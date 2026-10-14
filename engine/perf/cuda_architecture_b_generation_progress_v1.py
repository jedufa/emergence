"""Publish host-visible progress from an existing native RunControlV2."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Final


SCHEMA_VERSION: Final[str] = "eiel_cuda_architecture_b_generation_progress_v1"
DEFAULT_POLL_SECONDS: Final[float] = 0.001
_PHASES: Final[frozenset[str]] = frozenset(
    {"initial", "running", "final", "failure"}
)


class NativeGenerationProgressError(RuntimeError):
    """The observer could not publish an exact native progress document."""


def _fail(message: str) -> NativeGenerationProgressError:
    return NativeGenerationProgressError(message)


def _stable_id(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or not value.isascii()
        or not value.isprintable()
    ):
        raise _fail(f"{name} must be a nonempty stable ASCII identifier")
    return value


def _atomic_write_json_v1(path: Path, value: Mapping[str, Any]) -> None:
    """Replace one JSON document atomically without durability synchronization."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class NativeGenerationProgressPublisherV1:
    """Observe a RunControlV2 manifest without taking simulation ownership."""

    def __init__(
        self,
        run_control: Any,
        *,
        output_path: Path,
        route_id: str,
        run_id: str,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
    ) -> None:
        if not callable(getattr(run_control, "manifest", None)):
            raise _fail("run_control must expose callable manifest")
        if not isinstance(output_path, Path):
            raise _fail("output_path must be a pathlib.Path")
        if (
            type(poll_seconds) is not float
            or not 0.0005 <= poll_seconds <= 1.0
        ):
            raise _fail("poll_seconds must be an exact float between 0.0005 and 1.0")
        self._run_control = run_control
        self._output_path = output_path
        self._route_id = _stable_id(route_id, name="route_id")
        self._run_id = _stable_id(run_id, name="run_id")
        self._poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._closed = False
        self._last_generation: int | None = None
        self._background_error: BaseException | None = None

    @property
    def output_path(self) -> Path:
        return self._output_path

    def _manifest(self) -> dict[str, Any]:
        raw = self._run_control.manifest()
        if not isinstance(raw, Mapping):
            raise _fail("run-control manifest must be a mapping")
        manifest = dict(raw)
        if manifest.get("run_id") != self._run_id:
            raise _fail("run-control manifest run_id does not match publisher run_id")
        for name in (
            "active_generation",
            "start_generation",
            "requested_generations",
            "target_next_generation",
        ):
            if type(manifest.get(name)) is not int:
                raise _fail(f"run-control manifest {name} must be an exact integer")
        if type(manifest.get("status")) is not str or not manifest["status"]:
            raise _fail("run-control manifest status must be a nonempty string")
        for name in ("run_started", "sealed"):
            if type(manifest.get(name)) is not bool:
                raise _fail(f"run-control manifest {name} must be an exact boolean")
        return manifest

    def _payload(
        self,
        manifest: Mapping[str, Any],
        *,
        phase: str,
        failure: BaseException | None = None,
    ) -> dict[str, Any]:
        if phase not in _PHASES:
            raise AssertionError(f"unsupported progress phase: {phase}")
        active_generation = manifest["active_generation"]
        if phase == "final" and manifest["status"] == "generation_complete":
            completed_boundary = manifest["target_next_generation"]
        elif active_generation >= 0:
            completed_boundary = active_generation
        else:
            completed_boundary = manifest["start_generation"]
        return {
            "schema_version": SCHEMA_VERSION,
            "route_id": self._route_id,
            "run_id": self._run_id,
            "phase": phase,
            "active_generation": active_generation,
            "completed_generation_boundary": completed_boundary,
            "start_generation": manifest["start_generation"],
            "requested_generations": manifest["requested_generations"],
            "target_next_generation": manifest["target_next_generation"],
            "run_control_status": manifest["status"],
            "run_started": manifest["run_started"],
            "sealed": manifest["sealed"],
            "failure": (
                None
                if failure is None
                else {"type": type(failure).__name__, "message": str(failure)}
            ),
        }

    def _publish(
        self,
        manifest: Mapping[str, Any],
        *,
        phase: str,
        failure: BaseException | None = None,
    ) -> None:
        _atomic_write_json_v1(
            self._output_path,
            self._payload(manifest, phase=phase, failure=failure),
        )

    def _publish_running_if_new(self, manifest: Mapping[str, Any]) -> None:
        generation = manifest["active_generation"]
        if generation < 0 or generation == self._last_generation:
            return
        self._publish(manifest, phase="running")
        self._last_generation = generation

    def _observe(self) -> None:
        try:
            while not self._stop.wait(self._poll_seconds):
                self._publish_running_if_new(self._manifest())
        except BaseException as exc:
            self._background_error = exc
            self._stop.set()

    def start(self) -> NativeGenerationProgressPublisherV1:
        if self._started:
            raise _fail("native generation progress publisher is single-use")
        manifest = self._manifest()
        self._publish(manifest, phase="initial")
        self._started = True
        self._thread = threading.Thread(
            target=self._observe,
            name=f"{self._route_id}-generation-progress",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self, *, failure: BaseException | None = None) -> None:
        if not self._started:
            raise _fail("native generation progress publisher was not started")
        if self._closed:
            raise _fail("native generation progress publisher is already closed")
        self._stop.set()
        assert self._thread is not None
        self._thread.join(timeout=max(1.0, self._poll_seconds * 4.0))
        terminal_error: BaseException | None = None
        if self._thread.is_alive():
            terminal_error = _fail("native generation progress thread did not stop")
        manifest: dict[str, Any] | None = None
        try:
            manifest = self._manifest()
        except BaseException as exc:
            terminal_error = terminal_error or exc
        if manifest is not None:
            try:
                self._publish_running_if_new(manifest)
            except BaseException as exc:
                terminal_error = terminal_error or exc
            try:
                self._publish(
                    manifest,
                    phase="failure" if failure is not None else "final",
                    failure=failure,
                )
            except BaseException as exc:
                terminal_error = terminal_error or exc
        self._closed = True
        publication_error = self._background_error or terminal_error
        if publication_error is not None:
            raise _fail(
                "native generation progress publication failed: "
                f"{type(publication_error).__name__}: {publication_error}"
            ) from publication_error

    def __enter__(self) -> NativeGenerationProgressPublisherV1:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        del exc_type, traceback
        try:
            self.close(failure=exc)
        except BaseException as publication_error:
            if exc is None:
                raise
            exc.add_note(
                "native generation progress publication also failed: "
                f"{type(publication_error).__name__}: {publication_error}"
            )
        return False


__all__ = [
    "DEFAULT_POLL_SECONDS",
    "SCHEMA_VERSION",
    "NativeGenerationProgressError",
    "NativeGenerationProgressPublisherV1",
]
