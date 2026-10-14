from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

import pytest

from engine.perf import cuda_architecture_b_generation_progress_v1 as subject


class _Control:
    def __init__(self, run_id: str = "native-run") -> None:
        self._lock = threading.Lock()
        self._manifest: dict[str, Any] = {
            "run_id": run_id,
            "status": "initialized",
            "run_started": False,
            "sealed": False,
            "active_generation": -1,
            "start_generation": -1,
            "requested_generations": 0,
            "target_next_generation": -1,
        }

    def manifest(self) -> Mapping[str, Any]:
        with self._lock:
            return dict(self._manifest)

    def update(self, **values: Any) -> None:
        with self._lock:
            self._manifest.update(values)


def _wait_until(predicate: Any, *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError("timed out waiting for progress observer")


def test_publisher_emits_exact_schema_and_deduplicates_generations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    control = _Control()
    publications: list[dict[str, Any]] = []

    def capture(_path: Path, value: Mapping[str, Any]) -> None:
        publications.append(dict(value))

    monkeypatch.setattr(subject, "_atomic_write_json_v1", capture)
    publisher = subject.NativeGenerationProgressPublisherV1(
        control,
        output_path=tmp_path / "progress.json",
        route_id="resident-checkpoint-segment",
        run_id="native-run",
        poll_seconds=0.0005,
    )
    publisher.start()
    control.update(
        status="generation_start",
        run_started=True,
        active_generation=25,
        start_generation=25,
        requested_generations=75,
        target_next_generation=100,
    )
    _wait_until(lambda: len(publications) >= 2)
    time.sleep(0.003)
    control.update(status="tick_running")
    time.sleep(0.003)
    assert [row["active_generation"] for row in publications[1:]] == [25]

    control.update(active_generation=26)
    _wait_until(lambda: len(publications) >= 3)
    control.update(status="generation_complete", active_generation=99)
    publisher.close()

    assert [row["phase"] for row in publications] == [
        "initial",
        "running",
        "running",
        "running",
        "final",
    ]
    assert [row["active_generation"] for row in publications[1:-1]] == [25, 26, 99]
    assert set(publications[-1]) == {
        "schema_version",
        "route_id",
        "run_id",
        "phase",
        "active_generation",
        "completed_generation_boundary",
        "start_generation",
        "requested_generations",
        "target_next_generation",
        "run_control_status",
        "run_started",
        "sealed",
        "failure",
    }
    assert publications[-1]["completed_generation_boundary"] == 100
    assert publications[-1] == {
        "schema_version": subject.SCHEMA_VERSION,
        "route_id": "resident-checkpoint-segment",
        "run_id": "native-run",
        "phase": "final",
        "active_generation": 99,
        "completed_generation_boundary": 100,
        "start_generation": 25,
        "requested_generations": 75,
        "target_next_generation": 100,
        "run_control_status": "generation_complete",
        "run_started": True,
        "sealed": False,
        "failure": None,
    }
    assert publisher._thread is not None
    assert not publisher._thread.is_alive()


def test_context_publishes_failure_without_swallowing_original_error(
    tmp_path: Path,
) -> None:
    control = _Control()
    output = tmp_path / "progress.json"

    with pytest.raises(ValueError, match="native operation failed"):
        with subject.NativeGenerationProgressPublisherV1(
            control,
            output_path=output,
            route_id="ub100-repeat",
            run_id="native-run",
        ):
            control.update(
                status="failed",
                run_started=True,
                sealed=True,
                active_generation=7,
                start_generation=0,
                requested_generations=100,
                target_next_generation=100,
            )
            raise ValueError("native operation failed")

    assert json.loads(output.read_text(encoding="ascii")) == {
        "schema_version": subject.SCHEMA_VERSION,
        "route_id": "ub100-repeat",
        "run_id": "native-run",
        "phase": "failure",
        "active_generation": 7,
        "completed_generation_boundary": 7,
        "start_generation": 0,
        "requested_generations": 100,
        "target_next_generation": 100,
        "run_control_status": "failed",
        "run_started": True,
        "sealed": True,
        "failure": {"type": "ValueError", "message": "native operation failed"},
    }


def test_context_preserves_native_error_when_failure_publication_also_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    control = _Control()

    def fail_failure_publication(_path: Path, value: Mapping[str, Any]) -> None:
        if value["phase"] == "failure":
            raise OSError("injected terminal progress failure")

    monkeypatch.setattr(subject, "_atomic_write_json_v1", fail_failure_publication)
    with pytest.raises(ValueError, match="native operation failed") as captured:
        with subject.NativeGenerationProgressPublisherV1(
            control,
            output_path=tmp_path / "progress.json",
            route_id="dual-failure-route",
            run_id="native-run",
        ):
            control.update(
                status="failed",
                run_started=True,
                sealed=True,
                active_generation=7,
                start_generation=0,
                requested_generations=100,
                target_next_generation=100,
            )
            raise ValueError("native operation failed")

    assert captured.value.__notes__ == [
        "native generation progress publication also failed: "
        "NativeGenerationProgressError: native generation progress publication failed: "
        "OSError: injected terminal progress failure"
    ]


def test_atomic_writer_uses_same_directory_replace_without_fsync(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "nested" / "progress.json"
    replacements: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        source_path = Path(source)
        target_path = Path(target)
        replacements.append((source_path, target_path))
        assert source_path.parent == target_path.parent
        real_replace(source, target)

    monkeypatch.setattr(subject.os, "replace", replace)
    monkeypatch.setattr(
        subject.os,
        "fsync",
        lambda _descriptor: pytest.fail("progress publication must not fsync"),
    )

    subject._atomic_write_json_v1(output, {"phase": "initial"})
    subject._atomic_write_json_v1(output, {"phase": "final"})

    assert json.loads(output.read_text(encoding="ascii")) == {"phase": "final"}
    assert len(replacements) == 2
    assert not list(output.parent.glob(".progress.json.*.tmp"))


def test_background_write_error_is_surfaced_on_clean_shutdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    control = _Control()
    calls = 0

    def fail_running(_path: Path, value: Mapping[str, Any]) -> None:
        nonlocal calls
        calls += 1
        if value["phase"] == "running":
            raise OSError("injected progress write failure")

    monkeypatch.setattr(subject, "_atomic_write_json_v1", fail_running)
    publisher = subject.NativeGenerationProgressPublisherV1(
        control,
        output_path=tmp_path / "progress.json",
        route_id="write-error-route",
        run_id="native-run",
        poll_seconds=0.0005,
    ).start()
    control.update(
        status="generation_start",
        run_started=True,
        active_generation=0,
        start_generation=0,
        requested_generations=1,
        target_next_generation=1,
    )
    _wait_until(lambda: publisher._background_error is not None)

    with pytest.raises(
        subject.NativeGenerationProgressError,
        match="injected progress write failure",
    ):
        publisher.close()

    assert calls >= 2
    assert publisher._thread is not None
    assert not publisher._thread.is_alive()


def test_route_and_manifest_run_ids_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(subject.NativeGenerationProgressError, match="route_id"):
        subject.NativeGenerationProgressPublisherV1(
            _Control(),
            output_path=tmp_path / "progress.json",
            route_id=" route ",
            run_id="native-run",
        )

    publisher = subject.NativeGenerationProgressPublisherV1(
        _Control(run_id="other-run"),
        output_path=tmp_path / "progress.json",
        route_id="exact-route",
        run_id="native-run",
    )
    with pytest.raises(subject.NativeGenerationProgressError, match="does not match"):
        publisher.start()
