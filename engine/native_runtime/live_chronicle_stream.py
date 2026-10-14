"""Best-effort RAM-only live Chronicle projection for native run boundaries."""

from __future__ import annotations

import json
import os
import queue
import threading
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any


SCHEMA_VERSION = "eiel_native_live_chronicle_event_v1"
DEFAULT_ENDPOINT = (
    "http://127.0.0.1:8787/api/emergence/live_simulations/chronicle/live"
)
_send_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=256)
_sender_lock = threading.Lock()
_sender_thread: threading.Thread | None = None


def _enabled() -> bool:
    configured = str(os.environ.get("EIEL_LIVE_CHRONICLE", "auto")).strip().lower()
    if "PYTEST_CURRENT_TEST" in os.environ and configured == "auto":
        return False
    return configured not in {
        "0",
        "off",
        "false",
        "no",
    }


def event_from_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any] | None:
    """Project one already-exported native mailbox boundary."""

    progress = snapshot.get("compact_progress")
    if not isinstance(progress, Mapping):
        return None
    manifest = {
        "run_id": snapshot.get("run_id"),
        "controller_id": snapshot.get("controller_id"),
        "owner_epoch": snapshot.get("owner_epoch"),
        "last_boundary": {
            "completed_generation": snapshot.get("completed_generation"),
            "boundary_ordinal": snapshot.get("boundary_ordinal"),
            "trajectory_commit_frontier": snapshot.get("completed_generation"),
        },
    }
    event = event_from_manifest(manifest, last_boundary_ordinal=0)
    if event is not None and type(progress.get("alive_count")) is int:
        event["measurements"] = {"alive_count": int(progress["alive_count"])}
    return event


def event_from_manifest(
    manifest: Mapping[str, Any], *, last_boundary_ordinal: int
) -> dict[str, Any] | None:
    """Project one newly committed native boundary without inspecting trajectory state."""

    boundary = manifest.get("last_boundary")
    if not isinstance(boundary, Mapping):
        return None
    ordinal = boundary.get("boundary_ordinal")
    generation = boundary.get("completed_generation")
    if type(ordinal) is not int or type(generation) is not int:
        return None
    if ordinal <= last_boundary_ordinal:
        return None
    run_id = str(manifest.get("run_id") or "").strip()
    controller_id = str(manifest.get("controller_id") or "").strip()
    owner_epoch = manifest.get("owner_epoch")
    if not run_id or not controller_id or type(owner_epoch) is not int:
        return None
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "controller_id": controller_id,
        "owner_epoch": owner_epoch,
        "event_id": f"{controller_id}:boundary:{ordinal}",
        "event_type": "native_generation_boundary",
        "generation": generation,
        "step": 0,
        "boundary_ordinal": ordinal,
        "trajectory_commit_frontier": boundary.get(
            "trajectory_commit_frontier", generation
        ),
        "headline": "Native boundary committed on the GPU owner",
        "scientific_summary": (
            "The native owner committed this boundary. Detailed scientific "
            "events remain provisional until terminal Chronicle materialization."
        ),
        "source_completeness": "provisional_native_boundary",
        "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }


def post_event(
    event: Mapping[str, Any],
    *,
    endpoint: str | None = None,
    timeout_seconds: float = 0.2,
) -> None:
    """Send one bounded local event; callers intentionally ignore all failures."""

    target = str(
        endpoint
        or os.environ.get("EIEL_LIVE_CHRONICLE_ENDPOINT")
        or DEFAULT_ENDPOINT
    ).strip()
    payload = json.dumps(dict(event), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    if len(payload) > 16 * 1024:
        raise ValueError("live Chronicle event exceeds 16 KiB")
    request = urllib.request.Request(
        target,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
        if not 200 <= int(response.status) < 300:
            raise OSError(f"live Chronicle endpoint returned HTTP {response.status}")


def _send_worker() -> None:
    while True:
        event = _send_queue.get()
        try:
            post_event(event)
        except Exception:
            pass
        finally:
            _send_queue.task_done()


def enqueue_event(event: Mapping[str, Any]) -> None:
    """Queue without blocking native execution; a full queue drops presentation data."""

    global _sender_thread
    if not _enabled():
        return
    with _sender_lock:
        if _sender_thread is None or not _sender_thread.is_alive():
            _sender_thread = threading.Thread(
                target=_send_worker,
                name="eiel-live-chronicle-sender",
                daemon=True,
            )
            _sender_thread.start()
    try:
        _send_queue.put_nowait(dict(event))
    except queue.Full:
        pass


class NativeRunControlLivePublisher:
    """Observe committed RunControl boundaries from a failure-isolated daemon."""

    def __init__(
        self,
        run_control: Any,
        *,
        poll_seconds: float = 0.02,
        sender: Callable[[Mapping[str, Any]], None] = enqueue_event,
    ) -> None:
        self._run_control = run_control
        self._poll_seconds = max(0.005, float(poll_seconds))
        self._sender = sender
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_boundary_ordinal = 0

    def start(self) -> None:
        if not _enabled() or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="eiel-native-live-chronicle",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=0.1)

    def stop(self, *, final_manifest: Mapping[str, Any] | None = None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if final_manifest is not None:
            self.observe(final_manifest)

    def observe(self, manifest: Mapping[str, Any]) -> None:
        event = event_from_manifest(
            manifest,
            last_boundary_ordinal=self._last_boundary_ordinal,
        )
        if event is None:
            return
        self._last_boundary_ordinal = int(event["boundary_ordinal"])
        try:
            self._sender(event)
        except Exception:
            pass

    def _run(self) -> None:
        first_observation = True
        while not self._stop.is_set():
            try:
                manifest = self._run_control.manifest()
                if isinstance(manifest, Mapping):
                    self.observe(manifest)
            except Exception:
                pass
            if first_observation:
                self._ready.set()
                first_observation = False
            self._stop.wait(self._poll_seconds)


__all__ = (
    "DEFAULT_ENDPOINT",
    "NativeRunControlLivePublisher",
    "SCHEMA_VERSION",
    "enqueue_event",
    "event_from_manifest",
    "event_from_snapshot",
    "post_event",
)
