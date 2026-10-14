from __future__ import annotations

import threading

from native_runtime.live_chronicle_stream import (
    NativeRunControlLivePublisher,
    event_from_manifest,
    event_from_snapshot,
)


def _manifest(ordinal: int = 1) -> dict:
    return {
        "run_id": "gpu-world-a",
        "controller_id": "controller:gpu-world-a",
        "owner_epoch": 1,
        "last_boundary": {
            "completed_generation": ordinal - 1,
            "next_generation": ordinal,
            "boundary_ordinal": ordinal,
            "trajectory_commit_frontier": ordinal - 1,
        },
    }


def test_manifest_projection_is_one_provisional_committed_boundary() -> None:
    event = event_from_manifest(_manifest(), last_boundary_ordinal=0)

    assert event is not None
    assert event["event_type"] == "native_generation_boundary"
    assert event["generation"] == 0
    assert event["boundary_ordinal"] == 1
    assert event["source_completeness"] == "provisional_native_boundary"
    assert event_from_manifest(_manifest(), last_boundary_ordinal=1) is None


def test_live_publisher_is_deduplicated_and_failure_isolated() -> None:
    calls: list[dict] = []

    def sender(event):
        calls.append(dict(event))
        raise OSError("web UI unavailable")

    publisher = NativeRunControlLivePublisher(object(), sender=sender)
    publisher.observe(_manifest(1))
    publisher.observe(_manifest(1))
    publisher.observe(_manifest(2))

    assert [event["boundary_ordinal"] for event in calls] == [1, 2]


def test_mailbox_snapshot_projects_existing_compact_progress() -> None:
    snapshot = {
        "run_id": "gpu-world-a",
        "controller_id": "controller:gpu-world-a",
        "owner_epoch": 1,
        "boundary_ordinal": 3,
        "completed_generation": 2,
        "compact_progress": {"alive_count": 57},
    }

    event = event_from_snapshot(snapshot)

    assert event is not None
    assert event["generation"] == 2
    assert event["measurements"] == {"alive_count": 57}


def test_daemon_observes_boundary_while_owner_remains_live(monkeypatch) -> None:
    monkeypatch.setenv("EIEL_LIVE_CHRONICLE", "1")
    observed = threading.Event()
    state = {"manifest": {**_manifest(), "last_boundary": None}}

    class Control:
        def manifest(self):
            return dict(state["manifest"])

    calls: list[dict] = []

    def sender(event):
        calls.append(dict(event))
        observed.set()

    publisher = NativeRunControlLivePublisher(
        Control(), poll_seconds=0.005, sender=sender
    )
    publisher.start()
    state["manifest"] = _manifest(1)
    assert observed.wait(timeout=1.0)
    publisher.stop(final_manifest=_manifest(1))

    assert [event["boundary_ordinal"] for event in calls] == [1]
