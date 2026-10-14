from __future__ import annotations

from pathlib import Path
import threading

import pytest

from engine.option_d import (
    DrawOperation,
    DrawSchedule,
    OptionDContractError,
    RNGStateBundle,
    checkpoint_bytes,
    capture_checkpoint,
    execute_cpu,
    load_checkpoint,
    restore_checkpoint,
    state_from_seed,
    write_checkpoint_atomic,
    write_snapshot_atomic,
)


SOURCE = "test-source-b7e1856"
ARTIFACT = "test-option-d-artifact-v1"


def _successor() -> tuple[RNGStateBundle, DrawSchedule]:
    bundle = RNGStateBundle(
        (state_from_seed(101), state_from_seed(202)),
        source_binding=SOURCE,
    )
    schedule = DrawSchedule(
        (
            (DrawOperation.gauss(), DrawOperation.getrandbits(0)),
            (DrawOperation.random(), DrawOperation.randbelow(19)),
        )
    )
    return execute_cpu(bundle, schedule).successor, schedule


def test_checkpoint_round_trip_is_canonical_and_fresh_owner_safe() -> None:
    successor, schedule = _successor()
    first = checkpoint_bytes(
        successor,
        artifact_binding=ARTIFACT,
        schedule_sha256=schedule.digest(),
    )
    second = checkpoint_bytes(
        successor,
        artifact_binding=ARTIFACT,
        schedule_sha256=schedule.digest(),
    )

    restored = restore_checkpoint(
        first,
        expected_source_binding=SOURCE,
        expected_artifact_binding=ARTIFACT,
        expected_schedule_sha256=schedule.digest(),
    )
    assert first == second
    assert restored == successor
    assert restored is not successor
    assert all(a is not b for a, b in zip(restored.worlds, successor.worlds, strict=True))


def test_uninterrupted_equals_checkpoint_resume() -> None:
    initial = RNGStateBundle((state_from_seed(303),), source_binding=SOURCE)
    prefix_schedule = DrawSchedule(
        ((DrawOperation.gauss(), DrawOperation.random(), DrawOperation.randbelow(7)),)
    )
    suffix_schedule = DrawSchedule(
        ((DrawOperation.gauss(), DrawOperation.getrandbits(64), DrawOperation.random()),)
    )
    uninterrupted = execute_cpu(
        initial,
        DrawSchedule((prefix_schedule.worlds[0] + suffix_schedule.worlds[0],)),
    )
    prefix = execute_cpu(initial, prefix_schedule)
    data = checkpoint_bytes(prefix.successor, artifact_binding=ARTIFACT)
    restored = restore_checkpoint(
        data,
        expected_source_binding=SOURCE,
        expected_artifact_binding=ARTIFACT,
    )
    resumed = execute_cpu(restored, suffix_schedule)

    assert prefix.material.worlds[0] + resumed.material.worlds[0] == uninterrupted.material.worlds[0]
    assert resumed.successor == uninterrupted.successor


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source", "wrong-source", "source binding mismatch"),
        ("artifact", "wrong-artifact", "artifact binding mismatch"),
        ("schedule", "0" * 64, "schedule binding mismatch"),
    ],
)
def test_wrong_bindings_fail_closed(field: str, value: str, message: str) -> None:
    successor, schedule = _successor()
    data = checkpoint_bytes(
        successor,
        artifact_binding=ARTIFACT,
        schedule_sha256=schedule.digest(),
    )
    arguments = {
        "expected_source_binding": SOURCE,
        "expected_artifact_binding": ARTIFACT,
        "expected_schedule_sha256": schedule.digest(),
    }
    arguments[f"expected_{field}_binding" if field != "schedule" else "expected_schedule_sha256"] = value

    with pytest.raises(OptionDContractError, match=message):
        restore_checkpoint(data, **arguments)


def test_corruption_is_rejected() -> None:
    successor, _ = _successor()
    data = bytearray(checkpoint_bytes(successor, artifact_binding=ARTIFACT))
    offset = data.index(b"words") + 10
    data[offset] = ord("9") if data[offset] != ord("9") else ord("8")

    with pytest.raises(OptionDContractError):
        restore_checkpoint(
            bytes(data),
            expected_source_binding=SOURCE,
            expected_artifact_binding=ARTIFACT,
        )


def test_atomic_file_publication_and_load(tmp_path: Path) -> None:
    successor, schedule = _successor()
    destination = tmp_path / "owner" / "checkpoint.json"
    digest = write_checkpoint_atomic(
        destination,
        successor,
        artifact_binding=ARTIFACT,
        schedule_sha256=schedule.digest(),
    )
    restored = load_checkpoint(
        destination,
        expected_source_binding=SOURCE,
        expected_artifact_binding=ARTIFACT,
        expected_schedule_sha256=schedule.digest(),
    )

    assert len(digest) == 64
    assert restored == successor
    assert list(destination.parent.glob("*.tmp")) == []


def test_owner_snapshot_remains_exact_during_delayed_persistence(tmp_path: Path) -> None:
    initial = RNGStateBundle((state_from_seed(404),), source_binding=SOURCE)
    prefix_schedule = DrawSchedule(((DrawOperation.gauss(), DrawOperation.random()),))
    prefix = execute_cpu(initial, prefix_schedule)
    snapshot = capture_checkpoint(prefix.successor, artifact_binding=ARTIFACT)

    # The authoritative owner advances after capture while persistence is held.
    later = execute_cpu(
        prefix.successor,
        DrawSchedule(((DrawOperation.gauss(), DrawOperation.getrandbits(64)),)),
    )
    release = threading.Event()
    destination = tmp_path / "delayed" / "checkpoint.json"
    errors: list[BaseException] = []

    def persist() -> None:
        try:
            assert release.wait(timeout=5)
            write_snapshot_atomic(destination, snapshot)
        except BaseException as error:  # pragma: no cover - surfaced below
            errors.append(error)

    writer = threading.Thread(target=persist, name="option-d-checkpoint-writer")
    writer.start()
    assert later.successor != prefix.successor
    release.set()
    writer.join(timeout=5)
    assert not writer.is_alive()
    assert errors == []

    restored = load_checkpoint(
        destination,
        expected_source_binding=SOURCE,
        expected_artifact_binding=ARTIFACT,
    )
    assert restored == prefix.successor
    assert restored != later.successor
