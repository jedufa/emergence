from __future__ import annotations

import threading
from pathlib import Path

import pytest

from engine.campaigns.acs_checkpoint_staging import (
    ACSCheckpointStaging,
    ACSCheckpointStagingError,
)


class _ObservedResultWaitCondition(threading.Condition):
    def __init__(self, result_waiting: threading.Event) -> None:
        super().__init__()
        self._result_waiting = result_waiting

    def wait(self, timeout: float | None = None) -> bool:
        if threading.current_thread().name == "acs-result-consumer":
            self._result_waiting.set()
        return super().wait(timeout)


def test_native_boundary_exports_final_stream_synchronization_timing() -> None:
    source = (
        Path(__file__).parents[1]
        / "engine/perf/cuda_architecture_b_flattened_controller_v1_binding.cpp"
    ).read_text()

    assert "final_synchronization_started" in source
    assert 'receipt["final_stream_synchronization_time_ns"]' in source
    assert (
        'receipt["last_useful_kernel_completion_observed_host_time_ns"]'
        in source
    )


def test_two_slot_writer_is_fifo_and_advances_durable_frontier_on_consumption() -> None:
    release_first = threading.Event()
    calls: list[str] = []
    staging = ACSCheckpointStaging[str](slots=2)

    first = staging.submit(
        compute_frontier=1000,
        captured_bytes=10,
        persist=lambda: (release_first.wait(2), calls.append("first"), "a")[-1],
    )
    second = staging.submit(
        compute_frontier=2500,
        captured_bytes=20,
        persist=lambda: (calls.append("second"), "b")[-1],
    )

    before = staging.telemetry()
    assert before["compute_frontier"] == 2500
    assert before["durable_checkpoint_frontier"] == -1
    assert before["queue_occupancy_high_water"] == 2
    release_first.set()
    assert staging.result(first) == "a"
    assert staging.telemetry()["durable_checkpoint_frontier"] == -1
    staging.acknowledge_durable(first)
    assert staging.result(second) == "b"
    staging.acknowledge_durable(second)
    assert calls == ["first", "second"]
    after = staging.telemetry()
    assert after["durable_checkpoint_frontier"] == 2500
    assert after["frontier_lag"] == 0
    staging.close()


def test_one_slot_applies_backpressure_until_prior_result_is_released() -> None:
    writer_started = threading.Event()
    release_writer = threading.Event()
    submit_finished = threading.Event()
    staging = ACSCheckpointStaging[str](slots=1)

    def persist_first() -> str:
        writer_started.set()
        assert release_writer.wait(2)
        return "first"

    first = staging.submit(
        compute_frontier=1000,
        captured_bytes=10,
        persist=persist_first,
    )
    assert writer_started.wait(2)

    second_box: list[object] = []

    def submit_second() -> None:
        second_box.append(
            staging.submit(
                compute_frontier=2500,
                captured_bytes=20,
                persist=lambda: "second",
            )
        )
        submit_finished.set()

    submit_thread = threading.Thread(target=submit_second)
    submit_thread.start()
    assert not submit_finished.wait(0.05)
    release_writer.set()
    assert staging.result(first) == "first"
    staging.acknowledge_durable(first)
    assert submit_finished.wait(2)
    second = second_box[0]
    assert staging.result(second) == "second"  # type: ignore[arg-type]
    staging.acknowledge_durable(second)  # type: ignore[arg-type]
    submit_thread.join()
    assert staging.telemetry()["producer_blocked_time_ns"] > 0
    staging.close()


def test_capture_reservation_is_bounded_before_persistence_is_sealed() -> None:
    staging = ACSCheckpointStaging[str](slots=1)
    first = staging.begin_capture(compute_frontier=1000)
    assert staging.telemetry()["slot_states"] == ["CAPTURING"]
    second_started = threading.Event()
    second_box: list[object] = []

    def reserve_second() -> None:
        second_box.append(staging.begin_capture(compute_frontier=2500))
        second_started.set()

    reserve_thread = threading.Thread(target=reserve_second)
    reserve_thread.start()
    assert not second_started.wait(0.05)
    staging.seal_capture(first, captured_bytes=10, persist=lambda: "first")
    assert staging.result(first, producer_backpressure=True) == "first"
    staging.acknowledge_durable(first)
    assert second_started.wait(2)
    second = second_box[0]
    staging.seal_capture(  # type: ignore[arg-type]
        second,
        captured_bytes=20,
        persist=lambda: "second",
    )
    assert staging.result(second) == "second"  # type: ignore[arg-type]
    staging.acknowledge_durable(second)  # type: ignore[arg-type]
    reserve_thread.join()
    assert staging.telemetry()["queue_occupancy_high_water"] == 1
    staging.close()


def test_later_sealed_capture_cannot_overtake_unsealed_fifo_predecessor() -> None:
    staging = ACSCheckpointStaging[str](slots=2)
    first = staging.begin_capture(compute_frontier=1000)
    second = staging.begin_capture(compute_frontier=2500)
    second_started = threading.Event()
    staging.seal_capture(
        second,
        captured_bytes=20,
        persist=lambda: (second_started.set(), "second")[-1],
    )
    assert not second_started.wait(0.05)
    staging.seal_capture(first, captured_bytes=10, persist=lambda: "first")
    assert staging.result(first) == "first"
    staging.acknowledge_durable(first)
    assert staging.result(second) == "second"
    staging.acknowledge_durable(second)
    staging.close()


def test_writer_failure_latches_and_never_advances_durable_frontier() -> None:
    staging = ACSCheckpointStaging[str](slots=2)

    def fail() -> str:
        raise OSError("injected disk full")

    token = staging.submit(
        compute_frontier=1000,
        captured_bytes=10,
        persist=fail,
    )
    with pytest.raises(ACSCheckpointStagingError, match="injected disk full") as caught:
        staging.result(token)

    assert isinstance(caught.value.__cause__, OSError)
    telemetry = staging.telemetry()
    assert telemetry["failure_latched"] is True
    assert telemetry["durable_checkpoint_frontier"] == -1
    with pytest.raises(ACSCheckpointStagingError, match="injected disk full"):
        staging.submit(
            compute_frontier=2500,
            captured_bytes=20,
            persist=lambda: "unreachable",
        )
    staging.abort()


def test_results_cannot_publish_durable_frontiers_out_of_fifo_order() -> None:
    staging = ACSCheckpointStaging[str](slots=2)
    first = staging.submit(
        compute_frontier=1000,
        captured_bytes=10,
        persist=lambda: "first",
    )
    second = staging.submit(
        compute_frontier=2500,
        captured_bytes=20,
        persist=lambda: "second",
    )

    with pytest.raises(ACSCheckpointStagingError, match="FIFO"):
        staging.result(second)
    assert staging.telemetry()["durable_checkpoint_frontier"] == -1
    assert staging.result(first) == "first"
    staging.acknowledge_durable(first)
    assert staging.result(second) == "second"
    staging.acknowledge_durable(second)
    staging.close()


def test_close_rejects_unconsumed_persisting_snapshot() -> None:
    release = threading.Event()
    staging = ACSCheckpointStaging[str](slots=1)
    token = staging.submit(
        compute_frontier=1000,
        captured_bytes=10,
        persist=lambda: (release.wait(2), "done")[-1],
    )
    with pytest.raises(ACSCheckpointStagingError, match="unconsumed"):
        staging.close()
    release.set()
    assert staging.result(token) == "done"
    with pytest.raises(ACSCheckpointStagingError, match="unpublished"):
        staging.close()
    staging.acknowledge_durable(token)
    staging.close()


def test_abort_revokes_unsealed_capture_and_is_idempotent() -> None:
    staging = ACSCheckpointStaging[str](slots=1)
    token = staging.begin_capture(compute_frontier=1000)

    staging.abort()
    staging.abort()

    telemetry = staging.telemetry()
    assert telemetry["closed"] is True
    assert telemetry["aborted"] is True
    assert telemetry["abandoned_snapshot_count"] == 1
    assert telemetry["queue_occupancy"] == 0
    assert telemetry["slot_states"] == ["FREE"]
    assert telemetry["writer_alive"] is False
    for operation in (
        lambda: staging.ready(token),
        lambda: staging.result(token),
        lambda: staging.acknowledge_durable(token),
    ):
        with pytest.raises(ACSCheckpointStagingError, match="aborted"):
            operation()
    staging.close()


def test_abort_waits_for_running_work_and_never_starts_abandoned_callback() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    abort_returned = threading.Event()
    staging = ACSCheckpointStaging[str](slots=2)

    def persist_first() -> str:
        first_started.set()
        assert release_first.wait(2)
        return "first"

    first = staging.submit(
        compute_frontier=1000,
        captured_bytes=10,
        persist=persist_first,
    )
    assert first_started.wait(2)
    second = staging.submit(
        compute_frontier=2500,
        captured_bytes=20,
        persist=lambda: (second_started.set(), "second")[-1],
    )

    abort_thread = threading.Thread(
        target=lambda: (staging.abort(), abort_returned.set())
    )
    abort_thread.start()
    assert not abort_returned.wait(0.05)
    assert not second_started.is_set()
    release_first.set()
    assert abort_returned.wait(2)
    abort_thread.join()

    assert not second_started.wait(0.05)
    telemetry = staging.telemetry()
    assert telemetry["abandoned_snapshot_count"] == 2
    assert telemetry["queue_occupancy"] == 0
    assert telemetry["slot_states"] == ["FREE", "FREE"]
    assert telemetry["writer_alive"] is False
    for token in (first, second):
        with pytest.raises(ACSCheckpointStagingError, match="aborted"):
            staging.ready(token)


def test_abort_rejects_result_waiting_before_writer_dispatch() -> None:
    writer_gate = threading.Event()
    result_waiting = threading.Event()
    callback_started = threading.Event()
    abort_returned = threading.Event()
    result_outcome: list[object] = []

    def delayed_thread_factory(**kwargs: object) -> threading.Thread:
        target = kwargs["target"]
        assert callable(target)
        return threading.Thread(
            target=lambda: (writer_gate.wait(), target()),
            name=str(kwargs["name"]),
            daemon=bool(kwargs["daemon"]),
        )

    staging = ACSCheckpointStaging[str](
        slots=1,
        thread_factory=delayed_thread_factory,
    )
    staging._condition = _ObservedResultWaitCondition(result_waiting)
    token = staging.submit(
        compute_frontier=1000,
        captured_bytes=10,
        persist=lambda: (callback_started.set(), "unreachable")[-1],
    )

    def consume() -> None:
        try:
            result_outcome.append(staging.result(token))
        except BaseException as exc:
            result_outcome.append(exc)

    consumer = threading.Thread(
        target=consume,
        name="acs-result-consumer",
        daemon=True,
    )
    consumer.start()
    assert result_waiting.wait(2)
    aborter = threading.Thread(
        target=lambda: (staging.abort(), abort_returned.set())
    )
    aborter.start()
    with staging._condition:
        if not staging._aborted:
            staging._condition.wait(2)
        assert staging._aborted
    assert not abort_returned.is_set()

    writer_gate.set()
    assert abort_returned.wait(2)
    aborter.join()
    consumer.join(2)

    assert not consumer.is_alive()
    assert not callback_started.is_set()
    assert len(result_outcome) == 1
    assert isinstance(result_outcome[0], ACSCheckpointStagingError)
    assert "aborted" in str(result_outcome[0])


def test_abort_rejects_waiting_result_after_inflight_callback_finishes() -> None:
    writer_gate = threading.Event()
    writer_started = threading.Event()
    release_writer = threading.Event()
    result_waiting = threading.Event()
    abort_returned = threading.Event()
    result_outcome: list[object] = []

    def delayed_thread_factory(**kwargs: object) -> threading.Thread:
        target = kwargs["target"]
        assert callable(target)
        return threading.Thread(
            target=lambda: (writer_gate.wait(), target()),
            name=str(kwargs["name"]),
            daemon=bool(kwargs["daemon"]),
        )

    staging = ACSCheckpointStaging[str](
        slots=1,
        thread_factory=delayed_thread_factory,
    )
    staging._condition = _ObservedResultWaitCondition(result_waiting)

    def persist() -> str:
        writer_started.set()
        assert release_writer.wait(2)
        return "must-not-leak"

    token = staging.submit(
        compute_frontier=1000,
        captured_bytes=10,
        persist=persist,
    )
    writer_gate.set()
    assert writer_started.wait(2)

    def consume() -> None:
        try:
            result_outcome.append(staging.result(token))
        except BaseException as exc:
            result_outcome.append(exc)

    consumer = threading.Thread(
        target=consume,
        name="acs-result-consumer",
        daemon=True,
    )
    consumer.start()
    assert result_waiting.wait(2)
    aborter = threading.Thread(
        target=lambda: (staging.abort(), abort_returned.set())
    )
    aborter.start()
    with staging._condition:
        if not staging._aborted:
            staging._condition.wait(2)
        assert staging._aborted
    assert not abort_returned.is_set()

    release_writer.set()
    assert abort_returned.wait(2)
    aborter.join()
    consumer.join(2)

    assert not consumer.is_alive()
    assert len(result_outcome) == 1
    assert isinstance(result_outcome[0], ACSCheckpointStagingError)
    assert "aborted" in str(result_outcome[0])


def test_abort_revokes_consumed_but_unpublished_result() -> None:
    staging = ACSCheckpointStaging[str](slots=1)
    token = staging.submit(
        compute_frontier=1000,
        captured_bytes=10,
        persist=lambda: "complete-but-unpublished",
    )
    assert staging.result(token) == "complete-but-unpublished"

    staging.abort()

    telemetry = staging.telemetry()
    assert telemetry["abandoned_snapshot_count"] == 1
    assert telemetry["durable_checkpoint_frontier"] == -1
    assert telemetry["writer_alive"] is False
    with pytest.raises(ACSCheckpointStagingError, match="aborted"):
        staging.acknowledge_durable(token)


def test_close_joins_writer_after_failure_and_is_idempotent() -> None:
    staging = ACSCheckpointStaging[str](slots=1)

    def fail() -> str:
        raise OSError("injected close-after-failure")

    token = staging.submit(
        compute_frontier=1000,
        captured_bytes=10,
        persist=fail,
    )
    with pytest.raises(ACSCheckpointStagingError, match="close-after-failure"):
        staging.result(token)

    staging.close()
    staging.close()

    telemetry = staging.telemetry()
    assert telemetry["closed"] is True
    assert telemetry["failure_latched"] is True
    assert telemetry["writer_alive"] is False
