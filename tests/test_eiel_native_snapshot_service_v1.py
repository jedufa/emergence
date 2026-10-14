from __future__ import annotations

from threading import Event, Thread
from time import monotonic, sleep
from types import MappingProxyType

import pytest
import torch

from engine.native_runtime.cuda_snapshot_service_v1 import (
    NativeSnapshotBackpressureError,
    NativeSnapshotServiceClosedError,
    NativeSnapshotServiceError,
    NativeSnapshotServiceFailure,
    NativeSnapshotServiceV1,
    NativeSnapshotSlotStateV1,
    NativeSnapshotTimeoutError,
)


STATE_SHAPE = (2, 16)
METRICS_SHAPE = (2, 4, 3)


def _cpu_allocator(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device="cpu")


def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = monotonic() + timeout
    while not predicate():
        if monotonic() >= deadline:
            raise AssertionError("timed out waiting for snapshot service state")
        sleep(0.005)


class _CaptureFence:
    def __init__(self, submission_id: int, order: list[tuple[str, int]]) -> None:
        self._submission_id = submission_id
        self._order = order

    def wait(self) -> dict[str, int]:
        self._order.append(("wait", self._submission_id))
        return {"capture_sequence": self._submission_id, "d2h_bytes": 224}


def _service(
    *,
    capture,
    encoder,
    slot_count: int,
    state_shape: tuple[int, ...] = STATE_SHAPE,
    metrics_shape: tuple[int, ...] = METRICS_SHAPE,
) -> NativeSnapshotServiceV1:
    return NativeSnapshotServiceV1(
        state_shape=state_shape,
        active_science_metrics_shape=metrics_shape,
        slot_count=slot_count,
        capture=capture,
        encoder=encoder,
        slot_allocator=_cpu_allocator,
        allow_unpinned_test_slots=True,
    )


def test_bounded_fifo_service_waits_capture_and_reuses_dynamic_slots() -> None:
    order: list[tuple[str, int]] = []

    def capture(request):
        order.append(("capture", request.submission_id))
        request.state_destination.fill_(request.absolute_tick)
        request.active_science_metrics_destination.fill_(request.absolute_tick + 10)
        return _CaptureFence(request.submission_id, order)

    def encoder(request):
        order.append(("encode", request.submission_id))
        assert request.capture_metrics["capture_sequence"] == request.submission_id
        assert int(request.state[0, 0]) == request.absolute_tick
        assert int(request.active_science_metrics[0, 0, 0]) == (
            request.absolute_tick + 10
        )
        return {
            "absolute_tick": request.absolute_tick,
            "state_pointer": int(request.state.data_ptr()),
            "metrics_pointer": int(request.active_science_metrics.data_ptr()),
        }

    service = _service(capture=capture, encoder=encoder, slot_count=2)
    try:
        first = service.submit(absolute_tick=3, context={"plan": "bound"})
        second = service.submit(absolute_tick=7, context={"plan": "bound"})
        first_batch = service.drain(timeout=2)
        assert [row.submission_id for row in first_batch] == [
            first.submission_id,
            second.submission_id,
        ]
        assert [row.result["absolute_tick"] for row in first_batch] == [3, 7]
        assert [item for item in order if item[0] == "capture"] == [
            ("capture", 1),
            ("capture", 2),
        ]
        assert [item for item in order if item[0] == "encode"] == [
            ("encode", 1),
            ("encode", 2),
        ]
        for submission_id in (1, 2):
            assert order.index(("capture", submission_id)) < order.index(
                ("wait", submission_id)
            ) < order.index(("encode", submission_id))

        third = service.submit(absolute_tick=11)
        third_batch = service.drain(timeout=2)
        assert third_batch[0].submission_id == third.submission_id
        assert third.slot_index in {first.slot_index, second.slot_index}

        telemetry = service.telemetry()
        assert telemetry["slot_count"] == 2
        assert telemetry["slot_state_counts"] == {
            "FREE": 2,
            "CAPTURING": 0,
            "ENCODING": 0,
            "COMPLETED": 0,
            "FAILED": 0,
        }
        assert telemetry["slot_reuse_count"] == 1
        assert telemetry["queue_capacity"] == 2
        assert 1 <= telemetry["queue_high_water"] <= 2
        assert telemetry["in_use_slots_high_water"] == 2
        assert telemetry["state_bytes_per_slot"] == 32
        assert telemetry["active_science_metrics_bytes_per_slot"] == 192
        assert telemetry["bytes_per_slot"] == 224
        assert telemetry["resident_slot_bytes"] == 448
        assert telemetry["resident_pinned_bytes"] == 0
        assert telemetry["pinned_memory_verified"] is False
        assert telemetry["horizon_dependent_buffer_bytes"] == 0
        for name in (
            "producer_capture_elapsed_ns_total",
            "queue_wait_elapsed_ns_total",
            "worker_capture_wait_elapsed_ns_total",
            "worker_encode_elapsed_ns_total",
            "worker_job_elapsed_ns_total",
        ):
            assert telemetry[name] > 0
    finally:
        service.shutdown(drain=True, raise_on_failure=False)


def test_saturation_backpressures_until_a_completion_releases_the_slot() -> None:
    encode_started = Event()
    release_encode = Event()

    def capture(request):
        request.state_destination.fill_(request.absolute_tick)
        request.active_science_metrics_destination.zero_()
        return None

    def encoder(request):
        encode_started.set()
        if not release_encode.wait(2):
            raise AssertionError("test encoder release was not signalled")
        return {"absolute_tick": request.absolute_tick}

    service = _service(capture=capture, encoder=encoder, slot_count=1)
    blocked_error: list[BaseException] = []
    blocked_thread: Thread | None = None
    try:
        first = service.submit(absolute_tick=1)
        assert encode_started.wait(2)
        assert service.has_capacity is False
        assert service.has_pending is True
        assert service.ready is False
        assert service.telemetry()["slot_states"] == ["ENCODING"]
        with pytest.raises(NativeSnapshotBackpressureError, match="saturated"):
            service.submit(absolute_tick=2, block=False)

        def submit_blocked() -> None:
            try:
                service.submit(absolute_tick=2, block=True, timeout=2)
            except BaseException as exc:  # pragma: no cover - surfaced below
                blocked_error.append(exc)

        blocked_thread = Thread(target=submit_blocked)
        blocked_thread.start()
        _wait_until(lambda: service.telemetry()["backpressure_count"] == 2)
        assert blocked_thread.is_alive()

        release_encode.set()
        _wait_until(lambda: service.ready)
        with pytest.raises(
            NativeSnapshotBackpressureError, match="must be settled"
        ):
            service.submit(absolute_tick=2)
        first_completion = service.settle_oldest(timeout=2)
        assert first_completion.submission_id == first.submission_id
        blocked_thread.join(2)
        assert not blocked_thread.is_alive()
        assert len(blocked_error) == 1
        assert isinstance(blocked_error[0], NativeSnapshotBackpressureError)
        assert "must be settled" in str(blocked_error[0])
        second = service.submit(absolute_tick=2)
        second_completion = service.settle_oldest(timeout=2)
        assert second_completion.submission_id == second.submission_id
        telemetry = service.telemetry()
        assert telemetry["backpressure_count"] >= 3
        assert telemetry["producer_submit_blocked_count"] == 1
        assert telemetry["producer_blocked_count"] >= 1
        assert telemetry["producer_blocked_elapsed_ns_total"] > 0
        assert telemetry["worker_job_elapsed_ns_total"] > 0
        assert service.has_capacity is True
        assert service.has_pending is False
    finally:
        release_encode.set()
        if blocked_thread is not None:
            blocked_thread.join(2)
        service.shutdown(drain=True, raise_on_failure=False)


def test_normal_settle_wait_is_counted_as_producer_backpressure() -> None:
    encode_started = Event()
    release_encode = Event()

    def encoder(request):
        encode_started.set()
        if not release_encode.wait(2):
            raise AssertionError("test encoder release was not signalled")
        return {"absolute_tick": request.absolute_tick}

    service = _service(
        capture=lambda _request: None,
        encoder=encoder,
        slot_count=1,
    )
    releaser: Thread | None = None
    try:
        service.submit(absolute_tick=1)
        assert encode_started.wait(2)

        def release_later() -> None:
            sleep(0.05)
            release_encode.set()

        releaser = Thread(target=release_later)
        releaser.start()
        completion = service.settle_oldest(timeout=2)
        assert completion.absolute_tick == 1
        telemetry = service.telemetry()
        assert telemetry["producer_submit_blocked_count"] == 0
        assert telemetry["producer_settle_blocked_count"] == 1
        assert telemetry["producer_settle_blocked_elapsed_ns_total"] > 0
        assert telemetry["producer_blocked_count"] == 1
        assert telemetry["producer_blocked_elapsed_ns_total"] > 0
    finally:
        release_encode.set()
        if releaser is not None:
            releaser.join(2)
        service.shutdown(drain=True, raise_on_failure=False)


def test_shutdown_timeout_is_bounded_while_capture_holds_submit_lock() -> None:
    capture_started = Event()
    release_capture = Event()
    submit_errors: list[BaseException] = []

    def capture(_request):
        capture_started.set()
        if not release_capture.wait(2):
            raise AssertionError("test capture release was not signalled")
        return None

    service = _service(capture=capture, encoder=lambda _request: {}, slot_count=1)

    def submit() -> None:
        try:
            service.submit(absolute_tick=1)
        except BaseException as exc:  # pragma: no cover - surfaced below
            submit_errors.append(exc)

    submit_thread = Thread(target=submit)
    submit_thread.start()
    try:
        assert capture_started.wait(2)
        started = monotonic()
        with pytest.raises(
            NativeSnapshotTimeoutError, match="in-progress native snapshot capture"
        ):
            service.shutdown(timeout=0.05, raise_on_failure=False)
        assert monotonic() - started < 0.25
        telemetry = service.telemetry()
        assert telemetry["accepting"] is False
        assert telemetry["stopping"] is False

        release_capture.set()
        submit_thread.join(2)
        assert not submit_thread.is_alive()
        assert submit_errors == []
        completions = service.shutdown(
            drain=True,
            timeout=2,
            raise_on_failure=False,
        )
        assert [row.absolute_tick for row in completions] == [1]
    finally:
        release_capture.set()
        submit_thread.join(2)
        service.shutdown(drain=True, raise_on_failure=False)


def test_completion_consumers_are_serialized_during_drain() -> None:
    second_encode_started = Event()
    release_second = Event()
    drained: list[object] = []
    drain_errors: list[BaseException] = []

    def encoder(request):
        if request.submission_id == 2:
            second_encode_started.set()
            if not release_second.wait(2):
                raise AssertionError("test encoder release was not signalled")
        return {"absolute_tick": request.absolute_tick}

    service = _service(
        capture=lambda _request: None,
        encoder=encoder,
        slot_count=2,
    )
    service.submit(absolute_tick=1)
    service.submit(absolute_tick=2)
    assert second_encode_started.wait(2)

    def drain() -> None:
        try:
            drained.extend(service.drain(timeout=2))
        except BaseException as exc:  # pragma: no cover - surfaced below
            drain_errors.append(exc)

    drain_thread = Thread(target=drain)
    drain_thread.start()
    try:
        settlement_lock = getattr(service, "_settlement_lock")
        _wait_until(settlement_lock.locked)
        with pytest.raises(
            NativeSnapshotTimeoutError, match="completion consumer"
        ):
            service.settle_oldest(timeout=0.05)
        release_second.set()
        drain_thread.join(2)
        assert not drain_thread.is_alive()
        assert drain_errors == []
        assert [row.absolute_tick for row in drained] == [1, 2]
    finally:
        release_second.set()
        drain_thread.join(2)
        service.shutdown(drain=True, raise_on_failure=False)


def test_shutdown_finishes_encoding_and_cancels_only_queued_capture() -> None:
    encode_started = Event()
    release_encode = Event()

    def capture(request):
        request.state_destination.fill_(request.absolute_tick)
        request.active_science_metrics_destination.zero_()
        return None

    def encoder(request):
        encode_started.set()
        if not release_encode.wait(2):
            raise AssertionError("test encoder release was not signalled")
        return {"absolute_tick": request.absolute_tick}

    service = _service(capture=capture, encoder=encoder, slot_count=2)
    shutdown_result: dict[str, object] = {}
    shutdown_error: list[BaseException] = []
    blocked_submit_error: list[BaseException] = []
    blocked_submit_thread: Thread | None = None
    try:
        first = service.submit(absolute_tick=1)
        assert encode_started.wait(2)
        second = service.submit(absolute_tick=2)
        _wait_until(
            lambda: sorted(service.telemetry()["slot_states"])
            == ["CAPTURING", "ENCODING"]
        )
        assert service.cancel(second) is True
        assert service.cancel(first) is False

        def blocked_submit() -> None:
            try:
                service.submit(absolute_tick=3, block=True, timeout=3)
            except BaseException as exc:
                blocked_submit_error.append(exc)

        blocked_submit_thread = Thread(target=blocked_submit)
        blocked_submit_thread.start()
        _wait_until(lambda: service.telemetry()["backpressure_count"] == 1)

        def stop() -> None:
            try:
                shutdown_result["completions"] = service.shutdown(
                    cancel_pending=True,
                    drain=True,
                    timeout=3,
                    raise_on_failure=False,
                )
            except BaseException as exc:  # pragma: no cover - surfaced below
                shutdown_error.append(exc)

        shutdown_thread = Thread(target=stop)
        shutdown_thread.start()
        _wait_until(lambda: service.telemetry()["stopping"] is True)
        release_encode.set()
        shutdown_thread.join(3)
        assert not shutdown_thread.is_alive()
        assert shutdown_error == []
        blocked_submit_thread.join(2)
        assert len(blocked_submit_error) == 1
        assert isinstance(
            blocked_submit_error[0], NativeSnapshotServiceClosedError
        )
        completions = shutdown_result["completions"]
        assert [row.submission_id for row in completions] == [
            first.submission_id,
            second.submission_id,
        ]
        assert completions[0].state is NativeSnapshotSlotStateV1.COMPLETED
        assert completions[1].state is NativeSnapshotSlotStateV1.FAILED
        assert completions[1].cancelled is True
        telemetry = service.telemetry()
        assert telemetry["worker_alive"] is False
        assert telemetry["cancelled_count"] == 1
        assert telemetry["slot_state_counts"]["FREE"] == 2
        with pytest.raises(NativeSnapshotServiceClosedError):
            service.submit(absolute_tick=3)
    finally:
        release_encode.set()
        if blocked_submit_thread is not None:
            blocked_submit_thread.join(2)
        service.shutdown(drain=True, raise_on_failure=False)


def test_encoder_failure_propagates_in_fifo_and_fails_queued_work_closed() -> None:
    first_encoder_started = Event()
    fail_first = Event()

    def capture(request):
        request.state_destination.fill_(request.absolute_tick)
        request.active_science_metrics_destination.zero_()
        return None

    def encoder(request):
        if request.submission_id == 1:
            first_encoder_started.set()
            if not fail_first.wait(2):
                raise AssertionError("test failure was not released")
            raise ValueError("synthetic native encoder failure")
        raise AssertionError("queued work ran after service failure")

    service = _service(capture=capture, encoder=encoder, slot_count=2)
    try:
        first = service.submit(absolute_tick=1)
        assert first_encoder_started.wait(2)
        second = service.submit(absolute_tick=2)
        fail_first.set()
        with pytest.raises(
            NativeSnapshotServiceFailure, match="synthetic native encoder failure"
        ) as caught:
            service.drain(timeout=2)
        completions = caught.value.completions
        assert [row.submission_id for row in completions] == [
            first.submission_id,
            second.submission_id,
        ]
        assert [row.failure_phase for row in completions] == [
            "encode",
            "upstream_failure",
        ]
        assert all(
            row.state is NativeSnapshotSlotStateV1.FAILED for row in completions
        )
        telemetry = service.telemetry()
        assert telemetry["service_failed"] is True
        assert telemetry["failed_count"] == 2
        assert telemetry["slot_state_counts"]["FREE"] == 2
        with pytest.raises(NativeSnapshotServiceError, match="failed at submission 1"):
            service.submit(absolute_tick=3)
    finally:
        fail_first.set()
        service.shutdown(drain=True, raise_on_failure=False)


def test_submission_context_is_deep_frozen_bounded_and_accounted() -> None:
    encode_started = Event()
    release_encode = Event()

    def encoder(request):
        encode_started.set()
        if not release_encode.wait(2):
            raise AssertionError("test encoder release was not signalled")
        assert isinstance(request.context, MappingProxyType)
        nested = request.context["nested"]
        assert isinstance(nested, MappingProxyType)
        assert nested["selected"] == (1, 2)
        with pytest.raises(TypeError):
            request.context["stage_path"] = "/mutated"
        return {
            "stage_path": request.context["stage_path"],
            "selected": nested["selected"],
            "world_uid": request.context["world_uid"],
        }

    service = _service(
        capture=lambda _request: None,
        encoder=encoder,
        slot_count=1,
    )
    context = {
        "stage_path": "/bounded/stage.jsonl.gz",
        "nested": {"selected": [1, 2]},
        "world_uid": (1 << 64) - 1,
    }
    try:
        service.submit(absolute_tick=1, context=context)
        assert encode_started.wait(2)
        context["stage_path"] = "/caller-mutated"
        context["nested"]["selected"].append(3)
        active = service.telemetry()
        assert active["context_accounted_bytes_current"] > 0
        assert (
            active["context_accounted_bytes_high_water"]
            == active["context_accounted_bytes_current"]
        )
        assert active["context_accounted_bytes_per_submission_max"] > 0
        assert (
            active["context_resident_accounted_bytes"]
            == active["context_accounted_bytes_current"]
        )

        release_encode.set()
        completion = service.settle_oldest(timeout=2)
        assert completion.result == {
            "stage_path": "/bounded/stage.jsonl.gz",
            "selected": [1, 2],
            "world_uid": (1 << 64) - 1,
        }
        settled = service.telemetry()
        assert settled["context_accounted_bytes_current"] == 0
        assert settled["context_accounted_bytes_high_water"] > 0
        assert settled["context_resident_accounted_bytes"] == 0
    finally:
        release_encode.set()
        service.shutdown(drain=True, raise_on_failure=False)


@pytest.mark.parametrize(
    "context",
    [
        b"bytes-are-not-a-context",
        {"tensor": torch.zeros(1)},
        {"object": object()},
        {"integer": 1 << 64},
        {"integer": -(1 << 63) - 1},
        {"float": float("nan")},
        {"too_many": list(range(65))},
        {"too_deep": [[[[[[[1]]]]]]]},
        {"too_large": ["x" * 2048 for _ in range(64)]},
    ],
)
def test_submission_context_rejects_unbounded_or_non_value_data(context) -> None:
    service = _service(
        capture=lambda _request: None,
        encoder=lambda _request: {},
        slot_count=1,
    )
    try:
        with pytest.raises(NativeSnapshotServiceError):
            service.submit(absolute_tick=1, context=context)
        telemetry = service.telemetry()
        assert telemetry["submitted_count"] == 0
        assert telemetry["context_accounted_bytes_current"] == 0
    finally:
        service.shutdown(drain=True, raise_on_failure=False)


@pytest.mark.parametrize("integer", [-(1 << 63), (1 << 64) - 1])
def test_completion_accepts_full_64_bit_value_domain(integer: int) -> None:
    service = _service(
        capture=lambda _request: None,
        encoder=lambda _request: {"world_uid": integer},
        slot_count=1,
    )
    try:
        service.submit(absolute_tick=1)
        completion = service.settle_oldest(timeout=2)
        assert completion.result == {"world_uid": integer}
    finally:
        service.shutdown(drain=True, raise_on_failure=False)


@pytest.mark.parametrize(
    "result",
    [
        b"serialized-payload-must-not-cross-completion-boundary",
        {"tensor": torch.zeros(1)},
        {"too_many": list(range(4097))},
        {"integer": 1 << 64},
        {"integer": -(1 << 63) - 1},
    ],
)
def test_encoder_completion_must_be_a_compact_tensor_free_mapping(result) -> None:
    service = _service(
        capture=lambda _request: None,
        encoder=lambda _request: result,
        slot_count=1,
    )
    try:
        service.submit(absolute_tick=1)
        with pytest.raises(NativeSnapshotServiceFailure) as caught:
            service.drain(timeout=2)
        assert caught.value.completion.failure_phase == "encode_result"
        assert "completion" in caught.value.completion.failure_message
    finally:
        service.shutdown(drain=True, raise_on_failure=False)


@pytest.mark.parametrize("world_count", [257, 1024, 5000])
def test_dynamic_world_completion_remains_within_compact_bound(
    world_count: int,
) -> None:
    captured_worlds = [
        {
            "world_index": index,
            "world_uid": index + 1,
            "cell_id": index,
            "science_ownership": {
                "study_id": "EIEL-ECOLOGICAL-DEMAND-DISCRIMINATION-1024-V1",
                "arm_id": "arm-3697614b27184b1e9190569b2c5f1a9e",
                "replicate": index,
            },
            "science_world_sha256": {
                "algorithm": "sha256",
                "hex": f"{index % 16:x}" * 64,
            },
            "science_acquisition_sha256": {
                "algorithm": "sha256",
                "hex": f"{(index + 1) % 16:x}" * 64,
            },
            "active_agent_count": index % 513,
            "canonical_state_sha256": f"{index % 16:x}" * 64,
        }
        for index in range(world_count)
    ]
    service = _service(
        capture=lambda _request: None,
        encoder=lambda _request: {
            "selected_world_indices": list(range(world_count)),
            "selected_world_uids": list(range(1, world_count + 1)),
            "captured_worlds": captured_worlds,
        },
        slot_count=1,
        state_shape=(world_count, 1),
        metrics_shape=(world_count, 1),
    )
    try:
        service.submit(absolute_tick=1)
        completion = service.settle_oldest(timeout=2)
        assert completion.result == {
            "selected_world_indices": list(range(world_count)),
            "selected_world_uids": list(range(1, world_count + 1)),
            "captured_worlds": captured_worlds,
        }
        telemetry = service.telemetry()
        assert telemetry["descriptor_maximum_container_items"] >= world_count
    finally:
        service.shutdown(drain=True, raise_on_failure=False)


def test_unpinned_allocator_is_rejected_without_explicit_test_selector() -> None:
    with pytest.raises(NativeSnapshotServiceError, match="pinned CPU"):
        NativeSnapshotServiceV1(
            state_shape=STATE_SHAPE,
            active_science_metrics_shape=METRICS_SHAPE,
            slot_count=1,
            capture=lambda _request: None,
            encoder=lambda _request: {},
            slot_allocator=_cpu_allocator,
        )


def test_slot_allocator_rejects_overlapping_byte_ranges_with_distinct_pointers() -> None:
    backing = torch.empty(24, dtype=torch.uint8)

    def overlapping_allocator(
        shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        if dtype == torch.uint8:
            return backing[:16].reshape(shape)
        return backing[8:16].view(torch.int64).reshape(shape)

    with pytest.raises(NativeSnapshotServiceError, match="overlapping"):
        NativeSnapshotServiceV1(
            state_shape=(1, 16),
            active_science_metrics_shape=(1, 1),
            slot_count=1,
            capture=lambda _request: None,
            encoder=lambda _request: {},
            slot_allocator=overlapping_allocator,
            allow_unpinned_test_slots=True,
        )
