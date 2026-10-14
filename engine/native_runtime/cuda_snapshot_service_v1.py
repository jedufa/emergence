"""Bounded persistent capture/encode service for exact native snapshots.

The service owns a fixed number of reusable host slots.  Its default allocator
uses pinned CPU tensors; the explicit unpinned allocator escape exists only so
the state machine can be exercised on CPU-only test hosts.  Capture and encoder
functions are injected at coarse snapshot granularity.  They must not retain a
slot buffer after returning because the slot is reused after its completion is
consumed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
import sys
from threading import Condition, Lock, Thread
from time import monotonic, perf_counter_ns
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence


MIN_SIGNED_INT64_V1 = -(1 << 63)
MAX_SIGNED_INT64_V1 = (1 << 63) - 1
MAX_UNSIGNED_INT64_V1 = (1 << 64) - 1
NATIVE_SNAPSHOT_SERVICE_SCHEMA_V1 = "eiel_native_snapshot_service_v1"
MAX_COMPLETION_MAPPING_FIELDS_V1 = 64
MAX_COMPLETION_DEPTH_V1 = 6
MAX_COMPLETION_TEXT_BYTES_V1 = 16 << 10
DESCRIPTOR_BASE_CONTAINER_ITEMS_V1 = 64
DESCRIPTOR_BASE_NODES_V1 = 1024
DESCRIPTOR_NODES_PER_WORLD_V1 = 32
DESCRIPTOR_BASE_ACCOUNTED_BYTES_V1 = 64 << 10
# Conservative retained-heap accounting: every copied node is charged enough
# for its Python object/container entry as well as its encoded scalar payload.
DESCRIPTOR_NODE_ACCOUNTED_BYTES_V1 = 256
# The authenticated native publication descriptor carries parsed per-world
# cell/ownership/digest metadata, not only flat counters.  Keep its bound
# proportional to the plan's bounded per-world metadata envelope while still
# making the service completion finite and independent of payload bytes.
DESCRIPTOR_ACCOUNTED_BYTES_PER_WORLD_V1 = 16 << 10


class NativeSnapshotServiceError(RuntimeError):
    """Base error for the bounded native snapshot service."""


class NativeSnapshotBackpressureError(NativeSnapshotServiceError):
    """No reusable slot became available under the requested wait policy."""


class NativeSnapshotServiceClosedError(NativeSnapshotServiceError):
    """The service no longer accepts submissions."""


class NativeSnapshotTimeoutError(NativeSnapshotServiceError):
    """A bounded service wait expired."""


class NativeSnapshotSlotStateV1(str, Enum):
    FREE = "FREE"
    CAPTURING = "CAPTURING"
    ENCODING = "ENCODING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class NativeSnapshotSubmissionV1:
    submission_id: int
    slot_index: int
    absolute_tick: int
    _service_token: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class NativeSnapshotCaptureRequestV1:
    submission_id: int
    slot_index: int
    absolute_tick: int
    state_destination: Any
    active_science_metrics_destination: Any
    context: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class NativeSnapshotEncodeRequestV1:
    submission_id: int
    slot_index: int
    absolute_tick: int
    state: Any
    active_science_metrics: Any
    capture_metrics: Mapping[str, object]
    context: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class NativeSnapshotCompletionV1:
    submission_id: int
    slot_index: int
    absolute_tick: int
    state: NativeSnapshotSlotStateV1
    result: object | None
    capture_metrics: Mapping[str, object]
    failure_phase: str | None
    failure_type: str | None
    failure_message: str | None
    cancelled: bool


class NativeSnapshotServiceFailure(NativeSnapshotServiceError):
    """One or more bounded submissions failed on capture or encode."""

    def __init__(
        self,
        completion: NativeSnapshotCompletionV1,
        *,
        completions: Sequence[NativeSnapshotCompletionV1] = (),
    ) -> None:
        self.completion = completion
        self.completions = tuple(completions) or (completion,)
        phase = completion.failure_phase or "unknown"
        detail = completion.failure_message or "unknown failure"
        super().__init__(
            f"snapshot submission {completion.submission_id} failed during "
            f"{phase}: {detail}"
        )


class NativeSnapshotCancellation(NativeSnapshotServiceFailure):
    """A queued capture was cancelled after its buffer became safe to reuse."""


@dataclass(slots=True)
class _NativeSnapshotSlotV1:
    index: int
    state_buffer: Any
    metrics_buffer: Any
    state: NativeSnapshotSlotStateV1 = NativeSnapshotSlotStateV1.FREE
    submission_id: int | None = None
    absolute_tick: int | None = None
    context: Mapping[str, object] | None = None
    context_accounted_bytes: int = 0
    capture_fence: object | None = None
    capture_metrics: Mapping[str, object] = field(default_factory=dict)
    result: object | None = None
    failure_phase: str | None = None
    failure: BaseException | None = None
    cancelled: bool = False
    use_count: int = 0
    enqueued_ns: int = 0


@dataclass(frozen=True, slots=True)
class _DescriptorLimitsV1:
    maximum_container_items: int
    maximum_nodes: int
    maximum_accounted_bytes: int


def _checked_shape(value: Sequence[int], *, label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise NativeSnapshotServiceError(f"{label} must be an integer sequence")
    result = tuple(value)
    if not result or any(type(item) is not int or item <= 0 for item in result):
        raise NativeSnapshotServiceError(f"{label} must contain positive integers")
    elements = 1
    for item in result:
        if elements > sys.maxsize // item:
            raise NativeSnapshotServiceError(f"{label} exceeds host address space")
        elements *= item
    return result


def _checked_timeout(timeout: float | None) -> float | None:
    if timeout is None:
        return None
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or timeout < 0
    ):
        raise NativeSnapshotServiceError("snapshot service timeout is invalid")
    return float(timeout)


def _remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, deadline - monotonic())


def _tensor_nbytes(value: Any) -> int:
    return int(value.numel()) * int(value.element_size())


def _checked_scaled_limit(
    *, base: int, per_world: int, world_count: int, label: str
) -> int:
    if world_count > (sys.maxsize - base) // per_world:
        raise NativeSnapshotServiceError(f"{label} exceeds host address space")
    return base + world_count * per_world


def _descriptor_limits(world_count: int) -> _DescriptorLimitsV1:
    return _DescriptorLimitsV1(
        maximum_container_items=max(
            DESCRIPTOR_BASE_CONTAINER_ITEMS_V1, world_count
        ),
        maximum_nodes=_checked_scaled_limit(
            base=DESCRIPTOR_BASE_NODES_V1,
            per_world=DESCRIPTOR_NODES_PER_WORLD_V1,
            world_count=world_count,
            label="snapshot descriptor node bound",
        ),
        maximum_accounted_bytes=_checked_scaled_limit(
            base=DESCRIPTOR_BASE_ACCOUNTED_BYTES_V1,
            per_world=DESCRIPTOR_ACCOUNTED_BYTES_PER_WORLD_V1,
            world_count=world_count,
            label="snapshot descriptor byte bound",
        ),
    )


def _bounded_descriptor_mapping(
    value: object,
    *,
    limits: _DescriptorLimitsV1,
    label: str,
    immutable: bool,
) -> tuple[Mapping[str, object], int]:
    """Copy one strictly bounded JSON-like mapping and account its payload."""

    visited = 0
    accounted_bytes = 0

    def charge(byte_count: int) -> None:
        nonlocal accounted_bytes
        if byte_count > limits.maximum_accounted_bytes - accounted_bytes:
            raise NativeSnapshotServiceError(
                f"{label} exceeds the aggregate compact byte bound"
            )
        accounted_bytes += byte_count

    def normalize(item: object, *, depth: int) -> object:
        nonlocal visited
        visited += 1
        if visited > limits.maximum_nodes:
            raise NativeSnapshotServiceError(
                f"{label} exceeds the compact node bound"
            )
        if depth > MAX_COMPLETION_DEPTH_V1:
            raise NativeSnapshotServiceError(
                f"{label} exceeds the compact depth bound"
            )
        charge(DESCRIPTOR_NODE_ACCOUNTED_BYTES_V1)
        if item is None:
            return None
        if type(item) is bool:
            charge(1)
            return item
        if type(item) is int:
            if not MIN_SIGNED_INT64_V1 <= item <= MAX_UNSIGNED_INT64_V1:
                raise NativeSnapshotServiceError(
                    f"{label} integer exceeds the 64-bit value domain"
                )
            charge(8)
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise NativeSnapshotServiceError(
                    f"{label} contains a non-finite float"
                )
            charge(8)
            return item
        if type(item) is str:
            encoded = item.encode("utf-8")
            if len(encoded) > MAX_COMPLETION_TEXT_BYTES_V1:
                raise NativeSnapshotServiceError(
                    f"{label} text exceeds the compact bound"
                )
            charge(len(encoded))
            return item
        if isinstance(item, Mapping):
            if len(item) > MAX_COMPLETION_MAPPING_FIELDS_V1:
                raise NativeSnapshotServiceError(
                    f"{label} mapping exceeds the compact bound"
                )
            charge(8)
            normalized: dict[str, object] = {}
            for key, child in item.items():
                if type(key) is not str or not key:
                    raise NativeSnapshotServiceError(
                        f"{label} mapping key is invalid"
                    )
                encoded_key = key.encode("utf-8")
                if len(encoded_key) > 256:
                    raise NativeSnapshotServiceError(
                        f"{label} mapping key is invalid"
                    )
                charge(len(encoded_key) + 1)
                normalized[key] = normalize(child, depth=depth + 1)
            if immutable:
                return MappingProxyType(normalized)
            return normalized
        if type(item) in (list, tuple):
            if len(item) > limits.maximum_container_items:
                raise NativeSnapshotServiceError(
                    f"{label} container exceeds the compact bound"
                )
            charge(8)
            normalized_children = tuple(
                normalize(child, depth=depth + 1) for child in item
            )
            if immutable:
                return normalized_children
            return list(normalized_children)
        raise NativeSnapshotServiceError(
            f"{label} must not contain bytes, tensors, "
            f"or non-compact objects ({type(item).__name__})"
        )

    if not isinstance(value, Mapping):
        raise NativeSnapshotServiceError(f"{label} must be a compact mapping")
    result = normalize(value, depth=0)
    if not isinstance(result, Mapping):  # pragma: no cover - top-level invariant
        raise NativeSnapshotServiceError(f"{label} mapping normalization failed")
    return result, accounted_bytes


def _compact_completion_mapping(
    value: object,
    *,
    limits: _DescriptorLimitsV1,
    label: str = "snapshot encoder completion",
) -> Mapping[str, object]:
    result, _accounted_bytes = _bounded_descriptor_mapping(
        value,
        limits=limits,
        label=label,
        immutable=False,
    )
    return result


def _immutable_context_mapping(
    value: object | None, *, limits: _DescriptorLimitsV1
) -> tuple[Mapping[str, object] | None, int]:
    if value is None:
        return None, 0
    return _bounded_descriptor_mapping(
        value,
        limits=limits,
        label="snapshot submission context",
        immutable=True,
    )


def _default_pinned_allocator(shape: tuple[int, ...], dtype: Any) -> Any:
    import torch

    return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)


class NativeSnapshotServiceV1:
    """One-worker, bounded FIFO service over reusable full-snapshot slots.

    ``capture`` runs on the submitting/runtime-owner thread and may return
    ``None`` for a synchronous capture or an object exposing ``wait()``.  The
    persistent worker calls ``wait()`` before exposing either buffer to
    ``encoder``.  A capture fence's optional return value must be a mapping of
    compact timing/copy telemetry.

    ``encoder`` runs on the persistent worker and receives one coarse request.
    It must return a compact JSON-like mapping and must not retain either
    reusable buffer after returning.  Per-submission ``context`` is copied into
    a deeply immutable, 64-bit-value JSON-like mapping under envelope-derived
    cardinality and aggregate-byte bounds; cold plans belong in the injected
    closures rather than in this descriptor.  Provider code should settle the
    oldest completion before a nonblocking submit whenever ``has_capacity`` is
    false; internal submission waits never consume terminal slots.
    """

    def __init__(
        self,
        *,
        state_shape: Sequence[int],
        active_science_metrics_shape: Sequence[int],
        slot_count: int,
        capture: Callable[[NativeSnapshotCaptureRequestV1], object | None],
        encoder: Callable[[NativeSnapshotEncodeRequestV1], object],
        slot_allocator: Callable[[tuple[int, ...], Any], Any] | None = None,
        allow_unpinned_test_slots: bool = False,
        worker_name: str = "eiel-native-snapshot-v1",
    ) -> None:
        import torch

        self._state_shape = _checked_shape(state_shape, label="snapshot state shape")
        self._metrics_shape = _checked_shape(
            active_science_metrics_shape,
            label="snapshot active-science metrics shape",
        )
        if self._state_shape[0] != self._metrics_shape[0]:
            raise NativeSnapshotServiceError(
                "snapshot state and active-science metrics world counts differ"
            )
        self._world_count = self._state_shape[0]
        self._descriptor_limits = _descriptor_limits(self._world_count)
        if type(slot_count) is not int or slot_count <= 0:
            raise NativeSnapshotServiceError("snapshot slot_count must be positive")
        if not callable(capture) or not callable(encoder):
            raise NativeSnapshotServiceError(
                "snapshot capture and encoder must be callable"
            )
        if type(allow_unpinned_test_slots) is not bool:
            raise NativeSnapshotServiceError(
                "snapshot unpinned-test selector must be boolean"
            )
        if allow_unpinned_test_slots and slot_allocator is None:
            raise NativeSnapshotServiceError(
                "unpinned test slots require an explicit test allocator"
            )
        if not isinstance(worker_name, str) or not worker_name:
            raise NativeSnapshotServiceError("snapshot worker name is invalid")

        allocator = slot_allocator or _default_pinned_allocator
        self._capture = capture
        self._encoder = encoder
        self._slot_count = slot_count
        self._service_token = object()
        self._condition = Condition(Lock())
        self._submit_lock = Lock()
        self._settlement_lock = Lock()
        self._pending: deque[int] = deque()
        self._terminal: deque[int] = deque()
        self._finished_by_submission: dict[int, int] = {}
        self._slot_by_submission: dict[int, int] = {}
        self._free: deque[int] = deque(range(slot_count))
        self._next_submission_id = 1
        self._next_terminal_submission_id = 1
        self._accepting = True
        self._stopping = False
        self._worker_stopped = False
        self._service_failure: tuple[int, str, BaseException] | None = None

        self._submitted_count = 0
        self._completed_count = 0
        self._failed_count = 0
        self._cancelled_count = 0
        self._queue_high_water = 0
        self._terminal_high_water = 0
        self._in_use_high_water = 0
        self._reuse_count = 0
        self._backpressure_count = 0
        self._backpressure_timeout_count = 0
        self._producer_blocked_count = 0
        self._producer_submit_blocked_count = 0
        self._producer_settle_blocked_count = 0
        self._producer_blocked_elapsed_ns_total = 0
        self._producer_blocked_elapsed_ns_max = 0
        self._producer_settle_blocked_elapsed_ns_total = 0
        self._producer_settle_blocked_elapsed_ns_max = 0
        self._producer_capture_elapsed_ns_total = 0
        self._producer_capture_elapsed_ns_max = 0
        self._queue_wait_elapsed_ns_total = 0
        self._queue_wait_elapsed_ns_max = 0
        self._worker_capture_wait_elapsed_ns_total = 0
        self._worker_capture_wait_elapsed_ns_max = 0
        self._worker_encode_elapsed_ns_total = 0
        self._worker_encode_elapsed_ns_max = 0
        self._worker_job_elapsed_ns_total = 0
        self._worker_job_elapsed_ns_max = 0
        self._context_accounted_bytes_current = 0
        self._context_accounted_bytes_high_water = 0
        self._context_accounted_bytes_per_submission_max = 0

        self._slots: list[_NativeSnapshotSlotV1] = []
        extents: list[tuple[int, int]] = []
        for index in range(slot_count):
            state_buffer = allocator(self._state_shape, torch.uint8)
            metrics_buffer = allocator(self._metrics_shape, torch.int64)
            self._validate_slot_buffer(
                state_buffer,
                shape=self._state_shape,
                dtype=torch.uint8,
                label=f"snapshot state slot {index}",
                require_pinned=not allow_unpinned_test_slots,
            )
            self._validate_slot_buffer(
                metrics_buffer,
                shape=self._metrics_shape,
                dtype=torch.int64,
                label=f"snapshot metrics slot {index}",
                require_pinned=not allow_unpinned_test_slots,
            )
            for buffer in (state_buffer, metrics_buffer):
                start = int(buffer.data_ptr())
                end = start + _tensor_nbytes(buffer)
                if any(start < other_end and other_start < end for other_start, other_end in extents):
                    raise NativeSnapshotServiceError(
                        "snapshot slot allocator returned overlapping reusable buffers"
                    )
                extents.append((start, end))
            self._slots.append(
                _NativeSnapshotSlotV1(
                    index=index,
                    state_buffer=state_buffer,
                    metrics_buffer=metrics_buffer,
                )
            )

        self._state_bytes_per_slot = _tensor_nbytes(self._slots[0].state_buffer)
        self._metrics_bytes_per_slot = _tensor_nbytes(
            self._slots[0].metrics_buffer
        )
        self._bytes_per_slot = (
            self._state_bytes_per_slot + self._metrics_bytes_per_slot
        )
        self._pinned_memory_verified = not allow_unpinned_test_slots
        self._worker = Thread(target=self._worker_main, name=worker_name, daemon=True)
        self._worker.start()

    @staticmethod
    def _validate_slot_buffer(
        value: Any,
        *,
        shape: tuple[int, ...],
        dtype: Any,
        label: str,
        require_pinned: bool,
    ) -> None:
        if (
            tuple(value.shape) != shape
            or value.dtype != dtype
            or value.device.type != "cpu"
            or not value.is_contiguous()
            or (require_pinned and not value.is_pinned())
        ):
            memory = "pinned CPU" if require_pinned else "contiguous CPU"
            raise NativeSnapshotServiceError(
                f"{label} must be an exact {memory} tensor"
            )

    def __enter__(self) -> NativeSnapshotServiceV1:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.shutdown(
            cancel_pending=exc is not None,
            drain=True,
            raise_on_failure=exc is None,
        )

    def _raise_if_unavailable_locked(self) -> None:
        if not self._accepting:
            raise NativeSnapshotServiceClosedError(
                "native snapshot service is not accepting submissions"
            )
        if self._service_failure is not None:
            submission_id, phase, error = self._service_failure
            raise NativeSnapshotServiceError(
                f"native snapshot service failed at submission {submission_id} "
                f"during {phase}: {error}"
            )

    def submit(
        self,
        *,
        absolute_tick: int,
        context: Mapping[str, object] | None = None,
        block: bool = False,
        timeout: float | None = None,
    ) -> NativeSnapshotSubmissionV1:
        if (
            type(absolute_tick) is not int
            or absolute_tick < 0
            or absolute_tick > MAX_SIGNED_INT64_V1
        ):
            raise NativeSnapshotServiceError(
                "snapshot absolute_tick must be a non-negative signed-int64"
            )
        if type(block) is not bool:
            raise NativeSnapshotServiceError("snapshot block selector must be boolean")
        timeout = _checked_timeout(timeout)
        deadline = None if timeout is None else monotonic() + timeout

        with self._submit_lock:
            with self._condition:
                self._raise_if_unavailable_locked()
                waited = False
                wait_started_ns = 0
                try:
                    while not self._free:
                        self._raise_if_unavailable_locked()
                        if self._terminal:
                            if not waited:
                                self._backpressure_count += 1
                            raise NativeSnapshotBackpressureError(
                                "oldest native snapshot completion must be "
                                "settled before submission"
                            )
                        if not block:
                            self._backpressure_count += 1
                            raise NativeSnapshotBackpressureError(
                                "native snapshot slots are saturated"
                            )
                        if not waited:
                            self._backpressure_count += 1
                            self._producer_blocked_count += 1
                            self._producer_submit_blocked_count += 1
                            waited = True
                            wait_started_ns = perf_counter_ns()
                        remaining = _remaining(deadline)
                        if remaining == 0:
                            self._backpressure_timeout_count += 1
                            raise NativeSnapshotBackpressureError(
                                "timed out waiting for a native snapshot slot"
                            )
                        self._condition.wait(remaining)
                finally:
                    if wait_started_ns:
                        elapsed_ns = perf_counter_ns() - wait_started_ns
                        self._producer_blocked_elapsed_ns_total += elapsed_ns
                        self._producer_blocked_elapsed_ns_max = max(
                            self._producer_blocked_elapsed_ns_max, elapsed_ns
                        )

            # Only the serialized submitter freezes a context, and only after a
            # reusable slot is available.  Concurrent blocked producers cannot
            # accumulate unaccounted descriptor copies.
            frozen_context, context_accounted_bytes = _immutable_context_mapping(
                context,
                limits=self._descriptor_limits,
            )
            with self._condition:
                # Shutdown or worker failure may have occurred while copying the
                # caller descriptor; no slot has been claimed yet.
                self._raise_if_unavailable_locked()
                if not self._free:  # pragma: no cover - submit serialization
                    raise NativeSnapshotServiceError(
                        "snapshot free-slot invariant changed during context copy"
                    )
                slot = self._slots[self._free.popleft()]
                submission_id = self._next_submission_id
                self._next_submission_id += 1
                slot.state = NativeSnapshotSlotStateV1.CAPTURING
                slot.submission_id = submission_id
                slot.absolute_tick = absolute_tick
                slot.context = frozen_context
                slot.context_accounted_bytes = context_accounted_bytes
                slot.capture_fence = None
                slot.capture_metrics = {}
                slot.result = None
                slot.failure_phase = None
                slot.failure = None
                slot.cancelled = False
                slot.use_count += 1
                if slot.use_count > 1:
                    self._reuse_count += 1
                self._slot_by_submission[submission_id] = slot.index
                self._submitted_count += 1
                self._context_accounted_bytes_current += context_accounted_bytes
                self._context_accounted_bytes_high_water = max(
                    self._context_accounted_bytes_high_water,
                    self._context_accounted_bytes_current,
                )
                self._context_accounted_bytes_per_submission_max = max(
                    self._context_accounted_bytes_per_submission_max,
                    context_accounted_bytes,
                )
                self._in_use_high_water = max(
                    self._in_use_high_water,
                    self._slot_count - len(self._free),
                )

            request = NativeSnapshotCaptureRequestV1(
                submission_id=submission_id,
                slot_index=slot.index,
                absolute_tick=absolute_tick,
                state_destination=slot.state_buffer,
                active_science_metrics_destination=slot.metrics_buffer,
                context=frozen_context,
            )
            capture_started_ns = perf_counter_ns()
            try:
                fence = self._capture(request)
            except BaseException as exc:
                capture_elapsed_ns = perf_counter_ns() - capture_started_ns
                with self._condition:
                    self._producer_capture_elapsed_ns_total += capture_elapsed_ns
                    self._producer_capture_elapsed_ns_max = max(
                        self._producer_capture_elapsed_ns_max, capture_elapsed_ns
                    )
                    self._finish_failed_locked(
                        slot,
                        phase="capture",
                        error=exc,
                        poison=True,
                        cancelled=False,
                    )
                    completion = self._completion_from_slot_locked(slot)
                raise NativeSnapshotServiceFailure(completion) from exc

            capture_elapsed_ns = perf_counter_ns() - capture_started_ns
            with self._condition:
                self._producer_capture_elapsed_ns_total += capture_elapsed_ns
                self._producer_capture_elapsed_ns_max = max(
                    self._producer_capture_elapsed_ns_max, capture_elapsed_ns
                )
                slot.capture_fence = fence
                slot.enqueued_ns = perf_counter_ns()
                self._pending.append(slot.index)
                self._queue_high_water = max(
                    self._queue_high_water, len(self._pending)
                )
                self._condition.notify_all()
            return NativeSnapshotSubmissionV1(
                submission_id=submission_id,
                slot_index=slot.index,
                absolute_tick=absolute_tick,
                _service_token=self._service_token,
            )

    def _await_capture(self, fence: object | None) -> Mapping[str, object]:
        if fence is None:
            return {}
        waiter = getattr(fence, "wait", None)
        if not callable(waiter):
            raise NativeSnapshotServiceError(
                "snapshot capture fence does not expose wait()"
            )
        result = waiter()
        if result is None:
            return {}
        return _compact_completion_mapping(
            result,
            limits=self._descriptor_limits,
            label="snapshot capture fence telemetry",
        )

    def _worker_main(self) -> None:
        while True:
            with self._condition:
                while not self._pending:
                    if self._stopping:
                        self._worker_stopped = True
                        self._condition.notify_all()
                        return
                    self._condition.wait()
                slot = self._slots[self._pending.popleft()]
                fence = slot.capture_fence
                worker_job_started_ns = perf_counter_ns()
                queue_wait_elapsed_ns = worker_job_started_ns - slot.enqueued_ns
                self._queue_wait_elapsed_ns_total += queue_wait_elapsed_ns
                self._queue_wait_elapsed_ns_max = max(
                    self._queue_wait_elapsed_ns_max, queue_wait_elapsed_ns
                )

            capture_wait_started_ns = perf_counter_ns()
            try:
                capture_metrics = self._await_capture(fence)
            except BaseException as exc:
                capture_wait_elapsed_ns = perf_counter_ns() - capture_wait_started_ns
                worker_job_elapsed_ns = perf_counter_ns() - worker_job_started_ns
                with self._condition:
                    self._record_worker_elapsed_locked(
                        capture_wait_elapsed_ns=capture_wait_elapsed_ns,
                        encode_elapsed_ns=0,
                        worker_job_elapsed_ns=worker_job_elapsed_ns,
                    )
                    self._finish_failed_locked(
                        slot,
                        phase="capture_wait",
                        error=exc,
                        poison=True,
                        cancelled=False,
                    )
                continue

            capture_wait_elapsed_ns = perf_counter_ns() - capture_wait_started_ns
            with self._condition:
                slot.capture_metrics = capture_metrics
                if slot.cancelled:
                    self._record_worker_elapsed_locked(
                        capture_wait_elapsed_ns=capture_wait_elapsed_ns,
                        encode_elapsed_ns=0,
                        worker_job_elapsed_ns=(
                            perf_counter_ns() - worker_job_started_ns
                        ),
                    )
                    self._finish_failed_locked(
                        slot,
                        phase="cancelled",
                        error=NativeSnapshotServiceError(
                            "snapshot submission was cancelled"
                        ),
                        poison=False,
                        cancelled=True,
                    )
                    continue
                if self._service_failure is not None:
                    failed_submission, failed_phase, failure = self._service_failure
                    self._record_worker_elapsed_locked(
                        capture_wait_elapsed_ns=capture_wait_elapsed_ns,
                        encode_elapsed_ns=0,
                        worker_job_elapsed_ns=(
                            perf_counter_ns() - worker_job_started_ns
                        ),
                    )
                    self._finish_failed_locked(
                        slot,
                        phase="upstream_failure",
                        error=NativeSnapshotServiceError(
                            "snapshot service stopped after submission "
                            f"{failed_submission} failed during {failed_phase}: {failure}"
                        ),
                        poison=False,
                        cancelled=False,
                    )
                    continue
                slot.state = NativeSnapshotSlotStateV1.ENCODING
                request = NativeSnapshotEncodeRequestV1(
                    submission_id=slot.submission_id,
                    slot_index=slot.index,
                    absolute_tick=slot.absolute_tick,
                    state=slot.state_buffer,
                    active_science_metrics=slot.metrics_buffer,
                    capture_metrics=dict(slot.capture_metrics),
                    context=slot.context,
                )
                self._condition.notify_all()

            encode_started_ns = perf_counter_ns()
            try:
                encoded = self._encoder(request)
            except BaseException as exc:
                encode_elapsed_ns = perf_counter_ns() - encode_started_ns
                with self._condition:
                    self._record_worker_elapsed_locked(
                        capture_wait_elapsed_ns=capture_wait_elapsed_ns,
                        encode_elapsed_ns=encode_elapsed_ns,
                        worker_job_elapsed_ns=(
                            perf_counter_ns() - worker_job_started_ns
                        ),
                    )
                    self._finish_failed_locked(
                        slot,
                        phase="encode",
                        error=exc,
                        poison=True,
                        cancelled=False,
                    )
                continue

            try:
                result = _compact_completion_mapping(
                    encoded,
                    limits=self._descriptor_limits,
                )
            except BaseException as exc:
                encode_elapsed_ns = perf_counter_ns() - encode_started_ns
                with self._condition:
                    self._record_worker_elapsed_locked(
                        capture_wait_elapsed_ns=capture_wait_elapsed_ns,
                        encode_elapsed_ns=encode_elapsed_ns,
                        worker_job_elapsed_ns=(
                            perf_counter_ns() - worker_job_started_ns
                        ),
                    )
                    self._finish_failed_locked(
                        slot,
                        phase="encode_result",
                        error=exc,
                        poison=True,
                        cancelled=False,
                    )
                continue

            encode_elapsed_ns = perf_counter_ns() - encode_started_ns
            with self._condition:
                self._record_worker_elapsed_locked(
                    capture_wait_elapsed_ns=capture_wait_elapsed_ns,
                    encode_elapsed_ns=encode_elapsed_ns,
                    worker_job_elapsed_ns=perf_counter_ns() - worker_job_started_ns,
                )
                slot.result = result
                slot.state = NativeSnapshotSlotStateV1.COMPLETED
                self._completed_count += 1
                self._publish_terminal_locked(slot)

    def _record_worker_elapsed_locked(
        self,
        *,
        capture_wait_elapsed_ns: int,
        encode_elapsed_ns: int,
        worker_job_elapsed_ns: int,
    ) -> None:
        self._worker_capture_wait_elapsed_ns_total += capture_wait_elapsed_ns
        self._worker_capture_wait_elapsed_ns_max = max(
            self._worker_capture_wait_elapsed_ns_max, capture_wait_elapsed_ns
        )
        self._worker_encode_elapsed_ns_total += encode_elapsed_ns
        self._worker_encode_elapsed_ns_max = max(
            self._worker_encode_elapsed_ns_max, encode_elapsed_ns
        )
        self._worker_job_elapsed_ns_total += worker_job_elapsed_ns
        self._worker_job_elapsed_ns_max = max(
            self._worker_job_elapsed_ns_max, worker_job_elapsed_ns
        )

    def _finish_failed_locked(
        self,
        slot: _NativeSnapshotSlotV1,
        *,
        phase: str,
        error: BaseException,
        poison: bool,
        cancelled: bool,
    ) -> None:
        slot.state = NativeSnapshotSlotStateV1.FAILED
        slot.failure_phase = phase
        slot.failure = error
        slot.cancelled = cancelled
        self._failed_count += 1
        if cancelled:
            self._cancelled_count += 1
        if poison and self._service_failure is None:
            self._service_failure = (slot.submission_id, phase, error)
        self._publish_terminal_locked(slot)

    def _publish_terminal_locked(self, slot: _NativeSnapshotSlotV1) -> None:
        self._finished_by_submission[slot.submission_id] = slot.index
        while self._next_terminal_submission_id in self._finished_by_submission:
            ready = self._finished_by_submission.pop(
                self._next_terminal_submission_id
            )
            self._terminal.append(ready)
            self._next_terminal_submission_id += 1
        self._terminal_high_water = max(
            self._terminal_high_water, len(self._terminal)
        )
        self._condition.notify_all()

    def _completion_from_slot_locked(
        self, slot: _NativeSnapshotSlotV1
    ) -> NativeSnapshotCompletionV1:
        failure = slot.failure
        return NativeSnapshotCompletionV1(
            submission_id=slot.submission_id,
            slot_index=slot.index,
            absolute_tick=slot.absolute_tick,
            state=slot.state,
            result=slot.result,
            capture_metrics=dict(slot.capture_metrics),
            failure_phase=slot.failure_phase,
            failure_type=None if failure is None else type(failure).__name__,
            failure_message=None if failure is None else str(failure),
            cancelled=slot.cancelled,
        )

    def _consume_terminal_locked(self) -> NativeSnapshotCompletionV1:
        slot = self._slots[self._terminal.popleft()]
        completion = self._completion_from_slot_locked(slot)
        del self._slot_by_submission[slot.submission_id]
        self._context_accounted_bytes_current -= slot.context_accounted_bytes
        if self._context_accounted_bytes_current < 0:  # pragma: no cover
            raise NativeSnapshotServiceError(
                "snapshot context byte accounting underflow"
            )
        slot.state = NativeSnapshotSlotStateV1.FREE
        slot.submission_id = None
        slot.absolute_tick = None
        slot.context = None
        slot.context_accounted_bytes = 0
        slot.capture_fence = None
        slot.capture_metrics = {}
        slot.result = None
        slot.failure_phase = None
        slot.failure = None
        slot.cancelled = False
        slot.enqueued_ns = 0
        self._free.append(slot.index)
        self._condition.notify_all()
        return completion

    @staticmethod
    def _raise_completion_failure(
        completion: NativeSnapshotCompletionV1,
        *,
        completions: Sequence[NativeSnapshotCompletionV1] = (),
    ) -> None:
        error_type = (
            NativeSnapshotCancellation
            if completion.cancelled
            else NativeSnapshotServiceFailure
        )
        raise error_type(completion, completions=completions)

    def _record_producer_settle_wait(self, elapsed_ns: int) -> None:
        if elapsed_ns <= 0:
            return
        with self._condition:
            self._backpressure_count += 1
            self._producer_blocked_count += 1
            self._producer_settle_blocked_count += 1
            self._producer_blocked_elapsed_ns_total += elapsed_ns
            self._producer_blocked_elapsed_ns_max = max(
                self._producer_blocked_elapsed_ns_max, elapsed_ns
            )
            self._producer_settle_blocked_elapsed_ns_total += elapsed_ns
            self._producer_settle_blocked_elapsed_ns_max = max(
                self._producer_settle_blocked_elapsed_ns_max, elapsed_ns
            )

    def _acquire_settlement(
        self, *, block: bool, deadline: float | None
    ) -> int:
        if self._settlement_lock.acquire(blocking=False):
            return 0
        if not block:
            raise NativeSnapshotTimeoutError(
                "another native snapshot completion consumer is active"
            )
        started_ns = perf_counter_ns()
        if deadline is None:
            acquired = self._settlement_lock.acquire()
        else:
            acquired = self._settlement_lock.acquire(timeout=_remaining(deadline))
        elapsed_ns = perf_counter_ns() - started_ns
        if not acquired:
            self._record_producer_settle_wait(elapsed_ns)
            raise NativeSnapshotTimeoutError(
                "timed out waiting for the native snapshot completion consumer"
            )
        return elapsed_ns

    @property
    def has_capacity(self) -> bool:
        """Whether a nonblocking producer submission can claim a slot now."""

        with self._condition:
            return bool(
                self._free
                and self._accepting
                and self._service_failure is None
            )

    @property
    def has_pending(self) -> bool:
        """Whether any submitted snapshot still owns a bounded slot."""

        with self._condition:
            return bool(self._slot_by_submission)

    @property
    def ready(self) -> bool:
        """Whether the oldest FIFO completion can be settled without waiting."""

        with self._condition:
            return bool(self._terminal)

    def settle_oldest(
        self,
        *,
        block: bool = True,
        timeout: float | None = None,
        raise_on_failure: bool = True,
    ) -> NativeSnapshotCompletionV1:
        """Consume the oldest FIFO completion and release its reusable slot."""

        return self.next_completion(
            block=block,
            timeout=timeout,
            raise_on_failure=raise_on_failure,
        )

    def next_completion(
        self,
        *,
        block: bool = True,
        timeout: float | None = None,
        raise_on_failure: bool = True,
    ) -> NativeSnapshotCompletionV1:
        if type(block) is not bool or type(raise_on_failure) is not bool:
            raise NativeSnapshotServiceError(
                "snapshot completion selectors must be boolean"
            )
        timeout = _checked_timeout(timeout)
        deadline = None if timeout is None else monotonic() + timeout
        settle_wait_ns = self._acquire_settlement(block=block, deadline=deadline)
        try:
            with self._condition:
                while not self._terminal:
                    if not block:
                        raise NativeSnapshotTimeoutError(
                            "no native snapshot completion is ready"
                        )
                    if self._worker_stopped and not self._slot_by_submission:
                        raise NativeSnapshotServiceClosedError(
                            "native snapshot service has no remaining completions"
                        )
                    remaining = _remaining(deadline)
                    if remaining == 0:
                        raise NativeSnapshotTimeoutError(
                            "timed out waiting for native snapshot completion"
                        )
                    wait_started_ns = perf_counter_ns()
                    self._condition.wait(remaining)
                    settle_wait_ns += perf_counter_ns() - wait_started_ns
                completion = self._consume_terminal_locked()
        finally:
            self._settlement_lock.release()
            self._record_producer_settle_wait(settle_wait_ns)
        if completion.state is NativeSnapshotSlotStateV1.FAILED and raise_on_failure:
            self._raise_completion_failure(completion)
        return completion

    def drain(
        self,
        *,
        timeout: float | None = None,
        raise_on_failure: bool = True,
    ) -> tuple[NativeSnapshotCompletionV1, ...]:
        if type(raise_on_failure) is not bool:
            raise NativeSnapshotServiceError(
                "snapshot failure selector must be boolean"
            )
        timeout = _checked_timeout(timeout)
        deadline = None if timeout is None else monotonic() + timeout
        settle_wait_ns = self._acquire_settlement(block=True, deadline=deadline)
        try:
            with self._condition:
                target = set(self._slot_by_submission)
                while any(
                    self._slots[self._slot_by_submission[submission_id]].state
                    not in (
                        NativeSnapshotSlotStateV1.COMPLETED,
                        NativeSnapshotSlotStateV1.FAILED,
                    )
                    for submission_id in target
                ):
                    remaining = _remaining(deadline)
                    if remaining == 0:
                        raise NativeSnapshotTimeoutError(
                            "timed out draining native snapshot submissions"
                        )
                    wait_started_ns = perf_counter_ns()
                    self._condition.wait(remaining)
                    settle_wait_ns += perf_counter_ns() - wait_started_ns
                completions: list[NativeSnapshotCompletionV1] = []
                while target:
                    if not self._terminal:
                        raise NativeSnapshotServiceError(
                            "snapshot terminal ordering invariant failed"
                        )
                    completion = self._consume_terminal_locked()
                    if completion.submission_id not in target:
                        raise NativeSnapshotServiceError(
                            "snapshot drain crossed its bounded submission set"
                        )
                    target.remove(completion.submission_id)
                    completions.append(completion)
        finally:
            self._settlement_lock.release()
            self._record_producer_settle_wait(settle_wait_ns)
        result = tuple(completions)
        if raise_on_failure:
            failed = next(
                (
                    item
                    for item in result
                    if item.state is NativeSnapshotSlotStateV1.FAILED
                ),
                None,
            )
            if failed is not None:
                self._raise_completion_failure(failed, completions=result)
        return result

    def cancel(self, submission: NativeSnapshotSubmissionV1) -> bool:
        if (
            not isinstance(submission, NativeSnapshotSubmissionV1)
            or submission._service_token is not self._service_token
        ):
            raise NativeSnapshotServiceError(
                "snapshot submission belongs to another service"
            )
        with self._condition:
            index = self._slot_by_submission.get(submission.submission_id)
            if index is None:
                return False
            slot = self._slots[index]
            if slot.state is not NativeSnapshotSlotStateV1.CAPTURING:
                return False
            slot.cancelled = True
            self._condition.notify_all()
            return True

    def cancel_pending(self) -> int:
        with self._condition:
            cancelled = 0
            for slot in self._slots:
                if (
                    slot.state is NativeSnapshotSlotStateV1.CAPTURING
                    and not slot.cancelled
                ):
                    slot.cancelled = True
                    cancelled += 1
            self._condition.notify_all()
            return cancelled

    def shutdown(
        self,
        *,
        cancel_pending: bool = False,
        drain: bool = True,
        timeout: float | None = None,
        raise_on_failure: bool = True,
    ) -> tuple[NativeSnapshotCompletionV1, ...]:
        if (
            type(cancel_pending) is not bool
            or type(drain) is not bool
            or type(raise_on_failure) is not bool
        ):
            raise NativeSnapshotServiceError(
                "snapshot shutdown selectors must be boolean"
            )
        timeout = _checked_timeout(timeout)
        deadline = None if timeout is None else monotonic() + timeout
        # Close admission before waiting behind the serialized submitter.  An
        # already-admitted capture may still finish, but a finite shutdown
        # deadline must not wait indefinitely for its callback.
        with self._condition:
            self._accepting = False
            self._condition.notify_all()
        if deadline is None:
            submit_acquired = self._submit_lock.acquire()
        else:
            submit_acquired = self._submit_lock.acquire(timeout=_remaining(deadline))
        if not submit_acquired:
            raise NativeSnapshotTimeoutError(
                "timed out waiting for an in-progress native snapshot capture"
            )
        try:
            with self._condition:
                self._stopping = True
                if cancel_pending:
                    for slot in self._slots:
                        if slot.state is NativeSnapshotSlotStateV1.CAPTURING:
                            slot.cancelled = True
                self._condition.notify_all()
        finally:
            self._submit_lock.release()
        self._worker.join(_remaining(deadline))
        if self._worker.is_alive():
            raise NativeSnapshotTimeoutError(
                "timed out shutting down native snapshot worker"
            )
        if not drain:
            return ()
        return self.drain(
            timeout=_remaining(deadline), raise_on_failure=raise_on_failure
        )

    def telemetry(self) -> Mapping[str, object]:
        with self._condition:
            state_counts = {
                state.value: sum(slot.state is state for slot in self._slots)
                for state in NativeSnapshotSlotStateV1
            }
            in_use = self._slot_count - len(self._free)
            return {
                "schema_version": NATIVE_SNAPSHOT_SERVICE_SCHEMA_V1,
                "worker_count": 1,
                "worker_alive": self._worker.is_alive(),
                "accepting": self._accepting,
                "stopping": self._stopping,
                "service_failed": self._service_failure is not None,
                "slot_count": self._slot_count,
                "slot_states": [slot.state.value for slot in self._slots],
                "slot_state_counts": state_counts,
                "queue_capacity": self._slot_count,
                "queue_depth": len(self._pending),
                "queue_high_water": self._queue_high_water,
                "terminal_depth": len(self._terminal),
                "terminal_high_water": self._terminal_high_water,
                "in_use_slots": in_use,
                "in_use_slots_high_water": self._in_use_high_water,
                "submitted_count": self._submitted_count,
                "completed_count": self._completed_count,
                "failed_count": self._failed_count,
                "cancelled_count": self._cancelled_count,
                "slot_reuse_count": self._reuse_count,
                "backpressure_count": self._backpressure_count,
                "backpressure_timeout_count": self._backpressure_timeout_count,
                "producer_blocked_count": self._producer_blocked_count,
                "producer_submit_blocked_count": (
                    self._producer_submit_blocked_count
                ),
                "producer_settle_blocked_count": (
                    self._producer_settle_blocked_count
                ),
                "producer_blocked_elapsed_ns_total": (
                    self._producer_blocked_elapsed_ns_total
                ),
                "producer_blocked_elapsed_ns_max": (
                    self._producer_blocked_elapsed_ns_max
                ),
                "producer_settle_blocked_elapsed_ns_total": (
                    self._producer_settle_blocked_elapsed_ns_total
                ),
                "producer_settle_blocked_elapsed_ns_max": (
                    self._producer_settle_blocked_elapsed_ns_max
                ),
                "producer_capture_elapsed_ns_total": (
                    self._producer_capture_elapsed_ns_total
                ),
                "producer_capture_elapsed_ns_max": (
                    self._producer_capture_elapsed_ns_max
                ),
                "queue_wait_elapsed_ns_total": self._queue_wait_elapsed_ns_total,
                "queue_wait_elapsed_ns_max": self._queue_wait_elapsed_ns_max,
                "worker_capture_wait_elapsed_ns_total": (
                    self._worker_capture_wait_elapsed_ns_total
                ),
                "worker_capture_wait_elapsed_ns_max": (
                    self._worker_capture_wait_elapsed_ns_max
                ),
                "worker_encode_elapsed_ns_total": (
                    self._worker_encode_elapsed_ns_total
                ),
                "worker_encode_elapsed_ns_max": (
                    self._worker_encode_elapsed_ns_max
                ),
                "worker_job_elapsed_ns_total": self._worker_job_elapsed_ns_total,
                "worker_job_elapsed_ns_max": self._worker_job_elapsed_ns_max,
                "state_shape": list(self._state_shape),
                "active_science_metrics_shape": list(self._metrics_shape),
                "state_bytes_per_slot": self._state_bytes_per_slot,
                "active_science_metrics_bytes_per_slot": (
                    self._metrics_bytes_per_slot
                ),
                "bytes_per_slot": self._bytes_per_slot,
                "resident_slot_bytes": self._slot_count * self._bytes_per_slot,
                "resident_pinned_bytes": (
                    self._slot_count * self._bytes_per_slot
                    if self._pinned_memory_verified
                    else 0
                ),
                "pinned_memory_verified": self._pinned_memory_verified,
                "in_use_buffer_bytes_high_water": (
                    self._in_use_high_water * self._bytes_per_slot
                ),
                "descriptor_maximum_container_items": (
                    self._descriptor_limits.maximum_container_items
                ),
                "descriptor_maximum_nodes": self._descriptor_limits.maximum_nodes,
                "descriptor_maximum_accounted_bytes": (
                    self._descriptor_limits.maximum_accounted_bytes
                ),
                "context_accounted_bytes_current": (
                    self._context_accounted_bytes_current
                ),
                "context_accounted_bytes_high_water": (
                    self._context_accounted_bytes_high_water
                ),
                "context_resident_accounted_bytes": (
                    self._context_accounted_bytes_current
                ),
                "context_resident_accounted_bytes_high_water": (
                    self._context_accounted_bytes_high_water
                ),
                "context_accounted_bytes_per_submission_max": (
                    self._context_accounted_bytes_per_submission_max
                ),
                "horizon_dependent_buffer_bytes": 0,
            }


__all__ = [
    "NATIVE_SNAPSHOT_SERVICE_SCHEMA_V1",
    "NativeSnapshotBackpressureError",
    "NativeSnapshotCancellation",
    "NativeSnapshotCaptureRequestV1",
    "NativeSnapshotCompletionV1",
    "NativeSnapshotEncodeRequestV1",
    "NativeSnapshotServiceClosedError",
    "NativeSnapshotServiceError",
    "NativeSnapshotServiceFailure",
    "NativeSnapshotServiceV1",
    "NativeSnapshotSlotStateV1",
    "NativeSnapshotSubmissionV1",
    "NativeSnapshotTimeoutError",
]
