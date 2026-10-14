"""Authority-preserving owner for the CUDA-golden Full Evolution V2 transition."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Final, Mapping, Sequence

from .contract import (
    CUDA_NATIVE_TIME_AUTHORITY_V1,
    INITIAL_POPULATION_V2,
    MAX_CAPABILITY_COUNTER_INCREMENT_PER_TICK_V2,
    MAX_SIGNED_INT64,
    MAX_TOTAL_TICKS_COUNTER_SAFE_V2,
    RESIDENT_CAPACITY_C_V2,
    CudaFullEvolutionNativeContractV2,
    validate_cuda_full_evolution_contract_v2,
)
from .layout import (
    CUDA_NATIVE_LIFECYCLE_EVENT_ROW_BYTES_V2,
    CUDA_NATIVE_LIFECYCLE_EVENT_ROWS_PER_WORLD_PER_TICK_V2,
    CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3,
    CUDA_NATIVE_SPARSE_EVENT_WIRE_ABI_V3,
    state_layout_v2,
)
from .native_loader import (
    EXECUTION_PLAN_CLASS_V2,
    LoadedCudaFullEvolutionNativeV2,
    load_prebuilt_cuda_full_evolution_v2,
)

FULL_CAPABILITY_MASK_V2: Final[int] = (1 << 7) - 1
RNG_SEMANTIC_EPOCH_V3: Final[int] = 3
EVENT_DISPATCH_SCHEMA_V2: Final[str] = "eiel_full_evolution_cuda_event_dispatch_v2"
EVENT_SINK_RECEIPT_SCHEMA_V2: Final[str] = (
    "eiel_full_evolution_cuda_event_sink_receipt_v2"
)
EVENT_FRONTIER_COMMIT_V2: Final[str] = "exact_content_external_sink_receipt_v2"
SPARSE_EVENT_DISPATCH_SCHEMA_V3: Final[str] = (
    "eiel_full_evolution_cuda_sparse_event_dispatch_v3"
)
SPARSE_EVENT_SINK_RECEIPT_SCHEMA_V3: Final[str] = (
    "eiel_full_evolution_cuda_sparse_event_sink_receipt_v3"
)
SPARSE_EVENT_FRONTIER_COMMIT_V3: Final[str] = (
    "exact_immutable_sparse_content_external_sink_receipt_v3"
)
CHECKPOINT_SCHEMA_V2: Final[str] = "eiel_full_evolution_cuda_checkpoint_v2"
CHECKPOINT_CANONICAL_STATE_V2: Final[str] = (
    "immutable_cpu_bytes_with_content_layout_and_metadata_seals"
)
PUBLIC_STATE_PROJECTION_V2: Final[str] = "validated_detached_clone_only"
EXECUTION_PLAN_TOPOLOGY_SCHEMA_V2: Final[str] = (
    "eiel_full_evolution_cuda_execution_plan_topology_v2"
)
EXECUTION_PLAN_RECEIPT_SCHEMA_V2: Final[str] = (
    "eiel_full_evolution_cuda_execution_plan_receipt_v2"
)
NATIVE_EXECUTION_RECEIPT_SCHEMA_V2: Final[str] = (
    "eiel_full_evolution_cuda_native_execution_receipt_v2"
)
EXECUTION_PLAN_STREAM_POLICY_V2: Final[str] = (
    "immutable_construction_stream_including_default_stream_0"
)
FULL_CAPACITY_BENCHMARK_FIXTURE_V2: Final[str] = (
    "eiel_full_evolution_cuda_full_capacity_benchmark_fixture_v2"
)
EXECUTION_MODES_V2: Final[tuple[str, str]] = (
    "eager_monolith",
    "cuda_graph_monolith",
)
COUNTER_NAMES_V2: Final[tuple[str, ...]] = (
    "patch",
    "wdl_delayed_hazard",
    "jungle",
    "eed",
    "genome_v2_ordinary_birth",
    "genome_v2_mutation",
    "recombination_profile",
)


class CudaFullEvolutionRuntimeError(RuntimeError):
    """Raised when an owner would violate the fixed native runtime contract."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _tensor_bytes(value: Any) -> bytes:
    """Copy one tensor to immutable host bytes at an explicit boundary."""
    try:
        detached = value.detach()
        # PyTorch may silently compact a strided CUDA view on the device even
        # when cpu() appears before contiguous().  Authority-bound GPU payloads
        # must therefore already be contiguous or fail before allocating.
        if detached.is_cuda and not detached.is_contiguous():
            raise CudaFullEvolutionRuntimeError(
                "non-contiguous CUDA tensor projection is prohibited"
            )
        return detached.cpu().contiguous().numpy().tobytes(order="C")
    except CudaFullEvolutionRuntimeError:
        raise
    except Exception as exc:
        raise CudaFullEvolutionRuntimeError(
            f"cannot materialize authority-bound tensor bytes: {exc}"
        ) from exc


def _event_content_metadata(
    *,
    event_rows: Any,
    event_counts: Any,
    world_uids: tuple[int, ...],
    start_next_absolute_tick: int,
    end_next_absolute_tick: int,
    tick_count: int,
    artifact_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": EVENT_DISPATCH_SCHEMA_V2,
        "time_authority": CUDA_NATIVE_TIME_AUTHORITY_V1,
        "world_uids": list(world_uids),
        "start_next_absolute_tick": start_next_absolute_tick,
        "end_next_absolute_tick": end_next_absolute_tick,
        "tick_count": tick_count,
        "event_rows": {
            "dtype": str(event_rows.dtype),
            "shape": list(event_rows.shape),
        },
        "event_counts": {
            "dtype": str(event_counts.dtype),
            "shape": list(event_counts.shape),
        },
        "artifact_sha256": artifact_sha256,
        "manifest_sha256": manifest_sha256,
    }


def _event_content_sha256(
    *,
    event_rows: Any,
    event_counts: Any,
    world_uids: tuple[int, ...],
    start_next_absolute_tick: int,
    end_next_absolute_tick: int,
    tick_count: int,
    artifact_sha256: str,
    manifest_sha256: str,
) -> str:
    metadata = _event_content_metadata(
        event_rows=event_rows,
        event_counts=event_counts,
        world_uids=world_uids,
        start_next_absolute_tick=start_next_absolute_tick,
        end_next_absolute_tick=end_next_absolute_tick,
        tick_count=tick_count,
        artifact_sha256=artifact_sha256,
        manifest_sha256=manifest_sha256,
    )
    digest = hashlib.sha256()
    digest.update(b"EIEL-CUDA-FULL-EVOLUTION-V2-EVENTS\0")
    digest.update(_canonical_json_bytes(metadata))
    digest.update(b"\0ROWS\0")
    digest.update(_tensor_bytes(event_rows))
    digest.update(b"\0COUNTS\0")
    digest.update(_tensor_bytes(event_counts))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CudaFullEvolutionEventDispatchV2:
    """Owned event batch whose exact content must be receipted by a sink."""

    event_rows: Any
    event_counts: Any
    world_uids: tuple[int, ...]
    start_next_absolute_tick: int
    end_next_absolute_tick: int
    tick_count: int
    drain_elapsed_seconds: float
    content_hash_elapsed_seconds: float
    content_sha256: str
    artifact_sha256: str
    manifest_sha256: str
    schema_version: str = EVENT_DISPATCH_SCHEMA_V2
    time_authority: str = CUDA_NATIVE_TIME_AUTHORITY_V1


def _rehash_event_dispatch(dispatch: CudaFullEvolutionEventDispatchV2) -> str:
    if dispatch.schema_version != EVENT_DISPATCH_SCHEMA_V2:
        raise CudaFullEvolutionRuntimeError("event dispatch schema mismatch")
    if dispatch.time_authority != CUDA_NATIVE_TIME_AUTHORITY_V1:
        raise CudaFullEvolutionRuntimeError("event dispatch time authority mismatch")
    return _event_content_sha256(
        event_rows=dispatch.event_rows,
        event_counts=dispatch.event_counts,
        world_uids=dispatch.world_uids,
        start_next_absolute_tick=dispatch.start_next_absolute_tick,
        end_next_absolute_tick=dispatch.end_next_absolute_tick,
        tick_count=dispatch.tick_count,
        artifact_sha256=dispatch.artifact_sha256,
        manifest_sha256=dispatch.manifest_sha256,
    )


@dataclass(frozen=True, slots=True)
class CudaFullEvolutionEventSinkReceiptV2:
    """External sink assertion sealed to one exact drained event payload."""

    dispatch_content_sha256: str
    sink_id: str
    sink_receipt_id: str
    world_uids: tuple[int, ...]
    start_next_absolute_tick: int
    end_next_absolute_tick: int
    tick_count: int
    artifact_sha256: str
    manifest_sha256: str
    receipt_sha256: str
    schema_version: str = EVENT_SINK_RECEIPT_SCHEMA_V2
    time_authority: str = CUDA_NATIVE_TIME_AUTHORITY_V1


def _event_sink_receipt_body(
    receipt: CudaFullEvolutionEventSinkReceiptV2,
) -> dict[str, Any]:
    return {
        "schema_version": receipt.schema_version,
        "time_authority": receipt.time_authority,
        "dispatch_content_sha256": receipt.dispatch_content_sha256,
        "sink_id": receipt.sink_id,
        "sink_receipt_id": receipt.sink_receipt_id,
        "world_uids": list(receipt.world_uids),
        "start_next_absolute_tick": receipt.start_next_absolute_tick,
        "end_next_absolute_tick": receipt.end_next_absolute_tick,
        "tick_count": receipt.tick_count,
        "artifact_sha256": receipt.artifact_sha256,
        "manifest_sha256": receipt.manifest_sha256,
    }


def _checked_sink_token(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > 1024
    ):
        raise CudaFullEvolutionRuntimeError(
            f"{label} must be a nonempty trimmed string of at most 1024 characters"
        )
    return value


def seal_event_sink_receipt_v2(
    dispatch: CudaFullEvolutionEventDispatchV2,
    *,
    sink_id: str,
    sink_receipt_id: str,
) -> CudaFullEvolutionEventSinkReceiptV2:
    """Seal an external sink's durable reference to the exact dispatch bytes.

    This function proves content binding, not sink durability. The caller is
    responsible for invoking it only after its external sink has accepted the
    event payload.
    """
    if type(dispatch) is not CudaFullEvolutionEventDispatchV2:
        raise CudaFullEvolutionRuntimeError("sink receipt requires a V2 event dispatch")
    checked_sink_id = _checked_sink_token(sink_id, "sink_id")
    checked_sink_receipt_id = _checked_sink_token(sink_receipt_id, "sink_receipt_id")
    if not _valid_sha256(dispatch.content_sha256):
        raise CudaFullEvolutionRuntimeError("event dispatch content hash is invalid")
    if _rehash_event_dispatch(dispatch) != dispatch.content_sha256:
        raise CudaFullEvolutionRuntimeError(
            "event dispatch content changed before sink receipt sealing"
        )
    provisional = CudaFullEvolutionEventSinkReceiptV2(
        dispatch_content_sha256=dispatch.content_sha256,
        sink_id=checked_sink_id,
        sink_receipt_id=checked_sink_receipt_id,
        world_uids=dispatch.world_uids,
        start_next_absolute_tick=dispatch.start_next_absolute_tick,
        end_next_absolute_tick=dispatch.end_next_absolute_tick,
        tick_count=dispatch.tick_count,
        artifact_sha256=dispatch.artifact_sha256,
        manifest_sha256=dispatch.manifest_sha256,
        receipt_sha256="",
    )
    return CudaFullEvolutionEventSinkReceiptV2(
        dispatch_content_sha256=provisional.dispatch_content_sha256,
        sink_id=provisional.sink_id,
        sink_receipt_id=provisional.sink_receipt_id,
        world_uids=provisional.world_uids,
        start_next_absolute_tick=provisional.start_next_absolute_tick,
        end_next_absolute_tick=provisional.end_next_absolute_tick,
        tick_count=provisional.tick_count,
        artifact_sha256=provisional.artifact_sha256,
        manifest_sha256=provisional.manifest_sha256,
        receipt_sha256=_canonical_sha256(_event_sink_receipt_body(provisional)),
    )


def _sparse_event_content_metadata(
    *,
    packed_events: bytes,
    world_offsets: tuple[int, ...],
    event_counts: tuple[int, ...],
    final_next_agent_ids: tuple[int, ...],
    world_uids: tuple[int, ...],
    start_next_absolute_tick: int,
    end_next_absolute_tick: int,
    tick_count: int,
    artifact_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SPARSE_EVENT_DISPATCH_SCHEMA_V3,
        "time_authority": CUDA_NATIVE_TIME_AUTHORITY_V1,
        "wire_abi": CUDA_NATIVE_SPARSE_EVENT_WIRE_ABI_V3,
        "wire_endianness": "little",
        "packed_event_row_bytes": CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3,
        "packed_event_count": len(packed_events)
        // CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3,
        "packed_event_nbytes": len(packed_events),
        "world_offsets": list(world_offsets),
        "event_counts": list(event_counts),
        "final_next_agent_ids": list(final_next_agent_ids),
        "world_uids": list(world_uids),
        "start_next_absolute_tick": start_next_absolute_tick,
        "end_next_absolute_tick": end_next_absolute_tick,
        "tick_count": tick_count,
        "artifact_sha256": artifact_sha256,
        "manifest_sha256": manifest_sha256,
    }


def _sparse_event_content_sha256(
    *,
    packed_events: bytes,
    world_offsets: tuple[int, ...],
    event_counts: tuple[int, ...],
    final_next_agent_ids: tuple[int, ...],
    world_uids: tuple[int, ...],
    start_next_absolute_tick: int,
    end_next_absolute_tick: int,
    tick_count: int,
    artifact_sha256: str,
    manifest_sha256: str,
) -> str:
    metadata = _sparse_event_content_metadata(
        packed_events=packed_events,
        world_offsets=world_offsets,
        event_counts=event_counts,
        final_next_agent_ids=final_next_agent_ids,
        world_uids=world_uids,
        start_next_absolute_tick=start_next_absolute_tick,
        end_next_absolute_tick=end_next_absolute_tick,
        tick_count=tick_count,
        artifact_sha256=artifact_sha256,
        manifest_sha256=manifest_sha256,
    )
    digest = hashlib.sha256()
    digest.update(b"EIEL-CUDA-FULL-EVOLUTION-V3-SPARSE-EVENTS\0")
    digest.update(_canonical_json_bytes(metadata))
    digest.update(b"\0PACKED-ROWS\0")
    digest.update(packed_events)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CudaFullEvolutionSparseEventDispatchV3:
    """Immutable exact sparse lifecycle payload awaiting a durable sink ACK."""

    packed_events: bytes
    world_offsets: tuple[int, ...]
    event_counts: tuple[int, ...]
    final_next_agent_ids: tuple[int, ...]
    world_uids: tuple[int, ...]
    start_next_absolute_tick: int
    end_next_absolute_tick: int
    tick_count: int
    drain_elapsed_seconds: float
    count_projection_elapsed_seconds: float
    compaction_and_transfer_elapsed_seconds: float
    content_hash_elapsed_seconds: float
    content_sha256: str
    artifact_sha256: str
    manifest_sha256: str
    schema_version: str = SPARSE_EVENT_DISPATCH_SCHEMA_V3
    time_authority: str = CUDA_NATIVE_TIME_AUTHORITY_V1
    wire_abi: str = CUDA_NATIVE_SPARSE_EVENT_WIRE_ABI_V3

    @property
    def event_count(self) -> int:
        return len(self.packed_events) // CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3

    @property
    def wire_nbytes(self) -> int:
        return len(self.packed_events) + 8 * len(self.world_offsets)


def _rehash_sparse_event_dispatch_v3(
    dispatch: CudaFullEvolutionSparseEventDispatchV3,
) -> str:
    if dispatch.schema_version != SPARSE_EVENT_DISPATCH_SCHEMA_V3:
        raise CudaFullEvolutionRuntimeError("sparse event dispatch schema mismatch")
    if dispatch.time_authority != CUDA_NATIVE_TIME_AUTHORITY_V1:
        raise CudaFullEvolutionRuntimeError(
            "sparse event dispatch time authority mismatch"
        )
    if dispatch.wire_abi != CUDA_NATIVE_SPARSE_EVENT_WIRE_ABI_V3:
        raise CudaFullEvolutionRuntimeError("sparse event wire ABI mismatch")
    return _sparse_event_content_sha256(
        packed_events=dispatch.packed_events,
        world_offsets=dispatch.world_offsets,
        event_counts=dispatch.event_counts,
        final_next_agent_ids=dispatch.final_next_agent_ids,
        world_uids=dispatch.world_uids,
        start_next_absolute_tick=dispatch.start_next_absolute_tick,
        end_next_absolute_tick=dispatch.end_next_absolute_tick,
        tick_count=dispatch.tick_count,
        artifact_sha256=dispatch.artifact_sha256,
        manifest_sha256=dispatch.manifest_sha256,
    )


@dataclass(frozen=True, slots=True)
class CudaFullEvolutionSparseEventSinkReceiptV3:
    """External durable-sink assertion over one immutable sparse payload."""

    dispatch_content_sha256: str
    sink_id: str
    sink_receipt_id: str
    world_uids: tuple[int, ...]
    start_next_absolute_tick: int
    end_next_absolute_tick: int
    tick_count: int
    artifact_sha256: str
    manifest_sha256: str
    receipt_sha256: str
    schema_version: str = SPARSE_EVENT_SINK_RECEIPT_SCHEMA_V3
    time_authority: str = CUDA_NATIVE_TIME_AUTHORITY_V1
    wire_abi: str = CUDA_NATIVE_SPARSE_EVENT_WIRE_ABI_V3


def _sparse_event_sink_receipt_body_v3(
    receipt: CudaFullEvolutionSparseEventSinkReceiptV3,
) -> dict[str, Any]:
    return {
        "schema_version": receipt.schema_version,
        "time_authority": receipt.time_authority,
        "wire_abi": receipt.wire_abi,
        "dispatch_content_sha256": receipt.dispatch_content_sha256,
        "sink_id": receipt.sink_id,
        "sink_receipt_id": receipt.sink_receipt_id,
        "world_uids": list(receipt.world_uids),
        "start_next_absolute_tick": receipt.start_next_absolute_tick,
        "end_next_absolute_tick": receipt.end_next_absolute_tick,
        "tick_count": receipt.tick_count,
        "artifact_sha256": receipt.artifact_sha256,
        "manifest_sha256": receipt.manifest_sha256,
    }


def seal_sparse_event_sink_receipt_v3(
    dispatch: CudaFullEvolutionSparseEventDispatchV3,
    *,
    sink_id: str,
    sink_receipt_id: str,
) -> CudaFullEvolutionSparseEventSinkReceiptV3:
    """Seal a durable sink's exact reference to immutable sparse event bytes."""
    if type(dispatch) is not CudaFullEvolutionSparseEventDispatchV3:
        raise CudaFullEvolutionRuntimeError(
            "sparse sink receipt requires a V3 sparse event dispatch"
        )
    checked_sink_id = _checked_sink_token(sink_id, "sink_id")
    checked_sink_receipt_id = _checked_sink_token(sink_receipt_id, "sink_receipt_id")
    if not _valid_sha256(dispatch.content_sha256):
        raise CudaFullEvolutionRuntimeError(
            "sparse event dispatch content hash is invalid"
        )
    if _rehash_sparse_event_dispatch_v3(dispatch) != dispatch.content_sha256:
        raise CudaFullEvolutionRuntimeError(
            "sparse event dispatch content changed before sink receipt sealing"
        )
    provisional = CudaFullEvolutionSparseEventSinkReceiptV3(
        dispatch_content_sha256=dispatch.content_sha256,
        sink_id=checked_sink_id,
        sink_receipt_id=checked_sink_receipt_id,
        world_uids=dispatch.world_uids,
        start_next_absolute_tick=dispatch.start_next_absolute_tick,
        end_next_absolute_tick=dispatch.end_next_absolute_tick,
        tick_count=dispatch.tick_count,
        artifact_sha256=dispatch.artifact_sha256,
        manifest_sha256=dispatch.manifest_sha256,
        receipt_sha256="",
    )
    return CudaFullEvolutionSparseEventSinkReceiptV3(
        dispatch_content_sha256=provisional.dispatch_content_sha256,
        sink_id=provisional.sink_id,
        sink_receipt_id=provisional.sink_receipt_id,
        world_uids=provisional.world_uids,
        start_next_absolute_tick=provisional.start_next_absolute_tick,
        end_next_absolute_tick=provisional.end_next_absolute_tick,
        tick_count=provisional.tick_count,
        artifact_sha256=provisional.artifact_sha256,
        manifest_sha256=provisional.manifest_sha256,
        receipt_sha256=_canonical_sha256(
            _sparse_event_sink_receipt_body_v3(provisional)
        ),
    )


def _validate_sparse_event_projection_v3(
    *,
    packed_rows: Any,
    world_offsets: tuple[int, ...],
    event_counts: tuple[int, ...],
    final_next_agent_ids: tuple[int, ...],
    tick_count: int,
) -> None:
    """Validate one CPU projection without reconstructing the dense oracle."""
    import torch

    batch_size = len(event_counts)
    if (
        len(world_offsets) != batch_size + 1
        or len(final_next_agent_ids) != batch_size
        or world_offsets[0] != 0
        or any(count < 0 for count in event_counts)
        or any(
            world_offsets[index + 1] - world_offsets[index] != event_counts[index]
            for index in range(batch_size)
        )
    ):
        raise CudaFullEvolutionRuntimeError(
            "sparse event world offsets/counts are inconsistent"
        )
    event_count = world_offsets[-1]
    if (
        event_count < 0
        or tuple(packed_rows.shape)
        != (event_count, CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3)
        or packed_rows.dtype != torch.uint8
        or packed_rows.is_cuda
        or not packed_rows.is_contiguous()
    ):
        raise CudaFullEvolutionRuntimeError(
            "sparse event CPU projection shape or storage is invalid"
        )
    if event_count == 0:
        return

    words = packed_rows.view(torch.int64).view(event_count, 4)
    agent_id = words[:, 0]
    parent_a_id = words[:, 1]
    parent_b_id = words[:, 2]
    packed_key = words[:, 3]
    tick_offset = packed_key & 0x3F
    event_kind = (packed_key >> 6) & 0x3
    subject_slot = (packed_key >> 8) & 0x1FF
    site_index = (packed_key >> 17) & 0x3FF
    final_ids = torch.repeat_interleave(
        torch.tensor(final_next_agent_ids, dtype=torch.int64),
        torch.tensor(event_counts, dtype=torch.int64),
    )
    committed_birth = event_kind == 2
    valid = (packed_key >= 0) & (packed_key < (1 << 27))
    valid &= tick_offset < tick_count
    valid &= (event_kind == 1) | committed_birth
    valid &= subject_slot < RESIDENT_CAPACITY_C_V2
    valid &= site_index < 1024
    valid &= (agent_id > 0) & (agent_id < final_ids)
    valid &= (parent_a_id >= 0) & (parent_a_id < final_ids)
    valid &= (parent_b_id >= 0) & (parent_b_id < final_ids)
    valid &= ~committed_birth | (
        (parent_a_id > 0) & (parent_b_id > 0) & (parent_a_id != parent_b_id)
    )
    canonical_ordinal = (
        tick_offset * (2 * RESIDENT_CAPACITY_C_V2)
        + (event_kind - 1) * RESIDENT_CAPACITY_C_V2
        + subject_slot
    )
    if event_count > 1:
        same_world = torch.ones(event_count - 1, dtype=torch.bool)
        for boundary in world_offsets[1:-1]:
            if 0 < boundary < event_count:
                same_world[boundary - 1] = False
        valid_order = (
            ~same_world | (canonical_ordinal[1:] > canonical_ordinal[:-1])
        ).all()
    else:
        valid_order = torch.tensor(True)
    if not bool(valid.all().item()) or not bool(valid_order.item()):
        raise CudaFullEvolutionRuntimeError(
            "sparse lifecycle event payload validation failed"
        )


@dataclass(frozen=True, slots=True)
class _PendingDispatchV2:
    start_next_absolute_tick: int
    tick_count: int

    @property
    def end_next_absolute_tick(self) -> int:
        return self.start_next_absolute_tick + self.tick_count


@dataclass(frozen=True, slots=True)
class CudaFullEvolutionCheckpointV2:
    """Sealed canonical checkpoint with immutable host bytes and identity."""

    contract: CudaFullEvolutionNativeContractV2
    canonical_state: bytes
    canonical_state_shape: tuple[int, int]
    canonical_state_nbytes: int
    canonical_state_sha256: str
    canonical_state_layout_sha256: str
    initial_next_absolute_tick: int
    advanced_ticks: int
    acknowledged_event_next_tick: int
    last_sink_receipt_sha256: str | None
    world_uids: tuple[int, ...]
    rng_keys: tuple[int, ...]
    rng_semantic_epoch: int
    artifact_path: Path
    artifact_sha256: str
    manifest_sha256: str
    device_identity: Mapping[str, Any]
    population_fixture_id: str | None
    metadata_seal_sha256: str
    schema_version: str = CHECKPOINT_SCHEMA_V2
    time_authority: str = CUDA_NATIVE_TIME_AUTHORITY_V1


@dataclass(frozen=True, slots=True)
class CudaFullEvolutionCheckpointCaptureSourceV1:
    """Producer-thread borrow of resident state for bounded ACS capture.

    This is deliberately not a checkpoint and must never cross the detached
    writer boundary. The owning runtime remains live and immutable while the
    producer copies each contiguous canonical row through its two pinned
    staging slabs. Only the resulting host-file descriptor may be queued.
    """

    contract: CudaFullEvolutionNativeContractV2
    resident_state: Any
    canonical_bytes_per_world: int
    initial_next_absolute_tick: int
    advanced_ticks: int
    acknowledged_event_next_tick: int
    last_sink_receipt_sha256: str | None
    world_uids: tuple[int, ...]
    rng_keys: tuple[int, ...]
    rng_semantic_epoch: int
    artifact_path: Path
    artifact_sha256: str
    manifest_sha256: str
    device_identity: Mapping[str, Any]
    population_fixture_id: str | None
    schema_version: str = CHECKPOINT_SCHEMA_V2
    time_authority: str = CUDA_NATIVE_TIME_AUTHORITY_V1


# Transitional type name for callers; its payload is now the sealed V2 schema.
CudaFullEvolutionStateCloneV2 = CudaFullEvolutionCheckpointV2


def _canonical_state_layout_sha256(
    *, shape: tuple[int, int], nbytes: int, content_sha256: str
) -> str:
    return _canonical_sha256(
        {
            "dtype": "uint8",
            "shape": list(shape),
            "nbytes": nbytes,
            "content_sha256": content_sha256,
        }
    )


def _checkpoint_metadata_body(
    checkpoint: CudaFullEvolutionCheckpointV2,
) -> dict[str, Any]:
    return {
        "schema_version": checkpoint.schema_version,
        "time_authority": checkpoint.time_authority,
        "contract": checkpoint.contract.to_dict(),
        "canonical_state_shape": list(checkpoint.canonical_state_shape),
        "canonical_state_nbytes": checkpoint.canonical_state_nbytes,
        "canonical_state_sha256": checkpoint.canonical_state_sha256,
        "canonical_state_layout_sha256": (checkpoint.canonical_state_layout_sha256),
        "initial_next_absolute_tick": checkpoint.initial_next_absolute_tick,
        "advanced_ticks": checkpoint.advanced_ticks,
        "acknowledged_event_next_tick": checkpoint.acknowledged_event_next_tick,
        "last_sink_receipt_sha256": checkpoint.last_sink_receipt_sha256,
        "world_uids": list(checkpoint.world_uids),
        "rng_keys": list(checkpoint.rng_keys),
        "rng_semantic_epoch": checkpoint.rng_semantic_epoch,
        "artifact_sha256": checkpoint.artifact_sha256,
        "manifest_sha256": checkpoint.manifest_sha256,
        "device_identity": dict(checkpoint.device_identity),
        "population_fixture_id": checkpoint.population_fixture_id,
    }


def checkpoint_runtime_projection_from_capture_source_v1(
    source: CudaFullEvolutionCheckpointCaptureSourceV1,
    *,
    canonical_state_sha256: str,
) -> dict[str, Any]:
    """Seal scalar checkpoint authority after direct resident-row capture."""

    if type(source) is not CudaFullEvolutionCheckpointCaptureSourceV1:
        raise CudaFullEvolutionRuntimeError(
            "checkpoint capture source must use the exact producer-borrow schema"
        )
    contract = validate_cuda_full_evolution_contract_v2(source.contract)
    layout = state_layout_v2()
    if source.canonical_bytes_per_world != layout.canonical_checkpoint_bytes_per_world:
        raise CudaFullEvolutionRuntimeError(
            "checkpoint capture source canonical row width differs"
        )
    try:
        import torch

        valid_state = (
            isinstance(source.resident_state, torch.Tensor)
            and source.resident_state.dtype == torch.uint8
            and source.resident_state.is_cuda
            and source.resident_state.is_contiguous()
            and tuple(int(value) for value in source.resident_state.shape)
            == (contract.B, layout.state_bytes_per_world)
        )
    except (ImportError, RuntimeError, TypeError, ValueError):
        valid_state = False
    if not valid_state:
        raise CudaFullEvolutionRuntimeError(
            "checkpoint capture source is not the exact contiguous resident CUDA state"
        )
    if not _valid_sha256(canonical_state_sha256):
        raise CudaFullEvolutionRuntimeError(
            "checkpoint capture source state digest is invalid"
        )
    shape = (contract.B, source.canonical_bytes_per_world)
    nbytes = shape[0] * shape[1]
    if (
        source.schema_version != CHECKPOINT_SCHEMA_V2
        or source.time_authority != CUDA_NATIVE_TIME_AUTHORITY_V1
        or source.initial_next_absolute_tick != contract.start_next_tick
        or type(source.advanced_ticks) is not int
        or not 0 <= source.advanced_ticks <= contract.total_ticks
        or source.acknowledged_event_next_tick
        != source.initial_next_absolute_tick + source.advanced_ticks
        or (
            source.advanced_ticks == 0
            and source.last_sink_receipt_sha256 is not None
        )
        or (
            source.advanced_ticks > 0
            and not _valid_sha256(source.last_sink_receipt_sha256)
        )
        or len(source.world_uids) != contract.B
        or len(set(source.world_uids)) != contract.B
        or any(
            type(uid) is not int or not 0 < uid <= MAX_SIGNED_INT64
            for uid in source.world_uids
        )
        or len(source.rng_keys) != contract.B
        or any(
            type(key) is not int or not 0 <= key <= MAX_SIGNED_INT64
            for key in source.rng_keys
        )
        or source.rng_semantic_epoch != RNG_SEMANTIC_EPOCH_V3
        or not _valid_sha256(source.artifact_sha256)
        or not _valid_sha256(source.manifest_sha256)
        or not isinstance(source.artifact_path, Path)
        or source.population_fixture_id
        not in (None, FULL_CAPACITY_BENCHMARK_FIXTURE_V2)
    ):
        raise CudaFullEvolutionRuntimeError(
            "checkpoint capture source scalar authority is invalid"
        )
    expected_device_keys = {
        "index",
        "name",
        "uuid",
        "pci_domain_id",
        "pci_bus_id",
        "pci_device_id",
        "total_memory_bytes",
        "multiprocessor_count",
        "sm",
        "cuda_runtime",
    }
    if not isinstance(source.device_identity, Mapping) or set(
        source.device_identity
    ) != expected_device_keys:
        raise CudaFullEvolutionRuntimeError(
            "checkpoint capture source device identity is invalid"
        )
    body = {
        "schema_version": source.schema_version,
        "time_authority": source.time_authority,
        "contract": contract.to_dict(),
        "canonical_state_shape": list(shape),
        "canonical_state_nbytes": nbytes,
        "canonical_state_sha256": canonical_state_sha256,
        "canonical_state_layout_sha256": _canonical_state_layout_sha256(
            shape=shape,
            nbytes=nbytes,
            content_sha256=canonical_state_sha256,
        ),
        "initial_next_absolute_tick": source.initial_next_absolute_tick,
        "advanced_ticks": source.advanced_ticks,
        "acknowledged_event_next_tick": source.acknowledged_event_next_tick,
        "last_sink_receipt_sha256": source.last_sink_receipt_sha256,
        "world_uids": list(source.world_uids),
        "rng_keys": list(source.rng_keys),
        "rng_semantic_epoch": source.rng_semantic_epoch,
        "artifact_sha256": source.artifact_sha256,
        "manifest_sha256": source.manifest_sha256,
        "device_identity": dict(source.device_identity),
        "population_fixture_id": source.population_fixture_id,
    }
    return {**body, "metadata_seal_sha256": _canonical_sha256(body)}


def _validate_checkpoint_envelope(
    checkpoint: CudaFullEvolutionCheckpointV2,
) -> CudaFullEvolutionNativeContractV2:
    if type(checkpoint) is not CudaFullEvolutionCheckpointV2:
        raise CudaFullEvolutionRuntimeError("checkpoint must use the sealed V2 schema")
    if checkpoint.schema_version != CHECKPOINT_SCHEMA_V2:
        raise CudaFullEvolutionRuntimeError("checkpoint schema mismatch")
    if checkpoint.time_authority != CUDA_NATIVE_TIME_AUTHORITY_V1:
        raise CudaFullEvolutionRuntimeError("checkpoint time authority mismatch")
    if not isinstance(checkpoint.contract, CudaFullEvolutionNativeContractV2):
        raise CudaFullEvolutionRuntimeError(
            "checkpoint contract must be the immutable V2 contract object"
        )
    contract = validate_cuda_full_evolution_contract_v2(checkpoint.contract)
    if contract.total_ticks > MAX_TOTAL_TICKS_COUNTER_SAFE_V2:
        raise CudaFullEvolutionRuntimeError(
            "checkpoint exceeds the signed-int64 counter-safety horizon"
        )
    if (
        type(checkpoint.initial_next_absolute_tick) is not int
        or checkpoint.initial_next_absolute_tick != contract.start_next_tick
    ):
        raise CudaFullEvolutionRuntimeError(
            "checkpoint initial tick is not bound to contract.start_next_tick"
        )
    if type(checkpoint.advanced_ticks) is not int or not (
        0 <= checkpoint.advanced_ticks <= contract.total_ticks
    ):
        raise CudaFullEvolutionRuntimeError("checkpoint advanced tick count is invalid")
    expected_frontier = contract.start_next_tick + checkpoint.advanced_ticks
    if (
        type(checkpoint.acknowledged_event_next_tick) is not int
        or checkpoint.acknowledged_event_next_tick != expected_frontier
    ):
        raise CudaFullEvolutionRuntimeError(
            "checkpoint acknowledged event frontier is inconsistent"
        )
    if checkpoint.advanced_ticks == 0:
        if checkpoint.last_sink_receipt_sha256 is not None:
            raise CudaFullEvolutionRuntimeError(
                "initial checkpoint must not claim a sink receipt"
            )
    elif not _valid_sha256(checkpoint.last_sink_receipt_sha256):
        raise CudaFullEvolutionRuntimeError(
            "advanced checkpoint lacks a valid last sink receipt hash"
        )
    if (
        type(checkpoint.world_uids) is not tuple
        or len(checkpoint.world_uids) != contract.B
        or len(set(checkpoint.world_uids)) != contract.B
        or any(
            type(uid) is not int or not 0 < uid <= MAX_SIGNED_INT64
            for uid in checkpoint.world_uids
        )
    ):
        raise CudaFullEvolutionRuntimeError("checkpoint world UID identity is invalid")
    if (
        type(checkpoint.rng_keys) is not tuple
        or len(checkpoint.rng_keys) != contract.B
        or any(
            type(key) is not int or not 0 <= key <= MAX_SIGNED_INT64
            for key in checkpoint.rng_keys
        )
        or type(checkpoint.rng_semantic_epoch) is not int
        or checkpoint.rng_semantic_epoch != RNG_SEMANTIC_EPOCH_V3
    ):
        raise CudaFullEvolutionRuntimeError("checkpoint RNG identity is invalid")
    layout = state_layout_v2()
    expected_shape = (contract.B, layout.canonical_checkpoint_bytes_per_world)
    expected_nbytes = expected_shape[0] * expected_shape[1]
    if (
        type(checkpoint.canonical_state) is not bytes
        or type(checkpoint.canonical_state_shape) is not tuple
        or checkpoint.canonical_state_shape != expected_shape
        or type(checkpoint.canonical_state_nbytes) is not int
        or checkpoint.canonical_state_nbytes != expected_nbytes
        or len(checkpoint.canonical_state) != expected_nbytes
    ):
        raise CudaFullEvolutionRuntimeError("checkpoint canonical byte layout mismatch")
    actual_state_sha256 = hashlib.sha256(checkpoint.canonical_state).hexdigest()
    if (
        not _valid_sha256(checkpoint.canonical_state_sha256)
        or checkpoint.canonical_state_sha256 != actual_state_sha256
    ):
        raise CudaFullEvolutionRuntimeError("checkpoint canonical state hash mismatch")
    expected_layout_sha256 = _canonical_state_layout_sha256(
        shape=expected_shape,
        nbytes=expected_nbytes,
        content_sha256=actual_state_sha256,
    )
    if checkpoint.canonical_state_layout_sha256 != expected_layout_sha256:
        raise CudaFullEvolutionRuntimeError("checkpoint state shape/size seal mismatch")
    if not _valid_sha256(checkpoint.artifact_sha256) or not _valid_sha256(
        checkpoint.manifest_sha256
    ):
        raise CudaFullEvolutionRuntimeError("checkpoint artifact identity is invalid")
    if not isinstance(checkpoint.artifact_path, Path):
        raise CudaFullEvolutionRuntimeError("checkpoint artifact locator is invalid")
    expected_device_keys = {
        "index",
        "name",
        "uuid",
        "pci_domain_id",
        "pci_bus_id",
        "pci_device_id",
        "total_memory_bytes",
        "multiprocessor_count",
        "sm",
        "cuda_runtime",
    }
    if (
        not isinstance(checkpoint.device_identity, Mapping)
        or set(checkpoint.device_identity) != expected_device_keys
        or type(checkpoint.device_identity.get("index")) is not int
        or any(
            type(checkpoint.device_identity.get(key)) is not int
            for key in (
                "pci_domain_id",
                "pci_bus_id",
                "pci_device_id",
                "total_memory_bytes",
                "multiprocessor_count",
            )
        )
        or any(
            type(checkpoint.device_identity.get(key)) is not str
            or not checkpoint.device_identity.get(key)
            for key in ("name", "uuid", "sm", "cuda_runtime")
        )
    ):
        raise CudaFullEvolutionRuntimeError(
            "checkpoint selected-device identity is invalid"
        )
    if checkpoint.population_fixture_id not in (
        None,
        FULL_CAPACITY_BENCHMARK_FIXTURE_V2,
    ):
        raise CudaFullEvolutionRuntimeError(
            "checkpoint population fixture identity is invalid"
        )
    if not _valid_sha256(
        checkpoint.metadata_seal_sha256
    ) or checkpoint.metadata_seal_sha256 != _canonical_sha256(
        _checkpoint_metadata_body(checkpoint)
    ):
        raise CudaFullEvolutionRuntimeError("checkpoint metadata seal mismatch")
    return contract


class CudaFullEvolutionRuntimeV2:
    """Own private resident state; every public projection is a verified copy."""

    def __init__(
        self,
        *,
        contract: CudaFullEvolutionNativeContractV2,
        loaded: LoadedCudaFullEvolutionNativeV2,
        state: Any,
        event_rows: Any,
        event_counts: Any,
        reference_scratch: Any | None,
        initial_next_absolute_tick: int,
        expected_world_uids: tuple[int, ...],
        expected_rng_keys: tuple[int, ...],
        expected_rng_semantic_epoch: int,
        advanced_ticks: int = 0,
        last_sink_receipt_sha256: str | None = None,
        execution_mode: str = "eager_monolith",
        population_fixture_id: str | None = None,
    ) -> None:
        self.contract = validate_cuda_full_evolution_contract_v2(contract)
        if self.contract.total_ticks > MAX_TOTAL_TICKS_COUNTER_SAFE_V2:
            raise CudaFullEvolutionRuntimeError(
                "total_ticks exceeds the signed-int64 counter-safety horizon"
            )
        self.__loaded = loaded
        self.__state = state
        self.__event_rows = event_rows
        self.__event_counts = event_counts
        self.__reference_scratch = reference_scratch
        self.__initial_next_absolute_tick = initial_next_absolute_tick
        self.__expected_world_uids = tuple(expected_world_uids)
        self.__expected_rng_keys = tuple(expected_rng_keys)
        self.__expected_rng_semantic_epoch = expected_rng_semantic_epoch
        self.__advanced_ticks = advanced_ticks
        self.__last_sink_receipt_sha256 = last_sink_receipt_sha256
        if population_fixture_id not in (
            None,
            FULL_CAPACITY_BENCHMARK_FIXTURE_V2,
        ):
            raise CudaFullEvolutionRuntimeError(
                "population fixture identity is invalid"
            )
        self.__population_fixture_id = population_fixture_id
        self.__pending_dispatch: _PendingDispatchV2 | None = None
        self.__drained_dispatch: (
            CudaFullEvolutionEventDispatchV2
            | CudaFullEvolutionSparseEventDispatchV3
            | None
        ) = None
        self.__poisoned_reason: str | None = None
        if execution_mode not in EXECUTION_MODES_V2:
            raise CudaFullEvolutionRuntimeError(
                "execution_mode must be eager_monolith or cuda_graph_monolith"
            )
        self.__execution_mode = execution_mode
        self.__execution_plans: dict[int, Any] = {}
        self.__execution_plan_topologies: dict[int, dict[str, Any]] = {}
        self.__last_execution_plan_k: int | None = None
        self.__last_execution_receipt: dict[str, Any] | None = None
        self.__closed = False
        self.__native_execution_plan_type = getattr(
            self.__loaded.module, EXECUTION_PLAN_CLASS_V2, None
        )
        if not isinstance(self.__native_execution_plan_type, type):
            raise CudaFullEvolutionRuntimeError(
                "loaded binary does not export its native execution-plan type"
            )
        self._validate_owner_storage()
        self._validate_committed_frontier()
        try:
            for dispatch_temporal_k in range(1, self.contract.temporal_k + 1):
                plan = self.__loaded.module.create_execution_plan(
                    self.__execution_mode,
                    self.__state,
                    self.__event_rows,
                    self.__event_counts,
                    dispatch_temporal_k,
                    self.contract.temporal_k,
                    FULL_CAPABILITY_MASK_V2,
                    self.contract.profile_id,
                    self.__loaded.artifact_sha256,
                    self.__loaded.manifest_sha256,
                )
                self.__execution_plans[dispatch_temporal_k] = plan
                self.__execution_plan_topologies[dispatch_temporal_k] = (
                    self._bind_execution_plan_topology(
                        plan, dispatch_temporal_k=dispatch_temporal_k
                    )
                )
        except Exception as exc:
            for plan in self.__execution_plans.values():
                try:
                    plan.close()
                except Exception:
                    pass
            self.__execution_plans.clear()
            self.__execution_plan_topologies.clear()
            raise CudaFullEvolutionRuntimeError(
                f"cannot construct immutable native execution plan: {exc}"
            ) from exc

    @classmethod
    def initialize(
        cls,
        contract: CudaFullEvolutionNativeContractV2 | Mapping[str, Any],
        *,
        seed: int,
        next_absolute_tick: int | None = None,
        artifact_path: str | Path | None = None,
        device: str | Any = "cuda",
        enable_reference: bool = False,
        world_uids: Sequence[int] | None = None,
        first_world_uid: int = 1,
        execution_mode: str = "eager_monolith",
    ) -> "CudaFullEvolutionRuntimeV2":
        contract = validate_cuda_full_evolution_contract_v2(contract)
        if execution_mode not in EXECUTION_MODES_V2:
            raise CudaFullEvolutionRuntimeError(
                "execution_mode must be eager_monolith or cuda_graph_monolith"
            )
        if contract.total_ticks > MAX_TOTAL_TICKS_COUNTER_SAFE_V2:
            raise CudaFullEvolutionRuntimeError(
                "total_ticks exceeds the signed-int64 counter-safety horizon"
            )
        if type(seed) is not int or seed < 0 or seed > MAX_SIGNED_INT64:
            raise CudaFullEvolutionRuntimeError("seed must be an integer in [0, 2**63)")
        selected_tick = (
            contract.start_next_tick
            if next_absolute_tick is None
            else next_absolute_tick
        )
        if type(selected_tick) is not int or not 0 <= selected_tick <= MAX_SIGNED_INT64:
            raise CudaFullEvolutionRuntimeError(
                "next_absolute_tick must be an integer in [0, 2**63)"
            )
        if selected_tick != contract.start_next_tick:
            raise CudaFullEvolutionRuntimeError(
                "next_absolute_tick must equal contract.start_next_tick"
            )
        if type(enable_reference) is not bool:
            raise CudaFullEvolutionRuntimeError("enable_reference must be boolean")
        if world_uids is None:
            if type(first_world_uid) is not int or first_world_uid <= 0:
                raise CudaFullEvolutionRuntimeError("first_world_uid must be positive")
            uids = tuple(first_world_uid + index for index in range(contract.B))
        else:
            uids = tuple(world_uids)
        if (
            len(uids) != contract.B
            or len(set(uids)) != len(uids)
            or any(
                type(uid) is not int or not 0 < uid <= MAX_SIGNED_INT64 for uid in uids
            )
        ):
            raise CudaFullEvolutionRuntimeError(
                "world_uids must contain B unique positive signed-int64 identities"
            )
        import torch

        selected_device = torch.device(device)
        if selected_device.type != "cuda":
            raise CudaFullEvolutionRuntimeError("CUDA Native V2 requires a CUDA device")
        loaded = load_prebuilt_cuda_full_evolution_v2(
            artifact_path, device=selected_device
        )
        layout = state_layout_v2()
        state = torch.empty(
            (contract.B, layout.state_bytes_per_world),
            dtype=torch.uint8,
            device=selected_device,
        )
        event_rows = torch.zeros(
            (
                contract.B,
                CUDA_NATIVE_LIFECYCLE_EVENT_ROWS_PER_WORLD_PER_TICK_V2
                * contract.temporal_k,
                CUDA_NATIVE_LIFECYCLE_EVENT_ROW_BYTES_V2,
            ),
            dtype=torch.uint8,
            device=selected_device,
        )
        event_counts = torch.zeros(
            (contract.B,), dtype=torch.int32, device=selected_device
        )
        scratch = None
        if enable_reference:
            scratch = torch.zeros(
                (contract.B, layout.reference_scratch_bytes_per_world),
                dtype=torch.uint8,
                device=selected_device,
            )
        uid_tensor = torch.tensor(uids, dtype=torch.int64, device=selected_device)
        loaded.module.initialize(state, uid_tensor, seed, selected_tick)
        return cls(
            contract=contract,
            loaded=loaded,
            state=state,
            event_rows=event_rows,
            event_counts=event_counts,
            reference_scratch=scratch,
            initial_next_absolute_tick=selected_tick,
            expected_world_uids=uids,
            expected_rng_keys=(seed,) * contract.B,
            expected_rng_semantic_epoch=RNG_SEMANTIC_EPOCH_V3,
            execution_mode=execution_mode,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: CudaFullEvolutionCheckpointV2,
        *,
        artifact_path: str | Path | None = None,
        device: str | Any = "cuda",
        enable_reference: bool = False,
        execution_mode: str = "eager_monolith",
    ) -> "CudaFullEvolutionRuntimeV2":
        contract = _validate_checkpoint_envelope(checkpoint)
        if execution_mode not in EXECUTION_MODES_V2:
            raise CudaFullEvolutionRuntimeError(
                "execution_mode must be eager_monolith or cuda_graph_monolith"
            )
        if type(enable_reference) is not bool:
            raise CudaFullEvolutionRuntimeError("enable_reference must be boolean")
        import torch

        selected_device = torch.device(device)
        if selected_device.type != "cuda":
            raise CudaFullEvolutionRuntimeError("CUDA Native V2 requires a CUDA device")
        loaded = load_prebuilt_cuda_full_evolution_v2(
            artifact_path or checkpoint.artifact_path,
            device=selected_device,
        )
        if loaded.artifact_sha256 != checkpoint.artifact_sha256:
            raise CudaFullEvolutionRuntimeError("checkpoint artifact hash mismatch")
        if loaded.manifest_sha256 != checkpoint.manifest_sha256:
            raise CudaFullEvolutionRuntimeError("checkpoint manifest hash mismatch")
        if dict(loaded.device_identity) != dict(checkpoint.device_identity):
            raise CudaFullEvolutionRuntimeError(
                "checkpoint selected-device identity mismatch"
            )
        layout = state_layout_v2()
        state = torch.empty(
            (contract.B, layout.state_bytes_per_world),
            dtype=torch.uint8,
            device=selected_device,
        )
        canonical_cpu = torch.frombuffer(
            bytearray(checkpoint.canonical_state), dtype=torch.uint8
        ).view(checkpoint.canonical_state_shape)
        state[:, : layout.canonical_checkpoint_bytes_per_world].copy_(canonical_cpu)
        loaded.module.rebuild_derived_caches(state)
        event_rows = torch.zeros(
            (
                contract.B,
                CUDA_NATIVE_LIFECYCLE_EVENT_ROWS_PER_WORLD_PER_TICK_V2
                * contract.temporal_k,
                CUDA_NATIVE_LIFECYCLE_EVENT_ROW_BYTES_V2,
            ),
            dtype=torch.uint8,
            device=selected_device,
        )
        event_counts = torch.zeros(
            (contract.B,), dtype=torch.int32, device=selected_device
        )
        scratch = None
        if enable_reference:
            scratch = torch.zeros(
                (contract.B, layout.reference_scratch_bytes_per_world),
                dtype=torch.uint8,
                device=selected_device,
            )
        return cls(
            contract=contract,
            loaded=loaded,
            state=state,
            event_rows=event_rows,
            event_counts=event_counts,
            reference_scratch=scratch,
            initial_next_absolute_tick=checkpoint.initial_next_absolute_tick,
            expected_world_uids=checkpoint.world_uids,
            expected_rng_keys=checkpoint.rng_keys,
            expected_rng_semantic_epoch=checkpoint.rng_semantic_epoch,
            advanced_ticks=checkpoint.advanced_ticks,
            last_sink_receipt_sha256=checkpoint.last_sink_receipt_sha256,
            execution_mode=execution_mode,
            population_fixture_id=checkpoint.population_fixture_id,
        )

    def _bind_execution_plan_topology(
        self, plan: Any, *, dispatch_temporal_k: int
    ) -> dict[str, Any]:
        if type(plan) is not self.__native_execution_plan_type:
            raise CudaFullEvolutionRuntimeError(
                "execution plan is not the exact loaded native execution-plan type"
            )
        return self._bind_execution_plan_topology_payload(
            plan.topology_manifest(), dispatch_temporal_k=dispatch_temporal_k
        )

    def _bind_execution_plan_topology_payload(
        self, raw: Any, *, dispatch_temporal_k: int
    ) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise CudaFullEvolutionRuntimeError(
                "native execution-plan topology is not a mapping"
            )
        topology = json.loads(json.dumps(dict(raw), sort_keys=True))
        actual_device_index = int(self.__state.get_device())
        loaded_device_index = self.__loaded.device_identity.get("index")
        if (
            type(loaded_device_index) is not int
            or loaded_device_index != actual_device_index
        ):
            raise CudaFullEvolutionRuntimeError(
                "loaded artifact device identity does not match the owner tensor"
            )
        expected_keys = {
            "schema_version",
            "execution_mode",
            "profile_id",
            "artifact_sha256",
            "manifest_sha256",
            "native_plan_ordinal",
            "device_index",
            "B",
            "dispatch_temporal_k",
            "event_capacity_k",
            "capability_mask",
            "construction_kind",
            "construction_stream_handle",
            "graph_handle",
            "graph_exec_handle",
            "graph_node_count",
            "execution_kernel_count",
            "node_types",
            "kernel_nodes",
            "dependency_edges",
            "pointer_bindings",
            "strong_tensor_references",
            "pointer_substitution_allowed",
            "runtime_inspected",
            "declaration_is_execution_evidence",
            "launch_stream_policy",
        }
        expected_scalars = {
            "schema_version": EXECUTION_PLAN_TOPOLOGY_SCHEMA_V2,
            "execution_mode": self.__execution_mode,
            "profile_id": self.contract.profile_id,
            "artifact_sha256": self.__loaded.artifact_sha256,
            "manifest_sha256": self.__loaded.manifest_sha256,
            "device_index": actual_device_index,
            "B": self.contract.B,
            "dispatch_temporal_k": dispatch_temporal_k,
            "event_capacity_k": self.contract.temporal_k,
            "capability_mask": FULL_CAPABILITY_MASK_V2,
            "execution_kernel_count": 1,
            "strong_tensor_references": True,
            "pointer_substitution_allowed": False,
            "declaration_is_execution_evidence": False,
            "launch_stream_policy": EXECUTION_PLAN_STREAM_POLICY_V2,
        }
        if set(topology) != expected_keys or any(
            type(topology.get(key)) is not type(expected)
            or topology.get(key) != expected
            for key, expected in expected_scalars.items()
        ):
            raise CudaFullEvolutionRuntimeError(
                "native execution-plan topology binding mismatch"
            )
        native_plan_ordinal = topology["native_plan_ordinal"]
        handle_fields = (
            "construction_stream_handle",
            "graph_handle",
            "graph_exec_handle",
        )
        if (
            type(native_plan_ordinal) is not int
            or native_plan_ordinal < 1
            or any(
                type(topology[field]) is not int or topology[field] < 0
                for field in handle_fields
            )
        ):
            raise CudaFullEvolutionRuntimeError(
                "native execution-plan topology identity is invalid"
            )
        pointer_bindings = topology["pointer_bindings"]
        expected_pointer_bindings = {
            "state": int(self.__state.data_ptr()),
            "events": int(self.__event_rows.data_ptr()),
            "event_counts": int(self.__event_counts.data_ptr()),
        }
        if (
            not isinstance(pointer_bindings, dict)
            or set(pointer_bindings) != set(expected_pointer_bindings)
            or any(type(value) is not int for value in pointer_bindings.values())
            or pointer_bindings != expected_pointer_bindings
        ):
            raise CudaFullEvolutionRuntimeError(
                "native execution-plan pointer binding mismatch"
            )
        expected_kernel_nodes = [
            {
                "name": "advance_optimized_kernel",
                "node_type": "kernel",
                "runtime_graph_node": self.__execution_mode == "cuda_graph_monolith",
                "function_pointer_verified": True,
                "grid": [self.contract.B, 1, 1],
                "block": [256, 1, 1],
                "shared_memory_bytes": 0,
                "parameters": {
                    "state_pointer": expected_pointer_bindings["state"],
                    "state_stride_bytes": int(self.__state.stride(0)),
                    "event_pointer": expected_pointer_bindings["events"],
                    "event_stride_rows": int(self.__event_rows.stride(0))
                    // CUDA_NATIVE_LIFECYCLE_EVENT_ROW_BYTES_V2,
                    "event_count_pointer": expected_pointer_bindings["event_counts"],
                    "dispatch_temporal_k": dispatch_temporal_k,
                    "capability_mask": FULL_CAPABILITY_MASK_V2,
                },
            }
        ]
        graph_mode = self.__execution_mode == "cuda_graph_monolith"
        expected_graph_fields = {
            "construction_kind": (
                "explicit_cuda_graph_api_no_stream_capture"
                if graph_mode
                else "eager_kernel_launch"
            ),
            "graph_node_count": 1 if graph_mode else 0,
            "node_types": ["kernel"] if graph_mode else [],
            "kernel_nodes": expected_kernel_nodes,
            "dependency_edges": [],
            "runtime_inspected": graph_mode,
        }
        if any(
            type(topology.get(key)) is not type(expected)
            or topology.get(key) != expected
            for key, expected in expected_graph_fields.items()
        ):
            raise CudaFullEvolutionRuntimeError(
                "native execution-plan runtime topology mismatch"
            )
        if graph_mode:
            if topology["graph_handle"] == 0 or topology["graph_exec_handle"] == 0:
                raise CudaFullEvolutionRuntimeError(
                    "native CUDA Graph execution-plan handles are missing"
                )
        elif topology["graph_handle"] != 0 or topology["graph_exec_handle"] != 0:
            raise CudaFullEvolutionRuntimeError(
                "eager execution plan fabricated CUDA Graph handles"
            )
        topology["pointer_binding_identity_sha256"] = _canonical_sha256(
            {
                "pointers": pointer_bindings,
                "device": dict(self.__loaded.device_identity),
                "B": self.contract.B,
                "dispatch_temporal_k": dispatch_temporal_k,
                "event_capacity_k": self.contract.temporal_k,
                "capability_mask": FULL_CAPABILITY_MASK_V2,
                "profile_id": self.contract.profile_id,
                "artifact_sha256": self.__loaded.artifact_sha256,
                "manifest_sha256": self.__loaded.manifest_sha256,
            }
        )
        topology["device_identity"] = dict(self.__loaded.device_identity)
        topology["capture_identity_sha256"] = _canonical_sha256(
            {
                "construction_kind": topology.get("construction_kind"),
                "construction_stream_handle": topology.get(
                    "construction_stream_handle"
                ),
                "graph_handle": topology.get("graph_handle"),
                "graph_exec_handle": topology.get("graph_exec_handle"),
                "graph_node_count": topology.get("graph_node_count"),
                "execution_kernel_count": topology.get("execution_kernel_count"),
                "node_types": topology.get("node_types"),
                "kernel_nodes": topology.get("kernel_nodes"),
                "dependency_edges": topology.get("dependency_edges"),
                "strong_tensor_references": topology.get("strong_tensor_references"),
                "pointer_substitution_allowed": topology.get(
                    "pointer_substitution_allowed"
                ),
                "runtime_inspected": topology.get("runtime_inspected"),
                "launch_stream_policy": topology.get("launch_stream_policy"),
                "device_identity": topology["device_identity"],
                "B": self.contract.B,
                "dispatch_temporal_k": dispatch_temporal_k,
                "event_capacity_k": self.contract.temporal_k,
                "capability_mask": FULL_CAPABILITY_MASK_V2,
                "profile_id": self.contract.profile_id,
                "artifact_sha256": self.__loaded.artifact_sha256,
                "manifest_sha256": self.__loaded.manifest_sha256,
            }
        )
        topology["plan_identity_sha256"] = _canonical_sha256(
            {
                "native_plan_ordinal": topology.get("native_plan_ordinal"),
                "execution_mode": self.__execution_mode,
                "pointer_binding_identity_sha256": topology[
                    "pointer_binding_identity_sha256"
                ],
                "capture_identity_sha256": topology["capture_identity_sha256"],
                "profile_id": self.contract.profile_id,
                "artifact_sha256": self.__loaded.artifact_sha256,
                "manifest_sha256": self.__loaded.manifest_sha256,
            }
        )
        return topology

    def _bind_native_execution_receipt(
        self,
        raw: Any,
        *,
        topology: Mapping[str, Any],
        start_next_absolute_tick: int,
        tick_count: int,
    ) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            self._poison("native execution-plan launch receipt is missing")
        receipt = json.loads(json.dumps(dict(raw), sort_keys=True))
        expected_keys = {
            "schema_version",
            "execution_mode",
            "profile_id",
            "artifact_sha256",
            "manifest_sha256",
            "native_plan_ordinal",
            "launch_ordinal",
            "device_index",
            "launch_stream_handle",
            "construction_stream_handle",
            "launch_stream_policy",
            "start_next_absolute_tick",
            "end_next_absolute_tick",
            "tick_count",
            "B",
            "dispatch_temporal_k",
            "event_capacity_k",
            "capability_mask",
            "enqueued_world_ticks",
            "committed_world_ticks",
            "kernel",
            "graph_launched",
            "topology_graph_node_count",
            "topology_execution_kernel_count",
            "topology_node_types",
            "topology_kernel_nodes",
            "topology_dependency_edges",
            "pointer_bindings",
            "strong_tensor_references",
            "pointer_substitution_allowed",
            "runtime_inspected",
            "native_status",
            "event_status",
            "per_node_timing",
            "inter_node_gap_timing",
            "timing_fabrication_allowed",
        }
        expected_values = {
            "schema_version": NATIVE_EXECUTION_RECEIPT_SCHEMA_V2,
            "execution_mode": self.__execution_mode,
            "profile_id": self.contract.profile_id,
            "artifact_sha256": self.__loaded.artifact_sha256,
            "manifest_sha256": self.__loaded.manifest_sha256,
            "native_plan_ordinal": topology["native_plan_ordinal"],
            "device_index": int(self.__loaded.device_identity["index"]),
            "launch_stream_handle": topology["construction_stream_handle"],
            "construction_stream_handle": topology["construction_stream_handle"],
            "launch_stream_policy": EXECUTION_PLAN_STREAM_POLICY_V2,
            "start_next_absolute_tick": start_next_absolute_tick,
            "end_next_absolute_tick": start_next_absolute_tick + tick_count,
            "tick_count": tick_count,
            "B": self.contract.B,
            "dispatch_temporal_k": tick_count,
            "event_capacity_k": self.contract.temporal_k,
            "capability_mask": FULL_CAPABILITY_MASK_V2,
            "enqueued_world_ticks": self.contract.B * tick_count,
            "committed_world_ticks": 0,
            "kernel": "advance_optimized_kernel",
            "graph_launched": self.__execution_mode == "cuda_graph_monolith",
            "topology_graph_node_count": topology["graph_node_count"],
            "topology_execution_kernel_count": topology["execution_kernel_count"],
            "topology_node_types": topology["node_types"],
            "topology_kernel_nodes": topology["kernel_nodes"],
            "topology_dependency_edges": topology["dependency_edges"],
            "pointer_bindings": topology["pointer_bindings"],
            "strong_tensor_references": True,
            "pointer_substitution_allowed": False,
            "runtime_inspected": self.__execution_mode == "cuda_graph_monolith",
            "native_status": "enqueued_unvalidated",
            "event_status": "not_validated_by_native_launch",
            "per_node_timing": None,
            "inter_node_gap_timing": None,
            "timing_fabrication_allowed": False,
        }
        if set(receipt) != expected_keys or any(
            type(receipt.get(key)) is not type(expected) or receipt.get(key) != expected
            for key, expected in expected_values.items()
        ):
            self._poison("native execution-plan launch receipt binding mismatch")
        launch_ordinal = receipt["launch_ordinal"]
        if type(launch_ordinal) is not int or launch_ordinal < 1:
            self._poison("native execution-plan launch ordinal is invalid")
        return receipt

    def _validate_owner_storage(self) -> None:
        import torch

        layout = state_layout_v2()
        if not isinstance(self.__loaded, LoadedCudaFullEvolutionNativeV2):
            raise CudaFullEvolutionRuntimeError("native artifact owner is invalid")
        runtime_receipt = self.__loaded.manifest.get("runtime")
        if (
            not isinstance(runtime_receipt, Mapping)
            or runtime_receipt.get("dense_oracle_event_frontier_commit")
            != SPARSE_EVENT_FRONTIER_COMMIT_V3
            or runtime_receipt.get("checkpoint_schema") != CHECKPOINT_SCHEMA_V2
            or runtime_receipt.get("checkpoint_canonical_state")
            != CHECKPOINT_CANONICAL_STATE_V2
            or runtime_receipt.get("public_state_projection")
            != PUBLIC_STATE_PROJECTION_V2
        ):
            raise CudaFullEvolutionRuntimeError(
                "native manifest does not bind the runtime authority boundary"
            )
        if (
            self.__state.dtype != torch.uint8
            or not self.__state.is_cuda
            or not self.__state.is_contiguous()
            or tuple(self.__state.shape)
            != (self.contract.B, layout.state_bytes_per_world)
        ):
            raise CudaFullEvolutionRuntimeError("resident state owner layout mismatch")
        if self.__reference_scratch is not None and (
            self.__reference_scratch.dtype != torch.uint8
            or not self.__reference_scratch.is_cuda
            or not self.__reference_scratch.is_contiguous()
            or tuple(self.__reference_scratch.shape)
            != (self.contract.B, layout.reference_scratch_bytes_per_world)
            or self.__reference_scratch.device != self.__state.device
        ):
            raise CudaFullEvolutionRuntimeError(
                "reference scratch owner layout mismatch"
            )
        expected_event_shape = (
            self.contract.B,
            CUDA_NATIVE_LIFECYCLE_EVENT_ROWS_PER_WORLD_PER_TICK_V2
            * self.contract.temporal_k,
            CUDA_NATIVE_LIFECYCLE_EVENT_ROW_BYTES_V2,
        )
        if (
            self.__event_rows.dtype != torch.uint8
            or not self.__event_rows.is_cuda
            or not self.__event_rows.is_contiguous()
            or tuple(self.__event_rows.shape) != expected_event_shape
            or self.__event_rows.device != self.__state.device
            or self.__event_counts.dtype != torch.int32
            or not self.__event_counts.is_cuda
            or not self.__event_counts.is_contiguous()
            or tuple(self.__event_counts.shape) != (self.contract.B,)
            or self.__event_counts.device != self.__state.device
        ):
            raise CudaFullEvolutionRuntimeError("bounded event staging owner mismatch")
        if (
            self.__initial_next_absolute_tick != self.contract.start_next_tick
            or type(self.__advanced_ticks) is not int
            or not 0 <= self.__advanced_ticks <= self.contract.total_ticks
        ):
            raise CudaFullEvolutionRuntimeError(
                "owner absolute-tick frontier is invalid"
            )
        if (
            type(self.__expected_world_uids) is not tuple
            or len(self.__expected_world_uids) != self.contract.B
            or len(set(self.__expected_world_uids)) != self.contract.B
            or any(
                type(uid) is not int or not 0 < uid <= MAX_SIGNED_INT64
                for uid in self.__expected_world_uids
            )
        ):
            raise CudaFullEvolutionRuntimeError("owner world UID identity is invalid")
        if (
            type(self.__expected_rng_keys) is not tuple
            or len(self.__expected_rng_keys) != self.contract.B
            or any(
                type(key) is not int or not 0 <= key <= MAX_SIGNED_INT64
                for key in self.__expected_rng_keys
            )
            or type(self.__expected_rng_semantic_epoch) is not int
            or self.__expected_rng_semantic_epoch != RNG_SEMANTIC_EPOCH_V3
        ):
            raise CudaFullEvolutionRuntimeError("owner RNG identity is invalid")
        if self.__advanced_ticks == 0:
            if self.__last_sink_receipt_sha256 is not None:
                raise CudaFullEvolutionRuntimeError(
                    "initial owner must not claim a sink receipt"
                )
        elif not _valid_sha256(self.__last_sink_receipt_sha256):
            raise CudaFullEvolutionRuntimeError(
                "advanced owner lacks a valid sink receipt hash"
            )

    @property
    def state(self) -> Any:
        self._validate_committed_frontier()
        return self.__state.detach().clone()

    @property
    def reference_scratch(self) -> Any | None:
        self._validate_committed_frontier()
        if self.__reference_scratch is None:
            return None
        return self.__reference_scratch.detach().clone()

    @property
    def event_rows(self) -> Any:
        self._validate_committed_frontier()
        return self.__event_rows.detach().clone()

    @property
    def event_counts(self) -> Any:
        self._validate_committed_frontier()
        return self.__event_counts.detach().clone()

    @property
    def artifact_path(self) -> Path:
        return self.__loaded.artifact_path

    @property
    def artifact_sha256(self) -> str:
        return self.__loaded.artifact_sha256

    @property
    def manifest_sha256(self) -> str:
        return self.__loaded.manifest_sha256

    @property
    def remaining_ticks(self) -> int:
        pending_ticks = (
            0 if self.__pending_dispatch is None else self.__pending_dispatch.tick_count
        )
        return self.contract.total_ticks - self.__advanced_ticks - pending_ticks

    @property
    def advanced_ticks(self) -> int:
        """Return sink-receipted ticks, never merely dispatched ticks."""
        return self.__advanced_ticks

    @property
    def has_pending_dispatch(self) -> bool:
        return self.__pending_dispatch is not None

    @property
    def active_counts(self) -> tuple[int, ...]:
        """Return immutable active-count words after full committed-frontier validation."""
        rows = self._validate_committed_frontier()
        return tuple(int(row[3]) for row in rows)

    @property
    def committed_next_absolute_tick(self) -> int:
        return self.__initial_next_absolute_tick + self.__advanced_ticks

    @property
    def last_sink_receipt_sha256(self) -> str | None:
        return self.__last_sink_receipt_sha256

    @property
    def execution_mode(self) -> str:
        return self.__execution_mode

    @property
    def population_fixture_id(self) -> str | None:
        return self.__population_fixture_id

    @property
    def closed(self) -> bool:
        return self.__closed

    def execution_plan_receipt_v2(self) -> dict[str, Any]:
        """Return copied runtime evidence; declarations alone never prove a launch."""
        selected_k = self.__last_execution_plan_k or self.contract.temporal_k
        topology = self.__execution_plan_topologies.get(selected_k)
        if topology is None:
            raise CudaFullEvolutionRuntimeError("native execution plan is unavailable")
        return json.loads(
            json.dumps(
                {
                    "schema_version": EXECUTION_PLAN_RECEIPT_SCHEMA_V2,
                    "topology": topology,
                    "prepared_dispatch_temporal_k": list(
                        sorted(self.__execution_plan_topologies)
                    ),
                    "last_execution": self.__last_execution_receipt,
                    "closed": self.__closed,
                },
                sort_keys=True,
            )
        )

    def _ensure_usable(self, operation: str) -> None:
        if self.__closed:
            raise CudaFullEvolutionRuntimeError(
                f"owner is closed; {operation} is forbidden"
            )
        if self.__poisoned_reason is not None:
            raise CudaFullEvolutionRuntimeError(
                f"owner is poisoned; {operation} is forbidden: "
                f"{self.__poisoned_reason}"
            )

    def _ensure_no_pending(self, operation: str) -> None:
        self._ensure_usable(operation)
        if self.__pending_dispatch is not None:
            raise CudaFullEvolutionRuntimeError(
                f"{operation} requires drain and exact-content sink receipt "
                "acknowledgement of pending events"
            )

    def _poison(self, reason: str) -> None:
        self.__poisoned_reason = reason
        raise CudaFullEvolutionRuntimeError(
            "runtime authority validation failed; owner poisoned and staging "
            f"preserved: {reason}"
        )

    def _header_words_unchecked(self) -> Any:
        """Private header view used only at explicit validation boundaries."""
        import torch

        header_bytes = state_layout_v2().field("scalar_header_counters").bytes
        return self.__state[:, :header_bytes].view(torch.int64)

    def _validate_header_rows(
        self,
        rows: list[list[int]],
        *,
        expected_next_tick: int,
        expected_ticks_completed: int,
    ) -> None:
        if len(rows) != self.contract.B:
            self._poison("header batch extent mismatch")
        counter_bound = (
            MAX_CAPABILITY_COUNTER_INCREMENT_PER_TICK_V2 * expected_ticks_completed
        )
        for world, row in enumerate(rows):
            if len(row) != 24:
                self._poison(f"world {world} header word extent mismatch")
            values = tuple(int(value) for value in row)
            if any(value < 0 for value in values):
                self._poison(
                    f"world {world} contains signed-overflowed u64 header data"
                )
            if values[0] != self.__expected_world_uids[world]:
                self._poison(f"world {world} immutable world UID mismatch")
            if values[1] != expected_next_tick:
                self._poison(
                    f"world {world} next tick {values[1]} != {expected_next_tick}"
                )
            if values[4] != 0:
                self._poison(f"world {world} status is {values[4]}")
            if values[5] != expected_ticks_completed:
                self._poison(
                    f"world {world} ticks_completed {values[5]} != "
                    f"{expected_ticks_completed}"
                )
            if values[13] != self.__expected_rng_keys[world]:
                self._poison(f"world {world} immutable RNG key mismatch")
            if values[14] != self.__expected_rng_semantic_epoch:
                self._poison(f"world {world} RNG semantic epoch mismatch")
            if values[22] != 0 or values[23] != 0:
                self._poison(f"world {world} reserved header words are nonzero")
            capability_counters = values[6:13]
            if any(value > counter_bound for value in capability_counters):
                self._poison(f"world {world} capability counter exceeds safe bound")
            birth_count, death_count, mutation_count, recombination_count = values[
                15:19
            ]
            if birth_count > RESIDENT_CAPACITY_C_V2 * expected_ticks_completed:
                self._poison(f"world {world} birth count is impossible")
            if death_count > RESIDENT_CAPACITY_C_V2 * expected_ticks_completed:
                self._poison(f"world {world} death count is impossible")
            if mutation_count > 2 * 128 * birth_count:
                self._poison(f"world {world} mutation count is impossible")
            if recombination_count > 16 * birth_count:
                self._poison(f"world {world} recombination count is impossible")
            if values[10] != birth_count:
                self._poison(f"world {world} birth capability/count mismatch")
            if values[11] != mutation_count:
                self._poison(f"world {world} mutation capability/count mismatch")
            if values[12] != recombination_count:
                self._poison(f"world {world} recombination capability/count mismatch")
            fixture_population_delta = (
                RESIDENT_CAPACITY_C_V2 - INITIAL_POPULATION_V2
                if self.__population_fixture_id == FULL_CAPACITY_BENCHMARK_FIXTURE_V2
                else 0
            )
            expected_active = (
                INITIAL_POPULATION_V2
                + fixture_population_delta
                + birth_count
                - death_count
            )
            if not 0 <= expected_active <= RESIDENT_CAPACITY_C_V2:
                self._poison(f"world {world} lifecycle counts imply invalid activity")
            if values[3] != expected_active:
                self._poison(f"world {world} active count is inconsistent")
            if values[2] != (
                INITIAL_POPULATION_V2 + 1 + fixture_population_delta + birth_count
            ):
                self._poison(f"world {world} next agent ID is inconsistent")
            if values[2] > MAX_SIGNED_INT64:
                self._poison(f"world {world} next agent ID exceeds signed int64")
            if any(value > 1024 for value in values[19:22]):
                self._poison(f"world {world} bounded ecological summary is invalid")

    def _validate_committed_frontier(self) -> list[list[int]]:
        self._ensure_no_pending("committed frontier validation")
        rows = self._header_words_unchecked().cpu().tolist()
        self._validate_header_rows(
            rows,
            expected_next_tick=self.committed_next_absolute_tick,
            expected_ticks_completed=self.__advanced_ticks,
        )
        return rows

    def _validate_pending_frontier(
        self, pending: _PendingDispatchV2
    ) -> list[list[int]]:
        self._ensure_usable("pending frontier validation")
        if pending is not self.__pending_dispatch:
            self._poison("pending dispatch identity changed")
        try:
            rows = self._header_words_unchecked().cpu().tolist()
        except Exception as exc:
            self._poison(
                f"asynchronous native launch/frontier synchronization failed: {exc}"
            )
        self._validate_header_rows(
            rows,
            expected_next_tick=pending.end_next_absolute_tick,
            expected_ticks_completed=self.__advanced_ticks + pending.tick_count,
        )
        return rows

    def _checked_ticks(self, ticks: int) -> None:
        if type(ticks) is not int or ticks <= 0:
            raise CudaFullEvolutionRuntimeError("ticks must be a positive integer")
        if ticks > self.remaining_ticks:
            raise CudaFullEvolutionRuntimeError(
                "advance exceeds the contract's absolute tick horizon"
            )

    def _advance(self, ticks: int, *, reference: bool, capability_mask: int) -> None:
        self._ensure_no_pending("advance")
        self._checked_ticks(ticks)
        if ticks > self.contract.temporal_k:
            raise CudaFullEvolutionRuntimeError(
                "one public advance is exactly one dispatch and cannot exceed temporal_k"
            )
        if (
            type(capability_mask) is not int
            or not 0 <= capability_mask <= FULL_CAPABILITY_MASK_V2
        ):
            raise CudaFullEvolutionRuntimeError("diagnostic capability mask is invalid")
        if reference and self.__reference_scratch is None:
            raise CudaFullEvolutionRuntimeError(
                "reference scratch was not enabled when this owner was allocated"
            )
        start_tick = self.committed_next_absolute_tick
        if reference:
            self.__last_execution_receipt = None
            self.__last_execution_plan_k = None
            self.__loaded.module.advance_reference(
                self.__state,
                self.__reference_scratch,
                self.__event_rows,
                self.__event_counts,
                ticks,
                self.contract.temporal_k,
                capability_mask,
            )
        elif capability_mask != FULL_CAPABILITY_MASK_V2:
            self.__last_execution_receipt = None
            self.__last_execution_plan_k = None
            self.__loaded.module.advance(
                self.__state,
                self.__event_rows,
                self.__event_counts,
                ticks,
                self.contract.temporal_k,
                capability_mask,
            )
        else:
            plan = self.__execution_plans.get(ticks)
            topology = self.__execution_plan_topologies.get(ticks)
            if plan is None or topology is None:
                raise CudaFullEvolutionRuntimeError(
                    "native execution plan is unavailable"
                )
            if type(plan) is not self.__native_execution_plan_type:
                self._poison("native execution-plan object identity changed")
            try:
                plan.launch(start_tick, ticks)
                native_receipt = plan.last_execution_receipt()
            except Exception as exc:
                self._poison(f"native execution-plan launch failed: {exc}")
            native_receipt = self._bind_native_execution_receipt(
                native_receipt,
                topology=topology,
                start_next_absolute_tick=start_tick,
                tick_count=ticks,
            )
            self.__last_execution_receipt = {
                **native_receipt,
                "plan_identity_sha256": topology["plan_identity_sha256"],
                "pointer_binding_identity_sha256": topology[
                    "pointer_binding_identity_sha256"
                ],
                "capture_identity_sha256": topology["capture_identity_sha256"],
                "artifact_sha256": self.__loaded.artifact_sha256,
                "manifest_sha256": self.__loaded.manifest_sha256,
                "device_identity": dict(self.__loaded.device_identity),
                "runtime_status": "pending_event_validation",
                "event_status": "pending_not_drained",
            }
            self.__last_execution_plan_k = ticks
        self.__pending_dispatch = _PendingDispatchV2(start_tick, ticks)

    def advance_reference(self, ticks: int) -> None:
        self._advance(ticks, reference=True, capability_mask=FULL_CAPABILITY_MASK_V2)

    def advance(self, ticks: int) -> None:
        self._advance(ticks, reference=False, capability_mask=FULL_CAPABILITY_MASK_V2)

    def advance_diagnostic(
        self, ticks: int, *, capability_mask: int, reference: bool = False
    ) -> None:
        """Explicit causal-ablation hook; production calls always use the full mask."""
        self._advance(ticks, reference=reference, capability_mask=capability_mask)

    def drain_events(self, *, to_cpu: bool = True) -> CudaFullEvolutionEventDispatchV2:
        """Validate and copy one dispatch outside hot timing, then hash its content."""
        if type(to_cpu) is not bool:
            raise CudaFullEvolutionRuntimeError("to_cpu must be boolean")
        self._ensure_usable("event drain")
        pending = self.__pending_dispatch
        if pending is None:
            raise CudaFullEvolutionRuntimeError(
                "event drain requires a pending dispatch"
            )
        if self.__drained_dispatch is not None:
            raise CudaFullEvolutionRuntimeError(
                "pending dispatch was already drained and awaits a sink receipt"
            )

        drain_started = perf_counter()
        header_rows = self._validate_pending_frontier(pending)
        import torch

        # Project the bounded dispatch before decoding it.  Decoding the dense
        # event ABI on the device creates several event-sized temporaries; at a
        # correctly admitted high-B/K boundary those temporaries can exceed the
        # owner's remaining VRAM even though the resident simulation itself
        # fits.  Event validation and sink acknowledgement are explicitly host
        # boundaries and are outside transition timing, so the CPU projection
        # is the authoritative validation surface.
        validation_event_rows = self.__event_rows.detach().cpu()
        validation_event_counts = self.__event_counts.detach().cpu()
        if to_cpu:
            event_rows = validation_event_rows
            event_counts = validation_event_counts
        else:
            event_rows = self.__event_rows.detach().clone()
            event_counts = self.__event_counts.detach().clone()

        rows_per_tick = CUDA_NATIVE_LIFECYCLE_EVENT_ROWS_PER_WORLD_PER_TICK_V2
        words = validation_event_rows.view(torch.int64)
        tail = words[:, :, 5]
        event_kind = tail & 0xFFFF
        site_index = (tail >> 16) & 0xFFFF
        status = (tail >> 32) & 0xFFFF
        reserved = (tail >> 48) & 0xFFFF
        row_index = torch.arange(
            validation_event_rows.shape[1],
            dtype=torch.int64,
            device=validation_event_rows.device,
        ).view(1, -1)
        tick_offset = row_index // rows_per_tick
        within_tick = row_index % rows_per_tick
        canonical_slot = within_tick % RESIDENT_CAPACITY_C_V2
        expected_kind = torch.where(
            within_tick < RESIDENT_CAPACITY_C_V2,
            torch.ones_like(within_tick),
            torch.full_like(within_tick, 2),
        )
        committed = status == 1
        expected_event_id = (expected_kind << 48) | canonical_slot
        expected_tick = pending.start_next_absolute_tick + tick_offset
        event_counts_i64 = validation_event_counts.to(torch.int64)
        final_next_agent_ids = torch.tensor(
            [int(row[2]) for row in header_rows],
            dtype=torch.int64,
            device=validation_event_rows.device,
        ).view(-1, 1)
        valid = ((status == 0) | committed).all()
        valid = valid & (words[~committed] == 0).all()
        valid = valid & (~committed | (tick_offset < pending.tick_count)).all()
        valid = valid & (~committed | (event_kind == expected_kind)).all()
        valid = valid & (~committed | (words[:, :, 0] == expected_tick)).all()
        valid = valid & (~committed | (words[:, :, 1] == expected_event_id)).all()
        valid = valid & (~committed | (words[:, :, 2] > 0)).all()
        valid = valid & (~committed | (words[:, :, 2] < final_next_agent_ids)).all()
        valid = valid & (~committed | (words[:, :, 3] >= 0)).all()
        valid = valid & (~committed | (words[:, :, 4] >= 0)).all()
        valid = (
            valid
            & (
                ~committed
                | (words[:, :, 3] == 0)
                | (words[:, :, 3] < final_next_agent_ids)
            ).all()
        )
        valid = (
            valid
            & (
                ~committed
                | (words[:, :, 4] == 0)
                | (words[:, :, 4] < final_next_agent_ids)
            ).all()
        )
        committed_birth = committed & (expected_kind == 2)
        valid = valid & (~committed_birth | (words[:, :, 3] > 0)).all()
        valid = valid & (~committed_birth | (words[:, :, 4] > 0)).all()
        valid = valid & (~committed_birth | (words[:, :, 3] != words[:, :, 4])).all()
        valid = valid & (~committed | (site_index < 1024)).all()
        valid = valid & (reserved == 0).all()
        valid = valid & (event_counts_i64 >= 0).all()
        valid = valid & (event_counts_i64 <= rows_per_tick * pending.tick_count).all()
        valid = valid & (event_counts_i64 == committed.to(torch.int64).sum(dim=1)).all()
        if not bool(valid.item()):
            self._poison("bounded lifecycle event row/count validation failed")

        torch.cuda.synchronize(self.__state.device)
        drain_elapsed_seconds = perf_counter() - drain_started

        world_uids = tuple(int(row[0]) for row in header_rows)
        hash_started = perf_counter()
        content_sha256 = _event_content_sha256(
            event_rows=event_rows,
            event_counts=event_counts,
            world_uids=world_uids,
            start_next_absolute_tick=pending.start_next_absolute_tick,
            end_next_absolute_tick=pending.end_next_absolute_tick,
            tick_count=pending.tick_count,
            artifact_sha256=self.__loaded.artifact_sha256,
            manifest_sha256=self.__loaded.manifest_sha256,
        )
        content_hash_elapsed_seconds = perf_counter() - hash_started
        dispatch = CudaFullEvolutionEventDispatchV2(
            event_rows=event_rows,
            event_counts=event_counts,
            world_uids=world_uids,
            start_next_absolute_tick=pending.start_next_absolute_tick,
            end_next_absolute_tick=pending.end_next_absolute_tick,
            tick_count=pending.tick_count,
            drain_elapsed_seconds=drain_elapsed_seconds,
            content_hash_elapsed_seconds=content_hash_elapsed_seconds,
            content_sha256=content_sha256,
            artifact_sha256=self.__loaded.artifact_sha256,
            manifest_sha256=self.__loaded.manifest_sha256,
        )
        self.__drained_dispatch = dispatch
        if self.__last_execution_receipt is not None:
            self.__last_execution_receipt.update(
                {
                    "runtime_status": "event_content_validated_not_acknowledged",
                    "event_status": "drained_content_validated",
                    "event_content_sha256": content_sha256,
                    "event_counts": [
                        int(value) for value in event_counts.cpu().tolist()
                    ],
                    "committed_world_ticks": 0,
                }
            )
        return dispatch

    def drain_events_sparse_v3(
        self,
    ) -> CudaFullEvolutionSparseEventDispatchV3:
        """Compact, validate, and copy only committed lifecycle rows."""
        self._ensure_usable("sparse event drain")
        pending = self.__pending_dispatch
        if pending is None:
            raise CudaFullEvolutionRuntimeError(
                "sparse event drain requires a pending dispatch"
            )
        if self.__drained_dispatch is not None:
            raise CudaFullEvolutionRuntimeError(
                "pending dispatch was already drained and awaits a sink receipt"
            )

        drain_started = perf_counter()
        header_rows = self._validate_pending_frontier(pending)
        import torch

        count_projection_started = perf_counter()
        event_counts_cpu = self.__event_counts.detach().cpu()
        event_counts = tuple(int(value) for value in event_counts_cpu.tolist())
        maximum_per_world = (
            CUDA_NATIVE_LIFECYCLE_EVENT_ROWS_PER_WORLD_PER_TICK_V2 * pending.tick_count
        )
        if any(count < 0 or count > maximum_per_world for count in event_counts):
            self._poison("sparse event count projection exceeds dispatch bounds")
        world_offsets_list = [0]
        for count in event_counts:
            next_offset = world_offsets_list[-1] + count
            if next_offset > MAX_SIGNED_INT64 // CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3:
                self._poison("sparse event projection exceeds signed-int64 bytes")
            world_offsets_list.append(next_offset)
        world_offsets = tuple(world_offsets_list)
        count_projection_elapsed_seconds = perf_counter() - count_projection_started

        compaction_started = perf_counter()
        world_offsets_device = torch.tensor(
            world_offsets, dtype=torch.int64, device=self.__state.device
        )
        packed_rows_device = torch.empty(
            (world_offsets[-1], CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3),
            dtype=torch.uint8,
            device=self.__state.device,
        )
        validation_device = torch.empty(
            (self.contract.B,), dtype=torch.int32, device=self.__state.device
        )
        try:
            self.__loaded.module.compact_events_v3(
                self.__state,
                self.__event_rows,
                self.__event_counts,
                world_offsets_device,
                packed_rows_device,
                validation_device,
                pending.start_next_absolute_tick,
                pending.tick_count,
                self.contract.temporal_k,
            )
            validation = tuple(
                int(value) for value in validation_device.detach().cpu().tolist()
            )
            packed_rows_cpu = packed_rows_device.detach().cpu()
        except Exception as exc:
            self._poison(f"native sparse event compaction failed: {exc}")
        if any(validation):
            failures = [
                (world, value) for world, value in enumerate(validation) if value
            ][:8]
            self._poison(
                "native sparse event validation failed for world/error-bit pairs "
                f"{failures}"
            )
        final_next_agent_ids = tuple(int(row[2]) for row in header_rows)
        try:
            _validate_sparse_event_projection_v3(
                packed_rows=packed_rows_cpu,
                world_offsets=world_offsets,
                event_counts=event_counts,
                final_next_agent_ids=final_next_agent_ids,
                tick_count=pending.tick_count,
            )
        except CudaFullEvolutionRuntimeError as exc:
            self._poison(str(exc))
        packed_events = packed_rows_cpu.numpy().tobytes(order="C")
        compaction_and_transfer_elapsed_seconds = perf_counter() - compaction_started
        drain_elapsed_seconds = perf_counter() - drain_started

        world_uids = tuple(int(row[0]) for row in header_rows)
        hash_started = perf_counter()
        content_sha256 = _sparse_event_content_sha256(
            packed_events=packed_events,
            world_offsets=world_offsets,
            event_counts=event_counts,
            final_next_agent_ids=final_next_agent_ids,
            world_uids=world_uids,
            start_next_absolute_tick=pending.start_next_absolute_tick,
            end_next_absolute_tick=pending.end_next_absolute_tick,
            tick_count=pending.tick_count,
            artifact_sha256=self.__loaded.artifact_sha256,
            manifest_sha256=self.__loaded.manifest_sha256,
        )
        content_hash_elapsed_seconds = perf_counter() - hash_started
        dispatch = CudaFullEvolutionSparseEventDispatchV3(
            packed_events=packed_events,
            world_offsets=world_offsets,
            event_counts=event_counts,
            final_next_agent_ids=final_next_agent_ids,
            world_uids=world_uids,
            start_next_absolute_tick=pending.start_next_absolute_tick,
            end_next_absolute_tick=pending.end_next_absolute_tick,
            tick_count=pending.tick_count,
            drain_elapsed_seconds=drain_elapsed_seconds,
            count_projection_elapsed_seconds=count_projection_elapsed_seconds,
            compaction_and_transfer_elapsed_seconds=(
                compaction_and_transfer_elapsed_seconds
            ),
            content_hash_elapsed_seconds=content_hash_elapsed_seconds,
            content_sha256=content_sha256,
            artifact_sha256=self.__loaded.artifact_sha256,
            manifest_sha256=self.__loaded.manifest_sha256,
        )
        self.__drained_dispatch = dispatch
        if self.__last_execution_receipt is not None:
            self.__last_execution_receipt.update(
                {
                    "runtime_status": "sparse_event_content_validated_not_acknowledged",
                    "event_status": "sparse_v3_content_validated",
                    "event_wire_abi": CUDA_NATIVE_SPARSE_EVENT_WIRE_ABI_V3,
                    "event_content_sha256": content_sha256,
                    "event_counts": list(event_counts),
                    "sparse_event_count": dispatch.event_count,
                    "sparse_event_wire_nbytes": dispatch.wire_nbytes,
                    "committed_world_ticks": 0,
                }
            )
        return dispatch

    def acknowledge_event_drain(
        self, receipt: CudaFullEvolutionEventSinkReceiptV2
    ) -> None:
        """Commit host time only for a sealed receipt over unchanged event bytes."""
        self._ensure_usable("event sink receipt acknowledgement")
        pending = self.__pending_dispatch
        dispatch = self.__drained_dispatch
        if pending is None or dispatch is None:
            raise CudaFullEvolutionRuntimeError(
                "no drained dispatch awaits a sink receipt"
            )
        if type(dispatch) is not CudaFullEvolutionEventDispatchV2:
            raise CudaFullEvolutionRuntimeError(
                "pending dispatch uses the sparse V3 sink receipt path"
            )
        self._validate_pending_frontier(pending)
        try:
            if type(receipt) is not CudaFullEvolutionEventSinkReceiptV2:
                raise CudaFullEvolutionRuntimeError(
                    "acknowledgement requires a sealed V2 sink receipt"
                )
            if receipt.schema_version != EVENT_SINK_RECEIPT_SCHEMA_V2:
                raise CudaFullEvolutionRuntimeError("sink receipt schema mismatch")
            if receipt.time_authority != CUDA_NATIVE_TIME_AUTHORITY_V1:
                raise CudaFullEvolutionRuntimeError(
                    "sink receipt time authority mismatch"
                )
            _checked_sink_token(receipt.sink_id, "receipt sink_id")
            _checked_sink_token(receipt.sink_receipt_id, "receipt sink_receipt_id")
            if not _valid_sha256(
                receipt.receipt_sha256
            ) or receipt.receipt_sha256 != _canonical_sha256(
                _event_sink_receipt_body(receipt)
            ):
                raise CudaFullEvolutionRuntimeError("sink receipt seal mismatch")
            exact_receipt_identity = (
                receipt.dispatch_content_sha256 == dispatch.content_sha256
                and receipt.world_uids == dispatch.world_uids
                and receipt.start_next_absolute_tick
                == dispatch.start_next_absolute_tick
                and receipt.end_next_absolute_tick == dispatch.end_next_absolute_tick
                and receipt.tick_count == dispatch.tick_count
                and receipt.artifact_sha256 == dispatch.artifact_sha256
                and receipt.manifest_sha256 == dispatch.manifest_sha256
            )
            if not exact_receipt_identity:
                raise CudaFullEvolutionRuntimeError(
                    "sink receipt does not identify the pending dispatch"
                )
            if _rehash_event_dispatch(dispatch) != dispatch.content_sha256:
                raise CudaFullEvolutionRuntimeError(
                    "drained event content changed after receipt binding"
                )
        except CudaFullEvolutionRuntimeError as exc:
            self._poison(str(exc))
        self._validate_pending_frontier(pending)
        self.__advanced_ticks += pending.tick_count
        self.__last_sink_receipt_sha256 = receipt.receipt_sha256
        self.__pending_dispatch = None
        self.__drained_dispatch = None
        if self.__last_execution_receipt is not None:
            self.__last_execution_receipt.update(
                {
                    "runtime_status": "committed",
                    "event_status": "exact_content_sink_acknowledged",
                    "sink_receipt_sha256": receipt.receipt_sha256,
                    "committed_world_ticks": self.contract.B * pending.tick_count,
                    "committed_next_absolute_tick": self.committed_next_absolute_tick,
                }
            )

    def acknowledge_sparse_event_drain_v3(
        self, receipt: CudaFullEvolutionSparseEventSinkReceiptV3
    ) -> None:
        """Commit host time for one unchanged immutable sparse sink payload."""
        self._ensure_usable("sparse event sink receipt acknowledgement")
        pending = self.__pending_dispatch
        dispatch = self.__drained_dispatch
        if pending is None or dispatch is None:
            raise CudaFullEvolutionRuntimeError(
                "no drained dispatch awaits a sparse sink receipt"
            )
        if type(dispatch) is not CudaFullEvolutionSparseEventDispatchV3:
            raise CudaFullEvolutionRuntimeError(
                "pending dispatch uses the dense V2 sink receipt path"
            )
        self._validate_pending_frontier(pending)
        try:
            if type(receipt) is not CudaFullEvolutionSparseEventSinkReceiptV3:
                raise CudaFullEvolutionRuntimeError(
                    "acknowledgement requires a sealed V3 sparse sink receipt"
                )
            if receipt.schema_version != SPARSE_EVENT_SINK_RECEIPT_SCHEMA_V3:
                raise CudaFullEvolutionRuntimeError(
                    "sparse sink receipt schema mismatch"
                )
            if receipt.time_authority != CUDA_NATIVE_TIME_AUTHORITY_V1:
                raise CudaFullEvolutionRuntimeError(
                    "sparse sink receipt time authority mismatch"
                )
            if receipt.wire_abi != CUDA_NATIVE_SPARSE_EVENT_WIRE_ABI_V3:
                raise CudaFullEvolutionRuntimeError(
                    "sparse sink receipt wire ABI mismatch"
                )
            _checked_sink_token(receipt.sink_id, "receipt sink_id")
            _checked_sink_token(receipt.sink_receipt_id, "receipt sink_receipt_id")
            if not _valid_sha256(
                receipt.receipt_sha256
            ) or receipt.receipt_sha256 != _canonical_sha256(
                _sparse_event_sink_receipt_body_v3(receipt)
            ):
                raise CudaFullEvolutionRuntimeError("sparse sink receipt seal mismatch")
            exact_receipt_identity = (
                receipt.dispatch_content_sha256 == dispatch.content_sha256
                and receipt.world_uids == dispatch.world_uids
                and receipt.start_next_absolute_tick
                == dispatch.start_next_absolute_tick
                and receipt.end_next_absolute_tick == dispatch.end_next_absolute_tick
                and receipt.tick_count == dispatch.tick_count
                and receipt.artifact_sha256 == dispatch.artifact_sha256
                and receipt.manifest_sha256 == dispatch.manifest_sha256
            )
            if not exact_receipt_identity:
                raise CudaFullEvolutionRuntimeError(
                    "sparse sink receipt does not identify the pending dispatch"
                )
            if _rehash_sparse_event_dispatch_v3(dispatch) != dispatch.content_sha256:
                raise CudaFullEvolutionRuntimeError(
                    "sparse event content changed after receipt binding"
                )
        except CudaFullEvolutionRuntimeError as exc:
            self._poison(str(exc))
        self._validate_pending_frontier(pending)
        self.__advanced_ticks += pending.tick_count
        self.__last_sink_receipt_sha256 = receipt.receipt_sha256
        self.__pending_dispatch = None
        self.__drained_dispatch = None
        if self.__last_execution_receipt is not None:
            self.__last_execution_receipt.update(
                {
                    "runtime_status": "committed",
                    "event_status": "exact_sparse_content_sink_acknowledged",
                    "event_wire_abi": CUDA_NATIVE_SPARSE_EVENT_WIRE_ABI_V3,
                    "sink_receipt_sha256": receipt.receipt_sha256,
                    "committed_world_ticks": self.contract.B * pending.tick_count,
                    "committed_next_absolute_tick": self.committed_next_absolute_tick,
                }
            )

    def force_test_conditions(self) -> None:
        """Arrange deterministic rare-event sentinels without changing transition code."""
        self._validate_committed_frontier()
        self.__loaded.module.force_test_conditions(self.__state)
        self._validate_committed_frontier()

    def force_full_capacity_benchmark_fixture_v2(
        self,
    ) -> CudaFullEvolutionCheckpointV2:
        """Create and seal the distinct fresh-start technical C512 fixture."""
        import torch

        rows = self._validate_committed_frontier()
        if (
            self.__advanced_ticks != 0
            or self.__last_sink_receipt_sha256 is not None
            or self.__population_fixture_id is not None
            or any(int(row[3]) != INITIAL_POPULATION_V2 for row in rows)
        ):
            raise CudaFullEvolutionRuntimeError(
                "full-capacity benchmark fixture requires one fresh canonical P256 "
                "committed frontier"
            )
        current_stream_handle = int(
            torch.cuda.current_stream(self.__state.device).cuda_stream
        )
        if any(
            topology["construction_stream_handle"] != current_stream_handle
            for topology in self.__execution_plan_topologies.values()
        ):
            raise CudaFullEvolutionRuntimeError(
                "full-capacity benchmark fixture must use the immutable execution-plan "
                "stream"
            )
        try:
            self.__loaded.module.force_full_capacity_benchmark_fixture_v2(self.__state)
        except Exception as exc:
            self._poison(f"native full-capacity fixture construction failed: {exc}")
        self.__population_fixture_id = FULL_CAPACITY_BENCHMARK_FIXTURE_V2
        validated = self._validate_committed_frontier()
        if any(int(row[3]) != RESIDENT_CAPACITY_C_V2 for row in validated):
            self._poison("native full-capacity fixture did not produce exact C512")
        return self.checkpoint()

    def fork(self) -> "CudaFullEvolutionRuntimeV2":
        """Deep-copy a verified owner for exact GPU differential tests."""
        self._validate_committed_frontier()
        return type(self)(
            contract=self.contract,
            loaded=self.__loaded,
            state=self.__state.clone(),
            event_rows=self.__event_rows.clone(),
            event_counts=self.__event_counts.clone(),
            reference_scratch=(
                None
                if self.__reference_scratch is None
                else self.__reference_scratch.clone()
            ),
            initial_next_absolute_tick=self.__initial_next_absolute_tick,
            expected_world_uids=self.__expected_world_uids,
            expected_rng_keys=self.__expected_rng_keys,
            expected_rng_semantic_epoch=self.__expected_rng_semantic_epoch,
            advanced_ticks=self.__advanced_ticks,
            last_sink_receipt_sha256=self.__last_sink_receipt_sha256,
            execution_mode=self.__execution_mode,
            population_fixture_id=self.__population_fixture_id,
        )

    def close(self) -> None:
        """Synchronize/destroy plans; poisoned owners retain cleanup authority only."""
        if self.__closed:
            return
        if self.__poisoned_reason is None:
            self._ensure_no_pending("close")
        if self.__execution_plans:
            close_errors = []
            for plan in self.__execution_plans.values():
                try:
                    plan.close()
                except Exception as exc:
                    close_errors.append(str(exc))
            if close_errors:
                raise CudaFullEvolutionRuntimeError(
                    "cannot close every native execution plan safely: "
                    + "; ".join(close_errors)
                )
        self.__execution_plans.clear()
        self.__closed = True

    def checkpoint(self, *, to_cpu: bool = True) -> CudaFullEvolutionCheckpointV2:
        """Seal canonical planes as immutable CPU bytes after frontier validation."""
        if to_cpu is not True:
            raise CudaFullEvolutionRuntimeError(
                "sealed V2 checkpoints are always immutable CPU bytes"
            )
        self._validate_committed_frontier()
        canonical_bytes_per_world = (
            state_layout_v2().canonical_checkpoint_bytes_per_world
        )
        # The canonical columns are a non-contiguous view because each resident
        # row also contains rebuildable cache bytes.  Making that view contiguous
        # before the device-to-host boundary allocates a second, surface-sized
        # CUDA tensor and can exhaust VRAM after an otherwise admissible run.
        # Copy the already-contiguous resident allocation once, then compact the
        # canonical view on the CPU where checkpoint staging belongs.
        resident_state_cpu = self.__state.detach().cpu()
        canonical_tensor = resident_state_cpu[
            :, :canonical_bytes_per_world
        ].contiguous()
        del resident_state_cpu
        canonical_state = canonical_tensor.numpy().tobytes(order="C")
        shape = (self.contract.B, canonical_bytes_per_world)
        nbytes = len(canonical_state)
        content_sha256 = hashlib.sha256(canonical_state).hexdigest()
        layout_sha256 = _canonical_state_layout_sha256(
            shape=shape,
            nbytes=nbytes,
            content_sha256=content_sha256,
        )
        provisional = CudaFullEvolutionCheckpointV2(
            contract=self.contract,
            canonical_state=canonical_state,
            canonical_state_shape=shape,
            canonical_state_nbytes=nbytes,
            canonical_state_sha256=content_sha256,
            canonical_state_layout_sha256=layout_sha256,
            initial_next_absolute_tick=self.__initial_next_absolute_tick,
            advanced_ticks=self.__advanced_ticks,
            acknowledged_event_next_tick=self.committed_next_absolute_tick,
            last_sink_receipt_sha256=self.__last_sink_receipt_sha256,
            world_uids=self.__expected_world_uids,
            rng_keys=self.__expected_rng_keys,
            rng_semantic_epoch=self.__expected_rng_semantic_epoch,
            artifact_path=self.__loaded.artifact_path,
            artifact_sha256=self.__loaded.artifact_sha256,
            manifest_sha256=self.__loaded.manifest_sha256,
            device_identity=dict(self.__loaded.device_identity),
            population_fixture_id=self.__population_fixture_id,
            metadata_seal_sha256="",
        )
        return CudaFullEvolutionCheckpointV2(
            contract=provisional.contract,
            canonical_state=provisional.canonical_state,
            canonical_state_shape=provisional.canonical_state_shape,
            canonical_state_nbytes=provisional.canonical_state_nbytes,
            canonical_state_sha256=provisional.canonical_state_sha256,
            canonical_state_layout_sha256=(provisional.canonical_state_layout_sha256),
            initial_next_absolute_tick=provisional.initial_next_absolute_tick,
            advanced_ticks=provisional.advanced_ticks,
            acknowledged_event_next_tick=provisional.acknowledged_event_next_tick,
            last_sink_receipt_sha256=provisional.last_sink_receipt_sha256,
            world_uids=provisional.world_uids,
            rng_keys=provisional.rng_keys,
            rng_semantic_epoch=provisional.rng_semantic_epoch,
            artifact_path=provisional.artifact_path,
            artifact_sha256=provisional.artifact_sha256,
            manifest_sha256=provisional.manifest_sha256,
            device_identity=dict(provisional.device_identity),
            population_fixture_id=provisional.population_fixture_id,
            metadata_seal_sha256=_canonical_sha256(
                _checkpoint_metadata_body(provisional)
            ),
        )

    def checkpoint_capture_source_v1(
        self,
    ) -> CudaFullEvolutionCheckpointCaptureSourceV1:
        """Borrow resident state at an exact committed frontier for ACS2."""

        self._validate_committed_frontier()
        return CudaFullEvolutionCheckpointCaptureSourceV1(
            contract=self.contract,
            resident_state=self.__state.detach(),
            canonical_bytes_per_world=(
                state_layout_v2().canonical_checkpoint_bytes_per_world
            ),
            initial_next_absolute_tick=self.__initial_next_absolute_tick,
            advanced_ticks=self.__advanced_ticks,
            acknowledged_event_next_tick=self.committed_next_absolute_tick,
            last_sink_receipt_sha256=self.__last_sink_receipt_sha256,
            world_uids=self.__expected_world_uids,
            rng_keys=self.__expected_rng_keys,
            rng_semantic_epoch=self.__expected_rng_semantic_epoch,
            artifact_path=self.__loaded.artifact_path,
            artifact_sha256=self.__loaded.artifact_sha256,
            manifest_sha256=self.__loaded.manifest_sha256,
            device_identity=dict(self.__loaded.device_identity),
            population_fixture_id=self.__population_fixture_id,
        )

    def clone(self, *, to_cpu: bool = True) -> CudaFullEvolutionCheckpointV2:
        """Compatibility spelling; sealed V2 checkpoints are always immutable CPU bytes."""
        if to_cpu is not True:
            raise CudaFullEvolutionRuntimeError(
                "sealed V2 checkpoints are always immutable CPU bytes"
            )
        return self.checkpoint()

    def next_absolute_ticks(self) -> list[int]:
        rows = self._validate_committed_frontier()
        return [int(row[1]) for row in rows]

    def world_uids(self) -> list[int]:
        rows = self._validate_committed_frontier()
        return [int(row[0]) for row in rows]

    def counters(self) -> list[dict[str, int]]:
        rows = self._validate_committed_frontier()
        return [
            dict(zip(COUNTER_NAMES_V2, map(int, row[6:13]), strict=True))
            for row in rows
        ]


def initialize(
    contract: CudaFullEvolutionNativeContractV2 | Mapping[str, Any],
    *,
    seed: int,
    next_absolute_tick: int | None = None,
    artifact_path: str | Path | None = None,
    device: str | Any = "cuda",
    enable_reference: bool = False,
    world_uids: Sequence[int] | None = None,
    first_world_uid: int = 1,
    execution_mode: str = "eager_monolith",
) -> CudaFullEvolutionRuntimeV2:
    return CudaFullEvolutionRuntimeV2.initialize(
        contract,
        seed=seed,
        next_absolute_tick=next_absolute_tick,
        artifact_path=artifact_path,
        device=device,
        enable_reference=enable_reference,
        world_uids=world_uids,
        first_world_uid=first_world_uid,
        execution_mode=execution_mode,
    )


__all__ = [
    "CHECKPOINT_CANONICAL_STATE_V2",
    "CHECKPOINT_SCHEMA_V2",
    "COUNTER_NAMES_V2",
    "CudaFullEvolutionCheckpointCaptureSourceV1",
    "CudaFullEvolutionCheckpointV2",
    "CudaFullEvolutionEventDispatchV2",
    "CudaFullEvolutionEventSinkReceiptV2",
    "CudaFullEvolutionRuntimeError",
    "CudaFullEvolutionRuntimeV2",
    "CudaFullEvolutionStateCloneV2",
    "EVENT_DISPATCH_SCHEMA_V2",
    "EVENT_FRONTIER_COMMIT_V2",
    "EVENT_SINK_RECEIPT_SCHEMA_V2",
    "EXECUTION_MODES_V2",
    "EXECUTION_PLAN_RECEIPT_SCHEMA_V2",
    "EXECUTION_PLAN_STREAM_POLICY_V2",
    "EXECUTION_PLAN_TOPOLOGY_SCHEMA_V2",
    "FULL_CAPABILITY_MASK_V2",
    "FULL_CAPACITY_BENCHMARK_FIXTURE_V2",
    "MAX_CAPABILITY_COUNTER_INCREMENT_PER_TICK_V2",
    "MAX_TOTAL_TICKS_COUNTER_SAFE_V2",
    "NATIVE_EXECUTION_RECEIPT_SCHEMA_V2",
    "PUBLIC_STATE_PROJECTION_V2",
    "RNG_SEMANTIC_EPOCH_V3",
    "checkpoint_runtime_projection_from_capture_source_v1",
    "initialize",
    "seal_event_sink_receipt_v2",
]
