"""Transactional direct-sparse owner for CUDA-golden Full Evolution V2."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import stat
from time import perf_counter, perf_counter_ns, process_time_ns
from typing import Any, Final, Mapping, MutableMapping, Sequence

from .contract import (
    CUDA_NATIVE_TIME_AUTHORITY_V1,
    INITIAL_POPULATION_V2,
    MAX_CAPABILITY_COUNTER_INCREMENT_PER_TICK_V2,
    MAX_SIGNED_INT64,
    RESIDENT_CAPACITY_C_V2,
    CudaFullEvolutionNativeContractV2,
    validate_cuda_full_evolution_contract_v2,
)
from .layout import (
    CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3,
    CUDA_NATIVE_SPARSE_EVENT_WIRE_ABI_V3,
    state_layout_v2,
)
from .native_loader import (
    LoadedCudaFullEvolutionNativeV2,
    load_prebuilt_cuda_full_evolution_v2,
)
from .science_native_world_runtime_v1 import (
    NATIVE_WORLD_CONTROL_WIDTH_V1,
    ScienceNativeWorldRuntimeError,
    ScienceNativeWorldRuntimeProjectionV1,
)
from .runtime import (
    COUNTER_NAMES_V2,
    FULL_CAPABILITY_MASK_V2,
    FULL_CAPACITY_BENCHMARK_FIXTURE_V2,
    RNG_SEMANTIC_EPOCH_V3,
    CudaFullEvolutionCheckpointV2,
    CudaFullEvolutionCheckpointCaptureSourceV1,
    CudaFullEvolutionRuntimeError,
    CudaFullEvolutionSparseEventDispatchV3,
    _canonical_sha256,
    _canonical_state_layout_sha256,
    _checkpoint_metadata_body,
    _checked_sink_token,
    _rehash_sparse_event_dispatch_v3,
    _sparse_event_content_sha256,
    _validate_checkpoint_envelope,
    _validate_sparse_event_projection_v3,
    _valid_sha256,
)

DIRECT_SPARSE_OWNER_SCHEMA_V3: Final[str] = (
    "eiel_full_evolution_cuda_direct_sparse_owner_v3"
)
DIRECT_SPARSE_EXECUTION_RECEIPT_SCHEMA_V3: Final[str] = (
    "eiel_full_evolution_cuda_direct_sparse_execution_receipt_v3"
)
DIRECT_SPARSE_OVERFLOW_BIT_V3: Final[int] = 1 << 11
DIRECT_SPARSE_MAX_CAPACITY_ROWS_V3: Final[int] = 2 * RESIDENT_CAPACITY_C_V2 * 64
DIRECT_SPARSE_ADMISSION_MARGIN_BYTES_V3: Final[int] = 64 << 20
DIRECT_SPARSE_DURABLE_ENVELOPE_SCHEMA_V3: Final[str] = (
    "eiel_cuda_full_evolution_direct_sparse_durable_envelope_v3"
)
DIRECT_SPARSE_DURABLE_SINK_RECEIPT_SCHEMA_V3: Final[str] = (
    "eiel_cuda_full_evolution_direct_sparse_durable_sink_receipt_v3"
)
DIRECT_SPARSE_DURABLE_ENVELOPE_MAGIC_V3: Final[bytes] = b"EIELDSV3"
DIRECT_SPARSE_CLAIM_CLASS_V3: Final[str] = "TECHNICAL_ONLY"
DIRECT_SPARSE_EVENT_FRONTIER_COMMIT_V3: Final[str] = (
    "exact_canonical_local_envelope_inode_digest_open_descriptor_receipt_v3"
)
COMPACT_SCIENCE_CAPTURE_SCHEMA_V1: Final[str] = (
    "eiel_full_evolution_cuda_compact_science_capture_v1"
)
COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1: Final[tuple[str, ...]] = (
    "action_0_count",
    "action_1_count",
    "action_2_count",
    "action_3_count",
    "action_4_count",
    "action_5_count",
    "action_6_count",
    "alive_ticks",
    "reproduction_eligible_count",
    "mate_candidate_count",
    "birth_attempt_count",
    "successful_parent_count",
    "resource_uptake_total",
    "movement_cost_total",
    "movement_attempt_count",
    "movement_success_count",
    "health_damage_total",
    "energy_delta_total",
    "attack_attempt_count",
    "attack_hit_count",
    "births_as_parent_count",
)
COMPACT_SCIENCE_AGENT_METRIC_WIDTH_V1: Final[int] = len(
    COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1
)
COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1: Final[int] = 132
COMPACT_SCIENCE_BIRTH_PROVENANCE_ENCODING_V1: Final[str] = (
    "linked_locus_mutation_direction_u2_and_recombination_haplotype_u2_le_v1"
)
_VERIFIED_ENVELOPE_FACTORY_TOKEN_V3: Final[object] = object()
_PREPARED_ENVELOPE_FACTORY_TOKEN_V3: Final[object] = object()


@dataclass(frozen=True, slots=True)
class DirectSparseDispatchMetricsV3:
    snapshot_elapsed_seconds: float
    rollback_elapsed_seconds: float
    retry_count: int
    selected_capacity_rows_per_world: int
    kernel_elapsed_ms_diagnostic_only: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int


@dataclass(frozen=True, slots=True)
class CudaFullEvolutionPinnedSnapshotCaptureFenceV1:
    """Completion fence for one direct resident-to-pinned snapshot capture."""

    absolute_tick: int
    state_bytes: int
    active_science_metrics_bytes: int
    source_device: str
    start_event: Any
    end_event: Any

    def wait(self) -> Mapping[str, object]:
        """Wait only on the capture worker and return compact copy telemetry."""

        self.end_event.synchronize()
        return {
            "schema_version": "eiel_cuda_pinned_snapshot_capture_fence_v1",
            "absolute_tick": self.absolute_tick,
            "source_device": self.source_device,
            "state_bytes": self.state_bytes,
            "active_science_metrics_bytes": self.active_science_metrics_bytes,
            "d2h_bytes": self.state_bytes + self.active_science_metrics_bytes,
            "cuda_event_time_ns": round(
                float(self.start_event.elapsed_time(self.end_event)) * 1_000_000
            ),
        }


@dataclass(frozen=True, slots=True)
class CudaFullEvolutionScienceCaptureV1:
    """Detached native scientific primitives for one exact dispatch.

    The tensors are CPU projections made only after the native transition has
    completed its ordinary validation.  They are acquisition data, not state
    or RNG inputs, and are therefore deliberately absent from the resident
    checkpoint identity.
    """

    action_agent_ids: Any
    action_values: Any
    birth_child_ids: Any
    birth_parent_a_ids: Any
    birth_parent_b_ids: Any
    birth_mutation_counts: Any
    birth_child_slots: Any
    birth_sites: Any
    birth_counts: Any
    start_next_absolute_tick: int
    end_next_absolute_tick: int
    world_uids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CudaFullEvolutionCompactScienceCaptureV1:
    """Exact event-aligned metrics and hereditary provenance for one dispatch."""

    event_agent_metrics: Any
    event_birth_provenance: Any
    metric_fields: tuple[str, ...]
    birth_provenance_encoding: str
    start_next_absolute_tick: int
    end_next_absolute_tick: int
    world_uids: tuple[int, ...]
    world_offsets: tuple[int, ...]
    event_content_sha256: str

    def __post_init__(self) -> None:
        if self.metric_fields != COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1:
            raise CudaFullEvolutionRuntimeError(
                "compact science metric field registry differs"
            )
        if tuple(self.event_agent_metrics.shape) != (
            self.world_offsets[-1],
            COMPACT_SCIENCE_AGENT_METRIC_WIDTH_V1,
        ):
            raise CudaFullEvolutionRuntimeError(
                "compact science event metrics do not align with sparse events"
            )
        if (
            self.birth_provenance_encoding
            != COMPACT_SCIENCE_BIRTH_PROVENANCE_ENCODING_V1
            or tuple(self.event_birth_provenance.shape)
            != (
                self.world_offsets[-1],
                COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1,
            )
        ):
            raise CudaFullEvolutionRuntimeError(
                "compact science birth provenance does not align with sparse events"
            )


def _capture_direct_sparse_dispatch_authority_v3(
    dispatch: CudaFullEvolutionSparseEventDispatchV3,
) -> tuple[object, ...]:
    return (
        dispatch.packed_events,
        dispatch.world_offsets,
        dispatch.event_counts,
        dispatch.final_next_agent_ids,
        dispatch.world_uids,
        dispatch.start_next_absolute_tick,
        dispatch.end_next_absolute_tick,
        dispatch.tick_count,
        dispatch.drain_elapsed_seconds,
        dispatch.count_projection_elapsed_seconds,
        dispatch.compaction_and_transfer_elapsed_seconds,
        dispatch.content_hash_elapsed_seconds,
        dispatch.content_sha256,
        dispatch.artifact_sha256,
        dispatch.manifest_sha256,
        dispatch.schema_version,
        dispatch.time_authority,
        dispatch.wire_abi,
    )


def _direct_sparse_dispatch_matches_authority_v3(
    dispatch: CudaFullEvolutionSparseEventDispatchV3,
    authority: tuple[object, ...],
) -> bool:
    current = _capture_direct_sparse_dispatch_authority_v3(dispatch)
    return all(current[index] is authority[index] for index in range(5)) and (
        current[5:] == authority[5:]
    )


@dataclass(frozen=True, slots=True)
class _PendingDirectSparseDispatchV3:
    start_next_absolute_tick: int
    tick_count: int
    dispatch: CudaFullEvolutionSparseEventDispatchV3
    dispatch_authority: tuple[object, ...]

    @property
    def end_next_absolute_tick(self) -> int:
        return self.start_next_absolute_tick + self.tick_count

    def verify_dispatch(self, dispatch: CudaFullEvolutionSparseEventDispatchV3) -> None:
        if type(dispatch) is not CudaFullEvolutionSparseEventDispatchV3:
            raise CudaFullEvolutionRuntimeError(
                "pending authority requires an exact sparse V3 dispatch"
            )
        if dispatch is not self.dispatch:
            raise CudaFullEvolutionRuntimeError(
                "pending authority is bound to another dispatch object"
            )
        if not _direct_sparse_dispatch_matches_authority_v3(
            dispatch, self.dispatch_authority
        ):
            raise CudaFullEvolutionRuntimeError(
                "pending immutable sparse dispatch changed after owner validation"
            )


@dataclass(frozen=True, slots=True)
class CudaFullEvolutionDirectSparseDurableSinkReceiptV3:
    """Sealed dispatch plus exact local durable-envelope identity."""

    dispatch_content_sha256: str
    sink_id: str
    sink_receipt_id: str
    sample_id: str
    world_uids: tuple[int, ...]
    start_next_absolute_tick: int
    end_next_absolute_tick: int
    tick_count: int
    artifact_sha256: str
    manifest_sha256: str
    durable_path: str
    durable_device: int
    durable_inode: int
    durable_link_count: int
    durable_mode: int
    durable_nbytes: int
    durable_content_sha256: str
    durable_mtime_ns: int
    durable_ctime_ns: int
    receipt_sha256: str
    schema_version: str = DIRECT_SPARSE_DURABLE_SINK_RECEIPT_SCHEMA_V3
    time_authority: str = CUDA_NATIVE_TIME_AUTHORITY_V1
    wire_abi: str = CUDA_NATIVE_SPARSE_EVENT_WIRE_ABI_V3
    envelope_schema_version: str = DIRECT_SPARSE_DURABLE_ENVELOPE_SCHEMA_V3


def _durable_sink_receipt_body_v3(
    receipt: CudaFullEvolutionDirectSparseDurableSinkReceiptV3,
) -> dict[str, Any]:
    return {
        "schema_version": receipt.schema_version,
        "time_authority": receipt.time_authority,
        "wire_abi": receipt.wire_abi,
        "envelope_schema_version": receipt.envelope_schema_version,
        "dispatch_content_sha256": receipt.dispatch_content_sha256,
        "sink_id": receipt.sink_id,
        "sink_receipt_id": receipt.sink_receipt_id,
        "sample_id": receipt.sample_id,
        "world_uids": list(receipt.world_uids),
        "start_next_absolute_tick": receipt.start_next_absolute_tick,
        "end_next_absolute_tick": receipt.end_next_absolute_tick,
        "tick_count": receipt.tick_count,
        "artifact_sha256": receipt.artifact_sha256,
        "manifest_sha256": receipt.manifest_sha256,
        "durable_path": receipt.durable_path,
        "durable_device": receipt.durable_device,
        "durable_inode": receipt.durable_inode,
        "durable_link_count": receipt.durable_link_count,
        "durable_mode": receipt.durable_mode,
        "durable_nbytes": receipt.durable_nbytes,
        "durable_content_sha256": receipt.durable_content_sha256,
        "durable_mtime_ns": receipt.durable_mtime_ns,
        "durable_ctime_ns": receipt.durable_ctime_ns,
    }


def _validate_direct_sparse_dispatch_v3(
    dispatch: CudaFullEvolutionSparseEventDispatchV3,
) -> None:
    if type(dispatch) is not CudaFullEvolutionSparseEventDispatchV3:
        raise CudaFullEvolutionRuntimeError(
            "durable envelope requires an exact sparse V3 dispatch"
        )
    if (
        not _valid_sha256(dispatch.content_sha256)
        or _rehash_sparse_event_dispatch_v3(dispatch) != dispatch.content_sha256
    ):
        raise CudaFullEvolutionRuntimeError(
            "durable envelope dispatch content seal mismatch"
        )
    import torch

    packed_rows = (
        torch.empty((0, CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3), dtype=torch.uint8)
        if not dispatch.packed_events
        else torch.frombuffer(
            bytearray(dispatch.packed_events), dtype=torch.uint8
        ).view(dispatch.event_count, CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3)
    )
    _validate_sparse_event_projection_v3(
        packed_rows=packed_rows,
        world_offsets=dispatch.world_offsets,
        event_counts=dispatch.event_counts,
        final_next_agent_ids=dispatch.final_next_agent_ids,
        tick_count=dispatch.tick_count,
    )


def _encode_direct_sparse_durable_envelope_unchecked_v3(
    dispatch: CudaFullEvolutionSparseEventDispatchV3, *, sample_id: str
) -> bytes:
    checked_sample_id = _checked_sink_token(sample_id, "sample_id")
    metadata = {
        "schema_version": DIRECT_SPARSE_DURABLE_ENVELOPE_SCHEMA_V3,
        "claim_class": DIRECT_SPARSE_CLAIM_CLASS_V3,
        "sample_id": checked_sample_id,
        "dispatch_schema_version": dispatch.schema_version,
        "dispatch_time_authority": dispatch.time_authority,
        "wire_abi": dispatch.wire_abi,
        "dispatch_content_sha256": dispatch.content_sha256,
        "world_uids": list(dispatch.world_uids),
        "start_next_absolute_tick": dispatch.start_next_absolute_tick,
        "end_next_absolute_tick": dispatch.end_next_absolute_tick,
        "tick_count": dispatch.tick_count,
        "event_counts": list(dispatch.event_counts),
        "final_next_agent_ids": list(dispatch.final_next_agent_ids),
        "world_offset_count": len(dispatch.world_offsets),
        "packed_event_count": dispatch.event_count,
        "packed_event_nbytes": len(dispatch.packed_events),
        "artifact_sha256": dispatch.artifact_sha256,
        "manifest_sha256": dispatch.manifest_sha256,
    }
    header = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    offsets = b"".join(
        value.to_bytes(8, byteorder="little", signed=False)
        for value in dispatch.world_offsets
    )
    return b"".join(
        (
            DIRECT_SPARSE_DURABLE_ENVELOPE_MAGIC_V3,
            len(header).to_bytes(8, byteorder="little", signed=False),
            header,
            offsets,
            dispatch.packed_events,
        )
    )


def encode_direct_sparse_durable_envelope_v3(
    dispatch: CudaFullEvolutionSparseEventDispatchV3, *, sample_id: str
) -> bytes:
    _validate_direct_sparse_dispatch_v3(dispatch)
    return _encode_direct_sparse_durable_envelope_unchecked_v3(
        dispatch, sample_id=sample_id
    )


class PreparedDirectSparseDurableEnvelopeV3:
    """Owner-issued canonical bytes for one exact pending immutable dispatch."""

    __slots__ = (
        "__artifact_sha256",
        "__content_sha256",
        "__dispatch",
        "__dispatch_content_sha256",
        "__event_counts",
        "__final_next_agent_ids",
        "__manifest_sha256",
        "__packed_events",
        "__payload",
        "__sample_id",
        "__schema_version",
        "__tick_identity",
        "__time_authority",
        "__wire_abi",
        "__world_offsets",
        "__world_uids",
    )

    def __init__(
        self,
        *,
        dispatch: CudaFullEvolutionSparseEventDispatchV3,
        sample_id: str,
        payload: bytes,
        content_sha256: str,
        _factory_token: object,
    ) -> None:
        if _factory_token is not _PREPARED_ENVELOPE_FACTORY_TOKEN_V3:
            raise CudaFullEvolutionRuntimeError(
                "prepared durable envelope capabilities are owner-factory-owned"
            )
        if type(dispatch) is not CudaFullEvolutionSparseEventDispatchV3:
            raise CudaFullEvolutionRuntimeError(
                "prepared durable envelope requires an exact sparse V3 dispatch"
            )
        self.__dispatch = dispatch
        self.__sample_id = sample_id
        self.__payload = payload
        self.__content_sha256 = content_sha256
        # Retain the exact immutable constituent objects.  Identity checks are
        # constant-time for the 9-10 MB packed bytes and reject object.__setattr__
        # replacement without rehashing the already validated dispatch.
        self.__packed_events = dispatch.packed_events
        self.__world_offsets = dispatch.world_offsets
        self.__event_counts = dispatch.event_counts
        self.__final_next_agent_ids = dispatch.final_next_agent_ids
        self.__world_uids = dispatch.world_uids
        self.__tick_identity = (
            dispatch.start_next_absolute_tick,
            dispatch.end_next_absolute_tick,
            dispatch.tick_count,
        )
        self.__dispatch_content_sha256 = dispatch.content_sha256
        self.__artifact_sha256 = dispatch.artifact_sha256
        self.__manifest_sha256 = dispatch.manifest_sha256
        self.__schema_version = dispatch.schema_version
        self.__time_authority = dispatch.time_authority
        self.__wire_abi = dispatch.wire_abi

    @property
    def content_sha256(self) -> str:
        return self.__content_sha256

    @property
    def nbytes(self) -> int:
        return len(self.__payload)

    @property
    def sample_id(self) -> str:
        return self.__sample_id

    def verify_dispatch(
        self,
        dispatch: CudaFullEvolutionSparseEventDispatchV3 | None = None,
        *,
        sample_id: str | None = None,
    ) -> None:
        candidate = self.__dispatch if dispatch is None else dispatch
        if type(candidate) is not CudaFullEvolutionSparseEventDispatchV3:
            raise CudaFullEvolutionRuntimeError(
                "prepared durable envelope requires an exact sparse V3 dispatch"
            )
        if candidate is not self.__dispatch:
            raise CudaFullEvolutionRuntimeError(
                "prepared durable envelope is bound to another dispatch object"
            )
        if sample_id is not None and sample_id != self.__sample_id:
            raise CudaFullEvolutionRuntimeError(
                "prepared durable envelope sample identity mismatch"
            )
        if (
            candidate.packed_events is not self.__packed_events
            or candidate.world_offsets is not self.__world_offsets
            or candidate.event_counts is not self.__event_counts
            or candidate.final_next_agent_ids is not self.__final_next_agent_ids
            or candidate.world_uids is not self.__world_uids
            or (
                candidate.start_next_absolute_tick,
                candidate.end_next_absolute_tick,
                candidate.tick_count,
            )
            != self.__tick_identity
            or candidate.content_sha256 != self.__dispatch_content_sha256
            or candidate.artifact_sha256 != self.__artifact_sha256
            or candidate.manifest_sha256 != self.__manifest_sha256
            or candidate.schema_version != self.__schema_version
            or candidate.time_authority != self.__time_authority
            or candidate.wire_abi != self.__wire_abi
        ):
            raise CudaFullEvolutionRuntimeError(
                "prepared durable envelope dispatch changed after preparation"
            )

    def payload_for_publication(
        self, dispatch: CudaFullEvolutionSparseEventDispatchV3 | None = None
    ) -> bytes:
        self.verify_dispatch(dispatch, sample_id=self.__sample_id)
        return self.__payload

    def bound_dispatch(self) -> CudaFullEvolutionSparseEventDispatchV3:
        self.verify_dispatch(sample_id=self.__sample_id)
        return self.__dispatch


def _prepare_direct_sparse_durable_envelope_unchecked_v3(
    dispatch: CudaFullEvolutionSparseEventDispatchV3, *, sample_id: str
) -> PreparedDirectSparseDurableEnvelopeV3:
    checked_sample_id = _checked_sink_token(sample_id, "sample_id")
    payload = _encode_direct_sparse_durable_envelope_unchecked_v3(
        dispatch, sample_id=checked_sample_id
    )
    return PreparedDirectSparseDurableEnvelopeV3(
        dispatch=dispatch,
        sample_id=checked_sample_id,
        payload=payload,
        content_sha256=hashlib.sha256(payload).hexdigest(),
        _factory_token=_PREPARED_ENVELOPE_FACTORY_TOKEN_V3,
    )


def prepare_direct_sparse_durable_envelope_v3(
    dispatch: CudaFullEvolutionSparseEventDispatchV3, *, sample_id: str
) -> PreparedDirectSparseDurableEnvelopeV3:
    """Checked compatibility factory; the benchmark uses owner issuance."""
    _validate_direct_sparse_dispatch_v3(dispatch)
    return _prepare_direct_sparse_durable_envelope_unchecked_v3(
        dispatch, sample_id=sample_id
    )


def _read_exact_descriptor(descriptor: int, expected_nbytes: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < expected_nbytes:
        chunk = os.pread(descriptor, min(1 << 20, expected_nbytes - offset), offset)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
    if offset != expected_nbytes or os.pread(descriptor, 1, offset):
        raise CudaFullEvolutionRuntimeError(
            "durable envelope descriptor length changed during verification"
        )
    return b"".join(chunks)


def _sha256_exact_descriptor(descriptor: int, expected_nbytes: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < expected_nbytes:
        chunk = os.pread(descriptor, min(1 << 20, expected_nbytes - offset), offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    if offset != expected_nbytes or os.pread(descriptor, 1, offset):
        raise CudaFullEvolutionRuntimeError(
            "durable envelope descriptor length changed during verification"
        )
    return digest.hexdigest()


class VerifiedDirectSparseDurableEnvelopeV3:
    """Read-only descriptor held across receipt issuance and owner commit."""

    __slots__ = (
        "__closed",
        "__descriptor",
        "__dispatch",
        "__dispatch_content_sha256",
        "__identity",
        "__mode",
        "__nbytes",
        "__content_sha256",
        "__mtime_ns",
        "__ctime_ns",
        "__path",
        "__prepared",
        "__sample_id",
    )

    def __init__(
        self,
        *,
        descriptor: int,
        path: Path,
        identity: tuple[int, int],
        mode: int,
        nbytes: int,
        content_sha256: str,
        mtime_ns: int,
        ctime_ns: int,
        prepared: PreparedDirectSparseDurableEnvelopeV3,
        _factory_token: object,
    ) -> None:
        if _factory_token is not _VERIFIED_ENVELOPE_FACTORY_TOKEN_V3:
            raise CudaFullEvolutionRuntimeError(
                "verified durable envelope handles are factory-owned"
            )
        if type(prepared) is not PreparedDirectSparseDurableEnvelopeV3:
            raise CudaFullEvolutionRuntimeError(
                "verified durable envelope requires an exact prepared capability"
            )
        prepared.verify_dispatch(sample_id=prepared.sample_id)
        if nbytes != prepared.nbytes or content_sha256 != prepared.content_sha256:
            raise CudaFullEvolutionRuntimeError(
                "verified durable envelope byte identity differs from preparation"
            )
        self.__descriptor = descriptor
        self.__path = path
        self.__identity = identity
        self.__mode = mode
        self.__nbytes = nbytes
        self.__content_sha256 = content_sha256
        self.__mtime_ns = mtime_ns
        self.__ctime_ns = ctime_ns
        self.__prepared = prepared
        self.__dispatch = prepared.bound_dispatch()
        self.__dispatch_content_sha256 = self.__dispatch.content_sha256
        self.__sample_id = prepared.sample_id
        self.__closed = False

    def binding(self) -> dict[str, Any]:
        return self.verify()

    def _binding(self) -> dict[str, Any]:
        return {
            "durable_path": str(self.__path),
            "durable_device": self.__identity[0],
            "durable_inode": self.__identity[1],
            "durable_link_count": 1,
            "durable_mode": self.__mode,
            "durable_nbytes": self.__nbytes,
            "durable_content_sha256": self.__content_sha256,
            "durable_mtime_ns": self.__mtime_ns,
            "durable_ctime_ns": self.__ctime_ns,
        }

    def verify(
        self,
        *,
        dispatch: CudaFullEvolutionSparseEventDispatchV3 | None = None,
        dispatch_content_sha256: str | None = None,
        sample_id: str | None = None,
    ) -> dict[str, Any]:
        if self.__closed:
            raise CudaFullEvolutionRuntimeError("durable envelope handle is closed")
        self.__prepared.verify_dispatch(
            self.__dispatch if dispatch is None else dispatch,
            sample_id=self.__sample_id if sample_id is None else sample_id,
        )
        if dispatch is not None and dispatch is not self.__dispatch:
            raise CudaFullEvolutionRuntimeError(
                "durable envelope handle is bound to another dispatch object"
            )
        if (
            dispatch_content_sha256 is not None
            and dispatch_content_sha256 != self.__dispatch_content_sha256
        ):
            raise CudaFullEvolutionRuntimeError(
                "durable envelope handle dispatch identity mismatch"
            )
        if sample_id is not None and sample_id != self.__sample_id:
            raise CudaFullEvolutionRuntimeError(
                "durable envelope handle sample identity mismatch"
            )
        try:
            resolved_path = self.__path.resolve(strict=True)
            descriptor_state = os.fstat(self.__descriptor)
            path_state = os.lstat(self.__path)
        except OSError as exc:
            raise CudaFullEvolutionRuntimeError(
                f"durable envelope path is unavailable during verification: {exc}"
            ) from exc
        if (
            resolved_path != self.__path
            or not self.__path.is_absolute()
            or not stat.S_ISREG(descriptor_state.st_mode)
            or not stat.S_ISREG(path_state.st_mode)
            or (descriptor_state.st_dev, descriptor_state.st_ino) != self.__identity
            or (path_state.st_dev, path_state.st_ino) != self.__identity
            or descriptor_state.st_nlink != 1
            or path_state.st_nlink != 1
            or stat.S_IMODE(descriptor_state.st_mode) != self.__mode
            or stat.S_IMODE(path_state.st_mode) != self.__mode
            or descriptor_state.st_size != self.__nbytes
            or path_state.st_size != self.__nbytes
            or descriptor_state.st_mtime_ns != self.__mtime_ns
            or path_state.st_mtime_ns != self.__mtime_ns
            or descriptor_state.st_ctime_ns != self.__ctime_ns
            or path_state.st_ctime_ns != self.__ctime_ns
        ):
            raise CudaFullEvolutionRuntimeError(
                "durable envelope path no longer names the verified immutable inode"
            )
        return self._binding()

    def verify_durable_content(
        self,
        *,
        dispatch: CudaFullEvolutionSparseEventDispatchV3,
        sample_id: str,
    ) -> dict[str, Any]:
        """Independently rehash the held descriptor at the owner ACK seam."""
        self.verify(
            dispatch=dispatch,
            dispatch_content_sha256=dispatch.content_sha256,
            sample_id=sample_id,
        )
        actual_sha256 = _sha256_exact_descriptor(self.__descriptor, self.__nbytes)
        if actual_sha256 != self.__content_sha256:
            raise CudaFullEvolutionRuntimeError(
                "durable envelope descriptor content digest changed before ACK"
            )
        return self.verify(
            dispatch=dispatch,
            dispatch_content_sha256=dispatch.content_sha256,
            sample_id=sample_id,
        )

    def is_bound_to_prepared(
        self, prepared: PreparedDirectSparseDurableEnvelopeV3
    ) -> bool:
        return type(prepared) is PreparedDirectSparseDurableEnvelopeV3 and (
            prepared is self.__prepared
        )

    def close(self) -> None:
        if not self.__closed:
            os.close(self.__descriptor)
            self.__closed = True

    def __enter__(self) -> "VerifiedDirectSparseDurableEnvelopeV3":
        self.verify()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def open_verified_direct_sparse_durable_envelope_v3(
    dispatch: CudaFullEvolutionSparseEventDispatchV3,
    *,
    sample_id: str,
    durable_path: str | Path,
) -> VerifiedDirectSparseDurableEnvelopeV3:
    prepared = prepare_direct_sparse_durable_envelope_v3(dispatch, sample_id=sample_id)
    return open_verified_prepared_direct_sparse_durable_envelope_v3(
        prepared, durable_path=durable_path
    )


def open_verified_prepared_direct_sparse_durable_envelope_v3(
    prepared: PreparedDirectSparseDurableEnvelopeV3,
    *,
    durable_path: str | Path,
) -> VerifiedDirectSparseDurableEnvelopeV3:
    if type(prepared) is not PreparedDirectSparseDurableEnvelopeV3:
        raise CudaFullEvolutionRuntimeError(
            "durable envelope reopen requires an exact prepared capability"
        )
    prepared.verify_dispatch(sample_id=prepared.sample_id)
    path = Path(durable_path)
    if not path.is_absolute():
        raise CudaFullEvolutionRuntimeError("durable envelope path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CudaFullEvolutionRuntimeError(
            f"durable envelope path is unavailable: {exc}"
        ) from exc
    if path != resolved:
        raise CudaFullEvolutionRuntimeError(
            "durable envelope path must be canonical and symlink-free"
        )
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        descriptor_state = os.fstat(descriptor)
        path_state = os.lstat(path)
        identity = (descriptor_state.st_dev, descriptor_state.st_ino)
        mode = stat.S_IMODE(descriptor_state.st_mode)
        if (
            not stat.S_ISREG(descriptor_state.st_mode)
            or not stat.S_ISREG(path_state.st_mode)
            or (path_state.st_dev, path_state.st_ino) != identity
            or descriptor_state.st_nlink != 1
            or path_state.st_nlink != 1
            or mode != stat.S_IRUSR
            or stat.S_IMODE(path_state.st_mode) != stat.S_IRUSR
            or descriptor_state.st_size != prepared.nbytes
            or path_state.st_size != prepared.nbytes
        ):
            raise CudaFullEvolutionRuntimeError(
                "durable envelope immutable inode contract mismatch"
            )
        actual_sha256 = _sha256_exact_descriptor(descriptor, prepared.nbytes)
        descriptor_state = os.fstat(descriptor)
        path_state = os.lstat(path)
        if (
            actual_sha256 != prepared.content_sha256
            or path.resolve(strict=True) != path
            or (descriptor_state.st_dev, descriptor_state.st_ino) != identity
            or (path_state.st_dev, path_state.st_ino) != identity
            or descriptor_state.st_nlink != 1
            or path_state.st_nlink != 1
            or stat.S_IMODE(descriptor_state.st_mode) != stat.S_IRUSR
            or stat.S_IMODE(path_state.st_mode) != stat.S_IRUSR
            or descriptor_state.st_size != prepared.nbytes
            or path_state.st_size != prepared.nbytes
        ):
            raise CudaFullEvolutionRuntimeError(
                "durable envelope bytes or immutable inode contract mismatch"
            )
        return VerifiedDirectSparseDurableEnvelopeV3(
            descriptor=descriptor,
            path=path,
            identity=identity,
            mode=mode,
            nbytes=prepared.nbytes,
            content_sha256=prepared.content_sha256,
            mtime_ns=descriptor_state.st_mtime_ns,
            ctime_ns=descriptor_state.st_ctime_ns,
            prepared=prepared,
            _factory_token=_VERIFIED_ENVELOPE_FACTORY_TOKEN_V3,
        )
    except Exception:
        os.close(descriptor)
        raise


def seal_direct_sparse_durable_sink_receipt_v3(
    dispatch: CudaFullEvolutionSparseEventDispatchV3,
    *,
    sink_id: str,
    sink_receipt_id: str,
    sample_id: str,
    durable_envelope: VerifiedDirectSparseDurableEnvelopeV3,
) -> CudaFullEvolutionDirectSparseDurableSinkReceiptV3:
    if type(dispatch) is not CudaFullEvolutionSparseEventDispatchV3:
        raise CudaFullEvolutionRuntimeError(
            "durable sink receipt requires an exact sparse V3 dispatch"
        )
    if type(durable_envelope) is not VerifiedDirectSparseDurableEnvelopeV3:
        raise CudaFullEvolutionRuntimeError(
            "durable sink receipt requires an exact verified envelope handle"
        )
    checked_sink_id = _checked_sink_token(sink_id, "sink_id")
    checked_receipt_id = _checked_sink_token(sink_receipt_id, "sink_receipt_id")
    checked_sample_id = _checked_sink_token(sample_id, "sample_id")
    binding = durable_envelope.verify(
        dispatch=dispatch,
        dispatch_content_sha256=dispatch.content_sha256,
        sample_id=checked_sample_id,
    )
    provisional = CudaFullEvolutionDirectSparseDurableSinkReceiptV3(
        dispatch_content_sha256=dispatch.content_sha256,
        sink_id=checked_sink_id,
        sink_receipt_id=checked_receipt_id,
        sample_id=checked_sample_id,
        world_uids=dispatch.world_uids,
        start_next_absolute_tick=dispatch.start_next_absolute_tick,
        end_next_absolute_tick=dispatch.end_next_absolute_tick,
        tick_count=dispatch.tick_count,
        artifact_sha256=dispatch.artifact_sha256,
        manifest_sha256=dispatch.manifest_sha256,
        **binding,
        receipt_sha256="",
    )
    return replace(
        provisional,
        receipt_sha256=_canonical_sha256(_durable_sink_receipt_body_v3(provisional)),
    )


def validate_direct_sparse_durable_sink_receipt_v3(
    dispatch: CudaFullEvolutionSparseEventDispatchV3,
    receipt: CudaFullEvolutionDirectSparseDurableSinkReceiptV3,
    *,
    durable_envelope: VerifiedDirectSparseDurableEnvelopeV3,
) -> dict[str, Any]:
    """Revalidate the exact durable object immediately at the commit seam."""
    if type(dispatch) is not CudaFullEvolutionSparseEventDispatchV3:
        raise CudaFullEvolutionRuntimeError(
            "durable acknowledgement requires an exact sparse V3 dispatch"
        )
    if type(receipt) is not CudaFullEvolutionDirectSparseDurableSinkReceiptV3:
        raise CudaFullEvolutionRuntimeError(
            "acknowledgement requires an exact durable sparse V3 receipt"
        )
    if type(durable_envelope) is not VerifiedDirectSparseDurableEnvelopeV3:
        raise CudaFullEvolutionRuntimeError(
            "acknowledgement requires the exact verified envelope handle"
        )
    if (
        receipt.schema_version != DIRECT_SPARSE_DURABLE_SINK_RECEIPT_SCHEMA_V3
        or receipt.time_authority != CUDA_NATIVE_TIME_AUTHORITY_V1
        or receipt.wire_abi != CUDA_NATIVE_SPARSE_EVENT_WIRE_ABI_V3
        or receipt.envelope_schema_version != DIRECT_SPARSE_DURABLE_ENVELOPE_SCHEMA_V3
    ):
        raise CudaFullEvolutionRuntimeError("durable sparse receipt contract mismatch")
    _checked_sink_token(receipt.sink_id, "receipt sink_id")
    _checked_sink_token(receipt.sink_receipt_id, "receipt sink_receipt_id")
    checked_sample_id = _checked_sink_token(receipt.sample_id, "receipt sample_id")
    if not _valid_sha256(
        receipt.receipt_sha256
    ) or receipt.receipt_sha256 != _canonical_sha256(
        _durable_sink_receipt_body_v3(receipt)
    ):
        raise CudaFullEvolutionRuntimeError("durable sparse receipt seal mismatch")
    if (
        receipt.dispatch_content_sha256 != dispatch.content_sha256
        or receipt.world_uids != dispatch.world_uids
        or receipt.start_next_absolute_tick != dispatch.start_next_absolute_tick
        or receipt.end_next_absolute_tick != dispatch.end_next_absolute_tick
        or receipt.tick_count != dispatch.tick_count
        or receipt.artifact_sha256 != dispatch.artifact_sha256
        or receipt.manifest_sha256 != dispatch.manifest_sha256
    ):
        raise CudaFullEvolutionRuntimeError(
            "durable sparse receipt does not identify the pending dispatch"
        )
    binding = durable_envelope.verify_durable_content(
        dispatch=dispatch,
        sample_id=checked_sample_id,
    )
    if any(getattr(receipt, field) != value for field, value in binding.items()):
        raise CudaFullEvolutionRuntimeError(
            "durable sparse receipt does not match the open envelope handle"
        )
    if (
        receipt.durable_link_count != 1
        or receipt.durable_mode != stat.S_IRUSR
        or receipt.durable_nbytes != binding["durable_nbytes"]
        or receipt.durable_content_sha256 != binding["durable_content_sha256"]
    ):
        raise CudaFullEvolutionRuntimeError(
            "durable sparse receipt inode contract is invalid"
        )
    return binding


def _initial_capacity(temporal_k: int) -> int:
    requested = max(64, temporal_k * 8)
    return 1 << (requested - 1).bit_length()


def _checked_capacity(value: int, *, label: str) -> int:
    if (
        type(value) is not int
        or value < 1
        or value > DIRECT_SPARSE_MAX_CAPACITY_ROWS_V3
    ):
        raise CudaFullEvolutionRuntimeError(
            f"{label} must be an integer in [1,{DIRECT_SPARSE_MAX_CAPACITY_ROWS_V3}]"
        )
    return value


def _device_admission(
    torch: Any,
    *,
    device: Any,
    batch_size: int,
    capacity_rows: int,
    phase: str,
    compact_science_capture: bool = False,
    native_world_control: bool = False,
) -> dict[str, int]:
    if phase not in {"initial_owner", "replacement_staging", "packed_output"}:
        raise CudaFullEvolutionRuntimeError(
            f"unknown direct sparse VRAM admission phase: {phase}"
        )
    layout = state_layout_v2()
    state_bytes = batch_size * layout.state_bytes_per_world
    staging_bytes = batch_size * capacity_rows * CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3
    compact_metric_row_bytes = COMPACT_SCIENCE_AGENT_METRIC_WIDTH_V1 * 8
    compact_birth_provenance_row_bytes = (
        COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1
    )
    compact_persistent_bytes = (
        batch_size
        * RESIDENT_CAPACITY_C_V2
        * (compact_metric_row_bytes * 2 + compact_birth_provenance_row_bytes)
        if compact_science_capture
        else 0
    )
    compact_staging_bytes = (
        batch_size
        * capacity_rows
        * (compact_metric_row_bytes + compact_birth_provenance_row_bytes)
        if compact_science_capture
        else 0
    )
    world_control_bytes = (
        batch_size * 22 * 8 if native_world_control else 0
    )
    persistent_control_bytes = batch_size * 2 * 4
    pack_control_bytes = batch_size * 4 + (batch_size + 1) * 8
    control_bytes = persistent_control_bytes + pack_control_bytes
    owner_bytes = (
        2 * state_bytes
        + staging_bytes
        + persistent_control_bytes
        + compact_persistent_bytes
        + compact_staging_bytes
        + world_control_bytes
    )
    simultaneous_packed_bytes = staging_bytes + compact_staging_bytes
    estimated_peak_bytes = (
        owner_bytes
        + simultaneous_packed_bytes
        + pack_control_bytes
        + DIRECT_SPARSE_ADMISSION_MARGIN_BYTES_V3
    )
    if phase == "initial_owner":
        phase_required_free_bytes = estimated_peak_bytes
    elif phase == "replacement_staging":
        # CUDA's free-memory observation already excludes the live owner and
        # old staging. Keep that old staging until replacement allocation
        # succeeds and reserve the future simultaneous packed output too.
        phase_required_free_bytes = (
            staging_bytes
            + compact_staging_bytes
            + simultaneous_packed_bytes
            + pack_control_bytes
            + DIRECT_SPARSE_ADMISSION_MARGIN_BYTES_V3
        )
    else:
        # Persistent owner allocations are already excluded from free memory.
        phase_required_free_bytes = (
            simultaneous_packed_bytes
            + pack_control_bytes
            + DIRECT_SPARSE_ADMISSION_MARGIN_BYTES_V3
        )
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if phase_required_free_bytes > free_bytes:
        raise CudaFullEvolutionRuntimeError(
            f"direct sparse VRAM admission failed during {phase}: "
            "phase_required_free_bytes="
            f"{phase_required_free_bytes}, current_free_bytes={int(free_bytes)}"
        )
    return {
        "state_bytes": state_bytes,
        "rollback_state_bytes": state_bytes,
        "direct_sparse_staging_bytes": staging_bytes,
        "simultaneous_exact_packed_output_bytes": simultaneous_packed_bytes,
        "compact_science_persistent_bytes": compact_persistent_bytes,
        "compact_science_staging_bytes": compact_staging_bytes,
        "native_world_control_bytes": world_control_bytes,
        "control_bytes": control_bytes,
        "persistent_control_bytes": persistent_control_bytes,
        "pack_control_bytes": pack_control_bytes,
        "admission_margin_bytes": DIRECT_SPARSE_ADMISSION_MARGIN_BYTES_V3,
        "required_free_bytes": estimated_peak_bytes,
        "phase_required_free_bytes": phase_required_free_bytes,
        "observed_free_bytes": int(free_bytes),
        "total_device_bytes": int(total_bytes),
        "horizon_dependent_resident_bytes": 0,
    }


class CudaFullEvolutionDirectSparseOwnerV3:
    """Own an in-place transition as a rollback-capable sparse transaction."""

    def __init__(
        self,
        *,
        contract: CudaFullEvolutionNativeContractV2,
        loaded: LoadedCudaFullEvolutionNativeV2,
        state: Any,
        rollback_state: Any,
        direct_events: Any,
        event_counts: Any,
        advance_validation: Any,
        initial_capacity_rows: int,
        maximum_capacity_rows: int,
        initial_next_absolute_tick: int,
        advanced_ticks: int,
        world_uids: tuple[int, ...],
        rng_keys: tuple[int, ...],
        last_sink_receipt_sha256: str | None,
        population_fixture_id: str | None,
        admission: Mapping[str, int],
        capability_mask: int = FULL_CAPABILITY_MASK_V2,
        native_control_digest: str | None = None,
        native_control_vector: Mapping[str, Any] | None = None,
        native_world_controls: Sequence[Mapping[str, Any]] = (),
        science_world_control_table: Any | None = None,
        science_founder_application: Mapping[str, Any] | None = None,
        compact_science_capture: bool = False,
        science_agent_metrics: Any | None = None,
        science_agent_metrics_rollback: Any | None = None,
        science_event_metrics: Any | None = None,
        science_agent_birth_provenance: Any | None = None,
        science_event_birth_provenance: Any | None = None,
    ) -> None:
        self.contract = validate_cuda_full_evolution_contract_v2(contract)
        self.__loaded = loaded
        self.__state = state
        self.__rollback_state = rollback_state
        self.__direct_events = direct_events
        self.__event_counts = event_counts
        self.__advance_validation = advance_validation
        self.__capacity_rows = _checked_capacity(
            initial_capacity_rows, label="initial_capacity_rows"
        )
        self.__maximum_capacity_rows = _checked_capacity(
            maximum_capacity_rows, label="maximum_capacity_rows"
        )
        if self.__capacity_rows > self.__maximum_capacity_rows:
            raise CudaFullEvolutionRuntimeError(
                "initial direct sparse capacity exceeds maximum capacity"
            )
        self.__initial_next_absolute_tick = initial_next_absolute_tick
        self.__advanced_ticks = advanced_ticks
        self.__world_uids = world_uids
        self.__rng_keys = rng_keys
        self.__last_sink_receipt_sha256 = last_sink_receipt_sha256
        self.__population_fixture_id = population_fixture_id
        self.__admission = dict(admission)
        if (
            type(capability_mask) is not int
            or capability_mask < 0
            or capability_mask > FULL_CAPABILITY_MASK_V2
        ):
            raise CudaFullEvolutionRuntimeError(
                "native capability mask must fit the CUDA V2 scalar mask"
            )
        if (native_control_digest is None) != (native_control_vector is None):
            raise CudaFullEvolutionRuntimeError(
                "native control digest and vector must be supplied together"
            )
        if capability_mask != FULL_CAPABILITY_MASK_V2 and native_control_vector is None:
            raise CudaFullEvolutionRuntimeError(
                "a non-full native capability mask requires an exact control receipt"
            )
        if native_control_digest is not None and not _valid_sha256(
            native_control_digest
        ):
            raise CudaFullEvolutionRuntimeError(
                "native control digest must be lowercase SHA-256"
            )
        if native_control_vector is not None:
            if not isinstance(native_control_vector, Mapping):
                raise CudaFullEvolutionRuntimeError(
                    "native control vector must be an object"
                )
            if set(native_control_vector) != {"mutation.enabled"} or type(
                native_control_vector["mutation.enabled"]
            ) is not bool:
                raise CudaFullEvolutionRuntimeError(
                    "native control vector is not the closed mutation contract"
                )
            expected_mask = (
                FULL_CAPABILITY_MASK_V2
                if native_control_vector["mutation.enabled"]
                else FULL_CAPABILITY_MASK_V2 & ~(1 << 5)
            )
            if capability_mask != expected_mask:
                raise CudaFullEvolutionRuntimeError(
                    "native control mask differs from mutation.enabled"
                )
            expected_digest = _canonical_sha256(
                {
                    "schema_version": "eiel_science_cuda_v2_native_control_v1",
                    "capability_mask": capability_mask,
                    "control_vector": dict(native_control_vector),
                }
            )
            if native_control_digest != expected_digest:
                raise CudaFullEvolutionRuntimeError(
                    "native control digest does not match the applied vector"
                )
        self.__capability_mask = capability_mask
        self.__native_control_digest = native_control_digest
        self.__native_control_vector = (
            None
            if native_control_vector is None
            else dict(native_control_vector)
        )
        self.__native_world_projection: (
            ScienceNativeWorldRuntimeProjectionV1 | None
        ) = None
        self.__initial_live_counts = (INITIAL_POPULATION_V2,) * self.contract.B
        self.__science_world_control_table = None
        self.__science_founder_application = None
        self.__compact_science_capture = compact_science_capture
        self.__science_agent_metrics = science_agent_metrics
        self.__science_agent_metrics_rollback = science_agent_metrics_rollback
        self.__science_event_metrics = science_event_metrics
        self.__science_agent_birth_provenance = science_agent_birth_provenance
        self.__science_event_birth_provenance = science_event_birth_provenance
        if type(compact_science_capture) is not bool:
            raise CudaFullEvolutionRuntimeError(
                "compact science capture selector must be boolean"
            )
        if native_world_controls:
            if native_control_vector is not None or capability_mask != FULL_CAPABILITY_MASK_V2:
                raise CudaFullEvolutionRuntimeError(
                    "generic world controls cannot be combined with the legacy scalar control"
                )
            try:
                projection = ScienceNativeWorldRuntimeProjectionV1.from_controls(
                    native_world_controls
                )
            except ScienceNativeWorldRuntimeError as exc:
                raise CudaFullEvolutionRuntimeError(
                    f"native world-control projection is invalid: {exc}"
                ) from exc
            if len(projection.worlds) != self.contract.B:
                raise CudaFullEvolutionRuntimeError(
                    "native world-control count differs from physical B"
                )
            if science_world_control_table is None:
                raise CudaFullEvolutionRuntimeError(
                    "native world-control device table is absent"
                )
            expected_rows = projection.rows_at_tick(
                initial_next_absolute_tick + advanced_ticks
            )
            if (
                tuple(science_world_control_table.shape)
                != (self.contract.B, NATIVE_WORLD_CONTROL_WIDTH_V1)
                or science_world_control_table.dtype != __import__("torch").int64
                or not science_world_control_table.is_cuda
                or not science_world_control_table.is_contiguous()
                or tuple(tuple(int(item) for item in row) for row in (
                    science_world_control_table.detach().cpu().tolist()
                )) != expected_rows
            ):
                raise CudaFullEvolutionRuntimeError(
                    "native world-control device table differs from its resolved projection"
                )
            if not isinstance(science_founder_application, Mapping):
                raise CudaFullEvolutionRuntimeError(
                    "native founder binding is absent"
                )
            if (
                science_founder_application.get("registry_sha256")
                != projection.founder_registry_sha256
                or science_founder_application.get("control_set_sha256")
                != projection.control_set_sha256
            ):
                raise CudaFullEvolutionRuntimeError(
                    "native founder binding differs from the world-control set"
                )
            self.__native_world_projection = projection
            self.__initial_live_counts = tuple(
                int(world.payload["initialization"]["initial_live_count"])
                for world in projection.worlds
            )
            self.__science_world_control_table = science_world_control_table
            self.__science_founder_application = dict(science_founder_application)
        elif science_world_control_table is not None or science_founder_application is not None:
            raise CudaFullEvolutionRuntimeError(
                "unbound native world-control runtime state is present"
            )
        if compact_science_capture:
            if self.__native_world_projection is None:
                raise CudaFullEvolutionRuntimeError(
                    "compact science capture requires generic native-world controls"
                )
            if any(
                value is None
                for value in (
                    science_agent_metrics,
                    science_agent_metrics_rollback,
                    science_event_metrics,
                    science_agent_birth_provenance,
                    science_event_birth_provenance,
                )
            ):
                raise CudaFullEvolutionRuntimeError(
                    "compact science capture tensors are incomplete"
                )
        elif any(
            value is not None
            for value in (
                science_agent_metrics,
                science_agent_metrics_rollback,
                science_event_metrics,
                science_agent_birth_provenance,
                science_event_birth_provenance,
            )
        ):
            raise CudaFullEvolutionRuntimeError(
                "unbound compact science capture tensors are present"
            )
        self.__pending: _PendingDirectSparseDispatchV3 | None = None
        self.__pending_prepared_envelope: (
            PreparedDirectSparseDurableEnvelopeV3 | None
        ) = None
        self.__last_metrics: DirectSparseDispatchMetricsV3 | None = None
        self.__last_execution_receipt: dict[str, Any] | None = None
        self.__poisoned_reason: str | None = None
        self.__closed = False
        self._validate_storage()
        self._validate_frontier(
            expected_next_tick=self.committed_next_absolute_tick,
            expected_ticks_completed=self.__advanced_ticks,
        )

    @classmethod
    def initialize(
        cls,
        contract: CudaFullEvolutionNativeContractV2 | Mapping[str, Any],
        *,
        seed: int,
        artifact_path: str | Path,
        device: str | Any = "cuda",
        world_uids: Sequence[int] | None = None,
        first_world_uid: int = 1,
        initial_capacity_rows: int | None = None,
        maximum_capacity_rows: int = DIRECT_SPARSE_MAX_CAPACITY_ROWS_V3,
        capability_mask: int = FULL_CAPABILITY_MASK_V2,
        native_control_digest: str | None = None,
        native_control_vector: Mapping[str, Any] | None = None,
        native_world_controls: Sequence[Mapping[str, Any]] = (),
        compact_science_capture: bool = False,
    ) -> "CudaFullEvolutionDirectSparseOwnerV3":
        contract = validate_cuda_full_evolution_contract_v2(contract)
        if type(seed) is not int or not 0 <= seed <= MAX_SIGNED_INT64:
            raise CudaFullEvolutionRuntimeError("seed must be an integer in [0,2**63)")
        if world_uids is None:
            if type(first_world_uid) is not int or first_world_uid < 1:
                raise CudaFullEvolutionRuntimeError("first_world_uid must be positive")
            uids = tuple(first_world_uid + index for index in range(contract.B))
        else:
            uids = tuple(world_uids)
        if (
            len(uids) != contract.B
            or len(set(uids)) != contract.B
            or any(
                type(uid) is not int or not 0 < uid <= MAX_SIGNED_INT64 for uid in uids
            )
        ):
            raise CudaFullEvolutionRuntimeError(
                "world_uids must contain B unique positive signed-int64 identities"
            )
        capacity = _checked_capacity(
            (
                _initial_capacity(contract.temporal_k)
                if initial_capacity_rows is None
                else initial_capacity_rows
            ),
            label="initial_capacity_rows",
        )
        maximum = _checked_capacity(
            maximum_capacity_rows, label="maximum_capacity_rows"
        )
        if capacity > maximum:
            raise CudaFullEvolutionRuntimeError(
                "initial direct sparse capacity exceeds maximum capacity"
            )
        import torch

        selected_device = torch.device(device)
        if selected_device.type != "cuda":
            raise CudaFullEvolutionRuntimeError(
                "direct sparse owner requires a CUDA device"
            )
        loaded = load_prebuilt_cuda_full_evolution_v2(
            artifact_path, device=selected_device
        )
        try:
            native_world_projection = (
                ScienceNativeWorldRuntimeProjectionV1.from_controls(
                    native_world_controls
                )
                if native_world_controls
                else None
            )
        except ScienceNativeWorldRuntimeError as exc:
            raise CudaFullEvolutionRuntimeError(
                f"native world-control projection is invalid: {exc}"
            ) from exc
        if native_world_projection is not None and len(
            native_world_projection.worlds
        ) != contract.B:
            raise CudaFullEvolutionRuntimeError(
                "native world-control count differs from physical B"
            )
        admission = _device_admission(
            torch,
            device=selected_device,
            batch_size=contract.B,
            capacity_rows=capacity,
            phase="initial_owner",
            compact_science_capture=compact_science_capture,
            native_world_control=native_world_projection is not None,
        )
        layout = state_layout_v2()
        state = torch.empty(
            (contract.B, layout.state_bytes_per_world),
            dtype=torch.uint8,
            device=selected_device,
        )
        uid_tensor = torch.tensor(uids, dtype=torch.int64, device=selected_device)
        science_world_control_table = None
        science_founder_application = None
        if native_world_projection is None:
            loaded.module.initialize(state, uid_tensor, seed, contract.start_next_tick)
            rng_keys = (seed,) * contract.B
        else:
            science_world_control_table = torch.tensor(
                native_world_projection.rows_at_tick(contract.start_next_tick),
                dtype=torch.int64,
                device=selected_device,
            )
            loaded.module.initialize_science_worlds_v1(
                state,
                uid_tensor,
                science_world_control_table,
                contract.start_next_tick,
            )
            science_founder_application = (
                native_world_projection.apply_founders_to_state(state)
            )
            rng_keys = tuple(
                int(row[21])
                for row in native_world_projection.rows_at_tick(
                    contract.start_next_tick
                )
            )
        rollback_state = torch.empty_like(state)
        direct_events = torch.empty(
            (contract.B, capacity, CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3),
            dtype=torch.uint8,
            device=selected_device,
        )
        event_counts = torch.zeros(
            contract.B, dtype=torch.int32, device=selected_device
        )
        advance_validation = torch.zeros_like(event_counts)
        science_agent_metrics = None
        science_agent_metrics_rollback = None
        science_event_metrics = None
        science_agent_birth_provenance = None
        science_event_birth_provenance = None
        if compact_science_capture:
            science_agent_metrics = torch.zeros(
                (
                    contract.B,
                    RESIDENT_CAPACITY_C_V2,
                    COMPACT_SCIENCE_AGENT_METRIC_WIDTH_V1,
                ),
                dtype=torch.int64,
                device=selected_device,
            )
            science_agent_metrics_rollback = torch.empty_like(
                science_agent_metrics
            )
            science_event_metrics = torch.empty(
                (
                    contract.B,
                    capacity,
                    COMPACT_SCIENCE_AGENT_METRIC_WIDTH_V1,
                ),
                dtype=torch.int64,
                device=selected_device,
            )
            science_agent_birth_provenance = torch.empty(
                (
                    contract.B,
                    RESIDENT_CAPACITY_C_V2,
                    COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1,
                ),
                dtype=torch.uint8,
                device=selected_device,
            )
            science_event_birth_provenance = torch.empty(
                (
                    contract.B,
                    capacity,
                    COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1,
                ),
                dtype=torch.uint8,
                device=selected_device,
            )
        return cls(
            contract=contract,
            loaded=loaded,
            state=state,
            rollback_state=rollback_state,
            direct_events=direct_events,
            event_counts=event_counts,
            advance_validation=advance_validation,
            initial_capacity_rows=capacity,
            maximum_capacity_rows=maximum,
            initial_next_absolute_tick=contract.start_next_tick,
            advanced_ticks=0,
            world_uids=uids,
            rng_keys=rng_keys,
            last_sink_receipt_sha256=None,
            population_fixture_id=None,
            admission=admission,
            capability_mask=capability_mask,
            native_control_digest=native_control_digest,
            native_control_vector=native_control_vector,
            native_world_controls=native_world_controls,
            science_world_control_table=science_world_control_table,
            science_founder_application=science_founder_application,
            compact_science_capture=compact_science_capture,
            science_agent_metrics=science_agent_metrics,
            science_agent_metrics_rollback=science_agent_metrics_rollback,
            science_event_metrics=science_event_metrics,
            science_agent_birth_provenance=science_agent_birth_provenance,
            science_event_birth_provenance=science_event_birth_provenance,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: CudaFullEvolutionCheckpointV2,
        *,
        artifact_path: str | Path | None = None,
        device: str | Any = "cuda",
        initial_capacity_rows: int | None = None,
        maximum_capacity_rows: int = DIRECT_SPARSE_MAX_CAPACITY_ROWS_V3,
        capability_mask: int = FULL_CAPABILITY_MASK_V2,
        native_control_digest: str | None = None,
        native_control_vector: Mapping[str, Any] | None = None,
        native_world_controls: Sequence[Mapping[str, Any]] = (),
        compact_science_capture: bool = False,
        compact_science_agent_metrics: Any | None = None,
    ) -> "CudaFullEvolutionDirectSparseOwnerV3":
        contract = _validate_checkpoint_envelope(checkpoint)
        capacity = _checked_capacity(
            (
                _initial_capacity(contract.temporal_k)
                if initial_capacity_rows is None
                else initial_capacity_rows
            ),
            label="initial_capacity_rows",
        )
        maximum = _checked_capacity(
            maximum_capacity_rows, label="maximum_capacity_rows"
        )
        if capacity > maximum:
            raise CudaFullEvolutionRuntimeError(
                "initial direct sparse capacity exceeds maximum capacity"
            )
        import torch

        selected_device = torch.device(device)
        if selected_device.type != "cuda":
            raise CudaFullEvolutionRuntimeError(
                "direct sparse owner requires a CUDA device"
            )
        loaded = load_prebuilt_cuda_full_evolution_v2(
            artifact_path or checkpoint.artifact_path, device=selected_device
        )
        if loaded.artifact_sha256 != checkpoint.artifact_sha256:
            raise CudaFullEvolutionRuntimeError("checkpoint artifact hash mismatch")
        if loaded.manifest_sha256 != checkpoint.manifest_sha256:
            raise CudaFullEvolutionRuntimeError("checkpoint manifest hash mismatch")
        if dict(loaded.device_identity) != dict(checkpoint.device_identity):
            raise CudaFullEvolutionRuntimeError(
                "checkpoint selected-device identity mismatch"
            )
        admission = _device_admission(
            torch,
            device=selected_device,
            batch_size=contract.B,
            capacity_rows=capacity,
            phase="initial_owner",
            compact_science_capture=compact_science_capture,
            native_world_control=bool(native_world_controls),
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
        try:
            native_world_projection = (
                ScienceNativeWorldRuntimeProjectionV1.from_controls(
                    native_world_controls
                )
                if native_world_controls
                else None
            )
        except ScienceNativeWorldRuntimeError as exc:
            raise CudaFullEvolutionRuntimeError(
                f"native world-control projection is invalid: {exc}"
            ) from exc
        science_world_control_table = None
        science_founder_application = None
        if native_world_projection is not None:
            if len(native_world_projection.worlds) != contract.B:
                raise CudaFullEvolutionRuntimeError(
                    "native world-control count differs from physical B"
                )
            current_tick = (
                checkpoint.initial_next_absolute_tick + checkpoint.advanced_ticks
            )
            rows = native_world_projection.rows_at_tick(current_tick)
            expected_rng_keys = tuple(int(row[21]) for row in rows)
            if checkpoint.rng_keys != expected_rng_keys:
                raise CudaFullEvolutionRuntimeError(
                    "checkpoint RNG bindings differ from native world controls"
                )
            science_world_control_table = torch.tensor(
                rows, dtype=torch.int64, device=selected_device
            )
            science_founder_application = (
                native_world_projection.founder_binding_v1()
            )
        rollback_state = torch.empty_like(state)
        direct_events = torch.empty(
            (contract.B, capacity, CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3),
            dtype=torch.uint8,
            device=selected_device,
        )
        event_counts = torch.zeros(
            contract.B, dtype=torch.int32, device=selected_device
        )
        advance_validation = torch.zeros_like(event_counts)
        science_agent_metrics = None
        science_agent_metrics_rollback = None
        science_event_metrics = None
        science_agent_birth_provenance = None
        science_event_birth_provenance = None
        if compact_science_capture:
            expected_shape = (
                contract.B,
                RESIDENT_CAPACITY_C_V2,
                COMPACT_SCIENCE_AGENT_METRIC_WIDTH_V1,
            )
            if compact_science_agent_metrics is None:
                raise CudaFullEvolutionRuntimeError(
                    "compact science restore lacks active-agent metrics"
                )
            science_agent_metrics = torch.as_tensor(
                compact_science_agent_metrics,
                dtype=torch.int64,
                device=selected_device,
            ).contiguous()
            if tuple(science_agent_metrics.shape) != expected_shape:
                raise CudaFullEvolutionRuntimeError(
                    "compact science restore metrics have the wrong shape"
                )
            science_agent_metrics_rollback = torch.empty_like(
                science_agent_metrics
            )
            science_event_metrics = torch.empty(
                (
                    contract.B,
                    capacity,
                    COMPACT_SCIENCE_AGENT_METRIC_WIDTH_V1,
                ),
                dtype=torch.int64,
                device=selected_device,
            )
            science_agent_birth_provenance = torch.empty(
                (
                    contract.B,
                    RESIDENT_CAPACITY_C_V2,
                    COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1,
                ),
                dtype=torch.uint8,
                device=selected_device,
            )
            science_event_birth_provenance = torch.empty(
                (
                    contract.B,
                    capacity,
                    COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1,
                ),
                dtype=torch.uint8,
                device=selected_device,
            )
        return cls(
            contract=contract,
            loaded=loaded,
            state=state,
            rollback_state=rollback_state,
            direct_events=direct_events,
            event_counts=event_counts,
            advance_validation=advance_validation,
            initial_capacity_rows=capacity,
            maximum_capacity_rows=maximum,
            initial_next_absolute_tick=checkpoint.initial_next_absolute_tick,
            advanced_ticks=checkpoint.advanced_ticks,
            world_uids=checkpoint.world_uids,
            rng_keys=checkpoint.rng_keys,
            last_sink_receipt_sha256=checkpoint.last_sink_receipt_sha256,
            population_fixture_id=checkpoint.population_fixture_id,
            admission=admission,
            capability_mask=capability_mask,
            native_control_digest=native_control_digest,
            native_control_vector=native_control_vector,
            native_world_controls=native_world_controls,
            science_world_control_table=science_world_control_table,
            science_founder_application=science_founder_application,
            compact_science_capture=compact_science_capture,
            science_agent_metrics=science_agent_metrics,
            science_agent_metrics_rollback=science_agent_metrics_rollback,
            science_event_metrics=science_event_metrics,
            science_agent_birth_provenance=science_agent_birth_provenance,
            science_event_birth_provenance=science_event_birth_provenance,
        )

    @property
    def advanced_ticks(self) -> int:
        return self.__advanced_ticks

    @property
    def committed_next_absolute_tick(self) -> int:
        return self.__initial_next_absolute_tick + self.__advanced_ticks

    @property
    def remaining_ticks(self) -> int:
        return self.contract.total_ticks - self.__advanced_ticks

    @property
    def has_pending_dispatch(self) -> bool:
        return self.__pending is not None

    @property
    def selected_capacity_rows_per_world(self) -> int:
        return self.__capacity_rows

    @property
    def dispatch_metrics_v3(self) -> DirectSparseDispatchMetricsV3 | None:
        return self.__last_metrics

    @property
    def resource_admission_v3(self) -> Mapping[str, int]:
        return dict(self.__admission)

    @property
    def next_native_world_control_boundary_v1(self) -> int | None:
        if self.__native_world_projection is None:
            return None
        return self.__native_world_projection.next_boundary_after(
            self.committed_next_absolute_tick
        )

    @property
    def native_world_control_binding_v1(self) -> Mapping[str, Any] | None:
        if self.__native_world_projection is None:
            return None
        return {
            "schema_version": "eiel_science_native_world_runtime_binding_v1",
            "control_set_sha256": self.__native_world_projection.control_set_sha256,
            "founder_registry_sha256": (
                self.__native_world_projection.founder_registry_sha256
            ),
            "founder_application": dict(self.__science_founder_application),
        }

    def _ensure_usable(self, operation: str) -> None:
        if self.__closed:
            raise CudaFullEvolutionRuntimeError(f"{operation} on closed owner")
        if self.__poisoned_reason is not None:
            raise CudaFullEvolutionRuntimeError(
                f"{operation} on poisoned owner: {self.__poisoned_reason}"
            )

    def _ensure_no_pending(self, operation: str) -> None:
        self._ensure_usable(operation)
        if self.__pending is not None:
            raise CudaFullEvolutionRuntimeError(
                f"{operation} requires durable sparse sink acknowledgement"
            )

    def _poison(self, reason: str) -> None:
        self.__poisoned_reason = reason
        raise CudaFullEvolutionRuntimeError(
            f"direct sparse owner poisoned with staging preserved: {reason}"
        )

    def _restore_snapshot_or_poison(self, reason: str) -> None:
        import torch

        try:
            self.__state.copy_(self.__rollback_state)
            if self.__compact_science_capture:
                self.__science_agent_metrics.copy_(
                    self.__science_agent_metrics_rollback
                )
            torch.cuda.synchronize(self.__state.device)
        except Exception as rollback_error:
            combined = f"{reason}; full-state rollback failed: {rollback_error}"
            self.__poisoned_reason = combined
            raise CudaFullEvolutionRuntimeError(
                f"direct sparse owner poisoned after rollback failure: {combined}"
            ) from rollback_error

    def _restore_snapshot_and_poison(self, reason: str) -> None:
        self._restore_snapshot_or_poison(reason)
        self._poison(reason)

    def _validate_storage(self) -> None:
        import torch

        runtime_receipt = self.__loaded.manifest.get("runtime")
        if (
            not isinstance(runtime_receipt, Mapping)
            or runtime_receipt.get("production_owner")
            != "CudaFullEvolutionDirectSparseOwnerV3"
            or runtime_receipt.get("event_frontier_commit")
            != DIRECT_SPARSE_EVENT_FRONTIER_COMMIT_V3
            or runtime_receipt.get("horizon_dependent_resident_bytes") != 0
        ):
            raise CudaFullEvolutionRuntimeError(
                "native manifest does not bind the direct sparse owner authority"
            )
        layout = state_layout_v2()
        expected_state = (self.contract.B, layout.state_bytes_per_world)
        if (
            self.__state.dtype != torch.uint8
            or not self.__state.is_cuda
            or not self.__state.is_contiguous()
            or tuple(self.__state.shape) != expected_state
            or self.__rollback_state.dtype != torch.uint8
            or not self.__rollback_state.is_cuda
            or not self.__rollback_state.is_contiguous()
            or tuple(self.__rollback_state.shape) != expected_state
            or self.__rollback_state.device != self.__state.device
            or self.__direct_events.dtype != torch.uint8
            or not self.__direct_events.is_cuda
            or not self.__direct_events.is_contiguous()
            or tuple(self.__direct_events.shape)
            != (
                self.contract.B,
                self.__capacity_rows,
                CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3,
            )
            or self.__direct_events.device != self.__state.device
            or self.__event_counts.dtype != torch.int32
            or not self.__event_counts.is_cuda
            or not self.__event_counts.is_contiguous()
            or tuple(self.__event_counts.shape) != (self.contract.B,)
            or self.__event_counts.device != self.__state.device
            or self.__advance_validation.dtype != torch.int32
            or not self.__advance_validation.is_cuda
            or not self.__advance_validation.is_contiguous()
            or tuple(self.__advance_validation.shape) != (self.contract.B,)
            or self.__advance_validation.device != self.__state.device
            or (
                self.__native_world_projection is not None
                and (
                    self.__science_world_control_table.dtype != torch.int64
                    or not self.__science_world_control_table.is_cuda
                    or not self.__science_world_control_table.is_contiguous()
                    or tuple(self.__science_world_control_table.shape)
                    != (self.contract.B, NATIVE_WORLD_CONTROL_WIDTH_V1)
                    or self.__science_world_control_table.device != self.__state.device
                )
            )
            or (
                self.__compact_science_capture
                and (
                    self.__science_agent_metrics.dtype != torch.int64
                    or not self.__science_agent_metrics.is_cuda
                    or not self.__science_agent_metrics.is_contiguous()
                    or tuple(self.__science_agent_metrics.shape)
                    != (
                        self.contract.B,
                        RESIDENT_CAPACITY_C_V2,
                        COMPACT_SCIENCE_AGENT_METRIC_WIDTH_V1,
                    )
                    or self.__science_agent_metrics.device != self.__state.device
                    or self.__science_agent_metrics_rollback.dtype != torch.int64
                    or not self.__science_agent_metrics_rollback.is_cuda
                    or not self.__science_agent_metrics_rollback.is_contiguous()
                    or tuple(self.__science_agent_metrics_rollback.shape)
                    != tuple(self.__science_agent_metrics.shape)
                    or self.__science_agent_metrics_rollback.device
                    != self.__state.device
                    or self.__science_event_metrics.dtype != torch.int64
                    or not self.__science_event_metrics.is_cuda
                    or not self.__science_event_metrics.is_contiguous()
                    or tuple(self.__science_event_metrics.shape)
                    != (
                        self.contract.B,
                        self.__capacity_rows,
                        COMPACT_SCIENCE_AGENT_METRIC_WIDTH_V1,
                    )
                    or self.__science_event_metrics.device != self.__state.device
                    or self.__science_agent_birth_provenance.dtype != torch.uint8
                    or not self.__science_agent_birth_provenance.is_cuda
                    or not self.__science_agent_birth_provenance.is_contiguous()
                    or tuple(self.__science_agent_birth_provenance.shape)
                    != (
                        self.contract.B,
                        RESIDENT_CAPACITY_C_V2,
                        COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1,
                    )
                    or self.__science_agent_birth_provenance.device
                    != self.__state.device
                    or self.__science_event_birth_provenance.dtype != torch.uint8
                    or not self.__science_event_birth_provenance.is_cuda
                    or not self.__science_event_birth_provenance.is_contiguous()
                    or tuple(self.__science_event_birth_provenance.shape)
                    != (
                        self.contract.B,
                        self.__capacity_rows,
                        COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1,
                    )
                    or self.__science_event_birth_provenance.device
                    != self.__state.device
                )
            )
        ):
            raise CudaFullEvolutionRuntimeError(
                "direct sparse owner storage shape/type/device mismatch"
            )

    def _header_rows(self) -> list[list[int]]:
        import torch

        header_bytes = state_layout_v2().field("scalar_header_counters").bytes
        return self.__state[:, :header_bytes].view(torch.int64).cpu().tolist()

    def _validate_frontier(
        self, *, expected_next_tick: int, expected_ticks_completed: int
    ) -> list[list[int]]:
        rows = self._header_rows()
        counter_bound = (
            MAX_CAPABILITY_COUNTER_INCREMENT_PER_TICK_V2 * expected_ticks_completed
        )
        fixture_delta = (
            RESIDENT_CAPACITY_C_V2 - INITIAL_POPULATION_V2
            if self.__population_fixture_id == FULL_CAPACITY_BENCHMARK_FIXTURE_V2
            else 0
        )
        for world, raw in enumerate(rows):
            values = tuple(int(value) for value in raw)
            if len(values) != 24 or any(value < 0 for value in values):
                self._poison(f"world {world} header extent or signed range invalid")
            if (
                values[0] != self.__world_uids[world]
                or values[1] != expected_next_tick
                or values[4] != 0
                or values[5] != expected_ticks_completed
                or values[13] != self.__rng_keys[world]
                or values[14] != RNG_SEMANTIC_EPOCH_V3
                or values[22] != 0
                or values[23] != 0
            ):
                self._poison(f"world {world} immutable frontier identity mismatch")
            if any(value > counter_bound for value in values[6:13]):
                self._poison(f"world {world} capability counter exceeds safe bound")
            births, deaths, mutations, recombinations = values[15:19]
            if (
                births > RESIDENT_CAPACITY_C_V2 * expected_ticks_completed
                or deaths > RESIDENT_CAPACITY_C_V2 * expected_ticks_completed
                or mutations > 256 * births
                or recombinations > 16 * births
                or values[10] != births
                or values[11] != mutations
                or values[12] != recombinations
            ):
                self._poison(f"world {world} lifecycle counters are inconsistent")
            initial_live = self.__initial_live_counts[world]
            expected_active = initial_live + fixture_delta + births - deaths
            if not 0 <= expected_active <= RESIDENT_CAPACITY_C_V2:
                self._poison(f"world {world} active population is impossible")
            if values[3] != expected_active:
                self._poison(f"world {world} active population mismatch")
            if values[2] != initial_live + 1 + fixture_delta + births:
                self._poison(f"world {world} next agent ID mismatch")
            if any(value > 1024 for value in values[19:22]):
                self._poison(f"world {world} ecological summary is invalid")
        return rows

    def _grow_capacity(self, required_rows: int) -> None:
        import torch

        next_capacity = max(required_rows, self.__capacity_rows * 2)
        next_capacity = 1 << (next_capacity - 1).bit_length()
        if next_capacity > self.__maximum_capacity_rows:
            self._poison(
                "direct sparse overflow exceeds the configured maximum capacity"
            )
        torch.cuda.empty_cache()
        try:
            admission = _device_admission(
                torch,
                device=self.__state.device,
                batch_size=self.contract.B,
                capacity_rows=next_capacity,
                phase="replacement_staging",
                compact_science_capture=self.__compact_science_capture,
                native_world_control=self.__native_world_projection is not None,
            )
            replacement = torch.empty(
                (
                    self.contract.B,
                    next_capacity,
                    CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3,
                ),
                dtype=torch.uint8,
                device=self.__state.device,
            )
            science_replacement = (
                torch.empty(
                    (
                        self.contract.B,
                        next_capacity,
                        COMPACT_SCIENCE_AGENT_METRIC_WIDTH_V1,
                    ),
                    dtype=torch.int64,
                    device=self.__state.device,
                )
                if self.__compact_science_capture
                else None
            )
            science_birth_provenance_replacement = (
                torch.empty(
                    (
                        self.contract.B,
                        next_capacity,
                        COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1,
                    ),
                    dtype=torch.uint8,
                    device=self.__state.device,
                )
                if self.__compact_science_capture
                else None
            )
        except Exception as exc:
            self._poison(
                "direct sparse staging growth was rejected before replacing "
                f"the live staging tensor: {exc}"
            )
        old_staging = self.__direct_events
        old_capacity = self.__capacity_rows
        old_admission = self.__admission
        old_science_staging = self.__science_event_metrics
        old_science_birth_provenance_staging = (
            self.__science_event_birth_provenance
        )
        self.__direct_events = replacement
        if self.__compact_science_capture:
            self.__science_event_metrics = science_replacement
            self.__science_event_birth_provenance = (
                science_birth_provenance_replacement
            )
        self.__capacity_rows = next_capacity
        self.__admission = {**old_admission, **admission}
        try:
            self._validate_storage()
        except Exception as exc:
            self.__direct_events = old_staging
            self.__science_event_metrics = old_science_staging
            self.__science_event_birth_provenance = (
                old_science_birth_provenance_staging
            )
            self.__capacity_rows = old_capacity
            self.__admission = old_admission
            self._poison(f"direct sparse staging replacement validation failed: {exc}")
        del old_staging
        if old_science_staging is not None:
            del old_science_staging
        if old_science_birth_provenance_staging is not None:
            del old_science_birth_provenance_staging

    def prepare_sparse_event_durable_envelope_v3(
        self,
        dispatch: CudaFullEvolutionSparseEventDispatchV3,
        *,
        sample_id: str,
    ) -> PreparedDirectSparseDurableEnvelopeV3:
        """Issue canonical bytes only for this owner's exact pending dispatch."""
        self._ensure_usable("sparse event durable-envelope preparation")
        pending = self.__pending
        try:
            if pending is None:
                raise CudaFullEvolutionRuntimeError(
                    "no direct sparse dispatch awaits durable-envelope preparation"
                )
            if type(dispatch) is not CudaFullEvolutionSparseEventDispatchV3:
                raise CudaFullEvolutionRuntimeError(
                    "durable-envelope preparation requires an exact sparse V3 dispatch"
                )
            if dispatch is not pending.dispatch:
                raise CudaFullEvolutionRuntimeError(
                    "durable-envelope preparation is bound to another dispatch object"
                )
            pending.verify_dispatch(dispatch)
            already_prepared = self.__pending_prepared_envelope
            if already_prepared is not None:
                already_prepared.verify_dispatch(dispatch, sample_id=sample_id)
                return already_prepared
            # advance_and_drain_v3 centrally validated the CPU projection and
            # content seal immediately before installing this frozen object as
            # pending.  Re-encoding that 9-10 MB dispatch is unnecessary here.
            prepared = _prepare_direct_sparse_durable_envelope_unchecked_v3(
                dispatch, sample_id=sample_id
            )
            self.__pending_prepared_envelope = prepared
            return prepared
        except Exception as exc:
            if pending is None:
                raise
            self._restore_snapshot_and_poison(
                f"direct sparse durable-envelope preparation failed: {exc}"
            )

    def reject_pending_sparse_event_drain_v3(self, *, reason: str) -> None:
        """Rollback and poison after a post-transition sink publication fault."""
        self._ensure_usable("sparse event sink rejection")
        if self.__pending is None:
            raise CudaFullEvolutionRuntimeError(
                "no direct sparse dispatch awaits sink rejection"
            )
        checked_reason = _checked_sink_token(reason, "sink rejection reason")
        self._restore_snapshot_and_poison(checked_reason)

    def advance_and_drain_v3(
        self, ticks: int
    ) -> CudaFullEvolutionSparseEventDispatchV3:
        """Advance through the ordinary sparse transaction boundary.

        This compatibility entry point deliberately keeps the pre-science
        return type and allocation behaviour.  Scientific capture is an
        additive acquisition path below and never participates in the
        authoritative state/checkpoint contract.
        """
        dispatch, _capture = self._advance_and_drain_v3(
            ticks, capture_science=False, capture_compact_science=False
        )
        return dispatch

    def advance_and_drain_science_v1(
        self, ticks: int
    ) -> tuple[
        CudaFullEvolutionSparseEventDispatchV3,
        CudaFullEvolutionScienceCaptureV1,
    ]:
        """Advance one exact transaction and return detached science primitives."""
        dispatch, capture = self._advance_and_drain_v3(
            ticks, capture_science=True, capture_compact_science=False
        )
        if capture is None:  # pragma: no cover - defensive invariant
            raise CudaFullEvolutionRuntimeError(
                "native science capture returned no acquisition payload"
            )
        return dispatch, capture

    def advance_and_drain_compact_science_v1(
        self, ticks: int
    ) -> tuple[
        CudaFullEvolutionSparseEventDispatchV3,
        CudaFullEvolutionCompactScienceCaptureV1,
    ]:
        """Advance once and return event-aligned compact science metrics."""

        if not self.__compact_science_capture:
            raise CudaFullEvolutionRuntimeError(
                "compact science capture was not bound at owner construction"
            )
        dispatch, capture = self._advance_and_drain_v3(
            ticks, capture_science=False, capture_compact_science=True
        )
        if not isinstance(capture, CudaFullEvolutionCompactScienceCaptureV1):
            raise CudaFullEvolutionRuntimeError(
                "native compact science capture returned no acquisition payload"
            )
        return dispatch, capture

    def _advance_and_drain_v3(
        self,
        ticks: int,
        *,
        capture_science: bool,
        capture_compact_science: bool,
    ) -> tuple[
        CudaFullEvolutionSparseEventDispatchV3,
        CudaFullEvolutionScienceCaptureV1
        | CudaFullEvolutionCompactScienceCaptureV1
        | None,
    ]:
        self._ensure_no_pending("advance")
        if capture_science and capture_compact_science:
            raise CudaFullEvolutionRuntimeError(
                "dense and compact science capture cannot share one dispatch"
            )
        if capture_compact_science and not self.__compact_science_capture:
            raise CudaFullEvolutionRuntimeError(
                "compact science capture is not resident"
            )
        if type(ticks) is not int or ticks < 1:
            raise CudaFullEvolutionRuntimeError("ticks must be a positive integer")
        if ticks > self.contract.temporal_k or ticks > self.remaining_ticks:
            raise CudaFullEvolutionRuntimeError(
                "direct sparse advance exceeds temporal or absolute horizon"
            )
        if self.__native_world_projection is not None:
            start_tick = self.committed_next_absolute_tick
            next_boundary = self.__native_world_projection.next_boundary_after(
                start_tick
            )
            if next_boundary is not None and next_boundary < start_tick + ticks:
                raise CudaFullEvolutionRuntimeError(
                    "native world-control dispatch crosses an absolute-tick boundary"
                )
            import torch

            rows = self.__native_world_projection.rows_at_tick(start_tick)
            self.__science_world_control_table.copy_(
                torch.tensor(
                    rows,
                    dtype=torch.int64,
                    device=self.__state.device,
                )
            )
        self._validate_frontier(
            expected_next_tick=self.committed_next_absolute_tick,
            expected_ticks_completed=self.__advanced_ticks,
        )
        import torch

        torch.cuda.reset_peak_memory_stats(self.__state.device)
        science_action_ids = None
        science_action_values = None
        science_birth_child_ids = None
        science_birth_parent_a_ids = None
        science_birth_parent_b_ids = None
        science_birth_mutation_counts = None
        science_birth_child_slots = None
        science_birth_sites = None
        science_birth_counts = None
        if capture_science:
            science_action_ids = torch.empty(
                (self.contract.B, ticks, RESIDENT_CAPACITY_C_V2),
                dtype=torch.int64,
                device=self.__state.device,
            )
            science_action_values = torch.empty_like(science_action_ids, dtype=torch.uint8)
            birth_shape = (
                self.contract.B,
                ticks,
                RESIDENT_CAPACITY_C_V2 // 2,
            )
            science_birth_child_ids = torch.empty(
                birth_shape, dtype=torch.int64, device=self.__state.device
            )
            science_birth_parent_a_ids = torch.empty_like(science_birth_child_ids)
            science_birth_parent_b_ids = torch.empty_like(science_birth_child_ids)
            science_birth_mutation_counts = torch.empty(
                birth_shape, dtype=torch.int32, device=self.__state.device
            )
            science_birth_child_slots = torch.empty_like(science_birth_mutation_counts)
            science_birth_sites = torch.empty_like(science_birth_mutation_counts)
            science_birth_counts = torch.empty(
                (self.contract.B, ticks),
                dtype=torch.int32,
                device=self.__state.device,
            )
        drain_started = perf_counter()
        snapshot_started = perf_counter()
        try:
            self.__rollback_state.copy_(self.__state)
            if self.__compact_science_capture:
                self.__science_agent_metrics_rollback.copy_(
                    self.__science_agent_metrics
                )
        except Exception as exc:
            self._poison(f"full-state rollback snapshot failed before dispatch: {exc}")
        snapshot_elapsed = perf_counter() - snapshot_started
        rollback_elapsed = 0.0
        retry_count = 0
        kernel_elapsed_ms = 0.0
        count_projection_elapsed = 0.0
        start_tick = self.committed_next_absolute_tick
        while True:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            try:
                if capture_science:
                    science_entrypoint = (
                        self.__loaded.module.advance_direct_sparse_science_worlds_v1
                        if self.__native_world_projection is not None
                        else self.__loaded.module.advance_direct_sparse_science_v1
                    )
                    science_arguments = [
                        self.__state,
                        self.__direct_events,
                        self.__event_counts,
                        self.__advance_validation,
                        science_action_ids,
                        science_action_values,
                        science_birth_child_ids,
                        science_birth_parent_a_ids,
                        science_birth_parent_b_ids,
                        science_birth_mutation_counts,
                        science_birth_child_slots,
                        science_birth_sites,
                        science_birth_counts,
                    ]
                    if self.__native_world_projection is not None:
                        science_arguments.append(self.__science_world_control_table)
                    science_arguments.extend((ticks, self.__capacity_rows))
                    if self.__native_world_projection is None:
                        science_arguments.append(self.__capability_mask)
                    science_entrypoint(*science_arguments)
                elif capture_compact_science:
                    self.__loaded.module.advance_direct_sparse_compact_science_worlds_v1(
                        self.__state,
                        self.__direct_events,
                        self.__event_counts,
                        self.__advance_validation,
                        self.__science_world_control_table,
                        self.__science_agent_metrics,
                        self.__science_event_metrics,
                        self.__science_agent_birth_provenance,
                        self.__science_event_birth_provenance,
                        ticks,
                        self.__capacity_rows,
                    )
                else:
                    if self.__native_world_projection is None:
                        self.__loaded.module.advance_direct_sparse_v3(
                            self.__state,
                            self.__direct_events,
                            self.__event_counts,
                            self.__advance_validation,
                            ticks,
                            self.__capacity_rows,
                            self.__capability_mask,
                        )
                    else:
                        self.__loaded.module.advance_direct_sparse_worlds_v1(
                            self.__state,
                            self.__direct_events,
                            self.__event_counts,
                            self.__advance_validation,
                            self.__science_world_control_table,
                            ticks,
                            self.__capacity_rows,
                        )
                end_event.record()
            except Exception as exc:
                self._restore_snapshot_and_poison(
                    f"native direct sparse launch failed: {exc}"
                )
            projection_started = perf_counter()
            try:
                event_counts = tuple(
                    int(value) for value in self.__event_counts.cpu().tolist()
                )
                advance_validation = tuple(
                    int(value) for value in self.__advance_validation.cpu().tolist()
                )
                kernel_elapsed_ms += float(start_event.elapsed_time(end_event))
            except Exception as exc:
                self._restore_snapshot_and_poison(
                    f"direct sparse launch synchronization failed: {exc}"
                )
            count_projection_elapsed += perf_counter() - projection_started
            overflow_only = any(advance_validation) and all(
                value in (0, DIRECT_SPARSE_OVERFLOW_BIT_V3)
                for value in advance_validation
            )
            maximum_required = max(event_counts, default=0)
            if overflow_only and maximum_required > self.__capacity_rows:
                rollback_started = perf_counter()
                self._restore_snapshot_or_poison(
                    "direct sparse overflow full-state rollback failed"
                )
                rollback_elapsed += perf_counter() - rollback_started
                self._grow_capacity(maximum_required)
                retry_count += 1
                continue
            if any(advance_validation) or any(
                value < 0 or value > self.__capacity_rows for value in event_counts
            ):
                self._restore_snapshot_and_poison(
                    "direct sparse launch validation failed: "
                    f"bits={advance_validation[:8]}, max_count={maximum_required}"
                )
            break

        end_tick = start_tick + ticks
        try:
            header_rows = self._validate_frontier(
                expected_next_tick=end_tick,
                expected_ticks_completed=self.__advanced_ticks + ticks,
            )
            offsets_list = [0]
            for count in event_counts:
                next_offset = offsets_list[-1] + count
                if (
                    next_offset
                    > MAX_SIGNED_INT64 // CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3
                ):
                    raise CudaFullEvolutionRuntimeError(
                        "direct sparse packed output exceeds signed int64"
                    )
                offsets_list.append(next_offset)
            world_offsets = tuple(offsets_list)
            pack_admission = _device_admission(
                torch,
                device=self.__state.device,
                batch_size=self.contract.B,
                capacity_rows=self.__capacity_rows,
                phase="packed_output",
                compact_science_capture=self.__compact_science_capture,
                native_world_control=self.__native_world_projection is not None,
            )
            receipt_admission = dict(self.__admission)
            receipt_admission.update(
                {
                    "last_pack_phase_required_free_bytes": pack_admission[
                        "phase_required_free_bytes"
                    ],
                    "last_pack_observed_free_bytes": pack_admission[
                        "observed_free_bytes"
                    ],
                }
            )
            pack_started = perf_counter()
            offsets_device = torch.tensor(
                world_offsets, dtype=torch.int64, device=self.__state.device
            )
            packed_device = torch.empty(
                (world_offsets[-1], CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3),
                dtype=torch.uint8,
                device=self.__state.device,
            )
            packed_science_event_metrics_device = (
                torch.empty(
                    (
                        world_offsets[-1],
                        COMPACT_SCIENCE_AGENT_METRIC_WIDTH_V1,
                    ),
                    dtype=torch.int64,
                    device=self.__state.device,
                )
                if capture_compact_science
                else None
            )
            packed_science_event_birth_provenance_device = (
                torch.empty(
                    (
                        world_offsets[-1],
                        COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1,
                    ),
                    dtype=torch.uint8,
                    device=self.__state.device,
                )
                if capture_compact_science
                else None
            )
            pack_validation_device = torch.empty_like(self.__advance_validation)
            if capture_compact_science:
                self.__loaded.module.pack_direct_sparse_science_events_v1(
                    self.__state,
                    self.__direct_events,
                    self.__event_counts,
                    self.__advance_validation,
                    offsets_device,
                    packed_device,
                    pack_validation_device,
                    self.__science_agent_metrics,
                    self.__science_event_metrics,
                    packed_science_event_metrics_device,
                    self.__science_agent_birth_provenance,
                    self.__science_event_birth_provenance,
                    packed_science_event_birth_provenance_device,
                    ticks,
                    self.__capacity_rows,
                )
            else:
                self.__loaded.module.pack_direct_sparse_events_v3(
                    self.__state,
                    self.__direct_events,
                    self.__event_counts,
                    self.__advance_validation,
                    offsets_device,
                    packed_device,
                    pack_validation_device,
                    ticks,
                    self.__capacity_rows,
                )
            pack_validation = tuple(
                int(value) for value in pack_validation_device.cpu().tolist()
            )
            packed_cpu = packed_device.cpu()
            if any(pack_validation):
                raise CudaFullEvolutionRuntimeError(
                    f"direct sparse pack validation failed: {pack_validation[:8]}"
                )
            final_next_agent_ids = tuple(int(row[2]) for row in header_rows)
            _validate_sparse_event_projection_v3(
                packed_rows=packed_cpu,
                world_offsets=world_offsets,
                event_counts=event_counts,
                final_next_agent_ids=final_next_agent_ids,
                tick_count=ticks,
            )
            packed_events = packed_cpu.numpy().tobytes(order="C")
            pack_elapsed = perf_counter() - pack_started
            world_uids = tuple(int(row[0]) for row in header_rows)
            hash_started = perf_counter()
            content_sha256 = _sparse_event_content_sha256(
                packed_events=packed_events,
                world_offsets=world_offsets,
                event_counts=event_counts,
                final_next_agent_ids=final_next_agent_ids,
                world_uids=world_uids,
                start_next_absolute_tick=start_tick,
                end_next_absolute_tick=end_tick,
                tick_count=ticks,
                artifact_sha256=self.__loaded.artifact_sha256,
                manifest_sha256=self.__loaded.manifest_sha256,
            )
            hash_elapsed = perf_counter() - hash_started
            drain_elapsed = perf_counter() - drain_started
            dispatch = CudaFullEvolutionSparseEventDispatchV3(
                packed_events=packed_events,
                world_offsets=world_offsets,
                event_counts=event_counts,
                final_next_agent_ids=final_next_agent_ids,
                world_uids=world_uids,
                start_next_absolute_tick=start_tick,
                end_next_absolute_tick=end_tick,
                tick_count=ticks,
                drain_elapsed_seconds=drain_elapsed,
                count_projection_elapsed_seconds=count_projection_elapsed,
                compaction_and_transfer_elapsed_seconds=pack_elapsed,
                content_hash_elapsed_seconds=hash_elapsed,
                content_sha256=content_sha256,
                artifact_sha256=self.__loaded.artifact_sha256,
                manifest_sha256=self.__loaded.manifest_sha256,
            )
            pending = _PendingDirectSparseDispatchV3(
                start_next_absolute_tick=start_tick,
                tick_count=ticks,
                dispatch=dispatch,
                dispatch_authority=_capture_direct_sparse_dispatch_authority_v3(
                    dispatch
                ),
            )
            metrics = DirectSparseDispatchMetricsV3(
                snapshot_elapsed_seconds=snapshot_elapsed,
                rollback_elapsed_seconds=rollback_elapsed,
                retry_count=retry_count,
                selected_capacity_rows_per_world=self.__capacity_rows,
                kernel_elapsed_ms_diagnostic_only=kernel_elapsed_ms,
                peak_allocated_bytes=int(
                    torch.cuda.max_memory_allocated(self.__state.device)
                ),
                peak_reserved_bytes=int(
                    torch.cuda.max_memory_reserved(self.__state.device)
                ),
            )
            execution_receipt = {
                "schema_version": DIRECT_SPARSE_EXECUTION_RECEIPT_SCHEMA_V3,
                "owner_schema_version": DIRECT_SPARSE_OWNER_SCHEMA_V3,
                "profile_id": self.contract.profile_id,
                "artifact_sha256": self.__loaded.artifact_sha256,
                "manifest_sha256": self.__loaded.manifest_sha256,
                "device_identity": dict(self.__loaded.device_identity),
                "capability_mask": self.__capability_mask,
                "native_control": (
                    None
                    if self.__native_control_vector is None
                    else {
                        "schema_version": "eiel_science_cuda_v2_native_control_v1",
                        "capability_mask": self.__capability_mask,
                        "control_vector": dict(self.__native_control_vector),
                        "control_digest": {
                            "algorithm": "sha256",
                            "hex": self.__native_control_digest,
                        },
                    }
                ),
                "native_world_control": (
                    None
                    if self.__native_world_projection is None
                    else dict(self.native_world_control_binding_v1)
                ),
                "B": self.contract.B,
                "start_next_absolute_tick": start_tick,
                "end_next_absolute_tick": end_tick,
                "tick_count": ticks,
                "enqueued_world_ticks": self.contract.B * ticks,
                "committed_world_ticks": 0,
                "selected_capacity_rows_per_world": self.__capacity_rows,
                "retry_count": retry_count,
                "rollback_snapshot_full_state_bytes": int(
                    self.__rollback_state.numel()
                ),
                "pointers": {
                    "state": int(self.__state.data_ptr()),
                    "rollback_state": int(self.__rollback_state.data_ptr()),
                    "direct_sparse_staging": int(self.__direct_events.data_ptr()),
                    "event_counts": int(self.__event_counts.data_ptr()),
                    "advance_validation": int(self.__advance_validation.data_ptr()),
                    "science_world_controls": (
                        0
                        if self.__science_world_control_table is None
                        else int(self.__science_world_control_table.data_ptr())
                    ),
                    "science_agent_metrics": (
                        0
                        if self.__science_agent_metrics is None
                        else int(self.__science_agent_metrics.data_ptr())
                    ),
                    "science_event_metrics": (
                        0
                        if self.__science_event_metrics is None
                        else int(self.__science_event_metrics.data_ptr())
                    ),
                    "science_agent_birth_provenance": (
                        0
                        if self.__science_agent_birth_provenance is None
                        else int(self.__science_agent_birth_provenance.data_ptr())
                    ),
                    "science_event_birth_provenance": (
                        0
                        if self.__science_event_birth_provenance is None
                        else int(self.__science_event_birth_provenance.data_ptr())
                    ),
                },
                "event_wire_abi": dispatch.wire_abi,
                "event_content_sha256": dispatch.content_sha256,
                "event_count": dispatch.event_count,
                "event_status": (
                    "exact_direct_sparse_content_validated_not_acknowledged"
                ),
                "runtime_status": "pending_durable_sink_acknowledgement",
                "horizon_dependent_resident_bytes": 0,
                "resource_admission": receipt_admission,
                "science_capture": {
                    "enabled": capture_science or capture_compact_science,
                    "schema": (
                        COMPACT_SCIENCE_CAPTURE_SCHEMA_V1
                        if capture_compact_science
                        else (
                            "eiel_full_evolution_cuda_science_capture_v1"
                            if capture_science
                            else None
                        )
                    ),
                    "bytes": (
                        int(
                            packed_science_event_metrics_device.numel()
                            * packed_science_event_metrics_device.element_size()
                            + packed_science_event_birth_provenance_device.numel()
                            * packed_science_event_birth_provenance_device.element_size()
                        )
                        if capture_compact_science
                        else (0 if not capture_science else sum(
                            int(tensor.numel() * tensor.element_size())
                            for tensor in (
                                science_action_ids,
                                science_action_values,
                                science_birth_child_ids,
                                science_birth_parent_a_ids,
                                science_birth_parent_b_ids,
                                science_birth_mutation_counts,
                                science_birth_child_slots,
                                science_birth_sites,
                                science_birth_counts,
                            )
                        ))
                    ),
                },
            }
        except Exception as exc:
            self._restore_snapshot_and_poison(
                f"direct sparse post-transition drain failed: {exc}"
            )
        self.__pending = pending
        self.__last_metrics = metrics
        self.__last_execution_receipt = execution_receipt
        capture = None
        if capture_compact_science:
            capture = CudaFullEvolutionCompactScienceCaptureV1(
                event_agent_metrics=(
                    packed_science_event_metrics_device.cpu()
                ),
                event_birth_provenance=(
                    packed_science_event_birth_provenance_device.cpu()
                ),
                metric_fields=COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
                birth_provenance_encoding=(
                    COMPACT_SCIENCE_BIRTH_PROVENANCE_ENCODING_V1
                ),
                start_next_absolute_tick=start_tick,
                end_next_absolute_tick=end_tick,
                world_uids=world_uids,
                world_offsets=world_offsets,
                event_content_sha256=content_sha256,
            )
        elif capture_science:
            # event_counts.cpu() and packed_device.cpu() above synchronize the
            # current stream.  These copies are therefore detached acquisition
            # data and cannot race the resident state transition.
            capture = CudaFullEvolutionScienceCaptureV1(
                action_agent_ids=science_action_ids.cpu(),
                action_values=science_action_values.cpu(),
                birth_child_ids=science_birth_child_ids.cpu(),
                birth_parent_a_ids=science_birth_parent_a_ids.cpu(),
                birth_parent_b_ids=science_birth_parent_b_ids.cpu(),
                birth_mutation_counts=science_birth_mutation_counts.cpu(),
                birth_child_slots=science_birth_child_slots.cpu(),
                birth_sites=science_birth_sites.cpu(),
                birth_counts=science_birth_counts.cpu(),
                start_next_absolute_tick=start_tick,
                end_next_absolute_tick=end_tick,
                world_uids=world_uids,
            )
        return dispatch, capture

    def acknowledge_sparse_event_drain_v3(
        self,
        receipt: CudaFullEvolutionDirectSparseDurableSinkReceiptV3,
        *,
        durable_envelope: VerifiedDirectSparseDurableEnvelopeV3,
    ) -> None:
        self._ensure_usable("sparse event acknowledgement")
        pending = self.__pending
        if pending is None:
            raise CudaFullEvolutionRuntimeError(
                "no direct sparse dispatch awaits acknowledgement"
            )
        dispatch = pending.dispatch
        try:
            pending.verify_dispatch(dispatch)
            prepared = self.__pending_prepared_envelope
            if prepared is None or not durable_envelope.is_bound_to_prepared(prepared):
                raise CudaFullEvolutionRuntimeError(
                    "durable sparse acknowledgement does not carry the owner-issued "
                    "prepared capability"
                )
            binding = validate_direct_sparse_durable_sink_receipt_v3(
                dispatch,
                receipt,
                durable_envelope=durable_envelope,
            )
            self._validate_frontier(
                expected_next_tick=pending.end_next_absolute_tick,
                expected_ticks_completed=self.__advanced_ticks + pending.tick_count,
            )
        except Exception as exc:
            self._restore_snapshot_and_poison(
                f"durable sparse acknowledgement failed: {exc}"
            )
        previous_advanced_ticks = self.__advanced_ticks
        previous_sink_receipt_sha256 = self.__last_sink_receipt_sha256
        self.__advanced_ticks += pending.tick_count
        self.__last_sink_receipt_sha256 = receipt.receipt_sha256
        try:
            durable_envelope.verify(
                dispatch=dispatch,
                dispatch_content_sha256=dispatch.content_sha256,
                sample_id=receipt.sample_id,
            )
        except Exception as exc:
            self.__advanced_ticks = previous_advanced_ticks
            self.__last_sink_receipt_sha256 = previous_sink_receipt_sha256
            self._restore_snapshot_and_poison(
                f"durable envelope changed at commit boundary: {exc}"
            )
        self.__pending = None
        self.__pending_prepared_envelope = None
        if self.__last_execution_receipt is not None:
            self.__last_execution_receipt.update(
                {
                    "committed_world_ticks": self.contract.B * pending.tick_count,
                    "committed_next_absolute_tick": self.committed_next_absolute_tick,
                    "sink_receipt_sha256": receipt.receipt_sha256,
                    "durable_envelope": dict(binding),
                    "event_status": "exact_direct_sparse_content_sink_acknowledged",
                    "runtime_status": "committed",
                }
            )

    def counters(self) -> list[dict[str, int]]:
        self._ensure_no_pending("counter projection")
        rows = self._validate_frontier(
            expected_next_tick=self.committed_next_absolute_tick,
            expected_ticks_completed=self.__advanced_ticks,
        )
        return [
            dict(zip(COUNTER_NAMES_V2, map(int, row[6:13]), strict=True))
            for row in rows
        ]

    def force_full_capacity_benchmark_fixture_v2(
        self,
    ) -> CudaFullEvolutionCheckpointV2:
        self._ensure_no_pending("full-capacity fixture")
        rows = self._validate_frontier(
            expected_next_tick=self.committed_next_absolute_tick,
            expected_ticks_completed=self.__advanced_ticks,
        )
        if (
            self.__advanced_ticks != 0
            or self.__last_sink_receipt_sha256 is not None
            or self.__population_fixture_id is not None
            or any(int(row[3]) != INITIAL_POPULATION_V2 for row in rows)
        ):
            raise CudaFullEvolutionRuntimeError(
                "full-capacity fixture requires a fresh canonical P256 frontier"
            )
        try:
            self.__loaded.module.force_full_capacity_benchmark_fixture_v2(self.__state)
        except Exception as exc:
            self._poison(f"native full-capacity fixture construction failed: {exc}")
        self.__population_fixture_id = FULL_CAPACITY_BENCHMARK_FIXTURE_V2
        validated = self._validate_frontier(
            expected_next_tick=self.committed_next_absolute_tick,
            expected_ticks_completed=self.__advanced_ticks,
        )
        if any(int(row[3]) != RESIDENT_CAPACITY_C_V2 for row in validated):
            self._poison("full-capacity fixture did not produce exact C512")
        return self.checkpoint()

    def capture_native_snapshot_to_pinned_v1(
        self,
        *,
        absolute_tick: int,
        state_destination: Any,
        active_science_metrics_destination: Any,
    ) -> CudaFullEvolutionPinnedSnapshotCaptureFenceV1:
        """Issue one exact resident state/metrics D2H capture on the current stream.

        The caller owns both reusable destinations until the returned fence has
        completed.  This method does not build a Python metrics projection and
        does not create an intermediate device clone.  The state destination
        may bind either the complete resident row or its exact canonical
        checkpoint prefix.  The latter contains every snapshot source field
        while excluding non-authoritative expressed-cache scratch columns.
        """

        import torch

        self._ensure_no_pending("native snapshot capture")
        if (
            type(absolute_tick) is not int
            or absolute_tick < 0
            or absolute_tick > MAX_SIGNED_INT64
        ):
            raise CudaFullEvolutionRuntimeError(
                "native snapshot absolute_tick must be a non-negative signed-int64"
            )
        if absolute_tick != self.committed_next_absolute_tick:
            raise CudaFullEvolutionRuntimeError(
                "native snapshot tick differs from the committed absolute frontier"
            )
        if not self.__compact_science_capture:
            raise CudaFullEvolutionRuntimeError(
                "native snapshot active science metrics are not resident"
            )
        # The committed scalar frontier is already authoritative here.  Do not
        # synchronously project header rows through Python on the producer
        # boundary; the plan-bound native encoder validates every captured
        # selected-world UID and next_absolute_tick before encoding.

        canonical_state_bytes = int(
            state_layout_v2().canonical_checkpoint_bytes_per_world
        )
        full_state_shape = tuple(self.__state.shape)
        canonical_state_shape = (
            int(self.__state.shape[0]),
            canonical_state_bytes,
        )
        if tuple(getattr(state_destination, "shape", ())) == full_state_shape:
            state_source = self.__state
        elif tuple(getattr(state_destination, "shape", ())) == canonical_state_shape:
            state_source = self.__state[:, :canonical_state_bytes]
        else:
            raise CudaFullEvolutionRuntimeError(
                "native snapshot state destination shape must bind either the "
                "complete resident row or its canonical checkpoint prefix"
            )

        destinations = (
            (
                state_destination,
                state_source,
                torch.uint8,
                "state",
            ),
            (
                active_science_metrics_destination,
                self.__science_agent_metrics,
                torch.int64,
                "active science metrics",
            ),
        )
        destination_extents: list[tuple[int, int, str]] = []
        for destination, source, dtype, label in destinations:
            if (
                not isinstance(destination, torch.Tensor)
                or destination.dtype != dtype
                or destination.device.type != "cpu"
                or not destination.is_contiguous()
                or not destination.is_pinned()
                or tuple(destination.shape) != tuple(source.shape)
            ):
                raise CudaFullEvolutionRuntimeError(
                    f"native snapshot {label} destination must be an exact "
                    "contiguous pinned CPU tensor"
                )
            start = int(destination.data_ptr())
            end = start + int(destination.numel() * destination.element_size())
            for other_start, other_end, other_label in destination_extents:
                if start < other_end and other_start < end:
                    raise CudaFullEvolutionRuntimeError(
                        "native snapshot destination byte ranges overlap: "
                        f"{other_label} and {label}"
                    )
            destination_extents.append((start, end, label))

        stream = torch.cuda.current_stream(self.__state.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        try:
            state_destination.copy_(state_source, non_blocking=True)
            active_science_metrics_destination.copy_(
                self.__science_agent_metrics, non_blocking=True
            )
            end_event.record(stream)
        except Exception as exc:
            try:
                stream.synchronize()
            except Exception as synchronization_error:
                raise CudaFullEvolutionRuntimeError(
                    "native snapshot capture failed and its partial D2H copy "
                    f"could not be fenced: {synchronization_error}"
                ) from exc
            raise CudaFullEvolutionRuntimeError(
                f"native snapshot capture failed: {exc}"
            ) from exc
        return CudaFullEvolutionPinnedSnapshotCaptureFenceV1(
            absolute_tick=absolute_tick,
            state_bytes=int(state_source.numel() * state_source.element_size()),
            active_science_metrics_bytes=int(
                self.__science_agent_metrics.numel()
                * self.__science_agent_metrics.element_size()
            ),
            source_device=str(self.__state.device),
            start_event=start_event,
            end_event=end_event,
        )

    def state_projection(
        self,
        *,
        timing_diagnostics: MutableMapping[str, Any] | None = None,
    ) -> Any:
        if timing_diagnostics is None:
            self._ensure_no_pending("state projection")
            self._validate_frontier(
                expected_next_tick=self.committed_next_absolute_tick,
                expected_ticks_completed=self.__advanced_ticks,
            )
            return self.__state.detach().clone()
        total_wall_started = perf_counter_ns()
        total_cpu_started = process_time_ns()
        validation_wall_started = perf_counter_ns()
        validation_cpu_started = process_time_ns()
        self._ensure_no_pending("state projection")
        self._validate_frontier(
            expected_next_tick=self.committed_next_absolute_tick,
            expected_ticks_completed=self.__advanced_ticks,
        )
        validation_wall_ns = perf_counter_ns() - validation_wall_started
        validation_cpu_ns = process_time_ns() - validation_cpu_started
        projection_wall_started = perf_counter_ns()
        projection_cpu_started = process_time_ns()
        if self.__state.is_cuda:
            import torch

        projection_events = (
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            if self.__state.is_cuda
            else None
        )
        if projection_events is not None:
            projection_events[0].record()
        projection = self.__state.detach().clone()
        if projection_events is not None:
            projection_events[1].record()
            projection_events[1].synchronize()
        projection_wall_ns = perf_counter_ns() - projection_wall_started
        projection_cpu_ns = process_time_ns() - projection_cpu_started
        projection_cuda_event_ns = (
            round(
                float(
                    projection_events[0].elapsed_time(projection_events[1])
                )
                * 1_000_000
            )
            if projection_events is not None
            else None
        )
        if timing_diagnostics is not None:
            timing_diagnostics.clear()
            timing_diagnostics.update(
                {
                    "schema_version": (
                        "eiel_compact_science_state_projection_timing_v1"
                    ),
                    "claim_class": "TECHNICAL_ONLY",
                    "phases": {
                        "frontier_validation": {
                            "wall_time_ns": validation_wall_ns,
                            "cpu_time_ns": validation_cpu_ns,
                        },
                        "device_projection_clone_and_wait": {
                            "wall_time_ns": projection_wall_ns,
                            "cpu_time_ns": projection_cpu_ns,
                            "cuda_event_time_ns": projection_cuda_event_ns,
                        },
                    },
                    "projection_bytes": (
                        int(projection.numel()) * int(projection.element_size())
                    ),
                    "source_device": str(self.__state.device),
                    "projection_device": str(projection.device),
                    "timing_note": (
                        "benchmark-scoped timing synchronizes the clone end event; "
                        "CUDA event time isolates device execution while wall time "
                        "also includes the host wait"
                    ),
                    "total_wall_time_ns": perf_counter_ns()
                    - total_wall_started,
                    "total_cpu_time_ns": process_time_ns()
                    - total_cpu_started,
                }
            )
        return projection

    def compact_science_active_metrics_projection_v1(
        self,
        *,
        timing_diagnostics: MutableMapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Detach the exact active-slot lifetime counters at a frontier."""

        if timing_diagnostics is None:
            self._ensure_no_pending("compact science metrics projection")
            if not self.__compact_science_capture:
                raise CudaFullEvolutionRuntimeError(
                    "compact science metrics are not resident"
                )
            self._validate_frontier(
                expected_next_tick=self.committed_next_absolute_tick,
                expected_ticks_completed=self.__advanced_ticks,
            )
            projection = self.__science_agent_metrics.detach().cpu().contiguous()
            raw = projection.numpy().tobytes(order="C")
            return {
                "schema_version": (
                    "eiel_full_evolution_cuda_active_science_metrics_v1"
                ),
                "next_absolute_tick": self.committed_next_absolute_tick,
                "world_uids": list(self.__world_uids),
                "metric_fields": list(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1),
                "shape": list(projection.shape),
                "content_sha256": hashlib.sha256(raw).hexdigest(),
                "tensor": projection,
            }
        total_wall_started = perf_counter_ns()
        total_cpu_started = process_time_ns()
        validation_wall_started = perf_counter_ns()
        validation_cpu_started = process_time_ns()
        self._ensure_no_pending("compact science metrics projection")
        if not self.__compact_science_capture:
            raise CudaFullEvolutionRuntimeError(
                "compact science metrics are not resident"
            )
        self._validate_frontier(
            expected_next_tick=self.committed_next_absolute_tick,
            expected_ticks_completed=self.__advanced_ticks,
        )
        validation_wall_ns = perf_counter_ns() - validation_wall_started
        validation_cpu_ns = process_time_ns() - validation_cpu_started
        d2h_wall_started = perf_counter_ns()
        d2h_cpu_started = process_time_ns()
        if self.__science_agent_metrics.is_cuda:
            import torch

        d2h_events = (
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            if self.__science_agent_metrics.is_cuda
            else None
        )
        if d2h_events is not None:
            d2h_events[0].record()
        projection = self.__science_agent_metrics.detach().cpu().contiguous()
        if d2h_events is not None:
            d2h_events[1].record()
            d2h_events[1].synchronize()
        d2h_wall_ns = perf_counter_ns() - d2h_wall_started
        d2h_cpu_ns = process_time_ns() - d2h_cpu_started
        d2h_cuda_event_ns = (
            round(
                float(d2h_events[0].elapsed_time(d2h_events[1]))
                * 1_000_000
            )
            if d2h_events is not None
            else None
        )
        bytes_wall_started = perf_counter_ns()
        bytes_cpu_started = process_time_ns()
        raw = projection.numpy().tobytes(order="C")
        bytes_wall_ns = perf_counter_ns() - bytes_wall_started
        bytes_cpu_ns = process_time_ns() - bytes_cpu_started
        hash_wall_started = perf_counter_ns()
        hash_cpu_started = process_time_ns()
        content_sha256 = hashlib.sha256(raw).hexdigest()
        hash_wall_ns = perf_counter_ns() - hash_wall_started
        hash_cpu_ns = process_time_ns() - hash_cpu_started
        result = {
            "schema_version": "eiel_full_evolution_cuda_active_science_metrics_v1",
            "next_absolute_tick": self.committed_next_absolute_tick,
            "world_uids": list(self.__world_uids),
            "metric_fields": list(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1),
            "shape": list(projection.shape),
            "content_sha256": content_sha256,
            "tensor": projection,
        }
        if timing_diagnostics is not None:
            timing_diagnostics.clear()
            timing_diagnostics.update(
                {
                    "schema_version": (
                        "eiel_compact_science_metrics_projection_timing_v1"
                    ),
                    "claim_class": "TECHNICAL_ONLY",
                    "phases": {
                        "frontier_validation": {
                            "wall_time_ns": validation_wall_ns,
                            "cpu_time_ns": validation_cpu_ns,
                        },
                        "metrics_d2h_and_contiguous": {
                            "wall_time_ns": d2h_wall_ns,
                            "cpu_time_ns": d2h_cpu_ns,
                            "cuda_event_time_ns": d2h_cuda_event_ns,
                        },
                        "metrics_host_bytes_conversion": {
                            "wall_time_ns": bytes_wall_ns,
                            "cpu_time_ns": bytes_cpu_ns,
                        },
                        "metrics_content_hashing": {
                            "wall_time_ns": hash_wall_ns,
                            "cpu_time_ns": hash_cpu_ns,
                        },
                    },
                    "d2h_bytes": len(raw),
                    "source_device": str(self.__science_agent_metrics.device),
                    "projection_device": str(projection.device),
                    "total_wall_time_ns": perf_counter_ns()
                    - total_wall_started,
                    "total_cpu_time_ns": process_time_ns()
                    - total_cpu_started,
                }
            )
        return result

    def execution_receipt_v3(self) -> Mapping[str, Any] | None:
        return (
            None
            if self.__last_execution_receipt is None
            else deepcopy(self.__last_execution_receipt)
        )

    def checkpoint(self) -> CudaFullEvolutionCheckpointV2:
        self._ensure_no_pending("checkpoint")
        self._validate_frontier(
            expected_next_tick=self.committed_next_absolute_tick,
            expected_ticks_completed=self.__advanced_ticks,
        )
        canonical_bytes_per_world = (
            state_layout_v2().canonical_checkpoint_bytes_per_world
        )
        resident_cpu = self.__state.cpu()
        canonical_tensor = resident_cpu[:, :canonical_bytes_per_world].contiguous()
        canonical_state = canonical_tensor.numpy().tobytes(order="C")
        shape = (self.contract.B, canonical_bytes_per_world)
        content_sha256 = hashlib.sha256(canonical_state).hexdigest()
        layout_sha256 = _canonical_state_layout_sha256(
            shape=shape,
            nbytes=len(canonical_state),
            content_sha256=content_sha256,
        )
        provisional = CudaFullEvolutionCheckpointV2(
            contract=self.contract,
            canonical_state=canonical_state,
            canonical_state_shape=shape,
            canonical_state_nbytes=len(canonical_state),
            canonical_state_sha256=content_sha256,
            canonical_state_layout_sha256=layout_sha256,
            initial_next_absolute_tick=self.__initial_next_absolute_tick,
            advanced_ticks=self.__advanced_ticks,
            acknowledged_event_next_tick=self.committed_next_absolute_tick,
            last_sink_receipt_sha256=self.__last_sink_receipt_sha256,
            world_uids=self.__world_uids,
            rng_keys=self.__rng_keys,
            rng_semantic_epoch=RNG_SEMANTIC_EPOCH_V3,
            artifact_path=self.__loaded.artifact_path,
            artifact_sha256=self.__loaded.artifact_sha256,
            manifest_sha256=self.__loaded.manifest_sha256,
            device_identity=dict(self.__loaded.device_identity),
            population_fixture_id=self.__population_fixture_id,
            metadata_seal_sha256="",
        )
        return CudaFullEvolutionCheckpointV2(
            **{
                field: getattr(provisional, field)
                for field in provisional.__dataclass_fields__
                if field != "metadata_seal_sha256"
            },
            metadata_seal_sha256=_canonical_sha256(
                _checkpoint_metadata_body(provisional)
            ),
        )

    def checkpoint_capture_source_v1(
        self,
    ) -> CudaFullEvolutionCheckpointCaptureSourceV1:
        """Borrow resident state at an exact committed frontier for ACS2."""

        self._ensure_no_pending("checkpoint capture source")
        self._validate_frontier(
            expected_next_tick=self.committed_next_absolute_tick,
            expected_ticks_completed=self.__advanced_ticks,
        )
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
            world_uids=self.__world_uids,
            rng_keys=self.__rng_keys,
            rng_semantic_epoch=RNG_SEMANTIC_EPOCH_V3,
            artifact_path=self.__loaded.artifact_path,
            artifact_sha256=self.__loaded.artifact_sha256,
            manifest_sha256=self.__loaded.manifest_sha256,
            device_identity=dict(self.__loaded.device_identity),
            population_fixture_id=self.__population_fixture_id,
        )

    def close(self) -> None:
        if self.__closed:
            return
        if self.__poisoned_reason is None:
            self._ensure_no_pending("close")
        import torch

        device = self.__state.device
        torch.cuda.synchronize(device)
        self.__pending = None
        self.__pending_prepared_envelope = None
        self.__last_metrics = None
        self.__last_execution_receipt = None
        self.__state = None
        self.__rollback_state = None
        self.__direct_events = None
        self.__event_counts = None
        self.__advance_validation = None
        self.__science_world_control_table = None
        self.__science_agent_metrics = None
        self.__science_agent_metrics_rollback = None
        self.__science_event_metrics = None
        self.__science_agent_birth_provenance = None
        self.__science_event_birth_provenance = None
        self.__native_world_projection = None
        self.__science_founder_application = None
        self.__closed = True


__all__ = [
    "DIRECT_SPARSE_ADMISSION_MARGIN_BYTES_V3",
    "DIRECT_SPARSE_EXECUTION_RECEIPT_SCHEMA_V3",
    "DIRECT_SPARSE_MAX_CAPACITY_ROWS_V3",
    "DIRECT_SPARSE_OWNER_SCHEMA_V3",
    "DIRECT_SPARSE_OVERFLOW_BIT_V3",
    "CudaFullEvolutionDirectSparseOwnerV3",
    "CudaFullEvolutionCompactScienceCaptureV1",
    "CudaFullEvolutionPinnedSnapshotCaptureFenceV1",
    "CudaFullEvolutionScienceCaptureV1",
    "COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1",
    "COMPACT_SCIENCE_BIRTH_PROVENANCE_ENCODING_V1",
    "COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1",
    "COMPACT_SCIENCE_CAPTURE_SCHEMA_V1",
    "DirectSparseDispatchMetricsV3",
]
