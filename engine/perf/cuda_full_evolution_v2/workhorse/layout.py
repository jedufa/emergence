"""Checked O(S+C) state, scratch, event and VRAM layout for workhorse V2."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Final

from .contract import (
    BACKEND_PHASE_GRAPH_V2,
    BACKEND_FUSED_CTA_TEMPORAL_V2,
    FullEvolutionCudaWorkhorseContractV2,
    FullEvolutionWorkhorseContractError,
    WORKHORSE_CHECKPOINT_SCHEMA_V2,
    WORKHORSE_EVENT_ROW_BYTES_V2,
    WORKHORSE_MAX_GRID_Y_V2,
    WorkhorseCompileShapeV2,
    canonical_sha256,
    checked_add_i64,
    checked_multiply_i64,
    workhorse_artifact_identity_v2,
    workhorse_backend_v2,
    workhorse_backend_temporal_k_candidates_v2,
    workhorse_shape_v2,
)

HEADER_WORDS_V2: Final[int] = 32
HEADER_BYTES_V2: Final[int] = HEADER_WORDS_V2 * 8
GENOME_LOCI_V2: Final[int] = 32
WORKHORSE_MINIMUM_HEADROOM_NUMERATOR_V2: Final[int] = 1
WORKHORSE_MINIMUM_HEADROOM_DENOMINATOR_V2: Final[int] = 4
WORKHORSE_CUDA_ALLOCATOR_SEGMENT_QUANTUM_BYTES_V2: Final[int] = 2 * 1024 * 1024
WORKHORSE_OWNER_DEVICE_ALLOCATION_COUNT_V2: Final[int] = 4
WORKHORSE_OWNER_TRANSIENT_ALLOCATION_QUANTA_V2: Final[int] = 1

_DTYPE_BYTES: Final[dict[str, int]] = {
    "u8": 1,
    "u16": 2,
    "u32": 4,
    "i32": 4,
    "u64": 8,
}


def workhorse_allocator_owner_overhead_bound_v2(
    observed_reserved_overhead_bytes: int,
) -> int:
    """Return the explicit allocator bound used by every B/K ledger.

    A Workhorse owner has four long-lived CUDA tensor allocations. The native
    caching allocator may retain the remainder of a 2 MiB segment with any one
    allocation, and one further segment quantum bounds transient allocator
    activity around dispatch/canonicalization instrumentation. The physical
    B=1 reserved-memory observation remains authoritative whenever it is
    larger. This is a named, recomputable bound rather than an untracked
    percentage or tolerance.
    """

    if (
        type(observed_reserved_overhead_bytes) is not int
        or observed_reserved_overhead_bytes < 0
    ):
        raise FullEvolutionWorkhorseContractError(
            "observed allocator overhead must be a nonnegative integer"
        )
    explicit_floor = checked_multiply_i64(
        WORKHORSE_CUDA_ALLOCATOR_SEGMENT_QUANTUM_BYTES_V2,
        WORKHORSE_OWNER_DEVICE_ALLOCATION_COUNT_V2
        + WORKHORSE_OWNER_TRANSIENT_ALLOCATION_QUANTA_V2,
        "workhorse allocator owner overhead bound",
    )
    return max(observed_reserved_overhead_bytes, explicit_floor)


def _align(value: int, alignment: int, label: str) -> int:
    if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
        raise FullEvolutionWorkhorseContractError(
            f"{label} alignment must be a positive power of two"
        )
    padded = checked_add_i64(value, alignment - 1, f"{label} alignment")
    return padded & ~(alignment - 1)


@dataclass(frozen=True, slots=True)
class WorkhorseFieldV2:
    name: str
    dtype: str
    count: int
    offset: int
    nbytes: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "count": self.count,
            "offset": self.offset,
            "nbytes": self.nbytes,
        }


def _pack_fields(
    specifications: tuple[tuple[str, str, int], ...], *, label: str
) -> tuple[tuple[WorkhorseFieldV2, ...], int]:
    fields: list[WorkhorseFieldV2] = []
    cursor = 0
    for name, dtype, count in specifications:
        if dtype not in _DTYPE_BYTES:
            raise FullEvolutionWorkhorseContractError(f"unsupported dtype {dtype}")
        if type(count) is not int or count < 0:
            raise FullEvolutionWorkhorseContractError(
                f"{label}.{name} count must be a nonnegative integer"
            )
        width = _DTYPE_BYTES[dtype]
        cursor = _align(cursor, min(width, 8), f"{label}.{name}")
        nbytes = checked_multiply_i64(count, width, f"{label}.{name} bytes")
        fields.append(WorkhorseFieldV2(name, dtype, count, cursor, nbytes))
        cursor = checked_add_i64(cursor, nbytes, f"{label}.{name} end")
    return tuple(fields), _align(cursor, 256, f"{label} row")


@dataclass(frozen=True, slots=True)
class WorkhorseLayoutV2:
    shape: WorkhorseCompileShapeV2
    state_fields: tuple[WorkhorseFieldV2, ...]
    scratch_fields: tuple[WorkhorseFieldV2, ...]
    state_bytes_per_world: int
    scratch_bytes_per_world: int

    def field(self, name: str, *, scratch: bool = False) -> WorkhorseFieldV2:
        fields = self.scratch_fields if scratch else self.state_fields
        for field in fields:
            if field.name == name:
                return field
        raise KeyError(name)

    @property
    def event_rows_per_world_tick(self) -> int:
        return checked_multiply_i64(2, self.shape.capacity, "event rows per world tick")

    @property
    def event_bytes_per_world_tick(self) -> int:
        return checked_multiply_i64(
            self.event_rows_per_world_tick,
            WORKHORSE_EVENT_ROW_BYTES_V2,
            "event bytes per world tick",
        )

    @property
    def layout_sha256(self) -> str:
        return canonical_sha256(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "eiel_full_evolution_cuda_workhorse_layout_v2",
            **self.shape.identity_fields(),
            "header_bytes": HEADER_BYTES_V2,
            "genome_loci": GENOME_LOCI_V2,
            "state_fields": [field.to_dict() for field in self.state_fields],
            "scratch_fields": [field.to_dict() for field in self.scratch_fields],
            "state_bytes_per_world": self.state_bytes_per_world,
            "scratch_bytes_per_world": self.scratch_bytes_per_world,
            "event_row_bytes": WORKHORSE_EVENT_ROW_BYTES_V2,
            "event_rows_per_world_tick": self.event_rows_per_world_tick,
            "event_bytes_per_world_tick": self.event_bytes_per_world_tick,
            "resident_event_pages_per_world": 1,
            "resident_event_bytes_per_world": self.event_bytes_per_world_tick,
            "resident_event_growth_with_temporal_k_bytes": 0,
            "growth_law": "horizon_independent_O(S_plus_C)",
        }
        if include_hash:
            payload["layout_sha256"] = canonical_sha256(payload)
        return payload


def workhorse_layout_v2(
    shape: str | WorkhorseCompileShapeV2,
) -> WorkhorseLayoutV2:
    if isinstance(shape, str):
        shape = workhorse_shape_v2(shape)
    if workhorse_shape_v2(shape.shape_id) != shape:
        raise FullEvolutionWorkhorseContractError(
            "layout accepts only an exact reviewed specialization"
        )
    sites = shape.sites
    capacity = shape.capacity
    state_specs = (
        ("header", "u64", HEADER_WORDS_V2),
        ("occupancy", "u32", sites),
        ("resource", "u16", sites),
        ("patch_field", "u16", sites),
        ("patch_pending", "u16", sites),
        ("wdl_delay_0", "u16", sites),
        ("wdl_delay_1", "u16", sites),
        ("wdl_active", "u16", sites),
        ("jungle", "u8", sites),
        ("eed_field", "u16", sites),
        ("alive", "u8", capacity),
        ("active_slots_by_uid", "u32", capacity),
        ("position", "u32", capacity),
        ("energy", "i32", capacity),
        ("health", "i32", capacity),
        ("age", "u32", capacity),
        ("birth_tick", "u64", capacity),
        ("agent_id", "u64", capacity),
        ("parent_a_id", "u64", capacity),
        ("parent_b_id", "u64", capacity),
        ("genome_a", "u8", capacity * GENOME_LOCI_V2),
        ("genome_b", "u8", capacity * GENOME_LOCI_V2),
        ("expressed", "u8", capacity * GENOME_LOCI_V2),
    )
    # Every term is linear in S or C. Tile summaries are linear in ceil(C/256).
    scratch_specs = (
        ("move_target", "u32", capacity),
        ("move_accepted", "u8", capacity),
        ("move_winner_uid", "u64", sites),
        ("birth_winner_uid", "u64", sites),
        ("gather_amount", "u16", capacity),
        ("mate_slot", "u32", capacity),
        ("spawn_site", "u32", capacity),
        ("birth_accepted", "u8", capacity),
        ("birth_rank", "u32", capacity),
        ("active_slots", "u32", capacity),
        ("free_slots", "u32", capacity),
        ("tile_active_count", "u32", shape.rank_tiles),
        ("tile_free_count", "u32", shape.rank_tiles),
        ("tile_birth_count", "u32", shape.rank_tiles),
        ("world_control", "u64", 32),
    )
    state_fields, state_bytes = _pack_fields(state_specs, label="state")
    scratch_fields, scratch_bytes = _pack_fields(scratch_specs, label="scratch")
    expected_max = checked_add_i64(
        checked_multiply_i64(40, sites, "linear site bound"),
        checked_multiply_i64(176, capacity, "linear capacity bound"),
        "linear layout bound",
    )
    if state_bytes + scratch_bytes > expected_max + 4096:
        raise RuntimeError("workhorse layout escaped its O(S+C) audited bound")
    return WorkhorseLayoutV2(
        shape=shape,
        state_fields=state_fields,
        scratch_fields=scratch_fields,
        state_bytes_per_world=state_bytes,
        scratch_bytes_per_world=scratch_bytes,
    )


@dataclass(frozen=True, slots=True)
class WorkhorseResourceLedgerV2:
    shape_id: str
    B: int
    temporal_k: int
    state_bytes: int
    scratch_bytes: int
    event_bytes: int
    event_count_bytes: int
    checkpoint_canonical_bytes: int
    checkpoint_host_staging_slabs: int
    checkpoint_host_staging_bytes: int
    checkpoint_host_peak_bytes: int
    checkpoint_device_staging_bytes: int
    graph_object_bytes: int
    graph_exec_bytes: int
    allocator_owner_overhead_bytes: int
    driver_unattributed_owner_bytes: int
    process_baseline_bytes: int
    tensor_resident_bytes: int
    incremental_device_bytes: int
    projected_process_peak_bytes: int
    owner_components_receipt_sha256: str

    def __post_init__(self) -> None:
        shape = workhorse_shape_v2(self.shape_id)
        if type(self.B) is not int or not 1 <= self.B <= 65_535:
            raise FullEvolutionWorkhorseContractError(
                "resource ledger B must fit grid.y"
            )
        if self.temporal_k not in shape.temporal_k_candidates:
            raise FullEvolutionWorkhorseContractError(
                "resource ledger temporal K is not compiled"
            )
        for name in (
            "state_bytes",
            "scratch_bytes",
            "event_bytes",
            "event_count_bytes",
            "checkpoint_canonical_bytes",
            "checkpoint_host_staging_bytes",
            "checkpoint_host_peak_bytes",
            "checkpoint_device_staging_bytes",
            "graph_object_bytes",
            "graph_exec_bytes",
            "allocator_owner_overhead_bytes",
            "driver_unattributed_owner_bytes",
            "process_baseline_bytes",
            "tensor_resident_bytes",
            "incremental_device_bytes",
            "projected_process_peak_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise FullEvolutionWorkhorseContractError(
                    f"resource ledger {name} must be a nonnegative integer"
                )
        if self.checkpoint_host_staging_slabs != 2:
            raise FullEvolutionWorkhorseContractError(
                "resource ledger requires exactly two checkpoint staging slabs"
            )
        layout = workhorse_layout_v2(shape)
        expected_state = checked_multiply_i64(
            self.B, layout.state_bytes_per_world, "ledger state bytes"
        )
        expected_scratch = checked_multiply_i64(
            self.B, layout.scratch_bytes_per_world, "ledger scratch bytes"
        )
        expected_events = checked_multiply_i64(
            self.B,
            layout.event_bytes_per_world_tick,
            "ledger single sparse event page bytes",
        )
        expected_count = checked_multiply_i64(self.B, 8, "ledger event counts")
        tensor = 0
        for label, value in (
            ("state", expected_state),
            ("scratch", expected_scratch),
            ("events", expected_events),
            ("event counts", expected_count),
        ):
            tensor = checked_add_i64(tensor, value, f"ledger tensor {label}")
        incremental = tensor
        for label, value in (
            ("checkpoint device", self.checkpoint_device_staging_bytes),
            ("graph object", self.graph_object_bytes),
            ("graph exec", self.graph_exec_bytes),
            ("allocator", self.allocator_owner_overhead_bytes),
            ("driver residual", self.driver_unattributed_owner_bytes),
        ):
            incremental = checked_add_i64(
                incremental, value, f"ledger incremental {label}"
            )
        projected_process_peak = checked_add_i64(
            self.process_baseline_bytes,
            incremental,
            "ledger projected process peak",
        )
        if (
            self.state_bytes != expected_state
            or self.scratch_bytes != expected_scratch
            or self.event_bytes != expected_events
            or self.event_count_bytes != expected_count
            or self.checkpoint_canonical_bytes != expected_state
            or self.checkpoint_host_staging_bytes != 2 * expected_state
            or self.checkpoint_host_peak_bytes != 3 * expected_state
            or self.tensor_resident_bytes != tensor
            or self.incremental_device_bytes != incremental
            or self.projected_process_peak_bytes != projected_process_peak
        ):
            raise FullEvolutionWorkhorseContractError(
                "resource ledger byte topology is internally inconsistent"
            )
        _checked_sha256(
            self.owner_components_receipt_sha256,
            "resource ledger owner component receipt",
        )

    def to_dict(self) -> dict[str, int | str]:
        return {
            "schema_version": "eiel_full_evolution_cuda_workhorse_resource_ledger_v2",
            "shape_id": self.shape_id,
            "B": self.B,
            "temporal_k": self.temporal_k,
            "state_bytes": self.state_bytes,
            "scratch_bytes": self.scratch_bytes,
            "event_bytes": self.event_bytes,
            "event_count_bytes": self.event_count_bytes,
            "checkpoint_canonical_bytes": self.checkpoint_canonical_bytes,
            "checkpoint_host_staging_slabs": self.checkpoint_host_staging_slabs,
            "checkpoint_host_staging_bytes": self.checkpoint_host_staging_bytes,
            "checkpoint_host_peak_bytes": self.checkpoint_host_peak_bytes,
            "checkpoint_device_staging_bytes": self.checkpoint_device_staging_bytes,
            "graph_object_bytes": self.graph_object_bytes,
            "graph_exec_bytes": self.graph_exec_bytes,
            "allocator_owner_overhead_bytes": self.allocator_owner_overhead_bytes,
            "driver_unattributed_owner_bytes": self.driver_unattributed_owner_bytes,
            "process_baseline_bytes": self.process_baseline_bytes,
            "tensor_resident_bytes": self.tensor_resident_bytes,
            "incremental_device_bytes": self.incremental_device_bytes,
            "projected_process_peak_bytes": self.projected_process_peak_bytes,
            "owner_components_receipt_sha256": self.owner_components_receipt_sha256,
            "event_device_pages_per_world": 1,
            "event_page_rows_per_world": workhorse_layout_v2(
                self.shape_id
            ).event_rows_per_world_tick,
            "event_resident_growth_with_temporal_k_bytes": 0,
            "horizon_dependent_resident_bytes": 0,
            "arbitrary_allowance_bytes": 0,
        }


def _checked_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FullEvolutionWorkhorseContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class WorkhorseOwnerComponentsV2:
    """Measured non-tensor allocations for one exact backend/K owner.

    The values are retained rather than hidden behind a safety allowance.  A
    runtime receipt must derive them from the exclusive process/allocator and
    graph-owner measurements made before sealing a candidate envelope.
    """

    backend_id: str
    temporal_k: int
    graph_object_bytes: int
    graph_exec_bytes: int
    allocator_owner_overhead_bytes: int
    driver_unattributed_owner_bytes: int
    process_baseline_bytes: int
    checkpoint_device_staging_bytes_per_world: int
    measurement_receipt_sha256: str

    def __post_init__(self) -> None:
        workhorse_backend_v2(self.backend_id)
        for name in (
            "graph_object_bytes",
            "graph_exec_bytes",
            "allocator_owner_overhead_bytes",
            "driver_unattributed_owner_bytes",
            "process_baseline_bytes",
            "checkpoint_device_staging_bytes_per_world",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise FullEvolutionWorkhorseContractError(
                    f"owner component {name} must be a nonnegative integer"
                )
        if type(self.temporal_k) is not int or self.temporal_k <= 0:
            raise FullEvolutionWorkhorseContractError(
                "owner component temporal_k must be positive"
            )
        if self.backend_id != BACKEND_PHASE_GRAPH_V2 and (
            self.graph_object_bytes or self.graph_exec_bytes
        ):
            raise FullEvolutionWorkhorseContractError(
                "only phase_graph_v2 may own CUDA graph allocations"
            )
        _checked_sha256(
            self.measurement_receipt_sha256, "owner component measurement receipt"
        )

    def to_dict(self) -> dict[str, int | str]:
        return {
            "schema_version": (
                "eiel_full_evolution_cuda_workhorse_owner_components_v2"
            ),
            "backend_id": self.backend_id,
            "temporal_k": self.temporal_k,
            "graph_object_bytes": self.graph_object_bytes,
            "graph_exec_bytes": self.graph_exec_bytes,
            "allocator_owner_overhead_bytes": self.allocator_owner_overhead_bytes,
            "driver_unattributed_owner_bytes": self.driver_unattributed_owner_bytes,
            "process_baseline_bytes": self.process_baseline_bytes,
            "checkpoint_device_staging_bytes_per_world": (
                self.checkpoint_device_staging_bytes_per_world
            ),
            "measurement_receipt_sha256": self.measurement_receipt_sha256,
        }


def estimate_workhorse_resources_v2(
    contract: FullEvolutionCudaWorkhorseContractV2,
    *,
    owner_components: WorkhorseOwnerComponentsV2,
) -> WorkhorseResourceLedgerV2:
    if not isinstance(contract, FullEvolutionCudaWorkhorseContractV2):
        raise FullEvolutionWorkhorseContractError(
            "resource estimation requires an exact V2 contract"
        )
    if type(owner_components) is not WorkhorseOwnerComponentsV2:
        raise FullEvolutionWorkhorseContractError(
            "exact resource estimation requires measured owner components"
        )
    if (
        owner_components.backend_id != contract.backend_id
        or owner_components.temporal_k != contract.temporal_k
    ):
        raise FullEvolutionWorkhorseContractError(
            "owner components do not match the exact backend/K contract"
        )
    layout = workhorse_layout_v2(contract.shape)
    state_bytes = checked_multiply_i64(
        contract.B, layout.state_bytes_per_world, "batched state bytes"
    )
    scratch_bytes = checked_multiply_i64(
        contract.B, layout.scratch_bytes_per_world, "batched scratch bytes"
    )
    event_bytes = checked_multiply_i64(
        contract.B,
        layout.event_bytes_per_world_tick,
        "exact single 2*C*48 sparse event page bytes",
    )
    event_count_bytes = checked_multiply_i64(contract.B, 8, "event count bytes")
    checkpoint_canonical_bytes = state_bytes
    checkpoint_host_staging_slabs = 2
    checkpoint_host_staging_bytes = checked_multiply_i64(
        checkpoint_canonical_bytes,
        checkpoint_host_staging_slabs,
        "two bounded checkpoint host slabs",
    )
    checkpoint_host_peak_bytes = checked_add_i64(
        checkpoint_canonical_bytes,
        checkpoint_host_staging_bytes,
        "canonical checkpoint plus two host staging slabs",
    )
    checkpoint_device_staging_bytes = checked_multiply_i64(
        contract.B,
        owner_components.checkpoint_device_staging_bytes_per_world,
        "checkpoint device staging bytes",
    )
    tensor_resident_bytes = 0
    for label, value in (
        ("state", state_bytes),
        ("scratch", scratch_bytes),
        ("events", event_bytes),
        ("event counts", event_count_bytes),
    ):
        tensor_resident_bytes = checked_add_i64(
            tensor_resident_bytes, value, f"tensor resident after {label}"
        )
    incremental_device_bytes = tensor_resident_bytes
    for label, value in (
        ("checkpoint device staging", checkpoint_device_staging_bytes),
        ("graph object", owner_components.graph_object_bytes),
        ("graph exec", owner_components.graph_exec_bytes),
        ("allocator owner overhead", owner_components.allocator_owner_overhead_bytes),
        (
            "driver-unattributed measured owner",
            owner_components.driver_unattributed_owner_bytes,
        ),
    ):
        incremental_device_bytes = checked_add_i64(
            incremental_device_bytes, value, f"incremental device after {label}"
        )
    projected_process_peak_bytes = checked_add_i64(
        owner_components.process_baseline_bytes,
        incremental_device_bytes,
        "projected process peak",
    )
    return WorkhorseResourceLedgerV2(
        shape_id=contract.shape_id,
        B=contract.B,
        temporal_k=contract.temporal_k,
        state_bytes=state_bytes,
        scratch_bytes=scratch_bytes,
        event_bytes=event_bytes,
        event_count_bytes=event_count_bytes,
        checkpoint_canonical_bytes=checkpoint_canonical_bytes,
        checkpoint_host_staging_slabs=checkpoint_host_staging_slabs,
        checkpoint_host_staging_bytes=checkpoint_host_staging_bytes,
        checkpoint_host_peak_bytes=checkpoint_host_peak_bytes,
        checkpoint_device_staging_bytes=checkpoint_device_staging_bytes,
        graph_object_bytes=owner_components.graph_object_bytes,
        graph_exec_bytes=owner_components.graph_exec_bytes,
        allocator_owner_overhead_bytes=(
            owner_components.allocator_owner_overhead_bytes
        ),
        driver_unattributed_owner_bytes=(
            owner_components.driver_unattributed_owner_bytes
        ),
        process_baseline_bytes=owner_components.process_baseline_bytes,
        tensor_resident_bytes=tensor_resident_bytes,
        incremental_device_bytes=incremental_device_bytes,
        projected_process_peak_bytes=projected_process_peak_bytes,
        owner_components_receipt_sha256=(owner_components.measurement_receipt_sha256),
    )


@dataclass(frozen=True, slots=True)
class WorkhorseVramAdmissionV2:
    admitted: bool
    free_bytes: int
    total_device_bytes: int
    reserve_bytes: int
    usable_bytes: int
    required_bytes: int
    headroom_bytes: int
    reason: str

    def __post_init__(self) -> None:
        for name in (
            "free_bytes",
            "total_device_bytes",
            "reserve_bytes",
            "usable_bytes",
            "required_bytes",
            "headroom_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise FullEvolutionWorkhorseContractError(
                    f"admission {name} must be a nonnegative integer"
                )
        if type(self.admitted) is not bool or self.free_bytes > self.total_device_bytes:
            raise FullEvolutionWorkhorseContractError(
                "admission live-memory observation is invalid"
            )
        expected_usable = max(0, self.free_bytes - self.reserve_bytes)
        expected_admitted = self.required_bytes <= expected_usable
        if (
            self.usable_bytes != expected_usable
            or self.admitted is not expected_admitted
            or self.headroom_bytes != max(0, expected_usable - self.required_bytes)
            or self.reason
            != ("ADMITTED" if expected_admitted else "WAIT_VRAM_HEADROOM")
        ):
            raise FullEvolutionWorkhorseContractError(
                "admission ledger is internally inconsistent"
            )

    def to_dict(self) -> dict[str, int | str | bool]:
        return {
            "schema_version": "eiel_full_evolution_cuda_workhorse_vram_admission_v2",
            "admitted": self.admitted,
            "free_bytes": self.free_bytes,
            "total_device_bytes": self.total_device_bytes,
            "reserve_bytes": self.reserve_bytes,
            "usable_bytes": self.usable_bytes,
            "required_bytes": self.required_bytes,
            "headroom_bytes": self.headroom_bytes,
            "reason": self.reason,
        }


def admit_workhorse_vram_v2(
    contract: FullEvolutionCudaWorkhorseContractV2,
    *,
    free_bytes: int,
    total_device_bytes: int,
    owner_components: WorkhorseOwnerComponentsV2,
    reserve_bytes: int = 0,
) -> WorkhorseVramAdmissionV2:
    for name, value in (
        ("free_bytes", free_bytes),
        ("total_device_bytes", total_device_bytes),
        ("reserve_bytes", reserve_bytes),
    ):
        if type(value) is not int or value < 0:
            raise FullEvolutionWorkhorseContractError(
                f"{name} must be a nonnegative integer"
            )
    if free_bytes > total_device_bytes:
        raise FullEvolutionWorkhorseContractError(
            "free_bytes must not exceed total_device_bytes"
        )
    required = estimate_workhorse_resources_v2(
        contract, owner_components=owner_components
    ).incremental_device_bytes
    fraction_reserve = (
        total_device_bytes * WORKHORSE_MINIMUM_HEADROOM_NUMERATOR_V2
        + WORKHORSE_MINIMUM_HEADROOM_DENOMINATOR_V2
        - 1
    ) // WORKHORSE_MINIMUM_HEADROOM_DENOMINATOR_V2
    effective_reserve = max(reserve_bytes, fraction_reserve)
    usable = max(0, free_bytes - effective_reserve)
    admitted = required <= usable
    return WorkhorseVramAdmissionV2(
        admitted=admitted,
        free_bytes=free_bytes,
        total_device_bytes=total_device_bytes,
        reserve_bytes=effective_reserve,
        usable_bytes=usable,
        required_bytes=required,
        headroom_bytes=max(0, usable - required),
        reason="ADMITTED" if admitted else "WAIT_VRAM_HEADROOM",
    )


def workhorse_topology_resource_binding_v2(
    shape: str | WorkhorseCompileShapeV2,
    *,
    backend_id: str,
    temporal_k: int,
    runtime_inspection: Mapping[str, Any],
    graph_inspection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal compiled resources, launch geometry, and occupancy for one K.

    The binding is deliberately created before candidate generation.  It uses
    compiled function attributes and the driver's blocks-per-SM result rather
    than estimating occupancy from shape alone.
    """
    if isinstance(shape, str):
        shape = workhorse_shape_v2(shape)
    if workhorse_shape_v2(shape.shape_id) != shape:
        raise FullEvolutionWorkhorseContractError("shape must be exact")
    backend = workhorse_backend_v2(backend_id)
    if temporal_k not in shape.temporal_k_candidates:
        raise FullEvolutionWorkhorseContractError("temporal K is not compiled")
    if backend_id == "reference_phased_v2" and temporal_k != 1:
        raise FullEvolutionWorkhorseContractError("reference topology is K1 only")
    runtime = dict(runtime_inspection)
    required_positive = (
        "multiprocessor_count",
        "maximum_grid_x",
        "maximum_grid_y",
        "maximum_threads_per_block",
        "maximum_threads_per_sm",
        "registers_per_sm",
        "shared_memory_per_sm",
        "maximum_blocks_per_sm",
        "warp_size",
    )
    if (
        runtime.get("runtime_inspected") is not True
        or runtime.get("backend_id") != backend_id
        or runtime.get("profile_id") != shape.profile_id
        or runtime.get("route_id") != shape.route_id
        or any(
            type(runtime.get(field)) is not int or runtime[field] <= 0
            for field in required_positive
        )
        or runtime["maximum_threads_per_block"] < 256
        or runtime["warp_size"] <= 0
        or 256 % runtime["warp_size"]
    ):
        raise FullEvolutionWorkhorseContractError(
            "runtime device limits are missing or inconsistent"
        )
    resources = runtime.get("compiled_kernel_resources")
    expected_kernels = backend.topology_payload(shape).get(
        "kernels_per_dispatch"
    ) or backend.topology_payload(shape).get("kernels_per_tick")
    if (
        type(resources) is not list
        or len(resources) != len(expected_kernels)
        or [row.get("kernel") for row in resources if type(row) is dict]
        != list(expected_kernels)
    ):
        raise FullEvolutionWorkhorseContractError(
            "compiled resources do not cover the exact backend kernel sequence"
        )
    resource_rows: list[dict[str, Any]] = []
    regimes: list[dict[str, Any]] = []
    for resource in resources:
        if type(resource) is not dict or set(resource) != {
            "kernel",
            "registers_per_thread",
            "static_shared_bytes",
            "local_bytes_per_thread",
            "maximum_threads_per_block",
            "active_blocks_per_sm_at_compiled_block",
        }:
            raise FullEvolutionWorkhorseContractError(
                "compiled kernel resource row is not exact"
            )
        for field in (
            "registers_per_thread",
            "static_shared_bytes",
            "local_bytes_per_thread",
            "maximum_threads_per_block",
            "active_blocks_per_sm_at_compiled_block",
        ):
            if type(resource[field]) is not int or resource[field] < 0:
                raise FullEvolutionWorkhorseContractError(
                    f"compiled resource {field} is invalid"
                )
        active_blocks = resource["active_blocks_per_sm_at_compiled_block"]
        registers_per_cta = resource["registers_per_thread"] * 256
        theoretical_limits = [
            runtime["maximum_blocks_per_sm"],
            runtime["maximum_threads_per_sm"] // 256,
        ]
        if registers_per_cta:
            theoretical_limits.append(runtime["registers_per_sm"] // registers_per_cta)
        if resource["static_shared_bytes"]:
            theoretical_limits.append(
                runtime["shared_memory_per_sm"] // resource["static_shared_bytes"]
            )
        theoretical_max = min(theoretical_limits)
        if (
            resource["maximum_threads_per_block"] < 256
            or active_blocks <= 0
            or active_blocks > theoretical_max
        ):
            raise FullEvolutionWorkhorseContractError(
                "compiled blocks-per-SM result violates device resource limits"
            )
        kernel = resource["kernel"]
        if backend_id == BACKEND_FUSED_CTA_TEMPORAL_V2:
            ctas_per_world = 1
            occurrences = 1
        else:
            ctas_per_world = (
                1
                if kernel
                in {
                    "compact_birth_events_v3",
                    "compact_death_events_v3",
                    "rank_prefix_phase_v2",
                    "mate_spawn_phase_v2",
                    "publish_phase_v2",
                }
                else (
                    shape.site_tiles
                    if kernel == "environment_phase_v2"
                    else shape.agent_tiles
                )
            )
            occurrences = temporal_k
        resident_ctas = runtime["multiprocessor_count"] * active_blocks
        resident_worlds = max(1, resident_ctas // ctas_per_world)
        resource_rows.append(dict(resource))
        regimes.append(
            {
                "kernel": kernel,
                "ctas_per_world": ctas_per_world,
                "occurrences_per_dispatch": occurrences,
                "ctas_per_world_dispatch": ctas_per_world * occurrences,
                "active_blocks_per_sm": active_blocks,
                "resident_ctas_device": resident_ctas,
                "resident_worlds_per_wave": resident_worlds,
                "first_saturation_B": (resident_ctas + ctas_per_world - 1)
                // ctas_per_world,
                "theoretical_blocks_per_sm_limit": theoretical_max,
            }
        )

    graph_geometry: dict[str, Any] | None = None
    if backend_id == BACKEND_PHASE_GRAPH_V2:
        if type(graph_inspection) is not dict:
            raise FullEvolutionWorkhorseContractError(
                "phase graph topology binding requires captured B1 graph geometry"
            )
        graph = dict(graph_inspection)
        nodes = graph.get("nodes")
        kernels_per_tick = len(backend.topology_payload(shape)["kernels_per_tick"])
        expected_kernel_nodes = kernels_per_tick * temporal_k
        if (
            graph.get("runtime_inspected") is not True
            or graph.get("shape_id") != shape.shape_id
            or graph.get("backend_id") != backend_id
            or graph.get("B") != 1
            or graph.get("temporal_k") != temporal_k
            or graph.get("kernel_node_count") != expected_kernel_nodes
            or graph.get("memset_node_count") != 1
            or graph.get("node_count") != expected_kernel_nodes + 1
            or graph.get("edge_count") != expected_kernel_nodes
            or graph.get("linear_chain") is not True
            or graph.get("kernel_sequence_exact") is not True
            or graph.get("kernel_geometry_exact") is not True
            or type(nodes) is not list
            or len(nodes) != graph["node_count"]
        ):
            raise FullEvolutionWorkhorseContractError(
                "captured graph runtime geometry is invalid"
            )
        expected_sequence = (
            list(backend.topology_payload(shape)["kernels_per_tick"]) * temporal_k
        )
        kernel_nodes = [
            node
            for node in nodes
            if type(node) is dict and node.get("node_type") == "kernel"
        ]
        if (
            len(kernel_nodes) != len(expected_sequence)
            or [node.get("kernel") for node in kernel_nodes] != expected_sequence
            or any(
                type(node.get("block")) not in (list, tuple)
                or tuple(node["block"]) != (256, 1, 1)
                or type(node.get("grid")) not in (list, tuple)
                or len(node["grid"]) != 3
                or tuple(node["grid"])[1:] != (1, 1)
                or node["grid"][0]
                != (
                    1
                    if node["kernel"]
                    in {
                        "compact_birth_events_v3",
                        "compact_death_events_v3",
                        "rank_prefix_phase_v2",
                        "mate_spawn_phase_v2",
                        "publish_phase_v2",
                    }
                    else (
                        shape.site_tiles
                        if node["kernel"] == "environment_phase_v2"
                        else shape.agent_tiles
                    )
                )
                or node["grid"][0] > runtime["maximum_grid_x"]
                for node in kernel_nodes
            )
        ):
            raise FullEvolutionWorkhorseContractError(
                "captured graph node sequence/geometry is not exact"
            )
        graph_geometry = {
            "B": 1,
            "temporal_k": temporal_k,
            "node_count": graph["node_count"],
            "edge_count": graph["edge_count"],
            "kernel_node_count": graph["kernel_node_count"],
            "memset_node_count": graph["memset_node_count"],
            "nodes_sha256": canonical_sha256(nodes),
            "graph_object_device_bytes": graph.get("graph_object_device_bytes"),
            "graph_exec_device_bytes": graph.get("graph_exec_device_bytes"),
        }
        if any(
            type(graph_geometry[field]) is not int or graph_geometry[field] < 0
            for field in ("graph_object_device_bytes", "graph_exec_device_bytes")
        ):
            raise FullEvolutionWorkhorseContractError(
                "captured graph owner bytes are invalid"
            )
    elif graph_inspection is not None:
        raise FullEvolutionWorkhorseContractError(
            "non-graph backend must not supply graph geometry"
        )

    body = {
        "schema_version": "eiel_workhorse_topology_resource_binding_v2",
        "shape_id": shape.shape_id,
        "profile_id": shape.profile_id,
        "route_id": shape.route_id,
        "backend_id": backend_id,
        "temporal_k": temporal_k,
        "threads_per_cta": 256,
        "device_limits": {field: runtime[field] for field in required_positive},
        "compiled_kernel_resources": resource_rows,
        "occupancy_regimes": regimes,
        "maximum_launch_B": min(WORKHORSE_MAX_GRID_Y_V2, runtime["maximum_grid_y"]),
        "graph_geometry": graph_geometry,
    }
    return {**body, "topology_resource_sha256": canonical_sha256(body)}


def validate_workhorse_topology_resource_binding_v2(
    binding: Mapping[str, Any],
    *,
    shape: str | WorkhorseCompileShapeV2,
    backend_id: str,
    temporal_k: int,
) -> dict[str, Any]:
    """Reopen the exact sealed topology/resource projection."""
    if isinstance(shape, str):
        shape = workhorse_shape_v2(shape)
    if type(binding) is not dict:
        raise FullEvolutionWorkhorseContractError(
            "topology/resource binding must be an object"
        )
    body = {
        key: value
        for key, value in binding.items()
        if key != "topology_resource_sha256"
    }
    if (
        set(binding)
        != {
            "schema_version",
            "shape_id",
            "profile_id",
            "route_id",
            "backend_id",
            "temporal_k",
            "threads_per_cta",
            "device_limits",
            "compiled_kernel_resources",
            "occupancy_regimes",
            "maximum_launch_B",
            "graph_geometry",
            "topology_resource_sha256",
        }
        or binding.get("schema_version")
        != "eiel_workhorse_topology_resource_binding_v2"
        or binding.get("shape_id") != shape.shape_id
        or binding.get("profile_id") != shape.profile_id
        or binding.get("route_id") != shape.route_id
        or binding.get("backend_id") != backend_id
        or binding.get("temporal_k") != temporal_k
        or binding.get("threads_per_cta") != 256
        or binding.get("topology_resource_sha256") != canonical_sha256(body)
    ):
        raise FullEvolutionWorkhorseContractError(
            "topology/resource binding identity or seal changed"
        )
    # Re-derive from its persisted minimal runtime and graph projection.  This
    # proves resource changes cannot be hidden behind a copied binding hash.
    runtime = {
        "runtime_inspected": True,
        "backend_id": backend_id,
        "profile_id": shape.profile_id,
        "route_id": shape.route_id,
        **dict(binding["device_limits"]),
        "compiled_kernel_resources": binding["compiled_kernel_resources"],
    }
    graph_geometry = binding["graph_geometry"]
    # Full graph nodes cannot be reconstructed from their digest here, so the
    # creator remains the authority for node content; all admission-driving
    # fields are independently checked below.
    if backend_id == BACKEND_PHASE_GRAPH_V2:
        expected_kernel_nodes = (
            len(
                workhorse_backend_v2(backend_id).topology_payload(shape)[
                    "kernels_per_tick"
                ]
            )
            * temporal_k
        )
        if (
            type(graph_geometry) is not dict
            or graph_geometry.get("B") != 1
            or graph_geometry.get("temporal_k") != temporal_k
            or graph_geometry.get("kernel_node_count") != expected_kernel_nodes
            or graph_geometry.get("memset_node_count") != 1
            or graph_geometry.get("node_count") != expected_kernel_nodes + 1
            or graph_geometry.get("edge_count") != expected_kernel_nodes
            or not _checked_sha256(graph_geometry.get("nodes_sha256"), "graph nodes")
        ):
            raise FullEvolutionWorkhorseContractError(
                "sealed graph geometry is invalid"
            )
    elif graph_geometry is not None:
        raise FullEvolutionWorkhorseContractError(
            "non-graph topology contains graph geometry"
        )
    if (
        type(binding.get("maximum_launch_B")) is not int
        or binding["maximum_launch_B"]
        != min(WORKHORSE_MAX_GRID_Y_V2, runtime["maximum_grid_y"])
        or type(binding.get("occupancy_regimes")) is not list
        or len(binding["occupancy_regimes"])
        != len(binding["compiled_kernel_resources"])
    ):
        raise FullEvolutionWorkhorseContractError(
            "topology admission limits are invalid"
        )
    for resource, regime in zip(
        binding["compiled_kernel_resources"], binding["occupancy_regimes"]
    ):
        active = resource["active_blocks_per_sm_at_compiled_block"]
        kernel = resource["kernel"]
        expected_ctas = (
            1
            if backend_id == BACKEND_FUSED_CTA_TEMPORAL_V2
            or kernel
            in {
                "compact_birth_events_v3",
                "compact_death_events_v3",
                "rank_prefix_phase_v2",
                "mate_spawn_phase_v2",
                "publish_phase_v2",
            }
            else (
                shape.site_tiles
                if kernel == "environment_phase_v2"
                else shape.agent_tiles
            )
        )
        expected_occurrences = (
            1 if backend_id == BACKEND_FUSED_CTA_TEMPORAL_V2 else temporal_k
        )
        ctas = regime.get("ctas_per_world")
        expected_resident_ctas = runtime["multiprocessor_count"] * active
        theoretical_limits = [
            runtime["maximum_blocks_per_sm"],
            runtime["maximum_threads_per_sm"] // 256,
        ]
        registers_per_cta = resource["registers_per_thread"] * 256
        if registers_per_cta:
            theoretical_limits.append(runtime["registers_per_sm"] // registers_per_cta)
        if resource["static_shared_bytes"]:
            theoretical_limits.append(
                runtime["shared_memory_per_sm"] // resource["static_shared_bytes"]
            )
        theoretical_max = min(theoretical_limits)
        if (
            regime.get("kernel") != resource["kernel"]
            or ctas != expected_ctas
            or regime.get("occurrences_per_dispatch") != expected_occurrences
            or regime.get("ctas_per_world_dispatch")
            != expected_ctas * expected_occurrences
            or regime.get("active_blocks_per_sm") != active
            or regime.get("resident_ctas_device") != expected_resident_ctas
            or regime.get("resident_worlds_per_wave")
            != max(1, expected_resident_ctas // ctas)
            or regime.get("first_saturation_B")
            != (expected_resident_ctas + ctas - 1) // ctas
            or regime.get("theoretical_blocks_per_sm_limit") != theoretical_max
            or resource["maximum_threads_per_block"] < 256
            or active <= 0
            or active > theoretical_max
        ):
            raise FullEvolutionWorkhorseContractError(
                "topology occupancy regime does not recompute"
            )
    return dict(binding)


def workhorse_b_candidates_v2(
    shape: str | WorkhorseCompileShapeV2,
    *,
    backend_id: str,
    temporal_k: int,
    free_bytes: int,
    total_device_bytes: int,
    owner_components: WorkhorseOwnerComponentsV2,
    topology_resource_binding: Mapping[str, Any],
    reserve_bytes: int = 0,
) -> tuple[int, ...]:
    """Seal one exact backend/K candidate set around its live VRAM maximum."""
    if isinstance(shape, str):
        shape = workhorse_shape_v2(shape)
    if workhorse_shape_v2(shape.shape_id) != shape:
        raise FullEvolutionWorkhorseContractError("shape must be exact")
    topology = validate_workhorse_topology_resource_binding_v2(
        topology_resource_binding,
        shape=shape,
        backend_id=backend_id,
        temporal_k=temporal_k,
    )
    if owner_components.backend_id != backend_id:
        raise FullEvolutionWorkhorseContractError("backend owner mismatch")
    if owner_components.temporal_k != temporal_k:
        raise FullEvolutionWorkhorseContractError("temporal K owner mismatch")
    graph_geometry = topology["graph_geometry"]
    if backend_id == BACKEND_PHASE_GRAPH_V2:
        if (
            owner_components.graph_object_bytes
            != graph_geometry["graph_object_device_bytes"]
            or owner_components.graph_exec_bytes
            != graph_geometry["graph_exec_device_bytes"]
        ):
            raise FullEvolutionWorkhorseContractError(
                "phase graph owner bytes differ from captured graph geometry"
            )
    elif owner_components.graph_object_bytes or owner_components.graph_exec_bytes:
        raise FullEvolutionWorkhorseContractError(
            "non-graph backend cannot claim graph owner bytes"
        )
    if temporal_k not in shape.temporal_k_candidates:
        raise FullEvolutionWorkhorseContractError("temporal K is not compiled")
    for name, value in (
        ("free_bytes", free_bytes),
        ("total_device_bytes", total_device_bytes),
        ("reserve_bytes", reserve_bytes),
    ):
        if type(value) is not int or value < 0:
            raise FullEvolutionWorkhorseContractError(
                f"{name} must be a nonnegative integer"
            )
    if free_bytes > total_device_bytes:
        raise FullEvolutionWorkhorseContractError(
            "free_bytes must not exceed total_device_bytes"
        )
    fraction_reserve = (
        total_device_bytes * WORKHORSE_MINIMUM_HEADROOM_NUMERATOR_V2
        + WORKHORSE_MINIMUM_HEADROOM_DENOMINATOR_V2
        - 1
    ) // WORKHORSE_MINIMUM_HEADROOM_DENOMINATOR_V2
    effective_reserve = max(reserve_bytes, fraction_reserve)
    usable = max(0, free_bytes - effective_reserve)
    one = FullEvolutionCudaWorkhorseContractV2(
        shape_id=shape.shape_id,
        backend_id=backend_id,
        B=1,
        temporal_k=temporal_k,
        total_ticks=temporal_k,
    )
    one_ledger = estimate_workhorse_resources_v2(one, owner_components=owner_components)
    fixed = checked_add_i64(
        checked_add_i64(
            checked_add_i64(
                owner_components.graph_object_bytes,
                owner_components.graph_exec_bytes,
                "fixed graph owner bytes",
            ),
            owner_components.allocator_owner_overhead_bytes,
            "fixed graph/allocator owner bytes",
        ),
        owner_components.driver_unattributed_owner_bytes,
        "fixed backend owner bytes",
    )
    per_world = one_ledger.incremental_device_bytes - fixed
    maximum_vram = (
        0 if per_world <= 0 or usable < fixed else (usable - fixed) // per_world
    )
    maximum = min(topology["maximum_launch_B"], maximum_vram)
    raw = {
        1,
        2,
        4,
        8,
        16,
        32,
        68,
        136,
        max(1, shape.batch_seed // 2),
        shape.batch_seed,
        shape.batch_seed * 2,
        maximum,
        maximum - 1,
        maximum + 1,
        maximum // 2,
    }
    sm_count = topology["device_limits"]["multiprocessor_count"]
    raw.update({max(1, sm_count // 2), sm_count, sm_count * 2, sm_count * 4})
    for regime in topology["occupancy_regimes"]:
        for wave in {
            regime["resident_worlds_per_wave"],
            regime["first_saturation_B"],
        }:
            raw.update(
                {
                    wave - 1,
                    wave,
                    wave + 1,
                    wave * 2,
                    wave * 4,
                    wave * 8,
                    (maximum // wave) * wave,
                    ((maximum + wave - 1) // wave) * wave,
                }
            )
    return tuple(
        sorted(value for value in raw if 1 <= value <= topology["maximum_launch_B"])
    )


def workhorse_candidate_envelope_v2(
    shape: str | WorkhorseCompileShapeV2,
    *,
    backend_id: str,
    free_bytes: int,
    total_device_bytes: int,
    owner_components_by_k: dict[int, WorkhorseOwnerComponentsV2],
    topology_resource_bindings_by_k: Mapping[int, Mapping[str, Any]],
    live_observation_authority_binding_sha256: str,
    reserve_bytes: int = 0,
) -> dict[str, Any]:
    """Return the complete sealed B/K plan for one exact backend mode."""
    if isinstance(shape, str):
        shape = workhorse_shape_v2(shape)
    expected_k = workhorse_backend_temporal_k_candidates_v2(shape, backend_id)
    if set(owner_components_by_k) != set(expected_k):
        raise FullEvolutionWorkhorseContractError(
            "owner components must cover every compiled temporal K exactly"
        )
    if set(topology_resource_bindings_by_k) != set(expected_k):
        raise FullEvolutionWorkhorseContractError(
            "topology/resource bindings must cover every compiled temporal K exactly"
        )
    _checked_sha256(
        live_observation_authority_binding_sha256,
        "candidate-envelope live observation authority",
    )
    per_k = []
    for temporal_k in expected_k:
        candidates = workhorse_b_candidates_v2(
            shape,
            backend_id=backend_id,
            temporal_k=temporal_k,
            free_bytes=free_bytes,
            total_device_bytes=total_device_bytes,
            owner_components=owner_components_by_k[temporal_k],
            topology_resource_binding=topology_resource_bindings_by_k[temporal_k],
            reserve_bytes=reserve_bytes,
        )
        admissions = []
        for B in candidates:
            contract = FullEvolutionCudaWorkhorseContractV2(
                shape_id=shape.shape_id,
                backend_id=backend_id,
                B=B,
                temporal_k=temporal_k,
                total_ticks=temporal_k,
            )
            admission = admit_workhorse_vram_v2(
                contract,
                free_bytes=free_bytes,
                total_device_bytes=total_device_bytes,
                owner_components=owner_components_by_k[temporal_k],
                reserve_bytes=reserve_bytes,
            )
            admissions.append({"B": B, "admission": admission.to_dict()})
        admitted = [row["B"] for row in admissions if row["admission"]["admitted"]]
        topology = validate_workhorse_topology_resource_binding_v2(
            topology_resource_bindings_by_k[temporal_k],
            shape=shape,
            backend_id=backend_id,
            temporal_k=temporal_k,
        )
        one = FullEvolutionCudaWorkhorseContractV2(
            shape_id=shape.shape_id,
            backend_id=backend_id,
            B=1,
            temporal_k=temporal_k,
            total_ticks=temporal_k,
        )
        one_ledger = estimate_workhorse_resources_v2(
            one, owner_components=owner_components_by_k[temporal_k]
        )
        fixed = (
            owner_components_by_k[temporal_k].graph_object_bytes
            + owner_components_by_k[temporal_k].graph_exec_bytes
            + owner_components_by_k[temporal_k].allocator_owner_overhead_bytes
            + owner_components_by_k[temporal_k].driver_unattributed_owner_bytes
        )
        per_world_bytes = one_ledger.incremental_device_bytes - fixed
        effective_reserve = max(
            reserve_bytes,
            (
                total_device_bytes * WORKHORSE_MINIMUM_HEADROOM_NUMERATOR_V2
                + WORKHORSE_MINIMUM_HEADROOM_DENOMINATOR_V2
                - 1
            )
            // WORKHORSE_MINIMUM_HEADROOM_DENOMINATOR_V2,
        )
        usable = max(0, free_bytes - effective_reserve)
        exact_vram_maximum = (
            0
            if per_world_bytes <= 0 or usable < fixed
            else (usable - fixed) // per_world_bytes
        )
        exact_launch_maximum = min(exact_vram_maximum, topology["maximum_launch_B"])
        per_k.append(
            {
                "temporal_k": temporal_k,
                "B_candidates": list(candidates),
                "exact_algebraic_vram_maximum_B": exact_vram_maximum,
                "exact_launch_admissible_maximum_B": exact_launch_maximum,
                "maximum_vram_admitted_B": max(admitted, default=0),
                "vram_maximum_adjacent_B": sorted(
                    value
                    for value in {
                        exact_launch_maximum - 1,
                        exact_launch_maximum,
                        exact_launch_maximum + 1,
                    }
                    if 1 <= value <= topology["maximum_launch_B"]
                ),
                "topology_resource_binding": topology,
                "rows": admissions,
            }
        )
    body = {
        "schema_version": ("eiel_full_evolution_cuda_workhorse_candidate_envelope_v2"),
        "shape_id": shape.shape_id,
        "backend_id": backend_id,
        "free_bytes": free_bytes,
        "total_device_bytes": total_device_bytes,
        "reserve_bytes_requested": reserve_bytes,
        "live_observation_authority_binding_sha256": (
            live_observation_authority_binding_sha256
        ),
        "temporal_k": list(expected_k),
        "per_k": per_k,
        "complete": True,
    }
    return {**body, "candidate_envelope_sha256": canonical_sha256(body)}


def workhorse_checkpoint_identity_v2(
    contract: FullEvolutionCudaWorkhorseContractV2,
    *,
    checkpoint_at_tick: int,
    artifact_identity_sha256: str,
    state_content_sha256: str,
    event_sink_receipt_path: str,
    attempt_authority_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if type(checkpoint_at_tick) is not int or checkpoint_at_tick < 0:
        raise FullEvolutionWorkhorseContractError(
            "checkpoint_at_tick must be nonnegative"
        )
    if (
        not contract.start_next_tick
        <= checkpoint_at_tick
        <= (contract.start_next_tick + contract.total_ticks)
    ):
        raise FullEvolutionWorkhorseContractError(
            "checkpoint tick lies outside the exact contract interval"
        )
    for label, value in (
        ("artifact_identity_sha256", artifact_identity_sha256),
        ("state_content_sha256", state_content_sha256),
    ):
        _checked_sha256(value, label)
    # Import at the authority boundary to keep layout arithmetic independent of
    # filesystem policy while ensuring a hash string can never stand in for an
    # acknowledged external sink receipt.
    from .authority import (
        _event_attempt_view_v2,
        validate_workhorse_event_sink_receipt_v2,
    )

    attempt_view = _event_attempt_view_v2(attempt_authority_binding, historical=True)
    attempt_binding = attempt_view["binding"]

    sink_ack = validate_workhorse_event_sink_receipt_v2(
        event_sink_receipt_path,
        expected_artifact_identity_sha256=artifact_identity_sha256,
        expected_end_next_absolute_tick=checkpoint_at_tick,
        expected_attempt_authority_binding_sha256=attempt_view["binding_sha256"],
        expected_post_state_sha256=state_content_sha256,
    )
    payload = {
        "schema_version": WORKHORSE_CHECKPOINT_SCHEMA_V2,
        "profile_id": contract.profile_id,
        "route_id": contract.route_id,
        "shape_id": contract.shape_id,
        "semantic_transition_sha256": contract.semantic_transition_sha256,
        "artifact_identity_sha256": artifact_identity_sha256,
        "state_content_sha256": state_content_sha256,
        "attempt_authority_binding": attempt_binding,
        "attempt_authority_binding_sha256": attempt_view["binding_sha256"],
        "attempt_lineage_sha256": attempt_view["lineage_sha256"],
        "rng_identity": sink_ack["rng_identity"],
        "rng_identity_sha256": sink_ack["rng_identity_sha256"],
        "pre_state_sha256": sink_ack["pre_state_sha256"],
        "post_state_sha256": sink_ack["post_state_sha256"],
        "dispatch_identity_sha256": sink_ack["dispatch_identity_sha256"],
        "event_sink_receipt": sink_ack,
        "event_sink_ack_sha256": sink_ack["receipt_file_sha256"],
        "checkpoint_at_tick": checkpoint_at_tick,
        "next_absolute_tick": checkpoint_at_tick,
        "B": contract.B,
        "layout_sha256": workhorse_layout_v2(contract.shape).layout_sha256,
    }
    return {**payload, "checkpoint_identity_sha256": canonical_sha256(payload)}


def workhorse_attempt_identity_v2(
    contract: FullEvolutionCudaWorkhorseContractV2,
    *,
    artifact_sha256: str,
    sealed_artifact_identity_sha256: str,
    device_uuid: str,
    target_sm: str,
    resource_ledger: WorkhorseResourceLedgerV2,
    admission: WorkhorseVramAdmissionV2,
    governing_contracts: dict[str, dict[str, str]],
    attempt_authority_binding: Mapping[str, Any],
    historical_authority: bool = False,
) -> dict[str, Any]:
    _checked_sha256(artifact_sha256, "artifact_sha256")
    _checked_sha256(
        sealed_artifact_identity_sha256,
        "sealed_artifact_identity_sha256",
    )
    for label, value in (("device_uuid", device_uuid), ("target_sm", target_sm)):
        if type(value) is not str or not value.strip() or value != value.strip():
            raise FullEvolutionWorkhorseContractError(
                f"{label} must be a nonempty trimmed string"
            )
    if type(resource_ledger) is not WorkhorseResourceLedgerV2:
        raise FullEvolutionWorkhorseContractError(
            "attempt identity requires the exact resource ledger"
        )
    if type(admission) is not WorkhorseVramAdmissionV2:
        raise FullEvolutionWorkhorseContractError(
            "attempt identity requires the exact live VRAM admission"
        )
    if (
        resource_ledger.shape_id != contract.shape_id
        or resource_ledger.B != contract.B
        or resource_ledger.temporal_k != contract.temporal_k
        or admission.required_bytes != resource_ledger.incremental_device_bytes
    ):
        raise FullEvolutionWorkhorseContractError(
            "attempt resource/admission binding does not match B/K"
        )
    expected_contract_keys = {
        "workhorse_semantic_contract",
        "performance_contract",
    }
    if type(governing_contracts) is not dict or set(governing_contracts) != (
        expected_contract_keys
    ):
        raise FullEvolutionWorkhorseContractError(
            "attempt identity requires the workhorse semantic and performance contracts"
        )
    semantic_binding = governing_contracts["workhorse_semantic_contract"]
    performance_binding = governing_contracts["performance_contract"]
    if (
        type(semantic_binding) is not dict
        or set(semantic_binding)
        != {
            "path",
            "source_sha256",
            "semantic_transition_id",
            "semantic_transition_sha256",
        }
        or semantic_binding.get("semantic_transition_id")
        != contract.semantic_transition_id
        or semantic_binding.get("semantic_transition_sha256")
        != contract.semantic_transition_sha256
        or type(semantic_binding.get("path")) is not str
        or not semantic_binding["path"]
        or type(performance_binding) is not dict
        or set(performance_binding) != {"path", "sha256"}
        or type(performance_binding.get("path")) is not str
        or not performance_binding["path"]
    ):
        raise FullEvolutionWorkhorseContractError(
            "invalid workhorse governing contract binding"
        )
    _checked_sha256(semantic_binding["source_sha256"], "semantic source hash")
    _checked_sha256(
        semantic_binding["semantic_transition_sha256"], "semantic transition hash"
    )
    _checked_sha256(performance_binding["sha256"], "performance contract hash")
    from .authority import workhorse_governing_contract_bindings_v2

    if governing_contracts != workhorse_governing_contract_bindings_v2():
        raise FullEvolutionWorkhorseContractError(
            "attempt governing contract hashes are not current"
        )
    from .attempt_authority import validate_workhorse_attempt_authority_binding_v2

    authority_binding = validate_workhorse_attempt_authority_binding_v2(
        attempt_authority_binding, historical=historical_authority
    )
    authority_request = authority_binding["request"]
    authority_response = authority_binding["response"]
    if (
        authority_request.get("phase")
        not in {"CALIBRATION", "EXECUTION_ROW", "CLOSEOUT"}
        or authority_request.get("contract") != contract.to_dict()
        or authority_request.get("artifact_sha256") != artifact_sha256
        or authority_request.get("sealed_artifact_identity_sha256")
        != sealed_artifact_identity_sha256
        or authority_request.get("device_uuid") != device_uuid
        or admission.free_bytes != authority_response.get("device_free_bytes")
        or admission.total_device_bytes != authority_response.get("device_total_bytes")
    ):
        raise FullEvolutionWorkhorseContractError(
            "attempt identity differs from injected Scheduler/lease authority"
        )
    artifact_identity = workhorse_artifact_identity_v2(
        contract.shape, contract.backend_id
    )
    payload = {
        "schema_version": "eiel_full_evolution_cuda_workhorse_attempt_identity_v2",
        "contract": contract.to_dict(),
        "artifact_identity_sha256": artifact_identity["identity_sha256"],
        "artifact_sha256": artifact_sha256,
        "sealed_artifact_identity_sha256": sealed_artifact_identity_sha256,
        "device_uuid": device_uuid,
        "target_sm": target_sm,
        "B": contract.B,
        "temporal_k": contract.temporal_k,
        "resource_ledger": resource_ledger.to_dict(),
        "live_vram_admission": admission.to_dict(),
        "governing_contracts": governing_contracts,
        "attempt_authority_binding": authority_binding,
        "attempt_authority_binding_sha256": authority_binding[
            "authority_binding_sha256"
        ],
        "attempt_lineage_sha256": authority_binding["attempt_lineage_sha256"],
    }
    return {**payload, "attempt_identity_sha256": canonical_sha256(payload)}


__all__ = [
    "GENOME_LOCI_V2",
    "HEADER_BYTES_V2",
    "HEADER_WORDS_V2",
    "WORKHORSE_CUDA_ALLOCATOR_SEGMENT_QUANTUM_BYTES_V2",
    "WORKHORSE_MINIMUM_HEADROOM_DENOMINATOR_V2",
    "WORKHORSE_MINIMUM_HEADROOM_NUMERATOR_V2",
    "WORKHORSE_OWNER_DEVICE_ALLOCATION_COUNT_V2",
    "WORKHORSE_OWNER_TRANSIENT_ALLOCATION_QUANTA_V2",
    "WorkhorseFieldV2",
    "WorkhorseLayoutV2",
    "WorkhorseOwnerComponentsV2",
    "WorkhorseResourceLedgerV2",
    "WorkhorseVramAdmissionV2",
    "admit_workhorse_vram_v2",
    "estimate_workhorse_resources_v2",
    "workhorse_attempt_identity_v2",
    "workhorse_b_candidates_v2",
    "workhorse_candidate_envelope_v2",
    "workhorse_checkpoint_identity_v2",
    "workhorse_layout_v2",
    "workhorse_allocator_owner_overhead_bound_v2",
    "workhorse_topology_resource_binding_v2",
    "validate_workhorse_topology_resource_binding_v2",
]
