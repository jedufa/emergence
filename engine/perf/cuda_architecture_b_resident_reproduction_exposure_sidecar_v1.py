"""Generic CUDA-resident reproduction exposure ledger and CPU projector.

The sidecar records reproduction attempts and the authoritative eligible-parent
snapshot referenced by each attempt.  It is independent of campaign identity,
generation count, seed set, checkpoint cadence, reproduction policy, and world
shape.  Parent contributions are deliberately projected from successful
attempt rows instead of stored as a second source of truth.

Allocation and restore are fail-closed CUDA operations.  Checkpoint projections
are explicit, hash-bound CPU transport values.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Final, Mapping, Sequence

import torch


SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_resident_reproduction_exposure_sidecar_v2"
)
CHECKPOINT_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_resident_reproduction_exposure_checkpoint_v3"
)
PROJECTOR_SCHEMA_VERSION: Final[str] = (
    "eiel_resident_reproduction_exposure_projector_v1"
)
CONTINUITY_SCHEMA_VERSION: Final[str] = (
    "eiel_resident_reproduction_exposure_continuity_v1"
)
SEGMENT_OUTPUT_SCHEMA_VERSION: Final[str] = (
    "eiel_resident_reproduction_exposure_segment_output_v1"
)
SEGMENT_DRAIN_SCHEMA_VERSION: Final[str] = (
    "eiel_resident_reproduction_exposure_segment_drain_v1"
)

PATH_GENERATION_BOUNDARY_REFILL: Final[int] = 0
PATH_ORDINARY_LIVE_BIRTH: Final[int] = 1
VALID_PATH_CODES: Final[frozenset[int]] = frozenset(
    (PATH_GENERATION_BOUNDARY_REFILL, PATH_ORDINARY_LIVE_BIRTH)
)

NO_PATH: Final[int] = -1
NO_REFERENCE: Final[int] = -1
NO_AGENT_ID: Final[int] = -1
NO_GENERATION: Final[int] = -1
NO_TICK: Final[int] = -1
NO_FAILURE: Final[int] = 0
INACTIVE_FAILURE: Final[int] = -1
PARENT_ROLE_PRIMARY: Final[int] = 0
PARENT_ROLE_MATE: Final[int] = 1
PARENT_ROLE_COUNT: Final[int] = 2
STORAGE_MODE_EXACT_FULL: Final[str] = "exact_full"
STORAGE_MODE_COMPACT_COMPLETE: Final[str] = "compact_complete"
STORAGE_MODE_EXACT_FULL_CODE: Final[int] = 0
STORAGE_MODE_COMPACT_COMPLETE_CODE: Final[int] = 1
STORAGE_MODE_CODES: Final[Mapping[str, int]] = {
    STORAGE_MODE_EXACT_FULL: STORAGE_MODE_EXACT_FULL_CODE,
    STORAGE_MODE_COMPACT_COMPLETE: STORAGE_MODE_COMPACT_COMPLETE_CODE,
}

CELL_STATUS_OK: Final[int] = 0
CELL_STATUS_DISABLED: Final[int] = 1
CELL_STATUS_ATTEMPT_CAPACITY_EXCEEDED: Final[int] = 2
CELL_STATUS_SNAPSHOT_CAPACITY_EXCEEDED: Final[int] = 3
CELL_STATUS_AGGREGATE_CAPACITY_EXCEEDED: Final[int] = 4
CELL_STATUS_AUTHORITATIVE_COMMIT_MISMATCH: Final[int] = 5
FAILURE_PENDING_COMMIT: Final[int] = (1 << 31) - 1
FAILURE_AUTHORITATIVE_COMMIT_ABORTED: Final[int] = (1 << 31) - 2
LIVE_BIRTH_FAILURE_ID_BY_CODE_V1: Final[Mapping[int, str]] = {
    1: "birth_fail_no_space",
    2: "birth_fail_buffer",
    3: "birth_fail_no_free_slot",
    4: "birth_fail_no_mate",
    5: "birth_fail_conception",
    6: "birth_fail_gestation",
    7: "birth_fail_health",
    8: "birth_fail_cooldown",
    9: "birth_fail_no_energy",
    10: "birth_fail_already_mated",
    11: "birth_fail_try_probability",
}
NO_COMMIT_ORDINAL: Final[int] = -1
NO_DONOR_ORDINAL: Final[int] = 255
DONOR_ORDINAL_CAPACITY: Final[int] = 7
VALID_CELL_STATUSES: Final[frozenset[int]] = frozenset(
    (
        CELL_STATUS_OK,
        CELL_STATUS_DISABLED,
        CELL_STATUS_ATTEMPT_CAPACITY_EXCEEDED,
        CELL_STATUS_SNAPSHOT_CAPACITY_EXCEEDED,
        CELL_STATUS_AGGREGATE_CAPACITY_EXCEEDED,
        CELL_STATUS_AUTHORITATIVE_COMMIT_MISMATCH,
    )
)

TENSOR_NAMES_V1: Final[tuple[str, ...]] = (
    "cell_enabled",
    "storage_mode_codes",
    "attempt_paths",
    "attempt_generations",
    "attempt_ticks",
    "attempt_eligible_snapshot_refs",
    "attempt_eligible_counts",
    "attempt_selected_parent_ids",
    "attempt_success",
    "attempt_failure_codes",
    "snapshot_paths",
    "snapshot_generations",
    "snapshot_ticks",
    "snapshot_eligible_counts",
    "snapshot_eligible_parent_ids",
    "aggregate_agent_seen",
    "aggregate_agent_ids",
    "aggregate_primary_opportunity_counts",
    "aggregate_mate_opportunity_counts",
    "attempt_cursors",
    "snapshot_cursors",
    "aggregate_seen_counts",
    "cell_status",
    "attempt_commit_ordinals",
    "attempt_child_agent_ids",
    "attempt_donor_ordinals",
    "attempt_donor_draw_counts",
    "qualification_attempt_capacity_enabled",
    "qualification_effective_attempt_capacity",
    "registered_path_codes",
    "path_rng_counter_states",
    "path_counter_write_counts",
    "path_event_row_write_counts",
    "path_rng_state_write_counts",
)


class CudaResidentReproductionExposureError(ValueError):
    """Raised when reproduction exposure is incomplete or contradictory."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _require_cuda_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise CudaResidentReproductionExposureError(
            "reproduction exposure owner requires an explicit CUDA device"
        )
    if not torch.cuda.is_available():
        raise CudaResidentReproductionExposureError(
            "CUDA is unavailable; CPU fallback is prohibited"
        )
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _copy_cpu_tensor_to_resident_cuda_v1(
    value: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    if value.device.type != "cpu" or device.type != "cuda":
        raise CudaResidentReproductionExposureError(
            "checkpoint transport requires a CPU source and CUDA destination"
        )
    resident = torch.empty_like(value, device=device)
    resident.copy_(value, non_blocking=False)
    return resident


def _tensor_manifest(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().to(device="cpu").contiguous()
    raw = value.numpy().tobytes(order="C")
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


@dataclass(frozen=True)
class ResidentReproductionExposureShapeV1:
    """Declarative B/E/S/C capacities for the resident exposure ABI."""

    batch_cells: int
    attempt_capacity: int
    snapshot_capacity: int
    eligible_capacity: int
    storage_mode: str = STORAGE_MODE_EXACT_FULL
    aggregate_capacity: int = 1
    registered_path_codes: tuple[int, ...] = (
        PATH_GENERATION_BOUNDARY_REFILL,
        PATH_ORDINARY_LIVE_BIRTH,
    )

    def __post_init__(self) -> None:
        values = (
            self.batch_cells,
            self.attempt_capacity,
            self.snapshot_capacity,
            self.eligible_capacity,
            self.aggregate_capacity,
        )
        if any(type(value) is not int or value < 1 for value in values):
            raise CudaResidentReproductionExposureError(
                "reproduction exposure B/E/S/C capacities must be positive integers"
            )
        if self.storage_mode not in STORAGE_MODE_CODES:
            raise CudaResidentReproductionExposureError(
                "reproduction exposure storage mode is unsupported"
            )
        codes = tuple(self.registered_path_codes)
        if (
            not codes
            or len(set(codes)) != len(codes)
            or any(type(code) is not int or code < 0 or code > 127 for code in codes)
        ):
            raise CudaResidentReproductionExposureError(
                "registered reproduction path codes must be unique nonnegative int8 values"
            )
        object.__setattr__(self, "registered_path_codes", codes)

    def payload(self) -> dict[str, Any]:
        return {
            "batch_cells": self.batch_cells,
            "attempt_capacity": self.attempt_capacity,
            "snapshot_capacity": self.snapshot_capacity,
            "eligible_capacity": self.eligible_capacity,
            "storage_mode": self.storage_mode,
            "aggregate_capacity": self.aggregate_capacity,
            "registered_path_codes": list(self.registered_path_codes),
        }


@dataclass(frozen=True)
class ReproductionExposureExistingCountsV1:
    """Optional independent per-cell counters used for exact cross-checking."""

    attempts: tuple[int, ...]
    successes: tuple[int, ...]
    failures: tuple[int, ...]

    def __post_init__(self) -> None:
        lengths = {len(self.attempts), len(self.successes), len(self.failures)}
        if len(lengths) != 1 or any(
            type(value) is not int or value < 0
            for rows in (self.attempts, self.successes, self.failures)
            for value in rows
        ):
            raise CudaResidentReproductionExposureError(
                "existing reproduction counts must be equal-length nonnegative integers"
            )
        if any(
            attempts != successes + failures
            for attempts, successes, failures in zip(
                self.attempts,
                self.successes,
                self.failures,
                strict=True,
            )
        ):
            raise CudaResidentReproductionExposureError(
                "existing attempts must equal successes plus failures"
            )


@dataclass(frozen=True)
class ReproductionExposureAttemptV1:
    cell_index: int
    attempt_index: int
    path: int
    generation: int
    tick: int
    primary_eligible_snapshot_ref: int
    mate_eligible_snapshot_ref: int
    primary_eligible_count: int
    mate_eligible_count: int
    selected_parent_0_id: int | None
    selected_parent_1_id: int | None
    success: bool
    failure_code: int
    commit_ordinal: int | None
    child_agent_id: int | None
    donor_draw_ordinals: tuple[int, ...]


@dataclass(frozen=True)
class ReproductionExposureSnapshotV1:
    cell_index: int
    snapshot_index: int
    path: int
    generation: int
    tick: int
    eligible_parent_ids: tuple[int, ...]


@dataclass(frozen=True)
class ReproductionExposureCellProjectionV1:
    cell_index: int
    enabled: bool
    status: int
    attempt_count: int
    success_count: int
    failure_count: int
    selected_parent_counts: Mapping[int, int]
    parent_contributions: Mapping[int, int]
    primary_eligibility_opportunities: Mapping[int, int]
    mate_eligibility_opportunities: Mapping[int, int]
    path_instrumentation: tuple[Mapping[str, int], ...]


@dataclass(frozen=True)
class ResidentReproductionExposureProjectionV1:
    schema_version: str
    shape: ResidentReproductionExposureShapeV1
    attempts: tuple[ReproductionExposureAttemptV1, ...]
    snapshots: tuple[ReproductionExposureSnapshotV1, ...]
    cells: tuple[ReproductionExposureCellProjectionV1, ...]


@dataclass(frozen=True)
class ReproductionExposureContinuityV1:
    """Bounded checkpoint state committing every previously drained segment."""

    segment_count: int
    cell_chain_sha256: tuple[str, ...]
    cumulative_attempt_counts: tuple[int, ...]
    cumulative_success_counts: tuple[int, ...]
    cumulative_failure_counts: tuple[int, ...]
    cumulative_snapshot_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        rows = (
            self.cell_chain_sha256,
            self.cumulative_attempt_counts,
            self.cumulative_success_counts,
            self.cumulative_failure_counts,
            self.cumulative_snapshot_counts,
        )
        lengths = {len(row) for row in rows}
        if (
            type(self.segment_count) is not int
            or self.segment_count < 0
            or len(lengths) != 1
            or not lengths
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in self.cell_chain_sha256
            )
            or any(
                type(value) is not int or value < 0
                for row in rows[1:]
                for value in row
            )
            or any(
                attempts != successes + failures
                for attempts, successes, failures in zip(
                    self.cumulative_attempt_counts,
                    self.cumulative_success_counts,
                    self.cumulative_failure_counts,
                    strict=True,
                )
            )
        ):
            raise CudaResidentReproductionExposureError(
                "reproduction exposure continuity is invalid"
            )

    @classmethod
    def origin(cls, batch_cells: int) -> "ReproductionExposureContinuityV1":
        if type(batch_cells) is not int or batch_cells < 1:
            raise CudaResidentReproductionExposureError(
                "continuity batch size must be positive"
            )
        zeros = (0,) * batch_cells
        return cls(
            segment_count=0,
            cell_chain_sha256=("0" * 64,) * batch_cells,
            cumulative_attempt_counts=zeros,
            cumulative_success_counts=zeros,
            cumulative_failure_counts=zeros,
            cumulative_snapshot_counts=zeros,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": CONTINUITY_SCHEMA_VERSION,
            "segment_count": self.segment_count,
            "cell_chain_sha256": list(self.cell_chain_sha256),
            "cumulative_attempt_counts": list(self.cumulative_attempt_counts),
            "cumulative_success_counts": list(self.cumulative_success_counts),
            "cumulative_failure_counts": list(self.cumulative_failure_counts),
            "cumulative_snapshot_counts": list(self.cumulative_snapshot_counts),
        }


@dataclass(frozen=True)
class ResidentReproductionExposureSegmentDrainV1:
    """Exact host output plus the bounded empty checkpoint for the next segment."""

    schema_version: str
    segment_output: Mapping[str, Any]
    continuation_checkpoint: Mapping[str, Any]


def _expected_tensor_contract(
    shape: ResidentReproductionExposureShapeV1,
) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    b = shape.batch_cells
    e = shape.attempt_capacity
    s = shape.snapshot_capacity
    c = shape.eligible_capacity
    k = shape.aggregate_capacity
    p = len(shape.registered_path_codes)
    return {
        "cell_enabled": (torch.bool, (b,)),
        "storage_mode_codes": (torch.int8, (b,)),
        "attempt_paths": (torch.int8, (b, e)),
        "attempt_generations": (torch.int64, (b, e)),
        "attempt_ticks": (torch.int64, (b, e)),
        "attempt_eligible_snapshot_refs": (
            torch.int64,
            (b, e, PARENT_ROLE_COUNT),
        ),
        "attempt_eligible_counts": (
            torch.int64,
            (b, e, PARENT_ROLE_COUNT),
        ),
        "attempt_selected_parent_ids": (
            torch.int64,
            (b, e, PARENT_ROLE_COUNT),
        ),
        "attempt_success": (torch.bool, (b, e)),
        "attempt_failure_codes": (torch.int32, (b, e)),
        "snapshot_paths": (torch.int8, (b, s)),
        "snapshot_generations": (torch.int64, (b, s)),
        "snapshot_ticks": (torch.int64, (b, s)),
        "snapshot_eligible_counts": (torch.int64, (b, s)),
        "snapshot_eligible_parent_ids": (torch.int64, (b, s, c)),
        "aggregate_agent_seen": (torch.bool, (b, k)),
        "aggregate_agent_ids": (torch.int64, (b, k)),
        "aggregate_primary_opportunity_counts": (torch.int64, (b, k)),
        "aggregate_mate_opportunity_counts": (torch.int64, (b, k)),
        "attempt_cursors": (torch.int64, (b,)),
        "snapshot_cursors": (torch.int64, (b,)),
        "aggregate_seen_counts": (torch.int64, (b,)),
        "cell_status": (torch.int32, (b,)),
        "attempt_commit_ordinals": (torch.int64, (b, e)),
        "attempt_child_agent_ids": (torch.int64, (b, e)),
        "attempt_donor_ordinals": (
            torch.uint8,
            (b, e, DONOR_ORDINAL_CAPACITY),
        ),
        "attempt_donor_draw_counts": (torch.int64, (b, e)),
        "qualification_attempt_capacity_enabled": (torch.bool, (b,)),
        "qualification_effective_attempt_capacity": (torch.int64, (b,)),
        "registered_path_codes": (torch.int8, (p,)),
        "path_rng_counter_states": (torch.int64, (b, p)),
        "path_counter_write_counts": (torch.int64, (b, p)),
        "path_event_row_write_counts": (torch.int64, (b, p)),
        "path_rng_state_write_counts": (torch.int64, (b, p)),
    }


def _validate_tensors(
    *,
    shape: ResidentReproductionExposureShapeV1,
    tensors: Mapping[str, torch.Tensor],
    require_device_type: str,
) -> None:
    expected = _expected_tensor_contract(shape)
    if set(tensors) != set(TENSOR_NAMES_V1):
        raise CudaResidentReproductionExposureError(
            "reproduction exposure tensor set is incomplete"
        )
    devices = set()
    for name, (dtype, tensor_shape) in expected.items():
        value = tensors[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != dtype
            or tuple(value.shape) != tensor_shape
            or value.requires_grad
            or not value.is_contiguous()
        ):
            raise CudaResidentReproductionExposureError(
                f"{name} contradicts the resident reproduction exposure ABI"
            )
        devices.add(value.device)
    if len(devices) != 1 or next(iter(devices)).type != require_device_type:
        raise CudaResidentReproductionExposureError(
            f"reproduction exposure tensors must share one {require_device_type} device"
        )

    attempt_cursors = tensors["attempt_cursors"]
    snapshot_cursors = tensors["snapshot_cursors"]
    aggregate_seen_counts = tensors["aggregate_seen_counts"]
    if bool(
        (
            (attempt_cursors < 0)
            | (attempt_cursors > shape.attempt_capacity)
            | (snapshot_cursors < 0)
            | (snapshot_cursors > shape.snapshot_capacity)
            | (aggregate_seen_counts < 0)
            | (aggregate_seen_counts > shape.aggregate_capacity)
        )
        .any()
        .item()
    ):
        raise CudaResidentReproductionExposureError(
            "reproduction exposure cursor exceeds its declared capacity"
        )
    storage_mode_codes = tensors["storage_mode_codes"]
    expected_storage_mode = STORAGE_MODE_CODES[shape.storage_mode]
    if bool((storage_mode_codes != expected_storage_mode).any().item()):
        raise CudaResidentReproductionExposureError(
            "storage mode tensor differs from the checkpoint-bound shape"
        )
    status = tensors["cell_status"]
    valid_status = torch.zeros_like(status, dtype=torch.bool)
    for code in VALID_CELL_STATUSES:
        valid_status |= status == code
    if bool((~valid_status).any().item()):
        raise CudaResidentReproductionExposureError(
            "reproduction exposure cell status is unknown"
        )
    if tuple(tensors["registered_path_codes"].tolist()) != shape.registered_path_codes:
        raise CudaResidentReproductionExposureError(
            "resident path-code axis differs from its checkpoint-bound shape"
        )
    for name in (
        "path_rng_counter_states",
        "path_counter_write_counts",
        "path_event_row_write_counts",
        "path_rng_state_write_counts",
    ):
        if bool((tensors[name] < 0).any().item()):
            raise CudaResidentReproductionExposureError(
                f"{name} cannot contain negative instrumentation values"
            )

    enabled = tensors["cell_enabled"]
    qualification_enabled = tensors["qualification_attempt_capacity_enabled"]
    qualification_capacity = tensors[
        "qualification_effective_attempt_capacity"
    ]
    if bool(
        (
            (qualification_capacity < 0)
            | (qualification_capacity > shape.attempt_capacity)
            | (
                qualification_enabled
                & (qualification_capacity >= shape.attempt_capacity)
            )
        )
        .any()
        .item()
    ):
        raise CudaResidentReproductionExposureError(
            "qualification attempt capacity is outside the declared envelope"
        )
    if bool((qualification_enabled & ~enabled).any().item()):
        raise CudaResidentReproductionExposureError(
            "qualification attempt capacity requires an enabled cell"
        )
    disabled = ~enabled
    if bool(
        (
            disabled
            & (
                (attempt_cursors != 0)
                | (snapshot_cursors != 0)
                | (aggregate_seen_counts != 0)
                | (status != CELL_STATUS_DISABLED)
            )
        )
        .any()
        .item()
    ) or bool((enabled & (status == CELL_STATUS_DISABLED)).any().item()):
        raise CudaResidentReproductionExposureError(
            "disabled cell cursor or status contradicts sentinel semantics"
        )

    attempt_rows = (
        torch.arange(shape.attempt_capacity, device=attempt_cursors.device)[None, :]
        < attempt_cursors[:, None]
    )
    snapshot_rows = (
        torch.arange(shape.snapshot_capacity, device=snapshot_cursors.device)[None, :]
        < snapshot_cursors[:, None]
    )
    attempt_paths = tensors["attempt_paths"]
    snapshot_paths = tensors["snapshot_paths"]
    valid_attempt_paths = torch.zeros_like(attempt_paths, dtype=torch.bool)
    valid_snapshot_paths = torch.zeros_like(snapshot_paths, dtype=torch.bool)
    for path_code in shape.registered_path_codes:
        valid_attempt_paths |= attempt_paths == path_code
        valid_snapshot_paths |= snapshot_paths == path_code
    if bool((attempt_rows & ~valid_attempt_paths).any().item()) or bool(
        ((~attempt_rows) & (attempt_paths != NO_PATH)).any().item()
    ):
        raise CudaResidentReproductionExposureError("attempt path rows are invalid")
    if bool((snapshot_rows & ~valid_snapshot_paths).any().item()) or bool(
        ((~snapshot_rows) & (snapshot_paths != NO_PATH)).any().item()
    ):
        raise CudaResidentReproductionExposureError("snapshot path rows are invalid")

    snapshot_refs = tensors["attempt_eligible_snapshot_refs"]
    safe_refs = snapshot_refs.clamp(min=0, max=shape.snapshot_capacity - 1)
    referenced_paths = snapshot_paths[:, :, None].expand(
        -1, -1, PARENT_ROLE_COUNT
    ).gather(1, safe_refs)
    inactive_reference_drift = bool(
        ((~attempt_rows[:, :, None]) & (snapshot_refs != NO_REFERENCE))
        .any()
        .item()
    )
    if shape.storage_mode == STORAGE_MODE_EXACT_FULL:
        invalid_reference = bool(
            (
                attempt_rows[:, :, None]
                & (
                    (snapshot_refs < 0)
                    | (snapshot_refs >= snapshot_cursors[:, None, None])
                    | (referenced_paths != attempt_paths[:, :, None])
                )
            )
            .any()
            .item()
        )
    else:
        invalid_reference = bool(
            (
                attempt_rows[:, :, None]
                & (snapshot_refs != NO_REFERENCE)
            )
            .any()
            .item()
        ) or bool((snapshot_cursors != 0).any().item())
    if invalid_reference or inactive_reference_drift:
        raise CudaResidentReproductionExposureError(
            "attempt role-specific eligible snapshot reference is invalid"
        )

    attempt_generations = tensors["attempt_generations"]
    attempt_ticks = tensors["attempt_ticks"]
    attempt_eligible_counts = tensors["attempt_eligible_counts"]
    if bool(
        (
            attempt_rows
            & ((attempt_generations < 0) | (attempt_ticks < 0))
        ).any().item()
    ) or bool(
        (
            (~attempt_rows)
            & (
                (attempt_generations != NO_GENERATION)
                | (attempt_ticks != NO_TICK)
            )
        ).any().item()
    ) or bool(
        (
            attempt_rows[:, :, None]
            & (attempt_eligible_counts < 0)
        ).any().item()
    ) or bool(
        (
            attempt_rows
            & (
                attempt_eligible_counts[
                    :, :, PARENT_ROLE_PRIMARY
                ] < 1
            )
        ).any().item()
    ) or bool(
        (
            (~attempt_rows[:, :, None])
            & (attempt_eligible_counts != 0)
        ).any().item()
    ):
        raise CudaResidentReproductionExposureError(
            "attempt coordinates or eligible counts are invalid"
        )

    generations = tensors["snapshot_generations"]
    ticks = tensors["snapshot_ticks"]
    counts = tensors["snapshot_eligible_counts"]
    if bool(
        (
            snapshot_rows
            & (
                (generations < 0)
                | (ticks < 0)
                | (counts < 0)
                | (counts > shape.eligible_capacity)
            )
        )
        .any()
        .item()
    ) or bool(
        (
            (~snapshot_rows)
            & (
                (generations != NO_GENERATION)
                | (ticks != NO_TICK)
                | (counts != 0)
            )
        )
        .any()
        .item()
    ):
        raise CudaResidentReproductionExposureError(
            "eligible snapshot coordinates or counts are invalid"
        )
    eligible_ids = tensors["snapshot_eligible_parent_ids"]
    validation_column_chunk = 8
    for column_start in range(0, shape.eligible_capacity, validation_column_chunk):
        column_end = min(
            shape.eligible_capacity,
            column_start + validation_column_chunk,
        )
        columns = torch.arange(
            column_start,
            column_end,
            device=counts.device,
        )[None, None, :]
        eligible_columns = (
            (columns < counts[:, :, None]) & snapshot_rows[:, :, None]
        )
        ids = eligible_ids[:, :, column_start:column_end]
        if bool((eligible_columns & (ids < 0)).any().item()) or bool(
            ((~eligible_columns) & (ids != NO_AGENT_ID)).any().item()
        ):
            raise CudaResidentReproductionExposureError(
                "eligible parent ID rows contradict snapshot counts"
            )
    for cell in range(shape.batch_cells):
        for snapshot in range(int(snapshot_cursors[cell].item())):
            count = int(counts[cell, snapshot].item())
            values = eligible_ids[cell, snapshot, :count]
            if int(torch.unique(values).numel()) != count:
                raise CudaResidentReproductionExposureError(
                    "eligible parent IDs must be unique within a snapshot"
                )
    if shape.storage_mode == STORAGE_MODE_EXACT_FULL:
        for cell in range(shape.batch_cells):
            for attempt in range(int(attempt_cursors[cell].item())):
                for role in (PARENT_ROLE_PRIMARY, PARENT_ROLE_MATE):
                    ref = int(snapshot_refs[cell, attempt, role].item())
                    if (
                        int(attempt_eligible_counts[cell, attempt, role].item())
                        != int(counts[cell, ref].item())
                    ):
                        raise CudaResidentReproductionExposureError(
                            "attempt eligible count differs from exact snapshot"
                        )

    aggregate_seen = tensors["aggregate_agent_seen"]
    aggregate_ids = tensors["aggregate_agent_ids"]
    aggregate_primary = tensors["aggregate_primary_opportunity_counts"]
    aggregate_mate = tensors["aggregate_mate_opportunity_counts"]
    for cell in range(shape.batch_cells):
        seen_count = int(aggregate_seen_counts[cell].item())
        active_ids = aggregate_ids[cell, aggregate_seen[cell]]
        if (
            bool((active_ids < 0).any().item())
            or int(torch.unique(active_ids).numel()) != seen_count
            or bool(
                (
                    (~aggregate_seen[cell])
                    & (aggregate_ids[cell] != NO_AGENT_ID)
                )
                .any()
                .item()
            )
            or int(aggregate_seen[cell].sum(dtype=torch.int64).item())
            != seen_count
            or bool(
                (
                    aggregate_seen[cell]
                    & (
                        (aggregate_primary[cell] < 0)
                        | (aggregate_mate[cell] < 0)
                    )
                )
                .any()
                .item()
            )
            or bool(
                (
                    (~aggregate_seen[cell])
                    & (
                        (aggregate_primary[cell] != 0)
                        | (aggregate_mate[cell] != 0)
                    )
                )
                .any()
                .item()
            )
        ):
            raise CudaResidentReproductionExposureError(
                "eligibility opportunity aggregate rows are invalid"
            )
    if shape.storage_mode == STORAGE_MODE_EXACT_FULL and bool(
        (aggregate_seen_counts != 0).any().item()
    ):
        raise CudaResidentReproductionExposureError(
            "exact-full exposure must not write compact aggregates"
        )
    if shape.storage_mode == STORAGE_MODE_COMPACT_COMPLETE:
        for cell in range(shape.batch_cells):
            if int(status[cell].item()) != CELL_STATUS_OK:
                continue
            attempts = int(attempt_cursors[cell].item())
            if (
                int(aggregate_primary[cell].sum().item())
                != int(
                    attempt_eligible_counts[
                        cell, :attempts, PARENT_ROLE_PRIMARY
                    ].sum().item()
                )
                or int(aggregate_mate[cell].sum().item())
                != int(
                    attempt_eligible_counts[
                        cell, :attempts, PARENT_ROLE_MATE
                    ].sum().item()
                )
            ):
                raise CudaResidentReproductionExposureError(
                    "compact eligibility aggregates contradict attempt counts"
                )
    selected_ids = tensors["attempt_selected_parent_ids"]
    selected_0 = selected_ids[:, :, PARENT_ROLE_PRIMARY]
    selected_1 = selected_ids[:, :, PARENT_ROLE_MATE]
    success = tensors["attempt_success"]
    failures = tensors["attempt_failure_codes"]
    if bool(((~attempt_rows) & success).any().item()) or bool(
        (
            (~attempt_rows[:, :, None])
            & (selected_ids != NO_AGENT_ID)
        ).any().item()
    ) or bool(
        ((~attempt_rows) & (failures != INACTIVE_FAILURE)).any().item()
    ):
        raise CudaResidentReproductionExposureError(
            "inactive attempt rows do not use exact sentinels"
        )
    if bool((attempt_rows & success & (failures != NO_FAILURE)).any().item()) or bool(
        (attempt_rows & ~success & (failures <= NO_FAILURE)).any().item()
    ):
        raise CudaResidentReproductionExposureError(
            "attempt success and failure code contradict each other"
        )
    commit_ordinals = tensors["attempt_commit_ordinals"]
    child_ids = tensors["attempt_child_agent_ids"]
    donor_counts = tensors["attempt_donor_draw_counts"]
    donor_ordinals = tensors["attempt_donor_ordinals"]
    pending = attempt_rows & (failures == FAILURE_PENDING_COMMIT)
    committed = attempt_rows & success
    uncommitted = attempt_rows & ~success
    donor_columns = (
        torch.arange(DONOR_ORDINAL_CAPACITY, device=donor_counts.device)[
            None, None, :
        ]
        < donor_counts[:, :, None]
    ) & committed[:, :, None]
    if (
        bool(
            (
                (~attempt_rows)
                & (
                    (commit_ordinals != NO_COMMIT_ORDINAL)
                    | (child_ids != NO_AGENT_ID)
                    | (donor_counts != 0)
                )
            ).any().item()
        )
        or bool(
            (
                committed
                & (
                    (commit_ordinals < 0)
                    | (child_ids < 0)
                    | ~((donor_counts == 0) | (donor_counts == 7))
                )
            ).any().item()
        )
        or bool(
            (
                uncommitted
                & (
                    (child_ids != NO_AGENT_ID)
                    | (donor_counts != 0)
                    | (~pending & (commit_ordinals != NO_COMMIT_ORDINAL))
                    | (pending & (commit_ordinals < 0))
                )
            ).any().item()
        )
        or bool((donor_columns & (donor_ordinals > 1)).any().item())
        or bool(
            (
                (~donor_columns)
                & (donor_ordinals != NO_DONOR_ORDINAL)
            ).any().item()
        )
    ):
        raise CudaResidentReproductionExposureError(
            "attempt authoritative commit fields are invalid"
        )
    if bool(
        (
            attempt_rows
            & success
            & (
                (selected_0 < 0)
                | (selected_1 < 0)
                | (selected_0 == selected_1)
            )
        )
        .any()
        .item()
    ):
        raise CudaResidentReproductionExposureError(
            "successful attempt requires two distinct selected parent IDs"
        )
    for cell in range(shape.batch_cells):
        compact_opportunities = {
            int(aggregate_ids[cell, row].item()): (
                int(aggregate_primary[cell, row].item()),
                int(aggregate_mate[cell, row].item()),
            )
            for row in torch.nonzero(
                aggregate_seen[cell],
                as_tuple=False,
            )
            .flatten()
            .tolist()
        }
        for attempt in range(int(attempt_cursors[cell].item())):
            for role in (PARENT_ROLE_PRIMARY, PARENT_ROLE_MATE):
                selected = int(selected_ids[cell, attempt, role].item())
                if selected == NO_AGENT_ID:
                    continue
                if shape.storage_mode == STORAGE_MODE_EXACT_FULL:
                    ref = int(snapshot_refs[cell, attempt, role].item())
                    count = int(counts[cell, ref].item())
                    eligible = set(eligible_ids[cell, ref, :count].tolist())
                    present = selected in eligible
                else:
                    opportunities = compact_opportunities.get(selected)
                    present = (
                        opportunities is not None
                        and opportunities[role] > 0
                    )
                if not present:
                    raise CudaResidentReproductionExposureError(
                        "selected parent is absent from its role-specific eligible set"
                    )


def _projection_digest(
    *,
    shape: ResidentReproductionExposureShapeV1,
    tensors: Mapping[str, torch.Tensor],
    continuity: ReproductionExposureContinuityV1,
) -> tuple[str, dict[str, dict[str, Any]]]:
    manifests = {name: _tensor_manifest(tensors[name]) for name in TENSOR_NAMES_V1}
    digest = _canonical_sha256(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "owner_schema_version": SCHEMA_VERSION,
            "shape": shape.payload(),
            "continuity": continuity.payload(),
            "tensor_manifests": manifests,
        }
    )
    return digest, manifests


@dataclass(slots=True)
class CudaResidentReproductionExposureSidecarV1:
    """Typed CUDA owner for reproduction attempts and eligible-set snapshots."""

    shape: ResidentReproductionExposureShapeV1
    cell_enabled: torch.Tensor
    storage_mode_codes: torch.Tensor
    attempt_paths: torch.Tensor
    attempt_generations: torch.Tensor
    attempt_ticks: torch.Tensor
    attempt_eligible_snapshot_refs: torch.Tensor
    attempt_eligible_counts: torch.Tensor
    attempt_selected_parent_ids: torch.Tensor
    attempt_success: torch.Tensor
    attempt_failure_codes: torch.Tensor
    snapshot_paths: torch.Tensor
    snapshot_generations: torch.Tensor
    snapshot_ticks: torch.Tensor
    snapshot_eligible_counts: torch.Tensor
    snapshot_eligible_parent_ids: torch.Tensor
    aggregate_agent_seen: torch.Tensor
    aggregate_agent_ids: torch.Tensor
    aggregate_primary_opportunity_counts: torch.Tensor
    aggregate_mate_opportunity_counts: torch.Tensor
    attempt_cursors: torch.Tensor
    snapshot_cursors: torch.Tensor
    aggregate_seen_counts: torch.Tensor
    cell_status: torch.Tensor
    attempt_commit_ordinals: torch.Tensor
    attempt_child_agent_ids: torch.Tensor
    attempt_donor_ordinals: torch.Tensor
    attempt_donor_draw_counts: torch.Tensor
    qualification_attempt_capacity_enabled: torch.Tensor
    qualification_effective_attempt_capacity: torch.Tensor
    registered_path_codes: torch.Tensor
    path_rng_counter_states: torch.Tensor
    path_counter_write_counts: torch.Tensor
    path_event_row_write_counts: torch.Tensor
    path_rng_state_write_counts: torch.Tensor
    continuity: ReproductionExposureContinuityV1

    @property
    def device(self) -> torch.device:
        return self.cell_enabled.device

    def tensor_map(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in TENSOR_NAMES_V1}

    def validate_exact(self) -> None:
        _validate_tensors(
            shape=self.shape,
            tensors=self.tensor_map(),
            require_device_type="cuda",
        )
        if (
            not isinstance(self.continuity, ReproductionExposureContinuityV1)
            or len(self.continuity.cell_chain_sha256) != self.shape.batch_cells
        ):
            raise CudaResidentReproductionExposureError(
                "reproduction exposure continuity does not match B"
            )

    def arm_qualification_attempt_capacity_v1(
        self,
        *,
        target_cell_index: int,
        effective_capacity: int,
    ) -> None:
        """Arm one default-off, resident attempt-capacity qualification fault."""

        self.validate_exact()
        if (
            type(target_cell_index) is not int
            or target_cell_index < 0
            or target_cell_index >= self.shape.batch_cells
            or type(effective_capacity) is not int
            or effective_capacity < 0
            or effective_capacity >= self.shape.attempt_capacity
        ):
            raise CudaResidentReproductionExposureError(
                "qualification attempt-capacity target is outside the resident envelope"
            )
        if (
            not bool(self.cell_enabled[target_cell_index].item())
            or int(self.cell_status[target_cell_index].item()) != CELL_STATUS_OK
            or bool(self.qualification_attempt_capacity_enabled.any().item())
        ):
            raise CudaResidentReproductionExposureError(
                "qualification attempt-capacity fault requires one fresh enabled target"
            )
        self.qualification_effective_attempt_capacity[target_cell_index] = (
            effective_capacity
        )
        self.qualification_attempt_capacity_enabled[target_cell_index] = True
        self.validate_exact()

    def export_checkpoint_projection(self) -> dict[str, Any]:
        self.validate_exact()
        tensors = {
            name: value.detach().to(device="cpu").contiguous().clone()
            for name, value in self.tensor_map().items()
        }
        digest, manifests = _projection_digest(
            shape=self.shape,
            tensors=tensors,
            continuity=self.continuity,
        )
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "owner_schema_version": SCHEMA_VERSION,
            "shape": self.shape.payload(),
            "continuity": self.continuity.payload(),
            "tensors": tensors,
            "tensor_manifests": manifests,
            "projection_sha256": digest,
        }

    def drain_segment(self) -> ResidentReproductionExposureSegmentDrainV1:
        """Synchronize, project, and reset raw buffers for the next segment."""

        drained = drain_resident_reproduction_exposure_checkpoint_v1(
            self.export_checkpoint_projection()
        )
        shape, tensors, continuity = _validated_checkpoint_transport(
            drained.continuation_checkpoint
        )
        if shape != self.shape:
            raise CudaResidentReproductionExposureError(
                "drained reproduction exposure shape changed"
            )
        for name, value in tensors.items():
            getattr(self, name).copy_(value, non_blocking=False)
        self.continuity = continuity
        self.validate_exact()
        return drained

    @classmethod
    def from_checkpoint_projection(
        cls,
        projection: Mapping[str, Any],
        *,
        device: str | torch.device,
    ) -> "CudaResidentReproductionExposureSidecarV1":
        resolved = _require_cuda_device(device)
        shape, tensors, continuity = _validated_checkpoint_transport(projection)
        owner = cls(
            shape=shape,
            continuity=continuity,
            **{
                name: _copy_cpu_tensor_to_resident_cuda_v1(
                    value,
                    device=resolved,
                )
                for name, value in tensors.items()
            },
        )
        owner.validate_exact()
        return owner


def _validated_checkpoint_transport(
    projection: Mapping[str, Any],
) -> tuple[
    ResidentReproductionExposureShapeV1,
    dict[str, torch.Tensor],
    ReproductionExposureContinuityV1,
]:
    if (
        not isinstance(projection, Mapping)
        or projection.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or projection.get("owner_schema_version") != SCHEMA_VERSION
        or not isinstance(projection.get("shape"), Mapping)
        or not isinstance(projection.get("continuity"), Mapping)
        or not isinstance(projection.get("tensors"), Mapping)
    ):
        raise CudaResidentReproductionExposureError(
            "reproduction exposure checkpoint identity is invalid"
        )
    try:
        shape = ResidentReproductionExposureShapeV1(
            **{
                name: projection["shape"][name]
                for name in (
                    "batch_cells",
                    "attempt_capacity",
                    "snapshot_capacity",
                    "eligible_capacity",
                    "storage_mode",
                    "aggregate_capacity",
                    "registered_path_codes",
                )
            }
        )
        tensors = {
            name: projection["tensors"][name] for name in TENSOR_NAMES_V1
        }
        continuity_payload = projection["continuity"]
        if continuity_payload.get("schema_version") != CONTINUITY_SCHEMA_VERSION:
            raise KeyError("continuity.schema_version")
        continuity = ReproductionExposureContinuityV1(
            segment_count=continuity_payload["segment_count"],
            cell_chain_sha256=tuple(continuity_payload["cell_chain_sha256"]),
            cumulative_attempt_counts=tuple(
                continuity_payload["cumulative_attempt_counts"]
            ),
            cumulative_success_counts=tuple(
                continuity_payload["cumulative_success_counts"]
            ),
            cumulative_failure_counts=tuple(
                continuity_payload["cumulative_failure_counts"]
            ),
            cumulative_snapshot_counts=tuple(
                continuity_payload["cumulative_snapshot_counts"]
            ),
        )
    except (KeyError, TypeError) as exc:
        raise CudaResidentReproductionExposureError(
            "reproduction exposure checkpoint is incomplete"
        ) from exc
    _validate_tensors(shape=shape, tensors=tensors, require_device_type="cpu")
    if len(continuity.cell_chain_sha256) != shape.batch_cells:
        raise CudaResidentReproductionExposureError(
            "reproduction exposure continuity does not match B"
        )
    digest, manifests = _projection_digest(
        shape=shape,
        tensors=tensors,
        continuity=continuity,
    )
    if (
        projection.get("tensor_manifests") != manifests
        or projection.get("projection_sha256") != digest
    ):
        raise CudaResidentReproductionExposureError(
            "reproduction exposure checkpoint hash mismatch"
        )
    return shape, tensors, continuity


def allocate_cuda_resident_reproduction_exposure_sidecar_v1(
    shape: ResidentReproductionExposureShapeV1,
    *,
    cell_enabled: Sequence[bool] | torch.Tensor,
    device: str | torch.device,
) -> CudaResidentReproductionExposureSidecarV1:
    """Allocate sentinel-filled resident tensors for a native producer."""

    if not isinstance(shape, ResidentReproductionExposureShapeV1):
        raise CudaResidentReproductionExposureError(
            "reproduction exposure shape must use the typed B/E/S/C contract"
        )
    resolved = _require_cuda_device(device)
    if isinstance(cell_enabled, torch.Tensor):
        if (
            cell_enabled.dtype != torch.bool
            or tuple(cell_enabled.shape) != (shape.batch_cells,)
        ):
            raise CudaResidentReproductionExposureError(
                "cell_enabled must be a bool vector matching B"
            )
        enabled = cell_enabled.to(device=resolved).contiguous().clone()
    else:
        values = tuple(cell_enabled)
        if len(values) != shape.batch_cells or any(type(value) is not bool for value in values):
            raise CudaResidentReproductionExposureError(
                "cell_enabled must contain one bool per batch cell"
            )
        enabled = torch.tensor(values, dtype=torch.bool, device=resolved)

    b = shape.batch_cells
    e = shape.attempt_capacity
    s = shape.snapshot_capacity
    c = shape.eligible_capacity
    k = shape.aggregate_capacity
    owner = CudaResidentReproductionExposureSidecarV1(
        shape=shape,
        cell_enabled=enabled,
        storage_mode_codes=torch.full(
            (b,),
            STORAGE_MODE_CODES[shape.storage_mode],
            dtype=torch.int8,
            device=resolved,
        ),
        attempt_paths=torch.full((b, e), NO_PATH, dtype=torch.int8, device=resolved),
        attempt_generations=torch.full(
            (b, e), NO_GENERATION, dtype=torch.int64, device=resolved
        ),
        attempt_ticks=torch.full(
            (b, e), NO_TICK, dtype=torch.int64, device=resolved
        ),
        attempt_eligible_snapshot_refs=torch.full(
            (b, e, PARENT_ROLE_COUNT),
            NO_REFERENCE,
            dtype=torch.int64,
            device=resolved,
        ),
        attempt_eligible_counts=torch.zeros(
            (b, e, PARENT_ROLE_COUNT),
            dtype=torch.int64,
            device=resolved,
        ),
        attempt_selected_parent_ids=torch.full(
            (b, e, PARENT_ROLE_COUNT),
            NO_AGENT_ID,
            dtype=torch.int64,
            device=resolved,
        ),
        attempt_success=torch.zeros((b, e), dtype=torch.bool, device=resolved),
        attempt_failure_codes=torch.full(
            (b, e), INACTIVE_FAILURE, dtype=torch.int32, device=resolved
        ),
        snapshot_paths=torch.full(
            (b, s), NO_PATH, dtype=torch.int8, device=resolved
        ),
        snapshot_generations=torch.full(
            (b, s), NO_GENERATION, dtype=torch.int64, device=resolved
        ),
        snapshot_ticks=torch.full(
            (b, s), NO_TICK, dtype=torch.int64, device=resolved
        ),
        snapshot_eligible_counts=torch.zeros(
            (b, s), dtype=torch.int64, device=resolved
        ),
        snapshot_eligible_parent_ids=torch.full(
            (b, s, c), NO_AGENT_ID, dtype=torch.int64, device=resolved
        ),
        aggregate_agent_seen=torch.zeros(
            (b, k), dtype=torch.bool, device=resolved
        ),
        aggregate_agent_ids=torch.full(
            (b, k), NO_AGENT_ID, dtype=torch.int64, device=resolved
        ),
        aggregate_primary_opportunity_counts=torch.zeros(
            (b, k), dtype=torch.int64, device=resolved
        ),
        aggregate_mate_opportunity_counts=torch.zeros(
            (b, k), dtype=torch.int64, device=resolved
        ),
        attempt_cursors=torch.zeros((b,), dtype=torch.int64, device=resolved),
        snapshot_cursors=torch.zeros((b,), dtype=torch.int64, device=resolved),
        aggregate_seen_counts=torch.zeros(
            (b,), dtype=torch.int64, device=resolved
        ),
        cell_status=torch.where(
            enabled,
            torch.full((b,), CELL_STATUS_OK, dtype=torch.int32, device=resolved),
            torch.full(
                (b,), CELL_STATUS_DISABLED, dtype=torch.int32, device=resolved
            ),
        ),
        attempt_commit_ordinals=torch.full(
            (b, e), NO_COMMIT_ORDINAL, dtype=torch.int64, device=resolved
        ),
        attempt_child_agent_ids=torch.full(
            (b, e), NO_AGENT_ID, dtype=torch.int64, device=resolved
        ),
        attempt_donor_ordinals=torch.full(
            (b, e, DONOR_ORDINAL_CAPACITY),
            NO_DONOR_ORDINAL,
            dtype=torch.uint8,
            device=resolved,
        ),
        attempt_donor_draw_counts=torch.zeros(
            (b, e), dtype=torch.int64, device=resolved
        ),
        qualification_attempt_capacity_enabled=torch.zeros(
            (b,), dtype=torch.bool, device=resolved
        ),
        qualification_effective_attempt_capacity=torch.full(
            (b,), e, dtype=torch.int64, device=resolved
        ),
        registered_path_codes=torch.tensor(
            shape.registered_path_codes, dtype=torch.int8, device=resolved
        ),
        path_rng_counter_states=torch.zeros(
            (b, len(shape.registered_path_codes)),
            dtype=torch.int64,
            device=resolved,
        ),
        path_counter_write_counts=torch.zeros(
            (b, len(shape.registered_path_codes)),
            dtype=torch.int64,
            device=resolved,
        ),
        path_event_row_write_counts=torch.zeros(
            (b, len(shape.registered_path_codes)),
            dtype=torch.int64,
            device=resolved,
        ),
        path_rng_state_write_counts=torch.zeros(
            (b, len(shape.registered_path_codes)),
            dtype=torch.int64,
            device=resolved,
        ),
        continuity=ReproductionExposureContinuityV1.origin(b),
    )
    owner.validate_exact()
    return owner


def project_resident_reproduction_exposure_v1(
    projection: Mapping[str, Any],
    *,
    existing_counts: ReproductionExposureExistingCountsV1 | None = None,
) -> ResidentReproductionExposureProjectionV1:
    """Project typed rows and successful parent contributions from a checkpoint."""

    shape, tensors, _continuity = _validated_checkpoint_transport(projection)
    if bool(
        (
            tensors["attempt_failure_codes"]
            == FAILURE_PENDING_COMMIT
        ).any().item()
    ):
        raise CudaResidentReproductionExposureError(
            "cannot project exposure while authoritative commits are pending"
        )
    attempts: list[ReproductionExposureAttemptV1] = []
    snapshots: list[ReproductionExposureSnapshotV1] = []
    cells: list[ReproductionExposureCellProjectionV1] = []

    if existing_counts is not None and (
        not isinstance(existing_counts, ReproductionExposureExistingCountsV1)
        or len(existing_counts.attempts) != shape.batch_cells
    ):
        raise CudaResidentReproductionExposureError(
            "existing count cross-check must match the projected batch"
        )

    for cell in range(shape.batch_cells):
        snapshot_count = int(tensors["snapshot_cursors"][cell].item())
        for row in range(snapshot_count):
            eligible_count = int(
                tensors["snapshot_eligible_counts"][cell, row].item()
            )
            snapshots.append(
                ReproductionExposureSnapshotV1(
                    cell_index=cell,
                    snapshot_index=row,
                    path=int(tensors["snapshot_paths"][cell, row].item()),
                    generation=int(
                        tensors["snapshot_generations"][cell, row].item()
                    ),
                    tick=int(tensors["snapshot_ticks"][cell, row].item()),
                    eligible_parent_ids=tuple(
                        int(value)
                        for value in tensors["snapshot_eligible_parent_ids"][
                            cell, row, :eligible_count
                        ].tolist()
                    ),
                )
            )

        selected_counts: dict[int, int] = {}
        contributions: dict[int, int] = {}
        aggregate_rows = (
            torch.nonzero(
                tensors["aggregate_agent_seen"][cell],
                as_tuple=False,
            )
            .flatten()
            .tolist()
        )
        primary_opportunities = {
            int(tensors["aggregate_agent_ids"][cell, row].item()): int(
                tensors["aggregate_primary_opportunity_counts"][
                    cell, row
                ].item()
            )
            for row in aggregate_rows
            if int(
                tensors["aggregate_primary_opportunity_counts"][
                    cell, row
                ].item()
            )
            > 0
        }
        aggregate_rows = (
            torch.nonzero(
                tensors["aggregate_agent_seen"][cell],
                as_tuple=False,
            )
            .flatten()
            .tolist()
        )
        mate_opportunities = {
            int(tensors["aggregate_agent_ids"][cell, row].item()): int(
                tensors["aggregate_mate_opportunity_counts"][
                    cell, row
                ].item()
            )
            for row in aggregate_rows
            if int(
                tensors["aggregate_mate_opportunity_counts"][
                    cell, row
                ].item()
            )
            > 0
        }
        attempt_count = int(tensors["attempt_cursors"][cell].item())
        success_count = 0
        for row in range(attempt_count):
            selected_row = tensors["attempt_selected_parent_ids"][cell, row]
            parent_0_raw = int(selected_row[PARENT_ROLE_PRIMARY].item())
            parent_1_raw = int(selected_row[PARENT_ROLE_MATE].item())
            succeeded = bool(tensors["attempt_success"][cell, row].item())
            attempt = ReproductionExposureAttemptV1(
                cell_index=cell,
                attempt_index=row,
                path=int(tensors["attempt_paths"][cell, row].item()),
                generation=int(
                    tensors["attempt_generations"][cell, row].item()
                ),
                tick=int(tensors["attempt_ticks"][cell, row].item()),
                primary_eligible_snapshot_ref=int(
                    tensors["attempt_eligible_snapshot_refs"][
                        cell, row, PARENT_ROLE_PRIMARY
                    ].item()
                ),
                mate_eligible_snapshot_ref=int(
                    tensors["attempt_eligible_snapshot_refs"][
                        cell, row, PARENT_ROLE_MATE
                    ].item()
                ),
                primary_eligible_count=int(
                    tensors["attempt_eligible_counts"][
                        cell, row, PARENT_ROLE_PRIMARY
                    ].item()
                ),
                mate_eligible_count=int(
                    tensors["attempt_eligible_counts"][
                        cell, row, PARENT_ROLE_MATE
                    ].item()
                ),
                selected_parent_0_id=(
                    None if parent_0_raw == NO_AGENT_ID else parent_0_raw
                ),
                selected_parent_1_id=(
                    None if parent_1_raw == NO_AGENT_ID else parent_1_raw
                ),
                success=succeeded,
                failure_code=int(
                    tensors["attempt_failure_codes"][cell, row].item()
                ),
                commit_ordinal=(
                    None
                    if int(
                        tensors["attempt_commit_ordinals"][cell, row].item()
                    ) == NO_COMMIT_ORDINAL
                    else int(
                        tensors["attempt_commit_ordinals"][cell, row].item()
                    )
                ),
                child_agent_id=(
                    None
                    if int(
                        tensors["attempt_child_agent_ids"][cell, row].item()
                    ) == NO_AGENT_ID
                    else int(
                        tensors["attempt_child_agent_ids"][cell, row].item()
                    )
                ),
                donor_draw_ordinals=tuple(
                    int(value)
                    for value in tensors["attempt_donor_ordinals"][
                        cell,
                        row,
                        : int(
                            tensors["attempt_donor_draw_counts"][
                                cell, row
                            ].item()
                        ),
                    ].tolist()
                ),
            )
            attempts.append(attempt)
            if shape.storage_mode == STORAGE_MODE_EXACT_FULL:
                for role, ref in enumerate(
                    (
                        attempt.primary_eligible_snapshot_ref,
                        attempt.mate_eligible_snapshot_ref,
                    )
                ):
                    eligible_count = int(
                        tensors["snapshot_eligible_counts"][cell, ref].item()
                    )
                    target = (
                        primary_opportunities
                        if role == PARENT_ROLE_PRIMARY
                        else mate_opportunities
                    )
                    for parent_id in tensors[
                        "snapshot_eligible_parent_ids"
                    ][cell, ref, :eligible_count].tolist():
                        key = int(parent_id)
                        target[key] = target.get(key, 0) + 1
            selected_ids = tuple(
                value
                for value in (
                    attempt.selected_parent_0_id,
                    attempt.selected_parent_1_id,
                )
                if value is not None
            )
            for parent_id in selected_ids:
                selected_counts[parent_id] = selected_counts.get(parent_id, 0) + 1
            if succeeded:
                success_count += 1
                for parent_id in selected_ids:
                    contributions[parent_id] = contributions.get(parent_id, 0) + 1
        failure_count = attempt_count - success_count
        if existing_counts is not None and (
            attempt_count != existing_counts.attempts[cell]
            or success_count != existing_counts.successes[cell]
            or failure_count != existing_counts.failures[cell]
        ):
            raise CudaResidentReproductionExposureError(
                f"existing reproduction counts contradict exposure cell {cell}"
            )
        cells.append(
            ReproductionExposureCellProjectionV1(
                cell_index=cell,
                enabled=bool(tensors["cell_enabled"][cell].item()),
                status=int(tensors["cell_status"][cell].item()),
                attempt_count=attempt_count,
                success_count=success_count,
                failure_count=failure_count,
                selected_parent_counts=dict(sorted(selected_counts.items())),
                parent_contributions=dict(sorted(contributions.items())),
                primary_eligibility_opportunities=dict(
                    sorted(primary_opportunities.items())
                ),
                mate_eligibility_opportunities=dict(
                    sorted(mate_opportunities.items())
                ),
                path_instrumentation=tuple(
                    {
                        "path_code": path_code,
                        "rng_counter_state": int(
                            tensors["path_rng_counter_states"][
                                cell, path_index
                            ].item()
                        ),
                        "counter_write_count": int(
                            tensors["path_counter_write_counts"][
                                cell, path_index
                            ].item()
                        ),
                        "event_row_write_count": int(
                            tensors["path_event_row_write_counts"][
                                cell, path_index
                            ].item()
                        ),
                        "rng_state_write_count": int(
                            tensors["path_rng_state_write_counts"][
                                cell, path_index
                            ].item()
                        ),
                    }
                    for path_index, path_code in enumerate(
                        shape.registered_path_codes
                    )
                ),
            )
        )

    return ResidentReproductionExposureProjectionV1(
        schema_version=PROJECTOR_SCHEMA_VERSION,
        shape=shape,
        attempts=tuple(attempts),
        snapshots=tuple(snapshots),
        cells=tuple(cells),
    )


def _attempt_payload(
    row: ReproductionExposureAttemptV1,
    *,
    storage_mode: str,
) -> dict[str, Any]:
    authoritative_commit = None
    if row.success:
        authority = {
            "generation": row.generation,
            "tick": row.tick,
            "path_id": row.path,
            "selected_parent_ids": [
                row.selected_parent_0_id,
                row.selected_parent_1_id,
            ],
            "child_agent_id": row.child_agent_id,
            "donor_draw_ordinals": list(row.donor_draw_ordinals),
        }
        authoritative_commit = {
            **authority,
            "commit_authority_sha256": _canonical_sha256(authority),
        }
    payload = {
        "attempt_index": row.attempt_index,
        "path": row.path,
        "generation": row.generation,
        "tick": row.tick,
        "eligible_parent_counts": [
            row.primary_eligible_count,
            row.mate_eligible_count,
        ],
        "selected_parent_ids": [
            row.selected_parent_0_id,
            row.selected_parent_1_id,
        ],
        "success": row.success,
        "failure_code": row.failure_code,
        "commit_ordinal": row.commit_ordinal,
        "authoritative_commit": authoritative_commit,
    }
    if storage_mode == STORAGE_MODE_EXACT_FULL:
        payload["eligible_snapshot_refs"] = [
            row.primary_eligible_snapshot_ref,
            row.mate_eligible_snapshot_ref,
        ]
    return payload


def _snapshot_payload(row: ReproductionExposureSnapshotV1) -> dict[str, Any]:
    return {
        "snapshot_index": row.snapshot_index,
        "path": row.path,
        "generation": row.generation,
        "tick": row.tick,
        "eligible_parent_ids": list(row.eligible_parent_ids),
    }


def _count_payload(values: Mapping[int, int]) -> list[dict[str, int]]:
    return [
        {"agent_id": agent_id, "count": count}
        for agent_id, count in sorted(values.items())
    ]


def _reset_cpu_tensors(
    shape: ResidentReproductionExposureShapeV1,
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    reset = {name: value.clone() for name, value in tensors.items()}
    reset["attempt_paths"].fill_(NO_PATH)
    reset["attempt_generations"].fill_(NO_GENERATION)
    reset["attempt_ticks"].fill_(NO_TICK)
    reset["attempt_eligible_snapshot_refs"].fill_(NO_REFERENCE)
    reset["attempt_eligible_counts"].zero_()
    reset["attempt_selected_parent_ids"].fill_(NO_AGENT_ID)
    reset["attempt_success"].zero_()
    reset["attempt_failure_codes"].fill_(INACTIVE_FAILURE)
    reset["attempt_commit_ordinals"].fill_(NO_COMMIT_ORDINAL)
    reset["attempt_child_agent_ids"].fill_(NO_AGENT_ID)
    reset["attempt_donor_ordinals"].fill_(NO_DONOR_ORDINAL)
    reset["attempt_donor_draw_counts"].zero_()
    reset["snapshot_paths"].fill_(NO_PATH)
    reset["snapshot_generations"].fill_(NO_GENERATION)
    reset["snapshot_ticks"].fill_(NO_TICK)
    reset["snapshot_eligible_counts"].zero_()
    reset["snapshot_eligible_parent_ids"].fill_(NO_AGENT_ID)
    reset["aggregate_agent_seen"].zero_()
    reset["aggregate_agent_ids"].fill_(NO_AGENT_ID)
    reset["aggregate_primary_opportunity_counts"].zero_()
    reset["aggregate_mate_opportunity_counts"].zero_()
    reset["attempt_cursors"].zero_()
    reset["snapshot_cursors"].zero_()
    reset["aggregate_seen_counts"].zero_()
    enabled = reset["cell_enabled"]
    previous_status = tensors["cell_status"]
    reset["cell_status"] = torch.where(
        ~enabled,
        torch.full((shape.batch_cells,), CELL_STATUS_DISABLED, dtype=torch.int32),
        torch.where(
            previous_status == CELL_STATUS_OK,
            torch.full((shape.batch_cells,), CELL_STATUS_OK, dtype=torch.int32),
            previous_status,
        ),
    ).contiguous()
    return reset


def _checkpoint_from_cpu_tensors(
    *,
    shape: ResidentReproductionExposureShapeV1,
    tensors: Mapping[str, torch.Tensor],
    continuity: ReproductionExposureContinuityV1,
) -> dict[str, Any]:
    _validate_tensors(shape=shape, tensors=tensors, require_device_type="cpu")
    digest, manifests = _projection_digest(
        shape=shape,
        tensors=tensors,
        continuity=continuity,
    )
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "owner_schema_version": SCHEMA_VERSION,
        "shape": shape.payload(),
        "continuity": continuity.payload(),
        "tensors": dict(tensors),
        "tensor_manifests": manifests,
        "projection_sha256": digest,
    }


def drain_resident_reproduction_exposure_checkpoint_v1(
    projection: Mapping[str, Any],
) -> ResidentReproductionExposureSegmentDrainV1:
    """Drain one exact segment and return bounded continuation state.

    Raw attempt and eligible-set rows are emitted once in ``segment_output``.
    The continuation checkpoint retains only empty fixed-capacity buffers,
    cumulative counters, and a per-cell hash chain that commits segment order
    and bytes. Overflow status is sticky per cell; healthy enabled cells return
    to ``CELL_STATUS_OK``.
    """

    shape, tensors, before = _validated_checkpoint_transport(projection)
    projected = project_resident_reproduction_exposure_v1(projection)
    segment_index = before.segment_count
    cell_payloads: list[dict[str, Any]] = []
    chain_values: list[str] = []
    attempts_after: list[int] = []
    successes_after: list[int] = []
    failures_after: list[int] = []
    snapshots_after: list[int] = []
    for cell in projected.cells:
        attempts = [
            _attempt_payload(row, storage_mode=shape.storage_mode)
            for row in projected.attempts
            if row.cell_index == cell.cell_index
        ]
        snapshots = [
            _snapshot_payload(row)
            for row in projected.snapshots
            if row.cell_index == cell.cell_index
        ]
        cumulative_attempts = (
            before.cumulative_attempt_counts[cell.cell_index]
            + cell.attempt_count
        )
        cumulative_successes = (
            before.cumulative_success_counts[cell.cell_index]
            + cell.success_count
        )
        cumulative_failures = (
            before.cumulative_failure_counts[cell.cell_index]
            + cell.failure_count
        )
        cumulative_snapshots = (
            before.cumulative_snapshot_counts[cell.cell_index]
            + len(snapshots)
        )
        payload = {
            "cell_index": cell.cell_index,
            "enabled": cell.enabled,
            "status": cell.status,
            "attempt_count": cell.attempt_count,
            "success_count": cell.success_count,
            "failure_count": cell.failure_count,
            "snapshot_count": len(snapshots),
            "attempts": attempts,
            "eligible_parent_snapshots": snapshots,
            "selected_parent_counts": _count_payload(
                cell.selected_parent_counts
            ),
            "parent_contribution_counts": _count_payload(
                cell.parent_contributions
            ),
            "primary_eligibility_opportunity_counts": _count_payload(
                cell.primary_eligibility_opportunities
            ),
            "mate_eligibility_opportunity_counts": _count_payload(
                cell.mate_eligibility_opportunities
            ),
            "path_instrumentation": [
                dict(row) for row in cell.path_instrumentation
            ],
            "cumulative_counts_after_segment": {
                "attempts": cumulative_attempts,
                "successes": cumulative_successes,
                "failures": cumulative_failures,
                "eligible_snapshots": cumulative_snapshots,
            },
        }
        segment_sha256 = _canonical_sha256(
            {
                "schema_version": SEGMENT_OUTPUT_SCHEMA_VERSION,
                "segment_index": segment_index,
                "cell": payload,
            }
        )
        chain_sha256 = _canonical_sha256(
            {
                "schema_version": CONTINUITY_SCHEMA_VERSION,
                "segment_index": segment_index,
                "previous_chain_sha256": before.cell_chain_sha256[
                    cell.cell_index
                ],
                "segment_sha256": segment_sha256,
            }
        )
        payload["segment_sha256"] = segment_sha256
        payload["previous_chain_sha256"] = before.cell_chain_sha256[
            cell.cell_index
        ]
        payload["chain_sha256_after_segment"] = chain_sha256
        cell_payloads.append(payload)
        chain_values.append(chain_sha256)
        attempts_after.append(cumulative_attempts)
        successes_after.append(cumulative_successes)
        failures_after.append(cumulative_failures)
        snapshots_after.append(cumulative_snapshots)

    after = ReproductionExposureContinuityV1(
        segment_count=segment_index + 1,
        cell_chain_sha256=tuple(chain_values),
        cumulative_attempt_counts=tuple(attempts_after),
        cumulative_success_counts=tuple(successes_after),
        cumulative_failure_counts=tuple(failures_after),
        cumulative_snapshot_counts=tuple(snapshots_after),
    )
    segment_core = {
        "schema_version": SEGMENT_OUTPUT_SCHEMA_VERSION,
        "segment_index": segment_index,
        "capacity_semantics": {
            **shape.payload(),
            "scope": "per_execution_segment_raw_buffer_v1",
            "overflow": "cell_local_sticky_fail_closed_v1",
        },
        "cells": cell_payloads,
        "continuity_after_segment": after.payload(),
    }
    segment_output = {
        **segment_core,
        "segment_output_sha256": _canonical_sha256(segment_core),
    }
    continuation = _checkpoint_from_cpu_tensors(
        shape=shape,
        tensors=_reset_cpu_tensors(shape, tensors),
        continuity=after,
    )
    return ResidentReproductionExposureSegmentDrainV1(
        schema_version=SEGMENT_DRAIN_SCHEMA_VERSION,
        segment_output=segment_output,
        continuation_checkpoint=continuation,
    )
