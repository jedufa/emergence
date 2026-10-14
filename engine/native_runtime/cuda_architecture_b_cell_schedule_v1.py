from __future__ import annotations

"""Metadata-only per-cell schedule for a future resident Architecture B owner.

This module composes accepted contracts. It does not define branch identity,
RNG inheritance, Patch P2 parameter encoding, compact summaries, checkpoint
cloning, or automatic batch sizing. It never launches CUDA or a simulation.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from heapq import heappop, heappush
import json
from math import gcd
from typing import Any, Final

from engine.perf.cuda_architecture_b_auto_batch_v1 import (
    RECEIPT_SCHEMA_VERSION as AUTO_BATCH_RECEIPT_SCHEMA_VERSION_V1,
)
from engine.perf.cuda_architecture_b_cell_decision_summary_v1 import (
    CELL_STATUS_SUCCESS_V1,
    CELL_STATUS_UNEXECUTED_V1,
    CellDecisionSummaryContractError,
    CellDecisionSummaryV1,
    validate_cell_decision_summary_v1,
)
from engine.perf.cuda_architecture_b_checkpoint_branch import (
    CudaCheckpointBranch,
    CudaCheckpointBranchError,
    canonical_branch_json_bytes,
    validate_checkpoint_branch,
    validate_checkpoint_branch_batch,
)
from engine.perf.cuda_architecture_b_patch_p2_cell_params_v1 import (
    PatchP2CellParamsError,
    PatchP2CellParamsV1,
    pack_patch_p2_cell_params_v1,
    patch_p2_cell_params_digest_v1,
    unpack_patch_p2_cell_params_v1,
)


SCHEDULE_SPEC_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_cuda_architecture_b_cell_schedule_spec_v1"
)
SCHEDULE_RECEIPT_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_cuda_architecture_b_cell_schedule_receipt_v1"
)
MUTATION_RECEIPT_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_cuda_architecture_b_cell_schedule_mutation_receipt_v1"
)
DISPATCH_PLAN_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_cuda_architecture_b_cell_dispatch_plan_v1"
)

CELL_STATE_ACTIVE_V1: Final[str] = "active"
CELL_STATE_COMPLETED_V1: Final[str] = "completed"
CELL_STATE_CANCELLED_V1: Final[str] = "cancelled"
CELL_STATE_FAILED_V1: Final[str] = "failed"
TERMINAL_CELL_STATES_V1: Final[frozenset[str]] = frozenset(
    {CELL_STATE_COMPLETED_V1, CELL_STATE_CANCELLED_V1, CELL_STATE_FAILED_V1}
)

COMMAND_CHECKPOINT_V1: Final[str] = "checkpoint"
COMMAND_CHECKPOINT_AND_FORK_V1: Final[str] = "checkpoint_and_fork"
COMMAND_CANCEL_V1: Final[str] = "cancel"
BOUNDARY_COMMAND_KINDS_V1: Final[frozenset[str]] = frozenset(
    {
        COMMAND_CHECKPOINT_V1,
        COMMAND_CHECKPOINT_AND_FORK_V1,
        COMMAND_CANCEL_V1,
    }
)

_SCHEDULE_FIELDS: Final[frozenset[str]] = frozenset(
    {"schema_version", "controller_id", "cell_count", "cells", "auto_batch_receipt"}
)
_CELL_FIELDS: Final[frozenset[str]] = frozenset(
    {"cell_index", "stable_cell_id", "generation_limit", "branch", "parameters"}
)
_COMMAND_FIELDS: Final[frozenset[str]] = frozenset(
    {"command_id", "cell_index", "generation_boundary", "kind"}
)
_CHECKPOINT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "command_id",
        "cell_index",
        "generation_boundary",
        "checkpoint_projection_sha256",
        "rng_state_sha256",
    }
)
_FORK_FIELDS: Final[frozenset[str]] = frozenset(
    {"command_id", "cell_index", "generation_boundary", "branch"}
)
_CANCEL_FIELDS: Final[frozenset[str]] = frozenset(
    {"command_id", "cell_index", "generation_boundary"}
)
_MAX_INDEX: Final[int] = (1 << 63) - 1


class CellScheduleContractError(ValueError):
    """Raised when schedule input or accepted-contract evidence is invalid."""


class CellScheduleStateError(RuntimeError):
    """Raised when a valid request contradicts the current schedule state."""


def _fail(path: str, message: str) -> None:
    raise CellScheduleContractError(f"{path}: {message}")


def _state_fail(path: str, message: str) -> None:
    raise CellScheduleStateError(f"{path}: {message}")


def _exact_fields(value: Any, expected: frozenset[str], *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be a mapping")
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        _fail(path, f"field mismatch: missing={missing}, unknown={unknown}")
    return value


def _exact_int(value: Any, *, path: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= _MAX_INDEX:
        _fail(path, f"must be an exact integer in [{minimum}, {_MAX_INDEX}]")
    return value


def _stable_identifier(value: Any, *, path: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 256
        or value.strip() != value
        or not value.isascii()
        or not value.isprintable()
        or "\x00" in value
    ):
        _fail(path, "must be a nonempty stable ASCII identifier")
    return value


def _sha256_hex(value: Any, *, path: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(path, "must be exactly 64 lowercase hexadecimal characters")
    return value


def _plain_json(value: Any) -> Any:
    return json.loads(canonical_branch_json_bytes(value))


def _canonical_sha256(value: Any) -> str:
    return sha256(canonical_branch_json_bytes(value)).hexdigest()


def _accepted_generation_chunks(value: Any) -> tuple[int, ...]:
    path = "accepted_generation_chunks"
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(path, "must be a nonempty sequence of exact positive integers")
    chunks = tuple(
        _exact_int(item, path=f"{path}[{index}]", minimum=1)
        for index, item in enumerate(value)
    )
    if not chunks:
        _fail(path, "must not be empty")
    if len(chunks) != len(set(chunks)):
        _fail(path, "must not contain duplicate chunk sizes")
    return tuple(sorted(chunks, reverse=True))


def _select_exact_generation_chunk(
    spans: Sequence[int],
    *,
    accepted_chunks: tuple[int, ...],
    maximum: int,
) -> int | None:
    """Select the largest call that leaves every safe span exactly reachable."""

    usable = tuple(chunk for chunk in accepted_chunks if chunk <= maximum)
    if not usable:
        return None

    common_divisor = 0
    for chunk in usable:
        common_divisor = gcd(common_divisor, chunk)
    normalized_chunks = tuple(chunk // common_divisor for chunk in usable)
    modulus = min(normalized_chunks)
    distances: dict[int, int] = {0: 0}
    settled: dict[int, int] = {}
    pending: list[tuple[int, int]] = [(0, 0)]

    def is_reachable(span: int) -> bool:
        if span < 0 or span % common_divisor:
            return False
        normalized_span = span // common_divisor
        target_residue = normalized_span % modulus
        while target_residue not in settled:
            if not pending:
                return False
            distance, residue = heappop(pending)
            if residue in settled:
                continue
            settled[residue] = distance
            for chunk in normalized_chunks:
                next_residue = (residue + chunk) % modulus
                next_distance = distance + chunk
                previous_distance = distances.get(next_residue)
                if previous_distance is None or next_distance < previous_distance:
                    distances[next_residue] = next_distance
                    heappush(pending, (next_distance, next_residue))
        return settled[target_residue] <= normalized_span

    if any(not is_reachable(span) for span in spans):
        return None
    shortest_span = min(spans)
    for chunk in usable:
        if chunk <= shortest_span and all(
            is_reachable(span - chunk) for span in spans
        ):
            return chunk
    return None


@dataclass(frozen=True, slots=True)
class CellScheduleMutationReceiptV1:
    receipt_type: str
    accepted: bool
    disposition: str
    controller_id: str
    cell_index: int | None
    generation_boundary: int | None
    state: str | None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MUTATION_RECEIPT_SCHEMA_VERSION_V1,
            "receipt_type": self.receipt_type,
            "accepted": self.accepted,
            "disposition": self.disposition,
            "controller_id": self.controller_id,
            "cell_index": self.cell_index,
            "generation_boundary": self.generation_boundary,
            "state": self.state,
            "details": _plain_json(self.details),
            "contract": {
                "metadata_only": True,
                "cuda_execution_performed": False,
                "simulation_execution_performed": False,
                "production_promotion_claimed": False,
            },
        }


@dataclass(frozen=True, slots=True)
class CellDispatchCohortPlanV1:
    cohort_index: int
    cell_indices: tuple[int, ...]
    stable_cell_ids: tuple[int, ...]
    start_generation: int
    generation_count: int
    stop_reason: str
    stop_cell_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cohort_index": self.cohort_index,
            "physical_batch_cell_count": len(self.cell_indices),
            "cell_indices": list(self.cell_indices),
            "stable_cell_ids": list(self.stable_cell_ids),
            "start_generation": self.start_generation,
            "generation_count": self.generation_count,
            "stop_reason": self.stop_reason,
            "stop_cell_indices": list(self.stop_cell_indices),
            "shared_generation_coordinate": True,
        }


@dataclass(frozen=True, slots=True)
class CellDispatchPlanReceiptV1:
    controller_id: str
    schedule_sha256: str
    max_generations_per_dispatch: int
    cohorts: tuple[CellDispatchCohortPlanV1, ...]
    terminal_cell_indices: tuple[int, ...]
    blocked_active_cells: tuple[Mapping[str, Any], ...]

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": DISPATCH_PLAN_SCHEMA_VERSION_V1,
            "controller_id": self.controller_id,
            "schedule_sha256": self.schedule_sha256,
            "max_generations_per_dispatch": self.max_generations_per_dispatch,
            "cohort_count": len(self.cohorts),
            "cohorts": [cohort.to_dict() for cohort in self.cohorts],
            "terminal_cell_indices": list(self.terminal_cell_indices),
            "blocked_active_cells": [
                _plain_json(cell) for cell in self.blocked_active_cells
            ],
            "selection_policy": {
                "primary": "largest_safe_same_generation_physical_batch",
                "secondary": (
                    "largest_accepted_chunk_preserving_exact_boundary_decomposition"
                ),
                "cohort_order": "current_generation_ascending",
                "cell_order": "stable_cell_id_then_cell_index",
            },
            "contract": {
                "read_only": True,
                "idempotent": True,
                "metadata_only": True,
                "shared_generation_count_per_cohort": True,
                "live_controller_integration": False,
                "cuda_execution_performed": False,
                "simulation_execution_performed": False,
                "performance_claimed": False,
                "production_promotion_claimed": False,
            },
        }

    @property
    def plan_sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["plan_sha256"] = self.plan_sha256
        return payload


@dataclass(frozen=True, slots=True)
class _CellIdentityV1:
    cell_index: int
    stable_cell_id: int
    generation_limit: int
    start_generation: int
    branch: CudaCheckpointBranch
    parameters: PatchP2CellParamsV1
    parameter_digest: Mapping[str, Any]

    @property
    def branch_id(self) -> str:
        return self.branch.branch_id

    @property
    def parent_branch_id(self) -> str | None:
        return self.branch.rng_lineage_provenance["source_branch_id"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_index": self.cell_index,
            "stable_cell_id": self.stable_cell_id,
            "generation_limit": self.generation_limit,
            "start_generation": self.start_generation,
            "branch_id": self.branch_id,
            "parent_branch_id": self.parent_branch_id,
            "branch": self.branch.to_record(),
            "parameters": self.parameters.to_dict(),
            "parameter_digest": dict(self.parameter_digest),
            "rng_lineage": self.branch.rng_lineage_provenance,
        }


@dataclass(frozen=True, slots=True)
class _BoundaryCommandV1:
    command_id: str
    cell_index: int
    generation_boundary: int
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "cell_index": self.cell_index,
            "generation_boundary": self.generation_boundary,
            "kind": self.kind,
        }


@dataclass(frozen=True, slots=True)
class _CheckpointBoundaryV1:
    command_id: str
    cell_index: int
    generation_boundary: int
    checkpoint_projection_sha256: str
    rng_state_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "cell_index": self.cell_index,
            "generation_boundary": self.generation_boundary,
            "generation_boundary_complete": True,
            "checkpoint_projection_sha256": self.checkpoint_projection_sha256,
            "rng_state_sha256": self.rng_state_sha256,
        }


@dataclass(slots=True)
class _CellRuntimeV1:
    identity: _CellIdentityV1
    state: str = CELL_STATE_ACTIVE_V1
    completed_generation_count: int = 0
    compact_summary_row_frontier: int = 0
    status_prefix: tuple[int, ...] = ()
    commands: dict[int, _BoundaryCommandV1] = field(default_factory=dict)
    checkpoints: dict[int, _CheckpointBoundaryV1] = field(default_factory=dict)
    fork_branches: dict[int, CudaCheckpointBranch] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity.to_dict(),
            "state": self.state,
            "terminal": self.state in TERMINAL_CELL_STATES_V1,
            "completed_generation_count": self.completed_generation_count,
            "current_generation": (
                self.identity.start_generation + self.completed_generation_count
            ),
            "compact_summary_row_frontier": self.compact_summary_row_frontier,
            "commands": [
                command.to_dict()
                for _, command in sorted(self.commands.items())
            ],
            "completed_checkpoint_boundaries": [
                checkpoint.to_dict()
                for _, checkpoint in sorted(self.checkpoints.items())
            ],
            "fork_bindings": [
                {
                    "generation_boundary": boundary,
                    "branch": branch.to_record(),
                    "checkpoint_clone_plan_required": True,
                    "checkpoint_clone_execution_performed": False,
                }
                for boundary, branch in sorted(self.fork_branches.items())
            ],
        }


def _accepted_auto_batch_binding(
    receipt: Any, *, cell_count: int
) -> tuple[dict[str, Any], str]:
    if not isinstance(receipt, Mapping):
        _fail("schedule.auto_batch_receipt", "must be an accepted auto-batch receipt")
    try:
        plain = _plain_json(receipt)
    except (CudaCheckpointBranchError, TypeError, ValueError) as exc:
        raise CellScheduleContractError(
            f"schedule.auto_batch_receipt: is not canonical JSON: {exc}"
        ) from exc
    if plain.get("schema_version") != AUTO_BATCH_RECEIPT_SCHEMA_VERSION_V1:
        _fail("schedule.auto_batch_receipt.schema_version", "is not the accepted v1 schema")
    if "selected_batch_size" in plain:
        selected = plain.get("decision") == "selected"
        selected_count = plain.get("selected_batch_size")
        receipt_form = "combined"
    else:
        selected = plain.get("status") == "selected"
        selected_count = plain.get("selected_batch_cells")
        receipt_form = "split"
    if not selected:
        _fail("schedule.auto_batch_receipt", "must contain an accepted selection")
    if type(selected_count) is not int or selected_count != cell_count:
        _fail(
            "schedule.auto_batch_receipt",
            "selected batch size must exactly match configurable cell_count",
        )
    digest = _canonical_sha256(plain)
    return (
        {
            "schema_version": plain["schema_version"],
            "receipt_form": receipt_form,
            "selected_cell_count": selected_count,
            "receipt_sha256": digest,
            "selection_reused_without_recalculation": True,
            "hardware_probed_by_schedule": False,
            "performance_qualified_by_schedule": False,
        },
        digest,
    )


def _canonical_parameters(value: Any, *, path: str) -> PatchP2CellParamsV1:
    if not isinstance(value, PatchP2CellParamsV1):
        _fail(path, "must be an accepted PatchP2CellParamsV1 record")
    try:
        return unpack_patch_p2_cell_params_v1(pack_patch_p2_cell_params_v1(value))
    except PatchP2CellParamsError as exc:
        raise CellScheduleContractError(f"{path}: {exc}") from exc


class CudaArchitectureBCellScheduleV1:
    """Thin per-cell state machine over accepted Architecture B contracts."""

    def __init__(
        self,
        *,
        controller_id: str,
        cells: tuple[_CellIdentityV1, ...],
        auto_batch_binding: Mapping[str, Any],
        auto_batch_receipt_sha256: str,
    ) -> None:
        self._controller_id = controller_id
        self._cells = {
            cell.cell_index: _CellRuntimeV1(identity=cell) for cell in cells
        }
        self._stable_cell_index = {
            cell.stable_cell_id: cell.cell_index for cell in cells
        }
        self._auto_batch_binding = _plain_json(auto_batch_binding)
        self._auto_batch_receipt_sha256 = auto_batch_receipt_sha256
        identity_payload = {
            "schema_version": SCHEDULE_SPEC_SCHEMA_VERSION_V1,
            "controller_id": controller_id,
            "cell_count": len(cells),
            "cells": [cell.to_dict() for cell in cells],
            "auto_batch_receipt_sha256": auto_batch_receipt_sha256,
        }
        self._schedule_sha256 = _canonical_sha256(identity_payload)

    @property
    def cell_count(self) -> int:
        return len(self._cells)

    @property
    def schedule_sha256(self) -> str:
        return self._schedule_sha256

    def _runtime(self, cell_index: Any, *, path: str) -> _CellRuntimeV1:
        index = _exact_int(cell_index, path=path)
        runtime = self._cells.get(index)
        if runtime is None:
            _fail(path, "is outside the configured cell range")
        return runtime

    def _receipt(
        self,
        *,
        receipt_type: str,
        accepted: bool,
        disposition: str,
        runtime: _CellRuntimeV1 | None = None,
        generation_boundary: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> CellScheduleMutationReceiptV1:
        return CellScheduleMutationReceiptV1(
            receipt_type=receipt_type,
            accepted=accepted,
            disposition=disposition,
            controller_id=self._controller_id,
            cell_index=(runtime.identity.cell_index if runtime is not None else None),
            generation_boundary=generation_boundary,
            state=(runtime.state if runtime is not None else None),
            details={} if details is None else details,
        )

    @staticmethod
    def _pending_boundary(
        runtime: _CellRuntimeV1,
    ) -> tuple[int, _BoundaryCommandV1] | None:
        for boundary, command in sorted(runtime.commands.items()):
            if command.kind == COMMAND_CHECKPOINT_V1:
                settled = boundary in runtime.checkpoints
            elif command.kind == COMMAND_CHECKPOINT_AND_FORK_V1:
                settled = (
                    boundary in runtime.checkpoints
                    and boundary in runtime.fork_branches
                )
            else:
                settled = runtime.state == CELL_STATE_CANCELLED_V1
            if settled:
                continue
            if boundary < runtime.completed_generation_count:
                _state_fail(
                    "dispatch",
                    f"cell {runtime.identity.cell_index} crossed pending "
                    f"boundary {boundary}",
                )
            return boundary, command
        return None

    def plan_dispatch_cohorts(
        self,
        *,
        max_generations_per_dispatch: int,
        accepted_generation_chunks: Sequence[int],
    ) -> CellDispatchPlanReceiptV1:
        maximum = _exact_int(
            max_generations_per_dispatch,
            path="max_generations_per_dispatch",
            minimum=1,
        )
        accepted_chunks = _accepted_generation_chunks(accepted_generation_chunks)
        eligible_by_generation: dict[
            int, list[tuple[_CellRuntimeV1, int, int | None, str | None]]
        ] = {}
        terminal: list[_CellRuntimeV1] = []
        blocked: list[dict[str, Any]] = []

        for runtime in self._cells.values():
            if runtime.state in TERMINAL_CELL_STATES_V1:
                terminal.append(runtime)
                continue
            remaining = (
                runtime.identity.generation_limit
                - runtime.completed_generation_count
            )
            pending = self._pending_boundary(runtime)
            pending_boundary = pending[0] if pending is not None else None
            pending_kind = pending[1].kind if pending is not None else None
            pending_distance = (
                pending_boundary - runtime.completed_generation_count
                if pending_boundary is not None
                else None
            )
            safe_span = min(
                remaining,
                pending_distance if pending_distance is not None else _MAX_INDEX,
            )
            if safe_span <= 0:
                blocked.append(
                    {
                        "cell_index": runtime.identity.cell_index,
                        "stable_cell_id": runtime.identity.stable_cell_id,
                        "current_generation": (
                            runtime.identity.start_generation
                            + runtime.completed_generation_count
                        ),
                        "reason": (
                            "pending_boundary"
                            if pending_boundary is not None
                            else "generation_limit"
                        ),
                        "pending_boundary": pending_boundary,
                        "pending_command_kind": pending_kind,
                    }
                )
                continue
            current_generation = (
                runtime.identity.start_generation
                + runtime.completed_generation_count
            )
            eligible_by_generation.setdefault(current_generation, []).append(
                (runtime, safe_span, pending_boundary, pending_kind)
            )

        if not eligible_by_generation:
            if terminal and not blocked:
                reason = "all cells are terminal"
            else:
                reason = "all active cells are stopped at safe boundaries"
            _state_fail("dispatch", f"no safe progress: {reason}")

        cohorts: list[CellDispatchCohortPlanV1] = []
        for start_generation in sorted(eligible_by_generation):
            entries = sorted(
                eligible_by_generation[start_generation],
                key=lambda entry: (
                    entry[0].identity.stable_cell_id,
                    entry[0].identity.cell_index,
                ),
            )
            generation_count = _select_exact_generation_chunk(
                tuple(entry[1] for entry in entries),
                accepted_chunks=accepted_chunks,
                maximum=maximum,
            )
            if generation_count is None:
                spans = ", ".join(
                    f"cell {entry[0].identity.cell_index}={entry[1]}"
                    for entry in entries
                )
                _state_fail(
                    "dispatch",
                    "no exact decomposition for safe boundary spans "
                    f"at generation {start_generation}: {spans}; "
                    f"accepted_generation_chunks={list(accepted_chunks)}, "
                    f"max_generations_per_dispatch={maximum}",
                )
            stop_cells: set[int] = set()
            stop_reasons: set[str] = set()
            for runtime, safe_span, pending_boundary, _pending_kind in entries:
                cell_index = runtime.identity.cell_index
                completed = runtime.completed_generation_count
                if completed + generation_count == runtime.identity.generation_limit:
                    stop_reasons.add("generation_limit")
                    stop_cells.add(cell_index)
                if (
                    pending_boundary is not None
                    and completed + generation_count == pending_boundary
                ):
                    stop_reasons.add("pending_boundary")
                    stop_cells.add(cell_index)
                if generation_count == maximum:
                    stop_reasons.add("max_generations_per_dispatch")
                    stop_cells.add(cell_index)
            if not stop_reasons:
                stop_reasons.add("accepted_generation_chunk")
            ordered_reasons = tuple(
                reason
                for reason in (
                    "pending_boundary",
                    "generation_limit",
                    "max_generations_per_dispatch",
                    "accepted_generation_chunk",
                )
                if reason in stop_reasons
            )
            if not ordered_reasons:
                raise AssertionError("dispatch cohort has no limiting reason")
            cohorts.append(
                CellDispatchCohortPlanV1(
                    cohort_index=len(cohorts),
                    cell_indices=tuple(
                        entry[0].identity.cell_index for entry in entries
                    ),
                    stable_cell_ids=tuple(
                        entry[0].identity.stable_cell_id for entry in entries
                    ),
                    start_generation=start_generation,
                    generation_count=generation_count,
                    stop_reason="+".join(ordered_reasons),
                    stop_cell_indices=tuple(sorted(stop_cells)),
                )
            )

        terminal_indices = tuple(
            runtime.identity.cell_index
            for runtime in sorted(
                terminal,
                key=lambda runtime: (
                    runtime.identity.stable_cell_id,
                    runtime.identity.cell_index,
                ),
            )
        )
        blocked_rows = tuple(
            sorted(
                blocked,
                key=lambda row: (row["stable_cell_id"], row["cell_index"]),
            )
        )
        return CellDispatchPlanReceiptV1(
            controller_id=self._controller_id,
            schedule_sha256=self._schedule_sha256,
            max_generations_per_dispatch=maximum,
            cohorts=tuple(cohorts),
            terminal_cell_indices=terminal_indices,
            blocked_active_cells=blocked_rows,
        )

    def request_boundary_command(
        self, value: Mapping[str, Any]
    ) -> CellScheduleMutationReceiptV1:
        raw = _exact_fields(value, _COMMAND_FIELDS, path="command")
        command_id = _stable_identifier(raw["command_id"], path="command.command_id")
        runtime = self._runtime(raw["cell_index"], path="command.cell_index")
        boundary = _exact_int(
            raw["generation_boundary"], path="command.generation_boundary"
        )
        kind = raw["kind"]
        if kind not in BOUNDARY_COMMAND_KINDS_V1:
            _fail("command.kind", f"must be one of {sorted(BOUNDARY_COMMAND_KINDS_V1)}")
        if runtime.state in TERMINAL_CELL_STATES_V1:
            _state_fail("command", f"cell is terminal in state {runtime.state!r}")
        if not runtime.completed_generation_count <= boundary <= runtime.identity.generation_limit:
            _state_fail(
                "command.generation_boundary",
                "must not be behind progress or beyond the per-cell generation limit",
            )
        command = _BoundaryCommandV1(
            command_id=command_id,
            cell_index=runtime.identity.cell_index,
            generation_boundary=boundary,
            kind=kind,
        )
        first = runtime.commands.get(boundary)
        if first is not None:
            if first == command:
                return self._receipt(
                    receipt_type="boundary_command",
                    accepted=True,
                    disposition="idempotent_first_command",
                    runtime=runtime,
                    generation_boundary=boundary,
                    details={"winning_command": first.to_dict()},
                )
            return self._receipt(
                receipt_type="boundary_command",
                accepted=False,
                disposition="rejected_first_command_wins",
                runtime=runtime,
                generation_boundary=boundary,
                details={"winning_command": first.to_dict()},
            )
        runtime.commands[boundary] = command
        return self._receipt(
            receipt_type="boundary_command",
            accepted=True,
            disposition="accepted_first_command",
            runtime=runtime,
            generation_boundary=boundary,
            details={"winning_command": command.to_dict()},
        )

    def _candidate_progress(
        self, runtime: _CellRuntimeV1, statuses: tuple[int, ...]
    ) -> tuple[str, int, int, tuple[int, ...]]:
        limit = runtime.identity.generation_limit
        if any(status != CELL_STATUS_UNEXECUTED_V1 for status in statuses[limit:]):
            _state_fail(
                f"summary.cell[{runtime.identity.cell_index}]",
                "contains a published row beyond its generation limit",
            )
        prefix: list[int] = []
        seen_unexecuted = False
        seen_failure = False
        successful = 0
        for row_index, status in enumerate(statuses[:limit]):
            if status == CELL_STATUS_UNEXECUTED_V1:
                seen_unexecuted = True
                continue
            if seen_unexecuted:
                _state_fail(
                    f"summary.cell[{runtime.identity.cell_index}].row[{row_index}]",
                    "published rows must form one contiguous prefix",
                )
            if seen_failure:
                _state_fail(
                    f"summary.cell[{runtime.identity.cell_index}].row[{row_index}]",
                    "no row may follow a failed terminal row",
                )
            prefix.append(status)
            if status == CELL_STATUS_SUCCESS_V1:
                successful += 1
            else:
                seen_failure = True

        frozen_prefix = tuple(prefix)
        if frozen_prefix[: len(runtime.status_prefix)] != runtime.status_prefix:
            _state_fail(
                f"summary.cell[{runtime.identity.cell_index}]",
                "rewrites or regresses the accepted compact-summary prefix",
            )
        frontier = len(frozen_prefix)
        if frontier < runtime.compact_summary_row_frontier:
            _state_fail(
                f"summary.cell[{runtime.identity.cell_index}]",
                "compact-summary row frontier regressed",
            )
        if runtime.state in TERMINAL_CELL_STATES_V1:
            if frozen_prefix != runtime.status_prefix:
                _state_fail(
                    f"summary.cell[{runtime.identity.cell_index}]",
                    "cannot advance a terminal cell",
                )
            return runtime.state, runtime.completed_generation_count, frontier, frozen_prefix

        for boundary, command in runtime.commands.items():
            if runtime.completed_generation_count < boundary < frontier:
                if command.kind == COMMAND_CANCEL_V1:
                    _state_fail(
                        f"summary.cell[{runtime.identity.cell_index}]",
                        f"progress crossed pending cancellation boundary {boundary}",
                    )
                if boundary not in runtime.checkpoints:
                    _state_fail(
                        f"summary.cell[{runtime.identity.cell_index}]",
                        f"progress crossed incomplete checkpoint boundary {boundary}",
                    )

        if seen_failure:
            state = CELL_STATE_FAILED_V1
        elif successful == limit:
            terminal_command = runtime.commands.get(limit)
            state = (
                CELL_STATE_ACTIVE_V1
                if terminal_command is not None
                and terminal_command.kind == COMMAND_CANCEL_V1
                else CELL_STATE_COMPLETED_V1
            )
        else:
            state = CELL_STATE_ACTIVE_V1
        return state, successful, frontier, frozen_prefix

    def apply_compact_summary(
        self, summary: CellDecisionSummaryV1
    ) -> CellScheduleMutationReceiptV1:
        try:
            validate_cell_decision_summary_v1(summary)
        except CellDecisionSummaryContractError as exc:
            raise CellScheduleContractError(f"summary: {exc}") from exc
        if summary.cell_count != self.cell_count:
            _fail("summary.cell_count", "must exactly match schedule cell_count")
        maximum_limit = max(
            runtime.identity.generation_limit for runtime in self._cells.values()
        )
        if summary.generation_count != maximum_limit:
            _fail(
                "summary.generation_count",
                "must equal the maximum per-cell generation limit",
            )
        summary_ids = tuple(int(value) for value in summary.stable_cell_ids)
        if set(summary_ids) != set(self._stable_cell_index):
            _fail("summary.stable_cell_ids", "must exactly match the schedule identities")

        candidates: dict[int, tuple[str, int, int, tuple[int, ...]]] = {}
        for summary_cell_index, stable_cell_id in enumerate(summary_ids):
            runtime = self._cells[self._stable_cell_index[stable_cell_id]]
            statuses = tuple(
                int(value) for value in summary.status[:, summary_cell_index]
            )
            candidates[runtime.identity.cell_index] = self._candidate_progress(
                runtime, statuses
            )

        updates: list[dict[str, Any]] = []
        for cell_index, candidate in sorted(candidates.items()):
            runtime = self._cells[cell_index]
            previous = {
                "state": runtime.state,
                "completed_generation_count": runtime.completed_generation_count,
                "compact_summary_row_frontier": runtime.compact_summary_row_frontier,
            }
            (
                runtime.state,
                runtime.completed_generation_count,
                runtime.compact_summary_row_frontier,
                runtime.status_prefix,
            ) = candidate
            updates.append(
                {
                    "cell_index": cell_index,
                    "previous": previous,
                    "current": {
                        "state": runtime.state,
                        "completed_generation_count": runtime.completed_generation_count,
                        "compact_summary_row_frontier": runtime.compact_summary_row_frontier,
                    },
                }
            )
        return self._receipt(
            receipt_type="compact_summary_progress",
            accepted=True,
            disposition="accepted_transactional_prefix_update",
            details={"cell_updates": updates},
        )

    def record_checkpoint_boundary(
        self, value: Mapping[str, Any]
    ) -> CellScheduleMutationReceiptV1:
        raw = _exact_fields(value, _CHECKPOINT_FIELDS, path="checkpoint")
        command_id = _stable_identifier(
            raw["command_id"], path="checkpoint.command_id"
        )
        runtime = self._runtime(raw["cell_index"], path="checkpoint.cell_index")
        boundary = _exact_int(
            raw["generation_boundary"], path="checkpoint.generation_boundary"
        )
        command = runtime.commands.get(boundary)
        if command is None or command.command_id != command_id or command.kind not in {
            COMMAND_CHECKPOINT_V1,
            COMMAND_CHECKPOINT_AND_FORK_V1,
        }:
            _state_fail(
                "checkpoint",
                "must match the first accepted checkpoint-capable boundary command",
            )
        if boundary > runtime.completed_generation_count:
            _state_fail("checkpoint", "generation boundary is not complete")
        checkpoint = _CheckpointBoundaryV1(
            command_id=command_id,
            cell_index=runtime.identity.cell_index,
            generation_boundary=boundary,
            checkpoint_projection_sha256=_sha256_hex(
                raw["checkpoint_projection_sha256"],
                path="checkpoint.checkpoint_projection_sha256",
            ),
            rng_state_sha256=_sha256_hex(
                raw["rng_state_sha256"], path="checkpoint.rng_state_sha256"
            ),
        )
        existing = runtime.checkpoints.get(boundary)
        if existing is not None and existing != checkpoint:
            _state_fail("checkpoint", "boundary already has different checkpoint evidence")
        runtime.checkpoints[boundary] = checkpoint
        return self._receipt(
            receipt_type="checkpoint_boundary",
            accepted=True,
            disposition=("idempotent_checkpoint" if existing else "checkpoint_complete"),
            runtime=runtime,
            generation_boundary=boundary,
            details={"checkpoint": checkpoint.to_dict()},
        )

    def bind_fork_branch(
        self, value: Mapping[str, Any]
    ) -> CellScheduleMutationReceiptV1:
        raw = _exact_fields(value, _FORK_FIELDS, path="fork")
        command_id = _stable_identifier(raw["command_id"], path="fork.command_id")
        runtime = self._runtime(raw["cell_index"], path="fork.cell_index")
        boundary = _exact_int(raw["generation_boundary"], path="fork.generation_boundary")
        command = runtime.commands.get(boundary)
        if (
            command is None
            or command.command_id != command_id
            or command.kind != COMMAND_CHECKPOINT_AND_FORK_V1
        ):
            _state_fail(
                "fork", "must match the first accepted checkpoint-and-fork command"
            )
        checkpoint = runtime.checkpoints.get(boundary)
        if checkpoint is None:
            _state_fail("fork", "requires a complete checkpoint boundary")
        try:
            branch = validate_checkpoint_branch(raw["branch"])
        except CudaCheckpointBranchError as exc:
            raise CellScheduleContractError(f"fork.branch: {exc}") from exc
        identity = branch.branch_identity
        source_checkpoint = identity["source_checkpoint"]
        source_lineage = identity["source_lineage"]
        expected_generation = runtime.identity.start_generation + boundary
        if (
            source_checkpoint["generation"] != expected_generation
            or source_checkpoint["projection_sha256"]
            != checkpoint.checkpoint_projection_sha256
            or source_lineage["branch_id"] != runtime.identity.branch_id
            or source_lineage["rng_state_sha256"] != checkpoint.rng_state_sha256
        ):
            _state_fail(
                "fork.branch",
                "does not match the complete checkpoint, parent branch, and RNG lineage",
            )
        all_branch_ids = {
            child.identity.branch_id for child in self._cells.values()
        } | {
            child.branch_id
            for cell in self._cells.values()
            for child in cell.fork_branches.values()
        }
        if branch.branch_id in all_branch_ids:
            existing = runtime.fork_branches.get(boundary)
            if existing is None or existing.branch_id != branch.branch_id:
                _state_fail("fork.branch", "duplicates an existing branch_id")
        existing = runtime.fork_branches.get(boundary)
        if existing is not None and existing.branch_id != branch.branch_id:
            _state_fail("fork", "boundary already has a different fork branch")
        runtime.fork_branches[boundary] = branch
        return self._receipt(
            receipt_type="fork_binding",
            accepted=True,
            disposition=("idempotent_fork_binding" if existing else "fork_branch_bound"),
            runtime=runtime,
            generation_boundary=boundary,
            details={
                "branch": branch.to_record(),
                "checkpoint_clone_contract": (
                    "cuda_architecture_b_checkpoint_clone_v1"
                ),
                "boundary_controller_contract": (
                    "cuda_architecture_b_boundary_controller_v1"
                ),
                "checkpoint_clone_plan_required": True,
                "checkpoint_clone_execution_performed": False,
            },
        )

    def settle_cancellation(
        self, value: Mapping[str, Any]
    ) -> CellScheduleMutationReceiptV1:
        raw = _exact_fields(value, _CANCEL_FIELDS, path="cancellation")
        command_id = _stable_identifier(
            raw["command_id"], path="cancellation.command_id"
        )
        runtime = self._runtime(raw["cell_index"], path="cancellation.cell_index")
        boundary = _exact_int(
            raw["generation_boundary"], path="cancellation.generation_boundary"
        )
        if runtime.state in TERMINAL_CELL_STATES_V1:
            _state_fail("cancellation", f"cell is terminal in state {runtime.state!r}")
        command = runtime.commands.get(boundary)
        if (
            command is None
            or command.command_id != command_id
            or command.kind != COMMAND_CANCEL_V1
        ):
            _state_fail("cancellation", "must match the first accepted cancel command")
        if (
            runtime.completed_generation_count != boundary
            or runtime.compact_summary_row_frontier != boundary
        ):
            _state_fail("cancellation", "requested generation boundary is not settled")
        runtime.state = CELL_STATE_CANCELLED_V1
        return self._receipt(
            receipt_type="cancellation",
            accepted=True,
            disposition="cancelled_at_generation_boundary",
            runtime=runtime,
            generation_boundary=boundary,
            details={"winning_command": command.to_dict()},
        )

    def to_dict(self) -> dict[str, Any]:
        cells = [self._cells[index].to_dict() for index in sorted(self._cells)]
        state_counts = {
            state: sum(cell["state"] == state for cell in cells)
            for state in (
                CELL_STATE_ACTIVE_V1,
                CELL_STATE_COMPLETED_V1,
                CELL_STATE_CANCELLED_V1,
                CELL_STATE_FAILED_V1,
            )
        }
        return {
            "schema_version": SCHEDULE_RECEIPT_SCHEMA_VERSION_V1,
            "controller_id": self._controller_id,
            "schedule_sha256": self._schedule_sha256,
            "cell_count": self.cell_count,
            "maximum_generation_limit": max(
                runtime.identity.generation_limit for runtime in self._cells.values()
            ),
            "state_counts": state_counts,
            "auto_batch_binding": _plain_json(self._auto_batch_binding),
            "cells": cells,
            "contract": {
                "metadata_only": True,
                "live_controller_integration": False,
                "cuda_execution_performed": False,
                "simulation_execution_performed": False,
                "production_promotion_claimed": False,
                "branch_and_rng_authority": "cuda_architecture_b_checkpoint_branch",
                "parameter_authority": "cuda_architecture_b_patch_p2_cell_params_v1",
                "summary_authority": "cuda_architecture_b_cell_decision_summary_v1",
                "batch_authority": "cuda_architecture_b_auto_batch_v1",
                "fork_execution_authority": (
                    "cuda_architecture_b_boundary_controller_v1+"
                    "cuda_architecture_b_checkpoint_clone_v1"
                ),
            },
        }


def create_cell_schedule_v1(
    specification: Mapping[str, Any],
) -> CudaArchitectureBCellScheduleV1:
    raw = _exact_fields(specification, _SCHEDULE_FIELDS, path="schedule")
    if raw["schema_version"] != SCHEDULE_SPEC_SCHEMA_VERSION_V1:
        _fail("schedule.schema_version", "is not the supported v1 schema")
    controller_id = _stable_identifier(
        raw["controller_id"], path="schedule.controller_id"
    )
    cell_count = _exact_int(raw["cell_count"], path="schedule.cell_count", minimum=1)
    cells_value = raw["cells"]
    if not isinstance(cells_value, Sequence) or isinstance(
        cells_value, (str, bytes, bytearray)
    ):
        _fail("schedule.cells", "must be a sequence")
    if len(cells_value) != cell_count:
        _fail("schedule.cells", "length must exactly match configurable cell_count")

    auto_batch_binding, receipt_digest = _accepted_auto_batch_binding(
        raw["auto_batch_receipt"], cell_count=cell_count
    )
    parsed: list[tuple[int, int, int, CudaCheckpointBranch, PatchP2CellParamsV1]] = []
    for position, value in enumerate(cells_value):
        path = f"schedule.cells[{position}]"
        cell = _exact_fields(value, _CELL_FIELDS, path=path)
        cell_index = _exact_int(cell["cell_index"], path=f"{path}.cell_index")
        stable_cell_id = _exact_int(
            cell["stable_cell_id"], path=f"{path}.stable_cell_id"
        )
        generation_limit = _exact_int(
            cell["generation_limit"], path=f"{path}.generation_limit", minimum=1
        )
        try:
            branch = validate_checkpoint_branch(cell["branch"])
        except CudaCheckpointBranchError as exc:
            raise CellScheduleContractError(f"{path}.branch: {exc}") from exc
        parameters = _canonical_parameters(cell["parameters"], path=f"{path}.parameters")
        parsed.append(
            (cell_index, stable_cell_id, generation_limit, branch, parameters)
        )

    if sorted(row[0] for row in parsed) != list(range(cell_count)):
        _fail("schedule.cells.cell_index", "must be unique and contiguous from zero")
    stable_ids = [row[1] for row in parsed]
    if len(stable_ids) != len(set(stable_ids)):
        _fail("schedule.cells.stable_cell_id", "must be unique")
    branch_ids = [row[3].branch_id for row in parsed]
    if len(branch_ids) != len(set(branch_ids)):
        _fail("schedule.cells.branch", "contains duplicate branch_id values")
    try:
        validated_branches = validate_checkpoint_branch_batch(
            tuple(row[3] for row in sorted(parsed))
        )
    except CudaCheckpointBranchError as exc:
        raise CellScheduleContractError(f"schedule.cells.branch: {exc}") from exc

    identities: list[_CellIdentityV1] = []
    for row, branch in zip(sorted(parsed), validated_branches, strict=True):
        cell_index, stable_cell_id, generation_limit, _branch, parameters = row
        identities.append(
            _CellIdentityV1(
                cell_index=cell_index,
                stable_cell_id=stable_cell_id,
                generation_limit=generation_limit,
                start_generation=int(
                    branch.branch_identity["source_checkpoint"]["generation"]
                ),
                branch=branch,
                parameters=parameters,
                parameter_digest=patch_p2_cell_params_digest_v1(parameters),
            )
        )
    return CudaArchitectureBCellScheduleV1(
        controller_id=controller_id,
        cells=tuple(identities),
        auto_batch_binding=auto_batch_binding,
        auto_batch_receipt_sha256=receipt_digest,
    )


__all__ = [
    "BOUNDARY_COMMAND_KINDS_V1",
    "CELL_STATE_ACTIVE_V1",
    "CELL_STATE_CANCELLED_V1",
    "CELL_STATE_COMPLETED_V1",
    "CELL_STATE_FAILED_V1",
    "COMMAND_CANCEL_V1",
    "COMMAND_CHECKPOINT_AND_FORK_V1",
    "COMMAND_CHECKPOINT_V1",
    "CellDispatchCohortPlanV1",
    "CellDispatchPlanReceiptV1",
    "CudaArchitectureBCellScheduleV1",
    "CellScheduleContractError",
    "CellScheduleMutationReceiptV1",
    "CellScheduleStateError",
    "MUTATION_RECEIPT_SCHEMA_VERSION_V1",
    "DISPATCH_PLAN_SCHEMA_VERSION_V1",
    "SCHEDULE_RECEIPT_SCHEMA_VERSION_V1",
    "SCHEDULE_SPEC_SCHEMA_VERSION_V1",
    "TERMINAL_CELL_STATES_V1",
    "create_cell_schedule_v1",
]
