from __future__ import annotations

"""Deterministic metadata-only planner for one resident generation boundary.

Scientific score construction remains owned by the caller that produced the
compact ``CellDecisionSummaryV1``. This module only applies an explicit,
mechanical pruning/fork policy and composes existing branch and clone contracts.
It never imports a tensor framework, inspects resident allocation contents, or
executes a transfer.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Final

from .cuda_architecture_b_cell_decision_summary_v1 import (
    CellDecisionSummaryContractError,
    CellDecisionSummaryV1,
    RankedCellDecisionV1,
    rank_cell_decision_summary_cpu_oracle_v1,
)
from .cuda_architecture_b_checkpoint_branch import (
    CudaCheckpointBranch,
    CudaCheckpointBranchError,
    create_checkpoint_branch,
    validate_branch_key,
    validate_checkpoint_branch_batch,
)
from .cuda_architecture_b_checkpoint_clone_v1 import (
    CheckpointCloneContractError,
    CheckpointCloneFieldDescriptorV1,
    CheckpointClonePlanV1,
    plan_checkpoint_clone_v1,
)


BOUNDARY_PLAN_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_cuda_architecture_b_boundary_controller_plan_v1"
)
BOUNDARY_CLONE_PLAN_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_cuda_architecture_b_boundary_controller_clone_plan_v1"
)
BOUNDARY_BRANCH_TRAJECTORY_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_cuda_architecture_b_boundary_branch_trajectory_v1"
)
RANK_TIEBREAK_REJECT_V1: Final[str] = "reject"
RANK_TIEBREAK_STABLE_CELL_ID_ASCENDING_V1: Final[str] = (
    "stable_cell_id_ascending"
)
SEEDED_STABLE_CELL_ID_SHA256_ASSIGNMENT_V1: Final[str] = (
    "seeded_stable_cell_id_sha256"
)
CONTAINS_FULL_SIMULATION_STATE_V1: Final[bool] = False
FULL_STATE_PAYLOAD_ACCEPTED_V1: Final[bool] = False
DEVICE_TO_HOST_OPERATIONS_V1: Final[int] = 0
DEVICE_TO_HOST_BYTES_V1: Final[int] = 0
SCIENTIFIC_SCORING_SEMANTICS_SELECTED_V1: Final[bool] = False

_ASSIGNMENT_DOMAIN_V1: Final[bytes] = b"EIEL\0ARCHB_BOUNDARY_ASSIGNMENT\0V1\0"
_MAX_UINT64: Final[int] = (1 << 64) - 1
_SHA256_HEX_LENGTH: Final[int] = 64
_POLICY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "policy_id",
        "policy_digest",
        "generation_index",
        "source_generation",
        "generation_boundary_complete",
        "prune_count",
        "planner_seed",
        "rank_tiebreak",
        "fork_source_assignment",
        "branch_key_prefix",
        "source_checkpoint_projection_sha256",
        "state_manifest",
        "source_lineage",
    }
)
_SOURCE_LINEAGE_FIELDS: Final[frozenset[str]] = frozenset(
    {"stable_cell_id", "source_rng_state_sha256", "source_branch_id"}
)
_FIELD_DESCRIPTOR_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "field_name",
        "name",
        "pointer_identity",
        "shape",
        "cell_stride",
        "cell_stride_bytes",
        "bytes_per_cell",
        "element_size_bytes",
        "semantic_classification",
        "copy_classification",
    }
)


class BoundaryControllerContractError(ValueError):
    """Raised when a boundary plan would be unsafe or ambiguous."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256(encoded).hexdigest()


def _fail(path: str, message: str) -> None:
    raise BoundaryControllerContractError(f"{path}: {message}")


def _exact_int(value: Any, *, path: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(path, f"must be an exact integer >= {minimum}")
    return value


def _sha256_hex(value: Any, *, path: str) -> str:
    if (
        type(value) is not str
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(path, "must be exactly 64 lowercase hexadecimal characters")
    return value


def _stable_ascii_identifier(value: Any, *, path: str) -> str:
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


def _optional_source_branch_id(value: Any, *, path: str) -> str | None:
    if value is None:
        return None
    return _stable_ascii_identifier(value, path=path)


@dataclass(frozen=True)
class BoundarySourceLineageV1:
    """Compact caller receipt needed for exact RNG inheritance."""

    stable_cell_id: int
    source_rng_state_sha256: str
    source_branch_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_cell_id": self.stable_cell_id,
            "source_rng_state_sha256": self.source_rng_state_sha256,
            "source_branch_id": self.source_branch_id,
        }


@dataclass(frozen=True)
class BoundaryPruningForkPolicyV1:
    """Caller-declared mechanical policy; it contains no scoring formula."""

    policy_id: str
    policy_digest: str
    generation_index: int
    source_generation: int
    generation_boundary_complete: bool
    prune_count: int
    planner_seed: int
    rank_tiebreak: str
    fork_source_assignment: str
    branch_key_prefix: str
    source_checkpoint_projection_sha256: str
    state_manifest: Mapping[str, Any]
    source_lineage: tuple[BoundarySourceLineageV1, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_digest": self.policy_digest,
            "generation_index": self.generation_index,
            "source_generation": self.source_generation,
            "generation_boundary_complete": self.generation_boundary_complete,
            "prune_count": self.prune_count,
            "planner_seed": self.planner_seed,
            "rank_tiebreak": self.rank_tiebreak,
            "fork_source_assignment": self.fork_source_assignment,
            "branch_key_prefix": self.branch_key_prefix,
            "source_checkpoint_projection_sha256": (
                self.source_checkpoint_projection_sha256
            ),
            "state_manifest": json.loads(
                json.dumps(
                    self.state_manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
            ),
            "source_lineage": [row.to_dict() for row in self.source_lineage],
        }


@dataclass(frozen=True)
class BoundaryCellV1:
    rank_index: int
    cell_index: int
    stable_cell_id: int

    def to_dict(self) -> dict[str, int]:
        return {
            "rank_index": self.rank_index,
            "cell_index": self.cell_index,
            "stable_cell_id": self.stable_cell_id,
        }


@dataclass(frozen=True)
class BoundaryForkEntryV1:
    fork_entry_index: int
    source_cell_index: int
    source_stable_cell_id: int
    target_cell_index: int
    target_stable_cell_id: int
    clone_source_plan_index: int
    branch_key: str
    branch_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fork_entry_index": self.fork_entry_index,
            "source_cell_index": self.source_cell_index,
            "source_stable_cell_id": self.source_stable_cell_id,
            "target_cell_index": self.target_cell_index,
            "target_stable_cell_id": self.target_stable_cell_id,
            "clone_source_plan_index": self.clone_source_plan_index,
            "branch_key": self.branch_key,
            "branch_id": self.branch_id,
        }


@dataclass(frozen=True)
class BoundaryClonePlanV1:
    """Small multi-source wrapper around existing single-source clone plans."""

    source_plans: tuple[CheckpointClonePlanV1, ...]

    @property
    def source_plan_count(self) -> int:
        return len(self.source_plans)

    @property
    def target_count(self) -> int:
        return sum(len(plan.target_cell_indices) for plan in self.source_plans)

    @property
    def clone_entry_count(self) -> int:
        return sum(plan.clone_entry_count for plan in self.source_plans)

    @property
    def total_copy_bytes(self) -> int:
        return sum(plan.total_copy_bytes for plan in self.source_plans)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": BOUNDARY_CLONE_PLAN_SCHEMA_VERSION_V1,
            "source_plan_count": self.source_plan_count,
            "target_count": self.target_count,
            "clone_entry_count": self.clone_entry_count,
            "total_copy_bytes": self.total_copy_bytes,
            "source_plans": [plan.to_dict() for plan in self.source_plans],
            "source_preservation_contract": {
                "sources_are_read_only": True,
                "targets_exclude_all_sources": True,
                "target_slots_are_unique": True,
            },
            "transfer_contract": {
                "copy_kind": "device_to_device",
                "accepted_tensor_state_only": True,
                "full_semantic_checkpoint_clone_qualified": False,
                "device_to_host_operations": DEVICE_TO_HOST_OPERATIONS_V1,
                "device_to_host_bytes": DEVICE_TO_HOST_BYTES_V1,
                "hidden_host_materialization": False,
                "execution_performed": False,
                "zero_d2h_execution_qualified": False,
            },
        }

    @property
    def plan_sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["plan_sha256"] = self.plan_sha256
        return payload


@dataclass(frozen=True)
class CudaArchitectureBBoundaryPlanV1:
    generation_index: int
    source_generation: int
    score_policy_id: str
    score_policy_digest: str
    pruning_fork_policy: BoundaryPruningForkPolicyV1
    kept_source_cells: tuple[BoundaryCellV1, ...]
    pruned_source_cells: tuple[BoundaryCellV1, ...]
    target_cells: tuple[BoundaryCellV1, ...]
    source_to_target_clone_entries: tuple[BoundaryForkEntryV1, ...]
    branch_records: tuple[CudaCheckpointBranch, ...]
    clone_plan: BoundaryClonePlanV1

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": BOUNDARY_PLAN_SCHEMA_VERSION_V1,
            "generation_index": self.generation_index,
            "source_generation": self.source_generation,
            "score_contract": {
                "score_policy_id": self.score_policy_id,
                "score_policy_digest": self.score_policy_digest,
                "caller_basis_used_verbatim": True,
                "scientific_scoring_semantics_selected": (
                    SCIENTIFIC_SCORING_SEMANTICS_SELECTED_V1
                ),
                "rank_tiebreak": self.pruning_fork_policy.rank_tiebreak,
            },
            "pruning_fork_policy": self.pruning_fork_policy.to_dict(),
            "kept_source_cells": [cell.to_dict() for cell in self.kept_source_cells],
            "pruned_source_cells": [
                cell.to_dict() for cell in self.pruned_source_cells
            ],
            "target_cells": [cell.to_dict() for cell in self.target_cells],
            "source_to_target_clone_entries": [
                entry.to_dict() for entry in self.source_to_target_clone_entries
            ],
            "branch_records": [branch.to_record() for branch in self.branch_records],
            "rng_lineage": [
                branch.rng_lineage_provenance for branch in self.branch_records
            ],
            "clone_plan": self.clone_plan.to_dict(),
            "boundary_contract": {
                "contains_full_simulation_state": (
                    CONTAINS_FULL_SIMULATION_STATE_V1
                ),
                "full_state_payload_accepted": FULL_STATE_PAYLOAD_ACCEPTED_V1,
                "device_to_host_operations": DEVICE_TO_HOST_OPERATIONS_V1,
                "device_to_host_bytes": DEVICE_TO_HOST_BYTES_V1,
                "hidden_host_materialization": False,
                "execution_performed": False,
                "source_cells_preserved_by_plan": True,
                "targets_are_distinct_slots": True,
                "live_fork_execution_qualified": False,
                "full_semantic_checkpoint_clone_qualified": False,
            },
        }

    @property
    def plan_sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["plan_sha256"] = self.plan_sha256
        return payload


def _normalize_source_lineage(value: Any) -> tuple[BoundarySourceLineageV1, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        _fail("policy.source_lineage", "must be a sequence of compact records")
    normalized: list[BoundarySourceLineageV1] = []
    stable_ids: set[int] = set()
    for index, raw in enumerate(value):
        path = f"policy.source_lineage[{index}]"
        if isinstance(raw, BoundarySourceLineageV1):
            record = raw
        elif isinstance(raw, Mapping):
            unknown = set(raw) - _SOURCE_LINEAGE_FIELDS
            missing = _SOURCE_LINEAGE_FIELDS - set(raw) - {"source_branch_id"}
            if unknown or missing:
                _fail(
                    path,
                    "fields do not match the compact schema; "
                    f"unknown={sorted(unknown)}, "
                    f"missing={sorted(missing)}",
                )
            record = BoundarySourceLineageV1(
                stable_cell_id=raw.get("stable_cell_id"),
                source_rng_state_sha256=raw.get("source_rng_state_sha256"),
                source_branch_id=raw.get("source_branch_id"),
            )
        else:
            _fail(path, "must be BoundarySourceLineageV1 or a mapping")
        stable_id = _exact_int(
            record.stable_cell_id, path=f"{path}.stable_cell_id"
        )
        if stable_id in stable_ids:
            _fail(path, f"stable_cell_id collision for {stable_id}")
        stable_ids.add(stable_id)
        normalized.append(
            BoundarySourceLineageV1(
                stable_cell_id=stable_id,
                source_rng_state_sha256=_sha256_hex(
                    record.source_rng_state_sha256,
                    path=f"{path}.source_rng_state_sha256",
                ),
                source_branch_id=_optional_source_branch_id(
                    record.source_branch_id, path=f"{path}.source_branch_id"
                ),
            )
        )
    if not normalized:
        _fail("policy.source_lineage", "must contain at least one record")
    return tuple(sorted(normalized, key=lambda row: row.stable_cell_id))


def _normalize_policy(
    value: BoundaryPruningForkPolicyV1 | Mapping[str, Any],
) -> BoundaryPruningForkPolicyV1:
    if isinstance(value, BoundaryPruningForkPolicyV1):
        raw = value.to_dict()
    elif isinstance(value, Mapping):
        unknown = set(value) - _POLICY_FIELDS
        missing = _POLICY_FIELDS - set(value)
        if unknown or missing:
            _fail(
                "policy",
                f"fields do not match the compact schema; unknown={sorted(unknown)}, "
                f"missing={sorted(missing)}",
            )
        raw = dict(value)
    else:
        _fail("policy", "must be BoundaryPruningForkPolicyV1 or a mapping")

    policy_id = _stable_ascii_identifier(raw["policy_id"], path="policy.policy_id")
    policy_digest = _sha256_hex(
        raw["policy_digest"], path="policy.policy_digest"
    )
    generation_index = _exact_int(
        raw["generation_index"], path="policy.generation_index"
    )
    source_generation = _exact_int(
        raw["source_generation"], path="policy.source_generation"
    )
    if raw["generation_boundary_complete"] is not True:
        _fail("policy.generation_boundary_complete", "must be exactly true")
    prune_count = _exact_int(raw["prune_count"], path="policy.prune_count", minimum=1)
    planner_seed = _exact_int(raw["planner_seed"], path="policy.planner_seed")
    if planner_seed > _MAX_UINT64:
        _fail("policy.planner_seed", "must fit uint64")
    rank_tiebreak = raw["rank_tiebreak"]
    if rank_tiebreak not in {
        RANK_TIEBREAK_REJECT_V1,
        RANK_TIEBREAK_STABLE_CELL_ID_ASCENDING_V1,
    }:
        _fail(
            "policy.rank_tiebreak",
            "must be 'reject' or explicit 'stable_cell_id_ascending'",
        )
    assignment = raw["fork_source_assignment"]
    if assignment != SEEDED_STABLE_CELL_ID_SHA256_ASSIGNMENT_V1:
        _fail(
            "policy.fork_source_assignment",
            f"must be {SEEDED_STABLE_CELL_ID_SHA256_ASSIGNMENT_V1!r}",
        )
    branch_key_prefix = raw["branch_key_prefix"]
    try:
        validate_branch_key(branch_key_prefix)
    except CudaCheckpointBranchError as exc:
        raise BoundaryControllerContractError(
            f"policy.branch_key_prefix: {exc}"
        ) from exc
    state_manifest_value = raw["state_manifest"]
    if not isinstance(state_manifest_value, Mapping):
        _fail("policy.state_manifest", "must be a compact metadata mapping")
    try:
        state_manifest = json.loads(
            json.dumps(
                state_manifest_value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise BoundaryControllerContractError(
            f"policy.state_manifest: must be canonical JSON metadata: {exc}"
        ) from exc

    return BoundaryPruningForkPolicyV1(
        policy_id=policy_id,
        policy_digest=policy_digest,
        generation_index=generation_index,
        source_generation=source_generation,
        generation_boundary_complete=True,
        prune_count=prune_count,
        planner_seed=planner_seed,
        rank_tiebreak=rank_tiebreak,
        fork_source_assignment=assignment,
        branch_key_prefix=branch_key_prefix,
        source_checkpoint_projection_sha256=_sha256_hex(
            raw["source_checkpoint_projection_sha256"],
            path="policy.source_checkpoint_projection_sha256",
        ),
        state_manifest=state_manifest,
        source_lineage=_normalize_source_lineage(raw["source_lineage"]),
    )


def _strict_field_descriptors(
    field_descriptors: Sequence[
        CheckpointCloneFieldDescriptorV1 | Mapping[str, Any]
    ],
) -> tuple[CheckpointCloneFieldDescriptorV1 | Mapping[str, Any], ...]:
    if not isinstance(field_descriptors, Sequence) or isinstance(
        field_descriptors, (str, bytes, bytearray)
    ):
        _fail("field_descriptors", "must be a nonempty sequence")
    if not field_descriptors:
        _fail("field_descriptors", "must be nonempty")
    result: list[CheckpointCloneFieldDescriptorV1 | Mapping[str, Any]] = []
    for index, descriptor in enumerate(field_descriptors):
        if isinstance(descriptor, CheckpointCloneFieldDescriptorV1):
            result.append(descriptor)
            continue
        if not isinstance(descriptor, Mapping):
            _fail(
                f"field_descriptors[{index}]",
                "must be CheckpointCloneFieldDescriptorV1 or a mapping",
            )
        unknown = set(descriptor) - _FIELD_DESCRIPTOR_FIELDS
        if unknown:
            _fail(
                f"field_descriptors[{index}]",
                "contains non-descriptor payload fields: " + ", ".join(sorted(unknown)),
            )
        result.append(dict(descriptor))
    return tuple(result)


def _cell(row: RankedCellDecisionV1) -> BoundaryCellV1:
    return BoundaryCellV1(
        rank_index=row.rank_index,
        cell_index=row.cell_index,
        stable_cell_id=row.stable_cell_id,
    )


def _reject_ambiguous_ties(
    ranking: Sequence[RankedCellDecisionV1],
    *,
    rank_tiebreak: str,
) -> None:
    tied: dict[tuple[float, ...], list[int]] = {}
    for row in ranking:
        tied.setdefault(row.score_basis, []).append(row.stable_cell_id)
    collisions = [sorted(ids) for ids in tied.values() if len(ids) > 1]
    if collisions and rank_tiebreak != RANK_TIEBREAK_STABLE_CELL_ID_ASCENDING_V1:
        _fail(
            "policy.rank_tiebreak",
            "ambiguous score-basis ties require explicit "
            f"{RANK_TIEBREAK_STABLE_CELL_ID_ASCENDING_V1!r}; ties={collisions}",
        )


def _assigned_source(
    kept_by_stable_id: Sequence[RankedCellDecisionV1],
    *,
    target_stable_cell_id: int,
    policy: BoundaryPruningForkPolicyV1,
) -> RankedCellDecisionV1:
    preimage = (
        _ASSIGNMENT_DOMAIN_V1
        + bytes.fromhex(policy.policy_digest)
        + policy.planner_seed.to_bytes(8, "big", signed=False)
        + target_stable_cell_id.to_bytes(8, "big", signed=False)
    )
    selector = int.from_bytes(sha256(preimage).digest(), "big")
    return kept_by_stable_id[selector % len(kept_by_stable_id)]


def _branch_key(
    policy: BoundaryPruningForkPolicyV1,
    *,
    source_stable_cell_id: int,
    target_stable_cell_id: int,
) -> str:
    key = (
        f"{policy.branch_key_prefix}.g{policy.source_generation}."
        f"s{source_stable_cell_id}.t{target_stable_cell_id}"
    )
    try:
        return validate_branch_key(key)
    except CudaCheckpointBranchError as exc:
        raise BoundaryControllerContractError(
            f"derived branch key is invalid; use a shorter branch_key_prefix: {exc}"
        ) from exc


def _trajectory_projection(
    *,
    summary: CellDecisionSummaryV1,
    policy: BoundaryPruningForkPolicyV1,
    source_stable_cell_id: int,
    target_stable_cell_id: int,
) -> dict[str, Any]:
    return {
        "schema_version": BOUNDARY_BRANCH_TRAJECTORY_SCHEMA_VERSION_V1,
        "score_policy_id": summary.score_policy_id,
        "score_policy_digest": summary.score_policy_digest.hex(),
        "pruning_fork_policy_id": policy.policy_id,
        "pruning_fork_policy_digest": policy.policy_digest,
        "generation_index": policy.generation_index,
        "planner_seed": policy.planner_seed,
        "rank_tiebreak": policy.rank_tiebreak,
        "fork_source_assignment": policy.fork_source_assignment,
        "source_stable_cell_id": source_stable_cell_id,
        "target_stable_cell_id": target_stable_cell_id,
    }


def _build_boundary_plan(
    summary: CellDecisionSummaryV1,
    field_descriptors: Sequence[
        CheckpointCloneFieldDescriptorV1 | Mapping[str, Any]
    ],
    policy: BoundaryPruningForkPolicyV1 | Mapping[str, Any],
) -> CudaArchitectureBBoundaryPlanV1:
    normalized_policy = _normalize_policy(policy)
    descriptors = _strict_field_descriptors(field_descriptors)
    rankings = rank_cell_decision_summary_cpu_oracle_v1(summary)
    if normalized_policy.generation_index >= len(rankings):
        _fail("policy.generation_index", "is outside the compact summary")
    ranking = rankings[normalized_policy.generation_index]
    if len(ranking) != summary.cell_count:
        _fail(
            "summary",
            "selected boundary must have one successful rankable row per cell",
        )
    if normalized_policy.prune_count >= len(ranking):
        _fail("policy.prune_count", "must leave at least one kept source cell")
    _reject_ambiguous_ties(
        ranking, rank_tiebreak=normalized_policy.rank_tiebreak
    )

    kept_rows = tuple(ranking[: -normalized_policy.prune_count])
    pruned_rows = tuple(ranking[-normalized_policy.prune_count :])
    kept_by_stable_id = tuple(sorted(kept_rows, key=lambda row: row.stable_cell_id))
    targets_by_stable_id = tuple(
        sorted(pruned_rows, key=lambda row: row.stable_cell_id)
    )
    lineage_by_stable_id = {
        row.stable_cell_id: row for row in normalized_policy.source_lineage
    }
    kept_ids = {row.stable_cell_id for row in kept_rows}
    lineage_ids = set(lineage_by_stable_id)
    if lineage_ids != kept_ids:
        _fail(
            "policy.source_lineage",
            "must contain exactly the kept source stable_cell_ids; "
            f"missing={sorted(kept_ids - lineage_ids)}, "
            f"unexpected={sorted(lineage_ids - kept_ids)}",
        )

    assignments: list[tuple[RankedCellDecisionV1, RankedCellDecisionV1]] = []
    for target in targets_by_stable_id:
        assignments.append(
            (
                _assigned_source(
                    kept_by_stable_id,
                    target_stable_cell_id=target.stable_cell_id,
                    policy=normalized_policy,
                ),
                target,
            )
        )
    target_indices = [target.cell_index for _, target in assignments]
    if len(target_indices) != len(set(target_indices)):
        _fail("targets", "target slot collision")
    source_indices = {source.cell_index for source, _ in assignments}
    if source_indices.intersection(target_indices):
        _fail("targets", "source and target slots must be distinct")

    targets_by_source: dict[int, list[int]] = {}
    source_row_by_id: dict[int, RankedCellDecisionV1] = {}
    for source, target in assignments:
        source_row_by_id[source.stable_cell_id] = source
        targets_by_source.setdefault(source.stable_cell_id, []).append(
            target.cell_index
        )

    source_plans: list[CheckpointClonePlanV1] = []
    source_plan_index_by_stable_id: dict[int, int] = {}
    for source_stable_id in sorted(targets_by_source):
        source = source_row_by_id[source_stable_id]
        clone = plan_checkpoint_clone_v1(
            descriptors,
            state_manifest=normalized_policy.state_manifest,
            generation_boundary_complete=True,
            source_cell_index=source.cell_index,
            target_cell_indices=targets_by_source[source_stable_id],
        )
        if any(field.shape[0] != summary.cell_count for field in clone.physical_fields):
            _fail(
                "field_descriptors",
                "resident descriptor cell count must exactly match summary.cell_count",
            )
        source_plan_index_by_stable_id[source_stable_id] = len(source_plans)
        source_plans.append(clone)
    clone_plan = BoundaryClonePlanV1(source_plans=tuple(source_plans))

    branches: list[CudaCheckpointBranch] = []
    for source, target in assignments:
        lineage = lineage_by_stable_id[source.stable_cell_id]
        key = _branch_key(
            normalized_policy,
            source_stable_cell_id=source.stable_cell_id,
            target_stable_cell_id=target.stable_cell_id,
        )
        branches.append(
            create_checkpoint_branch(
                branch_key=key,
                source_checkpoint_projection_sha256=(
                    normalized_policy.source_checkpoint_projection_sha256
                ),
                source_generation=normalized_policy.source_generation,
                trajectory_projection=_trajectory_projection(
                    summary=summary,
                    policy=normalized_policy,
                    source_stable_cell_id=source.stable_cell_id,
                    target_stable_cell_id=target.stable_cell_id,
                ),
                source_rng_state_sha256=lineage.source_rng_state_sha256,
                inherited_rng_state_sha256=lineage.source_rng_state_sha256,
                source_branch_id=lineage.source_branch_id,
            )
        )
    validated_branches = validate_checkpoint_branch_batch(branches)

    fork_entries = tuple(
        BoundaryForkEntryV1(
            fork_entry_index=index,
            source_cell_index=source.cell_index,
            source_stable_cell_id=source.stable_cell_id,
            target_cell_index=target.cell_index,
            target_stable_cell_id=target.stable_cell_id,
            clone_source_plan_index=source_plan_index_by_stable_id[
                source.stable_cell_id
            ],
            branch_key=branch.branch_key,
            branch_id=branch.branch_id,
        )
        for index, ((source, target), branch) in enumerate(
            zip(assignments, validated_branches, strict=True)
        )
    )
    return CudaArchitectureBBoundaryPlanV1(
        generation_index=normalized_policy.generation_index,
        source_generation=normalized_policy.source_generation,
        score_policy_id=summary.score_policy_id,
        score_policy_digest=summary.score_policy_digest.hex(),
        pruning_fork_policy=normalized_policy,
        kept_source_cells=tuple(_cell(row) for row in kept_rows),
        pruned_source_cells=tuple(_cell(row) for row in pruned_rows),
        target_cells=tuple(_cell(row) for row in targets_by_stable_id),
        source_to_target_clone_entries=fork_entries,
        branch_records=validated_branches,
        clone_plan=clone_plan,
    )


def plan_cuda_architecture_b_boundary_controller_v1(
    summary: CellDecisionSummaryV1,
    field_descriptors: Sequence[
        CheckpointCloneFieldDescriptorV1 | Mapping[str, Any]
    ],
    policy: BoundaryPruningForkPolicyV1 | Mapping[str, Any],
) -> CudaArchitectureBBoundaryPlanV1:
    """Build one deterministic, source-preserving boundary metadata plan."""

    try:
        plan = _build_boundary_plan(summary, field_descriptors, policy)
        json.dumps(plan.to_dict(), allow_nan=False)
        return plan
    except BoundaryControllerContractError:
        raise
    except (
        CellDecisionSummaryContractError,
        CheckpointCloneContractError,
        CudaCheckpointBranchError,
        TypeError,
        ValueError,
    ) as exc:
        raise BoundaryControllerContractError(
            f"boundary plan rejected by composed contract: {exc}"
        ) from exc


# Concise aliases for callers that name the result rather than the controller.
plan_cuda_architecture_b_boundary_v1 = (
    plan_cuda_architecture_b_boundary_controller_v1
)
build_cuda_architecture_b_boundary_plan_v1 = (
    plan_cuda_architecture_b_boundary_controller_v1
)


__all__ = [
    "BOUNDARY_BRANCH_TRAJECTORY_SCHEMA_VERSION_V1",
    "BOUNDARY_CLONE_PLAN_SCHEMA_VERSION_V1",
    "BOUNDARY_PLAN_SCHEMA_VERSION_V1",
    "CONTAINS_FULL_SIMULATION_STATE_V1",
    "DEVICE_TO_HOST_BYTES_V1",
    "DEVICE_TO_HOST_OPERATIONS_V1",
    "FULL_STATE_PAYLOAD_ACCEPTED_V1",
    "RANK_TIEBREAK_REJECT_V1",
    "RANK_TIEBREAK_STABLE_CELL_ID_ASCENDING_V1",
    "SCIENTIFIC_SCORING_SEMANTICS_SELECTED_V1",
    "SEEDED_STABLE_CELL_ID_SHA256_ASSIGNMENT_V1",
    "BoundaryCellV1",
    "BoundaryClonePlanV1",
    "BoundaryControllerContractError",
    "BoundaryForkEntryV1",
    "BoundaryPruningForkPolicyV1",
    "BoundarySourceLineageV1",
    "CudaArchitectureBBoundaryPlanV1",
    "build_cuda_architecture_b_boundary_plan_v1",
    "plan_cuda_architecture_b_boundary_controller_v1",
    "plan_cuda_architecture_b_boundary_v1",
]
