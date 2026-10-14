"""Execute the two preregistered candidate storage modes and compare them.

This boundary is qualification-only.  It consumes the immutable mixed-mode
composer artifact, derives both resident requests from its auxiliary parity
rows, executes the existing candidate segment owner, and reports only the five
comparisons preregistered by that artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Final

from engine.native_runtime.inheritance_cross_path_qualified_artifact_v1 import (
    CandidateInheritanceCrossPathArtifactV1,
    InheritanceCrossPathCandidateArtifactLoaderV1,
    UPSTREAM_BINDING_IDS,
)

from .canonical import canonical_sha256
from .inheritance_cross_path_candidate_plan_v1 import (
    InheritanceCrossPathCandidatePlanV1,
)
from .inheritance_cross_path_candidate_segment_owner_v1 import (
    CandidateExposureCapacityV1,
    InheritanceCrossPathCandidateSegmentOwnerV1,
    InheritanceCrossPathCandidateSegmentRequestV1,
)
from .production_engine_provider import PROFILE_ID
from .inheritance_cross_path_storage_parity_v1 import (
    CANDIDATE_STORAGE_PARITY_COMPARISON_RECEIPT_SCHEMA_VERSION,
)
from .mixed_mode_resident_isolation_plan_composer_v1 import (
    STORAGE_PARITY_FIELDS,
    verify_composed_plan_artifact_v1,
)
from .providers import ResidentBatchRequestV1


RESULT_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_candidate_storage_parity_execution_v1"
)
COMPARISON_RECEIPT_SCHEMA_VERSION: Final[str] = (
    CANDIDATE_STORAGE_PARITY_COMPARISON_RECEIPT_SCHEMA_VERSION
)
_STORAGE_MODES: Final[tuple[str, str]] = ("exact_full", "compact_complete")
_ELIGIBILITY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "eligible_snapshot_refs",
        "primary_eligible_snapshot_ref",
        "mate_eligible_snapshot_ref",
    }
)
_AGGREGATE_FIELDS: Final[tuple[str, str]] = (
    "primary_eligibility_opportunity_counts",
    "mate_eligibility_opportunity_counts",
)


class InheritanceCrossPathCandidateStorageParityExecutorError(RuntimeError):
    """A parity execution lacks preregistered authority or exact evidence."""


def _fail(
    message: str,
) -> InheritanceCrossPathCandidateStorageParityExecutorError:
    return InheritanceCrossPathCandidateStorageParityExecutorError(message)


def _copy(value: Any, *, name: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise _fail(f"{name} must contain canonical JSON data") from exc


def _digest(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _fail(f"{name} must be a lowercase SHA-256 digest")
    return value


def _safe_id(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in {".", ".."}
    ):
        raise _fail(f"{name} must be a safe non-empty identifier")
    return value


def _absolute_root(value: Any, *, name: str) -> Path:
    if not isinstance(value, str):
        raise _fail(f"{name} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or path.is_symlink():
        raise _fail(f"{name} must be an absolute non-symlink path")
    return path


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _fail(f"{name} must be an integer >= {minimum}")
    return value


def _artifact_sha256s(
    plan: InheritanceCrossPathCandidatePlanV1,
) -> dict[str, str]:
    artifacts = plan.plan.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        UPSTREAM_BINDING_IDS
    ):
        raise _fail("candidate plan lacks its four artifact bindings")
    result: dict[str, str] = {}
    for artifact_id in UPSTREAM_BINDING_IDS:
        row = artifacts[artifact_id]
        if not isinstance(row, Mapping):
            raise _fail(f"candidate artifact row is malformed: {artifact_id}")
        result[artifact_id] = _digest(
            row.get("sha256"), name=f"artifacts.{artifact_id}.sha256"
        )
    return result


def _candidate_loader_matches(
    loader: InheritanceCrossPathCandidateArtifactLoaderV1,
    plan: InheritanceCrossPathCandidatePlanV1,
) -> None:
    if not isinstance(loader, InheritanceCrossPathCandidateArtifactLoaderV1):
        raise _fail("candidate loader must be the qualification-only loader")
    if (
        loader.candidate_authority_sha256 != plan.candidate_authority_sha256
        or loader.candidate_bindings != _artifact_sha256s(plan)
    ):
        raise _fail("candidate loader differs from its immutable candidate plan")


def _template_for(
    plan: InheritanceCrossPathCandidatePlanV1,
    physical: Mapping[str, Any],
    logical_template_cell_ids: Mapping[str, str] | None,
) -> dict[str, Any]:
    cells = plan.plan.get("cells")
    config = physical.get("config")
    if not isinstance(cells, list) or not isinstance(config, Mapping):
        raise _fail("parity request or candidate template cells are malformed")
    logical_id = _safe_id(
        config.get("logical_cell_id"), name="physical_order.logical_cell_id"
    )
    mapped_id = (
        logical_template_cell_ids.get(logical_id)
        if logical_template_cell_ids is not None
        else logical_id
    )
    direct = [
        cell
        for cell in cells
        if isinstance(cell, Mapping) and cell.get("cell_id") == mapped_id
    ]
    if len(direct) == 1:
        matches = direct
    else:
        matches = []
        for cell in cells:
            dimensions = cell.get("dimensions") if isinstance(cell, Mapping) else None
            if (
                isinstance(dimensions, Mapping)
                and dimensions.get("inheritance_mode") == physical.get("mode_id")
                and dimensions.get("reproduction_path")
                == config.get("reproduction_path")
                and dimensions.get("paired_seed") == physical.get("seed")
            ):
                matches.append(cell)
    if len(matches) != 1:
        raise _fail(f"candidate template selection is not unique: {logical_id}")
    template = _copy(matches[0], name=f"candidate template {logical_id}")
    dimensions = template.get("dimensions")
    if (
        not isinstance(dimensions, Mapping)
        or dimensions.get("inheritance_mode") != physical.get("mode_id")
        or dimensions.get("reproduction_path") != config.get("reproduction_path")
        or dimensions.get("paired_seed") != physical.get("seed")
    ):
        raise _fail(f"candidate template differs from preregistration: {logical_id}")
    return template


def _candidate_request(
    *,
    auxiliary: Mapping[str, Any],
    candidate_plan: InheritanceCrossPathCandidatePlanV1,
    logical_template_cell_ids: Mapping[str, str] | None,
) -> InheritanceCrossPathCandidateSegmentRequestV1:
    physical_order = auxiliary.get("physical_order")
    if not isinstance(physical_order, list) or not physical_order:
        raise _fail("auxiliary parity request has no physical order")
    ordered = sorted(
        physical_order,
        key=lambda row: row.get("batch_position") if isinstance(row, Mapping) else -1,
    )
    if [
        row.get("batch_position") if isinstance(row, Mapping) else None
        for row in ordered
    ] != list(range(len(ordered))):
        raise _fail("auxiliary parity batch positions are not contiguous")

    cells: list[dict[str, Any]] = []
    for position, physical in enumerate(ordered):
        if not isinstance(physical, Mapping):
            raise _fail("auxiliary physical-order row is malformed")
        occurrence_id = _safe_id(
            physical.get("occurrence_id"), name="physical_order.occurrence_id"
        )
        template = _template_for(candidate_plan, physical, logical_template_cell_ids)
        template["cell_id"] = occurrence_id
        if "ordinal" in template:
            template["ordinal"] = position
        cells.append(template)
    cell_ids = tuple(cell["cell_id"] for cell in cells)
    if len(set(cell_ids)) != len(cell_ids):
        raise _fail("auxiliary parity occurrence IDs are not unique")

    generations = _integer(auxiliary.get("generations"), name="generations", minimum=1)
    ticks = _integer(
        auxiliary.get("ticks_per_generation"),
        name="ticks_per_generation",
        minimum=1,
    )
    split = _integer(
        auxiliary.get("checkpoint_split_generation"),
        name="checkpoint_split_generation",
        minimum=1,
    )
    if split >= generations:
        raise _fail("parity checkpoint split must be interior")
    if auxiliary.get("reporting_enabled") is not True:
        raise _fail("preregistered storage parity requires reporting")

    request_id = _safe_id(auxiliary.get("request_id"), name="request_id")
    plan = _copy(candidate_plan.plan, name="candidate plan")
    plan.pop("plan_sha256", None)
    plan.pop("comparison_branches", None)
    plan["cells"] = cells
    plan["physical_batches"] = [
        {"ordinal": 0, "batch_id": request_id, "cell_ids": list(cell_ids)}
    ]
    execution = dict(plan.get("execution") or {})
    execution.pop("technical_gate", None)
    execution.update(
        {
            "generations": generations,
            "ticks_per_generation": ticks,
            "checkpoints": [split, generations],
            "checkpoint_storage_mode": "disk",
            "reporting_enabled": True,
        }
    )
    plan["execution"] = execution
    plan["plan_sha256"] = canonical_sha256(plan)

    parameter_names = tuple(sorted(cells[0].get("parameters", {})))
    seed_names = tuple(sorted(cells[0].get("seeds", {})))
    if (
        not parameter_names
        or not seed_names
        or any(
            tuple(sorted(cell.get("parameters", {}))) != parameter_names
            or tuple(sorted(cell.get("seeds", {}))) != seed_names
            for cell in cells
        )
    ):
        raise _fail("candidate parity templates have inconsistent parameter fields")
    batch_request = ResidentBatchRequestV1(
        profile_id=PROFILE_ID,
        batch_id=request_id,
        cell_ids=cell_ids,
        generations=generations,
        ticks_per_generation=ticks,
        checkpoint_milestones=(split, generations),
        parameter_arrays=tuple(
            (
                name,
                tuple(cell["parameters"][name] for cell in cells),
            )
            for name in parameter_names
        ),
        seed_arrays=tuple(
            (name, tuple(cell["seeds"][name] for cell in cells)) for name in seed_names
        ),
        agent_id_allocation_generation_count=generations,
        checkpoint_storage_mode="disk",
        reporting_enabled=True,
        execution_route_sha256=candidate_plan.candidate_authority_sha256,
    )
    return InheritanceCrossPathCandidateSegmentRequestV1(
        request_id=request_id,
        candidate_authority_sha256=candidate_plan.candidate_authority_sha256,
        candidate_artifact_sha256s=_artifact_sha256s(candidate_plan),
        plan=plan,
        batch_request=batch_request,
        runtime_root=_absolute_root(auxiliary.get("run_root"), name="run_root"),
        exposure_capacity=CandidateExposureCapacityV1(
            storage_mode=str(auxiliary["storage_mode"]),
            attempt_capacity=int(auxiliary["exposure"]["A"]),
            snapshot_capacity=int(auxiliary["exposure"]["S"]),
            eligible_capacity=int(auxiliary["exposure"]["C"]),
            aggregate_capacity=int(auxiliary["exposure"]["K"]),
        ),
        split_generation=split,
        fixed_action_required=False,
        storage_observation_required=True,
    )


def _verify_result_binding(
    result: Mapping[str, Any],
    *,
    request: InheritanceCrossPathCandidateSegmentRequestV1,
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise _fail("candidate owner result must be an object")
    row = _copy(result, name="candidate owner result")
    supplied = _digest(row.pop("result_sha256", None), name="result_sha256")
    if canonical_sha256(row) != supplied:
        raise _fail("candidate result digest is stale")
    row["result_sha256"] = supplied
    expected = {
        "request_sha256": request.request_sha256,
        "candidate_authority_sha256": request.candidate_authority_sha256,
        "plan_sha256": request.plan["plan_sha256"],
        "cell_ids": list(request.batch_request.cell_ids),
        "reporting_enabled": True,
        "production_authority": False,
        "fresh_owner": True,
    }
    if any(row.get(name) != value for name, value in expected.items()):
        raise _fail("candidate result differs from its request binding")
    if row.get("candidate_artifact_sha256s") != dict(
        request.candidate_artifact_sha256s
    ):
        raise _fail("candidate result binds another artifact chain")

    source_owner = _safe_id(row.get("source_owner_id"), name="source_owner_id")
    terminal_owner = _safe_id(row.get("terminal_owner_id"), name="terminal_owner_id")
    if source_owner == terminal_owner:
        raise _fail("candidate fresh-owner identities are not disjoint")
    restores = row.get("restore_receipts")
    if not isinstance(restores, list) or len(restores) != len(
        request.batch_request.cell_ids
    ):
        raise _fail("candidate result lacks fresh-owner restore receipts")
    for cell_id, receipt in zip(request.batch_request.cell_ids, restores, strict=True):
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("cell_id") != cell_id
            or receipt.get("source_owner_id") != source_owner
            or receipt.get("terminal_owner_id") != terminal_owner
            or receipt.get("checkpoint_reloaded") is not True
            or receipt.get("checkpoint_source_sha256")
            != receipt.get("restored_source_sha256")
        ):
            raise _fail("candidate fresh-owner restore binding is incomplete")

    segments = row.get("segments")
    split = request.split_generation
    generations = request.batch_request.generations
    expected_ranges = ((0, split), (split, generations))
    if not isinstance(segments, list) or len(segments) != len(expected_ranges):
        raise _fail("candidate result lacks its two checkpoint segments")
    for index, (segment, bounds) in enumerate(
        zip(segments, expected_ranges, strict=True)
    ):
        if (
            not isinstance(segment, Mapping)
            or segment.get("segment_index") != index
            or (
                segment.get("start_generation"),
                segment.get("end_generation"),
            )
            != bounds
            or segment.get("cell_ids") != list(request.batch_request.cell_ids)
            or not isinstance(segment.get("mechanics_checkpoint_digests"), Mapping)
            or not isinstance(segment.get("segment_artifacts"), list)
            or len(segment["segment_artifacts"]) != len(request.batch_request.cell_ids)
        ):
            raise _fail("candidate segment/result binding is incomplete")
    terminal_checkpoint = row.get("terminal_checkpoint")
    if (
        not isinstance(terminal_checkpoint, Mapping)
        or terminal_checkpoint.get("generation") != generations
        or segments[-1].get("checkpoint") != terminal_checkpoint
        or not isinstance(row.get("metrics_rows"), list)
        or not isinstance(row.get("provider_result"), Mapping)
    ):
        raise _fail("candidate terminal checkpoint/result binding is incomplete")
    receipts = row.get("qualification_receipts")
    dispatch = (
        receipts.get("resident_dispatch_observations")
        if isinstance(receipts, Mapping)
        else None
    )
    if not isinstance(dispatch, list) or len(dispatch) != len(segments):
        raise _fail("candidate result lacks ordered resident dispatch evidence")
    for index, observation in enumerate(dispatch):
        if (
            not isinstance(observation, Mapping)
            or observation.get("dispatch_index") != index
            or not isinstance(observation.get("cell_state_sha256"), list)
            or not isinstance(observation.get("cell_rng_sha256"), list)
            or len(observation["cell_state_sha256"])
            != len(request.batch_request.cell_ids)
            or len(observation["cell_rng_sha256"])
            != len(request.batch_request.cell_ids)
        ):
            raise _fail("candidate resident dispatch evidence is malformed")
        for field in ("cell_state_sha256", "cell_rng_sha256"):
            for value in observation[field]:
                _digest(value, name=f"resident_dispatch.{field}")
    return row


def _storage_observations(
    result: Mapping[str, Any],
    *,
    storage_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for segment in result["segments"]:
        for artifact in segment["segment_artifacts"]:
            capacity = (
                artifact.get("capacity_semantics")
                if isinstance(artifact, Mapping)
                else None
            )
            cell = artifact.get("cell") if isinstance(artifact, Mapping) else None
            rows = cell.get("attempts") if isinstance(cell, Mapping) else None
            if (
                not isinstance(capacity, Mapping)
                or capacity.get("storage_mode") != storage_mode
                or not isinstance(rows, list)
            ):
                raise _fail(f"{storage_mode} exposure observation is malformed")
            normalized_rows: list[dict[str, Any]] = []
            for attempt in rows:
                if not isinstance(attempt, Mapping):
                    raise _fail(f"{storage_mode} attempt row is malformed")
                present = _ELIGIBILITY_FIELDS.intersection(attempt)
                if storage_mode == "exact_full":
                    if not isinstance(attempt.get("eligible_snapshot_refs"), list):
                        raise _fail(
                            "exact_full attempt lacks authoritative eligibility"
                        )
                elif present:
                    raise _fail(
                        "compact attempt exposes forbidden per-attempt eligibility"
                    )
                normalized = dict(attempt)
                for field in _ELIGIBILITY_FIELDS:
                    normalized.pop(field, None)
                normalized_rows.append(normalized)
            aggregate = {}
            for field in _AGGREGATE_FIELDS:
                value = cell.get(field)
                if not isinstance(value, list):
                    raise _fail(f"{storage_mode} exposure lacks aggregate eligibility")
                aggregate[field] = value
            identity = {
                "segment_index": segment["segment_index"],
                "cell_id": artifact.get("cell_id"),
            }
            attempts.append({**identity, "attempts": normalized_rows})
            aggregates.append({**identity, **aggregate})
    return attempts, aggregates


def _comparison_receipt(
    *,
    composed_artifact_sha256: str,
    comparison_contract: Mapping[str, Any],
    exact_request: InheritanceCrossPathCandidateSegmentRequestV1,
    compact_request: InheritanceCrossPathCandidateSegmentRequestV1,
    exact_result: Mapping[str, Any],
    compact_result: Mapping[str, Any],
) -> dict[str, Any]:
    contract_fields = comparison_contract.get("required_comparisons")
    if contract_fields != list(STORAGE_PARITY_FIELDS):
        raise _fail("comparison field set differs from preregistration")
    exact_attempts, exact_aggregates = _storage_observations(
        exact_result, storage_mode="exact_full"
    )
    compact_attempts, compact_aggregates = _storage_observations(
        compact_result, storage_mode="compact_complete"
    )
    exact_dispatch = exact_result["qualification_receipts"][
        "resident_dispatch_observations"
    ]
    compact_dispatch = compact_result["qualification_receipts"][
        "resident_dispatch_observations"
    ]
    comparison_values = {
        "attempt_outcome_parent_commit_donor_identity_exact": (
            exact_attempts,
            compact_attempts,
        ),
        "aggregate_eligibility_equivalent": (
            exact_aggregates,
            compact_aggregates,
        ),
        "final_state_exact": (
            [row["cell_state_sha256"] for row in exact_dispatch],
            [row["cell_state_sha256"] for row in compact_dispatch],
        ),
        "rng_state_exact": (
            [row["cell_rng_sha256"] for row in exact_dispatch],
            [row["cell_rng_sha256"] for row in compact_dispatch],
        ),
        "checkpoint_exact": (
            [row["mechanics_checkpoint_digests"] for row in exact_result["segments"]],
            [row["mechanics_checkpoint_digests"] for row in compact_result["segments"]],
        ),
    }
    if set(comparison_values) != set(STORAGE_PARITY_FIELDS):
        raise _fail("implemented comparison set differs from preregistration")
    comparisons: dict[str, Any] = {}
    for field in STORAGE_PARITY_FIELDS:
        exact_value, compact_value = comparison_values[field]
        if exact_value != compact_value:
            raise _fail(f"cross-storage parity failed: {field}")
        comparisons[field] = {
            "ok": True,
            "evidence_sha256": canonical_sha256(exact_value),
        }

    owner_ids = {
        exact_result["source_owner_id"],
        exact_result["terminal_owner_id"],
        compact_result["source_owner_id"],
        compact_result["terminal_owner_id"],
    }
    if len(owner_ids) != 4:
        raise _fail("cross-storage executions did not use four fresh owner identities")
    unsigned = {
        "schema_version": COMPARISON_RECEIPT_SCHEMA_VERSION,
        "composed_plan_artifact_sha256": composed_artifact_sha256,
        "comparison_contract_sha256": _digest(
            comparison_contract.get("comparison_contract_sha256"),
            name="comparison_contract_sha256",
        ),
        "candidate_plan_sha256s": {
            "exact_full": exact_request.plan["plan_sha256"],
            "compact_complete": compact_request.plan["plan_sha256"],
        },
        "candidate_request_sha256s": {
            "exact_full": exact_request.request_sha256,
            "compact_complete": compact_request.request_sha256,
        },
        "candidate_result_sha256s": {
            "exact_full": exact_result["result_sha256"],
            "compact_complete": compact_result["result_sha256"],
        },
        "fresh_owner_identity_sha256": canonical_sha256(sorted(owner_ids)),
        "reporting_enabled": True,
        "production_authority": False,
        "comparisons": comparisons,
    }
    return {**unsigned, "comparison_receipt_sha256": canonical_sha256(unsigned)}


@dataclass(frozen=True, slots=True)
class InheritanceCrossPathCandidateStorageParityExecutionV1:
    """The two bound candidate results and their narrow parity receipt."""

    exact_full_request: InheritanceCrossPathCandidateSegmentRequestV1
    compact_complete_request: InheritanceCrossPathCandidateSegmentRequestV1
    exact_full_result: Mapping[str, Any]
    compact_complete_result: Mapping[str, Any]
    comparison_receipt: Mapping[str, Any]

    def payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "production_authority": False,
            "requests": {
                "exact_full": self.exact_full_request.payload(),
                "compact_complete": self.compact_complete_request.payload(),
            },
            "results": {
                "exact_full": _copy(self.exact_full_result, name="exact_full_result"),
                "compact_complete": _copy(
                    self.compact_complete_result,
                    name="compact_complete_result",
                ),
            },
            "comparison_receipt": _copy(
                self.comparison_receipt, name="comparison_receipt"
            ),
        }
        return {**payload, "execution_sha256": canonical_sha256(payload)}


class InheritanceCrossPathCandidateStorageParityExecutorV1:
    """Tombstone for the retired cross-storage candidate executor."""

    def __init__(
        self,
        *,
        composed_plan_artifact: Mapping[str, Any],
        exact_full_plan: InheritanceCrossPathCandidatePlanV1,
        compact_complete_plan: InheritanceCrossPathCandidatePlanV1,
        exact_full_loader: InheritanceCrossPathCandidateArtifactLoaderV1,
        compact_complete_loader: InheritanceCrossPathCandidateArtifactLoaderV1,
        logical_template_cell_ids: Mapping[str, str] | None = None,
        candidate_owner: InheritanceCrossPathCandidateSegmentOwnerV1 | None = None,
    ) -> None:
        raise _fail(
            "P12_RETIRED_INHERITANCE_AUTHORITY "
            "candidate storage-parity execution is disabled"
        )
        # Historical implementation retained temporarily for evidence reading.
        verified = verify_composed_plan_artifact_v1(composed_plan_artifact)
        if (
            type(exact_full_plan) is not InheritanceCrossPathCandidatePlanV1
            or exact_full_plan.storage_mode != "exact_full"
            or type(compact_complete_plan) is not InheritanceCrossPathCandidatePlanV1
            or compact_complete_plan.storage_mode != "compact_complete"
        ):
            raise _fail("storage parity requires typed exact and compact plans")
        _candidate_loader_matches(exact_full_loader, exact_full_plan)
        _candidate_loader_matches(compact_complete_loader, compact_complete_plan)
        parity = verified.get("storage_mode_parity")
        if not isinstance(parity, Mapping):
            raise _fail("verified composed artifact lacks storage parity")
        exact_auxiliary = parity.get("exact_full_request")
        compact_auxiliary = parity.get("compact_complete_request")
        comparison = parity.get("comparison_contract")
        if not all(
            isinstance(value, Mapping)
            for value in (exact_auxiliary, compact_auxiliary, comparison)
        ):
            raise _fail("verified storage parity rows are malformed")
        for mode, auxiliary, plan in (
            ("exact_full", exact_auxiliary, exact_full_plan),
            ("compact_complete", compact_auxiliary, compact_complete_plan),
        ):
            if (
                auxiliary.get("storage_mode") != mode
                or auxiliary.get("candidate_authority_sha256")
                != plan.candidate_authority_sha256
                or auxiliary.get("candidate_artifact_sha256")
                != plan.binding_manifest_sha256
            ):
                raise _fail(f"{mode} preregistration differs from its candidate plan")
        roots = {
            _absolute_root(auxiliary[field], name=f"{mode}.{field}")
            for mode, auxiliary in (
                ("exact_full", exact_auxiliary),
                ("compact_complete", compact_auxiliary),
            )
            for field in ("run_root", "owner_root", "checkpoint_root")
        }
        if len(roots) != 6:
            raise _fail("cross-storage runtime roots are not disjoint")
        physical_order = exact_auxiliary.get("physical_order")
        logical_ids = (
            {
                row.get("config", {}).get("logical_cell_id")
                for row in physical_order
                if isinstance(row, Mapping) and isinstance(row.get("config"), Mapping)
            }
            if isinstance(physical_order, list)
            else set()
        )
        if logical_template_cell_ids is not None and (
            not isinstance(logical_template_cell_ids, Mapping)
            or set(logical_template_cell_ids) != logical_ids
            or any(
                not isinstance(template_id, str) or not template_id
                for template_id in logical_template_cell_ids.values()
            )
            or len(set(logical_template_cell_ids.values()))
            != len(logical_template_cell_ids)
        ):
            raise _fail("logical template-cell mapping must cover parity cells exactly")

        exact_request = _candidate_request(
            auxiliary=exact_auxiliary,
            candidate_plan=exact_full_plan,
            logical_template_cell_ids=logical_template_cell_ids,
        )
        compact_request = _candidate_request(
            auxiliary=compact_auxiliary,
            candidate_plan=compact_complete_plan,
            logical_template_cell_ids=logical_template_cell_ids,
        )
        if (
            exact_request.batch_request.cell_ids
            != compact_request.batch_request.cell_ids
            or exact_request.batch_request.generations
            != compact_request.batch_request.generations
            or exact_request.batch_request.ticks_per_generation
            != compact_request.batch_request.ticks_per_generation
            or exact_request.batch_request.checkpoint_milestones
            != compact_request.batch_request.checkpoint_milestones
            or exact_request.batch_request.parameter_arrays
            != compact_request.batch_request.parameter_arrays
            or exact_request.batch_request.seed_arrays
            != compact_request.batch_request.seed_arrays
        ):
            raise _fail("candidate requests differ outside storage authority")
        exact_artifacts = _artifact_sha256s(exact_full_plan)
        compact_artifacts = _artifact_sha256s(compact_complete_plan)
        if (
            exact_artifacts["cuda_extension"] != compact_artifacts["cuda_extension"]
            or exact_artifacts["extension_manifest"]
            != compact_artifacts["extension_manifest"]
            or exact_artifacts["inheritance_registry"]
            == compact_artifacts["inheritance_registry"]
            or exact_artifacts["telemetry_contract"]
            == compact_artifacts["telemetry_contract"]
        ):
            raise _fail("candidate artifact chains do not vary only storage authority")
        owner = candidate_owner or InheritanceCrossPathCandidateSegmentOwnerV1()
        if not callable(getattr(owner, "execute", None)):
            raise _fail("candidate_owner must expose execute")

        self.composed_plan_artifact = verified
        self.exact_full_plan = exact_full_plan
        self.compact_complete_plan = compact_complete_plan
        self.exact_full_loader = exact_full_loader
        self.compact_complete_loader = compact_complete_loader
        self.candidate_owner = owner
        self.exact_full_request = exact_request
        self.compact_complete_request = compact_request
        self.comparison_contract = dict(comparison)

    def execute(
        self,
    ) -> InheritanceCrossPathCandidateStorageParityExecutionV1:
        """Load, execute, bind, and compare both preregistered candidates."""

        exact_candidate = self.exact_full_loader()
        compact_candidate = self.compact_complete_loader()
        for mode, candidate in (
            ("exact_full", exact_candidate),
            ("compact_complete", compact_candidate),
        ):
            if not isinstance(candidate, CandidateInheritanceCrossPathArtifactV1):
                raise _fail(f"{mode} loader returned another artifact type")
        exact_result = _verify_result_binding(
            self.candidate_owner.execute(exact_candidate, self.exact_full_request),
            request=self.exact_full_request,
        )
        compact_result = _verify_result_binding(
            self.candidate_owner.execute(
                compact_candidate, self.compact_complete_request
            ),
            request=self.compact_complete_request,
        )
        receipt = _comparison_receipt(
            composed_artifact_sha256=self.composed_plan_artifact["artifact_sha256"],
            comparison_contract=self.comparison_contract,
            exact_request=self.exact_full_request,
            compact_request=self.compact_complete_request,
            exact_result=exact_result,
            compact_result=compact_result,
        )
        return InheritanceCrossPathCandidateStorageParityExecutionV1(
            exact_full_request=self.exact_full_request,
            compact_complete_request=self.compact_complete_request,
            exact_full_result=exact_result,
            compact_complete_result=compact_result,
            comparison_receipt=receipt,
        )


__all__ = [
    "COMPARISON_RECEIPT_SCHEMA_VERSION",
    "InheritanceCrossPathCandidateStorageParityExecutionV1",
    "InheritanceCrossPathCandidateStorageParityExecutorError",
    "InheritanceCrossPathCandidateStorageParityExecutorV1",
    "RESULT_SCHEMA_VERSION",
]
