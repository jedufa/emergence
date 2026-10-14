"""Exact-full versus compact-complete real-resident parity verifier."""

from __future__ import annotations

from typing import Any, Mapping

from .canonical import canonical_sha256
from .inheritance_cross_path_qualification_receipts_v1 import (
    PARITY_DIMENSION_IDS,
    PARITY_EXACTNESS_IDS,
    CrossStorageParityEvidenceBindingV1,
    CrossStorageParityPlanV1,
    CrossStorageParityResultV1,
    CrossStorageResidentCaptureV1,
)
from .mixed_mode_resident_isolation_plan_composer_v1 import (
    STORAGE_COMPACT_COMPLETE_REQUEST_ID,
    STORAGE_EXACT_FULL_REQUEST_ID,
    STORAGE_PARITY_COMPARISON_SCHEMA_VERSION,
    STORAGE_PARITY_FIELDS,
    STORAGE_PARITY_REQUEST_SCHEMA_VERSION,
)

CANDIDATE_STORAGE_PARITY_COMPARISON_RECEIPT_SCHEMA_VERSION = (
    "eiel_inheritance_cross_path_candidate_storage_parity_comparison_v1"
)


class InheritanceCrossPathStorageParityError(ValueError):
    """The two qualification-only executions are not comparable or exact."""


def _fail(message: str) -> InheritanceCrossPathStorageParityError:
    return InheritanceCrossPathStorageParityError(message)


def _attempt_authority(
    cell: Mapping[str, Any],
    *,
    exact_full: bool,
) -> str:
    attempts = cell.get("attempts")
    if not isinstance(attempts, list):
        raise _fail("storage observation lacks exact attempt rows")
    normalized = []
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise _fail("storage observation attempt row is malformed")
        row = dict(attempt)
        eligibility_fields = {
            "eligible_snapshot_refs",
            "primary_eligible_snapshot_ref",
            "mate_eligible_snapshot_ref",
        }
        present = eligibility_fields.intersection(row)
        if exact_full and not present:
            raise _fail("exact_full attempt lacks authoritative eligibility sets")
        if not exact_full and present:
            raise _fail("compact attempt exposes forbidden per-attempt eligibility")
        if exact_full:
            for field in eligibility_fields:
                row.pop(field, None)
        normalized.append(row)
    return canonical_sha256(normalized)


def _eligibility_authority(cell: Mapping[str, Any]) -> str:
    names = (
        "primary_eligibility_opportunity_counts",
        "mate_eligibility_opportunity_counts",
    )
    if any(not isinstance(cell.get(name), list) for name in names):
        raise _fail("storage observation lacks aggregate eligibility")
    return canonical_sha256({name: cell[name] for name in names})


def cross_storage_candidate_dimension_sha256s_v1(
    plan: Mapping[str, Any],
    *,
    cell_ids: list[str] | tuple[str, ...],
) -> dict[str, str]:
    """Project the exact dimension convention used by preregistration."""

    cells = plan.get("cells")
    execution = plan.get("execution")
    if (
        not isinstance(cells, list)
        or not isinstance(execution, Mapping)
        or [cell.get("cell_id") for cell in cells if isinstance(cell, Mapping)]
        != list(cell_ids)
        or any(
            not isinstance(cell.get("parameters"), Mapping)
            or not isinstance(cell.get("seeds"), Mapping)
            for cell in cells
            if isinstance(cell, Mapping)
        )
    ):
        raise _fail("candidate plan lacks ordered parity dimensions")
    dimensions = {
        "cells": canonical_sha256(cells),
        "cell_order": canonical_sha256(list(cell_ids)),
        "modes": canonical_sha256(
            [dict(cell["parameters"]) for cell in cells]
        ),
        "seeds": canonical_sha256([dict(cell["seeds"]) for cell in cells]),
        "generations": canonical_sha256(execution.get("generations")),
        "checkpoints": canonical_sha256(execution.get("checkpoints")),
    }
    if set(dimensions) != set(PARITY_DIMENSION_IDS):
        raise _fail("candidate parity dimension IDs drifted")
    return dimensions


def compose_cross_storage_resident_captures_v1(
    *,
    preregistered_plan: CrossStorageParityPlanV1,
    exact_full_result_sha256: str,
    compact_complete_result_sha256: str,
) -> tuple[CrossStorageResidentCaptureV1, CrossStorageResidentCaptureV1]:
    """Compose the two typed capture payloads used by the physical manifest."""

    if not isinstance(preregistered_plan, CrossStorageParityPlanV1):
        raise _fail("cross-storage captures require their typed preregistered plan")
    for name, value in (
        ("exact_full_result_sha256", exact_full_result_sha256),
        ("compact_complete_result_sha256", compact_complete_result_sha256),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise _fail(f"{name} must be a lowercase SHA-256 digest")
    exact = CrossStorageResidentCaptureV1(
        storage_mode="exact_full",
        capture_sha256=exact_full_result_sha256,
        extension_sha256=preregistered_plan.exact_full_artifacts[
            "cuda_extension"
        ],
        extension_manifest_sha256=preregistered_plan.exact_full_artifacts[
            "extension_manifest"
        ],
        native_abi=preregistered_plan.native_abi,
        registry_sha256=preregistered_plan.exact_full_artifacts[
            "inheritance_registry"
        ],
        telemetry_contract_sha256=preregistered_plan.exact_full_artifacts[
            "telemetry_contract"
        ],
        dimension_sha256s=preregistered_plan.dimension_sha256s,
    )
    compact = CrossStorageResidentCaptureV1(
        storage_mode="compact_complete",
        capture_sha256=compact_complete_result_sha256,
        extension_sha256=preregistered_plan.compact_complete_artifacts[
            "cuda_extension"
        ],
        extension_manifest_sha256=preregistered_plan.compact_complete_artifacts[
            "extension_manifest"
        ],
        native_abi=preregistered_plan.native_abi,
        registry_sha256=preregistered_plan.compact_complete_artifacts[
            "inheritance_registry"
        ],
        telemetry_contract_sha256=preregistered_plan.compact_complete_artifacts[
            "telemetry_contract"
        ],
        dimension_sha256s=preregistered_plan.dimension_sha256s,
    )
    return exact, compact


def _verified_auxiliary_request(
    value: Mapping[str, Any],
    *,
    storage_mode: str,
    request_id: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{storage_mode} auxiliary request is malformed")
    row = dict(value)
    supplied = row.pop("request_sha256", None)
    if (
        not isinstance(supplied, str)
        or canonical_sha256(row) != supplied
        or row.get("schema_version") != STORAGE_PARITY_REQUEST_SCHEMA_VERSION
        or row.get("request_id") != request_id
        or row.get("execution_route") != "auxiliary_storage_parity_only"
        or row.get("isolation_callback_authorized") is not False
        or row.get("storage_mode") != storage_mode
        or row.get("reporting_enabled") is not True
    ):
        raise _fail(f"{storage_mode} auxiliary request is not preregistered")
    exposure = row.get("exposure")
    if not isinstance(exposure, Mapping):
        raise _fail(f"{storage_mode} auxiliary exposure is malformed")
    if storage_mode == "exact_full":
        valid_eligibility = (
            exposure.get("eligibility_representation")
            == "role_specific_per_attempt_parent_id_sets_v1"
            and exposure.get("per_attempt_eligible_parent_id_sets_available")
            is True
        )
    else:
        valid_eligibility = (
            exposure.get("eligibility_representation")
            == "role_specific_aggregate_opportunity_counts_v1"
            and exposure.get("per_attempt_eligible_parent_id_sets_available")
            is False
            and exposure.get("S") == 1
        )
    if not valid_eligibility:
        raise _fail(f"{storage_mode} auxiliary eligibility authority is invalid")
    return {**row, "request_sha256": supplied}


def _verify_comparison_contract(
    value: Mapping[str, Any],
    *,
    exact_request_sha256: str,
    compact_request_sha256: str,
) -> str:
    if not isinstance(value, Mapping):
        raise _fail("storage parity comparison contract is malformed")
    row = dict(value)
    supplied = row.pop("comparison_contract_sha256", None)
    if (
        not isinstance(supplied, str)
        or canonical_sha256(row) != supplied
        or row.get("schema_version")
        != STORAGE_PARITY_COMPARISON_SCHEMA_VERSION
        or row.get("exact_full_request_sha256") != exact_request_sha256
        or row.get("compact_complete_request_sha256")
        != compact_request_sha256
        or row.get("required_comparisons") != list(STORAGE_PARITY_FIELDS)
        or row.get("exact_full_eligible_parent_id_sets_authoritative")
        is not True
        or row.get("compact_per_attempt_eligible_parent_id_sets_forbidden")
        is not True
        or row.get("compact_eligibility_comparison")
        != "role_specific_aggregate_opportunity_counts_v1"
        or row.get("promotion_authority") is not False
    ):
        raise _fail("storage parity comparison contract is invalid")
    return supplied


def _verified_candidate_result(
    value: Mapping[str, Any],
    *,
    storage_mode: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{storage_mode} candidate result is not qualification evidence")
    row = dict(value)
    supplied = row.pop("result_sha256", None)
    if not isinstance(supplied, str) or canonical_sha256(row) != supplied:
        raise _fail(f"{storage_mode} candidate result digest is stale")
    return {**row, "result_sha256": supplied}


def _verify_execution_comparison_receipt(
    value: Mapping[str, Any],
    *,
    comparison_contract_sha256: str,
    exact_result: Mapping[str, Any],
    compact_result: Mapping[str, Any],
) -> None:
    if not isinstance(value, Mapping):
        raise _fail("candidate execution comparison receipt is malformed")
    row = dict(value)
    supplied = row.pop("comparison_receipt_sha256", None)
    expected_plan_sha256s = {
        "exact_full": exact_result.get("plan_sha256"),
        "compact_complete": compact_result.get("plan_sha256"),
    }
    expected_request_sha256s = {
        "exact_full": exact_result.get("request_sha256"),
        "compact_complete": compact_result.get("request_sha256"),
    }
    expected_result_sha256s = {
        "exact_full": exact_result.get("result_sha256"),
        "compact_complete": compact_result.get("result_sha256"),
    }
    comparisons = row.get("comparisons")
    if (
        not isinstance(supplied, str)
        or canonical_sha256(row) != supplied
        or row.get("schema_version")
        != CANDIDATE_STORAGE_PARITY_COMPARISON_RECEIPT_SCHEMA_VERSION
        or row.get("comparison_contract_sha256") != comparison_contract_sha256
        or row.get("candidate_plan_sha256s") != expected_plan_sha256s
        or row.get("candidate_request_sha256s") != expected_request_sha256s
        or row.get("candidate_result_sha256s") != expected_result_sha256s
        or row.get("reporting_enabled") is not True
        or row.get("production_authority") is not False
        or not isinstance(comparisons, Mapping)
        or set(comparisons) != set(STORAGE_PARITY_FIELDS)
        or any(
            not isinstance(comparisons.get(field), Mapping)
            or comparisons[field].get("ok") is not True
            or not isinstance(comparisons[field].get("evidence_sha256"), str)
            for field in STORAGE_PARITY_FIELDS
        )
    ):
        raise _fail("candidate execution comparison receipt is invalid")


def verify_cross_path_storage_parity_v1(
    *,
    preregistered_plan: CrossStorageParityPlanV1,
    exact_full_request: Mapping[str, Any],
    compact_complete_request: Mapping[str, Any],
    comparison_contract: Mapping[str, Any],
    execution_comparison_receipt: Mapping[str, Any],
    capture_manifest_sha256: str,
    exact_full_result: Mapping[str, Any],
    compact_complete_result: Mapping[str, Any],
) -> CrossStorageParityResultV1:
    """Return the committed typed result only after every obligation is exact."""

    if not isinstance(preregistered_plan, CrossStorageParityPlanV1):
        raise _fail("cross-storage parity requires its typed preregistered plan")
    exact_request = _verified_auxiliary_request(
        exact_full_request,
        storage_mode="exact_full",
        request_id=STORAGE_EXACT_FULL_REQUEST_ID,
    )
    compact_request = _verified_auxiliary_request(
        compact_complete_request,
        storage_mode="compact_complete",
        request_id=STORAGE_COMPACT_COMPLETE_REQUEST_ID,
    )
    comparison_sha256 = _verify_comparison_contract(
        comparison_contract,
        exact_request_sha256=exact_request["request_sha256"],
        compact_request_sha256=compact_request["request_sha256"],
    )
    exact_full_result = _verified_candidate_result(
        exact_full_result,
        storage_mode="exact_full",
    )
    compact_complete_result = _verified_candidate_result(
        compact_complete_result,
        storage_mode="compact_complete",
    )
    _verify_execution_comparison_receipt(
        execution_comparison_receipt,
        comparison_contract_sha256=comparison_sha256,
        exact_result=exact_full_result,
        compact_result=compact_complete_result,
    )
    if (
        not isinstance(capture_manifest_sha256, str)
        or len(capture_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in capture_manifest_sha256
        )
    ):
        raise _fail("capture manifest SHA-256 is invalid")
    if (
        exact_request.get("candidate_authority_sha256")
        == compact_request.get("candidate_authority_sha256")
        or exact_request.get("candidate_artifact_sha256")
        == compact_request.get("candidate_artifact_sha256")
        or exact_request.get("physical_order")
        != compact_request.get("physical_order")
        or any(
            exact_request.get(field) != compact_request.get(field)
            for field in (
                "generations",
                "ticks_per_generation",
                "checkpoint_split_generation",
            )
        )
        or any(
            exact_request["exposure"].get(field)
            != compact_request["exposure"].get(field)
            for field in ("A", "C", "K")
        )
    ):
        raise _fail("auxiliary parity requests differ outside storage authority")
    for name, result in (
        ("exact_full", exact_full_result),
        ("compact_complete", compact_complete_result),
    ):
        if (
            not isinstance(result, Mapping)
            or result.get("production_authority") is not False
            or not isinstance(result.get("result_sha256"), str)
            or not isinstance(result.get("segments"), list)
            or not isinstance(result.get("qualification_receipts"), Mapping)
            or not isinstance(result.get("parity_dimension_sha256s"), Mapping)
            or not isinstance(
                result.get("candidate_artifact_sha256s"), Mapping
            )
            or result.get("candidate_authority_sha256")
            != (
                exact_request["candidate_authority_sha256"]
                if name == "exact_full"
                else compact_request["candidate_authority_sha256"]
            )
        ):
            raise _fail(f"{name} candidate result is not qualification evidence")
    if (
        exact_full_result.get("cell_ids") != compact_complete_result.get("cell_ids")
        or len(exact_full_result["segments"])
        != len(compact_complete_result["segments"])
    ):
        raise _fail("storage candidates differ in physical cells or segments")
    if (
        dict(exact_full_result["candidate_artifact_sha256s"])
        != dict(preregistered_plan.exact_full_artifacts)
        or dict(compact_complete_result["candidate_artifact_sha256s"])
        != dict(preregistered_plan.compact_complete_artifacts)
        or dict(exact_full_result["parity_dimension_sha256s"])
        != dict(preregistered_plan.dimension_sha256s)
        or dict(compact_complete_result["parity_dimension_sha256s"])
        != dict(preregistered_plan.dimension_sha256s)
    ):
        raise _fail("candidate artifact chain or dimensions differ from preregistration")

    exact_dispatch = exact_full_result["qualification_receipts"].get(
        "resident_dispatch_observations"
    )
    compact_dispatch = compact_complete_result["qualification_receipts"].get(
        "resident_dispatch_observations"
    )
    if (
        not isinstance(exact_dispatch, list)
        or not exact_dispatch
        or exact_dispatch != compact_dispatch
    ):
        raise _fail("storage candidates differ in resident state or RNG")

    for exact_segment, compact_segment in zip(
        exact_full_result["segments"],
        compact_complete_result["segments"],
        strict=True,
    ):
        if (
            exact_segment.get("start_generation")
            != compact_segment.get("start_generation")
            or exact_segment.get("end_generation")
            != compact_segment.get("end_generation")
            or exact_segment.get("cell_ids") != compact_segment.get("cell_ids")
            or exact_segment.get("mechanics_checkpoint_digests")
            != compact_segment.get("mechanics_checkpoint_digests")
        ):
            raise _fail("storage candidates differ in checkpointed mechanics")
        exact_artifacts = exact_segment.get("segment_artifacts")
        compact_artifacts = compact_segment.get("segment_artifacts")
        if (
            not isinstance(exact_artifacts, list)
            or not isinstance(compact_artifacts, list)
            or len(exact_artifacts) != len(compact_artifacts)
        ):
            raise _fail("storage candidates lack ordered exposure artifacts")
        for exact_artifact, compact_artifact in zip(
            exact_artifacts, compact_artifacts, strict=True
        ):
            exact_capacity = exact_artifact.get("capacity_semantics")
            compact_capacity = compact_artifact.get("capacity_semantics")
            if (
                not isinstance(exact_capacity, Mapping)
                or exact_capacity.get("storage_mode") != "exact_full"
                or not isinstance(compact_capacity, Mapping)
                or compact_capacity.get("storage_mode") != "compact_complete"
                or exact_artifact.get("cell_id") != compact_artifact.get("cell_id")
            ):
                raise _fail("storage mode authority or physical order is invalid")
            exact_cell = exact_artifact.get("cell")
            compact_cell = compact_artifact.get("cell")
            if not isinstance(exact_cell, Mapping) or not isinstance(
                compact_cell, Mapping
            ):
                raise _fail("storage cell exposure is malformed")
            attempt = _attempt_authority(exact_cell, exact_full=True)
            eligibility = _eligibility_authority(exact_cell)
            if (
                attempt != _attempt_authority(compact_cell, exact_full=False)
                or eligibility != _eligibility_authority(compact_cell)
            ):
                raise _fail("storage candidates differ in attempts or eligibility")
    exact, compact = compose_cross_storage_resident_captures_v1(
        preregistered_plan=preregistered_plan,
        exact_full_result_sha256=exact_full_result["result_sha256"],
        compact_complete_result_sha256=compact_complete_result["result_sha256"],
    )
    return CrossStorageParityResultV1(
        plan_sha256=preregistered_plan.plan_sha256,
        capture_manifest_sha256=capture_manifest_sha256,
        exact_full=exact,
        compact_complete=compact,
        exactness={name: True for name in PARITY_EXACTNESS_IDS},
    )


def verify_cross_path_storage_parity_evidence_v1(
    *,
    preregistered_plan: CrossStorageParityPlanV1,
    exact_full_request: Mapping[str, Any],
    compact_complete_request: Mapping[str, Any],
    comparison_contract: Mapping[str, Any],
    execution_comparison_receipt: Mapping[str, Any],
    capture_manifest_sha256: str,
    exact_full_result: Mapping[str, Any],
    compact_complete_result: Mapping[str, Any],
) -> tuple[
    CrossStorageParityPlanV1,
    CrossStorageParityResultV1,
    CrossStorageParityEvidenceBindingV1,
]:
    """Return the canonical typed plan, verifier result, and evidence binding."""

    result = verify_cross_path_storage_parity_v1(
        preregistered_plan=preregistered_plan,
        exact_full_request=exact_full_request,
        compact_complete_request=compact_complete_request,
        comparison_contract=comparison_contract,
        execution_comparison_receipt=execution_comparison_receipt,
        capture_manifest_sha256=capture_manifest_sha256,
        exact_full_result=exact_full_result,
        compact_complete_result=compact_complete_result,
    )
    evidence = CrossStorageParityEvidenceBindingV1(
        parity_plan_sha256=preregistered_plan.plan_sha256,
        capture_manifest_sha256=result.capture_manifest_sha256,
        parity_result_sha256=result.result_sha256,
        exact_full=result.exact_full,
        compact_complete=result.compact_complete,
        exactness=result.exactness,
    )
    return preregistered_plan, result, evidence


__all__ = [
    "CANDIDATE_STORAGE_PARITY_COMPARISON_RECEIPT_SCHEMA_VERSION",
    "InheritanceCrossPathStorageParityError",
    "compose_cross_storage_resident_captures_v1",
    "cross_storage_candidate_dimension_sha256s_v1",
    "verify_cross_path_storage_parity_evidence_v1",
    "verify_cross_path_storage_parity_v1",
]
