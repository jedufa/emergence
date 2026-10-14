from __future__ import annotations

from copy import deepcopy

import pytest

from engine.campaigns.inheritance_cross_path_qualification_receipts_v1 import (
    CROSS_STORAGE_PARITY_RESULT_SCHEMA_VERSION,
    PARITY_DIMENSION_IDS,
    CrossStorageParityPlanV1,
)
from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.inheritance_cross_path_storage_parity_v1 import (
    InheritanceCrossPathStorageParityError,
    verify_cross_path_storage_parity_evidence_v1,
    verify_cross_path_storage_parity_v1,
)
from engine.campaigns.mixed_mode_resident_isolation_plan_composer_v1 import (
    STORAGE_PARITY_COMPARISON_SCHEMA_VERSION,
    STORAGE_PARITY_FIELDS,
    STORAGE_PARITY_REQUEST_SCHEMA_VERSION,
)


DIMENSIONS = {
    name: f"{index + 1:x}" * 64
    for index, name in enumerate(PARITY_DIMENSION_IDS)
}
EXACT_ARTIFACTS = {
    "cuda_extension": "a" * 64,
    "extension_manifest": "b" * 64,
    "inheritance_registry": "c" * 64,
    "telemetry_contract": "d" * 64,
}
COMPACT_ARTIFACTS = {
    **EXACT_ARTIFACTS,
    "inheritance_registry": "e" * 64,
    "telemetry_contract": "f" * 64,
}
EXACT_AUTHORITY = "7" * 64
COMPACT_AUTHORITY = "9" * 64
EXACT_CANDIDATE_ARTIFACT = "1" * 64
COMPACT_CANDIDATE_ARTIFACT = "2" * 64


def _plan() -> CrossStorageParityPlanV1:
    return CrossStorageParityPlanV1(
        native_abi="cross-path-abi-v1",
        exact_full_artifacts=EXACT_ARTIFACTS,
        compact_complete_artifacts=COMPACT_ARTIFACTS,
        dimension_sha256s=DIMENSIONS,
    )


def _request(storage_mode: str) -> dict:
    exact = storage_mode == "exact_full"
    payload = {
        "schema_version": STORAGE_PARITY_REQUEST_SCHEMA_VERSION,
        "request_id": (
            "storage-exact-full" if exact else "storage-compact-complete"
        ),
        "execution_route": "auxiliary_storage_parity_only",
        "isolation_callback_authorized": False,
        "candidate_authority_sha256": (
            EXACT_AUTHORITY if exact else COMPACT_AUTHORITY
        ),
        "candidate_artifact_sha256": (
            EXACT_CANDIDATE_ARTIFACT if exact else COMPACT_CANDIDATE_ARTIFACT
        ),
        "storage_mode": storage_mode,
        "run_root": f"/runtime/{storage_mode}",
        "owner_root": f"/owners/{storage_mode}",
        "checkpoint_root": f"/checkpoints/{storage_mode}",
        "generations": 1,
        "ticks_per_generation": 3,
        "checkpoint_split_generation": None,
        "reporting_enabled": True,
        "exposure": {
            "A": 8,
            "S": 8 if exact else 1,
            "C": 4,
            "K": 4,
            "eligibility_representation": (
                "role_specific_per_attempt_parent_id_sets_v1"
                if exact
                else "role_specific_aggregate_opportunity_counts_v1"
            ),
            "per_attempt_eligible_parent_id_sets_available": exact,
        },
        "physical_order": [{"occurrence_id": "cell-a", "batch_position": 0}],
    }
    return {**payload, "request_sha256": canonical_sha256(payload)}


def _contract(exact: dict, compact: dict) -> dict:
    payload = {
        "schema_version": STORAGE_PARITY_COMPARISON_SCHEMA_VERSION,
        "exact_full_request_sha256": exact["request_sha256"],
        "compact_complete_request_sha256": compact["request_sha256"],
        "required_comparisons": list(STORAGE_PARITY_FIELDS),
        "exact_full_eligible_parent_id_sets_authoritative": True,
        "compact_per_attempt_eligible_parent_id_sets_forbidden": True,
        "compact_eligibility_comparison": (
            "role_specific_aggregate_opportunity_counts_v1"
        ),
        "promotion_authority": False,
    }
    return {
        **payload,
        "comparison_contract_sha256": canonical_sha256(payload),
    }


def _result(storage_mode: str, result_digit: str):
    exact = storage_mode == "exact_full"
    attempt = {
        "attempt_index": 0,
        "path": 1,
        "generation": 0,
        "tick": 3,
        "eligible_parent_counts": [2, 1],
        "selected_parent_ids": [11, 13],
        "success": True,
        "failure_code": 0,
        "commit_ordinal": 0,
        "authoritative_commit": {
            "child_agent_id": 19,
            "donor_draw_ordinals": [0, 1],
            "commit_authority_sha256": "8" * 64,
        },
    }
    if exact:
        attempt["eligible_snapshot_refs"] = [0, 1]
    payload = {
        "production_authority": False,
        "request_sha256": result_digit * 64,
        "candidate_authority_sha256": (
            EXACT_AUTHORITY if exact else COMPACT_AUTHORITY
        ),
        "plan_sha256": (
            EXACT_CANDIDATE_ARTIFACT
            if exact
            else COMPACT_CANDIDATE_ARTIFACT
        ),
        "cell_ids": ["cell-a"],
        "candidate_artifact_sha256s": (
            EXACT_ARTIFACTS
            if storage_mode == "exact_full"
            else COMPACT_ARTIFACTS
        ),
        "parity_dimension_sha256s": DIMENSIONS,
        "qualification_receipts": {
            "resident_dispatch_observations": [
                {
                    "dispatch_index": 0,
                    "cell_state_sha256": ["3" * 64],
                    "cell_rng_sha256": ["4" * 64],
                }
            ]
        },
        "segments": [
            {
                "start_generation": 0,
                "end_generation": 1,
                "cell_ids": ["cell-a"],
                "mechanics_checkpoint_digests": {"cell-a": "5" * 64},
                "segment_artifacts": [
                    {
                        "cell_id": "cell-a",
                        "capacity_semantics": {"storage_mode": storage_mode},
                        "cell": {
                            "attempts": [attempt],
                            "primary_eligibility_opportunity_counts": [
                                {"agent_id": 11, "count": 1},
                                {"agent_id": 13, "count": 1},
                            ],
                            "mate_eligibility_opportunity_counts": [
                                {"agent_id": 13, "count": 1}
                            ],
                        },
                    }
                ],
            }
        ],
    }
    return {**payload, "result_sha256": canonical_sha256(payload)}


def _reseal_result(result: dict) -> dict:
    result.pop("result_sha256", None)
    result["result_sha256"] = canonical_sha256(result)
    return result


def _execution_receipt(contract: dict, exact: dict, compact: dict) -> dict:
    payload = {
        "schema_version": (
            "eiel_inheritance_cross_path_candidate_storage_parity_comparison_v1"
        ),
        "composed_plan_artifact_sha256": "a" * 64,
        "comparison_contract_sha256": contract["comparison_contract_sha256"],
        "candidate_plan_sha256s": {
            "exact_full": exact["plan_sha256"],
            "compact_complete": compact["plan_sha256"],
        },
        "candidate_request_sha256s": {
            "exact_full": exact["request_sha256"],
            "compact_complete": compact["request_sha256"],
        },
        "candidate_result_sha256s": {
            "exact_full": exact["result_sha256"],
            "compact_complete": compact["result_sha256"],
        },
        "fresh_owner_identity_sha256": "b" * 64,
        "reporting_enabled": True,
        "production_authority": False,
        "comparisons": {
            field: {"ok": True, "evidence_sha256": "c" * 64}
            for field in STORAGE_PARITY_FIELDS
        },
    }
    return {
        **payload,
        "comparison_receipt_sha256": canonical_sha256(payload),
    }


def test_projects_exact_attempt_eligibility_state_rng_and_checkpoint_parity() -> None:
    plan = _plan()
    exact_request = _request("exact_full")
    compact_request = _request("compact_complete")
    contract = _contract(exact_request, compact_request)
    exact = _result("exact_full", "1")
    compact = _result("compact_complete", "2")
    typed_plan, result, evidence = verify_cross_path_storage_parity_evidence_v1(
        preregistered_plan=plan,
        exact_full_request=exact_request,
        compact_complete_request=compact_request,
        comparison_contract=contract,
        execution_comparison_receipt=_execution_receipt(
            contract, exact, compact
        ),
        capture_manifest_sha256="d" * 64,
        exact_full_result=exact,
        compact_complete_result=compact,
    )

    assert result.payload()["schema_version"] == (
        CROSS_STORAGE_PARITY_RESULT_SCHEMA_VERSION
    )
    assert result.plan_sha256 == plan.plan_sha256
    assert result.exact_full.capture_sha256 == exact["result_sha256"]
    assert result.compact_complete.capture_sha256 == compact["result_sha256"]
    assert len(result.result_sha256) == 64
    assert typed_plan is plan
    assert evidence.parity_result_sha256 == result.result_sha256


def test_rejects_rng_or_attempt_drift() -> None:
    exact_request = _request("exact_full")
    compact_request = _request("compact_complete")
    contract = _contract(exact_request, compact_request)
    exact = _result("exact_full", "1")
    compact = _result("compact_complete", "2")
    compact["qualification_receipts"]["resident_dispatch_observations"][0][
        "cell_rng_sha256"
    ][0] = "9" * 64
    _reseal_result(compact)
    with pytest.raises(
        InheritanceCrossPathStorageParityError,
        match="state or RNG",
    ):
        verify_cross_path_storage_parity_v1(
            preregistered_plan=_plan(),
            exact_full_request=exact_request,
            compact_complete_request=compact_request,
            comparison_contract=contract,
            execution_comparison_receipt=_execution_receipt(
                contract, exact, compact
            ),
            capture_manifest_sha256="d" * 64,
            exact_full_result=exact,
            compact_complete_result=compact,
        )

    compact = _result("compact_complete", "2")
    compact["segments"][0]["segment_artifacts"][0]["cell"]["attempts"][0][
        "authoritative_commit"
    ]["child_agent_id"] = 20
    _reseal_result(compact)
    with pytest.raises(
        InheritanceCrossPathStorageParityError,
        match="attempts or eligibility",
    ):
        verify_cross_path_storage_parity_v1(
            preregistered_plan=_plan(),
            exact_full_request=exact_request,
            compact_complete_request=compact_request,
            comparison_contract=contract,
            execution_comparison_receipt=_execution_receipt(
                contract, exact, compact
            ),
            capture_manifest_sha256="d" * 64,
            exact_full_result=deepcopy(exact),
            compact_complete_result=compact,
        )


def test_rejects_contract_drift_and_compact_per_attempt_eligibility() -> None:
    exact_request = _request("exact_full")
    compact_request = _request("compact_complete")
    contract = _contract(exact_request, compact_request)
    exact = _result("exact_full", "1")
    compact = _result("compact_complete", "2")
    contract["promotion_authority"] = True
    with pytest.raises(
        InheritanceCrossPathStorageParityError,
        match="comparison contract",
    ):
        verify_cross_path_storage_parity_v1(
            preregistered_plan=_plan(),
            exact_full_request=exact_request,
            compact_complete_request=compact_request,
            comparison_contract=contract,
            execution_comparison_receipt=_execution_receipt(
                contract, exact, compact
            ),
            capture_manifest_sha256="d" * 64,
            exact_full_result=exact,
            compact_complete_result=compact,
        )

    compact = _result("compact_complete", "2")
    compact["segments"][0]["segment_artifacts"][0]["cell"]["attempts"][0][
        "eligible_snapshot_refs"
    ] = [-1, -1]
    _reseal_result(compact)
    contract = _contract(exact_request, compact_request)
    with pytest.raises(
        InheritanceCrossPathStorageParityError,
        match="forbidden per-attempt eligibility",
    ):
        verify_cross_path_storage_parity_v1(
            preregistered_plan=_plan(),
            exact_full_request=exact_request,
            compact_complete_request=compact_request,
            comparison_contract=contract,
            execution_comparison_receipt=_execution_receipt(
                contract, exact, compact
            ),
            capture_manifest_sha256="d" * 64,
            exact_full_result=exact,
            compact_complete_result=compact,
        )


def test_rejects_stale_candidate_result_or_execution_receipt() -> None:
    exact_request = _request("exact_full")
    compact_request = _request("compact_complete")
    contract = _contract(exact_request, compact_request)
    exact = _result("exact_full", "1")
    compact = _result("compact_complete", "2")
    exact["plan_sha256"] = "f" * 64
    with pytest.raises(
        InheritanceCrossPathStorageParityError,
        match="result digest is stale",
    ):
        verify_cross_path_storage_parity_v1(
            preregistered_plan=_plan(),
            exact_full_request=exact_request,
            compact_complete_request=compact_request,
            comparison_contract=contract,
            execution_comparison_receipt=_execution_receipt(
                contract, exact, compact
            ),
            capture_manifest_sha256="d" * 64,
            exact_full_result=exact,
            compact_complete_result=compact,
        )

    exact = _result("exact_full", "1")
    receipt = _execution_receipt(contract, exact, compact)
    receipt["candidate_plan_sha256s"]["exact_full"] = "f" * 64
    with pytest.raises(
        InheritanceCrossPathStorageParityError,
        match="execution comparison receipt is invalid",
    ):
        verify_cross_path_storage_parity_v1(
            preregistered_plan=_plan(),
            exact_full_request=exact_request,
            compact_complete_request=compact_request,
            comparison_contract=contract,
            execution_comparison_receipt=receipt,
            capture_manifest_sha256="d" * 64,
            exact_full_result=exact,
            compact_complete_result=compact,
        )
