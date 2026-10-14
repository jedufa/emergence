from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.scripts.eiel_option_d_m28_env_policy_receipt_contract import (
    DECISION_WITHHELD,
    REQUIRED_OPERATORS,
    decide,
)


_DIGEST = "0" * 64


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _m27_receipt(path: Path) -> Path:
    return _write_json(
        path,
        {
            "decision": "M27-A-ENV-POLICY-PCG64-SURFACES-IDENTIFIED-PROMOTION-WITHHELD",
            "discovered_operations": [
                {
                    "operator_id": "environment.food_spawn.uniform_or_bounded",
                    "physical_stream_id": "numpy.environment_policy",
                },
                {
                    "operator_id": "policy.action_gumbel.float32",
                    "physical_stream_id": "numpy.environment_policy",
                },
            ],
        },
    )


def _common(operator_id: str) -> dict[str, Any]:
    return {
        "operator_id": operator_id,
        "physical_stream_id": "numpy.environment_policy",
        "operation_receipt_schema_version": "synthetic_env_policy_receipt_v1",
        "status_ok": True,
        "promotion_qualified": True,
        "promotion_blockers": [],
        "raw_pcg64_delta_available": True,
        "semantic_draws": 4,
        "raw64_draws": 2,
        "raw32_draws": 0,
        "state_before_digest": _DIGEST,
        "state_after_digest": _DIGEST,
        "buffer_before_digest": _DIGEST,
        "buffer_after_digest": _DIGEST,
        "has_uint32_before": False,
        "has_uint32_after": True,
        "uinteger_before": 0,
        "uinteger_after": 123,
        "checkpoint_continuation_covered": True,
        "failure_atomicity_covered": True,
        "authority_class": "measurement_only_not_route_promotion",
    }


def _food_receipt() -> dict[str, Any]:
    row = _common("environment.food_spawn.uniform_or_bounded")
    row.update(
        {
            "food_spawn_attempts": 4,
            "food_spawn_successes": 3,
            "bounded_rejection_attempts": 1,
            "bounded_rejection_checkpoint_continuation_covered": True,
        }
    )
    return row


def _policy_receipt() -> dict[str, Any]:
    row = _common("policy.action_gumbel.float32")
    row.update(
        {
            "policy_rows_requested": 4,
            "policy_cache_hit_rows": 1,
            "policy_cache_miss_rows": 3,
            "policy_rows_generated": 3,
            "gumbel_float32_draws": 4,
            "post_action_state_buffer_bound": True,
        }
    )
    return row


def test_m28_defines_contract_and_withholds_when_receipts_absent(
    tmp_path: Path,
) -> None:
    m27 = _m27_receipt(tmp_path / "m27.json")

    result = decide(m27_receipt_path=m27, operation_receipt_paths=[])

    assert result["ok"] is True
    assert result["decision"] == DECISION_WITHHELD
    assert result["contract_defined"] is True
    assert result["operation_receipts_qualified"] is False
    assert result["missing_operation_receipts"] == list(REQUIRED_OPERATORS)
    assert result["route_promotion_authorized"] is False
    assert "env_policy_operation_receipts_missing" in result["promotion_blockers"]
    assert len(result["receipt_sha256"]) == 64


def test_m28_accepts_complete_operation_receipts_but_keeps_route_withheld(
    tmp_path: Path,
) -> None:
    m27 = _m27_receipt(tmp_path / "m27.json")
    food = _write_json(tmp_path / "food.json", _food_receipt())
    policy = _write_json(tmp_path / "policy.json", _policy_receipt())

    result = decide(
        m27_receipt_path=m27,
        operation_receipt_paths=[food, policy],
    )

    assert result["ok"] is True
    assert result["operation_receipts_qualified"] is True
    assert result["qualified_operations"] == sorted(REQUIRED_OPERATORS)
    assert result["route_promotion_authorized"] is False
    assert result["promotion_blockers"] == [
        "route_level_checkpoint_projection_migration_not_bound",
        "route_level_removal_and_performance_evidence_absent",
    ]


def test_m28_fails_closed_for_malformed_policy_receipt(tmp_path: Path) -> None:
    m27 = _m27_receipt(tmp_path / "m27.json")
    policy_payload = _policy_receipt()
    policy_payload["promotion_blockers"] = ["raw_counter_missing"]
    policy_payload["raw_pcg64_delta_available"] = False
    policy_payload["policy_cache_hit_rows"] = 4
    policy_payload["policy_cache_miss_rows"] = 4
    policy = _write_json(tmp_path / "policy.json", policy_payload)
    food = _write_json(tmp_path / "food.json", _food_receipt())

    result = decide(
        m27_receipt_path=m27,
        operation_receipt_paths=[food, policy],
    )

    assert result["ok"] is True
    assert result["operation_receipts_qualified"] is False
    policy_validation = [
        row
        for row in result["operation_validations"]
        if row["operator_id"] == "policy.action_gumbel.float32"
    ][0]
    assert "promotion_blockers_not_empty" in policy_validation["failures"]
    assert "raw_pcg64_delta_not_available" in policy_validation["failures"]
    assert "policy_cache_rows_must_sum_to_requested_rows" in policy_validation[
        "failures"
    ]
    assert (
        "operation_receipt_unqualified:policy.action_gumbel.float32"
        in result["promotion_blockers"]
    )


def test_m28_rejects_wrong_m27_operation_inventory(tmp_path: Path) -> None:
    m27 = _write_json(
        tmp_path / "m27.json",
        {
            "decision": "M27-A-ENV-POLICY-PCG64-SURFACES-IDENTIFIED-PROMOTION-WITHHELD",
            "discovered_operations": [
                {"operator_id": "policy.action_gumbel.float32"}
            ],
        },
    )

    result = decide(m27_receipt_path=m27, operation_receipt_paths=[])

    assert result["ok"] is False
    assert "m27_required_operations_not_exactly_bound" in result[
        "promotion_blockers"
    ]
