from __future__ import annotations

from tools.evidence.reward_audit import audit_reward_leakage


FIELDS = ("direct_cognition_reward_present_flag", "direct_prediction_error_selection_pressure_present_flag")


def test_reward_audit_safe_case_passes() -> None:
    audit = audit_reward_leakage([{"run_id": "r1", "final_metrics": {field: "0" for field in FIELDS}}], FIELDS)
    assert audit["ok"] is True
    assert audit["severity"] == "ok"


def test_reward_audit_direct_cognition_reward_is_fatal() -> None:
    audit = audit_reward_leakage([{"run_id": "r1", "final_metrics": {"direct_cognition_reward_present_flag": "1"}}], FIELDS)
    assert audit["severity"] == "fatal"
    assert audit["active_fields"][0]["field"] == "direct_cognition_reward_present_flag"


def test_reward_audit_ambiguous_missing_fields_downgrades() -> None:
    audit = audit_reward_leakage([{"run_id": "r1", "final_metrics": {"direct_cognition_reward_present_flag": "0"}}], FIELDS)
    assert audit["severity"] == "ambiguous"
    assert audit["missing_fields"] == ["direct_prediction_error_selection_pressure_present_flag"]


def test_reward_audit_unexpected_reward_fields_are_named() -> None:
    audit = audit_reward_leakage(
        [{"run_id": "r1", "final_metrics": {"direct_perception_reward_present": "1", "diversity_reward_present": "0"}}],
        (),
    )
    assert audit["severity"] == "fatal"
    assert "direct_perception_reward_present" in {row["field"] for row in audit["active_fields"]}
    assert audit["ambiguous_fields"] == ["direct_perception_reward_present", "diversity_reward_present"]


def test_reward_audit_treats_true_disabled_flag_as_inactive_control() -> None:
    audit = audit_reward_leakage(
        [{"run_id": "r1", "final_metrics": {"prediction_direct_selection_pressure_disabled_flag": "1"}}],
        (),
    )

    assert audit["severity"] == "ok"
    assert audit["active_fields"] == []
    assert audit["ambiguous_fields"] == []
    assert audit["inactive_control_fields"] == ["prediction_direct_selection_pressure_disabled_flag"]


def test_reward_audit_treats_false_disabled_flag_as_fatal() -> None:
    audit = audit_reward_leakage(
        [{"run_id": "r1", "final_metrics": {"prediction_direct_selection_pressure_disabled_flag": "0"}}],
        (),
    )

    assert audit["severity"] == "fatal"
    assert audit["active_fields"][0]["field"] == "prediction_direct_selection_pressure_disabled_flag"
