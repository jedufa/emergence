from __future__ import annotations

from tools.bin import mosaic_ea0_policy_eligibility_audit as ea0


def test_mosaic_ea0_required_outputs_include_gate1_and_closeout_artifacts() -> None:
    assert "mosaic_ea0_decision_contract_audit.json" in ea0.REQUIRED_OUTPUTS
    assert "mosaic_ea0_final_decision.json" in ea0.REQUIRED_OUTPUTS
    assert "mosaic_ea0_cleanup_record.json" in ea0.REQUIRED_OUTPUTS


def test_mosaic_ea0_not_evaluated_payload_is_explicit() -> None:
    payload = ea0.not_evaluated("mosaic_ea0_policy_reachability_summary", "gate failed")

    assert payload["status"] == "NOT_EVALUATED"
    assert payload["reason"] == "gate failed"
