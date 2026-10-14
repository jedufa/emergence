from __future__ import annotations

import math

import pytest

from evolution.structural_causal_robustness import (
    REP2B_GRAPH_DIGEST,
    action_trace_outcomes,
    base_reproduction_summary,
    cost_sanity,
    decision_summary,
    developmental_robustness,
    evaluate_fixture,
    field_coverage,
    freeze_robustness_battery_plan,
    inheritance_robustness,
    integrity_summary,
    live_robustness_summary,
    operator_supply,
    recombination_robustness,
    repeat_parity_summary,
    reward_audit,
    saturation_summary,
    selection_readiness_skipped,
    unsupported_graph_summary,
)


PLAN = freeze_robustness_battery_plan()
PLAN_BY_ID = {row["fixture_id"]: row for row in PLAN}


def test_rep2d_base_reproduces_rep2b() -> None:
    summary = base_reproduction_summary()
    assert summary["graph_digest_match"] is True
    assert summary["field_delta_match"] is True
    assert summary["repeat_parity_match"] is True
    assert summary["graph_digest"] == REP2B_GRAPH_DIGEST


def test_rep2d_disabled_compatibility_fixture() -> None:
    row = evaluate_fixture(PLAN_BY_ID["seed_live_off"])
    assert row["unsupported_reason"] == "disabled"
    assert row["changed_fields"] == []


@pytest.mark.parametrize("fixture_id", [row["fixture_id"] for row in PLAN])
def test_rep2d_frozen_fixture_evaluates_without_nonfinite(fixture_id: str) -> None:
    row = evaluate_fixture(PLAN_BY_ID[fixture_id])
    assert row["repeat_match"] is True
    assert row["random_draw_delta"] == 0
    for value in row["effective_fields_after"].values():
        assert math.isfinite(float(value))


def test_rep2d_zero_weight_and_zero_gate() -> None:
    zero_weight = evaluate_fixture(PLAN_BY_ID["zero_weight"])
    zero_gate = evaluate_fixture(PLAN_BY_ID["zero_gate"])
    assert zero_weight["changed_fields"] == []
    assert "drive_logit_gain" not in zero_gate["changed_fields"]
    assert "drive_inertia" not in zero_gate["changed_fields"]


def test_rep2d_edge_removal_and_bounds() -> None:
    seed = evaluate_fixture(PLAN_BY_ID["seed_live_on"])
    removed = evaluate_fixture(PLAN_BY_ID["removed_first_edge"])
    assert "aggression_bias" in seed["changed_fields"]
    assert "aggression_bias" not in removed["changed_fields"]
    for row in (seed, removed):
        for value in row["effective_fields_after"].values():
            assert math.isfinite(float(value))


def test_rep2d_positive_monotonic_and_negative_sign_cases() -> None:
    rows = [evaluate_fixture(PLAN_BY_ID[fid]) for fid in ("edge0_weight_0p0", "edge0_weight_0p25", "edge0_weight_1p0", "edge0_weight_2p0", "edge0_weight_neg1p0")]
    deltas = [float(row["delta_per_field"]["aggression_bias"]) for row in rows[:4]]
    assert deltas == sorted(deltas)
    assert float(rows[-1]["delta_per_field"]["aggression_bias"]) < 0.0


def test_rep2d_aggregation_order_and_digest_stability() -> None:
    seed = evaluate_fixture(PLAN_BY_ID["seed_live_on"])
    shuffled = evaluate_fixture(PLAN_BY_ID["order_shuffle"])
    duplicate = evaluate_fixture(PLAN_BY_ID["duplicate_edge_aggregation"])
    assert seed["graph_digest"] == shuffled["graph_digest"]
    assert duplicate["graph_supported"] is True


@pytest.mark.parametrize(
    "fixture_id",
    [
        "unsupported_cycle",
        "unsupported_missing_source",
        "unsupported_wrong_type",
        "unsupported_duplicate_module_id",
        "unsupported_nonfinite_weight",
        "unsupported_cap_overflow",
        "unsupported_slot_route",
    ],
)
def test_rep2d_unsupported_controls_fail_closed(fixture_id: str) -> None:
    row = evaluate_fixture(PLAN_BY_ID[fixture_id])
    assert row["graph_supported"] is False
    assert row["unsupported_reason"]


def test_rep2d_robustness_summaries() -> None:
    rows = [evaluate_fixture(fixture) for fixture in PLAN]
    robustness = live_robustness_summary(rows)
    unsupported = unsupported_graph_summary(rows)
    saturation = saturation_summary(rows)
    assert robustness["bounded"] is True
    assert robustness["monotonicity"]["zero_connection_contribution_ok"] is True
    assert robustness["sign_behavior"]["hidden_absolute_value_behavior"] is False
    assert unsupported["unsupported_count"] >= 1
    assert saturation["classification"] == "no_saturation"


def test_rep2d_operator_supply_and_field_coverage() -> None:
    rows = [evaluate_fixture(fixture) for fixture in PLAN]
    op_rows, op_summary = operator_supply(seed_count=2)
    coverage = field_coverage(rows, op_rows)
    assert op_summary["attempts"] == 28
    assert op_summary["valid_live_rate"] > 0.0
    assert "aggression_bias" in coverage["action_reachable"]


def test_rep2d_inheritance_recombination_developmental() -> None:
    inheritance = inheritance_robustness()
    recombination = recombination_robustness()
    developmental = developmental_robustness()
    assert inheritance["inherited_live_variant_count"] >= 1
    assert recombination["recombined_live_variant_count"] >= 1
    assert developmental["records"]


def test_rep2d_action_reach_summary_and_gap() -> None:
    rows, summary, gap = action_trace_outcomes()
    assert rows
    assert summary["action_reach_present"] is True
    assert summary["action_score_change_count"] > 0
    assert gap["action_reach_blocked"] is False


def test_rep2d_selection_readiness_null_when_skipped() -> None:
    summary, rows, lineage = selection_readiness_skipped("behavior_reach_weak")
    assert summary["selection_readiness_supported"] is None
    assert rows == []
    assert lineage["lineage_confounded"] is None


def test_rep2d_reward_audit_and_cost_sanity() -> None:
    _op_rows, op_summary = operator_supply(seed_count=1)
    reward = reward_audit()
    cost = cost_sanity(op_summary)
    assert reward["reward_audit_passed"] is True
    assert cost["cost_sanity_passed"] is True


def test_rep2d_integrity_decision_repeat_parity() -> None:
    rows = [evaluate_fixture(fixture) for fixture in PLAN[:8]]
    action_rows, action_summary, _gap = action_trace_outcomes()
    behavior = {"behavior_reach_present": False, "cell_count": 0}
    reward = reward_audit()
    checkpoint = {"checkpoint_status": "passed"}
    w6 = {"w6_status": "passed"}
    integrity = integrity_summary(rows, action_summary, behavior, reward, checkpoint, w6)
    decision = decision_summary(live_robustness_summary(rows), {"valid_live_rate": 1.0}, action_summary, behavior, checkpoint, w6)
    repeat = repeat_parity_summary(rows, action_rows, [])
    assert integrity["integrity_failure"] is False
    assert decision["decision"] == "REP-2D-B"
    assert repeat["action_trace"] is True
