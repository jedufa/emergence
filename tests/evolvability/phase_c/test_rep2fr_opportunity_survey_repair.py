from __future__ import annotations

from pathlib import Path

from evolution.structural_action_opportunity_repair import (
    CANONICAL_COUNTERFACTUAL_FIELD_DELTAS,
    corrected_selected_contexts,
    decision_summary,
    make_rep2fr_sim,
    metric_contract_v2,
    original_survey_plan,
    rep2f_survey_validity_review,
    repeat_parity_summary,
)
from simulation.runner_pkg.runner_action_opportunity import empty_opportunity_summary


def test_rep2fr_metric_contract_separates_actual_and_potential_linked_fields() -> None:
    contract = metric_contract_v2()
    assert contract["schema_version"] == "structural_action_opportunity_v2"
    assert contract["v1_compatibility"]
    assert contract["potential_field_linked_opportunity"]["does_not_require_live_field_change"] is True
    assert contract["counterfactual_canonical_effect"]["adds_random_draws"] == 0


def test_rep2fr_prior_gate_validity_review_marks_survey_invalid() -> None:
    review = rep2f_survey_validity_review(
        {"classification": "no_existing_scenario_exposure"},
        [{"passes_opportunity_gate": False}],
    )
    assert review["corrected_status"] == "SURVEY_GATE_INVALID"
    assert review["conclusion_supported"] is False
    assert review["linked_field_opportunity_forced_zero"] is True
    assert review["near_boundary_forced_zero"] is True


def test_rep2fr_live_off_potential_opportunity_nonzero_actual_zero(tmp_path: Path) -> None:
    sim = make_rep2fr_sim(
        tmp_path / "sim",
        seed=11,
        context="crowding_existing",
        live_enabled=False,
        generations=1,
        steps=2,
        pop=4,
        metrics=False,
    )
    sim.run(start_gen=0, skip_init=False)
    total = sim._structural_action_opportunity_total
    assert total["schema_version"] == "structural_action_opportunity_v2"
    assert total["evaluated_decisions"] > 0
    assert total["potential_linked_opportunity_count"] > 0
    assert total["linked_field_active_count"] == 0
    assert total["linked_field_opportunity_count"] == 0


def test_rep2fr_margin_telemetry_measured_not_invented(tmp_path: Path) -> None:
    sim = make_rep2fr_sim(
        tmp_path / "sim",
        seed=12,
        context="clean_control",
        live_enabled=False,
        generations=1,
        steps=2,
        pop=4,
        metrics=False,
    )
    sim.run(start_gen=0, skip_init=False)
    total = sim._structural_action_opportunity_total
    assert total["margin_evaluable_count"] == total["evaluated_decisions"]
    assert total["margin_not_evaluable_count"] == 0
    assert total["margin_mean"] is not None
    assert sum(total["margin_histogram"].values()) == total["margin_evaluable_count"]


def test_rep2fr_disabled_summary_keeps_null_discipline() -> None:
    summary = empty_opportunity_summary(enabled=False)
    assert summary["enabled"] is False
    assert summary["evaluated_decisions"] is None
    assert summary["margin_evaluable_count"] is None
    assert summary["not_evaluable_reason_counts"] == {"reporting_disabled": 1}


def test_rep2fr_corrected_survey_plan_is_outcome_blind() -> None:
    plan = original_survey_plan()
    assert plan["frozen_before_structural_outcomes"] is True
    assert plan["live_semantics"] == "off_for_action_execution"
    forbidden = set(plan["forbidden_ranking_fields"])
    assert {"live_on_behavior", "survival", "births", "lineage"}.issubset(forbidden)


def test_rep2fr_corrected_gate_can_pass_live_off_scenario() -> None:
    selected = corrected_selected_contexts(
        [
            {
                "scenario_name": "candidate",
                "passes_corrected_opportunity_gate": True,
                "corrected_opportunity_score": 100,
            }
        ]
    )
    assert selected["outcome_blind_confirmation"] is True
    assert selected["selected_contexts"][0]["scenario_name"] == "candidate"


def test_rep2fr_counterfactual_field_delta_fixed() -> None:
    assert CANONICAL_COUNTERFACTUAL_FIELD_DELTAS["aggression_bias"] > 0.0
    assert CANONICAL_COUNTERFACTUAL_FIELD_DELTAS["drive_logit_gain"] > 0.0


def test_rep2fr_repeat_parity_runs_cells_sequentially(tmp_path: Path) -> None:
    summary = repeat_parity_summary(tmp_path, include_readiness=True)
    assert summary["representative_cell_match"] is True
    assert summary["readiness_cell_match"] is True
    assert summary["representative_hash_a"] == summary["representative_hash_b"]
    assert summary["readiness_hash_a"] == summary["readiness_hash_b"]
    assert summary["passed"] is True


def test_rep2fr_decision_fails_closed_on_repeat_parity_failure() -> None:
    decision = decision_summary(
        {"decision_qualifier": "SELECTION_READINESS_NOT_SUPPORTED"},
        {"checkpoint_status": "passed"},
        {"w6_status": "limited"},
        {"passed": False},
    )
    assert decision["decision"] == "REP-2FR-D"
    assert decision["qualifier"] == "INTEGRITY_FAILURE"
