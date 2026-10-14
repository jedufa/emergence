from __future__ import annotations

from simulation.runner_pkg.runner_action_opportunity import empty_opportunity_summary
from evolution.structural_action_opportunity_gate import (
    checkpoint_resume_summary,
    decision_summary,
    make_rep2f_sim,
    opportunity_code_map,
    opportunity_metric_contract,
    opportunity_survey_plan,
    selected_contexts,
)


def test_rep2f_default_off_summary_uses_nulls() -> None:
    summary = empty_opportunity_summary(enabled=False)
    assert summary["enabled"] is False
    assert summary["evaluated_decisions"] is None
    assert summary["linked_field_opportunity_count"] is None
    assert summary["not_evaluable_reason_counts"] == {"reporting_disabled": 1}


def test_rep2f_metric_contract_distinguishes_measured_zero_and_null() -> None:
    contract = opportunity_metric_contract()
    assert "zero means measured zero" in contract["measured_zero_vs_null"]
    assert "adjacent live target" in contract["attack_target_available_count"]
    assert "food at current cell" in contract["gather_resource_available_count"]
    assert "policy_margin_not_captured" in contract["near_boundary_count"]


def test_rep2f_code_map_is_reporting_only_and_deterministic() -> None:
    code = opportunity_code_map()
    assert code["telemetry_status"] == "reporting_only_default_off_repair"
    assert code["random_draws_added"] == 0
    assert code["state_mutation_added"] is False


def test_rep2f_reporting_enabled_accumulates_decisions(tmp_path) -> None:
    sim = make_rep2f_sim(
        tmp_path / "sim",
        seed=123,
        context="clean_control",
        live_enabled=False,
        generations=1,
        steps=2,
        pop=4,
        metrics=False,
    )
    sim.run(start_gen=0, skip_init=False)
    total = getattr(sim, "_structural_action_opportunity_total", {})
    assert total["enabled"] is True
    assert total["evaluated_decisions"] > 0
    assert total["not_evaluable_reason_counts"]["policy_margin_not_captured"] == total["evaluated_decisions"]


def test_rep2f_survey_plan_is_outcome_blind() -> None:
    plan = opportunity_survey_plan()
    forbidden = set(plan["forbidden_ranking_fields"])
    assert plan["frozen_before_structural_outcomes"] is True
    assert plan["live_semantics"] == "off"
    assert {"survival", "births", "structural_effect"}.issubset(forbidden)
    assert "linked_field_opportunity_count" in plan["ranking_components"]


def test_rep2f_selected_contexts_excludes_nonpassing_rows() -> None:
    selected = selected_contexts(
        [
            {"scenario_name": "qualifies", "passes_opportunity_gate": True, "opportunity_score": 10},
            {"scenario_name": "blocked", "passes_opportunity_gate": False, "opportunity_score": 20},
        ]
    )
    assert selected["outcome_blind_confirmation"] is True
    assert [row["scenario_name"] for row in selected["selected_contexts"]] == ["qualifies"]
    assert [row["scenario_name"] for row in selected["excluded_scenarios"]] == ["blocked"]


def test_rep2f_decision_no_existing_exposure_routes_to_b() -> None:
    decision = decision_summary(
        {"decision_qualifier": "NO_EXISTING_SCENARIO_EXPOSURE"},
        {"checkpoint_status": "passed"},
        {"w6_status": "passed"},
    )
    assert decision["decision"] == "REP-2F-B"
    assert decision["qualifier"] == "NO_EXISTING_SCENARIO_EXPOSURE"


def test_rep2f_checkpoint_preserves_reporting_counters(tmp_path) -> None:
    checkpoint = checkpoint_resume_summary(tmp_path)
    assert checkpoint["checkpoint_status"] == "passed"
    assert checkpoint["opportunity_counters_match"] is True
