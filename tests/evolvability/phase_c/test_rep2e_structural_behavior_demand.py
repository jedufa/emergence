from __future__ import annotations

from agents.genome import Genome
from agents.structural_payload_activation import (
    STRUCTURAL_PAYLOAD_SEED_ABLATION_REMOVE_CONNECTIONS,
    STRUCTURAL_PAYLOAD_SEED_ABLATION_ZERO_CONNECTION_GATES,
    ablate_seed_payload,
    minimal_canonical_seed_payload,
)
from evolution.structural_behavior_demand import (
    ablation_design,
    action_choice_causal_map,
    action_choice_code_map,
    field_consumption_map,
    freeze_behavior_demand_fixture_plan,
    graph_arms,
    margin_diagnosis,
    payload_remove_connections,
    reporting_summary,
    run_chosen_action_fixtures,
    run_sequence_fixtures,
    selection_readiness_skipped,
)
from evolution.structural_mutation_viability import base_fixture_genome


def test_rep2e_fixture_plan_is_frozen_and_has_required_families() -> None:
    plan = freeze_behavior_demand_fixture_plan()
    families = {row["behavior_family"] for row in plan["fixtures"]}
    assert plan["frozen_before_outcomes"] is True
    assert "aggression_attack_opportunity" in families
    assert "drive_choice_competition" in families
    assert "temperature_near_tie_choice" in families
    assert "unavailable_target_control" in families


def test_rep2e_action_choice_maps_cover_live_fields() -> None:
    code = action_choice_code_map()
    causal = action_choice_causal_map()
    fields = field_consumption_map()
    assert "runner_policy.py" in " ".join(code["source_files"])
    assert "top-two margin" in causal["map"]
    assert "aggression_bias" in fields["consumed_by_one_step_logits"]
    assert "drive_inertia" in fields["consumed_by_short_sequence_drive_state"]


def test_rep2e_matched_ablation_payloads() -> None:
    genome = base_fixture_genome()
    payload = minimal_canonical_seed_payload(genome)
    zero = ablate_seed_payload(payload, STRUCTURAL_PAYLOAD_SEED_ABLATION_ZERO_CONNECTION_GATES)
    removed = ablate_seed_payload(payload, STRUCTURAL_PAYLOAD_SEED_ABLATION_REMOVE_CONNECTIONS)
    assert len(zero["connections"]) == len(payload["connections"])
    assert any(conn["weight"] == 0.0 for conn in zero["connections"] if conn.get("value_type") == "gate")
    assert removed["connections"] == []
    assert Genome.from_dict(payload_remove_connections()).connections == []


def test_rep2e_chosen_action_reach_and_margin_diagnosis() -> None:
    rows, summary, margin = run_chosen_action_fixtures()
    assert rows
    assert summary["canonical_chosen_action_reach"] is True
    assert summary["chosen_action_change_count"] >= 1
    assert margin["near_boundary_count"] >= 1
    assert margin_diagnosis(rows)["no_flip_count"] >= 1


def test_rep2e_sequence_fixture_reports_divergence_status() -> None:
    traces, summary = run_sequence_fixtures()
    assert traces
    assert summary["classification"] in {
        "canonical_divergence",
        "mutation_generated_divergence",
        "diagnostic_only_divergence",
        "no_sequence_divergence",
    }
    assert summary["sequence_count"] >= 4


def test_rep2e_ablation_design_records_partial_zero_gate() -> None:
    design = ablation_design()
    assert design["default_off_config"] == "structural_payload_seed_ablation"
    assert design["primary_fixture_ablation"] == "remove_connections"
    assert "zero_gate_note" in design


def test_rep2e_selection_readiness_skipped_uses_null_support() -> None:
    summary, rows, lineage, cost = selection_readiness_skipped("gate_not_passed")
    assert summary["selection_readiness_supported"] is None
    assert rows == []
    assert lineage["lineage_confounded"] is None
    assert cost["cost_sanity_passed"] is True


def test_rep2e_reporting_summary_uses_null_when_not_evaluated() -> None:
    selection, _rows, _lineage, _cost = selection_readiness_skipped("gate_not_passed")
    payload = reporting_summary(
        {"row_count": 1, "score_change_count": 1, "probability_change_count": 1, "chosen_action_change_count": 1, "executed_action_change_count": 1},
        {"canonical_sequence_divergence": False, "mutation_generated_sequence_divergence": False, "diagnostic_sequence_divergence": False},
        {"canonical_behavior_reach": False, "mutation_generated_behavior_reach": False, "diagnostic_behavior_reach": None},
        {"classification": "no_blocker_behavior_reach_present"},
        {"classification": "behavior_opportunity_weak", "opportunity_counts": {}},
        selection,
        {"reward_audit_passed": True},
        {"checkpoint_status": "passed"},
        {"w6_status": "passed"},
    )
    assert payload["selection_readiness_supported"] is None
    assert payload["diagnostic_behavior_reach"] is None
