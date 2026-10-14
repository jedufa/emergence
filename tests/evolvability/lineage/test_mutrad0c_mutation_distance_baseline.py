from __future__ import annotations

import pytest

from tools.bin.mutrad0c_mutation_distance_baseline import classify_metrics, decide, summarize_cell_events


def test_mutrad0c_summarizes_birth_distance_components() -> None:
    events = [
        {
            "event_type": "birth_success",
            "child_lineage_id": "L1",
            "parent_child_distance": 2,
            "parent_child_genome_distance_mean": 0.10,
            "parent_child_genome_distance_max": 0.15,
            "parent_child_genome_scalar_distance_mean": 0.10,
            "parent_child_genome_memory_field_delta": 0.0,
            "parent_child_genome_prediction_field_delta": 0.05,
            "parent_child_genome_module_gate_delta": 0.0,
        },
        {
            "event_type": "birth_success",
            "child_lineage_id": "L2",
            "parent_child_distance": 4,
            "parent_child_genome_distance_mean": 0.20,
            "parent_child_genome_distance_max": 0.25,
            "parent_child_genome_scalar_distance_mean": 0.20,
            "parent_child_genome_memory_field_delta": 0.10,
            "parent_child_genome_prediction_field_delta": 0.0,
            "parent_child_genome_module_gate_delta": 0.0,
        },
        {"event_type": "birth_fail_no_space"},
    ]

    summary = summarize_cell_events(events)

    assert summary["birth_success_count"] == 2
    assert summary["failed_birth_count_if_available"] == 1
    assert summary["offspring_effective_lineages"] == 2.0
    assert summary["parent_child_genome_distance_mean_event_mean"] == pytest.approx(0.15)
    assert summary["memory_dim_delta_event_count"] == 1
    assert summary["pred_lr_delta_event_count"] == 1
    assert summary["module_gate_delta_event_count"] == 0


def test_mutrad0c_decision_a_for_complete_narrow_local_reporting() -> None:
    cell = {
        "context_id": "flat_control",
        "seed": 101,
        "birth_success_count": 3,
        "parent_child_genome_distance_mean_present_birth_rows": 3,
        "parent_child_genome_distance_max_present_birth_rows": 3,
        "parent_child_genome_scalar_distance_mean_present_birth_rows": 3,
        "parent_child_genome_memory_field_delta_present_birth_rows": 3,
        "parent_child_genome_prediction_field_delta_present_birth_rows": 3,
        "parent_child_genome_module_gate_delta_present_birth_rows": 3,
        "parent_child_genome_distance_mean_event_mean": 0.05,
        "parent_child_genome_distance_max_event_max": 0.10,
        "module_gate_delta_event_count": 0,
        "memory_dim_delta_event_count": 0,
        "pred_lr_delta_event_count": 1,
        "final_module_count_max": 0,
        "final_connection_count_max": 0,
        "final_c1_operator_applied_nonempty_count": 0,
    }

    classification = classify_metrics([cell], [{"context_id": "flat_control"}])
    decision = decide(classification)

    assert classification["missing_distance_field_cell_count"] == 0
    assert classification["mutation_appears_narrow_local_by_reporting_proxy"] is True
    assert decision["decision"] == "MUTRAD-0C-BASE-A"


def test_mutrad0c_decision_b_for_missing_distance_fields() -> None:
    cell = {
        "context_id": "flat_control",
        "seed": 101,
        "birth_success_count": 2,
        "parent_child_genome_distance_mean_present_birth_rows": 1,
        "parent_child_genome_distance_max_present_birth_rows": 2,
        "parent_child_genome_scalar_distance_mean_present_birth_rows": 2,
        "parent_child_genome_memory_field_delta_present_birth_rows": 2,
        "parent_child_genome_prediction_field_delta_present_birth_rows": 2,
        "parent_child_genome_module_gate_delta_present_birth_rows": 2,
    }

    classification = classify_metrics([cell], [{"context_id": "flat_control"}])
    decision = decide(classification)

    assert classification["missing_distance_field_cell_count"] == 1
    assert decision["decision"] == "MUTRAD-0C-BASE-B"
