from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "tools/bin/mem_memory_demand_interpretation_gate.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("mem_memory_demand_interpretation_gate", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_rows() -> list[dict]:
    rows: list[dict] = []
    for demand, selected in (
        ("flat_control_clean", 0.10),
        ("cue_observation_off_control", 0.12),
        ("hazard_off_control", 0.11),
        ("memmax_short_gap", 0.20),
        ("memmax_too_long_gap", 0.15),
    ):
        rows.append(
            {
                "memory_arm": "memory_normal",
                "demand_cell": demand,
                "seed": 101,
                "selected_minus_population_final": selected,
                "descendant_minus_initial_final": 0.10,
                "effective_memory_final": 4.0,
                "memory_cost_mean_final": 0.01,
                "lineage_top1_alive_share_final": 0.42,
                "initial_distribution_audited": True,
                "initial_distribution_matched": True,
                "behavior_coupling_metric_available": True,
                "behavior_coupling_supported": True,
            }
        )
    rows.append(
        {
            "memory_arm": "memory_normal",
            "demand_cell": "memmax_useful_gap",
            "seed": 101,
            "selected_minus_population_final": 2.0,
            "descendant_minus_initial_final": 1.0,
            "effective_memory_final": 8.0,
            "memory_cost_mean_final": 0.02,
            "lineage_top1_alive_share_final": 0.35,
            "initial_distribution_audited": True,
            "initial_distribution_matched": True,
            "behavior_coupling_metric_available": True,
            "behavior_coupling_supported": True,
            "hidden_window_behavior_delta": 0.5,
        }
    )
    for demand in (
        "flat_control_clean",
        "cue_observation_off_control",
        "hazard_off_control",
        "memmax_short_gap",
        "memmax_useful_gap",
        "memmax_too_long_gap",
    ):
        rows.append(
            {
                "memory_arm": "memory_ablated",
                "demand_cell": demand,
                "seed": 101,
                "selected_minus_population_final": 0.0,
                "descendant_minus_initial_final": 0.0,
                "effective_memory_final": 0.0,
                "memory_cost_mean_final": 0.0,
                "lineage_top1_alive_share_final": 0.35,
                "initial_distribution_audited": True,
                "initial_distribution_matched": True,
                "behavior_coupling_metric_available": True,
                "behavior_coupling_supported": True,
            }
        )
    return rows


def _gate(evaluation: dict, gate_id: str) -> dict:
    for gate in evaluation["gates"]:
        if gate["gate_id"] == gate_id:
            return gate
    raise AssertionError(f"missing gate: {gate_id}")


def test_positive_fixture_passes_all_interpretation_gates() -> None:
    helper = _load_helper()
    evaluation = helper.evaluate_rows(_fixture_rows(), dataset_id="positive_fixture")
    assert evaluation["overall_status"] == "pass"
    assert evaluation["positive_memory_demand_interpretation_supported"] is True
    assert all(gate["status"] == "pass" for gate in evaluation["gates"])
    json.dumps(evaluation, sort_keys=True)


def test_ablated_control_pressure_fails_closed() -> None:
    helper = _load_helper()
    rows = _fixture_rows()
    for row in rows:
        if row["memory_arm"] == "memory_ablated" and row["demand_cell"] == "memmax_useful_gap":
            row["selected_minus_population_final"] = 3.0
    evaluation = helper.evaluate_rows(rows, dataset_id="ablated_invalid_fixture")
    assert evaluation["overall_status"] == "fail_closed"
    assert _gate(evaluation, "effective_memory_gate")["status"] == "fail"
    assert _gate(evaluation, "control_specificity_gate")["status"] == "fail"


def test_selected_parent_only_pressure_fails_descendant_and_timing_gates() -> None:
    helper = _load_helper()
    rows = _fixture_rows()
    for row in rows:
        if row["memory_arm"] == "memory_normal" and row["demand_cell"] == "memmax_useful_gap":
            row["descendant_minus_initial_final"] = 0.0
    evaluation = helper.evaluate_rows(rows, dataset_id="selected_parent_only_fixture")
    assert evaluation["overall_status"] == "fail_closed"
    assert _gate(evaluation, "descendant_movement_gate")["status"] == "fail"
    assert _gate(evaluation, "timing_replacement_gate")["status"] == "fail"


def test_lineage_dominated_fixture_fails_lineage_gate() -> None:
    helper = _load_helper()
    rows = _fixture_rows()
    for row in rows:
        if row["memory_arm"] == "memory_normal" and row["demand_cell"] == "memmax_useful_gap":
            row["lineage_top1_alive_share_final"] = 0.95
    evaluation = helper.evaluate_rows(rows, dataset_id="lineage_dominated_fixture")
    assert evaluation["overall_status"] == "fail_closed"
    assert _gate(evaluation, "founder_lineage_gate")["status"] == "fail"


def test_cost_unknown_fixture_cannot_support_costly_memory_claim() -> None:
    helper = _load_helper()
    rows = _fixture_rows()
    for row in rows:
        if row["memory_arm"] == "memory_normal":
            row["memory_cost_mean_final"] = 0.0
    evaluation = helper.evaluate_rows(rows, dataset_id="cost_unknown_fixture")
    assert evaluation["overall_status"] == "fail_closed"
    assert _gate(evaluation, "cost_gate")["status"] == "fail"
