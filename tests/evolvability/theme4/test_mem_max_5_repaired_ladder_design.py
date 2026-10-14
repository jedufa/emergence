from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "tools/bin/mem_max_5_repaired_ladder_design.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("mem_max_5_repaired_ladder_design", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mem_max_5_decision_is_reporting_repair_b() -> None:
    helper = _load_helper()
    fixtures = helper.gate_evaluator_fixture_results()
    decision = helper.decision_payload(fixtures)
    assert decision["decision"] == "MEM-MAX-5B"
    assert decision["repaired_ladder_designed"] is True
    assert decision["scientific_registry_created"] is False
    assert decision["scientific_campaign_or_preflight_run"] is False
    assert decision["gate_inputs_planned"] is True
    assert decision["gate_inputs_currently_available_for_future_run"] is False
    json.dumps(decision, sort_keys=True)


def test_gate_mapping_covers_all_mem_max_4_gates() -> None:
    helper = _load_helper()
    mapping = helper.gate_requirement_mapping()
    gate_ids = {gate["gate_id"] for gate in mapping["gates"]}
    assert gate_ids == {
        "effective_memory_gate",
        "descendant_movement_gate",
        "control_specificity_gate",
        "behavior_coupling_gate",
        "founder_lineage_gate",
        "cost_gate",
        "timing_replacement_gate",
    }
    assert all(gate["planned_outputs"] for gate in mapping["gates"])


def test_repaired_ladder_is_small_and_has_ablated_controls() -> None:
    helper = _load_helper()
    ladder = helper.repaired_ladder_design()
    assert ladder["planned_runs_if_repaired_reporting_exists"] == 12
    assert ladder["scientific_registry_created"] is False
    cells = ladder["cells"]
    arms = {cell["memory_arm"] for cell in cells}
    demands = {cell["demand_cell"] for cell in cells}
    assert arms == {"memory_normal", "memory_ablated"}
    assert "memmax_useful_gap" in demands
    assert "cue_observation_off_control" in demands
    assert "hazard_off_control" in demands


def test_gate_fixtures_prove_pass_and_fail_closed_cases() -> None:
    helper = _load_helper()
    results = helper.gate_evaluator_fixture_results()["fixtures"]
    assert results["synthetic_all_gates_pass"]["overall_status"] == "pass"
    assert results["synthetic_all_gates_pass"]["blocking_gate_count"] == 0
    assert results["missing_behavior_coupling"]["overall_status"] == "fail_closed"
    assert results["missing_behavior_coupling"]["behavior_gate_status"] == "unknown"
    assert results["mem_max_2_recheck"]["overall_status"] == "fail_closed"
    assert results["mem_max_3_recheck"]["overall_status"] == "fail_closed"
