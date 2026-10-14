from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "tools/bin/mem_max_6_reporting_repair_and_preflight.py"
REGISTRY = ROOT / "_spec/registry/v1/mem_max_6_repaired_memory_demand_tiny_preflight.yaml"


def _load_helper():
    spec = importlib.util.spec_from_file_location("mem_max_6_reporting_repair_and_preflight", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_is_tiny_gate_wired_and_behavior_neutral() -> None:
    helper = _load_helper()
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    validation = helper.validate_registry(registry)
    assert validation["valid"], validation["errors"]
    assert validation["planned_runs"] == 10
    assert registry["matrix_order"] == ["memory_arm", "demand_cell", "seed"]


def test_reporting_field_map_covers_required_concepts() -> None:
    helper = _load_helper()
    field_map = helper.reporting_repair_field_map()
    actual = field_map["actual_field_map"]
    for required in (
        "mem_max_hidden_window_behavior_metric_available",
        "mem_max_hidden_window_behavior_coupling_supported",
        "mem_max_initial_memory_dim_histogram",
        "mem_max_founder_distribution_matched_or_audited",
        "mem_max_selected_parent_to_descendant_replacement_supported",
        "mem_max_timing_replacement_metric_available",
        "mem_max_gate_input_schema_version",
    ):
        assert required in actual
        assert actual[required]
    assert field_map["behavior_neutrality"]["reporting_only"] is True


def test_mem_max_6_gate_fixtures_cover_all_required_failures() -> None:
    helper = _load_helper()
    results = helper.gate_evaluator_fixture_results()
    assert results["status"] == "passed"
    summary = results["summary"]
    assert summary["all_gates_pass"]["overall_status"] == "pass"
    for name in (
        "missing_behavior_coupling",
        "unmatched_initial_distribution",
        "selected_parent_only",
        "ablated_useful_gap_stronger",
        "top_lineage_dominance",
        "mem_max_2_recheck",
        "mem_max_3_recheck",
    ):
        assert summary[name]["overall_status"] == "fail_closed"
    json.dumps(results["summary"], sort_keys=True)
