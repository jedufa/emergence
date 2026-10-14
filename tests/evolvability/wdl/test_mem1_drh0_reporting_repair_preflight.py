from __future__ import annotations

import json
from pathlib import Path

from tools.bin.mem1_drh0_reporting_repair_preflight_report import build_summary


def test_mem1_drh0_reporting_repair_preflight_gate_blocks_uninterpretable_memory_arm_path():
    fixture_path = Path(__file__).with_name("fixtures") / "mem1_drh0_reporting_preflight_inputs.json"
    summary = build_summary(inputs=json.loads(fixture_path.read_text(encoding="utf-8")))

    assert summary["base_mem1_decision"] == "MEM-1-B"
    assert summary["source_integrity"]["reporting_clean_for_fixture_scope"] is True
    assert summary["pre_run_gate"]["gate"] == "PRE-RUN-BLOCKED"
    assert summary["decision"]["label"] == "MEM1-DRH0-RP1-B"
    assert summary["campaigns_run"] == 0
    assert summary["preflights_run"] == 0

    surfaces = {item["surface"]: item for item in summary["reporting_surfaces"]}
    assert surfaces["cue_visible_phase_rows"]["status"] == "pass"
    assert surfaces["hidden_window_rows"]["status"] == "pass"
    assert surfaces["pre_consequence_refuge_arrival_or_use"]["status"] == "pass_fixture_level"
    assert surfaces["memory_normal_vs_ablated_arm_labeling"]["status"] == "blocked_for_preflight"
    assert surfaces["lineage_sanity"]["status"] == "not_applicable_fixture_only"

    requirements = {item["requirement"]: item["passed"] for item in summary["pre_run_gate"]["requirements"]}
    assert requirements["hidden-window and pre-consequence refuge-use metrics are emitted"] is True
    assert requirements["memory-normal and memory-ablated can be labeled cleanly"] is False
    assert requirements["output can be interpreted as preflight only"] is False
