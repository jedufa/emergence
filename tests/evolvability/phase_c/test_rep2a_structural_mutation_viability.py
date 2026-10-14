from __future__ import annotations

import json
import random

from evolution.operators_c1 import operator_catalog
from evolution.structural_mutation_viability import (
    CONNECTION_EXECUTION_STATUS,
    apply_single_operator,
    changed_effective_fields,
    complexity_cost,
    constraint_probe_summary,
    empty_control_genome,
    inheritance_and_retention_summary,
    integrity_summary,
    operator_battery,
    operator_map,
    phenotype_reach_class,
    phenotype_reach_summary,
    recombination_summary,
    rep2a_decision,
    reporting_disabled_summary,
    seeded_fixture_genome,
    stable_hash,
    stable_json,
    viability_summary_payload,
)
from tools.scripts.rep2a_structural_mutation_viability_evidence import build_outputs


def test_rep2a_reporting_disabled_is_noop_and_consumes_no_random_draws() -> None:
    rng = random.Random(1234)
    before = rng.getstate()
    summary = reporting_disabled_summary()
    after = rng.getstate()
    assert summary["reporting_enabled"] is False
    assert summary["behavior_change"] is False
    assert summary["random_draws_consumed"] == 0
    assert before == after


def test_rep2a_default_off_control_remains_empty() -> None:
    genome = empty_control_genome()
    assert genome.modules == []
    assert genome.connections == []


def test_rep2a_operator_map_classifies_every_catalog_operator() -> None:
    mapped = {row["operator_name"] for row in operator_map()}
    catalog = {spec.op_id for spec in operator_catalog()}
    assert mapped == catalog
    assert all(row["phenotype_reach_class"] in {"live_module_effect", "stored_connection_only", "mixed", "no_effect", "mixed_stored_without_live_effect"} for row in operator_map())


def test_rep2a_duplicate_is_valid_neutral_module_change_on_rep1c_seed() -> None:
    row = apply_single_operator("duplicate_with_decay", (0.1, 0.2, 0.3))
    assert row["classification"] == "valid_neutral_representation_change"
    assert row["module_count_delta"] == 1
    assert row["changed_effective_fields"] == []


def test_rep2a_gate_drift_is_stored_only_when_weight_changes() -> None:
    row = apply_single_operator("gate_drift", (0.1, 0.2))
    assert row["classification"] == "valid_stored_only_change"
    assert row["phenotype_reach_class"] == "stored_connection_only"
    assert row["connection_execution_status"] == CONNECTION_EXECUTION_STATUS


def test_rep2a_gate_drift_exact_duplicate_is_explicit_noop() -> None:
    row = apply_single_operator("gate_drift", (0.4, 0.5))
    assert row["classification"] == "noop"
    assert row["payload_changed"] is False


def test_rep2a_invalid_rejection_probe_has_explicit_reason() -> None:
    probe = constraint_probe_summary()
    assert probe["invalid_rejection"]["rejected"] is True
    assert "DET_GENOME_B2_TYPE_MISMATCH" in probe["invalid_rejection"]["error_code"]


def test_rep2a_module_and_connection_cap_blocks_are_counted() -> None:
    probe = constraint_probe_summary()
    assert probe["module_cap"]["attempted"] == 14
    assert probe["module_cap"]["kept"] == 12
    assert probe["module_cap"]["dropped"] == 2
    assert probe["connection_cap"]["attempted"] == 36
    assert probe["connection_cap"]["kept"] == 32
    assert probe["connection_cap"]["dropped"] == 4


def test_rep2a_locality_block_is_counted() -> None:
    probe = constraint_probe_summary()
    assert probe["locality"]["attempted"] == 1
    assert probe["locality"]["kept"] == 0
    assert probe["locality"]["dropped"] == 1


def test_rep2a_fusion_compression_reduces_structure_validly() -> None:
    row = apply_single_operator("fusion_merge_compression", (0.1, 0.2))
    assert row["classification"] == "valid_compression"
    assert row["module_count_delta"] == -1
    assert row["catastrophic_integrity_failure"] is False


def test_rep2a_hand_authored_live_module_change_classifier() -> None:
    before = seeded_fixture_genome().canonical_payload()
    after = json.loads(stable_json(before))
    for module in after["modules"]:
        if module["module_id"] == "m010_behavior_bias":
            module["params"]["aggression_bias"] = 0.33
    assert changed_effective_fields(before, after) == ["aggression_bias"]
    assert phenotype_reach_class(before, after) == "live_module_effect"


def test_rep2a_current_operator_battery_has_no_live_effects() -> None:
    summary = phenotype_reach_summary(operator_battery())
    assert summary["connection_execution_status"] == "not_live"
    assert summary["live_module_effect_count"] == 0
    assert summary["exact_effective_fields_changed"] == []


def test_rep2a_inheritance_records_changed_structures() -> None:
    retention = inheritance_and_retention_summary(operator_battery())
    assert retention["inherited_change_count"] >= 1
    assert all(row["originating_parent"] == "fixture_parent" for row in retention["records"])


def test_rep2a_short_and_medium_retention_are_exact_streaks() -> None:
    retention = inheritance_and_retention_summary(operator_battery())
    assert retention["short_retained_change_count"] == retention["inherited_change_count"]
    assert retention["medium_retained_change_count"] == retention["inherited_change_count"]


def test_rep2a_loss_and_empty_collapse_are_separate() -> None:
    retention = inheritance_and_retention_summary(operator_battery())
    assert retention["lost_change_count"] == 0
    assert retention["empty_collapse_count"] == 0


def test_rep2a_recombination_path_preserves_or_explains_change() -> None:
    summary = recombination_summary()
    assert summary["change_survived_or_loss_explained"] is True
    assert summary["child_module_count"] >= 3


def test_rep2a_complexity_cost_before_after_values_are_present() -> None:
    row = apply_single_operator("edge_split_insertion", (0.4, 0.5, 0.6))
    before = complexity_cost(row["before_payload"])
    after = complexity_cost(row["after_payload"])
    assert before["representation_cost"] >= 0.0
    assert after["representation_cost"] >= before["representation_cost"]


def test_rep2a_integrity_selects_limited_rep2a_b_decision() -> None:
    battery = operator_battery()
    integrity = integrity_summary(battery, inheritance_and_retention_summary(battery))
    decision = rep2a_decision(integrity)
    assert integrity["zero_integrity_failures"] is True
    assert integrity["live_module_effect_changes"] == 0
    assert decision["decision"] == "REP-2A-B"


def test_rep2a_stable_json_is_byte_stable() -> None:
    payload = viability_summary_payload()
    assert stable_json(payload) == stable_json(payload)
    assert stable_hash(payload) == stable_hash(payload)


def test_rep2a_metrics_isolation_does_not_mutate_seed_genome() -> None:
    genome = seeded_fixture_genome()
    before = genome.canonical_hash()
    _ = viability_summary_payload()
    after = genome.canonical_hash()
    assert before == after


def test_rep2a_evidence_builder_writes_checkpoint_and_w6_support(tmp_path) -> None:
    build_outputs(tmp_path, [])
    checkpoint = json.loads((tmp_path / "rep2a_checkpoint_resume_summary.json").read_text(encoding="utf-8"))
    w6 = json.loads((tmp_path / "rep2a_w6_support_summary.json").read_text(encoding="utf-8"))
    assert checkpoint["payloads_match"] is True
    assert checkpoint["random_stream_changes_introduced_by_reporting"] is False
    assert w6["silent_omission"] is False
    assert w6["equivalent_report_facing_outputs"] is True
