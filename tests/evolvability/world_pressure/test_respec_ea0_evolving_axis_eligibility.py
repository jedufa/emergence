from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.bin import respec_ea0_evolving_axis_eligibility as ea0


def test_respec_ea0_records_live_d1_evidence_sha_and_prompt_reconciliation() -> None:
    verification = ea0.d1_evidence_verification()

    assert verification["status"] == "PASS"
    assert verification["d1_decision"] == "RESPEC-D1-B"
    assert verification["actual_d1_registry_sha256"] == ea0.EXPECTED_D1_REGISTRY_SHA256
    assert verification["actual_d1_zip_sha256"] != ea0.PROMPT_EXPECTED_D1_ZIP_SHA256
    assert verification["prompt_sha_reconciliation"] == "PROMPT_STALE_LIVE_FILESYSTEM_USED"


def test_respec_ea0_offspring_probe_uses_live_crossover_and_supplies_both_directions() -> None:
    rows = ea0._offspring_probe_rows(samples_per_case=400)
    outcomes = ea0.mixed_world_arm_outcomes()
    arms = dict(outcomes["arms"])
    birth_rate = ea0._mean([float(dict(arms[arm])["successful_births_final_mean"]) for arm in arms]) / 48.0
    summary = ea0.mutation_supply_summary(rows, birth_rate)
    neutral = next(row for row in summary["case_summaries"] if row["case"] == "neutral_x_neutral")

    assert summary["status"] == "PASS"
    assert summary["both_direction_reachability"] is True
    assert neutral["positive_fraction"] > 0.0
    assert neutral["negative_fraction"] > 0.0
    assert neutral["clamp_fraction"] == 0.0
    assert summary["neutral_parent_expected_new_non_neutral_variants_per_100_births"] > 0.0


def test_respec_ea0_classifies_d1_mixed_world_as_directional_b_not_two_niche_ready() -> None:
    balance = ea0.mixed_world_resource_balance()
    outcomes = ea0.mixed_world_arm_outcomes()
    classification = ea0.mixed_world_classification(balance, outcomes)
    seed_support = outcomes["seed_support"]

    assert balance["resource_b_to_a_success_ratio"] > 1.0
    assert seed_support["neutral_gt_a_favored_births"] == 4
    assert seed_support["b_favored_gt_a_favored_births"] == 4
    assert classification["status"] == "FAIL"
    assert classification["classification"] == "DIRECTIONAL_B_OPTIMUM_WITH_NEUTRAL_B_PLATEAU"
    assert classification["two_niche_campaign_authorized"] is False


def test_respec_ea0_final_decision_blocks_registry() -> None:
    payloads, _rows = ea0.build_payloads(samples_per_case=400)
    decision = payloads["respec_ea0_final_decision.json"]
    registry = payloads["respec_ea0_registry_contract.json"]
    later_gate = payloads["respec_ea0_observation_action_audit.json"]

    assert decision["decision"] == "RESPEC-EA0-B"
    assert decision["campaign_authorized"] is False
    assert decision["failed_gate"] == 4
    assert registry["status"] == "BLOCKED_NOT_AUTHORIZED"
    assert registry["registry_created"] is False
    assert later_gate["status"] == "NOT_EVALUATED"


def test_respec_ea0_write_outputs_creates_complete_compact_package(tmp_path: Path, monkeypatch) -> None:
    evidence_dir = tmp_path / "evidence"
    report_path = tmp_path / "world-pressure_EIEL_RESPEC_EA0_Evolving_Axis_Selection_Eligibility_2026_07_11.md"
    monkeypatch.setattr(ea0, "EVIDENCE_DIR", evidence_dir)
    monkeypatch.setattr(ea0, "REPORT_PATH", report_path)

    result = ea0.write_outputs(samples_per_case=400)

    assert result["decision"] == "RESPEC-EA0-B"
    assert report_path.exists()
    assert (evidence_dir / report_path.name).exists()
    assert Path(result["evidence_zip"]).exists()
    manifest = json.loads((evidence_dir / "respec_ea0_evidence_bundle_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sha256"] == result["evidence_zip_sha256"]
    assert "respec_ea0_final_decision.json" in manifest["members"]
    with (evidence_dir / "respec_ea0_offspring_axis_probe.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
