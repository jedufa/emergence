from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tests.evidence.cogp1_fixtures import cogp1_manifest
from tools.evidence.contracts import check_campaign_schema
from tools.evidence.decisions import analyze_profile
from tools.evidence.generate_campaign_decision_report import main as report_main


def _decision(manifest_path: Path) -> dict:
    schema = check_campaign_schema(manifest_path)
    return analyze_profile(schema, "COG-P1")


def test_cog_p1_signal_supported_recommends_a(tmp_path: Path) -> None:
    decision = _decision(cogp1_manifest(tmp_path, signal="supported"))

    assert decision["decision"] == "A"
    assert decision["pressure_signal_supported"] is True
    assert any("bounded COG-P1 go criterion" in claim for claim in decision["claims_allowed"])
    assert any("Do not claim cognition emergence" in claim for claim in decision["claims_not_allowed"])


def test_cog_p1_no_signal_recommends_b(tmp_path: Path) -> None:
    decision = _decision(cogp1_manifest(tmp_path, signal="none"))

    assert decision["decision"] == "B"
    assert decision["pressure_signal_supported"] is False


def test_cog_p1_missing_run_recommends_c(tmp_path: Path) -> None:
    manifest_path = cogp1_manifest(tmp_path, signal="supported")
    run_root = manifest_path.parent / "runs"
    summary_path = run_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runs"] = summary["runs"][:-1]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    decision = _decision(manifest_path)
    assert decision["decision"] == "C"
    assert "interpretation" in " ".join(decision["rationale"]) or "coverage" in " ".join(decision["rationale"])


def test_cog_p1_reward_leakage_recommends_d(tmp_path: Path) -> None:
    manifest_path = cogp1_manifest(tmp_path, signal="supported")
    run_root = manifest_path.parent / "runs"
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    metrics_path = Path(summary["runs"][0]["run_dir"]) / "metrics.csv"
    row = next(csv.DictReader(metrics_path.open(encoding="utf-8")))
    row["direct_cognition_reward_present_flag"] = "1"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    decision = _decision(manifest_path)
    assert decision["decision"] == "D"
    assert decision["reward_audit"]["active_fields"]


def test_placeholder_profiles_fail_clearly(tmp_path: Path) -> None:
    manifest_path = cogp1_manifest(tmp_path, signal="supported")
    with pytest.raises(NotImplementedError):
        analyze_profile(check_campaign_schema(manifest_path), "COG-P2")


def test_report_cli_writes_required_artifacts(tmp_path: Path) -> None:
    manifest_path = cogp1_manifest(tmp_path, signal="supported")
    out_dir = tmp_path / "report"

    assert report_main(["--manifest", str(manifest_path), "--profile", "COG-P1", "--out-dir", str(out_dir), "--deterministic"]) == 0
    for filename in (
        "decision_report.md",
        "decision_summary.json",
        "paired_comparisons.csv",
        "run_status.csv",
        "anomaly_summary.json",
        "schema_status.json",
        "campaign_evidence_manifest.json",
    ):
        assert (out_dir / filename).exists()
    summary = json.loads((out_dir / "decision_summary.json").read_text(encoding="utf-8"))
    assert summary["decision"] == "A"
