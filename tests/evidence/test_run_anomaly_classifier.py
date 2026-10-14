from __future__ import annotations

import json
from pathlib import Path

from tests.evidence.test_schema_checker import build_cogp1_fixture
from tools.evidence.contracts import check_campaign_schema
from tools.evidence.profiles import get_profile
from tools.evidence.runs import classify_runs, run_status_csv_rows


def test_run_status_table_is_csv_serializable(tmp_path: Path) -> None:
    manifest_path, _runs = build_cogp1_fixture(tmp_path)
    result = check_campaign_schema(manifest_path)

    rows = run_status_csv_rows(result["run_status"])
    assert rows
    assert {"campaign_id", "profile", "run_id", "scenario_id", "status"}.issubset(rows[0])


def test_unknown_extra_scenario_is_classified(tmp_path: Path) -> None:
    manifest_path, runs = build_cogp1_fixture(tmp_path)
    summary_path = Path(runs[0]["run_dir"]).parents[0] / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    extra = dict(summary["runs"][0])
    extra["run_id"] = "extra-run"
    extra["axes"] = dict(extra["axes"])
    extra["axes"]["ecology_context"] = "unknown_context"
    summary["runs"].append(extra)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = check_campaign_schema(manifest_path)
    counts = result["anomaly_summary"]["counts"]
    assert result["ok"] is False
    assert counts["unknown_scenario"] >= 1
    assert counts["extra_run"] >= 1


def test_classifier_direct_api_reports_anomaly_summary(tmp_path: Path) -> None:
    manifest_path, _runs = build_cogp1_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    classified = classify_runs(manifest, get_profile("COG-P1"))

    assert len(classified["rows"]) == 120
    assert classified["anomaly_summary"]["counts"]["complete"] == 120
