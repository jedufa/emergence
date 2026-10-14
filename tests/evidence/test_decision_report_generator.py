from __future__ import annotations

from pathlib import Path

from tests.evidence.cogp1_fixtures import cogp1_manifest
from tools.evidence.reports import generate_campaign_decision_report


def test_markdown_report_contains_required_sections(tmp_path: Path) -> None:
    manifest_path = cogp1_manifest(tmp_path, signal="supported")
    out_dir = tmp_path / "report"
    generate_campaign_decision_report(manifest_path, "COG-P1", out_dir, generated_at="2026-07-02T00:00:00Z")

    text = (out_dir / "decision_report.md").read_text(encoding="utf-8")
    for heading in (
        "Executive Verdict",
        "Campaign Design",
        "Run Completeness",
        "Schema Contract Status",
        "Determinism/Repeat Status",
        "Primary Deltas",
        "Pressure-Specificity Result",
        "Lineage/Founder Arm Summary",
        "Behavior Proxy Summary",
        "Cost Summary",
        "Anomalies/Failures",
        "A/B/C/D Decision",
        "Claims Allowed",
        "Claims Not Allowed",
        "Recommended Next Action",
    ):
        assert f"## {heading}" in text
