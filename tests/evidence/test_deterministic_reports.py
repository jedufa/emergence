from __future__ import annotations

from pathlib import Path

from tests.evidence.cogp1_fixtures import cogp1_manifest
from tools.evidence.reports import generate_campaign_decision_report


ARTIFACTS = (
    "decision_report.md",
    "decision_summary.json",
    "paired_comparisons.csv",
    "run_status.csv",
    "anomaly_summary.json",
    "schema_status.json",
)


def test_deterministic_reports_are_byte_for_byte_equal(tmp_path: Path) -> None:
    manifest_path = cogp1_manifest(tmp_path, signal="supported")
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    generate_campaign_decision_report(manifest_path, "COG-P1", out_a, deterministic=True)
    generate_campaign_decision_report(manifest_path, "COG-P1", out_b, deterministic=True)

    for artifact in ARTIFACTS:
        assert (out_a / artifact).read_bytes() == (out_b / artifact).read_bytes()


def test_explicit_generated_at_is_stable(tmp_path: Path) -> None:
    manifest_path = cogp1_manifest(tmp_path, signal="supported")
    out = tmp_path / "report"
    generate_campaign_decision_report(manifest_path, "COG-P1", out, generated_at="2026-07-02T00:00:00Z")

    assert b"2026-07-02T00:00:00Z" in (out / "decision_report.md").read_bytes()
