from __future__ import annotations

import json
from pathlib import Path

from tools.evidence.update_campaign_index import update_campaign_index


def _campaign(root: Path, name: str, decision: str, timestamp: str) -> None:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "decision_summary.json").write_text(
        json.dumps(
            {
                "campaign_id": "COG-P1",
                "profile": "COG-P1",
                "decision": decision,
                "git_commit": f"commit-{name}",
                "branch": f"branch-{name}",
                "generated_at": timestamp,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    for filename in (
        "campaign_evidence_manifest.json",
        "decision_report.md",
        "paired_comparisons.csv",
        "anomaly_summary.json",
        "schema_status.json",
    ):
        (folder / filename).write_text("x\n", encoding="utf-8")


def test_campaign_index_is_idempotent_and_lists_invalid_campaigns(tmp_path: Path) -> None:
    _campaign(tmp_path, "old", "B", "2026-07-01T00:00:00Z")
    _campaign(tmp_path, "new", "D", "2026-07-02T00:00:00Z")
    (tmp_path / "broken").mkdir()

    path = update_campaign_index(tmp_path)
    first = path.read_text(encoding="utf-8")
    second_path = update_campaign_index(tmp_path)
    second = second_path.read_text(encoding="utf-8")

    assert first == second
    assert first.count("| COG-P1 | COG-P1 |") == 2
    assert "missing_artifacts" in first
    assert "commit-new" in first
    assert "[decision_report.md](new/decision_report.md)" in first
