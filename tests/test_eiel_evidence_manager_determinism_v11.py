from __future__ import annotations

import json
from pathlib import Path

from tests.evidence.cogp1_fixtures import cogp1_manifest
from tools.actors import evidence_manager

FIXED = "2026-08-03T12:00:00Z"


def test_repeated_deterministic_assessment_and_export_are_byte_stable(tmp_path: Path, capsys) -> None:
    manifest = cogp1_manifest(tmp_path / "campaign", signal="supported")
    state = tmp_path / "state"
    reports = tmp_path / "reports"
    args = [
        "assess", str(manifest), "--state-root", str(state), "--out-dir", str(reports),
        "--generated-at", FIXED, "--deterministic",
    ]
    assert evidence_manager.main(args) == 0
    first = json.loads(capsys.readouterr().out)
    report_bytes = (reports / "decision_report.md").read_bytes()
    assert evidence_manager.main(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert first["assessment_id"] == second["assessment_id"]
    assert (reports / "decision_report.md").read_bytes() == report_bytes

    export_a = tmp_path / "a.zip"
    export_b = tmp_path / "b.zip"
    for target in (export_a, export_b):
        assert evidence_manager.main([
            "export", first["assessment_id"], "--export-output", str(target), "--state-root", str(state),
        ]) == 0
        capsys.readouterr()
    assert export_a.read_bytes() == export_b.read_bytes()
