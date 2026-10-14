from __future__ import annotations

import json
from pathlib import Path

from tests.evidence.cogp1_fixtures import cogp1_manifest
from tests.eiel_actor_v11_helpers import authorize_mission, run_eiel, scientific_mission_spec
from tools.actors import evidence_manager

FIXED = "2026-08-03T12:00:00Z"


def _authorized_mission(tmp_path: Path, *, state: Path) -> str:
    mission_path = tmp_path / "mission.json"
    mission_path.write_text(json.dumps(scientific_mission_spec(), sort_keys=True), encoding="utf-8")
    created = run_eiel(
        ["missionPlanner", "create", "--input", str(mission_path), "--ready", "--generated-at", FIXED],
        state_root=state,
    )
    mission_id = created["mission"]["mission_id"]
    authorized = authorize_mission(
        mission_id,
        state_root=state,
        ready=True,
        generated_at="2026-08-03T12:00:01Z",
        authority_domain="campaign_decision",
    )
    assert authorized["status"] == "READY"
    return mission_id


def test_evidence_manager_inspects_assesses_and_exports(tmp_path: Path, capsys) -> None:
    manifest = cogp1_manifest(tmp_path / "campaign", signal="supported")
    state = tmp_path / "state"
    reports = tmp_path / "reports"
    mission_id = _authorized_mission(tmp_path, state=state)

    assert evidence_manager.main(["inspect", str(manifest), "--state-root", str(state)]) == 0
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["inspection"]["readiness"] == "READY_FOR_ASSESSMENT"

    assert evidence_manager.main([
        "assess", str(manifest), "--state-root", str(state), "--out-dir", str(reports),
        "--mission", mission_id, "--generated-at", FIXED, "--deterministic",
    ]) == 0
    assessed = json.loads(capsys.readouterr().out)
    assessment_id = assessed["assessment_id"]
    assert assessed["assessment"]["procedural_decision"] == "A"
    assert assessed["assessment"]["scientific_qualification_effect"] == "none"

    export = tmp_path / "assessment.zip"
    assert evidence_manager.main([
        "export", assessment_id, "--export-output", str(export), "--state-root", str(state),
    ]) == 0
    exported = json.loads(capsys.readouterr().out)
    assert exported["file_count"] >= 8
    assert export.is_file()


def test_evidence_manager_prepare_fails_closed_on_missing_metadata(tmp_path: Path, capsys) -> None:
    campaign = tmp_path / "empty-campaign"
    campaign.mkdir()
    assert evidence_manager.main([
        "prepare", str(campaign), "--profile", "COG-P1", "--generated-at", FIXED,
    ]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "DRAFT"
    assert result["unresolved_fields"]
    assert Path(result["manifest_path"]).name == "campaign_evidence_manifest.draft.json"


def test_evidence_manager_discover_validate_status_list_and_show(tmp_path: Path, capsys) -> None:
    manifest = cogp1_manifest(tmp_path / "campaign", signal="supported")
    canonical_manifest = manifest.parent / "campaign_evidence_manifest.json"
    canonical_manifest.write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = canonical_manifest
    state = tmp_path / "state"
    reports = tmp_path / "reports"
    mission_id = _authorized_mission(tmp_path, state=state)

    assert evidence_manager.main(["discover", "--root", str(tmp_path)]) == 0
    discovered = json.loads(capsys.readouterr().out)
    assert discovered["count"] >= 1
    assert any(row.get("readiness") == "READY_FOR_ASSESSMENT" for row in discovered["campaigns"])

    assert evidence_manager.main(["validate", str(manifest)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["inspection"]["readiness"] == "READY_FOR_ASSESSMENT"

    assert evidence_manager.main([
        "assess", str(manifest), "--state-root", str(state), "--out-dir", str(reports),
        "--mission", mission_id, "--generated-at", FIXED, "--deterministic",
    ]) == 0
    assessed = json.loads(capsys.readouterr().out)
    assessment_id = assessed["assessment_id"]

    assert evidence_manager.main(["status", assessment_id, "--state-root", str(state)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["assessment"]["decision"] == "A"

    assert evidence_manager.main(["list", "--state-root", str(state)]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["count"] == 1

    assert evidence_manager.main(["show", assessment_id, "--state-root", str(state)]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["record"]["record_id"] == assessment_id
