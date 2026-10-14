from __future__ import annotations

import json
from pathlib import Path

from tests.eiel_actor_v11_helpers import REPO_ROOT, write_preflight_receipt
from tools.actors.common import new_record
from tools.actors import mission_planner
from tools.actors.store import ActorRecordStore

FIXED = "2026-08-03T12:00:00Z"


def _spec() -> dict:
    return {
        "title": "Bounded memory robustness battery",
        "mission_type": "ROBUSTNESS_BATTERY",
        "question": "Does the bounded WDL memory signal survive fixed hazard variation?",
        "decision_to_enable": "Close or retain the WDL-specific robustness claim.",
        "scope": ["three preregistered delayed-hazard variants"],
        "out_of_scope": ["coefficient tuning", "general cognition"],
        "controls": ["memory normal", "memory ablated"],
        "success_criteria": ["predeclared robustness decision rule passes"],
        "failure_criteria": ["predeclared robustness decision rule fails"],
        "stopping_rules": ["stop after the fixed matrix"],
        "decision_rule": "Apply the versioned evidence profile without post-hoc widening.",
        "claim_ceiling": "Bounded WDL-specific memory robustness only.",
        "required_evidence": ["paired outputs", "reward-leakage audit"],
        "technical_qualification_requirements": ["current CUDA route authority"],
        "resource_demand": {"device": "cuda", "minimum_vram_available_gib": 4.0},
        "tasks": ["implement variants", "run tests", "run campaign", "assess evidence"],
        "required_closeout_refs": [
            "evidence_refs", "qualification_refs", "resource_refs", "report_refs",
            "campaign_output_refs", "source_identity", "librarian_document_ids",
        ],
    }


def _create_and_authorize(source: Path, state: Path, capsys) -> tuple[str, dict]:
    assert mission_planner.main([
        "create", "--input", str(source), "--ready", "--state-root", str(state),
        "--generated-at", FIXED,
    ]) == 0
    created = json.loads(capsys.readouterr().out)
    mission_id = created["mission"]["mission_id"]
    assert created["mission"]["status"] == "DRAFT"
    assert created["status"] == "BLOCKED"
    assert created["next_actions"][0]["action"] == "run_evidence_preflight"

    receipt = write_preflight_receipt(state / "fixture-preflight.json")
    assert mission_planner.main([
        "preflight", mission_id,
        "--preflight-receipt", str(receipt),
        "--authorization-basis", "operator_instruction",
        "--operator-instruction-ref", "test operator approval",
        "--repo-root", str(REPO_ROOT),
        "--ready", "--state-root", str(state),
        "--generated-at", "2026-08-03T12:00:30Z",
    ]) == 0
    authorized = json.loads(capsys.readouterr().out)
    assert authorized["mission"]["status"] == "READY"
    assert authorized["mission"]["authority_ready"] is True
    return mission_id, authorized


def test_mission_planner_create_preflight_update_history_closeout(tmp_path: Path, capsys) -> None:
    source = tmp_path / "mission.json"
    source.write_text(json.dumps(_spec()), encoding="utf-8")
    state = tmp_path / "state"
    mission_id, authorized = _create_and_authorize(source, state, capsys)

    qualification_id = "QCHK-TEST-ACCEPTED"
    resource_id = "RADM-TEST-ACCEPTED"
    evidence_id = "EDM-TEST-DECIDED"
    ActorRecordStore(state, "qualifications").put(new_record(
        actor="qualificationManager", record_type="technical_qualification_check",
        record_id=qualification_id, status="QUALIFIED", payload={"result": {"decision": "QUALIFIED"}},
        generated_at=FIXED,
    ))
    ActorRecordStore(state, "resources").put(new_record(
        actor="resourceManager", record_type="resource_admission_recommendation",
        record_id=resource_id, status="ADMIT", payload={"decision": "ADMIT"},
        generated_at=FIXED,
    ))
    ActorRecordStore(state, "evidence-decisions").put(new_record(
        actor="evidenceManager", record_type="evidence_assessment",
        record_id=evidence_id, status="DECIDED_B", payload={"procedural_decision": "B"},
        generated_at=FIXED,
    ))
    campaign_output = tmp_path / "campaign-output"
    campaign_output.mkdir()
    report = tmp_path / "implementation-report.md"
    report.write_text(authorized["evidence_preflight_markdown"] + "\n# Test report\n", encoding="utf-8")

    assert mission_planner.main([
        "update", mission_id, "--qualification-ref", qualification_id,
        "--resource-ref", resource_id, "--status", "ACTIVE", "--state-root", str(state),
        "--generated-at", "2026-08-03T12:01:00Z",
    ]) == 0
    capsys.readouterr()

    assert mission_planner.main([
        "closeout", mission_id,
        "--outcome", "completed",
        "--decision", "negative result accepted",
        "--no-claims-statement", "No general memory or cognition claim is made.",
        "--evidence-ref", evidence_id,
        "--qualification-ref", qualification_id,
        "--resource-ref", resource_id,
        "--campaign-output-ref", str(campaign_output),
        "--report-ref", str(report),
        "--librarian-document-id", "EIEL-DOC-000005",
        "--source-commit", "test-source-commit",
        "--strict-references",
        "--state-root", str(state),
        "--generated-at", "2026-08-03T12:02:00Z",
    ]) == 0
    closed = json.loads(capsys.readouterr().out)
    assert closed["mission"]["status"] == "CLOSED"
    assert closed["closeout"]["evidence_preflight_markdown"].startswith("## Evidence Preflight")

    assert mission_planner.main(["history", mission_id, "--state-root", str(state)]) == 0
    history = json.loads(capsys.readouterr().out)
    assert history["revision_count"] == 4


def test_mission_planner_rejects_execution_and_missing_stop_rules(tmp_path: Path) -> None:
    raw = _spec()
    raw["commands"] = ["python run.py"]
    raw["stopping_rules"] = []
    normalized = mission_planner.normalize_mission_spec(raw)
    validation = mission_planner.validate_mission_spec(normalized)
    assert validation["valid"] is False
    assert any("executable field" in error for error in validation["errors"])
    assert any("stopping_rules" in error for error in validation["errors"])
    assert any("preflight" in error.lower() for error in validation["errors"])


def test_mission_planner_validate_is_blocked_until_preflight_then_recovers(tmp_path: Path, capsys) -> None:
    source = tmp_path / "mission.json"
    source.write_text(json.dumps(_spec()), encoding="utf-8")
    state = tmp_path / "state"

    assert mission_planner.main(["validate", str(source), "--state-root", str(state)]) == 1
    blocked = json.loads(capsys.readouterr().out)
    assert blocked["status"] == "BLOCKED"
    assert blocked["next_actions"][0]["action"] == "run_evidence_preflight"

    mission_id, _ = _create_and_authorize(source, state, capsys)

    assert mission_planner.main(["validate", mission_id, "--state-root", str(state)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["validation"]["valid"] is True

    assert mission_planner.main(["inspect", mission_id, "--state-root", str(state)]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["record"]["record_id"] == mission_id

    assert mission_planner.main(["status", "--state-root", str(state)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["count"] == 1
    assert status["status_counts"] == {"READY": 1}
