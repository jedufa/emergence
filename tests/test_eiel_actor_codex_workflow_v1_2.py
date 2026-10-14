from __future__ import annotations

import json
from pathlib import Path

from tests.evidence.cogp1_fixtures import cogp1_manifest
from tests.eiel_actor_v11_helpers import FIXED, run_eiel, write_safe_erm_report, write_test_qualification_register


def _mission(path: Path, manifest: Path, campaign_spec: Path) -> Path:
    raw_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "eiel_mission_specification_v2",
        "title": "Authority-integrated Codex actor workflow validation",
        "mission_type": "IMPLEMENTATION_VALIDATION",
        "question": "Can Codex complete and recover the authority-first actor workflow without reconstructing repository rules?",
        "decision_to_enable": "Close or repair the local actor-manager implementation before merge.",
        "scope": ["four local actor CLIs", "real Librarian and qualification authority", "fixed EDL fixture"],
        "out_of_scope": ["campaign execution", "Mission Control", "scientific qualification", "mechanics changes"],
        "controls": [],
        "success_criteria": ["all authority-first workflow stages complete through fresh subprocesses"],
        "failure_criteria": ["any actor invents authority, launches work, loses history, or bypasses preflight"],
        "stopping_rules": ["stop after the fixed workflow and accept the bounded result without tuning"],
        "decision_rule": "All authority, resource, qualification, EDL, recovery, and closeout gates must pass.",
        "claim_ceiling": "Implementation and regression validity of the local actor workflow only.",
        "required_evidence": ["real Librarian receipt", "real qualification resolution", "resource receipt", "fixed EDL assessment"],
        "technical_qualification_requirements": ["current b1w32p256c512gNt160cpu authority"],
        "qualification_request": {"track_label": "b1w32p256c512gNt160cpu"},
        "resource_demand": {
            "schema_version": "eiel_resource_demand_v1",
            "device": "cpu",
            "minimum_ram_available_gib": 1.0,
            "minimum_storage_available_gib": 1.0,
            "workload_class": "actor_workflow_verification",
        },
        "tasks": ["bind authority", "resolve qualification", "assess resources", "handoff external campaign", "assess evidence", "close"],
        "references": {
            "qualification_record_ids": [],
            "resource_record_ids": [],
            "evidence_assessment_ids": [],
            "campaign_spec_paths": [str(campaign_spec)],
            "campaign_output_paths": [],
            "implementation_report_refs": [],
            "librarian_document_ids": [],
        },
        "source_identity": {"source_commit": raw_manifest["git_commit"]},
        "required_closeout_refs": [
            "evidence_refs",
            "qualification_refs",
            "resource_refs",
            "report_refs",
            "campaign_output_refs",
            "source_identity",
            "librarian_document_ids",
        ],
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_real_authority_first_codex_workflow_and_recovery(tmp_path: Path) -> None:
    state = tmp_path / "state"
    authority_register = write_test_qualification_register(tmp_path / "qualification-authority.json")
    manifest = cogp1_manifest(tmp_path / "campaign", signal="supported")
    campaign_spec = tmp_path / "campaign-spec.json"
    campaign_spec.write_text('{"schema_version":"external_campaign_spec_reference_v1"}\n', encoding="utf-8")
    mission_file = _mission(tmp_path / "mission.json", manifest, campaign_spec)
    report = write_safe_erm_report(tmp_path / "erm.json")

    created = run_eiel(
        ["missionPlanner", "create", "--input", str(mission_file), "--ready", "--generated-at", FIXED],
        state_root=state,
    )
    mission_id = created["mission"]["mission_id"]
    assert created["status"] == "BLOCKED"
    assert created["mission"]["status"] == "DRAFT"

    preflight = run_eiel(
        [
            "missionPlanner",
            "preflight",
            mission_id,
            "--topic",
            "qualification authority register",
            "--collection",
            "all",
            "--authority-domain",
            "execution_route",
            "--authorization-basis",
            "operator_instruction",
            "--operator-instruction-ref",
            "explicit V1.2 actor hardening and maximum-verification instruction",
            "--ready",
            "--generated-at",
            "2026-08-03T12:00:01Z",
        ],
        state_root=state,
    )
    assert preflight["status"] == "READY"
    assert preflight["authority_context"]["preflight"]["authority_domains"] == ["execution_route"]
    assert preflight["authority_context"]["preflight"]["fail_closed"] is False
    assert preflight["evidence_preflight_markdown"].startswith("## Evidence Preflight\n")

    # Fresh process recovery uses retained, hash-verified authority rather than chat context.
    authority = run_eiel(["missionPlanner", "authority", mission_id], state_root=state)
    assert authority["status"] == "AUTHORIZED"
    assert authority["authority_validation"]["valid"] is True
    next_step = run_eiel(["missionPlanner", "next", mission_id], state_root=state)
    assert next_step["workflow"]["missing_items"] == ["qualification reference"]

    qualified = run_eiel(
        [
            "qualificationManager",
            "check",
            "--mission",
            mission_id,
            "--store",
            "--authority-register",
            str(authority_register),
            "--generated-at",
            "2026-08-03T12:00:02Z",
        ],
        state_root=state,
    )
    assert qualified["decision"] == "QUALIFIED"
    qualification_id = qualified["check_id"]
    run_eiel(
        ["missionPlanner", "update", mission_id, "--qualification-ref", qualification_id, "--generated-at", "2026-08-03T12:00:03Z"],
        state_root=state,
    )

    resource = run_eiel(
        [
            "resourceManager",
            "assess",
            "--mission",
            mission_id,
            "--report",
            str(report),
            "--generated-at",
            "2026-08-03T12:00:04Z",
        ],
        state_root=state,
    )
    assert resource["assessment"]["decision"] == "ADMIT"
    resource_id = resource["assessment_id"]
    active = run_eiel(
        [
            "missionPlanner",
            "update",
            mission_id,
            "--resource-ref",
            resource_id,
            "--status",
            "ACTIVE",
            "--generated-at",
            "2026-08-03T12:00:05Z",
        ],
        state_root=state,
    )
    assert active["mission"]["status"] == "ACTIVE"
    assert active["mission"]["next_actions"][0]["action"] == "execute_externally"
    assert "tools.scripts.eiel_campaign" in active["mission"]["next_actions"][0]["command"]

    # Campaign execution remains external. Once Codex records the resulting output,
    # the mission advances to evidence inspection without the planner pretending it ran anything.
    output_recorded = run_eiel(
        [
            "missionPlanner",
            "update",
            mission_id,
            "--campaign-output-ref",
            str(manifest.parent),
            "--generated-at",
            "2026-08-03T12:00:05.500000Z",
        ],
        state_root=state,
    )
    assert output_recorded["mission"]["next_actions"][0]["action"] == "inspect_evidence"

    report_path = tmp_path / "implementation-report.md"
    report_path.write_text(
        preflight["evidence_preflight_markdown"]
        + "\n# Actor workflow validation\n\nThis is bounded implementation evidence only.\n",
        encoding="utf-8",
    )
    run_eiel(
        [
            "missionPlanner",
            "update",
            mission_id,
            "--report-ref",
            str(report_path),
            "--source-commit",
            json.loads(manifest.read_text(encoding="utf-8"))["git_commit"],
            "--generated-at",
            "2026-08-03T12:00:06Z",
        ],
        state_root=state,
    )

    evidence = run_eiel(
        [
            "evidenceManager",
            "assess",
            str(manifest),
            "--mission",
            mission_id,
            "--out-dir",
            str(tmp_path / "decision"),
            "--deterministic",
            "--generated-at",
            "2026-08-03T12:00:07Z",
        ],
        state_root=state,
    )
    evidence_id = evidence["assessment_id"]
    assert evidence["assessment"]["procedural_decision"] == "A"
    assert evidence["assessment"]["scientific_qualification_effect"] == "none"
    assert evidence["assessment"]["authority_context"]["preflight"]["receipt_sha256"]

    run_eiel(
        [
            "missionPlanner",
            "update",
            mission_id,
            "--evidence-ref",
            evidence_id,
            "--status",
            "EVIDENCE_PENDING",
            "--generated-at",
            "2026-08-03T12:00:08Z",
        ],
        state_root=state,
    )
    closed = run_eiel(
        [
            "missionPlanner",
            "closeout",
            mission_id,
            "--outcome",
            "completed",
            "--decision",
            "authority-first local actor workflow verified",
            "--no-claims-statement",
            "No scientific, intelligence, emergence, or general cognition claim is made.",
            "--evidence-ref",
            evidence_id,
            "--qualification-ref",
            qualification_id,
            "--resource-ref",
            resource_id,
            "--report-ref",
            str(report_path),
            "--campaign-output-ref",
            str(manifest.parent),
            "--source-commit",
            json.loads(manifest.read_text(encoding="utf-8"))["git_commit"],
            "--librarian-document-id",
            "EIEL-DOC-000005",
            "--strict-references",
            "--generated-at",
            "2026-08-03T12:00:09Z",
        ],
        state_root=state,
    )
    assert closed["mission"]["status"] == "CLOSED"

    final = run_eiel(["missionPlanner", "handoff", mission_id], state_root=state)
    history = run_eiel(["missionPlanner", "history", mission_id], state_root=state)
    assert final["handoff"]["missing_items"] == []
    assert final["handoff"]["evidence_preflight_markdown"].startswith("## Evidence Preflight\n")
    assert history["revision_count"] >= 7
