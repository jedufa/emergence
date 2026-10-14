from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence

from tests.evidence.cogp1_fixtures import cogp1_manifest
from tests.eiel_actor_v11_helpers import authorize_mission, write_test_qualification_register
from tools import resource_monitor as erm

ROOT = Path(__file__).resolve().parents[1]
EIEL = ROOT / "tools" / "bin" / "eiel"
FIXED = "2026-08-03T12:00:00Z"
GIB = 1024**3


def _run(args: Sequence[str], *, expected: int | set[int] = 0) -> dict[str, Any]:
    proc = subprocess.run(
        [str(EIEL), *args],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        text=True,
        capture_output=True,
        check=False,
    )
    allowed = {expected} if isinstance(expected, int) else set(expected)
    assert proc.returncode in allowed, (proc.args, proc.returncode, proc.stdout, proc.stderr)
    assert proc.stdout.strip(), (proc.args, proc.stderr)
    payload = json.loads(proc.stdout)
    for field in ("status", "reason_codes", "problems", "warnings", "next_actions", "related_records"):
        assert field in payload, (field, payload)
    return payload


def _write_erm_report(path: Path, *, observed_at: str = FIXED) -> Path:
    report = {
        "schema_version": erm.SCHEMA_VERSION,
        "generated_at_utc": observed_at,
        "mode": "read_only_stateless_fresh_inspection",
        "non_goals": ["no_scheduler", "no_reservation", "no_launch", "no_process_kill"],
        "cpu": {"logical_threads": 8, "utilization_percent": 10.0},
        "memory": {"ram_available_bytes": 12 * GIB},
        "gpu": {"available": False, "gpus": [], "processes": []},
        "storage": {"roots": [{"path": str(path.parent), "free_bytes": 100 * GIB}]},
        "running_eiel_workloads": [],
        "slot_heartbeats": [],
        "campaign_manifests": [],
        "ste": {"available": False},
        "blockers": [],
        "recommendation": {"value": "SAFE", "confidence": "HIGH", "reason": "workflow fixture"},
        "confidence": "HIGH",
        "blocker_classification": [],
        "recommended_poll_interval_seconds": 30,
        "next_check_after_seconds": 30,
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _write_mission(path: Path, manifest: Path) -> Path:
    raw_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    spec = {
        "title": "Codex-realistic COG-P1 actor verification",
        "mission_type": "SCIENTIFIC_PROBE",
        "question": "Does the fixed perception fixture satisfy the predeclared COG-P1 decision rule?",
        "decision_to_enable": "Record the bounded procedural COG-P1 result and close the verification mission.",
        "scope": ["one fixed COG-P1 evidence fixture", "actor CLI workflow verification"],
        "out_of_scope": ["new mechanics", "campaign tuning", "general cognition claims"],
        "controls": ["perception normal", "perception ablated"],
        "success_criteria": ["implemented COG-P1 profile returns its predeclared result"],
        "failure_criteria": ["schema, pairing, reward, qualification, or source binding blocks assessment"],
        "stopping_rules": ["stop after the fixed actor workflow and recovery checks"],
        "decision_rule": "Use the implemented COG-P1 profile without post-hoc widening.",
        "claim_ceiling": "Procedural COG-P1 fixture result and actor usability only.",
        "required_evidence": ["paired outputs", "reward-leakage audit", "technical integrity record"],
        "technical_qualification_requirements": ["current admitted CPU route"],
        "resource_demand": {"device": "cpu", "minimum_ram_available_gib": 1.0},
        "qualification_request": {"track_label": "b1w32p256c512gNt160cpu"},
        "tasks": ["create mission", "check route", "assess resources", "assess evidence", "close mission"],
        "source_identity": {"source_commit": raw_manifest["git_commit"]},
        "required_closeout_refs": [
            "evidence_refs", "qualification_refs", "resource_refs", "report_refs",
            "campaign_output_refs", "source_identity", "librarian_document_ids",
        ],
    }
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def test_codex_subprocess_workflow_and_interruption_recovery(tmp_path: Path) -> None:
    state = tmp_path / "actor-state"
    authority_register = write_test_qualification_register(tmp_path / "qualification-authority.json")
    manifest = cogp1_manifest(tmp_path / "campaign", signal="supported")
    mission_file = _write_mission(tmp_path / "mission.json", manifest)
    erm_report = _write_erm_report(tmp_path / "erm.json")
    implementation_report = tmp_path / "implementation-report.md"
    implementation_report.write_text("## Evidence Preflight\n\n- Authority consulted: eiel-auth-test-campaign-decision\n- Evidence IDs: EIEL-EV-000001\n- Document IDs: EIEL-DOC-000005\n- Claims consulted: eiel-claim-test-bounded\n- Limitations reviewed: eiel-limitation-test-bounded\n- Unknown authority: None\n- Catalog build ID: test-bound-by-mission\n- Maintained source digest: test-bound-by-mission\n\n# Codex actor verification\n", encoding="utf-8")

    created = _run([
        "missionPlanner", "create", "--input", str(mission_file), "--ready",
        "--state-root", str(state), "--generated-at", FIXED,
    ])
    mission_id = created["mission"]["mission_id"]
    assert created["mission"]["status"] == "DRAFT"
    assert created["next_actions"][0]["action"] == "run_evidence_preflight"
    authorize_mission(
        mission_id, state_root=state, ready=True,
        generated_at="2026-08-03T12:00:30Z",
    )

    # Fresh subprocess recovery after mission creation and retained authority binding.
    recovered = _run(["missionPlanner", "next", mission_id, "--state-root", str(state)])
    assert recovered["workflow"]["completed_stages"] == [
        "mission_created", "evidence_preflight_bound", "authority_bound", "mission_validated"
    ]
    assert recovered["workflow"]["missing_items"] == ["qualification reference"]

    qualified = _run([
        "qualificationManager", "check", "--mission", mission_id, "--store",
        "--authority-register", str(authority_register), "--state-root", str(state), "--generated-at", "2026-08-03T12:01:00Z",
    ])
    qualification_id = qualified["check_id"]
    assert qualified["decision"] == "QUALIFIED"
    assert qualified["next_actions"][0]["command"].endswith(f"--qualification-ref {qualification_id}")

    _run([
        "missionPlanner", "update", mission_id, "--qualification-ref", qualification_id,
        "--state-root", str(state), "--generated-at", "2026-08-03T12:02:00Z",
    ])
    resources = _run([
        "resourceManager", "assess", "--mission", mission_id, "--report", str(erm_report),
        "--state-root", str(state), "--generated-at", "2026-08-03T12:03:00Z",
    ])
    resource_id = resources["assessment_id"]
    assert resources["assessment"]["decision"] == "ADMIT"

    active = _run([
        "missionPlanner", "update", mission_id, "--resource-ref", resource_id, "--status", "ACTIVE",
        "--state-root", str(state), "--generated-at", "2026-08-03T12:04:00Z",
    ])
    assert active["mission"]["status"] == "ACTIVE"

    # A new process can reconstruct state without prior in-memory context.
    status = _run(["missionPlanner", "status", mission_id, "--state-root", str(state)])
    assert {"qualification_checked", "resources_assessed", "execution_started"}.issubset(
        set(status["mission"]["completed_stages"])
    )
    assert status["mission"]["next_actions"][0]["action"] == "execute_externally"
    history = _run(["missionPlanner", "history", mission_id, "--state-root", str(state)])
    assert history["revision_count"] == 4

    _run([
        "missionPlanner", "update", mission_id,
        "--campaign-output-ref", str(manifest.parent),
        "--report-ref", str(implementation_report),
        "--state-root", str(state), "--generated-at", "2026-08-03T12:05:00Z",
    ])
    assessed = _run([
        "evidenceManager", "assess", str(manifest), "--mission", mission_id,
        "--state-root", str(state), "--out-dir", str(tmp_path / "evidence-report"),
        "--deterministic", "--generated-at", "2026-08-03T12:06:00Z",
    ])
    evidence_id = assessed["assessment_id"]
    assert assessed["assessment"]["procedural_decision"] == "A"
    assert qualification_id in assessed["assessment"]["technical_qualification_refs"]
    assert assessed["next_actions"][0]["command"].endswith(f"--evidence-ref {evidence_id}")

    _run([
        "missionPlanner", "update", mission_id, "--evidence-ref", evidence_id,
        "--status", "EVIDENCE_PENDING", "--state-root", str(state),
        "--generated-at", "2026-08-03T12:07:00Z",
    ])
    next_before_close = _run(["missionPlanner", "next", mission_id, "--state-root", str(state)])
    assert next_before_close["workflow"]["next_actions"][0]["action"] == "closeout"

    closed = _run([
        "missionPlanner", "closeout", mission_id,
        "--outcome", "completed",
        "--decision", "bounded procedural result recorded",
        "--no-claims-statement", "No intelligence, emergence, or general cognition claim is made.",
        "--evidence-ref", evidence_id,
        "--qualification-ref", qualification_id,
        "--resource-ref", resource_id,
        "--report-ref", str(implementation_report),
        "--campaign-output-ref", str(manifest.parent),
        "--source-commit", json.loads(manifest.read_text(encoding="utf-8"))["git_commit"],
        "--librarian-document-id", "EIEL-DOC-000005",
        "--strict-references",
        "--state-root", str(state), "--generated-at", "2026-08-03T12:08:00Z",
    ])
    assert closed["mission"]["status"] == "CLOSED"

    final_handoff = _run(["missionPlanner", "handoff", mission_id, "--state-root", str(state)])
    assert final_handoff["handoff"]["status"] == "CLOSED"
    assert final_handoff["handoff"]["missing_items"] == []
    assert final_handoff["handoff"]["next_actions"][0]["action"] == "none"

    # Deterministic export proves the retained assessment can be transferred twice identically.
    export_a = tmp_path / "assessment-a.zip"
    export_b = tmp_path / "assessment-b.zip"
    first_export = _run([
        "evidenceManager", "export", evidence_id, "--export-output", str(export_a),
        "--state-root", str(state),
    ])
    second_export = _run([
        "evidenceManager", "export", evidence_id, "--export-output", str(export_b),
        "--state-root", str(state),
    ])
    assert first_export["export_sha256"] == second_export["export_sha256"]
    assert export_a.read_bytes() == export_b.read_bytes()
