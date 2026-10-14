from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from tests.evidence.cogp1_fixtures import cogp1_manifest
from tests.eiel_actor_v11_helpers import authorize_mission, write_test_qualification_register

ROOT = Path(__file__).resolve().parents[1]
FIXED = "2026-08-03T12:00:00Z"
GIB = 1024**3


def run_cli(*args: str, expected: int = 0) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(
        [sys.executable, "-m", "tools.eiel", *args],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == expected, (
        f"command returned {proc.returncode}, expected {expected}: {args}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "Traceback" not in proc.stdout + proc.stderr
    return json.loads(proc.stdout)


def write_erm_report(path: Path, *, blocked: bool = False, generated_at: str = FIXED) -> None:
    blockers = [{"class": "GPU_BLOCKER", "kind": "active_gpu_workload", "pid": 4242}] if blocked else []
    report = {
        "schema_version": "eiel_resource_monitor_report_v1",
        "generated_at_utc": generated_at,
        "mode": "read_only_stateless_fresh_inspection",
        "non_goals": ["no_scheduler", "no_reservation", "no_launch", "no_process_kill"],
        "cpu": {"logical_threads": 8, "utilization_percent": 12.0},
        "memory": {"ram_available_bytes": 12 * GIB},
        "gpu": {
            "available": True,
            "gpus": [{"index": 0, "vram_free_bytes": 5 * GIB, "vram_total_bytes": 6 * GIB}],
            "processes": [{"pid": 4242}] if blocked else [],
        },
        "storage": {"roots": [{"path": str(path.parent), "free_bytes": 100 * GIB}]},
        "running_eiel_workloads": [{"pid": 4242, "device": "cuda"}] if blocked else [],
        "slot_heartbeats": [],
        "campaign_manifests": [],
        "ste": {"available": False},
        "blockers": blockers,
        "recommendation": {
            "value": "WAIT" if blocked else "SAFE",
            "confidence": "HIGH",
            "reason": "deterministic Codex workflow fixture",
        },
        "confidence": "HIGH",
        "blocker_classification": ["GPU_BLOCKER"] if blocked else [],
        "recommended_poll_interval_seconds": 30,
        "next_check_after_seconds": 30,
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def write_mission(path: Path) -> None:
    mission = {
        "title": "Codex-realistic COG-P1 actor workflow verification",
        "mission_type": "SCIENTIFIC_PROBE",
        "question": "Does the fixed COG-P1 fixture meet its preregistered bounded criterion?",
        "decision_to_enable": "Verify the actor workflow from mission creation through bounded evidence closeout.",
        "scope": ["one existing deterministic COG-P1 fixture"],
        "out_of_scope": ["new mechanics", "campaign execution", "general cognition claims"],
        "controls": ["perception normal", "perception ablated"],
        "success_criteria": ["the implemented COG-P1 procedural A criterion is met"],
        "failure_criteria": ["the implemented COG-P1 procedural criterion is not met"],
        "stopping_rules": ["stop after one fixed end-to-end verification; do not tune"],
        "decision_rule": "Use the implemented COG-P1 EDL profile without post-hoc widening.",
        "claim_ceiling": "Procedural COG-P1 fixture decision only.",
        "required_evidence": ["paired outputs", "reward-leakage audit", "technical integrity record"],
        "technical_qualification_requirements": ["current admitted CUDA route"],
        "qualification_request": {"track_label": "b48w16p64c128gNt160gpu"},
        "resource_demand": {
            "device": "cuda",
            "minimum_ram_available_gib": 4,
            "minimum_vram_available_gib": 4,
            "minimum_storage_available_gib": 1,
            "exclusive_gpu": True,
        },
        "tasks": [
            "create and validate mission",
            "check technical authority",
            "assess resources",
            "assess evidence",
            "close mission",
        ],
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
    path.write_text(json.dumps(mission), encoding="utf-8")


def test_codex_end_to_end_workflow_and_fresh_process_recovery(tmp_path: Path) -> None:
    state = tmp_path / "state"
    authority_register = write_test_qualification_register(tmp_path / "qualification-authority.json")
    mission_path = tmp_path / "mission.json"
    report_path = tmp_path / "erm.json"
    implementation_report = tmp_path / "implementation_report.md"
    manifest = cogp1_manifest(tmp_path / "campaign", signal="supported")
    write_mission(mission_path)
    write_erm_report(report_path)
    implementation_report.write_text("## Evidence Preflight\n\n- Authority consulted: eiel-auth-test-campaign-decision\n- Evidence IDs: EIEL-EV-000001\n- Document IDs: EIEL-DOC-000005\n- Claims consulted: eiel-claim-test-bounded\n- Limitations reviewed: eiel-limitation-test-bounded\n- Unknown authority: None\n- Catalog build ID: test-bound-by-mission\n- Maintained source digest: test-bound-by-mission\n\n# Codex actor workflow verification\n", encoding="utf-8")

    created = run_cli(
        "missionPlanner", "create", "--input", str(mission_path), "--ready",
        "--state-root", str(state), "--generated-at", FIXED,
    )
    mission_id = created["mission"]["mission_id"]
    assert created["status"] == "BLOCKED"
    assert created["next_actions"][0]["action"] == "run_evidence_preflight"
    authorize_mission(
        mission_id, state_root=state, ready=True,
        generated_at="2026-08-03T12:00:00.500000Z",
    )

    # Every invocation below is a fresh subprocess. This is the intended cheap-Codex
    # recovery model, not an in-memory happy-path masquerading as one.
    recovered = run_cli("missionPlanner", "status", mission_id, "--state-root", str(state))
    assert recovered["mission"]["missing_items"] == ["qualification reference"]
    inspected = run_cli("missionPlanner", "inspect", mission_id, "--state-root", str(state))
    assert inspected["record"]["record_id"] == mission_id
    history = run_cli("missionPlanner", "history", mission_id, "--state-root", str(state))
    assert history["revision_count"] == 2

    qualification = run_cli(
        "qualificationManager", "check", "--mission", mission_id, "--store",
        "--authority-register", str(authority_register), "--state-root", str(state), "--generated-at", "2026-08-03T12:00:01Z",
    )
    qualification_id = qualification["check_id"]
    assert qualification["decision"] == "QUALIFIED"
    assert qualification["full_result"]["record_id"] == "b48w16p64c128gNt160gpu"
    assert qualification["next_actions"][0]["command"] == (
        f"python -m tools.eiel missionPlanner update {mission_id} --qualification-ref {qualification_id}"
    )
    run_cli(
        "missionPlanner", "update", mission_id, "--qualification-ref", qualification_id,
        "--state-root", str(state), "--generated-at", "2026-08-03T12:00:02Z",
    )

    resources = run_cli(
        "resourceManager", "assess", "--mission", mission_id, "--report", str(report_path),
        "--state-root", str(state), "--generated-at", "2026-08-03T12:00:03Z",
    )
    resource_id = resources["assessment_id"]
    assert resources["assessment"]["decision"] == "ADMIT"
    assert resources["next_actions"][-1]["command"] == (
        f"python -m tools.eiel missionPlanner update {mission_id} --resource-ref {resource_id}"
    )
    run_cli(
        "missionPlanner", "update", mission_id, "--resource-ref", resource_id, "--status", "ACTIVE",
        "--state-root", str(state), "--generated-at", "2026-08-03T12:00:04Z",
    )
    run_cli(
        "missionPlanner", "update", mission_id,
        "--campaign-output-ref", str(manifest.parent),
        "--report-ref", str(implementation_report),
        "--source-commit", "abc",
        "--state-root", str(state), "--generated-at", "2026-08-03T12:00:05Z",
    )

    evidence = run_cli(
        "evidenceManager", "assess", str(manifest), "--state-root", str(state),
        "--out-dir", str(tmp_path / "reports"), "--generated-at", "2026-08-03T12:00:06Z",
        "--deterministic", "--mission", mission_id,
    )
    evidence_id = evidence["assessment_id"]
    assert evidence["assessment"]["procedural_decision"] == "A"
    assert evidence["assessment"]["scientific_qualification_effect"] == "none"
    assert evidence["next_actions"][0]["command"] == (
        f"python -m tools.eiel missionPlanner update {mission_id} --evidence-ref {evidence_id}"
    )
    run_cli(
        "missionPlanner", "update", mission_id, "--evidence-ref", evidence_id,
        "--status", "EVIDENCE_PENDING", "--state-root", str(state),
        "--generated-at", "2026-08-03T12:00:07Z",
    )

    closed = run_cli(
        "missionPlanner", "closeout", mission_id,
        "--outcome", "completed",
        "--decision", "procedural fixture A",
        "--no-claims-statement", "No scientific or general cognition claim is made.",
        "--evidence-ref", evidence_id,
        "--qualification-ref", qualification_id,
        "--resource-ref", resource_id,
        "--report-ref", str(implementation_report),
        "--campaign-output-ref", str(manifest.parent),
        "--source-commit", "abc",
        "--librarian-document-id", "EIEL-DOC-000005",
        "--strict-references",
        "--state-root", str(state), "--generated-at", "2026-08-03T12:00:08Z",
    )
    assert closed["mission"]["status"] == "CLOSED"

    final_status = run_cli("missionPlanner", "status", mission_id, "--state-root", str(state))
    final_history = run_cli("missionPlanner", "history", mission_id, "--state-root", str(state))
    handoff = run_cli("missionPlanner", "handoff", mission_id, "--state-root", str(state))
    assert final_history["revision_count"] == 7
    assert final_status["mission"]["completed_stages"][-1] == "closed"
    assert handoff["handoff"]["references"]["qualification_record_ids"] == [qualification_id]
    assert handoff["handoff"]["references"]["resource_record_ids"] == [resource_id]
    assert handoff["handoff"]["references"]["evidence_assessment_ids"] == [evidence_id]
    assert handoff["next_actions"][0]["description"].startswith("Mission is closed")


def test_interrupted_codex_gets_exact_next_action_from_new_process(tmp_path: Path) -> None:
    state = tmp_path / "state"
    mission_path = tmp_path / "mission.json"
    write_mission(mission_path)
    created = run_cli(
        "missionPlanner", "create", "--input", str(mission_path), "--ready",
        "--state-root", str(state), "--generated-at", FIXED,
    )
    mission_id = created["mission"]["mission_id"]
    authorize_mission(
        mission_id, state_root=state, ready=True,
        generated_at="2026-08-03T12:00:00.500000Z",
    )

    next_one = run_cli("missionPlanner", "next", mission_id, "--state-root", str(state))
    status = run_cli("missionPlanner", "status", mission_id, "--state-root", str(state))
    history = run_cli("missionPlanner", "history", mission_id, "--state-root", str(state))
    assert next_one["workflow"]["completed_stages"] == ["mission_created", "evidence_preflight_bound", "authority_bound", "mission_validated"]
    assert next_one["workflow"]["missing_items"] == ["qualification reference"]
    assert next_one["next_actions"][0]["command"] == (
        f"python -m tools.eiel qualificationManager check --mission {mission_id} --store"
    )
    assert status["mission"]["status"] == "READY"
    assert history["revision_count"] == 2
