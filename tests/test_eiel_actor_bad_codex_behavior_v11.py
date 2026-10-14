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


def run_cli(*args: str, expected: int) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(
        [sys.executable, "-m", "tools.eiel", *args], cwd=ROOT, env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert proc.returncode == expected, f"{args}\nstdout={proc.stdout}\nstderr={proc.stderr}"
    assert "Traceback" not in proc.stdout + proc.stderr
    return json.loads(proc.stdout)


def valid_mission() -> dict[str, Any]:
    return {
        "title": "Bounded validation",
        "mission_type": "SCIENTIFIC_PROBE",
        "question": "Does the bounded condition differ from control?",
        "decision_to_enable": "Close or retain one bounded claim.",
        "scope": ["fixed matrix"],
        "out_of_scope": ["tuning"],
        "controls": ["normal", "ablated"],
        "success_criteria": ["predeclared rule passes"],
        "failure_criteria": ["predeclared rule fails"],
        "stopping_rules": ["stop after fixed matrix"],
        "decision_rule": "Apply profile V1.",
        "claim_ceiling": "Bounded result only.",
        "required_evidence": ["paired outputs", "reward-leakage audit"],
        "technical_qualification_requirements": ["current route"],
        "qualification_request": {"track_label": "b1w32p256c512gNt160cpu"},
        "resource_demand": {"device": "cpu"},
        "tasks": ["validate"],
    }


def erm_report(*, generated_at: str, gpu_available: bool = True) -> dict[str, Any]:
    return {
        "schema_version": "eiel_resource_monitor_report_v1",
        "generated_at_utc": generated_at,
        "mode": "read_only_stateless_fresh_inspection",
        "non_goals": ["no_scheduler"],
        "cpu": {"logical_threads": 8, "utilization_percent": 10.0},
        "memory": {"ram_available_bytes": 8 * GIB},
        "gpu": {"available": gpu_available, "gpus": [], "processes": []},
        "storage": {"roots": [{"path": "/tmp", "free_bytes": 20 * GIB}]},
        "running_eiel_workloads": [], "slot_heartbeats": [], "campaign_manifests": [],
        "ste": {"available": False}, "blockers": [] if gpu_available else [{"class": "UNKNOWN", "kind": "gpu_telemetry_unavailable"}],
        "recommendation": {"value": "UNKNOWN" if not gpu_available else "SAFE", "confidence": "LOW" if not gpu_available else "HIGH", "reason": "fixture"},
        "confidence": "LOW" if not gpu_available else "HIGH", "blocker_classification": ["UNKNOWN"] if not gpu_available else [],
        "recommended_poll_interval_seconds": 30, "next_check_after_seconds": 30,
    }


def test_mission_planner_blocks_execution_missing_rules_invalid_transition_and_incomplete_closeout(tmp_path: Path) -> None:
    forbidden = run_cli("missionPlanner", "run", expected=2)
    assert forbidden["reason_codes"] == ["forbidden_actor_operation"]

    bad = valid_mission()
    bad["stopping_rules"] = []
    bad["nested"] = {"command": "python run.py"}
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    validation = run_cli("missionPlanner", "validate", str(bad_path), expected=1)
    assert "mission_field_missing:stopping_rules" in validation["reason_codes"]
    assert any("executable field" in problem for problem in validation["problems"])
    assert validation["next_actions"]

    state = tmp_path / "state"
    authority_register = write_test_qualification_register(tmp_path / "qualification-authority.json")
    good_path = tmp_path / "good.json"
    good_path.write_text(json.dumps(valid_mission()), encoding="utf-8")
    created = run_cli(
        "missionPlanner", "create", "--input", str(good_path),
        "--state-root", str(state), "--generated-at", FIXED, expected=0,
    )
    mission_id = created["mission"]["mission_id"]
    invalid_transition = run_cli(
        "missionPlanner", "update", mission_id, "--status", "ACTIVE",
        "--state-root", str(state), expected=2,
    )
    assert invalid_transition["reason_codes"] == ["mission_state_transition_invalid"]

    authorize_mission(
        mission_id, state_root=state, ready=True,
        generated_at="2026-08-03T12:01:00Z",
    )
    qualification = run_cli(
        "qualificationManager", "check", "--mission", mission_id, "--store",
        "--authority-register", str(authority_register), "--state-root", str(state), "--generated-at", "2026-08-03T12:01:10Z", expected=0,
    )
    report_path = tmp_path / "erm-active.json"
    report_path.write_text(json.dumps(erm_report(generated_at=FIXED)), encoding="utf-8")
    resources = run_cli(
        "resourceManager", "assess", "--mission", mission_id, "--report", str(report_path),
        "--state-root", str(state), "--generated-at", "2026-08-03T12:01:20Z", expected=0,
    )
    run_cli(
        "missionPlanner", "update", mission_id, "--status", "ACTIVE",
        "--qualification-ref", qualification["check_id"],
        "--resource-ref", resources["assessment_id"],
        "--state-root", str(state), "--generated-at", "2026-08-03T12:02:00Z", expected=0,
    )
    incomplete = run_cli(
        "missionPlanner", "closeout", mission_id,
        "--outcome", "done", "--decision", "bounded", "--no-claims-statement", "No broad claim.",
        "--state-root", str(state), expected=2,
    )
    assert incomplete["reason_codes"] == ["mission_closeout_incomplete"]
    assert any("required reference" in problem for problem in incomplete["problems"])


def test_qualification_manager_fails_closed_without_authority_or_structured_evidence(tmp_path: Path) -> None:
    authority_register = write_test_qualification_register(tmp_path / "qualification-authority.json")
    unknown = run_cli("qualificationManager", "check", "--track-label", "NOT-A-TRACK", "--authority-register", str(authority_register), expected=1)
    assert unknown["status"] == "NO_CURRENT_AUTHORITY"
    assert unknown["ok"] is False
    assert unknown["next_actions"]

    evidence = tmp_path / "prose-like.json"
    evidence.write_text(json.dumps({"decision": "A", "comment": "looks good"}), encoding="utf-8")
    rejected = run_cli(
        "qualificationManager", "decide", "--evidence", str(evidence),
        "--state-root", str(tmp_path / "state"), "--generated-at", FIXED, expected=1,
    )
    assert rejected["status"] == "NOT_QUALIFIED"
    assert any("structured qualification evidence" in problem for problem in rejected["problems"])

    forbidden = run_cli("qualificationManager", "execute", expected=2)
    assert forbidden["reason_codes"] == ["forbidden_actor_operation"]


def test_resource_manager_rejects_missing_stale_unknown_and_forbidden_operations(tmp_path: Path) -> None:
    missing = run_cli("resourceManager", "assess", expected=2)
    assert missing["reason_codes"] == ["missing_resource_demand"]

    stale_path = tmp_path / "stale.json"
    stale_path.write_text(json.dumps(erm_report(generated_at="2026-08-03T10:00:00Z")), encoding="utf-8")
    stale = run_cli(
        "resourceManager", "assess", "--device", "cpu", "--report", str(stale_path),
        "--generated-at", FIXED, "--state-root", str(tmp_path / "state-a"), expected=1,
    )
    assert stale["status"] == "MANUAL_REVIEW"
    assert "stale_resource_observation" in stale["reason_codes"]

    unknown_path = tmp_path / "unknown.json"
    unknown_path.write_text(json.dumps(erm_report(generated_at=FIXED, gpu_available=False)), encoding="utf-8")
    unknown = run_cli(
        "resourceManager", "assess", "--device", "cuda", "--report", str(unknown_path),
        "--generated-at", FIXED, "--state-root", str(tmp_path / "state-b"), expected=1,
    )
    assert unknown["status"] == "MANUAL_REVIEW"
    assert "gpu_state_unknown" in unknown["reason_codes"]

    forbidden = run_cli("resourceManager", "reserve", expected=2)
    assert forbidden["reason_codes"] == ["forbidden_actor_operation"]


def test_evidence_manager_blocks_missing_manifest_unknown_profile_bad_authority_and_source_mismatch(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    missing = run_cli("evidenceManager", "inspect", str(empty), expected=2)
    assert missing["reason_codes"] == ["campaign_evidence_manifest_missing"]
    assert missing["next_actions"][0]["command"].startswith("python -m tools.eiel evidenceManager prepare")

    manifest = cogp1_manifest(tmp_path / "campaign", signal="supported")
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["profile"] = "UNKNOWN-PROFILE"
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    unknown = run_cli("evidenceManager", "inspect", str(manifest), expected=1)
    assert unknown["status"] == "PROFILE_NOT_IMPLEMENTED"

    manifest = cogp1_manifest(tmp_path / "campaign-2", signal="supported")
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["technical_qualification_required"] = True
    raw["technical_qualification_refs"] = ["INVENTED-QUALIFICATION"]
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    invalid_authority = run_cli(
        "evidenceManager", "inspect", str(manifest),
        "--state-root", str(tmp_path / "state"), expected=1,
    )
    assert invalid_authority["status"] == "TECHNICAL_QUALIFICATION_INVALID"
    assert any("unknown technical qualification reference" in problem for problem in invalid_authority["problems"])

    source_mismatch = run_cli(
        "evidenceManager", "validate", str(manifest), "--expected-source-commit", "different",
        "--state-root", str(tmp_path / "state"), expected=1,
    )
    assert "source_identity_mismatch" in source_mismatch["reason_codes"]

    forbidden = run_cli("evidenceManager", "claim", expected=2)
    assert forbidden["reason_codes"] == ["forbidden_actor_operation"]
