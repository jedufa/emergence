from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence

from tools import resource_monitor as erm
from tests.eiel_actor_v11_helpers import authorize_mission

ROOT = Path(__file__).resolve().parents[1]
EIEL = ROOT / "tools" / "bin" / "eiel"
GIB = 1024**3


def _run(args: Sequence[str], expected: int | set[int]) -> dict[str, Any]:
    proc = subprocess.run(
        [str(EIEL), *args], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)},
        text=True, capture_output=True, check=False,
    )
    allowed = {expected} if isinstance(expected, int) else expected
    assert proc.returncode in allowed, (proc.args, proc.returncode, proc.stdout, proc.stderr)
    payload = json.loads(proc.stdout)
    assert payload["status"] in {"BLOCKED", "DRAFT", "MANUAL_REVIEW", "WAIT", "NOT_QUALIFIED", "FULL_QUALIFICATION_REQUIRED"} or payload["ok"] is False
    assert payload["reason_codes"]
    assert payload["problems"]
    return payload


def _valid_mission() -> dict[str, Any]:
    return {
        "title": "Bad-Codex guard test",
        "mission_type": "SCIENTIFIC_PROBE",
        "question": "Does the guard reject unsafe progression?",
        "decision_to_enable": "Record the guard result.",
        "scope": ["actor guard"],
        "out_of_scope": ["campaign execution"],
        "controls": ["normal", "ablated"],
        "success_criteria": ["unsafe action rejected"],
        "failure_criteria": ["unsafe action accepted"],
        "stopping_rules": ["stop after rejection"],
        "decision_rule": "Reject missing authority.",
        "claim_ceiling": "Actor guard only.",
        "required_evidence": ["reward-leakage audit"],
        "technical_qualification_requirements": ["current route"],
        "resource_demand": {"device": "cpu"},
        "qualification_request": {"track_label": "b1w32p256c512gNt160cpu"},
        "tasks": ["verify rejection"],
    }


def test_forbidden_actor_operations_are_structured() -> None:
    for args in (
        ["missionPlanner", "run"],
        ["qualificationManager", "execute"],
        ["resourceManager", "reserve"],
        ["evidenceManager", "tune"],
    ):
        result = _run(args, expected=2)
        assert "forbidden_actor_operation" in result["reason_codes"]
        assert result["next_actions"]


def test_mission_rejects_missing_fields_executable_content_and_unsafe_activation(tmp_path: Path) -> None:
    state = tmp_path / "state"
    invalid = _valid_mission()
    invalid["stopping_rules"] = []
    invalid["commands"] = ["python run.py"]
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    validation = _run([
        "missionPlanner", "validate", str(invalid_path), "--state-root", str(state),
    ], expected=1)
    assert any("stopping_rules" in problem for problem in validation["problems"])
    assert any("executable field" in problem for problem in validation["problems"])

    mission_path = tmp_path / "mission.json"
    mission_path.write_text(json.dumps(_valid_mission()), encoding="utf-8")
    created = subprocess.run(
        [str(EIEL), "missionPlanner", "create", "--input", str(mission_path), "--ready", "--state-root", str(state)],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, text=True, capture_output=True, check=False,
    )
    assert created.returncode == 0
    mission_id = json.loads(created.stdout)["mission"]["mission_id"]
    authorize_mission(mission_id, state_root=state, ready=True)
    activated = _run([
        "missionPlanner", "update", mission_id, "--status", "ACTIVE", "--state-root", str(state),
    ], expected=2)
    assert "mission_state_prerequisites_missing" in activated["reason_codes"]
    assert any("qualification" in problem.lower() for problem in activated["problems"])


def test_resource_stale_unknown_and_destructive_requests_fail_closed(tmp_path: Path) -> None:
    state = tmp_path / "state"
    stale = {
        "schema_version": erm.SCHEMA_VERSION,
        "generated_at_utc": "2026-08-03T10:00:00Z",
        "mode": "read_only_stateless_fresh_inspection",
        "non_goals": ["no_scheduler"],
        "cpu": {"logical_threads": 8},
        "memory": {"ram_available_bytes": 8 * GIB},
        "gpu": {"available": False, "gpus": [], "processes": []},
        "storage": {"roots": [{"free_bytes": 20 * GIB}]},
        "running_eiel_workloads": [], "slot_heartbeats": [], "campaign_manifests": [],
        "ste": {"available": False}, "blockers": [],
        "recommendation": {"value": "SAFE", "confidence": "HIGH"},
        "confidence": "HIGH", "blocker_classification": [],
        "recommended_poll_interval_seconds": 30, "next_check_after_seconds": 30,
    }
    report = tmp_path / "stale.json"
    report.write_text(json.dumps(stale), encoding="utf-8")
    result = _run([
        "resourceManager", "assess", "--device", "cpu", "--report", str(report),
        "--state-root", str(state), "--generated-at", "2026-08-03T12:00:00Z",
    ], expected=1)
    assert result["assessment"]["decision"] == "MANUAL_REVIEW"
    assert "stale_resource_observation" in result["reason_codes"]

    missing = _run(["resourceManager", "assess", "--state-root", str(state)], expected=2)
    assert "missing_resource_demand" in missing["reason_codes"]


def test_qualification_rejects_bare_prose_decision(tmp_path: Path) -> None:
    evidence = tmp_path / "prose-like.json"
    evidence.write_text(json.dumps({"decision": "A", "note": "looks good"}), encoding="utf-8")
    result = _run([
        "qualificationManager", "decide", "--evidence", str(evidence),
        "--state-root", str(tmp_path / "state"),
    ], expected=1)
    assert result["qualification_decision"]["accepted_for_envelope_expansion"] is False
    assert any("structured qualification evidence bindings are missing" in p for p in result["problems"])
