from __future__ import annotations

import json
from pathlib import Path

from tests.eiel_actor_v11_helpers import FIXED, authorize_mission, run_eiel, scientific_mission_spec, write_safe_erm_report, write_test_qualification_register


def test_fresh_codex_process_recovers_exact_next_action(tmp_path: Path) -> None:
    state = tmp_path / "state"
    authority_register = write_test_qualification_register(tmp_path / "qualification-authority.json")
    source = tmp_path / "mission.json"
    source.write_text(json.dumps(scientific_mission_spec()), encoding="utf-8")
    created = run_eiel(
        ["missionPlanner", "create", "--input", str(source), "--ready", "--generated-at", FIXED],
        state_root=state,
    )
    mission_id = created["mission"]["mission_id"]
    assert created["mission"]["status"] == "DRAFT"
    authorize_mission(
        mission_id, state_root=state, ready=True, generated_at="2026-08-03T12:00:30Z"
    )

    qualified = run_eiel(
        ["qualificationManager", "check", "--mission", mission_id, "--store", "--authority-register", str(authority_register), "--generated-at", "2026-08-03T12:01:00Z"],
        state_root=state,
    )
    qcheck_id = qualified["check_id"]
    run_eiel(
        ["missionPlanner", "update", mission_id, "--qualification-ref", qcheck_id, "--generated-at", "2026-08-03T12:02:00Z"],
        state_root=state,
    )

    # A separate process sees exactly what remains instead of rereading project history.
    status = run_eiel(["missionPlanner", "status", mission_id], state_root=state)
    assert status["mission"]["completed_stages"] == ["mission_created", "evidence_preflight_bound", "authority_bound", "mission_validated", "qualification_checked"]
    assert "resource assessment reference" in status["mission"]["missing_items"]
    assert status["next_actions"][0]["command"].startswith("python -m tools.eiel resourceManager assess")

    report = write_safe_erm_report(tmp_path / "report.json")
    resource = run_eiel(
        ["resourceManager", "assess", "--mission", mission_id, "--report", str(report), "--generated-at", "2026-08-03T12:03:00Z"],
        state_root=state,
    )
    run_eiel(
        [
            "missionPlanner", "update", mission_id, "--resource-ref", resource["assessment_id"],
            "--status", "ACTIVE", "--generated-at", "2026-08-03T12:04:00Z",
        ],
        state_root=state,
    )

    recovered = run_eiel(["missionPlanner", "handoff", mission_id], state_root=state)
    assert recovered["handoff"]["status"] == "ACTIVE"
    assert recovered["handoff"]["next_actions"][0]["action"] == "execute_externally"
    assert qcheck_id in recovered["handoff"]["references"]["qualification_record_ids"]
    assert resource["assessment_id"] in recovered["handoff"]["references"]["resource_record_ids"]
    history = run_eiel(["missionPlanner", "history", mission_id], state_root=state)
    assert history["revision_count"] == 4
