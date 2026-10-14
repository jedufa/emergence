from __future__ import annotations

import concurrent.futures
from pathlib import Path

from tests.eiel_actor_v11_helpers import run_eiel
from tests.eiel_actor_v14_helpers import create_work, make_bootstrap_repo, run_bootstrap, start_session
from tools.actors.common import revise_record
from tools.actors.portfolio import work_store
from tools.actors.work_manager import persist_work_record_fixture


def test_work_manager_tracks_fifo_priority_and_open_portfolio(tmp_path: Path) -> None:
    state = tmp_path / "state"
    create_work(state, work_id="WORK-FIRST", title="First", generated_at="2026-08-04T00:00:00Z", priority="P3")
    create_work(state, work_id="WORK-SECOND", title="Second", generated_at="2026-08-04T00:01:00Z", priority="P0")
    fifo = run_eiel(["workManager", "list", "--open-only", "--view", "fifo", "--state-root", str(state)], state_root=state)
    assert [row["work_id"] for row in fifo["work_items"]] == ["WORK-FIRST", "WORK-SECOND"]
    priority = run_eiel(["workManager", "list", "--open-only", "--view", "priority", "--state-root", str(state)], state_root=state)
    assert [row["work_id"] for row in priority["work_items"]] == ["WORK-SECOND", "WORK-FIRST"]


def test_unpackaged_work_defaults_to_nonblocking_record_only_completion(tmp_path: Path) -> None:
    state = tmp_path / "state"
    created = run_eiel(
        [
            "workManager", "create",
            "--work-id", "WORK-OPTIONAL",
            "--title", "Optional workflow record",
            "--summary", "Track operator-authorized work without a package.",
            "--status", "ACTIVE",
            "--state-root", str(state),
        ],
        state_root=state,
    )["work"]
    assert created["completion_policy"] == "RECORD_ONLY"

    completed = run_eiel(
        ["workManager", "complete", "WORK-OPTIONAL", "--state-root", str(state)],
        state_root=state,
    )["work"]
    assert completed["status"] == "COMPLETED"
    assert completed["completion_basis"] == "RECORD_ONLY_COMPLETION"
    assert completed["accepted"] is False

    legacy = run_eiel(
        [
            "workManager", "create",
            "--work-id", "WORK-LEGACY-FORMAL",
            "--title", "Unbound formal placeholder",
            "--summary", "Retain compatibility with an older unbound record.",
            "--completion-policy", "REPOSITORY_DELIVERY",
            "--status", "ACTIVE",
            "--state-root", str(state),
        ],
        state_root=state,
    )["work"]
    assert legacy["completion_policy"] == "REPOSITORY_DELIVERY"
    reconciled = run_eiel(
        ["workManager", "complete", "WORK-LEGACY-FORMAL", "--state-root", str(state)],
        state_root=state,
    )["work"]
    assert reconciled["completion_policy"] == "RECORD_ONLY"
    assert reconciled["accepted"] is False


def test_administrative_reconciliation_repairs_invalid_completion_policy(tmp_path: Path) -> None:
    state = tmp_path / "state"
    create_work(
        state,
        work_id="WORK-LEGACY-MALFORMED",
        title="Malformed legacy record",
        generated_at="2026-08-04T00:00:00Z",
    )
    record = work_store(state).get("WORK-LEGACY-MALFORMED")
    malformed_payload = dict(record["payload"])
    malformed_payload["completion_policy"] = "NOT_A_POLICY"
    malformed = revise_record(
        record,
        payload=malformed_payload,
        status="BACKLOG",
        action="fixture_corrupt_completion_policy",
        generated_at="2026-08-04T00:01:00Z",
    )
    persist_work_record_fixture(malformed, state_root=state, allow_update=True)

    reconciled = run_eiel(
        [
            "workManager", "reconcile", "WORK-LEGACY-MALFORMED",
            "--disposition", "OBSOLETE",
            "--reason", "Retire malformed legacy state.",
            "--acknowledge-no-acceptance",
            "--state-root", str(state),
            "--generated-at", "2026-08-04T00:02:00Z",
        ],
        state_root=state,
    )["work"]

    assert reconciled["status"] == "OBSOLETE"
    assert reconciled["completion_policy"] == "RECORD_ONLY"
    assert reconciled["administrative_reconciliation"]["payload_repairs"] == [{
        "field": "completion_policy",
        "previous_value": "NOT_A_POLICY",
        "replacement_value": "RECORD_ONLY",
    }]


def test_work_manager_dependency_chain_releases_only_after_success(tmp_path: Path) -> None:
    state = tmp_path / "state"
    create_work(state, work_id="WORK-VRAM", title="VRAM reduction", generated_at="2026-08-04T00:00:00Z", status="ACTIVE")
    create_work(state, work_id="WORK-QUAL", title="W32 qualification", generated_at="2026-08-04T00:01:00Z", dependency=["WORK-VRAM"], work_type="QUALIFICATION")
    create_work(state, work_id="WORK-CAMPAIGN", title="W32 campaign", generated_at="2026-08-04T00:02:00Z", dependency=["WORK-QUAL"], work_type="SCIENCE")
    before = run_eiel(["workManager", "graph", "WORK-CAMPAIGN", "--state-root", str(state)], state_root=state)
    assert before["work"]["WORK-CAMPAIGN"]["derived_status"] == "BLOCKED_BY_DEPENDENCY"
    run_eiel(["workManager", "complete", "WORK-VRAM", "--force", "--record-only-reason", "Fixture dependency completed.", "--state-root", str(state), "--generated-at", "2026-08-05T00:00:00Z"], state_root=state)
    qual = run_eiel(["workManager", "inspect", "WORK-QUAL", "--state-root", str(state)], state_root=state)
    assert qual["work"]["status"] == "BLOCKED"
    assert qual["work"]["derived_status"] == "READY_TO_RESUME"
    run_eiel(["workManager", "resume", "WORK-QUAL", "--state-root", str(state), "--generated-at", "2026-08-05T00:01:00Z"], state_root=state)
    run_eiel(["workManager", "complete", "WORK-QUAL", "--force", "--record-only-reason", "Fixture qualification dependency completed.", "--state-root", str(state), "--generated-at", "2026-08-06T00:00:00Z"], state_root=state)
    campaign = run_eiel(["workManager", "inspect", "WORK-CAMPAIGN", "--state-root", str(state)], state_root=state)
    assert campaign["work"]["status"] == "BLOCKED"
    assert campaign["work"]["derived_status"] == "READY_TO_RESUME"


def test_discontinued_dependency_does_not_release_blocked_work(tmp_path: Path) -> None:
    state = tmp_path / "state"
    create_work(state, work_id="WORK-BLOCKER", title="Blocker", generated_at="2026-08-04T00:00:00Z", status="ACTIVE")
    create_work(state, work_id="WORK-PARENT", title="Parent", generated_at="2026-08-04T00:01:00Z", dependency=["WORK-BLOCKER"])
    run_eiel(["workManager", "discontinue", "WORK-BLOCKER", "--force", "--state-root", str(state), "--generated-at", "2026-08-05T00:00:00Z"], state_root=state)
    parent = run_eiel(["workManager", "inspect", "WORK-PARENT", "--state-root", str(state)], state_root=state)
    assert parent["work"]["derived_status"] == "BLOCKED_BY_TERMINAL_DEPENDENCY"
    assert parent["work"]["dependencies"]["terminal_unsatisfied"] == [{"status": "DISCONTINUED", "work_id": "WORK-BLOCKER"}]


def test_operational_continuity_creates_successor_chain_without_rewriting_history(tmp_path: Path) -> None:
    state = tmp_path / "state"
    create_work(state, work_id="WORK-80-WRC-01", title="WRC 01", generated_at="2026-08-04T00:00:00Z", status="ACTIVE")
    for index in range(2, 7):
        create_work(
            state,
            work_id=f"WORK-80-WRC-0{index}",
            title=f"WRC 0{index}",
            generated_at=f"2026-08-04T00:0{index}:00Z",
            dependency=[f"WORK-80-WRC-0{index - 1}"],
        )
    for index in range(1, 7):
        run_eiel(
            ["workManager", "discontinue", f"WORK-80-WRC-0{index}", "--force", "--state-root", str(state)],
            state_root=state,
        )
    first = run_eiel(
        ["workManager", "continue", "WORK-80-WRC-06", "--reason", "Restore the WRC sequence.", "--state-root", str(state)],
        state_root=state,
    )
    assert first["continuation_status"] == "AUTO_RESOLVED"
    assert first["blocker"]["operator_intervention_required"] is False
    assert first["blocker"]["continuation_possible"] is True
    successor = first["continuation_work"]
    assert successor["status"] == "BLOCKED"
    assert successor["dependency_work_ids"]
    historical = run_eiel(["workManager", "inspect", "WORK-80-WRC-06", "--state-root", str(state)], state_root=state)
    assert historical["work"]["status"] == "DISCONTINUED"
    repeated = run_eiel(["workManager", "continue", "WORK-80-WRC-06", "--state-root", str(state)], state_root=state)
    assert repeated["created"] is False
    assert repeated["continuation_work"]["work_id"] == successor["work_id"]


def test_continuation_routes_missing_qualification_to_a_narrow_prerequisite(tmp_path: Path) -> None:
    state = tmp_path / "state"
    run_eiel(
        [
            "workManager", "create", "--work-id", "WORK-QUAL-ROUTE",
            "--title", "Qualification route", "--work-type", "QUALIFICATION",
            "--completion-policy", "QUALIFICATION_DECISION", "--status", "ACTIVE",
            "--state-root", str(state),
        ],
        state_root=state,
    )
    routed = run_eiel(
        ["workManager", "continue", "WORK-QUAL-ROUTE", "--state-root", str(state)],
        state_root=state,
    )
    assert routed["blocker"] == {
        "blocker_class": "SOFT",
        "blocker_scope": "action",
        "operator_intervention_required": False,
        "reconciliation_action": "qualification_prerequisite_routed",
        "continuation_possible": True,
    }
    assert routed["work"]["status"] == "BLOCKED"
    assert routed["prerequisite_work"]["status"] == "READY"


def test_dependency_cycle_is_rejected(tmp_path: Path) -> None:
    state = tmp_path / "state"
    create_work(state, work_id="WORK-A", title="A", generated_at="2026-08-04T00:00:00Z")
    create_work(state, work_id="WORK-B", title="B", generated_at="2026-08-04T00:01:00Z", dependency=["WORK-A"])
    result = run_eiel(
        ["workManager", "dependency", "add", "WORK-A", "WORK-B", "--state-root", str(state), "--generated-at", "2026-08-04T00:02:00Z"],
        state_root=state,
        expected=2,
    )
    assert result["reason_codes"] == ["work_validation_failed"]
    assert any("cycle" in problem for problem in result["problems"])


def test_work_history_is_append_only_across_pause_resume_and_obsolete(tmp_path: Path) -> None:
    state = tmp_path / "state"
    create_work(state, work_id="WORK-HISTORY", title="History", generated_at="2026-08-04T00:00:00Z", status="ACTIVE")
    run_eiel(["workManager", "pause", "WORK-HISTORY", "--next-action", "Resume later", "--state-root", str(state), "--generated-at", "2026-08-04T01:00:00Z"], state_root=state)
    run_eiel(["workManager", "resume", "WORK-HISTORY", "--state-root", str(state), "--generated-at", "2026-08-05T00:00:00Z"], state_root=state)
    run_eiel(["workManager", "obsolete", "WORK-HISTORY", "--force", "--state-root", str(state), "--generated-at", "2026-08-06T00:00:00Z"], state_root=state)
    history = run_eiel(["workManager", "history", "WORK-HISTORY", "--state-root", str(state)], state_root=state)
    assert [row["revision"] for row in history["history"]] == [1, 2, 3, 4]
    assert [row["status"] for row in history["history"]] == ["ACTIVE", "PAUSED", "ACTIVE", "OBSOLETE"]


def test_discontinued_work_reopens_with_restored_next_action_and_immutable_history(tmp_path: Path) -> None:
    state = tmp_path / "state"
    create_work(
        state,
        work_id="WORK-REOPEN",
        title="Reopen",
        generated_at="2026-08-04T00:00:00Z",
        status="ACTIVE",
        next_action="Run the retained validation.",
    )
    run_eiel(
        [
            "workManager", "update", "WORK-REOPEN",
            "--next-command", "pytest -q tests/test_retained.py",
            "--state-root", str(state),
            "--generated-at", "2026-08-04T00:01:00Z",
        ],
        state_root=state,
    )
    run_eiel(
        [
            "workManager", "discontinue", "WORK-REOPEN", "--force",
            "--note", "Operator stopped this attempt.",
            "--state-root", str(state),
            "--generated-at", "2026-08-04T00:02:00Z",
        ],
        state_root=state,
    )
    closed_record = run_eiel(
        ["workManager", "inspect", "WORK-REOPEN", "--state-root", str(state)],
        state_root=state,
    )["record"]
    closed_path = state / "work" / "history" / "WORK-REOPEN" / "000003.json"
    closed_bytes = closed_path.read_bytes()

    reopened = run_eiel(
        [
            "workManager", "reopen", "WORK-REOPEN",
            "--reason", "Operator wants to continue the same bounded Work.",
            "--state-root", str(state),
            "--generated-at", "2026-08-04T00:03:00Z",
        ],
        state_root=state,
    )

    assert reopened["work"]["status"] == "READY"
    reopened_record = run_eiel(
        ["workManager", "inspect", "WORK-REOPEN", "--state-root", str(state)],
        state_root=state,
    )["record"]
    assert reopened_record["previous_record_sha256"] == closed_record["record_sha256"]
    assert reopened_record["payload"]["next_action"] == "Run the retained validation."
    assert reopened_record["payload"]["next_command"] == "pytest -q tests/test_retained.py"
    assert reopened_record["payload"]["terminal_transition"] is None
    assert reopened_record["payload"]["accepted"] is False
    assert reopened_record["payload"]["completion_basis"] is None
    assert reopened_record["history"] == [{
        "action": "reopen",
        "at_utc": "2026-08-04T00:03:00Z",
        "note": "Operator wants to continue the same bounded Work.",
        "revision": 4,
        "status": "READY",
    }]
    assert closed_path.read_bytes() == closed_bytes


def test_discontinued_work_can_be_reopened_repeatedly_with_explicit_overrides(tmp_path: Path) -> None:
    state = tmp_path / "state"
    create_work(
        state,
        work_id="WORK-REOPEN-TWICE",
        title="Reopen twice",
        generated_at="2026-08-04T00:00:00Z",
        status="ACTIVE",
    )
    run_eiel(
        ["workManager", "discontinue", "WORK-REOPEN-TWICE", "--force", "--state-root", str(state), "--generated-at", "2026-08-04T00:01:00Z"],
        state_root=state,
    )
    run_eiel(
        [
            "workManager", "reopen", "WORK-REOPEN-TWICE", "--reason", "First reopen",
            "--next-action", "Use the replacement action.",
            "--next-command", "pytest -q replacement",
            "--state-root", str(state), "--generated-at", "2026-08-04T00:02:00Z",
        ],
        state_root=state,
    )
    first_record = run_eiel(
        ["workManager", "inspect", "WORK-REOPEN-TWICE", "--state-root", str(state)],
        state_root=state,
    )["record"]
    assert first_record["payload"]["next_action"] == "Use the replacement action."
    assert first_record["payload"]["next_command"] == "pytest -q replacement"
    run_eiel(
        ["workManager", "discontinue", "WORK-REOPEN-TWICE", "--force", "--state-root", str(state), "--generated-at", "2026-08-04T00:03:00Z"],
        state_root=state,
    )
    second = run_eiel(
        ["workManager", "reopen", "WORK-REOPEN-TWICE", "--reason", "Second reopen", "--state-root", str(state), "--generated-at", "2026-08-04T00:04:00Z"],
        state_root=state,
    )
    assert second["work"]["status"] == "READY"
    history = run_eiel(["workManager", "history", "WORK-REOPEN-TWICE", "--state-root", str(state)], state_root=state)
    assert [row["status"] for row in history["history"]] == ["ACTIVE", "DISCONTINUED", "READY", "DISCONTINUED", "READY"]


def test_reopen_rejects_non_discontinued_and_administratively_reconciled_work(tmp_path: Path) -> None:
    state = tmp_path / "state"
    create_work(state, work_id="WORK-STILL-ACTIVE", title="Still active", generated_at="2026-08-04T00:00:00Z", status="ACTIVE")
    active = run_eiel(
        ["workManager", "reopen", "WORK-STILL-ACTIVE", "--reason", "Invalid", "--state-root", str(state)],
        state_root=state,
        expected=2,
    )
    assert active["reason_codes"] == ["work_reopen_status_invalid"]

    create_work(state, work_id="WORK-BLANK-REASON", title="Blank reason", generated_at="2026-08-04T00:00:30Z", status="ACTIVE")
    run_eiel(
        ["workManager", "discontinue", "WORK-BLANK-REASON", "--force", "--state-root", str(state)],
        state_root=state,
    )
    blank_reason = run_eiel(
        ["workManager", "reopen", "WORK-BLANK-REASON", "--reason", "   ", "--state-root", str(state)],
        state_root=state,
        expected=2,
    )
    assert blank_reason["reason_codes"] == ["work_reopen_reason_required"]

    create_work(state, work_id="WORK-RECONCILED", title="Reconciled", generated_at="2026-08-04T00:01:00Z", status="ACTIVE")
    run_eiel(
        [
            "workManager", "reconcile", "WORK-RECONCILED",
            "--disposition", "DISCONTINUE", "--reason", "Repair corrupted lifecycle state.",
            "--acknowledge-no-acceptance", "--state-root", str(state),
            "--generated-at", "2026-08-04T00:02:00Z",
        ],
        state_root=state,
    )
    reconciled = run_eiel(
        ["workManager", "reopen", "WORK-RECONCILED", "--reason", "Invalid", "--state-root", str(state)],
        state_root=state,
        expected=2,
    )
    assert reconciled["reason_codes"] == ["work_reopen_administrative_reconciliation"]


def test_reopen_rejects_completed_obsolete_and_superseded_work(tmp_path: Path) -> None:
    state = tmp_path / "state"
    create_work(state, work_id="WORK-COMPLETED", title="Completed", generated_at="2026-08-04T00:00:00Z", status="ACTIVE")
    run_eiel(
        [
            "workManager", "complete", "WORK-COMPLETED",
            "--record-only-reason", "Fixture completed.",
            "--state-root", str(state), "--generated-at", "2026-08-04T00:01:00Z",
        ],
        state_root=state,
    )
    create_work(state, work_id="WORK-OBSOLETE", title="Obsolete", generated_at="2026-08-04T00:02:00Z", status="ACTIVE")
    run_eiel(
        ["workManager", "obsolete", "WORK-OBSOLETE", "--force", "--state-root", str(state), "--generated-at", "2026-08-04T00:03:00Z"],
        state_root=state,
    )
    create_work(state, work_id="WORK-SUPERSEDED", title="Superseded", generated_at="2026-08-04T00:04:00Z", status="ACTIVE")
    create_work(state, work_id="WORK-REPLACEMENT", title="Replacement", generated_at="2026-08-04T00:05:00Z")
    run_eiel(
        ["workManager", "supersede", "WORK-SUPERSEDED", "--with", "WORK-REPLACEMENT", "--state-root", str(state), "--generated-at", "2026-08-04T00:06:00Z"],
        state_root=state,
    )

    for work_id in ("WORK-COMPLETED", "WORK-OBSOLETE", "WORK-SUPERSEDED"):
        result = run_eiel(
            ["workManager", "reopen", work_id, "--reason", "Invalid", "--state-root", str(state)],
            state_root=state,
            expected=2,
        )
        assert result["reason_codes"] == ["work_reopen_status_invalid"]


def test_reopen_preserves_dependency_gates(tmp_path: Path) -> None:
    state = tmp_path / "state"
    create_work(state, work_id="WORK-PENDING-DEPENDENCY", title="Pending dependency", generated_at="2026-08-04T00:00:00Z", status="ACTIVE")
    create_work(
        state,
        work_id="WORK-DEPENDENT-REOPEN",
        title="Dependent reopen",
        generated_at="2026-08-04T00:01:00Z",
        dependency=["WORK-PENDING-DEPENDENCY"],
    )
    run_eiel(
        ["workManager", "discontinue", "WORK-DEPENDENT-REOPEN", "--force", "--state-root", str(state), "--generated-at", "2026-08-04T00:02:00Z"],
        state_root=state,
    )
    dependent = run_eiel(
        ["workManager", "reopen", "WORK-DEPENDENT-REOPEN", "--reason", "Invalid", "--state-root", str(state)],
        state_root=state,
        expected=2,
    )
    assert dependent["reason_codes"] == ["work_reopen_dependencies_unresolved"]


def test_work_cannot_complete_while_session_is_active(tmp_path: Path) -> None:
    # Session-specific setup is covered in session tests; this test uses a small sealed record fixture.
    from tools.actors.common import new_record
    from tools.actors.portfolio import SESSION_RECORD_TYPE
    from tools.actors.session_manager import persist_session_record_fixture

    state = tmp_path / "state"
    create_work(state, work_id="WORK-ACTIVE-SESSION", title="Active session", generated_at="2026-08-04T00:00:00Z", status="ACTIVE")
    session = new_record(
        actor="sessionManager",
        record_type=SESSION_RECORD_TYPE,
        record_id="SES-ACTIVE-SESSION",
        status="ACTIVE",
        payload={"schema_version": "eiel_codex_session_v1", "work_id": "WORK-ACTIVE-SESSION"},
        generated_at="2026-08-04T00:01:00Z",
    )
    persist_session_record_fixture(session, state_root=state)
    blocked = run_eiel(["workManager", "complete", "WORK-ACTIVE-SESSION", "--record-only-reason", "Fixture completed.", "--state-root", str(state)], state_root=state, expected=2)
    assert blocked["reason_codes"] == ["work_active_session_exists"]


def test_work_cannot_complete_while_session_is_non_terminal(tmp_path: Path) -> None:
    state = tmp_path / "state"
    repo = make_bootstrap_repo(tmp_path)
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    create_work(state, work_id="WORK-RECON", title="Non-terminal session", generated_at="2026-08-04T00:01:00Z", status="ACTIVE")
    session = start_session(
        state,
        repo=repo,
        work_id="WORK-RECON",
        bootstrap_id=bootstrap_id,
        codex_id="codex-work",
        generated_at="2026-08-04T00:02:00Z",
    )
    session_id = session["session"]["session_id"]
    run_eiel(
        ["sessionManager", "interrupt", session_id, "--interrupted-during", "manual review", "--next-action", "resume after reconciliation", "--state-root", str(state), "--generated-at", "2026-08-04T00:03:00Z"],
        state_root=state,
    )
    blocked = run_eiel(["workManager", "discontinue", "WORK-RECON", "--state-root", str(state), "--generated-at", "2026-08-04T01:00:00Z"], state_root=state, expected=2)
    assert blocked["reason_codes"] == ["work_active_session_exists"]


def test_concurrent_idempotent_work_creation_does_not_corrupt_store(tmp_path: Path) -> None:
    state = tmp_path / "state"
    command = [
        "workManager", "create", "--work-id", "WORK-CONCURRENT", "--title", "Concurrent", "--summary", "Concurrent",
        "--work-type", "ENGINEERING", "--priority", "P2", "--status", "BACKLOG", "--next-action", "Continue",
        "--state-root", str(state), "--generated-at", "2026-08-04T00:00:00Z",
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: run_eiel(command, state_root=state), range(4)))
    assert all(result["work"]["work_id"] == "WORK-CONCURRENT" for result in results)
    listed = run_eiel(["workManager", "list", "--state-root", str(state)], state_root=state)
    assert listed["count"] == 1
    history = run_eiel(["workManager", "history", "WORK-CONCURRENT", "--state-root", str(state)], state_root=state)
    assert history["count"] == 1


def test_work_supersession_preserves_old_and_new_identity(tmp_path: Path) -> None:
    state = tmp_path / "state"
    create_work(state, work_id="WORK-OLD", title="Old", generated_at="2026-08-04T00:00:00Z")
    create_work(state, work_id="WORK-NEW", title="New", generated_at="2026-08-04T00:01:00Z")
    result = run_eiel(["workManager", "supersede", "WORK-OLD", "--with", "WORK-NEW", "--state-root", str(state), "--generated-at", "2026-08-04T00:02:00Z"], state_root=state)
    assert result["work"]["status"] == "SUPERSEDED"
    old = run_eiel(["workManager", "inspect", "WORK-OLD", "--state-root", str(state)], state_root=state)
    assert old["record"]["payload"]["superseded_by_work_id"] == "WORK-NEW"
    new = run_eiel(["workManager", "inspect", "WORK-NEW", "--state-root", str(state)], state_root=state)
    assert new["work"]["status"] == "BACKLOG"
