from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tests.eiel_actor_v11_helpers import run_eiel
from tests.eiel_actor_v14_helpers import create_work, make_bootstrap_repo, run_bootstrap, start_session


def test_workspace_recovers_the_only_resumable_session(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    create_work(state, work_id="WORK-ONLY", title="Only work", generated_at="2026-08-04T00:01:00Z", status="ACTIVE")
    session = start_session(state, repo=repo, work_id="WORK-ONLY", bootstrap_id=bootstrap_id, codex_id="codex-1", generated_at="2026-08-04T00:02:00Z")
    session_id = session["session"]["session_id"]
    run_eiel(["sessionManager", "interrupt", session_id, "--interrupted-during", "test run", "--next-action", "Rerun the failed test.", "--next-command", "pytest -q tests/test_x.py", "--state-root", str(state), "--generated-at", "2026-08-04T00:03:00Z"], state_root=state)
    workspace = run_eiel(["workspace", "open", "--bootstrap-record", bootstrap_id, "--state-root", str(state)], state_root=state)
    assert workspace["status"] == "SESSION_RESUME_REQUIRED"
    assert workspace["workspace"]["selected_session"]["record_id"] == session_id
    receipt = workspace["workspace"]["context_receipt"]
    assert receipt["currentness_claim"] == "CONTEXT_SNAPSHOT_WITHOUT_DECLARED_SCOPE"
    assert receipt["work"]["package_digest"] is None
    assert workspace["workspace"]["context_receipt_validation"]["valid"] is True
    assert "sessionManager resume" in workspace["next_actions"][0]["command"]
    assert "pytest -q tests/test_x.py" not in workspace["next_actions"][0]["command"]


def test_workspace_exposes_multiple_sessions_without_blocking_authorized_work(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    create_work(state, work_id="WORK-A", title="A", generated_at="2026-08-04T00:01:00Z", status="ACTIVE")
    create_work(state, work_id="WORK-B", title="B", generated_at="2026-08-04T00:02:00Z", status="ACTIVE")
    start_session(state, repo=repo, work_id="WORK-A", bootstrap_id=bootstrap_id, codex_id="codex-a", generated_at="2026-08-04T00:03:00Z")
    start_session(state, repo=repo, work_id="WORK-B", bootstrap_id=bootstrap_id, codex_id="codex-b", generated_at="2026-08-04T00:04:00Z")
    result = run_eiel(["workspace", "open", "--bootstrap-record", bootstrap_id, "--state-root", str(state)], state_root=state)
    assert result["status"] == "WORKSPACE_READY"
    assert result["execution_blocked"] is False
    assert result["workflow_administration_required"] is True
    assert result["problems"] == []
    assert result["workspace"]["portfolio"]["resumable_session_count"] == 2
    selected = run_eiel(["workspace", "open", "--bootstrap-record", bootstrap_id, "--work-id", "WORK-B", "--state-root", str(state)], state_root=state)
    assert selected["workspace"]["selected_work"]["record_id"] == "WORK-B"
    assert selected["workspace"]["selected_session"]["payload"]["current_codex_id"] == "codex-b"


def test_workspace_nested_w32_qualification_vram_scenario(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]

    create_work(state, work_id="WORK-VRAM", title="VRAM reduction", generated_at="2026-08-04T00:01:00Z", status="ACTIVE", priority="P0")
    create_work(state, work_id="WORK-W32-QUAL", title="W32 qualification", generated_at="2026-08-04T00:02:00Z", dependency=["WORK-VRAM"], priority="P1", work_type="QUALIFICATION")
    create_work(state, work_id="WORK-W32-CAMPAIGN", title="W32 campaign", generated_at="2026-08-04T00:03:00Z", dependency=["WORK-W32-QUAL"], priority="P1", work_type="SCIENCE")

    vram_session = start_session(state, repo=repo, work_id="WORK-VRAM", bootstrap_id=bootstrap_id, codex_id="codex-vram-original", generated_at="2026-08-04T00:04:00Z")
    vram_session_id = vram_session["session"]["session_id"]
    run_eiel(["sessionManager", "interrupt", vram_session_id, "--interrupted-during", "alias lifetime probe", "--next-action", "Resume the VRAM probe.", "--state-root", str(state), "--generated-at", "2026-08-05T00:00:00Z"], state_root=state)
    run_eiel(["sessionManager", "resume", vram_session_id, "--codex-id", "codex-vram-replacement", "--takeover-reason", "five-day pause", "--state-root", str(state), "--generated-at", "2026-08-10T00:00:00Z"], state_root=state)
    run_eiel(["sessionManager", "complete", vram_session_id, "--final-step", "VRAM repair validated", "--state-root", str(state), "--generated-at", "2026-08-11T00:00:00Z"], state_root=state)
    run_eiel(["workManager", "complete", "WORK-VRAM", "--last-completed-step", "VRAM repair validated", "--state-root", str(state), "--generated-at", "2026-08-11T00:01:00Z"], state_root=state)

    qualification = run_eiel(["workManager", "inspect", "WORK-W32-QUAL", "--state-root", str(state)], state_root=state)
    assert qualification["work"]["status"] == "BLOCKED"
    assert qualification["work"]["derived_status"] == "READY_TO_RESUME"
    run_eiel(["workManager", "resume", "WORK-W32-QUAL", "--state-root", str(state), "--generated-at", "2026-08-11T00:02:00Z"], state_root=state)
    qualification_session = start_session(state, repo=repo, work_id="WORK-W32-QUAL", bootstrap_id=bootstrap_id, codex_id="codex-qualification", generated_at="2026-08-11T00:03:00Z")
    qualification_session_id = qualification_session["session"]["session_id"]
    run_eiel(["sessionManager", "complete", qualification_session_id, "--final-step", "route qualified", "--state-root", str(state), "--generated-at", "2026-08-12T00:00:00Z"], state_root=state)
    run_eiel(["workManager", "complete", "WORK-W32-QUAL", "--last-completed-step", "route qualified", "--state-root", str(state), "--generated-at", "2026-08-12T00:01:00Z"], state_root=state)

    campaign = run_eiel(["workManager", "inspect", "WORK-W32-CAMPAIGN", "--state-root", str(state)], state_root=state)
    assert campaign["work"]["status"] == "BLOCKED"
    assert campaign["work"]["derived_status"] == "READY_TO_RESUME"
    graph = run_eiel(["workManager", "graph", "WORK-W32-CAMPAIGN", "--state-root", str(state)], state_root=state)
    assert graph["cycle"] is None
    assert graph["work"]["WORK-VRAM"]["status"] == "COMPLETED"
    assert graph["work"]["WORK-W32-QUAL"]["status"] == "COMPLETED"
    assert graph["work"]["WORK-W32-CAMPAIGN"]["status"] == "BLOCKED"
    assert graph["work"]["WORK-W32-CAMPAIGN"]["derived_status"] == "READY_TO_RESUME"


def test_workspace_reports_missing_bootstrap_without_blocking_authorized_work(tmp_path: Path) -> None:
    state = tmp_path / "state"
    result = run_eiel(["workspace", "open", "--state-root", str(state)], state_root=state)
    assert result["status"] == "WORKSPACE_ASSISTANCE_UNAVAILABLE"
    assert result["reason_codes"] == ["workspace_bootstrap_missing"]
    assert result["execution_blocked"] is False
    assert result["problems"] == []
    assert result["warnings"] == ["no retained bootstrap receipt exists for the current repository"]


def test_workspace_implicit_open_does_not_reuse_receipt_after_head_change(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    original = run_bootstrap(repo=repo, state_root=state)
    changed = repo / "engine" / "post_bootstrap_change.py"
    changed.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", str(changed)], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "advance fixture head"], check=True)

    result = run_eiel(
        ["workspace", "open", "--repo-root", str(repo), "--state-root", str(state)],
        state_root=state,
    )

    assert result["status"] == "WORKSPACE_ASSISTANCE_UNAVAILABLE"
    assert result["reason_codes"] == ["workspace_bootstrap_missing"]
    assert result["execution_blocked"] is False
    assert result["warnings"] == [
        "no retained bootstrap receipt matches the current repository state"
    ]
    command = result["next_actions"][0]["command"]
    assert f"bootstrap refresh {original['bootstrap']['bootstrap_id']}" in command
    assert "<task>" not in command


def test_workspace_does_not_adopt_a_single_session_from_another_repository_lane(
    tmp_path: Path,
) -> None:
    first_repo = make_bootstrap_repo(tmp_path / "first")
    second_repo = make_bootstrap_repo(tmp_path / "second")
    state = tmp_path / "state"
    first_bootstrap = run_bootstrap(repo=first_repo, state_root=state)["bootstrap"]["bootstrap_id"]
    create_work(
        state,
        work_id="WORK-OTHER-LANE",
        title="Other lane",
        generated_at="2026-08-04T00:01:00Z",
        status="ACTIVE",
    )
    other_session = start_session(
        state,
        repo=first_repo,
        work_id="WORK-OTHER-LANE",
        bootstrap_id=first_bootstrap,
        codex_id="codex-other-lane",
        generated_at="2026-08-04T00:02:00Z",
    )["session"]["session_id"]
    second_bootstrap = run_bootstrap(repo=second_repo, state_root=state)["bootstrap"]["bootstrap_id"]

    result = run_eiel(
        [
            "start",
            "--repo-root",
            str(second_repo),
            "--bootstrap-record",
            second_bootstrap,
            "--state-root",
            str(state),
        ],
        state_root=state,
    )

    assert result["status"] == "WORKSPACE_READY"
    assert result["execution_blocked"] is False
    assert result["workspace"].get("session") is None
    assert other_session in result["workspace"]["portfolio"].get("resumable_session_ids", []) or (
        result["workspace"]["portfolio"]["resumable_session_count"] == 1
    )
    assert any("other repository lanes" in warning for warning in result["warnings"])


def test_workspace_explicitly_reports_optional_bridge_absence(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    result = run_eiel(["workspace", "open", "--bootstrap-record", bootstrap_id, "--state-root", str(state)], state_root=state)
    assert result["workspace"]["bridge"]["required"] is False
    assert result["workspace"]["bridge"]["status"] == "not_installed_optional"
    assert result["workspace"]["bridge"]["automatic_execution"] is False


def _write_qualification_evidence(path: Path, *, decision: str, obligations: list[str]) -> None:
    path.write_text(
        json.dumps({
            "decision": decision,
            "required_obligations": obligations,
            "satisfied_recipe_obligations": list(obligations),
            "required_obligations_sha256": "evidence-sha-ignored",
            "evidence_bindings": [
                {"evidence_id": f"evidence-{decision}", "kind": "test-fixture", "uri": "file://test", "sha256": "ignored"},
            ],
        }),
        encoding="utf-8",
    )


def test_workspace_distinguishes_committed_implementation_from_blocked_qualification(
    tmp_path: Path,
) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    create_work(state, work_id="WORK-BLOCKED-QUAL", title="Committed implementation", status="COMPLETED", generated_at="2026-08-04T00:01:00Z")
    evidence = tmp_path / "blocked_qualification.json"
    _write_qualification_evidence(evidence, decision="B", obligations=["direct-resident-route"])
    decision = run_eiel(
        ["qualificationManager", "decide", "--evidence", str(evidence), "--state-root", str(state), "--generated-at", "2026-08-04T00:02:00Z"],
        state_root=state,
        expected=(0, 1),
    )["decision_id"]
    run_eiel(
        [
            "workManager",
            "update",
            "WORK-BLOCKED-QUAL",
            "--qualification-ref",
            decision,
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T00:03:00Z",
        ],
        state_root=state,
    )
    workspace = run_eiel(
        [
            "workspace",
            "open",
            "--bootstrap-record",
            bootstrap_id,
            "--work-id",
            "WORK-BLOCKED-QUAL",
            "--state-root",
            str(state),
        ],
        state_root=state,
    )
    implqual = workspace["workspace"]["implementation_and_qualification"]
    assert implqual["implementation"]["status"] == "RECORD_ONLY_COMPLETION"
    assert "no implementation acceptance" in implqual["implementation"]["reason"]
    assert implqual["qualification"]["status"] in {"BLOCKED", "NOT_QUALIFIED"}


def test_qualified_qualification_does_not_force_committed_implementation(
    tmp_path: Path,
) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    create_work(state, work_id="WORK-QUAL-PRESENT", title="Qualified route", status="READY", generated_at="2026-08-04T00:01:00Z")
    evidence = tmp_path / "qualified_implementation.json"
    _write_qualification_evidence(evidence, decision="A", obligations=["direct-resident-route"])
    decision = run_eiel(
        ["qualificationManager", "decide", "--evidence", str(evidence), "--state-root", str(state), "--generated-at", "2026-08-04T00:02:00Z"],
        state_root=state,
    )["decision_id"]
    run_eiel(
        [
            "workManager",
            "update",
            "WORK-QUAL-PRESENT",
            "--qualification-ref",
            decision,
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T00:03:00Z",
        ],
        state_root=state,
    )
    workspace = run_eiel(
        [
            "workspace",
            "open",
            "--bootstrap-record",
            bootstrap_id,
            "--work-id",
            "WORK-QUAL-PRESENT",
            "--state-root",
            str(state),
        ],
        state_root=state,
    )
    implqual = workspace["workspace"]["implementation_and_qualification"]
    assert implqual["qualification"]["status"] == "QUALIFIED"
    assert implqual["implementation"]["status"] != "COMMITTED"
    assert implqual["implementation"]["status"] == "PRESENT"


def test_workspace_audit_and_handoff_surface_distinct_states(
    tmp_path: Path,
) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    create_work(state, work_id="WORK-AUDIT-HANDOFF", title="Surface split", status="READY", generated_at="2026-08-04T00:01:00Z")
    evidence = tmp_path / "qualified_handoff.json"
    _write_qualification_evidence(evidence, decision="A", obligations=["direct-resident-route"])
    decision = run_eiel(
        ["qualificationManager", "decide", "--evidence", str(evidence), "--state-root", str(state), "--generated-at", "2026-08-04T00:02:00Z"],
        state_root=state,
    )["decision_id"]
    run_eiel(
        [
            "workManager",
            "update",
            "WORK-AUDIT-HANDOFF",
            "--qualification-ref",
            decision,
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T00:03:00Z",
        ],
        state_root=state,
    )

    handoff = run_eiel(
        [
            "workspace",
            "handoff",
            "--bootstrap-record",
            bootstrap_id,
            "--work-id",
            "WORK-AUDIT-HANDOFF",
            "--state-root",
            str(state),
        ],
        state_root=state,
    )
    assert handoff["status"] == "HANDOFF_READY"
    handoff_payload = handoff["handoff"]
    assert handoff_payload["implementation_and_qualification"]["implementation"]["status"] == "PRESENT"
    assert handoff_payload["implementation_and_qualification"]["qualification"]["status"] == "QUALIFIED"

    audit = run_eiel(
        ["workspace", "audit", "--repo-root", str(repo), "--state-root", str(state), "--generated-at", "2026-08-04T00:04:00Z"],
        state_root=state,
    )
    rows = audit["audit"]["implementation_and_qualification"]
    match = [row for row in rows if row["work_id"] == "WORK-AUDIT-HANDOFF"][0]
    assert match["implementation"]["status"] == "PRESENT"
    assert match["qualification"]["status"] == "QUALIFIED"


def test_workspace_stale_receipt_advisory_preserves_refresh_context(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    (repo / "engine/dirty.py").write_text("x=1\n", encoding="utf-8")
    state = tmp_path / "state"
    bootstrap = run_bootstrap(
        repo=repo,
        state_root=state,
        acknowledge_dirty=True,
        authority_domain="execution_route",
        generated_at="2026-08-04T00:00:00Z",
    )
    bootstrap_id = bootstrap["bootstrap"]["bootstrap_id"]
    advisory = run_eiel(
        [
            "workspace",
            "open",
            "--bootstrap-record",
            bootstrap_id,
            "--repo-root",
            str(repo),
            "--max-bootstrap-age-hours",
            "0.01",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T01:00:00Z",
        ],
        state_root=state,
    )
    assert advisory["execution_blocked"] is False
    assert advisory["problems"] == []
    assert advisory["workflow_administration_required"] is True
    command = advisory["next_actions"][0]["command"]
    assert "--authority-domain" in command
    assert "execution_route" in command
    assert "--acknowledge-dirty" in command
    assert "--state-root" in command


def test_workspace_reports_bootstrap_authority_domain_mismatch_as_advisory(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap = run_bootstrap(
        repo=repo,
        state_root=state,
        authority_domain="execution_route",
        generated_at="2026-08-04T00:00:00Z",
    )
    bootstrap_id = bootstrap["bootstrap"]["bootstrap_id"]
    advisory = run_eiel(
        [
            "workspace",
            "open",
            "--repo-root",
            str(repo),
            "--bootstrap-record",
            bootstrap_id,
            "--authority-domain",
            "campaign_decision",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T01:00:00Z",
        ],
        state_root=state,
    )
    assert advisory["status"] == "WORKSPACE_ASSISTANCE_UNAVAILABLE"
    assert advisory["reason_codes"] == ["workspace_bootstrap_refresh_required"]
    assert advisory["execution_blocked"] is False
    assert advisory["problems"] == []
    assert any("authority domain" in warning.lower() for warning in advisory["warnings"])


def test_workspace_audit_requires_session_reconciliation_for_terminal_work(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]

    create_work(state, work_id="WORK-DEADLOCK", title="Deadlock repair", generated_at="2026-08-04T00:01:00Z", status="ACTIVE")
    session = start_session(
        state,
        repo=repo,
        work_id="WORK-DEADLOCK",
        bootstrap_id=bootstrap_id,
        codex_id="codex-deadlock",
        generated_at="2026-08-04T00:02:00Z",
    )
    session_id = session["session"]["session_id"]
    run_eiel([
        "sessionManager",
        "interrupt",
        session_id,
        "--interrupted-during",
        "deadlock repair fixture",
        "--next-action",
        "Replay after lifecycle repair",
        "--state-root",
        str(state),
        "--generated-at",
        "2026-08-04T00:03:00Z",
    ], state_root=state)
    rejected = run_eiel([
        "workManager",
        "discontinue",
        "WORK-DEADLOCK",
        "--force",
        "--state-root",
        str(state),
        "--generated-at",
        "2026-08-04T01:00:00Z",
    ], state_root=state, expected=2)
    assert rejected["reason_codes"] == ["work_force_terminal_requires_reconcile"]
    run_eiel([
        "workManager",
        "reconcile",
        "WORK-DEADLOCK",
        "--disposition",
        "DISCONTINUE",
        "--reason",
        "explicit fixture reconciliation",
        "--acknowledge-no-acceptance",
        "--state-root",
        str(state),
        "--generated-at",
        "2026-08-04T01:00:30Z",
    ], state_root=state)

    inconsistent = run_eiel(["workspace", "audit", "--repo-root", str(repo), "--state-root", str(state)], state_root=state, expected=2)
    assert inconsistent["status"] == "INCONSISTENT"
    assert any("non-terminal session" in str(problem).lower() for problem in inconsistent["audit"]["problems"])

    run_eiel([
        "sessionManager",
        "reconcile-terminal-parent",
        session_id,
        "--status",
        "DISCONTINUED",
        "--reason",
        "terminal reconciliation fixture repair",
        "--state-root",
        str(state),
        "--generated-at",
        "2026-08-04T02:00:00Z",
    ], state_root=state)

    consistent = run_eiel(["workspace", "audit", "--repo-root", str(repo), "--state-root", str(state)], state_root=state)
    assert consistent["status"] == "CONSISTENT"
    handoff = run_eiel(
        [
            "workspace",
            "handoff",
            "--work-id",
            "WORK-DEADLOCK",
            "--bootstrap-record",
            bootstrap_id,
            "--state-root",
            str(state),
        ],
        state_root=state,
    )


def test_workspace_audit_treats_deleted_terminal_worktree_as_historical(tmp_path: Path) -> None:
    session_repo = make_bootstrap_repo(tmp_path / "session")
    audit_repo = make_bootstrap_repo(tmp_path / "audit")
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=session_repo, state_root=state)["bootstrap"]["bootstrap_id"]
    create_work(
        state,
        work_id="WORK-HISTORICAL-AUDIT",
        title="Historical audit fixture",
        generated_at="2026-08-04T00:01:00Z",
        status="ACTIVE",
    )
    session_id = start_session(
        state,
        repo=session_repo,
        work_id="WORK-HISTORICAL-AUDIT",
        bootstrap_id=bootstrap_id,
        codex_id="codex-historical-audit",
        generated_at="2026-08-04T00:02:00Z",
    )["session"]["session_id"]
    run_eiel(
        [
            "sessionManager", "complete", session_id,
            "--final-step", "historical fixture complete",
            "--state-root", str(state),
            "--generated-at", "2026-08-04T00:03:00Z",
        ],
        state_root=state,
    )
    run_eiel(
        [
            "sessionManager", "context-receipt", session_id,
            "--repo-root", str(session_repo),
            "--attach",
            "--state-root", str(state),
            "--generated-at", "2026-08-04T00:04:00Z",
        ],
        state_root=state,
    )
    run_eiel(
        [
            "workManager", "complete", "WORK-HISTORICAL-AUDIT",
            "--record-only-reason", "historical fixture completion",
            "--state-root", str(state),
            "--generated-at", "2026-08-04T00:05:00Z",
        ],
        state_root=state,
    )
    session_repo.rename(tmp_path / "deleted-session-worktree")

    audit = run_eiel(
        ["workspace", "audit", "--repo-root", str(audit_repo), "--state-root", str(state)],
        state_root=state,
    )
    assert audit["status"] == "CONSISTENT_WITH_WARNINGS"
    assert audit["audit"]["problems"] == []
    assert any(
        "repository currentness was not revalidated" in warning
        for warning in audit["audit"]["warnings"]
    )
