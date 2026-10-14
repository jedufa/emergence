from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tests.eiel_actor_v11_helpers import run_eiel
from tests.eiel_actor_v14_helpers import create_work, make_bootstrap_repo, run_bootstrap, start_session


def _setup(tmp_path: Path) -> tuple[Path, Path, str]:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap = run_bootstrap(repo=repo, state_root=state)
    bootstrap_id = bootstrap["bootstrap"]["bootstrap_id"]
    create_work(state, work_id="WORK-SESSION", title="Session work", generated_at="2026-08-04T00:01:00Z", status="ACTIVE")
    return repo, state, bootstrap_id


def test_session_records_interruption_and_replacement_codex_without_losing_original(tmp_path: Path) -> None:
    repo, state, bootstrap_id = _setup(tmp_path)
    created = start_session(state, repo=repo, work_id="WORK-SESSION", bootstrap_id=bootstrap_id, codex_id="codex-original", generated_at="2026-08-04T00:02:00Z")
    session_id = created["session"]["session_id"]
    run_eiel(
        [
            "sessionManager", "interrupt", session_id,
            "--interrupted-during", "qualification evidence review",
            "--next-action", "Resume the evidence review from row 18.",
            "--next-command", "eiel qualificationManager check --request request.json",
            "--state-root", str(state),
            "--generated-at", "2026-08-05T00:00:00Z",
        ],
        state_root=state,
    )
    resumed = run_eiel(
        [
            "sessionManager", "resume", session_id,
            "--codex-id", "codex-replacement",
            "--takeover-reason", "original Codex unavailable after five days",
            "--state-root", str(state),
            "--generated-at", "2026-08-10T00:00:00Z",
        ],
        state_root=state,
    )
    session = resumed["session"]
    assert session["original_codex_id"] == "codex-original"
    assert session["current_codex_id"] == "codex-replacement"
    assert session["preferred_continuation_codex_id"] == "codex-original"
    assert [row["role"] for row in session["codex_participants"]] == ["original", "takeover"]
    assert session["next_action"] == "Resume the evidence review from row 18."


def test_session_checkpoint_retains_commands_steps_and_artifacts(tmp_path: Path) -> None:
    repo, state, bootstrap_id = _setup(tmp_path)
    created = start_session(state, repo=repo, work_id="WORK-SESSION", bootstrap_id=bootstrap_id, codex_id="codex-1", generated_at="2026-08-04T00:02:00Z")
    session_id = created["session"]["session_id"]
    result = run_eiel(
        [
            "sessionManager", "checkpoint", session_id,
            "--current-stage", "validation",
            "--current-step", "run focused tests",
            "--completed-step", "implementation complete",
            "--success-command", "python -m py_compile tools/actors/session_manager.py",
            "--success-result", "passed",
            "--next-action", "Run the workflow tests.",
            "--next-command", "pytest -q tests/test_eiel_session_manager_v1_4.py",
            "--artifact-ref", "/tmp/output.json",
            "--report-ref", "reports/session.md",
            "--state-root", str(state),
            "--generated-at", "2026-08-04T01:00:00Z",
        ],
        state_root=state,
    )
    session = result["session"]
    assert session["current_stage"] == "validation"
    assert session["completed_steps"] == ["implementation complete"]
    assert session["last_successful_command"]["result"] == "passed"
    assert session["references"]["artifact_paths"] == ["/tmp/output.json"]
    handoff = run_eiel(["sessionManager", "handoff", session_id, "--state-root", str(state)], state_root=state)
    assert handoff["handoff"]["next_command"] == "pytest -q tests/test_eiel_session_manager_v1_4.py"


def test_terminal_reference_repair_reclassifies_invalid_evidence_path(tmp_path: Path) -> None:
    repo, state, bootstrap_id = _setup(tmp_path)
    created = start_session(
        state,
        repo=repo,
        work_id="WORK-SESSION",
        bootstrap_id=bootstrap_id,
        codex_id="codex-1",
        generated_at="2026-08-04T00:02:00Z",
    )
    session_id = created["session"]["session_id"]
    invalid_path = "/tmp/emergence-test/storage/fixture/evidence"
    run_eiel(
        [
            "sessionManager", "checkpoint", session_id,
            "--evidence-ref", invalid_path,
            "--state-root", str(state),
            "--generated-at", "2026-08-04T00:03:00Z",
        ],
        state_root=state,
    )
    run_eiel(
        [
            "sessionManager", "complete", session_id,
            "--final-step", "terminal fixture complete",
            "--state-root", str(state),
            "--generated-at", "2026-08-04T00:04:00Z",
        ],
        state_root=state,
    )
    run_eiel(
        [
            "workManager", "complete", "WORK-SESSION",
            "--record-only-reason", "terminal fixture completion",
            "--last-completed-step", "terminal fixture complete",
            "--state-root", str(state),
            "--generated-at", "2026-08-04T00:05:00Z",
        ],
        state_root=state,
    )
    inconsistent = run_eiel(
        ["workspace", "audit", "--state-root", str(state)],
        state_root=state,
        expected=2,
    )
    assert invalid_path in " ".join(inconsistent["problems"])
    repaired = run_eiel(
        [
            "sessionManager", "repair-references", session_id,
            "--reason", "The runtime directory was stored in the evidence-ID namespace.",
            "--remove-evidence-ref", invalid_path,
            "--artifact-ref", invalid_path,
            "--state-root", str(state),
            "--generated-at", "2026-08-04T00:06:00Z",
        ],
        state_root=state,
    )
    assert repaired["session"]["references"]["evidence_assessment_ids"] == []
    assert repaired["session"]["references"]["artifact_paths"] == [invalid_path]
    consistent = run_eiel(["workspace", "audit", "--state-root", str(state)], state_root=state)
    assert consistent["status"] == "CONSISTENT"
    assert consistent["audit"]["problems"] == []
    record = json.loads((state / "sessions" / "records" / f"{session_id}.json").read_text(encoding="utf-8"))
    assert record["history"][-1]["action"] == "repair_references"


def test_parallel_sessions_require_distinct_worktree_and_branch_ownership(tmp_path: Path) -> None:
    repo, state, bootstrap_id = _setup(tmp_path)
    start_session(state, repo=repo, work_id="WORK-SESSION", bootstrap_id=bootstrap_id, codex_id="codex-1", generated_at="2026-08-04T00:02:00Z")
    blocked = run_eiel(
        [
            "sessionManager", "start", "WORK-SESSION",
            "--bootstrap-record", bootstrap_id,
            "--codex-id", "codex-2",
            "--repo-root", str(repo),
            "--state-root", str(state),
            "--generated-at", "2026-08-04T00:03:00Z",
        ],
        state_root=state,
        expected=2,
    )
    assert blocked["reason_codes"] == ["session_active_conflict"]
    overlapping = run_eiel(
        [
            "sessionManager", "start", "WORK-SESSION",
            "--bootstrap-record", bootstrap_id,
            "--codex-id", "codex-2",
            "--repo-root", str(repo),
            "--allow-parallel",
            "--state-root", str(state),
            "--generated-at", "2026-08-04T00:03:00Z",
        ],
        state_root=state,
        expected=2,
    )
    assert overlapping["reason_codes"] == ["session_parallel_repository_conflict"]

    lane_b = tmp_path / "lane-b"
    subprocess.run(
        ["git", "worktree", "add", "-q", "-b", "lane-b", str(lane_b), "HEAD"],
        cwd=repo,
        check=True,
    )
    for marker in ("tests", "docs", "evidences"):
        (lane_b / marker).mkdir(parents=True, exist_ok=True)
    lane_b_bootstrap = run_bootstrap(
        repo=lane_b,
        state_root=state,
        generated_at="2026-08-04T00:04:00Z",
    )["bootstrap"]["bootstrap_id"]
    parallel = start_session(
        state,
        repo=lane_b,
        work_id="WORK-SESSION",
        bootstrap_id=lane_b_bootstrap,
        codex_id="codex-2",
        generated_at="2026-08-04T00:05:00Z",
        allow_parallel=True,
    )
    assert parallel["session"]["status"] == "ACTIVE"
    assert parallel["session"]["record_sha256"]


def test_session_cannot_start_for_blocked_or_terminal_work(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    create_work(state, work_id="WORK-BLOCKER", title="Blocker", generated_at="2026-08-04T00:01:00Z", status="ACTIVE")
    create_work(state, work_id="WORK-BLOCKED", title="Blocked", generated_at="2026-08-04T00:02:00Z", dependency=["WORK-BLOCKER"])
    blocked = run_eiel(
        ["sessionManager", "start", "WORK-BLOCKED", "--bootstrap-record", bootstrap_id, "--codex-id", "codex", "--repo-root", str(repo), "--state-root", str(state)],
        state_root=state,
        expected=2,
    )
    assert blocked["reason_codes"] == ["session_work_dependencies_unresolved"]
    run_eiel(["workManager", "complete", "WORK-BLOCKER", "--force", "--state-root", str(state), "--generated-at", "2026-08-05T00:00:00Z"], state_root=state)
    run_eiel(["workManager", "discontinue", "WORK-BLOCKED", "--force", "--state-root", str(state), "--generated-at", "2026-08-06T00:00:00Z"], state_root=state)
    terminal = run_eiel(
        ["sessionManager", "start", "WORK-BLOCKED", "--bootstrap-record", bootstrap_id, "--codex-id", "codex", "--repo-root", str(repo), "--state-root", str(state)],
        state_root=state,
        expected=2,
    )
    assert terminal["reason_codes"] == ["session_work_not_open"]


def test_session_completion_does_not_complete_work_implicitly(tmp_path: Path) -> None:
    repo, state, bootstrap_id = _setup(tmp_path)
    session_id = start_session(state, repo=repo, work_id="WORK-SESSION", bootstrap_id=bootstrap_id, codex_id="codex-1", generated_at="2026-08-04T00:02:00Z")["session"]["session_id"]
    run_eiel(["sessionManager", "complete", session_id, "--final-step", "handoff written", "--state-root", str(state), "--generated-at", "2026-08-05T00:00:00Z"], state_root=state)
    work = run_eiel(["workManager", "inspect", "WORK-SESSION", "--state-root", str(state)], state_root=state)
    assert work["work"]["status"] == "ACTIVE"


def test_session_reconcile_terminal_parent_marks_session_terminal(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    create_work(state, work_id="WORK-SESSION-RECON", title="Reconciliation path", generated_at="2026-08-04T00:01:00Z", status="ACTIVE")
    session = start_session(
        state,
        repo=repo,
        work_id="WORK-SESSION-RECON",
        bootstrap_id=bootstrap_id,
        codex_id="codex-recon",
        generated_at="2026-08-04T00:02:00Z",
    )
    session_id = session["session"]["session_id"]
    run_eiel(
        ["sessionManager", "pause", session_id, "--next-action", "Wait for terminal work transition", "--state-root", str(state), "--generated-at", "2026-08-04T00:03:00Z"],
        state_root=state,
    )
    rejected = run_eiel(
        ["workManager", "discontinue", "WORK-SESSION-RECON", "--force", "--state-root", str(state), "--generated-at", "2026-08-04T01:00:00Z"],
        state_root=state,
        expected=2,
    )
    assert rejected["reason_codes"] == ["work_force_terminal_requires_reconcile"]
    run_eiel(
        [
            "workManager", "reconcile", "WORK-SESSION-RECON",
            "--disposition", "DISCONTINUE",
            "--reason", "explicit administrative fixture reconciliation",
            "--acknowledge-no-acceptance",
            "--state-root", str(state),
            "--generated-at", "2026-08-04T01:00:00Z",
        ],
        state_root=state,
    )
    reconciled = run_eiel(
        [
            "sessionManager",
            "reconcile-terminal-parent",
            session_id,
            "--status",
            "DISCONTINUED",
            "--reason",
            "experimental stop: terminal parent work",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T02:00:00Z",
        ],
        state_root=state,
    )
    assert reconciled["session"]["status"] == "DISCONTINUED"
    assert reconciled["session"]["next_action"] is None
    assert reconciled["session"]["next_command"] is None
    history = run_eiel(["sessionManager", "history", session_id, "--state-root", str(state)], state_root=state)
    assert history["history"][-1]["status"] == "DISCONTINUED"
    assert history["history"][-1]["revision"] > 1
    inspected = run_eiel(["sessionManager", "inspect", session_id, "--state-root", str(state)], state_root=state)
    assert inspected["record"]["payload"]["terminal_reconciliation"]["reason"] == "experimental stop: terminal parent work"
    work = run_eiel(["workManager", "inspect", "WORK-SESSION-RECON", "--state-root", str(state)], state_root=state)
    assert work["work"]["status"] == "DISCONTINUED"
    repeat = run_eiel(
        [
            "sessionManager",
            "reconcile-terminal-parent",
            session_id,
            "--status",
            "DISCONTINUED",
            "--reason",
            "second attempt should fail clearly",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T02:10:00Z",
        ],
        state_root=state,
    )
    assert repeat["session"]["status"] == "DISCONTINUED"
    assert repeat["session"]["next_action"] is None
    assert repeat["session"]["next_command"] is None
    history = run_eiel(["sessionManager", "history", session_id, "--state-root", str(state)], state_root=state)
    assert history["history"][-1]["status"] == "DISCONTINUED"
    assert history["history"][-1]["revision"] > 2


def test_session_reconcile_terminal_parent_survives_missing_recorded_worktree(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    create_work(
        state,
        work_id="WORK-MISSING-WORKTREE",
        title="Missing worktree reconciliation",
        generated_at="2026-08-04T00:01:00Z",
        status="ACTIVE",
    )
    session_id = start_session(
        state,
        repo=repo,
        work_id="WORK-MISSING-WORKTREE",
        bootstrap_id=bootstrap_id,
        codex_id="codex-missing-worktree",
        generated_at="2026-08-04T00:02:00Z",
    )["session"]["session_id"]
    run_eiel(
        [
            "workManager", "reconcile", "WORK-MISSING-WORKTREE",
            "--disposition", "DISCONTINUE",
            "--reason", "fixture parent retirement",
            "--acknowledge-no-acceptance",
            "--state-root", str(state),
            "--generated-at", "2026-08-04T01:00:00Z",
        ],
        state_root=state,
    )

    recorded_root = str(repo.resolve())
    repo.rename(tmp_path / "removed-worktree")
    strict = run_eiel(
        [
            "sessionManager", "reconcile-terminal-parent", session_id,
            "--status", "DISCONTINUED",
            "--reason", "explicit refresh remains strict",
            "--repo-root", recorded_root,
            "--state-root", str(state),
        ],
        state_root=state,
        expected=2,
    )
    assert strict["reason_codes"] == ["git_worktree_missing"]

    reconciled = run_eiel(
        [
            "sessionManager", "reconcile-terminal-parent", session_id,
            "--status", "DISCONTINUED",
            "--reason", "historical closeout after worktree removal",
            "--state-root", str(state),
            "--generated-at", "2026-08-04T02:00:00Z",
        ],
        state_root=state,
    )
    assert reconciled["session"]["status"] == "DISCONTINUED"
    inspected = run_eiel(
        ["sessionManager", "inspect", session_id, "--state-root", str(state)],
        state_root=state,
    )
    assert inspected["record"]["payload"]["repository"]["repo_root"] == recorded_root


def test_session_reconcile_terminal_parent_preserves_drifted_repository_snapshot(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    create_work(
        state,
        work_id="WORK-DRIFTED-WORKTREE",
        title="Drifted worktree reconciliation",
        generated_at="2026-08-04T00:01:00Z",
        status="ACTIVE",
    )
    created = start_session(
        state,
        repo=repo,
        work_id="WORK-DRIFTED-WORKTREE",
        bootstrap_id=bootstrap_id,
        codex_id="codex-drifted-worktree",
        generated_at="2026-08-04T00:02:00Z",
    )
    session_id = created["session"]["session_id"]
    recorded_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    run_eiel(
        [
            "workManager", "reconcile", "WORK-DRIFTED-WORKTREE",
            "--disposition", "DISCONTINUE",
            "--reason", "fixture parent retirement",
            "--acknowledge-no-acceptance",
            "--state-root", str(state),
            "--generated-at", "2026-08-04T01:00:00Z",
        ],
        state_root=state,
    )
    (repo / "drift.txt").write_text("later repository state\n", encoding="utf-8")
    subprocess.run(["git", "add", "drift.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "later fixture commit"], cwd=repo, check=True)

    reconciled = run_eiel(
        [
            "sessionManager", "reconcile-terminal-parent", session_id,
            "--status", "DISCONTINUED",
            "--reason", "historical closeout after repository drift",
            "--state-root", str(state),
            "--generated-at", "2026-08-04T02:00:00Z",
        ],
        state_root=state,
    )
    assert reconciled["session"]["status"] == "DISCONTINUED"
    inspected = run_eiel(
        ["sessionManager", "inspect", session_id, "--state-root", str(state)],
        state_root=state,
    )
    assert inspected["record"]["payload"]["repository"]["head"] == recorded_head


def test_session_reconcile_terminal_parent_fails_for_nonterminal_parent_work(tmp_path: Path) -> None:
    repo, state, bootstrap_id = _setup(tmp_path)
    session_id = start_session(state, repo=repo, work_id="WORK-SESSION", bootstrap_id=bootstrap_id, codex_id="codex-1", generated_at="2026-08-04T00:02:00Z")["session"]["session_id"]
    blocked = run_eiel(
        [
            "sessionManager",
            "reconcile-terminal-parent",
            session_id,
            "--status",
            "DISCONTINUED",
            "--reason",
            "should fail while work remains active",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T01:00:00Z",
        ],
        state_root=state,
        expected=2,
    )
    assert blocked["reason_codes"] == ["session_parent_not_terminal"]


def test_session_history_survives_fresh_process_recovery(tmp_path: Path) -> None:
    repo, state, bootstrap_id = _setup(tmp_path)
    session_id = start_session(state, repo=repo, work_id="WORK-SESSION", bootstrap_id=bootstrap_id, codex_id="codex-1", generated_at="2026-08-04T00:02:00Z")["session"]["session_id"]
    run_eiel(["sessionManager", "pause", session_id, "--next-action", "Continue tomorrow", "--state-root", str(state), "--generated-at", "2026-08-04T23:00:00Z"], state_root=state)
    status = run_eiel(["sessionManager", "status", session_id, "--state-root", str(state)], state_root=state)
    assert status["session"]["status"] == "PAUSED"
    assert status["session"]["next_action"] == "Continue tomorrow"
    history = run_eiel(["sessionManager", "history", session_id, "--state-root", str(state)], state_root=state)
    assert history["count"] == 2
    assert [row["status"] for row in history["history"]] == ["ACTIVE", "PAUSED"]


def test_session_pause_records_handoff_after_a_commit_stales_bootstrap(tmp_path: Path) -> None:
    repo, state, bootstrap_id = _setup(tmp_path)
    session_id = start_session(
        state,
        repo=repo,
        work_id="WORK-SESSION",
        bootstrap_id=bootstrap_id,
        codex_id="codex-1",
        generated_at="2026-08-04T00:02:00Z",
    )["session"]["session_id"]
    (repo / "handoff-boundary.txt").write_text("commit after bootstrap\n", encoding="utf-8")
    subprocess.run(["git", "add", "handoff-boundary.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "Stale bootstrap fixture"], cwd=repo, check=True)

    paused = run_eiel(
        [
            "sessionManager", "pause", session_id,
            "--next-action", "Resume from the durable handoff.",
            "--state-root", str(state),
            "--generated-at", "2026-08-04T23:00:00Z",
        ],
        state_root=state,
    )

    assert paused["session"]["status"] == "PAUSED"
    assert paused["session"]["next_action"] == "Resume from the durable handoff."


def test_session_rejects_bootstrap_from_different_repository(tmp_path: Path) -> None:
    repo_a = make_bootstrap_repo(tmp_path / "a")
    repo_b = make_bootstrap_repo(tmp_path / "b")
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo_a, state_root=state)["bootstrap"]["bootstrap_id"]
    create_work(state, work_id="WORK-CROSS-REPO", title="Cross repo", generated_at="2026-08-04T00:01:00Z", status="ACTIVE")
    result = run_eiel(
        ["sessionManager", "start", "WORK-CROSS-REPO", "--bootstrap-record", bootstrap_id, "--codex-id", "codex", "--repo-root", str(repo_b), "--state-root", str(state)],
        state_root=state,
        expected=2,
    )
    assert result["reason_codes"] == ["session_bootstrap_invalid"]
