from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from tests.eiel_actor_v11_helpers import EIEL, REPO_ROOT, run_eiel
from tests.eiel_actor_v14_helpers import create_work, make_bootstrap_repo, run_bootstrap, start_session
from tools.actors.common import ActorError, advisory_lock, atomic_write_text
from tools.actors.portfolio import session_store, work_store


def _tree_hashes(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_same_head_working_tree_drift_is_nonblocking_recovery_advice(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    (repo / "AGENTS.md").write_text("# changed after Bootstrap\n", encoding="utf-8")

    opened = run_eiel(
        ["workspace", "open", "--repo-root", str(repo), "--bootstrap-record", bootstrap_id, "--state-root", str(state)],
        state_root=state,
    )
    assert opened["execution_blocked"] is False
    assert opened["problems"] == []
    assert opened["workflow_administration_required"] is True
    assert any("working-tree/index" in warning for warning in opened["warnings"])


def test_active_session_rejects_silent_identity_takeover(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    create_work(state, work_id="WORK-TAKEOVER", title="Takeover", generated_at="2026-08-06T08:00:00Z", status="ACTIVE")
    session_id = start_session(
        state,
        repo=repo,
        work_id="WORK-TAKEOVER",
        bootstrap_id=bootstrap_id,
        codex_id="codex-original",
        generated_at="2026-08-06T08:01:00Z",
    )["session"]["session_id"]

    rejected = run_eiel(
        [
            "sessionManager", "resume", session_id,
            "--codex-id", "codex-other",
            "--takeover-reason", "unauthorized active transfer",
            "--repo-root", str(repo),
            "--state-root", str(state),
        ],
        state_root=state,
        expected=2,
    )
    assert rejected["reason_codes"] == ["session_active_takeover_forbidden"]
    retained = session_store(state).get_read_only(session_id)
    assert retained["payload"]["current_codex_id"] == "codex-original"


def test_terminal_parent_workspace_never_exposes_work_command(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    create_work(state, work_id="WORK-TERMINAL", title="Terminal", generated_at="2026-08-06T08:00:00Z", status="ACTIVE")
    session_id = start_session(
        state,
        repo=repo,
        work_id="WORK-TERMINAL",
        bootstrap_id=bootstrap_id,
        codex_id="codex-terminal",
        generated_at="2026-08-06T08:01:00Z",
    )["session"]["session_id"]
    run_eiel(
        [
            "sessionManager", "interrupt", session_id,
            "--interrupted-during", "fixture",
            "--next-action", "Run unsafe command",
            "--next-command", "python -c 'print(unsafe)'",
            "--state-root", str(state),
        ],
        state_root=state,
    )
    run_eiel(
        [
            "workManager", "reconcile", "WORK-TERMINAL",
            "--disposition", "DISCONTINUE",
            "--reason", "explicit administrative repair",
            "--acknowledge-no-acceptance",
            "--state-root", str(state),
        ],
        state_root=state,
    )

    opened = run_eiel(
        [
            "workspace", "open", "--repo-root", str(repo),
            "--bootstrap-record", bootstrap_id,
            "--work-id", "WORK-TERMINAL",
            "--session-id", session_id,
            "--state-root", str(state),
        ],
        state_root=state,
    )
    assert opened["status"] == "TERMINAL_PARENT_RECONCILIATION_REQUIRED"
    assert opened["execution_blocked"] is False
    assert opened["workflow_administration_required"] is True
    command = opened["next_actions"][0].get("command") or ""
    assert "print(unsafe)" not in command
    assert "sessionManager resume" not in command


def test_workspace_audit_fails_closed_for_invalid_explicit_repository(tmp_path: Path) -> None:
    state = tmp_path / "state"
    invalid = tmp_path / "not-a-repository"
    invalid.mkdir()
    result = run_eiel(
        ["workspace", "audit", "--repo-root", str(invalid), "--state-root", str(state)],
        state_root=state,
        expected=2,
    )
    assert result["status"] == "BLOCKED"
    assert result["reason_codes"] == ["git_worktree_missing"]


def test_workspace_audit_does_not_repair_torn_current_state(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    run_bootstrap(repo=repo, state_root=state)
    create_work(state, work_id="WORK-TORN", title="Torn", generated_at="2026-08-06T08:00:00Z", status="READY")
    current = work_store(state).record_path("WORK-TORN")
    current.unlink()
    before = _tree_hashes(state)

    result = run_eiel(
        ["workspace", "audit", "--repo-root", str(repo), "--state-root", str(state)],
        state_root=state,
    )
    after = _tree_hashes(state)
    assert result["status"] == "CONSISTENT_WITH_WARNINGS"
    assert any("work recoverable stale state: WORK-TORN" in warning for warning in result["warnings"])
    assert not current.exists()
    assert before == after


def test_concurrent_session_start_has_one_owner(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    create_work(state, work_id="WORK-RACE", title="Race", generated_at="2026-08-06T08:00:00Z", status="ACTIVE")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["EIEL_ACTOR_STATE_ROOT"] = str(state)

    def command(codex_id: str) -> list[str]:
        return [
            str(EIEL), "sessionManager", "start", "WORK-RACE",
            "--bootstrap-record", bootstrap_id,
            "--codex-id", codex_id,
            "--repo-root", str(repo),
            "--stage", "implementation",
            "--current-step", "race",
            "--next-action", "Continue race fixture",
            "--state-root", str(state),
            "--generated-at", "2026-08-06T08:01:00Z",
        ]

    first = subprocess.Popen(command("codex-race-a"), cwd=REPO_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    second = subprocess.Popen(command("codex-race-b"), cwd=REPO_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    first_out, first_err = first.communicate(timeout=90)
    second_out, second_err = second.communicate(timeout=90)
    assert sorted([first.returncode, second.returncode]) == [0, 2], (first_out, first_err, second_out, second_err)
    active = [
        row for row in session_store(state).list_read_only()
        if row.get("status") == "ACTIVE"
    ]
    assert len(active) == 1


def test_lock_failure_is_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import fcntl

    def fail_lock(*_args, **_kwargs):
        raise OSError("fixture lock failure")

    monkeypatch.setattr(fcntl, "flock", fail_lock)
    with pytest.raises(ActorError) as exc:
        with advisory_lock(tmp_path / "state" / ".lock"):
            raise AssertionError("unlocked body must not execute")
    assert exc.value.code == "actor_lock_unavailable"


def test_atomic_write_fsyncs_file_and_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[int] = []
    real_fsync = os.fsync

    def tracking_fsync(fd: int) -> None:
        calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", tracking_fsync)
    target = atomic_write_text(tmp_path / "durable" / "record.txt", "durable\n")
    assert target.read_text(encoding="utf-8") == "durable\n"
    assert len(calls) >= 2


def test_work_closure_and_session_start_share_one_lifecycle_lock(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    create_work(state, work_id="WORK-CLOSE-RACE", title="Close race", generated_at="2026-08-06T08:00:00Z", status="ACTIVE")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["EIEL_ACTOR_STATE_ROOT"] = str(state)
    start_cmd = [
        str(EIEL), "sessionManager", "start", "WORK-CLOSE-RACE",
        "--bootstrap-record", bootstrap_id,
        "--codex-id", "codex-close-race",
        "--repo-root", str(repo),
        "--stage", "implementation",
        "--current-step", "race",
        "--next-action", "Continue race fixture",
        "--state-root", str(state),
        "--generated-at", "2026-08-06T08:01:00Z",
    ]
    close_cmd = [
        str(EIEL), "workManager", "discontinue", "WORK-CLOSE-RACE",
        "--reason", "concurrent close fixture",
        "--state-root", str(state),
        "--generated-at", "2026-08-06T08:01:00Z",
    ]
    start = subprocess.Popen(start_cmd, cwd=REPO_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    close = subprocess.Popen(close_cmd, cwd=REPO_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    start_out, start_err = start.communicate(timeout=90)
    close_out, close_err = close.communicate(timeout=90)
    assert sorted([start.returncode, close.returncode]) == [0, 2], (start_out, start_err, close_out, close_err)

    work = work_store(state).get_read_only("WORK-CLOSE-RACE")
    active = [
        row for row in session_store(state).list_read_only()
        if row.get("status") == "ACTIVE" and (row.get("payload") or {}).get("work_id") == "WORK-CLOSE-RACE"
    ]
    assert not (work.get("status") in {"COMPLETED", "DISCONTINUED", "OBSOLETE"} and active)


def test_bootstrap_revalidates_environment_fingerprint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tools.actors import bootstrap as bootstrap_module
    from tools.actors.bootstrap import bootstrap_store, git_repository_identity, validate_bootstrap_record

    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    record = bootstrap_store(state).get_read_only(bootstrap_id)
    live_repository = git_repository_identity(repo)
    original = bootstrap_module._environment_payload

    def changed_environment(*args, **kwargs):
        payload = original(*args, **kwargs)
        payload["fingerprint_sha256"] = "sha256:" + ("f" * 64)
        return payload

    monkeypatch.setattr(bootstrap_module, "_environment_payload", changed_environment)
    problems = validate_bootstrap_record(
        record,
        expected_repository=live_repository,
        revalidate_environment=True,
        max_age_hours=None,
    )
    assert "bootstrap Python/package/Torch/CUDA environment fingerprint changed" in problems


def test_direct_session_inspect_withholds_stale_active_command(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    create_work(state, work_id="WORK-STALE-INSPECT", title="Stale inspect", generated_at="2026-08-06T08:00:00Z", status="ACTIVE")
    session_id = start_session(
        state,
        repo=repo,
        work_id="WORK-STALE-INSPECT",
        bootstrap_id=bootstrap_id,
        codex_id="codex-stale-inspect",
        generated_at="2026-08-06T08:01:00Z",
    )["session"]["session_id"]
    run_eiel(
        [
            "sessionManager", "checkpoint", session_id,
            "--next-action", "Run retained command",
            "--next-command", "python -c 'print(stale)'",
            "--state-root", str(state),
        ],
        state_root=state,
    )
    (repo / "AGENTS.md").write_text("# same-HEAD drift after checkpoint\n", encoding="utf-8")

    inspected = run_eiel(
        ["sessionManager", "inspect", session_id, "--state-root", str(state)],
        state_root=state,
    )
    assert inspected["status"] == "BLOCKED"
    commands = [str(item.get("command") or "") for item in inspected.get("next_actions") or []]
    assert not any("print(stale)" in command for command in commands)
    assert any("repository" in problem.lower() for problem in inspected.get("problems") or [])


def test_workspace_handoff_has_material_digest_independent_of_timestamp(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]

    first = run_eiel(
        [
            "workspace", "handoff", "--repo-root", str(repo),
            "--bootstrap-record", bootstrap_id,
            "--state-root", str(state),
            "--generated-at", "2026-08-06T08:01:00Z",
        ],
        state_root=state,
    )
    second = run_eiel(
        [
            "workspace", "handoff", "--repo-root", str(repo),
            "--bootstrap-record", bootstrap_id,
            "--state-root", str(state),
            "--generated-at", "2026-08-06T09:01:00Z",
        ],
        state_root=state,
    )
    assert first["handoff"]["generated_at_utc"] != second["handoff"]["generated_at_utc"]
    assert first["handoff"]["handoff_sha256"] == second["handoff"]["handoff_sha256"]
    assert len(first["handoff"]["handoff_sha256"]) == 64
    from tools.actors.canonical import canonical_sha256
    from tools.actors.workspace import _handoff_material
    assert canonical_sha256(_handoff_material(first["handoff"])) == first["handoff"]["handoff_sha256"]


def test_context_identity_changes_when_session_owner_changes() -> None:
    from tools.actors.context_receipt import session_context_sha256

    base = {
        "record_id": "SES-CONTEXT-OWNER",
        "status": "INTERRUPTED",
        "payload": {
            "session_id": "SES-CONTEXT-OWNER",
            "work_id": "WORK-CONTEXT-OWNER",
            "original_codex_id": "codex-a",
            "current_codex_id": "codex-a",
            "preferred_continuation_codex_id": "codex-a",
            "codex_participants": [{"codex_id": "codex-a", "role": "original"}],
        },
    }
    changed = json.loads(json.dumps(base))
    changed["payload"]["current_codex_id"] = "codex-b"
    changed["payload"]["preferred_continuation_codex_id"] = "codex-b"
    changed["payload"]["codex_participants"].append({"codex_id": "codex-b", "role": "takeover"})
    assert session_context_sha256(base) != session_context_sha256(changed)


def test_workspace_withholds_command_when_bound_work_revision_changes(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    create_work(state, work_id="WORK-COMMAND-BINDING", title="Command binding", generated_at="2026-08-06T08:00:00Z", status="ACTIVE")
    session_id = start_session(
        state,
        repo=repo,
        work_id="WORK-COMMAND-BINDING",
        bootstrap_id=bootstrap_id,
        codex_id="codex-command-binding",
        generated_at="2026-08-06T08:01:00Z",
    )["session"]["session_id"]
    run_eiel(
        [
            "sessionManager", "checkpoint", session_id,
            "--next-action", "Run bound command",
            "--next-command", "python -c 'print(bound)'",
            "--state-root", str(state),
        ],
        state_root=state,
    )
    run_eiel(
        [
            "workManager", "update", "WORK-COMMAND-BINDING",
            "--next-action", "Work authority changed after checkpoint",
            "--state-root", str(state),
        ],
        state_root=state,
    )

    opened = run_eiel(
        [
            "workspace", "open", "--repo-root", str(repo),
            "--bootstrap-record", bootstrap_id,
            "--work-id", "WORK-COMMAND-BINDING",
            "--session-id", session_id,
            "--state-root", str(state),
        ],
        state_root=state,
        expected=2,
    )
    assert opened["status"] == "SESSION_COMMAND_REFRESH_REQUIRED"
    commands = [str(item.get("command") or "") for item in opened.get("next_actions") or []]
    assert not any("print(bound)" in command for command in commands)
    assert opened.get("problems")
