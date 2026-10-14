from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from tests.eiel_actor_v11_helpers import run_eiel
from tests.eiel_scheduler_v110_helpers import (
    commit_all,
    fresh_bootstrap,
    make_scheduler_fixture,
    package_payload,
)
from tools.actors.common import ActorError, new_record, revise_record
from tools.actors.portfolio import session_store, work_store
from tools.actors.store import ActorRecordStore
from tools.actors.work_manager import assignment_candidate_for_work
from tools.scheduler.contracts import assignment_store, dispatch_store, lease_store
from tools.scheduler.dispatch import (
    authorize_dispatch,
    bind_worker_thread_identity,
    claim_assignment,
    compile_assignment,
    mark_assignment_running,
    recover_preparing,
    request_assignment,
    set_dispatch_enabled,
    supersede_assignment_for_manual_takeover,
    worker_open,
)
from tools.scheduler.launcher import (
    _start_wrapped_process,
    process_identity_is_live,
    terminate_process_group,
)
from tools.scheduler.runtime import load_runtime, reconcile, run_once, start_polling


def test_materially_different_dispatch_cannot_reuse_ready_assignment(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    first = compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    original = dispatch_store(fixture.state).get(fixture.dispatch_id)
    payload = dict(original["payload"])
    payload["worker_profile"] = {
        **dict(payload["worker_profile"]),
        "fake_behavior": "blocked",
        "fake_fixture_content": "DIFFERENT\n",
    }
    payload["lease_policy"] = {"duration_seconds": 7200, "heartbeat_interval_seconds": 60}
    second = authorize_dispatch(payload, state_root=fixture.state, dispatch_id="DSP-DIFFERENT-POLICY")
    with pytest.raises(ActorError) as exc:
        compile_assignment(second["record_id"], state_root=fixture.state)
    assert exc.value.code == "assignment_authorization_conflict"
    assert assignment_store(fixture.state).get(first["record_id"])["status"] == "READY"


def test_different_worktree_at_same_head_cannot_reuse_assignment(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path / "base")
    first = compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    other = tmp_path / "other-worktree"
    subprocess.run(["git", "worktree", "add", "-q", "-b", "scheduler-other", str(other), "HEAD"], cwd=fixture.repo, check=True)
    for marker_dir in ("tests", "docs", "evidences"):
        (other / marker_dir).mkdir(parents=True, exist_ok=True)
    fresh_bootstrap(other, fixture.state)
    original = dispatch_store(fixture.state).get(fixture.dispatch_id)
    payload = dict(original["payload"])
    payload["repository"] = {"repo_root": str(other), "expected_branch": "scheduler-other"}
    second = authorize_dispatch(payload, state_root=fixture.state, dispatch_id="DSP-OTHER-WORKTREE")
    with pytest.raises(ActorError) as exc:
        compile_assignment(second["record_id"], state_root=fixture.state)
    assert exc.value.code == "assignment_authorization_conflict"
    material = assignment_store(fixture.state).get(first["record_id"])["payload"]["material"]
    assert material["repository"]["repo_root"] == str(fixture.repo.resolve())


def test_manual_active_session_is_not_scheduler_candidate(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    started = run_eiel([
        "sessionManager", "start", fixture.work_id,
        "--bootstrap-record", fixture.bootstrap_id,
        "--codex-id", "MANUAL-CODEX",
        "--repo-root", str(fixture.repo),
        "--role", "IMPLEMENTATION_AUTHORITY",
        "--state-root", str(fixture.state),
    ], state_root=fixture.state)
    assert started["session"]["status"] == "ACTIVE"
    candidate = assignment_candidate_for_work(work_store(fixture.state).get(fixture.work_id), state_root=fixture.state)
    assert candidate["eligible"] is False
    assert "active_session_exists" in candidate["reason_codes"]
    assert request_assignment(state_root=fixture.state) is None


def test_actor_store_recovers_history_written_before_current(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.actors.store as store_module

    store = ActorRecordStore(tmp_path, "crash-records")
    first = new_record(
        actor="test", record_type="fixture", record_id="REC-CRASH",
        status="READY", payload={"value": 1}, generated_at="2026-08-05T00:00:00Z",
        action="create",
    )
    store.put(first)
    second = revise_record(
        first, payload={"value": 2}, status="READY", action="advance",
        generated_at="2026-08-05T00:01:00Z", note="same transition",
    )
    real_atomic = store_module.atomic_write_json
    crashed = {"done": False}

    def crash_before_current(path, value):
        if Path(path) == store.record_path("REC-CRASH") and int(value.get("revision") or 0) == 2 and not crashed["done"]:
            crashed["done"] = True
            raise RuntimeError("injected crash after history write")
        return real_atomic(path, value)

    monkeypatch.setattr(store_module, "atomic_write_json", crash_before_current)
    with pytest.raises(RuntimeError):
        store.put(second, allow_update=True)
    monkeypatch.setattr(store_module, "atomic_write_json", real_atomic)

    recovered = store.get("REC-CRASH")
    assert recovered["revision"] == 2
    assert recovered["payload"] == {"value": 2}
    retry = revise_record(
        first, payload={"value": 2}, status="READY", action="advance",
        generated_at="2026-08-05T00:02:00Z", note="same transition",
    )
    store.put(retry, allow_update=True)
    assert store.get("REC-CRASH")["revision"] == 2


def test_thread_identity_binding_rejects_unswept_expired_lease(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    assignment = compile_assignment(fixture.dispatch_id, state_root=fixture.state, generated_at="2026-08-05T00:00:00Z")
    assignment, lease = claim_assignment(
        assignment["record_id"], scheduler_id="SCH-EXPIRY", state_root=fixture.state,
        generated_at="2026-08-05T00:00:00Z",
    )
    worker_id = f"codex-{assignment['record_id'].lower()}"
    provisional = f"EIEL-PROVISIONAL-{worker_id}"
    assignment, lease = mark_assignment_running(
        assignment["record_id"], lease["record_id"],
        {"kind": "codex_exec", "worker_id": worker_id, "thread_id": provisional},
        state_root=fixture.state, generated_at="2026-08-05T00:00:00Z",
    )
    worker_open(
        assignment["record_id"], lease["record_id"], state_root=fixture.state,
        worker_id=worker_id, codex_id=provisional, generated_at="2026-08-05T00:00:00Z",
    )
    with pytest.raises(ActorError) as exc:
        bind_worker_thread_identity(
            assignment["record_id"], lease["record_id"], state_root=fixture.state,
            worker_id=worker_id, thread_id="REAL-THREAD",
            generated_at="2026-08-05T01:00:00Z",
        )
    assert exc.value.code == "lease_expired"


def test_process_group_cleanup_kills_child_after_leader_exit(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    script = tmp_path / "leader.py"
    script.write_text(
        "import subprocess,sys\n"
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
        f"open({str(child_pid_path)!r},'w').write(str(p.pid))\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen([sys.executable, str(script)], start_new_session=True)
    for _ in range(100):
        if child_pid_path.is_file():
            break
        time.sleep(0.02)
    child_pid = int(child_pid_path.read_text())
    proc.wait(timeout=5)
    terminate_process_group(proc, grace_seconds=0.2)
    for _ in range(100):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        stat = Path(f"/proc/{child_pid}/stat")
        if stat.is_file() and stat.read_text().split()[2] == "Z":
            break
        time.sleep(0.02)
    else:
        pytest.fail("child survived cleanup after leader exit")


def test_cleanup_failure_remains_visible_and_recoverable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    import tools.scheduler.dispatch as dispatch

    real_invalidate = dispatch.invalidate_prepared_session_for_assignment

    def fail_projection(**_kwargs):
        raise ActorError("projection failed", code="projection_failed")

    def fail_cleanup(*_args, **_kwargs):
        raise ActorError("cleanup failed", code="cleanup_failed")

    monkeypatch.setattr(dispatch, "_workspace_projection", fail_projection)
    monkeypatch.setattr(dispatch, "invalidate_prepared_session_for_assignment", fail_cleanup)
    with pytest.raises(ActorError) as exc:
        compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    assert exc.value.code == "assignment_cleanup_required"
    pending = next(assignment_store(fixture.state).get(row["record_id"]) for row in assignment_store(fixture.state).list())
    assert pending["status"] == "CLEANUP_REQUIRED"
    monkeypatch.setattr(dispatch, "invalidate_prepared_session_for_assignment", real_invalidate)
    recovered = recover_preparing(state_root=fixture.state)
    assert recovered[0]["status"] == "INVALID"
    session = session_store(fixture.state).get(pending["payload"]["preparation"]["session_id"])
    assert session["payload"]["prepared_assignment_id"] is None


def test_wrapper_bounds_stderr_during_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_SCHEDULER_TEST_MODE", "1")
    logs = tmp_path / "logs"
    command = [
        sys.executable, "-c",
        "import sys,time\nfor _ in range(1000):\n sys.stderr.write('x'*4096); sys.stderr.flush(); time.sleep(0.001)\ntime.sleep(0.5)",
    ]
    proc, _identity, _handshake, _ack = _start_wrapped_process(
        target_command=command,
        logs=logs,
        stdout_path=logs / "stdout.log",
        stderr_path=logs / "stderr.log",
        identity={"kind": "fake", "worker_id": "bounded-log", "thread_id": "FAKE"},
        on_started=None,
        maximum_log_bytes=64 * 1024,
    )
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not (logs / "stderr.log").is_file():
        time.sleep(0.02)
    time.sleep(0.2)
    assert (logs / "stderr.log").stat().st_size <= 64 * 1024
    assert proc.wait(timeout=10) == 0
    terminate_process_group(proc, grace_seconds=0.2)
    assert (logs / "stderr.log").stat().st_size <= 64 * 1024


def test_dispatch_disable_between_activation_and_ack_prevents_worker_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    import tools.scheduler.runtime as runtime

    real_acknowledge = runtime.acknowledge_worker_launch

    def revoke_then_ack(*args, **kwargs):
        set_dispatch_enabled(fixture.dispatch_id, enabled=False, state_root=fixture.state)
        return real_acknowledge(*args, **kwargs)

    monkeypatch.setattr(runtime, "acknowledge_worker_launch", revoke_then_ack)
    with pytest.raises(ActorError):
        run_once(state_root=fixture.state)
    assert (fixture.repo / "tools/scheduler_fixture.py").read_text(encoding="utf-8") == "VALUE = 'baseline'\n"
    assignments = [assignment_store(fixture.state).get(row["record_id"]) for row in assignment_store(fixture.state).list()]
    assert all(row["status"] not in {"READY", "CLAIMED", "RUNNING"} for row in assignments)


def test_start_polling_reconciles_worker_surviving_scheduler_crash(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    dispatch = dispatch_store(fixture.state).get(fixture.dispatch_id)
    payload = dict(dispatch["payload"])
    payload["worker_profile"] = {**dict(payload["worker_profile"]), "fake_sleep_seconds": 3.0, "timeout_seconds": 30}
    revised = revise_record(
        dispatch, payload=payload, status="ENABLED", action="test_slow_worker",
        generated_at="2026-08-05T00:00:00Z",
    )
    dispatch_store(fixture.state).put(revised, allow_update=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["EIEL_SCHEDULER_TEST_MODE"] = "1"
    code = (
        "from tools.scheduler.runtime import run_once; "
        f"run_once(state_root={str(fixture.state)!r})"
    )
    parent = subprocess.Popen([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], env=env)
    row = None
    for _ in range(300):
        runtime = load_runtime(fixture.state)
        processes = dict(runtime.get("processes") or {})
        if processes:
            row = next(iter(processes.values()))
            identity = dict(row.get("worker_identity") or {})
            if row.get("status") == "RUNNING" and process_identity_is_live(identity.get("pid"), identity.get("process_start_identity")):
                break
        if parent.poll() is not None:
            break
        time.sleep(0.02)
    assert row is not None and row.get("status") == "RUNNING"
    identity = dict(row["worker_identity"])
    os.kill(parent.pid, signal.SIGKILL)
    parent.wait(timeout=5)
    assert process_identity_is_live(identity.get("pid"), identity.get("process_start_identity"))

    result = start_polling(
        state_root=fixture.state, poll_interval_seconds=0.1,
        max_assignments=0, max_idle_cycles=1,
    )
    assert result["events"][0]["status"] == "STARTUP_RECONCILED"
    assignments = [assignment_store(fixture.state).get(row["record_id"]) for row in assignment_store(fixture.state).list()]
    assert any(row["status"] == "CLOSED" for row in assignments)
    assert load_runtime(fixture.state)["processes"] == {}


def test_actor_store_recovers_current_written_before_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.actors.store as store_module

    store = ActorRecordStore(tmp_path, "index-crash-records")
    first = new_record(
        actor="test", record_type="fixture", record_id="REC-INDEX-CRASH",
        status="READY", payload={"value": 1}, generated_at="2026-08-05T00:00:00Z",
        action="create",
    )
    store.put(first)
    second = revise_record(
        first, payload={"value": 2}, status="READY", action="advance",
        generated_at="2026-08-05T00:01:00Z", note="index crash",
    )
    real_atomic = store_module.atomic_write_json
    crashed = {"done": False}

    def crash_index(path, value):
        if Path(path) == store.index_path and not crashed["done"]:
            crashed["done"] = True
            raise RuntimeError("injected crash after current write")
        return real_atomic(path, value)

    monkeypatch.setattr(store_module, "atomic_write_json", crash_index)
    with pytest.raises(RuntimeError):
        store.put(second, allow_update=True)
    monkeypatch.setattr(store_module, "atomic_write_json", real_atomic)

    recovered = store.get("REC-INDEX-CRASH")
    assert recovered["revision"] == 2
    index = json.loads(store.index_path.read_text(encoding="utf-8"))
    row = next(item for item in index["records"] if item["record_id"] == "REC-INDEX-CRASH")
    assert row["revision"] == 2
    assert row["record_sha256"] == recovered["record_sha256"]


def test_same_dispatch_cannot_reuse_assignment_after_branch_drift(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    first = compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    subprocess.run(["git", "checkout", "-q", "-b", "drifted-branch"], cwd=fixture.repo, check=True)
    with pytest.raises(ActorError) as exc:
        compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    assert exc.value.code == "assignment_stale"
    assert assignment_store(fixture.state).get(first["record_id"])["status"] == "READY"


def test_revised_dispatch_cannot_reuse_preparing_assignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    import tools.scheduler.dispatch as dispatch_module

    def interrupt_before_session(*_args, **_kwargs):
        raise KeyboardInterrupt("injected hard stop while PREPARING")

    monkeypatch.setattr(dispatch_module, "prepare_session_for_assignment", interrupt_before_session)
    with pytest.raises(KeyboardInterrupt):
        compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    preparing = next(
        assignment_store(fixture.state).get(row["record_id"])
        for row in assignment_store(fixture.state).list()
    )
    assert preparing["status"] == "PREPARING"

    dispatch = dispatch_store(fixture.state).get(fixture.dispatch_id)
    payload = dict(dispatch["payload"])
    payload["worker_profile"] = {
        **dict(payload["worker_profile"]),
        "fake_behavior": "blocked",
        "fake_fixture_content": "revised while preparing\n",
    }
    revised = revise_record(
        dispatch,
        payload=payload,
        status="ENABLED",
        action="revise_dispatch_during_preparation",
        generated_at="2026-08-05T00:01:00Z",
    )
    dispatch_store(fixture.state).put(revised, allow_update=True)

    monkeypatch.undo()
    with pytest.raises(ActorError) as exc:
        compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    assert exc.value.code == "assignment_authorization_conflict"
    assert assignment_store(fixture.state).get(preparing["record_id"])["status"] == "PREPARING"


def test_candidate_rejection_does_not_block_later_work(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path, work_id="WORK-MANUAL-A", queue_priority=1)
    started = run_eiel([
        "sessionManager", "start", fixture.work_id,
        "--bootstrap-record", fixture.bootstrap_id,
        "--codex-id", "MANUAL-A",
        "--repo-root", str(fixture.repo),
        "--role", "IMPLEMENTATION_AUTHORITY",
        "--state-root", str(fixture.state),
    ], state_root=fixture.state)
    assert started["session"]["status"] == "ACTIVE"

    work_b = "WORK-SCHED-B"
    content_b = fixture.repo / "tools/scheduler_fixture_b.py"
    content_b.write_text("VALUE = 'baseline-b'\n", encoding="utf-8")
    package_b = fixture.repo / "work_packages/scheduler_b.yaml"
    package_b.write_text(
        yaml.safe_dump(package_payload(work_b, content_file="tools/scheduler_fixture_b.py"), sort_keys=False),
        encoding="utf-8",
    )
    commit_all(fixture.repo, "add second scheduler work")
    bootstrap_b = fresh_bootstrap(fixture.repo, fixture.state)
    run_eiel([
        "workManager", "create", "--work-package", str(package_b),
        "--repo-root", str(fixture.repo), "--status", "ACTIVE",
        "--next-action", "Execute work B.", "--state-root", str(fixture.state),
    ], state_root=fixture.state)
    dispatch_b = authorize_dispatch({
        "work_id": work_b,
        "enabled": True,
        "allowed_authority_phases": ["implementation"],
        "worker_profile": {
            "launcher_type": "fake", "fake_behavior": "checkpoint",
            "fake_fixture_content": "VALUE = 'worker-b'\n",
            "sandbox": "workspace-write", "ephemeral": True,
        },
        "repository": {"repo_root": str(fixture.repo), "expected_branch": None},
        "queue": {"priority": 2},
        "automation_policy": {
            "allow_fresh_session": True, "allow_resume": True,
            "allow_independent_review": True, "allow_repair": True,
        },
        "lease_policy": {"duration_seconds": 3600, "heartbeat_interval_seconds": 5},
        "created_by": "HUMAN_OPERATOR",
    }, state_root=fixture.state)
    assert bootstrap_b

    assignment, rejections = request_assignment(state_root=fixture.state, with_diagnostics=True)
    assert assignment is not None
    assert assignment["payload"]["material"]["work"]["record_id"] == work_b
    assert assignment["payload"]["material"]["dispatch_request"]["record_id"] == dispatch_b["record_id"]
    # Work A is filtered by Work Manager before compilation.  The important
    # property is that it does not abort selection of the later valid Work B.
    assert all(row.get("work_id") != fixture.work_id for row in rejections)


def test_wrapper_success_removes_background_descendants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_SCHEDULER_TEST_MODE", "1")
    logs = tmp_path / "wrapper-success"
    child_pid_path = tmp_path / "background.pid"
    command = [
        sys.executable,
        "-c",
        (
            "import subprocess,sys; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            f"open({str(child_pid_path)!r},'w').write(str(p.pid))"
        ),
    ]
    proc, _identity, _handshake, _ack = _start_wrapped_process(
        target_command=command,
        logs=logs,
        stdout_path=logs / "stdout.log",
        stderr_path=logs / "stderr.log",
        identity={"kind": "fake", "worker_id": "background-cleanup", "thread_id": "FAKE"},
        on_started=None,
    )
    assert proc.wait(timeout=10) == 0
    assert child_pid_path.is_file()
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    for _ in range(100):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        stat = Path(f"/proc/{child_pid}/stat")
        if stat.is_file() and stat.read_text(encoding="utf-8").split()[2] == "Z":
            break
        time.sleep(0.02)
    else:
        pytest.fail("successful wrapper exit left a background descendant alive")


def test_manual_takeover_cleanup_failure_remains_visible_and_retryable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    assignment = compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    session_id = assignment["payload"]["material"]["session"]["record_id"]
    import tools.scheduler.dispatch as dispatch_module

    real_detach = dispatch_module.invalidate_prepared_session_for_assignment
    failed = {"done": False}

    def fail_once(*args, **kwargs):
        if not failed["done"]:
            failed["done"] = True
            raise ActorError("injected Session cleanup failure", code="injected_cleanup_failure")
        return real_detach(*args, **kwargs)

    monkeypatch.setattr(dispatch_module, "invalidate_prepared_session_for_assignment", fail_once)
    with pytest.raises(ActorError) as exc:
        supersede_assignment_for_manual_takeover(
            assignment["record_id"],
            state_root=fixture.state,
            reason="manual fallback",
            generated_at="2026-08-05T01:00:00Z",
        )
    assert exc.value.code == "injected_cleanup_failure"

    stranded = assignment_store(fixture.state).get(assignment["record_id"])
    assert stranded["status"] == "SUPERSEDED"
    assert stranded["payload"]["operational"]["manual_takeover_cleanup_required"] is True
    pending = session_store(fixture.state).get(session_id)
    assert pending["payload"]["prepared_assignment_id"] == assignment["record_id"]

    completed = supersede_assignment_for_manual_takeover(
        assignment["record_id"],
        state_root=fixture.state,
        reason="manual fallback",
        generated_at="2026-08-05T01:00:01Z",
    )
    assert completed["status"] == "SUPERSEDED"
    assert completed["payload"]["operational"]["manual_takeover_cleanup_required"] is False
    detached = session_store(fixture.state).get(session_id)
    assert detached["status"] == "INTERRUPTED"
    assert detached["payload"]["worker_identity_state"] == "DETACHED"
    assert detached["payload"]["prepared_assignment_id"] is None


def test_launch_failure_keeps_runtime_row_when_worker_loss_write_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    import tools.scheduler.runtime as runtime_module

    class FailingLauncher:
        def launch(self, **_kwargs):
            raise RuntimeError("injected launcher failure")

    monkeypatch.setattr(runtime_module, "_launcher_for_assignment", lambda _assignment: FailingLauncher())

    def fail_loss(*_args, **_kwargs):
        raise ActorError("injected worker-loss write failure", code="injected_loss_failure")

    monkeypatch.setattr(runtime_module, "record_execution_loss", fail_loss)
    with pytest.raises(ActorError) as exc:
        run_once(state_root=fixture.state)
    assert exc.value.code == "scheduler_worker_loss_recovery_required"

    runtime = load_runtime(fixture.state)
    assert len(runtime["processes"]) == 1
    row = next(iter(runtime["processes"].values()))
    assert row["recovery_required"] is True
    assert "injected_loss_failure" in row["recovery_error"]
    assignment_id = row["assignment_id"]
    assert assignment_store(fixture.state).get(assignment_id)["status"] in {"CLAIMED", "RUNNING"}

    monkeypatch.undo()
    recovered = reconcile(state_root=fixture.state, launch_grace_seconds=-1.0)
    assert recovered["outcomes"]
    assert load_runtime(fixture.state)["processes"] == {}
    assert assignment_store(fixture.state).get(assignment_id)["status"] in {"EXPIRED", "INVALID", "SUPERSEDED"}
