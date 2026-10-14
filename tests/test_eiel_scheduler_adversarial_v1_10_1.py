from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.eiel_actor_v11_helpers import run_eiel
from tests.eiel_scheduler_v110_helpers import fresh_bootstrap, make_scheduler_fixture
from tools.actors.common import ActorError, revise_record, utc_now
from tools.actors.portfolio import bootstrap_store, session_store
from tools.scheduler.contracts import (
    assignment_material_sha256,
    assignment_store,
    dispatch_store,
    lease_store,
    normalize_dispatch_payload,
    result_store,
)
from tools.scheduler.dispatch import (
    authorize_dispatch,
    claim_assignment,
    compile_assignment,
    expire_leases,
    heartbeat_lease,
    mark_assignment_running,
    request_assignment,
    set_dispatch_enabled,
    submit_worker_result,
    worker_open,
)
from tools.scheduler.launcher import FakeWorkerLauncher, terminate_process_group
from tools.scheduler.worker_protocol import result_from_repository


def _activate_fake(fixture, *, generated_at: str = "2026-08-05T00:00:00Z"):
    assignment = request_assignment(state_root=fixture.state, generated_at=generated_at)
    assert assignment is not None
    assignment, lease = claim_assignment(
        assignment["record_id"], scheduler_id="SCH-ADVERSARIAL", state_root=fixture.state, generated_at=generated_at
    )
    worker_id = f"fake-{assignment['record_id'].lower()}"
    thread_id = f"FAKE-THREAD-{assignment['record_id']}"
    assignment, lease = mark_assignment_running(
        assignment["record_id"], lease["record_id"],
        {"kind": "fake", "worker_id": worker_id, "thread_id": thread_id},
        state_root=fixture.state, generated_at=generated_at,
    )
    session = worker_open(
        assignment["record_id"], lease["record_id"], state_root=fixture.state,
        worker_id=worker_id, codex_id=thread_id, generated_at=generated_at,
    )
    return assignment, lease, session, worker_id, thread_id


def test_result_submission_retry_converges_after_first_durable_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    assignment, lease, session, worker_id, thread_id = _activate_fake(fixture)
    run_eiel([
        "sessionManager", "checkpoint", session["record_id"],
        "--next-action", "Continue safely.", "--state-root", str(fixture.state),
    ], state_root=fixture.state)
    run_eiel([
        "sessionManager", "pause", session["record_id"],
        "--next-action", "Continue safely.", "--state-root", str(fixture.state),
    ], state_root=fixture.state)
    result = result_from_repository(
        assignment_id=assignment["record_id"], lease_id=lease["record_id"], state_root=fixture.state,
        worker_identity={"kind": "fake", "worker_id": worker_id, "pid": None, "thread_id": thread_id},
        outcome="CHECKPOINTED",
        started_at_utc="2026-08-05T00:00:00Z",
        ended_at_utc="2026-08-05T00:00:30Z",
    )

    import tools.scheduler.dispatch as dispatch
    real_finalize = dispatch._finalize_result_submission
    calls = {"count": 0}

    def crash_once(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("injected crash after Worker Result write")
        return real_finalize(**kwargs)

    monkeypatch.setattr(dispatch, "_finalize_result_submission", crash_once)
    with pytest.raises(RuntimeError):
        submit_worker_result(result, state_root=fixture.state, generated_at="2026-08-05T00:01:00Z")
    assert len(result_store(fixture.state).list()) == 1
    monkeypatch.setattr(dispatch, "_finalize_result_submission", real_finalize)
    stored, closed_assignment, closed_lease = submit_worker_result(result, state_root=fixture.state, generated_at="2026-08-05T00:01:01Z")
    assert stored["record_id"]
    assert closed_assignment["status"] == "CLOSED"
    assert closed_lease["status"] == "CLOSED"
    assert len(result_store(fixture.state).list()) == 1


def test_failed_link_detaches_prepared_session_and_retry_uses_new_assignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    import tools.scheduler.dispatch as dispatch
    real_projection = dispatch._workspace_projection

    def fail_projection(**_kwargs):
        raise ActorError("injected projection failure", code="injected")

    monkeypatch.setattr(dispatch, "_workspace_projection", fail_projection)
    with pytest.raises(ActorError):
        compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    sessions = [session_store(fixture.state).get(row["record_id"]) for row in session_store(fixture.state).list()]
    assert len(sessions) == 1
    assert sessions[0]["status"] == "INTERRUPTED"
    assert sessions[0]["payload"]["worker_identity_state"] == "DETACHED"
    assert sessions[0]["payload"]["prepared_assignment_id"] is None
    invalid_ids = {row["record_id"] for row in assignment_store(fixture.state).list() if row["status"] == "INVALID"}
    assert len(invalid_ids) == 1

    monkeypatch.setattr(dispatch, "_workspace_projection", real_projection)
    retried = compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    assert retried["status"] == "READY"
    assert retried["record_id"] not in invalid_ids


def test_prepared_session_can_be_taken_over_manually_and_assignment_is_superseded(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    assignment = request_assignment(state_root=fixture.state)
    assert assignment is not None
    session_id = assignment["payload"]["material"]["session"]["record_id"]
    result = run_eiel([
        "sessionManager", "resume", session_id,
        "--take-over-prepared", "--codex-id", "MANUAL-CODEX-1",
        "--bootstrap-record", fixture.bootstrap_id, "--repo-root", str(fixture.repo),
        "--state-root", str(fixture.state),
    ], state_root=fixture.state)
    assert result["session"]["status"] == "ACTIVE"
    session = session_store(fixture.state).get(session_id)
    assert session["payload"]["worker_identity_state"] == "ACTIVE"
    assert session["payload"]["prepared_assignment_id"] is None
    assert assignment_store(fixture.state).get(assignment["record_id"])["status"] == "SUPERSEDED"


@pytest.mark.parametrize("target_kind", ["entry", "package"])
def test_uncommitted_assignment_input_drift_blocks_claim(tmp_path: Path, target_kind: str) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    assignment = request_assignment(state_root=fixture.state)
    assert assignment is not None
    target = fixture.repo / "tools/scheduler_fixture.py" if target_kind == "entry" else fixture.package_path
    target.write_text(target.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
    with pytest.raises(ActorError) as exc:
        claim_assignment(assignment["record_id"], scheduler_id="SCH-DRIFT", state_root=fixture.state)
    assert exc.value.code in {"scheduler_worktree_dirty", "assignment_stale"}


def test_disabling_dispatch_supersedes_ready_assignment(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    assignment = request_assignment(state_root=fixture.state)
    assert assignment is not None
    set_dispatch_enabled(fixture.dispatch_id, enabled=False, state_root=fixture.state)
    assert assignment_store(fixture.state).get(assignment["record_id"])["status"] == "SUPERSEDED"
    with pytest.raises(ActorError):
        claim_assignment(assignment["record_id"], scheduler_id="SCH-DISABLED", state_root=fixture.state)


def test_expired_attempt_rejects_late_new_result(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    assignment, lease, session, worker_id, thread_id = _activate_fake(fixture, generated_at="2026-08-05T00:00:00Z")
    run_eiel([
        "sessionManager", "checkpoint", session["record_id"], "--next-action", "Continue safely.",
        "--state-root", str(fixture.state),
    ], state_root=fixture.state)
    result = result_from_repository(
        assignment_id=assignment["record_id"], lease_id=lease["record_id"], state_root=fixture.state,
        worker_identity={"kind": "fake", "worker_id": worker_id, "pid": None, "thread_id": thread_id},
        outcome="CHECKPOINTED", ended_at_utc="2026-08-05T02:00:00Z",
    )
    expire_leases(state_root=fixture.state, generated_at="2026-08-05T02:00:00Z")
    with pytest.raises(ActorError):
        submit_worker_result(result, state_root=fixture.state, generated_at="2026-08-05T02:00:01Z")


def test_ready_assignment_material_rewrite_is_detected_even_with_new_digest(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    assignment = request_assignment(state_root=fixture.state)
    assert assignment is not None
    payload = dict(assignment["payload"])
    material = dict(payload["material"])
    material["next_safe_action"] = {"description": "mutated"}
    payload["material"] = material
    payload["assignment_material_sha256"] = assignment_material_sha256(material)
    revised = revise_record(assignment, payload=payload, status="READY", action="adversarial_rewrite", generated_at=utc_now())
    assignment_store(fixture.state).put(revised, allow_update=True)
    with pytest.raises(ActorError) as exc:
        claim_assignment(assignment["record_id"], scheduler_id="SCH-MUTATED", state_root=fixture.state)
    assert exc.value.code == "assignment_immutability_violation"


def test_second_dispatch_authorization_conflicts_with_existing_ready_assignment(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    first = compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    original = dispatch_store(fixture.state).get(fixture.dispatch_id)
    second_dispatch = authorize_dispatch(
        dict(original["payload"]), state_root=fixture.state, dispatch_id="DSP-SECOND-AUTHORIZATION"
    )
    with pytest.raises(ActorError) as exc:
        compile_assignment(second_dispatch["record_id"], state_root=fixture.state)
    assert exc.value.code == "assignment_authorization_conflict"
    assert assignment_store(fixture.state).get(first["record_id"])["status"] == "READY"
    ready = [row for row in assignment_store(fixture.state).list() if row["status"] == "READY"]
    assert len(ready) == 1


def test_bootstrap_record_digest_drift_blocks_claim(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    assignment = request_assignment(state_root=fixture.state)
    assert assignment is not None
    store = bootstrap_store(fixture.state)
    record = store.get(fixture.bootstrap_id)
    payload = dict(record["payload"])
    payload["checks"] = [*list(payload.get("checks") or []), {"name": "adversarial", "status": "PASS", "message": "changed"}]
    revised = revise_record(record, payload=payload, status=record["status"], action="adversarial_revision", generated_at=utc_now())
    store.put(revised, allow_update=True)
    with pytest.raises(ActorError) as exc:
        claim_assignment(assignment["record_id"], scheduler_id="SCH-BOOTSTRAP", state_root=fixture.state)
    assert exc.value.code == "assignment_stale"


def test_lease_policy_and_heartbeat_owner_are_strict(tmp_path: Path) -> None:
    with pytest.raises(ActorError):
        normalize_dispatch_payload({
            "work_id": "WORK-X", "repository": {"repo_root": str(tmp_path)},
            "lease_policy": {"duration_seconds": 30, "heartbeat_interval_seconds": 20},
        })
    fixture = make_scheduler_fixture(tmp_path / "fixture")
    assignment, lease, _session, worker_id, thread_id = _activate_fake(fixture)
    with pytest.raises(ActorError) as exc:
        heartbeat_lease(
            lease["record_id"], state_root=fixture.state,
            worker={"worker_id": "another-worker"}, generated_at="2026-08-05T00:01:00Z",
        )
    assert exc.value.code == "lease_worker_mismatch"
    renewed = heartbeat_lease(
        lease["record_id"], state_root=fixture.state,
        worker={"worker_id": worker_id, "thread_id": thread_id}, generated_at="2026-08-05T00:01:01Z",
    )
    assert renewed["status"] == "ACTIVE"


def test_scheduler_v1_rejects_higher_authority_and_dangerous_profile(tmp_path: Path) -> None:
    with pytest.raises(ActorError):
        normalize_dispatch_payload({
            "work_id": "WORK-X", "repository": {"repo_root": str(tmp_path)},
            "allowed_authority_phases": ["engineering"],
        })
    with pytest.raises(ActorError):
        normalize_dispatch_payload({
            "work_id": "WORK-X", "repository": {"repo_root": str(tmp_path)},
            "worker_profile": {"launcher_type": "codex_exec", "sandbox": "danger-full-access"},
        })


def test_terminate_process_group_kills_descendants(tmp_path: Path) -> None:
    script = tmp_path / "tree.py"
    child_pid_file = tmp_path / "child.pid"
    script.write_text(
        "import subprocess,sys,time,os\n"
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
        f"open({str(child_pid_file)!r},'w').write(str(p.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen([sys.executable, str(script)], start_new_session=True)
    for _ in range(100):
        if child_pid_file.exists():
            break
        time.sleep(0.02)
    child_pid = int(child_pid_file.read_text())
    terminate_process_group(proc, grace_seconds=0.2)
    assert proc.poll() is not None
    for _ in range(100):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        # A zombie has no capability to keep editing; confirm that state.
        stat = Path(f"/proc/{child_pid}/stat")
        assert stat.is_file() and stat.read_text().split()[2] == "Z"


def _latest_bootstrap_id(state: Path) -> str:
    rows = bootstrap_store(state).list()
    assert rows
    return str(sorted(rows, key=lambda row: (row.get("updated_at_utc") or "", row["record_id"]))[-1]["record_id"])


def test_old_interrupted_implementation_cannot_bypass_review_required(tmp_path: Path) -> None:
    from tools.scheduler.runtime import run_once
    fixture = make_scheduler_fixture(tmp_path, fake_behavior="implementation-complete")
    assert run_once(state_root=fixture.state)["status"] == "RESULT_SUBMITTED"
    set_dispatch_enabled(fixture.dispatch_id, enabled=False, state_root=fixture.state)
    bootstrap_id = _latest_bootstrap_id(fixture.state)
    started = run_eiel([
        "sessionManager", "start", fixture.work_id,
        "--bootstrap-record", bootstrap_id, "--codex-id", "OLD-CODEX",
        "--repo-root", str(fixture.repo), "--role", "IMPLEMENTATION_AUTHORITY",
        "--allow-parallel", "--state-root", str(fixture.state),
    ], state_root=fixture.state)
    old_id = started["session"]["session_id"]
    run_eiel([
        "sessionManager", "interrupt", old_id, "--interrupted-during", "old work",
        "--next-action", "Resume old implementation", "--state-root", str(fixture.state),
    ], state_root=fixture.state)
    review_dispatch = authorize_dispatch({
        "work_id": fixture.work_id,
        "repository": {"repo_root": str(fixture.repo)},
        "allowed_authority_phases": ["review"],
        "worker_profile": {"launcher_type": "fake", "fake_behavior": "review-accept"},
    }, state_root=fixture.state)
    candidates = [row for row in __import__("tools.scheduler.dispatch", fromlist=["dispatch_candidates"]).dispatch_candidates(state_root=fixture.state) if row["dispatch_id"] == review_dispatch["record_id"]]
    assert candidates[0]["eligible"] is True
    assert candidates[0]["authority_phase"] == "review"
    assert candidates[0].get("existing_session_id") != old_id


def test_review_identity_overlap_starts_reduced_assurance_review(tmp_path: Path) -> None:
    from tools.scheduler.runtime import run_once
    fixture = make_scheduler_fixture(tmp_path, fake_behavior="implementation-complete")
    assert run_once(state_root=fixture.state)["status"] == "RESULT_SUBMITTED"
    implementation_sessions = [session_store(fixture.state).get(row["record_id"]) for row in session_store(fixture.state).list()]
    implementation = next(row for row in implementation_sessions if row["payload"].get("session_outcome") == "IMPLEMENTATION_COMPLETE")
    same_codex = implementation["payload"]["current_codex_id"]
    set_dispatch_enabled(fixture.dispatch_id, enabled=False, state_root=fixture.state)
    bootstrap_id = _latest_bootstrap_id(fixture.state)
    reduced = run_eiel([
        "sessionManager", "start", fixture.work_id,
        "--bootstrap-record", bootstrap_id, "--codex-id", same_codex,
        "--repo-root", str(fixture.repo), "--role", "REVIEW_AUTHORITY",
        "--review-target-session", implementation["record_id"],
        "--state-root", str(fixture.state),
    ], state_root=fixture.state)
    assert reduced["session"]["session_role"] == "REVIEW_AUTHORITY"


def test_scheduler_generated_at_requires_explicit_test_mode(tmp_path: Path) -> None:
    from tools.scheduler.cli import assignment_main
    assert assignment_main(["candidates", "--state-root", str(tmp_path), "--generated-at", "2026-08-05T00:00:00Z"]) == 2
    os.environ["EIEL_SCHEDULER_TEST_MODE"] = "1"
    try:
        assert assignment_main(["candidates", "--state-root", str(tmp_path), "--generated-at", "2026-08-05T00:00:00Z"]) == 0
    finally:
        os.environ.pop("EIEL_SCHEDULER_TEST_MODE", None)


def test_runtime_process_rows_are_forgotten_and_run_logs_are_bounded(tmp_path: Path) -> None:
    from tools.scheduler.runtime import load_runtime, run_once
    fixture = make_scheduler_fixture(tmp_path, fake_behavior="checkpoint")
    for index in range(3):
        event = run_once(state_root=fixture.state)
        assert event["status"] == "RESULT_SUBMITTED"
    runtime = load_runtime(fixture.state)
    assert runtime["processes"] == {}
    runs_root = fixture.state / "scheduler" / "runs"
    assert len([path for path in runs_root.iterdir() if path.is_dir()]) <= 20


def test_preexpiry_accepted_result_can_finish_finalization_after_lease_expiry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    assignment, lease, session, worker_id, thread_id = _activate_fake(fixture, generated_at="2026-08-05T00:00:00Z")
    run_eiel([
        "sessionManager", "checkpoint", session["record_id"], "--next-action", "Continue safely.",
        "--state-root", str(fixture.state),
    ], state_root=fixture.state)
    run_eiel([
        "sessionManager", "pause", session["record_id"], "--next-action", "Continue safely.",
        "--state-root", str(fixture.state),
    ], state_root=fixture.state)
    result = result_from_repository(
        assignment_id=assignment["record_id"], lease_id=lease["record_id"], state_root=fixture.state,
        worker_identity={"kind": "fake", "worker_id": worker_id, "pid": None, "thread_id": thread_id},
        outcome="CHECKPOINTED",
        started_at_utc="2026-08-05T00:00:00Z",
        ended_at_utc="2026-08-05T00:00:30Z",
    )

    import tools.scheduler.dispatch as dispatch
    real_finalize = dispatch._finalize_result_submission
    monkeypatch.setattr(dispatch, "_finalize_result_submission", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError):
        submit_worker_result(result, state_root=fixture.state, generated_at="2026-08-05T00:00:01Z")
    monkeypatch.setattr(dispatch, "_finalize_result_submission", real_finalize)

    expire_leases(state_root=fixture.state, generated_at="2026-08-05T02:00:00Z")
    stored, closed_assignment, closed_lease = submit_worker_result(
        result, state_root=fixture.state, generated_at="2026-08-05T02:00:01Z"
    )
    assert stored["record_id"]
    assert closed_assignment["status"] == "CLOSED"
    assert closed_lease["status"] == "CLOSED"


def test_worker_result_rejects_final_branch_change(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    assignment, lease, session, worker_id, thread_id = _activate_fake(fixture)
    run_eiel([
        "sessionManager", "checkpoint", session["record_id"], "--next-action", "Continue safely.",
        "--state-root", str(fixture.state),
    ], state_root=fixture.state)
    run_eiel([
        "sessionManager", "pause", session["record_id"], "--next-action", "Continue safely.",
        "--state-root", str(fixture.state),
    ], state_root=fixture.state)
    subprocess.run(["git", "switch", "-q", "-c", "unexpected-worker-branch"], cwd=fixture.repo, check=True)
    result = result_from_repository(
        assignment_id=assignment["record_id"], lease_id=lease["record_id"], state_root=fixture.state,
        worker_identity={"kind": "fake", "worker_id": worker_id, "pid": None, "thread_id": thread_id},
        outcome="CHECKPOINTED",
        started_at_utc="2026-08-05T00:00:00Z",
        ended_at_utc="2026-08-05T00:00:30Z",
    )
    with pytest.raises(ActorError):
        submit_worker_result(result, state_root=fixture.state)


@pytest.mark.parametrize("crash_point", ["assignment_result_submitted", "assignment_closed", "lease_close"])
def test_result_submission_retry_converges_after_each_terminal_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_point: str
) -> None:
    from tools.actors.store import ActorRecordStore

    fixture = make_scheduler_fixture(tmp_path)
    assignment, lease, session, worker_id, thread_id = _activate_fake(fixture)
    run_eiel([
        "sessionManager", "checkpoint", session["record_id"], "--next-action", "Continue safely.",
        "--state-root", str(fixture.state),
    ], state_root=fixture.state)
    run_eiel([
        "sessionManager", "pause", session["record_id"], "--next-action", "Continue safely.",
        "--state-root", str(fixture.state),
    ], state_root=fixture.state)
    result = result_from_repository(
        assignment_id=assignment["record_id"], lease_id=lease["record_id"], state_root=fixture.state,
        worker_identity={"kind": "fake", "worker_id": worker_id, "pid": None, "thread_id": thread_id},
        outcome="CHECKPOINTED",
        started_at_utc="2026-08-05T00:00:00Z",
        ended_at_utc="2026-08-05T00:00:30Z",
    )

    original_put = ActorRecordStore.put
    crashed = {"done": False}

    def crash_after_selected_write(self, record, *, allow_update=False):
        status = str(record.get("status") or "")
        should_crash = (
            (crash_point == "assignment_result_submitted" and self.namespace == "assignments" and status == "RESULT_SUBMITTED")
            or (crash_point == "assignment_closed" and self.namespace == "assignments" and status == "CLOSED")
            or (crash_point == "lease_close" and self.namespace == "leases" and status == "CLOSED")
        )
        path = original_put(self, record, allow_update=allow_update)
        if should_crash and not crashed["done"]:
            crashed["done"] = True
            raise RuntimeError(f"injected crash after {crash_point}")
        return path

    monkeypatch.setattr(ActorRecordStore, "put", crash_after_selected_write)
    with pytest.raises(RuntimeError):
        submit_worker_result(result, state_root=fixture.state, generated_at="2026-08-05T00:01:00Z")
    monkeypatch.setattr(ActorRecordStore, "put", original_put)

    stored, closed_assignment, closed_lease = submit_worker_result(
        result, state_root=fixture.state, generated_at="2026-08-05T00:01:01Z"
    )
    assert crashed["done"] is True
    assert stored["record_id"]
    assert closed_assignment["status"] == "CLOSED"
    assert closed_lease["status"] == "CLOSED"
    assert len(result_store(fixture.state).list()) == 1
