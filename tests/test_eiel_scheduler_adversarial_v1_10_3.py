from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.eiel_actor_v11_helpers import run_eiel
from tests.eiel_scheduler_v110_helpers import make_scheduler_fixture
from tools.actors.common import ActorError, atomic_write_json
from tools.actors.portfolio import session_store
from tools.scheduler.contracts import assignment_store, lease_store
from tools.scheduler.dispatch import (
    claim_assignment,
    compile_assignment,
    expire_leases,
    mark_assignment_running,
    submit_worker_result,
    worker_open,
)
from tools.scheduler.launcher import _start_wrapped_process, process_start_identity, terminate_process_group
from tools.scheduler.runtime import load_runtime, reconcile, save_runtime, scheduler_status
from tools.scheduler.worker_protocol import result_from_repository, write_worker_result


def _running_checkpointed(tmp_path: Path, *, generated: bool = False):
    fixture = make_scheduler_fixture(tmp_path)
    assignment = compile_assignment(
        fixture.dispatch_id,
        state_root=fixture.state,
        generated_at="2026-08-05T00:00:00Z" if generated else None,
    )
    claimed, lease = claim_assignment(
        assignment["record_id"],
        scheduler_id="SCH-RESULT-1103",
        state_root=fixture.state,
        generated_at="2026-08-05T00:00:01Z" if generated else None,
    )
    claimed, lease = mark_assignment_running(
        claimed["record_id"],
        lease["record_id"],
        {"kind": "fake", "worker_id": "result-worker"},
        state_root=fixture.state,
        generated_at="2026-08-05T00:00:02Z" if generated else None,
    )
    worker_open(
        claimed["record_id"],
        lease["record_id"],
        state_root=fixture.state,
        worker_id="result-worker",
        codex_id="result-codex",
        generated_at="2026-08-05T00:00:03Z" if generated else None,
    )
    session_id = claimed["payload"]["material"]["session"]["record_id"]
    run_eiel(
        ["sessionManager", "checkpoint", session_id, "--next-action", "continue", "--state-root", str(fixture.state)],
        state_root=fixture.state,
    )
    run_eiel(
        ["sessionManager", "pause", session_id, "--next-action", "continue", "--state-root", str(fixture.state)],
        state_root=fixture.state,
    )
    return fixture, claimed, lease


def _dead_runtime_row(fixture, assignment, lease, result_path: Path) -> None:
    runtime = load_runtime(fixture.state)
    runtime["processes"] = {
        assignment["record_id"]: {
            "assignment_id": assignment["record_id"],
            "lease_id": lease["record_id"],
            "worker_identity": {"pid": 999999, "process_start_identity": "missing"},
            "result_path": str(result_path),
            "status": "RUNNING",
        }
    }
    save_runtime(fixture.state, runtime)


def test_preexpiry_result_submitted_after_expiry_is_accepted(tmp_path: Path) -> None:
    fixture, assignment, lease = _running_checkpointed(tmp_path, generated=True)
    result = result_from_repository(
        assignment_id=assignment["record_id"],
        lease_id=lease["record_id"],
        state_root=fixture.state,
        worker_identity={"kind": "fake", "worker_id": "result-worker"},
        outcome="CHECKPOINTED",
        started_at_utc="2026-08-05T00:00:04Z",
        ended_at_utc="2026-08-05T00:59:59Z",
    )
    result_path = Path(assignment["payload"]["material"]["result_contract"]["result_path"])
    write_worker_result(result_path, result)
    expire_leases(state_root=fixture.state, generated_at="2026-08-05T02:00:00Z")
    assert assignment_store(fixture.state).get(assignment["record_id"])["status"] == "EXPIRED"
    _dead_runtime_row(fixture, assignment, lease, result_path)
    outcome = reconcile(state_root=fixture.state)
    assert any(row["status"] == "RESULT_SUBMITTED" for row in outcome["outcomes"]), outcome
    assert assignment_store(fixture.state).get(assignment["record_id"])["status"] == "CLOSED"
    assert lease_store(fixture.state).get(lease["record_id"])["status"] == "CLOSED"


def test_result_ending_exactly_at_expiry_is_rejected(tmp_path: Path) -> None:
    fixture, assignment, lease = _running_checkpointed(tmp_path, generated=True)
    expires_at = lease["payload"]["expires_at_utc"]
    result = result_from_repository(
        assignment_id=assignment["record_id"],
        lease_id=lease["record_id"],
        state_root=fixture.state,
        worker_identity={"kind": "fake", "worker_id": "result-worker"},
        outcome="CHECKPOINTED",
        started_at_utc="2026-08-05T00:00:04Z",
        ended_at_utc=expires_at,
    )
    with pytest.raises(ActorError) as exc:
        submit_worker_result(result, state_root=fixture.state, generated_at="2026-08-05T02:00:00Z")
    assert exc.value.code == "worker_result_invalid"
    assert any("before Lease expiry" in problem for problem in exc.value.problems)


def test_partial_result_is_pending_then_converges(tmp_path: Path) -> None:
    fixture, assignment, lease = _running_checkpointed(tmp_path)
    result = result_from_repository(
        assignment_id=assignment["record_id"], lease_id=lease["record_id"], state_root=fixture.state,
        worker_identity={"kind": "fake", "worker_id": "result-worker"}, outcome="CHECKPOINTED",
    )
    result_path = Path(assignment["payload"]["material"]["result_contract"]["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, sort_keys=True).encode("utf-8")
    result_path.write_bytes(encoded[: len(encoded) // 2])
    _dead_runtime_row(fixture, assignment, lease, result_path)
    first = reconcile(state_root=fixture.state)
    assert any(row["status"] == "RESULT_PENDING" for row in first["outcomes"])
    atomic_write_json(result_path, result)
    second = reconcile(state_root=fixture.state)
    assert any(row["status"] == "RESULT_SUBMITTED" for row in second["outcomes"])


def test_stable_invalid_result_is_preserved_and_becomes_loss(tmp_path: Path) -> None:
    fixture, assignment, lease = _running_checkpointed(tmp_path)
    result_path = Path(assignment["payload"]["material"]["result_contract"]["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("{not-json}\n", encoding="utf-8")
    _dead_runtime_row(fixture, assignment, lease, result_path)
    first = reconcile(state_root=fixture.state)
    assert any(row["status"] == "RESULT_PENDING" for row in first["outcomes"])
    second = reconcile(state_root=fixture.state)
    assert any(row["status"] == "PROCESS_FAILURE" for row in second["outcomes"])
    assert result_path.is_file()
    assert assignment_store(fixture.state).get(assignment["record_id"])["status"] == "EXPIRED"
    details = dict(load_runtime(fixture.state)["last_event"].get("details") or {})
    assert details["result_path"] == str(result_path)
    assert details["errors"]


def test_zero_byte_result_can_be_replaced_by_valid_result(tmp_path: Path) -> None:
    fixture, assignment, lease = _running_checkpointed(tmp_path)
    result = result_from_repository(
        assignment_id=assignment["record_id"], lease_id=lease["record_id"], state_root=fixture.state,
        worker_identity={"kind": "fake", "worker_id": "result-worker"}, outcome="CHECKPOINTED",
    )
    result_path = Path(assignment["payload"]["material"]["result_contract"]["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_bytes(b"")
    _dead_runtime_row(fixture, assignment, lease, result_path)
    assert any(row["status"] == "RESULT_PENDING" for row in reconcile(state_root=fixture.state)["outcomes"])
    write_worker_result(result_path, result)
    assert any(row["status"] == "RESULT_SUBMITTED" for row in reconcile(state_root=fixture.state)["outcomes"])


def test_thread_identity_sidecar_survives_log_rotation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_SCHEDULER_TEST_MODE", "1")
    logs = tmp_path / "logs"
    thread_id = "THREAD-DURABLE-1103"
    code = (
        "import json,sys\n"
        f"print(json.dumps({{'type':'thread.started','thread_id':{thread_id!r}}}), flush=True)\n"
        "for i in range(12000): print(json.dumps({'type':'event','payload':'x'*256}), flush=True)\n"
    )
    proc, identity, _handshake, _ack = _start_wrapped_process(
        target_command=[sys.executable, "-c", code],
        logs=logs,
        stdout_path=logs / "events.jsonl",
        stderr_path=logs / "stderr.log",
        identity={"kind": "codex_exec", "worker_id": "sidecar", "thread_id": "PROVISIONAL"},
        on_started=None,
        maximum_log_bytes=32 * 1024,
    )
    assert proc.wait(timeout=20) == 0
    terminate_process_group(proc, grace_seconds=0.2)
    sidecar = json.loads(Path(identity["thread_identity_path"]).read_text(encoding="utf-8"))
    assert sidecar["status"] == "BOUND"
    assert sidecar["thread_id"] == thread_id
    assert (logs / "events.jsonl").stat().st_size <= 32 * 1024


def test_conflicting_thread_id_sidecar_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_SCHEDULER_TEST_MODE", "1")
    logs = tmp_path / "logs"
    code = (
        "import json\n"
        "print(json.dumps({'type':'thread.started','thread_id':'T-1'}), flush=True)\n"
        "print(json.dumps({'type':'thread.started','thread_id':'T-2'}), flush=True)\n"
    )
    proc, identity, _handshake, _ack = _start_wrapped_process(
        target_command=[sys.executable, "-c", code], logs=logs,
        stdout_path=logs / "events.jsonl", stderr_path=logs / "stderr.log",
        identity={"kind": "codex_exec", "worker_id": "conflict", "thread_id": "PROVISIONAL"},
        on_started=None,
    )
    assert proc.wait(timeout=10) == 125
    terminate_process_group(proc, grace_seconds=0.2)
    sidecar = json.loads(Path(identity["thread_identity_path"]).read_text(encoding="utf-8"))
    assert sidecar["status"] == "CONFLICT"


def test_direct_launcher_without_activation_callback_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EIEL_SCHEDULER_TEST_MODE", raising=False)
    marker = tmp_path / "executed.txt"
    logs = tmp_path / "logs"
    with pytest.raises(ActorError) as exc:
        _start_wrapped_process(
            target_command=[sys.executable, "-c", f"open({str(marker)!r},'w').write('bad')"],
            logs=logs, stdout_path=logs / "stdout.log", stderr_path=logs / "stderr.log",
            identity={"kind": "fake", "worker_id": "no-callback", "thread_id": "FAKE"},
            on_started=None,
        )
    assert exc.value.code == "worker_launch_activation_required"
    time.sleep(0.1)
    assert not marker.exists()


def test_wrapper_rejects_wrong_ack_contract(tmp_path: Path) -> None:
    marker = tmp_path / "executed.txt"
    logs = tmp_path / "logs"

    def wrong_ack(bound):
        ack = Path(bound["ack_path"])
        atomic_write_json(ack, {
            "schema_version": "eiel_scheduler_launch_ack_v1",
            "assignment_id": "WRONG",
            "lease_id": bound.get("lease_id") or "",
            "launch_nonce": bound.get("launch_nonce") or "",
        })

    proc, _identity, _handshake, _ack = _start_wrapped_process(
        target_command=[sys.executable, "-c", f"open({str(marker)!r},'w').write('bad')"],
        logs=logs, stdout_path=logs / "stdout.log", stderr_path=logs / "stderr.log",
        identity={
            "kind": "fake", "worker_id": "wrong-ack", "thread_id": "FAKE",
            "assignment_id": "ASSIGN-ACK", "lease_id": "LEASE-ACK", "launch_nonce": "NONCE-ACK",
        },
        on_started=wrong_ack,
    )
    assert proc.wait(timeout=10) == 126
    terminate_process_group(proc, grace_seconds=0.2)
    assert not marker.exists()


def test_scheduler_status_reports_lifecycle_buckets(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    status = scheduler_status(state_root=fixture.state)
    records = status["records"]
    for key in (
        "claimable_assignments", "executing_assignments", "recovery_required_assignments",
        "result_pending_assignments",
    ):
        assert key in records
    assert records["claimable_assignments"] == 1


def test_launcher_host_capability_fails_clearly_off_linux(tmp_path: Path) -> None:
    from tools.scheduler.launcher import launcher_host_capability

    capability = launcher_host_capability(platform_name="darwin", proc_root=tmp_path)
    assert capability["ok"] is False
    assert "require Linux" in capability["reason"]


def test_scheduler_modules_remain_optional_and_domain_bounded() -> None:
    scheduler = Path("tools/scheduler")
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in scheduler.glob("*.py")
    )
    for forbidden in (
        "engine.simulation",
        "engine.evolution",
        "engine.env",
        "tools.actors.mission_planner",
        "tools.actors.evidence_manager",
        "tools.actors.qualification_manager",
    ):
        assert forbidden not in source
    runtime = (scheduler / "runtime.py").read_text(encoding="utf-8")
    assert "from tools.scheduler.result_settlement import" in runtime
    assert "def _result_fingerprint" not in runtime
    assert not (scheduler / "assignment_manager.py").exists()
    assert not (scheduler / "recovery_daemon.py").exists()


def test_unbound_runtime_row_never_kills_unowned_process_group(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    try:
        runtime = load_runtime(fixture.state)
        runtime["processes"] = {
            "ASSIGN-UNBOUND": {
                "assignment_id": "ASSIGN-UNBOUND",
                "lease_id": "LEASE-UNBOUND",
                "worker_identity": {
                    "pid": proc.pid,
                    "process_start_identity": process_start_identity(proc.pid),
                },
                "result_path": str(fixture.state / "missing-unbound-result.json"),
                "status": "RUNNING",
            }
        }
        save_runtime(fixture.state, runtime)
        outcome = reconcile(state_root=fixture.state)
        assert outcome["outcomes"][0]["status"] == "PROCESS_FAILURE"
        assert proc.poll() is None
    finally:
        terminate_process_group(proc)
