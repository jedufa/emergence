from __future__ import annotations

from pathlib import Path

from tests.eiel_actor_v11_helpers import run_eiel
from tests.eiel_scheduler_v110_helpers import make_scheduler_fixture
from tools.actors.common import read_json
from tools.actors.portfolio import session_store
from tools.scheduler.contracts import assignment_store, lease_store
from tools.scheduler.dispatch import (
    claim_assignment,
    compile_assignment,
    expire_leases,
    mark_assignment_running,
    worker_open,
)
from tools.scheduler.launcher import process_start_identity
from tools.scheduler.runtime import load_runtime, reconcile, save_runtime
from tools.scheduler.worker_protocol import result_from_repository, write_worker_result


def test_lease_expiry_records_worker_loss(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    assignment = compile_assignment(fixture.dispatch_id, state_root=fixture.state, generated_at="2026-08-05T00:00:00Z")
    claimed, lease = claim_assignment(
        assignment["record_id"], scheduler_id="SCH-EXPIRY", state_root=fixture.state,
        generated_at="2026-08-05T00:00:01Z",
    )
    mark_assignment_running(
        claimed["record_id"], lease["record_id"], {"kind": "fake", "worker_id": "expiry-worker"},
        state_root=fixture.state, generated_at="2026-08-05T00:00:02Z",
    )
    worker_open(
        claimed["record_id"], lease["record_id"], state_root=fixture.state,
        worker_id="expiry-worker", codex_id="expiry-codex", generated_at="2026-08-05T00:00:03Z",
    )
    expired = expire_leases(state_root=fixture.state, generated_at="2026-08-05T02:00:00Z")
    assert len(expired) == 1
    assert lease_store(fixture.state).get(lease["record_id"])["status"] == "EXPIRED"
    assert assignment_store(fixture.state).get(assignment["record_id"])["status"] == "EXPIRED"


def test_reconcile_marks_vanished_worker_as_loss(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    assignment = compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    claimed, lease = claim_assignment(assignment["record_id"], scheduler_id="SCH-RECON", state_root=fixture.state)
    mark_assignment_running(claimed["record_id"], lease["record_id"], {"kind": "fake", "worker_id": "lost", "pid": 999999, "process_start_identity": "missing"}, state_root=fixture.state)
    runtime = load_runtime(fixture.state)
    result_path = Path(claimed["payload"]["material"]["result_contract"]["result_path"])
    runtime["processes"] = {
        claimed["record_id"]: {
            "assignment_id": claimed["record_id"], "lease_id": lease["record_id"],
            "worker_identity": {"pid": 999999, "process_start_identity": "missing"},
            "result_path": str(result_path), "status": "RUNNING",
        }
    }
    save_runtime(fixture.state, runtime)
    outcome = reconcile(state_root=fixture.state)
    assert outcome["outcomes"][0]["status"] == "PROCESS_FAILURE"
    session_id = claimed["payload"]["material"]["session"]["record_id"]
    assert session_store(fixture.state).get(session_id)["status"] == "INTERRUPTED"


def test_reconcile_rejects_unbound_live_process(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    import os
    runtime = load_runtime(fixture.state)
    runtime["processes"] = {
        "ASSIGN-LIVE": {
            "assignment_id": "ASSIGN-LIVE", "lease_id": "LEASE-LIVE",
            "worker_identity": {"pid": os.getpid(), "process_start_identity": process_start_identity(os.getpid())},
            "result_path": str(fixture.state / "missing-result.json"), "status": "RUNNING",
        }
    }
    save_runtime(fixture.state, runtime)
    outcome = reconcile(state_root=fixture.state)
    assert outcome["outcomes"][0]["status"] == "PROCESS_FAILURE"


def test_reconcile_submits_result_written_before_scheduler_restart(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    assignment = compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    claimed, lease = claim_assignment(assignment["record_id"], scheduler_id="SCH-RECON", state_root=fixture.state)
    claimed, lease = mark_assignment_running(claimed["record_id"], lease["record_id"], {"kind": "fake", "worker_id": "reconcile-worker"}, state_root=fixture.state)
    worker_open(claimed["record_id"], lease["record_id"], state_root=fixture.state, worker_id="reconcile-worker", codex_id="reconcile-codex")
    session_id = claimed["payload"]["material"]["session"]["record_id"]
    run_eiel(["sessionManager", "checkpoint", session_id, "--next-action", "continue", "--state-root", str(fixture.state)], state_root=fixture.state)
    run_eiel(["sessionManager", "pause", session_id, "--next-action", "continue", "--state-root", str(fixture.state)], state_root=fixture.state)
    result = result_from_repository(
        assignment_id=claimed["record_id"], lease_id=lease["record_id"], state_root=fixture.state,
        worker_identity={"kind": "fake", "worker_id": "reconcile-worker"}, outcome="CHECKPOINTED",
    )
    result_path = Path(claimed["payload"]["material"]["result_contract"]["result_path"])
    write_worker_result(result_path, result)
    runtime = load_runtime(fixture.state)
    runtime["processes"] = {
        claimed["record_id"]: {
            "assignment_id": claimed["record_id"], "lease_id": lease["record_id"],
            "worker_identity": {"pid": 999999, "process_start_identity": "missing"},
            "result_path": str(result_path), "status": "RUNNING",
        }
    }
    save_runtime(fixture.state, runtime)
    outcome = reconcile(state_root=fixture.state)
    assert outcome["outcomes"][0]["status"] == "RESULT_SUBMITTED"
    assert assignment_store(fixture.state).get(claimed["record_id"])["status"] == "CLOSED"
