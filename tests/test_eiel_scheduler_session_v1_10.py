from __future__ import annotations

from pathlib import Path

import pytest

from tests.eiel_scheduler_v110_helpers import fresh_bootstrap, make_scheduler_fixture
from tools.actors.common import ActorError
from tools.actors.portfolio import session_store
from tools.scheduler.dispatch import (
    claim_assignment,
    compile_assignment,
    mark_assignment_running,
    record_execution_loss,
    worker_open,
)


def _prepared(tmp_path: Path):
    fixture = make_scheduler_fixture(tmp_path)
    assignment = compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    session_id = assignment["payload"]["material"]["session"]["record_id"]
    return fixture, assignment, session_id


def test_prepared_session_has_no_fabricated_codex_identity(tmp_path: Path) -> None:
    fixture, assignment, session_id = _prepared(tmp_path)
    session = session_store(fixture.state).get(session_id)
    payload = session["payload"]
    assert session["status"] == "CREATED"
    assert payload["worker_identity_state"] == "PENDING"
    assert payload["original_codex_id"] is None
    assert payload["current_codex_id"] is None
    assert payload["codex_participants"] == []
    assert payload["prepared_assignment_id"] == assignment["record_id"]


def test_worker_activation_binds_real_assignment_lease_and_identity(tmp_path: Path) -> None:
    fixture, assignment, session_id = _prepared(tmp_path)
    claimed, lease = claim_assignment(assignment["record_id"], scheduler_id="SCH-TEST", state_root=fixture.state)
    mark_assignment_running(claimed["record_id"], lease["record_id"], {"kind": "fake", "worker_id": "worker-1", "pid": 1}, state_root=fixture.state)
    active = worker_open(claimed["record_id"], lease["record_id"], state_root=fixture.state, worker_id="worker-1", codex_id="codex-1")
    payload = active["payload"]
    assert active["status"] == "ACTIVE"
    assert payload["scheduler_assignment_id"] == claimed["record_id"]
    assert payload["scheduler_lease_id"] == lease["record_id"]
    assert payload["scheduler_worker_id"] == "worker-1"
    assert payload["current_codex_id"] == "codex-1"


def test_worker_open_rejects_identity_that_does_not_match_active_lease(tmp_path: Path) -> None:
    fixture, assignment, _ = _prepared(tmp_path)
    claimed, lease = claim_assignment(assignment["record_id"], scheduler_id="SCH-TEST", state_root=fixture.state)
    mark_assignment_running(claimed["record_id"], lease["record_id"], {"kind": "fake", "worker_id": "worker-1"}, state_root=fixture.state)
    with pytest.raises(ActorError) as excinfo:
        worker_open(claimed["record_id"], lease["record_id"], state_root=fixture.state, worker_id="worker-2", codex_id="codex-2")
    assert excinfo.value.code == "lease_worker_mismatch"


def test_activation_is_idempotent_for_exact_same_worker(tmp_path: Path) -> None:
    fixture, assignment, _ = _prepared(tmp_path)
    claimed, lease = claim_assignment(assignment["record_id"], scheduler_id="SCH-TEST", state_root=fixture.state)
    mark_assignment_running(claimed["record_id"], lease["record_id"], {"kind": "fake", "worker_id": "worker-1"}, state_root=fixture.state)
    first = worker_open(claimed["record_id"], lease["record_id"], state_root=fixture.state, worker_id="worker-1", codex_id="codex-1")
    second = worker_open(claimed["record_id"], lease["record_id"], state_root=fixture.state, worker_id="worker-1", codex_id="codex-1")
    assert first["record_sha256"] == second["record_sha256"]


def test_worker_loss_interrupts_session_without_changing_work(tmp_path: Path) -> None:
    fixture, assignment, session_id = _prepared(tmp_path)
    claimed, lease = claim_assignment(assignment["record_id"], scheduler_id="SCH-TEST", state_root=fixture.state)
    mark_assignment_running(claimed["record_id"], lease["record_id"], {"kind": "fake", "worker_id": "worker-1"}, state_root=fixture.state)
    worker_open(claimed["record_id"], lease["record_id"], state_root=fixture.state, worker_id="worker-1", codex_id="codex-1")
    expired_assignment, expired_lease = record_execution_loss(claimed["record_id"], lease["record_id"], state_root=fixture.state, reason="TEST_LOSS")
    session = session_store(fixture.state).get(session_id)
    assert expired_assignment["status"] == "EXPIRED"
    assert expired_lease["status"] == "EXPIRED"
    assert session["status"] == "INTERRUPTED"
    assert session["payload"]["worker_loss"]["reason"] == "TEST_LOSS"


def test_fresh_assignment_can_resume_interrupted_session(tmp_path: Path) -> None:
    fixture, assignment, session_id = _prepared(tmp_path)
    claimed, lease = claim_assignment(assignment["record_id"], scheduler_id="SCH-TEST", state_root=fixture.state)
    mark_assignment_running(claimed["record_id"], lease["record_id"], {"kind": "fake", "worker_id": "worker-1"}, state_root=fixture.state)
    worker_open(claimed["record_id"], lease["record_id"], state_root=fixture.state, worker_id="worker-1", codex_id="codex-1")
    record_execution_loss(claimed["record_id"], lease["record_id"], state_root=fixture.state, reason="TEST_LOSS")
    # Repository HEAD did not change, so the current Bootstrap remains valid.
    resumed = compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    assert resumed["record_id"] != assignment["record_id"]
    assert resumed["payload"]["material"]["session"]["record_id"] == session_id
    assert resumed["payload"]["material"]["session"]["action"] == "RESUME"


def test_wrong_worker_cannot_take_over_active_session(tmp_path: Path) -> None:
    fixture, assignment, _ = _prepared(tmp_path)
    claimed, lease = claim_assignment(assignment["record_id"], scheduler_id="SCH-TEST", state_root=fixture.state)
    mark_assignment_running(claimed["record_id"], lease["record_id"], {"kind": "fake", "worker_id": "worker-1"}, state_root=fixture.state)
    worker_open(claimed["record_id"], lease["record_id"], state_root=fixture.state, worker_id="worker-1", codex_id="codex-1")
    with pytest.raises(ActorError):
        worker_open(claimed["record_id"], lease["record_id"], state_root=fixture.state, worker_id="worker-2", codex_id="codex-2")
