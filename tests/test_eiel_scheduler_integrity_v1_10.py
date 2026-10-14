from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tests.eiel_scheduler_v110_helpers import make_scheduler_fixture
from tools.actors.common import ActorError
from tools.actors.portfolio import session_store
from tools.scheduler.dispatch import (
    claim_assignment,
    compile_assignment,
    mark_assignment_running,
    submit_worker_result,
    validate_worker_result,
    worker_open,
)
from tools.scheduler.worker_protocol import result_from_repository


def _active(tmp_path: Path):
    fixture = make_scheduler_fixture(tmp_path)
    assignment = compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    claimed, lease = claim_assignment(assignment["record_id"], scheduler_id="SCH-INTEGRITY", state_root=fixture.state)
    worker = {"kind": "fake", "worker_id": "worker-integrity", "pid": 1}
    claimed, lease = mark_assignment_running(claimed["record_id"], lease["record_id"], worker, state_root=fixture.state)
    session = worker_open(claimed["record_id"], lease["record_id"], state_root=fixture.state, worker_id="worker-integrity", codex_id="codex-integrity")
    return fixture, claimed, lease, session


def _checkpoint_result(fixture, assignment, lease):
    session_id = assignment["payload"]["material"]["session"]["record_id"]
    from tests.eiel_actor_v11_helpers import run_eiel
    run_eiel(["sessionManager", "checkpoint", session_id, "--next-action", "continue", "--state-root", str(fixture.state)], state_root=fixture.state)
    run_eiel(["sessionManager", "pause", session_id, "--next-action", "continue", "--state-root", str(fixture.state)], state_root=fixture.state)
    return result_from_repository(
        assignment_id=assignment["record_id"], lease_id=lease["record_id"], state_root=fixture.state,
        worker_identity={"kind": "fake", "worker_id": "worker-integrity"}, outcome="CHECKPOINTED"
    )


def test_duplicate_claim_is_rejected(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    assignment = compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    claim_assignment(assignment["record_id"], scheduler_id="SCH-ONE", state_root=fixture.state)
    with pytest.raises(ActorError):
        claim_assignment(assignment["record_id"], scheduler_id="SCH-TWO", state_root=fixture.state)


def test_worker_result_binds_assignment_context_and_lease(tmp_path: Path) -> None:
    fixture, assignment, lease, _ = _active(tmp_path)
    result = _checkpoint_result(fixture, assignment, lease)
    validation = validate_worker_result(result, assignment=assignment, lease=lease, state_root=fixture.state)
    assert validation["valid"] is True


def test_tampered_assignment_digest_is_rejected(tmp_path: Path) -> None:
    fixture, assignment, lease, _ = _active(tmp_path)
    result = _checkpoint_result(fixture, assignment, lease)
    result["assignment_material_sha256"] = "0" * 64
    with pytest.raises(ActorError, match="integrity"):
        submit_worker_result(result, state_root=fixture.state)


def test_tampered_session_snapshot_is_rejected(tmp_path: Path) -> None:
    fixture, assignment, lease, _ = _active(tmp_path)
    result = _checkpoint_result(fixture, assignment, lease)
    result["session"]["revision_after"] += 1
    with pytest.raises(ActorError):
        submit_worker_result(result, state_root=fixture.state)


def test_wrong_worker_identity_is_rejected(tmp_path: Path) -> None:
    fixture, assignment, lease, _ = _active(tmp_path)
    result = _checkpoint_result(fixture, assignment, lease)
    result["worker_identity"]["worker_id"] = "impostor"
    with pytest.raises(ActorError):
        submit_worker_result(result, state_root=fixture.state)


def test_exact_duplicate_result_submission_is_idempotent(tmp_path: Path) -> None:
    fixture, assignment, lease, _ = _active(tmp_path)
    result = _checkpoint_result(fixture, assignment, lease)
    first, closed_assignment, closed_lease = submit_worker_result(result, state_root=fixture.state)
    second, same_assignment, same_lease = submit_worker_result(result, state_root=fixture.state)
    assert first["record_id"] == second["record_id"]
    assert closed_assignment["record_sha256"] == same_assignment["record_sha256"]
    assert closed_lease["record_sha256"] == same_lease["record_sha256"]
