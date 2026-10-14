from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.actors.common import ActorError, new_record, utc_now
from tools.actors.store import ActorRecordStore
from tests.eiel_scheduler_v110_helpers import make_scheduler_fixture
from tools.actors.portfolio import session_store, work_store
from tools.scheduler.contracts import add_seconds, assignment_store, lease_store, result_store
from tools.scheduler.dispatch import authorize_dispatch, set_dispatch_enabled
from tools.scheduler import runtime as scheduler_runtime
from tools.scheduler.runtime import _scheduler_execution_lock, load_runtime, run_once, scheduler_status, start_polling


def test_single_worker_execution_lock_rejects_second_scheduler(tmp_path: Path) -> None:
    state = tmp_path / "state"
    with _scheduler_execution_lock(state):
        with pytest.raises(ActorError) as excinfo:
            run_once(state_root=state)
    assert excinfo.value.code == "scheduler_already_running"


def test_run_once_executes_fake_worker_and_submits_structured_result(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path, fake_behavior="implementation-complete")
    event = run_once(state_root=fixture.state)
    assert event["status"] == "RESULT_SUBMITTED"
    assert event["assignment"]["status"] == "CLOSED"
    assert event["lease"]["status"] == "CLOSED"
    assert event["worker_result"]["payload"]["outcome"] == "COMPLETED"
    session_id = event["worker_result"]["payload"]["session_id"]
    assert session_store(fixture.state).get(session_id)["status"] == "COMPLETED"
    assert work_store(fixture.state).get(fixture.work_id)["status"] == "ACTIVE"


def test_run_once_persists_three_lease_attempt_binding_before_provider_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    set_dispatch_enabled(fixture.dispatch_id, enabled=False, state_root=fixture.state)
    now = utc_now()
    resources = ActorRecordStore(fixture.state, "resources")
    observation = new_record(
        actor="resourceManager",
        record_type="resource_observation",
        record_id="ROBS-RUNTIME-ATTEMPT",
        status="OBSERVED",
        payload={
            "report": {
                "schema_version": "eiel_resource_monitor_v1",
                "generated_at_utc": now,
            }
        },
        generated_at=now,
    )
    resources.put(observation)
    recommendation = new_record(
        actor="resourceManager",
        record_type="resource_admission_recommendation",
        record_id="RADM-RUNTIME-ATTEMPT",
        status="ADMIT",
        payload={"observation_id": observation["record_id"], "decision": "ADMIT"},
        generated_at=now,
    )
    resources.put(recommendation)
    attempt_id = "ATTEMPT-RUNTIME-THREE-LEASES"
    continuation = {
        "schema_version": "crca_canonical_continuation_lease_v1",
        "lease_id": "CONTINUATION-RUNTIME-THREE-LEASES",
        "run_id": "RUN-RUNTIME-THREE-LEASES",
        "attempt_id": attempt_id,
        "restore_checkpoint_digest": None,
        "lineage_scope_digest": {"algorithm": "sha256", "hex": "5" * 64},
        "owner_id": "runtime-owner",
        "lease_epoch": 1,
        "fencing_token_digest": {"algorithm": "sha256", "hex": "6" * 64},
        "prior_lease_digest": None,
        "granted_at_utc": now,
        "expires_at_utc": add_seconds(now, 3600),
    }
    dispatch = authorize_dispatch(
        {
            "work_id": fixture.work_id,
            "enabled": True,
            "allowed_authority_phases": ["implementation"],
            "worker_profile": {
                "launcher_type": "fake",
                "fake_behavior": "checkpoint",
                "sandbox": "workspace-write",
                "ephemeral": True,
            },
            "repository": {"repo_root": str(fixture.repo)},
            "resource_admission_required": True,
            "resource_admission": {
                "schema_version": "crca_scheduler_resource_admission_binding_v1",
                "attempt_admission_request_sha256": "1" * 64,
                "resource_scope_sha256": "2" * 64,
                "resource_ids": ["host:cpu"],
                "resource_observation_id": observation["record_id"],
                "resource_observation_record_sha256": observation["record_sha256"],
                "resource_recommendation_id": recommendation["record_id"],
                "resource_recommendation_record_sha256": recommendation[
                    "record_sha256"
                ],
                "start_window_seconds": 60,
                "ttl_seconds": 120,
                "max_observation_age_seconds": 300,
                "attempt_lease_admission": {
                    "schema_version": "crca_scheduler_attempt_lease_admission_v1",
                    "attempt_id": attempt_id,
                    "canonical_continuation_lease": continuation,
                },
            },
            "lease_policy": {
                "duration_seconds": 3600,
                "heartbeat_interval_seconds": 5,
            },
            "created_by": "HUMAN_OPERATOR",
        },
        state_root=fixture.state,
    )
    observed_before_launcher = False
    original_launcher = scheduler_runtime._launcher_for_assignment

    def require_binding_before_launcher(assignment):
        nonlocal observed_before_launcher
        binding_root = (
            fixture.state / "scheduler" / "attempt-lease-bindings" / attempt_id
        )
        observed_before_launcher = len(list(binding_root.glob("*.json"))) == 1
        assert observed_before_launcher
        return original_launcher(assignment)

    monkeypatch.setattr(
        scheduler_runtime, "_launcher_for_assignment", require_binding_before_launcher
    )
    event = run_once(state_root=fixture.state, dispatch_id=str(dispatch["record_id"]))
    assert event["status"] == "RESULT_SUBMITTED"
    assert observed_before_launcher is True
    bindings = list(
        (fixture.state / "scheduler" / "attempt-lease-bindings" / attempt_id).glob(
            "*.json"
        )
    )
    assert len(bindings) == 1
    payload = json.loads(bindings[0].read_text(encoding="utf-8"))
    assert payload["attempt_id"] == attempt_id
    assert payload["scheduler_assignment_lease_id"]
    assert payload["resource_start_lease_id"]
    assert payload["canonical_continuation_lease_id"] == continuation["lease_id"]


def test_run_once_with_no_dispatch_is_clean_idle(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    from tools.scheduler.dispatch import set_dispatch_enabled
    set_dispatch_enabled(fixture.dispatch_id, enabled=False, state_root=fixture.state)
    event = run_once(state_root=fixture.state)
    assert event["status"] == "NO_ASSIGNMENT"
    assert load_runtime(fixture.state)["counters"]["idle_cycles"] == 1


def test_worker_process_failure_records_loss_and_preserves_work(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path, fake_behavior="crash")
    event = run_once(state_root=fixture.state)
    assert event["status"] == "PROCESS_FAILURE"
    assert event["assignment"]["status"] == "EXPIRED"
    assert event["lease"]["status"] == "EXPIRED"
    sessions = [session_store(fixture.state).get(row["record_id"]) for row in session_store(fixture.state).list()]
    assert sessions[0]["status"] == "INTERRUPTED"
    assert work_store(fixture.state).get(fixture.work_id)["status"] == "ACTIVE"


def test_polling_scheduler_processes_one_assignment_then_stops(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path, fake_behavior="checkpoint")
    result = start_polling(state_root=fixture.state, poll_interval_seconds=0.1, max_assignments=1, max_idle_cycles=1)
    assert result["status"] == "STOPPED"
    assert result["assignments_processed"] == 1
    assert load_runtime(fixture.state)["status"] == "STOPPED"


def test_scheduler_status_is_advisory_and_manual_path_unblocked(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    status = scheduler_status(state_root=fixture.state)
    assert status["optionality_audit"] == "NOT_MEASURED"
    assert status["records"]["worker_results"] == 0
