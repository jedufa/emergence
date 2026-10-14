from __future__ import annotations

from pathlib import Path

import pytest

from tests.eiel_scheduler_v110_helpers import make_scheduler_fixture
from tools.actors.common import ActorError
from tools.actors.portfolio import session_store, work_store
from tools.scheduler.contracts import (
    assignment_store,
    dispatch_store,
    normalize_dispatch_payload,
)
from tools.scheduler.dispatch import (
    authorize_dispatch,
    compile_assignment,
    dispatch_candidates,
    recover_preparing,
    request_assignment,
    set_dispatch_enabled,
)


def test_assignment_linker_compiles_exact_repository_truth(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    assignment = compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    payload = assignment["payload"]
    material = payload["material"]
    assert assignment["status"] == "READY"
    assert material["work"]["record_id"] == fixture.work_id
    assert material["work"]["work_package_binding"]["sha256"]
    assert material["session"]["status"] == "CREATED"
    assert material["repository"]["expected_head"]
    assert material["bootstrap_expectations"]["verify_only"] is True
    assert material["result_contract"]["prose_parsing_allowed"] is False
    assert payload["assignment_material_sha256"]


def test_compilation_is_idempotent_for_same_truth(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    first = compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    second = compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    assert first["record_id"] == second["record_id"]
    assert first["record_sha256"] == second["record_sha256"]
    assert len(assignment_store(fixture.state).list()) == 1


def test_no_dispatch_means_no_assignment(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    set_dispatch_enabled(fixture.dispatch_id, enabled=False, state_root=fixture.state)
    assert request_assignment(state_root=fixture.state) is None
    candidates = dispatch_candidates(state_root=fixture.state)
    assert candidates[0]["eligible"] is False
    assert "dispatch_disabled" in candidates[0]["reason_codes"]


def test_queue_priority_deterministically_selects_candidate(tmp_path: Path) -> None:
    first = make_scheduler_fixture(tmp_path / "a", work_id="WORK-SCHED-A", queue_priority=200)
    # Both repos intentionally share one state root for the portfolio test.
    second_repo = first.repo.parent / "second"
    # Reuse helper with a separate tmp then reauthorize into first state is needlessly theatrical;
    # priority ordering itself is covered by two Dispatch Requests on the same Work.
    high = authorize_dispatch(
        {
            "work_id": first.work_id,
            "repository": {"repo_root": str(first.repo)},
            "allowed_authority_phases": ["implementation"],
            "queue": {"priority": 5},
            "worker_profile": {"launcher_type": "fake"},
        },
        state_root=first.state,
    )
    candidates = dispatch_candidates(state_root=first.state)
    assert candidates[0]["dispatch_id"] == high["record_id"]
    assert candidates[0]["queue_priority"] == 5


def test_preparing_recovery_does_not_duplicate_session(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    assignment = compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    session_id = assignment["payload"]["material"]["session"]["record_id"]
    before = len(session_store(fixture.state).list())
    recovered = recover_preparing(state_root=fixture.state)
    assert recovered == []
    assert len(session_store(fixture.state).list()) == before
    assert session_store(fixture.state).exists(session_id)


def test_linker_fails_closed_when_bootstrap_head_is_stale(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    (fixture.repo / "README.changed.md").write_text("head drift\n", encoding="utf-8")
    import subprocess
    subprocess.run(["git", "add", "README.changed.md"], cwd=fixture.repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "head drift"], cwd=fixture.repo, check=True)
    with pytest.raises(ActorError) as caught:
        compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    assert caught.value.code == "dispatch_bootstrap_missing"


def test_resource_admission_requires_versioned_exact_scheduler_binding(tmp_path: Path) -> None:
    with pytest.raises(ActorError) as caught:
        authorize_dispatch(
            {
                "work_id": "WORK-RESOURCE-DEFERRED",
                "repository": {"repo_root": str(tmp_path)},
                "resource_admission_required": True,
            },
            state_root=tmp_path / "state",
        )
    assert caught.value.code == "scheduler_resource_admission_binding_invalid"

    exact = normalize_dispatch_payload(
        {
            "work_id": "WORK-RESOURCE-BOUND",
            "repository": {"repo_root": str(tmp_path)},
            "resource_admission_required": True,
            "resource_admission": {
                "schema_version": "crca_scheduler_resource_admission_binding_v1",
                "attempt_admission_request_sha256": "1" * 64,
                "resource_scope_sha256": "2" * 64,
                "resource_ids": ["host:cpu"],
                "resource_observation_id": "ROBS-EXACT",
                "resource_observation_record_sha256": "3" * 64,
                "resource_recommendation_id": "RADM-EXACT",
                "resource_recommendation_record_sha256": "4" * 64,
                "start_window_seconds": 60,
                "ttl_seconds": 120,
                "max_observation_age_seconds": 300,
                "attempt_lease_admission": {
                    "schema_version": "crca_scheduler_attempt_lease_admission_v1",
                    "attempt_id": "ATTEMPT-RESOURCE-BOUND",
                    "canonical_continuation_lease": {
                        "schema_version": "crca_canonical_continuation_lease_v1",
                        "lease_id": "CONTINUATION-RESOURCE-BOUND",
                        "run_id": "RUN-RESOURCE-BOUND",
                        "attempt_id": "ATTEMPT-RESOURCE-BOUND",
                        "restore_checkpoint_digest": None,
                        "lineage_scope_digest": {
                            "algorithm": "sha256",
                            "hex": "5" * 64,
                        },
                        "owner_id": "runtime-owner",
                        "lease_epoch": 1,
                        "fencing_token_digest": {
                            "algorithm": "sha256",
                            "hex": "6" * 64,
                        },
                        "prior_lease_digest": None,
                        "granted_at_utc": "2026-08-23T04:00:00Z",
                        "expires_at_utc": "2026-08-23T05:00:00Z",
                    },
                },
            },
        }
    )
    binding = exact["resource_admission"]
    assert binding["schema_version"] == "crca_scheduler_resource_admission_binding_v1"
    assert binding["resource_ids"] == ["host:cpu"]
