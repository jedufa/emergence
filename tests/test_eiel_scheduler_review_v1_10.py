from __future__ import annotations

from pathlib import Path

from tests.eiel_actor_v11_helpers import run_eiel
from tests.eiel_actor_v14_helpers import run_bootstrap
from tests.eiel_scheduler_v110_helpers import make_scheduler_fixture
from tools.scheduler.dispatch import authorize_dispatch, set_dispatch_enabled
from tools.scheduler.runtime import run_once


def _authorize(fixture, *, phase: str, behavior: str, content: str | None = None):
    if phase == "review":
        run_bootstrap(
            repo=fixture.repo,
            state_root=fixture.state,
            task_mode="review",
            edit_authorized=False,
            write_scope=(),
        )
    elif phase in {"implementation", "repair"}:
        run_bootstrap(repo=fixture.repo, state_root=fixture.state, task_mode="implementation")
    raw = {
        "work_id": fixture.work_id,
        "repository": {"repo_root": str(fixture.repo)},
        "allowed_authority_phases": [phase],
        "worker_profile": {"launcher_type": "fake", "fake_behavior": behavior, "fake_fixture_content": content or "VALUE='review'\n"},
        "queue": {"priority": 1},
        "automation_policy": {"allow_fresh_session": True, "allow_resume": True, "allow_independent_review": True, "allow_repair": True},
    }
    return authorize_dispatch(raw, state_root=fixture.state)


def _status(fixture):
    return run_eiel(["workManager", "inspect", fixture.work_id, "--state-root", str(fixture.state)], state_root=fixture.state)["work"]


def test_implementation_review_repair_acceptance_flow(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path, fake_behavior="implementation-complete", fixture_content="VALUE='implementation'\n")
    assert run_once(state_root=fixture.state)["status"] == "RESULT_SUBMITTED"
    assert _status(fixture)["derived_status"] == "REVIEW_REQUIRED"

    set_dispatch_enabled(fixture.dispatch_id, enabled=False, state_root=fixture.state)
    review_changes = _authorize(fixture, phase="review", behavior="review-changes")
    assert run_once(state_root=fixture.state)["status"] == "RESULT_SUBMITTED"
    assert _status(fixture)["derived_status"] == "CHANGES_REQUIRED"

    set_dispatch_enabled(review_changes["record_id"], enabled=False, state_root=fixture.state)
    repair = _authorize(fixture, phase="repair", behavior="repair-complete", content="VALUE='repair'\n")
    assert run_once(state_root=fixture.state)["status"] == "RESULT_SUBMITTED"
    assert _status(fixture)["derived_status"] == "REVIEW_REQUIRED"

    set_dispatch_enabled(repair["record_id"], enabled=False, state_root=fixture.state)
    _authorize(fixture, phase="review", behavior="review-accept")
    assert run_once(state_root=fixture.state)["status"] == "RESULT_SUBMITTED"
    status = _status(fixture)
    assert status["derived_status"] == "READY_TO_ACCEPT"
    review_id = status["review"]["accepted_review_sessions"][0]["session_id"]
    completed = run_eiel(["workManager", "complete", fixture.work_id, "--review-session", review_id, "--state-root", str(fixture.state)], state_root=fixture.state)
    assert completed["work"]["status"] == "COMPLETED"


def test_scheduler_never_completes_work_automatically(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path, fake_behavior="implementation-complete")
    run_once(state_root=fixture.state)
    assert _status(fixture)["status"] == "ACTIVE"
    assert _status(fixture)["derived_status"] == "REVIEW_REQUIRED"


def test_reviewed_ready_work_can_complete_but_unreviewed_ready_work_cannot(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(
        tmp_path,
        work_status="READY",
        fake_behavior="implementation-complete",
        fixture_content="VALUE='implementation-ready'\n",
    )
    try:
        run_eiel(
            ["workManager", "complete", fixture.work_id, "--state-root", str(fixture.state)],
            state_root=fixture.state,
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("unreviewed READY Work unexpectedly completed")

    assert run_once(state_root=fixture.state)["status"] == "RESULT_SUBMITTED"
    set_dispatch_enabled(fixture.dispatch_id, enabled=False, state_root=fixture.state)
    _authorize(fixture, phase="review", behavior="review-accept")
    assert run_once(state_root=fixture.state)["status"] == "RESULT_SUBMITTED"
    status = _status(fixture)
    assert status["status"] == "READY"
    assert status["derived_status"] == "READY_TO_ACCEPT"
    review_id = status["review"]["accepted_review_sessions"][0]["session_id"]
    completed = run_eiel(
        ["workManager", "complete", fixture.work_id, "--review-session", review_id, "--state-root", str(fixture.state)],
        state_root=fixture.state,
    )
    assert completed["work"]["status"] == "COMPLETED"
