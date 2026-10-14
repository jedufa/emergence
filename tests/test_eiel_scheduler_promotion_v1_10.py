from __future__ import annotations

from pathlib import Path

from tests.eiel_actor_v11_helpers import run_eiel
from tests.eiel_scheduler_v110_helpers import make_scheduler_fixture
from tools.scheduler.dispatch import dispatch_candidates, request_assignment
from tools.scheduler.runtime import run_once


def test_promotion_stops_automation_and_requires_revised_package(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path, fake_behavior="promotion")
    event = run_once(state_root=fixture.state)
    assert event["status"] == "RESULT_SUBMITTED"
    assert event["worker_result"]["payload"]["outcome"] == "PROMOTION_REQUIRED"
    work = run_eiel(["workManager", "inspect", fixture.work_id, "--state-root", str(fixture.state)], state_root=fixture.state)["work"]
    assert work["derived_status"] == "PROMOTION_REQUIRED"
    candidates = dispatch_candidates(state_root=fixture.state)
    assert candidates[0]["eligible"] is False
    assert "promotion_required" in candidates[0]["reason_codes"]
    assert request_assignment(state_root=fixture.state) is None
