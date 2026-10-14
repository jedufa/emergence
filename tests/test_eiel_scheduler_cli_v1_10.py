from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.eiel_actor_v11_helpers import run_eiel
from tests.eiel_scheduler_v110_helpers import make_scheduler_fixture

ROOT = Path(__file__).resolve().parents[1]


def test_optional_cli_surfaces_are_discoverable_without_changing_manual_start() -> None:
    help_result = subprocess.run(
        [sys.executable, "-m", "tools.eiel", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Optional unattended execution" in help_result.stdout
    assert "assignment" in help_result.stdout
    assert "scheduler" in help_result.stdout
    start_help = subprocess.run(
        [sys.executable, "-m", "tools.eiel", "start", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "workspace open" in start_help.stdout


def test_cli_run_once_executes_only_explicitly_dispatched_work(tmp_path: Path) -> None:
    fixture = make_scheduler_fixture(tmp_path, fake_behavior="checkpoint")
    listed = run_eiel(
        ["assignment", "list-dispatch", "--state-root", str(fixture.state)],
        state_root=fixture.state,
    )
    assert listed["count"] == 1
    result = run_eiel(
        ["scheduler", "run-once", "--state-root", str(fixture.state)],
        state_root=fixture.state,
    )
    assert result["scheduler"]["status"] == "RESULT_SUBMITTED"
    manual = run_eiel(
        ["workManager", "inspect", fixture.work_id, "--state-root", str(fixture.state)],
        state_root=fixture.state,
    )
    assert manual["work"]["status"] == "ACTIVE"
