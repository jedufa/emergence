from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from tests.eiel_actor_v11_helpers import REPO_ROOT
from tools.actors.release import ACTOR_MANAGERS_VERSION


def _copy_minimal_tooling(tmp_path: Path) -> Path:
    root = tmp_path / "copy"
    shutil.copytree(REPO_ROOT / "tools", root / "tools")
    (root / "tools/scheduler").exists() and shutil.rmtree(root / "tools/scheduler")
    return root


def _run(root: Path, *args: str):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    return subprocess.run([sys.executable, "-m", "tools.eiel", *args], cwd=root, env=env, text=True, capture_output=True, check=False, timeout=30)


def test_physical_scheduler_removal_leaves_manual_cli_available(tmp_path: Path) -> None:
    root = _copy_minimal_tooling(tmp_path)
    version = _run(root, "--version")
    help_result = _run(root, "--help")
    work_help = _run(root, "workManager", "--help")
    session_help = _run(root, "sessionManager", "--help")
    assert version.returncode == 0 and ACTOR_MANAGERS_VERSION in version.stdout
    assert help_result.returncode == 0 and "bootstrap" in help_result.stdout
    assert work_help.returncode == 0
    assert session_help.returncode == 0


def test_missing_scheduler_fails_only_optional_command(tmp_path: Path) -> None:
    root = _copy_minimal_tooling(tmp_path)
    optional = _run(root, "scheduler", "status")
    manual = _run(root, "workManager", "--help")
    assert optional.returncode == 2
    assert "Optional EIEL Scheduler is not installed" in optional.stderr
    assert manual.returncode == 0


def test_physical_scheduler_removal_allows_manual_takeover_of_prepared_session(tmp_path: Path) -> None:
    from tests.eiel_scheduler_v110_helpers import make_scheduler_fixture
    from tools.scheduler.dispatch import compile_assignment
    from tools.actors.portfolio import session_store

    fixture = make_scheduler_fixture(tmp_path / "fixture")
    assignment = compile_assignment(fixture.dispatch_id, state_root=fixture.state)
    session_id = assignment["payload"]["material"]["session"]["record_id"]
    assert session_store(fixture.state).get(session_id)["payload"]["worker_identity_state"] == "PENDING"

    root = _copy_minimal_tooling(tmp_path / "removed")
    result = _run(
        root,
        "sessionManager",
        "resume",
        session_id,
        "--take-over-prepared",
        "--codex-id",
        "MANUAL-THREAD-1",
        "--takeover-reason",
        "Optional Scheduler was physically removed; operator resumed the prepared Session manually.",
        "--repo-root",
        str(fixture.repo),
        "--state-root",
        str(fixture.state),
    )
    assert result.returncode == 0, result.stderr
    resumed = session_store(fixture.state).get(session_id)
    assert resumed["status"] == "ACTIVE"
    assert resumed["payload"]["worker_identity_state"] == "ACTIVE"
    assert resumed["payload"]["current_codex_id"] == "MANUAL-THREAD-1"
    assert resumed["payload"].get("prepared_assignment_id") is None
