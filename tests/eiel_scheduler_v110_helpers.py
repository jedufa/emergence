from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tests.eiel_actor_v11_helpers import run_eiel
from tests.eiel_actor_v14_helpers import make_bootstrap_repo, run_bootstrap
from tools.actors.portfolio import bootstrap_store, session_store, work_store
from tools.scheduler.dispatch import authorize_dispatch


@dataclass
class SchedulerFixture:
    repo: Path
    state: Path
    bootstrap_id: str
    work_id: str
    package_path: Path
    dispatch_id: str


def package_payload(work_id: str, *, package_version: int = 1, content_file: str = "tools/scheduler_fixture.py") -> dict[str, Any]:
    return {
        "schema_version": "eiel_work_package_v1",
        "work_id": work_id,
        "package_version": package_version,
        "title": "Optional Scheduler bounded fixture",
        "goal": "Modify one bounded fixture through an immutable Assignment.",
        "scope": ["Scheduler fixture only."],
        "owned_files": [content_file],
        "forbidden_files": ["engine/**"],
        "dependencies": [],
        "acceptance_criteria": ["A bounded commit exists and independent review may follow."],
        "tests": ["git diff --check"],
        "evidence": ["Worker Result and commit."],
        "stop_conditions": ["Stop at the authority-phase boundary."],
        "recommended_authority": "IMPLEMENTATION_AUTHORITY",
        "recommended_model": "fake",
        "uncertainty_class": "U2",
        "read_contract": {
            "entry_files": [content_file],
            "required_documents": [],
            "search_roots": ["tools"],
            "excluded_roots": ["engine", "evidences"],
            "known_symbols": [],
            "maximum_initial_files": 2,
            "broad_search_allowed": False,
        },
        "output_policy": {
            "default_output_mode": "compact",
            "maximum_display_lines": 80,
            "test_output": "failures_and_summary",
            "git_output": "short_status_or_stat",
            "json_output": "selected_fields",
            "log_output": "tail_only",
            "recursive_output_requires_reason": True,
        },
        "execution_budget": {
            "expected_tool_calls": 10,
            "handoff_recommended_after_tool_calls": 20,
            "maximum_compactions": 1,
            "expected_duration_minutes": 30,
            "context_health_policy": "advisory",
        },
    }


def commit_all(repo: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def fresh_bootstrap(repo: Path, state: Path) -> str:
    return run_bootstrap(repo=repo, state_root=state, acknowledge_dirty=False)["bootstrap"]["bootstrap_id"]


def make_scheduler_fixture(
    tmp_path: Path,
    *,
    work_id: str = "WORK-SCHED-110",
    work_status: str = "ACTIVE",
    fake_behavior: str = "checkpoint",
    phases: tuple[str, ...] = ("implementation", "repair", "review"),
    fixture_content: str = "VALUE = 'scheduler-worker'\n",
    queue_priority: int = 100,
) -> SchedulerFixture:
    repo = make_bootstrap_repo(tmp_path)
    target = repo / "tools/scheduler_fixture.py"
    target.write_text("VALUE = 'baseline'\n", encoding="utf-8")
    package_path = repo / "work_packages/scheduler.yaml"
    package_path.parent.mkdir(parents=True, exist_ok=True)
    package_path.write_text(yaml.safe_dump(package_payload(work_id), sort_keys=False), encoding="utf-8")
    commit_all(repo, "scheduler fixture")
    state = tmp_path / "state"
    bootstrap_id = fresh_bootstrap(repo, state)
    run_eiel(
        [
            "workManager", "create", "--work-package", str(package_path), "--repo-root", str(repo),
            "--status", work_status, "--next-action", "Execute the bounded scheduler fixture.",
            "--state-root", str(state),
        ],
        state_root=state,
    )
    dispatch = authorize_dispatch(
        {
            "work_id": work_id,
            "enabled": True,
            "allowed_authority_phases": list(phases),
            "worker_profile": {
                "launcher_type": "fake",
                "fake_behavior": fake_behavior,
                "fake_fixture_content": fixture_content,
                "sandbox": "workspace-write",
                "ephemeral": True,
            },
            "repository": {"repo_root": str(repo), "expected_branch": None},
            "queue": {"priority": queue_priority},
            "automation_policy": {
                "allow_fresh_session": True,
                "allow_resume": True,
                "allow_independent_review": True,
                "allow_repair": True,
            },
            "lease_policy": {"duration_seconds": 3600, "heartbeat_interval_seconds": 5},
            "created_by": "HUMAN_OPERATOR",
        },
        state_root=state,
    )
    return SchedulerFixture(repo, state, bootstrap_id, work_id, package_path, str(dispatch["record_id"]))
