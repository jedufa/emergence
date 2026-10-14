from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from tests.eiel_actor_v11_helpers import REPO_ROOT, run_eiel
from tests.eiel_actor_v14_helpers import create_work, make_bootstrap_repo, run_bootstrap
from tests.test_eiel_work_package_v1_8 import (
    _commit,
    _create_bound,
    _package_value,
    _start,
    _write_package,
)
from tools.actors.bootstrap import bootstrap_refresh_command
from tools.actors.workspace import _implementation_and_qualification_state


def _run_cli(args: list[str], *, state_root: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    if state_root is not None:
        env["EIEL_ACTOR_STATE_ROOT"] = str(state_root)
    return subprocess.run(
        [sys.executable, "-m", "tools.eiel", *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


@pytest.mark.parametrize(
    "tool",
    [
        "bootstrap",
        "workspace",
        "workManager",
        "sessionManager",
        "missionPlanner",
        "qualificationManager",
        "resourceManager",
        "evidenceManager",
        "bridgeManager",
        "assignment",
        "scheduler",
    ],
)
def test_human_errors_are_respected_across_operator_tools(tmp_path: Path, tool: str) -> None:
    result = _run_cli([tool, "--human"], state_root=tmp_path / tool)
    assert result.returncode == 2, (tool, result.stdout, result.stderr)
    rendered = result.stdout.strip() or result.stderr.strip()
    assert rendered.startswith("ERROR:"), (tool, rendered)
    assert not rendered.startswith("{")


def test_parse_error_honors_human_mode_and_persists_structured_output(tmp_path: Path) -> None:
    output = tmp_path / "error.json"
    result = _run_cli(
        [
            "workManager",
            "inspect",
            "--human",
            "--output",
            str(output),
            "--state-root",
            str(tmp_path / "state"),
        ]
    )
    assert result.returncode == 2
    assert result.stdout.startswith("ERROR:")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED"
    assert payload["reason_codes"] == ["cli_usage_error"]
    assert payload["next_actions"][0]["action"] == "show_help"


def test_scheduler_success_has_concise_human_summary(tmp_path: Path) -> None:
    result = _run_cli(
        ["scheduler", "status", "--state-root", str(tmp_path / "state"), "--human"]
    )
    assert result.returncode == 0, result.stderr
    assert "Scheduler:" in result.stdout
    assert not result.stdout.lstrip().startswith("{")
    assert "Next:" in result.stdout or "Count" in result.stdout or "Counts:" in result.stdout


def test_bootstrap_refresh_command_preserves_required_choices_and_executes(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    (repo / ".venv/bin/python").unlink()
    first = run_eiel(
        [
            "bootstrap",
            "inspect",
            "--repo-root",
            str(repo),
            "--task-mode",
            "mechanical",
            "--topic",
            "refresh fidelity fixture",
            "--operator-instruction-ref",
            "operator approved mechanical verification",
            "--gpu-required",
            "no",
            "--allow-current-python",
            "--preflight-exemption-reason",
            "bounded mechanical transfer verification",
            "--write-scope",
            "tools",
            "--expected-validation",
            "python -m py_compile tools/example.py",
            "--collection",
            "documents",
            "--limit",
            "7",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-06T07:00:00Z",
        ],
        state_root=state,
    )
    bootstrap_id = first["bootstrap"]["bootstrap_id"]

    marker = repo / "README.md"
    marker.write_text("new head\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "advance head"], cwd=repo, check=True)

    blocked = run_eiel(
        [
            "workspace",
            "open",
            "--repo-root",
            str(repo),
            "--bootstrap-record",
            bootstrap_id,
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-06T07:05:00Z",
        ],
        state_root=state,
        expected=2,
    )
    command = blocked["next_actions"][0]["command"]
    argv = shlex.split(command)
    assert "--allow-current-python" in argv
    assert argv[argv.index("--preflight-exemption-reason") + 1] == "bounded mechanical transfer verification"
    assert argv[argv.index("--gpu-required") + 1] == "no"
    assert argv[argv.index("--collection") + 1] == "documents"
    assert argv[argv.index("--limit") + 1] == "7"
    assert argv[argv.index("--state-root") + 1] == str(state.resolve())

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    refreshed = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert refreshed.returncode == 0, refreshed.stdout + refreshed.stderr
    refreshed_payload = json.loads(refreshed.stdout)
    assert refreshed_payload["bootstrap"]["readiness"] in {"READY", "READY_WITH_WARNINGS"}



def test_legacy_bootstrap_refresh_infers_explicit_python_and_mechanical_exemption(tmp_path: Path) -> None:
    legacy = {
        "payload": {
            "task": {
                "mode": "mechanical",
                "topic": "legacy refresh fixture",
                "operator_instruction_ref": "legacy operator instruction",
                "gpu_required": False,
                "write_scope": [{"path": "tools"}],
                "expected_validation": ["python -m py_compile tools/example.py"],
            },
            "environment": {"source": "explicit", "python": "/opt/eiel/bin/python"},
            "authority": {"preflight": {"collection": "documents"}},
            "checks": [
                {
                    "name": "librarian_preflight",
                    "status": "WARN",
                    "message": "mechanical exemption",
                    "details": {"reason": "legacy bounded exemption"},
                }
            ],
        }
    }
    command = bootstrap_refresh_command(
        legacy,
        repo_root=tmp_path,
        state_root=tmp_path / "state",
    )
    argv = shlex.split(command)
    assert argv[argv.index("--python") + 1] == "/opt/eiel/bin/python"
    assert argv[argv.index("--preflight-exemption-reason") + 1] == "legacy bounded exemption"
    assert argv[argv.index("--collection") + 1] == "documents"
    assert argv[argv.index("--gpu-required") + 1] == "no"

def test_review_session_defaults_are_review_specific(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package = _write_package(repo, _package_value())
    bootstrap0 = run_bootstrap(
        repo=repo,
        state_root=state,
        generated_at="2026-08-06T07:10:00Z",
    )["bootstrap"]["bootstrap_id"]
    _create_bound(repo, state, package, generated_at="2026-08-06T07:11:00Z")
    implementation = _start(
        repo,
        state,
        bootstrap0,
        codex_id="implementation-thread",
        generated_at="2026-08-06T07:12:00Z",
    )
    delivery_commit = _commit(repo, "tools/bounded_component.py", "VALUE = 1\n", "bounded delivery")
    from tests.test_eiel_work_package_v1_8 import _refresh_bootstrap
    _refresh_bootstrap(repo, state, implementation["session_id"], generated_at="2026-08-06T07:12:30Z")
    run_eiel(
        [
            "sessionManager",
            "complete",
            implementation["session_id"],
            "--delivery-commit",
            delivery_commit,
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-06T07:13:00Z",
        ],
        state_root=state,
    )
    bootstrap1 = run_bootstrap(
        repo=repo,
        state_root=state,
        generated_at="2026-08-06T07:14:00Z",
    )["bootstrap"]["bootstrap_id"]
    review = _start(
        repo,
        state,
        bootstrap1,
        codex_id="independent-review-thread",
        role="REVIEW_AUTHORITY",
        target=implementation["session_id"],
        generated_at="2026-08-06T07:15:00Z",
    )
    assert review["current_stage"] == "review"
    assert review["current_step"] == "review the exact bound delivery"
    assert review["next_action"] == (
        "Review the exact delivery Session and commit against the bound Work Package."
    )
    assert review["next_command"] is None


def test_terminal_work_has_no_false_missing_next_action_warning_and_human_says_none(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    run_eiel(
        [
            "workManager",
            "create",
            "--work-id",
            "WORK-TERMINAL",
            "--title",
            "Terminal warning fixture",
            "--summary",
            "Terminal warning fixture",
            "--work-type",
            "ENGINEERING",
            "--priority",
            "P2",
            "--status",
            "ACTIVE",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-06T07:20:00Z",
        ],
        state_root=state,
    )
    completed = run_eiel(
        [
            "workManager",
            "complete",
            "WORK-TERMINAL",
            "--force",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-06T07:21:00Z",
        ],
        state_root=state,
    )["work"]
    assert completed["status"] == "COMPLETED"
    assert "no next action is recorded" not in completed["validation"]["warnings"]

    human = _run_cli(
        [
            "workManager",
            "inspect",
            "WORK-TERMINAL",
            "--state-root",
            str(state),
            "--human",
        ],
        state_root=state,
    )
    assert human.returncode == 0
    assert human.stdout.rstrip("\n").splitlines() == [
        "Work WORK-TERMINAL",
        "  Status: COMPLETED (COMPLETED)",
        "  Title: Terminal warning fixture",
        "  Summary: Terminal warning fixture",
        "  Next WI: none",
    ]


def test_non_applicable_qualification_is_not_required(tmp_path: Path) -> None:
    repository = {"head": "abc123"}
    engineering = {
        "record_id": "WORK-ENGINEERING",
        "status": "ACTIVE",
        "payload": {"work_type": "ENGINEERING", "references": {}},
    }
    qualification = {
        "record_id": "WORK-QUALIFICATION",
        "status": "ACTIVE",
        "payload": {"work_type": "QUALIFICATION", "references": {}},
    }
    normal = _implementation_and_qualification_state(
        repository,
        engineering,
        state_root=tmp_path / "state",
    )
    required = _implementation_and_qualification_state(
        repository,
        qualification,
        state_root=tmp_path / "state",
    )
    assert normal["qualification"]["status"] == "NOT_REQUIRED"
    assert required["qualification"]["status"] == "UNKNOWN"


def test_workspace_audit_human_output_lists_warning_details(tmp_path: Path) -> None:
    state = tmp_path / "state"
    create_work(
        state,
        work_id="WORK-AUDIT-WARNING",
        title="Audit warning fixture",
        generated_at="2026-08-06T07:30:00Z",
        status="ACTIVE",
        next_action="Start the bounded Session.",
    )
    result = _run_cli(
        ["workspace", "audit", "--state-root", str(state), "--human"],
        state_root=state,
    )
    assert result.returncode == 0
    assert "Warnings: 1" in result.stdout
    assert "Warning: active work WORK-AUDIT-WARNING has no retained session" in result.stdout
    assert "Next:" in result.stdout


def test_help_separates_read_only_daily_and_advanced_surfaces() -> None:
    top = _run_cli(["--help"])
    session = _run_cli(["sessionManager", "--help"])
    assignment = _run_cli(["assignment", "--help"])
    assert top.returncode == session.returncode == assignment.returncode == 0
    assert "does not resume a Session" in top.stdout
    assert "Codex uses structured JSON by default. Human operators add --human." in top.stdout
    assert "Daily:" in session.stdout
    assert "Scheduler-internal/advanced:" in session.stdout
    assert "Scheduler-internal/advanced:" in assignment.stdout


def test_operator_documents_are_bounded_and_transfer_safe() -> None:
    quickstart = (REPO_ROOT / "OPERATOR_QUICKSTART.md").read_text(encoding="utf-8")
    transfer = (REPO_ROOT / "TRANSFER_README.md").read_text(encoding="utf-8")
    root_readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    guide = (REPO_ROOT / "tools/actors/CODEX_OPERATOR_GUIDE.md").read_text(encoding="utf-8")
    actors = (REPO_ROOT / "tools/actors/README.md").read_text(encoding="utf-8")
    scheduler = (REPO_ROOT / "tools/scheduler/README.md").read_text(encoding="utf-8")

    assert len(quickstart.splitlines()) <= 100
    assert "latest durable Session checkpoint" in quickstart
    assert "standalone Emergence worktree" in transfer
    assert "Do not create Bootstrap, Work, Session" in transfer
    assert "For Codex, use the default structured JSON" in root_readme
    assert "every new or resumed Codex turn" not in guide
    assert "operator-supplied" in guide
    assert "tools/actors/CONTEXT_EPOCH_POLICY.md" in actors
    assert "--poll-seconds 300" in scheduler
    assert "--poll-interval-seconds" not in scheduler
