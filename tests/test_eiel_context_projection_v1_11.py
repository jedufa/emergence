from __future__ import annotations

import hashlib
import json
import subprocess
import argparse
import shlex
from pathlib import Path

from jsonschema import Draft202012Validator

from tests.eiel_actor_v11_helpers import run_eiel
from tests.eiel_actor_v14_helpers import make_bootstrap_repo, run_bootstrap
from tests.test_eiel_context_efficiency_v1_9 import _setup
from tools.actors.workspace import _compact_full_view_command
from tools.actors.context_projection import (
    COMPACT_WORKSPACE_SCHEMA_VERSION,
    project_compact_workspace_guidance,
    project_initial_context_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "tools/actors/schemas/eiel_codex_workspace_compact_v1.schema.json"


def _common(repo: Path, state: Path, bootstrap_id: str, session_id: str) -> list[str]:
    return [
        "--bootstrap-record",
        bootstrap_id,
        "--work-id",
        "WORK-CONTEXT-19",
        "--session-id",
        session_id,
        "--repo-root",
        str(repo),
        "--state-root",
        str(state),
        "--generated-at",
        "2026-08-04T00:04:00Z",
    ]


def _cli(args: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python", "-m", "tools.eiel", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()




def test_full_view_command_shell_quotes_exact_bound_paths() -> None:
    repo = "/tmp/repo $HOME with space's"
    state = "/tmp/state `uname` with space"
    args = argparse.Namespace(
        repo_root=repo,
        state_root=state,
        bootstrap_record="BST-ONE",
        authority_domain="implementation",
        work_id="WORK-ONE",
        session_id="SES-ONE",
        allow_stale_bootstrap=False,
        max_bootstrap_age_hours=None,
    )
    payload = {
        "workspace": {
            "repository": {"root": repo},
            "state_root": state,
            "bootstrap": {"record_id": "BST-ONE"},
            "selected_work": {"record_id": "WORK-ONE"},
            "selected_session": {"record_id": "SES-ONE"},
        }
    }
    command = _compact_full_view_command(args, payload)
    words = shlex.split(command)
    assert words[:5] == ["python", "-m", "tools.eiel", "workspace", "open"]
    assert words[words.index("--repo-root") + 1] == repo
    assert words[words.index("--state-root") + 1] == state

def test_compact_projection_is_schema_valid_bounded_and_preserves_critical_truth(tmp_path: Path) -> None:
    repo, state, bootstrap_id, session_id, _ = _setup(tmp_path)
    common = _common(repo, state, bootstrap_id, session_id)

    full = _cli(["workspace", "open", *common])
    explicit_full = _cli(["workspace", "open", "--full", *common])
    compact = _cli(["workspace", "open", "--compact", *common])

    assert full.returncode == explicit_full.returncode == compact.returncode == 0
    assert full.stdout == explicit_full.stdout
    full_payload = json.loads(full.stdout)
    compact_payload = json.loads(compact.stdout)

    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(compact_payload)

    assert compact_payload["workspace"]["schema_version"] == COMPACT_WORKSPACE_SCHEMA_VERSION
    assert compact_payload["status"] == full_payload["status"]
    assert compact_payload["reason_codes"] == full_payload["reason_codes"]
    assert compact_payload["problems"] == full_payload["problems"]
    assert compact_payload["next_actions"][0] == full_payload["next_actions"][0]
    assert compact_payload["workspace"]["repository"]["head"] == full_payload["workspace"]["repository"]["head"]
    assert compact_payload["workspace"]["work"]["work_id"] == full_payload["workspace"]["selected_work"]["record_id"]
    assert compact_payload["workspace"]["session"]["session_id"] == full_payload["workspace"]["selected_session"]["record_id"]
    assert compact_payload["workspace"]["work"]["package"]["sha256"] == full_payload["workspace"]["selected_work"]["payload"]["work_package_binding"]["sha256"]
    assert compact_payload["workspace"]["context_receipt"]["receipt_sha256"] == full_payload["workspace"]["context_receipt"]["receipt_sha256"]

    display_limit = compact_payload["workspace"]["output_policy"]["maximum_display_lines"]
    assert len(compact.stdout.splitlines()) <= display_limit
    assert len(compact.stdout.encode()) <= int(len(full.stdout.encode()) * 0.30)


def test_shortcuts_default_compact_but_output_file_retains_full_payload(tmp_path: Path) -> None:
    repo, state, bootstrap_id, session_id, _ = _setup(tmp_path)
    common = _common(repo, state, bootstrap_id, session_id)
    output = tmp_path / "workspace-full.json"

    result = _cli(["start", *common, "--output", str(output)])
    assert result.returncode == 0
    displayed = json.loads(result.stdout)
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert displayed["workspace"]["schema_version"] == COMPACT_WORKSPACE_SCHEMA_VERSION
    assert stored["workspace"]["schema_version"] == "eiel_codex_workspace_v2"
    assert displayed["status"] == stored["status"]
    assert displayed["workspace"]["references"]["full_output_path"] == str(output)

    for shortcut in ("resume", "status"):
        projected = _cli([shortcut, *common])
        assert projected.returncode == 0
        assert json.loads(projected.stdout)["workspace"]["schema_version"] == COMPACT_WORKSPACE_SCHEMA_VERSION

    full = _cli(["start", "--full", *common])
    human = _cli(["start", "--human", *common])
    assert json.loads(full.stdout)["workspace"]["schema_version"] == "eiel_codex_workspace_v2"
    assert human.stdout.startswith("Workspace:")


def test_compact_projection_is_read_only(tmp_path: Path) -> None:
    repo, state, bootstrap_id, session_id, _ = _setup(tmp_path)
    before = _tree_digest(state)
    result = _cli(["start", *_common(repo, state, bootstrap_id, session_id)])
    after = _tree_digest(state)
    assert result.returncode == 0
    assert before == after


def test_initial_context_metrics_warn_without_blocking(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "small.py").write_text("print('ok')\n", encoding="utf-8")
    (repo / "large.md").write_text("x" * 70_000, encoding="utf-8")
    metrics = project_initial_context_metrics(
        {
            "entry_files": ["small.py"],
            "required_documents": ["large.md", "missing.md"],
        },
        repo_root=repo,
    )
    assert metrics["status"] == "ADVISORY_WARNING"
    assert metrics["initial_total_bytes"] > 65_536
    assert metrics["missing_initial_files"] == ["missing.md"]
    assert metrics["warning_reasons"]

    compact = project_compact_workspace_guidance(
        {
            "guidance_schema_version": "eiel_codex_guidance_v1",
            "status": "WORKSPACE_READY",
            "reason_codes": ["workspace_ready"],
            "problems": [],
            "warnings": [],
            "next_actions": [],
            "related_records": [],
            "ok": True,
            "workspace": {
                "schema_version": "eiel_codex_workspace_v2",
                "tooling_version": "1.11.0",
                "repository": {"root": str(repo), "branch": "main", "head": "abc", "dirty": False},
                "bootstrap": {"record_id": "BST-TEST", "status": "READY", "age_hours": 0},
                "portfolio": {"open_work_count": 0, "resumable_session_count": 0},
                "initial_context": {"entry_files": ["small.py"], "required_documents": ["large.md", "missing.md"]},
                "output_policy": {"maximum_display_lines": 120},
                "context_health": {"status": "NOT_OBSERVED", "handoff_recommended": False},
            },
        }
    )
    assert compact["ok"] is True
    assert compact["workspace"]["initial_context_metrics"]["status"] == "ADVISORY_WARNING"
    assert any(item.startswith("Initial context advisory:") for item in compact["warnings"])


def test_ambiguous_work_remains_fail_closed_and_lists_candidates(tmp_path: Path) -> None:
    repo, state, bootstrap_id, session_id, package_path = _setup(tmp_path)
    import yaml

    second = yaml.safe_load(package_path.read_text(encoding="utf-8"))
    second["work_id"] = "WORK-CONTEXT-20"
    second["title"] = "Second bounded package"
    second_path = repo / "work_packages/context_efficiency_2.yaml"
    second_path.write_text(yaml.safe_dump(second, sort_keys=False), encoding="utf-8")
    subprocess.run(["git", "add", str(second_path.relative_to(repo))], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "second package"], cwd=repo, check=True)
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    run_eiel(
        ["sessionManager", "attach-bootstrap", session_id, "--bootstrap-record", bootstrap_id, "--repo-root", str(repo), "--state-root", str(state)],
        state_root=state,
    )
    run_eiel(["sessionManager", "discontinue", session_id, "--state-root", str(state)], state_root=state)
    run_eiel(
        [
            "workManager",
            "create",
            "--work-package",
            str(second_path),
            "--repo-root",
            str(repo),
            "--status",
            "ACTIVE",
            "--next-action",
            "Implement the second package.",
            "--state-root",
            str(state),
        ],
        state_root=state,
    )

    result = _cli(["start", "--bootstrap-record", bootstrap_id, "--repo-root", str(repo), "--state-root", str(state)])
    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["status"] == "WORKSPACE_READY"
    assert payload["execution_blocked"] is False
    assert payload["workflow_administration_required"] is True
    assert sorted(payload["workspace"]["portfolio"]["candidate_work_ids"]) == ["WORK-CONTEXT-19", "WORK-CONTEXT-20"]
    assert payload["problems"] == []


def test_workspace_missing_bootstrap_human_path_is_nonblocking_and_human(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "empty-state"
    result = _cli(["workspace", "open", "--human", "--repo-root", str(repo), "--state-root", str(state)])
    assert result.returncode == 0
    assert result.stdout.startswith("Workspace assistance unavailable")
    assert not result.stdout.lstrip().startswith("{")


def test_planned_stop_human_output_confirms_receipt_identity_path_and_digest(tmp_path: Path) -> None:
    repo, state, _, session_id, _ = _setup(tmp_path)
    receipt = _cli(
        [
            "sessionManager",
            "context-receipt",
            session_id,
            "--repo-root",
            str(repo),
            "--state-root",
            str(state),
            "--attach",
            "--human",
        ]
    )
    assert receipt.returncode == 0
    assert "Context receipt: CTX-" in receipt.stdout
    assert "Receipt SHA-256:" in receipt.stdout
    assert "Receipt path:" in receipt.stdout

    handoff = _cli(
        [
            "sessionManager",
            "handoff",
            session_id,
            "--repo-root",
            str(repo),
            "--state-root",
            str(state),
            "--human",
        ]
    )
    assert handoff.returncode == 0
    assert "Context receipt: CTX-" in handoff.stdout
    assert "Receipt SHA-256:" in handoff.stdout
    assert "Receipt path:" in handoff.stdout


def test_example_package_does_not_force_reread_of_global_operator_authority() -> None:
    import yaml

    package = yaml.safe_load((ROOT / "tools/actors/examples/work_package.example.yaml").read_text(encoding="utf-8"))
    required = set(package["read_contract"]["required_documents"])
    assert "AGENTS.md" not in required
    assert "tools/actors/CODEX_OPERATOR_GUIDE.md" not in required
