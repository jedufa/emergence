from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from jsonschema import Draft202012Validator

from tests.eiel_actor_v11_helpers import run_eiel
from tests.eiel_actor_v14_helpers import create_work, make_bootstrap_repo, run_bootstrap, start_session
from tools.actors.release import ACTOR_MANAGERS_VERSION, version_text

T0 = "2026-08-04T02:00:00Z"
T1 = "2026-08-04T02:10:00Z"
T2 = "2026-08-04T03:00:00Z"


def _bootstrap_id(result: dict) -> str:
    return str(result["bootstrap"]["bootstrap_id"])


def test_workspace_is_single_codex_front_door_and_handoff_is_deterministic(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap = run_bootstrap(repo=repo, state_root=state, generated_at=T0)
    bootstrap_id = _bootstrap_id(bootstrap)
    create_work(
        state,
        work_id="WORK-CONSOLIDATION",
        title="Consolidate the Codex operator path",
        generated_at=T0,
        status="READY",
        next_action="Run the consolidated actor verification.",
    )
    session = start_session(
        state,
        repo=repo,
        work_id="WORK-CONSOLIDATION",
        bootstrap_id=bootstrap_id,
        codex_id="codex-v16",
        generated_at=T0,
    )
    session_id = str(session["session"]["session_id"])

    opened = run_eiel(
        [
            "start",
            "--repo-root",
            str(repo),
            "--bootstrap-record",
            bootstrap_id,
            "--state-root",
            str(state),
            "--generated-at",
            T1,
        ],
        state_root=state,
    )
    assert opened["status"] == "WORKSPACE_READY"
    assert opened["workspace"]["schema_version"] == "eiel_codex_workspace_compact_v1"
    assert opened["workspace"]["session"]["session_id"] == session_id
    assert opened["workspace"]["tooling_version"] == ACTOR_MANAGERS_VERSION

    full_opened = run_eiel(
        [
            "start", "--full",
            "--repo-root", str(repo),
            "--bootstrap-record", bootstrap_id,
            "--state-root", str(state),
            "--generated-at", T1,
        ],
        state_root=state,
    )
    assert full_opened["workspace"]["schema_version"] == "eiel_codex_workspace_v2"
    assert full_opened["workspace"]["selected_session"]["record_id"] == session_id

    args = [
        "handoff",
        "--repo-root",
        str(repo),
        "--bootstrap-record",
        bootstrap_id,
        "--session-id",
        session_id,
        "--state-root",
        str(state),
        "--generated-at",
        T1,
    ]
    first = run_eiel(args, state_root=state)
    second = run_eiel(args, state_root=state)
    assert first == second
    assert first["status"] == "HANDOFF_READY"
    assert first["handoff"]["session"]["record_id"] == session_id
    assert first["handoff"]["record_hashes"]["session"]

    audit = run_eiel(
        ["audit", "--repo-root", str(repo), "--state-root", str(state), "--generated-at", T1],
        state_root=state,
    )
    assert audit["status"] == "CONSISTENT"
    assert audit["audit"]["problems"] == []


def test_head_change_is_advisory_for_execution_and_rebinds_selected_session(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    first_bootstrap = run_bootstrap(repo=repo, state_root=state, generated_at=T0)
    first_id = _bootstrap_id(first_bootstrap)
    create_work(
        state,
        work_id="WORK-HEAD-DRIFT",
        title="Resume after repository change",
        generated_at=T0,
        status="READY",
    )
    session = start_session(
        state,
        repo=repo,
        work_id="WORK-HEAD-DRIFT",
        bootstrap_id=first_id,
        codex_id="codex-original",
        generated_at=T0,
    )
    session_id = str(session["session"]["session_id"])

    (repo / "new_file.txt").write_text("new commit\n", encoding="utf-8")
    subprocess.run(["git", "add", "new_file.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "advance fixture"], cwd=repo, check=True)

    advisory = run_eiel(
        [
            "workspace",
            "open",
            "--repo-root",
            str(repo),
            "--bootstrap-record",
            first_id,
            "--state-root",
            str(state),
            "--generated-at",
            T2,
        ],
        state_root=state,
    )
    assert advisory["execution_blocked"] is False
    assert advisory["problems"] == []
    assert any("current HEAD" in warning for warning in advisory["warnings"])

    fresh = run_bootstrap(repo=repo, state_root=state, generated_at=T2)
    fresh_id = _bootstrap_id(fresh)
    selected = run_eiel(
        [
            "workspace",
            "open",
            "--repo-root",
            str(repo),
            "--bootstrap-record",
            fresh_id,
            "--session-id",
            session_id,
            "--state-root",
            str(state),
            "--generated-at",
            T2,
        ],
        state_root=state,
    )
    assert selected["status"] == "SESSION_BOOTSTRAP_REFRESH_REQUIRED"
    assert selected["execution_blocked"] is False
    assert selected["problems"] == []
    assert "--bootstrap-record" in selected["next_actions"][0]["command"]

    resumed = run_eiel(
        [
            "sessionManager",
            "resume",
            session_id,
            "--bootstrap-record",
            fresh_id,
            "--repo-root",
            str(repo),
            "--codex-id",
            "codex-original",
            "--state-root",
            str(state),
            "--generated-at",
            T2,
        ],
        state_root=state,
    )
    assert resumed["session"]["bootstrap_record_id"] == fresh_id
    assert resumed["session"]["bootstrap_history"] == [first_id, fresh_id]
    assert resumed["session"]["current_codex_id"] == "codex-original"

    ready = run_eiel(
        [
            "workspace",
            "open",
            "--repo-root",
            str(repo),
            "--bootstrap-record",
            fresh_id,
            "--session-id",
            session_id,
            "--state-root",
            str(state),
            "--generated-at",
            T2,
        ],
        state_root=state,
    )
    assert ready["status"] == "WORKSPACE_READY"


def test_workspace_reports_stale_bootstrap_without_blocking_authorized_work(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap = run_bootstrap(
        repo=repo,
        state_root=state,
        generated_at="2026-08-01T00:00:00Z",
    )
    bootstrap_id = _bootstrap_id(bootstrap)
    advisory = run_eiel(
        [
            "workspace",
            "open",
            "--repo-root",
            str(repo),
            "--bootstrap-record",
            bootstrap_id,
            "--max-bootstrap-age-hours",
            "24",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T00:00:00Z",
        ],
        state_root=state,
    )
    assert advisory["execution_blocked"] is False
    assert advisory["problems"] == []
    assert any("stale" in warning.lower() for warning in advisory["warnings"])

    allowed = run_eiel(
        [
            "workspace",
            "open",
            "--repo-root",
            str(repo),
            "--bootstrap-record",
            bootstrap_id,
            "--allow-stale-bootstrap",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T00:00:00Z",
        ],
        state_root=state,
    )
    assert allowed["status"] == "NO_OPEN_WORK"


def test_workspace_audit_detects_missing_cross_manager_reference(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap = run_bootstrap(repo=repo, state_root=state, generated_at=T0)
    bootstrap_id = _bootstrap_id(bootstrap)
    create_work(
        state,
        work_id="WORK-AUDIT-MISSING-REF",
        title="Audit a missing qualification reference",
        generated_at=T0,
        status="READY",
    )
    run_eiel(
        [
            "workManager",
            "update",
            "WORK-AUDIT-MISSING-REF",
            "--qualification-ref",
            "QUAL-MISSING",
            "--state-root",
            str(state),
            "--generated-at",
            T1,
        ],
        state_root=state,
    )
    start_session(
        state,
        repo=repo,
        work_id="WORK-AUDIT-MISSING-REF",
        bootstrap_id=bootstrap_id,
        codex_id="codex-audit",
        generated_at=T1,
    )
    audit = run_eiel(
        ["workspace", "audit", "--repo-root", str(repo), "--state-root", str(state), "--generated-at", T1],
        state_root=state,
        expected=2,
    )
    assert audit["status"] == "INCONSISTENT"
    assert any("QUAL-MISSING" in problem for problem in audit["audit"]["problems"])


def test_top_level_readme_and_version_surface_are_present() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "README.md").is_file()
    result = subprocess.run(
        [str(root / "tools/bin/eiel"), "--version"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == version_text()
    text = (root / "tools/actors/CODEX_OPERATOR_GUIDE.md").read_text(encoding="utf-8")
    assert f"EIEL V{ACTOR_MANAGERS_VERSION.rsplit('.', 1)[0]}" in text
    assert "eiel start" in text
    assert "eiel audit" in text
    assert "eiel handoff" in text



def _tree_hashes(root: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    if not root.exists():
        return rows
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rows[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows


def test_workspace_v2_schemas_validate_real_outputs_and_examples(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    actors = root / "tools" / "actors"
    workspace_schema = json.loads((actors / "schemas" / "eiel_codex_workspace_v2.schema.json").read_text(encoding="utf-8"))
    handoff_schema = json.loads((actors / "schemas" / "eiel_codex_workspace_handoff_v1.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(workspace_schema)
    Draft202012Validator.check_schema(handoff_schema)
    example = json.loads((actors / "examples" / "codex_workspace_v2.example.json").read_text(encoding="utf-8"))
    Draft202012Validator(workspace_schema).validate(example)

    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = _bootstrap_id(run_bootstrap(repo=repo, state_root=state, generated_at=T0))
    create_work(state, work_id="WORK-SCHEMA", title="Schema validation", generated_at=T0, status="READY")
    session_id = str(start_session(state, repo=repo, work_id="WORK-SCHEMA", bootstrap_id=bootstrap_id, codex_id="codex-schema", generated_at=T0)["session"]["session_id"])
    run_eiel(
        [
            "sessionManager", "checkpoint", session_id,
            "--next-action", "Validate the bound schema fixture.",
            "--next-command", "python -m tools.eiel --version",
            "--state-root", str(state),
            "--generated-at", T1,
        ],
        state_root=state,
    )
    opened = run_eiel(["start", "--full", "--repo-root", str(repo), "--bootstrap-record", bootstrap_id, "--session-id", session_id, "--state-root", str(state), "--generated-at", T1], state_root=state)
    handoff = run_eiel(["handoff", "--repo-root", str(repo), "--bootstrap-record", bootstrap_id, "--session-id", session_id, "--state-root", str(state), "--generated-at", T1], state_root=state)
    Draft202012Validator(workspace_schema).validate(opened["workspace"])
    Draft202012Validator(handoff_schema).validate(handoff["handoff"])


def test_workspace_audit_is_strictly_read_only(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = _bootstrap_id(run_bootstrap(repo=repo, state_root=state, generated_at=T0))
    create_work(state, work_id="WORK-AUDIT-READONLY", title="Audit without mutation", generated_at=T0, status="READY")
    start_session(state, repo=repo, work_id="WORK-AUDIT-READONLY", bootstrap_id=bootstrap_id, codex_id="codex-audit", generated_at=T0)
    before = _tree_hashes(state)
    result = run_eiel(["audit", "--repo-root", str(repo), "--state-root", str(state), "--generated-at", T1], state_root=state)
    after = _tree_hashes(state)
    assert result["status"] == "CONSISTENT"
    assert before == after


def test_workspace_implicit_bootstrap_is_scoped_to_current_repository(tmp_path: Path) -> None:
    repo_a = make_bootstrap_repo(tmp_path / "a")
    repo_b = make_bootstrap_repo(tmp_path / "b")
    state = tmp_path / "state"
    a_id = _bootstrap_id(run_bootstrap(repo=repo_a, state_root=state, generated_at="2026-08-04T01:00:00Z"))
    b_id = _bootstrap_id(run_bootstrap(repo=repo_b, state_root=state, generated_at="2026-08-04T02:00:00Z"))
    assert a_id != b_id
    opened = run_eiel(
        ["start", "--repo-root", str(repo_a), "--state-root", str(state), "--generated-at", "2026-08-04T03:00:00Z"],
        state_root=state,
    )
    assert opened["workspace"]["bootstrap"]["record_id"] == a_id
    assert opened["workspace"]["repository"]["root"] == str(repo_a.resolve())


def test_workspace_next_commands_preserve_repository_and_state_roots(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "custom-state"
    bootstrap_id = _bootstrap_id(run_bootstrap(repo=repo, state_root=state, generated_at=T0))
    create_work(state, work_id="WORK-EXACT-COMMAND", title="Exact command", generated_at=T0, status="READY")
    opened = run_eiel(
        ["start", "--bootstrap-record", bootstrap_id, "--state-root", str(state), "--generated-at", T1],
        state_root=state,
    )
    command = opened["next_actions"][0]["command"]
    assert "workManager list --open-only" in command
    assert f'--state-root "{state.resolve()}"' in command
