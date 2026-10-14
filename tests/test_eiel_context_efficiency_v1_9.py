from __future__ import annotations

import json
import subprocess
from pathlib import Path

from jsonschema import Draft202012Validator

from tests.eiel_actor_v11_helpers import run_eiel
from tests.eiel_actor_v14_helpers import make_bootstrap_repo, run_bootstrap
from tools.actors.context_receipt import (
    build_context_receipt,
    project_context_health,
    validate_context_receipt,
    verify_receipt_seal,
)
from tools.actors.portfolio import bootstrap_store, session_store, work_store
from tools.actors.work_package import (
    DEFAULT_EXECUTION_BUDGET,
    load_work_package,
    normalize_work_package,
    package_path_matches,
    work_package_contract_sha256,
)
from tools.analysis.codex_session_efficiency import analyze_session

ROOT = Path(__file__).resolve().parents[1]
ACTORS = ROOT / "tools" / "actors"


def _base_package(*, with_context: bool = True) -> dict:
    package = {
        "schema_version": "eiel_work_package_v1",
        "work_id": "WORK-CONTEXT-19",
        "package_version": 1,
        "title": "Context-efficiency hardening",
        "goal": "Implement one bounded context-efficiency slice.",
        "scope": ["Context receipt and advisory context health only."],
        "owned_files": ["tools/component.py", "tests/test_component.py"],
        "forbidden_files": ["engine/**"],
        "dependencies": [],
        "acceptance_criteria": ["Focused context-efficiency tests pass."],
        "tests": ["pytest -q tests/test_component.py"],
        "evidence": ["Focused test result and Context Receipt."],
        "stop_conditions": ["Stop at an authority-phase boundary."],
        "recommended_authority": "IMPLEMENTATION_AUTHORITY",
        "recommended_model": "GPT-5.3 Spark High",
        "uncertainty_class": "U2",
    }
    if with_context:
        package.update(
            {
                "read_contract": {
                    "entry_files": ["tools/component.py", "tests/test_component.py"],
                    "required_documents": ["docs/context_epoch_policy.md"],
                    "search_roots": ["tools", "tests"],
                    "excluded_roots": ["evidences"],
                    "known_symbols": ["component_entry"],
                    "maximum_initial_files": 4,
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
                    "expected_tool_calls": 3,
                    "handoff_recommended_after_tool_calls": 5,
                    "maximum_compactions": 1,
                    "expected_duration_minutes": 30,
                    "context_health_policy": "advisory",
                },
            }
        )
    return package


def _setup(tmp_path: Path) -> tuple[Path, Path, str, str, Path]:
    repo = make_bootstrap_repo(tmp_path)
    (repo / "tools/component.py").write_text("def component_entry():\n    return 'ok'\n", encoding="utf-8")
    (repo / "tests/test_component.py").write_text("def test_component():\n    assert True\n", encoding="utf-8")
    (repo / "docs/context_epoch_policy.md").write_text("# Context epoch policy\n", encoding="utf-8")
    package_path = repo / "work_packages/context_efficiency.yaml"
    package_path.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    package_path.write_text(yaml.safe_dump(_base_package(), sort_keys=False), encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "context package fixture"], cwd=repo, check=True)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    run_eiel(
        [
            "workManager",
            "create",
            "--work-package",
            str(package_path),
            "--repo-root",
            str(repo),
            "--status",
            "ACTIVE",
            "--next-action",
            "Implement the bounded package.",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T00:01:00Z",
        ],
        state_root=state,
    )
    session = run_eiel(
        [
            "sessionManager",
            "start",
            "WORK-CONTEXT-19",
            "--bootstrap-record",
            bootstrap_id,
            "--codex-id",
            "codex-context-1",
            "--repo-root",
            str(repo),
            "--model",
            "GPT-5.3 Spark High",
            "--reasoning-level",
            "HIGH",
            "--current-step",
            "implement receipt",
            "--next-action",
            "Run focused tests.",
            "--next-command",
            "pytest -q tests/test_component.py",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T00:02:00Z",
        ],
        state_root=state,
    )["session"]
    return repo, state, bootstrap_id, session["session_id"], package_path


def _records(state: Path, session_id: str):
    session = session_store(state).get(session_id)
    work = work_store(state).get(str(session["payload"]["work_id"]))
    bootstrap = bootstrap_store(state).get(str(session["payload"]["bootstrap_record_id"]))
    return work, session, bootstrap


def test_optional_contracts_are_canonical_and_legacy_packages_remain_legacy() -> None:
    package = _base_package()
    normalized = normalize_work_package(package)
    assert normalized["read_contract"]["maximum_initial_files"] == 4
    assert normalized["output_policy"]["maximum_display_lines"] == 80
    assert normalized["execution_budget"]["handoff_recommended_after_tool_calls"] == 5
    assert work_package_contract_sha256(normalized) == work_package_contract_sha256(normalize_work_package(package))

    legacy = normalize_work_package(_base_package(with_context=False))
    assert "read_contract" not in legacy
    assert "output_policy" not in legacy
    assert "execution_budget" not in legacy

    historical = load_work_package(ACTORS / "examples/work_package_legacy_v1_8.example.yaml")
    assert work_package_contract_sha256(historical) == "90c6ea4722f34503a72dee388b38a9e87f3a68f3c3fb08d09bfc53251df577b0"



def test_optional_contracts_yaml_json_identity(tmp_path: Path) -> None:
    package = _base_package()
    json_path = tmp_path / "package.json"
    yaml_path = tmp_path / "package.yaml"
    json_path.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    import yaml

    yaml_path.write_text(yaml.safe_dump(package, sort_keys=False), encoding="utf-8")
    from_json = load_work_package(json_path)
    from_yaml = load_work_package(yaml_path)
    assert from_json == from_yaml
    assert work_package_contract_sha256(from_json) == work_package_contract_sha256(from_yaml)

def test_segment_aware_globs_do_not_cross_directories() -> None:
    assert package_path_matches("docs/readme.md", "docs/*.md")
    assert not package_path_matches("docs/private/readme.md", "docs/*.md")
    assert package_path_matches("docs/private/readme.md", "docs/**/*.md")


def test_context_receipt_is_deterministic_compact_and_schema_valid(tmp_path: Path) -> None:
    repo, state, _, session_id, _ = _setup(tmp_path)
    work, session, bootstrap = _records(state, session_id)
    first = build_context_receipt(work_record=work, session_record=session, bootstrap_record=bootstrap, repo_root=repo)
    second = build_context_receipt(work_record=work, session_record=session, bootstrap_record=bootstrap, repo_root=repo)
    assert first == second
    assert verify_receipt_seal(first) == []
    schema = json.loads((ACTORS / "schemas/eiel_context_receipt_v1.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(first)
    text = json.dumps(first, sort_keys=True)
    for forbidden in ("raw_command_output", "search_history", "reasoning_trace", "tool_transcript"):
        assert forbidden not in text
    assert first["scope"]["entry_files"] == ["tools/component.py", "tests/test_component.py"]
    assert first["context_health"]["status"] == "NOT_OBSERVED"


def test_context_receipt_staleness_distinguishes_unrelated_and_relevant_head_drift(tmp_path: Path) -> None:
    repo, state, _, session_id, _ = _setup(tmp_path)
    work, session, bootstrap = _records(state, session_id)
    receipt = build_context_receipt(work_record=work, session_record=session, bootstrap_record=bootstrap, repo_root=repo)

    (repo / "README.unrelated.md").write_text("unrelated\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.unrelated.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "unrelated change"], cwd=repo, check=True)
    unrelated = validate_context_receipt(receipt, work_record=work, session_record=session, repo_root=repo)
    assert unrelated["valid"] is True
    assert unrelated["warnings"]

    (repo / "tools/component.py").write_text("def component_entry():\n    return 'changed'\n", encoding="utf-8")
    subprocess.run(["git", "add", "tools/component.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "relevant change"], cwd=repo, check=True)
    relevant = validate_context_receipt(receipt, work_record=work, session_record=session, repo_root=repo)
    assert relevant["valid"] is False
    assert any("relevant repository paths changed" in row or "file changed" in row for row in relevant["errors"])


def test_missing_required_entry_file_fails_receipt_generation(tmp_path: Path) -> None:
    repo, state, _, session_id, _ = _setup(tmp_path)
    work, session, bootstrap = _records(state, session_id)
    (repo / "tools/component.py").unlink()
    try:
        build_context_receipt(work_record=work, session_record=session, bootstrap_record=bootstrap, repo_root=repo)
    except Exception as exc:
        assert "required initial files are missing" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing required entry file did not fail closed")


def test_workspace_projects_context_health_and_handoff_receipt(tmp_path: Path) -> None:
    repo, state, bootstrap_id, session_id, _ = _setup(tmp_path)
    checkpoint = run_eiel(
        [
            "sessionManager",
            "checkpoint",
            session_id,
            "--tool-call-count",
            "5",
            "--compaction-count",
            "1",
            "--broad-output-events",
            "2",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T00:03:00Z",
        ],
        state_root=state,
    )
    assert checkpoint["session"]["context_health"]["status"] == "HANDOFF_RECOMMENDED"
    opened = run_eiel(
        [
            "workspace",
            "open",
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
        ],
        state_root=state,
    )
    workspace = opened["workspace"]
    assert opened["status"] == "CONTEXT_HANDOFF_RECOMMENDED"
    assert workspace["context_receipt"]["schema_version"] == "eiel_context_receipt_v1"
    assert workspace["initial_context"]["maximum_initial_files"] == 4
    assert opened["next_actions"][0]["command"].startswith("python -m tools.eiel sessionManager context-receipt")

    attached = run_eiel(
        [
            "sessionManager", "context-receipt", session_id,
            "--repo-root", str(repo), "--state-root", str(state), "--attach",
        ],
        state_root=state,
    )
    assert attached["receipt_validation"]["valid"] is True
    assert attached["session"]["context_receipt"]["receipt_id"] == attached["context_receipt"]["receipt_id"]

    handoff = run_eiel(
        ["sessionManager", "handoff", session_id, "--repo-root", str(repo), "--state-root", str(state)],
        state_root=state,
    )
    assert handoff["handoff"]["context_receipt"]["receipt_id"].startswith("CTX-")
    assert Path(handoff["handoff"]["context_receipt_path"]).is_file()


def test_context_health_is_advisory_and_does_not_invent_telemetry() -> None:
    unseen = project_context_health(
        None,
        execution_budget=DEFAULT_EXECUTION_BUDGET,
        session_status="ACTIVE",
        session_outcome=None,
    )
    assert unseen["status"] == "NOT_OBSERVED"
    assert unseen["tool_call_count"] is None
    assert unseen["policy"] == "advisory"


def test_analyzer_counts_one_logical_compaction_and_uses_reported_usage(tmp_path: Path) -> None:
    rows = [
        {"timestamp": "2026-08-04T00:00:00Z", "type": "session_meta", "payload": {"type": "session_meta", "session_id": "x"}},
        {"timestamp": "2026-08-04T00:00:01Z", "type": "event_msg", "payload": {"type": "task_started", "model_context_window": 1000}},
        {"timestamp": "2026-08-04T00:00:02Z", "type": "compacted", "payload": {}},
        {"timestamp": "2026-08-04T00:00:02Z", "type": "event_msg", "payload": {"type": "context_compacted"}},
        {"timestamp": "2026-08-04T00:00:03Z", "type": "event_msg", "payload": {"type": "token_count", "info": {"last_token_usage": {"input_tokens": 5}, "total_token_usage": {"input_tokens": 10, "cached_input_tokens": 8}}}},
    ]
    path = tmp_path / "session.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = analyze_session(path, label="synthetic")
    assert result["context_compactions"] == 1
    assert result["cumulative_model_processing"]["input_tokens"] == 10
    assert result["summed_turn_processing"]["input_tokens"] == 5


def test_context_receipt_rejects_same_head_uncommitted_relevant_drift(tmp_path: Path) -> None:
    repo, state, _, session_id, _ = _setup(tmp_path)
    work, session, bootstrap = _records(state, session_id)
    receipt = build_context_receipt(work_record=work, session_record=session, bootstrap_record=bootstrap, repo_root=repo)
    (repo / "tools/new_relevant_file.py").write_text("VALUE = 1\n", encoding="utf-8")
    result = validate_context_receipt(receipt, work_record=work, session_record=session, repo_root=repo)
    assert result["head_changed"] is False
    assert result["working_state_changed"] is True
    assert result["valid"] is False
    assert any("relevant repository paths changed" in row for row in result["errors"])
