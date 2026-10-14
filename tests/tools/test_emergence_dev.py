from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import tools.emergence_dev.codemods as codemods_module
import tools.emergence_dev.tasks as tasks_module
import tools.emergence_dev.validation as validation_module
from tools.emergence_dev.core import DevAutomationError, build_context
from tools.emergence_dev.codemods import apply_codemod, plan_codemod
from tools.emergence_dev.tasks import (
    begin_task,
    build_change_plan,
    checkpoint_task,
    resume_task,
    scope_add,
    task_status,
    path_hash,
)
from tools.emergence_dev.validation import (
    _compact_receipt,
    finish_task,
    record_external_validation,
    review_task,
    triage_task,
    validate_task,
)


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _config(*, default_budget: int = 5, extended_budget: int = 10, relationship: bool = False) -> str:
    text = f"""\
schema_version = "emergence_dev_config_v1"

[index]
roots = ["engine", "tests"]
config_roots = []
extensions = [".py"]
exclude = ["**/__pycache__/**"]

[limits]
context_tokens = 4000
internal_context_tokens = 3500
candidate_files = 6
default_owned_files = {default_budget}
extended_owned_files = {extended_budget}
max_validation_commands = 10
log_tail_lines = 30

[[critical_rules]]
id = "checkpoint"
paths = ["engine/**/*checkpoint*.py", "engine/**/checkpoint*.py"]
keywords = ["checkpoint"]
notes = ["Check checkpoint compatibility."]
mandatory = true
required_validation = "checkpoint-replay"

[[critical_rules]]
id = "native-cuda-parity"
paths = ["engine/**/*cuda*.py", "engine/**/cuda*.py"]
keywords = ["cuda", "native"]
notes = ["Check native CUDA parity."]
mandatory = true
required_validation = "native-cuda-parity"

[[critical_rules]]
id = "authoritative-schema"
paths = ["engine/**/*schema*.py", "engine/**/schema*.py"]
keywords = ["schema"]
notes = ["Check schema consumers."]
mandatory = true
required_validation = "schema-consumer-compatibility"

[[validation_rules]]
id = "python"
paths = ["**/*.py"]
kind = "compile"
"""
    if relationship:
        text += """\

[[relationships]]
id = "generated"
source = "engine/source.py"
counterparts = ["engine/generated.py"]
generator = [
  "{python}",
  "tools/generate.py",
  "--source",
  "{source}",
  "--output",
  "{output}"
]
"""
    return text


def _project(
    path: Path,
    *,
    default_budget: int = 5,
    extended_budget: int = 10,
    relationship: bool = False,
) -> Path:
    path.mkdir()
    (path / "engine").mkdir()
    (path / "tests").mkdir()
    (path / "pyproject.toml").write_text(
        '[project]\nname = "emergence"\nversion = "0.0.0"\n',
        encoding="utf-8",
    )
    (path / "emergence-dev.toml").write_text(
        _config(
            default_budget=default_budget,
            extended_budget=extended_budget,
            relationship=relationship,
        ),
        encoding="utf-8",
    )
    (path / ".gitignore").write_text(
        "__pycache__/\n.pytest_cache/\n*.pyc\n",
        encoding="utf-8",
    )
    (path / "AGENTS.md").write_text("# Test instructions\n", encoding="utf-8")
    (path / "engine" / "__init__.py").write_text("", encoding="utf-8")
    (path / "engine" / "sample.py").write_text(
        '"""Sample runtime value."""\n\n\ndef runtime_value() -> int:\n    return 1\n',
        encoding="utf-8",
    )
    (path / "tests" / "test_sample.py").write_text(
        "from engine.sample import runtime_value\n\n\ndef test_runtime_value() -> None:\n"
        "    assert runtime_value() == 1\n",
        encoding="utf-8",
    )
    if relationship:
        (path / "tools").mkdir()
        (path / "tools" / "generate.py").write_text(
            "import argparse\n"
            "from pathlib import Path\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--source', type=Path, required=True)\n"
            "p.add_argument('--output', type=Path, required=True)\n"
            "a = p.parse_args()\n"
            "a.output.write_text(a.source.read_text(encoding='utf-8').replace('VALUE', 'GENERATED_VALUE'), encoding='utf-8')\n",
            encoding="utf-8",
        )
        (path / "engine" / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
        (path / "engine" / "generated.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(path, "init", "-q")
    _git(path, "add", ".")
    _git(
        path,
        "-c",
        "user.name=Emergence Test",
        "-c",
        "user.email=emergence-test@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    return path


def test_prepare_is_bounded_advisory_and_relocatable(tmp_path: Path) -> None:
    first = _project(tmp_path / "first")
    second = tmp_path / "second"
    shutil.copytree(first, second, ignore=shutil.ignore_patterns(".git"))
    state_a = tmp_path / "state-a"
    state_b = tmp_path / "state-b"

    context_a = build_context(
        "repair runtime value resolution and its tests",
        project_root=first,
        state_root=state_a,
    )
    context_b = build_context(
        "repair runtime value resolution and its tests",
        project_root=second,
        state_root=state_b,
    )

    assert context_a["schema_version"] == "emergence_dev_context_v1"
    assert context_a["advisory"] is True
    assert context_a["estimated_context_tokens"] <= 4000
    assert context_a["candidates"][0]["path"] == "engine/sample.py"
    assert any(row["path"] == "tests/test_sample.py" for row in context_a["suggested_tests"])
    assert context_a["index_sha256"] == context_b["index_sha256"]
    assert _git(first, "status", "--short") == ""


def test_prepare_cache_is_reused_for_unchanged_tree(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    state = tmp_path / "state"
    first = build_context("runtime value", project_root=root, state_root=state)
    cache = state / "cache" / "source_index_v1.json"
    first_mtime = cache.stat().st_mtime_ns
    second = build_context("runtime value", project_root=root, state_root=state)
    assert cache.stat().st_mtime_ns == first_mtime
    assert first["index_sha256"] == second["index_sha256"]


def test_begin_requires_explicit_adoption_of_dirty_path(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    state = tmp_path / "state"
    source = root / "engine" / "sample.py"
    source.write_text(source.read_text(encoding="utf-8") + "\n# existing dirt\n", encoding="utf-8")
    digest = path_hash(root, "engine/sample.py")

    with pytest.raises(DevAutomationError, match="requires --adopt"):
        begin_task(
            "repair runtime value",
            ["engine/sample.py"],
            project_root=root,
            state_root=state,
        )

    created = begin_task(
        "repair runtime value",
        ["engine/sample.py"],
        project_root=root,
        state_root=state,
        adoptions=[f"engine/sample.py={digest}"],
    )
    status = task_status(created["task_id"], state_root=state)
    assert status["owned_delta"] == []
    source.write_text(source.read_text(encoding="utf-8") + "# task change\n", encoding="utf-8")
    assert task_status(created["task_id"], state_root=state)["owned_delta"] == ["engine/sample.py"]


def test_nested_project_requires_explicit_adoption_of_dirty_path(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    project_parent = repository / "experiments" / "projects"
    project_parent.mkdir(parents=True)
    root = _project(project_parent / "emergence")
    shutil.rmtree(root / ".git")
    _git(repository, "init", "-q")
    _git(repository, "add", ".")
    _git(
        repository,
        "-c",
        "user.name=Emergence Test",
        "-c",
        "user.email=emergence-test@example.invalid",
        "commit",
        "-qm",
        "nested fixture",
    )

    source = root / "engine" / "sample.py"
    source.write_text(source.read_text(encoding="utf-8") + "\nVALUE = 2\n", encoding="utf-8")
    assert tasks_module.git_status(root) == {"engine/sample.py": " M"}

    state = tmp_path / "state"
    with pytest.raises(
        DevAutomationError,
        match=r"already-dirty owned path requires --adopt engine/sample\.py=",
    ):
        begin_task(
            "nested dirty edit",
            ["engine/sample.py"],
            project_root=root,
            state_root=state,
        )

    digest = path_hash(root, "engine/sample.py")
    assert digest is not None
    created = begin_task(
        "nested dirty edit",
        ["engine/sample.py"],
        project_root=root,
        state_root=state,
        adoptions=[f"engine/sample.py={digest}"],
    )
    baseline = json.loads(
        (Path(created["task_dir"]) / "baseline.json").read_text(encoding="utf-8")
    )
    assert baseline["preexisting_dirty"] == {"engine/sample.py": " M"}
    assert baseline["adopted"] == ["engine/sample.py"]


def test_scope_budget_is_soft_then_hash_gated(tmp_path: Path) -> None:
    root = _project(tmp_path / "project", default_budget=2, extended_budget=3)
    for name in ("two.py", "three.py"):
        (root / "engine" / name).write_text(f'NAME = "{name}"\n', encoding="utf-8")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Emergence Test",
        "-c",
        "user.email=emergence-test@example.invalid",
        "commit",
        "-qm",
        "more files",
    )
    state = tmp_path / "state"
    task = begin_task(
        "change several files",
        ["engine/sample.py"],
        project_root=root,
        state_root=state,
    )["task_id"]
    added = scope_add(
        task,
        "engine/two.py",
        "test",
        justification="add the direct regression test dependency",
        state_root=state,
    )
    assert added["owned_count"] == 2
    with pytest.raises(DevAutomationError, match="approve-scope") as caught:
        scope_add(
            task,
            "engine/three.py",
            "cross-surface",
            justification="the public consumer must change with the implementation",
            state_root=state,
        )
    approval = re.search(r"--approve-scope ([0-9a-f]{64})", str(caught.value))
    assert approval
    added = scope_add(
        task,
        "engine/three.py",
        "cross-surface",
        justification="the public consumer must change with the implementation",
        state_root=state,
        approve_scope=approval.group(1),
    )
    assert added["owned_count"] == 3


def test_change_plan_selects_only_tests_related_to_owned_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path / "project")
    state = tmp_path / "state"
    created = begin_task(
        "repair task ownership",
        ["engine/sample.py", "tests/test_sample.py"],
        project_root=root,
        state_root=state,
    )
    context = build_context(
        "repair task ownership",
        project_root=root,
        state_root=state,
    )
    context["suggested_tests"] = [
        {
            "path": "tests/test_sample.py",
            "reason": "direct_import",
            "imports": ["engine.sample"],
        },
        {
            "path": "tests/test_unrelated.py",
            "reason": "direct_import",
            "imports": ["engine.unrelated"],
        },
    ]
    monkeypatch.setattr(tasks_module, "build_context", lambda *args, **kwargs: context)

    plan = build_change_plan(created["task_id"], state_root=state)
    assert plan["selected_tests"] == [
        {"path": "tests/test_sample.py", "reason": "owned_test"}
    ]


def test_generated_counterpart_blocks_incomplete_scope(tmp_path: Path) -> None:
    root = _project(tmp_path / "project", relationship=True)
    state = tmp_path / "state"
    task = begin_task(
        "update generated source",
        ["engine/source.py"],
        project_root=root,
        state_root=state,
    )["task_id"]
    (root / "engine" / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    plan = build_change_plan(task, state_root=state)
    assert plan["status"] == "blocked"
    assert any("engine/generated.py" in blocker for blocker in plan["blockers"])


def test_validation_review_finish_and_resume(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    state = tmp_path / "state"
    task = begin_task(
        "adjust runtime value implementation",
        ["engine/sample.py"],
        project_root=root,
        state_root=state,
    )["task_id"]
    source = root / "engine" / "sample.py"
    source.write_text(
        '"""Sample runtime value."""\n\n\ndef runtime_value() -> int:\n    value = 1\n    return value\n',
        encoding="utf-8",
    )

    plan = build_change_plan(task, state_root=state)
    assert plan["status"] == "ready"
    assert any(row["path"] == "tests/test_sample.py" for row in plan["selected_tests"])
    validation = validate_task(task, state_root=state)
    assert validation["status"] == "passed"
    triage = triage_task(task, state_root=state)
    assert triage["failure_groups"] == []
    review = review_task(task, state_root=state)
    assert review["status"] == "passed"
    closeout = finish_task(task, state_root=state)
    assert Path(closeout["json_path"]).is_file()
    assert Path(closeout["markdown_path"]).is_file()

    checkpoint_task(
        task,
        note="validated",
        next_command=f"emergence-dev change review --task {task}",
        state_root=state,
    )
    resumed = resume_task(task, state_root=state)
    assert resumed["drift"] is False
    assert resumed["next_command"].endswith(task)


def test_external_validation_closes_only_current_verified_obligation(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path / "project")
    state = tmp_path / "state"
    cuda_source = root / "engine" / "perf" / "cuda_feature.py"
    task = begin_task(
        "verify CUDA parity",
        ["engine/perf/cuda_feature.py", "tests/test_sample.py"],
        project_root=root,
        state_root=state,
    )["task_id"]
    cuda_source.parent.mkdir(parents=True, exist_ok=True)
    cuda_source.write_text("CUDA_PARITY = True\n", encoding="utf-8")

    initial = build_change_plan(task, state_root=state)
    assert initial["mandatory_outstanding_obligations"][0]["id"] == (
        "native-cuda-parity"
    )
    failed = record_external_validation(
        task,
        "native-cuda-parity",
        [sys.executable, "-m", "pytest", "-q", "tests/missing.py"],
        evidence_kind="pytest",
        state_root=state,
    )
    assert failed["status"] == "failed"
    assert build_change_plan(
        task, state_root=state
    )["mandatory_outstanding_obligations"]

    passed = record_external_validation(
        task,
        "native-cuda-parity",
        [sys.executable, "-m", "pytest", "-q", "tests/test_sample.py"],
        evidence_kind="pytest",
        state_root=state,
    )
    assert passed["status"] == "passed"
    assert passed["junit_summary"] == {
        "tests": 1,
        "passed": 1,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    verified = build_change_plan(task, state_root=state)
    assert verified["mandatory_outstanding_obligations"] == []
    assert verified["critical_obligations"][0]["status"] == "verified"
    assert validate_task(task, state_root=state)["status"] == "passed"
    assert review_task(task, state_root=state)["status"] == "passed"

    Path(passed["junit_path"]).write_text("<tampered />", encoding="utf-8")
    assert build_change_plan(
        task, state_root=state
    )["mandatory_outstanding_obligations"]

    cuda_source.write_text("CUDA_PARITY = False\n", encoding="utf-8")
    stale = build_change_plan(task, state_root=state)
    assert stale["mandatory_outstanding_obligations"]


def test_failure_triage_and_stale_validation(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    state = tmp_path / "state"
    task = begin_task(
        "break runtime value for triage",
        ["engine/sample.py"],
        project_root=root,
        state_root=state,
    )["task_id"]
    source = root / "engine" / "sample.py"
    source.write_text(
        '"""Sample runtime value."""\n\n\ndef runtime_value() -> int:\n    return 2\n',
        encoding="utf-8",
    )
    validation = validate_task(task, state_root=state)
    assert validation["status"] == "failed"
    triage = triage_task(task, state_root=state)
    assert any(row["kind"] == "pytest" for row in triage["failure_groups"])
    assert review_task(task, state_root=state)["status"] == "blocked"

    source.write_text(
        '"""Sample runtime value."""\n\n\ndef runtime_value() -> int:\n    return 1\n',
        encoding="utf-8",
    )
    assert validate_task(task, state_root=state)["status"] == "passed"
    source.write_text(source.read_text(encoding="utf-8") + "\n# later drift\n", encoding="utf-8")
    review = review_task(task, state_root=state)
    assert any("stale" in blocker for blocker in review["blockers"])


def test_review_detects_undeclared_and_untracked_hardcoded_paths(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    state = tmp_path / "state"
    task = begin_task(
        "add portable helper",
        ["engine/portable.py"],
        project_root=root,
        state_root=state,
    )["task_id"]
    (root / "engine" / "portable.py").write_text(
        'ROOT = "/tmp/emergence-test/user/causyl/runtime"\n',
        encoding="utf-8",
    )
    (root / "engine" / "undeclared.py").write_text("VALUE = 1\n", encoding="utf-8")
    plan = build_change_plan(task, state_root=state)
    assert plan["status"] == "blocked"
    assert "engine/undeclared.py" in plan["blockers"][0]

    (root / "engine" / "undeclared.py").unlink()
    validation = validate_task(task, state_root=state)
    assert validation["status"] == "passed"
    review = review_task(task, state_root=state)
    assert any("fixed Causyl path" in blocker for blocker in review["blockers"])


def test_codemod_exact_insert_atomic_idempotent_and_stale(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    state = tmp_path / "state"
    task = begin_task(
        "mechanically update runtime value",
        ["engine/sample.py"],
        project_root=root,
        state_root=state,
    )["task_id"]
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": "emergence_dev_codemod_spec_v1",
                "operations": [
                    {
                        "kind": "replace_exact",
                        "path": "engine/sample.py",
                        "old": "    return 1\n",
                        "new": "    value = 1\n    return value\n",
                        "expected_count": 1,
                    },
                    {
                        "kind": "insert_after",
                        "path": "engine/sample.py",
                        "anchor": '"""Sample runtime value."""',
                        "text": "\n# Generated by a reviewed mechanical plan.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    planned = plan_codemod(task, spec, state_root=state)
    planned_payload = json.loads(Path(planned["plan_path"]).read_text(encoding="utf-8"))
    assert planned_payload["application_contract"] == {
        "maximum_live_outputs": 1,
        "atomicity": "exception_atomic",
        "process_crash_recovery": False,
    }
    applied = apply_codemod(
        task,
        planned["plan_path"],
        planned["plan_sha256"],
        state_root=state,
    )
    assert applied["status"] == "applied"
    assert "value = 1" in (root / "engine" / "sample.py").read_text(encoding="utf-8")
    repeated = apply_codemod(
        task,
        planned["plan_path"],
        planned["plan_sha256"],
        state_root=state,
    )
    assert repeated["status"] == "already_applied"

    stale_spec = tmp_path / "stale.json"
    stale_spec.write_text(
        json.dumps(
            {
                "schema_version": "emergence_dev_codemod_spec_v1",
                "operations": [
                    {
                        "kind": "replace_exact",
                        "path": "engine/sample.py",
                        "old": "value = 1",
                        "new": "value = int(1)",
                        "expected_count": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    stale = plan_codemod(task, stale_spec, state_root=state)
    source = root / "engine" / "sample.py"
    source.write_text(source.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
    with pytest.raises(DevAutomationError, match="stale"):
        apply_codemod(task, stale["plan_path"], stale["plan_sha256"], state_root=state)


def test_codemod_runs_only_configured_staged_generator(tmp_path: Path) -> None:
    root = _project(tmp_path / "project", relationship=True)
    state = tmp_path / "state"
    task = begin_task(
        "regenerate counterpart",
        ["engine/source.py", "engine/generated.py"],
        project_root=root,
        state_root=state,
    )["task_id"]
    (root / "engine" / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    spec = tmp_path / "generator.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": "emergence_dev_codemod_spec_v1",
                "operations": [
                    {
                        "kind": "run_generator",
                        "relationship_id": "generated",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    planned = plan_codemod(task, spec, state_root=state)
    assert (root / "engine" / "generated.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    applied = apply_codemod(
        task,
        planned["plan_path"],
        planned["plan_sha256"],
        state_root=state,
    )
    assert applied["status"] == "applied"
    assert (root / "engine" / "generated.py").read_text(encoding="utf-8") == "GENERATED_VALUE = 2\n"


def _replace_spec(path: Path, operations: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "emergence_dev_codemod_spec_v1",
                "operations": operations,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_context_uses_conservative_internal_ceiling_and_keeps_warnings(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    dynamic = root / "engine" / "simulation" / "checkpoint_unicode.py"
    dynamic.parent.mkdir()
    dynamic.write_text(
        "VALUE = '".encode().decode() + ("λ界🙂" * 3000) + "'\nRESULT = getattr(object(), 'x', None)\n",
        encoding="utf-8",
    )
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Emergence Test",
        "-c",
        "user.email=emergence-test@example.invalid",
        "commit",
        "-qm",
        "non ascii fixture",
    )
    state = tmp_path / "state"
    first = build_context("checkpoint λ界🙂 compatibility", project_root=root, state_root=state)
    second = build_context("checkpoint λ界🙂 compatibility", project_root=root, state_root=state)
    assert first["context_sha256"] == second["context_sha256"]
    assert first["estimated_context_tokens"] <= 3500
    assert first["context_budget"]["estimator"] == "utf8_bytes_ceil_div_3"
    assert "checkpoint-replay" in first["context_markdown"]
    assert "Dynamic coupling detected" in first["context_markdown"]


@pytest.mark.parametrize(
    ("relative", "objective", "required"),
    [
        ("engine/simulation/checkpoint_runtime.py", "change checkpoint replay", "checkpoint-replay"),
        ("engine/native/cuda_dispatch.py", "change CUDA dispatch", "native-cuda-parity"),
        ("engine/schemas/public_schema.py", "change schema contract", "schema-consumer-compatibility"),
    ],
)
def test_critical_surfaces_fail_closed_at_review_and_finish(
    tmp_path: Path,
    relative: str,
    objective: str,
    required: str,
) -> None:
    root = _project(tmp_path / "project")
    source = root / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    test_path = root / "tests" / f"test_{source.stem}.py"
    module = relative.removesuffix(".py").replace("/", ".")
    test_path.write_text(f"from {module} import VALUE\n\ndef test_value():\n    assert VALUE == 1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Emergence Test",
        "-c",
        "user.email=emergence-test@example.invalid",
        "commit",
        "-qm",
        "critical fixture",
    )
    state = tmp_path / "state"
    task = begin_task(objective, [relative], project_root=root, state_root=state)["task_id"]
    source.write_text("VALUE = int(1)\n", encoding="utf-8")
    receipt = validate_task(task, state_root=state)
    assert receipt["status"] == "passed"
    assert receipt["validations"]["mandatory_external"][required] == "outstanding"
    review = review_task(task, state_root=state)
    assert any(required in blocker for blocker in review["blockers"])
    with pytest.raises(DevAutomationError, match="mandatory obligation"):
        finish_task(task, state_root=state)


def test_event_chain_detects_tail_deletion_and_corruption(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    state = tmp_path / "state"
    created = begin_task("event integrity", ["engine/sample.py"], project_root=root, state_root=state)
    checkpoint_task(
        created["task_id"],
        note="checkpoint",
        next_command="true",
        state_root=state,
    )
    event_path = Path(created["task_dir"]) / "events.jsonl"
    lines = event_path.read_text(encoding="utf-8").splitlines()
    event_path.write_text(lines[0] + "\n", encoding="utf-8")
    with pytest.raises(DevAutomationError, match="event head"):
        task_status(created["task_id"], state_root=state)

    other = begin_task("event corruption", ["engine/sample.py"], project_root=root, state_root=tmp_path / "other")
    other_events = Path(other["task_dir"]) / "events.jsonl"
    other_events.write_text(
        other_events.read_text(encoding="utf-8").replace("task_started", "task_tampered"),
        encoding="utf-8",
    )
    with pytest.raises(DevAutomationError, match="event chain"):
        task_status(other["task_id"], state_root=tmp_path / "other")


def test_event_head_recovers_one_process_killed_valid_extension_idempotently(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path / "project")
    state = tmp_path / "state"
    created = begin_task(
        "recover one event",
        ["engine/sample.py"],
        project_root=root,
        state_root=state,
    )
    task_dir = Path(created["task_dir"])
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os\n"
                "from pathlib import Path\n"
                "import tools.emergence_dev.tasks as tasks\n"
                "real = tasks.atomic_write_json\n"
                "def interrupt(path, value):\n"
                "    if path.name == 'events_head.json':\n"
                "        os._exit(91)\n"
                "    real(path, value)\n"
                "tasks.atomic_write_json = interrupt\n"
                f"tasks.checkpoint_task({created['task_id']!r}, "
                f"note='durable event', next_command='true', "
                f"state_root=Path({str(state)!r}))\n"
            ),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 91

    first = task_status(created["task_id"], state_root=state)
    assert first["event_count"] == 2
    assert len(first["event_recoveries"]) == 1
    receipt = first["event_recoveries"][0]
    assert receipt["status"] == "recovered_valid_log_extension"
    assert len(receipt["recovered_event_sha256s"]) == 1
    head_after_first = (task_dir / "events_head.json").read_bytes()

    second = task_status(created["task_id"], state_root=state)
    assert second["event_recoveries"] == first["event_recoveries"]
    assert (task_dir / "events_head.json").read_bytes() == head_after_first


def test_event_head_recovers_multiple_valid_extensions(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    state = tmp_path / "state"
    created = begin_task(
        "recover several events",
        ["engine/sample.py"],
        project_root=root,
        state_root=state,
    )
    task_dir = Path(created["task_dir"])
    original_head = json.loads(
        (task_dir / "events_head.json").read_text(encoding="utf-8")
    )
    checkpoint_task(
        created["task_id"],
        note="first",
        next_command="one",
        state_root=state,
    )
    checkpoint_task(
        created["task_id"],
        note="second",
        next_command="two",
        state_root=state,
    )
    tasks_module.atomic_write_json(task_dir / "events_head.json", original_head)

    recovered = task_status(created["task_id"], state_root=state)
    assert recovered["event_count"] == 3
    receipt = recovered["event_recoveries"][0]
    assert receipt["prior_head"]["event_count"] == 1
    assert receipt["recovered_head"]["event_count"] == 3
    assert len(receipt["recovered_event_sha256s"]) == 2


def test_event_head_ahead_broken_chain_and_altered_event_fail_closed(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path / "project")

    ahead_state = tmp_path / "ahead-state"
    ahead = begin_task(
        "head ahead",
        ["engine/sample.py"],
        project_root=root,
        state_root=ahead_state,
    )
    ahead_dir = Path(ahead["task_dir"])
    head = json.loads((ahead_dir / "events_head.json").read_text(encoding="utf-8"))
    tasks_module.atomic_write_json(
        ahead_dir / "events_head.json",
        {**head, "event_count": 2, "last_event_sha256": "f" * 64},
    )
    with pytest.raises(DevAutomationError, match="ahead of append-only log"):
        task_status(ahead["task_id"], state_root=ahead_state)

    broken_state = tmp_path / "broken-state"
    broken = begin_task(
        "broken previous hash",
        ["engine/sample.py"],
        project_root=root,
        state_root=broken_state,
    )
    broken_dir = Path(broken["task_dir"])
    row = {
        "schema_version": tasks_module.EVENT_VERSION,
        "sequence": 2,
        "created_at": "2026-07-26T00:00:00+00:00",
        "kind": "checkpoint",
        "data": {"note": "broken"},
        "previous_event_sha256": "0" * 64,
    }
    row["event_sha256"] = tasks_module.canonical_sha256(row)
    with (broken_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(tasks_module.canonical_json_line(row))
    with pytest.raises(DevAutomationError, match="event chain"):
        task_status(broken["task_id"], state_root=broken_state)

    altered_state = tmp_path / "altered-state"
    altered = begin_task(
        "altered event",
        ["engine/sample.py"],
        project_root=root,
        state_root=altered_state,
    )
    altered_events = Path(altered["task_dir"]) / "events.jsonl"
    altered_events.write_text(
        altered_events.read_text(encoding="utf-8").replace(
            "task_started", "task_altered"
        ),
        encoding="utf-8",
    )
    with pytest.raises(DevAutomationError, match="event chain"):
        task_status(altered["task_id"], state_root=altered_state)


def test_deleted_valid_event_tail_remains_rejected(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    state = tmp_path / "state"
    created = begin_task(
        "deleted event tail",
        ["engine/sample.py"],
        project_root=root,
        state_root=state,
    )
    checkpoint_task(
        created["task_id"],
        note="must not disappear",
        next_command="true",
        state_root=state,
    )
    event_path = Path(created["task_dir"]) / "events.jsonl"
    lines = event_path.read_text(encoding="utf-8").splitlines()
    event_path.write_text(lines[0] + "\n", encoding="utf-8")
    with pytest.raises(DevAutomationError, match="ahead of append-only log"):
        task_status(created["task_id"], state_root=state)


def test_ownership_fingerprints_detect_mode_delete_rename_and_symlink_target(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    link = root / "engine" / "owned-link.py"
    link.symlink_to("sample.py")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Emergence Test",
        "-c",
        "user.email=emergence-test@example.invalid",
        "commit",
        "-qm",
        "link fixture",
    )
    state = tmp_path / "state"
    task = begin_task(
        "fingerprint ownership",
        ["engine/sample.py", "engine/owned-link.py"],
        project_root=root,
        state_root=state,
    )["task_id"]
    os.chmod(root / "engine" / "sample.py", 0o755)
    link.unlink()
    link.symlink_to("__init__.py")
    assert task_status(task, state_root=state)["owned_delta"] == [
        "engine/owned-link.py",
        "engine/sample.py",
    ]
    (root / "engine" / "sample.py").unlink()
    (root / "engine" / "renamed.py").write_text("VALUE = 1\n", encoding="utf-8")
    status = task_status(task, state_root=state)
    assert "engine/sample.py" in status["owned_delta"]
    assert "engine/renamed.py" in status["new_undeclared_dirty"]


def test_unrelated_dirty_state_is_preserved_and_drift_blocks(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    unrelated = root / "notes.txt"
    unrelated.write_text("preexisting\n", encoding="utf-8")
    state = tmp_path / "state"
    task = begin_task("owned edit", ["engine/sample.py"], project_root=root, state_root=state)["task_id"]
    (root / "engine" / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    assert unrelated.read_text(encoding="utf-8") == "preexisting\n"
    unrelated.write_text("drifted\n", encoding="utf-8")
    plan = build_change_plan(task, state_root=state)
    assert any("pre-existing dirty paths drifted" in blocker for blocker in plan["blockers"])
    assert unrelated.read_text(encoding="utf-8") == "drifted\n"


def test_validation_plan_drift_and_malformed_junit_are_explicit(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    extra = root / "engine" / "extra.py"
    extra.write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Emergence Test",
        "-c",
        "user.email=emergence-test@example.invalid",
        "commit",
        "-qm",
        "extra fixture",
    )
    state = tmp_path / "state"
    task = begin_task("validation drift", ["engine/sample.py"], project_root=root, state_root=state)["task_id"]
    source = root / "engine" / "sample.py"
    source.write_text(source.read_text(encoding="utf-8") + "\n# validated\n", encoding="utf-8")
    receipt = validate_task(task, state_root=state)
    assert receipt["status"] == "passed"
    scope_add(
        task,
        "engine/extra.py",
        "cross-surface",
        justification="the reviewed plan now includes a second consumer",
        state_root=state,
    )
    assert any("change plan" in blocker for blocker in review_task(task, state_root=state)["blockers"])

    failing_root = _project(tmp_path / "failing")
    failing_state = tmp_path / "failing-state"
    failing_task = begin_task(
        "malformed junit",
        ["engine/sample.py"],
        project_root=failing_root,
        state_root=failing_state,
    )["task_id"]
    failing_source = failing_root / "engine" / "sample.py"
    failing_source.write_text(failing_source.read_text(encoding="utf-8").replace("return 1", "return 2"), encoding="utf-8")
    failed = validate_task(failing_task, state_root=failing_state)
    Path(failed["junit_path"]).write_text("<broken", encoding="utf-8")
    triage = triage_task(failing_task, state_root=failing_state)
    assert any(row["kind"] == "junit_parse_error" for row in triage["failure_groups"])


def test_validation_preserves_venv_interpreter_and_serializes_native_builds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project(tmp_path / "project")
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(sys.executable)
    monkeypatch.setattr(validation_module.sys, "executable", str(venv_python))
    plan, _ = validation_module._validation_plan(
        "task",
        {
            "owned": ["engine/sample.py", "tests/test_sample.py"],
            "owned_delta": ["engine/sample.py"],
            "selected_tests": [
                {"path": "tests/test_sample.py", "reason": "owned_test"}
            ],
            "plan_sha256": "a" * 64,
        },
        project_root=root,
        task_dir=tmp_path / "task",
    )

    compile_step = next(row for row in plan["steps"] if row["id"] == "python-compile")
    pytest_step = next(row for row in plan["steps"] if row["id"] == "targeted-pytest")
    assert compile_step["argv"][0] == str(venv_python)
    assert pytest_step["argv"][0] == str(venv_python)
    assert pytest_step["env"]["MAX_JOBS"] == "1"


def test_validation_digest_uses_original_output_while_tail_is_redacted() -> None:
    raw = {
        "status": "passed",
        "plan_sha256": "a" * 64,
        "steps": [
            {
                "id": "one",
                "status": "passed",
                "returncode": 0,
                "argv": ["tool"],
                "cwd": ".",
                "stdout": "token=super-secret",
                "stderr": "",
            }
        ],
    }
    receipt = _compact_receipt(
        "task",
        {"plan_sha256": "b" * 64, "mandatory_outstanding_obligations": []},
        raw,
        owned_hashes={},
        junit_path=None,
        tail_lines=10,
    )
    assert receipt["steps"][0]["stdout_sha256"] == __import__("hashlib").sha256(
        b"token=super-secret"
    ).hexdigest()
    assert "super-secret" not in receipt["steps"][0]["stdout_tail"]


def test_codemod_rejects_unsafe_plans_without_tracked_writes(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    state = tmp_path / "state"
    task = begin_task("safe codemod", ["engine/sample.py"], project_root=root, state_root=state)["task_id"]
    before = _git(root, "status", "--short")
    nonunique = _replace_spec(
        tmp_path / "nonunique.json",
        [{"kind": "insert_after", "path": "engine/sample.py", "anchor": "\n", "text": "# x\n"}],
    )
    with pytest.raises(DevAutomationError, match="requires one anchor"):
        plan_codemod(task, nonunique, state_root=state)
    undeclared = _replace_spec(
        tmp_path / "undeclared.json",
        [{"kind": "replace_exact", "path": "tests/test_sample.py", "old": "1", "new": "2", "expected_count": 1}],
    )
    with pytest.raises(DevAutomationError, match="not task-owned"):
        plan_codemod(task, undeclared, state_root=state)
    valid = _replace_spec(
        tmp_path / "valid.json",
        [{"kind": "replace_exact", "path": "engine/sample.py", "old": "return 1", "new": "return int(1)", "expected_count": 1}],
    )
    planned = plan_codemod(task, valid, state_root=state)
    assert _git(root, "status", "--short") == before
    with pytest.raises(DevAutomationError, match="reviewed hash"):
        apply_codemod(task, planned["plan_path"], "0" * 64, state_root=state)


def test_codemod_head_drift_symlink_and_shell_metacharacters_fail_safe(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    metachar = root / "engine" / "value;touch PWNED.py"
    metachar.write_text("VALUE = 1\n", encoding="utf-8")
    link = root / "engine" / "link.py"
    link.symlink_to("sample.py")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Emergence Test",
        "-c",
        "user.email=emergence-test@example.invalid",
        "commit",
        "-qm",
        "unsafe names fixture",
    )
    state = tmp_path / "state"
    task = begin_task(
        "metacharacter safety",
        ["engine/sample.py", "engine/value;touch PWNED.py"],
        project_root=root,
        state_root=state,
    )["task_id"]
    spec = _replace_spec(
        tmp_path / "meta.json",
        [{"kind": "replace_exact", "path": "engine/value;touch PWNED.py", "old": "1", "new": "2", "expected_count": 1}],
    )
    planned = plan_codemod(task, spec, state_root=state)
    apply_codemod(task, planned["plan_path"], planned["plan_sha256"], state_root=state)
    assert not (root / "PWNED.py").exists()

    link_state = tmp_path / "link-state"
    link_task = begin_task("link safety", ["engine/link.py"], project_root=root, state_root=link_state)["task_id"]
    link_spec = _replace_spec(
        tmp_path / "link.json",
        [{"kind": "replace_exact", "path": "engine/link.py", "old": "return 1", "new": "return 2", "expected_count": 1}],
    )
    with pytest.raises(DevAutomationError, match="symlink"):
        plan_codemod(link_task, link_spec, state_root=link_state)

    drift_state = tmp_path / "drift-state"
    drift_task = begin_task("head drift", ["engine/sample.py"], project_root=root, state_root=drift_state)["task_id"]
    drift_spec = _replace_spec(
        tmp_path / "drift.json",
        [{"kind": "replace_exact", "path": "engine/sample.py", "old": "return 1", "new": "return int(1)", "expected_count": 1}],
    )
    drift_plan = plan_codemod(drift_task, drift_spec, state_root=drift_state)
    (root / "AGENTS.md").write_text("# changed\n", encoding="utf-8")
    _git(root, "add", "AGENTS.md")
    _git(
        root,
        "-c",
        "user.name=Emergence Test",
        "-c",
        "user.email=emergence-test@example.invalid",
        "commit",
        "-qm",
        "head drift",
    )
    with pytest.raises(DevAutomationError, match="drift"):
        apply_codemod(
            drift_task,
            drift_plan["plan_path"],
            drift_plan["plan_sha256"],
            state_root=drift_state,
        )


def test_codemod_multifile_rejected_and_single_file_exception_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path / "project")
    second = root / "engine" / "two.py"
    second.write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Emergence Test",
        "-c",
        "user.email=emergence-test@example.invalid",
        "commit",
        "-qm",
        "two files",
    )
    state = tmp_path / "state"
    task = begin_task(
        "exception atomic transform",
        ["engine/sample.py", "engine/two.py"],
        project_root=root,
        state_root=state,
    )["task_id"]
    spec = _replace_spec(
        tmp_path / "multi.json",
        [
            {"kind": "replace_exact", "path": "engine/sample.py", "old": "return 1", "new": "return int(1)", "expected_count": 1},
            {"kind": "replace_exact", "path": "engine/two.py", "old": "1", "new": "2", "expected_count": 1},
        ],
    )
    planned = plan_codemod(task, spec, state_root=state)
    originals = {
        path: (root / path).read_bytes()
        for path in ("engine/sample.py", "engine/two.py")
    }
    with pytest.raises(DevAutomationError, match="requires exactly one live output"):
        apply_codemod(
            task,
            planned["plan_path"],
            planned["plan_sha256"],
            state_root=state,
        )
    assert all((root / path).read_bytes() == original for path, original in originals.items())

    single_spec = _replace_spec(
        tmp_path / "single.json",
        [
            {
                "kind": "replace_exact",
                "path": "engine/sample.py",
                "old": "return 1",
                "new": "return int(1)",
                "expected_count": 1,
            }
        ],
    )
    single = plan_codemod(task, single_spec, state_root=state)
    real_replace = codemods_module.os.replace

    def fail_live_replace(source: object, target: object) -> None:
        if (
            Path(target).name == "sample.py"
            and "emergence-dev-" in Path(source).name
            and "rollback" not in Path(source).name
        ):
            raise OSError("injected replacement failure")
        real_replace(source, target)

    monkeypatch.setattr(codemods_module.os, "replace", fail_live_replace)
    with pytest.raises(OSError, match="injected"):
        apply_codemod(
            task,
            single["plan_path"],
            single["plan_sha256"],
            state_root=state,
        )
    assert all((root / path).read_bytes() == original for path, original in originals.items())
    monkeypatch.setattr(codemods_module.os, "replace", real_replace)

    real_hash = codemods_module.path_hash

    def mismatching_hash(project: Path, relative: str) -> str | None:
        if "int(1)" in (project / relative).read_text(encoding="utf-8"):
            return "f" * 64
        return real_hash(project, relative)

    monkeypatch.setattr(codemods_module, "path_hash", mismatching_hash)
    with pytest.raises(DevAutomationError, match="postimage"):
        apply_codemod(
            task,
            single["plan_path"],
            single["plan_sha256"],
            state_root=state,
        )
    assert all((root / path).read_bytes() == original for path, original in originals.items())


def test_failing_generator_never_writes_live_output(tmp_path: Path) -> None:
    root = _project(tmp_path / "project", relationship=True)
    generator = root / "tools" / "generate.py"
    generator.write_text("raise SystemExit(7)\n", encoding="utf-8")
    _git(root, "add", "tools/generate.py")
    _git(
        root,
        "-c",
        "user.name=Emergence Test",
        "-c",
        "user.email=emergence-test@example.invalid",
        "commit",
        "-qm",
        "failing generator",
    )
    state = tmp_path / "state"
    task = begin_task(
        "generator failure",
        ["engine/source.py", "engine/generated.py"],
        project_root=root,
        state_root=state,
    )["task_id"]
    (root / "engine" / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    before = (root / "engine" / "generated.py").read_bytes()
    spec = _replace_spec(
        tmp_path / "generator.json",
        [{"kind": "run_generator", "relationship_id": "generated"}],
    )
    with pytest.raises(DevAutomationError, match="generator failed"):
        plan_codemod(task, spec, state_root=state)
    assert (root / "engine" / "generated.py").read_bytes() == before
