"""Targeted validation, failure triage, diff review, and final receipts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from engine.validation_plans import execute_plan

from .core import (
    DevAutomationError,
    atomic_write_json,
    canonical_sha256,
    load_config,
    resolve_state_root,
)
from .tasks import (
    append_event,
    build_change_plan,
    load_task,
    task_events,
    task_status,
)


VALIDATION_RECEIPT_VERSION = "emergence_dev_validation_receipt_v1"
TRIAGE_VERSION = "emergence_dev_triage_v1"
REVIEW_VERSION = "emergence_dev_review_v1"
CLOSEOUT_VERSION = "emergence_dev_closeout_v1"
EXTERNAL_VALIDATION_RECEIPT_VERSION = (
    "emergence_dev_external_validation_receipt_v1"
)
CPU_VALIDATION_INHERITED_ENV = (
    "COMPILER_PATH",
    "CPLUS_INCLUDE_PATH",
    "GCC_EXEC_PREFIX",
    "LIBRARY_PATH",
    "PATH",
)
_SECRET_RE = re.compile(
    r"(?i)(authorization:\s*bearer\s+)[^\s]+|"
    r"((?:api[_-]?key|password|passwd|secret|token)\s*[=:]\s*)[^\s,;]+"
)
_LOCATION_RE = re.compile(r"(?P<path>(?:[A-Za-z]:)?[^:\n ]+\.py):(?P<line>[0-9]+)")


def _active_python_executable() -> str:
    return os.path.abspath(sys.executable)


def _redact(value: str) -> str:
    return _SECRET_RE.sub(lambda match: (match.group(1) or match.group(2) or "") + "[REDACTED]", value)


def _tail(value: str, limit: int) -> str:
    lines = value.splitlines()
    return "\n".join(lines[-limit:])


def _step_id(prefix: str, value: str) -> str:
    rendered = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return f"{prefix}-{rendered}"[:64].rstrip("-")


def record_external_validation(
    task_id: str,
    obligation_id: str,
    command: Iterable[str],
    *,
    evidence_kind: str,
    timeout_seconds: int = 1800,
    state_root: str | Path | None = None,
) -> dict[str, Any]:
    """Execute and bind one mandatory external validation to current scope."""

    resolved_state = resolve_state_root(state_root)
    task_dir, baseline = load_task(resolved_state, task_id)
    root = Path(baseline["project_root"])
    status = task_status(task_id, state_root=resolved_state)
    plan = build_change_plan(task_id, state_root=resolved_state)
    obligation = next(
        (
            row
            for row in plan["critical_obligations"]
            if row["id"] == obligation_id and row["mandatory"] is True
        ),
        None,
    )
    if obligation is None:
        raise DevAutomationError(
            f"mandatory external obligation is not active: {obligation_id}"
        )
    argv = [str(value) for value in command]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv or any(not value for value in argv):
        raise DevAutomationError("external validation command is empty")
    if evidence_kind != "pytest":
        raise DevAutomationError(
            "external validation evidence kind is unsupported"
        )
    if any(
        value == "--junitxml" or value.startswith("--junitxml=")
        for value in argv
    ):
        raise DevAutomationError(
            "external pytest validation owns its JUnit output path"
        )
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 3600:
        raise DevAutomationError(
            "external validation timeout must be between 1 and 3600 seconds"
        )
    if _SECRET_RE.search(" ".join(argv)):
        raise DevAutomationError(
            "external validation command appears to contain secret material"
        )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    junit_path = (
        task_dir
        / "external_validation"
        / obligation_id
        / f"{run_id}.junit.xml"
    )
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    executed_argv = [*argv, f"--junitxml={junit_path}"]
    try:
        completed = subprocess.run(
            executed_argv,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        stderr = f"{stderr}\nexternal validation timed out".strip()
    junit_summary: dict[str, int] | None = None
    junit_sha256: str | None = None
    try:
        junit_root = ET.parse(junit_path).getroot()
        suites = [
            row
            for row in junit_root.iter("testsuite")
            if not list(row.iterfind("testsuite"))
        ]
        tests = sum(int(row.get("tests", "0")) for row in suites)
        failures = sum(int(row.get("failures", "0")) for row in suites)
        errors = sum(int(row.get("errors", "0")) for row in suites)
        skipped = sum(int(row.get("skipped", "0")) for row in suites)
        junit_summary = {
            "tests": tests,
            "passed": tests - failures - errors - skipped,
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
        }
        junit_sha256 = hashlib.sha256(junit_path.read_bytes()).hexdigest()
    except (OSError, ET.ParseError, TypeError, ValueError):
        junit_summary = None
    evidence_passed = (
        returncode == 0
        and junit_summary is not None
        and junit_summary["passed"] > 0
        and junit_summary["failures"] == 0
        and junit_summary["errors"] == 0
        and junit_summary["skipped"] == 0
    )
    receipt: dict[str, Any] = {
        "schema_version": EXTERNAL_VALIDATION_RECEIPT_VERSION,
        "task_id": task_id,
        "obligation_id": obligation_id,
        "required_validation": obligation["required_validation"],
        "status": "passed" if evidence_passed else "failed",
        "returncode": returncode,
        "evidence_kind": evidence_kind,
        "argv": executed_argv,
        "timeout_seconds": timeout_seconds,
        "baseline_sha256": baseline["baseline_sha256"],
        "head": status["current_head"],
        "owned_hashes": status["owned_hashes"],
        "config_sha256": plan["config_sha256"],
        "junit_path": str(junit_path),
        "junit_sha256": junit_sha256,
        "junit_summary": junit_summary,
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "stdout_tail": _redact(_tail(stdout, 30)),
        "stderr_tail": _redact(_tail(stderr, 30)),
        "completed_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    receipt_dir = task_dir / "external_validation" / obligation_id
    receipt_path = receipt_dir / f"{receipt['receipt_sha256']}.json"
    atomic_write_json(receipt_path, receipt)
    append_event(
        task_dir,
        "external_validation_recorded",
        {
            "receipt": receipt,
            "receipt_path": str(receipt_path),
        },
    )
    return {
        **receipt,
        "receipt_path": str(receipt_path),
    }


def _validation_plan(
    task_id: str,
    plan: Mapping[str, Any],
    *,
    project_root: Path,
    task_dir: Path,
) -> tuple[dict[str, Any], Path | None]:
    config = load_config(project_root)
    owned_delta = list(plan["owned_delta"])
    steps: list[dict[str, Any]] = []
    git = shutil.which("git")
    if git is None:
        raise DevAutomationError("git executable is unavailable")
    steps.append(
        {
            "id": "diff-check",
            "argv": [str(Path(git).resolve()), "diff", "--check", "--", *plan["owned"]],
            "cwd": ".",
            "timeout_seconds": 120,
            "env": {},
        }
    )
    python_files = sorted(path for path in plan["owned"] if path.endswith(".py"))
    if python_files:
        steps.append(
            {
                "id": "python-compile",
                "argv": [
                    _active_python_executable(),
                    "-m",
                    "compileall",
                    "-q",
                    "-f",
                    *python_files,
                ],
                "cwd": ".",
                "timeout_seconds": 300,
                "env": {"PYTHONDONTWRITEBYTECODE": "1"},
            }
        )
    junit_path: Path | None = None
    selected_tests = sorted(
        {
            row["path"]
            for row in plan["selected_tests"]
            if (project_root / row["path"]).is_file()
        }
    )
    if selected_tests:
        junit_path = task_dir / "junit" / f"{plan['plan_sha256']}.xml"
        junit_path.parent.mkdir(parents=True, exist_ok=True)
        steps.append(
            {
                "id": "targeted-pytest",
                "argv": [
                    _active_python_executable(),
                    "-m",
                    "pytest",
                    "-q",
                    "--disable-warnings",
                    f"--junitxml={junit_path}",
                    *selected_tests,
                ],
                "cwd": ".",
                "timeout_seconds": 1800,
                "env": {
                    "MAX_JOBS": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            }
        )
    register = "docs/100_governance-authority/110_authority-registers/EIEL-DOC-000007--eiel-unfinished-work-register.md"
    if register in owned_delta:
        steps.append(
            {
                "id": "register-html-check",
                "argv": [
                    _active_python_executable(),
                    "tools/scripts/render_eiel_unfinished_work_register_html.py",
                    "--check",
                ],
                "cwd": ".",
                "timeout_seconds": 120,
                "env": {"PYTHONDONTWRITEBYTECODE": "1"},
            }
        )
    if len(steps) > config.limits["max_validation_commands"]:
        raise DevAutomationError(
            f"validation plan has {len(steps)} commands, exceeding configured maximum "
            f"{config.limits['max_validation_commands']}"
        )
    plan_id = _step_id("dev", plan["plan_sha256"][:16])
    return (
        {
            "schema_version": "eiel_validation_plan_v1",
            "plan_id": plan_id,
            "description": f"Targeted Emergence development validation for {task_id}",
            "cpu_only": True,
            "defaults": {
                "cwd": ".",
                "timeout_seconds": 300,
                "inherit_env": [
                    key for key in CPU_VALIDATION_INHERITED_ENV if key in os.environ
                ],
                "env": {},
            },
            "steps": steps,
        },
        junit_path,
    )


def _compact_receipt(
    task_id: str,
    change_plan: Mapping[str, Any],
    raw: Mapping[str, Any],
    *,
    owned_hashes: Mapping[str, str | None],
    junit_path: Path | None,
    tail_lines: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for raw_step in raw["steps"]:
        original_stdout = raw_step["stdout"]
        original_stderr = raw_step["stderr"]
        stdout = _redact(original_stdout)
        stderr = _redact(original_stderr)
        rows.append(
            {
                "id": raw_step["id"],
                "status": raw_step["status"],
                "returncode": raw_step["returncode"],
                "argv": raw_step["argv"],
                "cwd": raw_step["cwd"],
                "stdout_sha256": hashlib.sha256(original_stdout.encode("utf-8")).hexdigest(),
                "stderr_sha256": hashlib.sha256(original_stderr.encode("utf-8")).hexdigest(),
                "stdout_tail": _tail(stdout, tail_lines),
                "stderr_tail": _tail(stderr, tail_lines),
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": VALIDATION_RECEIPT_VERSION,
        "task_id": task_id,
        "change_plan_sha256": change_plan["plan_sha256"],
        "engine_validation_plan_sha256": raw["plan_sha256"],
        "status": raw["status"],
        "cpu_only": True,
        "owned_hashes": dict(sorted(owned_hashes.items())),
        "validations": {
            "cpu_targeted": raw["status"],
            "mandatory_external": {
                row["required_validation"]: "outstanding"
                for row in change_plan["mandatory_outstanding_obligations"]
            },
        },
        "mandatory_outstanding_obligations": change_plan[
            "mandatory_outstanding_obligations"
        ],
        "junit_path": str(junit_path) if junit_path and junit_path.is_file() else None,
        "steps": rows,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_task(task_id: str, *, state_root: str | Path | None = None) -> dict[str, Any]:
    resolved_state = resolve_state_root(state_root)
    task_dir, baseline = load_task(resolved_state, task_id)
    root = Path(baseline["project_root"])
    change_plan = build_change_plan(task_id, state_root=resolved_state)
    if change_plan["status"] != "ready":
        raise DevAutomationError(f"change plan is blocked: {change_plan['blockers']}")
    status = task_status(task_id, state_root=resolved_state)
    plan, junit_path = _validation_plan(
        task_id,
        change_plan,
        project_root=root,
        task_dir=task_dir,
    )
    raw = execute_plan(plan, root=root)
    redacted_raw = dict(raw)
    redacted_raw["steps"] = [
        {**row, "stdout": _redact(row["stdout"]), "stderr": _redact(row["stderr"])}
        for row in raw["steps"]
    ]
    raw_path = task_dir / "validation" / f"{change_plan['plan_sha256']}.raw.json"
    compact_path = task_dir / "validation" / f"{change_plan['plan_sha256']}.json"
    atomic_write_json(raw_path, redacted_raw)
    config = load_config(root)
    compact = _compact_receipt(
        task_id,
        change_plan,
        raw,
        owned_hashes=status["owned_hashes"],
        junit_path=junit_path,
        tail_lines=config.limits["log_tail_lines"],
    )
    atomic_write_json(compact_path, compact)
    append_event(
        task_dir,
        "validation_completed",
        {
            "receipt_sha256": compact["receipt_sha256"],
            "receipt_path": str(compact_path),
            "status": compact["status"],
            "owned_hashes": compact["owned_hashes"],
        },
    )
    compact["receipt_path"] = str(compact_path)
    compact["raw_receipt_path"] = str(raw_path)
    return compact


def _latest_validation(task_dir: Path) -> dict[str, Any] | None:
    candidates = sorted((task_dir / "validation").glob("*.json")) if (task_dir / "validation").is_dir() else []
    candidates = [path for path in candidates if not path.name.endswith(".raw.json")]
    if not candidates:
        return None
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    try:
        payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DevAutomationError(f"cannot read validation receipt: {candidates[-1]}") from exc
    claimed = payload.get("receipt_sha256")
    unhashed = dict(payload)
    unhashed.pop("receipt_sha256", None)
    if payload.get("schema_version") != VALIDATION_RECEIPT_VERSION or canonical_sha256(unhashed) != claimed:
        raise DevAutomationError(f"validation receipt integrity check failed: {candidates[-1]}")
    return payload


def triage_task(task_id: str, *, state_root: str | Path | None = None) -> dict[str, Any]:
    resolved_state = resolve_state_root(state_root)
    task_dir, _ = load_task(resolved_state, task_id)
    receipt = _latest_validation(task_dir)
    if receipt is None:
        raise DevAutomationError("task has no validation receipt")
    failures: list[dict[str, Any]] = []
    junit_path = receipt.get("junit_path")
    if junit_path and Path(junit_path).is_file():
        try:
            root = ET.parse(junit_path).getroot()
        except (ET.ParseError, OSError) as exc:
            failures.append(
                {
                    "kind": "junit_parse_error",
                    "path": str(junit_path),
                    "message": _redact(str(exc))[:500],
                }
            )
        else:
            for case in root.iter("testcase"):
                failure = case.find("failure")
                error = case.find("error")
                node = failure if failure is not None else error
                if node is None:
                    continue
                message = _redact(node.get("message", "") or (node.text or "").splitlines()[0])
                failures.append(
                    {
                        "kind": "pytest",
                        "test": f"{case.get('classname', '')}::{case.get('name', '')}".strip(":"),
                        "message": message[:500],
                    }
                )
    for step in receipt["steps"]:
        if step["status"] == "passed":
            continue
        combined = f"{step['stdout_tail']}\n{step['stderr_tail']}"
        locations = sorted(
            {
                f"{match.group('path')}:{match.group('line')}"
                for match in _LOCATION_RE.finditer(combined)
            }
        )[:10]
        failures.append(
            {
                "kind": "command",
                "step": step["id"],
                "status": step["status"],
                "locations": locations,
                "message": _tail(combined, 20),
            }
        )
    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in failures:
        identity = canonical_sha256(row)
        if identity not in seen:
            seen.add(identity)
            deduplicated.append(row)
    payload: dict[str, Any] = {
        "schema_version": TRIAGE_VERSION,
        "task_id": task_id,
        "validation_receipt_sha256": receipt["receipt_sha256"],
        "status": receipt["status"],
        "failure_groups": deduplicated,
        "next_command": (
            f"emergence-dev change review --task {task_id}"
            if receipt["status"] == "passed"
            else f"emergence-dev change validate --task {task_id}"
        ),
    }
    payload["triage_sha256"] = canonical_sha256(payload)
    return payload


def _added_diff(root: Path, owned: Iterable[str]) -> list[str]:
    import subprocess

    completed = subprocess.run(
        ["git", "diff", "--unified=0", "--no-color", "--", *owned],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise DevAutomationError(completed.stderr.strip() or "git diff failed")
    added = [
        line[1:]
        for line in completed.stdout.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    for relative in owned:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=root,
            check=False,
            capture_output=True,
        )
        path = root / relative
        if tracked.returncode != 0 and path.is_file():
            try:
                added.extend(path.read_text(encoding="utf-8").splitlines())
            except (OSError, UnicodeDecodeError):
                continue
    return added


def review_task(task_id: str, *, state_root: str | Path | None = None) -> dict[str, Any]:
    resolved_state = resolve_state_root(state_root)
    task_dir, baseline = load_task(resolved_state, task_id)
    root = Path(baseline["project_root"])
    status = task_status(task_id, state_root=resolved_state)
    plan = build_change_plan(task_id, state_root=resolved_state)
    receipt = _latest_validation(task_dir)
    blockers = list(plan["blockers"])
    warnings: list[str] = []
    if not status["owned_delta"]:
        blockers.append("task has no task-owned delta")
    source_delta = [
        path
        for path in status["owned_delta"]
        if path.endswith(".py") and not path.startswith("tests/")
    ]
    if source_delta and not plan["selected_tests"]:
        blockers.append("changed Python source has no selected test")
    if receipt is None:
        blockers.append("task has not been validated")
    else:
        if receipt["status"] != "passed":
            blockers.append("latest validation did not pass")
        if receipt.get("change_plan_sha256") != plan["plan_sha256"]:
            blockers.append("latest validation is stale for the current change plan")
        if receipt["owned_hashes"] != status["owned_hashes"]:
            blockers.append("latest validation is stale for the current owned file hashes")
    for obligation in plan["mandatory_outstanding_obligations"]:
        blockers.append(
            "mandatory obligation remains outstanding: "
            f"{obligation['id']} requires {obligation['required_validation']}"
        )
    added = _added_diff(root, status["owned"])
    hardcoded = sorted({line.strip() for line in added if re.search(r"/(?:home|Users)/[^/\s]+", line)})
    if hardcoded:
        blockers.append("new fixed Causyl path literal detected")
    secrets = sorted({line.strip() for line in added if _SECRET_RE.search(line)})
    if secrets:
        blockers.append("possible secret material detected in added lines")
    if any("schema_version" in line for line in added):
        warnings.append("schema-version text changed; confirm compatibility and consumers")
    if any(re.match(r"\s*(class|def|async def)\s+[^_]", line) for line in added):
        warnings.append("public Python symbols were added or changed")
    for obligation in plan["critical_obligations"]:
        warnings.append(f"critical obligation {obligation['id']}: {' '.join(obligation['notes'])}")
    payload: dict[str, Any] = {
        "schema_version": REVIEW_VERSION,
        "task_id": task_id,
        "status": "blocked" if blockers else "passed",
        "owned_delta": status["owned_delta"],
        "change_plan_sha256": plan["plan_sha256"],
        "validation_receipt_sha256": receipt["receipt_sha256"] if receipt else None,
        "blockers": blockers,
        "warnings": warnings,
        "hardcoded_path_additions": hardcoded,
        "possible_secret_additions": ["[REDACTED]" for _ in secrets],
        "advisory_findings": True,
    }
    payload["review_sha256"] = canonical_sha256(payload)
    return payload


def _render_closeout(payload: Mapping[str, Any]) -> str:
    validation = payload["validation"]
    return "\n".join(
        [
            f"# Emergence development closeout: {payload['task_id']}",
            "",
            f"- Objective: {payload['objective']}",
            f"- Baseline: `{payload['baseline_head']}`",
            f"- Final HEAD: `{payload['final_head']}`",
            f"- Owned files: {len(payload['owned'])}",
            f"- Modified owned files: {len(payload['owned_delta'])}",
            f"- Validation: {validation['status']} (`{validation['receipt_sha256']}`)",
            f"- Review: {payload['review']['status']} (`{payload['review']['review_sha256']}`)",
            "",
            "## Modified files",
            "",
            *[f"- `{path}`" for path in payload["owned_delta"]],
            "",
            "## Validation steps",
            "",
            *[f"- `{row['id']}`: {row['status']}" for row in validation["steps"]],
            "",
            "This receipt covers only the declared file delta and listed CPU validation.",
            "",
        ]
    )


def finish_task(
    task_id: str,
    *,
    state_root: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved_state = resolve_state_root(state_root)
    task_dir, baseline = load_task(resolved_state, task_id)
    status = task_status(task_id, state_root=resolved_state)
    review = review_task(task_id, state_root=resolved_state)
    if review["status"] != "passed":
        raise DevAutomationError(f"change review is blocked: {review['blockers']}")
    validation = _latest_validation(task_dir)
    if validation is None:
        raise DevAutomationError("task has no validation receipt")
    payload: dict[str, Any] = {
        "schema_version": CLOSEOUT_VERSION,
        "task_id": task_id,
        "objective": baseline["objective"],
        "baseline_head": baseline["head"],
        "final_head": status["current_head"],
        "owned": status["owned"],
        "owned_delta": status["owned_delta"],
        "owned_hashes": status["owned_hashes"],
        "scope_additions": [
            event["data"]
            for event in task_events(task_dir)
            if event["kind"] == "scope_added"
        ],
        "mandatory_outstanding_obligations": validation.get(
            "mandatory_outstanding_obligations", []
        ),
        "validation": validation,
        "review": review,
    }
    payload["closeout_sha256"] = canonical_sha256(payload)
    json_path = task_dir / "closeout.json"
    markdown_path = Path(report_path).resolve() if report_path else task_dir / "closeout.md"
    atomic_write_json(json_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_closeout(payload), encoding="utf-8")
    append_event(
        task_dir,
        "task_finished",
        {
            "closeout_sha256": payload["closeout_sha256"],
            "json_path": str(json_path),
            "markdown_path": str(markdown_path),
        },
    )
    return {
        "ok": True,
        "task_id": task_id,
        "closeout_sha256": payload["closeout_sha256"],
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }
