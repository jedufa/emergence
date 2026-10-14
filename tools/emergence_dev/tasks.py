"""Append-only task contracts and scope planning."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .core import (
    DevAutomationError,
    atomic_write_json,
    build_context,
    canonical_sha256,
    file_sha256,
    find_project_root,
    load_config,
    resolve_state_root,
)


TASK_VERSION = "emergence_dev_task_v1"
EVENT_VERSION = "emergence_dev_task_event_v1"
PLAN_VERSION = "emergence_dev_change_plan_v1"
_TASK_RE = re.compile(r"[0-9]{8}T[0-9]{6}Z-[a-z0-9][a-z0-9-]{0,48}-[0-9a-f]{8}\Z")
_REASON_CODES = frozenset(
    {
        "configuration",
        "cross-surface",
        "generated",
        "native-cuda",
        "schema",
        "checkpoint",
        "test",
    }
)


def _git(root: Path, *argv: str) -> str:
    completed = subprocess.run(
        ["git", *argv],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise DevAutomationError(completed.stderr.strip() or f"git {' '.join(argv)} failed")
    return completed.stdout.strip()


def git_status(root: Path) -> dict[str, str]:
    project_prefix = _git(root, "rev-parse", "--show-prefix")

    def project_relative(path: str) -> str:
        if not project_prefix:
            return path
        if not path.startswith(project_prefix):
            raise DevAutomationError(f"git status path escapes project root: {path}")
        relative = path[len(project_prefix) :]
        if not relative:
            raise DevAutomationError(
                f"git status path does not name a project file: {path}"
            )
        return relative

    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", "."],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise DevAutomationError(completed.stderr.decode("utf-8", "replace").strip())
    parts = completed.stdout.split(b"\0")
    rows: dict[str, str] = {}
    index = 0
    while index < len(parts):
        raw = parts[index]
        index += 1
        if not raw:
            continue
        text = raw.decode("utf-8", "surrogateescape")
        if len(text) < 4:
            continue
        code = text[:2]
        path = project_relative(text[3:])
        rows[path] = code
        if "R" in code or "C" in code:
            if index < len(parts) and parts[index]:
                original = project_relative(
                    parts[index].decode("utf-8", "surrogateescape")
                )
                rows[original] = code
                index += 1
    return dict(sorted(rows.items()))


def normalize_owned_path(root: Path, value: str | Path) -> str:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    resolved = candidate.parent.resolve(strict=False) / candidate.name
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise DevAutomationError(f"task-owned path escapes project root: {value}") from exc
    if resolved.is_dir() and not resolved.is_symlink():
        raise DevAutomationError(f"task-owned path must name a file, not a directory: {value}")
    rendered = relative.as_posix()
    if not rendered or rendered == ".":
        raise DevAutomationError("task-owned path must not be the project root")
    return rendered


def path_hash(root: Path, relative: str) -> str | None:
    path = root / relative
    if not os.path.lexists(path):
        return None
    metadata = path.lstat()
    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode):
        return canonical_sha256(
            {"kind": "symlink", "mode": mode, "target": os.readlink(path)}
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise DevAutomationError(f"task path is not a regular file: {relative}")
    return canonical_sha256(
        {"kind": "file", "mode": mode, "content_sha256": file_sha256(path)}
    )


def planned_text_hash(root: Path, relative: str, content: str) -> str:
    path = root / relative
    mode = stat.S_IMODE(path.lstat().st_mode) if os.path.lexists(path) else 0o644
    return canonical_sha256(
        {
            "kind": "file",
            "mode": mode,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
    )


def snapshot_paths(root: Path, paths: Iterable[str]) -> dict[str, str | None]:
    return {path: path_hash(root, path) for path in sorted(set(paths))}


def _slug(value: str) -> str:
    words = re.findall(r"[a-z0-9]+", value.lower())
    return "-".join(words[:6])[:48] or "task"


def _task_dir(state_root: Path, task_id: str) -> Path:
    if _TASK_RE.fullmatch(task_id) is None:
        raise DevAutomationError(f"invalid task id: {task_id}")
    path = state_root / "tasks" / task_id
    if not path.is_dir():
        raise DevAutomationError(f"unknown task: {task_id}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DevAutomationError(f"cannot load task state: {path}") from exc
    if not isinstance(payload, dict):
        raise DevAutomationError(f"task state is not an object: {path}")
    return payload


def load_task(state_root: Path, task_id: str) -> tuple[Path, dict[str, Any]]:
    directory = _task_dir(state_root, task_id)
    baseline = _load_json(directory / "baseline.json")
    if baseline.get("schema_version") != TASK_VERSION:
        raise DevAutomationError(f"unsupported task schema: {baseline.get('schema_version')!r}")
    return directory, baseline


def _read_events(directory: Path) -> list[dict[str, Any]]:
    path = directory / "events.jsonl"
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DevAutomationError(f"cannot read task events: {path}") from exc
    previous = ""
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DevAutomationError(f"task event log is invalid: {path}") from exc
        claimed = row.get("event_sha256")
        unhashed = dict(row)
        unhashed.pop("event_sha256", None)
        if row.get("previous_event_sha256", "") != previous or canonical_sha256(unhashed) != claimed:
            raise DevAutomationError(f"task event chain is invalid: {path}")
        previous = claimed
        rows.append(row)
    head_path = directory / "events_head.json"
    head = _load_json(head_path)
    count = head.get("event_count")
    last = head.get("last_event_sha256")
    if (
        head.get("schema_version") != "emergence_dev_event_head_v1"
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or not isinstance(last, str)
    ):
        raise DevAutomationError(f"task event head is invalid: {head_path}")
    if count == len(rows) and last == previous:
        return rows
    if count > len(rows):
        raise DevAutomationError(
            f"task event head is ahead of append-only log: {head_path}"
        )
    expected_prefix = "" if count == 0 else rows[count - 1]["event_sha256"]
    if last != expected_prefix:
        raise DevAutomationError(
            f"task event head is not a valid prefix of append-only log: {head_path}"
        )
    recovered_head = {
        "schema_version": "emergence_dev_event_head_v1",
        "event_count": len(rows),
        "last_event_sha256": previous,
    }
    recovery: dict[str, Any] = {
        "schema_version": "emergence_dev_event_head_recovery_v1",
        "task_id": directory.name,
        "status": "recovered_valid_log_extension",
        "prior_head": head,
        "recovered_head": recovered_head,
        "recovered_event_sha256s": [
            row["event_sha256"] for row in rows[count:]
        ],
    }
    recovery["recovery_receipt_sha256"] = canonical_sha256(recovery)
    receipt_path = (
        directory
        / "recovery_receipts"
        / (
            f"events-{count}-{len(rows)}-"
            f"{recovery['recovery_receipt_sha256'][:16]}.json"
        )
    )
    atomic_write_json(receipt_path, recovery)
    atomic_write_json(head_path, recovered_head)
    return rows


def task_events(directory: Path) -> list[dict[str, Any]]:
    """Return the verified append-only event stream."""
    return _read_events(directory)


def event_recovery_receipts(directory: Path) -> list[dict[str, Any]]:
    """Return verified deterministic event-head recovery receipts."""
    recovery_dir = directory / "recovery_receipts"
    if not recovery_dir.is_dir():
        return []
    receipts: list[dict[str, Any]] = []
    for path in sorted(recovery_dir.glob("events-*.json")):
        receipt = _load_json(path)
        claimed = receipt.get("recovery_receipt_sha256")
        unhashed = dict(receipt)
        unhashed.pop("recovery_receipt_sha256", None)
        if (
            receipt.get("schema_version")
            != "emergence_dev_event_head_recovery_v1"
            or canonical_sha256(unhashed) != claimed
        ):
            raise DevAutomationError(
                f"task event recovery receipt is invalid: {path}"
            )
        receipts.append(receipt)
    return receipts


def append_event(directory: Path, kind: str, data: Mapping[str, Any]) -> dict[str, Any]:
    events = _read_events(directory)
    row: dict[str, Any] = {
        "schema_version": EVENT_VERSION,
        "sequence": len(events) + 1,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "kind": kind,
        "data": dict(data),
        "previous_event_sha256": events[-1]["event_sha256"] if events else "",
    }
    row["event_sha256"] = canonical_sha256(row)
    line = canonical_json_line(row)
    descriptor = os.open(directory / "events.jsonl", os.O_WRONLY | os.O_APPEND)
    try:
        os.write(descriptor, line.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    atomic_write_json(
        directory / "events_head.json",
        {
            "schema_version": "emergence_dev_event_head_v1",
            "event_count": len(events) + 1,
            "last_event_sha256": row["event_sha256"],
        },
    )
    return row


def canonical_json_line(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def _adoptions(values: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        path, separator, digest = value.rpartition("=")
        if not separator or not path or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise DevAutomationError("--adopt values must use PATH=SHA256")
        result[path] = digest
    return result


def begin_task(
    objective: str,
    owned: Iterable[str],
    *,
    project_root: str | Path | None = None,
    state_root: str | Path | None = None,
    adoptions: Iterable[str] = (),
) -> dict[str, Any]:
    root = find_project_root(project_root)
    config = load_config(root)
    resolved_state = resolve_state_root(state_root)
    objective = objective.strip()
    if not objective:
        raise DevAutomationError("objective must not be empty")
    owned_paths = sorted({normalize_owned_path(root, value) for value in owned})
    if not owned_paths:
        raise DevAutomationError("at least one --own path is required")
    default_limit = config.limits["default_owned_files"]
    if len(owned_paths) > default_limit:
        raise DevAutomationError(
            f"change begin accepts at most {default_limit} paths; add justified paths with change scope-add"
        )
    dirty = git_status(root)
    normalized_adoptions = {
        normalize_owned_path(root, path): digest for path, digest in _adoptions(adoptions).items()
    }
    for path in owned_paths:
        if path not in dirty:
            continue
        digest = path_hash(root, path)
        if digest is None:
            raise DevAutomationError(
                f"already-dirty untracked path requires a file before adoption: {path}"
            )
        if normalized_adoptions.get(path) != digest:
            raise DevAutomationError(
                f"already-dirty owned path requires --adopt {path}={digest}"
            )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    identity = canonical_sha256(
        {
            "objective": objective,
            "head": _git(root, "rev-parse", "HEAD"),
            "owned": owned_paths,
            "timestamp": timestamp,
        }
    )[:8]
    task_id = f"{timestamp}-{_slug(objective)}-{identity}"
    directory = resolved_state / "tasks" / task_id
    directory.mkdir(parents=True, exist_ok=False)
    baseline: dict[str, Any] = {
        "schema_version": TASK_VERSION,
        "task_id": task_id,
        "objective": objective,
        "project_root": str(root),
        "config_sha256": config.config_sha256,
        "head": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "branch", "--show-current"),
        "preexisting_dirty": dirty,
        "preexisting_dirty_hashes": snapshot_paths(root, dirty),
        "owned": owned_paths,
        "owned_reasons": {path: "initial" for path in owned_paths},
        "owned_baseline_hashes": snapshot_paths(root, owned_paths),
        "adopted": sorted(normalized_adoptions),
        "default_file_budget": default_limit,
        "extended_file_budget": config.limits["extended_owned_files"],
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    baseline["baseline_sha256"] = canonical_sha256(baseline)
    atomic_write_json(directory / "baseline.json", baseline)
    (directory / "events.jsonl").write_text("", encoding="utf-8")
    atomic_write_json(
        directory / "events_head.json",
        {
            "schema_version": "emergence_dev_event_head_v1",
            "event_count": 0,
            "last_event_sha256": "",
        },
    )
    append_event(directory, "task_started", {"owned": owned_paths})
    return {
        "ok": True,
        "task_id": task_id,
        "task_dir": str(directory),
        "baseline_sha256": baseline["baseline_sha256"],
    }


def _owned_projection(baseline: Mapping[str, Any], events: Iterable[Mapping[str, Any]]) -> tuple[list[str], dict[str, str]]:
    owned = list(baseline["owned"])
    reasons = dict(baseline["owned_reasons"])
    for event in events:
        if event["kind"] == "scope_added":
            path = event["data"]["path"]
            if path not in owned:
                owned.append(path)
            reasons[path] = event["data"]["reason"]
    return sorted(owned), reasons


def task_status(task_id: str, *, state_root: str | Path | None = None) -> dict[str, Any]:
    resolved_state = resolve_state_root(state_root)
    directory, baseline = load_task(resolved_state, task_id)
    root = Path(baseline["project_root"])
    events = _read_events(directory)
    owned, reasons = _owned_projection(baseline, events)
    current_hashes = snapshot_paths(root, owned)
    baseline_hashes = dict(baseline["owned_baseline_hashes"])
    for event in events:
        if event["kind"] == "scope_added":
            baseline_hashes[event["data"]["path"]] = event["data"]["baseline_hash"]
    for path in owned:
        baseline_hashes.setdefault(path, None)
    delta = sorted(path for path in owned if current_hashes[path] != baseline_hashes[path])
    current_dirty = git_status(root)
    preexisting_dirty = dict(baseline["preexisting_dirty"])
    new_undeclared = sorted(path for path in current_dirty if path not in preexisting_dirty and path not in owned)
    dirty_drift: list[str] = []
    for path in preexisting_dirty:
        if path in owned:
            continue
        old_hash = baseline["preexisting_dirty_hashes"].get(path)
        try:
            current_hash = path_hash(root, path)
        except DevAutomationError:
            current_hash = None
        if current_hash != old_hash:
            dirty_drift.append(path)
    head = _git(root, "rev-parse", "HEAD")
    return {
        "schema_version": TASK_VERSION,
        "task_id": task_id,
        "objective": baseline["objective"],
        "baseline_head": baseline["head"],
        "current_head": head,
        "head_drift": head != baseline["head"],
        "owned": owned,
        "owned_reasons": reasons,
        "owned_delta": delta,
        "owned_hashes": current_hashes,
        "new_undeclared_dirty": new_undeclared,
        "preexisting_dirty_drift": sorted(dirty_drift),
        "event_count": len(events),
        "event_recoveries": event_recovery_receipts(directory),
        "latest_event": events[-1] if events else None,
        "task_dir": str(directory),
    }


def scope_add(
    task_id: str,
    path: str,
    reason: str,
    *,
    justification: str,
    state_root: str | Path | None = None,
    approve_scope: str | None = None,
    adopt_sha256: str | None = None,
) -> dict[str, Any]:
    resolved_state = resolve_state_root(state_root)
    directory, baseline = load_task(resolved_state, task_id)
    root = Path(baseline["project_root"])
    normalized = normalize_owned_path(root, path)
    if reason not in _REASON_CODES:
        raise DevAutomationError(f"scope reason must be one of {sorted(_REASON_CODES)}")
    justification = justification.strip()
    if not justification:
        raise DevAutomationError("scope justification must not be empty")
    events = _read_events(directory)
    owned, owned_reasons = _owned_projection(baseline, events)
    if normalized in owned:
        raise DevAutomationError(f"path is already task-owned: {normalized}")
    current_dirty = git_status(root)
    current_hash = path_hash(root, normalized)
    if normalized in current_dirty and adopt_sha256 != current_hash:
        if current_hash is None:
            raise DevAutomationError(
                f"dirty path must exist before it can be adopted: {normalized}"
            )
        raise DevAutomationError(
            f"dirty path requires --adopt-sha256 {current_hash}: {normalized}"
        )
    proposed = sorted([*owned, normalized])
    default_limit = int(baseline["default_file_budget"])
    extended_limit = int(baseline["extended_file_budget"])
    if len(proposed) > extended_limit:
        raise DevAutomationError(f"scope exceeds the hard limit of {extended_limit} files")
    config = load_config(root)
    proposed_set = set(proposed)
    relationship_snapshot = []
    for relationship in config.relationships:
        source = relationship.get("source")
        counterparts = [
            item for item in relationship.get("counterparts", []) if isinstance(item, str)
        ]
        if source in proposed_set or any(item in proposed_set for item in counterparts):
            relationship_snapshot.append(
                {
                    "id": relationship.get("id", "unnamed"),
                    "missing_from_scope": sorted(
                        item
                        for item in [source, *counterparts]
                        if isinstance(item, str) and item not in proposed_set
                    ),
                }
            )
    critical_snapshot = []
    for rule in config.critical_rules:
        patterns = [item for item in rule.get("paths", []) if isinstance(item, str)]
        matching = sorted(
            item
            for item in proposed
            if any(fnmatch.fnmatch(item, pattern) for pattern in patterns)
        )
        if matching:
            critical_snapshot.append(
                {
                    "id": rule.get("id", "unnamed"),
                    "matching_paths": matching,
                    "mandatory": bool(rule.get("mandatory", False)),
                    "required_validation": rule.get("required_validation"),
                }
            )
    approval_manifest = {
        "schema_version": "emergence_dev_scope_manifest_v1",
        "task_id": task_id,
        "owned": proposed,
        "owned_reasons": {**owned_reasons, normalized: reason},
        "new_path": normalized,
        "reason": reason,
        "justification": justification,
        "generated_relationships": relationship_snapshot,
        "critical_obligations": critical_snapshot,
    }
    approval_hash = canonical_sha256(approval_manifest)
    if len(proposed) > default_limit and approve_scope != approval_hash:
        raise DevAutomationError(
            f"scope exceeds the default {default_limit}-file budget; "
            f"review the obligation snapshot and repeat with --approve-scope {approval_hash}"
        )
    append_event(
        directory,
        "scope_added",
        {
            "path": normalized,
            "reason": reason,
            "justification": justification,
            "baseline_hash": current_hash,
            "adopted": normalized in current_dirty,
            "scope_manifest": approval_manifest,
            "scope_manifest_sha256": approval_hash,
            "scope_approval_sha256": approval_hash if len(proposed) > default_limit else None,
        },
    )
    return {
        "ok": True,
        "task_id": task_id,
        "owned_count": len(proposed),
        "path": normalized,
        "reason": reason,
        "justification": justification,
        "scope_manifest_sha256": approval_hash,
    }


def _plan_relationships(config: Any, owned: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for relationship in config.relationships:
        source = relationship.get("source")
        counterparts = [item for item in relationship.get("counterparts", []) if isinstance(item, str)]
        related = source in owned or any(item in owned for item in counterparts)
        if not related:
            continue
        missing = [item for item in [source, *counterparts] if isinstance(item, str) and item not in owned]
        rows.append({**dict(relationship), "missing_from_scope": missing})
        if missing:
            blockers.append(
                f"generated relationship {relationship.get('id', 'unnamed')} is missing owned paths: {missing}"
            )
    return rows, blockers


def _imports_owned_module(imports: Iterable[str], owned: set[str]) -> bool:
    modules: set[str] = set()
    for path in owned:
        if not path.endswith(".py") or path.startswith("tests/"):
            continue
        module = path.removesuffix(".py").replace("/", ".")
        modules.add(module)
        if module.startswith("engine."):
            modules.add(module.removeprefix("engine."))
    return any(
        imported == module
        or imported.startswith(f"{module}.")
        or module.startswith(f"{imported}.")
        for imported in imports
        for module in modules
    )


def build_change_plan(task_id: str, *, state_root: str | Path | None = None) -> dict[str, Any]:
    resolved_state = resolve_state_root(state_root)
    directory, baseline = load_task(resolved_state, task_id)
    status = task_status(task_id, state_root=resolved_state)
    root = Path(baseline["project_root"])
    config = load_config(root)
    context = build_context(
        baseline["objective"],
        project_root=root,
        state_root=resolved_state,
    )
    owned = set(status["owned"])
    relationships, blockers = _plan_relationships(config, owned)
    if status["new_undeclared_dirty"]:
        blockers.append(f"new undeclared dirty paths: {status['new_undeclared_dirty']}")
    if status["preexisting_dirty_drift"]:
        blockers.append(f"pre-existing dirty paths drifted: {status['preexisting_dirty_drift']}")
    if status["head_drift"]:
        blockers.append("HEAD changed since task begin")
    selected_tests = {
        row["path"]: row["reason"]
        for row in context["suggested_tests"]
        if row["reason"] != "lexical_advisory"
        and _imports_owned_module(row.get("imports", ()), owned)
    }
    external_receipts: dict[str, Mapping[str, Any]] = {}
    for event in reversed(task_events(directory)):
        if event["kind"] != "external_validation_recorded":
            continue
        receipt = event["data"].get("receipt")
        if not isinstance(receipt, Mapping):
            continue
        obligation_id = receipt.get("obligation_id")
        if not isinstance(obligation_id, str) or obligation_id in external_receipts:
            continue
        unsigned = dict(receipt)
        supplied_receipt_sha256 = unsigned.pop("receipt_sha256", None)
        junit_summary = receipt.get("junit_summary")
        receipt_path_value = event["data"].get("receipt_path")
        junit_path_value = receipt.get("junit_path")
        try:
            receipt_path = Path(str(receipt_path_value)).resolve()
            receipt_path.relative_to(directory.resolve())
            junit_path = Path(str(junit_path_value)).resolve()
            junit_path.relative_to(directory.resolve())
            persisted_receipt = json.loads(
                receipt_path.read_text(encoding="utf-8")
            )
            evidence_files_match = (
                not receipt_path.is_symlink()
                and not junit_path.is_symlink()
                and persisted_receipt == receipt
                and file_sha256(junit_path) == receipt.get("junit_sha256")
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
        ):
            evidence_files_match = False
        if (
            receipt.get("schema_version")
            != "emergence_dev_external_validation_receipt_v1"
            or receipt.get("status") != "passed"
            or supplied_receipt_sha256 != canonical_sha256(unsigned)
            or receipt.get("baseline_sha256")
            != baseline["baseline_sha256"]
            or receipt.get("head") != status["current_head"]
            or receipt.get("owned_hashes") != status["owned_hashes"]
            or receipt.get("config_sha256") != config.config_sha256
            or receipt.get("evidence_kind") != "pytest"
            or not isinstance(receipt.get("junit_sha256"), str)
            or len(receipt["junit_sha256"]) != 64
            or not isinstance(junit_summary, Mapping)
            or junit_summary.get("passed", 0) <= 0
            or junit_summary.get("failures") != 0
            or junit_summary.get("errors") != 0
            or junit_summary.get("skipped") != 0
            or not evidence_files_match
        ):
            continue
        external_receipts[obligation_id] = receipt
    for path in owned:
        if path.startswith("tests/") and Path(path).name.startswith("test_"):
            selected_tests[path] = "owned_test"
    critical = []
    for rule in config.critical_rules:
        patterns = [item for item in rule.get("paths", []) if isinstance(item, str)]
        matching = sorted(path for path in owned if any(fnmatch.fnmatch(path, pattern) for pattern in patterns))
        if matching:
            external_receipt = external_receipts.get(
                str(rule.get("id", "unnamed"))
            )
            required_validation = rule.get("required_validation")
            externally_verified = (
                bool(rule.get("mandatory", False))
                and external_receipt is not None
                and external_receipt.get("required_validation")
                == required_validation
            )
            critical.append(
                {
                    "id": rule.get("id", "unnamed"),
                    "matching_paths": matching,
                    "notes": list(rule.get("notes", [])),
                    "mandatory": bool(rule.get("mandatory", False)),
                    "required_validation": required_validation,
                    "status": (
                        "verified"
                        if externally_verified
                        else (
                            "outstanding"
                            if rule.get("mandatory", False)
                            else "advisory"
                        )
                    ),
                    "external_validation_receipt_sha256": (
                        external_receipt.get("receipt_sha256")
                        if externally_verified
                        else None
                    ),
                }
            )
    mandatory_outstanding = [
        row for row in critical if row["mandatory"] and row["status"] == "outstanding"
    ]
    plan: dict[str, Any] = {
        "schema_version": PLAN_VERSION,
        "task_id": task_id,
        "objective": baseline["objective"],
        "baseline_sha256": baseline["baseline_sha256"],
        "config_sha256": config.config_sha256,
        "context_sha256": context["context_sha256"],
        "owned": status["owned"],
        "owned_delta": status["owned_delta"],
        "file_budget": {
            "default": baseline["default_file_budget"],
            "extended": baseline["extended_file_budget"],
            "current": len(status["owned"]),
        },
        "relationships": relationships,
        "critical_obligations": critical,
        "mandatory_outstanding_obligations": mandatory_outstanding,
        "selected_tests": [
            {"path": path, "reason": reason}
            for path, reason in sorted(selected_tests.items())
        ],
        "blockers": blockers,
        "status": "blocked" if blockers else "ready",
        "advisory_selection": True,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    plan_path = directory / "plans" / f"{plan['plan_sha256']}.json"
    atomic_write_json(plan_path, plan)
    append_event(
        directory,
        "change_planned",
        {"plan_sha256": plan["plan_sha256"], "status": plan["status"]},
    )
    plan["plan_path"] = str(plan_path)
    return plan


def checkpoint_task(
    task_id: str,
    *,
    note: str,
    next_command: str,
    state_root: str | Path | None = None,
) -> dict[str, Any]:
    if not note.strip() or not next_command.strip():
        raise DevAutomationError("checkpoint note and next command must not be empty")
    resolved_state = resolve_state_root(state_root)
    directory, _ = load_task(resolved_state, task_id)
    status = task_status(task_id, state_root=resolved_state)
    event = append_event(
        directory,
        "checkpoint",
        {
            "note": note.strip(),
            "next_command": next_command.strip(),
            "owned_hashes": status["owned_hashes"],
            "current_head": status["current_head"],
        },
    )
    return {"ok": True, "task_id": task_id, "event_sha256": event["event_sha256"]}


def resume_task(task_id: str, *, state_root: str | Path | None = None) -> dict[str, Any]:
    resolved_state = resolve_state_root(state_root)
    directory, _ = load_task(resolved_state, task_id)
    events = _read_events(directory)
    status = task_status(task_id, state_root=resolved_state)
    checkpoint = next((row for row in reversed(events) if row["kind"] == "checkpoint"), None)
    next_command = (
        checkpoint["data"]["next_command"]
        if checkpoint
        else f"emergence-dev change plan --task {task_id}"
    )
    drift = bool(
        status["head_drift"]
        or status["new_undeclared_dirty"]
        or status["preexisting_dirty_drift"]
    )
    return {
        "ok": not drift,
        "task_id": task_id,
        "drift": drift,
        "status": status,
        "checkpoint": checkpoint,
        "next_command": next_command,
    }
