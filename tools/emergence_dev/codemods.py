"""Narrow, hash-bound, single-output exception-atomic transformations."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .core import (
    DevAutomationError,
    atomic_write_json,
    canonical_sha256,
    load_config,
    resolve_state_root,
)
from .tasks import (
    append_event,
    load_task,
    normalize_owned_path,
    path_hash,
    planned_text_hash,
    task_status,
)


SPEC_VERSION = "emergence_dev_codemod_spec_v1"
PLAN_VERSION = "emergence_dev_codemod_plan_v1"
RECEIPT_VERSION = "emergence_dev_codemod_receipt_v1"
_SPEC_KEYS = frozenset({"schema_version", "operations"})
_COMMON_OPERATION_KEYS = frozenset({"kind", "path"})


def _object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DevAutomationError(f"{field} must be an object with string keys")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], allowed: set[str] | frozenset[str], *, field: str) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise DevAutomationError(f"{field} contains unsupported fields: {unknown}")


def _load_spec(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DevAutomationError(f"cannot load codemod spec: {path}") from exc
    spec = _object(payload, field="spec")
    _exact_keys(spec, _SPEC_KEYS, field="spec")
    if spec.get("schema_version") != SPEC_VERSION:
        raise DevAutomationError(f"codemod spec schema_version must be {SPEC_VERSION!r}")
    operations = spec.get("operations")
    if not isinstance(operations, list) or not operations:
        raise DevAutomationError("codemod spec operations must be a non-empty array")
    return spec


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise DevAutomationError(f"codemod path is not readable UTF-8 text: {path}") from exc


def _generator_output(
    relationship: Mapping[str, Any],
    *,
    root: Path,
    task_dir: Path,
) -> tuple[str, str, list[str], str]:
    source = relationship.get("source")
    counterparts = relationship.get("counterparts")
    generator = relationship.get("generator")
    if (
        not isinstance(source, str)
        or not isinstance(counterparts, list)
        or len(counterparts) != 1
        or not isinstance(counterparts[0], str)
        or not isinstance(generator, list)
        or not generator
        or not all(isinstance(item, str) and item for item in generator)
    ):
        raise DevAutomationError("generator relationship requires one source, one output, and argv")
    output = counterparts[0]
    with tempfile.TemporaryDirectory(prefix="emergence-dev-generator-", dir=task_dir) as temporary:
        stage = Path(temporary)
        staged_source = stage / source
        staged_output = stage / output
        staged_source.parent.mkdir(parents=True, exist_ok=True)
        staged_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / source, staged_source)
        if (root / output).is_file():
            shutil.copy2(root / output, staged_output)
        substitutions = {
            "{python}": str(Path(sys.executable).resolve()),
            "{source}": str(staged_source),
            "{output}": str(staged_output),
        }
        argv = [substitutions.get(item, item) for item in generator]
        executable = Path(argv[0])
        if not executable.is_absolute():
            located = shutil.which(argv[0])
            if located is None:
                raise DevAutomationError(f"generator executable is unavailable: {argv[0]}")
            argv[0] = str(Path(located).resolve())
        completed = subprocess.run(
            argv,
            cwd=root,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
            env={
                "CUDA_VISIBLE_DEVICES": "",
                "NVIDIA_VISIBLE_DEVICES": "void",
                "ROCR_VISIBLE_DEVICES": "",
                "HIP_VISIBLE_DEVICES": "",
            },
        )
        if completed.returncode != 0:
            tail = "\n".join((completed.stderr or completed.stdout).splitlines()[-20:])
            raise DevAutomationError(f"approved generator failed: {tail}")
        return source, output, argv, _read_text(staged_output)


def plan_codemod(
    task_id: str,
    spec_path: str | Path,
    *,
    state_root: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    state = resolve_state_root(state_root)
    task_dir, baseline = load_task(state, task_id)
    root = Path(baseline["project_root"])
    config = load_config(root)
    status = task_status(task_id, state_root=state)
    if status["new_undeclared_dirty"] or status["preexisting_dirty_drift"] or status["head_drift"]:
        raise DevAutomationError("task drift must be resolved before planning a codemod")
    owned = set(status["owned"])
    spec = _load_spec(Path(spec_path))
    writes: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    relationships = {
        row.get("id"): row for row in config.relationships if isinstance(row.get("id"), str)
    }
    for index, raw_operation in enumerate(spec["operations"]):
        operation = _object(raw_operation, field=f"operations[{index}]")
        kind = operation.get("kind")
        if kind == "replace_exact":
            _exact_keys(
                operation,
                _COMMON_OPERATION_KEYS | {"old", "new", "expected_count"},
                field=f"operations[{index}]",
            )
            path = normalize_owned_path(root, operation.get("path", ""))
            old = operation.get("old")
            new = operation.get("new")
            expected = operation.get("expected_count")
            if (
                not isinstance(old, str)
                or not old
                or not isinstance(new, str)
                or isinstance(expected, bool)
                or not isinstance(expected, int)
                or expected < 1
            ):
                raise DevAutomationError("replace_exact requires old, new, and positive expected_count")
            if path not in owned:
                raise DevAutomationError(f"codemod path is not task-owned: {path}")
            if (root / path).is_symlink():
                raise DevAutomationError(f"codemod refuses symlink path: {path}")
            current = writes.get(path, {}).get("content", _read_text(root / path))
            if current.count(old) != expected:
                raise DevAutomationError(
                    f"replace_exact expected {expected} occurrences in {path}, found {current.count(old)}"
                )
            transformed = current.replace(old, new)
            writes[path] = {
                "preimage_sha256": path_hash(root, path),
                "content": transformed,
                "postimage_sha256": planned_text_hash(root, path, transformed),
            }
            summaries.append({"kind": kind, "path": path, "expected_count": expected})
        elif kind == "insert_after":
            _exact_keys(
                operation,
                _COMMON_OPERATION_KEYS | {"anchor", "text"},
                field=f"operations[{index}]",
            )
            path = normalize_owned_path(root, operation.get("path", ""))
            anchor = operation.get("anchor")
            text = operation.get("text")
            if not isinstance(anchor, str) or not anchor or not isinstance(text, str) or not text:
                raise DevAutomationError("insert_after requires non-empty anchor and text")
            if path not in owned:
                raise DevAutomationError(f"codemod path is not task-owned: {path}")
            if (root / path).is_symlink():
                raise DevAutomationError(f"codemod refuses symlink path: {path}")
            current = writes.get(path, {}).get("content", _read_text(root / path))
            combined = anchor + text
            if combined in current:
                transformed = current
            else:
                if current.count(anchor) != 1:
                    raise DevAutomationError(
                        f"insert_after requires one anchor in {path}, found {current.count(anchor)}"
                    )
                transformed = current.replace(anchor, combined, 1)
            writes[path] = {
                "preimage_sha256": path_hash(root, path),
                "content": transformed,
                "postimage_sha256": planned_text_hash(root, path, transformed),
            }
            summaries.append({"kind": kind, "path": path})
        elif kind == "run_generator":
            _exact_keys(operation, {"kind", "relationship_id"}, field=f"operations[{index}]")
            relationship_id = operation.get("relationship_id")
            if not isinstance(relationship_id, str) or relationship_id not in relationships:
                raise DevAutomationError(f"unknown approved generator relationship: {relationship_id!r}")
            source, path, argv, transformed = _generator_output(
                relationships[relationship_id],
                root=root,
                task_dir=task_dir,
            )
            if source not in owned or path not in owned:
                raise DevAutomationError(
                    f"generator source and output must both be task-owned: {source}, {path}"
                )
            if (root / source).is_symlink() or (root / path).is_symlink():
                raise DevAutomationError("approved generator refuses symlink source or output")
            writes[path] = {
                "preimage_sha256": path_hash(root, path),
                "content": transformed,
                "postimage_sha256": planned_text_hash(root, path, transformed),
            }
            summaries.append(
                {"kind": kind, "path": path, "relationship_id": relationship_id, "argv": argv}
            )
        else:
            raise DevAutomationError(f"unsupported codemod operation kind: {kind!r}")
    plan: dict[str, Any] = {
        "schema_version": PLAN_VERSION,
        "task_id": task_id,
        "baseline_sha256": baseline["baseline_sha256"],
        "spec_sha256": canonical_sha256(spec),
        "application_contract": {
            "maximum_live_outputs": 1,
            "atomicity": "exception_atomic",
            "process_crash_recovery": False,
        },
        "operations": summaries,
        "writes": dict(sorted(writes.items())),
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    plan_path = Path(output).resolve() if output else task_dir / "codemods" / f"{plan['plan_sha256']}.json"
    atomic_write_json(plan_path, plan)
    append_event(
        task_dir,
        "codemod_planned",
        {"plan_sha256": plan["plan_sha256"], "paths": sorted(writes)},
    )
    return {
        "ok": True,
        "task_id": task_id,
        "plan_sha256": plan["plan_sha256"],
        "plan_path": str(plan_path),
        "operations": summaries,
        "writes": [
            {
                "path": path,
                "preimage_sha256": row["preimage_sha256"],
                "postimage_sha256": row["postimage_sha256"],
            }
            for path, row in sorted(writes.items())
        ],
    }


def _load_plan(path: Path, expected_hash: str) -> dict[str, Any]:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DevAutomationError(f"cannot load codemod plan: {path}") from exc
    if not isinstance(plan, dict) or plan.get("schema_version") != PLAN_VERSION:
        raise DevAutomationError(f"codemod plan schema_version must be {PLAN_VERSION!r}")
    claimed = plan.get("plan_sha256")
    unhashed = dict(plan)
    unhashed.pop("plan_sha256", None)
    actual = canonical_sha256(unhashed)
    if claimed != actual or expected_hash != actual:
        raise DevAutomationError("codemod plan hash does not match reviewed hash")
    return plan


def _exception_atomic_transaction(
    root: Path,
    writes: Mapping[str, Mapping[str, Any]],
) -> dict[str, str | None]:
    originals: dict[str, bytes | None] = {}
    modes: dict[str, int] = {}
    staged: dict[str, Path] = {}
    try:
        for relative, row in writes.items():
            target = root / relative
            originals[relative] = target.read_bytes() if target.exists() else None
            modes[relative] = target.stat().st_mode if target.exists() else 0o100644
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.emergence-dev-{os.getpid()}.tmp")
            temporary.write_text(row["content"], encoding="utf-8")
            os.chmod(temporary, modes[relative])
            staged[relative] = temporary
        for relative, temporary in staged.items():
            os.replace(temporary, root / relative)
        actual = {relative: path_hash(root, relative) for relative in writes}
        expected = {
            relative: row["postimage_sha256"] for relative, row in writes.items()
        }
        if actual != expected:
            raise DevAutomationError("codemod postimage verification failed")
        return actual
    except BaseException:
        for relative, original in originals.items():
            target = root / relative
            if original is None:
                target.unlink(missing_ok=True)
            else:
                temporary = target.with_name(f".{target.name}.emergence-dev-rollback.tmp")
                temporary.write_bytes(original)
                os.chmod(temporary, modes[relative])
                os.replace(temporary, target)
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)


def apply_codemod(
    task_id: str,
    plan_path: str | Path,
    plan_hash: str,
    *,
    state_root: str | Path | None = None,
) -> dict[str, Any]:
    state = resolve_state_root(state_root)
    task_dir, baseline = load_task(state, task_id)
    root = Path(baseline["project_root"])
    plan = _load_plan(Path(plan_path), plan_hash)
    if plan.get("task_id") != task_id or plan.get("baseline_sha256") != baseline["baseline_sha256"]:
        raise DevAutomationError("codemod plan belongs to a different task baseline")
    status = task_status(task_id, state_root=state)
    if status["new_undeclared_dirty"] or status["preexisting_dirty_drift"] or status["head_drift"]:
        raise DevAutomationError("task drift must be resolved before applying a codemod")
    if len(plan["writes"]) != 1:
        raise DevAutomationError(
            "V1 codemod apply is exception-atomic and requires exactly one live output; "
            "split multi-file plans into separately reviewed applications"
        )
    current = {path: path_hash(root, path) for path in plan["writes"]}
    preimages = {path: row["preimage_sha256"] for path, row in plan["writes"].items()}
    postimages = {path: row["postimage_sha256"] for path, row in plan["writes"].items()}
    if current == postimages:
        return {
            "ok": True,
            "status": "already_applied",
            "task_id": task_id,
            "plan_sha256": plan_hash,
        }
    if current != preimages:
        raise DevAutomationError("codemod plan is stale for the current file hashes")
    actual = _exception_atomic_transaction(root, plan["writes"])
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_VERSION,
        "task_id": task_id,
        "plan_sha256": plan_hash,
        "status": "applied",
        "application_contract": "single_output_exception_atomic",
        "postimage_hashes": actual,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    receipt_path = task_dir / "codemods" / f"{plan_hash}.receipt.json"
    atomic_write_json(receipt_path, receipt)
    append_event(
        task_dir,
        "codemod_applied",
        {"plan_sha256": plan_hash, "receipt_sha256": receipt["receipt_sha256"]},
    )
    return {**receipt, "receipt_path": str(receipt_path), "ok": True}
