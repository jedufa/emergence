"""Deterministic bounded settlement of structured Worker Result files.

This module is mechanical execution plumbing. It owns no Work, Session,
Assignment, or Scheduler authority; callers provide the durable runtime-state
mutation and result-submission operations.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tools.actors.common import ActorError, utc_now

RuntimeMutator = Callable[[str | Path | None, Callable[[dict[str, Any]], None]], dict[str, Any]]
ResultSubmitter = Callable[[Path, str | Path | None], tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]

DEFAULT_GRACE_SECONDS = 0.25
DEFAULT_VALIDATION_ATTEMPTS = 2


def _fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "mtime_ns": path.stat().st_mtime_ns,
    }


def _observe(
    *,
    mutate_runtime: RuntimeMutator,
    state_root: str | Path | None,
    assignment_id: str,
    lease_id: str,
    result_path: Path,
) -> dict[str, Any]:
    fingerprint = _fingerprint(result_path)
    holder: dict[str, Any] = {}

    def mutate(runtime: dict[str, Any]) -> None:
        processes = dict(runtime.get("processes") or {})
        row = dict(processes.get(assignment_id) or {})
        previous = dict(row.get("result_settlement") or {})
        same = all(previous.get(key) == fingerprint[key] for key in ("size", "sha256", "mtime_ns"))
        now = utc_now()
        settlement = {
            "status": "RESULT_PENDING",
            "first_seen_at_utc": previous.get("first_seen_at_utc") if same else now,
            "last_seen_at_utc": now,
            **fingerprint,
            "stable_observations": int(previous.get("stable_observations") or 0) + 1 if same else 1,
            "validation_attempts": int(previous.get("validation_attempts") or 0) if same else 0,
            "validation_errors": list(previous.get("validation_errors") or []) if same else [],
        }
        row.update(
            {
                "assignment_id": assignment_id,
                "lease_id": lease_id,
                "result_path": str(result_path),
                "status": "RESULT_PENDING",
                "result_settlement": settlement,
                "updated_at_utc": now,
            }
        )
        processes[assignment_id] = row
        runtime["processes"] = processes
        runtime["last_event"] = {
            "kind": "worker_result_pending",
            "assignment_id": assignment_id,
            "at_utc": now,
            "stable_observations": settlement["stable_observations"],
        }
        holder.update(settlement)

    mutate_runtime(state_root, mutate)
    return holder


def _record_validation_failure(
    *,
    mutate_runtime: RuntimeMutator,
    state_root: str | Path | None,
    assignment_id: str,
    error: ActorError,
) -> dict[str, Any]:
    holder: dict[str, Any] = {}

    def mutate(runtime: dict[str, Any]) -> None:
        processes = dict(runtime.get("processes") or {})
        row = dict(processes.get(assignment_id) or {})
        settlement = dict(row.get("result_settlement") or {})
        settlement["validation_attempts"] = int(settlement.get("validation_attempts") or 0) + 1
        settlement["validation_errors"] = [str(error), *list(error.problems or [])]
        settlement["last_validation_at_utc"] = utc_now()
        row["result_settlement"] = settlement
        row["status"] = "RESULT_PENDING"
        row["updated_at_utc"] = utc_now()
        processes[assignment_id] = row
        runtime["processes"] = processes
        holder.update(settlement)

    mutate_runtime(state_root, mutate)
    return holder


def attempt_result_settlement(
    *,
    mutate_runtime: RuntimeMutator,
    submit_result: ResultSubmitter,
    state_root: str | Path | None,
    assignment_id: str,
    lease_id: str,
    result_path: Path,
    validation_attempts: int = DEFAULT_VALIDATION_ATTEMPTS,
) -> dict[str, Any]:
    """Converge one exited worker result without treating partial bytes as loss."""

    if not result_path.is_file():
        return {"status": "RESULT_MISSING"}
    try:
        observed = _observe(
            mutate_runtime=mutate_runtime,
            state_root=state_root,
            assignment_id=assignment_id,
            lease_id=lease_id,
            result_path=result_path,
        )
    except OSError as exc:
        return {"status": "RESULT_PENDING", "error": str(exc)}
    if int(observed.get("stable_observations") or 0) < 2:
        return {"status": "RESULT_PENDING", "observation": observed}
    try:
        result_record, assignment, lease = submit_result(result_path, state_root)
    except ActorError as exc:
        failed = _record_validation_failure(
            mutate_runtime=mutate_runtime,
            state_root=state_root,
            assignment_id=assignment_id,
            error=exc,
        )
        if int(failed.get("validation_attempts") or 0) < max(1, int(validation_attempts)):
            return {"status": "RESULT_PENDING", "observation": failed, "error": str(exc)}
        return {
            "status": "RESULT_INVALID",
            "error": str(exc),
            "errors": list(failed.get("validation_errors") or []),
            "result_path": str(result_path),
        }
    return {
        "status": "RESULT_SUBMITTED",
        "result_record": result_record,
        "assignment": assignment,
        "lease": lease,
    }


def attempt_result_settlement_with_wait(
    *,
    mutate_runtime: RuntimeMutator,
    submit_result: ResultSubmitter,
    state_root: str | Path | None,
    assignment_id: str,
    lease_id: str,
    result_path: Path,
    grace_seconds: float = DEFAULT_GRACE_SECONDS,
    validation_attempts: int = DEFAULT_VALIDATION_ATTEMPTS,
) -> dict[str, Any]:
    settlement = attempt_result_settlement(
        mutate_runtime=mutate_runtime,
        submit_result=submit_result,
        state_root=state_root,
        assignment_id=assignment_id,
        lease_id=lease_id,
        result_path=result_path,
        validation_attempts=validation_attempts,
    )
    if settlement.get("status") == "RESULT_PENDING":
        time.sleep(max(0.0, float(grace_seconds)))
        settlement = attempt_result_settlement(
            mutate_runtime=mutate_runtime,
            submit_result=submit_result,
            state_root=state_root,
            assignment_id=assignment_id,
            lease_id=lease_id,
            result_path=result_path,
            validation_attempts=validation_attempts,
        )
    return settlement


__all__ = [
    "DEFAULT_GRACE_SECONDS",
    "DEFAULT_VALIDATION_ATTEMPTS",
    "attempt_result_settlement",
    "attempt_result_settlement_with_wait",
]
