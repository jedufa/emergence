"""Structured worker opening and result protocol for EIEL Assignments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from tools.actors.bootstrap import git_repository_identity
from tools.actors.common import ActorError, atomic_write_json, utc_now
from tools.actors.portfolio import session_store
from tools.scheduler.contracts import (
    WORKER_RESULT_SCHEMA_VERSION,
    assignment_store,
    lease_store,
    load_assignment_payload,
    load_lease_payload,
    normalize_worker_result_payload,
)
from tools.scheduler.dispatch import worker_open


def open_worker(
    assignment_id: str,
    lease_id: str,
    *,
    state_root: str | Path | None,
    worker_id: str,
    codex_id: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    return worker_open(
        assignment_id,
        lease_id,
        state_root=state_root,
        worker_id=worker_id,
        codex_id=codex_id,
        generated_at=generated_at,
    )


def result_from_repository(
    *,
    assignment_id: str,
    lease_id: str,
    state_root: str | Path | None,
    worker_identity: Mapping[str, Any],
    outcome: str,
    validation_status: str = "NOT_RUN",
    validation_commands: list[str] | None = None,
    validation_gaps: list[str] | None = None,
    validation_decision_impact: str | None = None,
    report_paths: list[str] | None = None,
    blockers: list[str] | None = None,
    started_at_utc: str | None = None,
    ended_at_utc: str | None = None,
    exit_code: int = 0,
) -> dict[str, Any]:
    assignment = assignment_store(state_root).get(assignment_id)
    lease = lease_store(state_root).get(lease_id)
    ap = load_assignment_payload(assignment.get("payload"), require_ready=True)
    material = dict(ap.get("material") or {})
    lease_payload = load_lease_payload(lease.get("payload"))
    if lease_payload.get("assignment_id") != assignment_id:
        raise ActorError("Lease does not bind the requested Assignment", code="lease_assignment_mismatch")
    session_id = str(dict(material.get("session") or {}).get("record_id") or "")
    session = session_store(state_root).get(session_id)
    session_payload = dict(session.get("payload") or {})
    repo = dict(material.get("repository") or {})
    live = git_repository_identity(repo.get("repo_root"))
    timestamp = utc_now(ended_at_utc)
    result = {
        "schema_version": WORKER_RESULT_SCHEMA_VERSION,
        "assignment_id": assignment_id,
        "assignment_material_sha256": ap.get("assignment_material_sha256"),
        "context_receipt_sha256": dict(material.get("continuation") or {}).get("context_receipt_sha256"),
        "lease_id": lease_id,
        "work_id": dict(material.get("work") or {}).get("record_id"),
        "session_id": session_id,
        "worker_identity": dict(worker_identity),
        "outcome": outcome,
        "repository": {
            "head_before": repo.get("expected_head"),
            "head_after": live.get("head"),
            "working_directory": repo.get("working_directory") or repo.get("repo_root"),
            "commit_created": str(repo.get("expected_head")) != str(live.get("head")),
            "branch": live.get("branch"),
        },
        "session": {
            "revision_after": session.get("revision"),
            "record_sha256_after": session.get("record_sha256"),
            "status_after": session.get("status"),
            "session_outcome": session_payload.get("session_outcome"),
        },
        "execution": {
            "started_at_utc": utc_now(started_at_utc) if started_at_utc else str(lease.get("created_at_utc") or timestamp),
            "ended_at_utc": timestamp,
            "exit_code": int(exit_code),
        },
        "validation": {
            "status": validation_status,
            "commands": list(validation_commands or []),
            "report_paths": list(report_paths or []),
            "gaps": list(validation_gaps or []),
            "decision_impact": validation_decision_impact,
        },
        "blockers": list(blockers or []),
    }
    return normalize_worker_result_payload(result)


def write_worker_result(path: str | Path, result: Mapping[str, Any]) -> Path:
    normalized = normalize_worker_result_payload(result)
    return atomic_write_json(path, normalized)


__all__ = ["open_worker", "result_from_repository", "write_worker_result"]
