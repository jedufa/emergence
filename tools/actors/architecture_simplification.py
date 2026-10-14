"""Canonical simplified projections for retained v1 actor records.

The v1 storage vocabulary remains readable for historical compatibility.  New
operator projections use a smaller semantic model so redundant stored statuses
and copied sidecars cannot acquire independent lifecycle authority.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from tools.actors.contract_versions import (
    SCHEDULER_ASSIGNMENT_PROJECTION_VERSION,
    SESSION_ATTEMPT_PROJECTION_VERSION,
    WORK_LIFECYCLE_PROJECTION_VERSION,
)

WORK_CANONICAL_STATES = ("OPEN", "BLOCKED", "COMPLETED", "CLOSED")
SESSION_CANONICAL_STATES = ("ACTIVE", "PAUSED", "BLOCKED", "COMPLETED", "CLOSED")

_WORK_CLOSED = {"DISCONTINUED", "OBSOLETE", "SUPERSEDED"}
_SESSION_CLOSED = {"DISCONTINUED", "OBSOLETE", "SUPERSEDED"}
_SESSION_RESUMABLE = {"CREATED", "PAUSED", "INTERRUPTED", "BLOCKED"}
_ASSIGNMENT_TRANSIENT = {"PREPARING", "CLEANUP_REQUIRED"}


def canonical_work_lifecycle(
    record: Mapping[str, Any],
    *,
    sessions: Sequence[Mapping[str, Any]] | None = None,
    derived_status: str | None = None,
) -> dict[str, Any]:
    """Project one Work record onto the four-state architecture contract.

    Stored v1 states remain exact historical evidence.  OPEN readiness and
    execution activity are derived, not competing lifecycle states.
    """

    stored = str(record.get("status") or "BACKLOG").upper()
    payload = dict(record.get("payload") or {})
    session_rows = [dict(row) for row in (sessions or []) if isinstance(row, Mapping)]
    active = [row for row in session_rows if str(row.get("status") or "").upper() == "ACTIVE"]
    resumable = [
        row for row in session_rows
        if str(row.get("status") or "").upper() in _SESSION_RESUMABLE
    ]

    if stored == "COMPLETED":
        state = "COMPLETED"
    elif stored == "BLOCKED":
        state = "BLOCKED"
    elif stored in _WORK_CLOSED:
        state = "CLOSED"
    else:
        state = "OPEN"

    if active:
        execution_state = "ACTIVE"
    elif resumable:
        execution_state = "PAUSED"
    else:
        execution_state = "IDLE"

    derived = str(derived_status or "").upper() or None
    if state != "OPEN":
        readiness = "NOT_APPLICABLE"
    elif derived in {
        "BLOCKED_BY_DEPENDENCY",
        "BLOCKED_BY_TERMINAL_DEPENDENCY",
        "CHANGES_REQUIRED",
        "PROMOTION_REQUIRED",
    }:
        readiness = "BLOCKED"
    elif derived in {"READY", "REVIEW_REQUIRED", "READY_TO_ACCEPT"}:
        readiness = "READY"
    else:
        readiness = "OPEN"

    closure_reason = None
    replacement_work_id = None
    if state == "CLOSED":
        closure_reason = stored.lower()
        replacement_work_id = (
            payload.get("superseded_by_work_id") if stored == "SUPERSEDED" else None
        )

    planning_fields_authority = "PRE_SESSION_ONLY" if not session_rows else "SESSION_DERIVED"
    return {
        "schema_version": WORK_LIFECYCLE_PROJECTION_VERSION,
        "state": state,
        "stored_v1_status": stored,
        "stored_v1_status_authoritative_for_history": True,
        "readiness": readiness,
        "execution_state": execution_state,
        "active_session_ids": [str(row.get("session_id") or "") for row in active],
        "resumable_session_ids": [str(row.get("session_id") or "") for row in resumable],
        "closure_reason": closure_reason,
        "replacement_work_id": replacement_work_id,
        "operator_hold_reason": payload.get("blocked_reason") if stored == "PAUSED" else None,
        "planning_fields_authority": planning_fields_authority,
        "parent_work_id_semantics": "DESCRIPTIVE_ONLY",
        "broad_references_semantics": "NAVIGATION_ONLY",
        "resource_totals_semantics": "DERIVED_FROM_IMMUTABLE_WRA_ARTIFACTS",
    }


def canonical_session_attempt(
    record: Mapping[str, Any],
    *,
    work_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project one Session onto the five-state attempt contract."""

    stored = str(record.get("status") or "CREATED").upper()
    payload = dict(record.get("payload") or {})
    work_payload = dict(work_record.get("payload") or {}) if isinstance(work_record, Mapping) else {}

    if stored == "ACTIVE":
        state = "ACTIVE"
        pause_kind = None
    elif stored in {"CREATED", "PAUSED", "INTERRUPTED"}:
        state = "PAUSED"
        pause_kind = {
            "CREATED": "NOT_STARTED",
            "PAUSED": "OPERATOR_PAUSE",
            "INTERRUPTED": "INTERRUPTION",
        }[stored]
    elif stored == "BLOCKED":
        state = "BLOCKED"
        pause_kind = None
    elif stored == "COMPLETED":
        state = "COMPLETED"
        pause_kind = None
    elif stored in _SESSION_CLOSED:
        state = "CLOSED"
        pause_kind = None
    else:
        state = "CLOSED"
        pause_kind = None

    verdict = str(payload.get("review_verdict") or "").upper() or None
    canonical_verdict = None if verdict == "BLOCKED" else verdict
    participants = [
        dict(row) for row in (payload.get("codex_participants") or [])
        if isinstance(row, Mapping)
    ]
    current_codex_id = str(payload.get("current_codex_id") or "") or None
    original_codex_id = (
        str(participants[0].get("codex_id") or "") or None
        if participants else str(payload.get("original_codex_id") or "") or None
    )

    scheduler_origin = {
        "assignment_id": payload.get("scheduler_assignment_id") or payload.get("prepared_assignment_id"),
        "assignment_material_sha256": payload.get("scheduler_assignment_digest"),
        "lease_id": payload.get("scheduler_lease_id"),
        "worker_id": payload.get("scheduler_worker_id"),
        "semantics": "IMMUTABLE_ORIGIN_REFERENCES_ONLY",
    }
    resource_accounting = dict(payload.get("resource_accounting") or {})
    wra_refs = {
        "active_capture_id": resource_accounting.get("active_capture_id"),
        "latest_usage_artifact": resource_accounting.get("latest_usage_artifact"),
        "semantics": "ARTIFACT_REFERENCES_ONLY_IN_CANONICAL_PROJECTION",
    }

    return {
        "schema_version": SESSION_ATTEMPT_PROJECTION_VERSION,
        "state": state,
        "stored_v1_status": stored,
        "stored_v1_status_authoritative_for_history": True,
        "pause_kind": pause_kind,
        "closure_reason": stored.lower() if state == "CLOSED" else None,
        "review_verdict": canonical_verdict,
        "legacy_blocked_review_verdict": verdict == "BLOCKED",
        "mission_id": work_payload.get("mission_id", payload.get("mission_id")),
        "title": work_payload.get("title", payload.get("title")),
        "mission_and_title_source": "WORK_RECORD" if work_payload else "LEGACY_SESSION_COPY",
        "current_codex_id": current_codex_id,
        "original_codex_id": original_codex_id,
        "participant_count": len(participants),
        "preferred_continuation_identity_semantics": "DERIVED_FROM_PARTICIPANTS_AND_POLICY",
        "bootstrap_semantics": "TRANSITION_VALIDATION_SNAPSHOT",
        "progress_semantics": "CURRENT_STEP_PLUS_TYPED_CHECKPOINT_EVIDENCE",
        "scheduler_origin": scheduler_origin,
        "wra_refs": wra_refs,
        "context_semantics": "BOUNDED_CONTINUATION_PROJECTION",
    }


def canonical_assignment_state(status: str) -> dict[str, Any]:
    """Expose Scheduler transients without granting them product lifecycle meaning."""

    stored = str(status or "").upper()
    if stored in _ASSIGNMENT_TRANSIENT:
        state = "COMPILATION_TRANSIENT"
    elif stored in {"READY", "CLAIMED", "RUNNING", "RESULT_SUBMITTED"}:
        state = stored
    elif stored == "CLOSED":
        state = "CLOSED"
    else:
        state = "CLOSED"
    return {
        "schema_version": SCHEDULER_ASSIGNMENT_PROJECTION_VERSION,
        "state": state,
        "stored_v1_status": stored,
        "precreation_protocol": "LEGACY_COMPATIBILITY_LIMITATION" if stored in _ASSIGNMENT_TRANSIENT else None,
    }


__all__ = [
    "SCHEDULER_ASSIGNMENT_PROJECTION_VERSION",
    "SESSION_ATTEMPT_PROJECTION_VERSION",
    "SESSION_CANONICAL_STATES",
    "WORK_CANONICAL_STATES",
    "WORK_LIFECYCLE_PROJECTION_VERSION",
    "canonical_assignment_state",
    "canonical_session_attempt",
    "canonical_work_lifecycle",
]
