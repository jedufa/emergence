"""Small typed closure contracts shared by Work and Session records.

These records deliberately remain embedded in the manager-owned Work/Session
payloads.  The module freezes their schemas and validation semantics without
creating another authority surface or another manager.
"""

from __future__ import annotations

from typing import Any, Mapping

from tools.actors.common import ActorError, require_explicit_schema_version, require_known_fields, validate_timestamp

WORK_EVIDENCE_REQUIREMENT_SCHEMA_VERSION = "eiel_work_evidence_requirement_v1"
WORK_HOLD_SCHEMA_VERSION = "eiel_work_hold_v1"
DELIVERY_COMPONENT_SCHEMA_VERSION = "eiel_delivery_component_v1"
SESSION_ATTEMPT_DISPOSITION_SCHEMA_VERSION = "eiel_session_attempt_disposition_v1"
SESSION_ARTIFACT_DISPOSITION_SCHEMA_VERSION = "eiel_session_artifact_disposition_v1"
REVIEW_PROGRESS_SCHEMA_VERSION = "eiel_review_progress_v1"
REPAIR_LINEAGE_SCHEMA_VERSION = "eiel_repair_lineage_v1"

EVIDENCE_REQUIREMENT_MODES = ("GATE", "PROVENANCE_ONLY")
HOLD_KINDS = ("REVIEW_DEFERRED", "RESOURCE_WAIT", "OPERATOR_DECISION")
DELIVERY_COMPONENT_DISPOSITIONS = ("INCLUDED", "MODIFIED", "REFERENCE_ONLY", "REJECTED")
ATTEMPT_DISPOSITIONS = ("SUCCEEDED", "FAILED", "ABANDONED", "NOT_NEEDED", "DUPLICATE", "SUPERSEDED")
ARTIFACT_DISPOSITIONS = ("INCLUDED", "RETAINED_REFERENCE", "REJECTED", "DISCARDED")
REVIEW_PROGRESS_STATES = ("NOT_STARTED", "IN_PROGRESS", "DEFERRED", "COMPLETED")


def _object(raw: Any, *, label: str, schema: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ActorError(f"{label} must be an object", code="schema_record_malformed")
    require_explicit_schema_version(raw, expected=schema, label=label)
    require_known_fields(raw, allowed=allowed, label=label)
    return dict(raw)


def load_work_evidence_requirement(raw: Any) -> dict[str, Any]:
    value = _object(
        raw,
        label="Work evidence requirement",
        schema=WORK_EVIDENCE_REQUIREMENT_SCHEMA_VERSION,
        allowed={
            "schema_version", "mode", "required_assessment_ids", "accepted_decisions",
            "supports_proposition", "recorded_at_utc",
        },
    )
    mode = str(value.get("mode") or "").upper()
    if mode not in EVIDENCE_REQUIREMENT_MODES:
        raise ActorError("Work evidence requirement mode is invalid", code="schema_record_malformed")
    assessment_ids = sorted({str(item).strip().upper() for item in value.get("required_assessment_ids") or [] if str(item).strip()})
    decisions = sorted({str(item).strip().upper() for item in value.get("accepted_decisions") or [] if str(item).strip()})
    proposition = str(value.get("supports_proposition") or "").strip()
    if mode == "GATE" and (not assessment_ids or not decisions or not proposition):
        raise ActorError(
            "gating evidence requirement needs assessment IDs, accepted decisions, and proposition",
            code="schema_record_malformed",
        )
    value.update({
        "mode": mode,
        "required_assessment_ids": assessment_ids,
        "accepted_decisions": decisions,
        "supports_proposition": proposition or None,
        "recorded_at_utc": validate_timestamp(str(value.get("recorded_at_utc") or "")),
    })
    return value


def load_work_hold(raw: Any) -> dict[str, Any]:
    value = _object(
        raw,
        label="Work hold",
        schema=WORK_HOLD_SCHEMA_VERSION,
        allowed={
            "schema_version", "kind", "reason", "authority", "created_at_utc",
            "resume_condition", "resume_after_utc",
        },
    )
    kind = str(value.get("kind") or "").upper()
    if kind not in HOLD_KINDS:
        raise ActorError("Work hold kind is invalid", code="schema_record_malformed")
    authority = value.get("authority")
    if not isinstance(authority, Mapping) or not str(authority.get("role") or "").strip():
        raise ActorError("Work hold requires attributable authority", code="schema_record_malformed")
    reason = str(value.get("reason") or "").strip()
    resume_condition = str(value.get("resume_condition") or "").strip()
    if not reason or not resume_condition:
        raise ActorError("Work hold requires reason and resume_condition", code="schema_record_malformed")
    value.update({
        "kind": kind,
        "reason": reason,
        "authority": dict(authority),
        "created_at_utc": validate_timestamp(str(value.get("created_at_utc") or "")),
        "resume_condition": resume_condition,
        "resume_after_utc": validate_timestamp(str(value["resume_after_utc"])) if value.get("resume_after_utc") else None,
    })
    return value


def load_delivery_component(raw: Any) -> dict[str, Any]:
    value = _object(
        raw,
        label="delivery component",
        schema=DELIVERY_COMPONENT_SCHEMA_VERSION,
        allowed={
            "schema_version", "source_session_id", "source_commit", "contribution_paths",
            "artifact_refs", "disposition", "reason", "recorded_at_utc",
        },
    )
    disposition = str(value.get("disposition") or "").upper()
    if disposition not in DELIVERY_COMPONENT_DISPOSITIONS:
        raise ActorError("delivery component disposition is invalid", code="schema_record_malformed")
    session_id = str(value.get("source_session_id") or "").strip().upper()
    reason = str(value.get("reason") or "").strip()
    if not session_id.startswith("SES-") or not reason:
        raise ActorError("delivery component requires source Session and reason", code="schema_record_malformed")
    value.update({
        "source_session_id": session_id,
        "source_commit": str(value.get("source_commit") or "").strip().lower() or None,
        "contribution_paths": sorted({str(item).strip() for item in value.get("contribution_paths") or [] if str(item).strip()}),
        "artifact_refs": [dict(item) for item in value.get("artifact_refs") or [] if isinstance(item, Mapping)],
        "disposition": disposition,
        "reason": reason,
        "recorded_at_utc": validate_timestamp(str(value.get("recorded_at_utc") or "")),
    })
    return value


def load_attempt_disposition(raw: Any) -> dict[str, Any]:
    value = _object(
        raw,
        label="Session attempt disposition",
        schema=SESSION_ATTEMPT_DISPOSITION_SCHEMA_VERSION,
        allowed={
            "schema_version", "disposition", "reason", "authority", "prior_session_ref",
            "continuation_ref", "recorded_at_utc",
        },
    )
    disposition = str(value.get("disposition") or "").upper()
    if disposition not in ATTEMPT_DISPOSITIONS:
        raise ActorError("Session attempt disposition is invalid", code="schema_record_malformed")
    if not str(value.get("reason") or "").strip():
        raise ActorError("Session attempt disposition requires reason", code="schema_record_malformed")
    if not isinstance(value.get("authority"), Mapping):
        raise ActorError("Session attempt disposition requires authority", code="schema_record_malformed")
    value.update({
        "disposition": disposition,
        "reason": str(value.get("reason")).strip(),
        "authority": dict(value.get("authority") or {}),
        "prior_session_ref": dict(value.get("prior_session_ref") or {}) if isinstance(value.get("prior_session_ref"), Mapping) else None,
        "continuation_ref": dict(value.get("continuation_ref") or {}) if isinstance(value.get("continuation_ref"), Mapping) else None,
        "recorded_at_utc": validate_timestamp(str(value.get("recorded_at_utc") or "")),
    })
    return value


def load_artifact_disposition(raw: Any) -> dict[str, Any]:
    value = _object(
        raw,
        label="Session artifact disposition",
        schema=SESSION_ARTIFACT_DISPOSITION_SCHEMA_VERSION,
        allowed={"schema_version", "artifact_ref", "disposition", "reason", "recorded_at_utc"},
    )
    disposition = str(value.get("disposition") or "").upper()
    if disposition not in ARTIFACT_DISPOSITIONS or not isinstance(value.get("artifact_ref"), Mapping):
        raise ActorError("Session artifact disposition is invalid", code="schema_record_malformed")
    if not str(value.get("reason") or "").strip():
        raise ActorError("Session artifact disposition requires reason", code="schema_record_malformed")
    value.update({
        "artifact_ref": dict(value["artifact_ref"]),
        "disposition": disposition,
        "reason": str(value["reason"]).strip(),
        "recorded_at_utc": validate_timestamp(str(value.get("recorded_at_utc") or "")),
    })
    return value


def load_review_progress(raw: Any) -> dict[str, Any]:
    value = _object(
        raw,
        label="Review progress",
        schema=REVIEW_PROGRESS_SCHEMA_VERSION,
        allowed={
            "schema_version", "state", "completed_obligations", "remaining_obligations",
            "last_finding_id", "defer_reason", "resume_condition", "recorded_at_utc",
        },
    )
    state = str(value.get("state") or "").upper()
    if state not in REVIEW_PROGRESS_STATES:
        raise ActorError("Review progress state is invalid", code="schema_record_malformed")
    if state == "DEFERRED" and (not str(value.get("defer_reason") or "").strip() or not str(value.get("resume_condition") or "").strip()):
        raise ActorError("deferred Review progress requires reason and resume condition", code="schema_record_malformed")
    value.update({
        "state": state,
        "completed_obligations": sorted({str(item).strip() for item in value.get("completed_obligations") or [] if str(item).strip()}),
        "remaining_obligations": sorted({str(item).strip() for item in value.get("remaining_obligations") or [] if str(item).strip()}),
        "last_finding_id": str(value.get("last_finding_id") or "").strip() or None,
        "defer_reason": str(value.get("defer_reason") or "").strip() or None,
        "resume_condition": str(value.get("resume_condition") or "").strip() or None,
        "recorded_at_utc": validate_timestamp(str(value.get("recorded_at_utc") or "")),
    })
    return value


def load_repair_lineage(raw: Any) -> dict[str, Any]:
    value = _object(
        raw,
        label="repair lineage",
        schema=REPAIR_LINEAGE_SCHEMA_VERSION,
        allowed={
            "schema_version", "source_review_ref", "finding_ids", "repair_session_ref",
            "resolution", "waiver_authority", "rereview_session_ref", "recorded_at_utc",
        },
    )
    if not isinstance(value.get("source_review_ref"), Mapping):
        raise ActorError("repair lineage requires source review reference", code="schema_record_malformed")
    finding_ids = sorted({str(item).strip() for item in value.get("finding_ids") or [] if str(item).strip()})
    if not finding_ids:
        raise ActorError("repair lineage requires finding IDs", code="schema_record_malformed")
    resolution = str(value.get("resolution") or "").upper()
    if resolution not in {"OPEN", "REPAIRED", "WAIVED", "REJECTED"}:
        raise ActorError("repair lineage resolution is invalid", code="schema_record_malformed")
    value.update({
        "source_review_ref": dict(value["source_review_ref"]),
        "finding_ids": finding_ids,
        "repair_session_ref": dict(value.get("repair_session_ref") or {}) if isinstance(value.get("repair_session_ref"), Mapping) else None,
        "resolution": resolution,
        "waiver_authority": dict(value.get("waiver_authority") or {}) if isinstance(value.get("waiver_authority"), Mapping) else None,
        "rereview_session_ref": dict(value.get("rereview_session_ref") or {}) if isinstance(value.get("rereview_session_ref"), Mapping) else None,
        "recorded_at_utc": validate_timestamp(str(value.get("recorded_at_utc") or "")),
    })
    if resolution == "WAIVED" and value["waiver_authority"] is None:
        raise ActorError("waived finding requires waiver authority", code="schema_record_malformed")
    return value


__all__ = [name for name in globals() if name.startswith("load_") or name.endswith("_SCHEMA_VERSION")]
