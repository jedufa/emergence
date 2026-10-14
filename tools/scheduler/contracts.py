"""Deterministic records and contracts for the optional EIEL worker scheduler.

The scheduler package owns no project truth.  These helpers normalize and
validate the four scheduler-facing record payloads and provide file-backed
stores that reuse the existing immutable ActorRecordStore envelope.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.actors.canonical import canonical_bytes, canonical_sha256
from tools.actors.contract_versions import WORKER_RESULT_SCHEMA_VERSION
from tools.actors.common import (
    ActorError,
    advisory_lock,
    actor_state_root,
    atomic_write_json,
    canonical_host_actor_state_root,
    read_json,
    require_explicit_schema_version,
    stable_id,
    validate_timestamp,
)
from tools.actors.portfolio import normalize_string_list
from tools.actors.store import ActorRecordStore, scoped_actor_store

DISPATCH_SCHEMA_VERSION = "eiel_dispatch_request_v1"
ASSIGNMENT_SCHEMA_VERSION = "eiel_worker_assignment_v1"
LEASE_SCHEMA_VERSION = "eiel_assignment_lease_v1"
SCHEDULER_LAUNCH_ACK_SCHEMA_VERSION = "eiel_scheduler_launch_ack_v1"

DISPATCH_NAMESPACE = "dispatch-requests"
ASSIGNMENT_NAMESPACE = "assignments"
LEASE_NAMESPACE = "leases"
RESULT_NAMESPACE = "worker-results"
SCHEDULER_DIRECTORY = "scheduler"

AUTHORITY_PHASES = ("implementation", "review", "repair")
ASSIGNMENT_STATUSES = (
    "PREPARING", "CLEANUP_REQUIRED", "READY", "CLAIMED", "RUNNING", "RESULT_SUBMITTED",
    "CLOSED", "INVALID", "SUPERSEDED", "EXPIRED",
)
LEASE_STATUSES = ("ACTIVE", "EXPIRED", "REVOKED", "CLOSED")

# Executable lifecycle ownership.  Declaring status strings without one
# source-to-target policy left every recovery writer to invent its own rules.
# Self-loops are retained only where the record is being enriched without
# changing lifecycle ownership (heartbeat, launch acknowledgement, retryable
# cleanup, and idempotent terminal reconciliation).
ASSIGNMENT_ALLOWED_TRANSITIONS = {
    "PREPARING": {"PREPARING", "READY", "CLEANUP_REQUIRED", "INVALID", "SUPERSEDED"},
    "CLEANUP_REQUIRED": {"CLEANUP_REQUIRED", "INVALID", "SUPERSEDED"},
    "READY": {"READY", "CLAIMED", "SUPERSEDED"},
    "CLAIMED": {"CLAIMED", "RUNNING", "EXPIRED", "SUPERSEDED"},
    "RUNNING": {"RUNNING", "RESULT_SUBMITTED", "EXPIRED", "SUPERSEDED"},
    "RESULT_SUBMITTED": {"RESULT_SUBMITTED", "CLOSED"},
    "CLOSED": {"CLOSED"},
    "INVALID": {"INVALID"},
    "SUPERSEDED": {"SUPERSEDED"},
    "EXPIRED": {"EXPIRED", "RESULT_SUBMITTED"},
}

LEASE_ALLOWED_TRANSITIONS = {
    "ACTIVE": {"ACTIVE", "EXPIRED", "REVOKED", "CLOSED"},
    "EXPIRED": {"EXPIRED", "CLOSED"},
    "REVOKED": {"REVOKED"},
    "CLOSED": {"CLOSED"},
}
RESULT_OUTCOMES = (
    "COMPLETED", "CHECKPOINTED", "BLOCKED", "PROMOTION_REQUIRED",
    "ASSIGNMENT_STALE", "INTEGRITY_FAILURE", "PROCESS_FAILURE",
    "OPERATOR_DECISION_REQUIRED",
)
VALIDATION_STATUSES = ("PASSED", "FAILED", "NOT_RUN", "PARTIAL", "BLOCKED")
LAUNCHER_TYPES = ("fake", "codex_exec")
RESOURCE_ADMISSION_BINDING_SCHEMA_VERSION = (
    "crca_scheduler_resource_admission_binding_v1"
)
ATTEMPT_LEASE_ADMISSION_SCHEMA_VERSION = (
    "crca_scheduler_attempt_lease_admission_v1"
)
CANONICAL_CONTINUATION_LEASE_SCHEMA_VERSION = (
    "crca_canonical_continuation_lease_v1"
)


def scheduler_root(state_root: str | Path | None = None) -> Path:
    return actor_state_root(state_root) / SCHEDULER_DIRECTORY


def dispatch_store(
    state_root: str | Path | None = None,
    *,
    write_scope: bool = False,
) -> ActorRecordStore:
    if write_scope:
        return scoped_actor_store(
            scheduler_root(state_root),
            DISPATCH_NAMESPACE,
            actor="assignmentLinker",
            record_types=("dispatch_request",),
            reference_root=actor_state_root(state_root),
        )
    return ActorRecordStore(
        scheduler_root(state_root),
        DISPATCH_NAMESPACE,
        reference_root=actor_state_root(state_root),
    )


def dispatch_writer_store(state_root: str | Path | None = None) -> ActorRecordStore:
    return dispatch_store(state_root, write_scope=True)


def assignment_store(
    state_root: str | Path | None = None,
    *,
    write_scope: bool = False,
) -> ActorRecordStore:
    if write_scope:
        return scoped_actor_store(
            scheduler_root(state_root),
            ASSIGNMENT_NAMESPACE,
            actor="assignmentLinker",
            record_types=("assignment",),
            reference_root=actor_state_root(state_root),
        )
    return ActorRecordStore(
        scheduler_root(state_root),
        ASSIGNMENT_NAMESPACE,
        reference_root=actor_state_root(state_root),
    )


def assignment_writer_store(state_root: str | Path | None = None) -> ActorRecordStore:
    return assignment_store(state_root, write_scope=True)


def lease_store(
    state_root: str | Path | None = None,
    *,
    write_scope: bool = False,
) -> ActorRecordStore:
    if write_scope:
        return scoped_actor_store(
            scheduler_root(state_root),
            LEASE_NAMESPACE,
            actor="scheduler",
            record_types=("assignment_lease",),
            reference_root=actor_state_root(state_root),
        )
    return ActorRecordStore(
        scheduler_root(state_root),
        LEASE_NAMESPACE,
        reference_root=actor_state_root(state_root),
    )


def lease_writer_store(state_root: str | Path | None = None) -> ActorRecordStore:
    return lease_store(state_root, write_scope=True)


def result_store(
    state_root: str | Path | None = None,
    *,
    write_scope: bool = False,
) -> ActorRecordStore:
    if write_scope:
        return scoped_actor_store(
            scheduler_root(state_root),
            RESULT_NAMESPACE,
            actor="scheduler",
            record_types=("worker_result",),
            reference_root=actor_state_root(state_root),
        )
    return ActorRecordStore(
        scheduler_root(state_root),
        RESULT_NAMESPACE,
        reference_root=actor_state_root(state_root),
    )


def result_writer_store(state_root: str | Path | None = None) -> ActorRecordStore:
    return result_store(state_root, write_scope=True)


def resource_store(state_root: str | Path | None = None) -> ActorRecordStore:
    """Read exact Resource Manager observations without granting RM authority."""

    return ActorRecordStore(actor_state_root(state_root), "resources")


def parse_utc(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(validate_timestamp(value).replace("Z", "+00:00"))


def add_seconds(value: str, seconds: int | float) -> str:
    timestamp = parse_utc(value) + dt.timedelta(seconds=max(0, float(seconds)))
    return timestamp.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def dispatch_id_for(value: Mapping[str, Any]) -> str:
    return stable_id("DSP", dict(value), length=16)


def assignment_id_for(value: Mapping[str, Any]) -> str:
    return stable_id("ASSIGN", dict(value), length=16)


def lease_id_for(value: Mapping[str, Any]) -> str:
    return stable_id("LEASE", dict(value), length=16)


def result_id_for(value: Mapping[str, Any]) -> str:
    return stable_id("RESULT", dict(value), length=16)


def scheduler_id_for(value: Mapping[str, Any]) -> str:
    return stable_id("SCH", dict(value), length=12)


def _bool(raw: Mapping[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ActorError(f"{key} must be boolean", code="scheduler_contract_invalid")
    return value


def normalize_resource_admission_binding(
    raw: Mapping[str, Any] | None,
) -> dict[str, Any]:
    value = dict(raw or {})
    if value.get("schema_version") != RESOURCE_ADMISSION_BINDING_SCHEMA_VERSION:
        raise ActorError(
            "resource admission requires the versioned exact binding",
            code="scheduler_resource_admission_binding_invalid",
        )
    text_fields = (
        "attempt_admission_request_sha256",
        "resource_scope_sha256",
        "resource_observation_id",
        "resource_observation_record_sha256",
        "resource_recommendation_id",
        "resource_recommendation_record_sha256",
    )
    for field in text_fields:
        text = str(value.get(field) or "").strip()
        if field.endswith("sha256") and (
            len(text) != 64 or any(character not in "0123456789abcdef" for character in text)
        ):
            raise ActorError(
                f"resource admission {field} must be exact SHA-256",
                code="scheduler_resource_admission_binding_invalid",
            )
        if not text:
            raise ActorError(
                f"resource admission {field} is required",
                code="scheduler_resource_admission_binding_invalid",
            )
        value[field] = text
    resource_ids = tuple(sorted(set(
        str(item).strip() for item in value.get("resource_ids") or [] if str(item).strip()
    )))
    if not resource_ids:
        raise ActorError(
            "resource admission needs at least one exact resource ID",
            code="scheduler_resource_admission_binding_invalid",
        )
    start_window_seconds = int(value.get("start_window_seconds") or 300)
    ttl_seconds = int(value.get("ttl_seconds") or start_window_seconds)
    max_observation_age_seconds = int(
        value.get("max_observation_age_seconds") or 300
    )
    if not 30 <= start_window_seconds <= 3600:
        raise ActorError(
            "resource start window must be between 30 and 3600 seconds",
            code="scheduler_resource_admission_binding_invalid",
        )
    if not start_window_seconds <= ttl_seconds <= 7200:
        raise ActorError(
            "resource lease TTL must cover the start window and be at most 7200 seconds",
            code="scheduler_resource_admission_binding_invalid",
        )
    if not 1 <= max_observation_age_seconds <= 3600:
        raise ActorError(
            "resource observation age bound must be between 1 and 3600 seconds",
            code="scheduler_resource_admission_binding_invalid",
        )
    attempt_binding = dict(value.get("attempt_lease_admission") or {})
    if attempt_binding.get("schema_version") != ATTEMPT_LEASE_ADMISSION_SCHEMA_VERSION:
        raise ActorError(
            "resource admission requires an exact Attempt lease binding",
            code="scheduler_attempt_lease_binding_invalid",
        )
    continuation = dict(attempt_binding.get("canonical_continuation_lease") or {})
    expected_continuation_fields = {
        "schema_version",
        "lease_id",
        "run_id",
        "attempt_id",
        "restore_checkpoint_digest",
        "lineage_scope_digest",
        "owner_id",
        "lease_epoch",
        "fencing_token_digest",
        "prior_lease_digest",
        "granted_at_utc",
        "expires_at_utc",
    }
    if (
        continuation.get("schema_version")
        != CANONICAL_CONTINUATION_LEASE_SCHEMA_VERSION
        or set(continuation) != expected_continuation_fields
    ):
        raise ActorError(
            "canonical continuation lease wire record is incomplete or unknown",
            code="scheduler_attempt_lease_binding_invalid",
        )
    for field in ("lease_id", "run_id", "attempt_id", "owner_id"):
        if not isinstance(continuation.get(field), str) or not continuation[field].strip():
            raise ActorError(
                f"canonical continuation lease {field} is required",
                code="scheduler_attempt_lease_binding_invalid",
            )
    if int(continuation.get("lease_epoch") or 0) < 1:
        raise ActorError(
            "canonical continuation lease epoch must be positive",
            code="scheduler_attempt_lease_binding_invalid",
        )
    for field in ("lineage_scope_digest", "fencing_token_digest"):
        digest = dict(continuation.get(field) or {})
        if (
            digest.get("algorithm") != "sha256"
            or not isinstance(digest.get("hex"), str)
            or len(digest["hex"]) != 64
            or any(character not in "0123456789abcdef" for character in digest["hex"])
        ):
            raise ActorError(
                f"canonical continuation lease {field} is not exact SHA-256",
                code="scheduler_attempt_lease_binding_invalid",
            )
    for field in ("restore_checkpoint_digest", "prior_lease_digest"):
        if continuation.get(field) is None:
            continue
        digest = dict(continuation.get(field) or {})
        if (
            digest.get("algorithm") != "sha256"
            or not isinstance(digest.get("hex"), str)
            or len(digest["hex"]) != 64
            or any(character not in "0123456789abcdef" for character in digest["hex"])
        ):
            raise ActorError(
                f"canonical continuation lease {field} is not exact SHA-256",
                code="scheduler_attempt_lease_binding_invalid",
            )
    granted = validate_timestamp(str(continuation.get("granted_at_utc") or ""))
    expires = validate_timestamp(str(continuation.get("expires_at_utc") or ""))
    if parse_utc(expires) <= parse_utc(granted):
        raise ActorError(
            "canonical continuation lease interval is invalid",
            code="scheduler_attempt_lease_binding_invalid",
        )
    attempt_id = str(attempt_binding.get("attempt_id") or "")
    if attempt_id != continuation["attempt_id"]:
        raise ActorError(
            "Attempt ID differs from the continuation lease",
            code="scheduler_attempt_lease_binding_invalid",
        )
    continuation_digest = canonical_sha256(continuation)
    owner_digest = canonical_sha256(
        {
            "owner_id": continuation["owner_id"],
            "lease_digest": {
                "algorithm": "sha256",
                "hex": continuation_digest,
            },
            "lease_epoch": int(continuation["lease_epoch"]),
            "fencing_token_digest": continuation["fencing_token_digest"],
        }
    )
    return {
        "schema_version": RESOURCE_ADMISSION_BINDING_SCHEMA_VERSION,
        **{field: value[field] for field in text_fields},
        "resource_ids": list(resource_ids),
        "start_window_seconds": start_window_seconds,
        "ttl_seconds": ttl_seconds,
        "max_observation_age_seconds": max_observation_age_seconds,
        "attempt_lease_admission": {
            "schema_version": ATTEMPT_LEASE_ADMISSION_SCHEMA_VERSION,
            "attempt_id": attempt_id,
            "canonical_continuation_lease": continuation,
            "canonical_continuation_lease_sha256": continuation_digest,
            "continuation_owner_sha256": owner_digest,
        },
    }


def normalize_worker_profile(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    value = dict(raw or {})
    launcher_type = str(value.get("launcher_type") or "fake").strip().lower()
    if launcher_type not in LAUNCHER_TYPES:
        raise ActorError(
            f"launcher_type must be one of {', '.join(LAUNCHER_TYPES)}",
            code="scheduler_worker_profile_invalid",
        )
    sandbox = str(value.get("sandbox") or "workspace-write").strip()
    if sandbox not in {"read-only", "workspace-write"}:
        raise ActorError("unsupported Codex sandbox profile", code="scheduler_worker_profile_invalid")
    test_knobs = {key for key in ("timeout_seconds", "fake_sleep_seconds") if key in value}
    if test_knobs and os.environ.get("EIEL_SCHEDULER_TEST_MODE") != "1":
        raise ActorError(
            "fake worker timing knobs are test-only",
            code="scheduler_worker_profile_invalid",
            problems=sorted(test_knobs),
        )
    profile = {
        "profile_id": str(value.get("profile_id") or launcher_type).strip(),
        "launcher_type": launcher_type,
        "model": str(value.get("model") or "").strip() or None,
        "reasoning_level": str(value.get("reasoning_level") or "").strip().upper() or None,
        "profile": str(value.get("profile") or "").strip() or None,
        "sandbox": sandbox,
        "ephemeral": _bool(value, "ephemeral", True),
        "fake_behavior": str(value.get("fake_behavior") or "checkpoint").strip().lower(),
        "fake_fixture_content": str(value.get("fake_fixture_content") or "scheduler fake worker fixture\n"),
    }
    if "timeout_seconds" in value:
        profile["timeout_seconds"] = max(10, int(value["timeout_seconds"]))
    if "fake_sleep_seconds" in value:
        profile["fake_sleep_seconds"] = max(0.0, float(value["fake_sleep_seconds"]))
    return profile



def normalize_lease_policy(raw: Mapping[str, Any] | None) -> dict[str, int]:
    value = dict(raw or {})
    duration = int(value.get("duration_seconds", 3600))
    heartbeat = int(value.get("heartbeat_interval_seconds", 60))
    if duration < 30:
        raise ActorError("Lease duration must be at least 30 seconds", code="lease_policy_invalid")
    if heartbeat < 1 or heartbeat * 3 > duration:
        raise ActorError(
            "heartbeat_interval_seconds must be at least 1 and no more than one third of duration_seconds",
            code="lease_policy_invalid",
        )
    return {"duration_seconds": duration, "heartbeat_interval_seconds": heartbeat}


def normalize_dispatch_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(raw)
    if "schema_version" in value:
        require_explicit_schema_version(
            value,
            expected=DISPATCH_SCHEMA_VERSION,
            label="Dispatch Request",
        )
    work_id = str(value.get("work_id") or "").strip().upper()
    if not work_id.startswith("WORK-"):
        raise ActorError("Dispatch Request work_id must start with WORK-", code="dispatch_work_id_invalid")
    requested_allowed = [
        str(row).strip().lower()
        for row in (value.get("allowed_authority_phases") or ["implementation"])
    ]
    requested_allowed = list(dict.fromkeys(row for row in requested_allowed if row))
    unsupported = sorted(row for row in requested_allowed if row not in AUTHORITY_PHASES)
    # Eligibility treats this field as a set.  Canonicalize it in the frozen
    # authority-phase order so equivalent Dispatch Requests cannot acquire
    # different record IDs or Assignment linker digests merely by reordering
    # the JSON array.
    allowed = [phase for phase in AUTHORITY_PHASES if phase in requested_allowed]
    if unsupported or not allowed:
        raise ActorError(
            "Dispatch Request contains unsupported authority phases",
            code="dispatch_phase_invalid",
            problems=unsupported or ["at least one authority phase is required"],
        )
    repository = dict(value.get("repository") or {})
    repo_root = str(repository.get("repo_root") or repository.get("working_directory") or "").strip()
    if not repo_root:
        raise ActorError("Dispatch Request repository.repo_root is required", code="dispatch_repository_missing")
    queue = dict(value.get("queue") or {})
    resource_admission_required = _bool(
        value, "resource_admission_required", False
    )
    resource_admission = (
        normalize_resource_admission_binding(
            value.get("resource_admission")
            if isinstance(value.get("resource_admission"), Mapping)
            else None
        )
        if resource_admission_required
        else None
    )
    if not resource_admission_required and value.get("resource_admission") is not None:
        raise ActorError(
            "resource admission binding is present while admission is disabled",
            code="scheduler_resource_admission_binding_invalid",
        )
    automation = dict(value.get("automation_policy") or {})
    payload = {
        "schema_version": DISPATCH_SCHEMA_VERSION,
        "work_id": work_id,
        "enabled": _bool(value, "enabled", True),
        "allowed_authority_phases": allowed,
        "worker_profile": normalize_worker_profile(value.get("worker_profile") if isinstance(value.get("worker_profile"), Mapping) else None),
        "repository": {
            "repo_root": str(Path(repo_root).expanduser().resolve()),
            "expected_branch": str(repository.get("expected_branch") or "").strip() or None,
        },
        "queue": {
            "priority": int(queue.get("priority", queue.get("rank", 100))),
        },
        "automation_policy": {
            "allow_fresh_session": _bool(automation, "allow_fresh_session", True),
            "allow_resume": _bool(automation, "allow_resume", True),
            "allow_independent_review": _bool(automation, "allow_independent_review", True),
            "allow_repair": _bool(automation, "allow_repair", True),
            "allow_work_completion": False,
        },
        "lease_policy": normalize_lease_policy(value.get("lease_policy") if isinstance(value.get("lease_policy"), Mapping) else None),
        "resource_admission_required": resource_admission_required,
        "resource_admission": resource_admission,
        "created_by": str(value.get("created_by") or "HUMAN_OPERATOR").strip().upper(),
        "note": str(value.get("note") or "").strip() or None,
    }
    return payload


def assignment_material_sha256(material: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(material))


def normalize_assignment_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(raw)
    if "schema_version" in value:
        require_explicit_schema_version(
            value,
            expected=ASSIGNMENT_SCHEMA_VERSION,
            label="Worker Assignment",
        )
    material = value.get("material")
    normalized_material = dict(material) if isinstance(material, Mapping) else None
    digest = str(value.get("assignment_material_sha256") or "").strip() or None
    if normalized_material is not None:
        actual = assignment_material_sha256(normalized_material)
        if digest and digest != actual:
            raise ActorError("Assignment material digest mismatch", code="assignment_digest_mismatch")
        digest = actual
    return {
        "schema_version": ASSIGNMENT_SCHEMA_VERSION,
        "material": normalized_material,
        "assignment_material_sha256": digest,
        "preparation": dict(value.get("preparation") or {}),
        "operational": {
            "current_lease_id": dict(value.get("operational") or {}).get("current_lease_id"),
            "result_id": dict(value.get("operational") or {}).get("result_id"),
            "claim_count": int(dict(value.get("operational") or {}).get("claim_count", 0)),
        },
    }


def verify_assignment_payload(payload: Mapping[str, Any], *, require_ready: bool = False) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != ASSIGNMENT_SCHEMA_VERSION:
        errors.append("unsupported Assignment schema_version")
    material = payload.get("material")
    digest = payload.get("assignment_material_sha256")
    if material is not None:
        if not isinstance(material, Mapping):
            errors.append("Assignment material must be an object")
        elif digest != assignment_material_sha256(material):
            errors.append("Assignment material digest mismatch")
    elif require_ready:
        errors.append("READY Assignment has no material")
    return errors


def normalize_lease_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(raw)
    if "schema_version" in value:
        require_explicit_schema_version(
            value,
            expected=LEASE_SCHEMA_VERSION,
            label="Assignment Lease",
        )
    worker = dict(value.get("worker") or {})
    policy = normalize_lease_policy(value)
    return {
        "schema_version": LEASE_SCHEMA_VERSION,
        "assignment_id": str(value.get("assignment_id") or "").strip().upper(),
        "assignment_material_sha256": str(value.get("assignment_material_sha256") or "").strip(),
        "scheduler_id": str(value.get("scheduler_id") or "").strip().upper(),
        "claimed_at_utc": validate_timestamp(str(value.get("claimed_at_utc") or "")),
        "expires_at_utc": validate_timestamp(str(value.get("expires_at_utc") or "")),
        "last_heartbeat_at_utc": validate_timestamp(str(value.get("last_heartbeat_at_utc") or value.get("claimed_at_utc") or "")),
        "duration_seconds": policy["duration_seconds"],
        "heartbeat_interval_seconds": policy["heartbeat_interval_seconds"],
        "worker": {
            "kind": str(worker.get("kind") or "").strip() or None,
            "worker_id": str(worker.get("worker_id") or "").strip() or None,
            "pid": int(worker["pid"]) if worker.get("pid") is not None else None,
            "process_start_identity": str(worker.get("process_start_identity") or "").strip() or None,
            "thread_id": str(worker.get("thread_id") or "").strip() or None,
        },
    }


def normalize_worker_result_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(raw)
    if "schema_version" in value:
        require_explicit_schema_version(
            value,
            expected=WORKER_RESULT_SCHEMA_VERSION,
            label="Worker Result",
        )
    outcome = str(value.get("outcome") or "").strip().upper()
    if outcome not in RESULT_OUTCOMES:
        raise ActorError(
            "unsupported Worker Result outcome",
            code="worker_result_outcome_invalid",
            problems=[outcome or "<empty>"],
        )
    validation = dict(value.get("validation") or {})
    validation_status = str(validation.get("status") or "NOT_RUN").strip().upper()
    if validation_status not in VALIDATION_STATUSES:
        raise ActorError("unsupported Worker Result validation status", code="worker_result_validation_invalid")
    validation_gaps = normalize_string_list(validation.get("gaps"))
    decision_impact = str(validation.get("decision_impact") or "").strip() or None
    if validation_status == "PARTIAL" and (not validation_gaps or not decision_impact):
        raise ActorError(
            "PARTIAL validation requires explicit gaps and decision_impact",
            code="worker_result_validation_partial_incomplete",
        )
    if validation_status == "PASSED" and validation_gaps:
        raise ActorError(
            "PASSED validation cannot retain unresolved gaps",
            code="worker_result_validation_contradictory",
        )
    worker = dict(value.get("worker_identity") or {})
    normalized_worker = {
        "kind": str(worker.get("kind") or "").strip(),
        "worker_id": str(worker.get("worker_id") or "").strip(),
        "pid": int(worker["pid"]) if worker.get("pid") is not None else None,
        "process_start_identity": str(worker.get("process_start_identity") or "").strip() or None,
        "thread_id": str(worker.get("thread_id") or "").strip() or None,
    }
    if not normalized_worker["kind"] or not normalized_worker["worker_id"]:
        raise ActorError("Worker Result requires kind and worker_id", code="worker_result_identity_invalid")
    return {
        "schema_version": WORKER_RESULT_SCHEMA_VERSION,
        "assignment_id": str(value.get("assignment_id") or "").strip().upper(),
        "assignment_material_sha256": str(value.get("assignment_material_sha256") or "").strip(),
        "context_receipt_sha256": str(value.get("context_receipt_sha256") or "").strip(),
        "lease_id": str(value.get("lease_id") or "").strip().upper(),
        "work_id": str(value.get("work_id") or "").strip().upper(),
        "session_id": str(value.get("session_id") or "").strip().upper(),
        "worker_identity": normalized_worker,
        "outcome": outcome,
        "completion_scope": "ASSIGNMENT_ATTEMPT",
        "repository": dict(value.get("repository") or {}),
        "session": dict(value.get("session") or {}),
        "execution": dict(value.get("execution") or {}),
        "validation": {
            "status": validation_status,
            "commands": normalize_string_list(validation.get("commands")),
            "report_paths": normalize_string_list(validation.get("report_paths") or value.get("report_paths")),
            "gaps": validation_gaps,
            "decision_impact": decision_impact,
        },
        "blockers": normalize_string_list(value.get("blockers")),
    }


def load_dispatch_payload(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ActorError("Dispatch Request must be an object", code="schema_record_malformed")
    require_explicit_schema_version(
        raw,
        expected=DISPATCH_SCHEMA_VERSION,
        label="Dispatch Request",
    )
    return normalize_dispatch_payload(raw)


def load_assignment_payload(raw: Any, *, require_ready: bool = False) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ActorError("Worker Assignment must be an object", code="schema_record_malformed")
    require_explicit_schema_version(
        raw,
        expected=ASSIGNMENT_SCHEMA_VERSION,
        label="Worker Assignment",
    )
    value = normalize_assignment_payload(raw)
    errors = verify_assignment_payload(value, require_ready=require_ready)
    if errors:
        raise ActorError(
            "invalid Worker Assignment",
            code="schema_record_malformed",
            problems=errors,
        )
    return value


def load_lease_payload(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ActorError("Assignment Lease must be an object", code="schema_record_malformed")
    require_explicit_schema_version(
        raw,
        expected=LEASE_SCHEMA_VERSION,
        label="Assignment Lease",
    )
    return normalize_lease_payload(raw)


def load_worker_result_payload(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ActorError("Worker Result must be an object", code="schema_record_malformed")
    require_explicit_schema_version(
        raw,
        expected=WORKER_RESULT_SCHEMA_VERSION,
        label="Worker Result",
    )
    return normalize_worker_result_payload(raw)


def phase_to_role(phase: str) -> str:
    normalized = str(phase).strip().lower()
    mapping = {
        "implementation": "IMPLEMENTATION_AUTHORITY",
        "repair": "IMPLEMENTATION_AUTHORITY",
        "review": "REVIEW_AUTHORITY",
    }
    if normalized not in mapping:
        raise ActorError("unsupported Assignment authority phase", code="assignment_phase_invalid")
    return mapping[normalized]


def records(store_obj: ActorRecordStore, *, read_only: bool = False) -> list[dict[str, Any]]:
    """Load scheduler records through strict current or forensic read-only paths."""

    payload_loaders = {
        "dispatch-requests": load_dispatch_payload,
        "assignments": load_assignment_payload,
        "leases": load_lease_payload,
        "worker-results": load_worker_result_payload,
    }
    rows: list[dict[str, Any]] = []
    for index in (store_obj.list_read_only() if read_only else store_obj.list()):
        if not index.get("record_id") or index.get("status") == "INVALID":
            continue
        record = (store_obj.get_read_only(str(index["record_id"])) if read_only else store_obj.get(str(index["record_id"])))
        loader = payload_loaders.get(store_obj.namespace)
        if loader is not None:
            loader(record.get("payload"))
        rows.append(record)
    return rows


def active_lease_for_assignment(assignment_id: str, *, state_root: str | Path | None = None, now: str | None = None) -> dict[str, Any] | None:
    current = parse_utc(now) if now else None
    for lease in records(lease_store(state_root)):
        if lease.get("status") != "ACTIVE":
            continue
        payload = dict(lease.get("payload") or {})
        if payload.get("assignment_id") != assignment_id:
            continue
        if current is not None and parse_utc(str(payload.get("expires_at_utc"))) <= current:
            continue
        return lease
    return None


__all__ = [
    "ASSIGNMENT_SCHEMA_VERSION", "ASSIGNMENT_STATUSES", "AUTHORITY_PHASES",
    "DISPATCH_SCHEMA_VERSION", "LEASE_SCHEMA_VERSION", "LEASE_STATUSES",
    "RESULT_OUTCOMES", "WORKER_RESULT_SCHEMA_VERSION", "VALIDATION_STATUSES",
    "active_lease_for_assignment", "add_seconds", "assignment_id_for",
    "assignment_material_sha256", "assignment_store", "dispatch_id_for",
    "dispatch_store", "lease_id_for", "lease_store", "load_assignment_payload",
    "load_dispatch_payload", "load_lease_payload", "load_worker_result_payload", "normalize_assignment_payload",
    "normalize_dispatch_payload", "normalize_lease_payload", "normalize_lease_policy", "normalize_resource_admission_binding", "normalize_worker_profile",
    "normalize_worker_result_payload", "parse_utc", "phase_to_role", "records",
    "result_id_for", "result_store", "scheduler_id_for", "scheduler_root",
    "verify_assignment_payload",
]
