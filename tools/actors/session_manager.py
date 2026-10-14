"""Durable short-term working memory for Codex sessions on EIEL work.

A session records where one Codex execution attempt stopped, the exact next
safe action, repository identity, commands, artifacts, and takeover history.
It never launches Codex, schedules work, or changes Work Manager records.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.actors.canonical import canonical_sha256
from tools.actors.acceptance import retain_and_validate_review_proof
from tools.actors.architecture_simplification import canonical_session_attempt
from tools.actors.bootstrap import (
    EDITING_TASK_MODES,
    bootstrap_refresh_command,
    git_repository_identity,
    validate_bootstrap_record,
)
from tools.actors.closure_contracts import (
    ARTIFACT_DISPOSITIONS,
    ATTEMPT_DISPOSITIONS,
    REPAIR_LINEAGE_SCHEMA_VERSION,
    REVIEW_PROGRESS_SCHEMA_VERSION,
    SESSION_ARTIFACT_DISPOSITION_SCHEMA_VERSION,
    SESSION_ATTEMPT_DISPOSITION_SCHEMA_VERSION,
    load_artifact_disposition,
    load_attempt_disposition,
    load_repair_lineage,
    load_review_progress,
)
from tools.actors.context_receipt import (
    build_context_receipt,
    epoch_for_session,
    normalize_observed_context_health,
    project_context_health,
    validate_context_receipt,
    write_context_receipt,
)
from tools.actors.common import (
    ActorError,
    apply_dotted_patch,
    emit_actor_error,
    emit_payload,
    new_record,
    parse_set_values,
    read_structured,
    require_explicit_schema_version,
    require_known_fields,
    require_test_fixture_context,
    revise_record,
    stable_id,
    timestamp_age_hours,
    utc_now,
    with_guidance,
)
from tools.actors.delivery_semantics import (
    canonicalize_delivery_evidence,
    canonicalize_review_evidence,
    canonicalize_review_target,
    delivery_commit as delivery_commit_value,
    delivery_session_id,
    delivery_sufficiency,
    delivering_codex_identities,
)
from tools.actors.lifecycle_policy import (
    AUTHORITATIVE_REVIEW_BINDING_SCHEMA_VERSION,
    actor_record_ref,
    work_execution_ref,
    codex_identities,
    context_contract_status,
    declared_identity_assurance,
    duplicate_delivery_claims,
    lifecycle_decision_snapshot,
    normalize_identity_assurance,
    require_open_work,
    require_record_ref,
    scheduler_identity_assurance,
)
from tools.actors.portfolio import (
    BOOTSTRAP_ID_PREFIX,
    SESSION_ACTIVE_STATUSES,
    SESSION_ID_PREFIX,
    SESSION_RECORD_TYPE,
    SESSION_RESUMABLE_STATUSES,
    SESSION_TERMINAL_STATUSES,
    WORK_ID_PREFIX,
    WORK_OPEN_STATUSES,
    WORK_TERMINAL_STATUSES,
    bootstrap_store,
    conversation_id_from_url,
    dependency_state,
    normalize_reference_map,
    normalize_string_list,
    parallel_session_ownership_problems,
    session_store,
    session_writer_store,
    sessions_for_work,
    validate_prefixed_id,
    work_lifecycle_lock,
    work_store,
)
from tools.actors.reference_integrity import actor_source_ref
from tools.actors.repository_truth import (
    ARTIFACT_REFERENCE_SCHEMA_VERSION,
    capture_adoption_receipt,
    capture_delivery_proof,
)
from tools.actors.release import operator_metadata
from tools.actors.resource_accounting import (
    capture_start,
    finalize_active_capture,
    finalize_session,
    load_session_resource_accounting,
    normalize_session_resource_accounting,
    require_resource_thread_available,
    resource_thread_lock,
    validate_session_resource_accounting,
)
from tools.actors.session_cli import build_parser as build_session_parser
from tools.actors.work_package import (
    AUTHORITY_ROLES,
    WORK_PACKAGE_BINDING_SCHEMA_VERSION,
    UNCERTAINTY_CLASSES,
    VERIFICATION_EXECUTION_STRICT,
    VERIFICATION_HISTORICAL_AUDIT,
    delivery_recorded_at,
    load_work_package,
    normalize_work_package_binding,
    normalize_work_package_chain,
    work_package_chain_bindings,
    resolve_work_package_binding,
    resolve_delivery_lineage,
    uncertainty_is_promotion,
    validate_changed_paths,
    verify_work_package_binding,
)

# Git discovery delegates to bootstrap.git_repository_identity, whose bounded
# read-only implementation uses subprocess.run rather than shell execution.

ACTOR_NAME = "sessionManager"
SESSION_SCHEMA_VERSION = "eiel_codex_session_v1"
NEXT_COMMAND_PRECONDITIONS_SCHEMA_VERSION = "eiel_next_command_preconditions_v1"
SESSION_DELIVERY_EVIDENCE_SCHEMA_VERSION = "eiel_session_delivery_evidence_v4"
SESSION_DELIVERY_EVIDENCE_LEGACY_SCHEMA_VERSION = "eiel_session_delivery_evidence_v3"
SESSION_DELIVERY_EVIDENCE_OLDEST_SCHEMA_VERSION = "eiel_session_delivery_evidence_v2"
REVIEW_EVIDENCE_SCHEMA_VERSION = "eiel_review_evidence_v4"
REVIEW_EVIDENCE_LEGACY_SCHEMA_VERSION = "eiel_review_evidence_v3"
REVIEW_EVIDENCE_OLDEST_SCHEMA_VERSION = "eiel_review_evidence_v2"
REVIEW_TARGET_SCHEMA_VERSION = "eiel_review_target_v2"
REVIEW_TARGET_LEGACY_SCHEMA_VERSION = "eiel_review_target_v1"
DELIVERY_MATERIAL_REF_SCHEMA_VERSION = "eiel_session_delivery_material_ref_v2"
DELIVERY_MATERIAL_REF_LEGACY_SCHEMA_VERSION = "eiel_session_delivery_material_ref_v1"
SESSION_STATUSES = (
    "CREATED",
    "ACTIVE",
    "PAUSED",
    "INTERRUPTED",
    "BLOCKED",
    "COMPLETED",
    "DISCONTINUED",
    "OBSOLETE",
    "SUPERSEDED",
)
SESSION_PAYLOAD_FIELDS = frozenset({
    "schema_version", "session_id", "work_id", "mission_id", "title",
    "bootstrap_record_id", "bootstrap_history", "original_codex_id",
    "original_codex_thread_url", "current_codex_id", "current_codex_thread_url",
    "preferred_continuation_codex_id", "preferred_continuation_codex_thread_url",
    "codex_participants", "repository", "current_stage", "current_step",
    "completed_steps", "last_successful_command", "last_failed_command",
    "interrupted_during", "blocked_reason", "blocked_by_work_ids", "next_action",
    "next_command", "next_command_preconditions", "session_creation_ordinal",
    "notes", "references", "parallel_session_acknowledged", "session_role",
    "model_assignment", "work_package_binding", "context_contract_status",
    "session_outcome", "review_verdict", "promotion_request", "delivery_evidence",
    "delivery_base_commit", "review_target", "review_evidence", "epoch",
    "authoritative_review_binding",
    "context_health", "context_receipt", "prepared_assignment_id",
    "scheduler_assignment_id", "scheduler_assignment_digest", "scheduler_lease_id",
    "scheduler_worker_id", "assignment_history", "worker_loss",
    "worker_identity_state", "identity_assurance", "resource_accounting",
    "resumed_from", "validation_events", "validation_gaps",
    "validation_decision_impact", "epoch_history", "terminal_transition",
    "attempt_disposition", "artifact_dispositions", "review_progress", "repair_lineage",
})
REVIEW_EVIDENCE_FIELDS = frozenset({
    "schema_version",
    "delivery_session_id",
    "review_session_id",
    "review_scope",
    "integrated_delivery_lineage",
    "delivering_codex_identities",
    "reviewing_codex_identities",
    "delivery_identity_assurance",
    "reviewing_identity_assurance",
    "identity_separation_assurance",
    "review_authority",
    "reviewed_commit",
    "reviewed_package_sha256",
    "reviewed_package_version",
    "tests_reviewed",
    "evidence_reviewed",
    "acceptance_criteria_reviewed",
    "known_limitations",
    "findings",
    "repair_disposition",
    "obligation_proof",
    "delivery_sufficiency",
    "acceptance_scope",
    "supersedes_review",
    "delivery_model_assignment",
    "review_model_assignment",
    "decision_snapshot",
    "reviewed_at_utc",
})
LEGACY_REVIEW_EVIDENCE_FIELDS = frozenset({
    "schema_version",
    "implementation_session_id",
    "review_session_id",
    "implementing_codex_identities",
    "reviewing_codex_identities",
    "implementing_identity_assurance",
    "reviewing_identity_assurance",
    "identity_separation_assurance",
    "review_authority",
    "reviewed_commit",
    "reviewed_package_sha256",
    "reviewed_package_version",
    "tests_reviewed",
    "evidence_reviewed",
    "acceptance_criteria_reviewed",
    "known_limitations",
    "findings",
    "repair_disposition",
    "obligation_proof",
    "implementation_sufficiency",
    "acceptance_scope",
    "supersedes_review",
    "implementation_model_assignment",
    "review_model_assignment",
    "decision_snapshot",
    "reviewed_at_utc",
})
SESSION_OUTCOMES = (
    "ARCHITECTURE_COMPLETE",
    "ENGINEERING_COMPLETE",
    "IMPLEMENTATION_COMPLETE",
    "DELIVERY_RECORDED",
    "EXTERNAL_DELIVERY_ADOPTED",
    "REVIEW_COMPLETE",
    "PROMOTION_REQUESTED",
    "SESSION_COMPLETE",
)
REVIEW_VERDICTS = ("ACCEPT", "CHANGES_REQUIRED", "REJECT", "PROMOTE", "BLOCKED")
DELIVERY_OUTCOMES = {
    "ARCHITECTURE_COMPLETE",
    "ENGINEERING_COMPLETE",
    "IMPLEMENTATION_COMPLETE",
    "DELIVERY_RECORDED",
    "EXTERNAL_DELIVERY_ADOPTED",
}
ROLE_DEFAULT_OUTCOME = {
    "ARCHITECTURE_AUTHORITY": "ARCHITECTURE_COMPLETE",
    "ENGINEERING_AUTHORITY": "ENGINEERING_COMPLETE",
    "IMPLEMENTATION_AUTHORITY": "DELIVERY_RECORDED",
    "REVIEW_AUTHORITY": "REVIEW_COMPLETE",
}
ALLOWED_TRANSITIONS = {
    "CREATED": {"CREATED", "ACTIVE", "PAUSED", "INTERRUPTED", "BLOCKED", "DISCONTINUED", "OBSOLETE", "SUPERSEDED"},
    "ACTIVE": {"ACTIVE", "PAUSED", "INTERRUPTED", "BLOCKED", "COMPLETED", "DISCONTINUED", "OBSOLETE", "SUPERSEDED"},
    "PAUSED": {"PAUSED", "ACTIVE", "INTERRUPTED", "BLOCKED", "DISCONTINUED", "OBSOLETE", "SUPERSEDED"},
    "INTERRUPTED": {"INTERRUPTED", "ACTIVE", "PAUSED", "BLOCKED", "DISCONTINUED", "OBSOLETE", "SUPERSEDED"},
    "BLOCKED": {"BLOCKED", "ACTIVE", "PAUSED", "INTERRUPTED", "DISCONTINUED", "OBSOLETE", "SUPERSEDED"},
    "COMPLETED": {"COMPLETED", "SUPERSEDED"},
    "DISCONTINUED": {"DISCONTINUED", "SUPERSEDED"},
    "OBSOLETE": {"OBSOLETE", "SUPERSEDED"},
    "SUPERSEDED": {"SUPERSEDED"},
}
def _require_package_delivery_role(
    package_binding: Mapping[str, Any] | None,
    requested_role: str,
) -> None:
    """Treat the package delivery authority as binding, not advisory.

    Review Authority is a separate phase.  Every non-review Session bound to a
    Work Package must use the package's exact delivery authority; callers may
    no longer relabel the contract by passing ``--role``.
    """

    if not package_binding or requested_role == "REVIEW_AUTHORITY":
        return
    required = str(dict(package_binding).get("recommended_authority") or "").strip().upper()
    if required and requested_role != required:
        raise ActorError(
            "Session role contradicts the Work Package required delivery authority",
            code="session_package_authority_mismatch",
            problems=[f"Required: {required}", f"Requested: {requested_role}"],
        )


def _require_bootstrap_session_authority(
    bootstrap: Mapping[str, Any],
    requested_role: str,
) -> None:
    """Compose Bootstrap task/edit authority into Session authority.

    Bootstrap is repository/environment readiness, but an editing Session may
    not turn a mechanical or read-only receipt into write authority merely by
    selecting an authoritative-looking role on the command line.  Review is a
    separate read-oriented phase and intentionally retains compatibility with
    implementation-shaped bootstraps created before v1.18e.
    """

    role = str(requested_role or "").strip().upper()
    if role == "REVIEW_AUTHORITY":
        return
    task = dict(dict(bootstrap.get("payload") or {}).get("task") or {})
    mode = str(task.get("mode") or "").strip().lower()
    problems: list[str] = []
    if mode not in EDITING_TASK_MODES:
        problems.append(
            f"Bootstrap task mode {mode or '<missing>'} does not authorize a write-capable {role} Session"
        )
    if not bool(task.get("edit_authorized")):
        problems.append("Bootstrap did not record explicit edit authorization")
    if not list(task.get("write_scope") or []):
        problems.append("Bootstrap did not retain a bounded write scope")
    if problems:
        raise ActorError(
            "Bootstrap task/edit authority does not authorize the requested Session role",
            code="session_bootstrap_authority_mismatch",
            problems=problems,
            related_records=[str(bootstrap.get("record_id") or "")],
        )


REFERENCE_FIELDS = (
    "qualification_record_ids",
    "resource_record_ids",
    "evidence_assessment_ids",
    "artifact_paths",
    "report_paths",
    "librarian_document_ids",
    "prompt_paths",
    "external_urls",
)

# Checkpoint is a progress projection, not a second writer for Session identity,
# authority, review, Scheduler binding, context-contract, epoch, or WRA truth.
SESSION_CHECKPOINT_EDITABLE_ROOTS = frozenset({
    "current_stage",
    "current_step",
    "completed_steps",
    "next_action",
    "next_command",
    "interrupted_during",
    "blocked_reason",
    "last_successful_command",
    "last_failed_command",
    "validation_events",
    "validation_gaps",
    "validation_decision_impact",
    "references",
    "context_health",
    "notes",
})


def _store(args: argparse.Namespace):
    return session_store(args.state_root, write_scope=True)


def persist_session_record_fixture(
    record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    allow_update: bool = False,
) -> Path:
    """Test-only raw fixture seam retained inside the canonical Session owner."""

    require_test_fixture_context()
    return session_store(state_root, write_scope=True).put(record, allow_update=allow_update)


def load_session_delivery_evidence(raw: Any) -> dict[str, Any]:
    """Load current or retained historical delivery evidence.

    V4 is the first generic-delivery schema. V2/V3 remain readable, but are
    canonicalized in memory to ``delivery_*`` names and are never emitted by
    current writers.
    """

    if not isinstance(raw, Mapping):
        raise ActorError("Session delivery evidence must be an object", code="schema_record_malformed")
    schema = require_explicit_schema_version(
        raw,
        expected=SESSION_DELIVERY_EVIDENCE_SCHEMA_VERSION,
        supported=(
            SESSION_DELIVERY_EVIDENCE_LEGACY_SCHEMA_VERSION,
            SESSION_DELIVERY_EVIDENCE_OLDEST_SCHEMA_VERSION,
        ),
        label="Session delivery evidence",
    )
    value = canonicalize_delivery_evidence(raw)
    if schema == SESSION_DELIVERY_EVIDENCE_SCHEMA_VERSION:
        forbidden = sorted(
            field for field in ("implementation_session_id", "implementation_commit", "implementation_sufficiency")
            if field in raw
        )
        if forbidden:
            raise ActorError(
                "current delivery evidence uses obsolete implementation-specific fields",
                code="schema_record_malformed",
                problems=[f"obsolete field: {field}" for field in forbidden],
            )
        required = (
            "delivery_session_id", "base_commit", "delivery_commit",
            "repo_root", "work_package_sha256", "changed_files", "worktree_clean",
            "delivery_kind", "delivery_sufficiency", "delivery_proof",
        )
    else:
        required = (
            "implementation_session_id", "base_commit", "implementation_commit",
            "repo_root", "work_package_sha256", "changed_files", "worktree_clean",
        )
    missing = [field for field in required if field not in raw]
    if missing:
        raise ActorError(
            "Session delivery evidence is malformed",
            code="schema_record_malformed",
            problems=[f"missing field: {field}" for field in missing],
        )
    if schema in {SESSION_DELIVERY_EVIDENCE_SCHEMA_VERSION, SESSION_DELIVERY_EVIDENCE_LEGACY_SCHEMA_VERSION}:
        if delivery_sufficiency(value) != "UNASSESSED":
            raise ActorError(
                "delivery evidence cannot claim delivery sufficiency",
                code="session_delivery_sufficiency_invalid",
            )
        if not isinstance(raw.get("delivery_proof"), Mapping):
            raise ActorError(
                "current delivery evidence requires a durable repository delivery proof",
                code="session_delivery_proof_missing",
            )
        kind = str(raw.get("delivery_kind") or "")
        if kind not in {"MANAGED_SESSION_DELIVERY", "EXTERNAL_DELIVERY_ADOPTED"}:
            raise ActorError("unsupported repository delivery kind", code="session_delivery_kind_invalid")
        if kind == "EXTERNAL_DELIVERY_ADOPTED" and not isinstance(raw.get("adoption_receipt"), Mapping):
            raise ActorError(
                "external delivery adoption requires an immutable adoption receipt",
                code="repository_adoption_receipt_missing",
            )
    value["source_schema_version"] = schema
    return value

def load_review_evidence(raw: Any, *, allow_legacy: bool = False) -> dict[str, Any]:
    """Load one durable Review Authority decision without laundering history."""

    if not isinstance(raw, Mapping):
        raise ActorError("Review evidence must be an object", code="schema_record_malformed")
    if not str(raw.get("schema_version") or "").strip():
        if allow_legacy:
            return dict(raw)
        raise ActorError(
            "current Review evidence requires an explicit schema_version",
            code="schema_version_missing",
        )
    schema = require_explicit_schema_version(
        raw,
        expected=REVIEW_EVIDENCE_SCHEMA_VERSION,
        supported=(REVIEW_EVIDENCE_LEGACY_SCHEMA_VERSION, REVIEW_EVIDENCE_OLDEST_SCHEMA_VERSION),
        label="Review evidence",
    )
    require_known_fields(
        raw,
        allowed=(REVIEW_EVIDENCE_FIELDS if schema == REVIEW_EVIDENCE_SCHEMA_VERSION else LEGACY_REVIEW_EVIDENCE_FIELDS),
        label="Review evidence",
    )
    if schema == REVIEW_EVIDENCE_SCHEMA_VERSION:
        forbidden = sorted(
            field for field in (
                "implementation_session_id", "implementing_codex_identities",
                "implementing_identity_assurance", "implementation_sufficiency",
                "implementation_model_assignment",
            ) if field in raw
        )
        if forbidden:
            raise ActorError(
                "current Review evidence uses obsolete implementation-specific fields",
                code="schema_record_malformed",
                problems=[f"obsolete field: {field}" for field in forbidden],
            )
        required = (
            "delivery_session_id", "review_session_id", "reviewed_commit",
            "reviewed_package_sha256", "delivery_sufficiency", "acceptance_scope",
            "known_limitations", "supersedes_review", "reviewed_at_utc",
        )
    else:
        required = (
            "implementation_session_id", "review_session_id", "reviewed_commit",
            "reviewed_package_sha256", "implementation_sufficiency", "acceptance_scope",
            "known_limitations", "supersedes_review", "reviewed_at_utc",
        )
    missing = [field for field in required if field not in raw]
    if missing:
        raise ActorError(
            "Review evidence is malformed",
            code="schema_record_malformed",
            problems=[f"missing field: {field}" for field in missing],
        )
    value = canonicalize_review_evidence(raw)
    if delivery_sufficiency(value) not in {"SUFFICIENT", "INSUFFICIENT", "UNASSESSED"}:
        raise ActorError("Review evidence delivery_sufficiency is invalid", code="schema_record_malformed")
    if value.get("acceptance_scope") not in {None, "UNCONDITIONAL", "WITH_LIMITS"}:
        raise ActorError("Review evidence acceptance_scope is invalid", code="schema_record_malformed")
    limitations = value.get("known_limitations")
    if not isinstance(limitations, list) or any(not str(item).strip() for item in limitations):
        raise ActorError(
            "Review evidence known_limitations must be a non-empty-string array",
            code="schema_record_malformed",
        )
    proof_ref = value.get("obligation_proof")
    if proof_ref is not None:
        if not isinstance(proof_ref, Mapping):
            raise ActorError("Review obligation proof reference must be an object", code="schema_record_malformed")
        require_explicit_schema_version(
            proof_ref,
            expected=ARTIFACT_REFERENCE_SCHEMA_VERSION,
            label="Review obligation proof reference",
        )
    supersedes = value.get("supersedes_review")
    if supersedes is not None:
        if not isinstance(supersedes, Mapping):
            raise ActorError("superseded Review reference must be an object", code="schema_record_malformed")
        require_explicit_schema_version(
            supersedes,
            expected="eiel_actor_record_ref_v1",
            label="superseded Review reference",
        )
        require_known_fields(
            supersedes,
            allowed={"schema_version", "record_id", "revision", "record_sha256", "status"},
            label="superseded Review reference",
        )
        missing_ref = [
            field for field in ("record_id", "revision", "record_sha256", "status")
            if field not in supersedes or supersedes.get(field) in (None, "")
        ]
        if missing_ref:
            raise ActorError(
                "superseded Review reference is incomplete",
                code="schema_record_malformed",
                problems=[f"missing field: {field}" for field in missing_ref],
            )
    if value.get("acceptance_scope") == "WITH_LIMITS" and not limitations:
        raise ActorError("WITH_LIMITS review evidence requires at least one limitation", code="schema_record_malformed")
    if value.get("acceptance_scope") == "UNCONDITIONAL" and limitations:
        raise ActorError("UNCONDITIONAL review evidence cannot retain limitations", code="schema_record_malformed")
    assurance = str(value.get("identity_separation_assurance") or "")
    allowed_assurance = (
        {
            "DECLARED_SEPARATION", "INDEPENDENT_SCHEDULER_BOUND",
            "REVIEWER_INDEPENDENCE_UNAVAILABLE",
        }
        if schema in {REVIEW_EVIDENCE_SCHEMA_VERSION, REVIEW_EVIDENCE_LEGACY_SCHEMA_VERSION}
        else {"DECLARED", "SCHEDULER_BOUND"}
    )
    if assurance not in allowed_assurance:
        raise ActorError(
            "Review evidence identity_separation_assurance is invalid for its schema version",
            code="schema_record_malformed",
        )
    value["source_schema_version"] = schema
    return value

def load_next_command_preconditions(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ActorError("next-command preconditions must be an object", code="schema_record_malformed")
    require_explicit_schema_version(
        raw,
        expected=NEXT_COMMAND_PRECONDITIONS_SCHEMA_VERSION,
        label="next-command preconditions",
    )
    return dict(raw)


def load_session_payload(raw: Any) -> dict[str, Any]:
    """Strict read-only loader for one persisted Codex Session payload."""

    if not isinstance(raw, Mapping):
        raise ActorError("Codex Session payload must be an object", code="schema_record_malformed")
    require_explicit_schema_version(raw, expected=SESSION_SCHEMA_VERSION, label="Codex Session")
    require_known_fields(raw, allowed=SESSION_PAYLOAD_FIELDS, label="Codex Session payload")
    value = dict(raw)
    binding = value.get("work_package_binding")
    if binding is not None:
        if not isinstance(binding, Mapping):
            raise ActorError("Work Package binding must be an object", code="schema_record_malformed")
        require_explicit_schema_version(
            binding,
            expected=WORK_PACKAGE_BINDING_SCHEMA_VERSION,
            label="Work Package binding",
        )
    accounting = value.get("resource_accounting")
    if accounting is not None:
        load_session_resource_accounting(accounting)
    preconditions = value.get("next_command_preconditions")
    if preconditions is not None:
        load_next_command_preconditions(preconditions)
    delivery = value.get("delivery_evidence")
    if delivery is not None:
        value["delivery_evidence"] = load_session_delivery_evidence(delivery)
    binding_kind = str(dict(binding or {}).get("sha256_kind") or "raw_file_sha256_v1")
    target = value.get("review_target")
    if target is not None:
        value["review_target"] = load_review_target(
            target, allow_legacy=bool(binding) and binding_kind == "raw_file_sha256_v1"
        )
    review = value.get("review_evidence")
    if review is not None:
        value["review_evidence"] = load_review_evidence(
            review, allow_legacy=bool(binding) and binding_kind == "raw_file_sha256_v1"
        )
    if value.get("attempt_disposition") is not None:
        load_attempt_disposition(value["attempt_disposition"])
    for row in value.get("artifact_dispositions") or []:
        load_artifact_disposition(row)
    if value.get("review_progress") is not None:
        load_review_progress(value["review_progress"])
    if value.get("repair_lineage") is not None:
        load_repair_lineage(value["repair_lineage"])
    return value


def load_review_target(raw: Any, *, allow_legacy: bool = False) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ActorError("Review target must be an object", code="schema_record_malformed")
    if not str(raw.get("schema_version") or "").strip():
        if allow_legacy:
            return dict(raw)
        raise ActorError("Review target schema_version is missing", code="schema_version_missing")
    schema = require_explicit_schema_version(
        raw,
        expected=REVIEW_TARGET_SCHEMA_VERSION,
        supported=(REVIEW_TARGET_LEGACY_SCHEMA_VERSION,),
        label="Review target",
    )
    value = canonicalize_review_target(raw)
    if schema == REVIEW_TARGET_SCHEMA_VERSION and "implementation_session_id" in raw:
        raise ActorError(
            "current Review target uses obsolete implementation_session_id",
            code="schema_record_malformed",
        )
    material = value.get("delivery_material")
    if not isinstance(material, Mapping):
        raise ActorError("Review target has no material delivery reference", code="schema_record_malformed")
    require_explicit_schema_version(
        material,
        expected=DELIVERY_MATERIAL_REF_SCHEMA_VERSION,
        supported=(DELIVERY_MATERIAL_REF_LEGACY_SCHEMA_VERSION,),
        label="review delivery material",
    )
    value["source_schema_version"] = schema
    return value

def _session_repository(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "repo_root": identity.get("root"),
        "branch": identity.get("branch"),
        "head": identity.get("head"),
        "status_short_branch": identity.get("status_short_branch"),
        "working_state_sha256": identity.get("working_state_sha256"),
        "repository_state_sha256": identity.get("repository_state_sha256"),
        "changed_paths": list(identity.get("changed_paths") or []),
        "dirty": bool(identity.get("dirty")),
    }


def _repository_matches_session(stored: Mapping[str, Any], live: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if Path(str(stored.get("repo_root") or "/")).resolve() != Path(str(live.get("root") or "/")).resolve():
        errors.append("Session repository root changed")
    for field, label in (
        ("head", "HEAD"),
        ("branch", "branch"),
        ("repository_state_sha256", "working-tree/index state"),
    ):
        if str(stored.get(field) or "") != str(live.get(field) or ""):
            errors.append(f"Session repository {label} changed since its last authoritative checkpoint")
    return errors


def _bind_next_command_preconditions(
    payload: dict[str, Any],
    *,
    work_record: Mapping[str, Any],
    bootstrap_record: Mapping[str, Any],
    repository: Mapping[str, Any],
) -> None:
    if not payload.get("next_command"):
        payload["next_command_preconditions"] = None
        return
    payload["next_command_preconditions"] = {
        "schema_version": NEXT_COMMAND_PRECONDITIONS_SCHEMA_VERSION,
        "work_record_sha256": work_record.get("record_sha256"),
        "bootstrap_record_id": bootstrap_record.get("record_id"),
        "package_sha256": dict(payload.get("work_package_binding") or {}).get("sha256"),
        "repository_state_sha256": repository.get("repository_state_sha256"),
        "current_codex_id": payload.get("current_codex_id"),
    }




def _is_provisional_codex_id(value: Any) -> bool:
    text = str(value or "").strip().upper()
    return text.startswith("EIEL-PROVISIONAL-") or text.startswith("PENDING-CODEX-")

def _normalize_payload(raw: Mapping[str, Any], *, session_id: str | None, generated_at: str, allow_prepared: bool = False) -> dict[str, Any]:
    if "schema_version" in raw:
        require_explicit_schema_version(raw, expected=SESSION_SCHEMA_VERSION, label="Codex Session")
    payload = dict(raw)
    payload["schema_version"] = SESSION_SCHEMA_VERSION
    payload["work_id"] = validate_prefixed_id(str(payload.get("work_id") or ""), WORK_ID_PREFIX, label="work item")
    payload["mission_id"] = str(payload.get("mission_id") or "").strip().upper() or None
    payload["title"] = str(payload.get("title") or "").strip() or None
    payload["bootstrap_record_id"] = validate_prefixed_id(str(payload.get("bootstrap_record_id") or ""), BOOTSTRAP_ID_PREFIX, label="bootstrap")
    payload["bootstrap_history"] = normalize_string_list(
        payload.get("bootstrap_history") or [payload["bootstrap_record_id"]]
    )
    if payload["bootstrap_record_id"] not in payload["bootstrap_history"]:
        payload["bootstrap_history"].append(payload["bootstrap_record_id"])
    original_url = str(payload.get("original_codex_thread_url") or payload.get("current_codex_thread_url") or "").strip() or None
    original_id = str(payload.get("original_codex_id") or payload.get("current_codex_id") or conversation_id_from_url(original_url) or "").strip() or None
    identity_state = str(payload.get("worker_identity_state") or ("PENDING" if allow_prepared and not original_id else "ACTIVE")).strip().upper()
    # A scheduler-prepared Session may not yet have a Codex identity.  The
    # same is true when a worker disappears before opening the Session, or
    # when an unused prepared Session is detached during compilation failure.
    # Those non-active states must still be able to close through the bounded
    # recovery path without fabricating a conversation identifier.
    if not original_id and not (allow_prepared and identity_state in {"PENDING", "LOST", "DETACHED"}):
        raise ActorError("active Codex session requires a Codex conversation or session identifier", code="session_codex_identity_missing")
    payload["worker_identity_state"] = identity_state
    payload["original_codex_id"] = original_id
    payload["original_codex_thread_url"] = original_url
    payload["current_codex_id"] = str(payload.get("current_codex_id") or original_id or "").strip() or None
    payload["current_codex_thread_url"] = str(payload.get("current_codex_thread_url") or original_url or "").strip() or None
    payload["preferred_continuation_codex_id"] = str(payload.get("preferred_continuation_codex_id") or original_id or "").strip() or None
    payload["identity_assurance"] = normalize_identity_assurance(
        payload.get("identity_assurance"), fallback_codex_id=payload.get("current_codex_id")
    )
    participants = payload.get("codex_participants") if isinstance(payload.get("codex_participants"), list) else []
    if not participants and original_id:
        participants = [{
            "codex_id": original_id,
            "thread_url": original_url,
            "joined_at_utc": generated_at,
            "role": "original",
            "reason": "session created",
        }]
    payload["codex_participants"] = participants
    repository = dict(payload.get("repository") or {})
    payload["repository"] = repository
    payload["current_stage"] = str(payload.get("current_stage") or "bootstrap_complete").strip()
    payload["current_step"] = str(payload.get("current_step") or "begin bounded work").strip()
    payload["completed_steps"] = normalize_string_list(payload.get("completed_steps"))
    payload["last_successful_command"] = payload.get("last_successful_command")
    payload["last_failed_command"] = payload.get("last_failed_command")
    payload["validation_events"] = [
        dict(row) for row in (payload.get("validation_events") or []) if isinstance(row, Mapping)
    ]
    payload["validation_gaps"] = normalize_string_list(payload.get("validation_gaps"))
    payload["validation_decision_impact"] = (
        str(payload.get("validation_decision_impact") or "").strip() or None
    )
    payload["resumed_from"] = (
        dict(payload.get("resumed_from") or {})
        if isinstance(payload.get("resumed_from"), Mapping)
        else None
    )
    payload["interrupted_during"] = str(payload.get("interrupted_during") or "").strip() or None
    payload["blocked_reason"] = str(payload.get("blocked_reason") or "").strip() or None
    payload["blocked_by_work_ids"] = normalize_string_list(payload.get("blocked_by_work_ids"))
    payload["next_action"] = str(payload.get("next_action") or "").strip() or None
    payload["next_command"] = str(payload.get("next_command") or "").strip() or None
    payload["next_command_preconditions"] = (
        dict(payload.get("next_command_preconditions") or {})
        if isinstance(payload.get("next_command_preconditions"), Mapping)
        else None
    )
    payload["session_creation_ordinal"] = int(payload.get("session_creation_ordinal") or 1)
    payload["notes"] = [dict(row) if isinstance(row, Mapping) else {"text": str(row)} for row in (payload.get("notes") or [])]
    payload["references"] = normalize_reference_map(payload.get("references"), fields=REFERENCE_FIELDS)
    payload["parallel_session_acknowledged"] = bool(payload.get("parallel_session_acknowledged"))
    payload["session_role"] = str(payload.get("session_role") or "IMPLEMENTATION_AUTHORITY").strip().upper()
    model_assignment = payload.get("model_assignment")
    if isinstance(model_assignment, Mapping):
        model_assignment = dict(model_assignment)
        model_assignment["provider"] = str(model_assignment.get("provider") or "OPENAI").strip().upper()
        model_assignment["model"] = str(model_assignment.get("model") or "").strip() or None
        model_assignment["reasoning_level"] = str(model_assignment.get("reasoning_level") or "").strip().upper() or None
        model_assignment["selected_by"] = str(model_assignment.get("selected_by") or "HUMAN_OPERATOR").strip().upper()
    else:
        model_assignment = None
    payload["model_assignment"] = model_assignment
    payload["work_package_binding"] = normalize_work_package_binding(payload.get("work_package_binding"))
    payload["context_contract_status"] = (
        dict(payload.get("context_contract_status") or {})
        if isinstance(payload.get("context_contract_status"), Mapping)
        else None
    )
    payload["session_outcome"] = str(payload.get("session_outcome") or "").strip().upper() or None
    payload["review_verdict"] = str(payload.get("review_verdict") or "").strip().upper() or None
    payload["promotion_request"] = dict(payload.get("promotion_request") or {}) if isinstance(payload.get("promotion_request"), Mapping) else None
    payload["delivery_evidence"] = dict(payload.get("delivery_evidence") or {}) if isinstance(payload.get("delivery_evidence"), Mapping) else None
    payload["delivery_base_commit"] = str(payload.get("delivery_base_commit") or "").strip() or None
    payload["review_target"] = dict(payload.get("review_target") or {}) if isinstance(payload.get("review_target"), Mapping) else None
    payload["review_evidence"] = dict(payload.get("review_evidence") or {}) if isinstance(payload.get("review_evidence"), Mapping) else None
    payload["authoritative_review_binding"] = (
        dict(payload.get("authoritative_review_binding") or {})
        if isinstance(payload.get("authoritative_review_binding"), Mapping)
        else None
    )
    identity = {
        "work_id": payload["work_id"],
        "original_codex_id": original_id,
        "created_at": generated_at,
        "creation_ordinal": payload.get("session_creation_ordinal"),
    }
    resolved = validate_prefixed_id(session_id, SESSION_ID_PREFIX, label="session") if session_id else stable_id("SES", identity)
    payload["session_id"] = resolved
    payload["epoch"] = epoch_for_session(
        payload.get("epoch") if isinstance(payload.get("epoch"), Mapping) else None,
        session_id=resolved,
        authority_role=payload["session_role"],
        generated_at=generated_at,
    )
    payload["epoch_history"] = [
        dict(row) for row in (payload.get("epoch_history") or []) if isinstance(row, Mapping)
    ]
    payload["context_health"] = normalize_observed_context_health(
        payload.get("context_health") if isinstance(payload.get("context_health"), Mapping) else None
    )
    payload["context_receipt"] = (
        dict(payload.get("context_receipt") or {})
        if isinstance(payload.get("context_receipt"), Mapping)
        else None
    )
    payload["prepared_assignment_id"] = str(payload.get("prepared_assignment_id") or "").strip().upper() or None
    payload["scheduler_assignment_id"] = str(payload.get("scheduler_assignment_id") or "").strip().upper() or None
    payload["scheduler_assignment_digest"] = str(payload.get("scheduler_assignment_digest") or "").strip() or None
    payload["scheduler_lease_id"] = str(payload.get("scheduler_lease_id") or "").strip().upper() or None
    payload["scheduler_worker_id"] = str(payload.get("scheduler_worker_id") or "").strip() or None
    payload["assignment_history"] = [dict(row) for row in (payload.get("assignment_history") or []) if isinstance(row, Mapping)]
    payload["worker_loss"] = dict(payload.get("worker_loss") or {}) if isinstance(payload.get("worker_loss"), Mapping) else None
    payload["resource_accounting"] = normalize_session_resource_accounting(
        payload.get("resource_accounting") if isinstance(payload.get("resource_accounting"), Mapping) else None
    )
    payload["attempt_disposition"] = (
        load_attempt_disposition(payload.get("attempt_disposition"))
        if isinstance(payload.get("attempt_disposition"), Mapping)
        else None
    )
    payload["artifact_dispositions"] = [
        load_artifact_disposition(row)
        for row in (payload.get("artifact_dispositions") or [])
        if isinstance(row, Mapping)
    ]
    payload["review_progress"] = (
        load_review_progress(payload.get("review_progress"))
        if isinstance(payload.get("review_progress"), Mapping)
        else None
    )
    payload["repair_lineage"] = (
        load_repair_lineage(payload.get("repair_lineage"))
        if isinstance(payload.get("repair_lineage"), Mapping)
        else None
    )
    payload["terminal_transition"] = (
        dict(payload.get("terminal_transition") or {})
        if isinstance(payload.get("terminal_transition"), Mapping)
        else None
    )
    return payload


def _current_authoritative_review_binding(
    work_record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    read_only: bool = False,
) -> dict[str, Any] | None:
    # Local import keeps the Session manager from acquiring Work-manager
    # ownership while still sharing its authoritative review-lineage decision.
    from tools.actors.work_manager import authoritative_review_binding_for_work

    return authoritative_review_binding_for_work(
        work_record,
        state_root=state_root,
        read_only=read_only,
    )


def _validate_payload(
    payload: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    record_status: str | None = None,
    verify_bootstrap: bool = True,
    allow_terminal_work: bool = False,
    allow_historical_package: bool = False,
    read_only: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if payload.get("schema_version") != SESSION_SCHEMA_VERSION:
        errors.append("unsupported session schema_version")
    if payload.get("resource_accounting") is None:
        warnings.append("legacy Session record has no WRA-1 accounting projection")
    else:
        errors.extend(validate_session_resource_accounting(payload.get("resource_accounting")))
    if payload.get("attempt_disposition") is not None:
        try:
            load_attempt_disposition(payload.get("attempt_disposition"))
        except ActorError as exc:
            errors.append(f"invalid attempt_disposition: {exc}")
    for index, row in enumerate(payload.get("artifact_dispositions") or []):
        try:
            load_artifact_disposition(row)
        except ActorError as exc:
            errors.append(f"invalid artifact_dispositions[{index}]: {exc}")
    if payload.get("review_progress") is not None:
        try:
            load_review_progress(payload.get("review_progress"))
        except ActorError as exc:
            errors.append(f"invalid review_progress: {exc}")
    if payload.get("repair_lineage") is not None:
        try:
            load_repair_lineage(payload.get("repair_lineage"))
        except ActorError as exc:
            errors.append(f"invalid repair_lineage: {exc}")
    try:
        work_store_obj = work_store(state_root)
        work_getter = work_store_obj.get_read_only if read_only else work_store_obj.get
        work = work_getter(str(payload.get("work_id") or ""))
    except ActorError as exc:
        errors.append(str(exc))
        work = None
    if work and work.get("status") in WORK_TERMINAL_STATUSES and not allow_terminal_work:
        errors.append(f"work item is terminal: {work.get('status')}")
    mission_id = payload.get("mission_id")
    if mission_id and not str(mission_id).startswith("MIS-"):
        errors.append("mission_id must start with MIS-")

    validation_events = [
        dict(row) for row in (payload.get("validation_events") or []) if isinstance(row, Mapping)
    ]
    for index, event in enumerate(validation_events):
        event_status = str(event.get("status") or "").strip().upper()
        if event_status not in {"PASSED", "FAILED", "PARTIAL", "SKIPPED"}:
            errors.append(f"validation_events[{index}] has unsupported status")
        if not str(event.get("command") or event.get("check") or "").strip():
            errors.append(f"validation_events[{index}] requires command or check")
        if event_status == "PARTIAL":
            gaps = normalize_string_list(event.get("gaps"))
            impact = str(event.get("decision_impact") or "").strip()
            if not gaps or not impact:
                errors.append(
                    f"validation_events[{index}] PARTIAL requires nonempty gaps and decision_impact"
                )
    validation_gaps = normalize_string_list(payload.get("validation_gaps"))
    validation_impact = str(payload.get("validation_decision_impact") or "").strip()
    if validation_gaps and not validation_impact:
        errors.append("validation_gaps require validation_decision_impact")
    if validation_impact and not validation_gaps:
        errors.append("validation_decision_impact requires validation_gaps")

    resumed_from = payload.get("resumed_from")
    if resumed_from is not None:
        if not isinstance(resumed_from, Mapping):
            errors.append("resumed_from must be an object")
        else:
            resumed = dict(resumed_from)
            if resumed.get("schema_version") != "eiel_session_resume_cause_v1":
                errors.append("resumed_from has unsupported schema_version")
            session_ref = resumed.get("session")
            if not isinstance(session_ref, Mapping):
                errors.append("resumed_from requires exact Session reference")
            else:
                if str(session_ref.get("record_id") or "") != str(payload.get("session_id") or ""):
                    errors.append("resumed_from Session identity differs from current Session")
                if not str(session_ref.get("record_sha256") or ""):
                    errors.append("resumed_from Session reference requires record_sha256")
            if not str(resumed.get("context_receipt_id") or ""):
                errors.append("resumed_from requires context_receipt_id")
            if not str(resumed.get("context_receipt_sha256") or ""):
                errors.append("resumed_from requires context_receipt_sha256")

    epoch_history = [
        dict(row) for row in (payload.get("epoch_history") or []) if isinstance(row, Mapping)
    ]
    epoch_ids = [str(row.get("epoch_id") or "") for row in epoch_history if str(row.get("epoch_id") or "")]
    if len(epoch_ids) != len(set(epoch_ids)):
        errors.append("epoch_history repeats an epoch_id")
    current_epoch_id = str(dict(payload.get("epoch") or {}).get("epoch_id") or "")
    if current_epoch_id and current_epoch_id in set(epoch_ids):
        errors.append("epoch_history must not repeat the current epoch")
    identity_state = str(payload.get("worker_identity_state") or "ACTIVE")
    identity_pending = identity_state == "PENDING"
    identity_detached = identity_state == "DETACHED"
    identity_lost_before_activation = (
        identity_state == "LOST"
        and not str(payload.get("original_codex_id") or "").strip()
        and not str(payload.get("current_codex_id") or "").strip()
    )
    if not identity_pending and not identity_detached and not identity_lost_before_activation and not str(payload.get("current_codex_id") or "").strip():
        errors.append("current_codex_id is required")
    if not identity_pending and not identity_detached and not identity_lost_before_activation and not str(payload.get("original_codex_id") or "").strip():
        errors.append("original_codex_id is required")
    if identity_pending:
        if not str(payload.get("prepared_assignment_id") or "").startswith("ASSIGN-"):
            errors.append("prepared Session requires prepared_assignment_id")
        if payload.get("codex_participants"):
            errors.append("prepared Session must not fabricate Codex participants")
    if identity_lost_before_activation:
        if payload.get("codex_participants"):
            errors.append("pre-activation-lost Session must not fabricate Codex participants")
    if identity_detached:
        if payload.get("prepared_assignment_id"):
            errors.append("detached Session must not retain prepared_assignment_id")
        if payload.get("scheduler_assignment_id") or payload.get("scheduler_lease_id") or payload.get("scheduler_worker_id"):
            errors.append("detached Session must not retain scheduler ownership")
        if payload.get("codex_participants"):
            errors.append("detached Session must not fabricate Codex participants")
    if str(payload.get("session_role") or "IMPLEMENTATION_AUTHORITY") not in AUTHORITY_ROLES:
        errors.append("session_role must be one of " + ", ".join(AUTHORITY_ROLES))
    epoch = dict(payload.get("epoch") or {})
    expected_phase = {
        "ARCHITECTURE_AUTHORITY": "architecture",
        "ENGINEERING_AUTHORITY": "engineering",
        "IMPLEMENTATION_AUTHORITY": "implementation",
        "REVIEW_AUTHORITY": "review",
    }.get(str(payload.get("session_role") or "IMPLEMENTATION_AUTHORITY"))
    if epoch and epoch.get("authority_phase") != expected_phase:
        errors.append("session epoch authority_phase does not match session_role")
    try:
        normalize_observed_context_health(
            payload.get("context_health") if isinstance(payload.get("context_health"), Mapping) else None
        )
    except ActorError as exc:
        errors.append(str(exc))
    outcome = payload.get("session_outcome")
    if outcome and outcome not in SESSION_OUTCOMES:
        errors.append("session_outcome must be one of " + ", ".join(SESSION_OUTCOMES))
    verdict = payload.get("review_verdict")
    if verdict and verdict not in REVIEW_VERDICTS:
        errors.append("review_verdict must be one of " + ", ".join(REVIEW_VERDICTS))
    if verdict and payload.get("session_role") != "REVIEW_AUTHORITY":
        errors.append("review_verdict is valid only for REVIEW_AUTHORITY sessions")
    if outcome == "REVIEW_COMPLETE" and payload.get("session_role") != "REVIEW_AUTHORITY":
        errors.append("REVIEW_COMPLETE requires session_role REVIEW_AUTHORITY")
    if payload.get("session_role") == "REVIEW_AUTHORITY" and outcome == "REVIEW_COMPLETE" and not verdict:
        errors.append("completed Review Authority session requires review_verdict")

    # Composite Session lifecycle invariants. Status is an envelope field, but
    # outcomes/verdicts/promotion evidence live in the payload; validating them
    # independently allowed impossible Cartesian combinations to persist.
    status = str(record_status or "").strip().upper() or None
    promotion_request = payload.get("promotion_request")
    has_promotion_request = isinstance(promotion_request, Mapping) and bool(promotion_request)
    terminal_outcomes = set(DELIVERY_OUTCOMES) | {"REVIEW_COMPLETE", "SESSION_COMPLETE"}
    if status is not None and outcome in terminal_outcomes and status != "COMPLETED":
        errors.append(f"session_outcome {outcome} requires Session status COMPLETED")
    if outcome == "PROMOTION_REQUESTED" and not has_promotion_request:
        errors.append("PROMOTION_REQUESTED requires structured promotion_request")
    if outcome != "PROMOTION_REQUESTED" and has_promotion_request:
        errors.append("promotion_request is valid only with session_outcome PROMOTION_REQUESTED")
    if payload.get("session_role") == "REVIEW_AUTHORITY":
        if verdict and outcome != "REVIEW_COMPLETE":
            errors.append("review_verdict requires session_outcome REVIEW_COMPLETE")
        if has_promotion_request:
            errors.append("Review Authority Session may not carry implementation promotion_request")
    elif verdict:
        errors.append("non-review Session may not carry review_verdict")

    if record_status in SESSION_TERMINAL_STATUSES and not isinstance(payload.get("attempt_disposition"), Mapping):
        warnings.append("legacy terminal Session has no attempt_disposition")
    if record_status not in SESSION_TERMINAL_STATUSES and payload.get("attempt_disposition") is not None:
        errors.append("non-terminal Session may not carry attempt_disposition")
    if payload.get("session_role") == "REVIEW_AUTHORITY":
        progress = dict(payload.get("review_progress") or {})
        if record_status == "COMPLETED" and progress and progress.get("state") != "COMPLETED":
            errors.append("completed Review Session review_progress must be COMPLETED")
        if record_status in {"PAUSED", "BLOCKED"} and progress.get("state") == "DEFERRED":
            if not progress.get("defer_reason") or not progress.get("resume_condition"):
                errors.append("deferred Review requires reason and resume condition")
    elif payload.get("review_progress") is not None or payload.get("repair_lineage") is not None:
        errors.append("review_progress and repair_lineage are valid only for Review Authority Sessions")

    package_binding = payload.get("work_package_binding")
    binding_kind = str(dict(package_binding or {}).get("sha256_kind") or "raw_file_sha256_v1")
    legacy_v17_binding = bool(package_binding) and binding_kind == "raw_file_sha256_v1"
    if outcome in DELIVERY_OUTCOMES and not isinstance(payload.get("delivery_evidence"), Mapping):
        (warnings if legacy_v17_binding else errors).append(
            "legacy V1.7 delivery session has no exact delivery_evidence"
            if legacy_v17_binding
            else "completed delivery session requires delivery_evidence"
        )
    if payload.get("session_role") == "REVIEW_AUTHORITY" and not isinstance(payload.get("review_target"), Mapping):
        (warnings if legacy_v17_binding else errors).append(
            "legacy V1.7 Review Authority session has no explicit review_target"
            if legacy_v17_binding
            else "Review Authority session requires an explicit review_target"
        )
    if outcome == "REVIEW_COMPLETE" and not isinstance(payload.get("review_evidence"), Mapping):
        (warnings if legacy_v17_binding else errors).append(
            "legacy V1.7 review has no structured review_evidence"
            if legacy_v17_binding
            else "completed Review Authority session requires review_evidence"
        )
    elif isinstance(payload.get("review_evidence"), Mapping):
        try:
            load_review_evidence(
                payload.get("review_evidence"),
                allow_legacy=legacy_v17_binding,
            )
        except ActorError as exc:
            target = warnings if legacy_v17_binding else errors
            target.append(f"invalid Review evidence: {exc}")
            target.extend(str(problem) for problem in (exc.problems or []))
    if work:
        work_payload = dict(work.get("payload") or {})
        expected_binding = dict(work_payload.get("work_package_binding") or {})
        expected_chain = normalize_work_package_chain(
            work_payload.get("work_package_chain"), fallback=expected_binding
        )
        expected_digests = {
            str(dict(row["binding"]).get("sha256") or "") for row in expected_chain
        }
        if expected_binding:
            if not package_binding:
                if not allow_historical_package:
                    errors.append("session is missing the Work Item's Work Package binding")
            else:
                repository_root = dict(payload.get("repository") or {}).get("repo_root")
                package = verify_work_package_binding(
                    package_binding,
                    repo_root=repository_root,
                    expected_work_id=str(payload.get("work_id") or ""),
                    verification_mode=(
                        VERIFICATION_HISTORICAL_AUDIT
                        if allow_historical_package or read_only
                        else VERIFICATION_EXECUTION_STRICT
                    ),
                    state_root=state_root,
                )
                errors.extend(package["errors"])
                warnings.extend(package["warnings"])
                if (
                    str(dict(package_binding).get("sha256") or "") not in expected_digests
                    and not allow_historical_package
                ):
                    errors.append("session Work Package digest does not match a Work Item package-chain binding")
        elif package_binding and not allow_historical_package:
            errors.append("session has a Work Package binding but its Work Item does not")

        retained_review_binding = (
            dict(payload.get("authoritative_review_binding") or {})
            if isinstance(payload.get("authoritative_review_binding"), Mapping)
            else None
        )
        if retained_review_binding:
            if retained_review_binding.get("schema_version") != AUTHORITATIVE_REVIEW_BINDING_SCHEMA_VERSION:
                errors.append("unsupported authoritative_review_binding schema_version")
            binding_without_digest = dict(retained_review_binding)
            observed_binding_digest = str(binding_without_digest.pop("binding_sha256", "") or "")
            expected_binding_digest = canonical_sha256(binding_without_digest)
            if not observed_binding_digest or observed_binding_digest != expected_binding_digest:
                errors.append("authoritative_review_binding digest mismatch")

        current_review_binding = _current_authoritative_review_binding(
            work,
            state_root=state_root,
            read_only=read_only,
        )
        current_verdict = str(dict(current_review_binding or {}).get("verdict") or "")
        open_implementation_repair = (
            status not in SESSION_TERMINAL_STATUSES
            and str(payload.get("session_role") or "") == "IMPLEMENTATION_AUTHORITY"
            and current_verdict in {"CHANGES_REQUIRED", "REJECT", "BLOCKED"}
        )
        if open_implementation_repair and not retained_review_binding:
            errors.append(
                "repair Session requires an exact authoritative_review_binding"
            )
        if retained_review_binding and current_review_binding != retained_review_binding:
            message = "authoritative review changed after this Session was authorized"
            (warnings if allow_historical_package else errors).append(message)
    if not str(payload.get("next_action") or "").strip():
        warnings.append("session has no exact next action")
    if verify_bootstrap:
        try:
            bootstrap_store_obj = bootstrap_store(state_root)
            bootstrap_getter = bootstrap_store_obj.get_read_only if read_only else bootstrap_store_obj.get
            bootstrap = bootstrap_getter(str(payload.get("bootstrap_record_id") or ""))
        except ActorError as exc:
            errors.append(str(exc))
        else:
            expected_root = dict(payload.get("repository") or {}).get("repo_root")
            expected_head = dict(payload.get("repository") or {}).get("head")
            errors.extend(
                validate_bootstrap_record(
                    bootstrap,
                    expected_repo_root=expected_root,
                    expected_head=str(expected_head) if expected_head else None,
                )
            )
    return {"valid": not errors, "errors": errors, "warnings": warnings}


def _work_readiness(state_root: str | Path | None, work_id: str) -> dict[str, Any]:
    work = work_store(state_root).get(work_id)
    payload = dict(work.get("payload") or {})
    dependencies = dependency_state(state_root, payload.get("dependency_work_ids") or [])
    return {"record": work, "dependencies": dependencies}


def _package_binding_for_session(
    work: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    role: str,
) -> dict[str, Any] | None:
    """Select the deterministic current package without mutating Work."""

    payload = dict(work.get("payload") or {})
    fallback = dict(payload.get("work_package_binding") or {})
    chain = normalize_work_package_chain(payload.get("work_package_chain"), fallback=fallback)
    if not chain:
        return fallback or None
    if len(chain) == 1:
        return dict(chain[0]["binding"])
    # Import lazily: Work Manager remains the owner of derived chain/review
    # state, while Session Manager only asks for the next immutable binding.
    from tools.actors.work_manager import review_state_for_work

    state = review_state_for_work(work, state_root=state_root)
    if role == "REVIEW_AUTHORITY":
        if state.get("intermediate_review_required"):
            return dict(state.get("current_package_binding") or {})
        if state.get("all_packages_delivered"):
            return dict(chain[-1]["binding"])
        raise ActorError(
            "final Review Authority Session cannot start before every Work Package is delivered",
            code="review_chain_incomplete",
            next_actions=[{"action": "start_next_package", "description": "Complete the next eligible implementation package first."}],
        )
    binding = state.get("next_package_binding")
    if binding:
        return dict(binding)
    if state.get("all_packages_delivered") or state.get("intermediate_review_required"):
        raise ActorError(
            "implementation cannot advance while the package chain is at a review gate",
            code="work_package_review_gate",
            next_actions=[{"action": "start_review", "description": "Complete the required substantive review before continuing; use reduced assurance if no independent reviewer is immediately available."}],
        )
    return dict(chain[0]["binding"])


def _bootstrap_refresh_command(
    bootstrap: Mapping[str, Any],
    *,
    repo_root: str,
    state_root: str | Path | None = None,
    acknowledge_dirty: bool = False,
) -> str:
    return bootstrap_refresh_command(
        bootstrap,
        repo_root=repo_root,
        state_root=state_root,
        acknowledge_dirty=acknowledge_dirty,
    )


def _refresh_or_validate_bootstrap(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    previous_id = str(payload.get("bootstrap_record_id") or "")
    previous = bootstrap_store(args.state_root).get(previous_id)
    repository = dict(payload.get("repository") or {})
    requested_root = Path(args.repo_root or repository.get("repo_root") or Path.cwd()).expanduser().resolve()
    current_identity = git_repository_identity(requested_root)
    bootstrap = previous
    if args.bootstrap_record:
        bootstrap_id = validate_prefixed_id(args.bootstrap_record, BOOTSTRAP_ID_PREFIX, label="bootstrap")
        bootstrap = bootstrap_store(args.state_root).get(bootstrap_id)
    stored_repository = dict(payload.get("repository") or {})
    repository_errors = _repository_matches_session(stored_repository, current_identity)
    # An explicit replacement Bootstrap is a bounded reconciliation request:
    # validate it against the live repository below, then update the Session's
    # recorded repository snapshot. Without one, retain the strict guard so a
    # bare resume cannot silently discard provenance about repository drift.
    if repository_errors and not args.bootstrap_record:
        raise ActorError(
            "session repository state changed since its last authoritative checkpoint",
            code="session_repository_state_drift",
            problems=repository_errors,
            next_actions=[{
                "action": "refresh_bootstrap",
                "description": "Create a current Bootstrap and pass it to resume to reconcile the Session record.",
            }],
            related_records=[str(payload.get("session_id") or "")],
        )
    errors = validate_bootstrap_record(
        bootstrap,
        expected_repo_root=current_identity["root"],
        expected_head=current_identity["head"],
        revalidate_environment=True,
        max_age_hours=None if args.allow_stale_bootstrap else args.max_bootstrap_age_hours,
        now=args.generated_at,
    )
    if errors:
        raise ActorError(
            "session bootstrap must be refreshed before resume",
            code="session_bootstrap_refresh_required",
            problems=errors,
            next_actions=[
                {
                    "action": "refresh_bootstrap",
                    "description": "Run a fresh bootstrap against the current repository and HEAD.",
                    "command": _bootstrap_refresh_command(
                        previous,
                        repo_root=current_identity["root"],
                        state_root=args.state_root,
                        acknowledge_dirty=bool(current_identity.get("dirty")),
                    ),
                },
                {
                    "action": "resume_with_fresh_bootstrap",
                    "description": "Resume this session with the new BST record.",
                    "command": f"python -m tools.eiel sessionManager resume {payload.get('session_id')} --bootstrap-record <BST-ID> --codex-id '<codex-id>'",
                },
            ],
            related_records=[previous_id],
        )
    new_id = str(bootstrap["record_id"])
    history = normalize_string_list(payload.get("bootstrap_history") or [previous_id])
    if new_id not in history:
        history.append(new_id)
    payload["bootstrap_history"] = history
    payload["bootstrap_record_id"] = new_id
    payload["repository"] = _session_repository(current_identity)
    return payload


def _active_session_conflicts(state_root: str | Path | None, work_id: str, *, exclude: str | None = None) -> list[str]:
    return [
        str(row["record_id"])
        for row in sessions_for_work(state_root, work_id)
        if row.get("status") in SESSION_ACTIVE_STATUSES and str(row.get("record_id")) != str(exclude or "")
    ]


def _git(repo_root: str | Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(Path(repo_root).expanduser().resolve()), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _repository_family_id(repo_root: str | Path) -> str:
    """Return a stable identity shared by all worktrees of one repository."""

    root = Path(repo_root).expanduser().resolve()
    result = _git(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if result.returncode == 0 and str(result.stdout or "").strip():
        return str(Path(str(result.stdout).strip()).expanduser().resolve())
    # Older Git versions may not support --path-format.  A fallback is still
    # deterministic and retains the previous same-worktree protection.
    result = _git(root, "rev-parse", "--git-common-dir")
    if result.returncode == 0 and str(result.stdout or "").strip():
        candidate = Path(str(result.stdout).strip()).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        return str(candidate.resolve())
    return str(root)


def _porcelain_paths(output: str) -> list[str]:
    paths: list[str] = []
    for entry in output.split("\x00"):
        if not entry:
            continue
        path = entry[3:] if len(entry) >= 3 and entry[2] == " " else entry
        if path and path not in paths:
            paths.append(path)
    return paths


def _repository_delivery_evidence(
    record: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    delivery_commit: str,
    adopt_existing_commit: bool = False,
    adoption_reason: str | None = None,
    external_producer: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    repository = dict(payload.get("repository") or {})
    repo_root = str(repository.get("repo_root") or "").strip()
    session_base_commit = str(payload.get("delivery_base_commit") or repository.get("head") or "").strip()
    if not repo_root or not session_base_commit:
        raise ActorError("delivery session has no bound repository identity", code="session_repository_missing")
    requested = str(delivery_commit or "").strip()
    if not requested:
        raise ActorError(
            "package-bound delivery completion requires --delivery-commit",
            code="session_delivery_commit_required",
            next_actions=[{
                "action": "name_delivery_commit",
                "description": "Commit the bounded delivery and repeat completion with its exact Git commit.",
                "command": f"python -m tools.eiel sessionManager complete {record.get('record_id')} --delivery-commit $(git rev-parse HEAD)",
            }],
        )
    resolved = _git(repo_root, "rev-parse", "--verify", f"{requested}^{{commit}}")
    if resolved.returncode != 0 or not resolved.stdout.strip():
        raise ActorError(
            "delivery commit is not available in the bound repository",
            code="session_delivery_commit_invalid",
            problems=[resolved.stderr.strip() or requested],
        )
    implementation_commit = resolved.stdout.strip().lower()
    current = git_repository_identity(repo_root)
    if str(current.get("head") or "").lower() != implementation_commit:
        raise ActorError(
            "delivery worktree HEAD must equal the explicit delivery commit",
            code="session_delivery_commit_mismatch",
            problems=[f"Delivery commit: {implementation_commit}", f"Current HEAD: {current.get('head')}"],
        )

    binding = dict(payload.get("work_package_binding") or {})
    verified = verify_work_package_binding(
        binding,
        repo_root=repo_root,
        expected_work_id=str(payload.get("work_id") or ""),
        verification_mode=VERIFICATION_EXECUTION_STRICT,
        state_root=state_root,
    )
    if not verified["valid"] or not isinstance(verified.get("package"), Mapping):
        raise ActorError(
            "delivery Work Package binding is not valid",
            code="work_package_binding_invalid",
            problems=verified["errors"],
            warnings=verified["warnings"],
        )
    package = dict(verified["package"])

    adoption_receipt_ref: dict[str, Any] | None = None
    proof_base_commit = session_base_commit
    delivery_kind = "MANAGED_SESSION_DELIVERY"
    if adopt_existing_commit:
        parents = _git(repo_root, "show", "-s", "--format=%P", implementation_commit)
        parent_values = [value for value in parents.stdout.strip().split() if value] if parents.returncode == 0 else []
        if not parent_values:
            raise ActorError(
                "cannot adopt a root commit without an explicit historical baseline",
                code="repository_adoption_base_missing",
            )
        proof_base_commit = parent_values[0]
        _, adoption_receipt_ref = capture_adoption_receipt(
            state_root=state_root,
            repo_root=repo_root,
            work_id=str(payload.get("work_id") or ""),
            session_id=str(record.get("record_id") or ""),
            adopted_commit=implementation_commit,
            reason=str(adoption_reason or ""),
            external_producer=external_producer,
            generated_at=generated_at,
        )
        delivery_kind = "EXTERNAL_DELIVERY_ADOPTED"
    else:
        ancestry = _git(repo_root, "merge-base", "--is-ancestor", session_base_commit, implementation_commit)
        if ancestry.returncode != 0:
            raise ActorError(
                "delivery commit does not descend from the Session base commit",
                code="session_delivery_commit_not_descendant",
                problems=[f"Base: {session_base_commit}", f"Delivery: {implementation_commit}"],
            )

    status = _git(repo_root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames")
    if status.returncode != 0:
        raise ActorError(
            "cannot inspect delivery worktree status",
            code="session_repository_inspection_failed",
            problems=[status.stderr.strip() or "git status failed"],
        )
    dirty_paths = _porcelain_paths(status.stdout)

    diff = _git(repo_root, "diff", "--name-only", "--no-renames", "-z", f"{proof_base_commit}..{implementation_commit}")
    if diff.returncode != 0:
        raise ActorError(
            "cannot calculate files changed by the delivery",
            code="session_repository_inspection_failed",
            problems=[diff.stderr.strip() or "git diff failed"],
        )
    changed_files = [item for item in diff.stdout.split("\x00") if item]
    if not changed_files and not adopt_existing_commit:
        raise ActorError(
            "a managed delivery Session may not claim an empty repository delivery",
            code="session_empty_delivery_forbidden",
            problems=["Use --adopt-existing-commit with an explicit adoption reason for pre-existing unmanaged work."],
        )

    protected_package_paths: set[str] = set()
    for store_obj in (work_store(state_root), session_store(state_root)):
        for index_row in store_obj.list_read_only():
            try:
                historical = store_obj.get_read_only(str(index_row.get("record_id") or ""))
            except ActorError:
                continue
            historical_binding = dict(dict(historical.get("payload") or {}).get("work_package_binding") or {})
            historical_path = str(historical_binding.get("path") or "").strip()
            if historical_path:
                protected_package_paths.add(historical_path)
    protected_changes = sorted(set(changed_files).intersection(protected_package_paths))
    if protected_changes:
        raise ActorError(
            "delivery changes a current or historical Work Package source document",
            code="work_package_history_path_changed",
            problems=[f"protected Work Package path changed: {value}" for value in protected_changes],
        )

    fence = validate_changed_paths(changed_files, package, repo_root=repo_root)
    if not fence["valid"]:
        raise ActorError(
            "delivery changed files outside the Work Package fence",
            code="work_package_file_fence_violation",
            problems=[
                *(f"outside owned_files: {value}" for value in fence["outside_owned_files"]),
                *(f"forbidden file changed: {value}" for value in fence["forbidden_files_changed"]),
            ],
        )

    dirty_fence = validate_changed_paths(dirty_paths, package, repo_root=repo_root)
    dirty_changed = set(dirty_fence["changed_files"])
    unrelated_dirty = set(dirty_fence["outside_owned_files"])
    forbidden_dirty = set(dirty_fence["forbidden_files_changed"])
    protected_dirty = dirty_changed.intersection(protected_package_paths)
    owned_dirty = dirty_changed - unrelated_dirty
    fatal_dirty = sorted(owned_dirty | forbidden_dirty | protected_dirty)
    if fatal_dirty:
        raise ActorError(
            "package-owned, forbidden, or package-contract paths are dirty",
            code="session_delivery_worktree_dirty",
            problems=[f"dirty delivery path: {value}" for value in fatal_dirty],
        )
    unrelated = sorted(unrelated_dirty - forbidden_dirty - protected_dirty)

    # A second live Session may not silently own the same repository paths.
    ownership_conflicts: list[str] = []
    repository_family = _repository_family_id(repo_root)
    for index_row in session_store(state_root).list_read_only():
        other_id = str(index_row.get("record_id") or "")
        if not other_id or other_id == str(record.get("record_id") or ""):
            continue
        try:
            other = session_store(state_root).get_read_only(other_id)
        except ActorError:
            continue
        if str(other.get("status") or "") not in SESSION_ACTIVE_STATUSES:
            continue
        other_payload = dict(other.get("payload") or {})
        if str(other_payload.get("session_role") or "") == "REVIEW_AUTHORITY":
            continue
        other_root = str(dict(other_payload.get("repository") or {}).get("repo_root") or "")
        if not other_root or _repository_family_id(other_root) != repository_family:
            continue
        other_binding = dict(other_payload.get("work_package_binding") or {})
        other_check = verify_work_package_binding(
            other_binding,
            repo_root=other_root,
            expected_work_id=str(other_payload.get("work_id") or ""),
            verification_mode=VERIFICATION_HISTORICAL_AUDIT,
            state_root=state_root,
        )
        if not isinstance(other_check.get("package"), Mapping):
            continue
        other_fence = validate_changed_paths(changed_files, dict(other_check["package"]), repo_root=other_root)
        overlapping = sorted(set(other_fence["changed_files"]) - set(other_fence["outside_owned_files"]))
        if overlapping:
            ownership_conflicts.append(f"{other_id} owns overlapping paths: {', '.join(overlapping)}")
    if ownership_conflicts:
        raise ActorError(
            "delivery overlaps another live Session authority",
            code="session_delivery_authority_conflict",
            problems=ownership_conflicts,
        )

    package_sha = str(binding.get("sha256") or "")
    duplicates = duplicate_delivery_claims(
        sessions_for_work(state_root, str(payload.get("work_id") or ""), read_only=True),
        current_session_id=str(record.get("record_id") or ""),
        work_id=str(payload.get("work_id") or ""),
        package_sha256=package_sha,
        delivery_commit=implementation_commit,
    )
    if duplicates:
        raise ActorError(
            "delivery commit is already owned by another completed Session",
            code="session_delivery_commit_already_claimed",
            problems=["Claiming Sessions: " + ", ".join(duplicates)],
            related_records=[str(record.get("record_id") or ""), *duplicates],
        )

    _, proof_ref = capture_delivery_proof(
        state_root=state_root,
        repo_root=repo_root,
        work_id=str(payload.get("work_id") or ""),
        session_id=str(record.get("record_id") or ""),
        package_sha256=package_sha,
        base_commit=proof_base_commit,
        delivery_commit=implementation_commit,
        changed_files=fence["changed_files"],
        delivery_kind=delivery_kind,
        adopted_from=adoption_receipt_ref,
        generated_at=generated_at,
    )

    return {
        "schema_version": SESSION_DELIVERY_EVIDENCE_SCHEMA_VERSION,
        "delivery_session_id": str(record.get("record_id") or ""),
        "base_commit": proof_base_commit,
        "session_base_commit": session_base_commit,
        "delivery_commit": implementation_commit,
        "branch": current.get("branch"),
        "repo_root": repo_root,
        "work_package_sha256": package_sha,
        "work_package_version": binding.get("package_version"),
        "changed_files": fence["changed_files"],
        "worktree_clean": not dirty_paths,
        "unrelated_dirty_paths": unrelated,
        "delivery_kind": delivery_kind,
        "delivery_sufficiency": "UNASSESSED",
        "delivery_proof": proof_ref,
        "adoption_receipt": adoption_receipt_ref,
        "recorded_at_utc": None,
    }


def _completed_delivery_sessions(
    state_root: str | Path | None,
    work_id: str,
    *,
    package_sha256: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in sessions_for_work(state_root, work_id):
        payload = dict(row.get("payload") or {})
        if row.get("status") != "COMPLETED":
            continue
        if payload.get("session_outcome") not in DELIVERY_OUTCOMES:
            continue
        if payload.get("session_role") == "REVIEW_AUTHORITY":
            continue
        if package_sha256:
            observed = str(dict(payload.get("work_package_binding") or {}).get("sha256") or "")
            if observed != package_sha256:
                continue
        if not isinstance(payload.get("delivery_evidence"), Mapping):
            continue
        rows.append(dict(row))
    return list(resolve_delivery_lineage(rows, state_root=state_root)["ordered"])


def _select_review_target(
    *,
    state_root: str | Path | None,
    work_id: str,
    package_sha256: str,
    requested_session_id: str | None,
) -> dict[str, Any]:
    deliveries = _completed_delivery_sessions(state_root, work_id, package_sha256=package_sha256)
    if not deliveries:
        raise ActorError(
            "review cannot start before a package delivery session is complete",
            code="review_delivery_missing",
            related_records=[work_id],
        )
    lineage = resolve_delivery_lineage(deliveries, state_root=state_root)
    if lineage["conflict"]:
        raise ActorError(
            "review cannot start while completed delivery commits diverge or cannot be verified",
            code="review_delivery_lineage_conflict",
            problems=list(lineage.get("problems") or []),
            related_records=[work_id, *list(lineage.get("conflicting_session_ids") or [])],
        )
    authoritative = lineage.get("authoritative")
    if not authoritative:
        raise ActorError(
            "review cannot resolve one authoritative delivery",
            code="review_delivery_missing",
            related_records=[work_id],
        )
    if requested_session_id:
        requested = str(requested_session_id).strip().upper()
        if str(authoritative.get("record_id") or "") != requested:
            raise ActorError(
                "the requested review target is not the authoritative delivery for this Work Package",
                code="review_target_invalid",
                problems=[f"Authoritative delivery: {authoritative.get('record_id')}"],
                related_records=[work_id, requested, str(authoritative.get("record_id") or "")],
            )
        delivery = authoritative
    else:
        delivery = authoritative
    delivery_payload = dict(delivery.get("payload") or {})
    evidence = dict(delivery_payload.get("delivery_evidence") or {})
    integrated_lineage: list[dict[str, Any]] = []
    try:
        from tools.actors.work_manager import review_state_for_work

        work_record = work_store(state_root).get_read_only(work_id)
        chain_state = review_state_for_work(work_record, state_root=state_root)
        for entry in chain_state.get("package_chain") or []:
            package_binding = dict(entry.get("binding") or {})
            delivery_id = str(entry.get("delivery_session_id") or "")
            delivery_record = session_store(state_root).get_read_only(delivery_id) if delivery_id else None
            delivery_evidence = dict(dict(delivery_record or {}).get("payload") or {}).get("delivery_evidence") or {}
            integrated_lineage.append({
                "sequence": entry.get("sequence"),
                "package_path": package_binding.get("path"),
                "package_version": package_binding.get("package_version"),
                "package_sha256": package_binding.get("sha256"),
                "implementation_session_id": delivery_id or None,
                "delivery_commit": delivery_commit_value(delivery_evidence),
                "delivery_record": actor_record_ref(delivery_record) if delivery_record else None,
            })
    except ActorError:
        integrated_lineage = []
    delivery_material = {
        "schema_version": DELIVERY_MATERIAL_REF_SCHEMA_VERSION,
        "delivery_session_id": str(delivery.get("record_id") or ""),
        "terminal_outcome": str(delivery_payload.get("session_outcome") or ""),
        "delivery_evidence_sha256": canonical_sha256(evidence),
        "delivery_commit": delivery_commit_value(evidence),
        "work_package_sha256": str(evidence.get("work_package_sha256") or package_sha256),
        "work_package_version": evidence.get("work_package_version"),
        "identity_assurance": normalize_identity_assurance(
            delivery_payload.get("identity_assurance"),
            fallback_codex_id=delivery_payload.get("current_codex_id"),
        ),
    }
    return {
        "schema_version": REVIEW_TARGET_SCHEMA_VERSION,
        "delivery_session_id": str(delivery.get("record_id") or ""),
        "delivery_material": delivery_material,
        "delivery_evidence_sha256": canonical_sha256(evidence),
        "delivering_codex_identities": sorted(codex_identities(delivery)),
        "delivery_identity_assurance": normalize_identity_assurance(
            delivery_payload.get("identity_assurance"),
            fallback_codex_id=delivery_payload.get("current_codex_id"),
        ),
        "reviewed_commit": delivery_commit_value(evidence),
        "reviewed_package_sha256": str(evidence.get("work_package_sha256") or package_sha256),
        "reviewed_package_version": evidence.get("work_package_version"),
        "review_scope": "FINAL_INTEGRATED" if integrated_lineage and str(delivery.get("record_id") or "") == str(integrated_lineage[-1].get("implementation_session_id") or "") else "INTERMEDIATE_PACKAGE",
        "integrated_delivery_lineage": integrated_lineage,
    }


def _ensure_review_assurance(
    *,
    state_root: str | Path | None,
    work_id: str,
    review_record: Mapping[str, Any],
    review_target: Mapping[str, Any],
) -> dict[str, Any]:
    review_target = canonicalize_review_target(review_target)
    target_id = delivery_session_id(review_target)
    if not target_id:
        raise ActorError("review target has no delivery Session", code="review_target_invalid")
    try:
        implementation = session_store(state_root).get(target_id)
    except ActorError as exc:
        raise ActorError(
            "review target implementation session is missing",
            code="review_target_invalid",
            problems=[str(exc)],
            related_records=[work_id, target_id],
        ) from exc
    implementation_payload = dict(implementation.get("payload") or {})
    delivery = dict(implementation_payload.get("delivery_evidence") or {})
    target_schema = str(review_target.get("schema_version") or "")
    if target_schema not in {REVIEW_TARGET_SCHEMA_VERSION, REVIEW_TARGET_LEGACY_SCHEMA_VERSION}:
        raise ActorError("unsupported review target schema_version", code="schema_version_unsupported")
    delivery_material = dict(review_target.get("delivery_material") or {})
    if delivery_material:
        require_explicit_schema_version(
            delivery_material,
            expected=DELIVERY_MATERIAL_REF_SCHEMA_VERSION,
            supported=(DELIVERY_MATERIAL_REF_LEGACY_SCHEMA_VERSION,),
            label="review delivery material",
        )
        material_session_field = (
            "delivery_session_id"
            if str(delivery_material.get("schema_version") or "") == DELIVERY_MATERIAL_REF_SCHEMA_VERSION
            else "implementation_session_id"
        )
        material_commit_field = (
            "delivery_commit"
            if str(delivery_material.get("schema_version") or "") == DELIVERY_MATERIAL_REF_SCHEMA_VERSION
            else "implementation_commit"
        )
        expected_material = {
            material_session_field: str(implementation.get("record_id") or ""),
            "terminal_outcome": str(implementation_payload.get("session_outcome") or ""),
            "delivery_evidence_sha256": canonical_sha256(delivery),
            material_commit_field: delivery_commit_value(delivery),
            "work_package_sha256": str(delivery.get("work_package_sha256") or ""),
            "work_package_version": delivery.get("work_package_version"),
        }
        for field, expected_value in expected_material.items():
            if delivery_material.get(field) != expected_value:
                raise ActorError(
                    f"review target delivery material changed: {field}",
                    code="lifecycle_decision_stale",
                    related_records=[work_id, target_id],
                )
    else:
        target_record_ref = dict(review_target.get("implementation_record") or {})
        if target_record_ref:
            require_record_ref(target_record_ref, implementation, label="review target implementation Session")
    target_delivery_digest = str(review_target.get("delivery_evidence_sha256") or "")
    if target_delivery_digest and target_delivery_digest != canonical_sha256(delivery):
        raise ActorError(
            "review target delivery evidence changed after review start",
            code="lifecycle_decision_stale",
            related_records=[work_id, target_id],
        )
    if implementation.get("status") != "COMPLETED" or implementation_payload.get("session_outcome") not in DELIVERY_OUTCOMES:
        raise ActorError("review target is not a completed delivery session", code="review_target_invalid")
    expected_commit = delivery_commit_value(delivery)
    expected_package = str(delivery.get("work_package_sha256") or "")
    if str(review_target.get("reviewed_commit") or "") != expected_commit:
        raise ActorError("review target commit does not match delivery evidence", code="review_target_invalid")
    if str(review_target.get("reviewed_package_sha256") or "") != expected_package:
        raise ActorError("review target package digest does not match delivery evidence", code="review_target_invalid")
    implementation_ids = {value for value in codex_identities(implementation) if not _is_provisional_codex_id(value)}
    review_ids = {value for value in codex_identities(review_record) if not _is_provisional_codex_id(value)}
    overlap = sorted(implementation_ids.intersection(review_ids))
    # Operational continuity permits a reduced-assurance review whenever an
    # independent identity is unavailable. The terminal review evidence records
    # that limitation explicitly; reviewer identity never blocks acceptance.
    return implementation


def _active_command_authorization_problems(
    record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
) -> list[str]:
    payload = dict(record.get("payload") or {})
    repository = dict(payload.get("repository") or {})
    root = repository.get("repo_root")
    if not root:
        return ["Session repository binding is incomplete"]
    try:
        live_repository = git_repository_identity(root)
    except ActorError as exc:
        return [str(exc)]
    problems = _repository_matches_session(repository, live_repository)
    preconditions = dict(payload.get("next_command_preconditions") or {})
    if not payload.get("next_command"):
        return problems
    if preconditions.get("schema_version") != NEXT_COMMAND_PRECONDITIONS_SCHEMA_VERSION:
        problems.append("Session next command has no current authorization preconditions")
        return problems
    try:
        work = work_store(state_root).get_read_only(str(payload.get("work_id") or ""))
        bootstrap = bootstrap_store(state_root).get_read_only(str(payload.get("bootstrap_record_id") or ""))
    except ActorError as exc:
        problems.append(str(exc))
        return problems
    problems.extend(
        validate_bootstrap_record(
            bootstrap,
            expected_repo_root=live_repository.get("root"),
            expected_head=str(live_repository.get("head") or "") or None,
            revalidate_environment=True,
        )
    )
    expected = {
        "work_record_sha256": work.get("record_sha256"),
        "bootstrap_record_id": bootstrap.get("record_id"),
        "package_sha256": dict(payload.get("work_package_binding") or {}).get("sha256"),
        "repository_state_sha256": live_repository.get("repository_state_sha256"),
        "current_codex_id": payload.get("current_codex_id"),
    }
    for field, value in expected.items():
        if preconditions.get(field) != value:
            problems.append(f"Session next command precondition is stale: {field}")
    return problems


def _summary(
    record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    authorize_live: bool = False,
) -> dict[str, Any]:
    payload = dict(record.get("payload") or {})
    status = str(record.get("status") or "CREATED")
    validation = _validate_payload(
        payload,
        state_root=state_root,
        record_status=status,
        verify_bootstrap=status not in SESSION_TERMINAL_STATUSES,
        allow_terminal_work=status in SESSION_TERMINAL_STATUSES,
        allow_historical_package=status in SESSION_TERMINAL_STATUSES,
        read_only=True,
    )
    authorization_problems = (
        _active_command_authorization_problems(record, state_root=state_root)
        if status in SESSION_RESUMABLE_STATUSES and authorize_live
        else []
    )
    if authorization_problems:
        validation["errors"] = list(validation.get("errors") or []) + authorization_problems
        validation["valid"] = False
    next_actions: list[dict[str, Any]] = []
    outcome = payload.get("session_outcome")
    verdict = payload.get("review_verdict")
    if outcome == "PROMOTION_REQUESTED":
        promotion_request = dict(payload.get("promotion_request") or {})
        next_actions.append(
            {
                "action": "return_to_engineering_authority",
                "description": "Do not resume this implementation attempt. Engineering Authority must resolve the promotion and issue a revised or superseding Work Package.",
                "command": f"python -m tools.eiel workManager inspect {payload.get('work_id')}",
                "promotion_request_ref": {
                    "session_id": record.get("record_id"),
                    "revision": record.get("revision"),
                    "record_sha256": record.get("record_sha256"),
                    "requested_uncertainty_class": promotion_request.get("requested_uncertainty_class"),
                    "recommended_authority": promotion_request.get("recommended_authority"),
                    "promotion_request_sha256": canonical_sha256(promotion_request),
                },
            }
        )
    elif status in SESSION_TERMINAL_STATUSES and outcome in DELIVERY_OUTCOMES:
        next_actions.append(
            {
                "action": "start_review",
                "description": "Start a Review Authority session for this Work Package; use an independent identity when immediately available, otherwise record reduced assurance.",
                "command": f"python -m tools.eiel sessionManager start {payload.get('work_id')} --role REVIEW_AUTHORITY --review-target-session {record.get('record_id')} --codex-id '<codex-id>' --bootstrap-record <BST-ID>",
            }
        )
    elif status in SESSION_TERMINAL_STATUSES and outcome == "REVIEW_COMPLETE" and verdict == "ACCEPT":
        next_actions.append(
            {
                "action": "accept_work",
                "description": "Complete the Work Item through the accepted evidence-entailing review.",
                "command": f"python -m tools.eiel workManager complete {payload.get('work_id')} --review-session {record.get('record_id')}",
            }
        )
    elif status in SESSION_TERMINAL_STATUSES and outcome == "REVIEW_COMPLETE" and verdict in {"CHANGES_REQUIRED", "REJECT", "BLOCKED"}:
        next_actions.append(
            {
                "action": "return_for_repair",
                "description": "Inspect the review findings and start a bounded repair or replacement implementation session.",
                "command": f"python -m tools.eiel workManager inspect {payload.get('work_id')}",
            }
        )
    elif status in SESSION_TERMINAL_STATUSES and outcome == "REVIEW_COMPLETE" and verdict == "PROMOTE":
        next_actions.append(
            {
                "action": "return_to_engineering_authority",
                "description": "The review identified higher uncertainty. Engineering Authority must resolve or supersede the Work Package.",
                "command": f"python -m tools.eiel workManager inspect {payload.get('work_id')}",
            }
        )
    elif status in SESSION_TERMINAL_STATUSES:
        next_actions.append({"action": "none", "description": f"Session is {status.lower()} and must not be resumed."})
    elif status in {"PAUSED", "INTERRUPTED", "BLOCKED", "CREATED"}:
        if payload.get("worker_identity_state") == "PENDING" and payload.get("prepared_assignment_id"):
            next_actions.extend(
                [
                    {
                        "action": "scheduler_assignment_owns_session",
                        "description": "This Session is prepared for a Scheduler Assignment and is not an ordinary resumable Session.",
                        "command": "python -m tools.eiel assignment status",
                    },
                    {
                        "action": "manual_takeover",
                        "description": "Explicitly supersede the unstarted Scheduler attempt and resume manually.",
                        "command": (
                            f"python -m tools.eiel sessionManager resume {record['record_id']} "
                            "--take-over-prepared --takeover-reason '<reason>' --codex-id '<codex-id>'"
                        ),
                    },
                ]
            )
        elif authorization_problems:
            next_actions.append(
                {
                    "action": "refresh_bootstrap",
                    "description": "Refresh the repository/Bootstrap binding before resuming this Session.",
                    "command": f"python -m tools.eiel sessionManager inspect {record['record_id']}",
                }
            )
        else:
            next_actions.append({"action": "resume", "description": "Resume this exact session with the original Codex or an explicit takeover.", "command": f"python -m tools.eiel sessionManager resume {record['record_id']} --codex-id '<codex-id>'"})
    elif status == "ACTIVE":
        if authorization_problems:
            next_actions.append(
                {
                    "action": "refresh_or_reconcile_session",
                    "description": "The retained command is no longer authorized by the live repository or actor state.",
                    "command": f"python -m tools.eiel sessionManager checkpoint {record['record_id']}",
                }
            )
        elif payload.get("next_action"):
            next_actions.append({"action": "continue", "description": str(payload["next_action"]), **({"command": payload["next_command"]} if payload.get("next_command") else {})})
        else:
            next_actions.append({"action": "checkpoint", "description": "Record the current stage and exact next action before stopping.", "command": f"python -m tools.eiel sessionManager checkpoint {record['record_id']} --next-action '<next action>'"})

    # Session outcomes are historical facts.  Current Work lifecycle, package
    # integrity and delivery/review lineage decide what may happen now.
    current_work_summary: dict[str, Any] | None = None
    current_work_record: dict[str, Any] | None = None
    try:
        work_record = work_store(state_root).get_read_only(str(payload.get("work_id") or ""))
        current_work_record = dict(work_record)
        from tools.actors.work_manager import summarize_work_record

        current_work_summary = summarize_work_record(
            work_record,
            state_root=state_root,
            read_only=True,
            repo_root=dict(payload.get("repository") or {}).get("repo_root"),
        )
    except ActorError as exc:
        validation["errors"] = [*list(validation.get("errors") or []), str(exc)]
        validation["valid"] = False
    if current_work_summary is not None:
        work_errors = [
            str(value)
            for value in dict(current_work_summary.get("validation") or {}).get("errors") or []
        ]
        if work_errors or not bool(current_work_summary.get("valid", True)):
            validation["errors"] = [*list(validation.get("errors") or []), *work_errors]
            validation["valid"] = False
            next_actions = [
                {
                    "action": "repair_work_integrity",
                    "description": "Repair the authoritative Work or Work Package before continuing this historical Session.",
                    "command": f"python -m tools.eiel workManager inspect {payload.get('work_id')}",
                }
            ]
        elif str(current_work_summary.get("status") or "") in WORK_TERMINAL_STATUSES:
            next_actions = [
                {
                    "action": "none",
                    "description": "The parent Work is terminal; this Session has no executable continuation.",
                }
            ]
        elif str(current_work_summary.get("derived_status") or "") in {
            "PROMOTION_REQUIRED",
            "REVIEW_REQUIRED",
            "READY_TO_ACCEPT",
            "CHANGES_REQUIRED",
            "BLOCKED_BY_DEPENDENCY",
            "BLOCKED_BY_TERMINAL_DEPENDENCY",
        }:
            next_actions = list(current_work_summary.get("next_actions") or [])[:1]
    package = None
    package_binding = payload.get("work_package_binding")
    if package_binding:
        package_validation = verify_work_package_binding(
            package_binding,
            repo_root=dict(payload.get("repository") or {}).get("repo_root"),
            expected_work_id=str(payload.get("work_id") or ""),
            state_root=state_root,
        )
        package = package_validation.get("package")
    from tools.actors.work_package import effective_execution_budget
    execution_budget = effective_execution_budget(dict(package or {}))
    context_health = project_context_health(
        payload.get("context_health") if isinstance(payload.get("context_health"), Mapping) else None,
        execution_budget=execution_budget,
        session_status=status,
        session_outcome=str(outcome or "") or None,
        epoch=payload.get("epoch") if isinstance(payload.get("epoch"), Mapping) else None,
    )
    retained_delivery = (
        dict(payload.get("delivery_evidence") or {})
        if isinstance(payload.get("delivery_evidence"), Mapping)
        else None
    )
    delivery_projection = None
    if retained_delivery is not None:
        delivery_projection = {
            "delivery_session_id": delivery_session_id(retained_delivery),
            "delivery_base_commit": retained_delivery.get("base_commit"),
            "delivery_commit": delivery_commit_value(retained_delivery),
            "delivery_kind": retained_delivery.get("delivery_kind"),
            "delivery_sufficiency": delivery_sufficiency(retained_delivery),
            "delivery_proof": retained_delivery.get("delivery_proof"),
            "recorded_at_utc": retained_delivery.get("recorded_at_utc"),
        }
    canonical_attempt = canonical_session_attempt(
        record,
        work_record=current_work_record,
    )
    return {
        "session_id": record.get("record_id"),
        "work_id": payload.get("work_id"),
        "mission_id": canonical_attempt.get("mission_id"),
        "title": canonical_attempt.get("title"),
        "status": status,
        "status_scope": "SESSION_ATTEMPT",
        "status_label": f"Session {status}",
        "session_attempt_status": status,
        "canonical_session_attempt": canonical_attempt,
        "canonical_session_state": canonical_attempt["state"],
        "canonical_review_verdict": canonical_attempt.get("review_verdict"),
        "revision": record.get("revision"),
        "original_codex_id": payload.get("original_codex_id"),
        "original_codex_thread_url": payload.get("original_codex_thread_url"),
        "current_codex_id": payload.get("current_codex_id"),
        "current_codex_thread_url": payload.get("current_codex_thread_url"),
        "preferred_continuation_codex_id": payload.get("preferred_continuation_codex_id"),
        "codex_participants": payload.get("codex_participants"),
        "repository": payload.get("repository"),
        "bootstrap_record_id": payload.get("bootstrap_record_id"),
        "bootstrap_history": payload.get("bootstrap_history"),
        "current_stage": payload.get("current_stage"),
        "current_step": payload.get("current_step"),
        "completed_steps": payload.get("completed_steps"),
        "interrupted_during": payload.get("interrupted_during"),
        "blocked_reason": payload.get("blocked_reason"),
        "blocked_by_work_ids": payload.get("blocked_by_work_ids"),
        "next_action": payload.get("next_action"),
        "next_command": payload.get("next_command"),
        "last_successful_command": payload.get("last_successful_command"),
        "last_failed_command": payload.get("last_failed_command"),
        "references": payload.get("references"),
        "notes": payload.get("notes"),
        "session_role": payload.get("session_role") or "IMPLEMENTATION_AUTHORITY",
        "model_selection": payload.get("model_assignment"),
        "model_assignment": payload.get("model_assignment"),
        "identity_assurance": payload.get("identity_assurance"),
        "work_package_binding": payload.get("work_package_binding"),
        "context_contract_status": payload.get("context_contract_status"),
        "session_outcome": payload.get("session_outcome"),
        "review_verdict": payload.get("review_verdict"),
        "promotion_request": payload.get("promotion_request"),
        "delivery_evidence": payload.get("delivery_evidence"),
        "delivery": delivery_projection,
        "review_target": payload.get("review_target"),
        "review_evidence": payload.get("review_evidence"),
        "authoritative_review_binding": payload.get("authoritative_review_binding"),
        "epoch": payload.get("epoch"),
        "context_health": context_health,
        "context_receipt": payload.get("context_receipt"),
        "resumed_from": payload.get("resumed_from"),
        "validation_events": payload.get("validation_events") or [],
        "validation_gaps": payload.get("validation_gaps") or [],
        "validation_decision_impact": payload.get("validation_decision_impact"),
        "epoch_history": payload.get("epoch_history") or [],
        "terminal_transition": payload.get("terminal_transition"),
        "attempt_disposition": payload.get("attempt_disposition"),
        "artifact_dispositions": payload.get("artifact_dispositions") or [],
        "review_progress": payload.get("review_progress"),
        "repair_lineage": payload.get("repair_lineage"),
        "authority_binding": {
            "schema_version": "eiel_authority_binding_projection_v1",
            "record_owner": "SESSION_MANAGER",
            "record_authority": "SESSION_MANAGER",
            "attempt_owner": "SESSION_MANAGER",
            "authority_role": payload.get("session_role") or "IMPLEMENTATION_AUTHORITY",
            "authority_domain": "CODEX_EXECUTION_ATTEMPT",
            "authorization_basis": (
                "BOUND_WORK_PACKAGE" if payload.get("work_package_binding") else "HUMAN_OPERATOR_SESSION_RECORD"
            ),
            "current_technical_authority": payload.get("session_role") or "IMPLEMENTATION_AUTHORITY",
            "session_role": payload.get("session_role") or "IMPLEMENTATION_AUTHORITY",
            "identity_assurance": payload.get("identity_assurance") or "DECLARED",
        },
        "projection_freshness": {
            "schema_version": "eiel_projection_freshness_v1",
            "authoritative_record": actor_record_ref(record),
            "parent_work_record": actor_record_ref(current_work_record) if current_work_record else None,
            "parent_work_execution_ref": work_execution_ref(current_work_record) if current_work_record else None,
            "bootstrap_record_id": payload.get("bootstrap_record_id"),
            "status": "EXACT_RECORD_BOUND" if current_work_record else "PARENT_WORK_UNAVAILABLE",
        },
        "resource_accounting": normalize_session_resource_accounting(
            payload.get("resource_accounting") if isinstance(payload.get("resource_accounting"), Mapping) else None
        ),
        "worker_identity_state": payload.get("worker_identity_state") or "ACTIVE",
        "prepared_assignment_id": payload.get("prepared_assignment_id"),
        "scheduler_assignment_id": payload.get("scheduler_assignment_id"),
        "scheduler_assignment_digest": payload.get("scheduler_assignment_digest"),
        "scheduler_lease_id": payload.get("scheduler_lease_id"),
        "scheduler_worker_id": payload.get("scheduler_worker_id"),
        "assignment_history": payload.get("assignment_history") or [],
        "worker_loss": payload.get("worker_loss"),
        "valid": validation["valid"],
        "validation": validation,
        "next_actions": next_actions,
        "updated_at_utc": record.get("updated_at_utc"),
        "record_sha256": record.get("record_sha256"),
    }


def summarize_session_record(
    record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    authorize_live: bool = False,
) -> dict[str, Any]:
    """Public owner-qualified Session projection."""

    return _summary(record, state_root=state_root, authorize_live=authorize_live)


def _transition(record: Mapping[str, Any], new_status: str) -> None:
    current = str(record.get("status") or "CREATED")
    if new_status not in ALLOWED_TRANSITIONS.get(current, set()):
        raise ActorError(
            f"invalid session transition {current} -> {new_status}",
            code="session_transition_invalid",
            problems=[f"Allowed transitions from {current}: {', '.join(sorted(ALLOWED_TRANSITIONS.get(current, set())))}"],
            related_records=[str(record.get("record_id"))],
        )


def _session_authority(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "actor": ACTOR_NAME,
        "role": str(payload.get("session_role") or ""),
        "codex_id": payload.get("current_codex_id"),
        "identity_assurance": normalize_identity_assurance(
            payload.get("identity_assurance"), fallback_codex_id=payload.get("current_codex_id")
        ),
        "source": "bounded_manager_command",
    }


def _default_attempt_disposition(*, status: str, action: str, payload: Mapping[str, Any], reason: str, generated_at: str | None) -> dict[str, Any]:
    outcome = str(payload.get("session_outcome") or "")
    if status == "COMPLETED":
        disposition = "SUCCEEDED"
    elif status == "DISCONTINUED":
        disposition = "ABANDONED"
    elif status == "OBSOLETE":
        disposition = "NOT_NEEDED"
    elif status == "SUPERSEDED":
        disposition = "SUPERSEDED"
    elif str(dict(payload.get("worker_loss") or {}).get("reason") or ""):
        disposition = "FAILED"
    else:
        disposition = "FAILED" if outcome in {"FAILED", "PROCESS_FAILURE"} else "ABANDONED"
    return load_attempt_disposition({
        "schema_version": SESSION_ATTEMPT_DISPOSITION_SCHEMA_VERSION,
        "disposition": disposition,
        "reason": reason or f"{action}: {status}",
        "authority": _session_authority(payload),
        "prior_session_ref": None,
        "continuation_ref": None,
        "recorded_at_utc": utc_now(generated_at),
    })


def _default_artifact_dispositions(payload: Mapping[str, Any], *, generated_at: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    references = dict(payload.get("references") or {})
    for category in sorted(references):
        values = references.get(category) or []
        if not isinstance(values, list):
            continue
        for index, value in enumerate(values):
            ref = dict(value) if isinstance(value, Mapping) else {"category": category, "value": value}
            rows.append(load_artifact_disposition({
                "schema_version": SESSION_ARTIFACT_DISPOSITION_SCHEMA_VERSION,
                "artifact_ref": ref,
                "disposition": "RETAINED_REFERENCE",
                "reason": f"retained terminal Session reference from {category}[{index}]",
                "recorded_at_utc": utc_now(generated_at),
            }))
    for field in ("delivery_evidence", "review_evidence", "context_receipt"):
        value = payload.get(field)
        if isinstance(value, Mapping):
            rows.append(load_artifact_disposition({
                "schema_version": SESSION_ARTIFACT_DISPOSITION_SCHEMA_VERSION,
                "artifact_ref": {"field": field, "sha256": canonical_sha256(value)},
                "disposition": "INCLUDED",
                "reason": f"authoritative {field} retained on terminal Session",
                "recorded_at_utc": utc_now(generated_at),
            }))
    return rows


def _review_finding_ids(*, session_id: str, findings: Sequence[str]) -> list[str]:
    return [stable_id("FIND", {"review_session_id": session_id, "index": index, "finding": text})
            for index, text in enumerate(findings)]


def _revise_session_record(
    record: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    status: str,
    action: str,
    state_root: str | Path | None,
    generated_at: str | None,
    note: str | None = None,
    allow_terminal_work: bool = False,
    verify_bootstrap: bool = True,
    validation_message: str = "session validation failed",
) -> dict[str, Any]:
    """Apply every durable Session status write through one policy gateway.

    Scheduler integration and recovery paths still own their specific identity,
    repository, Assignment, Lease, and loss guards.  This gateway owns the
    shared source-to-target policy, cross-field Session validation, immutable
    record revision, and store write.  Keeping those responsibilities here
    prevents recovery code from quietly acquiring a second lifecycle model.
    """

    _transition(record, status)
    raw_payload = dict(payload)
    payload = _normalize_payload(
        raw_payload,
        session_id=str(record.get("record_id") or raw_payload.get("session_id") or ""),
        generated_at=str(record.get("created_at_utc") or utc_now(generated_at)),
        allow_prepared=str(raw_payload.get("worker_identity_state") or "") in {"PENDING", "LOST", "DETACHED"},
    )
    if status in SESSION_TERMINAL_STATUSES and str(record.get("status") or "") != status:
        identity = normalize_identity_assurance(
            payload.get("identity_assurance"), fallback_codex_id=payload.get("current_codex_id")
        )
        reason = str(
            note
            or payload.get("blocked_reason")
            or payload.get("session_outcome")
            or f"{action} through bounded Session Manager lifecycle command"
        ).strip()
        payload["terminal_transition"] = {
            "schema_version": "eiel_terminal_transition_v1",
            "reason": reason,
            "authority": {
                "actor": ACTOR_NAME,
                "role": str(payload.get("session_role") or ""),
                "codex_id": payload.get("current_codex_id"),
                "identity_assurance": identity,
                "source": "bounded_manager_command",
            },
            "prior_record": actor_record_ref(record),
            "replacement_state": {
                "record_id": str(record.get("record_id") or ""),
                "revision": int(record.get("revision") or 0) + 1,
                "status": status,
                "payload_sha256": canonical_sha256(payload),
            },
            "action": action,
            "recorded_at_utc": utc_now(generated_at),
        }
        if not isinstance(payload.get("attempt_disposition"), Mapping):
            payload["attempt_disposition"] = _default_attempt_disposition(
                status=status, action=action, payload=payload, reason=reason, generated_at=generated_at
            )
        if not payload.get("artifact_dispositions"):
            payload["artifact_dispositions"] = _default_artifact_dispositions(payload, generated_at=generated_at)
    validation = _validate_payload(
        payload,
        state_root=state_root,
        record_status=status,
        verify_bootstrap=verify_bootstrap,
        allow_terminal_work=allow_terminal_work,
        allow_historical_package=status in SESSION_TERMINAL_STATUSES,
    )
    if not validation["valid"]:
        raise ActorError(
            validation_message,
            code="session_validation_failed",
            problems=validation["errors"],
            warnings=validation["warnings"],
        )
    revised = revise_record(
        record,
        payload=payload,
        status=status,
        action=action,
        generated_at=utc_now(generated_at),
        note=note,
    )
    session_store(state_root, write_scope=True).put(revised, allow_update=True)
    return revised


def attach_session_resource_accounting(
    session_id: str,
    accounting: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    generated_at: str | None = None,
    action: str,
    note: str | None = None,
) -> dict[str, Any]:
    """Attach WRA evidence through the Session Manager's authoritative writer."""

    normalized_id = validate_prefixed_id(session_id, SESSION_ID_PREFIX, label="session")
    initial = session_store(state_root).get_read_only(normalized_id)
    work_id = str(dict(initial.get("payload") or {}).get("work_id") or "")
    with work_lifecycle_lock(state_root, work_id):
        record = session_store(state_root).get(normalized_id)
        normalized = load_session_resource_accounting(accounting)
        payload = dict(record.get("payload") or {})
        if normalized == payload.get("resource_accounting"):
            return dict(record)
        payload["resource_accounting"] = normalized
        payload = _normalize_payload(
            payload,
            session_id=normalized_id,
            generated_at=str(record.get("created_at_utc") or utc_now(generated_at)),
        )
        return _revise_session_record(
            record,
            payload=payload,
            status=str(record.get("status") or "ACTIVE"),
            action=action,
            state_root=state_root,
            generated_at=generated_at,
            note=note,
            allow_terminal_work=str(record.get("status") or "") in SESSION_TERMINAL_STATUSES,
        )


def _revise(
    args: argparse.Namespace,
    record: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    status: str,
    action: str,
    note: str | None,
    allow_terminal_work: bool = False,
    verify_bootstrap: bool = True,
) -> dict[str, Any]:
    return _revise_session_record(
        record,
        payload=payload,
        status=status,
        state_root=args.state_root,
        generated_at=args.generated_at,
        action=action,
        note=note,
        allow_terminal_work=allow_terminal_work,
        verify_bootstrap=verify_bootstrap,
    )


def _guided(
    payload: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    authorize_live: bool = False,
) -> dict[str, Any]:
    summary = _summary(record, state_root=state_root, authorize_live=authorize_live)
    return with_guidance(
        payload,
        status="BLOCKED" if not summary["valid"] else summary["status"],
        reason_codes=["session_validation_failed"] if not summary["valid"] else [f"session_{summary['status'].lower()}"],
        problems=summary["validation"]["errors"],
        warnings=summary["validation"]["warnings"],
        next_actions=summary["next_actions"],
        related_records=[summary["session_id"], summary["work_id"], *([summary["mission_id"]] if summary.get("mission_id") else [])],
    )


def _prepare_session_for_assignment_locked(
    *,
    state_root: str | Path | None,
    work_id: str,
    bootstrap_record_id: str,
    repo_root: str | Path,
    assignment_id: str,
    session_id: str,
    role: str,
    worker_profile: Mapping[str, Any] | None = None,
    review_target_session: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Prepare a Session without inventing a live Codex identity."""

    timestamp = utc_now(generated_at)
    work_id = validate_prefixed_id(work_id, WORK_ID_PREFIX, label="work item")
    readiness = _work_readiness(state_root, work_id)
    work = readiness["record"]
    if work.get("status") not in WORK_OPEN_STATUSES:
        raise ActorError(f"work item is not open: {work.get('status')}", code="session_work_not_open")
    if not readiness["dependencies"]["all_satisfied"]:
        raise ActorError("work dependencies are unresolved", code="session_work_dependencies_unresolved")
    conflicts = _active_session_conflicts(state_root, work_id)
    if conflicts:
        raise ActorError("another active session already owns this work item", code="session_active_conflict", problems=conflicts)
    bootstrap_id = validate_prefixed_id(bootstrap_record_id, BOOTSTRAP_ID_PREFIX, label="bootstrap")
    bootstrap = bootstrap_store(state_root).get(bootstrap_id)
    repo_identity = git_repository_identity(repo_root)
    bootstrap_errors = validate_bootstrap_record(
        bootstrap,
        expected_repository=repo_identity,
        revalidate_environment=True,
        max_age_hours=None,
        now=timestamp,
    )
    if bootstrap_errors:
        raise ActorError("bootstrap receipt is not valid for prepared Session", code="session_bootstrap_invalid", problems=bootstrap_errors)
    work_payload = dict(work.get("payload") or {})
    requested_role = str(role or dict(work_payload.get("work_package_binding") or {}).get("recommended_authority") or "IMPLEMENTATION_AUTHORITY").upper()
    package_binding = _package_binding_for_session(work, state_root=state_root, role=requested_role)
    _require_package_delivery_role(package_binding, requested_role)
    _require_bootstrap_session_authority(bootstrap, requested_role)
    review_target = None
    if requested_role == "REVIEW_AUTHORITY":
        if not package_binding:
            raise ActorError("Review Authority sessions require a bound Work Package", code="review_work_package_missing")
        review_target = _select_review_target(
            state_root=state_root,
            work_id=work_id,
            package_sha256=str(package_binding.get("sha256") or ""),
            requested_session_id=review_target_session,
        )
        reviewed_commit = str(review_target.get("reviewed_commit") or "")
        if reviewed_commit and reviewed_commit != str(repo_identity.get("head") or ""):
            raise ActorError("review worktree HEAD does not match implementation commit", code="review_commit_mismatch")
    authoritative_review_binding = None
    if requested_role == "IMPLEMENTATION_AUTHORITY":
        candidate_binding = _current_authoritative_review_binding(
            work,
            state_root=state_root,
        )
        if str(dict(candidate_binding or {}).get("verdict") or "") in {
            "CHANGES_REQUIRED",
            "REJECT",
            "BLOCKED",
        }:
            authoritative_review_binding = candidate_binding
    profile = dict(worker_profile or {})
    selected_model = str(profile.get("model") or dict(package_binding or {}).get("recommended_model") or "").strip() or None
    context_status = _delegated_context_contract_status(
        package_binding=package_binding,
        repo_root=repo_root,
        work_id=work_id,
        role=requested_role,
        model=selected_model,
        acknowledge_defaults=False,
        state_root=state_root,
    )
    raw = {
        "work_id": work_id,
        "mission_id": work_payload.get("mission_id"),
        "title": work_payload.get("title"),
        "bootstrap_record_id": bootstrap_id,
        "original_codex_id": None,
        "current_codex_id": None,
        "preferred_continuation_codex_id": None,
        "codex_participants": [],
        "worker_identity_state": "PENDING",
        "identity_assurance": declared_identity_assurance(codex_id=None, source="scheduler_prepared"),
        "context_contract_status": context_status,
        "prepared_assignment_id": assignment_id,
        "current_stage": "assignment_prepared",
        "current_step": work_payload.get("current_step") or "await worker activation",
        "next_action": work_payload.get("next_action") or "Activate the prepared Session from the immutable Assignment.",
        "next_command": None,
        "repository": _session_repository(repo_identity),
        "parallel_session_acknowledged": False,
        "session_role": requested_role,
        "model_assignment": {
            "provider": "OPENAI",
            "model": profile.get("model"),
            "reasoning_level": profile.get("reasoning_level"),
            "selected_by": "HUMAN_OPERATOR",
        },
        "work_package_binding": package_binding,
        "session_outcome": None,
        "review_verdict": None,
        "promotion_request": None,
        "delivery_evidence": None,
        "review_target": review_target,
        "review_evidence": None,
        "authoritative_review_binding": authoritative_review_binding,
        "assignment_history": [{"assignment_id": assignment_id, "event": "PREPARED", "at_utc": timestamp}],
    }
    payload = _normalize_payload(raw, session_id=session_id, generated_at=timestamp, allow_prepared=True)
    validation = _validate_payload(payload, state_root=state_root, record_status="CREATED")
    if not validation["valid"]:
        raise ActorError("prepared Session validation failed", code="session_validation_failed", problems=validation["errors"])
    from tools.scheduler.contracts import assignment_store as scheduler_assignment_store

    assignment = scheduler_assignment_store(state_root).get_read_only(assignment_id)
    record = new_record(
        actor=ACTOR_NAME,
        record_type=SESSION_RECORD_TYPE,
        record_id=payload["session_id"],
        status="CREATED",
        payload=payload,
        generated_at=timestamp,
        source_refs=[
            actor_source_ref("bootstrap", bootstrap),
            actor_source_ref("work", work),
            actor_source_ref("assignment", assignment),
        ],
        action="prepare_assignment_session",
        note="Prepared without a live Codex identity",
    )
    session_store(state_root, write_scope=True).put(record)
    return record


def prepare_session_for_assignment(
    *,
    state_root: str | Path | None,
    work_id: str,
    bootstrap_record_id: str,
    repo_root: str | Path,
    assignment_id: str,
    session_id: str,
    role: str,
    worker_profile: Mapping[str, Any] | None = None,
    review_target_session: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Prepare one scheduler Session under the Work lifecycle lock."""

    normalized_work_id = validate_prefixed_id(work_id, WORK_ID_PREFIX, label="work item")
    with work_lifecycle_lock(state_root, normalized_work_id):
        return _prepare_session_for_assignment_locked(
            state_root=state_root,
            work_id=normalized_work_id,
            bootstrap_record_id=bootstrap_record_id,
            repo_root=repo_root,
            assignment_id=assignment_id,
            session_id=session_id,
            role=role,
            worker_profile=worker_profile,
            review_target_session=review_target_session,
            generated_at=generated_at,
        )


def _activate_session_for_assignment_locked(
    session_id: str,
    *,
    state_root: str | Path | None,
    assignment_id: str,
    assignment_digest: str,
    lease_id: str,
    worker_id: str,
    codex_id: str,
    repo_root: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Activate a prepared or resumable Session for one exact worker attempt."""

    timestamp = utc_now(generated_at)
    store_obj = session_store(state_root)
    record = store_obj.get(validate_prefixed_id(session_id, SESSION_ID_PREFIX, label="session"))
    payload = dict(record.get("payload") or {})
    prepared = payload.get("worker_identity_state") == "PENDING"
    if prepared and str(payload.get("prepared_assignment_id") or "") != assignment_id:
        raise ActorError("prepared Session belongs to another Assignment", code="session_assignment_mismatch")
    if record.get("status") not in {"CREATED", "PAUSED", "INTERRUPTED", "BLOCKED", "ACTIVE"}:
        raise ActorError("Session cannot be activated for Assignment", code="session_assignment_activation_invalid")
    if record.get("status") == "ACTIVE":
        if (
            payload.get("scheduler_assignment_id") == assignment_id
            and payload.get("scheduler_lease_id") == lease_id
            and payload.get("scheduler_worker_id") == worker_id
        ):
            return record
        raise ActorError("Session is already active under another worker", code="session_active_conflict")
    repo_identity = git_repository_identity(repo_root)
    retained_repo = dict(payload.get("repository") or {})
    repository_problems = _repository_matches_session(retained_repo, repo_identity)
    if repository_problems:
        raise ActorError(
            "Session repository changed since assignment preparation",
            code="session_repository_state_mismatch",
            problems=repository_problems,
        )
    participant = {
        "codex_id": codex_id,
        "thread_url": None,
        "joined_at_utc": timestamp,
        "role": "original" if not payload.get("original_codex_id") else "takeover",
        "reason": f"Assignment {assignment_id} activated by worker {worker_id}",
    }
    payload["original_codex_id"] = payload.get("original_codex_id") or codex_id
    payload["current_codex_id"] = codex_id
    payload["preferred_continuation_codex_id"] = codex_id
    payload["codex_participants"] = [*list(payload.get("codex_participants") or []), participant]
    payload["worker_identity_state"] = "ACTIVE"
    payload["scheduler_assignment_id"] = assignment_id
    payload["scheduler_assignment_digest"] = assignment_digest
    payload["scheduler_lease_id"] = lease_id
    payload["scheduler_worker_id"] = worker_id
    payload["identity_assurance"] = scheduler_identity_assurance(
        codex_id=codex_id, worker_id=worker_id, lease_id=lease_id
    )
    payload["worker_loss"] = None
    payload["current_stage"] = "assignment_active"
    payload["assignment_history"] = [
        *list(payload.get("assignment_history") or []),
        {"assignment_id": assignment_id, "lease_id": lease_id, "worker_id": worker_id, "event": "ACTIVATED", "at_utc": timestamp},
    ]
    return _revise_session_record(
        record,
        payload=payload,
        status="ACTIVE",
        action="activate_assignment",
        state_root=state_root,
        generated_at=timestamp,
        note=f"Activated by {worker_id}",
        validation_message="activated Session validation failed",
    )


def activate_session_for_assignment(
    session_id: str,
    *,
    state_root: str | Path | None,
    assignment_id: str,
    assignment_digest: str,
    lease_id: str,
    worker_id: str,
    codex_id: str,
    repo_root: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Activate one scheduler Session under its Work lifecycle lock."""

    normalized_session_id = validate_prefixed_id(session_id, SESSION_ID_PREFIX, label="session")
    existing = session_store(state_root).get_read_only(normalized_session_id)
    work_id = validate_prefixed_id(str((existing.get("payload") or {}).get("work_id") or ""), WORK_ID_PREFIX, label="work item")
    with work_lifecycle_lock(state_root, work_id):
        return _activate_session_for_assignment_locked(
            normalized_session_id,
            state_root=state_root,
            assignment_id=assignment_id,
            assignment_digest=assignment_digest,
            lease_id=lease_id,
            worker_id=worker_id,
            codex_id=codex_id,
            repo_root=repo_root,
            generated_at=generated_at,
        )


def _bind_session_codex_thread_identity_locked(
    session_id: str,
    *,
    state_root: str | Path | None,
    assignment_id: str,
    lease_id: str,
    worker_id: str,
    thread_id: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Replace a provisional process identity with the real Codex thread ID."""

    actual = str(thread_id or "").strip()
    if not actual or _is_provisional_codex_id(actual):
        raise ActorError("real Codex thread identity is required", code="session_codex_identity_missing")
    timestamp = utc_now(generated_at)
    store_obj = session_store(state_root)
    record = store_obj.get(validate_prefixed_id(session_id, SESSION_ID_PREFIX, label="session"))
    payload = dict(record.get("payload") or {})
    if record.get("status") != "ACTIVE":
        raise ActorError("Session is not active", code="session_assignment_activation_invalid")
    if payload.get("scheduler_assignment_id") != assignment_id or payload.get("scheduler_lease_id") != lease_id or payload.get("scheduler_worker_id") != worker_id:
        raise ActorError("Codex thread binding does not match active Assignment", code="session_assignment_mismatch")
    current = str(payload.get("current_codex_id") or "")
    if current == actual:
        return record
    if current and not _is_provisional_codex_id(current):
        raise ActorError("Session already binds a different real Codex identity", code="session_codex_identity_conflict")
    participants = [row for row in list(payload.get("codex_participants") or []) if not _is_provisional_codex_id(dict(row).get("codex_id"))]
    participants.append({
        "codex_id": actual,
        "thread_url": None,
        "joined_at_utc": timestamp,
        "role": "original" if not payload.get("original_codex_id") or _is_provisional_codex_id(payload.get("original_codex_id")) else "takeover",
        "reason": f"Real Codex thread bound for Assignment {assignment_id}",
    })
    if not payload.get("original_codex_id") or _is_provisional_codex_id(payload.get("original_codex_id")):
        payload["original_codex_id"] = actual
    payload["current_codex_id"] = actual
    payload["preferred_continuation_codex_id"] = actual
    payload["identity_assurance"] = scheduler_identity_assurance(
        codex_id=actual,
        worker_id=str(payload.get("scheduler_worker_id") or "") or None,
        lease_id=str(payload.get("scheduler_lease_id") or "") or None,
    )
    payload["codex_participants"] = participants
    payload["assignment_history"] = [*list(payload.get("assignment_history") or []), {
        "assignment_id": assignment_id, "lease_id": lease_id, "worker_id": worker_id,
        "codex_thread_id": actual, "event": "CODEX_THREAD_BOUND", "at_utc": timestamp,
    }]
    candidate = dict(record)
    candidate["payload"] = payload
    candidate["status"] = "ACTIVE"
    resource_accounting, _ = capture_start(
        candidate,
        state_root=state_root,
        generated_at=generated_at,
    )
    payload["resource_accounting"] = resource_accounting
    return _revise_session_record(
        record,
        payload=payload,
        status="ACTIVE",
        action="bind_codex_thread",
        state_root=state_root,
        generated_at=timestamp,
        validation_message="Codex thread binding validation failed",
    )


def bind_session_codex_thread_identity(
    session_id: str,
    *,
    state_root: str | Path | None,
    assignment_id: str,
    lease_id: str,
    worker_id: str,
    thread_id: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    normalized_session_id = validate_prefixed_id(session_id, SESSION_ID_PREFIX, label="session")
    existing = session_store(state_root).get_read_only(normalized_session_id)
    work_id = str(dict(existing.get("payload") or {}).get("work_id") or "")
    with work_lifecycle_lock(state_root, work_id):
        with resource_thread_lock(state_root, thread_id):
            require_resource_thread_available(
                state_root,
                thread_id,
                exclude_session_id=normalized_session_id,
            )
            return _bind_session_codex_thread_identity_locked(
                normalized_session_id,
                state_root=state_root,
                assignment_id=assignment_id,
                lease_id=lease_id,
                worker_id=worker_id,
                thread_id=thread_id,
                generated_at=generated_at,
            )


def _attach_session_bootstrap_after_commit_locked(
    session_id: str,
    *,
    state_root: str | Path | None,
    bootstrap_record_id: str,
    repo_root: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Bind a fresh verify-only Bootstrap to any active Session after a commit."""

    timestamp = utc_now(generated_at)
    store_obj = session_store(state_root)
    record = store_obj.get(validate_prefixed_id(session_id, SESSION_ID_PREFIX, label="session"))
    if record.get("status") != "ACTIVE":
        raise ActorError("Bootstrap refresh requires an active Session", code="session_bootstrap_attach_invalid")
    payload = dict(record.get("payload") or {})
    identity = git_repository_identity(repo_root)
    bootstrap = bootstrap_store(state_root).get(
        validate_prefixed_id(bootstrap_record_id, BOOTSTRAP_ID_PREFIX, label="bootstrap record")
    )
    errors = validate_bootstrap_record(
        bootstrap, expected_repo_root=identity["root"], expected_head=identity["head"], max_age_hours=None
    )
    if errors:
        raise ActorError("Bootstrap is not valid for the committed worker repository", code="session_bootstrap_invalid", problems=errors)
    prior = str(payload.get("bootstrap_record_id") or "")
    payload["delivery_base_commit"] = payload.get("delivery_base_commit") or str(
        dict(payload.get("repository") or {}).get("head") or ""
    ).strip() or None
    payload["bootstrap_record_id"] = str(bootstrap.get("record_id"))
    payload["repository"] = {
        "repo_root": identity["root"],
        "branch": identity.get("branch"),
        "head": identity["head"],
        "status_short_branch": identity.get("status_short_branch"),
    }
    history = normalize_string_list(payload.get("bootstrap_history") or ([prior] if prior else []))
    if str(bootstrap.get("record_id")) not in history:
        history.append(str(bootstrap.get("record_id")))
    payload["bootstrap_history"] = history
    return _revise_session_record(
        record,
        payload=payload,
        status="ACTIVE",
        action="attach_worker_bootstrap",
        state_root=state_root,
        generated_at=timestamp,
        validation_message="Session Bootstrap attachment validation failed",
    )



def attach_session_bootstrap_after_commit(
    session_id: str,
    *,
    state_root: str | Path | None,
    bootstrap_record_id: str,
    repo_root: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    normalized_session_id = validate_prefixed_id(session_id, SESSION_ID_PREFIX, label="session")
    existing = session_store(state_root).get_read_only(normalized_session_id)
    work_id = str(dict(existing.get("payload") or {}).get("work_id") or "")
    with work_lifecycle_lock(state_root, work_id):
        return _attach_session_bootstrap_after_commit_locked(
            normalized_session_id,
            state_root=state_root,
            bootstrap_record_id=bootstrap_record_id,
            repo_root=repo_root,
            generated_at=generated_at,
        )
def _invalidate_prepared_session_for_assignment_locked(
    session_id: str,
    *,
    state_root: str | Path | None,
    assignment_id: str,
    reason: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Detach an unused prepared Session after Assignment compilation fails."""

    timestamp = utc_now(generated_at)
    store_obj = session_store(state_root)
    record = store_obj.get(validate_prefixed_id(session_id, SESSION_ID_PREFIX, label="session"))
    payload = dict(record.get("payload") or {})
    if payload.get("worker_identity_state") != "PENDING" or payload.get("prepared_assignment_id") != assignment_id:
        return record
    prior_next_action = payload.get("next_action")
    prior_next_command = payload.get("next_command")
    payload.update({
        "worker_identity_state": "DETACHED",
        "prepared_assignment_id": None,
        "scheduler_assignment_id": None,
        "scheduler_assignment_digest": None,
        "scheduler_lease_id": None,
        "scheduler_worker_id": None,
        "worker_loss": {
            "assignment_id": assignment_id, "lease_id": None, "reason": str(reason),
            "recorded_at_utc": timestamp, "prepared_only": True,
            "prior_next_action": prior_next_action, "prior_next_command": prior_next_command,
        },
        "interrupted_during": "assignment preparation",
        "next_action": prior_next_action or "Resume this Session manually or compile a fresh Assignment.",
        "next_command": prior_next_command,
    })
    return _revise_session_record(
        record,
        payload=payload,
        status="INTERRUPTED",
        action="invalidate_prepared_assignment",
        state_root=state_root,
        generated_at=timestamp,
        note=str(reason),
        validation_message="detached prepared Session validation failed",
    )

def _record_worker_loss_locked(
    session_id: str,
    *,
    state_root: str | Path | None,
    assignment_id: str,
    lease_id: str | None,
    reason: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Record mechanical worker loss without interpreting engineering failure."""

    timestamp = utc_now(generated_at)
    store_obj = session_store(state_root)
    record = store_obj.get(validate_prefixed_id(session_id, SESSION_ID_PREFIX, label="session"))
    if record.get("status") in SESSION_TERMINAL_STATUSES:
        return record
    payload = dict(record.get("payload") or {})
    bound = str(payload.get("scheduler_assignment_id") or payload.get("prepared_assignment_id") or "")
    if bound and bound != assignment_id:
        raise ActorError("worker-loss Assignment does not match Session", code="session_assignment_mismatch")
    if payload.get("scheduler_lease_id") and lease_id and payload.get("scheduler_lease_id") != lease_id:
        raise ActorError("worker-loss Lease does not match Session", code="session_lease_mismatch")
    payload["worker_identity_state"] = "LOST"
    # The expired Assignment no longer owns continuation. Retain its identity
    # in worker_loss, but release the live scheduler bindings so the exact
    # interrupted Session can be selected by the next Assignment.
    payload["prepared_assignment_id"] = None
    payload["scheduler_assignment_id"] = None
    payload["scheduler_lease_id"] = None
    payload["scheduler_worker_id"] = None
    prior_next_action = payload.get("next_action")
    prior_next_command = payload.get("next_command")
    payload["worker_loss"] = {
        "assignment_id": assignment_id,
        "lease_id": lease_id,
        "reason": str(reason),
        "recorded_at_utc": timestamp,
        "prior_next_action": prior_next_action,
        "prior_next_command": prior_next_command,
        "mechanical_recovery_advice": "Inspect the loss, then resume manually or compile a fresh Assignment.",
    }
    payload["interrupted_during"] = payload.get("current_step") or "scheduler worker execution"
    payload["next_action"] = prior_next_action or "Resume this exact Session after inspecting the mechanical worker loss."
    payload["next_command"] = prior_next_command
    payload["current_codex_id"] = payload.get("current_codex_id")
    return _revise_session_record(
        record,
        payload=payload,
        status="INTERRUPTED",
        action="record_worker_loss",
        state_root=state_root,
        generated_at=timestamp,
        note=str(reason),
        validation_message="worker-loss Session validation failed",
    )


def invalidate_prepared_session_for_assignment(
    session_id: str,
    *,
    state_root: str | Path | None,
    assignment_id: str,
    reason: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    normalized = validate_prefixed_id(session_id, SESSION_ID_PREFIX, label="session")
    initial = session_store(state_root).get_read_only(normalized)
    work_id = str(dict(initial.get("payload") or {}).get("work_id") or "")
    with work_lifecycle_lock(state_root, work_id):
        return _invalidate_prepared_session_for_assignment_locked(
            normalized,
            state_root=state_root,
            assignment_id=assignment_id,
            reason=reason,
            generated_at=generated_at,
        )


def record_worker_loss(
    session_id: str,
    *,
    state_root: str | Path | None,
    assignment_id: str,
    lease_id: str | None,
    reason: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    normalized = validate_prefixed_id(session_id, SESSION_ID_PREFIX, label="session")
    initial = session_store(state_root).get_read_only(normalized)
    work_id = str(dict(initial.get("payload") or {}).get("work_id") or "")
    with work_lifecycle_lock(state_root, work_id):
        return _record_worker_loss_locked(
            normalized,
            state_root=state_root,
            assignment_id=assignment_id,
            lease_id=lease_id,
            reason=reason,
            generated_at=generated_at,
        )


def command_prepare_assignment(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    profile = {"model": args.model, "reasoning_level": args.reasoning_level}
    record = prepare_session_for_assignment(
        state_root=args.state_root,
        work_id=args.work_id,
        bootstrap_record_id=args.bootstrap_record,
        repo_root=args.repo_root,
        assignment_id=args.assignment_id,
        session_id=args.session_id,
        role=args.role,
        worker_profile=profile,
        review_target_session=args.review_target_session,
        generated_at=args.generated_at,
    )
    return _guided({"ok": True, "created": True, "session": _summary(record, state_root=args.state_root)}, record, state_root=args.state_root), 0


def command_activate_assignment(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = activate_session_for_assignment(
        args.session_id,
        state_root=args.state_root,
        assignment_id=args.assignment_id,
        assignment_digest=args.assignment_digest,
        lease_id=args.lease_id,
        worker_id=args.worker_id,
        codex_id=args.codex_id,
        repo_root=args.repo_root,
        generated_at=args.generated_at,
    )
    return _guided({"ok": True, "session": _summary(record, state_root=args.state_root)}, record, state_root=args.state_root), 0


def command_worker_loss(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = record_worker_loss(
        args.session_id,
        state_root=args.state_root,
        assignment_id=args.assignment_id,
        lease_id=args.lease_id,
        reason=args.reason,
        generated_at=args.generated_at,
    )
    return _guided({"ok": True, "session": _summary(record, state_root=args.state_root)}, record, state_root=args.state_root), 0


def _delegated_context_contract_status(
    *,
    package_binding: Mapping[str, Any] | None,
    repo_root: str | Path,
    work_id: str,
    role: str,
    model: str | None,
    acknowledge_defaults: bool,
    state_root: str | Path | None = None,
) -> dict[str, Any] | None:
    if not package_binding:
        return None
    verified = verify_work_package_binding(
        package_binding,
        repo_root=repo_root,
        expected_work_id=work_id,
        verification_mode=VERIFICATION_EXECUTION_STRICT,
        state_root=state_root,
    )
    if not verified["valid"] or not isinstance(verified.get("package"), Mapping):
        raise ActorError(
            "Work Package binding is invalid before Session delegation",
            code="work_package_binding_invalid",
            problems=list(verified.get("errors") or []),
            warnings=list(verified.get("warnings") or []),
        )
    status = context_contract_status(dict(verified["package"]))
    delegated_spark = role == "IMPLEMENTATION_AUTHORITY" and "SPARK" in str(model or "").upper()
    if delegated_spark and not status["delegated_ready"] and not acknowledge_defaults:
        raise ActorError(
            "delegated Spark Work Package lacks explicit context contracts",
            code="work_package_context_contract_required",
            problems=[
                *(["Missing: " + ", ".join(status["missing_fields"])] if status["missing_fields"] else []),
                *(
                    ["read_contract must name entry files, required documents, search roots, or known symbols"]
                    if not status["has_entry_guidance"]
                    else []
                ),
            ],
            next_actions=[{
                "action": "revise_work_package_context_contract",
                "description": "Populate read_contract, output_policy, and execution_budget before delegating to Spark.",
            }],
            related_records=[work_id],
        )
    status["defaults_acknowledged"] = bool(acknowledge_defaults and not status["delegated_ready"] )
    return status


def _command_start_locked(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    timestamp = utc_now(args.generated_at)
    work_id = validate_prefixed_id(args.work_id, WORK_ID_PREFIX, label="work item")
    readiness = _work_readiness(args.state_root, work_id)
    work = readiness["record"]
    if work.get("status") not in WORK_OPEN_STATUSES:
        raise ActorError(f"work item is not open: {work.get('status')}", code="session_work_not_open")
    if not readiness["dependencies"]["all_satisfied"]:
        raise ActorError("work dependencies are unresolved", code="session_work_dependencies_unresolved", problems=[json.dumps(readiness["dependencies"], sort_keys=True)])
    conflicts = _active_session_conflicts(args.state_root, work_id)
    if conflicts and not args.allow_parallel:
        raise ActorError(
            "another active session already owns this work item",
            code="session_active_conflict",
            problems=[f"Active sessions: {', '.join(conflicts)}"],
            next_actions=[{"action": "resume_existing", "description": "Resume or explicitly close the existing active session.", "command": f"python -m tools.eiel sessionManager inspect {conflicts[0]}"}],
        )
    bootstrap_id = validate_prefixed_id(args.bootstrap_record, BOOTSTRAP_ID_PREFIX, label="bootstrap")
    bootstrap = bootstrap_store(args.state_root).get(bootstrap_id)
    repo_root = Path(args.repo_root or Path.cwd()).resolve()
    repo_identity = git_repository_identity(repo_root)
    bootstrap_errors = validate_bootstrap_record(
        bootstrap,
        expected_repository=repo_identity,
        revalidate_environment=True,
        max_age_hours=None if args.allow_stale_bootstrap else args.max_bootstrap_age_hours,
        now=args.generated_at,
    )
    if bootstrap_errors:
        raise ActorError("bootstrap receipt is not valid for this session", code="session_bootstrap_invalid", problems=bootstrap_errors, related_records=[bootstrap_id])
    raw = read_structured(args.input, name="session") if args.input else {}
    work_payload = dict(work.get("payload") or {})
    requested_role = str(
        args.role or raw.get("session_role") or dict(work_payload.get("work_package_binding") or {}).get("recommended_authority")
        or "IMPLEMENTATION_AUTHORITY"
    ).strip().upper()
    package_binding = _package_binding_for_session(work, state_root=args.state_root, role=requested_role)
    _require_package_delivery_role(package_binding, requested_role)
    _require_bootstrap_session_authority(bootstrap, requested_role)
    model_assignment = raw.get("model_assignment") if isinstance(raw.get("model_assignment"), Mapping) else None
    if args.model or args.reasoning_level:
        model_assignment = {
            "provider": "OPENAI",
            "model": args.model,
            "reasoning_level": args.reasoning_level,
            "selected_by": "HUMAN_OPERATOR",
        }
    selected_model = str(dict(model_assignment or {}).get("model") or dict(package_binding or {}).get("recommended_model") or "").strip() or None
    context_status = _delegated_context_contract_status(
        package_binding=package_binding,
        repo_root=repo_root,
        work_id=work_id,
        role=requested_role,
        model=selected_model,
        acknowledge_defaults=bool(getattr(args, "acknowledge_default_context_contracts", False)),
        state_root=args.state_root,
    )
    if conflicts and args.allow_parallel:
        active_records = [
            session_store(args.state_root).get_read_only(session_id)
            for session_id in conflicts
        ]
        candidate = {
            "record_id": str(args.session_id or "<new-session>"),
            "status": "ACTIVE",
            "payload": {
                "session_role": requested_role,
                "repository": _session_repository(repo_identity),
                "parallel_session_acknowledged": True,
            },
        }
        ownership_problems = parallel_session_ownership_problems([*active_records, candidate])
        if ownership_problems:
            raise ActorError(
                "parallel Sessions require concrete non-overlapping repository ownership",
                code="session_parallel_repository_conflict",
                problems=ownership_problems,
                related_records=[work_id, *conflicts],
            )
    review_target = None
    if requested_role == "REVIEW_AUTHORITY":
        if not package_binding:
            raise ActorError(
                "Review Authority sessions require a bound Work Package",
                code="review_work_package_missing",
                related_records=[work_id],
            )
        package_sha = str(package_binding.get("sha256") or "")
        review_target = _select_review_target(
            state_root=args.state_root,
            work_id=work_id,
            package_sha256=package_sha,
            requested_session_id=args.review_target_session,
        )
        requested_codex_id = str(
            args.codex_id
            or conversation_id_from_url(args.codex_url)
            or raw.get("original_codex_id")
            or raw.get("current_codex_id")
            or ""
        ).strip()
        # Same-identity review is a reduced-assurance fallback. Completion
        # records it as REVIEWER_INDEPENDENCE_UNAVAILABLE rather than claiming
        # independent review, so administrative identity scarcity cannot halt
        # otherwise bounded work.
        reviewed_commit = str(review_target.get("reviewed_commit") or "")
        if reviewed_commit and str(repo_identity.get("head") or "") != reviewed_commit:
            raise ActorError(
                "review worktree HEAD does not match the implementation commit being reviewed",
                code="review_commit_mismatch",
                problems=[
                    f"Reviewed commit: {reviewed_commit}",
                    f"Current HEAD: {repo_identity.get('head')}",
                ],
                next_actions=[
                    {
                        "action": "checkout_reviewed_commit",
                        "description": "Use a clean review worktree at the exact implementation commit, then start review again.",
                    }
                ],
            )

    authoritative_review_binding = None
    if requested_role == "IMPLEMENTATION_AUTHORITY":
        candidate_binding = _current_authoritative_review_binding(
            work,
            state_root=args.state_root,
        )
        if str(dict(candidate_binding or {}).get("verdict") or "") in {
            "CHANGES_REQUIRED",
            "REJECT",
            "BLOCKED",
        }:
            authoritative_review_binding = candidate_binding

    if requested_role == "REVIEW_AUTHORITY":
        default_stage = "review"
        default_step = "review the exact bound delivery"
        default_next_action = (
            "Review the exact delivery Session and commit against the bound Work Package."
        )
        default_next_command = None
    else:
        default_stage = "bootstrap_complete"
        default_step = work_payload.get("current_step") or "begin bounded work"
        default_next_action = (
            work_payload.get("next_action") or "Begin the bounded work package."
        )
        default_next_command = work_payload.get("next_command")

    raw.update({
        "work_id": work_id,
        "mission_id": args.mission_id or work_payload.get("mission_id"),
        "title": args.title or work_payload.get("title"),
        "bootstrap_record_id": bootstrap_id,
        "original_codex_id": args.codex_id or conversation_id_from_url(args.codex_url),
        "original_codex_thread_url": args.codex_url,
        "current_codex_id": args.codex_id or conversation_id_from_url(args.codex_url),
        "current_codex_thread_url": args.codex_url,
        "identity_assurance": declared_identity_assurance(
            codex_id=args.codex_id or conversation_id_from_url(args.codex_url)
        ),
        "context_contract_status": context_status,
        "current_stage": args.stage or raw.get("current_stage") or default_stage,
        "current_step": args.current_step or raw.get("current_step") or default_step,
        "next_action": args.next_action or raw.get("next_action") or default_next_action,
        "next_command": args.next_command or raw.get("next_command") or default_next_command,
        "repository": _session_repository(repo_identity),
        "parallel_session_acknowledged": bool(args.allow_parallel),
        "session_role": requested_role,
        "model_assignment": model_assignment,
        "work_package_binding": package_binding,
        "session_outcome": None,
        "review_verdict": None,
        "promotion_request": None,
        "delivery_evidence": None,
        "delivery_base_commit": str(repo_identity.get("head") or "") if package_binding and requested_role != "REVIEW_AUTHORITY" else None,
        "review_target": review_target,
        "review_evidence": None,
        "authoritative_review_binding": authoritative_review_binding,
    })
    raw["session_creation_ordinal"] = len(sessions_for_work(args.state_root, work_id)) + 1
    payload = _normalize_payload(raw, session_id=args.session_id, generated_at=timestamp)
    _bind_next_command_preconditions(
        payload, work_record=work, bootstrap_record=bootstrap, repository=repo_identity
    )
    validation = _validate_payload(payload, state_root=args.state_root, record_status="ACTIVE")
    if not validation["valid"]:
        raise ActorError("session validation failed", code="session_validation_failed", problems=validation["errors"], warnings=validation["warnings"])
    source_refs = [
        actor_source_ref("bootstrap", bootstrap),
        actor_source_ref("work", work),
        *(
            [
                {
                    "kind": "work_package",
                    "path": package_binding.get("path"),
                    "sha256": package_binding.get("sha256"),
                }
            ]
            if package_binding
            else []
        ),
    ]
    record = new_record(
        actor=ACTOR_NAME,
        record_type=SESSION_RECORD_TYPE,
        record_id=payload["session_id"],
        status="ACTIVE",
        payload=payload,
        generated_at=timestamp,
        source_refs=source_refs,
        note=args.note,
    )
    resource_accounting, token_capture = capture_start(
        record,
        state_root=args.state_root,
        generated_at=args.generated_at,
    )
    payload["resource_accounting"] = resource_accounting
    record = new_record(
        actor=ACTOR_NAME,
        record_type=SESSION_RECORD_TYPE,
        record_id=payload["session_id"],
        status="ACTIVE",
        payload=payload,
        generated_at=timestamp,
        source_refs=source_refs,
        note=args.note,
    )
    path = _store(args).put(record)
    return _guided(
        {
            "ok": True,
            "created": True,
            "record_path": str(path),
            "session": _summary(record, state_root=args.state_root),
            "token_capture": token_capture,
        },
        record,
        state_root=args.state_root,
    ), 0


def command_start(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    work_id = validate_prefixed_id(args.work_id, WORK_ID_PREFIX, label="work item")
    requested_id = str(args.codex_id or conversation_id_from_url(args.codex_url) or "").strip()
    if not requested_id and args.input:
        raw = read_structured(args.input, name="session")
        requested_id = str(
            raw.get("original_codex_id")
            or raw.get("current_codex_id")
            or conversation_id_from_url(raw.get("original_codex_thread_url"))
            or conversation_id_from_url(raw.get("current_codex_thread_url"))
            or ""
        ).strip()
        if requested_id:
            # Freeze the identity used by the inner normalizer to the same
            # value protected by the thread-attribution lock.
            args.codex_id = requested_id
    with work_lifecycle_lock(args.state_root, work_id):
        if not requested_id:
            return _command_start_locked(args)
        with resource_thread_lock(args.state_root, requested_id):
            require_resource_thread_available(args.state_root, requested_id)
            return _command_start_locked(args)


def command_list(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    rows = []
    store_obj = _store(args)
    for index_row in store_obj.list_read_only():
        if index_row.get("status") == "INVALID" or index_row.get("integrity_classification") == "RECOVERABLE_STALE":
            rows.append(index_row)
            continue
        record = store_obj.get_read_only(str(index_row.get("record_id") or ""))
        summary = _summary(record, state_root=args.state_root, authorize_live=True)
        if args.status and summary["status"] not in args.status:
            continue
        if args.work_id and summary["work_id"] != args.work_id:
            continue
        if args.open_only and summary["status"] not in SESSION_RESUMABLE_STATUSES:
            continue
        rows.append(summary)
    rows.sort(key=lambda row: (str(row.get("updated_at_utc")), str(row.get("session_id"))), reverse=True)
    return with_guidance(
        {"ok": True, "count": len(rows), "sessions": rows},
        status="SESSION_LIST_READY",
        reason_codes=["session_list_ready"],
        next_actions=[{"action": "select_session", "description": "Select one session explicitly when several are resumable.", "command": "python -m tools.eiel sessionManager inspect <SES-ID>"}] if rows else [{"action": "start_session", "description": "No matching session exists; start one for an authorized work item."}],
        related_records=[str(row.get("session_id")) for row in rows],
    ), 0


def command_inspect(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    session_id = validate_prefixed_id(args.session_id, SESSION_ID_PREFIX, label="session")
    record = _store(args).get_read_only(session_id)
    return _guided(
        {"ok": True, "record": record, "session": _summary(record, state_root=args.state_root, authorize_live=True)},
        record,
        state_root=args.state_root,
        authorize_live=True,
    ), 0


def _command_checkpoint_locked(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = _store(args).get(args.session_id)
    payload = dict(record.get("payload") or {})
    token_usage_artifacts: list[dict[str, Any]] = []
    patch: dict[str, Any] = {}
    if args.patch:
        patch.update(read_structured(args.patch, name="session patch"))
    patch.update(parse_set_values(args.set_values))
    requested_patch_roots = {str(key).split(".", 1)[0] for key in patch}
    manager_owned_patch_roots = requested_patch_roots.intersection({
        "schema_version", "session_id", "work_id", "bootstrap_record_id",
        "bootstrap_history", "original_codex_id", "original_codex_thread_url",
        "current_codex_id", "current_codex_thread_url",
        "preferred_continuation_codex_id",
        "preferred_continuation_codex_thread_url", "codex_participants",
        "repository", "parallel_session_acknowledged", "session_role",
        "model_assignment", "work_package_binding", "session_outcome",
        "review_verdict", "promotion_request", "delivery_evidence",
        "delivery_base_commit", "review_target", "review_evidence", "epoch",
        "context_receipt", "resource_accounting", "prepared_assignment_id",
        "scheduler_assignment_id", "scheduler_assignment_digest",
        "scheduler_lease_id", "scheduler_worker_id", "assignment_history",
        "worker_loss", "worker_identity_state", "identity_assurance",
        "session_creation_ordinal", "context_contract_status",
    })
    forbidden = sorted(requested_patch_roots - SESSION_CHECKPOINT_EDITABLE_ROOTS)
    if forbidden:
        raise ActorError(
            "generic Session checkpoint accepts progress fields only",
            code="session_managed_field_patch_forbidden",
            problems=[
                "Fields requiring a bounded owner path: " + ", ".join(forbidden),
                *(
                    ["Recognized manager-owned fields: " + ", ".join(sorted(manager_owned_patch_roots))]
                    if manager_owned_patch_roots
                    else []
                ),
                "Checkpoint-editable fields: " + ", ".join(sorted(SESSION_CHECKPOINT_EDITABLE_ROOTS)),
            ],
            next_actions=[
                {
                    "action": "use_bounded_command",
                    "description": (
                        "Use Session start, identity binding, Scheduler integration, review, "
                        "promotion, context-receipt, epoch, or WRA attachment APIs for authoritative fields."
                    ),
                }
            ],
        )
    if patch:
        payload = apply_dotted_patch(payload, patch)
    for key in ("current_stage", "current_step", "next_action", "next_command", "interrupted_during", "blocked_reason"):
        value = getattr(args, key, None)
        if value is not None:
            payload[key] = value
    if args.completed_step:
        payload["completed_steps"] = normalize_string_list([*normalize_string_list(payload.get("completed_steps")), *args.completed_step])
    if args.success_command:
        payload["last_successful_command"] = {"command": args.success_command, "result": args.success_result, "at_utc": utc_now(args.generated_at)}
    if args.failed_command:
        payload["last_failed_command"] = {"command": args.failed_command, "result": args.failed_result, "at_utc": utc_now(args.generated_at)}
    validation_status = str(getattr(args, "validation_status", None) or "").strip().upper()
    validation_command = str(getattr(args, "validation_command", None) or "").strip()
    validation_gaps = normalize_string_list(getattr(args, "validation_gap", None) or [])
    validation_impact = str(getattr(args, "validation_decision_impact", None) or "").strip() or None
    if validation_status:
        if not validation_command:
            raise ActorError(
                "validation checkpoint requires --validation-command",
                code="session_validation_event_command_missing",
            )
        if validation_status == "PARTIAL" and (not validation_gaps or not validation_impact):
            raise ActorError(
                "PARTIAL validation requires --validation-gap and --validation-decision-impact",
                code="session_validation_partial_incomplete",
            )
        if validation_status == "PASSED" and validation_gaps:
            raise ActorError(
                "PASSED validation may not declare unresolved gaps",
                code="session_validation_contradictory",
            )
        event = {
            "schema_version": "eiel_session_validation_event_v1",
            "status": validation_status,
            "command": validation_command,
            "gaps": validation_gaps,
            "decision_impact": validation_impact,
            "recorded_at_utc": utc_now(args.generated_at),
        }
        payload["validation_events"] = [
            *[dict(row) for row in (payload.get("validation_events") or []) if isinstance(row, Mapping)],
            event,
        ]
    if validation_gaps or validation_impact:
        if not validation_gaps or not validation_impact:
            raise ActorError(
                "validation gaps and decision impact must be recorded together",
                code="session_validation_gap_incomplete",
            )
        payload["validation_gaps"] = validation_gaps
        payload["validation_decision_impact"] = validation_impact
    for field, values in {
        "qualification_record_ids": args.qualification_ref,
        "resource_record_ids": args.resource_ref,
        "evidence_assessment_ids": args.evidence_ref,
        "artifact_paths": args.artifact_ref,
        "report_paths": args.report_ref,
        "librarian_document_ids": args.librarian_document_id,
        "prompt_paths": args.prompt_ref,
        "external_urls": args.external_url,
    }.items():
        if values:
            refs = normalize_reference_map(payload.get("references"), fields=REFERENCE_FIELDS)
            refs[field] = normalize_string_list([*refs[field], *values])
            payload["references"] = refs
    if any(
        value is not None
        for value in (args.tool_call_count, args.compaction_count, args.broad_output_events, args.shell_output_tokens)
    ) or args.handoff_recommended:
        health = dict(payload.get("context_health") or {})
        for key in ("tool_call_count", "compaction_count", "broad_output_events", "shell_output_tokens"):
            value = getattr(args, key, None)
            if value is not None:
                health[key] = value
        health["source"] = "manual"
        health["observed_at_utc"] = utc_now(args.generated_at)
        if args.handoff_recommended:
            health["handoff_recommended"] = True
        payload["context_health"] = normalize_observed_context_health(health)
    if args.epoch_boundary_reason:
        boundary_at = utc_now(args.generated_at)
        candidate = dict(record)
        candidate["payload"] = payload
        resource_accounting, token_usage_artifacts = finalize_active_capture(
            candidate,
            state_root=args.state_root,
            generated_at=boundary_at,
        )
        payload["resource_accounting"] = resource_accounting
        epoch = dict(payload.get("epoch") or {})
        epoch["ended_at_utc"] = boundary_at
        epoch["boundary_reason"] = args.epoch_boundary_reason
        payload["epoch"] = epoch_for_session(
            epoch,
            session_id=str(record.get("record_id")),
            authority_role=str(payload.get("session_role") or "IMPLEMENTATION_AUTHORITY"),
            generated_at=str(record.get("created_at_utc") or utc_now(args.generated_at)),
        )
        health = dict(payload.get("context_health") or {})
        health["handoff_recommended"] = True
        payload["context_health"] = normalize_observed_context_health(health)
    live_repository = git_repository_identity(dict(payload.get("repository") or {}).get("repo_root"))
    payload["repository"] = _session_repository(live_repository)
    work = work_store(args.state_root).get(str(payload.get("work_id") or ""))
    if str(record.get("status") or "") not in SESSION_TERMINAL_STATUSES:
        bootstrap = bootstrap_store(args.state_root).get(str(payload.get("bootstrap_record_id") or ""))
        _bind_next_command_preconditions(
            payload, work_record=work, bootstrap_record=bootstrap, repository=live_repository
        )
    revised = _revise(args, record, payload=payload, status=str(record.get("status")), action="checkpoint", note=args.note)
    if args.context_receipt_output:
        revised, receipt, receipt_path = _generate_and_attach_context_receipt(
            args, revised, output=args.context_receipt_output, action="checkpoint_context_receipt"
        )
        response_body: dict[str, Any] = {
            "ok": True,
            "session": _summary(revised, state_root=args.state_root),
            "context_receipt": receipt,
            "context_receipt_path": str(receipt_path),
        }
        if token_usage_artifacts:
            response_body["token_usage_artifacts"] = token_usage_artifacts
        return _guided(response_body, revised, state_root=args.state_root), 0
    response_body = {"ok": True, "session": _summary(revised, state_root=args.state_root)}
    if token_usage_artifacts:
        response_body["token_usage_artifacts"] = token_usage_artifacts
    return _guided(response_body, revised, state_root=args.state_root), 0


def command_checkpoint(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    initial = _store(args).get_read_only(args.session_id)
    work_id = str(dict(initial.get("payload") or {}).get("work_id") or "")
    with work_lifecycle_lock(args.state_root, work_id):
        return _command_checkpoint_locked(args)


def _generate_context_receipt_for_record(
    args: argparse.Namespace,
    record: Mapping[str, Any],
    *,
    read_only: bool = False,
) -> dict[str, Any]:
    payload = dict(record.get("payload") or {})
    work_records = work_store(args.state_root)
    bootstrap_records = bootstrap_store(args.state_root)
    work_get = work_records.get_read_only if read_only else work_records.get
    bootstrap_get = bootstrap_records.get_read_only if read_only else bootstrap_records.get
    work = work_get(str(payload.get("work_id") or ""))
    bootstrap = bootstrap_get(str(payload.get("bootstrap_record_id") or ""))
    return build_context_receipt(
        work_record=work,
        session_record=record,
        bootstrap_record=bootstrap,
        repo_root=getattr(args, "repo_root", None) or dict(payload.get("repository") or {}).get("repo_root"),
        state_root=args.state_root,
    )


def _generate_and_attach_context_receipt(
    args: argparse.Namespace,
    record: Mapping[str, Any],
    *,
    output: str | Path | None,
    action: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    receipt = _generate_context_receipt_for_record(args, record)
    path = write_context_receipt(receipt, state_root=args.state_root, output=output)
    payload = dict(record.get("payload") or {})
    payload["context_receipt"] = {
        "receipt_id": receipt["receipt_id"],
        "path": str(path),
        "sha256": receipt["receipt_sha256"],
        "generated_at_utc": receipt["generated_at_utc"],
    }
    revised = _revise(
        args,
        record,
        payload=payload,
        status=str(record.get("status")),
        action=action,
        note=getattr(args, "note", None),
        allow_terminal_work=str(record.get("status")) in SESSION_TERMINAL_STATUSES,
    )
    return revised, receipt, path


def _command_context_receipt_locked(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = _store(args).get(args.session_id)
    if args.attach:
        record, receipt, path = _generate_and_attach_context_receipt(
            args, record, output=args.receipt_output, action="attach_context_receipt"
        )
    else:
        receipt = _generate_context_receipt_for_record(args, record)
        path = write_context_receipt(receipt, state_root=args.state_root, output=args.receipt_output)
    validation = validate_context_receipt(
        receipt,
        work_record=work_store(args.state_root).get(str(dict(record.get("payload") or {}).get("work_id") or "")),
        session_record=record,
        repo_root=args.repo_root or dict(dict(record.get("payload") or {}).get("repository") or {}).get("repo_root"),
        state_root=args.state_root,
    )
    if not validation["valid"]:
        raise ActorError(
            "generated Context Receipt did not validate",
            code="context_receipt_invalid",
            problems=validation["errors"],
            warnings=validation["warnings"],
        )
    return _guided(
        {
            "ok": True,
            "session": _summary(record, state_root=args.state_root),
            "context_receipt": receipt,
            "context_receipt_path": str(path),
            "receipt_validation": validation,
        },
        record,
        state_root=args.state_root,
    ), 0


def command_context_receipt(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    initial = _store(args).get_read_only(args.session_id)
    work_id = str(dict(initial.get("payload") or {}).get("work_id") or "")
    with work_lifecycle_lock(args.state_root, work_id):
        return _command_context_receipt_locked(args)


def _command_note_locked(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = _store(args).get(args.session_id)
    payload = dict(record.get("payload") or {})
    payload["notes"] = [*list(payload.get("notes") or []), {"at_utc": utc_now(args.generated_at), "text": args.text}]
    revised = _revise(args, record, payload=payload, status=str(record.get("status")), action="note", note=args.note)
    return _guided({"ok": True, "session": _summary(revised, state_root=args.state_root)}, revised, state_root=args.state_root), 0


def command_note(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    initial = _store(args).get_read_only(args.session_id)
    work_id = str(dict(initial.get("payload") or {}).get("work_id") or "")
    with work_lifecycle_lock(args.state_root, work_id):
        return _command_note_locked(args)


def _status_transition_locked(args: argparse.Namespace, *, status: str, action: str, patch: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    record = _store(args).get(args.session_id)
    payload = dict(record.get("payload") or {})
    work_id = str(payload.get("work_id") or "")
    work = work_store(args.state_root).get(work_id)
    require_open_work(work, action=action)
    payload.update(dict(patch))
    token_usage_artifacts: list[dict[str, Any]] = []
    if status in SESSION_TERMINAL_STATUSES:
        candidate = dict(record)
        candidate["payload"] = payload
        resource_accounting, token_usage_artifacts = finalize_session(
            candidate,
            state_root=args.state_root,
            generated_at=args.generated_at,
        )
        payload["resource_accounting"] = resource_accounting
    live_repository = git_repository_identity(dict(payload.get("repository") or {}).get("repo_root"))
    payload["repository"] = _session_repository(live_repository)
    if status not in SESSION_TERMINAL_STATUSES:
        bootstrap = bootstrap_store(args.state_root).get(str(payload.get("bootstrap_record_id") or ""))
        _bind_next_command_preconditions(
            payload, work_record=work, bootstrap_record=bootstrap, repository=live_repository
        )
    revised = _revise(
        args,
        record,
        payload=payload,
        status=status,
        action=action,
        note=getattr(args, "note", None),
        # Pausing or interrupting only preserves a recovery boundary; it does
        # not authorize another execution.  A stale Bootstrap must therefore
        # not prevent recording the exact state needed to recover safely.
        verify_bootstrap=(
            status not in SESSION_TERMINAL_STATUSES
            and action not in {"pause", "interrupt"}
        ),
    )
    response = _guided({"ok": True, "session": _summary(revised, state_root=args.state_root)}, revised, state_root=args.state_root)
    if token_usage_artifacts:
        response["token_usage_artifacts"] = token_usage_artifacts
    return response, 0


def _status_transition(args: argparse.Namespace, *, status: str, action: str, patch: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    initial = _store(args).get_read_only(args.session_id)
    work_id = str(dict(initial.get("payload") or {}).get("work_id") or "")
    with work_lifecycle_lock(args.state_root, work_id):
        return _status_transition_locked(args, status=status, action=action, patch=patch)


def command_pause(args: argparse.Namespace):
    return _status_transition(args, status="PAUSED", action="pause", patch={"next_action": args.next_action, "next_command": args.next_command})


def command_interrupt(args: argparse.Namespace):
    return _status_transition(args, status="INTERRUPTED", action="interrupt", patch={"interrupted_during": args.interrupted_during, "next_action": args.next_action, "next_command": args.next_command})


def command_block(args: argparse.Namespace):
    return _status_transition(args, status="BLOCKED", action="block", patch={"blocked_reason": args.reason, "blocked_by_work_ids": args.blocked_by, "next_action": args.next_action or "Complete or repair the blocking work.", "next_command": args.next_command})


def command_attach_bootstrap(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = attach_session_bootstrap_after_commit(
        args.session_id,
        state_root=args.state_root,
        bootstrap_record_id=args.bootstrap_record,
        repo_root=args.repo_root,
        generated_at=args.generated_at,
    )
    return with_guidance(
        {"ok": True, "session": _summary(record, state_root=args.state_root)},
        status="BOOTSTRAP_ATTACHED",
        reason_codes=["session_bootstrap_attached"],
        next_actions=[],
        related_records=[str(record.get("record_id") or ""), str(args.bootstrap_record)],
    ), 0


def _resume_context_receipt(
    args: argparse.Namespace,
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, bool]:
    """Resolve or create the exact pre-resume Context Receipt."""

    payload = dict(record.get("payload") or {})
    retained = dict(payload.get("context_receipt") or {})
    generated = False
    if retained.get("path"):
        path = Path(str(retained["path"]))
        if not path.is_file():
            raise ActorError(
                "retained Context Receipt is missing",
                code="session_context_receipt_missing",
                related_records=[str(record.get("record_id") or "")],
            )
        from tools.actors.common import read_json
        receipt = read_json(path, name="Context Receipt")
    else:
        receipt = _generate_context_receipt_for_record(args, record)
        path = write_context_receipt(receipt, state_root=args.state_root)
        generated = True
    work = work_store(args.state_root).get(str(payload.get("work_id") or ""))
    validation = validate_context_receipt(
        receipt,
        work_record=work,
        session_record=record,
        repo_root=args.repo_root or dict(payload.get("repository") or {}).get("repo_root"),
        state_root=args.state_root,
    )
    if not validation.get("valid"):
        raise ActorError(
            "pre-resume Context Receipt is not current",
            code="session_context_receipt_stale",
            problems=list(validation.get("errors") or []),
            warnings=list(validation.get("warnings") or []),
            related_records=[str(record.get("record_id") or ""), str(receipt.get("receipt_id") or "")],
        )
    return receipt, path, generated


def _command_resume_locked(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = _store(args).get(args.session_id)
    payload = dict(record.get("payload") or {})
    resume_receipt, resume_receipt_path, resume_receipt_generated = _resume_context_receipt(args, record)
    previous_epoch_token_artifacts: list[dict[str, Any]] = []
    if payload.get("session_outcome") == "PROMOTION_REQUESTED":
        raise ActorError(
            "a promotion-requested session may not be resumed",
            code="session_promotion_unresolved",
            problems=["Engineering Authority must resolve the higher uncertainty and issue a revised or superseding Work Package."],
            next_actions=[
                {
                    "action": "inspect_work",
                    "description": "Inspect the Work Item and promotion request before creating a replacement session.",
                    "command": f"python -m tools.eiel workManager inspect {payload.get('work_id')}",
                }
            ],
            related_records=[str(record.get("record_id")), str(payload.get("work_id"))],
        )
    work_id = str(payload.get("work_id"))
    readiness = _work_readiness(args.state_root, work_id)
    if readiness["record"].get("status") in WORK_TERMINAL_STATUSES:
        raise ActorError("session work item is terminal", code="session_work_terminal")
    if not readiness["dependencies"]["all_satisfied"]:
        raise ActorError("session work dependencies are unresolved", code="session_work_dependencies_unresolved", problems=[json.dumps(readiness["dependencies"], sort_keys=True)])
    conflicts = _active_session_conflicts(args.state_root, work_id, exclude=str(record["record_id"]))
    if conflicts and not args.allow_parallel:
        raise ActorError("another active session already owns this work item", code="session_active_conflict", problems=conflicts)
    prepared_id = ""
    prepared_takeover = payload.get("worker_identity_state") == "PENDING"
    if prepared_takeover:
        if not getattr(args, "take_over_prepared", False):
            raise ActorError(
                "prepared Session requires explicit manual takeover",
                code="session_prepared_takeover_required",
                next_actions=[{
                    "action": "manual_takeover",
                    "description": "Detach the unstarted Scheduler attempt and resume this Session manually.",
                    "command": (
                        f"python -m tools.eiel sessionManager resume {record.get('record_id')} "
                        "--take-over-prepared --takeover-reason '<reason>' --codex-id '<codex-id>'"
                    ),
                }],
            )
        if not str(args.takeover_reason or "").strip():
            raise ActorError(
                "prepared Session manual takeover requires an explicit reason",
                code="session_takeover_reason_missing",
                related_records=[str(record.get("record_id")), str(payload.get("prepared_assignment_id") or "")],
            )
        prepared_id = str(payload.get("prepared_assignment_id") or "")

    # Validate repository, Bootstrap and requested identity before any durable
    # Scheduler transition.  The previous order superseded READY Assignments
    # and only then discovered a missing takeover reason or stale Bootstrap.
    payload = _refresh_or_validate_bootstrap(args, payload)
    new_url = args.codex_url or payload.get("current_codex_thread_url")
    new_id = args.codex_id or conversation_id_from_url(new_url) or payload.get("current_codex_id")
    if not new_id:
        raise ActorError("resume requires a Codex identity", code="session_codex_identity_missing")
    current_id = str(payload.get("current_codex_id") or "")
    if str(record.get("status")) == "ACTIVE" and str(new_id) != current_id:
        raise ActorError(
            "a different Codex identity may not take over an ACTIVE Session",
            code="session_active_takeover_forbidden",
            problems=[f"Current identity: {current_id}", f"Requested identity: {new_id}"],
            next_actions=[{
                "action": "interrupt_then_resume",
                "description": "The current owner must pause or interrupt the Session before another identity resumes it.",
            }],
            related_records=[str(record.get("record_id"))],
        )
    if str(new_id) != current_id and not str(args.takeover_reason or "").strip():
        raise ActorError(
            "identity takeover requires an explicit reason",
            code="session_takeover_reason_missing",
            related_records=[str(record.get("record_id"))],
        )

    prior_epoch = dict(payload.get("epoch") or {})
    payload["resumed_from"] = {
        "schema_version": "eiel_session_resume_cause_v1",
        "session": actor_record_ref(record),
        "context_receipt_id": str(resume_receipt.get("receipt_id") or ""),
        "context_receipt_sha256": str(resume_receipt.get("receipt_sha256") or ""),
        "context_receipt_path": str(resume_receipt_path),
        "boundary_reason": str(prior_epoch.get("boundary_reason") or record.get("status") or "resume"),
        "interruption_summary": {
            "interrupted_during": payload.get("interrupted_during"),
            "blocked_reason": payload.get("blocked_reason"),
            "last_failed_command": payload.get("last_failed_command"),
            "validation_gaps": list(payload.get("validation_gaps") or []),
            "validation_decision_impact": payload.get("validation_decision_impact"),
        },
    }
    payload["context_receipt"] = {
        "receipt_id": resume_receipt.get("receipt_id"),
        "path": str(resume_receipt_path),
        "sha256": resume_receipt.get("receipt_sha256"),
        "generated_at_utc": resume_receipt.get("generated_at_utc"),
    }
    if prepared_takeover:
        payload["assignment_history"] = [*list(payload.get("assignment_history") or []), {
            "assignment_id": prepared_id,
            "event": "MANUAL_TAKEOVER",
            "reason": str(args.takeover_reason),
            "at_utc": utc_now(args.generated_at),
        }]
        payload["prepared_assignment_id"] = None
        payload["scheduler_assignment_id"] = None
        payload["scheduler_assignment_digest"] = None
        payload["scheduler_lease_id"] = None
        payload["scheduler_worker_id"] = None
        payload["worker_identity_state"] = "DETACHED"
        payload["worker_loss"] = {
            "assignment_id": prepared_id, "lease_id": None, "reason": str(args.takeover_reason),
            "recorded_at_utc": utc_now(args.generated_at), "prepared_only": True,
        }
    if str(new_id) != current_id or (new_url and new_url != payload.get("current_codex_thread_url")):
        payload["codex_participants"] = [
            *list(payload.get("codex_participants") or []),
            {
                "codex_id": str(new_id),
                "thread_url": new_url,
                "joined_at_utc": utc_now(args.generated_at),
                "role": "takeover" if str(new_id) != str(payload.get("original_codex_id")) else "returning_original",
                "reason": args.takeover_reason or "session resumed",
            },
        ]
    payload["current_codex_id"] = str(new_id)
    payload["current_codex_thread_url"] = new_url
    payload["identity_assurance"] = declared_identity_assurance(
        codex_id=str(new_id), source="manual_resume"
    )
    if payload.get("worker_identity_state") == "DETACHED":
        payload["worker_identity_state"] = "ACTIVE"
        payload["original_codex_id"] = payload.get("original_codex_id") or str(new_id)
        payload["preferred_continuation_codex_id"] = str(new_id)
    previous_epoch = dict(payload.get("epoch") or {})
    # Only an ended epoch becomes historical.  A prepared or paused Session
    # may resume inside the same still-open epoch; copying that live epoch into
    # history would silently change Context Epoch cardinality and make the
    # current epoch appear twice.
    if previous_epoch.get("ended_at_utc"):
        if previous_epoch not in list(payload.get("epoch_history") or []):
            payload["epoch_history"] = [*list(payload.get("epoch_history") or []), previous_epoch]
        # Older records may contain an ended Context Epoch whose token segment
        # was never closed.  Do not query today's cumulative counter for that
        # historical boundary; seal the gap as explicit UNAVAILABLE evidence.
        candidate = dict(record)
        candidate["payload"] = payload
        resource_accounting, previous_epoch_token_artifacts = finalize_active_capture(
            candidate,
            state_root=args.state_root,
            generated_at=str(previous_epoch.get("ended_at_utc")),
            unavailable_reason=(
                "Context Epoch ended before WRA-1 recorded its END snapshot; "
                "the historical boundary cannot be reconstructed from current cumulative counters"
            ),
        )
        payload["resource_accounting"] = resource_accounting
        payload["epoch"] = epoch_for_session(
            None,
            session_id=str(record.get("record_id")),
            authority_role=str(payload.get("session_role") or "IMPLEMENTATION_AUTHORITY"),
            generated_at=utc_now(args.generated_at),
        )
        payload["context_health"] = normalize_observed_context_health(None)
        payload["context_receipt"] = None
    payload["interrupted_during"] = None
    payload["blocked_reason"] = None
    payload["blocked_by_work_ids"] = []
    payload["parallel_session_acknowledged"] = bool(args.allow_parallel)
    current_repository = git_repository_identity(dict(payload.get("repository") or {}).get("repo_root"))
    payload["repository"] = _session_repository(current_repository)
    current_work = work_store(args.state_root).get(work_id)
    current_bootstrap = bootstrap_store(args.state_root).get(str(payload.get("bootstrap_record_id") or ""))
    _bind_next_command_preconditions(
        payload, work_record=current_work, bootstrap_record=current_bootstrap, repository=current_repository
    )
    candidate = dict(record)
    candidate["payload"] = payload
    candidate["status"] = "ACTIVE"
    resource_accounting, token_capture = capture_start(
        candidate,
        state_root=args.state_root,
        generated_at=args.generated_at,
    )
    payload["resource_accounting"] = resource_accounting
    precommit_validation = _validate_payload(
        payload,
        state_root=args.state_root,
        record_status="ACTIVE",
    )
    if not precommit_validation["valid"]:
        raise ActorError(
            "resumed Session validation failed before Scheduler takeover commit",
            code="session_validation_failed",
            problems=precommit_validation["errors"],
            warnings=precommit_validation["warnings"],
        )
    if prepared_id:
        try:
            from tools.scheduler.dispatch import supersede_assignment_for_manual_takeover
        except ImportError:
            # Manual recovery remains available if the optional Scheduler
            # package was physically removed after preparing the Session.
            pass
        else:
            supersede_assignment_for_manual_takeover(
                prepared_id,
                state_root=args.state_root,
                reason=str(args.takeover_reason),
                generated_at=args.generated_at,
                detach_session=False,
            )
    revised = _revise(args, record, payload=payload, status="ACTIVE", action="resume", note=args.note)
    response_body: dict[str, Any] = {
        "ok": True,
        "session": _summary(revised, state_root=args.state_root),
        "token_capture": token_capture,
        "resume_context_receipt": {
            "receipt_id": resume_receipt.get("receipt_id"),
            "path": str(resume_receipt_path),
            "generated_for_resume": resume_receipt_generated,
        },
    }
    if previous_epoch_token_artifacts:
        response_body["previous_epoch_token_artifacts"] = previous_epoch_token_artifacts
    return _guided(response_body, revised, state_root=args.state_root), 0


def command_resume(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    initial = _store(args).get(args.session_id)
    initial_payload = dict(initial.get("payload") or {})
    work_id = str(initial_payload.get("work_id") or "")
    requested_id = str(
        args.codex_id
        or conversation_id_from_url(args.codex_url)
        or initial_payload.get("current_codex_id")
        or ""
    ).strip()
    with work_lifecycle_lock(args.state_root, work_id):
        with resource_thread_lock(args.state_root, requested_id):
            require_resource_thread_available(
                args.state_root,
                requested_id,
                exclude_session_id=str(initial.get("record_id") or ""),
            )
            return _command_resume_locked(args)


def _command_complete_locked(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = _store(args).get(args.session_id)
    payload = dict(record.get("payload") or {})
    work_id = str(payload.get("work_id") or "")
    work = work_store(args.state_root).get(work_id)
    require_open_work(work, action="complete Session")
    role = str(payload.get("session_role") or "IMPLEMENTATION_AUTHORITY")
    package_binding = dict(payload.get("work_package_binding") or {})
    default_outcome = ROLE_DEFAULT_OUTCOME.get(role, "SESSION_COMPLETE") if package_binding else "SESSION_COMPLETE"
    outcome = str(args.outcome or default_outcome).strip().upper()
    if outcome == "PROMOTION_REQUESTED":
        raise ActorError(
            "use request-promotion instead of completing a promotion request",
            code="session_promotion_command_required",
        )
    verdict = str(args.review_verdict or "").strip().upper() or None
    delivery_evidence = payload.get("delivery_evidence")
    review_evidence = payload.get("review_evidence")
    if role == "REVIEW_AUTHORITY":
        if outcome != "REVIEW_COMPLETE":
            raise ActorError(
                "Review Authority sessions must complete with REVIEW_COMPLETE",
                code="review_outcome_invalid",
            )
        if not verdict:
            raise ActorError(
                "Review Authority completion requires --review-verdict",
                code="review_verdict_missing",
            )
        target = dict(payload.get("review_target") or {})
        candidate_payload = dict(payload)
        candidate_payload["session_outcome"] = outcome
        candidate_payload["review_verdict"] = verdict
        candidate_record = dict(record)
        candidate_record["payload"] = candidate_payload
        implementation = _ensure_review_assurance(
            state_root=args.state_root,
            work_id=str(payload.get("work_id") or ""),
            review_record=candidate_record,
            review_target=target,
        )
        repo_root = str(dict(payload.get("repository") or {}).get("repo_root") or "")
        current = git_repository_identity(repo_root)
        status = _git(repo_root, "status", "--porcelain=v1", "-z")
        if status.returncode != 0 or status.stdout:
            raise ActorError(
                "review must finish from a clean worktree",
                code="review_worktree_dirty",
                problems=[status.stderr.strip() or "uncommitted or untracked files are present"],
            )
        reviewed_commit = str(target.get("reviewed_commit") or "")
        if str(current.get("head") or "") != reviewed_commit:
            raise ActorError(
                "review completion no longer points at the reviewed implementation commit",
                code="review_commit_mismatch",
                problems=[f"Reviewed: {reviewed_commit}", f"Current HEAD: {current.get('head')}"],
            )
        tests_reviewed = normalize_string_list(args.tests_reviewed)
        evidence_reviewed = normalize_string_list(args.evidence_reviewed)
        acceptance_reviewed = normalize_string_list(args.acceptance_reviewed)
        known_limitations = normalize_string_list(args.known_limitation)
        findings = normalize_string_list(args.review_finding)
        disposition = str(args.repair_disposition or "").strip().upper() or None
        obligation_proof_ref: dict[str, Any] | None = None
        obligation_proof: dict[str, Any] | None = None
        superseded_review_ref: dict[str, Any] | None = None
        superseded_review_id = str(getattr(args, "supersedes_review", None) or "").strip()
        if superseded_review_id:
            prior = session_store(args.state_root).get(superseded_review_id)
            prior_payload = dict(prior.get("payload") or {})
            prior_target = dict(prior_payload.get("review_target") or {})
            if (
                prior.get("status") != "COMPLETED"
                or prior_payload.get("session_role") != "REVIEW_AUTHORITY"
                or str(prior_payload.get("work_id") or "") != str(payload.get("work_id") or "")
                or delivery_session_id(prior_target) != delivery_session_id(target)
                or str(prior_target.get("reviewed_commit") or "") != reviewed_commit
                or str(prior_target.get("reviewed_package_sha256") or "") != str(target.get("reviewed_package_sha256") or "")
            ):
                raise ActorError(
                    "--supersedes-review must identify a completed Review Session for the same delivery",
                    code="review_supersession_invalid",
                    related_records=[superseded_review_id],
                )
            superseded_review_ref = actor_record_ref(prior)
        if verdict == "ACCEPT":
            package_check = verify_work_package_binding(
                package_binding,
                repo_root=repo_root,
                expected_work_id=str(payload.get("work_id") or ""),
                verification_mode=VERIFICATION_EXECUTION_STRICT,
                state_root=args.state_root,
            )
            if not package_check["valid"] or not isinstance(package_check.get("package"), Mapping):
                raise ActorError(
                    "review cannot ACCEPT an invalid Work Package binding",
                    code="work_package_binding_invalid",
                    problems=list(package_check.get("errors") or []),
                    warnings=list(package_check.get("warnings") or []),
                )
            declared_package = dict(package_check["package"])
            if not str(getattr(args, "review_proof", None) or "").strip():
                raise ActorError(
                    "ACCEPT requires --review-proof with PASS results and content-addressed evidence for every obligation",
                    code="review_obligation_proof_missing",
                    problems=["Label-only --acceptance-reviewed/--tests-reviewed/--evidence-reviewed values are not proof."],
                )
            obligation_proof, obligation_proof_ref = retain_and_validate_review_proof(
                args.review_proof,
                state_root=args.state_root,
                work_id=str(payload.get("work_id") or ""),
                review_session_id=str(record.get("record_id") or ""),
                reviewed_commit=reviewed_commit,
                reviewed_package_sha256=str(target.get("reviewed_package_sha256") or ""),
                package=declared_package,
                generated_at=args.generated_at,
            )
            # Legacy coverage strings remain display-only and are derived from
            # the proof when callers omit them.  They can never authorize ACCEPT.
            acceptance_reviewed = acceptance_reviewed or normalize_string_list(declared_package.get("acceptance_criteria"))
            tests_reviewed = tests_reviewed or normalize_string_list(declared_package.get("tests"))
            evidence_reviewed = evidence_reviewed or normalize_string_list(declared_package.get("evidence"))
            known_limitations = sorted(set([*known_limitations, *normalize_string_list(obligation_proof.get("limitations"))]))
        if verdict == "CHANGES_REQUIRED":
            if not findings:
                raise ActorError("CHANGES_REQUIRED requires at least one --review-finding", code="review_findings_missing")
            if disposition != "WITHIN_PACKAGE":
                raise ActorError(
                    "CHANGES_REQUIRED must identify a bounded repair inside the current Work Package",
                    code="review_repair_disposition_invalid",
                    problems=["Use --repair-disposition WITHIN_PACKAGE, or use PROMOTE when a revised package is required."],
                )
        if verdict == "PROMOTE" and disposition != "REVISED_PACKAGE_REQUIRED":
            raise ActorError(
                "PROMOTE requires --repair-disposition REVISED_PACKAGE_REQUIRED",
                code="review_repair_disposition_invalid",
            )
        if verdict in {"REJECT", "BLOCKED", "PROMOTE"} and not findings:
            raise ActorError(f"{verdict} requires at least one --review-finding", code="review_findings_missing")
        implementation_payload = dict(implementation.get("payload") or {})
        review_evidence = {
            "schema_version": REVIEW_EVIDENCE_SCHEMA_VERSION,
            "delivery_session_id": delivery_session_id(target),
            "review_session_id": str(record.get("record_id") or ""),
            "delivering_codex_identities": delivering_codex_identities(target),
            "reviewing_codex_identities": sorted(codex_identities(record)),
            "delivery_identity_assurance": normalize_identity_assurance(
                implementation_payload.get("identity_assurance"),
                fallback_codex_id=str(implementation_payload.get("current_codex_id") or "") or None,
            ),
            "reviewing_identity_assurance": normalize_identity_assurance(
                payload.get("identity_assurance"),
                fallback_codex_id=str(payload.get("current_codex_id") or "") or None,
            ),
            "identity_separation_assurance": (
                "REVIEWER_INDEPENDENCE_UNAVAILABLE"
                if not {value for value in codex_identities(record) if not _is_provisional_codex_id(value)}
                or not {value for value in codex_identities(implementation) if not _is_provisional_codex_id(value)}
                or set(codex_identities(record)).intersection(codex_identities(implementation))
                else "INDEPENDENT_SCHEDULER_BOUND"
                if normalize_identity_assurance(payload.get("identity_assurance"))["level"] == "SCHEDULER_BOUND"
                and normalize_identity_assurance(implementation_payload.get("identity_assurance"))["level"] == "SCHEDULER_BOUND"
                else "DECLARED_SEPARATION"
            ),
            "review_authority": role,
            "reviewed_commit": reviewed_commit,
            "reviewed_package_sha256": str(target.get("reviewed_package_sha256") or ""),
            "reviewed_package_version": target.get("reviewed_package_version"),
            "review_scope": target.get("review_scope") or "PACKAGE",
            "integrated_delivery_lineage": list(target.get("integrated_delivery_lineage") or []),
            "tests_reviewed": tests_reviewed,
            "evidence_reviewed": evidence_reviewed,
            "acceptance_criteria_reviewed": acceptance_reviewed,
            "known_limitations": known_limitations,
            "findings": findings,
            "repair_disposition": disposition,
            "obligation_proof": obligation_proof_ref,
            "delivery_sufficiency": (
                str(dict(obligation_proof or {}).get("implementation_sufficiency") or "UNASSESSED")
            ),
            "acceptance_scope": "WITH_LIMITS" if known_limitations else ("UNCONDITIONAL" if verdict == "ACCEPT" else None),
            "supersedes_review": superseded_review_ref,
            "delivery_model_assignment": implementation_payload.get("model_assignment"),
            "review_model_assignment": payload.get("model_assignment"),
            "decision_snapshot": lifecycle_decision_snapshot(
                work_record=work,
                session_record=record,
                implementation_record=implementation,
                review_record=record,
                package_sha256=str(target.get("reviewed_package_sha256") or ""),
                delivery_commit=reviewed_commit,
            ),
            "reviewed_at_utc": utc_now(args.generated_at),
        }
    elif verdict:
        raise ActorError(
            "--review-verdict is valid only for Review Authority sessions",
            code="review_verdict_role_invalid",
        )
    elif package_binding:
        if bool(getattr(args, "adopt_existing_commit", False)):
            outcome = "EXTERNAL_DELIVERY_ADOPTED"
        allowed_outcomes = {ROLE_DEFAULT_OUTCOME.get(role)}
        if bool(getattr(args, "adopt_existing_commit", False)):
            # Adoption records an already committed, provenance-checked repair;
            # it is not an implementation-authority-only capability.
            allowed_outcomes.add("EXTERNAL_DELIVERY_ADOPTED")
        if role == "IMPLEMENTATION_AUTHORITY":
            allowed_outcomes.add("IMPLEMENTATION_COMPLETE")  # legacy explicit spelling, provenance only
        if outcome not in allowed_outcomes:
            raise ActorError(
                f"{role} session bound to a Work Package must complete with one of {sorted(value for value in allowed_outcomes if value)}",
                code="session_outcome_role_mismatch",
            )
        if outcome in DELIVERY_OUTCOMES:
            delivery_evidence = _repository_delivery_evidence(
                record,
                payload,
                state_root=args.state_root,
                delivery_commit=str(args.delivery_commit or ""),
                adopt_existing_commit=bool(getattr(args, "adopt_existing_commit", False)),
                adoption_reason=getattr(args, "adoption_reason", None),
                external_producer=getattr(args, "external_producer", None),
                generated_at=args.generated_at,
            )
        delivery_evidence["decision_snapshot"] = lifecycle_decision_snapshot(
            work_record=work,
            session_record=record,
            package_sha256=str(package_binding.get("sha256") or ""),
            delivery_commit=delivery_commit_value(delivery_evidence),
        )
        delivery_evidence["recorded_at_utc"] = utc_now(args.generated_at)
    review_progress = payload.get("review_progress")
    repair_lineage = payload.get("repair_lineage")
    if role == "REVIEW_AUTHORITY":
        finding_ids = _review_finding_ids(
            session_id=str(record.get("record_id") or ""),
            findings=normalize_string_list(dict(review_evidence or {}).get("findings")),
        )
        review_progress = load_review_progress({
            "schema_version": REVIEW_PROGRESS_SCHEMA_VERSION,
            "state": "COMPLETED",
            "completed_obligations": sorted(set([
                *normalize_string_list(dict(review_evidence or {}).get("tests_reviewed")),
                *normalize_string_list(dict(review_evidence or {}).get("evidence_reviewed")),
                *normalize_string_list(dict(review_evidence or {}).get("acceptance_criteria_reviewed")),
            ])),
            "remaining_obligations": [],
            "last_finding_id": finding_ids[-1] if finding_ids else None,
            "defer_reason": None,
            "resume_condition": None,
            "recorded_at_utc": utc_now(args.generated_at),
        })
        if finding_ids:
            repair_lineage = load_repair_lineage({
                "schema_version": REPAIR_LINEAGE_SCHEMA_VERSION,
                "source_review_ref": {
                    "schema_version": "eiel_actor_record_ref_v1",
                    "record_id": str(record.get("record_id") or ""),
                    "revision": int(record.get("revision") or 0) + 1,
                    "record_sha256": "pending-terminal-revision",
                    "status": "COMPLETED",
                },
                "finding_ids": finding_ids,
                "repair_session_ref": None,
                "resolution": "OPEN" if verdict != "ACCEPT" else "REPAIRED",
                "waiver_authority": None,
                "rereview_session_ref": None,
                "recorded_at_utc": utc_now(args.generated_at),
            })

    epoch = dict(payload.get("epoch") or {})
    epoch["ended_at_utc"] = utc_now(args.generated_at)
    epoch["boundary_reason"] = f"authority phase completed: {outcome}"
    health = dict(payload.get("context_health") or {})
    health["handoff_recommended"] = True
    resource_accounting, token_usage_artifacts = finalize_session(
        record,
        state_root=args.state_root,
        generated_at=args.generated_at,
    )
    patch = {
        "current_step": args.final_step or "complete",
        "next_action": None,
        "next_command": None,
        "interrupted_during": None,
        "blocked_reason": None,
        "blocked_by_work_ids": [],
        "session_outcome": outcome,
        "review_verdict": verdict,
        "delivery_evidence": delivery_evidence,
        "review_evidence": review_evidence,
        "review_progress": review_progress,
        "repair_lineage": repair_lineage,
        "epoch": epoch_for_session(
            epoch,
            session_id=str(record.get("record_id")),
            authority_role=role,
            generated_at=str(record.get("created_at_utc") or utc_now(args.generated_at)),
        ),
        "context_health": normalize_observed_context_health(health),
        "resource_accounting": resource_accounting,
    }
    response, code = _status_transition_locked(args, status="COMPLETED", action="complete", patch=patch)
    response["token_usage_artifacts"] = token_usage_artifacts
    return response, code


def command_complete(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    initial = _store(args).get_read_only(args.session_id)
    work_id = str(dict(initial.get("payload") or {}).get("work_id") or "")
    with work_lifecycle_lock(args.state_root, work_id):
        return _command_complete_locked(args)


def _command_request_promotion_locked(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = _store(args).get(args.session_id)
    payload = dict(record.get("payload") or {})
    if record.get("status") in SESSION_TERMINAL_STATUSES:
        raise ActorError(
            "terminal sessions cannot request promotion",
            code="session_terminal",
            related_records=[str(record.get("record_id"))],
        )
    package_binding = dict(payload.get("work_package_binding") or {})
    if not package_binding:
        raise ActorError(
            "promotion request requires a bound Work Package",
            code="work_package_binding_missing",
            related_records=[str(record.get("record_id")), str(payload.get("work_id"))],
        )
    reason = str(args.reason or "").strip()
    safe_state = str(args.safe_state or "").strip()
    if not reason or not safe_state:
        raise ActorError(
            "promotion requires a non-empty reason and safe repository state",
            code="promotion_details_missing",
        )
    current_class = str(package_binding.get("uncertainty_class") or "").upper()
    requested_class = str(args.requested_class).upper()
    if not uncertainty_is_promotion(current_class, requested_class):
        raise ActorError(
            "requested uncertainty class must be higher than the Work Package class",
            code="promotion_class_invalid",
            problems=[f"Current: {current_class}; requested: {requested_class}"],
        )
    recommended_authority = str(
        args.recommended_authority
        or ("ARCHITECTURE_AUTHORITY" if requested_class == "U5" else "ENGINEERING_AUTHORITY")
    ).upper()
    request = {
        "current_uncertainty_class": current_class,
        "requested_uncertainty_class": requested_class,
        "reason": reason,
        "affected_files_or_contracts": normalize_string_list(args.affected),
        "safe_repository_state": safe_state,
        "recommended_authority": recommended_authority,
        "requested_at_utc": utc_now(args.generated_at),
    }
    epoch = dict(payload.get("epoch") or {})
    epoch["ended_at_utc"] = utc_now(args.generated_at)
    epoch["boundary_reason"] = "promotion requested"
    health = dict(payload.get("context_health") or {})
    health["handoff_recommended"] = True
    resource_accounting, token_usage_artifacts = finalize_session(
        record,
        state_root=args.state_root,
        generated_at=args.generated_at,
    )
    patch = {
        "session_outcome": "PROMOTION_REQUESTED",
        "promotion_request": request,
        "review_verdict": "PROMOTE" if payload.get("session_role") == "REVIEW_AUTHORITY" else None,
        "blocked_reason": reason,
        "blocked_by_work_ids": [],
        "next_action": "Stop implementation. Return the Work Package and promotion request to Engineering Authority.",
        "next_command": f"python -m tools.eiel workManager inspect {payload.get('work_id')}",
        "interrupted_during": payload.get("current_step") or "bounded implementation",
        "epoch": epoch_for_session(
            epoch,
            session_id=str(record.get("record_id")),
            authority_role=str(payload.get("session_role") or "IMPLEMENTATION_AUTHORITY"),
            generated_at=str(record.get("created_at_utc") or utc_now(args.generated_at)),
        ),
        "context_health": normalize_observed_context_health(health),
        "resource_accounting": resource_accounting,
    }
    response, code = _status_transition_locked(args, status="BLOCKED", action="request_promotion", patch=patch)
    response["token_usage_artifacts"] = token_usage_artifacts
    return response, code


def command_request_promotion(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    initial = _store(args).get_read_only(args.session_id)
    work_id = str(dict(initial.get("payload") or {}).get("work_id") or "")
    with work_lifecycle_lock(args.state_root, work_id):
        return _command_request_promotion_locked(args)


def command_review_progress(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    initial = _store(args).get_read_only(args.session_id)
    payload = dict(initial.get("payload") or {})
    if payload.get("session_role") != "REVIEW_AUTHORITY":
        raise ActorError("review-progress is valid only for Review Authority Sessions", code="review_progress_role_invalid")
    state = str(args.state or "IN_PROGRESS").upper()
    progress = load_review_progress({
        "schema_version": REVIEW_PROGRESS_SCHEMA_VERSION,
        "state": state,
        "completed_obligations": normalize_string_list(args.completed_obligation),
        "remaining_obligations": normalize_string_list(args.remaining_obligation),
        "last_finding_id": args.last_finding_id,
        "defer_reason": args.defer_reason,
        "resume_condition": args.resume_condition,
        "recorded_at_utc": utc_now(args.generated_at),
    })
    status = "PAUSED" if state == "DEFERRED" else str(initial.get("status") or "ACTIVE")
    patch = {"review_progress": progress}
    if state == "DEFERRED":
        patch.update({
            "next_action": args.resume_condition,
            "next_command": None,
            "blocked_reason": args.defer_reason,
        })
    return _status_transition(args, status=status, action="review_progress", patch=patch)


def command_defer_review(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    args.state = "DEFERRED"
    args.completed_obligation = []
    args.remaining_obligation = list(args.remaining_obligation or [])
    args.last_finding_id = None
    args.defer_reason = args.reason
    args.resume_condition = args.resume_condition
    return command_review_progress(args)


def command_discontinue(args: argparse.Namespace):
    return _status_transition(args, status="DISCONTINUED", action="discontinue", patch={"next_action": None, "next_command": None})


def command_obsolete(args: argparse.Namespace):
    return _status_transition(args, status="OBSOLETE", action="obsolete", patch={"next_action": None, "next_command": None})


def _command_reconcile_terminal_parent_locked(args: argparse.Namespace):
    record = _store(args).get(args.session_id)
    payload = dict(record.get("payload") or {})
    work_id = str(payload.get("work_id") or "")
    if not work_id:
        raise ActorError("session has no linked work item", code="session_missing_work")
    try:
        work = work_store(args.state_root).get(work_id)
    except ActorError as exc:
        raise ActorError("session work item is missing", code="session_work_missing", related_records=[work_id], problems=[str(exc)])
    current_status = str(record.get("status") or "ACTIVE")
    if current_status in SESSION_TERMINAL_STATUSES and current_status != args.status:
        raise ActorError("session is already terminal", code="session_already_terminal", related_records=[str(record["record_id"])])
    if str(work.get("status")) not in WORK_TERMINAL_STATUSES:
        raise ActorError(
            "session parent work is not terminal",
            code="session_parent_not_terminal",
            related_records=[work_id],
            problems=[f"Observed work status: {work.get('status')}"],
        )
    refresh_repository = bool(args.bootstrap_record or args.repo_root)
    if refresh_repository:
        # Terminal-parent reconciliation is an administrative closeout, not a
        # continuation attempt.  Preserve the recorded repository and
        # Bootstrap as historical provenance unless the operator explicitly
        # asks to replace them.  A deleted or repointed worktree must not
        # strand a Session after its parent Work is already terminal.
        payload = _refresh_or_validate_bootstrap(args, payload)
    candidate = dict(record)
    candidate["payload"] = payload
    resource_accounting, token_usage_artifacts = finalize_session(
        candidate,
        state_root=args.state_root,
        generated_at=args.generated_at,
    )
    payload.update(
        {
            "terminal_reconciliation": {
                "work_id": work_id,
                "work_status": str(work.get("status")),
                "requested_status": args.status,
                "reason": args.reason,
                "requested_at": utc_now(args.generated_at),
            },
            "next_action": None,
            "next_command": None,
            "interrupted_during": None,
            "blocked_reason": None,
            "resource_accounting": resource_accounting,
        }
    )
    note = (
        f"reconcile terminal parent: {work_id} / reason: {args.reason}"
        + (" [refresh bootstrap metadata]" if refresh_repository else " [retain historical repository metadata]")
    )
    revised = _revise(
        args,
        record,
        payload=payload,
        status=args.status,
        action="reconcile_terminal_parent",
        note=note,
        allow_terminal_work=True,
        verify_bootstrap=refresh_repository,
    )
    response = _guided({"ok": True, "session": _summary(revised, state_root=args.state_root)}, revised, state_root=args.state_root)
    response["token_usage_artifacts"] = token_usage_artifacts
    return response, 0


def command_reconcile_terminal_parent(args: argparse.Namespace):
    initial = _store(args).get_read_only(args.session_id)
    work_id = str(dict(initial.get("payload") or {}).get("work_id") or "")
    with work_lifecycle_lock(args.state_root, work_id):
        return _command_reconcile_terminal_parent_locked(args)


def _command_repair_references_locked(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = _store(args).get(args.session_id)
    if str(record.get("status") or "") not in SESSION_TERMINAL_STATUSES:
        raise ActorError(
            "reference repair is restricted to terminal historical Sessions",
            code="session_reference_repair_requires_terminal",
            problems=[f"Observed Session status: {record.get('status')}"],
        )
    removals = normalize_string_list(args.remove_evidence_ref)
    if not removals and not args.artifact_ref and not args.report_ref:
        raise ActorError(
            "reference repair requires an evidence reference removal or replacement artifact/report path",
            code="session_reference_repair_empty",
        )
    payload = dict(record.get("payload") or {})
    references = normalize_reference_map(payload.get("references"), fields=REFERENCE_FIELDS)
    existing_evidence = references["evidence_assessment_ids"]
    unknown = sorted(set(removals) - set(existing_evidence))
    if unknown:
        raise ActorError(
            "reference repair attempted to remove an unbound evidence reference",
            code="session_reference_repair_reference_missing",
            problems=unknown,
        )
    references["evidence_assessment_ids"] = [value for value in existing_evidence if value not in set(removals)]
    references["artifact_paths"] = normalize_string_list([*references["artifact_paths"], *args.artifact_ref])
    references["report_paths"] = normalize_string_list([*references["report_paths"], *args.report_ref])
    payload["references"] = references
    reason = str(args.reason).strip()
    note = (
        f"administrative reference repair: {reason}; removed evidence refs={removals}; "
        f"added artifact refs={normalize_string_list(args.artifact_ref)}; "
        f"added report refs={normalize_string_list(args.report_ref)}"
    )
    revised = _revise(
        args,
        record,
        payload=payload,
        status=str(record.get("status")),
        action="repair_references",
        note=note,
        allow_terminal_work=True,
    )
    return _guided({"ok": True, "session": _summary(revised, state_root=args.state_root)}, revised, state_root=args.state_root), 0


def command_repair_references(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    initial = _store(args).get_read_only(args.session_id)
    work_id = str(dict(initial.get("payload") or {}).get("work_id") or "")
    with work_lifecycle_lock(args.state_root, work_id):
        return _command_repair_references_locked(args)


def _command_supersede_locked(args: argparse.Namespace):
    replacement = _store(args).get(args.with_session_id)
    if replacement.get("status") in SESSION_TERMINAL_STATUSES:
        raise ActorError("replacement session is terminal", code="session_replacement_terminal")
    replacement_payload = dict(replacement.get("payload") or {})
    current = _store(args).get(args.session_id)
    if str(replacement_payload.get("work_id") or "") != str(dict(current.get("payload") or {}).get("work_id") or ""):
        raise ActorError("replacement Session belongs to another Work Item", code="session_replacement_work_mismatch")
    return _status_transition_locked(args, status="SUPERSEDED", action="supersede", patch={"superseded_by_session_id": replacement["record_id"], "next_action": None, "next_command": None})


def command_supersede(args: argparse.Namespace):
    initial = _store(args).get_read_only(args.session_id)
    work_id = str(dict(initial.get("payload") or {}).get("work_id") or "")
    with work_lifecycle_lock(args.state_root, work_id):
        return _command_supersede_locked(args)


def command_history(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    session_id = validate_prefixed_id(args.session_id, SESSION_ID_PREFIX, label="session")
    rows = _store(args).history(session_id)
    return {"ok": True, "session_id": session_id, "history": rows, "count": len(rows)}, 0


def command_handoff(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = _store(args).get_read_only(args.session_id)
    summary = _summary(record, state_root=args.state_root, authorize_live=True)
    payload = dict(record.get("payload") or {})
    receipt = (
        _generate_context_receipt_for_record(args, record, read_only=True)
        if payload.get("work_package_binding")
        else None
    )
    receipt_path = (
        write_context_receipt(receipt, state_root=args.state_root, output=args.receipt_output)
        if receipt is not None
        else None
    )
    handoff = {
        "schema_version": "eiel_codex_session_handoff_v1",
        **operator_metadata(),
        "session_id": summary["session_id"],
        "work_id": summary["work_id"],
        "mission_id": summary["mission_id"],
        "status": summary["status"],
        "original_codex_id": summary["original_codex_id"],
        "original_codex_thread_url": summary["original_codex_thread_url"],
        "current_codex_id": summary["current_codex_id"],
        "current_codex_thread_url": summary["current_codex_thread_url"],
        "preferred_continuation_codex_id": summary["preferred_continuation_codex_id"],
        "session_role": summary["session_role"],
        "model_selection": summary.get("model_selection"),
        "model_assignment": summary["model_assignment"],
        "work_package_binding": summary["work_package_binding"],
        "session_outcome": summary["session_outcome"],
        "review_verdict": summary["review_verdict"],
        "promotion_request": summary["promotion_request"],
        "delivery_evidence": summary["delivery_evidence"],
        "review_target": summary["review_target"],
        "review_evidence": summary["review_evidence"],
        "repository": summary["repository"],
        "bootstrap_record_id": summary["bootstrap_record_id"],
        "bootstrap_history": summary["bootstrap_history"],
        "current_stage": summary["current_stage"],
        "current_step": summary["current_step"],
        "completed_steps": summary["completed_steps"],
        "last_successful_command": summary["last_successful_command"],
        "last_failed_command": summary["last_failed_command"],
        "validation_events": summary.get("validation_events") or [],
        "validation_gaps": summary.get("validation_gaps") or [],
        "validation_decision_impact": summary.get("validation_decision_impact"),
        "resumed_from": summary.get("resumed_from"),
        "epoch_history": summary.get("epoch_history") or [],
        "terminal_transition": summary.get("terminal_transition"),
        "attempt_disposition": summary.get("attempt_disposition"),
        "artifact_dispositions": summary.get("artifact_dispositions") or [],
        "review_progress": summary.get("review_progress"),
        "repair_lineage": summary.get("repair_lineage"),
        "authority_binding": summary.get("authority_binding"),
        "projection_freshness": summary.get("projection_freshness"),
        "interrupted_during": summary["interrupted_during"],
        "blocked_reason": summary["blocked_reason"],
        "next_action": summary["next_action"],
        "next_command": summary["next_command"],
        "references": summary["references"],
        "notes": summary["notes"],
        "epoch": summary.get("epoch"),
        "context_health": summary.get("context_health"),
        "context_receipt": receipt,
        "context_receipt_path": str(receipt_path) if receipt_path else None,
        "record_sha256": record.get("record_sha256"),
    }
    return _guided(
        {"ok": True, "session": summary, "handoff": handoff},
        record,
        state_root=args.state_root,
        authorize_live=True,
    ), 0


def _human(payload: Mapping[str, Any]) -> str:
    session = dict(payload.get("session") or {}) if isinstance(payload.get("session"), Mapping) else {}
    handoff = dict(payload.get("handoff") or {}) if isinstance(payload.get("handoff"), Mapping) else {}
    receipt = (
        dict(payload.get("context_receipt") or {})
        if isinstance(payload.get("context_receipt"), Mapping)
        else dict(handoff.get("context_receipt") or {})
        if isinstance(handoff.get("context_receipt"), Mapping)
        else {}
    )
    receipt_path = payload.get("context_receipt_path") or handoff.get("context_receipt_path")
    if session:
        lines = [
            f"Session {session.get('session_id')}",
            f"Session status: {session.get('status')}",
            f"Work: {session.get('work_id')}",
            f"Role: {session.get('session_role')}",
            f"Outcome: {session.get('session_outcome') or 'pending'}",
            f"Codex: {session.get('current_codex_id')}",
            f"Next: {session.get('next_action') or 'none'}",
        ]
        if receipt:
            lines.extend(
                [
                    f"Context receipt: {receipt.get('receipt_id')}",
                    f"Receipt SHA-256: {receipt.get('receipt_sha256')}",
                    f"Receipt path: {receipt_path or 'not written'}",
                ]
            )
        return "\n".join(lines) + "\n"
    if "sessions" in payload:
        lines = [f"Sessions: {payload.get('count', 0)}"]
        for row in payload.get("sessions") or []:
            lines.append(f"{row.get('session_id')} {row.get('status')} {row.get('work_id')} {row.get('current_codex_id')}")
        return "\n".join(lines) + "\n"
    return json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"


def build_parser() -> argparse.ArgumentParser:
    return build_session_parser(session_statuses=SESSION_STATUSES, session_outcomes=SESSION_OUTCOMES, review_verdicts=REVIEW_VERDICTS)


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw:
        raw = ["list", "--open-only"]
    args: argparse.Namespace | None = None
    try:
        args = build_parser().parse_args(raw)
        if args.command == "status":
            if args.session_id:
                args.command = "inspect"
                payload, code = command_inspect(args)
            else:
                args.command = "list"
                args.work_id = None
                args.status = None
                args.open_only = True
                payload, code = command_list(args)
        else:
            handlers = {
                "start": command_start,
                "prepare-assignment": command_prepare_assignment,
                "activate-assignment": command_activate_assignment,
                "attach-bootstrap": command_attach_bootstrap,
                "worker-loss": command_worker_loss,
                "list": command_list,
                "inspect": command_inspect,
                "checkpoint": command_checkpoint,
                "context-receipt": command_context_receipt,
                "note": command_note,
                "pause": command_pause,
                "interrupt": command_interrupt,
                "block": command_block,
                "request-promotion": command_request_promotion,
                "resume": command_resume,
                "complete": command_complete,
                "review-progress": command_review_progress,
                "defer-review": command_defer_review,
                "discontinue": command_discontinue,
                "obsolete": command_obsolete,
                "reconcile-terminal-parent": command_reconcile_terminal_parent,
                "supersede": command_supersede,
                "repair-references": command_repair_references,
                "history": command_history,
                "handoff": command_handoff,
            }
            payload, code = handlers[args.command](args)
        emit_payload(payload, output=args.output, human_text=_human(payload), json_output=not args.human)
        return code
    except ActorError as exc:
        command = raw[0] if raw else "unknown"
        emit_actor_error(
            exc,
            actor=ACTOR_NAME,
            command=command,
            raw_args=raw,
            parsed_args=args,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
