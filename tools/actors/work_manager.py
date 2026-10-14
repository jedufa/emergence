"""Persistent portfolio manager for EIEL engineering and scientific work.

Work Manager records what work exists across many concurrent or paused tracks.
It owns work-item lifecycle and dependency state only.  It does not execute
commands, launch Codex, run campaigns, qualify routes, or make scientific
choices.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from tools.actors.bootstrap import git_repository_identity
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.actors.acceptance import (
    WORK_TERMINAL_DECISION_RECEIPT_SCHEMA_VERSION,
    load_review_proof_artifact,
    load_work_terminal_decision_receipt,
    retain_work_terminal_decision_receipt,
)
from tools.actors.architecture_simplification import canonical_work_lifecycle
from tools.actors.common import (
    ActorError,
    actor_state_root,
    apply_dotted_patch,
    canonical_sha256,
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
    utc_now,
    with_guidance,
)
from tools.actors.closure_contracts import (
    DELIVERY_COMPONENT_SCHEMA_VERSION,
    WORK_EVIDENCE_REQUIREMENT_SCHEMA_VERSION,
    WORK_HOLD_SCHEMA_VERSION,
    load_delivery_component,
    load_work_evidence_requirement,
    load_work_hold,
)
from tools.actors.delivery_semantics import (
    delivery_commit as delivery_commit_value,
    delivery_session_id,
    delivery_sufficiency,
)
from tools.actors.lifecycle_policy import (
    AUTHORITATIVE_REVIEW_BINDING_SCHEMA_VERSION,
    actor_record_ref,
    work_execution_ref,
    lifecycle_decision_snapshot,
    require_record_ref,
    verify_lifecycle_decision_snapshot,
)
from tools.actors.portfolio import (
    PortfolioSnapshot,
    WORK_ID_PREFIX,
    WORK_OPEN_STATUSES,
    WORK_RECORD_TYPE,
    WORK_TERMINAL_STATUSES,
    SESSION_ACTIVE_STATUSES,
    SESSION_TERMINAL_STATUSES,
    build_portfolio_snapshot,
    dependency_graph,
    dependency_state,
    evidence_store,
    find_dependency_cycle,
    normalize_reference_map,
    normalize_string_list,
    qualification_store,
    sessions_for_work,
    session_store,
    validate_prefixed_id,
    work_lifecycle_lock,
    work_lifecycle_locks,
    work_store,
    work_writer_store,
)
from tools.actors.repository_truth import (
    DELIVERY_PROOF_SCHEMA_VERSION,
    INTEGRATION_RECEIPT_SCHEMA_VERSION,
    capture_integration_receipt,
    load_repository_truth_artifact,
)
from tools.actors.release import operator_metadata
from tools.actors.resource_accounting import (
    WORK_ACCOUNTING_SCHEMA_VERSION,
    aggregate_work_usage,
    load_resource_estimate,
    load_work_resource_accounting,
    normalize_resource_estimate,
    normalize_work_resource_accounting,
    validate_resource_estimate,
    work_accounting_projection,
)
from tools.actors.work_cli import build_parser as build_work_parser
from tools.actors.work_package import (
    LEGACY_HASH_KIND,
    WORK_PACKAGE_BINDING_SCHEMA_VERSION,
    VERIFICATION_EXECUTION_STRICT,
    VERIFICATION_HISTORICAL_AUDIT,
    archive_work_package_contract,
    bind_work_package,
    delivery_recorded_at,
    normalize_work_package_chain,
    normalize_work_package_binding,
    resolve_delivery_lineage,
    review_recorded_at,
    verify_work_package_binding,
    work_package_material_equal,
)

ACTOR_NAME = "workManager"
WORK_SCHEMA_VERSION = "eiel_work_item_v1"
WORK_ACCEPTANCE_EVIDENCE_SCHEMA_VERSION = "eiel_work_acceptance_evidence_v2"
WORK_ACCEPTANCE_EVIDENCE_LEGACY_SCHEMA_VERSION = "eiel_work_acceptance_evidence_v1"
WORK_QUALIFICATION_EVIDENCE_SCHEMA_VERSION = "eiel_work_qualification_acceptance_evidence_v1"
WORK_COMPLETION_POLICIES = ("REPOSITORY_DELIVERY", "QUALIFICATION_DECISION", "RECORD_ONLY")
WORK_COMPLETION_BASES = (
    "ACCEPTED_REPOSITORY_DELIVERY",
    "ADMITTED_TECHNICAL_QUALIFICATION",
    "RECORD_ONLY_COMPLETION",
)
WORK_TYPES = (
    "ENGINEERING",
    "SCIENCE",
    "QUALIFICATION",
    "INFRASTRUCTURE",
    "DOCUMENTATION",
    "EVIDENCE",
    "OPERATIONS",
    "RESEARCH",
    "OTHER",
)
WORK_STATUSES = (
    "BACKLOG",
    "READY",
    "ACTIVE",
    "PAUSED",
    "BLOCKED",
    "COMPLETED",
    "DISCONTINUED",
    "OBSOLETE",
    "SUPERSEDED",
)
PRIORITIES = ("P0", "P1", "P2", "P3", "P4")
WORK_PAYLOAD_FIELDS = frozenset({
    "schema_version", "work_id", "title", "summary", "work_type", "subsystem",
    "priority", "mission_id", "parent_work_id", "dependency_work_ids", "tags",
    "current_stage", "current_step", "last_completed_step", "next_action",
    "next_command", "blocked_reason", "preferred_codex_conversation_id",
    "preferred_codex_thread_url", "superseded_by_work_id", "work_package_binding",
    "accepted_review_session_id", "acceptance_evidence",
    "administrative_reconciliation", "references", "resource_estimate",
    "resource_accounting", "created_order_key", "completion_policy",
    "completion_basis", "completion_reason", "repository_integration_receipt",
    "terminal_decision_receipt", "acceptance_scope", "accepted",
    "package_binding_history", "promotion_resolution", "terminal_transition",
    "evidence_requirement", "hold", "delivery_components",
    "continuation", "storage_evidence_roots", "storage_merge_closeout",
})
PRIORITY_RANK = {value: index for index, value in enumerate(PRIORITIES)}
ALLOWED_TRANSITIONS = {
    "BACKLOG": {"BACKLOG", "READY", "ACTIVE", "PAUSED", "BLOCKED", "DISCONTINUED", "OBSOLETE", "SUPERSEDED"},
    "READY": {"READY", "ACTIVE", "PAUSED", "BLOCKED", "DISCONTINUED", "OBSOLETE", "SUPERSEDED"},
    "ACTIVE": {"ACTIVE", "PAUSED", "BLOCKED", "COMPLETED", "DISCONTINUED", "OBSOLETE", "SUPERSEDED"},
    "PAUSED": {"PAUSED", "READY", "ACTIVE", "BLOCKED", "DISCONTINUED", "OBSOLETE", "SUPERSEDED"},
    "BLOCKED": {"BLOCKED", "READY", "ACTIVE", "PAUSED", "DISCONTINUED", "OBSOLETE", "SUPERSEDED"},
    "COMPLETED": {"COMPLETED", "SUPERSEDED"},
    "DISCONTINUED": {"DISCONTINUED", "SUPERSEDED"},
    "OBSOLETE": {"OBSOLETE", "SUPERSEDED"},
    "SUPERSEDED": {"SUPERSEDED"},
}
REFERENCE_FIELDS = (
    "qualification_record_ids",
    "resource_record_ids",
    "evidence_assessment_ids",
    "artifact_paths",
    "report_paths",
    "librarian_document_ids",
    "external_urls",
)

# Generic update is metadata-only.  Lifecycle, identity, package, dependency,
# acceptance, supersession, estimate, and accounting truth each have their own
# bounded writer.
WORK_UPDATE_EDITABLE_ROOTS = frozenset({
    "title",
    "summary",
    "subsystem",
    "priority",
    "mission_id",
    "tags",
    "current_stage",
    "current_step",
    "last_completed_step",
    "next_action",
    "next_command",
    "preferred_codex_conversation_id",
    "preferred_codex_thread_url",
    "references",
    "storage_evidence_roots",
})


def _store(args: argparse.Namespace):
    return work_store(args.state_root, write_scope=True)


def persist_work_record_fixture(
    record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    allow_update: bool = False,
) -> Path:
    """Test-only raw fixture seam retained inside the canonical Work owner."""

    require_test_fixture_context()
    return work_store(state_root, write_scope=True).put(record, allow_update=allow_update)


def load_work_acceptance_evidence(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ActorError("Work acceptance evidence must be an object", code="schema_record_malformed")
    schema = require_explicit_schema_version(
        raw,
        expected=WORK_ACCEPTANCE_EVIDENCE_SCHEMA_VERSION,
        supported=(WORK_ACCEPTANCE_EVIDENCE_LEGACY_SCHEMA_VERSION,),
        label="Work acceptance evidence",
    )
    if schema == WORK_ACCEPTANCE_EVIDENCE_SCHEMA_VERSION:
        required = (
            "decision_snapshot", "accepted_review", "accepted_delivery",
            "obligation_proof", "repository_integration_receipt",
            "acceptance_scope", "known_limitations", "recorded_at_utc",
        )
        missing = [field for field in required if field not in raw]
        if missing:
            raise ActorError(
                "Work acceptance evidence is malformed",
                code="schema_record_malformed",
                problems=[f"missing field: {field}" for field in missing],
            )
        scope = str(raw.get("acceptance_scope") or "")
        if scope not in {"UNCONDITIONAL", "WITH_LIMITS"}:
            raise ActorError(
                "Work acceptance evidence has an invalid acceptance scope",
                code="schema_record_malformed",
            )
        limitations = raw.get("known_limitations")
        if not isinstance(limitations, list):
            raise ActorError(
                "Work acceptance evidence known_limitations must be an array",
                code="schema_record_malformed",
            )
        if scope == "WITH_LIMITS" and not limitations:
            raise ActorError(
                "WITH_LIMITS acceptance requires at least one retained limitation",
                code="schema_record_malformed",
            )
        if scope == "UNCONDITIONAL" and limitations:
            raise ActorError(
                "UNCONDITIONAL acceptance cannot retain hidden limitations",
                code="schema_record_malformed",
            )
    return dict(raw)


def load_work_payload(raw: Any) -> dict[str, Any]:
    """Strict read-only loader for one persisted Work Item payload."""

    if not isinstance(raw, Mapping):
        raise ActorError("Work Item payload must be an object", code="schema_record_malformed")
    require_explicit_schema_version(raw, expected=WORK_SCHEMA_VERSION, label="Work Item")
    require_known_fields(raw, allowed=WORK_PAYLOAD_FIELDS, label="Work Item payload")
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
    estimate = value.get("resource_estimate")
    loaded_estimate = load_resource_estimate(estimate) if estimate is not None else normalize_resource_estimate(None)
    accounting = value.get("resource_accounting")
    if accounting is not None:
        load_work_resource_accounting(accounting, estimate=loaded_estimate)
    evidence = value.get("acceptance_evidence")
    if evidence is not None:
        if not isinstance(evidence, Mapping):
            raise ActorError("Work acceptance evidence must be an object", code="schema_record_malformed")
        if evidence.get("schema_version") == WORK_QUALIFICATION_EVIDENCE_SCHEMA_VERSION:
            require_known_fields(
                evidence,
                allowed={
                    "schema_version", "qualification_record", "decision_status",
                    "source_commit", "repository_state_sha256", "acceptance_scope",
                    "limitations", "evidence_bindings", "recorded_at_utc",
                },
                label="Work qualification acceptance evidence",
            )
            for field in (
                "qualification_record", "decision_status", "source_commit",
                "repository_state_sha256", "acceptance_scope", "limitations",
                "evidence_bindings", "recorded_at_utc",
            ):
                if field not in evidence:
                    raise ActorError(
                        "Work qualification acceptance evidence is malformed",
                        code="schema_record_malformed",
                        problems=[f"missing field: {field}"],
                    )
        else:
            load_work_acceptance_evidence(evidence)
    if value.get("evidence_requirement") is not None:
        load_work_evidence_requirement(value.get("evidence_requirement"))
    if value.get("hold") is not None:
        load_work_hold(value.get("hold"))
    for row in value.get("delivery_components") or []:
        load_delivery_component(row)
    terminal = value.get("terminal_decision_receipt")
    if terminal is not None:
        require_explicit_schema_version(
            terminal,
            expected="eiel_repository_truth_artifact_ref_v1",
            label="Work terminal decision receipt reference",
        )
    closeout = value.get("storage_merge_closeout")
    if closeout is not None:
        require_explicit_schema_version(
            closeout,
            expected="eiel_repository_truth_artifact_ref_v1",
            label="Work storage merge closeout reference",
        )
    return value


def _verify_work_binding_at_available_root(
    binding: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    work_id: str,
    repo_root: str | Path | None,
    read_only: bool,
    allow_historical_package_binding: bool = False,
) -> dict[str, Any]:
    roots: list[str] = []
    for candidate in (
        repo_root,
        dict(binding).get("bound_repo_root"),
        *(
            dict(dict(row.get("payload") or {}).get("repository") or {}).get("repo_root")
            for row in sessions_for_work(state_root, work_id, read_only=read_only)
        ),
    ):
        value = str(candidate or "").strip()
        if value and value not in roots:
            roots.append(value)
    results: list[dict[str, Any]] = []
    for root in roots or [None]:
        result = verify_work_package_binding(
            binding,
            repo_root=root,
            expected_work_id=work_id,
            verification_mode=(
                VERIFICATION_HISTORICAL_AUDIT
                if read_only or allow_historical_package_binding
                else VERIFICATION_EXECUTION_STRICT
            ),
            state_root=state_root,
        )
        results.append(result)
        if result["valid"]:
            result["resolved_repo_root"] = root
            return result
    result = results[0]
    if len(results) > 1:
        result = dict(result)
        result["errors"] = [
            *list(result.get("errors") or []),
            "Work Package did not verify at any known repository root: " + ", ".join(roots),
        ]
    return result



def _normalize_payload(raw: Mapping[str, Any], *, work_id: str | None, generated_at: str) -> dict[str, Any]:
    if "schema_version" in raw:
        require_explicit_schema_version(raw, expected=WORK_SCHEMA_VERSION, label="Work Item")
    payload = dict(raw)
    payload["schema_version"] = WORK_SCHEMA_VERSION
    payload["title"] = str(payload.get("title") or "").strip()
    payload["summary"] = str(payload.get("summary") or "").strip()
    payload["work_type"] = str(payload.get("work_type") or "ENGINEERING").upper()
    default_completion_policy = (
        "REPOSITORY_DELIVERY"
        if payload.get("work_package_binding")
        else "QUALIFICATION_DECISION"
        if payload["work_type"] == "QUALIFICATION"
        else "RECORD_ONLY"
    )
    payload["completion_policy"] = str(payload.get("completion_policy") or default_completion_policy).strip().upper()
    payload["completion_basis"] = str(payload.get("completion_basis") or "").strip().upper() or None
    payload["completion_reason"] = str(payload.get("completion_reason") or "").strip() or None
    payload["repository_integration_receipt"] = dict(payload.get("repository_integration_receipt") or {}) if isinstance(payload.get("repository_integration_receipt"), Mapping) else None
    payload["terminal_decision_receipt"] = dict(payload.get("terminal_decision_receipt") or {}) if isinstance(payload.get("terminal_decision_receipt"), Mapping) else None
    payload["package_binding_history"] = [
        dict(row) for row in (payload.get("package_binding_history") or []) if isinstance(row, Mapping)
    ]
    payload["promotion_resolution"] = (
        dict(payload.get("promotion_resolution") or {})
        if isinstance(payload.get("promotion_resolution"), Mapping) else None
    )
    payload["terminal_transition"] = (
        dict(payload.get("terminal_transition") or {})
        if isinstance(payload.get("terminal_transition"), Mapping) else None
    )
    payload["storage_evidence_roots"] = sorted(
        normalize_string_list(payload.get("storage_evidence_roots"))
    )
    payload["storage_merge_closeout"] = (
        dict(payload.get("storage_merge_closeout") or {})
        if isinstance(payload.get("storage_merge_closeout"), Mapping)
        else None
    )
    payload["evidence_requirement"] = (
        load_work_evidence_requirement(payload.get("evidence_requirement"))
        if payload.get("evidence_requirement") is not None else None
    )
    payload["hold"] = (
        load_work_hold(payload.get("hold")) if payload.get("hold") is not None else None
    )
    payload["delivery_components"] = [
        load_delivery_component(row) for row in (payload.get("delivery_components") or [])
    ]
    payload["acceptance_scope"] = str(payload.get("acceptance_scope") or "").strip().upper() or None
    payload["accepted"] = bool(payload.get("accepted"))
    payload["subsystem"] = str(payload.get("subsystem") or "").strip() or None
    payload["priority"] = str(payload.get("priority") or "P2").upper()
    payload["mission_id"] = str(payload.get("mission_id") or "").strip().upper() or None
    payload["parent_work_id"] = str(payload.get("parent_work_id") or "").strip().upper() or None
    # Work dependencies and tags are set-valued contracts.  Their order does
    # not change readiness or classification, so retain one canonical order in
    # the durable actor record instead of manufacturing revisions from array
    # presentation alone.
    payload["dependency_work_ids"] = sorted(normalize_string_list(payload.get("dependency_work_ids")))
    payload["tags"] = sorted(normalize_string_list(payload.get("tags")))
    payload["current_stage"] = str(payload.get("current_stage") or "").strip() or None
    payload["current_step"] = str(payload.get("current_step") or "").strip() or None
    payload["last_completed_step"] = str(payload.get("last_completed_step") or "").strip() or None
    payload["next_action"] = str(payload.get("next_action") or "").strip() or None
    payload["next_command"] = str(payload.get("next_command") or "").strip() or None
    payload["blocked_reason"] = str(payload.get("blocked_reason") or "").strip() or None
    payload["preferred_codex_conversation_id"] = str(payload.get("preferred_codex_conversation_id") or "").strip() or None
    payload["preferred_codex_thread_url"] = str(payload.get("preferred_codex_thread_url") or "").strip() or None
    payload["superseded_by_work_id"] = str(payload.get("superseded_by_work_id") or "").strip().upper() or None
    payload["work_package_binding"] = normalize_work_package_binding(payload.get("work_package_binding"))
    chain_source = payload.get("work_package_chain")
    fallback_binding = payload.get("work_package_binding")
    if (
        isinstance(chain_source, list)
        and len(chain_source) == 1
        and isinstance(fallback_binding, Mapping)
        and isinstance(chain_source[0], Mapping)
    ):
        chain_binding = chain_source[0].get("binding") if isinstance(chain_source[0].get("binding"), Mapping) else chain_source[0]
        if str(dict(chain_binding).get("sha256") or "") != str(dict(fallback_binding).get("sha256") or ""):
            chain_source = None
    payload["work_package_chain"] = normalize_work_package_chain(
        chain_source, fallback=fallback_binding
    )
    if payload["work_package_binding"]:
        payload["completion_policy"] = "REPOSITORY_DELIVERY"
    payload["accepted_review_session_id"] = str(payload.get("accepted_review_session_id") or "").strip().upper() or None
    payload["acceptance_evidence"] = dict(payload.get("acceptance_evidence") or {}) if isinstance(payload.get("acceptance_evidence"), Mapping) else None
    payload["administrative_reconciliation"] = dict(payload.get("administrative_reconciliation") or {}) if isinstance(payload.get("administrative_reconciliation"), Mapping) else None
    payload["continuation"] = (
        dict(payload.get("continuation") or {})
        if isinstance(payload.get("continuation"), Mapping) else None
    )
    payload["lifecycle_repair"] = dict(payload.get("lifecycle_repair") or {}) if isinstance(payload.get("lifecycle_repair"), Mapping) else None
    payload["references"] = normalize_reference_map(payload.get("references"), fields=REFERENCE_FIELDS)
    payload["resource_estimate"] = normalize_resource_estimate(
        payload.get("resource_estimate") if isinstance(payload.get("resource_estimate"), Mapping) else None,
        generated_at=generated_at,
    )
    payload["resource_accounting"] = normalize_work_resource_accounting(
        payload.get("resource_accounting") if isinstance(payload.get("resource_accounting"), Mapping) else None,
        estimate=payload["resource_estimate"],
    )
    payload["created_order_key"] = str(payload.get("created_order_key") or generated_at)
    identity = {
        "title": payload["title"],
        "work_type": payload["work_type"],
        "created_order_key": payload["created_order_key"],
        "parent_work_id": payload["parent_work_id"],
    }
    resolved = validate_prefixed_id(work_id, WORK_ID_PREFIX, label="work item") if work_id else stable_id("WORK", identity)
    payload["work_id"] = resolved
    return payload


def _validate_payload(
    payload: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    existing_id: str | None = None,
    status: str | None = None,
    read_only: bool = False,
    repo_root: str | Path | None = None,
    snapshot: PortfolioSnapshot | None = None,
    allow_historical_package_binding: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    # An administrative terminal reconciliation preserves history without
    # claiming delivery acceptance.  Its immutable contract receipt is the
    # authoritative evidence when the historical checkout has since gone away.
    historical_package_binding = allow_historical_package_binding or (
        str(status or "").upper() in {"DISCONTINUED", "OBSOLETE"}
        and isinstance(payload.get("administrative_reconciliation"), Mapping)
    )
    if payload.get("schema_version") != WORK_SCHEMA_VERSION:
        errors.append("unsupported work schema_version")
    if not str(payload.get("title") or "").strip():
        errors.append("title is required")
    if str(payload.get("work_type")) not in WORK_TYPES:
        errors.append(f"work_type must be one of {', '.join(WORK_TYPES)}")
    completion_policy = str(payload.get("completion_policy") or "")
    if completion_policy not in WORK_COMPLETION_POLICIES:
        errors.append("completion_policy must be one of " + ", ".join(WORK_COMPLETION_POLICIES))
    if payload.get("work_package_binding") and completion_policy != "REPOSITORY_DELIVERY":
        errors.append("Work Package binding requires completion_policy REPOSITORY_DELIVERY")
    completion_basis = str(payload.get("completion_basis") or "")
    if completion_basis and completion_basis not in WORK_COMPLETION_BASES:
        errors.append("completion_basis is invalid")
    if str(status or "").upper() == "COMPLETED":
        if not completion_basis:
            errors.append("COMPLETED Work requires an explicit completion_basis")
        if completion_policy == "REPOSITORY_DELIVERY" and completion_basis != "ACCEPTED_REPOSITORY_DELIVERY":
            errors.append("repository-delivery Work may complete only from accepted repository delivery")
        if completion_policy == "QUALIFICATION_DECISION" and completion_basis != "ADMITTED_TECHNICAL_QUALIFICATION":
            errors.append("qualification Work may complete only from an admitted technical qualification decision")
        if completion_policy == "RECORD_ONLY" and completion_basis != "RECORD_ONLY_COMPLETION":
            errors.append("record-only Work has an invalid completion basis")
        if payload.get("terminal_decision_receipt") is None:
            errors.append("COMPLETED Work requires an immutable terminal_decision_receipt")
    evidence_requirement = payload.get("evidence_requirement")
    if evidence_requirement is not None:
        try:
            requirement = load_work_evidence_requirement(evidence_requirement)
        except ActorError as exc:
            errors.append(str(exc))
            requirement = None
        if requirement and requirement.get("mode") == "GATE" and str(status or "").upper() == "COMPLETED":
            store_obj = evidence_store(state_root)
            for assessment_id in requirement.get("required_assessment_ids") or []:
                try:
                    assessment = (store_obj.get_read_only if read_only else store_obj.get)(assessment_id)
                except ActorError as exc:
                    errors.append(f"required evidence assessment {assessment_id} is unavailable: {exc}")
                    continue
                decision = str(dict(assessment.get("payload") or {}).get("procedural_decision") or "").upper()
                if decision not in set(requirement.get("accepted_decisions") or []):
                    errors.append(
                        f"required evidence assessment {assessment_id} decision {decision or '<missing>'} "
                        "does not satisfy the Work evidence gate"
                    )
    hold = payload.get("hold")
    if hold is not None:
        try:
            loaded_hold = load_work_hold(hold)
        except ActorError as exc:
            errors.append(str(exc))
            loaded_hold = None
        if loaded_hold and str(status or "").upper() not in {"PAUSED", "BLOCKED"}:
            errors.append("Work with an active hold must remain PAUSED or BLOCKED")
    for index, component in enumerate(payload.get("delivery_components") or []):
        try:
            loaded_component = load_delivery_component(component)
        except ActorError as exc:
            errors.append(f"delivery_components[{index}]: {exc}")
            continue
        try:
            session = (session_store(state_root).get_read_only if read_only else session_store(state_root).get)(
                str(loaded_component.get("source_session_id") or "")
            )
            if str(dict(session.get("payload") or {}).get("work_id") or "") != str(payload.get("work_id") or ""):
                errors.append(f"delivery_components[{index}] belongs to another Work Item")
        except ActorError as exc:
            errors.append(f"delivery_components[{index}] source Session is unavailable: {exc}")
    if str(payload.get("priority")) not in PRIORITIES:
        errors.append(f"priority must be one of {', '.join(PRIORITIES)}")
    if payload.get("resource_estimate") is None:
        warnings.append("legacy Work record has no WRA-1 resource estimate; defaults are projected read-only")
    else:
        errors.extend(validate_resource_estimate(payload.get("resource_estimate")))
    accounting = payload.get("resource_accounting")
    if accounting is None:
        warnings.append("legacy Work record has no WRA-1 accounting projection")
    elif not isinstance(accounting, Mapping):
        errors.append("resource_accounting must be an object")
    elif dict(accounting).get("schema_version") != WORK_ACCOUNTING_SCHEMA_VERSION:
        errors.append("unsupported Work resource_accounting schema_version")
    work_id = str(payload.get("work_id") or "")
    try:
        validate_prefixed_id(work_id, WORK_ID_PREFIX, label="work item")
    except ActorError as exc:
        errors.append(str(exc))
    dependencies = normalize_string_list(payload.get("dependency_work_ids"))
    if work_id in dependencies:
        errors.append("work item cannot depend on itself")
    ws = work_store(state_root)
    getter = ws.get_read_only if read_only else ws.get
    for dependency in dependencies:
        try:
            validate_prefixed_id(dependency, WORK_ID_PREFIX, label="dependency work item")
            if snapshot is not None:
                if dependency not in snapshot.work_by_id:
                    raise ActorError(f"unknown work record: {dependency}")
            else:
                getter(dependency)
        except ActorError as exc:
            errors.append(str(exc))
    parent = payload.get("parent_work_id")
    if parent:
        try:
            validate_prefixed_id(str(parent), WORK_ID_PREFIX, label="parent work item")
            if snapshot is not None:
                if str(parent) not in snapshot.work_by_id:
                    raise ActorError(f"unknown work record: {parent}")
            else:
                getter(str(parent))
        except ActorError as exc:
            errors.append(str(exc))
    mission = payload.get("mission_id")
    if mission and not str(mission).startswith("MIS-"):
        errors.append("mission_id must start with MIS-")
    package_binding = payload.get("work_package_binding")
    try:
        package_chain = normalize_work_package_chain(
            payload.get("work_package_chain"), fallback=package_binding
        )
    except ActorError as exc:
        errors.extend([str(exc), *list(exc.problems or [])])
        package_chain = []
    if (
        package_binding
        and len(package_chain) == 1
        and str(dict(package_chain[0]["binding"]).get("sha256") or "")
        != str(dict(package_binding).get("sha256") or "")
    ):
        package_chain = normalize_work_package_chain(None, fallback=package_binding)
    if package_binding and package_chain:
        if str(dict(package_binding).get("sha256") or "") != str(
            dict(package_chain[0]["binding"]).get("sha256") or ""
        ):
            errors.append("work_package_binding must remain the first Work Package chain binding")
    for chain_entry in package_chain:
        package = _verify_work_binding_at_available_root(
            dict(chain_entry["binding"]),
            state_root=state_root,
            work_id=work_id,
            repo_root=repo_root,
            read_only=read_only,
            allow_historical_package_binding=historical_package_binding,
        )
        errors.extend(package["errors"])
        warnings.extend(package["warnings"])
        if package.get("package") is not None:
            package_dependencies = normalize_string_list(dict(package["package"]).get("dependencies"))
            if sorted(package_dependencies) != sorted(dependencies):
                errors.append(
                    "work dependency_work_ids do not match the bound Work Package dependencies"
                )
    accepted_review = payload.get("accepted_review_session_id")
    if accepted_review and not str(accepted_review).startswith("SES-"):
        errors.append("accepted_review_session_id must start with SES-")
    elif accepted_review:
        try:
            if snapshot is not None:
                retained_review = snapshot.session_by_id.get(str(accepted_review))
                if retained_review is None:
                    raise ActorError(f"unknown session record: {accepted_review}")
                review_record = dict(retained_review)
            else:
                session_getter = (
                    session_store(state_root).get_read_only if read_only else session_store(state_root).get
                )
                review_record = session_getter(str(accepted_review))
            review_payload = dict(review_record.get("payload") or {})
            if review_payload.get("session_role") != "REVIEW_AUTHORITY":
                errors.append("accepted_review_session_id does not identify a Review Authority Session")
            if str(review_payload.get("work_id") or "") != work_id:
                errors.append("accepted_review_session_id belongs to another Work Item")
        except ActorError as exc:
            errors.append(f"accepted_review_session_id is not resolvable: {exc}")
    superseded_by = payload.get("superseded_by_work_id")
    if superseded_by:
        try:
            validate_prefixed_id(str(superseded_by), WORK_ID_PREFIX, label="superseding work item")
            if snapshot is not None:
                if str(superseded_by) not in snapshot.work_by_id:
                    raise ActorError(f"unknown work record: {superseded_by}")
            else:
                getter(str(superseded_by))
        except ActorError as exc:
            errors.append(f"superseded_by_work_id is not resolvable: {exc}")
    graph = dependency_graph(state_root, read_only=read_only, snapshot=snapshot)
    graph[work_id] = dependencies
    cycle = find_dependency_cycle(graph)
    if cycle:
        errors.append("dependency cycle detected: " + " -> ".join(cycle))
    lifecycle_status = str(status or "").upper()
    if lifecycle_status != "COMPLETED" and (
        completion_basis
        or payload.get("completion_reason")
        or payload.get("repository_integration_receipt") is not None
        or payload.get("terminal_decision_receipt") is not None
        or payload.get("acceptance_scope") is not None
        or bool(payload.get("accepted"))
    ):
        errors.append("completion evidence fields are valid only on COMPLETED Work")
    if (
        lifecycle_status not in WORK_TERMINAL_STATUSES
        and not payload.get("next_action")
        and not payload.get("next_command")
    ):
        warnings.append("no next action is recorded")
    if payload.get("preferred_codex_thread_url") and not payload.get("preferred_codex_conversation_id"):
        warnings.append("preferred Codex thread URL exists without an explicit conversation ID")
    return {"valid": not errors, "errors": errors, "warnings": warnings}


def _codex_identities(session: Mapping[str, Any]) -> set[str]:
    payload = dict(session.get("payload") or {})
    identities = {
        str(value).strip()
        for value in (payload.get("original_codex_id"), payload.get("current_codex_id"))
        if str(value or "").strip()
    }
    for participant in payload.get("codex_participants") or []:
        if isinstance(participant, Mapping) and str(participant.get("codex_id") or "").strip():
            identities.add(str(participant["codex_id"]).strip())
    return identities


def _package_chain_state(
    package_chain: Sequence[Mapping[str, Any]],
    *,
    delivery_sessions: Sequence[Mapping[str, Any]],
    reviews: Sequence[Mapping[str, Any]],
    accepted_reviews: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive the next package from immutable delivery/review lineage."""

    entries: list[dict[str, Any]] = []
    current_index: int | None = None
    accepted_by_target = {
        str(row.get("target_session_id") or "")
        for row in accepted_reviews
        if row.get("valid_acceptance")
    }
    for index, chain_entry in enumerate(package_chain):
        binding = dict(chain_entry["binding"])
        digest = str(binding.get("sha256") or "")
        deliveries = [
            dict(row) for row in delivery_sessions
            if str(dict(dict(row.get("payload") or {}).get("work_package_binding") or {}).get("sha256") or "") == digest
        ]
        delivery = deliveries[-1] if deliveries else None
        delivery_id = str(delivery.get("record_id") or "") if delivery else None
        requires_review = bool(chain_entry.get("requires_intermediate_review")) and index < len(package_chain) - 1
        review_for_delivery = [
            dict(row) for row in reviews
            if str(row.get("target_session_id") or "") == str(delivery_id or "")
        ]
        accepted = bool(delivery_id and delivery_id in accepted_by_target)
        entry = {
            "sequence": int(chain_entry.get("sequence") or index + 1),
            "binding": binding,
            "requires_intermediate_review": requires_review,
            "delivery_session_id": delivery_id,
            "implementation_complete": bool(delivery),
            "intermediate_review_sessions": review_for_delivery,
            "intermediate_review_accepted": accepted,
        }
        entries.append(entry)
        if current_index is None and (not delivery or (requires_review and not accepted)):
            current_index = index
    all_delivered = bool(entries) and all(bool(row["implementation_complete"]) for row in entries)
    intermediate_gate = next(
        (row for row in entries if row["implementation_complete"] and row["requires_intermediate_review"] and not row["intermediate_review_accepted"]),
        None,
    )
    if intermediate_gate:
        current_index = int(intermediate_gate["sequence"]) - 1
    next_entry = entries[current_index] if current_index is not None and current_index < len(entries) else None
    return {
        "entries": entries,
        "current_package_index": current_index,
        "current_package_binding": dict(next_entry["binding"]) if next_entry else None,
        "next_package_binding": dict(next_entry["binding"]) if next_entry and not next_entry["implementation_complete"] else None,
        "next_package_requires_intermediate_review": bool(next_entry and next_entry["requires_intermediate_review"]),
        "intermediate_review_required": bool(intermediate_gate),
        "all_delivered": all_delivered,
        "final_delivery_session_id": entries[-1].get("delivery_session_id") if entries else None,
        "final_review_sessions": [
            dict(row) for row in reviews
            if entries and str(row.get("target_session_id") or "") == str(entries[-1].get("delivery_session_id") or "")
        ],
        "accepted_intermediate_review_sessions": [
            dict(row) for row in accepted_reviews
            if entries and str(row.get("target_session_id") or "") != str(entries[-1].get("delivery_session_id") or "")
        ],
    }
def _terminal_decision_receipt_errors(
    work_record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
) -> list[str]:
    if str(work_record.get("status") or "") != "COMPLETED":
        return []
    payload = dict(work_record.get("payload") or {})
    reference = payload.get("terminal_decision_receipt")
    if not isinstance(reference, Mapping):
        return ["COMPLETED Work has no immutable terminal decision receipt"]
    try:
        receipt = load_work_terminal_decision_receipt(reference, state_root=state_root)
    except ActorError as exc:
        return [str(exc), *list(exc.problems or [])]
    errors: list[str] = []
    if str(receipt.get("work_id") or "") != str(work_record.get("record_id") or ""):
        errors.append("Work terminal decision receipt belongs to another Work Item")
    decision = dict(receipt.get("terminal_decision") or {})
    for field, expected in (
        ("final_status", "COMPLETED"),
        ("completion_policy", str(payload.get("completion_policy") or "")),
        ("completion_basis", str(payload.get("completion_basis") or "")),
        ("completion_reason", str(payload.get("completion_reason") or "")),
        ("accepted", bool(payload.get("accepted"))),
        ("acceptance_scope", str(payload.get("acceptance_scope") or "")),
        ("accepted_review_session_id", str(payload.get("accepted_review_session_id") or "") or None),
    ):
        if decision.get(field) != expected:
            errors.append(f"Work terminal decision receipt differs from terminal Work field: {field}")
    return errors


def verify_work_acceptance_evidence(
    work_record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
) -> list[str]:
    """Reconstruct and verify the exact historical acceptance decision graph."""

    payload = dict(work_record.get("payload") or {})
    completion_policy = str(payload.get("completion_policy") or "").upper()
    completion_basis = str(payload.get("completion_basis") or "").upper()
    receipt_errors = _terminal_decision_receipt_errors(work_record, state_root=state_root)
    if work_record.get("status") == "COMPLETED" and completion_basis == "RECORD_ONLY_COMPLETION":
        errors: list[str] = list(receipt_errors)
        if completion_policy != "RECORD_ONLY":
            errors.append("RECORD_ONLY completion basis requires RECORD_ONLY completion_policy")
        if payload.get("work_package_binding"):
            errors.append("package-bound Work cannot use RECORD_ONLY completion")
        if payload.get("accepted") is not False:
            errors.append("RECORD_ONLY completion must state accepted=false")
        if payload.get("acceptance_evidence") is not None:
            errors.append("RECORD_ONLY completion must not carry repository acceptance evidence")
        if payload.get("repository_integration_receipt") is not None:
            errors.append("RECORD_ONLY completion must not carry a repository integration receipt")
        return errors
    if work_record.get("status") == "COMPLETED" and completion_basis == "ADMITTED_TECHNICAL_QUALIFICATION":
        errors = list(receipt_errors)
        if completion_policy != "QUALIFICATION_DECISION":
            errors.append("technical qualification basis requires QUALIFICATION_DECISION completion_policy")
        raw = payload.get("acceptance_evidence")
        if not isinstance(raw, Mapping):
            errors.append("qualification Work has no retained qualification acceptance evidence")
            return errors
        evidence = dict(raw)
        if evidence.get("schema_version") != WORK_QUALIFICATION_EVIDENCE_SCHEMA_VERSION:
            errors.append("unsupported Work qualification acceptance evidence schema_version")
            return errors
        reference = evidence.get("qualification_record")
        if not isinstance(reference, Mapping):
            errors.append("Work qualification evidence has no exact decision record reference")
            return errors
        try:
            decision_record = qualification_store(state_root).get_revision_read_only(
                str(reference.get("record_id") or ""),
                int(reference.get("revision") or 0),
            )
            require_record_ref(reference, decision_record, label="technical qualification decision")
        except (ActorError, ValueError, TypeError) as exc:
            errors.append(f"technical qualification decision is not exactly resolvable: {exc}")
            return errors
        decision = dict(dict(decision_record.get("payload") or {}).get("decision") or {})
        if str(decision_record.get("status") or "") not in {"QUALIFIED", "QUALIFIED_WITH_LIMITS"}:
            errors.append("retained technical qualification decision is not positive")
        if not bool(decision.get("ok")) or not bool(decision.get("accepted_for_envelope_expansion")):
            errors.append("retained technical qualification decision does not admit the bounded envelope")
        if str(evidence.get("source_commit") or "") != str(decision.get("source_commit") or ""):
            errors.append("Work qualification source commit differs from retained decision")
        if list(evidence.get("limitations") or []) != list(decision.get("limitations") or []):
            errors.append("Work qualification limitations differ from retained decision")
        if not list(evidence.get("evidence_bindings") or []):
            errors.append("Work qualification acceptance has no content-addressed evidence bindings")
        return list(dict.fromkeys(errors))
    raw_evidence = payload.get("acceptance_evidence")
    if raw_evidence is None:
        return ["accepted repository delivery has no acceptance_evidence"] if work_record.get("status") == "COMPLETED" else []
    try:
        evidence = load_work_acceptance_evidence(raw_evidence)
    except ActorError as exc:
        return [str(exc), *list(exc.problems or [])]
    if str(evidence.get("schema_version") or "") == WORK_ACCEPTANCE_EVIDENCE_LEGACY_SCHEMA_VERSION:
        return [
            "legacy Work acceptance evidence is historical compatibility material only and cannot establish current accepted repository delivery"
        ]

    errors = [*receipt_errors, *verify_lifecycle_decision_snapshot(evidence.get("decision_snapshot"))]
    snapshot = dict(evidence.get("decision_snapshot") or {})
    review_ref = evidence.get("accepted_review")
    delivery_ref = evidence.get("accepted_delivery")
    if not isinstance(review_ref, Mapping):
        errors.append("Work acceptance evidence accepted_review reference is missing")
        review_ref = {}
    if not isinstance(delivery_ref, Mapping):
        errors.append("Work acceptance evidence accepted_delivery reference is missing")
        delivery_ref = {}

    accepted_review_id = str(payload.get("accepted_review_session_id") or "")
    if accepted_review_id != str(dict(review_ref).get("record_id") or ""):
        errors.append("Work accepted_review_session_id does not match acceptance evidence")

    def exact_record(store_obj: Any, reference: Mapping[str, Any], label: str) -> dict[str, Any] | None:
        record_id = str(reference.get("record_id") or "")
        revision = int(reference.get("revision") or 0)
        if not record_id or revision < 1:
            errors.append(f"{label} has invalid exact historical coordinates")
            return None
        try:
            record = store_obj.get_revision_read_only(record_id, revision)
            require_record_ref(reference, record, label=label)
            return record
        except ActorError as exc:
            errors.append(f"{label} is not exactly resolvable: {exc}")
            errors.extend(str(value) for value in (exc.problems or []))
            return None

    work_ref = snapshot.get("work") if isinstance(snapshot.get("work"), Mapping) else {}
    session_ref = snapshot.get("session") if isinstance(snapshot.get("session"), Mapping) else {}
    delivery_snapshot_ref = snapshot.get("delivery", snapshot.get("implementation"))
    implementation_ref = delivery_snapshot_ref if isinstance(delivery_snapshot_ref, Mapping) else {}
    snapshot_review_ref = snapshot.get("review") if isinstance(snapshot.get("review"), Mapping) else {}

    exact_work = exact_record(work_store(state_root), work_ref, "accepted Work decision input")
    exact_review = exact_record(session_store(state_root), dict(review_ref), "accepted Review Session")
    exact_delivery = exact_record(session_store(state_root), dict(delivery_ref), "accepted delivery Session")

    for label, observed, expected in (
        ("decision session", session_ref, dict(review_ref)),
        ("decision review", snapshot_review_ref, dict(review_ref)),
        ("decision implementation", implementation_ref, dict(delivery_ref)),
    ):
        if observed != expected:
            errors.append(f"Work acceptance evidence {label} reference differs from retained exact reference")

    if exact_work and str(exact_work.get("record_id") or "") != str(work_record.get("record_id") or ""):
        errors.append("Work acceptance snapshot belongs to another Work Item")
    if exact_work and int(exact_work.get("revision") or 0) >= int(work_record.get("revision") or 0):
        errors.append("Work acceptance snapshot does not precede the retained terminal Work revision")

    package_sha = str(dict(payload.get("work_package_binding") or {}).get("sha256") or "")
    if str(snapshot.get("package_sha256") or "") != package_sha:
        errors.append("Work acceptance snapshot package digest differs from retained Work binding")

    if exact_review:
        rp = dict(exact_review.get("payload") or {})
        review_evidence = dict(rp.get("review_evidence") or {})
        review_target = dict(rp.get("review_target") or {})
        if exact_review.get("status") != "COMPLETED":
            errors.append("accepted Review Session exact revision is not COMPLETED")
        if rp.get("session_role") != "REVIEW_AUTHORITY":
            errors.append("accepted Review Session exact revision is not Review Authority")
        if rp.get("session_outcome") != "REVIEW_COMPLETE":
            errors.append("accepted Review Session exact revision outcome is not REVIEW_COMPLETE")
        if rp.get("review_verdict") != "ACCEPT":
            errors.append("accepted Review Session exact revision verdict is not ACCEPT")
        if str(rp.get("work_id") or "") != str(work_record.get("record_id") or ""):
            errors.append("accepted Review Session belongs to another Work Item")
        delivery_id = str(dict(delivery_ref).get("record_id") or "")
        reviewed_commit = str(review_evidence.get("reviewed_commit") or "")
        if delivery_session_id(review_evidence) != delivery_id:
            errors.append("accepted Review evidence does not point to accepted delivery Session")
        if str(review_evidence.get("reviewed_package_sha256") or "") != package_sha:
            errors.append("accepted Review evidence package digest differs from Work binding")
        if delivery_session_id(review_target) != delivery_id:
            errors.append("accepted Review target does not point to accepted delivery Session")
        if str(review_target.get("reviewed_commit") or "") != reviewed_commit:
            errors.append("accepted Review target commit differs from Review evidence")
        if str(snapshot.get("delivery_commit") or "") != reviewed_commit:
            errors.append("Work acceptance snapshot delivery commit differs from Review evidence")
        proof_ref = review_evidence.get("obligation_proof")
        if not isinstance(proof_ref, Mapping):
            errors.append("accepted Review has no evidence-entailing obligation proof")
        else:
            try:
                proof = load_review_proof_artifact(proof_ref, state_root=state_root)
                if str(proof.get("work_id") or "") != str(work_record.get("record_id") or ""):
                    errors.append("accepted Review obligation proof belongs to another Work")
                if str(proof.get("review_session_id") or "") != str(exact_review.get("record_id") or ""):
                    errors.append("accepted Review obligation proof belongs to another Review Session")
                if str(proof.get("reviewed_commit") or "") != reviewed_commit:
                    errors.append("accepted Review obligation proof commit differs from Review evidence")
                if str(proof.get("reviewed_package_sha256") or "") != package_sha:
                    errors.append("accepted Review obligation proof package differs from Work binding")
                if str(proof.get("implementation_sufficiency") or "") != "SUFFICIENT":
                    errors.append("accepted Review did not establish implementation sufficiency")
            except ActorError as exc:
                errors.append(f"accepted Review obligation proof is invalid: {exc}")
                errors.extend(str(value) for value in (exc.problems or []))

    if exact_delivery:
        dp = dict(exact_delivery.get("payload") or {})
        delivery_evidence = dict(dp.get("delivery_evidence") or {})
        if exact_delivery.get("status") != "COMPLETED":
            errors.append("accepted delivery Session exact revision is not COMPLETED")
        if dp.get("session_role") == "REVIEW_AUTHORITY":
            errors.append("accepted delivery Session is a Review Authority Session")
        if dp.get("session_outcome") not in {
            "ARCHITECTURE_COMPLETE", "ENGINEERING_COMPLETE", "IMPLEMENTATION_COMPLETE",
            "DELIVERY_RECORDED", "EXTERNAL_DELIVERY_ADOPTED",
        }:
            errors.append("accepted delivery Session has no completed delivery outcome")
        if str(dp.get("work_id") or "") != str(work_record.get("record_id") or ""):
            errors.append("accepted delivery Session belongs to another Work Item")
        implementation_commit = delivery_commit_value(delivery_evidence)
        if implementation_commit != str(snapshot.get("delivery_commit") or ""):
            errors.append("accepted delivery Session commit differs from Work acceptance snapshot")
        if str(delivery_evidence.get("work_package_sha256") or "") != package_sha:
            errors.append("accepted delivery Session package digest differs from Work binding")
        proof_ref = delivery_evidence.get("delivery_proof")
        if not isinstance(proof_ref, Mapping):
            errors.append("accepted delivery Session has no durable repository delivery proof")
        else:
            try:
                proof = load_repository_truth_artifact(
                    proof_ref,
                    state_root=state_root,
                    expected_schema_version=DELIVERY_PROOF_SCHEMA_VERSION,
                )
                if str(proof.get("work_id") or "") != str(work_record.get("record_id") or ""):
                    errors.append("delivery proof belongs to another Work")
                if str(proof.get("session_id") or "") != str(exact_delivery.get("record_id") or ""):
                    errors.append("delivery proof belongs to another Session")
                if str(proof.get("delivery_commit") or "") != implementation_commit:
                    errors.append("delivery proof commit differs from delivery evidence")
            except ActorError as exc:
                errors.append(f"accepted delivery proof is invalid: {exc}")
                errors.extend(str(value) for value in (exc.problems or []))

    integration_ref = evidence.get("repository_integration_receipt")
    if not isinstance(integration_ref, Mapping):
        errors.append("Work acceptance evidence integration receipt is missing")
    else:
        try:
            integration = load_repository_truth_artifact(
                integration_ref,
                state_root=state_root,
                expected_schema_version=INTEGRATION_RECEIPT_SCHEMA_VERSION,
            )
            if str(integration.get("work_id") or "") != str(work_record.get("record_id") or ""):
                errors.append("integration receipt belongs to another Work")
            if str(integration.get("review_session_id") or "") != accepted_review_id:
                errors.append("integration receipt belongs to another Review Session")
            if str(integration.get("reviewed_commit") or "") != str(snapshot.get("delivery_commit") or ""):
                errors.append("integration receipt reviewed commit differs from acceptance snapshot")
            payload_integration_ref = payload.get("repository_integration_receipt")
            if not isinstance(payload_integration_ref, Mapping) or dict(payload_integration_ref) != dict(integration_ref):
                errors.append("Work repository integration receipt differs from acceptance evidence")
        except ActorError as exc:
            errors.append(f"integration receipt is invalid: {exc}")
            errors.extend(str(value) for value in (exc.problems or []))

    if exact_review and exact_delivery:
        overlap = sorted(_codex_identities(exact_review).intersection(_codex_identities(exact_delivery)))
        review_assurance = str(dict(dict(exact_review.get("payload") or {}).get("review_evidence") or {}).get("identity_separation_assurance") or "")
        if overlap and review_assurance != "REVIEWER_INDEPENDENCE_UNAVAILABLE":
            errors.append(
                "accepted review is not independent from delivery Codex identity: " + ", ".join(overlap)
            )

    return list(dict.fromkeys(errors))


def review_state_for_work(
    record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    read_only: bool = False,
    snapshot: PortfolioSnapshot | None = None,
) -> dict[str, Any]:
    """Return immutable delivery lineage and one authoritative review verdict."""

    payload = dict(record.get("payload") or {})
    package_binding = payload.get("work_package_binding")
    package_chain = normalize_work_package_chain(
        payload.get("work_package_chain"), fallback=package_binding
    )
    if (
        len(package_chain) == 1
        and package_binding
        and str(dict(package_chain[0]["binding"]).get("sha256") or "")
        != str(dict(package_binding).get("sha256") or "")
    ):
        package_chain = normalize_work_package_chain(None, fallback=package_binding)
    chain_shas = {
        str(dict(entry["binding"]).get("sha256") or "") for entry in package_chain
    }
    package_sha = str(dict(package_binding or {}).get("sha256") or "")
    package_hash_kind = str(dict(package_binding or {}).get("sha256_kind") or LEGACY_HASH_KIND)
    legacy_v17_package = bool(package_binding) and package_hash_kind == LEGACY_HASH_KIND
    sessions = sessions_for_work(state_root, str(record.get("record_id")), read_only=read_only, snapshot=snapshot)
    delivery_sessions: list[dict[str, Any]] = []
    for row in sessions:
        session_payload = dict(row.get("payload") or {})
        session_sha = str(dict(session_payload.get("work_package_binding") or {}).get("sha256") or "")
        if (
            row.get("status") == "COMPLETED"
            and session_payload.get("session_outcome")
            in {
                "ARCHITECTURE_COMPLETE",
                "ENGINEERING_COMPLETE",
                "IMPLEMENTATION_COMPLETE",
                "DELIVERY_RECORDED",
                "EXTERNAL_DELIVERY_ADOPTED",
            }
            and session_payload.get("session_role") != "REVIEW_AUTHORITY"
            and (not chain_shas or session_sha in chain_shas)
            and (isinstance(session_payload.get("delivery_evidence"), Mapping) or legacy_v17_package)
        ):
            delivery_sessions.append(dict(row))

    # A sequential chain intentionally contains ancestor commits. Resolve
    # lineage independently for each package so WP1 remains part of the final
    # integrated delivery rather than being treated as a superseded repair.
    chain_lineages: list[dict[str, Any]] = []
    authoritative_deliveries: list[dict[str, Any]] = []
    for chain_entry in package_chain:
        digest = str(dict(chain_entry["binding"]).get("sha256") or "")
        rows = [
            row for row in delivery_sessions
            if str(dict(dict(row.get("payload") or {}).get("work_package_binding") or {}).get("sha256") or "") == digest
        ]
        if legacy_v17_package:
            rows.sort(key=lambda row: (delivery_recorded_at(row), str(row.get("record_id") or "")), reverse=True)
            lineage = {"ordered": rows, "authoritative": rows[0] if rows else None, "conflict": False, "conflicting_session_ids": [], "superseded_session_ids": [], "problems": []}
        else:
            lineage = resolve_delivery_lineage(rows)
        chain_lineages.append(lineage)
        if lineage.get("authoritative"):
            authoritative_deliveries.append(dict(lineage["authoritative"]))
    delivery_sessions = authoritative_deliveries
    latest_delivery = authoritative_deliveries[-1] if authoritative_deliveries else None
    latest_delivery_payload = dict(latest_delivery.get("payload") or {}) if latest_delivery else {}
    latest_delivery_evidence = dict(latest_delivery_payload.get("delivery_evidence") or {})
    latest_delivery_id = str(latest_delivery.get("record_id") or "") if latest_delivery else ""
    latest_commit = delivery_commit_value(latest_delivery_evidence)
    lineage = {
        "conflict": any(bool(row.get("conflict")) for row in chain_lineages),
        "conflicting_session_ids": [
            session_id for row in chain_lineages for session_id in list(row.get("conflicting_session_ids") or [])
        ],
        "superseded_session_ids": [
            session_id for row in chain_lineages for session_id in list(row.get("superseded_session_ids") or [])
        ],
        "problems": [problem for row in chain_lineages for problem in list(row.get("problems") or [])],
    }

    reviews: list[dict[str, Any]] = []
    current_reviews: list[dict[str, Any]] = []
    review_rows = [row for row in sessions if dict(row.get("payload") or {}).get("session_role") == "REVIEW_AUTHORITY"]
    review_rows.sort(
        key=lambda row: (review_recorded_at(row), str(row.get("record_id") or "")),
        reverse=True,
    )
    deliveries_by_id = {
        str(row.get("record_id") or ""): row for row in delivery_sessions
    }
    final_package_sha = str(
        dict(package_chain[-1]["binding"]).get("sha256") or ""
    ) if package_chain else package_sha
    for row in review_rows:
        session_payload = dict(row.get("payload") or {})
        verdict = session_payload.get("review_verdict")
        review_sha = str(dict(session_payload.get("work_package_binding") or {}).get("sha256") or "")
        target = dict(session_payload.get("review_target") or {})
        evidence = dict(session_payload.get("review_evidence") or {})
        target_id = delivery_session_id(target)
        if legacy_v17_package and not target_id:
            target_id = latest_delivery_id
        target_delivery = deliveries_by_id.get(target_id)
        target_evidence = dict(dict(target_delivery or {}).get("payload") or {}).get("delivery_evidence") or {}
        package_match = not chain_shas or review_sha in chain_shas
        if legacy_v17_package:
            target_is_latest = bool(latest_delivery_id)
            commit_match = True
            evidence_match = True
        else:
            target_is_latest = bool(target_delivery)
            expected_target_commit = delivery_commit_value(target_evidence)
            commit_match = bool(expected_target_commit) and str(target.get("reviewed_commit") or "") == expected_target_commit
            evidence_match = (
                target_id in deliveries_by_id
                and delivery_session_id(evidence) == target_id
                and str(evidence.get("reviewed_commit") or "") == expected_target_commit
                and str(evidence.get("reviewed_package_sha256") or "") == review_sha
            )
        reasons: list[str] = []
        review_warnings: list[str] = []
        if legacy_v17_package:
            review_warnings.append(
                "legacy V1.7 review lacks exact delivery-session and commit binding; preserved for backward compatibility"
            )
        if lineage.get("conflict"):
            reasons.append("completed deliveries have unresolved divergent or unverifiable commit lineage")
        if row.get("status") != "COMPLETED":
            reasons.append("review session is not completed")
        if session_payload.get("session_outcome") != "REVIEW_COMPLETE":
            reasons.append("review session outcome is not REVIEW_COMPLETE")
        if not package_match:
            reasons.append("review session is not bound to the current Work Package digest")
        if not target_is_latest:
            reasons.append("review does not target an authoritative package delivery session")
        if not commit_match:
            reasons.append("review does not target the authoritative delivery commit")
        if row.get("status") == "COMPLETED" and not evidence_match:
            reasons.append("review evidence does not bind the authoritative session, commit, and package digest")
        if target_delivery:
            overlap = sorted(_codex_identities(target_delivery).intersection(_codex_identities(row)))
            if overlap:
                review_warnings.append(
                    "reviewer independence unavailable; same Codex identity retained as reduced assurance: "
                    + ", ".join(overlap)
                )
        elif verdict == "ACCEPT" and package_match and not lineage.get("conflict"):
            reasons.append("no completed delivery session is available for review")
        structurally_current = not reasons
        review = {
            "session_id": row.get("record_id"),
            "status": row.get("status"),
            "verdict": verdict,
            "package_match": package_match,
            "target_session_id": target_id or None,
            "reviewed_commit": target.get("reviewed_commit"),
            "targets_latest_delivery": target_is_latest and commit_match,
            "is_final_review": bool(target_id == latest_delivery_id and review_sha == final_package_sha),
            "valid_acceptance": False,
            "authoritative": False,
            "superseded": False,
            "legacy_acceptance": legacy_v17_package,
            "problems": reasons,
            "warnings": review_warnings,
            "reviewed_at_utc": review_recorded_at(row),
            "updated_at_utc": row.get("updated_at_utc"),
        }
        reviews.append(review)
        if structurally_current:
            current_reviews.append(review)

    accepted: list[dict[str, Any]] = []
    accepted_intermediate: list[dict[str, Any]] = []
    authoritative_review = None
    for review in current_reviews:
        if review.get("is_final_review"):
            if authoritative_review is None:
                authoritative_review = review
            else:
                review["superseded"] = True
                review["problems"].append(
                    f"review verdict is superseded by authoritative review {authoritative_review['session_id']}"
                )
    for review in current_reviews:
        if review.get("superseded"):
            continue
        review["authoritative"] = True
        review["valid_acceptance"] = review.get("verdict") == "ACCEPT"
        if review["valid_acceptance"]:
            if review.get("is_final_review"):
                accepted.append(review)
            else:
                accepted_intermediate.append(review)

    promotion_sessions: list[dict[str, Any]] = []
    for row in sessions:
        session_payload = dict(row.get("payload") or {})
        session_sha = str(dict(session_payload.get("work_package_binding") or {}).get("sha256") or "")
        package_match = not chain_shas or session_sha in chain_shas
        explicit_request = package_match and session_payload.get("session_outcome") == "PROMOTION_REQUESTED"
        review_promotion = bool(
            authoritative_review
            and str(authoritative_review.get("session_id") or "") == str(row.get("record_id") or "")
            and authoritative_review.get("verdict") == "PROMOTE"
        )
        if explicit_request or review_promotion:
            promotion_sessions.append(
                {
                    "session_id": row.get("record_id"),
                    "status": row.get("status"),
                    "package_sha256": session_sha or None,
                    "promotion_request": dict(session_payload.get("promotion_request") or {}),
                    "source": "review_verdict" if review_promotion else "session_request",
                }
            )
    latest_verdict = authoritative_review["verdict"] if authoritative_review else None
    chain_state = _package_chain_state(
        package_chain,
        delivery_sessions=delivery_sessions,
        reviews=reviews,
        accepted_reviews=[*accepted, *accepted_intermediate],
    )
    return {
        "required": bool(package_binding),
        "implementation_complete": bool(delivery_sessions),
        "delivery_complete": bool(delivery_sessions),
        "implementation_session_ids": [str(row.get("record_id")) for row in delivery_sessions],
        "latest_delivery_session_id": latest_delivery_id or None,
        "latest_delivery_commit": latest_commit or None,
        "delivery_conflict": bool(lineage.get("conflict")),
        "conflicting_delivery_session_ids": list(lineage.get("conflicting_session_ids") or []),
        "superseded_delivery_session_ids": list(lineage.get("superseded_session_ids") or []),
        "delivery_lineage_problems": list(lineage.get("problems") or []),
        "review_sessions": reviews,
        "accepted_review_sessions": accepted,
        "authoritative_review_session_id": authoritative_review.get("session_id") if authoritative_review else None,
        "review_conflict": False,
        "conflicting_review_session_ids": [],
        "review_conflict_problems": [],
        "latest_verdict": latest_verdict,
        "promotion_sessions": promotion_sessions,
        "package_chain": chain_state["entries"],
        "current_package_index": chain_state["current_package_index"],
        "current_package_binding": chain_state.get("current_package_binding"),
        "next_package_binding": chain_state.get("next_package_binding"),
        "next_package_requires_intermediate_review": chain_state.get("next_package_requires_intermediate_review", False),
        "intermediate_review_required": chain_state.get("intermediate_review_required", False),
        "all_packages_delivered": chain_state["all_delivered"],
        "final_delivery_session_id": chain_state.get("final_delivery_session_id"),
        "final_review_sessions": chain_state.get("final_review_sessions", []),
        "accepted_intermediate_review_sessions": chain_state.get("accepted_intermediate_review_sessions", []),
        "multi_package": len(package_chain) > 1,
    }


def authoritative_review_binding_for_work(
    record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    read_only: bool = False,
    snapshot: PortfolioSnapshot | None = None,
) -> dict[str, Any] | None:
    """Return the exact current review decision coordinates for one Work."""

    review_state = review_state_for_work(
        record, state_root=state_root, read_only=read_only, snapshot=snapshot
    )
    review_id = str(review_state.get("authoritative_review_session_id") or "")
    if not review_id:
        return None
    review = next(
        (
            dict(row)
            for row in review_state.get("review_sessions") or []
            if str(dict(row).get("session_id") or "") == review_id
        ),
        None,
    )
    if review is None:
        return None
    binding = {
        "schema_version": AUTHORITATIVE_REVIEW_BINDING_SCHEMA_VERSION,
        "review_session_id": review_id,
        "review_revision": int(review.get("revision") or 0),
        "review_record_sha256": str(review.get("record_sha256") or ""),
        "verdict": str(review.get("verdict") or "") or None,
        "reviewed_delivery_session_id": str(review.get("target_session_id") or "") or None,
        "reviewed_commit": str(review.get("reviewed_commit") or "") or None,
        "reviewed_package_sha256": str(review.get("reviewed_package_sha256") or "") or None,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    return binding


def _derived_state(
    record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    read_only: bool = False,
    snapshot: PortfolioSnapshot | None = None,
) -> dict[str, Any]:
    payload = dict(record.get("payload") or {})
    work_id = str(record.get("record_id"))
    dependencies = dependency_state(
        state_root,
        payload.get("dependency_work_ids") or [],
        read_only=read_only,
        snapshot=snapshot,
    )
    session_records = sessions_for_work(
        state_root,
        work_id,
        read_only=read_only,
        snapshot=snapshot,
    )
    session_rows = [
        {
            "session_id": row.get("record_id"),
            "status": row.get("status"),
            "updated_at_utc": row.get("updated_at_utc"),
            "current_step": dict(row.get("payload") or {}).get("current_step"),
            "next_action": dict(row.get("payload") or {}).get("next_action"),
            "current_codex_id": dict(row.get("payload") or {}).get("current_codex_id"),
            "session_role": dict(row.get("payload") or {}).get("session_role"),
            "session_outcome": dict(row.get("payload") or {}).get("session_outcome"),
            "review_verdict": dict(row.get("payload") or {}).get("review_verdict"),
            "promotion_request": dict(row.get("payload") or {}).get("promotion_request"),
            "authoritative_review_binding": dict(row.get("payload") or {}).get("authoritative_review_binding"),
            "work_package_binding": dict(row.get("payload") or {}).get("work_package_binding"),
            "worker_identity_state": dict(row.get("payload") or {}).get("worker_identity_state"),
            "prepared_assignment_id": dict(row.get("payload") or {}).get("prepared_assignment_id"),
            "scheduler_assignment_id": dict(row.get("payload") or {}).get("scheduler_assignment_id"),
        }
        for row in session_records
    ]
    active_sessions = [row for row in session_rows if row["status"] in SESSION_ACTIVE_STATUSES]
    review = review_state_for_work(
        record,
        state_root=state_root,
        read_only=read_only,
        snapshot=snapshot,
    )
    status = str(record.get("status") or "BACKLOG")
    if status in WORK_OPEN_STATUSES and review["promotion_sessions"]:
        derived = "PROMOTION_REQUIRED"
    elif status in WORK_OPEN_STATUSES and dependencies["terminal_unsatisfied"]:
        derived = "BLOCKED_BY_TERMINAL_DEPENDENCY"
    elif status in WORK_OPEN_STATUSES and (dependencies["pending"] or dependencies["missing"]):
        derived = "BLOCKED_BY_DEPENDENCY"
    elif status in WORK_OPEN_STATUSES and active_sessions:
        derived = status
    elif status in WORK_OPEN_STATUSES and review.get("intermediate_review_required"):
        derived = "CHANGES_REQUIRED" if review["latest_verdict"] in {"CHANGES_REQUIRED", "REJECT", "BLOCKED"} else "REVIEW_REQUIRED"
    elif status in WORK_OPEN_STATUSES and review.get("all_packages_delivered") and not review["accepted_review_sessions"]:
        derived = "CHANGES_REQUIRED" if review["latest_verdict"] in {"CHANGES_REQUIRED", "REJECT", "BLOCKED"} else "REVIEW_REQUIRED"
    elif status in WORK_OPEN_STATUSES and not review.get("multi_package") and review["required"] and review["implementation_complete"] and not review["accepted_review_sessions"]:
        derived = "CHANGES_REQUIRED" if review["latest_verdict"] in {"CHANGES_REQUIRED", "REJECT", "BLOCKED"} else "REVIEW_REQUIRED"
    elif status in WORK_OPEN_STATUSES and review["accepted_review_sessions"]:
        derived = "READY_TO_ACCEPT"
    elif status == "BLOCKED" and dependencies["all_satisfied"]:
        derived = "READY_TO_RESUME"
    else:
        derived = status
    next_actions: list[dict[str, Any]] = []
    if status in WORK_TERMINAL_STATUSES:
        next_actions.append({"action": "none", "description": f"Work is {status.lower()} and is not resumable."})
    elif review["promotion_sessions"]:
        promotion = review["promotion_sessions"][0]
        next_actions.append({
            "action": "return_to_engineering_authority",
            "description": "Implementation discovered more uncertainty than the approved Work Package permits. Do not redesign inside this session.",
            "command": f"python -m tools.eiel sessionManager inspect {promotion['session_id']}",
            "promotion_request_ref": {
                "session_id": promotion.get("session_id"),
                "revision": promotion.get("revision"),
                "record_sha256": promotion.get("record_sha256"),
                "package_sha256": promotion.get("package_sha256"),
                "requested_uncertainty_class": promotion.get("requested_uncertainty_class"),
                "recommended_authority": promotion.get("recommended_authority"),
                "promotion_request_sha256": promotion.get("promotion_request_sha256"),
            },
        })
    elif dependencies["terminal_unsatisfied"]:
        next_actions.append({
            "action": "resolve_terminal_dependency",
            "description": "A dependency ended without completion. Replace, supersede, or explicitly redesign the blocked work.",
        })
    elif dependencies["pending"] or dependencies["missing"]:
        next_actions.append({
            "action": "complete_dependencies",
            "description": "Complete or repair the listed dependency work before resuming this item.",
            "command": "python -m tools.eiel workManager graph " + work_id,
        })
    elif status == "BLOCKED":
        next_actions.append({"action": "resume_work", "description": "All dependencies are complete; resume the work explicitly.", "command": f"python -m tools.eiel workManager resume {work_id}"})
    elif active_sessions:
        next_actions.append({"action": "resume_session", "description": "Continue the active session.", "command": f"python -m tools.eiel sessionManager inspect {active_sessions[0]['session_id']}"})
    elif review.get("intermediate_review_required"):
        intermediate = next(
            (row for row in review.get("package_chain") or [] if row.get("requires_intermediate_review") and row.get("implementation_complete") and not row.get("intermediate_review_accepted")),
            None,
        )
        next_actions.append({
            "action": "start_review",
            "description": "This Work Package requires substantive intermediate review before the next package can start; use reduced assurance if no independent reviewer is immediately available.",
            "command": f"python -m tools.eiel sessionManager start {work_id} --bootstrap-record '<BST-ID>' --codex-id '<codex-id>' --role REVIEW_AUTHORITY --review-target-session {dict(intermediate or {}).get('delivery_session_id')}",
        })
    elif review.get("all_packages_delivered") and not review["accepted_review_sessions"]:
        if review["latest_verdict"] in {"CHANGES_REQUIRED", "REJECT", "BLOCKED"}:
            next_actions.append({
                "action": "repair_implementation",
                "description": "Final substantive review did not accept the integrated delivery. Start a bounded repair session for the affected package.",
                "command": f"python -m tools.eiel sessionManager start {work_id} --bootstrap-record '<BST-ID>' --codex-id '<codex-id>' --role IMPLEMENTATION_AUTHORITY",
            })
        else:
            next_actions.append({
                "action": "start_review",
                "description": "All Work Packages are delivered. Start the final Review Authority session; use an independent identity when immediately available, otherwise record reduced assurance.",
                "command": f"python -m tools.eiel sessionManager start {work_id} --bootstrap-record '<BST-ID>' --codex-id '<codex-id>' --role REVIEW_AUTHORITY --review-target-session {review.get('final_delivery_session_id') or review.get('latest_delivery_session_id')}",
            })
    elif not review.get("multi_package") and review["required"] and review["implementation_complete"] and not review["accepted_review_sessions"]:
        if review["latest_verdict"] in {"CHANGES_REQUIRED", "REJECT", "BLOCKED"}:
            next_actions.append({
                "action": "repair_implementation",
                "description": "Substantive review did not accept the implementation. Start a bounded repair session from the same Work Package or promote the work if its contract is insufficient.",
                "command": f"python -m tools.eiel sessionManager start {work_id} --bootstrap-record '<BST-ID>' --codex-id '<codex-id>' --role IMPLEMENTATION_AUTHORITY",
            })
        else:
            next_actions.append({
                "action": "start_review",
                "description": "Implementation is complete but Work is not accepted. Start a Review Authority session; use reduced assurance if no independent reviewer is immediately available.",
                "command": f"python -m tools.eiel sessionManager start {work_id} --bootstrap-record '<BST-ID>' --codex-id '<codex-id>' --role REVIEW_AUTHORITY --review-target-session {review.get('latest_delivery_session_id')}",
            })
    elif review["accepted_review_sessions"]:
        accepted = review["accepted_review_sessions"][0]
        next_actions.append({
            "action": "accept_work",
            "description": "An evidence-entailing Review Authority session accepted the bound implementation. Complete the Work explicitly.",
            "command": f"python -m tools.eiel workManager complete {work_id} --review-session {accepted['session_id']}",
        })
    elif review.get("next_package_binding"):
        binding = dict(review["next_package_binding"])
        next_actions.append({
            "action": "start_next_package",
            "description": f"Start implementation Session for Work Package {binding.get('path')} (package {review.get('current_package_index', 0) + 1}).",
            "command": f"python -m tools.eiel sessionManager start {work_id} --bootstrap-record '<BST-ID>' --codex-id '<codex-id>' --role IMPLEMENTATION_AUTHORITY",
        })
    elif session_rows:
        prepared = [
            row
            for row in session_rows
            if row.get("worker_identity_state") == "PENDING"
            and row.get("prepared_assignment_id")
        ]
        if prepared:
            prepared.sort(
                key=lambda row: (
                    str(row.get("updated_at_utc") or ""),
                    str(row.get("session_id") or ""),
                ),
                reverse=True,
            )
            selected = prepared[0]
            next_actions.append(
                {
                    "action": "scheduler_assignment_owns_session",
                    "description": (
                        "A READY Scheduler Assignment owns this prepared Session. "
                        "Let the Scheduler claim it or perform an explicit manual takeover."
                    ),
                    "command": "python -m tools.eiel assignment status",
                }
            )
            next_actions.append(
                {
                    "action": "manual_takeover",
                    "description": "Explicitly supersede the unstarted Scheduler attempt and resume manually.",
                    "command": (
                        f"python -m tools.eiel sessionManager resume {selected['session_id']} "
                        "--take-over-prepared --takeover-reason '<reason>' --codex-id '<current-codex-id>'"
                    ),
                }
            )
        else:
            resumable = [
                row
                for row in session_rows
                if row["status"] in {"CREATED", "PAUSED", "INTERRUPTED", "BLOCKED"}
                and row.get("worker_identity_state") != "PENDING"
                and not row.get("prepared_assignment_id")
            ]
            if resumable:
                next_actions.append({"action": "resume_session", "description": "Resume the most recent interrupted or paused session.", "command": f"python -m tools.eiel sessionManager resume {resumable[0]['session_id']} --codex-id '<current-codex-id>'"})
            else:
                next_actions.append({"action": "start_session", "description": "Start a new Codex session for this work item.", "command": f"python -m tools.eiel sessionManager start {work_id} --bootstrap-record '<BST-ID>' --codex-id '<codex-id>'"})
    else:
        next_actions.append({"action": "start_session", "description": "Start the first Codex session for this work item.", "command": f"python -m tools.eiel sessionManager start {work_id} --bootstrap-record '<BST-ID>' --codex-id '<codex-id>'"})
    if payload.get("next_action") and not (
        review.get("multi_package") and review.get("next_package_binding")
    ):
        next_actions.insert(0, {"action": "recorded_next_action", "description": str(payload["next_action"]), **({"command": payload["next_command"]} if payload.get("next_command") else {})})
    return {
        "work_id": work_id,
        "status": status,
        "derived_status": derived,
        "dependencies": dependencies,
        "sessions": session_rows,
        "active_session_count": len(active_sessions),
        "review": review,
        "next_actions": next_actions,
    }


def _repository_truth_projection(
    record: Mapping[str, Any],
    *,
    review: Mapping[str, Any],
    repo_root: str | Path | None,
) -> dict[str, Any]:
    """Compare open Work truth with an explicitly selected repository checkout.

    A delivery branch ending at the reviewed commit is not an integration.  A
    later checkout that contains that delivery while Work is still open is a
    repository/lifecycle contradiction that must be reconciled explicitly.
    """

    if not repo_root:
        return {"status": "NOT_CHECKED", "violation": False, "current_head": None, "delivery_commit": None}
    delivery_commit = str(review.get("latest_delivery_commit") or "").strip().lower()
    if not delivery_commit:
        return {"status": "NO_DELIVERY", "violation": False, "current_head": None, "delivery_commit": None}
    try:
        current = git_repository_identity(repo_root)
    except ActorError as exc:
        return {"status": "REPOSITORY_UNAVAILABLE", "violation": False, "current_head": None, "delivery_commit": delivery_commit, "problems": [str(exc)]}
    current_head = str(current.get("head") or "").lower()
    if current_head == delivery_commit:
        relation = "DELIVERY_HEAD"
    else:
        ancestor = subprocess.run(
            ["git", "-C", str(Path(repo_root).expanduser().resolve()), "merge-base", "--is-ancestor", delivery_commit, current_head],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        )
        relation = "DESCENDANT_INTEGRATION" if ancestor.returncode == 0 else "UNRELATED_OR_REWRITTEN"
    violation = str(record.get("status") or "") != "COMPLETED" and relation == "DESCENDANT_INTEGRATION"
    return {
        "status": relation,
        "violation": violation,
        "current_head": current_head,
        "delivery_commit": delivery_commit,
        "problem": (
            "repository contains the managed delivery in a later commit while Work remains unresolved"
            if violation else None
        ),
    }


def _summary(
    record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    read_only: bool = False,
    repo_root: str | Path | None = None,
    snapshot: PortfolioSnapshot | None = None,
) -> dict[str, Any]:
    payload = dict(record.get("payload") or {})
    projected_estimate = normalize_resource_estimate(
        payload.get("resource_estimate") if isinstance(payload.get("resource_estimate"), Mapping) else None,
        generated_at=str(record.get("created_at_utc") or utc_now()),
    )
    projected_accounting = normalize_work_resource_accounting(
        payload.get("resource_accounting") if isinstance(payload.get("resource_accounting"), Mapping) else None,
        estimate=projected_estimate,
    )
    derived = _derived_state(
        record,
        state_root=state_root,
        read_only=read_only,
        snapshot=snapshot,
    )
    repository_truth = _repository_truth_projection(
        record, review=derived["review"], repo_root=repo_root
    )
    validation = _validate_payload(
        payload,
        state_root=state_root,
        existing_id=str(record.get("record_id")),
        status=str(record.get("status") or ""),
        read_only=read_only,
        repo_root=repo_root,
        snapshot=snapshot,
    )
    acceptance_errors = verify_work_acceptance_evidence(record, state_root=state_root)
    repository_errors = (
        [str(repository_truth.get("problem"))]
        if repository_truth.get("violation") and repository_truth.get("problem")
        else []
    )
    if acceptance_errors or repository_errors:
        validation = {
            **validation,
            "valid": False,
            "errors": [
                *list(validation.get("errors") or []),
                *acceptance_errors,
                *repository_errors,
            ],
        }
    if str(record.get("status") or "") != "COMPLETED":
        acceptance_status = "NOT_DECIDED"
    elif str(payload.get("completion_basis") or "") == "RECORD_ONLY_COMPLETION":
        acceptance_status = "NOT_APPLICABLE"
    elif not bool(payload.get("accepted")):
        acceptance_status = "NOT_ACCEPTED"
    elif str(payload.get("completion_basis") or "") == "ADMITTED_TECHNICAL_QUALIFICATION":
        acceptance_status = (
            "TECHNICALLY_QUALIFIED_WITH_LIMITS"
            if str(payload.get("acceptance_scope") or "") == "WITH_LIMITS"
            else "TECHNICALLY_QUALIFIED"
        )
    else:
        acceptance_status = (
            "ACCEPTED_WITH_LIMITS"
            if str(payload.get("acceptance_scope") or "") == "WITH_LIMITS"
            else "ACCEPTED"
        )
    canonical_lifecycle = canonical_work_lifecycle(
        record,
        sessions=derived["sessions"],
        derived_status=derived["derived_status"],
    )
    return {
        "work_id": record.get("record_id"),
        "title": payload.get("title"),
        "summary": payload.get("summary"),
        "work_type": payload.get("work_type"),
        "subsystem": payload.get("subsystem"),
        "priority": payload.get("priority"),
        "status": record.get("status"),
        "status_scope": "WORK_LIFECYCLE",
        "status_label": f"Work {record.get('status')}",
        "work_lifecycle_status": record.get("status"),
        "canonical_work_lifecycle": canonical_lifecycle,
        "canonical_work_state": canonical_lifecycle["state"],
        "work_acceptance_status": acceptance_status,
        "derived_status": derived["derived_status"],
        "mission_id": payload.get("mission_id"),
        "parent_work_id": payload.get("parent_work_id"),
        "dependency_work_ids": payload.get("dependency_work_ids"),
        "current_stage": payload.get("current_stage"),
        "current_step": payload.get("current_step"),
        "last_completed_step": payload.get("last_completed_step"),
        "next_action": payload.get("next_action"),
        "next_command": payload.get("next_command"),
        "work_package_binding": payload.get("work_package_binding"),
        "work_package_chain": payload.get("work_package_chain"),
        "package_binding_history": payload.get("package_binding_history") or [],
        "promotion_resolution": payload.get("promotion_resolution"),
        "terminal_transition": payload.get("terminal_transition"),
        "authority_binding": {
            "schema_version": "eiel_authority_binding_projection_v1",
            "record_owner": "WORK_MANAGER",
            "record_authority": "WORK_MANAGER",
            "lifecycle_owner": "WORK_MANAGER",
            "acceptance_owner": "WORK_MANAGER",
            "authority_role": dict(payload.get("work_package_binding") or {}).get("recommended_authority"),
            "authority_domain": payload.get("subsystem") or payload.get("work_type"),
            "authorization_basis": (
                "BOUND_WORK_PACKAGE" if payload.get("work_package_binding") else "HUMAN_OPERATOR_WORK_RECORD"
            ),
            "current_technical_authority": None,
            "required_delivery_authority": dict(payload.get("work_package_binding") or {}).get("recommended_authority"),
            "identity_assurance": "EXACT_RECORD_BOUND",
        },
        "projection_freshness": {
            "schema_version": "eiel_projection_freshness_v1",
            "authoritative_record": actor_record_ref(record),
            "material_execution_ref": work_execution_ref(record),
            "status": "EXACT_RECORD_BOUND",
        },
        "completion_policy": payload.get("completion_policy"),
        "completion_basis": payload.get("completion_basis"),
        "completion_reason": payload.get("completion_reason"),
        "accepted": payload.get("accepted"),
        "acceptance_scope": payload.get("acceptance_scope"),
        "repository_integration_receipt": payload.get("repository_integration_receipt"),
        "storage_evidence_roots": payload.get("storage_evidence_roots"),
        "storage_merge_closeout": payload.get("storage_merge_closeout"),
        "terminal_decision_receipt": payload.get("terminal_decision_receipt"),
        "accepted_review_session_id": payload.get("accepted_review_session_id"),
        "acceptance_evidence": payload.get("acceptance_evidence"),
        "repository_truth": repository_truth,
        "administrative_reconciliation": payload.get("administrative_reconciliation"),
        "lifecycle_repair": payload.get("lifecycle_repair"),
        "resource_estimate": projected_estimate,
        "resource_accounting": projected_accounting,
        "updated_at_utc": record.get("updated_at_utc"),
        "created_at_utc": record.get("created_at_utc"),
        "revision": record.get("revision"),
        "valid": validation["valid"],
        "validation": validation,
        "dependencies": derived["dependencies"],
        "sessions": derived["sessions"],
        "active_session_count": derived["active_session_count"],
        "review": derived["review"],
        "next_actions": derived["next_actions"],
        "references": payload.get("references"),
        "record_sha256": record.get("record_sha256"),
    }


def summarize_work_record(
    record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    read_only: bool = False,
    repo_root: str | Path | None = None,
    snapshot: PortfolioSnapshot | None = None,
) -> dict[str, Any]:
    """Public summary with an explicit forensic read-only mode."""

    return _summary(
        record,
        state_root=state_root,
        read_only=read_only,
        repo_root=repo_root,
        snapshot=snapshot,
    )


def scheduler_eligibility_candidate_for_work(
    record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    snapshot: PortfolioSnapshot | None = None,
) -> dict[str, Any]:
    """Return deterministic scheduler eligibility facts without scheduling work.

    This is a read-only Work Manager query.  It does not inspect Dispatch
    Requests, claim work, prepare Sessions, or launch workers.
    """

    summary = _summary(record, state_root=state_root, snapshot=snapshot)
    payload = dict(record.get("payload") or {})
    reasons: list[str] = []
    eligible = True
    phase: str | None = None
    existing_session_id: str | None = None
    review_target_session_id: str | None = None

    if record.get("status") not in WORK_OPEN_STATUSES:
        eligible = False
        reasons.append("work_not_open")
    if not summary["valid"]:
        eligible = False
        reasons.append("work_invalid")
    if not payload.get("work_package_binding"):
        eligible = False
        reasons.append("work_package_missing")
    dependencies = dict(summary.get("dependencies") or {})
    if not dependencies.get("all_satisfied", False):
        eligible = False
        reasons.append("dependencies_unresolved")
    derived = str(summary.get("derived_status") or "")
    if derived in {"PROMOTION_REQUIRED", "READY_TO_ACCEPT", "BLOCKED_BY_DEPENDENCY", "BLOCKED_BY_TERMINAL_DEPENDENCY"}:
        eligible = False
        reasons.append(derived.lower())

    sessions = list(summary.get("sessions") or [])
    current_package_sha = str(dict(payload.get("work_package_binding") or {}).get("sha256") or "")
    # Authority transitions are current repository truth and always outrank
    # historical resumability. Old interrupted implementation/review Sessions
    # may not bypass a required review, repair, acceptance, or promotion gate.
    if derived == "REVIEW_REQUIRED":
        phase = "review"
        review_target_session_id = str(dict(summary.get("review") or {}).get("latest_delivery_session_id") or "") or None
    elif derived == "CHANGES_REQUIRED":
        phase = "repair"
        current_review_binding = authoritative_review_binding_for_work(
            record,
            state_root=state_root,
            read_only=False,
            snapshot=snapshot,
        )
        repair_sessions = [
            row
            for row in sessions
            if row.get("status") in {"CREATED", "PAUSED", "INTERRUPTED", "BLOCKED"}
            and row.get("worker_identity_state") != "PENDING"
            and not row.get("prepared_assignment_id")
            and row.get("session_role") == "IMPLEMENTATION_AUTHORITY"
            and str(dict(row.get("work_package_binding") or {}).get("sha256") or current_package_sha)
            == current_package_sha
            and dict(row.get("authoritative_review_binding") or {}) == dict(current_review_binding or {})
            and str(row.get("session_outcome") or "") != "PROMOTION_REQUESTED"
        ]
        if repair_sessions:
            repair_sessions.sort(
                key=lambda row: (
                    str(row.get("updated_at_utc") or ""),
                    str(row.get("session_id") or ""),
                ),
                reverse=True,
            )
            existing_session_id = str(repair_sessions[0].get("session_id") or "") or None
    else:
        resumable = [
            row for row in sessions
            if row.get("status") in {"CREATED", "PAUSED", "INTERRUPTED", "BLOCKED"}
            and row.get("worker_identity_state") != "PENDING"
            and not row.get("prepared_assignment_id")
            and str(dict(row.get("work_package_binding") or {}).get("sha256") or current_package_sha) == current_package_sha
            and str(row.get("session_outcome") or "") != "PROMOTION_REQUESTED"
        ]
        manual_only = [
            row
            for row in resumable
            if str(row.get("session_role") or "")
            not in {"IMPLEMENTATION_AUTHORITY", "REVIEW_AUTHORITY"}
        ]
        if manual_only:
            eligible = False
            reasons.append("higher_authority_manual_only")
            resumable = []
        if resumable:
            resumable.sort(key=lambda row: (str(row.get("updated_at_utc") or ""), str(row.get("session_id") or "")), reverse=True)
            existing_session_id = str(resumable[0].get("session_id") or "") or None
            role = str(resumable[0].get("session_role") or "")
            phase = {
                "IMPLEMENTATION_AUTHORITY": "implementation",
                "REVIEW_AUTHORITY": "review",
            }.get(role)
            if phase is None:
                eligible = False
                reasons.append("higher_authority_manual_only")
                existing_session_id = None
    if eligible and phase is None:
        if summary.get("active_session_count"):
            eligible = False
            reasons.append("active_session_exists")
        else:
            binding = dict(payload.get("work_package_binding") or {})
            recommended = str(binding.get("recommended_authority") or "IMPLEMENTATION_AUTHORITY")
            if recommended in {"ARCHITECTURE_AUTHORITY", "ENGINEERING_AUTHORITY"}:
                eligible = False
                reasons.append("higher_authority_manual_only")
            else:
                phase = {
                    "IMPLEMENTATION_AUTHORITY": "implementation",
                    "REVIEW_AUTHORITY": "review",
                }.get(recommended)
                if phase is None:
                    eligible = False
                    reasons.append("unsupported_delivery_authority")

    if eligible:
        reasons.append("eligible")
    return {
        "candidate_kind": "SCHEDULER_ELIGIBILITY_CANDIDATE",
        "work_id": str(record.get("record_id") or ""),
        "title": summary.get("title"),
        "priority": summary.get("priority"),
        "status": record.get("status"),
        "derived_status": derived,
        "eligible": eligible,
        "reason_codes": reasons,
        "authority_phase": phase,
        "existing_session_id": existing_session_id,
        "review_target_session_id": review_target_session_id,
        "work_package_binding": payload.get("work_package_binding"),
        "created_at_utc": record.get("created_at_utc"),
    }


def assignment_candidate_for_work(
    record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    snapshot: PortfolioSnapshot | None = None,
) -> dict[str, Any]:
    """Compatibility alias for the namespaced Scheduler eligibility projection."""

    return scheduler_eligibility_candidate_for_work(
        record, state_root=state_root, snapshot=snapshot
    )


def eligible_assignment_candidates(*, state_root: str | Path | None) -> list[dict[str, Any]]:
    snapshot = build_portfolio_snapshot(state_root, read_only=False)
    rows: list[dict[str, Any]] = []
    for work_id, record in snapshot.work_by_id.items():
        rows.append(
            scheduler_eligibility_candidate_for_work(
                record,
                state_root=state_root,
                snapshot=snapshot,
            )
        )
    phase_rank = {"implementation": 3, "repair": 1, "review": 2, "architecture": 4, "engineering": 4}
    rows.sort(key=lambda row: (
        not bool(row["eligible"]),
        PRIORITY_RANK.get(str(row.get("priority")), 99),
        phase_rank.get(str(row.get("authority_phase")), 99),
        str(row.get("created_at_utc") or ""),
        str(row.get("work_id") or ""),
    ))
    return rows


def command_automation_candidates(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    rows = eligible_assignment_candidates(state_root=args.state_root)
    if args.eligible_only:
        rows = [row for row in rows if row["eligible"]]
    return with_guidance(
        {"ok": True, "candidate_count": len(rows), "candidates": rows},
        status="AUTOMATION_CANDIDATES_READY",
        reason_codes=["work_automation_candidates"],
        next_actions=[],
    ), 0


def _transition(
    record: Mapping[str, Any],
    new_status: str,
    *,
    allow_reviewed_ready_completion: bool = False,
    allow_incomplete_chain_repair: bool = False,
    allow_discontinued_reopen: bool = False,
) -> None:
    current = str(record.get("status") or "BACKLOG")
    if current == "READY" and new_status == "COMPLETED" and allow_reviewed_ready_completion:
        return
    if current == "COMPLETED" and new_status == "READY" and allow_incomplete_chain_repair:
        return
    if current == "DISCONTINUED" and new_status == "READY" and allow_discontinued_reopen:
        return
    if new_status not in ALLOWED_TRANSITIONS.get(current, set()):
        raise ActorError(
            f"invalid work transition {current} -> {new_status}",
            code="work_transition_invalid",
            problems=[f"Allowed transitions from {current}: {', '.join(sorted(ALLOWED_TRANSITIONS.get(current, set())))}"],
            related_records=[str(record.get("record_id"))],
        )


def _revise_work_record(
    record: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    status: str,
    action: str,
    state_root: str | Path | None,
    generated_at: str | None,
    note: str | None,
    repo_root: str | Path | None = None,
    allow_reviewed_ready_completion: bool = False,
    allow_incomplete_chain_repair: bool = False,
    allow_discontinued_reopen: bool = False,
    allow_historical_package_binding: bool = False,
) -> dict[str, Any]:
    """Single executable gateway for every durable Work revision."""

    _transition(
        record,
        status,
        allow_reviewed_ready_completion=allow_reviewed_ready_completion,
        allow_incomplete_chain_repair=allow_incomplete_chain_repair,
        allow_discontinued_reopen=allow_discontinued_reopen,
    )
    payload = dict(payload)
    if status in WORK_TERMINAL_STATUSES and str(record.get("status") or "") != status:
        reason = str(
            note
            or payload.get("completion_reason")
            or payload.get("blocked_reason")
            or dict(payload.get("administrative_reconciliation") or {}).get("reason")
            or f"{action} through bounded Work Manager lifecycle command"
        ).strip()
        payload["terminal_transition"] = {
            "schema_version": "eiel_terminal_transition_v1",
            "reason": reason,
            "authority": {
                "actor": ACTOR_NAME,
                "role": "HUMAN_OPERATOR",
                "identity_assurance": "DECLARED",
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
    validation = _validate_payload(
        payload,
        state_root=state_root,
        existing_id=str(record.get("record_id")),
        status=status,
        repo_root=repo_root,
        allow_historical_package_binding=allow_historical_package_binding,
    )
    if not validation["valid"]:
        raise ActorError(
            "work item validation failed",
            code="work_validation_failed",
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
    work_store(state_root, write_scope=True).put(revised, allow_update=True)
    return revised


def _revise(
    args: argparse.Namespace,
    record: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    status: str,
    action: str,
    note: str | None,
    allow_reviewed_ready_completion: bool = False,
    allow_incomplete_chain_repair: bool = False,
    allow_discontinued_reopen: bool = False,
    allow_historical_package_binding: bool = False,
) -> dict[str, Any]:
    return _revise_work_record(
        record,
        payload=payload,
        state_root=args.state_root,
        status=status,
        action=action,
        generated_at=args.generated_at,
        note=note,
        repo_root=getattr(args, "repo_root", None),
        allow_reviewed_ready_completion=allow_reviewed_ready_completion,
        allow_incomplete_chain_repair=allow_incomplete_chain_repair,
        allow_discontinued_reopen=allow_discontinued_reopen,
        allow_historical_package_binding=allow_historical_package_binding,
    )


def attach_work_resource_accounting(
    work_id: str,
    accounting: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    generated_at: str | None = None,
    action: str = "finalize_resource_accounting",
    note: str | None = None,
) -> dict[str, Any]:
    """Attach a WRA projection without transferring Work write authority."""

    normalized_id = validate_prefixed_id(work_id, WORK_ID_PREFIX, label="work item")
    with work_lifecycle_lock(state_root, normalized_id):
        record = work_store(state_root).get(normalized_id)
        if str(record.get("status") or "") not in WORK_TERMINAL_STATUSES:
            raise ActorError(
                "Work resource accounting may be finalized only after terminal disposition",
                code="token_finalize_work_not_terminal",
                related_records=[normalized_id],
            )
        payload = dict(record.get("payload") or {})
        estimate = load_resource_estimate(payload.get("resource_estimate"))
        normalized = load_work_resource_accounting(accounting, estimate=estimate)
        if normalized == payload.get("resource_accounting"):
            return dict(record)
        payload["resource_accounting"] = normalized
        payload = _normalize_payload(
            payload,
            work_id=normalized_id,
            generated_at=str(record.get("created_at_utc") or utc_now(generated_at)),
        )
        return _revise_work_record(
            record,
            payload=payload,
            status=str(record.get("status") or "BACKLOG"),
            action=action,
            state_root=state_root,
            generated_at=generated_at,
            note=note,
        )


def _guided(payload: Mapping[str, Any], record: Mapping[str, Any], *, state_root: str | Path | None) -> dict[str, Any]:
    summary = _summary(record, state_root=state_root)
    problems = summary["validation"]["errors"]
    warnings = summary["validation"]["warnings"]
    status = "BLOCKED" if problems else str(summary["derived_status"])
    next_actions = list(summary["next_actions"])
    if problems:
        # A malformed or stale authoritative Work record may be inspected, but
        # none of its cached start/resume/review/accept prose is executable.
        next_actions = [
            {
                "action": "repair_work_integrity",
                "description": "Repair the Work Package, reference, or acceptance evidence before any lifecycle action.",
                "command": f"python -m tools.eiel workManager inspect {summary['work_id']}",
            }
        ]
    return with_guidance(
        payload,
        status=status,
        reason_codes=[f"work_{status.lower()}"],
        problems=problems,
        warnings=warnings,
        next_actions=next_actions,
        related_records=[summary["work_id"], *summary.get("dependency_work_ids", [])],
    )


def _bind_package(
    raw: dict[str, Any],
    *,
    path: str,
    repo_root: str | Path,
    expected_work_id: str | None,
    state_root: str | Path | None,
) -> tuple[dict[str, Any], str]:
    binding, package = bind_work_package(
        path,
        repo_root=repo_root,
        expected_work_id=expected_work_id,
        require_git_provenance=True,
    )
    binding = archive_work_package_contract(binding, package, state_root=state_root)
    existing_dependencies = normalize_string_list(raw.get("dependency_work_ids"))
    package_dependencies = normalize_string_list(package.get("dependencies"))
    if existing_dependencies and sorted(existing_dependencies) != sorted(package_dependencies):
        raise ActorError(
            "declared Work dependencies do not match the Work Package",
            code="work_package_dependency_mismatch",
            problems=[
                f"Work dependencies: {existing_dependencies}",
                f"Work Package dependencies: {package_dependencies}",
            ],
        )
    raw["dependency_work_ids"] = package_dependencies
    raw["work_package_binding"] = binding
    raw["work_package_chain"] = [{"sequence": 1, "binding": binding, "requires_intermediate_review": False}]
    raw["title"] = str(raw.get("title") or package.get("title") or package["work_id"]).strip()
    raw["summary"] = str(raw.get("summary") or package.get("goal") or "").strip()
    raw["accepted_review_session_id"] = None
    return raw, str(package["work_id"])


def _bind_package_chain(
    raw: dict[str, Any],
    *,
    paths: Sequence[str],
    repo_root: str | Path,
    expected_work_id: str | None,
    state_root: str | Path | None,
    intermediate_review_sequences: Sequence[int] = (),
) -> tuple[dict[str, Any], str]:
    if not paths:
        raise ActorError("at least one Work Package is required", code="work_package_chain_invalid")
    bindings: list[dict[str, Any]] = []
    packages: list[dict[str, Any]] = []
    resolved_work_id = expected_work_id
    for path in paths:
        binding, package = bind_work_package(
            path,
            repo_root=repo_root,
            expected_work_id=resolved_work_id,
            require_git_provenance=True,
        )
        if resolved_work_id is None:
            resolved_work_id = str(package["work_id"])
        bindings.append(archive_work_package_contract(binding, package, state_root=state_root))
        packages.append(package)
    work_ids = {str(package["work_id"]) for package in packages}
    if len(work_ids) != 1:
        raise ActorError("all Work Packages in a chain must belong to one Work Item", code="work_package_chain_invalid")
    raw["work_package_binding"] = bindings[0]
    raw["work_package_chain"] = [
        {"sequence": index, "binding": binding, "requires_intermediate_review": index in set(intermediate_review_sequences)}
        for index, binding in enumerate(bindings, start=1)
    ]
    raw["dependency_work_ids"] = normalize_string_list(packages[0].get("dependencies"))
    raw["title"] = str(raw.get("title") or packages[0].get("title") or packages[0]["work_id"]).strip()
    raw["summary"] = str(raw.get("summary") or packages[0].get("goal") or "").strip()
    raw["accepted_review_session_id"] = None
    return raw, str(resolved_work_id or packages[0]["work_id"])


def command_create(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    timestamp = utc_now(args.generated_at)
    raw = read_structured(args.input, name="work item") if args.input else {}
    for key in ("title", "summary", "work_type", "completion_policy", "subsystem", "priority", "mission_id", "parent_work_id", "current_stage", "current_step", "last_completed_step", "next_action", "next_command", "preferred_codex_conversation_id", "preferred_codex_thread_url"):
        value = getattr(args, key, None)
        if value is not None:
            raw[key] = value
    if args.dependency:
        raw["dependency_work_ids"] = [*normalize_string_list(raw.get("dependency_work_ids")), *args.dependency]
    if args.tag:
        raw["tags"] = [*normalize_string_list(raw.get("tags")), *args.tag]
    if getattr(args, "storage_evidence_root", None):
        raw["storage_evidence_roots"] = normalize_string_list(
            [
                *normalize_string_list(raw.get("storage_evidence_roots")),
                *list(args.storage_evidence_root or []),
            ]
        )
    if any(
        value is not None
        for value in (
            getattr(args, "estimated_tokens_low", None),
            getattr(args, "estimated_tokens_expected", None),
            getattr(args, "estimated_tokens_high", None),
            getattr(args, "estimate_confidence", None),
        )
    ):
        current_estimate = dict(raw.get("resource_estimate") or {}) if isinstance(raw.get("resource_estimate"), Mapping) else {}
        token_estimate = dict(current_estimate.get("estimated_tokens") or {})
        for field, value in (
            ("low", getattr(args, "estimated_tokens_low", None)),
            ("expected", getattr(args, "estimated_tokens_expected", None)),
            ("high", getattr(args, "estimated_tokens_high", None)),
        ):
            if value is not None:
                token_estimate[field] = value
        current_estimate["estimated_tokens"] = token_estimate
        if getattr(args, "estimate_confidence", None) is not None:
            current_estimate["confidence"] = args.estimate_confidence
        current_estimate["method"] = "static"
        raw["resource_estimate"] = current_estimate
    resolved_work_id = args.work_id
    if args.work_package:
        raw, package_work_id = _bind_package(
            raw,
            path=args.work_package[0],
            repo_root=args.repo_root or Path.cwd(),
            expected_work_id=args.work_id,
            state_root=args.state_root,
        )
        if len(args.work_package) > 1:
            raw, package_work_id = _bind_package_chain(
                raw,
                paths=args.work_package,
                repo_root=args.repo_root or Path.cwd(),
                expected_work_id=args.work_id,
                state_root=args.state_root,
                intermediate_review_sequences=args.intermediate_review_sequence,
            )
        else:
            for row in raw.get("work_package_chain") or []:
                row["requires_intermediate_review"] = int(row.get("sequence") or 0) in set(args.intermediate_review_sequence)
        resolved_work_id = args.work_id or package_work_id
    payload = _normalize_payload(raw, work_id=resolved_work_id, generated_at=timestamp)
    status = args.status or ("BLOCKED" if payload["dependency_work_ids"] else "BACKLOG")
    if status in WORK_TERMINAL_STATUSES:
        raise ActorError(
            "Work Items must enter terminal states through bounded lifecycle commands, not creation",
            code="work_terminal_initial_state_forbidden",
            problems=[f"Requested initial status: {status}"],
        )
    dependency_snapshot = dependency_state(args.state_root, payload["dependency_work_ids"]) if payload["dependency_work_ids"] else None
    if dependency_snapshot and not dependency_snapshot["all_satisfied"]:
        if status == "ACTIVE":
            raise ActorError("work cannot start while dependencies are unresolved", code="work_dependencies_unresolved")
        if status in {"BACKLOG", "READY"}:
            status = "BLOCKED"
            payload["blocked_reason"] = payload.get("blocked_reason") or "Waiting for dependency work to complete."
    validation = _validate_payload(payload, state_root=args.state_root, status=status)
    if not validation["valid"]:
        raise ActorError("work item validation failed", code="work_validation_failed", problems=validation["errors"], warnings=validation["warnings"])
    record = new_record(
        actor=ACTOR_NAME,
        record_type=WORK_RECORD_TYPE,
        record_id=payload["work_id"],
        status=status,
        payload=payload,
        generated_at=timestamp,
        source_refs=[
            *([{"kind": "input", "uri": str(Path(args.input).resolve())}] if args.input else []),
            *(
                [
                    {
                        "kind": "work_package",
                        "path": dict(payload.get("work_package_binding") or {}).get("path"),
                        "sha256": dict(payload.get("work_package_binding") or {}).get("sha256"),
                    }
                ]
                if payload.get("work_package_binding")
                else []
            ),
        ],
        note=args.note,
    )
    path = _store(args).put(record)
    return _guided({"ok": True, "created": True, "record_path": str(path), "work": _summary(record, state_root=args.state_root)}, record, state_root=args.state_root), 0


def command_list(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    snapshot = build_portfolio_snapshot(args.state_root, read_only=True)
    rows = [dict(row) for row in snapshot.invalid_work_rows]
    for record in snapshot.work_by_id.values():
        payload = dict(record.get("payload") or {})
        if args.status and record.get("status") not in args.status:
            continue
        if args.open_only and record.get("status") not in WORK_OPEN_STATUSES:
            continue
        if args.priority and payload.get("priority") not in args.priority:
            continue
        summary = _summary(
            record,
            state_root=args.state_root,
            read_only=True,
            snapshot=snapshot,
        )
        rows.append(summary)
    if args.view == "priority":
        rows.sort(key=lambda row: (PRIORITY_RANK.get(str(row.get("priority")), 99), str(row.get("created_at_utc")), str(row.get("work_id"))))
    elif args.view == "recent":
        rows.sort(key=lambda row: (str(row.get("updated_at_utc")), str(row.get("work_id"))), reverse=True)
    elif args.view == "blocked":
        rows = [row for row in rows if str(row.get("derived_status", "")).startswith("BLOCKED") or row.get("status") == "BLOCKED"]
        rows.sort(key=lambda row: (PRIORITY_RANK.get(str(row.get("priority")), 99), str(row.get("created_at_utc"))))
    else:
        rows.sort(key=lambda row: (str(row.get("created_at_utc")), str(row.get("work_id"))))
    payload = with_guidance(
        {"ok": True, "count": len(rows), "view": args.view, "work_items": rows},
        status="PORTFOLIO_READY",
        reason_codes=["work_portfolio_listed"],
        next_actions=[{"action": "select_work", "description": "Select a work ID explicitly before resuming when several items are open.", "command": "python -m tools.eiel workManager inspect <WORK-ID>"}] if rows else [{"action": "create_work", "description": "No matching work exists; create an authorized work item.", "command": "python -m tools.eiel workManager create --title '<title>' --summary '<summary>'"}],
        related_records=[str(row.get("work_id")) for row in rows],
    )
    return payload, 0


def command_inspect(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    work_id = validate_prefixed_id(args.work_id, WORK_ID_PREFIX, label="work item")
    record = _store(args).get_read_only(work_id)
    snapshot = build_portfolio_snapshot(args.state_root, read_only=True)
    return _guided(
        {
            "ok": True,
            "record": record,
            "work": _summary(
                record,
                state_root=args.state_root,
                read_only=True,
                repo_root=args.repo_root,
                snapshot=snapshot,
            ),
        },
        record,
        state_root=args.state_root,
    ), 0


def _command_update_locked(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = _store(args).get(args.work_id)
    payload = dict(record.get("payload") or {})
    patch: dict[str, Any] = {}
    if args.patch:
        patch.update(read_structured(args.patch, name="work patch"))
    patch.update(parse_set_values(args.set_values))
    requested_patch_roots = {str(key).split(".", 1)[0] for key in patch}
    # Explicit inventory retained beside the allow-list so static ownership
    # audits can prove every manager-owned root remains outside generic update.
    manager_owned_patch_roots = requested_patch_roots.intersection({
        "created_order_key",
        "work_package_binding",
        "accepted_review_session_id",
        "acceptance_evidence",
        "administrative_reconciliation",
        "superseded_by_work_id",
        "resource_estimate",
        "resource_accounting",
        "package_binding_history",
        "promotion_resolution",
        "terminal_transition",
        "evidence_requirement",
        "hold",
        "delivery_components",
    })
    forbidden = sorted(requested_patch_roots - WORK_UPDATE_EDITABLE_ROOTS)
    if forbidden:
        raise ActorError(
            "generic Work update accepts metadata fields only",
            code="work_managed_field_patch_forbidden",
            problems=[
                "Fields requiring a bounded owner command: " + ", ".join(forbidden),
                *(
                    ["Recognized manager-owned fields: " + ", ".join(sorted(manager_owned_patch_roots))]
                    if manager_owned_patch_roots
                    else []
                ),
                "Editable fields: " + ", ".join(sorted(WORK_UPDATE_EDITABLE_ROOTS)),
            ],
            next_actions=[
                {
                    "action": "use_bounded_command",
                    "description": (
                        "Use Work Package, dependency, lifecycle, review, supersession, "
                        "resource-estimate, or resource-accounting owner APIs for authoritative fields."
                    ),
                }
            ],
        )
    if patch:
        payload = apply_dotted_patch(payload, patch)
    if args.work_package:
        if str(record.get("status")) in WORK_TERMINAL_STATUSES:
            raise ActorError(
                "terminal Work cannot receive or replace a Work Package",
                code="work_package_terminal_work",
                related_records=[args.work_id],
            )
        linked_sessions = sessions_for_work(args.state_root, args.work_id)
        non_terminal = [
            row
            for row in linked_sessions
            if row.get("status") not in SESSION_TERMINAL_STATUSES
        ]
        if non_terminal:
            raise ActorError(
                "cannot replace a Work Package while linked sessions remain non-terminal",
                code="work_package_session_active",
                problems=[f"Sessions: {', '.join(str(row['record_id']) for row in non_terminal)}"],
                next_actions=[{"action": "close_session", "description": "Complete, discontinue, obsolete, or supersede the old bound session before replacing the contract."}],
            )
        previous_binding = dict(payload.get("work_package_binding") or {})
        if linked_sessions and not previous_binding:
            raise ActorError(
                "cannot add the first Work Package after sessions already exist",
                code="work_package_history_unbound",
                problems=["Historical sessions cannot be retroactively assigned an engineering contract."],
                next_actions=[
                    {
                        "action": "create_superseding_work",
                        "description": "Create or supersede with a new Work Item whose package is bound before its first session.",
                    }
                ],
            )
        previous_package = None
        if previous_binding:
            previous_verified = verify_work_package_binding(
                previous_binding,
                repo_root=args.repo_root or previous_binding.get("bound_repo_root") or Path.cwd(),
                expected_work_id=args.work_id,
                verification_mode=VERIFICATION_HISTORICAL_AUDIT,
                state_root=args.state_root,
            )
            if not previous_verified["valid"] or previous_verified.get("package") is None:
                raise ActorError(
                    "existing Work Package binding must be valid before it can be revised",
                    code="work_package_existing_binding_invalid",
                    problems=previous_verified["errors"],
                    warnings=previous_verified["warnings"],
                )
            previous_package = dict(previous_verified["package"])
        if len(args.work_package) > 1:
            revised_payload, _ = _bind_package_chain(
                payload,
                paths=args.work_package,
                repo_root=args.repo_root or Path.cwd(),
                expected_work_id=args.work_id,
                state_root=args.state_root,
                intermediate_review_sequences=args.intermediate_review_sequence,
            )
        else:
            revised_payload, _ = _bind_package(
                payload,
                path=args.work_package[0],
                repo_root=args.repo_root or Path.cwd(),
                expected_work_id=args.work_id,
                state_root=args.state_root,
            )
        revised_binding = dict(revised_payload.get("work_package_binding") or {})
        revised_verified = verify_work_package_binding(
            revised_binding,
            repo_root=args.repo_root or revised_binding.get("bound_repo_root") or Path.cwd(),
            expected_work_id=args.work_id,
            verification_mode=VERIFICATION_EXECUTION_STRICT,
            state_root=args.state_root,
        )
        if not revised_verified["valid"] or revised_verified.get("package") is None:
            raise ActorError(
                "revised Work Package binding is invalid",
                code="work_package_binding_invalid",
                problems=revised_verified["errors"],
                warnings=revised_verified["warnings"],
            )
        revised_package = dict(revised_verified["package"])
        material_changed = bool(
            previous_package is not None
            and not work_package_material_equal(previous_package, revised_package)
        )
        if linked_sessions and material_changed:
            if revised_binding.get("path") == previous_binding.get("path"):
                raise ActorError(
                    "a material Work Package revision must use a new repository path",
                    code="work_package_history_path_reused",
                    problems=[
                        f"Existing path: {previous_binding.get('path')}",
                        "Preserve the old package file so historical sessions remain verifiable.",
                    ],
                )
            if int(revised_binding.get("package_version") or 0) <= int(previous_binding.get("package_version") or 0):
                raise ActorError(
                    "a material Work Package revision must increase package_version",
                    code="work_package_version_not_increased",
                    problems=[
                        f"Existing version: {previous_binding.get('package_version')}",
                        f"Proposed version: {revised_binding.get('package_version')}",
                    ],
                )
        if previous_binding and material_changed:
            decision_basis = dict(revised_package.get("decision_basis") or {})
            revision_reason = str(
                decision_basis.get("reason_summary")
                or getattr(args, "note", None)
                or "Operator-authorized Work Package revision."
            ).strip()
            source_transition = None
            current_review = review_state_for_work(record, state_root=args.state_root)
            promotion_rows = list(current_review.get("promotion_sessions") or [])
            if promotion_rows:
                source_transition = {
                    "schema_version": "eiel_promotion_source_ref_v1",
                    **{key: promotion_rows[0].get(key) for key in (
                        "session_id", "revision", "record_sha256", "package_sha256",
                        "promotion_request_sha256", "requested_uncertainty_class", "recommended_authority",
                    )},
                }
            history_row = {
                "schema_version": "eiel_work_package_rebind_v1",
                "previous_binding": previous_binding,
                "replacement_binding": revised_binding,
                "reason": revision_reason,
                "source_transition": source_transition,
                "changed_material_fields": sorted(
                    key for key in set(previous_package or {}) | set(revised_package)
                    if (previous_package or {}).get(key) != revised_package.get(key)
                ),
                "recorded_at_utc": utc_now(args.generated_at),
            }
            revised_payload["package_binding_history"] = [
                *list(payload.get("package_binding_history") or []), history_row
            ]
            if source_transition:
                revised_payload["promotion_resolution"] = {
                    "schema_version": "eiel_promotion_resolution_v1",
                    "source_promotion": source_transition,
                    "previous_package_binding": previous_binding,
                    "replacement_package_binding": revised_binding,
                    "deciding_authority": {
                        "role": "ENGINEERING_AUTHORITY",
                        "identity_assurance": "DECLARED",
                        "source": "work_package_revision",
                    },
                    "reason": revision_reason,
                    "resolved_at_utc": utc_now(args.generated_at),
                }
        payload = revised_payload
    for key in ("title", "summary", "subsystem", "priority", "mission_id", "current_stage", "current_step", "last_completed_step", "next_action", "next_command", "preferred_codex_conversation_id", "preferred_codex_thread_url"):
        value = getattr(args, key, None)
        if value is not None:
            payload[key] = value
    for field, values in {
        "qualification_record_ids": args.qualification_ref,
        "resource_record_ids": args.resource_ref,
        "evidence_assessment_ids": args.evidence_ref,
        "artifact_paths": args.artifact_ref,
        "report_paths": args.report_ref,
        "librarian_document_ids": args.librarian_document_id,
        "external_urls": args.external_url,
    }.items():
        if values:
            refs = normalize_reference_map(payload.get("references"), fields=REFERENCE_FIELDS)
            refs[field] = normalize_string_list([*refs[field], *values])
            payload["references"] = refs
    if getattr(args, "storage_evidence_root", None):
        payload["storage_evidence_roots"] = normalize_string_list(
            [
                *normalize_string_list(payload.get("storage_evidence_roots")),
                *list(args.storage_evidence_root or []),
            ]
        )
    payload = _normalize_payload(
        payload,
        work_id=str(record.get("record_id")),
        generated_at=str(record.get("created_at_utc") or utc_now(args.generated_at)),
    )
    # Metadata update preserves lifecycle status by construction.
    status = str(record.get("status"))
    revised = _revise(args, record, payload=payload, status=status, action="update", note=args.note)
    return _guided({"ok": True, "work": _summary(revised, state_root=args.state_root)}, revised, state_root=args.state_root), 0


def command_update(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    work_id = validate_prefixed_id(args.work_id, WORK_ID_PREFIX, label="work item")
    with work_lifecycle_lock(args.state_root, work_id):
        return _command_update_locked(args)


def _simple_transition_locked(
    args: argparse.Namespace,
    *,
    status: str,
    action: str,
    payload_patch: Mapping[str, Any] | None = None,
    allow_reviewed_ready_completion: bool = False,
) -> tuple[dict[str, Any], int]:
    record = _store(args).get(args.work_id)
    payload = dict(record.get("payload") or {})
    if payload_patch:
        payload.update(dict(payload_patch))
    if status in {"READY", "ACTIVE"}:
        if payload.get("hold") is not None:
            retained_hold = load_work_hold(payload.get("hold"))
            raise ActorError(
                "work cannot resume while a machine-readable hold remains active",
                code="work_hold_active",
                problems=[f"Hold kind: {retained_hold.get('kind')}", str(retained_hold.get('resume_condition') or '')],
                next_actions=[{"action": "clear_work_hold", "description": "Clear the hold only after its retained resume condition is satisfied."}],
            )
        dep = dependency_state(args.state_root, payload.get("dependency_work_ids") or [])
        if not dep["all_satisfied"]:
            raise ActorError(
                "work dependencies are not complete",
                code="work_dependencies_unresolved",
                problems=[json.dumps(dep, sort_keys=True)],
                next_actions=[{"action": "inspect_dependencies", "description": "Inspect and complete dependency work.", "command": f"python -m tools.eiel workManager graph {args.work_id}"}],
            )
    if status in WORK_TERMINAL_STATUSES:
        non_terminal = [
            row for row in sessions_for_work(args.state_root, args.work_id)
            if row.get("status") not in SESSION_TERMINAL_STATUSES
        ]
        if non_terminal:
            if getattr(args, "force", False):
                raise ActorError(
                    "ordinary forced closure may not strand non-terminal Sessions",
                    code="work_force_terminal_requires_reconcile",
                    problems=[f"Sessions: {', '.join(str(row['record_id']) for row in non_terminal)}"],
                    next_actions=[{
                        "action": "administrative_reconcile",
                        "description": "Use workManager reconcile with explicit no-acceptance acknowledgement, then reconcile each Session.",
                        "command": (
                            f"python -m tools.eiel workManager reconcile {args.work_id} "
                            "--disposition DISCONTINUE --reason '<reason>' --acknowledge-no-acceptance"
                        ),
                    }],
                )
            raise ActorError(
                "work cannot be closed while linked sessions remain non-terminal",
                code="work_active_session_exists",
                problems=[f"Sessions: {', '.join(str(row['record_id']) for row in non_terminal)}"],
                next_actions=[{"action": "close_session", "description": "Pause, interrupt, complete, or discontinue the active session first."}],
            )
    revised = _revise(
        args,
        record,
        payload=payload,
        status=status,
        action=action,
        note=getattr(args, "note", None),
        allow_reviewed_ready_completion=allow_reviewed_ready_completion,
    )
    return _guided({"ok": True, "work": _summary(revised, state_root=args.state_root)}, revised, state_root=args.state_root), 0


def _simple_transition(
    args: argparse.Namespace,
    *,
    status: str,
    action: str,
    payload_patch: Mapping[str, Any] | None = None,
    allow_reviewed_ready_completion: bool = False,
) -> tuple[dict[str, Any], int]:
    work_id = validate_prefixed_id(args.work_id, WORK_ID_PREFIX, label="work item")
    with work_lifecycle_lock(args.state_root, work_id):
        return _simple_transition_locked(
            args,
            status=status,
            action=action,
            payload_patch=payload_patch,
            allow_reviewed_ready_completion=allow_reviewed_ready_completion,
        )


def command_set_evidence_requirement(args: argparse.Namespace):
    work_id = validate_prefixed_id(args.work_id, WORK_ID_PREFIX, label="work item")
    with work_lifecycle_lock(args.state_root, work_id):
        record = _store(args).get(work_id)
        payload = dict(record.get("payload") or {})
        requirement = {
            "schema_version": WORK_EVIDENCE_REQUIREMENT_SCHEMA_VERSION,
            "mode": str(args.mode).upper(),
            "required_assessment_ids": list(args.assessment_id or []),
            "accepted_decisions": list(args.accepted_decision or []),
            "supports_proposition": args.supports_proposition,
            "recorded_at_utc": utc_now(args.generated_at),
        }
        payload["evidence_requirement"] = load_work_evidence_requirement(requirement)
        revised = _revise(args, record, payload=payload, status=str(record.get("status")), action="set_evidence_requirement", note=args.note)
        return _guided({"ok": True, "work": _summary(revised, state_root=args.state_root)}, revised, state_root=args.state_root), 0


def command_hold(args: argparse.Namespace):
    work_id = validate_prefixed_id(args.work_id, WORK_ID_PREFIX, label="work item")
    with work_lifecycle_lock(args.state_root, work_id):
        record = _store(args).get(work_id)
        payload = dict(record.get("payload") or {})
        hold = {
            "schema_version": WORK_HOLD_SCHEMA_VERSION,
            "kind": str(args.kind).upper(),
            "reason": args.reason,
            "authority": {"role": args.authority_role, "identity": args.authority_identity},
            "created_at_utc": utc_now(args.generated_at),
            "resume_condition": args.resume_condition,
            "resume_after_utc": args.resume_after_utc,
        }
        payload["hold"] = load_work_hold(hold)
        payload["next_action"] = args.resume_condition
        revised = _revise(args, record, payload=payload, status="PAUSED", action="hold", note=args.note)
        return _guided({"ok": True, "work": _summary(revised, state_root=args.state_root)}, revised, state_root=args.state_root), 0


def command_clear_hold(args: argparse.Namespace):
    work_id = validate_prefixed_id(args.work_id, WORK_ID_PREFIX, label="work item")
    with work_lifecycle_lock(args.state_root, work_id):
        record = _store(args).get(work_id)
        payload = dict(record.get("payload") or {})
        if payload.get("hold") is None:
            raise ActorError("Work has no active hold", code="work_hold_missing")
        payload["hold"] = None
        payload["next_action"] = args.next_action
        revised = _revise(args, record, payload=payload, status="READY", action="clear_hold", note=args.note)
        return _guided({"ok": True, "work": _summary(revised, state_root=args.state_root)}, revised, state_root=args.state_root), 0


def command_add_delivery_component(args: argparse.Namespace):
    work_id = validate_prefixed_id(args.work_id, WORK_ID_PREFIX, label="work item")
    with work_lifecycle_lock(args.state_root, work_id):
        record = _store(args).get(work_id)
        payload = dict(record.get("payload") or {})
        component = load_delivery_component({
            "schema_version": DELIVERY_COMPONENT_SCHEMA_VERSION,
            "source_session_id": args.source_session_id,
            "source_commit": args.source_commit,
            "contribution_paths": list(args.contribution_path or []),
            "artifact_refs": [],
            "disposition": args.disposition,
            "reason": args.reason,
            "recorded_at_utc": utc_now(args.generated_at),
        })
        existing = list(payload.get("delivery_components") or [])
        if any(canonical_sha256(row) == canonical_sha256(component) for row in existing):
            revised = record
        else:
            payload["delivery_components"] = [*existing, component]
            revised = _revise(args, record, payload=payload, status=str(record.get("status")), action="add_delivery_component", note=args.note)
        return _guided({"ok": True, "work": _summary(revised, state_root=args.state_root)}, revised, state_root=args.state_root), 0


def command_pause(args: argparse.Namespace):
    return _simple_transition(args, status="PAUSED", action="pause", payload_patch={"next_action": args.next_action, "next_command": args.next_command})


def _command_block_locked(args: argparse.Namespace):
    record = _store(args).get(args.work_id)
    payload = dict(record.get("payload") or {})
    payload["dependency_work_ids"] = normalize_string_list([*normalize_string_list(payload.get("dependency_work_ids")), *args.blocked_by])
    payload["blocked_reason"] = args.reason
    payload["next_action"] = args.next_action or "Complete or repair the blocking dependency work."
    payload["next_command"] = args.next_command
    revised = _revise(args, record, payload=payload, status="BLOCKED", action="block", note=args.note)
    return _guided({"ok": True, "work": _summary(revised, state_root=args.state_root)}, revised, state_root=args.state_root), 0


def command_block(args: argparse.Namespace):
    work_id = validate_prefixed_id(args.work_id, WORK_ID_PREFIX, label="work item")
    with work_lifecycle_lock(args.state_root, work_id):
        return _command_block_locked(args)


def command_resume(args: argparse.Namespace):
    return _simple_transition(args, status="READY" if args.ready else "ACTIVE", action="resume", payload_patch={"blocked_reason": None})


def command_reopen(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    work_id = validate_prefixed_id(args.work_id, WORK_ID_PREFIX, label="work item")
    reason = str(args.reason or "").strip()
    if not reason:
        raise ActorError(
            "reopening Work requires a non-empty reason",
            code="work_reopen_reason_required",
            related_records=[work_id],
        )
    with work_lifecycle_lock(args.state_root, work_id):
        store = _store(args)
        record = store.get(work_id)
        status = str(record.get("status") or "")
        if status != "DISCONTINUED":
            raise ActorError(
                "only ordinarily discontinued Work can be reopened",
                code="work_reopen_status_invalid",
                problems=[f"Observed status: {status or '<missing>'}"],
                related_records=[work_id],
            )

        payload = dict(record.get("payload") or {})
        if payload.get("administrative_reconciliation") is not None:
            raise ActorError(
                "administratively reconciled Work cannot be reopened",
                code="work_reopen_administrative_reconciliation",
                problems=["Create replacement Work after repairing the underlying lifecycle inconsistency."],
                related_records=[work_id],
            )
        if payload.get("hold") is not None:
            raise ActorError(
                "discontinued Work with an active hold cannot reopen as READY",
                code="work_reopen_hold_active",
                next_actions=[{
                    "action": "create_replacement_work",
                    "description": "Create replacement Work that preserves the hold and its resume condition.",
                }],
                related_records=[work_id],
            )
        dependencies = dependency_state(args.state_root, payload.get("dependency_work_ids") or [])
        if not dependencies["all_satisfied"]:
            raise ActorError(
                "discontinued Work dependencies are not complete",
                code="work_reopen_dependencies_unresolved",
                problems=[json.dumps(dependencies, sort_keys=True)],
                next_actions=[{
                    "action": "inspect_dependencies",
                    "description": "Resolve the retained dependency graph before reopening Work as READY.",
                    "command": f"python -m tools.eiel workManager graph {work_id}",
                }],
                related_records=[work_id],
            )

        terminal = dict(payload.get("terminal_transition") or {})
        prior_ref = dict(terminal.get("prior_record") or {})
        replacement = dict(terminal.get("replacement_state") or {})
        if (
            terminal.get("action") != "discontinue"
            or replacement.get("record_id") != work_id
            or replacement.get("status") != "DISCONTINUED"
            or int(replacement.get("revision") or 0) != int(record.get("revision") or 0)
        ):
            raise ActorError(
                "Work lacks an ordinary discontinuation transition that can be reopened",
                code="work_reopen_terminal_transition_invalid",
                related_records=[work_id],
            )
        prior_revision = int(prior_ref.get("revision") or 0)
        prior = store.get_revision_read_only(work_id, prior_revision)
        if (
            str(prior.get("record_sha256") or "") != str(prior_ref.get("record_sha256") or "")
            or str(prior.get("record_id") or "") != work_id
            or str(prior.get("status") or "") in WORK_TERMINAL_STATUSES
        ):
            raise ActorError(
                "Work discontinuation does not bind a valid open predecessor",
                code="work_reopen_history_invalid",
                related_records=[work_id],
            )

        prior_payload = dict(prior.get("payload") or {})
        payload["next_action"] = args.next_action or prior_payload.get("next_action")
        payload["next_command"] = args.next_command or prior_payload.get("next_command")
        payload["blocked_reason"] = None
        # The immutable discontinued revision retains the terminal transition.
        # The current open projection must not claim that it remains terminal.
        payload["terminal_transition"] = None
        revised = _revise(
            args,
            record,
            payload=payload,
            status="READY",
            action="reopen",
            note=reason if not args.note else f"{reason} Note: {str(args.note).strip()}",
            allow_discontinued_reopen=True,
        )
        return _guided(
            {"ok": True, "work": _summary(revised, state_root=args.state_root)},
            revised,
            state_root=args.state_root,
        ), 0


def _qualification_completion_evidence(
    record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    repo_root: str | Path | None,
    generated_at: str | None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    payload = dict(record.get("payload") or {})
    references = normalize_reference_map(payload.get("references"), fields=REFERENCE_FIELDS)
    decision_ids = list(references.get("qualification_record_ids") or [])
    if len(decision_ids) != 1:
        raise ActorError(
            "qualification Work completion requires exactly one retained technical decision",
            code="work_qualification_decision_required",
            problems=[f"Observed qualification references: {decision_ids}"],
        )
    decision_record = qualification_store(state_root).get(decision_ids[0])
    if str(decision_record.get("record_type") or "") != "technical_qualification_decision":
        raise ActorError(
            "qualification reference is not a technical qualification decision",
            code="work_qualification_decision_invalid",
        )
    decision = dict(dict(decision_record.get("payload") or {}).get("decision") or {})
    status = str(decision_record.get("status") or "")
    if (
        status not in {"QUALIFIED", "QUALIFIED_WITH_LIMITS"}
        or not bool(decision.get("ok"))
        or not bool(decision.get("accepted_for_envelope_expansion"))
    ):
        raise ActorError(
            "technical qualification decision does not admit the bounded envelope",
            code="work_qualification_not_admitted",
            problems=[f"Decision status: {status}", *list(decision.get("errors") or [])],
            related_records=[decision_ids[0]],
        )
    bindings = [dict(row) for row in decision.get("evidence_bindings") or [] if isinstance(row, Mapping)]
    if not bindings:
        raise ActorError(
            "technical qualification decision has no content-addressed evidence bindings",
            code="work_qualification_evidence_missing",
            related_records=[decision_ids[0]],
        )
    source_commit = str(decision.get("source_commit") or "").strip().lower()
    if not repo_root:
        raise ActorError(
            "qualification Work completion requires --repo-root for current-support validation",
            code="work_qualification_repository_required",
        )
    repository = git_repository_identity(repo_root)
    if source_commit != str(repository.get("head") or "").lower():
        raise ActorError(
            "technical qualification decision is stale for the current repository",
            code="work_qualification_stale",
            problems=[f"Qualified commit: {source_commit}", f"Current HEAD: {repository.get('head')}"],
            related_records=[decision_ids[0]],
        )
    limitations = normalize_string_list(decision.get("limitations"))
    acceptance_scope = "WITH_LIMITS" if limitations else "TECHNICAL_ONLY"
    evidence = {
        "schema_version": WORK_QUALIFICATION_EVIDENCE_SCHEMA_VERSION,
        "qualification_record": actor_record_ref(decision_record),
        "decision_status": status,
        "source_commit": source_commit,
        "repository_state_sha256": repository.get("repository_state_sha256"),
        "acceptance_scope": acceptance_scope,
        "limitations": limitations,
        "evidence_bindings": bindings,
        "recorded_at_utc": utc_now(generated_at),
    }
    return evidence, acceptance_scope, decision_record


def _command_complete_locked(args: argparse.Namespace):
    record = _store(args).get(args.work_id)
    payload = dict(record.get("payload") or {})
    completion_policy = str(payload.get("completion_policy") or "").upper()
    patch: dict[str, Any] = {
        "last_completed_step": args.last_completed_step,
        "next_action": None,
        "next_command": None,
    }
    if getattr(args, "storage_evidence_root", None):
        patch["storage_evidence_roots"] = normalize_string_list(
            [
                *normalize_string_list(payload.get("storage_evidence_roots")),
                *list(args.storage_evidence_root or []),
            ]
        )
    qualification_record: dict[str, Any] | None = None

    if completion_policy == "REPOSITORY_DELIVERY" and not payload.get("work_package_binding"):
        # Older records and explicitly formalized placeholders may retain the
        # repository-delivery policy without ever acquiring a package. Closing
        # such a record as RECORD_ONLY is safe: it preserves history while
        # asserting neither delivery acceptance nor repository provenance.
        completion_policy = "RECORD_ONLY"
        patch["completion_policy"] = completion_policy

    if completion_policy == "REPOSITORY_DELIVERY":
        review = review_state_for_work(record, state_root=args.state_root)
        if review.get("delivery_conflict"):
            conflict_ids = list(review.get("conflicting_delivery_session_ids") or [])
            raise ActorError(
                "Work Package has divergent or unverifiable delivery lineage",
                code="work_delivery_lineage_conflict",
                problems=[
                    *list(review.get("delivery_lineage_problems") or []),
                    *(["Conflicting delivery sessions: " + ", ".join(conflict_ids)] if conflict_ids else []),
                ],
                related_records=[args.work_id, *conflict_ids],
            )
        if review.get("review_conflict"):
            conflict_ids = list(review.get("conflicting_review_session_ids") or [])
            raise ActorError(
                "current Review Authority verdicts conflict and require explicit supersession or adjudication",
                code="work_review_conflict",
                problems=list(review.get("review_conflict_problems") or []),
                related_records=[args.work_id, *conflict_ids],
            )
        if review["promotion_sessions"]:
            promotion_ids = sorted(str(row["session_id"]) for row in review["promotion_sessions"])
            raise ActorError(
                "Work Package has an unresolved promotion and cannot be completed",
                code="work_promotion_unresolved",
                problems=["Promotion sessions: " + ", ".join(promotion_ids)],
                related_records=[args.work_id, *promotion_ids],
            )
        selected = str(args.review_session or "").strip().upper()
        accepted = {str(row["session_id"]): row for row in review["accepted_review_sessions"]}
        if not selected:
            candidates = sorted(accepted)
            raise ActorError(
                "Work Package delivery requires an explicit evidence-entailing accepted review",
                code="work_review_required",
                problems=[
                    "No --review-session was supplied.",
                    *([f"Accepted review sessions: {', '.join(candidates)}"] if candidates else ["No valid ACCEPT review session exists."]),
                ],
            )
        if selected not in accepted:
            observed = next((row for row in review["review_sessions"] if str(row["session_id"]) == selected), None)
            problems = [
                *list(dict(observed or {}).get("problems") or []),
                *list(dict(observed or {}).get("eligibility_reasons") or []),
            ]
            if observed:
                problems.append(f"Review verdict: {observed.get('verdict')}")
            raise ActorError(
                "review session does not authorize Work completion",
                code="work_review_not_accepted",
                problems=problems or ["The named Session is not a current evidence-entailing ACCEPT review."],
                related_records=[args.work_id, selected],
            )
        accepted_review = session_store(args.state_root).get(selected)
        accepted_payload = dict(accepted_review.get("payload") or {})
        review_evidence = dict(accepted_payload.get("review_evidence") or {})
        if (
            delivery_sufficiency(review_evidence) != "SUFFICIENT"
            or not isinstance(review_evidence.get("obligation_proof"), Mapping)
        ):
            raise ActorError(
                "accepted review does not prove implementation sufficiency and obligation truth",
                code="work_review_entailment_missing",
            )
        implementation_id = delivery_session_id(review_evidence)
        implementation = session_store(args.state_root).get(implementation_id)
        implementation_payload = dict(implementation.get("payload") or {})
        delivery_evidence = dict(implementation_payload.get("delivery_evidence") or {})
        delivery_proof_ref = delivery_evidence.get("delivery_proof")
        if not isinstance(delivery_proof_ref, Mapping):
            raise ActorError(
                "accepted delivery has no durable repository proof",
                code="work_delivery_proof_missing",
            )
        repo_root = str(args.repo_root or delivery_evidence.get("repo_root") or "").strip()
        if not repo_root:
            raise ActorError("accepted delivery has no repository root", code="work_repository_missing")
        current_repository = git_repository_identity(repo_root)
        current_head = str(current_repository.get("head") or "").lower()
        reviewed_commit = str(review_evidence.get("reviewed_commit") or "").lower()
        requested_integration = str(getattr(args, "integration_commit", None) or "").strip().lower()
        if current_head != reviewed_commit and not requested_integration:
            raise ActorError(
                "repository advanced beyond the reviewed delivery; explicit integration verification is required",
                code="work_repository_advanced_since_review",
                problems=[f"Reviewed: {reviewed_commit}", f"Current HEAD: {current_head}"],
                next_actions=[{
                    "action": "verify_integration",
                    "description": "Name the current integration commit so EIEL can verify reviewed-path content equivalence.",
                    "command": f"python -m tools.eiel workManager complete {args.work_id} --review-session {selected} --integration-commit {current_head}",
                }],
            )
        integration_commit = requested_integration or current_head
        if integration_commit != current_head:
            raise ActorError(
                "--integration-commit must equal the current repository HEAD",
                code="work_integration_commit_mismatch",
                problems=[f"Requested: {integration_commit}", f"Current HEAD: {current_head}"],
            )
        _, integration_ref = capture_integration_receipt(
            state_root=args.state_root,
            repo_root=repo_root,
            work_id=args.work_id,
            review_session_id=selected,
            delivery_proof_ref=dict(delivery_proof_ref),
            integration_commit=integration_commit,
            note=getattr(args, "integration_note", None),
            generated_at=args.generated_at,
        )
        acceptance_scope = str(review_evidence.get("acceptance_scope") or "UNCONDITIONAL")
        patch.update({
            "completion_basis": "ACCEPTED_REPOSITORY_DELIVERY",
            "completion_reason": "Exact delivery, obligation proof, evidence-entailing review, and repository integration verified.",
            "accepted": True,
            "acceptance_scope": acceptance_scope,
            "accepted_review_session_id": selected,
            "repository_integration_receipt": integration_ref,
            "acceptance_evidence": {
                "schema_version": WORK_ACCEPTANCE_EVIDENCE_SCHEMA_VERSION,
                "decision_snapshot": lifecycle_decision_snapshot(
                    work_record=record,
                    session_record=accepted_review,
                    implementation_record=implementation,
                    review_record=accepted_review,
                    package_sha256=str(dict(payload.get("work_package_binding") or {}).get("sha256") or ""),
                    delivery_commit=reviewed_commit,
                ),
                "accepted_review": actor_record_ref(accepted_review),
                "accepted_delivery": actor_record_ref(implementation),
                "obligation_proof": dict(review_evidence["obligation_proof"]),
                "repository_integration_receipt": integration_ref,
                "acceptance_scope": acceptance_scope,
                "known_limitations": list(review_evidence.get("known_limitations") or []),
                "recorded_at_utc": utc_now(args.generated_at),
            },
        })
    elif completion_policy == "QUALIFICATION_DECISION":
        if str(payload.get("work_type") or "") != "QUALIFICATION":
            raise ActorError(
                "QUALIFICATION_DECISION completion is reserved for QUALIFICATION Work",
                code="work_qualification_scope_invalid",
            )
        qualification_evidence, acceptance_scope, qualification_record = _qualification_completion_evidence(
            record,
            state_root=args.state_root,
            repo_root=args.repo_root,
            generated_at=args.generated_at,
        )
        patch.update({
            "completion_basis": "ADMITTED_TECHNICAL_QUALIFICATION",
            "completion_reason": "Exact evidence-bound technical qualification decision admitted the bounded envelope for the current repository.",
            "accepted": True,
            "acceptance_scope": acceptance_scope,
            "accepted_review_session_id": None,
            "acceptance_evidence": qualification_evidence,
            "repository_integration_receipt": None,
        })
    elif completion_policy == "RECORD_ONLY":
        if payload.get("work_package_binding"):
            raise ActorError(
                "repository-delivery Work cannot be converted to record-only completion",
                code="work_record_only_scope_invalid",
            )
        reason = str(
            getattr(args, "record_only_reason", None)
            or args.last_completed_step
            or "Operator-authorized record-only completion."
        ).strip()
        delivery_sessions = [
            row for row in sessions_for_work(args.state_root, args.work_id)
            if row.get("status") == "COMPLETED"
            and str(dict(row.get("payload") or {}).get("session_outcome") or "") in {
                "ARCHITECTURE_COMPLETE", "ENGINEERING_COMPLETE", "IMPLEMENTATION_COMPLETE",
                "DELIVERY_RECORDED", "EXTERNAL_DELIVERY_ADOPTED",
            }
        ]
        if delivery_sessions:
            raise ActorError(
                "record-only Work cannot hide retained repository delivery Sessions",
                code="work_record_only_delivery_conflict",
                related_records=[str(row.get("record_id") or "") for row in delivery_sessions],
            )
        patch.update({
            "completion_basis": "RECORD_ONLY_COMPLETION",
            "completion_reason": reason,
            "accepted": False,
            "acceptance_scope": "NOT_APPLICABLE",
            "accepted_review_session_id": None,
            "acceptance_evidence": None,
            "repository_integration_receipt": None,
        })
    else:
        raise ActorError("unsupported Work completion_policy", code="work_completion_policy_invalid")

    usage_artifact, usage_reference = aggregate_work_usage(
        record,
        sessions_for_work(args.state_root, args.work_id),
        state_root=args.state_root,
        generated_at=args.generated_at,
    )
    patch["resource_accounting"] = work_accounting_projection(
        payload, artifact_reference=usage_reference, artifact=usage_artifact
    )
    _, terminal_reference = retain_work_terminal_decision_receipt(
        record,
        state_root=args.state_root,
        final_status="COMPLETED",
        completion_policy=completion_policy,
        completion_basis=str(patch.get("completion_basis") or ""),
        completion_reason=str(patch.get("completion_reason") or ""),
        accepted=bool(patch.get("accepted")),
        acceptance_scope=str(patch.get("acceptance_scope") or ""),
        accepted_review_session_id=str(patch.get("accepted_review_session_id") or "") or None,
        acceptance_evidence=(
            dict(patch.get("acceptance_evidence") or {})
            if isinstance(patch.get("acceptance_evidence"), Mapping)
            else None
        ),
        repository_integration_receipt=(
            dict(patch.get("repository_integration_receipt") or {})
            if isinstance(patch.get("repository_integration_receipt"), Mapping)
            else None
        ),
        qualification_decision=(
            actor_record_ref(qualification_record) if qualification_record is not None else None
        ),
        dependency_state=dependency_state(args.state_root, payload.get("dependency_work_ids") or []),
        generated_at=args.generated_at,
    )
    patch["terminal_decision_receipt"] = terminal_reference
    transition = _simple_transition_locked(
        args,
        status="COMPLETED",
        action="complete",
        payload_patch=patch,
        allow_reviewed_ready_completion=bool(
            str(record.get("status") or "") == "READY"
            and patch.get("accepted_review_session_id")
        ),
    )
    if (
        completion_policy == "REPOSITORY_DELIVERY"
        and isinstance(patch.get("repository_integration_receipt"), Mapping)
    ):
        from tools.actors.storage_lifecycle import closeout_merged_work_storage_v1

        completed = _store(args).get(args.work_id)
        completed_payload = dict(completed.get("payload") or {})
        closeout = closeout_merged_work_storage_v1(
            work_id=args.work_id,
            payload=completed_payload,
            integration_reference=dict(patch["repository_integration_receipt"]),
            repo_root=str(args.repo_root or repo_root),
            state_root=args.state_root,
            generated_at=args.generated_at,
        )
        completed_payload["storage_merge_closeout"] = dict(closeout["receipt"])
        completed_payload = _normalize_payload(
            completed_payload,
            work_id=args.work_id,
            generated_at=str(completed.get("created_at_utc") or utc_now(args.generated_at)),
        )
        revised = _revise(
            args,
            completed,
            payload=completed_payload,
            status="COMPLETED",
            action="storage_merge_closeout",
            note=args.note,
        )
        response = dict(transition[0])
        response["storage_merge_closeout"] = closeout
        response["work"] = _summary(revised, state_root=args.state_root)
        return response, transition[1]
    return transition


def command_complete(args: argparse.Namespace):
    work_id = validate_prefixed_id(args.work_id, WORK_ID_PREFIX, label="work item")
    with work_lifecycle_lock(args.state_root, work_id):
        return _command_complete_locked(args)


def _command_repair_chain_locked(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """Reopen only a terminal Work closed at an intermediate chain stage."""

    record = _store(args).get(args.work_id)
    if str(record.get("status") or "") != "COMPLETED":
        raise ActorError(
            "incomplete-chain repair requires a completed Work record",
            code="work_chain_repair_not_completed",
            problems=[f"Observed status: {record.get('status')}"],
        )
    if not args.acknowledge_intermediate_completion:
        raise ActorError(
            "incomplete-chain repair requires explicit acknowledgement of intermediate completion",
            code="work_chain_repair_ack_required",
            next_actions=[
                {
                    "action": "acknowledge_intermediate_completion",
                    "description": "Repeat with --acknowledge-intermediate-completion after confirming the accepted review was not final.",
                }
            ],
        )

    payload = dict(record.get("payload") or {})
    package_chain = normalize_work_package_chain(
        payload.get("work_package_chain"),
        fallback=payload.get("work_package_binding"),
    )
    if len(package_chain) < 2:
        raise ActorError(
            "incomplete-chain repair requires a multi-package Work Package chain",
            code="work_chain_repair_not_multi_package",
        )
    review = review_state_for_work(record, state_root=args.state_root)
    active_sessions = [
        str(row.get("record_id"))
        for row in sessions_for_work(args.state_root, args.work_id)
        if row.get("status") not in SESSION_TERMINAL_STATUSES
    ]
    if active_sessions:
        raise ActorError(
            "incomplete-chain repair cannot strand active Sessions",
            code="work_chain_repair_active_session",
            problems=["Sessions: " + ", ".join(active_sessions)],
        )
    if review.get("delivery_conflict"):
        raise ActorError(
            "incomplete-chain repair cannot bypass divergent delivery lineage",
            code="work_chain_repair_delivery_conflict",
            problems=list(review.get("delivery_lineage_problems") or []),
        )
    if review.get("promotion_sessions"):
        raise ActorError(
            "incomplete-chain repair cannot bypass an unresolved promotion",
            code="work_chain_repair_promotion_unresolved",
            related_records=[str(row.get("session_id")) for row in review["promotion_sessions"]],
        )
    if review.get("all_packages_delivered"):
        raise ActorError(
            "Work Package chain has no undelivered next package",
            code="work_chain_repair_no_next_package",
        )
    if review.get("accepted_review_sessions"):
        raise ActorError(
            "incomplete-chain repair cannot bypass an accepted final review",
            code="work_chain_repair_final_review_present",
        )

    accepted_review_id = str(payload.get("accepted_review_session_id") or "")
    accepted_review = None
    if accepted_review_id:
        accepted_review = session_store(args.state_root).get_read_only(accepted_review_id)
    accepted_review_payload = dict(accepted_review.get("payload") or {}) if accepted_review else {}
    current_binding = dict(payload.get("work_package_binding") or {})
    if not current_binding:
        raise ActorError(
            "completed Work has no current Work Package binding to repair",
            code="work_chain_repair_binding_missing",
        )
    if (
        not accepted_review_id
        or not accepted_review
        or accepted_review.get("status") != "COMPLETED"
        or accepted_review_payload.get("session_role") != "REVIEW_AUTHORITY"
        or accepted_review_payload.get("review_verdict") != "ACCEPT"
        or str(dict(accepted_review_payload.get("work_package_binding") or {}).get("sha256") or "")
        != str(current_binding.get("sha256") or "")
    ):
        raise ActorError(
            "completed Work does not identify an accepted review for its current intermediate package",
            code="work_chain_repair_intermediate_review_missing",
            problems=[
                "The repair is limited to historical completion after an accepted intermediate package review.",
            ],
        )

    target_id = delivery_session_id(accepted_review_payload.get("review_target"))
    target = session_store(args.state_root).get_read_only(target_id) if target_id else None
    target_payload = dict(target.get("payload") or {}) if target else {}
    if (
        not target
        or target.get("status") != "COMPLETED"
        or target_payload.get("session_role") == "REVIEW_AUTHORITY"
        or str(dict(target_payload.get("work_package_binding") or {}).get("sha256") or "")
        != str(current_binding.get("sha256") or "")
    ):
        raise ActorError(
            "accepted intermediate review does not target a completed delivery for the current package",
            code="work_chain_repair_delivery_target_invalid",
            related_records=[accepted_review_id, target_id] if target_id else [accepted_review_id],
        )

    current_index = next(
        (
            index
            for index, row in enumerate(package_chain)
            if str(dict(row["binding"]).get("sha256") or "")
            == str(current_binding.get("sha256") or "")
        ),
        None,
    )
    repaired_chain = [dict(row) for row in package_chain]
    if current_index is None:
        # A legacy single-package rebind replaced the current chain binding
        # without updating the later chain entries. The next package is the
        # first undelivered chain entry; replace its immediate predecessor.
        first_undelivered = review.get("current_package_index")
        if first_undelivered is None or int(first_undelivered) <= 0:
            raise ActorError(
                "cannot locate the intermediate package sequence in the legacy chain",
                code="work_chain_repair_sequence_unknown",
            )
        current_index = int(first_undelivered) - 1
        repaired_chain[current_index]["binding"] = current_binding
    current_sequence = int(repaired_chain[current_index].get("sequence") or current_index + 1)
    if current_sequence >= len(repaired_chain):
        raise ActorError(
            "accepted review is for the final package; no chain repair is applicable",
            code="work_chain_repair_final_package",
        )
    next_binding = dict(repaired_chain[current_index + 1]["binding"])
    if any(
        str(dict(row["binding"]).get("sha256") or "")
        == str(current_binding.get("sha256") or "")
        for index, row in enumerate(repaired_chain)
        if index != current_index
    ):
        raise ActorError(
            "current Work Package binding would duplicate another chain entry",
            code="work_chain_repair_duplicate_binding",
        )

    first_binding = dict(package_chain[0]["binding"])
    if current_index == 0:
        first_binding = current_binding
    next_sequence = int(repaired_chain[current_index + 1].get("sequence") or current_index + 2)
    timestamp = utc_now(args.generated_at)
    repair = {
        "schema_version": "eiel_work_lifecycle_repair_v1",
        "kind": "reopen_incomplete_package_chain",
        "repaired_at_utc": timestamp,
        "previous_status": str(record.get("status")),
        "previous_record_sha256": str(record.get("record_sha256") or ""),
        "accepted_intermediate_review_session_id": accepted_review_id,
        "accepted_intermediate_package_sha256": str(current_binding.get("sha256") or ""),
        "next_package_sequence": next_sequence,
        "next_package_path": next_binding.get("path"),
        "reason": str(args.reason).strip(),
    }
    payload["work_package_binding"] = first_binding
    payload["work_package_chain"] = repaired_chain
    payload["accepted_review_session_id"] = None
    payload["acceptance_evidence"] = None
    payload["current_stage"] = f"Work Package {next_sequence} implementation"
    payload["current_step"] = f"Start Work Package sequence {next_sequence} after lifecycle repair"
    payload["last_completed_step"] = "Accepted intermediate Work Package review"
    payload["next_action"] = f"Start Work Package sequence {next_sequence} implementation session"
    payload["next_command"] = None
    payload["lifecycle_repair"] = repair
    revised = _revise(
        args,
        record,
        payload=payload,
        status="READY",
        action="repair_incomplete_package_chain",
        note=str(args.reason).strip(),
        allow_incomplete_chain_repair=True,
    )
    return _guided(
        {"ok": True, "repaired": True, "repair": repair, "work": _summary(revised, state_root=args.state_root)},
        revised,
        state_root=args.state_root,
    ), 0


def command_repair_chain(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    work_id = validate_prefixed_id(args.work_id, WORK_ID_PREFIX, label="work item")
    with work_lifecycle_lock(args.state_root, work_id):
        return _command_repair_chain_locked(args)


def _command_reconcile_locked(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """Administratively retire corrupted state without implying acceptance."""

    record = _store(args).get(args.work_id)
    if str(record.get("status")) == "COMPLETED":
        raise ActorError(
            "accepted completed Work may not be administratively reconciled",
            code="work_reconciliation_completed_forbidden",
            related_records=[args.work_id],
        )
    reason = str(args.reason or "").strip()
    if not reason:
        raise ActorError("administrative reconciliation requires a reason", code="work_reconciliation_reason_missing")
    if not args.acknowledge_no_acceptance:
        raise ActorError(
            "administrative reconciliation must explicitly acknowledge that it is not acceptance",
            code="work_reconciliation_ack_required",
            next_actions=[
                {
                    "action": "acknowledge_no_acceptance",
                    "description": "Repeat with --acknowledge-no-acceptance after confirming the Work will not release dependencies.",
                }
            ],
        )
    target = "DISCONTINUED" if args.disposition == "DISCONTINUE" else "OBSOLETE"
    active = [
        str(row.get("record_id"))
        for row in sessions_for_work(args.state_root, args.work_id)
        if row.get("status") not in SESSION_TERMINAL_STATUSES
    ]
    payload = dict(record.get("payload") or {})
    previous_completion_policy = str(payload.get("completion_policy") or "")
    completion_policy = previous_completion_policy.upper()
    repaired_completion_policy = None
    if completion_policy not in WORK_COMPLETION_POLICIES:
        replacement_policy = (
            "REPOSITORY_DELIVERY"
            if payload.get("work_package_binding")
            else "QUALIFICATION_DECISION"
            if str(payload.get("work_type") or "").upper() == "QUALIFICATION"
            else "RECORD_ONLY"
        )
        payload["completion_policy"] = replacement_policy
        repaired_completion_policy = {
            "field": "completion_policy",
            "previous_value": previous_completion_policy,
            "replacement_value": replacement_policy,
        }
    payload["accepted_review_session_id"] = None
    payload["administrative_reconciliation"] = {
        "reason": reason,
        "disposition": target,
        "previous_work_status": str(record.get("status") or ""),
        "active_session_ids_at_reconciliation": active,
        "acceptance_implied": False,
        "dependencies_released": False,
        "recorded_at_utc": utc_now(args.generated_at),
        "recorded_by": "HUMAN_OPERATOR",
    }
    if repaired_completion_policy is not None:
        payload["administrative_reconciliation"]["payload_repairs"] = [repaired_completion_policy]
    payload["next_action"] = (
        "Reconcile each listed Session against the terminal parent Work Item."
        if active
        else None
    )
    payload["next_command"] = (
        f"python -m tools.eiel sessionManager reconcile-terminal-parent {active[0]} --status DISCONTINUED --reason '<same administrative reason>'"
        if active
        else None
    )
    revised = _revise(
        args,
        record,
        payload=payload,
        status=target,
        action="administrative_reconcile",
        note=args.note or f"administrative reconciliation: {reason}",
        allow_historical_package_binding=True,
    )
    return _guided(
        {"ok": True, "accepted": False, "work": _summary(revised, state_root=args.state_root)},
        revised,
        state_root=args.state_root,
    ), 0


def command_reconcile(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    work_id = validate_prefixed_id(args.work_id, WORK_ID_PREFIX, label="work item")
    with work_lifecycle_lock(args.state_root, work_id):
        return _command_reconcile_locked(args)


def command_discontinue(args: argparse.Namespace):
    return _simple_transition(args, status="DISCONTINUED", action="discontinue", payload_patch={"next_action": None, "next_command": None})


def command_obsolete(args: argparse.Namespace):
    return _simple_transition(args, status="OBSOLETE", action="obsolete", payload_patch={"next_action": None, "next_command": None})


def command_supersede(args: argparse.Namespace):
    work_id = validate_prefixed_id(args.work_id, WORK_ID_PREFIX, label="work item")
    replacement_id = validate_prefixed_id(args.with_work_id, WORK_ID_PREFIX, label="replacement work item")
    with work_lifecycle_locks(args.state_root, [work_id, replacement_id]):
        replacement = _store(args).get(replacement_id)
        if replacement.get("status") in WORK_TERMINAL_STATUSES:
            raise ActorError("replacement work is already terminal", code="work_replacement_terminal")
        return _simple_transition_locked(args, status="SUPERSEDED", action="supersede", payload_patch={"superseded_by_work_id": replacement["record_id"], "next_action": None, "next_command": None})


def _command_dependency_locked(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = _store(args).get(args.work_id)
    if str(record.get("status") or "") in WORK_TERMINAL_STATUSES:
        raise ActorError(
            "terminal Work dependency graph is immutable",
            code="work_terminal_dependency_mutation_forbidden",
            related_records=[str(record.get("record_id") or "")],
        )
    payload = dict(record.get("payload") or {})
    dependencies = normalize_string_list(payload.get("dependency_work_ids"))
    if args.operation == "add":
        dependencies = normalize_string_list([*dependencies, args.dependency_work_id])
    else:
        dependencies = [value for value in dependencies if value != args.dependency_work_id]
    payload["dependency_work_ids"] = dependencies
    status = "BLOCKED" if dependencies and not dependency_state(args.state_root, dependencies)["all_satisfied"] else str(record.get("status"))
    revised = _revise(args, record, payload=payload, status=status, action=f"dependency_{args.operation}", note=args.note)
    return _guided({"ok": True, "work": _summary(revised, state_root=args.state_root)}, revised, state_root=args.state_root), 0


def _continuation_successor(
    source: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    generated_at: str | None,
    reason: str,
    resolving: set[str] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Create or reuse a safe successor for terminal unsuccessful Work.

    Historical records are never rewritten.  A successor deliberately starts
    without a copied Work Package binding: a package is an immutable contract
    for its original Work ID and must be explicitly rebound before delivery.
    """

    source_id = str(source.get("record_id") or "")
    source_payload = dict(source.get("payload") or {})
    if str(source.get("status") or "") not in WORK_TERMINAL_STATUSES - {"COMPLETED"}:
        return dict(source), False
    resolving = set(resolving or ())
    if source_id in resolving:
        raise ActorError(
            "continuation cannot resolve a cyclic historical dependency graph",
            code="work_continuation_cycle",
            related_records=sorted([*resolving, source_id]),
        )
    resolving.add(source_id)
    store = work_store(state_root, write_scope=True)
    candidates = []
    for row in store.list_read_only():
        candidate = store.get_read_only(str(row.get("record_id") or ""))
        candidate_payload = dict(candidate.get("payload") or {})
        if (
            str(candidate_payload.get("parent_work_id") or "") == source_id
            and isinstance(candidate_payload.get("continuation"), Mapping)
        ):
            candidates.append(candidate)
    candidates.sort(key=lambda row: (str(row.get("created_at_utc") or ""), str(row.get("record_id") or "")), reverse=True)
    reusable = next((row for row in candidates if str(row.get("status") or "") not in WORK_TERMINAL_STATUSES), None)
    if reusable is not None:
        return dict(reusable), False

    dependencies: list[str] = []
    for dependency_id in normalize_string_list(source_payload.get("dependency_work_ids")):
        dependency = store.get_read_only(dependency_id)
        if str(dependency.get("status") or "") in WORK_TERMINAL_STATUSES - {"COMPLETED"}:
            successor, _ = _continuation_successor(
                dependency,
                state_root=state_root,
                generated_at=generated_at,
                reason=reason,
                resolving=resolving,
            )
            dependencies.append(str(successor.get("record_id") or ""))
        else:
            dependencies.append(dependency_id)
    generation = len(candidates) + 1
    work_id = stable_id("WORK", {"continuation_of": source_id, "generation": generation})
    timestamp = utc_now(generated_at)
    raw = {
        "title": f"{source_payload.get('title') or source_id} — continuation {generation}",
        "summary": source_payload.get("summary") or f"Continuation of {source_id}.",
        "work_type": source_payload.get("work_type") or "ENGINEERING",
        "completion_policy": "RECORD_ONLY",
        "subsystem": source_payload.get("subsystem"),
        "priority": source_payload.get("priority") or "P2",
        "mission_id": source_payload.get("mission_id"),
        "parent_work_id": source_id,
        "dependency_work_ids": dependencies,
        "tags": [*normalize_string_list(source_payload.get("tags")), "operational-continuity"],
        "current_stage": "operational_continuation",
        "current_step": "Reconcile inherited administrative state and continue safe work.",
        "next_action": source_payload.get("next_action") or "Resume the bounded continuation work.",
        "next_command": source_payload.get("next_command"),
        "references": source_payload.get("references"),
        "continuation": {
            "schema_version": "eiel_work_continuation_v1",
            "predecessor": actor_record_ref(source),
            "reason": reason,
            "created_at_utc": timestamp,
            "copied_package_binding": False,
            "administrative_state_reconciled": True,
        },
    }
    payload = _normalize_payload(raw, work_id=work_id, generated_at=timestamp)
    dependency_snapshot = dependency_state(state_root, dependencies) if dependencies else {"all_satisfied": True}
    status = "READY" if dependency_snapshot["all_satisfied"] else "BLOCKED"
    if status == "BLOCKED":
        payload["blocked_reason"] = "Waiting for successful continuation of inherited dependency work."
    validation = _validate_payload(payload, state_root=state_root, status=status)
    if not validation["valid"]:
        raise ActorError("continuation Work validation failed", code="work_continuation_validation_failed", problems=validation["errors"])
    record = new_record(
        actor=ACTOR_NAME,
        record_type=WORK_RECORD_TYPE,
        record_id=work_id,
        status=status,
        payload=payload,
        generated_at=timestamp,
        source_refs=[],
        note=f"Operational continuity: {reason}",
    )
    store.put(record)
    return record, True


def _auto_reconcile_open_work(
    record: Mapping[str, Any], *, args: argparse.Namespace, reason: str
) -> tuple[dict[str, Any], bool]:
    """Clear only machine-resolvable administrative stops on an open Work."""

    payload = dict(record.get("payload") or {})
    changed = False
    hold = payload.get("hold")
    if hold is not None and str(dict(hold).get("kind") or "") in {"REVIEW_DEFERRED", "OPERATOR_DECISION"}:
        payload["hold"] = None
        payload["next_action"] = payload.get("next_action") or "Continue after automatic administrative reconciliation."
        changed = True
    dependencies = dependency_state(args.state_root, payload.get("dependency_work_ids") or [])
    target_status = str(record.get("status") or "BACKLOG")
    if target_status in {"BLOCKED", "PAUSED"} and dependencies["all_satisfied"] and payload.get("hold") is None:
        target_status = "READY"
        payload["blocked_reason"] = None
        changed = True
    if not changed:
        return dict(record), False
    payload["administrative_reconciliation"] = {
        "reason": reason,
        "disposition": "AUTO_CONTINUE",
        "previous_work_status": str(record.get("status") or ""),
        "acceptance_implied": False,
        "dependencies_released": False,
        "recorded_at_utc": utc_now(args.generated_at),
        "recorded_by": "workManager operational continuity resolver",
    }
    return _revise(args, record, payload=payload, status=target_status, action="auto_reconcile_continuation", note=reason), True


def _ensure_qualification_prerequisite(
    record: Mapping[str, Any], *, args: argparse.Namespace, reason: str
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Route a missing qualification decision to bounded prerequisite Work.

    The route records no qualification result.  It only gives the missing
    artifact a runnable, provenance-preserving owner while leaving unrelated
    engineering Work untouched.
    """

    payload = dict(record.get("payload") or {})
    references = normalize_reference_map(payload.get("references"), fields=REFERENCE_FIELDS)
    existing_ids = list(references.get("qualification_record_ids") or [])
    if existing_ids:
        return dict(record), {}, False
    store = work_store(args.state_root, write_scope=True)
    parent_id = str(record.get("record_id") or "")
    candidates = []
    for row in store.list_read_only():
        candidate = store.get_read_only(str(row.get("record_id") or ""))
        candidate_payload = dict(candidate.get("payload") or {})
        continuation = dict(candidate_payload.get("continuation") or {})
        if continuation.get("qualification_prerequisite_for") == parent_id:
            candidates.append(candidate)
    candidates.sort(key=lambda row: (str(row.get("created_at_utc") or ""), str(row.get("record_id") or "")), reverse=True)
    prerequisite = next((row for row in candidates if str(row.get("status") or "") not in WORK_TERMINAL_STATUSES), None)
    created = False
    if prerequisite is None:
        generation = len(candidates) + 1
        timestamp = utc_now(args.generated_at)
        prerequisite_id = stable_id("WORK", {"qualification_prerequisite_for": parent_id, "generation": generation})
        prerequisite_payload = _normalize_payload(
            {
                "title": f"Construct qualification evidence for {parent_id}",
                "summary": "Produce or bind the exact technical qualification decision required by the parent Work.",
                "work_type": "ENGINEERING",
                "completion_policy": "RECORD_ONLY",
                "priority": payload.get("priority") or "P2",
                "parent_work_id": parent_id,
                "tags": ["operational-continuity", "qualification-prerequisite"],
                "current_stage": "qualification_prerequisite",
                "next_action": "Construct or bind the required qualification decision from authoritative current source.",
                "continuation": {
                    "schema_version": "eiel_work_continuation_v1",
                    "qualification_prerequisite_for": parent_id,
                    "reason": reason,
                    "created_at_utc": timestamp,
                    "acceptance_implied": False,
                },
            },
            work_id=prerequisite_id,
            generated_at=timestamp,
        )
        prerequisite = new_record(
            actor=ACTOR_NAME,
            record_type=WORK_RECORD_TYPE,
            record_id=prerequisite_id,
            status="READY",
            payload=prerequisite_payload,
            generated_at=timestamp,
            source_refs=[],
            note="Automatic operational-continuity qualification prerequisite.",
        )
        store.put(prerequisite)
        created = True
    prerequisite_id = str(prerequisite.get("record_id") or "")
    dependencies = normalize_string_list([*(payload.get("dependency_work_ids") or []), prerequisite_id])
    if prerequisite_id not in payload.get("dependency_work_ids") or str(record.get("status") or "") != "BLOCKED":
        payload["dependency_work_ids"] = dependencies
        payload["blocked_reason"] = "Exact qualification decision is not yet bound; automatic prerequisite Work was routed."
        payload["administrative_reconciliation"] = {
            "reason": reason,
            "disposition": "QUALIFICATION_PREREQUISITE_ROUTED",
            "previous_work_status": str(record.get("status") or ""),
            "acceptance_implied": False,
            "dependencies_released": False,
            "recorded_at_utc": utc_now(args.generated_at),
            "recorded_by": "workManager operational continuity resolver",
        }
        record = _revise(args, record, payload=payload, status="BLOCKED", action="route_qualification_prerequisite", note=reason)
    return dict(record), dict(prerequisite), created


def command_continue(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """Resolve administrative lifecycle friction without asserting delivery."""

    work_id = validate_prefixed_id(args.work_id, WORK_ID_PREFIX, label="work item")
    reason = str(args.reason or "Operational continuity requested by operator.").strip()
    with work_lifecycle_lock(args.state_root, work_id):
        record = _store(args).get(work_id)
        status = str(record.get("status") or "")
        if status in WORK_TERMINAL_STATUSES - {"COMPLETED"}:
            successor, created = _continuation_successor(
                record, state_root=args.state_root, generated_at=args.generated_at, reason=reason
            )
            return _guided(
                {
                    "ok": True,
                    "continuation_status": "AUTO_RESOLVED",
                    "blocker": {
                        "blocker_class": "SOFT",
                        "blocker_scope": "work",
                        "operator_intervention_required": False,
                        "reconciliation_action": "successor_created_or_reused",
                        "continuation_possible": True,
                    },
                    "created": created,
                    "historical_work_id": work_id,
                    "continuation_work": _summary(successor, state_root=args.state_root),
                },
                successor,
                state_root=args.state_root,
            ), 0
        payload = dict(record.get("payload") or {})
        if (
            str(payload.get("completion_policy") or "") == "QUALIFICATION_DECISION"
            and not normalize_reference_map(payload.get("references"), fields=REFERENCE_FIELDS).get("qualification_record_ids")
        ):
            revised, prerequisite, created = _ensure_qualification_prerequisite(record, args=args, reason=reason)
            return _guided(
                {
                    "ok": True,
                    "continuation_status": "AUTO_RESOLVED",
                    "blocker": {
                        "blocker_class": "SOFT",
                        "blocker_scope": "action",
                        "operator_intervention_required": False,
                        "reconciliation_action": "qualification_prerequisite_routed",
                        "continuation_possible": True,
                    },
                    "work": _summary(revised, state_root=args.state_root),
                    "prerequisite_work": _summary(prerequisite, state_root=args.state_root),
                    "prerequisite_created": created,
                },
                revised,
                state_root=args.state_root,
            ), 0
        revised, changed = _auto_reconcile_open_work(record, args=args, reason=reason)
        return _guided(
            {
                "ok": True,
                "continuation_status": "AUTO_RESOLVED" if changed else "NO_ADMINISTRATIVE_REPAIR_NEEDED",
                "blocker": {
                    "blocker_class": "SOFT",
                    "blocker_scope": "action",
                    "operator_intervention_required": False,
                    "reconciliation_action": "administrative_state_reconciled" if changed else "none_required",
                    "continuation_possible": True,
                },
                "work": _summary(revised, state_root=args.state_root),
            },
            revised,
            state_root=args.state_root,
        ), 0


def command_dependency(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    work_id = validate_prefixed_id(args.work_id, WORK_ID_PREFIX, label="work item")
    dependency_id = validate_prefixed_id(args.dependency_work_id, WORK_ID_PREFIX, label="dependency work item")
    with work_lifecycle_locks(args.state_root, [work_id, dependency_id]):
        return _command_dependency_locked(args)


def _refresh_records(args: argparse.Namespace) -> list[str]:
    """Automatically release dependency-satisfied administrative blocks.

    This does not turn an unsuccessful terminal dependency into success; it
    only advances a mutable Work after all retained dependencies are complete.
    """

    ready: list[str] = []
    store_obj = _store(args)
    for row in store_obj.list_read_only():
        if row.get("status") != "BLOCKED":
            continue
        work_id = str(row["record_id"])
        with work_lifecycle_lock(args.state_root, work_id):
            record = store_obj.get(work_id)
            payload = dict(record.get("payload") or {})
            dep = dependency_state(args.state_root, payload.get("dependency_work_ids") or [])
            if dep["all_satisfied"] and payload.get("hold") is None:
                payload["blocked_reason"] = None
                payload["administrative_reconciliation"] = {
                    "reason": "Dependencies completed; automatic operational-continuity release.",
                    "disposition": "AUTO_RESUME",
                    "previous_work_status": "BLOCKED",
                    "acceptance_implied": False,
                    "dependencies_released": False,
                    "recorded_at_utc": utc_now(args.generated_at),
                    "recorded_by": "workManager dependency resolver",
                }
                _revise(args, record, payload=payload, status="READY", action="auto_release_dependencies", note="Automatic release after successful dependency completion.")
                ready.append(work_id)
    return sorted(ready)


def command_refresh(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    ready = _refresh_records(args)
    return with_guidance(
        {
            "ok": True,
            "mutated": bool(ready),
            "released_count": len(ready),
            "released_work_ids": ready,
            "ready_to_resume_count": len(ready),
            "ready_to_resume_work_ids": ready,
        },
        status="PORTFOLIO_REFRESHED",
        reason_codes=["work_dependency_readiness_derived"],
        next_actions=[
            {
                "action": "continue_released_work",
                "description": "Dependencies are complete and the Work was automatically released to READY.",
                "command": f"python -m tools.eiel workManager continue {ready[0]}",
            }
        ] if ready else [{"action": "none", "description": "No blocked Work is ready to resume."}],
        related_records=ready,
    ), 0


def command_graph(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    graph = dependency_graph(args.state_root, read_only=True)
    if args.work_id:
        work_id = validate_prefixed_id(args.work_id, WORK_ID_PREFIX, label="work item")
        if work_id not in graph:
            raise ActorError(f"unknown work item: {work_id}")
        wanted: set[str] = set()
        stack = [work_id]
        while stack:
            node = stack.pop()
            if node in wanted:
                continue
            wanted.add(node)
            stack.extend(graph.get(node, []))
        graph = {node: graph.get(node, []) for node in sorted(wanted)}
    edges = [{"from": node, "depends_on": dependency} for node in sorted(graph) for dependency in sorted(graph[node])]
    cycle = find_dependency_cycle(graph)
    states = {
        node: _summary(_store(args).get_read_only(node), state_root=args.state_root, read_only=True)
        for node in graph
        if _store(args).exists(node)
    }
    return with_guidance(
        {"ok": cycle is None, "graph": graph, "edges": edges, "cycle": cycle, "work": states},
        status="BLOCKED" if cycle else "DEPENDENCY_GRAPH_READY",
        reason_codes=["work_dependency_cycle"] if cycle else ["work_dependency_graph_ready"],
        problems=["Dependency cycle: " + " -> ".join(cycle)] if cycle else [],
        next_actions=[{"action": "repair_cycle", "description": "Remove or redesign a cyclic dependency."}] if cycle else [{"action": "complete_leaf_dependencies", "description": "Work from dependency leaves toward the blocked parent."}],
        related_records=list(graph),
    ), 0 if cycle is None else 2


def command_history(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    work_id = validate_prefixed_id(args.work_id, WORK_ID_PREFIX, label="work item")
    rows = _store(args).history(work_id)
    return {"ok": True, "work_id": work_id, "history": rows, "count": len(rows)}, 0


def command_next(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = _store(args).get_read_only(args.work_id)
    summary = _summary(record, state_root=args.state_root, read_only=True)
    return _guided(
        {"ok": True, "work": summary, "next_actions": summary["next_actions"]},
        record,
        state_root=args.state_root,
    ), 0


def command_handoff(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = _store(args).get_read_only(args.work_id)
    summary = _summary(record, state_root=args.state_root, read_only=True)
    payload = dict(record.get("payload") or {})
    handoff = {
        "schema_version": "eiel_work_handoff_v1",
        **operator_metadata(),
        "work_id": summary["work_id"],
        "title": summary["title"],
        "status": summary["status"],
        "derived_status": summary["derived_status"],
        "mission_id": summary["mission_id"],
        "current_stage": summary["current_stage"],
        "current_step": summary["current_step"],
        "last_completed_step": summary["last_completed_step"],
        "next_action": summary["next_action"],
        "next_command": summary["next_command"],
        "work_package_binding": summary["work_package_binding"],
        "review": summary["review"],
        "accepted_review_session_id": summary["accepted_review_session_id"],
        "dependencies": summary["dependencies"],
        "sessions": summary["sessions"],
        "references": payload.get("references"),
        "preferred_codex_conversation_id": payload.get("preferred_codex_conversation_id"),
        "preferred_codex_thread_url": payload.get("preferred_codex_thread_url"),
        "record_sha256": record.get("record_sha256"),
    }
    return _guided({"ok": True, "work": summary, "handoff": handoff}, record, state_root=args.state_root), 0


def _human(payload: Mapping[str, Any]) -> str:
    if "work" in payload and isinstance(payload["work"], Mapping):
        row = payload["work"]
        status = str(row.get("status") or "UNKNOWN").upper()
        actions = list(payload.get("next_actions") or row.get("next_actions") or [])
        next_text = row.get("next_action")
        if not next_text and status in WORK_TERMINAL_STATUSES:
            next_text = "none"
        elif not next_text and actions:
            next_text = dict(actions[0]).get("description")
        lines = [
            f"Work {row.get('work_id')}",
            f"Work status: {row.get('status')} ({row.get('derived_status')})",
            f"Title: {row.get('title')}",
            f"Next: {next_text or 'not recorded'}",
        ]
        warnings = list(payload.get("warnings") or [])
        if warnings:
            lines.append(f"Warnings: {len(warnings)}")
            lines.extend(f"Warning: {warning}" for warning in warnings[:3])
        return "\n".join(lines) + "\n"
    if "work_items" in payload:
        lines = [f"Work items: {payload.get('count', 0)}"]
        for row in payload.get("work_items") or []:
            lines.append(f"{row.get('work_id')} {row.get('priority')} {row.get('status')} {row.get('title')}")
        return "\n".join(lines) + "\n"
    return json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"


def build_parser() -> argparse.ArgumentParser:
    return build_work_parser(work_types=WORK_TYPES, priorities=PRIORITIES, work_statuses=WORK_STATUSES)


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw:
        raw = ["list", "--open-only"]
    args: argparse.Namespace | None = None
    try:
        args = build_parser().parse_args(raw)
        if args.command == "status":
            if args.work_id:
                args.command = "inspect"
                payload, code = command_inspect(args)
            else:
                args.command = "list"
                args.status = None
                args.priority = None
                args.open_only = True
                args.view = "fifo"
                payload, code = command_list(args)
        else:
            handlers = {
                "create": command_create,
                "list": command_list,
                "inspect": command_inspect,
                "update": command_update,
                "evidence-requirement": command_set_evidence_requirement,
                "hold": command_hold,
                "clear-hold": command_clear_hold,
                "add-delivery-component": command_add_delivery_component,
                "pause": command_pause,
                "block": command_block,
                "resume": command_resume,
                "reopen": command_reopen,
                "continue": command_continue,
                "complete": command_complete,
                "repair-chain": command_repair_chain,
                "reconcile": command_reconcile,
                "discontinue": command_discontinue,
                "obsolete": command_obsolete,
                "supersede": command_supersede,
                "dependency": command_dependency,
                "refresh": command_refresh,
                "graph": command_graph,
                "history": command_history,
                "next": command_next,
                "handoff": command_handoff,
                "automation-candidates": command_automation_candidates,
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
