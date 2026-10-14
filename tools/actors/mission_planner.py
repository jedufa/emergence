"""Deterministic, execution-free EIEL Mission Planner CLI for Codex."""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.actors.canonical import canonical_sha256
from tools.actors.common import (
    ActorArgumentParser,
    ActorError,
    actor_state_root,
    apply_dotted_patch,
    emit_actor_error,
    emit_payload,
    file_sha256,
    forbidden_operation_error,
    new_record,
    parse_set_values,
    read_structured,
    repository_root,
    require_explicit_schema_version,
    revise_record,
    stable_id,
    utc_now,
    validate_record_id,
    with_guidance,
)
from tools.actors.authority import (
    AUTHORIZATION_BASES,
    build_authority_context,
    evidence_preflight_markdown,
    normalize_authority_context,
    retain_preflight_receipt,
    run_librarian_preflight,
    validate_authority_context,
    validate_librarian_document_ids,
    validate_managed_document_path,
)
from tools.actors.store import ActorRecordStore
from tools.actors.release import operator_metadata

ACTOR_NAME = "missionPlanner"
STORE_NAMESPACE = "missions"
MISSION_SCHEMA_VERSION = "eiel_mission_specification_v2"
LEGACY_MISSION_SCHEMA_VERSIONS = {"eiel_mission_specification_v1"}
MISSION_RECORD_TYPE = "mission_specification"
MISSION_TYPES = (
    "IMPLEMENTATION_VALIDATION",
    "TECHNICAL_QUALIFICATION",
    "BASELINE",
    "SMOKE",
    "REHEARSAL",
    "SCIENTIFIC_PROBE",
    "CONFIRMATION",
    "ROBUSTNESS_BATTERY",
    "ABLATION",
    "SENSITIVITY",
    "PERFORMANCE",
    "FORENSICS",
    "BRANCH_REPLAY",
    "CLOSURE",
)
SCIENTIFIC_MISSION_TYPES = {
    "SCIENTIFIC_PROBE", "CONFIRMATION", "ROBUSTNESS_BATTERY", "ABLATION", "SENSITIVITY"
}
MISSION_STATUSES = (
    "DRAFT", "READY", "ACTIVE", "BLOCKED", "EVIDENCE_PENDING", "CLOSED", "SUPERSEDED", "ABANDONED"
)
ALLOWED_TRANSITIONS = {
    "DRAFT": {"DRAFT", "READY", "BLOCKED", "ABANDONED"},
    "READY": {"READY", "ACTIVE", "BLOCKED", "SUPERSEDED", "ABANDONED"},
    "ACTIVE": {"ACTIVE", "BLOCKED", "EVIDENCE_PENDING", "CLOSED", "ABANDONED"},
    "BLOCKED": {"BLOCKED", "READY", "ACTIVE", "ABANDONED", "SUPERSEDED"},
    "EVIDENCE_PENDING": {"EVIDENCE_PENDING", "ACTIVE", "BLOCKED", "CLOSED", "ABANDONED"},
    "CLOSED": {"CLOSED", "SUPERSEDED"},
    "SUPERSEDED": {"SUPERSEDED"},
    "ABANDONED": {"ABANDONED"},
}
TEXT_FIELDS = ("title", "question", "decision_to_enable", "decision_rule")
LIST_FIELDS = (
    "scope", "out_of_scope", "success_criteria", "failure_criteria", "stopping_rules",
    "required_evidence", "technical_qualification_requirements", "tasks",
)
REFERENCE_FIELDS = (
    "qualification_record_ids",
    "resource_record_ids",
    "evidence_assessment_ids",
    "campaign_spec_paths",
    "campaign_output_paths",
    "implementation_report_refs",
    "librarian_document_ids",
)
REFERENCE_ARGUMENT_FIELDS = {
    "qualification_ref": "qualification_record_ids",
    "resource_ref": "resource_record_ids",
    "evidence_ref": "evidence_assessment_ids",
    "campaign_spec_ref": "campaign_spec_paths",
    "campaign_output_ref": "campaign_output_paths",
    "implementation_report_ref": "implementation_report_refs",
    "librarian_document_id": "librarian_document_ids",
}
DEFAULT_CLOSEOUT_REQUIREMENTS = ("evidence_refs", "qualification_refs", "librarian_document_ids")
FORBIDDEN_EXECUTION_FIELDS = {
    "commands", "shell_commands", "run_command", "launcher", "subprocess", "execution_plan",
    "command", "shell", "run_script", "process_launcher",
}
FORBIDDEN_OPERATIONS = {
    "run": "Mission Planner records the plan; Codex must use existing campaign tooling to execute it.",
    "execute": "Mission Planner records the plan; Codex must use existing campaign tooling to execute it.",
    "launch": "Mission Planner has no launch authority.",
    "schedule": "Mission Planner has no scheduler or background process.",
}


def _store(args: argparse.Namespace) -> ActorRecordStore:
    return ActorRecordStore(actor_state_root(args.state_root), STORE_NAMESPACE)


def _common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-root")
    parser.add_argument("--generated-at")
    parser.add_argument("--output")
    parser.add_argument("--human", action="store_true")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _default_references() -> dict[str, list[str]]:
    return {name: [] for name in REFERENCE_FIELDS}


def _default_template(mission_type: str) -> dict[str, Any]:
    kind = str(mission_type).upper()
    if kind not in MISSION_TYPES:
        raise ActorError(f"unknown mission type: {mission_type}", code="mission_type_unknown")
    template: dict[str, Any] = {
        "schema_version": MISSION_SCHEMA_VERSION,
        "mission_type": kind,
        "title": "<concise mission title>",
        "question": "<one decision-relevant question>",
        "decision_to_enable": "<the programme decision this mission enables>",
        "scope": ["<included work>"],
        "out_of_scope": ["<explicit exclusion>"],
        "controls": [],
        "success_criteria": ["<predeclared success criterion>"],
        "failure_criteria": ["<predeclared negative criterion>"],
        "stopping_rules": ["stop after the fixed, approved scope"],
        "decision_rule": "<versioned bounded decision rule>",
        "claim_ceiling": "No claim beyond the declared mission scope.",
        "required_evidence": ["<required evidence artifact>"],
        "technical_qualification_requirements": ["current route authority or an explicit qualification plan"],
        "resource_demand": {"device": "none" if kind == "CLOSURE" else "either"},
        "qualification_request": {},
        "tasks": ["<first bounded task>"],
        "references": _default_references(),
        "source_identity": {},
        "required_closeout_refs": list(DEFAULT_CLOSEOUT_REQUIREMENTS),
        "authority_context": normalize_authority_context(None),
    }
    if kind in SCIENTIFIC_MISSION_TYPES:
        template["controls"] = ["normal condition", "matched control or ablation"]
        template["required_evidence"] = ["paired outputs", "reward-leakage audit", "technical integrity record"]
        template["required_closeout_refs"] = [
            "evidence_refs", "qualification_refs", "resource_refs", "report_refs", "campaign_output_refs", "source_identity", "librarian_document_ids"
        ]
    elif kind == "TECHNICAL_QUALIFICATION":
        template["claim_ceiling"] = "Technical route trust only; no scientific claim."
        template["required_evidence"] = ["route identity", "repeat parity", "checkpoint/replay evidence", "durable report"]
        template["required_closeout_refs"] = ["qualification_refs", "resource_refs", "report_refs", "source_identity", "librarian_document_ids"]
    elif kind == "IMPLEMENTATION_VALIDATION":
        template["claim_ceiling"] = "Implementation and regression validity only."
        template["required_closeout_refs"] = ["qualification_refs", "report_refs", "source_identity", "librarian_document_ids"]
    elif kind == "PERFORMANCE":
        template["claim_ceiling"] = "Performance under the measured route and envelope only."
        template["required_evidence"] = ["timing protocol", "repeat measurements", "resource observation"]
        template["required_closeout_refs"] = ["qualification_refs", "resource_refs", "report_refs", "source_identity", "librarian_document_ids"]
    elif kind == "CLOSURE":
        template["claim_ceiling"] = "Closure of the named topic only."
        template["required_closeout_refs"] = ["evidence_refs", "report_refs", "librarian_document_ids"]
    return template


def migrate_mission_v1_to_v2(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the one declared historical Mission schema without erasing it."""

    require_explicit_schema_version(
        raw,
        expected=MISSION_SCHEMA_VERSION,
        supported=sorted(LEGACY_MISSION_SCHEMA_VERSIONS),
        label="Mission Specification",
    )
    if raw.get("schema_version") == MISSION_SCHEMA_VERSION:
        return copy.deepcopy(dict(raw))
    source = copy.deepcopy(dict(raw))
    migrated = copy.deepcopy(source)
    migrated["schema_version"] = MISSION_SCHEMA_VERSION
    references = dict(migrated.get("references") or {})
    references.setdefault("campaign_spec_paths", [])
    migrated["references"] = references
    migrated["authority_context"] = normalize_authority_context(None)
    migrated["schema_migration"] = {
        "source_schema_version": source.get("schema_version"),
        "source_sha256": canonical_sha256(source),
        "source_payload": source,
        "migration": "mission_specification_v1_to_v2_add_authority_context",
    }
    return migrated


def load_mission_spec(raw: Any, *, mission_id: str | None = None) -> dict[str, Any]:
    """Strict loader for current and explicitly supported historical Missions."""

    if not isinstance(raw, Mapping):
        raise ActorError("Mission Specification must be an object", code="schema_record_malformed")
    source = dict(raw)
    if source.get("record_type") == MISSION_RECORD_TYPE and isinstance(source.get("payload"), Mapping):
        source = dict(source["payload"])
    if isinstance(source.get("mission"), Mapping):
        source = dict(source["mission"])
    require_explicit_schema_version(
        source,
        expected=MISSION_SCHEMA_VERSION,
        supported=sorted(LEGACY_MISSION_SCHEMA_VERSIONS),
        label="Mission Specification",
    )
    if source.get("schema_version") in LEGACY_MISSION_SCHEMA_VERSIONS:
        source = migrate_mission_v1_to_v2(source)
    return normalize_mission_spec(source, mission_id=mission_id)



def normalize_mission_spec(raw: Mapping[str, Any], *, mission_id: str | None = None) -> dict[str, Any]:
    source = dict(raw)
    if source.get("record_type") == MISSION_RECORD_TYPE and isinstance(source.get("payload"), Mapping):
        source = dict(source["payload"])
    if isinstance(source.get("mission"), Mapping):
        source = dict(source["mission"])
    if "schema_version" in source:
        require_explicit_schema_version(
            source,
            expected=MISSION_SCHEMA_VERSION,
            supported=sorted(LEGACY_MISSION_SCHEMA_VERSIONS),
            label="Mission Specification",
        )
        if source.get("schema_version") in LEGACY_MISSION_SCHEMA_VERSIONS:
            source = migrate_mission_v1_to_v2(source)
    payload = dict(source)
    payload["schema_version"] = MISSION_SCHEMA_VERSION
    payload["authority_context"] = normalize_authority_context(payload.get("authority_context"))
    payload["mission_type"] = str(payload.get("mission_type") or "IMPLEMENTATION_VALIDATION").upper()
    payload["execution_authority"] = "external_codex_only"
    payload["planner_authority"] = "planning_tracking_history_only_no_execution"
    for field in LIST_FIELDS:
        payload[field] = _as_list(payload.get(field))
    payload["controls"] = _as_list(payload.get("controls"))
    payload["parent_mission_ids"] = [str(item) for item in _as_list(payload.get("parent_mission_ids"))]
    payload["required_closeout_refs"] = [
        str(item) for item in _as_list(payload.get("required_closeout_refs") or DEFAULT_CLOSEOUT_REQUIREMENTS)
    ]
    resource = payload.get("resource_demand")
    payload["resource_demand"] = dict(resource) if isinstance(resource, Mapping) else {}
    request = payload.get("qualification_request")
    payload["qualification_request"] = dict(request) if isinstance(request, Mapping) else {}
    source_identity = payload.get("source_identity")
    payload["source_identity"] = dict(source_identity) if isinstance(source_identity, Mapping) else {}
    refs = _default_references()
    if isinstance(payload.get("references"), Mapping):
        for name in REFERENCE_FIELDS:
            refs[name] = [str(item) for item in _as_list(dict(payload["references"]).get(name))]
    payload["references"] = refs
    identity_payload = {
        key: value for key, value in payload.items()
        if key not in {"mission_id", "created_at_utc", "updated_at_utc", "closeout"}
    }
    resolved_id = validate_record_id(mission_id or str(payload.get("mission_id") or stable_id("MIS", identity_payload)))
    if not resolved_id.startswith("MIS-"):
        raise ActorError("mission_id must start with MIS-", code="mission_id_invalid")
    payload["mission_id"] = resolved_id
    return payload


def _walk_forbidden(value: Any, *, path: str = "") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            dotted = f"{path}.{key_text}" if path else key_text
            if key_text.lower() in FORBIDDEN_EXECUTION_FIELDS:
                errors.append(f"Mission Planner forbids executable field: {dotted}")
            errors.extend(_walk_forbidden(nested, path=dotted))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            errors.extend(_walk_forbidden(nested, path=f"{path}[{index}]"))
    return errors


def _field_action(mission_id: Any, field: str) -> dict[str, Any]:
    return {
        "action": "repair_mission_field",
        "description": f"Provide a valid value for {field}.",
        "command": f"python -m tools.eiel missionPlanner update {mission_id or '<mission-id>'} --set '{field}=<JSON-value>'",
    }


def validate_mission_spec(payload: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    missing_fields: list[str] = []
    if payload.get("schema_version") != MISSION_SCHEMA_VERSION:
        errors.append("unsupported mission schema_version")
    try:
        mission_id = validate_record_id(str(payload.get("mission_id") or ""))
        if not mission_id.startswith("MIS-"):
            errors.append("mission_id must start with MIS-")
    except ActorError as exc:
        errors.append(str(exc))
    mission_type = str(payload.get("mission_type") or "").upper()
    if mission_type not in MISSION_TYPES:
        errors.append(f"mission_type must be one of {', '.join(MISSION_TYPES)}")
    for field in TEXT_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} must be a non-empty string")
            missing_fields.append(field)
    claim = payload.get("claim_ceiling")
    if not isinstance(claim, str) or not claim.strip():
        errors.append("claim_ceiling must be a non-empty string")
        missing_fields.append("claim_ceiling")
    for field in LIST_FIELDS:
        value = payload.get(field)
        if not isinstance(value, list) or not value:
            errors.append(f"{field} must be a non-empty list")
            missing_fields.append(field)
    resource = payload.get("resource_demand")
    if not isinstance(resource, Mapping):
        errors.append("resource_demand must be an object")
        missing_fields.append("resource_demand")
        resource = {}
    elif str(resource.get("device") or "").lower() not in {"cpu", "cuda", "either", "none"}:
        errors.append("resource_demand.device must be cpu, cuda, either, or none")
    if payload.get("execution_authority") != "external_codex_only":
        errors.append("Mission Planner cannot own execution authority")
    errors.extend(_walk_forbidden(payload))
    if "scientific_decision" in payload or "scientific_claim_approved" in payload:
        errors.append("Mission Planner cannot approve scientific claims")
    refs = payload.get("references")
    if not isinstance(refs, Mapping):
        errors.append("references must be an object")
    else:
        for name in REFERENCE_FIELDS:
            if not isinstance(refs.get(name), list):
                errors.append(f"references.{name} must be a list")
        errors.extend(validate_librarian_document_ids(refs.get("librarian_document_ids") or []))
    if mission_type in SCIENTIFIC_MISSION_TYPES:
        if not list(payload.get("controls") or []):
            errors.append("scientific mission requires declared controls")
            missing_fields.append("controls")
        if not any("reward" in str(item).lower() for item in payload.get("required_evidence") or []):
            errors.append("scientific mission requires an explicit reward-leakage audit")
            missing_fields.append("required_evidence.reward_leakage_audit")
    elif mission_type == "TECHNICAL_QUALIFICATION":
        if not payload.get("qualification_request"):
            warnings.append("technical qualification mission has no exact qualification_request yet")
    if not list(payload.get("parent_mission_ids") or []):
        warnings.append("mission has no parent mission; valid for a root mission")
    authority_validation = validate_authority_context(
        payload.get("authority_context"),
        mission_type=mission_type,
        resource_demand=resource if isinstance(resource, Mapping) else {},
    )
    errors.extend(authority_validation["errors"])
    warnings.extend(authority_validation["warnings"])
    missing_fields.extend(authority_validation["missing_fields"])
    reason_codes = []
    for field in missing_fields:
        reason_codes.append("mission_field_missing:" + field)
    if errors and not reason_codes:
        reason_codes.append("mission_validation_failed")
    next_actions = [_field_action(payload.get("mission_id"), field) for field in sorted(set(missing_fields))]
    if any(str(field).startswith("authority_context") for field in missing_fields):
        next_actions = [{
            "action": "run_evidence_preflight",
            "description": "Bind the mandatory Knowledge Librarian preflight and the authorizing operator/work-package basis.",
            "command": (
                f"python -m tools.eiel missionPlanner preflight {payload.get('mission_id') or '<mission-id>'} "
                "--topic '<exact topic>' --authorization-basis operator_instruction "
                "--operator-instruction-ref '<operator instruction>'"
            ),
        }, *[row for row in next_actions if "authority_context" not in str(row.get("command"))]]
    return {
        "schema_version": "eiel_mission_validation_v2",
        "mission_id": payload.get("mission_id"),
        "valid": not errors,
        "ready_for_work": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "missing_fields": sorted(set(missing_fields)),
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "next_actions": next_actions,
        "claim_ceiling": payload.get("claim_ceiling"),
        "execution_authority": payload.get("execution_authority"),
        "authority": authority_validation,
    }


def _references(payload: Mapping[str, Any]) -> dict[str, list[str]]:
    refs = _default_references()
    raw = payload.get("references")
    if isinstance(raw, Mapping):
        for name in REFERENCE_FIELDS:
            refs[name] = [str(item) for item in _as_list(raw.get(name))]
    closeout = payload.get("closeout")
    if isinstance(closeout, Mapping):
        mapping = {
            "qualification_record_ids": "qualification_refs",
            "resource_record_ids": "resource_refs",
            "evidence_assessment_ids": "evidence_refs",
            "campaign_output_paths": "campaign_output_refs",
            "implementation_report_refs": "report_refs",
            "librarian_document_ids": "librarian_document_ids",
        }
        for target, source in mapping.items():
            refs[target] = sorted({*refs[target], *[str(item) for item in _as_list(closeout.get(source))]})
    return refs


def _actor_reference_state(
    payload: Mapping[str, Any],
    *,
    state_root: Path | None,
) -> dict[str, Any]:
    """Resolve the operational meaning of cross-actor references.

    Mission Planner remains read-only toward the other actors.  It only checks
    their sealed records so a weak Codex cannot mistake a blocked qualification
    or a WAIT recommendation for permission to begin work.
    """

    refs = _references(payload)
    state = {
        "qualification": {"required": bool(payload.get("technical_qualification_requirements")), "present": bool(refs["qualification_record_ids"]), "accepted": False, "problems": []},
        "resource": {"required": str(dict(payload.get("resource_demand") or {}).get("device") or "").lower() != "none", "present": bool(refs["resource_record_ids"]), "accepted": False, "problems": []},
        "evidence": {"present": bool(refs["evidence_assessment_ids"]), "accepted": bool(refs["evidence_assessment_ids"]), "problems": []},
    }
    if state_root is None:
        state["qualification"]["accepted"] = state["qualification"]["present"]
        state["resource"]["accepted"] = state["resource"]["present"] or not state["resource"]["required"]
        return state

    stores = {
        "qualification": ActorRecordStore(state_root, "qualifications"),
        "resource": ActorRecordStore(state_root, "resources"),
        "evidence": ActorRecordStore(state_root, "evidence-decisions"),
    }
    accepted_statuses = {
        "qualification": {"QUALIFIED", "QUALIFIED_WITH_LIMITS", "QUALIFIED_WITH_EXCLUSIONS"},
        "resource": {"ADMIT", "ADMIT_WITH_LIMITS"},
        "evidence": {"DECIDED_A", "DECIDED_B", "DECIDED_C", "DECIDED_D"},
    }
    field_map = {
        "qualification": "qualification_record_ids",
        "resource": "resource_record_ids",
        "evidence": "evidence_assessment_ids",
    }
    for kind, field in field_map.items():
        statuses: list[str] = []
        for record_id in refs[field]:
            try:
                actor_record = stores[kind].get(record_id)
            except ActorError:
                state[kind]["problems"].append(f"unknown {kind} actor record: {record_id}")
                continue
            statuses.append(str(actor_record.get("status") or "UNKNOWN"))
        if statuses:
            state[kind]["statuses"] = statuses
            state[kind]["accepted"] = any(status in accepted_statuses[kind] for status in statuses)
            if not state[kind]["accepted"]:
                state[kind]["problems"].append(
                    f"no {kind} reference has an accepted status; observed {sorted(set(statuses))}"
                )
    if not state["resource"]["required"]:
        state["resource"]["accepted"] = True
    return state


def _workflow(record: Mapping[str, Any], *, state_root: Path | None = None) -> dict[str, Any]:
    payload = dict(record.get("payload") or {})
    validation = validate_mission_spec(payload)
    authority = dict(validation.get("authority") or {})
    refs = _references(payload)
    actor_refs = _actor_reference_state(payload, state_root=state_root)
    status = str(record.get("status") or "DRAFT")
    completed = ["mission_created"]
    if isinstance(payload.get("authority_context"), Mapping) and dict(payload["authority_context"]).get("preflight"):
        completed.append("evidence_preflight_bound")
    if authority.get("ready_for_work"):
        completed.append("authority_bound")
    if validation["valid"]:
        completed.append("mission_validated")
    if actor_refs["qualification"]["accepted"]:
        completed.append("qualification_checked")
    if actor_refs["resource"]["accepted"]:
        completed.append("resources_assessed")
    if status in {"ACTIVE", "EVIDENCE_PENDING", "CLOSED"}:
        completed.append("execution_started")
    if refs["campaign_output_paths"]:
        completed.append("campaign_output_linked")
    if refs["evidence_assessment_ids"]:
        completed.append("evidence_assessed")
    if status == "CLOSED":
        completed.append("closed")
    missing: list[str] = []
    next_actions: list[dict[str, Any]] = []
    mission_id = str(record.get("record_id"))
    if not authority.get("ready_for_work"):
        missing.extend(authority.get("missing_fields") or authority.get("errors") or ["authority binding"])
        next_actions.append({
            "action": "run_evidence_preflight",
            "description": "Run and bind the mandatory Knowledge Librarian preflight plus the authorizing operator/work-package basis.",
            "command": f"python -m tools.eiel missionPlanner preflight {mission_id} --topic '<exact topic>' --authorization-basis operator_instruction --operator-instruction-ref '<operator instruction>'",
        })
    elif not validation["valid"]:
        missing.extend(validation["missing_fields"] or validation["errors"])
        next_actions.extend(validation["next_actions"])
    elif status == "DRAFT":
        next_actions.append({
            "action": "mark_ready",
            "description": "The mission is valid. Mark it READY before work begins.",
            "command": f"python -m tools.eiel missionPlanner update {mission_id} --status READY",
        })
    elif status == "READY" and not actor_refs["qualification"]["accepted"]:
        missing.append("qualification reference")
        missing.extend(actor_refs["qualification"]["problems"])
        next_actions.append({
            "action": "check_qualification",
            "description": "Resolve the mission's exact technical route authority.",
            "command": f"python -m tools.eiel qualificationManager check --mission {mission_id} --store",
        })
    elif status == "READY" and not actor_refs["resource"]["accepted"]:
        missing.append("resource assessment reference")
        missing.extend(actor_refs["resource"]["problems"])
        next_actions.append({
            "action": "assess_resources",
            "description": "Assess the mission against a fresh or supplied ERM observation.",
            "command": f"python -m tools.eiel resourceManager assess --mission {mission_id}",
        })
    elif status == "READY":
        next_actions.append({
            "action": "begin_external_work",
            "description": "Qualification and resources are recorded. Codex may begin the bounded work externally.",
            "command": f"python -m tools.eiel missionPlanner update {mission_id} --status ACTIVE",
        })
    elif status == "ACTIVE" and not refs["campaign_output_paths"]:
        missing.append("campaign output reference")
        campaign_spec = refs["campaign_spec_paths"][0] if refs["campaign_spec_paths"] else "<campaign-spec>"
        next_actions.append({
            "action": "execute_externally",
            "description": (
                "Codex must validate and run the approved campaign through the existing campaign owner, "
                "then attach the resulting output path to this mission. Mission Planner does not execute it."
            ),
            "command": f"python -m tools.scripts.eiel_campaign run {shlex.quote(campaign_spec)}",
            "follow_up_command": (
                f"python -m tools.eiel missionPlanner update {mission_id} "
                "--campaign-output-ref '<campaign-output-path>'"
            ),
            "authority_owner": "tools.scripts.eiel_campaign",
        })
    elif status == "ACTIVE" and not refs["evidence_assessment_ids"]:
        missing.append("evidence assessment reference")
        campaign = refs["campaign_output_paths"][0] if refs["campaign_output_paths"] else "<campaign-output>"
        next_actions.append({
            "action": "inspect_evidence",
            "description": "Inspect and assess the completed campaign outputs.",
            "command": f"python -m tools.eiel evidenceManager inspect {campaign}",
        })
    elif status in {"ACTIVE", "EVIDENCE_PENDING"} and refs["evidence_assessment_ids"]:
        next_actions.append({
            "action": "closeout",
            "description": "All primary actor references exist. Prepare the bounded mission closeout.",
            "command": f"python -m tools.eiel missionPlanner closeout {mission_id} --closeout <closeout.json>",
        })
    elif status == "EVIDENCE_PENDING":
        missing.append("evidence assessment reference")
        next_actions.append({"action": "complete_evidence", "description": "Complete the Evidence Manager assessment, then attach its ID."})
    elif status == "BLOCKED":
        next_actions.append({"action": "resolve_blocker", "description": "Inspect the mission and repair the recorded blocker before resuming."})
    elif status == "CLOSED":
        next_actions.append({"action": "none", "description": "Mission is closed. No further action is authorized by this record."})
    elif status in {"SUPERSEDED", "ABANDONED"}:
        next_actions.append({"action": "none", "description": f"Mission is {status.lower()} and must not be resumed."})
    related = []
    for name, values in refs.items():
        for value in values:
            related.append({"record_id": value, "kind": name})
    return {
        "schema_version": "eiel_mission_workflow_state_v2",
        "mission_id": mission_id,
        "status": status,
        "completed_stages": completed,
        "missing_items": missing,
        "next_actions": next_actions,
        "references": refs,
        "authority_state": authority,
        "evidence_preflight": dict(payload.get("authority_context") or {}).get("preflight"),
        "actor_reference_state": actor_refs,
        "related_records": related,
    }


def _mission_summary(record: Mapping[str, Any], *, state_root: Path | None = None) -> dict[str, Any]:
    payload = dict(record.get("payload") or {})
    validation = validate_mission_spec(payload)
    workflow = _workflow(record, state_root=state_root)
    return {
        "mission_id": record.get("record_id"),
        "title": payload.get("title"),
        "mission_type": payload.get("mission_type"),
        "status": record.get("status"),
        "revision": record.get("revision"),
        "valid": validation["valid"],
        "error_count": len(validation["errors"]),
        "warning_count": len(validation["warnings"]),
        "authority_ready": bool(dict(validation.get("authority") or {}).get("ready_for_work")),
        "authorization_basis": dict(payload.get("authority_context") or {}).get("authorization_basis"),
        "catalog_build_id": dict(dict(payload.get("authority_context") or {}).get("preflight") or {}).get("catalog_build_id"),
        "updated_at_utc": record.get("updated_at_utc"),
        "completed_stages": workflow["completed_stages"],
        "missing_items": workflow["missing_items"],
        "next_actions": workflow["next_actions"],
        "related_records": workflow["related_records"],
    }


def _human_summary(payload: Mapping[str, Any]) -> str:
    if "mission" in payload and isinstance(payload["mission"], Mapping):
        mission = payload["mission"]
        next_rows = mission.get("next_actions") or payload.get("next_actions") or []
        next_text = next_rows[0].get("description") if next_rows and isinstance(next_rows[0], Mapping) else "none"
        return (
            f"Mission {mission.get('mission_id')}\nStatus: {mission.get('status')}\n"
            f"Title: {mission.get('title')}\nValid: {mission.get('valid')}\nNext: {next_text}\n"
        )
    if "missions" in payload:
        return f"Missions: {payload.get('count', 0)}\n" + "\n".join(
            f"{row.get('mission_id')} {row.get('status')} {row.get('title')}" for row in payload.get("missions") or []
        )
    return str(dict(payload))


def _guided_mission_payload(
    base: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    state_root: Path | None = None,
    ok: bool = True,
) -> dict[str, Any]:
    workflow = _workflow(record, state_root=state_root)
    validation = validate_mission_spec(dict(record.get("payload") or {}))
    status = str(record.get("status") or "DRAFT") if validation["valid"] else "BLOCKED"
    reason_codes = validation["reason_codes"] if not validation["valid"] else [f"mission_{status.lower()}"]
    return with_guidance(
        base,
        status=status,
        reason_codes=reason_codes,
        problems=validation["errors"],
        warnings=validation["warnings"],
        next_actions=workflow["next_actions"],
        related_records=workflow["related_records"],
    )


def command_template(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    template = _default_template(args.mission_type)
    if args.template_output:
        from tools.actors.common import atomic_write_json
        atomic_write_json(args.template_output, template)
    return with_guidance(
        {"ok": True, "mission_type": args.mission_type, "template": template, "template_path": args.template_output},
        status="TEMPLATE_READY",
        reason_codes=["mission_template_ready"],
        next_actions=[{
            "action": "create_mission",
            "description": "Fill the placeholders, then create the mission record.",
            "command": f"python -m tools.eiel missionPlanner create --input {args.template_output or '<mission.json>'} --ready",
        }],
    ), 0


def command_create(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    state_root = actor_state_root(args.state_root)
    raw: dict[str, Any] = read_structured(args.input, name="mission specification") if args.input else {}
    direct = {
        "title": args.title,
        "question": args.question,
        "decision_to_enable": args.decision_to_enable,
        "mission_type": args.mission_type,
        "decision_rule": args.decision_rule,
        "claim_ceiling": args.claim_ceiling,
    }
    raw.update({key: value for key, value in direct.items() if value is not None})
    for field in (*LIST_FIELDS, "controls"):
        value = getattr(args, field, None)
        if value:
            raw[field] = value
    if args.device:
        resource = dict(raw.get("resource_demand") or {})
        resource["device"] = args.device
        raw["resource_demand"] = resource
    spec = normalize_mission_spec(raw, mission_id=args.mission_id)
    validation = validate_mission_spec(spec)
    record = new_record(
        actor=ACTOR_NAME,
        record_type=MISSION_RECORD_TYPE,
        record_id=str(spec["mission_id"]),
        status="READY" if validation["valid"] and args.ready else "DRAFT",
        payload=spec,
        generated_at=utc_now(args.generated_at),
        source_refs=({"kind": "input", "uri": str(Path(args.input).resolve())},) if args.input else (),
        note=args.note,
    )
    path = _store(args).put(record)
    result = {
        "ok": True,
        "created": True,
        "record_path": str(path),
        "mission": {**_mission_summary(record, state_root=state_root), "validation": validation},
    }
    return _guided_mission_payload(result, record, state_root=state_root), 0


def command_preflight(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """Delegate to the Knowledge Librarian and bind its exact receipt."""

    state_root = actor_state_root(args.state_root)
    store = _store(args)
    record = store.get(args.mission_id)
    current_status = str(record.get("status") or "DRAFT")
    if current_status in {"ACTIVE", "EVIDENCE_PENDING", "CLOSED", "SUPERSEDED", "ABANDONED"}:
        raise ActorError(
            "mission authority binding is frozen after work begins",
            code="mission_authority_frozen",
            problems=[f"Current mission status is {current_status}."],
            next_actions=[{
                "action": "create_successor_mission",
                "description": "Create a successor mission if the authority basis materially changed.",
                "command": "python -m tools.eiel missionPlanner create --input <successor-mission.json>",
            }],
            related_records=[args.mission_id],
        )
    payload = dict(record.get("payload") or {})
    topic = args.topic or str(payload.get("question") or payload.get("title") or "").strip()
    if not topic:
        raise ActorError("preflight topic is required", code="preflight_topic_missing")
    repo = Path(args.repo_root).expanduser().resolve() if args.repo_root else repository_root()
    active_goal = args.active_goal
    if not active_goal and (repo / "ACTIVE_GOAL.md").is_file():
        active_goal = str(repo / "ACTIVE_GOAL.md")

    if args.authorization_basis == "mechanical_preflight_exemption":
        receipt = None
        retained = None
    elif args.preflight_receipt:
        receipt = read_structured(args.preflight_receipt, name="Knowledge Librarian preflight receipt")
        retained = retain_preflight_receipt(receipt, state_root=state_root)
    else:
        receipt = run_librarian_preflight(
            repo_root=repo,
            source_root=args.source_root,
            topic=topic,
            collection=args.collection,
            limit=args.limit,
            authority_domain=args.authority_domain,
        )
        retained = retain_preflight_receipt(receipt, state_root=state_root)

    context = build_authority_context(
        authorization_basis=args.authorization_basis,
        repo_root=repo,
        state_root=state_root,
        preflight_receipt=receipt,
        preflight_receipt_path=retained,
        operator_instruction_ref=args.operator_instruction_ref,
        work_package_path=args.work_package,
        active_goal_path=active_goal,
        preflight_fail_closed_acknowledged=args.acknowledge_fail_closed,
        preflight_exemption_reason=args.preflight_exemption_reason,
        repository_fallback_reason=args.repository_fallback_reason,
    )
    updated_payload = normalize_mission_spec(
        {**payload, "authority_context": context},
        mission_id=str(record.get("record_id")),
    )
    validation = validate_mission_spec(updated_payload)
    next_status = "READY" if args.ready and validation["valid"] else "DRAFT"
    _validate_transition(current_status, next_status)
    source_refs = list(record.get("source_refs") or [])
    if retained is not None and context.get("preflight"):
        source_refs.append({
            "kind": "knowledge_librarian_preflight",
            "uri": str(retained),
            "sha256": file_sha256(retained),
            "receipt_sha256": context["preflight"]["receipt_sha256"],
            "catalog_build_id": context["preflight"]["catalog_build_id"],
        })
    revised = revise_record(
        record,
        payload=updated_payload,
        status=next_status,
        action="bind_evidence_preflight",
        generated_at=utc_now(args.generated_at),
        note=args.note,
        source_refs=source_refs,
    )
    path = store.put(revised, allow_update=True)
    result = {
        "ok": validation["valid"],
        "record_path": str(path),
        "preflight_receipt_path": str(retained) if retained else None,
        "authority_context": context,
        "evidence_preflight_markdown": evidence_preflight_markdown(context.get("preflight")),
        "mission": {**_mission_summary(revised, state_root=state_root), "validation": validation},
    }
    return _guided_mission_payload(result, revised, state_root=state_root), 0 if validation["valid"] else 1


def command_authority(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    state_root = actor_state_root(args.state_root)
    record = _store(args).get(args.mission_id)
    payload = dict(record.get("payload") or {})
    validation = validate_authority_context(
        payload.get("authority_context"),
        mission_type=str(payload.get("mission_type") or ""),
        resource_demand=dict(payload.get("resource_demand") or {}),
    )
    context = normalize_authority_context(payload.get("authority_context"))
    return with_guidance(
        {
            "ok": validation["valid"],
            "mission_id": args.mission_id,
            "authority_context": context,
            "authority_validation": validation,
            "evidence_preflight_markdown": evidence_preflight_markdown(context.get("preflight")),
        },
        status="AUTHORIZED" if validation["valid"] else "BLOCKED",
        reason_codes=["mission_authority_bound"] if validation["valid"] else ["mission_authority_incomplete"],
        problems=validation["errors"],
        warnings=validation["warnings"],
        next_actions=_workflow(record, state_root=state_root)["next_actions"],
        related_records=[args.mission_id],
    ), 0 if validation["valid"] else 1


def command_validate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if Path(args.target).is_file():
        spec = normalize_mission_spec(read_structured(args.target, name="mission specification"))
        validation = validate_mission_spec(spec)
        payload = with_guidance(
            {"ok": validation["valid"], "validation": validation},
            status="VALID" if validation["valid"] else "BLOCKED",
            reason_codes=validation["reason_codes"] or ["mission_valid"],
            problems=validation["errors"],
            warnings=validation["warnings"],
            next_actions=validation["next_actions"],
        )
        return payload, 0 if validation["valid"] else 1
    state_root = actor_state_root(args.state_root)
    record = _store(args).get(args.target)
    validation = validate_mission_spec(dict(record.get("payload") or {}))
    return _guided_mission_payload(
        {"ok": validation["valid"], "validation": validation, "mission": _mission_summary(record, state_root=state_root)},
        record,
        state_root=state_root,
    ), 0 if validation["valid"] else 1


def command_inspect(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    state_root = actor_state_root(args.state_root)
    record = _store(args).get(args.mission_id)
    validation = validate_mission_spec(dict(record.get("payload") or {}))
    return _guided_mission_payload(
        {"ok": not bool(validation["errors"]), "record": record, "validation": validation, "mission": _mission_summary(record, state_root=state_root)},
        record,
        state_root=state_root,
        ok=not bool(validation["errors"]),
    ), 0 if not validation["errors"] else 1


def command_status(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    state_root = actor_state_root(args.state_root)
    store = _store(args)
    if args.mission_id:
        record = store.get(args.mission_id)
        mission = _mission_summary(record, state_root=state_root)
        return _guided_mission_payload(
            {"ok": mission["valid"], "mission": mission}, record, state_root=state_root
        ), 0 if mission["valid"] else 1
    missions = []
    for row in store.list():
        missions.append(
            row if row.get("status") == "INVALID" else _mission_summary(store.get(str(row["record_id"])), state_root=state_root)
        )
    counts = Counter(str(row.get("status") or "UNKNOWN") for row in missions)
    blocked = [row for row in missions if row.get("status") in {"BLOCKED", "INVALID"}]
    next_rows = [row for row in missions if row.get("status") not in {"CLOSED", "SUPERSEDED", "ABANDONED"}]
    return with_guidance(
        {
            "ok": not blocked,
            "count": len(missions),
            "status_counts": dict(sorted(counts.items())),
            "missions": missions,
        },
        status="READY" if not blocked else "ATTENTION_REQUIRED",
        reason_codes=["mission_backlog_loaded"],
        problems=[f"{len(blocked)} mission records require attention"] if blocked else [],
        next_actions=(next_rows[0].get("next_actions") or []) if next_rows else [],
    ), 0


def _validate_transition(current: str, target: str) -> None:
    allowed = ALLOWED_TRANSITIONS.get(current, {current})
    if target not in allowed:
        raise ActorError(
            f"invalid mission state transition: {current} -> {target}",
            code="mission_state_transition_invalid",
            problems=[f"Allowed next states from {current}: {', '.join(sorted(allowed))}"],
            next_actions=[{
                "action": "choose_valid_status",
                "description": f"Use one of: {', '.join(sorted(allowed))}",
            }],
        )


def _transition_prerequisite_errors(
    payload: Mapping[str, Any],
    *,
    next_status: str,
    state_root: Path,
) -> list[str]:
    errors: list[str] = []
    if next_status == "CLOSED":
        return ["use 'python -m tools.eiel missionPlanner closeout' so required evidence and authority references are checked"]
    if next_status not in {"ACTIVE", "EVIDENCE_PENDING"}:
        return errors
    actor_state = _actor_reference_state(payload, state_root=state_root)
    if actor_state["qualification"]["required"] and not actor_state["qualification"]["accepted"]:
        errors.append("ACTIVE work requires an accepted technical qualification check record")
        errors.extend(actor_state["qualification"]["problems"])
    if actor_state["resource"]["required"] and not actor_state["resource"]["accepted"]:
        errors.append("ACTIVE work requires an ADMIT or ADMIT_WITH_LIMITS resource assessment record")
        errors.extend(actor_state["resource"]["problems"])
    if next_status == "EVIDENCE_PENDING" and not _references(payload)["campaign_output_paths"]:
        errors.append("EVIDENCE_PENDING requires at least one campaign output path")
    return list(dict.fromkeys(errors))


def command_update(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    state_root = actor_state_root(args.state_root)
    store = _store(args)
    record = store.get(args.mission_id)
    payload = dict(record.get("payload") or {})
    patch: dict[str, Any] = {}
    if args.patch:
        patch.update(read_structured(args.patch, name="mission patch"))
    patch.update(parse_set_values(args.set_values or []))
    protected_roots = {
        "authority_context",
        "execution_authority",
        "planner_authority",
        "schema_version",
        "mission_id",
    }
    attempted_protected = sorted(
        {
            str(key).split(".", 1)[0]
            for key in patch
            if str(key).split(".", 1)[0] in protected_roots
        }
    )
    if attempted_protected:
        raise ActorError(
            "mission authority and identity fields cannot be changed through generic update",
            code="mission_protected_field_update_forbidden",
            problems=[f"Protected field: {field}" for field in attempted_protected],
            next_actions=[{
                "action": "use_preflight_command",
                "description": "Bind authority only through the validated preflight command.",
                "command": f"python -m tools.eiel missionPlanner preflight {args.mission_id} --help",
            }],
            related_records=[args.mission_id],
        )
    refs = dict(payload.get("references") or _default_references())
    ref_inputs = {
        "qualification_record_ids": args.qualification_ref,
        "resource_record_ids": args.resource_ref,
        "evidence_assessment_ids": args.evidence_ref,
        "campaign_spec_paths": args.campaign_spec_ref,
        "campaign_output_paths": args.campaign_output_ref,
        "implementation_report_refs": args.report_ref,
        "librarian_document_ids": args.librarian_document_id,
    }
    for field, values in ref_inputs.items():
        if values:
            refs[field] = _append_unique(_as_list(refs.get(field)), values)
    if any(ref_inputs.values()):
        patch["references"] = refs
    source_identity = dict(payload.get("source_identity") or {})
    if args.source_commit:
        source_identity["source_commit"] = args.source_commit
    if args.snapshot_sha256:
        source_identity["snapshot_sha256"] = args.snapshot_sha256
    if args.source_commit or args.snapshot_sha256:
        patch["source_identity"] = source_identity
    updated_payload = normalize_mission_spec(apply_dotted_patch(payload, patch), mission_id=str(record.get("record_id")))
    if str(updated_payload.get("mission_id")) != str(record.get("record_id")):
        raise ActorError("mission_id is immutable", code="mission_id_immutable")
    validation = validate_mission_spec(updated_payload)
    current_status = str(record.get("status") or "DRAFT").upper()
    next_status = str(args.status or current_status).upper()
    if next_status not in MISSION_STATUSES:
        raise ActorError(f"status must be one of {', '.join(MISSION_STATUSES)}", code="mission_status_invalid")
    _validate_transition(current_status, next_status)
    if next_status in {"READY", "ACTIVE", "EVIDENCE_PENDING", "CLOSED"} and not validation["valid"]:
        raise ActorError(
            "cannot advance an invalid mission",
            code="mission_invalid_for_state",
            problems=validation["errors"],
            next_actions=validation["next_actions"],
            related_records=[args.mission_id],
        )
    prerequisite_errors = _transition_prerequisite_errors(
        updated_payload,
        next_status=next_status,
        state_root=state_root,
    )
    if prerequisite_errors:
        raise ActorError(
            "mission state prerequisites are not satisfied",
            code="mission_state_prerequisites_missing",
            problems=prerequisite_errors,
            next_actions=_workflow(record, state_root=state_root)["next_actions"],
            related_records=[args.mission_id],
        )
    revised = revise_record(
        record,
        payload=updated_payload,
        status=next_status,
        action="update",
        generated_at=utc_now(args.generated_at),
        note=args.note,
    )
    path = store.put(revised, allow_update=True)
    return _guided_mission_payload(
        {"ok": validation["valid"], "record_path": str(path), "mission": {**_mission_summary(revised, state_root=state_root), "validation": validation}},
        revised,
        state_root=state_root,
    ), 0 if validation["valid"] else 1


def command_history(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    state_root = actor_state_root(args.state_root)
    rows = _store(args).history(args.mission_id)
    record = _store(args).get(args.mission_id)
    return _guided_mission_payload(
        {"ok": True, "mission_id": args.mission_id, "revision_count": len(rows), "history": rows, "mission": _mission_summary(record, state_root=state_root)},
        record,
        state_root=state_root,
    ), 0


def command_next(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    state_root = actor_state_root(args.state_root)
    record = _store(args).get(args.mission_id)
    workflow = _workflow(record, state_root=state_root)
    return _guided_mission_payload(
        {"ok": True, "mission": _mission_summary(record, state_root=state_root), "workflow": workflow},
        record,
        state_root=state_root,
    ), 0


def _append_unique(target: list[str], values: Iterable[Any]) -> list[str]:
    return sorted({*[str(item) for item in target], *[str(item) for item in values if str(item)]})


def _closeout_missing(payload: Mapping[str, Any], closeout: Mapping[str, Any]) -> list[str]:
    required = [str(item) for item in payload.get("required_closeout_refs") or DEFAULT_CLOSEOUT_REQUIREMENTS]
    missing: list[str] = []
    for field in required:
        if field == "source_identity":
            source = closeout.get("source_identity") or payload.get("source_identity")
            if not isinstance(source, Mapping) or not (source.get("source_commit") or source.get("snapshot_sha256")):
                missing.append("source_identity")
        elif not _as_list(closeout.get(field)):
            missing.append(field)
    return missing


def _strict_reference_errors(args: argparse.Namespace, closeout: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    root = actor_state_root(args.state_root)
    stores = {
        "evidence_refs": ActorRecordStore(root, "evidence-decisions"),
        "qualification_refs": ActorRecordStore(root, "qualifications"),
        "resource_refs": ActorRecordStore(root, "resources"),
    }
    for field, store in stores.items():
        for record_id in _as_list(closeout.get(field)):
            try:
                record = store.get(str(record_id))
            except ActorError:
                errors.append(f"{field} references unknown or invalid actor record: {record_id}")
                continue
            status = str(record.get("status") or "UNKNOWN")
            accepted = {
                "evidence_refs": {"DECIDED_A", "DECIDED_B", "DECIDED_C", "DECIDED_D"},
                "qualification_refs": {"QUALIFIED", "QUALIFIED_WITH_LIMITS", "QUALIFIED_WITH_EXCLUSIONS"},
                "resource_refs": {"ADMIT", "ADMIT_WITH_LIMITS"},
            }[field]
            if status not in accepted:
                errors.append(f"{field} record {record_id} has non-accepted status {status}")
    errors.extend(validate_librarian_document_ids(
        _as_list(closeout.get("librarian_document_ids")),
        repo_root=repository_root(),
        require_registered=True,
    ))
    for path in _as_list(closeout.get("report_refs")):
        report_path = Path(str(path)).expanduser()
        if not report_path.exists():
            if args.strict_references:
                errors.append(f"closeout path does not exist: {path}")
            continue
        errors.extend(validate_managed_document_path(report_path, repo_root=repository_root()))
        if report_path.is_file() and report_path.suffix.lower() in {".md", ".markdown"}:
            text = report_path.read_text(encoding="utf-8", errors="replace")
            if not text.lstrip().startswith("## Evidence Preflight"):
                errors.append(
                    f"substantial Markdown report must begin with the mandatory Evidence Preflight section: {report_path}"
                )
    if args.strict_references:
        for path in _as_list(closeout.get("campaign_output_refs")):
            if not Path(str(path)).expanduser().exists():
                errors.append(f"closeout path does not exist: {path}")
    return errors


def command_closeout(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    state_root = actor_state_root(args.state_root)
    store = _store(args)
    record = store.get(args.mission_id)
    current_status = str(record.get("status") or "DRAFT")
    _validate_transition(current_status, "CLOSED")
    payload = dict(record.get("payload") or {})
    validation = validate_mission_spec(payload)
    if not validation["valid"]:
        raise ActorError("cannot close an invalid mission", code="mission_closeout_invalid", problems=validation["errors"], next_actions=validation["next_actions"])
    closeout: dict[str, Any] = read_structured(args.closeout, name="mission closeout") if args.closeout else {}
    for key, value in {
        "outcome": args.outcome,
        "decision": args.decision,
        "no_claims_statement": args.no_claims_statement,
    }.items():
        if value is not None:
            closeout[key] = value
    mappings = {
        "evidence_refs": args.evidence_ref,
        "qualification_refs": args.qualification_ref,
        "resource_refs": args.resource_ref,
        "report_refs": [*args.report_ref, *args.implementation_report_ref],
        "campaign_output_refs": args.campaign_output_ref,
        "librarian_document_ids": args.librarian_document_id,
    }
    for field, values in mappings.items():
        closeout[field] = _append_unique(_as_list(closeout.get(field)), values or [])
    source_identity = dict(closeout.get("source_identity") or payload.get("source_identity") or {})
    if args.source_commit:
        source_identity["source_commit"] = args.source_commit
    if args.snapshot_sha256:
        source_identity["snapshot_sha256"] = args.snapshot_sha256
    closeout["source_identity"] = source_identity
    authority_context = normalize_authority_context(payload.get("authority_context"))
    closeout["authority_context"] = authority_context
    closeout["evidence_preflight"] = authority_context.get("preflight")
    closeout["evidence_preflight_markdown"] = evidence_preflight_markdown(
        authority_context.get("preflight")
    )
    closeout_errors = [
        f"closeout.{field} must be a non-empty string"
        for field in ("outcome", "decision", "no_claims_statement")
        if not isinstance(closeout.get(field), str) or not str(closeout.get(field)).strip()
    ]
    missing = _closeout_missing(payload, closeout)
    closeout_errors.extend(f"closeout missing required reference: {field}" for field in missing)
    closeout_errors.extend(_strict_reference_errors(args, closeout))
    if closeout_errors:
        raise ActorError(
            "mission closeout is incomplete",
            code="mission_closeout_incomplete",
            problems=closeout_errors,
            next_actions=[{
                "action": "supply_closeout_refs",
                "description": "Attach every reference declared by required_closeout_refs, then rerun closeout.",
                "command": f"python -m tools.eiel missionPlanner inspect {args.mission_id}",
            }],
            related_records=[args.mission_id],
        )
    payload["closeout"] = closeout
    refs = dict(payload.get("references") or _default_references())
    cross_map = {
        "qualification_record_ids": "qualification_refs",
        "resource_record_ids": "resource_refs",
        "evidence_assessment_ids": "evidence_refs",
        "campaign_output_paths": "campaign_output_refs",
        "implementation_report_refs": "report_refs",
        "librarian_document_ids": "librarian_document_ids",
    }
    for target, source in cross_map.items():
        refs[target] = _append_unique(_as_list(refs.get(target)), _as_list(closeout.get(source)))
    payload["references"] = refs
    payload["source_identity"] = source_identity
    revised = revise_record(
        record,
        payload=payload,
        status="CLOSED",
        action="closeout",
        generated_at=utc_now(args.generated_at),
        note=args.note,
    )
    path = store.put(revised, allow_update=True)
    return _guided_mission_payload(
        {"ok": True, "record_path": str(path), "mission": _mission_summary(revised, state_root=state_root), "closeout": closeout, "warnings": []},
        revised,
        state_root=state_root,
    ), 0


def command_handoff(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    state_root = actor_state_root(args.state_root)
    record = _store(args).get(args.mission_id)
    payload = dict(record.get("payload") or {})
    workflow = _workflow(record, state_root=state_root)
    handoff = {
        "schema_version": "eiel_mission_codex_handoff_v2",
        **operator_metadata(),
        "mission_id": args.mission_id,
        "title": payload.get("title"),
        "mission_type": payload.get("mission_type"),
        "status": record.get("status"),
        "question": payload.get("question"),
        "decision_to_enable": payload.get("decision_to_enable"),
        "claim_ceiling": payload.get("claim_ceiling"),
        "completed_stages": workflow["completed_stages"],
        "missing_items": workflow["missing_items"],
        "next_actions": workflow["next_actions"],
        "references": workflow["references"],
        "closeout": payload.get("closeout"),
        "authority_context": payload.get("authority_context"),
        "evidence_preflight_markdown": evidence_preflight_markdown(
            dict(payload.get("authority_context") or {}).get("preflight")
        ),
        "execution_handoff": {
            "owner": "existing_eiel_campaign_tooling",
            "planner_executes": False,
            "preflight_command": "python -m tools.scripts.eiel_campaign preflight <campaign-spec>",
            "run_command": "python -m tools.scripts.eiel_campaign run <campaign-spec>",
        },
        "record_sha256": record.get("record_sha256"),
    }
    return _guided_mission_payload(
        {"ok": True, "mission": _mission_summary(record, state_root=state_root), "handoff": handoff},
        record,
        state_root=state_root,
    ), 0


def build_parser() -> argparse.ArgumentParser:
    parser = ActorArgumentParser(prog="python -m tools.eiel missionPlanner", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True, parser_class=ActorArgumentParser)

    template = sub.add_parser("template", help="Produce a mission-class template for Codex.")
    _common_options(template)
    template.add_argument("mission_type", choices=MISSION_TYPES)
    template.add_argument("--template-output")

    create = sub.add_parser("create", help="Create a file-backed mission specification.")
    _common_options(create)
    create.add_argument("--input")
    create.add_argument("--mission-id")
    create.add_argument("--title")
    create.add_argument("--question")
    create.add_argument("--decision-to-enable")
    create.add_argument("--mission-type", choices=MISSION_TYPES)
    create.add_argument("--decision-rule")
    create.add_argument("--claim-ceiling")
    create.add_argument("--device", choices=("cpu", "cuda", "either", "none"))
    for field in (*LIST_FIELDS, "controls"):
        create.add_argument("--" + field.replace("_", "-"), dest=field, action="append")
    create.add_argument("--ready", action="store_true")
    create.add_argument("--note")

    preflight = sub.add_parser("preflight", help="Run or bind the mandatory Knowledge Librarian Evidence Preflight.")
    _common_options(preflight)
    preflight.add_argument("mission_id")
    preflight.add_argument("--topic")
    preflight.add_argument("--preflight-receipt")
    preflight.add_argument("--repo-root")
    preflight.add_argument("--source-root", default="tools/evidence/librarian/catalog_source/00-catalog")
    preflight.add_argument("--collection", choices=("evidence", "documents", "all"), default="all")
    preflight.add_argument("--authority-domain")
    preflight.add_argument("--limit", type=int, default=20)
    preflight.add_argument("--authorization-basis", choices=AUTHORIZATION_BASES, required=True)
    preflight.add_argument("--operator-instruction-ref")
    preflight.add_argument("--work-package")
    preflight.add_argument("--active-goal")
    preflight.add_argument("--acknowledge-fail-closed", action="store_true")
    preflight.add_argument("--preflight-exemption-reason")
    preflight.add_argument("--repository-fallback-reason")
    preflight.add_argument("--ready", action="store_true")
    preflight.add_argument("--note")

    authority = sub.add_parser("authority", help="Show the retained mission authority and Evidence Preflight binding.")
    _common_options(authority)
    authority.add_argument("mission_id")

    validate = sub.add_parser("validate", help="Validate a mission record or input file.")
    _common_options(validate)
    validate.add_argument("target")

    inspect = sub.add_parser("inspect", help="Show one mission, workflow state, and next action.")
    _common_options(inspect)
    inspect.add_argument("mission_id")

    status = sub.add_parser("status", help="Show one mission or the mission backlog.")
    _common_options(status)
    status.add_argument("mission_id", nargs="?")

    update = sub.add_parser("update", help="Patch mission fields and record an immutable revision.")
    _common_options(update)
    update.add_argument("mission_id")
    update.add_argument("--patch")
    update.add_argument("--set", dest="set_values", action="append", default=[])
    update.add_argument("--status", choices=MISSION_STATUSES)
    update.add_argument("--qualification-ref", action="append", default=[])
    update.add_argument("--resource-ref", action="append", default=[])
    update.add_argument("--evidence-ref", action="append", default=[])
    update.add_argument("--campaign-spec-ref", action="append", default=[])
    update.add_argument("--campaign-output-ref", action="append", default=[])
    update.add_argument("--report-ref", action="append", default=[])
    update.add_argument("--librarian-document-id", action="append", default=[])
    update.add_argument("--source-commit")
    update.add_argument("--snapshot-sha256")
    update.add_argument("--note")

    history = sub.add_parser("history", help="Show immutable mission revision snapshots.")
    _common_options(history)
    history.add_argument("mission_id")

    next_cmd = sub.add_parser("next", help="Show the exact next action for an interrupted Codex session.")
    _common_options(next_cmd)
    next_cmd.add_argument("mission_id")

    closeout = sub.add_parser("closeout", help="Close a mission with required cross-actor references.")
    _common_options(closeout)
    closeout.add_argument("mission_id")
    closeout.add_argument("--closeout")
    closeout.add_argument("--outcome")
    closeout.add_argument("--decision")
    closeout.add_argument("--no-claims-statement")
    closeout.add_argument("--evidence-ref", action="append", default=[])
    closeout.add_argument("--qualification-ref", action="append", default=[])
    closeout.add_argument("--resource-ref", action="append", default=[])
    closeout.add_argument("--report-ref", action="append", default=[])
    closeout.add_argument("--implementation-report-ref", action="append", default=[])
    closeout.add_argument("--campaign-output-ref", action="append", default=[])
    closeout.add_argument("--librarian-document-id", action="append", default=[])
    closeout.add_argument("--source-commit")
    closeout.add_argument("--snapshot-sha256")
    closeout.add_argument("--strict-references", action="store_true")
    closeout.add_argument("--note")

    handoff = sub.add_parser("handoff", help="Produce a compact Codex recovery or final handoff record.")
    _common_options(handoff)
    handoff.add_argument("mission_id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] in FORBIDDEN_OPERATIONS:
        exc = forbidden_operation_error(f"eiel {ACTOR_NAME}", raw_args[0], FORBIDDEN_OPERATIONS[raw_args[0]])
        emit_actor_error(
            exc,
            actor=ACTOR_NAME,
            command=raw_args[0],
            raw_args=raw_args,
        )
        return 2
    args: argparse.Namespace | None = None
    try:
        args = build_parser().parse_args(raw_args)
        handlers = {
            "template": command_template,
            "create": command_create,
            "preflight": command_preflight,
            "authority": command_authority,
            "validate": command_validate,
            "inspect": command_inspect,
            "status": command_status,
            "update": command_update,
            "history": command_history,
            "next": command_next,
            "closeout": command_closeout,
            "handoff": command_handoff,
        }
        payload, code = handlers[args.command](args)
        emit_payload(payload, output=args.output, human_text=_human_summary(payload), json_output=not args.human)
        return code
    except ActorError as exc:
        command = raw_args[0] if raw_args else "unknown"
        emit_actor_error(
            exc,
            actor=ACTOR_NAME,
            command=command,
            raw_args=raw_args,
            parsed_args=args,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
