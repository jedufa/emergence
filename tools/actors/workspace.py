"""Consolidated read-only Codex front door for EIEL.

Workspace combines a bootstrap receipt with the Work Manager, Session Manager,
and optional Mission Planner context.  It also audits cross-manager record
integrity and can export one deterministic resume handoff.  It never changes
actor records, never guesses between multiple open tracks, and never treats an
administrative workflow obligation as execution authority.
"""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.actors.bootstrap import (
    READY_STATUSES,
    assess_bootstrap_record,
    bootstrap_refresh_command,
    git_repository_identity,
)
from tools.actors.context_receipt import build_context_receipt, validate_context_receipt
from tools.actors.context_projection import project_compact_workspace_guidance
from tools.actors.canonical import canonical_sha256
from tools.actors.common import (
    ActorArgumentParser,
    ActorError,
    priority_ordered_findings,
    EIEL_LAUNCHER,
    actor_state_root,
    emit_actor_error,
    emit_payload,
    file_sha256,
    read_json,
    repository_root,
    require_explicit_schema_version,
    timestamp_age_hours,
    utc_now,
    with_guidance,
)
from tools.actors.portfolio import (
    BOOTSTRAP_ID_PREFIX,
    SESSION_RESUMABLE_STATUSES,
    SESSION_TERMINAL_STATUSES,
    WORK_ID_PREFIX,
    WORK_OPEN_STATUSES,
    WORK_TERMINAL_STATUSES,
    bootstrap_store,
    dependency_graph,
    dependency_state,
    evidence_store,
    find_dependency_cycle,
    latest_bootstrap_record,
    mission_store,
    normalize_string_list,
    open_sessions,
    open_work,
    parallel_session_ownership_problems,
    qualification_store,
    resource_store,
    session_store,
    sessions_for_work,
    validate_prefixed_id,
    work_store,
)
from tools.actors.release import ACTOR_MANAGERS_VERSION, OPERATOR_CONTRACT_VERSION
from tools.actors.cli_compatibility import command_entry
from tools.actors.resource_accounting import validate_session_resource_accounting_binding
from tools.actors.reference_integrity import (
    REFERENCE_CORRUPT,
    REFERENCE_EXTERNAL_UNVERIFIABLE,
    REFERENCE_OPTIONAL_ABSENT,
    REFERENCE_RECOVERABLE_STALE,
    REFERENCE_SUPPORTED_HISTORICAL,
    audit_scheduler_reference_graph,
    validate_record_source_references,
)
from tools.actors.work_manager import summarize_work_record
from tools.actors.session_manager import NEXT_COMMAND_PRECONDITIONS_SCHEMA_VERSION, summarize_session_record
from tools.scheduler.contracts import (
    assignment_store,
    dispatch_store,
    lease_store,
    result_store,
)
from tools.actors.work_package import (
    AUTHORITY_ROLES,
    effective_execution_budget,
    effective_output_policy,
    effective_read_contract,
    verify_work_package_binding,
)

ACTOR_NAME = "workspace"
WORKSPACE_SCHEMA_VERSION = "eiel_codex_workspace_v2"
LEGACY_WORKSPACE_SCHEMA_VERSIONS = ("eiel_codex_workspace_v1",)
WORKSPACE_HANDOFF_SCHEMA_VERSION = "eiel_codex_workspace_handoff_v1"
WORKSPACE_AUDIT_SCHEMA_VERSION = "eiel_codex_workspace_audit_v1"

def load_workspace_artifact(raw: Any) -> dict[str, Any]:
    """Load a full Workspace projection, including the retained read-only v1."""

    if not isinstance(raw, Mapping):
        raise ActorError("Workspace artifact must be an object", code="schema_record_malformed")
    value = dict(raw)
    workspace = dict(value.get("workspace") or value)
    require_explicit_schema_version(
        workspace,
        expected=WORKSPACE_SCHEMA_VERSION,
        supported=LEGACY_WORKSPACE_SCHEMA_VERSIONS,
        label="Workspace artifact",
    )
    for field in ("bootstrap", "portfolio"):
        if field not in workspace:
            raise ActorError(
                "Workspace artifact is malformed",
                code="schema_record_malformed",
                problems=[f"missing field: {field}"],
            )
    return workspace


def load_workspace_audit_artifact(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ActorError("Workspace audit artifact must be an object", code="schema_record_malformed")
    value = dict(raw)
    audit = dict(value.get("audit") or value)
    require_explicit_schema_version(
        audit, expected=WORKSPACE_AUDIT_SCHEMA_VERSION, label="Workspace audit artifact"
    )
    return audit

BOOTSTRAP_REFRESH_RECOMMENDATION_HOURS = 24.0
DELIVERY_SESSION_OUTCOMES = {
    "ARCHITECTURE_COMPLETE",
    "ENGINEERING_COMPLETE",
    "IMPLEMENTATION_COMPLETE",  # legacy provenance spelling
    "DELIVERY_RECORDED",
    "EXTERNAL_DELIVERY_ADOPTED",
}

_QUALIFIED_DECISIONS = {
    "QUALIFIED",
    "QUALIFIED_WITH_EXCLUSIONS",
    "QUALIFIED_WITH_LIMITS",
}
_BLOCKED_DECISIONS = {
    "BLOCKED",
    "FULL_QUALIFICATION_REQUIRED",
    "NOT_QUALIFIED",
    "SUPERSEDED",
    "REJECTED",
    "EXPIRED",
    "INCONSISTENT_AUTHORITY_RECORD",
    "REVALIDATION_REQUIRED",
}

_TASK_MODE_CONTEXT = {
    "review": {
        "task_class": "review",
        "required_authority_domain": "review",
        "required_session_role": "REVIEW_AUTHORITY",
    },
    "diagnosis": {
        "task_class": "diagnosis",
        "required_authority_domain": "engineering",
        "required_session_role": "ENGINEERING_AUTHORITY",
    },
    "implementation": {
        "task_class": "implementation",
        "required_authority_domain": "implementation",
        "required_session_role": "IMPLEMENTATION_AUTHORITY",
    },
    "campaign": {
        "task_class": "campaign",
        "required_authority_domain": "campaign",
        "required_session_role": "ENGINEERING_AUTHORITY",
    },
    "qualification": {
        "task_class": "qualification",
        "required_authority_domain": "qualification",
        "required_session_role": "ENGINEERING_AUTHORITY",
    },
    "documentation": {
        "task_class": "documentation",
        "required_authority_domain": "repository",
        "required_session_role": "IMPLEMENTATION_AUTHORITY",
    },
    "evidence": {
        "task_class": "evidence",
        "required_authority_domain": "evidence",
        "required_session_role": "ENGINEERING_AUTHORITY",
    },
    "mechanical": {
        "task_class": "repository_maintenance",
        "required_authority_domain": "repository",
        "required_session_role": "IMPLEMENTATION_AUTHORITY",
    },
}

REFERENCE_NAMESPACE_MAP = {
    "qualification_record_ids": "qualifications",
    "resource_record_ids": "resources",
    "evidence_assessment_ids": "evidence-decisions",
}


def _common(
    parser: argparse.ArgumentParser,
    *,
    include_bootstrap: bool = True,
    include_projection: bool = False,
) -> None:
    parser.add_argument("--state-root")
    parser.add_argument("--repo-root")
    parser.add_argument("--generated-at")
    if include_bootstrap:
        parser.add_argument("--bootstrap-record")
        parser.add_argument("--authority-domain")
        parser.add_argument("--max-bootstrap-age-hours", type=float)
        parser.add_argument("--allow-stale-bootstrap", action="store_true")
    parser.add_argument("--output", help="Write the complete deterministic payload to this JSON file.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--human", action="store_true", help="Display the concise human summary.")
    if include_projection:
        modes.add_argument(
            "--compact",
            action="store_true",
            help="Display the structured compact opening while --output, when used, retains the full payload.",
        )
        modes.add_argument("--full", action="store_true", help="Display the complete Workspace payload.")


def _state_flag(args: argparse.Namespace) -> str:
    return f"--state-root {json.dumps(str(actor_state_root(args.state_root)))}"


def _repo_flag(repository: Mapping[str, Any]) -> str:
    return f"--repo-root {json.dumps(str(repository.get('root') or Path.cwd()))}"


def _bootstrap_refresh_command(
    record: Mapping[str, Any],
    repository: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    requested_authority_domain: str | None = None,
) -> str:
    return bootstrap_refresh_command(
        record,
        repo_root=str(repository.get("root") or Path.cwd()),
        state_root=state_root,
        requested_authority_domain=requested_authority_domain,
        acknowledge_dirty=bool(repository.get("dirty")),
    )


def _bootstrap(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], float, dict[str, Any]]:
    # An explicit receipt may be opened from an installed CLI or orchestration
    # checkout, so its bound repository is authoritative unless --repo-root is
    # provided.  Without an explicit receipt, however, Workspace selects only a
    # receipt for the current worktree.  It never grabs the globally newest
    # receipt from another branch/worktree or a prior repository state merely
    # because that file is newer.
    store = bootstrap_store(args.state_root)
    if args.bootstrap_record:
        bootstrap_id = validate_prefixed_id(args.bootstrap_record, BOOTSTRAP_ID_PREFIX, label="bootstrap")
        record = store.get_read_only(bootstrap_id)
        retained_repository = dict(dict(record.get("payload") or {}).get("repository") or {})
        target_root = args.repo_root or retained_repository.get("root") or Path.cwd()
        repository = git_repository_identity(target_root)
    else:
        repository = git_repository_identity(args.repo_root or Path.cwd())
        matching: list[dict[str, Any]] = []
        same_worktree_receipts: list[dict[str, Any]] = []
        for row in store.list_read_only():
            if row.get("status") == "INVALID":
                continue
            try:
                candidate = store.get_read_only(str(row.get("record_id")))
            except ActorError:
                continue
            candidate_root = dict(dict(candidate.get("payload") or {}).get("repository") or {}).get("root")
            if not candidate_root:
                continue
            try:
                same_root = Path(str(candidate_root)).expanduser().resolve() == Path(str(repository["root"])).resolve()
            except OSError:
                same_root = False
            if same_root:
                same_worktree_receipts.append(candidate)
                candidate_repository = dict(dict(candidate.get("payload") or {}).get("repository") or {})
                if all(
                    str(candidate_repository.get(field) or "") == str(repository.get(field) or "")
                    for field in (
                        "head",
                        "branch",
                        "working_state_sha256",
                        "repository_state_sha256",
                    )
                ):
                    matching.append(candidate)
        matching.sort(
            key=lambda row: (str(row.get("updated_at_utc") or ""), str(row.get("record_id") or "")),
            reverse=True,
        )
        record = matching[0] if matching else None
        if record is None:
            same_worktree_receipts.sort(
                key=lambda row: (str(row.get("updated_at_utc") or ""), str(row.get("record_id") or "")),
                reverse=True,
            )
            latest_same_worktree = same_worktree_receipts[0] if same_worktree_receipts else None
            if latest_same_worktree is not None:
                command = (
                    f"{EIEL_LAUNCHER} bootstrap refresh {latest_same_worktree['record_id']} "
                    + _repo_flag(repository)
                    + " "
                    + _state_flag(args)
                )
                if args.authority_domain:
                    command += " --authority-domain " + json.dumps(str(args.authority_domain))
                missing_message = "no retained bootstrap receipt matches the current repository state"
                description = (
                    "Refresh the newest same-worktree Bootstrap using its retained non-secret "
                    "classification inputs. This repairs Workspace assistance and does not grant execution authority."
                )
            else:
                command = (
                    f"{EIEL_LAUNCHER} bootstrap inspect --repo-root " + json.dumps(repository["root"])
                    + " --task-mode implementation --topic '<task>' --operator-instruction-ref '<instruction>' --edit-authorized"
                    + " --write-scope '<scope>' --expected-validation '<test>'"
                )
                if args.authority_domain:
                    command += " --authority-domain " + json.dumps(str(args.authority_domain))
                command += " " + _state_flag(args)
                missing_message = "no retained bootstrap receipt exists for the current repository"
                description = "Run the fail-closed repository bootstrap before opening a workspace."
            raise ActorError(
                missing_message,
                code="workspace_bootstrap_missing",
                next_actions=[
                    {
                        "action": "run_bootstrap",
                        "description": description,
                        "command": command,
                    }
                ],
            )
    assessment = assess_bootstrap_record(
        record,
        expected_repository=repository,
        expected_authority_domain=args.authority_domain,
        revalidate_environment=True,
        max_age_hours=None if args.allow_stale_bootstrap else args.max_bootstrap_age_hours,
        now=args.generated_at,
    )
    blocking_errors = [
        *assessment["integrity_errors"],
        *assessment["compatibility_errors"],
        *assessment["readiness_errors"],
    ]
    if blocking_errors:
        raise ActorError(
            "workspace bootstrap receipt cannot safely support recovery for this repository",
            code="workspace_bootstrap_refresh_required",
            problems=blocking_errors,
            warnings=assessment["staleness_errors"],
            next_actions=[
                {
                    "action": "refresh_bootstrap",
                    "description": "Refresh the recovery receipt; this repairs Workspace assistance and does not grant execution authority.",
                    "command": _bootstrap_refresh_command(
                        record,
                        repository,
                        state_root=args.state_root,
                        requested_authority_domain=args.authority_domain,
                    ),
                }
            ],
            related_records=[str(record.get("record_id"))],
        )
    generated = str(dict(record.get("payload") or {}).get("generated_at_utc") or record.get("updated_at_utc"))
    return record, repository, timestamp_age_hours(generated, now=args.generated_at), assessment


def _mission_summary(state_root: str | Path | None, mission_id: str | None) -> dict[str, Any] | None:
    if not mission_id:
        return None
    try:
        record = mission_store(state_root).get_read_only(mission_id)
    except ActorError:
        return {"mission_id": mission_id, "status": "REFERENCE_ONLY_NOT_IN_LOCAL_STATE"}
    payload = dict(record.get("payload") or {})
    return {
        "mission_id": record.get("record_id"),
        "status": record.get("status"),
        "title": payload.get("title"),
        "question": payload.get("question"),
        "decision_to_enable": payload.get("decision_to_enable"),
        "updated_at_utc": record.get("updated_at_utc"),
        "record_sha256": record.get("record_sha256"),
    }


def _optional_bridge_summary(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Report optional bridge presence in the repository bound by Bootstrap."""

    root = Path(repo_root).expanduser().resolve() / "bridge" if repo_root else repository_root() / "bridge"
    available = (root / "chatgpt-bridge-manager").is_file() and (root / "package.json").is_file()
    return {
        "required": False,
        "status": "available_optional" if available else "not_installed_optional",
        "manager_command": f"{EIEL_LAUNCHER} bridgeManager status",
        "automatic_execution": False,
        "authority_effect": "none",
    }


def _selected_context(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    works = open_work(args.state_root, read_only=True)
    sessions = open_sessions(args.state_root, read_only=True)
    selected_work = None
    selected_session = None
    if getattr(args, "work_id", None):
        work_id = validate_prefixed_id(args.work_id, WORK_ID_PREFIX, label="work item")
        selected_work = work_store(args.state_root).get_read_only(work_id)
        sessions = [row for row in sessions if dict(row.get("payload") or {}).get("work_id") == work_id]
    if getattr(args, "session_id", None):
        selected_session = session_store(args.state_root).get_read_only(args.session_id)
        session_work_id = str(dict(selected_session.get("payload") or {}).get("work_id") or "")
        if selected_work and selected_work.get("record_id") != session_work_id:
            raise ActorError(
                "selected session does not belong to selected work item",
                code="workspace_selection_mismatch",
                related_records=[str(selected_work["record_id"]), str(selected_session["record_id"])],
            )
        selected_work = selected_work or work_store(args.state_root).get_read_only(session_work_id)
        sessions = [selected_session]
    return works, sessions, selected_work, selected_session


def _qualification_records(state_root: str | Path | None, references: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    refs = normalize_string_list(dict(references or {}).get("qualification_record_ids"))
    if not refs:
        return []
    store = qualification_store(state_root)
    rows: list[dict[str, Any]] = []
    for record_id in refs:
        try:
            record = store.get_read_only(record_id)
        except ActorError:
            rows.append({"record_id": record_id, "status": "MISSING"})
            continue
        payload = dict(record.get("payload") or {})
        status = str(record.get("status") or "UNKNOWN")
        result = dict(payload.get("result") or {})
        imported = dict(payload.get("decision") or {}) if isinstance(payload.get("decision"), Mapping) else {}
        decision = str(result.get("decision") or imported.get("decision_status") or status or "").upper()
        rows.append({
            "record_id": record_id,
            "record_type": str(record.get("record_type") or ""),
            "status": status,
            "decision": decision,
            "authority": (result.get("record_id") if isinstance(result.get("record_id"), str) else None),
            "reasons": list(result.get("reasons") or []),
            "limitations": list(result.get("limitations") or imported.get("limitations") or []),
            "required_evidence": list(result.get("next_required_evidence") or imported.get("errors") or []),
            "next_required_action": result.get("next_required_action") if isinstance(result.get("next_required_action"), str) else None,
            "source_commit": imported.get("source_commit"),
            "plan_sha256": imported.get("plan_sha256"),
            "accepted_for_envelope_expansion": bool(imported.get("accepted_for_envelope_expansion")) if imported else decision in _QUALIFIED_DECISIONS,
            "updated_at_utc": str(record.get("updated_at_utc") or ""),
            "record_sha256": record.get("record_sha256"),
        })
    return sorted(rows, key=lambda row: (row.get("updated_at_utc") or row.get("record_id") or ""), reverse=True)


def _resolve_implementation_status(work_record: Mapping[str, Any] | None) -> tuple[str, str]:
    if not work_record:
        return "UNKNOWN", "No selected work is currently bound to this Workspace context."
    work_status = str(work_record.get("status") or "UNKNOWN")
    payload = dict(work_record.get("payload") or {})
    if work_status == "COMPLETED":
        basis = str(payload.get("completion_basis") or "")
        if basis == "ACCEPTED_REPOSITORY_DELIVERY":
            return "ACCEPTED_REPOSITORY_DELIVERY", "Work completed from exact reviewed and integrated repository delivery."
        if basis == "RECORD_ONLY_COMPLETION":
            return "RECORD_ONLY_COMPLETION", "Work closed as a non-repository record; no implementation acceptance is implied."
        return "COMPLETED_UNCLASSIFIED", "Legacy Work completion has no explicit completion basis and is not current acceptance."
    if work_status in WORK_TERMINAL_STATUSES:
        return "TERMINATED", f"Work record is terminal with status {work_status}."
    if work_status in WORK_OPEN_STATUSES:
        return "PRESENT", f"Work record is active in implementation state {work_status}."
    if work_status == "UNKNOWN":
        return "UNKNOWN", "Work status is not available in this record."
    return work_status, f"Work status is {work_status}."


def _qualify_qualification(
    records: list[dict[str, Any]],
    *,
    required: bool,
    repository_head: str | None = None,
) -> dict[str, Any]:
    if not records:
        return {
            "status": "UNKNOWN" if required else "NOT_REQUIRED",
            "admitted": not required,
            "reason": (
                "Qualification is required for this work context, but no qualification record is attached."
                if required
                else "This work context declares no qualification requirement or qualification binding."
            ),
            "current_authority": None,
            "required_evidence": [],
            "limitations": [],
            "records": [],
        }
    current_head = str(repository_head or "").lower()
    for row in records:
        decision = str(row.get("decision") or "").upper()
        row_status = str(row.get("status") or "").upper()
        source_commit = str(row.get("source_commit") or "").lower()
        if decision in _QUALIFIED_DECISIONS:
            if source_commit and current_head and source_commit != current_head:
                return {
                    "status": "REVALIDATION_REQUIRED",
                    "admitted": False,
                    "reason": "Qualification evidence is bound to a different repository commit.",
                    "current_authority": row.get("authority"),
                    "required_evidence": [f"Revalidate qualification at repository commit {current_head}."],
                    "limitations": list(row.get("limitations") or []),
                    "records": records,
                }
            limitations = list(row.get("limitations") or [])
            exact_status = "QUALIFIED_WITH_LIMITS" if limitations and decision == "QUALIFIED" else decision
            return {
                "status": exact_status,
                "admitted": bool(row.get("accepted_for_envelope_expansion", True)),
                "reason": (str((row.get("reasons") or [None])[0]) if row.get("reasons") else "Qualification record is accepted within its exact declared envelope."),
                "current_authority": row.get("authority"),
                "required_evidence": list(row.get("required_evidence") or []),
                "limitations": limitations,
                "records": records,
            }
        if decision in _BLOCKED_DECISIONS or row_status in _BLOCKED_DECISIONS or decision == "QUALIFICATION_DECISION_A_IMPORTED_UNVERIFIED":
            return {
                "status": decision or row_status or "BLOCKED",
                "admitted": False,
                "reason": (str((row.get("reasons") or [None])[0]) if row.get("reasons") else f"Qualification record status is {decision or row_status}."),
                "current_authority": row.get("authority"),
                "required_evidence": list(row.get("required_evidence") or []),
                "limitations": list(row.get("limitations") or []),
                "records": records,
            }
    return {
        "status": str(records[0].get("decision") or records[0].get("status") or "UNKNOWN"),
        "admitted": False,
        "reason": records[0].get("next_required_action") or "No explicit qualification decision status is available.",
        "current_authority": records[0].get("authority"),
        "required_evidence": list(records[0].get("required_evidence") or []),
        "limitations": list(records[0].get("limitations") or []),
        "records": records,
    }


def _implementation_and_qualification_state(
    repository: Mapping[str, Any],
    work_record: Mapping[str, Any] | None,
    *,
    state_root: str | Path | None,
    qualification_required: bool = False,
) -> dict[str, Any]:
    if not work_record:
        return {
            "implementation": {"status": "UNKNOWN", "reason": "No selected work is currently bound to this Workspace context.", "commit": "", "work_id": None, "work_status": "UNKNOWN"},
            "qualification": {
                "status": "UNKNOWN",
                "reason": "No qualification records are currently attached to this work item.",
                "current_authority": None,
                "required_evidence": [],
                "records": [],
            },
        }
    work_payload = dict(work_record.get("payload") or {})
    impl_status, impl_reason = _resolve_implementation_status(work_record)
    references = dict(work_payload.get("references") or {})
    record_ids = normalize_string_list(references.get("qualification_record_ids"))
    required = (
        bool(qualification_required)
        or str(work_payload.get("work_type") or "").upper() == "QUALIFICATION"
        or bool(record_ids)
    )
    qualifications = _qualify_qualification(
        _qualification_records(state_root, references),
        required=required,
        repository_head=str(repository.get("head") or "") or None,
    )
    return {
        "implementation": {
            "status": impl_status,
            "reason": impl_reason,
            "commit": str(repository.get("head") or ""),
            "work_id": str(work_record.get("record_id") or work_payload.get("work_id") or ""),
            "work_status": str(work_record.get("status") or work_payload.get("status") or "UNKNOWN"),
        },
        "qualification": qualifications,
    }


def _task_context(
    bootstrap: Mapping[str, Any],
    implementation_and_qualification: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile one portable task-class and authority contract.

    Authority domain is advisory routing information.  Session role is the
    mechanically valid role accepted by Session Manager.  Keeping the two
    fields separate prevents Workspace from emitting an authority label that
    the next command cannot actually use.
    """

    task = dict(dict(bootstrap.get("payload") or {}).get("task") or {})
    mode = str(task.get("mode") or "").strip().lower()
    if mode not in _TASK_MODE_CONTEXT:
        raise ActorError(
            f"unsupported Bootstrap task mode in Workspace context: {mode or '<missing>'}",
            code="workspace_task_mode_invalid",
        )
    context = dict(_TASK_MODE_CONTEXT[mode])
    explicit_domain = str(task.get("authority_domain") or "").strip()
    if explicit_domain:
        context["required_authority_domain"] = explicit_domain
    qualification = dict(implementation_and_qualification.get("qualification") or {})
    qualification_status = str(qualification.get("status") or "UNKNOWN").upper()
    qualification_required = mode == "qualification" or qualification_status == "BLOCKED"
    context.update(
        {
            "bootstrap_task_mode": mode,
            "qualification_required": qualification_required,
            "qualification_status": qualification_status,
            "qualification_reason": str(qualification.get("reason") or ""),
        }
    )
    if context["required_session_role"] not in AUTHORITY_ROLES:
        raise ActorError(
            "Workspace task context produced an invalid Session role",
            code="workspace_task_role_invalid",
        )
    return context


def _session_refresh_problems(
    session: Mapping[str, Any],
    *,
    bootstrap: Mapping[str, Any],
    repository: Mapping[str, Any],
) -> list[str]:
    payload = dict(session.get("payload") or {})
    problems: list[str] = []
    if str(payload.get("bootstrap_record_id") or "") != str(bootstrap.get("record_id") or ""):
        problems.append("selected session is bound to an older bootstrap receipt")
    session_repo = dict(payload.get("repository") or {})
    if session_repo.get("repo_root") and Path(str(session_repo["repo_root"])).resolve() != Path(str(repository["root"])).resolve():
        problems.append("selected session repository root differs from the current repository")
    if session_repo.get("head") and str(session_repo["head"]) != str(repository["head"]):
        problems.append("selected session repository HEAD differs from the current HEAD")
    if session_repo.get("branch") and str(session_repo["branch"]) != str(repository.get("branch") or ""):
        problems.append("selected session repository branch differs from the current branch")
    if session_repo.get("repository_state_sha256") and str(session_repo["repository_state_sha256"]) != str(repository.get("repository_state_sha256") or ""):
        problems.append("selected session working-tree/index state differs from its last retained checkpoint")
    return problems


def _session_matches_repository_lane(
    session: Mapping[str, Any], repository: Mapping[str, Any]
) -> bool:
    """Return whether an unselected Session belongs to this worktree lane.

    HEAD and working-state drift are deliberately excluded: those are ordinary
    recovery facts for a Session already bound to the same root and branch.
    A different root or branch is another lane and must never be adopted merely
    because it is the only globally resumable Session.
    """

    session_repository = dict(dict(session.get("payload") or {}).get("repository") or {})
    session_root = session_repository.get("repo_root")
    session_branch = session_repository.get("branch")
    if not session_root or not session_branch:
        return False
    try:
        same_root = Path(str(session_root)).resolve() == Path(str(repository["root"])).resolve()
    except OSError:
        return False
    return same_root and str(session_branch) == str(repository.get("branch") or "")


def _next_command_precondition_problems(
    session: Mapping[str, Any],
    *,
    work: Mapping[str, Any] | None,
    bootstrap: Mapping[str, Any],
    repository: Mapping[str, Any],
) -> list[str]:
    payload = dict(session.get("payload") or {})
    preconditions = dict(payload.get("next_command_preconditions") or {})
    if not payload.get("next_command"):
        return ["selected session has no retained work command"]
    if preconditions.get("schema_version") != NEXT_COMMAND_PRECONDITIONS_SCHEMA_VERSION:
        return ["selected session command has no current authorization preconditions"]
    expected = {
        "work_record_sha256": work.get("record_sha256") if work else None,
        "bootstrap_record_id": bootstrap.get("record_id"),
        "package_sha256": dict(payload.get("work_package_binding") or {}).get("sha256"),
        "repository_state_sha256": repository.get("repository_state_sha256"),
        "current_codex_id": payload.get("current_codex_id"),
    }
    problems: list[str] = []
    labels = {
        "work_record_sha256": "Work revision",
        "bootstrap_record_id": "Bootstrap binding",
        "package_sha256": "Work Package binding",
        "repository_state_sha256": "repository state",
        "current_codex_id": "Codex ownership",
    }
    for field, current in expected.items():
        if preconditions.get(field) != current:
            problems.append(f"selected session command {labels[field]} precondition is stale")
    return problems


def _canonical_work_gate(
    summary: Mapping[str, Any] | None,
    *,
    work: Mapping[str, Any] | None,
    session: Mapping[str, Any] | None,
    bootstrap: Mapping[str, Any],
    repository: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Return a hard Work-authority decision before local projections.

    Workspace joins repository, Bootstrap, Context Receipt, and Session facts,
    but it does not own Work lifecycle or review lineage.  This gate prevents a
    locally plausible Session from replacing a newer Work decision.
    """

    if not summary or not work:
        return None
    work_id = str(work.get("record_id") or summary.get("work_id") or "")
    validation = dict(summary.get("validation") or {})
    validation_errors = [str(value) for value in (validation.get("errors") or [])]
    validation_warnings = [str(value) for value in (validation.get("warnings") or [])]
    if validation_errors or not bool(summary.get("valid", True)):
        package_problem = any("package" in value.lower() for value in validation_errors)
        return {
            "status": "WORK_PACKAGE_REFRESH_REQUIRED" if package_problem else "WORK_INTEGRITY_REPAIR_REQUIRED",
            "problems": validation_errors or ["selected Work failed authoritative validation"],
            "warnings": validation_warnings,
            "next_actions": [
                {
                    "action": "repair_work_integrity",
                    "description": "Repair the authoritative Work or Work Package before starting, resuming, reviewing, or accepting anything.",
                    "command": f"{EIEL_LAUNCHER} workManager inspect {work_id} {_state_flag(args)}",
                }
            ],
        }

    stored_status = str(work.get("status") or summary.get("status") or "")
    session_status = str((session or {}).get("status") or "")
    if stored_status in WORK_TERMINAL_STATUSES:
        if session and session_status not in SESSION_TERMINAL_STATUSES:
            return {
                "status": "TERMINAL_PARENT_RECONCILIATION_REQUIRED",
                "problems": ["selected non-terminal Session has a terminal parent Work"],
                "warnings": validation_warnings,
                "next_actions": [
                    {
                        "action": "reconcile_terminal_parent",
                        "description": "Do not execute or resume this Session. Reconcile it against the terminal Work explicitly.",
                        "command": f"{EIEL_LAUNCHER} workManager inspect {work_id} {_state_flag(args)}",
                    }
                ],
            }
        return {
            "status": "WORK_TERMINAL",
            "problems": [],
            "warnings": validation_warnings,
            "next_actions": [
                {
                    "action": "none",
                    "description": f"Work is {stored_status.lower()} and has no executable next action.",
                    "command": None,
                }
            ],
        }

    derived = str(summary.get("derived_status") or stored_status)
    if derived == "PROMOTION_REQUIRED":
        return {
            "status": "PROMOTION_REQUIRED",
            "problems": [],
            "warnings": validation_warnings,
            "next_actions": list(summary.get("next_actions") or [])[:1],
        }
    if derived == "REVIEW_REQUIRED":
        delivery_id = str(dict(summary.get("review") or {}).get("latest_delivery_session_id") or "<SES-DELIVERY-ID>")
        return {
            "status": "WORK_REVIEW_REQUIRED",
            "problems": [],
            "warnings": validation_warnings,
            "next_actions": [
                {
                    "action": "start_review",
                    "description": "Review the authoritative latest delivery, not an older locally accepted Session; use reduced assurance if independence is unavailable.",
                    "command": (
                        f"{EIEL_LAUNCHER} sessionManager start {work_id} --role REVIEW_AUTHORITY "
                        f"--review-target-session {delivery_id} --bootstrap-record {bootstrap['record_id']} "
                        f"--codex-id '<codex-id>' {_repo_flag(repository)} {_state_flag(args)}"
                    ),
                }
            ],
        }
    if derived == "READY_TO_ACCEPT":
        accepted = list(dict(summary.get("review") or {}).get("accepted_review_sessions") or [])
        review_id = str(dict(accepted[0]).get("session_id") if accepted else "<SES-REVIEW-ID>")
        return {
            "status": "WORK_READY_TO_ACCEPT",
            "problems": [],
            "warnings": validation_warnings,
            "next_actions": [
                {
                    "action": "accept_work",
                    "description": "Complete the Work through the current authoritative accepted review.",
                    "command": f"{EIEL_LAUNCHER} workManager complete {work_id} --review-session {review_id} {_state_flag(args)}",
                }
            ],
        }
    if derived == "CHANGES_REQUIRED":
        return {
            "status": "WORK_CHANGES_REQUIRED",
            "problems": [],
            "warnings": validation_warnings,
            "next_actions": [
                {
                    "action": "start_repair_session",
                    "description": "Start a bounded repair against the current Work decision.",
                    "command": (
                        f"{EIEL_LAUNCHER} sessionManager start {work_id} --role IMPLEMENTATION_AUTHORITY "
                        f"--bootstrap-record {bootstrap['record_id']} --codex-id '<codex-id>' "
                        f"{_repo_flag(repository)} {_state_flag(args)}"
                    ),
                }
            ],
        }
    return None


def command_open(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    bootstrap, repository, bootstrap_age_hours, bootstrap_assessment = _bootstrap(args)
    works, sessions, selected_work, selected_session = _selected_context(args)

    status_counts = Counter(str(row.get("status")) for row in works)
    all_open_sessions = open_sessions(args.state_root, read_only=True)
    session_counts = Counter(str(row.get("status")) for row in all_open_sessions)
    next_actions: list[dict[str, Any]] = []
    status = "WORKSPACE_READY"
    problems: list[str] = []
    warnings: list[str] = []
    administrative_obligations: list[str] = []
    if bootstrap_age_hours > BOOTSTRAP_REFRESH_RECOMMENDATION_HOURS:
        warnings.append(
            f"bootstrap receipt is {bootstrap_age_hours:.2f} hours old; refresh is recommended before substantial new work"
        )

    bootstrap_staleness = list(bootstrap_assessment.get("staleness_errors") or [])
    warnings.extend(
        f"Bootstrap recovery receipt is stale: {problem}" for problem in bootstrap_staleness
    )
    if bootstrap_staleness:
        administrative_obligations.extend(bootstrap_staleness)
        next_actions.append(
            {
                "action": "refresh_bootstrap",
                "description": "Optionally refresh Bootstrap to improve recovery accuracy; staleness does not revoke operator authorization.",
                "command": _bootstrap_refresh_command(
                    bootstrap,
                    repository,
                    state_root=args.state_root,
                    requested_authority_domain=args.authority_domain,
                ),
            }
        )

    explicit_selection = bool(getattr(args, "work_id", None) or getattr(args, "session_id", None))
    lane_sessions = sessions
    if not explicit_selection:
        lane_sessions = [
            session for session in sessions if _session_matches_repository_lane(session, repository)
        ]
        unrelated_count = len(sessions) - len(lane_sessions)
        if unrelated_count:
            obligation = (
                f"{unrelated_count} resumable Session(s) belong to other repository lanes and were not selected."
            )
            administrative_obligations.append(obligation)
            warnings.append(obligation)

    if selected_session is None and len(lane_sessions) == 1:
        selected_session = lane_sessions[0]
        selected_work = selected_work or work_store(args.state_root).get_read_only(
            str(dict(selected_session.get("payload") or {}).get("work_id"))
        )

    selected_work_summary = (
        summarize_work_record(selected_work, state_root=args.state_root, read_only=True)
        if selected_work
        else None
    )
    selected_session_summary = (
        summarize_session_record(selected_session, state_root=args.state_root, authorize_live=False)
        if selected_session
        else None
    )
    work_gate = _canonical_work_gate(
        selected_work_summary,
        work=selected_work,
        session=selected_session,
        bootstrap=bootstrap,
        repository=repository,
        args=args,
    )

    if work_gate is not None:
        status = str(work_gate["status"])
        administrative_obligations.extend(list(work_gate.get("problems") or []))
        warnings.extend(
            f"Managed Work record requires reconciliation: {problem}"
            for problem in work_gate.get("problems") or []
        )
        warnings.extend(list(work_gate.get("warnings") or []))
        next_actions.extend(list(work_gate.get("next_actions") or []))
    elif selected_session:
        refresh_problems = _session_refresh_problems(
            selected_session, bootstrap=bootstrap, repository=repository
        )
        session_payload = dict(selected_session.get("payload") or {})
        work_payload = dict(selected_work.get("payload") or {}) if selected_work else {}
        dependency = dependency_state(args.state_root, work_payload.get("dependency_work_ids") or [], read_only=True)
        session_outcome = str(session_payload.get("session_outcome") or "")
        review_verdict = str(session_payload.get("review_verdict") or "")
        session_status = str(selected_session.get("status") or "")
        if (not selected_work) or (
            str(selected_work.get("status") or "") in WORK_TERMINAL_STATUSES
            and session_status not in SESSION_TERMINAL_STATUSES
        ):
            status = "TERMINAL_PARENT_RECONCILIATION_REQUIRED"
            obligation = "selected non-terminal Session has no open parent Work record"
            administrative_obligations.append(obligation)
            warnings.append(f"Workflow obligation: {obligation}")
            next_actions.append(
                {
                    "action": "reconcile_terminal_parent",
                    "description": "Do not execute or resume this Session. Inspect and explicitly reconcile the terminal parent lifecycle.",
                    "command": (
                        f"{EIEL_LAUNCHER} workManager inspect {selected_work['record_id']} {_state_flag(args)}"
                        if selected_work
                        else f"{EIEL_LAUNCHER} sessionManager inspect {selected_session['record_id']} {_state_flag(args)}"
                    ),
                }
            )
        elif session_payload.get("worker_identity_state") == "PENDING" and session_payload.get("prepared_assignment_id"):
            status = "SCHEDULER_ASSIGNMENT_OWNS_SESSION"
            next_actions.extend(
                [
                    {
                        "action": "scheduler_assignment_owns_session",
                        "description": "A Scheduler Assignment owns this prepared Session; do not resume it through the ordinary path.",
                        "command": f"{EIEL_LAUNCHER} assignment status {_state_flag(args)}",
                    },
                    {
                        "action": "manual_takeover",
                        "description": "Explicitly supersede the unstarted Scheduler attempt and resume manually.",
                        "command": (
                            f"{EIEL_LAUNCHER} sessionManager resume {selected_session['record_id']} "
                            "--take-over-prepared --takeover-reason '<reason>' --codex-id '<codex-id>' "
                            f"--bootstrap-record {bootstrap['record_id']} {_repo_flag(repository)} {_state_flag(args)}"
                        ),
                    },
                ]
            )
        elif session_outcome == "PROMOTION_REQUESTED":
            status = "PROMOTION_REQUIRED"
            next_actions.append(
                {
                    "action": "return_to_engineering_authority",
                    "description": "The implementation session stopped because uncertainty exceeded the approved Work Package. Do not resume it.",
                    "command": f"{EIEL_LAUNCHER} workManager inspect {selected_work['record_id']} {_state_flag(args)}" if selected_work else None,
                }
            )
        elif str(selected_session.get("status")) in SESSION_TERMINAL_STATUSES and session_outcome in DELIVERY_SESSION_OUTCOMES:
            status = "WORK_REVIEW_REQUIRED"
            next_actions.append(
                {
                    "action": "start_review",
                    "description": "Implementation is complete. Start a Review Authority session for the exact Work Package; use reduced assurance if independence is unavailable.",
                    "command": (
                        f"{EIEL_LAUNCHER} sessionManager start {selected_work['record_id']} --role REVIEW_AUTHORITY "
                        f"--review-target-session {selected_session['record_id']} "
                        f"--bootstrap-record {bootstrap['record_id']} --codex-id '<codex-id>' "
                        f"{_repo_flag(repository)} {_state_flag(args)}"
                    ) if selected_work else None,
                }
            )
        elif str(selected_session.get("status")) in SESSION_TERMINAL_STATUSES and session_outcome == "REVIEW_COMPLETE":
            if review_verdict == "ACCEPT":
                status = "WORK_READY_TO_ACCEPT"
                next_actions.append(
                    {
                        "action": "accept_work",
                        "description": "Evidence-entailing review accepted the implementation. Complete the Work Item explicitly.",
                        "command": f"{EIEL_LAUNCHER} workManager complete {selected_work['record_id']} --review-session {selected_session['record_id']} {_state_flag(args)}" if selected_work else None,
                    }
                )
            elif review_verdict == "PROMOTE":
                status = "PROMOTION_REQUIRED"
                next_actions.append(
                    {
                        "action": "return_to_engineering_authority",
                        "description": "Review found higher uncertainty than the Work Package permits.",
                        "command": f"{EIEL_LAUNCHER} workManager inspect {selected_work['record_id']} {_state_flag(args)}" if selected_work else None,
                    }
                )
            else:
                status = "WORK_CHANGES_REQUIRED"
                next_actions.append(
                    {
                        "action": "start_repair_session",
                        "description": "Review did not accept the implementation. Start a bounded repair session or promote the Work Package.",
                        "command": (
                            f"{EIEL_LAUNCHER} sessionManager start {selected_work['record_id']} --role IMPLEMENTATION_AUTHORITY "
                            f"--bootstrap-record {bootstrap['record_id']} --codex-id '<codex-id>' "
                            f"{_repo_flag(repository)} {_state_flag(args)}"
                        ) if selected_work else None,
                    }
                )
        elif refresh_problems:
            status = "SESSION_BOOTSTRAP_REFRESH_REQUIRED"
            administrative_obligations.extend(refresh_problems)
            warnings.extend(f"Workflow obligation: {problem}" for problem in refresh_problems)
            next_actions.append(
                {
                    "action": "resume_with_current_bootstrap",
                    "description": "Resume the session while binding the current verified bootstrap receipt.",
                    "command": (
                        f"{EIEL_LAUNCHER} sessionManager resume {selected_session['record_id']} "
                        f"--bootstrap-record {bootstrap['record_id']} --codex-id '<codex-id>' "
                        f"{_repo_flag(repository)} {_state_flag(args)}"
                    ),
                }
            )
        elif not dependency["all_satisfied"]:
            status = "SESSION_BLOCKED_BY_WORK"
            obligation = "selected Work record has unresolved dependencies"
            administrative_obligations.append(obligation)
            warnings.append(f"Workflow obligation: {obligation}")
            blocker = (dependency["pending"] or dependency["terminal_unsatisfied"] or [{"work_id": "<WORK-ID>"}])[0]
            next_actions.append(
                {
                    "action": "inspect_blocker",
                    "description": "Complete or repair the blocking work before resuming this session.",
                    "command": f"{EIEL_LAUNCHER} workManager inspect {blocker.get('work_id')} {_state_flag(args)}",
                }
            )
        elif str(selected_session.get("status")) in SESSION_TERMINAL_STATUSES:
            status = "SESSION_TERMINAL"
            obligation = f"selected Session is {str(selected_session.get('status')).lower()}"
            administrative_obligations.append(obligation)
            warnings.append(f"Workflow obligation: {obligation}")
            next_actions.append(
                {
                    "action": "start_replacement_session",
                    "description": "Start a new session for the still-open work item if continuation is authorized.",
                    "command": (
                        f"{EIEL_LAUNCHER} sessionManager start {selected_work['record_id']} "
                        f"--bootstrap-record {bootstrap['record_id']} --codex-id '<codex-id>' "
                        f"{_repo_flag(repository)} {_state_flag(args)}"
                    ) if selected_work else None,
                }
            )
        elif session_status != "ACTIVE":
            status = "SESSION_RESUME_REQUIRED"
            next_actions.append(
                {
                    "action": "resume_selected_session",
                    "description": "The Session is retained but not ACTIVE. Resume it explicitly before executing repository work.",
                    "command": (
                        f"{EIEL_LAUNCHER} sessionManager resume {selected_session['record_id']} "
                        f"--bootstrap-record {bootstrap['record_id']} --codex-id '<codex-id>' "
                        f"{_repo_flag(repository)} {_state_flag(args)}"
                    ),
                }
            )
        else:
            command_problems = _next_command_precondition_problems(
                selected_session,
                work=selected_work,
                bootstrap=bootstrap,
                repository=repository,
            )
            if command_problems:
                status = "SESSION_COMMAND_REFRESH_REQUIRED"
                administrative_obligations.extend(command_problems)
                warnings.extend(f"Workflow obligation: {problem}" for problem in command_problems)
                next_actions.append(
                    {
                        "action": "refresh_session_checkpoint",
                        "description": "Refresh the ACTIVE Session checkpoint before relying on its historical work command.",
                        "command": f"{EIEL_LAUNCHER} sessionManager checkpoint {selected_session['record_id']} {_state_flag(args)}",
                    }
                )
            else:
                next_actions.append(
                    {
                        "action": "continue_active_session",
                        "description": session_payload.get("next_action") or "Continue the ACTIVE Session.",
                        "command": session_payload.get("next_command"),
                    }
                )
    elif len(lane_sessions) > 1:
        status = "WORKSPACE_READY"
        obligation = "Several Sessions are resumable in this repository lane; none was selected implicitly."
        administrative_obligations.append(obligation)
        warnings.append(f"Workflow choice available: {obligation}")
        next_actions.append(
            {
                "action": "select_session",
                "description": "List sessions and reopen the workspace with one explicit session ID.",
                "command": f"{EIEL_LAUNCHER} sessionManager list --open-only {_state_flag(args)}",
            }
        )
    elif selected_work:
        work_payload = dict(selected_work.get("payload") or {})
        dependency = dependency_state(args.state_root, work_payload.get("dependency_work_ids") or [], read_only=True)
        derived_status = str(dict(selected_work_summary or {}).get("derived_status") or "")
        if derived_status == "PROMOTION_REQUIRED":
            status = "PROMOTION_REQUIRED"
            next_actions.extend(list(dict(selected_work_summary or {}).get("next_actions") or [])[:1])
        elif derived_status == "REVIEW_REQUIRED":
            status = "WORK_REVIEW_REQUIRED"
            next_actions.append(
                {
                    "action": "start_review",
                    "description": "Implementation is complete. Start a Review Authority session; use reduced assurance if independence is unavailable.",
                    "command": (
                        f"{EIEL_LAUNCHER} sessionManager start {selected_work['record_id']} --role REVIEW_AUTHORITY "
                        f"--review-target-session {dict(dict(selected_work_summary or {}).get('review') or {}).get('latest_delivery_session_id')} "
                        f"--bootstrap-record {bootstrap['record_id']} --codex-id '<codex-id>' "
                        f"{_repo_flag(repository)} {_state_flag(args)}"
                    ),
                }
            )
        elif derived_status == "READY_TO_ACCEPT":
            status = "WORK_READY_TO_ACCEPT"
            accepted = list(dict(dict(selected_work_summary or {}).get("review") or {}).get("accepted_review_sessions") or [])
            review_id = dict(accepted[0]).get("session_id") if accepted else "<SES-REVIEW-ID>"
            next_actions.append(
                {
                    "action": "accept_work",
                    "description": "Evidence-entailing review accepted the implementation. Complete the Work Item explicitly.",
                    "command": f"{EIEL_LAUNCHER} workManager complete {selected_work['record_id']} --review-session {review_id} {_state_flag(args)}",
                }
            )
        elif derived_status == "CHANGES_REQUIRED":
            status = "WORK_CHANGES_REQUIRED"
            next_actions.append(
                {
                    "action": "start_repair_session",
                    "description": "Review did not accept the implementation. Start a bounded repair session or promote the Work Package.",
                    "command": (
                        f"{EIEL_LAUNCHER} sessionManager start {selected_work['record_id']} --role IMPLEMENTATION_AUTHORITY "
                        f"--bootstrap-record {bootstrap['record_id']} --codex-id '<codex-id>' "
                        f"{_repo_flag(repository)} {_state_flag(args)}"
                    ),
                }
            )
        elif dict(selected_work_summary or {}).get("review", {}).get("next_package_binding"):
            next_actions.extend(list(dict(selected_work_summary or {}).get("next_actions") or [])[:1])
        elif not dependency["all_satisfied"]:
            status = "WORK_BLOCKED_BY_DEPENDENCY"
            obligation = "selected Work record has unresolved dependencies"
            administrative_obligations.append(obligation)
            warnings.append(f"Workflow obligation: {obligation}")
            blocker = (dependency["pending"] or dependency["terminal_unsatisfied"] or [{"work_id": "<WORK-ID>"}])[0]
            next_actions.append(
                {
                    "action": "inspect_blocker",
                    "description": "Complete or repair the blocking work before starting a session.",
                    "command": f"{EIEL_LAUNCHER} workManager inspect {blocker.get('work_id')} {_state_flag(args)}",
                }
            )
        else:
            next_actions.append(
                {
                    "action": "start_session",
                    "description": "No resumable session exists for the selected work item; start a new one.",
                    "command": (
                        f"{EIEL_LAUNCHER} sessionManager start {selected_work['record_id']} "
                        f"--bootstrap-record {bootstrap['record_id']} --codex-id '<codex-id>' "
                        f"{_repo_flag(repository)} {_state_flag(args)}"
                    ),
                }
            )
    elif works:
        status = "WORKSPACE_READY"
        obligation = f"{len(works)} open Work Item(s) are available but none was explicitly selected."
        administrative_obligations.append(obligation)
        warnings.append(f"Workflow choice available: {obligation}")
        next_actions.append(
            {
                "action": "select_work",
                "description": "Optionally list open Work Items and reopen with --work-id when one is relevant to the operator request.",
                "command": f"{EIEL_LAUNCHER} workManager list --open-only --view priority {_state_flag(args)}",
            }
        )
    else:
        status = "NO_OPEN_WORK"
        next_actions.append(
            {
                "action": "create_work",
                "description": "Proceed from the explicit operator instruction; optionally create a Work Item when durable tracking would help.",
                "command": f"{EIEL_LAUNCHER} workManager create --title '<title>' --summary '<scope>' {_state_flag(args)}",
            }
        )

    work_payload = dict(selected_work.get("payload") or {}) if selected_work else {}
    session_payload = dict(selected_session.get("payload") or {}) if selected_session else {}
    mission_id = session_payload.get("mission_id") or work_payload.get("mission_id")
    implementation_and_qualification = _implementation_and_qualification_state(
        repository,
        selected_work,
        state_root=args.state_root,
        qualification_required=(
            str(dict(dict(bootstrap.get("payload") or {}).get("task") or {}).get("mode") or "").lower()
            == "qualification"
        ),
    )
    task_context = _task_context(bootstrap, implementation_and_qualification)
    context_receipt = None
    context_receipt_validation = None
    initial_context = None
    output_policy = None
    execution_budget = None
    context_health = None
    if selected_work and selected_session:
        try:
            context_receipt = build_context_receipt(
                work_record=selected_work,
                session_record=selected_session,
                bootstrap_record=bootstrap,
                repo_root=repository["root"],
                state_root=args.state_root,
            )
            context_receipt_validation = validate_context_receipt(
                context_receipt,
                work_record=selected_work,
                session_record=selected_session,
                repo_root=repository["root"],
                state_root=args.state_root,
            )
            if not context_receipt_validation["valid"]:
                receipt_errors = list(context_receipt_validation["errors"])
                administrative_obligations.extend(receipt_errors)
                warnings.extend(
                    f"Context Receipt refresh obligation: {problem}"
                    for problem in receipt_errors
                )
            warnings.extend(context_receipt_validation["warnings"])
            scope = dict(context_receipt.get("scope") or {})
            initial_context = {
                "entry_files": list(scope.get("entry_files") or []),
                "required_documents": list(scope.get("required_documents") or []),
                "search_roots": list(scope.get("search_roots") or []),
                "excluded_roots": list(scope.get("excluded_roots") or []),
                "known_symbols": list(scope.get("known_symbols") or []),
                "maximum_initial_files": scope.get("maximum_initial_files"),
                "broad_search_allowed": scope.get("broad_search_allowed"),
            }
            output_policy = context_receipt.get("output_policy")
            execution_budget = context_receipt.get("execution_budget")
            context_health = context_receipt.get("context_health")
        except ActorError as exc:
            receipt_errors = list(exc.problems or [str(exc)])
            administrative_obligations.extend(receipt_errors)
            warnings.extend(
                f"Context Receipt assistance unavailable: {problem}"
                for problem in receipt_errors
            )
            warnings.extend(list(exc.warnings or []))
    elif selected_work:
        binding = dict(work_payload.get("work_package_binding") or {})
        if binding:
            verified = verify_work_package_binding(
                binding,
                repo_root=repository["root"],
                expected_work_id=str(selected_work.get("record_id")),
                state_root=args.state_root,
            )
            if verified.get("package"):
                package = dict(verified["package"])
                initial_context = effective_read_contract(package)
                output_policy = effective_output_policy(package)
                execution_budget = effective_execution_budget(package)

    if (
        selected_session
        and context_health
        and context_health.get("handoff_recommended")
        and str(selected_session.get("status")) not in SESSION_TERMINAL_STATUSES
        and status == "WORKSPACE_READY"
    ):
        status = "CONTEXT_HANDOFF_RECOMMENDED"
        warnings.extend(str(reason) for reason in context_health.get("reasons") or [])
        next_actions.insert(
            0,
            {
                "action": "checkpoint_context_epoch",
                "description": "Context health recommends ending this conversation at the current safe checkpoint.",
                "command": (
                    f"{EIEL_LAUNCHER} sessionManager context-receipt {selected_session['record_id']} "
                    f"--attach {_repo_flag(repository)} {_state_flag(args)}"
                ),
            },
        )
    portfolio_work_rows = []
    for row in works:
        summary = summarize_work_record(row, state_root=args.state_root, read_only=True)
        binding = dict(summary.get("work_package_binding") or {})
        portfolio_work_rows.append(
            {
                "work_id": row.get("record_id"),
                "status": row.get("status"),
                "derived_status": summary.get("derived_status"),
                "title": dict(row.get("payload") or {}).get("title"),
                "priority": dict(row.get("payload") or {}).get("priority"),
                "next_action": dict(row.get("payload") or {}).get("next_action"),
                "work_package": (
                    {
                        "path": binding.get("path"),
                        "sha256": binding.get("sha256"),
                        "recommended_authority": binding.get("recommended_authority"),
                        "recommended_model": binding.get("recommended_model"),
                        "uncertainty_class": binding.get("uncertainty_class"),
                    }
                    if binding
                    else None
                ),
            }
        )
    invoked_as = str(getattr(args, "invoked_as", "") or getattr(args, "command", "") or "open")
    cli_entry = command_entry(invoked_as)
    payload = {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "operation": "INSPECTION_ONLY",
        "cli_compatibility": cli_entry,
        "operator_contract_version": OPERATOR_CONTRACT_VERSION,
        "tooling_version": ACTOR_MANAGERS_VERSION,
        "state_root": str(actor_state_root(args.state_root)),
        "repository": repository,
        "bootstrap": {
            "record_id": bootstrap.get("record_id"),
            "status": bootstrap.get("status"),
            "age_hours": round(bootstrap_age_hours, 4),
            "task": dict(bootstrap.get("payload") or {}).get("task"),
            "environment": dict(bootstrap.get("payload") or {}).get("environment"),
            "record_sha256": bootstrap.get("record_sha256"),
            "assessment": bootstrap_assessment,
        },
        "portfolio": {
            "open_work_count": len(works),
            "work_status_counts": dict(status_counts),
            "resumable_session_count": len(all_open_sessions),
            "session_status_counts": dict(session_counts),
            "open_work": portfolio_work_rows,
            "resumable_sessions": [
                {
                    "session_id": row.get("record_id"),
                    "work_id": dict(row.get("payload") or {}).get("work_id"),
                    "status": row.get("status"),
                    "current_codex_id": dict(row.get("payload") or {}).get("current_codex_id"),
                    "current_step": dict(row.get("payload") or {}).get("current_step"),
                    "next_action": dict(row.get("payload") or {}).get("next_action"),
                }
                for row in all_open_sessions
            ],
        },
        "selected_work": (
            {
                "record_id": selected_work.get("record_id"),
                "status": selected_work.get("status"),
                "payload": work_payload,
                "summary": selected_work_summary,
                "record_sha256": selected_work.get("record_sha256"),
            }
            if selected_work
            else None
        ),
        "selected_session": (
            {
                "record_id": selected_session.get("record_id"),
                "status": selected_session.get("status"),
                "payload": session_payload,
                "summary": selected_session_summary,
                "record_sha256": selected_session.get("record_sha256"),
            }
            if selected_session
            else None
        ),
        "implementation_and_qualification": implementation_and_qualification,
        "task_context": task_context,
        "context_receipt": context_receipt,
        "context_receipt_validation": context_receipt_validation,
        "initial_context": initial_context,
        "output_policy": output_policy,
        "execution_budget": execution_budget,
        "context_health": context_health,
        "mission": _mission_summary(args.state_root, str(mission_id) if mission_id else None),
        "bridge": _optional_bridge_summary(repository["root"]),
        "authority_and_model_policy": {
            "operator_instruction_is_primary_operational_authority": True,
            "workflow_records_are_optional_assistance": True,
            "administrative_incompleteness_blocks_execution": False,
            "managed_acceptance_and_claim_gates_remain_strict": True,
            "model_recommendation_is_advisory": True,
            "human_selects_model": True,
            "workspace_recommends_next_action_but_does_not_accept_work": True,
        },
        "workflow_administration": {
            "required_for_operator_authorized_execution": False,
            "obligations": administrative_obligations,
        },
        "normal_path": [
            "explicit_operator_instruction",
            "bootstrap",
            "workspace",
            "workManager",
            "sessionManager",
            "missionPlanner_when_needed",
            "qualificationManager_when_needed",
            "resourceManager_when_needed",
            "existing_campaign_tools",
            "evidenceManager_when_needed",
            "knowledge_librarian",
        ],
    }
    result = with_guidance(
        {
            "ok": not problems,
            "execution_blocked": bool(problems),
            "workflow_administration_required": bool(administrative_obligations),
            "workspace": payload,
        },
        status=status,
        reason_codes=[status.lower()],
        problems=problems,
        warnings=warnings,
        next_actions=next_actions,
        related_records=[
            str(bootstrap["record_id"]),
            *([str(selected_work["record_id"])] if selected_work else []),
            *([str(selected_session["record_id"])] if selected_session else []),
            *([str(mission_id)] if mission_id else []),
        ],
    )
    return result, 0 if not problems else 2


def _check_reference(
    *,
    state_root: str | Path | None,
    field: str,
    record_id: str,
) -> str | None:
    stores = {
        "qualification_record_ids": qualification_store(state_root),
        "resource_record_ids": resource_store(state_root),
        "evidence_assessment_ids": evidence_store(state_root),
    }
    try:
        stores[field].get_read_only(record_id)
    except ActorError as exc:
        return str(exc)
    return None


def _resolve_current_or_history(store_obj: Any, record_id: str) -> tuple[bool, str | None]:
    """Resolve a durable identity through current state or sealed history."""

    try:
        store_obj.get_read_only(record_id)
        return True, None
    except ActorError as current_error:
        try:
            history = store_obj.history_read_only(record_id)
        except ActorError as history_error:
            return False, f"current={current_error}; history={history_error}"
        if not history:
            return False, str(current_error)
        return True, "resolved through immutable history because current state is unavailable"


def _find_graph_cycle(graph: Mapping[str, Sequence[str]]) -> list[str] | None:
    visiting: set[str] = set()
    visited: set[str] = set()
    trail: list[str] = []

    def visit(node: str) -> list[str] | None:
        if node in visiting:
            index = trail.index(node)
            return [*trail[index:], node]
        if node in visited:
            return None
        visiting.add(node)
        trail.append(node)
        for parent in graph.get(node, []):
            cycle = visit(str(parent))
            if cycle:
                return cycle
        trail.pop()
        visiting.remove(node)
        visited.add(node)
        return None

    for node in sorted(graph):
        cycle = visit(node)
        if cycle:
            return cycle
    return None


def _audit_store(
    namespace: str,
    store_obj: Any,
    *,
    live_repository_source_record_ids: set[str] | None = None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    problems: list[str] = []
    warnings: list[str] = []
    rows = store_obj.list_read_only()
    valid = 0
    recoverable_count = 0
    for row in rows:
        record_id = str(row.get("record_id") or "<unknown>")
        if row.get("status") == "INVALID":
            problems.append(
                f"{namespace} current/history state is corrupt: {record_id}: "
                f"{row.get('error') or row.get('errors')}"
            )
            continue
        if row.get("integrity_classification") == "RECOVERABLE_STALE" or row.get("status") == "RECOVERABLE_STALE":
            recoverable_count += 1
            warnings.append(
                f"{namespace} recoverable stale state: {record_id}: "
                f"{row.get('error') or 'explicit promotion/index rebuild required'}"
            )
            continue
        valid += 1
        try:
            current = store_obj.get_read_only(record_id)
            history = store_obj.history_read_only(record_id)
        except ActorError as exc:
            problems.append(f"{namespace} record/history failure: {record_id}: {exc}")
            continue
        if not history:
            problems.append(f"{namespace} record has no immutable history: {record_id}")
        else:
            latest = history[-1]
            if int(latest.get("revision") or 0) != int(current.get("revision") or 0):
                problems.append(f"{namespace} current revision does not match latest history: {record_id}")
            if latest.get("record_sha256") != current.get("record_sha256"):
                problems.append(f"{namespace} current hash does not match latest history: {record_id}")
        source_validation = validate_record_source_references(
            current, state_root=store_obj.reference_root
        )
        for error in source_validation.get("errors") or []:
            historical_repository_source = (
                live_repository_source_record_ids is not None
                and record_id not in live_repository_source_record_ids
                and "repository source path is not available" in str(error)
            )
            target = warnings if historical_repository_source else problems
            qualifier = "historical " if historical_repository_source else ""
            target.append(f"{namespace} {record_id} {qualifier}source reference: {error}")
        for warning in source_validation.get("warnings") or []:
            warnings.append(f"{namespace} {record_id} source reference: {warning}")

    index_audit = store_obj.audit_index_read_only()
    classification = str(index_audit.get("classification") or "CORRUPT")
    if classification == "CORRUPT":
        for finding in index_audit.get("corrupt_findings") or []:
            problems.append(f"{namespace} store integrity: {finding}")
        for finding in index_audit.get("recoverable_findings") or []:
            warnings.append(f"{namespace} derived/recoverable integrity: {finding}")
    elif classification == "RECOVERABLE_STALE":
        for finding in index_audit.get("recoverable_findings") or []:
            warnings.append(f"{namespace} recoverable stale integrity: {finding}")

    return problems, warnings, {
        "records": len(rows),
        "valid_records": valid,
        "recoverable_records": recoverable_count,
        "integrity_classification": classification,
        "repair_actions": list(index_audit.get("repair_actions") or []),
        "index_record_count": int(index_audit.get("index_record_count") or 0),
    }


def command_audit(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    state_root = actor_state_root(args.state_root)
    repository = git_repository_identity(args.repo_root or repository_root())
    live_bootstrap_ids: set[str] = set()
    for row in session_store(state_root).list_read_only():
        if row.get("status") in SESSION_TERMINAL_STATUSES | {"INVALID", "RECOVERABLE_STALE"}:
            continue
        session = session_store(state_root).get_read_only(str(row["record_id"]))
        bootstrap_id = str(dict(session.get("payload") or {}).get("bootstrap_record_id") or "")
        if bootstrap_id:
            live_bootstrap_ids.add(bootstrap_id)
    stores = {
        "bootstrap": bootstrap_store(state_root),
        "work": work_store(state_root),
        "sessions": session_store(state_root),
        "missions": mission_store(state_root),
        "qualifications": qualification_store(state_root),
        "resources": resource_store(state_root),
        "evidence-decisions": evidence_store(state_root),
        "scheduler/dispatch-requests": dispatch_store(state_root),
        "scheduler/assignments": assignment_store(state_root),
        "scheduler/leases": lease_store(state_root),
        "scheduler/worker-results": result_store(state_root),
    }
    problems: list[str] = []
    warnings: list[str] = []
    store_summaries: dict[str, Any] = {}
    implementation_and_qualification: list[dict[str, Any]] = []
    for namespace, store_obj in stores.items():
        p, w, summary = _audit_store(
            namespace,
            store_obj,
            live_repository_source_record_ids=(
                live_bootstrap_ids if namespace == "bootstrap" else None
            ),
        )
        problems.extend(p)
        warnings.extend(w)
        store_summaries[namespace] = summary

    graph = dependency_graph(state_root, read_only=True)
    cycle = find_dependency_cycle(graph)
    if cycle:
        problems.append("work dependency cycle: " + " -> ".join(cycle))

    active_sessions_by_work: dict[str, list[str]] = defaultdict(list)
    for row in session_store(state_root).list_read_only():
        if row.get("status") in {"INVALID", "RECOVERABLE_STALE"}:
            continue
        record = session_store(state_root).get_read_only(str(row["record_id"]))
        payload = dict(record.get("payload") or {})
        session_id = str(record["record_id"])
        work_id = str(payload.get("work_id") or "")
        try:
            work = work_store(state_root).get_read_only(work_id)
        except ActorError:
            problems.append(f"session {session_id} references missing work {work_id}")
            work = None
        session_role = str(payload.get("session_role") or "IMPLEMENTATION_AUTHORITY")
        if session_role not in AUTHORITY_ROLES:
            problems.append(f"session {session_id} has invalid authority role {session_role}")
        session_binding = payload.get("work_package_binding")
        if work:
            work_binding = dict(dict(work.get("payload") or {}).get("work_package_binding") or {})
            session_terminal = str(record.get("status")) in SESSION_TERMINAL_STATUSES
            if work_binding:
                if not session_binding:
                    if not session_terminal:
                        problems.append(f"session {session_id} is missing work package binding for {work_id}")
                else:
                    session_repository = dict(payload.get("repository") or {})
                    package_validation = verify_work_package_binding(
                        session_binding,
                        repo_root=session_repository.get("repo_root"),
                        expected_work_id=work_id,
                        state_root=state_root,
                    )
                    for error in package_validation["errors"]:
                        problems.append(f"session {session_id} work package: {error}")
                    for warning in package_validation["warnings"]:
                        warnings.append(f"session {session_id} work package: {warning}")
                    if (
                        str(dict(session_binding).get("sha256") or "")
                        != str(work_binding.get("sha256") or "")
                        and not session_terminal
                    ):
                        problems.append(f"session {session_id} work package digest differs from work {work_id}")
            elif session_binding and not session_terminal:
                problems.append(f"session {session_id} has a work package binding but work {work_id} does not")
        current_bootstrap_id = str(payload.get("bootstrap_record_id") or "")
        resolved, resolution_note = _resolve_current_or_history(bootstrap_store(state_root), current_bootstrap_id)
        if not resolved:
            problems.append(f"session {session_id} references missing bootstrap {current_bootstrap_id}: {resolution_note}")
        elif resolution_note:
            warnings.append(f"session {session_id} bootstrap {current_bootstrap_id}: {resolution_note}")
        seen_bootstraps: set[str] = set()
        for historical_bootstrap_id in normalize_string_list(payload.get("bootstrap_history")):
            if historical_bootstrap_id in seen_bootstraps:
                problems.append(f"session {session_id} repeats Bootstrap history identity {historical_bootstrap_id}")
                continue
            seen_bootstraps.add(historical_bootstrap_id)
            resolved, resolution_note = _resolve_current_or_history(
                bootstrap_store(state_root), historical_bootstrap_id
            )
            if not resolved:
                problems.append(
                    f"session {session_id} references missing historical bootstrap "
                    f"{historical_bootstrap_id}: {resolution_note}"
                )
            elif resolution_note:
                warnings.append(
                    f"session {session_id} historical bootstrap {historical_bootstrap_id}: {resolution_note}"
                )
        mission_id = payload.get("mission_id")
        if mission_id:
            try:
                mission_store(state_root).get_read_only(str(mission_id))
            except ActorError:
                warnings.append(f"session {session_id} references mission not present in local state: {mission_id}")
        for blocker in normalize_string_list(payload.get("blocked_by_work_ids")):
            if not work_store(state_root).exists(blocker):
                problems.append(f"session {session_id} references missing blocking work {blocker}")
        for field in REFERENCE_NAMESPACE_MAP:
            for record_id in normalize_string_list(dict(payload.get("references") or {}).get(field)):
                error = _check_reference(state_root=state_root, field=field, record_id=record_id)
                if error:
                    problems.append(f"session {session_id} {field}: {error}")
        if work and payload.get("resource_accounting") is not None:
            for error in validate_session_resource_accounting_binding(
                payload.get("resource_accounting"),
                session_record=record,
                work_record=work,
                state_root=state_root,
            ):
                problems.append(f"session {session_id} resource accounting: {error}")
        receipt_ref = payload.get("context_receipt")
        if work and isinstance(receipt_ref, Mapping) and receipt_ref:
            receipt_path = Path(str(receipt_ref.get("path") or ""))
            if not receipt_path.is_file():
                problems.append(f"session {session_id} Context Receipt is missing: {receipt_path}")
            else:
                try:
                    retained_receipt = read_json(receipt_path, name=f"Context Receipt for {session_id}")
                except ActorError as exc:
                    problems.append(f"session {session_id} Context Receipt: {exc}")
                else:
                    retained_sha = str(retained_receipt.get("receipt_sha256") or "")
                    referenced_sha = str(receipt_ref.get("sha256") or "")
                    if referenced_sha and referenced_sha != retained_sha:
                        problems.append(f"session {session_id} Context Receipt reference digest mismatch")
                    validation = validate_context_receipt(
                        retained_receipt,
                        work_record=work,
                        session_record=record,
                        repo_root=dict(payload.get("repository") or {}).get("repo_root"),
                        state_root=state_root,
                        historical_repository=session_terminal,
                    )
                    for error in validation.get("errors") or []:
                        problems.append(f"session {session_id} Context Receipt: {error}")
                    for warning in validation.get("warnings") or []:
                        warnings.append(f"session {session_id} Context Receipt: {warning}")
        if str(record.get("status")) == "ACTIVE":
            active_sessions_by_work[work_id].append(session_id)
        if work and str(work.get("status")) in WORK_TERMINAL_STATUSES and str(record.get("status")) not in SESSION_TERMINAL_STATUSES:
            problems.append(f"non-terminal session {session_id} belongs to terminal work {work_id}")

    for work_id, session_ids in active_sessions_by_work.items():
        if len(session_ids) > 1:
            rows = [session_store(state_root).get_read_only(session_id) for session_id in session_ids]
            ownership_problems = parallel_session_ownership_problems(rows)
            for problem in ownership_problems:
                problems.append(f"work {work_id}: {problem}")

    for row in work_store(state_root).list_read_only():
        if row.get("status") in {"INVALID", "RECOVERABLE_STALE"}:
            continue
        record = work_store(state_root).get_read_only(str(row["record_id"]))
        payload = dict(record.get("payload") or {})
        work_id = str(record["record_id"])
        parent = payload.get("parent_work_id")
        if parent and not work_store(state_root).exists(str(parent)):
            problems.append(f"work {work_id} references missing parent {parent}")
        superseded_by = str(payload.get("superseded_by_work_id") or "")
        if superseded_by:
            resolved, resolution_note = _resolve_current_or_history(work_store(state_root), superseded_by)
            if not resolved:
                problems.append(
                    f"work {work_id} references missing superseding work {superseded_by}: {resolution_note}"
                )
            elif superseded_by == work_id:
                problems.append(f"work {work_id} supersedes itself")
            elif resolution_note:
                warnings.append(f"work {work_id} superseding work {superseded_by}: {resolution_note}")
        deps = dependency_state(state_root, payload.get("dependency_work_ids") or [], read_only=True)
        status = str(record.get("status"))
        work_summary = summarize_work_record(record, state_root=state_root, read_only=True)
        accepted_review_id = str(payload.get("accepted_review_session_id") or "")
        if accepted_review_id:
            resolved, resolution_note = _resolve_current_or_history(
                session_store(state_root), accepted_review_id
            )
            if not resolved:
                problems.append(
                    f"work {work_id} references missing accepted review Session "
                    f"{accepted_review_id}: {resolution_note}"
                )
            else:
                try:
                    accepted_review = session_store(state_root).get_read_only(accepted_review_id)
                except ActorError:
                    accepted_review = None
                if accepted_review is not None:
                    accepted_payload = dict(accepted_review.get("payload") or {})
                    if str(accepted_payload.get("work_id") or "") != work_id:
                        problems.append(
                            f"work {work_id} accepted review {accepted_review_id} belongs to "
                            f"work {accepted_payload.get('work_id')}"
                        )
                    if str(accepted_payload.get("session_role") or "") != "REVIEW_AUTHORITY":
                        problems.append(
                            f"work {work_id} accepted review {accepted_review_id} is not Review Authority"
                        )
                    if str(accepted_payload.get("review_verdict") or "") != "ACCEPT":
                        problems.append(
                            f"work {work_id} accepted review {accepted_review_id} has no ACCEPT verdict"
                        )
                elif resolution_note:
                    warnings.append(
                        f"work {work_id} accepted review {accepted_review_id}: {resolution_note}; "
                        "semantic validation is limited to retained immutable history"
                    )
        binding = payload.get("work_package_binding")
        if binding:
            # V1 audit did not elevate generic Work warnings. Keep that legacy
            # behavior and add strict validation only for the new package contract.
            for error in dict(work_summary.get("validation") or {}).get("errors") or []:
                problems.append(f"work {work_id}: {error}")
            for warning in dict(work_summary.get("validation") or {}).get("warnings") or []:
                warnings.append(f"work {work_id}: {warning}")
            review = dict(work_summary.get("review") or {})
            for review_row in review.get("review_sessions") or []:
                for error in dict(review_row).get("problems") or []:
                    problems.append(f"work {work_id} review {review_row.get('session_id')}: {error}")
                for warning in dict(review_row).get("warnings") or []:
                    warnings.append(f"work {work_id} review {review_row.get('session_id')}: {warning}")
            if status == "COMPLETED":
                accepted_id = str(payload.get("accepted_review_session_id") or "")
                accepted = {
                    str(item.get("session_id")): item
                    for item in review.get("accepted_review_sessions") or []
                }
                if not accepted_id:
                    problems.append(f"completed package-bound work {work_id} has no accepted review session")
                elif accepted_id not in accepted:
                    problems.append(f"completed package-bound work {work_id} cites invalid accepted review {accepted_id}")
        if status in {"READY", "ACTIVE"} and not deps["all_satisfied"]:
            problems.append(f"work {work_id} is {status} while dependencies are unresolved")
        if status == "BLOCKED" and deps["all_satisfied"]:
            warnings.append(f"work {work_id} is still BLOCKED although all dependencies are complete")
        mission_id = payload.get("mission_id")
        if mission_id:
            try:
                mission_store(state_root).get_read_only(str(mission_id))
            except ActorError:
                warnings.append(f"work {work_id} references mission not present in local state: {mission_id}")
        for field in REFERENCE_NAMESPACE_MAP:
            for record_id in normalize_string_list(dict(payload.get("references") or {}).get(field)):
                error = _check_reference(state_root=state_root, field=field, record_id=record_id)
                if error:
                    problems.append(f"work {work_id} {field}: {error}")
        implqual = _implementation_and_qualification_state(
            repository,
            record,
            state_root=state_root,
        )
        implementation_and_qualification.append(
            {
                "work_id": work_id,
                "implementation": implqual.get("implementation", {}),
                "qualification": implqual.get("qualification", {}),
            }
        )
        if status == "ACTIVE" and not sessions_for_work(state_root, work_id, read_only=True):
            warnings.append(f"active work {work_id} has no retained session")

    mission_graph: dict[str, list[str]] = {}
    for row in mission_store(state_root).list_read_only():
        if row.get("status") in {"INVALID", "RECOVERABLE_STALE"}:
            continue
        mission_id = str(row.get("record_id") or "")
        try:
            mission = mission_store(state_root).get_read_only(mission_id)
        except ActorError as exc:
            problems.append(f"mission {mission_id} cannot be read: {exc}")
            continue
        mission_payload = dict(mission.get("payload") or {})
        parents = normalize_string_list(mission_payload.get("parent_mission_ids"))
        mission_graph[mission_id] = parents
        if mission_id in parents:
            problems.append(f"mission {mission_id} references itself as a parent")
        for parent_id in parents:
            resolved, resolution_note = _resolve_current_or_history(mission_store(state_root), parent_id)
            if not resolved:
                problems.append(f"mission {mission_id} references missing parent mission {parent_id}: {resolution_note}")
            elif resolution_note:
                warnings.append(f"mission {mission_id} parent {parent_id}: {resolution_note}")
        mission_refs = dict(mission_payload.get("references") or {})
        closeout = dict(mission_payload.get("closeout") or {})
        for field, closeout_field in (
            ("qualification_record_ids", "qualification_refs"),
            ("resource_record_ids", "resource_refs"),
            ("evidence_assessment_ids", "evidence_refs"),
        ):
            values = [
                *normalize_string_list(mission_refs.get(field)),
                *normalize_string_list(closeout.get(closeout_field)),
            ]
            for record_id in sorted(set(values)):
                error = _check_reference(state_root=state_root, field=field, record_id=record_id)
                if error:
                    problems.append(f"mission {mission_id} {field}: {error}")
    mission_cycle = _find_graph_cycle(mission_graph)
    if mission_cycle:
        problems.append("mission parent cycle: " + " -> ".join(mission_cycle))

    # Scheduler state is execution state, not a second source of project
    # truth.  Traverse both directions of the durable execution graph so a
    # clean Actor-store audit cannot hide a dangling prepared Session,
    # Assignment, Lease, Worker Result, or retained Context Receipt binding.
    scheduler_integrity = audit_scheduler_reference_graph(state_root=state_root)
    problems.extend(scheduler_integrity.get("errors") or [])
    warnings.extend(scheduler_integrity.get("warnings") or [])

    # Preserve the earlier live-Work gate explicitly.  The composite graph
    # proves identity/reachability; this check proves that an open Assignment
    # remains runnable against current project truth rather than merely
    # pointing to a valid historical Work revision.
    for row in assignment_store(state_root).list_read_only():
        if row.get("status") in {"INVALID", "RECOVERABLE_STALE"}:
            continue
        assignment = assignment_store(state_root).get_read_only(str(row.get("record_id") or ""))
        assignment_id = str(assignment.get("record_id") or "")
        material = dict(dict(assignment.get("payload") or {}).get("material") or {})
        work_material = dict(material.get("work") or {})
        assigned_work_id = str(work_material.get("record_id") or work_material.get("work_id") or "")
        if not assigned_work_id or not work_store(state_root).exists(assigned_work_id):
            continue
        live_work = work_store(state_root).get_read_only(assigned_work_id)
        if str(assignment.get("status") or "") in {"PREPARING", "CLEANUP_REQUIRED", "READY", "CLAIMED", "RUNNING", "RESULT_SUBMITTED"}:
            if str(live_work.get("status") or "") in WORK_TERMINAL_STATUSES:
                problems.append(
                    f"scheduler assignment {assignment_id} remains {assignment.get('status')} for terminal work {assigned_work_id}"
                )
            material_revision = work_material.get("revision")
            material_sha = str(work_material.get("record_sha256") or "")
            if material_revision not in (None, live_work.get("revision")) or (
                material_sha and material_sha != str(live_work.get("record_sha256") or "")
            ):
                problems.append(f"scheduler assignment {assignment_id} retains stale Work identity for {assigned_work_id}")

    problems = list(dict.fromkeys(problems))
    warnings = list(dict.fromkeys(warnings))
    scheduler_classes = dict(scheduler_integrity.get("classifications") or {})
    scheduler_errors = set(scheduler_integrity.get("errors") or [])
    scheduler_warnings = set(scheduler_integrity.get("warnings") or [])
    structured_classes = {
        classification: list(scheduler_classes.get(classification) or [])
        for classification in (
            REFERENCE_CORRUPT,
            REFERENCE_RECOVERABLE_STALE,
            REFERENCE_SUPPORTED_HISTORICAL,
            REFERENCE_OPTIONAL_ABSENT,
            REFERENCE_EXTERNAL_UNVERIFIABLE,
        )
    }
    for item in problems:
        if item in scheduler_errors:
            continue
        structured_classes[REFERENCE_CORRUPT].append({
            "source_kind": "workspace",
            "source_id": None,
            "edge": "composite_audit",
            "target_kind": None,
            "target_id": None,
            "classification": REFERENCE_CORRUPT,
            "errors": [item],
            "warnings": [],
        })
    for item in warnings:
        if item in scheduler_warnings:
            continue
        lower = item.lower()
        classification = (
            REFERENCE_SUPPORTED_HISTORICAL
            if "histor" in lower
            else REFERENCE_EXTERNAL_UNVERIFIABLE
            if "external" in lower or "not present in local state" in lower
            else REFERENCE_RECOVERABLE_STALE
        )
        structured_classes[classification].append({
            "source_kind": "workspace",
            "source_id": None,
            "edge": "composite_audit",
            "target_kind": None,
            "target_id": None,
            "classification": classification,
            "errors": [],
            "warnings": [item],
        })
    status = "INCONSISTENT" if problems else "CONSISTENT_WITH_WARNINGS" if warnings else "CONSISTENT"
    next_actions: list[dict[str, Any]] = []
    if problems:
        next_actions.append(
            {
                "action": "repair_record_integrity",
                "description": "Repair the listed cross-record or lifecycle defects before relying on automatic recovery.",
            }
        )
    elif warnings:
        next_actions.append(
            {
                "action": "review_warnings",
                "description": "Review stale lifecycle states; no record was changed automatically.",
            }
        )
    else:
        next_actions.append(
            {
                "action": "open_workspace",
                "description": "Retained Codex operating state is internally consistent.",
                "command": f"{EIEL_LAUNCHER} workspace open {_state_flag(args)}",
            }
        )
    payload = with_guidance(
        {
            "ok": not problems,
            "audit": {
                "schema_version": WORKSPACE_AUDIT_SCHEMA_VERSION,
                "tooling_version": ACTOR_MANAGERS_VERSION,
                "generated_at_utc": utc_now(args.generated_at),
                "state_root": str(state_root),
                "implementation_and_qualification": implementation_and_qualification,
                "store_summaries": store_summaries,
                "dependency_nodes": len(graph),
                "integrity_classifications": structured_classes,
                "scheduler_reference_graph": scheduler_integrity,
                "problems": problems,
                "warnings": warnings,
            },
        },
        status=status,
        reason_codes=[status.lower()],
        problems=problems,
        warnings=warnings,
        next_actions=next_actions,
    )
    code = 2 if problems or (warnings and args.strict) else 0
    return payload, code


def _handoff_material(handoff: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable handoff identity without presentation-time fields."""

    material = copy.deepcopy(dict(handoff))
    material.pop("generated_at_utc", None)
    material.pop("handoff_sha256", None)
    bootstrap = dict(material.get("bootstrap") or {})
    bootstrap.pop("age_hours", None)
    material["bootstrap"] = bootstrap or None
    return material


def load_workspace_handoff_artifact(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ActorError("Workspace handoff artifact must be an object", code="schema_record_malformed")
    value = dict(raw)
    handoff = dict(value.get("handoff") or value)
    require_explicit_schema_version(
        handoff,
        expected=WORKSPACE_HANDOFF_SCHEMA_VERSION,
        label="Workspace handoff artifact",
    )
    observed = handoff.get("handoff_sha256")
    actual = canonical_sha256(_handoff_material(handoff))
    if not isinstance(observed, str) or observed != actual:
        raise ActorError(
            "Workspace handoff digest mismatch",
            code="workspace_handoff_digest_mismatch",
        )
    return handoff



def command_handoff(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    opened, code = command_open(args)
    if code != 0:
        return opened, code
    workspace = dict(opened.get("workspace") or {})
    selected_work = dict(workspace.get("selected_work") or {})
    selected_session = dict(workspace.get("selected_session") or {})
    action = (opened.get("next_actions") or [{}])[0]
    handoff = {
        "schema_version": WORKSPACE_HANDOFF_SCHEMA_VERSION,
        "operator_contract_version": OPERATOR_CONTRACT_VERSION,
        "tooling_version": ACTOR_MANAGERS_VERSION,
        "generated_at_utc": utc_now(args.generated_at),
        "repository": workspace.get("repository"),
        "state_root": workspace.get("state_root"),
        "bootstrap": workspace.get("bootstrap"),
        "implementation_and_qualification": workspace.get("implementation_and_qualification"),
        "task_context": workspace.get("task_context"),
        "context_receipt": workspace.get("context_receipt"),
        "initial_context": workspace.get("initial_context"),
        "output_policy": workspace.get("output_policy"),
        "execution_budget": workspace.get("execution_budget"),
        "context_health": workspace.get("context_health"),
        "work": selected_work or None,
        "session": selected_session or None,
        "mission": workspace.get("mission"),
        "bridge": workspace.get("bridge"),
        "next_action": action,
        "resume_rule": "Execute only the recorded next command after checking its referenced manager status; exit code zero alone is not authorization.",
        "record_hashes": {
            "bootstrap": dict(workspace.get("bootstrap") or {}).get("record_sha256"),
            "work": selected_work.get("record_sha256") if selected_work else None,
            "session": selected_session.get("record_sha256") if selected_session else None,
            "mission": dict(workspace.get("mission") or {}).get("record_sha256"),
        },
    }
    handoff["handoff_sha256"] = canonical_sha256(_handoff_material(handoff))
    return with_guidance(
        {"ok": True, "handoff": handoff},
        status="HANDOFF_READY",
        reason_codes=["workspace_handoff_ready"],
        next_actions=[action] if action else [],
        related_records=[
            value
            for value in (
                dict(workspace.get("bootstrap") or {}).get("record_id"),
                selected_work.get("record_id") if selected_work else None,
                selected_session.get("record_id") if selected_session else None,
                dict(workspace.get("mission") or {}).get("mission_id"),
            )
            if value
        ],
    ), 0


def _human(payload: Mapping[str, Any]) -> str:
    def append_findings(lines: list[str], source: Mapping[str, Any]) -> None:
        problems = priority_ordered_findings(source.get("problems") or payload.get("problems") or [])
        warnings = priority_ordered_findings(source.get("warnings") or payload.get("warnings") or [])
        if problems:
            lines.append(f"Problems: {len(problems)}")
            lines.extend(f"Problem: {problem}" for problem in problems[:5])
        if warnings:
            lines.append(f"Warnings: {len(warnings)}")
            lines.extend(f"Warning: {warning}" for warning in warnings[:5])
        if len(problems) > 5 or len(warnings) > 5:
            lines.append("Details: rerun without --human or use --output <path> for the full structured payload.")

    if "audit" in payload:
        audit = dict(payload.get("audit") or {})
        lines = [f"Workspace audit: {payload.get('status')}"]
        append_findings(lines, audit)
        if not audit.get("problems") and not audit.get("warnings"):
            lines.extend(("Problems: 0", "Warnings: 0"))
        actions = list(payload.get("next_actions") or [])
        if actions:
            action = dict(actions[0])
            lines.append(f"Next: {action.get('description') or 'none'}")
            if action.get("command"):
                lines.append(f"Command: {action['command']}")
        return "\n".join(lines) + "\n"
    if "handoff" in payload:
        row = dict(payload.get("handoff") or {})
        work = dict(row.get("work") or {})
        session = dict(row.get("session") or {})
        implqual = row.get("implementation_and_qualification") or {}
        implementation = dict(implqual.get("implementation") or {})
        qualification = dict(implqual.get("qualification") or {})
        action = dict(row.get("next_action") or {})
        context_health = dict(row.get("context_health") or {})
        lines = [
            "EIEL Codex resume handoff",
            f"Work: {work.get('record_id') or 'none'}",
            f"Session: {session.get('record_id') or 'none'}",
            f"Implementation: {implementation.get('status') or 'UNKNOWN'}",
            f"Qualification: {qualification.get('status') or 'UNKNOWN'}",
            f"Context health: {context_health.get('status') or 'NOT_AVAILABLE'}",
            f"Next: {action.get('description') or 'none'}",
        ]
        if action.get("command"):
            lines.append(f"Command: {action['command']}")
        append_findings(lines, payload)
        return "\n".join(lines) + "\n"
    ws = dict(payload.get("workspace") or {})
    portfolio = dict(ws.get("portfolio") or {})
    selected_work = dict(ws.get("selected_work") or {})
    selected_session = dict(ws.get("selected_session") or {})
    implqual = dict(ws.get("implementation_and_qualification") or {})
    implementation = dict(implqual.get("implementation") or {})
    qualification = dict(implqual.get("qualification") or {})
    lines = [
        f"Workspace: {payload.get('status')}",
        f"Repository: {dict(ws.get('repository') or {}).get('root')}",
        f"Bootstrap: {dict(ws.get('bootstrap') or {}).get('record_id')}",
        f"Open work: {portfolio.get('open_work_count', 0)}",
        f"Resumable sessions: {portfolio.get('resumable_session_count', 0)}",
        f"Implementation: {implementation.get('status') or 'UNKNOWN'}",
        f"Qualification: {qualification.get('status') or 'UNKNOWN'}",
        f"Context health: {dict(ws.get('context_health') or {}).get('status') or 'NOT_AVAILABLE'}",
    ]
    if selected_work:
        lines.append(f"Work: {selected_work.get('record_id')} {dict(selected_work.get('payload') or {}).get('title')}")
    if selected_session:
        lines.append(f"Session: {selected_session.get('record_id')} {selected_session.get('status')}")
    actions = payload.get("next_actions") or []
    if actions:
        lines.append(f"Next: {actions[0].get('description')}")
        if actions[0].get("command"):
            lines.append(f"Command: {actions[0]['command']}")
    append_findings(lines, payload)
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = ActorArgumentParser(prog=f"{EIEL_LAUNCHER} workspace", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True, parser_class=ActorArgumentParser)
    for name, help_text in (
        ("open", "Open the current operational context without mutating it."),
        ("status", "Alias for workspace open."),
    ):
        cmd = sub.add_parser(name, help=help_text)
        _common(cmd, include_projection=True)
        cmd.add_argument("--work-id")
        cmd.add_argument("--session-id")
        cmd.add_argument("--invoked-as", help=argparse.SUPPRESS)
    audit = sub.add_parser("audit", help="Audit record integrity, cross-manager references, dependencies, and lifecycle consistency.")
    _common(audit, include_bootstrap=False)
    audit.add_argument("--strict", action="store_true")
    handoff = sub.add_parser("handoff", help="Export one consolidated deterministic Codex resume handoff.")
    _common(handoff)
    handoff.add_argument("--work-id")
    handoff.add_argument("--session-id")
    return parser


def _compact_full_view_command(args: argparse.Namespace, payload: Mapping[str, Any]) -> str:
    workspace = dict(payload.get("workspace") or {})
    repository = dict(workspace.get("repository") or {})
    bootstrap = dict(workspace.get("bootstrap") or {})
    selected_work = dict(workspace.get("selected_work") or {})
    selected_session = dict(workspace.get("selected_session") or {})
    parts = [f"{EIEL_LAUNCHER} workspace open --full"]
    for flag, value in (
        ("--repo-root", args.repo_root or repository.get("root")),
        ("--state-root", args.state_root or workspace.get("state_root")),
        ("--bootstrap-record", getattr(args, "bootstrap_record", None) or bootstrap.get("record_id")),
        ("--authority-domain", getattr(args, "authority_domain", None)),
        ("--work-id", getattr(args, "work_id", None) or selected_work.get("record_id")),
        ("--session-id", getattr(args, "session_id", None) or selected_session.get("record_id")),
    ):
        if value:
            parts.append(f"{flag} {shlex.quote(str(value))}")
    if getattr(args, "allow_stale_bootstrap", False):
        parts.append("--allow-stale-bootstrap")
    if getattr(args, "max_bootstrap_age_hours", None) is not None:
        parts.append(f"--max-bootstrap-age-hours {args.max_bootstrap_age_hours}")
    return " ".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw:
        raw = ["open"]
    args: argparse.Namespace | None = None
    try:
        args = build_parser().parse_args(raw)
        handlers = {
            "open": command_open,
            "status": command_open,
            "audit": command_audit,
            "handoff": command_handoff,
        }
        payload, code = handlers[args.command](args)
        display_payload = None
        if getattr(args, "compact", False):
            display_payload = project_compact_workspace_guidance(
                payload,
                full_view_command=_compact_full_view_command(args, payload),
                full_output_path=args.output,
            )
        emit_payload(
            payload,
            output=args.output,
            human_text=_human(payload),
            json_output=not args.human,
            display_payload=display_payload,
            compact_json=bool(display_payload),
            maximum_display_lines=int(
                dict(dict(display_payload or {}).get("workspace") or {})
                .get("output_policy", {})
                .get("maximum_display_lines", 120)
                if display_payload
                else 120
            ),
        )
        return code
    except ActorError as exc:
        command = raw[0] if raw else "open"
        if exc.code in {"workspace_bootstrap_missing", "workspace_bootstrap_refresh_required"}:
            warnings = list(dict.fromkeys([str(exc), *list(exc.problems or []), *list(exc.warnings or [])]))
            payload = with_guidance(
                {
                    "ok": True,
                    "execution_blocked": False,
                    "workflow_administration_required": True,
                    "workspace": None,
                },
                status="WORKSPACE_ASSISTANCE_UNAVAILABLE",
                reason_codes=[exc.code],
                problems=[],
                warnings=warnings,
                next_actions=list(exc.next_actions or []),
                related_records=list(exc.related_records or []),
            )
            human, output = (bool(getattr(args, "human", False)), getattr(args, "output", None))
            lines = [
                "Workspace assistance unavailable; operator-authorized work is not blocked.",
                *[f"Warning: {warning}" for warning in warnings[:5]],
            ]
            if exc.next_actions:
                action = dict(exc.next_actions[0])
                lines.append(f"Optional next: {action.get('description')}")
                if action.get("command"):
                    lines.append(f"Command: {action['command']}")
            emit_payload(
                payload,
                output=output,
                human_text="\n".join(lines) + "\n",
                json_output=not human,
            )
            return 0
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
