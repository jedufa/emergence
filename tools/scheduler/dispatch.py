"""Assignment Linker and record operations for the optional EIEL scheduler.

This module compiles existing repository truth into immutable Assignments.  It
never chooses a scientific objective, rewrites Work, completes Work, repairs a
repository, or launches a worker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from tools.actors.bootstrap import EDITING_TASK_MODES, git_repository_identity, validate_bootstrap_record
from tools.actors.common import (
    ActorError,
    advisory_lock,
    atomic_write_json,
    new_record,
    revise_record,
    read_json,
    stable_id,
    utc_now,
    require_test_fixture_context,
)
from tools.actors.context_receipt import build_context_receipt, validate_context_receipt, write_context_receipt
from tools.actors.canonical import canonical_sha256
from tools.actors.portfolio import bootstrap_store, session_store, work_store
from tools.actors.reference_integrity import actor_source_ref
from tools.actors.session_manager import (
    activate_session_for_assignment,
    bind_session_codex_thread_identity,
    invalidate_prepared_session_for_assignment,
    prepare_session_for_assignment,
    record_worker_loss,
)
from tools.actors.work_manager import eligible_assignment_candidates, scheduler_eligibility_candidate_for_work
from tools.actors.work_package import verify_work_package_binding
from tools.actors import workspace as workspace_actor
from tools.scheduler.contracts import (
    ASSIGNMENT_ALLOWED_TRANSITIONS,
    ASSIGNMENT_SCHEMA_VERSION,
    DISPATCH_SCHEMA_VERSION,
    LEASE_ALLOWED_TRANSITIONS,
    LEASE_SCHEMA_VERSION,
    SCHEDULER_LAUNCH_ACK_SCHEMA_VERSION,
    WORKER_RESULT_SCHEMA_VERSION,
    active_lease_for_assignment,
    add_seconds,
    assignment_id_for,
    assignment_store,
    assignment_writer_store,
    dispatch_id_for,
    dispatch_store,
    dispatch_writer_store,
    lease_id_for,
    lease_store,
    lease_writer_store,
    load_assignment_payload,
    load_dispatch_payload,
    load_lease_payload,
    load_worker_result_payload,
    normalize_assignment_payload,
    normalize_dispatch_payload,
    normalize_lease_payload,
    normalize_worker_result_payload,
    parse_utc,
    phase_to_role,
    records,
    result_id_for,
    result_store,
    result_writer_store,
    scheduler_root,
    verify_assignment_payload,
)

ACTOR_ASSIGNMENT = "assignmentLinker"
ACTOR_SCHEDULER = "scheduler"

PHASE_URGENCY = {"implementation": 3, "repair": 1, "review": 2}
WORK_PRIORITY = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4}


def persist_dispatch_record_fixture(record: Mapping[str, Any], *, state_root: str | Path | None, allow_update: bool = False) -> Path:
    require_test_fixture_context()
    return dispatch_store(state_root, write_scope=True).put(record, allow_update=allow_update)


def persist_assignment_record_fixture(record: Mapping[str, Any], *, state_root: str | Path | None, allow_update: bool = False) -> Path:
    require_test_fixture_context()
    return assignment_store(state_root, write_scope=True).put(record, allow_update=allow_update)


def persist_lease_record_fixture(record: Mapping[str, Any], *, state_root: str | Path | None, allow_update: bool = False) -> Path:
    require_test_fixture_context()
    return lease_store(state_root, write_scope=True).put(record, allow_update=allow_update)


def persist_worker_result_record_fixture(record: Mapping[str, Any], *, state_root: str | Path | None, allow_update: bool = False) -> Path:
    require_test_fixture_context()
    return result_store(state_root, write_scope=True).put(record, allow_update=allow_update)


def _assignment_payload(record: Mapping[str, Any], *, require_ready: bool = False) -> dict[str, Any]:
    return load_assignment_payload(record.get("payload"), require_ready=require_ready)


def _lease_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return load_lease_payload(record.get("payload"))


def _dispatch_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return load_dispatch_payload(record.get("payload"))


def _require_scheduler_transition(
    record: Mapping[str, Any],
    target_status: str,
    *,
    machine: str,
    allowed: Mapping[str, set[str]],
) -> None:
    current = str(record.get("status") or "")
    if target_status not in allowed.get(current, set()):
        raise ActorError(
            f"invalid {machine} transition {current or '<missing>'} -> {target_status}",
            code=f"{machine}_transition_invalid",
            problems=[
                f"Allowed transitions from {current or '<missing>'}: "
                + ", ".join(sorted(allowed.get(current, set())))
            ],
            related_records=[str(record.get("record_id") or "")],
        )


def _revise_assignment(
    record: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    status: str,
    action: str,
    state_root: str | Path | None,
    generated_at: str | None,
    note: str | None = None,
) -> dict[str, Any]:
    """Single executable gateway for durable Assignment lifecycle writes."""

    _require_scheduler_transition(
        record,
        status,
        machine="assignment",
        allowed=ASSIGNMENT_ALLOWED_TRANSITIONS,
    )
    normalized = normalize_assignment_payload(payload)
    revised = revise_record(
        record,
        payload=normalized,
        status=status,
        action=action,
        generated_at=utc_now(generated_at),
        note=note,
    )
    assignment_store(state_root, write_scope=True).put(revised, allow_update=True)
    return revised


def _revise_lease(
    record: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    status: str,
    action: str,
    state_root: str | Path | None,
    generated_at: str | None,
    note: str | None = None,
) -> dict[str, Any]:
    """Single executable gateway for durable Lease lifecycle writes."""

    _require_scheduler_transition(
        record,
        status,
        machine="lease",
        allowed=LEASE_ALLOWED_TRANSITIONS,
    )
    normalized = normalize_lease_payload(payload)
    revised = revise_record(
        record,
        payload=normalized,
        status=status,
        action=action,
        generated_at=utc_now(generated_at),
        note=note,
    )
    lease_store(state_root, write_scope=True).put(revised, allow_update=True)
    return revised



def _assignment_lock_path(state_root: str | Path | None, assignment_id: str) -> Path:
    return scheduler_root(state_root) / "lifecycle-locks" / f"{str(assignment_id).upper()}.lock"


def _dispatch_lock_path(state_root: str | Path | None, dispatch_id: str) -> Path:
    return scheduler_root(state_root) / "dispatch-locks" / f"{str(dispatch_id).upper()}.lock"


def _require_clean_repository(repository: Mapping[str, Any], *, stage: str) -> None:
    if repository.get("dirty"):
        raise ActorError(
            f"unattended Scheduler requires a clean worktree at {stage}",
            code="scheduler_worktree_dirty",
            problems=[str(row) for row in repository.get("dirty_entries") or []],
        )


def _record_at_revision(store_obj, record_id: str, revision: int) -> dict[str, Any]:
    for row in store_obj.history(record_id):
        if int(row.get("revision") or 0) == int(revision):
            return read_json(Path(str(row["path"])), name=f"{record_id} revision {revision}")
    raise ActorError(f"historical record revision is missing: {record_id} r{revision}", code="scheduler_history_missing")


def _verify_ready_material_immutable(assignment: Mapping[str, Any], *, state_root: str | Path | None) -> None:
    if assignment.get("status") == "PREPARING":
        return
    store_obj = assignment_store(state_root)
    ready_snapshot = None
    for row in store_obj.history(str(assignment.get("record_id"))):
        record = read_json(Path(str(row["path"])), name="Assignment history")
        if record.get("status") == "READY":
            ready_snapshot = record
            break
    if ready_snapshot is None:
        raise ActorError("Assignment has no immutable READY snapshot", code="assignment_ready_snapshot_missing")
    current_payload = _assignment_payload(assignment)
    ready_payload = load_assignment_payload(ready_snapshot.get("payload"), require_ready=True)
    if current_payload.get("material") != ready_payload.get("material") or current_payload.get("assignment_material_sha256") != ready_payload.get("assignment_material_sha256"):
        raise ActorError("READY Assignment material was mutated", code="assignment_immutability_violation")


def _validate_dispatch_binding(material: Mapping[str, Any], *, state_root: str | Path | None, require_enabled: bool = True) -> dict[str, Any]:
    frozen = dict(material.get("dispatch_request") or {})
    dispatch = dispatch_store(state_root).get(str(frozen.get("record_id") or ""))
    payload = load_dispatch_payload(dispatch.get("payload"))
    if require_enabled and (dispatch.get("status") != "ENABLED" or not payload.get("enabled", True)):
        raise ActorError("Dispatch Request was disabled", code="dispatch_not_enabled")
    if int(dispatch.get("revision") or 0) != int(frozen.get("revision") or 0) or dispatch.get("record_sha256") != frozen.get("record_sha256"):
        raise ActorError("Dispatch Request changed after Assignment compilation", code="assignment_stale")
    return dispatch


def _require_monotonic_event_time(
    timestamp: str,
    *references: str | None,
    label: str,
) -> None:
    """Reject caller-supplied scheduler time that moves authority backwards."""

    observed = parse_utc(timestamp)
    for reference in references:
        if reference and observed < parse_utc(str(reference)):
            raise ActorError(
                f"{label} timestamp precedes durable scheduler state",
                code="scheduler_time_non_monotonic",
                problems=[f"event={timestamp}", f"durable={reference}"],
            )


def _validate_live_assignment_inputs(
    assignment: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    allow_active_session: bool = False,
) -> dict[str, Any]:
    payload = load_assignment_payload(assignment.get("payload"), require_ready=True)
    _verify_ready_material_immutable(assignment, state_root=state_root)
    material = dict(payload.get("material") or {})
    dispatch = _validate_dispatch_binding(material, state_root=state_root)

    repo = dict(material.get("repository") or {})
    live = git_repository_identity(repo.get("repo_root"))
    if live.get("head") != repo.get("expected_head") or live.get("branch") != repo.get("expected_branch"):
        raise ActorError("Assignment repository expectations are stale", code="assignment_stale")
    _require_clean_repository(live, stage="Assignment validation")

    bootstrap_info = dict(material.get("bootstrap_expectations") or {})
    bootstrap = bootstrap_store(state_root).get(str(bootstrap_info.get("record_id") or ""))
    if bootstrap.get("record_sha256") != bootstrap_info.get("record_sha256"):
        raise ActorError("Assignment Bootstrap record changed", code="assignment_stale")
    bootstrap_errors = validate_bootstrap_record(bootstrap, expected_repo_root=live["root"], expected_head=live["head"], max_age_hours=None)
    if bootstrap_errors:
        raise ActorError("Assignment Bootstrap is stale", code="assignment_stale", problems=bootstrap_errors)

    work_info = dict(material.get("work") or {})
    work = work_store(state_root).get(str(work_info.get("record_id") or ""))
    if work.get("record_sha256") != work_info.get("record_sha256"):
        raise ActorError("Assignment Work changed", code="assignment_stale")
    binding = dict(dict(work.get("payload") or {}).get("work_package_binding") or {})
    if binding.get("sha256") != dict(work_info.get("work_package_binding") or {}).get("sha256"):
        raise ActorError("Assignment Work Package binding changed", code="assignment_stale")
    package_check = verify_work_package_binding(
        binding,
        repo_root=live["root"],
        expected_work_id=str(work.get("record_id") or ""),
        state_root=state_root,
    )
    if not package_check.get("valid"):
        raise ActorError("Assignment Work Package file changed", code="assignment_stale", problems=list(package_check.get("errors") or []))

    session_info = dict(material.get("session") or {})
    session_id = str(session_info.get("record_id") or "")
    session = session_store(state_root).get(session_id)
    phase = str(dict(material.get("phase") or {}).get("kind") or "")
    expected_role = phase_to_role(phase)
    actual_role = str(dict(session.get("payload") or {}).get("session_role") or "")
    if actual_role != expected_role:
        raise ActorError(
            "Assignment Session authority role does not match its frozen phase",
            code="assignment_session_role_mismatch",
            problems=[f"phase={phase}", f"expected={expected_role}", f"observed={actual_role or '<empty>'}"],
            related_records=[session_id],
        )
    session_snapshot = session
    if session.get("record_sha256") != session_info.get("record_sha256"):
        sp = dict(session.get("payload") or {})
        exact_activation = allow_active_session and (
            session.get("status") == "ACTIVE"
            and sp.get("scheduler_assignment_id") == assignment.get("record_id")
            and sp.get("scheduler_assignment_digest") == payload.get("assignment_material_sha256")
        )
        if not exact_activation:
            raise ActorError("Assignment Session changed", code="assignment_stale")
        session_snapshot = _record_at_revision(session_store(state_root), session_id, int(session_info.get("revision") or 0))

    continuation = dict(material.get("continuation") or {})
    receipt_path = Path(str(continuation.get("context_receipt_path") or ""))
    if not receipt_path.is_file():
        raise ActorError("Assignment Context Receipt is missing", code="assignment_stale")
    receipt = read_json(receipt_path, name="Context Receipt")
    if receipt.get("receipt_sha256") != continuation.get("context_receipt_sha256"):
        raise ActorError("Assignment Context Receipt changed", code="assignment_stale")
    receipt_check = validate_context_receipt(
        receipt,
        work_record=work,
        session_record=session_snapshot,
        repo_root=live["root"],
        state_root=state_root,
    )
    if not receipt_check.get("valid"):
        receipt_errors = [str(value) for value in receipt_check.get("errors") or []]
        review_stale = any("authoritative review" in value.lower() for value in receipt_errors)
        raise ActorError(
            "Assignment authoritative review binding is stale"
            if review_stale
            else "Assignment Context Receipt is stale",
            code="assignment_review_binding_stale" if review_stale else "assignment_stale",
            problems=receipt_errors,
            warnings=list(receipt_check.get("warnings") or []),
        )
    return {"payload": payload, "material": material, "dispatch": dispatch, "repository": live, "bootstrap": bootstrap, "work": work, "session": session, "receipt": receipt}


def _bootstrap_authorizes_assignment_role(record: Mapping[str, Any], required_role: str) -> bool:
    """Return whether one live Bootstrap is compatible with an Assignment role.

    Repository/HEAD freshness alone is insufficient: a fresh Review Assignment
    must not inherit an implementation-shaped, write-authorized Bootstrap merely
    because it was created later.  Conversely, an editing Assignment must retain
    explicit edit authorization and a bounded write scope.  This check selects
    a compatible receipt; Session Manager remains the final authority gate.
    """

    task = dict(dict(record.get("payload") or {}).get("task") or {})
    mode = str(task.get("mode") or "").strip().lower()
    role = str(required_role or "").strip().upper()
    if role == "REVIEW_AUTHORITY":
        return mode in {"review", "diagnosis"} and not bool(task.get("edit_authorized"))
    if role == "IMPLEMENTATION_AUTHORITY":
        return (
            mode in EDITING_TASK_MODES
            and bool(task.get("edit_authorized"))
            and bool(list(task.get("write_scope") or []))
        )
    return False


def _matching_bootstrap(
    *,
    state_root: str | Path | None,
    repo_identity: Mapping[str, Any],
    required_role: str,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for row in bootstrap_store(state_root).list():
        if not row.get("record_id") or row.get("status") == "INVALID":
            continue
        record = bootstrap_store(state_root).get(str(row["record_id"]))
        errors = validate_bootstrap_record(
            record,
            expected_repo_root=str(repo_identity.get("root")),
            expected_head=str(repo_identity.get("head")),
            max_age_hours=None,
        )
        if not errors and _bootstrap_authorizes_assignment_role(record, required_role):
            candidates.append(record)
    if not candidates:
        mode = "review" if str(required_role).upper() == "REVIEW_AUTHORITY" else "implementation"
        raise ActorError(
            "No current role-compatible Bootstrap receipt matches the Dispatch Request repository and HEAD",
            code="dispatch_bootstrap_missing",
            problems=[f"required_role={required_role}"],
            next_actions=[{
                "action": "bootstrap",
                "description": "Run a fresh role-compatible verify-only Bootstrap before requesting an Assignment.",
                "command": (
                    f"python -m tools.eiel bootstrap inspect --repo-root "
                    f"{json.dumps(str(repo_identity.get('root')))} --task-mode {mode} ..."
                ),
            }],
        )
    return sorted(candidates, key=lambda row: (str(row.get("updated_at_utc") or ""), str(row.get("record_id") or "")), reverse=True)[0]


def _bootstrap_for_assignment_candidate(
    *,
    state_root: str | Path | None,
    repo_identity: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Use the exact resumed Session Bootstrap instead of a merely newer one.

    A resumable Session and its Context Receipt are causally bound to the
    Session's current Bootstrap (or an explicitly retained historical one).
    Selecting the newest repository-wide Bootstrap during compilation can make
    Workspace reject an otherwise valid exact resume, and worse, can silently
    relabel which authority receipt authorized that continuation.  Fresh
    Sessions still use the newest matching repository receipt.
    """

    session_id = str(candidate.get("existing_session_id") or "")
    if not session_id:
        return _matching_bootstrap(
            state_root=state_root,
            repo_identity=repo_identity,
            required_role=phase_to_role(str(candidate.get("authority_phase") or "")),
        )

    session = session_store(state_root).get(session_id)
    session_payload = dict(session.get("payload") or {})
    bootstrap_id = str(session_payload.get("bootstrap_record_id") or "")
    if not bootstrap_id:
        raise ActorError(
            "resumed Session has no exact Bootstrap binding",
            code="assignment_session_bootstrap_missing",
            related_records=[session_id],
        )
    bootstrap = bootstrap_store(state_root).get(bootstrap_id)
    errors = validate_bootstrap_record(
        bootstrap,
        expected_repo_root=str(repo_identity.get("root") or ""),
        expected_head=str(repo_identity.get("head") or ""),
        max_age_hours=None,
    )
    if errors:
        raise ActorError(
            "resumed Session Bootstrap is stale for the current repository",
            code="assignment_session_bootstrap_stale",
            problems=errors,
            related_records=[session_id, bootstrap_id],
            next_actions=[{
                "action": "attach_fresh_bootstrap",
                "description": "Attach a fresh verify-only Bootstrap to the exact Session before compiling another Assignment.",
                "command": (
                    f"python -m tools.eiel sessionManager attach-bootstrap {session_id} "
                    f"--bootstrap-record <BST-ID> --repo-root {json.dumps(str(repo_identity.get('root') or ''))}"
                ),
            }],
        )
    return bootstrap


def _workspace_projection(
    *,
    state_root: str | Path | None,
    repo_root: str,
    bootstrap_id: str,
    work_id: str,
    session_id: str,
) -> dict[str, Any]:
    args = SimpleNamespace(
        state_root=str(state_root) if state_root is not None else None,
        repo_root=repo_root,
        generated_at=None,
        bootstrap_record=bootstrap_id,
        authority_domain=None,
        max_bootstrap_age_hours=None,
        allow_stale_bootstrap=True,
        output=None,
        human=False,
        work_id=work_id,
        session_id=session_id,
    )
    payload, code = workspace_actor.command_open(args)
    if code != 0:
        raise ActorError("Workspace refused the exact Work/Session projection", code="assignment_workspace_blocked")
    workspace = dict(payload.get("workspace") or payload)
    actions = list(payload.get("next_actions") or workspace.get("next_actions") or [])
    workspace["_guidance_next_actions"] = actions
    return workspace


def authorize_dispatch(
    raw: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    generated_at: str | None = None,
    dispatch_id: str | None = None,
) -> dict[str, Any]:
    timestamp = utc_now(generated_at)
    payload = normalize_dispatch_payload(raw)
    work = work_store(state_root).get(payload["work_id"])
    repo_identity = git_repository_identity(dict(payload["repository"]).get("repo_root"))
    _require_clean_repository(repo_identity, stage="Dispatch authorization")
    expected_branch = dict(payload["repository"]).get("expected_branch")
    if expected_branch and expected_branch != repo_identity.get("branch"):
        raise ActorError(
            "Dispatch Request expected branch does not match repository",
            code="dispatch_repository_mismatch",
            problems=[f"Expected {expected_branch}; observed {repo_identity.get('branch')}"]
        )
    identity = {
        "work_id": payload["work_id"],
        "package_sha256": dict(dict(work.get("payload") or {}).get("work_package_binding") or {}).get("sha256"),
        "repo_root": repo_identity.get("root"),
        "created_at": timestamp,
        "dispatch_payload": payload,
    }
    record_id = dispatch_id or dispatch_id_for(identity)
    record = new_record(
        actor=ACTOR_ASSIGNMENT,
        record_type="dispatch_request",
        record_id=record_id,
        status="ENABLED" if payload["enabled"] else "DISABLED",
        payload=payload,
        generated_at=timestamp,
        source_refs=[actor_source_ref("work", work)],
        action="authorize_dispatch",
        note=payload.get("note"),
    )
    dispatch_store(state_root, write_scope=True).put(record)
    return record

def set_dispatch_enabled(dispatch_id: str, *, enabled: bool, state_root: str | Path | None, generated_at: str | None = None, note: str | None = None) -> dict[str, Any]:
    # Launch ACK and revocation share the Dispatch lock.  This closes the
    # otherwise tiny but real interval between final live-input validation and
    # releasing the worker wrapper to exec its target.
    with advisory_lock(_dispatch_lock_path(state_root, dispatch_id)):
        store = dispatch_store(state_root, write_scope=True)
        record = store.get(dispatch_id)
        payload = _dispatch_payload(record)
        payload["enabled"] = bool(enabled)
        timestamp = utc_now(generated_at)
        revised = revise_record(
            record,
            payload=payload,
            status="ENABLED" if enabled else "DISABLED",
            action="enable_dispatch" if enabled else "disable_dispatch",
            generated_at=timestamp,
            note=note,
        )
        store.put(revised, allow_update=True)
        if not enabled:
            for assignment in records(assignment_store(state_root)):
                material = dict(_assignment_payload(assignment).get("material") or {})
                preparation = dict(_assignment_payload(assignment).get("preparation") or {})
                bound_dispatch = str(dict(material.get("dispatch_request") or {}).get("record_id") or preparation.get("dispatch_id") or "")
                if bound_dispatch != dispatch_id:
                    continue
                status = str(assignment.get("status") or "")
                operational = dict(_assignment_payload(assignment).get("operational") or {})
                pre_ack_running = status == "RUNNING" and not operational.get("launch_acknowledged_at_utc")
                if status not in {"PREPARING", "READY", "CLAIMED"} and not pre_ack_running:
                    continue
                supersede_assignment_for_manual_takeover(
                    str(assignment.get("record_id")),
                    state_root=state_root,
                    reason="DISPATCH_DISABLED",
                    generated_at=timestamp,
                    allow_unacknowledged_running=pre_ack_running,
                )
        return revised

def dispatch_candidates(*, state_root: str | Path | None) -> list[dict[str, Any]]:
    work_candidates = {row["work_id"]: row for row in eligible_assignment_candidates(state_root=state_root)}
    rows: list[dict[str, Any]] = []
    for dispatch in records(dispatch_store(state_root)):
        dp = load_dispatch_payload(dispatch.get("payload"))
        candidate = dict(work_candidates.get(str(dp.get("work_id") or "")) or {
            "work_id": dp.get("work_id"), "eligible": False, "reason_codes": ["work_missing"],
            "priority": "P4", "authority_phase": None, "created_at_utc": dispatch.get("created_at_utc"),
        })
        reasons = list(candidate.get("reason_codes") or [])
        eligible = bool(candidate.get("eligible"))
        if dispatch.get("status") != "ENABLED" or not dp.get("enabled", True):
            eligible = False
            reasons.append("dispatch_disabled")
        phase = candidate.get("authority_phase")
        allowed = set(dp.get("allowed_authority_phases") or [])
        if phase and phase not in allowed:
            eligible = False
            reasons.append("phase_not_authorized")
        policy = dict(dp.get("automation_policy") or {})
        required_flag = {
            "review": "allow_independent_review",
            "repair": "allow_repair",
        }.get(str(phase), "allow_resume" if candidate.get("existing_session_id") else "allow_fresh_session")
        if not policy.get(required_flag, False):
            eligible = False
            reasons.append(f"policy_{required_flag}_false")
        rows.append({
            "dispatch_id": dispatch.get("record_id"),
            "dispatch_revision": dispatch.get("revision"),
            "dispatch_sha256": dispatch.get("record_sha256"),
            "work_id": candidate.get("work_id"),
            "work_priority": candidate.get("priority"),
            "authority_phase": phase,
            # Queue eligibility is a cheap, read-only screening result.  It is
            # intentionally not a claim that live compilation will pass the
            # repository, Bootstrap, receipt, package, or ownership gates.
            "eligible": eligible,
            "queue_eligible": eligible,
            "eligibility_scope": "queue_only",
            "reason_codes": list(dict.fromkeys(reasons)),
            "queue_priority": int(dict(dp.get("queue") or {}).get("priority", 100)),
            "created_at_utc": candidate.get("created_at_utc") or dispatch.get("created_at_utc"),
            "existing_session_id": candidate.get("existing_session_id"),
            "review_target_session_id": candidate.get("review_target_session_id"),
        })
    rows.sort(key=lambda row: (
        not row["eligible"],
        int(row.get("queue_priority", 100)),
        WORK_PRIORITY.get(str(row.get("work_priority")), 99),
        PHASE_URGENCY.get(str(row.get("authority_phase")), 99),
        str(row.get("created_at_utc") or ""),
        str(row.get("work_id") or ""),
        str(row.get("dispatch_id") or ""),
    ))
    return rows


def _link_intent_payload(
    *,
    dispatch: Mapping[str, Any],
    work: Mapping[str, Any],
    candidate: Mapping[str, Any],
    repository: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the complete immutable human-authorized compilation intent."""

    work_payload = dict(work.get("payload") or {})
    dispatch_payload = _dispatch_payload(dispatch)
    return {
        "schema_version": "eiel_assignment_link_intent_v1",
        "dispatch": {
            "record_id": dispatch.get("record_id"),
            "revision": dispatch.get("revision"),
            "record_sha256": dispatch.get("record_sha256"),
            "payload": dispatch_payload,
        },
        "work": {
            "record_id": work.get("record_id"),
            "revision": work.get("revision"),
            "record_sha256": work.get("record_sha256"),
            "work_package_sha256": dict(work_payload.get("work_package_binding") or {}).get("sha256"),
        },
        "repository": {
            "repo_root": repository.get("root"),
            "working_directory": repository.get("root"),
            "branch": repository.get("branch"),
            "head": repository.get("head"),
        },
        "authority_phase": candidate.get("authority_phase"),
        "existing_session_id": candidate.get("existing_session_id"),
        "review_target_session_id": candidate.get("review_target_session_id"),
        "bootstrap": {
            "record_id": bootstrap.get("record_id"),
            "revision": bootstrap.get("revision"),
            "record_sha256": bootstrap.get("record_sha256"),
        },
    }


def _execution_uniqueness_key(*, work: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[str, str, str, str]:
    binding = dict(dict(work.get("payload") or {}).get("work_package_binding") or {})
    return (
        str(work.get("record_id") or ""),
        str(binding.get("sha256") or ""),
        str(candidate.get("authority_phase") or ""),
        str(candidate.get("review_target_session_id") or ""),
    )


def _preparing_payload(*, dispatch: Mapping[str, Any], work: Mapping[str, Any], candidate: Mapping[str, Any], repository: Mapping[str, Any], bootstrap: Mapping[str, Any], session_id: str, timestamp: str, link_intent_sha256: str) -> dict[str, Any]:
    return normalize_assignment_payload({
        "material": None,
        "assignment_material_sha256": None,
        "preparation": {
            "phase": "PREPARING",
            "dispatch_id": dispatch.get("record_id"),
            "dispatch_revision": dispatch.get("revision"),
            "dispatch_sha256": dispatch.get("record_sha256"),
            "work_id": work.get("record_id"),
            "work_revision": work.get("revision"),
            "work_sha256": work.get("record_sha256"),
            "session_id": session_id,
            "authority_phase": candidate.get("authority_phase"),
            "existing_session_id": candidate.get("existing_session_id"),
            "review_target_session_id": candidate.get("review_target_session_id"),
            "bootstrap_record_id": bootstrap.get("record_id"),
            "repository": dict(repository),
            "link_intent_sha256": link_intent_sha256,
            "started_at_utc": timestamp,
        },
        "operational": {"current_lease_id": None, "result_id": None, "claim_count": 0},
    })


def _exact_open_assignment_for_dispatch(
    *,
    state_root: str | Path | None,
    dispatch: Mapping[str, Any],
    work: Mapping[str, Any],
    candidate: Mapping[str, Any],
    repository: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return an exact open Assignment for one unchanged Dispatch revision.

    A Session prepared by the Assignment becomes visible to Work Manager on the
    second compilation attempt, so recomputing the pre-creation link intent
    would not be stable.  Exact same-Dispatch idempotence is instead proven from
    the frozen Dispatch revision/digest and the PREPARING/READY repository truth.
    Distinct Dispatch authorizations still use the complete link-intent digest.
    """

    for record in records(assignment_store(state_root)):
        if record.get("status") not in {"PREPARING", "CLEANUP_REQUIRED", "READY", "CLAIMED", "RUNNING", "RESULT_SUBMITTED"}:
            continue
        payload = _assignment_payload(record)
        material = dict(payload.get("material") or {})
        prep = dict(payload.get("preparation") or {})
        frozen = dict(material.get("dispatch_request") or {})
        prior_id = str(frozen.get("record_id") or prep.get("dispatch_id") or "")
        if prior_id != str(dispatch.get("record_id") or ""):
            continue
        prior_revision = int(frozen.get("revision") or prep.get("dispatch_revision") or 0)
        prior_sha = str(frozen.get("record_sha256") or prep.get("dispatch_sha256") or "")
        mismatches: list[str] = []
        if prior_revision != int(dispatch.get("revision") or 0) or prior_sha != str(dispatch.get("record_sha256") or ""):
            mismatches.append("Dispatch revision/digest")
        prior_work = dict(material.get("work") or {})
        if str(prior_work.get("record_id") or prep.get("work_id") or "") != str(work.get("record_id") or ""):
            mismatches.append("Work")
        prior_phase = str(dict(material.get("phase") or {}).get("kind") or prep.get("authority_phase") or "")
        if prior_phase != str(candidate.get("authority_phase") or ""):
            mismatches.append("authority phase")
        prior_review = str(prep.get("review_target_session_id") or "")
        current_review = str(candidate.get("review_target_session_id") or "")
        if prior_review != current_review:
            mismatches.append("review target")
        prior_session = str(dict(material.get("session") or {}).get("record_id") or prep.get("session_id") or "")
        current_session = str(candidate.get("existing_session_id") or "")
        if current_session and current_session != prior_session:
            mismatches.append("Session target")
        prior_repo = dict(material.get("repository") or prep.get("repository") or {})
        prior_root = str(prior_repo.get("repo_root") or prior_repo.get("root") or "")
        if Path(prior_root).resolve() != Path(str(repository.get("root") or "")).resolve():
            mismatches.append("repository root")
        prior_branch = str(prior_repo.get("expected_branch") or prior_repo.get("branch") or "")
        if prior_branch != str(repository.get("branch") or ""):
            mismatches.append("branch")
        prior_head = str(prior_repo.get("expected_head") or prior_repo.get("head") or "")
        if prior_head != str(repository.get("head") or ""):
            mismatches.append("HEAD")
        prior_bootstrap = str(dict(material.get("bootstrap_expectations") or {}).get("record_id") or prep.get("bootstrap_record_id") or "")
        if prior_bootstrap != str(bootstrap.get("record_id") or ""):
            mismatches.append("Bootstrap")
        if mismatches:
            repository_drift = {"repository root", "branch", "HEAD", "Bootstrap"}
            drift_only = all(item in repository_drift for item in mismatches)
            raise ActorError(
                "open Assignment no longer matches live repository truth" if drift_only else
                "open Assignment binds an older or different revision of this Dispatch Request",
                code="assignment_stale" if drift_only else "assignment_authorization_conflict",
                problems=mismatches,
                related_records=[str(record.get("record_id") or "")],
            )
        return record
    return None


def _equivalent_open_assignment(*, state_root: str | Path | None, work: Mapping[str, Any], candidate: Mapping[str, Any], link_intent_sha256: str) -> dict[str, Any] | None:
    requested_key = _execution_uniqueness_key(work=work, candidate=candidate)
    for record in records(assignment_store(state_root)):
        if record.get("status") not in {"PREPARING", "CLEANUP_REQUIRED", "READY", "CLAIMED", "RUNNING", "RESULT_SUBMITTED"}:
            continue
        payload = _assignment_payload(record)
        material = dict(payload.get("material") or {})
        prep = dict(payload.get("preparation") or {})
        prior_work = dict(material.get("work") or {})
        prior_key = (
            str(prep.get("work_id") or prior_work.get("record_id") or ""),
            str(dict(prior_work.get("work_package_binding") or {}).get("sha256") or ""),
            str(prep.get("authority_phase") or dict(material.get("phase") or {}).get("kind") or ""),
            str(prep.get("review_target_session_id") or ""),
        )
        # PREPARING records may not yet have material; use the current Work
        # package only for that exact Work when the preparation record predates
        # V1.10.3.
        if not prior_key[1] and prior_key[0] == requested_key[0]:
            prior_key = (prior_key[0], requested_key[1], prior_key[2], prior_key[3])
        if prior_key != requested_key:
            continue
        prior_digest = str(material.get("link_intent_sha256") or prep.get("link_intent_sha256") or "")
        if prior_digest == link_intent_sha256:
            return record
        raise ActorError(
            "another open Assignment binds a different authorization for the same execution phase",
            code="assignment_authorization_conflict",
            related_records=[str(record.get("record_id") or "")],
        )
    return None

def compile_assignment(dispatch_id: str, *, state_root: str | Path | None, generated_at: str | None = None) -> dict[str, Any]:
    timestamp = utc_now(generated_at)
    root = scheduler_root(state_root)
    with advisory_lock(root / "assignment_compiler.lock"):
        dispatch = dispatch_store(state_root).get(dispatch_id)
        if dispatch.get("status") != "ENABLED":
            raise ActorError("Dispatch Request is not enabled", code="dispatch_not_enabled")
        dp = load_dispatch_payload(dispatch.get("payload"))
        work = work_store(state_root).get(str(dp.get("work_id") or ""))
        candidate = scheduler_eligibility_candidate_for_work(work, state_root=state_root)
        if not candidate["eligible"]:
            raise ActorError("Work is not eligible for Assignment compilation", code="work_not_assignment_eligible", problems=candidate["reason_codes"])
        phase = str(candidate.get("authority_phase") or "")
        if phase not in {"implementation", "review", "repair"}:
            raise ActorError("higher authority phases are manual-only in Scheduler V1", code="assignment_phase_not_authorized")
        if phase not in set(dp.get("allowed_authority_phases") or []):
            raise ActorError("Authority phase is not authorized by Dispatch Request", code="assignment_phase_not_authorized")
        policy = dict(dp.get("automation_policy") or {})
        policy_key = {"review": "allow_independent_review", "repair": "allow_repair"}.get(phase, "allow_resume" if candidate.get("existing_session_id") else "allow_fresh_session")
        if not policy.get(policy_key, False):
            raise ActorError("Dispatch Request policy does not authorize this phase", code="assignment_policy_not_authorized")
        repository = git_repository_identity(dict(dp.get("repository") or {}).get("repo_root"))
        _require_clean_repository(repository, stage="Assignment compilation")
        expected_branch = dict(dp.get("repository") or {}).get("expected_branch")
        if expected_branch and expected_branch != repository.get("branch"):
            raise ActorError("Dispatch Request branch does not match repository", code="assignment_repository_mismatch")
        binding = dict(dict(work.get("payload") or {}).get("work_package_binding") or {})
        package_check = verify_work_package_binding(
            binding,
            repo_root=repository["root"],
            expected_work_id=str(work.get("record_id") or ""),
            state_root=state_root,
        )
        if not package_check.get("valid"):
            raise ActorError("Work Package binding is stale", code="assignment_stale", problems=list(package_check.get("errors") or []))
        bootstrap = _bootstrap_for_assignment_candidate(
            state_root=state_root,
            repo_identity=repository,
            candidate=candidate,
        )
        exact_prior = _exact_open_assignment_for_dispatch(
            state_root=state_root, dispatch=dispatch, work=work,
            candidate=candidate, repository=repository, bootstrap=bootstrap,
        )
        if exact_prior:
            if exact_prior.get("status") == "CLEANUP_REQUIRED":
                raise ActorError(
                    "prior Assignment requires prepared-Session cleanup before recompilation",
                    code="assignment_cleanup_required",
                    related_records=[str(exact_prior.get("record_id") or "")],
                )
            if exact_prior.get("status") == "PREPARING":
                return exact_prior
            # Same Dispatch identity is not enough.  The frozen Assignment must
            # still match live Work, Bootstrap, repository and Context Receipt
            # truth before idempotent reuse is allowed.
            _validate_live_assignment_inputs(
                exact_prior, state_root=state_root,
                allow_active_session=exact_prior.get("status") in {"CLAIMED", "RUNNING", "RESULT_SUBMITTED"},
            )
            return exact_prior
        link_intent = _link_intent_payload(
            dispatch=dispatch, work=work, candidate=candidate,
            repository=repository, bootstrap=bootstrap,
        )
        link_intent_sha256 = canonical_sha256(link_intent)
        prior = _equivalent_open_assignment(
            state_root=state_root, work=work, candidate=candidate,
            link_intent_sha256=link_intent_sha256,
        )
        if prior:
            if prior.get("status") == "CLEANUP_REQUIRED":
                raise ActorError(
                    "prior Assignment requires prepared-Session cleanup before recompilation",
                    code="assignment_cleanup_required",
                    related_records=[str(prior.get("record_id") or "")],
                )
            if prior.get("status") != "PREPARING":
                return prior
        seed = {"link_intent_sha256": link_intent_sha256}
        store = assignment_store(state_root, write_scope=True)
        attempt = 1
        assignment_id = assignment_id_for(seed)
        while store.exists(assignment_id):
            existing_record = store.get(assignment_id)
            if existing_record.get("status") in {"PREPARING", "CLEANUP_REQUIRED", "READY", "CLAIMED", "RUNNING", "RESULT_SUBMITTED"}:
                break
            attempt += 1
            assignment_id = assignment_id_for({**seed, "attempt": attempt})
        preparing = prior
        preparing_payload = _assignment_payload(preparing) if preparing is not None else {}
        session_id = str(dict(preparing_payload.get("preparation") or {}).get("session_id") or candidate.get("existing_session_id") or stable_id("SES", {"assignment_id": assignment_id}, length=16))
        created_session = False
        if preparing is None:
            preparing = new_record(
                actor=ACTOR_ASSIGNMENT,
                record_type="assignment",
                record_id=assignment_id,
                status="PREPARING",
                payload=_preparing_payload(dispatch=dispatch, work=work, candidate=candidate, repository=repository, bootstrap=bootstrap, session_id=session_id, timestamp=timestamp, link_intent_sha256=link_intent_sha256),
                generated_at=timestamp,
                source_refs=[
                    actor_source_ref("dispatch_request", dispatch),
                    actor_source_ref("work", work),
                    actor_source_ref("bootstrap", bootstrap),
                ],
                action="prepare_assignment",
            )
            store.put(preparing)
        try:
            if candidate.get("existing_session_id") or session_store(state_root).exists(session_id):
                session = session_store(state_root).get(session_id)
            else:
                session = prepare_session_for_assignment(
                    state_root=state_root,
                    work_id=str(work.get("record_id")),
                    bootstrap_record_id=str(bootstrap.get("record_id")),
                    repo_root=str(repository.get("root")),
                    assignment_id=assignment_id,
                    session_id=session_id,
                    role=phase_to_role(phase),
                    worker_profile=dict(dp.get("worker_profile") or {}),
                    review_target_session=candidate.get("review_target_session_id"),
                    generated_at=timestamp,
                )
                created_session = True
            session_role = str(dict(session.get("payload") or {}).get("session_role") or "")
            required_role = phase_to_role(phase)
            if session_role != required_role:
                raise ActorError(
                    "existing Session authority role does not match Assignment phase",
                    code="assignment_session_role_mismatch",
                    problems=[f"phase={phase}", f"expected={required_role}", f"observed={session_role or '<empty>'}"],
                    related_records=[str(session.get("record_id") or "")],
                )
            workspace = _workspace_projection(
                state_root=state_root,
                repo_root=str(repository.get("root")),
                bootstrap_id=str(bootstrap.get("record_id")),
                work_id=str(work.get("record_id")),
                session_id=session_id,
            )
            receipt = build_context_receipt(
                work_record=work,
                session_record=session,
                bootstrap_record=bootstrap,
                repo_root=str(repository.get("root")),
                state_root=state_root,
            )
            receipt_path = write_context_receipt(receipt, state_root=state_root)
            actions = list(workspace.get("_guidance_next_actions") or [])
            session_payload = dict(session.get("payload") or {})
            if candidate.get("existing_session_id"):
                primary = {
                    "action": "resume_selected_session",
                    "description": str(
                        session_payload.get("next_action")
                        or "Resume the exact bound Session."
                    ),
                    "command": session_payload.get("next_command"),
                    "target_session_id": session.get("record_id"),
                    "required_authority_role": phase_to_role(phase),
                    "authority_phase": phase,
                }
            else:
                primary = dict(actions[0]) if actions else {
                    "action": "execute_assignment",
                    "description": str(
                        session_payload.get("next_action")
                        or "Execute the bounded Work Package."
                    ),
                    "required_authority_role": phase_to_role(phase),
                    "authority_phase": phase,
                }
            primary = {key: value for key, value in primary.items() if value is not None}
            primary["action_sha256"] = canonical_sha256(primary)
            material = {
                "schema_version": ASSIGNMENT_SCHEMA_VERSION,
                "created_at_utc": timestamp,
                "link_intent_sha256": link_intent_sha256,
                "dispatch_request": {"record_id": dispatch.get("record_id"), "revision": dispatch.get("revision"), "record_sha256": dispatch.get("record_sha256")},
                "work": {
                    "record_id": work.get("record_id"), "status": work.get("status"), "revision": work.get("revision"),
                    "record_sha256": work.get("record_sha256"), "derived_status": candidate.get("derived_status"),
                    "work_package_binding": binding,
                },
                "session": {
                    "record_id": session.get("record_id"), "status": session.get("status"), "revision": session.get("revision"),
                    "record_sha256": session.get("record_sha256"),
                    "action": "RESUME" if candidate.get("existing_session_id") else "ACTIVATE_PREPARED",
                },
                "phase": {"kind": phase, "authority_role": phase_to_role(phase)},
                "workspace_handoff": {
                    "status": workspace.get("status"), "initial_context": workspace.get("initial_context"),
                    "output_policy": workspace.get("output_policy"), "execution_budget": workspace.get("execution_budget"),
                    "context_health": workspace.get("context_health"), "next_safe_action": primary,
                    "record_hashes": {"work": work.get("record_sha256"), "session": session.get("record_sha256"), "bootstrap": bootstrap.get("record_sha256")},
                },
                "bootstrap_expectations": {
                    "record_id": bootstrap.get("record_id"), "record_sha256": bootstrap.get("record_sha256"),
                    "expected_repo_root": repository.get("root"), "expected_branch": repository.get("branch"),
                    "expected_head": repository.get("head"), "verify_only": True,
                },
                "worker_profile": dict(dp.get("worker_profile") or {}),
                "repository": {
                    "repo_root": repository.get("root"), "working_directory": repository.get("root"),
                    "expected_branch": repository.get("branch"), "expected_head": repository.get("head"),
                    "worktree_action": "USE_EXISTING", "worktree_mutation_allowed": False,
                },
                "continuation": {
                    "context_receipt_id": receipt.get("receipt_id"), "context_receipt_sha256": receipt.get("receipt_sha256"),
                    "context_receipt_path": str(receipt_path), "authority_phase": phase,
                    "authoritative_review_binding": receipt.get("authoritative_review_binding"),
                },
                "next_safe_action": primary,
                "lease_policy": dict(dp.get("lease_policy") or {}),
                "resource_admission_required": bool(
                    dp.get("resource_admission_required", False)
                ),
                "resource_admission": (
                    dict(dp.get("resource_admission") or {})
                    if dp.get("resource_admission_required", False)
                    else None
                ),
                "result_contract": {
                    "result_path": str(root / "results" / f"{assignment_id}.json"),
                    "schema_version": WORKER_RESULT_SCHEMA_VERSION, "worker_must_not_complete_work": True,
                    "prose_parsing_allowed": False,
                },
            }
            payload = normalize_assignment_payload({
                "material": material,
                "preparation": {**dict(_assignment_payload(preparing).get("preparation") or {}), "phase": "READY", "completed_at_utc": timestamp},
                "operational": dict(_assignment_payload(preparing).get("operational") or {}),
            })
            errors = verify_assignment_payload(payload, require_ready=True)
            if errors:
                raise ActorError("Assignment material validation failed", code="assignment_invalid", problems=errors)
            ready = _revise_assignment(
                preparing,
                payload=payload,
                status="READY",
                action="assignment_ready",
                state_root=state_root,
                generated_at=timestamp,
                note="Immutable Assignment compiled from repository truth",
            )
            return ready
        except Exception as exc:
            # A failed compiler may not pretend cleanup succeeded.  Keep a
            # visible, non-claimable state until the exact prepared Session is
            # detached or manually taken over.
            cleanup_error: Exception | None = None
            try:
                if session_store(state_root).exists(session_id):
                    invalidate_prepared_session_for_assignment(
                        session_id, state_root=state_root, assignment_id=assignment_id,
                        reason=f"ASSIGNMENT_COMPILATION_FAILED:{exc}", generated_at=timestamp,
                    )
            except Exception as cleanup_exc:  # deliberately retained as durable state
                cleanup_error = cleanup_exc
            current = store.get(assignment_id)
            payload = _assignment_payload(current)
            preparation = dict(payload.get("preparation") or {})
            if cleanup_error is not None:
                preparation.update({
                    "phase": "CLEANUP_REQUIRED",
                    "failure": str(exc),
                    "cleanup_failure": str(cleanup_error),
                    "failed_at_utc": timestamp,
                })
                payload["preparation"] = preparation
                pending = _revise_assignment(
                    current,
                    payload=payload,
                    status="CLEANUP_REQUIRED",
                    action="assignment_cleanup_required",
                    state_root=state_root,
                    generated_at=timestamp,
                    note=f"Compilation failed and prepared Session cleanup failed: {cleanup_error}",
                )
                raise ActorError(
                    "Assignment compilation failed and prepared Session cleanup is required",
                    code="assignment_cleanup_required",
                    problems=[str(exc), str(cleanup_error)],
                    related_records=[assignment_id, session_id],
                ) from exc
            preparation.update({"phase": "FAILED", "failure": str(exc), "failed_at_utc": timestamp})
            payload["preparation"] = preparation
            failed = _revise_assignment(
                current,
                payload=payload,
                status="INVALID",
                action="assignment_preparation_failed",
                state_root=state_root,
                generated_at=timestamp,
                note=str(exc),
            )
            raise

CANDIDATE_REJECTION_CODES = {
    "work_not_assignment_eligible",
    "dispatch_not_enabled",
    "assignment_policy_not_authorized",
    "assignment_phase_not_authorized",
    "assignment_repository_mismatch",
    "assignment_stale",
    "session_active_conflict",
    "assignment_authorization_conflict",
    "assignment_cleanup_required",
    "scheduler_worktree_dirty",
}


def request_assignment(
    *,
    state_root: str | Path | None,
    dispatch_id: str | None = None,
    work_id: str | None = None,
    generated_at: str | None = None,
    with_diagnostics: bool = False,
) -> dict[str, Any] | None | tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if dispatch_id:
        result = compile_assignment(dispatch_id, state_root=state_root, generated_at=generated_at)
        return (result, []) if with_diagnostics else result
    candidates = [row for row in dispatch_candidates(state_root=state_root) if row["eligible"]]
    if work_id:
        candidates = [row for row in candidates if row.get("work_id") == work_id]
    rejections: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            result = compile_assignment(str(candidate["dispatch_id"]), state_root=state_root, generated_at=generated_at)
            return (result, rejections) if with_diagnostics else result
        except ActorError as exc:
            if exc.code not in CANDIDATE_REJECTION_CODES:
                raise
            rejections.append({
                "dispatch_id": candidate.get("dispatch_id"),
                "work_id": candidate.get("work_id"),
                "code": exc.code,
                "error": str(exc),
                "related_records": list(exc.related_records or []),
            })
    return (None, rejections) if with_diagnostics else None


def recover_preparing(*, state_root: str | Path | None, generated_at: str | None = None) -> list[dict[str, Any]]:
    timestamp = utc_now(generated_at)
    recovered: list[dict[str, Any]] = []
    store = assignment_store(state_root)
    for record in records(store):
        if record.get("status") not in {"PREPARING", "CLEANUP_REQUIRED"}:
            continue
        assignment_id = str(record.get("record_id") or "")
        preparation = dict(_assignment_payload(record).get("preparation") or {})
        if record.get("status") == "CLEANUP_REQUIRED":
            session_id = str(preparation.get("session_id") or "")
            try:
                if session_id and session_store(state_root).exists(session_id):
                    invalidate_prepared_session_for_assignment(
                        session_id, state_root=state_root, assignment_id=assignment_id,
                        reason="RECOVER_ASSIGNMENT_CLEANUP", generated_at=timestamp,
                    )
                current = store.get(assignment_id)
                payload = _assignment_payload(current)
                prep = dict(payload.get("preparation") or {})
                prep.update({"phase": "FAILED", "cleanup_recovered_at_utc": timestamp})
                payload["preparation"] = prep
                invalid = _revise_assignment(
                    current,
                    payload=payload,
                    status="INVALID",
                    action="recover_assignment_cleanup",
                    state_root=state_root,
                    generated_at=timestamp,
                    note="Prepared Session cleanup completed; failed Assignment remains invalid.",
                )
                recovered.append(invalid)
            except ActorError as exc:
                current = store.get(assignment_id)
                payload = _assignment_payload(current)
                prep = dict(payload.get("preparation") or {})
                prep["cleanup_failure"] = str(exc)
                prep["cleanup_last_attempt_at_utc"] = timestamp
                payload["preparation"] = prep
                pending = _revise_assignment(
                    current,
                    payload=payload,
                    status="CLEANUP_REQUIRED",
                    action="recover_assignment_cleanup_failed",
                    state_root=state_root,
                    generated_at=timestamp,
                    note=str(exc),
                )
                recovered.append(pending)
            continue
        dispatch_id = str(preparation.get("dispatch_id") or "")
        if not dispatch_id:
            recovered.append(record)
            continue
        try:
            recovered.append(compile_assignment(dispatch_id, state_root=state_root, generated_at=timestamp))
        except ActorError as exc:
            current = store.get(assignment_id)
            if current.get("status") != "PREPARING":
                recovered.append(current)
                continue
            payload = _assignment_payload(current)
            prep = dict(payload.get("preparation") or {})
            session_id = str(prep.get("session_id") or "")
            cleanup_error: ActorError | None = None
            try:
                if session_id and session_store(state_root).exists(session_id):
                    invalidate_prepared_session_for_assignment(
                        session_id, state_root=state_root, assignment_id=assignment_id,
                        reason=f"RECOVER_PREPARING_FAILED:{exc.code}", generated_at=timestamp,
                    )
            except ActorError as cleanup_exc:
                cleanup_error = cleanup_exc
            prep.update({
                "recovery_failure_code": exc.code,
                "recovery_failure": str(exc),
                "recovery_last_attempt_at_utc": timestamp,
            })
            if cleanup_error is not None:
                prep.update({"phase": "CLEANUP_REQUIRED", "cleanup_failure": str(cleanup_error)})
                status = "CLEANUP_REQUIRED"
                action = "recover_preparing_cleanup_required"
                note = f"Recovery failed and Session cleanup remains required: {cleanup_error}"
            else:
                prep.update({"phase": "FAILED", "cleanup_recovered_at_utc": timestamp})
                status = "INVALID"
                action = "recover_preparing_failed"
                note = f"PREPARING recovery failed closed: {exc.code}"
            payload["preparation"] = prep
            revised = _revise_assignment(
                current,
                payload=payload,
                status=status,
                action=action,
                state_root=state_root,
                generated_at=timestamp,
                note=note,
            )
            recovered.append(revised)
    return recovered


def claim_assignment(assignment_id: str, *, scheduler_id: str, state_root: str | Path | None, generated_at: str | None = None, duration_seconds: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    timestamp = utc_now(generated_at)
    if duration_seconds is not None:
        raise ActorError("claim-time Lease duration override is not supported", code="lease_override_not_supported")
    with advisory_lock(_assignment_lock_path(state_root, assignment_id)):
        store = assignment_store(state_root)
        assignment = store.get(assignment_id)
        if assignment.get("status") != "READY":
            raise ActorError("Assignment is not READY", code="assignment_not_claimable")
        _require_monotonic_event_time(
            timestamp,
            str(assignment.get("updated_at_utc") or ""),
            label="Assignment claim",
        )
        checked = _validate_live_assignment_inputs(assignment, state_root=state_root)
        payload = dict(checked["payload"])
        existing = active_lease_for_assignment(assignment_id, state_root=state_root, now=timestamp)
        if existing:
            raise ActorError("Assignment already has an active Lease", code="assignment_already_claimed", related_records=[str(existing.get("record_id"))])
        policy = dict(dict(payload.get("material") or {}).get("lease_policy") or {})
        duration = int(policy.get("duration_seconds") or 3600)
        heartbeat = int(policy.get("heartbeat_interval_seconds") or 60)
        seed = {"assignment_id": assignment_id, "assignment_digest": payload.get("assignment_material_sha256"), "scheduler_id": scheduler_id, "claimed_at": timestamp, "claim": int(dict(payload.get("operational") or {}).get("claim_count", 0)) + 1}
        lease_id = lease_id_for(seed)
        lease_payload = normalize_lease_payload({
            "assignment_id": assignment_id,
            "assignment_material_sha256": payload.get("assignment_material_sha256"),
            "scheduler_id": scheduler_id,
            "claimed_at_utc": timestamp,
            "expires_at_utc": add_seconds(timestamp, duration),
            "last_heartbeat_at_utc": timestamp,
            "duration_seconds": duration,
            "heartbeat_interval_seconds": heartbeat,
            "worker": {},
        })
        lease = new_record(actor=ACTOR_SCHEDULER, record_type="assignment_lease", record_id=lease_id, status="ACTIVE", payload=lease_payload, generated_at=timestamp, source_refs=[actor_source_ref("assignment", assignment)], action="claim_assignment")
        lease_store(state_root, write_scope=True).put(lease)
        op = dict(payload.get("operational") or {})
        op.update({"current_lease_id": lease_id, "claim_count": int(op.get("claim_count", 0)) + 1})
        payload["operational"] = op
        claimed = _revise_assignment(
            assignment,
            payload=payload,
            status="CLAIMED",
            action="claim_assignment",
            state_root=state_root,
            generated_at=timestamp,
            note=f"Claimed by {scheduler_id}",
        )
        return claimed, lease

def mark_assignment_running(assignment_id: str, lease_id: str, worker_identity: Mapping[str, Any], *, state_root: str | Path | None, generated_at: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    timestamp = utc_now(generated_at)
    with advisory_lock(_assignment_lock_path(state_root, assignment_id)):
        assignments = assignment_store(state_root)
        assignment = assignments.get(assignment_id)
        _verify_ready_material_immutable(assignment, state_root=state_root)
        leases = lease_store(state_root)
        lease = leases.get(lease_id)
        if assignment.get("status") not in {"CLAIMED", "RUNNING"}:
            raise ActorError("Assignment is not claimed", code="assignment_not_running")
        lp = _lease_payload(lease)
        if lp.get("assignment_id") != assignment_id or lease.get("status") != "ACTIVE":
            raise ActorError("Lease does not own Assignment", code="lease_assignment_mismatch")
        _require_monotonic_event_time(
            timestamp,
            str(assignment.get("updated_at_utc") or ""),
            str(lp.get("last_heartbeat_at_utc") or lp.get("claimed_at_utc") or ""),
            label="Assignment worker start",
        )
        if parse_utc(str(lp.get("expires_at_utc"))) <= parse_utc(timestamp):
            raise ActorError("Lease has expired", code="lease_expired")
        bound = dict(lp.get("worker") or {})
        incoming = {key: value for key, value in dict(worker_identity).items() if value is not None}
        for key in ("kind", "worker_id", "pid", "process_start_identity", "thread_id"):
            if bound.get(key) is not None and incoming.get(key) is not None and bound.get(key) != incoming.get(key):
                raise ActorError(f"Lease worker {key} is immutable", code="lease_worker_mismatch")
        for key, value in incoming.items():
            if key in {"kind", "worker_id", "pid", "process_start_identity", "thread_id"} and bound.get(key) is None:
                bound[key] = value
        lp["worker"] = bound
        duration = int(lp.get("duration_seconds") or 3600)
        lp["last_heartbeat_at_utc"] = timestamp
        lp["expires_at_utc"] = add_seconds(timestamp, duration)
        revised_lease = _revise_lease(
            lease,
            payload=lp,
            status="ACTIVE",
            action="worker_started",
            state_root=state_root,
            generated_at=timestamp,
        )
        if assignment.get("status") == "RUNNING":
            return assignment, revised_lease
        revised_assignment = _revise_assignment(
            assignment,
            payload=_assignment_payload(assignment),
            status="RUNNING",
            action="worker_started",
            state_root=state_root,
            generated_at=timestamp,
        )
        return revised_assignment, revised_lease

def heartbeat_lease(lease_id: str, *, state_root: str | Path | None, worker: Mapping[str, Any] | None = None, generated_at: str | None = None) -> dict[str, Any]:
    timestamp = utc_now(generated_at)
    store = lease_store(state_root)
    lease = store.get(lease_id)
    payload = _lease_payload(lease)
    assignment_id = str(payload.get("assignment_id") or "")
    with advisory_lock(_assignment_lock_path(state_root, assignment_id)):
        lease = store.get(lease_id)
        payload = _lease_payload(lease)
        if lease.get("status") != "ACTIVE":
            raise ActorError("Lease is not active", code="lease_not_active")
        try:
            assignment = assignment_store(state_root).get(assignment_id)
        except ActorError as exc:
            raise ActorError(
                "Lease Assignment is missing or invalid",
                code="lease_assignment_stale",
                problems=[str(exc)],
                related_records=[lease_id, assignment_id],
            ) from exc
        assignment_payload = _assignment_payload(assignment)
        if str(assignment.get("status") or "") not in {"CLAIMED", "RUNNING"}:
            raise ActorError(
                f"Lease Assignment is not heartbeat-compatible: {assignment.get('status')}",
                code="lease_assignment_stale",
                related_records=[lease_id, assignment_id],
            )
        if str(payload.get("assignment_material_sha256") or "") != str(
            assignment_payload.get("assignment_material_sha256") or ""
        ):
            raise ActorError(
                "Lease Assignment material digest changed",
                code="lease_assignment_stale",
                related_records=[lease_id, assignment_id],
            )
        _require_monotonic_event_time(
            timestamp,
            str(assignment.get("updated_at_utc") or ""),
            str(payload.get("last_heartbeat_at_utc") or payload.get("claimed_at_utc") or ""),
            label="Lease heartbeat",
        )
        if parse_utc(str(payload.get("expires_at_utc"))) <= parse_utc(timestamp):
            raise ActorError("Lease has expired", code="lease_expired")
        if worker:
            bound = dict(payload.get("worker") or {})
            for key, value in dict(worker).items():
                if value is not None and bound.get(key) != value:
                    raise ActorError("heartbeat worker identity does not match Lease owner", code="lease_worker_mismatch")
        duration = int(payload.get("duration_seconds") or 3600)
        payload["last_heartbeat_at_utc"] = timestamp
        payload["expires_at_utc"] = add_seconds(timestamp, duration)
        revised = _revise_lease(
            lease,
            payload=payload,
            status="ACTIVE",
            action="heartbeat",
            state_root=state_root,
            generated_at=timestamp,
        )
        return revised

def _expected_outcome(session: Mapping[str, Any]) -> set[str]:
    payload = dict(session.get("payload") or {})
    if payload.get("session_outcome") == "PROMOTION_REQUESTED":
        return {"PROMOTION_REQUIRED"}
    if session.get("status") == "COMPLETED":
        return {"COMPLETED"}
    if session.get("status") == "BLOCKED":
        return {"BLOCKED", "CHECKPOINTED"}
    if session.get("status") in {"PAUSED", "INTERRUPTED"}:
        return {"CHECKPOINTED", "PROCESS_FAILURE", "ASSIGNMENT_STALE", "INTEGRITY_FAILURE", "OPERATOR_DECISION_REQUIRED"}
    if session.get("status") in {"ACTIVE", "CREATED"}:
        return {"PROCESS_FAILURE", "ASSIGNMENT_STALE", "INTEGRITY_FAILURE", "OPERATOR_DECISION_REQUIRED"}
    return set()


def validate_worker_result(result: Mapping[str, Any], *, assignment: Mapping[str, Any], lease: Mapping[str, Any], state_root: str | Path | None, now: str | None = None) -> dict[str, Any]:
    payload = load_worker_result_payload(result)
    errors: list[str] = []
    utc_now(now)  # validate any supplied timestamp without making it result authority
    ap = load_assignment_payload(assignment.get("payload"), require_ready=True)
    material = dict(ap.get("material") or {})
    execution = dict(payload.get("execution") or {})
    started = None
    ended = None
    try:
        started = parse_utc(str(execution.get("started_at_utc") or ""))
        ended = parse_utc(str(execution.get("ended_at_utc") or ""))
        if ended < started:
            errors.append("Worker Result execution timestamps are reversed")
    except Exception:
        errors.append("Worker Result execution timestamps are invalid")

    if assignment.get("status") not in {"RUNNING", "EXPIRED"}:
        errors.append("Worker Result requires a RUNNING or recoverable EXPIRED Assignment")
    if payload["assignment_id"] != assignment.get("record_id"):
        errors.append("Worker Result assignment_id mismatch")
    if payload["assignment_material_sha256"] != ap.get("assignment_material_sha256"):
        errors.append("Worker Result Assignment digest mismatch")
    if payload["lease_id"] != lease.get("record_id"):
        errors.append("Worker Result Lease mismatch")
    lp = load_lease_payload(lease.get("payload"))
    if lease.get("status") not in {"ACTIVE", "EXPIRED"}:
        errors.append("Worker Result requires an ACTIVE or recoverable EXPIRED Lease")
    try:
        claimed_at = parse_utc(str(lp.get("claimed_at_utc") or ""))
        expires_at = parse_utc(str(lp.get("expires_at_utc") or ""))
        if started is not None and started < claimed_at:
            errors.append("Worker Result execution started before Lease claim")
        # Completion exactly at expiry is rejected.  A delayed Scheduler may
        # submit later, but the worker itself must have completed while the
        # Lease was unquestionably valid.
        if ended is not None and ended >= expires_at:
            errors.append("Worker Result execution did not end before Lease expiry")
    except Exception:
        errors.append("Worker Result Lease timestamps are invalid")
    if assignment.get("status") == "EXPIRED" and lease.get("status") != "EXPIRED":
        errors.append("EXPIRED Assignment requires an EXPIRED Lease for result recovery")
    if lp.get("assignment_id") != assignment.get("record_id"):
        errors.append("Lease Assignment mismatch")
    if lp.get("assignment_material_sha256") != ap.get("assignment_material_sha256"):
        errors.append("Lease Assignment digest mismatch")
    # Result validation occurs after the bounded worker may have committed and
    # refreshed Session truth. Re-validating the opening Context Receipt here
    # would incorrectly reject every legitimate implementation. Preserve only
    # immutable Assignment/Dispatch/Work-Package checks; final repository and
    # Session state are verified below.
    try:
        _verify_ready_material_immutable(assignment, state_root=state_root)
        _validate_dispatch_binding(material, state_root=state_root, require_enabled=False)
        work_info_live = dict(material.get("work") or {})
        work_live = work_store(state_root).get(str(work_info_live.get("record_id") or ""))
        binding_live = dict(dict(work_live.get("payload") or {}).get("work_package_binding") or {})
        package_check_live = verify_work_package_binding(
            binding_live,
            repo_root=str(dict(material.get("repository") or {}).get("repo_root") or ""),
            expected_work_id=str(work_live.get("record_id") or ""),
            state_root=state_root,
        )
        if not package_check_live.get("valid"):
            errors.extend(list(package_check_live.get("errors") or []))
    except ActorError as exc:
        errors.extend([str(exc), *list(exc.problems or [])])
    work_info = dict(material.get("work") or {})
    session_info = dict(material.get("session") or {})
    continuation = dict(material.get("continuation") or {})
    if payload["work_id"] != work_info.get("record_id"):
        errors.append("Worker Result Work mismatch")
    if payload["session_id"] != session_info.get("record_id"):
        errors.append("Worker Result Session mismatch")
    if payload["context_receipt_sha256"] != continuation.get("context_receipt_sha256"):
        errors.append("Worker Result Context Receipt mismatch")
    try:
        sessions = session_store(state_root)
        session = sessions.get(str(session_info.get("record_id") or ""))
    except ActorError as exc:
        errors.append(str(exc))
        session = None
    if session:
        result_session = dict(payload.get("session") or {})
        # A Lease sweep may append a worker-loss Session revision after the
        # worker already completed and wrote its result.  Immutable history is
        # therefore consulted for the exact result-bound revision rather than
        # treating the later administrative revision as proof the result is
        # false.
        if (
            int(result_session.get("revision_after", -1)) != int(session.get("revision") or 0)
            or result_session.get("record_sha256_after") != session.get("record_sha256")
        ):
            historical_session_id = str(session_info.get("record_id") or "")
            for historical in reversed(sessions.history(historical_session_id)):
                if (
                    int(result_session.get("revision_after", -1)) == int(historical.get("revision") or 0)
                    and result_session.get("record_sha256_after") == historical.get("record_sha256")
                ):
                    # ``history()`` is intentionally a compact display projection.
                    # Worker Result settlement needs the exact immutable payload
                    # because outcome and worker identity are authoritative fields.
                    # Load and verify the full sealed revision instead of treating
                    # the history summary as source truth.
                    session = sessions.get_revision_read_only(
                        historical_session_id, int(historical.get("revision") or 0)
                    )
                    if session.get("record_sha256") != historical.get("record_sha256"):
                        errors.append("historical Session revision digest mismatch")
                    break
        session_payload = dict(session.get("payload") or {})
        if int(result_session.get("revision_after", -1)) != int(session.get("revision") or 0):
            errors.append("Worker Result Session revision mismatch")
        if result_session.get("record_sha256_after") != session.get("record_sha256"):
            errors.append("Worker Result Session digest mismatch")
        if result_session.get("status_after") != session.get("status"):
            errors.append("Worker Result Session status mismatch")
        if result_session.get("session_outcome") != session_payload.get("session_outcome"):
            errors.append("Worker Result Session outcome mismatch")
        if payload["outcome"] not in _expected_outcome(session):
            errors.append("Worker Result outcome is inconsistent with Session truth")
        result_worker = dict(payload.get("worker_identity") or {})
        bound_worker = session_payload.get("scheduler_worker_id")
        if bound_worker and str(result_worker.get("worker_id") or "") != str(bound_worker):
            errors.append("Worker Result worker identity mismatch")
        profile = dict(material.get("worker_profile") or {})
        current_codex = str(session_payload.get("current_codex_id") or "")
        if profile.get("launcher_type") == "codex_exec":
            thread_id = str(result_worker.get("thread_id") or "")
            if not thread_id:
                errors.append("real Codex Worker Result is missing thread_id")
            if thread_id and thread_id != current_codex:
                errors.append("Worker Result Codex thread identity mismatch")
            if current_codex.startswith(("EIEL-PROVISIONAL-", "PENDING-CODEX-")):
                errors.append("Session still has only a provisional Codex identity")
        # A completed post-commit delivery must bind a fresh Bootstrap for the
        # live HEAD before it can trigger independent review.
        if payload["outcome"] == "COMPLETED":
            try:
                bootstrap = bootstrap_store(state_root).get(str(session_payload.get("bootstrap_record_id") or ""))
                repo_info = dict(material.get("repository") or {})
                live = git_repository_identity(repo_info.get("repo_root"))
                bootstrap_errors = validate_bootstrap_record(bootstrap, expected_repo_root=live["root"], expected_head=live["head"], max_age_hours=None)
                if bootstrap_errors:
                    errors.append("completed worker did not refresh Session Bootstrap to the final HEAD")
            except ActorError as exc:
                errors.append(str(exc))
    repo_info = dict(material.get("repository") or {})
    result_repo = dict(payload.get("repository") or {})
    if result_repo.get("head_before") != repo_info.get("expected_head"):
        errors.append("Worker Result repository head_before mismatch")
    try:
        live = git_repository_identity(repo_info.get("repo_root"))
        if result_repo.get("head_after") != live.get("head"):
            errors.append("Worker Result repository head_after does not match live repository")
        if result_repo.get("branch") != live.get("branch") or live.get("branch") != repo_info.get("expected_branch"):
            errors.append("Worker Result repository branch mismatch")
        if bool(result_repo.get("commit_created")) != (str(result_repo.get("head_before")) != str(result_repo.get("head_after"))):
            errors.append("Worker Result commit_created is inconsistent with repository heads")
        if Path(str(result_repo.get("working_directory") or "/")).resolve() != Path(str(repo_info.get("working_directory") or repo_info.get("repo_root"))).resolve():
            errors.append("Worker Result repository path mismatch")
        if live.get("dirty"):
            errors.append("Worker Result repository is dirty")
    except ActorError as exc:
        errors.append(str(exc))
    if payload["outcome"] == "COMPLETED" and str(dict(payload.get("validation") or {}).get("status") or "") not in {"PASSED", "PARTIAL"}:
        errors.append("COMPLETED Worker Result requires PASSED or PARTIAL validation")
    return {"valid": not errors, "errors": list(dict.fromkeys(errors)), "payload": payload}

def _finalize_result_submission(
    *,
    result_record: Mapping[str, Any],
    assignment_id: str,
    lease_id: str,
    state_root: str | Path | None,
    generated_at: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    assignments = assignment_store(state_root)
    leases = lease_store(state_root)
    assignment = assignments.get(assignment_id)
    lease = leases.get(lease_id)
    if str(assignment.get("status") or "") in {"SUPERSEDED", "INVALID"}:
        raise ActorError(
            "Worker Result cannot resurrect an Assignment ended by supersession or invalidation",
            code="worker_result_assignment_terminal_conflict",
            related_records=[assignment_id],
        )
    result_id = str(result_record.get("record_id") or "")
    ap = _assignment_payload(assignment)
    op = dict(ap.get("operational") or {})
    if op.get("result_id") not in {None, result_id}:
        raise ActorError("Assignment already binds another Worker Result", code="worker_result_conflict")
    if op.get("result_id") != result_id or assignment.get("status") not in {"RESULT_SUBMITTED", "CLOSED"}:
        op["result_id"] = result_id
        ap["operational"] = op
        if assignment.get("status") not in {"RESULT_SUBMITTED", "CLOSED"}:
            assignment = _revise_assignment(
                assignment,
                payload=ap,
                status="RESULT_SUBMITTED",
                action="submit_result",
                state_root=state_root,
                generated_at=generated_at,
            )
    if assignment.get("status") != "CLOSED":
        assignment = _revise_assignment(
            assignment,
            payload=ap,
            status="CLOSED",
            action="close_assignment",
            state_root=state_root,
            generated_at=generated_at,
        )
    if lease.get("status") != "CLOSED":
        if lease.get("status") not in {"ACTIVE", "EXPIRED"}:
            raise ActorError("accepted Worker Result cannot close this Lease state", code="worker_result_lifecycle_conflict")
        action = "close_lease" if lease.get("status") == "ACTIVE" else "finalize_preexpiry_accepted_result"
        lease = _revise_lease(
            lease,
            payload=_lease_payload(lease),
            status="CLOSED",
            action=action,
            state_root=state_root,
            generated_at=generated_at,
        )
    return dict(result_record), assignment, lease


def submit_worker_result(result: Mapping[str, Any], *, state_root: str | Path | None, generated_at: str | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    timestamp = utc_now(generated_at)
    normalized = load_worker_result_payload(result)
    assignment_id = normalized["assignment_id"]
    with advisory_lock(_assignment_lock_path(state_root, assignment_id)):
        assignment = assignment_store(state_root).get(assignment_id)
        lease = lease_store(state_root).get(normalized["lease_id"])
        result_id = result_id_for({
            "assignment_id": normalized["assignment_id"],
            "assignment_digest": normalized["assignment_material_sha256"],
            "lease_id": normalized["lease_id"],
            "session_sha256": dict(normalized.get("session") or {}).get("record_sha256_after"),
            "outcome": normalized["outcome"],
        })
        results = result_store(state_root, write_scope=True)
        if results.exists(result_id):
            existing = results.get(result_id)
            if existing.get("payload") != normalized:
                raise ActorError("Worker Result identity collision", code="worker_result_conflict")
            return _finalize_result_submission(
                result_record=existing,
                assignment_id=assignment_id,
                lease_id=normalized["lease_id"],
                state_root=state_root,
                generated_at=timestamp,
            )
        validation = validate_worker_result(normalized, assignment=assignment, lease=lease, state_root=state_root, now=timestamp)
        if not validation["valid"]:
            raise ActorError("Worker Result failed integrity validation", code="worker_result_invalid", problems=validation["errors"])
        result_record = new_record(
            actor=ACTOR_SCHEDULER,
            record_type="worker_result",
            record_id=result_id,
            status=normalized["outcome"],
            payload=normalized,
            generated_at=timestamp,
            source_refs=[
                actor_source_ref("assignment", assignment),
                actor_source_ref("lease", lease),
            ],
            action="submit_worker_result",
        )
        results.put(result_record)
        return _finalize_result_submission(
            result_record=result_record,
            assignment_id=assignment_id,
            lease_id=normalized["lease_id"],
            state_root=state_root,
            generated_at=timestamp,
        )

def record_execution_loss(
    assignment_id: str,
    lease_id: str,
    *,
    state_root: str | Path | None,
    reason: str,
    generated_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Close one lost execution without inventing an engineering result."""
    timestamp = utc_now(generated_at)
    with advisory_lock(_assignment_lock_path(state_root, assignment_id)):
        assignments = assignment_store(state_root)
        leases = lease_store(state_root)
        assignment = assignments.get(assignment_id)
        lease = leases.get(lease_id)
        lease_payload = _lease_payload(lease)
        if lease_payload.get("assignment_id") != assignment_id:
            raise ActorError("Lease does not own Assignment", code="lease_assignment_mismatch")
        if assignment.get("status") in {"RESULT_SUBMITTED", "CLOSED"}:
            return assignment, lease
        if lease.get("status") == "ACTIVE":
            lease = _revise_lease(
                lease,
                payload=lease_payload,
                status="EXPIRED",
                action="record_worker_loss",
                state_root=state_root,
                generated_at=timestamp,
                note=reason,
            )
        if assignment.get("status") in {"CLAIMED", "RUNNING"}:
            material = dict(_assignment_payload(assignment).get("material") or {})
            session_id = str(dict(material.get("session") or {}).get("record_id") or "")
            if session_id:
                try:
                    record_worker_loss(session_id, state_root=state_root, assignment_id=assignment_id, lease_id=lease_id, reason=reason, generated_at=timestamp)
                except ActorError as exc:
                    if exc.code not in {"session_terminal", "session_transition_invalid"}:
                        raise
            assignment = _revise_assignment(
                assignment,
                payload=_assignment_payload(assignment),
                status="EXPIRED",
                action="record_worker_loss",
                state_root=state_root,
                generated_at=timestamp,
                note=reason,
            )
        return assignment, lease

def expire_leases(*, state_root: str | Path | None, generated_at: str | None = None) -> list[dict[str, Any]]:
    timestamp = utc_now(generated_at)
    expired: list[dict[str, Any]] = []
    for index in lease_store(state_root).list():
        if index.get("status") != "ACTIVE":
            continue
        lease = lease_store(state_root).get(str(index.get("record_id")))
        payload = _lease_payload(lease)
        if parse_utc(str(payload.get("expires_at_utc"))) > parse_utc(timestamp):
            continue
        assignment_id = str(payload.get("assignment_id") or "")
        with advisory_lock(_assignment_lock_path(state_root, assignment_id)):
            lease = lease_store(state_root).get(str(lease.get("record_id")))
            if lease.get("status") != "ACTIVE":
                continue
            payload = _lease_payload(lease)
            if parse_utc(str(payload.get("expires_at_utc"))) > parse_utc(timestamp):
                continue
            revised = _revise_lease(
                lease,
                payload=payload,
                status="EXPIRED",
                action="expire_lease",
                state_root=state_root,
                generated_at=timestamp,
            )
            assignment = assignment_store(state_root).get(assignment_id)
            if assignment.get("status") in {"CLAIMED", "RUNNING"}:
                material = dict(_assignment_payload(assignment).get("material") or {})
                session_id = str(dict(material.get("session") or {}).get("record_id") or "")
                if session_id:
                    record_worker_loss(session_id, state_root=state_root, assignment_id=assignment_id, lease_id=str(lease.get("record_id")), reason="LEASE_EXPIRED", generated_at=timestamp)
                expired_assignment = _revise_assignment(
                    assignment,
                    payload=_assignment_payload(assignment),
                    status="EXPIRED",
                    action="lease_expired",
                    state_root=state_root,
                    generated_at=timestamp,
                )
            expired.append(revised)
    return expired

def worker_open(assignment_id: str, lease_id: str, *, state_root: str | Path | None, worker_id: str, codex_id: str, generated_at: str | None = None) -> dict[str, Any]:
    with advisory_lock(_assignment_lock_path(state_root, assignment_id)):
        assignment = assignment_store(state_root).get(assignment_id)
        lease = lease_store(state_root).get(lease_id)
        if assignment.get("status") not in {"CLAIMED", "RUNNING"}:
            raise ActorError("Worker cannot open an unclaimed Assignment", code="assignment_not_claimed")
        checked = _validate_live_assignment_inputs(assignment, state_root=state_root, allow_active_session=True)
        ap = dict(checked["payload"])
        material = dict(checked["material"])
        lp = _lease_payload(lease)
        if lease.get("status") != "ACTIVE" or lp.get("assignment_id") != assignment_id or lp.get("assignment_material_sha256") != ap.get("assignment_material_sha256"):
            raise ActorError("Worker Lease is not valid for Assignment", code="lease_assignment_mismatch")
        if parse_utc(str(lp.get("expires_at_utc"))) <= parse_utc(utc_now(generated_at)):
            raise ActorError("Worker Lease has expired", code="lease_expired")
        lease_worker_id = str(dict(lp.get("worker") or {}).get("worker_id") or "")
        if lease_worker_id and lease_worker_id != worker_id:
            raise ActorError("Worker identity does not match the active Lease", code="lease_worker_mismatch", related_records=[lease_id])
        session_info = dict(material.get("session") or {})
        activated = activate_session_for_assignment(
            str(session_info.get("record_id")),
            state_root=state_root,
            assignment_id=assignment_id,
            assignment_digest=str(ap.get("assignment_material_sha256")),
            lease_id=lease_id,
            worker_id=worker_id,
            codex_id=codex_id,
            repo_root=str(dict(material.get("repository") or {}).get("repo_root")),
            generated_at=generated_at,
        )
        return activated


def acknowledge_worker_launch(
    assignment_id: str,
    lease_id: str,
    ack_path: str | Path,
    *,
    state_root: str | Path | None,
    launch_nonce: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Commit the final human authorization check and release the ACK gate."""

    assignment = assignment_store(state_root).get(assignment_id)
    material = dict(_assignment_payload(assignment).get("material") or {})
    dispatch_id = str(dict(material.get("dispatch_request") or {}).get("record_id") or "")
    if not dispatch_id:
        raise ActorError("Assignment has no Dispatch binding", code="assignment_dispatch_missing")
    timestamp = utc_now(generated_at)
    with advisory_lock(_dispatch_lock_path(state_root, dispatch_id)):
        with advisory_lock(_assignment_lock_path(state_root, assignment_id)):
            assignment = assignment_store(state_root).get(assignment_id)
            if assignment.get("status") != "RUNNING":
                raise ActorError("Assignment is not RUNNING at launch ACK", code="assignment_not_running")
            checked = _validate_live_assignment_inputs(assignment, state_root=state_root, allow_active_session=True)
            lease = lease_store(state_root).get(lease_id)
            lp = _lease_payload(lease)
            if lease.get("status") != "ACTIVE" or lp.get("assignment_id") != assignment_id:
                raise ActorError("Lease is not active at launch ACK", code="lease_assignment_mismatch")
            if parse_utc(str(lp.get("expires_at_utc"))) <= parse_utc(timestamp):
                raise ActorError("Lease expired before launch ACK", code="lease_expired")
            payload = _assignment_payload(assignment)
            operational = dict(payload.get("operational") or {})
            if not operational.get("launch_acknowledged_at_utc"):
                operational["launch_acknowledged_at_utc"] = timestamp
                payload["operational"] = operational
                assignment = _revise_assignment(
                    assignment,
                    payload=payload,
                    status="RUNNING",
                    action="launch_acknowledged",
                    state_root=state_root,
                    generated_at=timestamp,
                )
            atomic_write_json(
                Path(ack_path),
                {
                    "schema_version": SCHEDULER_LAUNCH_ACK_SCHEMA_VERSION,
                    "assignment_id": assignment_id,
                    "lease_id": lease_id,
                    "launch_nonce": str(launch_nonce),
                },
            )
            return assignment


def bind_worker_thread_identity(assignment_id: str, lease_id: str, *, state_root: str | Path | None, worker_id: str, thread_id: str, generated_at: str | None = None) -> dict[str, Any]:
    with advisory_lock(_assignment_lock_path(state_root, assignment_id)):
        assignment = assignment_store(state_root).get(assignment_id)
        material = dict(_assignment_payload(assignment).get("material") or {})
        lease = lease_store(state_root).get(lease_id)
        lp = _lease_payload(lease)
        timestamp = utc_now(generated_at)
        if assignment.get("status") != "RUNNING":
            raise ActorError("Assignment is not RUNNING for thread binding", code="assignment_not_running")
        if lease.get("status") != "ACTIVE" or lp.get("assignment_id") != assignment_id:
            raise ActorError("Lease is not active for thread binding", code="lease_assignment_mismatch")
        if parse_utc(str(lp.get("expires_at_utc"))) <= parse_utc(timestamp):
            raise ActorError("Lease expired before Codex thread binding", code="lease_expired")
        bound = dict(lp.get("worker") or {})
        if bound.get("worker_id") != worker_id:
            raise ActorError("thread binding worker does not match Lease", code="lease_worker_mismatch")
        prior_thread = str(bound.get("thread_id") or "")
        if prior_thread and not prior_thread.startswith(("EIEL-PROVISIONAL-", "PENDING-CODEX-")) and prior_thread != thread_id:
            raise ActorError("Lease already binds another Codex thread", code="lease_worker_mismatch")
        bound["thread_id"] = thread_id
        lp["worker"] = bound
        revised_lease = _revise_lease(
            lease,
            payload=lp,
            status="ACTIVE",
            action="bind_codex_thread",
            state_root=state_root,
            generated_at=timestamp,
        )
        session_id = str(dict(material.get("session") or {}).get("record_id") or "")
        return bind_session_codex_thread_identity(
            session_id,
            state_root=state_root,
            assignment_id=assignment_id,
            lease_id=lease_id,
            worker_id=worker_id,
            thread_id=thread_id,
            generated_at=generated_at,
        )


def supersede_assignment_for_manual_takeover(assignment_id: str, *, state_root: str | Path | None, reason: str, generated_at: str | None = None, detach_session: bool = True, allow_unacknowledged_running: bool = False) -> dict[str, Any] | None:
    """Supersede one Scheduler attempt and make its Session manually recoverable.

    Cleanup is intentionally retryable.  A prior call may have durably
    superseded the Assignment before Session cleanup failed.  In that case an
    exact retry must finish the cleanup instead of returning early and hiding
    the orphaned prepared Session.
    """

    if not assignment_id or not assignment_store(state_root).exists(assignment_id):
        return None
    timestamp = utc_now(generated_at)
    with advisory_lock(_assignment_lock_path(state_root, assignment_id)):
        store = assignment_store(state_root)
        assignment = store.get(assignment_id)
        if assignment.get("status") in {"RESULT_SUBMITTED", "CLOSED", "INVALID", "EXPIRED"}:
            # RESULT_SUBMITTED is settlement-owned. Manual takeover after a
            # validated result exists would fork the authoritative lifecycle.
            return assignment

        payload = _assignment_payload(assignment)
        operational = dict(payload.get("operational") or {})
        already_superseded = assignment.get("status") == "SUPERSEDED"
        if assignment.get("status") == "RUNNING":
            if not allow_unacknowledged_running or operational.get("launch_acknowledged_at_utc"):
                raise ActorError("running Assignment requires worker-loss recovery before manual takeover", code="manual_takeover_worker_running")

        lease_id = str(operational.get("current_lease_id") or "")
        if not already_superseded and lease_id and lease_store(state_root).exists(lease_id):
            lease = lease_store(state_root).get(lease_id)
            if lease.get("status") == "ACTIVE":
                lease = _revise_lease(
                    lease,
                    payload=_lease_payload(lease),
                    status="REVOKED",
                    action="manual_takeover",
                    state_root=state_root,
                    generated_at=timestamp,
                    note=reason,
                )

        if not already_superseded:
            operational["manual_takeover_cleanup_required"] = bool(detach_session or allow_unacknowledged_running)
            operational["manual_takeover_reason"] = reason
            payload["operational"] = operational
