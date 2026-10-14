"""Cross-record and source-reference integrity for durable EIEL actor state.

The file-backed Actor stores deliberately stay small.  This module provides the
repository-wide graph layer that can dereference typed actor references without
turning every manager into a second, slightly different integrity engine.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping

from tools.actors.common import (
    ACTOR_SOURCE_REF_KINDS,
    EXTERNAL_SOURCE_REF_KINDS,
    EXTERNAL_UNVERIFIABLE_SOURCE_KIND,
    ActorError,
    file_sha256,
    validate_source_ref_shape,
)
from tools.actors.store import ActorRecordStore

REFERENCE_VALID = "VALID"
REFERENCE_SUPPORTED_HISTORICAL = "SUPPORTED_HISTORICAL"
REFERENCE_EXTERNAL_UNVERIFIABLE = "EXTERNAL_UNVERIFIABLE"
REFERENCE_OPTIONAL_ABSENT = "OPTIONAL_ABSENT"
REFERENCE_RECOVERABLE_STALE = "RECOVERABLE_STALE"
REFERENCE_CORRUPT = "CORRUPT"

ACTOR_REFERENCE_TARGETS: dict[str, tuple[str, frozenset[str]]] = {
    "work": ("work", frozenset({"work_item"})),
    "session": ("sessions", frozenset({"codex_session"})),
    "bootstrap": ("bootstrap", frozenset({"repository_bootstrap"})),
    "mission": ("missions", frozenset({"mission_specification"})),
    "technical_qualification": (
        "qualifications",
        frozenset({"technical_qualification_plan", "technical_qualification_decision", "technical_qualification_check"}),
    ),
    "qualification": (
        "qualifications",
        frozenset({"technical_qualification_plan", "technical_qualification_decision", "technical_qualification_check"}),
    ),
    "resource_observation": ("resources", frozenset({"resource_observation"})),
    "resource_assessment": ("resources", frozenset({"resource_admission_recommendation"})),
    "evidence_assessment": ("evidence-decisions", frozenset({"evidence_assessment"})),
    "dispatch_request": ("dispatch-requests", frozenset({"dispatch_request"})),
    "assignment": ("assignments", frozenset({"assignment"})),
    "lease": ("leases", frozenset({"assignment_lease"})),
    "worker_result": ("worker-results", frozenset({"worker_result"})),
}

SCHEDULER_REFERENCE_NAMESPACES = frozenset(
    {"dispatch-requests", "assignments", "leases", "worker-results"}
)


def actor_source_ref(kind: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """Build an exact immutable actor source reference."""

    kind_value = str(kind).strip()
    if kind_value not in ACTOR_REFERENCE_TARGETS:
        raise ActorError(f"unknown actor source reference kind: {kind_value}", code="source_reference_kind_unknown")
    return {
        "schema_version": "eiel_actor_record_ref_v1",
        "kind": kind_value,
        "record_id": str(record.get("record_id") or ""),
        "record_type": str(record.get("record_type") or ""),
        "revision": int(record.get("revision") or 0),
        "status": str(record.get("status") or ""),
        "record_sha256": str(record.get("record_sha256") or ""),
    }


def _actor_reference_result(ref: Mapping[str, Any], *, state_root: str | Path) -> dict[str, Any]:
    kind = str(ref.get("kind") or "")
    namespace, expected_types = ACTOR_REFERENCE_TARGETS[kind]
    actor_root = Path(state_root).expanduser().resolve()
    store_root = (
        actor_root / "scheduler"
        if namespace in SCHEDULER_REFERENCE_NAMESPACES
        else actor_root
    )
    store = ActorRecordStore(
        store_root,
        namespace,
        reference_root=actor_root,
    )
    record_id = str(ref.get("record_id") or "")
    revision = ref.get("revision")
    try:
        if isinstance(revision, int) and not isinstance(revision, bool):
            record = store.get_revision_read_only(record_id, revision)
            classification = REFERENCE_SUPPORTED_HISTORICAL
        else:
            record = store.get_read_only(record_id)
            classification = REFERENCE_VALID
    except ActorError as exc:
        return {
            "classification": REFERENCE_CORRUPT,
            "kind": kind,
            "record_id": record_id,
            "namespace": namespace,
            "errors": [f"source reference target is not resolvable: {exc}", *[str(v) for v in (exc.problems or [])]],
            "warnings": [],
        }
    errors: list[str] = []
    warnings: list[str] = []
    if str(record.get("record_type") or "") not in expected_types:
        errors.append(
            f"source reference kind {kind!r} expects record_type in {sorted(expected_types)}, "
            f"observed {record.get('record_type')!r}"
        )
    declared_type = ref.get("record_type")
    if declared_type is not None and str(declared_type) != str(record.get("record_type") or ""):
        errors.append("source reference record_type differs from resolved target")
    declared_status = ref.get("status")
    if declared_status is not None and str(declared_status) != str(record.get("status") or ""):
        errors.append("source reference status differs from resolved target")
    digest = ref.get("record_sha256", ref.get("sha256"))
    if digest is not None and str(digest) != str(record.get("record_sha256") or ""):
        errors.append("source reference digest differs from resolved target")
    if revision is None or digest is None:
        warnings.append("actor source reference is current-only rather than exact historical")
    return {
        "classification": REFERENCE_CORRUPT if errors else classification,
        "kind": kind,
        "record_id": record_id,
        "namespace": namespace,
        "resolved_record": record,
        "errors": errors,
        "warnings": warnings,
    }


def _external_reference_result(
    ref: Mapping[str, Any], *, repo_root: str | Path | None = None
) -> dict[str, Any]:
    kind = str(ref.get("kind") or "")
    locator = str(ref.get("uri") if ref.get("uri") is not None else ref.get("path") or "")
    path = Path(locator).expanduser()
    if not path.is_absolute() and repo_root is not None:
        path = Path(repo_root).expanduser().resolve() / path
    warnings: list[str] = []
    errors: list[str] = []
    if kind == "repository":
        if not path.is_dir():
            errors.append(f"repository source path is not available: {path}")
        elif ref.get("commit"):
            result = subprocess.run(
                ["git", "-C", str(path), "cat-file", "-e", f"{ref['commit']}^{{commit}}"],
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                errors.append(f"repository source commit is not resolvable: {ref['commit']}")
    elif path.is_file():
        if ref.get("sha256") and kind != "work_package" and file_sha256(path) != str(ref.get("sha256")):
            errors.append("external source file digest mismatch")
        if kind == "work_package" and ref.get("sha256"):
            warnings.append("Work Package source digest is semantic and is verified through its contract receipt")
    elif path.exists():
        warnings.append(f"external source locator is not a regular file: {path}")
    else:
        warnings.append(f"external source is not locally resolvable: {locator}")
    return {
        "classification": REFERENCE_CORRUPT if errors else (
            REFERENCE_EXTERNAL_UNVERIFIABLE if warnings else REFERENCE_VALID
        ),
        "kind": kind,
        "locator": locator,
        "errors": errors,
        "warnings": warnings,
    }


def resolve_source_reference(
    raw: Mapping[str, Any],
    *,
    state_root: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    shape_errors = validate_source_ref_shape(raw)
    if shape_errors:
        return {
            "classification": REFERENCE_CORRUPT,
            "kind": str(dict(raw).get("kind") or "") if isinstance(raw, Mapping) else "",
            "errors": shape_errors,
            "warnings": [],
        }
    ref = dict(raw)
    kind = str(ref.get("kind") or "")
    if kind in ACTOR_SOURCE_REF_KINDS:
        return _actor_reference_result(ref, state_root=state_root)
    if kind in EXTERNAL_SOURCE_REF_KINDS:
        return _external_reference_result(ref, repo_root=repo_root)
    if kind == EXTERNAL_UNVERIFIABLE_SOURCE_KIND:
        return {
            "classification": REFERENCE_EXTERNAL_UNVERIFIABLE,
            "kind": kind,
            "locator": str(ref.get("uri") or ""),
            "errors": [],
            "warnings": [str(ref.get("reason") or "external source is intentionally unverifiable")],
        }
    return {
        "classification": REFERENCE_CORRUPT,
        "kind": kind,
        "errors": [f"unknown source reference kind: {kind}"],
        "warnings": [],
    }


def validate_record_source_references(
    record: Mapping[str, Any],
    *,
    state_root: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    for index, raw in enumerate(record.get("source_refs") or []):
        result = resolve_source_reference(raw, state_root=state_root, repo_root=repo_root)
        result = {"index": index, **result}
        rows.append(result)
        errors.extend(f"source_refs[{index}]: {value}" for value in result.get("errors") or [])
        warnings.extend(f"source_refs[{index}]: {value}" for value in result.get("warnings") or [])
    return {
        "classification": REFERENCE_CORRUPT if errors else (
            REFERENCE_EXTERNAL_UNVERIFIABLE if warnings else REFERENCE_VALID
        ),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "references": rows,
    }


def _resolve_graph_target(
    store: ActorRecordStore,
    record_id: str | None,
    *,
    required: bool,
    allow_historical: bool,
) -> dict[str, Any]:
    """Resolve one durable graph target without repairing current state."""

    normalized = str(record_id or "").strip().upper()
    if not normalized:
        classification = REFERENCE_CORRUPT if required else REFERENCE_OPTIONAL_ABSENT
        return {
            "classification": classification,
            "record_id": normalized or None,
            "record": None,
            "errors": ["required reference is absent"] if required else [],
            "warnings": [],
        }
    try:
        record = store.get_read_only(normalized)
    except ActorError as current_error:
        try:
            history = store.history_read_only(normalized)
        except ActorError:
            history = []
        if history:
            latest = history[-1]
            try:
                record = store.get_revision_read_only(
                    normalized,
                    int(latest.get("revision") or 0),
                )
            except ActorError as history_error:
                return {
                    "classification": REFERENCE_CORRUPT,
                    "record_id": normalized,
                    "record": None,
                    "errors": [
                        f"reference target current state is missing and immutable history is unreadable: {history_error}"
                    ],
                    "warnings": [],
                }
            if allow_historical:
                return {
                    "classification": REFERENCE_SUPPORTED_HISTORICAL,
                    "record_id": normalized,
                    "record": record,
                    "errors": [],
                    "warnings": ["target resolves only through immutable history"],
                }
            return {
                "classification": REFERENCE_CORRUPT,
                "record_id": normalized,
                "record": record,
                "errors": ["active reference target has no current record; immutable history alone is insufficient"],
                "warnings": [],
            }
        return {
            "classification": REFERENCE_CORRUPT,
            "record_id": normalized,
            "record": None,
            "errors": [f"reference target is not resolvable: {current_error}"],
            "warnings": [],
        }
    return {
        "classification": REFERENCE_VALID,
        "record_id": normalized,
        "record": record,
        "errors": [],
        "warnings": [],
    }


def _exact_snapshot_target(
    store: ActorRecordStore,
    snapshot: Mapping[str, Any] | None,
    *,
    required: bool,
) -> dict[str, Any]:
    """Resolve one frozen record snapshot by exact revision and digest."""

    value = dict(snapshot or {})
    record_id = str(value.get("record_id") or value.get("work_id") or "").strip().upper()
    if not record_id:
        classification = REFERENCE_CORRUPT if required else REFERENCE_OPTIONAL_ABSENT
        return {
            "classification": classification,
            "record_id": None,
            "record": None,
            "errors": ["required frozen snapshot has no record_id"] if required else [],
            "warnings": [],
        }
    revision = value.get("revision")
    digest = str(value.get("record_sha256") or "")
    errors: list[str] = []
    record: dict[str, Any] | None = None
    if isinstance(revision, int) and not isinstance(revision, bool) and revision > 0:
        try:
            record = store.get_revision_read_only(record_id, revision)
        except ActorError as exc:
            errors.append(f"frozen snapshot revision is not resolvable: {exc}")
    else:
        errors.append("frozen snapshot requires a positive integer revision")
    if record is not None and digest and digest != str(record.get("record_sha256") or ""):
        errors.append("frozen snapshot digest differs from immutable target")
    if not digest:
        errors.append("frozen snapshot requires record_sha256")
    return {
        "classification": REFERENCE_CORRUPT if errors else REFERENCE_SUPPORTED_HISTORICAL,
        "record_id": record_id,
        "record": record,
        "errors": errors,
        "warnings": [] if errors else ["exact immutable revision resolved"],
    }


def audit_scheduler_reference_graph(
    *,
    state_root: str | Path,
) -> dict[str, Any]:
    """Audit the durable Session/Scheduler execution graph without mutation.

    This is the Scheduler portion of the repository-wide integrity traversal.
    Workspace combines it with Actor-store, Context Receipt, Work Package, WRA,
    Mission, review and qualification checks.  Every edge receives one explicit
    classification so optional absence, immutable history and corruption are
    not collapsed into the same vague warning bucket.
    """

    # Local imports prevent the durable reference registry from becoming a
    # lifecycle writer or introducing import cycles between managers.
    from tools.actors.portfolio import session_store, work_store
    from tools.scheduler.contracts import (
        assignment_store,
        dispatch_store,
        lease_store,
        result_store,
    )

    actor_root = Path(state_root).expanduser().resolve()
    works = work_store(actor_root)
    sessions = session_store(actor_root)
    dispatches = dispatch_store(actor_root)
    assignments = assignment_store(actor_root)
    leases = lease_store(actor_root)
    results = result_store(actor_root)

    edges: list[dict[str, Any]] = []

    def add_edge(
        *,
        source_kind: str,
        source_id: str,
        edge: str,
        target_kind: str,
        result: Mapping[str, Any],
        extra_errors: list[str] | None = None,
        extra_warnings: list[str] | None = None,
    ) -> None:
        errors = [*list(result.get("errors") or []), *list(extra_errors or [])]
        warnings = [*list(result.get("warnings") or []), *list(extra_warnings or [])]
        classification = str(result.get("classification") or REFERENCE_CORRUPT)
        if errors:
            classification = REFERENCE_CORRUPT
        elif classification == REFERENCE_VALID and warnings:
            classification = REFERENCE_RECOVERABLE_STALE
        edges.append(
            {
                "source_kind": source_kind,
                "source_id": source_id,
                "edge": edge,
                "target_kind": target_kind,
                "target_id": result.get("record_id"),
                "classification": classification,
                "errors": list(dict.fromkeys(str(item) for item in errors if str(item))),
                "warnings": list(dict.fromkeys(str(item) for item in warnings if str(item))),
            }
        )

    def current_records(store: ActorRecordStore) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for row in store.list_read_only():
            if row.get("status") in {"INVALID", "RECOVERABLE_STALE"}:
                continue
            record_id = str(row.get("record_id") or "")
            if not record_id:
                continue
            try:
                records.append(store.get_read_only(record_id))
            except ActorError:
                # Store/index audit reports the current/history defect.  The
                # graph must remain read-only and must not opportunistically
                # promote history merely to enumerate an edge source.
                continue
        return records

    # Dispatch Request -> Work project truth.
    for dispatch in current_records(dispatches):
        dispatch_id = str(dispatch.get("record_id") or "")
        payload = dict(dispatch.get("payload") or {})
        enabled = str(dispatch.get("status") or "") == "ENABLED"
        resolved = _resolve_graph_target(
            works,
            str(payload.get("work_id") or ""),
            required=True,
            allow_historical=not enabled,
        )
        add_edge(
            source_kind="dispatch_request",
            source_id=dispatch_id,
            edge="work_id",
            target_kind="work",
            result=resolved,
        )

    assignment_rows = current_records(assignments)
    lease_rows = current_records(leases)
    result_rows = current_records(results)
    assignment_open_statuses = {
        "PREPARING", "CLEANUP_REQUIRED", "READY", "CLAIMED", "RUNNING", "RESULT_SUBMITTED"
    }

    # Assignment frozen inputs and mutable operational references.
    for assignment in assignment_rows:
        assignment_id = str(assignment.get("record_id") or "")
        status = str(assignment.get("status") or "")
        payload = dict(assignment.get("payload") or {})
        material = dict(payload.get("material") or {})
        preparation = dict(payload.get("preparation") or {})

        dispatch_snapshot = dict(material.get("dispatch_request") or {})
        if not dispatch_snapshot and preparation.get("dispatch_id"):
            dispatch_snapshot = {
                "record_id": preparation.get("dispatch_id"),
                "revision": preparation.get("dispatch_revision"),
                "record_sha256": preparation.get("dispatch_sha256"),
            }
        add_edge(
            source_kind="assignment",
            source_id=assignment_id,
            edge="dispatch_request",
            target_kind="dispatch_request",
            result=_exact_snapshot_target(dispatches, dispatch_snapshot, required=True),
        )

        work_snapshot = dict(material.get("work") or {})
        if not work_snapshot and preparation.get("work_id"):
            work_snapshot = {
                "record_id": preparation.get("work_id"),
                "revision": preparation.get("work_revision"),
                "record_sha256": preparation.get("work_sha256"),
            }
        add_edge(
            source_kind="assignment",
            source_id=assignment_id,
            edge="work",
            target_kind="work",
            result=_exact_snapshot_target(works, work_snapshot, required=True),
        )

        session_snapshot = dict(material.get("session") or {})
        if not session_snapshot and preparation.get("session_id"):
            # PREPARING may exist briefly before the Session is created.  Once
            # READY or later, exact Session identity is mandatory.
            session_snapshot = {"record_id": preparation.get("session_id")}
        session_required = status not in {"PREPARING", "CLEANUP_REQUIRED"}
        if session_snapshot.get("revision"):
            session_result = _exact_snapshot_target(
                sessions,
                session_snapshot,
                required=session_required,
            )
        else:
            session_result = _resolve_graph_target(
                sessions,
                str(session_snapshot.get("record_id") or ""),
                required=session_required,
                allow_historical=status not in assignment_open_statuses,
            )
        add_edge(
            source_kind="assignment",
            source_id=assignment_id,
            edge="session",
            target_kind="session",
            result=session_result,
        )

        continuation = dict(material.get("continuation") or {})
        receipt_id = str(continuation.get("context_receipt_id") or "").strip().upper()
        receipt_path = Path(str(continuation.get("context_receipt_path") or "")).expanduser()
        receipt_sha = str(continuation.get("context_receipt_sha256") or "")
        receipt_errors: list[str] = []
        receipt_warnings: list[str] = []
        if material:
            if not receipt_id:
                receipt_errors.append("Assignment continuation is missing context_receipt_id")
            if not receipt_path.is_file():
                receipt_errors.append(f"Assignment Context Receipt is missing: {receipt_path}")
            else:
                try:
                    import json

                    retained = json.loads(receipt_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    receipt_errors.append(f"Assignment Context Receipt is unreadable: {exc}")
                else:
                    observed_id = str(retained.get("receipt_id") or "")
                    observed_sha = str(retained.get("receipt_sha256") or "")
                    if observed_id != receipt_id:
                        receipt_errors.append("Assignment Context Receipt identity mismatch")
                    if not receipt_sha or receipt_sha != observed_sha:
                        receipt_errors.append("Assignment Context Receipt digest mismatch")
        receipt_result = {
            "classification": (
                REFERENCE_CORRUPT
                if receipt_errors
                else REFERENCE_VALID
                if material
                else REFERENCE_OPTIONAL_ABSENT
            ),
            "record_id": receipt_id or None,
            "errors": receipt_errors,
            "warnings": receipt_warnings,
        }
        add_edge(
            source_kind="assignment",
            source_id=assignment_id,
            edge="context_receipt",
            target_kind="context_receipt",
            result=receipt_result,
        )

        operational = dict(payload.get("operational") or {})
        lease_id = str(operational.get("current_lease_id") or "").strip().upper()
        lease_required = status in {"CLAIMED", "RUNNING", "RESULT_SUBMITTED"}
        lease_result = _resolve_graph_target(
            leases,
            lease_id,
            required=lease_required,
            allow_historical=status not in assignment_open_statuses,
        )
        lease_errors: list[str] = []
        lease_record = lease_result.get("record")
        if isinstance(lease_record, Mapping):
            lp = dict(lease_record.get("payload") or {})
            if str(lp.get("assignment_id") or "") != assignment_id:
                lease_errors.append("Lease points to a different Assignment")
            if str(lp.get("assignment_material_sha256") or "") != str(payload.get("assignment_material_sha256") or ""):
                lease_errors.append("Lease Assignment digest mismatch")
        add_edge(
            source_kind="assignment",
            source_id=assignment_id,
            edge="operational.current_lease_id",
            target_kind="lease",
            result=lease_result,
            extra_errors=lease_errors,
        )

        result_id = str(operational.get("result_id") or "").strip().upper()
        result_required = status in {"RESULT_SUBMITTED", "CLOSED"}
        result_result = _resolve_graph_target(
            results,
            result_id,
            required=result_required,
            allow_historical=status not in assignment_open_statuses,
        )
        result_errors: list[str] = []
        result_record = result_result.get("record")
        if isinstance(result_record, Mapping):
            rp = dict(result_record.get("payload") or {})
            if str(rp.get("assignment_id") or "") != assignment_id:
                result_errors.append("Worker Result points to a different Assignment")
            if lease_id and str(rp.get("lease_id") or "") != lease_id:
                result_errors.append("Worker Result points to a different Lease")
            if str(rp.get("assignment_material_sha256") or "") != str(payload.get("assignment_material_sha256") or ""):
                result_errors.append("Worker Result Assignment digest mismatch")
        add_edge(
            source_kind="assignment",
            source_id=assignment_id,
            edge="operational.result_id",
            target_kind="worker_result",
            result=result_result,
            extra_errors=result_errors,
        )

    # Declared Session scheduler bindings must resolve to the exact execution
    # records. Scheduler participation is optional: a directly started Session
    # with no scheduler provenance is not corrupt merely because it has no
    # Assignment or Lease. Once any scheduler binding is declared, the graph
    # remains strict and rejects partial or ghost active bindings.
    for session in current_records(sessions):
        session_id = str(session.get("record_id") or "")
        status = str(session.get("status") or "")
        payload = dict(session.get("payload") or {})
        identity_state = str(payload.get("worker_identity_state") or "ACTIVE")
        prepared_id = str(payload.get("prepared_assignment_id") or "").strip().upper()
        scheduler_assignment_id = str(payload.get("scheduler_assignment_id") or "").strip().upper()
        scheduler_lease_id = str(payload.get("scheduler_lease_id") or "").strip().upper()
        scheduler_bound = any(
            (
                prepared_id,
                scheduler_assignment_id,
                scheduler_lease_id,
                str(payload.get("scheduler_worker_id") or "").strip(),
                list(payload.get("assignment_history") or []),
            )
        )

        prepared_required = identity_state in {"PENDING", "LOST"} and not scheduler_assignment_id
        prepared_result = _resolve_graph_target(
            assignments,
            prepared_id,
            required=prepared_required,
            allow_historical=not prepared_required,
        )
        prepared_errors: list[str] = []
        prepared_record = prepared_result.get("record")
        if isinstance(prepared_record, Mapping):
            ap = dict(prepared_record.get("payload") or {})
            material = dict(ap.get("material") or {})
            prep = dict(ap.get("preparation") or {})
            bound_session = str(dict(material.get("session") or {}).get("record_id") or prep.get("session_id") or "")
            if bound_session and bound_session != session_id:
                prepared_errors.append("prepared Assignment points to a different Session")
            if prepared_required and str(prepared_record.get("status") or "") not in {
                "PREPARING", "CLEANUP_REQUIRED", "READY"
            }:
                prepared_errors.append("prepared Session references an Assignment that is not preparatory/ready")
        add_edge(
            source_kind="session",
            source_id=session_id,
            edge="prepared_assignment_id",
            target_kind="assignment",
            result=prepared_result,
            extra_errors=prepared_errors,
        )

        active_required = scheduler_bound and status == "ACTIVE" and identity_state == "ACTIVE"
        assignment_result = _resolve_graph_target(
            assignments,
            scheduler_assignment_id,
            required=active_required,
            allow_historical=not active_required,
        )
        assignment_errors: list[str] = []
        assignment_record = assignment_result.get("record")
        if isinstance(assignment_record, Mapping):
            ap = dict(assignment_record.get("payload") or {})
            material = dict(ap.get("material") or {})
            bound_session = str(dict(material.get("session") or {}).get("record_id") or "")
            if bound_session and bound_session != session_id:
                assignment_errors.append("Scheduler Assignment points to a different Session")
            declared_digest = str(payload.get("scheduler_assignment_digest") or "")
            if declared_digest and declared_digest != str(ap.get("assignment_material_sha256") or ""):
                assignment_errors.append("Session Scheduler Assignment digest mismatch")
        add_edge(
            source_kind="session",
            source_id=session_id,
            edge="scheduler_assignment_id",
            target_kind="assignment",
            result=assignment_result,
            extra_errors=assignment_errors,
        )

        lease_result = _resolve_graph_target(
            leases,
            scheduler_lease_id,
            required=active_required,
            allow_historical=not active_required,
        )
        lease_errors = []
        lease_record = lease_result.get("record")
        if isinstance(lease_record, Mapping):
            lp = dict(lease_record.get("payload") or {})
            if scheduler_assignment_id and str(lp.get("assignment_id") or "") != scheduler_assignment_id:
                lease_errors.append("Session Lease points to a different Assignment")
            if active_required and str(lease_record.get("status") or "") != "ACTIVE":
                lease_errors.append("active Session does not bind an ACTIVE Lease")
            worker_id = str(payload.get("scheduler_worker_id") or "")
            lease_worker = str(dict(lp.get("worker") or {}).get("worker_id") or "")
            if worker_id and lease_worker and worker_id != lease_worker:
                lease_errors.append("Session Scheduler worker identity differs from Lease")
        add_edge(
            source_kind="session",
            source_id=session_id,
            edge="scheduler_lease_id",
            target_kind="lease",
            result=lease_result,
            extra_errors=lease_errors,
        )

    # Every retained Lease and Worker Result is checked in the reverse
    # direction as well, so a dangling object cannot hide merely because its
    # Assignment lost the corresponding forward pointer.
    for lease in lease_rows:
        lease_id = str(lease.get("record_id") or "")
        status = str(lease.get("status") or "")
        payload = dict(lease.get("payload") or {})
        assignment_id = str(payload.get("assignment_id") or "").strip().upper()
        resolved = _resolve_graph_target(
            assignments,
            assignment_id,
            required=True,
            allow_historical=status != "ACTIVE",
        )
        errors: list[str] = []
        assignment = resolved.get("record")
        if isinstance(assignment, Mapping):
            ap = dict(assignment.get("payload") or {})
            if str(payload.get("assignment_material_sha256") or "") != str(ap.get("assignment_material_sha256") or ""):
                errors.append("Lease Assignment digest mismatch")
            if status == "ACTIVE" and str(dict(ap.get("operational") or {}).get("current_lease_id") or "") != lease_id:
                errors.append("ACTIVE Lease is not the Assignment current_lease_id")
        add_edge(
            source_kind="lease",
            source_id=lease_id,
            edge="assignment_id",
            target_kind="assignment",
            result=resolved,
            extra_errors=errors,
        )

    for result in result_rows:
        result_id = str(result.get("record_id") or "")
        payload = dict(result.get("payload") or {})
        assignment_id = str(payload.get("assignment_id") or "").strip().upper()
        lease_id = str(payload.get("lease_id") or "").strip().upper()
        assignment_result = _resolve_graph_target(
            assignments,
            assignment_id,
            required=True,
            allow_historical=True,
        )
        assignment_errors: list[str] = []
        assignment = assignment_result.get("record")
        if isinstance(assignment, Mapping):
            ap = dict(assignment.get("payload") or {})
            if str(payload.get("assignment_material_sha256") or "") != str(ap.get("assignment_material_sha256") or ""):
                assignment_errors.append("Worker Result Assignment digest mismatch")
            material = dict(ap.get("material") or {})
            if str(payload.get("work_id") or "") != str(dict(material.get("work") or {}).get("record_id") or ""):
                assignment_errors.append("Worker Result work_id differs from Assignment")
            if str(payload.get("session_id") or "") != str(dict(material.get("session") or {}).get("record_id") or ""):
                assignment_errors.append("Worker Result session_id differs from Assignment")
            continuation = dict(material.get("continuation") or {})
            if str(payload.get("context_receipt_sha256") or "") != str(continuation.get("context_receipt_sha256") or ""):
                assignment_errors.append("Worker Result Context Receipt digest differs from Assignment")
            op_result = str(dict(ap.get("operational") or {}).get("result_id") or "")
            if op_result and op_result != result_id:
                assignment_errors.append("Assignment points to a different Worker Result")
        add_edge(
            source_kind="worker_result",
            source_id=result_id,
            edge="assignment_id",
            target_kind="assignment",
            result=assignment_result,
            extra_errors=assignment_errors,
        )
        lease_result = _resolve_graph_target(
            leases,
            lease_id,
            required=True,
            allow_historical=True,
        )
        lease_errors: list[str] = []
        lease = lease_result.get("record")
        if isinstance(lease, Mapping):
            lp = dict(lease.get("payload") or {})
            if str(lp.get("assignment_id") or "") != assignment_id:
                lease_errors.append("Worker Result Lease points to a different Assignment")
        add_edge(
            source_kind="worker_result",
            source_id=result_id,
            edge="lease_id",
            target_kind="lease",
            result=lease_result,
            extra_errors=lease_errors,
        )

    grouped = {
        classification: [row for row in edges if row["classification"] == classification]
        for classification in (
            REFERENCE_CORRUPT,
            REFERENCE_RECOVERABLE_STALE,
            REFERENCE_SUPPORTED_HISTORICAL,
            REFERENCE_OPTIONAL_ABSENT,
            REFERENCE_EXTERNAL_UNVERIFIABLE,
            REFERENCE_VALID,
        )
    }
    errors = [
        f"{row['source_kind']} {row['source_id']} {row['edge']}: {error}"
        for row in grouped[REFERENCE_CORRUPT]
        for error in row.get("errors") or ["reference is corrupt"]
    ]
    warnings = [
        f"{row['source_kind']} {row['source_id']} {row['edge']}: {warning}"
        for classification in (
            REFERENCE_RECOVERABLE_STALE,
            REFERENCE_EXTERNAL_UNVERIFIABLE,
        )
        for row in grouped[classification]
        for warning in row.get("warnings") or [classification]
    ]
    return {
        "schema_version": "eiel_composite_reference_audit_v1",
        "valid": not errors,
        "counts": {key: len(value) for key, value in grouped.items()},
        "classifications": grouped,
        "edges": edges,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
    }
