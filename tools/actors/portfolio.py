"""Shared portfolio helpers for EIEL Work and Session Managers.

This module contains only deterministic, file-backed lookup and validation
helpers.  It does not execute work, launch processes, or mutate another
manager's authoritative records.
"""

from __future__ import annotations

import itertools
import re
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.actors.common import ActorError, actor_state_root, advisory_lock, validate_record_id
from tools.actors.store import ActorRecordStore, scoped_actor_store

WORK_NAMESPACE = "work"
SESSION_NAMESPACE = "sessions"
BOOTSTRAP_NAMESPACE = "bootstrap"
MISSION_NAMESPACE = "missions"
QUALIFICATION_NAMESPACE = "qualifications"
RESOURCE_NAMESPACE = "resources"
EVIDENCE_NAMESPACE = "evidence-decisions"

WORK_RECORD_TYPE = "work_item"
SESSION_RECORD_TYPE = "codex_session"
BOOTSTRAP_RECORD_TYPE = "repository_bootstrap"

WORK_ID_PREFIX = "WORK-"
SESSION_ID_PREFIX = "SES-"
BOOTSTRAP_ID_PREFIX = "BST-"

WORK_OPEN_STATUSES = {"BACKLOG", "READY", "ACTIVE", "PAUSED", "BLOCKED"}
WORK_SUCCESS_STATUS = "COMPLETED"
WORK_TERMINAL_STATUSES = {"COMPLETED", "DISCONTINUED", "OBSOLETE", "SUPERSEDED"}
SESSION_RESUMABLE_STATUSES = {"CREATED", "ACTIVE", "PAUSED", "INTERRUPTED", "BLOCKED"}
SESSION_ACTIVE_STATUSES = {"ACTIVE"}
SESSION_TERMINAL_STATUSES = {"COMPLETED", "DISCONTINUED", "OBSOLETE", "SUPERSEDED"}

CHATGPT_CONVERSATION_RE = re.compile(r"(?:^|/)c/([0-9a-fA-F-]{16,})")


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Command-scoped current-state projection loaded exactly once.

    It is deliberately not durable authority.  The snapshot exists only for
    one command/query and replaces repeated repository-wide scans while a Work
    summary asks for dependencies, Sessions, delivery lineage, and review
    state.
    """

    work_by_id: Mapping[str, Mapping[str, Any]]
    session_by_id: Mapping[str, Mapping[str, Any]]
    sessions_by_work: Mapping[str, tuple[Mapping[str, Any], ...]]
    dependency_graph: Mapping[str, tuple[str, ...]]
    invalid_work_rows: tuple[Mapping[str, Any], ...] = ()
    invalid_session_rows: tuple[Mapping[str, Any], ...] = ()
    read_only: bool = True


def build_portfolio_snapshot(
    state_root: str | Path | None = None,
    *,
    read_only: bool = True,
) -> PortfolioSnapshot:
    """Load current Work and Session state once for one bounded command."""

    def load(store_obj: ActorRecordStore) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        rows = store_obj.list_read_only() if read_only else store_obj.list()
        getter = store_obj.get_read_only if read_only else store_obj.get
        values: dict[str, dict[str, Any]] = {}
        invalid: list[dict[str, Any]] = []
        for row in rows:
            record_id = str(row.get("record_id") or "")
            if (
                not record_id
                or row.get("status") == "INVALID"
                or row.get("integrity_classification") == "RECOVERABLE_STALE"
            ):
                invalid.append(dict(row))
                continue
            try:
                values[record_id] = getter(record_id)
            except ActorError as exc:
                invalid.append({**dict(row), "status": "INVALID", "error": str(exc)})
        return values, invalid

    works, invalid_works = load(work_store(state_root))
    sessions, invalid_sessions = load(session_store(state_root))
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in sessions.values():
        work_id = str(dict(record.get("payload") or {}).get("work_id") or "")
        if work_id:
            grouped.setdefault(work_id, []).append(record)
    ordered_groups = {
        work_id: tuple(sorted(
            values,
            key=lambda row: (
                str(row.get("updated_at_utc") or ""),
                str(row.get("record_id") or ""),
            ),
            reverse=True,
        ))
        for work_id, values in grouped.items()
    }
    graph = {
        work_id: tuple(normalize_string_list(dict(record.get("payload") or {}).get("dependency_work_ids")))
        for work_id, record in works.items()
    }
    return PortfolioSnapshot(
        work_by_id=works,
        session_by_id=sessions,
        sessions_by_work=ordered_groups,
        dependency_graph=graph,
        invalid_work_rows=tuple(invalid_works),
        invalid_session_rows=tuple(invalid_sessions),
        read_only=read_only,
    )


def store(root: str | Path | None, namespace: str) -> ActorRecordStore:
    return ActorRecordStore(actor_state_root(root), namespace)


def work_store(
    root: str | Path | None = None,
    *,
    write_scope: bool = False,
) -> ActorRecordStore:
    """Return the Work read adapter or the explicit Work Manager writer."""

    if write_scope:
        return scoped_actor_store(
            actor_state_root(root),
            WORK_NAMESPACE,
            actor="workManager",
            record_types=(WORK_RECORD_TYPE,),
        )
    return store(root, WORK_NAMESPACE)


def work_writer_store(root: str | Path | None = None) -> ActorRecordStore:
    """Compatibility alias for the manager-scoped Work store."""

    return work_store(root, write_scope=True)


def session_store(
    root: str | Path | None = None,
    *,
    write_scope: bool = False,
) -> ActorRecordStore:
    """Return the Session read adapter or the explicit Session Manager writer."""

    if write_scope:
        return scoped_actor_store(
            actor_state_root(root),
            SESSION_NAMESPACE,
            actor="sessionManager",
            record_types=(SESSION_RECORD_TYPE,),
        )
    return store(root, SESSION_NAMESPACE)


def session_writer_store(root: str | Path | None = None) -> ActorRecordStore:
    """Compatibility alias for the manager-scoped Session store."""

    return session_store(root, write_scope=True)


def bootstrap_store(
    root: str | Path | None = None,
    *,
    write_scope: bool = False,
) -> ActorRecordStore:
    """Return the Bootstrap read adapter or the explicit Bootstrap writer."""

    if write_scope:
        return scoped_actor_store(
            actor_state_root(root),
            BOOTSTRAP_NAMESPACE,
            actor="bootstrap",
            record_types=(BOOTSTRAP_RECORD_TYPE,),
        )
    return store(root, BOOTSTRAP_NAMESPACE)


def bootstrap_writer_store(root: str | Path | None = None) -> ActorRecordStore:
    """Compatibility alias for the manager-scoped Bootstrap store."""

    return bootstrap_store(root, write_scope=True)


def mission_store(root: str | Path | None = None) -> ActorRecordStore:
    return store(root, MISSION_NAMESPACE)


def qualification_store(root: str | Path | None = None) -> ActorRecordStore:
    return store(root, QUALIFICATION_NAMESPACE)


def resource_store(root: str | Path | None = None) -> ActorRecordStore:
    return store(root, RESOURCE_NAMESPACE)


def evidence_store(root: str | Path | None = None) -> ActorRecordStore:
    return store(root, EVIDENCE_NAMESPACE)


def validate_prefixed_id(value: str, prefix: str, *, label: str) -> str:
    record_id = validate_record_id(value)
    if not record_id.startswith(prefix):
        raise ActorError(
            f"{label} must start with {prefix}",
            code=f"{label.replace(' ', '_')}_id_invalid",
            problems=[f"Observed {record_id!r}."],
        )
    return record_id


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    rows = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
    normalized = [str(item).strip() for item in rows if str(item).strip()]
    # A set has no semantic order.  Preserving its interpreter-dependent
    # iteration order leaks PYTHONHASHSEED into actor records and stable IDs.
    # Lists and tuples retain their explicit order because several callers use
    # this helper for genuinely ordered histories and command sequences.
    if isinstance(value, (set, frozenset)):
        normalized.sort()
    return list(dict.fromkeys(normalized))


def parallel_session_ownership_problems(
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return concrete repository-ownership conflicts for parallel Sessions.

    An acknowledgement bit records operator intent.  It does not prove that
    two implementation-capable Sessions can mutate safely in parallel.  Until
    the repository has an explicit file-scope lease, the supported proof is a
    distinct Git worktree *and* distinct branch for every pair of write-capable
    Sessions.  Review Authority Sessions do not claim the implementation
    mutation surface and are therefore excluded from this pairwise check.
    """

    writers: list[tuple[str, str, str]] = []
    for record in records:
        payload = dict(record.get("payload") or {})
        if str(record.get("status") or "") != "ACTIVE":
            continue
        if str(payload.get("session_role") or "").upper() == "REVIEW_AUTHORITY":
            continue
        repository = dict(payload.get("repository") or {})
        writers.append(
            (
                str(record.get("record_id") or "<candidate>"),
                str(repository.get("repo_root") or "").strip(),
                str(repository.get("branch") or "").strip(),
            )
        )

    problems: list[str] = []
    for left, right in itertools.combinations(writers, 2):
        left_id, left_root, left_branch = left
        right_id, right_root, right_branch = right
        if not left_root or not right_root or not left_branch or not right_branch:
            problems.append(
                f"parallel Sessions {left_id} and {right_id} lack complete worktree/branch ownership proof"
            )
            continue
        try:
            same_root = Path(left_root).expanduser().resolve() == Path(right_root).expanduser().resolve()
        except OSError:
            same_root = left_root == right_root
        overlaps: list[str] = []
        if same_root:
            overlaps.append("worktree")
        if left_branch == right_branch:
            overlaps.append("branch")
        if overlaps:
            problems.append(
                f"parallel Sessions {left_id} and {right_id} share " + " and ".join(overlaps)
            )
    return problems


def normalize_reference_map(
    raw: Mapping[str, Any] | None,
    *,
    fields: Sequence[str],
) -> dict[str, list[str]]:
    source = dict(raw or {})
    return {field: normalize_string_list(source.get(field)) for field in fields}


def work_lifecycle_lock(state_root: str | Path | None, work_id: str):
    """Serialize Work/Session lifecycle decisions for one Work Item.

    The invariant is Work-scoped, so the lock must be Work-scoped too. Store
    namespace locks remain responsible for record integrity underneath it.
    """

    normalized = validate_prefixed_id(work_id, WORK_ID_PREFIX, label="work item")
    return advisory_lock(actor_state_root(state_root) / "lifecycle-locks" / f"{normalized}.lock")


@contextmanager
def work_lifecycle_locks(state_root: str | Path | None, work_ids: Sequence[str]):
    """Acquire several Work lifecycle locks in deterministic order.

    Cross-Work operations such as dependency mutation and supersession must not
    observe one Work before another process changes its peer. Sorted acquisition
    avoids lock-order deadlocks without introducing another authority surface.
    """

    normalized = sorted({
        validate_prefixed_id(value, WORK_ID_PREFIX, label="work item")
        for value in work_ids
        if str(value or "").strip()
    })
    with ExitStack() as stack:
        for work_id in normalized:
            stack.enter_context(work_lifecycle_lock(state_root, work_id))
        yield


def records(store_obj: ActorRecordStore, *, read_only: bool = False) -> Iterable[dict[str, Any]]:
    list_rows = store_obj.list_read_only() if read_only else store_obj.list()
    getter = store_obj.get_read_only if read_only else store_obj.get
    for row in list_rows:
        record_id = row.get("record_id")
        if not record_id or row.get("status") == "INVALID":
            continue
        try:
            yield getter(str(record_id))
        except ActorError:
            continue


def sessions_for_work(
    state_root: str | Path | None,
    work_id: str,
    *,
    read_only: bool = False,
    snapshot: PortfolioSnapshot | None = None,
) -> list[dict[str, Any]]:
    normalized = validate_prefixed_id(work_id, WORK_ID_PREFIX, label="work item")
    if snapshot is not None:
        return [dict(record) for record in snapshot.sessions_by_work.get(normalized, ())]
    rows = [
        record
        for record in records(session_store(state_root), read_only=read_only)
        if str(dict(record.get("payload") or {}).get("work_id") or "") == normalized
    ]
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("updated_at_utc") or ""),
            str(row.get("record_id") or ""),
        ),
        reverse=True,
    )


def open_sessions(
    state_root: str | Path | None = None,
    *,
    read_only: bool = False,
    snapshot: PortfolioSnapshot | None = None,
) -> list[dict[str, Any]]:
    source = snapshot.session_by_id.values() if snapshot is not None else records(session_store(state_root), read_only=read_only)
    rows = [
        record
        for record in source
        if record.get("status") in SESSION_RESUMABLE_STATUSES
        and str(dict(record.get("payload") or {}).get("session_outcome") or "") != "PROMOTION_REQUESTED"
    ]
    return sorted(
        rows,
        key=lambda row: (str(row.get("updated_at_utc") or ""), str(row.get("record_id") or "")),
        reverse=True,
    )


def open_work(
    state_root: str | Path | None = None,
    *,
    read_only: bool = False,
    snapshot: PortfolioSnapshot | None = None,
) -> list[dict[str, Any]]:
    source = snapshot.work_by_id.values() if snapshot is not None else records(work_store(state_root), read_only=read_only)
    rows = [record for record in source if record.get("status") in WORK_OPEN_STATUSES]
    return sorted(
        rows,
        key=lambda row: (str(row.get("created_at_utc") or ""), str(row.get("record_id") or "")),
    )


def conversation_id_from_url(value: str | None) -> str | None:
    if not value:
        return None
    match = CHATGPT_CONVERSATION_RE.search(str(value))
    return match.group(1) if match else None


def verify_cross_record(
    state_root: str | Path | None,
    *,
    namespace: str,
    record_id: str,
    acceptable_statuses: Sequence[str] | None = None,
    read_only: bool = False,
) -> dict[str, Any]:
    selected_store = store(state_root, namespace)
    row = (selected_store.get_read_only if read_only else selected_store.get)(record_id)
    if acceptable_statuses and str(row.get("status")) not in set(acceptable_statuses):
        raise ActorError(
            f"{namespace} record {record_id} is not in an acceptable state",
            code="cross_record_status_not_acceptable",
            problems=[
                f"Observed status {row.get('status')}; accepted states are {', '.join(acceptable_statuses)}."
            ],
            related_records=[record_id],
        )
    return row


def dependency_graph(
    state_root: str | Path | None = None,
    *,
    read_only: bool = False,
    snapshot: PortfolioSnapshot | None = None,
) -> dict[str, list[str]]:
    if snapshot is not None:
        return {work_id: list(values) for work_id, values in snapshot.dependency_graph.items()}
    graph: dict[str, list[str]] = {}
    for record in records(work_store(state_root), read_only=read_only):
        payload = dict(record.get("payload") or {})
        graph[str(record["record_id"])] = normalize_string_list(payload.get("dependency_work_ids"))
    return graph


def find_dependency_cycle(graph: Mapping[str, Sequence[str]]) -> list[str] | None:
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        if node in visiting:
            try:
                index = stack.index(node)
            except ValueError:
                index = 0
            return [*stack[index:], node]
        if node in visited:
            return None
        visiting.add(node)
        stack.append(node)
        for dependency in graph.get(node, ()):  # dependencies not yet created are handled separately
            cycle = visit(str(dependency))
            if cycle:
                return cycle
        stack.pop()
        visiting.remove(node)
        visited.add(node)
        return None

    for node in sorted(graph):
        cycle = visit(node)
        if cycle:
            return cycle
    return None


def dependency_state(
    state_root: str | Path | None,
    dependency_ids: Sequence[str],
    *,
    read_only: bool = False,
    snapshot: PortfolioSnapshot | None = None,
) -> dict[str, Any]:
    satisfied: list[str] = []
    pending: list[dict[str, str]] = []
    terminal_unsatisfied: list[dict[str, str]] = []
    missing: list[str] = []
    ws = work_store(state_root)
    getter = ws.get_read_only if read_only else ws.get
    for value in normalize_string_list(dependency_ids):
        try:
            if snapshot is not None:
                retained = snapshot.work_by_id.get(value)
                if retained is None:
                    raise ActorError(f"unknown work record: {value}")
                record = dict(retained)
            else:
                record = getter(value)
        except ActorError:
            missing.append(value)
            continue
        status = str(record.get("status") or "UNKNOWN")
        if status == WORK_SUCCESS_STATUS:
            satisfied.append(value)
        elif status in WORK_TERMINAL_STATUSES:
            terminal_unsatisfied.append({"work_id": value, "status": status})
        else:
            pending.append({"work_id": value, "status": status})
    return {
        "satisfied": satisfied,
        "pending": pending,
        "terminal_unsatisfied": terminal_unsatisfied,
        "missing": missing,
        "all_satisfied": not pending and not terminal_unsatisfied and not missing,
    }


def latest_bootstrap_record(state_root: str | Path | None = None, *, read_only: bool = False) -> dict[str, Any] | None:
    rows = sorted(
        records(bootstrap_store(state_root), read_only=read_only),
        key=lambda row: (str(row.get("updated_at_utc") or ""), str(row.get("record_id") or "")),
        reverse=True,
    )
    return rows[0] if rows else None


__all__ = [
    "BOOTSTRAP_ID_PREFIX",
    "BOOTSTRAP_NAMESPACE",
    "BOOTSTRAP_RECORD_TYPE",
    "MISSION_NAMESPACE",
    "QUALIFICATION_NAMESPACE",
    "RESOURCE_NAMESPACE",
    "EVIDENCE_NAMESPACE",
    "PortfolioSnapshot",
    "SESSION_ACTIVE_STATUSES",
    "SESSION_ID_PREFIX",
    "SESSION_NAMESPACE",
    "SESSION_RECORD_TYPE",
    "SESSION_RESUMABLE_STATUSES",
    "SESSION_TERMINAL_STATUSES",
    "WORK_ID_PREFIX",
    "WORK_NAMESPACE",
    "WORK_OPEN_STATUSES",
    "WORK_RECORD_TYPE",
    "WORK_SUCCESS_STATUS",
    "WORK_TERMINAL_STATUSES",
    "bootstrap_store",
    "bootstrap_writer_store",
    "build_portfolio_snapshot",
    "conversation_id_from_url",
    "dependency_graph",
    "dependency_state",
    "find_dependency_cycle",
    "latest_bootstrap_record",
    "mission_store",
    "qualification_store",
    "resource_store",
    "evidence_store",
    "normalize_reference_map",
    "normalize_string_list",
    "parallel_session_ownership_problems",
    "open_sessions",
    "open_work",
    "records",
    "session_store",
    "session_writer_store",
    "sessions_for_work",
    "store",
    "validate_prefixed_id",
    "verify_cross_record",
    "work_store",
    "work_writer_store",
    "work_lifecycle_lock",
    "work_lifecycle_locks",
]
