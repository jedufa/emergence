"""Pure lifecycle policy shared by Work, Session, Workspace, and Scheduler.

This module centralizes cross-record invariants.  It owns no records, launches
nothing, and performs no filesystem writes.  Managers remain the authorities
for their own records; this module merely prevents each caller from inventing a
slightly different definition of freshness, identity assurance, or readiness.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from tools.actors.canonical import canonical_sha256
from tools.actors.common import ActorError
from tools.actors.delivery_semantics import delivery_commit as delivery_commit_value
from tools.actors.portfolio import WORK_OPEN_STATUSES

IDENTITY_ASSURANCE_DECLARED = "DECLARED"
IDENTITY_ASSURANCE_SCHEDULER_BOUND = "SCHEDULER_BOUND"
IDENTITY_ASSURANCE_LEVELS = (
    IDENTITY_ASSURANCE_DECLARED,
    IDENTITY_ASSURANCE_SCHEDULER_BOUND,
)
DECISION_SNAPSHOT_SCHEMA_VERSION = "eiel_lifecycle_decision_snapshot_v2"
LEGACY_DECISION_SNAPSHOT_SCHEMA_VERSION = "eiel_lifecycle_decision_snapshot_v1"
RECORD_REF_SCHEMA_VERSION = "eiel_actor_record_ref_v1"
AUTHORITATIVE_REVIEW_BINDING_SCHEMA_VERSION = "eiel_authoritative_review_binding_v1"
CONTEXT_CONTRACT_FIELDS = ("read_contract", "output_policy", "execution_budget")

WORK_EXECUTION_REF_SCHEMA_VERSION = "eiel_work_execution_ref_v1"
WRA_CONTINUATION_CAPSULE_SCHEMA_VERSION = "eiel_wra_continuation_capsule_v1"


def work_execution_material(work_record: Mapping[str, Any]) -> dict[str, Any]:
    """Return only immutable Work-contract facts relevant to a Session.

    Lifecycle labels, dependencies, holds, acceptance projections and display
    guidance deliberately do not participate.  They describe portfolio state;
    they are not execution permission and must not revoke an otherwise current
    Context Receipt after a safe administrative reconciliation.
    """

    payload = dict(work_record.get("payload") or {})
    binding = dict(payload.get("work_package_binding") or {})
    return {
        "work_id": str(work_record.get("record_id") or payload.get("work_id") or ""),
        "work_package": {
            "sha256": str(binding.get("sha256") or "") or None,
            "package_version": int(binding.get("package_version") or 0) or None,
            "recommended_authority": str(binding.get("recommended_authority") or "") or None,
            "uncertainty_class": str(binding.get("uncertainty_class") or "") or None,
        },
        "completion_policy": str(payload.get("completion_policy") or ""),
    }


def work_execution_ref(work_record: Mapping[str, Any]) -> dict[str, Any]:
    material = work_execution_material(work_record)
    return {
        "schema_version": WORK_EXECUTION_REF_SCHEMA_VERSION,
        "material": material,
        "sha256": canonical_sha256(material),
    }


def wra_continuation_capsule(session_record: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(session_record.get("payload") or {})
    accounting = dict(payload.get("resource_accounting") or {})
    captures = [dict(row) for row in (accounting.get("captures") or []) if isinstance(row, Mapping)]
    problems: list[str] = []
    for capture in captures:
        status = str(capture.get("status") or "")
        if status in {"UNAVAILABLE", "STARTED"}:
            problems.append(f"capture {capture.get('capture_id') or '<unknown>'}: {status}")
    latest = (
        dict(accounting.get("latest_usage_artifact") or {})
        if isinstance(accounting.get("latest_usage_artifact"), Mapping)
        else None
    )
    return {
        "schema_version": WRA_CONTINUATION_CAPSULE_SCHEMA_VERSION,
        "wra_status": (
            "UNAVAILABLE" if any(str(row.get("status") or "") == "UNAVAILABLE" for row in captures)
            else "OPEN" if accounting.get("active_capture_id")
            else "RECORDED" if captures or latest
            else "NOT_RECORDED"
        ),
        "problem_count": len(problems),
        "problems": problems,
        "latest_usage_artifact_ref": latest,
    }


def actor_record_ref(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable coordinates used in a multi-record decision."""

    return {
        "schema_version": RECORD_REF_SCHEMA_VERSION,
        "record_id": str(record.get("record_id") or ""),
        "revision": int(record.get("revision") or 0),
        "record_sha256": str(record.get("record_sha256") or ""),
        "status": str(record.get("status") or ""),
    }


def require_record_ref(reference: Mapping[str, Any], record: Mapping[str, Any], *, label: str) -> None:
    expected = actor_record_ref(record)
    problems = []
    if reference.get("schema_version") != RECORD_REF_SCHEMA_VERSION:
        problems.append(
            f"{label} schema_version changed: expected {RECORD_REF_SCHEMA_VERSION!r}, "
            f"observed {reference.get('schema_version')!r}"
        )
    for field in ("record_id", "revision", "record_sha256", "status"):
        if reference.get(field) != expected.get(field):
            problems.append(
                f"{label} {field} changed: expected {reference.get(field)!r}, observed {expected.get(field)!r}"
            )
    if problems:
        raise ActorError(
            f"{label} changed after the lifecycle decision was prepared",
            code="lifecycle_decision_stale",
            problems=problems,
            related_records=[str(expected.get("record_id") or "")],
        )


def lifecycle_decision_snapshot(
    *,
    work_record: Mapping[str, Any],
    session_record: Mapping[str, Any],
    package_sha256: str | None,
    implementation_record: Mapping[str, Any] | None = None,
    review_record: Mapping[str, Any] | None = None,
    delivery_commit: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": DECISION_SNAPSHOT_SCHEMA_VERSION,
        "work": actor_record_ref(work_record),
        "session": actor_record_ref(session_record),
        "package_sha256": str(package_sha256 or "") or None,
        "delivery_commit": str(delivery_commit or "") or None,
        "delivery": actor_record_ref(implementation_record) if implementation_record else None,
        "review": actor_record_ref(review_record) if review_record else None,
    }
    payload["snapshot_sha256"] = canonical_sha256(payload)
    return payload


def load_lifecycle_decision_snapshot(raw: Any) -> dict[str, Any]:
    """Strictly load a current or retained historical decision snapshot."""

    if not isinstance(raw, Mapping):
        raise ActorError("lifecycle decision snapshot must be an object", code="schema_record_malformed")
    errors = verify_lifecycle_decision_snapshot(raw)
    if errors:
        raise ActorError(
            "lifecycle decision snapshot is invalid",
            code="schema_record_malformed",
            problems=errors,
        )
    return dict(raw)


def verify_lifecycle_decision_snapshot(raw: Any) -> list[str]:
    """Verify the immutable seal and exact reference shape of a decision snapshot."""

    if not isinstance(raw, Mapping):
        return ["lifecycle decision snapshot must be an object"]
    snapshot = dict(raw)
    errors: list[str] = []
    schema_version = snapshot.get("schema_version")
    if schema_version not in {DECISION_SNAPSHOT_SCHEMA_VERSION, LEGACY_DECISION_SNAPSHOT_SCHEMA_VERSION}:
        errors.append("lifecycle decision snapshot has unsupported schema_version")
    expected = str(snapshot.get("snapshot_sha256") or "")
    material = dict(snapshot)
    material.pop("snapshot_sha256", None)
    actual = canonical_sha256(material)
    if not expected or expected != actual:
        errors.append("lifecycle decision snapshot digest mismatch")
    for field in ("work", "session"):
        reference = snapshot.get(field)
        if not isinstance(reference, Mapping):
            errors.append(f"lifecycle decision snapshot {field} reference is missing")
            continue
        if reference.get("schema_version") != RECORD_REF_SCHEMA_VERSION:
            errors.append(f"lifecycle decision snapshot {field} reference has unsupported schema_version")
        if not str(reference.get("record_id") or ""):
            errors.append(f"lifecycle decision snapshot {field} reference has no record_id")
        if int(reference.get("revision") or 0) < 1:
            errors.append(f"lifecycle decision snapshot {field} reference has invalid revision")
        if not str(reference.get("record_sha256") or ""):
            errors.append(f"lifecycle decision snapshot {field} reference has no record_sha256")
    optional_fields = (
        ("delivery", "review")
        if schema_version == DECISION_SNAPSHOT_SCHEMA_VERSION
        else ("implementation", "review")
    )
    for field in optional_fields:
        reference = snapshot.get(field)
        if reference is None:
            continue
        if not isinstance(reference, Mapping):
            errors.append(f"lifecycle decision snapshot {field} reference must be an object or null")
            continue
        if reference.get("schema_version") != RECORD_REF_SCHEMA_VERSION:
            errors.append(f"lifecycle decision snapshot {field} reference has unsupported schema_version")
        if not str(reference.get("record_id") or "") or int(reference.get("revision") or 0) < 1:
            errors.append(f"lifecycle decision snapshot {field} reference has invalid identity")
    return list(dict.fromkeys(errors))


def codex_identities(record: Mapping[str, Any]) -> set[str]:
    payload = dict(record.get("payload") or {})
    identities = {
        str(value).strip()
        for value in (payload.get("original_codex_id"), payload.get("current_codex_id"))
        if str(value or "").strip()
    }
    for participant in payload.get("codex_participants") or []:
        if isinstance(participant, Mapping) and str(participant.get("codex_id") or "").strip():
            identities.add(str(participant["codex_id"]).strip())
    return identities


def declared_identity_assurance(*, codex_id: str | None, source: str = "manual_cli") -> dict[str, Any]:
    return {
        "level": IDENTITY_ASSURANCE_DECLARED,
        "source": str(source),
        "codex_id": str(codex_id or "") or None,
        "worker_id": None,
        "lease_id": None,
        "override_allowed": True,
    }


def scheduler_identity_assurance(
    *, codex_id: str | None, worker_id: str | None, lease_id: str | None
) -> dict[str, Any]:
    return {
        "level": IDENTITY_ASSURANCE_SCHEDULER_BOUND,
        "source": "scheduler_assignment",
        "codex_id": str(codex_id or "") or None,
        "worker_id": str(worker_id or "") or None,
        "lease_id": str(lease_id or "") or None,
        "override_allowed": False,
    }


def normalize_identity_assurance(raw: Any, *, fallback_codex_id: str | None = None) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return declared_identity_assurance(codex_id=fallback_codex_id, source="legacy_or_manual")
    level = str(raw.get("level") or IDENTITY_ASSURANCE_DECLARED).strip().upper()
    if level not in IDENTITY_ASSURANCE_LEVELS:
        raise ActorError(
            "identity assurance has an unsupported explicit level",
            code="identity_assurance_unsupported",
            problems=[
                f"Observed: {level or '<empty>'}",
                "Supported: " + ", ".join(IDENTITY_ASSURANCE_LEVELS),
            ],
        )
    return {
        "level": level,
        "source": str(raw.get("source") or "legacy_or_manual"),
        "codex_id": str(raw.get("codex_id") or fallback_codex_id or "") or None,
        "worker_id": str(raw.get("worker_id") or "") or None,
        "lease_id": str(raw.get("lease_id") or "") or None,
        "override_allowed": bool(raw.get("override_allowed", level == IDENTITY_ASSURANCE_DECLARED)),
    }


def require_open_work(work_record: Mapping[str, Any], *, action: str) -> None:
    status = str(work_record.get("status") or "")
    if status not in WORK_OPEN_STATUSES:
        raise ActorError(
            f"cannot {action} while parent Work is {status or 'unknown'}",
            code="session_parent_work_terminal",
            related_records=[str(work_record.get("record_id") or "")],
        )


def context_contract_status(package: Mapping[str, Any]) -> dict[str, Any]:
    explicit = [field for field in CONTEXT_CONTRACT_FIELDS if isinstance(package.get(field), Mapping)]
    missing = [field for field in CONTEXT_CONTRACT_FIELDS if field not in explicit]
    read_contract = dict(package.get("read_contract") or {})
    has_entry_guidance = any(
        read_contract.get(field)
        for field in ("entry_files", "required_documents", "search_roots", "known_symbols")
    )
    return {
        "explicit_fields": explicit,
        "missing_fields": missing,
        "has_entry_guidance": bool(has_entry_guidance),
        "delegated_ready": not missing and bool(has_entry_guidance),
    }


def duplicate_delivery_claims(
    sessions: Sequence[Mapping[str, Any]],
    *,
    current_session_id: str,
    work_id: str,
    package_sha256: str,
    delivery_commit: str,
) -> list[str]:
    conflicts: list[str] = []
    for row in sessions:
        if str(row.get("record_id") or "") == current_session_id:
            continue
        payload = dict(row.get("payload") or {})
        evidence = dict(payload.get("delivery_evidence") or {})
        if row.get("status") != "COMPLETED":
            continue
        if str(payload.get("work_id") or "") != work_id:
            continue
        if str(evidence.get("work_package_sha256") or "") != package_sha256:
            continue
        if delivery_commit_value(evidence) == delivery_commit:
            conflicts.append(str(row.get("record_id") or ""))
    return sorted(conflicts)
