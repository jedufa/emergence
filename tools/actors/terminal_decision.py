"""Immutable terminal Work decision receipts.

The Work record remains the authoritative lifecycle object.  This module retains
one compact, content-addressed decision receipt alongside it so a later audit can
reconstruct why terminal state was entered without depending on mutable Git
objects, current projections, or a best-effort reading of unrelated revisions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from tools.actors.common import (
    ActorError,
    actor_state_root,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    read_structured,
    require_explicit_schema_version,
    stable_id,
    utc_now,
)
from tools.actors.lifecycle_policy import actor_record_ref

TERMINAL_DECISION_RECEIPT_SCHEMA_VERSION = "eiel_work_terminal_decision_receipt_v1"
TERMINAL_DECISION_RECEIPT_REF_SCHEMA_VERSION = "eiel_work_terminal_decision_receipt_ref_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> Path:
    expected = (
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")
    if path.exists():
        if path.read_bytes() != expected:
            raise ActorError(
                "immutable terminal decision receipt conflict",
                code="work_terminal_decision_receipt_conflict",
                problems=[str(path)],
            )
        return path
    return atomic_write_json(path, value)


def retain_terminal_decision_receipt(
    *,
    state_root: str | Path | None,
    work_record: Mapping[str, Any],
    terminal_status: str,
    decision_payload: Mapping[str, Any],
    deciding_authority: Mapping[str, Any],
    generated_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retain the exact terminal decision material before the final revision.

    ``work_record`` is the authoritative predecessor.  ``decision_payload`` is
    the normalized terminal payload that will be written by Work Manager.  The
    resulting reference is embedded into that final Work revision, closing the
    provenance loop without a circular dependency on the final record hash.
    """

    payload = dict(decision_payload)
    acceptance = payload.get("acceptance_evidence")
    accounting = payload.get("resource_accounting")
    authority = dict(deciding_authority)
    identity_assurance = dict(authority.get("decision_identity_assurance") or {})
    assurance_level = str(identity_assurance.get("level") or "DECLARED").upper()
    if assurance_level not in {"DECLARED", "SCHEDULER_BOUND"}:
        raise ActorError(
            "terminal decision authority assurance is invalid",
            code="work_terminal_decision_authority_invalid",
        )
    if str(authority.get("schema_version") or "") != "eiel_terminal_deciding_authority_v1":
        raise ActorError(
            "terminal decision authority schema is invalid",
            code="work_terminal_decision_authority_invalid",
        )
    if str(authority.get("actor") or "") != "workManager":
        raise ActorError(
            "terminal decision authority actor is invalid",
            code="work_terminal_decision_authority_invalid",
        )
    if str(authority.get("authority_relation") or "") != "work_lifecycle_owner":
        raise ActorError(
            "terminal decision authority relation is invalid",
            code="work_terminal_decision_authority_invalid",
        )
    if str(authority.get("assurance") or "") != "manager_enforced_exact_predecessor_and_evidence":
        raise ActorError(
            "terminal decision authority lifecycle assurance is invalid",
            code="work_terminal_decision_authority_invalid",
        )
    if not str(authority.get("decision_actor_kind") or "").strip():
        raise ActorError(
            "terminal decision authority has no actor kind",
            code="work_terminal_decision_authority_invalid",
        )
    if not str(authority.get("decision_actor_id") or "").strip():
        raise ActorError(
            "terminal decision authority has no actor identity",
            code="work_terminal_decision_authority_invalid",
        )

    receipt: dict[str, Any] = {
        "schema_version": TERMINAL_DECISION_RECEIPT_SCHEMA_VERSION,
        "receipt_id": None,
        "work_id": str(payload.get("work_id") or work_record.get("record_id") or ""),
        "terminal_status": str(terminal_status or "").upper(),
        "completion_policy": str(payload.get("completion_policy") or "").upper(),
        "completion_basis": str(payload.get("completion_basis") or "").upper(),
        "completion_reason": str(payload.get("completion_reason") or "").strip(),
        "accepted": bool(payload.get("accepted")),
        "acceptance_scope": str(payload.get("acceptance_scope") or "").upper() or None,
        "accepted_review_session_id": payload.get("accepted_review_session_id"),
        "predecessor_work_record": actor_record_ref(work_record),
        "deciding_authority": authority,
        "provenance_grade": (
            f"SELF_CONTAINED_ACCEPTED_REPOSITORY_DELIVERY_{assurance_level}"
            if bool(payload.get("accepted"))
            else f"SELF_CONTAINED_NON_ACCEPTING_RECORD_CLOSURE_{assurance_level}"
        ),
        "work_package_sha256": str(dict(payload.get("work_package_binding") or {}).get("sha256") or "") or None,
        "acceptance_evidence_sha256": canonical_sha256(acceptance) if isinstance(acceptance, Mapping) else None,
        "repository_integration_receipt": (
            dict(payload.get("repository_integration_receipt") or {})
            if isinstance(payload.get("repository_integration_receipt"), Mapping)
            else None
        ),
        "resource_accounting_artifact": (
            dict(dict(accounting or {}).get("artifact") or {})
            if isinstance(accounting, Mapping) and isinstance(dict(accounting).get("artifact"), Mapping)
            else None
        ),
        "decided_at_utc": utc_now(generated_at),
    }
    if not receipt["work_id"]:
        raise ActorError("terminal decision receipt has no Work identity", code="work_terminal_decision_identity_missing")
    if receipt["terminal_status"] != "COMPLETED":
        raise ActorError("terminal decision receipt currently supports COMPLETED only", code="work_terminal_decision_status_invalid")
    if not receipt["completion_basis"] or not receipt["completion_reason"]:
        raise ActorError("terminal decision receipt requires basis and reason", code="work_terminal_decision_incomplete")
    receipt["receipt_id"] = stable_id(
        "WORKDEC",
        {key: value for key, value in receipt.items() if key != "receipt_id"},
    )
    material = dict(receipt)
    receipt["receipt_sha256"] = canonical_sha256(material)
    path = actor_state_root(state_root) / "terminal-work-decisions" / f"{receipt['receipt_id']}.json"
    _write_immutable(path, receipt)
    reference = {
        "schema_version": TERMINAL_DECISION_RECEIPT_REF_SCHEMA_VERSION,
        "artifact_id": receipt["receipt_id"],
        "artifact_schema_version": TERMINAL_DECISION_RECEIPT_SCHEMA_VERSION,
        "path": str(path),
        "state_root_locator": path.relative_to(actor_state_root(state_root)).as_posix(),
        "sha256": file_sha256(path),
        "canonical_sha256": canonical_sha256(receipt),
    }
    return receipt, reference


def load_terminal_decision_receipt(
    reference: Mapping[str, Any],
    *,
    state_root: str | Path | None,
) -> dict[str, Any]:
    require_explicit_schema_version(
        reference,
        expected=TERMINAL_DECISION_RECEIPT_REF_SCHEMA_VERSION,
        label="terminal Work decision receipt reference",
    )
    candidates = [Path(str(reference.get("path") or ""))]
    locator = str(reference.get("state_root_locator") or "").strip()
    if locator:
        candidates.append(actor_state_root(state_root) / locator)
    path = next((candidate for candidate in candidates if str(candidate) and candidate.is_file()), None)
    if path is None:
        raise ActorError("terminal Work decision receipt is missing", code="work_terminal_decision_receipt_missing")
    if file_sha256(path) != str(reference.get("sha256") or ""):
        raise ActorError("terminal Work decision receipt file digest mismatch", code="work_terminal_decision_receipt_digest_mismatch")
    value = read_structured(path, name="terminal Work decision receipt")
    require_explicit_schema_version(
        value,
        expected=TERMINAL_DECISION_RECEIPT_SCHEMA_VERSION,
        label="terminal Work decision receipt",
    )
    material = dict(value)
    embedded = str(material.pop("receipt_sha256", "") or "")
    if not _SHA256_RE.fullmatch(embedded) or canonical_sha256(material) != embedded:
        raise ActorError("terminal Work decision receipt canonical digest mismatch", code="work_terminal_decision_receipt_digest_mismatch")
    if str(reference.get("artifact_id") or "") != str(value.get("receipt_id") or ""):
        raise ActorError("terminal Work decision receipt identity mismatch", code="work_terminal_decision_receipt_identity_mismatch")
    authority = value.get("deciding_authority")
    if not isinstance(authority, Mapping):
        raise ActorError(
            "terminal Work decision receipt has no deciding authority",
            code="work_terminal_decision_authority_invalid",
        )
    identity_assurance = authority.get("decision_identity_assurance")
    if not isinstance(identity_assurance, Mapping):
        raise ActorError(
            "terminal Work decision receipt has no identity assurance",
            code="work_terminal_decision_authority_invalid",
        )
    assurance_level = str(identity_assurance.get("level") or "").upper()
    if assurance_level not in {"DECLARED", "SCHEDULER_BOUND"}:
        raise ActorError(
            "terminal Work decision receipt has invalid identity assurance",
            code="work_terminal_decision_authority_invalid",
        )
    expected_grade = (
        f"SELF_CONTAINED_ACCEPTED_REPOSITORY_DELIVERY_{assurance_level}"
        if bool(value.get("accepted"))
        else f"SELF_CONTAINED_NON_ACCEPTING_RECORD_CLOSURE_{assurance_level}"
    )
    if str(value.get("provenance_grade") or "") != expected_grade:
        raise ActorError(
            "terminal Work decision receipt provenance grade does not match identity assurance",
            code="work_terminal_decision_authority_invalid",
        )
    return dict(value)


__all__ = [
    "TERMINAL_DECISION_RECEIPT_REF_SCHEMA_VERSION",
    "TERMINAL_DECISION_RECEIPT_SCHEMA_VERSION",
    "load_terminal_decision_receipt",
    "retain_terminal_decision_receipt",
]
