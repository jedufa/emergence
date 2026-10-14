"""Run artifact lifecycle contracts and fail-closed storage utilities.

This module is deliberately a contract/observation layer.  It does not own
campaign execution, scientific semantics, qualification decisions, or
destructive cleanup.  Campaign plans bind the contract; the existing runner,
evidence closeout, and prune tooling consume the resulting receipts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
from pathlib import Path
import shutil
from typing import Any, Final

from .canonical import canonical_sha256, pretty_json
from .storage_availability import (
    DATA2_FILESYSTEM_ROOT,
    DATA2_LIFECYCLE_MANAGEMENT_ROOT,
    StorageAvailabilityPolicyError,
    build_storage_availability_snapshot_v1,
    evidence_backed_execution_reserve_v1,
    load_data2_storage_availability_policy_v1,
    observe_filesystem_identity_v1,
    storage_availability_policy_binding_v1,
    validate_data2_run_root_v1,
    validate_data2_storage_availability_policy_v1,
    validate_filesystem_identity_v1,
)


LIFECYCLE_SCHEMA_VERSION: Final[str] = "eiel_run_artifact_lifecycle_contract_v1"
DISPOSITION_SCHEMA_VERSION: Final[str] = "eiel_artifact_disposition_manifest_v1"
LEGACY_RECONSTRUCTION_SCHEMA_VERSION: Final[str] = (
    "eiel_legacy_evidence_reconstruction_v1"
)
LEGACY_REVIEW_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_legacy_evidence_review_receipt_v1"
)
STORAGE_RECEIPT_SCHEMA_VERSION: Final[str] = "eiel_storage_disposition_receipt_v1"
RUN_PASSPORT_SCHEMA_VERSION: Final[str] = "eiel_run_passport_v1"
STORAGE_FAILURE_SCHEMA_VERSION: Final[str] = "eiel_storage_budget_failure_receipt_v1"
STORAGE_GUARD_SCHEMA_VERSION: Final[str] = "eiel_run_storage_guard_v1"
DATA2_STORAGE_RESERVATION_SCHEMA_VERSION: Final[str] = (
    "eiel_data2_storage_reservation_v1"
)
PROCESS_OWNER_IDENTITY_SCHEMA_VERSION: Final[str] = "eiel_process_owner_identity_v1"
RUN_FAILURE_SCHEMA_VERSION: Final[str] = "eiel_run_artifact_failure_receipt_v1"
PROGRESSIVE_STORAGE_RESERVATION_MODE_V1: Final[str] = (
    "bounded_incremental_guarded_v1"
)

CHECKPOINT_TRANSIT: Final[str] = "CHECKPOINT_TRANSIT"
SCIENCE_COMPACT: Final[str] = "SCIENCE_COMPACT"
QUALIFICATION: Final[str] = "QUALIFICATION"
DEBUG_FORENSIC: Final[str] = "DEBUG_FORENSIC"
LIFECYCLE_PROFILES: Final[tuple[str, ...]] = (
    CHECKPOINT_TRANSIT,
    SCIENCE_COMPACT,
    QUALIFICATION,
    DEBUG_FORENSIC,
)

RETENTION_CLASSES: Final[tuple[str, ...]] = (
    "R0",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "RX",
)
DECISION_EVIDENCE_CLASSES: Final[tuple[str, ...]] = ("A", "B", "C")
OUTPUT_PROFILES: Final[tuple[str, ...]] = (
    "compact_default",
    "compact_funnel",
    "fast_scan",
    "repeat_parity",
    "debug_event_trace",
    "full_replay",
)
PROFILE_OUTPUT_BINDINGS: Final[dict[str, str]] = {
    CHECKPOINT_TRANSIT: "compact_default",
    SCIENCE_COMPACT: "compact_default",
    QUALIFICATION: "compact_default",
    DEBUG_FORENSIC: "debug_event_trace",
}
CHECKPOINT_ROLES: Final[tuple[str, ...]] = (
    "intermediate",
    "transit_durable",
    "terminal",
    "branch_origin",
    "qualification_required",
)
CHECKPOINT_AUTHORITIES: Final[tuple[str, ...]] = (
    "legacy_generation",
    "absolute_tick",
)
UNKNOWN_ARTIFACT_POLICY: Final[str] = "RX_BLOCKS_AUTO_DELETE"
UNKNOWN_ARTIFACT_SCOPES: Final[tuple[str, ...]] = ("global", "exact_object")
LEGACY_RECONSTRUCTION_STATUSES: Final[tuple[str, ...]] = (
    "REQUIRES_REVIEW",
    "ACCEPTED",
    "REJECTED",
)
LEGACY_REVIEW_STATUSES: Final[tuple[str, ...]] = ("required", "accepted", "rejected")
LEGACY_REVIEW_DECISIONS: Final[tuple[str, ...]] = (
    "KEEP_ACTIVE",
    "HOLD_DIAGNOSTIC",
    "RELEASE",
    "RETAIN_DURABLE",
    "RX",
)
LIFECYCLE_REQUIRED_PRUNE_PREDICATES: Final[tuple[str, ...]] = (
    "valid_disposition_manifest",
    "unknown_artifacts_resolved",
    "no_live_references",
    "audit_hold_released_or_not_required",
    "qualification_hold_released_or_not_required",
    "replacement_verified_or_not_required",
    "explicit_checkpoint_roles_resolved",
    "storage_disposition_receipt",
)

_ID_RE = re.compile(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*\Z")
_CHECKPOINT_RE = re.compile(r"(?:bundle_gen|generation[_-]?)(\d+)", re.IGNORECASE)


class ArtifactLifecycleError(ValueError):
    """A lifecycle contract, disposition, or storage receipt is invalid."""


def closeout_report_only_run_v1(**kwargs: Any) -> dict[str, Any]:
    """Terminal report-only policy through the existing lifecycle facade."""
    from .report_only_closeout import closeout_report_only_run_v1 as closeout

    return closeout(**kwargs)


def read_report_markdown_v1(path: Path) -> str:
    """Read the report body used by a resumable lifecycle closeout."""
    from .report_only_closeout import read_report_markdown_v1 as read_report

    return read_report(path)


class StorageBudgetExceeded(RuntimeError):
    """A run crossed a declared lifecycle storage safety boundary."""

    def __init__(self, message: str, *, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.receipt = dict(receipt)


def _exact(value: Mapping[str, Any], allowed: set[str], *, path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ArtifactLifecycleError(f"{path} contains unsupported fields: {unknown}")


def _text(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactLifecycleError(f"{path} must be a non-empty string")
    return value.strip()


def _identifier(value: Any, *, path: str) -> str:
    result = _text(value, path=path)
    if _ID_RE.fullmatch(result) is None:
        raise ArtifactLifecycleError(f"{path} must be a stable identifier")
    return result


def _boolean(value: Any, *, path: str) -> bool:
    if type(value) is not bool:
        raise ArtifactLifecycleError(f"{path} must be boolean")
    return value


def _bytes(value: Any, *, path: str) -> int:
    if type(value) is not int or value <= 0:
        raise ArtifactLifecycleError(f"{path} must be a positive integer byte quota")
    return value


def _string_list(value: Any, *, path: str, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ArtifactLifecycleError(f"{path} must be an array of strings")
    if len(value) != len(set(value)):
        raise ArtifactLifecycleError(f"{path} must not contain duplicates")
    if nonempty and not value:
        raise ArtifactLifecycleError(f"{path} must not be empty")
    return list(value)


def _optional_sha256(value: Any, *, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ArtifactLifecycleError(f"{path} must be a lowercase SHA-256 or null")
    return value


def validate_legacy_reconstruction_metadata_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the review gate attached to a reconstructed legacy manifest.

    This metadata is deliberately small and explicit.  It records why a
    pre-lifecycle root was reconstructed, but it is not itself release
    authority.  Release/archive authority is unlocked only by an accepted
    review receipt bound to the candidate manifest.
    """

    if not isinstance(value, Mapping):
        raise ArtifactLifecycleError("legacy_reconstruction must be an object")
    raw = dict(value)
    _exact(
        raw,
        {
            "schema_version",
            "status",
            "source_manifest_status",
            "method",
            "created_at_utc",
            "scope",
            "evidence_basis",
            "review",
        },
        path="legacy_reconstruction",
    )
    if raw.get("schema_version") != LEGACY_RECONSTRUCTION_SCHEMA_VERSION:
        raise ArtifactLifecycleError("legacy reconstruction schema mismatch")
    status = _text(raw.get("status"), path="legacy_reconstruction.status")
    if status not in LEGACY_RECONSTRUCTION_STATUSES:
        raise ArtifactLifecycleError("legacy reconstruction status is unsupported")
    if raw.get("source_manifest_status") != "ABSENT":
        raise ArtifactLifecycleError(
            "legacy reconstruction source_manifest_status must be ABSENT"
        )
    method = _text(raw.get("method"), path="legacy_reconstruction.method")
    created = _text(
        raw.get("created_at_utc"), path="legacy_reconstruction.created_at_utc"
    )

    scope = raw.get("scope")
    if not isinstance(scope, Mapping):
        raise ArtifactLifecycleError("legacy_reconstruction.scope must be an object")
    scope = dict(scope)
    _exact(
        scope,
        {
            "max_objects",
            "max_bytes",
            "inventory_complete",
            "candidate_object_count",
            "candidate_bytes",
        },
        path="legacy_reconstruction.scope",
    )
    for field in (
        "max_objects",
        "max_bytes",
        "candidate_object_count",
        "candidate_bytes",
    ):
        if type(scope.get(field)) is not int or int(scope[field]) < 0:
            raise ArtifactLifecycleError(
                f"legacy_reconstruction.scope.{field} must be a non-negative integer"
            )
    if int(scope["max_objects"]) < 1 or int(scope["max_bytes"]) < 1:
        raise ArtifactLifecycleError(
            "legacy reconstruction scope limits must be positive"
        )
    scope["inventory_complete"] = _boolean(
        scope.get("inventory_complete"),
        path="legacy_reconstruction.scope.inventory_complete",
    )

    evidence_basis = raw.get("evidence_basis")
    if not isinstance(evidence_basis, list):
        raise ArtifactLifecycleError(
            "legacy_reconstruction.evidence_basis must be an array"
        )
    normalized_basis: list[dict[str, str]] = []
    for index, item in enumerate(evidence_basis):
        if not isinstance(item, Mapping):
            raise ArtifactLifecycleError(
                f"legacy reconstruction evidence basis {index} is malformed"
            )
        row = dict(item)
        _exact(
            row,
            {"kind", "path", "sha256"},
            path=f"legacy_reconstruction.evidence_basis[{index}]",
        )
        normalized_basis.append(
            {
                "kind": _text(
                    row.get("kind"),
                    path=f"legacy_reconstruction.evidence_basis[{index}].kind",
                ),
                "path": _text(
                    row.get("path"),
                    path=f"legacy_reconstruction.evidence_basis[{index}].path",
                ),
                "sha256": _text(
                    row.get("sha256"),
                    path=f"legacy_reconstruction.evidence_basis[{index}].sha256",
                ),
            }
        )
        if re.fullmatch(r"[0-9a-f]{64}", normalized_basis[-1]["sha256"]) is None:
            raise ArtifactLifecycleError(
                f"legacy reconstruction evidence basis {index} has an invalid SHA-256"
            )

    review = raw.get("review")
    if not isinstance(review, Mapping):
        raise ArtifactLifecycleError("legacy_reconstruction.review must be an object")
    review = dict(review)
    _exact(
        review,
        {
            "required",
            "status",
            "receipt_sha256",
            "reviewer",
            "accepted_at_utc",
            "allow_automatic_release",
        },
        path="legacy_reconstruction.review",
    )
    normalized_review = {
        "required": _boolean(
            review.get("required"), path="legacy_reconstruction.review.required"
        ),
        "status": _text(
            review.get("status"), path="legacy_reconstruction.review.status"
        ),
        "receipt_sha256": _optional_sha256(
            review.get("receipt_sha256"),
            path="legacy_reconstruction.review.receipt_sha256",
        ),
        "reviewer": None
        if review.get("reviewer") is None
        else _text(
            review.get("reviewer"), path="legacy_reconstruction.review.reviewer"
        ),
        "accepted_at_utc": None
        if review.get("accepted_at_utc") is None
        else _text(
            review.get("accepted_at_utc"),
            path="legacy_reconstruction.review.accepted_at_utc",
        ),
        "allow_automatic_release": _boolean(
            review.get("allow_automatic_release"),
            path="legacy_reconstruction.review.allow_automatic_release",
        ),
    }
    if normalized_review["status"] not in LEGACY_REVIEW_STATUSES:
        raise ArtifactLifecycleError(
            "legacy reconstruction review status is unsupported"
        )
    if status == "REQUIRES_REVIEW":
        if (
            normalized_review["status"] != "required"
            or normalized_review["required"] is not True
        ):
            raise ArtifactLifecycleError(
                "REQUIRES_REVIEW legacy reconstruction must retain a required review"
            )
        if normalized_review["receipt_sha256"] is not None:
            raise ArtifactLifecycleError(
                "REQUIRES_REVIEW legacy reconstruction cannot carry a review receipt"
            )
    elif status == "ACCEPTED":
        if (
            normalized_review["status"] != "accepted"
            or normalized_review["required"] is not True
        ):
            raise ArtifactLifecycleError(
                "ACCEPTED legacy reconstruction must carry an accepted review"
            )
        if normalized_review["receipt_sha256"] is None:
            raise ArtifactLifecycleError(
                "ACCEPTED legacy reconstruction must bind a review receipt"
            )
        if (
            normalized_review["reviewer"] is None
            or normalized_review["accepted_at_utc"] is None
        ):
            raise ArtifactLifecycleError(
                "ACCEPTED legacy reconstruction must identify its reviewer and time"
            )
    else:
        if (
            normalized_review["status"] != "rejected"
            or normalized_review["required"] is not True
        ):
            raise ArtifactLifecycleError(
                "REJECTED legacy reconstruction must retain a rejected review"
            )

    return {
        "schema_version": LEGACY_RECONSTRUCTION_SCHEMA_VERSION,
        "status": status,
        "source_manifest_status": "ABSENT",
        "method": method,
        "created_at_utc": created,
        "scope": scope,
        "evidence_basis": normalized_basis,
        "review": normalized_review,
    }


def validate_legacy_evidence_review_receipt_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an explicit semantic review of a reconstructed legacy root."""

    if not isinstance(value, Mapping):
        raise ArtifactLifecycleError("legacy evidence review receipt must be an object")
    raw = dict(value)
    supplied = raw.pop("receipt_sha256", None)
    _exact(
        raw,
        {
            "schema_version",
            "manifest_sha256",
            "runtime_root",
            "decision",
            "reviewer",
            "reviewed_at_utc",
            "evidence_references",
            "artifact_decisions",
            "terminal_status",
            "live_reference_status",
            "audit_hold_status",
            "qualification_hold_status",
            "allow_automatic_release",
        },
        path="legacy_evidence_review_receipt",
    )
    if raw.get("schema_version") != LEGACY_REVIEW_RECEIPT_SCHEMA_VERSION:
        raise ArtifactLifecycleError("legacy evidence review receipt schema mismatch")
    manifest_sha256 = raw.get("manifest_sha256")
    if (
        not isinstance(manifest_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None
    ):
        raise ArtifactLifecycleError("legacy review receipt manifest_sha256 is invalid")
    decision = _text(raw.get("decision"), path="legacy review receipt decision")
    if decision not in {"accepted", "rejected"}:
        raise ArtifactLifecycleError("legacy review receipt decision is unsupported")
    _text(raw.get("reviewer"), path="legacy review receipt reviewer")
    _text(raw.get("reviewed_at_utc"), path="legacy review receipt reviewed_at_utc")
    evidence_references = raw.get("evidence_references")
    if not isinstance(evidence_references, list):
        raise ArtifactLifecycleError(
            "legacy review receipt evidence_references must be an array"
        )
    normalized_refs: list[dict[str, str]] = []
    for index, item in enumerate(evidence_references):
        if not isinstance(item, Mapping):
            raise ArtifactLifecycleError(
                f"legacy review evidence reference {index} is malformed"
            )
        row = dict(item)
        _exact(
            row,
            {"kind", "path", "sha256"},
            path=f"legacy review evidence_references[{index}]",
        )
        normalized = {
            "kind": _text(
                row.get("kind"), path=f"legacy review evidence_references[{index}].kind"
            ),
            "path": _text(
                row.get("path"), path=f"legacy review evidence_references[{index}].path"
            ),
            "sha256": _text(
                row.get("sha256"),
                path=f"legacy review evidence_references[{index}].sha256",
            ),
        }
        if re.fullmatch(r"[0-9a-f]{64}", normalized["sha256"]) is None:
            raise ArtifactLifecycleError(
                f"legacy review evidence reference {index} has an invalid SHA-256"
            )
        normalized_refs.append(normalized)
    if decision == "accepted" and not normalized_refs:
        raise ArtifactLifecycleError(
            "accepted legacy review must cite evidence references"
        )

    artifact_decisions = raw.get("artifact_decisions")
    if not isinstance(artifact_decisions, list):
        raise ArtifactLifecycleError(
            "legacy review artifact_decisions must be an array"
        )
    normalized_decisions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(artifact_decisions):
        if not isinstance(item, Mapping):
            raise ArtifactLifecycleError(
                f"legacy artifact decision {index} is malformed"
            )
        row = dict(item)
        _exact(
            row,
            {
                "path",
                "disposition",
                "retention_class",
                "release_eligible",
                "reason",
                "live_reference_status",
                "replacement_verified",
                "decision_evidence_class",
                "replacement_artifact_identity",
                "consumer_references",
                "reproducible",
            },
            path=f"legacy review artifact_decisions[{index}]",
        )
        path = (
            _text(
                row.get("path"), path=f"legacy review artifact_decisions[{index}].path"
            )
            .replace("\\", "/")
            .strip("/")
        )
        if not path or path in seen:
            raise ArtifactLifecycleError(
                "legacy review artifact decision paths must be unique"
            )
        seen.add(path)
        disposition = _text(
            row.get("disposition"),
            path=f"legacy review artifact_decisions[{index}].disposition",
        )
        if disposition not in LEGACY_REVIEW_DECISIONS:
            raise ArtifactLifecycleError(
                "legacy review artifact disposition is unsupported"
            )
        retention_class = _text(
            row.get("retention_class"),
            path=f"legacy review artifact_decisions[{index}].retention_class",
        )
        if retention_class not in RETENTION_CLASSES:
            raise ArtifactLifecycleError(
                "legacy review artifact retention class is unsupported"
            )
        release_eligible = _boolean(
            row.get("release_eligible"),
            path=f"legacy review artifact_decisions[{index}].release_eligible",
        )
        replacement_verified = _boolean(
            row.get("replacement_verified"),
            path=f"legacy review artifact_decisions[{index}].replacement_verified",
        )
        consumer_references = row.get("consumer_references")
        if not isinstance(consumer_references, list) or not all(
            isinstance(item, str) and item.strip() for item in consumer_references
        ):
            raise ArtifactLifecycleError(
                f"legacy review artifact_decisions[{index}].consumer_references must be an array of strings"
            )
        if len(consumer_references) != len(set(consumer_references)):
            raise ArtifactLifecycleError(
                f"legacy review artifact_decisions[{index}].consumer_references must be unique"
            )
        reproducible = _boolean(
            row.get("reproducible"),
            path=f"legacy review artifact_decisions[{index}].reproducible",
        )
        if disposition == "RELEASE":
            if retention_class not in {"R0", "R1", "R2"} or not release_eligible:
                raise ArtifactLifecycleError(
                    "legacy RELEASE decisions require eligible R0-R2 rows"
                )
            if retention_class == "R2" and not replacement_verified:
                raise ArtifactLifecycleError(
                    "legacy R2 release requires a verified replacement"
                )
        elif release_eligible:
            raise ArtifactLifecycleError(
                "only RELEASE decisions may be release eligible"
            )
        normalized_decisions.append(
            {
                "path": path,
                "disposition": disposition,
                "retention_class": retention_class,
                "release_eligible": release_eligible,
                "reason": _text(
                    row.get("reason"),
                    path=f"legacy review artifact_decisions[{index}].reason",
                ),
                "live_reference_status": _text(
                    row.get("live_reference_status"),
                    path=f"legacy review artifact_decisions[{index}].live_reference_status",
                ),
                "replacement_verified": replacement_verified,
                "decision_evidence_class": row.get("decision_evidence_class"),
                "replacement_artifact_identity": row.get(
                    "replacement_artifact_identity"
                ),
                "consumer_references": list(consumer_references),
                "reproducible": reproducible,
            }
        )
    for status_name in (
        "terminal_status",
        "live_reference_status",
        "audit_hold_status",
        "qualification_hold_status",
    ):
        raw[status_name] = _text(
            raw.get(status_name), path=f"legacy review receipt {status_name}"
        )
    raw["allow_automatic_release"] = _boolean(
        raw.get("allow_automatic_release"),
        path="legacy review receipt allow_automatic_release",
    )
    if raw["allow_automatic_release"] and raw["decision"] == "rejected":
        raise ArtifactLifecycleError(
            "rejected legacy review cannot enable automatic release"
        )
    if not isinstance(supplied, str) or supplied != canonical_sha256(raw):
        raise ArtifactLifecycleError("legacy review receipt SHA-256 mismatch")
    return {
        **raw,
        "evidence_references": normalized_refs,
        "artifact_decisions": normalized_decisions,
        "receipt_sha256": supplied,
    }


def legacy_reconstruction_is_accepted_v1(manifest: Mapping[str, Any]) -> bool:
    metadata = manifest.get("legacy_reconstruction")
    if not isinstance(metadata, Mapping):
        return True
    return (
        metadata.get("status") == "ACCEPTED"
        and isinstance(metadata.get("review"), Mapping)
        and metadata["review"].get("status") == "accepted"
        and isinstance(metadata["review"].get("receipt_sha256"), str)
    )


LIFECYCLE_EMERGENCY_RESERVE_CAP_BYTES: Final[int] = 4 * 1024**3


def _partition_budget(total: int, profile: str) -> dict[str, int]:
    if total < 4:
        raise ArtifactLifecycleError(
            "storage.total_max_bytes must be at least four bytes"
        )
    ratios = {
        CHECKPOINT_TRANSIT: (20, 60, 15),
        SCIENCE_COMPACT: (20, 50, 25),
        QUALIFICATION: (45, 30, 20),
        DEBUG_FORENSIC: (60, 20, 15),
    }[profile]
    # Keep the contract's internal failure allowance bounded by an observed
    # filesystem-scale margin.  The former total/20 rule turned a large
    # logical quota into tens or hundreds of GiB of admission floor; the
    # data2 availability policy now carries the measured execution reserve.
    reserve = max(1, min(total // 20, LIFECYCLE_EMERGENCY_RESERVE_CAP_BYTES))
    remaining = total - reserve
    scratch = max(1, remaining * ratios[0] // 100)
    checkpoint = max(1, remaining * ratios[1] // 100)
    durable = remaining - scratch - checkpoint
    if durable <= 0:
        raise ArtifactLifecycleError("storage quotas cannot be partitioned safely")
    return {
        "total_max_bytes": total,
        "scratch_max_bytes": scratch,
        "checkpoint_max_bytes": checkpoint,
        "durable_max_bytes": durable,
        "emergency_reserve_bytes": reserve,
    }


def _checkpoint_roles(
    schedule: Sequence[int],
    *,
    profile: str,
    authority: str = "legacy_generation",
    time_authority: str | None = None,
) -> list[dict[str, Any]]:
    if time_authority is not None:
        authority = (
            "absolute_tick"
            if time_authority == "absolute_tick_v1"
            else "legacy_generation"
        )
    if authority not in CHECKPOINT_AUTHORITIES:
        raise ArtifactLifecycleError("checkpoint authority is unsupported")
    ordered = sorted({int(value) for value in schedule})
    if not ordered:
        return []
    terminal = ordered[-1]
    rows: list[dict[str, Any]] = []
    for coordinate in ordered:
        role = "terminal" if coordinate == terminal else "intermediate"
        if profile == QUALIFICATION and coordinate == terminal:
            role = "qualification_required"
        retention_class = (
            "R4" if role in {"terminal", "qualification_required"} else "R1"
        )
        coordinate_field = (
            "checkpoint_tick" if authority == "absolute_tick" else "generation"
        )
        identity = (
            f"checkpoint-tick-{coordinate:012d}"
            if authority == "absolute_tick"
            else f"checkpoint-{coordinate:09d}"
        )
        rows.append(
            {
                "identity": identity,
                coordinate_field: coordinate,
                "role": role,
                "retention_class": retention_class,
            }
        )
    return rows


def _profile_payload(
    profile: str,
    *,
    total_max_bytes: int,
    checkpoint_schedule: Sequence[int],
    checkpoint_authority: str = "legacy_generation",
    time_authority: str | None = None,
) -> dict[str, Any]:
    if time_authority is not None:
        checkpoint_authority = (
            "absolute_tick"
            if time_authority == "absolute_tick_v1"
            else "legacy_generation"
        )
    if profile not in LIFECYCLE_PROFILES:
        raise ArtifactLifecycleError(
            f"lifecycle_profile must be one of {list(LIFECYCLE_PROFILES)}"
        )
    output_profile = {
        CHECKPOINT_TRANSIT: "compact_default",
        SCIENCE_COMPACT: "compact_default",
        QUALIFICATION: "compact_default",
        DEBUG_FORENSIC: "debug_event_trace",
    }[profile]
    policies = {
        CHECKPOINT_TRANSIT: ("forbidden", "forbidden", False),
        SCIENCE_COMPACT: ("forbidden", "forbidden", False),
        QUALIFICATION: ("forbidden", "forbidden", False),
        DEBUG_FORENSIC: ("bounded", "allowed", True),
    }
    raw_trace, event_trace, ring_enabled = policies[profile]
    audit_binding = "required" if profile == SCIENCE_COMPACT else "not_required"
    qualification_binding = "required" if profile == QUALIFICATION else "not_required"
    default_class = {
        CHECKPOINT_TRANSIT: "R1",
        SCIENCE_COMPACT: "R3",
        QUALIFICATION: "R4",
        DEBUG_FORENSIC: "R1",
    }[profile]
    decision_class = "A" if profile == SCIENCE_COMPACT else None
    required_closeout = [
        "run_passport",
        "artifact_disposition_manifest",
    ]
    if profile in {SCIENCE_COMPACT, QUALIFICATION}:
        required_closeout.append("evidence_closeout_or_qualification_receipt")
    required_closeout.append("storage_disposition_receipt")
    storage = _partition_budget(total_max_bytes, profile)
    checkpoints = {
        (
            "checkpoint_ticks"
            if checkpoint_authority == "absolute_tick"
            else "schedule"
        ): sorted({int(value) for value in checkpoint_schedule}),
        "roles": _checkpoint_roles(
            checkpoint_schedule,
            profile=profile,
            authority=checkpoint_authority,
        ),
    }
    if checkpoint_authority == "absolute_tick":
        checkpoints["authority"] = "absolute_tick"
    return {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "lifecycle_profile": profile,
        "storage": storage,
        "observability": {
            "output_profile": output_profile,
            "raw_tick_trace_policy": raw_trace,
            "debug_event_trace_policy": event_trace,
            "ring_buffer": {
                "enabled": ring_enabled,
                "max_bytes": min(256 * 1024 * 1024, storage["scratch_max_bytes"])
                if ring_enabled
                else 1,
            },
        },
        "checkpoints": checkpoints,
        "closeout": {
            "required_types": required_closeout,
            "compact_replacement_verification_required": profile
            in {SCIENCE_COMPACT, DEBUG_FORENSIC},
        },
        "retention": {
            "default_class": default_class,
            "decision_evidence_class": decision_class,
            "audit_hold_binding": audit_binding,
            "qualification_hold_binding": qualification_binding,
            "unknown_artifact_policy": UNKNOWN_ARTIFACT_POLICY,
        },
        "prune": {
            # New, explicitly lifecycle-bound runs release exact terminal
            # operational material by default.  DEBUG_FORENSIC remains held
            # until an explicit diagnostic-closeout review; legacy-derived
            # contracts are forced back to false by the resolver below.
            "automatic_prune_allowed": profile != DEBUG_FORENSIC,
            "required_predicates": list(LIFECYCLE_REQUIRED_PRUNE_PREDICATES),
        },
    }


def _validate_contract_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(value)
    supplied_hash = raw.pop("contract_sha256", None)
    _exact(
        raw,
        {
            "schema_version",
            "lifecycle_profile",
            "storage",
            "observability",
            "checkpoints",
            "closeout",
            "retention",
            "prune",
        },
        path="artifact_lifecycle",
    )
    if raw.get("schema_version") != LIFECYCLE_SCHEMA_VERSION:
        raise ArtifactLifecycleError(
            f"artifact_lifecycle.schema_version must be {LIFECYCLE_SCHEMA_VERSION!r}"
        )
    profile = _text(
        raw.get("lifecycle_profile"), path="artifact_lifecycle.lifecycle_profile"
    )
    if profile not in LIFECYCLE_PROFILES:
        raise ArtifactLifecycleError(
            "artifact_lifecycle.lifecycle_profile is unsupported"
        )

    storage = raw.get("storage")
    if not isinstance(storage, Mapping):
        raise ArtifactLifecycleError("artifact_lifecycle.storage must be an object")
    storage = dict(storage)
    _exact(
        storage,
        {
            "total_max_bytes",
            "scratch_max_bytes",
            "checkpoint_max_bytes",
            "durable_max_bytes",
            "emergency_reserve_bytes",
        },
        path="artifact_lifecycle.storage",
    )
    storage = {
        name: _bytes(storage.get(name), path=f"artifact_lifecycle.storage.{name}")
        for name in (
            "total_max_bytes",
            "scratch_max_bytes",
            "checkpoint_max_bytes",
            "durable_max_bytes",
            "emergency_reserve_bytes",
        )
    }
    if (
        storage["scratch_max_bytes"]
        + storage["checkpoint_max_bytes"]
        + storage["durable_max_bytes"]
        + storage["emergency_reserve_bytes"]
        > storage["total_max_bytes"]
    ):
        raise ArtifactLifecycleError(
            "artifact_lifecycle.storage subquotas plus emergency reserve exceed total_max_bytes"
        )

    observability = raw.get("observability")
    if not isinstance(observability, Mapping):
        raise ArtifactLifecycleError(
            "artifact_lifecycle.observability must be an object"
        )
    observability = dict(observability)
    _exact(
        observability,
        {
            "output_profile",
            "raw_tick_trace_policy",
            "debug_event_trace_policy",
            "ring_buffer",
            "override_reason",
        },
        path="artifact_lifecycle.observability",
    )
    output_profile = _identifier(
        observability.get("output_profile"),
        path="artifact_lifecycle.observability.output_profile",
    )
    if output_profile not in OUTPUT_PROFILES:
        raise ArtifactLifecycleError(
            "artifact_lifecycle.observability.output_profile is unsupported"
        )
    default_output_profile = PROFILE_OUTPUT_BINDINGS[profile]
    raw_override_reason = observability.get("override_reason")
    override_reason = None
    if raw_override_reason is not None:
        override_reason = _text(
            raw_override_reason,
            path="artifact_lifecycle.observability.override_reason",
        )
    if output_profile != default_output_profile and override_reason is None:
        raise ArtifactLifecycleError(
            "artifact_lifecycle.observability.output_profile override requires "
            "a declared override_reason"
        )
    raw_policy = _identifier(
        observability.get("raw_tick_trace_policy"),
        path="artifact_lifecycle.observability.raw_tick_trace_policy",
    )
    event_policy = _identifier(
        observability.get("debug_event_trace_policy"),
        path="artifact_lifecycle.observability.debug_event_trace_policy",
    )
    if raw_policy not in {"forbidden", "allowed", "bounded"}:
        raise ArtifactLifecycleError("raw_tick_trace_policy is unsupported")
    if event_policy not in {"forbidden", "allowed", "bounded"}:
        raise ArtifactLifecycleError("debug_event_trace_policy is unsupported")
    ring = observability.get("ring_buffer")
    if not isinstance(ring, Mapping):
        raise ArtifactLifecycleError(
            "artifact_lifecycle.observability.ring_buffer must be an object"
        )
    ring = dict(ring)
    _exact(
        ring,
        {"enabled", "max_bytes"},
        path="artifact_lifecycle.observability.ring_buffer",
    )
    ring_enabled = _boolean(
        ring.get("enabled"), path="artifact_lifecycle.observability.ring_buffer.enabled"
    )
    ring_max = _bytes(
        ring.get("max_bytes"),
        path="artifact_lifecycle.observability.ring_buffer.max_bytes",
    )
    if ring_enabled and raw_policy != "bounded" and event_policy != "bounded":
        raise ArtifactLifecycleError("ring_buffer requires a bounded trace policy")
    if ring_enabled and ring_max > storage["scratch_max_bytes"]:
        raise ArtifactLifecycleError(
            "ring_buffer.max_bytes cannot exceed storage.scratch_max_bytes"
        )
    if (
        profile in {CHECKPOINT_TRANSIT, SCIENCE_COMPACT, QUALIFICATION}
        and (raw_policy != "forbidden" or event_policy != "forbidden")
        and override_reason is None
    ):
        raise ArtifactLifecycleError(
            f"{profile} trace overrides require a declared override_reason"
        )
    if (
        profile == QUALIFICATION
        and (
            raw_policy != "forbidden"
            or event_policy != "forbidden"
            or output_profile != default_output_profile
        )
        and override_reason is None
    ):
        raise ArtifactLifecycleError(
            "QUALIFICATION observability escalation requires a declared override_reason"
        )
    observability = {
        "output_profile": output_profile,
        "raw_tick_trace_policy": raw_policy,
        "debug_event_trace_policy": event_policy,
        "ring_buffer": {"enabled": ring_enabled, "max_bytes": ring_max},
    }
    if override_reason is not None:
        observability["override_reason"] = override_reason

    checkpoints = raw.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise ArtifactLifecycleError("artifact_lifecycle.checkpoints must be an object")
    checkpoints = dict(checkpoints)
    _exact(
        checkpoints,
        {"schedule", "checkpoint_ticks", "authority", "roles"},
        path="artifact_lifecycle.checkpoints",
    )
    authority = checkpoints.get("authority", "legacy_generation")
    if authority not in CHECKPOINT_AUTHORITIES:
        raise ArtifactLifecycleError(
            "artifact_lifecycle.checkpoints.authority is unsupported"
        )
    if authority == "absolute_tick":
        if "schedule" in checkpoints:
            raise ArtifactLifecycleError(
                "absolute-tick checkpoints must not carry a generation schedule"
            )
        schedule = checkpoints.get("checkpoint_ticks")
        coordinate_field = "checkpoint_tick"
    else:
        if "checkpoint_ticks" in checkpoints:
            raise ArtifactLifecycleError(
                "legacy-generation checkpoints must not carry checkpoint_ticks"
            )
        schedule = checkpoints.get("schedule")
        coordinate_field = "generation"
    if not isinstance(schedule, list) or any(
        type(item) is not int or item < 0 for item in schedule
    ):
        raise ArtifactLifecycleError(
            "artifact_lifecycle.checkpoints coordinate schedule is invalid"
        )
    if schedule != sorted(set(schedule)):
        raise ArtifactLifecycleError(
            "artifact_lifecycle.checkpoints coordinates must be sorted and unique"
        )
    roles = checkpoints.get("roles")
    if not isinstance(roles, list):
        raise ArtifactLifecycleError(
            "artifact_lifecycle.checkpoints.roles must be an array"
        )
    normalized_roles: list[dict[str, Any]] = []
    role_ids: set[str] = set()
    role_coordinates: set[int] = set()
    for index, item in enumerate(roles):
        if not isinstance(item, Mapping):
            raise ArtifactLifecycleError(
                f"artifact_lifecycle.checkpoints.roles[{index}] must be an object"
            )
        row = dict(item)
        _exact(
            row,
            {
                "identity",
                "generation",
                "checkpoint_tick",
                "role",
                "retention_class",
            },
            path=f"artifact_lifecycle.checkpoints.roles[{index}]",
        )
        identity = _identifier(
            row.get("identity"),
            path=f"artifact_lifecycle.checkpoints.roles[{index}].identity",
        )
        if coordinate_field == "checkpoint_tick" and "generation" in row:
            raise ArtifactLifecycleError(
                f"artifact_lifecycle.checkpoints.roles[{index}] uses generation in an absolute-tick contract"
            )
        if coordinate_field == "generation" and "checkpoint_tick" in row:
            raise ArtifactLifecycleError(
                f"artifact_lifecycle.checkpoints.roles[{index}] uses checkpoint_tick in a legacy contract"
            )
        coordinate = row.get(coordinate_field)
        if type(coordinate) is not int or coordinate < 0 or coordinate not in schedule:
            raise ArtifactLifecycleError(
                f"artifact_lifecycle.checkpoints.roles[{index}].{coordinate_field} is not in the declared schedule"
            )
        role = _identifier(
            row.get("role"), path=f"artifact_lifecycle.checkpoints.roles[{index}].role"
        )
        if role not in CHECKPOINT_ROLES:
            raise ArtifactLifecycleError(f"checkpoint role {role!r} is unsupported")
        retention_class = _text(
            row.get("retention_class"),
            path=f"artifact_lifecycle.checkpoints.roles[{index}].retention_class",
        )
        if retention_class not in RETENTION_CLASSES:
            raise ArtifactLifecycleError("checkpoint retention class is unsupported")
        if identity in role_ids or coordinate in role_coordinates:
            raise ArtifactLifecycleError(
                "checkpoint roles must identify unique identities and coordinates"
            )
        role_ids.add(identity)
        role_coordinates.add(coordinate)
        normalized_roles.append(
            {
                "identity": identity,
                coordinate_field: coordinate,
                "role": role,
                "retention_class": retention_class,
            }
        )
    if set(schedule) != role_coordinates:
        raise ArtifactLifecycleError(
            "checkpoint roles must cover the declared checkpoint schedule"
        )
    checkpoints = {
        ("checkpoint_ticks" if authority == "absolute_tick" else "schedule"): list(
            schedule
        ),
        "roles": normalized_roles,
    }
    if authority == "absolute_tick":
        checkpoints["authority"] = authority

    closeout = raw.get("closeout")
    if not isinstance(closeout, Mapping):
        raise ArtifactLifecycleError("artifact_lifecycle.closeout must be an object")
    closeout = dict(closeout)
    _exact(
        closeout,
        {"required_types", "compact_replacement_verification_required"},
        path="artifact_lifecycle.closeout",
    )
    closeout = {
        "required_types": _string_list(
            closeout.get("required_types"),
            path="artifact_lifecycle.closeout.required_types",
            nonempty=True,
        ),
        "compact_replacement_verification_required": _boolean(
            closeout.get("compact_replacement_verification_required"),
            path="artifact_lifecycle.closeout.compact_replacement_verification_required",
        ),
    }

    retention = raw.get("retention")
    if not isinstance(retention, Mapping):
        raise ArtifactLifecycleError("artifact_lifecycle.retention must be an object")
    retention = dict(retention)
    _exact(
        retention,
        {
            "default_class",
            "decision_evidence_class",
            "audit_hold_binding",
            "qualification_hold_binding",
            "unknown_artifact_policy",
            "unknown_artifact_scope",
        },
        path="artifact_lifecycle.retention",
    )
    default_class = _text(
        retention.get("default_class"),
        path="artifact_lifecycle.retention.default_class",
    )
    if default_class not in RETENTION_CLASSES:
        raise ArtifactLifecycleError(
            "artifact_lifecycle.retention.default_class is unsupported"
        )
    decision_class = retention.get("decision_evidence_class")
    if decision_class is not None:
        decision_class = _text(
            decision_class, path="artifact_lifecycle.retention.decision_evidence_class"
        )
        if decision_class not in DECISION_EVIDENCE_CLASSES:
            raise ArtifactLifecycleError(
                "artifact_lifecycle.retention.decision_evidence_class is unsupported"
            )
    unknown_artifact_scope = retention.get("unknown_artifact_scope")
    retention = {
        "default_class": default_class,
        "decision_evidence_class": decision_class,
        "audit_hold_binding": _identifier(
            retention.get("audit_hold_binding"),
            path="artifact_lifecycle.retention.audit_hold_binding",
        ),
        "qualification_hold_binding": _identifier(
            retention.get("qualification_hold_binding"),
            path="artifact_lifecycle.retention.qualification_hold_binding",
        ),
        "unknown_artifact_policy": _text(
            retention.get("unknown_artifact_policy"),
            path="artifact_lifecycle.retention.unknown_artifact_policy",
        ),
    }
    if unknown_artifact_scope is not None:
        unknown_artifact_scope = _identifier(
            unknown_artifact_scope,
            path="artifact_lifecycle.retention.unknown_artifact_scope",
        )
        if unknown_artifact_scope not in UNKNOWN_ARTIFACT_SCOPES:
            raise ArtifactLifecycleError(
                "retention.unknown_artifact_scope is unsupported"
            )
        retention["unknown_artifact_scope"] = unknown_artifact_scope
    if retention["audit_hold_binding"] not in {"required", "not_required"}:
        raise ArtifactLifecycleError("retention.audit_hold_binding is unsupported")
    if retention["qualification_hold_binding"] not in {"required", "not_required"}:
        raise ArtifactLifecycleError(
            "retention.qualification_hold_binding is unsupported"
        )
    if retention["unknown_artifact_policy"] != UNKNOWN_ARTIFACT_POLICY:
        raise ArtifactLifecycleError(
            "unknown_artifact_policy must be RX_BLOCKS_AUTO_DELETE"
        )

    prune = raw.get("prune")
    if not isinstance(prune, Mapping):
        raise ArtifactLifecycleError("artifact_lifecycle.prune must be an object")
    prune = dict(prune)
    _exact(
        prune,
        {"automatic_prune_allowed", "required_predicates"},
        path="artifact_lifecycle.prune",
    )
    automatic = _boolean(
        prune.get("automatic_prune_allowed"),
        path="artifact_lifecycle.prune.automatic_prune_allowed",
    )
    predicates = _string_list(
        prune.get("required_predicates"),
        path="artifact_lifecycle.prune.required_predicates",
        nonempty=True,
    )
    if not set(LIFECYCLE_REQUIRED_PRUNE_PREDICATES).issubset(predicates):
        raise ArtifactLifecycleError(
            "prune.required_predicates must include all lifecycle safety predicates"
        )
    if automatic and profile == DEBUG_FORENSIC:
        raise ArtifactLifecycleError(
            "DEBUG_FORENSIC cannot enable automatic prune before diagnostic closeout"
        )
    if automatic and not set(LIFECYCLE_REQUIRED_PRUNE_PREDICATES).issubset(predicates):
        raise ArtifactLifecycleError(
            "automatic prune requires all lifecycle safety predicates"
        )
    prune = {"automatic_prune_allowed": automatic, "required_predicates": predicates}

    canonical = {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "lifecycle_profile": profile,
        "storage": storage,
        "observability": observability,
        "checkpoints": checkpoints,
        "closeout": closeout,
        "retention": retention,
        "prune": prune,
    }
    if supplied_hash is not None and supplied_hash != canonical_sha256(canonical):
        raise ArtifactLifecycleError("artifact_lifecycle contract SHA-256 mismatch")
    canonical["contract_sha256"] = canonical_sha256(canonical)
    return canonical


@dataclass(frozen=True, slots=True)
class RunArtifactLifecycleContractV1:
    """Canonical immutable contract carried by a compiled run plan."""

    schema_version: str
    lifecycle_profile: str
    storage: Mapping[str, int]
    observability: Mapping[str, Any]
    checkpoints: Mapping[str, Any]
    closeout: Mapping[str, Any]
    retention: Mapping[str, Any]
    prune: Mapping[str, Any]

    def payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        core = {
            "schema_version": self.schema_version,
            "lifecycle_profile": self.lifecycle_profile,
            "storage": dict(self.storage),
            "observability": dict(self.observability),
            "checkpoints": dict(self.checkpoints),
            "closeout": dict(self.closeout),
            "retention": dict(self.retention),
            "prune": dict(self.prune),
        }
        if include_hash:
            core["contract_sha256"] = canonical_sha256(core)
        return core

    @property
    def contract_sha256(self) -> str:
        return canonical_sha256(self.payload(include_hash=False))


def validate_run_artifact_lifecycle_contract_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly validate and canonicalize a lifecycle contract payload."""

    return _validate_contract_payload(value)


def lifecycle_profile_contract_v1(
    profile: str,
    *,
    total_max_bytes: int = 10 * 1024 * 1024 * 1024,
    checkpoint_schedule: Sequence[int] = (),
    checkpoint_authority: str = "legacy_generation",
) -> dict[str, Any]:
    """Return one bounded profile with a complete V1 contract."""

    return _validate_contract_payload(
        _profile_payload(
            profile,
            total_max_bytes=_bytes(total_max_bytes, path="total_max_bytes"),
            checkpoint_schedule=checkpoint_schedule,
            checkpoint_authority=checkpoint_authority,
        )
    )


def _normalize_legacy_checkpoint_contract_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate the pre-native checkpoint spelling at the import boundary.

    Older campaign documents used ``time_authority``/``tick`` and carried a
    hash for that representation.  The active contract is
    ``authority``/``checkpoint_tick``; the old hash cannot bind the translated
    payload and is therefore deliberately discarded before canonicalization.
    """

    raw = dict(value)
    checkpoints = raw.get("checkpoints")
    if not isinstance(checkpoints, Mapping) or "time_authority" not in checkpoints:
        return raw
    old = dict(checkpoints)
    time_authority = old.pop("time_authority")
    roles = old.get("roles", [])
    if time_authority == "absolute_tick_v1":
        schedule = old.pop("schedule", old.pop("checkpoint_ticks", []))
        translated_roles = []
        for item in roles:
            row = dict(item)
            if "tick" in row:
                row["checkpoint_tick"] = row.pop("tick")
            translated_roles.append(row)
        raw["checkpoints"] = {
            "authority": "absolute_tick",
            "checkpoint_ticks": list(schedule),
            "roles": translated_roles,
        }
    elif time_authority == "generation":
        schedule = old.pop("checkpoint_ticks", old.pop("schedule", []))
        translated_roles = []
        for item in roles:
            row = dict(item)
            if "checkpoint_tick" in row:
                row["generation"] = row.pop("checkpoint_tick")
            translated_roles.append(row)
        raw["checkpoints"] = {
            "schedule": list(schedule),
            "roles": translated_roles,
        }
    else:
        raise ArtifactLifecycleError(
            "legacy artifact_lifecycle.checkpoints.time_authority is unsupported"
        )
    raw.pop("contract_sha256", None)
    return raw


def resolve_run_artifact_lifecycle_v1(
    value: Mapping[str, Any] | None,
    *,
    total_max_bytes: int | None = None,
    checkpoint_schedule: Sequence[int] = (),
    terminal_generation: int | None = None,
    terminal_tick: int | None = None,
    checkpoint_authority: str = "legacy_generation",
    time_authority: str | None = None,
    legacy: bool = False,
) -> dict[str, Any]:
    """Resolve an explicit contract or derive a conservative legacy default."""

    if time_authority is not None:
        checkpoint_authority = (
            "absolute_tick"
            if time_authority == "absolute_tick_v1"
            else "legacy_generation"
        )
    if checkpoint_authority not in CHECKPOINT_AUTHORITIES:
        raise ArtifactLifecycleError("unsupported lifecycle checkpoint authority")
    schedule = sorted({int(item) for item in checkpoint_schedule})
    terminal_coordinate = (
        terminal_tick
        if checkpoint_authority == "absolute_tick"
        else terminal_generation
    )
    if terminal_coordinate is not None and terminal_coordinate >= 0:
        schedule = sorted({*schedule, int(terminal_coordinate)})
    if value is None:
        total = int(total_max_bytes or 10 * 1024 * 1024 * 1024)
        payload = _profile_payload(
            CHECKPOINT_TRANSIT,
            total_max_bytes=total,
            checkpoint_schedule=schedule,
            checkpoint_authority=checkpoint_authority,
        )
        # Legacy plans inherit a conservative contract but never inherit
        # destructive cleanup authority.
        payload["prune"] = {
            "automatic_prune_allowed": False,
            "required_predicates": list(LIFECYCLE_REQUIRED_PRUNE_PREDICATES),
        }
        return _validate_contract_payload(payload)
    if not isinstance(value, Mapping):
        raise ArtifactLifecycleError("artifact_lifecycle must be an object")
    contract = _validate_contract_payload(
        _normalize_legacy_checkpoint_contract_v1(value)
    )
    checkpoint_contract = contract["checkpoints"]
    declared_authority = checkpoint_contract.get("authority", "legacy_generation")
    declared_schedule = checkpoint_contract.get(
        "checkpoint_ticks" if declared_authority == "absolute_tick" else "schedule",
        [],
    )
    if declared_authority != checkpoint_authority or declared_schedule != schedule:
        if legacy:
            contract = rebind_lifecycle_checkpoint_schedule_v1(
                contract,
                schedule,
                checkpoint_authority=checkpoint_authority,
            )
        else:
            raise ArtifactLifecycleError(
                "artifact_lifecycle checkpoints must bind the campaign checkpoint authority and schedule"
            )
    if total_max_bytes is not None and contract["storage"]["total_max_bytes"] > int(
        total_max_bytes
    ):
        raise ArtifactLifecycleError(
            "artifact_lifecycle.storage.total_max_bytes exceeds execution.budgets.disk_bytes"
        )
    return contract


def ensure_lifecycle_contract_v1(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return the plan-bound contract, deriving a safe legacy one if absent."""

    raw = plan.get("artifact_lifecycle")
    execution = plan.get("execution")
    if not isinstance(execution, Mapping):
        execution = {}
    budgets = execution.get("budgets")
    if not isinstance(budgets, Mapping):
        budgets = {}
    tick_native = (
        execution.get("execution_route_identity") == "tick_native_execution_route_v1"
    )
    checkpoints = (
        execution.get("checkpoint_ticks", [])
        if tick_native
        else execution.get("checkpoints", plan.get("checkpoints", []))
    )
    if not isinstance(checkpoints, list):
        checkpoints = []
    generations = execution.get("generations")
    terminal = (
        int(generations) if type(generations) is int and generations >= 0 else None
    )
    total_ticks = execution.get("total_ticks")
    tick_terminal = (
        int(total_ticks) if type(total_ticks) is int and total_ticks >= 0 else None
    )
    total = budgets.get("disk_bytes")
    total_bytes = int(total) if type(total) is int and total > 0 else None
    if isinstance(raw, Mapping):
        contract = _validate_contract_payload(raw)
        if tick_native and contract["checkpoints"].get("authority") != "absolute_tick":
            raise ArtifactLifecycleError(
                "tick-native execution requires absolute checkpoint_ticks lifecycle authority"
            )
        if (
            total_bytes is not None
            and contract["storage"]["total_max_bytes"] > total_bytes
        ):
            raise ArtifactLifecycleError(
                "artifact_lifecycle.storage.total_max_bytes exceeds execution.budgets.disk_bytes"
            )
        return contract
    return resolve_run_artifact_lifecycle_v1(
        None,
        total_max_bytes=total_bytes,
        checkpoint_schedule=checkpoints,
        terminal_generation=terminal if not tick_native else None,
        terminal_tick=tick_terminal if tick_native else None,
        checkpoint_authority=("absolute_tick" if tick_native else "legacy_generation"),
        legacy=raw is None,
    )


def rebind_lifecycle_checkpoint_schedule_v1(
    contract: Mapping[str, Any],
    schedule: Sequence[int],
    *,
    checkpoint_authority: str | None = None,
) -> dict[str, Any]:
    """Bind a tick-native or QRR schedule without changing other policy."""

    current = _validate_contract_payload(contract)
    if checkpoint_authority is None:
        checkpoint_authority = current["checkpoints"].get(
            "authority", "legacy_generation"
        )
    if checkpoint_authority not in CHECKPOINT_AUTHORITIES:
        raise ArtifactLifecycleError("checkpoint authority is unsupported")
    current["checkpoints"] = {
        (
            "checkpoint_ticks"
            if checkpoint_authority == "absolute_tick"
            else "schedule"
        ): sorted({int(item) for item in schedule}),
        "roles": _checkpoint_roles(
            schedule,
            profile=str(current["lifecycle_profile"]),
            authority=checkpoint_authority,
        ),
    }
    if checkpoint_authority == "absolute_tick":
        current["checkpoints"]["authority"] = checkpoint_authority
    current.pop("contract_sha256", None)
    return _validate_contract_payload(current)


def checkpoint_role_for_generation(
    contract: Mapping[str, Any], generation: int
) -> dict[str, Any] | None:
    current = _validate_contract_payload(contract)
    for row in current["checkpoints"]["roles"]:
        if row.get("generation") == generation:
            return dict(row)
    return None


def checkpoint_role_for_tick(
    contract: Mapping[str, Any], checkpoint_tick: int
) -> dict[str, Any] | None:
    current = _validate_contract_payload(contract)
    if current["checkpoints"].get("authority") != "absolute_tick":
        return None
    for row in current["checkpoints"]["roles"]:
        if row.get("checkpoint_tick") == checkpoint_tick:
            return dict(row)
    return None


def _atomic_json_target_stat(parent_fd: int, target_name: str) -> os.stat_result | None:
    """Read a target safely across a concurrent atomic replacement.

    Overlay filesystems may transiently expose a regular target with
    ``st_nlink == 0`` while another writer completes ``os.replace``.  Yield
    and retry that one transient state; persistent zero-link, hard-linked,
    symlink, and non-regular targets remain fail-closed at the caller.
    """

    observed: os.stat_result | None = None
    for attempt in range(8):
        try:
            observed = os.stat(
                target_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 0:
            return observed
        if attempt < 7:
            os.sched_yield()
    return observed


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    target = Path(os.path.abspath(os.fspath(path.expanduser())))
    target.parent.mkdir(parents=True, exist_ok=True)
    parent_fd = os.open(
        target.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    temporary_name: str | None = None
    descriptor: int | None = None
    try:
        observed = _atomic_json_target_stat(parent_fd, target.name)
        if observed is not None and (
            not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1
        ):
            raise ArtifactLifecycleError(
                f"refusing unsafe canonical JSON target: {target}"
            )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        for _ in range(128):
            candidate = f".{target.name}.{secrets.token_hex(12)}.tmp"
            try:
                descriptor = os.open(candidate, flags, 0o644, dir_fd=parent_fd)
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if descriptor is None or temporary_name is None:
            raise ArtifactLifecycleError(
                f"cannot allocate unique canonical JSON staging file: {target}"
            )
        payload = pretty_json(dict(value)).encode("utf-8")
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        observed = _atomic_json_target_stat(parent_fd, target.name)
        if observed is not None and (
            not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1
        ):
            raise ArtifactLifecycleError(
                f"refusing unsafe canonical JSON target: {target}"
            )
        os.replace(
            temporary_name,
            target.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary_name = None
        os.fsync(parent_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)
    return target


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_sealed_artifact_inventory_row_v1(
    *,
    run_root: Path,
    path: Path,
    artifact_family: str,
    retention_class: str,
    owner_writer: str,
    **metadata: Any,
) -> dict[str, Any]:
    """Build one exact producer-owned inventory row at materialization time."""

    root = run_root.expanduser().resolve()
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise ArtifactLifecycleError("sealed artifact cannot be a symlink")
    target = unresolved.resolve(strict=True)
    try:
        relative = str(target.relative_to(root)).replace("\\", "/")
    except ValueError as exc:
        raise ArtifactLifecycleError(
            "sealed artifact escapes the runtime root"
        ) from exc
    observed = os.lstat(target)
    if retention_class not in RETENTION_CLASSES:
        raise ArtifactLifecycleError("sealed artifact retention class is unsupported")
    core = {
        "path": relative,
        "artifact_family": _text(
            artifact_family, path="sealed_artifact.artifact_family"
        ),
        "retention_class": retention_class,
        "owner_writer": _text(owner_writer, path="sealed_artifact.owner_writer"),
        **metadata,
    }
    if os.path.isfile(target):
        if observed.st_nlink != 1:
            raise ArtifactLifecycleError("sealed artifact file cannot be hard-linked")
        before = (
            observed.st_dev,
            observed.st_ino,
            observed.st_size,
            observed.st_mtime_ns,
        )
        digest = _sha256_file(target)
        after = os.lstat(target)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) != before:
            raise ArtifactLifecycleError("sealed artifact changed while hashing")
        return {
            **core,
            "artifact_type": "file",
            "bytes": int(observed.st_size),
            "sha256": digest,
        }
    if not os.path.isdir(target):
        raise ArtifactLifecycleError(
            "sealed artifact must be a regular file or directory"
        )
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for member in sorted(target.rglob("*"), key=lambda item: str(item)):
        member_stat = os.lstat(member)
        if member.is_symlink():
            raise ArtifactLifecycleError("sealed artifact tree cannot contain symlinks")
        if not member.is_file():
            continue
        if member_stat.st_nlink != 1:
            raise ArtifactLifecycleError(
                "sealed artifact tree cannot contain hard links"
            )
        before = (
            member_stat.st_dev,
            member_stat.st_ino,
            member_stat.st_size,
            member_stat.st_mtime_ns,
        )
        digest = _sha256_file(member)
        after = os.lstat(member)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) != before:
            raise ArtifactLifecycleError("sealed artifact tree changed while hashing")
        total_bytes += int(member_stat.st_size)
        entries.append(
            {
                "path": str(member.relative_to(target)).replace("\\", "/"),
                "bytes": int(member_stat.st_size),
                "sha256": digest,
            }
        )
    return {
        **core,
        "artifact_type": "sealed_tree",
        "bytes": total_bytes,
        "tree_sha256": canonical_sha256(entries),
    }


def build_run_passport_v1(
    plan: Mapping[str, Any],
    *,
    run_root: Path,
    terminal_status: str,
    checkpoint_catalog: Sequence[Mapping[str, Any]] = (),
    terminal_digest: str | None = None,
    source_identity: Mapping[str, Any] | None = None,
    provider_identity: Mapping[str, Any] | None = None,
    config_identity: str | None = None,
    seed_identity: str | None = None,
    disposition_manifest_path: str | None = None,
    disposition_receipt_path: str | None = None,
) -> dict[str, Any]:
    lifecycle = ensure_lifecycle_contract_v1(plan)
    artifacts = plan.get("artifacts")
    source = dict(source_identity or {})
    if not source and isinstance(artifacts, Mapping):
        source = {
            "artifact_hashes": {
                str(name): str(row.get("sha256"))
                for name, row in sorted(artifacts.items())
                if isinstance(row, Mapping) and isinstance(row.get("sha256"), str)
            }
        }
    cells = plan.get("cells", [])
    if config_identity is None:
        config_identity = canonical_sha256(
            [
                {
                    "cell_id": cell.get("cell_id"),
                    "parameters": cell.get("parameters", {}),
                }
                for cell in cells
                if isinstance(cell, Mapping)
            ]
        )
    if seed_identity is None:
        seed_identity = canonical_sha256(
            [
                {"cell_id": cell.get("cell_id"), "seeds": cell.get("seeds", {})}
                for cell in cells
                if isinstance(cell, Mapping)
            ]
        )
    native_lifecycle = lifecycle["checkpoints"].get("authority") == "absolute_tick"
    coordinate_name = "checkpoint_tick" if native_lifecycle else "generation"
    catalog = []
    for row in checkpoint_catalog:
        if not isinstance(row, Mapping):
            continue
        catalog.append(
            {
                key: row[key]
                for key in (
                    "identity",
                    coordinate_name,
                    "role",
                    "retention_class",
                    "path",
                    "sha256",
                    "state_sha256",
                )
                if key in row
            }
        )
    catalog.sort(
        key=lambda row: (
            int(row.get(coordinate_name, -1)),
            str(row.get("identity", "")),
        )
    )
    core: dict[str, Any] = {
        "schema_version": RUN_PASSPORT_SCHEMA_VERSION,
        "campaign_id": str(plan.get("campaign_id", "")),
        "run_id": run_root.expanduser().resolve().name,
        "plan_sha256": str(plan.get("plan_sha256", "")),
        "qualification_key_sha256": plan.get("qualification_key_sha256"),
        "source_identity": source,
        "config_identity": config_identity,
        "seed_identity": seed_identity,
        "provider_identity": dict(provider_identity or {}),
        "checkpoint_catalog": catalog,
        "terminal_status": terminal_status,
        "terminal_digest": terminal_digest,
        "lifecycle_profile": lifecycle["lifecycle_profile"],
        "lifecycle_contract_sha256": lifecycle["contract_sha256"],
        "disposition_manifest_path": disposition_manifest_path,
        "disposition_receipt_path": disposition_receipt_path,
    }
    return {**core, "passport_sha256": canonical_sha256(core)}


def write_run_passport_v1(path: Path, passport: Mapping[str, Any]) -> Path:
    value = dict(passport)
    expected = value.pop("passport_sha256", None)
    actual = canonical_sha256(value)
    if expected != actual:
        raise ArtifactLifecycleError("run passport SHA-256 mismatch")
    return _atomic_json(path, {**value, "passport_sha256": expected})


def build_run_artifact_failure_receipt_v1(
    plan: Mapping[str, Any],
    *,
    run_root: Path,
    stage: str,
    failure_type: str,
    message: str,
    status: str = "failed",
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Build a minimal fail-closed receipt for an abnormal run exit.

    This is intentionally smaller than terminal disposition.  It marks the
    run as quarantined and gives later closeout a durable handoff point without
    attempting to inventory or release a potentially incomplete run tree.
    """

    if status not in {"failed", "incomplete"}:
        raise ArtifactLifecycleError(
            "failure receipt status must be failed or incomplete"
        )
    lifecycle: dict[str, Any] | None
    try:
        lifecycle = ensure_lifecycle_contract_v1(plan)
    except (ArtifactLifecycleError, TypeError, ValueError):
        lifecycle = None
    root = run_root.expanduser().resolve()
    failure_type = _text(failure_type, path="failure.type")
    stage = _text(stage, path="failure.stage")
    message = str(message or "")[:4096]
    core = {
        "schema_version": RUN_FAILURE_SCHEMA_VERSION,
        "campaign_id": plan.get("campaign_id"),
        "run_id": root.name,
        "runtime_root": str(root),
        "plan_sha256": plan.get("plan_sha256"),
        "lifecycle_profile": (
            lifecycle.get("lifecycle_profile") if lifecycle is not None else None
        ),
        "lifecycle_contract_sha256": (
            lifecycle.get("contract_sha256") if lifecycle is not None else None
        ),
        "status": status,
        "failure": {
            "stage": stage,
            "type": failure_type,
            "message": message,
        },
        "quarantine": {
            "status": "QUARANTINED",
            "automatic_prune_allowed": False,
            "reason": "abnormal execution has no trusted terminal disposition",
            "release_requires": [
                "explicit_lifecycle_closeout",
                "valid_disposition_manifest",
                "no_live_references",
                "audit_hold_released_or_not_required",
                "qualification_hold_released_or_not_required",
            ],
        },
        "disposition_manifest_path": None,
        "timestamp_utc": timestamp_utc or _now(),
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def validate_run_artifact_failure_receipt_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactLifecycleError("run failure receipt must be an object")
    raw = dict(value)
    supplied = raw.pop("receipt_sha256", None)
    _exact(
        raw,
        {
            "schema_version",
            "campaign_id",
            "run_id",
            "runtime_root",
            "plan_sha256",
            "lifecycle_profile",
            "lifecycle_contract_sha256",
            "status",
            "failure",
            "quarantine",
            "disposition_manifest_path",
            "timestamp_utc",
        },
        path="run_failure_receipt",
    )
    if raw.get("schema_version") != RUN_FAILURE_SCHEMA_VERSION:
        raise ArtifactLifecycleError("run failure receipt schema mismatch")
    if raw.get("status") not in {"failed", "incomplete"}:
        raise ArtifactLifecycleError("run failure receipt status is unsupported")
    quarantine = raw.get("quarantine")
    if (
        not isinstance(quarantine, Mapping)
        or quarantine.get("status") != "QUARANTINED"
        or quarantine.get("automatic_prune_allowed") is not False
    ):
        raise ArtifactLifecycleError(
            "run failure receipt is not fail-closed quarantined"
        )
    failure = raw.get("failure")
    _exact(
        failure if isinstance(failure, Mapping) else {},
        {"stage", "type", "message"},
        path="run_failure_receipt.failure",
    )
    _exact(
        quarantine if isinstance(quarantine, Mapping) else {},
        {"status", "automatic_prune_allowed", "reason", "release_requires"},
        path="run_failure_receipt.quarantine",
    )
    if (
        not isinstance(failure, Mapping)
        or not failure.get("type")
        or not failure.get("stage")
    ):
        raise ArtifactLifecycleError("run failure receipt lacks failure identity")
    if supplied != canonical_sha256(raw):
        raise ArtifactLifecycleError("run failure receipt SHA-256 mismatch")
    return {**raw, "receipt_sha256": supplied}


def write_run_artifact_failure_receipt_v1(
    path: Path, receipt: Mapping[str, Any]
) -> Path:
    value = validate_run_artifact_failure_receipt_v1(receipt)
    return _atomic_json(path, value)


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


_SCIENCE_REQUEST_ATTEMPT_PREFIX_V1 = (
    r"(?:batches/batch-[0-9]{4}/attempt-[0-9]{4}/)?"
)

def _lifecycle_classification_relative_v1(relative: str) -> str:
    """Strip the canonical run-root wrapper around one attempt root.

    Child lifecycle manifests are rooted at the attempt directory, while the
    science front door's storage guard scans the enclosing run root.  Keep the
    original relative path for receipts, but classify both views against the
    same provider-owned artifact layout.
    """

    match = re.fullmatch(
        r"batches/batch-[0-9]{4}/attempt-[0-9]{4}/(batches/.+)",
        relative,
    )
    if match is None:
        return relative
    candidate = match.group(1)
    if re.match(r"batches/batch-[0-9]{4}/attempt-[0-9]{4}/", candidate):
        return relative
    return candidate


def _is_direct_sparse_semantic_event_envelope_v1(relative: str) -> bool:
    """Identify the exact provider-owned sparse event transport payload."""

    relative = _lifecycle_classification_relative_v1(relative)
    return (
        re.fullmatch(
            _SCIENCE_REQUEST_ATTEMPT_PREFIX_V1
            + r"batches/[A-Za-z0-9._-]+/scratch/events/"
            r"dispatch-tick-[0-9]{12}-[0-9]{12}/"
            r"direct-sparse-events-v3\.bin",
            relative,
        )
        is not None
    )


def _is_compact_science_durable_payload_v1(relative: str) -> bool:
    """Identify exact native compact-science reducer inputs."""

    relative = _lifecycle_classification_relative_v1(relative)
    batch = r"batches/[A-Za-z0-9._-]+/science-compact/"
    patterns = (
        batch
        + r"dispatches/ticks-[0-9]{12}-[0-9]{12}/"
        + r"(?:sparse_lifecycle_events_v3\.bin(?:\.gz)?|"
        + r"event_agent_metrics_i64le\.bin(?:\.gz)?|"
        + r"birth_provenance_packed_u4\.bin(?:\.gz)?|manifest\.json)",
        batch + r"snapshots/tick-[0-9]{12}/(?:snapshot\.jsonl\.gz|manifest\.json)",
        batch
        + r"checkpoint-frontiers/tick-[0-9]{12}/"
        + r"(?:active_agent_metrics_i64le\.gz|manifest\.json)",
        batch + r"capture-index-terminal\.json",
    )
    return any(re.fullmatch(pattern, relative) is not None for pattern in patterns)


def _is_compact_science_async_staging_payload_v1(relative: str) -> bool:
    """Identify exact in-flight compact-science publication staging payloads."""

    relative = _lifecycle_classification_relative_v1(relative)
    batch = r"batches/[A-Za-z0-9._-]+/science-compact/"
    staging = (
        batch
        + r"dispatches/\.ticks-[0-9]{12}-[0-9]{12}\.async-stage-[0-9a-f]{32}/"
    )
    payload = (
        r"(?:sparse_lifecycle_events_v3\.bin\.gz|"
        + r"event_agent_metrics_i64le\.bin\.gz|"
        + r"birth_provenance_packed_u4\.bin\.gz|manifest\.json)"
    )
    return re.fullmatch(
        staging + r"(?:" + payload + r"|\." + payload + r"\.[A-Za-z0-9_-]+\.tmp)",
        relative,
    ) is not None


def _is_workhorse_exact_event_sink_payload_v2(relative: str) -> bool:
    """Identify exact Workhorse V2/V3 ACK payloads, not optional traces."""

    semantic = (
        r"batches/[A-Za-z0-9._-]+/workhorse-semantic-event-sink/"
        r"(?:[A-Za-z0-9_-]+/)?[0-9a-f]{64}/"
        r"(?:event_(?:rows|counts)\.bin(?:\.zlib|\.zst)?|packed_events\.bin)"
    )
    measured = (
        r"batches/[A-Za-z0-9._-]+/event-sink/"
        r"b[0-9]+-k[0-9]+-(?:warmup|sample)-[0-9]+/"
        r"[0-9a-f]{64}/"
        r"(?:event_(?:rows|counts)\.bin(?:\.zlib|\.zst)?|packed_events\.bin)"
    )
    return (
        re.fullmatch(semantic, relative) is not None
        or re.fullmatch(measured, relative) is not None
    )


def _is_workhorse_semantic_fragment_payload_v2(relative: str) -> bool:
    """Identify exact raw components of one sealed semantic-case fragment."""

    return (
        re.fullmatch(
            r"batches/[A-Za-z0-9._-]+/workhorse-semantic-fragment/"
            r"(?:[A-Za-z0-9_]+\.(?:input_)?(?:state|scratch|events|counts)"
            r"|base_checkpoint_state)\.bin",
            relative,
        )
        is not None
    )


def _is_workhorse_semantic_fragment_manifest_v2(relative: str) -> bool:
    return (
        re.fullmatch(
            r"batches/[A-Za-z0-9._-]+/workhorse-semantic-fragment/"
            r"semantic_case_fragment\.json",
            relative,
        )
        is not None
    )


def _zone(relative: str) -> str:
    if _is_compact_science_async_staging_payload_v1(relative):
        return "scratch"
    if _is_compact_science_durable_payload_v1(relative):
        return "durable"
    if (
        _is_direct_sparse_semantic_event_envelope_v1(relative)
        or _is_workhorse_exact_event_sink_payload_v2(relative)
        or _is_workhorse_semantic_fragment_payload_v2(relative)
    ):
        return "scratch"
    if _is_workhorse_semantic_fragment_manifest_v2(relative):
        return "durable"
    first = relative.split("/", 1)[0]
    if first in {
        "control",
        "checkpoints",
        "durable",
        "scratch",
        "cache",
        "quarantine",
        "closeout",
    }:
        return first
    lower = relative.lower()
    # The resident event sink is authoritative checkpoint/restore state.  It
    # contains the word "event", but it is not an optional debug event trace;
    # classify it before the generic observability rules below so its bytes
    # are accounted for by the bounded checkpoint budget.
    if "event_history_v1" in lower:
        return "checkpoints"
    if "checkpoint_bundles_v2" in lower or "/checkpoint" in lower:
        return "checkpoints"
    if "lineage_reproduction_funnel_events" in lower:
        return "scratch"
    if "lineage_reproduction_funnel_generation_summary" in lower:
        return "durable"
    if "typed_lifecycle_events" in lower:
        return "scratch"
    if "compact_generation_summaries" in lower:
        return "durable"
    if "evidence_closeout" in lower or "chronicle_archive" in lower:
        return "closeout"
    if any(token in lower for token in ("trace", "debug", "event_jsonl")):
        return "scratch"
    if any(
        token in lower
        for token in (
            "manifest",
            "receipt",
            "passport",
            "config",
            "completion",
            "contract",
            "execution_plan",
            "preflight",
            "metrics",
            "comparison",
            "runability",
        )
    ):
        return "control"
    return "unknown"


def _artifact_family(relative: str) -> str:
    lower = relative.lower()
    if "event_history_v1" in lower:
        return "native_runtime.event_history"
    if "lineage_reproduction_funnel_events" in lower:
        return "lineage_reproduction_funnel.events"
    if "lineage_reproduction_funnel_generation_summary" in lower:
        return "lineage_reproduction_funnel.generation_summary"
    if "typed_lifecycle_events" in lower:
        return "native_runtime.typed_lifecycle_events"
    if "compact_generation_summaries" in lower:
        return "native_runtime.compact_generation_summaries"
    return _zone(relative)


def _owner_writer(relative: str) -> str:
    lower = relative.lower()
    if "event_history_v1" in lower:
        return "native_resident_event_sink"
    if "lineage_reproduction_funnel" in lower:
        return "LineageReproductionFunnelWriter"
    if "typed_lifecycle_events" in lower:
        return "write_typed_lifecycle_public_outputs"
    if "compact_generation_summaries" in lower:
        return "native_cuda_architecture_b_runner_owner"
    return "run"


def _checkpoint_generation(relative: str) -> int | None:
    match = _CHECKPOINT_RE.search(relative)
    return int(match.group(1)) if match else None


def _checkpoint_role_map(contract: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    coordinate_field = (
        "checkpoint_tick"
        if contract["checkpoints"].get("authority") == "absolute_tick"
        else "generation"
    )
    return {
        int(row[coordinate_field]): dict(row)
        for row in contract["checkpoints"]["roles"]
        if isinstance(row, Mapping) and type(row.get(coordinate_field)) is int
    }


def _explicit_checkpoint_catalog_row(
    relative: str, checkpoint_catalog: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    normalized = relative.replace("\\", "/").rstrip("/")
    matches: list[dict[str, Any]] = []
    for item in checkpoint_catalog:
        if not isinstance(item, Mapping):
            continue
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        candidate = raw_path.replace("\\", "/").strip("/")
        if normalized == candidate or normalized.startswith(candidate + "/"):
            matches.append(dict(item))
    if not matches:
        return None
    matches.sort(key=lambda row: len(str(row.get("path") or "")), reverse=True)
    return matches[0]


def _artifact_class(
    relative: str,
    *,
    contract: Mapping[str, Any],
    terminal_status: str,
    live_reference_status: str,
    audit_hold_status: str,
    qualification_hold_status: str,
    replacement_verified: bool,
    checkpoint_catalog: Sequence[Mapping[str, Any]],
) -> tuple[str, str | None, str, bool]:
    lower = relative.lower()
    explicit_checkpoint = _explicit_checkpoint_catalog_row(relative, checkpoint_catalog)
    zone = "checkpoints" if explicit_checkpoint is not None else _zone(relative)
    role: str | None = None
    lifecycle_violation = False
    if zone == "unknown":
        return (
            "RX",
            None,
            "artifact family is not classified by the lifecycle contract",
            False,
        )
    if zone == "checkpoints":
        if explicit_checkpoint is not None:
            role_row = explicit_checkpoint
        elif checkpoint_catalog and (
            "checkpoint" in lower or "bundle_gen" in lower or "generation" in lower
        ):
            return (
                "RX",
                None,
                "checkpoint lacks an explicit lifecycle catalog identity/role",
                False,
            )
        else:
            generation = _checkpoint_generation(relative)
            role_row = (
                _checkpoint_role_map(contract).get(generation)
                if generation is not None
                else None
            )
        if role_row is None:
            return (
                "RX",
                None,
                "checkpoint has no explicit lifecycle identity/role",
                False,
            )
        role_value = role_row.get("role")
        if not isinstance(role_value, str) or role_value not in CHECKPOINT_ROLES:
            return "RX", None, "checkpoint has an unsupported lifecycle role", False
        if explicit_checkpoint is not None and (
            not isinstance(role_row.get("identity"), str)
            or not role_row["identity"].strip()
        ):
            return (
                "RX",
                role_value,
                "checkpoint lacks an explicit lifecycle identity",
                False,
            )
        role = role_value
        fallback_retention = None
        coordinate_value = role_row.get("checkpoint_tick", role_row.get("generation"))
        if type(coordinate_value) is int:
            fallback_retention = (
                _checkpoint_role_map(contract)
                .get(coordinate_value, {})
                .get("retention_class")
            )
        retention_class = str(
            role_row.get("retention_class") or fallback_retention or "RX"
        )
        if retention_class not in RETENTION_CLASSES:
            return (
                "RX",
                role,
                "checkpoint has an unsupported lifecycle retention class",
                False,
            )
    elif zone == "control":
        retention_class = "R5"
    elif zone == "closeout":
        retention_class = "R5" if "receipt" in lower or "manifest" in lower else "R3"
    elif zone == "durable":
        retention_class = "R3"
    elif zone == "cache":
        retention_class = "R2"
    else:
        retention_class = "R0"
        if any(token in lower for token in ("raw_tick", "event", "trace", "debug")):
            raw_policy = contract["observability"]["raw_tick_trace_policy"]
            event_policy = contract["observability"]["debug_event_trace_policy"]
            if "event" in lower or "debug" in lower:
                allowed = event_policy != "forbidden"
            else:
                allowed = raw_policy != "forbidden"
            if not allowed:
                lifecycle_violation = True
                return (
                    "RX",
                    role,
                    "undeclared trace violates the lifecycle observability policy",
                    False,
                )
    no_live_reference = live_reference_status == "none"
    # Audit/evidence holds protect durable decision material.  They do not
    # freeze ordinary R0-R2 scratch/rebuildable objects once their own
    # consumer/replacement predicates are satisfied.
    holds_released = (
        qualification_hold_status in {"released", "not_required"}
        or retention_class in {"R0", "R1", "R2"}
    ) and (
        audit_hold_status in {"released", "not_required"}
        or retention_class in {"R0", "R1", "R2"}
    )
    if retention_class in {"R0", "R1"}:
        eligible = (
            terminal_status == "completed"
            and no_live_reference
            and holds_released
            and not lifecycle_violation
        )
        reason = (
            "terminal, unreferenced, and holds released"
            if eligible
            else "requires clean terminal state, no references, and released holds"
        )
    elif retention_class == "R2":
        eligible = (
            terminal_status == "completed"
            and no_live_reference
            and replacement_verified
            and holds_released
        )
        reason = (
            "canonical/replacement verified and unreferenced"
            if eligible
            else "requires verified replacement, no references, and released holds"
        )
    elif retention_class == "R3":
        eligible = (
            terminal_status == "completed"
            and no_live_reference
            and replacement_verified
            and holds_released
        )
        reason = (
            "accepted compact replacement verified"
            if eligible
            else "requires accepted compact replacement and released holds"
        )
    elif retention_class == "R4":
        eligible = (
            terminal_status == "completed" and no_live_reference and holds_released
        )
        reason = (
            "explicit replay/qualification release state"
            if eligible
            else "replay or qualification release is still required"
        )
    else:
        eligible = False
        reason = "durable authority is never automatically released"
    return retention_class, role, reason, eligible


def _declared_artifact_decision(
    retention_class: str,
    *,
    terminal_status: str,
    live_reference_status: str,
    audit_hold_status: str,
    qualification_hold_status: str,
    replacement_verified: bool,
) -> tuple[str, str, bool]:
    if retention_class not in RETENTION_CLASSES:
        return "RX", "declared retention class is unsupported", False
    no_live_reference = live_reference_status == "none"
    holds_released = (
        qualification_hold_status in {"released", "not_required"}
        or retention_class in {"R0", "R1", "R2"}
    ) and (
        audit_hold_status in {"released", "not_required"}
        or retention_class in {"R0", "R1", "R2"}
    )
    if retention_class in {"R0", "R1"}:
        eligible = (
            terminal_status == "completed" and no_live_reference and holds_released
        )
        reason = (
            "terminal, unreferenced, and holds released"
            if eligible
            else "requires clean terminal state, no references, and released holds"
        )
    elif retention_class == "R2":
        eligible = (
            terminal_status == "completed"
            and no_live_reference
            and replacement_verified
            and holds_released
        )
        reason = (
            "canonical/replacement verified and unreferenced"
            if eligible
            else "requires verified replacement, no references, and released holds"
        )
    elif retention_class == "R3":
        eligible = (
            terminal_status == "completed"
            and no_live_reference
            and replacement_verified
            and holds_released
        )
        reason = (
            "accepted compact replacement verified; archive may replace source"
            if eligible
            else "durable compact evidence requires verified replacement and released holds"
        )
    elif retention_class == "R4":
        eligible = False
        reason = "replay or qualification-critical material remains protected"
    elif retention_class == "R5":
        eligible = False
        reason = "durable authority is never automatically released"
    else:
        eligible = False
        reason = "artifact purpose or ownership is unresolved"
    disposition = (
        "RELEASE"
        if eligible and retention_class in {"R0", "R1", "R2"}
        else "RETAIN_DURABLE"
        if retention_class in {"R3", "R4", "R5"}
        else "RX"
        if retention_class == "RX"
        else "KEEP_ACTIVE"
    )
    return disposition, reason, eligible


def _declared_inventory_rows(
    inventory: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    terminal_status: str,
    live_reference_status: str,
    audit_hold_status: str,
    qualification_hold_status: str,
    replacement_verified: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(inventory):
        if not isinstance(item, Mapping):
            raise ArtifactLifecycleError(f"artifact inventory row {index} is malformed")
        row = dict(item)
        relative = str(row.get("path") or "").replace("\\", "/").strip("/")
        if not relative or relative in seen:
            raise ArtifactLifecycleError(
                "artifact inventory paths must be non-empty and unique"
            )
        candidate = root / relative
        try:
            candidate.resolve(strict=False).relative_to(root)
        except ValueError as exc:
            raise ArtifactLifecycleError(
                f"artifact inventory path escapes runtime root: {relative}"
            ) from exc
        seen.add(relative)
        size = row.get("bytes")
        if type(size) is not int or size < 0:
            raise ArtifactLifecycleError(
                f"artifact inventory row {index} has invalid bytes"
            )
        sha256 = row.get("sha256")
        tree_sha256 = row.get("tree_sha256")
        if not (
            isinstance(sha256, str) and re.fullmatch(r"[0-9a-f]{64}", sha256)
        ) and not (
            isinstance(tree_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", tree_sha256)
        ):
            raise ArtifactLifecycleError(
                f"artifact inventory row {index} lacks a sealed SHA-256 identity"
            )
        retention_class = str(row.get("retention_class") or "RX")
        row_live = str(row.get("live_reference_status") or live_reference_status)
        row_audit = str(row.get("audit_hold_status") or audit_hold_status)
        row_qualification = str(
            row.get("qualification_hold_status") or qualification_hold_status
        )
        row_replacement = bool(row.get("replacement_verified", replacement_verified))
        disposition, reason, eligible = _declared_artifact_decision(
            retention_class,
            terminal_status=terminal_status,
            live_reference_status=row_live,
            audit_hold_status=row_audit,
            qualification_hold_status=row_qualification,
            replacement_verified=row_replacement,
        )
        artifact_type = str(
            row.get("artifact_type")
            or ("sealed_tree" if tree_sha256 is not None else "file")
        )
        rows.append(
            {
                "artifact_id": str(
                    row.get("artifact_id")
                    or f"artifact-{canonical_sha256({'path': relative, 'sha256': sha256 or tree_sha256})[:20]}"
                ),
                "path": relative,
                "artifact_family": str(row.get("artifact_family") or "declared"),
                "artifact_type": artifact_type,
                "owner_writer": str(row.get("owner_writer") or "declared-producer"),
                "bytes": size,
                **({"sha256": sha256} if isinstance(sha256, str) else {}),
                **(
                    {"tree_sha256": tree_sha256} if isinstance(tree_sha256, str) else {}
                ),
                "retention_class": retention_class
                if retention_class in RETENTION_CLASSES
                else "RX",
                "decision_evidence_class": row.get("decision_evidence_class"),
                "replacement_artifact_identity": row.get(
                    "replacement_artifact_identity"
                ),
                "live_reference_status": row_live,
                "audit_hold_status": row_audit,
                "qualification_hold_status": row_qualification,
                "release_predicate": str(row.get("release_predicate") or reason),
                "release_eligible": eligible,
                "exact_object_identity": True,
                "disposition": disposition,
                "reason": str(row.get("reason") or reason),
                "checkpoint_role": row.get("checkpoint_role"),
                "checkpoint_identity": row.get("checkpoint_identity"),
                "checkpoint_run_association": row.get("checkpoint_run_association"),
                **(
                    {
                        "evidence_certificate_id": row.get("evidence_certificate_id"),
                        "evidence_certificate_sha256": row.get(
                            "evidence_certificate_sha256"
                        ),
                        "evidence_class": row.get("evidence_class"),
                        "certificate_disposition": row.get("certificate_disposition"),
                        "downstream_consumers": list(
                            row.get("downstream_consumers", [])
                        ),
                        "last_use": row.get("last_use"),
                    }
                    if row.get("evidence_certificate_id") is not None
                    else {}
                ),
            }
        )
    return rows


def _aggregate_rows(
    rows: list[dict[str, Any]], *, max_objects: int
) -> list[dict[str, Any]]:
    if len(rows) <= max_objects:
        return sorted(rows, key=lambda row: str(row["path"]))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["artifact_family"]), str(row["retention_class"])), []
        ).append(row)
    result: list[dict[str, Any]] = []
    for (family, retention_class), members in sorted(grouped.items()):
        entries = [
            {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
            for row in sorted(members, key=lambda item: str(item["path"]))
        ]
        result.append(
            {
                "artifact_id": f"family-{canonical_sha256(entries)[:20]}",
                "path": family,
                "paths": [row["path"] for row in entries],
                "artifact_family": family,
                "artifact_type": "aggregated_family",
                "owner_writer": "run-family-inventory",
                "bytes": sum(int(row["bytes"]) for row in members),
                "manifest_sha256": canonical_sha256(entries),
                "retention_class": retention_class,
                "decision_evidence_class": None,
                "replacement_artifact_identity": None,
                "live_reference_status": members[0]["live_reference_status"],
                "audit_hold_status": members[0]["audit_hold_status"],
                "qualification_hold_status": members[0]["qualification_hold_status"],
                "release_predicate": members[0]["release_predicate"],
                "release_eligible": False,
                "exact_object_identity": False,
                "disposition": "RX",
                "reason": "aggregated legacy inventory is advisory and cannot authorize deletion",
                "checkpoint_role": members[0].get("checkpoint_role"),
                "checkpoint_identity": members[0].get("checkpoint_identity"),
                "checkpoint_run_association": members[0].get(
                    "checkpoint_run_association"
                ),
            }
        )
    return result


def build_artifact_disposition_manifest_v1(
    plan: Mapping[str, Any],
    *,
    run_root: Path,
    terminal_status: str = "unknown",
    live_reference_status: str = "unknown",
    audit_hold_status: str | None = None,
    qualification_hold_status: str | None = None,
    replacement_verified: bool = False,
    replacement_artifacts: Sequence[Mapping[str, Any]] = (),
    checkpoint_catalog: Sequence[Mapping[str, Any]] = (),
    artifact_inventory: Sequence[Mapping[str, Any]] | None = None,
    inventory_complete: bool = False,
    max_objects: int = 4096,
    legacy_reconstruction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = ensure_lifecycle_contract_v1(plan)
    root = run_root.expanduser().resolve()
    normalized_checkpoint_catalog: list[dict[str, Any]] = []
    for item in checkpoint_catalog:
        if not isinstance(item, Mapping):
            continue
        row = dict(item)
        raw_path = row.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            candidate = Path(raw_path).expanduser()
            if candidate.is_absolute():
                try:
                    row["path"] = str(candidate.resolve().relative_to(root))
                except ValueError:
                    row["path"] = str(candidate.resolve())
        normalized_checkpoint_catalog.append(row)
    audit_status = audit_hold_status or (
        "active"
        if contract["retention"]["audit_hold_binding"] == "required"
        else "not_required"
    )
    qualification_status = qualification_hold_status or (
        "active"
        if contract["retention"]["qualification_hold_binding"] == "required"
        else "not_required"
    )
    normalized_legacy_reconstruction = (
        validate_legacy_reconstruction_metadata_v1(legacy_reconstruction)
        if legacy_reconstruction is not None
        else None
    )
    rows: list[dict[str, Any]] = []
    inventory_mode = artifact_inventory is not None
    if artifact_inventory is not None:
        rows = _declared_inventory_rows(
            artifact_inventory,
            root=root,
            terminal_status=terminal_status,
            live_reference_status=live_reference_status,
            audit_hold_status=audit_status,
            qualification_hold_status=qualification_status,
            replacement_verified=replacement_verified,
        )
    elif root.exists():
        for path in sorted(root.rglob("*"), key=lambda item: str(item)):
            if not path.is_file() or path.is_symlink():
                continue
            relative = _relative(path, root)
            explicit_checkpoint = _explicit_checkpoint_catalog_row(
                relative, normalized_checkpoint_catalog
            )
            retention_class, checkpoint_role, reason, release_eligible = (
                _artifact_class(
                    relative,
                    contract=contract,
                    terminal_status=terminal_status,
                    live_reference_status=live_reference_status,
                    audit_hold_status=audit_status,
                    qualification_hold_status=qualification_status,
                    replacement_verified=replacement_verified,
                    checkpoint_catalog=normalized_checkpoint_catalog,
                )
            )
            size = int(path.stat().st_size)
            sha256 = _sha256_file(path)
            rows.append(
                {
                    "artifact_id": f"artifact-{canonical_sha256({'path': relative, 'sha256': sha256})[:20]}",
                    "path": relative,
                    "artifact_family": _artifact_family(relative),
                    "artifact_type": path.suffix.lower().lstrip(".") or "file",
                    "owner_writer": _owner_writer(relative),
                    "bytes": size,
                    "sha256": sha256,
                    "retention_class": retention_class,
                    "decision_evidence_class": contract["retention"][
                        "decision_evidence_class"
                    ],
                    "replacement_artifact_identity": None,
                    "live_reference_status": live_reference_status,
                    "audit_hold_status": audit_status,
                    "qualification_hold_status": qualification_status,
                    "release_predicate": reason,
                    "release_eligible": bool(
                        retention_class != "RX" and release_eligible
                    ),
                    "exact_object_identity": True,
                    "disposition": (
                        "RELEASE"
                        if retention_class in {"R0", "R1", "R2"} and release_eligible
                        else "RETAIN_DURABLE"
                        if retention_class in {"R3", "R4", "R5"}
                        else "RX"
                        if retention_class == "RX"
                        else "KEEP_ACTIVE"
                    ),
                    "reason": reason,
                    "checkpoint_role": checkpoint_role,
                    "checkpoint_identity": (
                        explicit_checkpoint.get("identity")
                        if explicit_checkpoint is not None
                        else (
                            _checkpoint_role_map(contract)
                            .get(_checkpoint_generation(relative), {})
                            .get("identity")
                            if _checkpoint_generation(relative) is not None
                            else None
                        )
                    ),
                    "checkpoint_run_association": (
                        {"campaign_id": plan.get("campaign_id"), "run_id": root.name}
                        if checkpoint_role is not None
                        else None
                    ),
                }
            )
    bounded_max_objects = max(1, int(max_objects))
    if inventory_mode and len(rows) > bounded_max_objects:
        raise ArtifactLifecycleError(
            "producer-declared inventory exceeds max_objects; register a sealed tree, segment, or bundle"
        )
    if not inventory_mode:
        rows = _aggregate_rows(rows, max_objects=bounded_max_objects)
    summary: dict[str, Any] = {
        "produced_bytes": sum(int(row["bytes"]) for row in rows),
        "release_eligible_bytes": sum(
            int(row["bytes"]) for row in rows if row["release_eligible"]
        ),
        "unresolved_rx_bytes": sum(
            int(row["bytes"]) for row in rows if row["retention_class"] == "RX"
        ),
        "by_retention_class": {
            retention_class: {
                "objects": sum(
                    1 for row in rows if row["retention_class"] == retention_class
                ),
                "bytes": sum(
                    int(row["bytes"])
                    for row in rows
                    if row["retention_class"] == retention_class
                ),
            }
            for retention_class in RETENTION_CLASSES
        },
        "lifecycle_violations": sum(
            1
            for row in rows
            if "violates the lifecycle observability policy" in str(row.get("reason"))
        ),
    }
    unknown_artifact_scope = contract["retention"].get(
        "unknown_artifact_scope", "global"
    )
    release_rows = [
        row
        for row in rows
        if row.get("release_eligible") is True
        and row.get("exact_object_identity") is True
        and row.get("retention_class") in {"R0", "R1", "R2"}
    ]
    audit_clear_for_release = audit_status in {"released", "not_required"} or bool(
        release_rows
        and all(
            row.get("retention_class") in {"R0", "R1", "R2"} for row in release_rows
        )
    )
    qualification_clear_for_release = qualification_status in {
        "released",
        "not_required",
    } or bool(
        release_rows
        and all(
            row.get("retention_class") in {"R0", "R1", "R2"} for row in release_rows
        )
    )
    prune = {
        "automatic_prune_allowed": bool(contract["prune"]["automatic_prune_allowed"]),
        "automatic_prune_eligible": bool(
            contract["prune"]["automatic_prune_allowed"]
            and (
                normalized_legacy_reconstruction is None
                or normalized_legacy_reconstruction["status"] == "ACCEPTED"
            )
            and (not inventory_mode or inventory_complete)
            and (
                unknown_artifact_scope == "exact_object"
                or summary["unresolved_rx_bytes"] == 0
            )
            and any(
                row["release_eligible"]
                and row.get("exact_object_identity") is True
                and row["retention_class"] in {"R0", "R1", "R2"}
                for row in rows
            )
            and audit_clear_for_release
            and qualification_clear_for_release
        ),
        "required_predicates": list(contract["prune"]["required_predicates"]),
        "actually_deleted": 0,
    }
    core = {
        "schema_version": DISPOSITION_SCHEMA_VERSION,
        "campaign_id": plan.get("campaign_id"),
        "run_id": root.name,
        "runtime_root": str(root),
        "plan_sha256": plan.get("plan_sha256"),
        "lifecycle_profile": contract["lifecycle_profile"],
        "lifecycle_contract_sha256": contract["contract_sha256"],
        "terminal_status": terminal_status,
        "inventory": {
            "mode": "producer_declared" if inventory_mode else "legacy_terminal_scan",
            "complete": bool(inventory_complete if inventory_mode else True),
        },
        "checkpoint_catalog": normalized_checkpoint_catalog,
        "artifacts": rows,
        "replacement_artifacts": [dict(row) for row in replacement_artifacts],
        "retention": {
            "audit_hold_status": audit_status,
            "qualification_hold_status": qualification_status,
            "unknown_artifact_policy": UNKNOWN_ARTIFACT_POLICY,
            "unknown_artifact_scope": unknown_artifact_scope,
        },
        "summary": summary,
        "prune": prune,
    }
    if normalized_legacy_reconstruction is not None:
        core["legacy_reconstruction"] = normalized_legacy_reconstruction
    return {**core, "manifest_sha256": canonical_sha256(core)}


def build_active_artifact_disposition_manifest_v1(
    plan: Mapping[str, Any],
    *,
    run_root: Path,
    artifact_inventory: Sequence[Mapping[str, Any]],
    active_release_authority: Mapping[str, Any],
    max_objects: int = 4096,
) -> dict[str, Any]:
    """Build an exact, nonterminal release authority for a safe boundary.

    This is deliberately distinct from terminal disposition.  It is accepted
    by the existing hot-release actuator only together with an in-process
    owner capability and an exact active-boundary authority.  It cannot be
    used by the ordinary terminal/archive path.
    """

    if not isinstance(active_release_authority, Mapping):
        raise ArtifactLifecycleError(
            "active release authority must be a canonical object"
        )
    try:
        authority = json.loads(
            json.dumps(
                dict(active_release_authority),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactLifecycleError(
            "active release authority is not canonical JSON"
        ) from exc
    if not isinstance(authority, dict) or not authority:
        raise ArtifactLifecycleError("active release authority is empty")
    if not artifact_inventory:
        raise ArtifactLifecycleError("active release inventory cannot be empty")
    candidate = build_artifact_disposition_manifest_v1(
        plan,
        run_root=run_root,
        terminal_status="completed",
        live_reference_status="none",
        audit_hold_status="not_required",
        qualification_hold_status="not_required",
        replacement_verified=True,
        artifact_inventory=artifact_inventory,
        inventory_complete=True,
        max_objects=max_objects,
    )
    core = dict(candidate)
    core.pop("manifest_sha256", None)
    core["terminal_status"] = "active"
    core["active_release_boundary"] = True
    core["active_release_authority"] = authority
    rows = []
    for raw_row in core["artifacts"]:
        row = dict(raw_row)
        if row.get("retention_class") not in {"R0", "R1", "R2"}:
            raise ArtifactLifecycleError(
                "active release authority contains a protected artifact"
            )
        if row.get("exact_object_identity") is not True:
            raise ArtifactLifecycleError(
                "active release authority contains a non-exact artifact"
            )
        if row.get("release_eligible") is not True:
            raise ArtifactLifecycleError(
                "active release inventory row is not release eligible"
            )
        row["release_predicate"] = (
            "active exact release boundary, decision settled, and no bound child "
            "references this parent package"
        )
        row["reason"] = row["release_predicate"]
        row["disposition"] = "RELEASE"
        rows.append(row)
    core["artifacts"] = rows
    prune = dict(core["prune"])
    prune["automatic_prune_allowed"] = True
    prune["automatic_prune_eligible"] = True
    prune["actually_deleted"] = 0
    core["prune"] = prune
    return {**core, "manifest_sha256": canonical_sha256(core)}


def validate_artifact_disposition_manifest_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactLifecycleError("disposition manifest must be an object")
    raw = dict(value)
    supplied = raw.pop("manifest_sha256", None)
    if raw.get("schema_version") != DISPOSITION_SCHEMA_VERSION:
        raise ArtifactLifecycleError("disposition manifest schema mismatch")
    active_boundary = raw.get("active_release_boundary", False)
    if active_boundary is True:
        if raw.get("terminal_status") != "active":
            raise ArtifactLifecycleError(
                "active release disposition must have active terminal status"
            )
        if not isinstance(raw.get("active_release_authority"), Mapping):
            raise ArtifactLifecycleError(
                "active release disposition lacks its exact authority"
            )
    elif active_boundary is not False or "active_release_authority" in raw:
        raise ArtifactLifecycleError(
            "active release disposition boundary fields are inconsistent"
        )
    artifacts = raw.get("artifacts")
    if not isinstance(artifacts, list):
        raise ArtifactLifecycleError("disposition manifest artifacts must be an array")
    for index, row in enumerate(artifacts):
        if not isinstance(row, Mapping):
            raise ArtifactLifecycleError(
                f"disposition manifest artifact {index} is malformed"
            )
        if row.get("retention_class") not in RETENTION_CLASSES:
            raise ArtifactLifecycleError(
                f"disposition artifact {index} has an unsupported retention class"
            )
        if row.get("retention_class") == "RX" and row.get("release_eligible") is True:
            raise ArtifactLifecycleError("RX artifacts cannot be release eligible")
        if (
            row.get("release_eligible") is True
            and row.get("exact_object_identity") is not True
        ):
            raise ArtifactLifecycleError(
                "release-eligible artifacts require an exact object identity"
            )
    if "legacy_reconstruction" in raw:
        raw["legacy_reconstruction"] = validate_legacy_reconstruction_metadata_v1(
            raw["legacy_reconstruction"]
        )
    if supplied != canonical_sha256(raw):
        raise ArtifactLifecycleError("disposition manifest SHA-256 mismatch")
    return {**raw, "manifest_sha256": supplied}


def apply_legacy_evidence_review_v1(
    manifest: Mapping[str, Any],
    review_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Promote one reviewed legacy candidate into normal lifecycle authority.

    The receipt must account for every reconstructed row.  This is the
    one-time semantic bridge for roots created before lifecycle manifests
    existed; it never changes the existing prune/archive actuator.
    """

    candidate = validate_artifact_disposition_manifest_v1(manifest)
    metadata = candidate.get("legacy_reconstruction")
    if not isinstance(metadata, Mapping):
        raise ArtifactLifecycleError(
            "legacy evidence review can only promote a reconstructed manifest"
        )
    if metadata.get("status") != "REQUIRES_REVIEW":
        raise ArtifactLifecycleError("legacy manifest is not awaiting review")
    receipt = validate_legacy_evidence_review_receipt_v1(review_receipt)
    if receipt["manifest_sha256"] != candidate["manifest_sha256"]:
        raise ArtifactLifecycleError(
            "legacy review receipt is bound to a different candidate manifest"
        )
    runtime_root = candidate.get("runtime_root")
    if (
        not isinstance(runtime_root, str)
        or Path(runtime_root).expanduser().resolve()
        != Path(receipt["runtime_root"]).expanduser().resolve()
    ):
        raise ArtifactLifecycleError(
            "legacy review receipt is bound to a different runtime root"
        )
    decisions = {str(row["path"]): dict(row) for row in receipt["artifact_decisions"]}
    manifest_rows = [row for row in candidate["artifacts"] if isinstance(row, Mapping)]
    manifest_paths = {str(row.get("path")) for row in manifest_rows}
    if set(decisions) != manifest_paths:
        missing = sorted(manifest_paths - set(decisions))
        extra = sorted(set(decisions) - manifest_paths)
        raise ArtifactLifecycleError(
            f"legacy review must decide every reconstructed row (missing={missing}, extra={extra})"
        )
    if receipt["decision"] != "accepted":
        rejected = dict(candidate)
        rejected_metadata = dict(metadata)
        rejected_metadata["status"] = "REJECTED"
        rejected_review = dict(rejected_metadata["review"])
        rejected_review.update(
            {
                "status": "rejected",
                "reviewer": receipt["reviewer"],
                "accepted_at_utc": None,
                "receipt_sha256": receipt["receipt_sha256"],
                "allow_automatic_release": False,
            }
        )
        rejected_metadata["review"] = rejected_review
        rejected["legacy_reconstruction"] = rejected_metadata
        rejected.pop("manifest_sha256", None)
        return {**rejected, "manifest_sha256": canonical_sha256(rejected)}

    if receipt["allow_automatic_release"] and receipt["terminal_status"] != "completed":
        raise ArtifactLifecycleError(
            "automatic legacy release requires a completed terminal status"
        )
    if (
        receipt["allow_automatic_release"]
        and receipt["live_reference_status"] != "none"
    ):
        raise ArtifactLifecycleError(
            "automatic legacy release requires no live references"
        )
    root = Path(runtime_root).expanduser().resolve()
    if receipt["allow_automatic_release"] and RunStorageGuard.run_root_is_reserved(
        root
    ):
        raise ArtifactLifecycleError(
            "active run storage reservation blocks legacy review acceptance"
        )

    updated_rows: list[dict[str, Any]] = []
    for original in manifest_rows:
        row = dict(original)
        decision = decisions[str(row["path"])]
        if (
            decision["release_eligible"]
            and row.get("exact_object_identity") is not True
        ):
            raise ArtifactLifecycleError(
                f"legacy review cannot release a non-exact row: {row['path']}"
            )
        if decision["release_eligible"] and decision["live_reference_status"] != "none":
            raise ArtifactLifecycleError(
                f"legacy release row retains a live-reference hold: {row['path']}"
            )
        row.update(
            {
                "retention_class": decision["retention_class"],
                "disposition": decision["disposition"],
                "release_eligible": decision["release_eligible"],
                "release_predicate": decision["reason"],
                "reason": decision["reason"],
                "live_reference_status": decision["live_reference_status"],
                "replacement_verified": decision["replacement_verified"],
                "decision_evidence_class": decision["decision_evidence_class"],
                "replacement_artifact_identity": decision[
                    "replacement_artifact_identity"
                ],
                "consumer_references": decision["consumer_references"],
                "reproducible": decision["reproducible"],
            }
        )
        updated_rows.append(row)

    legacy_metadata = dict(metadata)
    legacy_metadata["status"] = "ACCEPTED"
    legacy_review = dict(legacy_metadata["review"])
    legacy_review.update(
        {
            "status": "accepted",
            "receipt_sha256": receipt["receipt_sha256"],
            "reviewer": receipt["reviewer"],
            "accepted_at_utc": receipt["reviewed_at_utc"],
            "allow_automatic_release": receipt["allow_automatic_release"],
        }
    )
    legacy_metadata["review"] = legacy_review
    core = dict(candidate)
    core.pop("manifest_sha256", None)
    core["terminal_status"] = receipt["terminal_status"]
    core["artifacts"] = updated_rows
    retention = dict(core.get("retention") or {})
    retention["audit_hold_status"] = receipt["audit_hold_status"]
    retention["qualification_hold_status"] = receipt["qualification_hold_status"]
    if receipt["allow_automatic_release"]:
        # Every row has been enumerated by the review receipt, so residual RX
        # rows are exact objects rather than an unknown tree-wide scope.
        retention["unknown_artifact_scope"] = "exact_object"
    core["retention"] = retention
    core["legacy_reconstruction"] = legacy_metadata

    release_rows = [
        row
        for row in updated_rows
        if row.get("release_eligible") is True
        and row.get("exact_object_identity") is True
        and row.get("retention_class") in {"R0", "R1", "R2"}
    ]
    unresolved_nonexact_rx = any(
        row.get("retention_class") == "RX"
        and row.get("exact_object_identity") is not True
        for row in updated_rows
    )
    prune = dict(core.get("prune") or {})
    prune["automatic_prune_allowed"] = receipt["allow_automatic_release"]
    prune["automatic_prune_eligible"] = bool(
        receipt["allow_automatic_release"]
        and core["inventory"].get("complete") is True
        and core["terminal_status"] == "completed"
        and not unresolved_nonexact_rx
        and bool(release_rows)
        and (
            receipt["qualification_hold_status"] in {"released", "not_required"}
            or all(
                row.get("retention_class") in {"R0", "R1", "R2"} for row in release_rows
            )
        )
    )
    prune["actually_deleted"] = 0
    core["prune"] = prune
    core["summary"] = {
        "produced_bytes": sum(int(row.get("bytes", 0)) for row in updated_rows),
        "release_eligible_bytes": sum(
            int(row.get("bytes", 0))
            for row in updated_rows
            if row.get("release_eligible") is True
        ),
        "unresolved_rx_bytes": sum(
            int(row.get("bytes", 0))
            for row in updated_rows
            if row.get("retention_class") == "RX"
        ),
        "by_retention_class": {
            retention_class: {
                "objects": sum(
                    1
                    for row in updated_rows
                    if row.get("retention_class") == retention_class
                ),
                "bytes": sum(
                    int(row.get("bytes", 0))
                    for row in updated_rows
                    if row.get("retention_class") == retention_class
                ),
            }
            for retention_class in RETENTION_CLASSES
        },
        "lifecycle_violations": sum(
            1
            for row in updated_rows
            if "violates the lifecycle observability policy" in str(row.get("reason"))
        ),
    }
    return {**core, "manifest_sha256": canonical_sha256(core)}


def disposition_manifest_authorizes_prune_v1(
    manifest: Mapping[str, Any],
    *,
    selected_paths: Sequence[str] | None = None,
    allow_active_release: bool = False,
) -> dict[str, Any]:
    value = validate_artifact_disposition_manifest_v1(manifest)
    active_boundary = value.get("active_release_boundary") is True
    if active_boundary and not allow_active_release:
        raise ArtifactLifecycleError(
            "active release disposition requires the owning boundary actuator"
        )
    if allow_active_release and not active_boundary:
        raise ArtifactLifecycleError(
            "active release actuator requires an active release disposition"
        )
    prune = value.get("prune")
    retention = value.get("retention")
    if not isinstance(prune, Mapping) or not isinstance(retention, Mapping):
        raise ArtifactLifecycleError(
            "disposition manifest lacks prune/retention authority"
        )
    if not legacy_reconstruction_is_accepted_v1(value):
        raise ArtifactLifecycleError(
            "legacy reconstructed disposition requires an accepted evidence review"
        )
    if (
        prune.get("automatic_prune_allowed") is not True
        or prune.get("automatic_prune_eligible") is not True
    ):
        raise ArtifactLifecycleError(
            "disposition manifest does not authorize destructive cleanup"
        )
    rows = value["artifacts"]
    selected = (
        None
        if selected_paths is None
        else {str(path).replace("\\", "/") for path in selected_paths}
    )
    if selected is None:
        selected_rows = [
            row
            for row in rows
            if row.get("release_eligible") is True
            and row.get("exact_object_identity") is True
        ]
    else:
        selected_rows = [row for row in rows if str(row.get("path")) in selected]
        if len(selected_rows) != len(selected):
            raise ArtifactLifecycleError(
                "selected prune path is absent from disposition authority"
            )
    blocked = [
        row
        for row in selected_rows
        if row.get("retention_class") == "RX"
        or row.get("release_eligible") is not True
        or row.get("exact_object_identity") is not True
    ]
    if blocked:
        raise ArtifactLifecycleError(
            "disposition authority contains protected or ineligible selected artifacts"
        )
    if retention.get("qualification_hold_status") not in {
        "released",
        "not_required",
    } and any(
        row.get("retention_class") not in {"R0", "R1", "R2"} for row in selected_rows
    ):
        raise ArtifactLifecycleError(
            "qualification hold blocks destructive cleanup for qualification evidence"
        )
    if retention.get("audit_hold_status") not in {"released", "not_required"} and any(
        row.get("retention_class") not in {"R0", "R1", "R2"} for row in selected_rows
    ):
        raise ArtifactLifecycleError(
            "audit hold blocks destructive cleanup for durable evidence"
        )
    return {
        "schema_version": "eiel_disposition_prune_authorization_v1",
        "manifest_sha256": value["manifest_sha256"],
        "selected_count": len(selected_rows),
        "selected_bytes": sum(int(row.get("bytes", 0)) for row in selected_rows),
        "decision": "allowed",
    }


def build_storage_disposition_receipt_v1(
    *,
    manifest: Mapping[str, Any],
    bytes_before: int,
    bytes_retained: int,
    bytes_released: int,
    bytes_unresolved_rx: int,
    removed: Sequence[Mapping[str, Any]] = (),
    archived: Sequence[Mapping[str, Any]] = (),
    action_failures: Sequence[Mapping[str, Any]] = (),
    skipped_paths: Sequence[str] = (),
    authorization_reason: str,
    replacement_references: Sequence[Mapping[str, Any]] = (),
    retention_references: Sequence[str] = (),
    plan_sha256: str | None = None,
    timestamp_utc: str | None = None,
    result: str = "dry_run",
) -> dict[str, Any]:
    validated = validate_artifact_disposition_manifest_v1(manifest)
    core = {
        "schema_version": STORAGE_RECEIPT_SCHEMA_VERSION,
        "campaign_id": validated.get("campaign_id"),
        "run_id": validated.get("run_id"),
        "lifecycle_contract_sha256": validated.get("lifecycle_contract_sha256"),
        "manifest_sha256": validated["manifest_sha256"],
        "plan_sha256": plan_sha256 or validated.get("plan_sha256"),
        "bytes_before": int(bytes_before),
        "bytes_retained": int(bytes_retained),
        "bytes_released": int(bytes_released),
        "bytes_unresolved_RX": int(bytes_unresolved_rx),
        "removed": [dict(row) for row in removed],
        "authorization_reason": authorization_reason,
        "replacement_references": [dict(row) for row in replacement_references],
        "retention_references": list(retention_references),
        "timestamp_utc": timestamp_utc or "UNSET",
        "result": result,
    }
    if archived:
        core["archived"] = [dict(row) for row in archived]
    if action_failures:
        core["action_failures"] = [dict(row) for row in action_failures]
    if skipped_paths:
        core["skipped_paths"] = [str(path) for path in skipped_paths]
    if any(
        int(core[name]) < 0
        for name in (
            "bytes_before",
            "bytes_retained",
            "bytes_released",
            "bytes_unresolved_RX",
        )
    ):
        raise ArtifactLifecycleError(
            "storage disposition byte counters cannot be negative"
        )
    return {**core, "receipt_sha256": canonical_sha256(core)}


def write_storage_disposition_receipt_v1(
    path: Path, receipt: Mapping[str, Any]
) -> Path:
    value = dict(receipt)
    expected = value.pop("receipt_sha256", None)
    if expected != canonical_sha256(value):
        raise ArtifactLifecycleError("storage disposition receipt SHA-256 mismatch")
    return _atomic_json(path, {**value, "receipt_sha256": expected})


def _run_artifact_transaction_lock_path_v1(run_root: Path) -> Path:
    requested = Path(os.path.abspath(os.path.expanduser(os.fspath(run_root))))
    resolved = requested.resolve(strict=False)
    is_data2 = (
        requested == DATA2_FILESYSTEM_ROOT
        or DATA2_FILESYSTEM_ROOT in requested.parents
        or resolved == DATA2_FILESYSTEM_ROOT
        or DATA2_FILESYSTEM_ROOT in resolved.parents
    )
    if is_data2:
        try:
            resolved = validate_data2_run_root_v1(requested)
        except StorageAvailabilityPolicyError as exc:
            raise ArtifactLifecycleError(str(exc)) from exc
        lock_root = DATA2_LIFECYCLE_MANAGEMENT_ROOT / "transactions"
    else:
        lock_root = resolved.parent / ".eiel-lifecycle-transactions"
        for candidate in (resolved, *resolved.parents):
            if candidate.name == "causyl-runtime":
                lock_root = candidate / "lifecycle" / "transactions"
                break
    identity = canonical_sha256(
        {
            "schema_version": "eiel_run_artifact_transaction_identity_v1",
            "run_root": str(resolved),
        }
    )
    return lock_root / f"{identity}.lock"


def _run_roots_overlap_v1(left: Path | str, right: Path | str) -> bool:
    left_root = Path(os.path.abspath(os.path.expanduser(os.fspath(left)))).resolve(
        strict=False
    )
    right_root = Path(os.path.abspath(os.path.expanduser(os.fspath(right)))).resolve(
        strict=False
    )
    return (
        left_root == right_root
        or left_root in right_root.parents
        or right_root in left_root.parents
    )


@contextmanager
def run_artifact_transaction_fence_v1(
    run_root: Path, *, blocking: bool = True
) -> Iterator[None]:
    """Serialize reseal, actuation, final receipt, and pointer consumption per run."""

    if type(blocking) is not bool:
        raise ArtifactLifecycleError("transaction fence blocking flag must be boolean")
    lock_path = _run_artifact_transaction_lock_path_v1(run_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.parent.is_symlink():
        raise ArtifactLifecycleError(
            "run artifact transaction root cannot be a symlink"
        )
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
        0o600,
    )
    handle = os.fdopen(descriptor, "a+b")
    flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
        fcntl.flock(handle.fileno(), flags)
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def register_pending_durable_archive_v1(
    manifest_path: Path,
    *,
    archive_root: Path | None = None,
    _transaction_fence_held: bool = False,
) -> Path | None:
    """Register one terminal manifest for the bounded archive/release worker.

    This is an atomic pointer, not a second artifact registry.  The worker
    reads only these pointers and the exact manifest-bound objects.  A run
    with no durable rows is still registered when its accepted disposition
    authorizes exact R0-R2 release.
    """

    manifest_file = manifest_path.expanduser().resolve()
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactLifecycleError(
            f"cannot read disposition manifest: {manifest_file}"
        ) from exc
    validated = validate_artifact_disposition_manifest_v1(manifest)
    if not legacy_reconstruction_is_accepted_v1(validated):
        raise ArtifactLifecycleError(
            "legacy reconstructed durable material requires an accepted evidence review"
        )
    try:
        root = Path(str(validated["runtime_root"])).expanduser().resolve()
        manifest_file.relative_to(root)
    except (KeyError, ValueError, OSError) as exc:
        raise ArtifactLifecycleError(
            "disposition manifest is not contained by its runtime root"
        ) from exc
    if type(_transaction_fence_held) is not bool:
        raise ArtifactLifecycleError("transaction fence ownership flag must be boolean")
    if not _transaction_fence_held:
        with run_artifact_transaction_fence_v1(root):
            return register_pending_durable_archive_v1(
                manifest_file,
                archive_root=archive_root,
                _transaction_fence_held=True,
            )
    durable = [
        row
        for row in validated.get("artifacts", [])
        if isinstance(row, Mapping)
        and row.get("retention_class") in {"R3", "R4", "R5"}
        and row.get("exact_object_identity") is True
    ]
    releasable = [
        row
        for row in validated.get("artifacts", [])
        if isinstance(row, Mapping)
        and row.get("retention_class") in {"R0", "R1", "R2"}
        and row.get("release_eligible") is True
        and row.get("exact_object_identity") is True
    ]
    if not durable and not (
        releasable
        and validated.get("prune", {}).get("automatic_prune_eligible") is True
    ):
        return None
    # Disposition manifests may be sealed under a nested control/evidence
    # directory while their inventory and receipts are bound to the owning
    # runtime root.  Bind the pointer to that declared runtime root rather
    # than to the manifest's storage directory.
    if root == DATA2_FILESYSTEM_ROOT or DATA2_FILESYSTEM_ROOT in root.parents:
        lifecycle_root = Path(
            os.path.abspath(os.fspath(DATA2_LIFECYCLE_MANAGEMENT_ROOT.expanduser()))
        )
        if lifecycle_root.resolve(strict=False) != lifecycle_root or not (
            lifecycle_root == DATA2_FILESYSTEM_ROOT
            or DATA2_FILESYSTEM_ROOT in lifecycle_root.parents
        ):
            raise ArtifactLifecycleError(
                "canonical configured runtime root lifecycle queue root is aliased or escapes"
            )
        pending_root = lifecycle_root / "pending"
    else:
        pending_root: Path | None = None
        for candidate in (root, *root.parents):
            if candidate.name == "causyl-runtime":
                pending_root = candidate / "lifecycle" / "pending"
                break
        if pending_root is None:
            pending_root = root.parent / ".eiel-lifecycle-pending"
    configured_archive_root = (
        archive_root.expanduser()
        if archive_root is not None
        else Path(
            os.environ.get(
                "EIEL_COLD_STORAGE_ROOT",
                "/mnt/Projects/_Emergence-cold-storage/eiel-artifacts",
            )
        ).expanduser()
    )
    # Pointer registration is part of the fast terminal closeout boundary.
    # Keep this conversion lexical: Path.resolve() probes every path component
    # and can synchronously block the run on an unavailable FUSE/HDD archive.
    # The detached archive worker owns archive-root resolution and I/O.
    configured_archive_root = Path(os.path.abspath(os.fspath(configured_archive_root)))
    pointer_core = {
        "schema_version": "eiel_durable_archive_pointer_v1",
        "run_root": str(root),
        "manifest_path": str(manifest_file),
        "manifest_sha256": validated["manifest_sha256"],
        "archive_root": str(configured_archive_root),
        "registered_at_utc": _now(),
    }
    pointer = {
        **pointer_core,
        "pointer_sha256": canonical_sha256(pointer_core),
    }
    root_identity = canonical_sha256({"run_root": str(root)})[:16]
    pointer_prefix = f"{root.name}-{root_identity}-"
    pointer_path = pending_root / (
        f"{pointer_prefix}{validated['manifest_sha256'][:16]}.json"
    )
    written = _atomic_json(pointer_path, pointer)

    # A bounded terminal reseal replaces the manifest authority for this exact
    # run. Preserve older pointers as audit evidence, but remove them from the
    # active worker queue so they cannot become permanent fail-closed retries.
    for prior_path in sorted(pending_root.glob("*.json")):
        if prior_path == written:
            continue
        try:
            prior_stat = os.lstat(prior_path)
            if not stat.S_ISREG(prior_stat.st_mode) or prior_stat.st_nlink != 1:
                continue
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(prior, Mapping):
            continue
        prior_core = dict(prior)
        prior_sha256 = prior_core.pop("pointer_sha256", None)
        if (
            prior.get("schema_version") != "eiel_durable_archive_pointer_v1"
            or not isinstance(prior_sha256, str)
            or prior_sha256 != canonical_sha256(prior_core)
            or prior.get("run_root") != str(root)
            or prior.get("manifest_sha256") == validated["manifest_sha256"]
        ):
            continue
        superseded_root = pending_root / "superseded"
        superseded_root.mkdir(parents=True, exist_ok=True)
        superseded_path = superseded_root / prior_path.name
        suffix = 0
        while superseded_path.exists() or superseded_path.is_symlink():
            suffix += 1
            superseded_path = superseded_root / (
                f"{prior_path.stem}-{prior_sha256[:12]}-{suffix}{prior_path.suffix}"
            )
        os.replace(prior_path, superseded_path)
        for directory in (superseded_root, pending_root):
            directory_fd = os.open(
                directory,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    return written


def _path_zone(relative: str) -> str:
    zone = _zone(relative)
    return (
        "scratch"
        if zone in {"unknown", "cache", "scratch", "quarantine"}
        else "checkpoints"
        if zone == "checkpoints"
        else "durable"
    )


def progressive_storage_reservation_window_bytes_v1(
    estimate: Mapping[str, Any] | None,
) -> int | None:
    """Validate and return a provider-owned incremental reservation window."""

    if not isinstance(estimate, Mapping):
        return None
    mode = estimate.get("storage_reservation_mode")
    if mode is None:
        return None
    if mode != PROGRESSIVE_STORAGE_RESERVATION_MODE_V1:
        raise ArtifactLifecycleError(
            "resource estimate storage_reservation_mode is unsupported"
        )
    if estimate.get("storage_reservation_guard_scope") != (
        "provider_dispatch_and_checkpoint_boundary_v1"
    ):
        raise ArtifactLifecycleError(
            "progressive storage reservation lacks its provider guard scope"
        )
    names = (
        "storage_reservation_window_bytes_upper_bound",
        "storage_reservation_control_bytes_upper_bound",
        "storage_reservation_dispatch_increment_bytes_upper_bound",
        "storage_reservation_boundary_increment_bytes_upper_bound",
        "storage_reservation_emergency_checkpoint_bytes_upper_bound",
        "storage_reservation_total_envelope_bytes_upper_bound",
    )
    values = {name: estimate.get(name) for name in names}
    if any(type(value) is not int or value < 0 for value in values.values()):
        raise ArtifactLifecycleError(
            "progressive storage reservation byte bounds are invalid"
        )
    window = int(values["storage_reservation_window_bytes_upper_bound"])
    total = int(values["storage_reservation_total_envelope_bytes_upper_bound"])
    disk = estimate.get("disk_bytes_upper_bound")
    expected_window = (
        int(values["storage_reservation_control_bytes_upper_bound"])
        + int(values["storage_reservation_dispatch_increment_bytes_upper_bound"])
        + max(
            int(values["storage_reservation_boundary_increment_bytes_upper_bound"]),
            int(values["storage_reservation_emergency_checkpoint_bytes_upper_bound"]),
        )
    )
    if (
        window <= 0
        or type(disk) is not int
        or disk <= 0
        or total != disk
        or window != expected_window
        or window > total
    ):
        raise ArtifactLifecycleError(
            "progressive storage reservation arithmetic differs from the total envelope"
        )
    return window


def _exact_storage_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    path: str,
) -> None:
    if set(value) != expected:
        raise ArtifactLifecycleError(
            f"{path} fields differ: expected {sorted(expected)}, got {sorted(value)}"
        )


def validate_process_owner_identity_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactLifecycleError("process owner identity must be an object")
    raw = dict(value)
    supplied_sha256 = raw.pop("identity_sha256", None)
    _exact_storage_fields(
        raw,
        {"schema_version", "boot_id", "pid", "process_start_ticks"},
        path="process owner identity",
    )
    if raw.get("schema_version") != PROCESS_OWNER_IDENTITY_SCHEMA_VERSION:
        raise ArtifactLifecycleError("process owner identity schema is unsupported")
    boot_id = raw.get("boot_id")
    if (
        not isinstance(boot_id, str)
        or re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            boot_id,
        )
        is None
    ):
        raise ArtifactLifecycleError("process owner identity boot_id is invalid")
    if type(raw.get("pid")) is not int or raw["pid"] <= 0:
        raise ArtifactLifecycleError("process owner identity pid is invalid")
    if (
        type(raw.get("process_start_ticks")) is not int
        or raw["process_start_ticks"] < 0
    ):
        raise ArtifactLifecycleError(
            "process owner identity process_start_ticks is invalid"
        )
    if not isinstance(supplied_sha256, str) or supplied_sha256 != canonical_sha256(raw):
        raise ArtifactLifecycleError("process owner identity SHA-256 mismatch")
    return {**raw, "identity_sha256": supplied_sha256}


def observe_process_owner_identity_v1(pid: int) -> dict[str, Any]:
    if type(pid) is not int or pid <= 0:
        raise ArtifactLifecycleError("process owner identity pid is invalid")
    try:
        boot_id = (
            Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        )
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ProcessLookupError(pid) from exc
    except (OSError, UnicodeError) as exc:
        raise ArtifactLifecycleError(
            f"cannot observe process owner identity for PID {pid}"
        ) from exc
    closing_parenthesis = stat_text.rfind(")")
    fields = (
        stat_text[closing_parenthesis + 2 :].split() if closing_parenthesis >= 0 else []
    )
    if len(fields) <= 19:
        raise ArtifactLifecycleError(
            f"cannot parse process start identity for PID {pid}"
        )
    try:
        process_start_ticks = int(fields[19])
    except ValueError as exc:
        raise ArtifactLifecycleError(
            f"cannot parse process start identity for PID {pid}"
        ) from exc
    core = {
        "schema_version": PROCESS_OWNER_IDENTITY_SCHEMA_VERSION,
        "boot_id": boot_id,
        "pid": pid,
        "process_start_ticks": process_start_ticks,
    }
    return validate_process_owner_identity_v1(
        {**core, "identity_sha256": canonical_sha256(core)}
    )


def _validate_storage_availability_policy_binding_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactLifecycleError(
            "storage availability policy binding must be an object"
        )
    raw = dict(value)
    _exact_storage_fields(
        raw,
        {"policy_id", "policy_version", "policy_sha256"},
        path="storage availability policy binding",
    )
    if not isinstance(raw.get("policy_id"), str) or not raw["policy_id"]:
        raise ArtifactLifecycleError("storage availability policy_id is invalid")
    if type(raw.get("policy_version")) is not int or raw["policy_version"] <= 0:
        raise ArtifactLifecycleError("storage availability policy_version is invalid")
    if (
        _optional_sha256(
            raw.get("policy_sha256"),
            path="storage availability policy binding.policy_sha256",
        )
        is None
    ):
        raise ArtifactLifecycleError("storage availability policy SHA-256 is required")
    return raw


def _data2_reservation_id_v1(
    *,
    run_root: Path,
    lifecycle_contract_sha256: str,
    storage_availability_policy: Mapping[str, Any],
    filesystem_identity: Mapping[str, Any],
    measured_run_filesystem_identity: Mapping[str, Any],
) -> str:
    return canonical_sha256(
        {
            "run_root": str(validate_data2_run_root_v1(run_root)),
            "lifecycle_contract_sha256": lifecycle_contract_sha256,
            "storage_availability_policy": dict(storage_availability_policy),
            "filesystem_identity": dict(filesystem_identity),
            "measured_run_filesystem_identity": dict(measured_run_filesystem_identity),
        }
    )[:24]


def build_data2_storage_reservation_v1(
    *,
    run_root: Path,
    lifecycle_contract_sha256: str,
    planned_bytes: int,
    remaining_bytes: int,
    owner_identity: Mapping[str, Any],
    updated_at_utc: str,
    storage_availability_policy: Mapping[str, Any],
    filesystem_identity: Mapping[str, Any],
    measured_run_filesystem_identity: Mapping[str, Any],
    filesystem_total_bytes: int,
    filesystem_free_bytes: int,
    active_other_remaining_bytes: int,
    intrinsic_protected_free_floor_bytes: int,
    admission_protected_free_floor_bytes: int,
    usable_headroom_bytes: int,
    admissible_headroom_bytes: int,
    post_candidate_headroom_bytes: int,
) -> dict[str, Any]:
    root = validate_data2_run_root_v1(run_root)
    contract_sha256 = _optional_sha256(
        lifecycle_contract_sha256,
        path="data2 reservation.lifecycle_contract_sha256",
    )
    if contract_sha256 is None:
        raise ArtifactLifecycleError(
            "data2 reservation.lifecycle_contract_sha256 is required"
        )
    policy_binding = _validate_storage_availability_policy_binding_v1(
        storage_availability_policy
    )
    canonical_identity = validate_filesystem_identity_v1(filesystem_identity)
    measured_identity = validate_filesystem_identity_v1(
        measured_run_filesystem_identity
    )
    if measured_identity != canonical_identity:
        raise ArtifactLifecycleError(
            "measured run filesystem does not match the bound configured runtime root filesystem"
        )
    owner = validate_process_owner_identity_v1(owner_identity)
    for name, value in (
        ("planned_bytes", planned_bytes),
        ("remaining_bytes", remaining_bytes),
        ("filesystem_total_bytes", filesystem_total_bytes),
        ("filesystem_free_bytes", filesystem_free_bytes),
        ("active_other_remaining_bytes", active_other_remaining_bytes),
        ("intrinsic_protected_free_floor_bytes", intrinsic_protected_free_floor_bytes),
        ("admission_protected_free_floor_bytes", admission_protected_free_floor_bytes),
        ("usable_headroom_bytes", usable_headroom_bytes),
        ("admissible_headroom_bytes", admissible_headroom_bytes),
        ("post_candidate_headroom_bytes", post_candidate_headroom_bytes),
    ):
        if type(value) is not int or value < 0:
            raise ArtifactLifecycleError(
                f"data2 reservation.{name} must be a non-negative integer"
            )
    if filesystem_total_bytes <= 0 or filesystem_free_bytes > filesystem_total_bytes:
        raise ArtifactLifecycleError("data2 reservation filesystem capacity is invalid")
    if remaining_bytes > planned_bytes:
        raise ArtifactLifecycleError(
            "data2 reservation remaining_bytes exceeds planned_bytes"
        )
    if admission_protected_free_floor_bytes < intrinsic_protected_free_floor_bytes:
        raise ArtifactLifecycleError(
            "data2 reservation admission floor cannot weaken its intrinsic floor"
        )
    expected_usable = max(
        0, filesystem_free_bytes - admission_protected_free_floor_bytes
    )
    if usable_headroom_bytes != expected_usable:
        raise ArtifactLifecycleError(
            "data2 reservation usable headroom is inconsistent"
        )
    expected_admissible = max(
        0,
        usable_headroom_bytes - active_other_remaining_bytes,
    )
    if admissible_headroom_bytes != expected_admissible:
        raise ArtifactLifecycleError(
            "data2 reservation admissible headroom is inconsistent"
        )
    if remaining_bytes > admissible_headroom_bytes:
        raise ArtifactLifecycleError(
            "data2 reservation remaining bytes exceed admissible headroom"
        )
    if post_candidate_headroom_bytes != admissible_headroom_bytes - remaining_bytes:
        raise ArtifactLifecycleError(
            "data2 reservation post-candidate headroom is inconsistent"
        )
    timestamp = _text(updated_at_utc, path="data2 reservation.updated_at_utc")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArtifactLifecycleError(
            "data2 reservation.updated_at_utc must be an ISO-8601 timestamp"
        ) from exc
    if not timestamp.endswith(
        "Z"
    ) or parsed_timestamp.utcoffset() != timezone.utc.utcoffset(parsed_timestamp):
        raise ArtifactLifecycleError(
            "data2 reservation.updated_at_utc must be normalized UTC"
        )
    reservation_id = _data2_reservation_id_v1(
        run_root=root,
        lifecycle_contract_sha256=contract_sha256,
        storage_availability_policy=policy_binding,
        filesystem_identity=canonical_identity,
        measured_run_filesystem_identity=measured_identity,
    )
    core = {
        "schema_version": DATA2_STORAGE_RESERVATION_SCHEMA_VERSION,
        "reservation_id": reservation_id,
        "run_root": str(root),
        "lifecycle_contract_sha256": contract_sha256,
        "planned_bytes": planned_bytes,
        "remaining_bytes": remaining_bytes,
        "pid": owner["pid"],
        "owner_identity": owner,
        "status": "active",
        "updated_at_utc": timestamp,
        "storage_availability_policy": policy_binding,
        "filesystem_identity": canonical_identity,
        "measured_run_filesystem_identity": measured_identity,
        "filesystem_total_bytes": filesystem_total_bytes,
        "filesystem_free_bytes": filesystem_free_bytes,
        "active_other_remaining_bytes": active_other_remaining_bytes,
        "intrinsic_protected_free_floor_bytes": intrinsic_protected_free_floor_bytes,
        "admission_protected_free_floor_bytes": admission_protected_free_floor_bytes,
        "usable_headroom_bytes": usable_headroom_bytes,
        "admissible_headroom_bytes": admissible_headroom_bytes,
        "post_candidate_headroom_bytes": post_candidate_headroom_bytes,
    }
    return {**core, "reservation_sha256": canonical_sha256(core)}


def validate_data2_storage_reservation_v1(
    value: Mapping[str, Any],
    *,
    filename: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactLifecycleError("data2 storage reservation must be an object")
    raw = dict(value)
    supplied_sha256 = raw.pop("reservation_sha256", None)
    _exact_storage_fields(
        raw,
        {
            "schema_version",
            "reservation_id",
            "run_root",
            "lifecycle_contract_sha256",
            "planned_bytes",
            "remaining_bytes",
            "pid",
            "owner_identity",
            "status",
            "updated_at_utc",
            "storage_availability_policy",
            "filesystem_identity",
            "measured_run_filesystem_identity",
            "filesystem_total_bytes",
            "filesystem_free_bytes",
            "active_other_remaining_bytes",
            "intrinsic_protected_free_floor_bytes",
            "admission_protected_free_floor_bytes",
            "usable_headroom_bytes",
            "admissible_headroom_bytes",
            "post_candidate_headroom_bytes",
        },
        path="data2 storage reservation",
    )
    if raw.get("schema_version") != DATA2_STORAGE_RESERVATION_SCHEMA_VERSION:
        raise ArtifactLifecycleError("data2 storage reservation schema is unsupported")
    if raw.get("status") != "active":
        raise ArtifactLifecycleError("data2 storage reservation status is unsupported")
    owner = raw.get("owner_identity")
    expected = build_data2_storage_reservation_v1(
        run_root=Path(str(raw.get("run_root"))),
        lifecycle_contract_sha256=raw.get("lifecycle_contract_sha256"),
        planned_bytes=raw.get("planned_bytes"),
        remaining_bytes=raw.get("remaining_bytes"),
        owner_identity=owner,
        updated_at_utc=raw.get("updated_at_utc"),
        storage_availability_policy=raw.get("storage_availability_policy"),
        filesystem_identity=raw.get("filesystem_identity"),
        measured_run_filesystem_identity=raw.get("measured_run_filesystem_identity"),
        filesystem_total_bytes=raw.get("filesystem_total_bytes"),
        filesystem_free_bytes=raw.get("filesystem_free_bytes"),
        active_other_remaining_bytes=raw.get("active_other_remaining_bytes"),
        intrinsic_protected_free_floor_bytes=raw.get(
            "intrinsic_protected_free_floor_bytes"
        ),
        admission_protected_free_floor_bytes=raw.get(
            "admission_protected_free_floor_bytes"
        ),
        usable_headroom_bytes=raw.get("usable_headroom_bytes"),
        admissible_headroom_bytes=raw.get("admissible_headroom_bytes"),
        post_candidate_headroom_bytes=raw.get("post_candidate_headroom_bytes"),
    )
    if raw.get("pid") != expected["pid"]:
        raise ArtifactLifecycleError("data2 reservation pid/owner binding mismatch")
    if raw.get("reservation_id") != expected["reservation_id"]:
        raise ArtifactLifecycleError("data2 reservation identity binding mismatch")
    if (
        not isinstance(supplied_sha256, str)
        or supplied_sha256 != expected["reservation_sha256"]
    ):
        raise ArtifactLifecycleError("data2 reservation SHA-256 mismatch")
    if filename is not None and filename != f"{expected['reservation_id']}.json":
        raise ArtifactLifecycleError("data2 reservation filename binding mismatch")
    return expected


class RunStorageGuard:
    """Bounded filesystem observer for one run; never deletes data."""

    def __init__(
        self,
        contract: Mapping[str, Any],
        *,
        run_root: Path,
        planned_bytes: int | None = None,
        reservation_window_bytes: int | None = None,
        disk_usage: Callable[[str], Any] = shutil.disk_usage,
        reservation_root: Path | None = None,
        minimum_free_fraction: float | None = None,
        data2_policy_loader: Callable[[], Mapping[str, Any]] | None = None,
        filesystem_identity_provider: Callable[[Path], Mapping[str, Any]] | None = None,
        process_identity_provider: Callable[[int], Mapping[str, Any]] | None = None,
        allow_data2_test_reservation_root: bool = False,
        same_owner_nested_parent_run_root: str | Path | None = None,
    ) -> None:
        self.contract = _validate_contract_payload(contract)
        requested_run_root = Path(
            os.path.abspath(os.path.expanduser(os.fspath(run_root)))
        )
        self.run_root = requested_run_root.resolve(strict=False)
        self.planned_bytes = planned_bytes
        if (
            reservation_window_bytes is not None
            and (
                type(reservation_window_bytes) is not int
                or reservation_window_bytes <= 0
            )
        ):
            raise ArtifactLifecycleError(
                "reservation_window_bytes must be a positive integer or null"
            )
        if (
            reservation_window_bytes is not None
            and planned_bytes is not None
            and reservation_window_bytes > planned_bytes
        ):
            raise ArtifactLifecycleError(
                "reservation_window_bytes cannot exceed the total planned bytes"
            )
        self.reservation_window_bytes = reservation_window_bytes
        self._disk_usage = disk_usage
        self._is_data2 = (
            requested_run_root == DATA2_FILESYSTEM_ROOT
            or DATA2_FILESYSTEM_ROOT in requested_run_root.parents
            or self.run_root == DATA2_FILESYSTEM_ROOT
            or DATA2_FILESYSTEM_ROOT in self.run_root.parents
        )
        if self._is_data2:
            try:
                self.run_root = validate_data2_run_root_v1(requested_run_root)
            except StorageAvailabilityPolicyError as exc:
                raise ArtifactLifecycleError(str(exc)) from exc
            if self.planned_bytes is not None and (
                type(self.planned_bytes) is not int or self.planned_bytes < 0
            ):
                raise ArtifactLifecycleError(
                    "planned_bytes must be a non-negative integer or null"
                )
        self._same_owner_nested_parent_run_root: Path | None = None
        if same_owner_nested_parent_run_root is not None:
            requested_parent_root = Path(
                os.path.abspath(
                    os.path.expanduser(os.fspath(same_owner_nested_parent_run_root))
                )
            )
            if self._is_data2:
                try:
                    parent_root = validate_data2_run_root_v1(requested_parent_root)
                except StorageAvailabilityPolicyError as exc:
                    raise ArtifactLifecycleError(str(exc)) from exc
            else:
                parent_root = requested_parent_root.resolve(strict=False)
            if parent_root == self.run_root or parent_root not in self.run_root.parents:
                raise ArtifactLifecycleError(
                    "same-owner nested reservation parent must strictly contain "
                    "the candidate run root"
                )
            self._same_owner_nested_parent_run_root = parent_root
        self.storage_availability_policy: dict[str, Any] | None = None
        self.storage_availability_policy_binding: dict[str, Any] | None = None
        self.filesystem_identity: dict[str, Any] | None = None
        self.measured_run_filesystem_identity: dict[str, Any] | None = None
        self.owner_identity: dict[str, Any] | None = None
        self._filesystem_identity_provider = (
            observe_filesystem_identity_v1
            if filesystem_identity_provider is None
            else filesystem_identity_provider
        )
        self._process_identity_provider = (
            observe_process_owner_identity_v1
            if process_identity_provider is None
            else process_identity_provider
        )
        if type(allow_data2_test_reservation_root) is not bool:
            raise ArtifactLifecycleError(
                "allow_data2_test_reservation_root must be boolean"
            )
        self._acquired_reservation_sha256: str | None = None
        # A provider-side guard can be nested under the canonical front-door
        # guard for the same bound run.  An exact same-root guard adopts the
        # live reservation.  A deliberately bound child-root guard may keep
        # its own, potentially larger, envelope, but it can never treat an
        # unrelated parent/child reservation as its own or release the parent.
        # The parent and child plans may intentionally have different
        # lifecycle envelopes; the explicit root binding, strict containment,
        # and same-owner check are the authority for this nested relationship.
        self._adopted_existing_reservation = False
        if self._is_data2:
            policy_loader = (
                load_data2_storage_availability_policy_v1
                if data2_policy_loader is None
                else data2_policy_loader
            )
            try:
                self.storage_availability_policy = (
                    validate_data2_storage_availability_policy_v1(policy_loader())
                )
                self.storage_availability_policy_binding = (
                    storage_availability_policy_binding_v1(
                        self.storage_availability_policy
                    )
                )
                self.filesystem_identity = validate_filesystem_identity_v1(
                    self._filesystem_identity_provider(DATA2_FILESYSTEM_ROOT),
                    observed_path=DATA2_FILESYSTEM_ROOT,
                )
                measured_path = self._filesystem_root()
                self.measured_run_filesystem_identity = validate_filesystem_identity_v1(
                    self._filesystem_identity_provider(measured_path),
                    observed_path=measured_path,
                )
                if self.measured_run_filesystem_identity != self.filesystem_identity:
                    raise StorageAvailabilityPolicyError(
                        "measured run filesystem does not match configured runtime root"
                    )
                self.owner_identity = validate_process_owner_identity_v1(
                    self._process_identity_provider(os.getpid())
                )
                if self.owner_identity["pid"] != os.getpid():
                    raise ArtifactLifecycleError(
                        "process owner identity does not bind the current process"
                    )
            except (
                ArtifactLifecycleError,
                StorageAvailabilityPolicyError,
                OSError,
                TypeError,
                ValueError,
            ) as exc:
                raise ArtifactLifecycleError(
                    "canonical configured runtime root storage availability policy or filesystem "
                    f"identity is unavailable or invalid: {exc}"
                ) from exc
        requested_reservation_root = (
            reservation_root.expanduser().resolve()
            if reservation_root is not None
            else self._default_reservation_root()
        )
        if self._is_data2:
            canonical_reservation_root = (
                DATA2_LIFECYCLE_MANAGEMENT_ROOT / "reservations"
            )
            if (
                requested_reservation_root != canonical_reservation_root
                and not allow_data2_test_reservation_root
            ):
                raise ArtifactLifecycleError(
                    "canonical configured runtime root guards must share the lifecycle reservation root"
                )
        self.reservation_root = requested_reservation_root
        if minimum_free_fraction is None:
            # The checked-in data2 policy supplies an evidence-backed reserve;
            # admission no longer invents a global 20% protected floor.
            minimum_free_fraction = 0.0
        if not 0.0 <= float(minimum_free_fraction) < 1.0:
            raise ArtifactLifecycleError(
                "minimum_free_fraction must be at least zero and below one"
            )
        self.minimum_free_fraction = float(minimum_free_fraction)
        if self._is_data2:
            if (
                self.storage_availability_policy_binding is None
                or self.filesystem_identity is None
                or self.measured_run_filesystem_identity is None
            ):
                raise ArtifactLifecycleError(
                    "canonical configured runtime root storage availability authority is not bound"
                )
            self.reservation_id = _data2_reservation_id_v1(
                run_root=self.run_root,
                lifecycle_contract_sha256=self.contract["contract_sha256"],
                storage_availability_policy=self.storage_availability_policy_binding,
                filesystem_identity=self.filesystem_identity,
                measured_run_filesystem_identity=(
                    self.measured_run_filesystem_identity
                ),
            )
        else:
            self.reservation_id = canonical_sha256(
                {
                    "run_root": str(self.run_root),
                    "lifecycle_contract_sha256": self.contract["contract_sha256"],
                }
            )[:24]

    def _default_reservation_root(self) -> Path:
        if self._is_data2:
            return DATA2_LIFECYCLE_MANAGEMENT_ROOT / "reservations"
        for candidate in (self.run_root, *self.run_root.parents):
            if candidate.name == "causyl-runtime":
                return candidate / "lifecycle" / "reservations"
        return self.run_root.parent / ".eiel-storage-reservations"

    @staticmethod
    def _pid_is_live(pid: Any) -> bool:
        if type(pid) is not int or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _reservation_path(self) -> Path:
        return self.reservation_root / f"{self.reservation_id}.json"

    def _reservation_lock(self) -> Any:
        self.reservation_root.mkdir(parents=True, exist_ok=True)
        handle = (self.reservation_root / ".lock").open("a+b")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return handle

    def _authoritative_filesystem_identities(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if (
            not self._is_data2
            or self.filesystem_identity is None
            or self.measured_run_filesystem_identity is None
        ):
            raise ArtifactLifecycleError(
                "canonical configured runtime root filesystem identity is not bound"
            )
        try:
            canonical = validate_filesystem_identity_v1(
                self._filesystem_identity_provider(DATA2_FILESYSTEM_ROOT),
                observed_path=DATA2_FILESYSTEM_ROOT,
            )
            measured_path = self._filesystem_root()
            measured = validate_filesystem_identity_v1(
                self._filesystem_identity_provider(measured_path),
                observed_path=measured_path,
            )
        except (StorageAvailabilityPolicyError, OSError, TypeError, ValueError) as exc:
            raise ArtifactLifecycleError(
                f"cannot re-observe configured runtime root filesystem identity: {exc}"
            ) from exc
        if canonical != measured:
            raise ArtifactLifecycleError(
                "measured run filesystem does not match the configured runtime root filesystem"
            )
        if (
            canonical != self.filesystem_identity
            or measured != self.measured_run_filesystem_identity
        ):
            raise ArtifactLifecycleError(
                "configured runtime root filesystem identity changed after guard construction"
            )
        return canonical, measured

    def _owner_liveness(
        self,
        owner_identity: Mapping[str, Any],
    ) -> tuple[bool | None, str | None]:
        try:
            owner = validate_process_owner_identity_v1(owner_identity)
        except ArtifactLifecycleError as exc:
            return None, str(exc)
        pid = owner["pid"]
        if not self._pid_is_live(pid):
            return False, None
        try:
            observed = validate_process_owner_identity_v1(
                self._process_identity_provider(pid)
            )
        except ProcessLookupError:
            return False, None
        except (ArtifactLifecycleError, OSError, TypeError, ValueError) as exc:
            return None, f"cannot verify process owner identity: {exc}"
        return observed == owner, None

    def _read_data2_reservation(self, path: Path) -> dict[str, Any]:
        if path.is_symlink() or not path.is_file():
            raise ArtifactLifecycleError(
                f"{path.name}: reservation must be a regular non-symlink file"
            )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactLifecycleError(
                f"{path.name}: cannot read reservation: {type(exc).__name__}"
            ) from exc
        try:
            return validate_data2_storage_reservation_v1(
                value,
                filename=path.name,
            )
        except ArtifactLifecycleError as exc:
            raise ArtifactLifecycleError(f"{path.name}: {exc}") from exc

    def _active_reservations(self) -> tuple[list[dict[str, Any]], list[str]]:
        rows: list[dict[str, Any]] = []
        errors: list[str] = []
        if not self.reservation_root.is_dir():
            return rows, errors
        for path in sorted(self.reservation_root.glob("*.json")):
            if self._is_data2:
                try:
                    row = self._read_data2_reservation(path)
                except ArtifactLifecycleError as exc:
                    errors.append(str(exc))
                    continue
                live, liveness_error = self._owner_liveness(row["owner_identity"])
                if liveness_error is not None:
                    errors.append(f"{path.name}: {liveness_error}")
                    continue
                if not live:
                    continue
                if (
                    row["storage_availability_policy"]
                    != self.storage_availability_policy_binding
                ):
                    errors.append(
                        f"{path.name}: storage availability policy binding mismatch"
                    )
                    continue
                if row["filesystem_identity"] != self.filesystem_identity:
                    errors.append(f"{path.name}: filesystem identity binding mismatch")
                    continue
                if (
                    row["measured_run_filesystem_identity"]
                    != self.measured_run_filesystem_identity
                ):
                    errors.append(
                        f"{path.name}: measured filesystem identity binding mismatch"
                    )
                    continue
                rows.append(row)
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                errors.append(f"{path.name}: {type(exc).__name__}")
                continue
            if not isinstance(value, Mapping):
                errors.append(f"{path.name}: non-object reservation")
                continue
            row = dict(value)
            if row.get("status") != "active":
                continue
            if not self._pid_is_live(row.get("pid")):
                continue
            remaining = row.get("remaining_bytes")
            if type(remaining) is not int or remaining < 0:
                errors.append(f"{path.name}: invalid remaining_bytes")
                continue
            rows.append(row)
        return rows, errors

    def _storage_availability_snapshot(
        self,
        *,
        free_bytes: int | None,
        filesystem_total_bytes: int | None,
        filesystem_identity: Mapping[str, Any] | None = None,
        execution_reserve_bytes: int | None = None,
    ) -> dict[str, Any] | None:
        if not self._is_data2:
            return None
        if self.storage_availability_policy is None or self.filesystem_identity is None:
            raise ArtifactLifecycleError(
                "canonical configured runtime root storage availability authority is not bound"
            )
        return build_storage_availability_snapshot_v1(
            policy=self.storage_availability_policy,
            filesystem_identity=(
                self.filesystem_identity
                if filesystem_identity is None
                else filesystem_identity
            ),
            filesystem_total_bytes=filesystem_total_bytes,
            filesystem_free_bytes=free_bytes,
            lifecycle_reserve_bytes=int(
                self.contract["storage"]["emergency_reserve_bytes"]
            ),
            configured_minimum_free_fraction=self.minimum_free_fraction,
            execution_reserve_bytes=execution_reserve_bytes,
        )

    def _execution_reserve_bytes(self, *, total_run_bytes: int) -> int | None:
        if not self._is_data2 or self.storage_availability_policy is None:
            return None
        remaining = (
            int(self.reservation_window_bytes)
            if self.reservation_window_bytes is not None
            else max(0, int(self.planned_bytes or 0) - int(total_run_bytes))
        )
        reserve = evidence_backed_execution_reserve_v1(
            policy=self.storage_availability_policy,
            remaining_run_bytes=remaining,
            reservation_window_bytes=int(self.reservation_window_bytes or 0),
            lifecycle_reserve_bytes=int(
                self.contract["storage"]["emergency_reserve_bytes"]
            ),
        )
        return int(reserve["execution_reserve_bytes"])

    def _reservation_snapshot(
        self,
        *,
        free_bytes: int | None,
        filesystem_total_bytes: int | None,
        total_run_bytes: int,
        include_candidate: bool,
    ) -> dict[str, Any]:
        rows, errors = self._active_reservations()
        authoritative_identity = self.filesystem_identity
        measured_identity = self.measured_run_filesystem_identity
        filesystem_identity_verified = not self._is_data2
        if self._is_data2:
            try:
                authoritative_identity, measured_identity = (
                    self._authoritative_filesystem_identities()
                )
                filesystem_identity_verified = True
            except ArtifactLifecycleError as exc:
                errors.append(str(exc))
                filesystem_identity_verified = False
        if self._is_data2 and self.owner_identity is not None:
            for row in rows:
                if (
                    row["run_root"] == str(self.run_root)
                    and row["lifecycle_contract_sha256"]
                    == self.contract["contract_sha256"]
                    and row["owner_identity"] != self.owner_identity
                ):
                    errors.append(
                        "duplicate live owner for the same run and lifecycle contract"
                    )
        other_rows = [
            row for row in rows if row.get("reservation_id") != self.reservation_id
        ]
        for row in other_rows:
            row_root = row.get("run_root")
            if isinstance(row_root, str) and _run_roots_overlap_v1(
                self.run_root, row_root
            ):
                same_owner_nested_parent = (
                    self._same_owner_nested_parent_run_root is not None
                    and row_root == str(self._same_owner_nested_parent_run_root)
                    and self._same_owner_nested_parent_run_root in self.run_root.parents
                    and (
                        (
                            self._is_data2
                            and row.get("owner_identity") == self.owner_identity
                        )
                        or (
                            not self._is_data2
                            and row.get("pid") == os.getpid()
                        )
                    )
                )
                if not same_owner_nested_parent:
                    errors.append(
                        "live reservation root overlaps candidate run root: "
                        + row_root
                    )
        current_remaining = (
            int(self.reservation_window_bytes)
            if self.reservation_window_bytes is not None
            else (
                max(0, int(self.planned_bytes) - total_run_bytes)
                if self.planned_bytes is not None
                else 0
            )
        )
        candidate_bounded = (
            self.reservation_window_bytes is not None
            or self.planned_bytes is not None
            or self.minimum_free_fraction == 0.0
        )
        other_remaining = sum(int(row["remaining_bytes"]) for row in other_rows)
        candidate_remaining = current_remaining if include_candidate else 0
        reserved_remaining = other_remaining + candidate_remaining
        availability = self._storage_availability_snapshot(
            free_bytes=(free_bytes if filesystem_identity_verified else None),
            filesystem_total_bytes=(
                filesystem_total_bytes if filesystem_identity_verified else None
            ),
            filesystem_identity=authoritative_identity,
            execution_reserve_bytes=self._execution_reserve_bytes(
                total_run_bytes=total_run_bytes
            ),
        )
        if availability is not None:
            current_protected_floor = availability["protected_free_floor_bytes"]
        else:
            fraction_floor = (
                int(filesystem_total_bytes * self.minimum_free_fraction)
                if filesystem_total_bytes is not None
                else 0
            )
            current_protected_floor = max(
                int(self.contract["storage"]["emergency_reserve_bytes"]),
                fraction_floor,
            )
        active_other_floors: list[int] = []
        for row in other_rows:
            stored_floor = row.get("intrinsic_protected_free_floor_bytes")
            stored_total = row.get("filesystem_total_bytes")
            if type(stored_floor) is not int or stored_floor < 0:
                continue
            effective_floor = stored_floor
            if (
                type(filesystem_total_bytes) is int
                and filesystem_total_bytes > 0
                and type(stored_total) is int
                and stored_total > 0
                and filesystem_total_bytes != stored_total
            ):
                # The V1 reservation binds the effective protected floor but
                # not its individual configured-fraction and lifecycle-reserve
                # components.  A capacity change would make exact component
                # recomputation ambiguous, so fail closed until the active V1
                # reservation finishes instead of weakening or over-scaling
                # its configured policy.
                errors.append(
                    "active reservation filesystem capacity changed while its "
                    "protected floor is bound"
                )
            active_other_floors.append(effective_floor)
        active_other_protected_floor = max(active_other_floors, default=0)
        protected_floor = (
            max(current_protected_floor, active_other_protected_floor)
            if current_protected_floor is not None
            else None
        )
        usable_headroom = (
            max(0, free_bytes - protected_floor)
            if free_bytes is not None and protected_floor is not None
            else None
        )
        admissible_headroom = (
            max(0, usable_headroom - other_remaining)
            if usable_headroom is not None
            else None
        )
        post_candidate_headroom = (
            max(0, admissible_headroom - candidate_remaining)
            if admissible_headroom is not None
            else None
        )
        required_free = (
            protected_floor + reserved_remaining
            if protected_floor is not None
            else None
        )
        result = {
            "reservation_root": str(self.reservation_root),
            "reservation_id": self.reservation_id,
            "active_other_reservations": len(other_rows),
            "active_other_remaining_bytes": other_remaining,
            "candidate_remaining_bytes": candidate_remaining,
            "reservation_window_bytes": self.reservation_window_bytes,
            "candidate_bounded": candidate_bounded,
            "minimum_free_fraction": self.minimum_free_fraction,
            "intrinsic_protected_floor_bytes": current_protected_floor,
            "active_other_protected_floor_bytes": active_other_protected_floor,
            "protected_floor_bytes": protected_floor,
            "usable_headroom_bytes": usable_headroom,
            "admissible_headroom_bytes": admissible_headroom,
            "post_candidate_headroom_bytes": post_candidate_headroom,
            "required_free_bytes": required_free,
            "free_bytes": free_bytes,
            "errors": errors,
            "filesystem_identity_verified": filesystem_identity_verified,
            "ok": (
                free_bytes is not None
                and required_free is not None
                and not errors
                and (not include_candidate or candidate_bounded)
                and free_bytes >= required_free
            ),
        }
        if availability is not None:
            result["storage_availability"] = availability
            result["filesystem_identity"] = authoritative_identity
            result["measured_run_filesystem_identity"] = measured_identity
        return result

    def reserve(self) -> dict[str, Any]:
        lock = self._reservation_lock()
        try:
            snapshot = self.snapshot(include_reservations=False)
            reservation = self._reservation_snapshot(
                free_bytes=snapshot["free_bytes"],
                filesystem_total_bytes=snapshot["filesystem_total_bytes"],
                total_run_bytes=int(snapshot["total_run_bytes"]),
                include_candidate=True,
            )
            if not snapshot["ok"] or not reservation["ok"]:
                return {**snapshot, "reservation": reservation, "ok": False}
            remaining = int(reservation["candidate_remaining_bytes"])
            if self._is_data2:
                path = self._reservation_path()
                adopted_live_reservation = False
                if path.exists() or path.is_symlink():
                    try:
                        existing = self._read_data2_reservation(path)
                    except ArtifactLifecycleError as exc:
                        reservation["errors"].append(str(exc))
                        reservation["ok"] = False
                        return {**snapshot, "reservation": reservation, "ok": False}
                    live, liveness_error = self._owner_liveness(
                        existing["owner_identity"]
                    )
                    if liveness_error is not None:
                        reservation["errors"].append(liveness_error)
                        reservation["ok"] = False
                        return {**snapshot, "reservation": reservation, "ok": False}
                    if live:
                        if self._acquired_reservation_sha256 is None:
                            # The canonical execution front door admits the
                            # run once and the provider may create a second
                            # guard for the same bound run while that outer
                            # reservation is still live.  Adopt only that
                            # exact same-owner reservation; a live reservation
                            # for another owner or a different planned
                            # envelope remains fail-closed.
                            if existing["owner_identity"] != self.owner_identity:
                                reservation["errors"].append(
                                    "live reservation ownership differs from this guard"
                                )
                                reservation["ok"] = False
                                return {
                                    **snapshot,
                                    "reservation": reservation,
                                    "ok": False,
                                }
                            if existing["planned_bytes"] != self.planned_bytes:
                                reservation["errors"].append(
                                    "live reservation planned-bytes binding differs from this guard"
                                )
                                reservation["ok"] = False
                                return {
                                    **snapshot,
                                    "reservation": reservation,
                                    "ok": False,
                                }
                            self._acquired_reservation_sha256 = existing[
                                "reservation_sha256"
                            ]
                            self._adopted_existing_reservation = True
                            adopted_live_reservation = True
                        elif self._adopted_existing_reservation:
                            if (
                                existing["owner_identity"] != self.owner_identity
                                or existing["planned_bytes"] != self.planned_bytes
                            ):
                                reservation["errors"].append(
                                    "adopted live reservation binding changed"
                                )
                                reservation["ok"] = False
                                return {
                                    **snapshot,
                                    "reservation": reservation,
                                    "ok": False,
                                }
                            # A same-owner sibling guard may have refreshed the
                            # receipt.  Follow its digest without treating that
                            # expected nested update as foreign mutation.
                            self._acquired_reservation_sha256 = existing[
                                "reservation_sha256"
                            ]
                            adopted_live_reservation = True
                        if (
                            existing["owner_identity"] != self.owner_identity
                            or existing["reservation_sha256"]
                            != self._acquired_reservation_sha256
                        ):
                            reservation["errors"].append(
                                "live reservation ownership changed after acquisition"
                            )
                            reservation["ok"] = False
                            return {
                                **snapshot,
                                "reservation": reservation,
                                "ok": False,
                            }
                if adopted_live_reservation:
                    # The outer guard owns publication and release.  Returning
                    # the current admission snapshot is sufficient for the
                    # nested provider guard and avoids changing the outer
                    # receipt digest on every dispatch boundary.
                    return {**snapshot, "reservation": reservation, "ok": True}
                if (
                    self.owner_identity is None
                    or self.storage_availability_policy_binding is None
                    or reservation.get("filesystem_identity") is None
                    or reservation.get("measured_run_filesystem_identity") is None
                    or type(snapshot.get("filesystem_total_bytes")) is not int
                    or type(snapshot.get("free_bytes")) is not int
                    or self.planned_bytes is None
                ):
                    reservation["errors"].append(
                        "authoritative configured runtime root reservation inputs are incomplete"
                    )
                    reservation["ok"] = False
                    return {**snapshot, "reservation": reservation, "ok": False}
                try:
                    final_identity, final_measured_identity = (
                        self._authoritative_filesystem_identities()
                    )
                except ArtifactLifecycleError as exc:
                    reservation["errors"].append(str(exc))
                    reservation["ok"] = False
                    return {**snapshot, "reservation": reservation, "ok": False}
                if (
                    final_identity != reservation["filesystem_identity"]
                    or final_measured_identity
                    != reservation["measured_run_filesystem_identity"]
                ):
                    reservation["errors"].append(
                        "configured runtime root filesystem identity changed before reservation write"
                    )
                    reservation["ok"] = False
                    return {**snapshot, "reservation": reservation, "ok": False}
                receipt = build_data2_storage_reservation_v1(
                    run_root=self.run_root,
                    lifecycle_contract_sha256=self.contract["contract_sha256"],
                    planned_bytes=self.planned_bytes,
                    remaining_bytes=remaining,
                    owner_identity=self.owner_identity,
                    updated_at_utc=_now(),
                    storage_availability_policy=(
                        self.storage_availability_policy_binding
                    ),
                    filesystem_identity=final_identity,
                    measured_run_filesystem_identity=final_measured_identity,
                    filesystem_total_bytes=snapshot["filesystem_total_bytes"],
                    filesystem_free_bytes=snapshot["free_bytes"],
                    active_other_remaining_bytes=reservation[
                        "active_other_remaining_bytes"
                    ],
                    intrinsic_protected_free_floor_bytes=reservation[
                        "intrinsic_protected_floor_bytes"
                    ],
                    admission_protected_free_floor_bytes=reservation[
                        "protected_floor_bytes"
                    ],
                    usable_headroom_bytes=reservation["usable_headroom_bytes"],
                    admissible_headroom_bytes=reservation["admissible_headroom_bytes"],
                    post_candidate_headroom_bytes=reservation[
                        "post_candidate_headroom_bytes"
                    ],
                )
                _atomic_json(path, receipt)
                self._acquired_reservation_sha256 = receipt["reservation_sha256"]
                return {**snapshot, "reservation": reservation, "ok": True}
            core = {
                "schema_version": STORAGE_GUARD_SCHEMA_VERSION,
                "reservation_id": self.reservation_id,
                "run_root": str(self.run_root),
                "lifecycle_contract_sha256": self.contract["contract_sha256"],
                "planned_bytes": self.planned_bytes,
                "reservation_window_bytes": self.reservation_window_bytes,
                "remaining_bytes": remaining,
                "pid": os.getpid(),
                "status": "active",
                "updated_at_utc": _now(),
            }
            _atomic_json(
                self._reservation_path(),
                {**core, "reservation_sha256": canonical_sha256(core)},
            )
            return {**snapshot, "reservation": reservation, "ok": True}
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    def release(self) -> None:
        lock = self._reservation_lock()
        try:
            path = self._reservation_path()
            if self._is_data2:
                if self._adopted_existing_reservation:
                    self._acquired_reservation_sha256 = None
                    self._adopted_existing_reservation = False
                    return
                if not path.exists() and not path.is_symlink():
                    return
                existing = self._read_data2_reservation(path)
                self._authoritative_filesystem_identities()
                if (
                    self.owner_identity is None
                    or existing["owner_identity"] != self.owner_identity
                ):
                    raise ArtifactLifecycleError(
                        "refusing to release a reservation owned by another process"
                    )
                if (
                    existing["storage_availability_policy"]
                    != self.storage_availability_policy_binding
                    or existing["filesystem_identity"] != self.filesystem_identity
                    or existing["measured_run_filesystem_identity"]
                    != self.measured_run_filesystem_identity
                ):
                    raise ArtifactLifecycleError(
                        "refusing to release a reservation with stale authority bindings"
                    )
                if (
                    self._acquired_reservation_sha256 is not None
                    and existing["reservation_sha256"]
                    != self._acquired_reservation_sha256
                ):
                    raise ArtifactLifecycleError(
                        "refusing to release a reservation changed after acquisition"
                    )
                path.unlink()
                self._acquired_reservation_sha256 = None
                return
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    def validate_active_release_owner_v1(
        self,
        *,
        run_root: Path,
        lifecycle_contract_sha256: str,
        reservation_lock_held: bool = False,
    ) -> None:
        """Authenticate the current guard for an in-run hot release.

        The normal prune actuator refuses every reserved root.  This narrow
        capability is only available to the guard that acquired the exact
        live reservation for the same root and lifecycle contract.
        """

        expected_root = Path(run_root).expanduser().resolve()
        if self.run_root != expected_root:
            raise ArtifactLifecycleError(
                "active release owner is bound to a different runtime root"
            )
        if lifecycle_contract_sha256 != self.contract["contract_sha256"]:
            raise ArtifactLifecycleError(
                "active release owner is bound to a different lifecycle contract"
            )
        if type(reservation_lock_held) is not bool:
            raise ArtifactLifecycleError(
                "active release reservation lock flag must be boolean"
            )
        path = self._reservation_path()
        lock = None if reservation_lock_held else self._reservation_lock()
        try:
            if path.is_symlink() or not path.is_file():
                raise ArtifactLifecycleError(
                    "active release owner reservation is not live"
                )
            if self._is_data2:
                existing = self._read_data2_reservation(path)
                self._authoritative_filesystem_identities()
                if (
                    self.owner_identity is None
                    or existing["owner_identity"] != self.owner_identity
                    or existing["run_root"] != str(self.run_root)
                    or existing["lifecycle_contract_sha256"]
                    != self.contract["contract_sha256"]
                    or (
                        self._acquired_reservation_sha256 is not None
                        and existing["reservation_sha256"]
                        != self._acquired_reservation_sha256
                    )
                ):
                    raise ArtifactLifecycleError(
                        "active release owner reservation identity changed"
                    )
                live, error = self._owner_liveness(existing["owner_identity"])
                if error is not None or live is not True:
                    raise ArtifactLifecycleError(
                        "active release owner process is not authenticated"
                    )
                return
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ArtifactLifecycleError(
                    "active release owner reservation is unreadable"
                ) from exc
            if not isinstance(value, Mapping):
                raise ArtifactLifecycleError(
                    "active release owner reservation is malformed"
                )
            if (
                value.get("status") != "active"
                or value.get("run_root") != str(self.run_root)
                or value.get("lifecycle_contract_sha256")
                != self.contract["contract_sha256"]
                or value.get("pid") != os.getpid()
            ):
                raise ArtifactLifecycleError(
                    "active release owner reservation identity changed"
                )
        finally:
            if lock is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                lock.close()

    @classmethod
    def release_for_run(
        cls,
        contract: Mapping[str, Any],
        *,
        run_root: Path,
        reservation_root: Path | None = None,
        data2_policy_loader: Callable[[], Mapping[str, Any]] | None = None,
        filesystem_identity_provider: Callable[[Path], Mapping[str, Any]] | None = None,
        process_identity_provider: Callable[[int], Mapping[str, Any]] | None = None,
        allow_data2_test_reservation_root: bool = False,
    ) -> None:
        cls(
            contract,
            run_root=run_root,
            reservation_root=reservation_root,
            data2_policy_loader=data2_policy_loader,
            filesystem_identity_provider=filesystem_identity_provider,
            process_identity_provider=process_identity_provider,
            allow_data2_test_reservation_root=allow_data2_test_reservation_root,
        ).release()

    @classmethod
    def has_active_reservation(
        cls,
        contract: Mapping[str, Any],
        *,
        run_root: Path,
        reservation_root: Path | None = None,
        data2_policy_loader: Callable[[], Mapping[str, Any]] | None = None,
        filesystem_identity_provider: Callable[[Path], Mapping[str, Any]] | None = None,
        process_identity_provider: Callable[[int], Mapping[str, Any]] | None = None,
        allow_data2_test_reservation_root: bool = False,
    ) -> bool:
        guard = cls(
            contract,
            run_root=run_root,
            reservation_root=reservation_root,
            data2_policy_loader=data2_policy_loader,
            filesystem_identity_provider=filesystem_identity_provider,
            process_identity_provider=process_identity_provider,
            allow_data2_test_reservation_root=allow_data2_test_reservation_root,
        )
        rows, errors = guard._active_reservations()
        if errors:
            return True
        return any(row.get("reservation_id") == guard.reservation_id for row in rows)

    @classmethod
    def run_root_is_reserved(
        cls,
        run_root: Path,
        *,
        reservation_root: Path | None = None,
        data2_policy_loader: Callable[[], Mapping[str, Any]] | None = None,
        filesystem_identity_provider: Callable[[Path], Mapping[str, Any]] | None = None,
        process_identity_provider: Callable[[int], Mapping[str, Any]] | None = None,
        allow_data2_test_reservation_root: bool = False,
    ) -> bool:
        requested_root = Path(os.path.abspath(os.path.expanduser(os.fspath(run_root))))
        root = requested_root.resolve(strict=False)
        is_data2 = (
            requested_root == DATA2_FILESYSTEM_ROOT
            or DATA2_FILESYSTEM_ROOT in requested_root.parents
            or root == DATA2_FILESYSTEM_ROOT
            or DATA2_FILESYSTEM_ROOT in root.parents
        )
        if is_data2:
            try:
                root = validate_data2_run_root_v1(requested_root)
            except StorageAvailabilityPolicyError:
                return True
        if type(allow_data2_test_reservation_root) is not bool:
            return True
        canonical_reservation_root = DATA2_LIFECYCLE_MANAGEMENT_ROOT / "reservations"
        if is_data2 and reservation_root is None:
            reservation_root = canonical_reservation_root
        elif reservation_root is None:
            for candidate in (root, *root.parents):
                if candidate.name == "causyl-runtime":
                    reservation_root = candidate / "lifecycle" / "reservations"
                    break
            else:
                reservation_root = root.parent / ".eiel-storage-reservations"
        elif is_data2:
            requested_reservation_root = reservation_root.expanduser().resolve()
            if (
                requested_reservation_root != canonical_reservation_root
                and not allow_data2_test_reservation_root
            ):
                return True
        directory = reservation_root.expanduser().resolve()
        if not directory.is_dir():
            return False
        canonical_policy_binding: dict[str, Any] | None = None
        canonical_filesystem_identity: dict[str, Any] | None = None
        if is_data2:
            policy_loader = (
                load_data2_storage_availability_policy_v1
                if data2_policy_loader is None
                else data2_policy_loader
            )
            identity_provider = (
                observe_filesystem_identity_v1
                if filesystem_identity_provider is None
                else filesystem_identity_provider
            )
            owner_provider = (
                observe_process_owner_identity_v1
                if process_identity_provider is None
                else process_identity_provider
            )
            try:
                canonical_policy_binding = storage_availability_policy_binding_v1(
                    policy_loader()
                )
                canonical_filesystem_identity = validate_filesystem_identity_v1(
                    identity_provider(DATA2_FILESYSTEM_ROOT),
                    observed_path=DATA2_FILESYSTEM_ROOT,
                )
            except (StorageAvailabilityPolicyError, OSError, TypeError, ValueError):
                return True
        for path in directory.glob("*.json"):
            if is_data2:
                if path.is_symlink() or not path.is_file():
                    return True
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    row = validate_data2_storage_reservation_v1(
                        value,
                        filename=path.name,
                    )
                    measured_identity = validate_filesystem_identity_v1(
                        identity_provider(Path(row["run_root"])),
                        observed_path=Path(row["run_root"]),
                    )
                except (
                    ArtifactLifecycleError,
                    StorageAvailabilityPolicyError,
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                ):
                    return True
                if (
                    row["storage_availability_policy"] != canonical_policy_binding
                    or row["filesystem_identity"] != canonical_filesystem_identity
                    or row["measured_run_filesystem_identity"] != measured_identity
                    or measured_identity != canonical_filesystem_identity
                ):
                    return True
                owner = row["owner_identity"]
                if not cls._pid_is_live(owner["pid"]):
                    continue
                try:
                    observed_owner = validate_process_owner_identity_v1(
                        owner_provider(owner["pid"])
                    )
                except ProcessLookupError:
                    continue
                except (ArtifactLifecycleError, OSError, TypeError, ValueError):
                    return True
                if observed_owner != owner:
                    continue
                if _run_roots_overlap_v1(row["run_root"], root):
                    return True
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                return True
            if not isinstance(value, Mapping):
                return True
            if (
                value.get("status") == "active"
                and isinstance(value.get("run_root"), str)
                and _run_roots_overlap_v1(value["run_root"], root)
                and cls._pid_is_live(value.get("pid"))
            ):
                return True
        return False

    def _filesystem_root(self) -> Path:
        candidate = self.run_root
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        return candidate

    def snapshot(self, *, include_reservations: bool = True) -> dict[str, Any]:
        by_zone = {"scratch": 0, "checkpoints": 0, "durable": 0}
        unknown = 0
        trace_violations: list[str] = []
        if self.run_root.exists():
            for path in self.run_root.rglob("*"):
                if not path.is_file() or path.is_symlink():
                    continue
                relative = _relative(path, self.run_root)
                size = int(path.stat().st_size)
                zone = _zone(relative)
                if zone == "unknown":
                    unknown += size
                else:
                    by_zone[_path_zone(relative)] += size
                lower = relative.lower()
                if (
                    _is_direct_sparse_semantic_event_envelope_v1(relative)
                    or _is_compact_science_durable_payload_v1(relative)
                    or _is_compact_science_async_staging_payload_v1(relative)
                    or _is_workhorse_exact_event_sink_payload_v2(relative)
                    or _is_workhorse_semantic_fragment_payload_v2(relative)
                ):
                    trace_kind = None
                elif "event" in lower or "debug" in lower:
                    trace_kind = "event"
                elif any(token in lower for token in ("raw_tick", "trace")):
                    trace_kind = "raw"
                else:
                    trace_kind = None
                if trace_kind is not None:
                    policy = self.contract["observability"][
                        "debug_event_trace_policy"
                        if trace_kind == "event"
                        else "raw_tick_trace_policy"
                    ]
                    if policy == "forbidden" and _zone(relative) not in {
                        "control",
                        "checkpoints",
                        "durable",
                        "closeout",
                    }:
                        trace_violations.append(relative)
        total = sum(by_zone.values()) + unknown
        filesystem_root = self._filesystem_root()
        authoritative_identity = self.filesystem_identity
        measured_identity = self.measured_run_filesystem_identity
        filesystem_identity_error: str | None = None
        try:
            if self._is_data2:
                authoritative_identity, measured_identity = (
                    self._authoritative_filesystem_identities()
                )
            usage = self._disk_usage(str(filesystem_root))
            free_bytes = int(usage.free)
            total_filesystem_bytes = int(usage.total)
            if self._is_data2:
                observed_after, measured_after = (
                    self._authoritative_filesystem_identities()
                )
                if (
                    observed_after != authoritative_identity
                    or measured_after != measured_identity
                ):
                    raise ArtifactLifecycleError(
                        "configured runtime root filesystem identity changed during snapshot"
                    )
        except (
            ArtifactLifecycleError,
            OSError,
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            free_bytes = None
            total_filesystem_bytes = None
            if self._is_data2:
                filesystem_identity_error = str(exc)
        storage = self.contract["storage"]
        violations: list[dict[str, Any]] = []
        availability = self._storage_availability_snapshot(
            free_bytes=free_bytes,
            filesystem_total_bytes=total_filesystem_bytes,
            filesystem_identity=authoritative_identity,
            execution_reserve_bytes=self._execution_reserve_bytes(
                total_run_bytes=total
            ),
        )
        for zone, limit_name in (
            ("scratch", "scratch_max_bytes"),
            ("checkpoints", "checkpoint_max_bytes"),
            ("durable", "durable_max_bytes"),
        ):
            if by_zone[zone] > storage[limit_name]:
                violations.append(
                    {
                        "code": f"{zone}_quota_exceeded",
                        "bytes": by_zone[zone],
                        "limit_bytes": storage[limit_name],
                    }
                )
        if total > storage["total_max_bytes"]:
            violations.append(
                {
                    "code": "total_quota_exceeded",
                    "bytes": total,
                    "limit_bytes": storage["total_max_bytes"],
                }
            )
        if free_bytes is not None and free_bytes < storage["emergency_reserve_bytes"]:
            violations.append(
                {
                    "code": "emergency_reserve_breached",
                    "free_bytes": free_bytes,
                    "reserve_bytes": storage["emergency_reserve_bytes"],
                }
            )
        if availability is not None and availability["status"] != "OBSERVED":
            violations.append(
                {
                    "code": "storage_availability_unobserved",
                    "policy_sha256": availability["policy_sha256"],
                }
            )
        if filesystem_identity_error is not None:
            violations.append(
                {
                    "code": "filesystem_identity_unverified",
                    "message": filesystem_identity_error,
                }
            )
        elif (
            availability is not None
            and int(availability["protected_floor_deficit_bytes"]) > 0
        ):
            violations.append(
                {
                    "code": "protected_free_floor_breached",
                    "free_bytes": free_bytes,
                    "protected_free_floor_bytes": availability[
                        "protected_free_floor_bytes"
                    ],
                    "deficit_bytes": availability["protected_floor_deficit_bytes"],
                }
            )
        if trace_violations:
            violations.append(
                {
                    "code": "undeclared_raw_trace",
                    "paths": sorted(trace_violations)[:32],
                    "count": len(trace_violations),
                }
            )
        result = {
            "schema_version": STORAGE_GUARD_SCHEMA_VERSION,
            "run_root": str(self.run_root),
            "lifecycle_profile": self.contract["lifecycle_profile"],
            "lifecycle_contract_sha256": self.contract["contract_sha256"],
            "bytes_by_zone": by_zone,
            "unknown_bytes": unknown,
            "total_run_bytes": total,
            "free_bytes": free_bytes,
            "filesystem_total_bytes": total_filesystem_bytes,
            "emergency_reserve_bytes": storage["emergency_reserve_bytes"],
            "planned_bytes": self.planned_bytes,
            "violations": violations,
            "ok": not violations,
        }
        if availability is not None:
            result.update(
                {
                    "storage_availability_policy": (
                        self.storage_availability_policy_binding
                    ),
                    "filesystem_identity": self.filesystem_identity,
                    "authoritative_filesystem_identity": authoritative_identity,
                    "measured_run_filesystem_identity": measured_identity,
                    "filesystem_identity_verified": (filesystem_identity_error is None),
                    "protected_free_floor_bytes": availability[
                        "protected_free_floor_bytes"
                    ],
                    "usable_headroom_bytes": availability["usable_headroom_bytes"],
                    "storage_availability": availability,
                }
            )
        if include_reservations:
            reservation = self._reservation_snapshot(
                free_bytes=free_bytes,
                filesystem_total_bytes=total_filesystem_bytes,
                total_run_bytes=total,
                include_candidate=True,
            )
            result["reservation"] = reservation
            if availability is not None:
                result["admissible_headroom_bytes"] = reservation[
                    "admissible_headroom_bytes"
                ]
                result["post_candidate_headroom_bytes"] = reservation[
                    "post_candidate_headroom_bytes"
                ]
            if not reservation["ok"]:
                result["violations"].append(
                    {
                        "code": "aggregate_storage_reservation_blocked",
                        "required_bytes": reservation["required_free_bytes"],
                        "free_bytes": reservation["free_bytes"],
                        "reservation_errors": list(reservation["errors"]),
                        "candidate_bounded": reservation["candidate_bounded"],
                    }
                )
                result["ok"] = False
        return result

    def preflight(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        snapshot["phase"] = "preflight"
        return snapshot

    def enforce(
        self,
        *,
        stage: str,
        pre_admission_reclaimer: Callable[["RunStorageGuard", Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Admit once, reclaim exact lifecycle material once, then retry.

        The callback is deliberately injected by the canonical execution
        front door.  It may use only manifest-authorized lifecycle actuators;
        this guard never discovers or deletes unrelated paths itself.  One
        bounded retry prevents storage pressure from becoming an unbounded
        reclaim loop.
        """

        snapshot = self.reserve()
        reclamation: Mapping[str, Any] | None = None
        if not snapshot["ok"] and pre_admission_reclaimer is not None:
            try:
                reclamation = pre_admission_reclaimer(self, snapshot)
            except Exception as exc:  # safety boundary: retry remains fail-closed
                reclamation = {
                    "schema_version": "eiel_pre_admission_reclamation_receipt_v1",
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                    "bytes_reclaimed": 0,
                }
            snapshot = self.reserve()
            if reclamation is not None:
                snapshot = {**snapshot, "pre_admission_reclamation": dict(reclamation)}
        if snapshot["ok"]:
            snapshot["stage"] = stage
            return snapshot
        receipt_core = {
            "schema_version": STORAGE_FAILURE_SCHEMA_VERSION,
            "run_root": str(self.run_root),
            "lifecycle_profile": self.contract["lifecycle_profile"],
            "lifecycle_contract_sha256": self.contract["contract_sha256"],
            "stage": stage,
            "status": "STORAGE_BUDGET_BLOCKED",
            "snapshot": snapshot,
        }
        receipt = {**receipt_core, "receipt_sha256": canonical_sha256(receipt_core)}
        if self.run_root:
            _atomic_json(self.run_root / "storage_budget_failure_receipt.json", receipt)
        raise StorageBudgetExceeded(
            "run storage lifecycle guard blocked continued execution",
            receipt=receipt,
        )


__all__ = [
    "ArtifactLifecycleError",
    "CHECKPOINT_TRANSIT",
    "CHECKPOINT_AUTHORITIES",
    "DEBUG_FORENSIC",
    "DECISION_EVIDENCE_CLASSES",
    "DATA2_STORAGE_RESERVATION_SCHEMA_VERSION",
    "DISPOSITION_SCHEMA_VERSION",
    "LIFECYCLE_PROFILES",
    "LIFECYCLE_REQUIRED_PRUNE_PREDICATES",
    "LIFECYCLE_SCHEMA_VERSION",
    "PROGRESSIVE_STORAGE_RESERVATION_MODE_V1",
    "QUALIFICATION",
    "RETENTION_CLASSES",
    "RunArtifactLifecycleContractV1",
    "RunStorageGuard",
    "RUN_FAILURE_SCHEMA_VERSION",
    "SCIENCE_COMPACT",
    "STORAGE_FAILURE_SCHEMA_VERSION",
    "STORAGE_RECEIPT_SCHEMA_VERSION",
    "StorageBudgetExceeded",
    "build_active_artifact_disposition_manifest_v1",
    "build_artifact_disposition_manifest_v1",
    "build_data2_storage_reservation_v1",
    "build_run_artifact_failure_receipt_v1",
    "build_run_passport_v1",
    "build_sealed_artifact_inventory_row_v1",
    "build_storage_disposition_receipt_v1",
    "checkpoint_role_for_generation",
    "checkpoint_role_for_tick",
    "disposition_manifest_authorizes_prune_v1",
    "ensure_lifecycle_contract_v1",
    "lifecycle_profile_contract_v1",
    "observe_process_owner_identity_v1",
    "progressive_storage_reservation_window_bytes_v1",
    "rebind_lifecycle_checkpoint_schedule_v1",
    "resolve_run_artifact_lifecycle_v1",
    "run_artifact_transaction_fence_v1",
    "validate_artifact_disposition_manifest_v1",
    "validate_data2_storage_reservation_v1",
    "validate_process_owner_identity_v1",
    "validate_run_artifact_failure_receipt_v1",
    "validate_run_artifact_lifecycle_contract_v1",
    "write_run_passport_v1",
    "write_run_artifact_failure_receipt_v1",
    "write_storage_disposition_receipt_v1",
    "register_pending_durable_archive_v1",
]
