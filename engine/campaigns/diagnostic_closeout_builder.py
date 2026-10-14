"""Bounded preparation of complete DEBUG_FORENSIC closeout requests.

The builder reads one explicitly named completed disposition manifest and one
explicit compact review spec.  It never discovers runs, changes source bytes,
or grants release authority.  Its output is a complete evidence-certificate
request plus a compact, hash-bound same-identity review receipt and report.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Final

from .artifact_lifecycle import (
    DEBUG_FORENSIC,
    validate_artifact_disposition_manifest_v1,
    validate_run_artifact_lifecycle_contract_v1,
)
from .canonical import canonical_sha256, pretty_json
from .evidence_certificate import (
    DIAGNOSTIC_SOURCE_LIVE_MARKER_CLOSEOUT_SCHEMA_VERSION,
    EvidenceCertificateError,
    validate_diagnostic_source_live_marker_closeout_v1,
    validate_diagnostic_source_live_marker_freshness_v1,
    validate_evidence_certificate_request_v1,
)

DEBUG_CLOSEOUT_REVIEW_SPEC_SCHEMA_VERSION: Final[str] = (
    "eiel_debug_forensic_closeout_review_spec_v1"
)
DEBUG_CLOSEOUT_REVIEW_INPUT_SCHEMA_VERSION: Final[str] = (
    "eiel_debug_forensic_closeout_review_input_v1"
)
DEBUG_CLOSEOUT_REVIEW_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_debug_forensic_closeout_review_receipt_v1"
)
DEBUG_CLOSEOUT_PREPARATION_PLAN_SCHEMA_VERSION: Final[str] = (
    "eiel_debug_forensic_closeout_preparation_plan_v1"
)
MAX_DEBUG_CLOSEOUT_SOURCE_ROWS: Final[int] = 50_000
MAX_DEBUG_CLOSEOUT_MANIFEST_BYTES: Final[int] = 16 * 1024 * 1024
MAX_DEBUG_CLOSEOUT_REFERENCE_COUNT: Final[int] = 32
MAX_DEBUG_CLOSEOUT_REFERENCE_BYTES: Final[int] = 16 * 1024 * 1024
MAX_DEBUG_CLOSEOUT_REQUEST_BYTES: Final[int] = 8 * 1024 * 1024
MAX_DEBUG_CLOSEOUT_REPORT_BYTES: Final[int] = 256 * 1024
MAX_DEBUG_CLOSEOUT_RECEIPT_BYTES: Final[int] = 256 * 1024

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*\Z")
_DECISIONS = {
    "ACCEPTED",
    "REJECTED",
    "VALID_NEGATIVE",
    "NO_SUPPORTED_RESULT",
    "RAW_DIAGNOSTIC_ABANDONED",
}
_SPECIAL_NO_RESULT_DECISIONS = {
    "NO_SUPPORTED_RESULT",
    "RAW_DIAGNOSTIC_ABANDONED",
}
_REVIEW_STATUSES = {"accepted", "accepted_with_limitations"}
_REVIEW_INDEPENDENCE = {
    "INDEPENDENT_REVIEW",
    "REVIEWER_INDEPENDENCE_UNAVAILABLE",
}
_CRITERION_STATUSES = {"passed", "failed", "valid_negative"}
_SUPERSEDED_EVIDENCE_CLASSES = {
    "OPERATIONAL_STATE",
    "RAW_TRANSIENT_EVIDENCE",
}
_RETENTION_CLASSES = {"R0", "R1", "R2", "R3", "R4", "R5", "RX"}


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceCertificateError(f"{path} must be an object")
    return dict(value)


def _exact_fields(value: Mapping[str, Any], expected: set[str], *, path: str) -> None:
    if set(value) != expected:
        raise EvidenceCertificateError(
            f"{path} fields differ: expected {sorted(expected)}, got {sorted(value)}"
        )


def _text(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceCertificateError(f"{path} must be a non-empty string")
    return value.strip()


def _identifier(value: Any, *, path: str) -> str:
    result = _text(value, path=path)
    if _ID_RE.fullmatch(result) is None:
        raise EvidenceCertificateError(f"{path} must be a stable identifier")
    return result


def _sha256(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise EvidenceCertificateError(f"{path} must be a lowercase SHA-256")
    return value


def _strings(value: Any, *, path: str, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise EvidenceCertificateError(f"{path} must be an array of strings")
    result = [item.strip() for item in value]
    if len(result) != len(set(result)):
        raise EvidenceCertificateError(f"{path} must not contain duplicates")
    if nonempty and not result:
        raise EvidenceCertificateError(f"{path} must not be empty")
    return result


def _objects(value: Any, *, path: str, nonempty: bool = False) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(
        isinstance(row, Mapping) for row in value
    ):
        raise EvidenceCertificateError(f"{path} must be an array of objects")
    result = [dict(row) for row in value]
    if nonempty and not result:
        raise EvidenceCertificateError(f"{path} must not be empty")
    return result


def _timestamp(value: Any, *, path: str) -> str:
    text = _text(value, path=path)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceCertificateError(f"{path} must be ISO-8601 UTC") from exc
    if not text.endswith("Z") or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise EvidenceCertificateError(f"{path} must be normalized UTC")
    return text


def _safe_relative(value: Any, *, path: str) -> str:
    text = _text(value, path=path).replace("\\", "/").strip("/")
    candidate = Path(text)
    if candidate.is_absolute() or not text or ".." in candidate.parts:
        raise EvidenceCertificateError(f"{path} must be a safe relative path")
    return text


def _validate_live_state(value: Any) -> dict[str, bool]:
    raw = _mapping(value, path="review_spec.live_state")
    names = {"writer", "reference", "resume", "reducer", "consumer"}
    _exact_fields(raw, names, path="review_spec.live_state")
    result: dict[str, bool] = {}
    for name in sorted(names):
        if type(raw[name]) is not bool:
            raise EvidenceCertificateError(
                f"review_spec.live_state.{name} must be boolean"
            )
        result[name] = raw[name]
    active = [name for name, enabled in result.items() if enabled]
    if active:
        raise EvidenceCertificateError(
            "DEBUG closeout review retains live state: " + ",".join(active)
        )
    return result


def _validate_exact_holds(value: Any) -> dict[str, list[str]]:
    raw = _mapping(value, path="review_spec.active_exact_byte_holds")
    names = {"forensic", "audit", "qualification", "replay"}
    _exact_fields(raw, names, path="review_spec.active_exact_byte_holds")
    result = {
        name: _strings(raw[name], path=f"review_spec.active_exact_byte_holds.{name}")
        for name in sorted(names)
    }
    active = [f"{name}:{hold}" for name, values in result.items() for hold in values]
    if active:
        raise EvidenceCertificateError(
            "DEBUG closeout review retains active exact-byte holds: " + ",".join(active)
        )
    return result


def _validate_source_live_marker_closeout(
    value: Any,
    *,
    selected_artifact_ids: Sequence[str],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return validate_diagnostic_source_live_marker_closeout_v1(
        value,
        selected_artifact_ids=selected_artifact_ids,
        source_manifest=source_manifest,
    )


def validate_debug_closeout_review_spec_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one explicit same-identity substantive review specification."""

    raw = _mapping(value, path="review_spec")
    supplied_sha256 = raw.pop("spec_sha256", None)
    expected = {
        "schema_version",
        "programme_id",
        "work_id",
        "obligation_id",
        "bounded_claim",
        "reviewed_at_utc",
        "reviewer",
        "source",
        "source_manifest",
        "artifact_lifecycle_contract",
        "decision",
        "conclusion",
        "limitations",
        "criteria",
        "measurements",
        "control_receipts",
        "superseded_artifacts",
        "unresolved_anomalies",
        "live_state",
        "active_exact_byte_holds",
        "source_live_marker_closeout",
        "reproduction",
        "environment",
        "device",
        "no_claims",
    }
    _exact_fields(raw, expected, path="review_spec")
    if raw.get("schema_version") != DEBUG_CLOSEOUT_REVIEW_SPEC_SCHEMA_VERSION:
        raise EvidenceCertificateError("DEBUG closeout review spec schema mismatch")

    reviewer = _mapping(raw.get("reviewer"), path="review_spec.reviewer")
    _exact_fields(
        reviewer,
        {"identity", "independence", "status"},
        path="review_spec.reviewer",
    )
    reviewer_status = _text(reviewer.get("status"), path="review_spec.reviewer.status")
    if reviewer_status not in _REVIEW_STATUSES:
        raise EvidenceCertificateError("DEBUG closeout review is not accepted")
    independence = _text(
        reviewer.get("independence"), path="review_spec.reviewer.independence"
    )
    if independence not in _REVIEW_INDEPENDENCE:
        raise EvidenceCertificateError(
            "DEBUG closeout review independence is unsupported"
        )

    source = _mapping(raw.get("source"), path="review_spec.source")
    _exact_fields(source, {"repository", "commit_sha"}, path="review_spec.source")
    commit_sha = _text(source.get("commit_sha"), path="review_spec.source.commit_sha")
    if re.fullmatch(r"[0-9a-f]{40,64}", commit_sha) is None:
        raise EvidenceCertificateError(
            "review_spec.source.commit_sha must be a lowercase Git/content identity"
        )
    source_manifest = _mapping(
        raw.get("source_manifest"), path="review_spec.source_manifest"
    )
    _exact_fields(
        source_manifest,
        {
            "path",
            "manifest_sha256",
            "campaign_id",
            "run_id",
            "runtime_root",
            "plan_sha256",
            "lifecycle_contract_sha256",
        },
        path="review_spec.source_manifest",
    )
    source_runtime_root = _text(
        source_manifest.get("runtime_root"),
        path="review_spec.source_manifest.runtime_root",
    )
    if (
        not Path(source_runtime_root).is_absolute()
        or str(Path(source_runtime_root).expanduser().resolve(strict=False))
        != source_runtime_root
    ):
        raise EvidenceCertificateError(
            "review_spec.source_manifest.runtime_root must be normalized and absolute"
        )
    normalized_source_manifest = {
        "path": _safe_relative(
            source_manifest.get("path"), path="review_spec.source_manifest.path"
        ),
        "manifest_sha256": _sha256(
            source_manifest.get("manifest_sha256"),
            path="review_spec.source_manifest.manifest_sha256",
        ),
        "campaign_id": _text(
            source_manifest.get("campaign_id"),
            path="review_spec.source_manifest.campaign_id",
        ),
        "run_id": _text(
            source_manifest.get("run_id"),
            path="review_spec.source_manifest.run_id",
        ),
        "runtime_root": source_runtime_root,
        "plan_sha256": _sha256(
            source_manifest.get("plan_sha256"),
            path="review_spec.source_manifest.plan_sha256",
        ),
        "lifecycle_contract_sha256": _sha256(
            source_manifest.get("lifecycle_contract_sha256"),
            path="review_spec.source_manifest.lifecycle_contract_sha256",
        ),
    }

    contract = validate_run_artifact_lifecycle_contract_v1(
        _mapping(
            raw.get("artifact_lifecycle_contract"),
            path="review_spec.artifact_lifecycle_contract",
        )
    )
    if (
        contract.get("lifecycle_profile") != DEBUG_FORENSIC
        or contract.get("prune", {}).get("automatic_prune_allowed") is not False
    ):
        raise EvidenceCertificateError(
            "DEBUG closeout spec must preserve its non-prunable DEBUG_FORENSIC contract"
        )

    decision = _text(raw.get("decision"), path="review_spec.decision")
    if decision not in _DECISIONS:
        raise EvidenceCertificateError("DEBUG closeout review decision is unsupported")
    limitations = _strings(
        raw.get("limitations"), path="review_spec.limitations", nonempty=True
    )
    if decision in _SPECIAL_NO_RESULT_DECISIONS and reviewer_status != (
        "accepted_with_limitations"
    ):
        raise EvidenceCertificateError(
            "no-result or abandoned diagnostics require accepted_with_limitations"
        )

    criteria: list[dict[str, str]] = []
    for index, row in enumerate(
        _objects(raw.get("criteria"), path="review_spec.criteria", nonempty=True)
    ):
        _exact_fields(
            row,
            {"criterion", "status", "detail"},
            path=f"review_spec.criteria[{index}]",
        )
        status = _text(row.get("status"), path=f"review_spec.criteria[{index}].status")
        if status not in _CRITERION_STATUSES:
            raise EvidenceCertificateError("DEBUG closeout criterion is incomplete")
        criteria.append(
            {
                "criterion": _text(
                    row.get("criterion"),
                    path=f"review_spec.criteria[{index}].criterion",
                ),
                "status": status,
                "detail": _text(
                    row.get("detail"), path=f"review_spec.criteria[{index}].detail"
                ),
            }
        )
    if decision in _SPECIAL_NO_RESULT_DECISIONS and not any(
        row["status"] in {"failed", "valid_negative"} for row in criteria
    ):
        raise EvidenceCertificateError(
            "no-result or abandoned diagnostics require a failed/valid-negative criterion"
        )

    control_receipts: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    seen_control_artifacts: set[str] = set()
    for index, row in enumerate(
        _objects(
            raw.get("control_receipts"),
            path="review_spec.control_receipts",
            nonempty=True,
        )
    ):
        _exact_fields(
            row,
            {"artifact_id", "path", "role", "bytes", "sha256"},
            path=f"review_spec.control_receipts[{index}]",
        )
        role = _identifier(
            row.get("role"), path=f"review_spec.control_receipts[{index}].role"
        )
        if role in seen_roles:
            raise EvidenceCertificateError("control receipt roles must be unique")
        seen_roles.add(role)
        artifact_id = _identifier(
            row.get("artifact_id"),
            path=f"review_spec.control_receipts[{index}].artifact_id",
        )
        if artifact_id in seen_control_artifacts:
            raise EvidenceCertificateError(
                "control receipt artifact IDs must be unique"
            )
        seen_control_artifacts.add(artifact_id)
        control_receipts.append(
            {
                "artifact_id": artifact_id,
                "path": _safe_relative(
                    row.get("path"),
                    path=f"review_spec.control_receipts[{index}].path",
                ),
                "role": role,
                "bytes": row.get("bytes"),
                "sha256": _sha256(
                    row.get("sha256"),
                    path=f"review_spec.control_receipts[{index}].sha256",
                ),
            }
        )
        if type(control_receipts[-1]["bytes"]) is not int or not (
            0 <= control_receipts[-1]["bytes"] <= MAX_DEBUG_CLOSEOUT_REFERENCE_BYTES
        ):
            raise EvidenceCertificateError(
                f"review_spec.control_receipts[{index}].bytes is invalid"
            )
    if len(control_receipts) > MAX_DEBUG_CLOSEOUT_REFERENCE_COUNT:
        raise EvidenceCertificateError("too many DEBUG closeout control receipts")
    if not any(
        "result" in row["role"].lower() or "decision" in row["role"].lower()
        for row in control_receipts
    ):
        raise EvidenceCertificateError(
            "DEBUG closeout requires a manifest-bound result or decision artifact; "
            "terminal status alone is not substantive evidence"
        )

    superseded: list[dict[str, str]] = []
    seen_artifacts: set[str] = set()
    for index, row in enumerate(
        _objects(
            raw.get("superseded_artifacts"),
            path="review_spec.superseded_artifacts",
            nonempty=True,
        )
    ):
        _exact_fields(
            row,
            {"artifact_id", "evidence_class", "rationale"},
            path=f"review_spec.superseded_artifacts[{index}]",
        )
        artifact_id = _identifier(
            row.get("artifact_id"),
            path=f"review_spec.superseded_artifacts[{index}].artifact_id",
        )
        if artifact_id in seen_artifacts:
            raise EvidenceCertificateError("superseded artifact IDs must be unique")
        seen_artifacts.add(artifact_id)
        evidence_class = _text(
            row.get("evidence_class"),
            path=f"review_spec.superseded_artifacts[{index}].evidence_class",
        )
        if evidence_class not in _SUPERSEDED_EVIDENCE_CLASSES:
            raise EvidenceCertificateError(
                "superseded DEBUG rows must be explicitly raw/transient or operational"
            )
        superseded.append(
            {
                "artifact_id": artifact_id,
                "evidence_class": evidence_class,
                "rationale": _text(
                    row.get("rationale"),
                    path=f"review_spec.superseded_artifacts[{index}].rationale",
                ),
            }
        )

    reproduction = _mapping(raw.get("reproduction"), path="review_spec.reproduction")
    _exact_fields(
        reproduction,
        {
            "prerequisites",
            "commands",
            "verification_rules",
            "expected_results",
            "guarantee",
        },
        path="review_spec.reproduction",
    )
    guarantee = _text(
        reproduction.get("guarantee"), path="review_spec.reproduction.guarantee"
    )
    if guarantee not in {"BIT_IDENTITY", "SEMANTIC_EQUIVALENCE", "REVALIDATION_ONLY"}:
        raise EvidenceCertificateError("review reproduction guarantee is unsupported")

    no_claims = _strings(
        raw.get("no_claims"), path="review_spec.no_claims", nonempty=True
    )
    lowered_claims = " ".join(no_claims).lower()
    if "scientific" not in lowered_claims or "qualification" not in lowered_claims:
        raise EvidenceCertificateError(
            "DEBUG closeout review must disclaim scientific and qualification claims"
        )
    anomalies = _strings(
        raw.get("unresolved_anomalies"),
        path="review_spec.unresolved_anomalies",
    )
    if anomalies:
        raise EvidenceCertificateError(
            "DEBUG closeout review cannot retain unresolved anomalies"
        )
    environment = _mapping(raw.get("environment"), path="review_spec.environment")
    device = _mapping(raw.get("device"), path="review_spec.device")
    measurements = _mapping(raw.get("measurements"), path="review_spec.measurements")
    if not environment or not device:
        raise EvidenceCertificateError(
            "DEBUG closeout review requires environment and device descriptions"
        )

    normalized = {
        "schema_version": DEBUG_CLOSEOUT_REVIEW_SPEC_SCHEMA_VERSION,
        "programme_id": _identifier(
            raw.get("programme_id"), path="review_spec.programme_id"
        ),
        "work_id": _identifier(raw.get("work_id"), path="review_spec.work_id"),
        "obligation_id": _identifier(
            raw.get("obligation_id"), path="review_spec.obligation_id"
        ),
        "bounded_claim": _text(
            raw.get("bounded_claim"), path="review_spec.bounded_claim"
        ),
        "reviewed_at_utc": _timestamp(
            raw.get("reviewed_at_utc"), path="review_spec.reviewed_at_utc"
        ),
        "reviewer": {
            "identity": _identifier(
                reviewer.get("identity"), path="review_spec.reviewer.identity"
            ),
            "independence": independence,
            "status": reviewer_status,
        },
        "source": {
            "repository": _text(
                source.get("repository"), path="review_spec.source.repository"
            ),
            "commit_sha": commit_sha,
        },
        "source_manifest": normalized_source_manifest,
        "artifact_lifecycle_contract": contract,
        "decision": decision,
        "conclusion": _text(raw.get("conclusion"), path="review_spec.conclusion"),
        "limitations": limitations,
        "criteria": criteria,
        "measurements": measurements,
        "control_receipts": control_receipts,
        "superseded_artifacts": superseded,
        "unresolved_anomalies": anomalies,
        "live_state": _validate_live_state(raw.get("live_state")),
        "active_exact_byte_holds": _validate_exact_holds(
            raw.get("active_exact_byte_holds")
        ),
        "source_live_marker_closeout": _validate_source_live_marker_closeout(
            raw.get("source_live_marker_closeout"),
            selected_artifact_ids=[row["artifact_id"] for row in superseded],
            source_manifest=normalized_source_manifest,
        ),
        "reproduction": {
            "prerequisites": _strings(
                reproduction.get("prerequisites"),
                path="review_spec.reproduction.prerequisites",
                nonempty=True,
            ),
            "commands": _strings(
                reproduction.get("commands"),
                path="review_spec.reproduction.commands",
                nonempty=True,
            ),
            "verification_rules": _strings(
                reproduction.get("verification_rules"),
                path="review_spec.reproduction.verification_rules",
                nonempty=True,
            ),
            "expected_results": _objects(
                reproduction.get("expected_results"),
                path="review_spec.reproduction.expected_results",
                nonempty=True,
            ),
            "guarantee": guarantee,
        },
        "environment": environment,
        "device": device,
        "no_claims": no_claims,
    }
    expected_sha256 = canonical_sha256(normalized)
    if supplied_sha256 != expected_sha256:
        raise EvidenceCertificateError("DEBUG closeout review spec SHA-256 mismatch")
    return {**normalized, "spec_sha256": supplied_sha256}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_source_file_v1(*, root: Path, relative: str, label: str) -> Path:
    candidate = root
    for part in Path(relative).parts:
        candidate = candidate / part
        try:
            observed = os.lstat(candidate)
        except FileNotFoundError as exc:
            raise EvidenceCertificateError(f"{label} is absent: {relative}") from exc
        if stat.S_ISLNK(observed.st_mode):
            raise EvidenceCertificateError(f"{label} cannot traverse a symlink")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EvidenceCertificateError(f"{label} escapes evidence_root") from exc
    return resolved


def _validate_control_receipts(
    rows: list[dict[str, Any]],
    *,
    evidence_root: Path,
    source_rows_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    root = evidence_root.expanduser().resolve(strict=True)
    for row in rows:
        artifact_id = str(row["artifact_id"])
        source_row = source_rows_by_id.get(artifact_id)
        if source_row is None:
            raise EvidenceCertificateError(
                f"supporting artifact is absent from source manifest: {artifact_id}"
            )
        if source_row.get("path") != row["path"]:
            raise EvidenceCertificateError(
                f"supporting artifact path differs from source manifest: {artifact_id}"
            )
        if "sha256" not in source_row:
            raise EvidenceCertificateError(
                f"supporting artifact must be one exact regular file: {artifact_id}"
            )
        if (
            source_row.get("bytes") != row["bytes"]
            or source_row.get("sha256") != row["sha256"]
            or source_row.get("exact_object_identity") is not True
        ):
            raise EvidenceCertificateError(
                f"supporting artifact identity differs from source manifest: {artifact_id}"
            )
        path = _resolve_source_file_v1(
            root=root,
            relative=str(row["path"]),
            label=f"supporting artifact {artifact_id}",
        )
        observed = os.lstat(path)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_size > MAX_DEBUG_CLOSEOUT_REFERENCE_BYTES
            or observed.st_size != row["bytes"]
        ):
            raise EvidenceCertificateError(
                "control receipt must be a bounded single-link regular file"
            )
        if _sha256_file(path) != row["sha256"]:
            raise EvidenceCertificateError(
                f"control receipt SHA-256 mismatch: {row['role']}"
            )
        validated.append(
            {
                "artifact_id": artifact_id,
                "role": row["role"],
                "path": str(row["path"]),
                "sha256": row["sha256"],
                "bytes": int(observed.st_size),
            }
        )
    return validated


def _load_source_manifest(
    *, evidence_root: Path, source_manifest_path: Path
) -> tuple[dict[str, Any], str]:
    root = evidence_root.expanduser()
    if root.is_symlink():
        raise EvidenceCertificateError("evidence_root cannot be a symlink")
    root = root.resolve(strict=True)
    unresolved = source_manifest_path.expanduser()
    if not unresolved.is_absolute():
        unresolved = root / unresolved
    if unresolved.is_symlink():
        raise EvidenceCertificateError(
            "source disposition manifest cannot be a symlink"
        )
    path = unresolved.resolve(strict=True)
    try:
        relative = str(path.relative_to(root)).replace("\\", "/")
    except ValueError as exc:
        raise EvidenceCertificateError(
            "source disposition manifest must be inside evidence_root"
        ) from exc
    observed = os.lstat(path)
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or observed.st_size > MAX_DEBUG_CLOSEOUT_MANIFEST_BYTES
    ):
        raise EvidenceCertificateError(
            "source disposition manifest must be a bounded single-link regular file"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceCertificateError(
            "source disposition manifest cannot be decoded"
        ) from exc
    manifest = validate_artifact_disposition_manifest_v1(raw)
    runtime_root = Path(str(manifest.get("runtime_root") or "")).expanduser()
    if runtime_root.is_symlink() or runtime_root.resolve(strict=True) != root:
        raise EvidenceCertificateError(
            "DEBUG closeout requires one exact source runtime root"
        )
    if (
        manifest.get("terminal_status") != "completed"
        or manifest.get("lifecycle_profile") != DEBUG_FORENSIC
        or manifest.get("inventory") != {"mode": "producer_declared", "complete": True}
    ):
        raise EvidenceCertificateError(
            "DEBUG closeout requires a completed complete DEBUG_FORENSIC manifest"
        )
    artifacts = manifest.get("artifacts", [])
    if (
        not isinstance(artifacts, list)
        or not artifacts
        or len(artifacts) > MAX_DEBUG_CLOSEOUT_SOURCE_ROWS
    ):
        raise EvidenceCertificateError("DEBUG source manifest row count is unsupported")
    return manifest, relative


def _row_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    digest_name = "sha256" if "sha256" in row else "tree_sha256"
    return {
        "artifact_id": row["artifact_id"],
        "path": row["path"],
        "bytes": row["bytes"],
        "retention_class": row["retention_class"],
        digest_name: row[digest_name],
    }


def _retained_evidence_class(source_class: str) -> str:
    if source_class == "R5":
        return "DURABLE_EVIDENCE_CERTIFICATE"
    if source_class == "R4":
        return "SCIENTIFIC_ENGINEERING_ASSET"
    return "RAW_TRANSIENT_EVIDENCE"


def draft_debug_closeout_review_spec_v1(
    *,
    evidence_root: Path,
    source_manifest_path: Path,
    artifact_lifecycle_contract: Mapping[str, Any],
    review_input: Mapping[str, Any],
    supporting_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Draft one fully bound spec from explicit review facts and one manifest.

    Retention-class selectors are deliberate compact instructions over the one
    named manifest.  The resulting spec expands them to exact artifact IDs.
    Nothing is discovered outside that manifest and its explicitly named
    supporting result/control files.
    """

    manifest, manifest_relative = _load_source_manifest(
        evidence_root=evidence_root,
        source_manifest_path=source_manifest_path,
    )
    contract = validate_run_artifact_lifecycle_contract_v1(
        _mapping(artifact_lifecycle_contract, path="review_input.lifecycle_contract")
    )
    if contract["contract_sha256"] != manifest["lifecycle_contract_sha256"]:
        raise EvidenceCertificateError(
            "draft lifecycle contract differs from the exact source manifest"
        )
    source_manifest_identity = {
        "path": manifest_relative,
        "manifest_sha256": manifest["manifest_sha256"],
        "campaign_id": manifest["campaign_id"],
        "run_id": manifest["run_id"],
        "runtime_root": manifest["runtime_root"],
        "plan_sha256": manifest["plan_sha256"],
        "lifecycle_contract_sha256": manifest["lifecycle_contract_sha256"],
    }
    raw = _mapping(review_input, path="review_input")
    expected_input_fields = {
        "schema_version",
        "programme_id",
        "work_id",
        "obligation_id",
        "bounded_claim",
        "reviewed_at_utc",
        "reviewer",
        "decision",
        "conclusion",
        "limitations",
        "criteria",
        "measurements",
        "supersession",
        "source_live_marker_audit",
        "reproduction",
        "environment",
        "device",
        "no_claims",
        "confirmations",
    }
    _exact_fields(raw, expected_input_fields, path="review_input")
    if raw.get("schema_version") != DEBUG_CLOSEOUT_REVIEW_INPUT_SCHEMA_VERSION:
        raise EvidenceCertificateError("DEBUG closeout review input schema mismatch")

    confirmations = _mapping(
        raw.get("confirmations"), path="review_input.confirmations"
    )
    confirmation_names = {
        "complete_inventory_reviewed",
        "no_unresolved_anomalies",
        "no_live_state",
        "no_active_exact_byte_holds",
    }
    _exact_fields(confirmations, confirmation_names, path="review_input.confirmations")
    if any(confirmations.get(name) is not True for name in confirmation_names):
        raise EvidenceCertificateError(
            "DEBUG closeout draft requires every explicit closeout confirmation"
        )

    source_rows = [dict(row) for row in manifest["artifacts"]]
    by_id = {str(row.get("artifact_id")): row for row in source_rows}
    by_path = {str(row.get("path")): row for row in source_rows}
    if len(by_id) != len(source_rows) or len(by_path) != len(source_rows):
        raise EvidenceCertificateError("source manifest row identities are ambiguous")

    controls: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    seen_paths: set[str] = set()
    for index, supplied in enumerate(supporting_artifacts):
        item = _mapping(supplied, path=f"supporting_artifacts[{index}]")
        _exact_fields(item, {"role", "path"}, path=f"supporting_artifacts[{index}]")
        role = _identifier(item.get("role"), path=f"supporting_artifacts[{index}].role")
        relative = _safe_relative(
            item.get("path"), path=f"supporting_artifacts[{index}].path"
        )
        if role in seen_roles or relative in seen_paths:
            raise EvidenceCertificateError(
                "supporting artifact roles and paths must be unique"
            )
        seen_roles.add(role)
        seen_paths.add(relative)
        source_row = by_path.get(relative)
        if source_row is None:
            raise EvidenceCertificateError(
                f"supporting artifact is absent from source manifest: {relative}"
            )
        if "sha256" not in source_row:
            raise EvidenceCertificateError(
                f"supporting artifact must be an exact regular file: {relative}"
            )
        controls.append(
            {
                "artifact_id": str(source_row["artifact_id"]),
                "path": relative,
                "role": role,
                "bytes": int(source_row["bytes"]),
                "sha256": str(source_row["sha256"]),
            }
        )
    if not controls:
        raise EvidenceCertificateError(
            "DEBUG closeout draft requires a named manifest-bound result/control artifact"
        )
    _validate_control_receipts(
        controls,
        evidence_root=evidence_root,
        source_rows_by_id=by_id,
    )
    execution_requests = [row for row in controls if row["role"] == "execution-request"]
    if len(execution_requests) != 1:
        raise EvidenceCertificateError(
            "DEBUG closeout draft requires exactly one execution-request support"
        )

    def control_git_identity(
        row: Mapping[str, Any], *, required: bool
    ) -> dict[str, str] | None:
        path = _resolve_source_file_v1(
            root=evidence_root.expanduser().resolve(strict=True),
            relative=str(row["path"]),
            label=f"supporting artifact {row['artifact_id']}",
        )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            if required:
                raise EvidenceCertificateError(
                    "manifest-bound execution request is not valid JSON"
                ) from exc
            return None
        if not isinstance(payload, Mapping):
            if required:
                raise EvidenceCertificateError(
                    "manifest-bound execution request is not a JSON object"
                )
            return None
        source_identity = payload.get("source_identity")
        git = (
            source_identity.get("git") if isinstance(source_identity, Mapping) else None
        )
        if not isinstance(git, Mapping):
            if required:
                raise EvidenceCertificateError(
                    "manifest-bound execution request lacks source_identity.git"
                )
            return None
        repository = git.get("repository_root")
        commit = git.get("commit")
        if (
            not isinstance(repository, str)
            or not Path(repository).is_absolute()
            or not isinstance(commit, str)
            or re.fullmatch(r"[0-9a-f]{40,64}", commit) is None
        ):
            raise EvidenceCertificateError(
                f"supporting artifact {row['role']} has invalid Git source identity"
            )
        return {"repository": repository, "commit_sha": commit}

    source = control_git_identity(execution_requests[0], required=True)
    assert source is not None
    for row in controls:
        if row["role"] == "execution-request":
            continue
        observed_source = control_git_identity(row, required=False)
        if observed_source is not None and observed_source != source:
            raise EvidenceCertificateError(
                f"supporting artifact {row['role']} Git identity differs from execution request"
            )

    supersession = _mapping(raw.get("supersession"), path="review_input.supersession")
    _exact_fields(
        supersession,
        {
            "retention_classes",
            "artifact_ids",
            "preserve_artifact_ids",
            "evidence_class",
            "rationale",
        },
        path="review_input.supersession",
    )
    selected_classes = set(
        _strings(
            supersession.get("retention_classes"),
            path="review_input.supersession.retention_classes",
        )
    )
    if not selected_classes <= {"R0", "R1", "R2", "R3"}:
        raise EvidenceCertificateError(
            "DEBUG draft supersession selectors are limited to R0-R3"
        )
    explicit_ids = set(
        _strings(
            supersession.get("artifact_ids"),
            path="review_input.supersession.artifact_ids",
        )
    )
    preserve_ids = set(
        _strings(
            supersession.get("preserve_artifact_ids"),
            path="review_input.supersession.preserve_artifact_ids",
        )
    )
    missing_ids = sorted((explicit_ids | preserve_ids) - set(by_id))
    if missing_ids:
        raise EvidenceCertificateError(
            "draft artifact selectors are absent from source manifest: "
            + ",".join(missing_ids)
        )
    supporting_ids = {str(row["artifact_id"]) for row in controls}
    if explicit_ids & (preserve_ids | supporting_ids):
        raise EvidenceCertificateError(
            "an explicitly superseded artifact is also explicitly preserved"
        )
    selected_ids = {
        artifact_id
        for artifact_id, row in by_id.items()
        if str(row.get("retention_class")) in selected_classes
    }
    selected_ids |= explicit_ids
    selected_ids -= preserve_ids | supporting_ids
    if not selected_ids:
        raise EvidenceCertificateError("DEBUG closeout draft selects no source rows")
    invalid_selected = sorted(
        artifact_id
        for artifact_id in selected_ids
        if str(by_id[artifact_id].get("retention_class"))
        not in {"R0", "R1", "R2", "R3"}
    )
    if invalid_selected:
        raise EvidenceCertificateError(
            "draft attempts to supersede protected R4/R5/RX rows: "
            + ",".join(invalid_selected)
        )
    evidence_class = _text(
        supersession.get("evidence_class"),
        path="review_input.supersession.evidence_class",
    )
    if evidence_class not in _SUPERSEDED_EVIDENCE_CLASSES:
        raise EvidenceCertificateError(
            "draft supersession evidence class must be raw/transient or operational"
        )
    rationale = _text(
        supersession.get("rationale"), path="review_input.supersession.rationale"
    )
    superseded = [
        {
            "artifact_id": artifact_id,
            "evidence_class": evidence_class,
            "rationale": rationale,
        }
        for artifact_id in sorted(selected_ids)
    ]
    live_marker_audit = _mapping(
        raw.get("source_live_marker_audit"),
        path="review_input.source_live_marker_audit",
    )
    _exact_fields(
        live_marker_audit,
        {"status", "audited_at_utc", "checks", "scope", "audit_method"},
        path="review_input.source_live_marker_audit",
    )
    source_live_marker_closeout = {
        "schema_version": DIAGNOSTIC_SOURCE_LIVE_MARKER_CLOSEOUT_SCHEMA_VERSION,
        **live_marker_audit,
        "source_manifest_sha256": manifest["manifest_sha256"],
        "source_campaign_id": manifest["campaign_id"],
        "source_run_id": manifest["run_id"],
        "source_runtime_root": manifest["runtime_root"],
        "selected_artifact_count": len(selected_ids),
        "selected_artifact_ids_sha256": canonical_sha256(sorted(selected_ids)),
    }

    spec_core = {
        "schema_version": DEBUG_CLOSEOUT_REVIEW_SPEC_SCHEMA_VERSION,
        "programme_id": raw.get("programme_id"),
        "work_id": raw.get("work_id"),
        "obligation_id": raw.get("obligation_id"),
        "bounded_claim": raw.get("bounded_claim"),
        "reviewed_at_utc": raw.get("reviewed_at_utc"),
        "reviewer": raw.get("reviewer"),
        "source": source,
        "source_manifest": source_manifest_identity,
        "artifact_lifecycle_contract": contract,
        "decision": raw.get("decision"),
        "conclusion": raw.get("conclusion"),
        "limitations": raw.get("limitations"),
        "criteria": raw.get("criteria"),
        "measurements": raw.get("measurements"),
        "control_receipts": controls,
        "superseded_artifacts": superseded,
        "unresolved_anomalies": [],
        "live_state": {
            "writer": False,
            "reference": False,
            "resume": False,
            "reducer": False,
            "consumer": False,
        },
        "active_exact_byte_holds": {
            "forensic": [],
            "audit": [],
            "qualification": [],
            "replay": [],
        },
        "source_live_marker_closeout": source_live_marker_closeout,
        "reproduction": raw.get("reproduction"),
        "environment": raw.get("environment"),
        "device": raw.get("device"),
        "no_claims": raw.get("no_claims"),
    }
    spec = {**spec_core, "spec_sha256": canonical_sha256(spec_core)}
    validated_spec = validate_debug_closeout_review_spec_v1(spec)
    validate_diagnostic_source_live_marker_freshness_v1(
        validated_spec["source_live_marker_closeout"]
    )
    return validated_spec


def build_debug_closeout_bundle_v1(
    *,
    evidence_root: Path,
    source_manifest_path: Path,
    review_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a complete request/review/report bundle without writing anything."""

    spec = validate_debug_closeout_review_spec_v1(review_spec)
    manifest, manifest_relative = _load_source_manifest(
        evidence_root=evidence_root,
        source_manifest_path=source_manifest_path,
    )
    source_identity = {
        "path": manifest_relative,
        "manifest_sha256": manifest["manifest_sha256"],
        "campaign_id": manifest["campaign_id"],
        "run_id": manifest["run_id"],
        "runtime_root": manifest["runtime_root"],
        "plan_sha256": manifest["plan_sha256"],
        "lifecycle_contract_sha256": manifest["lifecycle_contract_sha256"],
    }
    if spec["source_manifest"] != source_identity:
        raise EvidenceCertificateError(
            "review spec is bound to a different source manifest/run/root identity"
        )
    validate_diagnostic_source_live_marker_freshness_v1(
        spec["source_live_marker_closeout"]
    )
    if spec["artifact_lifecycle_contract"].get("contract_sha256") != manifest.get(
        "lifecycle_contract_sha256"
    ):
        raise EvidenceCertificateError(
            "review lifecycle contract differs from the exact source manifest"
        )
    source_rows = [dict(row) for row in manifest["artifacts"]]
    by_id = {str(row.get("artifact_id")): row for row in source_rows}
    if len(by_id) != len(source_rows):
        raise EvidenceCertificateError("source manifest artifact IDs are ambiguous")
    controls = _validate_control_receipts(
        spec["control_receipts"],
        evidence_root=evidence_root,
        source_rows_by_id=by_id,
    )
    supporting_ids = {row["artifact_id"] for row in controls}
    selected_spec = {row["artifact_id"]: row for row in spec["superseded_artifacts"]}
    missing = sorted(set(selected_spec) - set(by_id))
    if missing:
        raise EvidenceCertificateError(
            "explicit superseded artifact IDs are absent: " + ",".join(missing)
        )
    overlapping_support = sorted(set(selected_spec) & supporting_ids)
    if overlapping_support:
        raise EvidenceCertificateError(
            "supporting result/control artifacts must remain preserved: "
            + ",".join(overlapping_support)
        )
    for artifact_id, selected in selected_spec.items():
        row = by_id[artifact_id]
        source_class = str(row.get("retention_class"))
        if source_class not in {"R0", "R1", "R2", "R3"}:
            raise EvidenceCertificateError(
                f"explicit superseded artifact {artifact_id} is protected as {source_class}"
            )
        if row.get("live_reference_status") not in {"none", "active"}:
            raise EvidenceCertificateError(
                f"explicit superseded artifact {artifact_id} has unknown source live state"
            )
        if row.get("audit_hold_status") not in {
            "released",
            "not_required",
        } or row.get("qualification_hold_status") not in {
            "released",
            "not_required",
        }:
            raise EvidenceCertificateError(
                f"explicit superseded artifact {artifact_id} lacks closed source exact-byte holds"
            )
        if selected["evidence_class"] not in _SUPERSEDED_EVIDENCE_CLASSES:
            raise EvidenceCertificateError(
                f"explicit superseded artifact {artifact_id} is not raw/transient"
            )

    selected_rows = [_row_projection(by_id[row_id]) for row_id in sorted(selected_spec)]
    protected_rows = [
        _row_projection(row)
        for row in source_rows
        if str(row["artifact_id"]) not in selected_spec
    ]
    selected_bytes = sum(int(row["bytes"]) for row in selected_rows)
    protected_bytes = sum(int(row["bytes"]) for row in protected_rows)
    class_counts = Counter(str(row["retention_class"]) for row in source_rows)
    class_bytes = Counter()
    for row in source_rows:
        class_bytes[str(row["retention_class"])] += int(row["bytes"])
    selected_class_counts = Counter(
        str(by_id[row_id]["retention_class"]) for row_id in selected_spec
    )
    protected_audit_rows = sum(
        1
        for row in source_rows
        if str(row["artifact_id"]) not in selected_spec
        and row.get("audit_hold_status") == "active"
    )
    protected_qualification_rows = sum(
        1
        for row in source_rows
        if str(row["artifact_id"]) not in selected_spec
        and row.get("qualification_hold_status") == "active"
    )
    row_accounting = {
        "source_row_count": len(source_rows),
        "source_bytes": sum(int(row["bytes"]) for row in source_rows),
        "selected_row_count": len(selected_rows),
        "selected_bytes": selected_bytes,
        "selected_rows_sha256": canonical_sha256(selected_rows),
        "selected_by_source_retention_class": dict(
            sorted(selected_class_counts.items())
        ),
        "protected_row_count": len(protected_rows),
        "protected_bytes": protected_bytes,
        "protected_rows_sha256": canonical_sha256(protected_rows),
        "protected_audit_hold_rows": protected_audit_rows,
        "protected_qualification_hold_rows": protected_qualification_rows,
        "by_source_retention_class": {
            name: {"count": class_counts[name], "bytes": class_bytes[name]}
            for name in sorted(class_counts)
        },
    }
    request_projection = {
        "source_manifest": source_identity,
        "review_spec_sha256": spec["spec_sha256"],
        "decision": spec["decision"],
        "conclusion": spec["conclusion"],
        "limitations": spec["limitations"],
        "no_claims": spec["no_claims"],
        "row_accounting": row_accounting,
        "control_receipts_sha256": canonical_sha256(controls),
    }
    review_core = {
        "schema_version": DEBUG_CLOSEOUT_REVIEW_RECEIPT_SCHEMA_VERSION,
        "status": spec["reviewer"]["status"],
        "reviewer": spec["reviewer"],
        "reviewed_at_utc": spec["reviewed_at_utc"],
        "source_manifest": source_identity,
        "review_spec_sha256": spec["spec_sha256"],
        "request_projection_sha256": canonical_sha256(request_projection),
        "decision": spec["decision"],
        "conclusion": spec["conclusion"],
        "limitations": spec["limitations"],
        "criteria": spec["criteria"],
        "control_receipts": controls,
        "unresolved_anomalies": spec["unresolved_anomalies"],
        "live_state": spec["live_state"],
        "active_exact_byte_holds": spec["active_exact_byte_holds"],
        "source_live_marker_closeout": spec["source_live_marker_closeout"],
        "row_accounting": row_accounting,
        "reproduction": spec["reproduction"],
        "no_claims": spec["no_claims"],
        "assurance_statement": (
            "independent substantive review"
            if spec["reviewer"]["independence"] == "INDEPENDENT_REVIEW"
            else "same-identity reduced-assurance review; independent reviewer unavailable"
        ),
        "raw_payload_embedded": False,
    }
    review_receipt = {
        **review_core,
        "receipt_sha256": canonical_sha256(review_core),
    }
    if (
        len(pretty_json(review_receipt).encode("utf-8"))
        > MAX_DEBUG_CLOSEOUT_RECEIPT_BYTES
    ):
        raise EvidenceCertificateError("DEBUG closeout review receipt exceeds its cap")

    gate_id = "DEBUG-CLOSEOUT"
    plan_id = f"debug-closeout-{manifest['manifest_sha256'][:16]}"
    plan_identity = {
        "programme_id": spec["programme_id"],
        "plan_id": plan_id,
        "ordered_gates": [gate_id],
    }
    plan_sha256 = canonical_sha256(plan_identity)
    closure_core = {
        "schema_version": "eiel_programme_gate_closure_receipt_v1",
        "programme_id": spec["programme_id"],
        "plan_id": plan_id,
        "plan_sha256": plan_sha256,
        "gate_id": gate_id,
        "closed_gates": [gate_id],
        "status": "CLOSED",
        "decision_receipt_sha256": review_receipt["receipt_sha256"],
    }
    closure_receipt = {
        **closure_core,
        "receipt_sha256": canonical_sha256(closure_core),
    }

    artifacts: list[dict[str, Any]] = []
    for source_row in source_rows:
        artifact_id = str(source_row["artifact_id"])
        source_class = str(source_row["retention_class"])
        if source_class not in _RETENTION_CLASSES:
            raise EvidenceCertificateError(
                f"source artifact {artifact_id} retention class is unsupported"
            )
        digest_name = "sha256" if "sha256" in source_row else "tree_sha256"
        if (
            source_row.get("exact_object_identity") is not True
            or type(source_row.get("bytes")) is not int
            or int(source_row["bytes"]) < 0
            or not isinstance(source_row.get(digest_name), str)
        ):
            raise EvidenceCertificateError(
                f"source artifact {artifact_id} lacks exact sealed identity"
            )
        selected = selected_spec.get(artifact_id)
        closeout_disposition = (
            "SUPERSEDED_BY_CERTIFICATE" if selected is not None else "RETAIN"
        )
        evidence_class = (
            selected["evidence_class"]
            if selected is not None
            else _retained_evidence_class(source_class)
        )
        artifacts.append(
            {
                "artifact_id": artifact_id,
                "path": _safe_relative(
                    source_row.get("path"), path=f"source artifact {artifact_id}.path"
                ),
                "artifact_family": _identifier(
                    source_row.get("artifact_family"),
                    path=f"source artifact {artifact_id}.artifact_family",
                ),
                "owner_writer": _identifier(
                    source_row.get("owner_writer"),
                    path=f"source artifact {artifact_id}.owner_writer",
                ),
                "evidence_class": evidence_class,
                "reproduction_role": (
                    "raw-diagnostic-superseded"
                    if selected is not None
                    else "source-manifest-preserved"
                ),
                "produced_by": gate_id,
                "consumers": [],
                "last_use": None,
                "live_state": dict(spec["live_state"]),
                "forensic_hold": {
                    "status": "closed" if selected is not None else "not_applicable",
                    "closure_receipt_sha256": (
                        closure_receipt["receipt_sha256"]
                        if selected is not None
                        else None
                    ),
                },
                "retention_authorities": [],
                "policy": "UNKNOWN_POLICY" if source_class == "RX" else "KNOWN",
                "dependency_state": "known",
                "source_artifact_id": artifact_id,
                "source_retention_class": source_class,
                "diagnostic_closeout_disposition": closeout_disposition,
                "source_manifest_binding": {
                    "manifest_path": manifest_relative,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "artifact_id": artifact_id,
                    "bytes": int(source_row["bytes"]),
                    "retention_class": source_class,
                    digest_name: source_row[digest_name],
                },
            }
        )

    request = {
        "schema_version": "eiel_evidence_certificate_request_v1",
        "programme_id": spec["programme_id"],
        "work_id": spec["work_id"],
        "gate_id": gate_id,
        "obligation_id": spec["obligation_id"],
        "bounded_claim": spec["bounded_claim"],
        "sealed_at_utc": spec["reviewed_at_utc"],
        "identity": {
            "source": spec["source"],
            "artifact_hashes": {
                "source-manifest": manifest["manifest_sha256"],
                "review-spec": spec["spec_sha256"],
                "review-receipt": review_receipt["receipt_sha256"],
            },
            "runtime": {
                "route": "debug-closeout-builder-v1",
                "source_campaign_id": manifest["campaign_id"],
                "source_run_id": manifest["run_id"],
            },
            "config_sha256": spec["artifact_lifecycle_contract"]["contract_sha256"],
            "request_sha256": spec["spec_sha256"],
            "plan_sha256": plan_sha256,
            "environment": spec["environment"],
            "device": spec["device"],
            "rng_bindings": [],
            "rng_not_applicable_reason": (
                "storage-only same-identity diagnostic closeout review"
            ),
        },
        "programme_plan": {
            "plan_id": plan_id,
            "plan_sha256": plan_sha256,
            "closure_receipt_sha256": closure_receipt["receipt_sha256"],
            "closure_receipt": closure_receipt,
            "ordered_gates": [gate_id],
            "closed_gates": [gate_id],
        },
        "artifact_lifecycle_contract": spec["artifact_lifecycle_contract"],
        "governing_procedures": ["EIEL-ALCM-1", "EIEL-ESAB-1"],
        "evidence_obligations": [
            {
                "obligation_id": spec["obligation_id"],
                "description": (
                    "same-identity substantive diagnostic closeout and complete source-row accounting"
                ),
                "status": (
                    "valid_negative"
                    if spec["decision"] in _SPECIAL_NO_RESULT_DECISIONS
                    else "satisfied"
                ),
            }
        ],
        "validation": {
            "mechanical_receipts": [
                {
                    "receipt_id": row["role"],
                    "status": "passed",
                    "sha256": row["sha256"],
                }
                for row in controls
            ],
            "semantic_review": {
                "status": spec["reviewer"]["status"],
                "reviewer": spec["reviewer"]["identity"],
                "independence": spec["reviewer"]["independence"],
                "receipt_sha256": review_receipt["receipt_sha256"],
            },
        },
        "result": {
            "decision": spec["decision"],
            "criteria": spec["criteria"],
            "measurements": {
                **spec["measurements"],
                "source_rows": len(source_rows),
                "selected_rows": len(selected_rows),
                "selected_bytes": selected_bytes,
                "protected_rows": len(protected_rows),
                "protected_bytes": protected_bytes,
            },
            "digests": {
                "source_manifest": manifest["manifest_sha256"],
                "selected_rows": row_accounting["selected_rows_sha256"],
                "protected_rows": row_accounting["protected_rows_sha256"],
                "review_receipt": review_receipt["receipt_sha256"],
                "control_receipts": canonical_sha256(controls),
                "supporting_artifacts": controls,
                "source_live_marker_closeout": spec["source_live_marker_closeout"],
            },
            "limitations": spec["limitations"],
            "no_claims": spec["no_claims"],
        },
        "reproduction": spec["reproduction"],
        "qualification_context": {
            "status": "not_applicable",
            "authority_reference": None,
            "exact_required_artifact_ids": [],
        },
        "audit_context": {
            "status": "released",
            "authority_reference": review_receipt["receipt_sha256"],
            "exact_required_artifact_ids": [],
        },
        "inventory_complete": True,
        "artifacts": artifacts,
        "diagnostic_closeout": {
            "source_manifest": source_identity,
            "substantive_review": {
                "status": spec["reviewer"]["status"],
                "reviewer": spec["reviewer"]["identity"],
                "receipt_sha256": review_receipt["receipt_sha256"],
            },
            "closure_receipt_sha256": closure_receipt["receipt_sha256"],
            "conclusion": spec["conclusion"],
            "limitations": spec["limitations"],
            "unresolved_anomalies": spec["unresolved_anomalies"],
            "live_state": spec["live_state"],
            "active_exact_byte_holds": spec["active_exact_byte_holds"],
            "superseded_artifact_ids": sorted(selected_spec),
            "source_live_marker_closeout": spec["source_live_marker_closeout"],
        },
    }
    normalized_request = validate_evidence_certificate_request_v1(request)
    request_bytes = pretty_json(normalized_request).encode("utf-8")
    if len(request_bytes) > MAX_DEBUG_CLOSEOUT_REQUEST_BYTES:
        raise EvidenceCertificateError("DEBUG closeout request exceeds its cap")

    report_lines = [
        "# DEBUG_FORENSIC diagnostic closeout review",
        "",
        f"- Source: `{manifest['campaign_id']}/{manifest['run_id']}`",
        f"- Source manifest: `{manifest_relative}`",
        f"- Source manifest SHA-256: `{manifest['manifest_sha256']}`",
        f"- Decision: `{spec['decision']}`",
        f"- Review: `{spec['reviewer']['status']}` by `{spec['reviewer']['identity']}`",
        f"- Independence: `{spec['reviewer']['independence']}`",
        f"- Review receipt SHA-256: `{review_receipt['receipt_sha256']}`",
        f"- Request SHA-256: `{canonical_sha256(normalized_request)}`",
        "",
        "## Conclusion",
        "",
        spec["conclusion"],
        "",
        "## Exact accounting",
        "",
        f"- Source rows/bytes: {len(source_rows)} / {row_accounting['source_bytes']}",
        f"- Superseded rows/bytes: {len(selected_rows)} / {selected_bytes}",
        f"- Preserved rows/bytes: {len(protected_rows)} / {protected_bytes}",
        f"- Superseded set digest: `{row_accounting['selected_rows_sha256']}`",
        f"- Preserved set digest: `{row_accounting['protected_rows_sha256']}`",
        f"- Preserved audit-hold rows: {protected_audit_rows}",
        f"- Preserved qualification-hold rows: {protected_qualification_rows}",
        "",
        "## Source live-marker closure",
        "",
        f"- Status: `{spec['source_live_marker_closeout']['status']}`",
        f"- Audited at: `{spec['source_live_marker_closeout']['audited_at_utc']}`",
        f"- Selected-set digest: `{spec['source_live_marker_closeout']['selected_artifact_ids_sha256']}`",
        f"- Scope: `{spec['source_live_marker_closeout']['scope']}`",
        f"- Audit method: {spec['source_live_marker_closeout']['audit_method']}",
        "- The review receipt supersedes only the stale source live-reference marker; "
        "it does not supersede any audit, qualification, replay, or other exact-byte hold.",
        "",
        "## Manifest-bound supporting result/control artifacts",
        "",
    ]
    for row in controls:
        report_lines.append(
            f"- `{row['role']}` — `{row['artifact_id']}` — `{row['path']}` — "
            f"{row['bytes']} bytes — SHA-256 `{row['sha256']}`"
        )
    report_lines.extend(
        [
            "",
            "### Explicitly superseded raw/transient rows",
            "",
        ]
    )
    for row in selected_rows:
        selected = selected_spec[str(row["artifact_id"])]
        report_lines.append(
            f"- `{row['artifact_id']}` — `{row['path']}` — {row['bytes']} bytes — "
            f"source `{row['retention_class']}` — {selected['rationale']}"
        )
    report_lines.extend(["", "## Limitations", ""])
    report_lines.extend(f"- {value}" for value in spec["limitations"])
    report_lines.extend(["", "## Claim boundary", ""])
    report_lines.extend(f"- {value}" for value in spec["no_claims"])
    report_lines.extend(["", "## Reproduction and verification", ""])
    report_lines.extend(
        f"1. {value}" for value in spec["reproduction"]["prerequisites"]
    )
    report_lines.append("")
    report_lines.extend(f"- `{value}`" for value in spec["reproduction"]["commands"])
    report_lines.append("")
    report_lines.extend(
        f"- {value}" for value in spec["reproduction"]["verification_rules"]
    )
    report_lines.extend(
        [
            "",
            "The complete machine-readable request carries every source row and exact binding. "
            "This report carries only explicit release rows plus digests and summaries; it embeds no raw payload.",
            "",
        ]
    )
    report = "\n".join(report_lines)
    report_bytes = report.encode("utf-8")
    if len(report_bytes) > MAX_DEBUG_CLOSEOUT_REPORT_BYTES:
        raise EvidenceCertificateError("DEBUG closeout report exceeds its cap")
    return {
        "request": normalized_request,
        "request_sha256": canonical_sha256(normalized_request),
        "review_receipt": review_receipt,
        "review_receipt_sha256": review_receipt["receipt_sha256"],
        "report_markdown": report,
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "row_accounting": row_accounting,
        "source_manifest": source_identity,
        "review_spec_sha256": spec["spec_sha256"],
        "raw_payload_embedded": False,
        "release_authorized": False,
    }


def validate_debug_closeout_review_receipt_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _mapping(value, path="debug_closeout_review_receipt")
    supplied = receipt.pop("receipt_sha256", None)
    if (
        receipt.get("schema_version") != DEBUG_CLOSEOUT_REVIEW_RECEIPT_SCHEMA_VERSION
        or receipt.get("raw_payload_embedded") is not False
        or receipt.get("status") not in _REVIEW_STATUSES
        or supplied != canonical_sha256(receipt)
    ):
        raise EvidenceCertificateError("DEBUG closeout review receipt is invalid")
    reviewer = _mapping(receipt.get("reviewer"), path="review_receipt.reviewer")
    independence = reviewer.get("independence")
    if independence not in _REVIEW_INDEPENDENCE:
        raise EvidenceCertificateError("DEBUG closeout review assurance is misstated")
    expected_assurance = (
        "independent substantive review"
        if independence == "INDEPENDENT_REVIEW"
        else "same-identity reduced-assurance review; independent reviewer unavailable"
    )
    if receipt.get("assurance_statement") != expected_assurance:
        raise EvidenceCertificateError("DEBUG closeout review assurance is misstated")
    live = receipt.get("live_state")
    if (
        receipt.get("unresolved_anomalies") != []
        or not isinstance(live, Mapping)
        or any(live.values())
    ):
        raise EvidenceCertificateError("DEBUG closeout review retains unresolved state")
    holds = receipt.get("active_exact_byte_holds", {})
    if not isinstance(holds, Mapping) or any(holds.values()):
        raise EvidenceCertificateError("DEBUG closeout review retains exact-byte holds")
    live_closeout = _mapping(
        receipt.get("source_live_marker_closeout"),
        path="review_receipt.source_live_marker_closeout",
    )
    source_manifest = _mapping(
        receipt.get("source_manifest"), path="review_receipt.source_manifest"
    )
    closeout_checks = live_closeout.get("checks")
    row_accounting = receipt.get("row_accounting")
    if (
        live_closeout.get("schema_version")
        != DIAGNOSTIC_SOURCE_LIVE_MARKER_CLOSEOUT_SCHEMA_VERSION
        or live_closeout.get("status") != "CLOSED_BY_CURRENT_STATE_AUDIT"
        or live_closeout.get("scope") != "SOURCE_MANIFEST_LIVE_REFERENCE_MARKER_ONLY"
        or live_closeout.get("source_manifest_sha256")
        != source_manifest.get("manifest_sha256")
        or live_closeout.get("source_campaign_id") != source_manifest.get("campaign_id")
        or live_closeout.get("source_run_id") != source_manifest.get("run_id")
        or live_closeout.get("source_runtime_root")
        != source_manifest.get("runtime_root")
        or not isinstance(live_closeout.get("selected_artifact_ids_sha256"), str)
        or _SHA256_RE.fullmatch(live_closeout["selected_artifact_ids_sha256"]) is None
        or not isinstance(row_accounting, Mapping)
        or live_closeout.get("selected_artifact_count")
        != row_accounting.get("selected_row_count")
        or not isinstance(closeout_checks, Mapping)
        or any(
            closeout_checks.get(name) is not False
            for name in (
                "writer",
                "reference",
                "resume",
                "reducer",
                "consumer",
                "canonical_reservation_overlap",
            )
        )
        or closeout_checks.get("relay_process_references") != []
    ):
        raise EvidenceCertificateError("DEBUG source live-marker closeout is invalid")
    return {**receipt, "receipt_sha256": supplied}


__all__ = [
    "DEBUG_CLOSEOUT_PREPARATION_PLAN_SCHEMA_VERSION",
    "DEBUG_CLOSEOUT_REVIEW_INPUT_SCHEMA_VERSION",
    "DEBUG_CLOSEOUT_REVIEW_RECEIPT_SCHEMA_VERSION",
    "DEBUG_CLOSEOUT_REVIEW_SPEC_SCHEMA_VERSION",
    "MAX_DEBUG_CLOSEOUT_REPORT_BYTES",
    "MAX_DEBUG_CLOSEOUT_REQUEST_BYTES",
    "MAX_DEBUG_CLOSEOUT_SOURCE_ROWS",
    "build_debug_closeout_bundle_v1",
    "draft_debug_closeout_review_spec_v1",
    "validate_debug_closeout_review_receipt_v1",
    "validate_debug_closeout_review_spec_v1",
]
