"""Evidence-entailing review proof contract for Work Package acceptance."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

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
from tools.actors.portfolio import normalize_string_list
from tools.actors.repository_truth import ARTIFACT_REFERENCE_SCHEMA_VERSION
from tools.actors.lifecycle_policy import actor_record_ref

REVIEW_PROOF_SCHEMA_VERSION = "eiel_review_obligation_proof_v2"
LEGACY_REVIEW_PROOF_SCHEMA_VERSION = "eiel_review_obligation_proof_v1"
EXECUTION_RECEIPT_SCHEMA_VERSION = "eiel_review_execution_receipt_v1"
REVIEW_PROOF_ARTIFACT_SCHEMA_VERSION = "eiel_review_obligation_proof_artifact_v1"
WORK_TERMINAL_DECISION_RECEIPT_SCHEMA_VERSION = "eiel_work_terminal_decision_receipt_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_KINDS = {"acceptance_criterion", "test", "evidence_item"}


def _write_immutable(path: Path, value: Mapping[str, Any]) -> Path:
    if path.exists():
        existing = path.read_bytes()
        expected = (__import__("json").dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
        if existing != expected:
            raise ActorError("immutable review proof conflict", code="review_proof_artifact_conflict", problems=[str(path)])
        return path
    return atomic_write_json(path, value)


def _artifact_path(uri: str, *, source_dir: Path) -> Path | None:
    value = str(uri or "").strip()
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).expanduser().resolve()
    if parsed.scheme:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = source_dir / candidate
    return candidate.resolve()


def load_review_proof_artifact(
    reference: Mapping[str, Any],
    *,
    state_root: str | Path | None,
) -> dict[str, Any]:
    require_explicit_schema_version(
        reference, expected=ARTIFACT_REFERENCE_SCHEMA_VERSION, label="review proof reference"
    )
    locator = str(reference.get("state_root_locator") or "").strip()
    candidates = [Path(str(reference.get("path") or ""))]
    if locator:
        candidates.append(actor_state_root(state_root) / locator)
    path = next((candidate for candidate in candidates if str(candidate) and candidate.is_file()), None)
    if path is None:
        raise ActorError("review proof artifact is missing", code="review_proof_artifact_missing")
    if file_sha256(path) != str(reference.get("sha256") or ""):
        raise ActorError("review proof artifact digest mismatch", code="review_proof_artifact_digest_mismatch")
    value = read_structured(path, name="review proof artifact")
    require_explicit_schema_version(
        value, expected=REVIEW_PROOF_ARTIFACT_SCHEMA_VERSION, label="review proof artifact"
    )
    material = dict(value)
    embedded = str(material.pop("artifact_sha256", ""))
    if not _SHA256_RE.fullmatch(embedded) or canonical_sha256(material) != embedded:
        raise ActorError("review proof canonical digest mismatch", code="review_proof_artifact_digest_mismatch")
    if str(reference.get("artifact_id") or "") != str(value.get("artifact_id") or ""):
        raise ActorError("review proof artifact identity mismatch", code="review_proof_artifact_identity_mismatch")
    return dict(value)


def retain_and_validate_review_proof(
    proof_path: str | Path,
    *,
    state_root: str | Path | None,
    work_id: str,
    review_session_id: str,
    reviewed_commit: str,
    reviewed_package_sha256: str,
    package: Mapping[str, Any],
    generated_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = read_structured(proof_path, name="review obligation proof")
    require_explicit_schema_version(
        raw, expected=REVIEW_PROOF_SCHEMA_VERSION, label="review obligation proof"
    )
    if str(raw.get("work_id") or "") != work_id:
        raise ActorError("review proof belongs to another Work Item", code="review_proof_identity_mismatch")
    if str(raw.get("reviewed_commit") or "").lower() != str(reviewed_commit).lower():
        raise ActorError("review proof commit does not match review target", code="review_proof_identity_mismatch")
    if str(raw.get("reviewed_package_sha256") or "") != reviewed_package_sha256:
        raise ActorError("review proof package does not match review target", code="review_proof_identity_mismatch")
    if str(raw.get("implementation_sufficiency") or "").upper() != "SUFFICIENT":
        raise ActorError(
            "ACCEPT requires an explicit implementation-sufficiency conclusion",
            code="review_implementation_sufficiency_missing",
        )
    declared = {
        "acceptance_criterion": set(normalize_string_list(package.get("acceptance_criteria"))),
        "test": set(normalize_string_list(package.get("tests"))),
        "evidence_item": set(normalize_string_list(package.get("evidence"))),
    }
    observed: dict[str, set[str]] = {key: set() for key in _ALLOWED_KINDS}
    problems: list[str] = []
    normalized_results: list[dict[str, Any]] = []
    results = raw.get("obligation_results")
    if not isinstance(results, list):
        problems.append("obligation_results must be an array")
        results = []
    for index, row in enumerate(results):
        if not isinstance(row, Mapping):
            problems.append(f"obligation result {index} is not an object")
            continue
        kind = str(row.get("kind") or "")
        obligation = str(row.get("obligation") or "").strip()
        status = str(row.get("status") or "").upper()
        refs = [dict(ref) for ref in row.get("evidence_refs") or [] if isinstance(ref, Mapping)]
        execution = dict(row.get("execution_receipt") or {}) if isinstance(row.get("execution_receipt"), Mapping) else None
        if kind not in _ALLOWED_KINDS:
            problems.append(f"obligation result {index} has unsupported kind: {kind}")
            continue
        if not obligation:
            problems.append(f"obligation result {index} has no obligation text")
            continue
        if obligation in observed[kind]:
            problems.append(f"duplicate {kind}: {obligation}")
        observed[kind].add(obligation)
        if status != "PASS":
            problems.append(f"{kind} did not PASS: {obligation} ({status or 'missing status'})")
        if not refs:
            problems.append(f"{kind} has no evidence reference: {obligation}")
        if kind in {"test", "evidence_item"}:
            if execution is None:
                problems.append(f"{kind} has no execution provenance: {obligation}")
            else:
                if execution.get("schema_version") != EXECUTION_RECEIPT_SCHEMA_VERSION:
                    problems.append(f"{kind} execution receipt has unsupported schema: {obligation}")
                if str(execution.get("reviewed_commit") or "").lower() != str(reviewed_commit).lower():
                    problems.append(f"{kind} execution receipt commit mismatch: {obligation}")
                mode = str(execution.get("execution_mode") or "").upper()
                if mode not in {"REVIEWER_EXECUTED", "EXTERNAL_IMPORTED"}:
                    problems.append(f"{kind} execution receipt is inherited/unproven: {obligation}")
                if str(execution.get("review_session_id") or "") != str(review_session_id):
                    problems.append(f"{kind} execution receipt is not bound to this Review Session: {obligation}")
                relation = str(execution.get("evidence_relation") or "").upper()
                expected_relation = (
                    "EXECUTED_IN_REVIEW" if mode == "REVIEWER_EXECUTED" else "IMPORTED_AND_VERIFIED"
                )
                if relation != expected_relation:
                    problems.append(
                        f"{kind} execution receipt has invalid evidence relation for {mode or 'unknown mode'}: {obligation}"
                    )
                if mode == "EXTERNAL_IMPORTED" and not str(execution.get("source_execution_id") or "").strip():
                    problems.append(f"{kind} imported execution receipt has no source execution identity: {obligation}")
                if str(execution.get("status") or "").upper() != "PASS":
                    problems.append(f"{kind} execution receipt did not PASS: {obligation}")
                if not str(execution.get("executor_id") or "").strip():
                    problems.append(f"{kind} execution receipt has no executor: {obligation}")
                output_sha = str(execution.get("output_sha256") or "").lower()
                if not _SHA256_RE.fullmatch(output_sha):
                    problems.append(f"{kind} execution receipt has invalid output_sha256: {obligation}")
                if not isinstance(execution.get("environment"), Mapping):
                    problems.append(f"{kind} execution receipt has no environment binding: {obligation}")
        seen_refs: set[tuple[str, str]] = set()
        for ref in refs:
            uri = str(ref.get("uri") or ref.get("path") or "").strip()
            if not uri:
                problems.append(f"{kind} evidence reference has no uri: {obligation}")
            sha = str(ref.get("sha256") or "").lower()
            if not _SHA256_RE.fullmatch(sha):
                problems.append(f"{kind} evidence reference has invalid sha256: {obligation}")
            key = (uri, sha)
            if key in seen_refs:
                problems.append(f"{kind} contains a duplicate evidence reference: {obligation}")
            seen_refs.add(key)
            local = _artifact_path(uri, source_dir=Path(proof_path).expanduser().resolve().parent)
            if local is not None:
                if not local.is_file():
                    problems.append(f"{kind} evidence artifact does not exist: {local}")
                elif _SHA256_RE.fullmatch(sha):
                    observed_sha = file_sha256(local)
                    if observed_sha != sha:
                        problems.append(
                            f"{kind} evidence artifact digest mismatch for {obligation}: "
                            f"expected {sha}, observed {observed_sha}"
                        )
        normalized_results.append({
            "kind": kind,
            "obligation": obligation,
            "status": status,
            "evidence_refs": refs,
            "execution_receipt": execution,
        })
    for kind, expected in declared.items():
        for value in sorted(expected - observed[kind]):
            problems.append(f"missing proven {kind}: {value}")
        for value in sorted(observed[kind] - expected):
            problems.append(f"undeclared proven {kind}: {value}")
    if problems:
        raise ActorError(
            "ACCEPT review proof does not entail the Work Package obligations",
            code="review_obligation_proof_invalid",
            problems=problems,
        )
    artifact: dict[str, Any] = {
        "schema_version": REVIEW_PROOF_ARTIFACT_SCHEMA_VERSION,
        "artifact_id": None,
        "work_id": work_id,
        "review_session_id": review_session_id,
        "reviewed_commit": str(reviewed_commit).lower(),
        "reviewed_package_sha256": reviewed_package_sha256,
        "implementation_sufficiency": "SUFFICIENT",
        "obligation_results": sorted(normalized_results, key=lambda row: (row["kind"], row["obligation"])),
        "limitations": normalize_string_list(raw.get("limitations")),
        "source_document_sha256": file_sha256(proof_path),
        "recorded_at_utc": utc_now(generated_at),
    }
    artifact["artifact_id"] = stable_id("REVPROOF", {key: value for key, value in artifact.items() if key != "artifact_id"})
    material = dict(artifact)
    artifact["artifact_sha256"] = canonical_sha256(material)
    path = actor_state_root(state_root) / "review-proofs" / f"{artifact['artifact_id']}.json"
    _write_immutable(path, artifact)
    reference = {
        "schema_version": ARTIFACT_REFERENCE_SCHEMA_VERSION,
        "artifact_id": artifact["artifact_id"],
        "artifact_schema_version": REVIEW_PROOF_ARTIFACT_SCHEMA_VERSION,
        "path": str(path),
        "state_root_locator": path.relative_to(actor_state_root(state_root)).as_posix(),
        "sha256": file_sha256(path),
        "canonical_sha256": canonical_sha256(artifact),
    }
    return artifact, reference


def retain_work_terminal_decision_receipt(
    work_record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    final_status: str,
    completion_policy: str,
    completion_basis: str,
    completion_reason: str,
    accepted: bool,
    acceptance_scope: str,
    accepted_review_session_id: str | None,
    acceptance_evidence: Mapping[str, Any] | None,
    repository_integration_receipt: Mapping[str, Any] | None,
    qualification_decision: Mapping[str, Any] | None = None,
    dependency_state: Mapping[str, Any] | None = None,
    generated_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retain one self-contained immutable terminal Work decision.

    The current Work record remains the lifecycle head.  This receipt freezes
    the exact predecessor, completion contract, supporting evidence and
    authority statement so later recovery does not need to reconstruct the
    terminal decision from mutable projections or surviving Git objects.
    """

    payload = dict(work_record.get("payload") or {})
    artifact: dict[str, Any] = {
        "schema_version": WORK_TERMINAL_DECISION_RECEIPT_SCHEMA_VERSION,
        "artifact_id": None,
        "work_id": str(work_record.get("record_id") or payload.get("work_id") or ""),
        "pre_terminal_work": actor_record_ref(work_record),
        "work_contract": {
            "work_type": str(payload.get("work_type") or ""),
            "work_package_binding": dict(payload.get("work_package_binding") or {}) or None,
            "dependency_work_ids": list(payload.get("dependency_work_ids") or []),
        },
        "terminal_decision": {
            "final_status": str(final_status),
            "completion_policy": str(completion_policy),
            "completion_basis": str(completion_basis),
            "completion_reason": str(completion_reason),
            "accepted": bool(accepted),
            "acceptance_scope": str(acceptance_scope),
            "accepted_review_session_id": str(accepted_review_session_id or "") or None,
        },
        "acceptance_evidence": dict(acceptance_evidence or {}) or None,
        "repository_integration_receipt": dict(repository_integration_receipt or {}) or None,
        "qualification_decision": dict(qualification_decision or {}) or None,
        "dependency_state": dict(dependency_state or {}) or None,
        "deciding_authority": {
            "owner": "workManager",
            "authority_relation": "work_lifecycle_owner",
            "action": "complete",
            "assurance": "repository_local_policy_enforced",
        },
        "recorded_at_utc": utc_now(generated_at),
    }
    artifact["artifact_id"] = stable_id(
        "WDEC",
        {key: value for key, value in artifact.items() if key != "artifact_id"},
    )
    material = dict(artifact)
    artifact["artifact_sha256"] = canonical_sha256(material)
    path = actor_state_root(state_root) / "work-terminal-decisions" / f"{artifact['artifact_id']}.json"
    _write_immutable(path, artifact)
    reference = {
        "schema_version": ARTIFACT_REFERENCE_SCHEMA_VERSION,
        "artifact_id": artifact["artifact_id"],
        "artifact_schema_version": WORK_TERMINAL_DECISION_RECEIPT_SCHEMA_VERSION,
        "path": str(path),
        "state_root_locator": path.relative_to(actor_state_root(state_root)).as_posix(),
        "sha256": file_sha256(path),
        "canonical_sha256": canonical_sha256(artifact),
    }
    return artifact, reference


def load_work_terminal_decision_receipt(
    reference: Mapping[str, Any],
    *,
    state_root: str | Path | None,
) -> dict[str, Any]:
    require_explicit_schema_version(
        reference,
        expected=ARTIFACT_REFERENCE_SCHEMA_VERSION,
        label="Work terminal decision receipt reference",
    )
    locator = str(reference.get("state_root_locator") or "").strip()
    candidates = [Path(str(reference.get("path") or ""))]
    if locator:
        candidates.append(actor_state_root(state_root) / locator)
    path = next((candidate for candidate in candidates if str(candidate) and candidate.is_file()), None)
    if path is None:
        raise ActorError(
            "Work terminal decision receipt is missing",
            code="work_terminal_decision_receipt_missing",
        )
    if file_sha256(path) != str(reference.get("sha256") or ""):
        raise ActorError(
            "Work terminal decision receipt digest mismatch",
            code="work_terminal_decision_receipt_digest_mismatch",
        )
    value = read_structured(path, name="Work terminal decision receipt")
    require_explicit_schema_version(
        value,
        expected=WORK_TERMINAL_DECISION_RECEIPT_SCHEMA_VERSION,
        label="Work terminal decision receipt",
    )
    material = dict(value)
    embedded = str(material.pop("artifact_sha256", "") or "")
    if not _SHA256_RE.fullmatch(embedded) or canonical_sha256(material) != embedded:
        raise ActorError(
            "Work terminal decision receipt canonical digest mismatch",
            code="work_terminal_decision_receipt_digest_mismatch",
        )
    if str(reference.get("artifact_id") or "") != str(value.get("artifact_id") or ""):
        raise ActorError(
            "Work terminal decision receipt identity mismatch",
            code="work_terminal_decision_receipt_identity_mismatch",
        )
    return dict(value)


__all__ = [
    "EXECUTION_RECEIPT_SCHEMA_VERSION",
    "LEGACY_REVIEW_PROOF_SCHEMA_VERSION",
    "REVIEW_PROOF_ARTIFACT_SCHEMA_VERSION",
    "REVIEW_PROOF_SCHEMA_VERSION",
    "WORK_TERMINAL_DECISION_RECEIPT_SCHEMA_VERSION",
    "load_review_proof_artifact",
    "load_work_terminal_decision_receipt",
    "retain_and_validate_review_proof",
    "retain_work_terminal_decision_receipt",
]
