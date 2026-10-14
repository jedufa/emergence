"""Compact deterministic continuation receipts for Codex context epochs.

A Context Receipt is a projection over repository, Work Package, Work, and
Session truth. It does not create authority, retain transcripts, launch models,
or replace the underlying records.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from tools.actors.canonical import canonical_sha256, pretty_json
from tools.actors.bootstrap import git_repository_identity
from tools.actors.common import (
    ActorError,
    actor_state_root,
    advisory_lock,
    atomic_write_json,
    file_sha256,
    require_explicit_schema_version,
    stable_id,
    validate_timestamp,
)
from tools.actors.portfolio import bootstrap_store, work_store
from tools.actors.delivery_semantics import delivery_commit as delivery_commit_value
from tools.actors.lifecycle_policy import work_execution_ref, wra_continuation_capsule
from tools.actors.work_package import (
    effective_execution_budget,
    effective_output_policy,
    effective_read_contract,
    package_path_matches,
    verify_work_package_binding,
)

CONTEXT_RECEIPT_SCHEMA_VERSION = "eiel_context_receipt_v1"
CONTEXT_EVIDENCE_ARTIFACT_SCHEMA_VERSION = "eiel_context_evidence_artifact_v1"
CONTEXT_RECEIPT_ID_PREFIX = "CTX"
CONTEXT_RECEIPT_DIRECTORY = "context-receipts"
CONTEXT_HEALTH_STATUSES = (
    "NOT_OBSERVED",
    "HEALTHY",
    "CAUTION",
    "HANDOFF_RECOMMENDED",
)
AUTHORITY_PHASES = {
    "ARCHITECTURE_AUTHORITY": "architecture",
    "ENGINEERING_AUTHORITY": "engineering",
    "IMPLEMENTATION_AUTHORITY": "implementation",
    "REVIEW_AUTHORITY": "review",
}
PHASE_BOUNDARY_OUTCOMES = {
    "ARCHITECTURE_COMPLETE",
    "ENGINEERING_COMPLETE",
    "IMPLEMENTATION_COMPLETE",
    "REVIEW_COMPLETE",
    "PROMOTION_REQUESTED",
}


def epoch_for_session(
    raw: Mapping[str, Any] | None,
    *,
    session_id: str,
    authority_role: str,
    generated_at: str,
) -> dict[str, Any]:
    value = dict(raw or {})
    started = validate_timestamp(str(value.get("started_at_utc") or generated_at))
    phase = str(value.get("authority_phase") or AUTHORITY_PHASES.get(authority_role, "implementation")).strip().lower()
    epoch_id = str(value.get("epoch_id") or stable_id("EPOCH", {
        "session_id": session_id,
        "authority_phase": phase,
        "started_at_utc": started,
    })).strip().upper()
    ended = value.get("ended_at_utc")
    return {
        "epoch_id": epoch_id,
        "authority_phase": phase,
        "started_at_utc": started,
        "ended_at_utc": validate_timestamp(str(ended)) if ended else None,
        "boundary_reason": str(value.get("boundary_reason") or "").strip() or None,
    }


def normalize_observed_context_health(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    value = dict(raw or {})

    def count(name: str) -> int | None:
        raw_value = value.get(name)
        if raw_value is None or raw_value == "":
            return None
        if not isinstance(raw_value, int) or isinstance(raw_value, bool) or raw_value < 0:
            raise ActorError(
                f"context_health.{name} must be a non-negative integer or null",
                code="context_health_invalid",
            )
        return int(raw_value)

    source = str(value.get("source") or "unavailable").strip().lower()
    if source not in {"manual", "codex_log", "unavailable"}:
        raise ActorError(
            "context_health.source must be manual, codex_log, or unavailable",
            code="context_health_invalid",
        )
    return {
        "tool_call_count": count("tool_call_count"),
        "compaction_count": count("compaction_count"),
        "broad_output_events": count("broad_output_events"),
        "shell_output_tokens": count("shell_output_tokens"),
        "source": source,
        "observed_at_utc": validate_timestamp(str(value["observed_at_utc"])) if value.get("observed_at_utc") else None,
        "handoff_recommended": bool(value.get("handoff_recommended")),
    }


def project_context_health(
    observed: Mapping[str, Any] | None,
    *,
    execution_budget: Mapping[str, Any],
    session_status: str,
    session_outcome: str | None,
    epoch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = normalize_observed_context_health(observed)
    reasons: list[str] = []
    status = "HEALTHY"
    handoff = bool(metrics["handoff_recommended"])
    tool_calls = metrics["tool_call_count"]
    compactions = metrics["compaction_count"]
    broad_events = metrics["broad_output_events"]
    threshold = int(execution_budget.get("handoff_recommended_after_tool_calls") or 50)
    maximum_compactions = int(execution_budget.get("maximum_compactions") or 1)

    if session_outcome in PHASE_BOUNDARY_OUTCOMES:
        handoff = True
        reasons.append(f"authority phase reached boundary: {session_outcome}")
    if epoch and epoch.get("boundary_reason"):
        handoff = True
        reasons.append(str(epoch["boundary_reason"]))
    if tool_calls is not None:
        if tool_calls >= threshold:
            handoff = True
            reasons.append(f"tool-call observation {tool_calls} reached advisory handoff threshold {threshold}")
        elif tool_calls >= max(1, int(threshold * 0.8)):
            status = "CAUTION"
            reasons.append(f"tool-call observation {tool_calls} is near advisory threshold {threshold}")
    if compactions is not None:
        if compactions >= maximum_compactions:
            handoff = True
            reasons.append(f"compaction observation {compactions} reached advisory limit {maximum_compactions}")
        elif compactions > 0:
            status = "CAUTION"
            reasons.append(f"context has compacted {compactions} time(s)")
    if broad_events is not None and broad_events > 0:
        status = "CAUTION" if not handoff else status
        reasons.append(f"{broad_events} broad-output event(s) were recorded")
    if handoff:
        status = "HANDOFF_RECOMMENDED"
    elif all(metrics[name] is None for name in ("tool_call_count", "compaction_count", "broad_output_events", "shell_output_tokens")):
        status = "NOT_OBSERVED"
        reasons.append("Codex context telemetry was not supplied; no counters were invented")
    elif not reasons:
        reasons.append("observed context remains inside advisory package budget")
    if str(session_status).upper() in {"COMPLETED", "DISCONTINUED", "OBSOLETE", "SUPERSEDED"} and not handoff:
        status = "HANDOFF_RECOMMENDED"
        handoff = True
        reasons.append("session is terminal; continue in a fresh conversation only if Work remains open")
    return {
        **metrics,
        "status": status,
        "reasons": list(dict.fromkeys(reasons)),
        "handoff_recommended": handoff,
        "policy": str(execution_budget.get("context_health_policy") or "advisory"),
        "budget": dict(execution_budget),
    }


def session_context_material(session_record: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(session_record.get("payload") or {})
    fields = (
        "session_id",
        "work_id",
        "mission_id",
        "bootstrap_record_id",
        "session_role",
        "original_codex_id",
        "current_codex_id",
        "current_codex_thread_url",
        "preferred_continuation_codex_id",
        "codex_participants",
        "worker_identity_state",
        "model_assignment",
        "work_package_binding",
        "repository",
        "current_stage",
        "current_step",
        "completed_steps",
        "last_successful_command",
        "last_failed_command",
        "validation_events",
        "validation_gaps",
        "validation_decision_impact",
        "resumed_from",
        "interrupted_during",
        "blocked_reason",
        "blocked_by_work_ids",
        "next_action",
        "next_command",
        "session_outcome",
        "review_verdict",
        "promotion_request",
        "delivery_evidence",
        "review_target",
        "review_evidence",
        "authoritative_review_binding",
        "references",
        "epoch",
        "epoch_history",
        "context_health",
    )
    return {
        "record_id": session_record.get("record_id"),
        "status": session_record.get("status"),
        "payload": {field: payload.get(field) for field in fields},
    }


def session_context_sha256(session_record: Mapping[str, Any]) -> str:
    return canonical_sha256(session_context_material(session_record))


def _repository_file_identity(repo_root: Path, relative_path: str, *, scope: str) -> dict[str, Any]:
    path = repo_root / PurePosixPath(relative_path)
    exists = path.is_file()
    return {
        "path": relative_path,
        "scope": scope,
        "exists": exists,
        "sha256": file_sha256(path) if exists else None,
    }


def _authority_documents(
    repo_root: Path,
    *,
    package: Mapping[str, Any],
    bootstrap_record: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    read_contract = effective_read_contract(package)
    candidates: list[tuple[str, str]] = [("AGENTS.md", "repository")]
    for path in read_contract["required_documents"]:
        candidates.append((path, "work_package"))
    if bootstrap_record:
        task = dict(dict(bootstrap_record.get("payload") or {}).get("task") or {})
        ref = str(task.get("operator_instruction_ref") or "").strip()
        if ref and not Path(ref).is_absolute():
            candidates.append((PurePosixPath(ref).as_posix(), "bootstrap"))
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path, scope in candidates:
        if path in seen:
            continue
        seen.add(path)
        output.append(_repository_file_identity(repo_root, path, scope=scope))
    return output


def _critical_files(repo_root: Path, package: Mapping[str, Any]) -> list[dict[str, Any]]:
    read_contract = effective_read_contract(package)
    rows: list[tuple[str, str]] = []
    rows.extend((path, "entry_file") for path in read_contract["entry_files"])
    rows.extend((path, "required_document") for path in read_contract["required_documents"])
    # Exact owned files are decision-relevant and cheap to bind. Globbed or
    # directory scopes are checked through Git path-diff classification.
    rows.extend(
        (path, "owned_file")
        for path in sorted(str(value) for value in (package.get("owned_files") or []))
        if not any(token in path for token in ("*", "?", "[")) and (repo_root / path).is_file()
    )
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path, scope in rows:
        if path in seen:
            continue
        seen.add(path)
        output.append(_repository_file_identity(repo_root, path, scope=scope))
    return output


def _evidence_artifact_identity(
    raw_path: str,
    *,
    repo_root: Path,
    evidence_kind: str,
) -> dict[str, Any]:
    """Content-address one decision-relevant evidence path when available."""

    written = str(raw_path or "").strip()
    candidate = Path(written).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()
    exists = resolved.is_file()
    return {
        "schema_version": CONTEXT_EVIDENCE_ARTIFACT_SCHEMA_VERSION,
        "kind": str(evidence_kind),
        "path": written,
        "resolved_path": str(resolved),
        "exists": exists,
        "sha256": file_sha256(resolved) if exists else None,
        "binding_status": "CONTENT_ADDRESSED" if exists else "UNRESOLVED_LEGACY_REFERENCE",
    }


def _dependency_closure(
    work_record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
) -> dict[str, Any]:
    """Retain exact dependency truth visible at receipt creation.

    This is a compact recursive closure over authoritative Work records, not a
    second dependency engine.  The Work Manager remains the owner of lifecycle
    truth; the receipt merely binds the exact status, acceptance scope,
    limitations, and execution material that continuation relied upon.
    """

    store = work_store(state_root)
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    queue = [str(value).strip().upper() for value in dict(work_record.get("payload") or {}).get("dependency_work_ids") or [] if str(value).strip()]
    while queue:
        dependency_id = queue.pop(0)
        if dependency_id in seen:
            continue
        seen.add(dependency_id)
        try:
            dependency = store.get_read_only(dependency_id)
        except ActorError:
            rows.append({
                "work_id": dependency_id,
                "status": "MISSING",
                "record_sha256": None,
                "work_execution_ref": None,
                "acceptance_scope": None,
                "limitations": [],
                "terminal_decision_receipt": None,
                "dependency_work_ids": [],
            })
            continue
        payload = dict(dependency.get("payload") or {})
        acceptance = dict(payload.get("acceptance_evidence") or {})
        child_ids = sorted({str(value).strip().upper() for value in payload.get("dependency_work_ids") or [] if str(value).strip()})
        rows.append({
            "work_id": dependency_id,
            "status": str(dependency.get("status") or "UNKNOWN"),
            "record_sha256": dependency.get("record_sha256"),
            "work_execution_ref": work_execution_ref(dependency),
            "acceptance_scope": payload.get("acceptance_scope"),
            "limitations": sorted({str(value).strip() for value in acceptance.get("known_limitations") or [] if str(value).strip()}),
            "terminal_decision_receipt": dict(payload.get("terminal_decision_receipt") or {}) if isinstance(payload.get("terminal_decision_receipt"), Mapping) else None,
            "dependency_work_ids": child_ids,
        })
        queue.extend(value for value in child_ids if value not in seen)
    rows.sort(key=lambda row: str(row.get("work_id") or ""))
    material = {
        "schema_version": "eiel_dependency_closure_v1",
        "root_work_id": str(work_record.get("record_id") or ""),
        "dependencies": rows,
    }
    material["closure_sha256"] = canonical_sha256(material)
    return material


def _declared_scope_material(*, package: Mapping[str, Any], read_contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "owned_paths": sorted(str(value) for value in (package.get("owned_files") or [])),
        "forbidden_paths": sorted(str(value) for value in (package.get("forbidden_files") or [])),
        **dict(read_contract),
    }


def build_context_receipt(
    *,
    work_record: Mapping[str, Any],
    session_record: Mapping[str, Any],
    bootstrap_record: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
    state_root: str | Path | None = None,
) -> dict[str, Any]:
    work_payload = dict(work_record.get("payload") or {})
    session_payload = dict(session_record.get("payload") or {})
    binding = dict(session_payload.get("work_package_binding") or work_payload.get("work_package_binding") or {})
    repository = git_repository_identity(
        repo_root
        or dict(session_payload.get("repository") or {}).get("repo_root")
        or binding.get("bound_repo_root")
    )
    if binding:
        verified = verify_work_package_binding(
            binding,
            repo_root=repository["root"],
            expected_work_id=str(session_payload.get("work_id") or work_record.get("record_id") or ""),
            state_root=state_root,
        )
        if not verified["valid"] or not isinstance(verified.get("package"), Mapping):
            raise ActorError(
                "Context Receipt cannot use an invalid Work Package binding",
                code="context_receipt_package_invalid",
                problems=list(verified.get("errors") or []),
                warnings=list(verified.get("warnings") or []),
            )
        package = dict(verified["package"])
    else:
        # A Context Receipt is useful recovery assistance even when the operator
        # chose not to create a Work Package. Preserve the exact repository,
        # Work, Session, progress, validation, and evidence state without
        # inventing a file fence or acceptance contract.
        package = {}
    read_contract = effective_read_contract(package)
    output_policy = effective_output_policy(package)
    execution_budget = effective_execution_budget(package)
    epoch = epoch_for_session(
        session_payload.get("epoch") if isinstance(session_payload.get("epoch"), Mapping) else None,
        session_id=str(session_record.get("record_id") or session_payload.get("session_id")),
        authority_role=str(session_payload.get("session_role") or "IMPLEMENTATION_AUTHORITY"),
        generated_at=str(session_record.get("created_at_utc") or session_record.get("updated_at_utc")),
    )
    health = project_context_health(
        session_payload.get("context_health") if isinstance(session_payload.get("context_health"), Mapping) else None,
        execution_budget=execution_budget,
        session_status=str(session_record.get("status") or "UNKNOWN"),
        session_outcome=str(session_payload.get("session_outcome") or "") or None,
        epoch=epoch,
    )
    authority_documents = _authority_documents(
        Path(repository["root"]), package=package, bootstrap_record=bootstrap_record
    )
    repository_root = Path(repository["root"])
    critical_files = _critical_files(repository_root, package)
    missing_required = [
        row["path"]
        for row in critical_files
        if row.get("scope") in {"entry_file", "required_document"} and not row.get("exists")
    ]
    if missing_required:
        raise ActorError(
            "Context Receipt required initial files are missing",
            code="context_receipt_required_file_missing",
            problems=missing_required,
        )
    references = dict(session_payload.get("references") or {})
    evidence_artifacts = [
        *[
            _evidence_artifact_identity(
                str(path), repo_root=repository_root, evidence_kind="artifact"
            )
            for path in references.get("artifact_paths") or []
        ],
        *[
            _evidence_artifact_identity(
                str(path), repo_root=repository_root, evidence_kind="report"
            )
            for path in references.get("report_paths") or []
        ],
    ]
    retained_review_binding = (
        dict(session_payload.get("authoritative_review_binding") or {})
        if isinstance(session_payload.get("authoritative_review_binding"), Mapping)
        else None
    )
    if state_root is not None:
        from tools.actors.work_manager import authoritative_review_binding_for_work

        current_review_binding = authoritative_review_binding_for_work(
            work_record,
            state_root=state_root,
            read_only=True,
        )
        if retained_review_binding and retained_review_binding != current_review_binding:
            raise ActorError(
                "Context Receipt cannot bind a stale authoritative review",
                code="context_receipt_review_binding_stale",
                problems=[
                    f"retained review: {retained_review_binding.get('review_session_id')}",
                    f"current review: {dict(current_review_binding or {}).get('review_session_id')}",
                ],
            )
        requires_review_binding = (
            str(session_record.get("status") or "")
            not in {"COMPLETED", "DISCONTINUED", "OBSOLETE", "SUPERSEDED"}
            and str(session_payload.get("session_role") or "") == "IMPLEMENTATION_AUTHORITY"
            and str(dict(current_review_binding or {}).get("verdict") or "")
            in {"CHANGES_REQUIRED", "REJECT", "BLOCKED"}
        )
        if requires_review_binding and not retained_review_binding:
            raise ActorError(
                "repair Context Receipt requires the current authoritative review binding",
                code="context_receipt_review_binding_missing",
            )
    declared_scope = _declared_scope_material(package=package, read_contract=read_contract)
    dependency_closure = _dependency_closure(work_record, state_root=state_root)
    material = {
        "schema_version": CONTEXT_RECEIPT_SCHEMA_VERSION,
        "currentness_claim": (
            "CONTEXT_CURRENT_TO_DECLARED_SCOPE"
            if binding
            else "CONTEXT_SNAPSHOT_WITHOUT_DECLARED_SCOPE"
        ),
        "declared_scope_sha256": canonical_sha256(declared_scope),
        "dependency_closure": dependency_closure,
        "repository": {
            "root": repository["root"],
            "worktree": repository["root"],
            "branch": repository.get("branch"),
            "commit": repository["head"],
            "working_state_sha256": repository.get("working_state_sha256"),
            "repository_state_sha256": repository.get("repository_state_sha256"),
            "changed_paths": list(repository.get("changed_paths") or []),
        },
        "work": {
            "work_id": str(work_record.get("record_id")),
            "status": str(work_record.get("status")),
            "package_path": binding.get("path"),
            "package_version": binding.get("package_version"),
            "package_digest": binding.get("sha256"),
            "objective": (
                package.get("goal")
                or work_payload.get("summary")
                or work_payload.get("title")
                or str(work_record.get("record_id") or "Unpackaged operator-authorized work")
            ),
            "authority_role": session_payload.get("session_role"),
            "recommended_model": package.get("recommended_model"),
            "uncertainty_class": package.get("uncertainty_class"),
        },
        "session": {
            "session_id": str(session_record.get("record_id")),
            "status": str(session_record.get("status")),
            "authority_role": session_payload.get("session_role"),
            "current_codex_id": session_payload.get("current_codex_id"),
            "epoch": epoch,
        },
        "scope": declared_scope,
        "output_policy": output_policy,
        "execution_budget": execution_budget,
        "progress": {
            "completed_steps": list(session_payload.get("completed_steps") or []),
            "current_stage": session_payload.get("current_stage"),
            "current_step": session_payload.get("current_step"),
            "interrupted_step": session_payload.get("interrupted_during"),
            "next_action": session_payload.get("next_action"),
            "next_command": session_payload.get("next_command"),
            "unresolved_blockers": [
                value
                for value in [session_payload.get("blocked_reason"), *(session_payload.get("blocked_by_work_ids") or [])]
                if value
            ],
            "resumed_from": (
                dict(session_payload.get("resumed_from") or {})
                if isinstance(session_payload.get("resumed_from"), Mapping)
                else None
            ),
        },
        "validation": {
            "tests_completed": [
                dict(row)
                for row in (session_payload.get("validation_events") or [])
                if isinstance(row, Mapping) and str(row.get("status") or "").upper() == "PASSED"
            ],
            "validation_events": [
                dict(row) for row in (session_payload.get("validation_events") or []) if isinstance(row, Mapping)
            ],
            "unresolved_gaps": list(session_payload.get("validation_gaps") or []),
            "decision_impact": session_payload.get("validation_decision_impact"),
            "last_successful_command": session_payload.get("last_successful_command"),
            "evidence_produced": list(references.get("evidence_assessment_ids") or []),
            "artifact_paths": list(references.get("artifact_paths") or []),
            "report_paths": list(references.get("report_paths") or []),
            "evidence_artifacts": evidence_artifacts,
            "last_failed_command": session_payload.get("last_failed_command"),
            "delivery_commit": delivery_commit_value(dict(session_payload.get("delivery_evidence") or {})) or None,
            "review_state": session_payload.get("review_verdict"),
            "qualification_records": list(references.get("qualification_record_ids") or []),
        },
        "resource_accounting": wra_continuation_capsule(session_record),
        "authoritative_review_binding": retained_review_binding,
        "context_health": health,
        "authority_documents": authority_documents,
        "critical_files": critical_files,
        "source_bindings": {
            "bootstrap_record_id": str(bootstrap_record.get("record_id")) if bootstrap_record else session_payload.get("bootstrap_record_id"),
            "work_record_sha256": work_record.get("record_sha256"),
            "work_execution_ref": work_execution_ref(work_record),
            "session_context_sha256": session_context_sha256(session_record),
            "package_sha256": binding.get("sha256"),
            "authoritative_review": retained_review_binding,
        },
    }
    # ``generated_at_utc`` remains outside the v1 identity contract for
    # compatibility, so derive it from immutable Session creation time rather
    # than mutable revision time.  Regenerating identical receipt material now
    # produces byte-identical v1 output instead of replacing one identity with
    # a different seal.
    generated_at_utc = str(session_record.get("created_at_utc") or session_record.get("updated_at_utc"))
    receipt_id = stable_id(CONTEXT_RECEIPT_ID_PREFIX, material)
    receipt = {
        **material,
        "receipt_id": receipt_id,
        "generated_at_utc": generated_at_utc,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def context_receipt_path(state_root: str | Path | None, session_id: str, receipt_id: str) -> Path:
    return actor_state_root(state_root) / CONTEXT_RECEIPT_DIRECTORY / session_id / f"{receipt_id}.json"


def write_context_receipt(
    receipt: Mapping[str, Any],
    *,
    state_root: str | Path | None = None,
    output: str | Path | None = None,
) -> Path:
    errors = verify_receipt_seal(receipt)
    if errors:
        raise ActorError(
            "refusing to persist an invalid Context Receipt",
            code="context_receipt_invalid",
            problems=errors,
        )
    target = Path(output).expanduser().resolve() if output else context_receipt_path(
        state_root, str(dict(receipt.get("session") or {}).get("session_id")), str(receipt.get("receipt_id"))
    )
    expected_bytes = pretty_json(dict(receipt)).encode("utf-8")
    lock_path = target.with_name(f".{target.name}.lock")
    with advisory_lock(lock_path):
        if target.exists():
            if not target.is_file():
                raise ActorError(
                    "Context Receipt target exists but is not a file",
                    code="context_receipt_immutable_conflict",
                    problems=[str(target)],
                )
            existing = target.read_bytes()
            if existing == expected_bytes:
                return target
            raise ActorError(
                "Context Receipt identity already exists with different sealed bytes",
                code="context_receipt_immutable_conflict",
                problems=[str(target)],
                related_records=[str(receipt.get("receipt_id") or "")],
            )
        return atomic_write_json(target, dict(receipt))


def verify_receipt_seal(receipt: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if receipt.get("schema_version") != CONTEXT_RECEIPT_SCHEMA_VERSION:
        errors.append("unsupported Context Receipt schema_version")
    expected = str(receipt.get("receipt_sha256") or "")
    material = dict(receipt)
    material.pop("receipt_sha256", None)
    actual = canonical_sha256(material)
    if expected != actual:
        errors.append("Context Receipt digest mismatch")
    expected_id = stable_id(CONTEXT_RECEIPT_ID_PREFIX, {
        key: value for key, value in material.items() if key not in {"receipt_id", "generated_at_utc"}
    })
    if str(receipt.get("receipt_id") or "") != expected_id:
        errors.append("Context Receipt ID does not match its material payload")
    return errors


def load_context_receipt(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ActorError("Context Receipt must be an object", code="schema_record_malformed")
    require_explicit_schema_version(
        raw, expected=CONTEXT_RECEIPT_SCHEMA_VERSION, label="Context Receipt"
    )
    errors = verify_receipt_seal(raw)
    if errors:
        code = (
            "context_receipt_digest_mismatch"
            if any("digest mismatch" in error for error in errors)
            else "schema_record_malformed"
        )
        raise ActorError("invalid Context Receipt", code=code, problems=errors)
    return dict(raw)



def _git_changed_paths(repo_root: Path, old_commit: str, new_commit: str) -> tuple[list[str], str | None]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--name-only", f"{old_commit}..{new_commit}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return [], result.stderr.strip() or "git diff failed"
    return [row.strip() for row in result.stdout.splitlines() if row.strip()], None


def validate_context_receipt(
    receipt: Mapping[str, Any],
    *,
    work_record: Mapping[str, Any],
    session_record: Mapping[str, Any],
    repo_root: str | Path | None = None,
    state_root: str | Path | None = None,
    historical_repository: bool = False,
) -> dict[str, Any]:
    seal_errors = verify_receipt_seal(receipt)
    compatibility_errors = [
        value for value in seal_errors if "unsupported Context Receipt schema_version" in value
    ]
    integrity_errors = [value for value in seal_errors if value not in compatibility_errors]
    staleness_errors: list[str] = []
    errors = staleness_errors
    warnings: list[str] = []
    session_payload = dict(session_record.get("payload") or {})
    work_payload = dict(work_record.get("payload") or {})
    session_repository = dict(session_payload.get("repository") or {})
    if historical_repository:
        recorded_root = str(session_repository.get("repo_root") or "").strip()
        if not recorded_root:
            integrity_errors.append("terminal Session has no recorded repository root")
        repository = {
            "root": recorded_root,
            "branch": session_repository.get("branch"),
            "head": session_repository.get("head"),
            "working_state_sha256": session_repository.get("working_state_sha256"),
            "changed_paths": list(session_repository.get("changed_paths") or []),
        }
    else:
        repository = git_repository_identity(repo_root or session_repository.get("repo_root"))
    receipt_repo = dict(receipt.get("repository") or {})
    receipt_work = dict(receipt.get("work") or {})
    receipt_session = dict(receipt.get("session") or {})
    source = dict(receipt.get("source_bindings") or {})

    if Path(str(receipt_repo.get("root") or "/")).resolve() != Path(repository["root"]).resolve():
        errors.append("Context Receipt belongs to another repository")
    if receipt_work.get("work_id") != work_record.get("record_id"):
        errors.append("Context Receipt belongs to another Work Item")
    if receipt_session.get("session_id") != session_record.get("record_id"):
        errors.append("Context Receipt belongs to another Session")
    if receipt_session.get("authority_role") != session_payload.get("session_role"):
        errors.append("Context Receipt authority role changed")
    current_binding = dict(session_payload.get("work_package_binding") or work_payload.get("work_package_binding") or {})
    if receipt_work.get("package_digest") != current_binding.get("sha256"):
        errors.append("Context Receipt Work Package digest is stale")
    claim = str(receipt.get("currentness_claim") or "")
    if claim not in {
        "",
        "CONTEXT_CURRENT_TO_DECLARED_SCOPE",
        "CONTEXT_SNAPSHOT_WITHOUT_DECLARED_SCOPE",
    }:
        compatibility_errors.append("Context Receipt currentness claim is unsupported")
    if claim == "CONTEXT_SNAPSHOT_WITHOUT_DECLARED_SCOPE":
        warnings.append(
            "Context Receipt has no Work Package scope; it preserves recovery provenance without claiming managed acceptance."
        )
    receipt_scope = dict(receipt.get("scope") or {})
    observed_scope_digest = str(receipt.get("declared_scope_sha256") or "")
    if observed_scope_digest and observed_scope_digest != canonical_sha256(receipt_scope):
        integrity_errors.append("Context Receipt declared scope digest mismatch")
    retained_closure = receipt.get("dependency_closure")
    if retained_closure is not None:
        if not isinstance(retained_closure, Mapping):
            integrity_errors.append("Context Receipt dependency closure is malformed")
        else:
            closure_value = dict(retained_closure)
            observed_closure_digest = str(closure_value.pop("closure_sha256", "") or "")
            if not observed_closure_digest or observed_closure_digest != canonical_sha256(closure_value):
                integrity_errors.append("Context Receipt dependency closure digest mismatch")
            current_closure = _dependency_closure(work_record, state_root=state_root)
            if dict(retained_closure) != current_closure:
                staleness_errors.append("Context Receipt dependency closure changed")
    elif dict(work_payload).get("dependency_work_ids"):
        warnings.append("legacy Context Receipt has no dependency closure")
    expected_work_execution_ref = work_execution_ref(work_record)
    retained_work_execution_ref = source.get("work_execution_ref")
    if isinstance(retained_work_execution_ref, Mapping):
        if dict(retained_work_execution_ref) != expected_work_execution_ref:
            (warnings if historical_repository else errors).append(
                "Context Receipt Work execution material changed after the terminal Session snapshot"
            )
        elif source.get("work_record_sha256") != work_record.get("record_sha256"):
            warnings.append(
                "Work record revision changed only outside execution-authority material"
            )
    elif source.get("work_record_sha256") != work_record.get("record_sha256"):
        errors.append("legacy Context Receipt Work record changed")
    else:
        warnings.append(
            "Context Receipt uses legacy whole-record Work freshness; refresh to material work_execution_ref"
        )
    if source.get("session_context_sha256") != session_context_sha256(session_record):
        (warnings if historical_repository else errors).append(
            "Context Receipt Session continuation state changed after the terminal snapshot"
        )

    receipt_review_binding = (
        dict(receipt.get("authoritative_review_binding") or {})
        if isinstance(receipt.get("authoritative_review_binding"), Mapping)
        else None
    )
    session_review_binding = (
        dict(session_payload.get("authoritative_review_binding") or {})
        if isinstance(session_payload.get("authoritative_review_binding"), Mapping)
        else None
    )
    if receipt_review_binding != session_review_binding:
        errors.append("Context Receipt authoritative review binding differs from its Session")
    if dict(source.get("authoritative_review") or {}) != dict(receipt_review_binding or {}):
        errors.append("Context Receipt authoritative review source binding is inconsistent")
    if state_root is not None:
        from tools.actors.work_manager import authoritative_review_binding_for_work

        current_review_binding = authoritative_review_binding_for_work(
            work_record,
            state_root=state_root,
            read_only=True,
        )
        current_verdict = str(dict(current_review_binding or {}).get("verdict") or "")
        requires_review_binding = (
            str(session_record.get("status") or "")
            not in {"COMPLETED", "DISCONTINUED", "OBSOLETE", "SUPERSEDED"}
            and str(session_payload.get("session_role") or "") == "IMPLEMENTATION_AUTHORITY"
            and current_verdict in {"CHANGES_REQUIRED", "REJECT", "BLOCKED"}
        )
        if requires_review_binding and not receipt_review_binding:
            errors.append("Context Receipt repair authorization has no authoritative review binding")
        if receipt_review_binding and receipt_review_binding != current_review_binding:
            errors.append("Context Receipt authoritative review changed")

    receipt_bootstrap_id = str(source.get("bootstrap_record_id") or "")
    current_bootstrap_id = str(session_payload.get("bootstrap_record_id") or "")
    bootstrap_history = [
        str(value)
        for value in (session_payload.get("bootstrap_history") or [])
        if str(value).strip()
    ]
    if not receipt_bootstrap_id:
        errors.append("Context Receipt has no Bootstrap source binding")
    elif receipt_bootstrap_id != current_bootstrap_id and receipt_bootstrap_id not in bootstrap_history:
        errors.append("Context Receipt Bootstrap is not the Session current or retained historical Bootstrap")
    elif state_root is None:
        warnings.append(
            "Context Receipt Bootstrap identity was checked against the Session binding "
            "but not globally resolved because state_root was not supplied"
        )
    else:
        store_obj = bootstrap_store(state_root)
        try:
            store_obj.get_read_only(receipt_bootstrap_id)
        except ActorError:
            try:
                history_rows = store_obj.history_read_only(receipt_bootstrap_id)
                if not history_rows:
                    raise ActorError("Bootstrap immutable history is empty")
                latest_revision = int(history_rows[-1].get("revision") or 0)
                store_obj.get_revision_read_only(receipt_bootstrap_id, latest_revision)
            except ActorError as exc:
                errors.append(
                    f"Context Receipt Bootstrap source is not resolvable: {receipt_bootstrap_id}: {exc}"
                )

    repo_path = Path(repository["root"] or "/")
    repository_files = [
        *(receipt.get("authority_documents") or []),
        *(receipt.get("critical_files") or []),
    ]
    if historical_repository:
        if repository_files:
            warnings.append(
                "terminal Session repository files are sealed historical provenance; "
                "live file currentness was not revalidated"
            )
    else:
        for row in repository_files:
            value = dict(row)
            path = repo_path / str(value.get("path") or "")
            if value.get("exists") and not path.is_file():
                errors.append(f"required Context Receipt file disappeared: {value.get('path')}")
            elif path.is_file() and value.get("sha256") and file_sha256(path) != value.get("sha256"):
                errors.append(f"Context Receipt file changed: {value.get('path')}")

    for row in dict(receipt.get("validation") or {}).get("evidence_artifacts") or []:
        value = dict(row) if isinstance(row, Mapping) else {}
        if value.get("schema_version") != CONTEXT_EVIDENCE_ARTIFACT_SCHEMA_VERSION:
            integrity_errors.append("Context Receipt evidence artifact has unsupported schema_version")
            continue
        resolved = Path(str(value.get("resolved_path") or ""))
        if value.get("exists"):
            if not resolved.is_file():
                errors.append(f"Context Receipt evidence artifact disappeared: {value.get('path')}")
            elif not value.get("sha256") or file_sha256(resolved) != value.get("sha256"):
                errors.append(f"Context Receipt evidence artifact changed: {value.get('path')}")
        else:
            warnings.append(
                f"Context Receipt retains an unresolved legacy evidence reference: {value.get('path')}"
            )

    old_commit = str(receipt_repo.get("commit") or "")
    new_commit = str(repository.get("head") or "")
    head_changed = False if historical_repository else bool(old_commit and old_commit != new_commit)
    branch_changed = False if historical_repository else (
        str(receipt_repo.get("branch") or "") != str(repository.get("branch") or "")
    )
    working_state_changed = False if historical_repository else (
        str(receipt_repo.get("working_state_sha256") or "")
        != str(repository.get("working_state_sha256") or "")
    )
    changed_path_set: set[str] = set()
    if head_changed:
        committed_paths, diff_error = _git_changed_paths(repo_path, old_commit, new_commit)
        if diff_error:
            errors.append(f"cannot classify Context Receipt HEAD drift: {diff_error}")
        else:
            changed_path_set.update(committed_paths)
    if working_state_changed:
        changed_path_set.update(str(path) for path in (receipt_repo.get("changed_paths") or []) if path)
        changed_path_set.update(str(path) for path in (repository.get("changed_paths") or []) if path)
    changed_paths = sorted(changed_path_set)

    if historical_repository:
        warnings.append(
            "terminal Session repository currentness was not revalidated; "
            "the recorded repository binding remains immutable provenance"
        )

    if branch_changed:
        errors.append("Context Receipt repository branch changed")

    if (head_changed or working_state_changed) and not current_binding:
        staleness_errors.append(
            "repository changed after an unscoped Context Receipt; path relevance cannot be classified"
        )
    elif head_changed or working_state_changed:
        package_verification = verify_work_package_binding(
            current_binding,
            repo_root=repo_path,
            expected_work_id=str(work_record.get("record_id")),
            state_root=state_root,
        )
        package = dict(package_verification.get("package") or {})
        read_contract = effective_read_contract(package) if package else {}
        exact_or_glob_patterns = [
            str(current_binding.get("path") or ""),
            *(package.get("owned_files") or []),
            *(read_contract.get("entry_files") or []),
            *(read_contract.get("required_documents") or []),
        ]
        search_roots = [str(value).strip("/") for value in (read_contract.get("search_roots") or []) if str(value).strip("/")]
        excluded_roots = [str(value).strip("/") for value in (read_contract.get("excluded_roots") or []) if str(value).strip("/")]

        def within_root(path: str, root: str) -> bool:
            normalized = PurePosixPath(path).as_posix().strip("/")
            return normalized == root or normalized.startswith(root + "/")

        relevant = [
            path
            for path in changed_paths
            if not any(within_root(path, root) for root in excluded_roots)
            and (
                any(pattern and package_path_matches(path, pattern, repo_root=repo_path) for pattern in exact_or_glob_patterns)
                or any(within_root(path, root) for root in search_roots)
            )
        ]
        if relevant:
            errors.append("Context Receipt relevant repository paths changed: " + ", ".join(sorted(relevant)))
        elif working_state_changed and not changed_paths:
            errors.append("Context Receipt working-tree/index digest changed but changed paths could not be classified")
        else:
            warnings.append(
                "repository changed only outside the receipt's declared working set; refresh is recommended but continuation is allowed"
            )
    status = (
        "INVALID" if integrity_errors
        else "INCOMPATIBLE" if compatibility_errors
        else "STALE" if staleness_errors
        else "CURRENT"
    )
    combined_errors = [*integrity_errors, *compatibility_errors, *staleness_errors]
    return {
        "status": status,
        "integrity_valid": not integrity_errors,
        "compatible": not compatibility_errors,
        "current": not staleness_errors,
        "currentness_claim": claim or "CONTEXT_CURRENT_TO_DECLARED_SCOPE",
        "valid": status == "CURRENT",
        "integrity_errors": integrity_errors,
        "compatibility_errors": compatibility_errors,
        "staleness_errors": staleness_errors,
        "errors": combined_errors,
        "warnings": warnings,
        "head_changed": head_changed,
        "branch_changed": branch_changed,
        "working_state_changed": working_state_changed,
        "changed_paths": changed_paths,
        "current_repository": repository,
    }


__all__ = [
    "CONTEXT_HEALTH_STATUSES",
    "CONTEXT_EVIDENCE_ARTIFACT_SCHEMA_VERSION",
    "CONTEXT_RECEIPT_SCHEMA_VERSION",
    "build_context_receipt",
    "context_receipt_path",
    "epoch_for_session",
    "normalize_observed_context_health",
    "project_context_health",
    "session_context_sha256",
    "validate_context_receipt",
    "verify_receipt_seal",
    "write_context_receipt",
]
