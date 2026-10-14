"""Authority and Knowledge Librarian preflight contracts for EIEL actors.

The actor managers retain authority; they never create it.  This module binds
an exact operator/work-package/active-goal authorization to the exact Knowledge
Librarian preflight receipt used before substantive work.  It delegates catalog
resolution to the existing Knowledge Librarian and verifies retained receipts,
source digests, and file bindings on every fresh CLI process.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.actors.canonical import canonical_sha256
from tools.actors.common import (
    ActorError,
    atomic_write_json,
    file_sha256,
    read_json,
    require_explicit_schema_version,
)
from tools.actors.instruction_binding import bind_operator_instruction, instruction_is_authoritative
from tools.evidence.librarian import wp4

AUTHORITY_CONTEXT_SCHEMA_VERSION = "eiel_mission_authority_context_v1"
PREFLIGHT_RECEIPT_SCHEMA_VERSION = "eiel_evidence_preflight_receipt_v1"
PREFLIGHT_BINDING_SCHEMA_VERSION = "eiel_actor_preflight_binding_v1"
AUTHORIZATION_BASES = (
    "operator_instruction",
    "approved_work_package",
    "active_goal",
    "mechanical_preflight_exemption",
)
HIGHER_AUTHORITY_BASES = {"operator_instruction", "approved_work_package"}
DEFAULT_SOURCE_ROOT = Path("tools/evidence/librarian/catalog_source/00-catalog")
DOCUMENT_ID_RE = re.compile(r"^EIEL-DOC-\d{6}$")


class AuthorityError(ActorError):
    """Fail-closed authority/preflight error."""


def empty_authority_context() -> dict[str, Any]:
    return {
        "schema_version": AUTHORITY_CONTEXT_SCHEMA_VERSION,
        "authorization_basis": None,
        "operator_instruction_ref": None,
        "operator_instruction_binding": None,
        "approved_work_package": None,
        "active_goal": None,
        "preflight": None,
        "preflight_fail_closed_acknowledged": False,
        "preflight_exemption": None,
        "repository_fallback_reason": None,
    }


def _resolve_repo_path(path: str | Path, *, repo_root: str | Path) -> Path:
    source = Path(path).expanduser()
    if source.is_absolute():
        return source.resolve()
    return (Path(repo_root).expanduser().resolve() / source).resolve()


def _file_binding(path: str | Path, *, kind: str, repo_root: str | Path) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    resolved = _resolve_repo_path(path, repo_root=repo)
    if not resolved.is_file():
        raise AuthorityError(
            f"{kind} does not exist: {resolved}",
            code=f"{kind.replace(' ', '_')}_missing",
            problems=[f"The declared {kind} cannot be verified."],
        )
    try:
        stored_path = resolved.relative_to(repo).as_posix()
        path_kind = "repository_relative"
    except ValueError:
        stored_path = str(resolved)
        path_kind = "absolute"
    return {
        "path": stored_path,
        "path_kind": path_kind,
        "sha256": file_sha256(resolved),
    }


def _verify_file_binding(
    binding: Mapping[str, Any] | None,
    *,
    kind: str,
    repo_root: str | Path,
) -> list[str]:
    if not isinstance(binding, Mapping):
        return [f"{kind} binding is missing"]
    raw_path = str(binding.get("path") or "").strip()
    expected = str(binding.get("sha256") or "").strip().lower()
    if not raw_path:
        return [f"{kind}.path is missing"]
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        return [f"{kind}.sha256 is invalid"]
    resolved = _resolve_repo_path(raw_path, repo_root=repo_root)
    if not resolved.is_file():
        return [f"{kind} no longer exists: {resolved}"]
    actual = file_sha256(resolved)
    if actual != expected:
        return [f"{kind} hash changed: expected {expected}, observed {actual}"]
    return []


def validate_preflight_receipt(receipt: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if receipt.get("schema_version") != PREFLIGHT_RECEIPT_SCHEMA_VERSION:
        errors.append("unsupported Knowledge Librarian preflight schema_version")
    for field in ("topic", "catalog_build_id", "source_digest", "authority_resolution_state"):
        if not isinstance(receipt.get(field), str) or not str(receipt.get(field)).strip():
            errors.append(f"preflight.{field} must be a non-empty string")
    authority_filter = receipt.get("authority_domain_filter")
    if authority_filter is not None and not isinstance(authority_filter, str):
        errors.append("preflight.authority_domain_filter must be a string or null")
    for field in (
        "authority_consulted",
        "evidence_ids",
        "artifact_ids",
        "raw_asset_ids",
        "claims_consulted",
        "document_ids",
        "limitations_reviewed",
        "unresolved_authority_gaps",
        "applicable_dossiers",
    ):
        if not isinstance(receipt.get(field), list):
            errors.append(f"preflight.{field} must be a list")
    if not isinstance(receipt.get("fail_closed"), bool):
        errors.append("preflight.fail_closed must be boolean")
    if not isinstance(receipt.get("fallback_repository_search_allowed"), bool):
        errors.append("preflight.fallback_repository_search_allowed must be boolean")
    for document_id in receipt.get("document_ids") or []:
        if not DOCUMENT_ID_RE.match(str(document_id)):
            errors.append(f"invalid Knowledge Librarian document ID: {document_id}")
    return errors


def load_preflight_receipt(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Load one retained Knowledge Librarian receipt without version inference."""

    require_explicit_schema_version(
        raw,
        expected=PREFLIGHT_RECEIPT_SCHEMA_VERSION,
        label="Knowledge Librarian preflight receipt",
    )
    receipt = dict(raw)
    errors = validate_preflight_receipt(receipt)
    if errors:
        raise AuthorityError(
            "Knowledge Librarian preflight receipt is malformed",
            code="schema_record_malformed",
            problems=errors,
        )
    return receipt


def load_preflight_binding(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Load the compact durable binding embedded in an authority context."""

    require_explicit_schema_version(
        raw,
        expected=PREFLIGHT_BINDING_SCHEMA_VERSION,
        label="actor preflight binding",
    )
    binding = dict(raw)
    errors: list[str] = []
    for field in (
        "receipt_sha256",
        "topic",
        "catalog_build_id",
        "maintained_source_digest",
        "authority_resolution_state",
    ):
        if not isinstance(binding.get(field), str) or not str(binding.get(field)).strip():
            errors.append(f"preflight binding {field} must be a non-empty string")
    for field in (
        "authority_assertion_ids",
        "authority_domains",
        "evidence_ids",
        "artifact_ids",
        "raw_asset_ids",
        "document_ids",
        "claim_ids",
        "limitation_ids",
        "unknown_authority_gaps",
        "applicable_dossiers",
    ):
        if not isinstance(binding.get(field), list):
            errors.append(f"preflight binding {field} must be a list")
    if not isinstance(binding.get("fail_closed"), bool):
        errors.append("preflight binding fail_closed must be boolean")
    if not isinstance(binding.get("fallback_repository_search_allowed"), bool):
        errors.append("preflight binding fallback_repository_search_allowed must be boolean")
    expected = str(binding.get("receipt_sha256") or "").lower()
    if expected and not re.fullmatch(r"[0-9a-f]{64}", expected):
        errors.append("preflight binding receipt_sha256 is invalid")
    if errors:
        raise AuthorityError(
            "actor preflight binding is malformed",
            code="schema_record_malformed",
            problems=errors,
        )
    return binding


def compact_preflight_receipt(
    receipt: Mapping[str, Any],
    *,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    errors = validate_preflight_receipt(receipt)
    if errors:
        raise AuthorityError(
            "Knowledge Librarian preflight receipt is invalid",
            code="preflight_receipt_invalid",
            problems=errors,
        )
    receipt_payload = dict(receipt)
    authorities = [row for row in receipt_payload.get("authority_consulted") or [] if isinstance(row, Mapping)]
    return {
        "schema_version": PREFLIGHT_BINDING_SCHEMA_VERSION,
        "receipt_sha256": canonical_sha256(receipt_payload),
        "receipt_path": str(Path(receipt_path).expanduser().resolve()) if receipt_path else None,
        "topic": receipt_payload["topic"],
        "collection": receipt_payload.get("collection", "all"),
        "authority_domain_filter": receipt_payload.get("authority_domain_filter"),
        "catalog_build_id": receipt_payload["catalog_build_id"],
        "maintained_source_digest": receipt_payload["source_digest"],
        "authority_resolution_state": receipt_payload["authority_resolution_state"],
        "fail_closed": bool(receipt_payload["fail_closed"]),
        "fallback_repository_search_allowed": bool(receipt_payload["fallback_repository_search_allowed"]),
        "authority_assertion_ids": [
            str(row.get("authority_assertion_id")) for row in authorities if row.get("authority_assertion_id")
        ],
        "authority_domains": sorted({str(row.get("domain")) for row in authorities if row.get("domain")}),
        "evidence_ids": [str(item) for item in receipt_payload.get("evidence_ids") or []],
        "artifact_ids": [str(item) for item in receipt_payload.get("artifact_ids") or []],
        "raw_asset_ids": [str(item) for item in receipt_payload.get("raw_asset_ids") or []],
        "document_ids": [str(item) for item in receipt_payload.get("document_ids") or []],
        "claim_ids": [str(item) for item in receipt_payload.get("claims_consulted") or []],
        "limitation_ids": [str(item) for item in receipt_payload.get("limitations_reviewed") or []],
        "unknown_authority_gaps": [str(item) for item in receipt_payload.get("unresolved_authority_gaps") or []],
        "applicable_dossiers": [str(item) for item in receipt_payload.get("applicable_dossiers") or []],
        "next_step": receipt_payload.get("next_step"),
    }


def run_librarian_preflight(
    *,
    repo_root: str | Path,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    topic: str,
    collection: str = "all",
    limit: int = 20,
    authority_domain: str | None = None,
) -> dict[str, Any]:
    """Run the existing local Knowledge Librarian preflight implementation."""

    repo = Path(repo_root).expanduser().resolve()
    source = _resolve_repo_path(source_root, repo_root=repo)
    if not repo.is_dir():
        raise AuthorityError(f"repository root does not exist: {repo}", code="repository_root_missing")
    if not source.is_dir():
        raise AuthorityError(
            f"Knowledge Librarian source root does not exist: {source}",
            code="librarian_source_root_missing",
        )
    try:
        catalog = wp4.build_catalog(source, repo)
        receipt = wp4.evidence_preflight(
            catalog,
            str(topic),
            limit=int(limit),
            collection=str(collection),
            authority_domain=str(authority_domain).strip() if authority_domain else None,
        )
    except (wp4.LibrarianError, OSError) as exc:
        raise AuthorityError(
            "Knowledge Librarian preflight could not establish repository identity",
            code="librarian_preflight_failed",
            problems=[str(exc)],
            next_actions=[{
                "action": "run_from_repository",
                "description": "Run the preflight from the real Git checkout and retain its JSON receipt.",
                "command": (
                    "python -m tools.evidence.librarian preflight "
                    f"--topic {topic!r} --collection {collection} "
                    + (f"--authority-domain {authority_domain} " if authority_domain else "")
                    + "--source-root tools/evidence/librarian/catalog_source/00-catalog --json"
                ),
            }],
        ) from exc
    return dict(receipt)


def retain_preflight_receipt(
    receipt: Mapping[str, Any],
    *,
    state_root: str | Path,
) -> Path:
    errors = validate_preflight_receipt(receipt)
    if errors:
        raise AuthorityError(
            "cannot retain invalid Knowledge Librarian preflight receipt",
            code="preflight_receipt_invalid",
            problems=errors,
        )
    digest = canonical_sha256(dict(receipt))
    path = Path(state_root).expanduser().resolve() / "authority" / "preflights" / f"{digest}.json"
    if path.is_file():
        existing = read_json(path, name="retained preflight receipt")
        if canonical_sha256(existing) != digest:
            raise AuthorityError(f"retained preflight hash conflict: {path}", code="preflight_receipt_conflict")
        return path
    return atomic_write_json(path, dict(receipt))


def normalize_authority_context(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    context = empty_authority_context()
    if isinstance(raw, Mapping):
        if "schema_version" in raw:
            require_explicit_schema_version(
                raw,
                expected=AUTHORITY_CONTEXT_SCHEMA_VERSION,
                label="mission authority context",
            )
        context.update(dict(raw))
    context["schema_version"] = AUTHORITY_CONTEXT_SCHEMA_VERSION
    basis = context.get("authorization_basis")
    context["authorization_basis"] = str(basis).strip().lower() if basis else None
    for field in ("approved_work_package", "active_goal", "preflight", "preflight_exemption", "operator_instruction_binding"):
        value = context.get(field)
        context[field] = dict(value) if isinstance(value, Mapping) else None
    acknowledged = bool(
        context.get("preflight_fail_closed_acknowledged")
        or context.get("preflight_override_acknowledged")
    )
    context["preflight_fail_closed_acknowledged"] = acknowledged
    # Read the obsolete V1.1 spelling above, but never emit it again.
    context.pop("preflight_override_acknowledged", None)
    return context


def load_authority_context(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Load a persisted authority context while preserving fail-closed semantics."""

    require_explicit_schema_version(
        raw,
        expected=AUTHORITY_CONTEXT_SCHEMA_VERSION,
        label="mission authority context",
    )
    context = normalize_authority_context(raw)
    preflight = context.get("preflight")
    if preflight is not None:
        context["preflight"] = load_preflight_binding(preflight)
    return context


def build_authority_context(
    *,
    authorization_basis: str,
    repo_root: str | Path,
    state_root: str | Path | None = None,
    preflight_receipt: Mapping[str, Any] | None = None,
    preflight_receipt_path: str | Path | None = None,
    operator_instruction_ref: str | None = None,
    work_package_path: str | Path | None = None,
    active_goal_path: str | Path | None = None,
    preflight_fail_closed_acknowledged: bool = False,
    preflight_exemption_reason: str | None = None,
    repository_fallback_reason: str | None = None,
) -> dict[str, Any]:
    basis = str(authorization_basis).strip().lower()
    if basis not in AUTHORIZATION_BASES:
        raise AuthorityError(
            f"authorization_basis must be one of {', '.join(AUTHORIZATION_BASES)}",
            code="authorization_basis_invalid",
        )
    context = empty_authority_context()
    context["authorization_basis"] = basis
    instruction_binding = bind_operator_instruction(
        operator_instruction_ref,
        state_root=state_root,
        base_dir=repo_root,
        allow_inline_authority=(basis == "operator_instruction"),
    )
    context["operator_instruction_binding"] = instruction_binding
    if instruction_binding and instruction_binding.get("binding_kind") == "CONTENT_ADDRESSED_FILE":
        context["operator_instruction_ref"] = str(
            instruction_binding.get("source_path") or instruction_binding.get("state_root_locator") or ""
        ) or None
    elif instruction_binding:
        context["operator_instruction_ref"] = str(instruction_binding.get("label") or "").strip() or None
    else:
        context["operator_instruction_ref"] = None
    context["preflight_fail_closed_acknowledged"] = bool(preflight_fail_closed_acknowledged)
    context["repository_fallback_reason"] = str(repository_fallback_reason).strip() if repository_fallback_reason else None
    if work_package_path:
        context["approved_work_package"] = _file_binding(
            work_package_path, kind="approved work package", repo_root=repo_root
        )
    if active_goal_path:
        context["active_goal"] = _file_binding(active_goal_path, kind="active goal", repo_root=repo_root)
    if preflight_receipt is not None:
        context["preflight"] = compact_preflight_receipt(
            preflight_receipt,
            receipt_path=preflight_receipt_path,
        )
    if basis == "mechanical_preflight_exemption":
        context["preflight_exemption"] = {
            "classification": "mechanical_non_substantive",
            "reason": str(preflight_exemption_reason or "").strip(),
        }
    return normalize_authority_context(context)


def _current_catalog_identity(repo_root: str | Path, source_root: str | Path) -> tuple[str, str]:
    source = _resolve_repo_path(source_root, repo_root=repo_root)
    digest = wp4.maintained_source_digest(source)
    build_id = "eiel-build-wp4-" + hashlib.sha256(digest.encode()).hexdigest()[:20]
    return build_id, digest


def _verify_preflight_binding(
    preflight: Mapping[str, Any],
    *,
    repo_root: str | Path,
    source_root: str | Path,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if preflight.get("schema_version") != PREFLIGHT_BINDING_SCHEMA_VERSION:
        errors.append("authority_context.preflight has unsupported schema_version")
        return errors, warnings
    expected = str(preflight.get("receipt_sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        errors.append("authority_context.preflight.receipt_sha256 is invalid")
    receipt_path = preflight.get("receipt_path")
    if not receipt_path:
        errors.append("authority_context.preflight.receipt_path is required for durable verification")
    else:
        path = Path(str(receipt_path)).expanduser().resolve()
        if not path.is_file():
            errors.append(f"retained Knowledge Librarian preflight receipt is missing: {path}")
        else:
            try:
                receipt = read_json(path, name="retained Knowledge Librarian preflight receipt")
            except ActorError as exc:
                errors.append(str(exc))
            else:
                receipt_errors = validate_preflight_receipt(receipt)
                errors.extend(receipt_errors)
                actual = canonical_sha256(receipt)
                if expected and actual != expected:
                    errors.append(
                        f"retained preflight receipt hash changed: expected {expected}, observed {actual}"
                    )
                if not receipt_errors:
                    compact = compact_preflight_receipt(receipt, receipt_path=path)
                    for field in (
                        "topic",
                        "collection",
                        "authority_domain_filter",
                        "catalog_build_id",
                        "maintained_source_digest",
                        "authority_resolution_state",
                        "fail_closed",
                        "fallback_repository_search_allowed",
                        "authority_assertion_ids",
                        "authority_domains",
                        "evidence_ids",
                        "artifact_ids",
                        "raw_asset_ids",
                        "document_ids",
                        "claim_ids",
                        "limitation_ids",
                        "unknown_authority_gaps",
                        "applicable_dossiers",
                        "next_step",
                    ):
                        if compact.get(field) != preflight.get(field):
                            errors.append(f"authority_context.preflight.{field} does not match retained receipt")
    try:
        current_build, current_digest = _current_catalog_identity(repo_root, source_root)
    except (OSError, wp4.LibrarianError) as exc:
        errors.append(f"current Knowledge Librarian catalog identity cannot be verified: {exc}")
    else:
        if str(preflight.get("maintained_source_digest") or "") != current_digest:
            errors.append(
                "Knowledge Librarian maintained source changed after preflight; rerun preflight "
                f"(bound {preflight.get('maintained_source_digest')}, current {current_digest})"
            )
        if str(preflight.get("catalog_build_id") or "") != current_build:
            errors.append(
                "Knowledge Librarian catalog build changed after preflight; rerun preflight "
                f"(bound {preflight.get('catalog_build_id')}, current {current_build})"
            )
    return errors, warnings


def validate_authority_context(
    context: Mapping[str, Any] | None,
    *,
    mission_type: str,
    resource_demand: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    verify_bindings: bool = True,
) -> dict[str, Any]:
    normalized = normalize_authority_context(context)
    errors: list[str] = []
    warnings: list[str] = []
    missing_fields: list[str] = []
    basis = normalized.get("authorization_basis")
    repo = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()

    if basis not in AUTHORIZATION_BASES:
        errors.append("authority_context.authorization_basis is required")
        missing_fields.append("authority_context.authorization_basis")
    elif basis == "operator_instruction":
        if not str(normalized.get("operator_instruction_ref") or "").strip():
            errors.append("operator_instruction authorization requires operator_instruction_ref")
            missing_fields.append("authority_context.operator_instruction_ref")
        if not instruction_is_authoritative(normalized.get("operator_instruction_binding")):
            errors.append("operator_instruction authorization requires digest-bound instruction content")
            missing_fields.append("authority_context.operator_instruction_binding")
    elif basis == "approved_work_package":
        if not normalized.get("approved_work_package"):
            errors.append("approved_work_package authorization requires a bound work package")
            missing_fields.append("authority_context.approved_work_package")
    elif basis == "active_goal":
        if not normalized.get("active_goal"):
            errors.append("active_goal authorization requires a bound ACTIVE_GOAL.md")
            missing_fields.append("authority_context.active_goal")
        if not normalized.get("approved_work_package"):
            errors.append("active_goal authorization requires its approved work package binding")
            missing_fields.append("authority_context.approved_work_package")
    elif basis == "mechanical_preflight_exemption":
        exemption = normalized.get("preflight_exemption") or {}
        reason = str(exemption.get("reason") or "").strip()
        device = str(dict(resource_demand or {}).get("device") or "none").lower()
        if str(mission_type).upper() not in {"SMOKE", "CLOSURE"} or device not in {"none", "cpu"}:
            errors.append("mechanical preflight exemption is limited to non-scientific SMOKE or CLOSURE work")
        if not reason:
            errors.append("mechanical preflight exemption requires a reason")
            missing_fields.append("authority_context.preflight_exemption.reason")
        if not str(normalized.get("operator_instruction_ref") or "").strip():
            errors.append("mechanical preflight exemption requires an explicit operator instruction reference")
            missing_fields.append("authority_context.operator_instruction_ref")

    if verify_bindings:
        if normalized.get("approved_work_package"):
            errors.extend(
                _verify_file_binding(
                    normalized.get("approved_work_package"),
                    kind="approved work package",
                    repo_root=repo,
                )
            )
        if normalized.get("active_goal"):
            errors.extend(
                _verify_file_binding(
                    normalized.get("active_goal"), kind="active goal", repo_root=repo
                )
            )

    preflight = normalized.get("preflight")
    if basis != "mechanical_preflight_exemption":
        if not isinstance(preflight, Mapping):
            errors.append("substantive mission requires a retained Knowledge Librarian preflight")
            missing_fields.append("authority_context.preflight")
        else:
            if verify_bindings:
                preflight_errors, preflight_warnings = _verify_preflight_binding(
                    preflight,
                    repo_root=repo,
                    source_root=source_root,
                )
                errors.extend(preflight_errors)
                warnings.extend(preflight_warnings)
            if bool(preflight.get("fallback_repository_search_allowed")) and not str(
                normalized.get("repository_fallback_reason") or ""
            ).strip():
                warnings.append(
                    "preflight permits bounded repository fallback but no fallback reason is recorded"
                )
            if bool(preflight.get("fail_closed")):
                if (
                    basis in HIGHER_AUTHORITY_BASES
                    and normalized.get("preflight_fail_closed_acknowledged") is True
                ):
                    warnings.append(
                        "Knowledge Librarian preflight remains fail-closed; work proceeds only under the recorded higher-priority authorization and must preserve every listed authority gap"
                    )
                else:
                    errors.append(
                        "Knowledge Librarian preflight is fail-closed; narrow the authority domain/topic or explicitly acknowledge the operator/work-package authorization"
                    )
                    missing_fields.append("authority_context.preflight_fail_closed_acknowledged")

    return {
        "schema_version": "eiel_mission_authority_validation_v1",
        "valid": not errors,
        "ready_for_work": not errors,
        "authorization_basis": basis,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "missing_fields": sorted(set(missing_fields)),
        "preflight": dict(preflight) if isinstance(preflight, Mapping) else None,
    }


def evidence_preflight_markdown(binding: Mapping[str, Any] | None) -> str:
    row = dict(binding or {})
    authority = ", ".join(str(item) for item in row.get("authority_assertion_ids") or []) or "None recorded"
    evidence = ", ".join(str(item) for item in row.get("evidence_ids") or []) or "None recorded"
    documents = ", ".join(str(item) for item in row.get("document_ids") or []) or "None recorded"
    claims = ", ".join(str(item) for item in row.get("claim_ids") or []) or "None recorded"
    limitations = ", ".join(str(item) for item in row.get("limitation_ids") or []) or "None recorded"
    gaps = "; ".join(str(item) for item in row.get("unknown_authority_gaps") or []) or "None"
    return "\n".join(
        [
            "## Evidence Preflight",
            "",
            f"- Authority consulted: {authority}",
            f"- Evidence IDs: {evidence}",
            f"- Document IDs: {documents}",
            f"- Claims consulted: {claims}",
            f"- Limitations reviewed: {limitations}",
            f"- Unknown authority: {gaps}",
            f"- Catalog build ID: {row.get('catalog_build_id') or 'Not recorded'}",
            f"- Maintained source digest: {row.get('maintained_source_digest') or 'Not recorded'}",
        ]
    ) + "\n"


def _registered_document_ids(repo_root: str | Path) -> set[str]:
    source = _resolve_repo_path(DEFAULT_SOURCE_ROOT, repo_root=repo_root)
    objects, _ = wp4.load_source_accessions(source)
    return {str(row.get("document_id")) for row in objects.get("documents") or [] if row.get("document_id")}


def validate_librarian_document_ids(
    values: Sequence[Any],
    *,
    repo_root: str | Path | None = None,
    require_registered: bool = False,
) -> list[str]:
    errors = [
        f"invalid Knowledge Librarian document ID: {value}"
        for value in values
        if not DOCUMENT_ID_RE.match(str(value))
    ]
    if require_registered and repo_root is not None:
        try:
            registered = _registered_document_ids(repo_root)
        except (OSError, wp4.LibrarianError) as exc:
            errors.append(f"Knowledge Librarian document catalog cannot be verified: {exc}")
        else:
            for value in values:
                if DOCUMENT_ID_RE.match(str(value)) and str(value) not in registered:
                    errors.append(f"Knowledge Librarian document ID is not registered: {value}")
    return errors


def validate_managed_document_path(path: str | Path, *, repo_root: str | Path) -> list[str]:
    resolved = Path(path).expanduser().resolve()
    root = Path(repo_root).expanduser().resolve()
    docs_root = root / "docs"
    try:
        resolved.relative_to(docs_root)
    except ValueError:
        return []
    if resolved.name == "000_knowledge-catalog.html":
        return []
    if not re.match(r"^EIEL-DOC-\d{6}--", resolved.name):
        return [
            "managed docs/ path lacks a permanent Knowledge Librarian document ID: "
            + str(resolved)
        ]
    return []
