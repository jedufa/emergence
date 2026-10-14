"""Managed merge closeout and direct-Git reconciliation for runtime evidence.

The Work Manager calls this module only after accepted repository integration
has been durably recorded.  It consumes explicitly named evidence roots (or
already-registered artifact-path references that resolve to an exact
disposition manifest) and delegates deletion to the existing lifecycle prune
actuator.  It does not infer ownership by scanning a repository or ``configured runtime root``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any

from tools.actors.common import ActorError, canonical_sha256
from tools.actors.repository_truth import (
    ARTIFACT_REFERENCE_SCHEMA_VERSION,
    capture_storage_merge_closeout_receipt,
    load_repository_truth_artifact,
)

STORAGE_CLOSEOUT_REFERENCE_SCHEMA_VERSION = ARTIFACT_REFERENCE_SCHEMA_VERSION


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_text(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise ActorError(
            "cannot verify repository integration for storage closeout",
            code="storage_merge_closeout_git_verification_failed",
            problems=[str(result.stderr or "").strip() or "git command failed"],
        )
    return str(result.stdout or "").strip()


def _candidate_paths(payload: Mapping[str, Any], explicit: Sequence[str] | None) -> list[Path]:
    raw: list[str] = []
    raw.extend(str(value) for value in (explicit or []) if str(value).strip())
    raw.extend(
        str(value)
        for value in (payload.get("storage_evidence_roots") or [])
        if str(value).strip()
    )
    references = payload.get("references")
    if isinstance(references, Mapping):
        # Generic artifact references are often source files or reports.  They
        # become storage authority only when they already resolve to an exact
        # disposition manifest/root; ordinary paths are never interpreted as
        # deletion targets.
        for value in references.get("artifact_paths") or []:
            candidate = Path(str(value)).expanduser()
            if candidate.name == "artifact_disposition_manifest.json" or (
                candidate.is_dir()
                and (candidate / "artifact_disposition_manifest.json").is_file()
            ):
                raw.append(str(candidate))
    result: list[Path] = []
    seen: set[str] = set()
    for value in raw:
        path = Path(value).expanduser()
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _manifest_for_candidate(candidate: Path) -> tuple[Path, Path]:
    """Resolve only an explicitly named root or exact manifest file."""

    if candidate.is_symlink():
        raise RuntimeError("storage closeout candidate is a symlink")
    if candidate.is_file():
        if candidate.name != "artifact_disposition_manifest.json":
            raise RuntimeError(
                "storage closeout file candidate is not an exact disposition manifest"
            )
        manifest_path = candidate.resolve(strict=True)
        root = manifest_path.parent.resolve()
    elif candidate.is_dir():
        root = candidate.resolve()
        manifest_path = root / "artifact_disposition_manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise RuntimeError(
                "storage closeout root has no exact disposition manifest"
            )
    else:
        raise RuntimeError("storage closeout candidate is missing")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("storage closeout manifest is unsafe or missing")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("storage closeout manifest is unreadable") from exc
    if not isinstance(raw, Mapping):
        raise RuntimeError("storage closeout manifest is not an object")
    lifecycle = importlib.import_module("engine.campaigns.artifact_lifecycle")
    manifest = lifecycle.validate_artifact_disposition_manifest_v1(raw)
    bound_root = Path(str(manifest.get("runtime_root") or "")).expanduser().resolve()
    if bound_root != root or manifest_path.resolve().parent != root:
        raise RuntimeError("storage closeout manifest root identity differs")
    if manifest.get("terminal_status") != "completed":
        raise RuntimeError("storage closeout requires completed work")
    return root, manifest_path


def _durable_knowledge(
    *,
    root: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    references = payload.get("references")
    report_refs = (
        [str(value) for value in references.get("report_paths") or []]
        if isinstance(references, Mapping)
        else []
    )
    certificate_names = (
        "evidence_certificate.json",
        "evidence_closeout_receipt.json",
        "closeout_certificate.json",
    )
    report_names = ("report.md", "report.json", "closeout_report.md", "closeout_report.json")
    reproduction_names = (
        "run_passport.json",
        "reproduction_recipe.json",
        "reproduction_recipe.md",
        "artifact_lifecycle_contract.json",
    )

    def existing(names: Sequence[str]) -> list[str]:
        return [str(root / name) for name in names if (root / name).is_file()]

    certificates = existing(certificate_names)
    reports = [path for path in report_refs if Path(path).is_file()]
    reports.extend(existing(report_names))
    reproductions = existing(reproduction_names)
    return {
        "certificate_paths": sorted(set(certificates)),
        "report_paths": sorted(set(reports)),
        "reproduction_paths": sorted(set(reproductions)),
        "certificate_durable": bool(certificates),
        "report_durable": bool(reports),
        "reproduction_durable": bool(reproductions),
    }


def _root_result(
    *,
    root: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    lifecycle = importlib.import_module("engine.campaigns.artifact_lifecycle")
    prune_module = importlib.import_module("tools.scripts.eiel_runtime_storage_prune")
    categories = prune_module.CATEGORIES
    apply_prune_plan = prune_module.apply_prune_plan
    build_prune_plan = prune_module.build_prune_plan

    knowledge = _durable_knowledge(root=root, payload=payload)
    total_bytes = int(manifest.get("summary", {}).get("produced_bytes", 0))
    result: dict[str, Any] = {
        "run_root": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "status": "HELD",
        "bytes_before": total_bytes,
        "bytes_reclaimed": 0,
        "bytes_retained": total_bytes,
        "knowledge": knowledge,
    }
    if not all(
        knowledge[key]
        for key in ("certificate_durable", "report_durable", "reproduction_durable")
    ):
        result["reason"] = "durable certificate/report/reproduction set is incomplete"
        return result
    if lifecycle.RunStorageGuard.run_root_is_reserved(root):
        result["reason"] = "active reservation or unresolved writer protects the root"
        return result
    manifest_prune = manifest.get("prune")
    if not isinstance(manifest_prune, Mapping) or manifest_prune.get("automatic_prune_eligible") is not True:
        result["reason"] = "manifest is not automatically releasable"
        return result
    release_authority = manifest.get("release_authority")
    if isinstance(release_authority, Mapping) and release_authority.get("kind") in {
        "EVIDENCE_CERTIFICATE_GATE_CLOSURE",
        "EVIDENCE_CERTIFICATE_DIAGNOSTIC_CLOSEOUT",
    }:
        result["reason"] = "certificate-backed release requires exact close-gate context"
        return result
    plan = build_prune_plan(root, set(categories), manifest)
    selected_bytes = int(plan.get("selected_bytes") or 0)
    result["selected_bytes"] = selected_bytes
    if not plan.get("selected"):
        result.update({"status": "NOOP", "bytes_before": 0, "bytes_retained": int(result["bytes_before"])})
        return result
    deletion = apply_prune_plan(
        root,
        plan,
        disposition_manifest=manifest,
        current_disposition_manifest_path=manifest_path,
        _transaction_fence_held=True,
    )
    selected_by_path = {
        str(row.get("path")): row
        for row in plan.get("selected", [])
        if isinstance(row, Mapping)
    }
    recovered_paths = {str(path) for path in (deletion.get("recovered_released") or [])}
    absent_paths = {str(path) for path in (deletion.get("already_absent") or [])}
    reclaimed_rows = [
        row
        for row in (deletion.get("deleted_rows") or [])
        if str(row.get("path")) not in recovered_paths
    ]
    reclaimed = sum(int(row.get("bytes") or 0) for row in reclaimed_rows)
    already_released = sum(
        int(selected_by_path[path].get("bytes") or 0)
        for path in recovered_paths
        if path in selected_by_path
    )
    already_absent = sum(
        int(selected_by_path[path].get("bytes") or 0)
        for path in absent_paths
        if path in selected_by_path
    )
    current_selected_bytes = max(0, selected_bytes - already_released - already_absent)
    result.update(
        {
            "status": (
                "RELEASED"
                if reclaimed and not deletion.get("failures")
                else "PARTIAL"
                if deletion.get("failures")
                else "NOOP"
            ),
            "bytes_before": current_selected_bytes,
            "bytes_reclaimed": reclaimed,
            "bytes_already_released": already_released,
            "bytes_already_absent": already_absent,
            "bytes_retained": max(
                0,
                total_bytes - reclaimed - already_released - already_absent,
            ),
            "deleted": list(deletion.get("deleted") or []),
            "recovered_released": sorted(recovered_paths),
            "already_absent": sorted(absent_paths),
            "failures": list(deletion.get("failures") or []),
        }
    )
    return result


def closeout_merged_work_storage_v1(
    *,
    work_id: str,
    payload: Mapping[str, Any],
    integration_reference: Mapping[str, Any],
    repo_root: str | Path,
    state_root: str | Path | None,
    evidence_roots: Sequence[str] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Seal a managed integration's storage disposition and release its rows."""

    if payload.get("accepted") is not True:
        raise ActorError(
            "storage merge closeout requires accepted repository integration",
            code="storage_merge_closeout_not_accepted",
        )
    integration = load_repository_truth_artifact(
        integration_reference,
        state_root=state_root,
        expected_schema_version="eiel_repository_integration_receipt_v1",
    )
    if str(integration.get("work_id") or "") != str(work_id):
        raise ActorError(
            "storage merge closeout integration receipt belongs to another Work",
            code="storage_merge_closeout_work_mismatch",
        )
    repo = Path(repo_root).expanduser().resolve()
    lifecycle = importlib.import_module("engine.campaigns.artifact_lifecycle")
    current_head = _git_text(repo, "rev-parse", "HEAD").lower()
    integration_sha = str(integration.get("integration_commit") or "").lower()
    if not integration_sha or current_head != integration_sha:
        raise ActorError(
            "storage merge closeout requires the recorded integration commit to remain HEAD",
            code="storage_merge_closeout_head_mismatch",
            problems=[f"recorded integration: {integration_sha}", f"current HEAD: {current_head}"],
        )
    dirty = _git_text(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise ActorError(
            "storage merge closeout requires a clean integrated target checkout",
            code="storage_merge_closeout_dirty_repository",
            problems=[dirty[:4096]],
        )
    target_branch = _git_text(repo, "branch", "--show-current") or None
    if target_branch is None:
        raise ActorError(
            "storage merge closeout requires a named target branch",
            code="storage_merge_closeout_target_branch_missing",
        )
    roots = _candidate_paths(payload, evidence_roots)
    root_results: list[dict[str, Any]] = []
    for candidate in roots:
        try:
            root, manifest_path = _manifest_for_candidate(candidate)
            with lifecycle.run_artifact_transaction_fence_v1(root):
                manifest = lifecycle.validate_artifact_disposition_manifest_v1(
                    json.loads(manifest_path.read_text(encoding="utf-8"))
                )
                root_results.append(
                    _root_result(
                        root=root,
                        manifest_path=manifest_path,
                        manifest=manifest,
                        payload=payload,
                    )
                )
        except Exception as exc:  # exact root remains held and is reported
            root_results.append(
                {
                    "candidate": str(candidate),
                    "status": "HELD",
                    "bytes_before": 0,
                    "bytes_reclaimed": 0,
                    "bytes_retained": 0,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    bytes_before = sum(int(row.get("bytes_before") or 0) for row in root_results)
    bytes_reclaimed = sum(int(row.get("bytes_reclaimed") or 0) for row in root_results)
    bytes_retained = sum(int(row.get("bytes_retained") or 0) for row in root_results)
    unresolved = [
        str(row.get("reason"))
        for row in root_results
        if row.get("status") == "HELD" and row.get("reason")
    ]
    if not root_results:
        status = "NOOP"
    elif unresolved and bytes_reclaimed == 0:
        status = "HELD"
    elif unresolved:
        status = "PARTIAL"
    else:
        status = "RELEASED" if bytes_reclaimed else "NOOP"
    delivery_sha = str(integration.get("reviewed_commit") or integration_sha)
    acceptance = {
        "accepted": bool(payload.get("accepted")),
        "acceptance_scope": payload.get("acceptance_scope"),
        "completion_basis": payload.get("completion_basis"),
        "accepted_review_session_id": payload.get("accepted_review_session_id"),
    }
    knowledge = _durable_knowledge(
        root=Path(roots[0]).expanduser().resolve() if roots else repo,
        payload=payload,
    )
    body, reference = capture_storage_merge_closeout_receipt(
        state_root=state_root,
        work_id=work_id,
        source_branch=(dict(payload.get("work_package_binding") or {}).get("bound_branch")
                       if isinstance(payload.get("work_package_binding"), Mapping)
                       else None),
        source_worktree=(dict(payload.get("work_package_binding") or {}).get("bound_repo_root")
                         if isinstance(payload.get("work_package_binding"), Mapping)
                         else None),
        delivery_sha=delivery_sha,
        target_branch=target_branch,
        integration_sha=integration_sha,
        integration_relation=str(integration.get("relation") or "unknown"),
        acceptance_state=acceptance,
        certificate_report_reproduction=knowledge,
        evidence_roots=root_results,
        bytes_before=bytes_before,
        bytes_reclaimed=bytes_reclaimed,
        bytes_retained=bytes_retained,
        unresolved_unknown_policy_bytes=0,
        status=status,
        retained_reasons=unresolved,
        generated_at=generated_at,
    )
    return {
        "status": status,
        "receipt": reference,
        "receipt_artifact": body,
        "evidence_roots": root_results,
        "bytes_before": bytes_before,
        "bytes_reclaimed": bytes_reclaimed,
        "bytes_retained": bytes_retained,
        "unresolved": unresolved,
        "archive_policy": "DIRECT_HOT_RELEASE_ONLY_NO_HDD",
        "recorded_at_utc": body.get("recorded_at_utc", _now()),
    }


__all__ = ["closeout_merged_work_storage_v1"]
