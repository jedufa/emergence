"""Read-only compact projections for bounded Codex context openings.

This module owns no project state.  It derives a small presentation from the
full Workspace result and from files already named by the Work Package read
contract.  Full manager records remain authoritative and available through the
existing inspect/full-output routes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.actors.common import ActorError, EIEL_LAUNCHER, require_explicit_schema_version

COMPACT_WORKSPACE_SCHEMA_VERSION = "eiel_codex_workspace_compact_v1"
INITIAL_CONTEXT_TOTAL_BYTES_WARNING = 64 * 1024
INITIAL_CONTEXT_SINGLE_FILE_BYTES_WARNING = 32 * 1024
MAX_LARGEST_INITIAL_FILES = 5


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _drop_none(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _file_lines(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return None


def project_initial_context_metrics(
    initial_context: Mapping[str, Any] | None,
    *,
    repo_root: str | Path | None,
) -> dict[str, Any]:
    """Measure the exact initial files already named by a read contract.

    Metrics are advisory and read-only.  Paths are resolved beneath the bound
    repository; missing, non-file, or escaping paths are reported rather than
    guessed or followed.
    """

    contract = _mapping(initial_context)
    root = Path(repo_root or ".").expanduser().resolve()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    missing: list[str] = []
    invalid: list[str] = []

    for category, values in (
        ("entry_file", contract.get("entry_files")),
        ("required_document", contract.get("required_documents")),
    ):
        for raw in _strings(values):
            if raw in seen:
                continue
            seen.add(raw)
            candidate = (root / raw).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                invalid.append(raw)
                continue
            if not candidate.is_file():
                missing.append(raw)
                continue
            try:
                size = candidate.stat().st_size
            except OSError:
                missing.append(raw)
                continue
            rows.append(
                {
                    "path": raw,
                    "category": category,
                    "bytes": int(size),
                    "lines": _file_lines(candidate),
                }
            )

    rows.sort(key=lambda item: (-int(item["bytes"]), str(item["path"])))
    total_bytes = sum(int(item["bytes"]) for item in rows)
    total_lines = sum(int(item["lines"] or 0) for item in rows)
    largest = rows[:MAX_LARGEST_INITIAL_FILES]
    oversized = [
        str(item["path"])
        for item in rows
        if int(item["bytes"]) > INITIAL_CONTEXT_SINGLE_FILE_BYTES_WARNING
    ]
    warning_reasons: list[str] = []
    if total_bytes > INITIAL_CONTEXT_TOTAL_BYTES_WARNING:
        warning_reasons.append(
            f"initial context is {total_bytes} bytes, above the {INITIAL_CONTEXT_TOTAL_BYTES_WARNING}-byte advisory threshold"
        )
    if oversized:
        warning_reasons.append(
            "large initial files exceed the per-file advisory threshold: " + ", ".join(oversized)
        )
    if missing:
        warning_reasons.append("initial files are missing: " + ", ".join(sorted(missing)))
    if invalid:
        warning_reasons.append("initial paths escape the repository: " + ", ".join(sorted(invalid)))

    return {
        "status": "ADVISORY_WARNING" if warning_reasons else "WITHIN_ADVISORY_RANGE",
        "initial_file_count": len(seen),
        "existing_file_count": len(rows),
        "initial_total_bytes": total_bytes,
        "initial_total_lines": total_lines,
        "largest_initial_files": largest,
        "missing_initial_files": sorted(missing),
        "invalid_initial_paths": sorted(invalid),
        "warning_reasons": warning_reasons,
        "thresholds": {
            "total_bytes_warning": INITIAL_CONTEXT_TOTAL_BYTES_WARNING,
            "single_file_bytes_warning": INITIAL_CONTEXT_SINGLE_FILE_BYTES_WARNING,
        },
    }


def _package_projection(binding: Mapping[str, Any] | None) -> dict[str, Any] | None:
    row = _mapping(binding)
    if not row:
        return None
    return {
        "path": row.get("path"),
        "package_version": row.get("package_version"),
        "sha256": row.get("sha256"),
        "recommended_authority": row.get("recommended_authority"),
        "recommended_model": row.get("recommended_model"),
        "uncertainty_class": row.get("uncertainty_class"),
    }


def _work_projection(selected: Mapping[str, Any] | None) -> dict[str, Any] | None:
    row = _mapping(selected)
    if not row:
        return None
    payload = _mapping(row.get("payload"))
    summary = _mapping(row.get("summary"))
    return _drop_none(
        {
            "work_id": row.get("record_id"),
            "status": row.get("status"),
            "derived_status": summary.get("derived_status"),
            "title": payload.get("title") or summary.get("title"),
            "summary": payload.get("summary") or summary.get("summary"),
            "priority": payload.get("priority") or summary.get("priority"),
            "current_stage": payload.get("current_stage") or summary.get("current_stage"),
            "current_step": payload.get("current_step") or summary.get("current_step"),
            "next_action": payload.get("next_action") or summary.get("next_action"),
            "next_command": payload.get("next_command") or summary.get("next_command"),
            "package": _package_projection(
                payload.get("work_package_binding") or summary.get("work_package_binding")
            ),
        }
    )


def _session_projection(selected: Mapping[str, Any] | None) -> dict[str, Any] | None:
    row = _mapping(selected)
    if not row:
        return None
    payload = _mapping(row.get("payload"))
    model = _mapping(payload.get("model_selection") or payload.get("model_assignment"))
    health = _mapping(payload.get("context_health"))
    return _drop_none(
        {
            "session_id": row.get("record_id"),
            "status": row.get("status"),
            "work_id": payload.get("work_id"),
            "role": payload.get("session_role"),
            "current_codex_id": payload.get("current_codex_id"),
            "model": model.get("model"),
            "reasoning_level": model.get("reasoning_level"),
            "current_stage": payload.get("current_stage"),
            "current_step": payload.get("current_step"),
            "next_action": payload.get("next_action"),
            "next_command": payload.get("next_command"),
            "context_health_status": health.get("status"),
        }
    )


def _portfolio_projection(portfolio: Mapping[str, Any] | None) -> dict[str, Any]:
    row = _mapping(portfolio)
    open_work = list(row.get("open_work") or [])
    resumable = list(row.get("resumable_sessions") or [])
    result: dict[str, Any] = {
        "open_work_count": row.get("open_work_count", len(open_work)),
        "resumable_session_count": row.get("resumable_session_count", len(resumable)),
    }
    if int(result["open_work_count"] or 0) > 1:
        result["candidate_work_ids"] = [
            str(item.get("work_id")) for item in open_work if isinstance(item, Mapping) and item.get("work_id")
        ]
    if int(result["resumable_session_count"] or 0) > 1:
        result["resumable_session_ids"] = [
            str(item.get("session_id"))
            for item in resumable
            if isinstance(item, Mapping) and item.get("session_id")
        ]
    return result


def load_compact_workspace_artifact(raw: Any) -> dict[str, Any]:
    """Load the persisted compact Workspace response without normalization."""

    if not isinstance(raw, Mapping):
        raise ActorError("compact Workspace artifact must be an object", code="schema_record_malformed")
    value = dict(raw)
    workspace = value.get("workspace")
    if not isinstance(workspace, Mapping):
        raise ActorError("compact Workspace artifact has no workspace object", code="schema_record_malformed")
    require_explicit_schema_version(
        workspace,
        expected=COMPACT_WORKSPACE_SCHEMA_VERSION,
        label="compact Workspace",
    )
    return value



def project_compact_workspace_guidance(
    full_payload: Mapping[str, Any],
    *,
    full_view_command: str | None = None,
    full_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Derive one deterministic compact opening contract from full Workspace truth."""

    result = _mapping(full_payload)
    workspace = _mapping(result.get("workspace"))
    repository = _mapping(workspace.get("repository"))
    bootstrap = _mapping(workspace.get("bootstrap"))
    receipt = _mapping(workspace.get("context_receipt"))
    receipt_validation = _mapping(workspace.get("context_receipt_validation"))
    initial_context = _mapping(workspace.get("initial_context"))
    metrics = project_initial_context_metrics(initial_context, repo_root=repository.get("root"))

    warnings = _strings(result.get("warnings"))
    for reason in metrics.get("warning_reasons") or []:
        rendered = "Initial context advisory: " + str(reason)
        if rendered not in warnings:
            warnings.append(rendered)

    task_context = _mapping(workspace.get("task_context"))
    context_health = _mapping(workspace.get("context_health"))
    output_policy = _mapping(workspace.get("output_policy"))
    metrics_view: dict[str, Any] | None = None
    if initial_context:
        metrics_view = {
            "status": metrics.get("status"),
            "file_count": metrics.get("initial_file_count"),
            "existing_file_count": metrics.get("existing_file_count"),
            "total_bytes": metrics.get("initial_total_bytes"),
            "total_lines": metrics.get("initial_total_lines"),
            "largest_files": [
                {"path": row.get("path"), "bytes": row.get("bytes")}
                for row in list(metrics.get("largest_initial_files") or [])[:3]
            ],
        }
        if metrics.get("missing_initial_files"):
            metrics_view["missing_files"] = metrics.get("missing_initial_files")
        if metrics.get("invalid_initial_paths"):
            metrics_view["invalid_paths"] = metrics.get("invalid_initial_paths")

    health_view = {
        "status": context_health.get("status"),
        "handoff_recommended": bool(context_health.get("handoff_recommended")),
    }
    if context_health.get("handoff_recommended"):
        health_view["reasons"] = list(context_health.get("reasons") or [])

    receipt_view: dict[str, Any] | None = None
    if receipt:
        receipt_view = _drop_none(
            {
                "receipt_id": receipt.get("receipt_id"),
                "receipt_sha256": receipt.get("receipt_sha256"),
                "valid": receipt_validation.get("valid"),
            }
        )
        if receipt_validation.get("errors"):
            receipt_view["errors"] = list(receipt_validation.get("errors") or [])
        if receipt_validation.get("warnings"):
            receipt_view["warnings"] = list(receipt_validation.get("warnings") or [])

    compact_workspace = {
        "schema_version": COMPACT_WORKSPACE_SCHEMA_VERSION,
        "tooling_version": workspace.get("tooling_version"),
        "repository": _drop_none(
            {
                "root": repository.get("root"),
                "branch": repository.get("branch"),
                "head": repository.get("head"),
                "dirty": bool(repository.get("dirty")),
            }
        ),
        "bootstrap": _drop_none(
            {
                "record_id": bootstrap.get("record_id"),
                "status": bootstrap.get("status"),
                "age_hours": bootstrap.get("age_hours"),
            }
        ),
        "portfolio": _portfolio_projection(workspace.get("portfolio")),
        "work": _work_projection(workspace.get("selected_work")),
        "session": _session_projection(workspace.get("selected_session")),
        "task_context": _drop_none(
            {
                "task_class": task_context.get("task_class"),
                "required_authority_domain": task_context.get("required_authority_domain"),
                "required_session_role": task_context.get("required_session_role"),
                "qualification_required": task_context.get("qualification_required"),
                "qualification_status": task_context.get("qualification_status"),
            }
        ) or None,
        "initial_context": initial_context or None,
        "initial_context_metrics": metrics_view,
        "execution_budget": workspace.get("execution_budget"),
        "output_policy": _drop_none(
            {
                "default_output_mode": output_policy.get("default_output_mode"),
                "maximum_display_lines": output_policy.get("maximum_display_lines"),
                "test_output": output_policy.get("test_output"),
                "git_output": output_policy.get("git_output"),
                "json_output": output_policy.get("json_output"),
                "log_output": output_policy.get("log_output"),
            }
        ) or None,
        "context_health": health_view,
        "context_receipt": receipt_view,
        "workflow_administration": workspace.get("workflow_administration"),
        "references": _drop_none(
            {
                "full_view_command": full_view_command or f"{EIEL_LAUNCHER} workspace open --full",
                "full_output_path": str(full_output_path) if full_output_path else None,
            }
        ),
    }

    # Omit null conditional surfaces.  Their authoritative full records remain
    # available through the referenced full view.
    compact_workspace = {key: value for key, value in compact_workspace.items() if value is not None}

    return {
        "guidance_schema_version": result.get("guidance_schema_version"),
        "status": result.get("status"),
        "reason_codes": list(result.get("reason_codes") or []),
        "problems": list(result.get("problems") or []),
        "warnings": warnings,
        "next_actions": list(result.get("next_actions") or []),
        "related_records": list(result.get("related_records") or []),
        "ok": bool(result.get("ok")),
        "execution_blocked": bool(result.get("execution_blocked")),
        "workflow_administration_required": bool(result.get("workflow_administration_required")),
        "workspace": compact_workspace,
    }


__all__ = [
    "COMPACT_WORKSPACE_SCHEMA_VERSION",
    "INITIAL_CONTEXT_SINGLE_FILE_BYTES_WARNING",
    "INITIAL_CONTEXT_TOTAL_BYTES_WARNING",
    "load_compact_workspace_artifact",
    "project_compact_workspace_guidance",
    "project_initial_context_metrics",
]
