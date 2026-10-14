"""Bounded pre-admission reclamation through the existing lifecycle queue.

This module is intentionally a small adapter, not a storage manager.  It
consumes the exact pointers already produced by terminal lifecycle closeout,
delegates authority and deletion to the checked-in prune actuator, and makes
one bounded pass before the storage guard retries admission.  It never walks
``configured runtime root`` or copies material to cold storage.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from collections.abc import Mapping
import stat
from typing import Any

from .artifact_lifecycle import (
    DATA2_FILESYSTEM_ROOT,
    DATA2_LIFECYCLE_MANAGEMENT_ROOT,
    ArtifactLifecycleError,
    RunStorageGuard,
    _atomic_json,
    validate_artifact_disposition_manifest_v1,
    validate_data2_run_root_v1,
)
from .canonical import canonical_sha256


PRE_ADMISSION_RECLAMATION_SCHEMA_VERSION = (
    "eiel_pre_admission_reclamation_receipt_v1"
)
POINTER_SCHEMA_VERSION = "eiel_durable_archive_pointer_v1"
MAX_PRE_ADMISSION_CANDIDATES = 64


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_beneath(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _queue_root(guard: RunStorageGuard) -> Path:
    if bool(getattr(guard, "_is_data2", False)):
        return DATA2_LIFECYCLE_MANAGEMENT_ROOT / "pending"
    root = guard.run_root.expanduser().resolve(strict=False)
    for candidate in (root, *root.parents):
        if candidate.name == "causyl-runtime":
            return candidate / "lifecycle" / "pending"
    return root.parent / ".eiel-lifecycle-pending"


def _read_pointer(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactLifecycleError("lifecycle pointer is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactLifecycleError("lifecycle pointer is unreadable") from exc
    if not isinstance(value, Mapping):
        raise ArtifactLifecycleError("lifecycle pointer is not an object")
    pointer = dict(value)
    supplied = pointer.pop("pointer_sha256", None)
    if set(pointer) != {
        "schema_version",
        "run_root",
        "manifest_path",
        "manifest_sha256",
        "archive_root",
        "registered_at_utc",
    }:
        raise ArtifactLifecycleError("lifecycle pointer fields differ")
    if pointer.get("schema_version") != POINTER_SCHEMA_VERSION:
        raise ArtifactLifecycleError("lifecycle pointer schema is unsupported")
    if not isinstance(supplied, str) or supplied != canonical_sha256(pointer):
        raise ArtifactLifecycleError("lifecycle pointer digest differs")
    return {**pointer, "pointer_sha256": supplied}


def _pending_pointer_sort_key(path: Path) -> tuple[int, int, str]:
    """Match the lifecycle worker's fair order without following symlinks."""

    try:
        observed = path.lstat()
    except OSError:
        return (1, 0, path.name)
    valid_regular = stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1
    return (0 if valid_regular else 1, int(observed.st_mtime_ns), path.name)


def _candidate_from_pointer(
    pointer_path: Path,
    *,
    guard: RunStorageGuard,
) -> tuple[Path, Path, dict[str, Any]]:
    pointer = _read_pointer(pointer_path)
    raw_root = Path(str(pointer["run_root"])).expanduser()
    if not raw_root.is_absolute() or raw_root.is_symlink():
        raise ArtifactLifecycleError("lifecycle pointer run root is unsafe")
    root = raw_root.resolve(strict=False)
    if root == guard.run_root or _is_beneath(root, guard.run_root) or _is_beneath(
        guard.run_root, root
    ):
        raise ArtifactLifecycleError("candidate lifecycle root overlaps the admitted run")
    if bool(getattr(guard, "_is_data2", False)):
        root = validate_data2_run_root_v1(raw_root)
    if RunStorageGuard.run_root_is_reserved(root):
        raise ArtifactLifecycleError("candidate lifecycle root has an active reservation")
    manifest_path = Path(str(pointer["manifest_path"])).expanduser()
    if not manifest_path.is_absolute() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise ArtifactLifecycleError("candidate disposition manifest is unsafe or missing")
    manifest_path = manifest_path.resolve(strict=True)
    if not _is_beneath(manifest_path, root):
        raise ArtifactLifecycleError("candidate disposition manifest escapes its root")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactLifecycleError("candidate disposition manifest is unreadable") from exc
    if not isinstance(manifest, Mapping):
        raise ArtifactLifecycleError("candidate disposition manifest is not an object")
    validated = validate_artifact_disposition_manifest_v1(manifest)
    if validated.get("runtime_root") != str(root):
        raise ArtifactLifecycleError("candidate manifest runtime root differs")
    if validated.get("manifest_sha256") != pointer.get("manifest_sha256"):
        raise ArtifactLifecycleError("candidate manifest digest differs from pointer")
    if validated.get("terminal_status") not in {"completed", "COMPLETE", "terminal"}:
        raise ArtifactLifecycleError("candidate lifecycle work is not terminally complete")
    prune = validated.get("prune")
    if not isinstance(prune, Mapping) or prune.get("automatic_prune_eligible") is not True:
        raise ArtifactLifecycleError("candidate lifecycle manifest is not automatically releasable")
    return root, manifest_path, validated


def _write_receipt(
    *,
    receipt_root: Path,
    core: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    receipt_root = receipt_root.expanduser().resolve()
    receipt_root.mkdir(parents=True, exist_ok=True)
    body = dict(core)
    name = f"pre-admission-{canonical_sha256(body)[:24]}.json"
    path = receipt_root / name
    body["receipt_path"] = str(path)
    body["receipt_sha256"] = canonical_sha256(body)
    return _atomic_json(path, body), body


def reclaim_before_storage_wait_v1(
    guard: RunStorageGuard,
    failed_snapshot: Mapping[str, Any],
    *,
    pending_root: Path | None = None,
    receipt_root: Path | None = None,
    max_candidates: int = MAX_PRE_ADMISSION_CANDIDATES,
) -> dict[str, Any]:
    """Release exact completed-work R0-R2 material and return one receipt.

    The pass is bounded by pointer count and intentionally does not consume or
    mutate queue pointers.  A later archive/release worker can retry a failed
    candidate with its normal fairness rules.
    """

    if not isinstance(failed_snapshot, Mapping):
        raise ArtifactLifecycleError("failed storage snapshot must be an object")
    if type(max_candidates) is not int or not 1 <= max_candidates <= 1024:
        raise ArtifactLifecycleError("max_candidates is outside the bounded range")
    queue = (pending_root or _queue_root(guard)).expanduser()
    if queue.is_symlink() or not queue.is_dir():
        return {
            "schema_version": PRE_ADMISSION_RECLAMATION_SCHEMA_VERSION,
            "status": "NO_RECLAIMABLE_MATERIAL",
            "trigger": "before_storage_wait",
            "queue_root": str(queue),
            "candidate_limit": max_candidates,
            "candidates_examined": 0,
            "bytes_before": 0,
            "bytes_reclaimed": 0,
            "bytes_retained": 0,
            "unresolved": ["lifecycle pointer queue is absent"],
            "archive_policy": "DIRECT_HOT_RELEASE_ONLY_NO_HDD",
            "failed_snapshot": dict(failed_snapshot),
            "recorded_at_utc": _now(),
        }

    attempts: list[dict[str, Any]] = []
    bytes_before = 0
    bytes_reclaimed = 0
    bytes_retained = 0
    examined = 0
    # The background actuator rotates failed/busy pointers by mtime.  Sorting
    # by path name here permanently revisits the same prefix and can starve a
    # valid releasable pointer from the bounded pre-admission pass.  Reuse the
    # same stable, symlink-safe fairness order while leaving queue state
    # untouched.
    pending_pointers = sorted(queue.glob("*.json"), key=_pending_pointer_sort_key)
    for pointer_path in pending_pointers[:max_candidates]:
        examined += 1
        attempt: dict[str, Any] = {"pointer": str(pointer_path), "status": "SKIPPED"}
        try:
            root, manifest_path, manifest = _candidate_from_pointer(
                pointer_path,
                guard=guard,
            )
            # The checked-in prune actuator is an operator/script boundary.
            # Load it through its existing runtime plugin seam rather than
            # adding a new static campaigns -> tools.scripts dependency to
            # the frozen modular import graph.
            import importlib

            prune = importlib.import_module(
                "tools.scripts.eiel_runtime_storage_prune"
            )
            CATEGORIES = prune.CATEGORIES
            apply_prune_plan = prune.apply_prune_plan
            build_prune_plan = prune.build_prune_plan

            plan = build_prune_plan(root, set(CATEGORIES), manifest)
            selected_bytes = int(plan.get("selected_bytes") or 0)
            candidate_total_bytes = int(
                dict(manifest.get("summary") or {}).get("produced_bytes") or 0
            )
            if not plan.get("selected"):
                bytes_retained += candidate_total_bytes
                attempt.update(
                    {
                        "status": "NOOP",
                        "run_root": str(root),
                        "manifest_sha256": manifest.get("manifest_sha256"),
                        "selected_bytes": 0,
                        "bytes_reclaimed": 0,
                        "bytes_retained": candidate_total_bytes,
                    }
                )
                attempts.append(attempt)
                continue
            result = apply_prune_plan(
                root,
                plan,
                disposition_manifest=manifest,
                current_disposition_manifest_path=manifest_path,
            )
            reclaimed_rows = list(result.get("deleted_rows") or [])
            recovered_paths = {
                str(path) for path in (result.get("recovered_released") or [])
            }
            absent_paths = {
                str(path) for path in (result.get("already_absent") or [])
            }
            selected_by_path = {
                str(row.get("path")): row
                for row in plan.get("selected", [])
                if isinstance(row, Mapping)
            }
            reclaimed_rows = [
                row
                for row in reclaimed_rows
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
            bytes_before += max(0, selected_bytes - already_released - already_absent)
            bytes_reclaimed += reclaimed
            bytes_retained += max(
                0,
                candidate_total_bytes - reclaimed - already_released - already_absent,
            )
            attempt.update(
                {
                    "status": (
                        "RELEASED"
                        if reclaimed and not result.get("failures")
                        else "PARTIAL"
                        if result.get("failures")
                        else "NOOP"
                    ),
                    "run_root": str(root),
                    "manifest_sha256": manifest.get("manifest_sha256"),
                    "selected_bytes": selected_bytes,
                    "bytes_before": max(0, selected_bytes - already_released - already_absent),
                    "bytes_reclaimed": reclaimed,
                    "bytes_already_released": already_released,
                    "bytes_already_absent": already_absent,
                    "bytes_retained": max(
                        0,
                        candidate_total_bytes
                        - reclaimed
                        - already_released
                        - already_absent,
                    ),
                    "deleted": list(result.get("deleted") or []),
                    "recovered_released": sorted(recovered_paths),
                    "already_absent": sorted(absent_paths),
                    "failures": list(result.get("failures") or []),
                }
            )
            if result.get("failures"):
                attempt["reason"] = (
                    "prune actuator reported "
                    f"{len(result['failures'])} failure(s); candidate remains protected"
                )
        except Exception as exc:  # each pointer remains fail-closed and auditable
            attempt.update(
                {
                    "status": "HELD",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "bytes_reclaimed": 0,
                }
            )
            bytes_retained += int(attempt.get("selected_bytes") or 0)
        attempts.append(attempt)

    unresolved = [
        str(attempt.get("reason"))
        for attempt in attempts
        if attempt.get("status") in {"HELD", "PARTIAL"} and attempt.get("reason")
    ]
    released_count = sum(1 for attempt in attempts if attempt.get("bytes_reclaimed", 0) > 0)
    if bytes_reclaimed:
        status = "RECLAIMED" if not unresolved else "PARTIAL"
    elif unresolved:
        status = "NO_RECLAIMABLE_MATERIAL"
    else:
        status = "NO_RECLAIMABLE_MATERIAL"
    core = {
        "schema_version": PRE_ADMISSION_RECLAMATION_SCHEMA_VERSION,
        "status": status,
        "trigger": "before_storage_wait",
        "queue_root": str(queue.resolve(strict=False)),
        "candidate_limit": max_candidates,
        "candidates_examined": examined,
        "candidates_released": released_count,
        "bytes_before": bytes_before,
        "bytes_reclaimed": bytes_reclaimed,
        "bytes_retained": bytes_retained,
        "unresolved": unresolved,
        "attempts": attempts,
        "archive_policy": "DIRECT_HOT_RELEASE_ONLY_NO_HDD",
        "failed_snapshot": dict(failed_snapshot),
        "recorded_at_utc": _now(),
    }
    try:
        path, written = _write_receipt(
            receipt_root=receipt_root or (queue / "pre_admission"),
            core=core,
        )
        core = written
    except OSError as exc:
        core["receipt_write_error"] = f"{type(exc).__name__}: {exc}"
    if "receipt_sha256" not in core:
        core["receipt_sha256"] = canonical_sha256(
            {key: value for key, value in core.items() if key != "receipt_sha256"}
        )
    return core


__all__ = [
    "MAX_PRE_ADMISSION_CANDIDATES",
    "PRE_ADMISSION_RECLAMATION_SCHEMA_VERSION",
    "reclaim_before_storage_wait_v1",
]
