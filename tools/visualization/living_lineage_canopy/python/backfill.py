"""Read-only, deterministic checkpoint-bundle playback backfill."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .production_exporter import cross_check_gv2, export_gv2_checkpoint_run
from .provenance import output_hash, write_canonical_json
from .relay import RELAY_SCHEMA_VERSION, RELAY_STATUS_SCHEMA_VERSION, validate_relay_manifest


CATALOG_SCHEMA_VERSION = "eiel_canopy_backfill_catalog_v1"
PROJECTION_NAME = "native_runtime_checkpoint_projection.json.gz"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _key(source: Path) -> str:
    return hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:20]


def _atomic_directory(target: Path, files: dict[str, dict[str, Any]]) -> None:
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for name, value in files.items():
            write_canonical_json(temporary / name, value)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def discover_sources(
    roots: Iterable[Path], *, max_depth: int = 12, include: str | None = None, exclude: str | None = None, max_count: int | None = None
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Boundedly discover bundle-v2 roots without reading or hashing payloads."""

    candidates: dict[Path, None] = {}
    issues: list[dict[str, Any]] = []
    for raw_root in roots:
        root = raw_root.expanduser()
        if not root.is_dir():
            issues.append({"source_root": str(root), "catalog_status": "SOURCE_OFFLINE", "reason": "root is unavailable", "retryable": True})
            continue
        root = root.resolve()
        for current, directories, files in os.walk(root, topdown=True, followlinks=False, onerror=lambda error: issues.append({"source_root": str(error.filename or root), "catalog_status": "SOURCE_UNREADABLE", "reason": str(error), "retryable": True})):
            if max_count is not None and len(candidates) >= max_count:
                break
            current_path = Path(current)
            try:
                relative = current_path.relative_to(root)
            except ValueError:
                directories[:] = []
                continue
            directories[:] = [name for name in directories if not (current_path / name).is_symlink()]
            if len(relative.parts) >= max_depth:
                directories[:] = []
            if PROJECTION_NAME not in files and "native_runtime_checkpoint_projection.json" not in files:
                continue
            parts = current_path.parts
            if "checkpoint_bundles_v2" not in parts:
                continue
            source = Path(*parts[:parts.index("checkpoint_bundles_v2")])
            text = str(source)
            if (include and include not in text) or (exclude and exclude in text):
                continue
            candidates[source] = None
            if max_count is not None and len(candidates) >= max_count:
                break
    return sorted(candidates), issues


def backfill_sources(
    sources: Iterable[Path], *, derived_root: Path, dry_run: bool = False, generated_at: str | None = None
) -> dict[str, Any]:
    """Backfill only sources whose strict identity cross-check passes.

    Source runs are never written. Existing valid output is retained verbatim;
    invalid existing output fails closed instead of being replaced.
    """

    derived_root = derived_root.resolve()
    rows: list[dict[str, Any]] = []
    for raw_source in sources:
        source = raw_source.resolve()
        output_root = derived_root / _key(source) / "lineage-playback-relay"
        manifest_path = output_root / "relay_manifest.json"
        if manifest_path.is_file():
            validate_relay_manifest(manifest_path)
            rows.append({"source_root": str(source), "status": "existing", "catalog_status": "PLAYBACK_VALID", "output_root": str(output_root)})
            continue
        try:
            playback, provenance, summary = export_gv2_checkpoint_run(
                source, generated_at=generated_at or _now(), command="eiel canopy checkpoint backfill"
            )
            cross_check = cross_check_gv2(source, playback)
            if not cross_check["ok"]:
                raise ValueError("identity cross-check failed: " + "; ".join(cross_check["mismatches"]))
        except (OSError, ValueError) as error:
            rows.append({"source_root": str(source), "source_run_id": source.name, "status": "failure", "catalog_status": "INSUFFICIENT_IDENTITY", "reason": str(error), "retryable": False})
            continue
        row = {
            "source_root": str(source), "source_run_id": source.name, "status": "dry_run" if dry_run else "generated",
            "output_root": str(output_root), "classification": summary["classification"],
            "semantics_mode": summary["semantics_mode"], "playback_sha256": output_hash(playback),
        }
        if not dry_run:
            output_root.parent.mkdir(parents=True, exist_ok=True)
            written_at = generated_at or _now()
            manifest = {
                "schema_version": RELAY_SCHEMA_VERSION, "sequence_number": 1, "source_run_id": source.name,
                "source_root": str(source), "source_state": "source_complete",
                "current_source_classification": summary["classification"], "semantics_mode": summary["semantics_mode"],
                "branch_semantics": playback["metadata"]["branch_semantics"], "latest_completed_source_boundary": playback["metadata"]["last_tick"],
                "latest_exported_playback_boundary": playback["metadata"]["last_tick"], "playback_file_name": "playback_snapshot.json",
                "playback_sha256": output_hash(playback), "provenance_file_name": "provenance.json", "provenance_sha256": output_hash(provenance),
                "status_file_name": "relay_status.json", "cross_check_file_name": "cross_check.json", "cross_check_sha256": output_hash(cross_check),
                "source_lag": 0, "last_successful_update": written_at, "warnings": playback["metadata"]["warnings"],
            }
            status = {"schema_version": RELAY_STATUS_SCHEMA_VERSION, "state": "source_complete", "source_boundary": playback["metadata"]["last_tick"], "playback_boundary": playback["metadata"]["last_tick"], "last_successful_update": written_at, "incomplete_artifact_warnings": playback["metadata"]["warnings"]}
            _atomic_directory(output_root, {"playback_snapshot.json": playback, "provenance.json": provenance, "cross_check.json": cross_check, "relay_status.json": status, "relay_manifest.json": manifest})
        row["catalog_status"] = "BACKFILLABLE" if dry_run else "BACKFILLED"
        rows.append(row)
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("catalog_status") or row["status"]).upper()
        counts[status] = counts.get(status, 0) + 1
    catalog = {"schema_version": CATALOG_SCHEMA_VERSION, "generated_at": generated_at or _now(), "rows": sorted(rows, key=lambda row: row["source_root"]), "counts_by_classification": dict(sorted(counts.items()))}
    if not dry_run:
        derived_root.mkdir(parents=True, exist_ok=True)
        write_canonical_json(derived_root / "catalog.json", catalog, overwrite=True)
    return catalog


def bulk_backfill(
    roots: Iterable[Path], *, derived_root: Path, dry_run: bool = False, max_depth: int = 12, include: str | None = None, exclude: str | None = None, max_count: int | None = None, generated_at: str | None = None
) -> dict[str, Any]:
    sources, issues = discover_sources(roots, max_depth=max_depth, include=include, exclude=exclude, max_count=max_count)
    catalog = backfill_sources(sources, derived_root=derived_root, dry_run=dry_run, generated_at=generated_at)
    catalog["roots_scanned"] = [str(root) for root in roots]
    catalog["scan_issues"] = issues
    catalog["candidate_count"] = len(sources)
    for issue in issues:
        catalog["rows"].append(issue)
    if not dry_run:
        write_canonical_json(derived_root.resolve() / "catalog.json", catalog, overwrite=True)
    return catalog
