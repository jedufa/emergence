"""Fail-closed filesystem contract for resumable WRC Atlas campaigns.

This is deliberately an Atlas-level ledger, not a simulator checkpoint or a
campaign launcher.  A world can be reused only when its identity and every
required durable artifact are hash verified.  Presence of a directory, log,
or timestamp is never completion evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from engine.wrc.reducer.canonical import canonical_sha256, pretty_json
from engine.wrc.reducer.state import ReducerStateError, load_checkpoint


CAMPAIGN_SCHEMA_VERSION = "eiel_wrc_atlas_resume_v1"
WORLD_COMPLETE_SCHEMA_VERSION = "eiel_wrc_world_complete_v1"
MANIFEST_NAME = "campaign_manifest.json"
STATUS_NAME = "status.tsv"
HISTORY_NAME = "resume_history.jsonl"


class AtlasResumeError(ValueError):
    """A campaign root, planned cell, or completion receipt is unsafe."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(pretty_json(value), encoding="utf-8")
    os.replace(temporary, path)


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AtlasResumeError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise AtlasResumeError(f"{label} root must be an object")
    return value


def _verified(value: Mapping[str, Any], *, hash_name: str, label: str) -> dict[str, Any]:
    material = dict(value)
    actual = material.pop(hash_name, None)
    if not isinstance(actual, str) or actual != canonical_sha256(material):
        raise AtlasResumeError(f"{label} hash verification failed")
    return material


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise AtlasResumeError("unsafe world path") from exc
    return candidate


def _cell(value: Mapping[str, Any]) -> dict[str, Any]:
    required = ("world_id", "parameter_id", "parameter_value", "seed", "expected_rows", "route_identity")
    if not all(name in value for name in required):
        raise AtlasResumeError("planned cell lacks required identity")
    world_id = value["world_id"]
    if not isinstance(world_id, str) or not world_id or "/" in world_id or "\\" in world_id:
        raise AtlasResumeError("planned world ID is unsafe")
    seed = value["seed"]
    rows = value["expected_rows"]
    if isinstance(seed, bool) or not isinstance(seed, int) or isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise AtlasResumeError("planned cell seed or expected_rows is invalid")
    if not isinstance(value["parameter_id"], str) or not value["parameter_id"]:
        raise AtlasResumeError("planned parameter identity is invalid")
    if not isinstance(value["route_identity"], Mapping) or not value["route_identity"]:
        raise AtlasResumeError("planned route identity is invalid")
    return {
        "world_id": world_id,
        "parameter_id": value["parameter_id"],
        "parameter_value": value["parameter_value"],
        "seed": seed,
        "expected_rows": rows,
        "route_identity": dict(value["route_identity"]),
    }


def campaign_manifest(identity: Mapping[str, Any], cells: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(identity, Mapping) or not identity:
        raise AtlasResumeError("campaign identity must be a nonempty object")
    normalized = [_cell(value) for value in cells]
    if not normalized:
        raise AtlasResumeError("campaign must plan at least one cell")
    ids = [value["world_id"] for value in normalized]
    if len(ids) != len(set(ids)):
        raise AtlasResumeError("campaign has duplicate planned world IDs")
    value: dict[str, Any] = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_identity": dict(identity),
        "planned_cells": sorted(normalized, key=lambda row: row["world_id"]),
    }
    value["manifest_sha256"] = canonical_sha256(value)
    return value


def initialize_campaign(root: str | Path, *, identity: Mapping[str, Any], cells: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    root_path = Path(root).resolve()
    manifest = campaign_manifest(identity, cells)
    path = root_path / MANIFEST_NAME
    if path.exists():
        existing = _verified(_read_object(path, label="campaign manifest"), hash_name="manifest_sha256", label="campaign manifest")
        if existing != {key: value for key, value in manifest.items() if key != "manifest_sha256"}:
            raise AtlasResumeError("campaign root identity differs; refusing a fresh root")
        return {**existing, "manifest_sha256": canonical_sha256(existing)}
    root_path.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, manifest)
    return manifest


def load_manifest(root: str | Path) -> dict[str, Any]:
    value = _read_object(Path(root) / MANIFEST_NAME, label="campaign manifest")
    material = _verified(value, hash_name="manifest_sha256", label="campaign manifest")
    if material.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        raise AtlasResumeError("unsupported campaign manifest schema")
    identity = material.get("campaign_identity")
    cells = material.get("planned_cells")
    if not isinstance(identity, Mapping) or not isinstance(cells, list):
        raise AtlasResumeError("campaign manifest is malformed")
    rebuilt = campaign_manifest(identity, cells)
    if rebuilt["manifest_sha256"] != value["manifest_sha256"]:
        raise AtlasResumeError("campaign manifest is not canonical")
    return rebuilt


def _metrics_valid(path: Path, *, expected_rows: int) -> tuple[bool, str | None]:
    if not path.is_file():
        return False, "metrics_missing"
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeDecodeError):
        return False, "metrics_malformed"
    if len(rows) != expected_rows:
        return False, "metrics_row_count"
    if not rows:
        return False, "metrics_header_only"
    for row in rows:
        if not row:
            return False, "metrics_empty_row"
        for raw in row.values():
            if raw is None or raw == "":
                continue
            try:
                if not math.isfinite(float(raw)):
                    return False, "metrics_nonfinite"
            except ValueError:
                continue
    return True, None


def _retention_valid(path: Path, *, cell: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        value = _read_object(path, label="retention manifest")
        material = _verified(value, hash_name="manifest_sha256", label="retention manifest")
    except AtlasResumeError:
        return False
    return bool(material.get("complete") is True and material.get("world_identity_sha256") == canonical_sha256(cell))


def _validated_final_reducer(path: Path, *, cell: Mapping[str, Any]) -> dict[str, str] | None:
    """Return only a finalized reducer checkpoint bound to this Atlas cell."""

    if not path.is_file():
        return None
    try:
        checkpoint = load_checkpoint(path)
    except ReducerStateError:
        return None
    identity = checkpoint.identity
    expected_identity = {
        "world_id": cell["world_id"],
        "parameter_id": cell["parameter_id"],
        "parameter_value": cell["parameter_value"],
        "seed": cell["seed"],
    }
    observed_identity = {
        "world_id": identity.world_id,
        "parameter_id": identity.parameter_id,
        "parameter_value": identity.parameter_value,
        "seed": identity.seed,
    }
    if (
        not checkpoint.finalized
        or checkpoint.finalized_result is None
        or canonical_sha256(observed_identity) != canonical_sha256(expected_identity)
    ):
        return None
    payload = checkpoint.payload()
    return {
        "checkpoint_sha256": str(payload["checkpoint_sha256"]),
        "identity_sha256": identity.sha256,
    }


def publish_world_complete(root: str | Path, *, world_id: str, reducer_state: str | Path, metrics: str | Path | None = None, retention_manifest: str | Path | None = None) -> dict[str, Any]:
    """Atomically publish a reusable world only after every required input verifies."""

    root_path = Path(root).resolve()
    manifest = load_manifest(root_path)
    by_id = {cell["world_id"]: cell for cell in manifest["planned_cells"]}
    cell = by_id.get(world_id)
    if cell is None:
        raise AtlasResumeError("world is not planned by this campaign")
    world_dir = _safe_child(root_path, f"worlds/{world_id}")
    if world_dir.exists() and not world_dir.is_dir():
        raise AtlasResumeError("world path is not a directory")
    reducer = Path(reducer_state).resolve()
    reducer_receipt = _validated_final_reducer(reducer, cell=cell)
    if reducer_receipt is None:
        raise AtlasResumeError("reducer state is not a finalized checkpoint bound to this world")
    metrics_path = Path(metrics).resolve() if metrics is not None else world_dir / "metrics.csv"
    retention_path = Path(retention_manifest).resolve() if retention_manifest is not None else world_dir / "retention_manifest.json"
    metrics_ok, metrics_error = _metrics_valid(metrics_path, expected_rows=cell["expected_rows"])
    retention_ok = _retention_valid(retention_path, cell=cell)
    if not metrics_ok and not retention_ok:
        raise AtlasResumeError(f"world outputs are not reusable: {metrics_error or 'retention_invalid'}")
    world_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": WORLD_COMPLETE_SCHEMA_VERSION,
        "cell": cell,
        "cell_sha256": canonical_sha256(cell),
        "reducer_state": {
            "path": reducer.relative_to(world_dir).as_posix() if reducer.is_relative_to(world_dir) else str(reducer),
            "sha256": _sha256_file(reducer),
            **reducer_receipt,
        },
        "metrics": None if not metrics_ok else {"path": metrics_path.relative_to(world_dir).as_posix() if metrics_path.is_relative_to(world_dir) else str(metrics_path), "sha256": _sha256_file(metrics_path), "row_count": cell["expected_rows"]},
        "retention_manifest": None if not retention_ok else {"path": retention_path.relative_to(world_dir).as_posix() if retention_path.is_relative_to(world_dir) else str(retention_path), "sha256": _sha256_file(retention_path)},
    }
    payload["world_complete_sha256"] = canonical_sha256(payload)
    _atomic_json(world_dir / "world_complete.json", payload)
    return payload


def _complete_reason(root: Path, cell: Mapping[str, Any]) -> tuple[bool, str]:
    world_dir = _safe_child(root, f"worlds/{cell['world_id']}")
    marker_path = world_dir / "world_complete.json"
    if not marker_path.is_file():
        return False, "missing"
    try:
        marker = _read_object(marker_path, label="world completion marker")
        material = _verified(marker, hash_name="world_complete_sha256", label="world completion marker")
    except AtlasResumeError:
        return False, "invalid_marker"
    if material.get("schema_version") != WORLD_COMPLETE_SCHEMA_VERSION or material.get("cell") != cell or material.get("cell_sha256") != canonical_sha256(cell):
        return False, "identity_mismatch"
    reducer = material.get("reducer_state")
    if not isinstance(reducer, Mapping) or not all(
        isinstance(reducer.get(name), str)
        for name in ("path", "sha256", "checkpoint_sha256", "identity_sha256")
    ):
        return False, "reducer_missing"
    reducer_path = _safe_child(world_dir, str(reducer["path"])) if not Path(str(reducer["path"])).is_absolute() else Path(str(reducer["path"]))
    reducer_receipt = _validated_final_reducer(reducer_path, cell=cell)
    if (
        not reducer_path.is_file()
        or _sha256_file(reducer_path) != reducer["sha256"]
        or reducer_receipt is None
        or reducer_receipt["checkpoint_sha256"] != reducer["checkpoint_sha256"]
        or reducer_receipt["identity_sha256"] != reducer["identity_sha256"]
    ):
        return False, "reducer_invalid"
    metrics = material.get("metrics")
    if isinstance(metrics, Mapping) and isinstance(metrics.get("path"), str) and isinstance(metrics.get("sha256"), str):
        metrics_path = _safe_child(world_dir, str(metrics["path"])) if not Path(str(metrics["path"])).is_absolute() else Path(str(metrics["path"]))
        valid, _ = _metrics_valid(metrics_path, expected_rows=cell["expected_rows"])
        if valid and _sha256_file(metrics_path) == metrics["sha256"]:
            return True, "complete"
    retention = material.get("retention_manifest")
    if isinstance(retention, Mapping) and isinstance(retention.get("path"), str) and isinstance(retention.get("sha256"), str):
        retention_path = _safe_child(world_dir, str(retention["path"])) if not Path(str(retention["path"])).is_absolute() else Path(str(retention["path"]))
        if retention_path.is_file() and _sha256_file(retention_path) == retention["sha256"] and _retention_valid(retention_path, cell=cell):
            return True, "complete_retained"
    return False, "outputs_invalid"


def scan_campaign(root: str | Path, *, write_status: bool = True) -> dict[str, Any]:
    root_path = Path(root).resolve()
    manifest = load_manifest(root_path)
    rows: list[dict[str, str]] = []
    counts = {"complete": 0, "invalid": 0, "missing": 0, "rerun": 0}
    for cell in manifest["planned_cells"]:
        complete, reason = _complete_reason(root_path, cell)
        status = "complete" if complete else ("missing" if reason == "missing" else "invalid")
        counts[status] += 1
        if status == "invalid":
            counts["rerun"] += 1
        rows.append({"world_id": cell["world_id"], "status": status, "reason": reason})
    result: dict[str, Any] = {"campaign_manifest_sha256": manifest["manifest_sha256"], "rows": rows, "counts": counts}
    result["status_sha256"] = canonical_sha256(result)
    if write_status:
        text = "world_id\tstatus\treason\n" + "".join(f"{row['world_id']}\t{row['status']}\t{row['reason']}\n" for row in rows)
        path = root_path / STATUS_NAME
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, path)
    return result


def append_resume_history(root: str | Path, scan: Mapping[str, Any]) -> None:
    """Append an exact scan receipt; callers must opt in to history mutation."""
    required = {"campaign_manifest_sha256", "status_sha256", "counts"}
    if not required <= set(scan):
        raise AtlasResumeError("scan receipt is incomplete")
    root_path = Path(root)
    with (root_path / HISTORY_NAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({key: scan[key] for key in sorted(required)}, sort_keys=True) + "\n")


__all__ = [
    "AtlasResumeError", "CAMPAIGN_SCHEMA_VERSION", "HISTORY_NAME", "MANIFEST_NAME", "STATUS_NAME",
    "append_resume_history", "campaign_manifest", "initialize_campaign", "load_manifest",
    "publish_world_complete", "scan_campaign",
]
