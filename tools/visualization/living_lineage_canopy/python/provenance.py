from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable


NO_CLAIMS_STATEMENT = (
    "This exporter and prototype visualize lineage and agent lifecycle data. "
    "They do not show or prove: intelligence, emergence, cognition, memory selection, "
    "prediction selection, ecological specialization, durable strategy diversity, "
    "consciousness, culture, open-ended evolution. Visual clustering, lobe formation, "
    "dominance, extinction, persistence, or ancestry projections are display properties "
    "over supplied records. They are not new scientific findings."
)


def canonical_json_bytes(data: Any) -> bytes:
    return (json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def write_canonical_json(path: Path, data: Any, *, overwrite: bool = False) -> str:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(data)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_for(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def exporter_git_commit() -> str:
    return git_commit_for(Path(__file__).resolve().parents[4])


def source_file_manifest(files: Iterable[Path]) -> list[dict[str, str]]:
    rows = []
    for path in sorted({item.resolve() for item in files}, key=lambda item: str(item)):
        if path.is_file():
            rows.append({"path": str(path), "sha256": sha256_file(path)})
    return rows


def output_hash(data: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(data)).hexdigest()


def validate_provenance_binding(playback: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    if provenance.get("schema_version") != "eiel_lineage_playback_export_provenance_v1":
        raise ValueError("provenance.schema_version is unsupported")
    expected_hash = output_hash(playback)
    recorded_hash = provenance.get("output_sha256")
    if recorded_hash != expected_hash:
        raise ValueError(f"provenance output_sha256 mismatch: expected {expected_hash}, got {recorded_hash}")
    playback_metadata = playback.get("metadata") or {}
    if provenance.get("playback_schema_version") != playback_metadata.get("schema_version"):
        raise ValueError("provenance playback_schema_version does not match playback metadata")
    for key in ("source_classification", "semantics_mode"):
        if provenance.get(key) != playback_metadata.get(key):
            raise ValueError(f"provenance {key} does not match playback metadata")
    if provenance.get("dataset_id") != playback_metadata.get("dataset_id"):
        raise ValueError("provenance dataset_id does not match playback metadata")
    for key in ("source_cell_id",):
        if provenance.get(key) != playback_metadata.get(key):
            raise ValueError(f"provenance {key} does not match playback metadata")
    commit = provenance.get("exporter_git_commit")
    if not isinstance(commit, str) or not commit:
        raise ValueError("provenance.exporter_git_commit missing")
    return {
        "ok": True,
        "output_sha256": expected_hash,
        "exporter_git_commit": commit,
    }


def verify_source_file_hashes(provenance: dict[str, Any]) -> dict[str, Any]:
    checked = []
    mismatches = []
    for row in provenance.get("source_files") or []:
        path = Path(row.get("path", ""))
        expected = row.get("sha256")
        if not path.is_file():
            mismatches.append(f"missing source file: {path}")
            continue
        actual = sha256_file(path)
        checked.append({"path": str(path), "sha256": actual})
        if actual != expected:
            mismatches.append(f"source hash mismatch: {path}")
    if mismatches:
        raise ValueError("; ".join(mismatches))
    return {"ok": True, "checked": checked}
