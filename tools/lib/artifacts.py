"""Emergence-local artifact manifest helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_manifest(*, wrapper_dir: str | Path, job_id: str, job_spec_hash: str) -> Dict[str, Any]:
    """Build a deterministic wrapper artifact manifest."""

    root = Path(wrapper_dir)
    artifacts: List[Dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel == "artifact_manifest.json":
            continue
        artifacts.append(
            {
                "path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "schema_version": "v1",
        "job_id": str(job_id),
        "job_spec_hash": str(job_spec_hash),
        "wrapper_dir": root.resolve().as_posix(),
        "artifacts": artifacts,
    }


__all__ = ["build_artifact_manifest"]
