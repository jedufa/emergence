from __future__ import annotations

"""Canonical, atomic run-local storage for Scientific Chronicle V1."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from simulation.chronicle.schema import canonical_json, canonical_payload


ARTIFACT_NAMES = {
    "events": "formal_events.jsonl",
    "entries": "scientific_chronicle.jsonl",
    "episodes": "episodes.json",
    "contexts": "context_windows.jsonl",
    "summary": "scientific_chronicle_summary.json",
    "manifest": "chronicle_manifest.json",
    "storage": "chronicle_storage_summary.json",
    "index": "index.json",
}


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, canonical_json(payload) + "\n")


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    lines = [canonical_json(record) for record in records]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object in {path}:{line_number}")
        records.append(payload)
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_inventory(
    root: Path,
    *,
    logical_names: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    included = set(logical_names) if logical_names is not None else set(ARTIFACT_NAMES)
    rows: list[dict[str, Any]] = []
    for logical_name, filename in sorted(ARTIFACT_NAMES.items()):
        if logical_name not in included:
            continue
        path = root / filename
        if not path.exists():
            continue
        rows.append(
            {
                "artifact": logical_name,
                "relative_path": filename,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return canonical_payload(rows)
