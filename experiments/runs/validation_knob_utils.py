from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import yaml  # type: ignore


def _resolve_runs_root(raw: str | Path) -> Path:
    return Path(str(raw)).expanduser().resolve()


def _load_registry(registry_path: str | Path) -> Tuple[str, int]:
    doc = yaml.safe_load(Path(registry_path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("registry YAML must parse into a mapping")
    return str(doc.get("name") or ""), int(doc.get("version") or 1)


def _find_batch_dir(runs_root: str | Path, _registry_name: str, _registry_version: int, _batch_tag: str) -> Path | None:
    root = _resolve_runs_root(runs_root)
    if not root.exists():
        return None
    candidates = sorted(
        [p.parent for p in root.rglob("summary.json") if p.is_file()],
        key=lambda p: (float(p.stat().st_mtime), str(p)),
        reverse=True,
    )
    return candidates[0] if candidates else None


def _summary_runs(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    summary_path = p if p.name == "summary.json" else (p / "summary.json")
    if not summary_path.exists():
        return []
    try:
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    runs = obj.get("runs")
    return list(runs) if isinstance(runs, list) else []
