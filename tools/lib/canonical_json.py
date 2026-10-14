"""Emergence-local canonical JSON helpers.

This module replaces an older compatibility proxy that pointed at a shared
package not present in this experiment snapshot. Keep the API small and local
so tooling and docs builds can import it without external dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from engine.simulation.control_plane_pkg.util_json import canonical_json_str


def canonical_json(obj: Any) -> str:
    """Return canonical JSON text for ``obj``."""

    return canonical_json_str(obj)


def dump_canonical_to_file(obj: Any, path: str | Path) -> None:
    """Write canonical JSON with a trailing newline."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"{canonical_json(obj)}\n", encoding="utf-8")


__all__ = ["canonical_json", "dump_canonical_to_file"]
