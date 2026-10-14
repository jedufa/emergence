from __future__ import annotations

"""
simulation/metrics_pkg/metrics_logger_utils.py

Generic helpers shared across metrics logger submodules.

These functions are intentionally boring:
- safe numeric casts
- agent unwrapping (some call sites pass wrappers/tuples)
- genome dict probing and multi-name lookup
- biome fraction extraction
- deterministic Spearman correlation helper

Keep this module dependency-light to avoid circular imports.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _range_width(lo: float, hi: float) -> float:
    """Return a non-zero range width (avoid divide-by-zero)."""
    w = float(hi) - float(lo)
    return w if w > 1e-12 else 1.0


def _unwrap_agent(x: Any) -> Any:
    """Accept Agent or agent-like wrappers and return the Agent object."""
    if x is None:
        return None
    if hasattr(x, "genome") or hasattr(x, "stats") or hasattr(x, "agent_id"):
        return x
    if isinstance(x, (tuple, list)):
        for it in x:
            if hasattr(it, "genome") or hasattr(it, "stats") or hasattr(it, "agent_id"):
                return it
        return x[0] if len(x) > 0 else None
    if isinstance(x, dict):
        for k in ("agent", "a", "obj", "value"):
            it = x.get(k, None)
            if it is not None and (hasattr(it, "genome") or hasattr(it, "stats") or hasattr(it, "agent_id")):
                return it
        for it in x.values():
            if hasattr(it, "genome") or hasattr(it, "stats") or hasattr(it, "agent_id"):
                return it
        return None
    return x


def _genome_dict(g: Any) -> Optional[dict]:
    """Try to discover a dict backing a genome object."""
    if g is None:
        return None
    for attr in ("genes", "params", "gene", "values", "data"):
        d = getattr(g, attr, None)
        if isinstance(d, dict):
            return d
    return None


def _gget(g: Any, names: List[str], default: Any) -> Any:
    """Get an attribute (or dict entry) from genome by trying multiple names."""
    if g is None:
        return default
    for n in names:
        if hasattr(g, n):
            try:
                return getattr(g, n)
            except Exception:
                pass
    d = _genome_dict(g)
    if d is not None:
        for n in names:
            if n in d:
                return d.get(n, default)
    return default


def _as_float(x: Any, default: float = 0.0) -> float:
    """Convert to float with a safe fallback."""
    try:
        return float(x)
    except Exception:
        return float(default)


def _as_int(x: Any, default: int = 0) -> int:
    """Convert to int with a safe fallback."""
    try:
        return int(x)
    except Exception:
        return int(default)


def _biome_fracs(agents: Optional[List]) -> Tuple[float, float]:
    """Return (safe_frac, wild_frac) based on stats.biome_id when present."""
    if not agents:
        return 0.0, 0.0
    safe = 0
    wild = 0
    for raw in agents:
        a = _unwrap_agent(raw)
        if a is None:
            continue
        st = getattr(a, "stats", None)
        bid = 0
        if st is not None:
            try:
                bid = int(getattr(st, "biome_id", 0))
            except Exception:
                bid = 0
        if bid == 1:
            wild += 1
        else:
            safe += 1
    n = float(len(agents))
    if n <= 0:
        return 0.0, 0.0
    return float(safe) / n, float(wild) / n


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    """Deterministic rank transform with average ties (for Spearman)."""
    n = int(x.size)
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)

    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        xv = x[order[i]]
        while j < n and x[order[j]] == xv:
            j += 1
        avg = 0.5 * ((i + 1) + j)
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j

    return ranks


def _spearman(x: List[float], y: List[float]) -> float:
    """Simple Spearman correlation; deterministic; ignores non-finite values."""
    if x is None or y is None:
        return 0.0
    n = min(len(x), len(y))
    if n < 3:
        return 0.0
    xa = np.asarray(x[:n], dtype=np.float64)
    ya = np.asarray(y[:n], dtype=np.float64)

    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    if xa.size < 3:
        return 0.0

    rx = _rankdata_average_ties(xa)
    ry = _rankdata_average_ties(ya)

    rxm = rx - rx.mean()
    rym = ry - ry.mean()
    denom = float(np.sqrt((rxm * rxm).sum()) * np.sqrt((rym * rym).sum()))
    if denom <= 1e-12:
        return 0.0
    return float((rxm * rym).sum() / denom)
