from __future__ import annotations

"""
simulation/metrics_pkg/metrics_logger_memory.py

Memory / GRU-related metric helpers.

This is split out because memory traits tend to expand over time (new genes,
new buckets, new memory systems). Keeping it isolated prevents metrics_logger.py
from turning into a giant ball of cognitive spaghetti.
"""

from typing import Any, Dict, List, Optional

from simulation.metrics_pkg.metrics_logger_utils import _unwrap_agent, _gget, _as_int


_MEMORY_BUCKETS = (0, 8, 16, 32, 64)


def _gru_enabled_from_genome(g: Any) -> bool:
    """Best-effort detection of GRU usage from genome."""
    if g is None:
        return False
    v = _gget(g, ["use_gru", "gru_enabled", "enable_gru", "has_gru", "gru"], None)
    if v is not None:
        try:
            if callable(v):
                return bool(v())
            return bool(v)
        except Exception:
            pass
    md = _gget(g, ["memory_dim", "mem_dim", "memory_size", "gru_dim", "hidden_memory_dim"], 0)
    return _as_int(md, 0) > 0


def _gru_frac(agents: Optional[List]) -> float:
    """Fraction of agents with GRU enabled."""
    if not agents:
        return 0.0
    n = len(agents)
    gcount = 0
    for raw in agents:
        a = _unwrap_agent(raw)
        if a is None:
            continue
        gg = getattr(a, "genome", None)
        if _gru_enabled_from_genome(gg):
            gcount += 1
    return float(gcount) / float(n) if n > 0 else 0.0


def _memory_dim_list(agents: Optional[List]) -> List[int]:
    """Extract memory dimension from genome for each agent."""
    if not agents:
        return []
    out: List[int] = []
    for raw in agents:
        a = _unwrap_agent(raw)
        if a is None:
            continue
        g = getattr(a, "genome", None)
        d = _gget(g, ["memory_dim", "mem_dim", "memory_size", "gru_dim", "hidden_memory_dim"], 0)
        out.append(_as_int(d, 0))
    return out


def _memory_bucket_fracs(dims: List[int]) -> Dict[int, float]:
    """Return bucket fractions for memory dims."""
    if not dims:
        return {b: 0.0 for b in _MEMORY_BUCKETS}
    n = float(len(dims))
    counts = {b: 0 for b in _MEMORY_BUCKETS}
    for d in dims:
        if d in counts:
            counts[d] += 1
        else:
            nearest = min(_MEMORY_BUCKETS, key=lambda x: abs(x - d))
            counts[nearest] += 1
    return {b: float(counts[b]) / n for b in _MEMORY_BUCKETS}
