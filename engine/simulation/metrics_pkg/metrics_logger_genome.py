from __future__ import annotations

"""
simulation/metrics_pkg/metrics_logger_genome.py

Genome / gene statistic helpers for MetricsLogger.

Responsibilities:
- Safe extraction of common genes from agent-like objects
- Computation of population-level gene means/stds + diversity proxy
- Diversity collapse tracking (stateful via provided prev_streak)

This module is intentionally schema-tolerant:
it computes a diversity proxy without requiring knowledge of the full genome.
"""

from typing import Any, Dict, List, Sequence, Tuple

from simulation.metrics_pkg.stats import mean, std
from simulation.metrics_pkg.metrics_logger_utils import (
    _unwrap_agent,
    _gget,
    _as_float,
    _range_width,
)

# These bounds are used for normalizing diversity/variance metrics. Keep stable.
_GENE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "exploration_bias": (-1.0, 1.0),
    "aggression_bias": (-2.0, 2.0),
    "reproduction_threshold": (2.0, 90.0),
    "mutation_rate": (0.001, 0.20),
    # Hidden size is sometimes fixed; keep a tight range to avoid div-by-zero.
    "hidden_size": (64.0, 65.0),
}

_DISTANCE_GENE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "hidden_size": (64.0, 65.0),
    "exploration_bias": (-1.0, 1.0),
    "aggression_bias": (-2.0, 2.0),
    "reproduction_threshold": (2.0, 90.0),
    "mutation_rate": (0.001, 0.20),
    "action_temperature": (0.25, 2.5),
    "drive_inertia": (0.0, 0.999),
    "drive_noise": (0.0, 0.10),
    "drive_logit_gain": (0.0, 3.0),
    "pred_lr": (0.0, 1.0),
    "drive_sp_hunger": (0.0, 1.0),
    "drive_sp_pain": (0.0, 1.0),
    "drive_sp_fatigue": (0.0, 1.0),
    "drive_sp_fear": (0.0, 1.0),
    "exp_prior_strength": (0.0, 1.0),
    "exp_prior_noise": (0.0, 0.5),
    "drive_birth_noise": (0.0, 0.2),
    "memory_dim": (0.0, 64.0),
    "sensor_radius": (1.0, 3.0),
}

_SCALAR_DISTANCE_FIELDS = (
    "hidden_size",
    "exploration_bias",
    "aggression_bias",
    "reproduction_threshold",
    "mutation_rate",
    "action_temperature",
    "drive_inertia",
    "drive_noise",
    "drive_logit_gain",
    "drive_sp_hunger",
    "drive_sp_pain",
    "drive_sp_fatigue",
    "drive_sp_fear",
    "exp_prior_strength",
    "exp_prior_noise",
    "drive_birth_noise",
    "sensor_radius",
)

_MASK_DISTANCE_FIELDS = (
    "sensor_channel_mask",
    "introspection_channel_mask",
    "self_regulation_control_mask",
    "self_modification_target_mask",
    "directed_inheritance_channel_mask",
)


def safe_aggression_bias(raw_agent: Any) -> float:
    """Extract genome aggression bias safely from an agent-like object."""
    a = _unwrap_agent(raw_agent)
    if a is None:
        return 0.0
    g = getattr(a, "genome", None)
    v = _gget(g, ["aggression_bias", "agg_bias", "aggr_bias"], 0.0)
    return _as_float(v, 0.0)


def safe_exploration_bias(raw_agent: Any) -> float:
    """Extract genome exploration bias safely from an agent-like object."""
    a = _unwrap_agent(raw_agent)
    if a is None:
        return 0.0
    g = getattr(a, "genome", None)
    v = _gget(g, ["exploration_bias", "explore_bias", "exp_bias"], 0.0)
    return _as_float(v, 0.0)


def safe_repro_threshold(raw_agent: Any) -> float:
    """Extract reproduction threshold gene safely."""
    a = _unwrap_agent(raw_agent)
    if a is None:
        return 0.0
    g = getattr(a, "genome", None)
    v = _gget(g, ["reproduction_threshold", "repro_threshold", "repro_thr"], 0.0)
    return _as_float(v, 0.0)


def safe_mutation_rate(raw_agent: Any) -> float:
    """Extract mutation rate gene safely."""
    a = _unwrap_agent(raw_agent)
    if a is None:
        return 0.0
    g = getattr(a, "genome", None)
    v = _gget(g, ["mutation_rate", "mut_rate", "mu"], 0.0)
    return _as_float(v, 0.0)


def safe_hidden_size(raw_agent: Any) -> float:
    """Extract hidden size (or memory dim proxy) safely."""
    a = _unwrap_agent(raw_agent)
    if a is None:
        return 0.0
    g = getattr(a, "genome", None)
    v = _gget(g, ["hidden_size", "h", "h_size", "gru_hidden", "gru_hidden_size"], None)
    if v is None:
        # Fallback: memory dim is often the same concept in this codebase.
        v = _gget(g, ["memory_dim", "mem_dim", "gru_dim", "memory_size"], 0.0)
    return _as_float(v, 0.0)


def _genome_for(raw_agent: Any) -> Any:
    a = _unwrap_agent(raw_agent)
    if a is None:
        return None
    return getattr(a, "genome", None)


def _norm_abs_delta(a: Any, b: Any, key: str) -> float:
    lo, hi = _DISTANCE_GENE_BOUNDS.get(key, (0.0, 1.0))
    width = _range_width(lo, hi)
    return min(1.0, abs(_as_float(a, 0.0) - _as_float(b, 0.0)) / width)


def _bit_hamming_norm(a: Any, b: Any) -> float:
    try:
        ia = int(a)
        ib = int(b)
        x = ia ^ ib
    except Exception:
        return 0.0
    width = max(1, max(ia.bit_length(), ib.bit_length()))
    return float(int(x).bit_count()) / float(width)


def _module_gate_distance(parent_g: Any, child_g: Any) -> float:
    mask_deltas = [
        _bit_hamming_norm(getattr(parent_g, key, 0), getattr(child_g, key, 0))
        for key in _MASK_DISTANCE_FIELDS
    ]
    module_delta = min(1.0, abs(len(list(getattr(parent_g, "modules", []) or [])) - len(list(getattr(child_g, "modules", []) or []))) / 12.0)
    connection_delta = min(
        1.0,
        abs(len(list(getattr(parent_g, "connections", []) or [])) - len(list(getattr(child_g, "connections", []) or []))) / 32.0,
    )
    return float(mean(mask_deltas + [module_delta, connection_delta]))


def _distance_to_parent(parent_g: Any, child_g: Any) -> dict[str, float]:
    scalar = [
        _norm_abs_delta(getattr(parent_g, key, 0.0), getattr(child_g, key, 0.0), key)
        for key in _SCALAR_DISTANCE_FIELDS
    ]
    memory = mean([
        _norm_abs_delta(getattr(parent_g, "memory_dim", 0), getattr(child_g, "memory_dim", 0), "memory_dim"),
        1.0 if bool(getattr(parent_g, "use_gru", False)) != bool(getattr(child_g, "use_gru", False)) else 0.0,
    ])
    prediction = _norm_abs_delta(getattr(parent_g, "pred_lr", 0.0), getattr(child_g, "pred_lr", 0.0), "pred_lr")
    module_gate = _module_gate_distance(parent_g, child_g)
    all_components = scalar + [memory, prediction, module_gate]
    return {
        "total": float(mean(all_components)),
        "scalar": float(mean(scalar)),
        "memory": float(memory),
        "prediction": float(prediction),
        "module_gate": float(module_gate),
    }


def parent_child_genome_distance_summary(parent: Any, mate: Any, child: Any) -> dict[str, float]:
    """Report-only normalized genome distance from child to available parents."""
    child_g = _genome_for(child)
    parent_genomes = [g for g in (_genome_for(parent), _genome_for(mate)) if g is not None]
    if child_g is None or not parent_genomes:
        return {
            "parent_child_genome_distance_mean": 0.0,
            "parent_child_genome_distance_max": 0.0,
            "parent_child_genome_scalar_distance_mean": 0.0,
            "parent_child_genome_memory_field_delta": 0.0,
            "parent_child_genome_prediction_field_delta": 0.0,
            "parent_child_genome_module_gate_delta": 0.0,
        }
    rows = [_distance_to_parent(parent_g, child_g) for parent_g in parent_genomes]
    totals = [row["total"] for row in rows]
    return {
        "parent_child_genome_distance_mean": float(mean(totals)),
        "parent_child_genome_distance_max": float(max(totals) if totals else 0.0),
        "parent_child_genome_scalar_distance_mean": float(mean([row["scalar"] for row in rows])),
        "parent_child_genome_memory_field_delta": float(mean([row["memory"] for row in rows])),
        "parent_child_genome_prediction_field_delta": float(mean([row["prediction"] for row in rows])),
        "parent_child_genome_module_gate_delta": float(mean([row["module_gate"] for row in rows])),
    }


def genome_stats_from_agents(
    agents: List,
) -> Tuple[float, float, float, float, float, float, float, float, float, float, float]:
    """Compute mean/std for key genome traits + normalized diversity proxy.

    Returns:
        hs_mean, hs_std,
        exp_mean, exp_std,
        agg_mean, agg_std,
        repro_thr_mean, repro_thr_std,
        mut_rate_mean, mut_rate_std,
        diversity_norm_std_mean
    """
    return genome_stats_from_trait_values(
        [safe_hidden_size(a) for a in agents],
        [safe_exploration_bias(a) for a in agents],
        [safe_aggression_bias(a) for a in agents],
        [safe_repro_threshold(a) for a in agents],
        [safe_mutation_rate(a) for a in agents],
    )


def genome_stats_from_trait_values(
    hidden_sizes: Sequence[float],
    exploration_biases: Sequence[float],
    aggression_biases: Sequence[float],
    reproduction_thresholds: Sequence[float],
    mutation_rates: Sequence[float],
) -> Tuple[float, float, float, float, float, float, float, float, float, float, float]:
    """Compute the canonical genome aggregate contract from typed trait rows."""

    hs = [float(value) for value in hidden_sizes]
    ex = [float(value) for value in exploration_biases]
    ag = [float(value) for value in aggression_biases]
    rt = [float(value) for value in reproduction_thresholds]
    mr = [float(value) for value in mutation_rates]
    lengths = {len(hs), len(ex), len(ag), len(rt), len(mr)}
    if lengths == {0}:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if len(lengths) != 1:
        raise ValueError("genome trait rows must have equal non-zero lengths")

    def norm_std(vals: List[float], key: str) -> float:
        """Normalized standard deviation for diversity proxy."""
        if not vals:
            return 0.0
        lo, hi = _GENE_BOUNDS.get(key, (0.0, 1.0))
        w = _range_width(lo, hi)
        return float(std(vals)) / w

    diversity = float(mean([
        norm_std(hs, "hidden_size"),
        norm_std(ex, "exploration_bias"),
        norm_std(ag, "aggression_bias"),
        norm_std(rt, "reproduction_threshold"),
        norm_std(mr, "mutation_rate"),
    ]))

    return (
        float(mean(hs)), float(std(hs)),
        float(mean(ex)), float(std(ex)),
        float(mean(ag)), float(std(ag)),
        float(mean(rt)), float(std(rt)),
        float(mean(mr)), float(std(mr)),
        float(diversity),
    )


def update_collapse_state(
    diversity_norm_std_mean: float,
    sim_cfg: Any,
    prev_streak: int,
) -> Tuple[int, int]:
    """Track sustained diversity collapse.

    Args:
        diversity_norm_std_mean: diversity proxy (normalized).
        sim_cfg: sim config with optional thresholds/patience fields.
        prev_streak: previous consecutive-under-threshold streak.

    Returns:
        (collapse_flag, new_streak)
    """
    thr = float(getattr(sim_cfg, "diversity_collapse_threshold", 0.02)) if sim_cfg is not None else 0.02
    need = (
        int(getattr(sim_cfg, "diversity_collapse_streak_required"))
        if hasattr(sim_cfg, "diversity_collapse_streak_required")
        else int(getattr(sim_cfg, "diversity_collapse_patience", 5))
    ) if sim_cfg is not None else 5

    streak = int(prev_streak)
    if float(diversity_norm_std_mean) < thr:
        streak += 1
    else:
        streak = 0

    flag = 1 if streak >= need else 0
    return int(flag), int(streak)
