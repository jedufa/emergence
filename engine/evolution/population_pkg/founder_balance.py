from __future__ import annotations

"""Initial founder-lineage balance helpers.

This module is initialization-only. It assigns biological founder lineage
metadata for the initial population when explicitly enabled, and never enforces
lineage quotas after initialization.
"""

import math
from collections import Counter
from typing import Any, Iterable

from agents.agent import format_lineage_id, normalize_lineage_stats


FOUNDER_BALANCE_SCHEMA_VERSION = 1
FOUNDER_BALANCE_ASSIGNMENT_MODE = "deterministic_round_robin"


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def founder_balance_enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "lineage_founder_balance_enable", False)) if sim_cfg is not None else False


def validate_founder_balance_config(*, population_size: int, sim_cfg: Any) -> None:
    """Fail fast for invalid founder-balance initialization settings."""
    if not founder_balance_enabled(sim_cfg):
        return

    group_count = _as_int(getattr(sim_cfg, "lineage_founder_group_count", 0), 0)
    min_per_group = _as_int(getattr(sim_cfg, "lineage_founder_min_agents_per_group", 0), 0)
    schema_version = _as_int(getattr(sim_cfg, "lineage_founder_balance_schema_version", 0), 0)
    mode = str(getattr(sim_cfg, "lineage_founder_assignment_mode", "") or "")
    pop_n = _as_int(population_size, 0)

    if schema_version != FOUNDER_BALANCE_SCHEMA_VERSION:
        raise ValueError(
            f"lineage_founder_balance_schema_version must be {FOUNDER_BALANCE_SCHEMA_VERSION}, got {schema_version}"
        )
    if mode != FOUNDER_BALANCE_ASSIGNMENT_MODE:
        raise ValueError(
            "lineage_founder_assignment_mode must be "
            f"{FOUNDER_BALANCE_ASSIGNMENT_MODE!r}, got {mode!r}"
        )
    if pop_n <= 0:
        raise ValueError(f"population_size must be positive for founder balance, got {pop_n}")
    if group_count <= 0:
        raise ValueError(f"lineage_founder_group_count must be positive, got {group_count}")
    if min_per_group <= 0:
        raise ValueError(f"lineage_founder_min_agents_per_group must be positive, got {min_per_group}")
    if group_count > pop_n:
        raise ValueError(
            f"lineage_founder_group_count ({group_count}) cannot exceed population_size ({pop_n})"
        )
    if group_count * min_per_group > pop_n:
        raise ValueError(
            "lineage_founder_group_count * lineage_founder_min_agents_per_group "
            f"({group_count * min_per_group}) cannot exceed population_size ({pop_n})"
        )


def _share_metrics(counts: Iterable[int], total: int) -> dict[str, float]:
    values = [int(c) for c in counts if int(c) > 0]
    if total <= 0 or not values:
        return {
            "effective_lineages": 0.0,
            "entropy": 0.0,
            "top1_share": 0.0,
            "min_share": 0.0,
            "max_share": 0.0,
            "lineage_count": 0.0,
        }
    shares = [float(c) / float(total) for c in values]
    entropy = -sum(p * math.log(p) for p in shares if p > 0.0)
    return {
        "effective_lineages": float(math.exp(entropy)) if entropy > 0.0 else 1.0,
        "entropy": float(entropy),
        "top1_share": float(max(shares)),
        "min_share": float(min(shares)),
        "max_share": float(max(shares)),
        "lineage_count": float(len(values)),
    }


def _current_lineage_counts(agents: Iterable[Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for agent in agents:
        stats = normalize_lineage_stats(getattr(agent, "stats", None), getattr(agent, "agent_id", 0))
        if stats is None:
            continue
        lineage_id = str(getattr(stats, "lineage_id", "") or "")
        if lineage_id:
            counts[lineage_id] += 1
    return counts


def founder_balance_initial_metrics(population: Any) -> dict[str, float | int]:
    """Return initial founder-balance metrics attached to a population."""
    metrics = getattr(population, "_founder_balance_initial_metrics", None)
    if isinstance(metrics, dict):
        return dict(metrics)
    agents = sorted(getattr(population, "agents", {}).values(), key=lambda a: int(getattr(a, "agent_id", 0)))
    counts = _current_lineage_counts(agents)
    total = sum(counts.values())
    shares = _share_metrics(counts.values(), total)
    return {
        "founder_balance_enable": 0,
        "founder_group_count": 0,
        "founder_min_agents_per_group": 0,
        "founder_initial_effective_lineages": shares["effective_lineages"],
        "founder_initial_lineage_entropy": shares["entropy"],
        "founder_initial_top1_share": shares["top1_share"],
        "founder_initial_min_lineage_share": shares["min_share"],
        "founder_initial_max_lineage_share": shares["max_share"],
        "founder_initial_lineage_count": shares["lineage_count"],
        "founder_balance_violation_count": 0,
    }


def apply_founder_balance_initialization(population: Any) -> dict[str, float | int]:
    """Assign balanced founder lineage metadata to initial agents if enabled."""
    sim_cfg = getattr(population, "sim_cfg", None)
    pop_cfg = getattr(population, "pop_cfg", None)
    pop_n = _as_int(getattr(pop_cfg, "population_size", 0), 0)
    if not founder_balance_enabled(sim_cfg):
        metrics = founder_balance_initial_metrics(population)
        setattr(population, "_founder_balance_initial_metrics", metrics)
        return metrics

    validate_founder_balance_config(population_size=pop_n, sim_cfg=sim_cfg)
    group_count = _as_int(getattr(sim_cfg, "lineage_founder_group_count", 0), 0)
    min_per_group = _as_int(getattr(sim_cfg, "lineage_founder_min_agents_per_group", 0), 0)
    agents = sorted(getattr(population, "agents", {}).values(), key=lambda a: int(getattr(a, "agent_id", 0)))
    counts: Counter[int] = Counter()
    for idx, agent in enumerate(agents):
        group_id = (idx % group_count) + 1
        stats = normalize_lineage_stats(getattr(agent, "stats", None), getattr(agent, "agent_id", 0))
        if stats is None:
            continue
        stats.lineage_id = format_lineage_id(group_id)
        stats.founder_id = int(group_id)
        stats.parent_ids = tuple()
        stats.parent_lineage_ids = tuple()
        stats.birth_generation = 0
        stats.birth_tick = 0
        stats.lineage_depth = 0
        stats.patch_habitat_lineage_id = int(group_id)
        counts[group_id] += 1

    shares = _share_metrics(counts.values(), len(agents))
    violation_count = sum(1 for group_id in range(1, group_count + 1) if counts[group_id] < min_per_group)
    metrics = {
        "founder_balance_enable": 1,
        "founder_group_count": int(group_count),
        "founder_min_agents_per_group": int(min_per_group),
        "founder_initial_effective_lineages": shares["effective_lineages"],
        "founder_initial_lineage_entropy": shares["entropy"],
        "founder_initial_top1_share": shares["top1_share"],
        "founder_initial_min_lineage_share": shares["min_share"],
        "founder_initial_max_lineage_share": shares["max_share"],
        "founder_initial_lineage_count": shares["lineage_count"],
        "founder_balance_violation_count": int(violation_count),
    }
    setattr(population, "_founder_balance_initial_metrics", metrics)
    if sim_cfg is not None:
        for key, value in metrics.items():
            try:
                setattr(sim_cfg, f"_{key}", value)
            except Exception:
                pass
    return metrics
