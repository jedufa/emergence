from __future__ import annotations

"""Reporting-only birth-skew mechanism summaries.

The functions in this module derive compact lineage-share summaries from
already available generation snapshots. They do not influence selection,
reproduction, rewards, agent state, or environment state.
"""

import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "birth_skew_mechanism_reporting_v1"


def lineage_id(agent: Any) -> str | None:
    stats = getattr(agent, "stats", None)
    value = getattr(stats, "lineage_id", None) if stats is not None else None
    if value is None:
        value = getattr(agent, "lineage_id", None)
    if value is None or value == "":
        return None
    return str(value)


def _counter_from_agents(agents: Iterable[Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for agent in agents:
        lid = lineage_id(agent)
        if lid is not None:
            counts[lid] += 1
    return counts


def _counter_from_contexts(contexts: Iterable[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for context in contexts:
        if not isinstance(context, Mapping):
            continue
        value = context.get("lineage_id")
        if value is None or value == "":
            continue
        counts[str(value)] += 1
    return counts


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    out = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = float(count) / float(total)
        out -= p * math.log(p)
    return out


def share_summary(counts: Counter[str], *, top_n: int = 3) -> dict[str, Any]:
    total = int(sum(counts.values()))
    ranked = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    top = [
        {"lineage_id": str(lineage), "count": int(count), "share": float(count) / float(total) if total else 0.0}
        for lineage, count in ranked[:top_n]
    ]
    top1_share = top[0]["share"] if top else 0.0
    entropy = _entropy(counts)
    return {
        "total": total,
        "unique_lineages": int(len(counts)),
        "top1_lineage_id": top[0]["lineage_id"] if top else None,
        "top1_share": top1_share,
        "top_lineages": top,
        "entropy": entropy,
        "effective_lineages": float(math.exp(entropy)) if total else 0.0,
    }


def summarize_generation(
    *,
    generation: int,
    selected_parent_agents: Iterable[Any],
    alive_agents: Iterable[Any],
    birth_contexts: Iterable[Mapping[str, Any]],
    state: dict[str, Any] | None = None,
    top_n: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = dict(state or {})
    parent_counts = _counter_from_agents(selected_parent_agents)
    alive_counts = _counter_from_agents(alive_agents)
    birth_counts = _counter_from_contexts(birth_contexts)

    parent_summary = share_summary(parent_counts, top_n=top_n)
    alive_summary = share_summary(alive_counts, top_n=top_n)
    birth_summary = share_summary(birth_counts, top_n=top_n)

    if birth_summary["top1_share"] >= 0.8 and state.get("first_birth_dominance_generation") is None:
        state["first_birth_dominance_generation"] = int(generation)
    if alive_summary["top1_share"] >= 0.8 and state.get("first_alive_dominance_generation") is None:
        state["first_alive_dominance_generation"] = int(generation)

    birth_gen = state.get("first_birth_dominance_generation")
    alive_gen = state.get("first_alive_dominance_generation")
    lag = None
    if birth_gen is not None and alive_gen is not None:
        lag = int(alive_gen) - int(birth_gen)

    birth_lineage_ids = set(birth_counts.keys())
    nonzero_birth_lineages = set(state.get("nonzero_birth_lineages") or [])
    nonzero_birth_lineages.update(birth_lineage_ids)
    state["nonzero_birth_lineages"] = sorted(nonzero_birth_lineages)

    record = {
        "schema_version": SCHEMA_VERSION,
        "generation": int(generation),
        "reporting_only": True,
        "selected_parent_concentration": {
            "available": bool(parent_summary["total"] > 0),
            **parent_summary,
        },
        "birth_opportunity": {
            "available": bool(birth_summary["total"] > 0),
            **birth_summary,
            "cumulative_nonzero_birth_lineages": int(len(nonzero_birth_lineages)),
            "first_birth_dominance_generation": birth_gen,
            "first_alive_dominance_generation": alive_gen,
            "birth_to_alive_dominance_lag_generations": lag,
        },
        "alive_context": {
            "available": bool(alive_summary["total"] > 0),
            **alive_summary,
        },
        "refill_amplification": {
            "available": False,
            "reason": "Population cull/refill parent-source telemetry is not emitted by this passive generation snapshot.",
        },
        "offspring_locality": {
            "available": False,
            "reason": "Parent-child spatial distance telemetry is not emitted by this passive generation snapshot.",
        },
        "mechanics_intervention": False,
    }
    return record, state


def reporting_enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "birth_skew_mechanism_reporting_enable", False))


def maybe_write_birth_skew_mechanism_report(
    *,
    run_dir: str | os.PathLike[str] | None,
    generation: int,
    sim_cfg: Any,
    selected_parent_agents: Iterable[Any],
    alive_agents: Iterable[Any],
    birth_contexts: Iterable[Mapping[str, Any]],
    state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not reporting_enabled(sim_cfg):
        return None
    if not run_dir:
        return None

    top_n = int(getattr(sim_cfg, "birth_skew_mechanism_top_n", 3) or 3)
    record, new_state = summarize_generation(
        generation=generation,
        selected_parent_agents=selected_parent_agents,
        alive_agents=alive_agents,
        birth_contexts=birth_contexts,
        state=state,
        top_n=top_n,
    )
    metrics_dir = Path(run_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "birth_skew_mechanism.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
        f.write("\n")
    return {"path": str(path), "record": record, "state": new_state}

