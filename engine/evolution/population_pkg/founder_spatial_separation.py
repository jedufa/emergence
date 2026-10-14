from __future__ import annotations

"""Initialization-only founder spatial/ecological separation diagnostics.

This module never changes post-initialization movement, reproduction,
selection, rewards, quotas, or survival. It only provides deterministic
generation-zero placement when explicitly enabled.
"""

import json
import math
import os
from collections import Counter
from typing import Any, Iterable

import numpy as np

from agents.agent import normalize_lineage_stats


FOUNDER_SPATIAL_SCHEMA_VERSION = 1
FOUNDER_SPATIAL_ASSIGNMENT_MODES = {
    "none",
    "grid_quadrants",
    "max_distance",
    "patch_basin_round_robin",
}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_bool(value: Any) -> bool:
    return bool(value)


def founder_spatial_separation_enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "lineage_founder_spatial_separation_enable", False)) if sim_cfg is not None else False


def validate_founder_spatial_config(*, population_size: int, env_cfg: Any, sim_cfg: Any) -> None:
    if not founder_spatial_separation_enabled(sim_cfg):
        return
    schema_version = _as_int(getattr(sim_cfg, "lineage_founder_spatial_schema_version", 0), 0)
    mode = str(getattr(sim_cfg, "lineage_founder_spatial_assignment_mode", "none") or "none")
    group_count = _as_int(getattr(sim_cfg, "lineage_founder_spatial_group_count", 0), 0)
    min_distance = _as_int(getattr(sim_cfg, "lineage_founder_spatial_min_distance", 0), 0)
    width = _as_int(getattr(env_cfg, "width", 0), 0)
    height = _as_int(getattr(env_cfg, "height", 0), 0)
    pop_n = _as_int(population_size, 0)

    if schema_version != FOUNDER_SPATIAL_SCHEMA_VERSION:
        raise ValueError(
            f"lineage_founder_spatial_schema_version must be {FOUNDER_SPATIAL_SCHEMA_VERSION}, got {schema_version}"
        )
    if mode not in FOUNDER_SPATIAL_ASSIGNMENT_MODES:
        raise ValueError(
            "lineage_founder_spatial_assignment_mode must be one of "
            f"{sorted(FOUNDER_SPATIAL_ASSIGNMENT_MODES)!r}, got {mode!r}"
        )
    if pop_n <= 0:
        raise ValueError(f"population_size must be positive for founder spatial separation, got {pop_n}")
    if width <= 0 or height <= 0:
        raise ValueError(f"env width/height must be positive, got width={width} height={height}")
    if group_count < 0:
        raise ValueError(f"lineage_founder_spatial_group_count must be >= 0, got {group_count}")
    if group_count > pop_n:
        raise ValueError(
            f"lineage_founder_spatial_group_count ({group_count}) cannot exceed population_size ({pop_n})"
        )
    if min_distance < 0:
        raise ValueError(f"lineage_founder_spatial_min_distance must be >= 0, got {min_distance}")


def _agent_id(agent: Any) -> int:
    return _as_int(getattr(agent, "agent_id", 0), 0)


def _lineage_id(agent: Any) -> str:
    aid = _agent_id(agent)
    stats = normalize_lineage_stats(getattr(agent, "stats", None), aid)
    return str(getattr(stats, "lineage_id", f"L{aid:08d}") if stats is not None else f"L{aid:08d}")


def _founder_group_id(agent: Any) -> int:
    aid = _agent_id(agent)
    stats = normalize_lineage_stats(getattr(agent, "stats", None), aid)
    return _as_int(getattr(stats, "founder_id", aid) if stats is not None else aid, aid)


def _groups(agents: Iterable[Any], configured_group_count: int) -> list[tuple[str, list[Any]]]:
    ordered = sorted(list(agents or []), key=_agent_id)
    if configured_group_count > 0:
        grouped_by_lineage: dict[str, list[Any]] = {}
        for agent in ordered:
            grouped_by_lineage.setdefault(_lineage_id(agent), []).append(agent)
        if len(grouped_by_lineage) == configured_group_count:
            return [(lid, grouped_by_lineage[lid]) for lid in sorted(grouped_by_lineage.keys())]
        spatial_groups: dict[str, list[Any]] = {f"SG{idx + 1:08d}": [] for idx in range(configured_group_count)}
        for idx, agent in enumerate(ordered):
            spatial_groups[f"SG{(idx % configured_group_count) + 1:08d}"].append(agent)
        return [(gid, spatial_groups[gid]) for gid in sorted(spatial_groups.keys()) if spatial_groups[gid]]
    grouped: dict[str, list[Any]] = {}
    for agent in ordered:
        grouped.setdefault(_lineage_id(agent), []).append(agent)
    rows = [(lid, grouped[lid]) for lid in sorted(grouped.keys())]
    return rows


def _grid_anchor_candidates(width: int, height: int) -> list[tuple[int, int]]:
    corners = [
        (0, 0),
        (0, width - 1),
        (height - 1, 0),
        (height - 1, width - 1),
        (height // 2, width // 2),
        (0, width // 2),
        (height - 1, width // 2),
        (height // 2, 0),
        (height // 2, width - 1),
    ]
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for y, x in corners:
        pos = (max(0, min(height - 1, int(y))), max(0, min(width - 1, int(x))))
        if pos not in seen:
            out.append(pos)
            seen.add(pos)
    for y in range(height):
        for x in range(width):
            pos = (y, x)
            if pos not in seen:
                out.append(pos)
                seen.add(pos)
    return out


def _grid_quadrant_anchors(width: int, height: int, count: int) -> list[tuple[int, int]]:
    if count <= 0:
        return []
    rows = max(1, int(math.ceil(math.sqrt(count))))
    cols = max(1, int(math.ceil(count / float(rows))))
    anchors: list[tuple[int, int]] = []
    for idx in range(count):
        r = idx // cols
        c = idx % cols
        y = int(round(((r + 0.5) / rows) * max(1, height) - 0.5))
        x = int(round(((c + 0.5) / cols) * max(1, width) - 0.5))
        anchors.append((max(0, min(height - 1, y)), max(0, min(width - 1, x))))
    return anchors


def _max_distance_anchors(width: int, height: int, count: int) -> list[tuple[int, int]]:
    candidates = _grid_anchor_candidates(width, height)
    if count <= 0:
        return []
    anchors = [candidates[0]]
    remaining = candidates[1:]
    while len(anchors) < count and remaining:
        best = max(
            remaining,
            key=lambda pos: (
                min(abs(pos[0] - a[0]) + abs(pos[1] - a[1]) for a in anchors),
                -pos[0],
                -pos[1],
            ),
        )
        anchors.append(best)
        remaining = [p for p in remaining if p != best]
    return anchors


def _basin_id(env: Any, y: int, x: int) -> str:
    if env is None:
        return ""
    for name in ("patch_region_id", "region_id_map", "tier3_region_id"):
        arr = getattr(env, name, None)
        if arr is None:
            continue
        try:
            return f"{name}:{int(np.asarray(arr)[int(y), int(x)])}"
        except Exception:
            continue
    height = max(1, _as_int(getattr(getattr(env, "cfg", None), "height", 1), 1))
    width = max(1, _as_int(getattr(getattr(env, "cfg", None), "width", 1), 1))
    return f"coarse:{0 if y < height / 2 else 1}:{0 if x < width / 2 else 1}"


def _spiral_offsets(limit: int) -> list[tuple[int, int]]:
    offsets = [(0, 0)]
    radius = 1
    while len(offsets) < limit:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if max(abs(dy), abs(dx)) == radius:
                    offsets.append((dy, dx))
        radius += 1
    return offsets


def _place_group(
    *,
    anchor: tuple[int, int],
    agents: list[Any],
    occupied: set[tuple[int, int]],
    width: int,
    height: int,
) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    offsets = _spiral_offsets(width * height)
    ay, ax = anchor
    for agent in sorted(agents, key=_agent_id):
        placed = None
        for dy, dx in offsets:
            y = ay + dy
            x = ax + dx
            if 0 <= y < height and 0 <= x < width and (y, x) not in occupied:
                placed = (int(y), int(x))
                break
        if placed is None:
            raise RuntimeError("lineage founder spatial separation could not place all founders: no empty cells")
        occupied.add(placed)
        out[_agent_id(agent)] = placed
    return out


def build_founder_spatial_plan(*, agents: Iterable[Any], env_cfg: Any, sim_cfg: Any, env: Any = None) -> dict[str, Any]:
    agents_sorted = sorted(list(agents or []), key=_agent_id)
    validate_founder_spatial_config(
        population_size=len(agents_sorted),
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
    )
    width = _as_int(getattr(env_cfg, "width", 0), 0)
    height = _as_int(getattr(env_cfg, "height", 0), 0)
    mode = str(getattr(sim_cfg, "lineage_founder_spatial_assignment_mode", "none") or "none")
    configured_group_count = _as_int(getattr(sim_cfg, "lineage_founder_spatial_group_count", 0), 0)
    min_distance = _as_int(getattr(sim_cfg, "lineage_founder_spatial_min_distance", 0), 0)
    use_patch_basins = _as_bool(getattr(sim_cfg, "lineage_founder_spatial_use_patch_basins", False))
    groups = _groups(agents_sorted, configured_group_count)

    if mode == "none":
        anchors = [(0, 0) for _ in groups]
    elif mode == "grid_quadrants":
        anchors = _grid_quadrant_anchors(width, height, len(groups))
    elif mode in {"max_distance", "patch_basin_round_robin"}:
        anchors = _max_distance_anchors(width, height, len(groups))
    else:
        raise ValueError(f"unsupported founder spatial assignment mode: {mode!r}")

    occupied: set[tuple[int, int]] = set()
    agent_positions: dict[int, tuple[int, int]] = {}
    agent_spatial_group: dict[int, str] = {}
    agent_basin: dict[int, str] = {}
    group_records: list[dict[str, Any]] = []
    for idx, ((lineage_id, members), anchor) in enumerate(zip(groups, anchors), start=1):
        group_positions = _place_group(anchor=anchor, agents=members, occupied=occupied, width=width, height=height)
        agent_positions.update(group_positions)
        ys = [pos[0] for pos in group_positions.values()]
        xs = [pos[1] for pos in group_positions.values()]
        cy = float(sum(ys)) / float(max(1, len(ys)))
        cx = float(sum(xs)) / float(max(1, len(xs)))
        group_records.append(
            {
                "lineage_id": lineage_id,
                "founder_group_id": _founder_group_id(members[0]) if members else idx,
                "agent_count": len(members),
                "anchor_y": int(anchor[0]),
                "anchor_x": int(anchor[1]),
                "centroid_y": cy,
                "centroid_x": cx,
                "basin_id": _basin_id(env, int(round(cy)), int(round(cx))) if use_patch_basins else "",
            }
        )
        for agent in members:
            aid = _agent_id(agent)
            agent_spatial_group[aid] = lineage_id
            agent_basin[aid] = str(group_records[-1]["basin_id"])

    distances: list[float] = []
    for i, left in enumerate(group_records):
        for right in group_records[i + 1 :]:
            distances.append(
                abs(float(left["centroid_y"]) - float(right["centroid_y"]))
                + abs(float(left["centroid_x"]) - float(right["centroid_x"]))
            )
    violations = [d for d in distances if d < float(min_distance)]
    mapping = [
        {
            "agent_id": _agent_id(agent),
            "lineage_id": _lineage_id(agent),
            "spatial_group_id": agent_spatial_group.get(_agent_id(agent), ""),
            "founder_group_id": _founder_group_id(agent),
            "y": int(agent_positions[_agent_id(agent)][0]),
            "x": int(agent_positions[_agent_id(agent)][1]),
            "basin_id": agent_basin.get(_agent_id(agent), ""),
        }
        for agent in agents_sorted
        if _agent_id(agent) in agent_positions
    ]
    return {
        "schema_version": "lineage_founder_spatial_separation_v1",
        "enabled": True,
        "assignment_mode": mode,
        "group_count": len(groups),
        "configured_group_count": int(configured_group_count),
        "min_distance": int(min_distance),
        "min_distance_satisfied": not violations,
        "warning": "min_distance_not_feasible" if violations else "",
        "founder_collision_index": float(len(violations)) / float(max(1, len(distances))),
        "pairwise_distance_summary": {
            "count": len(distances),
            "min": float(min(distances)) if distances else 0.0,
            "mean": float(sum(distances) / float(len(distances))) if distances else 0.0,
            "max": float(max(distances)) if distances else 0.0,
        },
        "groups": group_records,
        "mapping": mapping,
    }


def disabled_founder_spatial_metrics() -> dict[str, Any]:
    return {
        "schema_version": "lineage_founder_spatial_separation_v1",
        "enabled": False,
        "assignment_mode": "none",
        "group_count": 0,
        "configured_group_count": 0,
        "min_distance": 0,
        "min_distance_satisfied": True,
        "warning": "",
        "founder_collision_index": 0.0,
        "pairwise_distance_summary": {"count": 0, "min": 0.0, "mean": 0.0, "max": 0.0},
        "groups": [],
        "mapping": [],
    }


def place_initial_founders(population: Any, env: Any, *, run_dir: str = "") -> dict[str, Any]:
    sim_cfg = getattr(population, "sim_cfg", None)
    if not founder_spatial_separation_enabled(sim_cfg):
        metrics = disabled_founder_spatial_metrics()
        setattr(population, "_founder_spatial_initial_metrics", metrics)
        return metrics
    plan = build_founder_spatial_plan(
        agents=getattr(population, "alive_agents")(),
        env_cfg=getattr(population, "env_cfg", getattr(env, "cfg", None)),
        sim_cfg=sim_cfg,
        env=env,
    )
    by_id = {int(row["agent_id"]): (int(row["y"]), int(row["x"])) for row in plan["mapping"]}
    for agent in getattr(population, "alive_agents")():
        env.add_agent(_agent_id(agent), pos=by_id[_agent_id(agent)])
    setattr(population, "_founder_spatial_initial_metrics", plan)
    if sim_cfg is not None:
        try:
            setattr(sim_cfg, "_lineage_founder_spatial_initial_metrics", plan)
        except Exception:
            pass
    if run_dir:
        write_founder_spatial_artifact(run_dir, plan)
    return plan


def write_founder_spatial_artifact(run_dir: str, plan: dict[str, Any]) -> None:
    try:
        out_dir = os.path.join(str(run_dir), "metrics")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "lineage_founder_spatial_initialization.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    except Exception:
        return


def founder_spatial_survival_record(*, generation: int, agents: Iterable[Any], initial_plan: Any = None) -> dict[str, Any]:
    if not isinstance(initial_plan, dict) or not initial_plan.get("enabled"):
        return {
            "schema_version": "lineage_founder_spatial_survival_v1",
            "generation": _as_int(generation, 0),
            "enabled": False,
            "groups": [],
        }
    initial_counts: Counter[str] = Counter()
    for row in initial_plan.get("mapping", []) or []:
        if isinstance(row, dict):
            initial_counts[str(row.get("lineage_id", "") or "")] += 1
    alive_counts: Counter[str] = Counter()
    for agent in agents or []:
        alive_counts[_lineage_id(agent)] += 1
    groups = []
    for lineage_id in sorted(initial_counts.keys()):
        start = int(initial_counts[lineage_id])
        alive = int(alive_counts.get(lineage_id, 0))
        groups.append(
            {
                "lineage_id": lineage_id,
                "initial_count": start,
                "alive_count": alive,
                "survival_fraction": float(min(alive, start)) / float(max(1, start)),
                "alive_fraction_relative_to_initial": float(alive) / float(max(1, start)),
            }
        )
    return {
        "schema_version": "lineage_founder_spatial_survival_v1",
        "generation": _as_int(generation, 0),
        "enabled": True,
        "groups": groups,
    }
