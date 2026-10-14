from __future__ import annotations

"""Compact reporting-only branch observability.

The writer is deliberately passive: it never draws RNG, never changes agent
state, and only emits compact rows when explicitly enabled.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from agents.agent import format_lineage_id, normalize_lineage_stats


SCHEMA_VERSION = "branch_observability_v1"
NICHE_SCHEMA_VERSION = "branch_niche_observability_v1"
BIRTH_EDGES_FILENAME = "branch_observability_birth_edges.jsonl"
LIFECYCLE_FILENAME = "branch_observability_agent_lifecycle.jsonl"
NICHE_LIFECYCLE_FILENAME = "branch_observability_niche_lifecycle.jsonl"
SUMMARY_FILENAME = "branch_observability_run_summary.json"


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _round(value: Any, places: int = 6) -> float:
    return round(_safe_float(value, 0.0), int(places))


def _counter() -> dict[str, int]:
    return {}


def _float_counter() -> dict[str, float]:
    return {}


def _counter_add(counter: dict[str, int], key: str, value: int = 1) -> None:
    counter[str(key)] = int(counter.get(str(key), 0)) + int(value)


def _float_counter_add(counter: dict[str, float], key: str, value: float) -> None:
    counter[str(key)] = float(counter.get(str(key), 0.0)) + float(value)


def _counter_merge(dst: dict[str, int], src: Mapping[str, Any] | None) -> None:
    for key, value in dict(src or {}).items():
        _counter_add(dst, str(key), _safe_int(value, 0))


def _float_counter_merge(dst: dict[str, float], src: Mapping[str, Any] | None) -> None:
    for key, value in dict(src or {}).items():
        _float_counter_add(dst, str(key), _safe_float(value, 0.0))


def _class_for_value(value: float, *, low: float, high: float, labels: tuple[str, str, str]) -> str:
    v = _safe_float(value, 0.0)
    if v >= float(high):
        return labels[2]
    if v <= float(low):
        return labels[0]
    return labels[1]


def _density_at(env: Any, y: int, x: int) -> float:
    try:
        grid = getattr(env, "agent_grid", None)
        cfg = getattr(env, "cfg", None)
        if grid is None:
            return 0.0
        radius = int(max(0, getattr(cfg, "patch_density_radius", 1)))
        include_self = bool(getattr(cfg, "patch_density_include_self", True))
        h, w = tuple(getattr(grid, "shape"))
        y0 = max(0, int(y) - radius)
        y1 = min(int(h), int(y) + radius + 1)
        x0 = max(0, int(x) - radius)
        x1 = min(int(w), int(x) + radius + 1)
        count = 0
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                if int(grid[yy, xx]) != -1:
                    count += 1
        if not include_self and int(grid[int(y), int(x)]) != -1:
            count -= 1
        if bool(getattr(cfg, "patch_density_normalize", False)):
            denom = float((2 * radius + 1) * (2 * radius + 1))
            if not include_self:
                denom -= 1.0
            if denom > 0.0:
                return float(count) / denom
        return float(count)
    except Exception:
        return 0.0


def _array_value(env: Any, name: str, y: int, x: int, default: float = 0.0) -> float:
    try:
        arr = getattr(env, name, None)
        if arr is None:
            return float(default)
        return _safe_float(arr[int(y), int(x)], default)
    except Exception:
        return float(default)


def niche_zone_at(env: Any, y: int, x: int) -> dict[str, Any]:
    """Return a deterministic compact ecological zone descriptor.

    This helper is read-only with respect to mechanics and avoids writing local
    density maps. It intentionally produces a small canonical class rather than
    a unique patch identifier.
    """
    try:
        if not bool(getattr(env, "patch_world_enable", False)):
            return {
                "zone_class": "flat",
                "zone_parts": ["flat"],
                "food": 0.0,
                "cover": 0.0,
                "shelter": 0.0,
                "waste": 0.0,
                "scent": 0.0,
                "density": 0.0,
                "movement_cost": 1.0,
                "recovery_activity": 0.0,
                "coverage": "flat_no_patch_world",
            }
        yy = max(0, min(int(getattr(env, "height", getattr(getattr(env, "cfg", None), "height", 0))) - 1, int(y)))
        xx = max(0, min(int(getattr(env, "width", getattr(getattr(env, "cfg", None), "width", 0))) - 1, int(x)))
    except Exception:
        return {"zone_class": "unknown", "zone_parts": ["unknown"], "coverage": "missing_position"}

    food = _array_value(env, "patch_food_amount", yy, xx)
    cover = _array_value(env, "patch_cover", yy, xx)
    shelter = _array_value(env, "patch_shelter", yy, xx)
    waste = _array_value(env, "patch_waste", yy, xx)
    scent = _array_value(env, "patch_scent", yy, xx)
    movement_cost = _array_value(env, "patch_movement_cost", yy, xx, 1.0)
    recovery = _array_value(env, "patch_recovery_activity", yy, xx)
    density = _density_at(env, yy, xx)
    parts = [
        _class_for_value(food, low=0.05, high=0.50, labels=("food_low", "food_mid", "food_high")),
        "shelter_high" if max(cover, shelter) >= 0.20 else "shelter_low",
        "waste_high" if waste >= 0.25 else "waste_low",
        "density_high" if density >= 4.0 else "density_low",
        "move_cost_high" if movement_cost > 1.25 else "move_cost_base",
        "recovery_high" if recovery >= 0.20 else "recovery_low",
    ]
    return {
        "zone_class": "|".join(parts),
        "zone_parts": parts,
        "food": _round(food),
        "cover": _round(cover),
        "shelter": _round(shelter),
        "waste": _round(waste),
        "scent": _round(scent),
        "density": _round(density),
        "movement_cost": _round(movement_cost),
        "recovery_activity": _round(recovery),
        "coverage": "patch_world",
    }


def _action_class(action: Any) -> str:
    try:
        action_i = int(action)
    except Exception:
        action_i = -1
    if action_i in {0, 1, 2, 3}:
        return "move"
    if action_i == 4:
        return "gather"
    if action_i == 5:
        return "rest"
    if action_i == 6:
        return "attack"
    return "unknown"


def _lineage(agent: Any, *, generation: int = 0, tick: int = 0) -> dict[str, Any]:
    aid = _safe_int(getattr(agent, "agent_id", 0), 0)
    stats = normalize_lineage_stats(getattr(agent, "stats", None), aid, generation=generation, tick=tick)
    if stats is None:
        return {
            "agent_id": aid,
            "founder_ancestry_id": format_lineage_id(aid),
            "biological_lineage_id": format_lineage_id(aid),
            "parent_ids": [],
            "birth_generation": int(generation),
            "birth_tick": int(tick),
            "lineage_depth": 0,
        }
    return {
        "agent_id": aid,
        "founder_ancestry_id": format_lineage_id(_safe_int(getattr(stats, "founder_id", aid), aid)),
        "biological_lineage_id": str(getattr(stats, "lineage_id", "") or format_lineage_id(aid)),
        "parent_ids": [_safe_int(x, -1) for x in tuple(getattr(stats, "parent_ids", ()) or ())],
        "birth_generation": _safe_int(getattr(stats, "birth_generation", generation), generation),
        "birth_tick": _safe_int(getattr(stats, "birth_tick", tick), tick),
        "lineage_depth": _safe_int(getattr(stats, "lineage_depth", 0), 0),
    }


def genome_signature(agent: Any) -> str:
    genome = getattr(agent, "genome", None)
    if genome is None:
        return ""
    for name in ("canonical_hash",):
        fn = getattr(genome, name, None)
        if callable(fn):
            try:
                return str(fn())
            except Exception:
                pass
    for name in ("to_dict", "to_raw_dict"):
        fn = getattr(genome, name, None)
        if callable(fn):
            try:
                return _stable_hash(fn())
            except Exception:
                pass
    return _stable_hash(str(genome))


def descriptor_payload(agent: Any) -> dict[str, Any]:
    genome = getattr(agent, "genome", None)
    stats = getattr(agent, "stats", None)
    memory_dim = _safe_int(getattr(genome, "memory_dim", 0) if genome is not None else 0, 0)
    payload = {
        "genome_signature": genome_signature(agent),
        "behavior_signature": _stable_hash(
            {
                "exploration_bias": round(_safe_float(getattr(genome, "exploration_bias", 0.0), 0.0), 6),
                "aggression_bias": round(_safe_float(getattr(genome, "aggression_bias", 0.0), 0.0), 6),
                "action_temperature": round(_safe_float(getattr(genome, "action_temperature", 1.0), 1.0), 6),
            }
        ),
        "ecology_signature": _stable_hash(
            {
                "energy_collected": round(_safe_float(getattr(stats, "energy_collected", 0.0), 0.0), 6),
                "health": round(_safe_float(getattr(agent, "health", 0.0), 0.0), 6),
                "injury": round(_safe_float(getattr(agent, "injury", 0.0), 0.0), 6),
            }
        ),
        "reproduction_signature": _stable_hash(
            {
                "offspring_count": _safe_int(getattr(stats, "offspring_count", 0) if stats is not None else 0, 0),
                "first_repro_step": _safe_int(getattr(stats, "first_repro_step", -1) if stats is not None else -1, -1),
                "reproduction_threshold": round(
                    _safe_float(getattr(genome, "reproduction_threshold", 0.0) if genome is not None else 0.0, 0.0), 6
                ),
            }
        ),
        "cognition_signature": _stable_hash(
            {
                "memory_dim": memory_dim,
                "use_gru": bool(getattr(genome, "use_gru", False) if genome is not None else False),
                "sensor_channel_mask": _safe_int(getattr(genome, "sensor_channel_mask", 0) if genome is not None else 0, 0),
            }
        ),
        "strategy_id": "",
        "memory_dim": memory_dim,
        "sensor_radius": _safe_int(getattr(genome, "sensor_radius", 0) if genome is not None else 0, 0),
    }
    payload["strategy_id"] = "strategy_" + _stable_hash(
        {k: payload[k] for k in ("behavior_signature", "ecology_signature", "reproduction_signature", "cognition_signature")}
    )[:10]
    return payload


class BranchObservabilityWriter:
    def __init__(self, sim: Any) -> None:
        cfg = getattr(sim, "sim_cfg", None)
        self.branch_enabled = bool(getattr(cfg, "branch_observability_enable", False))
        self.niche_enabled = bool(getattr(cfg, "branch_niche_observability_enable", False))
        self.enabled = bool(self.branch_enabled or self.niche_enabled)
        self.birth_edges_enable = bool(getattr(cfg, "branch_observability_birth_edges_enable", True))
        self.lifecycle_enable = bool(getattr(cfg, "branch_observability_lifecycle_enable", True))
        self.descriptor_enable = bool(getattr(cfg, "branch_observability_descriptor_snapshot_enable", True))
        self.output_profile = str(getattr(cfg, "branch_observability_output_profile", "compact_v1") or "compact_v1")
        self.warning_row_threshold = max(0, _safe_int(getattr(cfg, "branch_observability_warning_row_threshold", 100000), 100000))
        self.niche_lifecycle_enable = bool(getattr(cfg, "branch_niche_lifecycle_enable", True))
        self.niche_strategy_compression_enable = bool(getattr(cfg, "branch_niche_strategy_compression_enable", True))
        self.niche_output_profile = str(getattr(cfg, "branch_niche_output_profile", "compact_v1") or "compact_v1")
        self.niche_warning_row_threshold = max(0, _safe_int(getattr(cfg, "branch_niche_warning_row_threshold", 100000), 100000))
        self.run_id = str(getattr(cfg, "branch_observability_run_id", "") or getattr(cfg, "campaign_id", "") or Path(str(getattr(sim, "run_dir", ""))).name)
        self.candidate_a_enabled = bool(getattr(cfg, "lro1_reproductive_opportunity_enable", False))
        self.candidate_b_enabled = bool(getattr(cfg, "lro4_offspring_dispersal_enable", False))
        self.root = Path(str(getattr(sim, "run_dir", ""))) / "branch_observability"
        self._birth_edges: list[dict[str, Any]] = []
        self._birth_event_offset = len(self._read_jsonl(self.root / BIRTH_EDGES_FILENAME))
        self._seen_agents: set[int] = set()
        self._agent_snapshots: dict[int, dict[str, Any]] = self._load_lifecycle_snapshots(self.root / LIFECYCLE_FILENAME)
        self._niche_counters: dict[int, dict[str, Any]] = self._load_niche_snapshots(self.root / NICHE_LIFECYCLE_FILENAME)
        self._niche_tick_zone: dict[int, str] = {}
        self._finalized = False

    @classmethod
    def maybe_create(cls, sim: Any) -> "BranchObservabilityWriter | None":
        cfg = getattr(sim, "sim_cfg", None)
        if not (
            bool(getattr(cfg, "branch_observability_enable", False))
            or bool(getattr(cfg, "branch_niche_observability_enable", False))
        ):
            return None
        return cls(sim)

    def _niche_counter_for(self, agent: Any | int, *, generation: int = 0, tick: int = 0) -> dict[str, Any]:
        aid = _safe_int(getattr(agent, "agent_id", agent), -1)
        row = self._niche_counters.get(aid)
        if row is None:
            line = _lineage(agent, generation=generation, tick=tick) if not isinstance(agent, int) else {"agent_id": aid, "founder_ancestry_id": "", "biological_lineage_id": "", "parent_ids": [], "birth_generation": generation, "birth_tick": tick, "lineage_depth": 0}
            row = {
                "agent_id": aid,
                "founder_ancestry_id": str(line.get("founder_ancestry_id", "")),
                "biological_lineage_id": str(line.get("biological_lineage_id", "")),
                "parent_ids": list(line.get("parent_ids", []) or []),
                "birth_generation": _safe_int(line.get("birth_generation", generation), generation),
                "birth_tick": _safe_int(line.get("birth_tick", tick), tick),
                "lineage_depth": _safe_int(line.get("lineage_depth", 0), 0),
                "total_observed_ticks": 0,
                "zone_exposure_counts": _counter(),
                "zone_action_counts": _counter(),
                "movement_actions_by_zone": _counter(),
                "stay_actions_by_zone": _counter(),
                "gather_actions_by_zone": _counter(),
                "food_available_sum_by_zone": _float_counter(),
                "food_collected_by_zone": _float_counter(),
                "damage_by_zone": _float_counter(),
                "injury_by_zone": _float_counter(),
                "recovery_by_zone": _float_counter(),
                "successful_births_by_zone": _counter(),
                "offspring_zone_counts": _counter(),
                "transitions_between_zone_classes": _counter(),
                "shelter_cover_exposure_ticks": 0,
                "hazard_exposure_ticks": 0,
                "local_density_sum_by_zone": _float_counter(),
                "waste_sum_by_zone": _float_counter(),
                "scent_sum_by_zone": _float_counter(),
                "missing_data_flags": set(),
                "_last_zone_class": "",
            }
            self._niche_counters[aid] = row
        return row

    def record_niche_pre_env(self, sim: Any, *, generation: int, tick: int, alive: Iterable[Any], agent_ids_np: Any, actions_np: Any) -> None:
        if not (self.enabled and self.niche_enabled and self.niche_lifecycle_enable):
            return
        env = getattr(sim, "env", None)
        if env is None:
            return
        try:
            pos_y = getattr(env, "agent_pos_y")
            pos_x = getattr(env, "agent_pos_x")
            ids = list(getattr(agent_ids_np, "tolist", lambda: list(agent_ids_np))())
            actions = list(getattr(actions_np, "tolist", lambda: list(actions_np))())
        except Exception:
            return
        by_id = {_safe_int(getattr(agent, "agent_id", -1), -1): agent for agent in list(alive or [])}
        for aid_raw, action_raw in zip(ids, actions):
            aid = _safe_int(aid_raw, -1)
            agent = by_id.get(aid, aid)
            try:
                y = int(pos_y[aid])
                x = int(pos_x[aid])
            except Exception:
                continue
            if y < 0 or x < 0:
                continue
            zone = niche_zone_at(env, y, x)
            zone_class = str(zone.get("zone_class", "unknown"))
            action_class = _action_class(action_raw)
            rec = self._niche_counter_for(agent, generation=generation, tick=tick)
            rec["total_observed_ticks"] = int(rec.get("total_observed_ticks", 0)) + 1
            _counter_add(rec["zone_exposure_counts"], zone_class)
            _counter_add(rec["zone_action_counts"], f"{zone_class}:{action_class}")
            if action_class == "move":
                _counter_add(rec["movement_actions_by_zone"], zone_class)
            elif action_class == "gather":
                _counter_add(rec["gather_actions_by_zone"], zone_class)
            else:
                _counter_add(rec["stay_actions_by_zone"], zone_class)
            _float_counter_add(rec["food_available_sum_by_zone"], zone_class, _safe_float(zone.get("food"), 0.0))
            _float_counter_add(rec["local_density_sum_by_zone"], zone_class, _safe_float(zone.get("density"), 0.0))
            _float_counter_add(rec["waste_sum_by_zone"], zone_class, _safe_float(zone.get("waste"), 0.0))
            _float_counter_add(rec["scent_sum_by_zone"], zone_class, _safe_float(zone.get("scent"), 0.0))
            if max(_safe_float(zone.get("cover"), 0.0), _safe_float(zone.get("shelter"), 0.0)) >= 0.20:
                rec["shelter_cover_exposure_ticks"] = int(rec.get("shelter_cover_exposure_ticks", 0)) + 1
            if str(zone.get("coverage", "")) != "patch_world":
                rec["missing_data_flags"].add(str(zone.get("coverage", "missing_patch_world")))
            last = str(rec.get("_last_zone_class", "") or "")
            if last and last != zone_class:
                _counter_add(rec["transitions_between_zone_classes"], f"{last}->{zone_class}")
            rec["_last_zone_class"] = zone_class
            self._niche_tick_zone[aid] = zone_class

    def record_niche_post_env(self, *, agent_ids_np: Any, food_gained: Any, pre_health: Any, new_health: Any, pre_injury: Any, new_injury: Any) -> None:
        if not (self.enabled and self.niche_enabled and self.niche_lifecycle_enable):
            return
        try:
            ids = list(getattr(agent_ids_np, "tolist", lambda: list(agent_ids_np))())
        except Exception:
            return
        for i, aid_raw in enumerate(ids):
            aid = _safe_int(aid_raw, -1)
            if aid < 0:
                continue
            zone_class = str(self._niche_tick_zone.get(aid, "unknown"))
            rec = self._niche_counter_for(aid)
            try:
                fg = _safe_float(food_gained[i], 0.0)
            except Exception:
                fg = 0.0
            try:
                damage = max(0.0, _safe_float(pre_health[i], 0.0) - _safe_float(new_health[i], 0.0))
            except Exception:
                damage = 0.0
            try:
                injury = max(0.0, _safe_float(new_injury[i], 0.0) - _safe_float(pre_injury[i], 0.0))
            except Exception:
                injury = 0.0
            if fg > 0.0:
                _float_counter_add(rec["food_collected_by_zone"], zone_class, fg)
            if damage > 0.0:
                _float_counter_add(rec["damage_by_zone"], zone_class, damage)
            if injury > 0.0:
                _float_counter_add(rec["injury_by_zone"], zone_class, injury)

    def _record_agent_seen(self, agent: Any, *, generation: int = 0, tick: int = 0, kind: str = "observed") -> None:
        aid = _safe_int(getattr(agent, "agent_id", -1), -1)
        if aid >= 0:
            self._seen_agents.add(aid)
            self._agent_snapshots[aid] = self._snapshot_agent(agent, generation=generation, tick=tick, kind=kind)

    def _snapshot_agent(self, agent: Any, *, generation: int, tick: int, kind: str) -> dict[str, Any]:
        line = _lineage(agent, generation=generation, tick=tick)
        stats = getattr(agent, "stats", None)
        row: dict[str, Any] = {
            "agent_id": line["agent_id"],
            "founder_ancestry_id": line["founder_ancestry_id"],
            "biological_lineage_id": line["biological_lineage_id"],
            "parent_ids": line["parent_ids"],
            "birth_generation": line["birth_generation"],
            "birth_tick": line["birth_tick"],
            "last_observed_generation": int(generation),
            "last_observed_tick": int(tick),
            "snapshot_kind": str(kind),
            "alive": bool(getattr(agent, "alive", False)),
            "lifetime_ticks": _safe_int(getattr(stats, "survival_steps", 0) if stats is not None else 0, 0),
            "final_genome_signature": genome_signature(agent),
            "successful_children_count": _safe_int(getattr(stats, "offspring_count", 0) if stats is not None else 0, 0),
            "successful_births": _safe_int(getattr(stats, "offspring_count", 0) if stats is not None else 0, 0),
            "movement_summary": {"survival_steps": _safe_int(getattr(stats, "survival_steps", 0) if stats is not None else 0, 0)},
            "patch_use_summary": {"coverage": "not_available"},
            "food_use_summary": {"energy_collected": _safe_float(getattr(stats, "energy_collected", 0.0) if stats is not None else 0.0, 0.0)},
            "damage_summary": {"injury": _safe_float(getattr(agent, "injury", 0.0), 0.0), "health": _safe_float(getattr(agent, "health", 0.0), 0.0)},
        }
        if self.descriptor_enable:
            row.update(descriptor_payload(agent))
        return row

    def register_founders(self, agents: Iterable[Any], *, generation: int, tick: int) -> None:
        if not self.enabled:
            return
        for agent in sorted(list(agents or []), key=lambda a: _safe_int(getattr(a, "agent_id", 0), 0)):
            self._record_agent_seen(agent, generation=generation, tick=tick, kind="founder")

    def record_successful_birth(
        self,
        *,
        generation: int,
        tick: int,
        child: Any,
        primary_parent: Any,
        secondary_parent: Any | None,
        child_pos: tuple[int, int] | None = None,
        placement_fields: Mapping[str, Any] | None = None,
        env: Any | None = None,
    ) -> None:
        if not (self.enabled and self.birth_edges_enable):
            return
        self._record_agent_seen(child, generation=generation, tick=tick, kind="birth")
        self._record_agent_seen(primary_parent, generation=generation, tick=tick, kind="parent")
        if secondary_parent is not None:
            self._record_agent_seen(secondary_parent, generation=generation, tick=tick, kind="secondary_parent")
        child_lineage = _lineage(child, generation=generation, tick=tick)
        primary_lineage = _lineage(primary_parent, generation=generation, tick=tick)
        secondary_lineage = _lineage(secondary_parent, generation=generation, tick=tick) if secondary_parent is not None else {}
        child_id = _safe_int(getattr(child, "agent_id", -1), -1)
        parent_id = _safe_int(getattr(primary_parent, "agent_id", -1), -1)
        secondary_id = _safe_int(getattr(secondary_parent, "agent_id", -1), -1) if secondary_parent is not None else None
        birth_ordinal = self._birth_event_offset + len(self._birth_edges) + 1
        event_basis = {
            "run_id": self.run_id,
            "generation": int(generation),
            "tick": int(tick),
            "birth_ordinal": int(birth_ordinal),
            "child_agent_id": int(child_id),
            "primary_parent_agent_id": int(parent_id),
            "secondary_parent_agent_id": secondary_id,
        }
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "event_type": "birth_success",
            "run_id": self.run_id,
            "birth_event_id": "bo_birth_" + _stable_hash(event_basis)[:18],
            "generation": int(generation),
            "tick": int(tick),
            "child_agent_id": int(child_id),
            "primary_parent_agent_id": int(parent_id),
            "parent_agent_id": int(parent_id),
            "secondary_parent_agent_id": secondary_id,
            "primary_parent_founder_ancestry_id": primary_lineage.get("founder_ancestry_id", ""),
            "secondary_parent_founder_ancestry_id": secondary_lineage.get("founder_ancestry_id") if secondary_parent is not None else None,
            "parent_biological_lineage_id": primary_lineage.get("biological_lineage_id", ""),
            "child_founder_ancestry_id": child_lineage.get("founder_ancestry_id", ""),
            "child_biological_lineage_id": child_lineage.get("biological_lineage_id", ""),
            "child_lineage_depth": int(child_lineage.get("lineage_depth", 0) or 0),
            "deterministic_child_genome_signature": genome_signature(child),
            "deterministic_parent_genome_signature": genome_signature(primary_parent),
            "mutation_event_id": None,
            "mutation_event_coverage": "not_available",
            "birth_context": "live_tick_reproduction_success",
            "child_y": int(child_pos[0]) if child_pos is not None else None,
            "child_x": int(child_pos[1]) if child_pos is not None else None,
            "candidate_a_enabled": int(self.candidate_a_enabled),
            "candidate_b_enabled": int(self.candidate_b_enabled),
            "successful_placement_radius": placement_fields.get("lro4_offspring_dispersal_selected_radius") if placement_fields else None,
        }
        if self.descriptor_enable:
            row.update(descriptor_payload(child))
        self._birth_edges.append(row)
        self.root.mkdir(parents=True, exist_ok=True)
        self._append_jsonl(self.root / BIRTH_EDGES_FILENAME, row)
        if self.niche_enabled and env is not None:
            try:
                self._niche_counter_for(child, generation=generation, tick=tick)
                parent_zone = str(self._niche_tick_zone.get(parent_id, "unknown"))
                parent_rec = self._niche_counter_for(primary_parent, generation=generation, tick=tick)
                _counter_add(parent_rec["successful_births_by_zone"], parent_zone)
                if child_pos is not None:
                    child_zone = str(niche_zone_at(env, int(child_pos[0]), int(child_pos[1])).get("zone_class", "unknown"))
                    _counter_add(parent_rec["offspring_zone_counts"], child_zone)
            except Exception:
                pass

    def finalize(self, sim: Any, *, generation: int, tick: int, status: str = "complete") -> None:
        if not self.enabled or self._finalized:
            return
        self._finalized = True
        self.root.mkdir(parents=True, exist_ok=True)
        agents = sorted(list(getattr(getattr(sim, "pop", None), "agents", {}).values()), key=lambda a: _safe_int(getattr(a, "agent_id", 0), 0))
        for agent in agents:
            self._record_agent_seen(agent, generation=generation, tick=tick, kind="run_end")
        if self.birth_edges_enable:
            edge_path = self.root / BIRTH_EDGES_FILENAME
            rows_by_id = {
                str(row.get("birth_event_id", "")): row
                for row in self._read_jsonl(edge_path)
                if str(row.get("birth_event_id", ""))
            }
            for row in self._birth_edges:
                rows_by_id[str(row.get("birth_event_id", ""))] = row
            self._write_jsonl(
                edge_path,
                sorted(rows_by_id.values(), key=lambda row: (int(row.get("generation", 0) or 0), int(row.get("tick", 0) or 0), str(row.get("birth_event_id", "")))),
            )
        lifecycle_rows = []
        if self.lifecycle_enable:
            live_by_id = {_safe_int(getattr(agent, "agent_id", -1), -1): agent for agent in agents}
            for agent_id in sorted(set(self._agent_snapshots) | set(live_by_id)):
                agent = live_by_id.get(agent_id)
                lifecycle_rows.append(
                    self._lifecycle_row(
                        agent,
                        snapshot=self._agent_snapshots.get(agent_id),
                        generation=generation,
                        tick=tick,
                    )
                )
            self._write_jsonl(self.root / LIFECYCLE_FILENAME, lifecycle_rows)
        niche_rows = []
        if self.niche_enabled and self.niche_lifecycle_enable:
            for agent in agents:
                self._niche_counter_for(agent, generation=generation, tick=tick)
            for agent_id in sorted(self._niche_counters):
                niche_rows.append(self._niche_lifecycle_row(agent_id, generation=generation, tick=tick))
            self._write_jsonl(self.root / NICHE_LIFECYCLE_FILENAME, niche_rows)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": str(status),
            "output_profile": self.output_profile,
            "birth_edge_rows": len(self._birth_edges),
            "lifecycle_rows": len(lifecycle_rows),
            "estimated_rows": len(self._birth_edges) + len(lifecycle_rows),
            "warning_row_threshold": int(self.warning_row_threshold),
            "storage_warning": bool(self.warning_row_threshold and (len(self._birth_edges) + len(lifecycle_rows)) > self.warning_row_threshold),
            "birth_edges_file": BIRTH_EDGES_FILENAME if self.birth_edges_enable else "",
            "lifecycle_file": LIFECYCLE_FILENAME if self.lifecycle_enable else "",
            "niche_lifecycle_file": NICHE_LIFECYCLE_FILENAME if self.niche_enabled and self.niche_lifecycle_enable else "",
            "niche_lifecycle_rows": len(niche_rows),
            "branch_niche_observability_enable": int(self.niche_enabled),
            "mechanics_effect": "none_reporting_only",
        }
        (self.root / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _niche_lifecycle_row(self, agent_id: int, *, generation: int, tick: int) -> dict[str, Any]:
        rec = self._niche_counters.get(int(agent_id), {})
        ticks = max(0, _safe_int(rec.get("total_observed_ticks", 0), 0))
        exposure = dict(rec.get("zone_exposure_counts", {}) or {})
        top_zone = ""
        if exposure:
            top_zone = sorted(exposure.items(), key=lambda item: (-int(item[1]), str(item[0])))[0][0]
        movement_total = sum(_safe_int(v, 0) for v in dict(rec.get("movement_actions_by_zone", {}) or {}).values())
        gather_total = sum(_safe_int(v, 0) for v in dict(rec.get("gather_actions_by_zone", {}) or {}).values())
        transition_total = sum(_safe_int(v, 0) for v in dict(rec.get("transitions_between_zone_classes", {}) or {}).values())
        births_total = sum(_safe_int(v, 0) for v in dict(rec.get("successful_births_by_zone", {}) or {}).values())
        missing = sorted(str(x) for x in rec.get("missing_data_flags", set()) or [])
        return {
            "schema_version": NICHE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "agent_id": int(agent_id),
            "founder_ancestry_id": str(rec.get("founder_ancestry_id", "")),
            "biological_lineage_id": str(rec.get("biological_lineage_id", "")),
            "parent_ids": list(rec.get("parent_ids", []) or []),
            "birth_generation": _safe_int(rec.get("birth_generation", 0), 0),
            "birth_tick": _safe_int(rec.get("birth_tick", 0), 0),
            "final_generation": int(generation),
            "final_tick": int(tick),
            "lineage_depth": _safe_int(rec.get("lineage_depth", 0), 0),
            "total_observed_ticks": int(ticks),
            "dominant_zone_class": top_zone,
            "zone_exposure_counts": exposure,
            "zone_action_counts": dict(rec.get("zone_action_counts", {}) or {}),
            "movement_actions_by_zone": dict(rec.get("movement_actions_by_zone", {}) or {}),
            "stay_actions_by_zone": dict(rec.get("stay_actions_by_zone", {}) or {}),
            "gather_actions_by_zone": dict(rec.get("gather_actions_by_zone", {}) or {}),
            "food_available_sum_by_zone": {k: _round(v) for k, v in dict(rec.get("food_available_sum_by_zone", {}) or {}).items()},
            "food_collected_by_zone": {k: _round(v) for k, v in dict(rec.get("food_collected_by_zone", {}) or {}).items()},
            "damage_by_zone": {k: _round(v) for k, v in dict(rec.get("damage_by_zone", {}) or {}).items()},
            "injury_by_zone": {k: _round(v) for k, v in dict(rec.get("injury_by_zone", {}) or {}).items()},
            "recovery_by_zone": {k: _round(v) for k, v in dict(rec.get("recovery_by_zone", {}) or {}).items()},
            "successful_births_by_zone": dict(rec.get("successful_births_by_zone", {}) or {}),
            "offspring_zone_counts": dict(rec.get("offspring_zone_counts", {}) or {}),
            "transitions_between_zone_classes": dict(rec.get("transitions_between_zone_classes", {}) or {}),
            "local_density_sum_by_zone": {k: _round(v) for k, v in dict(rec.get("local_density_sum_by_zone", {}) or {}).items()},
            "waste_sum_by_zone": {k: _round(v) for k, v in dict(rec.get("waste_sum_by_zone", {}) or {}).items()},
            "scent_sum_by_zone": {k: _round(v) for k, v in dict(rec.get("scent_sum_by_zone", {}) or {}).items()},
            "shelter_cover_exposure_ticks": _safe_int(rec.get("shelter_cover_exposure_ticks", 0), 0),
            "hazard_exposure_ticks": _safe_int(rec.get("hazard_exposure_ticks", 0), 0),
            "movement_action_rate": _round(float(movement_total) / max(1, ticks)),
            "gather_action_rate": _round(float(gather_total) / max(1, ticks)),
            "transition_rate": _round(float(transition_total) / max(1, ticks)),
            "successful_birth_rate": _round(float(births_total) / max(1, ticks)),
            "missing_data_coverage": missing,
            "run_completion_status": "complete",
        }

    def _lifecycle_row(self, agent: Any | None, *, snapshot: Mapping[str, Any] | None, generation: int, tick: int) -> dict[str, Any]:
        if agent is None:
            snap = dict(snapshot or {})
            row: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "agent_id": _safe_int(snap.get("agent_id", -1), -1),
                "founder_ancestry_id": str(snap.get("founder_ancestry_id", "")),
                "biological_lineage_id": str(snap.get("biological_lineage_id", "")),
                "parent_ids": list(snap.get("parent_ids", []) or []),
                "birth_generation": _safe_int(snap.get("birth_generation", 0), 0),
                "birth_tick": _safe_int(snap.get("birth_tick", 0), 0),
                "final_generation": int(generation),
                "final_tick": int(tick),
                "alive_at_run_end": False,
                "lifetime_ticks": _safe_int(snap.get("lifetime_ticks", 0), 0),
                "final_genome_signature": str(snap.get("final_genome_signature", "")),
                "successful_children_count": _safe_int(snap.get("successful_children_count", 0), 0),
                "reproduction_attempts": None,
                "successful_births": _safe_int(snap.get("successful_births", 0), 0),
                "movement_summary": dict(snap.get("movement_summary", {}) or {}),
                "patch_use_summary": dict(snap.get("patch_use_summary", {}) or {"coverage": "not_available"}),
                "food_use_summary": dict(snap.get("food_use_summary", {}) or {}),
                "damage_summary": dict(snap.get("damage_summary", {}) or {}),
                "missing_descriptor_coverage_flags": ["not_live_at_run_end_snapshot_only"],
            }
            for key in (
                "genome_signature",
                "behavior_signature",
                "ecology_signature",
                "reproduction_signature",
                "cognition_signature",
                "strategy_id",
                "memory_dim",
                "sensor_radius",
            ):
                if key in snap:
                    row[key] = snap[key]
            return row

        line = _lineage(agent, generation=generation, tick=tick)
        stats = getattr(agent, "stats", None)
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "agent_id": line["agent_id"],
            "founder_ancestry_id": line["founder_ancestry_id"],
            "biological_lineage_id": line["biological_lineage_id"],
            "parent_ids": line["parent_ids"],
            "birth_generation": line["birth_generation"],
            "birth_tick": line["birth_tick"],
            "final_generation": int(generation),
            "final_tick": int(tick),
            "alive_at_run_end": bool(getattr(agent, "alive", False)),
            "lifetime_ticks": _safe_int(getattr(stats, "survival_steps", 0) if stats is not None else 0, 0),
            "final_genome_signature": genome_signature(agent),
            "successful_children_count": _safe_int(getattr(stats, "offspring_count", 0) if stats is not None else 0, 0),
            "reproduction_attempts": None,
            "successful_births": _safe_int(getattr(stats, "offspring_count", 0) if stats is not None else 0, 0),
            "movement_summary": {"survival_steps": _safe_int(getattr(stats, "survival_steps", 0) if stats is not None else 0, 0)},
            "patch_use_summary": {"coverage": "not_available"},
            "food_use_summary": {"energy_collected": _safe_float(getattr(stats, "energy_collected", 0.0) if stats is not None else 0.0, 0.0)},
            "damage_summary": {"injury": _safe_float(getattr(agent, "injury", 0.0), 0.0), "health": _safe_float(getattr(agent, "health", 0.0), 0.0)},
            "missing_descriptor_coverage_flags": [],
        }
        if self.descriptor_enable:
            row.update(descriptor_payload(agent))
        return row

    @staticmethod
    def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_stable_json(dict(row)) + "\n")
        os.replace(tmp, path)

    @staticmethod
    def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(_stable_json(dict(row)) + "\n")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
        return rows

    @classmethod
    def _load_lifecycle_snapshots(cls, path: Path) -> dict[int, dict[str, Any]]:
        snapshots: dict[int, dict[str, Any]] = {}
        for row in cls._read_jsonl(path):
            agent_id = _safe_int(row.get("agent_id", -1), -1)
            if agent_id < 0:
                continue
            snap = dict(row)
            snap.setdefault("last_observed_generation", snap.get("final_generation", snap.get("birth_generation", 0)))
            snap.setdefault("last_observed_tick", snap.get("final_tick", snap.get("birth_tick", 0)))
            snap.setdefault("snapshot_kind", "loaded_lifecycle")
            snapshots[agent_id] = snap
        return snapshots

    @classmethod
    def _load_niche_snapshots(cls, path: Path) -> dict[int, dict[str, Any]]:
        counters: dict[int, dict[str, Any]] = {}
        for row in cls._read_jsonl(path):
            agent_id = _safe_int(row.get("agent_id", -1), -1)
            if agent_id < 0:
                continue
            rec = {
                "agent_id": agent_id,
                "founder_ancestry_id": str(row.get("founder_ancestry_id", "")),
                "biological_lineage_id": str(row.get("biological_lineage_id", "")),
                "parent_ids": list(row.get("parent_ids", []) or []),
                "birth_generation": _safe_int(row.get("birth_generation", 0), 0),
                "birth_tick": _safe_int(row.get("birth_tick", 0), 0),
                "lineage_depth": _safe_int(row.get("lineage_depth", 0), 0),
                "total_observed_ticks": _safe_int(row.get("total_observed_ticks", 0), 0),
                "zone_exposure_counts": _counter(),
                "zone_action_counts": _counter(),
                "movement_actions_by_zone": _counter(),
                "stay_actions_by_zone": _counter(),
                "gather_actions_by_zone": _counter(),
                "food_available_sum_by_zone": _float_counter(),
                "food_collected_by_zone": _float_counter(),
                "damage_by_zone": _float_counter(),
                "injury_by_zone": _float_counter(),
                "recovery_by_zone": _float_counter(),
                "successful_births_by_zone": _counter(),
                "offspring_zone_counts": _counter(),
                "transitions_between_zone_classes": _counter(),
                "local_density_sum_by_zone": _float_counter(),
                "waste_sum_by_zone": _float_counter(),
                "scent_sum_by_zone": _float_counter(),
                "shelter_cover_exposure_ticks": _safe_int(row.get("shelter_cover_exposure_ticks", 0), 0),
                "hazard_exposure_ticks": _safe_int(row.get("hazard_exposure_ticks", 0), 0),
                "missing_data_flags": set(row.get("missing_data_coverage", []) or []),
                "_last_zone_class": str(row.get("dominant_zone_class", "") or ""),
            }
            for key in (
                "zone_exposure_counts",
                "zone_action_counts",
                "movement_actions_by_zone",
                "stay_actions_by_zone",
                "gather_actions_by_zone",
                "successful_births_by_zone",
                "offspring_zone_counts",
                "transitions_between_zone_classes",
            ):
                _counter_merge(rec[key], row.get(key, {}))
            for key in (
                "food_available_sum_by_zone",
                "food_collected_by_zone",
                "damage_by_zone",
                "injury_by_zone",
                "recovery_by_zone",
                "local_density_sum_by_zone",
                "waste_sum_by_zone",
                "scent_sum_by_zone",
            ):
                _float_counter_merge(rec[key], row.get(key, {}))
            counters[agent_id] = rec
        return counters


def branch_observability_schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "files": [BIRTH_EDGES_FILENAME, LIFECYCLE_FILENAME, NICHE_LIFECYCLE_FILENAME, SUMMARY_FILENAME],
        "default_off": True,
        "row_bounds": "one birth-edge row per successful child and one lifecycle row per agent",
        "niche_row_bounds": "one compact niche lifecycle row per observed agent when branch_niche_observability_enable=true",
        "mechanics_effect": "none_reporting_only",
    }
