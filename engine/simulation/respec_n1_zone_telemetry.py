from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from agents.resource_handling import RESOURCE_HANDLING_AXIS_FIELD, clamp_resource_handling_axis


SCHEMA_VERSION = "respec_n1_zone_telemetry_v1"
ZONE_A = "resource_a_zone"
ZONE_B = "resource_b_zone"
ZONE_NEUTRAL = "neutral_zone"
RESOURCE_NONE = 0
RESOURCE_A = 1
RESOURCE_B = 2


def enabled(obj: Any) -> bool:
    cfg = getattr(obj, "env_cfg", None) or getattr(obj, "cfg", None) or obj
    return bool(getattr(cfg, "respec_n1_zone_telemetry_enable", False))


def _float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


def _axis_arm(agent: Any) -> str:
    genome = getattr(agent, "genome", None)
    axis = clamp_resource_handling_axis(getattr(genome, RESOURCE_HANDLING_AXIS_FIELD, 0.0))
    if axis > 1e-9:
        return "a_favored"
    if axis < -1e-9:
        return "b_favored"
    return "neutral"


def _axis_value(agent: Any) -> float:
    genome = getattr(agent, "genome", None)
    return float(clamp_resource_handling_axis(getattr(genome, RESOURCE_HANDLING_AXIS_FIELD, 0.0)))


def _lineage_id(agent: Any) -> str:
    stats = getattr(agent, "stats", None)
    value = getattr(stats, "lineage_id", "") if stats is not None else ""
    if value:
        return str(value)
    try:
        return f"founder:{int(getattr(agent, 'agent_id', -1))}"
    except Exception:
        return "unknown"


def _age_steps(agent: Any) -> int:
    stats = getattr(agent, "stats", None)
    try:
        return int(getattr(stats, "survival_steps", 0) if stats is not None else 0)
    except Exception:
        return 0


def _zone_for_class(resource_class: int) -> str:
    if int(resource_class) == RESOURCE_A:
        return ZONE_A
    if int(resource_class) == RESOURCE_B:
        return ZONE_B
    return ZONE_NEUTRAL


def _zone_at(env: Any, y: int, x: int) -> str:
    try:
        classes = np.asarray(getattr(env, "respec0b_resource_class"))
        if y < 0 or x < 0 or y >= int(classes.shape[0]) or x >= int(classes.shape[1]):
            return ZONE_NEUTRAL
        return _zone_for_class(int(classes[int(y), int(x)]))
    except Exception:
        return ZONE_NEUTRAL


def _run_dir(runner: Any) -> Path:
    return Path(str(getattr(runner, "run_dir", "") or getattr(runner, "run_dir_path", "") or "."))


def _state(runner: Any) -> dict[str, Any]:
    state = getattr(runner, "_respec_n1_zone_telemetry_state", None)
    if isinstance(state, dict):
        return state
    state = {
        "schema_version": SCHEMA_VERSION,
        "by_arm_zone": defaultdict(lambda: defaultdict(float)),
        "by_lineage_zone": defaultdict(lambda: defaultdict(float)),
        "unique_agents": defaultdict(set),
        "unique_eligible_parent_agents": defaultdict(set),
        "unique_selected_parent_agents": defaultdict(set),
        "first_entry_generation": {},
        "current_zone": {},
        "current_dwell": defaultdict(int),
        "dwell_durations": defaultdict(list),
        "last_resource_success": {},
        "generation_rows": [],
        "history_rows": [],
        "capacity_budget_rows": [],
    }
    setattr(runner, "_respec_n1_zone_telemetry_state", state)
    return state


def _add(counter: Mapping[str, Any] | defaultdict, key: tuple[str, str], field: str, value: float = 1.0) -> None:
    counter[key][field] += float(value)


def _serial_counter(counter: Mapping[tuple[str, str], Mapping[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, fields in sorted(counter.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        out[f"{key[0]}|{key[1]}"] = {str(k): float(v) for k, v in sorted(fields.items())}
    return out


def layout_digest(env: Any) -> dict[str, Any]:
    classes = np.asarray(getattr(env, "respec0b_resource_class", np.zeros((0, 0))), dtype=np.int32)
    amount = np.asarray(getattr(env, "patch_food_amount", np.zeros_like(classes, dtype=np.float32)), dtype=np.float32)
    digest_payload = {
        "classes_shape": list(classes.shape),
        "classes_sha256": hashlib.sha256(np.ascontiguousarray(classes).tobytes()).hexdigest() if classes.size else "",
        "amount_shape": list(amount.shape),
        "amount_sha256": hashlib.sha256(np.ascontiguousarray(amount).tobytes()).hexdigest() if amount.size else "",
    }
    rows = []
    for resource_class, zone in ((RESOURCE_A, ZONE_A), (RESOURCE_B, ZONE_B), (RESOURCE_NONE, ZONE_NEUTRAL)):
        mask = classes == int(resource_class)
        rows.append(
            {
                "zone": zone,
                "resource_class": int(resource_class),
                "cell_count": int(np.sum(mask)) if classes.size else 0,
                "initial_resource_amount": float(np.sum(amount[mask])) if classes.size else 0.0,
                "resource_amount_at_summary": float(np.sum(amount[mask])) if classes.size else 0.0,
            }
        )
    return {"schema_version": SCHEMA_VERSION, "layout": digest_payload, "zones": rows}


def record_gather_events(
    env: Any,
    *,
    agent_ids: Any,
    actions: Any,
    ys: Any,
    xs: Any,
    resource_handling_axes: Any,
    events: list[dict[str, Any]],
) -> None:
    if not enabled(getattr(env, "cfg", None)):
        return
    try:
        setattr(env, "respec_n1_resource_gather_events_last", [dict(row) for row in events])
    except Exception:
        setattr(env, "respec_n1_resource_gather_events_last", [])


def clear_gather_events(env: Any) -> None:
    if enabled(getattr(env, "cfg", None)):
        setattr(env, "respec_n1_resource_gather_events_last", [])


def record_tick(
    runner: Any,
    *,
    generation: int,
    tick: int,
    alive: Iterable[Any],
    died_mask: Any,
) -> None:
    if not enabled(runner):
        return
    env = getattr(runner, "env", None)
    if env is None:
        return
    state = _state(runner)
    pos_y = np.asarray(getattr(env, "agent_pos_y", []), dtype=np.int32)
    pos_x = np.asarray(getattr(env, "agent_pos_x", []), dtype=np.int32)
    alive_list = list(alive or [])
    died = np.asarray(died_mask, dtype=bool).reshape((-1,)) if died_mask is not None else np.zeros((len(alive_list),), dtype=bool)
    amount = np.asarray(getattr(env, "patch_food_amount", np.zeros((0, 0))), dtype=np.float32)
    classes = np.asarray(getattr(env, "respec0b_resource_class", np.zeros_like(amount)), dtype=np.int32)
    threshold = _float(getattr(getattr(env, "cfg", None), "patch_food_presence_threshold", 0.5))

    for i, agent in enumerate(alive_list):
        try:
            aid = int(getattr(agent, "agent_id", -1))
        except Exception:
            continue
        if aid < 0 or aid >= len(pos_y) or aid >= len(pos_x):
            continue
        y = int(pos_y[aid])
        x = int(pos_x[aid])
        zone = _zone_at(env, y, x)
        arm = _axis_arm(agent)
        key = (arm, zone)
        lineage_key = (_lineage_id(agent), zone)
        _add(state["by_arm_zone"], key, "agent_ticks", 1.0)
        _add(state["by_arm_zone"], key, "axis_sum", _axis_value(agent))
        age_steps = _age_steps(agent)
        _add(state["by_arm_zone"], key, "age_steps_sum", float(age_steps))
        if age_steps <= 0:
            _add(state["by_arm_zone"], key, "age_bin_0_ticks", 1.0)
            _add(state["by_arm_zone"], key, "juvenile_ticks", 1.0)
        elif age_steps <= 80:
            _add(state["by_arm_zone"], key, "age_bin_1_to_80_ticks", 1.0)
            _add(state["by_arm_zone"], key, "mature_ticks", 1.0)
        else:
            _add(state["by_arm_zone"], key, "age_bin_gt_80_ticks", 1.0)
            _add(state["by_arm_zone"], key, "mature_ticks", 1.0)
        _add(state["by_lineage_zone"], lineage_key, "agent_ticks", 1.0)
        state["unique_agents"][key].add(aid)
        state["first_entry_generation"].setdefault((arm, zone), int(generation))
        prev = state["current_zone"].get(aid)
        if prev is not None and prev != zone:
            dwell_key = (arm, prev)
            state["dwell_durations"][dwell_key].append(int(state["current_dwell"].get(aid, 0)))
            state["current_dwell"][aid] = 0
        state["current_zone"][aid] = zone
        state["current_dwell"][aid] += 1
        if 0 <= y < amount.shape[0] and 0 <= x < amount.shape[1]:
            cls = int(classes[y, x]) if classes.size else 0
            if cls == RESOURCE_A and float(amount[y, x]) > threshold:
                _add(state["by_arm_zone"], key, "resource_a_exposure_ticks", 1.0)
            if cls == RESOURCE_B and float(amount[y, x]) > threshold:
                _add(state["by_arm_zone"], key, "resource_b_exposure_ticks", 1.0)
        if i < len(died) and bool(died[i]):
            _add(state["by_arm_zone"], key, "deaths", 1.0)
            _add(state["by_lineage_zone"], lineage_key, "deaths", 1.0)

    events = list(getattr(env, "respec_n1_resource_gather_events_last", []) or [])
    agent_by_id = {int(getattr(agent, "agent_id", -1)): agent for agent in alive_list}
    for event in events:
        aid = int(event.get("agent_id", -1))
        agent = agent_by_id.get(aid)
        arm = _axis_arm(agent) if agent is not None else "unknown"
        zone = str(event.get("zone") or ZONE_NEUTRAL)
        key = (arm, zone)
        lineage_key = (_lineage_id(agent), zone) if agent is not None else ("unknown", zone)
        _add(state["by_arm_zone"], key, "gather_attempts", _float(event.get("attempt")))
        _add(state["by_arm_zone"], key, "available_resource_attempts", _float(event.get("available_resource_attempt")))
        _add(state["by_arm_zone"], key, "stockout_attempts", _float(event.get("stockout_attempt")))
        _add(state["by_arm_zone"], key, "active_handling_ticks", _float(event.get("handling_tick")))
        _add(state["by_arm_zone"], key, "completions", _float(event.get("success")))
        _add(state["by_arm_zone"], key, "net_resource_energy", _float(event.get("resource_return")))
        _add(state["by_arm_zone"], key, "interruptions", _float(event.get("interruption")))
        if int(event.get("resource_class", 0)) == RESOURCE_A:
            _add(state["by_arm_zone"], key, "resource_a_completions", _float(event.get("success")))
            _add(state["by_lineage_zone"], lineage_key, "resource_a_completions", _float(event.get("success")))
            if _float(event.get("success")) > 0.0:
                state["last_resource_success"][aid] = {"resource": "A", "generation": int(generation), "tick": int(tick), "zone": zone}
        if int(event.get("resource_class", 0)) == RESOURCE_B:
            _add(state["by_arm_zone"], key, "resource_b_completions", _float(event.get("success")))
            _add(state["by_lineage_zone"], lineage_key, "resource_b_completions", _float(event.get("success")))
            if _float(event.get("success")) > 0.0:
                state["last_resource_success"][aid] = {"resource": "B", "generation": int(generation), "tick": int(tick), "zone": zone}


def _distance_summary(env: Any, agents: Iterable[Any]) -> dict[str, float]:
    classes = np.asarray(getattr(env, "respec0b_resource_class", np.zeros((0, 0))), dtype=np.int32)
    amount = np.asarray(getattr(env, "patch_food_amount", np.zeros_like(classes, dtype=np.float32)), dtype=np.float32)
    threshold = _float(getattr(getattr(env, "cfg", None), "patch_food_presence_threshold", 0.5))
    pos_y = np.asarray(getattr(env, "agent_pos_y", []), dtype=np.int32)
    pos_x = np.asarray(getattr(env, "agent_pos_x", []), dtype=np.int32)
    coords_a = np.argwhere((classes == RESOURCE_A) & (amount > threshold)) if classes.size else np.zeros((0, 2), dtype=np.int32)
    coords_b = np.argwhere((classes == RESOURCE_B) & (amount > threshold)) if classes.size else np.zeros((0, 2), dtype=np.int32)
    dist_a = []
    dist_b = []
    for agent in list(agents or []):
        aid = int(getattr(agent, "agent_id", -1))
        if aid < 0 or aid >= len(pos_y) or aid >= len(pos_x):
            continue
        y = int(pos_y[aid])
        x = int(pos_x[aid])
        if coords_a.size:
            dist_a.append(float(np.min(np.abs(coords_a[:, 0] - y) + np.abs(coords_a[:, 1] - x))))
        if coords_b.size:
            dist_b.append(float(np.min(np.abs(coords_b[:, 0] - y) + np.abs(coords_b[:, 1] - x))))
    return {
        "mean_distance_to_available_resource_a": float(np.mean(dist_a)) if dist_a else 0.0,
        "mean_distance_to_available_resource_b": float(np.mean(dist_b)) if dist_b else 0.0,
    }


def _capacity_budget_rows(env: Any, *, generation: int) -> list[dict[str, Any]]:
    snapshot = getattr(env, "respec_n2_capacity_budget_last", None)
    if not isinstance(snapshot, dict) or not snapshot:
        return []
    contract = snapshot.get("contract", {}) if isinstance(snapshot.get("contract", {}), dict) else {}
    rows = []
    for row in list(snapshot.get("zones", []) or []):
        if not isinstance(row, Mapping):
            continue
        rows.append(
            {
                "generation": int(generation),
                "schema_version": str(snapshot.get("schema_version", "")),
                "capacity_balanced_mixed_enabled": bool(snapshot.get("capacity_balanced_mixed_enabled", False)),
                "formula_version": str(contract.get("formula_version", "")),
                "zone": str(row.get("zone", "")),
                "resource_class": int(row.get("resource_class", 0)),
                "cell_count": int(row.get("cell_count", 0)),
                "fertility_mean": _float(row.get("fertility_mean")),
                "amount_before": _float(row.get("amount_before")),
                "amount_after": _float(row.get("amount_after")),
                "cap_total": _float(row.get("cap_total")),
                "available_cells_before": int(row.get("available_cells_before", 0)),
                "available_cells_after": int(row.get("available_cells_after", 0)),
                "available_amount_before": _float(row.get("available_amount_before")),
                "available_amount_after": _float(row.get("available_amount_after")),
                "base_regrowth_added": _float(row.get("base_regrowth_added")),
                "effective_regrowth_added": _float(row.get("effective_regrowth_added")),
                "energy_opportunity_added": _float(row.get("energy_opportunity_added")),
                "return_per_completion": _float(row.get("return_per_completion")),
                "stockout_cells_after": int(row.get("stockout_cells_after", 0)),
            }
        )
    return rows


def finish_generation(
    runner: Any,
    *,
    generation: int,
    alive_after: Iterable[Any],
    selected_parent_agents: Iterable[Any],
    birth_contexts: Iterable[Mapping[str, Any]],
    eligible_parent_agents: Iterable[Any] = (),
) -> None:
    if not enabled(runner):
        return
    env = getattr(runner, "env", None)
    if env is None:
        return
    state = _state(runner)
    pos_y = np.asarray(getattr(env, "agent_pos_y", []), dtype=np.int32)
    pos_x = np.asarray(getattr(env, "agent_pos_x", []), dtype=np.int32)
    eligible = list(eligible_parent_agents or [])
    for agent in eligible:
        aid = int(getattr(agent, "agent_id", -1))
        if aid < 0 or aid >= len(pos_y) or aid >= len(pos_x):
            continue
        zone = _zone_at(env, int(pos_y[aid]), int(pos_x[aid]))
        arm = _axis_arm(agent)
        key = (arm, zone)
        _add(state["by_arm_zone"], key, "eligible_parent_events", 1.0)
        _add(state["by_lineage_zone"], (_lineage_id(agent), zone), "eligible_parent_events", 1.0)
        state["unique_eligible_parent_agents"][key].add(aid)

    selected = list(selected_parent_agents or [])
    for agent in selected:
        aid = int(getattr(agent, "agent_id", -1))
        if aid < 0 or aid >= len(pos_y) or aid >= len(pos_x):
            continue
        zone = _zone_at(env, int(pos_y[aid]), int(pos_x[aid]))
        arm = _axis_arm(agent)
        key = (arm, zone)
        _add(state["by_arm_zone"], key, "selected_parent_events", 1.0)
        _add(state["by_lineage_zone"], (_lineage_id(agent), zone), "selected_parent_events", 1.0)
        state["unique_selected_parent_agents"][key].add(aid)

    agent_lookup = {int(getattr(agent, "agent_id", -1)): agent for agent in list(alive_after or [])}
    for agent in selected:
        agent_lookup.setdefault(int(getattr(agent, "agent_id", -1)), agent)
    for birth in list(birth_contexts or []):
        parent_id = int(birth.get("parent_id", birth.get("parent_agent_id", -1)))
        py = int(birth.get("parent_y", birth.get("agent_y", birth.get("child_y", -1))))
        px = int(birth.get("parent_x", birth.get("agent_x", birth.get("child_x", -1))))
        zone = _zone_at(env, py, px)
        parent = agent_lookup.get(parent_id)
        arm = _axis_arm(parent) if parent is not None else "unknown"
        _add(state["by_arm_zone"], (arm, zone), "births", 1.0)
        lineage = str(birth.get("parent_lineage_id", "") or (_lineage_id(parent) if parent is not None else "unknown"))
        _add(state["by_lineage_zone"], (lineage, zone), "births", 1.0)
        recent = state["last_resource_success"].get(parent_id, {})
        if recent.get("resource") == "A":
            _add(state["by_arm_zone"], (arm, zone), "births_after_a_completion", 1.0)
        if recent.get("resource") == "B":
            _add(state["by_arm_zone"], (arm, zone), "births_after_b_completion", 1.0)

    generation_rows = []
    for key, fields in sorted(state["by_arm_zone"].items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        arm, zone = key
        ticks = float(fields.get("agent_ticks", 0.0))
        handling = float(fields.get("active_handling_ticks", 0.0))
        dwell = state["dwell_durations"].get(key, [])
        generation_rows.append(
            {
                "generation": int(generation),
                "arm": arm,
                "zone": zone,
                "agent_ticks_cumulative": ticks,
                "unique_agents_cumulative": len(state["unique_agents"].get(key, set())),
                "juvenile_ticks": float(fields.get("juvenile_ticks", 0.0)),
                "mature_ticks": float(fields.get("mature_ticks", 0.0)),
                "age_bin_0_ticks": float(fields.get("age_bin_0_ticks", 0.0)),
                "age_bin_1_to_80_ticks": float(fields.get("age_bin_1_to_80_ticks", 0.0)),
                "age_bin_gt_80_ticks": float(fields.get("age_bin_gt_80_ticks", 0.0)),
                "mean_age_steps": float(fields.get("age_steps_sum", 0.0)) / ticks if ticks > 0.0 else 0.0,
                "first_entry_generation": int(state["first_entry_generation"].get(key, -1)),
                "mean_dwell_duration": float(np.mean(dwell)) if dwell else 0.0,
                "gather_attempts": float(fields.get("gather_attempts", 0.0)),
                "available_resource_attempts": float(fields.get("available_resource_attempts", 0.0)),
                "stockout_attempts": float(fields.get("stockout_attempts", 0.0)),
                "resource_a_exposure_ticks": float(fields.get("resource_a_exposure_ticks", 0.0)),
                "resource_b_exposure_ticks": float(fields.get("resource_b_exposure_ticks", 0.0)),
                "active_handling_ticks": handling,
                "completions": float(fields.get("completions", 0.0)),
                "resource_a_completions": float(fields.get("resource_a_completions", 0.0)),
                "resource_b_completions": float(fields.get("resource_b_completions", 0.0)),
                "completion_per_handling_tick": float(fields.get("completions", 0.0)) / handling if handling > 0.0 else 0.0,
                "net_resource_energy": float(fields.get("net_resource_energy", 0.0)),
                "births": float(fields.get("births", 0.0)),
                "births_after_a_completion": float(fields.get("births_after_a_completion", 0.0)),
                "births_after_b_completion": float(fields.get("births_after_b_completion", 0.0)),
                "births_per_1000_agent_ticks": 1000.0 * float(fields.get("births", 0.0)) / ticks if ticks > 0.0 else 0.0,
                "eligible_parent_events": float(fields.get("eligible_parent_events", 0.0)),
                "unique_eligible_parent_agents": len(state["unique_eligible_parent_agents"].get(key, set())),
                "selected_parent_events_per_1000_eligible_parent_events": 1000.0
                * float(fields.get("selected_parent_events", 0.0))
                / float(fields.get("eligible_parent_events", 0.0))
                if float(fields.get("eligible_parent_events", 0.0)) > 0.0
                else 0.0,
                "births_per_1000_eligible_parent_events": 1000.0
                * float(fields.get("births", 0.0))
                / float(fields.get("eligible_parent_events", 0.0))
                if float(fields.get("eligible_parent_events", 0.0)) > 0.0
                else 0.0,
                "selected_parent_events": float(fields.get("selected_parent_events", 0.0)),
                "unique_selected_parent_agents": len(state["unique_selected_parent_agents"].get(key, set())),
                "selected_parent_events_per_1000_agent_ticks": 1000.0
                * float(fields.get("selected_parent_events", 0.0))
                / ticks
                if ticks > 0.0
                else 0.0,
                "deaths": float(fields.get("deaths", 0.0)),
            }
        )
    state["generation_rows"] = generation_rows
    state["history_rows"].extend(dict(row) for row in generation_rows)
    budget_rows = _capacity_budget_rows(env, generation=int(generation))
    state["capacity_budget_rows"].extend(dict(row) for row in budget_rows)
    metrics_dir = _run_dir(runner) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    gen_path = metrics_dir / "respec_n1_zone_generation_summary.csv"
    history_rows = list(state["history_rows"])
    fields = list(history_rows[0].keys()) if history_rows else ["generation", "arm", "zone"]
    with gen_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(history_rows)

    if state["capacity_budget_rows"]:
        budget_path = metrics_dir / "respec_n2_capacity_budget_generation_summary.csv"
        budget_fields = list(state["capacity_budget_rows"][0].keys())
        with budget_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=budget_fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(state["capacity_budget_rows"])

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generation": int(generation),
        "layout": layout_digest(env),
        "arm_zone": generation_rows,
        "arm_zone_history_rows": len(history_rows),
        "capacity_budget": getattr(env, "respec_n2_capacity_budget_last", {}),
        "capacity_budget_history_rows": len(state["capacity_budget_rows"]),
        "lineage_zone": _serial_counter(state["by_lineage_zone"]),
        "distance_boundary": _distance_summary(env, alive_after),
    }
    (metrics_dir / "respec_n1_zone_run_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
