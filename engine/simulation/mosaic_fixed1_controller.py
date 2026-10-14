from __future__ import annotations

"""Default-off MOSAIC-FIXED-1 reactive validation probes.

The controller is experiment-only. It selects ordinary action IDs from ordinary
Patch World observations, then lets the existing environment path apply those
actions. Patch-class labels are used only by passive telemetry.
"""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from env.gridworld_pkg.constants import ACT_GATHER, ACT_MOVE_E, ACT_MOVE_N, ACT_MOVE_S, ACT_MOVE_W, ACT_REST
from env.gridworld_pkg.patch_world import (
    MOSAIC0A_PATCH_BURST,
    MOSAIC0A_PATCH_RENEWAL,
    PATCH_WORLD_OBSERVATION_CHANNEL_NAMES,
    patch_world_local_observation_channel_count,
    patch_world_observation_enabled,
)


SCHEMA_VERSION = "mosaic_fixed1_reactive_probe_v1"
FIXED1R_TELEMETRY_SCHEMA_VERSION = "mosaic_fixed1r_reciprocal_telemetry_v1"
ARM_NAMES = ("exploit_move", "balanced_generalist", "recovery_conservative")
MOVE_ACTIONS = (ACT_MOVE_N, ACT_MOVE_S, ACT_MOVE_E, ACT_MOVE_W)
MOVE_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
CLASS_NAMES = ("burst", "renewal")


ARM_RULES: dict[str, dict[str, float]] = {
    "exploit_move": {
        "gather_threshold": 0.70,
        "depart_threshold": 0.45,
        "pressure_weight": 0.15,
        "movement_cost_weight": 0.25,
        "rest_threshold": 9.0,
    },
    "balanced_generalist": {
        "gather_threshold": 0.55,
        "depart_threshold": 0.32,
        "pressure_weight": 0.45,
        "movement_cost_weight": 0.40,
        "rest_threshold": 1.60,
    },
    "recovery_conservative": {
        "gather_threshold": 0.42,
        "depart_threshold": 0.25,
        "pressure_weight": 0.90,
        "movement_cost_weight": 0.55,
        "rest_threshold": 0.95,
    },
}


def controller_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "arms": ARM_RULES,
        "available_channels": ["legacy_food_occupancy", *PATCH_WORLD_OBSERVATION_CHANNEL_NAMES[1:]],
        "actions": {
            "move_n": ACT_MOVE_N,
            "move_s": ACT_MOVE_S,
            "move_e": ACT_MOVE_E,
            "move_w": ACT_MOVE_W,
            "gather": ACT_GATHER,
            "rest": ACT_REST,
        },
        "hidden_state_excluded": [
            "mosaic0a_patch_class",
            "future food",
            "future recovery",
            "global maps",
            "lineage identity",
            "branch identity",
            "selection outcomes",
        ],
    }


def controller_digest() -> str:
    payload = json.dumps(controller_contract(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fixed1r_telemetry_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    if not bool(getattr(cfg, "mosaic_fixed1r_telemetry_enable", False)):
        return False
    if int(getattr(cfg, "mosaic_fixed1r_telemetry_schema_version", 1) or 1) != 1:
        raise ValueError("unsupported mosaic_fixed1r_telemetry_schema_version")
    return probe_enabled(runner)


def probe_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    if not bool(getattr(cfg, "mosaic_fixed1_probe_enable", False)):
        return False
    if int(getattr(cfg, "mosaic_fixed1_probe_schema_version", 1) or 1) != 1:
        raise ValueError("unsupported mosaic_fixed1_probe_schema_version")
    arm = str(getattr(cfg, "mosaic_fixed1_probe_arm", "") or "")
    if arm not in ARM_NAMES:
        raise ValueError(f"unsupported mosaic_fixed1_probe_arm: {arm}")
    if str(getattr(cfg, "mosaic_fixed1_probe_controller_version", "mosaic_fixed1_reactive_v1")) != "mosaic_fixed1_reactive_v1":
        raise ValueError("unsupported mosaic_fixed1_probe_controller_version")
    env_cfg = getattr(runner, "env_cfg", None)
    if bool(getattr(env_cfg, "patch_world_enable", False)) and not patch_world_observation_enabled(env_cfg):
        raise ValueError("mosaic_fixed1_probe_enable requires patch_world_observation_enable when patch_world_enable is true")
    return True


def _obs_views(runner: Any, agent_ids: np.ndarray) -> np.ndarray:
    cfg = getattr(runner, "env_cfg", None)
    r = int(getattr(cfg, "view_radius", 1) or 1)
    k = 2 * r + 1
    channels = int(patch_world_local_observation_channel_count(cfg))
    obs = np.asarray(runner.env.get_local_observations(agent_ids), dtype=np.float32)
    return obs.reshape((int(agent_ids.size), channels, k, k))


CHANNEL_NAMES = ("legacy_food_occupancy", *PATCH_WORLD_OBSERVATION_CHANNEL_NAMES[1:])


def _channel_index(name: str) -> int:
    return int(CHANNEL_NAMES.index(name))


LEGACY_FOOD_CH = _channel_index("legacy_food_occupancy")
FOOD_CH = _channel_index("patch_food_amount_norm")
WASTE_CH = _channel_index("patch_waste_norm")
DENSITY_CH = _channel_index("patch_density_norm")
MOVE_COST_CH = _channel_index("patch_movement_cost_norm")


def _channel_at(view: np.ndarray, channel: int, y: int, x: int, *, fallback_channel: int | None = None) -> float:
    if 0 <= int(channel) < int(view.shape[0]):
        return float(view[int(channel), int(y), int(x)])
    if fallback_channel is not None and 0 <= int(fallback_channel) < int(view.shape[0]):
        return float(view[int(fallback_channel), int(y), int(x)])
    return 0.0


def _best_move(view: np.ndarray, *, rules: dict[str, float], conservative: bool) -> int:
    r = view.shape[1] // 2
    best_action = ACT_MOVE_N
    best_score = -1.0e9
    for action, (dy, dx) in zip(MOVE_ACTIONS, MOVE_DELTAS):
        y = int(r + dy)
        x = int(r + dx)
        food = _channel_at(view, FOOD_CH, y, x, fallback_channel=LEGACY_FOOD_CH)
        pressure = 0.5 * _channel_at(view, WASTE_CH, y, x) + 0.5 * _channel_at(view, DENSITY_CH, y, x)
        move_cost = _channel_at(view, MOVE_COST_CH, y, x)
        if conservative:
            score = food - float(rules["pressure_weight"]) * pressure - float(rules["movement_cost_weight"]) * move_cost
        else:
            score = food - float(rules["movement_cost_weight"]) * move_cost
        if score > best_score:
            best_score = score
            best_action = int(action)
    return int(best_action)


def choose_actions(runner: Any, alive: list[Any], raw_actions: Any) -> np.ndarray:
    if not probe_enabled(runner):
        return np.asarray(raw_actions, dtype=np.int32)
    n = int(len(alive))
    if n <= 0:
        return np.empty((0,), dtype=np.int32)
    arm = str(getattr(runner.sim_cfg, "mosaic_fixed1_probe_arm"))
    rules = ARM_RULES[arm]
    agent_ids = np.asarray([int(getattr(agent, "agent_id")) for agent in alive], dtype=np.int32)
    views = _obs_views(runner, agent_ids)
    r = views.shape[2] // 2
    actions = np.empty((n,), dtype=np.int32)
    for i in range(n):
        view = views[i]
        current_food = _channel_at(view, FOOD_CH, r, r, fallback_channel=LEGACY_FOOD_CH)
        current_pressure = 0.5 * _channel_at(view, WASTE_CH, r, r) + 0.5 * _channel_at(view, DENSITY_CH, r, r)
        if arm == "exploit_move":
            if current_food >= float(rules["gather_threshold"]):
                actions[i] = np.int32(ACT_GATHER)
            else:
                actions[i] = np.int32(_best_move(view, rules=rules, conservative=False))
        elif arm == "recovery_conservative":
            if current_food >= float(rules["gather_threshold"]) and current_pressure < float(rules["rest_threshold"]):
                actions[i] = np.int32(ACT_GATHER)
            elif current_pressure >= float(rules["rest_threshold"]) or current_food <= float(rules["depart_threshold"]):
                actions[i] = np.int32(ACT_REST)
            else:
                actions[i] = np.int32(_best_move(view, rules=rules, conservative=True))
        else:
            if current_food >= float(rules["gather_threshold"]):
                actions[i] = np.int32(ACT_GATHER)
            elif current_pressure >= float(rules["rest_threshold"]):
                actions[i] = np.int32(ACT_REST)
            else:
                actions[i] = np.int32(_best_move(view, rules=rules, conservative=True))
    _record_pre_env(runner, agent_ids=agent_ids, actions=actions, views=views)
    return actions


def _state(runner: Any) -> dict[str, Any]:
    state = getattr(runner, "_mosaic_fixed1_state", None)
    if not isinstance(state, dict):
        state = {
            "schema_version": SCHEMA_VERSION,
            "fixed1r_schema_version": FIXED1R_TELEMETRY_SCHEMA_VERSION,
            "generations": {},
            "last_pre": {},
            "agent_class": {},
            "class_success_by_gen": {},
            "repro_recorded_generations": [],
        }
        setattr(runner, "_mosaic_fixed1_state", state)
    return state


def _gen_state(runner: Any) -> dict[str, float]:
    gen = int(getattr(runner, "_eiel_fdp_cur_gen", getattr(runner, "_current_generation", 0)) or 0)
    state = _state(runner)
    gens = state.setdefault("generations", {})
    row = gens.setdefault(str(gen), {"generation": float(gen)})
    return row


def _inc(row: dict[str, float], key: str, amount: float = 1.0) -> None:
    row[key] = float(row.get(key, 0.0) or 0.0) + float(amount)


def _agent_id(agent: Any) -> int:
    try:
        return int(getattr(agent, "agent_id"))
    except Exception:
        return -1


def _class_name(value: int) -> str:
    if int(value) == MOSAIC0A_PATCH_BURST:
        return "burst"
    if int(value) == MOSAIC0A_PATCH_RENEWAL:
        return "renewal"
    return "none"


def _record_pre_env(runner: Any, *, agent_ids: np.ndarray, actions: np.ndarray, views: np.ndarray) -> None:
    row = _gen_state(runner)
    fixed1r = fixed1r_telemetry_enabled(runner)
    cfg = getattr(runner, "env_cfg", None)
    threshold = float(getattr(cfg, "patch_food_presence_threshold", 0.5) or 0.5)
    env = getattr(runner, "env", None)
    ys = np.asarray(getattr(env, "agent_pos_y"))[agent_ids]
    xs = np.asarray(getattr(env, "agent_pos_x"))[agent_ids]
    classes = np.asarray(getattr(env, "mosaic0a_patch_class", np.zeros_like(getattr(env, "grid"))), dtype=np.int32)[ys, xs]
    food = np.asarray(getattr(env, "patch_food_amount", np.zeros_like(getattr(env, "grid"), dtype=np.float32)), dtype=np.float32)[ys, xs]
    movement_cost = np.asarray(
        getattr(env, "patch_movement_cost", np.ones_like(getattr(env, "grid"), dtype=np.float32)),
        dtype=np.float32,
    )[ys, xs]
    state = _state(runner)
    agent_class = state.setdefault("agent_class", {})
    for i, action in enumerate(np.asarray(actions, dtype=np.int32)):
        aid = int(agent_ids[i])
        cls = _class_name(int(classes[i]))
        f = float(food[i])
        move_cost = float(movement_cost[i])
        _inc(row, "agent_ticks")
        _inc(row, f"{cls}_dwell_ticks")
        if fixed1r and cls in CLASS_NAMES:
            _inc(row, f"{cls}_occupancy_ticks")
            _inc(row, f"{cls}_available_resource_agent_ticks", 1.0 if f > threshold else 0.0)
            _inc(row, f"{cls}_stockout_agent_ticks", 1.0 if f <= threshold else 0.0)
            _inc(row, f"{cls}_food_amount_agent_ticks", f)
            prev = str(agent_class.get(str(aid), ""))
            if prev != cls:
                _inc(row, f"{cls}_entries")
                if f > threshold:
                    _inc(row, f"{cls}_fresh_opportunity_entries")
        if int(action) in MOVE_ACTIONS:
            _inc(row, "movement_actions")
            _inc(row, f"{cls}_movement_actions")
            if fixed1r and cls in CLASS_NAMES:
                _inc(row, f"{cls}_movement_cost_before_use", move_cost)
            if float(food[i]) <= 0.45:
                _inc(row, "depleted_patch_departures")
                if fixed1r and cls in CLASS_NAMES:
                    _inc(row, f"{cls}_depleted_patch_departures")
        elif int(action) == ACT_GATHER:
            _inc(row, "gather_actions")
            _inc(row, f"{cls}_gather_attempts")
            if float(food[i]) > 0.5:
                _inc(row, f"{cls}_available_food_gather_attempts")
            else:
                _inc(row, "stockout_attempts")
                if fixed1r and cls in CLASS_NAMES:
                    _inc(row, f"{cls}_stockout_attempts")
        elif int(action) == ACT_REST:
            _inc(row, "rest_actions")
            _inc(row, f"{cls}_rest_actions")
        if fixed1r:
            agent_class[str(aid)] = cls
    _state(runner)["last_pre"] = {
        "agent_ids": [int(x) for x in agent_ids.tolist()],
        "actions": [int(x) for x in np.asarray(actions, dtype=np.int32).tolist()],
        "classes": [int(x) for x in classes.tolist()],
        "food": [float(x) for x in np.asarray(food, dtype=np.float32).tolist()],
    }


def record_post_env(runner: Any, *, agent_ids: Any, actions: Any, food_gained: Any) -> None:
    if not probe_enabled(runner):
        return
    row = _gen_state(runner)
    pre = dict(_state(runner).get("last_pre") or {})
    pre_ids = [int(x) for x in list(pre.get("agent_ids") or [])]
    pre_classes = [int(x) for x in list(pre.get("classes") or [])]
    cls_by_id = {aid: cls for aid, cls in zip(pre_ids, pre_classes)}
    ids = [int(x) for x in np.asarray(agent_ids, dtype=np.int32).tolist()]
    acts = [int(x) for x in np.asarray(actions, dtype=np.int32).tolist()]
    gains = [float(x) for x in np.asarray(food_gained, dtype=np.float32).tolist()]
    for aid, action, gain in zip(ids, acts, gains):
        cls = _class_name(int(cls_by_id.get(aid, 0)))
        if action in MOVE_ACTIONS:
            _inc(row, "successful_moves")
        if action == ACT_GATHER and gain > 0.0:
            _inc(row, f"{cls}_gather_successes")
            _inc(row, f"{cls}_successful_gathers")
            _inc(row, f"{cls}_food_gain", gain)
            if fixed1r_telemetry_enabled(runner) and cls in CLASS_NAMES:
                gen = str(int(getattr(runner, "_eiel_fdp_cur_gen", getattr(runner, "_current_generation", 0)) or 0))
                successes = _state(runner).setdefault("class_success_by_gen", {}).setdefault(gen, {})
                class_ids = successes.setdefault(cls, [])
                if int(aid) not in class_ids:
                    class_ids.append(int(aid))


SUMMARY_FIELDS = [
    "generation",
    "agent_ticks",
    "movement_actions",
    "successful_moves",
    "gather_actions",
    "rest_actions",
    "stockout_attempts",
    "depleted_patch_departures",
    "burst_dwell_ticks",
    "renewal_dwell_ticks",
    "burst_movement_actions",
    "renewal_movement_actions",
    "burst_gather_attempts",
    "renewal_gather_attempts",
    "burst_available_food_gather_attempts",
    "renewal_available_food_gather_attempts",
    "burst_gather_successes",
    "renewal_gather_successes",
    "burst_food_gain",
    "renewal_food_gain",
    "burst_rest_actions",
    "renewal_rest_actions",
]

FIXED1R_FIELDS = [
    "generation",
    "agent_ticks",
    "movement_actions",
    "gather_actions",
    "rest_actions",
]
for _cls in CLASS_NAMES:
    FIXED1R_FIELDS.extend(
        [
            f"{_cls}_available_resource_agent_ticks",
            f"{_cls}_stockout_agent_ticks",
            f"{_cls}_food_amount_agent_ticks",
            f"{_cls}_occupancy_ticks",
            f"{_cls}_entries",
            f"{_cls}_fresh_opportunity_entries",
            f"{_cls}_depleted_patch_departures",
            f"{_cls}_gather_attempts",
            f"{_cls}_successful_gathers",
            f"{_cls}_food_gain",
            f"{_cls}_stockout_attempts",
            f"{_cls}_movement_actions",
            f"{_cls}_movement_cost_before_use",
            f"{_cls}_net_return",
            f"{_cls}_eligible_parent_events_after_class_use",
            f"{_cls}_selected_parent_events_after_class_use",
            f"{_cls}_successful_births_after_class_use",
        ]
    )


def _record_fixed1r_reproduction_coupling(runner: Any, *, generation: int) -> None:
    if not fixed1r_telemetry_enabled(runner):
        return
    state = _state(runner)
    recorded = list(state.setdefault("repro_recorded_generations", []))
    gen_key = str(int(generation))
    if gen_key in recorded:
        return
    row = _gen_state(runner)
    successes_by_class = dict((state.get("class_success_by_gen") or {}).get(gen_key) or {})
    eligible = list(getattr(runner, "_eligible_parent_agents_gen", []) or [])
    selected = list(getattr(runner, "_selected_parent_agents_gen", []) or [])
    birth_contexts = list(getattr(runner, "_lineage_reproduction_funnel_birth_contexts_gen", []) or [])
    for cls in CLASS_NAMES:
        success_ids = {int(aid) for aid in list(successes_by_class.get(cls) or [])}
        eligible_ids = {_agent_id(agent) for agent in eligible if _agent_id(agent) in success_ids}
        selected_ids = {_agent_id(agent) for agent in selected if _agent_id(agent) in success_ids}
        _inc(row, f"{cls}_eligible_parent_events_after_class_use", len(eligible_ids))
        _inc(row, f"{cls}_selected_parent_events_after_class_use", len(selected_ids))
        births = 0
        for ctx in birth_contexts:
            try:
                parent_id = int(ctx.get("parent_id", ctx.get("parent_agent_id", -1)))
            except Exception:
                parent_id = -1
            if parent_id in success_ids:
                births += 1
        _inc(row, f"{cls}_successful_births_after_class_use", births)
    recorded.append(gen_key)
    state["repro_recorded_generations"] = recorded


def write_generation_summary(runner: Any, *, generation: int) -> None:
    if not probe_enabled(runner):
        return
    _record_fixed1r_reproduction_coupling(runner, generation=int(generation))
    run_dir = Path(str(getattr(runner, "run_dir", "") or ""))
    if not run_dir:
        return
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "mosaic_fixed1_generation_summary.csv"
    state = _state(runner)
    rows = []
    for key in sorted((state.get("generations") or {}).keys(), key=lambda x: int(x)):
        payload = dict((state.get("generations") or {}).get(key) or {})
        rows.append({field: payload.get(field, 0.0) for field in SUMMARY_FIELDS})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if fixed1r_telemetry_enabled(runner):
        fixed1r_rows = []
        for key in sorted((state.get("generations") or {}).keys(), key=lambda x: int(x)):
            payload = dict((state.get("generations") or {}).get(key) or {})
            out = {field: payload.get(field, 0.0) for field in FIXED1R_FIELDS}
            for cls in CLASS_NAMES:
                out[f"{cls}_net_return"] = float(out.get(f"{cls}_food_gain", 0.0) or 0.0) - float(
                    out.get(f"{cls}_movement_cost_before_use", 0.0) or 0.0
                )
            fixed1r_rows.append(out)
        fixed1r_path = metrics_dir / "mosaic_fixed1r_generation_summary.csv"
        with fixed1r_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIXED1R_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(fixed1r_rows)
        telemetry_payload = {
            "schema_version": FIXED1R_TELEMETRY_SCHEMA_VERSION,
            "recent_use_definition": "at least one successful gather from class during the just-completed generation",
            "arm_identity": "run-level sim_cfg.mosaic_fixed1_probe_arm",
            "net_return_definition": "observed class food gain minus class movement-cost exposure on movement actions",
            "fields": FIXED1R_FIELDS,
            "controller_digest": controller_digest(),
        }
        (metrics_dir / "mosaic_fixed1r_telemetry_contract.json").write_text(
            json.dumps(telemetry_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    contract_path = metrics_dir / "mosaic_fixed1_controller_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "arm": str(getattr(runner.sim_cfg, "mosaic_fixed1_probe_arm", "")),
                "controller_digest": controller_digest(),
                "contract": controller_contract(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
