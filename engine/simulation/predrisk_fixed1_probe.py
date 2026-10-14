from __future__ import annotations

"""Default-off PREDRISK-FIXED-1 fixed ordinary-policy probes.

The probes are experiment-only. They build frozen MLPPolicy weights, choose
ordinary action IDs from ordinary observations, and let the existing environment
path apply those actions. Hidden zone labels are used only by telemetry.
"""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agents.agent_policy import MLPPolicy
from env.gridworld_pkg.constants import ACT_GATHER, ACT_MOVE_E, ACT_MOVE_N, ACT_MOVE_S, ACT_MOVE_W, ACT_REST
from env.gridworld_pkg.jungle import compute_predator_risk_maps
from env.gridworld_pkg.patch_world import (
    PATCH_WORLD_OBSERVATION_CHANNEL_NAMES,
    PREDRISK_FIXED1_ZONE_EXPOSED_RICH,
    PREDRISK_FIXED1_ZONE_SAFE_COVERED,
    patch_world_local_observation_channel_count,
    patch_world_observation_enabled,
)


SCHEMA_VERSION = "predrisk_fixed1_mlp_probe_v1"
TELEMETRY_SCHEMA_VERSION = "predrisk_fixed1_telemetry_v1"
ARM_NAMES = ("risk_exploit", "balanced_generalist", "safe_cover")
MOVE_ACTIONS = (ACT_MOVE_N, ACT_MOVE_E, ACT_MOVE_S, ACT_MOVE_W)
MOVE_CELLS = ("north", "east", "south", "west")
OBS_CHANNELS = ("legacy_food_occupancy", *PATCH_WORLD_OBSERVATION_CHANNEL_NAMES[1:])
TELEMETRY_ZONE_NAMES = ("none", "exposed_rich", "safe_covered")
TELEMETRY_ZONE_FIELDS = (
    "agent_ticks",
    "available_resource_agent_ticks",
    "food_amount_sum",
    "cover_sum",
    "shelter_sum",
    "predator_risk_sum",
    "movement_cost_sum",
    "gather_attempts",
    "rest_actions",
    "move_actions",
    "movement_cost_before_move_actions",
    "gather_successes",
    "food_gain",
    "eligible_parent_events_after_zone_use",
    "selected_parent_events_after_zone_use",
    "successful_births_after_zone_use",
    "predator_events",
    "raw_predator_damage_cost",
    "effective_predator_damage_cost",
    "predator_damage_prevented",
)
SUMMARY_FIELDS = (
    "arm",
    "generation",
    "exposure_ticks",
    "cover_occupancy_ticks",
    *tuple(f"{zone}_{field}" for zone in TELEMETRY_ZONE_NAMES for field in TELEMETRY_ZONE_FIELDS),
)


def _obs_index(channel: str, cell: str) -> int:
    cell_index = {
        "north_west": 0,
        "north": 1,
        "north_east": 2,
        "west": 3,
        "center": 4,
        "east": 5,
        "south_west": 6,
        "south": 7,
        "south_east": 8,
    }[cell]
    return OBS_CHANNELS.index(channel) * 9 + cell_index


def _zero_policy() -> MLPPolicy:
    policy = MLPPolicy(obs_size=74, hidden_size=16, n_actions=7)
    with torch.no_grad():
        for param in policy.parameters():
            param.zero_()
        policy.net[2].weight.copy_(torch.eye(16))
    return policy


def build_probe_policy(arm: str) -> MLPPolicy:
    if arm not in ARM_NAMES:
        raise ValueError(f"unsupported predrisk_fixed1_probe_arm: {arm}")
    policy = _zero_policy()
    first = policy.net[0]
    head = policy.net[4]
    with torch.no_grad():
        if arm == "risk_exploit":
            food_w, cover_w, shelter_w, move_w, center_food_w, health_rest_w = 4.0, -0.7, -0.3, -0.2, 4.2, 2.5
        elif arm == "safe_cover":
            food_w, cover_w, shelter_w, move_w, center_food_w, health_rest_w = 1.0, 4.0, 2.2, -0.4, 1.6, 8.0
        else:
            food_w, cover_w, shelter_w, move_w, center_food_w, health_rest_w = 2.4, 1.8, 1.0, -0.3, 2.8, 4.6
        for unit, (action, cell) in enumerate(zip(MOVE_ACTIONS, MOVE_CELLS)):
            first.weight[unit, _obs_index("patch_food_amount_norm", cell)] = food_w
            first.weight[unit, _obs_index("patch_cover", cell)] = cover_w
            first.weight[unit, _obs_index("patch_shelter", cell)] = shelter_w
            first.weight[unit, _obs_index("patch_movement_cost_norm", cell)] = move_w
            head.weight[action, unit] = 4.0
        first.weight[8, _obs_index("patch_food_amount_norm", "center")] = center_food_w
        first.weight[8, _obs_index("patch_cover", "center")] = 0.3 if arm != "risk_exploit" else -0.2
        head.weight[ACT_GATHER, 8] = 4.0
        first.bias[9] = 2.2 if arm == "safe_cover" else 1.2
        first.weight[9, 73] = -6.5 if arm == "safe_cover" else -3.5
        head.weight[ACT_REST, 9] = health_rest_w
    return policy


def _state_dict_digest(policy: MLPPolicy) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        h.update(name.encode("utf-8"))
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def policy_hashes() -> dict[str, str]:
    return {arm: _state_dict_digest(build_probe_policy(arm)) for arm in ARM_NAMES}


def controller_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "arms": list(ARM_NAMES),
        "policy_type": "MLPPolicy",
        "obs_size": 74,
        "hidden_size": 16,
        "n_actions": 7,
        "policy_hashes": policy_hashes(),
        "ordinary_inputs": [*OBS_CHANNELS, "energy_norm", "health_norm"],
        "ordinary_actions": {
            "move_n": ACT_MOVE_N,
            "move_s": ACT_MOVE_S,
            "move_e": ACT_MOVE_E,
            "move_w": ACT_MOVE_W,
            "gather": ACT_GATHER,
            "rest": ACT_REST,
        },
        "hidden_state_excluded_from_action": [
            "predrisk_fixed1_zone_class",
            "jungle_predator_risk",
            "zone labels",
            "future state",
            "global map",
            "lineage identity",
            "branch identity",
            "selection outcomes",
            "reward",
        ],
    }


def probe_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    if not bool(getattr(cfg, "predrisk_fixed1_probe_enable", False)):
        return False
    if int(getattr(cfg, "predrisk_fixed1_probe_schema_version", 1) or 1) != 1:
        raise ValueError("unsupported predrisk_fixed1_probe_schema_version")
    arm = str(getattr(cfg, "predrisk_fixed1_probe_arm", "") or "")
    if arm not in ARM_NAMES:
        raise ValueError(f"unsupported predrisk_fixed1_probe_arm: {arm}")
    if str(getattr(cfg, "predrisk_fixed1_probe_controller_version", "predrisk_fixed1_mlp_v1")) != "predrisk_fixed1_mlp_v1":
        raise ValueError("unsupported predrisk_fixed1_probe_controller_version")
    env_cfg = getattr(runner, "env_cfg", None)
    if not bool(getattr(env_cfg, "patch_world_enable", False)):
        raise ValueError("predrisk_fixed1_probe_enable requires patch_world_enable")
    if not patch_world_observation_enabled(env_cfg):
        raise ValueError("predrisk_fixed1_probe_enable requires patch_world_observation_enable")
    if int(getattr(env_cfg, "view_radius", 1) or 1) != 1:
        raise ValueError("predrisk_fixed1_probe_enable currently requires view_radius=1")
    return True


def telemetry_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    if not bool(getattr(cfg, "predrisk_fixed1_telemetry_enable", False)):
        return False
    if int(getattr(cfg, "predrisk_fixed1_telemetry_schema_version", 1) or 1) != 1:
        raise ValueError("unsupported predrisk_fixed1_telemetry_schema_version")
    return probe_enabled(runner)


def _obs_batch(runner: Any, alive: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    agent_ids = np.asarray([int(getattr(agent, "agent_id")) for agent in alive], dtype=np.int32)
    cfg = getattr(runner, "env_cfg", None)
    channels = int(patch_world_local_observation_channel_count(cfg))
    views = np.asarray(runner.env.get_local_observations(agent_ids), dtype=np.float32).reshape(
        (int(agent_ids.size), channels, 3, 3)
    )
    local = views.reshape((int(agent_ids.size), channels * 9))
    obs = np.zeros((int(agent_ids.size), 74), dtype=np.float32)
    obs[:, : local.shape[1]] = local
    energy_norm = float(max(1e-9, float(getattr(runner.env_cfg, "initial_energy", 100.0) or 100.0)))
    health_norm = float(max(1e-9, float(getattr(runner.env_cfg, "initial_health", 100.0) or 100.0)))
    for i, agent in enumerate(alive):
        obs[i, 72] = np.float32(float(getattr(agent, "energy", energy_norm)) / energy_norm)
        obs[i, 73] = np.float32(float(getattr(agent, "health", health_norm)) / health_norm)
    return agent_ids, obs


def choose_actions(runner: Any, alive: list[Any], raw_actions: Any) -> np.ndarray:
    if not probe_enabled(runner):
        return np.asarray(raw_actions, dtype=np.int32)
    if not alive:
        return np.empty((0,), dtype=np.int32)
    arm = str(getattr(runner.sim_cfg, "predrisk_fixed1_probe_arm"))
    policy = _probe_policy_cache(runner, arm)
    agent_ids, obs = _obs_batch(runner, alive)
    with torch.no_grad():
        logits = policy(torch.from_numpy(obs)).detach().cpu().numpy()
    actions = np.asarray(np.argmax(logits, axis=1), dtype=np.int32)
    _record_pre_env(runner, agent_ids=agent_ids, actions=actions, obs=obs, logits=logits)
    return actions


def _probe_policy_cache(runner: Any, arm: str) -> MLPPolicy:
    cache = getattr(runner, "_predrisk_fixed1_policy_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(runner, "_predrisk_fixed1_policy_cache", cache)
    if arm not in cache:
        policy = build_probe_policy(arm)
        policy.eval()
        cache[arm] = policy
    return cache[arm]


def _state(runner: Any) -> dict[str, Any]:
    state = getattr(runner, "_predrisk_fixed1_state", None)
    if not isinstance(state, dict):
        state = {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "generations": {},
            "last_pre": {},
            "zone_success_by_gen": {},
            "repro_recorded_generations": [],
        }
        setattr(runner, "_predrisk_fixed1_state", state)
    return state


def _gen_row(runner: Any) -> dict[str, float]:
    gen = int(getattr(runner, "_eiel_fdp_cur_gen", getattr(runner, "_current_generation", 0)) or 0)
    gens = _state(runner).setdefault("generations", {})
    return gens.setdefault(str(gen), {"generation": float(gen)})


def _inc(row: dict[str, float], key: str, amount: float = 1.0) -> None:
    row[key] = float(row.get(key, 0.0) or 0.0) + float(amount)


def _zone_name(value: int) -> str:
    if int(value) == PREDRISK_FIXED1_ZONE_EXPOSED_RICH:
        return "exposed_rich"
    if int(value) == PREDRISK_FIXED1_ZONE_SAFE_COVERED:
        return "safe_covered"
    return "none"


def _agent_id(agent: Any) -> int:
    try:
        return int(getattr(agent, "agent_id"))
    except Exception:
        return -1


def _record_pre_env(runner: Any, *, agent_ids: np.ndarray, actions: np.ndarray, obs: np.ndarray, logits: np.ndarray) -> None:
    if not telemetry_enabled(runner):
        return
    row = _gen_row(runner)
    env = getattr(runner, "env", None)
    ys = np.asarray(getattr(env, "agent_pos_y"))[agent_ids]
    xs = np.asarray(getattr(env, "agent_pos_x"))[agent_ids]
    zones = np.asarray(getattr(env, "predrisk_fixed1_zone_class", np.zeros_like(getattr(env, "grid"))), dtype=np.int32)[ys, xs]
    risk = np.asarray(compute_predator_risk_maps(env)["risk"], dtype=np.float32)[ys, xs]
    cover = np.asarray(getattr(env, "patch_cover"), dtype=np.float32)[ys, xs]
    shelter = np.asarray(getattr(env, "patch_shelter"), dtype=np.float32)[ys, xs]
    food = np.asarray(getattr(env, "patch_food_amount"), dtype=np.float32)[ys, xs]
    move = np.asarray(getattr(env, "patch_movement_cost"), dtype=np.float32)[ys, xs]
    for i, aid in enumerate(agent_ids):
        zone = _zone_name(int(zones[i]))
        _inc(row, f"{zone}_agent_ticks")
        if float(food[i]) > 0.0:
            _inc(row, f"{zone}_available_resource_agent_ticks")
        _inc(row, f"{zone}_food_amount_sum", float(food[i]))
        _inc(row, f"{zone}_cover_sum", float(cover[i]))
        _inc(row, f"{zone}_shelter_sum", float(shelter[i]))
        _inc(row, f"{zone}_predator_risk_sum", float(risk[i]))
        _inc(row, f"{zone}_movement_cost_sum", float(move[i]))
        if int(actions[i]) == ACT_GATHER:
            _inc(row, f"{zone}_gather_attempts")
        if int(actions[i]) == ACT_REST:
            _inc(row, f"{zone}_rest_actions")
        if int(actions[i]) in {ACT_MOVE_N, ACT_MOVE_E, ACT_MOVE_S, ACT_MOVE_W}:
            _inc(row, f"{zone}_move_actions")
            _inc(row, f"{zone}_movement_cost_before_move_actions", float(move[i]))
        if zone == "exposed_rich":
            _inc(row, "exposure_ticks")
        if zone == "safe_covered":
            _inc(row, "cover_occupancy_ticks")
        _state(runner)["last_pre"][str(int(aid))] = {
            "zone": zone,
            "action": int(actions[i]),
            "risk": float(risk[i]),
            "food": float(food[i]),
            "cover": float(cover[i]),
            "shelter": float(shelter[i]),
            "top_logit": float(np.max(logits[i])),
        }


def record_post_env(runner: Any, *, agent_ids: np.ndarray, actions: np.ndarray, food_gained: Any) -> None:
    if not telemetry_enabled(runner):
        return
    row = _gen_row(runner)
    gained = np.asarray(food_gained, dtype=np.float32)
    last = _state(runner).get("last_pre", {})
    for i, aid in enumerate(agent_ids):
        pre = last.get(str(int(aid)), {})
        zone = str(pre.get("zone", "none"))
        gain = float(gained[i]) if i < int(gained.size) else 0.0
        if gain > 0.0:
            _inc(row, f"{zone}_gather_successes")
            _inc(row, f"{zone}_food_gain", gain)
            if zone in {"exposed_rich", "safe_covered"}:
                gen = str(int(getattr(runner, "_eiel_fdp_cur_gen", getattr(runner, "_current_generation", 0)) or 0))
                successes = _state(runner).setdefault("zone_success_by_gen", {}).setdefault(gen, {})
                zone_ids = successes.setdefault(zone, [])
                if int(aid) not in zone_ids:
                    zone_ids.append(int(aid))


def _record_reproduction_coupling(runner: Any, *, generation: int) -> None:
    state = _state(runner)
    recorded = list(state.setdefault("repro_recorded_generations", []))
    gen_key = str(int(generation))
    if gen_key in recorded:
        return
    row = _gen_row(runner)
    successes_by_zone = dict((state.get("zone_success_by_gen") or {}).get(gen_key) or {})
    eligible = list(getattr(runner, "_eligible_parent_agents_gen", []) or [])
    selected = list(getattr(runner, "_selected_parent_agents_gen", []) or [])
    birth_contexts = list(getattr(runner, "_lineage_reproduction_funnel_birth_contexts_gen", []) or [])
    for zone in ("exposed_rich", "safe_covered"):
        success_ids = {int(aid) for aid in list(successes_by_zone.get(zone) or [])}
        eligible_ids = {_agent_id(agent) for agent in eligible if _agent_id(agent) in success_ids}
        selected_ids = {_agent_id(agent) for agent in selected if _agent_id(agent) in success_ids}
        _inc(row, f"{zone}_eligible_parent_events_after_zone_use", len(eligible_ids))
        _inc(row, f"{zone}_selected_parent_events_after_zone_use", len(selected_ids))
        births = 0
        for ctx in birth_contexts:
            if not isinstance(ctx, dict):
                continue
            try:
                parent_id = int(ctx.get("parent_id", ctx.get("parent_agent_id", -1)))
            except Exception:
                parent_id = -1
            if parent_id in success_ids:
                births += 1
        _inc(row, f"{zone}_successful_births_after_zone_use", births)
    recorded.append(gen_key)
    state["repro_recorded_generations"] = recorded


def _record_predator_damage(runner: Any) -> None:
    row = _gen_row(runner)
    env = getattr(runner, "env", None)
    grid = getattr(env, "grid", None)
    classes = getattr(env, "predrisk_fixed1_zone_class", None)
    if grid is None or classes is None:
        return
    h, w = tuple(np.asarray(grid).shape)
    zone_classes = np.asarray(classes, dtype=np.int32)
    for ctx in list(getattr(runner, "_jungle_predator_event_contexts_gen", []) or []):
        if not isinstance(ctx, dict):
            continue
        try:
            y = max(0, min(int(h) - 1, int(float(ctx.get("y", 0)))))
            x = max(0, min(int(w) - 1, int(float(ctx.get("x", 0)))))
        except Exception:
            continue
        zone = _zone_name(int(zone_classes[y, x]))
        _inc(row, f"{zone}_predator_events")
        _inc(row, f"{zone}_raw_predator_damage_cost", float(ctx.get("raw_damage", 0.0) or 0.0))
        _inc(row, f"{zone}_effective_predator_damage_cost", float(ctx.get("effective_damage", 0.0) or 0.0))
        _inc(row, f"{zone}_predator_damage_prevented", float(ctx.get("damage_prevented", 0.0) or 0.0))


def write_generation_summary(runner: Any, *, generation: int) -> None:
    if not telemetry_enabled(runner):
        return
    _record_reproduction_coupling(runner, generation=int(generation))
    _record_predator_damage(runner)
    metrics_dir = Path(getattr(runner, "run_dir", ".") or ".") / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    row = dict(_gen_row(runner))
    row["generation"] = float(generation)
    row["arm"] = str(getattr(runner.sim_cfg, "predrisk_fixed1_probe_arm", ""))
    path = metrics_dir / "predrisk_fixed1_generation_summary.csv"
    fieldnames = list(SUMMARY_FIELDS)
    for key in sorted(row.keys()):
        if key not in fieldnames:
            fieldnames.append(key)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    (metrics_dir / "predrisk_fixed1_probe_contract.json").write_text(
        json.dumps(
            {
                **controller_contract(),
                "arm": str(getattr(runner.sim_cfg, "predrisk_fixed1_probe_arm", "")),
                "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
                "action_routing": "ordinary action IDs enter existing environment apply path",
                "zone_label_use": "telemetry only",
                "recent_zone_use_definition": "at least one successful gather in the zone during the just-completed generation",
                "net_return_fields": {
                    "food_energy": "<zone>_food_gain",
                    "movement_cost": "<zone>_movement_cost_before_move_actions",
                    "effective_predator_damage_cost": "<zone>_effective_predator_damage_cost",
                    "available_resource_denominator": "<zone>_available_resource_agent_ticks",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
