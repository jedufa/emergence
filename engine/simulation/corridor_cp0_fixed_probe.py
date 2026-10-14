from __future__ import annotations

"""Default-off CORRIDOR-CP0 frozen ordinary-MLP probe support.

The probes use the ordinary local Patch World observation vector and return
ordinary action IDs. Hidden CP0 zone labels are used only by construct scoring
and telemetry, never by policy inference.
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
from env.gridworld_pkg.patch_world import (
    CORRIDOR_CP0_ZONE_CORRIDOR,
    CORRIDOR_CP0_ZONE_HOME_LOCAL,
    CORRIDOR_CP0_ZONE_REMOTE_RICH,
    PATCH_WORLD_OBSERVATION_CHANNEL_NAMES,
    patch_world_local_observation_channel_count,
    patch_world_observation_enabled,
)


SCHEMA_VERSION = "corridor_cp0_fixed_probe_mlp_v1"
TELEMETRY_SCHEMA_VERSION = "corridor_cp0_route_telemetry_v1"
ARM_NAMES = ("remote_commuter", "balanced_generalist", "local_persistent")
MOVE_ACTIONS = (ACT_MOVE_N, ACT_MOVE_S, ACT_MOVE_E, ACT_MOVE_W)
MOVE_CELLS = ("north", "south", "east", "west")
OBS_CHANNELS = ("legacy_food_occupancy", *PATCH_WORLD_OBSERVATION_CHANNEL_NAMES[1:])
ZONE_NAMES = ("none", "home_local", "corridor", "remote_rich")
SUMMARY_FIELDS = (
    "arm",
    "generation",
    "home_local_ticks",
    "corridor_ticks",
    "remote_rich_ticks",
    "remote_successful_gathers",
    "local_successful_gathers",
    "completed_commute_cycles",
    "returns_to_home_after_remote_gather",
    "time_outside_home_eligibility",
    "eligible_parent_events_home",
    "selected_parent_events_after_completed_commute",
    "successful_births_after_completed_commute",
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
        raise ValueError(f"unsupported corridor_cp0_probe_arm: {arm}")
    policy = _zero_policy()
    first = policy.net[0]
    head = policy.net[4]
    with torch.no_grad():
        for unit, (action, cell) in enumerate(zip(MOVE_ACTIONS, MOVE_CELLS)):
            first.weight[unit, _obs_index("patch_food_amount_norm", cell)] = 2.0
            first.weight[unit, _obs_index("patch_cover", cell)] = -2.2
            first.weight[unit, _obs_index("patch_shelter", cell)] = -2.6
            first.weight[unit, _obs_index("patch_movement_cost_norm", cell)] = -0.8
            first.weight[unit, 72] = -1.4
            first.bias[unit] = 1.4
            head.weight[action, unit] = 4.0

        for unit, (action, cell) in enumerate(zip(MOVE_ACTIONS, MOVE_CELLS), start=4):
            first.weight[unit, _obs_index("patch_cover", cell)] = 2.2
            first.weight[unit, _obs_index("patch_shelter", cell)] = 2.8
            first.weight[unit, _obs_index("patch_movement_cost_norm", cell)] = -0.3
            first.weight[unit, 72] = 1.6
            first.bias[unit] = -1.6
            head.weight[action, unit] = 4.0

        first.weight[8, _obs_index("patch_food_amount_norm", "center")] = 7.0
        first.bias[8] = -4.2
        head.weight[ACT_GATHER, 8] = 5.0

        first.weight[9, 73] = -6.0
        first.bias[9] = 1.0
        head.weight[ACT_REST, 9] = 5.0

        if arm == "remote_commuter":
            first.weight[10, 72] = -3.0
            first.bias[10] = 2.2
            head.weight[ACT_MOVE_E, 10] = 1.4
            head.weight[ACT_MOVE_W, 10] = 1.4
        elif arm == "balanced_generalist":
            first.weight[10, 72] = -2.0
            first.bias[10] = 1.1
            head.weight[ACT_MOVE_E, 10] = 0.5
            head.weight[ACT_MOVE_W, 10] = 0.5
            head.bias[ACT_REST] = 0.2
        else:
            for unit, action in enumerate(MOVE_ACTIONS):
                head.weight[action, unit] *= 0.20
            first.weight[11, _obs_index("patch_shelter", "center")] = 4.0
            first.weight[11, _obs_index("patch_food_amount_norm", "center")] = 2.0
            first.bias[11] = 0.2
            head.weight[ACT_GATHER, 11] = 1.4
            head.weight[ACT_REST, 11] = 1.2
    policy.eval()
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
            "corridor_cp0_zone_class",
            "home eligibility",
            "route history",
            "global map",
            "future state",
            "lineage identity",
            "branch identity",
            "selection outcomes",
            "reward",
        ],
    }


def validate_controller_contract() -> dict[str, Any]:
    contract = controller_contract()
    excluded = set(contract["hidden_state_excluded_from_action"])
    return {
        "schema_version": "corridor_cp0_fixed_probe_contract_validation_v1",
        "ok": bool(
            contract["arms"] == list(ARM_NAMES)
            and contract["obs_size"] == 74
            and "corridor_cp0_zone_class" in excluded
            and all(len(value) == 64 for value in contract["policy_hashes"].values())
        ),
        "arms": list(contract["arms"]),
        "policy_hashes": dict(contract["policy_hashes"]),
        "zone_label_excluded": "corridor_cp0_zone_class" in excluded,
    }


def probe_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    if not bool(getattr(cfg, "corridor_cp0_probe_enable", False)):
        return False
    if int(getattr(cfg, "corridor_cp0_probe_schema_version", 1) or 1) != 1:
        raise ValueError("unsupported corridor_cp0_probe_schema_version")
    arm = str(getattr(cfg, "corridor_cp0_probe_arm", "") or "")
    if arm not in ARM_NAMES:
        raise ValueError(f"unsupported corridor_cp0_probe_arm: {arm}")
    if str(getattr(cfg, "corridor_cp0_probe_controller_version", "corridor_cp0_mlp_v1")) != "corridor_cp0_mlp_v1":
        raise ValueError("unsupported corridor_cp0_probe_controller_version")
    env_cfg = getattr(runner, "env_cfg", None)
    if not bool(getattr(env_cfg, "patch_world_enable", False)):
        raise ValueError("corridor_cp0_probe_enable requires patch_world_enable")
    if not patch_world_observation_enabled(env_cfg):
        raise ValueError("corridor_cp0_probe_enable requires patch_world_observation_enable")
    if int(getattr(env_cfg, "view_radius", 1) or 1) != 1:
        raise ValueError("corridor_cp0_probe_enable currently requires view_radius=1")
    return True


def telemetry_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    if not bool(getattr(cfg, "corridor_cp0_telemetry_enable", False)):
        return False
    if int(getattr(cfg, "corridor_cp0_telemetry_schema_version", 1) or 1) != 1:
        raise ValueError("unsupported corridor_cp0_telemetry_schema_version")
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
    arm = str(getattr(runner.sim_cfg, "corridor_cp0_probe_arm"))
    policy = _probe_policy_cache(runner, arm)
    agent_ids, obs = _obs_batch(runner, alive)
    with torch.no_grad():
        logits = policy(torch.from_numpy(obs)).detach().cpu().numpy()
    actions = np.asarray(np.argmax(logits, axis=1), dtype=np.int32)
    _record_pre_env(runner, agent_ids=agent_ids, actions=actions)
    return actions


def _probe_policy_cache(runner: Any, arm: str) -> MLPPolicy:
    cache = getattr(runner, "_corridor_cp0_policy_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(runner, "_corridor_cp0_policy_cache", cache)
    if arm not in cache:
        cache[arm] = build_probe_policy(arm)
    return cache[arm]


def _zone_name(value: int) -> str:
    if int(value) == CORRIDOR_CP0_ZONE_HOME_LOCAL:
        return "home_local"
    if int(value) == CORRIDOR_CP0_ZONE_CORRIDOR:
        return "corridor"
    if int(value) == CORRIDOR_CP0_ZONE_REMOTE_RICH:
        return "remote_rich"
    return "none"


def _state(runner: Any) -> dict[str, Any]:
    state = getattr(runner, "_corridor_cp0_state", None)
    if not isinstance(state, dict):
        state = {"schema_version": TELEMETRY_SCHEMA_VERSION, "generations": {}, "agent_route": {}}
        setattr(runner, "_corridor_cp0_state", state)
    return state


def _gen_row(runner: Any) -> dict[str, float]:
    gen = int(getattr(runner, "_eiel_fdp_cur_gen", getattr(runner, "_current_generation", 0)) or 0)
    return _state(runner).setdefault("generations", {}).setdefault(str(gen), {"generation": float(gen)})


def _inc(row: dict[str, float], key: str, amount: float = 1.0) -> None:
    row[key] = float(row.get(key, 0.0) or 0.0) + float(amount)


def _record_pre_env(runner: Any, *, agent_ids: np.ndarray, actions: np.ndarray) -> None:
    if not telemetry_enabled(runner):
        return
    row = _gen_row(runner)
    env = getattr(runner, "env", None)
    ys = np.asarray(getattr(env, "agent_pos_y"))[agent_ids]
    xs = np.asarray(getattr(env, "agent_pos_x"))[agent_ids]
    zones = np.asarray(getattr(env, "corridor_cp0_zone_class", np.zeros_like(getattr(env, "grid"))), dtype=np.int32)[ys, xs]
    route = _state(runner).setdefault("agent_route", {})
    for i, aid in enumerate(agent_ids):
        zone = _zone_name(int(zones[i]))
        _inc(row, f"{zone}_ticks")
        if zone != "home_local":
            _inc(row, "time_outside_home_eligibility")
        rec = route.setdefault(str(int(aid)), {"left_home": False, "remote_gathered": False})
        if zone != "home_local":
            rec["left_home"] = True
        if int(actions[i]) == ACT_GATHER and zone == "remote_rich":
            rec["remote_gathered"] = True
            _inc(row, "remote_successful_gathers")
        if int(actions[i]) == ACT_GATHER and zone == "home_local":
            _inc(row, "local_successful_gathers")
        if zone == "home_local" and bool(rec.get("left_home")) and bool(rec.get("remote_gathered")):
            _inc(row, "returns_to_home_after_remote_gather")
            _inc(row, "completed_commute_cycles")
            rec["left_home"] = False
            rec["remote_gathered"] = False


def write_generation_summary(runner: Any, *, generation: int) -> None:
    if not telemetry_enabled(runner):
        return
    row = dict(_gen_row(runner))
    row["generation"] = float(generation)
    row["arm"] = str(getattr(runner.sim_cfg, "corridor_cp0_probe_arm", ""))
    metrics_dir = Path(getattr(runner, "run_dir", ".") or ".") / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "corridor_cp0_generation_summary.csv"
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
    (metrics_dir / "corridor_cp0_probe_contract.json").write_text(
        json.dumps(controller_contract(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
