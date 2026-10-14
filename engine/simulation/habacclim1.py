from __future__ import annotations

"""Default-off HABACCLIM-1 acquired habitat-acclimation support."""

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
    HABACCLIM_HABITAT_A,
    HABACCLIM_HABITAT_B,
    PATCH_WORLD_OBSERVATION_CHANNEL_NAMES,
    habacclim_acclimation_rate,
    habacclim_enabled,
    habacclim_map_digest,
    habacclim_observation_enabled,
    habacclim_observation_scalar_index,
    patch_world_local_observation_channel_count,
    patch_world_total_observation_size,
)


SCHEMA_VERSION = "habacclim1_mlp_probe_v1"
TELEMETRY_SCHEMA_VERSION = "habacclim1_telemetry_v1"
ARM_NAMES = ("habitat_a_committed", "balanced_switcher", "habitat_b_committed")
OBS_CHANNELS = ("legacy_food_occupancy", *PATCH_WORLD_OBSERVATION_CHANNEL_NAMES[1:])
MOVE_ACTIONS = (ACT_MOVE_N, ACT_MOVE_E, ACT_MOVE_S, ACT_MOVE_W)
MOVE_CELLS = ("north", "east", "south", "west")
HABACCLIM1R_SELECTED_POLICY_HASHES = {
    "habitat_a_committed": "5bad94bda2227fef64059949f8eb4a9aef746f0507852a9194d8938317fe8963",
    "balanced_switcher": "6e876639362346699a265e3c046a53efbfb90b7dbcc2746fe2ded558bf337f91",
    "habitat_b_committed": "32c68423da0c86e94329b3f0646a1888402cc2a27cb3513af5c518455ffbc20b",
}
SUMMARY_FIELDS = (
    "arm",
    "generation",
    "habitat_a_ticks",
    "habitat_b_ticks",
    "transitions",
    "gather_attempts",
    "gather_successes",
    "raw_food_gain",
    "effective_food_gain",
    "processing_penalty_energy",
    "cumulative_mismatch",
    "matched_ticks",
    "mismatched_ticks",
    "transitional_ticks",
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


def habitat_target(value: int) -> float:
    if int(value) == int(HABACCLIM_HABITAT_A):
        return 1.0
    if int(value) == int(HABACCLIM_HABITAT_B):
        return -1.0
    return 0.0


def _state(runner: Any) -> dict[str, Any]:
    state = getattr(runner, "_habacclim1_state", None)
    if not isinstance(state, dict):
        state = {
            "last_habitat": {},
            **{field: 0.0 for field in SUMMARY_FIELDS if field not in {"arm", "generation"}},
        }
        setattr(runner, "_habacclim1_state", state)
    return state


def _penalty_enabled(cfg: Any) -> bool:
    control = str(getattr(cfg, "habacclim_control_mode", "acclimation_tradeoff") or "acclimation_tradeoff")
    return control in {"acclimation_tradeoff", "uncorrelated_habitat_cost"}


def _zero_lag(cfg: Any) -> bool:
    return str(getattr(cfg, "habacclim_control_mode", "") or "") == "zero_lag_control"


def apply_post_env(
    runner: Any,
    *,
    alive: list[Any],
    agent_ids: np.ndarray,
    actions: np.ndarray,
    new_energy: np.ndarray,
    food_gained: np.ndarray,
) -> None:
    """Apply HABACCLIM mismatch consequence and update acquired state."""
    env_cfg = getattr(runner, "env_cfg", None)
    if not habacclim_enabled(env_cfg):
        return
    env = getattr(runner, "env", None)
    classes = getattr(env, "habacclim_habitat_class", None)
    if classes is None:
        raise RuntimeError("habacclim_enable requires habacclim_habitat_class")
    pos_y = np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32)
    pos_x = np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32)
    class_arr = np.asarray(classes, dtype=np.int32)
    penalty_max = max(0.0, min(1.0, float(getattr(env_cfg, "habacclim_processing_penalty_max", 0.35) or 0.0)))
    rate = float(habacclim_acclimation_rate(env_cfg))
    state = _state(runner)
    penalty_on = _penalty_enabled(env_cfg)
    zero_lag = _zero_lag(env_cfg)

    for i, agent in enumerate(alive):
        aid = int(agent_ids[i])
        if aid < 0 or aid >= len(pos_y):
            continue
        y = int(pos_y[aid])
        x = int(pos_x[aid])
        if y < 0 or x < 0:
            continue
        habitat = int(class_arr[y, x])
        target = habitat_target(habitat)
        pre_state = float(getattr(agent, "habitat_acclimation_state", 0.0) or 0.0)
        if zero_lag:
            pre_state = target
        mismatch = abs(target - pre_state) / 2.0 if habitat in (HABACCLIM_HABITAT_A, HABACCLIM_HABITAT_B) else 0.0
        mismatch = max(0.0, min(1.0, float(mismatch)))
        raw_gain = max(0.0, float(food_gained[i]))
        loss = raw_gain * penalty_max * mismatch if penalty_on else 0.0
        if loss > 0.0:
            food_gained[i] = np.float32(max(0.0, raw_gain - loss))
            new_energy[i] = np.float32(max(0.0, float(new_energy[i]) - loss))

        if zero_lag:
            next_state = target
        else:
            next_state = max(-1.0, min(1.0, pre_state + rate * (target - pre_state)))
        agent.habitat_acclimation_state = float(next_state)

        if habitat == HABACCLIM_HABITAT_A:
            state["habitat_a_ticks"] = float(state.get("habitat_a_ticks", 0.0)) + 1.0
        elif habitat == HABACCLIM_HABITAT_B:
            state["habitat_b_ticks"] = float(state.get("habitat_b_ticks", 0.0)) + 1.0
        last = state["last_habitat"].get(aid)
        if last in (HABACCLIM_HABITAT_A, HABACCLIM_HABITAT_B) and habitat != last:
            state["transitions"] = float(state.get("transitions", 0.0)) + 1.0
        state["last_habitat"][aid] = habitat
        state["cumulative_mismatch"] = float(state.get("cumulative_mismatch", 0.0)) + mismatch
        state["raw_food_gain"] = float(state.get("raw_food_gain", 0.0)) + raw_gain
        state["effective_food_gain"] = float(state.get("effective_food_gain", 0.0)) + max(0.0, raw_gain - loss)
        state["processing_penalty_energy"] = float(state.get("processing_penalty_energy", 0.0)) + loss
        if int(actions[i]) == ACT_GATHER:
            state["gather_attempts"] = float(state.get("gather_attempts", 0.0)) + 1.0
            if raw_gain > 0.0:
                state["gather_successes"] = float(state.get("gather_successes", 0.0)) + 1.0
        if mismatch <= 0.25:
            state["matched_ticks"] = float(state.get("matched_ticks", 0.0)) + 1.0
        elif mismatch >= 0.75:
            state["mismatched_ticks"] = float(state.get("mismatched_ticks", 0.0)) + 1.0
        else:
            state["transitional_ticks"] = float(state.get("transitional_ticks", 0.0)) + 1.0


def _zero_policy() -> MLPPolicy:
    policy = MLPPolicy(obs_size=75, hidden_size=16, n_actions=7)
    with torch.no_grad():
        for param in policy.parameters():
            param.zero_()
        policy.net[2].weight.copy_(torch.eye(16))
    return policy


def build_probe_policy(arm: str) -> MLPPolicy:
    if arm not in ARM_NAMES:
        raise ValueError(f"unsupported habacclim1_probe_arm: {arm}")
    policy = _zero_policy()
    first = policy.net[0]
    head = policy.net[4]
    acclim_idx = 74
    with torch.no_grad():
        if arm == "habitat_a_committed":
            move_weights = {
                "patch_food_amount_norm": 3.4,
                "patch_scent_norm": 3.0,
                "patch_cover": -2.2,
                "patch_shelter": -2.4,
                "patch_movement_cost_norm": -0.7,
            }
        elif arm == "habitat_b_committed":
            move_weights = {
                "patch_food_amount_norm": 0.8,
                "patch_scent_norm": -2.6,
                "patch_cover": 3.0,
                "patch_shelter": 3.2,
                "patch_movement_cost_norm": -1.0,
            }
        else:
            move_weights = {
                "patch_food_amount_norm": 2.1,
                "patch_scent_norm": 0.4,
                "patch_cover": 1.3,
                "patch_shelter": 1.3,
                "patch_movement_cost_norm": -0.9,
            }
        for unit, (action, cell) in enumerate(zip(MOVE_ACTIONS, MOVE_CELLS)):
            for channel, weight in move_weights.items():
                first.weight[unit, _obs_index(channel, cell)] = weight
            first.weight[unit, 72] = -0.5
            first.bias[unit] = 0.5
            head.weight[action, unit] = 3.0
        first.weight[8, _obs_index("patch_food_amount_norm", "center")] = 6.0
        if arm == "habitat_a_committed":
            first.weight[8, _obs_index("patch_scent_norm", "center")] = 2.0
            first.weight[8, _obs_index("patch_shelter", "center")] = -1.0
        elif arm == "habitat_b_committed":
            first.weight[8, _obs_index("patch_shelter", "center")] = 2.0
            first.weight[8, _obs_index("patch_scent_norm", "center")] = -1.0
        else:
            first.weight[8, _obs_index("patch_food_amount_norm", "center")] = 6.0
            first.weight[8, _obs_index("patch_shelter", "center")] = 0.8
            first.weight[8, acclim_idx] = -0.2
        first.bias[8] = -2.0 if arm == "balanced_switcher" else -3.0
        head.weight[ACT_GATHER, 8] = 4.0
        first.weight[9, 73] = -5.0
        first.bias[9] = 1.2
        head.weight[ACT_REST, 9] = 4.0
        if arm == "habitat_a_committed":
            first.weight[10, acclim_idx] = 3.5
            first.bias[10] = 0.2
            head.weight[ACT_MOVE_E, 10] = 2.0
            head.weight[ACT_GATHER, 10] = 1.5
        elif arm == "habitat_b_committed":
            first.weight[10, acclim_idx] = -3.5
            first.bias[10] = 0.2
            head.weight[ACT_MOVE_W, 10] = 2.0
            head.weight[ACT_GATHER, 10] = 1.5
        else:
            first.weight[10, acclim_idx] = -2.2
            first.bias[10] = 0.3
            head.weight[ACT_MOVE_E, 10] = 1.1
            first.weight[11, acclim_idx] = 2.2
            first.bias[11] = 0.3
            head.weight[ACT_MOVE_W, 11] = 1.1
            head.bias[ACT_REST] = 0.05
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


def controller_contract(env_cfg: Any | None = None) -> dict[str, Any]:
    scalar_idx = 74 if env_cfg is None else int(habacclim_observation_scalar_index(env_cfg))
    return {
        "schema_version": SCHEMA_VERSION,
        "arms": list(ARM_NAMES),
        "policy_type": "MLPPolicy",
        "obs_size": 75,
        "hidden_size": 16,
        "n_actions": 7,
        "habitat_acclimation_state_norm_index": scalar_idx,
        "policy_hashes": policy_hashes(),
        "habacclim1r_selected_policy_hashes": HABACCLIM1R_SELECTED_POLICY_HASHES,
        "ordinary_inputs": [*OBS_CHANNELS, "energy_norm", "health_norm", "habitat_acclimation_state_norm"],
        "hidden_state_excluded_from_action": [
            "habacclim_habitat_class",
            "habitat label",
            "strategy label",
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
    if not bool(getattr(cfg, "habacclim1_probe_enable", False)):
        return False
    if int(getattr(cfg, "habacclim1_probe_schema_version", 1) or 1) != 1:
        raise ValueError("unsupported habacclim1_probe_schema_version")
    arm = str(getattr(cfg, "habacclim1_probe_arm", "") or "")
    if arm not in ARM_NAMES:
        raise ValueError(f"unsupported habacclim1_probe_arm: {arm}")
    if str(getattr(cfg, "habacclim1_probe_controller_version", "habacclim1_mlp_v1")) != "habacclim1_mlp_v1":
        raise ValueError("unsupported habacclim1_probe_controller_version")
    env_cfg = getattr(runner, "env_cfg", None)
    if not habacclim_enabled(env_cfg):
        raise ValueError("habacclim1_probe_enable requires habacclim_enable")
    if not habacclim_observation_enabled(env_cfg):
        raise ValueError("habacclim1_probe_enable requires habacclim_observation_enable")
    if int(patch_world_total_observation_size(env_cfg)) != 75:
        raise ValueError("habacclim1_probe_enable requires obs_size=75")
    if int(getattr(env_cfg, "view_radius", 1) or 1) != 1:
        raise ValueError("habacclim1_probe_enable currently requires view_radius=1")
    return True


def telemetry_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    if not bool(getattr(cfg, "habacclim1_telemetry_enable", False)):
        return False
    if int(getattr(cfg, "habacclim1_telemetry_schema_version", 1) or 1) != 1:
        raise ValueError("unsupported habacclim1_telemetry_schema_version")
    return probe_enabled(runner)


def _obs_batch(runner: Any, alive: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    agent_ids = np.asarray([int(getattr(agent, "agent_id")) for agent in alive], dtype=np.int32)
    env_cfg = getattr(runner, "env_cfg", None)
    channels = int(patch_world_local_observation_channel_count(env_cfg))
    views = np.asarray(runner.env.get_local_observations(agent_ids), dtype=np.float32).reshape(
        (int(agent_ids.size), channels, 3, 3)
    )
    local = views.reshape((int(agent_ids.size), channels * 9))
    obs = np.zeros((int(agent_ids.size), patch_world_total_observation_size(env_cfg)), dtype=np.float32)
    obs[:, : local.shape[1]] = local
    energy_norm = float(max(1e-9, float(getattr(runner.env_cfg, "max_energy", 160.0) or 160.0)))
    health_norm = float(max(1e-9, float(getattr(runner.env_cfg, "start_health", 12.0) or 12.0)))
    acclim_idx = int(habacclim_observation_scalar_index(env_cfg))
    for i, agent in enumerate(alive):
        obs[i, local.shape[1]] = np.float32(float(getattr(agent, "energy", energy_norm)) / energy_norm)
        obs[i, local.shape[1] + 1] = np.float32(float(getattr(agent, "health", health_norm)) / health_norm)
        obs[i, acclim_idx] = np.float32(float(getattr(agent, "habitat_acclimation_state", 0.0) or 0.0))
    return agent_ids, obs


def choose_actions(runner: Any, alive: list[Any], raw_actions: Any) -> np.ndarray:
    if not probe_enabled(runner):
        return np.asarray(raw_actions, dtype=np.int32)
    if not alive:
        return np.empty((0,), dtype=np.int32)
    arm = str(getattr(runner.sim_cfg, "habacclim1_probe_arm"))
    cache = getattr(runner, "_habacclim1_policy_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(runner, "_habacclim1_policy_cache", cache)
    if arm not in cache:
        cache[arm] = build_probe_policy(arm)
    agent_ids, obs = _obs_batch(runner, alive)
    with torch.no_grad():
        logits = cache[arm](torch.from_numpy(obs)).detach().cpu().numpy()
    actions = np.asarray(np.argmax(logits, axis=1), dtype=np.int32)
    setattr(runner, "_habacclim1_last_obs_sha256", hashlib.sha256(obs.tobytes(order="C")).hexdigest())
    setattr(runner, "_habacclim1_last_agent_ids", agent_ids)
    return actions


def write_generation_summary(runner: Any, *, generation: int) -> None:
    if not telemetry_enabled(runner):
        return
    metrics_dir = Path(str(getattr(runner, "run_dir", ""))) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    state = _state(runner)
    row = {field: state.get(field, 0.0) for field in SUMMARY_FIELDS}
    row["arm"] = str(getattr(runner.sim_cfg, "habacclim1_probe_arm", ""))
    row["generation"] = int(generation)
    path = metrics_dir / "habacclim1_generation_summary.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(SUMMARY_FIELDS))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    contract = controller_contract(getattr(runner, "env_cfg", None))
    env = getattr(runner, "env", None)
    if env is not None and hasattr(env, "habacclim_habitat_class"):
        contract["map_digest"] = habacclim_map_digest(getattr(env, "habacclim_habitat_class"))
    (metrics_dir / "habacclim1_probe_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
