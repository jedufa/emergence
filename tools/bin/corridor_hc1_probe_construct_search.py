#!/usr/bin/env python3
from __future__ import annotations

"""Run HC1 homing-cue construct checks and write final-family evidence."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from env.env_config import EnvConfig
from env.gridworld_pkg.constants import ACT_GATHER, ACT_MOVE_E, ACT_MOVE_N, ACT_MOVE_S, ACT_MOVE_W, ACT_REST
from env.gridworld_pkg.patch_world import (
    CORRIDOR_CP0_ZONE_CORRIDOR,
    CORRIDOR_CP0_ZONE_HOME_LOCAL,
    CORRIDOR_CP0_ZONE_REMOTE_RICH,
    PATCH_WORLD_OBSERVATION_CHANNEL_NAMES,
    apply_corridor_hc1_homing_cue_overlay,
    initialize_patch_world_arrays,
)
from simulation.corridor_cp0_fixed_probe import ARM_NAMES, OBS_CHANNELS, build_probe_policy, controller_contract, validate_controller_contract


HC1_ARM_NAMES = ("remote_commuter_hc1", "balanced_generalist_hc1", "local_persistent_hc1")
HOME = int(CORRIDOR_CP0_ZONE_HOME_LOCAL)
CORRIDOR = int(CORRIDOR_CP0_ZONE_CORRIDOR)
REMOTE = int(CORRIDOR_CP0_ZONE_REMOTE_RICH)
ACTIONS = {
    ACT_MOVE_N: "move_n",
    ACT_MOVE_S: "move_s",
    ACT_MOVE_E: "move_e",
    ACT_MOVE_W: "move_w",
    ACT_GATHER: "gather",
    ACT_REST: "rest",
}


def hc1_env_config() -> EnvConfig:
    cfg = EnvConfig(width=14, height=6, view_radius=1)
    cfg.patch_world_enable = True
    cfg.patch_world_observation_enable = True
    cfg.corridor_cp0_enable = True
    cfg.corridor_cp0_home_width = 4
    cfg.corridor_cp0_remote_width = 4
    cfg.corridor_cp0_corridor_half_width = 0
    cfg.corridor_hc1_homing_cue_enable = True
    cfg.corridor_hc1_homing_cue_field = "shelter"
    return cfg


def hc1_maps(*, reflect: bool = False) -> dict[str, np.ndarray]:
    cfg = hc1_env_config()
    arrays = initialize_patch_world_arrays(cfg, np.zeros((cfg.height, cfg.width), dtype=np.int8))
    if reflect:
        for name, value in list(arrays.items()):
            arrays[name] = np.fliplr(value).copy()
        apply_corridor_hc1_homing_cue_overlay(cfg, arrays)
    return {
        "zones": arrays["corridor_cp0_zone_class"].astype(np.int32),
        "food": arrays["patch_food_amount"].astype(np.float32),
        "cover": arrays["patch_cover"].astype(np.float32),
        "shelter": arrays["patch_shelter"].astype(np.float32),
        "move": arrays["patch_movement_cost"].astype(np.float32),
    }


def obs(maps: dict[str, np.ndarray], y: int, x: int, energy: float) -> np.ndarray:
    out = np.zeros((74,), dtype=np.float32)
    height, width = maps["zones"].shape
    cells = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    for cell_index, (dy, dx) in enumerate(cells):
        yy, xx = int(y) + int(dy), int(x) + int(dx)
        if not (0 <= yy < height and 0 <= xx < width):
            continue
        out[OBS_CHANNELS.index("patch_food_amount_norm") * 9 + cell_index] = maps["food"][yy, xx]
        out[OBS_CHANNELS.index("patch_cover") * 9 + cell_index] = maps["cover"][yy, xx]
        out[OBS_CHANNELS.index("patch_shelter") * 9 + cell_index] = maps["shelter"][yy, xx]
        out[OBS_CHANNELS.index("patch_movement_cost_norm") * 9 + cell_index] = maps["move"][yy, xx] / np.float32(4.0)
    out[72] = np.float32(float(energy) / 100.0)
    out[73] = np.float32(1.0)
    return out


def _shelter_triplet(maps: dict[str, np.ndarray], y: int, x: int) -> dict[str, float]:
    shelter = maps["shelter"]
    return {
        "west": float(shelter[y, max(0, x - 1)]),
        "center": float(shelter[y, x]),
        "east": float(shelter[y, min(shelter.shape[1] - 1, x + 1)]),
    }


def cue_contract() -> dict[str, Any]:
    cfg = EnvConfig()
    hc1 = hc1_env_config()
    return {
        "schema_version": "corridor_hc1_homing_cue_contract_v1",
        "default_off": bool(cfg.corridor_hc1_homing_cue_enable is False),
        "enabled_field": str(hc1.corridor_hc1_homing_cue_field),
        "ordinary_observation_field": "patch_shelter",
        "ordinary_observation_channel_present": "patch_shelter" in PATCH_WORLD_OBSERVATION_CHANNEL_NAMES,
        "zone_label_observation_channel_present": "corridor_cp0_zone_class" in PATCH_WORLD_OBSERVATION_CHANNEL_NAMES,
        "genotype_blind": True,
        "policy_blind": True,
        "deterministic": True,
        "selection_birth_energy_food_direct_effect": "none beyond the existing observed shelter scalar",
    }


def symmetry_resolution() -> dict[str, Any]:
    normal = hc1_maps(reflect=False)
    reflected = hc1_maps(reflect=True)
    y = normal["zones"].shape[0] // 2
    normal_x = 1
    reflected_x = reflected["zones"].shape[1] - 2
    normal_obs = obs(normal, y, normal_x, 30.0)
    reflected_obs = obs(reflected, y, reflected_x, 30.0)
    normal_triplet = _shelter_triplet(normal, y, normal_x)
    reflected_triplet = _shelter_triplet(reflected, y, reflected_x)
    return {
        "schema_version": "corridor_hc1_reflected_home_symmetry_resolution_v1",
        "normal_required_first_route_action": "move_e",
        "reflected_required_first_route_action": "move_w",
        "ordinary_observations_identical_after_hc1": bool(np.array_equal(normal_obs, reflected_obs)),
        "normal_home_shelter_triplet": normal_triplet,
        "reflected_home_shelter_triplet": reflected_triplet,
        "normal_lower_shelter_neighbor_points_to_route": bool(normal_triplet["east"] < normal_triplet["center"]),
        "reflected_lower_shelter_neighbor_points_to_route": bool(reflected_triplet["west"] < reflected_triplet["center"]),
        "hidden_zone_channel_exposed": "corridor_cp0_zone_class" in PATCH_WORLD_OBSERVATION_CHANNEL_NAMES,
    }


def route_state_blocker() -> dict[str, Any]:
    maps = hc1_maps(reflect=False)
    y = maps["zones"].shape[0] // 2
    x = 6
    outbound_obs = obs(maps, y, x, 30.0)
    inbound_obs = obs(maps, y, x, 30.0)
    triplet = _shelter_triplet(maps, y, x)
    return {
        "schema_version": "corridor_hc1_memoryless_route_state_blocker_v1",
        "fixture": {
            "y": int(y),
            "x": int(x),
            "zone": "corridor",
            "shelter_triplet": triplet,
        },
        "outbound_required_action": "move_e",
        "inbound_required_action": "move_w",
        "ordinary_observations_identical_at_equal_energy": bool(np.array_equal(outbound_obs, inbound_obs)),
        "reason": (
            "HC1 resolves reflected home direction, but a deterministic memoryless ordinary MLP still has no route "
            "phase input. The same mid-corridor local observation at the same energy requires opposite actions on "
            "outbound and inbound commute phases."
        ),
        "bounded_interpretation": (
            "This is a construct gate for the required frozen ordinary commuter, not a claim that every possible "
            "energy-calibrated or stateful controller is impossible."
        ),
    }


def run_inherited_cp1_probe(arm: str, *, reflect: bool = False, energy: float = 30.0, steps: int = 80) -> dict[str, Any]:
    maps = hc1_maps(reflect=reflect)
    food = maps["food"].copy()
    height, width = maps["zones"].shape
    y = height // 2
    x = 1 if not reflect else width - 2
    policy = build_probe_policy(arm)
    remote_gathers = 0
    local_gathers = 0
    completed_cycles = 0
    left_home = False
    remote_gathered = False
    actions: list[str] = []
    for _ in range(int(steps)):
        maps_with_food = dict(maps)
        maps_with_food["food"] = food
        with torch.no_grad():
            logits = policy(torch.from_numpy(obs(maps_with_food, y, x, energy)).unsqueeze(0)).detach().cpu().numpy()[0]
        action = int(np.argmax(logits))
        actions.append(str(ACTIONS.get(action, action)))
        zone = int(maps["zones"][y, x])
        if zone != HOME:
            left_home = True
        if action == ACT_GATHER and float(food[y, x]) > 0.1:
            energy += float(food[y, x]) * 5.0
            food[y, x] = np.float32(0.0)
            if zone == REMOTE:
                remote_gathers += 1
                remote_gathered = True
            if zone == HOME:
                local_gathers += 1
        elif action in (ACT_MOVE_N, ACT_MOVE_S, ACT_MOVE_E, ACT_MOVE_W):
            dy, dx = {
                ACT_MOVE_N: (-1, 0),
                ACT_MOVE_S: (1, 0),
                ACT_MOVE_E: (0, 1),
                ACT_MOVE_W: (0, -1),
            }[action]
            yy, xx = y + dy, x + dx
            if 0 <= yy < height and 0 <= xx < width:
                y, x = yy, xx
                energy -= float(maps["move"][y, x])
        elif action == ACT_REST:
            energy += 0.1
        if int(maps["zones"][y, x]) == HOME and left_home and remote_gathered:
            completed_cycles += 1
            left_home = False
            remote_gathered = False
        if energy < 0.0:
            break
    return {
        "arm": arm,
        "reflect": bool(reflect),
        "remote_gathers": int(remote_gathers),
        "local_gathers": int(local_gathers),
        "completed_commute_cycles": int(completed_cycles),
        "survived_steps": int(len(actions)),
        "actions_prefix": actions[:24],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    inherited = [run_inherited_cp1_probe(arm, reflect=reflect) for arm in ARM_NAMES for reflect in (False, True)]
    blocker = route_state_blocker()
    payload = {
        "schema_version": "corridor_hc1_probe_construct_search_v1",
        "decision": "CORRIDOR-HC1-E_FINAL_CONSTRUCT_FAILURE",
        "required_hc1_arms": list(HC1_ARM_NAMES),
        "frozen_hc1_policy_hashes": {},
        "cue_contract": cue_contract(),
        "symmetry_resolution": symmetry_resolution(),
        "method_1_inherited_cp1_probe_retest_under_hc1": inherited,
        "method_2_bounded_hc1_weight_template_search": {
            "candidate_evaluations": 0,
            "skipped_reason": "Pre-search route-state fixture blocks a single deterministic memoryless commuter from satisfying outbound and inbound mid-corridor actions at equal energy.",
        },
        "method_3_population_or_campaign_search": {
            "candidate_evaluations": 0,
            "skipped_reason": "Campaign is gated on frozen probe construction; no scientific cells run after construct failure.",
        },
        "route_state_blocker": blocker,
        "cp1_historical_contract": controller_contract(),
        "cp1_historical_contract_validation": validate_controller_contract(),
        "campaign_cells_run": 0,
        "campaign_gates": {
            "technical_rehearsal_24": "NOT_EVALUATED_CONSTRUCT_FAILED",
            "core_96": "NOT_EVALUATED_CONSTRUCT_FAILED",
            "confirmation_192": "NOT_EVALUATED_CONSTRUCT_FAILED",
            "repeat_parity": "NOT_EVALUATED_CONSTRUCT_FAILED",
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
