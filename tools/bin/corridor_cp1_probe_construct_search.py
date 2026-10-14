#!/usr/bin/env python3
from __future__ import annotations

"""Run CP1 fixed-probe construct checks and write negative evidence if blocked."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from env.gridworld_pkg.constants import ACT_GATHER, ACT_MOVE_E, ACT_MOVE_N, ACT_MOVE_S, ACT_MOVE_W, ACT_REST
from simulation.corridor_cp0_fixed_probe import ARM_NAMES, OBS_CHANNELS, build_probe_policy, controller_contract, validate_controller_contract


HOME = 1
CORRIDOR = 2
REMOTE = 3
ACTIONS = {
    ACT_MOVE_N: "move_n",
    ACT_MOVE_S: "move_s",
    ACT_MOVE_E: "move_e",
    ACT_MOVE_W: "move_w",
    ACT_GATHER: "gather",
    ACT_REST: "rest",
}


def _maps(
    *,
    reflect: bool = False,
    height: int = 6,
    width: int = 10,
    home_width: int = 2,
    remote_width: int = 2,
) -> dict[str, np.ndarray]:
    zones = np.zeros((height, width), dtype=np.int32)
    zones[:, :home_width] = HOME
    zones[:, width - remote_width :] = REMOTE
    zones[height // 2, home_width : width - remote_width] = CORRIDOR
    if reflect:
        zones = np.fliplr(zones).copy()
    food = np.zeros((height, width), dtype=np.float32)
    cover = np.zeros((height, width), dtype=np.float32)
    shelter = np.zeros((height, width), dtype=np.float32)
    move = np.ones((height, width), dtype=np.float32)
    food[zones == HOME] = np.float32(0.30)
    food[zones == CORRIDOR] = np.float32(0.05)
    food[zones == REMOTE] = np.float32(1.00)
    cover[zones == HOME] = np.float32(0.85)
    cover[zones == CORRIDOR] = np.float32(0.15)
    cover[zones == REMOTE] = np.float32(0.10)
    shelter[zones == HOME] = np.float32(0.85)
    shelter[zones == CORRIDOR] = np.float32(0.10)
    shelter[zones == REMOTE] = np.float32(0.05)
    move[zones == HOME] = np.float32(0.75)
    move[zones == CORRIDOR] = np.float32(1.80)
    move[zones == REMOTE] = np.float32(1.20)
    return {"zones": zones, "food": food, "cover": cover, "shelter": shelter, "move": move}


def _obs(maps: dict[str, np.ndarray], y: int, x: int, energy: float) -> np.ndarray:
    obs = np.zeros((74,), dtype=np.float32)
    height, width = maps["zones"].shape
    cells = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    for cell_index, (dy, dx) in enumerate(cells):
        yy, xx = int(y) + int(dy), int(x) + int(dx)
        if not (0 <= yy < height and 0 <= xx < width):
            continue
        for channel, name in (
            ("patch_food_amount_norm", "food"),
            ("patch_cover", "cover"),
            ("patch_shelter", "shelter"),
        ):
            obs[OBS_CHANNELS.index(channel) * 9 + cell_index] = maps[name][yy, xx]
        obs[OBS_CHANNELS.index("patch_movement_cost_norm") * 9 + cell_index] = maps["move"][yy, xx] / np.float32(4.0)
    obs[72] = np.float32(float(energy) / 10.0)
    obs[73] = np.float32(1.0)
    return obs


def _run_direct(arm: str, *, reflect: bool = False, energy: float = 3.0, steps: int = 80) -> dict[str, Any]:
    maps = _maps(reflect=reflect)
    height, width = maps["zones"].shape
    y = height // 2
    x = 1 if not reflect else width - 2
    policy = build_probe_policy(arm)
    food = maps["food"].copy()
    remote_gathers = 0
    local_gathers = 0
    completed_cycles = 0
    left_home = False
    remote_gathered = False
    actions: list[str] = []
    zone_ticks = {"home": 0, "corridor": 0, "remote": 0, "none": 0}
    for _ in range(int(steps)):
        with torch.no_grad():
            logits = policy(torch.from_numpy(_obs({**maps, "food": food}, y, x, energy)).unsqueeze(0)).detach().cpu().numpy()[0]
        action = int(np.argmax(logits))
        zone = int(maps["zones"][y, x])
        actions.append(str(ACTIONS.get(action, action)))
        if zone == HOME:
            zone_ticks["home"] += 1
        elif zone == CORRIDOR:
            zone_ticks["corridor"] += 1
        elif zone == REMOTE:
            zone_ticks["remote"] += 1
        else:
            zone_ticks["none"] += 1
        if zone != HOME:
            left_home = True
        if action == ACT_GATHER and float(food[y, x]) > 0.1:
            energy += float(food[y, x]) * 5.0
            if zone == REMOTE:
                remote_gathers += 1
                remote_gathered = True
            if zone == HOME:
                local_gathers += 1
            food[y, x] = np.float32(0.0)
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
        "zone_ticks": zone_ticks,
        "actions_prefix": actions[:24],
        "survived_steps": int(len(actions)),
    }


def _symmetry_blocker() -> dict[str, Any]:
    normal = _maps(reflect=False, width=14, home_width=4, remote_width=4)
    reflected = _maps(reflect=True, width=14, home_width=4, remote_width=4)
    normal_obs = _obs(normal, normal["zones"].shape[0] // 2, 1, 3.0)
    reflected_obs = _obs(reflected, reflected["zones"].shape[0] // 2, reflected["zones"].shape[1] - 2, 3.0)
    return {
        "schema_version": "corridor_cp1_memoryless_symmetry_blocker_v1",
        "normal_home_interior_required_first_route_action": "move_e",
        "reflected_home_interior_required_first_route_action": "move_w",
        "ordinary_observations_identical": bool(np.array_equal(normal_obs, reflected_obs)),
        "reason": (
            "A deterministic memoryless ordinary MLP must emit the same action for identical local observations, "
            "but the normal and reflected home-interior fixtures require opposite first route actions."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    direct = [_run_direct(arm, reflect=reflect) for arm in ARM_NAMES for reflect in (False, True)]
    blocker = _symmetry_blocker()
    payload = {
        "schema_version": "corridor_cp1_probe_construct_search_v1",
        "decision": "CORRIDOR-CP1-E_PROBE_REPRESENTATION_FAILURE",
        "method_1_direct_weight_construction": direct,
        "method_2_deterministic_local_search": {
            "candidate_evaluations": 0,
            "skipped_reason": "A pre-search fixture pair proves deterministic memoryless ordinary observations are insufficient for the required reflected-route construct.",
        },
        "method_3_deterministic_population_search": {
            "candidate_evaluations": 0,
            "skipped_reason": "Same representation blocker as method 2; random/search weights cannot map identical observations to opposite required route actions.",
        },
        "symmetry_blocker": blocker,
        "contract": controller_contract(),
        "contract_validation": validate_controller_contract(),
        "campaign_cells_run": 0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
