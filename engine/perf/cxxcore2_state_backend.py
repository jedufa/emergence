from __future__ import annotations

"""CXXCORE-2 persistent env-apply state-backend preflight helpers.

This module is diagnostic-only. It models the no-RNG CXXCORE-1 env-apply
subset while keeping copied state mutable across many ticks.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from perf.cxxcore_env_apply_shadow import EnvApplyShadowConfig, compare_shadow_deltas


SCHEMA_VERSION = "cxxcore2_state_backend_v1"


@dataclass(frozen=True)
class StateBackendInputs:
    config: EnvApplyShadowConfig
    agent_ids: np.ndarray
    initial_energy: np.ndarray
    initial_health: np.ndarray
    initial_injury: np.ndarray
    initial_scar: np.ndarray
    pos_y_by_id: np.ndarray
    pos_x_by_id: np.ndarray
    agent_grid: np.ndarray
    grid: np.ndarray
    move_cost_mults: np.ndarray
    juvenile_harvest_mults: np.ndarray
    actions_by_tick: np.ndarray


def as_backend_inputs(
    *,
    config: EnvApplyShadowConfig,
    agent_ids: Any,
    initial_energy: Any,
    initial_health: Any,
    initial_injury: Any,
    initial_scar: Any,
    pos_y_by_id: Any,
    pos_x_by_id: Any,
    agent_grid: Any,
    grid: Any,
    actions_by_tick: Any,
    move_cost_mults: Any | None = None,
    juvenile_harvest_mults: Any | None = None,
) -> StateBackendInputs:
    ids = np.ascontiguousarray(agent_ids, dtype=np.int32)
    actions = np.ascontiguousarray(actions_by_tick, dtype=np.int32)
    if actions.ndim != 2:
        raise ValueError("actions_by_tick must be a 2D int32 matrix")
    n = int(ids.shape[0])
    if int(actions.shape[1]) != n:
        raise ValueError("actions_by_tick columns must match agent_ids")
    energy = np.ascontiguousarray(initial_energy, dtype=np.float32)
    health = np.ascontiguousarray(initial_health, dtype=np.float32)
    injury = np.ascontiguousarray(initial_injury, dtype=np.float32)
    scar = np.ascontiguousarray(initial_scar, dtype=np.float32)
    for name, arr in (("energy", energy), ("health", health), ("injury", injury), ("scar", scar)):
        if int(arr.shape[0]) != n:
            raise ValueError(f"{name} length must match agent_ids")
    if move_cost_mults is None:
        move_mult = np.ones((n,), dtype=np.float32)
    else:
        move_mult = np.ascontiguousarray(move_cost_mults, dtype=np.float32)
    if juvenile_harvest_mults is None:
        harvest_mult = np.ones((n,), dtype=np.float32)
    else:
        harvest_mult = np.ascontiguousarray(juvenile_harvest_mults, dtype=np.float32)
    return StateBackendInputs(
        config=config,
        agent_ids=ids,
        initial_energy=energy,
        initial_health=health,
        initial_injury=injury,
        initial_scar=scar,
        pos_y_by_id=np.ascontiguousarray(pos_y_by_id, dtype=np.int32),
        pos_x_by_id=np.ascontiguousarray(pos_x_by_id, dtype=np.int32),
        agent_grid=np.ascontiguousarray(agent_grid, dtype=np.int32),
        grid=np.ascontiguousarray(grid, dtype=np.int8),
        move_cost_mults=move_mult,
        juvenile_harvest_mults=harvest_mult,
        actions_by_tick=actions,
    )


def run_backend_python_reference(inputs: StateBackendInputs, *, checkpoint_interval: int = 0) -> dict[str, np.ndarray]:
    """Run the no-RNG env-apply subset over mutable copied arrays."""
    ids = inputs.agent_ids
    n = int(ids.shape[0])
    order = np.argsort(ids, kind="mergesort")
    energy = inputs.initial_energy.astype(np.float32, copy=True)
    health = inputs.initial_health.astype(np.float32, copy=True)
    injury = inputs.initial_injury.astype(np.float32, copy=True)
    scar = inputs.initial_scar.astype(np.float32, copy=True)
    pos_y = inputs.pos_y_by_id.astype(np.int32, copy=True)
    pos_x = inputs.pos_x_by_id.astype(np.int32, copy=True)
    occ = inputs.agent_grid.astype(np.int32, copy=True)
    grid = inputs.grid.astype(np.int8, copy=True)
    died = np.zeros((n,), dtype=bool)
    checkpoint_interval = int(checkpoint_interval or 0)
    checkpoint_energy: list[np.ndarray] = []
    checkpoint_pos_y: list[np.ndarray] = []
    checkpoint_pos_x: list[np.ndarray] = []
    checkpoint_died: list[np.ndarray] = []

    cfg = inputs.config
    for tick in range(int(inputs.actions_by_tick.shape[0])):
        actions = inputs.actions_by_tick[tick]
        costs = np.empty((n,), dtype=np.float32)
        for src in order.tolist():
            action = int(actions[src])
            cost = float(cfg.move_cost) * float(inputs.move_cost_mults[src])
            if action == 4:
                cost = float(cfg.gather_cost)
            elif action == 5:
                cost = float(cfg.rest_cost)
            elif action == 6:
                cost = float(cfg.attack_cost)
            if float(cfg.action_cost_mult) != 1.0:
                cost *= float(cfg.action_cost_mult)
            if float(cfg.rest_cost_mult) != 1.0 and action == 5:
                cost *= float(cfg.rest_cost_mult)
            costs[src] = np.float32(cost)

        food_gained = np.zeros((n,), dtype=np.float32)
        for src in order.tolist():
            action = int(actions[src])
            aid = int(ids[src])
            y = int(pos_y[aid])
            x = int(pos_x[aid])
            if 0 <= action <= 3:
                ny = y + (-1 if action == 0 else 1 if action == 1 else 0)
                nx = x + (1 if action == 2 else -1 if action == 3 else 0)
                if 0 <= ny < int(cfg.height) and 0 <= nx < int(cfg.width) and int(occ[ny, nx]) == -1:
                    if int(occ[y, x]) == aid:
                        occ[y, x] = -1
                    occ[ny, nx] = aid
                    pos_y[aid] = np.int32(ny)
                    pos_x[aid] = np.int32(nx)
            elif action == 4:
                if int(grid[y, x]) == 1:
                    grid[y, x] = np.int8(0)
                    food_gained[src] = np.float32(float(cfg.food_energy) * float(inputs.juvenile_harvest_mults[src]))

        energy = energy - costs + food_gained
        energy = np.minimum(energy, np.float32(float(cfg.max_energy)))
        energy = np.maximum(energy, np.float32(0.0)).astype(np.float32, copy=False)
        died = ((energy <= 0.0) | (health <= 0.0)).astype(bool, copy=False)
        if checkpoint_interval > 0 and ((tick + 1) % checkpoint_interval == 0 or (tick + 1) == int(inputs.actions_by_tick.shape[0])):
            checkpoint_energy.append(energy.copy())
            checkpoint_pos_y.append(pos_y[ids].astype(np.int32, copy=True))
            checkpoint_pos_x.append(pos_x[ids].astype(np.int32, copy=True))
            checkpoint_died.append(died.copy())

    return {
        "new_energy": np.ascontiguousarray(energy),
        "new_health": np.ascontiguousarray(health),
        "new_injury": np.ascontiguousarray(injury),
        "new_scar": np.ascontiguousarray(scar),
        "died_mask": np.ascontiguousarray(died),
        "new_pos_y": np.ascontiguousarray(pos_y[ids].astype(np.int32, copy=False)),
        "new_pos_x": np.ascontiguousarray(pos_x[ids].astype(np.int32, copy=False)),
        "grid": np.ascontiguousarray(grid),
        "checkpoint_energy": np.ascontiguousarray(np.stack(checkpoint_energy)) if checkpoint_energy else np.empty((0, n), dtype=np.float32),
        "checkpoint_pos_y": np.ascontiguousarray(np.stack(checkpoint_pos_y)) if checkpoint_pos_y else np.empty((0, n), dtype=np.int32),
        "checkpoint_pos_x": np.ascontiguousarray(np.stack(checkpoint_pos_x)) if checkpoint_pos_x else np.empty((0, n), dtype=np.int32),
        "checkpoint_died": np.ascontiguousarray(np.stack(checkpoint_died)) if checkpoint_died else np.empty((0, n), dtype=bool),
    }


def compare_backend_states(
    expected: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    *,
    checkpoint_index: int | None = None,
    atol: float = 0.0,
) -> dict[str, Any]:
    fields = ("new_energy", "new_health", "new_injury", "new_scar", "died_mask", "new_pos_y", "new_pos_x", "grid")
    shadow_like_expected = {
        "new_energy": expected["new_energy"],
        "new_health": expected["new_health"],
        "new_injury": expected["new_injury"],
        "new_scar": expected["new_scar"],
        "food_gained": np.zeros_like(expected["new_energy"], dtype=np.float32),
        "died_mask": expected["died_mask"],
        "new_pos_y": expected["new_pos_y"],
        "new_pos_x": expected["new_pos_x"],
        "moved_mask": np.zeros_like(expected["died_mask"], dtype=bool),
        "energy_delta": np.zeros_like(expected["new_energy"], dtype=np.float32),
        "health_delta": np.zeros_like(expected["new_health"], dtype=np.float32),
        "injury_delta": np.zeros_like(expected["new_injury"], dtype=np.float32),
        "scar_delta": np.zeros_like(expected["new_scar"], dtype=np.float32),
    }
    shadow_like_actual = {key: np.copy(value) for key, value in shadow_like_expected.items()}
    for key in ("new_energy", "new_health", "new_injury", "new_scar", "died_mask", "new_pos_y", "new_pos_x"):
        shadow_like_actual[key] = actual[key]
    report = compare_shadow_deltas(shadow_like_expected, shadow_like_actual, atol=atol)
    grid_expected = np.asarray(expected["grid"])
    grid_actual = np.asarray(actual["grid"])
    grid_bad = np.argwhere(grid_expected != grid_actual)
    if grid_bad.size:
        y, x = [int(v) for v in grid_bad[0].tolist()]
        report["ok"] = False
        report["mismatch_count"] = int(report["mismatch_count"]) + int(grid_bad.shape[0])
        if report["first_divergence"] is None:
            report["first_divergence"] = {
                "field": "grid",
                "index": [y, x],
                "reason": "grid_mismatch",
                "python": int(grid_expected[y, x]),
                "candidate": int(grid_actual[y, x]),
                "abs_diff": None,
                "classification": "export/materialization",
            }
    if report.get("first_divergence") is not None:
        first = dict(report["first_divergence"])
        field = str(first.get("field", ""))
        if field in {"new_pos_y", "new_pos_x"}:
            first["classification"] = "movement"
        elif field == "grid":
            first["classification"] = "gather"
        elif field == "died_mask":
            first["classification"] = "death-mask"
        else:
            first["classification"] = "state-sync" if checkpoint_index is None else "checkpoint"
        first["checkpoint_index"] = checkpoint_index
        report["first_divergence"] = first
    report["schema_version"] = SCHEMA_VERSION
    report["fields_compared"] = list(fields)
    report["checkpoint_index"] = checkpoint_index
    return report
