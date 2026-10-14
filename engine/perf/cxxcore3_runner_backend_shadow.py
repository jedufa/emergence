from __future__ import annotations

"""CXXCORE-3 runner-adjacent persistent backend shadow helpers.

Diagnostic-only. Python remains authoritative; C++ may receive copied arrays
and mutate only backend-owned state.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from perf.cxxcore2_state_backend import StateBackendInputs, as_backend_inputs, compare_backend_states
from perf.cxxcore_env_apply_shadow import EnvApplyShadowConfig


SCHEMA_VERSION = "cxxcore3_runner_backend_shadow_v1"


@dataclass(frozen=True)
class RunnerBackendShadowInputs:
    state: StateBackendInputs
    cost_scalars_by_tick: np.ndarray


def as_runner_backend_shadow_inputs(
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
    cost_scalars_by_tick: Any | None = None,
    move_cost_mults: Any | None = None,
    juvenile_harvest_mults: Any | None = None,
) -> RunnerBackendShadowInputs:
    state = as_backend_inputs(
        config=config,
        agent_ids=agent_ids,
        initial_energy=initial_energy,
        initial_health=initial_health,
        initial_injury=initial_injury,
        initial_scar=initial_scar,
        pos_y_by_id=pos_y_by_id,
        pos_x_by_id=pos_x_by_id,
        agent_grid=agent_grid,
        grid=grid,
        actions_by_tick=actions_by_tick,
        move_cost_mults=move_cost_mults,
        juvenile_harvest_mults=juvenile_harvest_mults,
    )
    if cost_scalars_by_tick is None:
        costs = np.zeros(state.actions_by_tick.shape, dtype=np.float32)
    else:
        costs = np.ascontiguousarray(cost_scalars_by_tick, dtype=np.float32)
    if costs.shape != state.actions_by_tick.shape:
        raise ValueError("cost_scalars_by_tick shape must match actions_by_tick")
    return RunnerBackendShadowInputs(state=state, cost_scalars_by_tick=costs)


def deterministic_cost_scalar_matrix(agent_ids: Any, actions_by_tick: Any, *, enabled: bool) -> np.ndarray:
    """Small deterministic stand-in for already-expressed CostChargeSurface scalars.

    This intentionally models the fallback contract: Python precomputes scalar
    cost arrays and the backend applies them. It does not prove native
    CostChargeSurface expression.
    """
    actions = np.ascontiguousarray(actions_by_tick, dtype=np.int32)
    ids = np.ascontiguousarray(agent_ids, dtype=np.int32)
    if not enabled:
        return np.zeros(actions.shape, dtype=np.float32)
    base = 0.002 + (ids.astype(np.float32) % np.float32(7.0)) * np.float32(0.0001)
    action_term = np.maximum(actions.astype(np.float32), 0.0) * np.float32(0.00005)
    return np.ascontiguousarray(base.reshape((1, -1)) + action_term, dtype=np.float32)


def run_runner_backend_python_reference(inputs: RunnerBackendShadowInputs) -> dict[str, np.ndarray]:
    state = inputs.state
    ids = state.agent_ids
    n = int(ids.shape[0])
    order = np.argsort(ids, kind="mergesort")
    energy = state.initial_energy.astype(np.float32, copy=True)
    health = state.initial_health.astype(np.float32, copy=True)
    injury = state.initial_injury.astype(np.float32, copy=True)
    scar = state.initial_scar.astype(np.float32, copy=True)
    pos_y = state.pos_y_by_id.astype(np.int32, copy=True)
    pos_x = state.pos_x_by_id.astype(np.int32, copy=True)
    occ = state.agent_grid.astype(np.int32, copy=True)
    grid = state.grid.astype(np.int8, copy=True)
    died = np.zeros((n,), dtype=bool)
    cost_total = np.zeros((n,), dtype=np.float32)
    cfg = state.config
    for tick in range(int(state.actions_by_tick.shape[0])):
        actions = state.actions_by_tick[tick]
        costs = np.empty((n,), dtype=np.float32)
        for src in order.tolist():
            action = int(actions[src])
            cost = float(cfg.move_cost) * float(state.move_cost_mults[src])
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
                    food_gained[src] = np.float32(float(cfg.food_energy) * float(state.juvenile_harvest_mults[src]))
        tick_cost = inputs.cost_scalars_by_tick[tick].astype(np.float32, copy=False)
        cost_total = np.ascontiguousarray(cost_total + tick_cost)
        energy = energy - costs + food_gained - tick_cost
        energy = np.minimum(energy, np.float32(float(cfg.max_energy)))
        energy = np.maximum(energy, np.float32(0.0)).astype(np.float32, copy=False)
        died = ((energy <= 0.0) | (health <= 0.0)).astype(bool, copy=False)
    return {
        "new_energy": np.ascontiguousarray(energy),
        "new_health": np.ascontiguousarray(health),
        "new_injury": np.ascontiguousarray(injury),
        "new_scar": np.ascontiguousarray(scar),
        "died_mask": np.ascontiguousarray(died),
        "new_pos_y": np.ascontiguousarray(pos_y[ids].astype(np.int32, copy=False)),
        "new_pos_x": np.ascontiguousarray(pos_x[ids].astype(np.int32, copy=False)),
        "grid": np.ascontiguousarray(grid),
        "cost_total_by_agent": np.ascontiguousarray(cost_total),
        "cost_total": np.asarray([float(np.sum(cost_total, dtype=np.float64))], dtype=np.float64),
    }


def state_summary(payload: dict[str, np.ndarray]) -> dict[str, float]:
    energy = np.asarray(payload["new_energy"], dtype=np.float64)
    died = np.asarray(payload["died_mask"], dtype=bool)
    y = np.asarray(payload["new_pos_y"], dtype=np.float64)
    x = np.asarray(payload["new_pos_x"], dtype=np.float64)
    grid = np.asarray(payload["grid"])
    costs = np.asarray(payload.get("cost_total_by_agent", np.zeros_like(energy)), dtype=np.float64)
    return {
        "energy_sum": float(np.sum(energy, dtype=np.float64)),
        "energy_min": float(np.min(energy)) if energy.size else 0.0,
        "energy_max": float(np.max(energy)) if energy.size else 0.0,
        "died_count": float(np.sum(died)),
        "pos_checksum": float(np.sum(y * 1009.0 + x * 917.0, dtype=np.float64)),
        "grid_sum": float(np.sum(grid, dtype=np.float64)),
        "cost_total": float(np.sum(costs, dtype=np.float64)),
    }


def compare_runner_backend_states(
    expected: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    *,
    atol: float = 0.0,
) -> dict[str, Any]:
    report = compare_backend_states(expected, actual, atol=atol)
    cost_expected = np.asarray(expected.get("cost_total_by_agent", []), dtype=np.float64)
    cost_actual = np.asarray(actual.get("cost_total_by_agent", []), dtype=np.float64)
    if cost_expected.shape != cost_actual.shape:
        report["ok"] = False
        report["mismatch_count"] = int(report["mismatch_count"]) + 1
        report["first_divergence"] = report.get("first_divergence") or {
            "field": "cost_total_by_agent",
            "index": -1,
            "reason": "shape_mismatch",
            "python": list(cost_expected.shape),
            "candidate": list(cost_actual.shape),
            "abs_diff": None,
            "classification": "cost-charge",
        }
    elif cost_expected.size:
        diff = np.abs(cost_expected - cost_actual)
        bad = np.nonzero(diff > float(atol))[0]
        if bad.size:
            idx = int(bad[0])
            report["ok"] = False
            report["mismatch_count"] = int(report["mismatch_count"]) + int(bad.size)
            report["max_abs_diff"] = max(float(report.get("max_abs_diff", 0.0)), float(np.max(diff)))
            report["first_divergence"] = report.get("first_divergence") or {
                "field": "cost_total_by_agent",
                "index": idx,
                "reason": "float_mismatch",
                "python": float(cost_expected[idx]),
                "candidate": float(cost_actual[idx]),
                "abs_diff": float(diff[idx]),
                "classification": "cost-charge",
            }
    report["schema_version"] = SCHEMA_VERSION
    report["fields_compared"] = list(report.get("fields_compared", [])) + ["cost_total_by_agent", "cost_total"]
    return report


def compare_summaries(expected: dict[str, float], actual: dict[str, float], *, atol: float = 1.0e-6) -> dict[str, Any]:
    first = None
    mismatch_count = 0
    max_abs_diff = 0.0
    for key in sorted(expected):
        diff = abs(float(expected[key]) - float(actual.get(key, 0.0)))
        max_abs_diff = max(max_abs_diff, diff)
        if diff > float(atol):
            mismatch_count += 1
            first = first or {
                "field": key,
                "reason": "summary_mismatch",
                "python": float(expected[key]),
                "candidate": float(actual.get(key, 0.0)),
                "abs_diff": float(diff),
                "classification": "summary",
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": bool(mismatch_count == 0),
        "mismatch_count": int(mismatch_count),
        "max_abs_diff": float(max_abs_diff),
        "first_divergence": first,
        "fields_compared": sorted(expected),
    }


def audit_payload() -> dict[str, Any]:
    return {
        "schema_version": "cxxcore3_runner_backend_audit_v1",
        "cxxcore2_state_fields": [
            "agent_ids",
            "energy",
            "health",
            "injury",
            "scar",
            "pos_y_by_id",
            "pos_x_by_id",
            "agent_grid",
            "grid",
            "move_cost_mults",
            "juvenile_harvest_mults",
            "actions_by_tick",
        ],
        "new_cxxcore3_fields": ["cost_scalars_by_tick", "cost_total_by_agent", "sparse_summary"],
        "real_runner_fields_not_represented": [
            "agent Python objects",
            "genome/phenotype expression state",
            "CostChargeSurface object construction",
            "GRU upkeep state",
            "biome hazard random draws",
            "Tier3 transition random draws",
            "combat/injury/scar mutation internals",
            "checkpoint and metrics serialization state",
        ],
        "no_rng_operations_included": [
            "movement",
            "blocked movement",
            "action energy costs",
            "legacy grid gather",
            "death mask",
            "precomputed cost scalar application",
        ],
        "cost_charge_boundary": {
            "implemented": "precomputed per-agent/per-tick scalar fallback",
            "native_cost_surface_expression": False,
            "reason": "production CostChargeSurface still depends on Python object walking and phenotype/surface construction",
        },
        "rng_sources": [
            "validation random-walk override via rng_manager.policy_np.integers",
            "biome hazard via env rng random draws",
            "Tier3 transition/crossing risk via env rng random draws",
            "combat/injury/scar paths if enabled",
            "probe action generation is deterministic arithmetic, not RNG",
        ],
        "checkpoint_metrics": {
            "passive_readouts": ["final export", "sparse summary", "cost totals"],
            "excluded_mutations": ["checkpoint serialization", "runtime metrics definitions"],
        },
    }
