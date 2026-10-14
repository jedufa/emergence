from __future__ import annotations

"""CXXCORE-15 resident SoA tick-island helpers.

Diagnostic/default-off only. The resident model keeps copied SoA state across a
multi-tick window and exports sparse summaries plus final state, avoiding the
CXXCORE-14 pattern where missing transaction fields are reconstructed in Python
after native compute.
"""

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence

import numpy as np

from perf.cxxcore5_tick_island_layout_cache import (
    TickIslandLayoutCacheInputs,
    compare_runner_backend_states,
    run_tick_island_python_reference,
)
from perf.cxxcore8_guarded_runner_bridge import state_digest


SCHEMA_VERSION = "cxxcore15_resident_soa_tick_island_v1"
AUDIT_SCHEMA_VERSION = "cxxcore15_resident_ownership_audit_v1"
DECISION_LABELS = (
    "CXXCORE15-A-RESIDENT-SOA-TICK-ISLAND-GO",
    "CXXCORE15-B-RESIDENT-PARITY-ONLY-NO-SPEED",
    "CXXCORE15-C-LIVE-RUNNER-STATE-OWNERSHIP-TOO-WIDE",
    "CXXCORE15-D-RNG-SURFACE-PREFLIGHT-REQUIRED",
    "CXXCORE15-E-LARGER-TICK-LOOP-ISLAND-REQUIRED",
    "CXXCORE15-F-STOP-CXXCORE-ENV-TICK-PATH",
    "CXXCORE15-INCONCLUSIVE",
)
SUMMARY_NAMES = ("energy_sum", "energy_min", "energy_max", "died_count", "pos_checksum", "grid_sum", "cost_total")


@dataclass(frozen=True)
class ResidentSoAConfig:
    sparse_interval: int = 0
    export_final: bool = True
    unsupported_feature_reasons: tuple[str, ...] = ()
    parity_policy: str = "sparse_and_final"


@dataclass(frozen=True)
class ResidentSoAState:
    schema_version: str
    tick_inputs: TickIslandLayoutCacheInputs
    agent_ids: np.ndarray
    energy: np.ndarray
    health: np.ndarray
    injury: np.ndarray
    scar: np.ndarray
    pos_y_by_id: np.ndarray
    pos_x_by_id: np.ndarray
    agent_grid: np.ndarray
    resource_grid: np.ndarray
    actions_by_tick: np.ndarray
    cost_scalars_by_tick: np.ndarray
    move_cost_mults: np.ndarray
    juvenile_harvest_mults: np.ndarray
    layout_digest: str


@dataclass(frozen=True)
class ResidentRunResult:
    schema_version: str
    output: dict[str, np.ndarray]
    agent_grid: np.ndarray
    food_gained: np.ndarray
    moved_mask: np.ndarray
    views_dirty: bool
    sparse_ticks: np.ndarray
    sparse_summaries: np.ndarray
    final_summary: dict[str, float]
    diagnostics: dict[str, Any]


def resident_ownership_audit_payload() -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "materialization_cost_paid_in_cxxcore14": [
            "native output converted into Python arrays",
            "agent_grid reconstructed after native output",
            "food_gained and moved_mask reconstructed in Python",
            "final parity comparison re-ran copied-array Python reference",
            "full state copied into Python live-state carrier",
        ],
        "avoidable_when_resident": [
            "per-window Python reconstruction of agent_grid/food_gained/moved_mask",
            "per-tick full materialization",
            "full final export when sparse summaries are sufficient for lower-bound probes",
        ],
        "resident_arrays": [
            "energy",
            "health",
            "injury",
            "scar",
            "pos_y_by_id",
            "pos_x_by_id",
            "agent_grid",
            "resource_grid",
            "cost_total_by_agent",
            "food_gained",
            "moved_mask",
        ],
        "export_every_tick": [],
        "export_sparse_or_final": [
            "energy_sum",
            "died_count",
            "position checksum",
            "resource grid checksum",
            "cost total",
            "final full arrays for parity",
        ],
        "diagnostics_forcing_python_materialization": [
            "legacy metrics/checkpoint schemas if live runner adopted",
            "Python Agent.stats updates outside bounded harness",
            "Patch/Jungle and Tier3 side-effect counters when enabled",
        ],
        "unsupported_feature_fallbacks": unsupported_feature_fallbacks(),
        "minimal_safe_window": "captured no-RNG action/cost window with Python policy/cost inputs fixed before backend run",
        "checkpoint_schema_changed": False,
        "metrics_schema_changed": False,
        "native_rng_added": False,
    }


def unsupported_feature_fallbacks() -> list[str]:
    return [
        "Patch/Jungle dynamic arrays and counters",
        "Tier3 stochastic transition state",
        "biome hazard RNG",
        "validation RNG writeback unless actions are Python-precomputed",
        "reproduction/selection/cull/refill/lineage lifecycle",
        "checkpoint serialization ownership",
        "metrics schema ownership",
        "live policy/cognition inference ownership",
    ]


def as_resident_state(inputs: TickIslandLayoutCacheInputs) -> ResidentSoAState:
    state = inputs.backend.runner.state
    return ResidentSoAState(
        schema_version=SCHEMA_VERSION,
        tick_inputs=inputs,
        agent_ids=np.ascontiguousarray(state.agent_ids, dtype=np.int32),
        energy=np.ascontiguousarray(state.initial_energy, dtype=np.float32),
        health=np.ascontiguousarray(state.initial_health, dtype=np.float32),
        injury=np.ascontiguousarray(state.initial_injury, dtype=np.float32),
        scar=np.ascontiguousarray(state.initial_scar, dtype=np.float32),
        pos_y_by_id=np.ascontiguousarray(state.pos_y_by_id, dtype=np.int32),
        pos_x_by_id=np.ascontiguousarray(state.pos_x_by_id, dtype=np.int32),
        agent_grid=np.ascontiguousarray(state.agent_grid, dtype=np.int32),
        resource_grid=np.ascontiguousarray(state.grid, dtype=np.int8),
        actions_by_tick=np.ascontiguousarray(state.actions_by_tick, dtype=np.int32),
        cost_scalars_by_tick=np.ascontiguousarray(inputs.backend.runner.cost_scalars_by_tick, dtype=np.float32),
        move_cost_mults=np.ascontiguousarray(state.move_cost_mults, dtype=np.float32),
        juvenile_harvest_mults=np.ascontiguousarray(state.juvenile_harvest_mults, dtype=np.float32),
        layout_digest=str(inputs.backend.layout.digest),
    )


def reset_resident_state(state: ResidentSoAState) -> ResidentSoAState:
    return as_resident_state(state.tick_inputs)


def resident_state_digest(state: ResidentSoAState) -> str:
    return _stable_json_digest(
        {
            "agent_ids": _array_digest(state.agent_ids),
            "energy": _array_digest(state.energy),
            "health": _array_digest(state.health),
            "injury": _array_digest(state.injury),
            "scar": _array_digest(state.scar),
            "pos_y_by_id": _array_digest(state.pos_y_by_id),
            "pos_x_by_id": _array_digest(state.pos_x_by_id),
            "agent_grid": _array_digest(state.agent_grid),
            "resource_grid": _array_digest(state.resource_grid),
            "actions_by_tick": _array_digest(state.actions_by_tick),
            "cost_scalars_by_tick": _array_digest(state.cost_scalars_by_tick),
            "layout_digest": state.layout_digest,
        }
    )


def run_resident_python_reference(
    state: ResidentSoAState,
    *,
    config: ResidentSoAConfig | None = None,
    inject_divergence: bool = False,
) -> ResidentRunResult:
    cfg = config or ResidentSoAConfig()
    if cfg.unsupported_feature_reasons:
        raise ValueError("unsupported resident features: " + ",".join(cfg.unsupported_feature_reasons))
    tick_state = state.tick_inputs.backend.runner.state
    env_cfg = tick_state.config
    ids = state.agent_ids
    n = int(ids.shape[0])
    order = np.argsort(ids, kind="mergesort")
    energy = state.energy.astype(np.float32, copy=True)
    health = state.health.astype(np.float32, copy=True)
    injury = state.injury.astype(np.float32, copy=True)
    scar = state.scar.astype(np.float32, copy=True)
    pos_y = state.pos_y_by_id.astype(np.int32, copy=True)
    pos_x = state.pos_x_by_id.astype(np.int32, copy=True)
    occ = state.agent_grid.astype(np.int32, copy=True)
    grid = state.resource_grid.astype(np.int8, copy=True)
    food_gained_total = np.zeros((n,), dtype=np.float32)
    moved_mask = np.zeros((n,), dtype=bool)
    cost_total = np.zeros((n,), dtype=np.float32)
    died = np.zeros((n,), dtype=bool)
    sparse_ticks: list[int] = []
    sparse_summaries: list[np.ndarray] = []
    interval = int(cfg.sparse_interval or 0)

    for tick in range(int(state.actions_by_tick.shape[0])):
        actions = state.actions_by_tick[tick]
        costs = np.empty((n,), dtype=np.float32)
        for src in order.tolist():
            action = int(actions[src])
            cost = float(env_cfg.move_cost) * float(state.move_cost_mults[src])
            if action == 4:
                cost = float(env_cfg.gather_cost)
            elif action == 5:
                cost = float(env_cfg.rest_cost)
            elif action == 6:
                cost = float(env_cfg.attack_cost)
            if float(env_cfg.action_cost_mult) != 1.0:
                cost *= float(env_cfg.action_cost_mult)
            if float(env_cfg.rest_cost_mult) != 1.0 and action == 5:
                cost *= float(env_cfg.rest_cost_mult)
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
                if 0 <= ny < int(env_cfg.height) and 0 <= nx < int(env_cfg.width) and int(occ[ny, nx]) == -1:
                    if int(occ[y, x]) == aid:
                        occ[y, x] = -1
                    occ[ny, nx] = aid
                    pos_y[aid] = np.int32(ny)
                    pos_x[aid] = np.int32(nx)
                    moved_mask[src] = True
            elif action == 4 and int(grid[y, x]) == 1:
                grid[y, x] = np.int8(0)
                food_gained[src] = np.float32(float(env_cfg.food_energy) * float(state.juvenile_harvest_mults[src]))
        tick_cost = state.cost_scalars_by_tick[tick].astype(np.float32, copy=False)
        cost_total = np.ascontiguousarray(cost_total + tick_cost)
        food_gained_total = np.ascontiguousarray(food_gained_total + food_gained)
        energy = energy - costs + food_gained - tick_cost
        energy = np.minimum(energy, np.float32(float(env_cfg.max_energy)))
        energy = np.maximum(energy, np.float32(0.0)).astype(np.float32, copy=False)
        died = ((energy <= 0.0) | (health <= 0.0)).astype(bool, copy=False)
        if interval > 0 and ((tick + 1) % interval == 0 or (tick + 1) == int(state.actions_by_tick.shape[0])):
            sparse_ticks.append(tick + 1)
            sparse_summaries.append(_summary_array(ids=ids, energy=energy, died=died, pos_y=pos_y, pos_x=pos_x, grid=grid, cost_total=cost_total))

    if inject_divergence and energy.size:
        energy[0] = np.float32(float(energy[0]) + 1.0)
    output = {
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
    final_summary = _summary_dict(_summary_array(ids=ids, energy=energy, died=died, pos_y=pos_y, pos_x=pos_x, grid=grid, cost_total=cost_total))
    return ResidentRunResult(
        schema_version=SCHEMA_VERSION,
        output=output,
        agent_grid=np.ascontiguousarray(occ),
        food_gained=np.ascontiguousarray(food_gained_total),
        moved_mask=np.ascontiguousarray(moved_mask),
        views_dirty=bool(np.any(moved_mask) or np.any(food_gained_total > 0.0)),
        sparse_ticks=np.ascontiguousarray(sparse_ticks, dtype=np.int32),
        sparse_summaries=np.ascontiguousarray(np.vstack(sparse_summaries).astype(np.float64)) if sparse_summaries else np.empty((0, len(SUMMARY_NAMES)), dtype=np.float64),
        final_summary=final_summary,
        diagnostics={"schema_version": SCHEMA_VERSION, "ticks": int(state.actions_by_tick.shape[0]), "agents": n},
    )


def compare_resident_to_reference(
    inputs: TickIslandLayoutCacheInputs,
    resident: ResidentRunResult,
    *,
    atol: float = 1.0e-5,
) -> dict[str, Any]:
    expected = run_tick_island_python_reference(inputs)
    report = compare_runner_backend_states(expected, resident.output, atol=atol)
    report["state_digest"] = state_digest(resident.output)
    report["sparse_summary_rows"] = int(resident.sparse_summaries.shape[0])
    return report


def resident_result_payload(result: ResidentRunResult) -> dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "state_digest": state_digest(result.output),
        "agent_grid_digest": _array_digest(result.agent_grid),
        "resource_grid_digest": _array_digest(result.output["grid"]),
        "food_gained_sum": float(np.sum(result.food_gained, dtype=np.float64)),
        "moved_count": int(np.sum(result.moved_mask)),
        "views_dirty": bool(result.views_dirty),
        "sparse_ticks": result.sparse_ticks.tolist(),
        "final_summary": dict(result.final_summary),
        "diagnostics": dict(result.diagnostics),
    }


def _summary_array(
    *,
    ids: np.ndarray,
    energy: np.ndarray,
    died: np.ndarray,
    pos_y: np.ndarray,
    pos_x: np.ndarray,
    grid: np.ndarray,
    cost_total: np.ndarray,
) -> np.ndarray:
    y = pos_y[ids].astype(np.float64, copy=False)
    x = pos_x[ids].astype(np.float64, copy=False)
    e = energy.astype(np.float64, copy=False)
    return np.asarray(
        [
            float(np.sum(e, dtype=np.float64)),
            float(np.min(e)) if e.size else 0.0,
            float(np.max(e)) if e.size else 0.0,
            float(np.sum(died.astype(np.float64), dtype=np.float64)),
            float(np.sum(y * 1009.0 + x * 917.0, dtype=np.float64)),
            float(np.sum(grid.astype(np.float64), dtype=np.float64)),
            float(np.sum(cost_total.astype(np.float64), dtype=np.float64)),
        ],
        dtype=np.float64,
    )


def _summary_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(values[i]) for i, name in enumerate(SUMMARY_NAMES)}


def _array_digest(value: Any) -> str:
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _stable_json_digest(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "DECISION_LABELS",
    "SCHEMA_VERSION",
    "SUMMARY_NAMES",
    "ResidentRunResult",
    "ResidentSoAConfig",
    "ResidentSoAState",
    "as_resident_state",
    "compare_resident_to_reference",
    "reset_resident_state",
    "resident_ownership_audit_payload",
    "resident_result_payload",
    "resident_state_digest",
    "run_resident_python_reference",
    "unsupported_feature_fallbacks",
]
