from __future__ import annotations

"""CXXCORE-1 env-apply shadow helpers.

The authoritative simulation path remains Python. These helpers operate on
copied arrays and return proposed deltas/summaries for a deliberately narrow
no-RNG env-apply subset. Wider feature modes must be skipped until their state
and RNG contracts are explicit.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "cxxcore_env_apply_shadow_v1"
SUPPORTED_BACKENDS = ("python", "native_optional")
SUMMARY_SCHEMA_VERSION = "cxxcore_env_apply_shadow_summary_v1"


@dataclass(frozen=True)
class EnvApplyShadowConfig:
    height: int
    width: int
    move_cost: float
    gather_cost: float
    rest_cost: float
    attack_cost: float
    action_cost_mult: float
    rest_cost_mult: float
    food_energy: float
    max_energy: float


def _as_i32_1d(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D int32 array")
    return np.ascontiguousarray(arr)


def _as_f32_1d(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D float32 array")
    return np.ascontiguousarray(arr)


def _as_grid_i32(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D int32 array")
    return np.ascontiguousarray(arr)


def _as_grid_i8(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int8)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D int8 array")
    return np.ascontiguousarray(arr)


def feature_skip_reasons(env_cfg: Any, sim_cfg: Any = None) -> dict[str, bool]:
    """Return reasons that the first CXXCORE-1 shadow slice should skip.

    This first slice intentionally avoids RNG and broad behavior surfaces. A
    true value means the current tick shape is outside the no-RNG delta kernel.
    """
    return {
        "biome_rng": bool(getattr(env_cfg, "enable_biomes", False)),
        "patch_world": bool(getattr(env_cfg, "patch_world_enable", False)),
        "tier3_migration_pressure": bool(getattr(env_cfg, "migration_pressure_enable", False)),
        "combat": bool(getattr(env_cfg, "enable_combat", False)),
        "injury_cost_surface": bool(getattr(env_cfg, "enable_injury", False)),
        "scar_cost_surface": bool(getattr(env_cfg, "enable_scar", False)),
        "validation_random_walk": bool(getattr(env_cfg, "tier3_validation_force_random_walk", False)),
        "gru_upkeep": bool(
            float(getattr(env_cfg, "gru_cost_base", 0.0) or 0.0) > 0.0
            or float(getattr(env_cfg, "gru_cost_per_dim", 0.0) or 0.0) > 0.0
        ),
        "cognitive_cost_charging": bool(
            sim_cfg is not None
            and (
                bool(getattr(sim_cfg, "enable_experimental_perception_cost_charging", False))
                or bool(getattr(sim_cfg, "enable_experimental_sensor_radius_cost_charging", False))
                or bool(getattr(sim_cfg, "enable_experimental_prediction_cost_charging", False))
                or bool(getattr(sim_cfg, "enable_experimental_developmental_cost_charging", False))
                or bool(getattr(sim_cfg, "enable_experimental_introspection_cost_charging", False))
                or bool(getattr(sim_cfg, "enable_experimental_self_regulation_cost_charging", False))
                or bool(getattr(sim_cfg, "enable_experimental_self_modification_cost_charging", False))
            )
        ),
    }


def supported_for_first_slice(env_cfg: Any, sim_cfg: Any = None) -> tuple[bool, dict[str, bool]]:
    reasons = feature_skip_reasons(env_cfg, sim_cfg)
    return not any(reasons.values()), reasons


def config_from_env(env: Any) -> EnvApplyShadowConfig:
    cfg = getattr(env, "cfg")
    return EnvApplyShadowConfig(
        height=int(getattr(cfg, "height")),
        width=int(getattr(cfg, "width")),
        move_cost=float(getattr(cfg, "move_cost")),
        gather_cost=float(getattr(cfg, "gather_cost")),
        rest_cost=float(getattr(cfg, "rest_cost")),
        attack_cost=float(getattr(cfg, "attack_cost")),
        action_cost_mult=float(getattr(env, "action_cost_mult", 1.0)),
        rest_cost_mult=float(getattr(env, "rest_cost_mult", 1.0)),
        food_energy=float(getattr(cfg, "food_energy")),
        max_energy=float(getattr(cfg, "max_energy")),
    )


def compute_env_apply_shadow_python(
    *,
    config: EnvApplyShadowConfig,
    agent_ids: Any,
    actions: Any,
    energies: Any,
    healths: Any,
    injuries: Any,
    scars: Any,
    pos_y_by_id: Any,
    pos_x_by_id: Any,
    agent_grid: Any,
    grid: Any,
    move_cost_mults: Any | None = None,
    juvenile_harvest_mults: Any | None = None,
) -> dict[str, np.ndarray]:
    """Compute no-RNG env-apply deltas over copied arrays.

    Supported subset:
    - movement into empty cells;
    - legacy grid gather;
    - action costs and neutral per-agent multipliers;
    - max-energy clamp without injury/scar capacity penalties.
    """
    ids = _as_i32_1d(agent_ids, "agent_ids")
    acts = _as_i32_1d(actions, "actions")
    energy = _as_f32_1d(energies, "energies")
    health = _as_f32_1d(healths, "healths")
    injury = _as_f32_1d(injuries, "injuries")
    scar = _as_f32_1d(scars, "scars")
    pos_y = _as_i32_1d(pos_y_by_id, "pos_y_by_id")
    pos_x = _as_i32_1d(pos_x_by_id, "pos_x_by_id")
    occ = _as_grid_i32(agent_grid, "agent_grid").copy()
    food_grid = _as_grid_i8(grid, "grid").copy()

    n = int(ids.shape[0])
    for name, arr in (
        ("actions", acts),
        ("energies", energy),
        ("healths", health),
        ("injuries", injury),
        ("scars", scar),
    ):
        if int(arr.shape[0]) != n:
            raise ValueError(f"{name} length must match agent_ids")
    if n == 0:
        empty_f32 = np.empty((0,), dtype=np.float32)
        empty_bool = np.empty((0,), dtype=bool)
        empty_i32 = np.empty((0,), dtype=np.int32)
        return {
            "new_energy": empty_f32,
            "new_health": empty_f32,
            "new_injury": empty_f32,
            "new_scar": empty_f32,
            "food_gained": empty_f32,
            "died_mask": empty_bool,
            "new_pos_y": empty_i32,
            "new_pos_x": empty_i32,
            "moved_mask": empty_bool,
            "energy_delta": empty_f32,
            "health_delta": empty_f32,
            "injury_delta": empty_f32,
            "scar_delta": empty_f32,
        }

    if move_cost_mults is None:
        move_mult = np.ones((n,), dtype=np.float32)
    else:
        move_mult = _as_f32_1d(move_cost_mults, "move_cost_mults")
    if juvenile_harvest_mults is None:
        harvest_mult = np.ones((n,), dtype=np.float32)
    else:
        harvest_mult = _as_f32_1d(juvenile_harvest_mults, "juvenile_harvest_mults")
    if int(move_mult.shape[0]) != n or int(harvest_mult.shape[0]) != n:
        raise ValueError("multiplier lengths must match agent_ids")

    order = np.argsort(ids, kind="mergesort")
    inv = np.empty_like(order)
    inv[order] = np.arange(n, dtype=order.dtype)

    ids_o = ids[order]
    acts_o = acts[order]
    energy_o = energy[order]
    health_o = health[order]
    injury_o = injury[order]
    scar_o = scar[order]
    move_mult_o = move_mult[order]
    harvest_mult_o = harvest_mult[order]

    ys = pos_y[ids_o].astype(np.int32, copy=True)
    xs = pos_x[ids_o].astype(np.int32, copy=True)
    old_ys = ys.copy()
    old_xs = xs.copy()
    moved = np.zeros((n,), dtype=bool)

    costs = np.empty((n,), dtype=np.float32)
    for i in range(n):
        action = int(acts_o[i])
        cost = float(config.move_cost) * float(move_mult_o[i])
        if action == 4:
            cost = float(config.gather_cost)
        elif action == 5:
            cost = float(config.rest_cost)
        elif action == 6:
            cost = float(config.attack_cost)
        if float(config.action_cost_mult) != 1.0:
            cost *= float(config.action_cost_mult)
        if float(config.rest_cost_mult) != 1.0 and action == 5:
            cost *= float(config.rest_cost_mult)
        costs[i] = np.float32(cost)

    for i in range(n):
        action = int(acts_o[i])
        if action < 0 or action > 3:
            continue
        aid = int(ids_o[i])
        y = int(ys[i])
        x = int(xs[i])
        ny = y
        nx = x
        if action == 0:
            ny -= 1
        elif action == 1:
            ny += 1
        elif action == 2:
            nx += 1
        else:
            nx -= 1
        if ny < 0 or ny >= int(config.height) or nx < 0 or nx >= int(config.width):
            continue
        if int(occ[ny, nx]) != -1:
            continue
        if int(occ[y, x]) == aid:
            occ[y, x] = -1
        occ[ny, nx] = aid
        ys[i] = np.int32(ny)
        xs[i] = np.int32(nx)
        moved[i] = True

    food_gained = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        if int(acts_o[i]) != 4:
            continue
        y = int(ys[i])
        x = int(xs[i])
        if int(food_grid[y, x]) == 1:
            food_grid[y, x] = np.int8(0)
            food_gained[i] = np.float32(float(config.food_energy) * float(harvest_mult_o[i]))

    new_energy = energy_o - costs + food_gained
    new_energy = np.minimum(new_energy, np.float32(float(config.max_energy)))
    new_energy = np.maximum(new_energy, np.float32(0.0)).astype(np.float32, copy=False)
    new_health = health_o.copy()
    new_injury = injury_o.copy()
    new_scar = scar_o.copy()
    died = ((new_energy <= 0.0) | (new_health <= 0.0)).astype(bool, copy=False)

    new_pos_y = old_ys.copy()
    new_pos_x = old_xs.copy()
    new_pos_y[moved] = ys[moved]
    new_pos_x[moved] = xs[moved]

    def _unperm(arr: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(arr[inv])

    out_energy = _unperm(new_energy)
    out_health = _unperm(new_health)
    out_injury = _unperm(new_injury)
    out_scar = _unperm(new_scar)
    return {
        "new_energy": out_energy,
        "new_health": out_health,
        "new_injury": out_injury,
        "new_scar": out_scar,
        "food_gained": _unperm(food_gained),
        "died_mask": _unperm(died),
        "new_pos_y": _unperm(new_pos_y),
        "new_pos_x": _unperm(new_pos_x),
        "moved_mask": _unperm(moved),
        "energy_delta": np.ascontiguousarray(out_energy - energy),
        "health_delta": np.ascontiguousarray(out_health - health),
        "injury_delta": np.ascontiguousarray(out_injury - injury),
        "scar_delta": np.ascontiguousarray(out_scar - scar),
    }


def observed_delta_from_python(
    *,
    agent_ids: Any,
    pre_energy: Any,
    pre_health: Any,
    pre_injury: Any,
    pre_scar: Any,
    post_energy: Any,
    post_health: Any,
    post_injury: Any,
    post_scar: Any,
    food_gained: Any,
    died_mask: Any,
    pre_pos_y: Any,
    pre_pos_x: Any,
    post_pos_y_by_id: Any,
    post_pos_x_by_id: Any,
) -> dict[str, np.ndarray]:
    ids = _as_i32_1d(agent_ids, "agent_ids")
    pre_e = _as_f32_1d(pre_energy, "pre_energy")
    pre_h = _as_f32_1d(pre_health, "pre_health")
    pre_i = _as_f32_1d(pre_injury, "pre_injury")
    pre_s = _as_f32_1d(pre_scar, "pre_scar")
    post_e = _as_f32_1d(post_energy, "post_energy")
    post_h = _as_f32_1d(post_health, "post_health")
    post_i = _as_f32_1d(post_injury, "post_injury")
    post_s = _as_f32_1d(post_scar, "post_scar")
    fg = _as_f32_1d(food_gained, "food_gained")
    died = np.asarray(died_mask, dtype=bool)
    pre_y = _as_i32_1d(pre_pos_y, "pre_pos_y")
    pre_x = _as_i32_1d(pre_pos_x, "pre_pos_x")
    post_y_by_id = _as_i32_1d(post_pos_y_by_id, "post_pos_y_by_id")
    post_x_by_id = _as_i32_1d(post_pos_x_by_id, "post_pos_x_by_id")
    post_y = np.ascontiguousarray(post_y_by_id[ids].astype(np.int32, copy=False))
    post_x = np.ascontiguousarray(post_x_by_id[ids].astype(np.int32, copy=False))
    return {
        "new_energy": post_e,
        "new_health": post_h,
        "new_injury": post_i,
        "new_scar": post_s,
        "food_gained": fg,
        "died_mask": np.ascontiguousarray(died),
        "new_pos_y": post_y,
        "new_pos_x": post_x,
        "moved_mask": np.ascontiguousarray((post_y != pre_y) | (post_x != pre_x)),
        "energy_delta": np.ascontiguousarray(post_e - pre_e),
        "health_delta": np.ascontiguousarray(post_h - pre_h),
        "injury_delta": np.ascontiguousarray(post_i - pre_i),
        "scar_delta": np.ascontiguousarray(post_s - pre_s),
    }


def compare_shadow_deltas(
    expected: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    *,
    atol: float = 0.0,
) -> dict[str, Any]:
    fields = (
        "new_energy",
        "new_health",
        "new_injury",
        "new_scar",
        "food_gained",
        "died_mask",
        "new_pos_y",
        "new_pos_x",
        "moved_mask",
        "energy_delta",
        "health_delta",
        "injury_delta",
        "scar_delta",
    )
    max_abs_diff = 0.0
    first = None
    mismatch_count = 0
    for field in fields:
        exp = np.asarray(expected[field])
        got = np.asarray(actual[field])
        if exp.shape != got.shape:
            mismatch_count += 1
            first = first or {
                "field": field,
                "index": -1,
                "reason": "shape_mismatch",
                "python": list(exp.shape),
                "candidate": list(got.shape),
                "abs_diff": None,
            }
            continue
        if exp.dtype == np.bool_ or got.dtype == np.bool_ or np.issubdtype(exp.dtype, np.integer):
            bad = np.nonzero(exp != got)[0]
            if bad.size:
                mismatch_count += int(bad.size)
                idx = int(bad[0])
                first = first or {
                    "field": field,
                    "index": idx,
                    "reason": "exact_mismatch",
                    "python": bool(exp[idx]) if exp.dtype == np.bool_ else int(exp[idx]),
                    "candidate": bool(got[idx]) if got.dtype == np.bool_ else int(got[idx]),
                    "abs_diff": None,
                }
            continue
        diff = np.abs(exp.astype(np.float64, copy=False) - got.astype(np.float64, copy=False))
        if diff.size:
            max_abs_diff = max(max_abs_diff, float(np.max(diff)))
        bad = np.nonzero(diff > float(atol))[0]
        if bad.size:
            mismatch_count += int(bad.size)
            idx = int(bad[0])
            first = first or {
                "field": field,
                "index": idx,
                "reason": "float_mismatch",
                "python": float(exp[idx]),
                "candidate": float(got[idx]),
                "abs_diff": float(diff[idx]),
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": bool(mismatch_count == 0),
        "field_count": int(len(fields)),
        "mismatch_count": int(mismatch_count),
        "max_abs_diff": float(max_abs_diff),
        "first_divergence": first,
    }


def init_shadow_summary() -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "records": 0,
        "ticks": 0,
        "agents": 0,
        "cells": 0,
        "skipped": 0,
        "ok": True,
        "first_divergence": None,
        "skip_reason_counts": {},
        "timing_s": {
            "wall": 0.0,
            "python_env_apply": 0.0,
            "snapshot_copy": 0.0,
            "shadow_compute": 0.0,
            "materialize": 0.0,
            "compare": 0.0,
            "diagnostics": 0.0,
        },
    }


def update_shadow_summary(summary: dict[str, Any] | None, record: dict[str, Any]) -> dict[str, Any]:
    if summary is None:
        summary = init_shadow_summary()
    summary["records"] = int(summary.get("records", 0)) + 1
    summary["ticks"] = int(summary.get("ticks", 0)) + 1
    summary["agents"] = int(summary.get("agents", 0)) + int(record.get("agents", 0) or 0)
    summary["cells"] = int(record.get("cells", summary.get("cells", 0)) or 0)
    if record.get("skipped"):
        summary["skipped"] = int(summary.get("skipped", 0)) + 1
        reasons = dict(record.get("skip_reasons", {}) or {})
        counts = dict(summary.get("skip_reason_counts", {}) or {})
        for key, value in reasons.items():
            if value:
                counts[str(key)] = int(counts.get(str(key), 0)) + 1
        summary["skip_reason_counts"] = counts
    parity = record.get("parity")
    if isinstance(parity, dict):
        if not bool(parity.get("ok", False)):
            summary["ok"] = False
            if summary.get("first_divergence") is None:
                summary["first_divergence"] = parity.get("first_divergence")
    elif parity == "not_checked":
        summary["ok"] = bool(summary.get("ok", True))

    totals = dict(summary.get("timing_s", {}) or {})
    for key in ("wall", "python_env_apply", "snapshot_copy", "shadow_compute", "materialize", "compare", "diagnostics"):
        totals[key] = float(totals.get(key, 0.0)) + float(record.get("timing_s", {}).get(key, 0.0) or 0.0)
    summary["timing_s"] = totals
    return summary


def write_shadow_summary(run_dir: Any, summary: dict[str, Any], *, filename: str) -> Path | None:
    if not run_dir:
        return None
    out_dir = Path(str(run_dir)) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / str(filename or "cxxcore_env_apply_shadow_summary.json")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(out_path)
    return out_path
