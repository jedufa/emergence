#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "engine") not in sys.path:
    sys.path.insert(0, str(ROOT / "engine"))

from agents.cognition import compute_env_cost_charge_scalar_from_surface, express_cost_charge_surface  # noqa: E402
from agents.developmental_program_v1 import canonicalize_developmental_program  # noqa: E402
from agents.directed_inheritance import implemented_directed_inheritance_channel_mask  # noqa: E402
from agents.genome import Genome  # noqa: E402
from agents.introspection import implemented_introspection_channel_mask  # noqa: E402
from agents.perception import implemented_sensor_channel_mask  # noqa: E402
from agents.self_modification import implemented_self_modification_target_mask  # noqa: E402
from agents.self_regulation import implemented_self_regulation_control_mask  # noqa: E402
from env.env_config import EnvConfig  # noqa: E402
from env.gridworld_pkg.world import GridWorld  # noqa: E402
from perf.cxxcore3_runner_backend_shadow import (  # noqa: E402
    as_runner_backend_shadow_inputs,
    compare_runner_backend_states,
    compare_summaries,
    state_summary,
)
from perf.cxxcore4_cost_layout_backend import (  # noqa: E402
    as_cost_layout,
    as_cost_layout_backend_inputs,
    build_cost_layout_from_surfaces,
    compare_cost_vectors,
    cost_layout_audit_payload,
    cost_matrix_from_layout,
    evaluate_cost_layout_python,
    layout_summary,
    run_cost_layout_backend_python_reference,
)
from perf.cxxcore_env_apply_shadow import config_from_env  # noqa: E402
from perf.native.cost_charge_surface_kernel import COST_SURFACE_COLUMN_NAMES, pack_cost_charge_surfaces  # noqa: E402
from simulation.rng_manager import RNGManager  # noqa: E402
from simulation.runner_pkg.runner_config import SimConfig  # noqa: E402
from tools.scripts.eiel_cxxcore3_runner_backend_shadow_probe import (  # noqa: E402
    compile_cpp_kernel as compile_cxxcore3_kernel,
    load_runner_backend as load_cxxcore3_backend,
)


SUMMARY_NAMES = ("energy_sum", "energy_min", "energy_max", "died_count", "pos_checksum", "grid_sum", "cost_total")


def compile_cpp_kernel(build_dir: Path, *, cxx: str | None = None) -> dict[str, Any]:
    compiler = cxx or shutil.which("g++")
    if not compiler:
        return {"ok": False, "reason": "g++ not found", "library_path": "", "command": []}
    source_path = ROOT / "engine" / "perf" / "native" / "cxxcore4_cost_layout_backend_kernel.cpp"
    build_dir.mkdir(parents=True, exist_ok=True)
    library_path = build_dir / "libcxxcore4_cost_layout_backend.so"
    cmd = [
        str(compiler),
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        str(source_path),
        "-o",
        str(library_path),
    ]
    cp = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "ok": cp.returncode == 0,
        "reason": "" if cp.returncode == 0 else "compile failed",
        "command": cmd,
        "returncode": int(cp.returncode),
        "stdout_tail": cp.stdout[-4000:],
        "stderr_tail": cp.stderr[-4000:],
        "source_path": str(source_path),
        "library_path": str(library_path) if cp.returncode == 0 else "",
    }


def load_cost_layout_backend(library_path: Path):
    lib = ctypes.CDLL(str(library_path))
    f32p = ctypes.POINTER(ctypes.c_float)
    f64p = ctypes.POINTER(ctypes.c_double)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i8p = ctypes.POINTER(ctypes.c_int8)
    u8p = ctypes.POINTER(ctypes.c_uint8)

    scalar_fn = lib.cxxcore4_cost_layout_scalars
    scalar_fn.argtypes = [f64p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, f64p]
    scalar_fn.restype = ctypes.c_int

    backend_fn = lib.cxxcore4_backend_run_with_cost_layout
    backend_fn.argtypes = [
        i32p, f32p, f32p, f32p, f32p, i32p, i32p, ctypes.c_int64,
        i32p, i8p, f32p, f32p, i32p, f64p, ctypes.c_int64, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_int32, f32p, f32p, f32p,
        f32p, u8p, i32p, i32p, i8p, f32p, f64p, f64p,
    ]
    backend_fn.restype = ctypes.c_int

    def scalars(layout: Any) -> np.ndarray:
        rows = np.ascontiguousarray(layout.rows, dtype=np.float64)
        out = np.zeros((int(rows.shape[0]),), dtype=np.float64)
        status = scalar_fn(
            rows.ctypes.data_as(f64p),
            ctypes.c_int64(int(rows.shape[0])),
            ctypes.c_int64(int(rows.shape[1])),
            ctypes.c_int32(1 if layout.charge_channels else 0),
            ctypes.c_int32(1 if layout.charge_radius else 0),
            ctypes.c_int32(1 if layout.ledger_enabled else 0),
            out.ctypes.data_as(f64p),
        )
        if int(status) != 0:
            raise RuntimeError(f"CXXCORE-4 native scalar evaluator failed with status {int(status)}")
        return out

    def backend(inputs: Any, *, export_final: bool = True) -> dict[str, Any]:
        runner = inputs.runner
        state = runner.state
        layout = inputs.layout
        ids = np.ascontiguousarray(state.agent_ids, dtype=np.int32)
        rows = np.ascontiguousarray(layout.rows, dtype=np.float64)
        n = int(ids.shape[0])
        cells = int(state.config.height) * int(state.config.width)
        out_energy = np.zeros((n,), dtype=np.float32)
        out_health = np.zeros((n,), dtype=np.float32)
        out_injury = np.zeros((n,), dtype=np.float32)
        out_scar = np.zeros((n,), dtype=np.float32)
        out_died = np.zeros((n,), dtype=np.uint8)
        out_pos_y = np.zeros((n,), dtype=np.int32)
        out_pos_x = np.zeros((n,), dtype=np.int32)
        out_grid = np.zeros((cells,), dtype=np.int8)
        out_cost = np.zeros((n,), dtype=np.float32)
        out_layout_scalar = np.zeros((n,), dtype=np.float64)
        out_summary = np.zeros((len(SUMMARY_NAMES),), dtype=np.float64)
        status = backend_fn(
            ids.ctypes.data_as(i32p),
            np.ascontiguousarray(state.initial_energy, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(state.initial_health, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(state.initial_injury, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(state.initial_scar, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(state.pos_y_by_id, dtype=np.int32).ctypes.data_as(i32p),
            np.ascontiguousarray(state.pos_x_by_id, dtype=np.int32).ctypes.data_as(i32p),
            ctypes.c_int64(int(state.pos_y_by_id.shape[0])),
            np.ascontiguousarray(state.agent_grid, dtype=np.int32).ctypes.data_as(i32p),
            np.ascontiguousarray(state.grid, dtype=np.int8).ctypes.data_as(i8p),
            np.ascontiguousarray(state.move_cost_mults, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(state.juvenile_harvest_mults, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(state.actions_by_tick, dtype=np.int32).ctypes.data_as(i32p),
            rows.ctypes.data_as(f64p),
            ctypes.c_int64(int(rows.shape[1])),
            ctypes.c_int32(1 if layout.charge_channels else 0),
            ctypes.c_int32(1 if layout.charge_radius else 0),
            ctypes.c_int32(1 if layout.ledger_enabled else 0),
            ctypes.c_int64(int(state.actions_by_tick.shape[0])),
            ctypes.c_int64(n),
            ctypes.c_int64(int(state.config.height)),
            ctypes.c_int64(int(state.config.width)),
            ctypes.c_float(float(state.config.move_cost)),
            ctypes.c_float(float(state.config.gather_cost)),
            ctypes.c_float(float(state.config.rest_cost)),
            ctypes.c_float(float(state.config.attack_cost)),
            ctypes.c_float(float(state.config.action_cost_mult)),
            ctypes.c_float(float(state.config.rest_cost_mult)),
            ctypes.c_float(float(state.config.food_energy)),
            ctypes.c_float(float(state.config.max_energy)),
            ctypes.c_int32(1 if export_final else 0),
            out_energy.ctypes.data_as(f32p),
            out_health.ctypes.data_as(f32p),
            out_injury.ctypes.data_as(f32p),
            out_scar.ctypes.data_as(f32p),
            out_died.ctypes.data_as(u8p),
            out_pos_y.ctypes.data_as(i32p),
            out_pos_x.ctypes.data_as(i32p),
            out_grid.ctypes.data_as(i8p),
            out_cost.ctypes.data_as(f32p),
            out_layout_scalar.ctypes.data_as(f64p),
            out_summary.ctypes.data_as(f64p),
        )
        if int(status) != 0:
            raise RuntimeError(f"CXXCORE-4 native backend failed with status {int(status)}")
        return {
            "new_energy": out_energy,
            "new_health": out_health,
            "new_injury": out_injury,
            "new_scar": out_scar,
            "died_mask": out_died.astype(bool),
            "new_pos_y": out_pos_y,
            "new_pos_x": out_pos_x,
            "grid": out_grid.reshape((int(state.config.height), int(state.config.width))),
            "cost_total_by_agent": out_cost,
            "cost_total": np.asarray([float(np.sum(out_cost, dtype=np.float64))], dtype=np.float64),
            "layout_cost_scalar_by_agent": out_layout_scalar,
            "native_summary": {name: float(out_summary[i]) for i, name in enumerate(SUMMARY_NAMES)},
        }

    return scalars, backend


def _genome(agent_id: int) -> Genome:
    return Genome(
        hidden_size=64,
        exploration_bias=0.1,
        aggression_bias=0.2,
        reproduction_threshold=30.0,
        mutation_rate=0.03,
        action_temperature=1.0,
        use_gru=True,
        memory_dim=(0, 8, 16, 32, 64)[agent_id % 5],
        sensor_channel_mask=implemented_sensor_channel_mask(),
        sensor_radius=1 + int(agent_id % 3),
        introspection_channel_mask=implemented_introspection_channel_mask(),
        self_regulation_control_mask=implemented_self_regulation_control_mask(),
        self_modification_target_mask=implemented_self_modification_target_mask(),
        directed_inheritance_channel_mask=implemented_directed_inheritance_channel_mask(),
        drive_inertia=0.85,
        drive_noise=0.02,
        drive_logit_gain=1.0 + 0.01 * float(agent_id % 11),
        pred_lr=0.05 + 0.01 * float(agent_id % 31),
        developmental_program=canonicalize_developmental_program(None),
    )


def _agent(agent_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=int(agent_id),
        genome=_genome(int(agent_id)),
        energy=80.0,
        health=9.0,
        alive=True,
        drives=np.asarray([0.8, 0.2, 0.4, 0.6], dtype=np.float32),
        stats=SimpleNamespace(
            survival_steps=1 + int(agent_id % 40),
            pred_err_count=2,
            pred_err_sum=0.5,
            prediction_update_count=4,
            offspring_count=0,
            patch_habitat_lineage_id=1,
        ),
        directed_inheritance_payload=None,
    )


def _charged_cfg() -> SimConfig:
    sim_cfg = SimConfig(seed=17)
    sim_cfg.enable_experimental_prediction_cost_charging = True
    sim_cfg.prediction_cost_base = 0.01
    sim_cfg.prediction_cost_per_lr = 0.02
    sim_cfg.prediction_state_cost = 0.003
    sim_cfg.enable_experimental_perception_cost_charging = True
    sim_cfg.enable_experimental_sensor_radius_cost_charging = True
    sim_cfg.enable_developmental_cognition = True
    sim_cfg.developmental_cognition_mode = "linear_maturation"
    sim_cfg.enable_experimental_developmental_cost_charging = True
    sim_cfg.developmental_cognition_cost_base = 0.004
    sim_cfg.maturation_cost_per_tick = 0.002
    sim_cfg.enable_introspection_channels = True
    sim_cfg.force_all_introspection_channels = True
    sim_cfg.enable_experimental_introspection_cost_charging = True
    sim_cfg.enable_self_regulation = True
    sim_cfg.force_all_self_regulation_controls = True
    sim_cfg.enable_experimental_self_regulation_cost_charging = True
    sim_cfg.enable_self_modification = True
    sim_cfg.force_all_self_modification_targets = True
    sim_cfg.enable_experimental_self_modification_cost_charging = True
    sim_cfg.env_hotpath_inactive_phenotype_defaults_enable = True
    return sim_cfg


def _env_cfg(width: int, height: int) -> EnvConfig:
    return EnvConfig(
        width=width,
        height=height,
        food_spawn_per_step=0,
        max_food=0,
        enable_biomes=False,
        enable_combat=False,
        enable_injury=False,
        enable_scar=False,
        migration_pressure_enable=False,
        tier3_validation_force_random_walk=False,
        gru_cost_base=0.0,
        gru_cost_per_dim=0.0,
    )


def _capture_fixture(*, n: int, width: int, height: int, ticks: int, seed: int):
    if n > width * height:
        raise ValueError("fixture requires n <= width * height")
    rm = RNGManager(global_seed=seed)
    env_cfg = _env_cfg(width, height)
    env = GridWorld(env_cfg, seed=seed, rng=rm.env_np)
    for aid in range(n):
        env.add_agent(aid, pos=(int(aid // width), int(aid % width)))
    env.grid[:, :] = 0
    agent_ids = np.arange(n, dtype=np.int32)
    actions = np.empty((ticks, n), dtype=np.int32)
    for tick in range(ticks):
        actions[tick, :] = ((agent_ids * 3 + tick) % 7).astype(np.int32)
    if ticks > 0:
        gather0 = agent_ids[actions[0] == 4]
        env.grid[env.agent_pos_y[gather0], env.agent_pos_x[gather0]] = 1
    sim_cfg = _charged_cfg()
    agents = [_agent(int(aid)) for aid in agent_ids.tolist()]
    production_t0 = time.perf_counter()
    surfaces = [express_cost_charge_surface(agent.genome, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg) for agent in agents]
    production_scalars = np.ascontiguousarray(
        [compute_env_cost_charge_scalar_from_surface(surface, sim_cfg=sim_cfg) for surface in surfaces],
        dtype=np.float64,
    )
    production_s = time.perf_counter() - production_t0
    build_t0 = time.perf_counter()
    layout = build_cost_layout_from_surfaces(agent_ids=agent_ids, surfaces=surfaces, sim_cfg=sim_cfg)
    layout_build_s = time.perf_counter() - build_t0
    digest_t0 = time.perf_counter()
    stable_digest = as_cost_layout(
        agent_ids=agent_ids,
        rows=pack_cost_charge_surfaces(surfaces),
        charge_channels=layout.charge_channels,
        charge_radius=layout.charge_radius,
        ledger_enabled=layout.ledger_enabled,
    ).digest == layout.digest
    digest_s = time.perf_counter() - digest_t0
    inputs = as_cost_layout_backend_inputs(
        config=config_from_env(env),
        agent_ids=agent_ids,
        initial_energy=np.full((n,), 120.0, dtype=np.float32),
        initial_health=np.full((n,), 10.0, dtype=np.float32),
        initial_injury=np.zeros((n,), dtype=np.float32),
        initial_scar=np.zeros((n,), dtype=np.float32),
        pos_y_by_id=env.agent_pos_y.copy(),
        pos_x_by_id=env.agent_pos_x.copy(),
        agent_grid=env.agent_grid.copy(),
        grid=env.grid.copy(),
        actions_by_tick=actions,
        layout=layout,
        move_cost_mults=np.ones((n,), dtype=np.float32),
        juvenile_harvest_mults=np.ones((n,), dtype=np.float32),
    )
    timing = {
        "production_py_cost_s": float(production_s),
        "layout_build_s": float(layout_build_s),
        "layout_digest_validation_s": float(digest_s),
        "stable_digest": bool(stable_digest),
    }
    return inputs, production_scalars, timing, layout_summary(layout)


def _python_authoritative(inputs: Any) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    runner = inputs.runner
    state = runner.state
    rm = RNGManager(global_seed=12345)
    env = GridWorld(_env_cfg(int(state.config.width), int(state.config.height)), seed=12345, rng=rm.env_np)
    env.agent_pos_y[:] = state.pos_y_by_id
    env.agent_pos_x[:] = state.pos_x_by_id
    env.agent_grid[:, :] = state.agent_grid
    env.grid[:, :] = state.grid
    energy = state.initial_energy.copy()
    health = state.initial_health.copy()
    injury = state.initial_injury.copy()
    scar = state.initial_scar.copy()
    cost_total = np.zeros_like(energy, dtype=np.float32)
    died = np.zeros_like(energy, dtype=bool)
    timing = {"py_env_s": 0.0, "py_cost_s": 0.0, "action_prep_s": 0.0}
    for tick in range(int(state.actions_by_tick.shape[0])):
        t0 = time.perf_counter()
        actions = state.actions_by_tick[tick].astype(np.int32, copy=False)
        timing["action_prep_s"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        died, energy, health, _, injury, scar = env.step_agents_batch(
            agent_ids=state.agent_ids,
            actions=actions,
            energies=energy,
            healths=health,
            injuries=injury,
            scars=scar,
            move_cost_mults=state.move_cost_mults,
            juvenile_harvest_mults=state.juvenile_harvest_mults,
        )
        timing["py_env_s"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        tick_cost = runner.cost_scalars_by_tick[tick]
        cost_total = np.ascontiguousarray(cost_total + tick_cost)
        energy = np.maximum(0.0, energy - tick_cost).astype(np.float32, copy=False)
        died = ((energy <= 0.0) | (health <= 0.0)).astype(bool, copy=False)
        timing["py_cost_s"] += time.perf_counter() - t0
    return {
        "new_energy": energy,
        "new_health": health,
        "new_injury": injury,
        "new_scar": scar,
        "died_mask": died,
        "new_pos_y": env.agent_pos_y[state.agent_ids].astype(np.int32, copy=True),
        "new_pos_x": env.agent_pos_x[state.agent_ids].astype(np.int32, copy=True),
        "grid": env.grid.copy(),
        "cost_total_by_agent": cost_total,
        "cost_total": np.asarray([float(np.sum(cost_total, dtype=np.float64))], dtype=np.float64),
    }, timing


def _arm(shape: str, name: str, *, n: int, cells: int, ticks: int) -> dict[str, Any]:
    return {
        "shape": shape,
        "arm": name,
        "agents": int(n),
        "cells": int(cells),
        "ticks": int(ticks),
        "wall_s": 0.0,
        "production_py_cost_s": 0.0,
        "layout_build_s": 0.0,
        "layout_digest_validation_s": 0.0,
        "py_layout_eval_s": 0.0,
        "native_layout_eval_s": 0.0,
        "backend_compute_s": 0.0,
        "materialize_s": 0.0,
        "compare_s": 0.0,
        "py_env_s": 0.0,
        "py_cost_s": 0.0,
        "parity": "not_checked",
        "mismatch_count": 0,
        "max_abs_diff": 0.0,
        "first_divergence": None,
        "interpretation": "",
    }


def run_shape(shape_name: str, *, n: int, width: int, height: int, ticks: int, native: Any, cxxcore3: Any) -> dict[str, Any]:
    cells = width * height
    result: dict[str, Any] = {
        "shape": {"name": shape_name, "agents": n, "width": width, "height": height, "cells": cells, "ticks": ticks},
        "layout": {},
        "arms": [],
    }
    t0 = time.perf_counter()
    inputs, production_scalars, build_timing, summary = _capture_fixture(n=n, width=width, height=height, ticks=ticks, seed=41004)
    capture_total_s = time.perf_counter() - t0
    result["layout"] = summary | {"capture_total_s": float(capture_total_s), **build_timing}

    t0 = time.perf_counter()
    expected, py_timing = _python_authoritative(inputs)
    wall = time.perf_counter() - t0
    arm = _arm(shape_name, "baseline_python_authoritative_with_production_cost", n=n, cells=cells, ticks=ticks)
    arm.update(build_timing)
    arm["wall_s"] = wall + float(build_timing["production_py_cost_s"])
    arm["py_env_s"] = py_timing["py_env_s"]
    arm["py_cost_s"] = py_timing["py_cost_s"]
    arm["parity"] = "not_applicable"
    arm["interpretation"] = "Python authoritative env apply with production CostChargeSurface-derived layout scalar"
    result["arms"].append(arm)

    t0 = time.perf_counter()
    py_layout_scalars = evaluate_cost_layout_python(inputs.layout)
    py_eval_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    scalar_report = compare_cost_vectors(
        agent_ids=inputs.layout.agent_ids,
        production=production_scalars,
        layout=py_layout_scalars,
        actions_by_tick=inputs.runner.state.actions_by_tick,
    )
    compare_s = time.perf_counter() - t0
    arm = _arm(shape_name, "python_layout_cost_expression", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = py_eval_s + compare_s
    arm["py_layout_eval_s"] = py_eval_s
    arm["compare_s"] = compare_s
    arm["parity"] = "pass" if scalar_report["ok"] else "fail"
    arm["mismatch_count"] = int(scalar_report["mismatch_count"])
    arm["max_abs_diff"] = float(scalar_report["max_abs_diff"])
    arm["first_divergence"] = scalar_report["first_divergence"]
    arm["interpretation"] = "array layout scalar evaluator versus production object expression"
    result["arms"].append(arm)

    native_scalars, native_backend = native
    t0 = time.perf_counter()
    native_layout_scalars = native_scalars(inputs.layout)
    native_eval_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    native_report = compare_cost_vectors(
        agent_ids=inputs.layout.agent_ids,
        production=production_scalars,
        layout=native_layout_scalars,
        actions_by_tick=inputs.runner.state.actions_by_tick,
    )
    compare_s = time.perf_counter() - t0
    arm = _arm(shape_name, "native_layout_cost_expression", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = native_eval_s + compare_s
    arm["native_layout_eval_s"] = native_eval_s
    arm["compare_s"] = compare_s
    arm["parity"] = "pass" if native_report["ok"] else "fail"
    arm["mismatch_count"] = int(native_report["mismatch_count"])
    arm["max_abs_diff"] = float(native_report["max_abs_diff"])
    arm["first_divergence"] = native_report["first_divergence"]
    arm["interpretation"] = "native scalar evaluator over already-built layout rows"
    result["arms"].append(arm)

    t0 = time.perf_counter()
    _ = native_scalars(inputs.layout)
    arm = _arm(shape_name, "native_layout_kernel_only_microbench", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = time.perf_counter() - t0
    arm["native_layout_eval_s"] = arm["wall_s"]
    arm["parity"] = "not_applicable"
    arm["interpretation"] = "native scalar evaluator only; no production object walk or compare"
    result["arms"].append(arm)

    t0 = time.perf_counter()
    cxx3_inputs = as_runner_backend_shadow_inputs(
        config=inputs.runner.state.config,
        agent_ids=inputs.runner.state.agent_ids,
        initial_energy=inputs.runner.state.initial_energy,
        initial_health=inputs.runner.state.initial_health,
        initial_injury=inputs.runner.state.initial_injury,
        initial_scar=inputs.runner.state.initial_scar,
        pos_y_by_id=inputs.runner.state.pos_y_by_id,
        pos_x_by_id=inputs.runner.state.pos_x_by_id,
        agent_grid=inputs.runner.state.agent_grid,
        grid=inputs.runner.state.grid,
        actions_by_tick=inputs.runner.state.actions_by_tick,
        cost_scalars_by_tick=cost_matrix_from_layout(inputs.layout, inputs.runner.state.actions_by_tick),
        move_cost_mults=inputs.runner.state.move_cost_mults,
        juvenile_harvest_mults=inputs.runner.state.juvenile_harvest_mults,
    )
    cxx3_actual = cxxcore3(cxx3_inputs, export_final=True)
    compute_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    report = compare_runner_backend_states(expected, cxx3_actual, atol=1.0e-5)
    compare_s = time.perf_counter() - t0
    arm = _arm(shape_name, "cxxcore3_backend_with_precomputed_cost_scalar", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = compute_s + compare_s
    arm["backend_compute_s"] = compute_s
    arm["compare_s"] = compare_s
    arm["parity"] = "pass" if report["ok"] else "fail"
    arm["mismatch_count"] = int(report["mismatch_count"])
    arm["max_abs_diff"] = float(report["max_abs_diff"])
    arm["first_divergence"] = report["first_divergence"]
    arm["interpretation"] = "CXXCORE-3 scalar fallback/control using Python layout-computed scalars"
    result["arms"].append(arm)

    t0 = time.perf_counter()
    py_layout_expected = run_cost_layout_backend_python_reference(inputs)
    py_backend_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    report = compare_runner_backend_states(expected, py_layout_expected, atol=1.0e-5)
    compare_s = time.perf_counter() - t0
    arm = _arm(shape_name, "cxxcore4_backend_with_python_layout_cost", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = py_backend_s + compare_s
    arm["py_layout_eval_s"] = py_eval_s
    arm["backend_compute_s"] = py_backend_s
    arm["compare_s"] = compare_s
    arm["parity"] = "pass" if report["ok"] else "fail"
    arm["mismatch_count"] = int(report["mismatch_count"])
    arm["max_abs_diff"] = float(report["max_abs_diff"])
    arm["first_divergence"] = report["first_divergence"]
    arm["interpretation"] = "backend reference receives scalars computed from layout arrays"
    result["arms"].append(arm)

    t0 = time.perf_counter()
    actual = native_backend(inputs, export_final=True)
    compute_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    report = compare_runner_backend_states(expected, actual, atol=1.0e-5)
    compare_s = time.perf_counter() - t0
    arm = _arm(shape_name, "cxxcore4_backend_with_native_layout_cost", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = compute_s + compare_s
    arm["native_layout_eval_s"] = native_eval_s
    arm["backend_compute_s"] = compute_s
    arm["materialize_s"] = compute_s
    arm["compare_s"] = compare_s
    arm["parity"] = "pass" if report["ok"] else "fail"
    arm["mismatch_count"] = int(report["mismatch_count"])
    arm["max_abs_diff"] = float(report["max_abs_diff"])
    arm["first_divergence"] = report["first_divergence"]
    arm["interpretation"] = "persistent backend computes cost scalar from layout rows inside native loop"
    result["arms"].append(arm)

    t0 = time.perf_counter()
    summary_actual = native_backend(inputs, export_final=False)
    compute_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    summary_report = compare_summaries(state_summary(expected), summary_actual["native_summary"], atol=1.0e-4)
    compare_s = time.perf_counter() - t0
    arm = _arm(shape_name, "cxxcore4_backend_native_kernel_only_with_cost_layout", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = compute_s + compare_s
    arm["backend_compute_s"] = compute_s
    arm["compare_s"] = compare_s
    arm["parity"] = "pass" if summary_report["ok"] else "fail"
    arm["mismatch_count"] = int(summary_report["mismatch_count"])
    arm["max_abs_diff"] = float(summary_report["max_abs_diff"])
    arm["first_divergence"] = summary_report["first_divergence"]
    arm["interpretation"] = "native backend summary only over already-prepared backend/layout arrays"
    result["arms"].append(arm)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="CXXCORE-4 cost-layout/backend shadow probe.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--skip-large", action="store_true")
    args = parser.parse_args()

    compile4 = compile_cpp_kernel(args.build_dir / "cxxcore4")
    compile3 = compile_cxxcore3_kernel(args.build_dir / "cxxcore3")
    if not compile4.get("ok") or not compile3.get("ok"):
        payload = {
            "schema_version": "cxxcore4_cost_layout_backend_probe_v1",
            "native_compile": compile4,
            "cxxcore3_compile": compile3,
            "audit": cost_layout_audit_payload(),
            "shapes": [],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    native = load_cost_layout_backend(Path(str(compile4["library_path"])))
    cxxcore3 = load_cxxcore3_backend(Path(str(compile3["library_path"])))
    _warm_inputs, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=1, seed=99)
    _python_authoritative(_warm_inputs)
    native[0](_warm_inputs.layout)
    native[1](_warm_inputs, export_final=True)

    shapes = [("small", 256, 32, 32), ("medium", 1024, 64, 64)]
    if not args.skip_large:
        shapes.append(("large", 4096, 64, 64))
    payload = {
        "schema_version": "cxxcore4_cost_layout_backend_probe_v1",
        "native_compile": compile4,
        "cxxcore3_compile": compile3,
        "audit": cost_layout_audit_payload(),
        "cost_layout_fields": list(COST_SURFACE_COLUMN_NAMES),
        "shapes": [
            run_shape(name, n=n, width=w, height=h, ticks=int(args.ticks), native=native, cxxcore3=cxxcore3)
            for name, n, w, h in shapes
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
