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
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "engine") not in sys.path:
    sys.path.insert(0, str(ROOT / "engine"))

from env.env_config import EnvConfig  # noqa: E402
from env.gridworld_pkg.world import GridWorld  # noqa: E402
from perf.cxxcore2_state_backend import as_backend_inputs  # noqa: E402
from perf.cxxcore3_runner_backend_shadow import (  # noqa: E402
    audit_payload,
    as_runner_backend_shadow_inputs,
    compare_runner_backend_states,
    compare_summaries,
    deterministic_cost_scalar_matrix,
    run_runner_backend_python_reference,
    state_summary,
)
from perf.cxxcore_env_apply_shadow import config_from_env  # noqa: E402
from simulation.rng_manager import RNGManager  # noqa: E402
from tools.scripts.eiel_cxxcore2_state_backend_native_smoke import (  # noqa: E402
    compile_cpp_kernel as compile_cxxcore2_kernel,
    load_backend as load_cxxcore2_backend,
)


SUMMARY_NAMES = ("energy_sum", "energy_min", "energy_max", "died_count", "pos_checksum", "grid_sum", "cost_total")


def compile_cpp_kernel(build_dir: Path, *, cxx: str | None = None) -> dict[str, Any]:
    compiler = cxx or shutil.which("g++")
    if not compiler:
        return {"ok": False, "reason": "g++ not found", "library_path": "", "command": []}
    source_path = ROOT / "engine" / "perf" / "native" / "cxxcore3_runner_backend_kernel.cpp"
    build_dir.mkdir(parents=True, exist_ok=True)
    library_path = build_dir / "libcxxcore3_runner_backend.so"
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


def load_runner_backend(library_path: Path):
    lib = ctypes.CDLL(str(library_path))
    fn = lib.cxxcore3_backend_run_with_cost
    f32p = ctypes.POINTER(ctypes.c_float)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i8p = ctypes.POINTER(ctypes.c_int8)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    f64p = ctypes.POINTER(ctypes.c_double)
    fn.argtypes = [
        i32p, f32p, f32p, f32p, f32p, i32p, i32p, ctypes.c_int64,
        i32p, i8p, f32p, f32p, i32p, f32p, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_int32, f32p, f32p, f32p,
        f32p, u8p, i32p, i32p, i8p, f32p, f64p,
    ]
    fn.restype = ctypes.c_int

    def _call(inputs: Any, *, export_final: bool = True) -> dict[str, Any]:
        state = inputs.state
        ids = np.ascontiguousarray(state.agent_ids, dtype=np.int32)
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
        out_summary = np.zeros((len(SUMMARY_NAMES),), dtype=np.float64)
        status = fn(
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
            np.ascontiguousarray(inputs.cost_scalars_by_tick, dtype=np.float32).ctypes.data_as(f32p),
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
            out_summary.ctypes.data_as(f64p),
        )
        if int(status) != 0:
            raise RuntimeError(f"CXXCORE-3 native backend failed with status {int(status)}")
        payload = {
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
            "native_summary": {name: float(out_summary[i]) for i, name in enumerate(SUMMARY_NAMES)},
        }
        return payload

    return _call


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


def _capture_fixture(*, n: int, width: int, height: int, ticks: int, cost_enabled: bool, seed: int):
    if n > width * height:
        raise ValueError("fixture requires n <= width * height")
    rm = RNGManager(global_seed=seed)
    env = GridWorld(_env_cfg(width, height), seed=seed, rng=rm.env_np)
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
    costs = deterministic_cost_scalar_matrix(agent_ids, actions, enabled=cost_enabled)
    return as_runner_backend_shadow_inputs(
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
        cost_scalars_by_tick=costs,
        move_cost_mults=np.ones((n,), dtype=np.float32),
        juvenile_harvest_mults=np.ones((n,), dtype=np.float32),
    )


def _python_authoritative(inputs: Any) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    state = inputs.state
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
    timing = {"py_env_s": 0.0, "py_cost_s": 0.0, "action_prep_s": 0.0}
    died = np.zeros_like(energy, dtype=bool)
    for tick in range(int(state.actions_by_tick.shape[0])):
        t0 = time.perf_counter()
        actions = state.actions_by_tick[tick].astype(np.int32, copy=False)
        timing["action_prep_s"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        died, energy, health, food_gained, injury, scar = env.step_agents_batch(
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
        tick_cost = inputs.cost_scalars_by_tick[tick]
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
        "py_env_s": 0.0,
        "py_cost_s": 0.0,
        "capture_init_s": 0.0,
        "action_prep_s": 0.0,
        "cost_prep_s": 0.0,
        "cpp_compute_s": 0.0,
        "summary_export_s": 0.0,
        "compare_s": 0.0,
        "diagnostics_s": 0.0,
        "parity": "not_checked",
        "fields_compared": 0,
        "first_divergence": None,
        "interpretation": "",
    }


def run_shape(shape_name: str, *, n: int, width: int, height: int, ticks: int, backend: Any, cxxcore2: Any) -> dict[str, Any]:
    cells = width * height
    result = {"shape": {"name": shape_name, "agents": n, "width": width, "height": height, "cells": cells, "ticks": ticks}, "arms": []}

    t0 = time.perf_counter()
    no_cost_inputs = _capture_fixture(n=n, width=width, height=height, ticks=ticks, cost_enabled=False, seed=31001)
    capture_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    no_cost_expected, py_timing = _python_authoritative(no_cost_inputs)
    arm = _arm(shape_name, "baseline_python_authoritative_no_rng_no_cost", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = time.perf_counter() - t0
    arm["py_env_s"] = py_timing["py_env_s"]
    arm["py_cost_s"] = py_timing["py_cost_s"]
    arm["action_prep_s"] = py_timing["action_prep_s"]
    arm["capture_init_s"] = capture_s
    arm["parity"] = "not_applicable"
    arm["interpretation"] = "Python authoritative runner-compatible no-RNG fixture without cost scalars"
    result["arms"].append(arm)

    t0 = time.perf_counter()
    cost_inputs = _capture_fixture(n=n, width=width, height=height, ticks=ticks, cost_enabled=True, seed=31001)
    capture_cost_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    cost_expected, py_cost_timing = _python_authoritative(cost_inputs)
    arm = _arm(shape_name, "baseline_python_authoritative_no_rng_with_cost", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = time.perf_counter() - t0
    arm["py_env_s"] = py_cost_timing["py_env_s"]
    arm["py_cost_s"] = py_cost_timing["py_cost_s"]
    arm["action_prep_s"] = py_cost_timing["action_prep_s"]
    arm["capture_init_s"] = capture_cost_s
    arm["cost_prep_s"] = capture_cost_s
    arm["parity"] = "not_applicable"
    arm["interpretation"] = "Python authoritative fixture with precomputed cost-scalar fallback"
    result["arms"].append(arm)

    arm = _arm(shape_name, "cxxcore2_synthetic_state_backend_reference", n=n, cells=cells, ticks=ticks)
    if cxxcore2 is None:
        arm["parity"] = "unavailable"
    else:
        state = no_cost_inputs.state
        cxx2_inputs = as_backend_inputs(
            config=state.config,
            agent_ids=state.agent_ids,
            initial_energy=state.initial_energy,
            initial_health=state.initial_health,
            initial_injury=state.initial_injury,
            initial_scar=state.initial_scar,
            pos_y_by_id=state.pos_y_by_id,
            pos_x_by_id=state.pos_x_by_id,
            agent_grid=state.agent_grid,
            grid=state.grid,
            actions_by_tick=state.actions_by_tick,
            move_cost_mults=state.move_cost_mults,
            juvenile_harvest_mults=state.juvenile_harvest_mults,
        )
        t0 = time.perf_counter()
        cxx2_actual = cxxcore2(cxx2_inputs, export_final=True)
        arm["wall_s"] = time.perf_counter() - t0
        arm["cpp_compute_s"] = arm["wall_s"]
        arm["parity"] = "reference"
        arm["interpretation"] = f"CXXCORE-2 checksum={float(cxx2_actual['checksum']):.6f}"
    result["arms"].append(arm)

    t0 = time.perf_counter()
    actual = backend(no_cost_inputs, export_final=True)
    compute_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    parity = compare_runner_backend_states(no_cost_expected, actual)
    compare_s = time.perf_counter() - t0
    arm = _arm(shape_name, "cxxcore3_runner_backend_final_compare_no_cost", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = compute_s + compare_s
    arm["cpp_compute_s"] = compute_s
    arm["compare_s"] = compare_s
    arm["parity"] = "pass" if parity["ok"] else "fail"
    arm["fields_compared"] = len(parity.get("fields_compared", []))
    arm["first_divergence"] = parity.get("first_divergence")
    arm["interpretation"] = "runner-captured persistent backend final full compare, no cost"
    result["arms"].append(arm)

    t0 = time.perf_counter()
    summary_actual = backend(no_cost_inputs, export_final=False)
    compute_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    summary_report = compare_summaries(state_summary(no_cost_expected), summary_actual["native_summary"])
    compare_s = time.perf_counter() - t0
    arm = _arm(shape_name, "cxxcore3_runner_backend_sparse_summary_no_cost", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = compute_s + compare_s
    arm["cpp_compute_s"] = compute_s
    arm["summary_export_s"] = compute_s
    arm["compare_s"] = compare_s
    arm["parity"] = "pass" if summary_report["ok"] else "fail"
    arm["fields_compared"] = len(summary_report.get("fields_compared", []))
    arm["first_divergence"] = summary_report.get("first_divergence")
    arm["interpretation"] = "native sparse summary/checksum compare, no full export"
    result["arms"].append(arm)

    t0 = time.perf_counter()
    cost_actual = backend(cost_inputs, export_final=True)
    compute_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    parity = compare_runner_backend_states(cost_expected, cost_actual)
    compare_s = time.perf_counter() - t0
    arm = _arm(shape_name, "cxxcore3_runner_backend_final_compare_with_cost", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = compute_s + compare_s
    arm["cost_prep_s"] = capture_cost_s
    arm["cpp_compute_s"] = compute_s
    arm["compare_s"] = compare_s
    arm["parity"] = "pass" if parity["ok"] else "fail"
    arm["fields_compared"] = len(parity.get("fields_compared", []))
    arm["first_divergence"] = parity.get("first_divergence")
    arm["interpretation"] = "precomputed cost scalars applied inside backend loop"
    result["arms"].append(arm)

    t0 = time.perf_counter()
    summary_actual = backend(cost_inputs, export_final=False)
    compute_s = time.perf_counter() - t0
    arm = _arm(shape_name, "cxxcore3_runner_backend_kernel_loop_only_with_cost", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = compute_s
    arm["cost_prep_s"] = capture_cost_s
    arm["cpp_compute_s"] = compute_s
    arm["parity"] = "not_applicable"
    arm["interpretation"] = f"checksum-only cost fallback loop cost_total={summary_actual['native_summary']['cost_total']:.6f}"
    result["arms"].append(arm)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="CXXCORE-3 runner backend cost-shadow probe.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--skip-large", action="store_true")
    args = parser.parse_args()
    compile3 = compile_cpp_kernel(args.build_dir / "cxxcore3")
    compile2 = compile_cxxcore2_kernel(args.build_dir / "cxxcore2")
    if not compile3.get("ok"):
        payload = {"schema_version": "cxxcore3_runner_backend_probe_v1", "native_compile": compile3, "shapes": [], "audit": audit_payload()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    backend = load_runner_backend(Path(str(compile3["library_path"])))
    cxxcore2 = load_cxxcore2_backend(Path(str(compile2["library_path"]))) if compile2.get("ok") else None
    _warm_inputs = _capture_fixture(n=8, width=8, height=8, ticks=1, cost_enabled=False, seed=999)
    _python_authoritative(_warm_inputs)
    backend(_warm_inputs, export_final=True)
    shapes = [("small", 256, 32, 32), ("medium", 1024, 64, 64)]
    if not args.skip_large:
        shapes.append(("large", 4096, 64, 64))
    payload = {
        "schema_version": "cxxcore3_runner_backend_probe_v1",
        "native_compile": compile3,
        "cxxcore2_compile": compile2,
        "audit": audit_payload(),
        "shapes": [
            run_shape(name, n=n, width=w, height=h, ticks=int(args.ticks), backend=backend, cxxcore2=cxxcore2)
            for name, n, w, h in shapes
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
