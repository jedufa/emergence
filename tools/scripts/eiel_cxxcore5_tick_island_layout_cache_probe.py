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

from env.gridworld_pkg.world import GridWorld  # noqa: E402
from perf.cxxcore4_cost_layout_backend import compare_cost_vectors, evaluate_cost_layout_python  # noqa: E402
from perf.cxxcore5_tick_island_layout_cache import (  # noqa: E402
    CostLayoutCache,
    as_tick_island_inputs,
    audit_payload,
    cache_summary,
    checkpoint_metrics_boundary_classification,
    compare_runner_backend_states,
    compare_summaries,
    parity_payload,
    run_tick_island_python_reference,
    state_summary,
)
from perf.cxxcore_env_apply_shadow import config_from_env  # noqa: E402
from simulation.rng_manager import RNGManager  # noqa: E402
from tools.scripts.eiel_cxxcore4_cost_layout_backend_probe import (  # noqa: E402
    _agent,
    _charged_cfg,
    _env_cfg,
    _python_authoritative,
    load_cost_layout_backend,
)


SUMMARY_NAMES = ("energy_sum", "energy_min", "energy_max", "died_count", "pos_checksum", "grid_sum", "cost_total")


def compile_cpp_kernel(build_dir: Path, *, cxx: str | None = None) -> dict[str, Any]:
    compiler = cxx or shutil.which("g++")
    if not compiler:
        return {"ok": False, "reason": "g++ not found", "library_path": "", "command": []}
    source_path = ROOT / "engine" / "perf" / "native" / "cxxcore5_tick_island_kernel.cpp"
    build_dir.mkdir(parents=True, exist_ok=True)
    library_path = build_dir / "libcxxcore5_tick_island.so"
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


def load_tick_island_backend(library_path: Path):
    lib = ctypes.CDLL(str(library_path))
    fn = lib.cxxcore5_tick_island_run
    f32p = ctypes.POINTER(ctypes.c_float)
    f64p = ctypes.POINTER(ctypes.c_double)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i8p = ctypes.POINTER(ctypes.c_int8)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    fn.argtypes = [
        i32p, f32p, f32p, f32p, f32p, i32p, i32p, ctypes.c_int64,
        i32p, i8p, f32p, f32p, i32p, f64p, ctypes.c_int64, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_int32, f32p, f32p, f32p,
        f32p, u8p, i32p, i32p, i8p, f32p, f64p, f64p,
    ]
    fn.restype = ctypes.c_int

    def _call(inputs: Any, *, export_final: bool = True) -> dict[str, Any]:
        backend = inputs.backend
        state = backend.runner.state
        layout = backend.layout
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
            raise RuntimeError(f"CXXCORE-5 native tick island failed with status {int(status)}")
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

    return _call


def _capture_fixture(*, n: int, width: int, height: int, ticks: int, seed: int, age_shift: int = 0):
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
    if age_shift:
        for agent in agents[: max(1, min(8, len(agents)))]:
            agent.stats.survival_steps += int(age_shift)
    cache = CostLayoutCache(env_cfg=env_cfg, sim_cfg=sim_cfg)
    full_snapshot, full_info = cache.build_or_update(agents, force_full_rebuild=True)
    hit_snapshot, hit_info = cache.build_or_update(agents)
    if agents:
        agents[0].stats.survival_steps += 1
    update_snapshot, update_info = cache.build_or_update(agents)
    inputs = as_tick_island_inputs(
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
        cache_snapshot=hit_snapshot,
        move_cost_mults=np.ones((n,), dtype=np.float32),
        juvenile_harvest_mults=np.ones((n,), dtype=np.float32),
    )
    update_inputs = as_tick_island_inputs(
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
        cache_snapshot=update_snapshot,
        move_cost_mults=np.ones((n,), dtype=np.float32),
        juvenile_harvest_mults=np.ones((n,), dtype=np.float32),
    )
    return inputs, update_inputs, full_snapshot, full_info, hit_snapshot, hit_info, update_snapshot, update_info


def _arm(shape: str, name: str, *, n: int, cells: int, ticks: int) -> dict[str, Any]:
    return {
        "shape": shape,
        "arm": name,
        "agents": int(n),
        "cells": int(cells),
        "ticks": int(ticks),
        "wall_s": 0.0,
        "py_env_s": 0.0,
        "production_cost_s": 0.0,
        "layout_rebuild_update_s": 0.0,
        "cache_hit_pct": 0.0,
        "native_layout_eval_s": 0.0,
        "native_tick_s": 0.0,
        "compare_s": 0.0,
        "snapshot_copy_packing_s": 0.0,
        "materialization_s": 0.0,
        "diagnostics_overhead_s": 0.0,
        "parity": "not_checked",
        "mismatch_count": 0,
        "max_abs_diff": 0.0,
        "first_divergence": None,
        "cache_mode": "",
        "cache_hits": 0,
        "cache_misses": 0,
        "rows_rebuilt": 0,
        "full_rebuilds": 0,
        "interpretation": "",
    }


def _fill_parity(arm: dict[str, Any], payload: dict[str, Any]) -> None:
    arm["parity"] = payload["parity"]
    arm["mismatch_count"] = payload["mismatch_count"]
    arm["max_abs_diff"] = payload["max_abs_diff"]
    arm["first_divergence"] = payload["first_divergence"]
    arm["cache_mode"] = payload["cache_mode"]
    arm["cache_hits"] = payload["cache_hits"]
    arm["cache_misses"] = payload["cache_misses"]
    arm["rows_rebuilt"] = payload["rows_rebuilt"]
    arm["full_rebuilds"] = payload["full_rebuilds"]
    arm["parity_payload"] = payload


def run_shape(shape_name: str, *, n: int, width: int, height: int, ticks: int, native: Any, cxxcore4_native: Any) -> dict[str, Any]:
    cells = width * height
    result: dict[str, Any] = {
        "shape": {"name": shape_name, "agents": n, "width": width, "height": height, "cells": cells, "ticks": ticks},
        "arms": [],
        "cache": {},
    }
    t0 = time.perf_counter()
    inputs, update_inputs, full_snapshot, full_info, hit_snapshot, hit_info, update_snapshot, update_info = _capture_fixture(
        n=n,
        width=width,
        height=height,
        ticks=ticks,
        seed=51005,
    )
    capture_s = time.perf_counter() - t0
    result["cache"] = {
        "full_rebuild": cache_summary(full_snapshot),
        "hit": cache_summary(hit_snapshot),
        "row_update": cache_summary(update_snapshot),
    }

    t0 = time.perf_counter()
    expected, py_timing = _python_authoritative(inputs.backend)
    py_wall_s = time.perf_counter() - t0
    arm = _arm(shape_name, "baseline_python_authoritative_with_production_cost", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = py_wall_s + full_snapshot.production_cost_s
    arm["py_env_s"] = py_timing["py_env_s"]
    arm["production_cost_s"] = full_snapshot.production_cost_s
    arm["snapshot_copy_packing_s"] = capture_s
    arm["parity"] = "not_applicable"
    arm["interpretation"] = "Python authoritative env apply plus production object/surface cost expression"
    result["arms"].append(arm)

    scalar_report = compare_cost_vectors(
        agent_ids=full_snapshot.layout.agent_ids,
        production=full_info["production_scalars"],
        layout=full_info["layout_scalars"],
        actions_by_tick=inputs.backend.runner.state.actions_by_tick,
    )
    arm = _arm(shape_name, "python_authoritative_with_full_layout_rebuild", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = py_wall_s + full_snapshot.build_s
    arm["py_env_s"] = py_timing["py_env_s"]
    arm["production_cost_s"] = full_snapshot.production_cost_s
    arm["layout_rebuild_update_s"] = full_snapshot.build_s
    arm["cache_hit_pct"] = 0.0
    _fill_parity(arm, parity_payload(shape=shape_name, arm=arm["arm"], agents=n, ticks=ticks, cells=cells, report=scalar_report, snapshot=full_snapshot, cache_mode="full_rebuild"))
    arm["interpretation"] = "full rebuild layout scalar matches production expression"
    result["arms"].append(arm)

    hit_report = {"ok": hit_snapshot.layout.digest == full_snapshot.layout.digest, "mismatch_count": 0, "max_abs_diff": 0.0, "first_divergence": None}
    if not hit_report["ok"]:
        hit_report["mismatch_count"] = 1
        hit_report["first_divergence"] = {"field": "layout_digest", "python": full_snapshot.layout.digest, "candidate": hit_snapshot.layout.digest}
    arm = _arm(shape_name, "python_authoritative_with_cached_layout", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = py_wall_s + hit_snapshot.cache_lookup_s + hit_snapshot.digest_validation_s
    arm["py_env_s"] = py_timing["py_env_s"]
    arm["layout_rebuild_update_s"] = hit_snapshot.cache_lookup_s + hit_snapshot.digest_validation_s
    arm["cache_hit_pct"] = 100.0 * hit_snapshot.cache_hits / max(1, hit_snapshot.cache_hits + hit_snapshot.cache_misses)
    _fill_parity(arm, parity_payload(shape=shape_name, arm=arm["arm"], agents=n, ticks=ticks, cells=cells, report=hit_report, snapshot=hit_snapshot, cache_mode="hit"))
    arm["interpretation"] = "cache hit reuses identical layout digest without production object walk"
    result["arms"].append(arm)

    update_report = compare_cost_vectors(
        agent_ids=update_snapshot.layout.agent_ids,
        production=update_info["production_scalars"],
        layout=update_info["layout_scalars"],
        actions_by_tick=update_inputs.backend.runner.state.actions_by_tick,
    )
    arm = _arm(shape_name, "python_authoritative_with_row_update", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = py_wall_s + update_snapshot.update_s
    arm["py_env_s"] = py_timing["py_env_s"]
    arm["production_cost_s"] = update_snapshot.production_cost_s
    arm["layout_rebuild_update_s"] = update_snapshot.update_s
    arm["cache_hit_pct"] = 100.0 * update_snapshot.cache_hits / max(1, update_snapshot.cache_hits + update_snapshot.cache_misses)
    _fill_parity(arm, parity_payload(shape=shape_name, arm=arm["arm"], agents=n, ticks=ticks, cells=cells, report=update_report, snapshot=update_snapshot, cache_mode="row_update"))
    arm["interpretation"] = "conservative row update after age/developmental invalidation"
    result["arms"].append(arm)

    cxx4_backend = cxxcore4_native[1]
    t0 = time.perf_counter()
    cxx4_actual = cxx4_backend(inputs.backend, export_final=True)
    cxx4_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    cxx4_report = compare_runner_backend_states(expected, cxx4_actual, atol=1.0e-5)
    compare_s = time.perf_counter() - t0
    arm = _arm(shape_name, "cxxcore4_backend_with_native_layout_cost", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = cxx4_s + compare_s
    arm["native_tick_s"] = cxx4_s
    arm["compare_s"] = compare_s
    _fill_parity(arm, parity_payload(shape=shape_name, arm=arm["arm"], agents=n, ticks=ticks, cells=cells, report=cxx4_report, snapshot=hit_snapshot, cache_mode="hit"))
    arm["interpretation"] = "CXXCORE-4 backend regression/control"
    result["arms"].append(arm)

    t0 = time.perf_counter()
    actual = native(inputs, export_final=True)
    native_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    report = compare_runner_backend_states(expected, actual, atol=1.0e-5)
    compare_s = time.perf_counter() - t0
    arm = _arm(shape_name, "cxxcore5_tick_island_shadow_final_compare", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = native_s + compare_s
    arm["native_tick_s"] = native_s
    arm["materialization_s"] = native_s
    arm["compare_s"] = compare_s
    _fill_parity(arm, parity_payload(shape=shape_name, arm=arm["arm"], agents=n, ticks=ticks, cells=cells, report=report, snapshot=hit_snapshot, cache_mode="hit"))
    arm["interpretation"] = "CXXCORE-5 copied-state tick island, final full compare"
    result["arms"].append(arm)

    t0 = time.perf_counter()
    summary_actual = native(inputs, export_final=False)
    sparse_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    sparse_report = compare_summaries(state_summary(expected), summary_actual["native_summary"], atol=1.0e-4)
    compare_s = time.perf_counter() - t0
    arm = _arm(shape_name, "cxxcore5_tick_island_shadow_sparse_compare", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = sparse_s + compare_s
    arm["native_tick_s"] = sparse_s
    arm["compare_s"] = compare_s
    _fill_parity(arm, parity_payload(shape=shape_name, arm=arm["arm"], agents=n, ticks=ticks, cells=cells, report=sparse_report, snapshot=hit_snapshot, cache_mode="hit"))
    arm["interpretation"] = "CXXCORE-5 copied-state tick island, sparse summary compare"
    result["arms"].append(arm)

    t0 = time.perf_counter()
    kernel_only = native(inputs, export_final=False)
    kernel_s = time.perf_counter() - t0
    arm = _arm(shape_name, "cxxcore5_native_tick_island_kernel_only", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = kernel_s
    arm["native_tick_s"] = kernel_s
    arm["parity"] = "not_applicable"
    arm["cache_mode"] = "hit"
    arm["cache_hits"] = hit_snapshot.cache_hits
    arm["cache_misses"] = hit_snapshot.cache_misses
    arm["rows_rebuilt"] = hit_snapshot.rows_rebuilt
    arm["full_rebuilds"] = hit_snapshot.full_rebuilds
    arm["interpretation"] = f"already-prepared arrays, native summary cost_total={kernel_only['native_summary']['cost_total']:.6f}"
    result["arms"].append(arm)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="CXXCORE-5 tick-island layout-cache probe.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--skip-large", action="store_true")
    args = parser.parse_args()

    compile5 = compile_cpp_kernel(args.build_dir / "cxxcore5")
    compile4 = __import__(
        "tools.scripts.eiel_cxxcore4_cost_layout_backend_probe",
        fromlist=["compile_cpp_kernel"],
    ).compile_cpp_kernel(args.build_dir / "cxxcore4")
    if not compile5.get("ok") or not compile4.get("ok"):
        payload = {
            "schema_version": "cxxcore5_tick_island_layout_cache_probe_v1",
            "native_compile": compile5,
            "cxxcore4_compile": compile4,
            "audit": audit_payload(),
            "shapes": [],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    native = load_tick_island_backend(Path(str(compile5["library_path"])))
    cxxcore4_native = load_cost_layout_backend(Path(str(compile4["library_path"])))
    warm, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=1, seed=1)
    native(warm, export_final=True)

    shapes = [("small", 256, 32, 32), ("medium", 1024, 64, 64)]
    if not args.skip_large:
        shapes.append(("large", 4096, 64, 64))
    runner_pilot = ("runner_captured_replay", 128, 16, 16, min(240, int(args.ticks)))
    payload = {
        "schema_version": "cxxcore5_tick_island_layout_cache_probe_v1",
        "native_compile": compile5,
        "cxxcore4_compile": compile4,
        "audit": audit_payload(),
        "checkpoint_metrics_boundary": checkpoint_metrics_boundary_classification(),
        "runner_captured_pilot_mode": "captured-action replay fixture; live runner hook deferred to avoid runtime backend selection and RNG/checkpoint widening",
        "shapes": [
            run_shape(name, n=n, width=w, height=h, ticks=int(args.ticks), native=native, cxxcore4_native=cxxcore4_native)
            for name, n, w, h in shapes
        ]
        + [
            run_shape(
                runner_pilot[0],
                n=runner_pilot[1],
                width=runner_pilot[2],
                height=runner_pilot[3],
                ticks=runner_pilot[4],
                native=native,
                cxxcore4_native=cxxcore4_native,
            )
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
