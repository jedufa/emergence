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

from perf.cxxcore6_authoritative_pilot import (  # noqa: E402
    Cxxcore6PilotConfig,
    audit_payload,
    boundary_classification,
    result_payload,
    rng_preflight_audit,
    run_authoritative_pilot,
)
from perf.cxxcore5_tick_island_layout_cache import compare_runner_backend_states, compare_summaries, state_summary  # noqa: E402
from tools.scripts.eiel_cxxcore5_tick_island_layout_cache_probe import (  # noqa: E402
    _capture_fixture,
    _python_authoritative,
    compile_cpp_kernel as compile_cxxcore5_kernel,
    load_tick_island_backend as load_cxxcore5_backend,
)


SUMMARY_NAMES = ("energy_sum", "energy_min", "energy_max", "died_count", "pos_checksum", "grid_sum", "cost_total")


def compile_cpp_kernel(build_dir: Path, *, cxx: str | None = None) -> dict[str, Any]:
    compiler = cxx or shutil.which("g++")
    if not compiler:
        return {"ok": False, "reason": "g++ not found", "library_path": "", "command": []}
    source_path = ROOT / "engine" / "perf" / "native" / "cxxcore6_authoritative_pilot_kernel.cpp"
    build_dir.mkdir(parents=True, exist_ok=True)
    library_path = build_dir / "libcxxcore6_authoritative_pilot.so"
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


def load_authoritative_backend(library_path: Path):
    lib = ctypes.CDLL(str(library_path))
    fn = lib.cxxcore6_authoritative_pilot_run
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
            raise RuntimeError(f"CXXCORE-6 native authoritative pilot failed with status {int(status)}")
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


def _arm(shape: str, name: str, *, n: int, cells: int, ticks: int) -> dict[str, Any]:
    return {
        "shape": shape,
        "arm": name,
        "agents": int(n),
        "cells": int(cells),
        "ticks": int(ticks),
        "actions_covered": int(n) * int(ticks),
        "rows_covered": int(n),
        "wall_s": 0.0,
        "python_baseline_env_tick_s": 0.0,
        "python_remaining_work_s": 0.0,
        "layout_cache_lookup_s": 0.0,
        "layout_row_update_s": 0.0,
        "layout_full_rebuild_s": 0.0,
        "native_backend_compute_s": 0.0,
        "writeback_materialization_s": 0.0,
        "sparse_comparison_s": 0.0,
        "final_comparison_s": 0.0,
        "diagnostic_overhead_s": 0.0,
        "fallback_overhead_s": 0.0,
        "cache_hits": 0,
        "cache_misses": 0,
        "row_updates": 0,
        "parity": "not_checked",
        "first_divergence": None,
        "fallback_occurred": False,
        "fallback_reason": "",
        "interpretation": "",
    }


def _apply_result(arm: dict[str, Any], result: Any) -> None:
    payload = result_payload(result)
    timing = payload["timing"]
    diag = payload["diagnostics"]
    parity = payload.get("parity") or {}
    arm["wall_s"] = float(timing.get("total_wall_s", 0.0))
    arm["native_backend_compute_s"] = float(timing.get("native_backend_s", 0.0))
    arm["writeback_materialization_s"] = float(timing.get("writeback_materialization_s", 0.0))
    arm["sparse_comparison_s"] = float(timing.get("sparse_compare_s", 0.0))
    arm["final_comparison_s"] = float(timing.get("final_compare_s", 0.0))
    arm["fallback_overhead_s"] = float(timing.get("fallback_s", 0.0))
    arm["cache_hits"] = int(diag.get("cache_hits", 0))
    arm["cache_misses"] = int(diag.get("cache_misses", 0))
    arm["row_updates"] = int(diag.get("row_updates", 0))
    arm["fallback_occurred"] = bool(payload["fallback_occurred"])
    arm["fallback_reason"] = str(payload["fallback_reason"])
    if parity:
        arm["parity"] = "pass" if bool(parity.get("ok", False)) else "fail"
        arm["first_divergence"] = parity.get("first_divergence")
    elif payload["status"] == "authoritative":
        arm["parity"] = "not_applicable"
    elif payload["status"] == "fallback":
        arm["parity"] = "fallback"
    else:
        arm["parity"] = payload["status"]
    arm["result"] = payload


def run_shape(shape_name: str, *, n: int, width: int, height: int, ticks: int, native: Any, cxxcore5_native: Any) -> dict[str, Any]:
    cells = width * height
    result: dict[str, Any] = {
        "shape": {"name": shape_name, "agents": n, "width": width, "height": height, "cells": cells, "ticks": ticks},
        "arms": [],
    }
    inputs, _, full_snapshot, _, hit_snapshot, _, update_snapshot, _ = _capture_fixture(
        n=n,
        width=width,
        height=height,
        ticks=ticks,
        seed=61006,
    )
    t0 = time.perf_counter()
    expected, py_timing = _python_authoritative(inputs.backend)
    py_wall = time.perf_counter() - t0

    arm = _arm(shape_name, "baseline_python", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = py_wall + full_snapshot.production_cost_s
    arm["python_baseline_env_tick_s"] = py_timing["py_env_s"]
    arm["python_remaining_work_s"] = full_snapshot.production_cost_s
    arm["layout_full_rebuild_s"] = full_snapshot.build_s
    arm["cache_misses"] = full_snapshot.cache_misses
    arm["parity"] = "not_applicable"
    arm["interpretation"] = "current Python env/tick path plus production cost expression"
    result["arms"].append(arm)

    t0 = time.perf_counter()
    shadow_actual = cxxcore5_native(inputs, export_final=True)
    shadow_native_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    shadow_report = compare_runner_backend_states(expected, shadow_actual, atol=1.0e-5)
    shadow_compare_s = time.perf_counter() - t0
    arm = _arm(shape_name, "shadow_cxxcore5_final_compare", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = py_wall + shadow_native_s + shadow_compare_s
    arm["python_baseline_env_tick_s"] = py_timing["py_env_s"]
    arm["native_backend_compute_s"] = shadow_native_s
    arm["final_comparison_s"] = shadow_compare_s
    arm["cache_hits"] = hit_snapshot.cache_hits
    arm["cache_misses"] = hit_snapshot.cache_misses
    arm["parity"] = "pass" if shadow_report["ok"] else "fail"
    arm["first_divergence"] = shadow_report.get("first_divergence")
    arm["interpretation"] = "Python authoritative shadow/control; does not skip Python env apply"
    result["arms"].append(arm)

    cfg = Cxxcore6PilotConfig(cxxcore6_authoritative_pilot_enable=True, cxxcore6_authoritative_compare_mode="off")
    arm = _arm(shape_name, "authoritative_cxxcore6_no_compare", n=n, cells=cells, ticks=ticks)
    pilot = run_authoritative_pilot(inputs, config=cfg, native_backend=native, compare_mode="off")
    _apply_result(arm, pilot)
    arm["layout_cache_lookup_s"] = hit_snapshot.cache_lookup_s + hit_snapshot.digest_validation_s
    arm["interpretation"] = "C++ owns included no-RNG slice; no expensive reference compare"
    result["arms"].append(arm)

    cfg_sparse = Cxxcore6PilotConfig(cxxcore6_authoritative_pilot_enable=True, cxxcore6_authoritative_compare_mode="sparse")
    arm = _arm(shape_name, "authoritative_cxxcore6_sparse_compare", n=n, cells=cells, ticks=ticks)
    pilot = run_authoritative_pilot(inputs, config=cfg_sparse, native_backend=native, python_reference=expected, compare_mode="sparse")
    _apply_result(arm, pilot)
    arm["layout_cache_lookup_s"] = hit_snapshot.cache_lookup_s + hit_snapshot.digest_validation_s
    arm["interpretation"] = "C++ authoritative slice plus sparse summary guard"
    result["arms"].append(arm)

    cfg_final = Cxxcore6PilotConfig(cxxcore6_authoritative_pilot_enable=True, cxxcore6_authoritative_compare_mode="final_full")
    arm = _arm(shape_name, "authoritative_cxxcore6_final_compare", n=n, cells=cells, ticks=ticks)
    pilot = run_authoritative_pilot(inputs, config=cfg_final, native_backend=native, python_reference=expected, compare_mode="final_full")
    _apply_result(arm, pilot)
    arm["layout_cache_lookup_s"] = hit_snapshot.cache_lookup_s + hit_snapshot.digest_validation_s
    arm["interpretation"] = "C++ authoritative slice plus final full reference compare"
    result["arms"].append(arm)

    t0 = time.perf_counter()
    _ = native(inputs, export_final=False)
    kernel_s = time.perf_counter() - t0
    arm = _arm(shape_name, "kernel_loop_only", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = kernel_s
    arm["native_backend_compute_s"] = kernel_s
    arm["cache_hits"] = hit_snapshot.cache_hits
    arm["cache_misses"] = hit_snapshot.cache_misses
    arm["parity"] = "not_applicable"
    arm["interpretation"] = "native loop over already prepared arrays, no materialized final compare"
    result["arms"].append(arm)

    arm = _arm(shape_name, "cache_hit_layout", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = hit_snapshot.cache_lookup_s + hit_snapshot.digest_validation_s
    arm["layout_cache_lookup_s"] = arm["wall_s"]
    arm["cache_hits"] = hit_snapshot.cache_hits
    arm["cache_misses"] = hit_snapshot.cache_misses
    arm["parity"] = "pass"
    arm["interpretation"] = "layout cache-hit cost only"
    result["arms"].append(arm)

    arm = _arm(shape_name, "row_update_layout", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = update_snapshot.update_s
    arm["layout_row_update_s"] = update_snapshot.update_s
    arm["cache_hits"] = update_snapshot.cache_hits
    arm["cache_misses"] = update_snapshot.cache_misses
    arm["row_updates"] = update_snapshot.row_updates
    arm["parity"] = "pass"
    arm["interpretation"] = "one-row conservative cache update cost"
    result["arms"].append(arm)

    cfg_fb = Cxxcore6PilotConfig(cxxcore6_authoritative_pilot_enable=True, cxxcore6_authoritative_compare_mode="final_full")
    arm = _arm(shape_name, "fallback_python", n=n, cells=cells, ticks=ticks)
    pilot = run_authoritative_pilot(inputs, config=cfg_fb, native_backend=None, python_reference=expected, compare_mode="final_full")
    _apply_result(arm, pilot)
    arm["interpretation"] = "native unavailable fallback path"
    result["arms"].append(arm)

    if shape_name == "small":
        arm = _arm(shape_name, "authoritative_cxxcore6_injected_divergence_fallback", n=n, cells=cells, ticks=ticks)
        pilot = run_authoritative_pilot(
            inputs,
            config=cfg_final,
            native_backend=native,
            python_reference=expected,
            compare_mode="final_full",
            inject_divergence=True,
        )
        _apply_result(arm, pilot)
        arm["interpretation"] = "injected divergence proves fallback restores Python reference"
        result["arms"].append(arm)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="CXXCORE-6 guarded authoritative no-RNG pilot probe.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--skip-large", action="store_true")
    args = parser.parse_args()

    compile6 = compile_cpp_kernel(args.build_dir / "cxxcore6")
    compile5 = compile_cxxcore5_kernel(args.build_dir / "cxxcore5")
    if not compile6.get("ok") or not compile5.get("ok"):
        payload = {
            "schema_version": "cxxcore6_authoritative_pilot_probe_v1",
            "native_compile": compile6,
            "cxxcore5_compile": compile5,
            "audit": audit_payload(),
            "shapes": [],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    native = load_authoritative_backend(Path(str(compile6["library_path"])))
    cxxcore5_native = load_cxxcore5_backend(Path(str(compile5["library_path"])))
    warm, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=1, seed=6)
    native(warm, export_final=True)

    shapes = [("small", 256, 32, 32), ("medium", 1024, 64, 64)]
    if not args.skip_large:
        shapes.append(("large", 4096, 64, 64))
    runner_pilot = ("runner_captured_replay", 128, 16, 16, min(240, int(args.ticks)))
    payload = {
        "schema_version": "cxxcore6_authoritative_pilot_probe_v1",
        "native_compile": compile6,
        "cxxcore5_compile": compile5,
        "audit": audit_payload(),
        "boundary_classification": boundary_classification(),
        "rng_preflight": rng_preflight_audit(),
        "mini_live_run_smoke": {
            "implemented": False,
            "reason": "live runner hook deferred to avoid runtime backend selection and checkpoint/metrics widening; captured-action replay covered the no-RNG runner-compatible slice",
        },
        "shapes": [
            run_shape(name, n=n, width=w, height=h, ticks=int(args.ticks), native=native, cxxcore5_native=cxxcore5_native)
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
                cxxcore5_native=cxxcore5_native,
            )
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
