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

from perf.cxxcore5_tick_island_layout_cache import compare_runner_backend_states, state_summary  # noqa: E402
from perf.cxxcore6_authoritative_pilot import result_payload  # noqa: E402
from perf.cxxcore7_rng_precompute import (  # noqa: E402
    Cxxcore7PilotConfig,
    apply_validation_action_override,
    build_rng_payload,
    precompute_validation_action_override_ints,
    rng_surface_audit,
    run_authoritative_precomputed_rng_pilot,
    validate_rng_payload,
)
from simulation.rng_manager import RNGManager  # noqa: E402
from tools.scripts.eiel_cxxcore5_tick_island_layout_cache_probe import (  # noqa: E402
    _capture_fixture,
    _python_authoritative,
)
from tools.scripts.eiel_cxxcore6_authoritative_pilot_probe import SUMMARY_NAMES  # noqa: E402


def compile_cpp_kernel(build_dir: Path, *, cxx: str | None = None) -> dict[str, Any]:
    compiler = cxx or shutil.which("g++")
    if not compiler:
        return {"ok": False, "reason": "g++ not found", "library_path": "", "command": []}
    source_path = ROOT / "engine" / "perf" / "native" / "cxxcore7_rng_precompute_kernel.cpp"
    build_dir.mkdir(parents=True, exist_ok=True)
    library_path = build_dir / "libcxxcore7_rng_precompute.so"
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


def load_rng_precompute_backend(library_path: Path):
    lib = ctypes.CDLL(str(library_path))
    fn = lib.cxxcore7_rng_precompute_run
    f32p = ctypes.POINTER(ctypes.c_float)
    f64p = ctypes.POINTER(ctypes.c_double)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i8p = ctypes.POINTER(ctypes.c_int8)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    fn.argtypes = [
        i32p, f32p, f32p, f32p, f32p, i32p, i32p, ctypes.c_int64,
        i32p, i8p, f32p, f32p, i32p, ctypes.c_int32, i32p, ctypes.c_int64,
        f64p, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int32, f32p, f32p, f32p, f32p, u8p, i32p, i32p, i8p, f32p, f64p, f64p,
    ]
    fn.restype = ctypes.c_int

    def _call(inputs: Any, *, rng_payload: Any | None = None, export_final: bool = True) -> dict[str, Any]:
        backend = inputs.backend
        state = backend.runner.state
        layout = backend.layout
        ids = np.ascontiguousarray(state.agent_ids, dtype=np.int32)
        rows = np.ascontiguousarray(layout.rows, dtype=np.float64)
        n = int(ids.shape[0])
        ticks = int(state.actions_by_tick.shape[0])
        cells = int(state.config.height) * int(state.config.width)
        override_enable = 0
        override = np.empty((0,), dtype=np.int32)
        if rng_payload is not None and getattr(rng_payload, "validation_action_override_ints", None) is not None:
            override_enable = 1
            override = np.ascontiguousarray(rng_payload.validation_action_override_ints.values, dtype=np.int32).reshape(-1)
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
            ctypes.c_int32(override_enable),
            override.ctypes.data_as(i32p),
            ctypes.c_int64(int(override.size)),
            rows.ctypes.data_as(f64p),
            ctypes.c_int64(int(rows.shape[1])),
            ctypes.c_int32(1 if layout.charge_channels else 0),
            ctypes.c_int32(1 if layout.charge_radius else 0),
            ctypes.c_int32(1 if layout.ledger_enabled else 0),
            ctypes.c_int64(ticks),
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
            raise RuntimeError(f"CXXCORE-7 native RNG precompute pilot failed with status {int(status)}")
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


def _arm(shape: str, name: str, *, n: int, cells: int, ticks: int, surfaces: list[str]) -> dict[str, Any]:
    return {
        "shape": shape,
        "arm": name,
        "agents": int(n),
        "cells": int(cells),
        "ticks": int(ticks),
        "actions_covered": int(n) * int(ticks),
        "stochastic_surfaces": list(surfaces),
        "wall_s": 0.0,
        "python_reference_included_slice_s": 0.0,
        "rng_precompute_capture_s": 0.0,
        "snapshot_materialization_s": 0.0,
        "native_compute_s": 0.0,
        "writeback_materialization_s": 0.0,
        "sparse_compare_s": 0.0,
        "final_compare_s": 0.0,
        "fallback_overhead_s": 0.0,
        "parity": "not_checked",
        "first_divergence": None,
        "fallback_occurred": False,
        "fallback_reason": "",
        "interpretation": "",
    }


def _apply_result(arm: dict[str, Any], result: Any) -> None:
    payload = result_payload(result)
    timing = payload["timing"]
    parity = payload.get("parity") or {}
    arm["wall_s"] = float(timing.get("total_wall_s", 0.0))
    arm["python_reference_included_slice_s"] = float(timing.get("python_reference_s", 0.0))
    arm["native_compute_s"] = float(timing.get("native_backend_s", 0.0))
    arm["writeback_materialization_s"] = float(timing.get("writeback_materialization_s", 0.0))
    arm["sparse_compare_s"] = float(timing.get("sparse_compare_s", 0.0))
    arm["final_compare_s"] = float(timing.get("final_compare_s", 0.0))
    arm["fallback_overhead_s"] = float(timing.get("fallback_s", 0.0))
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


def run_shape(shape_name: str, *, n: int, width: int, height: int, ticks: int, native: Any) -> dict[str, Any]:
    cells = width * height
    surfaces = ["validation_action_override_ints"]
    result: dict[str, Any] = {
        "shape": {"name": shape_name, "agents": n, "width": width, "height": height, "cells": cells, "ticks": ticks},
        "arms": [],
    }
    inputs, _, full_snapshot, _, hit_snapshot, _, _, _ = _capture_fixture(
        n=n,
        width=width,
        height=height,
        ticks=ticks,
        seed=71007,
    )
    rng_t0 = time.perf_counter()
    override = precompute_validation_action_override_ints(
        rng=RNGManager(global_seed=71007).env_np,
        ticks=ticks,
        agents=n,
    )
    rng_payload = build_rng_payload(validation_action_override_ints=override)
    rng_s = time.perf_counter() - rng_t0
    ref_inputs = apply_validation_action_override(inputs, rng_payload)
    t0 = time.perf_counter()
    expected, py_timing = _python_authoritative(ref_inputs.backend)
    py_wall = time.perf_counter() - t0

    arm = _arm(shape_name, "baseline_python_authoritative_rng_surface", n=n, cells=cells, ticks=ticks, surfaces=surfaces)
    arm["wall_s"] = rng_s + py_wall + full_snapshot.production_cost_s
    arm["rng_precompute_capture_s"] = rng_s
    arm["python_reference_included_slice_s"] = py_timing["py_env_s"]
    arm["snapshot_materialization_s"] = full_snapshot.production_cost_s
    arm["parity"] = "not_applicable"
    arm["interpretation"] = "Python reference using precomputed validation action override action stream"
    result["arms"].append(arm)

    cfg = Cxxcore7PilotConfig(
        cxxcore7_rng_precompute_enable=True,
        cxxcore7_validation_action_override_enable=True,
        cxxcore6_authoritative_pilot_enable=True,
        cxxcore6_authoritative_compare_mode="off",
    )
    arm = _arm(shape_name, "cxxcore7_authoritative_precomputed_rng_no_compare", n=n, cells=cells, ticks=ticks, surfaces=surfaces)
    pilot = run_authoritative_precomputed_rng_pilot(inputs, config=cfg, rng_payload=rng_payload, native_backend=native)
    _apply_result(arm, pilot)
    arm["rng_precompute_capture_s"] = rng_s
    arm["interpretation"] = "C++ consumes precomputed Python action override array; Python included slice skipped"
    result["arms"].append(arm)

    cfg_sparse = Cxxcore7PilotConfig(
        cxxcore7_rng_precompute_enable=True,
        cxxcore7_validation_action_override_enable=True,
        cxxcore6_authoritative_pilot_enable=True,
        cxxcore6_authoritative_compare_mode="sparse",
    )
    arm = _arm(shape_name, "cxxcore7_authoritative_precomputed_rng_sparse_compare", n=n, cells=cells, ticks=ticks, surfaces=surfaces)
    pilot = run_authoritative_precomputed_rng_pilot(
        inputs,
        config=cfg_sparse,
        rng_payload=rng_payload,
        native_backend=native,
        python_reference=expected,
        compare_mode="sparse",
    )
    _apply_result(arm, pilot)
    arm["rng_precompute_capture_s"] = rng_s
    arm["interpretation"] = "C++ precomputed-RNG action override plus sparse summary guard"
    result["arms"].append(arm)

    cfg_final = Cxxcore7PilotConfig(
        cxxcore7_rng_precompute_enable=True,
        cxxcore7_validation_action_override_enable=True,
        cxxcore6_authoritative_pilot_enable=True,
        cxxcore6_authoritative_compare_mode="final_full",
    )
    arm = _arm(shape_name, "cxxcore7_authoritative_precomputed_rng_final_compare", n=n, cells=cells, ticks=ticks, surfaces=surfaces)
    pilot = run_authoritative_precomputed_rng_pilot(
        inputs,
        config=cfg_final,
        rng_payload=rng_payload,
        native_backend=native,
        python_reference=expected,
        compare_mode="final_full",
    )
    _apply_result(arm, pilot)
    arm["rng_precompute_capture_s"] = rng_s
    arm["interpretation"] = "C++ precomputed-RNG action override plus final full reference compare"
    result["arms"].append(arm)

    arm = _arm(shape_name, "runner_captured_replay_precomputed_rng", n=n, cells=cells, ticks=ticks, surfaces=surfaces)
    actual = native(inputs, rng_payload=rng_payload, export_final=True)
    report = compare_runner_backend_states(expected, actual, atol=1.0e-5)
    arm["wall_s"] = float(pilot.timing["native_backend_s"]) if "pilot" in locals() else 0.0
    arm["rng_precompute_capture_s"] = rng_s
    arm["parity"] = "pass" if report["ok"] else "fail"
    arm["first_divergence"] = report.get("first_divergence")
    arm["interpretation"] = "captured runner-compatible replay with precomputed action RNG surface"
    result["arms"].append(arm)

    if shape_name == "small":
        arm = _arm(shape_name, "missing_rng_payload_fallback", n=n, cells=cells, ticks=ticks, surfaces=surfaces)
        pilot = run_authoritative_precomputed_rng_pilot(inputs, config=cfg_final, rng_payload=None, native_backend=native)
        _apply_result(arm, pilot)
        arm["interpretation"] = "missing precomputed RNG array falls back without retaining C++ result"
        result["arms"].append(arm)

        arm = _arm(shape_name, "injected_divergence_fallback", n=n, cells=cells, ticks=ticks, surfaces=surfaces)
        pilot = run_authoritative_precomputed_rng_pilot(
            inputs,
            config=cfg_final,
            rng_payload=rng_payload,
            native_backend=native,
            python_reference=expected,
            compare_mode="final_full",
            inject_divergence=True,
        )
        _apply_result(arm, pilot)
        arm["rng_precompute_capture_s"] = rng_s
        arm["interpretation"] = "injected divergence proves fallback restores Python reference"
        result["arms"].append(arm)

    result["rng_payload"] = {
        "payload_digest": rng_payload.payload_digest,
        "validation": validate_rng_payload(rng_payload, ticks=ticks, agents=n),
        "override_digest": override.digest,
    }
    result["cache"] = {
        "cache_hits": int(hit_snapshot.cache_hits),
        "cache_misses": int(hit_snapshot.cache_misses),
        "full_rebuilds": int(hit_snapshot.full_rebuilds),
    }
    result["expected_summary"] = state_summary(expected)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="CXXCORE-7 precomputed RNG runner-bridge probe.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--skip-large", action="store_true")
    args = parser.parse_args()

    compile7 = compile_cpp_kernel(args.build_dir / "cxxcore7")
    payload = {
        "schema_version": "cxxcore7_rng_precompute_runner_bridge_probe_v1",
        "git": {
            "base_commit": "32cd7d88e0c7edd2e40ce92c59e91200bf9cc4fe",
        },
        "native_compile": compile7,
        "enabled_surfaces": ["validation_action_override_ints"],
        "rng_audit": rng_surface_audit(included_surfaces=("validation_action_override_ints",)),
        "guarded_runner_bridge_probe": {
            "implemented": False,
            "reason": "live runner mutation hook deferred; runner-captured replay covers the bounded action-RNG slice without backend selection or checkpoint/metrics ownership",
        },
        "shapes": [],
    }
    if not compile7.get("ok"):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    native = load_rng_precompute_backend(Path(str(compile7["library_path"])))
    warm, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=1, seed=7)
    warm_payload = build_rng_payload(
        validation_action_override_ints=precompute_validation_action_override_ints(
            rng=RNGManager(global_seed=7).env_np,
            ticks=1,
            agents=8,
        )
    )
    native(warm, rng_payload=warm_payload, export_final=True)

    shapes = [("small", 256, 32, 32), ("medium", 1024, 64, 64)]
    if not args.skip_large:
        shapes.append(("large", 4096, 64, 64))
    shapes.append(("runner_captured_replay", 128, 16, 16))
    payload["shapes"] = [
        run_shape(
            name,
            n=n,
            width=w,
            height=h,
            ticks=min(int(args.ticks), 240) if name == "runner_captured_replay" else int(args.ticks),
            native=native,
        )
        for name, n, w, h in shapes
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
