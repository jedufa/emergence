#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def compile_cpp_kernel(build_dir: Path, *, cxx: str | None = None) -> dict[str, Any]:
    compiler = cxx or shutil.which("g++")
    if not compiler:
        return {"ok": False, "reason": "g++ not found", "library_path": "", "command": []}
    repo_root = Path(__file__).resolve().parents[2]
    source_path = repo_root / "engine" / "perf" / "native" / "cxxcore2_state_backend_kernel.cpp"
    build_dir.mkdir(parents=True, exist_ok=True)
    library_path = build_dir / "libcxxcore2_state_backend.so"
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


def checkpoint_count(ticks: int, checkpoint_interval: int) -> int:
    ticks = int(ticks)
    checkpoint_interval = int(checkpoint_interval or 0)
    if ticks <= 0 or checkpoint_interval <= 0:
        return 0
    count = ticks // checkpoint_interval
    if ticks % checkpoint_interval:
        count += 1
    return int(count)


def load_backend(library_path: Path):
    lib = ctypes.CDLL(str(library_path))
    fn = lib.cxxcore2_backend_run_no_rng
    f32p = ctypes.POINTER(ctypes.c_float)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i8p = ctypes.POINTER(ctypes.c_int8)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    i64p = ctypes.POINTER(ctypes.c_int64)
    f64p = ctypes.POINTER(ctypes.c_double)
    fn.argtypes = [
        i32p, f32p, f32p, f32p, f32p, i32p, i32p, ctypes.c_int64,
        i32p, i8p, f32p, f32p, i32p, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_int64, ctypes.c_int32,
        f32p, f32p, f32p, f32p, u8p, i32p, i32p, i8p,
        f32p, i32p, i32p, u8p, i64p, f64p,
    ]
    fn.restype = ctypes.c_int

    def _call(inputs: Any, *, checkpoint_interval: int = 0, export_final: bool = True) -> dict[str, np.ndarray | float | int]:
        ids = np.ascontiguousarray(inputs.agent_ids, dtype=np.int32)
        n = int(ids.shape[0])
        ticks = int(inputs.actions_by_tick.shape[0])
        cells = int(inputs.config.height) * int(inputs.config.width)
        cks = checkpoint_count(ticks, checkpoint_interval)
        out_energy = np.zeros((n,), dtype=np.float32)
        out_health = np.zeros((n,), dtype=np.float32)
        out_injury = np.zeros((n,), dtype=np.float32)
        out_scar = np.zeros((n,), dtype=np.float32)
        out_died = np.zeros((n,), dtype=np.uint8)
        out_pos_y = np.zeros((n,), dtype=np.int32)
        out_pos_x = np.zeros((n,), dtype=np.int32)
        out_grid = np.zeros((cells,), dtype=np.int8)
        ck_energy = np.zeros((cks, n), dtype=np.float32)
        ck_pos_y = np.zeros((cks, n), dtype=np.int32)
        ck_pos_x = np.zeros((cks, n), dtype=np.int32)
        ck_died = np.zeros((cks, n), dtype=np.uint8)
        out_ck_count = ctypes.c_int64(0)
        out_checksum = ctypes.c_double(0.0)
        status = fn(
            ids.ctypes.data_as(i32p),
            np.ascontiguousarray(inputs.initial_energy, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.initial_health, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.initial_injury, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.initial_scar, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.pos_y_by_id, dtype=np.int32).ctypes.data_as(i32p),
            np.ascontiguousarray(inputs.pos_x_by_id, dtype=np.int32).ctypes.data_as(i32p),
            ctypes.c_int64(int(inputs.pos_y_by_id.shape[0])),
            np.ascontiguousarray(inputs.agent_grid, dtype=np.int32).ctypes.data_as(i32p),
            np.ascontiguousarray(inputs.grid, dtype=np.int8).ctypes.data_as(i8p),
            np.ascontiguousarray(inputs.move_cost_mults, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.juvenile_harvest_mults, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.actions_by_tick, dtype=np.int32).ctypes.data_as(i32p),
            ctypes.c_int64(ticks),
            ctypes.c_int64(n),
            ctypes.c_int64(int(inputs.config.height)),
            ctypes.c_int64(int(inputs.config.width)),
            ctypes.c_float(float(inputs.config.move_cost)),
            ctypes.c_float(float(inputs.config.gather_cost)),
            ctypes.c_float(float(inputs.config.rest_cost)),
            ctypes.c_float(float(inputs.config.attack_cost)),
            ctypes.c_float(float(inputs.config.action_cost_mult)),
            ctypes.c_float(float(inputs.config.rest_cost_mult)),
            ctypes.c_float(float(inputs.config.food_energy)),
            ctypes.c_float(float(inputs.config.max_energy)),
            ctypes.c_int64(int(checkpoint_interval or 0)),
            ctypes.c_int32(1 if export_final else 0),
            out_energy.ctypes.data_as(f32p),
            out_health.ctypes.data_as(f32p),
            out_injury.ctypes.data_as(f32p),
            out_scar.ctypes.data_as(f32p),
            out_died.ctypes.data_as(u8p),
            out_pos_y.ctypes.data_as(i32p),
            out_pos_x.ctypes.data_as(i32p),
            out_grid.ctypes.data_as(i8p),
            ck_energy.ctypes.data_as(f32p),
            ck_pos_y.ctypes.data_as(i32p),
            ck_pos_x.ctypes.data_as(i32p),
            ck_died.ctypes.data_as(u8p),
            ctypes.byref(out_ck_count),
            ctypes.byref(out_checksum),
        )
        if int(status) != 0:
            raise RuntimeError(f"CXXCORE-2 native backend failed with status {int(status)}")
        return {
            "new_energy": out_energy,
            "new_health": out_health,
            "new_injury": out_injury,
            "new_scar": out_scar,
            "died_mask": out_died.astype(bool),
            "new_pos_y": out_pos_y,
            "new_pos_x": out_pos_x,
            "grid": out_grid.reshape((int(inputs.config.height), int(inputs.config.width))),
            "checkpoint_energy": ck_energy,
            "checkpoint_pos_y": ck_pos_y,
            "checkpoint_pos_x": ck_pos_x,
            "checkpoint_died": ck_died.astype(bool),
            "checkpoint_count": int(out_ck_count.value),
            "checksum": float(out_checksum.value),
        }

    return _call


def _fixture():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "engine"))
    from perf.cxxcore_env_apply_shadow import EnvApplyShadowConfig  # noqa: WPS433
    from perf.cxxcore2_state_backend import as_backend_inputs  # noqa: WPS433

    config = EnvApplyShadowConfig(
        height=4,
        width=5,
        move_cost=1.0,
        gather_cost=0.5,
        rest_cost=0.1,
        attack_cost=2.0,
        action_cost_mult=1.0,
        rest_cost_mult=1.0,
        food_energy=3.0,
        max_energy=20.0,
    )
    agent_ids = np.asarray([7, 2, 5], dtype=np.int32)
    pos_y_by_id = np.full((8,), -1, dtype=np.int32)
    pos_x_by_id = np.full((8,), -1, dtype=np.int32)
    pos_y_by_id[[7, 2, 5]] = np.asarray([1, 1, 2], dtype=np.int32)
    pos_x_by_id[[7, 2, 5]] = np.asarray([1, 2, 0], dtype=np.int32)
    agent_grid = np.full((4, 5), -1, dtype=np.int32)
    agent_grid[1, 1] = 7
    agent_grid[1, 2] = 2
    agent_grid[2, 0] = 5
    grid = np.zeros((4, 5), dtype=np.int8)
    grid[1, 1] = 1
    actions = np.asarray([[4, 2, 0], [5, 5, 5], [3, 4, 1]], dtype=np.int32)
    return as_backend_inputs(
        config=config,
        agent_ids=agent_ids,
        initial_energy=np.asarray([10.0, 9.0, 8.0], dtype=np.float32),
        initial_health=np.asarray([9.0, 7.0, 4.0], dtype=np.float32),
        initial_injury=np.zeros((3,), dtype=np.float32),
        initial_scar=np.zeros((3,), dtype=np.float32),
        pos_y_by_id=pos_y_by_id,
        pos_x_by_id=pos_x_by_id,
        agent_grid=agent_grid,
        grid=grid,
        actions_by_tick=actions,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Optional CXXCORE-2 persistent state-backend native smoke.")
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    info = compile_cpp_kernel(args.build_dir)
    payload: dict[str, Any] = {"compile": info}
    if info.get("ok"):
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root / "engine"))
        from perf.cxxcore2_state_backend import compare_backend_states, run_backend_python_reference  # noqa: WPS433

        inputs = _fixture()
        expected = run_backend_python_reference(inputs, checkpoint_interval=2)
        actual = load_backend(Path(str(info["library_path"])))(inputs, checkpoint_interval=2)
        payload["parity"] = compare_backend_states(expected, actual)
        payload["checkpoint_count"] = int(actual["checkpoint_count"])
        payload["checksum"] = float(actual["checksum"])
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if info.get("ok") and payload.get("parity", {}).get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
