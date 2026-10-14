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
    source_path = repo_root / "engine" / "perf" / "native" / "cxxcore_env_apply_shadow_kernel.cpp"
    build_dir.mkdir(parents=True, exist_ok=True)
    library_path = build_dir / "libcxxcore1_env_apply_shadow.so"
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


def load_energy_delta(library_path: Path):
    lib = ctypes.CDLL(str(library_path))
    fn = lib.cxxcore1_energy_delta
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
    ]
    fn.restype = None

    def _call(pre_energy: np.ndarray, post_energy: np.ndarray) -> np.ndarray:
        pre = np.ascontiguousarray(pre_energy, dtype=np.float32)
        post = np.ascontiguousarray(post_energy, dtype=np.float32)
        if pre.shape != post.shape:
            raise ValueError("pre_energy and post_energy shape mismatch")
        out = np.zeros(pre.shape, dtype=np.float32)
        fn(
            pre.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            post.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(int(pre.size)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out

    return _call


def load_env_apply_no_rng(library_path: Path):
    lib = ctypes.CDLL(str(library_path))
    fn = lib.cxxcore1_env_apply_no_rng
    f32p = ctypes.POINTER(ctypes.c_float)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i8p = ctypes.POINTER(ctypes.c_int8)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    fn.argtypes = [
        i32p,
        i32p,
        f32p,
        f32p,
        f32p,
        f32p,
        i32p,
        i32p,
        ctypes.c_int64,
        i32p,
        i8p,
        ctypes.c_int64,
        ctypes.c_int64,
        f32p,
        f32p,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int64,
        f32p,
        f32p,
        f32p,
        f32p,
        f32p,
        u8p,
        i32p,
        i32p,
        u8p,
        f32p,
        f32p,
        f32p,
        f32p,
    ]
    fn.restype = ctypes.c_int

    def _call(*, config: Any, agent_ids: np.ndarray, actions: np.ndarray, energies: np.ndarray, healths: np.ndarray,
              injuries: np.ndarray, scars: np.ndarray, pos_y_by_id: np.ndarray, pos_x_by_id: np.ndarray,
              agent_grid: np.ndarray, grid: np.ndarray, move_cost_mults: np.ndarray,
              juvenile_harvest_mults: np.ndarray) -> dict[str, np.ndarray]:
        ids = np.ascontiguousarray(agent_ids, dtype=np.int32)
        acts = np.ascontiguousarray(actions, dtype=np.int32)
        energy = np.ascontiguousarray(energies, dtype=np.float32)
        health = np.ascontiguousarray(healths, dtype=np.float32)
        injury = np.ascontiguousarray(injuries, dtype=np.float32)
        scar = np.ascontiguousarray(scars, dtype=np.float32)
        pos_y = np.ascontiguousarray(pos_y_by_id, dtype=np.int32)
        pos_x = np.ascontiguousarray(pos_x_by_id, dtype=np.int32)
        occ = np.ascontiguousarray(agent_grid, dtype=np.int32)
        food_grid = np.ascontiguousarray(grid, dtype=np.int8)
        move_mult = np.ascontiguousarray(move_cost_mults, dtype=np.float32)
        harvest_mult = np.ascontiguousarray(juvenile_harvest_mults, dtype=np.float32)
        n = int(ids.size)
        out_energy = np.zeros((n,), dtype=np.float32)
        out_health = np.zeros((n,), dtype=np.float32)
        out_injury = np.zeros((n,), dtype=np.float32)
        out_scar = np.zeros((n,), dtype=np.float32)
        out_food = np.zeros((n,), dtype=np.float32)
        out_died = np.zeros((n,), dtype=np.uint8)
        out_pos_y = np.zeros((n,), dtype=np.int32)
        out_pos_x = np.zeros((n,), dtype=np.int32)
        out_moved = np.zeros((n,), dtype=np.uint8)
        out_e_delta = np.zeros((n,), dtype=np.float32)
        out_h_delta = np.zeros((n,), dtype=np.float32)
        out_i_delta = np.zeros((n,), dtype=np.float32)
        out_s_delta = np.zeros((n,), dtype=np.float32)
        status = fn(
            ids.ctypes.data_as(i32p),
            acts.ctypes.data_as(i32p),
            energy.ctypes.data_as(f32p),
            health.ctypes.data_as(f32p),
            injury.ctypes.data_as(f32p),
            scar.ctypes.data_as(f32p),
            pos_y.ctypes.data_as(i32p),
            pos_x.ctypes.data_as(i32p),
            ctypes.c_int64(int(pos_y.size)),
            occ.ctypes.data_as(i32p),
            food_grid.ctypes.data_as(i8p),
            ctypes.c_int64(int(config.height)),
            ctypes.c_int64(int(config.width)),
            move_mult.ctypes.data_as(f32p),
            harvest_mult.ctypes.data_as(f32p),
            ctypes.c_float(float(config.move_cost)),
            ctypes.c_float(float(config.gather_cost)),
            ctypes.c_float(float(config.rest_cost)),
            ctypes.c_float(float(config.attack_cost)),
            ctypes.c_float(float(config.action_cost_mult)),
            ctypes.c_float(float(config.rest_cost_mult)),
            ctypes.c_float(float(config.food_energy)),
            ctypes.c_float(float(config.max_energy)),
            ctypes.c_int64(n),
            out_energy.ctypes.data_as(f32p),
            out_health.ctypes.data_as(f32p),
            out_injury.ctypes.data_as(f32p),
            out_scar.ctypes.data_as(f32p),
            out_food.ctypes.data_as(f32p),
            out_died.ctypes.data_as(u8p),
            out_pos_y.ctypes.data_as(i32p),
            out_pos_x.ctypes.data_as(i32p),
            out_moved.ctypes.data_as(u8p),
            out_e_delta.ctypes.data_as(f32p),
            out_h_delta.ctypes.data_as(f32p),
            out_i_delta.ctypes.data_as(f32p),
            out_s_delta.ctypes.data_as(f32p),
        )
        if int(status) != 0:
            raise RuntimeError(f"C++ env-apply kernel failed with status {int(status)}")
        return {
            "new_energy": out_energy,
            "new_health": out_health,
            "new_injury": out_injury,
            "new_scar": out_scar,
            "food_gained": out_food,
            "died_mask": out_died.astype(bool),
            "new_pos_y": out_pos_y,
            "new_pos_x": out_pos_x,
            "moved_mask": out_moved.astype(bool),
            "energy_delta": out_e_delta,
            "health_delta": out_h_delta,
            "injury_delta": out_i_delta,
            "scar_delta": out_s_delta,
        }

    return _call


def main() -> int:
    parser = argparse.ArgumentParser(description="Optional CXXCORE-1 native compile/import smoke.")
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    info = compile_cpp_kernel(args.build_dir)
    payload: dict[str, Any] = {"compile": info}
    if info.get("ok"):
        call = load_energy_delta(Path(str(info["library_path"])))
        pre = np.asarray([1.0, 4.0, 8.5], dtype=np.float32)
        post = np.asarray([0.5, 5.0, 7.0], dtype=np.float32)
        actual = call(pre, post)
        expected = post - pre
        payload["parity"] = {
            "ok": bool(np.array_equal(actual, expected)),
            "actual": [float(v) for v in actual.tolist()],
            "expected": [float(v) for v in expected.tolist()],
        }
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root / "engine"))
        from perf.cxxcore_env_apply_shadow import (  # noqa: WPS433
            EnvApplyShadowConfig,
            compare_shadow_deltas,
            compute_env_apply_shadow_python,
        )

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
            max_energy=10.0,
        )
        agent_ids = np.asarray([7, 2, 5], dtype=np.int32)
        actions = np.asarray([4, 2, 0], dtype=np.int32)
        energies = np.asarray([5.0, 8.0, 0.75], dtype=np.float32)
        healths = np.asarray([9.0, 7.0, 4.0], dtype=np.float32)
        injuries = np.zeros((3,), dtype=np.float32)
        scars = np.zeros((3,), dtype=np.float32)
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
        move_mults = np.ones((3,), dtype=np.float32)
        harvest_mults = np.ones((3,), dtype=np.float32)
        py_expected = compute_env_apply_shadow_python(
            config=config,
            agent_ids=agent_ids,
            actions=actions,
            energies=energies,
            healths=healths,
            injuries=injuries,
            scars=scars,
            pos_y_by_id=pos_y_by_id,
            pos_x_by_id=pos_x_by_id,
            agent_grid=agent_grid,
            grid=grid,
            move_cost_mults=move_mults,
            juvenile_harvest_mults=harvest_mults,
        )
        native = load_env_apply_no_rng(Path(str(info["library_path"])))
        native_actual = native(
            config=config,
            agent_ids=agent_ids,
            actions=actions,
            energies=energies,
            healths=healths,
            injuries=injuries,
            scars=scars,
            pos_y_by_id=pos_y_by_id,
            pos_x_by_id=pos_x_by_id,
            agent_grid=agent_grid,
            grid=grid,
            move_cost_mults=move_mults,
            juvenile_harvest_mults=harvest_mults,
        )
        payload["env_apply_no_rng_parity"] = compare_shadow_deltas(py_expected, native_actual)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if (
        info.get("ok")
        and payload.get("parity", {}).get("ok")
        and payload.get("env_apply_no_rng_parity", {}).get("ok")
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
