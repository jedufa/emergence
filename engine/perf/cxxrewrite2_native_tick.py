from __future__ import annotations

"""CXXREWRITE-2 optional native tick/env-apply kernel.

The normal runner remains Python-authoritative. CXXREWRITE2 wires a guarded
native shadow comparator for the no-RNG movement/gather/rest/action-cost slice
over CXXREWRITE1 lifecycle-enabled runs. Native authoritative mutation is
intentionally unsupported in this first gate and fails closed when requested.
"""

import ctypes
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import time
from typing import Any

import numpy as np

from perf.cxxcore2_state_backend import as_backend_inputs, compare_backend_states, run_backend_python_reference
from perf.cxxcore_env_apply_shadow import config_from_env, observed_delta_from_python, supported_for_first_slice


SCHEMA_VERSION = "cxxrewrite2_native_tick_v1"
SUMMARY_SCHEMA_VERSION = "cxxrewrite2_native_tick_summary_v1"
BUILD_SCHEMA_VERSION = "cxxrewrite2_native_build_status_v1"
API_SCHEMA_VERSION = "cxxrewrite2_native_tick_api_v1"


@dataclass
class ResidentTickFrame:
    schema_version: str
    generation: int
    capacity: int
    energy_by_id: np.ndarray
    health_by_id: np.ndarray
    injury_by_id: np.ndarray
    scar_by_id: np.ndarray
    pos_y_by_id: np.ndarray
    pos_x_by_id: np.ndarray
    alive_by_id: np.ndarray
    agent_grid: np.ndarray
    grid: np.ndarray
    rebuild_count: int = 0
    missing_agent_syncs: int = 0
    pre_env_scalar_syncs: int = 0
    env_array_syncs: int = 0
    tick_updates: int = 0


def native_tick_requested(cfg: Any) -> bool:
    return bool(getattr(cfg, "cxxrewrite_native_tick_enable", False)) or bool(
        getattr(cfg, "cxxrewrite_native_tick_shadow_enable", False)
    ) or bool(getattr(cfg, "cxxrewrite_native_tick_authoritative_enable", False)) or bool(
        getattr(cfg, "cxxrewrite_native_tick_validate_shadow_enable", False)
    )


def _fail_closed(cfg: Any) -> bool:
    return bool(getattr(cfg, "cxxrewrite_native_tick_fail_closed", False)) or bool(
        getattr(cfg, "cxxrewrite_native_tick_authoritative_enable", False)
    )


def assert_native_authoritative_not_supported(cfg: Any) -> None:
    if bool(getattr(cfg, "cxxrewrite_native_tick_authoritative_enable", False)):
        raise RuntimeError(
            "CXXREWRITE2 native_tick_authoritative is not supported in this gate: "
            "the selected native slice is shadow-only because full tick semantics "
            "still include Python-owned reproduction, metrics, checkpoint, RNG, "
            "patch, injury, combat, and cost-charging surfaces."
        )


def native_api_schema() -> dict[str, Any]:
    return {
        "schema_version": API_SCHEMA_VERSION,
        "kernel": "cxxrewrite2_native_tick_run_no_rng",
        "mode": "shadow_compare_only",
        "inputs": [
            "agent_ids",
            "initial_energy",
            "initial_health",
            "initial_injury",
            "initial_scar",
            "initial_pos_y_by_id",
            "initial_pos_x_by_id",
            "initial_agent_grid",
            "initial_grid",
            "move_cost_mults",
            "juvenile_harvest_mults",
            "actions_by_tick",
            "height",
            "width",
            "move/gather/rest/attack/action/rest cost scalars",
            "food_energy",
            "max_energy",
        ],
        "outputs": [
            "new_energy",
            "new_health",
            "new_injury",
            "new_scar",
            "died_mask",
            "new_pos_y",
            "new_pos_x",
            "grid",
            "checkpoint arrays",
            "checksum",
        ],
        "included_semantics": ["movement", "gather", "rest/action costs", "death mask from energy/health"],
        "excluded_semantics": [
            "native policy",
            "native observation",
            "native RNG",
            "reproduction",
            "selection/cull/refill",
            "patch world",
            "biome hazard",
            "injury/scar cost penalties",
            "combat",
            "GRU upkeep",
            "cognitive cost charging",
            "metrics/checkpoint writer ownership",
        ],
    }


def selected_slice_audit() -> dict[str, Any]:
    return {
        "schema_version": "cxxrewrite2_selected_slice_audit_v1",
        "selected_slice": "no_rng_env_apply_movement_gather_rest_action_cost",
        "python_actions": "provided by existing runner policy/action path",
        "resident_array_contract": (
            "CXXREWRITE2 initializes a resident per-agent tick frame once per generation, "
            "uses resident scalar/env arrays as native inputs, and updates that frame from "
            "env outputs after each Python-authoritative tick. Existing agents are not "
            "repacked from Python Agent scalar fields for native shadow input each tick."
        ),
        "rng_handling": "no native RNG; selected slice skips RNG-bearing feature flags",
        "native_authoritative_supported": False,
        "authoritative_blocker": (
            "The runner still owns broader tick side effects in Python: patch emissions, "
            "hazards, injury/scar, combat, reproduction, selection/refill, metrics, "
            "checkpoints, and full Python Agent object state."
        ),
        "expected_python_work_removed_if_later_authoritative": [
            "env.step_agents_batch no-RNG movement/gather/action-cost arithmetic",
            "per-agent env position/food grid mutation for selected slice",
        ],
        "input_arrays": native_api_schema()["inputs"],
        "output_arrays": native_api_schema()["outputs"],
    }


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _copy_or_pad_i32(value: Any, size: int, *, fill: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int32).reshape(-1)
    out = np.full((int(size),), int(fill), dtype=np.int32)
    out[: min(int(size), int(arr.shape[0]))] = arr[: int(size)]
    return out


def _copy_or_pad_bool(value: Any, size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=bool).reshape(-1)
    out = np.zeros((int(size),), dtype=bool)
    out[: min(int(size), int(arr.shape[0]))] = arr[: int(size)]
    return out


def _grow_resident_frame(frame: ResidentTickFrame, new_capacity: int) -> None:
    new_cap = int(max(new_capacity, int(frame.capacity) * 2, 1))

    def _grow(arr: np.ndarray, fill: int | float | bool) -> np.ndarray:
        out = np.full((new_cap,), fill, dtype=arr.dtype)
        out[: int(arr.shape[0])] = arr
        return out

    frame.energy_by_id = _grow(frame.energy_by_id, 0.0)
    frame.health_by_id = _grow(frame.health_by_id, 0.0)
    frame.injury_by_id = _grow(frame.injury_by_id, 0.0)
    frame.scar_by_id = _grow(frame.scar_by_id, 0.0)
    frame.pos_y_by_id = _grow(frame.pos_y_by_id, -1)
    frame.pos_x_by_id = _grow(frame.pos_x_by_id, -1)
    frame.alive_by_id = _grow(frame.alive_by_id, False)
    frame.capacity = new_cap


def _sync_agents_into_resident_frame(frame: ResidentTickFrame, agents: list[Any]) -> None:
    for agent in agents:
        aid = _safe_int(getattr(agent, "agent_id", -1), -1)
        if aid < 0:
            continue
        if aid >= int(frame.capacity):
            _grow_resident_frame(frame, aid + 1)
        frame.energy_by_id[aid] = np.float32(float(getattr(agent, "energy", 0.0)))
        frame.health_by_id[aid] = np.float32(float(getattr(agent, "health", 0.0)))
        frame.injury_by_id[aid] = np.float32(float(getattr(agent, "injury", 0.0)))
        frame.scar_by_id[aid] = np.float32(float(getattr(agent, "scar", 0.0)))
        frame.alive_by_id[aid] = bool(getattr(agent, "alive", True))


def build_resident_tick_frame_from_runner(
    runner: Any,
    *,
    alive: Any,
    generation: int,
) -> ResidentTickFrame:
    agents = list(alive or [])
    env = getattr(runner, "env")
    max_aid = max([_safe_int(getattr(agent, "agent_id", -1), -1) for agent in agents] + [-1])
    pos_y_env = np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32)
    pos_x_env = np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32)
    alive_env = np.asarray(getattr(env, "agent_alive"), dtype=bool)
    capacity = max(int(max_aid) + 1, int(pos_y_env.shape[0]), int(pos_x_env.shape[0]), int(alive_env.shape[0]), 1)
    frame = ResidentTickFrame(
        schema_version=SCHEMA_VERSION,
        generation=int(generation),
        capacity=int(capacity),
        energy_by_id=np.zeros((capacity,), dtype=np.float32),
        health_by_id=np.zeros((capacity,), dtype=np.float32),
        injury_by_id=np.zeros((capacity,), dtype=np.float32),
        scar_by_id=np.zeros((capacity,), dtype=np.float32),
        pos_y_by_id=_copy_or_pad_i32(pos_y_env, capacity, fill=-1),
        pos_x_by_id=_copy_or_pad_i32(pos_x_env, capacity, fill=-1),
        alive_by_id=_copy_or_pad_bool(alive_env, capacity),
        agent_grid=np.asarray(getattr(env, "agent_grid"), dtype=np.int32).copy(),
        grid=np.asarray(getattr(env, "grid"), dtype=np.int8).copy(),
        rebuild_count=1,
    )
    _sync_agents_into_resident_frame(frame, agents)
    return frame


def ensure_resident_tick_frame(
    runner: Any,
    *,
    alive: Any,
    generation: int,
) -> ResidentTickFrame:
    frame = getattr(runner, "_cxxrewrite2_resident_tick_frame", None)
    if not isinstance(frame, ResidentTickFrame) or int(frame.generation) != int(generation):
        frame = build_resident_tick_frame_from_runner(runner, alive=alive, generation=generation)
        setattr(runner, "_cxxrewrite2_resident_tick_frame", frame)
        return frame
    agents = list(alive or [])
    max_aid = max([_safe_int(getattr(agent, "agent_id", -1), -1) for agent in agents] + [-1])
    if max_aid >= int(frame.capacity):
        _grow_resident_frame(frame, max_aid + 1)
    missing = [
        agent
        for agent in agents
        if _safe_int(getattr(agent, "agent_id", -1), -1) >= 0
        and not bool(frame.alive_by_id[_safe_int(getattr(agent, "agent_id", -1), -1)])
    ]
    if missing:
        frame.missing_agent_syncs += int(len(missing))
        _sync_agents_into_resident_frame(frame, missing)
    return frame


def commit_python_outputs_to_resident_frame(
    runner: Any,
    *,
    frame: ResidentTickFrame | None,
    agent_ids: Any,
    new_energy: Any,
    new_health: Any,
    new_injury: Any,
    new_scar: Any,
    died_mask: Any,
) -> None:
    if frame is None:
        return
    ids = np.asarray(agent_ids, dtype=np.int32)
    if ids.size and int(np.max(ids)) >= int(frame.capacity):
        _grow_resident_frame(frame, int(np.max(ids)) + 1)
    frame.energy_by_id[ids] = np.asarray(new_energy, dtype=np.float32)
    frame.health_by_id[ids] = np.asarray(new_health, dtype=np.float32)
    frame.injury_by_id[ids] = np.asarray(new_injury, dtype=np.float32)
    frame.scar_by_id[ids] = np.asarray(new_scar, dtype=np.float32)
    died = np.asarray(died_mask, dtype=bool)
    frame.alive_by_id[ids] = np.logical_not(died)
    env = getattr(runner, "env")
    pos_y = np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32)
    pos_x = np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32)
    alive_env = np.asarray(getattr(env, "agent_alive"), dtype=bool)
    cap = max(int(frame.capacity), int(pos_y.shape[0]), int(pos_x.shape[0]), int(alive_env.shape[0]))
    if cap > int(frame.capacity):
        _grow_resident_frame(frame, cap)
    frame.pos_y_by_id[: pos_y.shape[0]] = pos_y
    frame.pos_x_by_id[: pos_x.shape[0]] = pos_x
    frame.alive_by_id[: alive_env.shape[0]] = alive_env
    frame.agent_grid = np.asarray(getattr(env, "agent_grid"), dtype=np.int32).copy()
    frame.grid = np.asarray(getattr(env, "grid"), dtype=np.int8).copy()
    frame.tick_updates += 1


def sync_env_arrays_to_resident_frame(runner: Any, frame: ResidentTickFrame) -> None:
    env = getattr(runner, "env")
    pos_y = np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32)
    pos_x = np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32)
    alive_env = np.asarray(getattr(env, "agent_alive"), dtype=bool)
    cap = max(int(frame.capacity), int(pos_y.shape[0]), int(pos_x.shape[0]), int(alive_env.shape[0]))
    if cap > int(frame.capacity):
        _grow_resident_frame(frame, cap)
    frame.pos_y_by_id[: pos_y.shape[0]] = pos_y
    frame.pos_x_by_id[: pos_x.shape[0]] = pos_x
    frame.alive_by_id[: alive_env.shape[0]] = alive_env
    frame.agent_grid = np.asarray(getattr(env, "agent_grid"), dtype=np.int32).copy()
    frame.grid = np.asarray(getattr(env, "grid"), dtype=np.int8).copy()
    frame.env_array_syncs += 1


def compile_native_backend(build_dir: Path | str, *, cxx: str | None = None) -> dict[str, Any]:
    build = Path(build_dir)
    compiler = cxx or shutil.which("g++")
    source_path = Path(__file__).resolve().parent / "native" / "cxxrewrite2_native_tick_kernel.cpp"
    library_path = build / "libcxxrewrite2_native_tick.so"
    pybind11_status: dict[str, Any]
    try:
        import pybind11  # noqa: WPS433

        pybind11_status = {"available": True, "version": str(pybind11.__version__), "include": pybind11.get_include()}
    except Exception as exc:
        pybind11_status = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    info: dict[str, Any] = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "ok": False,
        "backend": "ctypes_shared_library",
        "pybind11": pybind11_status,
        "compiler": compiler or "",
        "python_version": sys.version,
        "platform": platform.platform(),
        "cc": str(sysconfig.get_config_var("CC") or ""),
        "source_path": str(source_path),
        "library_path": "",
        "command": [],
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
    }
    if not compiler:
        info["reason"] = "g++ not found"
        return info
    build.mkdir(parents=True, exist_ok=True)
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
    info.update(
        {
            "ok": cp.returncode == 0,
            "reason": "" if cp.returncode == 0 else "compile failed",
            "command": cmd,
            "returncode": int(cp.returncode),
            "stdout_tail": cp.stdout[-4000:],
            "stderr_tail": cp.stderr[-4000:],
            "library_path": str(library_path) if cp.returncode == 0 else "",
        }
    )
    return info


def native_backend_status(library_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(str(library_path)) if library_path else None
    pybind11_available = False
    pybind11_reason = ""
    try:
        import pybind11  # noqa: WPS433

        pybind11_available = True
        pybind11_reason = str(pybind11.__version__)
    except Exception as exc:
        pybind11_reason = f"{type(exc).__name__}: {exc}"
    return {
        "schema_version": BUILD_SCHEMA_VERSION,
        "backend": "ctypes_shared_library",
        "normal_import_requires_native": False,
        "pybind11_available": bool(pybind11_available),
        "pybind11_status": pybind11_reason,
        "gxx": shutil.which("g++") or "",
        "library_path": str(path) if path else "",
        "library_exists": bool(path.exists()) if path else False,
    }


def _checkpoint_count(ticks: int, checkpoint_interval: int) -> int:
    ticks = int(ticks)
    interval = int(checkpoint_interval or 0)
    if ticks <= 0 or interval <= 0:
        return 0
    return int((ticks + interval - 1) // interval)


def load_native_backend(library_path: str | Path):
    lib = ctypes.CDLL(str(library_path))
    fn = lib.cxxrewrite2_native_tick_run_no_rng
    f32p = ctypes.POINTER(ctypes.c_float)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i8p = ctypes.POINTER(ctypes.c_int8)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    f64p = ctypes.POINTER(ctypes.c_double)
    fn.argtypes = [
        i32p, f32p, f32p, f32p, f32p, i32p, i32p, ctypes.c_int64,
        i32p, i8p, f32p, f32p, i32p, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_int64, ctypes.c_int32,
        f32p, f32p, f32p, f32p, u8p, i32p, i32p, i8p,
        f32p, i32p, i32p, u8p, ctypes.POINTER(ctypes.c_int64), f64p,
    ]
    fn.restype = ctypes.c_int

    def _call(inputs: Any, *, checkpoint_interval: int = 0, export_final: bool = True) -> dict[str, Any]:
        ids = np.ascontiguousarray(inputs.agent_ids, dtype=np.int32)
        n = int(ids.shape[0])
        ticks = int(inputs.actions_by_tick.shape[0])
        cells = int(inputs.config.height) * int(inputs.config.width)
        cks = _checkpoint_count(ticks, checkpoint_interval)
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
            raise RuntimeError(f"CXXREWRITE2 native tick kernel failed with status {int(status)}")
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


def prepare_runner_shadow_snapshot(
    runner: Any,
    *,
    alive: Any,
    agent_ids_np: Any,
    actions_np: Any,
    energies: Any,
    healths: Any,
    injuries: Any,
    scars: Any,
    move_cost_mults: Any | None,
    juvenile_harvest_mults: Any | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    cfg = getattr(runner, "sim_cfg", None)
    if not native_tick_requested(cfg):
        return None, None
    assert_native_authoritative_not_supported(cfg)
    t0 = time.perf_counter()
    supported, skip_reasons = supported_for_first_slice(getattr(runner, "env_cfg", None), cfg)
    generation = int(
        runner._dev_generation_index()
        if hasattr(runner, "_dev_generation_index")
        else int(getattr(runner, "_eiel_fdp_cur_gen", getattr(runner, "_current_gen", -1)) or -1)
    )
    tick = int(
        runner._dev_tick_index()
        if hasattr(runner, "_dev_tick_index")
        else int(getattr(runner, "_eiel_fdp_cur_tick", getattr(runner, "tick", -1)) or -1)
    )
    ids = np.asarray(agent_ids_np, dtype=np.int32)
    record = {
        "schema_version": SCHEMA_VERSION,
        "generation": generation,
        "tick": tick,
        "agents": int(len(alive)),
        "cells": int(getattr(getattr(runner, "env_cfg", None), "height", 0) or 0)
        * int(getattr(getattr(runner, "env_cfg", None), "width", 0) or 0),
        "backend": "native_optional",
        "mode": "native_tick_shadow_compare"
        if bool(getattr(cfg, "cxxrewrite_native_tick_compare_enable", True))
        else "native_tick_shadow_no_compare",
        "skipped": not bool(supported),
        "skip_reasons": skip_reasons,
        "parity": "not_checked",
        "first_divergence": None,
        "timing_s": {
            "native_input_packing": 0.0,
            "native_kernel_compute": 0.0,
            "native_output_materialization": 0.0,
            "native_comparison": 0.0,
            "python_observed_materialization": 0.0,
            "total": 0.0,
        },
    }
    if not supported:
        record["timing_s"]["native_input_packing"] = float(time.perf_counter() - t0)
        if _fail_closed(cfg):
            raise RuntimeError(f"CXXREWRITE2 native tick unsupported feature shape: {skip_reasons!r}")
        return None, record
    try:
        env = getattr(runner, "env")
        frame = ensure_resident_tick_frame(runner, alive=alive, generation=generation)
        sync_env_arrays_to_resident_frame(runner, frame)
        if ids.size and int(np.max(ids)) >= int(frame.capacity):
            _grow_resident_frame(frame, int(np.max(ids)) + 1)
        resident_energy = frame.energy_by_id[ids].astype(np.float32, copy=True)
        resident_health = frame.health_by_id[ids].astype(np.float32, copy=True)
        resident_injury = frame.injury_by_id[ids].astype(np.float32, copy=True)
        resident_scar = frame.scar_by_id[ids].astype(np.float32, copy=True)
        reference_energy = np.asarray(energies, dtype=np.float32)
        reference_health = np.asarray(healths, dtype=np.float32)
        reference_injury = np.asarray(injuries, dtype=np.float32)
        reference_scar = np.asarray(scars, dtype=np.float32)
        scalar_drift = {
            "energy": float(np.max(np.abs(resident_energy - reference_energy))) if resident_energy.size else 0.0,
            "health": float(np.max(np.abs(resident_health - reference_health))) if resident_health.size else 0.0,
            "injury": float(np.max(np.abs(resident_injury - reference_injury))) if resident_injury.size else 0.0,
            "scar": float(np.max(np.abs(resident_scar - reference_scar))) if resident_scar.size else 0.0,
        }
        max_scalar_drift = max(float(v) for v in scalar_drift.values()) if scalar_drift else 0.0
        if max_scalar_drift > 1.0e-4:
            frame.energy_by_id[ids] = reference_energy
            frame.health_by_id[ids] = reference_health
            frame.injury_by_id[ids] = reference_injury
            frame.scar_by_id[ids] = reference_scar
            frame.pre_env_scalar_syncs += int(ids.shape[0])
            resident_energy = frame.energy_by_id[ids].astype(np.float32, copy=True)
            resident_health = frame.health_by_id[ids].astype(np.float32, copy=True)
            resident_injury = frame.injury_by_id[ids].astype(np.float32, copy=True)
            resident_scar = frame.scar_by_id[ids].astype(np.float32, copy=True)
        snapshot = {
            "config": config_from_env(env),
            "agent_ids": ids.copy(),
            "actions": np.asarray(actions_np, dtype=np.int32).copy(),
            "energies": resident_energy,
            "healths": resident_health,
            "injuries": resident_injury,
            "scars": resident_scar,
            "pos_y_by_id": frame.pos_y_by_id.copy(),
            "pos_x_by_id": frame.pos_x_by_id.copy(),
            "pre_pos_y": frame.pos_y_by_id[ids].copy(),
            "pre_pos_x": frame.pos_x_by_id[ids].copy(),
            "agent_grid": frame.agent_grid.copy(),
            "grid": frame.grid.copy(),
            "move_cost_mults": None if move_cost_mults is None else np.asarray(move_cost_mults, dtype=np.float32).copy(),
            "juvenile_harvest_mults": None
            if juvenile_harvest_mults is None
            else np.asarray(juvenile_harvest_mults, dtype=np.float32).copy(),
            "library_path": str(getattr(cfg, "cxxrewrite_native_tick_library_path", "") or ""),
            "resident_frame": frame,
            "resident_scalar_drift_vs_python_pre": scalar_drift,
            "resident_rebuild_count": int(frame.rebuild_count),
            "resident_missing_agent_syncs": int(frame.missing_agent_syncs),
            "resident_pre_env_scalar_syncs": int(frame.pre_env_scalar_syncs),
            "resident_env_array_syncs": int(frame.env_array_syncs),
            "resident_tick_updates": int(frame.tick_updates),
        }
        record["resident_frame_used"] = True
        record["resident_scalar_drift_vs_python_pre"] = scalar_drift
        record["resident_max_scalar_drift_vs_python_pre"] = float(max_scalar_drift)
        record["resident_rebuild_count"] = int(frame.rebuild_count)
        record["resident_missing_agent_syncs"] = int(frame.missing_agent_syncs)
        record["resident_pre_env_scalar_syncs"] = int(frame.pre_env_scalar_syncs)
        record["resident_env_array_syncs"] = int(frame.env_array_syncs)
        record["native_input_agent_object_scalar_reads"] = 0
    except Exception as exc:
        record["skipped"] = True
        record["skip_reasons"] = {"snapshot_exception": True, "detail": str(exc)}
        snapshot = None
        if _fail_closed(cfg):
            raise
    record["timing_s"]["native_input_packing"] = float(time.perf_counter() - t0)
    return snapshot, record


def run_runner_shadow_compare(
    runner: Any,
    *,
    snapshot: dict[str, Any] | None,
    record: dict[str, Any],
    new_energy: Any,
    new_health: Any,
    new_injury: Any,
    new_scar: Any,
    food_gained: Any,
    died_mask: Any,
) -> dict[str, Any]:
    cfg = getattr(runner, "sim_cfg", None)
    t_total = time.perf_counter()
    if snapshot is None:
        record["timing_s"]["total"] = float(time.perf_counter() - t_total)
        return record
    library_path = Path(str(snapshot.get("library_path", "")))
    if not library_path.exists():
        record["skipped"] = True
        record["skip_reasons"] = {"native_library_missing": True, "library_path": str(library_path)}
        if _fail_closed(cfg):
            raise RuntimeError(f"CXXREWRITE2 native library missing: {library_path}")
        record["timing_s"]["total"] = float(time.perf_counter() - t_total)
        return record

    t_pack = time.perf_counter()
    inputs = as_backend_inputs(
        config=snapshot["config"],
        agent_ids=snapshot["agent_ids"],
        initial_energy=snapshot["energies"],
        initial_health=snapshot["healths"],
        initial_injury=snapshot["injuries"],
        initial_scar=snapshot["scars"],
        pos_y_by_id=snapshot["pos_y_by_id"],
        pos_x_by_id=snapshot["pos_x_by_id"],
        agent_grid=snapshot["agent_grid"],
        grid=snapshot["grid"],
        actions_by_tick=np.asarray(snapshot["actions"], dtype=np.int32).reshape(1, -1),
        move_cost_mults=snapshot["move_cost_mults"],
        juvenile_harvest_mults=snapshot["juvenile_harvest_mults"],
    )
    record["timing_s"]["native_input_packing"] = float(
        record["timing_s"].get("native_input_packing", 0.0) + (time.perf_counter() - t_pack)
    )

    t_kernel = time.perf_counter()
    actual = load_native_backend(library_path)(inputs, checkpoint_interval=0)
    record["timing_s"]["native_kernel_compute"] = float(time.perf_counter() - t_kernel)

    compare_enabled = bool(getattr(cfg, "cxxrewrite_native_tick_compare_enable", True)) and bool(
        getattr(cfg, "cxxrewrite_native_tick_validate_shadow_enable", False)
    )
    if compare_enabled:
        t_mat = time.perf_counter()
        observed = observed_delta_from_python(
            agent_ids=snapshot["agent_ids"],
            pre_energy=snapshot["energies"],
            pre_health=snapshot["healths"],
            pre_injury=snapshot["injuries"],
            pre_scar=snapshot["scars"],
            post_energy=new_energy,
            post_health=new_health,
            post_injury=new_injury,
            post_scar=new_scar,
            food_gained=food_gained,
            died_mask=died_mask,
            pre_pos_y=snapshot["pre_pos_y"],
            pre_pos_x=snapshot["pre_pos_x"],
            post_pos_y_by_id=np.asarray(getattr(runner.env, "agent_pos_y"), dtype=np.int32),
            post_pos_x_by_id=np.asarray(getattr(runner.env, "agent_pos_x"), dtype=np.int32),
        )
        expected = {
            "new_energy": observed["new_energy"],
            "new_health": observed["new_health"],
            "new_injury": observed["new_injury"],
            "new_scar": observed["new_scar"],
            "died_mask": observed["died_mask"],
            "new_pos_y": observed["new_pos_y"],
            "new_pos_x": observed["new_pos_x"],
            "grid": np.asarray(getattr(runner.env, "grid"), dtype=np.int8).copy(),
        }
        record["timing_s"]["python_observed_materialization"] = float(time.perf_counter() - t_mat)
        t_compare = time.perf_counter()
        parity = compare_backend_states(expected, actual, atol=0.0)
        parity = _enrich_first_divergence(
            parity,
            generation=int(record.get("generation", -1)),
            tick=int(record.get("tick", -1)),
            agent_ids=np.asarray(snapshot["agent_ids"], dtype=np.int32),
        )
        record["timing_s"]["native_comparison"] = float(time.perf_counter() - t_compare)
        record["parity"] = parity
        record["first_divergence"] = parity.get("first_divergence")
        if not bool(parity.get("ok", False)) and _fail_closed(cfg):
            raise RuntimeError(f"CXXREWRITE2 native tick shadow divergence: {parity.get('first_divergence')!r}")
    else:
        record["parity"] = "not_checked"

    record["native_checksum"] = float(actual.get("checksum", 0.0))
    commit_python_outputs_to_resident_frame(
        runner,
        frame=snapshot.get("resident_frame"),
        agent_ids=snapshot["agent_ids"],
        new_energy=new_energy,
        new_health=new_health,
        new_injury=new_injury,
        new_scar=new_scar,
        died_mask=died_mask,
    )
    frame = snapshot.get("resident_frame")
    if isinstance(frame, ResidentTickFrame):
        record["resident_tick_updates"] = int(frame.tick_updates)
        record["resident_missing_agent_syncs"] = int(frame.missing_agent_syncs)
        record["resident_pre_env_scalar_syncs"] = int(frame.pre_env_scalar_syncs)
        record["resident_env_array_syncs"] = int(frame.env_array_syncs)
    record["timing_s"]["total"] = float(time.perf_counter() - t_total)
    return record


def _enrich_first_divergence(
    report: dict[str, Any],
    *,
    generation: int,
    tick: int,
    agent_ids: np.ndarray,
) -> dict[str, Any]:
    first = report.get("first_divergence")
    if not isinstance(first, dict):
        return report
    out = dict(first)
    idx = out.get("index")
    if isinstance(idx, int) and 0 <= idx < int(agent_ids.shape[0]):
        out["agent_id"] = int(agent_ids[idx])
        out["slot_id"] = int(idx)
    out["generation"] = int(generation)
    out["tick"] = int(tick)
    out.setdefault("classification", "native_compute")
    report["first_divergence"] = out
    return report


def init_native_tick_summary() -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "records": 0,
        "ticks": 0,
        "agents": 0,
        "cells": 0,
        "skipped": 0,
        "ok": True,
        "first_divergence": None,
        "parity_checked": 0,
        "native_checksums_seen": 0,
        "resident_frame_used_records": 0,
        "resident_missing_agent_syncs": 0,
        "resident_pre_env_scalar_syncs": 0,
        "resident_env_array_syncs": 0,
        "resident_rebuild_count_max": 0,
        "resident_max_scalar_drift_vs_python_pre": 0.0,
        "native_input_agent_object_scalar_reads": 0,
        "timing_s": {
            "native_input_packing": 0.0,
            "native_kernel_compute": 0.0,
            "native_output_materialization": 0.0,
            "native_comparison": 0.0,
            "python_observed_materialization": 0.0,
            "total": 0.0,
        },
        "skip_reason_counts": {},
    }


def update_native_tick_summary(summary: dict[str, Any] | None, record: dict[str, Any]) -> dict[str, Any]:
    if summary is None:
        summary = init_native_tick_summary()
    summary["records"] = int(summary.get("records", 0)) + 1
    summary["ticks"] = int(summary.get("ticks", 0)) + 1
    summary["agents"] = int(summary.get("agents", 0)) + int(record.get("agents", 0) or 0)
    summary["cells"] = int(record.get("cells", summary.get("cells", 0)) or 0)
    if record.get("skipped"):
        summary["skipped"] = int(summary.get("skipped", 0)) + 1
        counts = dict(summary.get("skip_reason_counts", {}) or {})
        for key, value in dict(record.get("skip_reasons", {}) or {}).items():
            if value:
                counts[str(key)] = int(counts.get(str(key), 0)) + 1
        summary["skip_reason_counts"] = counts
    parity = record.get("parity")
    if isinstance(parity, dict):
        summary["parity_checked"] = int(summary.get("parity_checked", 0)) + 1
        if not bool(parity.get("ok", False)):
            summary["ok"] = False
            if summary.get("first_divergence") is None:
                summary["first_divergence"] = parity.get("first_divergence")
    if "native_checksum" in record:
        summary["native_checksums_seen"] = int(summary.get("native_checksums_seen", 0)) + 1
    if bool(record.get("resident_frame_used", False)):
        summary["resident_frame_used_records"] = int(summary.get("resident_frame_used_records", 0)) + 1
    summary["resident_missing_agent_syncs"] = int(
        max(
            int(summary.get("resident_missing_agent_syncs", 0) or 0),
            int(record.get("resident_missing_agent_syncs", 0) or 0),
        )
    )
    summary["resident_pre_env_scalar_syncs"] = int(
        max(
            int(summary.get("resident_pre_env_scalar_syncs", 0) or 0),
            int(record.get("resident_pre_env_scalar_syncs", 0) or 0),
        )
    )
    summary["resident_env_array_syncs"] = int(
        max(
            int(summary.get("resident_env_array_syncs", 0) or 0),
            int(record.get("resident_env_array_syncs", 0) or 0),
        )
    )
    summary["resident_rebuild_count_max"] = int(
        max(
            int(summary.get("resident_rebuild_count_max", 0) or 0),
            int(record.get("resident_rebuild_count", 0) or 0),
        )
    )
    summary["resident_max_scalar_drift_vs_python_pre"] = float(
        max(
            float(summary.get("resident_max_scalar_drift_vs_python_pre", 0.0) or 0.0),
            float(record.get("resident_max_scalar_drift_vs_python_pre", 0.0) or 0.0),
        )
    )
    summary["native_input_agent_object_scalar_reads"] = int(
        int(summary.get("native_input_agent_object_scalar_reads", 0) or 0)
        + int(record.get("native_input_agent_object_scalar_reads", 0) or 0)
    )
    totals = dict(summary.get("timing_s", {}) or {})
    for key, value in dict(record.get("timing_s", {}) or {}).items():
        totals[str(key)] = float(totals.get(str(key), 0.0)) + float(value or 0.0)
    summary["timing_s"] = totals
    return summary


def write_native_tick_summary(run_dir: Any, summary: dict[str, Any], *, filename: str) -> Path | None:
    if not run_dir:
        return None
    out_dir = Path(str(run_dir)) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / str(filename or "cxxrewrite2_native_tick_summary.json")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(out_path)
    return out_path


def write_json(path: Path | str, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_status_payload(build_info: dict[str, Any]) -> dict[str, Any]:
    payload = dict(build_info)
    payload["api_schema"] = native_api_schema()
    return payload
