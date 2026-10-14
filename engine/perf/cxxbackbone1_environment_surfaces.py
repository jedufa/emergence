from __future__ import annotations

"""CXXBACKBONE-1 Patch/WDL/Jungle native array-surface contract."""

from dataclasses import dataclass
import ctypes
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "cxxbackbone1_environment_surfaces_v1"
SUMMARY_NAMES = ("patch_food_sum", "patch_visibility_sum", "jungle_damage_pressure_sum", "wdl_active_sum")


@dataclass(frozen=True)
class EnvironmentSurfacesInputV1:
    patch_food: np.ndarray
    patch_fertility: np.ndarray
    patch_cover: np.ndarray
    patch_shelter: np.ndarray
    jungle_predator_risk: np.ndarray
    jungle_terrain_protection: np.ndarray
    wdl_cue: np.ndarray
    wdl_active: np.ndarray
    wdl_timer: np.ndarray
    occupant_density: np.ndarray
    patch_regrow_rate: float
    patch_food_cap: float
    jungle_risk_decay: float
    wdl_cue_to_active_threshold: float
    wdl_delay_ticks: int


@dataclass(frozen=True)
class EnvironmentSurfacesResultV1:
    patch_food: np.ndarray
    patch_visibility: np.ndarray
    jungle_predator_risk: np.ndarray
    jungle_damage_pressure: np.ndarray
    wdl_cue: np.ndarray
    wdl_active: np.ndarray
    wdl_timer: np.ndarray
    summary: Mapping[str, float]
    timing: Mapping[str, float]
    executed_native: bool
    hidden_fallback: bool


def make_environment_surfaces_fixture(*, cells: int = 256, seed: int = 510) -> EnvironmentSurfacesInputV1:
    rng = np.random.default_rng(int(seed))
    return EnvironmentSurfacesInputV1(
        patch_food=rng.random((cells,), dtype=np.float32),
        patch_fertility=rng.random((cells,), dtype=np.float32),
        patch_cover=rng.random((cells,), dtype=np.float32) * np.float32(0.6),
        patch_shelter=rng.random((cells,), dtype=np.float32) * np.float32(0.5),
        jungle_predator_risk=rng.random((cells,), dtype=np.float32) * np.float32(0.4),
        jungle_terrain_protection=rng.random((cells,), dtype=np.float32) * np.float32(0.8),
        wdl_cue=rng.random((cells,), dtype=np.float32),
        wdl_active=np.zeros((cells,), dtype=np.float32),
        wdl_timer=np.asarray([-1 if i % 5 else 1 for i in range(cells)], dtype=np.int32),
        occupant_density=rng.random((cells,), dtype=np.float32) * np.float32(0.7),
        patch_regrow_rate=0.08,
        patch_food_cap=1.5,
        jungle_risk_decay=0.92,
        wdl_cue_to_active_threshold=0.75,
        wdl_delay_ticks=3,
    )


def run_environment_surfaces_python_reference(inputs: EnvironmentSurfacesInputV1) -> EnvironmentSurfacesResultV1:
    t0 = time.perf_counter()
    food = np.maximum(
        np.float32(0.0),
        np.minimum(
            np.float32(inputs.patch_food_cap),
            inputs.patch_food + np.float32(inputs.patch_regrow_rate) * inputs.patch_fertility * (np.float32(1.0) - inputs.occupant_density),
        ),
    ).astype(np.float32, copy=False)
    visibility = np.maximum(
        np.float32(0.0),
        np.minimum(np.float32(1.0), np.float32(1.0) - np.float32(0.5) * inputs.patch_cover - np.float32(0.25) * inputs.patch_shelter),
    ).astype(np.float32, copy=False)
    risk = np.maximum(np.float32(0.0), inputs.jungle_predator_risk * np.float32(inputs.jungle_risk_decay) + inputs.occupant_density * np.float32(0.05)).astype(np.float32, copy=False)
    pressure = np.maximum(np.float32(0.0), risk * (np.float32(1.0) - inputs.jungle_terrain_protection)).astype(np.float32, copy=False)
    cue = inputs.wdl_cue.astype(np.float32, copy=True)
    active = inputs.wdl_active.astype(np.float32, copy=True)
    timer = inputs.wdl_timer.astype(np.int32, copy=True)
    start = (cue >= np.float32(inputs.wdl_cue_to_active_threshold)) & (active <= np.float32(0.0)) & (timer < 0)
    dec = timer > 0
    activate = timer == 0
    timer[start] = np.int32(inputs.wdl_delay_ticks)
    timer[dec] = timer[dec] - np.int32(1)
    active[activate] = np.float32(1.0)
    timer[activate] = np.int32(-1)
    fade = (active > np.float32(0.0)) & (cue < np.float32(0.01))
    active[fade] = np.maximum(np.float32(0.0), active[fade] - np.float32(0.1))
    cue = np.maximum(np.float32(0.0), cue * np.float32(0.95)).astype(np.float32, copy=False)
    return EnvironmentSurfacesResultV1(
        patch_food=np.ascontiguousarray(food),
        patch_visibility=np.ascontiguousarray(visibility),
        jungle_predator_risk=np.ascontiguousarray(risk),
        jungle_damage_pressure=np.ascontiguousarray(pressure),
        wdl_cue=np.ascontiguousarray(cue),
        wdl_active=np.ascontiguousarray(active),
        wdl_timer=np.ascontiguousarray(timer),
        summary=_summary(food, visibility, pressure, active),
        timing={"wall_s": float(time.perf_counter() - t0)},
        executed_native=False,
        hidden_fallback=False,
    )


def compile_environment_surfaces_kernel(build_dir: Path | str, *, cxx: str | None = None) -> dict[str, Any]:
    compiler = cxx or shutil.which("g++")
    build = Path(build_dir)
    source_path = ROOT / "engine" / "perf" / "native" / "cxxbackbone1_environment_surfaces_kernel.cpp"
    library_path = build / "libcxxbackbone1_environment_surfaces.so"
    if not compiler:
        return {"ok": False, "reason": "g++ not found", "library_path": "", "command": [], "source_path": str(source_path)}
    build.mkdir(parents=True, exist_ok=True)
    cmd = [str(compiler), "-O3", "-std=c++17", "-shared", "-fPIC", str(source_path), "-o", str(library_path)]
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


def load_environment_surfaces_backend(library_path: Path | str):
    lib = ctypes.CDLL(str(library_path))
    fn = lib.cxxbackbone1_environment_surfaces_v1
    f32p = ctypes.POINTER(ctypes.c_float)
    i32p = ctypes.POINTER(ctypes.c_int32)
    f64p = ctypes.POINTER(ctypes.c_double)
    fn.argtypes = [
        f32p, f32p, f32p, f32p, f32p, f32p, f32p, f32p, i32p, f32p, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int32,
        f32p, f32p, f32p, f32p, f32p, f32p, i32p, f64p,
    ]
    fn.restype = ctypes.c_int

    def _call(inputs: EnvironmentSurfacesInputV1) -> EnvironmentSurfacesResultV1:
        prep_t0 = time.perf_counter()
        cells = int(inputs.patch_food.shape[0])
        out_food = np.zeros((cells,), dtype=np.float32)
        out_visibility = np.zeros((cells,), dtype=np.float32)
        out_risk = np.zeros((cells,), dtype=np.float32)
        out_pressure = np.zeros((cells,), dtype=np.float32)
        out_cue = np.zeros((cells,), dtype=np.float32)
        out_active = np.zeros((cells,), dtype=np.float32)
        out_timer = np.zeros((cells,), dtype=np.int32)
        out_summary = np.zeros((len(SUMMARY_NAMES),), dtype=np.float64)
        prep_s = time.perf_counter() - prep_t0
        compute_t0 = time.perf_counter()
        status = fn(
            np.ascontiguousarray(inputs.patch_food, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.patch_fertility, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.patch_cover, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.patch_shelter, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.jungle_predator_risk, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.jungle_terrain_protection, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.wdl_cue, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.wdl_active, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.wdl_timer, dtype=np.int32).ctypes.data_as(i32p),
            np.ascontiguousarray(inputs.occupant_density, dtype=np.float32).ctypes.data_as(f32p),
            ctypes.c_int64(cells),
            ctypes.c_float(float(inputs.patch_regrow_rate)),
            ctypes.c_float(float(inputs.patch_food_cap)),
            ctypes.c_float(float(inputs.jungle_risk_decay)),
            ctypes.c_float(float(inputs.wdl_cue_to_active_threshold)),
            ctypes.c_int32(int(inputs.wdl_delay_ticks)),
            out_food.ctypes.data_as(f32p),
            out_visibility.ctypes.data_as(f32p),
            out_risk.ctypes.data_as(f32p),
            out_pressure.ctypes.data_as(f32p),
            out_cue.ctypes.data_as(f32p),
            out_active.ctypes.data_as(f32p),
            out_timer.ctypes.data_as(i32p),
            out_summary.ctypes.data_as(f64p),
        )
        compute_s = time.perf_counter() - compute_t0
        if int(status) != 0:
            raise RuntimeError(f"CXXBACKBONE1 environment surfaces kernel failed with status {int(status)}")
        return EnvironmentSurfacesResultV1(
            patch_food=out_food,
            patch_visibility=out_visibility,
            jungle_predator_risk=out_risk,
            jungle_damage_pressure=out_pressure,
            wdl_cue=out_cue,
            wdl_active=out_active,
            wdl_timer=out_timer,
            summary={name: float(out_summary[i]) for i, name in enumerate(SUMMARY_NAMES)},
            timing={"setup_s": float(prep_s), "backend_compute_s": float(compute_s), "wall_s": float(prep_s + compute_s)},
            executed_native=True,
            hidden_fallback=False,
        )

    return _call


def compare_environment_surfaces_results(expected: EnvironmentSurfacesResultV1, observed: EnvironmentSurfacesResultV1, *, atol: float = 0.0) -> dict[str, Any]:
    for name in ("patch_food", "patch_visibility", "jungle_predator_risk", "jungle_damage_pressure", "wdl_cue", "wdl_active", "wdl_timer"):
        a = np.asarray(getattr(expected, name))
        b = np.asarray(getattr(observed, name))
        if a.shape != b.shape:
            return {"ok": False, "first_divergence": {"field": name, "expected_shape": list(a.shape), "observed_shape": list(b.shape)}}
        if a.dtype.kind == "f":
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            if bool(np.any(diff > float(atol))):
                idx = tuple(int(x) for x in np.argwhere(diff > float(atol))[0])
                return {"ok": False, "first_divergence": {"field": name, "index": list(idx), "expected": float(a[idx]), "observed": float(b[idx]), "abs_delta": float(diff[idx])}}
        elif not bool(np.array_equal(a, b)):
            idx = tuple(int(x) for x in np.argwhere(a != b)[0])
            return {"ok": False, "first_divergence": {"field": name, "index": list(idx), "expected": int(a[idx]), "observed": int(b[idx])}}
    for name in SUMMARY_NAMES:
        delta = abs(float(expected.summary[name]) - float(observed.summary[name]))
        if delta > float(atol):
            return {"ok": False, "first_divergence": {"field": f"summary.{name}", "expected": float(expected.summary[name]), "observed": float(observed.summary[name]), "abs_delta": delta}}
    return {"ok": True, "first_divergence": None, "state_digest": result_digest(observed)}


def authority_surfaces_payload(parity: Mapping[str, Any]) -> dict[str, Any]:
    exact = bool(parity.get("ok"))
    return {
        "schema_version": SCHEMA_VERSION,
        "native_authority_surfaces": {
            "patch_world_state": exact,
            "jungle_state": exact,
            "wdl_state": exact,
        },
        "unit_parity": "exact" if exact else "blocked",
        "first_divergence": parity.get("first_divergence"),
        "live_integrated": False,
        "checkpoint_integrated": False,
    }


def result_digest(result: EnvironmentSurfacesResultV1) -> str:
    return _json_digest(
        {
            "patch_food": _array_digest(result.patch_food),
            "patch_visibility": _array_digest(result.patch_visibility),
            "jungle_predator_risk": _array_digest(result.jungle_predator_risk),
            "jungle_damage_pressure": _array_digest(result.jungle_damage_pressure),
            "wdl_cue": _array_digest(result.wdl_cue),
            "wdl_active": _array_digest(result.wdl_active),
            "wdl_timer": _array_digest(result.wdl_timer),
            "summary": dict(result.summary),
        }
    )


def _summary(food: np.ndarray, visibility: np.ndarray, pressure: np.ndarray, active: np.ndarray) -> dict[str, float]:
    return {
        "patch_food_sum": float(np.sum(food.astype(np.float64), dtype=np.float64)),
        "patch_visibility_sum": float(np.sum(visibility.astype(np.float64), dtype=np.float64)),
        "jungle_damage_pressure_sum": float(np.sum(pressure.astype(np.float64), dtype=np.float64)),
        "wdl_active_sum": float(np.sum(active.astype(np.float64), dtype=np.float64)),
    }


def _array_digest(value: Any) -> str:
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _json_digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


__all__ = [
    "SCHEMA_VERSION",
    "EnvironmentSurfacesInputV1",
    "EnvironmentSurfacesResultV1",
    "authority_surfaces_payload",
    "compare_environment_surfaces_results",
    "compile_environment_surfaces_kernel",
    "load_environment_surfaces_backend",
    "make_environment_surfaces_fixture",
    "result_digest",
    "run_environment_surfaces_python_reference",
]
