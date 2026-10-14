from __future__ import annotations

"""RUNNERCORE-14 native C1 typed scalar projection preflight.

This module is default-off and diagnostic. It proves a compact C++ boundary for
numeric C1 typed-row projection fields without changing live simulation
semantics. Python still owns payload dictionaries, string fields, row objects,
hashes, RNG, and compatibility materialization.
"""

import ctypes
from dataclasses import dataclass
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
from typing import Any

import numpy as np


SCHEMA_VERSION = "runnercore14_native_c1_projection_v1"
BUILD_SCHEMA_VERSION = "runnercore14_native_c1_projection_build_v1"
API_SCHEMA_VERSION = "runnercore14_native_c1_projection_api_v1"

FLOAT_FIELDS = (
    "exploration_bias",
    "aggression_bias",
    "reproduction_threshold",
    "mutation_rate",
    "action_temperature",
    "drive_inertia",
    "drive_noise",
    "drive_logit_gain",
    "pred_lr",
    "drive_sp_hunger",
    "drive_sp_pain",
    "drive_sp_fatigue",
    "drive_sp_fear",
    "exp_prior_strength",
    "exp_prior_noise",
    "drive_birth_noise",
    "b3_locality_cost_total",
    "b3_locality_mean_distance",
    "b3_locality_max_distance",
)

INT_GENE_FIELDS = (
    "hidden_size",
    "use_gru",
    "memory_dim",
    "sensor_channel_mask",
    "sensor_radius",
    "introspection_channel_mask",
    "self_regulation_control_mask",
    "self_modification_target_mask",
    "directed_inheritance_channel_mask",
    "c1_operator_noop_count",
    "c2_matched_modules",
    "c2_matched_connections",
)

INT_DERIVED_FIELDS = ("modules_count", "connections_count")
INT_FIELDS = INT_GENE_FIELDS + INT_DERIVED_FIELDS


@dataclass(frozen=True)
class C1ProjectionBatch:
    schema_version: str
    float_cols: np.ndarray
    int_cols: np.ndarray
    payload_count: int
    float_fields: tuple[str, ...] = FLOAT_FIELDS
    int_fields: tuple[str, ...] = INT_FIELDS


def native_projection_requested(cfg: Any) -> bool:
    return bool(getattr(cfg, "runnercore14_native_c1_projection_enable", False)) or bool(
        getattr(cfg, "runnercore14_native_c1_projection_shadow_enable", False)
    )


def native_api_schema() -> dict[str, Any]:
    return {
        "schema_version": API_SCHEMA_VERSION,
        "kernel": "runnercore14_c1_project_rows",
        "mode": "native_numeric_c1_projection_preflight",
        "inputs": ["float_cols", "int_cols"],
        "float_fields": list(FLOAT_FIELDS),
        "int_fields": list(INT_FIELDS),
        "outputs": ["out_float_cols", "out_int_cols", "checksum"],
        "python_owned_surfaces": [
            "canonical C1 payload dictionaries",
            "string metadata fields",
            "payload hash and JSON canonicalization",
            "RunnerCoreC1TypedGenomeRow object construction",
            "policy objects and bank attach",
            "RNG and compatibility materialization",
        ],
        "normal_import_requires_native": False,
    }


def ownership_summary() -> dict[str, Any]:
    return {
        "schema_version": "runnercore14_native_c1_projection_ownership_v1",
        "native_cpp_status": "compiled optional ctypes kernel when requested",
        "live_authoritative": False,
        "default_off": True,
        "checkpoint_schema_changed": False,
        "metrics_schema_changed": False,
        "native_rng_added": False,
        "reward_changed": False,
        "selection_changed": False,
        "expected_next_boundary": (
            "Replace Python numeric field projection after C1 operator output with "
            "a batch/native row builder, while keeping payload hashes and unsupported "
            "modes fail-closed."
        ),
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return 1 if value else 0
        return int(value)
    except Exception:
        return int(default)


def build_projection_batch(payloads: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> C1ProjectionBatch:
    float_rows: list[list[float]] = []
    int_rows: list[list[int]] = []
    for payload in payloads:
        genes = payload.get("genes", {}) if isinstance(payload, dict) else {}
        if not isinstance(genes, dict):
            raise RuntimeError("RUNNERCORE14-UNSUPPORTED: C1 payload missing genes dict")
        modules = list(payload.get("modules", []) or []) if isinstance(payload, dict) else []
        connections = list(payload.get("connections", []) or []) if isinstance(payload, dict) else []
        float_rows.append(
            [
                _safe_float(payload.get(name, genes.get(name, 0.0)) if name.startswith("b3_") else genes.get(name, 0.0))
                for name in FLOAT_FIELDS
            ]
        )
        int_row = [_safe_int(genes.get(name, 0)) for name in INT_GENE_FIELDS]
        int_row.extend([len(modules), len(connections)])
        int_rows.append(int_row)
    n = len(float_rows)
    float_cols = np.ascontiguousarray(np.asarray(float_rows, dtype=np.float64).reshape((n, len(FLOAT_FIELDS))))
    int_cols = np.ascontiguousarray(np.asarray(int_rows, dtype=np.int32).reshape((n, len(INT_FIELDS))))
    return C1ProjectionBatch(
        schema_version=SCHEMA_VERSION,
        float_cols=float_cols,
        int_cols=int_cols,
        payload_count=int(n),
    )


def python_project_batch(batch: C1ProjectionBatch) -> dict[str, Any]:
    floats = np.ascontiguousarray(batch.float_cols, dtype=np.float64)
    ints = np.ascontiguousarray(batch.int_cols, dtype=np.int32)
    checksum = _checksum(floats, ints)
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "python_reference",
        "float_cols": floats,
        "int_cols": ints,
        "checksum": float(checksum),
    }


def compare_projection(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    atol: float = 0.0,
    checksum_atol: float | None = None,
) -> dict[str, Any]:
    exp_f = np.asarray(expected.get("float_cols"), dtype=np.float64)
    got_f = np.asarray(actual.get("float_cols"), dtype=np.float64)
    exp_i = np.asarray(expected.get("int_cols"), dtype=np.int32)
    got_i = np.asarray(actual.get("int_cols"), dtype=np.int32)
    if exp_f.shape != got_f.shape:
        return _shape_divergence("float_cols", exp_f, got_f)
    if exp_i.shape != got_i.shape:
        return _shape_divergence("int_cols", exp_i, got_i)
    f_bad = np.argwhere(np.abs(exp_f - got_f) > float(atol))
    if f_bad.size:
        row, col = [int(v) for v in f_bad[0].tolist()]
        return _value_divergence("float_cols", row, col, float(exp_f[row, col]), float(got_f[row, col]))
    i_bad = np.argwhere(exp_i != got_i)
    if i_bad.size:
        row, col = [int(v) for v in i_bad[0].tolist()]
        return _value_divergence("int_cols", row, col, int(exp_i[row, col]), int(got_i[row, col]))
    checksum_diff = abs(float(expected.get("checksum", 0.0)) - float(actual.get("checksum", 0.0)))
    checksum_tolerance = float(checksum_atol) if checksum_atol is not None else max(float(atol), 1.0e-4)
    return {
        "schema_version": "runnercore14_native_c1_projection_compare_v1",
        "ok": bool(checksum_diff <= checksum_tolerance),
        "classification": "native_c1_projection",
        "rows": int(exp_f.shape[0]),
        "float_cols": int(exp_f.shape[1]) if exp_f.ndim == 2 else 0,
        "int_cols": int(exp_i.shape[1]) if exp_i.ndim == 2 else 0,
        "max_abs_float_diff": float(np.max(np.abs(exp_f - got_f))) if exp_f.size else 0.0,
        "checksum_diff": float(checksum_diff),
        "checksum_tolerance": float(checksum_tolerance),
        "first_divergence": None,
    }


def compile_native_backend(build_dir: Path | str, *, cxx: str | None = None) -> dict[str, Any]:
    build = Path(build_dir)
    compiler = cxx or shutil.which("g++")
    source_path = Path(__file__).resolve().parent / "native" / "runnercore14_native_c1_projection_kernel.cpp"
    library_path = build / "librunnercore14_native_c1_projection.so"
    info: dict[str, Any] = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "ok": False,
        "backend": "ctypes_shared_library",
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
        "reason": "",
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
    return {
        "schema_version": BUILD_SCHEMA_VERSION,
        "backend": "ctypes_shared_library",
        "normal_import_requires_native": False,
        "gxx": shutil.which("g++") or "",
        "library_path": str(path) if path else "",
        "library_exists": bool(path.exists()) if path else False,
    }


def load_native_backend(library_path: str | Path):
    lib = ctypes.CDLL(str(library_path))
    fn = lib.runnercore14_c1_project_rows
    f64p = ctypes.POINTER(ctypes.c_double)
    i32p = ctypes.POINTER(ctypes.c_int32)
    fn.argtypes = [
        ctypes.c_int64,
        f64p,
        ctypes.c_int64,
        i32p,
        ctypes.c_int64,
        f64p,
        i32p,
        f64p,
    ]
    fn.restype = ctypes.c_int

    def _call(batch: C1ProjectionBatch) -> dict[str, Any]:
        float_cols = np.ascontiguousarray(batch.float_cols, dtype=np.float64)
        int_cols = np.ascontiguousarray(batch.int_cols, dtype=np.int32)
        out_float = np.zeros_like(float_cols)
        out_int = np.zeros_like(int_cols)
        checksum = ctypes.c_double(0.0)
        status = fn(
            ctypes.c_int64(int(batch.payload_count)),
            float_cols.ctypes.data_as(f64p),
            ctypes.c_int64(int(float_cols.shape[1]) if float_cols.ndim == 2 else 0),
            int_cols.ctypes.data_as(i32p),
            ctypes.c_int64(int(int_cols.shape[1]) if int_cols.ndim == 2 else 0),
            out_float.ctypes.data_as(f64p),
            out_int.ctypes.data_as(i32p),
            ctypes.byref(checksum),
        )
        if int(status) != 0:
            raise RuntimeError(f"RUNNERCORE14 native C1 projection failed with status {int(status)}")
        return {
            "schema_version": SCHEMA_VERSION,
            "backend": "native_ctypes_cpp",
            "float_cols": out_float,
            "int_cols": out_int,
            "checksum": float(checksum.value),
        }

    return _call


def _checksum(float_cols: np.ndarray, int_cols: np.ndarray) -> float:
    f_weights = np.arange(1, int(float_cols.shape[1]) + 1, dtype=np.float64).reshape((1, -1))
    i_weights = np.arange(101, 101 + int(int_cols.shape[1]), dtype=np.float64).reshape((1, -1))
    return float(
        np.sum(np.asarray(float_cols, dtype=np.float64) * f_weights, dtype=np.float64)
        + np.sum(np.asarray(int_cols, dtype=np.float64) * i_weights, dtype=np.float64)
    )


def _shape_divergence(field: str, expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    return {
        "schema_version": "runnercore14_native_c1_projection_compare_v1",
        "ok": False,
        "classification": "native_c1_projection_shape",
        "first_divergence": {
            "field": str(field),
            "python_shape": list(expected.shape),
            "native_shape": list(actual.shape),
        },
    }


def _value_divergence(field: str, row: int, col: int, expected: Any, actual: Any) -> dict[str, Any]:
    return {
        "schema_version": "runnercore14_native_c1_projection_compare_v1",
        "ok": False,
        "classification": "native_c1_projection_value",
        "first_divergence": {
            "field": str(field),
            "row": int(row),
            "column": int(col),
            "python_value": expected,
            "native_value": actual,
        },
    }
