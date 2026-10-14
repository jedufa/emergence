from __future__ import annotations

"""CXXREWRITE-4 resident upstream policy/observation scalar ownership.

This layer moves a bounded part of policy input materialization and post-env
scalar write-through onto the CXXREWRITE3 resident frame. It is default-off and
keeps Python policy/network execution authoritative; the native helper packs
resident policy heads for parity/timing evidence.
"""

import ctypes
from pathlib import Path
import json
import platform
import shutil
import subprocess
import sys
import sysconfig
import time
from typing import Any

import numpy as np

from perf.cxxrewrite3_resident_frame import (
    CxxRewrite3ResidentFrame,
    _generation_tick,
    _grow_frame,
    ensure_resident_frame,
    frame_digest,
)


SCHEMA_VERSION = "cxxrewrite4_upstream_resident_v1"
SUMMARY_SCHEMA_VERSION = "cxxrewrite4_upstream_summary_v1"
BUILD_SCHEMA_VERSION = "cxxrewrite4_native_pack_build_status_v1"
API_SCHEMA_VERSION = "cxxrewrite4_native_policy_pack_api_v1"

PACK_COLUMNS = [
    "energy_norm",
    "health_norm",
    "injury",
    "scar",
    "pos_y",
    "pos_x",
    "drive_hunger",
    "drive_pain",
    "drive_fatigue",
    "drive_fear",
    "action",
]
PACK_WEIGHTS = np.asarray([1.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0, 31.0], dtype=np.float64)


def resident_observation_requested(cfg: Any) -> bool:
    return bool(getattr(cfg, "cxxrewrite_resident_observation_enable", False)) or bool(
        getattr(cfg, "cxxrewrite_resident_observation_shadow_enable", False)
    ) or bool(getattr(cfg, "cxxrewrite_resident_observation_authoritative_enable", False)) or bool(
        getattr(cfg, "cxxrewrite_resident_observation_native_pack_enable", False)
    )


def resident_policyframe_requested(cfg: Any) -> bool:
    return bool(getattr(cfg, "cxxrewrite_resident_policyframe_enable", False)) or bool(
        getattr(cfg, "cxxrewrite_resident_policyframe_shadow_enable", False)
    ) or bool(getattr(cfg, "cxxrewrite_resident_policyframe_authoritative_enable", False)) or bool(
        getattr(cfg, "cxxrewrite_resident_policyframe_native_pack_enable", False)
    )


def resident_post_env_scalar_requested(cfg: Any) -> bool:
    return bool(getattr(cfg, "cxxrewrite_resident_post_env_scalar_enable", False)) or bool(
        getattr(cfg, "cxxrewrite_resident_post_env_scalar_shadow_enable", False)
    ) or bool(getattr(cfg, "cxxrewrite_resident_post_env_scalar_authoritative_enable", False)) or bool(
        getattr(cfg, "cxxrewrite_resident_generation_scalar_env_inputs_enable", False)
    )


def resident_generation_scalar_env_inputs_requested(cfg: Any) -> bool:
    return bool(getattr(cfg, "cxxrewrite_resident_generation_scalar_env_inputs_enable", False))


def resident_scalar_sync_fail_closed(cfg: Any) -> bool:
    return bool(getattr(cfg, "cxxrewrite_resident_post_env_scalar_fail_closed", False)) or bool(
        resident_generation_scalar_env_inputs_requested(cfg)
        and getattr(cfg, "cxxrewrite_resident_generation_runner_fail_closed", False)
    )


def native_pack_requested(cfg: Any) -> bool:
    return bool(getattr(cfg, "cxxrewrite_resident_observation_native_pack_enable", False)) or bool(
        getattr(cfg, "cxxrewrite_resident_policyframe_native_pack_enable", False)
    )


def upstream_requested(cfg: Any) -> bool:
    return bool(
        resident_observation_requested(cfg)
        or resident_policyframe_requested(cfg)
        or resident_post_env_scalar_requested(cfg)
        or native_pack_requested(cfg)
        or bool(getattr(cfg, "cxxrewrite_resident_upstream_validate_shadow_enable", False))
        or bool(getattr(cfg, "cxxrewrite_resident_upstream_forbid_hotpath_agent_sync", False))
    )


def _shadow_compare_requested(cfg: Any) -> bool:
    return bool(
        getattr(cfg, "cxxrewrite_resident_observation_shadow_enable", False)
        or getattr(cfg, "cxxrewrite_resident_policyframe_shadow_enable", False)
        or getattr(cfg, "cxxrewrite_resident_post_env_scalar_shadow_enable", False)
        or getattr(cfg, "cxxrewrite_resident_upstream_validate_shadow_enable", False)
    )


def _fail_closed(cfg: Any) -> bool:
    return bool(
        resident_scalar_sync_fail_closed(cfg)
        or getattr(cfg, "cxxrewrite_resident_observation_authoritative_enable", False)
        or getattr(cfg, "cxxrewrite_resident_policyframe_authoritative_enable", False)
        or getattr(cfg, "cxxrewrite_resident_post_env_scalar_authoritative_enable", False)
    )


def assert_authoritative_not_supported(cfg: Any) -> None:
    if bool(
        getattr(cfg, "cxxrewrite_resident_observation_authoritative_enable", False)
        or getattr(cfg, "cxxrewrite_resident_policyframe_authoritative_enable", False)
        or getattr(cfg, "cxxrewrite_resident_post_env_scalar_authoritative_enable", False)
    ):
        raise RuntimeError(
            "CXXREWRITE4 resident upstream authoritative mode is not supported in this gate: "
            "policy network execution, reproduction, metrics, checkpointing, and several "
            "non-env mutation paths remain Python-authoritative."
        )


def native_api_schema() -> dict[str, Any]:
    return {
        "schema_version": API_SCHEMA_VERSION,
        "kernel": "cxxrewrite4_pack_policy_heads",
        "mode": "resident_upstream_policy_head_packing",
        "inputs": [
            "agent_ids",
            "energy_by_id",
            "health_by_id",
            "injury_by_id",
            "scar_by_id",
            "pos_y_by_id",
            "pos_x_by_id",
            "action_by_id",
            "agent_id_to_slot",
            "drive arrays by slot",
            "energy/health normalization denominators",
        ],
        "outputs": ["packed Nx11 float32 policy head", "checksum"],
        "packed_columns": list(PACK_COLUMNS),
        "normal_import_requires_native": False,
    }


def resident_upstream_schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "resident_source": "CXXREWRITE3 resident frame plus runner drive arrays by slot",
        "policy_heads": ["energy_norm", "health_norm", "injury", "scar", "position", "drives", "last_action"],
        "post_env_write_through": [
            "env_apply arrays",
            "maintenance energy",
            "late prediction energy tax",
            "rest energy/health/injury/death",
            "delayed hazard / predator death sync",
        ],
        "native_pack_columns": list(PACK_COLUMNS),
        "authoritative_status": "fail_closed_in_cxxrewrite4",
    }


def blocker_map() -> dict[str, Any]:
    return {
        "schema_version": "cxxrewrite4_blocker_map_v1",
        "bounded_owner_moved": [
            "policy energy/health observation heads read from resident arrays",
            "drive policy heads read from runner slot arrays with resident slot mapping",
            "post-env scalar mutations mirror back into resident frame",
            "native policy-head pack consumes resident arrays directly",
        ],
        "remaining_python_owners": [
            "local env observation patch materialization",
            "genome/developmental expression policy scalars",
            "torch policy network execution and sampling",
            "birth/reproduction, metrics, checkpoint object graph",
        ],
        "authoritative_gate": "not safe until remaining Python owners are either resident-array backed or isolated behind explicit boundary contracts",
    }


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _generation_tick_safe(runner: Any) -> tuple[int, int]:
    try:
        return _generation_tick(runner)
    except Exception:
        return (
            _safe_int(getattr(runner, "_current_gen", getattr(runner, "gen", -1)), -1),
            _safe_int(getattr(runner, "tick", getattr(runner, "_pred_tick", -1)), -1),
        )


def _ensure_frame(runner: Any, alive: Any, *, generation: int | None = None, tick: int | None = None) -> CxxRewrite3ResidentFrame:
    gen, tk = _generation_tick_safe(runner)
    if generation is not None:
        gen = int(generation)
    if tick is not None:
        tk = int(tick)
    frame = ensure_resident_frame(runner, alive=alive, generation=gen, tick=tk)
    return frame


def _agent_ids_from_agents(agents: Any) -> np.ndarray:
    return np.asarray([_safe_int(getattr(agent, "agent_id", -1), -1) for agent in list(agents or [])], dtype=np.int32)


def _drive_arrays(runner: Any) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    try:
        if hasattr(runner, "_ensure_drive_arrays"):
            runner._ensure_drive_arrays()
    except Exception:
        pass
    h = getattr(runner, "_drive_hunger", None)
    p = getattr(runner, "_drive_pain", None)
    f = getattr(runner, "_drive_fatigue", None)
    fr = getattr(runner, "_drive_fear", None)
    if h is None or p is None or f is None or fr is None:
        return None, None, None, None
    return (
        np.asarray(h, dtype=np.float32),
        np.asarray(p, dtype=np.float32),
        np.asarray(f, dtype=np.float32),
        np.asarray(fr, dtype=np.float32),
    )


def _pack_drives_from_slots(runner: Any, frame: CxxRewrite3ResidentFrame, ids: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    h, p, f, fr = _drive_arrays(runner)
    if h is None or p is None or f is None or fr is None:
        return None, None
    slots = np.asarray(frame.agent_id_to_slot[ids], dtype=np.int32)
    valid = (slots >= 0) & (slots < int(h.shape[0])) & (slots < int(p.shape[0])) & (slots < int(f.shape[0])) & (slots < int(fr.shape[0]))
    if not bool(np.all(valid)):
        return None, slots
    drives = np.zeros((int(ids.shape[0]), 4), dtype=np.float32)
    drives[:, 0] = h[slots]
    drives[:, 1] = p[slots]
    drives[:, 2] = f[slots]
    drives[:, 3] = fr[slots]
    return drives, slots


def _sync_drive_slots_when_needed(runner: Any, agents: list[Any]) -> int:
    if not hasattr(runner, "_sync_drive_slot"):
        return 0
    drive_tokens = getattr(runner, "_drive_slot_token", None)
    syncs = 0
    for agent in agents:
        try:
            slot = int(getattr(agent, "slot", -1))
            token = int(getattr(agent, "slot_token", 0))
        except Exception:
            continue
        if slot < 0:
            continue
        needs_sync = True
        if drive_tokens is not None:
            try:
                needs_sync = int(drive_tokens[slot]) != int(token)
            except Exception:
                needs_sync = True
        if needs_sync:
            try:
                runner._sync_drive_slot(agent)
                syncs += 1
            except Exception:
                pass
    return int(syncs)


def python_pack_policy_heads(
    runner: Any,
    frame: CxxRewrite3ResidentFrame,
    agent_ids: Any,
    *,
    energy_norm_denom: float,
    health_norm_denom: float,
) -> dict[str, Any]:
    ids = np.asarray(agent_ids, dtype=np.int32).reshape(-1)
    e_den = float(energy_norm_denom) or 1.0
    h_den = float(health_norm_denom) or 1.0
    drives, _slots = _pack_drives_from_slots(runner, frame, ids)
    if drives is None:
        drives = np.zeros((int(ids.shape[0]), 4), dtype=np.float32)
    packed = np.column_stack(
        [
            frame.energy_by_id[ids] / np.float32(e_den),
            frame.health_by_id[ids] / np.float32(h_den),
            frame.injury_by_id[ids],
            frame.scar_by_id[ids],
            frame.pos_y_by_id[ids].astype(np.float32),
            frame.pos_x_by_id[ids].astype(np.float32),
            drives[:, 0],
            drives[:, 1],
            drives[:, 2],
            drives[:, 3],
            frame.action_by_id[ids].astype(np.float32),
        ]
    ).astype(np.float32, copy=False)
    return {"packed": packed, "checksum": float(np.sum(packed.astype(np.float64) * PACK_WEIGHTS.reshape((1, -1))))}


def compare_packed(expected: dict[str, Any], actual: dict[str, Any], *, atol: float = 0.0) -> dict[str, Any]:
    exp = np.asarray(expected.get("packed"), dtype=np.float32)
    got = np.asarray(actual.get("packed"), dtype=np.float32)
    if exp.shape != got.shape:
        return {
            "ok": False,
            "classification": "cxxrewrite4_native_policy_pack_shape",
            "first_divergence": {"field": "packed", "python_shape": list(exp.shape), "native_shape": list(got.shape)},
        }
    diff = np.abs(exp - got)
    bad = np.argwhere(diff > float(atol))
    if bad.size:
        row, col = [int(v) for v in bad[0].tolist()]
        return {
            "ok": False,
            "classification": "cxxrewrite4_native_policy_pack_value",
            "max_abs_diff": float(np.max(diff)),
            "first_divergence": {
                "field": "packed",
                "column_name": PACK_COLUMNS[col] if 0 <= col < len(PACK_COLUMNS) else str(col),
                "row": row,
                "column": col,
                "python_value": float(exp[row, col]),
                "native_value": float(got[row, col]),
                "abs_diff": float(diff[row, col]),
            },
        }
    checksum_diff = abs(float(expected.get("checksum", 0.0)) - float(actual.get("checksum", 0.0)))
    return {
        "ok": bool(checksum_diff <= max(float(atol), 1.0e-5)),
        "classification": "cxxrewrite4_native_policy_pack",
        "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
        "checksum_diff": float(checksum_diff),
        "first_divergence": None,
    }


def compile_native_backend(build_dir: Path | str, *, cxx: str | None = None) -> dict[str, Any]:
    build = Path(build_dir)
    compiler = cxx or shutil.which("g++")
    source_path = Path(__file__).resolve().parent / "native" / "cxxrewrite4_upstream_pack_kernel.cpp"
    library_path = build / "libcxxrewrite4_upstream_pack.so"
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


def load_native_backend(library_path: str | Path):
    lib = ctypes.CDLL(str(library_path))
    fn = lib.cxxrewrite4_pack_policy_heads
    i32p = ctypes.POINTER(ctypes.c_int32)
    f32p = ctypes.POINTER(ctypes.c_float)
    f64p = ctypes.POINTER(ctypes.c_double)
    fn.argtypes = [
        i32p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_float,
        f32p,
        f32p,
        f32p,
        f32p,
        i32p,
        i32p,
        i32p,
        i32p,
        f32p,
        f32p,
        f32p,
        f32p,
        f32p,
        f64p,
    ]
    fn.restype = ctypes.c_int

    def _call(
        runner: Any,
        frame: CxxRewrite3ResidentFrame,
        agent_ids: Any,
        *,
        energy_norm_denom: float,
        health_norm_denom: float,
    ) -> dict[str, Any]:
        ids = np.ascontiguousarray(np.asarray(agent_ids, dtype=np.int32).reshape(-1))
        h, p, f, fr = _drive_arrays(runner)
        if h is None or p is None or f is None or fr is None:
            drive_cap = 0
            h = p = f = fr = np.zeros((1,), dtype=np.float32)
        else:
            drive_cap = int(min(h.shape[0], p.shape[0], f.shape[0], fr.shape[0]))
        packed = np.zeros((int(ids.shape[0]), len(PACK_COLUMNS)), dtype=np.float32)
        checksum = ctypes.c_double(0.0)
        status = fn(
            ids.ctypes.data_as(i32p),
            ctypes.c_int32(int(ids.shape[0])),
            ctypes.c_int32(int(frame.capacity)),
            ctypes.c_int32(int(drive_cap)),
            ctypes.c_float(float(energy_norm_denom) or 1.0),
            ctypes.c_float(float(health_norm_denom) or 1.0),
            np.ascontiguousarray(frame.energy_by_id, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(frame.health_by_id, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(frame.injury_by_id, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(frame.scar_by_id, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(frame.pos_y_by_id, dtype=np.int32).ctypes.data_as(i32p),
            np.ascontiguousarray(frame.pos_x_by_id, dtype=np.int32).ctypes.data_as(i32p),
            np.ascontiguousarray(frame.action_by_id, dtype=np.int32).ctypes.data_as(i32p),
            np.ascontiguousarray(frame.agent_id_to_slot, dtype=np.int32).ctypes.data_as(i32p),
            np.ascontiguousarray(h, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(p, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(f, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(fr, dtype=np.float32).ctypes.data_as(f32p),
            packed.ctypes.data_as(f32p),
            ctypes.byref(checksum),
        )
        if int(status) != 0:
            raise RuntimeError(f"CXXREWRITE4 native policy-head pack failed with status {int(status)}")
        return {"packed": packed, "checksum": float(checksum.value)}

    return _call


def _native_pack_if_requested(
    runner: Any,
    frame: CxxRewrite3ResidentFrame,
    *,
    agent_ids: np.ndarray,
    energy_norm_denom: float,
    health_norm_denom: float,
    record: dict[str, Any],
) -> None:
    cfg = getattr(runner, "sim_cfg", None)
    if not native_pack_requested(cfg):
        return
    path = str(getattr(cfg, "cxxrewrite_resident_policyframe_native_pack_library_path", "") or "")
    if not path:
        path = str(getattr(cfg, "cxxrewrite_resident_observation_native_pack_library_path", "") or "")
    if not path:
        record["native_pack_skipped"] = True
        record["native_pack_skip_reason"] = "library_path_missing"
        frame.bump("cxxrewrite4_native_pack_skipped")
        if _fail_closed(cfg):
            raise RuntimeError("CXXREWRITE4 native policy-head pack requested but library path is missing")
        return
    lib_path = Path(path)
    if not lib_path.exists():
        record["native_pack_skipped"] = True
        record["native_pack_skip_reason"] = "library_missing"
        frame.bump("cxxrewrite4_native_pack_skipped")
        if _fail_closed(cfg):
            raise RuntimeError(f"CXXREWRITE4 native policy-head pack library missing: {lib_path}")
        return
    backend = getattr(runner, "_cxxrewrite4_native_pack_backend", None)
    if backend is None or getattr(runner, "_cxxrewrite4_native_pack_backend_path", "") != str(lib_path):
        backend = load_native_backend(lib_path)
        setattr(runner, "_cxxrewrite4_native_pack_backend", backend)
        setattr(runner, "_cxxrewrite4_native_pack_backend_path", str(lib_path))
    t0 = time.perf_counter()
    native = backend(
        runner,
        frame,
        agent_ids,
        energy_norm_denom=energy_norm_denom,
        health_norm_denom=health_norm_denom,
    )
    record["timing_s"]["native_policy_pack"] = float(record["timing_s"].get("native_policy_pack", 0.0)) + float(
        time.perf_counter() - t0
    )
    record["native_pack_used"] = True
    record["native_pack_checksum"] = float(native.get("checksum", 0.0))
    frame.bump("cxxrewrite4_native_pack_records")
    if _shadow_compare_requested(cfg):
        t1 = time.perf_counter()
        expected = python_pack_policy_heads(
            runner,
            frame,
            agent_ids,
            energy_norm_denom=energy_norm_denom,
            health_norm_denom=health_norm_denom,
        )
        report = compare_packed(expected, native, atol=0.0)
        record["native_pack_compare"] = report
        record["timing_s"]["native_policy_pack_compare"] = float(
            record["timing_s"].get("native_policy_pack_compare", 0.0)
        ) + float(time.perf_counter() - t1)
        if not bool(report.get("ok", False)):
            frame.first_divergence = report.get("first_divergence")
            if _fail_closed(cfg):
                raise RuntimeError(f"CXXREWRITE4 native policy-head pack parity failed: {report!r}")


def init_summary() -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "records": 0,
        "ok": True,
        "first_divergence": None,
        "observation_records": 0,
        "policyframe_records": 0,
        "action_records": 0,
        "post_env_records": 0,
        "agents": 0,
        "resident_observation_updates": 0,
        "resident_policyframe_updates": 0,
        "resident_action_updates": 0,
        "resident_post_env_scalar_updates": 0,
        "python_agent_scalar_reads_baseline": 0,
        "python_agent_scalar_reads_avoided": 0,
        "python_agent_scalar_writes_baseline": 0,
        "python_agent_scalar_writes_resident_first": 0,
        "python_post_env_agent_scalar_syncs": 0,
        "boundary_sync_records": 0,
        "hot_path_sync_records": 0,
        "drive_slot_syncs": 0,
        "native_pack_records": 0,
        "native_pack_skipped": 0,
        "native_pack_compare_checked": 0,
        "native_pack_compare_ok": True,
        "observation_digest_parity_ok": True,
        "action_array_parity_ok": True,
        "resident_generation_pre_env_records": 0,
        "resident_generation_pre_env_scalar_input_reads": 0,
        "max_observation_scalar_diff": 0.0,
        "timing_s": {},
    }


def update_summary(summary: dict[str, Any] | None, frame: CxxRewrite3ResidentFrame, record: dict[str, Any]) -> dict[str, Any]:
    if summary is None:
        summary = init_summary()
    summary["records"] = int(summary.get("records", 0)) + 1
    summary["agents"] = int(summary.get("agents", 0)) + int(record.get("agents", 0) or 0)
    for name in (
        "observation_records",
        "policyframe_records",
        "action_records",
        "post_env_records",
        "resident_observation_updates",
        "resident_policyframe_updates",
        "resident_action_updates",
        "resident_post_env_scalar_updates",
        "python_agent_scalar_reads_baseline",
        "python_agent_scalar_reads_avoided",
        "python_agent_scalar_writes_baseline",
        "python_agent_scalar_writes_resident_first",
        "python_post_env_agent_scalar_syncs",
        "boundary_sync_records",
        "hot_path_sync_records",
        "drive_slot_syncs",
        "resident_generation_pre_env_records",
        "resident_generation_pre_env_scalar_input_reads",
    ):
        summary[name] = int(summary.get(name, 0) or 0) + int(record.get(name, 0) or 0)
    counters = dict(getattr(frame, "counters", {}) or {})
    summary["native_pack_records"] = int(counters.get("cxxrewrite4_native_pack_records", summary.get("native_pack_records", 0)) or 0)
    summary["native_pack_skipped"] = int(counters.get("cxxrewrite4_native_pack_skipped", summary.get("native_pack_skipped", 0)) or 0)
    if bool(record.get("native_pack_skipped", False)):
        summary["native_pack_skipped"] = max(int(summary.get("native_pack_skipped", 0) or 0), 1)
    compare = record.get("native_pack_compare")
    if isinstance(compare, dict):
        summary["native_pack_compare_checked"] = int(summary.get("native_pack_compare_checked", 0)) + 1
        if not bool(compare.get("ok", False)):
            summary["native_pack_compare_ok"] = False
            summary["ok"] = False
            if summary.get("first_divergence") is None:
                summary["first_divergence"] = compare.get("first_divergence")
    parity = record.get("observation_parity")
    if isinstance(parity, dict):
        if not bool(parity.get("ok", True)):
            summary["observation_digest_parity_ok"] = False
            summary["ok"] = False
            if summary.get("first_divergence") is None:
                summary["first_divergence"] = parity.get("first_divergence")
        summary["max_observation_scalar_diff"] = float(
            max(float(summary.get("max_observation_scalar_diff", 0.0) or 0.0), float(parity.get("max_abs_diff", 0.0) or 0.0))
        )
    action_parity = record.get("action_array_parity")
    if isinstance(action_parity, dict) and not bool(action_parity.get("ok", True)):
        summary["action_array_parity_ok"] = False
        summary["ok"] = False
        if summary.get("first_divergence") is None:
            summary["first_divergence"] = action_parity.get("first_divergence")
    totals = dict(summary.get("timing_s", {}) or {})
    for key, value in dict(record.get("timing_s", {}) or {}).items():
        totals[str(key)] = float(totals.get(str(key), 0.0)) + float(value or 0.0)
    summary["timing_s"] = totals
    return summary


def write_summary(run_dir: Any, summary: dict[str, Any], *, filename: str = "cxxrewrite4_upstream_summary.json") -> Path | None:
    if not run_dir:
        return None
    out_dir = Path(str(run_dir)) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / str(filename)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(out_path)
    return out_path


def flush_summary(runner: Any) -> Path | None:
    summary = getattr(runner, "_cxxrewrite4_upstream_summary", None)
    if not isinstance(summary, dict):
        return None
    cfg = getattr(runner, "sim_cfg", None)
    if not bool(getattr(cfg, "cxxrewrite_resident_upstream_write_summary", True)):
        return None
    return write_summary(
        getattr(runner, "run_dir", None),
        summary,
        filename=str(getattr(cfg, "cxxrewrite_resident_upstream_summary_filename", "cxxrewrite4_upstream_summary.json")),
    )


def _update_and_write_summary(runner: Any, frame: CxxRewrite3ResidentFrame, record: dict[str, Any]) -> None:
    summary = update_summary(getattr(runner, "_cxxrewrite4_upstream_summary", None), frame, record)
    setattr(runner, "_cxxrewrite4_upstream_summary", summary)
    cfg = getattr(runner, "sim_cfg", None)
    if bool(getattr(cfg, "cxxrewrite_resident_upstream_defer_summary_write_enable", False)):
        return
    flush_summary(runner)


def _compare_observation_heads(
    *,
    alive: Any,
    energy_norm: np.ndarray,
    health_norm: np.ndarray,
    drives_np: np.ndarray | None,
    e_norm: float,
    h_norm: float,
) -> dict[str, Any]:
    agents = list(alive or [])
    py_e = np.asarray([_safe_float(getattr(agent, "energy", 0.0)) / float(e_norm or 1.0) for agent in agents], dtype=np.float32)
    py_h = np.asarray([_safe_float(getattr(agent, "health", 0.0)) / float(h_norm or 1.0) for agent in agents], dtype=np.float32)
    diffs = []
    if py_e.shape == energy_norm.shape:
        diffs.append(np.abs(py_e - energy_norm))
    if py_h.shape == health_norm.shape:
        diffs.append(np.abs(py_h - health_norm))
    max_diff = float(max([float(np.max(d)) for d in diffs if d.size] + [0.0]))
    ok = bool(max_diff <= 1.0e-6)
    first = None
    if not ok:
        diff = np.abs(py_e - energy_norm)
        if diff.size and float(np.max(diff)) > 1.0e-6:
            row = int(np.argmax(diff))
            first = {
                "field": "energy_norm",
                "row": row,
                "agent_id": _safe_int(getattr(agents[row], "agent_id", -1), -1) if row < len(agents) else -1,
                "python_value": float(py_e[row]),
                "resident_value": float(energy_norm[row]),
                "abs_diff": float(diff[row]),
            }
        else:
            diff = np.abs(py_h - health_norm)
            row = int(np.argmax(diff))
            first = {
                "field": "health_norm",
                "row": row,
                "agent_id": _safe_int(getattr(agents[row], "agent_id", -1), -1) if row < len(agents) else -1,
                "python_value": float(py_h[row]),
                "resident_value": float(health_norm[row]),
                "abs_diff": float(diff[row]),
            }
    return {
        "ok": ok,
        "classification": "resident_observation_heads",
        "max_abs_diff": max_diff,
        "drive_heads_present": drives_np is not None,
        "first_divergence": first,
    }


def prepare_policy_materialization(
    runner: Any,
    *,
    alive: Any,
    agent_ids_np: Any,
    e_norm: float,
    h_norm: float,
    stage: str,
) -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    if not upstream_requested(cfg):
        return None
    if not (resident_observation_requested(cfg) or resident_policyframe_requested(cfg) or native_pack_requested(cfg)):
        return None
    assert_authoritative_not_supported(cfg)
    t0 = time.perf_counter()
    agents = list(alive or [])
    ids = np.asarray(agent_ids_np, dtype=np.int32).reshape(-1)
    if ids.size == 0:
        return None
    frame = _ensure_frame(runner, agents)
    if ids.size and int(np.max(ids)) >= int(frame.capacity):
        _grow_frame(frame, int(np.max(ids)) + 1)
    hot_path_sync = False
    if not resident_post_env_scalar_requested(cfg) and not bool(
        getattr(cfg, "cxxrewrite_resident_upstream_forbid_hotpath_agent_sync", False)
    ):
        by_id = {_safe_int(getattr(agent, "agent_id", -1), -1): agent for agent in agents}
        for aid in ids.tolist():
            agent = by_id.get(int(aid))
            if agent is None:
                continue
            frame.energy_by_id[int(aid)] = np.float32(_safe_float(getattr(agent, "energy", 0.0)))
            frame.health_by_id[int(aid)] = np.float32(_safe_float(getattr(agent, "health", 0.0)))
            frame.injury_by_id[int(aid)] = np.float32(_safe_float(getattr(agent, "injury", 0.0)))
            frame.scar_by_id[int(aid)] = np.float32(_safe_float(getattr(agent, "scar", 0.0)))
            frame.alive_by_id[int(aid)] = bool(getattr(agent, "alive", False))
        frame.scalar_dirty_by_id[ids] = True
        hot_path_sync = True
    drive_slot_syncs = 0
    if bool(
        getattr(cfg, "cxxrewrite_resident_policyframe_enable", False)
        or getattr(cfg, "cxxrewrite_resident_policyframe_shadow_enable", False)
        or getattr(cfg, "cxxrewrite_resident_policyframe_native_pack_enable", False)
        or native_pack_requested(cfg)
    ):
        drive_slot_syncs = _sync_drive_slots_when_needed(runner, agents)
    use_observation = bool(
        getattr(cfg, "cxxrewrite_resident_observation_enable", False)
        or getattr(cfg, "cxxrewrite_resident_observation_authoritative_enable", False)
        or getattr(cfg, "cxxrewrite_resident_observation_native_pack_enable", False)
    )
    use_policyframe = bool(
        getattr(cfg, "cxxrewrite_resident_policyframe_enable", False)
        or getattr(cfg, "cxxrewrite_resident_policyframe_authoritative_enable", False)
        or getattr(cfg, "cxxrewrite_resident_policyframe_native_pack_enable", False)
    )
    energy_norm = (frame.energy_by_id[ids] / np.float32(float(e_norm) or 1.0)).astype(np.float32, copy=False)
    health_norm = (frame.health_by_id[ids] / np.float32(float(h_norm) or 1.0)).astype(np.float32, copy=False)
    drives_np, slots = _pack_drives_from_slots(runner, frame, ids)
    tokens = None
    if slots is not None:
        valid = (slots >= 0) & (slots < int(frame.slot_tokens.shape[0]))
        if bool(np.all(valid)):
            tokens = frame.slot_tokens[slots].astype(np.int64, copy=False)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": str(stage),
        "generation": int(frame.generation),
        "tick": int(frame.tick),
        "agents": int(ids.shape[0]),
        "observation_records": 1 if (resident_observation_requested(cfg) or use_observation) else 0,
        "policyframe_records": 1 if (resident_policyframe_requested(cfg) or use_policyframe) else 0,
        "resident_observation_updates": int(ids.shape[0]) if use_observation else 0,
        "resident_policyframe_updates": int(ids.shape[0]) if use_policyframe else 0,
        "python_agent_scalar_reads_baseline": int(ids.shape[0]) * 2,
        "python_agent_scalar_reads_avoided": int(ids.shape[0]) * 2
        if (use_observation or use_policyframe) and not hot_path_sync
        else 0,
        "hot_path_sync_records": 1 if hot_path_sync else 0,
        "boundary_sync_records": 1 if drive_slot_syncs else 0,
        "drive_slot_syncs": int(drive_slot_syncs),
        "native_pack_used": False,
        "native_pack_skipped": False,
        "digest": frame_digest(frame, agent_ids=ids),
        "timing_s": {
            "resident_policy_materialization": 0.0,
            "native_policy_pack": 0.0,
            "native_policy_pack_compare": 0.0,
            "total": 0.0,
        },
    }
    if _shadow_compare_requested(cfg):
        record["observation_parity"] = _compare_observation_heads(
            alive=agents,
            energy_norm=energy_norm,
            health_norm=health_norm,
            drives_np=drives_np,
            e_norm=float(e_norm) or 1.0,
            h_norm=float(h_norm) or 1.0,
        )
        if not bool(record["observation_parity"].get("ok", False)) and _fail_closed(cfg):
            raise RuntimeError(f"CXXREWRITE4 resident observation parity failed: {record['observation_parity']!r}")
    record["timing_s"]["resident_policy_materialization"] = float(time.perf_counter() - t0)
    _native_pack_if_requested(
        runner,
        frame,
        agent_ids=ids,
        energy_norm_denom=float(e_norm) or 1.0,
        health_norm_denom=float(h_norm) or 1.0,
        record=record,
    )
    record["timing_s"]["total"] = float(time.perf_counter() - t0)
    _update_and_write_summary(runner, frame, record)
    return {
        "frame": frame,
        "record": record,
        "use_resident_policy_heads": bool(use_observation or use_policyframe),
        "energy_norm": energy_norm,
        "health_norm": health_norm,
        "drives_np": drives_np if use_policyframe else None,
        "slots": slots,
        "tokens": tokens,
    }


def prepare_resident_generation_env_scalar_inputs(
    runner: Any,
    *,
    alive: Any,
    agent_ids_np: Any,
    stage: str = "resident_generation_pre_env",
) -> dict[str, np.ndarray] | None:
    cfg = getattr(runner, "sim_cfg", None)
    if not resident_generation_scalar_env_inputs_requested(cfg):
        return None
    assert_authoritative_not_supported(cfg)
    t0 = time.perf_counter()
    agents = list(alive or [])
    ids = np.asarray(agent_ids_np, dtype=np.int32).reshape(-1)
    if ids.size == 0:
        return {
            "energy": np.zeros((0,), dtype=np.float32),
            "health": np.zeros((0,), dtype=np.float32),
            "injury": np.zeros((0,), dtype=np.float32),
            "scar": np.zeros((0,), dtype=np.float32),
        }
    if bool(np.any(ids < 0)):
        if bool(getattr(cfg, "cxxrewrite_resident_generation_runner_fail_closed", False)):
            raise ValueError("resident generation scalar env inputs require non-negative agent ids")
        return None
    frame = _ensure_frame(runner, agents)
    if int(np.max(ids)) >= int(frame.capacity):
        _grow_frame(frame, int(np.max(ids)) + 1)
    max_id = int(np.max(ids))
    if max_id >= int(frame.alive_by_id.shape[0]):
        if bool(getattr(cfg, "cxxrewrite_resident_generation_runner_fail_closed", False)):
            raise ValueError("resident scalar frame capacity did not cover env input ids")
        return None
    energy = np.ascontiguousarray(frame.energy_by_id[ids], dtype=np.float32)
    health = np.ascontiguousarray(frame.health_by_id[ids], dtype=np.float32)
    injury = np.ascontiguousarray(frame.injury_by_id[ids], dtype=np.float32)
    scar = np.ascontiguousarray(frame.scar_by_id[ids], dtype=np.float32)
    frame.bump("resident_generation_pre_env_scalar_input_reads", int(ids.shape[0]) * 4)
    record = {
        "schema_version": SCHEMA_VERSION,
        "stage": str(stage),
        "generation": int(frame.generation),
        "tick": int(frame.tick),
        "agents": int(ids.shape[0]),
        "resident_generation_pre_env_records": 1,
        "resident_generation_pre_env_scalar_input_reads": int(ids.shape[0]) * 4,
        "python_agent_scalar_reads_baseline": int(ids.shape[0]) * 4,
        "python_agent_scalar_reads_avoided": int(ids.shape[0]) * 4,
        "timing_s": {"resident_generation_pre_env_scalar_inputs": float(time.perf_counter() - t0)},
    }
    _update_and_write_summary(runner, frame, record)
    return {"energy": energy, "health": health, "injury": injury, "scar": scar}


def record_action_array(runner: Any, *, alive: Any, agent_ids_np: Any, actions_np: Any, stage: str = "policy_actions") -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    if not upstream_requested(cfg):
        return None
    assert_authoritative_not_supported(cfg)
    t0 = time.perf_counter()
    agents = list(alive or [])
    ids = np.asarray(agent_ids_np, dtype=np.int32).reshape(-1)
    actions = np.asarray(actions_np, dtype=np.int32).reshape(-1)
    if ids.shape[0] != actions.shape[0]:
        if _fail_closed(cfg):
            raise ValueError("CXXREWRITE4 action array update requires aligned ids/actions")
        return None
    frame = _ensure_frame(runner, agents)
    if ids.size and int(np.max(ids)) >= int(frame.capacity):
        _grow_frame(frame, int(np.max(ids)) + 1)
    frame.action_by_id[ids] = actions
    frame.action_dirty_by_id[ids] = True
    frame.bump("cxxrewrite4_action_updates", int(ids.shape[0]))
    record = {
        "schema_version": SCHEMA_VERSION,
        "stage": str(stage),
        "generation": int(frame.generation),
        "tick": int(frame.tick),
        "agents": int(ids.shape[0]),
        "action_records": 1,
        "resident_action_updates": int(ids.shape[0]),
        "action_array_parity": {"ok": True, "classification": "resident_action_array", "first_divergence": None},
        "digest": frame_digest(frame, agent_ids=ids),
        "timing_s": {"resident_action_update": float(time.perf_counter() - t0)},
    }
    _update_and_write_summary(runner, frame, record)
    return record


def record_post_env_scalar_arrays(
    runner: Any,
    *,
    alive: Any,
    agent_ids_np: Any,
    new_energy: Any,
    new_health: Any,
    new_injury: Any,
    new_scar: Any,
    died_mask: Any,
    stage: str = "env_apply_arrays",
) -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    if not resident_post_env_scalar_requested(cfg):
        return None
    assert_authoritative_not_supported(cfg)
    t0 = time.perf_counter()
    ids = np.asarray(agent_ids_np, dtype=np.int32).reshape(-1)
    frame = _ensure_frame(runner, list(alive or []))
    if ids.size and int(np.max(ids)) >= int(frame.capacity):
        _grow_frame(frame, int(np.max(ids)) + 1)
    frame.energy_by_id[ids] = np.asarray(new_energy, dtype=np.float32).reshape(-1)
    frame.health_by_id[ids] = np.asarray(new_health, dtype=np.float32).reshape(-1)
    frame.injury_by_id[ids] = np.asarray(new_injury, dtype=np.float32).reshape(-1)
    frame.scar_by_id[ids] = np.asarray(new_scar, dtype=np.float32).reshape(-1)
    alive_after = np.logical_not(np.asarray(died_mask, dtype=bool).reshape(-1))
    frame.alive_by_id[ids] = alive_after
    env = getattr(runner, "env", None)
    if env is not None:
        pos_y = np.asarray(getattr(env, "agent_pos_y", frame.pos_y_by_id), dtype=np.int32)
        pos_x = np.asarray(getattr(env, "agent_pos_x", frame.pos_x_by_id), dtype=np.int32)
        env_alive = np.asarray(getattr(env, "agent_alive", frame.alive_by_id), dtype=bool)
        cap = max(int(frame.capacity), int(pos_y.shape[0]), int(pos_x.shape[0]), int(env_alive.shape[0]))
        if cap > int(frame.capacity):
            _grow_frame(frame, cap)
        frame.pos_y_by_id[: int(pos_y.shape[0])] = pos_y
        frame.pos_x_by_id[: int(pos_x.shape[0])] = pos_x
        frame.alive_by_id[: int(env_alive.shape[0])] = env_alive
        frame.env_dirty_by_id[: int(env_alive.shape[0])] = True
        frame.alive_by_id[ids] = alive_after
    frame.scalar_dirty_by_id[ids] = True
    frame.bump("cxxrewrite4_post_env_array_updates", int(ids.shape[0]))
    record = {
        "schema_version": SCHEMA_VERSION,
        "stage": str(stage),
        "generation": int(frame.generation),
        "tick": int(frame.tick),
        "agents": int(ids.shape[0]),
        "post_env_records": 1,
        "resident_post_env_scalar_updates": int(ids.shape[0]),
        "python_agent_scalar_writes_baseline": int(ids.shape[0]) * 4,
        "python_agent_scalar_writes_resident_first": int(ids.shape[0]) * 4,
        "timing_s": {"resident_post_env_array_update": float(time.perf_counter() - t0)},
    }
    _update_and_write_summary(runner, frame, record)
    return record


def sync_agents_scalar_state(runner: Any, agents: Any, *, stage: str) -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    if not resident_post_env_scalar_requested(cfg):
        return None
    assert_authoritative_not_supported(cfg)
    t0 = time.perf_counter()
    agent_list = [agent for agent in list(agents or []) if agent is not None]
    if not agent_list:
        return None
    frame = _ensure_frame(runner, agent_list)
    ids = _agent_ids_from_agents(agent_list)
    ids = ids[ids >= 0]
    if ids.size == 0:
        return None
    if int(np.max(ids)) >= int(frame.capacity):
        _grow_frame(frame, int(np.max(ids)) + 1)
    by_id = {_safe_int(getattr(agent, "agent_id", -1), -1): agent for agent in agent_list}
    for aid in ids.tolist():
        agent = by_id.get(int(aid))
        if agent is None:
            continue
        frame.energy_by_id[int(aid)] = np.float32(_safe_float(getattr(agent, "energy", 0.0)))
        frame.health_by_id[int(aid)] = np.float32(_safe_float(getattr(agent, "health", 0.0)))
        frame.injury_by_id[int(aid)] = np.float32(_safe_float(getattr(agent, "injury", 0.0)))
        frame.scar_by_id[int(aid)] = np.float32(_safe_float(getattr(agent, "scar", 0.0)))
        frame.alive_by_id[int(aid)] = bool(getattr(agent, "alive", False))
    frame.scalar_dirty_by_id[ids] = True
    frame.bump("cxxrewrite4_post_env_agent_syncs", int(ids.shape[0]))
    record = {
        "schema_version": SCHEMA_VERSION,
        "stage": str(stage),
        "generation": int(frame.generation),
        "tick": int(frame.tick),
        "agents": int(ids.shape[0]),
        "post_env_records": 1,
        "boundary_sync_records": 1,
        "resident_post_env_scalar_updates": int(ids.shape[0]),
        "python_post_env_agent_scalar_syncs": int(ids.shape[0]),
        "timing_s": {"resident_agent_scalar_sync": float(time.perf_counter() - t0)},
    }
    _update_and_write_summary(runner, frame, record)
    return record


def write_json(path: Path | str, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
