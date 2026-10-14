from __future__ import annotations

"""CXXREWRITE-3 resident native-entry frame and native packing helper.

This module deliberately targets the CXXREWRITE2 blocker: the native tick
kernel was fast, but its resident input frame was repaired from Python pre-env
arrays at every native entry. CXXREWRITE3 provides a default-off resident frame
that receives those pre-env arrays before CXXREWRITE2 packs native inputs, so
the CXXREWRITE2 repair counter measures zero when the frame is current.
"""

import ctypes
from dataclasses import dataclass, field
import hashlib
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


SCHEMA_VERSION = "cxxrewrite3_resident_frame_v1"
SUMMARY_SCHEMA_VERSION = "cxxrewrite3_resident_frame_summary_v1"
BUILD_SCHEMA_VERSION = "cxxrewrite3_native_pack_build_status_v1"
API_SCHEMA_VERSION = "cxxrewrite3_native_pack_api_v1"


@dataclass
class CxxRewrite3ResidentFrame:
    schema_version: str
    generation: int
    tick: int
    capacity: int
    slot_capacity: int
    slot_to_agent_id: np.ndarray
    agent_id_to_slot: np.ndarray
    slot_tokens: np.ndarray
    alive_by_id: np.ndarray
    energy_by_id: np.ndarray
    health_by_id: np.ndarray
    injury_by_id: np.ndarray
    scar_by_id: np.ndarray
    pos_y_by_id: np.ndarray
    pos_x_by_id: np.ndarray
    action_by_id: np.ndarray
    scalar_dirty_by_id: np.ndarray
    env_dirty_by_id: np.ndarray
    action_dirty_by_id: np.ndarray
    rebuild_count: int = 1
    counters: dict[str, int] = field(default_factory=dict)
    max_scalar_drift_vs_pre_env: float = 0.0
    native_pack_checksums: int = 0
    first_divergence: dict[str, Any] | None = None

    def bump(self, name: str, amount: int = 1) -> None:
        self.counters[str(name)] = int(self.counters.get(str(name), 0)) + int(amount)


def resident_frame_requested(cfg: Any) -> bool:
    return bool(getattr(cfg, "cxxrewrite_resident_frame_enable", False)) or bool(
        getattr(cfg, "cxxrewrite_resident_frame_shadow_enable", False)
    ) or bool(getattr(cfg, "cxxrewrite_resident_frame_authoritative_enable", False)) or bool(
        getattr(cfg, "cxxrewrite_resident_frame_validate_shadow_enable", False)
    ) or bool(getattr(cfg, "cxxrewrite_resident_frame_native_packing_enable", False)) or bool(
        getattr(cfg, "cxxrewrite_resident_frame_native_packing_shadow_enable", False)
    ) or bool(getattr(cfg, "cxxrewrite_resident_frame_native_packing_authoritative_enable", False))


def _fail_closed(cfg: Any) -> bool:
    return bool(getattr(cfg, "cxxrewrite_resident_frame_fail_closed", False)) or bool(
        getattr(cfg, "cxxrewrite_resident_frame_authoritative_enable", False)
    ) or bool(getattr(cfg, "cxxrewrite_resident_frame_native_packing_authoritative_enable", False))


def assert_authoritative_not_supported(cfg: Any) -> None:
    if bool(getattr(cfg, "cxxrewrite_resident_frame_authoritative_enable", False)) or bool(
        getattr(cfg, "cxxrewrite_resident_frame_native_packing_authoritative_enable", False)
    ):
        raise RuntimeError(
            "CXXREWRITE3 resident authoritative modes are not supported in this gate: "
            "the resident frame is a parity/timing contract and native-entry packing "
            "helper, while broader runner policy, reproduction, metrics, and "
            "checkpoint semantics remain Python-authoritative."
        )


def native_api_schema() -> dict[str, Any]:
    return {
        "schema_version": API_SCHEMA_VERSION,
        "kernel": "cxxrewrite3_pack_native_entry",
        "mode": "resident_frame_native_entry_packing",
        "inputs": [
            "agent_ids",
            "energy_by_id",
            "health_by_id",
            "injury_by_id",
            "scar_by_id",
            "pos_y_by_id",
            "pos_x_by_id",
            "actions_by_id",
        ],
        "outputs": ["packed Nx7 float32 head", "checksum"],
        "packed_columns": ["energy", "health", "injury", "scar", "pos_y", "pos_x", "action"],
        "normal_import_requires_native": False,
    }


def resident_frame_schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "identity": ["slot_to_agent_id", "agent_id_to_slot", "slot_tokens", "alive_by_id"],
        "scalar_state": ["energy_by_id", "health_by_id", "injury_by_id", "scar_by_id"],
        "env_state": ["pos_y_by_id", "pos_x_by_id"],
        "policy_native_entry": ["action_by_id"],
        "dirty_masks": ["scalar_dirty_by_id", "env_dirty_by_id", "action_dirty_by_id"],
        "sync_counters": [
            "initial_agent_scalar_reads",
            "pre_env_array_updates",
            "pre_env_scalar_syncs_eliminated",
            "post_env_array_updates",
            "native_pack_records",
            "native_pack_skipped",
            "native_input_agent_object_scalar_reads",
        ],
    }


def blocker_map() -> dict[str, Any]:
    return {
        "schema_version": "cxxrewrite3_blocker_map_v1",
        "cxxrewrite2_blocker": (
            "CXXREWRITE2 repaired resident scalar arrays inside native-entry packing "
            "when earlier Python-owned tick stages had changed agent scalar state."
        ),
        "boundaries": [
            {
                "file": "engine/simulation/runner_pkg/runner_ticks_env_apply.py",
                "function": "_apply_env_tick",
                "current_owner": "Python env/tick apply arrays plus Agent writeback",
                "proposed_resident_owner": "CXXREWRITE3 resident frame receives pre-env and post-env arrays",
                "native_can_consume_directly": True,
                "expected_sync_reduction": "CXXREWRITE2 resident_pre_env_scalar_syncs should drop to zero for enabled arms.",
            },
            {
                "file": "engine/simulation/runner_pkg/runner_policy.py",
                "function": "_batched_actions/_batched_actions_strided",
                "current_owner": "Python policy materialization and action sampling",
                "proposed_resident_owner": "future resident policy frame for observation and policy heads",
                "native_can_consume_directly": "partially; CXXREWRITE3 native pack consumes action/scalar/native-entry heads",
                "expected_sync_reduction": "lower-bound only until policy materialization itself writes resident arrays first.",
            },
            {
                "file": "engine/simulation/runner_pkg/runner_ticks_recovery.py",
                "function": "_apply_rest_and_recovery",
                "current_owner": "Python Agent scalar mutation after env apply",
                "proposed_resident_owner": "future scalar mutation owner or explicit post-stage array event",
                "native_can_consume_directly": False,
                "expected_sync_reduction": "not fully removable without wider post-env scalar ownership.",
            },
        ],
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


def _grow_1d(arr: np.ndarray, new_capacity: int, fill: int | float | bool) -> np.ndarray:
    if int(new_capacity) <= int(arr.shape[0]):
        return arr
    out = np.full((int(new_capacity),), fill, dtype=arr.dtype)
    out[: int(arr.shape[0])] = arr
    return out


def _grow_frame(frame: CxxRewrite3ResidentFrame, new_capacity: int) -> None:
    cap = int(max(new_capacity, int(frame.capacity) * 2, 1))
    frame.agent_id_to_slot = _grow_1d(frame.agent_id_to_slot, cap, -1)
    frame.alive_by_id = _grow_1d(frame.alive_by_id, cap, False)
    frame.energy_by_id = _grow_1d(frame.energy_by_id, cap, 0.0)
    frame.health_by_id = _grow_1d(frame.health_by_id, cap, 0.0)
    frame.injury_by_id = _grow_1d(frame.injury_by_id, cap, 0.0)
    frame.scar_by_id = _grow_1d(frame.scar_by_id, cap, 0.0)
    frame.pos_y_by_id = _grow_1d(frame.pos_y_by_id, cap, -1)
    frame.pos_x_by_id = _grow_1d(frame.pos_x_by_id, cap, -1)
    frame.action_by_id = _grow_1d(frame.action_by_id, cap, -1)
    frame.scalar_dirty_by_id = _grow_1d(frame.scalar_dirty_by_id, cap, False)
    frame.env_dirty_by_id = _grow_1d(frame.env_dirty_by_id, cap, False)
    frame.action_dirty_by_id = _grow_1d(frame.action_dirty_by_id, cap, False)
    frame.capacity = cap
    frame.bump("capacity_grows")


def _slot_token(agent: Any) -> int:
    return _safe_int(getattr(agent, "slot_token", getattr(agent, "_slot_token", 0)), 0)


def _slot_capacity(runner: Any, alive: list[Any]) -> int:
    pop = getattr(runner, "pop", None)
    cap = _safe_int(getattr(pop, "max_slots", 0), 0)
    if cap > 0:
        return cap
    slots = [_safe_int(getattr(agent, "slot", -1), -1) for agent in alive]
    return max([0, *slots]) + 1


def build_resident_frame_from_runner(runner: Any, *, alive: Any, generation: int, tick: int) -> CxxRewrite3ResidentFrame:
    agents = list(alive or [])
    env = getattr(runner, "env", None)
    env_alive = np.asarray(getattr(env, "agent_alive", np.zeros((1,), dtype=bool)), dtype=bool)
    pos_y = np.asarray(getattr(env, "agent_pos_y", np.full_like(env_alive, -1, dtype=np.int32)), dtype=np.int32)
    pos_x = np.asarray(getattr(env, "agent_pos_x", np.full_like(env_alive, -1, dtype=np.int32)), dtype=np.int32)
    max_aid = max([_safe_int(getattr(agent, "agent_id", -1), -1) for agent in agents] + [int(env_alive.shape[0]) - 1, 0])
    capacity = max(1, int(max_aid) + 1, int(pos_y.shape[0]), int(pos_x.shape[0]), int(env_alive.shape[0]))
    slots = _slot_capacity(runner, agents)
    frame = CxxRewrite3ResidentFrame(
        schema_version=SCHEMA_VERSION,
        generation=int(generation),
        tick=int(tick),
        capacity=int(capacity),
        slot_capacity=int(slots),
        slot_to_agent_id=np.full((int(slots),), -1, dtype=np.int32),
        agent_id_to_slot=np.full((int(capacity),), -1, dtype=np.int32),
        slot_tokens=np.zeros((int(slots),), dtype=np.int64),
        alive_by_id=np.zeros((int(capacity),), dtype=np.bool_),
        energy_by_id=np.zeros((int(capacity),), dtype=np.float32),
        health_by_id=np.zeros((int(capacity),), dtype=np.float32),
        injury_by_id=np.zeros((int(capacity),), dtype=np.float32),
        scar_by_id=np.zeros((int(capacity),), dtype=np.float32),
        pos_y_by_id=np.full((int(capacity),), -1, dtype=np.int32),
        pos_x_by_id=np.full((int(capacity),), -1, dtype=np.int32),
        action_by_id=np.full((int(capacity),), -1, dtype=np.int32),
        scalar_dirty_by_id=np.zeros((int(capacity),), dtype=np.bool_),
        env_dirty_by_id=np.zeros((int(capacity),), dtype=np.bool_),
        action_dirty_by_id=np.zeros((int(capacity),), dtype=np.bool_),
    )
    frame.pos_y_by_id[: int(pos_y.shape[0])] = pos_y
    frame.pos_x_by_id[: int(pos_x.shape[0])] = pos_x
    frame.alive_by_id[: int(env_alive.shape[0])] = env_alive
    for agent in agents:
        aid = _safe_int(getattr(agent, "agent_id", -1), -1)
        slot = _safe_int(getattr(agent, "slot", -1), -1)
        if aid < 0:
            continue
        if aid >= int(frame.capacity):
            _grow_frame(frame, aid + 1)
        frame.energy_by_id[aid] = np.float32(_safe_float(getattr(agent, "energy", 0.0)))
        frame.health_by_id[aid] = np.float32(_safe_float(getattr(agent, "health", 0.0)))
        frame.injury_by_id[aid] = np.float32(_safe_float(getattr(agent, "injury", 0.0)))
        frame.scar_by_id[aid] = np.float32(_safe_float(getattr(agent, "scar", 0.0)))
        frame.alive_by_id[aid] = bool(getattr(agent, "alive", True))
        if 0 <= slot < int(frame.slot_capacity):
            frame.slot_to_agent_id[slot] = np.int32(aid)
            frame.agent_id_to_slot[aid] = np.int32(slot)
            frame.slot_tokens[slot] = np.int64(_slot_token(agent))
    frame.bump("initial_agent_scalar_reads", len(agents))
    return frame


def ensure_resident_frame(runner: Any, *, alive: Any, generation: int, tick: int) -> CxxRewrite3ResidentFrame:
    frame = getattr(runner, "_cxxrewrite3_resident_frame", None)
    if not isinstance(frame, CxxRewrite3ResidentFrame) or int(frame.generation) != int(generation):
        frame = build_resident_frame_from_runner(runner, alive=alive, generation=generation, tick=tick)
        setattr(runner, "_cxxrewrite3_resident_frame", frame)
        return frame
    frame.tick = int(tick)
    agents = list(alive or [])
    max_aid = max([_safe_int(getattr(agent, "agent_id", -1), -1) for agent in agents] + [-1])
    if max_aid >= int(frame.capacity):
        _grow_frame(frame, max_aid + 1)
    missing = []
    for agent in agents:
        aid = _safe_int(getattr(agent, "agent_id", -1), -1)
        slot = _safe_int(getattr(agent, "slot", -1), -1)
        if aid < 0:
            continue
        if 0 <= slot < int(frame.slot_capacity) and int(frame.agent_id_to_slot[aid]) != slot:
            missing.append(agent)
    for agent in missing:
        aid = _safe_int(getattr(agent, "agent_id", -1), -1)
        slot = _safe_int(getattr(agent, "slot", -1), -1)
        frame.energy_by_id[aid] = np.float32(_safe_float(getattr(agent, "energy", 0.0)))
        frame.health_by_id[aid] = np.float32(_safe_float(getattr(agent, "health", 0.0)))
        frame.injury_by_id[aid] = np.float32(_safe_float(getattr(agent, "injury", 0.0)))
        frame.scar_by_id[aid] = np.float32(_safe_float(getattr(agent, "scar", 0.0)))
        frame.alive_by_id[aid] = bool(getattr(agent, "alive", True))
        if 0 <= slot < int(frame.slot_capacity):
            frame.slot_to_agent_id[slot] = np.int32(aid)
            frame.agent_id_to_slot[aid] = np.int32(slot)
            frame.slot_tokens[slot] = np.int64(_slot_token(agent))
    if missing:
        frame.bump("missing_agent_syncs", len(missing))
        frame.bump("initial_agent_scalar_reads", len(missing))
    return frame


def frame_digest(frame: CxxRewrite3ResidentFrame, *, agent_ids: Any | None = None) -> dict[str, Any]:
    ids = np.asarray(agent_ids, dtype=np.int32).reshape(-1) if agent_ids is not None else np.nonzero(frame.alive_by_id)[0].astype(np.int32)
    ids = ids[(ids >= 0) & (ids < int(frame.capacity))]

    def _hash(arr: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()

    payload = {
        "schema_version": "cxxrewrite3_frame_digest_v1",
        "generation": int(frame.generation),
        "tick": int(frame.tick),
        "count": int(ids.shape[0]),
        "agent_ids_sha256": _hash(ids.astype(np.int32, copy=False)),
        "energy_sha256": _hash(frame.energy_by_id[ids].astype(np.float32, copy=False)),
        "health_sha256": _hash(frame.health_by_id[ids].astype(np.float32, copy=False)),
        "injury_sha256": _hash(frame.injury_by_id[ids].astype(np.float32, copy=False)),
        "scar_sha256": _hash(frame.scar_by_id[ids].astype(np.float32, copy=False)),
        "pos_y_sha256": _hash(frame.pos_y_by_id[ids].astype(np.int32, copy=False)),
        "pos_x_sha256": _hash(frame.pos_x_by_id[ids].astype(np.int32, copy=False)),
        "action_sha256": _hash(frame.action_by_id[ids].astype(np.int32, copy=False)),
    }
    payload["digest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return payload


def python_pack_native_entry(frame: CxxRewrite3ResidentFrame, agent_ids: Any) -> dict[str, Any]:
    ids = np.asarray(agent_ids, dtype=np.int32).reshape(-1)
    packed = np.column_stack(
        [
            frame.energy_by_id[ids],
            frame.health_by_id[ids],
            frame.injury_by_id[ids],
            frame.scar_by_id[ids],
            frame.pos_y_by_id[ids].astype(np.float32),
            frame.pos_x_by_id[ids].astype(np.float32),
            frame.action_by_id[ids].astype(np.float32),
        ]
    ).astype(np.float32, copy=False)
    weights = np.asarray([1.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0], dtype=np.float64)
    return {"packed": packed, "checksum": float(np.sum(packed.astype(np.float64) * weights.reshape((1, -1))))}


def compare_packed(expected: dict[str, Any], actual: dict[str, Any], *, atol: float = 0.0) -> dict[str, Any]:
    exp = np.asarray(expected.get("packed"), dtype=np.float32)
    got = np.asarray(actual.get("packed"), dtype=np.float32)
    if exp.shape != got.shape:
        return {
            "ok": False,
            "classification": "native_packing_shape",
            "first_divergence": {"field": "packed", "python_shape": list(exp.shape), "native_shape": list(got.shape)},
        }
    diff = np.abs(exp - got)
    bad = np.argwhere(diff > float(atol))
    if bad.size:
        row, col = [int(v) for v in bad[0].tolist()]
        return {
            "ok": False,
            "classification": "native_packing_value",
            "max_abs_diff": float(np.max(diff)),
            "first_divergence": {
                "field": "packed",
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
        "classification": "native_packing",
        "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
        "checksum_diff": float(checksum_diff),
        "first_divergence": None,
    }


def compile_native_backend(build_dir: Path | str, *, cxx: str | None = None) -> dict[str, Any]:
    build = Path(build_dir)
    compiler = cxx or shutil.which("g++")
    source_path = Path(__file__).resolve().parent / "native" / "cxxrewrite3_resident_pack_kernel.cpp"
    library_path = build / "libcxxrewrite3_resident_pack.so"
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
    try:
        import pybind11  # noqa: WPS433

        pybind11_reason = str(pybind11.__version__)
        pybind11_available = True
    except Exception as exc:
        pybind11_reason = f"{type(exc).__name__}: {exc}"
        pybind11_available = False
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


def load_native_backend(library_path: str | Path):
    lib = ctypes.CDLL(str(library_path))
    fn = lib.cxxrewrite3_pack_native_entry
    i32p = ctypes.POINTER(ctypes.c_int32)
    f32p = ctypes.POINTER(ctypes.c_float)
    f64p = ctypes.POINTER(ctypes.c_double)
    fn.argtypes = [i32p, ctypes.c_int32, ctypes.c_int32, f32p, f32p, f32p, f32p, i32p, i32p, i32p, f32p, f64p]
    fn.restype = ctypes.c_int

    def _call(frame: CxxRewrite3ResidentFrame, agent_ids: Any) -> dict[str, Any]:
        ids = np.ascontiguousarray(np.asarray(agent_ids, dtype=np.int32).reshape(-1))
        packed = np.zeros((int(ids.shape[0]), 7), dtype=np.float32)
        checksum = ctypes.c_double(0.0)
        status = fn(
            ids.ctypes.data_as(i32p),
            ctypes.c_int32(int(ids.shape[0])),
            ctypes.c_int32(int(frame.capacity)),
            np.ascontiguousarray(frame.energy_by_id, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(frame.health_by_id, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(frame.injury_by_id, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(frame.scar_by_id, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(frame.pos_y_by_id, dtype=np.int32).ctypes.data_as(i32p),
            np.ascontiguousarray(frame.pos_x_by_id, dtype=np.int32).ctypes.data_as(i32p),
            np.ascontiguousarray(frame.action_by_id, dtype=np.int32).ctypes.data_as(i32p),
            packed.ctypes.data_as(f32p),
            ctypes.byref(checksum),
        )
        if int(status) != 0:
            raise RuntimeError(f"CXXREWRITE3 native resident pack failed with status {int(status)}")
        return {"packed": packed, "checksum": float(checksum.value)}

    return _call


def _generation_tick(runner: Any) -> tuple[int, int]:
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
    return generation, tick


def _sync_cxxrewrite2_frame(
    runner: Any,
    *,
    alive: Any,
    generation: int,
    agent_ids: np.ndarray,
    energies: np.ndarray,
    healths: np.ndarray,
    injuries: np.ndarray,
    scars: np.ndarray,
) -> None:
    try:
        from perf.cxxrewrite2_native_tick import ensure_resident_tick_frame  # noqa: WPS433

        frame2 = ensure_resident_tick_frame(runner, alive=alive, generation=generation)
        if agent_ids.size and int(np.max(agent_ids)) >= int(frame2.capacity):
            from perf.cxxrewrite2_native_tick import _grow_resident_frame  # noqa: WPS433

            _grow_resident_frame(frame2, int(np.max(agent_ids)) + 1)
        frame2.energy_by_id[agent_ids] = energies
        frame2.health_by_id[agent_ids] = healths
        frame2.injury_by_id[agent_ids] = injuries
        frame2.scar_by_id[agent_ids] = scars
    except Exception:
        return


def _native_pack_if_requested(
    runner: Any,
    frame: CxxRewrite3ResidentFrame,
    *,
    agent_ids: np.ndarray,
    record: dict[str, Any],
) -> None:
    cfg = getattr(runner, "sim_cfg", None)
    native_requested = bool(getattr(cfg, "cxxrewrite_resident_frame_native_packing_enable", False)) or bool(
        getattr(cfg, "cxxrewrite_resident_frame_native_packing_shadow_enable", False)
    ) or bool(getattr(cfg, "cxxrewrite_resident_frame_native_packing_authoritative_enable", False))
    compare = bool(getattr(cfg, "cxxrewrite_resident_frame_native_packing_compare_enable", True))
    if not native_requested:
        return
    path = str(getattr(cfg, "cxxrewrite_resident_frame_native_packing_library_path", "") or "")
    if not path:
        record["native_pack_skipped"] = True
        record["native_pack_skip_reason"] = "library_path_missing"
        frame.bump("native_pack_skipped")
        if _fail_closed(cfg):
            raise RuntimeError("CXXREWRITE3 native packing requested but library path is missing")
        return
    lib_path = Path(path)
    if not lib_path.exists():
        record["native_pack_skipped"] = True
        record["native_pack_skip_reason"] = "library_missing"
        frame.bump("native_pack_skipped")
        if _fail_closed(cfg):
            raise RuntimeError(f"CXXREWRITE3 native packing library missing: {lib_path}")
        return
    backend = getattr(runner, "_cxxrewrite3_native_pack_backend", None)
    if backend is None or getattr(runner, "_cxxrewrite3_native_pack_backend_path", "") != str(lib_path):
        backend = load_native_backend(lib_path)
        setattr(runner, "_cxxrewrite3_native_pack_backend", backend)
        setattr(runner, "_cxxrewrite3_native_pack_backend_path", str(lib_path))
    t0 = time.perf_counter()
    native = backend(frame, agent_ids)
    record["timing_s"]["native_packing_materialization"] += float(time.perf_counter() - t0)
    frame.native_pack_checksums += 1
    frame.bump("native_pack_records")
    record["native_pack_used"] = True
    record["native_pack_checksum"] = float(native.get("checksum", 0.0))
    if compare:
        t1 = time.perf_counter()
        expected = python_pack_native_entry(frame, agent_ids)
        report = compare_packed(expected, native, atol=0.0)
        record["timing_s"]["native_packing_comparison"] += float(time.perf_counter() - t1)
        record["native_pack_compare"] = report
        if not bool(report.get("ok", False)):
            frame.first_divergence = report.get("first_divergence")
            if _fail_closed(cfg):
                raise RuntimeError(f"CXXREWRITE3 native packing parity failed: {report!r}")


def stage_pre_env_native_entry(
    runner: Any,
    *,
    alive: Any,
    agent_ids_np: Any,
    actions_np: Any,
    energies: Any,
    healths: Any,
    injuries: Any,
    scars: Any,
) -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    if not resident_frame_requested(cfg):
        return None
    assert_authoritative_not_supported(cfg)
    t0 = time.perf_counter()
    generation, tick = _generation_tick(runner)
    ids = np.asarray(agent_ids_np, dtype=np.int32).reshape(-1)
    frame = ensure_resident_frame(runner, alive=alive, generation=generation, tick=tick)
    if ids.size and int(np.max(ids)) >= int(frame.capacity):
        _grow_frame(frame, int(np.max(ids)) + 1)
    e = np.asarray(energies, dtype=np.float32).reshape(-1)
    h = np.asarray(healths, dtype=np.float32).reshape(-1)
    inj = np.asarray(injuries, dtype=np.float32).reshape(-1)
    scar = np.asarray(scars, dtype=np.float32).reshape(-1)
    actions = np.asarray(actions_np, dtype=np.int32).reshape(-1)
    if not (ids.shape[0] == e.shape[0] == h.shape[0] == inj.shape[0] == scar.shape[0] == actions.shape[0]):
        raise ValueError("CXXREWRITE3 pre-env frame arrays must be aligned")
    drift = {
        "energy": float(np.max(np.abs(frame.energy_by_id[ids] - e))) if ids.size else 0.0,
        "health": float(np.max(np.abs(frame.health_by_id[ids] - h))) if ids.size else 0.0,
        "injury": float(np.max(np.abs(frame.injury_by_id[ids] - inj))) if ids.size else 0.0,
        "scar": float(np.max(np.abs(frame.scar_by_id[ids] - scar))) if ids.size else 0.0,
    }
    max_drift = max(float(v) for v in drift.values()) if drift else 0.0
    frame.max_scalar_drift_vs_pre_env = max(float(frame.max_scalar_drift_vs_pre_env), float(max_drift))
    frame.energy_by_id[ids] = e
    frame.health_by_id[ids] = h
    frame.injury_by_id[ids] = inj
    frame.scar_by_id[ids] = scar
    frame.action_by_id[ids] = actions
    frame.scalar_dirty_by_id[ids] = True
    frame.action_dirty_by_id[ids] = True
    frame.bump("pre_env_array_updates", int(ids.shape[0]))
    if max_drift > 1.0e-4:
        frame.bump("pre_env_scalar_syncs_eliminated", int(ids.shape[0]))
    _sync_cxxrewrite2_frame(
        runner,
        alive=alive,
        generation=generation,
        agent_ids=ids,
        energies=e,
        healths=h,
        injuries=inj,
        scars=scar,
    )
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generation": int(generation),
        "tick": int(tick),
        "agents": int(ids.shape[0]),
        "resident_frame_used": True,
        "mode": "resident_frame_native_entry",
        "max_scalar_drift_vs_pre_env": float(max_drift),
        "scalar_drift": drift,
        "native_input_agent_object_scalar_reads": 0,
        "native_pack_used": False,
        "native_pack_skipped": False,
        "first_divergence": None,
        "digest": frame_digest(frame, agent_ids=ids),
        "timing_s": {
            "resident_pre_env_array_update": 0.0,
            "native_packing_materialization": 0.0,
            "native_packing_comparison": 0.0,
            "post_env_array_update": 0.0,
            "total": 0.0,
        },
    }
    record["timing_s"]["resident_pre_env_array_update"] = float(time.perf_counter() - t0)
    _native_pack_if_requested(runner, frame, agent_ids=ids, record=record)
    record["timing_s"]["total"] = float(time.perf_counter() - t0)
    _update_and_write_summary(runner, frame, record)
    return record


def commit_post_env_arrays(
    runner: Any,
    *,
    agent_ids_np: Any,
    new_energy: Any,
    new_health: Any,
    new_injury: Any,
    new_scar: Any,
    died_mask: Any,
) -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    if not resident_frame_requested(cfg):
        return None
    frame = getattr(runner, "_cxxrewrite3_resident_frame", None)
    if not isinstance(frame, CxxRewrite3ResidentFrame):
        return None
    t0 = time.perf_counter()
    ids = np.asarray(agent_ids_np, dtype=np.int32).reshape(-1)
    if ids.size and int(np.max(ids)) >= int(frame.capacity):
        _grow_frame(frame, int(np.max(ids)) + 1)
    frame.energy_by_id[ids] = np.asarray(new_energy, dtype=np.float32).reshape(-1)
    frame.health_by_id[ids] = np.asarray(new_health, dtype=np.float32).reshape(-1)
    frame.injury_by_id[ids] = np.asarray(new_injury, dtype=np.float32).reshape(-1)
    frame.scar_by_id[ids] = np.asarray(new_scar, dtype=np.float32).reshape(-1)
    frame.alive_by_id[ids] = np.logical_not(np.asarray(died_mask, dtype=bool).reshape(-1))
    env = getattr(runner, "env", None)
    if env is not None:
        pos_y = np.asarray(getattr(env, "agent_pos_y", frame.pos_y_by_id), dtype=np.int32)
        pos_x = np.asarray(getattr(env, "agent_pos_x", frame.pos_x_by_id), dtype=np.int32)
        alive_env = np.asarray(getattr(env, "agent_alive", frame.alive_by_id), dtype=bool)
        cap = max(int(frame.capacity), int(pos_y.shape[0]), int(pos_x.shape[0]), int(alive_env.shape[0]))
        if cap > int(frame.capacity):
            _grow_frame(frame, cap)
        frame.pos_y_by_id[: int(pos_y.shape[0])] = pos_y
        frame.pos_x_by_id[: int(pos_x.shape[0])] = pos_x
        frame.alive_by_id[: int(alive_env.shape[0])] = alive_env
        frame.env_dirty_by_id[: int(alive_env.shape[0])] = True
    frame.scalar_dirty_by_id[ids] = True
    frame.bump("post_env_array_updates", int(ids.shape[0]))
    record = {
        "schema_version": SCHEMA_VERSION,
        "generation": int(frame.generation),
        "tick": int(frame.tick),
        "agents": int(ids.shape[0]),
        "post_env_array_update": True,
        "timing_s": {"post_env_array_update": float(time.perf_counter() - t0)},
    }
    _update_and_write_summary(runner, frame, record)
    return record


def init_summary() -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "records": 0,
        "pre_env_records": 0,
        "post_env_records": 0,
        "agents": 0,
        "ok": True,
        "first_divergence": None,
        "resident_frame_used_records": 0,
        "pre_env_array_updates": 0,
        "pre_env_scalar_syncs_eliminated": 0,
        "post_env_array_updates": 0,
        "missing_agent_syncs": 0,
        "initial_agent_scalar_reads": 0,
        "native_input_agent_object_scalar_reads": 0,
        "native_pack_records": 0,
        "native_pack_skipped": 0,
        "native_pack_compare_checked": 0,
        "native_pack_compare_ok": True,
        "max_scalar_drift_vs_pre_env": 0.0,
        "timing_s": {
            "resident_pre_env_array_update": 0.0,
            "native_packing_materialization": 0.0,
            "native_packing_comparison": 0.0,
            "post_env_array_update": 0.0,
            "total": 0.0,
        },
    }


def update_summary(summary: dict[str, Any] | None, frame: CxxRewrite3ResidentFrame, record: dict[str, Any]) -> dict[str, Any]:
    if summary is None:
        summary = init_summary()
    summary["records"] = int(summary.get("records", 0)) + 1
    summary["agents"] = int(summary.get("agents", 0)) + int(record.get("agents", 0) or 0)
    if bool(record.get("resident_frame_used", False)):
        summary["resident_frame_used_records"] = int(summary.get("resident_frame_used_records", 0)) + 1
        summary["pre_env_records"] = int(summary.get("pre_env_records", 0)) + 1
    if bool(record.get("post_env_array_update", False)):
        summary["post_env_records"] = int(summary.get("post_env_records", 0)) + 1
    counters = dict(frame.counters)
    for name in (
        "pre_env_array_updates",
        "pre_env_scalar_syncs_eliminated",
        "post_env_array_updates",
        "missing_agent_syncs",
        "initial_agent_scalar_reads",
        "native_pack_records",
        "native_pack_skipped",
    ):
        summary[name] = int(counters.get(name, summary.get(name, 0)) or 0)
    summary["native_input_agent_object_scalar_reads"] = int(
        int(summary.get("native_input_agent_object_scalar_reads", 0) or 0)
        + int(record.get("native_input_agent_object_scalar_reads", 0) or 0)
    )
    if bool(record.get("native_pack_used", False)):
        compare = record.get("native_pack_compare")
        if isinstance(compare, dict):
            summary["native_pack_compare_checked"] = int(summary.get("native_pack_compare_checked", 0)) + 1
            if not bool(compare.get("ok", False)):
                summary["native_pack_compare_ok"] = False
                summary["ok"] = False
                if summary.get("first_divergence") is None:
                    summary["first_divergence"] = compare.get("first_divergence")
    if bool(record.get("native_pack_skipped", False)):
        summary["native_pack_skipped"] = int(counters.get("native_pack_skipped", summary.get("native_pack_skipped", 0)) or 0)
    summary["max_scalar_drift_vs_pre_env"] = float(
        max(
            float(summary.get("max_scalar_drift_vs_pre_env", 0.0) or 0.0),
            float(record.get("max_scalar_drift_vs_pre_env", 0.0) or 0.0),
            float(frame.max_scalar_drift_vs_pre_env),
        )
    )
    totals = dict(summary.get("timing_s", {}) or {})
    for key, value in dict(record.get("timing_s", {}) or {}).items():
        totals[str(key)] = float(totals.get(str(key), 0.0)) + float(value or 0.0)
    summary["timing_s"] = totals
    return summary


def write_summary(run_dir: Any, summary: dict[str, Any], *, filename: str = "cxxrewrite3_resident_frame_summary.json") -> Path | None:
    if not run_dir:
        return None
    out_dir = Path(str(run_dir)) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / str(filename)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(out_path)
    return out_path


def _update_and_write_summary(runner: Any, frame: CxxRewrite3ResidentFrame, record: dict[str, Any]) -> None:
    summary = update_summary(getattr(runner, "_cxxrewrite3_resident_frame_summary", None), frame, record)
    setattr(runner, "_cxxrewrite3_resident_frame_summary", summary)
    if bool(getattr(getattr(runner, "sim_cfg", None), "cxxrewrite_resident_frame_write_summary", True)):
        write_summary(
            getattr(runner, "run_dir", None),
            summary,
            filename=str(
                getattr(
                    getattr(runner, "sim_cfg", None),
                    "cxxrewrite_resident_frame_summary_filename",
                    "cxxrewrite3_resident_frame_summary.json",
                )
            ),
        )


def write_json(path: Path | str, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
