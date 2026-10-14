from __future__ import annotations

"""CXXWAVE1F C++ backbone contracts and ctypes adapter.

This module is intentionally default-off. It provides a composable primitive
state contract for a live-facing scalar backbone: drives, chemistry,
maintenance state, and event-style summary counters. Runner integration must
still prove exact support envelopes before claiming native live authority.
"""

from dataclasses import dataclass
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "cxxwave1f_cpp_backbone_v1"
SCALAR_BACKBONE_SCHEMA_VERSION = "cxxwave1f_scalar_backbone_v1"
SCALAR_EVENT_TAPE_SCHEMA_VERSION = "cxxwave1f_scalar_event_tape_v1"
SCALAR_CHECKPOINT_SCHEMA_VERSION = "cxxwave1f_scalar_checkpoint_v1"
SUPPORT_SCHEMA_VERSION = "cxxwave1f_support_gate_v1"
SCALAR_EVENT_SUMMARY_NAMES = ("emitted_rows", "skipped_rows", "energy_sum", "health_sum")
SCALAR_CHECKPOINT_F32_COLUMNS = 27
SCALAR_CHECKPOINT_I32_COLUMNS = 5
SUMMARY_NAMES = (
    "drive_rows",
    "chemistry_rows",
    "maintenance_rows",
    "maintenance_schedule_requests",
    "maintenance_energy_cost_sum",
    "maintenance_opportunity_cost_sum",
    "maintenance_apply_count",
    "maintenance_applied_effect_sum",
    "episode0_selfharm_events",
)


@dataclass(frozen=True)
class Cxxwave1fScalarConfig:
    max_energy: float = 10.0
    start_health: float = 10.0
    hunger_gain: float = 0.08
    hunger_decay: float = 0.02
    pain_gain: float = 0.18
    pain_decay: float = 0.03
    fatigue_gain: float = 0.05
    fatigue_rest_relief: float = 0.10
    fatigue_decay: float = 0.01
    fear_gain: float = 0.22
    fear_decay: float = 0.03
    episode0_k: int = 20
    maintenance_enable: bool = True
    maintenance_delay_ticks: int = 0
    maintenance_base_effect: float = 0.2
    maintenance_diminishing_factor: float = 0.5
    maintenance_recent_decay: float = 0.9
    maintenance_energy_cost: float = 0.1
    maintenance_opportunity_cost_coeff: float = 0.0
    tick_index: int = 4


@dataclass
class Cxxwave1fScalarState:
    slots: np.ndarray
    actions: np.ndarray
    alive: np.ndarray
    energy: np.ndarray
    health: np.ndarray
    injury: np.ndarray
    last_energy_delta: np.ndarray
    last_health_delta: np.ndarray
    last_injury_delta: np.ndarray
    drives: np.ndarray
    prev_drive_delta: np.ndarray
    drive_inertia: np.ndarray
    drive_noise_scale: np.ndarray
    drive_noise_samples: np.ndarray
    survival_steps: np.ndarray
    drive_volatility: np.ndarray
    drive_jitter: np.ndarray
    episode0_drive_jitter_sum: np.ndarray
    episode0_selfharm_events: np.ndarray
    chemistry: np.ndarray
    chemistry_noise: np.ndarray
    maintenance: np.ndarray
    maintenance_opportunity: np.ndarray
    maintenance_requests: np.ndarray
    maintenance_applied: np.ndarray
    maintenance_energy_spent_total: np.ndarray
    maintenance_opportunity_cost_total: np.ndarray
    maintenance_applied_effect_total: np.ndarray

    def copy(self) -> "Cxxwave1fScalarState":
        return Cxxwave1fScalarState(**{name: np.asarray(value).copy() for name, value in self.__dict__.items()})


def cxxwave1f_capability_matrix() -> dict[str, Any]:
    rows = [
        {
            "block": "Block 1 live dispatch",
            "surface": "generation/tick dispatch",
            "status": "scaffolded_fail_closed",
            "native_authority": "none until exact live hook passes support gate",
            "blocker": "candidate dispatch still depends on RunnerCore support envelope",
        },
        {
            "block": "Block 3 environment",
            "surface": "post-env scalar state and event counters",
            "status": "partial_native_scalar_backbone",
            "native_authority": "drives consume post-env deltas; environment movement/gather remains upstream",
            "blocker": "Patch/Jungle/WDL env arrays are not yet fully owned by this kernel",
        },
        {
            "block": "Block 4 chemistry/maintenance",
            "surface": "drive, chemistry, maintenance rows",
            "status": "native_owned_for_supported_scalar_envelope",
            "native_authority": "primitive arrays updated in C++ from resident rows and explicit noise tapes",
            "blocker": "nonzero prediction horizon and nonzero maintenance opportunity scan fail closed in live mode",
        },
        {
            "block": "Block 5 events/reporting",
            "surface": "typed scalar event tape",
            "status": "native_typed_scalar_tape",
            "native_authority": "primitive event columns emitted in C++; public JSONL/CSV remains Python compatibility boundary",
            "blocker": "lineage/public rows are not byte-compatible from native yet",
        },
        {
            "block": "Block 6 checkpoint",
            "surface": "primitive scalar arrays",
            "status": "native_scalar_pack_restore",
            "native_authority": "scalar member pack/restore rows are owned by C++ and parity checked against Python",
            "blocker": "full RunnerCore checkpoint bundle still has non-scalar Python-owned members",
        },
    ]
    return {
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "rows": rows,
        "summary": {
            "native_owned_blocks": 3,
            "partial_blocks": 2,
            "live_gate_default": "closed",
            "support_policy": "open only when live hook proves exact parity and checkpoint continuation",
        },
    }


def live_support_report(*, prediction_horizon: int, maintenance_opportunity_cost_coeff: float, library_path: str) -> dict[str, Any]:
    blockers: list[str] = []
    if not library_path:
        blockers.append("cxxwave1f_scalar_backbone_library_path_missing")
    if int(prediction_horizon) > 0:
        blockers.append("prediction_horizon_positive_reads_pre_maintenance_energy")
    if float(maintenance_opportunity_cost_coeff) != 0.0:
        blockers.append("maintenance_opportunity_score_still_python_grid_scan")
    return {
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "supported": not blockers,
        "blockers": blockers,
        "decision": "CXXWAVE1F-SCALAR-LIVE-SUPPORTED" if not blockers else "CXXWAVE1F-SCALAR-LIVE-FAIL-CLOSED",
    }


def compile_scalar_backbone_kernel(build_dir: str | Path, *, cxx: str | None = None) -> dict[str, Any]:
    compiler = cxx or shutil.which("g++")
    build = Path(build_dir)
    source_path = Path(__file__).resolve().parent / "native" / "cxxwave1f_scalar_backbone_kernel.cpp"
    library_path = build / "libcxxwave1f_scalar_backbone.so"
    info: dict[str, Any] = {
        "schema_version": "cxxwave1f_scalar_backbone_build_v1",
        "ok": False,
        "compiler": compiler or "",
        "source_path": str(source_path),
        "library_path": "",
        "command": [],
        "reason": "",
    }
    if not compiler:
        info["reason"] = "g++ not found"
        return info
    build.mkdir(parents=True, exist_ok=True)
    cmd = [str(compiler), "-O3", "-std=c++17", "-shared", "-fPIC", str(source_path), "-o", str(library_path)]
    cp = subprocess.run(cmd, text=True, capture_output=True, check=False)
    info.update(
        {
            "ok": cp.returncode == 0,
            "returncode": int(cp.returncode),
            "command": cmd,
            "stdout_tail": cp.stdout[-4000:],
            "stderr_tail": cp.stderr[-4000:],
            "library_path": str(library_path) if cp.returncode == 0 else "",
            "reason": "" if cp.returncode == 0 else "compile failed",
        }
    )
    return info


def load_scalar_backbone_kernel(library_path: str | Path):
    lib = ctypes.CDLL(str(library_path))
    fn = lib.cxxwave1f_apply_scalar_backbone_v1
    i32p = ctypes.POINTER(ctypes.c_int32)
    i64 = ctypes.c_int64
    u8p = ctypes.POINTER(ctypes.c_uint8)
    f32p = ctypes.POINTER(ctypes.c_float)
    f64p = ctypes.POINTER(ctypes.c_double)
    fn.argtypes = [
        i32p,
        i32p,
        f32p,
        f32p,
        f32p,
        i64,
        i64,
        u8p,
        f32p,
        f32p,
        f32p,
        f32p,
        f32p,
        f32p,
        f32p,
        f32p,
        f32p,
        f32p,
        i32p,
        f32p,
        f32p,
        f32p,
        i32p,
        f32p,
        f32p,
        i32p,
        i32p,
        f32p,
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
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int32,
        f64p,
    ]
    fn.restype = ctypes.c_int
    evt_fn = lib.cxxwave1f_emit_scalar_event_tape_v1
    evt_fn.argtypes = [
        i32p,
        i32p,
        i64,
        i64,
        u8p,
        f32p,
        f32p,
        f32p,
        f32p,
        f32p,
        i32p,
        i32p,
        ctypes.c_int32,
        i32p,
        i32p,
        i32p,
        i32p,
        f32p,
        f32p,
        f32p,
        f32p,
        f64p,
    ]
    evt_fn.restype = ctypes.c_int
    pack_fn = lib.cxxwave1f_pack_scalar_checkpoint_v1
    pack_fn.argtypes = [
        i64,
        u8p,
        f32p,
        f32p,
        f32p,
        f32p,
        f32p,
        i32p,
        f32p,
        f32p,
        f32p,
        i32p,
        f32p,
        f32p,
        i32p,
        i32p,
        f32p,
        f32p,
        f32p,
        f32p,
        i32p,
        f64p,
    ]
    pack_fn.restype = ctypes.c_int
    restore_fn = lib.cxxwave1f_restore_scalar_checkpoint_v1
    restore_fn.argtypes = [
        i64,
        f32p,
        i32p,
        u8p,
        f32p,
        f32p,
        f32p,
        f32p,
        f32p,
        i32p,
        f32p,
        f32p,
        f32p,
        i32p,
        f32p,
        f32p,
        i32p,
        i32p,
        f32p,
        f32p,
        f32p,
        f64p,
    ]
    restore_fn.restype = ctypes.c_int

    def run(state: Cxxwave1fScalarState, cfg: Cxxwave1fScalarConfig) -> dict[str, Any]:
        summary = np.zeros((len(SUMMARY_NAMES),), dtype=np.float64)
        rc = fn(
            _i32(state.slots),
            _i32(state.actions),
            _f32(state.chemistry_noise),
            _f32(state.drive_noise_samples),
            _f32(state.maintenance_opportunity),
            ctypes.c_int64(int(state.slots.shape[0])),
            ctypes.c_int64(int(state.alive.shape[0])),
            _u8(state.alive),
            _f32(state.energy),
            _f32(state.health),
            _f32(state.injury),
            _f32(state.last_energy_delta),
            _f32(state.last_health_delta),
            _f32(state.last_injury_delta),
            _f32(state.drives),
            _f32(state.prev_drive_delta),
            _f32(state.drive_inertia),
            _f32(state.drive_noise_scale),
            _i32(state.survival_steps),
            _f32(state.drive_volatility),
            _f32(state.drive_jitter),
            _f32(state.episode0_drive_jitter_sum),
            _i32(state.episode0_selfharm_events),
            _f32(state.chemistry),
            _f32(state.maintenance),
            _i32(state.maintenance_requests),
            _i32(state.maintenance_applied),
            _f32(state.maintenance_energy_spent_total),
            _f32(state.maintenance_opportunity_cost_total),
            _f32(state.maintenance_applied_effect_total),
            ctypes.c_float(float(cfg.max_energy)),
            ctypes.c_float(float(cfg.start_health)),
            ctypes.c_float(float(cfg.hunger_gain)),
            ctypes.c_float(float(cfg.hunger_decay)),
            ctypes.c_float(float(cfg.pain_gain)),
            ctypes.c_float(float(cfg.pain_decay)),
            ctypes.c_float(float(cfg.fatigue_gain)),
            ctypes.c_float(float(cfg.fatigue_rest_relief)),
            ctypes.c_float(float(cfg.fatigue_decay)),
            ctypes.c_float(float(cfg.fear_gain)),
            ctypes.c_float(float(cfg.fear_decay)),
            ctypes.c_int32(int(cfg.episode0_k)),
            ctypes.c_int32(1 if cfg.maintenance_enable else 0),
            ctypes.c_int32(int(cfg.maintenance_delay_ticks)),
            ctypes.c_float(float(cfg.maintenance_base_effect)),
            ctypes.c_float(float(cfg.maintenance_diminishing_factor)),
            ctypes.c_float(float(cfg.maintenance_recent_decay)),
            ctypes.c_float(float(cfg.maintenance_energy_cost)),
            ctypes.c_float(float(cfg.maintenance_opportunity_cost_coeff)),
            ctypes.c_int32(int(cfg.tick_index)),
            summary.ctypes.data_as(f64p),
        )
        if int(rc) != 0:
            raise RuntimeError(f"cxxwave1f scalar backbone failed with code {int(rc)}")
        return {"schema_version": SCALAR_BACKBONE_SCHEMA_VERSION, "summary": _summary_dict(summary)}

    def emit_event_tape(state: Cxxwave1fScalarState, *, tick_index: int) -> dict[str, Any]:
        rows = int(state.slots.shape[0])
        event_type = np.zeros((rows,), dtype=np.int32)
        event_tick = np.zeros((rows,), dtype=np.int32)
        event_slot = np.zeros((rows,), dtype=np.int32)
        event_action = np.zeros((rows,), dtype=np.int32)
        value0 = np.zeros((rows,), dtype=np.float32)
        value1 = np.zeros((rows,), dtype=np.float32)
        value2 = np.zeros((rows,), dtype=np.float32)
        value3 = np.zeros((rows,), dtype=np.float32)
        summary = np.zeros((len(SCALAR_EVENT_SUMMARY_NAMES),), dtype=np.float64)
        rc = evt_fn(
            _i32(state.slots),
            _i32(state.actions),
            ctypes.c_int64(rows),
            ctypes.c_int64(int(state.alive.shape[0])),
            _u8(state.alive),
            _f32(state.energy),
            _f32(state.health),
            _f32(state.injury),
            _f32(state.drives),
            _f32(state.chemistry),
            _i32(state.maintenance_requests),
            _i32(state.maintenance_applied),
            ctypes.c_int32(int(tick_index)),
            event_type.ctypes.data_as(i32p),
            event_tick.ctypes.data_as(i32p),
            event_slot.ctypes.data_as(i32p),
            event_action.ctypes.data_as(i32p),
            value0.ctypes.data_as(f32p),
            value1.ctypes.data_as(f32p),
            value2.ctypes.data_as(f32p),
            value3.ctypes.data_as(f32p),
            summary.ctypes.data_as(f64p),
        )
        if int(rc) != 0:
            raise RuntimeError(f"cxxwave1f scalar event tape failed with code {int(rc)}")
        return {
            "schema_version": SCALAR_EVENT_TAPE_SCHEMA_VERSION,
            "event_type": event_type,
            "tick": event_tick,
            "slot": event_slot,
            "action": event_action,
            "value0": value0,
            "value1": value1,
            "value2": value2,
            "value3": value3,
            "summary": {name: float(summary[idx]) for idx, name in enumerate(SCALAR_EVENT_SUMMARY_NAMES)},
        }

    def pack_checkpoint(state: Cxxwave1fScalarState) -> dict[str, Any]:
        capacity = int(state.alive.shape[0])
        packed_f32 = np.zeros((capacity, SCALAR_CHECKPOINT_F32_COLUMNS), dtype=np.float32)
        packed_i32 = np.zeros((capacity, SCALAR_CHECKPOINT_I32_COLUMNS), dtype=np.int32)
        summary = np.zeros((3,), dtype=np.float64)
        rc = pack_fn(
            ctypes.c_int64(capacity),
            _u8(state.alive),
            _f32(state.energy),
            _f32(state.health),
            _f32(state.injury),
            _f32(state.drives),
            _f32(state.prev_drive_delta),
            _i32(state.survival_steps),
            _f32(state.drive_volatility),
            _f32(state.drive_jitter),
            _f32(state.episode0_drive_jitter_sum),
            _i32(state.episode0_selfharm_events),
            _f32(state.chemistry),
            _f32(state.maintenance),
            _i32(state.maintenance_requests),
            _i32(state.maintenance_applied),
            _f32(state.maintenance_energy_spent_total),
            _f32(state.maintenance_opportunity_cost_total),
            _f32(state.maintenance_applied_effect_total),
            packed_f32.ctypes.data_as(f32p),
            packed_i32.ctypes.data_as(i32p),
            summary.ctypes.data_as(f64p),
        )
        if int(rc) != 0:
            raise RuntimeError(f"cxxwave1f scalar checkpoint pack failed with code {int(rc)}")
        return {
            "schema_version": SCALAR_CHECKPOINT_SCHEMA_VERSION,
            "packed_f32": packed_f32,
            "packed_i32": packed_i32,
            "summary": {"capacity": float(summary[0]), "f32_checksum": float(summary[1]), "i32_checksum": float(summary[2])},
        }

    def restore_checkpoint(state: Cxxwave1fScalarState, payload: Mapping[str, Any]) -> dict[str, Any]:
        packed_f32 = np.ascontiguousarray(payload["packed_f32"], dtype=np.float32)
        packed_i32 = np.ascontiguousarray(payload["packed_i32"], dtype=np.int32)
        capacity = int(state.alive.shape[0])
        if tuple(packed_f32.shape) != (capacity, SCALAR_CHECKPOINT_F32_COLUMNS):
            raise ValueError("packed_f32 shape does not match state capacity")
        if tuple(packed_i32.shape) != (capacity, SCALAR_CHECKPOINT_I32_COLUMNS):
            raise ValueError("packed_i32 shape does not match state capacity")
        summary = np.zeros((2,), dtype=np.float64)
        rc = restore_fn(
            ctypes.c_int64(capacity),
            packed_f32.ctypes.data_as(f32p),
            packed_i32.ctypes.data_as(i32p),
            _u8(state.alive),
            _f32(state.energy),
            _f32(state.health),
            _f32(state.injury),
            _f32(state.drives),
            _f32(state.prev_drive_delta),
            _i32(state.survival_steps),
            _f32(state.drive_volatility),
            _f32(state.drive_jitter),
            _f32(state.episode0_drive_jitter_sum),
            _i32(state.episode0_selfharm_events),
            _f32(state.chemistry),
            _f32(state.maintenance),
            _i32(state.maintenance_requests),
            _i32(state.maintenance_applied),
            _f32(state.maintenance_energy_spent_total),
            _f32(state.maintenance_opportunity_cost_total),
            _f32(state.maintenance_applied_effect_total),
            summary.ctypes.data_as(f64p),
        )
        if int(rc) != 0:
            raise RuntimeError(f"cxxwave1f scalar checkpoint restore failed with code {int(rc)}")
        return {"schema_version": SCALAR_CHECKPOINT_SCHEMA_VERSION, "summary": {"capacity": float(summary[0]), "alive_count": float(summary[1])}}

    run.emit_event_tape = emit_event_tape  # type: ignore[attr-defined]
    run.pack_checkpoint = pack_checkpoint  # type: ignore[attr-defined]
    run.restore_checkpoint = restore_checkpoint  # type: ignore[attr-defined]
    return run


def run_scalar_backbone_reference(state: Cxxwave1fScalarState, cfg: Cxxwave1fScalarConfig) -> dict[str, Any]:
    summary = np.zeros((len(SUMMARY_NAMES),), dtype=np.float64)
    e_norm = np.float32(max(float(cfg.max_energy), 1.0e-6))
    h_norm = np.float32(max(float(cfg.start_health), 1.0e-6))
    for row, slot_raw in enumerate(np.asarray(state.slots, dtype=np.int32).tolist()):
        slot = int(slot_raw)
        if slot < 0 or slot >= int(state.alive.shape[0]) or not bool(state.alive[slot]):
            continue
        action = int(state.actions[row])
        prev = state.drives[slot].astype(np.float32, copy=True)
        e_v = np.float32(np.clip(state.energy[slot] / e_norm, 0.0, 1.0))
        h_v = np.float32(np.clip(state.health[slot] / h_norm, 0.0, 1.0))
        de = np.float32(state.last_energy_delta[slot])
        dh = np.float32(state.last_health_delta[slot])
        dinj = np.float32(state.last_injury_delta[slot])
        inertia = np.float32(state.drive_inertia[slot])
        target_hunger = np.float32(prev[0] + np.float32(cfg.hunger_gain) * (np.float32(1.0) - e_v))
        target_hunger = np.float32(target_hunger + np.float32(0.04) * (-de / e_norm) - np.float32(cfg.hunger_decay))
        if action == 4 and de > np.float32(0.0):
            target_hunger = np.float32(target_hunger - np.float32(0.20) * (de / e_norm))
        target_hunger = np.float32(np.clip(target_hunger, 0.0, 1.0))

        target_pain = np.float32(prev[1])
        if dinj > np.float32(0.0):
            target_pain = np.float32(target_pain + np.float32(cfg.pain_gain) * min(np.float32(1.0), dinj))
        if dh < np.float32(0.0):
            target_pain = np.float32(target_pain + np.float32(0.10) * min(np.float32(1.0), -dh / h_norm))
        target_pain = np.float32(np.clip(target_pain - np.float32(cfg.pain_decay), 0.0, 1.0))

        target_fatigue = np.float32(prev[2] - np.float32(cfg.fatigue_rest_relief) if action == 5 else prev[2] + np.float32(cfg.fatigue_gain))
        target_fatigue = np.float32(np.clip(target_fatigue - np.float32(cfg.fatigue_decay), 0.0, 1.0))

        target_fear = np.float32(prev[3])
        if dinj > np.float32(0.0):
            target_fear = np.float32(target_fear + np.float32(cfg.fear_gain) * min(np.float32(1.0), dinj))
        target_fear = np.float32(np.clip(target_fear + np.float32(0.05) * (np.float32(1.0) - h_v) - np.float32(cfg.fear_decay), 0.0, 1.0))

        inv = np.float32(1.0) - inertia
        new = np.asarray(
            [
                inertia * prev[0] + inv * target_hunger,
                inertia * prev[1] + inv * target_pain,
                inertia * prev[2] + inv * target_fatigue,
                inertia * prev[3] + inv * target_fear,
            ],
            dtype=np.float32,
        )
        if float(state.drive_noise_scale[slot]) > 0.0:
            new = new + np.float32(state.drive_noise_scale[slot]) * state.drive_noise_samples[row]
        np.clip(new, 0.0, 1.0, out=new)
        delta = new - prev
        jitter = np.sum(np.abs(delta - state.prev_drive_delta[slot]), dtype=np.float32)
        state.drives[slot] = new
        state.prev_drive_delta[slot] = delta
        state.drive_volatility[slot] += np.float32(np.sum(np.abs(delta), dtype=np.float32))
        state.drive_jitter[slot] += np.float32(jitter)
        if int(state.survival_steps[slot]) <= int(cfg.episode0_k):
            state.episode0_drive_jitter_sum[slot] += np.float32(jitter)
            if action == 6 and float(dinj) > 0.0:
                state.episode0_selfharm_events[slot] += np.int32(1)
                summary[8] += 1.0
        summary[0] += 1.0

        c = state.chemistry[slot]
        energy_n = np.float32(np.clip(state.energy[slot] / e_norm, 0.0, 1.0))
        health_n = np.float32(np.clip(state.health[slot] / h_norm, 0.0, 1.0))
        injury_n = np.float32(np.clip(state.injury[slot] / h_norm, 0.0, 1.0))
        stress = np.float32(np.clip(np.float32(0.55) * injury_n + np.float32(0.30) * state.drives[slot, 1] + np.float32(0.15) * (np.float32(1.0) - energy_n), 0.0, 1.0))
        energy_store, repair, acute, chronic, reg1, reg2 = [np.float32(x) for x in c.tolist()]
        acute_target = np.float32(np.clip(np.float32(1.25) * injury_n + np.float32(0.45) * stress + np.float32(0.20) * (np.float32(1.0) - health_n), 0.0, 1.0))
        acute = np.float32(np.clip(acute + np.float32(0.50) * (acute_target - acute) + np.float32(0.03) * reg1, 0.0, 1.0))
        chronic_target = np.float32(np.clip(np.float32(0.75) * chronic + np.float32(0.25) * acute, 0.0, 1.0))
        chronic = np.float32(np.clip(chronic + np.float32(0.07) * (chronic_target - chronic), 0.0, 1.0))
        energy_store = np.float32(np.clip(energy_store + np.float32(0.03) * energy_n - np.float32(0.02) * stress - np.float32(0.01) * acute, 0.0, 1.0))
        repair = np.float32(np.clip(repair + np.float32(0.04) * energy_n - np.float32(0.03) * acute - np.float32(0.01) * chronic, 0.0, 1.0))
        reg1 = np.float32(np.clip(np.float32(0.92) * reg1 + np.float32(0.08) * (np.float32(1.0) - acute) + np.float32(0.01) * state.chemistry_noise[row, 0], 0.0, 1.0))
        reg2 = np.float32(np.clip(np.float32(0.97) * reg2 + np.float32(0.03) * chronic + np.float32(0.01) * state.chemistry_noise[row, 1], 0.0, 1.0))
        acute_resolve = min(acute, np.float32(0.10) * repair + np.float32(0.04) * energy_store)
        acute = np.float32(np.clip(acute - acute_resolve, 0.0, 1.0))
        chronic = np.float32(np.clip(chronic + np.float32(0.25) * acute - np.float32(0.002) * repair, 0.0, 1.0))
        state.chemistry[slot] = np.asarray((energy_store, repair, acute, chronic, reg1, reg2), dtype=np.float32)
        summary[1] += 1.0

        if cfg.maintenance_enable:
            m = state.maintenance[slot]
            m[2] = np.float32(np.clip(m[2] * np.float32(cfg.maintenance_recent_decay), 0.0, 1.0))
            if action == 5:
                summary[3] += 1.0
                opp_cost = max(0.0, float(cfg.maintenance_opportunity_cost_coeff)) * float(state.maintenance_opportunity[row])
                total_cost = max(0.0, float(cfg.maintenance_energy_cost)) + opp_cost
                if total_cost > 0.0:
                    state.energy[slot] = np.float32(max(0.0, float(state.energy[slot]) - total_cost))
                    summary[4] += total_cost
                    summary[5] += opp_cost
                    state.maintenance_energy_spent_total[slot] += np.float32(total_cost)
                    state.maintenance_opportunity_cost_total[slot] += np.float32(opp_cost)
                m[0] = np.float32(np.clip(m[0] + np.float32(cfg.maintenance_base_effect), 0.0, 1.0))
                m[1] = np.float32(float(int(cfg.tick_index) + int(cfg.maintenance_delay_ticks)))
                m[3] = np.float32(float(int(cfg.tick_index)))
                state.maintenance_requests[slot] += np.int32(1)
            pending = float(m[0])
            due_tick = int(round(float(m[1])))
            if pending > 0.0 and due_tick >= 0 and int(cfg.tick_index) >= due_tick:
                recent_load = float(m[2])
                applied = float(np.clip(pending / (1.0 + max(0.0, float(cfg.maintenance_diminishing_factor)) * recent_load), 0.0, 1.0))
                state.chemistry[slot, 1] = np.float32(np.clip(state.chemistry[slot, 1] + 0.65 * applied, 0.0, 1.0))
                state.chemistry[slot, 2] = np.float32(np.clip(state.chemistry[slot, 2] - 0.55 * applied, 0.0, 1.0))
                state.chemistry[slot, 3] = np.float32(np.clip(state.chemistry[slot, 3] - 0.25 * applied, 0.0, 1.0))
                m[0] = np.float32(0.0)
                m[1] = np.float32(-1.0)
                m[2] = np.float32(np.clip(recent_load + applied, 0.0, 1.0))
                state.maintenance_applied[slot] += np.int32(1)
                state.maintenance_applied_effect_total[slot] += np.float32(applied)
                summary[6] += 1.0
                summary[7] += applied
            summary[2] += 1.0
    return {"schema_version": SCALAR_BACKBONE_SCHEMA_VERSION, "summary": _summary_dict(summary)}


def emit_scalar_event_tape_reference(state: Cxxwave1fScalarState, *, tick_index: int) -> dict[str, Any]:
    rows = int(state.slots.shape[0])
    event_type = np.zeros((rows,), dtype=np.int32)
    event_tick = np.zeros((rows,), dtype=np.int32)
    event_slot = np.zeros((rows,), dtype=np.int32)
    event_action = np.zeros((rows,), dtype=np.int32)
    value0 = np.zeros((rows,), dtype=np.float32)
    value1 = np.zeros((rows,), dtype=np.float32)
    value2 = np.zeros((rows,), dtype=np.float32)
    value3 = np.zeros((rows,), dtype=np.float32)
    summary = np.zeros((len(SCALAR_EVENT_SUMMARY_NAMES),), dtype=np.float64)
    for row, slot_raw in enumerate(np.asarray(state.slots, dtype=np.int32).tolist()):
        slot = int(slot_raw)
        action = int(state.actions[row])
        event_tick[row] = np.int32(tick_index)
        event_slot[row] = np.int32(slot)
        event_action[row] = np.int32(action)
        if slot < 0 or slot >= int(state.alive.shape[0]) or not bool(state.alive[slot]):
            summary[1] += 1.0
            continue
        event_type[row] = np.int32(4100 + action)
        value0[row] = np.float32(state.energy[slot])
        value1[row] = np.float32(state.health[slot])
        value2[row] = np.float32(np.sum(state.drives[slot], dtype=np.float32))
        value3[row] = np.float32(
            np.float32(state.chemistry[slot, 2] + state.chemistry[slot, 3])
            + np.float32(int(state.maintenance_requests[slot]) + int(state.maintenance_applied[slot])) * np.float32(0.001)
        )
        summary[0] += 1.0
        summary[2] += float(state.energy[slot])
        summary[3] += float(state.health[slot])
    return {
        "schema_version": SCALAR_EVENT_TAPE_SCHEMA_VERSION,
        "event_type": event_type,
        "tick": event_tick,
        "slot": event_slot,
        "action": event_action,
        "value0": value0,
        "value1": value1,
        "value2": value2,
        "value3": value3,
        "summary": {name: float(summary[idx]) for idx, name in enumerate(SCALAR_EVENT_SUMMARY_NAMES)},
    }


def compare_scalar_event_tapes(a: Mapping[str, Any], b: Mapping[str, Any], *, atol: float = 0.0) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for name in ("event_type", "tick", "slot", "action", "value0", "value1", "value2", "value3"):
        av = np.asarray(a[name])
        bv = np.asarray(b[name])
        if av.dtype.kind == "f" or bv.dtype.kind == "f":
            ok = bool(np.allclose(av, bv, atol=atol, rtol=0.0))
        else:
            ok = bool(np.array_equal(av, bv))
        if not ok:
            mismatches.append({"field": name, "a_digest": _array_digest(av), "b_digest": _array_digest(bv)})
    return {
        "schema_version": "cxxwave1f_scalar_event_tape_compare_v1",
        "ok": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:16],
        "digest": stable_digest({name: _array_digest(b[name]) for name in ("event_type", "tick", "slot", "action", "value0", "value1", "value2", "value3")}),
    }


def pack_scalar_checkpoint_reference(state: Cxxwave1fScalarState) -> dict[str, Any]:
    capacity = int(state.alive.shape[0])
    packed_f32 = np.zeros((capacity, SCALAR_CHECKPOINT_F32_COLUMNS), dtype=np.float32)
    packed_i32 = np.zeros((capacity, SCALAR_CHECKPOINT_I32_COLUMNS), dtype=np.int32)
    packed_f32[:, 0] = state.energy
    packed_f32[:, 1] = state.health
    packed_f32[:, 2] = state.injury
    packed_f32[:, 3:7] = state.drives
    packed_f32[:, 7:11] = state.prev_drive_delta
    packed_f32[:, 11] = state.drive_volatility
    packed_f32[:, 12] = state.drive_jitter
    packed_f32[:, 13] = state.episode0_drive_jitter_sum
    packed_f32[:, 14:20] = state.chemistry
    packed_f32[:, 20:24] = state.maintenance
    packed_f32[:, 24] = state.maintenance_energy_spent_total
    packed_f32[:, 25] = state.maintenance_opportunity_cost_total
    packed_f32[:, 26] = state.maintenance_applied_effect_total
    packed_i32[:, 0] = np.asarray(state.alive, dtype=np.int32)
    packed_i32[:, 1] = state.survival_steps
    packed_i32[:, 2] = state.episode0_selfharm_events
    packed_i32[:, 3] = state.maintenance_requests
    packed_i32[:, 4] = state.maintenance_applied
    f_weights = np.arange(1, SCALAR_CHECKPOINT_F32_COLUMNS + 1, dtype=np.float64).reshape((1, -1))
    i_weights = np.arange(1, SCALAR_CHECKPOINT_I32_COLUMNS + 1, dtype=np.float64).reshape((1, -1))
    return {
        "schema_version": SCALAR_CHECKPOINT_SCHEMA_VERSION,
        "packed_f32": packed_f32,
        "packed_i32": packed_i32,
        "summary": {
            "capacity": float(capacity),
            "f32_checksum": float(np.sum(packed_f32.astype(np.float64) * f_weights)),
            "i32_checksum": float(np.sum(packed_i32.astype(np.float64) * i_weights)),
        },
    }


def compare_scalar_checkpoints(a: Mapping[str, Any], b: Mapping[str, Any], *, atol: float = 0.0) -> dict[str, Any]:
    f_ok = bool(np.allclose(np.asarray(a["packed_f32"]), np.asarray(b["packed_f32"]), atol=atol, rtol=0.0))
    i_ok = bool(np.array_equal(np.asarray(a["packed_i32"]), np.asarray(b["packed_i32"])))
    return {
        "schema_version": "cxxwave1f_scalar_checkpoint_compare_v1",
        "ok": bool(f_ok and i_ok),
        "f32_digest": _array_digest(b["packed_f32"]),
        "i32_digest": _array_digest(b["packed_i32"]),
        "mismatches": [name for name, ok in (("packed_f32", f_ok), ("packed_i32", i_ok)) if not ok],
    }


def compare_scalar_states(a: Cxxwave1fScalarState, b: Cxxwave1fScalarState, *, atol: float = 1.0e-6) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for name in a.__dict__:
        av = np.asarray(getattr(a, name))
        bv = np.asarray(getattr(b, name))
        if av.dtype.kind in {"f"}:
            ok = bool(np.allclose(av, bv, atol=atol, rtol=0.0))
        else:
            ok = bool(np.array_equal(av, bv))
        if not ok:
            mismatches.append({"field": name, "a_digest": _array_digest(av), "b_digest": _array_digest(bv)})
    return {
        "schema_version": "cxxwave1f_scalar_backbone_compare_v1",
        "ok": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:16],
    }


def sample_scalar_backbone_state(*, rows: int = 16, capacity: int = 20) -> Cxxwave1fScalarState:
    rng = np.random.default_rng(11051)
    slots = np.arange(rows, dtype=np.int32) % capacity
    actions = np.asarray([0, 1, 2, 3, 4, 5, 6, 5] * ((rows + 7) // 8), dtype=np.int32)[:rows]
    alive = np.ones((capacity,), dtype=np.uint8)
    energy = rng.uniform(1.0, 9.0, size=capacity).astype(np.float32)
    health = rng.uniform(3.0, 10.0, size=capacity).astype(np.float32)
    injury = rng.uniform(0.0, 2.0, size=capacity).astype(np.float32)
    return Cxxwave1fScalarState(
        slots=np.ascontiguousarray(slots),
        actions=np.ascontiguousarray(actions),
        alive=np.ascontiguousarray(alive),
        energy=np.ascontiguousarray(energy),
        health=np.ascontiguousarray(health),
        injury=np.ascontiguousarray(injury),
        last_energy_delta=np.ascontiguousarray(rng.uniform(-0.6, 0.8, size=capacity).astype(np.float32)),
        last_health_delta=np.ascontiguousarray(rng.uniform(-0.3, 0.1, size=capacity).astype(np.float32)),
        last_injury_delta=np.ascontiguousarray(rng.uniform(0.0, 0.3, size=capacity).astype(np.float32)),
        drives=np.ascontiguousarray(rng.uniform(0.0, 1.0, size=(capacity, 4)).astype(np.float32)),
        prev_drive_delta=np.ascontiguousarray(rng.uniform(-0.05, 0.05, size=(capacity, 4)).astype(np.float32)),
        drive_inertia=np.ascontiguousarray(rng.uniform(0.1, 0.8, size=capacity).astype(np.float32)),
        drive_noise_scale=np.ascontiguousarray(np.zeros((capacity,), dtype=np.float32)),
        drive_noise_samples=np.ascontiguousarray(rng.normal(0.0, 1.0, size=(rows, 4)).astype(np.float32)),
        survival_steps=np.ascontiguousarray(rng.integers(0, 30, size=capacity, dtype=np.int32)),
        drive_volatility=np.ascontiguousarray(rng.uniform(0.0, 1.0, size=capacity).astype(np.float32)),
        drive_jitter=np.ascontiguousarray(rng.uniform(0.0, 1.0, size=capacity).astype(np.float32)),
        episode0_drive_jitter_sum=np.ascontiguousarray(rng.uniform(0.0, 1.0, size=capacity).astype(np.float32)),
        episode0_selfharm_events=np.ascontiguousarray(rng.integers(0, 3, size=capacity, dtype=np.int32)),
        chemistry=np.ascontiguousarray(rng.uniform(0.0, 1.0, size=(capacity, 6)).astype(np.float32)),
        chemistry_noise=np.ascontiguousarray(rng.normal(0.0, 1.0, size=(rows, 2)).astype(np.float32)),
        maintenance=np.ascontiguousarray(rng.uniform(0.0, 1.0, size=(capacity, 4)).astype(np.float32)),
        maintenance_opportunity=np.ascontiguousarray(np.zeros((rows,), dtype=np.float32)),
        maintenance_requests=np.ascontiguousarray(rng.integers(0, 3, size=capacity, dtype=np.int32)),
        maintenance_applied=np.ascontiguousarray(rng.integers(0, 3, size=capacity, dtype=np.int32)),
        maintenance_energy_spent_total=np.ascontiguousarray(rng.uniform(0.0, 1.0, size=capacity).astype(np.float32)),
        maintenance_opportunity_cost_total=np.ascontiguousarray(rng.uniform(0.0, 1.0, size=capacity).astype(np.float32)),
        maintenance_applied_effect_total=np.ascontiguousarray(rng.uniform(0.0, 1.0, size=capacity).astype(np.float32)),
    )


def scalar_state_payload(state: Cxxwave1fScalarState) -> dict[str, Any]:
    return {
        "schema_version": SCALAR_BACKBONE_SCHEMA_VERSION,
        "rows": int(state.slots.shape[0]),
        "capacity": int(state.alive.shape[0]),
        "digest": stable_digest({name: _array_digest(value) for name, value in sorted(state.__dict__.items())}),
    }


def _summary_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(values[idx]) for idx, name in enumerate(SUMMARY_NAMES)}


def _array_digest(value: Any) -> str:
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def stable_digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _i32(value: Any):
    return np.ascontiguousarray(value, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _u8(value: Any):
    return np.ascontiguousarray(value, dtype=np.uint8).ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def _f32(value: Any):
    return np.ascontiguousarray(value, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))


__all__ = [
    "Cxxwave1fScalarConfig",
    "Cxxwave1fScalarState",
    "SCALAR_CHECKPOINT_SCHEMA_VERSION",
    "SCALAR_EVENT_TAPE_SCHEMA_VERSION",
    "SCALAR_BACKBONE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SUMMARY_NAMES",
    "compare_scalar_checkpoints",
    "compare_scalar_event_tapes",
    "compare_scalar_states",
    "compile_scalar_backbone_kernel",
    "cxxwave1f_capability_matrix",
    "emit_scalar_event_tape_reference",
    "live_support_report",
    "load_scalar_backbone_kernel",
    "pack_scalar_checkpoint_reference",
    "run_scalar_backbone_reference",
    "sample_scalar_backbone_state",
    "scalar_state_payload",
]
