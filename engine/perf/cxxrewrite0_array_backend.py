from __future__ import annotations

"""CXXREWRITE-0 array-backend ownership scaffold.

Default-off engineering scaffold only. This module extracts the currently
Python-authoritative live runner state into a slot-indexed SoA snapshot and
validates that the copied arrays still match the live Python/env owners.

It does not mutate simulation state. Authoritative array mode is intentionally
blocked until churn ownership and sync/writeback are implemented.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np


SCHEMA_VERSION = "cxxrewrite0_array_backend_shadow_v1"
OWNERSHIP_SCHEMA_VERSION = "cxxrewrite0_ownership_map_v1"
N_ACTIONS = 7
N_DRIVES = 4


@dataclass(frozen=True)
class ArrayBackendSnapshot:
    schema_version: str
    generation: int
    phase: str
    slot_to_agent_id: np.ndarray
    slot_tokens: np.ndarray
    alive_mask: np.ndarray
    agent_ids: np.ndarray
    agent_id_to_slot_agent_ids: np.ndarray
    agent_id_to_slot_slots: np.ndarray
    pos_y: np.ndarray
    pos_x: np.ndarray
    energy: np.ndarray
    health: np.ndarray
    injury: np.ndarray
    age: np.ndarray
    hmem: np.ndarray
    drives: np.ndarray
    exp_dE: np.ndarray
    exp_dInj: np.ndarray
    action_buffer: np.ndarray
    observation_buffer: np.ndarray
    lineage_ids_by_slot: tuple[str, ...]
    founder_ids_by_slot: tuple[int, ...]
    churn_tape: tuple[dict[str, Any], ...]


def build_array_backend_snapshot(
    runner: Any,
    alive: Any = None,
    *,
    generation: int = -1,
    phase: str = "start_generation",
) -> ArrayBackendSnapshot:
    agents = _alive_agents(runner, alive)
    max_slot = max([_safe_int(getattr(a, "slot", -1), -1) for a in agents] + [-1])
    pop_max_slots = _safe_int(getattr(getattr(runner, "pop", None), "max_slots", 0), 0)
    cap = max(1, int(pop_max_slots), int(max_slot) + 1)

    slot_to_agent_id = np.full((cap,), -1, dtype=np.int64)
    slot_tokens = np.full((cap,), -1, dtype=np.int64)
    alive_mask = np.zeros((cap,), dtype=np.bool_)
    pos_y = np.full((cap,), -1, dtype=np.int32)
    pos_x = np.full((cap,), -1, dtype=np.int32)
    energy = np.zeros((cap,), dtype=np.float32)
    health = np.zeros((cap,), dtype=np.float32)
    injury = np.zeros((cap,), dtype=np.float32)
    age = np.zeros((cap,), dtype=np.int32)
    drives = np.zeros((cap, N_DRIVES), dtype=np.float32)
    exp_dE = np.zeros((cap, N_ACTIONS), dtype=np.float32)
    exp_dInj = np.zeros((cap, N_ACTIONS), dtype=np.float32)
    action_buffer = np.full((cap,), -1, dtype=np.int32)
    observation_buffer = np.empty((cap, 0), dtype=np.float32)
    lineage_ids = [""] * cap
    founder_ids = [-1] * cap

    env = getattr(runner, "env", None)
    env_pos_y = _env_array(env, "agent_pos_y")
    env_pos_x = _env_array(env, "agent_pos_x")
    env_alive = _env_array(env, "agent_alive")

    by_agent_id: list[tuple[int, int]] = []
    for agent in sorted(agents, key=lambda a: (_safe_int(getattr(a, "slot", -1), -1), _safe_int(getattr(a, "agent_id", -1), -1))):
        aid = _safe_int(getattr(agent, "agent_id", -1), -1)
        slot = _safe_int(getattr(agent, "slot", -1), -1)
        if aid < 0 or slot < 0 or slot >= cap:
            continue
        slot_to_agent_id[slot] = np.int64(aid)
        slot_tokens[slot] = np.int64(_safe_int(getattr(agent, "slot_token", 0), 0))
        alive_mask[slot] = True
        pos_y[slot] = np.int32(_position_value(env_pos_y, aid, getattr(agent, "y", -1)))
        pos_x[slot] = np.int32(_position_value(env_pos_x, aid, getattr(agent, "x", -1)))
        energy[slot] = np.float32(_safe_float(getattr(agent, "energy", 0.0), 0.0))
        health[slot] = np.float32(_safe_float(getattr(agent, "health", 0.0), 0.0))
        injury[slot] = np.float32(_safe_float(getattr(agent, "injury", 0.0), 0.0))
        age[slot] = np.int32(_safe_int(getattr(agent, "age", getattr(agent, "steps_alive", 0)), 0))

        d = np.asarray(getattr(agent, "drives", np.zeros((N_DRIVES,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        drives[slot, : min(N_DRIVES, d.shape[0])] = d[:N_DRIVES]

        stats = getattr(agent, "stats", None)
        de = np.asarray(getattr(stats, "exp_dE", np.zeros((N_ACTIONS,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        di = np.asarray(getattr(stats, "exp_dInj", np.zeros((N_ACTIONS,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        exp_dE[slot, : min(N_ACTIONS, de.shape[0])] = de[:N_ACTIONS]
        exp_dInj[slot, : min(N_ACTIONS, di.shape[0])] = di[:N_ACTIONS]

        lineage_ids[slot] = str(getattr(stats, "lineage_id", "") or getattr(agent, "lineage_id", "") or "")
        founder_ids[slot] = _safe_int(getattr(stats, "founder_id", getattr(agent, "founder_id", -1)), -1)
        by_agent_id.append((aid, slot))

    by_agent_id.sort(key=lambda item: item[0])
    agent_ids = np.asarray([aid for aid, _slot in by_agent_id], dtype=np.int64)
    agent_id_to_slot_slots = np.asarray([slot for _aid, slot in by_agent_id], dtype=np.int32)
    hmem = _extract_hmem_by_slot(runner, cap)

    return ArrayBackendSnapshot(
        schema_version=SCHEMA_VERSION,
        generation=int(generation),
        phase=str(phase),
        slot_to_agent_id=slot_to_agent_id,
        slot_tokens=slot_tokens,
        alive_mask=alive_mask,
        agent_ids=agent_ids,
        agent_id_to_slot_agent_ids=agent_ids.copy(),
        agent_id_to_slot_slots=agent_id_to_slot_slots,
        pos_y=pos_y,
        pos_x=pos_x,
        energy=energy,
        health=health,
        injury=injury,
        age=age,
        hmem=hmem,
        drives=drives,
        exp_dE=exp_dE,
        exp_dInj=exp_dInj,
        action_buffer=action_buffer,
        observation_buffer=observation_buffer,
        lineage_ids_by_slot=tuple(lineage_ids),
        founder_ids_by_slot=tuple(founder_ids),
        churn_tape=tuple(),
    )


def validate_snapshot_against_runner(snapshot: ArrayBackendSnapshot, runner: Any) -> dict[str, Any]:
    env = getattr(runner, "env", None)
    env_alive = _env_array(env, "agent_alive")
    env_pos_y = _env_array(env, "agent_pos_y")
    env_pos_x = _env_array(env, "agent_pos_x")
    agents = list(getattr(getattr(runner, "pop", None), "agents", {}).values())
    mismatches: list[dict[str, Any]] = []
    seen_slots: set[int] = set()

    for agent in sorted(agents, key=lambda a: _safe_int(getattr(a, "agent_id", -1), -1)):
        aid = _safe_int(getattr(agent, "agent_id", -1), -1)
        slot = _safe_int(getattr(agent, "slot", -1), -1)
        alive = bool(getattr(agent, "alive", False))
        if slot >= 0 and slot in seen_slots and alive:
            mismatches.append({"kind": "duplicate_slot", "agent_id": aid, "slot": slot})
        if slot >= 0 and alive:
            seen_slots.add(slot)
        if not alive:
            continue
        if slot < 0 or slot >= int(snapshot.slot_to_agent_id.shape[0]):
            mismatches.append({"kind": "invalid_alive_slot", "agent_id": aid, "slot": slot})
            continue
        if int(snapshot.slot_to_agent_id[slot]) != aid:
            mismatches.append(
                {
                    "kind": "slot_agent_id_mismatch",
                    "agent_id": aid,
                    "slot": slot,
                    "snapshot_agent_id": int(snapshot.slot_to_agent_id[slot]),
                }
            )
        if not bool(snapshot.alive_mask[slot]):
            mismatches.append({"kind": "alive_mask_false", "agent_id": aid, "slot": slot})
        if env_alive is not None and 0 <= aid < int(env_alive.shape[0]) and not bool(env_alive[aid]):
            mismatches.append({"kind": "env_alive_false", "agent_id": aid, "slot": slot})
        if env_pos_y is not None and 0 <= aid < int(env_pos_y.shape[0]) and int(snapshot.pos_y[slot]) != int(env_pos_y[aid]):
            mismatches.append({"kind": "pos_y_mismatch", "agent_id": aid, "slot": slot})
        if env_pos_x is not None and 0 <= aid < int(env_pos_x.shape[0]) and int(snapshot.pos_x[slot]) != int(env_pos_x[aid]):
            mismatches.append({"kind": "pos_x_mismatch", "agent_id": aid, "slot": slot})

    return {
        "schema_version": SCHEMA_VERSION,
        "ok": len(mismatches) == 0,
        "mismatch_count": int(len(mismatches)),
        "mismatches": mismatches[:64],
        "alive_slots": int(np.sum(snapshot.alive_mask)),
        "agent_count": int(snapshot.agent_ids.shape[0]),
        "slot_capacity": int(snapshot.slot_to_agent_id.shape[0]),
        "hmem_shape": [int(v) for v in snapshot.hmem.shape],
    }


def snapshot_digest(snapshot: ArrayBackendSnapshot) -> str:
    payload = {
        "schema_version": snapshot.schema_version,
        "generation": int(snapshot.generation),
        "phase": str(snapshot.phase),
        "slot_to_agent_id": _array_payload(snapshot.slot_to_agent_id),
        "slot_tokens": _array_payload(snapshot.slot_tokens),
        "alive_mask": _array_payload(snapshot.alive_mask.astype(np.uint8, copy=False)),
        "agent_ids": _array_payload(snapshot.agent_ids),
        "agent_id_to_slot_slots": _array_payload(snapshot.agent_id_to_slot_slots),
        "pos_y": _array_payload(snapshot.pos_y),
        "pos_x": _array_payload(snapshot.pos_x),
        "energy": _array_payload(snapshot.energy),
        "health": _array_payload(snapshot.health),
        "injury": _array_payload(snapshot.injury),
        "age": _array_payload(snapshot.age),
        "hmem": _array_payload(snapshot.hmem),
        "drives": _array_payload(snapshot.drives),
        "exp_dE": _array_payload(snapshot.exp_dE),
        "exp_dInj": _array_payload(snapshot.exp_dInj),
        "lineage_ids_by_slot": list(snapshot.lineage_ids_by_slot),
        "founder_ids_by_slot": list(snapshot.founder_ids_by_slot),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def record_generation_shadow(
    runner: Any,
    *,
    alive: Any,
    generation: int,
    phase: str = "start_generation",
    buckets: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    enabled = bool(getattr(cfg, "runner_array_backend_enable", False)) or bool(
        getattr(cfg, "cxxrewrite_array_backend_shadow_enable", False)
    )
    authoritative = bool(getattr(cfg, "cxxrewrite_array_backend_authoritative_enable", False))
    debug_sync = bool(getattr(cfg, "cxxrewrite_array_backend_debug_sync_enable", False))
    if authoritative:
        raise RuntimeError("CXXREWRITE0 authoritative array backend is not implemented; use shadow mode only")
    if not enabled and not debug_sync:
        return None

    t0 = time.perf_counter()
    snapshot = build_array_backend_snapshot(runner, alive, generation=int(generation), phase=str(phase))
    t1 = time.perf_counter()
    validation = validate_snapshot_against_runner(snapshot, runner)
    t2 = time.perf_counter()
    digest = snapshot_digest(snapshot)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generation": int(generation),
        "phase": str(phase),
        "digest": digest,
        "summary": snapshot_summary(snapshot),
        "validation": validation,
        "timing": {
            "build_s": float(t1 - t0),
            "validate_s": float(t2 - t1),
            "total_s": float(t2 - t0),
        },
        "decision_contract": "shadow_only_python_authoritative",
    }
    captures = getattr(runner, "_cxxrewrite0_array_backend_shadow_captures", None)
    if not isinstance(captures, list):
        captures = []
        setattr(runner, "_cxxrewrite0_array_backend_shadow_captures", captures)
    captures.append(payload)
    setattr(runner, "_cxxrewrite0_array_backend_last_digest", digest)
    if isinstance(buckets, dict):
        buckets["cxxrewrite0_array_backend_build"] = float(
            buckets.get("cxxrewrite0_array_backend_build", 0.0) + float(t1 - t0)
        )
        buckets["cxxrewrite0_array_backend_validate"] = float(
            buckets.get("cxxrewrite0_array_backend_validate", 0.0) + float(t2 - t1)
        )
    if debug_sync:
        _write_shadow_payload(runner, payload)
    if not bool(validation.get("ok", False)) and bool(getattr(cfg, "cxxrewrite_array_backend_fail_fast", False)):
        raise RuntimeError("CXXREWRITE0 array backend shadow validation failed")
    return payload


def snapshot_summary(snapshot: ArrayBackendSnapshot) -> dict[str, Any]:
    live = np.asarray(snapshot.alive_mask, dtype=bool)
    live_energy = snapshot.energy[live] if np.any(live) else np.asarray([], dtype=np.float32)
    return {
        "slot_capacity": int(snapshot.slot_to_agent_id.shape[0]),
        "alive_slots": int(np.sum(live)),
        "agent_count": int(snapshot.agent_ids.shape[0]),
        "hmem_shape": [int(v) for v in snapshot.hmem.shape],
        "lineage_slots": int(sum(1 for lid in snapshot.lineage_ids_by_slot if str(lid))),
        "energy_sum": float(np.sum(live_energy, dtype=np.float64)) if live_energy.size else 0.0,
        "health_sum": float(np.sum(snapshot.health[live], dtype=np.float64)) if np.any(live) else 0.0,
        "position_checksum": int(
            np.sum(snapshot.pos_y[live].astype(np.int64) * 1000003 + snapshot.pos_x[live].astype(np.int64), dtype=np.int64)
        )
        if np.any(live)
        else 0,
        "churn_tape_events": int(len(snapshot.churn_tape)),
    }


def ownership_map_payload() -> dict[str, Any]:
    return {
        "schema_version": OWNERSHIP_SCHEMA_VERSION,
        "selected_decision": "CXXREWRITE0-A-ARRAYBACKEND-FIRST-GO",
        "state_items": [
            {"item": "agent identity", "current_owner": "Population.agents and Agent.agent_id", "array_backend_field": "slot_to_agent_id, agent_id_to_slot"},
            {"item": "alive/dead flag", "current_owner": "Agent.alive plus env.agent_alive", "array_backend_field": "alive_mask"},
            {"item": "agent position", "current_owner": "GridWorld agent_pos_y/x and agent_grid", "array_backend_field": "pos_y, pos_x"},
            {"item": "energy/health/injury", "current_owner": "Python Agent objects", "array_backend_field": "energy, health, injury"},
            {"item": "age/lifecycle fields", "current_owner": "Python Agent objects and AgentStats", "array_backend_field": "age plus future lifecycle arrays"},
            {"item": "genome/policy bank references", "current_owner": "Agent.genome and runner policy banks", "array_backend_field": "slot mapping only in CXXREWRITE0"},
            {"item": "Hmem/GRU state", "current_owner": "runner._Hmem torch bank by slot", "array_backend_field": "hmem"},
            {"item": "prediction state", "current_owner": "AgentStats exp_dE/exp_dInj", "array_backend_field": "exp_dE, exp_dInj"},
            {"item": "stride/policy cache", "current_owner": "runner transient policy caches", "array_backend_field": "future resident cache, not authoritative in CXXREWRITE0"},
            {"item": "observations/actions", "current_owner": "runner policy path", "array_backend_field": "observation_buffer, action_buffer"},
            {"item": "patch/env arrays", "current_owner": "GridWorld arrays", "array_backend_field": "future env-array view, not copied by CXXREWRITE0"},
            {"item": "births/deaths/refill", "current_owner": "Population, runner tick births, env.remove_agent/add_agent", "array_backend_field": "future churn_tape"},
            {"item": "lineage IDs", "current_owner": "AgentStats", "array_backend_field": "lineage_ids_by_slot, founder_ids_by_slot"},
            {"item": "metrics/checkpoint", "current_owner": "metrics/checkpoint Python writers", "array_backend_field": "boundary sync only"},
        ],
        "authoritative_mode_status": "blocked_until_churn_tape_and_sync_writeback_exist",
    }


def churn_tape_contract_payload() -> dict[str, Any]:
    return {
        "schema_version": "cxxrewrite0_churn_tape_contract_v1",
        "events": [
            {"event": "death", "required_fields": ["tick", "agent_id", "slot", "reason", "energy", "health"]},
            {"event": "birth", "required_fields": ["tick", "agent_id", "slot", "parent_ids", "lineage_id", "founder_id"]},
            {"event": "refill", "required_fields": ["tick", "agent_id", "slot", "source", "lineage_id"]},
            {"event": "slot_free", "required_fields": ["tick", "agent_id", "slot", "slot_token"]},
            {"event": "slot_alloc", "required_fields": ["tick", "agent_id", "slot", "slot_token"]},
            {"event": "hmem_alloc", "required_fields": ["tick", "agent_id", "slot", "hidden_dim"]},
            {"event": "policy_bank_alloc", "required_fields": ["tick", "agent_id", "slot", "policy_kind"]},
        ],
        "ordering_rule": "events sorted by tick, then deterministic event class, then agent_id",
        "sync_rule": "Python Agent objects are synchronized from array state at generation/checkpoint/debug boundaries only",
    }


def _alive_agents(runner: Any, alive: Any) -> list[Any]:
    if alive is None:
        alive = getattr(getattr(runner, "pop", None), "alive_agents", lambda: [])()
    return [a for a in list(alive or []) if bool(getattr(a, "alive", False))]


def _extract_hmem_by_slot(runner: Any, cap: int) -> np.ndarray:
    hmem_obj = getattr(runner, "_Hmem", None)
    if hmem_obj is None:
        hmem_obj = getattr(runner, "gru_hmem", None)
    arr: np.ndarray
    try:
        if hasattr(hmem_obj, "detach"):
            arr = hmem_obj.detach().to("cpu").numpy()
        else:
            arr = np.asarray(hmem_obj)
    except Exception:
        arr = np.empty((0, 0), dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros((int(cap), 0), dtype=np.float32)
    out = np.zeros((int(cap), int(arr.shape[1])), dtype=np.float32)
    rows = min(int(cap), int(arr.shape[0]))
    out[:rows, :] = np.asarray(arr[:rows, :], dtype=np.float32)
    return np.ascontiguousarray(out)


def _write_shadow_payload(runner: Any, payload: dict[str, Any]) -> None:
    raw_run_dir = str(getattr(runner, "run_dir", "") or getattr(runner, "run_dir_path", "") or "")
    if not raw_run_dir:
        return
    run_dir = Path(raw_run_dir)
    out_dir = run_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cxxrewrite0_array_backend_shadow.jsonl"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
        f.write("\n")


def _array_payload(arr: np.ndarray) -> dict[str, Any]:
    a = np.ascontiguousarray(arr)
    return {
        "shape": [int(v) for v in a.shape],
        "dtype": str(a.dtype),
        "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
    }


def _env_array(env: Any, name: str) -> np.ndarray | None:
    try:
        value = getattr(env, name)
    except Exception:
        return None
    if value is None:
        return None
    try:
        return np.asarray(value)
    except Exception:
        return None


def _position_value(arr: np.ndarray | None, aid: int, fallback: Any) -> int:
    if arr is not None and 0 <= int(aid) < int(arr.shape[0]):
        return _safe_int(arr[int(aid)], -1)
    return _safe_int(fallback, -1)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)
