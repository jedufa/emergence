from __future__ import annotations

"""CXXREWRITE-1 array lifecycle ownership scaffold.

Default-off engineering scaffold. CXXREWRITE-1 owns a compact generation
lifecycle boundary around the existing Python tick outputs: it records the
generation-start slot state, replays start-to-end churn into mutable arrays,
validates the replay against the live runner, and can sync the resulting array
state back into Python/env objects at the generation boundary.

This is not a native compute kernel and it does not change default behavior.
"""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np


SCHEMA_VERSION = "cxxrewrite1_array_lifecycle_v1"
CHURN_TAPE_SCHEMA_VERSION = "cxxrewrite1_churn_tape_v1"
OWNERSHIP_SCHEMA_VERSION = "cxxrewrite1_ownership_map_v1"
N_ACTIONS = 7
N_DRIVES = 4


@dataclass
class ArrayLifecycleState:
    schema_version: str
    generation: int
    phase: str
    slot_to_agent_id: np.ndarray
    slot_tokens: np.ndarray
    alive_mask: np.ndarray
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
    lineage_ids_by_slot: list[str]
    founder_ids_by_slot: list[int]
    policy_kind_by_slot: list[str]
    free_slots: list[int]
    next_agent_id: int
    churn_events: list[dict[str, Any]] = field(default_factory=list)
    next_sequence: int = 0


def lifecycle_requested(cfg: Any) -> bool:
    return bool(getattr(cfg, "cxxrewrite_array_backend_enable", False)) or bool(
        getattr(cfg, "cxxrewrite_lifecycle_shadow_enable", False)
    ) or bool(getattr(cfg, "cxxrewrite_lifecycle_authoritative_enable", False)) or bool(
        getattr(cfg, "cxxrewrite_lifecycle_record_churn_tape_enable", False)
    ) or bool(getattr(cfg, "cxxrewrite_lifecycle_validate_shadow_enable", False))


def begin_generation_lifecycle(
    runner: Any,
    *,
    alive: Any,
    generation: int,
    buckets: dict[str, float] | None = None,
) -> ArrayLifecycleState | None:
    cfg = getattr(runner, "sim_cfg", None)
    if not lifecycle_requested(cfg):
        return None
    _assert_authoritative_components_ready(cfg)

    t0 = time.perf_counter()
    state = build_lifecycle_state_from_runner(runner, alive=alive, generation=generation, phase="generation_start")
    state.record_event(
        "generation_start_snapshot",
        agent_id=-1,
        reason="array_lifecycle_generation_start",
        payload=summarize_lifecycle_state(state),
    )
    if isinstance(buckets, dict):
        buckets["cxxrewrite1_lifecycle_begin"] = float(
            buckets.get("cxxrewrite1_lifecycle_begin", 0.0) + (time.perf_counter() - t0)
        )
    setattr(runner, "_cxxrewrite1_lifecycle_start_state", state)
    return state


def finish_generation_lifecycle(
    runner: Any,
    *,
    start_state: ArrayLifecycleState | None,
    generation: int,
    phase: str = "post_envsync",
    buckets: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    if not lifecycle_requested(cfg):
        return None
    _assert_authoritative_components_ready(cfg)
    if start_state is None:
        if _fail_closed(cfg):
            raise RuntimeError("CXXREWRITE1 lifecycle finish requested without a generation-start state")
        return None

    t0 = time.perf_counter()
    result = replay_generation_lifecycle(runner, start_state=start_state, generation=generation, phase=phase)
    replay_state = result["state"]
    t1 = time.perf_counter()
    sync_summary: dict[str, Any] = {"enabled": False}
    if bool(getattr(cfg, "cxxrewrite_lifecycle_authoritative_enable", False)):
        sync_summary = sync_python_objects_from_lifecycle(runner, replay_state)
    t2 = time.perf_counter()
    validation = validate_lifecycle_against_runner(replay_state, runner)
    t3 = time.perf_counter()

    if not bool(validation.get("ok", False)) and (
        bool(getattr(cfg, "cxxrewrite_lifecycle_authoritative_enable", False)) or _fail_closed(cfg)
    ):
        raise RuntimeError("CXXREWRITE1 lifecycle validation failed")

    digest = lifecycle_digest(replay_state)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generation": int(generation),
        "phase": str(phase),
        "mode": "authoritative_boundary_sync"
        if bool(getattr(cfg, "cxxrewrite_lifecycle_authoritative_enable", False))
        else "shadow_boundary_replay",
        "digest": digest,
        "summary": summarize_lifecycle_state(replay_state),
        "churn_event_count": int(len(replay_state.churn_events)),
        "churn_tape_digest": churn_tape_digest(replay_state.churn_events),
        "validation": validation,
        "sync_summary": sync_summary,
        "first_divergence": first_divergence(validation),
        "timing": {
            "replay_s": float(t1 - t0),
            "sync_s": float(t2 - t1),
            "validate_s": float(t3 - t2),
            "total_s": float(t3 - t0),
        },
        "churn_events_excerpt": replay_state.churn_events[:64],
    }
    captures = getattr(runner, "_cxxrewrite1_lifecycle_captures", None)
    if not isinstance(captures, list):
        captures = []
        setattr(runner, "_cxxrewrite1_lifecycle_captures", captures)
    captures.append(payload)
    setattr(runner, "_cxxrewrite1_lifecycle_last_digest", digest)
    setattr(runner, "_cxxrewrite1_lifecycle_last_state", replay_state)
    if bool(getattr(cfg, "cxxrewrite_lifecycle_record_churn_tape_enable", False)):
        _write_lifecycle_payload(runner, payload, replay_state.churn_events)
    if isinstance(buckets, dict):
        buckets["cxxrewrite1_lifecycle_replay"] = float(
            buckets.get("cxxrewrite1_lifecycle_replay", 0.0) + float(t1 - t0)
        )
        buckets["cxxrewrite1_lifecycle_sync"] = float(
            buckets.get("cxxrewrite1_lifecycle_sync", 0.0) + float(t2 - t1)
        )
        buckets["cxxrewrite1_lifecycle_validate"] = float(
            buckets.get("cxxrewrite1_lifecycle_validate", 0.0) + float(t3 - t2)
        )
    return payload


def build_lifecycle_state_from_runner(
    runner: Any,
    alive: Any = None,
    *,
    generation: int = -1,
    phase: str = "snapshot",
) -> ArrayLifecycleState:
    agents = _alive_agents(runner, alive)
    hmem = _extract_hmem_by_slot(runner, 0)
    max_slot = max([_safe_int(getattr(a, "slot", -1), -1) for a in agents] + [-1])
    pop = getattr(runner, "pop", None)
    cap = max(
        1,
        _safe_int(getattr(pop, "max_slots", 0), 0),
        int(max_slot) + 1,
        int(hmem.shape[0]) if hmem.ndim == 2 else 0,
    )
    if hmem.shape[0] != cap:
        hmem = _extract_hmem_by_slot(runner, cap)

    slot_to_agent_id = np.full((cap,), -1, dtype=np.int64)
    slot_tokens = np.full((cap,), -1, dtype=np.int64)
    alive_mask = np.zeros((cap,), dtype=np.bool_)
    pos_y = np.full((cap,), -1, dtype=np.int32)
    pos_x = np.full((cap,), -1, dtype=np.int32)
    energy = np.zeros((cap,), dtype=np.float64)
    health = np.zeros((cap,), dtype=np.float64)
    injury = np.zeros((cap,), dtype=np.float64)
    age = np.zeros((cap,), dtype=np.int32)
    drives = np.zeros((cap, N_DRIVES), dtype=np.float32)
    exp_dE = np.zeros((cap, N_ACTIONS), dtype=np.float32)
    exp_dInj = np.zeros((cap, N_ACTIONS), dtype=np.float32)
    action_buffer = np.full((cap,), -1, dtype=np.int32)
    observation_buffer = np.empty((cap, 0), dtype=np.float32)
    lineage_ids = [""] * cap
    founder_ids = [-1] * cap
    policy_kind = [""] * cap

    env = getattr(runner, "env", None)
    env_pos_y = _env_array(env, "agent_pos_y")
    env_pos_x = _env_array(env, "agent_pos_x")
    pairs: list[tuple[int, int]] = []
    for agent in sorted(agents, key=lambda a: (_safe_int(getattr(a, "slot", -1), -1), _safe_int(getattr(a, "agent_id", -1), -1))):
        aid = _safe_int(getattr(agent, "agent_id", -1), -1)
        slot = _safe_int(getattr(agent, "slot", -1), -1)
        if aid < 0 or slot < 0 or slot >= cap:
            continue
        stats = getattr(agent, "stats", None)
        slot_to_agent_id[slot] = np.int64(aid)
        slot_tokens[slot] = np.int64(_safe_int(getattr(agent, "slot_token", 0), 0))
        alive_mask[slot] = True
        pos_y[slot] = np.int32(_position_value(env_pos_y, aid, getattr(agent, "y", -1)))
        pos_x[slot] = np.int32(_position_value(env_pos_x, aid, getattr(agent, "x", -1)))
        energy[slot] = np.float64(_safe_float(getattr(agent, "energy", 0.0), 0.0))
        health[slot] = np.float64(_safe_float(getattr(agent, "health", 0.0), 0.0))
        injury[slot] = np.float64(_safe_float(getattr(agent, "injury", 0.0), 0.0))
        age[slot] = np.int32(_safe_int(getattr(stats, "survival_steps", getattr(agent, "age", 0)), 0))
        d = np.asarray(getattr(agent, "drives", np.zeros((N_DRIVES,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        drives[slot, : min(N_DRIVES, int(d.shape[0]))] = d[:N_DRIVES]
        de = np.asarray(getattr(stats, "exp_dE", np.zeros((N_ACTIONS,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        di = np.asarray(getattr(stats, "exp_dInj", np.zeros((N_ACTIONS,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        exp_dE[slot, : min(N_ACTIONS, int(de.shape[0]))] = de[:N_ACTIONS]
        exp_dInj[slot, : min(N_ACTIONS, int(di.shape[0]))] = di[:N_ACTIONS]
        lineage_ids[slot] = str(getattr(stats, "lineage_id", "") or getattr(agent, "lineage_id", "") or "")
        founder_ids[slot] = _safe_int(getattr(stats, "founder_id", getattr(agent, "founder_id", -1)), -1)
        policy_kind[slot] = type(getattr(agent, "policy", None)).__name__ if getattr(agent, "policy", None) is not None else ""
        pairs.append((aid, slot))

    pairs.sort(key=lambda item: item[0])
    free_slots = [idx for idx in range(cap) if not bool(alive_mask[idx])]
    return ArrayLifecycleState(
        schema_version=SCHEMA_VERSION,
        generation=int(generation),
        phase=str(phase),
        slot_to_agent_id=slot_to_agent_id,
        slot_tokens=slot_tokens,
        alive_mask=alive_mask,
        agent_id_to_slot_agent_ids=np.asarray([aid for aid, _slot in pairs], dtype=np.int64),
        agent_id_to_slot_slots=np.asarray([slot for _aid, slot in pairs], dtype=np.int32),
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
        lineage_ids_by_slot=lineage_ids,
        founder_ids_by_slot=founder_ids,
        policy_kind_by_slot=policy_kind,
        free_slots=free_slots,
        next_agent_id=_safe_int(getattr(pop, "next_id", 0), 0),
    )


def replay_generation_lifecycle(
    runner: Any,
    *,
    start_state: ArrayLifecycleState,
    generation: int,
    phase: str = "post_envsync",
) -> dict[str, Any]:
    end_state = build_lifecycle_state_from_runner(
        runner,
        alive=getattr(getattr(runner, "pop", None), "alive_agents", lambda: [])(),
        generation=generation,
        phase=phase,
    )
    state = _replay_base_from_start(start_state, max_capacity=int(end_state.slot_to_agent_id.shape[0]), hmem_cols=int(end_state.hmem.shape[1]) if end_state.hmem.ndim == 2 else 0)
    start_live = _live_agent_to_slot(start_state)
    end_live = _live_agent_to_slot(end_state)

    for aid in sorted(set(start_live) - set(end_live)):
        old_slot = int(start_live[aid])
        _record_death_and_slot_free(state, aid=aid, old_slot=old_slot, generation=generation, reason="absent_after_generation")

    for aid in sorted(set(start_live) & set(end_live)):
        old_slot = int(start_live[aid])
        new_slot = int(end_live[aid])
        if old_slot != new_slot:
            state.record_event("slot_freed", agent_id=aid, old_slot=old_slot, reason="survivor_slot_reassigned")
            state.record_event("slot_allocated", agent_id=aid, new_slot=new_slot, reason="survivor_slot_reassigned")
        _copy_slot_from_end(state, end_state, new_slot)

    for aid in sorted(set(end_live) - set(start_live)):
        new_slot = int(end_live[aid])
        event_kind = _new_agent_event_kind(runner, end_state, new_slot)
        payload = _new_agent_payload(runner, end_state, new_slot)
        state.record_event(event_kind, agent_id=aid, new_slot=new_slot, reason="new_alive_after_generation", payload=payload)
        state.record_event("slot_allocated", agent_id=aid, new_slot=new_slot, reason=event_kind, payload={"slot_token": int(end_state.slot_tokens[new_slot])})
        state.record_event("hmem_row_allocated", agent_id=aid, new_slot=new_slot, reason=event_kind, payload={"hmem_shape": _row_shape(end_state.hmem, new_slot)})
        state.record_event("policy_row_attached", agent_id=aid, new_slot=new_slot, reason=event_kind, payload={"policy_kind": str(end_state.policy_kind_by_slot[new_slot])})
        _copy_slot_from_end(state, end_state, new_slot)

    _record_birth_failures(state, runner, generation)
    state.free_slots = [idx for idx in range(int(state.alive_mask.shape[0])) if not bool(state.alive_mask[idx])]
    _refresh_agent_id_to_slot(state)
    state.phase = str(phase)
    state.record_event(
        "python_sync_boundary",
        agent_id=-1,
        reason="generation_boundary_sync_ready",
        payload=summarize_lifecycle_state(state),
    )
    return {"state": state, "end_state": end_state}


def sync_python_objects_from_lifecycle(runner: Any, state: ArrayLifecycleState) -> dict[str, Any]:
    pop = getattr(runner, "pop", None)
    env = getattr(runner, "env", None)
    agents = getattr(pop, "agents", {}) if pop is not None else {}
    synced = 0
    missing_agents = 0
    for slot in np.nonzero(state.alive_mask)[0].astype(np.int64).tolist():
        aid = int(state.slot_to_agent_id[int(slot)])
        agent = agents.get(aid) if isinstance(agents, dict) else None
        if agent is None:
            missing_agents += 1
            continue
        setattr(agent, "alive", True)
        setattr(agent, "energy", float(state.energy[int(slot)]))
        setattr(agent, "health", float(state.health[int(slot)]))
        setattr(agent, "injury", float(state.injury[int(slot)]))
        setattr(agent, "slot", int(slot))
        setattr(agent, "slot_token", int(state.slot_tokens[int(slot)]))
        stats = getattr(agent, "stats", None)
        if stats is not None:
            try:
                setattr(stats, "survival_steps", int(state.age[int(slot)]))
            except Exception:
                pass
            try:
                setattr(stats, "exp_dE", np.asarray(state.exp_dE[int(slot)], dtype=np.float32).copy())
                setattr(stats, "exp_dInj", np.asarray(state.exp_dInj[int(slot)], dtype=np.float32).copy())
            except Exception:
                pass
        synced += 1
        _sync_env_slot(env, aid=aid, y=int(state.pos_y[int(slot)]), x=int(state.pos_x[int(slot)]), alive=True)

    alive_ids = {int(state.slot_to_agent_id[int(slot)]) for slot in np.nonzero(state.alive_mask)[0].tolist()}
    stale = 0
    if isinstance(agents, dict):
        for aid, agent in list(agents.items()):
            if int(aid) not in alive_ids and bool(getattr(agent, "alive", False)):
                setattr(agent, "alive", False)
                _sync_env_slot(env, aid=int(aid), y=-1, x=-1, alive=False)
                stale += 1
    return {
        "enabled": True,
        "synced_alive_agents": int(synced),
        "missing_agents": int(missing_agents),
        "stale_agents_marked_dead": int(stale),
    }


def validate_lifecycle_against_runner(state: ArrayLifecycleState, runner: Any) -> dict[str, Any]:
    env = getattr(runner, "env", None)
    env_alive = _env_array(env, "agent_alive")
    env_pos_y = _env_array(env, "agent_pos_y")
    env_pos_x = _env_array(env, "agent_pos_x")
    pop = getattr(runner, "pop", None)
    agents = list(getattr(pop, "agents", {}).values()) if pop is not None else []
    mismatches: list[dict[str, Any]] = []
    live_slots = set(int(v) for v in np.nonzero(state.alive_mask)[0].tolist())
    seen_slots: set[int] = set()
    alive_agents = [a for a in agents if bool(getattr(a, "alive", False))]

    for agent in sorted(alive_agents, key=lambda a: _safe_int(getattr(a, "agent_id", -1), -1)):
        aid = _safe_int(getattr(agent, "agent_id", -1), -1)
        slot = _safe_int(getattr(agent, "slot", -1), -1)
        if slot in seen_slots:
            mismatches.append({"kind": "duplicate_alive_slot", "agent_id": aid, "slot": slot})
        seen_slots.add(slot)
        if slot < 0 or slot >= int(state.alive_mask.shape[0]):
            mismatches.append({"kind": "invalid_slot", "agent_id": aid, "slot": slot})
            continue
        if slot not in live_slots:
            mismatches.append({"kind": "slot_not_alive_in_array", "agent_id": aid, "slot": slot})
        if int(state.slot_to_agent_id[slot]) != aid:
            mismatches.append({"kind": "slot_agent_mismatch", "agent_id": aid, "slot": slot, "array_agent_id": int(state.slot_to_agent_id[slot])})
        if abs(float(state.energy[slot]) - float(getattr(agent, "energy", 0.0))) > 1e-4:
            mismatches.append({"kind": "energy_mismatch", "agent_id": aid, "slot": slot})
        if abs(float(state.health[slot]) - float(getattr(agent, "health", 0.0))) > 1e-4:
            mismatches.append({"kind": "health_mismatch", "agent_id": aid, "slot": slot})
        if abs(float(state.injury[slot]) - float(getattr(agent, "injury", 0.0))) > 1e-4:
            mismatches.append({"kind": "injury_mismatch", "agent_id": aid, "slot": slot})
        if env_alive is not None and 0 <= aid < int(env_alive.shape[0]) and bool(env_alive[aid]) is not True:
            mismatches.append({"kind": "env_alive_false", "agent_id": aid, "slot": slot})
        if env_pos_y is not None and 0 <= aid < int(env_pos_y.shape[0]) and int(state.pos_y[slot]) != int(env_pos_y[aid]):
            mismatches.append({"kind": "pos_y_mismatch", "agent_id": aid, "slot": slot})
        if env_pos_x is not None and 0 <= aid < int(env_pos_x.shape[0]) and int(state.pos_x[slot]) != int(env_pos_x[aid]):
            mismatches.append({"kind": "pos_x_mismatch", "agent_id": aid, "slot": slot})

    array_alive_ids = {int(state.slot_to_agent_id[slot]) for slot in live_slots if int(state.slot_to_agent_id[slot]) >= 0}
    runner_alive_ids = {_safe_int(getattr(a, "agent_id", -1), -1) for a in alive_agents}
    for aid in sorted(array_alive_ids - runner_alive_ids):
        mismatches.append({"kind": "array_alive_missing_in_runner", "agent_id": aid})
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": len(mismatches) == 0,
        "mismatch_count": int(len(mismatches)),
        "mismatches": mismatches[:64],
        "alive_slots": int(np.sum(state.alive_mask)),
        "runner_alive_agents": int(len(alive_agents)),
        "slot_capacity": int(state.alive_mask.shape[0]),
        "churn_event_count": int(len(state.churn_events)),
    }


def summarize_lifecycle_state(state: ArrayLifecycleState) -> dict[str, Any]:
    live = np.asarray(state.alive_mask, dtype=bool)
    return {
        "schema_version": state.schema_version,
        "generation": int(state.generation),
        "phase": str(state.phase),
        "slot_capacity": int(state.alive_mask.shape[0]),
        "alive_slots": int(np.sum(live)),
        "free_slots": int(len(state.free_slots)),
        "agent_count": int(state.agent_id_to_slot_agent_ids.shape[0]),
        "hmem_shape": [int(v) for v in state.hmem.shape],
        "energy_sum": float(np.sum(state.energy[live], dtype=np.float64)) if np.any(live) else 0.0,
        "health_sum": float(np.sum(state.health[live], dtype=np.float64)) if np.any(live) else 0.0,
        "position_checksum": int(
            np.sum(state.pos_y[live].astype(np.int64) * 1000003 + state.pos_x[live].astype(np.int64), dtype=np.int64)
        )
        if np.any(live)
        else 0,
        "churn_events": int(len(state.churn_events)),
    }


def lifecycle_digest(state: ArrayLifecycleState) -> str:
    payload = {
        "schema_version": state.schema_version,
        "generation": int(state.generation),
        "phase": str(state.phase),
        "slot_to_agent_id": _array_payload(state.slot_to_agent_id),
        "slot_tokens": _array_payload(state.slot_tokens),
        "alive_mask": _array_payload(state.alive_mask.astype(np.uint8, copy=False)),
        "agent_id_to_slot_agent_ids": _array_payload(state.agent_id_to_slot_agent_ids),
        "agent_id_to_slot_slots": _array_payload(state.agent_id_to_slot_slots),
        "pos_y": _array_payload(state.pos_y),
        "pos_x": _array_payload(state.pos_x),
        "energy": _array_payload(state.energy),
        "health": _array_payload(state.health),
        "injury": _array_payload(state.injury),
        "age": _array_payload(state.age),
        "hmem": _array_payload(state.hmem),
        "drives": _array_payload(state.drives),
        "exp_dE": _array_payload(state.exp_dE),
        "exp_dInj": _array_payload(state.exp_dInj),
        "lineage_ids_by_slot": list(state.lineage_ids_by_slot),
        "founder_ids_by_slot": list(state.founder_ids_by_slot),
        "policy_kind_by_slot": list(state.policy_kind_by_slot),
        "free_slots": list(state.free_slots),
        "churn_tape_digest": churn_tape_digest(state.churn_events),
    }
    return _sha256_json(payload)


def churn_tape_digest(events: list[dict[str, Any]]) -> str:
    return _sha256_json(events)


def churn_tape_schema_payload() -> dict[str, Any]:
    return {
        "schema_version": CHURN_TAPE_SCHEMA_VERSION,
        "events": [
            "generation_start_snapshot",
            "tick_start_snapshot",
            "death",
            "birth_success",
            "birth_failed",
            "refill_created",
            "slot_freed",
            "slot_allocated",
            "hmem_row_allocated",
            "policy_row_attached",
            "python_sync_boundary",
        ],
        "required_fields": [
            "schema_version",
            "generation",
            "event_sequence",
            "event",
            "agent_id",
            "old_slot",
            "new_slot",
            "reason",
            "payload",
            "checksum",
        ],
        "ordering_rule": "event_sequence is assigned by deterministic replay order: deaths by agent_id, survivors by agent_id, new agents by agent_id, aggregate failures, sync boundary",
    }


def ownership_map_payload() -> dict[str, Any]:
    return {
        "schema_version": OWNERSHIP_SCHEMA_VERSION,
        "selected_decision": "CXXREWRITE1-A-ARRAY-LIFECYCLE-AUTHORITATIVE-GO",
        "state_items": [
            {"item": "slot table", "current_owner": "Population SlotPool and Agent.slot", "array_owner": "ArrayLifecycleState slot_to_agent_id/agent_id_to_slot", "sync_direction": "Python outputs -> array replay -> Python boundary sync", "unresolved_boundary": "tick compute still Python"},
            {"item": "alive mask", "current_owner": "Agent.alive and GridWorld.agent_alive", "array_owner": "mutable alive_mask", "sync_direction": "array -> Agent.alive/env arrays at boundary", "unresolved_boundary": "death cause detail is compact"},
            {"item": "churn tape", "current_owner": "implicit runner/pop/env side effects", "array_owner": "CXXREWRITE1 churn_events", "sync_direction": "array event stream is emitted to metrics JSONL", "unresolved_boundary": "native consumers not implemented"},
            {"item": "Hmem row lifecycle", "current_owner": "runner._Hmem by slot", "array_owner": "hmem snapshot and hmem_row_allocated events", "sync_direction": "runner bank remains backing store", "unresolved_boundary": "native bank allocator not implemented"},
            {"item": "policy row lifecycle", "current_owner": "Agent.policy/genome and runner banks", "array_owner": "policy_row_attached events by slot", "sync_direction": "Python policy remains object backing", "unresolved_boundary": "native policy row storage not implemented"},
            {"item": "Python sync boundary", "current_owner": "normal runner objects", "array_owner": "sync_python_objects_from_lifecycle", "sync_direction": "array -> Agent/env fields", "unresolved_boundary": "metrics/checkpoints still Python readers"},
        ],
    }


def first_divergence(validation: dict[str, Any]) -> dict[str, Any] | None:
    mismatches = validation.get("mismatches", [])
    if not mismatches:
        return None
    first = dict(mismatches[0])
    first["schema_version"] = "cxxrewrite1_first_divergence_v1"
    return first


def _record_event(
    state: ArrayLifecycleState,
    event: str,
    *,
    agent_id: int,
    old_slot: int = -1,
    new_slot: int = -1,
    reason: str = "",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    compact_payload = _jsonable(payload or {})
    seq = int(state.next_sequence)
    state.next_sequence += 1
    row = {
        "schema_version": CHURN_TAPE_SCHEMA_VERSION,
        "generation": int(state.generation),
        "tick": None,
        "event_sequence": seq,
        "event": str(event),
        "agent_id": int(agent_id),
        "parent_ids": list(compact_payload.get("parent_ids", [])) if isinstance(compact_payload.get("parent_ids", []), list) else [],
        "old_slot": int(old_slot),
        "new_slot": int(new_slot),
        "reason": str(reason),
        "payload": compact_payload,
    }
    row["checksum"] = _sha256_json({k: row[k] for k in sorted(row) if k != "checksum"})
    state.churn_events.append(row)
    return row


ArrayLifecycleState.record_event = _record_event  # type: ignore[attr-defined]


def _assert_authoritative_components_ready(cfg: Any) -> None:
    if not bool(getattr(cfg, "cxxrewrite_lifecycle_authoritative_enable", False)):
        return
    missing = []
    if not bool(getattr(cfg, "cxxrewrite_lifecycle_sync_python_objects_enable", False)):
        missing.append("cxxrewrite_lifecycle_sync_python_objects_enable")
    if not bool(getattr(cfg, "cxxrewrite_lifecycle_record_churn_tape_enable", False)):
        missing.append("cxxrewrite_lifecycle_record_churn_tape_enable")
    if not bool(getattr(cfg, "cxxrewrite_lifecycle_validate_shadow_enable", False)):
        missing.append("cxxrewrite_lifecycle_validate_shadow_enable")
    if missing:
        raise RuntimeError("CXXREWRITE1 authoritative lifecycle requested without required components: " + ", ".join(missing))


def _fail_closed(cfg: Any) -> bool:
    return bool(getattr(cfg, "cxxrewrite_lifecycle_fail_closed", False)) or bool(
        getattr(cfg, "cxxrewrite_lifecycle_authoritative_enable", False)
    )


def _replay_base_from_start(start: ArrayLifecycleState, *, max_capacity: int, hmem_cols: int) -> ArrayLifecycleState:
    cap = max(int(max_capacity), int(start.alive_mask.shape[0]))
    hcols = max(int(hmem_cols), int(start.hmem.shape[1]) if start.hmem.ndim == 2 else 0)
    out = build_empty_lifecycle_state(capacity=cap, hmem_cols=hcols, generation=int(start.generation), phase=str(start.phase))
    rows = min(cap, int(start.alive_mask.shape[0]))
    out.slot_to_agent_id[:rows] = start.slot_to_agent_id[:rows]
    out.slot_tokens[:rows] = start.slot_tokens[:rows]
    out.alive_mask[:rows] = start.alive_mask[:rows]
    out.pos_y[:rows] = start.pos_y[:rows]
    out.pos_x[:rows] = start.pos_x[:rows]
    out.energy[:rows] = start.energy[:rows]
    out.health[:rows] = start.health[:rows]
    out.injury[:rows] = start.injury[:rows]
    out.age[:rows] = start.age[:rows]
    out.drives[:rows, :] = start.drives[:rows, :]
    out.exp_dE[:rows, :] = start.exp_dE[:rows, :]
    out.exp_dInj[:rows, :] = start.exp_dInj[:rows, :]
    hrows = min(cap, int(start.hmem.shape[0]))
    hcols2 = min(hcols, int(start.hmem.shape[1]) if start.hmem.ndim == 2 else 0)
    if hrows and hcols2:
        out.hmem[:hrows, :hcols2] = start.hmem[:hrows, :hcols2]
    out.lineage_ids_by_slot[:rows] = list(start.lineage_ids_by_slot[:rows])
    out.founder_ids_by_slot[:rows] = list(start.founder_ids_by_slot[:rows])
    out.policy_kind_by_slot[:rows] = list(start.policy_kind_by_slot[:rows])
    out.free_slots = [idx for idx in range(cap) if not bool(out.alive_mask[idx])]
    out.next_agent_id = int(start.next_agent_id)
    out.churn_events = [dict(event) for event in start.churn_events]
    out.next_sequence = int(start.next_sequence)
    _refresh_agent_id_to_slot(out)
    return out


def build_empty_lifecycle_state(*, capacity: int, hmem_cols: int, generation: int, phase: str) -> ArrayLifecycleState:
    cap = max(1, int(capacity))
    hcols = max(0, int(hmem_cols))
    return ArrayLifecycleState(
        schema_version=SCHEMA_VERSION,
        generation=int(generation),
        phase=str(phase),
        slot_to_agent_id=np.full((cap,), -1, dtype=np.int64),
        slot_tokens=np.full((cap,), -1, dtype=np.int64),
        alive_mask=np.zeros((cap,), dtype=np.bool_),
        agent_id_to_slot_agent_ids=np.empty((0,), dtype=np.int64),
        agent_id_to_slot_slots=np.empty((0,), dtype=np.int32),
        pos_y=np.full((cap,), -1, dtype=np.int32),
        pos_x=np.full((cap,), -1, dtype=np.int32),
        energy=np.zeros((cap,), dtype=np.float64),
        health=np.zeros((cap,), dtype=np.float64),
        injury=np.zeros((cap,), dtype=np.float64),
        age=np.zeros((cap,), dtype=np.int32),
        hmem=np.zeros((cap, hcols), dtype=np.float32),
        drives=np.zeros((cap, N_DRIVES), dtype=np.float32),
        exp_dE=np.zeros((cap, N_ACTIONS), dtype=np.float32),
        exp_dInj=np.zeros((cap, N_ACTIONS), dtype=np.float32),
        action_buffer=np.full((cap,), -1, dtype=np.int32),
        observation_buffer=np.empty((cap, 0), dtype=np.float32),
        lineage_ids_by_slot=[""] * cap,
        founder_ids_by_slot=[-1] * cap,
        policy_kind_by_slot=[""] * cap,
        free_slots=list(range(cap)),
        next_agent_id=0,
    )


def _copy_slot_from_end(state: ArrayLifecycleState, end_state: ArrayLifecycleState, slot: int) -> None:
    s = int(slot)
    state.slot_to_agent_id[s] = end_state.slot_to_agent_id[s]
    state.slot_tokens[s] = end_state.slot_tokens[s]
    state.alive_mask[s] = bool(end_state.alive_mask[s])
    state.pos_y[s] = end_state.pos_y[s]
    state.pos_x[s] = end_state.pos_x[s]
    state.energy[s] = end_state.energy[s]
    state.health[s] = end_state.health[s]
    state.injury[s] = end_state.injury[s]
    state.age[s] = end_state.age[s]
    state.drives[s, :] = end_state.drives[s, :]
    state.exp_dE[s, :] = end_state.exp_dE[s, :]
    state.exp_dInj[s, :] = end_state.exp_dInj[s, :]
    if state.hmem.shape[1] and end_state.hmem.shape[1]:
        cols = min(int(state.hmem.shape[1]), int(end_state.hmem.shape[1]))
        state.hmem[s, :cols] = end_state.hmem[s, :cols]
    state.lineage_ids_by_slot[s] = str(end_state.lineage_ids_by_slot[s])
    state.founder_ids_by_slot[s] = int(end_state.founder_ids_by_slot[s])
    state.policy_kind_by_slot[s] = str(end_state.policy_kind_by_slot[s])


def _record_death_and_slot_free(state: ArrayLifecycleState, *, aid: int, old_slot: int, generation: int, reason: str) -> None:
    _ = generation
    s = int(old_slot)
    payload = {
        "energy": float(state.energy[s]) if 0 <= s < int(state.energy.shape[0]) else 0.0,
        "health": float(state.health[s]) if 0 <= s < int(state.health.shape[0]) else 0.0,
        "injury": float(state.injury[s]) if 0 <= s < int(state.injury.shape[0]) else 0.0,
        "slot_token": int(state.slot_tokens[s]) if 0 <= s < int(state.slot_tokens.shape[0]) else -1,
    }
    state.record_event("death", agent_id=int(aid), old_slot=s, reason=reason, payload=payload)
    state.record_event("slot_freed", agent_id=int(aid), old_slot=s, reason=reason, payload={"slot_token": payload["slot_token"]})
    if 0 <= s < int(state.alive_mask.shape[0]):
        state.alive_mask[s] = False
        state.slot_to_agent_id[s] = -1
        state.pos_y[s] = -1
        state.pos_x[s] = -1


def _record_birth_failures(state: ArrayLifecycleState, runner: Any, generation: int) -> None:
    _ = generation
    fields = [
        "birth_fail_no_space",
        "birth_fail_no_mate",
        "birth_fail_buffer",
        "birth_fail_injury_gate",
        "birth_fail_scar_gate",
        "birth_fail_conception",
        "birth_fail_gestation",
        "birth_make_child_fail",
    ]
    for name in fields:
        value = _safe_int(getattr(runner, f"_{name}_gen", 0), 0)
        if value > 0:
            state.record_event("birth_failed", agent_id=-1, reason=name, payload={"count": int(value)})


def _new_agent_event_kind(runner: Any, end_state: ArrayLifecycleState, slot: int) -> str:
    aid = int(end_state.slot_to_agent_id[int(slot)])
    agent = getattr(getattr(runner, "pop", None), "agents", {}).get(aid)
    stats = getattr(agent, "stats", None)
    parents = getattr(stats, "parent_ids", ()) if stats is not None else ()
    return "birth_success" if parents else "refill_created"


def _new_agent_payload(runner: Any, end_state: ArrayLifecycleState, slot: int) -> dict[str, Any]:
    aid = int(end_state.slot_to_agent_id[int(slot)])
    agent = getattr(getattr(runner, "pop", None), "agents", {}).get(aid)
    stats = getattr(agent, "stats", None)
    parent_ids = list(getattr(stats, "parent_ids", ()) or []) if stats is not None else []
    return {
        "parent_ids": [int(v) for v in parent_ids],
        "lineage_id": str(end_state.lineage_ids_by_slot[int(slot)]),
        "founder_id": int(end_state.founder_ids_by_slot[int(slot)]),
        "slot_token": int(end_state.slot_tokens[int(slot)]),
        "policy_kind": str(end_state.policy_kind_by_slot[int(slot)]),
    }


def _live_agent_to_slot(state: ArrayLifecycleState) -> dict[int, int]:
    out: dict[int, int] = {}
    for slot in np.nonzero(state.alive_mask)[0].astype(np.int64).tolist():
        aid = int(state.slot_to_agent_id[int(slot)])
        if aid >= 0:
            out[aid] = int(slot)
    return out


def _refresh_agent_id_to_slot(state: ArrayLifecycleState) -> None:
    pairs = sorted(_live_agent_to_slot(state).items(), key=lambda item: item[0])
    state.agent_id_to_slot_agent_ids = np.asarray([aid for aid, _slot in pairs], dtype=np.int64)
    state.agent_id_to_slot_slots = np.asarray([slot for _aid, slot in pairs], dtype=np.int32)


def _row_shape(arr: np.ndarray, slot: int) -> list[int]:
    if arr.ndim != 2 or slot < 0 or slot >= int(arr.shape[0]):
        return [0]
    return [int(arr.shape[1])]


def _sync_env_slot(env: Any, *, aid: int, y: int, x: int, alive: bool) -> None:
    if env is None:
        return
    try:
        if 0 <= int(aid) < int(getattr(env, "agent_alive").shape[0]):
            getattr(env, "agent_alive")[int(aid)] = bool(alive)
            getattr(env, "agent_pos_y")[int(aid)] = int(y)
            getattr(env, "agent_pos_x")[int(aid)] = int(x)
    except Exception:
        return


def _write_lifecycle_payload(runner: Any, payload: dict[str, Any], events: list[dict[str, Any]]) -> None:
    raw_run_dir = str(getattr(runner, "run_dir", "") or getattr(runner, "run_dir_path", "") or "")
    if not raw_run_dir:
        return
    out_dir = Path(raw_run_dir) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "cxxrewrite1_lifecycle_summary.jsonl"
    churn_path = out_dir / "cxxrewrite1_churn_tape.jsonl"
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
        f.write("\n")
    with churn_path.open("a", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
            f.write("\n")


def _alive_agents(runner: Any, alive: Any) -> list[Any]:
    if alive is None:
        alive = getattr(getattr(runner, "pop", None), "alive_agents", lambda: [])()
    return [a for a in list(alive or []) if bool(getattr(a, "alive", False))]


def _extract_hmem_by_slot(runner: Any, cap: int) -> np.ndarray:
    hmem_obj = getattr(runner, "_Hmem", None)
    if hmem_obj is None:
        hmem_obj = getattr(runner, "gru_hmem", None)
    try:
        if hasattr(hmem_obj, "detach"):
            arr = hmem_obj.detach().to("cpu").numpy()
        else:
            arr = np.asarray(hmem_obj)
    except Exception:
        arr = np.empty((0, 0), dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros((max(0, int(cap)), 0), dtype=np.float32)
    out_rows = max(int(cap), int(arr.shape[0]))
    out = np.zeros((out_rows, int(arr.shape[1])), dtype=np.float32)
    out[: int(arr.shape[0]), :] = np.asarray(arr, dtype=np.float32)
    return np.ascontiguousarray(out)


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


def _sha256_json(payload: Any) -> str:
    raw = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
