from __future__ import annotations

"""Resident slot-state import for the RUNNERCORE-0 contract."""

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from simulation.runnercore.action_cache_contract import (
    is_slot_keyed_policy_cache_entry,
    resolve_policy_action_cache_entry,
)
from simulation.runnercore.profile import SUPPORTED_PROFILE


SCHEMA_VERSION = "runnercore0_resident_state_v1"
N_ACTIONS = 7
N_DRIVES = 4


@dataclass(frozen=True)
class RunnerCoreState:
    schema_version: str
    profile: str
    generation: int
    tick: int
    capacity: int
    alive_count: int
    agent_ids_by_alive_order: np.ndarray
    slot_to_agent_id: np.ndarray
    agent_id_to_slot_agent_ids: np.ndarray
    agent_id_to_slot_slots: np.ndarray
    slot_tokens: np.ndarray
    alive_mask: np.ndarray
    pos_y: np.ndarray
    pos_x: np.ndarray
    energy: np.ndarray
    health: np.ndarray
    injury: np.ndarray
    age: np.ndarray
    drives: np.ndarray
    hmem: np.ndarray
    policy_attack_bias: np.ndarray
    policy_action_temperature: np.ndarray
    policy_drive_logit_gain: np.ndarray
    policy_use_gru: np.ndarray
    action_cache_token: np.ndarray
    action_cache_cursor: np.ndarray
    action_cache_next: np.ndarray


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


def _alive_agents(runner: Any, alive: Any = None) -> list[Any]:
    if alive is not None:
        return list(alive)
    pop = getattr(runner, "pop", None)
    if pop is not None and hasattr(pop, "alive_agents"):
        return list(pop.alive_agents())
    agents = getattr(pop, "agents", {}) if pop is not None else {}
    if isinstance(agents, dict):
        return [agent for agent in agents.values() if bool(getattr(agent, "alive", True))]
    return [agent for agent in list(agents or []) if bool(getattr(agent, "alive", True))]


def _env_array(env: Any, name: str, dtype: Any, default_len: int) -> np.ndarray:
    arr = getattr(env, name, None)
    if arr is None:
        fill = False if dtype == np.bool_ else -1
        return np.full((int(default_len),), fill, dtype=dtype)
    return np.asarray(arr, dtype=dtype).reshape(-1)


def _slot_capacity(runner: Any, agents: list[Any]) -> int:
    pop = getattr(runner, "pop", None)
    pop_max = _safe_int(getattr(pop, "max_slots", 0), 0)
    max_slot = max([_safe_int(getattr(agent, "slot", -1), -1) for agent in agents] + [-1])
    return max(1, int(pop_max), int(max_slot) + 1)


def _extract_hmem_by_slot(runner: Any, capacity: int) -> np.ndarray:
    for name in ("_Hmem", "Hmem", "gru_hmem", "_gru_hmem"):
        raw = getattr(runner, name, None)
        if raw is None:
            continue
        try:
            if hasattr(raw, "detach"):
                arr = raw.detach().cpu().numpy()
            else:
                arr = np.asarray(raw)
            if arr.ndim == 1:
                arr = arr.reshape((arr.shape[0], 1))
            out = np.zeros((int(capacity), int(arr.shape[1])), dtype=np.float32)
            rows = min(int(capacity), int(arr.shape[0]))
            out[:rows, :] = np.asarray(arr[:rows, :], dtype=np.float32)
            return out
        except Exception:
            continue
    return np.zeros((int(capacity), 0), dtype=np.float32)


def _cache_row(cache: Any, slot: int, agent_id: int | None = None) -> tuple[int, int, int]:
    if agent_id is None:
        return -1, -1, -1
    value = resolve_policy_action_cache_entry(
        cache,
        slot=int(slot),
        agent_id=int(agent_id),
    )
    if value is None:
        return -1, -1, -1
    if isinstance(value, dict):
        return (
            _safe_int(value.get("slot_token", value.get("token", -1)), -1),
            _safe_int(value.get("cursor", value.get("idx", -1)), -1),
            _safe_int(value.get("next_action", value.get("action", -1)), -1),
        )
    if isinstance(value, (tuple, list)):
        slot_keyed = is_slot_keyed_policy_cache_entry(value)
        token_index = 1 if slot_keyed else 0
        actions_index = 2 if slot_keyed else 1
        cursor_index = 3 if slot_keyed else 2
        token = _safe_int(value[token_index], -1) if len(value) > token_index else -1
        if len(value) > actions_index:
            try:
                actions = np.asarray(value[actions_index], dtype=np.int32).reshape(-1)
            except Exception:
                actions = np.asarray([], dtype=np.int32)
            if int(actions.shape[0]) > 0:
                cursor = _safe_int(value[cursor_index], 0) if len(value) > cursor_index else 0
                action_index = cursor if 0 <= cursor < int(actions.shape[0]) else int(actions.shape[0]) - 1
                return token, cursor, _safe_int(actions[action_index], -1)
        cursor = _safe_int(value[cursor_index], -1) if len(value) > cursor_index else -1
        action = -1
        return token, cursor, action
    return -1, -1, -1


def import_generation_start(
    runner: Any,
    alive: Any = None,
    *,
    generation: int = -1,
    tick: int = -1,
    profile: str = SUPPORTED_PROFILE,
) -> RunnerCoreState:
    agents = _alive_agents(runner, alive)
    capacity = _slot_capacity(runner, agents)
    max_aid = max([_safe_int(getattr(agent, "agent_id", -1), -1) for agent in agents] + [-1])
    id_capacity = max(1, max_aid + 1)

    env = getattr(runner, "env", None)
    env_pos_y = _env_array(env, "agent_pos_y", np.int32, id_capacity)
    env_pos_x = _env_array(env, "agent_pos_x", np.int32, id_capacity)

    slot_to_agent_id = np.full((capacity,), -1, dtype=np.int64)
    slot_tokens = np.full((capacity,), -1, dtype=np.int64)
    alive_mask = np.zeros((capacity,), dtype=np.bool_)
    pos_y = np.full((capacity,), -1, dtype=np.int32)
    pos_x = np.full((capacity,), -1, dtype=np.int32)
    energy = np.zeros((capacity,), dtype=np.float32)
    health = np.zeros((capacity,), dtype=np.float32)
    injury = np.zeros((capacity,), dtype=np.float32)
    age = np.zeros((capacity,), dtype=np.int32)
    drives = np.zeros((capacity, N_DRIVES), dtype=np.float32)
    attack_bias = np.zeros((capacity,), dtype=np.float32)
    action_temperature = np.ones((capacity,), dtype=np.float32)
    drive_logit_gain = np.ones((capacity,), dtype=np.float32)
    use_gru = np.zeros((capacity,), dtype=np.bool_)
    action_cache_token = np.full((capacity,), -1, dtype=np.int64)
    action_cache_cursor = np.full((capacity,), -1, dtype=np.int32)
    action_cache_next = np.full((capacity,), -1, dtype=np.int32)

    by_agent_id: list[tuple[int, int]] = []
    cache = getattr(runner, "_policy_action_cache", getattr(runner, "policy_action_cache", None))
    for agent in agents:
        aid = _safe_int(getattr(agent, "agent_id", -1), -1)
        slot = _safe_int(getattr(agent, "slot", -1), -1)
        if aid < 0 or slot < 0 or slot >= capacity:
            continue
        genome = getattr(agent, "genome", None)
        slot_to_agent_id[slot] = np.int64(aid)
        slot_tokens[slot] = np.int64(_safe_int(getattr(agent, "slot_token", getattr(agent, "_slot_token", 0)), 0))
        alive_mask[slot] = True
        pos_y[slot] = np.int32(env_pos_y[aid] if 0 <= aid < int(env_pos_y.shape[0]) else getattr(agent, "y", -1))
        pos_x[slot] = np.int32(env_pos_x[aid] if 0 <= aid < int(env_pos_x.shape[0]) else getattr(agent, "x", -1))
        energy[slot] = np.float32(_safe_float(getattr(agent, "energy", 0.0), 0.0))
        health[slot] = np.float32(_safe_float(getattr(agent, "health", 0.0), 0.0))
        injury[slot] = np.float32(_safe_float(getattr(agent, "injury", 0.0), 0.0))
        age[slot] = np.int32(_safe_int(getattr(agent, "age", getattr(agent, "steps_alive", 0)), 0))

        d = np.asarray(getattr(agent, "drives", np.zeros((N_DRIVES,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        drives[slot, : min(N_DRIVES, int(d.shape[0]))] = d[:N_DRIVES]
        attack_bias[slot] = np.float32(_safe_float(getattr(genome, "aggression_bias", 0.0), 0.0))
        action_temperature[slot] = np.float32(_safe_float(getattr(genome, "action_temperature", 1.0), 1.0))
        drive_logit_gain[slot] = np.float32(_safe_float(getattr(genome, "drive_logit_gain", 1.0), 1.0))
        use_gru[slot] = bool(getattr(genome, "use_gru", False))
        tok, cursor, nxt = _cache_row(cache, slot, aid)
        action_cache_token[slot] = np.int64(tok)
        action_cache_cursor[slot] = np.int32(cursor)
        action_cache_next[slot] = np.int32(nxt)
        by_agent_id.append((aid, slot))

    by_agent_id.sort(key=lambda item: item[0])
    return RunnerCoreState(
        schema_version=SCHEMA_VERSION,
        profile=str(profile),
        generation=int(generation),
        tick=int(tick),
        capacity=int(capacity),
        alive_count=int(len(by_agent_id)),
        agent_ids_by_alive_order=np.asarray(
            [_safe_int(getattr(agent, "agent_id", -1), -1) for agent in agents],
            dtype=np.int64,
        ),
        slot_to_agent_id=slot_to_agent_id,
        agent_id_to_slot_agent_ids=np.asarray([aid for aid, _slot in by_agent_id], dtype=np.int64),
        agent_id_to_slot_slots=np.asarray([slot for _aid, slot in by_agent_id], dtype=np.int32),
        slot_tokens=slot_tokens,
        alive_mask=alive_mask,
        pos_y=pos_y,
        pos_x=pos_x,
        energy=energy,
        health=health,
        injury=injury,
        age=age,
        drives=drives,
        hmem=_extract_hmem_by_slot(runner, capacity),
        policy_attack_bias=attack_bias,
        policy_action_temperature=action_temperature,
        policy_drive_logit_gain=drive_logit_gain,
        policy_use_gru=use_gru,
        action_cache_token=action_cache_token,
        action_cache_cursor=action_cache_cursor,
        action_cache_next=action_cache_next,
    )


def state_digest(state: RunnerCoreState) -> dict[str, Any]:
    arrays = {
        "agent_ids_by_alive_order": state.agent_ids_by_alive_order,
        "slot_to_agent_id": state.slot_to_agent_id,
        "slot_tokens": state.slot_tokens,
        "alive_mask": state.alive_mask,
        "pos_y": state.pos_y,
        "pos_x": state.pos_x,
        "energy": state.energy,
        "health": state.health,
        "injury": state.injury,
        "age": state.age,
        "drives": state.drives,
        "hmem": state.hmem,
        "policy_attack_bias": state.policy_attack_bias,
        "policy_action_temperature": state.policy_action_temperature,
        "policy_drive_logit_gain": state.policy_drive_logit_gain,
        "policy_use_gru": state.policy_use_gru,
        "action_cache_token": state.action_cache_token,
        "action_cache_cursor": state.action_cache_cursor,
        "action_cache_next": state.action_cache_next,
    }
    h = hashlib.sha256()
    h.update(str(state.schema_version).encode("utf-8"))
    h.update(str(state.profile).encode("utf-8"))
    h.update(int(state.generation).to_bytes(8, "little", signed=True))
    h.update(int(state.tick).to_bytes(8, "little", signed=True))
    for name in sorted(arrays):
        arr = np.ascontiguousarray(arrays[name])
        h.update(name.encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(json.dumps([int(x) for x in arr.shape]).encode("utf-8"))
        h.update(arr.tobytes())
    return {
        "schema_version": "runnercore0_state_digest_v1",
        "generation": int(state.generation),
        "tick": int(state.tick),
        "profile": str(state.profile),
        "capacity": int(state.capacity),
        "alive_count": int(state.alive_count),
        "digest_sha256": h.hexdigest(),
    }


def state_summary_payload(state: RunnerCoreState) -> dict[str, Any]:
    digest = state_digest(state)
    return {
        "schema_version": "runnercore0_state_summary_v1",
        "digest": digest,
        "array_shapes": {
            "slot_to_agent_id": [int(x) for x in state.slot_to_agent_id.shape],
            "agent_ids_by_alive_order": [int(x) for x in state.agent_ids_by_alive_order.shape],
            "drives": [int(x) for x in state.drives.shape],
            "hmem": [int(x) for x in state.hmem.shape],
        },
        "slot_token_rows": int(state.slot_tokens.shape[0]),
        "alive_order_head": [int(x) for x in state.agent_ids_by_alive_order[:16].tolist()],
    }


def validate_state_against_runner(state: RunnerCoreState, runner: Any, alive: Any = None) -> dict[str, Any]:
    agents = _alive_agents(runner, alive)
    mismatches: list[dict[str, Any]] = []
    for agent in agents:
        aid = _safe_int(getattr(agent, "agent_id", -1), -1)
        slot = _safe_int(getattr(agent, "slot", -1), -1)
        if slot < 0 or slot >= int(state.capacity):
            mismatches.append({"agent_id": aid, "slot": slot, "field": "slot_range"})
            continue
        checks = {
            "slot_to_agent_id": int(state.slot_to_agent_id[slot]) == aid,
            "energy": np.isclose(float(state.energy[slot]), _safe_float(getattr(agent, "energy", 0.0), 0.0)),
            "health": np.isclose(float(state.health[slot]), _safe_float(getattr(agent, "health", 0.0), 0.0)),
            "injury": np.isclose(float(state.injury[slot]), _safe_float(getattr(agent, "injury", 0.0), 0.0)),
        }
        for field, ok in checks.items():
            if not bool(ok):
                mismatches.append({"agent_id": aid, "slot": slot, "field": field})
                break
    return {
        "schema_version": "runnercore0_state_validation_v1",
        "ok": not mismatches,
        "alive_count": int(len(agents)),
        "mismatch_count": int(len(mismatches)),
        "first_mismatches": mismatches[:8],
    }
