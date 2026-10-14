from __future__ import annotations

"""CXXBACKBONE-6 resident policy-service vertical-slice prototype."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from env.gridworld_pkg.patch_world import patch_world_observation_enabled
from simulation.runnercore.tick_state import slots_for_agent_ids


SCHEMA_VERSION = "cxxbackbone6_policy_service_v1"


@dataclass(frozen=True)
class ResidentRuntimeStateV6:
    schema_version: str
    generation: int
    tick: int
    capacity: int
    cache_stride: int
    alive_slots: np.ndarray
    alive_agent_ids: np.ndarray
    slot_tokens: np.ndarray
    cache_agent_id: np.ndarray
    cache_token: np.ndarray
    cache_cursor: np.ndarray
    cache_actions: np.ndarray
    pos_y_by_id: np.ndarray
    pos_x_by_id: np.ndarray
    flat_local_observations: np.ndarray
    action_output: np.ndarray
    support_blockers: tuple[str, ...]


def _bucket_add(buckets: dict[str, float] | None, name: str, value: float = 1.0) -> None:
    if buckets is None:
        return
    key = "cxxbackbone6_" + str(name)
    buckets[key] = float(buckets.get(key, 0.0) + float(value))


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def support_blockers(runner: Any) -> tuple[str, ...]:
    blockers: list[str] = []
    cfg = getattr(runner, "sim_cfg", None)
    env_cfg = getattr(runner, "env_cfg", None)
    if not _as_bool(getattr(cfg, "policy_deterministic_actions", False)):
        blockers.append("stochastic_policy_rng_not_yet_native")
    if int(getattr(cfg, "policy_tick_stride", 0) or 0) not in (1, 2):
        blockers.append("policy_tick_stride_must_be_1_or_2")
    if patch_world_observation_enabled(env_cfg):
        blockers.append("patch_observation_not_yet_owned_by_cxxbackbone6_service")
    if _as_bool(getattr(env_cfg, "wdl_delayed_hazard_enable", False)):
        blockers.append("wdl_not_supported")
    if _as_bool(getattr(cfg, "chemistry_functional_coupling_enable", False)):
        blockers.append("chemistry_observation_coupling_not_supported")
    return tuple(blockers)


def _refresh_flat_padded_views(env: Any) -> tuple[np.ndarray, np.ndarray]:
    if bool(getattr(env, "_views_dirty", True)) or getattr(env, "_occ_p", None) is None or getattr(env, "_food_p", None) is None:
        refresh = getattr(env, "_refresh_padded_views", None)
        if not callable(refresh):
            raise RuntimeError("env does not expose padded observation views")
        refresh()
    return np.asarray(env._occ_p, dtype=np.uint8), np.asarray(env._food_p, dtype=np.uint8)


def _flat_local_observations_from_arrays(runner: Any, agent_ids: np.ndarray) -> np.ndarray:
    env = runner.env
    occ_p, food_p = _refresh_flat_padded_views(env)
    ids = np.asarray(agent_ids, dtype=np.int32)
    n = int(ids.shape[0])
    r = int(getattr(runner.env_cfg, "view_radius", 1))
    k = int(2 * r + 1)
    ys = np.asarray(env.agent_pos_y, dtype=np.int32)[ids]
    xs = np.asarray(env.agent_pos_x, dtype=np.int32)[ids]
    out = np.empty((n, k * k), dtype=np.float32)
    for i in range(n):
        yp = int(ys[i]) + r
        xp = int(xs[i]) + r
        t = 0
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if int(occ_p[yp + dy, xp + dx]) != 0:
                    out[i, t] = np.float32(-1.0)
                else:
                    out[i, t] = np.float32(food_p[yp + dy, xp + dx])
                t += 1
    return out


def capture_resident_runtime_state_v6(
    runner: Any,
    state: Any,
    agent_ids: np.ndarray,
    *,
    local_observations: np.ndarray | None = None,
) -> ResidentRuntimeStateV6:
    ids = np.asarray(agent_ids, dtype=np.int32)
    slots = slots_for_agent_ids(state, ids.astype(np.int64, copy=False)).astype(np.int32, copy=False)
    if local_observations is None and not support_blockers(runner):
        local_observations = _flat_local_observations_from_arrays(runner, ids)
    local = (
        np.asarray(local_observations, dtype=np.float32)
        if local_observations is not None
        else np.empty((0, 0), dtype=np.float32)
    )
    return ResidentRuntimeStateV6(
        schema_version=SCHEMA_VERSION,
        generation=int(getattr(runner, "gen", 0) or 0),
        tick=int(getattr(runner, "tick", 0) or 0),
        capacity=int(getattr(state, "capacity", int(slots.shape[0]))),
        cache_stride=int(getattr(state, "stride", 1)),
        alive_slots=slots.copy(),
        alive_agent_ids=ids.copy(),
        slot_tokens=np.asarray(state.slot_tokens, dtype=np.int64).copy(),
        cache_agent_id=np.asarray(state.cache_agent_id, dtype=np.int64).copy(),
        cache_token=np.asarray(state.cache_token, dtype=np.int64).copy(),
        cache_cursor=np.asarray(state.cache_cursor, dtype=np.int32).copy(),
        cache_actions=np.asarray(state.cache_actions, dtype=np.int32).copy(),
        pos_y_by_id=np.asarray(getattr(runner.env, "agent_pos_y"), dtype=np.int32).copy(),
        pos_x_by_id=np.asarray(getattr(runner.env, "agent_pos_x"), dtype=np.int32).copy(),
        flat_local_observations=local.copy(),
        action_output=np.empty((int(ids.shape[0]),), dtype=np.int32),
        support_blockers=support_blockers(runner),
    )


def resident_runtime_state_v6_payload(snapshot: ResidentRuntimeStateV6) -> dict[str, Any]:
    payload = asdict(snapshot)
    for key, value in list(payload.items()):
        if isinstance(value, np.ndarray):
            payload[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sum": float(np.asarray(value, dtype=np.float64).sum()) if value.size else 0.0,
            }
    return payload


def execute_tick_flat_policy_service_v1(
    runner: Any,
    state: Any,
    agent_ids: np.ndarray,
    *,
    buckets: dict[str, float] | None = None,
) -> np.ndarray | None:
    blockers = support_blockers(runner)
    if blockers:
        _bucket_add(buckets, "unsupported_calls")
        return None
    ids_i64 = np.asarray(agent_ids, dtype=np.int64)
    ids_i32 = ids_i64.astype(np.int32, copy=False)
    n = int(ids_i32.shape[0])
    if n <= 0:
        return np.empty((0,), dtype=np.int32)
    stride = int(state.stride)
    slots = slots_for_agent_ids(state, ids_i64).astype(np.int32, copy=False)
    if bool(np.any(slots < 0)):
        _bucket_add(buckets, "unknown_agent_id_failures")
        return None

    _bucket_add(buckets, "service_execute_tick_calls")
    _bucket_add(buckets, "service_rows", float(n))
    cursors = np.asarray(state.cache_cursor[slots], dtype=np.int32)
    hit_mask = (
        (np.asarray(state.cache_agent_id[slots], dtype=np.int64) == ids_i64)
        & (np.asarray(state.cache_token[slots], dtype=np.int64) == np.asarray(state.slot_tokens[slots], dtype=np.int64))
        & (cursors >= np.int32(0))
        & (cursors < np.int32(stride))
    )
    out = np.empty((n,), dtype=np.int32)
    hit_idx = np.nonzero(hit_mask)[0].astype(np.int64)
    if int(hit_idx.shape[0]) > 0:
        hit_slots = slots[hit_idx]
        hit_cursors = cursors[hit_idx]
        out[hit_idx] = state.cache_actions[hit_slots, hit_cursors]
        state.cache_cursor[hit_slots] = (hit_cursors + np.int32(1)).astype(np.int32, copy=False)
        state.counters.action_cache_hits += int(hit_idx.shape[0])
    miss_idx = np.nonzero(~hit_mask)[0].astype(np.int64)
    _bucket_add(buckets, "cache_hits", float(hit_idx.shape[0]))
    _bucket_add(buckets, "cache_misses", float(miss_idx.shape[0]))
    if int(miss_idx.shape[0]) > 0:
        refill_ids = ids_i32[miss_idx].astype(np.int32, copy=False)
        refill_slots = slots[miss_idx].astype(np.int32, copy=False)
        order = np.argsort(refill_ids.astype(np.int64, copy=False), kind="stable")
        sorted_ids = refill_ids[order].astype(np.int32, copy=False)
        sorted_slots = refill_slots[order].astype(np.int32, copy=False)
        local = _flat_local_observations_from_arrays(runner, sorted_ids)
        from simulation.runnercore.action_cache import _compute_actions_k  # noqa: WPS433

        _bucket_add(buckets, "policy_math_delegated_to_compute_actions_k_rows", float(sorted_ids.shape[0]))
        actions_k = _compute_actions_k(
            runner,
            state,
            sorted_ids,
            k=stride,
            buckets=buckets,
            precomputed_slots=sorted_slots,
            precomputed_local_observations=local,
        )
        out_idx = miss_idx[order]
        out[out_idx] = actions_k[:, 0]
        state.cache_agent_id[sorted_slots] = sorted_ids.astype(np.int64, copy=False)
        state.cache_token[sorted_slots] = state.slot_tokens[sorted_slots]
        state.cache_cursor[sorted_slots] = np.int32(1)
        state.cache_actions[sorted_slots, :stride] = actions_k[:, :stride]
        state.counters.action_cache_misses += int(sorted_slots.shape[0])
        state.counters.action_cache_refill_rows += int(sorted_slots.shape[0])
        _bucket_add(buckets, "cache_refill_rows", float(sorted_slots.shape[0]))
        _bucket_add(buckets, "native_flat_observation_rows", float(sorted_ids.shape[0]))
    _bucket_add(buckets, "slot_lookups_inside_lane", 0.0)
    _bucket_add(buckets, "observation_callbacks_inside_lane", 0.0)
    return out.astype(np.int32, copy=False)
