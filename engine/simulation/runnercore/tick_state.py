from __future__ import annotations

"""RUNNERCORE-1 resident array tick state.

The candidate lane imports Python Agent objects at a generation boundary, then
uses this slot-indexed state as live truth for supported ticks. Agent mirrors are
updated only by explicit exporters.
"""

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np

from agents.agent import N_ACTIONS, N_DRIVES
from agents.chemistry import CHEMISTRY_SIZE, chemistry_from_values, zeros_chemistry
from agents.maintenance import MAINTENANCE_SIZE, maintenance_from_values, zeros_maintenance_state
from agents.cognition import (
    compute_env_cost_charge_scalar_from_surface,
    effective_memory_dim_for_genome,
    express_cost_charge_surface,
    memory_cost_components,
)
from agents.perception import express_perception
from simulation.runnercore.action_cache_contract import (
    is_slot_keyed_policy_cache_entry,
    resolve_policy_action_cache_entry,
)


SCHEMA_VERSION = "runnercore1_array_tick_state_v1"


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass
class RunnerCore1Counters:
    boundary_agent_import_rows: int = 0
    boundary_agent_export_rows: int = 0
    event_boundary_agent_export_rows: int = 0
    event_boundary_agent_create_rows: int = 0
    boundary_policy_bank_sync_rows: int = 0
    event_boundary_policy_bank_sync_rows: int = 0
    boundary_churn_materialization_rows: int = 0
    per_tick_agent_sync_rows: int = 0
    legacy_batched_actions_calls: int = 0
    legacy_batched_actions_k_calls: int = 0
    legacy_batched_actions_strided_calls: int = 0
    legacy_action_cache_refill_calls: int = 0
    action_cache_hits: int = 0
    action_cache_misses: int = 0
    action_cache_refill_rows: int = 0
    action_cache_invalidate_events: int = 0
    ticks_ran: int = 0
    env_deaths: int = 0
    death_events_applied: int = 0
    birth_events_applied: int = 0
    refill_events_applied: int = 0
    policy_row_attach_events: int = 0
    hmem_row_init_events: int = 0
    slot_token_attach_events: int = 0
    churn_events_total: int = 0
    unsupported_fallback_count: int = 0


@dataclass
class RunnerCore1TickState:
    schema_version: str
    profile: str
    generation: int
    capacity: int
    stride: int
    slot_to_agent_id: np.ndarray
    agent_id_to_slot_agent_ids: np.ndarray
    agent_id_to_slot_slots: np.ndarray
    slot_tokens: np.ndarray
    alive: np.ndarray
    energy: np.ndarray
    health: np.ndarray
    injury: np.ndarray
    scar: np.ndarray
    drives: np.ndarray
    chemistry: np.ndarray
    maintenance: np.ndarray
    aggr: np.ndarray
    temp: np.ndarray
    gain: np.ndarray
    drive_inertia: np.ndarray
    drive_noise: np.ndarray
    pred_lr: np.ndarray
    repro_threshold: np.ndarray
    gru_upkeep_cost: np.ndarray
    sensor_channel_mask: np.ndarray
    sensor_radius: np.ndarray
    chemistry_observation_fidelity: np.ndarray
    chemistry_observation_bias: np.ndarray
    cost_charge_scalar: np.ndarray
    cost_charge_maturation_scalar: np.ndarray
    use_gru: np.ndarray
    mem_dim: np.ndarray
    cache_agent_id: np.ndarray
    cache_token: np.ndarray
    cache_cursor: np.ndarray
    cache_actions: np.ndarray
    survival_steps: np.ndarray
    energy_collected: np.ndarray
    early_survival_steps: np.ndarray
    early_energy_collected: np.ndarray
    early_injury_gained: np.ndarray
    last_energy_delta: np.ndarray
    last_health_delta: np.ndarray
    last_injury_delta: np.ndarray
    last_action: np.ndarray
    last_last_action: np.ndarray
    action_switches: np.ndarray
    action_oscillations: np.ndarray
    episode0_switches: np.ndarray
    episode0_oscillations: np.ndarray
    attacks: np.ndarray
    idle_charge_paid: np.ndarray
    maintenance_requests: np.ndarray
    maintenance_applied: np.ndarray
    maintenance_energy_spent_total: np.ndarray
    maintenance_opportunity_cost_total: np.ndarray
    maintenance_applied_effect_total: np.ndarray
    selfharm_events: np.ndarray
    episode0_selfharm_events: np.ndarray
    drive_volatility: np.ndarray
    drive_jitter: np.ndarray
    episode0_drive_jitter_sum: np.ndarray
    prev_drive_delta: np.ndarray
    pred_err_sum: np.ndarray
    pred_err_count: np.ndarray
    prediction_update_count: np.ndarray
    prediction_state_delta_abs_sum: np.ndarray
    prediction_state_delta_abs_max: np.ndarray
    prediction_reset_count: np.ndarray
    episode0_pred_err_sum: np.ndarray
    episode0_pred_err_count: np.ndarray
    late_pred_err_sum: np.ndarray
    late_pred_err_count: np.ndarray
    adult_pred_err_ema: np.ndarray
    exp_dE: np.ndarray
    exp_dInj: np.ndarray
    pred_queue: list[deque]
    first_repro_step: list[Any]
    rest_streak: np.ndarray
    repro_cooldown: np.ndarray
    offspring_count: np.ndarray
    biome_id: np.ndarray
    early_life_weight: np.ndarray
    counters: RunnerCore1Counters = field(default_factory=RunnerCore1Counters)
    first_divergence: dict[str, Any] | None = None
    churn_events: list[dict[str, Any]] = field(default_factory=list)
    runnercore3_profile_buckets_ms: dict[str, float] = field(default_factory=dict)
    runnercore3_profile_counts: dict[str, int] = field(default_factory=dict)

    def slot_for_agent_id(self, agent_id: int) -> int:
        aid = int(agent_id)
        if aid < 0 or not self.agent_id_to_slot_agent_ids.size:
            return -1
        index = int(np.searchsorted(self.agent_id_to_slot_agent_ids, aid))
        if (
            index >= int(self.agent_id_to_slot_agent_ids.shape[0])
            or int(self.agent_id_to_slot_agent_ids[index]) != aid
        ):
            return -1
        return int(self.agent_id_to_slot_slots[index])


def runnercore3_profile_add(state: RunnerCore1TickState, name: str, seconds: float) -> None:
    if seconds < 0.0:
        return
    key = str(name)
    state.runnercore3_profile_buckets_ms[key] = float(state.runnercore3_profile_buckets_ms.get(key, 0.0)) + (
        float(seconds) * 1000.0
    )


def runnercore3_profile_count(state: RunnerCore1TickState, name: str, value: int = 1) -> None:
    key = str(name)
    state.runnercore3_profile_counts[key] = int(state.runnercore3_profile_counts.get(key, 0)) + int(value)


def slots_for_agent_ids(state: RunnerCore1TickState, agent_ids: Any) -> np.ndarray:
    ids = np.asarray(agent_ids, dtype=np.int64)
    out = np.full(ids.shape, -1, dtype=np.int32)
    keys = state.agent_id_to_slot_agent_ids
    if not keys.size:
        return out
    nonnegative = ids >= 0
    positions = np.searchsorted(keys, ids[nonnegative])
    within = positions < int(keys.shape[0])
    matched = np.zeros(positions.shape, dtype=np.bool_)
    matched[within] = keys[positions[within]] == ids[nonnegative][within]
    selected = np.flatnonzero(nonnegative)
    out.reshape(-1)[selected[matched]] = state.agent_id_to_slot_slots[
        positions[matched]
    ]
    return out


def _alive_agents(runner: Any) -> list[Any]:
    pop = getattr(runner, "pop", None)
    if pop is None:
        return []
    agents = list(pop.alive_agents()) if hasattr(pop, "alive_agents") else list(getattr(pop, "agents", {}).values())
    return [agent for agent in agents if _bool(getattr(agent, "alive", True))]


def _capacity(runner: Any, agents: list[Any]) -> int:
    pop = getattr(runner, "pop", None)
    max_slot = max([_as_int(getattr(agent, "slot", -1), -1) for agent in agents] + [-1])
    pop_slots = _as_int(getattr(pop, "max_slots", 0), 0)
    return max(1, pop_slots, max_slot + 1)


def _stat(agent: Any) -> Any:
    return getattr(agent, "stats", None)


def _queue_copy(value: Any) -> deque:
    try:
        return deque(value)
    except Exception:
        return deque()


def _fill_stat_arrays(state: RunnerCore1TickState, agent: Any, slot: int) -> None:
    st = _stat(agent)
    if st is None:
        return
    state.survival_steps[slot] = np.int32(_as_int(getattr(st, "survival_steps", 0), 0))
    state.energy_collected[slot] = np.float32(_as_float(getattr(st, "energy_collected", 0.0), 0.0))
    state.early_survival_steps[slot] = np.int32(_as_int(getattr(st, "early_survival_steps", 0), 0))
    state.early_energy_collected[slot] = np.float32(_as_float(getattr(st, "early_energy_collected", 0.0), 0.0))
    state.early_injury_gained[slot] = np.float32(_as_float(getattr(st, "early_injury_gained", 0.0), 0.0))
    state.last_energy_delta[slot] = np.float32(_as_float(getattr(st, "last_energy_delta", 0.0), 0.0))
    state.last_health_delta[slot] = np.float32(_as_float(getattr(st, "last_health_delta", 0.0), 0.0))
    state.last_injury_delta[slot] = np.float32(_as_float(getattr(st, "last_injury_delta", 0.0), 0.0))
    state.last_action[slot] = np.int32(_as_int(getattr(st, "last_action", -1), -1))
    state.last_last_action[slot] = np.int32(_as_int(getattr(st, "last_last_action", -1), -1))
    state.action_switches[slot] = np.int32(_as_int(getattr(st, "action_switches", 0), 0))
    state.action_oscillations[slot] = np.int32(_as_int(getattr(st, "action_oscillations", 0), 0))
    state.episode0_switches[slot] = np.int32(_as_int(getattr(st, "episode0_switches", 0), 0))
    state.episode0_oscillations[slot] = np.int32(_as_int(getattr(st, "episode0_oscillations", 0), 0))
    state.attacks[slot] = np.int32(_as_int(getattr(st, "attacks", 0), 0))
    state.idle_charge_paid[slot] = np.float32(_as_float(getattr(st, "idle_charge_paid", 0.0), 0.0))
    state.maintenance_requests[slot] = np.int32(_as_int(getattr(st, "maintenance_requests", 0), 0))
    state.maintenance_applied[slot] = np.int32(_as_int(getattr(st, "maintenance_applied", 0), 0))
    state.maintenance_energy_spent_total[slot] = np.float32(
        _as_float(getattr(st, "maintenance_energy_spent_total", 0.0), 0.0)
    )
    state.maintenance_opportunity_cost_total[slot] = np.float32(
        _as_float(getattr(st, "maintenance_opportunity_cost_total", 0.0), 0.0)
    )
    state.maintenance_applied_effect_total[slot] = np.float32(
        _as_float(getattr(st, "maintenance_applied_effect_total", 0.0), 0.0)
    )
    state.chemistry_observation_fidelity[slot] = np.float32(
        _as_float(getattr(st, "chemistry_observation_fidelity", 1.0), 1.0)
    )
    state.chemistry_observation_bias[slot] = np.float32(_as_float(getattr(st, "chemistry_observation_bias", 0.0), 0.0))
    state.selfharm_events[slot] = np.int32(_as_int(getattr(st, "selfharm_events", 0), 0))
    state.episode0_selfharm_events[slot] = np.int32(_as_int(getattr(st, "episode0_selfharm_events", 0), 0))
    state.drive_volatility[slot] = np.float32(_as_float(getattr(st, "drive_volatility", 0.0), 0.0))
    state.drive_jitter[slot] = np.float32(_as_float(getattr(st, "drive_jitter", 0.0), 0.0))
    state.episode0_drive_jitter_sum[slot] = np.float32(
        _as_float(getattr(st, "episode0_drive_jitter_sum", 0.0), 0.0)
    )
    prev_delta = np.asarray(getattr(st, "prev_drive_delta", np.zeros((N_DRIVES,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    state.prev_drive_delta[slot, : min(N_DRIVES, int(prev_delta.shape[0]))] = prev_delta[:N_DRIVES]
    state.pred_err_sum[slot] = np.float32(_as_float(getattr(st, "pred_err_sum", 0.0), 0.0))
    state.pred_err_count[slot] = np.int32(_as_int(getattr(st, "pred_err_count", 0), 0))
    state.prediction_update_count[slot] = np.int32(_as_int(getattr(st, "prediction_update_count", 0), 0))
    state.prediction_state_delta_abs_sum[slot] = np.float32(
        _as_float(getattr(st, "prediction_state_delta_abs_sum", 0.0), 0.0)
    )
    state.prediction_state_delta_abs_max[slot] = np.float32(
        _as_float(getattr(st, "prediction_state_delta_abs_max", 0.0), 0.0)
    )
    state.prediction_reset_count[slot] = np.int32(_as_int(getattr(st, "prediction_reset_count", 0), 0))
    state.episode0_pred_err_sum[slot] = np.float32(_as_float(getattr(st, "episode0_pred_err_sum", 0.0), 0.0))
    state.episode0_pred_err_count[slot] = np.int32(_as_int(getattr(st, "episode0_pred_err_count", 0), 0))
    state.late_pred_err_sum[slot] = np.float32(_as_float(getattr(st, "late_pred_err_sum", 0.0), 0.0))
    state.late_pred_err_count[slot] = np.int32(_as_int(getattr(st, "late_pred_err_count", 0), 0))
    state.adult_pred_err_ema[slot] = np.float32(_as_float(getattr(st, "adult_pred_err_ema", 0.0), 0.0))
    state.pred_queue[slot] = _queue_copy(getattr(st, "pred_queue", ()))
    state.first_repro_step[slot] = getattr(st, "first_repro_step", None)
    de = np.asarray(getattr(st, "exp_dE", np.zeros((N_ACTIONS,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    di = np.asarray(getattr(st, "exp_dInj", np.zeros((N_ACTIONS,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    state.exp_dE[slot, : min(N_ACTIONS, int(de.shape[0]))] = de[:N_ACTIONS]
    state.exp_dInj[slot, : min(N_ACTIONS, int(di.shape[0]))] = di[:N_ACTIONS]


def _fill_policy_observation_arrays(state: RunnerCore1TickState, runner: Any, genome: Any, slot: int) -> None:
    try:
        expr = express_perception(
            genome,
            env_cfg=getattr(runner, "env_cfg", None),
            sim_cfg=getattr(runner, "sim_cfg", None),
        )
        state.sensor_channel_mask[slot] = np.int32(int(getattr(expr, "active_sensor_channel_mask", 0)))
        state.sensor_radius[slot] = np.int32(int(getattr(expr, "active_sensor_radius", 1)))
    except Exception:
        state.sensor_channel_mask[slot] = np.int32(0)
        state.sensor_radius[slot] = np.int32(1)


def _fill_cost_charge_arrays(state: RunnerCore1TickState, runner: Any, agent: Any, genome: Any, slot: int) -> None:
    if not _bool(getattr(getattr(runner, "sim_cfg", None), "runnercore21_cost_charge_surface_arrays_enable", False)):
        state.cost_charge_scalar[slot] = np.float32(0.0)
        state.cost_charge_maturation_scalar[slot] = np.float32(0.0)
        return
    try:
        surface = express_cost_charge_surface(
            genome,
            agent=agent,
            env_cfg=getattr(runner, "env_cfg", None),
            sim_cfg=getattr(runner, "sim_cfg", None),
        )
        total = max(
            0.0,
            _as_float(compute_env_cost_charge_scalar_from_surface(surface, sim_cfg=getattr(runner, "sim_cfg", None)), 0.0),
        )
        maturation = max(0.0, _as_float(getattr(surface, "maturation_cost_charged", 0.0), 0.0))
        state.cost_charge_maturation_scalar[slot] = np.float32(maturation)
        state.cost_charge_scalar[slot] = np.float32(max(0.0, float(total) - float(maturation)))
    except Exception as exc:
        raise RuntimeError("runnercore21 cost charge surface scalar import failed") from exc


def sparse_agent_id_to_slot_pairs(
    agent_ids_or_dense: Any,
    slots: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize sparse pairs or a legacy dense reverse map to sorted pairs."""

    if slots is None:
        dense = np.asarray(agent_ids_or_dense, dtype=np.int32)
        if dense.ndim != 1:
            raise ValueError("legacy dense agent_id_to_slot must be rank one")
        agent_ids = np.flatnonzero(dense >= 0).astype(np.int64, copy=False)
        slot_values = dense[agent_ids].astype(np.int32, copy=True)
    else:
        agent_ids = np.asarray(agent_ids_or_dense, dtype=np.int64).reshape(-1)
        slot_values = np.asarray(slots, dtype=np.int32).reshape(-1)
        if agent_ids.shape != slot_values.shape:
            raise ValueError("sparse agent ID and slot arrays must have equal shape")
        if bool(np.any(agent_ids < 0)) or bool(np.any(slot_values < 0)):
            raise ValueError("sparse agent ID and slot arrays must be nonnegative")
        order = np.argsort(agent_ids, kind="stable")
        agent_ids = agent_ids[order]
        slot_values = slot_values[order]
    if agent_ids.size and (
        int(np.unique(agent_ids).size) != int(agent_ids.size)
        or int(np.unique(slot_values).size) != int(slot_values.size)
    ):
        raise ValueError("sparse agent ID and slot arrays must be one-to-one")
    return (
        np.ascontiguousarray(agent_ids, dtype=np.int64),
        np.ascontiguousarray(slot_values, dtype=np.int32),
    )


def replace_agent_id_to_slot_index(
    state: RunnerCore1TickState,
    agent_ids_or_dense: Any,
    slots: Any | None = None,
) -> None:
    agent_ids, slot_values = sparse_agent_id_to_slot_pairs(
        agent_ids_or_dense,
        slots,
    )
    if slot_values.size and bool(np.any(slot_values >= int(state.capacity))):
        raise ValueError("sparse agent slot exceeds resident capacity")
    state.agent_id_to_slot_agent_ids = agent_ids
    state.agent_id_to_slot_slots = slot_values


def bind_agent_id_to_slot(
    state: RunnerCore1TickState,
    agent_id: int,
    slot: int,
) -> None:
    aid = int(agent_id)
    slot_i = int(slot)
    if aid < 0 or slot_i < 0 or slot_i >= int(state.capacity):
        raise ValueError("agent ID binding is outside resident authority")
    keys = state.agent_id_to_slot_agent_ids
    values = state.agent_id_to_slot_slots
    keep = (keys != aid) & (values != slot_i)
    keys = keys[keep]
    values = values[keep]
    index = int(np.searchsorted(keys, aid))
    state.agent_id_to_slot_agent_ids = np.insert(keys, index, aid).astype(
        np.int64,
        copy=False,
    )
    state.agent_id_to_slot_slots = np.insert(values, index, slot_i).astype(
        np.int32,
        copy=False,
    )


def legacy_dense_agent_id_to_slot(
    state: RunnerCore1TickState,
    *,
    max_width: int,
) -> np.ndarray:
    """Materialize a bounded compatibility map for a legacy native call only."""

    keys = state.agent_id_to_slot_agent_ids
    width = max(1, int(keys[-1]) + 1) if keys.size else 1
    if width > int(max_width):
        raise ValueError(
            "legacy dense agent-ID grouping exceeds its bounded compatibility width"
        )
    dense = np.full((width,), -1, dtype=np.int32)
    if keys.size:
        dense[keys] = state.agent_id_to_slot_slots
    return dense


def invalidate_action_cache_slot(
    state: RunnerCore1TickState,
    slot: int,
    *,
    reason: str,
    agent_id: int | None = None,
) -> None:
    slot_i = int(slot)
    if slot_i < 0 or slot_i >= int(state.capacity):
        return
    state.cache_agent_id[slot_i] = np.int64(-1)
    state.cache_token[slot_i] = np.int64(-1)
    state.cache_cursor[slot_i] = np.int32(-1)
    state.cache_actions[slot_i, :] = np.int32(-1)
    state.counters.action_cache_invalidate_events += 1


def record_churn_event(
    state: RunnerCore1TickState,
    *,
    event_type: str,
    generation: int,
    tick: int,
    slot: int | None = None,
    agent_id: int | None = None,
    reason: str = "",
    **fields: Any,
) -> None:
    event = {
        "schema_version": "runnercore2_churn_event_v1",
        "order": int(len(state.churn_events)),
        "event_type": str(event_type),
        "generation": int(generation),
        "tick": int(tick),
        "slot": int(slot) if slot is not None else -1,
        "agent_id": int(agent_id) if agent_id is not None else -1,
        "reason": str(reason),
    }
    event.update(fields)
    state.churn_events.append(event)
    state.counters.churn_events_total += 1


def sync_agent_row_from_mirror(
    state: RunnerCore1TickState,
    runner: Any,
    agent: Any,
    *,
    is_new: bool = False,
    invalidate_cache_on_token_change: bool = True,
) -> int:
    aid = _as_int(getattr(agent, "agent_id", -1), -1)
    slot = _as_int(getattr(agent, "slot", -1), -1)
    if aid < 0 or slot < 0 or slot >= int(state.capacity):
        return -1
    old_token = int(state.slot_tokens[slot])
    token = _as_int(getattr(agent, "slot_token", 0), 0)
    genome = getattr(agent, "genome", None)
    profile_enabled = _bool(getattr(getattr(runner, "sim_cfg", None), "runnercore3_profile_enable", False))

    if hasattr(runner, "_sync_drive_slot"):
        runner._sync_drive_slot(agent)

    t_slot = time.perf_counter() if profile_enabled else 0.0
    state.slot_to_agent_id[slot] = np.int64(aid)
    bind_agent_id_to_slot(state, aid, slot)
    state.slot_tokens[slot] = np.int64(token)
    state.alive[slot] = bool(getattr(agent, "alive", True))
    if profile_enabled:
        runnercore3_profile_add(state, "slot_attach_ms", time.perf_counter() - t_slot)

    state.energy[slot] = np.float32(_as_float(getattr(agent, "energy", 0.0), 0.0))
    state.health[slot] = np.float32(_as_float(getattr(agent, "health", 0.0), 0.0))
    state.injury[slot] = np.float32(_as_float(getattr(agent, "injury", 0.0), 0.0))
    state.scar[slot] = np.float32(_as_float(getattr(agent, "scar", 0.0), 0.0))
    d = np.asarray(getattr(agent, "drives", np.zeros((N_DRIVES,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    state.drives[slot, :] = 0.0
    state.drives[slot, : min(N_DRIVES, int(d.shape[0]))] = d[:N_DRIVES]
    if hasattr(runner, "_drive_hunger") and getattr(runner, "_drive_hunger", None) is not None:
        if slot < int(runner._drive_hunger.shape[0]):
            state.drives[slot, 0] = np.float32(runner._drive_hunger[slot])
            state.drives[slot, 1] = np.float32(runner._drive_pain[slot])
            state.drives[slot, 2] = np.float32(runner._drive_fatigue[slot])
            state.drives[slot, 3] = np.float32(runner._drive_fear[slot])
    c_raw = getattr(agent, "chemistry", None)
    state.chemistry[slot, :] = chemistry_from_values(c_raw) if c_raw is not None else zeros_chemistry()
    m_raw = getattr(agent, "maintenance", None)
    state.maintenance[slot, :] = maintenance_from_values(m_raw) if m_raw is not None else zeros_maintenance_state()
    t_policy = time.perf_counter() if profile_enabled and genome is not None else 0.0
    if genome is not None:
        state.aggr[slot] = np.float32(_as_float(getattr(genome, "aggression_bias", 0.0), 0.0))
        state.temp[slot] = np.float32(_as_float(getattr(genome, "action_temperature", 1.0), 1.0))
        state.gain[slot] = np.float32(_as_float(getattr(genome, "drive_logit_gain", 1.0), 1.0))
        state.drive_inertia[slot] = np.float32(min(0.999, max(0.0, _as_float(getattr(genome, "drive_inertia", 0.85), 0.85))))
        state.drive_noise[slot] = np.float32(min(0.10, max(0.0, _as_float(getattr(genome, "drive_noise", 0.02), 0.02))))
        state.pred_lr[slot] = np.float32(min(1.0, max(0.0, _as_float(getattr(genome, "pred_lr", 0.08), 0.08))))
        state.repro_threshold[slot] = np.float32(_as_float(getattr(genome, "reproduction_threshold", 30.0), 30.0))
        state.use_gru[slot] = bool(getattr(genome, "use_gru", False))
        state.mem_dim[slot] = np.int32(effective_memory_dim_for_genome(genome, sim_cfg=getattr(runner, "sim_cfg", None)))
        _fill_policy_observation_arrays(state, runner, genome, slot)
        _fill_cost_charge_arrays(state, runner, agent, genome, slot)
        try:
            parts = memory_cost_components(genome, env_cfg=getattr(runner, "env_cfg", None), sim_cfg=getattr(runner, "sim_cfg", None))
            state.gru_upkeep_cost[slot] = np.float32(max(0.0, _as_float(parts.get("charged", 0.0), 0.0)))
        except Exception:
            state.gru_upkeep_cost[slot] = np.float32(0.0)
    if profile_enabled and genome is not None:
        runnercore3_profile_add(state, "policy_row_attach_ms", time.perf_counter() - t_policy)
    state.rest_streak[slot] = np.int32(_as_int(getattr(agent, "rest_streak", 0), 0))
    state.repro_cooldown[slot] = np.int32(_as_int(getattr(agent, "repro_cooldown", 0), 0))
    state.offspring_count[slot] = np.int32(_as_int(getattr(_stat(agent), "offspring_count", 0), 0))
    _fill_stat_arrays(state, agent, slot)
    if invalidate_cache_on_token_change and (is_new or int(old_token) != int(token)):
        t_inv = time.perf_counter() if profile_enabled else 0.0
        invalidate_action_cache_slot(state, slot, reason="slot_token_attach", agent_id=aid)
        if profile_enabled:
            runnercore3_profile_add(state, "action_cache_invalidate_ms", time.perf_counter() - t_inv)
        state.counters.slot_token_attach_events += 1
    return int(slot)


def sync_newborn_row_from_contract(
    state: RunnerCore1TickState,
    runner: Any,
    agent: Any,
    *,
    invalidate_cache_on_token_change: bool = True,
) -> int:
    """Attach a newborn row directly to resident arrays.

    This is deliberately narrower than ``sync_agent_row_from_mirror``. It only
    supports just-created children whose lifetime counters are at birth defaults,
    while preserving deterministic birth drives and inherited expectation priors.
    """
    aid = _as_int(getattr(agent, "agent_id", -1), -1)
    slot = _as_int(getattr(agent, "slot", -1), -1)
    if aid < 0 or slot < 0 or slot >= int(state.capacity):
        return -1
    old_token = int(state.slot_tokens[slot])
    token = _as_int(getattr(agent, "slot_token", 0), 0)
    genome = getattr(agent, "genome", None)
    if genome is None:
        return -1

    state.slot_to_agent_id[slot] = np.int64(aid)
    bind_agent_id_to_slot(state, aid, slot)
    state.slot_tokens[slot] = np.int64(token)
    state.alive[slot] = bool(getattr(agent, "alive", True))

    state.energy[slot] = np.float32(_as_float(getattr(agent, "energy", 0.0), 0.0))
    state.health[slot] = np.float32(_as_float(getattr(agent, "health", 0.0), 0.0))
    state.injury[slot] = np.float32(_as_float(getattr(agent, "injury", 0.0), 0.0))
    state.scar[slot] = np.float32(_as_float(getattr(agent, "scar", 0.0), 0.0))
    state.drives[slot, :] = 0.0
    d = np.asarray(getattr(agent, "drives", np.zeros((N_DRIVES,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    state.drives[slot, : min(N_DRIVES, int(d.shape[0]))] = d[:N_DRIVES]

    if hasattr(runner, "_ensure_drive_arrays"):
        runner._ensure_drive_arrays(required_slots=int(state.capacity))
    if hasattr(runner, "_drive_hunger") and getattr(runner, "_drive_hunger", None) is not None:
        if slot < int(runner._drive_hunger.shape[0]):
            runner._drive_hunger[slot] = np.float32(state.drives[slot, 0])
            runner._drive_pain[slot] = np.float32(state.drives[slot, 1])
            runner._drive_fatigue[slot] = np.float32(state.drives[slot, 2])
            runner._drive_fear[slot] = np.float32(state.drives[slot, 3])
            if getattr(runner, "_drive_slot_token", None) is not None and slot < int(runner._drive_slot_token.shape[0]):
                runner._drive_slot_token[slot] = np.int32(token)

    c_raw = getattr(agent, "chemistry", None)
    state.chemistry[slot, :] = chemistry_from_values(c_raw) if c_raw is not None else zeros_chemistry()
    m_raw = getattr(agent, "maintenance", None)
    state.maintenance[slot, :] = maintenance_from_values(m_raw) if m_raw is not None else zeros_maintenance_state()

    state.aggr[slot] = np.float32(_as_float(getattr(genome, "aggression_bias", 0.0), 0.0))
    state.temp[slot] = np.float32(_as_float(getattr(genome, "action_temperature", 1.0), 1.0))
    state.gain[slot] = np.float32(_as_float(getattr(genome, "drive_logit_gain", 1.0), 1.0))
    state.drive_inertia[slot] = np.float32(min(0.999, max(0.0, _as_float(getattr(genome, "drive_inertia", 0.85), 0.85))))
    state.drive_noise[slot] = np.float32(min(0.10, max(0.0, _as_float(getattr(genome, "drive_noise", 0.02), 0.02))))
    state.pred_lr[slot] = np.float32(min(1.0, max(0.0, _as_float(getattr(genome, "pred_lr", 0.08), 0.08))))
    state.repro_threshold[slot] = np.float32(_as_float(getattr(genome, "reproduction_threshold", 30.0), 30.0))
    state.use_gru[slot] = bool(getattr(genome, "use_gru", False))
    state.mem_dim[slot] = np.int32(effective_memory_dim_for_genome(genome, sim_cfg=getattr(runner, "sim_cfg", None)))
    _fill_policy_observation_arrays(state, runner, genome, slot)
    _fill_cost_charge_arrays(state, runner, agent, genome, slot)
    try:
        parts = memory_cost_components(genome, env_cfg=getattr(runner, "env_cfg", None), sim_cfg=getattr(runner, "sim_cfg", None))
        state.gru_upkeep_cost[slot] = np.float32(max(0.0, _as_float(parts.get("charged", 0.0), 0.0)))
    except Exception:
        state.gru_upkeep_cost[slot] = np.float32(0.0)

    state.rest_streak[slot] = np.int32(_as_int(getattr(agent, "rest_streak", 0), 0))
    state.repro_cooldown[slot] = np.int32(_as_int(getattr(agent, "repro_cooldown", 0), 0))
    state.offspring_count[slot] = np.int32(0)
    state.survival_steps[slot] = np.int32(0)
    state.energy_collected[slot] = np.float32(0.0)
    state.early_survival_steps[slot] = np.int32(0)
    state.early_energy_collected[slot] = np.float32(0.0)
    state.early_injury_gained[slot] = np.float32(0.0)
    state.last_energy_delta[slot] = np.float32(0.0)
    state.last_health_delta[slot] = np.float32(0.0)
    state.last_injury_delta[slot] = np.float32(0.0)
    state.last_action[slot] = np.int32(-1)
    state.last_last_action[slot] = np.int32(-1)
    state.action_switches[slot] = np.int32(0)
    state.action_oscillations[slot] = np.int32(0)
    state.episode0_switches[slot] = np.int32(0)
    state.episode0_oscillations[slot] = np.int32(0)
    state.attacks[slot] = np.int32(0)
    state.idle_charge_paid[slot] = np.float32(0.0)
    state.maintenance_requests[slot] = np.int32(0)
    state.maintenance_applied[slot] = np.int32(0)
    state.maintenance_energy_spent_total[slot] = np.float32(0.0)
    state.maintenance_opportunity_cost_total[slot] = np.float32(0.0)
    state.maintenance_applied_effect_total[slot] = np.float32(0.0)
    state.chemistry_observation_fidelity[slot] = np.float32(1.0)
    state.chemistry_observation_bias[slot] = np.float32(0.0)
    state.selfharm_events[slot] = np.int32(0)
    state.episode0_selfharm_events[slot] = np.int32(0)
    state.drive_volatility[slot] = np.float32(0.0)
    state.drive_jitter[slot] = np.float32(0.0)
    state.episode0_drive_jitter_sum[slot] = np.float32(0.0)
    state.prev_drive_delta[slot, :] = np.float32(0.0)
    state.pred_err_sum[slot] = np.float32(0.0)
    state.pred_err_count[slot] = np.int32(0)
    state.prediction_update_count[slot] = np.int32(0)
    state.prediction_state_delta_abs_sum[slot] = np.float32(0.0)
    state.prediction_state_delta_abs_max[slot] = np.float32(0.0)
    state.prediction_reset_count[slot] = np.int32(0)
    state.episode0_pred_err_sum[slot] = np.float32(0.0)
    state.episode0_pred_err_count[slot] = np.int32(0)
    state.late_pred_err_sum[slot] = np.float32(0.0)
    state.late_pred_err_count[slot] = np.int32(0)
    state.adult_pred_err_ema[slot] = np.float32(0.0)

    st = _stat(agent)
    state.exp_dE[slot, :] = np.float32(0.0)
    state.exp_dInj[slot, :] = np.float32(0.0)
    if st is not None:
        de = np.asarray(getattr(st, "exp_dE", np.zeros((N_ACTIONS,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        di = np.asarray(getattr(st, "exp_dInj", np.zeros((N_ACTIONS,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        state.exp_dE[slot, : min(N_ACTIONS, int(de.shape[0]))] = de[:N_ACTIONS]
        state.exp_dInj[slot, : min(N_ACTIONS, int(di.shape[0]))] = di[:N_ACTIONS]
    state.pred_queue[slot] = deque()
    state.first_repro_step[slot] = None
    state.biome_id[slot] = np.int32(0)
    state.early_life_weight[slot] = np.float32(1.0)

    if invalidate_cache_on_token_change and int(old_token) != int(token):
        invalidate_action_cache_slot(state, slot, reason="resident_newborn_attach", agent_id=aid)
        state.counters.slot_token_attach_events += 1
    return int(slot)


def import_generation_state(runner: Any, *, generation: int, profile: str) -> RunnerCore1TickState:
    agents = _alive_agents(runner)
    agents.sort(key=lambda agent: _as_int(getattr(agent, "agent_id", -1), -1))
    capacity = _capacity(runner, agents)
    stride = max(1, _as_int(getattr(getattr(runner, "sim_cfg", None), "policy_tick_stride", 1), 1))

    state = RunnerCore1TickState(
        schema_version=SCHEMA_VERSION,
        profile=str(profile),
        generation=int(generation),
        capacity=int(capacity),
        stride=int(stride),
        slot_to_agent_id=np.full((capacity,), -1, dtype=np.int64),
        agent_id_to_slot_agent_ids=np.empty((0,), dtype=np.int64),
        agent_id_to_slot_slots=np.empty((0,), dtype=np.int32),
        slot_tokens=np.full((capacity,), -1, dtype=np.int64),
        alive=np.zeros((capacity,), dtype=np.bool_),
        energy=np.zeros((capacity,), dtype=np.float32),
        health=np.zeros((capacity,), dtype=np.float32),
        injury=np.zeros((capacity,), dtype=np.float32),
        scar=np.zeros((capacity,), dtype=np.float32),
        drives=np.zeros((capacity, N_DRIVES), dtype=np.float32),
        chemistry=np.zeros((capacity, CHEMISTRY_SIZE), dtype=np.float32),
        maintenance=np.zeros((capacity, MAINTENANCE_SIZE), dtype=np.float32),
        aggr=np.zeros((capacity,), dtype=np.float32),
        temp=np.ones((capacity,), dtype=np.float32),
        gain=np.ones((capacity,), dtype=np.float32),
        drive_inertia=np.full((capacity,), 0.85, dtype=np.float32),
        drive_noise=np.full((capacity,), 0.02, dtype=np.float32),
        pred_lr=np.full((capacity,), 0.08, dtype=np.float32),
        repro_threshold=np.full((capacity,), 30.0, dtype=np.float32),
        gru_upkeep_cost=np.zeros((capacity,), dtype=np.float32),
        sensor_channel_mask=np.zeros((capacity,), dtype=np.int32),
        sensor_radius=np.ones((capacity,), dtype=np.int32),
        chemistry_observation_fidelity=np.ones((capacity,), dtype=np.float32),
        chemistry_observation_bias=np.zeros((capacity,), dtype=np.float32),
        cost_charge_scalar=np.zeros((capacity,), dtype=np.float32),
        cost_charge_maturation_scalar=np.zeros((capacity,), dtype=np.float32),
        use_gru=np.zeros((capacity,), dtype=np.bool_),
        mem_dim=np.zeros((capacity,), dtype=np.int32),
        cache_agent_id=np.full((capacity,), -1, dtype=np.int64),
        cache_token=np.full((capacity,), -1, dtype=np.int64),
        cache_cursor=np.full((capacity,), -1, dtype=np.int32),
        cache_actions=np.full((capacity, max(1, stride)), -1, dtype=np.int32),
        survival_steps=np.zeros((capacity,), dtype=np.int32),
        energy_collected=np.zeros((capacity,), dtype=np.float32),
        early_survival_steps=np.zeros((capacity,), dtype=np.int32),
        early_energy_collected=np.zeros((capacity,), dtype=np.float32),
        early_injury_gained=np.zeros((capacity,), dtype=np.float32),
        last_energy_delta=np.zeros((capacity,), dtype=np.float32),
        last_health_delta=np.zeros((capacity,), dtype=np.float32),
        last_injury_delta=np.zeros((capacity,), dtype=np.float32),
        last_action=np.full((capacity,), -1, dtype=np.int32),
        last_last_action=np.full((capacity,), -1, dtype=np.int32),
        action_switches=np.zeros((capacity,), dtype=np.int32),
        action_oscillations=np.zeros((capacity,), dtype=np.int32),
        episode0_switches=np.zeros((capacity,), dtype=np.int32),
        episode0_oscillations=np.zeros((capacity,), dtype=np.int32),
        attacks=np.zeros((capacity,), dtype=np.int32),
        idle_charge_paid=np.zeros((capacity,), dtype=np.float32),
        maintenance_requests=np.zeros((capacity,), dtype=np.int32),
        maintenance_applied=np.zeros((capacity,), dtype=np.int32),
        maintenance_energy_spent_total=np.zeros((capacity,), dtype=np.float32),
        maintenance_opportunity_cost_total=np.zeros((capacity,), dtype=np.float32),
        maintenance_applied_effect_total=np.zeros((capacity,), dtype=np.float32),
        selfharm_events=np.zeros((capacity,), dtype=np.int32),
        episode0_selfharm_events=np.zeros((capacity,), dtype=np.int32),
        drive_volatility=np.zeros((capacity,), dtype=np.float32),
        drive_jitter=np.zeros((capacity,), dtype=np.float32),
        episode0_drive_jitter_sum=np.zeros((capacity,), dtype=np.float32),
        prev_drive_delta=np.zeros((capacity, N_DRIVES), dtype=np.float32),
        pred_err_sum=np.zeros((capacity,), dtype=np.float32),
        pred_err_count=np.zeros((capacity,), dtype=np.int32),
        prediction_update_count=np.zeros((capacity,), dtype=np.int32),
        prediction_state_delta_abs_sum=np.zeros((capacity,), dtype=np.float32),
        prediction_state_delta_abs_max=np.zeros((capacity,), dtype=np.float32),
        prediction_reset_count=np.zeros((capacity,), dtype=np.int32),
        episode0_pred_err_sum=np.zeros((capacity,), dtype=np.float32),
        episode0_pred_err_count=np.zeros((capacity,), dtype=np.int32),
        late_pred_err_sum=np.zeros((capacity,), dtype=np.float32),
        late_pred_err_count=np.zeros((capacity,), dtype=np.int32),
        adult_pred_err_ema=np.zeros((capacity,), dtype=np.float32),
        exp_dE=np.zeros((capacity, N_ACTIONS), dtype=np.float32),
        exp_dInj=np.zeros((capacity, N_ACTIONS), dtype=np.float32),
        pred_queue=[deque() for _ in range(capacity)],
        first_repro_step=[None for _ in range(capacity)],
        rest_streak=np.zeros((capacity,), dtype=np.int32),
        repro_cooldown=np.zeros((capacity,), dtype=np.int32),
        offspring_count=np.zeros((capacity,), dtype=np.int32),
        biome_id=np.zeros((capacity,), dtype=np.int32),
        early_life_weight=np.ones((capacity,), dtype=np.float32),
    )

    if hasattr(runner, "_ensure_drive_arrays"):
        runner._ensure_drive_arrays(required_slots=int(capacity))

    cache = getattr(runner, "_policy_action_cache", getattr(runner, "policy_action_cache", None))
    mlp_agents: list[Any] = []
    gru_agents: list[Any] = []
    for agent in agents:
        aid = _as_int(getattr(agent, "agent_id", -1), -1)
        slot = _as_int(getattr(agent, "slot", -1), -1)
        if aid < 0 or slot < 0 or slot >= capacity:
            continue
        if hasattr(runner, "_sync_drive_slot"):
            runner._sync_drive_slot(agent)
        genome = getattr(agent, "genome", None)
        state.slot_to_agent_id[slot] = np.int64(aid)
        bind_agent_id_to_slot(state, aid, slot)
        state.slot_tokens[slot] = np.int64(_as_int(getattr(agent, "slot_token", 0), 0))
        if isinstance(cache, dict):
            _import_action_cache_row(state, cache, slot=slot, agent_id=aid)
        state.alive[slot] = True
        state.energy[slot] = np.float32(_as_float(getattr(agent, "energy", 0.0), 0.0))
        state.health[slot] = np.float32(_as_float(getattr(agent, "health", 0.0), 0.0))
        state.injury[slot] = np.float32(_as_float(getattr(agent, "injury", 0.0), 0.0))
        state.scar[slot] = np.float32(_as_float(getattr(agent, "scar", 0.0), 0.0))
        d = np.asarray(getattr(agent, "drives", np.zeros((N_DRIVES,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        state.drives[slot, : min(N_DRIVES, int(d.shape[0]))] = d[:N_DRIVES]
        if hasattr(runner, "_drive_hunger") and getattr(runner, "_drive_hunger", None) is not None:
            state.drives[slot, 0] = np.float32(runner._drive_hunger[slot])
            state.drives[slot, 1] = np.float32(runner._drive_pain[slot])
            state.drives[slot, 2] = np.float32(runner._drive_fatigue[slot])
            state.drives[slot, 3] = np.float32(runner._drive_fear[slot])
        c_raw = getattr(agent, "chemistry", None)
        state.chemistry[slot, :] = chemistry_from_values(c_raw) if c_raw is not None else zeros_chemistry()
        m_raw = getattr(agent, "maintenance", None)
        state.maintenance[slot, :] = maintenance_from_values(m_raw) if m_raw is not None else zeros_maintenance_state()
        if genome is not None:
            state.aggr[slot] = np.float32(_as_float(getattr(genome, "aggression_bias", 0.0), 0.0))
            state.temp[slot] = np.float32(_as_float(getattr(genome, "action_temperature", 1.0), 1.0))
            state.gain[slot] = np.float32(_as_float(getattr(genome, "drive_logit_gain", 1.0), 1.0))
            state.drive_inertia[slot] = np.float32(min(0.999, max(0.0, _as_float(getattr(genome, "drive_inertia", 0.85), 0.85))))
            state.drive_noise[slot] = np.float32(min(0.10, max(0.0, _as_float(getattr(genome, "drive_noise", 0.02), 0.02))))
            state.pred_lr[slot] = np.float32(min(1.0, max(0.0, _as_float(getattr(genome, "pred_lr", 0.08), 0.08))))
            state.repro_threshold[slot] = np.float32(_as_float(getattr(genome, "reproduction_threshold", 30.0), 30.0))
            state.use_gru[slot] = bool(getattr(genome, "use_gru", False))
            state.mem_dim[slot] = np.int32(effective_memory_dim_for_genome(genome, sim_cfg=getattr(runner, "sim_cfg", None)))
            _fill_policy_observation_arrays(state, runner, genome, slot)
            _fill_cost_charge_arrays(state, runner, agent, genome, slot)
            try:
                parts = memory_cost_components(genome, env_cfg=getattr(runner, "env_cfg", None), sim_cfg=getattr(runner, "sim_cfg", None))
                state.gru_upkeep_cost[slot] = np.float32(max(0.0, _as_float(parts.get("charged", 0.0), 0.0)))
            except Exception:
                state.gru_upkeep_cost[slot] = np.float32(0.0)
        state.rest_streak[slot] = np.int32(_as_int(getattr(agent, "rest_streak", 0), 0))
        state.repro_cooldown[slot] = np.int32(_as_int(getattr(agent, "repro_cooldown", 0), 0))
        state.offspring_count[slot] = np.int32(_as_int(getattr(_stat(agent), "offspring_count", 0), 0))
        _fill_stat_arrays(state, agent, slot)
        if bool(state.use_gru[slot]):
            gru_agents.append(agent)
        else:
            mlp_agents.append(agent)

    if mlp_agents and hasattr(runner, "_sync_bank_for_agents"):
        runner._sync_bank_for_agents(mlp_agents, buckets=None)
        state.counters.boundary_policy_bank_sync_rows += int(len(mlp_agents))
    if gru_agents and hasattr(runner, "_sync_gru_bank_for_agents"):
        runner._sync_gru_bank_for_agents(gru_agents, buckets=None)
        state.counters.boundary_policy_bank_sync_rows += int(len(gru_agents))
    state.counters.boundary_agent_import_rows = int(np.count_nonzero(state.alive))
    return state


def _import_action_cache_row(
    state: RunnerCore1TickState,
    cache: dict[Any, Any],
    *,
    slot: int,
    agent_id: int,
) -> None:
    value = resolve_policy_action_cache_entry(
        cache,
        slot=int(slot),
        agent_id=int(agent_id),
    )
    if value is None:
        return
    token = -1
    cursor = -1
    raw_actions: Any = ()
    if isinstance(value, dict):
        token = _as_int(value.get("slot_token", value.get("token", -1)), -1)
        cursor = _as_int(value.get("cursor", value.get("idx", -1)), -1)
        raw_actions = value.get("actions", value.get("action_cache_actions", ()))
    elif isinstance(value, (tuple, list)):
        slot_keyed = is_slot_keyed_policy_cache_entry(value)
        token_index = 1 if slot_keyed else 0
        actions_index = 2 if slot_keyed else 1
        cursor_index = 3 if slot_keyed else 2
        token = _as_int(value[token_index], -1) if len(value) > token_index else -1
        if len(value) > actions_index:
            raw_actions = value[actions_index]
        cursor = _as_int(value[cursor_index], -1) if len(value) > cursor_index else -1
    else:
        return
    if token < 0 or cursor < 0:
        return
    try:
        actions = np.asarray(raw_actions, dtype=np.int32).reshape(-1)
    except Exception:
        return
    if int(actions.shape[0]) <= 0:
        return
    slot_i = int(slot)
    width = min(int(state.cache_actions.shape[1]), int(actions.shape[0]))
    if width <= 0:
        return
    state.cache_agent_id[slot_i] = np.int64(int(agent_id))
    state.cache_token[slot_i] = np.int64(int(token))
    state.cache_cursor[slot_i] = np.int32(int(cursor))
    state.cache_actions[slot_i, :] = np.int32(-1)
    state.cache_actions[slot_i, :width] = actions[:width]


def alive_agent_ids(state: RunnerCore1TickState, runner: Any) -> np.ndarray:
    env_alive = np.asarray(getattr(getattr(runner, "env", None), "agent_alive"), dtype=np.bool_)
    alive_slots = np.nonzero(state.alive)[0]
    ids: list[int] = []
    for slot in alive_slots.tolist():
        aid = int(state.slot_to_agent_id[int(slot)])
        if 0 <= aid < int(env_alive.shape[0]) and bool(env_alive[aid]):
            ids.append(aid)
        else:
            state.alive[int(slot)] = False
    ids.sort()
    return np.asarray(ids, dtype=np.int32)


def _runnercore9_lazy_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return bool(
        _bool(getattr(cfg, "runnercore9_candidate_enable", False))
        and _bool(getattr(cfg, "runnercore9_lazy_newborn_materialization_enable", False))
    )


def _runnercore10_lazy_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return bool(
        _bool(getattr(cfg, "runnercore10_candidate_enable", False))
        and _bool(getattr(cfg, "runnercore10_typed_child_contract_enable", False))
        and _bool(getattr(cfg, "runnercore10_defer_direct_attached_compat_enable", False))
    )


def _runnercore11_lazy_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return bool(
        _bool(getattr(cfg, "runnercore11_candidate_enable", False))
        and _bool(getattr(cfg, "runnercore11_c1_typed_rows_enable", False))
    )


def _runnercore_lazy_enabled(runner: Any) -> bool:
    return bool(_runnercore9_lazy_enabled(runner) or _runnercore10_lazy_enabled(runner) or _runnercore11_lazy_enabled(runner))


def _runnercore9_stats(runner: Any) -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    if cfg is None:
        return None
    stats = getattr(cfg, "_runnercore9_c1_payload_or_lazy_newborn_stats", None)
    if not isinstance(stats, dict):
        stats = {
            "schema_version": "runnercore9_c1_payload_or_lazy_newborn_v1",
            "c1_payload_contract": "trace_only_c1_scalar_skip_parity_blocked",
            "lazy_newborn_contract": "resident_rows_defer_compat_agent_until_export",
            "native_cpp_status": "not_attached_python_c1_and_materialization_boundary",
            "newborn_agent_materializations_deferred": 0,
            "newborn_agent_materializations_on_demand": 0,
            "generation_end_materializations": 0,
            "checkpoint_forced_materializations": 0,
            "unsupported_reader_materializations": 0,
            "resident_rows_without_agent_objects": 0,
            "event_boundary_export_rows_avoided": 0,
            "hidden_agent_sync_rows": 0,
            "unsupported_fallback_count": 0,
        }
        try:
            setattr(cfg, "_runnercore9_c1_payload_or_lazy_newborn_stats", stats)
        except Exception:
            return None
    return stats


def _runnercore9_stat_add(runner: Any, key: str, value: int | float = 1) -> None:
    stats = _runnercore9_stats(runner)
    if stats is None:
        return
    old = stats.get(str(key), 0)
    if isinstance(value, float):
        stats[str(key)] = float(old or 0.0) + float(value)
    else:
        stats[str(key)] = int(old or 0) + int(value)


def _runnercore10_stats(runner: Any) -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    if cfg is None:
        return None
    stats = getattr(cfg, "_runnercore10_typed_child_stats", None)
    if not isinstance(stats, dict):
        stats = {
            "schema_version": "runnercore10_typed_child_genome_policy_row_v1",
            "typed_child_contract": "resident_child_genome_policy_row_v1",
            "native_cpp_status": "not_attached_python_typed_child_contract",
            "typed_child_specs_created": 0,
            "typed_child_specs_attached": 0,
            "typed_child_policy_rows_attached": 0,
            "typed_child_hmem_rows_attached": 0,
            "typed_child_direct_mlp_rows": 0,
            "typed_child_direct_gru_rows": 0,
            "typed_child_unsupported_rows": 0,
            "genome_from_dict_avoided": 0,
            "policy_object_creation_avoided": 0,
            "agent_materialization_avoided": 0,
            "compatibility_materializations_forced": 0,
            "newborn_agent_materializations_deferred": 0,
            "newborn_agent_materializations_on_demand": 0,
            "generation_end_materializations": 0,
            "checkpoint_forced_materializations": 0,
            "unsupported_reader_materializations": 0,
            "resident_rows_without_agent_objects": 0,
            "event_boundary_export_rows_avoided": 0,
            "hidden_agent_sync_rows": 0,
            "per_tick_agent_sync_rows": 0,
            "unsupported_typed_child_fallback_count": 0,
            "unsupported_fallback_count": 0,
            "forced_materialization_reasons": {},
        }
        try:
            setattr(cfg, "_runnercore10_typed_child_stats", stats)
        except Exception:
            return None
    return stats


def _runnercore10_stat_add(runner: Any, key: str, value: int | float = 1) -> None:
    stats = _runnercore10_stats(runner)
    if stats is None:
        return
    old = stats.get(str(key), 0)
    if isinstance(value, float):
        stats[str(key)] = float(old or 0.0) + float(value)
    else:
        stats[str(key)] = int(old or 0) + int(value)


def _runnercore10_reason_add(runner: Any, reason: str, value: int = 1) -> None:
    stats = _runnercore10_stats(runner)
    if stats is None:
        return
    reasons = stats.get("forced_materialization_reasons", {})
    if not isinstance(reasons, dict):
        reasons = {}
    key = str(reason)
    reasons[key] = int(reasons.get(key, 0) or 0) + int(value)
    stats["forced_materialization_reasons"] = reasons


def _runnercore11_stats(runner: Any) -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    if cfg is None:
        return None
    stats = getattr(cfg, "_runnercore11_c1_typed_genome_row_stats", None)
    if not isinstance(stats, dict):
        stats = {
            "schema_version": "runnercore11_c1_typed_genome_row_v1",
            "typed_c1_contract": "canonical_c1_payload_projected_to_resident_genome_row",
            "native_cpp_status": "not_attached_python_projection",
            "c1_operator_pipeline_calls": 0,
            "c1_typed_rows_created": 0,
            "c1_typed_rows_attached": 0,
            "policy_rows_attached": 0,
            "hmem_rows_attached": 0,
            "c1_typed_rows_missing_field_fail_closed": 0,
            "c1_typed_rows_unsupported_kind": 0,
            "c1_payload_dict_roundtrips_avoided": 0,
            "genome_from_dict_avoided": 0,
            "genome_from_dict_forced": 0,
            "genome_materialization_avoided": 0,
            "genome_materialization_forced": 0,
            "generation_end_proxy_agents_created": 0,
            "generation_end_proxy_genome_materialization_avoided": 0,
            "policy_object_creation_avoided": 0,
            "policy_object_creation_forced": 0,
            "forced_materialization_policy_source": 0,
            "forced_materialization_policy_bank_sync": 0,
            "forced_materialization_behavior_log": 0,
            "compatibility_export_forced": 0,
            "compatibility_export_avoided": 0,
            "birth_success_count": 0,
            "birth_failure_count": 0,
            "hidden_agent_sync_rows": 0,
            "per_tick_agent_sync_rows": 0,
            "unsupported_fallback_count": 0,
            "runnercore12_c1_payload_digest_reuse_enabled": 0,
            "runnercore12_c1_payload_digest_reuse_rows": 0,
            "runnercore12_c1_payload_digest_reuse_fallbacks": 0,
            "runnercore13_c1_projection_digest_sample_only_enabled": 0,
            "runnercore13_c1_projection_digest_full_rows": 0,
            "runnercore13_c1_projection_digest_skipped_rows": 0,
            "runnercore15_c1_noop_normalize_skip_enabled": 0,
            "runnercore15_c1_noop_normalize_skipped": 0,
            "runnercore16_c1_empty_structure_fastpath_enabled": 0,
            "runnercore16_c1_empty_structure_fastpath_hits": 0,
            "runnercore16_c1_empty_structure_fastpath_misses": 0,
            "runnercore16_c1_empty_structure_fastpath_noop_draws": 0,
            "runnercore17_resident_birth_parent_candidates": 0,
            "runnercore17_resident_birth_parent_specs_used": 0,
            "runnercore17_resident_birth_parent_state_refreshes": 0,
            "runnercore17_resident_birth_parent_materializations_avoided": 0,
            "runnercore17_resident_birth_parent_parent_import_syncs": 0,
            "runnercore17_resident_birth_parent_missing_specs": 0,
            "runnercore17_resident_birth_parent_unsupported_fallback_count": 0,
            "runnercore18_resident_parent_minimal_refresh_enabled": 0,
            "runnercore18_resident_parent_minimal_refreshes": 0,
            "runnercore18_resident_parent_full_refreshes": 0,
            "runnercore18_resident_parent_full_refresh_skipped_no_slot": 0,
            "runnercore18_resident_parent_minimal_import_syncs": 0,
            "runnercore18_resident_parent_unsupported_fallback_count": 0,
            "forced_materialization_reasons": {},
            "c1_projection_trace_rows": 0,
            "c1_projection_trace_sample": [],
            "required_field_failures": {},
        }
        try:
            setattr(cfg, "_runnercore11_c1_typed_genome_row_stats", stats)
        except Exception:
            return None
    return stats


def _runnercore11_stat_add(runner: Any, key: str, value: int | float = 1) -> None:
    stats = _runnercore11_stats(runner)
    if stats is None:
        return
    old = stats.get(str(key), 0)
    if isinstance(value, float):
        stats[str(key)] = float(old or 0.0) + float(value)
    else:
        stats[str(key)] = int(old or 0) + int(value)


def _runnercore11_reason_add(runner: Any, reason: str, value: int = 1) -> None:
    stats = _runnercore11_stats(runner)
    if stats is None:
        return
    reasons = stats.get("forced_materialization_reasons", {})
    if not isinstance(reasons, dict):
        reasons = {}
    key = str(reason)
    reasons[key] = int(reasons.get(key, 0) or 0) + int(value)
    stats["forced_materialization_reasons"] = reasons


def _runnercore9_materialize_lazy_agent(runner: Any, aid: int, *, boundary_reason: str) -> Any | None:
    pop = getattr(runner, "pop", None)
    agents = getattr(pop, "agents", {}) if pop is not None else {}
    if isinstance(agents, dict) and int(aid) in agents:
        return agents.get(int(aid))
    if _runnercore10_lazy_enabled(runner) or _runnercore11_lazy_enabled(runner):
        r10_specs = getattr(pop, "_runnercore10_typed_child_specs", {}) if pop is not None else {}
        if isinstance(r10_specs, dict):
            spec = r10_specs.pop(int(aid), None)
            if spec is not None and pop is not None and hasattr(pop, "materialize_child_row_spec"):
                r11_projection = bool(_runnercore11_lazy_enabled(runner) and getattr(spec, "c1_typed_genome_row", None) is not None)
                proxy_generation_end = bool(
                    r11_projection
                    and str(boundary_reason) == "generation_end"
                    and bool(
                        getattr(
                            getattr(runner, "sim_cfg", None),
                            "runnercore11_generation_end_proxy_agents_enable",
                            False,
                        )
                    )
                )
                if r11_projection:
                    _runnercore11_reason_add(runner, str(boundary_reason), 1)
                child = pop.materialize_child_row_spec(spec, materialize_genome=not proxy_generation_end)
                try:
                    setattr(
                        child,
                        "typed_materialization_state",
                        "materialized:generation_end_proxy_agent"
                        if proxy_generation_end
                        else f"materialized:{boundary_reason}",
                    )
                except Exception:
                    pass
                _runnercore10_stat_add(runner, "compatibility_materializations_forced", 1)
                _runnercore10_reason_add(runner, str(boundary_reason), 1)
                if str(boundary_reason) == "generation_end":
                    _runnercore10_stat_add(runner, "generation_end_materializations", 1)
                    if r11_projection:
                        if proxy_generation_end:
                            _runnercore11_reason_add(runner, "generation_end_proxy_agent", 1)
                        else:
                            _runnercore11_stat_add(runner, "forced_materialization_generation_end", 1)
                elif str(boundary_reason).startswith("runnercore2_"):
                    _runnercore10_stat_add(runner, "newborn_agent_materializations_on_demand", 1)
                    if r11_projection:
                        _runnercore11_stat_add(runner, "forced_materialization_runnercore2_birth_event_boundary", 1)
                    ids = set(int(x) for x in list(getattr(runner, "_runnercore9_on_demand_materialized_ids", []) or []))
                    ids.add(int(aid))
                    try:
                        runner._runnercore9_on_demand_materialized_ids = sorted(ids)
                    except Exception:
                        pass
                elif "checkpoint" in str(boundary_reason):
                    _runnercore10_stat_add(runner, "checkpoint_forced_materializations", 1)
                    if r11_projection:
                        _runnercore11_stat_add(runner, "forced_materialization_checkpoint", 1)
                else:
                    _runnercore10_stat_add(runner, "unsupported_reader_materializations", 1)
                    if r11_projection:
                        _runnercore11_stat_add(runner, "forced_materialization_debug_oracle", 1)
                return child
    specs = getattr(pop, "_runnercore9_lazy_child_specs", {}) if pop is not None else {}
    if not isinstance(specs, dict):
        return None
    spec = specs.pop(int(aid), None)
    if spec is None or pop is None or not hasattr(pop, "materialize_child_row_spec"):
        return None
    child = pop.materialize_child_row_spec(spec)
    if str(boundary_reason) == "generation_end":
        _runnercore9_stat_add(runner, "generation_end_materializations", 1)
    elif str(boundary_reason).startswith("runnercore2_"):
        _runnercore9_stat_add(runner, "newborn_agent_materializations_on_demand", 1)
        ids = set(int(x) for x in list(getattr(runner, "_runnercore9_on_demand_materialized_ids", []) or []))
        ids.add(int(aid))
        try:
            runner._runnercore9_on_demand_materialized_ids = sorted(ids)
        except Exception:
            pass
    else:
        _runnercore9_stat_add(runner, "unsupported_reader_materializations", 1)
    return child


def export_to_agents(
    state: RunnerCore1TickState,
    runner: Any,
    *,
    boundary_reason: str,
    agent_ids: Any | None = None,
) -> dict[str, Any]:
    pop = getattr(runner, "pop", None)
    agents = getattr(pop, "agents", {}) if pop is not None else {}
    agent_filter = None if agent_ids is None else {int(aid) for aid in agent_ids}
    exported = 0
    for slot in range(int(state.capacity)):
        aid = int(state.slot_to_agent_id[slot])
        if aid < 0:
            continue
        if agent_filter is not None and aid not in agent_filter:
            continue
        agent = agents.get(aid) if isinstance(agents, dict) else None
        if agent is None and _runnercore_lazy_enabled(runner) and bool(state.alive[slot]):
            agent = _runnercore9_materialize_lazy_agent(runner, aid, boundary_reason=str(boundary_reason))
        if agent is None:
            if _runnercore9_lazy_enabled(runner) and bool(state.alive[slot]):
                _runnercore9_stat_add(runner, "event_boundary_export_rows_avoided", 1)
            if _runnercore10_lazy_enabled(runner) and bool(state.alive[slot]):
                _runnercore10_stat_add(runner, "event_boundary_export_rows_avoided", 1)
            if _runnercore11_lazy_enabled(runner) and bool(state.alive[slot]):
                _runnercore11_stat_add(runner, "compatibility_export_avoided", 1)
            continue
        if (
            str(boundary_reason) == "generation_end"
            and str(getattr(agent, "typed_materialization_state", "")) == "resident_row_spec"
            and pop is not None
            and hasattr(pop, "materialize_child_row_spec")
        ):
            agent = pop.materialize_child_row_spec(agent)
            stats = getattr(getattr(runner, "sim_cfg", None), "_runnercore6_resident_child_row_stats", None)
            if isinstance(stats, dict):
                stats["runnercore25_generation_end_materializations"] = int(
                    stats.get("runnercore25_generation_end_materializations", 0) or 0
                ) + 1
            runnercore3_profile_count(state, "runnercore25_generation_end_materializations", 1)
        agent.alive = bool(state.alive[slot])
        agent.energy = float(state.energy[slot])
        agent.health = float(state.health[slot])
        agent.injury = float(state.injury[slot])
        agent.scar = float(state.scar[slot])
        agent.drives[:] = state.drives[slot]
        agent.chemistry = chemistry_from_values(state.chemistry[slot])
        agent.maintenance = maintenance_from_values(state.maintenance[slot])
        agent.rest_streak = int(state.rest_streak[slot])
        agent.repro_cooldown = int(state.repro_cooldown[slot])
        st = _stat(agent)
        if st is not None:
            st.survival_steps = int(state.survival_steps[slot])
            st.energy_collected = float(state.energy_collected[slot])
            st.early_survival_steps = int(state.early_survival_steps[slot])
            st.early_energy_collected = float(state.early_energy_collected[slot])
            st.early_injury_gained = float(state.early_injury_gained[slot])
            st.last_energy_delta = float(state.last_energy_delta[slot])
            st.last_health_delta = float(state.last_health_delta[slot])
            st.last_injury_delta = float(state.last_injury_delta[slot])
            st.last_action = int(state.last_action[slot])
            st.last_last_action = int(state.last_last_action[slot])
            st.action_switches = int(state.action_switches[slot])
            st.action_oscillations = int(state.action_oscillations[slot])
            st.episode0_switches = int(state.episode0_switches[slot])
            st.episode0_oscillations = int(state.episode0_oscillations[slot])
            st.attacks = int(state.attacks[slot])
            st.idle_charge_paid = float(state.idle_charge_paid[slot])
            st.maintenance_requests = int(state.maintenance_requests[slot])
            st.maintenance_applied = int(state.maintenance_applied[slot])
            st.maintenance_energy_spent_total = float(state.maintenance_energy_spent_total[slot])
            st.maintenance_opportunity_cost_total = float(state.maintenance_opportunity_cost_total[slot])
            st.maintenance_applied_effect_total = float(state.maintenance_applied_effect_total[slot])
            st.chemistry_observation_fidelity = float(state.chemistry_observation_fidelity[slot])
            st.chemistry_observation_bias = float(state.chemistry_observation_bias[slot])
            st.selfharm_events = int(state.selfharm_events[slot])
            st.episode0_selfharm_events = int(state.episode0_selfharm_events[slot])
            st.drive_volatility = float(state.drive_volatility[slot])
            st.drive_jitter = float(state.drive_jitter[slot])
            st.episode0_drive_jitter_sum = float(state.episode0_drive_jitter_sum[slot])
            st.prev_drive_delta = np.asarray(state.prev_drive_delta[slot], dtype=np.float32).copy()
            st.pred_err_sum = float(state.pred_err_sum[slot])
            st.pred_err_count = int(state.pred_err_count[slot])
            st.prediction_update_count = int(state.prediction_update_count[slot])
            st.prediction_state_delta_abs_sum = float(state.prediction_state_delta_abs_sum[slot])
            st.prediction_state_delta_abs_max = float(state.prediction_state_delta_abs_max[slot])
            st.prediction_reset_count = int(state.prediction_reset_count[slot])
            st.episode0_pred_err_sum = float(state.episode0_pred_err_sum[slot])
            st.episode0_pred_err_count = int(state.episode0_pred_err_count[slot])
            st.late_pred_err_sum = float(state.late_pred_err_sum[slot])
            st.late_pred_err_count = int(state.late_pred_err_count[slot])
            st.adult_pred_err_ema = float(state.adult_pred_err_ema[slot])
            st.exp_dE = np.asarray(state.exp_dE[slot], dtype=np.float32).copy()
            st.exp_dInj = np.asarray(state.exp_dInj[slot], dtype=np.float32).copy()
            st.pred_queue = deque(state.pred_queue[slot])
            st.first_repro_step = state.first_repro_step[slot]
            st.offspring_count = int(state.offspring_count[slot])
            st.biome_id = int(state.biome_id[slot])
            st.early_life_weight = float(state.early_life_weight[slot])
        exported += 1
    if hasattr(runner, "_drive_hunger") and getattr(runner, "_drive_hunger", None) is not None:
        n = min(int(state.capacity), int(runner._drive_hunger.shape[0]))
        runner._drive_hunger[:n] = state.drives[:n, 0]
        runner._drive_pain[:n] = state.drives[:n, 1]
        runner._drive_fatigue[:n] = state.drives[:n, 2]
        runner._drive_fear[:n] = state.drives[:n, 3]
    _export_action_cache_to_runner(state, runner)
    state.counters.boundary_agent_export_rows += int(exported)
    if str(boundary_reason).startswith("runnercore2_"):
        state.counters.event_boundary_agent_export_rows += int(exported)
        state.counters.boundary_churn_materialization_rows += int(exported)
    return {"boundary_reason": str(boundary_reason), "exported_rows": int(exported)}


def _export_action_cache_to_runner(state: RunnerCore1TickState, runner: Any) -> None:
    restored_cache: dict[int, list[Any]] = {}
    rows = min(
        int(state.capacity),
        int(state.slot_to_agent_id.shape[0]),
        int(state.alive.shape[0]),
        int(state.cache_token.shape[0]),
        int(state.cache_cursor.shape[0]),
        int(state.cache_actions.shape[0]),
    )
    for slot in range(rows):
        aid = int(state.slot_to_agent_id[slot])
        if aid < 0 or not bool(state.alive[slot]):
            continue
        token = int(state.cache_token[slot])
        cursor = int(state.cache_cursor[slot])
        if token < 0 or cursor < 0:
            continue
        actions = np.asarray(state.cache_actions[slot], dtype=np.int32).reshape(-1).copy()
        if int(actions.shape[0]) <= 0:
            continue
        restored_cache[aid] = [token, actions, cursor]
    runner._policy_action_cache = restored_cache


def tick_state_schema_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "identity_slot_state": [
            "alive",
            "slot_to_agent_id",
            "agent_id_to_slot_agent_ids",
            "agent_id_to_slot_slots",
            "slot_tokens",
        ],
        "policy_action_state": [
            "policy scalar arrays",
            "resident policy-observation mask/radius arrays",
            "GRU/memory metadata",
            "slot-indexed stride action cache arrays",
            "runner bank rows synced at import boundary",
        ],
        "env_scalar_state": ["energy", "health", "injury", "scar", "drives", "chemistry", "maintenance"],
        "stats_metric_state": [
            "common AgentStats counters",
            "prediction expectation arrays",
            "drive volatility/jitter arrays",
        ],
        "export_boundaries": ["generation_end", "fail_closed_restore", "optional_churn_boundary"],
        "status": "authoritative_for_supported_no_churn_tick_candidate",
    }


def state_summary_payload(state: RunnerCore1TickState) -> dict[str, Any]:
    return {
        "schema_version": "runnercore1_state_summary_v1",
        "profile": state.profile,
        "generation": int(state.generation),
        "capacity": int(state.capacity),
        "alive_count": int(np.count_nonzero(state.alive)),
        "ticks_ran": int(state.counters.ticks_ran),
        "boundary_agent_import_rows": int(state.counters.boundary_agent_import_rows),
        "boundary_agent_export_rows": int(state.counters.boundary_agent_export_rows),
        "event_boundary_agent_export_rows": int(state.counters.event_boundary_agent_export_rows),
        "event_boundary_agent_create_rows": int(state.counters.event_boundary_agent_create_rows),
        "boundary_policy_bank_sync_rows": int(state.counters.boundary_policy_bank_sync_rows),
        "event_boundary_policy_bank_sync_rows": int(state.counters.event_boundary_policy_bank_sync_rows),
        "boundary_churn_materialization_rows": int(state.counters.boundary_churn_materialization_rows),
        "per_tick_agent_sync_rows": int(state.counters.per_tick_agent_sync_rows),
        "legacy_batched_actions_calls": int(state.counters.legacy_batched_actions_calls),
        "legacy_batched_actions_k_calls": int(state.counters.legacy_batched_actions_k_calls),
        "legacy_batched_actions_strided_calls": int(state.counters.legacy_batched_actions_strided_calls),
        "legacy_action_cache_refill_calls": int(state.counters.legacy_action_cache_refill_calls),
        "action_cache_hits": int(state.counters.action_cache_hits),
        "action_cache_misses": int(state.counters.action_cache_misses),
        "action_cache_refill_rows": int(state.counters.action_cache_refill_rows),
        "action_cache_invalidate_events": int(state.counters.action_cache_invalidate_events),
        "env_deaths": int(state.counters.env_deaths),
        "death_events_applied": int(state.counters.death_events_applied),
        "birth_events_applied": int(state.counters.birth_events_applied),
        "refill_events_applied": int(state.counters.refill_events_applied),
        "policy_row_attach_events": int(state.counters.policy_row_attach_events),
        "hmem_row_init_events": int(state.counters.hmem_row_init_events),
        "slot_token_attach_events": int(state.counters.slot_token_attach_events),
        "churn_events_total": int(state.counters.churn_events_total),
        "churn_event_sample": state.churn_events[:16],
    }
