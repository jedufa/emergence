from __future__ import annotations

"""CXXCORE-16 guarded live runner resident-window pilot helpers.

Default-off diagnostic code. This module deliberately keeps integration narrow:
it can build a resident no-RNG window from live GridWorld/Agent state, run a
resident backend over prepared action/cost arrays, and materialize final state
at a sync point. It also reports when the real runner loop is too wide for a
safe resident window because per-tick diagnostics, drives, chemistry, metrics,
or checkpoint boundaries require Python state each tick.
"""

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Sequence

import numpy as np

from perf.cxxcore11_live_runner_transaction_hook import prepare_live_tick_inputs
from perf.cxxcore15_resident_soa_tick_island import (
    ResidentRunResult,
    ResidentSoAConfig,
    as_resident_state,
    compare_resident_to_reference,
    resident_result_payload,
    run_resident_python_reference,
)
from perf.cxxcore5_tick_island_layout_cache import CostLayoutCache, TickIslandLayoutCacheInputs, as_tick_island_inputs
from perf.cxxcore8_guarded_runner_bridge import state_digest
from perf.cxxcore_env_apply_shadow import config_from_env


SCHEMA_VERSION = "cxxcore16_live_runner_resident_window_v1"
AUDIT_SCHEMA_VERSION = "cxxcore16_live_runner_resident_window_audit_v1"
DECISION = "CXXCORE16-B-LIVE-RUNNER-BOUNDARY-TOO-WIDE"
MODES = (
    "off",
    "resident_window_shadow_compare",
    "resident_window_authoritative_no_rng_guarded",
    "resident_window_authoritative_no_rng_no_compare",
)


@dataclass(frozen=True)
class Cxxcore16ResidentWindowConfig:
    cxxcore16_resident_window_enable: bool = False
    cxxcore16_resident_window_authoritative_enable: bool = False
    cxxcore16_resident_window_mode: str = "off"
    cxxcore16_resident_window_compare_enable: bool = True
    cxxcore16_resident_window_sync_interval: int = 0
    cxxcore16_resident_window_max_ticks: int = 0
    cxxcore16_resident_window_allow_validation_rng_precompute: bool = False
    cxxcore16_resident_window_fail_closed: bool = True
    cxxcore16_resident_window_restore_on_divergence: bool = True


@dataclass(frozen=True)
class LiveResidentWindowRestoreToken:
    schema_version: str
    digest: str
    agent_ids: np.ndarray
    energy: np.ndarray
    health: np.ndarray
    injury: np.ndarray
    scar: np.ndarray
    alive: np.ndarray
    pos_y_by_id: np.ndarray
    pos_x_by_id: np.ndarray
    agent_grid: np.ndarray
    resource_grid: np.ndarray
    stats_survival_steps: np.ndarray
    stats_energy_collected: np.ndarray
    views_dirty: bool


@dataclass(frozen=True)
class Cxxcore16ResidentWindowResult:
    schema_version: str
    mode: str
    status: str
    decision: str
    output: dict[str, np.ndarray] | None
    parity: dict[str, Any] | None
    fallback_reason: str | None
    first_divergence: dict[str, Any] | None
    python_env_step_skipped: bool
    resident_windows: int
    fallback_count: int
    python_state_restored: bool
    timing: dict[str, float]
    diagnostics: dict[str, Any]


def config_from_sim_cfg(sim_cfg: Any) -> Cxxcore16ResidentWindowConfig:
    return Cxxcore16ResidentWindowConfig(
        cxxcore16_resident_window_enable=bool(getattr(sim_cfg, "cxxcore16_resident_window_enable", False)),
        cxxcore16_resident_window_authoritative_enable=bool(
            getattr(sim_cfg, "cxxcore16_resident_window_authoritative_enable", False)
        ),
        cxxcore16_resident_window_mode=str(getattr(sim_cfg, "cxxcore16_resident_window_mode", "off") or "off"),
        cxxcore16_resident_window_compare_enable=bool(
            getattr(sim_cfg, "cxxcore16_resident_window_compare_enable", True)
        ),
        cxxcore16_resident_window_sync_interval=int(
            getattr(sim_cfg, "cxxcore16_resident_window_sync_interval", 0) or 0
        ),
        cxxcore16_resident_window_max_ticks=int(getattr(sim_cfg, "cxxcore16_resident_window_max_ticks", 0) or 0),
        cxxcore16_resident_window_allow_validation_rng_precompute=bool(
            getattr(sim_cfg, "cxxcore16_resident_window_allow_validation_rng_precompute", False)
        ),
        cxxcore16_resident_window_fail_closed=bool(getattr(sim_cfg, "cxxcore16_resident_window_fail_closed", True)),
        cxxcore16_resident_window_restore_on_divergence=bool(
            getattr(sim_cfg, "cxxcore16_resident_window_restore_on_divergence", True)
        ),
    )


def live_runner_resident_window_audit() -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "live_runner_entry_points": [
            "runner_ticks_loop._run_ticks per-tick action/policy loop",
            "TickEnvApplyMixin._apply_env_tick",
            "GridWorld.step_agents_batch",
        ],
        "object_fields_mutated_per_tick": [
            "Agent.energy/health/injury/scar",
            "Agent.stats.last_energy_delta/last_health_delta/last_injury_delta",
            "Agent.stats.survival_steps downstream of env tick",
            "biome_id and Tier4/Tier3/Patch counters when enabled",
        ],
        "metrics_diagnostics_dependencies": [
            "memory_action_window_instrumentation pre/post env hooks",
            "prediction_demand_action_context_instrumentation pre/post env hooks",
            "identity drift update after env tick",
            "internal drives update after env tick",
            "chemistry state update after env tick",
            "cost charge diagnostics and expression recorders",
        ],
        "checkpoint_boundary_rule": "resident window must end/materialize before checkpoint; no checkpoint schema change",
        "supported_initial_slice": [
            "no Patch/Jungle dynamic arrays",
            "no biome hazard RNG",
            "no Tier3 stochastic transition/crossing RNG",
            "no reproduction/selection/cull/refill/lineage event inside window",
            "no checkpoint inside window",
            "no per-tick metrics requiring Python object state",
            "no independent native RNG",
        ],
        "eligibility_classes": eligibility_class_names(),
        "boundary_decision": DECISION,
        "reason_boundary_too_wide": (
            "the production _run_ticks loop requires policy/action generation, pre/post diagnostics, drives, "
            "chemistry, and metrics-visible Agent stats after every env tick; a multi-tick live resident window "
            "would either stale those reads or require owning broader runner lifecycle state"
        ),
        "checkpoint_schema_changed": False,
        "metrics_schema_changed": False,
        "native_rng_added": False,
    }


def eligibility_class_names() -> list[str]:
    return [
        "eligible_no_rng_resident_window",
        "unsupported_patch_jungle_dynamic",
        "unsupported_tier3_stochastic",
        "unsupported_biome_hazard_rng",
        "unsupported_checkpoint_boundary",
        "unsupported_metrics_dependency",
        "unsupported_object_stat_writeback",
        "unsupported_agent_lifecycle_event",
        "native_missing",
        "native_exception",
        "divergence_detected",
        "malformed_actions_or_costs",
    ]


def classify_eligibility(
    *,
    env: Any,
    env_cfg: Any,
    sim_cfg: Any,
    actions_by_tick: Any,
    checkpoint_boundary: bool = False,
    metrics_require_per_tick_python: bool = False,
    object_stats_require_per_tick_python: bool = False,
    agent_lifecycle_event: bool = False,
    native_backend: Any | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if bool(getattr(env, "patch_world_enable", False)) or bool(getattr(env_cfg, "patch_world_enable", False)):
        reasons.append("unsupported_patch_jungle_dynamic")
    if bool(getattr(env_cfg, "migration_pressure_enable", False)) or bool(getattr(env_cfg, "tier3_validation_force_random_walk", False)):
        reasons.append("unsupported_tier3_stochastic")
    if bool(getattr(env_cfg, "enable_biomes", False)) or float(getattr(env_cfg, "biome_hazard_prob", 0.0) or 0.0) > 0.0:
        reasons.append("unsupported_biome_hazard_rng")
    if checkpoint_boundary:
        reasons.append("unsupported_checkpoint_boundary")
    if metrics_require_per_tick_python:
        reasons.append("unsupported_metrics_dependency")
    if object_stats_require_per_tick_python:
        reasons.append("unsupported_object_stat_writeback")
    if agent_lifecycle_event:
        reasons.append("unsupported_agent_lifecycle_event")
    if native_backend is None:
        reasons.append("native_missing")
    actions = np.asarray(actions_by_tick)
    if actions.ndim != 2 or actions.size == 0:
        reasons.append("malformed_actions_or_costs")
    return {
        "eligible": not reasons,
        "classification": "eligible_no_rng_resident_window" if not reasons else reasons[0],
        "reasons": reasons,
    }


def prepare_live_window_inputs(
    *,
    env: Any,
    env_cfg: Any,
    sim_cfg: Any,
    alive: Sequence[Any],
    actions_by_tick: Any,
    energies: Any | None = None,
    healths: Any | None = None,
    injuries: Any | None = None,
    scars: Any | None = None,
    move_cost_mults: Any | None = None,
    juvenile_harvest_mults: Any | None = None,
    cache: CostLayoutCache | None = None,
) -> tuple[TickIslandLayoutCacheInputs, CostLayoutCache, dict[str, Any]]:
    t0 = time.perf_counter()
    agents = list(alive)
    ids = np.asarray([int(getattr(agent, "agent_id")) for agent in agents], dtype=np.int32)
    actions = np.ascontiguousarray(actions_by_tick, dtype=np.int32)
    if actions.ndim != 2 or int(actions.shape[1]) != int(ids.shape[0]):
        raise ValueError("actions_by_tick must be 2D and aligned to alive agents")
    cache_obj = cache if cache is not None else CostLayoutCache(env_cfg=env_cfg, sim_cfg=sim_cfg)
    snapshot, info = cache_obj.build_or_update(agents)
    n = int(ids.shape[0])
    inputs = as_tick_island_inputs(
        config=config_from_env(env),
        agent_ids=ids,
        initial_energy=np.asarray(
            energies if energies is not None else [float(getattr(agent, "energy")) for agent in agents],
            dtype=np.float32,
        ).copy(),
        initial_health=np.asarray(
            healths if healths is not None else [float(getattr(agent, "health")) for agent in agents],
            dtype=np.float32,
        ).copy(),
        initial_injury=np.asarray(
            injuries if injuries is not None else [float(getattr(agent, "injury", 0.0)) for agent in agents],
            dtype=np.float32,
        ).copy(),
        initial_scar=np.asarray(
            scars if scars is not None else [float(getattr(agent, "scar", 0.0)) for agent in agents],
            dtype=np.float32,
        ).copy(),
        pos_y_by_id=np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32).copy(),
        pos_x_by_id=np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32).copy(),
        agent_grid=np.asarray(getattr(env, "agent_grid"), dtype=np.int32).copy(),
        grid=np.asarray(getattr(env, "grid"), dtype=np.int8).copy(),
        actions_by_tick=actions,
        cache_snapshot=snapshot,
        move_cost_mults=np.ones((n,), dtype=np.float32) if move_cost_mults is None else np.asarray(move_cost_mults, dtype=np.float32).copy(),
        juvenile_harvest_mults=np.ones((n,), dtype=np.float32) if juvenile_harvest_mults is None else np.asarray(juvenile_harvest_mults, dtype=np.float32).copy(),
    )
    return inputs, cache_obj, {
        "build_s": float(time.perf_counter() - t0),
        "cache_mode": str(info.get("cache_mode", "")),
        "rows_rebuilt": int(snapshot.rows_rebuilt),
        "ticks": int(actions.shape[0]),
        "agents": n,
    }


def capture_restore_token(env: Any, alive: Sequence[Any]) -> LiveResidentWindowRestoreToken:
    agents = list(alive)
    ids = np.asarray([int(getattr(agent, "agent_id")) for agent in agents], dtype=np.int32)
    token = LiveResidentWindowRestoreToken(
        schema_version=SCHEMA_VERSION,
        digest="",
        agent_ids=ids,
        energy=np.asarray([float(getattr(agent, "energy")) for agent in agents], dtype=np.float32),
        health=np.asarray([float(getattr(agent, "health")) for agent in agents], dtype=np.float32),
        injury=np.asarray([float(getattr(agent, "injury", 0.0)) for agent in agents], dtype=np.float32),
        scar=np.asarray([float(getattr(agent, "scar", 0.0)) for agent in agents], dtype=np.float32),
        alive=np.asarray([bool(getattr(agent, "alive", True)) for agent in agents], dtype=bool),
        pos_y_by_id=np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32).copy(),
        pos_x_by_id=np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32).copy(),
        agent_grid=np.asarray(getattr(env, "agent_grid"), dtype=np.int32).copy(),
        resource_grid=np.asarray(getattr(env, "grid"), dtype=np.int8).copy(),
        stats_survival_steps=np.asarray(
            [int(getattr(getattr(agent, "stats", None), "survival_steps", 0) or 0) for agent in agents],
            dtype=np.int32,
        ),
        stats_energy_collected=np.asarray(
            [float(getattr(getattr(agent, "stats", None), "energy_collected", 0.0) or 0.0) for agent in agents],
            dtype=np.float32,
        ),
        views_dirty=bool(getattr(env, "_views_dirty", False)),
    )
    return LiveResidentWindowRestoreToken(**{**token.__dict__, "digest": _token_digest(token)})


def restore_from_token(env: Any, alive: Sequence[Any], token: LiveResidentWindowRestoreToken) -> None:
    agents = list(alive)
    for i, agent in enumerate(agents):
        agent.energy = float(token.energy[i])
        agent.health = float(token.health[i])
        agent.injury = float(token.injury[i])
        agent.scar = float(token.scar[i])
        agent.alive = bool(token.alive[i])
        stats = getattr(agent, "stats", None)
        if stats is not None:
            try:
                stats.survival_steps = int(token.stats_survival_steps[i])
                stats.energy_collected = float(token.stats_energy_collected[i])
            except Exception:
                pass
    getattr(env, "agent_pos_y")[...] = token.pos_y_by_id
    getattr(env, "agent_pos_x")[...] = token.pos_x_by_id
    getattr(env, "agent_grid")[...] = token.agent_grid
    getattr(env, "grid")[...] = token.resource_grid
    try:
        setattr(env, "_views_dirty", bool(token.views_dirty))
    except Exception:
        pass


def materialize_to_live(
    *,
    env: Any,
    alive: Sequence[Any],
    inputs: TickIslandLayoutCacheInputs,
    resident: ResidentRunResult,
) -> None:
    agents = list(alive)
    ids = np.asarray(inputs.backend.runner.state.agent_ids, dtype=np.int32)
    for i, agent in enumerate(agents):
        agent.energy = float(resident.output["new_energy"][i])
        agent.health = float(resident.output["new_health"][i])
        agent.injury = float(resident.output["new_injury"][i])
        agent.scar = float(resident.output["new_scar"][i])
        agent.alive = not bool(resident.output["died_mask"][i])
        st = getattr(agent, "stats", None)
        if st is not None:
            try:
                st.last_energy_delta = float(resident.output["new_energy"][i]) - float(inputs.backend.runner.state.initial_energy[i])
                st.last_health_delta = float(resident.output["new_health"][i]) - float(inputs.backend.runner.state.initial_health[i])
                st.last_injury_delta = float(resident.output["new_injury"][i]) - float(inputs.backend.runner.state.initial_injury[i])
                st.energy_collected = float(getattr(st, "energy_collected", 0.0) or 0.0) + float(resident.food_gained[i])
                st.survival_steps = int(getattr(st, "survival_steps", 0) or 0) + int(inputs.backend.runner.state.actions_by_tick.shape[0])
            except Exception:
                pass
    getattr(env, "agent_pos_y")[ids] = resident.output["new_pos_y"]
    getattr(env, "agent_pos_x")[ids] = resident.output["new_pos_x"]
    getattr(env, "agent_grid")[...] = resident.agent_grid
    getattr(env, "grid")[...] = resident.output["grid"]
    try:
        setattr(env, "_views_dirty", bool(resident.views_dirty))
    except Exception:
        pass


def run_live_resident_window(
    *,
    env: Any,
    env_cfg: Any,
    sim_cfg: Any,
    alive: Sequence[Any],
    actions_by_tick: Any,
    config: Cxxcore16ResidentWindowConfig,
    native_backend: Any | None = None,
    cache: CostLayoutCache | None = None,
    checkpoint_boundary: bool = False,
    metrics_require_per_tick_python: bool = False,
    object_stats_require_per_tick_python: bool = False,
    agent_lifecycle_event: bool = False,
    inject_divergence: bool = False,
    raise_exception: bool = False,
) -> Cxxcore16ResidentWindowResult:
    started = time.perf_counter()
    timing = _empty_timing()
    mode = str(config.cxxcore16_resident_window_mode)
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_schema_changed": False,
        "metrics_schema_changed": False,
        "native_rng_added": False,
        "native_direct_live_object_mutation": False,
    }
    if not bool(config.cxxcore16_resident_window_enable) or mode == "off":
        timing["total_wall_s"] = time.perf_counter() - started
        return _result(mode, "disabled", None, None, "resident_window_disabled", False, 0, 1, False, timing, diagnostics)
    eligibility = classify_eligibility(
        env=env,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        actions_by_tick=actions_by_tick,
        checkpoint_boundary=checkpoint_boundary,
        metrics_require_per_tick_python=metrics_require_per_tick_python,
        object_stats_require_per_tick_python=object_stats_require_per_tick_python,
        agent_lifecycle_event=agent_lifecycle_event,
        native_backend=native_backend,
    )
    diagnostics["eligibility"] = eligibility
    if not bool(eligibility["eligible"]):
        timing["total_wall_s"] = time.perf_counter() - started
        return _result(mode, "fallback", None, None, str(eligibility["classification"]), False, 0, 1, True, timing, diagnostics)

    token_t0 = time.perf_counter()
    token = capture_restore_token(env, alive)
    timing["restore_token_s"] = time.perf_counter() - token_t0
    try:
        if raise_exception:
            raise RuntimeError("injected CXXCORE-16 native exception")
        build_t0 = time.perf_counter()
        inputs, cache_obj, prep_diag = prepare_live_window_inputs(
            env=env,
            env_cfg=env_cfg,
            sim_cfg=sim_cfg,
            alive=alive,
            actions_by_tick=actions_by_tick,
            cache=cache,
        )
        timing["resident_backend_init_s"] = time.perf_counter() - build_t0
        diagnostics["prepare"] = prep_diag
        native_t0 = time.perf_counter()
        native_payload = native_backend(inputs)
        timing["resident_backend_compute_s"] = time.perf_counter() - native_t0
        resident = _coerce_resident_result(inputs, native_payload, inject_divergence=inject_divergence)
        parity = None
        if mode != "resident_window_authoritative_no_rng_no_compare" and bool(config.cxxcore16_resident_window_compare_enable):
            cmp_t0 = time.perf_counter()
            parity = compare_resident_to_reference(inputs, resident)
            timing["comparison_s"] = time.perf_counter() - cmp_t0
            if not bool(parity.get("ok", False)):
                if config.cxxcore16_resident_window_restore_on_divergence:
                    restore_t0 = time.perf_counter()
                    restore_from_token(env, alive, token)
                    timing["fallback_overhead_s"] += time.perf_counter() - restore_t0
                timing["total_wall_s"] = time.perf_counter() - started
                return _result(
                    mode,
                    "fallback",
                    resident.output,
                    parity,
                    "divergence_detected",
                    True,
                    1,
                    1,
                    True,
                    timing,
                    diagnostics,
                )
        elif mode == "resident_window_authoritative_no_rng_no_compare":
            parity = {"schema_version": SCHEMA_VERSION, "ok": True, "comparison_skipped": True, "first_divergence": None}
        if mode == "resident_window_shadow_compare":
            timing["total_wall_s"] = time.perf_counter() - started
            return _result(mode, "shadow_parity", resident.output, parity, None, False, 1, 0, True, timing, diagnostics)
        if not bool(config.cxxcore16_resident_window_authoritative_enable):
            timing["total_wall_s"] = time.perf_counter() - started
            return _result(
                mode,
                "fallback",
                resident.output,
                parity,
                "authoritative_disabled",
                False,
                0,
                1,
                True,
                timing,
                diagnostics,
            )
        mat_t0 = time.perf_counter()
        materialize_to_live(env=env, alive=alive, inputs=inputs, resident=resident)
        timing["final_materialization_s"] = time.perf_counter() - mat_t0
        timing["total_wall_s"] = time.perf_counter() - started
        return _result(mode, "authoritative_parity", resident.output, parity, None, True, 1, 0, False, timing, diagnostics)
    except Exception as exc:
        if config.cxxcore16_resident_window_fail_closed:
            raise
        restore_t0 = time.perf_counter()
        restore_from_token(env, alive, token)
        timing["fallback_overhead_s"] += time.perf_counter() - restore_t0
        timing["total_wall_s"] = time.perf_counter() - started
        return _result(
            mode,
            "fallback",
            None,
            None,
            "native_exception",
            False,
            0,
            1,
            True,
            timing,
            diagnostics | {"exception_type": type(exc).__name__, "exception": str(exc)},
        )


def result_payload(result: Cxxcore16ResidentWindowResult) -> dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "mode": result.mode,
        "status": result.status,
        "decision": result.decision,
        "parity": result.parity,
        "fallback_reason": result.fallback_reason,
        "first_divergence": result.first_divergence,
        "python_env_step_skipped": bool(result.python_env_step_skipped),
        "resident_windows": int(result.resident_windows),
        "fallback_count": int(result.fallback_count),
        "python_state_restored": bool(result.python_state_restored),
        "timing": dict(result.timing),
        "diagnostics": dict(result.diagnostics),
        "state_digest": state_digest(result.output),
    }


def _coerce_resident_result(
    inputs: TickIslandLayoutCacheInputs,
    payload: Any,
    *,
    inject_divergence: bool = False,
) -> ResidentRunResult:
    if isinstance(payload, ResidentRunResult):
        result = payload
    elif isinstance(payload, dict) and "output" in payload:
        result = ResidentRunResult(
            schema_version=payload.get("schema_version", "cxxcore16_native_payload_v1"),
            output={k: np.asarray(v).copy() for k, v in payload["output"].items()},
            agent_grid=np.asarray(payload["agent_grid"], dtype=np.int32).copy(),
            food_gained=np.asarray(payload["food_gained"], dtype=np.float32).copy(),
            moved_mask=np.asarray(payload["moved_mask"], dtype=bool).copy(),
            views_dirty=bool(payload["views_dirty"]),
            sparse_ticks=np.asarray(payload.get("sparse_ticks", []), dtype=np.int32).copy(),
            sparse_summaries=np.asarray(payload.get("sparse_summaries", np.empty((0, 7))), dtype=np.float64).copy(),
            final_summary=dict(payload.get("final_summary", {})),
            diagnostics=dict(payload.get("diagnostics", {})),
        )
    else:
        raise ValueError("native resident backend returned malformed payload")
    if inject_divergence and result.output["new_energy"].size:
        output = {k: np.asarray(v).copy() for k, v in result.output.items()}
        output["new_energy"][0] = np.float32(float(output["new_energy"][0]) + 1.0)
        result = ResidentRunResult(
            schema_version=result.schema_version,
            output=output,
            agent_grid=result.agent_grid,
            food_gained=result.food_gained,
            moved_mask=result.moved_mask,
            views_dirty=result.views_dirty,
            sparse_ticks=result.sparse_ticks,
            sparse_summaries=result.sparse_summaries,
            final_summary=result.final_summary,
            diagnostics=result.diagnostics,
        )
    return result


def _empty_timing() -> dict[str, float]:
    return {
        "total_wall_s": 0.0,
        "python_action_generation_s": 0.0,
        "python_cost_layout_generation_s": 0.0,
        "resident_backend_init_s": 0.0,
        "action_cost_ingest_s": 0.0,
        "resident_backend_compute_s": 0.0,
        "sparse_summary_s": 0.0,
        "final_materialization_s": 0.0,
        "comparison_s": 0.0,
        "fallback_overhead_s": 0.0,
        "metrics_checkpoint_materialization_s": 0.0,
        "restore_token_s": 0.0,
    }


def _result(
    mode: str,
    status: str,
    output: dict[str, np.ndarray] | None,
    parity: dict[str, Any] | None,
    fallback_reason: str | None,
    python_env_step_skipped: bool,
    resident_windows: int,
    fallback_count: int,
    python_state_restored: bool,
    timing: dict[str, float],
    diagnostics: dict[str, Any],
) -> Cxxcore16ResidentWindowResult:
    return Cxxcore16ResidentWindowResult(
        schema_version=SCHEMA_VERSION,
        mode=mode,
        status=status,
        decision=DECISION,
        output=output,
        parity=parity,
        fallback_reason=fallback_reason,
        first_divergence=(parity or {}).get("first_divergence") if parity else None,
        python_env_step_skipped=bool(python_env_step_skipped),
        resident_windows=int(resident_windows),
        fallback_count=int(fallback_count),
        python_state_restored=bool(python_state_restored),
        timing=timing,
        diagnostics=diagnostics,
    )


def _token_digest(token: LiveResidentWindowRestoreToken) -> str:
    return _stable_json_digest(
        {
            "agent_ids": _array_digest(token.agent_ids),
            "energy": _array_digest(token.energy),
            "health": _array_digest(token.health),
            "injury": _array_digest(token.injury),
            "scar": _array_digest(token.scar),
            "alive": _array_digest(token.alive),
            "pos_y_by_id": _array_digest(token.pos_y_by_id),
            "pos_x_by_id": _array_digest(token.pos_x_by_id),
            "agent_grid": _array_digest(token.agent_grid),
            "resource_grid": _array_digest(token.resource_grid),
            "stats_survival_steps": _array_digest(token.stats_survival_steps),
            "stats_energy_collected": _array_digest(token.stats_energy_collected),
            "views_dirty": bool(token.views_dirty),
        }
    )


def _array_digest(value: Any) -> str:
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _stable_json_digest(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "DECISION",
    "MODES",
    "SCHEMA_VERSION",
    "Cxxcore16ResidentWindowConfig",
    "Cxxcore16ResidentWindowResult",
    "LiveResidentWindowRestoreToken",
    "capture_restore_token",
    "classify_eligibility",
    "config_from_sim_cfg",
    "eligibility_class_names",
    "live_runner_resident_window_audit",
    "materialize_to_live",
    "prepare_live_window_inputs",
    "restore_from_token",
    "result_payload",
    "run_live_resident_window",
]
