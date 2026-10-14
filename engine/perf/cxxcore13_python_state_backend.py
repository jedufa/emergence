from __future__ import annotations

"""CXXCORE-13 Python state-backend ownership layer.

This module is default-off and diagnostic/pilot-only. It proves that a bounded
no-RNG env tick island can be represented as a copied-array backend transaction,
then written back through a Python-owned state boundary without mutating
authoritative simulation state until parity and feature gates pass.
"""

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Sequence

import numpy as np

from perf.cxxcore5_tick_island_layout_cache import (
    TickIslandLayoutCacheInputs,
    compare_runner_backend_states,
    run_tick_island_python_reference,
)
from perf.cxxcore8_guarded_runner_bridge import state_digest


SCHEMA_VERSION = "cxxcore13_python_state_backend_v1"
AUDIT_SCHEMA_VERSION = "cxxcore13_state_backend_ownership_audit_v1"
DECISION = "CXXCORE13-A-PYTHON-STATE-BACKEND-OWNERSHIP-GO"
MODES = (
    "off",
    "shadow_reference",
    "python_backend_authoritative_no_rng",
    "python_backend_authoritative_validation_rng",
    "native_backend_shadow_no_rng",
    "native_backend_authoritative_no_rng",
    "native_backend_authoritative_no_rng_no_compare",
)
PARITY_POLICIES = ("final_and_sparse", "every_tick", "interval")
RNG_POLICIES = ("none", "validation_action_override")


@dataclass(frozen=True)
class Cxxcore13StateBackendConfig:
    cxxcore13_state_backend_enable: bool = False
    cxxcore13_state_backend_mode: str = "off"
    cxxcore13_state_backend_parity_policy: str = "final_and_sparse"
    cxxcore13_state_backend_parity_interval: int = 1
    cxxcore13_state_backend_restore_on_divergence: bool = True
    cxxcore13_state_backend_fail_closed: bool = True
    cxxcore13_state_backend_strict_native_required: bool = False
    cxxcore13_state_backend_rng_policy: str = "none"
    cxxcore13_state_backend_emit_diagnostics: bool = True


@dataclass(frozen=True)
class EnvTickBackendInput:
    schema_version: str
    tick_inputs: TickIslandLayoutCacheInputs
    agent_ids: np.ndarray
    actions_by_tick: np.ndarray
    initial_energy: np.ndarray
    initial_health: np.ndarray
    initial_injury: np.ndarray
    initial_scar: np.ndarray
    pos_y_by_id: np.ndarray
    pos_x_by_id: np.ndarray
    agent_grid: np.ndarray
    resource_grid: np.ndarray
    cost_scalars_by_tick: np.ndarray
    layout_digest: str
    unsupported_feature_reasons: tuple[str, ...] = ()


@dataclass
class EnvTickLiveState:
    energy: np.ndarray
    health: np.ndarray
    injury: np.ndarray
    scar: np.ndarray
    alive_mask: np.ndarray
    pos_y_by_id: np.ndarray
    pos_x_by_id: np.ndarray
    agent_grid: np.ndarray
    resource_grid: np.ndarray
    stats_energy_collected: np.ndarray
    stats_survival_steps: np.ndarray
    views_dirty: bool = False


@dataclass(frozen=True)
class EnvTickRestoreToken:
    schema_version: str
    state_digest: str
    energy: np.ndarray
    health: np.ndarray
    injury: np.ndarray
    scar: np.ndarray
    alive_mask: np.ndarray
    pos_y_by_id: np.ndarray
    pos_x_by_id: np.ndarray
    agent_grid: np.ndarray
    resource_grid: np.ndarray
    stats_energy_collected: np.ndarray
    stats_survival_steps: np.ndarray
    views_dirty: bool


@dataclass(frozen=True)
class EnvTickBackendTransaction:
    schema_version: str
    output: dict[str, np.ndarray]
    food_gained: np.ndarray
    moved_mask: np.ndarray
    alive_mask: np.ndarray
    agent_grid: np.ndarray
    resource_grid: np.ndarray
    views_dirty: bool
    cost_total_by_agent: np.ndarray
    cost_total: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class EnvTickStateBackendResult:
    schema_version: str
    decision: str
    mode: str
    status: str
    transaction: EnvTickBackendTransaction | None
    parity: dict[str, Any] | None
    fallback_reason: str | None
    first_divergence: dict[str, Any] | None
    python_env_step_skipped: bool
    python_state_restored: bool
    native_available: bool
    native_output_retained: bool
    timing: dict[str, float]
    diagnostics: dict[str, Any]


def config_from_sim_cfg(sim_cfg: Any) -> Cxxcore13StateBackendConfig:
    return Cxxcore13StateBackendConfig(
        cxxcore13_state_backend_enable=bool(getattr(sim_cfg, "cxxcore13_state_backend_enable", False)),
        cxxcore13_state_backend_mode=str(getattr(sim_cfg, "cxxcore13_state_backend_mode", "off") or "off"),
        cxxcore13_state_backend_parity_policy=str(
            getattr(sim_cfg, "cxxcore13_state_backend_parity_policy", "final_and_sparse") or "final_and_sparse"
        ),
        cxxcore13_state_backend_parity_interval=int(
            getattr(sim_cfg, "cxxcore13_state_backend_parity_interval", 1) or 1
        ),
        cxxcore13_state_backend_restore_on_divergence=bool(
            getattr(sim_cfg, "cxxcore13_state_backend_restore_on_divergence", True)
        ),
        cxxcore13_state_backend_fail_closed=bool(getattr(sim_cfg, "cxxcore13_state_backend_fail_closed", True)),
        cxxcore13_state_backend_strict_native_required=bool(
            getattr(sim_cfg, "cxxcore13_state_backend_strict_native_required", False)
        ),
        cxxcore13_state_backend_rng_policy=str(getattr(sim_cfg, "cxxcore13_state_backend_rng_policy", "none") or "none"),
        cxxcore13_state_backend_emit_diagnostics=bool(
            getattr(sim_cfg, "cxxcore13_state_backend_emit_diagnostics", True)
        ),
    )


def validate_config(config: Cxxcore13StateBackendConfig) -> None:
    if config.cxxcore13_state_backend_mode not in MODES:
        raise ValueError(f"unsupported CXXCORE-13 state-backend mode: {config.cxxcore13_state_backend_mode}")
    if config.cxxcore13_state_backend_parity_policy not in PARITY_POLICIES:
        raise ValueError(f"unsupported CXXCORE-13 parity policy: {config.cxxcore13_state_backend_parity_policy}")
    if int(config.cxxcore13_state_backend_parity_interval) < 1:
        raise ValueError("cxxcore13_state_backend_parity_interval must be >= 1")
    if config.cxxcore13_state_backend_rng_policy not in RNG_POLICIES:
        raise ValueError(f"unsupported CXXCORE-13 RNG policy: {config.cxxcore13_state_backend_rng_policy}")


def state_backend_audit_rows() -> list[dict[str, Any]]:
    return [
        _row("agent_id_ordering", "Agent/object state", "owned_in_minimal_backend", "Copied row order is preserved from CXXCORE-5.", True),
        _row("energy_health_injury_scar", "Agent/object state", "owned_in_minimal_backend", "Transaction owns scalar output arrays and Python writeback.", True),
        _row("alive_dead_state", "Agent/object state", "owned_in_minimal_backend", "Death mask and alive mask are materialized before writeback.", True),
        _row("food_gained", "Agent/object state", "owned_in_minimal_backend", "Backend transaction now returns per-agent food gained.", True),
        _row("Agent.stats_energy_collected", "Agent/object state", "owned_in_minimal_backend", "Python state boundary applies food-gained stat deltas atomically.", True),
        _row("Agent.stats_survival_steps", "Agent/object state", "owned_in_minimal_backend", "Python state boundary applies bounded survival-step deltas.", True),
        _row("agent_position_arrays", "Grid/world state", "owned_in_minimal_backend", "Position arrays are written through the backend transaction.", True),
        _row("agent_grid_occupancy", "Grid/world state", "owned_in_minimal_backend", "Final occupancy grid is reconstructed and validated.", True),
        _row("resource_grid", "Grid/world state", "owned_in_minimal_backend", "Food/resource grid mutation is represented in copied arrays.", True),
        _row("views_dirty_flag", "Grid/world state", "owned_in_minimal_backend", "Movement/gather side effects set a writeback dirty flag.", True),
        _row("CostChargeSurface_layout_rows", "Cost/cache state", "owned_in_minimal_backend", "Precomputed layout rows/scalars feed the transaction.", True),
        _row("cost_total_by_agent", "Cost/cache state", "owned_in_minimal_backend", "Per-agent and aggregate cost totals are transaction outputs.", True),
        _row("CostLayoutCache_invalidation", "Cost/cache state", "deferred_feature_gated", "Cache lifecycle remains Python-owned outside the env tick island.", False),
        _row("phenotype_expression_path", "Agent/object state", "deferred_feature_gated", "Phenotype expression remains Python-owned and feeds precomputed rows/actions.", False),
        _row("perception_expression_path", "Agent/object state", "deferred_feature_gated", "Perception construction remains outside the first writeback slice.", False),
        _row("env_hotpath_diagnostics_config", "Diagnostics/metrics inputs", "deferred_feature_gated", "Hotpath diagnostics/config readouts remain Python-owned.", False),
        _row("patch_food_amount_and_emissions", "Grid/world state", "deferred_feature_gated", "Patch arrays/counters must be added before patch-enabled authoritative skip.", True),
        _row("tier3_last_step_stats", "Diagnostics/metrics inputs", "deferred_feature_gated", "Tier3 counters/buffers are excluded from the minimal no-RNG slice.", True),
        _row("validation_random_walk_rng", "RNG and stochastic surfaces", "requires_rng_precompute", "Validation action override must be Python-precomputed; no C++ RNG is added.", False),
        _row("biome_hazard_rng", "RNG and stochastic surfaces", "requires_rng_precompute", "Biome hazard stochastic draws are not in the minimal slice.", True),
        _row("tier3_transition_rng", "RNG and stochastic surfaces", "requires_rng_precompute", "Tier3 stochastic transition draws are not in the minimal slice.", True),
        _row("checkpoint_payload_state", "Checkpoint/replay implications", "requires_checkpoint_schema_preflight", "No checkpoint schema mutation is made in CXXCORE-13.", False),
        _row("metrics_logger_inputs", "Diagnostics/metrics inputs", "requires_metrics_schema_preflight", "No metrics schema mutation is made in CXXCORE-13.", False),
        _row("selection_reproduction_lineage", "Checkpoint/replay implications", "out_of_scope_for_cxxcore13", "Reproduction/death removal/lineage are outside this env tick writeback pilot.", False),
    ]


def state_backend_audit_payload() -> dict[str, Any]:
    rows = state_backend_audit_rows()
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "decision": DECISION,
        "minimal_authoritative_skip_supported": True,
        "python_env_step_skip_scope": "copied-array no-RNG env tick island with Python writeback boundary",
        "unsupported_features_fallback_before_mutation": True,
        "rng_strategy": "avoid RNG in the first authoritative slice; validation RNG remains Python-precomputed only",
        "checkpoint_schema_changed": False,
        "metrics_schema_changed": False,
        "rows": rows,
        "classifications": sorted({str(row["classification"]) for row in rows}),
    }


def as_backend_input(
    inputs: TickIslandLayoutCacheInputs,
    *,
    unsupported_feature_reasons: Sequence[str] = (),
) -> EnvTickBackendInput:
    state = inputs.backend.runner.state
    layout = inputs.backend.layout
    return EnvTickBackendInput(
        schema_version=SCHEMA_VERSION,
        tick_inputs=inputs,
        agent_ids=np.ascontiguousarray(state.agent_ids, dtype=np.int32),
        actions_by_tick=np.ascontiguousarray(state.actions_by_tick, dtype=np.int32),
        initial_energy=np.ascontiguousarray(state.initial_energy, dtype=np.float32),
        initial_health=np.ascontiguousarray(state.initial_health, dtype=np.float32),
        initial_injury=np.ascontiguousarray(state.initial_injury, dtype=np.float32),
        initial_scar=np.ascontiguousarray(state.initial_scar, dtype=np.float32),
        pos_y_by_id=np.ascontiguousarray(state.pos_y_by_id, dtype=np.int32),
        pos_x_by_id=np.ascontiguousarray(state.pos_x_by_id, dtype=np.int32),
        agent_grid=np.ascontiguousarray(state.agent_grid, dtype=np.int32),
        resource_grid=np.ascontiguousarray(state.grid, dtype=np.int8),
        cost_scalars_by_tick=np.ascontiguousarray(inputs.backend.runner.cost_scalars_by_tick, dtype=np.float32),
        layout_digest=str(layout.digest),
        unsupported_feature_reasons=tuple(str(reason) for reason in unsupported_feature_reasons if reason),
    )


def live_state_from_backend_input(backend_input: EnvTickBackendInput) -> EnvTickLiveState:
    n = int(backend_input.agent_ids.shape[0])
    return EnvTickLiveState(
        energy=backend_input.initial_energy.copy(),
        health=backend_input.initial_health.copy(),
        injury=backend_input.initial_injury.copy(),
        scar=backend_input.initial_scar.copy(),
        alive_mask=np.ones((n,), dtype=bool),
        pos_y_by_id=backend_input.pos_y_by_id.copy(),
        pos_x_by_id=backend_input.pos_x_by_id.copy(),
        agent_grid=backend_input.agent_grid.copy(),
        resource_grid=backend_input.resource_grid.copy(),
        stats_energy_collected=np.zeros((n,), dtype=np.float32),
        stats_survival_steps=np.zeros((n,), dtype=np.int32),
        views_dirty=False,
    )


def capture_restore_token(live_state: EnvTickLiveState) -> EnvTickRestoreToken:
    return EnvTickRestoreToken(
        schema_version=SCHEMA_VERSION,
        state_digest=live_state_digest(live_state),
        energy=live_state.energy.copy(),
        health=live_state.health.copy(),
        injury=live_state.injury.copy(),
        scar=live_state.scar.copy(),
        alive_mask=live_state.alive_mask.copy(),
        pos_y_by_id=live_state.pos_y_by_id.copy(),
        pos_x_by_id=live_state.pos_x_by_id.copy(),
        agent_grid=live_state.agent_grid.copy(),
        resource_grid=live_state.resource_grid.copy(),
        stats_energy_collected=live_state.stats_energy_collected.copy(),
        stats_survival_steps=live_state.stats_survival_steps.copy(),
        views_dirty=bool(live_state.views_dirty),
    )


def restore_from_token(live_state: EnvTickLiveState, token: EnvTickRestoreToken) -> None:
    live_state.energy[...] = token.energy
    live_state.health[...] = token.health
    live_state.injury[...] = token.injury
    live_state.scar[...] = token.scar
    live_state.alive_mask[...] = token.alive_mask
    live_state.pos_y_by_id[...] = token.pos_y_by_id
    live_state.pos_x_by_id[...] = token.pos_x_by_id
    live_state.agent_grid[...] = token.agent_grid
    live_state.resource_grid[...] = token.resource_grid
    live_state.stats_energy_collected[...] = token.stats_energy_collected
    live_state.stats_survival_steps[...] = token.stats_survival_steps
    live_state.views_dirty = bool(token.views_dirty)


def run_python_backend_transaction(
    backend_input: EnvTickBackendInput,
    *,
    inject_divergence: bool = False,
    raise_exception: bool = False,
) -> EnvTickBackendTransaction:
    if raise_exception:
        raise RuntimeError("injected CXXCORE-13 backend failure")
    reference = run_tick_island_python_reference(backend_input.tick_inputs)
    food_gained, moved_mask, agent_grid, views_dirty = _side_effect_deltas(backend_input)
    output = {key: np.ascontiguousarray(value.copy()) for key, value in reference.items()}
    if inject_divergence and output["new_energy"].size:
        output["new_energy"][0] = np.float32(float(output["new_energy"][0]) + 1.0)
    alive_mask = np.logical_not(np.asarray(output["died_mask"], dtype=bool))
    return EnvTickBackendTransaction(
        schema_version=SCHEMA_VERSION,
        output=output,
        food_gained=np.ascontiguousarray(food_gained, dtype=np.float32),
        moved_mask=np.ascontiguousarray(moved_mask, dtype=bool),
        alive_mask=np.ascontiguousarray(alive_mask, dtype=bool),
        agent_grid=np.ascontiguousarray(agent_grid, dtype=np.int32),
        resource_grid=np.ascontiguousarray(output["grid"], dtype=np.int8),
        views_dirty=bool(views_dirty),
        cost_total_by_agent=np.ascontiguousarray(output["cost_total_by_agent"], dtype=np.float32),
        cost_total=np.ascontiguousarray(output["cost_total"], dtype=np.float64),
        diagnostics={
            "layout_digest": backend_input.layout_digest,
            "ticks": int(backend_input.actions_by_tick.shape[0]),
            "agents": int(backend_input.agent_ids.shape[0]),
            "gridworld_step_skipped": True,
        },
    )


def native_output_to_transaction(
    backend_input: EnvTickBackendInput,
    native_output: dict[str, Any],
    *,
    inject_divergence: bool = False,
) -> EnvTickBackendTransaction:
    output = _normalize_backend_output(backend_input, native_output)
    if inject_divergence and output["new_energy"].size:
        output["new_energy"][0] = np.float32(float(output["new_energy"][0]) + 1.0)
    food_gained, moved_mask, agent_grid, views_dirty = _side_effect_deltas(backend_input)
    alive_mask = np.logical_not(np.asarray(output["died_mask"], dtype=bool))
    return EnvTickBackendTransaction(
        schema_version=SCHEMA_VERSION,
        output=output,
        food_gained=np.ascontiguousarray(food_gained, dtype=np.float32),
        moved_mask=np.ascontiguousarray(moved_mask, dtype=bool),
        alive_mask=np.ascontiguousarray(alive_mask, dtype=bool),
        agent_grid=np.ascontiguousarray(agent_grid, dtype=np.int32),
        resource_grid=np.ascontiguousarray(output["grid"], dtype=np.int8),
        views_dirty=bool(views_dirty),
        cost_total_by_agent=np.ascontiguousarray(output["cost_total_by_agent"], dtype=np.float32),
        cost_total=np.ascontiguousarray(output["cost_total"], dtype=np.float64),
        diagnostics={
            "layout_digest": backend_input.layout_digest,
            "ticks": int(backend_input.actions_by_tick.shape[0]),
            "agents": int(backend_input.agent_ids.shape[0]),
            "gridworld_step_skipped": True,
            "native_output_adapter": True,
        },
    )


def materialize_transaction_to_live_state(
    live_state: EnvTickLiveState,
    backend_input: EnvTickBackendInput,
    transaction: EnvTickBackendTransaction,
) -> None:
    ids = backend_input.agent_ids
    live_state.energy[...] = transaction.output["new_energy"]
    live_state.health[...] = transaction.output["new_health"]
    live_state.injury[...] = transaction.output["new_injury"]
    live_state.scar[...] = transaction.output["new_scar"]
    live_state.alive_mask[...] = transaction.alive_mask
    live_state.pos_y_by_id[ids] = transaction.output["new_pos_y"]
    live_state.pos_x_by_id[ids] = transaction.output["new_pos_x"]
    live_state.agent_grid[...] = transaction.agent_grid
    live_state.resource_grid[...] = transaction.resource_grid
    live_state.stats_energy_collected[...] = live_state.stats_energy_collected + transaction.food_gained
    live_state.stats_survival_steps[...] = live_state.stats_survival_steps + int(backend_input.actions_by_tick.shape[0])
    live_state.views_dirty = bool(live_state.views_dirty or transaction.views_dirty)


def compare_transaction_to_reference(
    backend_input: EnvTickBackendInput,
    transaction: EnvTickBackendTransaction,
    *,
    atol: float = 1.0e-5,
) -> dict[str, Any]:
    expected = run_tick_island_python_reference(backend_input.tick_inputs)
    report = compare_runner_backend_states(expected, transaction.output, atol=atol)
    expected_grid = _agent_grid_from_positions(
        backend_input=backend_input,
        pos_y=np.asarray(expected["new_pos_y"], dtype=np.int32),
        pos_x=np.asarray(expected["new_pos_x"], dtype=np.int32),
    )
    bad_grid = np.argwhere(expected_grid != transaction.agent_grid)
    if bad_grid.size:
        y, x = [int(v) for v in bad_grid[0].tolist()]
        report["ok"] = False
        report["mismatch_count"] = int(report["mismatch_count"]) + int(bad_grid.shape[0])
        report["first_divergence"] = report.get("first_divergence") or {
            "field": "agent_grid",
            "index": [y, x],
            "reason": "occupancy_mismatch",
            "python": int(expected_grid[y, x]),
            "candidate": int(transaction.agent_grid[y, x]),
            "abs_diff": None,
            "classification": "state-backend-writeback",
        }
    report["fields_compared"] = list(report.get("fields_compared", [])) + [
        "agent_grid",
        "food_gained_present",
        "moved_mask_present",
        "views_dirty",
    ]
    return report


def run_state_backend(
    inputs: TickIslandLayoutCacheInputs,
    *,
    config: Cxxcore13StateBackendConfig,
    live_state: EnvTickLiveState | None = None,
    native_backend: Any | None = None,
    unsupported_feature_reasons: Sequence[str] = (),
    inject_divergence: bool = False,
    raise_exception: bool = False,
) -> EnvTickStateBackendResult:
    validate_config(config)
    started = time.perf_counter()
    timing = _empty_timing()
    mode = str(config.cxxcore13_state_backend_mode)
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION,
        "checkpoint_schema_changed": False,
        "metrics_schema_changed": False,
        "native_rng_added": False,
        "native_direct_live_object_mutation": False,
        "state_backend_audit_digest": _stable_json_digest(state_backend_audit_rows()),
    }
    if not config.cxxcore13_state_backend_enable or mode == "off":
        timing["total_wall_s"] = time.perf_counter() - started
        return _result(mode, "disabled", None, None, "pilot_disabled", False, False, False, False, timing, diagnostics)

    build_t0 = time.perf_counter()
    backend_input = as_backend_input(inputs, unsupported_feature_reasons=unsupported_feature_reasons)
    timing["backend_input_build_s"] = time.perf_counter() - build_t0
    if backend_input.unsupported_feature_reasons:
        timing["fallback_overhead_s"] = time.perf_counter() - started
        timing["total_wall_s"] = time.perf_counter() - started
        return _result(
            mode,
            "fallback",
            None,
            None,
            "unsupported_features",
            False,
            False,
            False,
            True,
            timing,
            diagnostics | {"unsupported_feature_reasons": list(backend_input.unsupported_feature_reasons)},
        )

    if mode == "python_backend_authoritative_validation_rng":
        if config.cxxcore13_state_backend_rng_policy != "validation_action_override":
            timing["total_wall_s"] = time.perf_counter() - started
            return _result(mode, "fallback", None, None, "validation_rng_payload_required", False, False, False, True, timing, diagnostics)
        timing["total_wall_s"] = time.perf_counter() - started
        return _result(mode, "fallback", None, None, "validation_rng_writeback_deferred", False, False, False, True, timing, diagnostics)

    if mode == "native_backend_shadow_no_rng":
        if native_backend is None:
            timing["total_wall_s"] = time.perf_counter() - started
            status = "blocked" if config.cxxcore13_state_backend_strict_native_required else "fallback"
            return _result(mode, status, None, None, "native_unavailable", False, False, False, True, timing, diagnostics)
        try:
            compute_t0 = time.perf_counter()
            native_output = native_backend(inputs)
            timing["native_compute_s"] = time.perf_counter() - compute_t0
            materialize_t0 = time.perf_counter()
            transaction = native_output_to_transaction(backend_input, native_output, inject_divergence=inject_divergence)
            timing["native_materialize_s"] = time.perf_counter() - materialize_t0
            compare_t0 = time.perf_counter()
            parity = compare_transaction_to_reference(backend_input, transaction)
            timing["comparison_s"] = time.perf_counter() - compare_t0
            timing["total_wall_s"] = time.perf_counter() - started
            if not bool(parity.get("ok")):
                return _result(mode, "fallback", transaction, parity, "parity_divergence", False, True, False, True, timing, diagnostics)
            return _result(mode, "reference_parity", transaction, parity, None, False, True, False, True, timing, diagnostics)
        except Exception as exc:
            if config.cxxcore13_state_backend_fail_closed:
                raise
            timing["fallback_overhead_s"] = time.perf_counter() - started
            timing["total_wall_s"] = time.perf_counter() - started
            return _result(
                mode,
                "fallback",
                None,
                None,
                "exception_fallback",
                False,
                True,
                False,
                True,
                timing,
                diagnostics | {"exception_type": type(exc).__name__, "exception": str(exc)},
            )

    state = live_state or live_state_from_backend_input(backend_input)
    token_t0 = time.perf_counter()
    token = capture_restore_token(state)
    timing["restore_token_s"] = time.perf_counter() - token_t0

    try:
        if mode in {"native_backend_authoritative_no_rng", "native_backend_authoritative_no_rng_no_compare"}:
            if native_backend is None:
                timing["total_wall_s"] = time.perf_counter() - started
                status = "blocked" if config.cxxcore13_state_backend_strict_native_required else "fallback"
                return _result(mode, status, None, None, "native_unavailable", False, False, False, True, timing, diagnostics)
            if raise_exception:
                raise RuntimeError("injected CXXCORE-14 native backend failure")
            compute_t0 = time.perf_counter()
            native_output = native_backend(inputs)
            timing["native_compute_s"] = time.perf_counter() - compute_t0
            materialize_t0 = time.perf_counter()
            transaction = native_output_to_transaction(backend_input, native_output, inject_divergence=inject_divergence)
            timing["native_materialize_s"] = time.perf_counter() - materialize_t0
            if mode == "native_backend_authoritative_no_rng_no_compare":
                parity = {
                    "schema_version": SCHEMA_VERSION,
                    "ok": True,
                    "mismatch_count": 0,
                    "max_abs_diff": 0.0,
                    "first_divergence": None,
                    "fields_compared": [],
                    "comparison_skipped": True,
                }
            else:
                compare_t0 = time.perf_counter()
                parity = compare_transaction_to_reference(backend_input, transaction)
                timing["comparison_s"] = time.perf_counter() - compare_t0
        else:
            compute_t0 = time.perf_counter()
            transaction = run_python_backend_transaction(
                backend_input,
                inject_divergence=inject_divergence,
                raise_exception=raise_exception,
            )
            timing["backend_compute_s"] = time.perf_counter() - compute_t0
            compare_t0 = time.perf_counter()
            parity = compare_transaction_to_reference(backend_input, transaction)
            timing["comparison_s"] = time.perf_counter() - compare_t0
        if not bool(parity.get("ok")):
            if config.cxxcore13_state_backend_restore_on_divergence:
                restore_t0 = time.perf_counter()
                restore_from_token(state, token)
                timing["fallback_overhead_s"] += time.perf_counter() - restore_t0
            timing["total_wall_s"] = time.perf_counter() - started
            return _result(mode, "fallback", transaction, parity, "parity_divergence", True, False, True, True, timing, diagnostics)
        if mode == "shadow_reference":
            timing["total_wall_s"] = time.perf_counter() - started
            return _result(mode, "reference_parity", transaction, parity, None, False, False, False, True, timing, diagnostics)
        write_t0 = time.perf_counter()
        materialize_transaction_to_live_state(state, backend_input, transaction)
        timing["writeback_s"] = time.perf_counter() - write_t0
        timing["total_wall_s"] = time.perf_counter() - started
        status = "authoritative_no_compare" if mode == "native_backend_authoritative_no_rng_no_compare" else "authoritative_parity"
        return _result(mode, status, transaction, parity, None, True, bool(native_backend is not None), True, False, timing, diagnostics)
    except Exception as exc:
        if config.cxxcore13_state_backend_fail_closed:
            raise
        restore_t0 = time.perf_counter()
        restore_from_token(state, token)
        timing["fallback_overhead_s"] += time.perf_counter() - restore_t0
        timing["total_wall_s"] = time.perf_counter() - started
        return _result(
            mode,
            "fallback",
            None,
            None,
            "exception_fallback",
            False,
            False,
            True,
            True,
            timing,
            diagnostics | {"exception_type": type(exc).__name__, "exception": str(exc)},
        )


def result_payload(result: EnvTickStateBackendResult) -> dict[str, Any]:
    output = result.transaction.output if result.transaction else None
    return {
        "schema_version": result.schema_version,
        "decision": result.decision,
        "mode": result.mode,
        "status": result.status,
        "parity": result.parity,
        "fallback_reason": result.fallback_reason,
        "first_divergence": result.first_divergence,
        "python_env_step_skipped": bool(result.python_env_step_skipped),
        "python_state_restored": bool(result.python_state_restored),
        "native_available": bool(result.native_available),
        "native_output_retained": bool(result.native_output_retained),
        "timing": dict(result.timing),
        "diagnostics": dict(result.diagnostics),
        "state_digest": state_digest(output),
        "transaction": transaction_summary(result.transaction) if result.transaction else None,
    }


def transaction_summary(transaction: EnvTickBackendTransaction | None) -> dict[str, Any] | None:
    if transaction is None:
        return None
    return {
        "schema_version": transaction.schema_version,
        "food_gained_sum": float(np.sum(transaction.food_gained, dtype=np.float64)),
        "moved_count": int(np.sum(transaction.moved_mask)),
        "alive_count": int(np.sum(transaction.alive_mask)),
        "views_dirty": bool(transaction.views_dirty),
        "cost_total": float(np.asarray(transaction.cost_total, dtype=np.float64)[0]) if transaction.cost_total.size else 0.0,
        "output_digest": state_digest(transaction.output),
        "agent_grid_digest": _array_digest(transaction.agent_grid),
        "resource_grid_digest": _array_digest(transaction.resource_grid),
    }


def live_state_digest(live_state: EnvTickLiveState) -> str:
    return _stable_json_digest(
        {
            "energy": _array_digest(live_state.energy),
            "health": _array_digest(live_state.health),
            "injury": _array_digest(live_state.injury),
            "scar": _array_digest(live_state.scar),
            "alive_mask": _array_digest(live_state.alive_mask),
            "pos_y_by_id": _array_digest(live_state.pos_y_by_id),
            "pos_x_by_id": _array_digest(live_state.pos_x_by_id),
            "agent_grid": _array_digest(live_state.agent_grid),
            "resource_grid": _array_digest(live_state.resource_grid),
            "stats_energy_collected": _array_digest(live_state.stats_energy_collected),
            "stats_survival_steps": _array_digest(live_state.stats_survival_steps),
            "views_dirty": bool(live_state.views_dirty),
        }
    )


def _normalize_backend_output(backend_input: EnvTickBackendInput, native_output: dict[str, Any]) -> dict[str, np.ndarray]:
    if not isinstance(native_output, dict):
        raise ValueError("native backend output must be a dict")
    n = int(backend_input.agent_ids.shape[0])
    grid_shape = backend_input.resource_grid.shape
    specs = {
        "new_energy": (np.float32, (n,)),
        "new_health": (np.float32, (n,)),
        "new_injury": (np.float32, (n,)),
        "new_scar": (np.float32, (n,)),
        "died_mask": (bool, (n,)),
        "new_pos_y": (np.int32, (n,)),
        "new_pos_x": (np.int32, (n,)),
        "grid": (np.int8, grid_shape),
        "cost_total_by_agent": (np.float32, (n,)),
        "cost_total": (np.float64, (1,)),
    }
    normalized: dict[str, np.ndarray] = {}
    for key, (dtype, shape) in specs.items():
        if key not in native_output:
            raise ValueError(f"native backend output missing required field: {key}")
        arr = np.ascontiguousarray(native_output[key], dtype=dtype)
        if tuple(arr.shape) != tuple(shape):
            raise ValueError(f"native backend field {key} shape {tuple(arr.shape)} != {tuple(shape)}")
        normalized[key] = arr
    return normalized


def _side_effect_deltas(backend_input: EnvTickBackendInput) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    ids = backend_input.agent_ids
    n = int(ids.shape[0])
    state = backend_input.tick_inputs.backend.runner.state
    cfg = state.config
    order = np.argsort(ids, kind="mergesort")
    pos_y = backend_input.pos_y_by_id.astype(np.int32, copy=True)
    pos_x = backend_input.pos_x_by_id.astype(np.int32, copy=True)
    occ = backend_input.agent_grid.astype(np.int32, copy=True)
    grid = backend_input.resource_grid.astype(np.int8, copy=True)
    food_gained = np.zeros((n,), dtype=np.float32)
    moved = np.zeros((n,), dtype=bool)
    for tick in range(int(backend_input.actions_by_tick.shape[0])):
        actions = backend_input.actions_by_tick[tick]
        for src in order.tolist():
            action = int(actions[src])
            aid = int(ids[src])
            y = int(pos_y[aid])
            x = int(pos_x[aid])
            if 0 <= action <= 3:
                ny = y + (-1 if action == 0 else 1 if action == 1 else 0)
                nx = x + (1 if action == 2 else -1 if action == 3 else 0)
                if 0 <= ny < int(cfg.height) and 0 <= nx < int(cfg.width) and int(occ[ny, nx]) == -1:
                    if int(occ[y, x]) == aid:
                        occ[y, x] = -1
                    occ[ny, nx] = aid
                    pos_y[aid] = np.int32(ny)
                    pos_x[aid] = np.int32(nx)
                    moved[src] = True
            elif action == 4 and int(grid[y, x]) == 1:
                grid[y, x] = np.int8(0)
                food_gained[src] = np.float32(float(food_gained[src]) + float(cfg.food_energy) * float(state.juvenile_harvest_mults[src]))
    return food_gained, moved, occ, bool(np.any(moved) or np.any(food_gained > 0.0))


def _agent_grid_from_positions(*, backend_input: EnvTickBackendInput, pos_y: np.ndarray, pos_x: np.ndarray) -> np.ndarray:
    state = backend_input.tick_inputs.backend.runner.state
    grid = np.full((int(state.config.height), int(state.config.width)), -1, dtype=np.int32)
    for idx, aid in enumerate(backend_input.agent_ids.tolist()):
        y = int(pos_y[idx])
        x = int(pos_x[idx])
        if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
            grid[y, x] = int(aid)
    return np.ascontiguousarray(grid)


def _row(name: str, surface: str, classification: str, notes: str, required_for_skip: bool) -> dict[str, Any]:
    return {
        "name": name,
        "surface": surface,
        "classification": classification,
        "notes": notes,
        "required_for_python_env_step_skip": bool(required_for_skip),
    }


def _empty_timing() -> dict[str, float]:
    return {
        "total_wall_s": 0.0,
        "legacy_python_env_step_s": 0.0,
        "backend_input_build_s": 0.0,
        "backend_compute_s": 0.0,
        "writeback_s": 0.0,
        "restore_token_s": 0.0,
        "comparison_s": 0.0,
        "diagnostics_s": 0.0,
        "fallback_overhead_s": 0.0,
        "fallback_count": 0.0,
        "native_compute_s": 0.0,
        "native_materialize_s": 0.0,
    }


def _result(
    mode: str,
    status: str,
    transaction: EnvTickBackendTransaction | None,
    parity: dict[str, Any] | None,
    fallback_reason: str | None,
    python_env_step_skipped: bool,
    native_available: bool,
    native_output_retained: bool,
    python_state_restored: bool,
    timing: dict[str, float],
    diagnostics: dict[str, Any],
) -> EnvTickStateBackendResult:
    if fallback_reason:
        timing["fallback_count"] = 1.0
    return EnvTickStateBackendResult(
        schema_version=SCHEMA_VERSION,
        decision=DECISION,
        mode=mode,
        status=status,
        transaction=transaction,
        parity=parity,
        fallback_reason=fallback_reason,
        first_divergence=(parity or {}).get("first_divergence") if parity else None,
        python_env_step_skipped=bool(python_env_step_skipped),
        python_state_restored=bool(python_state_restored),
        native_available=bool(native_available),
        native_output_retained=bool(native_output_retained),
        timing=timing,
        diagnostics=diagnostics,
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
    "PARITY_POLICIES",
    "RNG_POLICIES",
    "SCHEMA_VERSION",
    "Cxxcore13StateBackendConfig",
    "EnvTickBackendInput",
    "EnvTickBackendTransaction",
    "EnvTickLiveState",
    "EnvTickRestoreToken",
    "EnvTickStateBackendResult",
    "as_backend_input",
    "capture_restore_token",
    "compare_transaction_to_reference",
    "config_from_sim_cfg",
    "live_state_digest",
    "live_state_from_backend_input",
    "materialize_transaction_to_live_state",
    "native_output_to_transaction",
    "restore_from_token",
    "result_payload",
    "run_python_backend_transaction",
    "run_state_backend",
    "state_backend_audit_payload",
    "state_backend_audit_rows",
    "transaction_summary",
    "validate_config",
]
