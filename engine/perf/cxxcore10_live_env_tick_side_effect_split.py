from __future__ import annotations

"""CXXCORE-10 live env-tick side-effect split preflight.

This module defines a default-off transaction boundary for env-tick materialized
outputs. It separates pure compute results from Python-owned writeback planning
without mutating live Agent or GridWorld objects.
"""

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from perf.cxxcore5_tick_island_layout_cache import (
    TickIslandLayoutCacheInputs,
    compare_runner_backend_states,
    run_tick_island_python_reference,
)
from perf.cxxcore7_rng_precompute import Cxxcore7RngPayload, apply_validation_action_override, validate_rng_payload
from perf.cxxcore8_guarded_runner_bridge import state_digest


SCHEMA_VERSION = "cxxcore10_live_env_tick_side_effect_split_v1"
SPLIT_MODES = (
    "off",
    "python_split_reference",
    "native_shadow_transaction",
    "native_guarded_transaction_no_rng",
    "native_guarded_transaction_validation_rng",
)
PARITY_POLICIES = ("final_and_sparse", "every_tick", "interval")
MATERIALIZED_FIELDS = (
    "new_energy",
    "new_health",
    "new_injury",
    "new_scar",
    "died_mask",
    "new_pos_y",
    "new_pos_x",
    "grid",
    "cost_total_by_agent",
    "cost_total",
)


@dataclass(frozen=True)
class Cxxcore10SplitConfig:
    cxxcore10_env_tick_split_enable: bool = False
    cxxcore10_env_tick_split_mode: str = "off"
    cxxcore10_env_tick_split_parity_interval: int = 1
    cxxcore10_env_tick_split_restore_on_divergence: bool = True
    cxxcore10_env_tick_split_emit_diagnostics: bool = True
    cxxcore10_env_tick_split_fail_closed: bool = True
    cxxcore10_env_tick_split_strict_native_required: bool = False
    cxxcore10_env_tick_split_parity_policy: str = "final_and_sparse"


@dataclass(frozen=True)
class EnvTickInputs:
    schema_version: str
    inputs: TickIslandLayoutCacheInputs
    state_digest: str
    ticks: int
    agents: int
    cells: int


@dataclass(frozen=True)
class EnvTickMaterializedResult:
    schema_version: str
    source: str
    fields: dict[str, np.ndarray]
    digest: str
    timing: dict[str, float]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class EnvTickSideEffectPlan:
    schema_version: str
    source_digest: str
    write_fields: tuple[str, ...]
    python_owned_side_effects: tuple[str, ...]
    metrics_schema_changed: bool
    checkpoint_schema_changed: bool
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class EnvTickWritebackResult:
    schema_version: str
    writeback_mode: str
    output: dict[str, np.ndarray]
    output_digest: str
    side_effect_plan: EnvTickSideEffectPlan
    timing: dict[str, float]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class Cxxcore10SplitResult:
    schema_version: str
    mode: str
    status: str
    materialized_source: str
    output: dict[str, np.ndarray] | None
    parity: dict[str, Any] | None
    fallback_reason: str | None
    first_divergence: dict[str, Any] | None
    retained_native_output: bool
    writeback_applied: bool
    python_state_restored: bool
    native_available: bool
    timing: dict[str, float]
    diagnostics: dict[str, Any]


def config_from_sim_cfg(sim_cfg: Any) -> Cxxcore10SplitConfig:
    return Cxxcore10SplitConfig(
        cxxcore10_env_tick_split_enable=bool(getattr(sim_cfg, "cxxcore10_env_tick_split_enable", False)),
        cxxcore10_env_tick_split_mode=str(getattr(sim_cfg, "cxxcore10_env_tick_split_mode", "off") or "off"),
        cxxcore10_env_tick_split_parity_interval=int(
            getattr(sim_cfg, "cxxcore10_env_tick_split_parity_interval", 1) or 1
        ),
        cxxcore10_env_tick_split_restore_on_divergence=bool(
            getattr(sim_cfg, "cxxcore10_env_tick_split_restore_on_divergence", True)
        ),
        cxxcore10_env_tick_split_emit_diagnostics=bool(
            getattr(sim_cfg, "cxxcore10_env_tick_split_emit_diagnostics", True)
        ),
        cxxcore10_env_tick_split_fail_closed=bool(getattr(sim_cfg, "cxxcore10_env_tick_split_fail_closed", True)),
        cxxcore10_env_tick_split_strict_native_required=bool(
            getattr(sim_cfg, "cxxcore10_env_tick_split_strict_native_required", False)
        ),
        cxxcore10_env_tick_split_parity_policy=str(
            getattr(sim_cfg, "cxxcore10_env_tick_split_parity_policy", "final_and_sparse") or "final_and_sparse"
        ),
    )


def as_env_tick_inputs(inputs: TickIslandLayoutCacheInputs) -> EnvTickInputs:
    state = inputs.backend.runner.state
    payload = {
        "agent_ids": np.asarray(state.agent_ids),
        "actions_by_tick": np.asarray(state.actions_by_tick),
        "initial_energy": np.asarray(state.initial_energy),
        "initial_health": np.asarray(state.initial_health),
        "initial_injury": np.asarray(state.initial_injury),
        "initial_scar": np.asarray(state.initial_scar),
        "pos_y_by_id": np.asarray(state.pos_y_by_id),
        "pos_x_by_id": np.asarray(state.pos_x_by_id),
        "agent_grid": np.asarray(state.agent_grid),
        "grid": np.asarray(state.grid),
        "move_cost_mults": np.asarray(state.move_cost_mults),
        "juvenile_harvest_mults": np.asarray(state.juvenile_harvest_mults),
        "cost_layout_rows": np.asarray(inputs.cache_snapshot.layout.rows),
    }
    return EnvTickInputs(
        schema_version=SCHEMA_VERSION,
        inputs=inputs,
        state_digest=state_digest(payload),
        ticks=int(state.actions_by_tick.shape[0]),
        agents=int(state.agent_ids.shape[0]),
        cells=int(state.config.height) * int(state.config.width),
    )


def materialize_python_reference(env_inputs: EnvTickInputs) -> EnvTickMaterializedResult:
    t0 = time.perf_counter()
    output = _normalize_materialized(run_tick_island_python_reference(env_inputs.inputs))
    return EnvTickMaterializedResult(
        schema_version=SCHEMA_VERSION,
        source="python_reference",
        fields=output,
        digest=state_digest(output),
        timing={"materialize_s": time.perf_counter() - t0},
        diagnostics={"source": "python_reference", "metrics_schema_changed": False, "checkpoint_schema_changed": False},
    )


def materialize_native_candidate(
    env_inputs: EnvTickInputs,
    *,
    native_backend: Any,
    rng_payload: Cxxcore7RngPayload | None = None,
    validation_rng: bool = False,
    inject_divergence: bool = False,
) -> EnvTickMaterializedResult:
    t0 = time.perf_counter()
    native_inputs = env_inputs.inputs
    if validation_rng:
        if rng_payload is None:
            raise ValueError("rng_payload_missing")
        native_inputs = apply_validation_action_override(env_inputs.inputs, rng_payload)
    raw = native_backend(native_inputs, rng_payload=rng_payload if validation_rng else None, export_final=True)
    fields = _normalize_materialized(raw)
    if inject_divergence and fields["new_energy"].size:
        fields = dict(fields)
        fields["new_energy"] = np.asarray(fields["new_energy"], dtype=np.float32).copy()
        fields["new_energy"][0] = np.float32(float(fields["new_energy"][0]) + 1.0)
    return EnvTickMaterializedResult(
        schema_version=SCHEMA_VERSION,
        source="native_candidate",
        fields=fields,
        digest=state_digest(fields),
        timing={"materialize_s": time.perf_counter() - t0},
        diagnostics={"source": "native_candidate", "metrics_schema_changed": False, "checkpoint_schema_changed": False},
    )


def build_side_effect_plan(materialized: EnvTickMaterializedResult) -> EnvTickSideEffectPlan:
    return EnvTickSideEffectPlan(
        schema_version=SCHEMA_VERSION,
        source_digest=materialized.digest,
        write_fields=MATERIALIZED_FIELDS,
        python_owned_side_effects=tuple(item["name"] for item in side_effect_taxonomy() if item["classification"].startswith("python_")),
        metrics_schema_changed=False,
        checkpoint_schema_changed=False,
        diagnostics={
            "diagnostic_payload_only": True,
            "native_mutates_live_python_objects": False,
            "selection_reproduction_lineage_owned_by": "python_out_of_scope",
        },
    )


def apply_python_owned_writeback(
    env_inputs: EnvTickInputs,
    materialized: EnvTickMaterializedResult,
    *,
    writeback_mode: str,
) -> EnvTickWritebackResult:
    t0 = time.perf_counter()
    plan = build_side_effect_plan(materialized)
    output = {name: np.asarray(materialized.fields[name]).copy() for name in MATERIALIZED_FIELDS}
    timing = {"writeback_s": time.perf_counter() - t0}
    return EnvTickWritebackResult(
        schema_version=SCHEMA_VERSION,
        writeback_mode=writeback_mode,
        output=output,
        output_digest=state_digest(output),
        side_effect_plan=plan,
        timing=timing,
        diagnostics={
            "input_state_digest": env_inputs.state_digest,
            "writeback_mode": writeback_mode,
            "metrics_schema_changed": False,
            "checkpoint_schema_changed": False,
            "live_python_objects_mutated": False,
        },
    )


def run_env_tick_split_transaction(
    inputs: TickIslandLayoutCacheInputs,
    *,
    config: Cxxcore10SplitConfig,
    native_backend: Any | None = None,
    rng_payload: Cxxcore7RngPayload | None = None,
    inject_divergence: bool = False,
) -> Cxxcore10SplitResult:
    mode = str(config.cxxcore10_env_tick_split_mode or "off")
    if mode not in SPLIT_MODES:
        raise ValueError(f"unsupported CXXCORE-10 split mode: {mode}")
    parity_policy = str(config.cxxcore10_env_tick_split_parity_policy or "final_and_sparse")
    if parity_policy not in PARITY_POLICIES:
        raise ValueError(f"unsupported CXXCORE-10 parity policy: {parity_policy}")

    timing = _empty_timing()
    started = time.perf_counter()
    env_inputs = as_env_tick_inputs(inputs)
    timing["input_capture_s"] = time.perf_counter() - started
    diagnostics = _base_diagnostics(env_inputs, mode=mode, parity_policy=parity_policy, config=config)

    if not bool(config.cxxcore10_env_tick_split_enable) or mode == "off":
        timing["total_wall_s"] = time.perf_counter() - started
        return _result(
            mode=mode,
            status="disabled",
            materialized_source="none",
            output=None,
            parity=None,
            fallback_reason="split_disabled",
            retained_native_output=False,
            writeback_applied=False,
            python_state_restored=False,
            native_available=bool(native_backend is not None),
            timing=timing,
            diagnostics=diagnostics,
        )

    ref_inputs = env_inputs
    validation_rng = mode == "native_guarded_transaction_validation_rng"
    if validation_rng:
        rng_t0 = time.perf_counter()
        if rng_payload is None:
            return _fallback(env_inputs, mode=mode, timing=timing, diagnostics=diagnostics, reason="rng_payload_missing", started=started)
        validation = validate_rng_payload(
            rng_payload,
            ticks=int(inputs.backend.runner.state.actions_by_tick.shape[0]),
            agents=int(inputs.backend.runner.state.actions_by_tick.shape[1]),
        )
        diagnostics["rng_validation"] = validation
        timing["rng_validation_s"] = time.perf_counter() - rng_t0
        if not validation["ok"]:
            return _fallback(env_inputs, mode=mode, timing=timing, diagnostics=diagnostics, reason="rng_payload_invalid", started=started)
        ref_inputs = as_env_tick_inputs(apply_validation_action_override(inputs, rng_payload))
    elif rng_payload is not None:
        diagnostics["rng_payload_ignored"] = True

    if mode == "python_split_reference":
        ref_t0 = time.perf_counter()
        py = materialize_python_reference(ref_inputs)
        timing["python_split_materialize_s"] = time.perf_counter() - ref_t0
        wb = apply_python_owned_writeback(ref_inputs, py, writeback_mode="python_reference")
        timing["python_writeback_s"] = wb.timing["writeback_s"]
        parity = compare_runner_backend_states(run_tick_island_python_reference(ref_inputs.inputs), wb.output, atol=0.0)
        timing["comparison_s"] = 0.0
        timing["total_wall_s"] = time.perf_counter() - started
        return _result(
            mode=mode,
            status="python_split_reference",
            materialized_source=py.source,
            output=wb.output,
            parity=parity,
            fallback_reason=None,
            retained_native_output=False,
            writeback_applied=True,
            python_state_restored=False,
            native_available=bool(native_backend is not None),
            timing=timing,
            diagnostics=diagnostics | {"writeback": wb.diagnostics, "side_effect_plan": _plan_payload(wb.side_effect_plan)},
        )

    if native_backend is None:
        if bool(config.cxxcore10_env_tick_split_strict_native_required and config.cxxcore10_env_tick_split_fail_closed):
            raise RuntimeError("CXXCORE-10 native backend required but unavailable")
        return _fallback(env_inputs, mode=mode, timing=timing, diagnostics=diagnostics, reason="native_unavailable", started=started)

    ref_t0 = time.perf_counter()
    py = materialize_python_reference(ref_inputs)
    timing["python_direct_reference_s"] = py.timing["materialize_s"]
    timing["python_split_materialize_s"] = time.perf_counter() - ref_t0

    native_t0 = time.perf_counter()
    try:
        native = materialize_native_candidate(
            env_inputs,
            native_backend=native_backend,
            rng_payload=rng_payload,
            validation_rng=validation_rng,
            inject_divergence=inject_divergence,
        )
    except Exception as exc:
        if config.cxxcore10_env_tick_split_fail_closed:
            raise
        return _fallback(
            env_inputs,
            mode=mode,
            timing=timing,
            diagnostics=diagnostics | {"exception_type": type(exc).__name__, "exception": str(exc)},
            reason="exception_fallback",
            started=started,
        )
    timing["native_backend_s"] = time.perf_counter() - native_t0

    cmp_t0 = time.perf_counter()
    parity = compare_runner_backend_states(py.fields, native.fields, atol=1.0e-5)
    timing["comparison_s"] = time.perf_counter() - cmp_t0
    if not bool(parity.get("ok", False)):
        if bool(config.cxxcore10_env_tick_split_restore_on_divergence):
            fallback_wb = apply_python_owned_writeback(ref_inputs, py, writeback_mode="fallback_python_reference")
            timing["fallback_restore_s"] = fallback_wb.timing["writeback_s"]
            timing["total_wall_s"] = time.perf_counter() - started
            return _result(
                mode=mode,
                status="fallback",
                materialized_source=py.source,
                output=fallback_wb.output,
                parity=parity,
                fallback_reason="divergence",
                retained_native_output=False,
                writeback_applied=False,
                python_state_restored=True,
                native_available=True,
                timing=timing,
                diagnostics=diagnostics | {"writeback": fallback_wb.diagnostics, "side_effect_plan": _plan_payload(fallback_wb.side_effect_plan)},
            )
        if bool(config.cxxcore10_env_tick_split_fail_closed):
            raise RuntimeError(f"CXXCORE-10 transaction divergence: {parity.get('first_divergence')!r}")

    source = py if mode == "native_shadow_transaction" else native
    wb_mode = "shadow_python_reference" if mode == "native_shadow_transaction" else "native_guarded_transaction"
    wb_t0 = time.perf_counter()
    wb = apply_python_owned_writeback(ref_inputs, source, writeback_mode=wb_mode)
    timing["python_writeback_s"] = time.perf_counter() - wb_t0
    timing["total_wall_s"] = time.perf_counter() - started
    return _result(
        mode=mode,
        status="shadow" if mode == "native_shadow_transaction" else "guarded_transaction",
        materialized_source=source.source,
        output=wb.output,
        parity=parity,
        fallback_reason=None,
        retained_native_output=mode != "native_shadow_transaction",
        writeback_applied=mode != "native_shadow_transaction",
        python_state_restored=False,
        native_available=True,
        timing=timing,
        diagnostics=diagnostics | {"writeback": wb.diagnostics, "side_effect_plan": _plan_payload(wb.side_effect_plan)},
    )


def result_payload(result: Cxxcore10SplitResult) -> dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "mode": result.mode,
        "status": result.status,
        "materialized_source": result.materialized_source,
        "parity": result.parity,
        "parity_passed": bool((result.parity or {}).get("ok", False)) if result.parity is not None else False,
        "fallback_reason": result.fallback_reason,
        "first_divergence": result.first_divergence,
        "retained_native_output": bool(result.retained_native_output),
        "writeback_applied": bool(result.writeback_applied),
        "python_state_restored": bool(result.python_state_restored),
        "native_available": bool(result.native_available),
        "timing": dict(result.timing),
        "diagnostics": dict(result.diagnostics),
        "state_digest": state_digest(result.output),
    }


def side_effect_taxonomy() -> list[dict[str, str]]:
    rows = [
        ("input_capture_actions", "pure_input_capture", "agent ids, actions, pre energy/health/injury/scar arrays"),
        ("validation_random_walk_action_override", "rng_payload_python_precomputed", "Python RNGManager schedules override arrays; C++ consumes arrays only"),
        ("cost_multiplier_arrays", "pure_input_capture", "move and juvenile multipliers are copied inputs"),
        ("CostChargeSurface_layout_cache", "pure_input_capture", "rebuildable layout/cache input, not checkpoint owned"),
        ("env_step_energy_health_injury_scar", "pure_compute_materialization", "materialized scalar outputs"),
        ("position_and_grid_outputs", "pure_compute_materialization", "materialized position/grid outputs"),
        ("resource_grid_gather_mutation", "pure_compute_materialization", "materialized grid/resource output"),
        ("death_mask_alive_dead_changes", "pure_compute_materialization", "death mask candidate output"),
        ("Agent.energy_health_injury_scar_writes", "python_object_writeback", "Python-owned writeback boundary"),
        ("position_grid_live_object_writes", "python_object_writeback", "Python-owned writeback boundary"),
        ("Agent.stats_last_deltas", "python_object_writeback", "Python-owned stat update"),
        ("survival_steps_energy_collected", "python_object_writeback", "Python-owned stat update"),
        ("death_counters_generation_bookkeeping", "python_side_effect_counter", "Python-owned generation counters"),
        ("Tier3_transition_crossing_accumulators", "python_side_effect_counter", "Python-owned from env last_step stats"),
        ("Tier4_exposure_and_death_bins", "python_side_effect_counter", "Python-owned age/exposure counters"),
        ("PatchWorld_harvest_movement_context", "python_side_effect_counter", "Python-owned patch accumulators"),
        ("Jungle_predator_reproduction_emissions", "out_of_scope_for_cxxcore10", "not owned by CXXCORE-10 split"),
        ("cost_charge_scalar_totals", "pure_compute_materialization", "candidate result fields, compared through transaction"),
        ("cost_charge_diagnostics", "diagnostic_payload_only", "out-of-band diagnostic payload only"),
        ("metrics_logger_input_fields", "metrics_input_only", "observed from Python state after writeback; schema unchanged"),
        ("checkpoint_relevant_state", "checkpoint_owned_python_state", "Python checkpoint schema unchanged"),
        ("biome_hazard_rng", "out_of_scope_for_cxxcore10", "deferred branch-dependent stochastic surface"),
        ("selection_reproduction_lineage_mutation", "out_of_scope_for_cxxcore10", "not read or written by split"),
    ]
    return [{"name": name, "classification": classification, "notes": notes} for name, classification, notes in rows]


def side_effect_taxonomy_payload() -> dict[str, Any]:
    return {
        "schema_version": "cxxcore10_side_effect_taxonomy_v1",
        "rows": side_effect_taxonomy(),
        "metrics_schema_changed": False,
        "checkpoint_schema_changed": False,
        "native_rng_added": False,
    }


def _normalize_materialized(raw: dict[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in MATERIALIZED_FIELDS:
        if name not in raw:
            raise ValueError(f"materialized field missing: {name}")
        arr = np.asarray(raw[name])
        if name == "died_mask":
            arr = arr.astype(bool, copy=False)
        elif name in {"new_pos_y", "new_pos_x"}:
            arr = arr.astype(np.int32, copy=False)
        elif name == "grid":
            arr = arr.astype(np.int8, copy=False)
        elif name == "cost_total":
            arr = arr.astype(np.float64, copy=False)
        else:
            arr = arr.astype(np.float32, copy=False)
        out[name] = np.ascontiguousarray(arr).copy()
    return out


def _base_diagnostics(
    env_inputs: EnvTickInputs,
    *,
    mode: str,
    parity_policy: str,
    config: Cxxcore10SplitConfig,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "parity_policy": parity_policy,
        "parity_interval": int(config.cxxcore10_env_tick_split_parity_interval),
        "agents": int(env_inputs.agents),
        "ticks": int(env_inputs.ticks),
        "cells": int(env_inputs.cells),
        "input_state_digest": env_inputs.state_digest,
        "metrics_schema_changed": False,
        "checkpoint_schema_changed": False,
        "native_rng_added": False,
        "native_mutates_live_python_objects": False,
    }


def _empty_timing() -> dict[str, float]:
    return {
        "total_wall_s": 0.0,
        "input_capture_s": 0.0,
        "python_direct_reference_s": 0.0,
        "python_split_materialize_s": 0.0,
        "native_backend_s": 0.0,
        "python_writeback_s": 0.0,
        "comparison_s": 0.0,
        "fallback_restore_s": 0.0,
        "rng_validation_s": 0.0,
        "diagnostics_s": 0.0,
        "cache_layout_refresh_s": 0.0,
    }


def _fallback(
    env_inputs: EnvTickInputs,
    *,
    mode: str,
    timing: dict[str, float],
    diagnostics: dict[str, Any],
    reason: str,
    started: float,
) -> Cxxcore10SplitResult:
    fallback_t0 = time.perf_counter()
    py = materialize_python_reference(env_inputs)
    wb = apply_python_owned_writeback(env_inputs, py, writeback_mode="fallback_python_reference")
    timing["fallback_restore_s"] = time.perf_counter() - fallback_t0
    timing["total_wall_s"] = time.perf_counter() - started
    return _result(
        mode=mode,
        status="fallback",
        materialized_source=py.source,
        output=wb.output,
        parity=None,
        fallback_reason=reason,
        retained_native_output=False,
        writeback_applied=False,
        python_state_restored=True,
        native_available=False,
        timing=timing,
        diagnostics=diagnostics | {"writeback": wb.diagnostics, "side_effect_plan": _plan_payload(wb.side_effect_plan)},
    )


def _result(
    *,
    mode: str,
    status: str,
    materialized_source: str,
    output: dict[str, np.ndarray] | None,
    parity: dict[str, Any] | None,
    fallback_reason: str | None,
    retained_native_output: bool,
    writeback_applied: bool,
    python_state_restored: bool,
    native_available: bool,
    timing: dict[str, float],
    diagnostics: dict[str, Any],
) -> Cxxcore10SplitResult:
    return Cxxcore10SplitResult(
        schema_version=SCHEMA_VERSION,
        mode=mode,
        status=status,
        materialized_source=materialized_source,
        output=output,
        parity=parity,
        fallback_reason=fallback_reason,
        first_divergence=(parity or {}).get("first_divergence") if parity else None,
        retained_native_output=retained_native_output,
        writeback_applied=writeback_applied,
        python_state_restored=python_state_restored,
        native_available=native_available,
        timing=timing,
        diagnostics=diagnostics,
    )


def _plan_payload(plan: EnvTickSideEffectPlan) -> dict[str, Any]:
    return {
        "schema_version": plan.schema_version,
        "source_digest": plan.source_digest,
        "write_fields": list(plan.write_fields),
        "python_owned_side_effects": list(plan.python_owned_side_effects),
        "metrics_schema_changed": bool(plan.metrics_schema_changed),
        "checkpoint_schema_changed": bool(plan.checkpoint_schema_changed),
        "diagnostics": dict(plan.diagnostics),
    }


__all__ = [
    "MATERIALIZED_FIELDS",
    "PARITY_POLICIES",
    "SCHEMA_VERSION",
    "SPLIT_MODES",
    "Cxxcore10SplitConfig",
    "Cxxcore10SplitResult",
    "EnvTickInputs",
    "EnvTickMaterializedResult",
    "EnvTickSideEffectPlan",
    "EnvTickWritebackResult",
    "apply_python_owned_writeback",
    "as_env_tick_inputs",
    "build_side_effect_plan",
    "config_from_sim_cfg",
    "materialize_native_candidate",
    "materialize_python_reference",
    "result_payload",
    "run_env_tick_split_transaction",
    "side_effect_taxonomy",
    "side_effect_taxonomy_payload",
]
