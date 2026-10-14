from __future__ import annotations

"""CXXCORE-9 guarded runner adapter contract.

Default-off guardrail layer for copied-state tick-island participation. This
module deliberately does not change normal runner execution. It packages the
lifecycle expected of a future live `_apply_env_tick` adapter: pre-state token,
native availability, Python reference, parity policy, fallback restoration, and
out-of-band diagnostics.
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
from perf.cxxcore7_rng_precompute import Cxxcore7RngPayload, validate_rng_payload
from perf.cxxcore8_guarded_runner_bridge import (
    Cxxcore8BridgeConfig,
    result_payload as cxxcore8_result_payload,
    run_guarded_runner_bridge,
    state_digest,
)


SCHEMA_VERSION = "cxxcore9_runner_adapter_v1"
ADAPTER_MODES = (
    "off",
    "shadow",
    "guarded_authoritative_no_rng",
    "guarded_authoritative_validation_rng",
)
PARITY_POLICIES = ("every_tick", "interval", "final_state")


@dataclass(frozen=True)
class Cxxcore9AdapterConfig:
    cxxcore9_runner_adapter_enable: bool = False
    cxxcore9_runner_adapter_mode: str = "off"
    cxxcore9_runner_adapter_parity_interval: int = 1
    cxxcore9_runner_adapter_restore_on_divergence: bool = True
    cxxcore9_runner_adapter_emit_diagnostics: bool = True
    cxxcore9_runner_adapter_fail_closed: bool = True
    cxxcore9_runner_adapter_strict_native_required: bool = False
    cxxcore9_runner_adapter_parity_policy: str = "every_tick"


@dataclass(frozen=True)
class Cxxcore9RestoreToken:
    schema_version: str
    pre_state_digest: str
    fields: tuple[str, ...]
    ticks: int
    agents: int
    cells: int


@dataclass(frozen=True)
class Cxxcore9AdapterResult:
    schema_version: str
    mode: str
    attempted_native: bool
    retained_native_output: bool
    parity_checked: bool
    parity_passed: bool
    fallback_reason: str | None
    first_divergence: dict[str, Any] | None
    timing: dict[str, float]
    output: dict[str, np.ndarray] | None
    restore_token: Cxxcore9RestoreToken | None
    diagnostics: dict[str, Any]
    native_available: bool
    fallback_occurred: bool
    python_state_restored: bool
    writeback_applied: bool


def config_from_sim_cfg(sim_cfg: Any) -> Cxxcore9AdapterConfig:
    return Cxxcore9AdapterConfig(
        cxxcore9_runner_adapter_enable=bool(getattr(sim_cfg, "cxxcore9_runner_adapter_enable", False)),
        cxxcore9_runner_adapter_mode=str(getattr(sim_cfg, "cxxcore9_runner_adapter_mode", "off") or "off"),
        cxxcore9_runner_adapter_parity_interval=int(
            getattr(sim_cfg, "cxxcore9_runner_adapter_parity_interval", 1) or 1
        ),
        cxxcore9_runner_adapter_restore_on_divergence=bool(
            getattr(sim_cfg, "cxxcore9_runner_adapter_restore_on_divergence", True)
        ),
        cxxcore9_runner_adapter_emit_diagnostics=bool(
            getattr(sim_cfg, "cxxcore9_runner_adapter_emit_diagnostics", True)
        ),
        cxxcore9_runner_adapter_fail_closed=bool(getattr(sim_cfg, "cxxcore9_runner_adapter_fail_closed", True)),
        cxxcore9_runner_adapter_strict_native_required=bool(
            getattr(sim_cfg, "cxxcore9_runner_adapter_strict_native_required", False)
        ),
        cxxcore9_runner_adapter_parity_policy=str(
            getattr(sim_cfg, "cxxcore9_runner_adapter_parity_policy", "every_tick") or "every_tick"
        ),
    )


def run_runner_adapter(
    inputs: TickIslandLayoutCacheInputs,
    *,
    config: Cxxcore9AdapterConfig,
    native_backend: Any | None,
    rng_payload: Cxxcore7RngPayload | None = None,
    inject_divergence: bool = False,
) -> Cxxcore9AdapterResult:
    mode = str(config.cxxcore9_runner_adapter_mode or "off")
    if mode not in ADAPTER_MODES:
        raise ValueError(f"unsupported CXXCORE-9 adapter mode: {mode}")
    parity_policy = str(config.cxxcore9_runner_adapter_parity_policy or "every_tick")
    if parity_policy not in PARITY_POLICIES:
        raise ValueError(f"unsupported CXXCORE-9 parity policy: {parity_policy}")

    timing = _empty_timing()
    total_t0 = time.perf_counter()
    diagnostics = _base_diagnostics(inputs, mode=mode, parity_policy=parity_policy, config=config)
    restore_token = _restore_token(inputs)
    timing["snapshot_restore_token_s"] = time.perf_counter() - total_t0

    if not bool(config.cxxcore9_runner_adapter_enable) or mode == "off":
        timing["total_wall_s"] = time.perf_counter() - total_t0
        return _result(
            mode=mode,
            attempted_native=False,
            retained_native_output=False,
            parity_checked=False,
            parity_passed=False,
            fallback_reason="adapter_disabled",
            first_divergence=None,
            timing=timing,
            output=None,
            restore_token=restore_token,
            diagnostics=diagnostics,
            native_available=bool(native_backend is not None),
            fallback_occurred=False,
            python_state_restored=False,
            writeback_applied=False,
        )

    if mode == "guarded_authoritative_validation_rng":
        rng_t0 = time.perf_counter()
        if rng_payload is None:
            return _fallback(
                inputs,
                mode=mode,
                timing=timing,
                diagnostics=diagnostics,
                restore_token=restore_token,
                reason="rng_payload_missing",
                native_available=bool(native_backend is not None),
                started=total_t0,
            )
        validation = validate_rng_payload(
            rng_payload,
            ticks=int(inputs.backend.runner.state.actions_by_tick.shape[0]),
            agents=int(inputs.backend.runner.state.actions_by_tick.shape[1]),
        )
        diagnostics["rng_validation"] = validation
        timing["rng_validation_s"] = time.perf_counter() - rng_t0
        if not validation["ok"]:
            return _fallback(
                inputs,
                mode=mode,
                timing=timing,
                diagnostics=diagnostics,
                restore_token=restore_token,
                reason="rng_payload_invalid",
                native_available=bool(native_backend is not None),
                started=total_t0,
            )
    elif rng_payload is not None:
        diagnostics["rng_payload_ignored"] = True

    if native_backend is None:
        if config.cxxcore9_runner_adapter_strict_native_required and config.cxxcore9_runner_adapter_fail_closed:
            raise RuntimeError("CXXCORE-9 native backend required but unavailable")
        return _fallback(
            inputs,
            mode=mode,
            timing=timing,
            diagnostics=diagnostics,
            restore_token=restore_token,
            reason="native_unavailable",
            native_available=False,
            started=total_t0,
        )

    bridge_mode = _bridge_mode(mode)
    bridge_cfg = Cxxcore8BridgeConfig(
        cxxcore8_guarded_runner_bridge_enable=True,
        cxxcore8_guarded_runner_bridge_mode=bridge_mode,
        cxxcore8_runner_bridge_compare_interval_ticks=max(0, int(config.cxxcore9_runner_adapter_parity_interval)),
        cxxcore8_runner_bridge_force_final_compare=True,
        cxxcore8_runner_bridge_fallback_on_mismatch=bool(config.cxxcore9_runner_adapter_restore_on_divergence),
        cxxcore8_runner_bridge_emit_compact_summary=bool(config.cxxcore9_runner_adapter_emit_diagnostics),
        cxxcore8_runner_bridge_allow_validation_rng_precompute=(mode == "guarded_authoritative_validation_rng"),
    )
    try:
        bridge = run_guarded_runner_bridge(
            inputs,
            config=bridge_cfg,
            native_backend=native_backend,
            rng_payload=rng_payload,
            inject_divergence=inject_divergence,
        )
    except Exception as exc:
        if config.cxxcore9_runner_adapter_fail_closed:
            raise
        return _fallback(
            inputs,
            mode=mode,
            timing=timing,
            diagnostics=diagnostics | {"exception_type": type(exc).__name__, "exception": str(exc)},
            restore_token=restore_token,
            reason="exception_fallback",
            native_available=True,
            started=total_t0,
        )

    bridge_payload = cxxcore8_result_payload(bridge)
    _merge_bridge_timing(timing, bridge_payload.get("timing") or {})
    parity = bridge_payload.get("parity") or {}
    fallback_reason = bridge_payload.get("fallback_reason") or None
    output = bridge.output
    timing["total_wall_s"] = time.perf_counter() - total_t0
    return _result(
        mode=mode,
        attempted_native=True,
        retained_native_output=bool(bridge_payload.get("cxx_result_retained", False)),
        parity_checked=bool(parity),
        parity_passed=bool(parity.get("ok", False)) if parity else False,
        fallback_reason=fallback_reason,
        first_divergence=parity.get("first_divergence") if parity else None,
        timing=timing,
        output=output,
        restore_token=restore_token,
        diagnostics=diagnostics | {"bridge": bridge_payload},
        native_available=bool(bridge_payload.get("native_available", False)),
        fallback_occurred=bool(bridge_payload.get("fallback_occurred", False)),
        python_state_restored=bool(bridge_payload.get("python_state_restored", False)),
        writeback_applied=bool(bridge_payload.get("writeback_applied", False)),
    )


def result_payload(result: Cxxcore9AdapterResult) -> dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "mode": result.mode,
        "attempted_native": bool(result.attempted_native),
        "retained_native_output": bool(result.retained_native_output),
        "parity_checked": bool(result.parity_checked),
        "parity_passed": bool(result.parity_passed),
        "fallback_reason": result.fallback_reason,
        "first_divergence": result.first_divergence,
        "timing": dict(result.timing),
        "restore_token": None if result.restore_token is None else result.restore_token.__dict__,
        "diagnostics": dict(result.diagnostics),
        "native_available": bool(result.native_available),
        "fallback_occurred": bool(result.fallback_occurred),
        "python_state_restored": bool(result.python_state_restored),
        "writeback_applied": bool(result.writeback_applied),
        "state_digest": state_digest(result.output),
    }


def adapter_boundary_map() -> dict[str, Any]:
    return {
        "schema_version": "cxxcore9_runner_adapter_boundary_map_v1",
        "phases": [
            {
                "phase": "pre_action_action_override",
                "classification": "Python authoritative only",
                "notes": "Validation random-walk override remains scheduled by Python RNGManager; C++ may consume copied arrays only.",
            },
            {
                "phase": "cost_multiplier_and_cost_layout_lifecycle",
                "classification": "Python authoritative only with rebuildable copied C++ layout input",
                "notes": "CostChargeSurface rows/cache are rebuildable diagnostics/backend input, not checkpoint-owned state.",
            },
            {
                "phase": "pre_state_snapshot_clone",
                "classification": "C++ copied-state only",
                "notes": "Adapter token covers arrays needed to restore or compare the selected tick-island slice.",
            },
            {
                "phase": "native_backend_execution",
                "classification": "C++ may materialize candidate output",
                "notes": "No independent RNG; no mutation of Python runner state before parity.",
            },
            {
                "phase": "python_reference_execution",
                "classification": "Python authoritative only",
                "notes": "Reference output remains available for fallback and parity.",
            },
            {
                "phase": "comparison",
                "classification": "C++ copied-state only",
                "notes": "Compare final materialized energy, health, injury, scar, death mask, position, grid, and cost totals.",
            },
            {
                "phase": "writeback",
                "classification": "adapter may write back under guarded pilot",
                "notes": "CXXCORE-9 wrapper can retain candidate output only inside copied-state pilot; live Agent/GridWorld writeback is deferred.",
            },
            {
                "phase": "fallback_restoration",
                "classification": "Python authoritative only",
                "notes": "On missing native, malformed RNG, divergence, or exception fallback returns Python reference and retains no native output.",
            },
            {
                "phase": "diagnostics_metrics_observation",
                "classification": "explicitly out of scope for metric definitions",
                "notes": "Adapter diagnostics are out-of-band summaries and do not feed metrics.csv definitions.",
            },
            {
                "phase": "death_stat_side_effect_update",
                "classification": "explicitly out of scope",
                "notes": "Runner death counters, Agent.stats deltas, Tier3/Patch/Tier4 accumulators, selection, reproduction, lineage, and checkpoints stay Python-owned.",
            },
        ],
        "restore_fields": restore_field_map(),
    }


def restore_field_map() -> list[str]:
    return [
        "agent_ids",
        "actions_by_tick",
        "initial_energy",
        "initial_health",
        "initial_injury",
        "initial_scar",
        "pos_y_by_id",
        "pos_x_by_id",
        "agent_grid",
        "grid",
        "move_cost_mults",
        "juvenile_harvest_mults",
        "cost_layout_rows",
        "cost_layout_flags",
        "cache_snapshot_digests",
    ]


def _bridge_mode(mode: str) -> str:
    if mode == "shadow":
        return "shadow"
    if mode == "guarded_authoritative_no_rng":
        return "authoritative_no_rng"
    if mode == "guarded_authoritative_validation_rng":
        return "authoritative_precomputed_rng"
    raise ValueError(f"mode has no bridge equivalent: {mode}")


def _restore_token(inputs: TickIslandLayoutCacheInputs) -> Cxxcore9RestoreToken:
    state = inputs.backend.runner.state
    pre_payload = {
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
    return Cxxcore9RestoreToken(
        schema_version=SCHEMA_VERSION,
        pre_state_digest=state_digest(pre_payload),
        fields=tuple(restore_field_map()),
        ticks=int(state.actions_by_tick.shape[0]),
        agents=int(state.agent_ids.shape[0]),
        cells=int(state.config.height) * int(state.config.width),
    )


def _base_diagnostics(
    inputs: TickIslandLayoutCacheInputs,
    *,
    mode: str,
    parity_policy: str,
    config: Cxxcore9AdapterConfig,
) -> dict[str, Any]:
    state = inputs.backend.runner.state
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "parity_policy": parity_policy,
        "parity_interval": int(config.cxxcore9_runner_adapter_parity_interval),
        "agents": int(state.agent_ids.shape[0]),
        "ticks": int(state.actions_by_tick.shape[0]),
        "cells": int(state.config.height) * int(state.config.width),
        "metrics_boundary": "no metrics definition changes; diagnostics are out-of-band",
        "checkpoint_boundary": "no checkpoint schema/state changes; adapter state is rebuildable from Python authoritative state",
        "selection_reproduction_lineage_boundary": "not read or written by adapter",
        "cpp_rng_boundary": "no independent C++ RNG",
        "live_writeback_boundary": "copied-state pilot only; live Agent/GridWorld writeback deferred",
    }


def _empty_timing() -> dict[str, float]:
    return {
        "total_wall_s": 0.0,
        "snapshot_restore_token_s": 0.0,
        "python_reference_s": 0.0,
        "rng_validation_s": 0.0,
        "native_backend_s": 0.0,
        "materialization_s": 0.0,
        "sparse_compare_s": 0.0,
        "final_compare_s": 0.0,
        "writeback_s": 0.0,
        "diagnostics_s": 0.0,
        "fallback_restoration_s": 0.0,
    }


def _merge_bridge_timing(timing: dict[str, float], bridge_timing: dict[str, Any]) -> None:
    timing["python_reference_s"] = float(bridge_timing.get("python_reference_s", 0.0))
    timing["native_backend_s"] = float(bridge_timing.get("native_compute_s", 0.0))
    timing["materialization_s"] = float(bridge_timing.get("materialization_s", 0.0))
    timing["sparse_compare_s"] = float(bridge_timing.get("sparse_compare_s", 0.0))
    timing["final_compare_s"] = float(bridge_timing.get("final_compare_s", 0.0))
    timing["writeback_s"] = float(bridge_timing.get("writeback_s", 0.0))
    timing["diagnostics_s"] = float(bridge_timing.get("diagnostics_s", 0.0))
    timing["fallback_restoration_s"] = float(bridge_timing.get("fallback_s", 0.0))


def _fallback(
    inputs: TickIslandLayoutCacheInputs,
    *,
    mode: str,
    timing: dict[str, float],
    diagnostics: dict[str, Any],
    restore_token: Cxxcore9RestoreToken,
    reason: str,
    native_available: bool,
    started: float,
) -> Cxxcore9AdapterResult:
    fallback_t0 = time.perf_counter()
    output = run_tick_island_python_reference(inputs)
    timing["fallback_restoration_s"] = time.perf_counter() - fallback_t0
    timing["total_wall_s"] = time.perf_counter() - started
    return _result(
        mode=mode,
        attempted_native=False,
        retained_native_output=False,
        parity_checked=False,
        parity_passed=False,
        fallback_reason=reason,
        first_divergence=None,
        timing=timing,
        output=output,
        restore_token=restore_token,
        diagnostics=diagnostics,
        native_available=native_available,
        fallback_occurred=True,
        python_state_restored=True,
        writeback_applied=False,
    )


def _result(
    *,
    mode: str,
    attempted_native: bool,
    retained_native_output: bool,
    parity_checked: bool,
    parity_passed: bool,
    fallback_reason: str | None,
    first_divergence: dict[str, Any] | None,
    timing: dict[str, float],
    output: dict[str, np.ndarray] | None,
    restore_token: Cxxcore9RestoreToken | None,
    diagnostics: dict[str, Any],
    native_available: bool,
    fallback_occurred: bool,
    python_state_restored: bool,
    writeback_applied: bool,
) -> Cxxcore9AdapterResult:
    return Cxxcore9AdapterResult(
        schema_version=SCHEMA_VERSION,
        mode=mode,
        attempted_native=attempted_native,
        retained_native_output=retained_native_output,
        parity_checked=parity_checked,
        parity_passed=parity_passed,
        fallback_reason=fallback_reason,
        first_divergence=first_divergence,
        timing=timing,
        output=output,
        restore_token=restore_token,
        diagnostics=diagnostics,
        native_available=native_available,
        fallback_occurred=fallback_occurred,
        python_state_restored=python_state_restored,
        writeback_applied=writeback_applied,
    )


def assert_output_matches_python_reference(
    inputs: TickIslandLayoutCacheInputs,
    result: Cxxcore9AdapterResult,
    *,
    atol: float = 1.0e-6,
) -> dict[str, Any]:
    expected = run_tick_island_python_reference(inputs)
    return compare_runner_backend_states(expected, result.output, atol=atol)


__all__ = [
    "ADAPTER_MODES",
    "PARITY_POLICIES",
    "SCHEMA_VERSION",
    "Cxxcore9AdapterConfig",
    "Cxxcore9AdapterResult",
    "Cxxcore9RestoreToken",
    "adapter_boundary_map",
    "assert_output_matches_python_reference",
    "config_from_sim_cfg",
    "restore_field_map",
    "result_payload",
    "run_runner_adapter",
]
