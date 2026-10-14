from __future__ import annotations

"""CXXCORE-11 default-off live runner transaction hook pilot.

The hook adapts live single-tick runner state into the CXXCORE-10 transaction
boundary. It deliberately keeps Python env-step execution authoritative in
CXXCORE-11, then compares/optionally retains materialized native scalar outputs
only after parity.
"""

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from perf.cxxcore5_tick_island_layout_cache import (
    CostLayoutCache,
    TickIslandLayoutCacheInputs,
    as_tick_island_inputs,
    compare_runner_backend_states,
    run_tick_island_python_reference,
)
from perf.cxxcore7_rng_precompute import (
    Cxxcore7RngPayload,
    _array_contract,
    build_rng_payload,
)
from perf.cxxcore8_guarded_runner_bridge import state_digest
from perf.cxxcore10_live_env_tick_side_effect_split import (
    Cxxcore10SplitConfig,
    result_payload as cxxcore10_result_payload,
    run_env_tick_split_transaction,
)
from perf.cxxcore_env_apply_shadow import config_from_env


SCHEMA_VERSION = "cxxcore11_live_runner_transaction_hook_v1"
LIVE_HOOK_MODES = (
    "off",
    "python_split_reference_live_hook",
    "native_shadow_live_hook",
    "native_guarded_no_rng_live_hook",
    "native_guarded_validation_rng_live_hook",
)
PARITY_POLICIES = ("final_and_sparse", "every_tick", "interval")


@dataclass(frozen=True)
class Cxxcore11LiveHookConfig:
    cxxcore11_live_runner_hook_enable: bool = False
    cxxcore11_live_runner_hook_mode: str = "off"
    cxxcore11_live_runner_hook_parity_interval: int = 1
    cxxcore11_live_runner_hook_restore_on_divergence: bool = True
    cxxcore11_live_runner_hook_fail_closed: bool = True
    cxxcore11_live_runner_hook_emit_diagnostics: bool = True
    cxxcore11_live_runner_hook_strict_native_required: bool = False
    cxxcore11_live_runner_hook_parity_policy: str = "final_and_sparse"
    cxxcore11_live_runner_hook_rng_policy: str = "none"


@dataclass(frozen=True)
class Cxxcore11LiveHookResult:
    schema_version: str
    mode: str
    status: str
    output: dict[str, np.ndarray] | None
    retained_native_output: bool
    apply_output_to_live_scalars: bool
    parity: dict[str, Any] | None
    live_observed_parity: dict[str, Any] | None
    fallback_reason: str | None
    first_divergence: dict[str, Any] | None
    python_state_restored: bool
    native_available: bool
    timing: dict[str, float]
    diagnostics: dict[str, Any]


def config_from_sim_cfg(sim_cfg: Any) -> Cxxcore11LiveHookConfig:
    return Cxxcore11LiveHookConfig(
        cxxcore11_live_runner_hook_enable=bool(getattr(sim_cfg, "cxxcore11_live_runner_hook_enable", False)),
        cxxcore11_live_runner_hook_mode=str(getattr(sim_cfg, "cxxcore11_live_runner_hook_mode", "off") or "off"),
        cxxcore11_live_runner_hook_parity_interval=int(
            getattr(sim_cfg, "cxxcore11_live_runner_hook_parity_interval", 1) or 1
        ),
        cxxcore11_live_runner_hook_restore_on_divergence=bool(
            getattr(sim_cfg, "cxxcore11_live_runner_hook_restore_on_divergence", True)
        ),
        cxxcore11_live_runner_hook_fail_closed=bool(getattr(sim_cfg, "cxxcore11_live_runner_hook_fail_closed", True)),
        cxxcore11_live_runner_hook_emit_diagnostics=bool(
            getattr(sim_cfg, "cxxcore11_live_runner_hook_emit_diagnostics", True)
        ),
        cxxcore11_live_runner_hook_strict_native_required=bool(
            getattr(sim_cfg, "cxxcore11_live_runner_hook_strict_native_required", False)
        ),
        cxxcore11_live_runner_hook_parity_policy=str(
            getattr(sim_cfg, "cxxcore11_live_runner_hook_parity_policy", "final_and_sparse") or "final_and_sparse"
        ),
        cxxcore11_live_runner_hook_rng_policy=str(
            getattr(sim_cfg, "cxxcore11_live_runner_hook_rng_policy", "none") or "none"
        ),
    )


def validate_config(config: Cxxcore11LiveHookConfig) -> None:
    if config.cxxcore11_live_runner_hook_mode not in LIVE_HOOK_MODES:
        raise ValueError(f"unsupported CXXCORE-11 live hook mode: {config.cxxcore11_live_runner_hook_mode}")
    if config.cxxcore11_live_runner_hook_parity_policy not in PARITY_POLICIES:
        raise ValueError(f"unsupported CXXCORE-11 parity policy: {config.cxxcore11_live_runner_hook_parity_policy}")
    if int(config.cxxcore11_live_runner_hook_parity_interval) < 1:
        raise ValueError("cxxcore11_live_runner_hook_parity_interval must be >= 1")
    if config.cxxcore11_live_runner_hook_rng_policy not in {"none", "validation_action_override"}:
        raise ValueError(f"unsupported CXXCORE-11 RNG policy: {config.cxxcore11_live_runner_hook_rng_policy}")


def live_hook_preflight_audit() -> dict[str, Any]:
    rows = [
        ("live_agent_action_arrays", "safe_input_capture", "agent ids, actions, energy/health/injury/scar copied before env step"),
        ("CostChargeSurface_layout_cache", "safe_input_capture", "rebuildable Python-derived layout; no checkpoint ownership"),
        ("validation_action_override", "python_precomputed_rng_payload", "Python RNGManager remains owner; payload copies action overrides"),
        ("cxxcore10_transaction_boundary", "native_materialized_compute", "C++ may materialize candidate arrays only"),
        ("live_Agent_scalar_writeback", "python_side_effect_writeback", "Python runner remains writer"),
        ("live_GridWorld_arrays", "python_side_effect_writeback", "Python env.step_agents_batch remains authoritative in CXXCORE-11"),
        ("Tier3_Tier4_Patch_counters", "python_diagnostic_counter", "Python-owned counters remain unchanged"),
        ("metrics_logger_inputs", "metrics_input_only", "observed from Python-owned state; schema unchanged"),
        ("checkpoint_payload_state", "checkpoint_owned_python_state", "checkpoint schema unchanged"),
    ]
    return {
        "schema_version": "cxxcore11_preflight_audit_v1",
        "rows": [{"name": n, "classification": c, "notes": notes} for n, c, notes in rows],
        "stop_gate_present": False,
        "direct_python_env_step_skip_supported": False,
        "reason_no_python_skip": "native materialized output does not own live agent_grid, Tier counters, patch counters, diagnostics, or checkpoint/metrics lifecycle",
        "decision_gate": "parity_hook_possible_but_python_env_step_remains_authoritative",
    }


def prepare_live_tick_inputs(
    *,
    env: Any,
    env_cfg: Any,
    sim_cfg: Any,
    alive: list[Any],
    agent_ids_np: Any,
    actions_np: Any,
    energies: Any,
    healths: Any,
    injuries: Any,
    scars: Any,
    move_cost_mults: Any | None,
    juvenile_harvest_mults: Any | None,
    cache: CostLayoutCache | None = None,
) -> tuple[TickIslandLayoutCacheInputs, CostLayoutCache, dict[str, Any]]:
    t0 = time.perf_counter()
    cache_obj = cache if cache is not None else CostLayoutCache(env_cfg=env_cfg, sim_cfg=sim_cfg)
    snapshot, info = cache_obj.build_or_update(alive)
    agent_ids = np.asarray(agent_ids_np, dtype=np.int32)
    inputs = as_tick_island_inputs(
        config=config_from_env(env),
        agent_ids=agent_ids,
        initial_energy=np.asarray(energies, dtype=np.float32).copy(),
        initial_health=np.asarray(healths, dtype=np.float32).copy(),
        initial_injury=np.asarray(injuries, dtype=np.float32).copy(),
        initial_scar=np.asarray(scars, dtype=np.float32).copy(),
        pos_y_by_id=np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32).copy(),
        pos_x_by_id=np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32).copy(),
        agent_grid=np.asarray(getattr(env, "agent_grid"), dtype=np.int32).copy(),
        grid=np.asarray(getattr(env, "grid"), dtype=np.int8).copy(),
        actions_by_tick=np.asarray(actions_np, dtype=np.int32).reshape(1, -1).copy(),
        cache_snapshot=snapshot,
        move_cost_mults=np.ones((agent_ids.shape[0],), dtype=np.float32)
        if move_cost_mults is None
        else np.asarray(move_cost_mults, dtype=np.float32).copy(),
        juvenile_harvest_mults=np.ones((agent_ids.shape[0],), dtype=np.float32)
        if juvenile_harvest_mults is None
        else np.asarray(juvenile_harvest_mults, dtype=np.float32).copy(),
    )
    diagnostics = {
        "cache_mode": str(info.get("cache_mode", "")),
        "changed_rows": [int(v) for v in info.get("changed_rows", [])],
        "cache_layout_refresh_s": float(time.perf_counter() - t0),
        "cache_hits": int(snapshot.cache_hits),
        "cache_misses": int(snapshot.cache_misses),
        "full_rebuilds": int(snapshot.full_rebuilds),
        "rows_rebuilt": int(snapshot.rows_rebuilt),
    }
    return inputs, cache_obj, diagnostics


def observed_materialized_from_live_python(
    *,
    inputs: TickIslandLayoutCacheInputs,
    env: Any,
    died_mask: Any,
    new_energy: Any,
    new_health: Any,
    new_injury: Any,
    new_scar: Any,
) -> dict[str, np.ndarray]:
    state = inputs.backend.runner.state
    agent_ids = np.asarray(state.agent_ids, dtype=np.int32)
    reference = run_tick_island_python_reference(inputs)
    return {
        "new_energy": np.asarray(new_energy, dtype=np.float32).copy(),
        "new_health": np.asarray(new_health, dtype=np.float32).copy(),
        "new_injury": np.asarray(new_injury, dtype=np.float32).copy(),
        "new_scar": np.asarray(new_scar, dtype=np.float32).copy(),
        "died_mask": np.asarray(died_mask, dtype=bool).copy(),
        "new_pos_y": np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32)[agent_ids].copy(),
        "new_pos_x": np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32)[agent_ids].copy(),
        "grid": np.asarray(getattr(env, "grid"), dtype=np.int8).copy(),
        "cost_total_by_agent": np.asarray(reference["cost_total_by_agent"], dtype=np.float32).copy(),
        "cost_total": np.asarray(reference["cost_total"], dtype=np.float64).copy(),
    }


def validation_rng_payload_from_actions(actions_by_tick: Any) -> Cxxcore7RngPayload:
    values = np.asarray(actions_by_tick, dtype=np.int32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    arr = _array_contract(
        name="validation_action_override_ints",
        values=np.ascontiguousarray(values, dtype=np.int32),
        source_rng_stream="rng_manager.policy_np",
        source_function="TickEnvApplyMixin._apply_env_tick validation random-walk override",
        generation_start=0,
        tick_start=0,
        draw_schedule="one int32 draw per active agent per tick, aligned to action matrix columns",
        consumed_conditionally=False,
        sentinel=-1,
        tolerance=0.0,
    )
    return build_rng_payload(validation_action_override_ints=arr)


def run_live_runner_transaction_hook(
    inputs: TickIslandLayoutCacheInputs,
    *,
    observed_python: dict[str, np.ndarray],
    config: Cxxcore11LiveHookConfig,
    native_backend: Any | None = None,
    rng_payload: Cxxcore7RngPayload | None = None,
    inject_divergence: bool = False,
    extra_diagnostics: dict[str, Any] | None = None,
) -> Cxxcore11LiveHookResult:
    validate_config(config)
    mode = str(config.cxxcore11_live_runner_hook_mode)
    timing = _empty_timing()
    started = time.perf_counter()
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "parity_policy": config.cxxcore11_live_runner_hook_parity_policy,
        "parity_interval": int(config.cxxcore11_live_runner_hook_parity_interval),
        "rng_policy": config.cxxcore11_live_runner_hook_rng_policy,
        "python_env_step_authoritative": True,
        "native_skips_python_env_step": False,
        "metrics_schema_changed": False,
        "checkpoint_schema_changed": False,
        "native_rng_added": False,
        "native_mutates_live_python_objects": False,
    } | dict(extra_diagnostics or {})
    if not bool(config.cxxcore11_live_runner_hook_enable) or mode == "off":
        timing["total_wall_s"] = time.perf_counter() - started
        return _result(
            mode=mode,
            status="disabled",
            output=None,
            retained_native_output=False,
            apply_output_to_live_scalars=False,
            parity=None,
            live_observed_parity=None,
            fallback_reason="hook_disabled",
            python_state_restored=False,
            native_available=bool(native_backend is not None),
            timing=timing,
            diagnostics=diagnostics,
        )
    if mode in {"native_shadow_live_hook", "native_guarded_no_rng_live_hook", "native_guarded_validation_rng_live_hook"}:
        if native_backend is None:
            if config.cxxcore11_live_runner_hook_strict_native_required and config.cxxcore11_live_runner_hook_fail_closed:
                raise RuntimeError("CXXCORE-11 native backend required but unavailable")
            return _fallback_result(
                mode=mode,
                observed_python=observed_python,
                timing=timing,
                started=started,
                diagnostics=diagnostics,
                reason="native_unavailable",
                native_available=False,
            )
    if mode == "native_guarded_validation_rng_live_hook" and rng_payload is None:
        return _fallback_result(
            mode=mode,
            observed_python=observed_python,
            timing=timing,
            started=started,
            diagnostics=diagnostics,
            reason="rng_payload_missing",
            native_available=bool(native_backend is not None),
        )

    cxx10_mode = _to_cxxcore10_mode(mode)
    cxx10_cfg = Cxxcore10SplitConfig(
        cxxcore10_env_tick_split_enable=True,
        cxxcore10_env_tick_split_mode=cxx10_mode,
        cxxcore10_env_tick_split_parity_interval=int(config.cxxcore11_live_runner_hook_parity_interval),
        cxxcore10_env_tick_split_restore_on_divergence=bool(config.cxxcore11_live_runner_hook_restore_on_divergence),
        cxxcore10_env_tick_split_emit_diagnostics=bool(config.cxxcore11_live_runner_hook_emit_diagnostics),
        cxxcore10_env_tick_split_fail_closed=bool(config.cxxcore11_live_runner_hook_fail_closed),
        cxxcore10_env_tick_split_strict_native_required=bool(config.cxxcore11_live_runner_hook_strict_native_required),
        cxxcore10_env_tick_split_parity_policy=str(config.cxxcore11_live_runner_hook_parity_policy),
    )
    try:
        cxx10 = run_env_tick_split_transaction(
            inputs,
            config=cxx10_cfg,
            native_backend=native_backend,
            rng_payload=rng_payload,
            inject_divergence=inject_divergence,
        )
    except Exception as exc:
        if config.cxxcore11_live_runner_hook_fail_closed:
            raise
        return _fallback_result(
            mode=mode,
            observed_python=observed_python,
            timing=timing,
            started=started,
            diagnostics=diagnostics | {"exception_type": type(exc).__name__, "exception": str(exc)},
            reason="exception_fallback",
            native_available=bool(native_backend is not None),
        )
    cxx10_payload = cxxcore10_result_payload(cxx10)
    _merge_cxxcore10_timing(timing, cxx10_payload.get("timing") or {})
    if cxx10.output is None:
        return _fallback_result(
            mode=mode,
            observed_python=observed_python,
            timing=timing,
            started=started,
            diagnostics=diagnostics | {"cxxcore10": cxx10_payload},
            reason=str(cxx10.fallback_reason or "transaction_no_output"),
            native_available=bool(cxx10.native_available),
        )

    cmp_t0 = time.perf_counter()
    live_parity = compare_runner_backend_states(observed_python, cxx10.output, atol=1.0e-5)
    timing["live_observed_compare_s"] = time.perf_counter() - cmp_t0
    fallback_reason = cxx10.fallback_reason
    if not bool(live_parity.get("ok", False)):
        fallback_reason = "live_observed_divergence"
        if config.cxxcore11_live_runner_hook_fail_closed and not config.cxxcore11_live_runner_hook_restore_on_divergence:
            raise RuntimeError(f"CXXCORE-11 live observed divergence: {live_parity.get('first_divergence')!r}")
    fallback = bool(fallback_reason)
    retained = bool(cxx10.retained_native_output and not fallback and mode.startswith("native_guarded"))
    output = cxx10.output if retained else observed_python
    timing["total_wall_s"] = time.perf_counter() - started
    return _result(
        mode=mode,
        status="fallback" if fallback else "live_hook",
        output=output,
        retained_native_output=retained,
        apply_output_to_live_scalars=retained,
        parity=cxx10.parity,
        live_observed_parity=live_parity,
        fallback_reason=fallback_reason,
        python_state_restored=bool(fallback),
        native_available=bool(cxx10.native_available),
        timing=timing,
        diagnostics=diagnostics | {"cxxcore10": cxx10_payload, "output_digest": state_digest(output)},
    )


def result_payload(result: Cxxcore11LiveHookResult) -> dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "mode": result.mode,
        "status": result.status,
        "retained_native_output": bool(result.retained_native_output),
        "apply_output_to_live_scalars": bool(result.apply_output_to_live_scalars),
        "parity": result.parity,
        "live_observed_parity": result.live_observed_parity,
        "fallback_reason": result.fallback_reason,
        "first_divergence": result.first_divergence,
        "python_state_restored": bool(result.python_state_restored),
        "native_available": bool(result.native_available),
        "timing": dict(result.timing),
        "diagnostics": dict(result.diagnostics),
        "state_digest": state_digest(result.output),
    }


def _to_cxxcore10_mode(mode: str) -> str:
    if mode == "python_split_reference_live_hook":
        return "python_split_reference"
    if mode == "native_shadow_live_hook":
        return "native_shadow_transaction"
    if mode == "native_guarded_no_rng_live_hook":
        return "native_guarded_transaction_no_rng"
    if mode == "native_guarded_validation_rng_live_hook":
        return "native_guarded_transaction_validation_rng"
    raise ValueError(f"mode has no CXXCORE-10 equivalent: {mode}")


def _empty_timing() -> dict[str, float]:
    return {
        "total_wall_s": 0.0,
        "input_capture_s": 0.0,
        "cache_layout_refresh_s": 0.0,
        "python_direct_reference_s": 0.0,
        "python_split_materialize_s": 0.0,
        "native_backend_s": 0.0,
        "python_writeback_s": 0.0,
        "comparison_s": 0.0,
        "live_observed_compare_s": 0.0,
        "fallback_restore_s": 0.0,
        "rng_validation_s": 0.0,
        "diagnostics_s": 0.0,
    }


def _merge_cxxcore10_timing(timing: dict[str, float], source: dict[str, Any]) -> None:
    for key in (
        "input_capture_s",
        "cache_layout_refresh_s",
        "python_direct_reference_s",
        "python_split_materialize_s",
        "native_backend_s",
        "python_writeback_s",
        "comparison_s",
        "fallback_restore_s",
        "rng_validation_s",
        "diagnostics_s",
    ):
        timing[key] = float(source.get(key, 0.0) or 0.0)


def _fallback_result(
    *,
    mode: str,
    observed_python: dict[str, np.ndarray],
    timing: dict[str, float],
    started: float,
    diagnostics: dict[str, Any],
    reason: str,
    native_available: bool,
) -> Cxxcore11LiveHookResult:
    timing["fallback_restore_s"] = time.perf_counter() - started
    timing["total_wall_s"] = time.perf_counter() - started
    return _result(
        mode=mode,
        status="fallback",
        output=observed_python,
        retained_native_output=False,
        apply_output_to_live_scalars=False,
        parity=None,
        live_observed_parity=None,
        fallback_reason=reason,
        python_state_restored=True,
        native_available=native_available,
        timing=timing,
        diagnostics=diagnostics | {"output_digest": state_digest(observed_python)},
    )


def _result(
    *,
    mode: str,
    status: str,
    output: dict[str, np.ndarray] | None,
    retained_native_output: bool,
    apply_output_to_live_scalars: bool,
    parity: dict[str, Any] | None,
    live_observed_parity: dict[str, Any] | None,
    fallback_reason: str | None,
    python_state_restored: bool,
    native_available: bool,
    timing: dict[str, float],
    diagnostics: dict[str, Any],
) -> Cxxcore11LiveHookResult:
    first = None
    if live_observed_parity and live_observed_parity.get("first_divergence") is not None:
        first = live_observed_parity.get("first_divergence")
    elif parity and parity.get("first_divergence") is not None:
        first = parity.get("first_divergence")
    return Cxxcore11LiveHookResult(
        schema_version=SCHEMA_VERSION,
        mode=mode,
        status=status,
        output=output,
        retained_native_output=retained_native_output,
        apply_output_to_live_scalars=apply_output_to_live_scalars,
        parity=parity,
        live_observed_parity=live_observed_parity,
        fallback_reason=fallback_reason,
        first_divergence=first,
        python_state_restored=python_state_restored,
        native_available=native_available,
        timing=timing,
        diagnostics=diagnostics,
    )


__all__ = [
    "LIVE_HOOK_MODES",
    "PARITY_POLICIES",
    "SCHEMA_VERSION",
    "Cxxcore11LiveHookConfig",
    "Cxxcore11LiveHookResult",
    "config_from_sim_cfg",
    "live_hook_preflight_audit",
    "observed_materialized_from_live_python",
    "prepare_live_tick_inputs",
    "result_payload",
    "run_live_runner_transaction_hook",
    "validate_config",
    "validation_rng_payload_from_actions",
]
