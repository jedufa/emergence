from __future__ import annotations

"""CXXCORE-12 live tick-island ownership audit.

This module is deliberately default-off. CXXCORE-12 decides whether the
CXXCORE-10/11 copied-state transaction can become a live authoritative
`GridWorld.step_agents_batch` replacement. The current answer is no: a Python
state-backend ownership layer is required before safely skipping the live Python
env step.
"""

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any

import numpy as np

from perf.cxxcore5_tick_island_layout_cache import (
    TickIslandLayoutCacheInputs,
    compare_runner_backend_states,
    run_tick_island_python_reference,
)
from perf.cxxcore7_rng_precompute import Cxxcore7RngPayload
from perf.cxxcore8_guarded_runner_bridge import state_digest
from perf.cxxcore11_live_runner_transaction_hook import (
    Cxxcore11LiveHookConfig,
    result_payload as cxxcore11_result_payload,
    run_live_runner_transaction_hook,
)


SCHEMA_VERSION = "cxxcore12_live_tick_island_ownership_v1"
AUDIT_SCHEMA_VERSION = "cxxcore12_ownership_audit_v1"
STATE_BACKEND_SCHEMA_VERSION = "cxxcore12_state_backend_requirement_v1"
DECISION = "CXXCORE12-B-STATE-BACKEND-REQUIRED"
MODES = (
    "off",
    "python_split_reference",
    "native_shadow_reference",
    "native_authoritative_no_rng_live_tick_island",
    "native_authoritative_validation_rng_live_tick_island",
)
PARITY_POLICIES = ("final_and_sparse", "every_tick", "interval")
RNG_POLICIES = ("none", "validation_action_override")


@dataclass(frozen=True)
class Cxxcore12LiveTickIslandConfig:
    cxxcore12_live_tick_island_enable: bool = False
    cxxcore12_live_tick_island_mode: str = "off"
    cxxcore12_live_tick_island_parity_policy: str = "final_and_sparse"
    cxxcore12_live_tick_island_parity_interval: int = 1
    cxxcore12_live_tick_island_restore_on_divergence: bool = True
    cxxcore12_live_tick_island_fail_closed: bool = True
    cxxcore12_live_tick_island_strict_native_required: bool = False
    cxxcore12_live_tick_island_rng_policy: str = "none"
    cxxcore12_live_tick_island_emit_diagnostics: bool = True


@dataclass(frozen=True)
class Cxxcore12ProbeResult:
    schema_version: str
    mode: str
    status: str
    decision: str
    output: dict[str, np.ndarray] | None
    parity: dict[str, Any] | None
    fallback_reason: str | None
    first_divergence: dict[str, Any] | None
    native_available: bool
    native_output_retained: bool
    python_env_step_skipped: bool
    python_state_restored: bool
    timing: dict[str, float]
    diagnostics: dict[str, Any]


def config_from_sim_cfg(sim_cfg: Any) -> Cxxcore12LiveTickIslandConfig:
    return Cxxcore12LiveTickIslandConfig(
        cxxcore12_live_tick_island_enable=bool(getattr(sim_cfg, "cxxcore12_live_tick_island_enable", False)),
        cxxcore12_live_tick_island_mode=str(getattr(sim_cfg, "cxxcore12_live_tick_island_mode", "off") or "off"),
        cxxcore12_live_tick_island_parity_policy=str(
            getattr(sim_cfg, "cxxcore12_live_tick_island_parity_policy", "final_and_sparse") or "final_and_sparse"
        ),
        cxxcore12_live_tick_island_parity_interval=int(
            getattr(sim_cfg, "cxxcore12_live_tick_island_parity_interval", 1) or 1
        ),
        cxxcore12_live_tick_island_restore_on_divergence=bool(
            getattr(sim_cfg, "cxxcore12_live_tick_island_restore_on_divergence", True)
        ),
        cxxcore12_live_tick_island_fail_closed=bool(getattr(sim_cfg, "cxxcore12_live_tick_island_fail_closed", True)),
        cxxcore12_live_tick_island_strict_native_required=bool(
            getattr(sim_cfg, "cxxcore12_live_tick_island_strict_native_required", False)
        ),
        cxxcore12_live_tick_island_rng_policy=str(
            getattr(sim_cfg, "cxxcore12_live_tick_island_rng_policy", "none") or "none"
        ),
        cxxcore12_live_tick_island_emit_diagnostics=bool(
            getattr(sim_cfg, "cxxcore12_live_tick_island_emit_diagnostics", True)
        ),
    )


def validate_config(config: Cxxcore12LiveTickIslandConfig) -> None:
    if config.cxxcore12_live_tick_island_mode not in MODES:
        raise ValueError(f"unsupported CXXCORE-12 live tick-island mode: {config.cxxcore12_live_tick_island_mode}")
    if config.cxxcore12_live_tick_island_parity_policy not in PARITY_POLICIES:
        raise ValueError(f"unsupported CXXCORE-12 parity policy: {config.cxxcore12_live_tick_island_parity_policy}")
    if int(config.cxxcore12_live_tick_island_parity_interval) < 1:
        raise ValueError("cxxcore12_live_tick_island_parity_interval must be >= 1")
    if config.cxxcore12_live_tick_island_rng_policy not in RNG_POLICIES:
        raise ValueError(f"unsupported CXXCORE-12 RNG policy: {config.cxxcore12_live_tick_island_rng_policy}")


def ownership_audit_rows() -> list[dict[str, Any]]:
    rows = [
        _row("agent_id_ordering", "Agent/object state", "native_transaction_owned_now", "CXXCORE-10/11 copied arrays preserve sorted/runner ordering for materialized compute.", True),
        _row("action_stream", "Agent/object state", "native_transaction_owned_now", "Copied action matrix can feed native no-RNG and validation-RNG modes.", True),
        _row("energy_health_injury_scar", "Agent/object state", "python_writeback_derived_now", "Native can materialize scalar outputs, but live Agent objects are still written by Python.", True),
        _row("alive_dead_state", "Agent/object state", "python_writeback_derived_now", "Death mask is materialized; Agent.alive transition and death bookkeeping remain Python-owned.", True),
        _row("food_gained", "Agent/object state", "requires_state_backend_before_live_ownership", "CXXCORE-10 materialized fields do not include food_gained, but runner stats and early-life metrics consume it.", True),
        _row("Agent.stats_delta_survival_energy", "Agent/object state", "python_authoritative_retained", "Stats are updated after env step from Python Agent objects and materialized arrays.", True),
        _row("agent_position_arrays", "Grid/world state", "python_writeback_derived_now", "Native returns new positions, but live env.agent_pos_y/x mutation is Python-owned.", True),
        _row("agent_grid_occupancy", "Grid/world state", "requires_state_backend_before_live_ownership", "CXXCORE-10 output lacks a live agent_grid result; reconstructing it after collisions is a state-backend responsibility.", True),
        _row("resource_grid", "Grid/world state", "python_writeback_derived_now", "Native returns grid, but live self.grid mutation and view-dirty semantics are Python-owned.", True),
        _row("views_dirty_flag", "Grid/world state", "requires_state_backend_before_live_ownership", "Python env sets _views_dirty from movement/gather side effects; native contract lacks this side-effect flag.", True),
        _row("patch_food_amount_and_emissions", "Grid/world state", "requires_state_backend_before_live_ownership", "Patch World mutates amount/scent/waste arrays and last-step counters inside step_agents_batch.", True),
        _row("patch_movement_context", "Grid/world state", "requires_state_backend_before_live_ownership", "Patch movement cost arrays and queued movement emissions are live env side effects.", True),
        _row("CostChargeSurface_layout_rows", "Cost/cache state", "native_transaction_owned_now", "Rows are copied/rebuildable and already feed CXXCORE-10 native materialization.", True),
        _row("CostLayoutCache_invalidation", "Cost/cache state", "python_authoritative_retained", "Cache hit/update lifecycle depends on Agent/genome/config Python objects.", True),
        _row("cost_total_by_agent", "Cost/cache state", "native_transaction_owned_now", "Materialized and compared by CXXCORE-10/11.", True),
        _row("cost_charge_diagnostics", "Diagnostics/metrics inputs", "python_authoritative_retained", "Diagnostic recorders and SoA summaries are Python-owned.", False),
        _row("validation_random_walk_rng", "RNG and stochastic surfaces", "native_transaction_owned_now", "Only safe through Python-precomputed action override arrays.", False),
        _row("biome_hazard_rng", "RNG and stochastic surfaces", "requires_rng_preflight_before_live_ownership", "GridWorld draws env.rng.random inside step_agents_batch when biomes are enabled.", True),
        _row("tier3_transition_rng", "RNG and stochastic surfaces", "requires_rng_preflight_before_live_ownership", "Tier3 crossing events draw randoms inside step_agents_batch.", True),
        _row("tier3_cc_neff_state", "Diagnostics/metrics inputs", "requires_state_backend_before_live_ownership", "Tier3 carrying-capacity EMA mutates _tier3_cc_Neff and last-step stats.", True),
        _row("tier3_last_step_stats", "Diagnostics/metrics inputs", "requires_state_backend_before_live_ownership", "Runner generation counters consume env.tier3_last_step_stats after env step.", True),
        _row("tier4_exposure_and_death_bins", "Diagnostics/metrics inputs", "python_authoritative_retained", "Runner computes age/exposure/death bins from Agent.stats and died_mask.", False),
        _row("env_apply_expression_diagnostics", "Diagnostics/metrics inputs", "python_authoritative_retained", "Hotpath diagnostics consume Python Agent/cognition calls.", False),
        _row("metrics_logger_inputs", "Diagnostics/metrics inputs", "requires_checkpoint_metrics_preflight_before_live_ownership", "Metrics read Python-owned counters and Agent.stats; schema unchanged in CXXCORE-12.", False),
        _row("checkpoint_payload_state", "Checkpoint/replay implications", "requires_checkpoint_metrics_preflight_before_live_ownership", "Any native-owned live state would need checkpoint/replay rules; CXXCORE-12 adds none.", True),
        _row("rng_stream_state", "Checkpoint/replay implications", "requires_rng_preflight_before_live_ownership", "Skipping Python env-step must not silently skip or advance Python RNG streams.", True),
        _row("selection_reproduction_lineage", "Checkpoint/replay implications", "out_of_scope_for_cxxcore12", "Not part of env tick-island ownership pilot.", False),
    ]
    return rows


def ownership_audit_payload() -> dict[str, Any]:
    rows = ownership_audit_rows()
    blockers = [row for row in rows if str(row["classification"]).startswith("requires_")]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "decision": DECISION,
        "path": "Path B - design-only state-backend requirement",
        "python_env_step_skip_safe_now": False,
        "native_authoritative_live_tick_island_supported_now": False,
        "primary_blocker": "CXXCORE-10/11 materialized contract does not own enough live GridWorld/runner side-effect state to skip GridWorld.step_agents_batch.",
        "rows": rows,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "required_contract": state_backend_requirement_payload(),
    }


def state_backend_requirement_payload() -> dict[str, Any]:
    return {
        "schema_version": STATE_BACKEND_SCHEMA_VERSION,
        "decision": DECISION,
        "required_before_live_authoritative_skip": True,
        "minimum_fields": [
            "agent_ids",
            "actions_by_tick",
            "energy",
            "health",
            "injury",
            "scar",
            "alive_mask",
            "died_mask",
            "food_gained",
            "agent_pos_y",
            "agent_pos_x",
            "agent_grid",
            "resource_grid",
            "views_dirty",
            "moved_mask",
            "cost_total_by_agent",
            "cost_total",
        ],
        "optional_but_required_when_feature_enabled": [
            "patch_food_amount",
            "patch_scent_emit_pending",
            "patch_waste_emit_pending",
            "patch_food_harvest_total_last",
            "patch_movement_energy_extra_last",
            "patch_movement_energy_event_count_last",
            "tier3_cc_Neff",
            "tier3_last_step_stats",
            "tier3_crossing_mask",
            "tier3_region_buffers",
            "hazard_rng_uniforms",
            "tier3_transition_rng_uniforms",
        ],
        "writeback_boundary": [
            "validate native output shapes/dtypes",
            "compare against separate reference run or saved checkpoint slice",
            "write env.agent_pos_y/x, env.agent_grid, env.grid/resource arrays atomically",
            "write Agent scalar fields through Python boundary",
            "write Agent.stats deltas, survival, and food-gained counters through Python boundary",
            "write Tier3/Tier4/Patch counters or explicitly prove disabled",
            "leave checkpoint and metrics schemas unchanged until a preflight owns them",
        ],
        "restore_token_fields": [
            "pre env.agent_pos_y/x",
            "pre env.agent_grid",
            "pre env.grid/resource arrays",
            "pre patch arrays/counters when enabled",
            "pre Tier3 mutable buffers/counters when enabled",
            "pre Agent scalar/stat values for alive batch",
            "pre RNG stream state when stochastic surfaces are included",
        ],
        "reason_cxxcore10_contract_insufficient": [
            "missing food_gained output for stats/metrics writeback",
            "missing agent_grid output or validated reconstruction contract",
            "missing patch-world mutated arrays and last-step counters",
            "missing Tier3 mutable state and stochastic draw precompute",
            "no checkpoint/replay rule for native-owned live state",
        ],
    }


def feasibility_payload() -> dict[str, Any]:
    return {
        "schema_version": "cxxcore12_feasibility_decision_v1",
        "decision": DECISION,
        "path_a_conditions": {
            "selected_slice_represented_by_cxxcore10_contract": False,
            "all_live_python_side_effects_materializable": False,
            "python_gridworld_step_can_be_skipped": False,
            "no_checkpoint_schema_required": True,
            "no_metrics_schema_change_required": True,
            "no_independent_cpp_rng_required": True,
            "fallback_can_restore_before_mutation": False,
        },
        "path_b_required": True,
        "next_step": "CXXCORE13-PYTHON-STATE-BACKEND-OWNERSHIP-LAYER",
    }


def run_cxxcore12_probe(
    inputs: TickIslandLayoutCacheInputs,
    *,
    config: Cxxcore12LiveTickIslandConfig,
    native_backend: Any | None = None,
    rng_payload: Cxxcore7RngPayload | None = None,
    inject_divergence: bool = False,
) -> Cxxcore12ProbeResult:
    validate_config(config)
    started = time.perf_counter()
    timing = _empty_timing()
    mode = str(config.cxxcore12_live_tick_island_mode)
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION,
        "mode": mode,
        "python_env_step_skipped": False,
        "metrics_schema_changed": False,
        "checkpoint_schema_changed": False,
        "native_rng_added": False,
        "state_backend_required": True,
        "ownership_audit_digest": _stable_json_digest(ownership_audit_rows()),
    }
    if not bool(config.cxxcore12_live_tick_island_enable) or mode == "off":
        timing["total_wall_s"] = time.perf_counter() - started
        return _result(
            mode=mode,
            status="disabled",
            output=None,
            parity=None,
            fallback_reason="pilot_disabled",
            native_available=bool(native_backend is not None),
            native_output_retained=False,
            python_env_step_skipped=False,
            python_state_restored=False,
            timing=timing,
            diagnostics=diagnostics,
        )

    if mode in {"native_authoritative_no_rng_live_tick_island", "native_authoritative_validation_rng_live_tick_island"}:
        timing["total_wall_s"] = time.perf_counter() - started
        return _result(
            mode=mode,
            status="blocked",
            output=None,
            parity=None,
            fallback_reason="state_backend_required",
            native_available=bool(native_backend is not None),
            native_output_retained=False,
            python_env_step_skipped=False,
            python_state_restored=False,
            timing=timing,
            diagnostics=diagnostics | {
                "blocked_by": state_backend_requirement_payload()["reason_cxxcore10_contract_insufficient"],
                "authoritative_arm_not_timed": True,
            },
        )

    observed_t0 = time.perf_counter()
    observed = run_tick_island_python_reference(inputs)
    timing["python_env_step_s"] = time.perf_counter() - observed_t0
    cxx11_mode = "python_split_reference_live_hook" if mode == "python_split_reference" else "native_shadow_live_hook"
    cxx11_cfg = Cxxcore11LiveHookConfig(
        cxxcore11_live_runner_hook_enable=True,
        cxxcore11_live_runner_hook_mode=cxx11_mode,
        cxxcore11_live_runner_hook_parity_interval=int(config.cxxcore12_live_tick_island_parity_interval),
        cxxcore11_live_runner_hook_restore_on_divergence=bool(config.cxxcore12_live_tick_island_restore_on_divergence),
        cxxcore11_live_runner_hook_fail_closed=bool(config.cxxcore12_live_tick_island_fail_closed),
        cxxcore11_live_runner_hook_emit_diagnostics=bool(config.cxxcore12_live_tick_island_emit_diagnostics),
        cxxcore11_live_runner_hook_strict_native_required=bool(config.cxxcore12_live_tick_island_strict_native_required),
        cxxcore11_live_runner_hook_parity_policy=str(config.cxxcore12_live_tick_island_parity_policy),
        cxxcore11_live_runner_hook_rng_policy=str(config.cxxcore12_live_tick_island_rng_policy),
    )
    try:
        cxx11 = run_live_runner_transaction_hook(
            inputs,
            observed_python=observed,
            config=cxx11_cfg,
            native_backend=native_backend,
            rng_payload=rng_payload,
            inject_divergence=inject_divergence,
            extra_diagnostics={"cxxcore12_probe": True},
        )
    except Exception as exc:
        if config.cxxcore12_live_tick_island_fail_closed:
            raise
        timing["fallback_overhead_s"] = time.perf_counter() - started
        timing["total_wall_s"] = time.perf_counter() - started
        return _result(
            mode=mode,
            status="fallback",
            output=observed,
            parity=None,
            fallback_reason="exception_fallback",
            native_available=bool(native_backend is not None),
            native_output_retained=False,
            python_env_step_skipped=False,
            python_state_restored=True,
            timing=timing,
            diagnostics=diagnostics | {"exception_type": type(exc).__name__, "exception": str(exc)},
        )
    cxx11_payload = cxxcore11_result_payload(cxx11)
    _merge_cxx11_timing(timing, cxx11_payload.get("timing") or {})
    parity = cxx11.parity if cxx11.fallback_reason else cxx11.live_observed_parity
    if cxx11.output is not None and not cxx11.fallback_reason:
        verify_t0 = time.perf_counter()
        parity = compare_runner_backend_states(observed, cxx11.output, atol=1.0e-5)
        timing["parity_s"] += time.perf_counter() - verify_t0
    timing["total_wall_s"] = time.perf_counter() - started
    return _result(
        mode=mode,
        status="reference_parity" if not cxx11.fallback_reason else "fallback",
        output=cxx11.output,
        parity=parity,
        fallback_reason=cxx11.fallback_reason,
        native_available=bool(cxx11.native_available),
        native_output_retained=bool(cxx11.retained_native_output),
        python_env_step_skipped=False,
        python_state_restored=bool(cxx11.python_state_restored),
        timing=timing,
        diagnostics=diagnostics | {"cxxcore11": cxx11_payload},
    )


def result_payload(result: Cxxcore12ProbeResult) -> dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "mode": result.mode,
        "status": result.status,
        "decision": result.decision,
        "parity": result.parity,
        "fallback_reason": result.fallback_reason,
        "first_divergence": result.first_divergence,
        "native_available": bool(result.native_available),
        "native_output_retained": bool(result.native_output_retained),
        "python_env_step_skipped": bool(result.python_env_step_skipped),
        "python_state_restored": bool(result.python_state_restored),
        "timing": dict(result.timing),
        "diagnostics": dict(result.diagnostics),
        "state_digest": state_digest(result.output),
    }


def _row(
    name: str,
    surface: str,
    classification: str,
    notes: str,
    required_for_skip: bool,
) -> dict[str, Any]:
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
        "python_action_prep_s": 0.0,
        "python_env_step_s": 0.0,
        "native_compute_s": 0.0,
        "writeback_s": 0.0,
        "cost_layout_cache_s": 0.0,
        "parity_s": 0.0,
        "diagnostics_s": 0.0,
        "fallback_overhead_s": 0.0,
    }


def _merge_cxx11_timing(timing: dict[str, float], source: dict[str, Any]) -> None:
    timing["native_compute_s"] = float(source.get("native_backend_s", 0.0) or 0.0)
    timing["writeback_s"] = float(source.get("python_writeback_s", 0.0) or 0.0)
    timing["cost_layout_cache_s"] = float(source.get("cache_layout_refresh_s", 0.0) or 0.0)
    timing["parity_s"] = float(source.get("comparison_s", 0.0) or 0.0) + float(
        source.get("live_observed_compare_s", 0.0) or 0.0
    )
    timing["diagnostics_s"] = float(source.get("diagnostics_s", 0.0) or 0.0)
    timing["fallback_overhead_s"] = float(source.get("fallback_restore_s", 0.0) or 0.0)


def _stable_json_digest(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _result(
    *,
    mode: str,
    status: str,
    output: dict[str, np.ndarray] | None,
    parity: dict[str, Any] | None,
    fallback_reason: str | None,
    native_available: bool,
    native_output_retained: bool,
    python_env_step_skipped: bool,
    python_state_restored: bool,
    timing: dict[str, float],
    diagnostics: dict[str, Any],
) -> Cxxcore12ProbeResult:
    return Cxxcore12ProbeResult(
        schema_version=SCHEMA_VERSION,
        mode=mode,
        status=status,
        decision=DECISION,
        output=output,
        parity=parity,
        fallback_reason=fallback_reason,
        first_divergence=(parity or {}).get("first_divergence") if parity else None,
        native_available=native_available,
        native_output_retained=native_output_retained,
        python_env_step_skipped=python_env_step_skipped,
        python_state_restored=python_state_restored,
        timing=timing,
        diagnostics=diagnostics,
    )


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "DECISION",
    "MODES",
    "PARITY_POLICIES",
    "RNG_POLICIES",
    "SCHEMA_VERSION",
    "STATE_BACKEND_SCHEMA_VERSION",
    "Cxxcore12LiveTickIslandConfig",
    "Cxxcore12ProbeResult",
    "config_from_sim_cfg",
    "feasibility_payload",
    "ownership_audit_payload",
    "ownership_audit_rows",
    "result_payload",
    "run_cxxcore12_probe",
    "state_backend_requirement_payload",
    "validate_config",
]
