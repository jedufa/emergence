from __future__ import annotations

"""CXXCORE-8 guarded live-runner bridge pilot helpers.

Diagnostic/default-off only. The bridge operates on copied runner/tick-island
state, compares against a Python reference from the same pre-state, and writes
back materialized C++ outputs only into the controlled bridge state after parity.
Normal simulation remains Python-authoritative unless an explicit probe calls
this helper.
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
    compare_summaries,
    run_tick_island_python_reference,
    state_summary,
)
from perf.cxxcore6_authoritative_pilot import _authoritative_materialization, _enrich_divergence_report
from perf.cxxcore7_rng_precompute import Cxxcore7RngPayload, apply_validation_action_override, validate_rng_payload


SCHEMA_VERSION = "cxxcore8_guarded_runner_bridge_v1"
BRIDGE_MODES = ("shadow", "authoritative_no_rng", "authoritative_precomputed_rng")


@dataclass(frozen=True)
class Cxxcore8BridgeConfig:
    cxxcore8_guarded_runner_bridge_enable: bool = False
    cxxcore8_guarded_runner_bridge_mode: str = "shadow"
    cxxcore8_runner_bridge_compare_interval_ticks: int = 0
    cxxcore8_runner_bridge_force_final_compare: bool = True
    cxxcore8_runner_bridge_fallback_on_mismatch: bool = True
    cxxcore8_runner_bridge_emit_compact_summary: bool = True
    cxxcore8_runner_bridge_allow_validation_rng_precompute: bool = False
    cxxcore8_runner_bridge_allow_biome_hazard_precompute: bool = False
    cxxcore8_runner_bridge_allow_tier3_precompute: bool = False


@dataclass(frozen=True)
class GuardedBridgeResult:
    schema_version: str
    mode: str
    status: str
    output: dict[str, np.ndarray] | None
    parity: dict[str, Any] | None
    sparse_parity: dict[str, Any] | None
    fallback_occurred: bool
    fallback_reason: str
    cxx_result_retained: bool
    python_state_restored: bool
    writeback_applied: bool
    native_available: bool
    timing: dict[str, float]
    diagnostics: dict[str, Any]


def config_from_sim_cfg(sim_cfg: Any) -> Cxxcore8BridgeConfig:
    return Cxxcore8BridgeConfig(
        cxxcore8_guarded_runner_bridge_enable=bool(getattr(sim_cfg, "cxxcore8_guarded_runner_bridge_enable", False)),
        cxxcore8_guarded_runner_bridge_mode=str(getattr(sim_cfg, "cxxcore8_guarded_runner_bridge_mode", "shadow") or "shadow"),
        cxxcore8_runner_bridge_compare_interval_ticks=int(
            getattr(sim_cfg, "cxxcore8_runner_bridge_compare_interval_ticks", 0) or 0
        ),
        cxxcore8_runner_bridge_force_final_compare=bool(
            getattr(sim_cfg, "cxxcore8_runner_bridge_force_final_compare", True)
        ),
        cxxcore8_runner_bridge_fallback_on_mismatch=bool(
            getattr(sim_cfg, "cxxcore8_runner_bridge_fallback_on_mismatch", True)
        ),
        cxxcore8_runner_bridge_emit_compact_summary=bool(
            getattr(sim_cfg, "cxxcore8_runner_bridge_emit_compact_summary", True)
        ),
        cxxcore8_runner_bridge_allow_validation_rng_precompute=bool(
            getattr(sim_cfg, "cxxcore8_runner_bridge_allow_validation_rng_precompute", False)
        ),
        cxxcore8_runner_bridge_allow_biome_hazard_precompute=bool(
            getattr(sim_cfg, "cxxcore8_runner_bridge_allow_biome_hazard_precompute", False)
        ),
        cxxcore8_runner_bridge_allow_tier3_precompute=bool(
            getattr(sim_cfg, "cxxcore8_runner_bridge_allow_tier3_precompute", False)
        ),
    )


def run_guarded_runner_bridge(
    inputs: TickIslandLayoutCacheInputs,
    *,
    config: Cxxcore8BridgeConfig,
    native_backend: Any | None,
    rng_payload: Cxxcore7RngPayload | None = None,
    inject_divergence: bool = False,
) -> GuardedBridgeResult:
    mode = str(config.cxxcore8_guarded_runner_bridge_mode or "shadow")
    if mode not in BRIDGE_MODES:
        raise ValueError(f"unsupported CXXCORE-8 bridge mode: {mode}")
    timing = {
        "total_wall_s": 0.0,
        "snapshot_copy_s": 0.0,
        "python_reference_s": 0.0,
        "rng_precompute_s": 0.0,
        "native_compute_s": 0.0,
        "materialization_s": 0.0,
        "sparse_compare_s": 0.0,
        "final_compare_s": 0.0,
        "writeback_s": 0.0,
        "fallback_s": 0.0,
        "diagnostics_s": 0.0,
    }
    state = inputs.backend.runner.state
    diagnostics = {
        "mode": mode,
        "agents": int(state.agent_ids.shape[0]),
        "ticks": int(state.actions_by_tick.shape[0]),
        "actions": int(state.agent_ids.shape[0]) * int(state.actions_by_tick.shape[0]),
        "cells": int(state.config.height) * int(state.config.width),
        "enabled_surfaces": list(rng_payload.enabled_surfaces if rng_payload is not None else ()),
        "checkpoint_boundary": "no checkpoint-required state added; bridge state is copied/rebuildable",
        "metrics_boundary": "no metrics definition changes; compact summaries are diagnostics only",
        "cpp_rng_boundary": "no independent C++ RNG",
    }
    total_t0 = time.perf_counter()
    if not bool(config.cxxcore8_guarded_runner_bridge_enable):
        timing["total_wall_s"] = time.perf_counter() - total_t0
        return GuardedBridgeResult(
            schema_version=SCHEMA_VERSION,
            mode=mode,
            status="disabled",
            output=None,
            parity=None,
            sparse_parity=None,
            fallback_occurred=False,
            fallback_reason="bridge_disabled",
            cxx_result_retained=False,
            python_state_restored=False,
            writeback_applied=False,
            native_available=bool(native_backend is not None),
            timing=timing,
            diagnostics=diagnostics,
        )
    snapshot_t0 = time.perf_counter()
    ref_inputs = inputs
    if mode == "authoritative_precomputed_rng":
        if not bool(config.cxxcore8_runner_bridge_allow_validation_rng_precompute):
            return _fallback_result(
                inputs,
                mode=mode,
                timing=timing,
                diagnostics=diagnostics,
                reason="validation_rng_precompute_not_allowed",
                native_available=bool(native_backend is not None),
                started=total_t0,
            )
        if rng_payload is None:
            return _fallback_result(
                inputs,
                mode=mode,
                timing=timing,
                diagnostics=diagnostics,
                reason="rng_payload_missing",
                native_available=bool(native_backend is not None),
                started=total_t0,
            )
        validation = validate_rng_payload(
            rng_payload,
            ticks=int(state.actions_by_tick.shape[0]),
            agents=int(state.actions_by_tick.shape[1]),
        )
        diagnostics["rng_validation"] = validation
        if not validation["ok"]:
            return _fallback_result(
                inputs,
                mode=mode,
                timing=timing,
                diagnostics=diagnostics,
                reason="rng_payload_invalid",
                native_available=bool(native_backend is not None),
                started=total_t0,
            )
        ref_inputs = apply_validation_action_override(inputs, rng_payload)
    elif mode == "authoritative_no_rng" and rng_payload is not None:
        diagnostics["ignored_rng_payload"] = True
    elif mode == "shadow" and rng_payload is not None:
        diagnostics["shadow_rng_payload_present"] = True
        ref_inputs = apply_validation_action_override(inputs, rng_payload)
    timing["snapshot_copy_s"] = time.perf_counter() - snapshot_t0

    reference_t0 = time.perf_counter()
    python_reference = run_tick_island_python_reference(ref_inputs)
    timing["python_reference_s"] = time.perf_counter() - reference_t0
    if native_backend is None:
        return _fallback_result(
            ref_inputs,
            mode=mode,
            timing=timing,
            diagnostics=diagnostics,
            reason="native_unavailable",
            native_available=False,
            started=total_t0,
            python_reference=python_reference,
        )

    native_t0 = time.perf_counter()
    actual = native_backend(inputs, rng_payload=rng_payload if mode in {"authoritative_precomputed_rng", "shadow"} else None, export_final=True)
    timing["native_compute_s"] = time.perf_counter() - native_t0
    materialize_t0 = time.perf_counter()
    retained = _authoritative_materialization(actual)
    timing["materialization_s"] = time.perf_counter() - materialize_t0
    if inject_divergence and "new_energy" in retained:
        retained = dict(retained)
        retained["new_energy"] = np.asarray(retained["new_energy"], dtype=np.float32).copy()
        if retained["new_energy"].size:
            retained["new_energy"][0] = np.float32(float(retained["new_energy"][0]) + 1.0)

    sparse_t0 = time.perf_counter()
    sparse = compare_summaries(state_summary(python_reference), actual["native_summary"], atol=1.0e-4)
    timing["sparse_compare_s"] = time.perf_counter() - sparse_t0
    final_t0 = time.perf_counter()
    parity = compare_runner_backend_states(python_reference, retained, atol=1.0e-5)
    timing["final_compare_s"] = time.perf_counter() - final_t0
    parity = _enrich_divergence_report(parity, fallback_occurred=False, fallback_reason="") or parity
    if not bool(parity.get("ok")):
        if bool(config.cxxcore8_runner_bridge_fallback_on_mismatch):
            timing["total_wall_s"] = time.perf_counter() - total_t0
            enriched = _enrich_divergence_report(parity, fallback_occurred=True, fallback_reason="divergence") or parity
            return GuardedBridgeResult(
                schema_version=SCHEMA_VERSION,
                mode=mode,
                status="fallback",
                output=python_reference,
                parity=enriched,
                sparse_parity=sparse,
                fallback_occurred=True,
                fallback_reason="divergence",
                cxx_result_retained=False,
                python_state_restored=True,
                writeback_applied=False,
                native_available=True,
                timing=timing,
                diagnostics=diagnostics | {"state_digest": state_digest(python_reference)},
            )
    write_t0 = time.perf_counter()
    output = python_reference if mode == "shadow" else retained
    timing["writeback_s"] = time.perf_counter() - write_t0
    timing["total_wall_s"] = time.perf_counter() - total_t0
    return GuardedBridgeResult(
        schema_version=SCHEMA_VERSION,
        mode=mode,
        status="shadow" if mode == "shadow" else "authoritative",
        output=output,
        parity=parity,
        sparse_parity=sparse,
        fallback_occurred=False,
        fallback_reason="",
        cxx_result_retained=mode != "shadow",
        python_state_restored=False,
        writeback_applied=mode != "shadow",
        native_available=True,
        timing=timing,
        diagnostics=diagnostics | {"state_digest": state_digest(output)},
    )


def result_payload(result: GuardedBridgeResult) -> dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "mode": result.mode,
        "status": result.status,
        "native_available": bool(result.native_available),
        "fallback_occurred": bool(result.fallback_occurred),
        "fallback_reason": result.fallback_reason,
        "cxx_result_retained": bool(result.cxx_result_retained),
        "python_state_restored": bool(result.python_state_restored),
        "writeback_applied": bool(result.writeback_applied),
        "parity": result.parity,
        "sparse_parity": result.sparse_parity,
        "timing": dict(result.timing),
        "diagnostics": dict(result.diagnostics),
    }


def bridge_audit() -> dict[str, Any]:
    return {
        "schema_version": "cxxcore8_guarded_runner_bridge_audit_v1",
        "insertion_point": "TickEnvApplyMixin._apply_env_tick after action override and cost multiplier materialization, before env.step_agents_batch writeback",
        "pilot_strategy": "copy pre-state, run Python reference from copy, run C++ materialization from same copy, compare, then write back only controlled bridge output",
        "python_state_mutated_by_slice": [
            "GridWorld agent_pos_y/agent_pos_x",
            "GridWorld agent_grid occupancy",
            "GridWorld resource grid",
            "Agent energy/health/injury/scar fields after runner writeback",
            "death count and per-agent stats deltas",
        ],
        "cpp_materialized_state": [
            "energy",
            "health",
            "injury",
            "scar",
            "died mask",
            "positions",
            "resource grid",
            "cost totals",
        ],
        "python_authoritative_state": [
            "Agent objects/genomes/stats outside materialized fields",
            "selection/reproduction/lineage",
            "checkpoint bundles",
            "metrics definitions",
            "RNG streams",
            "policy/cognition/learning",
        ],
        "fallback_strategy": "on native/missing RNG/divergence, use Python reference output and do not retain C++ output",
        "checkpoint_boundary": "no checkpoint state changes required",
        "metrics_boundary": "no metrics definition changes required",
        "rng_boundary": "no independent C++ RNG; validation action RNG remains Python-precomputed",
        "biome_hazard_expansion": "deferred; requires biome_grid/hazard_rate_map/config in native copied boundary",
        "tier3_transition_expansion": "design-only; branch-dependent crossing draw schedule needs a dedicated preflight",
    }


def state_digest(payload: dict[str, np.ndarray] | None) -> str:
    h = hashlib.sha256()
    if payload is None:
        return ""
    metadata = {"schema_version": SCHEMA_VERSION, "fields": sorted(payload)}
    h.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for key in sorted(payload):
        arr = np.ascontiguousarray(payload[key])
        h.update(key.encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(json.dumps(list(arr.shape)).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _fallback_result(
    inputs: TickIslandLayoutCacheInputs,
    *,
    mode: str,
    timing: dict[str, float],
    diagnostics: dict[str, Any],
    reason: str,
    native_available: bool,
    started: float,
    python_reference: dict[str, np.ndarray] | None = None,
) -> GuardedBridgeResult:
    fallback_t0 = time.perf_counter()
    output = python_reference if python_reference is not None else run_tick_island_python_reference(inputs)
    timing["fallback_s"] = time.perf_counter() - fallback_t0
    timing["total_wall_s"] = time.perf_counter() - started
    return GuardedBridgeResult(
        schema_version=SCHEMA_VERSION,
        mode=mode,
        status="fallback",
        output=output,
        parity=None,
        sparse_parity=None,
        fallback_occurred=True,
        fallback_reason=reason,
        cxx_result_retained=False,
        python_state_restored=True,
        writeback_applied=False,
        native_available=bool(native_available),
        timing=timing,
        diagnostics=diagnostics | {"state_digest": state_digest(output)},
    )


__all__ = [
    "BRIDGE_MODES",
    "SCHEMA_VERSION",
    "Cxxcore8BridgeConfig",
    "GuardedBridgeResult",
    "bridge_audit",
    "config_from_sim_cfg",
    "result_payload",
    "run_guarded_runner_bridge",
    "state_digest",
]
