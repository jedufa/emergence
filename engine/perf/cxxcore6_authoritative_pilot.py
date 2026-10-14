from __future__ import annotations

"""CXXCORE-6 guarded authoritative no-RNG pilot helpers.

Diagnostic/default-off only. The pilot lets the native CXXCORE tick island own
the included no-RNG slice for prepared fixtures while Python remains the normal
runtime authority unless explicit pilot flags are enabled.
"""

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np

from perf.cxxcore5_tick_island_layout_cache import (
    TickIslandLayoutCacheInputs,
    checkpoint_metrics_boundary_classification as cxxcore5_checkpoint_metrics_boundary,
    compare_runner_backend_states,
    compare_summaries,
    run_tick_island_python_reference,
    state_summary,
)


SCHEMA_VERSION = "cxxcore6_authoritative_pilot_v1"
COMPARE_MODES = ("off", "sparse", "periodic_full", "final_full")


@dataclass(frozen=True)
class Cxxcore6PilotConfig:
    cxxcore6_authoritative_pilot_enable: bool = False
    cxxcore6_authoritative_no_rng_only: bool = True
    cxxcore6_authoritative_compare_mode: str = "off"
    cxxcore6_authoritative_fallback_on_divergence: bool = True
    cxxcore6_authoritative_fail_fast_on_divergence: bool = False
    cxxcore6_force_python_baseline: bool = False
    cxxcore6_native_required: bool = False


@dataclass(frozen=True)
class AuthoritativePilotResult:
    schema_version: str
    status: str
    output: dict[str, np.ndarray] | None
    parity: dict[str, Any] | None
    native_available: bool
    fallback_occurred: bool
    fallback_reason: str
    python_state_restored: bool
    cxx_result_retained: bool
    timing: dict[str, float]
    diagnostics: dict[str, Any]


def config_from_sim_cfg(sim_cfg: Any) -> Cxxcore6PilotConfig:
    return Cxxcore6PilotConfig(
        cxxcore6_authoritative_pilot_enable=bool(getattr(sim_cfg, "cxxcore6_authoritative_pilot_enable", False)),
        cxxcore6_authoritative_no_rng_only=bool(getattr(sim_cfg, "cxxcore6_authoritative_no_rng_only", True)),
        cxxcore6_authoritative_compare_mode=str(getattr(sim_cfg, "cxxcore6_authoritative_compare_mode", "off") or "off"),
        cxxcore6_authoritative_fallback_on_divergence=bool(
            getattr(sim_cfg, "cxxcore6_authoritative_fallback_on_divergence", True)
        ),
        cxxcore6_authoritative_fail_fast_on_divergence=bool(
            getattr(sim_cfg, "cxxcore6_authoritative_fail_fast_on_divergence", False)
        ),
        cxxcore6_force_python_baseline=bool(getattr(sim_cfg, "cxxcore6_force_python_baseline", False)),
        cxxcore6_native_required=bool(getattr(sim_cfg, "cxxcore6_native_required", False)),
    )


def run_authoritative_pilot(
    inputs: TickIslandLayoutCacheInputs,
    *,
    config: Cxxcore6PilotConfig,
    native_backend: Callable[[Any], dict[str, Any]] | None,
    python_reference: dict[str, np.ndarray] | None = None,
    compare_mode: str | None = None,
    inject_divergence: bool = False,
) -> AuthoritativePilotResult:
    mode = str(compare_mode or config.cxxcore6_authoritative_compare_mode or "off")
    if mode not in COMPARE_MODES:
        raise ValueError(f"unsupported CXXCORE-6 compare mode: {mode}")
    timing = {
        "total_wall_s": 0.0,
        "python_reference_s": 0.0,
        "native_backend_s": 0.0,
        "writeback_materialization_s": 0.0,
        "sparse_compare_s": 0.0,
        "final_compare_s": 0.0,
        "fallback_s": 0.0,
        "diagnostics_s": 0.0,
    }
    diagnostics = {
        "compare_mode": mode,
        "no_rng_only": bool(config.cxxcore6_authoritative_no_rng_only),
        "cache_hits": int(inputs.cache_snapshot.cache_hits),
        "cache_misses": int(inputs.cache_snapshot.cache_misses),
        "row_updates": int(inputs.cache_snapshot.row_updates),
        "rows_rebuilt": int(inputs.cache_snapshot.rows_rebuilt),
        "full_rebuilds": int(inputs.cache_snapshot.full_rebuilds),
    }
    total_t0 = time.perf_counter()
    if not bool(config.cxxcore6_authoritative_pilot_enable):
        timing["total_wall_s"] = time.perf_counter() - total_t0
        return AuthoritativePilotResult(
            schema_version=SCHEMA_VERSION,
            status="disabled",
            output=None,
            parity=None,
            native_available=bool(native_backend is not None),
            fallback_occurred=False,
            fallback_reason="pilot_disabled",
            python_state_restored=False,
            cxx_result_retained=False,
            timing=timing,
            diagnostics=diagnostics,
        )
    if bool(config.cxxcore6_force_python_baseline):
        return _fallback(
            inputs,
            timing=timing,
            diagnostics=diagnostics,
            reason="force_python_baseline",
            native_available=bool(native_backend is not None),
            python_reference=python_reference,
            config=config,
            started=total_t0,
        )
    if not bool(config.cxxcore6_authoritative_no_rng_only):
        timing["total_wall_s"] = time.perf_counter() - total_t0
        return _fallback(
            inputs,
            timing=timing,
            diagnostics=diagnostics,
            reason="rng_required_surface_not_supported",
            native_available=bool(native_backend is not None),
            python_reference=python_reference,
            config=config,
            started=total_t0,
        )
    if native_backend is None:
        if bool(config.cxxcore6_native_required):
            raise RuntimeError("CXXCORE-6 native backend required but unavailable")
        return _fallback(
            inputs,
            timing=timing,
            diagnostics=diagnostics,
            reason="native_unavailable",
            native_available=False,
            python_reference=python_reference,
            config=config,
            started=total_t0,
        )

    native_t0 = time.perf_counter()
    actual = native_backend(inputs, export_final=True)
    timing["native_backend_s"] = time.perf_counter() - native_t0
    materialize_t0 = time.perf_counter()
    retained = _authoritative_materialization(actual)
    timing["writeback_materialization_s"] = time.perf_counter() - materialize_t0
    if inject_divergence and "new_energy" in retained:
        retained = dict(retained)
        retained["new_energy"] = np.asarray(retained["new_energy"], dtype=np.float32).copy()
        if retained["new_energy"].size:
            retained["new_energy"][0] = np.float32(float(retained["new_energy"][0]) + 1.0)

    parity = None
    if mode in {"sparse", "periodic_full", "final_full"}:
        if python_reference is None:
            ref_t0 = time.perf_counter()
            python_reference = run_tick_island_python_reference(inputs)
            timing["python_reference_s"] += time.perf_counter() - ref_t0
        if mode == "sparse":
            compare_t0 = time.perf_counter()
            parity = compare_summaries(state_summary(python_reference), actual["native_summary"], atol=1.0e-4)
            timing["sparse_compare_s"] = time.perf_counter() - compare_t0
        else:
            compare_t0 = time.perf_counter()
            parity = compare_runner_backend_states(python_reference, retained, atol=1.0e-5)
            timing["final_compare_s"] = time.perf_counter() - compare_t0
        if not bool(parity.get("ok")):
            if bool(config.cxxcore6_authoritative_fail_fast_on_divergence):
                raise RuntimeError(f"CXXCORE-6 divergence: {parity.get('first_divergence')!r}")
            if bool(config.cxxcore6_authoritative_fallback_on_divergence):
                timing["total_wall_s"] = time.perf_counter() - total_t0
                return AuthoritativePilotResult(
                    schema_version=SCHEMA_VERSION,
                    status="fallback",
                    output=python_reference,
                    parity=_enrich_divergence_report(parity, fallback_occurred=True, fallback_reason="divergence"),
                    native_available=True,
                    fallback_occurred=True,
                    fallback_reason="divergence",
                    python_state_restored=True,
                    cxx_result_retained=False,
                    timing=timing,
                    diagnostics=diagnostics,
                )

    timing["total_wall_s"] = time.perf_counter() - total_t0
    return AuthoritativePilotResult(
        schema_version=SCHEMA_VERSION,
        status="authoritative",
        output=retained,
        parity=_enrich_divergence_report(parity, fallback_occurred=False, fallback_reason="") if parity else None,
        native_available=True,
        fallback_occurred=False,
        fallback_reason="",
        python_state_restored=False,
        cxx_result_retained=True,
        timing=timing,
        diagnostics=diagnostics,
    )


def result_payload(result: AuthoritativePilotResult) -> dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "status": result.status,
        "native_available": bool(result.native_available),
        "fallback_occurred": bool(result.fallback_occurred),
        "fallback_reason": result.fallback_reason,
        "python_state_restored": bool(result.python_state_restored),
        "cxx_result_retained": bool(result.cxx_result_retained),
        "parity": result.parity,
        "timing": dict(result.timing),
        "diagnostics": dict(result.diagnostics),
    }


def audit_payload() -> dict[str, Any]:
    return {
        "schema_version": "cxxcore6_authoritative_pilot_audit_v1",
        "included_operations": [
            "copied persistent agent state arrays",
            "action stream movement/gather/rest/attack action-cost semantics from CXXCORE-5 no-RNG slice",
            "legacy grid gather/resource mutation",
            "energy, health, injury, scar pass-through/writeback arrays",
            "death-mask computation",
            "cost layout cache scalar application",
            "final and sparse parity summaries",
        ],
        "excluded_operations": [
            "reproduction and birth",
            "selection and cull/refill",
            "lineage mutation or identity changes",
            "checkpoint serialization as backend-owned state",
            "metrics definition changes",
            "biome hazard RNG",
            "Tier 3 transition/crossing RNG",
            "validation random-walk override RNG",
            "stochastic food spawning or environment initialization",
            "Python Agent object internals outside already materialized runner/env arrays",
        ],
        "mutable_state_written_by_pilot": [
            "energy array",
            "health/injury/scar arrays as pass-through materialized outputs",
            "died mask",
            "agent position arrays",
            "grid resource array",
            "cost total arrays",
        ],
        "python_state_still_authoritative": [
            "Agent objects/genomes/stats",
            "selection/reproduction/lineage state",
            "checkpoint bundle state",
            "metrics logger state",
            "RNG streams",
        ],
        "state_copied_into_cpp": [
            "agent ids",
            "energy/health/injury/scar",
            "position by id",
            "occupancy grid",
            "resource grid",
            "move/harvest multipliers",
            "action stream",
            "cost layout rows and flags",
        ],
        "state_written_back_from_cpp": [
            "energy/health/injury/scar",
            "died mask",
            "positions",
            "resource grid",
            "cost totals",
        ],
        "checkpoint_boundary": "no checkpoint-required state added; backend/cache state is rebuildable from Python authority",
        "metrics_boundary": "no metrics definitions changed; outputs are diagnostics/report-only unless explicit pilot caller records them",
        "rng_boundary": "no independent C++ RNG; RNG-required surfaces are excluded and trigger fallback/defer",
        "divergence_localization": [
            "field",
            "agent index/id when available",
            "grid coordinate for grid fields",
            "Python/reference value",
            "C++ authoritative value",
            "absolute difference",
            "fallback action",
        ],
        "fallback_kill_switch": [
            "pilot disabled by default",
            "native unavailable fallback",
            "rng-required-surface fallback",
            "divergence fallback when configured",
            "force Python baseline flag",
        ],
        "scope_gate": "guarded_authoritative_no_rng_pilot_feasible",
    }


def boundary_classification() -> list[dict[str, str]]:
    base = cxxcore5_checkpoint_metrics_boundary()
    return base + [
        {"state": "CXXCORE-6 pilot flags", "classification": "default-off runtime configuration"},
        {"state": "CXXCORE-6 materialized native outputs", "classification": "guarded authoritative pilot writeback state"},
        {"state": "CXXCORE-6 divergence reports", "classification": "runtime diagnostic output"},
        {"state": "RNG preflight arrays", "classification": "not implemented in CXXCORE-6 primary pilot"},
    ]


def rng_preflight_audit() -> dict[str, Any]:
    return {
        "schema_version": "cxxcore6_rng_preflight_audit_v1",
        "rng_sources_blocking_expansion": [
            "validation random-walk override via runner rng_manager.policy_np.integers",
            "biome hazard random draws inside GridWorld.step_agents_batch",
            "Tier 3 crossing/transition random draws inside GridWorld.step_agents_batch",
            "stochastic food spawning/environment initialization outside selected slice",
        ],
        "precomputed_rng_contract": {
            "status": "deferred",
            "reason": "no-RNG authoritative pilot already yielded useful timing/parity; stochastic surfaces should be a dedicated CXXCORE-7 preflight",
            "required_arrays": [
                "validation random-walk action overrides",
                "biome hazard uniform draws",
                "Tier 3 crossing/transition uniform draws",
            ],
        },
        "recommendation": "CXXCORE7-RNG-PRECOMPUTE-PREFLIGHT-GO",
    }


def _fallback(
    inputs: TickIslandLayoutCacheInputs,
    *,
    timing: dict[str, float],
    diagnostics: dict[str, Any],
    reason: str,
    native_available: bool,
    python_reference: dict[str, np.ndarray] | None,
    config: Cxxcore6PilotConfig,
    started: float,
) -> AuthoritativePilotResult:
    fallback_t0 = time.perf_counter()
    output = python_reference if python_reference is not None else run_tick_island_python_reference(inputs)
    timing["fallback_s"] = time.perf_counter() - fallback_t0
    timing["total_wall_s"] = time.perf_counter() - started
    return AuthoritativePilotResult(
        schema_version=SCHEMA_VERSION,
        status="fallback",
        output=output,
        parity=None,
        native_available=bool(native_available),
        fallback_occurred=True,
        fallback_reason=reason,
        python_state_restored=True,
        cxx_result_retained=False,
        timing=timing,
        diagnostics=diagnostics | {"native_required": bool(config.cxxcore6_native_required)},
    )


def _authoritative_materialization(actual: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "new_energy": np.ascontiguousarray(actual["new_energy"], dtype=np.float32),
        "new_health": np.ascontiguousarray(actual["new_health"], dtype=np.float32),
        "new_injury": np.ascontiguousarray(actual["new_injury"], dtype=np.float32),
        "new_scar": np.ascontiguousarray(actual["new_scar"], dtype=np.float32),
        "died_mask": np.ascontiguousarray(actual["died_mask"], dtype=bool),
        "new_pos_y": np.ascontiguousarray(actual["new_pos_y"], dtype=np.int32),
        "new_pos_x": np.ascontiguousarray(actual["new_pos_x"], dtype=np.int32),
        "grid": np.ascontiguousarray(actual["grid"]),
        "cost_total_by_agent": np.ascontiguousarray(actual["cost_total_by_agent"], dtype=np.float32),
        "cost_total": np.ascontiguousarray(actual["cost_total"], dtype=np.float64),
    }


def _enrich_divergence_report(
    parity: dict[str, Any] | None,
    *,
    fallback_occurred: bool,
    fallback_reason: str,
) -> dict[str, Any] | None:
    if parity is None:
        return None
    report = dict(parity)
    first = dict(report.get("first_divergence") or {})
    report["fallback_occurred"] = bool(fallback_occurred)
    report["fallback_reason"] = str(fallback_reason)
    if first:
        report["divergence"] = {
            "field_name": first.get("field"),
            "agent_id": first.get("agent_id"),
            "grid_coordinate": first.get("index") if first.get("field") == "grid" else None,
            "python_value": first.get("python", first.get("production")),
            "cxx_authoritative_value": first.get("candidate", first.get("layout")),
            "abs_diff": first.get("abs_diff"),
            "relative_diff": None,
            "classification": first.get("classification"),
            "fallback_occurred": bool(fallback_occurred),
            "fallback_reason": str(fallback_reason),
            "python_state_restored": bool(fallback_occurred),
            "cxx_result_retained": not bool(fallback_occurred),
        }
    return report


__all__ = [
    "COMPARE_MODES",
    "SCHEMA_VERSION",
    "AuthoritativePilotResult",
    "Cxxcore6PilotConfig",
    "audit_payload",
    "boundary_classification",
    "config_from_sim_cfg",
    "result_payload",
    "rng_preflight_audit",
    "run_authoritative_pilot",
]
