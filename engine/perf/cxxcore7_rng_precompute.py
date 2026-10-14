from __future__ import annotations

"""CXXCORE-7 precomputed Python RNG-array contract.

Diagnostic/default-off only. Python remains the authoritative RNG scheduler.
C++ may consume arrays produced by Python; it must not generate random values.
"""

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any

import numpy as np

from perf.cxxcore5_tick_island_layout_cache import TickIslandLayoutCacheInputs, as_tick_island_inputs
from perf.cxxcore6_authoritative_pilot import (
    AuthoritativePilotResult,
    Cxxcore6PilotConfig,
    _authoritative_materialization,
    _enrich_divergence_report,
    _fallback,
)
from perf.cxxcore5_tick_island_layout_cache import (
    compare_runner_backend_states,
    compare_summaries,
    run_tick_island_python_reference,
    state_summary,
)


SCHEMA_VERSION = "cxxcore7_rng_precompute_v1"
PILOT_SCHEMA_VERSION = "cxxcore7_rng_precompute_authoritative_pilot_v1"
COMPARE_MODES = ("off", "sparse", "periodic_full", "final_full")


@dataclass(frozen=True)
class PrecomputedRngArray:
    schema_version: str
    name: str
    values: np.ndarray
    source_rng_stream: str
    source_function: str
    generation_start: int
    tick_start: int
    ticks: int
    agents: int
    dtype: str
    draw_schedule: str
    consumed_conditionally: bool
    sentinel: int | float
    tolerance: float
    digest: str


@dataclass(frozen=True)
class Cxxcore7RngPayload:
    schema_version: str
    enabled_surfaces: tuple[str, ...]
    validation_action_override_ints: PrecomputedRngArray | None
    biome_hazard_uniforms: PrecomputedRngArray | None
    tier3_transition_uniforms: PrecomputedRngArray | None
    payload_digest: str
    audit: dict[str, Any]


@dataclass(frozen=True)
class Cxxcore7PilotConfig:
    cxxcore7_rng_precompute_enable: bool = False
    cxxcore7_validation_action_override_enable: bool = False
    cxxcore7_runner_bridge_probe_enable: bool = False
    cxxcore7_native_required: bool = False
    cxxcore6_authoritative_pilot_enable: bool = False
    cxxcore6_authoritative_compare_mode: str = "off"
    cxxcore6_authoritative_fallback_on_divergence: bool = True
    cxxcore6_authoritative_fail_fast_on_divergence: bool = False
    cxxcore6_force_python_baseline: bool = False


def config_from_sim_cfg(sim_cfg: Any) -> Cxxcore7PilotConfig:
    return Cxxcore7PilotConfig(
        cxxcore7_rng_precompute_enable=bool(getattr(sim_cfg, "cxxcore7_rng_precompute_enable", False)),
        cxxcore7_validation_action_override_enable=bool(
            getattr(sim_cfg, "cxxcore7_validation_action_override_enable", False)
        ),
        cxxcore7_runner_bridge_probe_enable=bool(getattr(sim_cfg, "cxxcore7_runner_bridge_probe_enable", False)),
        cxxcore7_native_required=bool(getattr(sim_cfg, "cxxcore7_native_required", False)),
        cxxcore6_authoritative_pilot_enable=bool(getattr(sim_cfg, "cxxcore6_authoritative_pilot_enable", False)),
        cxxcore6_authoritative_compare_mode=str(getattr(sim_cfg, "cxxcore6_authoritative_compare_mode", "off") or "off"),
        cxxcore6_authoritative_fallback_on_divergence=bool(
            getattr(sim_cfg, "cxxcore6_authoritative_fallback_on_divergence", True)
        ),
        cxxcore6_authoritative_fail_fast_on_divergence=bool(
            getattr(sim_cfg, "cxxcore6_authoritative_fail_fast_on_divergence", False)
        ),
        cxxcore6_force_python_baseline=bool(getattr(sim_cfg, "cxxcore6_force_python_baseline", False)),
    )


def precompute_validation_action_override_ints(
    *,
    rng: np.random.Generator,
    ticks: int,
    agents: int,
    generation_start: int = 0,
    tick_start: int = 0,
) -> PrecomputedRngArray:
    if int(ticks) < 0 or int(agents) < 0:
        raise ValueError("ticks and agents must be non-negative")
    values = np.ascontiguousarray(rng.integers(0, 4, size=(int(ticks), int(agents)), dtype=np.int32), dtype=np.int32)
    return _array_contract(
        name="validation_action_override_ints",
        values=values,
        source_rng_stream="rng_manager.policy_np",
        source_function="TickEnvApplyMixin._apply_env_tick validation random-walk override",
        generation_start=int(generation_start),
        tick_start=int(tick_start),
        draw_schedule="one int32 draw per active agent per tick, aligned to action matrix columns",
        consumed_conditionally=False,
        sentinel=-1,
        tolerance=0.0,
    )


def build_rng_payload(
    *,
    validation_action_override_ints: PrecomputedRngArray | None = None,
) -> Cxxcore7RngPayload:
    enabled = []
    if validation_action_override_ints is not None:
        enabled.append("validation_action_override_ints")
    audit = rng_surface_audit(included_surfaces=tuple(enabled))
    payload_digest = _stable_digest(
        {
            "schema_version": SCHEMA_VERSION,
            "enabled_surfaces": enabled,
            "validation_action_override_ints": (
                validation_action_override_ints.digest if validation_action_override_ints is not None else None
            ),
            "biome_hazard_uniforms": None,
            "tier3_transition_uniforms": None,
        }
    )
    return Cxxcore7RngPayload(
        schema_version=SCHEMA_VERSION,
        enabled_surfaces=tuple(enabled),
        validation_action_override_ints=validation_action_override_ints,
        biome_hazard_uniforms=None,
        tier3_transition_uniforms=None,
        payload_digest=payload_digest,
        audit=audit,
    )


def validate_rng_payload(payload: Cxxcore7RngPayload, *, ticks: int, agents: int) -> dict[str, Any]:
    errors: list[str] = []
    if payload.schema_version != SCHEMA_VERSION:
        errors.append("unsupported_schema_version")
    if "validation_action_override_ints" in payload.enabled_surfaces:
        arr = payload.validation_action_override_ints
        if arr is None:
            errors.append("validation_action_override_ints_missing")
        else:
            _validate_array(arr, ticks=ticks, agents=agents, dtype="int32", errors=errors)
            values = np.asarray(arr.values, dtype=np.int32)
            if values.size and (int(np.min(values)) < 0 or int(np.max(values)) > 3):
                errors.append("validation_action_override_ints_out_of_range")
    unsupported = set(payload.enabled_surfaces) - {"validation_action_override_ints"}
    for name in sorted(unsupported):
        errors.append(f"unsupported_surface:{name}")
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": not errors,
        "errors": errors,
        "payload_digest": payload.payload_digest,
        "enabled_surfaces": list(payload.enabled_surfaces),
    }


def apply_validation_action_override(
    inputs: TickIslandLayoutCacheInputs,
    payload: Cxxcore7RngPayload,
) -> TickIslandLayoutCacheInputs:
    state = inputs.backend.runner.state
    report = validate_rng_payload(
        payload,
        ticks=int(state.actions_by_tick.shape[0]),
        agents=int(state.actions_by_tick.shape[1]),
    )
    if not report["ok"]:
        raise ValueError(f"CXXCORE-7 RNG payload invalid: {report['errors']}")
    if payload.validation_action_override_ints is None:
        return inputs
    override = np.ascontiguousarray(payload.validation_action_override_ints.values, dtype=np.int32)
    return as_tick_island_inputs(
        config=state.config,
        agent_ids=state.agent_ids,
        initial_energy=state.initial_energy,
        initial_health=state.initial_health,
        initial_injury=state.initial_injury,
        initial_scar=state.initial_scar,
        pos_y_by_id=state.pos_y_by_id,
        pos_x_by_id=state.pos_x_by_id,
        agent_grid=state.agent_grid,
        grid=state.grid,
        actions_by_tick=override,
        cache_snapshot=inputs.cache_snapshot,
        move_cost_mults=state.move_cost_mults,
        juvenile_harvest_mults=state.juvenile_harvest_mults,
    )


def run_authoritative_precomputed_rng_pilot(
    inputs: TickIslandLayoutCacheInputs,
    *,
    config: Cxxcore7PilotConfig,
    rng_payload: Cxxcore7RngPayload | None,
    native_backend: Any | None,
    python_reference: dict[str, np.ndarray] | None = None,
    compare_mode: str | None = None,
    inject_divergence: bool = False,
) -> AuthoritativePilotResult:
    mode = str(compare_mode or config.cxxcore6_authoritative_compare_mode or "off")
    if mode not in COMPARE_MODES:
        raise ValueError(f"unsupported CXXCORE-7 compare mode: {mode}")
    cxx6_cfg = Cxxcore6PilotConfig(
        cxxcore6_authoritative_pilot_enable=bool(config.cxxcore6_authoritative_pilot_enable),
        cxxcore6_authoritative_no_rng_only=True,
        cxxcore6_authoritative_compare_mode=mode,
        cxxcore6_authoritative_fallback_on_divergence=bool(config.cxxcore6_authoritative_fallback_on_divergence),
        cxxcore6_authoritative_fail_fast_on_divergence=bool(config.cxxcore6_authoritative_fail_fast_on_divergence),
        cxxcore6_force_python_baseline=bool(config.cxxcore6_force_python_baseline),
        cxxcore6_native_required=bool(config.cxxcore7_native_required),
    )
    timing = {
        "total_wall_s": 0.0,
        "python_reference_s": 0.0,
        "rng_precompute_capture_s": 0.0,
        "native_backend_s": 0.0,
        "writeback_materialization_s": 0.0,
        "sparse_compare_s": 0.0,
        "final_compare_s": 0.0,
        "fallback_s": 0.0,
        "diagnostics_s": 0.0,
    }
    state = inputs.backend.runner.state
    diagnostics = {
        "compare_mode": mode,
        "rng_precompute_enabled": bool(config.cxxcore7_rng_precompute_enable),
        "validation_action_override_enabled": bool(config.cxxcore7_validation_action_override_enable),
        "enabled_surfaces": list(rng_payload.enabled_surfaces if rng_payload is not None else ()),
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
            schema_version=PILOT_SCHEMA_VERSION,
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
        ref_inputs = _reference_inputs(inputs, rng_payload)
        return _fallback(
            ref_inputs,
            timing=timing,
            diagnostics=diagnostics,
            reason="force_python_baseline",
            native_available=bool(native_backend is not None),
            python_reference=python_reference,
            config=cxx6_cfg,
            started=total_t0,
        )
    if not bool(config.cxxcore7_rng_precompute_enable):
        ref_inputs = _reference_inputs(inputs, rng_payload)
        return _fallback(
            ref_inputs,
            timing=timing,
            diagnostics=diagnostics,
            reason="rng_precompute_disabled",
            native_available=bool(native_backend is not None),
            python_reference=python_reference,
            config=cxx6_cfg,
            started=total_t0,
        )
    if rng_payload is None:
        ref_inputs = _reference_inputs(inputs, rng_payload)
        return _fallback(
            ref_inputs,
            timing=timing,
            diagnostics=diagnostics,
            reason="rng_payload_missing",
            native_available=bool(native_backend is not None),
            python_reference=python_reference,
            config=cxx6_cfg,
            started=total_t0,
        )
    validation = validate_rng_payload(
        rng_payload,
        ticks=int(state.actions_by_tick.shape[0]),
        agents=int(state.actions_by_tick.shape[1]),
    )
    diagnostics["rng_validation"] = validation
    if not validation["ok"]:
        ref_inputs = _reference_inputs(inputs, None)
        return _fallback(
            ref_inputs,
            timing=timing,
            diagnostics=diagnostics,
            reason="rng_payload_invalid",
            native_available=bool(native_backend is not None),
            python_reference=python_reference,
            config=cxx6_cfg,
            started=total_t0,
        )
    ref_inputs = apply_validation_action_override(inputs, rng_payload)
    if native_backend is None:
        if bool(config.cxxcore7_native_required):
            raise RuntimeError("CXXCORE-7 native backend required but unavailable")
        return _fallback(
            ref_inputs,
            timing=timing,
            diagnostics=diagnostics,
            reason="native_unavailable",
            native_available=False,
            python_reference=python_reference,
            config=cxx6_cfg,
            started=total_t0,
        )

    native_t0 = time.perf_counter()
    actual = native_backend(inputs, rng_payload=rng_payload, export_final=True)
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
            python_reference = run_tick_island_python_reference(ref_inputs)
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
                raise RuntimeError(f"CXXCORE-7 divergence: {parity.get('first_divergence')!r}")
            if bool(config.cxxcore6_authoritative_fallback_on_divergence):
                timing["total_wall_s"] = time.perf_counter() - total_t0
                return AuthoritativePilotResult(
                    schema_version=PILOT_SCHEMA_VERSION,
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
        schema_version=PILOT_SCHEMA_VERSION,
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


def rng_surface_audit(*, included_surfaces: tuple[str, ...] = ("validation_action_override_ints",)) -> dict[str, Any]:
    included = set(included_surfaces)
    rows = [
        {
            "surface": "validation random-walk action override",
            "file_function": "engine/simulation/runner_pkg/runner_ticks_env_apply.py::TickEnvApplyMixin._apply_env_tick",
            "rng_source": "self.rng_manager.policy_np.integers(0, 4, size=actions_np.shape[0], dtype=np.int32)",
            "draw_count_per_tick_agent": "one int32 per alive/action row when migration_pressure_enable and tier3_validation_force_random_walk are true",
            "branch_dependent": False,
            "precompute_feasible": True,
            "included_in_cxxcore7": "validation_action_override_ints" in included,
            "reason": "action stream is already a CXXCORE-6 copied input; precompute does not add state ownership",
        },
        {
            "surface": "biome ambient hazard",
            "file_function": "engine/env/gridworld_pkg/stepping.py::GridWorld.step_agents_batch/_hazard_kernel",
            "rng_source": "self.rng.random(n, dtype=np.float32)",
            "draw_count_per_tick_agent": "one uniform per canonicalized batch row when enable_biomes is true",
            "branch_dependent": False,
            "precompute_feasible": True,
            "included_in_cxxcore7": False,
            "reason": "requires biome_grid/hazard_rate_map/hazard config in native copied state; deferred to avoid widening CXXCORE-6 boundary",
        },
        {
            "surface": "Tier 3 transition/crossing risk",
            "file_function": "engine/env/gridworld_pkg/stepping.py::GridWorld.step_agents_batch",
            "rng_source": "self.rng.random(n, dtype=np.float32)",
            "draw_count_per_tick_agent": "one uniform per canonicalized batch row only when Tier 3 transition risk is enabled and crossings occurred",
            "branch_dependent": True,
            "precompute_feasible": True,
            "included_in_cxxcore7": False,
            "reason": "requires region grids/origin-destination transition state and branch-dependent crossing detection; deferred",
        },
        {
            "surface": "stochastic food spawning/environment initialization",
            "file_function": "engine/env/gridworld_pkg/food.py::FoodMixin.spawn_food and environment init paths",
            "rng_source": "self.rng.integers / self.rng.choice / self.rng.random",
            "draw_count_per_tick_agent": "not per-agent; branch/retry dependent on occupancy, weights, max_food, and spawn attempts",
            "branch_dependent": True,
            "precompute_feasible": False,
            "included_in_cxxcore7": False,
            "reason": "outside selected tick-island slice and retry schedule can change draw counts",
        },
    ]
    return {
        "schema_version": "cxxcore7_rng_surface_audit_v1",
        "rows": rows,
        "decision": "validation_action_override_ints_first_surface" if "validation_action_override_ints" in included else "no_surface_selected",
        "checkpoint_boundary": "no checkpoint-required state added; RNG arrays are report/probe inputs and Python RNG remains authoritative",
        "metrics_boundary": "no metrics definitions changed",
        "cpp_rng_boundary": "no independent C++ RNG, no C++ seeding, no C++ random library use",
    }


def _reference_inputs(inputs: TickIslandLayoutCacheInputs, payload: Cxxcore7RngPayload | None) -> TickIslandLayoutCacheInputs:
    if payload is None or payload.validation_action_override_ints is None:
        return inputs
    return apply_validation_action_override(inputs, payload)


def _array_contract(
    *,
    name: str,
    values: np.ndarray,
    source_rng_stream: str,
    source_function: str,
    generation_start: int,
    tick_start: int,
    draw_schedule: str,
    consumed_conditionally: bool,
    sentinel: int | float,
    tolerance: float,
) -> PrecomputedRngArray:
    arr = np.ascontiguousarray(values)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    digest = _array_digest(
        name=name,
        values=arr,
        source_rng_stream=source_rng_stream,
        source_function=source_function,
        generation_start=generation_start,
        tick_start=tick_start,
        draw_schedule=draw_schedule,
        consumed_conditionally=consumed_conditionally,
        sentinel=sentinel,
        tolerance=tolerance,
    )
    return PrecomputedRngArray(
        schema_version=SCHEMA_VERSION,
        name=name,
        values=arr,
        source_rng_stream=source_rng_stream,
        source_function=source_function,
        generation_start=int(generation_start),
        tick_start=int(tick_start),
        ticks=int(arr.shape[0]),
        agents=int(arr.shape[1]),
        dtype=str(arr.dtype),
        draw_schedule=draw_schedule,
        consumed_conditionally=bool(consumed_conditionally),
        sentinel=sentinel,
        tolerance=float(tolerance),
        digest=digest,
    )


def _validate_array(arr: PrecomputedRngArray, *, ticks: int, agents: int, dtype: str, errors: list[str]) -> None:
    values = np.asarray(arr.values)
    if arr.schema_version != SCHEMA_VERSION:
        errors.append(f"{arr.name}:unsupported_schema_version")
    if str(values.dtype) != dtype:
        errors.append(f"{arr.name}:dtype_mismatch")
    if tuple(values.shape) != (int(ticks), int(agents)):
        errors.append(f"{arr.name}:shape_mismatch")
    if arr.ticks != int(ticks) or arr.agents != int(agents):
        errors.append(f"{arr.name}:metadata_shape_mismatch")
    if _array_digest(
        name=arr.name,
        values=np.ascontiguousarray(values),
        source_rng_stream=arr.source_rng_stream,
        source_function=arr.source_function,
        generation_start=arr.generation_start,
        tick_start=arr.tick_start,
        draw_schedule=arr.draw_schedule,
        consumed_conditionally=arr.consumed_conditionally,
        sentinel=arr.sentinel,
        tolerance=arr.tolerance,
    ) != arr.digest:
        errors.append(f"{arr.name}:digest_mismatch")


def _array_digest(
    *,
    name: str,
    values: np.ndarray,
    source_rng_stream: str,
    source_function: str,
    generation_start: int,
    tick_start: int,
    draw_schedule: str,
    consumed_conditionally: bool,
    sentinel: int | float,
    tolerance: float,
) -> str:
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "source_rng_stream": source_rng_stream,
        "source_function": source_function,
        "generation_start": int(generation_start),
        "tick_start": int(tick_start),
        "shape": [int(v) for v in values.shape],
        "dtype": str(values.dtype),
        "draw_schedule": draw_schedule,
        "consumed_conditionally": bool(consumed_conditionally),
        "sentinel": sentinel,
        "tolerance": float(tolerance),
    }
    h = hashlib.sha256()
    h.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    h.update(np.ascontiguousarray(values).tobytes())
    return h.hexdigest()


def _stable_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


__all__ = [
    "COMPARE_MODES",
    "PILOT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "Cxxcore7PilotConfig",
    "Cxxcore7RngPayload",
    "PrecomputedRngArray",
    "apply_validation_action_override",
    "build_rng_payload",
    "config_from_sim_cfg",
    "precompute_validation_action_override_ints",
    "rng_surface_audit",
    "run_authoritative_precomputed_rng_pilot",
    "validate_rng_payload",
]
