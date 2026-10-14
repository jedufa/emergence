from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

try:
    from simulation.metrics_pkg.stats import mean, percentile, std
except ImportError:  # pragma: no cover - import fallback for standalone probes

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def std(xs: list[float]) -> float:
        if not xs:
            return 0.0
        m = mean(xs)
        return math.sqrt(mean([(x - m) ** 2 for x in xs]))

    def percentile(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        ys = sorted(xs)
        if len(ys) == 1:
            return float(ys[0])
        if p <= 0:
            return float(ys[0])
        if p >= 100:
            return float(ys[-1])
        pos = (p / 100.0) * (len(ys) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(ys[lo])
        frac = pos - lo
        return float(ys[lo] * (1.0 - frac) + ys[hi] * frac)


BEHAVIOR_ROW_FIELDS = (
    "generation",
    "agent_id",
    "cohort",
    "move_n",
    "move_s",
    "move_e",
    "move_w",
    "gather",
    "rest",
    "attack",
    "lifespan_steps",
    "energy_collected",
    "offspring",
    "attacks_made",
    "fitness",
    "alive_end",
    "eligible_for_selection",
    "kept_next",
    "injury_end",
    "injury_delta",
    "aggression_bias",
)

# Exact Flat resident reporting fairness surface after the chemistry closure.
NATIVE_FLAT_RESIDENT_REPORTING_COLUMNS = (
    "cog_capacity_mean",
    "cog_capacity_p50",
    "cog_capacity_p90",
    "cog_capacity_max",
    "cog_capacity_min",
    "cog_capacity_std",
    "cog_capacity_alive_mean",
    "cog_capacity_parent_mean",
    "cog_capacity_survivor_mean",
    "mem_evo_population_memory_dim_histogram",
    "mem_evo_selected_parent_memory_dim_histogram",
    "mem_evo_descendant_memory_dim_histogram",
    "mem_evo_lineage_memory_dim_summary_json",
    "mem_evo_effective_memory_dim_histogram",
    "prediction_lr_mean",
    "prediction_lr_p50",
    "prediction_lr_p90",
    "prediction_lr_max",
    "prediction_lr_std",
    "drive_gain_mean",
    "drive_gain_p90",
    "drive_noise_mean",
    "drive_inertia_mean",
    "drive_conflict_mean",
    "drive_conflict_p90",
    "hunger_sensitivity_mean",
    "fear_sensitivity_mean",
    "pain_sensitivity_mean",
    "fatigue_sensitivity_mean",
    "parent_cog_capacity_mean",
    "alive_cog_capacity_mean",
    "reproduced_cog_capacity_mean",
    "non_reproduced_cog_capacity_mean",
    "cognitive_selection_delta_parent_vs_alive",
    "drive_regulation_selection_delta_parent_vs_alive",
    "cognitive_trait_diversity",
    "cognitive_capacity_entropy",
    "drive_profile_diversity",
    "cognitive_strategy_count_approx",
    "juvenile_cog_capacity_mean",
    "adult_cog_capacity_mean",
    "elder_cog_capacity_mean",
    "cognition_by_age_slope",
    "juvenile_fraction",
    "adult_fraction",
    "mature_fraction",
    "adult_sensor_radius_mean",
    "adult_sensor_channel_count_mean",
    "reproduced_maturity_age_mean",
    "adult_survival_rate",
    "adult_reproduction_attempt_rate",
    "high_adult_cognition_low_developmental_cost_fraction",
    "adult_cognition_from_birth_fraction",
    "juvenile_high_cognition_fraction",
    "internal_energy_signal_mean",
    "internal_health_signal_mean",
    "internal_drive_conflict_signal_mean",
    "full_cognitive_strategy_count_approx",
    "lineage_cog_capacity_mean",
    "top_lineage_cog_capacity",
    "lineage_cognitive_diversity",
    "high_cognition_low_cost_fraction",
    "cognition_capacity_in_flat_world_baseline",
    "cog_capacity_alive_end_mean",
    "cog_capacity_drive_component_mean",
    "sensor_radius_cost_hypothetical_total",
)

if len(NATIVE_FLAT_RESIDENT_REPORTING_COLUMNS) != 66 or len(
    set(NATIVE_FLAT_RESIDENT_REPORTING_COLUMNS)
) != 66:
    raise RuntimeError("native Flat resident reporting contract must contain 66 unique columns")

NATIVE_LINEAGE_METRICS_INPUT_KEY = "native_lineage_metrics_input"
NATIVE_LINEAGE_METRICS_INPUT_SCHEMA_VERSION = "eiel_native_lineage_metrics_input_v1"

_NATIVE_LINEAGE_REQUIRED_TENSORS = (
    "alive_mask",
    "slot_to_agent_id",
    "pos_y",
    "pos_x",
    "energy",
    "policy_energy_observation",
    "health",
    "injury",
    "drives",
    "lineage_founder_id",
    "lineage_depth",
    "lineage_birth_generation",
    "lineage_birth_tick",
    "offspring_count",
    "first_repro_step",
    "survival_steps",
    "energy_collected",
    "pred_err_count",
    "pred_err_sum",
    "episode0_pred_err_count",
    "episode0_pred_err_sum",
    "drive_jitter",
    "genome_hidden_size",
    "genome_memory_dim",
    "policy_use_gru",
    "genome_exploration_bias",
    "genome_aggression_bias",
)

_NATIVE_CHILD_SCALAR_INT_FIELDS = (
    "genome_hidden_size",
    "genome_memory_dim",
    "genome_sensor_channel_mask",
    "genome_sensor_radius",
    "genome_introspection_channel_mask",
    "genome_self_regulation_control_mask",
    "genome_self_modification_target_mask",
    "genome_directed_inheritance_channel_mask",
)

_NATIVE_CHILD_SCALAR_FLOAT_FIELDS = (
    "genome_exploration_bias",
    "genome_aggression_bias",
    "genome_mutation_rate",
    "genome_reproduction_threshold",
    "genome_action_temperature",
    "genome_drive_inertia",
    "genome_drive_noise",
    "genome_drive_logit_gain",
    "genome_pred_lr",
    "genome_drive_sp_hunger",
    "genome_drive_sp_pain",
    "genome_drive_sp_fatigue",
    "genome_drive_sp_fear",
    "genome_exp_prior_strength",
    "genome_exp_prior_noise",
    "genome_drive_birth_noise",
)


def _tier4_reproduction_percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((float(p) / 100.0) * (len(ordered) - 1)))
    return float(ordered[max(0, min(len(ordered) - 1, index))])

DEFAULT_SURVIVAL_BINS_STEPS = (0, 10, 25, 50, 80, 120, 200)

_POLICY_CACHE_ROW_KEYS = (
    "policy_cache_rows_requested",
    "policy_cache_rows_hit",
    "policy_cache_rows_missed",
    "policy_cache_rows_refilled",
)
_POLICY_CACHE_TICK_KEYS = (
    "policy_cache_full_refill_ticks",
    "policy_cache_partial_refill_ticks",
)
_POLICY_CACHE_LEGACY_KEYS = (
    "policy_loop_tick_calls",
    "policy_refill_calls",
    "policy_refill_rows",
    "cache_hit_rows",
    "policy_cache_hit_calls",
)


def native_policy_cache_metrics(
    raw_result: Any | None = None,
    compact_result: Any | None = None,
) -> dict[str, int]:
    """Normalize native policy-cache counters across native result wrappers."""

    raw = _mapping(raw_result)
    compact = _mapping(compact_result)
    if not raw:
        raw = _mapping(_get(raw_result, "raw_result"))
    if not compact:
        compact = _mapping(_get(raw_result, "compact_result"))

    merged: dict[str, int] = {}
    for source in (raw, compact):
        if not source:
            continue
        counts = _policy_cache_metrics_from_result(source)
        _merge_policy_cache_counts(merged, counts, context="native raw/compact results")
    return _normalize_policy_cache_counts(merged)


def _policy_cache_metrics_from_result(source: Mapping[str, Any]) -> dict[str, int]:
    one_tick = _policy_cache_one_tick_result(source)
    counts = _normalize_policy_cache_counts(source, one_tick=one_tick)

    microcycles_value = _get(source, "microcycles")
    microcycles = (
        list(microcycles_value)
        if isinstance(microcycles_value, Sequence) and not isinstance(microcycles_value, (str, bytes, bytearray))
        else []
    )
    cycle_counts = [
        _policy_cache_metrics_from_result(cycle)
        for cycle in (_mapping(value) for value in microcycles)
        if cycle
    ]
    if cycle_counts:
        aggregate: dict[str, int] = {}
        for key in _POLICY_CACHE_ROW_KEYS + _POLICY_CACHE_TICK_KEYS + _POLICY_CACHE_LEGACY_KEYS:
            if all(key in cycle for cycle in cycle_counts):
                aggregate[key] = sum(int(cycle[key]) for cycle in cycle_counts)
        aggregate = _normalize_policy_cache_counts(aggregate)
        _merge_policy_cache_counts(counts, aggregate, context="native microcycle aggregate")

    is_multi_tick_microcycle = bool(_get(source, "native_lifecycle_microcycles", False)) and not one_tick
    if not is_multi_tick_microcycle:
        nested = _mapping(_get(source, "flat_generation_core"))
        if nested:
            _merge_policy_cache_counts(
                counts,
                _policy_cache_metrics_from_result(nested),
                context="native flat_generation_core",
            )
    return _normalize_policy_cache_counts(counts, one_tick=one_tick)


def _normalize_policy_cache_counts(
    source: Mapping[str, Any],
    *,
    one_tick: bool = False,
) -> dict[str, int]:
    requested = _policy_cache_count(source, "policy_cache_rows_requested")
    hit = _matching_policy_cache_count(source, "policy_cache_rows_hit", "cache_hit_rows")
    missed = _policy_cache_count(source, "policy_cache_rows_missed")
    refilled = _matching_policy_cache_count(source, "policy_cache_rows_refilled", "policy_refill_rows")

    if missed is not None and refilled is not None and missed != refilled:
        raise ValueError(
            "native policy-cache row counters violate "
            f"missed=refilled: missed={missed} refilled={refilled}"
        )
    if missed is None:
        missed = refilled
    if refilled is None:
        refilled = missed

    if requested is None and hit is not None and missed is not None:
        requested = hit + missed
    elif hit is None and requested is not None and missed is not None:
        hit = requested - missed
    elif missed is None and requested is not None and hit is not None:
        missed = requested - hit
        refilled = missed

    if hit is not None and hit < 0:
        raise ValueError("native policy-cache rows_hit cannot be negative")
    if missed is not None and missed < 0:
        raise ValueError("native policy-cache rows_missed cannot be negative")
    if requested is not None and hit is not None and missed is not None and requested != hit + missed:
        raise ValueError(
            "native policy-cache row counters violate "
            f"requested=hit+miss: requested={requested} hit={hit} missed={missed}"
        )
    if missed is not None and refilled is not None and missed != refilled:
        raise ValueError(
            "native policy-cache row counters violate "
            f"missed=refilled: missed={missed} refilled={refilled}"
        )

    policy_loop_tick_calls = _policy_cache_count(source, "policy_loop_tick_calls")
    policy_refill_calls = _policy_cache_count(source, "policy_refill_calls")
    policy_cache_hit_calls = _policy_cache_count(source, "policy_cache_hit_calls")
    full_present = "policy_cache_full_refill_ticks" in source
    partial_present = "policy_cache_partial_refill_ticks" in source
    full_refill_ticks = _policy_cache_count(source, "policy_cache_full_refill_ticks")
    partial_refill_ticks = _policy_cache_count(source, "policy_cache_partial_refill_ticks")

    if full_present or partial_present:
        full_refill_ticks = int(full_refill_ticks or 0)
        partial_refill_ticks = int(partial_refill_ticks or 0)
        refill_ticks = full_refill_ticks + partial_refill_ticks
        if policy_refill_calls is not None and policy_refill_calls != refill_ticks:
            raise ValueError(
                "native policy-cache tick counters violate "
                "policy_refill_calls=full+partial: "
                f"calls={policy_refill_calls} full={full_refill_ticks} partial={partial_refill_ticks}"
            )
        policy_refill_calls = refill_ticks
    elif policy_refill_calls is None and policy_loop_tick_calls is not None and policy_cache_hit_calls is not None:
        policy_refill_calls = policy_loop_tick_calls - policy_cache_hit_calls

    if policy_refill_calls is not None and policy_refill_calls < 0:
        raise ValueError("native policy-cache policy_refill_calls cannot be negative")
    if policy_loop_tick_calls is not None and policy_refill_calls is not None:
        if policy_refill_calls > policy_loop_tick_calls:
            raise ValueError(
                "native policy-cache policy_refill_calls cannot exceed policy_loop_tick_calls"
            )
        expected_hit_calls = policy_loop_tick_calls - policy_refill_calls
        if policy_cache_hit_calls is not None and policy_cache_hit_calls != expected_hit_calls:
            raise ValueError(
                "native policy-cache tick counters violate "
                "policy_cache_hit_calls=policy_loop_tick_calls-policy_refill_calls: "
                f"hit_calls={policy_cache_hit_calls} loop_calls={policy_loop_tick_calls} "
                f"refill_calls={policy_refill_calls}"
            )
        policy_cache_hit_calls = expected_hit_calls

    row_identity_available = requested is not None and hit is not None and missed is not None
    if one_tick and not (full_present or partial_present) and policy_refill_calls is not None:
        if policy_refill_calls > 1:
            raise ValueError("one-tick native policy-cache result reports more than one refill call")
        if policy_refill_calls == 0:
            full_refill_ticks = 0
            partial_refill_ticks = 0
        elif row_identity_available:
            if missed <= 0:
                raise ValueError("one-tick native policy-cache refill call has no missed rows")
            full_refill_ticks = int(hit == 0)
            partial_refill_ticks = int(hit > 0)

    if one_tick and row_identity_available and full_refill_ticks is not None and partial_refill_ticks is not None:
        expected_full = int(missed > 0 and hit == 0)
        expected_partial = int(missed > 0 and hit > 0)
        if (full_refill_ticks, partial_refill_ticks) != (expected_full, expected_partial):
            raise ValueError(
                "one-tick native policy-cache refill classification disagrees with row counters"
            )

    out: dict[str, int] = {}
    for key, value in (
        ("policy_cache_rows_requested", requested),
        ("policy_cache_rows_hit", hit),
        ("policy_cache_rows_missed", missed),
        ("policy_cache_rows_refilled", refilled),
        ("policy_cache_full_refill_ticks", full_refill_ticks),
        ("policy_cache_partial_refill_ticks", partial_refill_ticks),
        ("policy_loop_tick_calls", policy_loop_tick_calls),
        ("policy_refill_calls", policy_refill_calls),
        ("policy_cache_hit_calls", policy_cache_hit_calls),
    ):
        if value is not None:
            out[key] = int(value)
    if refilled is not None:
        out["policy_refill_rows"] = int(refilled)
    if hit is not None:
        out["cache_hit_rows"] = int(hit)
    return out


def _policy_cache_one_tick_result(source: Mapping[str, Any]) -> bool:
    for key in ("policy_loop_tick_calls", "native_tick_loop_count", "steps_per_generation"):
        value = _policy_cache_count(source, key)
        if value is not None:
            return value == 1
    schema = str(_get(source, "raw_result_schema_version", "") or _get(source, "schema_version", ""))
    return "action_cache_service" in schema


def _policy_cache_count(source: Mapping[str, Any], key: str) -> int | None:
    if key not in source or source[key] is None:
        return None
    try:
        value = int(source[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"native policy-cache counter {key} is not an integer") from exc
    if value < 0:
        raise ValueError(f"native policy-cache counter {key} cannot be negative")
    return value


def _matching_policy_cache_count(source: Mapping[str, Any], *keys: str) -> int | None:
    present = [(key, _policy_cache_count(source, key)) for key in keys if key in source and source[key] is not None]
    if not present:
        return None
    expected = int(present[0][1])
    for key, value in present[1:]:
        if value != expected:
            names = ", ".join(f"{name}={count}" for name, count in present)
            raise ValueError(f"native policy-cache legacy/canonical counters disagree: {names}")
    return expected


def _merge_policy_cache_counts(
    target: dict[str, int],
    incoming: Mapping[str, int],
    *,
    context: str,
) -> None:
    for key, value in incoming.items():
        count = int(value)
        if key in target and int(target[key]) != count:
            raise ValueError(
                f"native policy-cache counters disagree across {context}: "
                f"{key}={target[key]} versus {count}"
            )
        target[key] = count


def public_metrics_overlay(
    raw_result: Any | None = None,
    compact_result: Any | None = None,
    checkpoint_projection: Any | None = None,
    *,
    sim_cfg: Any | None = None,
    env_cfg: Any | None = None,
    metrics_generation: int | None = None,
) -> dict[str, Any]:
    """Build a side-effect-free metrics.csv overlay from native public outputs.

    The returned keys are canonical ``metrics.csv`` column names.  Values are
    emitted only when they can be derived from supplied native behavior rows,
    compact/public counters, or final live-row tensors.  Generation-time
    telemetry such as death age histograms and juvenile/adult exposure/death
    counts is intentionally not reconstructed here.
    """

    raw = _mapping(raw_result)
    compact = _mapping(compact_result)
    projection = _mapping(checkpoint_projection)

    if not raw and not compact and not projection:
        raw = _mapping(_get(raw_result, "raw_result"))
        compact = _mapping(_get(raw_result, "compact_result"))
        projection = _mapping(_get(raw_result, "checkpoint_projection"))

    if not compact:
        compact = _mapping(_get(raw_result, "compact_result"))
    if not projection:
        projection = _mapping(_get(raw_result, "checkpoint_projection"))

    tensors = _mapping(_get(projection, "tensors"))
    metadata = _mapping(_get(projection, "metadata"))
    scope_locked_flat = _cfg_bool(
        sim_cfg,
        "native_runtime_cxx26_scope_lock_enable",
        False,
    )
    if scope_locked_flat:
        from .flat_population_reporting import (
            validate_native_population_reporting_envelope,
            validated_native_flat_behavior_rows,
        )

        reporting_world = validate_native_population_reporting_envelope(
            sim_cfg=sim_cfg,
            env_cfg=env_cfg,
            metadata=metadata,
        )
        if reporting_world == "patch":
            _validate_patch_reporting_not_promoted(raw)

        expected_generation = _scope_locked_flat_reporting_generation(
            metadata,
            metrics_generation=metrics_generation,
        )
        behavior_rows = list(
            validated_native_flat_behavior_rows(
                _get(raw, "behavior_rows"),
                expected_generation=expected_generation,
            )
        )
    else:
        behavior_rows = _behavior_rows(_get(raw, "behavior_rows"))

    out: dict[str, Any] = {}
    _add_public_counts(out, raw=raw, compact=compact)
    out.update(native_policy_cache_metrics(raw, compact))
    _add_behavior_metrics(out, behavior_rows)
    live_slots, aid_to_slot = _live_slots(tensors)
    _add_final_tensor_metrics(
        out,
        tensors=tensors,
        live_slots=live_slots,
        aid_to_slot=aid_to_slot,
        behavior_rows=behavior_rows,
        sim_cfg=sim_cfg,
        env_cfg=env_cfg,
        metadata=metadata,
        metrics_generation=metrics_generation,
    )
    _add_flat_population_reporting_metrics(
        out,
        tensors=tensors,
        behavior_rows=behavior_rows,
        sim_cfg=sim_cfg,
        env_cfg=env_cfg,
        metadata=metadata,
        metrics_generation=metrics_generation,
    )
    lineage_metrics_requested = _cfg_bool(sim_cfg, "lineage_observability_enable", True)
    lineage_input: dict[str, Any] | None = None
    if lineage_metrics_requested and (sim_cfg is not None or "lineage_founder_id" in tensors):
        lineage_input = _native_lineage_metrics_input(
            raw=raw,
            tensors=tensors,
            metadata=metadata,
            behavior_rows=behavior_rows,
            aid_to_slot=aid_to_slot,
            sim_cfg=sim_cfg,
            env_cfg=env_cfg,
        )
        out[NATIVE_LINEAGE_METRICS_INPUT_KEY] = lineage_input
    if "birth_attempts" in out and "successful_births" in out and "first_repro_frac_alive" in out:
        out["behavior_reproduction_proxy"] = float(
            float(out["birth_attempts"])
            + float(out["successful_births"])
            + float(out["first_repro_frac_alive"])
        )
    if scope_locked_flat and str(metadata.get("world", "") or "").lower() == "patch":
        if lineage_input is None:
            raise ValueError("native Patch reporting requires lineage metrics input")
        from .patch_public_reporting import native_patch_public_metrics

        out.update(
            native_patch_public_metrics(
                raw_result=raw,
                checkpoint_projection=projection,
                lineage_input=lineage_input,
                existing_metrics=out,
                env_cfg=env_cfg,
                metrics_generation=_scope_locked_flat_reporting_generation(
                    metadata,
                    metrics_generation=metrics_generation,
                ),
            )
        )
    return out


native_public_metrics_overlay = public_metrics_overlay
build_public_metrics_overlay = public_metrics_overlay


def _add_flat_population_reporting_metrics(
    out: dict[str, Any],
    *,
    tensors: Mapping[str, Any],
    behavior_rows: list[dict[str, Any]],
    sim_cfg: Any | None,
    env_cfg: Any | None,
    metadata: Mapping[str, Any],
    metrics_generation: int | None,
) -> None:
    if not _cfg_bool(sim_cfg, "native_runtime_cxx26_scope_lock_enable", False):
        return

    # Local imports keep the native adapter and shared metrics package from
    # becoming import-time dependencies of each other.
    from .flat_population_reporting import native_flat_population_reporting_records
    from simulation.metrics_pkg.cognitive_status import compute_cognitive_status_metrics_from_records

    generation = _scope_locked_flat_reporting_generation(
        metadata,
        metrics_generation=metrics_generation,
    )
    from simulation.output_cost_timing import reporting_projection_phase, reporting_reducer_phase

    with reporting_projection_phase():
        records = native_flat_population_reporting_records(
            tensors=tensors,
            behavior_rows=behavior_rows,
            sim_cfg=sim_cfg,
            env_cfg=env_cfg,
            metadata=metadata,
            reporting_generation=generation,
        )
    with reporting_reducer_phase():
        reduced = compute_cognitive_status_metrics_from_records(
            records,
            env_cfg=env_cfg,
            sim_cfg=sim_cfg,
            generation=generation,
        )
    if not isinstance(reduced, Mapping):
        raise ValueError("native Flat resident reporting reducer did not return a mapping")
    missing = tuple(name for name in NATIVE_FLAT_RESIDENT_REPORTING_COLUMNS if name not in reduced)
    if missing:
        raise ValueError(f"native Flat resident reporting reducer is missing required columns: {missing}")
    # Publish the complete canonical reducer surface. Limiting this handoff to
    # the originally stale 66 columns would leave other Agent-derived metrics
    # dependent on the intentionally stale Python population.
    out.update(reduced)


def _validate_patch_reporting_not_promoted(raw: Mapping[str, Any]) -> None:
    """Keep Patch public reporting diagnostic-only until its promotion gate opens."""

    if raw.get("promotion_accepted") is not False:
        raise ValueError(
            "native Patch population reporting requires promotion_accepted=false"
        )


def _add_public_counts(out: dict[str, Any], *, raw: Mapping[str, Any], compact: Mapping[str, Any]) -> None:
    source = compact or raw
    count_map = {
        "alive": ("alive_count", "alive"),
        "successful_births": ("committed_birth_rows", "successful_births"),
        "failed_births": ("public_failed_birth_rows", "failed_birth_rows", "failed_births"),
        "birth_attempts": ("birth_attempt_rows", "birth_attempts"),
        "birth_pre_cooldown": ("birth_pre_cooldown",),
        "birth_pre_no_energy": ("birth_pre_no_energy",),
        "birth_pre_injury_gate": ("birth_pre_injury_gate",),
        "birth_pre_no_space": ("birth_pre_no_space",),
        "birth_pre_no_mate": ("birth_pre_no_mate",),
        "birth_fail_no_space": ("birth_fail_no_space",),
        "birth_fail_no_mate": ("birth_fail_no_mate",),
        "birth_fail_buffer": ("birth_fail_buffer",),
        "birth_fail_injury_gate": ("birth_fail_injury_gate",),
        "birth_fail_scar_gate": ("birth_fail_scar_gate",),
        "birth_attempt_conception_fail": ("birth_fail_conception",),
        "birth_fail_conception": ("birth_fail_conception",),
        "birth_gestation_fail": ("birth_fail_gestation",),
        "birth_fail_gestation": ("birth_fail_gestation",),
        "dev_stage_transition_count": ("dev_stage_transition_count",),
    }
    for column, names in count_map.items():
        value = _first_present(source, names)
        if value is not None:
            out[column] = max(0, _safe_int(value))

    injury_blocked = _first_present(source, ("injury_blocked_births",))
    if injury_blocked is not None:
        out["injury_blocked_births"] = max(0, _safe_int(injury_blocked))

    peak_injury = _first_present(source, ("peak_injury",))
    if peak_injury is not None:
        peak_value = _finite_float(peak_injury)
        if peak_value is not None:
            out["peak_injury"] = max(0.0, peak_value)

    if "dev_stage_transition_count" in out:
        transitions = int(out["dev_stage_transition_count"])
        out["dev_stage_eval_stage_counts_json"] = (
            json.dumps({"default": transitions}, separators=(",", ":")) if transitions > 0 else "{}"
        )


def _add_behavior_metrics(out: dict[str, Any], behavior_rows: list[dict[str, Any]]) -> None:
    if not behavior_rows:
        return

    cohort_rows = [row for row in behavior_rows if str(row.get("cohort", "") or "") == "start"]
    if cohort_rows:
        cohort_size = int(len(cohort_rows))
        cohort_survivors = int(sum(_safe_int(row.get("alive_end")) > 0 for row in cohort_rows))
        out["cohort_size"] = cohort_size
        out["cohort_survivors"] = cohort_survivors
        out["cohort_survival_rate"] = float(cohort_survivors) / float(cohort_size)

    action_totals = {
        "behavior_move_n_total": sum(_safe_int(row.get("move_n")) for row in behavior_rows),
        "behavior_move_s_total": sum(_safe_int(row.get("move_s")) for row in behavior_rows),
        "behavior_move_e_total": sum(_safe_int(row.get("move_e")) for row in behavior_rows),
        "behavior_move_w_total": sum(_safe_int(row.get("move_w")) for row in behavior_rows),
        "behavior_gather_total": sum(_safe_int(row.get("gather")) for row in behavior_rows),
        "behavior_rest_total": sum(_safe_int(row.get("rest")) for row in behavior_rows),
        "behavior_attack_total": sum(_safe_int(row.get("attack")) for row in behavior_rows),
    }
    action_totals["behavior_move_total"] = (
        action_totals["behavior_move_n_total"]
        + action_totals["behavior_move_s_total"]
        + action_totals["behavior_move_e_total"]
        + action_totals["behavior_move_w_total"]
    )
    action_totals["behavior_action_total"] = (
        action_totals["behavior_move_total"]
        + action_totals["behavior_gather_total"]
        + action_totals["behavior_rest_total"]
        + action_totals["behavior_attack_total"]
    )
    denom = float(max(1, action_totals["behavior_action_total"]))
    out.update({key: float(value) for key, value in action_totals.items()})
    out["behavior_move_fraction"] = float(action_totals["behavior_move_total"]) / denom
    out["behavior_gather_fraction"] = float(action_totals["behavior_gather_total"]) / denom
    out["behavior_rest_fraction"] = float(action_totals["behavior_rest_total"]) / denom
    out["behavior_attack_fraction"] = float(action_totals["behavior_attack_total"]) / denom

    live_rows = [row for row in behavior_rows if _safe_int(row.get("alive_end")) > 0]
    if live_rows:
        out["alive"] = int(len(live_rows))
        _add_distribution(out, "fitness", _row_floats(live_rows, "fitness"), mean_key="mean_fitness", std_key="std_fitness", p90_key="p90_fitness", max_key="max_fitness")
        _add_mean(out, "mean_survival_steps_total", _row_floats(live_rows, "lifespan_steps"))
        _add_distribution(out, "energy_collected", _row_floats(live_rows, "energy_collected"), mean_key="mean_energy_collected_total", std_key="std_energy_collected_total")
        _add_mean(out, "mean_offspring_total", _row_floats(live_rows, "offspring"))
        _add_mean(out, "mean_attacks_total", _row_floats(live_rows, "attacks_made"))
        aggression = _row_floats(live_rows, "aggression_bias")
        if aggression:
            out["mean_aggression_bias"] = float(mean(aggression))
            out["std_aggression_bias"] = float(std(aggression))
            out["p10_aggression_bias"] = float(percentile(aggression, 10))
            out["p90_aggression_bias"] = float(percentile(aggression, 90))
        injuries = _row_floats(live_rows, "injury_end")
        if injuries and not {"mean_injury", "p90_injury", "max_injury"} <= set(out):
            out["mean_injury"] = float(mean(injuries))
            out["p90_injury"] = float(percentile(injuries, 90))
            out["max_injury"] = float(max(injuries))

    first_repro = _row_floats(live_rows, "first_repro_step") if live_rows else []
    if first_repro:
        out["first_repro_step_mean_alive"] = float(mean(first_repro))
        out["first_repro_frac_alive"] = float(len(first_repro)) / float(max(1, len(live_rows)))
        out["tier4_fraction_reproduced_alive_end"] = out["first_repro_frac_alive"]
        out["tier4_first_repro_step_mean_reproducers_alive_end"] = out["first_repro_step_mean_alive"]
        out["tier4_first_repro_step_p50_reproducers_alive_end"] = _tier4_reproduction_percentile(first_repro, 50)
        out["tier4_first_repro_step_p90_reproducers_alive_end"] = _tier4_reproduction_percentile(first_repro, 90)


def _add_final_tensor_metrics(
    out: dict[str, Any],
    *,
    tensors: Mapping[str, Any],
    live_slots: list[int],
    aid_to_slot: dict[int, int],
    behavior_rows: list[dict[str, Any]],
    sim_cfg: Any | None,
    env_cfg: Any | None,
    metadata: Mapping[str, Any],
    metrics_generation: int | None = None,
) -> None:
    if not live_slots:
        return

    out["alive"] = int(len(live_slots))
    _fill_mean_if_absent(out, "mean_survival_steps_total", _tensor_live_floats(tensors, "survival_steps", live_slots))
    energy_collected_field = (
        "energy_collected_exact"
        if "energy_collected_exact" in tensors
        else "energy_collected"
    )
    _fill_distribution_if_absent(
        out,
        _tensor_live_floats(tensors, energy_collected_field, live_slots),
        mean_key="mean_energy_collected_total",
        std_key="std_energy_collected_total",
    )
    _fill_mean_if_absent(out, "mean_offspring_total", _tensor_live_floats(tensors, "offspring_count", live_slots))
    _fill_mean_if_absent(out, "mean_attacks_total", _tensor_live_floats(tensors, "attacks", live_slots))

    aggression = _tensor_live_floats(tensors, "genome_aggression_bias", live_slots)
    if aggression:
        out.setdefault("mean_aggression_bias", float(mean(aggression)))
        out.setdefault("std_aggression_bias", float(std(aggression)))
        out.setdefault("p10_aggression_bias", float(percentile(aggression, 10)))
        out.setdefault("p90_aggression_bias", float(percentile(aggression, 90)))

    injuries = _tensor_live_floats(tensors, "injury", live_slots)
    scars = _tensor_live_floats(tensors, "scar", live_slots)
    if injuries:
        out["mean_injury"] = float(mean(injuries))
        out["p90_injury"] = float(percentile(injuries, 90))
        out["max_injury"] = float(max(injuries))
    if scars:
        out["mean_scar"] = float(mean(scars))
        out["p90_scar"] = float(percentile(scars, 90))
        out["max_scar"] = float(max(scars))
    if injuries and scars:
        # MetricsLogger's public contract defaults this argument to 1.0; the
        # environment's action-cost scar weight is a separate semantic.
        scar_weight = 1.0
        impair = [float(injuries[i]) + scar_weight * float(scars[i]) for i in range(min(len(injuries), len(scars)))]
        _fill_mean_if_absent(out, "mean_impairment", impair)

    _add_chemistry(out, tensors=tensors, live_slots=live_slots, aid_to_slot=aid_to_slot, behavior_rows=behavior_rows, sim_cfg=sim_cfg)
    _add_biome_fractions(
        out,
        tensors=tensors,
        live_slots=live_slots,
        fail_closed=_cfg_bool(
            sim_cfg,
            "native_runtime_cxx26_scope_lock_enable",
            False,
        ),
    )
    _add_selected_parent_genome_metrics(
        out,
        tensors=tensors,
        aid_to_slot=aid_to_slot,
        behavior_rows=behavior_rows,
    )
    _add_early_lineage_metrics(
        out,
        tensors=tensors,
        live_slots=live_slots,
        metadata=metadata,
        metrics_generation=metrics_generation,
    )
    _add_drives(out, tensors=tensors, live_slots=live_slots)
    _add_coherence(
        out,
        tensors=tensors,
        live_slots=live_slots,
        aid_to_slot=aid_to_slot,
        behavior_rows=behavior_rows,
        sim_cfg=sim_cfg,
        metadata=metadata,
    )
    _add_memory_buckets(out, tensors=tensors, live_slots=live_slots)
    _add_first_repro_from_tensor(out, tensors=tensors, live_slots=live_slots)
    _add_alive_age_hist(out, tensors=tensors, live_slots=live_slots, env_cfg=env_cfg)


def _native_lineage_metrics_input(
    *,
    raw: Mapping[str, Any],
    tensors: Mapping[str, Any],
    metadata: Mapping[str, Any],
    behavior_rows: list[dict[str, Any]],
    aid_to_slot: Mapping[int, int],
    sim_cfg: Any,
    env_cfg: Any,
) -> dict[str, Any]:
    world = str(metadata.get("world", "") or "").strip().lower()
    if world == "patch":
        from .flat_population_reporting import validate_native_population_reporting_envelope

        validate_native_population_reporting_envelope(
            sim_cfg=sim_cfg,
            env_cfg=env_cfg,
            metadata=metadata,
        )
    elif world != "flat":
        raise ValueError(
            "native lineage metrics input requires an admitted Flat or Patch projection, "
            f"got {world or 'missing'}"
        )
    policy_family = str(metadata.get("native_runtime_fixed_policy_family", "") or "").strip().lower()
    if policy_family != "mlp":
        raise ValueError(
            "native lineage metrics input requires the accepted fixed-MLP reporting envelope: "
            f"native_runtime_fixed_policy_family={policy_family or 'missing'}"
        )
    missing = [name for name in _NATIVE_LINEAGE_REQUIRED_TENSORS if name not in tensors]
    if missing:
        raise ValueError(f"native lineage metrics projection is missing required tensors: {missing}")
    arrays = {name: _tensor_array(tensors[name]) for name in _NATIVE_LINEAGE_REQUIRED_TENSORS}
    if world == "patch":
        if "patch_habitat_lineage_id" not in tensors:
            raise ValueError(
                "native Patch lineage metrics projection is missing patch_habitat_lineage_id"
            )
        arrays["patch_habitat_lineage_id"] = _tensor_array(
            tensors["patch_habitat_lineage_id"]
        )
    if "energy_collected_exact" in tensors:
        arrays["energy_collected_exact"] = _tensor_array(tensors["energy_collected_exact"])
    invalid = [name for name, value in arrays.items() if value is None]
    if invalid:
        raise ValueError(f"native lineage metrics projection has unreadable tensors: {invalid}")
    if not behavior_rows:
        raise ValueError("native lineage metrics input requires native behavior rows")

    generations = {_safe_int(row.get("generation"), default=-1) for row in behavior_rows}
    if len(generations) != 1 or next(iter(generations)) < 0:
        raise ValueError(f"native lineage metrics behavior rows have inconsistent generations: {sorted(generations)}")
    generation = next(iter(generations))
    projection_generation = _safe_int(metadata.get("generation"), default=-1)
    if projection_generation not in {generation, generation + 1}:
        raise ValueError(
            "native lineage metrics projection/behavior generation mismatch: "
            f"behavior={generation} projection={projection_generation}"
        )

    live_rows = [row for row in behavior_rows if _safe_int(row.get("alive_end")) > 0]
    live_ids = [_safe_int(row.get("agent_id"), default=-1) for row in live_rows]
    if any(agent_id < 0 for agent_id in live_ids) or len(live_ids) != len(set(live_ids)):
        raise ValueError("native lineage metrics behavior rows have invalid or duplicate live agent IDs")

    records = []
    for row in sorted(live_rows, key=lambda item: _safe_int(item.get("agent_id"), default=-1)):
        agent_id = _safe_int(row.get("agent_id"), default=-1)
        slot = aid_to_slot.get(agent_id)
        if slot is not None:
            records.append(
                _native_lineage_projection_record(
                    row=row,
                    slot=int(slot),
                    arrays=arrays,
                    tensors=tensors,
                )
            )
            continue
        if str(row.get("cohort", "") or "") != "born":
            raise ValueError(
                "native lineage metrics live start-cohort agent is absent from the immutable projection: "
                f"agent_id={agent_id}"
            )
        records.append(
            _native_lineage_birth_record(
                row=row,
                raw=raw,
                tensors=tensors,
                metadata=metadata,
                arrays=arrays,
                aid_to_slot=aid_to_slot,
                generation=generation,
            )
        )

    agent_ids = [int(record["agent_id"]) for record in records]
    payload = {
        "schema_version": NATIVE_LINEAGE_METRICS_INPUT_SCHEMA_VERSION,
        "source": "native_public_outputs_and_single_checkpoint_projection",
        "world": world,
        "generation": generation,
        "projection_generation": projection_generation,
        "record_count": len(records),
        "agent_ids": agent_ids,
        "records": records,
    }
    if world == "patch":
        payload.update(
            {
                "patch_profile": metadata.get("native_runtime_patch_profile"),
                "patch_profile_scope": metadata.get("native_runtime_patch_profile_scope"),
                "patch_contract_schema_version": metadata.get(
                    "native_runtime_patch_contract_schema_version"
                ),
                "promotion_accepted": raw.get("promotion_accepted"),
            }
        )
    return payload


def _native_lineage_projection_record(
    *,
    row: Mapping[str, Any],
    slot: int,
    arrays: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> dict[str, Any]:
    from agents.modular_genome_v1 import canon_float

    agent_id = _safe_int(row.get("agent_id"), default=-1)
    projected_id = _native_slot_int(arrays, "slot_to_agent_id", slot)
    if projected_id != agent_id or not _truthy(_native_slot_value(arrays, "alive_mask", slot)):
        raise ValueError(
            "native lineage metrics slot identity mismatch: "
            f"agent_id={agent_id} slot={slot} projected_id={projected_id}"
        )
    founder_id = _native_slot_int(arrays, "lineage_founder_id", slot)
    if founder_id < 0:
        raise ValueError(f"native lineage metrics projection has invalid founder for agent_id={agent_id}")

    survival_steps = _safe_int(row.get("lifespan_steps"), default=-1)
    offspring_count = _safe_int(row.get("offspring"), default=-1)
    energy_collected = _lineage_finite(row.get("energy_collected"), context=f"behavior agent {agent_id} energy_collected")
    injury = _native_slot_float(arrays, "injury", slot)
    aggression = canon_float(
        _lineage_finite(row.get("aggression_bias"), context=f"behavior agent {agent_id} aggression_bias"),
        field="aggression_bias",
    )
    if survival_steps != _native_slot_int(arrays, "survival_steps", slot):
        raise ValueError(f"native lineage metrics survival_steps disagree for agent_id={agent_id}")
    if offspring_count != _native_slot_int(arrays, "offspring_count", slot):
        raise ValueError(f"native lineage metrics offspring_count disagree for agent_id={agent_id}")
    projected_energy_field = (
        "energy_collected_exact"
        if "energy_collected_exact" in arrays
        else "energy_collected"
    )
    if not _lineage_close(
        energy_collected,
        _native_slot_float(arrays, projected_energy_field, slot),
    ):
        raise ValueError(f"native lineage metrics energy_collected disagrees for agent_id={agent_id}")
    if not _lineage_close(
        _lineage_finite(row.get("injury_end"), context=f"behavior agent {agent_id} injury_end"),
        injury,
    ):
        raise ValueError(f"native lineage metrics injury disagrees for agent_id={agent_id}")
    projected_aggression = canon_float(
        _native_slot_float(arrays, "genome_aggression_bias", slot),
        field="aggression_bias",
    )
    if aggression != projected_aggression:
        raise ValueError(f"native lineage metrics aggression_bias disagrees for agent_id={agent_id}")

    first_repro_raw = _native_slot_int(arrays, "first_repro_step", slot)
    episode0_count = _native_slot_int(arrays, "episode0_pred_err_count", slot)
    episode0_sum = _native_slot_float(arrays, "episode0_pred_err_sum", slot)
    episode0_mean = episode0_sum / float(episode0_count) if episode0_count > 0 else 0.0
    identity_drift = (
        _native_tensor_slot_float(tensors, "identity_drift_ema", slot)
        if "identity_drift_ema" in tensors
        else 0.0
    )
    return _native_lineage_record(
        agent_id=agent_id,
        founder_id=founder_id,
        lineage_depth=_native_slot_int(arrays, "lineage_depth", slot),
        birth_generation=_native_slot_int(arrays, "lineage_birth_generation", slot),
        birth_tick=_native_slot_int(arrays, "lineage_birth_tick", slot),
        parent_ids=(),
        # Reference Agent.energy is maintained at Python float64 precision. The
        # mechanics tensor is float32, while this shadow preserves that public value.
        energy=_native_slot_float(arrays, "policy_energy_observation", slot),
        health=_native_slot_float(arrays, "health", slot),
        injury=injury,
        drives=_native_slot_vector(arrays, "drives", slot, expected=4),
        offspring_count=offspring_count,
        survival_steps=survival_steps,
        energy_collected=energy_collected,
        first_repro_step=None if first_repro_raw < 0 else first_repro_raw,
        pred_err_count=_native_slot_int(arrays, "pred_err_count", slot),
        pred_err_sum=_native_slot_float(arrays, "pred_err_sum", slot),
        episode0_pred_err_mean=episode0_mean,
        identity_drift_ema=identity_drift,
        drive_jitter=_native_slot_float(arrays, "drive_jitter", slot),
        hidden_size=_native_slot_int(arrays, "genome_hidden_size", slot),
        memory_dim=_native_slot_int(arrays, "genome_memory_dim", slot),
        use_gru=_truthy(_native_slot_value(arrays, "policy_use_gru", slot)),
        aggression_bias=aggression,
        exploration_bias=canon_float(
            _native_slot_float(arrays, "genome_exploration_bias", slot),
            field="exploration_bias",
        ),
        patch_habitat_lineage_id=_native_slot_int(
            arrays,
            "patch_habitat_lineage_id",
            slot,
        )
        if "patch_habitat_lineage_id" in arrays
        else 0,
        y=_native_slot_int(arrays, "pos_y", slot),
        x=_native_slot_int(arrays, "pos_x", slot),
    )


def _native_lineage_birth_record(
    *,
    row: Mapping[str, Any],
    raw: Mapping[str, Any],
    tensors: Mapping[str, Any],
    metadata: Mapping[str, Any],
    arrays: Mapping[str, Any],
    aid_to_slot: Mapping[int, int],
    generation: int,
) -> dict[str, Any]:
    from agents.modular_genome_v1 import canon_float

    child_id = _safe_int(row.get("agent_id"), default=-1)
    birth = _native_birth_public_context(raw, child_id=child_id)
    parent_id = int(birth["parent_id"])
    mate_id = int(birth["mate_id"])
    parent_slot = aid_to_slot.get(parent_id)
    mate_slot = aid_to_slot.get(mate_id if mate_id >= 0 else parent_id)
    if parent_slot is None or mate_slot is None:
        raise ValueError(
            "native lineage metrics cannot resolve projected parents for pre-cull child: "
            f"child_id={child_id} parent_id={parent_id} mate_id={mate_id}"
        )
    lifespan = _safe_int(row.get("lifespan_steps"), default=-1)
    offspring = _safe_int(row.get("offspring"), default=-1)
    if lifespan != 0 or offspring != 0:
        raise ValueError(
            "native lineage metrics cannot reconstruct a non-projected child after it accumulated state: "
            f"child_id={child_id} lifespan_steps={lifespan} offspring={offspring}"
        )

    event = _native_birth_event_context(
        tensors=tensors,
        metadata=metadata,
        child_id=child_id,
        generation=generation,
    )
    if int(event["parent_id"]) >= 0 and int(event["parent_id"]) != parent_id:
        raise ValueError(f"native lineage metrics birth parent disagrees for child_id={child_id}")
    parent_founder = _native_slot_int(arrays, "lineage_founder_id", int(parent_slot))
    mate_founder = _native_slot_int(arrays, "lineage_founder_id", int(mate_slot))
    founder_id = int(event["founder_id"])
    if founder_id < 0:
        founder_id = min(parent_founder, mate_founder)
    lineage_depth = int(event["lineage_depth"])
    if lineage_depth < 0:
        lineage_depth = max(
            _native_slot_int(arrays, "lineage_depth", int(parent_slot)),
            _native_slot_int(arrays, "lineage_depth", int(mate_slot)),
        ) + 1

    genome = _native_child_genome(
        raw=raw,
        row=row,
        child_id=child_id,
        child_slot=int(birth["child_slot"]),
        parent_slot=int(parent_slot),
        mate_slot=int(mate_slot),
        tensors=tensors,
        metadata=metadata,
    )
    aggression = canon_float(
        _lineage_finite(row.get("aggression_bias"), context=f"behavior child {child_id} aggression_bias"),
        field="aggression_bias",
    )
    if aggression != genome["aggression_bias"]:
        raise ValueError(f"native lineage metrics child aggression trace disagrees for child_id={child_id}")
    drives = tuple(float(value) for value in birth["drives"])
    position = birth["position"]
    patch_habitat_lineage_id = 0
    if "patch_habitat_lineage_id" in arrays:
        parent_patch_lineage = _native_slot_int(
            arrays, "patch_habitat_lineage_id", int(parent_slot)
        )
        mate_patch_lineage = _native_slot_int(
            arrays, "patch_habitat_lineage_id", int(mate_slot)
        )
        patch_habitat_lineage_id = min(
            parent_patch_lineage if parent_patch_lineage > 0 else parent_id,
            mate_patch_lineage
            if mate_patch_lineage > 0
            else (mate_id if mate_id >= 0 else parent_id),
        )
    return _native_lineage_record(
        agent_id=child_id,
        founder_id=founder_id,
        lineage_depth=lineage_depth,
        birth_generation=int(event["generation"]),
        birth_tick=int(event["tick"]),
        parent_ids=(parent_id,) if mate_id < 0 or mate_id == parent_id else (parent_id, mate_id),
        energy=float(event["energy"]),
        health=_lineage_finite(metadata.get("start_health"), context="native lineage metadata start_health"),
        injury=_lineage_finite(row.get("injury_end"), context=f"behavior child {child_id} injury_end"),
        drives=drives,
        offspring_count=offspring,
        survival_steps=lifespan,
        energy_collected=_lineage_finite(
            row.get("energy_collected"),
            context=f"behavior child {child_id} energy_collected",
        ),
        first_repro_step=None,
        pred_err_count=0,
        pred_err_sum=0.0,
        episode0_pred_err_mean=0.0,
        identity_drift_ema=0.0,
        drive_jitter=0.0,
        hidden_size=int(genome["hidden_size"]),
        memory_dim=int(genome["memory_dim"]),
        use_gru=bool(genome["use_gru"]),
        aggression_bias=float(genome["aggression_bias"]),
        exploration_bias=float(genome["exploration_bias"]),
        patch_habitat_lineage_id=patch_habitat_lineage_id,
        y=int(position[0]),
        x=int(position[1]),
    )


def _native_lineage_record(
    *,
    agent_id: int,
    founder_id: int,
    lineage_depth: int,
    birth_generation: int,
    birth_tick: int,
    parent_ids: Sequence[int],
    energy: float,
    health: float,
    injury: float,
    drives: Sequence[float],
    offspring_count: int,
    survival_steps: float,
    energy_collected: float,
    first_repro_step: int | None,
    pred_err_count: int,
    pred_err_sum: float,
    episode0_pred_err_mean: float,
    identity_drift_ema: float,
    drive_jitter: float,
    hidden_size: int,
    memory_dim: int,
    use_gru: bool,
    aggression_bias: float,
    exploration_bias: float,
    patch_habitat_lineage_id: int,
    y: int,
    x: int,
) -> dict[str, Any]:
    lineage_id = f"L{max(0, int(founder_id)):08d}"
    return {
        "agent_id": int(agent_id),
        "energy": float(energy),
        "health": float(health),
        "injury": float(injury),
        "drives": [float(value) for value in drives],
        "y": int(y),
        "x": int(x),
        "stats": {
            "lineage_id": lineage_id,
            "founder_id": int(founder_id),
            "parent_ids": [int(value) for value in parent_ids],
            "parent_lineage_ids": [],
            "birth_generation": int(birth_generation),
            "birth_tick": int(birth_tick),
            "lineage_depth": int(lineage_depth),
            "offspring_count": int(offspring_count),
            "survival_steps": float(survival_steps),
            "energy_collected": float(energy_collected),
            "first_repro_step": None if first_repro_step is None else int(first_repro_step),
            "pred_err_count": int(pred_err_count),
            "pred_err_sum": float(pred_err_sum),
            "episode0_pred_err_mean": float(episode0_pred_err_mean),
            "identity_drift_ema": float(identity_drift_ema),
            "drive_jitter": float(drive_jitter),
            "ecological_role": "general_agent",
            "role_source": "default",
            "role_confidence": None,
            "patch_habitat_lineage_id": int(patch_habitat_lineage_id),
        },
        "genome": {
            "hidden_size": int(hidden_size),
            "memory_dim": int(memory_dim),
            "use_gru": bool(use_gru),
            "aggression_bias": float(aggression_bias),
            "exploration_bias": float(exploration_bias),
        },
    }


def _native_birth_public_context(raw: Mapping[str, Any], *, child_id: int) -> dict[str, Any]:
    child_ids = _native_sequence(raw.get("child_agent_ids"), context="native child_agent_ids")
    matches = [index for index, value in enumerate(child_ids) if _safe_int(value, default=-1) == int(child_id)]
    if len(matches) != 1:
        raise ValueError(f"native lineage metrics expected one birth public row for child_id={child_id}")
    index = matches[0]
    parent_ids = _native_sequence(raw.get("parent_agent_ids"), context="native parent_agent_ids")
    mate_ids = _native_sequence(raw.get("mate_agent_ids"), context="native mate_agent_ids")
    child_slots = _native_sequence(raw.get("child_slots"), context="native child_slots")
    child_positions = _native_sequence(raw.get("child_positions"), context="native child_positions")
    for name, values in (
        ("parent_agent_ids", parent_ids),
        ("mate_agent_ids", mate_ids),
        ("child_slots", child_slots),
        ("child_positions", child_positions),
    ):
        if index >= len(values):
            raise ValueError(f"native lineage metrics {name} is incomplete for child_id={child_id}")
    drive_rows = _native_sequence(raw.get("child_drive_rows"), context="native child_drive_rows")
    matching_drives = [
        row
        for row in drive_rows
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes)) and len(row) >= 6
        and _safe_int(row[0], default=-1) == int(child_id)
    ]
    if len(matching_drives) != 1:
        raise ValueError(f"native lineage metrics expected one child drive row for child_id={child_id}")
    drive_row = matching_drives[0]
    if _safe_int(drive_row[1], default=-1) != _safe_int(child_slots[index], default=-2):
        raise ValueError(f"native lineage metrics child slot/drive row disagree for child_id={child_id}")
    drives = tuple(
        _lineage_finite(drive_row[offset], context=f"native child {child_id} drive[{offset - 2}]")
        for offset in range(2, 6)
    )
    position = child_positions[index]
    if not isinstance(position, Sequence) or isinstance(position, (str, bytes)) or len(position) != 2:
        raise ValueError(f"native lineage metrics child position is invalid for child_id={child_id}")
    return {
        "parent_id": _safe_int(parent_ids[index], default=-1),
        "mate_id": _safe_int(mate_ids[index], default=-1),
        "child_slot": _safe_int(child_slots[index], default=-1),
        "drives": drives,
        "position": (_safe_int(position[0]), _safe_int(position[1])),
    }


def _native_birth_event_context(
    *,
    tensors: Mapping[str, Any],
    metadata: Mapping[str, Any],
    child_id: int,
    generation: int,
) -> dict[str, Any]:
    names = (
        "typed_lifecycle_event_event_type",
        "typed_lifecycle_event_success",
        "typed_lifecycle_event_child_agent_id",
        "typed_lifecycle_event_parent_agent_id",
        "typed_lifecycle_event_lineage_founder_id",
        "typed_lifecycle_event_lineage_depth",
        "typed_lifecycle_event_generation",
        "typed_lifecycle_event_tick",
        "typed_lifecycle_event_value",
    )
    missing = [name for name in names if name not in tensors]
    if missing:
        raise ValueError(f"native lineage metrics birth event projection is missing tensors: {missing}")
    arrays = {name: _tensor_array(tensors[name]) for name in names}
    if any(value is None for value in arrays.values()):
        raise ValueError("native lineage metrics birth event projection has unreadable tensors")
    active_count = _safe_int(metadata.get("typed_lifecycle_event_active_count"), default=-1)
    if active_count < 0 or any(len(value) < active_count for value in arrays.values()):
        raise ValueError("native lineage metrics typed lifecycle active count is invalid")
    matches = []
    for index in range(active_count):
        if _safe_int(arrays["typed_lifecycle_event_event_type"][index]) != 1:
            continue
        if not _truthy(arrays["typed_lifecycle_event_success"][index]):
            continue
        if _safe_int(arrays["typed_lifecycle_event_child_agent_id"][index], default=-1) != int(child_id):
            continue
        matches.append(index)
    if len(matches) != 1:
        raise ValueError(f"native lineage metrics expected one successful birth event for child_id={child_id}")
    index = matches[0]
    event_generation = _safe_int(arrays["typed_lifecycle_event_generation"][index], default=-1)
    if event_generation != int(generation):
        raise ValueError(
            f"native lineage metrics birth event generation mismatch for child_id={child_id}: "
            f"expected={generation} actual={event_generation}"
        )
    return {
        "parent_id": _safe_int(arrays["typed_lifecycle_event_parent_agent_id"][index], default=-1),
        "founder_id": _safe_int(arrays["typed_lifecycle_event_lineage_founder_id"][index], default=-1),
        "lineage_depth": _safe_int(arrays["typed_lifecycle_event_lineage_depth"][index], default=-1),
        "generation": event_generation,
        "tick": _safe_int(arrays["typed_lifecycle_event_tick"][index], default=-1),
        "energy": _lineage_finite(
            arrays["typed_lifecycle_event_value"][index],
            context=f"native birth event energy for child_id={child_id}",
        ),
    }


def _native_child_genome(
    *,
    raw: Mapping[str, Any],
    row: Mapping[str, Any],
    child_id: int,
    child_slot: int,
    parent_slot: int,
    mate_slot: int,
    tensors: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    from agents.modular_genome_v1 import canon_float

    traces = _native_sequence(raw.get("birth_genome_trace_rows"), context="native birth_genome_trace_rows")
    matching_traces = [
        trace
        for trace in traces
        if isinstance(trace, Mapping) and _safe_int(trace.get("child_id"), default=-1) == int(child_id)
    ]
    if len(matching_traces) > 1:
        raise ValueError(f"native lineage metrics has duplicate birth genome traces for child_id={child_id}")
    if matching_traces:
        genes = matching_traces[0].get("child_genes")
        if not isinstance(genes, Mapping):
            raise ValueError(f"native lineage metrics child genome trace is incomplete for child_id={child_id}")
        required = ("hidden_size", "memory_dim", "use_gru", "aggression_bias", "exploration_bias")
        missing = [name for name in required if name not in genes]
        if missing:
            raise ValueError(f"native lineage metrics child genome trace is missing {missing} for child_id={child_id}")
        return {
            "hidden_size": _safe_int(genes["hidden_size"]),
            "memory_dim": _safe_int(genes["memory_dim"]),
            "use_gru": _truthy(genes["use_gru"]),
            "aggression_bias": canon_float(genes["aggression_bias"], field="aggression_bias"),
            "exploration_bias": canon_float(genes["exploration_bias"], field="exploration_bias"),
        }

    scalar_fields = _NATIVE_CHILD_SCALAR_INT_FIELDS + _NATIVE_CHILD_SCALAR_FLOAT_FIELDS
    missing = [name for name in scalar_fields if name not in tensors]
    if missing:
        raise ValueError(
            "native lineage metrics needs birth genome trace or complete scalar projection; "
            f"missing={missing} child_id={child_id}"
        )
    parent = _native_scalar_projection(tensors, slot=parent_slot)
    mate = _native_scalar_projection(tensors, slot=mate_slot)
    if parent["genome_hidden_size"] != mate["genome_hidden_size"]:
        raise ValueError(f"native lineage metrics needs birth genome trace for mixed hidden sizes, child_id={child_id}")
    candidate: dict[str, Any] = {}
    for name in _NATIVE_CHILD_SCALAR_INT_FIELDS:
        if name == "genome_memory_dim":
            if str(metadata.get("native_runtime_fixed_policy_family", "") or "").lower() == "mlp":
                candidate[name] = 0
            elif parent[name] == mate[name]:
                candidate[name] = int(parent[name])
            else:
                raise ValueError(f"native lineage metrics needs birth genome trace for mixed memory, child_id={child_id}")
        else:
            candidate[name] = int(parent[name])
    for name in _NATIVE_CHILD_SCALAR_FLOAT_FIELDS:
        base = 0.5 * (float(parent[name]) + float(mate[name]))
        candidate[name] = canon_float(base, field=name.removeprefix("genome_"))
    candidate["genome_aggression_bias"] = canon_float(
        _lineage_finite(row.get("aggression_bias"), context=f"behavior child {child_id} aggression_bias"),
        field="aggression_bias",
    )
    expected_hash = _native_child_scalar_hash(child_slot, candidate)
    observed_hash = _native_child_scalar_stage_hash(raw, child_id=child_id, child_slot=child_slot)
    if expected_hash != observed_hash:
        raise ValueError(
            "native lineage metrics cannot infer mutated child scalar genome without a trace: "
            f"child_id={child_id} expected_hash={expected_hash} observed_hash={observed_hash}"
        )
    return {
        "hidden_size": int(candidate["genome_hidden_size"]),
        "memory_dim": int(candidate["genome_memory_dim"]),
        "use_gru": False,
        "aggression_bias": float(candidate["genome_aggression_bias"]),
        "exploration_bias": float(candidate["genome_exploration_bias"]),
    }


def _native_scalar_projection(tensors: Mapping[str, Any], *, slot: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in _NATIVE_CHILD_SCALAR_INT_FIELDS:
        result[name] = _native_tensor_slot_int(tensors, name, slot)
    for name in _NATIVE_CHILD_SCALAR_FLOAT_FIELDS:
        result[name] = _native_tensor_slot_float(tensors, name, slot)
    return result


def _native_child_scalar_hash(child_slot: int, values: Mapping[str, Any]) -> int:
    mask = (1 << 64) - 1

    def mix(current: int, value: int) -> int:
        return ((current ^ (int(value) & mask)) * 1099511628211) & mask

    result = 1469598103934665603
    result = mix(result, child_slot)
    for name in _NATIVE_CHILD_SCALAR_INT_FIELDS:
        result = mix(result, int(values[name]))
    for name in _NATIVE_CHILD_SCALAR_FLOAT_FIELDS:
        bits = struct.unpack("<I", struct.pack("<f", float(values[name])))[0]
        result = mix(result, bits)
    return int(result & 0x7FFFFFFFFFFFFFFF)


def _native_child_scalar_stage_hash(raw: Mapping[str, Any], *, child_id: int, child_slot: int) -> int:
    rows = _native_sequence(raw.get("child_hash_chain_rows"), context="native child_hash_chain_rows")
    matches = [
        row
        for row in rows
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes)) and len(row) >= 5
        and _safe_int(row[0], default=-1) == int(child_id)
        and _safe_int(row[1], default=-1) == int(child_slot)
        and _safe_int(row[2], default=-1) == 1
        and str(row[3]) == "inherited_scalar_developmental"
    ]
    if len(matches) != 1:
        raise ValueError(f"native lineage metrics expected one child scalar hash for child_id={child_id}")
    return _safe_int(matches[0][4], default=-1)


def _native_sequence(value: Any, *, context: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{context} must be a sequence")
    try:
        return list(value)
    except Exception as exc:
        raise ValueError(f"{context} must be a sequence") from exc


def _native_slot_value(arrays: Mapping[str, Any], name: str, slot: int) -> Any:
    array = arrays.get(name)
    try:
        return array[slot]
    except Exception as exc:
        raise ValueError(f"native lineage metrics tensor {name} has no slot {slot}") from exc


def _native_slot_int(arrays: Mapping[str, Any], name: str, slot: int) -> int:
    value = _native_slot_value(arrays, name, slot)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"native lineage metrics tensor {name}[{slot}] is not an integer") from exc


def _native_slot_float(arrays: Mapping[str, Any], name: str, slot: int) -> float:
    return _lineage_finite(
        _native_slot_value(arrays, name, slot),
        context=f"native lineage metrics tensor {name}[{slot}]",
    )


def _native_slot_vector(arrays: Mapping[str, Any], name: str, slot: int, *, expected: int) -> tuple[float, ...]:
    value = _native_slot_value(arrays, name, slot)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != expected:
        raise ValueError(f"native lineage metrics tensor {name}[{slot}] must have length {expected}")
    return tuple(
        _lineage_finite(item, context=f"native lineage metrics tensor {name}[{slot}][{index}]")
        for index, item in enumerate(value)
    )


def _native_tensor_slot_int(tensors: Mapping[str, Any], name: str, slot: int) -> int:
    array = _tensor_array(tensors.get(name))
    if array is None:
        raise ValueError(f"native lineage metrics projection tensor {name} is unreadable")
    return _native_slot_int({name: array}, name, slot)


def _native_tensor_slot_float(tensors: Mapping[str, Any], name: str, slot: int) -> float:
    array = _tensor_array(tensors.get(name))
    if array is None:
        raise ValueError(f"native lineage metrics projection tensor {name} is unreadable")
    return _native_slot_float({name: array}, name, slot)


def _lineage_finite(value: Any, *, context: str) -> float:
    result = _finite_float(value)
    if result is None:
        raise ValueError(f"{context} is missing or non-finite")
    return float(result)


def _lineage_close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=5.0e-7)


def _add_biome_fractions(
    out: dict[str, Any],
    *,
    tensors: Mapping[str, Any],
    live_slots: list[int],
    fail_closed: bool = False,
) -> None:
    biome_grid = _tensor_array(tensors.get("biome_grid"))
    pos_y = _tensor_array(tensors.get("pos_y"))
    pos_x = _tensor_array(tensors.get("pos_x"))
    survival_steps = _tensor_array(tensors.get("survival_steps"))
    if not biome_grid or not pos_y or not pos_x or not survival_steps:
        if fail_closed:
            missing = [
                name
                for name, value in (
                    ("biome_grid", biome_grid),
                    ("pos_y", pos_y),
                    ("pos_x", pos_x),
                    ("survival_steps", survival_steps),
                )
                if not value
            ]
            raise ValueError(
                "scope-locked native Flat biome reporting has incomplete authority: "
                f"missing={missing}"
            )
        out["biome_safe_frac_alive"] = 1.0
        out["biome_wild_frac_alive"] = 0.0
        return

    wild = 0
    for slot in live_slots:
        if fail_closed:
            max_length = min(len(pos_y), len(pos_x), len(survival_steps))
            if slot >= max_length:
                raise ValueError(
                    "scope-locked native Flat biome reporting authority does not cover "
                    f"live slot {slot}"
                )
            age = _required_scope_int(
                survival_steps[slot],
                context=f"survival_steps[{slot}]",
                nonnegative=True,
            )
            y = _required_scope_int(pos_y[slot], context=f"pos_y[{slot}]")
            x = _required_scope_int(pos_x[slot], context=f"pos_x[{slot}]")
            if y < 0 or x < 0 or y >= len(biome_grid):
                raise ValueError(
                    f"native public metrics position is outside biome grid: slot={slot} y={y} x={x}"
                )
            grid_row = biome_grid[y]
            if not isinstance(grid_row, Sequence) or isinstance(
                grid_row,
                (str, bytes, bytearray),
            ) or x >= len(grid_row):
                raise ValueError(
                    f"native public metrics position is outside biome grid: slot={slot} y={y} x={x}"
                )
            biome = _required_scope_int(
                grid_row[x],
                context=f"biome_grid[{y}][{x}]",
            )
            if biome not in (0, 1):
                raise ValueError(
                    "scope-locked native Flat biome reporting requires biome IDs 0 or 1: "
                    f"biome_grid[{y}][{x}]={biome}"
                )
            # A final-tick birth has not completed an environment step, so its
            # reference stats.biome_id remains safe even on a wild spawn cell.
            wild += int(age > 0 and biome == 1)
            continue
        # Reference stats.biome_id starts safe and is updated only when a
        # resident completes an environment step. A final-tick birth therefore
        # remains safe even when its spawn cell is wild.
        if _safe_int(_index1(survival_steps, slot)) <= 0:
            continue
        y = _safe_int(_index1(pos_y, slot), default=-1)
        x = _safe_int(_index1(pos_x, slot), default=-1)
        if y < 0 or x < 0 or y >= len(biome_grid) or x >= len(biome_grid[y]):
            raise ValueError(f"native public metrics position is outside biome grid: slot={slot} y={y} x={x}")
        wild += int(_safe_int(biome_grid[y][x]) == 1)
    count = float(len(live_slots))
    out["biome_wild_frac_alive"] = float(wild) / count
    out["biome_safe_frac_alive"] = float(len(live_slots) - wild) / count


def _add_selected_parent_genome_metrics(
    out: dict[str, Any],
    *,
    tensors: Mapping[str, Any],
    aid_to_slot: Mapping[int, int],
    behavior_rows: list[dict[str, Any]],
) -> None:
    selected_ids = [
        _safe_int(row.get("agent_id"), default=-1)
        for row in behavior_rows
        if _safe_int(row.get("kept_next")) > 0
    ]
    if not selected_ids:
        return
    missing_ids = [agent_id for agent_id in selected_ids if agent_id not in aid_to_slot]
    if missing_ids:
        raise ValueError(
            "native public metrics selected-parent rows are absent from the owner projection: "
            f"{missing_ids}"
        )
    selected_slots = [int(aid_to_slot[agent_id]) for agent_id in selected_ids]
    trait_fields = (
        "genome_hidden_size",
        "genome_exploration_bias",
        "genome_aggression_bias",
        "genome_reproduction_threshold",
        "genome_mutation_rate",
    )
    missing_fields = [name for name in trait_fields if name not in tensors]
    if missing_fields:
        raise ValueError(f"native public metrics projection is missing genome fields: {missing_fields}")

    from agents.modular_genome_v1 import canon_float
    from simulation.metrics_pkg.metrics_logger_genome import genome_stats_from_trait_values

    hidden_sizes = _tensor_slot_floats(tensors, "genome_hidden_size", selected_slots)
    canonical_traits = [
        [canon_float(value, field=name) for value in _tensor_slot_floats(tensors, name, selected_slots)]
        for name in trait_fields[1:]
    ]
    stats = genome_stats_from_trait_values(hidden_sizes, *canonical_traits)
    for name, value in zip(
        (
            "gen_hidden_size_mean",
            "gen_hidden_size_std",
            "gen_exploration_bias_mean",
            "gen_exploration_bias_std",
            "gen_aggression_bias_mean",
            "gen_aggression_bias_std",
            "gen_reproduction_threshold_mean",
            "gen_reproduction_threshold_std",
            "gen_mutation_rate_mean",
            "gen_mutation_rate_std",
            "genome_diversity_norm_std_mean",
        ),
        stats,
    ):
        out[name] = float(value)


def _add_early_lineage_metrics(
    out: dict[str, Any],
    *,
    tensors: Mapping[str, Any],
    live_slots: list[int],
    metadata: Mapping[str, Any],
    metrics_generation: int | None = None,
) -> None:
    generation = int(metrics_generation) if metrics_generation is not None else _safe_int(metadata.get("generation"), default=-1)
    if generation not in (1, 2):
        return
    founder_ids = _tensor_slot_floats(tensors, "lineage_founder_id", live_slots)
    if len(founder_ids) != len(live_slots):
        raise ValueError("native public metrics projection has incomplete lineage founder identity")
    counts: dict[int, int] = {}
    for founder_id in founder_ids:
        key = int(founder_id)
        counts[key] = int(counts.get(key, 0)) + 1
    total = float(len(founder_ids))
    if total <= 0.0:
        effective = 0.0
        top1 = 0.0
    else:
        shares = [float(count) / total for count in counts.values()]
        entropy = -sum(share * math.log(share) for share in shares if share > 0.0)
        effective = float(math.exp(entropy)) if entropy > 0.0 else 1.0
        top1 = float(max(shares))
    out[f"early_lineage_effective_gen_{generation}"] = effective
    out[f"early_lineage_top1_share_gen_{generation}"] = top1


def _add_chemistry(
    out: dict[str, Any],
    *,
    tensors: Mapping[str, Any],
    live_slots: list[int],
    aid_to_slot: dict[int, int],
    behavior_rows: list[dict[str, Any]],
    sim_cfg: Any | None,
) -> None:
    chemistry = _tensor_array(tensors.get("chemistry"))
    if not chemistry:
        return

    columns = (
        ("chem_energy_store_mean", 0),
        ("chem_repair_reserve_mean", 1),
        ("chem_acute_damage_mean", 2),
        ("chem_chronic_damage_mean", 3),
        ("chem_regulator_1_mean", 4),
        ("chem_regulator_2_mean", 5),
    )
    channel_values: dict[int, list[float]] = {}
    for _, channel in columns:
        vals = [_finite_float(_index2(chemistry, slot, channel)) for slot in live_slots]
        channel_values[channel] = [v for v in vals if v is not None]
    for column, channel in columns:
        vals = channel_values[channel]
        if vals:
            out[column] = float(mean(vals))

    acute = channel_values.get(2, [])
    chronic = channel_values.get(3, [])
    if acute:
        out["chem_acute_damage_p90"] = float(percentile(acute, 90))
        acute_threshold = _cfg_float(sim_cfg, "chem_acute_high_threshold", 0.70)
        out["chem_acute_high_frac"] = float(sum(1 for x in acute if x >= acute_threshold)) / float(max(1, len(acute)))
    if chronic:
        out["chem_chronic_damage_p90"] = float(percentile(chronic, 90))
        chronic_threshold = _cfg_float(sim_cfg, "chem_chronic_high_threshold", 0.70)
        out["chem_chronic_high_frac"] = float(sum(1 for x in chronic if x >= chronic_threshold)) / float(max(1, len(chronic)))

    selected_slots = []
    for row in behavior_rows:
        if _safe_int(row.get("kept_next")) <= 0:
            continue
        aid = _safe_int(row.get("agent_id"), default=-1)
        slot = aid_to_slot.get(aid)
        if slot is not None:
            selected_slots.append(slot)
    if selected_slots:
        repair = [_finite_float(_index2(chemistry, slot, 1)) for slot in selected_slots]
        chronic_sel = [_finite_float(_index2(chemistry, slot, 3)) for slot in selected_slots]
        repair_vals = [v for v in repair if v is not None]
        chronic_vals = [v for v in chronic_sel if v is not None]
        if repair_vals:
            out["chem_repair_reserve_mean_selected_parents"] = float(mean(repair_vals))
        if chronic_vals:
            out["chem_chronic_damage_mean_selected_parents"] = float(mean(chronic_vals))


def _add_drives(out: dict[str, Any], *, tensors: Mapping[str, Any], live_slots: list[int]) -> None:
    drives = _tensor_array(tensors.get("drives"))
    if not drives:
        return
    for column_base, channel in (
        ("drive_hunger", 0),
        ("drive_pain", 1),
        ("drive_fatigue", 2),
        ("drive_fear", 3),
    ):
        vals = [_finite_float(_index2(drives, slot, channel)) for slot in live_slots]
        vals_f = [v for v in vals if v is not None]
        if vals_f:
            out[f"{column_base}_mean"] = float(mean(vals_f))
            out[f"{column_base}_std"] = float(std(vals_f))


def _add_coherence(
    out: dict[str, Any],
    *,
    tensors: Mapping[str, Any],
    live_slots: list[int],
    aid_to_slot: dict[int, int],
    behavior_rows: list[dict[str, Any]],
    sim_cfg: Any | None,
    metadata: Mapping[str, Any],
) -> None:
    switches = _tensor_live_floats(tensors, "action_switches", live_slots)
    oscillations = _tensor_live_floats(tensors, "action_oscillations", live_slots)
    drive_volatility = _tensor_live_floats(tensors, "drive_volatility", live_slots)
    drive_jitter = _tensor_live_floats(tensors, "drive_jitter", live_slots) or drive_volatility
    selfharm = _tensor_live_floats(tensors, "selfharm_events", live_slots)
    reset_shock = _tensor_live_floats(tensors, "birth_reset_shock", live_slots)
    eligible_slots = {
        aid_to_slot[aid]
        for row in behavior_rows
        if _safe_int(row.get("eligible_for_selection")) > 0
        for aid in [_safe_int(row.get("agent_id"), default=-1)]
        if aid in aid_to_slot
    }

    def selection_scoped(values: list[float]) -> list[float]:
        if not behavior_rows or len(values) != len(live_slots):
            return values
        return [value if live_slots[idx] in eligible_slots else 0.0 for idx, value in enumerate(values)]

    coh_switches = selection_scoped(switches)
    coh_oscillations = selection_scoped(oscillations)
    coh_drive_jitter = selection_scoped(drive_jitter)
    coh_selfharm = selection_scoped(selfharm)
    coh_reset_shock = selection_scoped(reset_shock)

    _fill_mean_if_absent(out, "coherence_switches_mean", switches)
    _fill_mean_if_absent(out, "coherence_oscillations_mean", oscillations)
    _fill_mean_if_absent(out, "drive_volatility_mean", drive_volatility)
    _fill_mean_if_absent(out, "drive_jitter_mean", drive_jitter)
    _fill_distribution_if_absent(
        out,
        _tensor_live_floats(tensors, "identity_drift_ema", live_slots),
        mean_key="identity_drift_mean",
        p90_key="identity_drift_p90",
        max_key="identity_drift_max",
    )

    if coh_switches and coh_oscillations:
        thrash = [
            (0.5 * coh_switches[i]) + coh_oscillations[i]
            for i in range(min(len(coh_switches), len(coh_oscillations)))
        ]
        _fill_mean_if_absent(out, "coh_thrash_mean", thrash)
    _fill_mean_if_absent(out, "coh_selfharm_mean", coh_selfharm)

    pred_err_mean = _ratio_by_slot(tensors, "pred_err_sum", "pred_err_count", live_slots)
    ep0_pred_mean = _ratio_by_slot(tensors, "episode0_pred_err_sum", "episode0_pred_err_count", live_slots)
    coh_pred_err_mean = selection_scoped(pred_err_mean)
    coh_ep0_pred_mean = selection_scoped(ep0_pred_mean)
    _fill_mean_if_absent(out, "coh_pred_err_mean", coh_pred_err_mean)
    _fill_mean_if_absent(out, "coh_reset_shock_mean", coh_reset_shock)
    _fill_mean_if_absent(out, "coh_ep0_pred_err_mean", coh_ep0_pred_mean)
    _fill_mean_if_absent(out, "episode0_pred_err_mean", ep0_pred_mean)

    ep0_drive = _tensor_live_floats(tensors, "episode0_drive_jitter_sum", live_slots)
    if ep0_drive:
        denom = max(1.0, _cfg_float(sim_cfg, "episode0_k", _meta_float(metadata, "episode0_k", 20.0)))
        vals = [value / denom for value in ep0_drive]
        _fill_mean_if_absent(out, "episode0_drive_jitter_mean", vals)
    _fill_mean_if_absent(out, "episode0_switches_mean", _tensor_live_floats(tensors, "episode0_switches", live_slots))
    _fill_mean_if_absent(out, "episode0_oscillations_mean", _tensor_live_floats(tensors, "episode0_oscillations", live_slots))
    _fill_mean_if_absent(
        out,
        "episode0_selfharm_mean",
        _tensor_live_floats(tensors, "episode0_selfharm_events", live_slots),
    )

    if coh_pred_err_mean or coh_switches or coh_oscillations or coh_drive_jitter or coh_selfharm or coh_reset_shock or coh_ep0_pred_mean:
        penalties = _coherence_penalties(
            pred_err=coh_pred_err_mean,
            switches=coh_switches,
            oscillations=coh_oscillations,
            drive_jitter=coh_drive_jitter,
            selfharm=coh_selfharm,
            reset_shock=coh_reset_shock,
            ep0_pred=coh_ep0_pred_mean,
            sim_cfg=sim_cfg,
            metadata=metadata,
        )
        _fill_mean_if_absent(out, "coh_penalty_total_mean", penalties)


def _add_memory_buckets(out: dict[str, Any], *, tensors: Mapping[str, Any], live_slots: list[int]) -> None:
    dims = _tensor_live_floats(tensors, "genome_memory_dim", live_slots) or _tensor_live_floats(tensors, "policy_mem_dim", live_slots)
    if not dims:
        return
    out.setdefault("mean_memory_dim_alive", float(mean(dims)))
    out.setdefault("p90_memory_dim_alive", float(percentile(dims, 90)))
    out.setdefault("max_memory_dim_alive", float(max(dims)))
    denom = float(max(1, len(dims)))
    for bucket in (0, 8, 16, 32, 64):
        out.setdefault(f"mem{bucket}_frac_alive", float(sum(1 for value in dims if int(value) == bucket)) / denom)

    gru = _tensor_live_floats(tensors, "policy_use_gru", live_slots)
    if gru:
        out.setdefault("gru_alive_frac", float(sum(1 for value in gru if value > 0.0)) / float(max(1, len(gru))))


def _add_first_repro_from_tensor(out: dict[str, Any], *, tensors: Mapping[str, Any], live_slots: list[int]) -> None:
    if "first_repro_frac_alive" in out:
        return
    steps = [value for value in _tensor_live_floats(tensors, "first_repro_step", live_slots) if value >= 0.0]
    if not steps:
        return
    frac = float(len(steps)) / float(max(1, len(live_slots)))
    out["first_repro_step_mean_alive"] = float(mean(steps))
    out["first_repro_frac_alive"] = frac
    out["tier4_fraction_reproduced_alive_end"] = frac
    out["tier4_first_repro_step_mean_reproducers_alive_end"] = float(mean(steps))
    out["tier4_first_repro_step_p50_reproducers_alive_end"] = _tier4_reproduction_percentile(steps, 50)
    out["tier4_first_repro_step_p90_reproducers_alive_end"] = _tier4_reproduction_percentile(steps, 90)


def _add_alive_age_hist(out: dict[str, Any], *, tensors: Mapping[str, Any], live_slots: list[int], env_cfg: Any | None) -> None:
    ages = _tensor_live_floats(tensors, "survival_steps", live_slots)
    if not ages:
        return
    edges = _cfg_sequence(env_cfg, "tier4_survival_bins_steps", DEFAULT_SURVIVAL_BINS_STEPS)
    if len(edges) < 2:
        return
    bins = [0 for _ in range(len(edges) - 1)]
    for raw_age in ages:
        age = int(raw_age)
        for idx in range(len(edges) - 1):
            if age >= int(edges[idx]) and age < int(edges[idx + 1]):
                bins[idx] += 1
                break
    out["tier4_bins_steps_json"] = json.dumps([int(x) for x in edges])
    out["tier4_alive_age_hist_json"] = json.dumps(bins)


def _coherence_penalties(
    *,
    pred_err: list[float],
    switches: list[float],
    oscillations: list[float],
    drive_jitter: list[float],
    selfharm: list[float],
    reset_shock: list[float],
    ep0_pred: list[float],
    sim_cfg: Any | None,
    metadata: Mapping[str, Any],
) -> list[float]:
    n = max(len(pred_err), len(switches), len(oscillations), len(drive_jitter), len(selfharm), len(reset_shock), len(ep0_pred))
    if n <= 0:
        return []
    if _cfg_bool(sim_cfg, "disable_prediction_error_selection_pressure", _meta_bool(metadata, "disable_prediction_error_selection_pressure", False)):
        pred_w = 0.0
        ep0_pred_w = 0.0
    else:
        pred_w = _cfg_float(sim_cfg, "coherence_pred_err_weight", _meta_float(metadata, "coherence_pred_err_weight", 1.25))
        ep0_pred_w = _cfg_float(sim_cfg, "coherence_episode0_pred_err_weight", _meta_float(metadata, "coherence_episode0_pred_err_weight", 0.75))
    thrash_w = _cfg_float(sim_cfg, "coherence_thrashing_weight", _meta_float(metadata, "coherence_thrashing_weight", 0.15))
    jitter_w = _cfg_float(sim_cfg, "coherence_drive_jitter_weight", _meta_float(metadata, "coherence_drive_jitter_weight", 0.08))
    selfharm_w = _cfg_float(sim_cfg, "coherence_selfharm_weight", _meta_float(metadata, "coherence_selfharm_weight", 3.0))
    reset_w = _cfg_float(sim_cfg, "coherence_reset_shock_weight", _meta_float(metadata, "coherence_reset_shock_weight", 0.35))
    penalties = []
    for i in range(n):
        switch = _value_at(switches, i)
        osc = _value_at(oscillations, i)
        thrash = 0.5 * switch + osc
        penalties.append(
            thrash_w * thrash
            + jitter_w * _value_at(drive_jitter, i)
            + selfharm_w * _value_at(selfharm, i)
            + pred_w * _value_at(pred_err, i)
            + reset_w * _value_at(reset_shock, i)
            + ep0_pred_w * _value_at(ep0_pred, i)
        )
    return penalties


def _behavior_rows(value: Any) -> list[dict[str, Any]]:
    rows = []
    if value is None:
        return rows
    try:
        iterable = list(value)
    except Exception:
        return rows
    for row in iterable:
        if isinstance(row, Mapping):
            rows.append(dict(row))
            continue
        if isinstance(row, (str, bytes)):
            continue
        if isinstance(row, Sequence):
            mapped = {}
            for idx, name in enumerate(BEHAVIOR_ROW_FIELDS):
                if idx < len(row):
                    mapped[name] = row[idx]
            rows.append(mapped)
    return rows


def _scope_locked_flat_reporting_generation(
    metadata: Mapping[str, Any],
    *,
    metrics_generation: Any,
) -> int:
    from .flat_population_reporting import resolve_native_flat_reporting_generation

    return resolve_native_flat_reporting_generation(
        metadata,
        reporting_generation=metrics_generation,
    )


def _required_scope_int(
    value: Any,
    *,
    context: str,
    nonnegative: bool = False,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"scope-locked native Flat reporting {context} must be an integer")
    result = int(value)
    if nonnegative and result < 0:
        raise ValueError(
            f"scope-locked native Flat reporting {context} must be nonnegative"
        )
    return result


def _live_slots(tensors: Mapping[str, Any]) -> tuple[list[int], dict[int, int]]:
    alive = _tensor_array(tensors.get("alive_mask"))
    agent_ids = _tensor_array(tensors.get("slot_to_agent_id"))
    if not alive or not agent_ids:
        return [], {}
    n = min(len(alive), len(agent_ids))
    slots = []
    aid_to_slot = {}
    for slot in range(n):
        aid = _safe_int(_index1(agent_ids, slot), default=-1)
        if aid < 0 or not _truthy(_index1(alive, slot)):
            continue
        slots.append(slot)
        aid_to_slot[aid] = slot
    return slots, aid_to_slot


def _tensor_live_floats(tensors: Mapping[str, Any], name: str, live_slots: list[int]) -> list[float]:
    return _tensor_slot_floats(tensors, name, live_slots)


def _tensor_slot_floats(tensors: Mapping[str, Any], name: str, slots: Sequence[int]) -> list[float]:
    array = _tensor_array(tensors.get(name))
    if not array:
        return []
    values = []
    for slot in slots:
        value = _finite_float(_index1(array, slot))
        if value is not None:
            values.append(value)
    return values


def _ratio_by_slot(tensors: Mapping[str, Any], sum_name: str, count_name: str, live_slots: list[int]) -> list[float]:
    sums = _tensor_array(tensors.get(sum_name))
    counts = _tensor_array(tensors.get(count_name))
    if not sums or not counts:
        return []
    values = []
    for slot in live_slots:
        total = _finite_float(_index1(sums, slot))
        count = _finite_float(_index1(counts, slot))
        if total is None or count is None or count <= 0.0:
            continue
        values.append(float(total) / float(count))
    return values


def _tensor_array(value: Any) -> Any:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
        if hasattr(value, "tolist"):
            return value.tolist()
    except Exception:
        return None
    if isinstance(value, (list, tuple)):
        return value
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _get(container: Any, key: str, default: Any = None) -> Any:
    if isinstance(container, Mapping):
        return container.get(key, default)
    return getattr(container, key, default)


def _first_present(source: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in source and source[name] is not None:
            return source[name]
    return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _row_floats(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = [_finite_float(row.get(key)) for row in rows if row.get(key) is not None]
    return [value for value in values if value is not None]


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() not in {"", "0", "false", "none"}
    return bool(value)


def _index1(array: Any, idx: int) -> Any:
    try:
        return array[idx]
    except Exception:
        return None


def _index2(array: Any, idx0: int, idx1: int) -> Any:
    try:
        return array[idx0][idx1]
    except Exception:
        return None


def _add_mean(out: dict[str, Any], key: str, values: list[float]) -> None:
    if values:
        out[key] = float(mean(values))


def _fill_mean_if_absent(out: dict[str, Any], key: str, values: list[float]) -> None:
    if key not in out:
        _add_mean(out, key, values)


def _add_distribution(
    out: dict[str, Any],
    _label: str,
    values: list[float],
    *,
    mean_key: str | None = None,
    std_key: str | None = None,
    p90_key: str | None = None,
    max_key: str | None = None,
) -> None:
    if not values:
        return
    if mean_key is not None:
        out[mean_key] = float(mean(values))
    if std_key is not None:
        out[std_key] = float(std(values))
    if p90_key is not None:
        out[p90_key] = float(percentile(values, 90))
    if max_key is not None:
        out[max_key] = float(max(values))


def _fill_distribution_if_absent(
    out: dict[str, Any],
    values: list[float],
    *,
    mean_key: str | None = None,
    std_key: str | None = None,
    p90_key: str | None = None,
    max_key: str | None = None,
) -> None:
    if mean_key is not None and mean_key in out:
        mean_key = None
    if std_key is not None and std_key in out:
        std_key = None
    if p90_key is not None and p90_key in out:
        p90_key = None
    if max_key is not None and max_key in out:
        max_key = None
    _add_distribution(
        out,
        "",
        values,
        mean_key=mean_key,
        std_key=std_key,
        p90_key=p90_key,
        max_key=max_key,
    )


def _cfg_float(cfg: Any | None, name: str, default: float) -> float:
    if isinstance(cfg, Mapping):
        return _meta_float(cfg, name, default)
    value = _finite_float(getattr(cfg, name, default))
    return float(default) if value is None else float(value)


def _cfg_bool(cfg: Any | None, name: str, default: bool) -> bool:
    if cfg is None:
        return bool(default)
    if isinstance(cfg, Mapping):
        return _meta_bool(cfg, name, default)
    return bool(getattr(cfg, name, default))


def _cfg_sequence(cfg: Any | None, name: str, default: Sequence[int]) -> list[int]:
    raw = _get(cfg, name, default)
    if raw is None:
        raw = default
    try:
        return [int(x) for x in list(raw)]
    except Exception:
        return [int(x) for x in default]


def _meta_float(metadata: Mapping[str, Any], name: str, default: float) -> float:
    value = _finite_float(metadata.get(name, default))
    return float(default) if value is None else float(value)


def _meta_bool(metadata: Mapping[str, Any], name: str, default: bool) -> bool:
    if name not in metadata:
        return bool(default)
    return _truthy(metadata.get(name))


def _value_at(values: list[float], idx: int) -> float:
    return float(values[idx]) if idx < len(values) else 0.0


def _episode0_denom(age: float, *, sim_cfg: Any | None, metadata: Mapping[str, Any]) -> float:
    k = _cfg_float(sim_cfg, "episode0_k", _meta_float(metadata, "episode0_k", 20.0))
    return float(max(1.0, min(float(age), float(k))))


def _selftest() -> None:
    raw = {
        "behavior_rows": [
            [0, 1, "start", 1, 0, 0, 0, 1, 0, 0, 12, 5.0, 1, 0, 12.5, 1, 1, 1, 0.2, 0.1, 0.3],
            [0, 2, "start", 0, 1, 0, 0, 0, 1, 0, 8, 2.0, 0, 0, 4.0, 1, 1, 0, 0.4, 0.0, 0.1],
        ],
        "committed_birth_rows": 1,
    }
    projection = {
        "tensors": {
            "alive_mask": [True, True],
            "slot_to_agent_id": [1, 2],
            "survival_steps": [12, 8],
            "injury": [0.2, 0.4],
            "scar": [0.0, 0.1],
            "drives": [[0.1, 0.2, 0.3, 0.4], [0.3, 0.2, 0.1, 0.0]],
            "chemistry": [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]],
            "action_switches": [2, 4],
            "action_oscillations": [1, 0],
            "drive_volatility": [0.5, 0.25],
            "genome_hidden_size": [64, 64],
            "genome_exploration_bias": [0.2, 0.1],
            "genome_aggression_bias": [0.3, 0.1],
            "genome_reproduction_threshold": [30.0, 35.0],
            "genome_mutation_rate": [0.05, 0.04],
        }
    }
    row = public_metrics_overlay(raw, checkpoint_projection=projection)
    assert row["alive"] == 2
    assert row["successful_births"] == 1
    assert row["behavior_action_total"] == 4.0
    assert row["mean_injury"] == 0.30000000000000004
    assert "tier4_deaths_age_hist_json" not in row


if __name__ == "__main__":
    _selftest()
