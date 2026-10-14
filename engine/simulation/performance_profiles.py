from __future__ import annotations

"""Named simulation performance profiles.

Profiles in this module are explicit, default-off implementation profiles. They
may reduce reporting/output cost or enable parity-proven equivalent hot paths,
but they must not change scientific dynamics: no policy, reward, selection,
reproduction, memory, prediction-state, or RNG behavior belongs here.
"""

from dataclasses import asdict, is_dataclass
from typing import Any


NO_PROFILE_NAMES = {"", "none", "current_evidence_profile"}
FAST_EXPLORATORY_PROFILE_V1 = "fast_exploratory_metrics_only_v1"
FAST_EXPLORATORY_PROFILE_V2 = "fast_exploratory_metrics_only_v2"
FAST_EXPLORATORY_PROFILE_V3 = "fast_exploratory_metrics_only_v3"
FAST_EXPLORATORY_PROFILE = FAST_EXPLORATORY_PROFILE_V1
FAST_EXPLORATORY_PARITY_PROFILE_V1 = "fast_exploratory_sampled_parity_v1"
FAST_EXPLORATORY_PARITY_PROFILE_V2 = "fast_exploratory_sampled_parity_v2"
FAST_EXPLORATORY_PARITY_PROFILE = FAST_EXPLORATORY_PARITY_PROFILE_V1
ACCEPTED_W6_FAST_PROFILE_V1 = "accepted_w6_fast_v1"
ACCEPTED_W6_FAST_PROFILE_V2 = "accepted_w6_fast_v2"
ACCEPTED_W6_FAST_PROFILE_V3 = "accepted_w6_fast_v3"
ACCEPTED_W6_FAST_PROFILE = ACCEPTED_W6_FAST_PROFILE_V3
W6_RUNNERCORE18_BRIDGE_PROFILE_V1 = "w6_runnercore18_bridge_v1"
W6_RUNNERCORE18_BRIDGE_PROFILE = W6_RUNNERCORE18_BRIDGE_PROFILE_V1
LRO3_CPUHOT_FAST_PROFILE_V1 = "lro3_cpuhot_fast_v1"
LRO3_CPUHOT_FAST_PROFILE = LRO3_CPUHOT_FAST_PROFILE_V1
LINEAGE_REPORT_FAST_PROFILE_V1 = "lineage_report_fast_v1"
LINEAGE_REPORT_FAST_PROFILE = LINEAGE_REPORT_FAST_PROFILE_V1
DEVELOPMENTAL_STAGE_CACHE_FAST_PROFILE_V1 = "developmental_stage_cache_fast_v1"
DEVELOPMENTAL_STAGE_CACHE_FAST_PROFILE = DEVELOPMENTAL_STAGE_CACHE_FAST_PROFILE_V1
EIEL_REPORT_COMPUTE_FAST_PROFILE_V1 = "eiel_report_compute_fast_v1"
EIEL_REPORT_COMPUTE_FAST_PROFILE = EIEL_REPORT_COMPUTE_FAST_PROFILE_V1

SUPPORTED_PROFILES = frozenset(
    {
        *NO_PROFILE_NAMES,
        FAST_EXPLORATORY_PROFILE_V1,
        FAST_EXPLORATORY_PROFILE_V2,
        FAST_EXPLORATORY_PROFILE_V3,
        FAST_EXPLORATORY_PARITY_PROFILE_V1,
        FAST_EXPLORATORY_PARITY_PROFILE_V2,
        ACCEPTED_W6_FAST_PROFILE_V1,
        ACCEPTED_W6_FAST_PROFILE_V2,
        ACCEPTED_W6_FAST_PROFILE_V3,
        W6_RUNNERCORE18_BRIDGE_PROFILE_V1,
        LRO3_CPUHOT_FAST_PROFILE_V1,
        LINEAGE_REPORT_FAST_PROFILE_V1,
        DEVELOPMENTAL_STAGE_CACHE_FAST_PROFILE_V1,
        EIEL_REPORT_COMPUTE_FAST_PROFILE_V1,
    }
)

CORE_INVARIANT_FIELDS = frozenset(
    {
        "seed",
        "generations",
        "steps_per_generation",
        "device",
        "bank_device",
        "policy_deterministic_actions",
        "policy_tick_stride",
        "mutation_rate",
        "mutation_std",
        "enable_experimental_prediction_cost_charging",
        "prediction_reward_enable",
        "direct_prediction_reward_enable",
        "direct_prediction_error_reward_enable",
        "selection_pressure",
        "survival_threshold",
        "reproduction_energy_threshold",
    }
)


def _snapshot(sim_cfg: Any) -> dict[str, Any]:
    if is_dataclass(sim_cfg):
        return dict(asdict(sim_cfg))
    return dict(getattr(sim_cfg, "__dict__", {}) or {})


def _set(sim_cfg: Any, changes: dict[str, Any], name: str, value: Any) -> None:
    before = getattr(sim_cfg, name, None)
    setattr(sim_cfg, name, value)
    after = getattr(sim_cfg, name, None)
    if before != after:
        changes[name] = {"before": before, "after": after}


def _apply_fast_exploratory_v1(sim_cfg: Any, changes: dict[str, Any]) -> None:
    _set(sim_cfg, changes, "memory_action_window_instrumentation_enable", False)
    _set(sim_cfg, changes, "memory_action_window_emit_jsonl", False)
    _set(sim_cfg, changes, "memory_action_window_include_lineage", False)
    _set(sim_cfg, changes, "memory_action_window_include_memory_state", False)
    _set(sim_cfg, changes, "memory_action_window_max_events_per_run", 0)

    _set(sim_cfg, changes, "prediction_demand_action_context_enable", False)
    _set(sim_cfg, changes, "prediction_demand_action_context_write_events", False)
    _set(sim_cfg, changes, "prediction_demand_action_context_max_events_per_run", 0)
    _set(sim_cfg, changes, "prediction_demand_action_context_sign_summary_enable", False)
    _set(sim_cfg, changes, "prediction_demand_action_context_sign_summary_include_windows", False)
    _set(sim_cfg, changes, "prediction_demand_action_context_sign_summary_write_jsonl", False)
    _set(sim_cfg, changes, "prediction_demand_action_context_sign_summary_write_compact_csv", False)
    _set(sim_cfg, changes, "prediction_demand_action_context_sign_summary_max_events_per_run", 0)
    _set(sim_cfg, changes, "enable_cognitive_status_metrics", False)
    _set(sim_cfg, changes, "a3_recovery_taxonomy_enable", False)


def _apply_fast_exploratory_v2(sim_cfg: Any, changes: dict[str, Any]) -> None:
    _apply_fast_exploratory_v1(sim_cfg, changes)


def _apply_accepted_w6_costsurface(sim_cfg: Any, changes: dict[str, Any]) -> None:
    _set(sim_cfg, changes, "env_hotpath_inactive_phenotype_defaults_enable", True)
    _set(sim_cfg, changes, "env_hotpath_cost_charge_surface_enable", True)
    _set(sim_cfg, changes, "env_hotpath_cost_charge_surface_guard_enable", False)
    _set(sim_cfg, changes, "native_cognitive_budget_cost_aggregation_enable", False)
    _set(sim_cfg, changes, "env_hotpath_direct_cost_scalar_enable", False)
    _set(sim_cfg, changes, "env_hotpath_direct_cost_scalar_mismatch_guard", False)


def _apply_w6_direct_reproduction_fastpath(sim_cfg: Any, changes: dict[str, Any]) -> None:
    _set(sim_cfg, changes, "cxxrewrite_reproduction_c2_payload_passthrough_enable", True)
    _set(sim_cfg, changes, "cxxrewrite_reproduction_c1_trusted_c2_payload_enable", True)
    _set(sim_cfg, changes, "cxxrewrite_reproduction_c2_trusted_parent_payloads_enable", True)
    _set(sim_cfg, changes, "cxxrewrite_reproduction_direct_child_payload_enable", True)
    _set(sim_cfg, changes, "cxxrewrite_reproduction_flat_policy_crossover_enable", True)


def _apply_w6_c1_canonical_fastpath(sim_cfg: Any, changes: dict[str, Any]) -> None:
    _set(sim_cfg, changes, "cxxrewrite_reproduction_c1_final_normalize_skip_enable", True)
    _set(sim_cfg, changes, "cxxrewrite_reproduction_c1_noop_normalize_skip_enable", True)
    _set(sim_cfg, changes, "cxxrewrite_reproduction_c1_empty_structure_fastpath_enable", True)


def _apply_runnercore18_supported_child_construction(sim_cfg: Any, changes: dict[str, Any]) -> None:
    """Apply the measured runnercore18_all harness lane to live/campaign config."""

    _set(sim_cfg, changes, "perf_aggperf3_timing_buckets_enable", True)

    _set(sim_cfg, changes, "runnercore4_profile_enable", True)
    _set(sim_cfg, changes, "runnercore4_candidate_enable", True)
    _set(sim_cfg, changes, "runnercore4_fail_closed_on_unsupported", True)
    _set(sim_cfg, changes, "runnercore4_allow_birth_boundary_agent_materialization", True)
    _set(sim_cfg, changes, "runnercore4_structural_payload_cache_enable", True)
    _set(sim_cfg, changes, "runnercore4_changed_parent_import_enable", True)

    _set(sim_cfg, changes, "runnercore5_profile_enable", True)
    _set(sim_cfg, changes, "runnercore5_candidate_enable", True)
    _set(sim_cfg, changes, "runnercore5_fail_closed_on_unsupported", True)
    _set(sim_cfg, changes, "runnercore5_allow_birth_boundary_agent_materialization", True)
    _set(sim_cfg, changes, "runnercore5_flat_self_policy_rng_enable", True)
    _set(sim_cfg, changes, "runnercore5_direct_newborn_mlp_bank_attach_enable", True)
    _set(sim_cfg, changes, "runnercore5_c1_canonical_boundary_fastpath_enable", True)

    _set(sim_cfg, changes, "runnercore6_profile_enable", True)
    _set(sim_cfg, changes, "runnercore6_candidate_enable", True)
    _set(sim_cfg, changes, "runnercore6_fail_closed_on_unsupported", True)
    _set(sim_cfg, changes, "runnercore6_allow_birth_boundary_agent_materialization", True)
    _set(sim_cfg, changes, "runnercore6_resident_newborn_row_attach_enable", True)
    _set(sim_cfg, changes, "runnercore6_trusted_canonical_genome_materialization_enable", False)

    _set(sim_cfg, changes, "runnercore7_self_mlp_policy_source_alias_enable", True)
    _set(sim_cfg, changes, "runnercore7_self_gru_policy_source_alias_enable", True)
    _set(sim_cfg, changes, "runnercore7_direct_newborn_gru_bank_attach_enable", True)
    _set(sim_cfg, changes, "runnercore7_static_reproduction_threshold_fastpath_enable", True)

    _set(sim_cfg, changes, "runnercore9_profile_enable", False)
    _set(sim_cfg, changes, "runnercore9_candidate_enable", False)
    _set(sim_cfg, changes, "runnercore9_c1_trace_enable", False)
    _set(sim_cfg, changes, "runnercore9_c1_scalar_op_normalize_skip_enable", False)
    _set(sim_cfg, changes, "runnercore9_lazy_newborn_materialization_enable", False)

    _set(sim_cfg, changes, "runnercore10_profile_enable", False)
    _set(sim_cfg, changes, "runnercore10_candidate_enable", False)
    _set(sim_cfg, changes, "runnercore10_typed_child_contract_enable", False)
    _set(sim_cfg, changes, "runnercore10_defer_direct_attached_compat_enable", False)

    _set(sim_cfg, changes, "runnercore11_profile_enable", True)
    _set(sim_cfg, changes, "runnercore11_candidate_enable", True)
    _set(sim_cfg, changes, "runnercore11_fail_closed_on_unsupported", True)
    _set(sim_cfg, changes, "runnercore11_c1_typed_rows_enable", True)
    _set(sim_cfg, changes, "runnercore11_fast_weighted_food_spawn_enable", True)
    _set(sim_cfg, changes, "runnercore11_generation_end_proxy_agents_enable", False)

    _set(sim_cfg, changes, "runnercore12_c1_payload_digest_reuse_enable", True)
    _set(sim_cfg, changes, "runnercore13_c1_projection_digest_sample_only_enable", True)
    _set(sim_cfg, changes, "runnercore15_c1_noop_normalize_skip_enable", True)
    _set(sim_cfg, changes, "runnercore16_c1_empty_structure_fastpath_enable", True)
    _set(sim_cfg, changes, "runnercore16_c1_empty_structure_fastpath_fail_closed", True)
    _set(sim_cfg, changes, "runnercore17_resident_birth_parent_enable", True)
    _set(sim_cfg, changes, "runnercore17_resident_birth_parent_fail_closed", True)
    _set(sim_cfg, changes, "runnercore18_resident_parent_minimal_refresh_enable", True)
    _set(sim_cfg, changes, "runnercore19_maintenance_arrays_enable", True)
    _set(sim_cfg, changes, "runnercore20_policy_observation_arrays_enable", True)
    _set(sim_cfg, changes, "runnercore21_cost_charge_surface_arrays_enable", True)
    _set(sim_cfg, changes, "runnercore22_full_birth_boundary_bridge_enable", True)
    _set(sim_cfg, changes, "runnercore23_jungle_predator_bridge_enable", False)
    _set(sim_cfg, changes, "runnercore24_native_jungle_predator_arrays_enable", True)
    _set(sim_cfg, changes, "runnercore27_full_birth_cooldown_prefilter_enable", True)
    _set(sim_cfg, changes, "runnercore31_full_birth_static_dev_threshold_enable", True)
    _set(sim_cfg, changes, "runnercore33_stride1_direct_action_enable", True)
    _set(sim_cfg, changes, "runnercore34_full_birth_resident_decision_enable", True)
    _set(sim_cfg, changes, "runnercore42_full_birth_patch_context_cache_enable", True)
    _set(sim_cfg, changes, "runnercore43_full_birth_threshold_tick_cache_enable", True)
    _set(sim_cfg, changes, "runnercore47_developmental_trace_buffered_emit_enable", True)
    _set(sim_cfg, changes, "runnercore45_jungle_predator_context_cache_enable", True)
    _set(sim_cfg, changes, "runnercore48_vectorized_chemistry_observation_enable", True)
    _set(sim_cfg, changes, "runnercore49_repro_viability_tick_cache_enable", True)
    _set(sim_cfg, changes, "runnercore50_spawn_first_valid_enable", True)


def _apply_lro3_cpuhot_dev_stage_cache(sim_cfg: Any, changes: dict[str, Any]) -> None:
    _set(sim_cfg, changes, "cxxrewrite_resident_dev_stage_cache_enable", True)
    _set(sim_cfg, changes, "cxxrewrite_resident_dev_stage_bounds_cache_enable", False)


def _apply_lineage_report_fast(sim_cfg: Any, changes: dict[str, Any]) -> None:
    _set(sim_cfg, changes, "lineage_reproduction_funnel_incremental_summary_cache_enable", True)


def _apply_developmental_stage_cache_fast(sim_cfg: Any, changes: dict[str, Any]) -> None:
    _set(sim_cfg, changes, "cxxrewrite_resident_dev_stage_cache_enable", True)
    _set(sim_cfg, changes, "cxxrewrite_resident_dev_stage_bounds_cache_enable", False)


def _apply_eiel_report_compute_fast(sim_cfg: Any, changes: dict[str, Any]) -> None:
    _apply_lineage_report_fast(sim_cfg, changes)
    _apply_developmental_stage_cache_fast(sim_cfg, changes)


def _restore_report_facing_metrics(sim_cfg: Any, changes: dict[str, Any], before: dict[str, Any]) -> None:
    for name in (
        "behavior_log_enable",
        "enable_cognitive_status_metrics",
        "a3_recovery_taxonomy_enable",
    ):
        if name in before:
            _set(sim_cfg, changes, name, before.get(name))


def apply_performance_profile(sim_cfg: Any, profile_name: str | None) -> dict[str, Any]:
    """Apply a named performance profile to ``sim_cfg``.

    Returns a compact audit object listing changed knobs and invariant-field
    checks. Unknown profiles raise ``ValueError`` to prevent silent benchmark
    contamination.
    """

    requested = str(profile_name or "").strip()
    profile = requested or "none"
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"unknown performance profile: {profile!r}")

    before = _snapshot(sim_cfg)
    changes: dict[str, Any] = {}
    if profile in NO_PROFILE_NAMES:
        return {
            "profile": profile,
            "changed_knobs": changes,
            "invariant_fields_changed": {},
            "scope": "no-op",
        }

    if profile == LRO3_CPUHOT_FAST_PROFILE_V1:
        _apply_lro3_cpuhot_dev_stage_cache(sim_cfg, changes)
        after = _snapshot(sim_cfg)
        invariant_changes = {
            name: {"before": before.get(name), "after": after.get(name)}
            for name in sorted(CORE_INVARIANT_FIELDS)
            if before.get(name) != after.get(name)
        }
        return {
            "profile": profile,
            "changed_knobs": changes,
            "invariant_fields_changed": invariant_changes,
            "scope": "lro3_cpuhot_dev_stage_cache_only",
        }

    # Shared fast-profile policy: sharply reduce high-volume output cost while
    # retaining core metrics.csv/perf_buckets.csv monitoring. The lineage report
    # profile is intentionally narrower and must preserve caller-selected output
    # and checkpoint knobs.
    if profile not in {
        LINEAGE_REPORT_FAST_PROFILE_V1,
        DEVELOPMENTAL_STAGE_CACHE_FAST_PROFILE_V1,
        EIEL_REPORT_COMPUTE_FAST_PROFILE_V1,
    }:
        _set(sim_cfg, changes, "behavior_log_enable", False)
        _set(sim_cfg, changes, "checkpoint_bundle_v2_period_gen", 0)

    if profile == FAST_EXPLORATORY_PROFILE_V1:
        _apply_fast_exploratory_v1(sim_cfg, changes)

    elif profile in {FAST_EXPLORATORY_PROFILE_V2, FAST_EXPLORATORY_PROFILE_V3}:
        _apply_fast_exploratory_v2(sim_cfg, changes)
        if profile == FAST_EXPLORATORY_PROFILE_V3:
            _set(sim_cfg, changes, "perf_aggperf3_timing_buckets_enable", True)
            _set(sim_cfg, changes, "perf_policy_stride_slot_cache_enable", True)

    elif profile == ACCEPTED_W6_FAST_PROFILE_V1:
        _apply_fast_exploratory_v2(sim_cfg, changes)
        _apply_accepted_w6_costsurface(sim_cfg, changes)

    elif profile in {ACCEPTED_W6_FAST_PROFILE_V2, ACCEPTED_W6_FAST_PROFILE_V3}:
        _apply_fast_exploratory_v2(sim_cfg, changes)
        _apply_accepted_w6_costsurface(sim_cfg, changes)
        _apply_w6_direct_reproduction_fastpath(sim_cfg, changes)
        if profile == ACCEPTED_W6_FAST_PROFILE_V3:
            _apply_w6_c1_canonical_fastpath(sim_cfg, changes)

    elif profile == W6_RUNNERCORE18_BRIDGE_PROFILE_V1:
        _apply_fast_exploratory_v2(sim_cfg, changes)
        _restore_report_facing_metrics(sim_cfg, changes, before)
        _apply_accepted_w6_costsurface(sim_cfg, changes)
        _apply_w6_direct_reproduction_fastpath(sim_cfg, changes)
        _apply_w6_c1_canonical_fastpath(sim_cfg, changes)
        _apply_runnercore18_supported_child_construction(sim_cfg, changes)

    elif profile == LINEAGE_REPORT_FAST_PROFILE_V1:
        _apply_lineage_report_fast(sim_cfg, changes)

    elif profile == DEVELOPMENTAL_STAGE_CACHE_FAST_PROFILE_V1:
        _apply_developmental_stage_cache_fast(sim_cfg, changes)

    elif profile == EIEL_REPORT_COMPUTE_FAST_PROFILE_V1:
        _apply_eiel_report_compute_fast(sim_cfg, changes)

    elif profile == FAST_EXPLORATORY_PARITY_PROFILE_V1:
        _set(sim_cfg, changes, "memory_action_window_emit_jsonl", True)
        _set(sim_cfg, changes, "memory_action_window_include_lineage", True)
        _set(sim_cfg, changes, "memory_action_window_include_memory_state", True)
        _set(sim_cfg, changes, "memory_action_window_max_events_per_run", 128)
        _set(sim_cfg, changes, "memory_action_window_sample_tick_period", 4)
        _set(sim_cfg, changes, "memory_action_window_sample_generation_period", 1)
        _set(sim_cfg, changes, "memory_action_window_sample_first_n_generations", 2)

        _set(sim_cfg, changes, "prediction_demand_action_context_write_events", False)
        _set(sim_cfg, changes, "prediction_demand_action_context_max_events_per_run", 128)
        _set(sim_cfg, changes, "prediction_demand_action_context_sample_tick_period", 4)
        _set(sim_cfg, changes, "prediction_demand_action_context_sample_generation_period", 1)
        _set(sim_cfg, changes, "prediction_demand_action_context_sample_first_n_generations", 2)
        _set(sim_cfg, changes, "prediction_demand_action_context_sign_summary_include_windows", True)
        _set(sim_cfg, changes, "prediction_demand_action_context_sign_summary_write_jsonl", True)
        _set(sim_cfg, changes, "prediction_demand_action_context_sign_summary_write_compact_csv", True)
        _set(sim_cfg, changes, "prediction_demand_action_context_sign_summary_max_events_per_run", 128)

    elif profile == FAST_EXPLORATORY_PARITY_PROFILE_V2:
        _set(sim_cfg, changes, "memory_action_window_emit_jsonl", True)
        _set(sim_cfg, changes, "memory_action_window_include_lineage", False)
        _set(sim_cfg, changes, "memory_action_window_include_memory_state", False)
        _set(sim_cfg, changes, "memory_action_window_max_events_per_run", 64)
        _set(sim_cfg, changes, "memory_action_window_sample_tick_period", 8)
        _set(sim_cfg, changes, "memory_action_window_sample_generation_period", 1)
        _set(sim_cfg, changes, "memory_action_window_sample_first_n_generations", 2)

        _set(sim_cfg, changes, "prediction_demand_action_context_write_events", False)
        _set(sim_cfg, changes, "prediction_demand_action_context_max_events_per_run", 64)
        _set(sim_cfg, changes, "prediction_demand_action_context_sample_tick_period", 8)
        _set(sim_cfg, changes, "prediction_demand_action_context_sample_generation_period", 1)
        _set(sim_cfg, changes, "prediction_demand_action_context_sample_first_n_generations", 2)
        _set(sim_cfg, changes, "prediction_demand_action_context_sign_summary_include_windows", True)
        _set(sim_cfg, changes, "prediction_demand_action_context_sign_summary_write_jsonl", True)
        _set(sim_cfg, changes, "prediction_demand_action_context_sign_summary_write_compact_csv", True)
        _set(sim_cfg, changes, "prediction_demand_action_context_sign_summary_max_events_per_run", 64)

    after = _snapshot(sim_cfg)
    invariant_changes = {
        name: {"before": before.get(name), "after": after.get(name)}
        for name in sorted(CORE_INVARIANT_FIELDS)
        if before.get(name) != after.get(name)
    }
    scope = "diagnostic_output_only"
    if profile == FAST_EXPLORATORY_PROFILE_V3:
        scope = "diagnostic_output_plus_default_off_policy_runner_fast_path"
    elif profile == ACCEPTED_W6_FAST_PROFILE_V1:
        scope = "diagnostic_output_plus_accepted_w6_costsurface_hotpath"
    elif profile == ACCEPTED_W6_FAST_PROFILE_V2:
        scope = "diagnostic_output_plus_accepted_w6_costsurface_plus_direct_reproduction_fastpath"
    elif profile == ACCEPTED_W6_FAST_PROFILE_V3:
        scope = (
            "diagnostic_output_plus_accepted_w6_costsurface_plus_direct_reproduction"
            "_plus_c1_canonical_fastpath"
        )
    elif profile == W6_RUNNERCORE18_BRIDGE_PROFILE_V1:
        scope = (
            "diagnostic_output_plus_accepted_w6_costsurface_plus_direct_reproduction"
            "_plus_c1_canonical_plus_runnercore18_bridge"
        )
    elif profile == LINEAGE_REPORT_FAST_PROFILE_V1:
        scope = "lineage_reproduction_funnel_incremental_summary_cache_only"
    elif profile == DEVELOPMENTAL_STAGE_CACHE_FAST_PROFILE_V1:
        scope = "developmental_stage_expression_cache_only"
    elif profile == EIEL_REPORT_COMPUTE_FAST_PROFILE_V1:
        scope = "lineage_report_plus_developmental_stage_expression_cache_only"
    return {
        "profile": profile,
        "changed_knobs": changes,
        "invariant_fields_changed": invariant_changes,
        "scope": scope,
    }
