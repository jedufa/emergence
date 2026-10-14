from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np

import pytest

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.metrics import MetricsLogger
from simulation.performance_profiles import (
    ACCEPTED_W6_FAST_PROFILE,
    ACCEPTED_W6_FAST_PROFILE_V1,
    ACCEPTED_W6_FAST_PROFILE_V2,
    ACCEPTED_W6_FAST_PROFILE_V3,
    CORE_INVARIANT_FIELDS,
    DEVELOPMENTAL_STAGE_CACHE_FAST_PROFILE,
    EIEL_REPORT_COMPUTE_FAST_PROFILE,
    FAST_EXPLORATORY_PARITY_PROFILE,
    FAST_EXPLORATORY_PARITY_PROFILE_V2,
    FAST_EXPLORATORY_PROFILE,
    FAST_EXPLORATORY_PROFILE_V2,
    FAST_EXPLORATORY_PROFILE_V3,
    LRO3_CPUHOT_FAST_PROFILE,
    LINEAGE_REPORT_FAST_PROFILE,
    W6_RUNNERCORE18_BRIDGE_PROFILE,
    apply_performance_profile,
)
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _evidence_like_cfg() -> SimConfig:
    cfg = SimConfig(
        seed=123,
        generations=3,
        steps_per_generation=40,
        device="cuda",
        bank_device="cuda",
        behavior_log_enable=True,
        policy_deterministic_actions=True,
        policy_tick_stride=2,
    )
    cfg.checkpoint_bundle_v2_period_gen = 1
    cfg.memory_action_window_instrumentation_enable = True
    cfg.memory_action_window_emit_jsonl = True
    cfg.memory_action_window_include_lineage = True
    cfg.memory_action_window_include_memory_state = True
    cfg.memory_action_window_max_events_per_run = 0
    cfg.prediction_demand_action_context_enable = True
    cfg.prediction_demand_action_context_write_events = True
    cfg.prediction_demand_action_context_sign_summary_enable = True
    cfg.prediction_demand_action_context_sign_summary_include_windows = True
    cfg.prediction_demand_action_context_sign_summary_write_jsonl = True
    cfg.prediction_demand_action_context_sign_summary_write_compact_csv = False
    cfg.prediction_demand_action_context_sign_summary_max_events_per_run = 0
    return cfg


def test_fast_profile_only_changes_diagnostic_output_knobs() -> None:
    cfg = _evidence_like_cfg()
    before = {name: getattr(cfg, name, None) for name in CORE_INVARIANT_FIELDS}

    audit = apply_performance_profile(cfg, FAST_EXPLORATORY_PROFILE)

    assert audit["scope"] == "diagnostic_output_only"
    assert audit["invariant_fields_changed"] == {}
    assert {name: getattr(cfg, name, None) for name in CORE_INVARIANT_FIELDS} == before
    assert cfg.behavior_log_enable is False
    assert cfg.checkpoint_bundle_v2_period_gen == 0
    assert getattr(cfg, "memory_action_window_instrumentation_enable") is False
    assert cfg.prediction_demand_action_context_enable is False
    assert cfg.prediction_demand_action_context_sign_summary_enable is False
    assert cfg.prediction_demand_action_context_sign_summary_include_windows is False
    assert cfg.prediction_demand_action_context_sign_summary_write_compact_csv is False
    assert cfg.enable_cognitive_status_metrics is False
    assert cfg.a3_recovery_taxonomy_enable is False


def test_parity_profile_keeps_richer_diagnostic_windows_but_samples() -> None:
    cfg = _evidence_like_cfg()

    audit = apply_performance_profile(cfg, FAST_EXPLORATORY_PARITY_PROFILE)

    assert audit["invariant_fields_changed"] == {}
    assert cfg.behavior_log_enable is False
    assert cfg.prediction_demand_action_context_sign_summary_include_windows is True
    assert cfg.prediction_demand_action_context_sign_summary_write_jsonl is True
    assert cfg.prediction_demand_action_context_sample_tick_period == 4
    assert getattr(cfg, "memory_action_window_sample_tick_period") == 4
    assert getattr(cfg, "memory_action_window_sample_first_n_generations") == 2


def test_fast_profile_v2_uses_telemetry_off_metrics_only_surface() -> None:
    cfg = _evidence_like_cfg()
    before = {name: getattr(cfg, name, None) for name in CORE_INVARIANT_FIELDS}

    audit = apply_performance_profile(cfg, FAST_EXPLORATORY_PROFILE_V2)

    assert audit["invariant_fields_changed"] == {}
    assert {name: getattr(cfg, name, None) for name in CORE_INVARIANT_FIELDS} == before
    assert cfg.behavior_log_enable is False
    assert cfg.checkpoint_bundle_v2_period_gen == 0
    assert getattr(cfg, "memory_action_window_instrumentation_enable") is False
    assert getattr(cfg, "memory_action_window_emit_jsonl") is False
    assert cfg.prediction_demand_action_context_enable is False
    assert cfg.prediction_demand_action_context_sign_summary_enable is False
    assert cfg.enable_cognitive_status_metrics is False
    assert cfg.a3_recovery_taxonomy_enable is False
    assert {"behavior_log_enable", "checkpoint_bundle_v2_period_gen"}.issubset(audit["changed_knobs"])


def test_lro3_cpuhot_profile_enables_stage_cache_without_output_knobs() -> None:
    cfg = _evidence_like_cfg()
    before = {name: getattr(cfg, name, None) for name in CORE_INVARIANT_FIELDS}

    audit = apply_performance_profile(cfg, LRO3_CPUHOT_FAST_PROFILE)

    assert audit["scope"] == "lro3_cpuhot_dev_stage_cache_only"
    assert audit["invariant_fields_changed"] == {}
    assert {name: getattr(cfg, name, None) for name in CORE_INVARIANT_FIELDS} == before
    assert cfg.behavior_log_enable is True
    assert cfg.checkpoint_bundle_v2_period_gen == 1
    assert cfg.memory_action_window_instrumentation_enable is True
    assert cfg.prediction_demand_action_context_enable is True
    assert cfg.cxxrewrite_resident_dev_stage_cache_enable is True
    assert cfg.cxxrewrite_resident_dev_stage_bounds_cache_enable is False
    assert audit["changed_knobs"] == {
        "cxxrewrite_resident_dev_stage_cache_enable": {"before": False, "after": True}
    }


def test_accepted_w6_fast_v1_preserves_original_w6_costsurface_profile() -> None:
    cfg = _evidence_like_cfg()
    before = {name: getattr(cfg, name, None) for name in CORE_INVARIANT_FIELDS}

    audit = apply_performance_profile(cfg, ACCEPTED_W6_FAST_PROFILE_V1)

    assert audit["scope"] == "diagnostic_output_plus_accepted_w6_costsurface_hotpath"
    assert audit["invariant_fields_changed"] == {}
    assert {name: getattr(cfg, name, None) for name in CORE_INVARIANT_FIELDS} == before
    assert cfg.env_hotpath_inactive_phenotype_defaults_enable is True
    assert cfg.env_hotpath_cost_charge_surface_enable is True
    assert cfg.env_hotpath_cost_charge_surface_guard_enable is False
    assert cfg.cxxrewrite_reproduction_c2_payload_passthrough_enable is False
    assert cfg.cxxrewrite_reproduction_c1_trusted_c2_payload_enable is False
    assert cfg.cxxrewrite_reproduction_c2_trusted_parent_payloads_enable is False
    assert cfg.cxxrewrite_reproduction_direct_child_payload_enable is False
    assert cfg.cxxrewrite_reproduction_c1_final_normalize_skip_enable is False
    assert cfg.cxxrewrite_reproduction_c1_noop_normalize_skip_enable is False
    assert cfg.cxxrewrite_reproduction_c1_empty_structure_fastpath_enable is False
    assert cfg.cxxrewrite_reproduction_flat_policy_crossover_enable is False
    assert cfg.runnercore11_c1_typed_rows_enable is False
    assert cfg.runnercore18_resident_parent_minimal_refresh_enable is False
    assert cfg.runnercore19_maintenance_arrays_enable is False
    assert cfg.runnercore20_policy_observation_arrays_enable is False
    assert cfg.runnercore21_cost_charge_surface_arrays_enable is False
    assert cfg.runnercore22_full_birth_boundary_bridge_enable is False
    assert cfg.runnercore23_jungle_predator_bridge_enable is False
    assert cfg.runnercore24_native_jungle_predator_arrays_enable is False
    assert cfg.runnercore25_full_birth_resident_newborn_attach_enable is False
    assert cfg.runnercore26_vectorized_jungle_predator_sampling_enable is False
    assert cfg.runnercore27_full_birth_cooldown_prefilter_enable is False
    assert cfg.runnercore30_full_birth_branch_prefilter_enable is False
    assert cfg.runnercore31_full_birth_static_dev_threshold_enable is False
    assert cfg.runnercore32_full_birth_inline_grid_scan_enable is False
    assert cfg.runnercore33_stride1_direct_action_enable is False
    assert cfg.runnercore34_full_birth_resident_decision_enable is False


def test_accepted_w6_fast_v2_preserves_direct_reproduction_profile() -> None:
    cfg = _evidence_like_cfg()
    before = {name: getattr(cfg, name, None) for name in CORE_INVARIANT_FIELDS}

    audit = apply_performance_profile(cfg, ACCEPTED_W6_FAST_PROFILE_V2)

    assert audit["scope"] == "diagnostic_output_plus_accepted_w6_costsurface_plus_direct_reproduction_fastpath"
    assert audit["invariant_fields_changed"] == {}
    assert {name: getattr(cfg, name, None) for name in CORE_INVARIANT_FIELDS} == before
    assert cfg.env_hotpath_inactive_phenotype_defaults_enable is True
    assert cfg.env_hotpath_cost_charge_surface_enable is True
    assert cfg.cxxrewrite_reproduction_c2_payload_passthrough_enable is True
    assert cfg.cxxrewrite_reproduction_c1_trusted_c2_payload_enable is True
    assert cfg.cxxrewrite_reproduction_c2_trusted_parent_payloads_enable is True
    assert cfg.cxxrewrite_reproduction_direct_child_payload_enable is True
    assert cfg.cxxrewrite_reproduction_flat_policy_crossover_enable is True
    assert cfg.cxxrewrite_reproduction_c1_final_normalize_skip_enable is False
    assert cfg.cxxrewrite_reproduction_c1_noop_normalize_skip_enable is False
    assert cfg.cxxrewrite_reproduction_c1_empty_structure_fastpath_enable is False


def test_accepted_w6_fast_profile_combines_positive_w6_knobs_only() -> None:
    cfg = _evidence_like_cfg()
    before = {name: getattr(cfg, name, None) for name in CORE_INVARIANT_FIELDS}

    audit = apply_performance_profile(cfg, ACCEPTED_W6_FAST_PROFILE)

    assert ACCEPTED_W6_FAST_PROFILE == ACCEPTED_W6_FAST_PROFILE_V3
    assert (
        audit["scope"]
        == "diagnostic_output_plus_accepted_w6_costsurface_plus_direct_reproduction_plus_c1_canonical_fastpath"
    )
    assert audit["invariant_fields_changed"] == {}
    assert {name: getattr(cfg, name, None) for name in CORE_INVARIANT_FIELDS} == before
    assert cfg.behavior_log_enable is False
    assert cfg.checkpoint_bundle_v2_period_gen == 0
    assert cfg.memory_action_window_instrumentation_enable is False
    assert cfg.prediction_demand_action_context_enable is False
    assert cfg.enable_cognitive_status_metrics is False
    assert cfg.a3_recovery_taxonomy_enable is False
    assert cfg.env_hotpath_inactive_phenotype_defaults_enable is True
    assert cfg.env_hotpath_cost_charge_surface_enable is True
    assert cfg.env_hotpath_cost_charge_surface_guard_enable is False
    assert cfg.native_cognitive_budget_cost_aggregation_enable is False
    assert cfg.env_hotpath_direct_cost_scalar_enable is False
    assert cfg.env_hotpath_direct_cost_scalar_mismatch_guard is False
    assert cfg.cxxrewrite_reproduction_c2_payload_passthrough_enable is True
    assert cfg.cxxrewrite_reproduction_c1_trusted_c2_payload_enable is True
    assert cfg.cxxrewrite_reproduction_c2_trusted_parent_payloads_enable is True
    assert cfg.cxxrewrite_reproduction_direct_child_payload_enable is True
    assert cfg.cxxrewrite_reproduction_flat_policy_crossover_enable is True
    assert cfg.cxxrewrite_reproduction_c1_final_normalize_skip_enable is True
    assert cfg.cxxrewrite_reproduction_c1_noop_normalize_skip_enable is True
    assert cfg.cxxrewrite_reproduction_c1_empty_structure_fastpath_enable is True
    assert cfg.cxxrewrite_reproduction_c1_empty_structure_fastpath_fail_closed is False
    assert cfg.env_hotpath_budget_ledger_scalar_fastpath_enable is False
    assert cfg.perf_aggperf3_timing_buckets_enable is False
    assert cfg.perf_policy_stride_slot_cache_enable is False
    assert cfg.runnercore1_candidate_enable is False
    assert cfg.runnercore11_c1_typed_rows_enable is False
    assert cfg.runnercore18_resident_parent_minimal_refresh_enable is False
    assert cfg.runnercore19_maintenance_arrays_enable is False
    assert cfg.runnercore20_policy_observation_arrays_enable is False
    assert cfg.runnercore21_cost_charge_surface_arrays_enable is False
    assert cfg.runnercore22_full_birth_boundary_bridge_enable is False
    assert cfg.runnercore23_jungle_predator_bridge_enable is False
    assert cfg.runnercore24_native_jungle_predator_arrays_enable is False
    assert cfg.runnercore25_full_birth_resident_newborn_attach_enable is False
    assert cfg.runnercore26_vectorized_jungle_predator_sampling_enable is False
    assert cfg.runnercore27_full_birth_cooldown_prefilter_enable is False


def test_w6_runnercore18_bridge_profile_matches_measured_runnercore18_lane() -> None:
    cfg = _evidence_like_cfg()
    before = {name: getattr(cfg, name, None) for name in CORE_INVARIANT_FIELDS}

    audit = apply_performance_profile(cfg, W6_RUNNERCORE18_BRIDGE_PROFILE)

    assert (
        audit["scope"]
        == "diagnostic_output_plus_accepted_w6_costsurface_plus_direct_reproduction"
        "_plus_c1_canonical_plus_runnercore18_bridge"
    )
    assert audit["invariant_fields_changed"] == {}
    assert {name: getattr(cfg, name, None) for name in CORE_INVARIANT_FIELDS} == before

    assert cfg.behavior_log_enable is True
    assert cfg.checkpoint_bundle_v2_period_gen == 0
    assert cfg.memory_action_window_instrumentation_enable is False
    assert cfg.prediction_demand_action_context_enable is False
    assert cfg.enable_cognitive_status_metrics is True
    assert cfg.a3_recovery_taxonomy_enable is True


    assert cfg.env_hotpath_inactive_phenotype_defaults_enable is True
    assert cfg.env_hotpath_cost_charge_surface_enable is True
    assert cfg.env_hotpath_cost_charge_surface_guard_enable is False
    assert cfg.cxxrewrite_reproduction_c2_payload_passthrough_enable is True
    assert cfg.cxxrewrite_reproduction_c1_trusted_c2_payload_enable is True
    assert cfg.cxxrewrite_reproduction_c2_trusted_parent_payloads_enable is True
    assert cfg.cxxrewrite_reproduction_direct_child_payload_enable is True
    assert cfg.cxxrewrite_reproduction_flat_policy_crossover_enable is True
    assert cfg.cxxrewrite_reproduction_c1_final_normalize_skip_enable is True
    assert cfg.cxxrewrite_reproduction_c1_noop_normalize_skip_enable is True
    assert cfg.cxxrewrite_reproduction_c1_empty_structure_fastpath_enable is True

    assert cfg.perf_aggperf3_timing_buckets_enable is True
    assert cfg.runnercore4_profile_enable is True
    assert cfg.runnercore4_candidate_enable is True
    assert cfg.runnercore4_fail_closed_on_unsupported is True
    assert cfg.runnercore4_allow_birth_boundary_agent_materialization is True
    assert cfg.runnercore4_structural_payload_cache_enable is True
    assert cfg.runnercore4_changed_parent_import_enable is True

    assert cfg.runnercore5_profile_enable is True
    assert cfg.runnercore5_candidate_enable is True
    assert cfg.runnercore5_fail_closed_on_unsupported is True
    assert cfg.runnercore5_allow_birth_boundary_agent_materialization is True
    assert cfg.runnercore5_flat_self_policy_rng_enable is True
    assert cfg.runnercore5_direct_newborn_mlp_bank_attach_enable is True
    assert cfg.runnercore5_c1_canonical_boundary_fastpath_enable is True

    assert cfg.runnercore6_profile_enable is True
    assert cfg.runnercore6_candidate_enable is True
    assert cfg.runnercore6_fail_closed_on_unsupported is True
    assert cfg.runnercore6_allow_birth_boundary_agent_materialization is True
    assert cfg.runnercore6_resident_newborn_row_attach_enable is True
    assert cfg.runnercore6_trusted_canonical_genome_materialization_enable is False

    assert cfg.runnercore7_self_mlp_policy_source_alias_enable is True
    assert cfg.runnercore7_self_gru_policy_source_alias_enable is True
    assert cfg.runnercore7_direct_newborn_gru_bank_attach_enable is True
    assert cfg.runnercore7_static_reproduction_threshold_fastpath_enable is True

    assert cfg.runnercore9_profile_enable is False
    assert cfg.runnercore9_candidate_enable is False
    assert cfg.runnercore9_c1_trace_enable is False
    assert cfg.runnercore9_lazy_newborn_materialization_enable is False
    assert cfg.runnercore10_profile_enable is False
    assert cfg.runnercore10_candidate_enable is False
    assert cfg.runnercore10_typed_child_contract_enable is False
    assert cfg.runnercore10_defer_direct_attached_compat_enable is False

    assert cfg.runnercore11_profile_enable is True
    assert cfg.runnercore11_candidate_enable is True
    assert cfg.runnercore11_fail_closed_on_unsupported is True
    assert cfg.runnercore11_c1_typed_rows_enable is True
    assert cfg.runnercore11_fast_weighted_food_spawn_enable is True
    assert cfg.runnercore11_generation_end_proxy_agents_enable is False
    assert cfg.runnercore12_c1_payload_digest_reuse_enable is True
    assert cfg.runnercore13_c1_projection_digest_sample_only_enable is True
    assert cfg.runnercore15_c1_noop_normalize_skip_enable is True
    assert cfg.runnercore16_c1_empty_structure_fastpath_enable is True
    assert cfg.runnercore16_c1_empty_structure_fastpath_fail_closed is True
    assert cfg.runnercore17_resident_birth_parent_enable is True
    assert cfg.runnercore17_resident_birth_parent_fail_closed is True
    assert cfg.runnercore18_resident_parent_minimal_refresh_enable is True
    assert cfg.runnercore19_maintenance_arrays_enable is True
    assert cfg.runnercore20_policy_observation_arrays_enable is True
    assert cfg.runnercore21_cost_charge_surface_arrays_enable is True
    assert cfg.runnercore22_full_birth_boundary_bridge_enable is True
    assert cfg.runnercore23_jungle_predator_bridge_enable is False
    assert cfg.runnercore24_native_jungle_predator_arrays_enable is True
    assert cfg.runnercore25_full_birth_resident_newborn_attach_enable is False
    assert cfg.runnercore26_vectorized_jungle_predator_sampling_enable is False
    assert cfg.runnercore27_full_birth_cooldown_prefilter_enable is True
    assert cfg.runnercore30_full_birth_branch_prefilter_enable is False
    assert cfg.runnercore31_full_birth_static_dev_threshold_enable is True
    assert cfg.runnercore32_full_birth_inline_grid_scan_enable is False
    assert cfg.runnercore33_stride1_direct_action_enable is True
    assert cfg.runnercore34_full_birth_resident_decision_enable is True
    assert cfg.runnercore42_full_birth_patch_context_cache_enable is True
    assert cfg.runnercore43_full_birth_threshold_tick_cache_enable is True
    assert cfg.runnercore47_developmental_trace_buffered_emit_enable is True
    assert cfg.runnercore45_jungle_predator_context_cache_enable is True
    assert cfg.runnercore48_vectorized_chemistry_observation_enable is True
    assert cfg.runnercore49_repro_viability_tick_cache_enable is True
    assert cfg.runnercore50_spawn_first_valid_enable is True


def test_lineage_report_fast_profile_only_enables_incremental_summary_cache() -> None:
    cfg = _evidence_like_cfg()
    cfg.lineage_reproduction_funnel_reporting_enable = True
    cfg.lineage_reproduction_funnel_write_events = True
    cfg.lineage_reproduction_funnel_write_scan_action_events = True
    before = {name: getattr(cfg, name, None) for name in CORE_INVARIANT_FIELDS}

    audit = apply_performance_profile(cfg, LINEAGE_REPORT_FAST_PROFILE)

    assert audit["scope"] == "lineage_reproduction_funnel_incremental_summary_cache_only"
    assert audit["invariant_fields_changed"] == {}
    assert {name: getattr(cfg, name, None) for name in CORE_INVARIANT_FIELDS} == before
    assert cfg.lineage_reproduction_funnel_incremental_summary_cache_enable is True
    assert cfg.behavior_log_enable is True
    assert cfg.checkpoint_bundle_v2_period_gen == 1
    assert cfg.memory_action_window_instrumentation_enable is True
    assert cfg.prediction_demand_action_context_enable is True
    assert set(audit["changed_knobs"]) == {"lineage_reproduction_funnel_incremental_summary_cache_enable"}


def test_developmental_stage_cache_fast_profile_only_enables_stage_cache() -> None:
    cfg = _evidence_like_cfg()
    cfg.cxxrewrite_resident_dev_stage_bounds_cache_enable = True
    before = {name: getattr(cfg, name, None) for name in CORE_INVARIANT_FIELDS}

    audit = apply_performance_profile(cfg, DEVELOPMENTAL_STAGE_CACHE_FAST_PROFILE)

    assert audit["scope"] == "developmental_stage_expression_cache_only"
    assert audit["invariant_fields_changed"] == {}
    assert {name: getattr(cfg, name, None) for name in CORE_INVARIANT_FIELDS} == before
    assert cfg.cxxrewrite_resident_dev_stage_cache_enable is True
    assert cfg.cxxrewrite_resident_dev_stage_bounds_cache_enable is False
    assert cfg.behavior_log_enable is True
    assert cfg.checkpoint_bundle_v2_period_gen == 1
    assert cfg.memory_action_window_instrumentation_enable is True
    assert cfg.prediction_demand_action_context_enable is True
    assert set(audit["changed_knobs"]) == {
        "cxxrewrite_resident_dev_stage_cache_enable",
        "cxxrewrite_resident_dev_stage_bounds_cache_enable",
    }

def test_eiel_report_compute_fast_profile_combines_accepted_cache_knobs_only() -> None:
    cfg = _evidence_like_cfg()
    cfg.lineage_reproduction_funnel_reporting_enable = True
    cfg.lineage_reproduction_funnel_write_events = True
    cfg.lineage_reproduction_funnel_write_scan_action_events = True
    cfg.cxxrewrite_resident_dev_stage_bounds_cache_enable = True
    before = {name: getattr(cfg, name, None) for name in CORE_INVARIANT_FIELDS}

    audit = apply_performance_profile(cfg, EIEL_REPORT_COMPUTE_FAST_PROFILE)

    assert audit["scope"] == "lineage_report_plus_developmental_stage_expression_cache_only"
    assert audit["invariant_fields_changed"] == {}
    assert {name: getattr(cfg, name, None) for name in CORE_INVARIANT_FIELDS} == before
    assert cfg.lineage_reproduction_funnel_incremental_summary_cache_enable is True
    assert cfg.cxxrewrite_resident_dev_stage_cache_enable is True
    assert cfg.cxxrewrite_resident_dev_stage_bounds_cache_enable is False
    assert cfg.behavior_log_enable is True
    assert cfg.checkpoint_bundle_v2_period_gen == 1
    assert cfg.memory_action_window_instrumentation_enable is True
    assert cfg.prediction_demand_action_context_enable is True
    assert set(audit["changed_knobs"]) == {
        "lineage_reproduction_funnel_incremental_summary_cache_enable",
        "cxxrewrite_resident_dev_stage_cache_enable",
        "cxxrewrite_resident_dev_stage_bounds_cache_enable",
    }


def test_accepted_w6_fast_profile_can_disable_checkpoint_knob_override() -> None:
    cfg = _evidence_like_cfg()
    cfg.checkpoint_bundle_v2_period_gen = 1

    audit = apply_performance_profile(cfg, ACCEPTED_W6_FAST_PROFILE)

    assert cfg.checkpoint_bundle_v2_period_gen == 0
    assert audit["changed_knobs"]["checkpoint_bundle_v2_period_gen"] == {"before": 1, "after": 0}


def test_fast_profile_v3_adds_default_off_policy_runner_fast_path() -> None:
    cfg = _evidence_like_cfg()
    before = {name: getattr(cfg, name, None) for name in CORE_INVARIANT_FIELDS}

    audit = apply_performance_profile(cfg, FAST_EXPLORATORY_PROFILE_V3)

    assert audit["scope"] == "diagnostic_output_plus_default_off_policy_runner_fast_path"
    assert audit["invariant_fields_changed"] == {}
    assert {name: getattr(cfg, name, None) for name in CORE_INVARIANT_FIELDS} == before
    assert cfg.behavior_log_enable is False
    assert cfg.checkpoint_bundle_v2_period_gen == 0
    assert cfg.memory_action_window_instrumentation_enable is False
    assert cfg.prediction_demand_action_context_enable is False
    assert cfg.perf_aggperf3_timing_buckets_enable is True
    assert cfg.perf_policy_drive_packed_k_staging_enable is False
    assert cfg.perf_policy_stride_slot_cache_enable is True
    assert cfg.perf_skip_dev_expression_prewarm_enable is False


def test_sampled_parity_v2_keeps_bounded_diagnostic_presence() -> None:
    cfg = _evidence_like_cfg()

    audit = apply_performance_profile(cfg, FAST_EXPLORATORY_PARITY_PROFILE_V2)

    assert audit["invariant_fields_changed"] == {}
    assert cfg.behavior_log_enable is False
    assert cfg.checkpoint_bundle_v2_period_gen == 0
    assert getattr(cfg, "memory_action_window_emit_jsonl") is True
    assert getattr(cfg, "memory_action_window_include_lineage") is False
    assert getattr(cfg, "memory_action_window_include_memory_state") is False
    assert getattr(cfg, "memory_action_window_sample_tick_period") == 8
    assert cfg.prediction_demand_action_context_write_events is False
    assert cfg.prediction_demand_action_context_sign_summary_include_windows is True
    assert cfg.prediction_demand_action_context_sign_summary_write_jsonl is True
    assert cfg.prediction_demand_action_context_sign_summary_max_events_per_run == 64


def test_noop_profile_does_not_mutate_config() -> None:
    cfg = _evidence_like_cfg()
    before = dict(cfg.__dict__)

    audit = apply_performance_profile(cfg, "current_evidence_profile")

    assert audit["scope"] == "no-op"
    assert dict(cfg.__dict__) == before


def test_unknown_profile_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown performance profile"):
        apply_performance_profile(_evidence_like_cfg(), "unsafe_fast_mode")


def test_memory_action_window_sampling_knobs_remain_dynamic() -> None:
    sim_fields = {field.name for field in fields(SimConfig)}
    assert "memory_action_window_sample_tick_period" not in sim_fields
    assert "memory_action_window_sample_generation_period" not in sim_fields
    assert "memory_action_window_sample_first_n_generations" not in sim_fields


def _smoke_env_cfg() -> EnvConfig:
    cfg = EnvConfig(width=6, height=6, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 4
    cfg.max_food = 24
    cfg.patch_world_enable = True
    cfg.patch_world_observation_enable = True
    cfg.wdl_delayed_hazard_enable = True
    cfg.wdl_delayed_hazard_observation_enable = True
    cfg.wdl_delayed_hazard_delay_steps = 4
    cfg.wdl_delayed_hazard_cue_duration_steps = 2
    cfg.wdl_delayed_hazard_active_duration_steps = 3
    cfg.wdl_delayed_hazard_event_period_steps = 12
    cfg.wdl_delayed_hazard_radius = 1
    cfg.wdl_delayed_hazard_intensity = 0.05
    cfg.prediction_demand_enable = True
    cfg.prediction_demand_observation_enable = True
    cfg.prediction_demand_delay_min = 4
    cfg.prediction_demand_delay_max = 6
    cfg.prediction_demand_active_duration = 2
    cfg.prediction_demand_event_period_steps = 10
    cfg.prediction_demand_radius = 1
    return cfg


def _smoke_sim_cfg(seed: int) -> SimConfig:
    cfg = SimConfig(
        seed=int(seed),
        generations=1,
        steps_per_generation=8,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=True,
        policy_deterministic_actions=True,
        policy_tick_stride=2,
    )
    cfg.phase_e_e1_enable = False
    cfg.phase_e_e2_enable = False
    cfg.epi_enable = False
    cfg.enable_cognitive_budget_ledger = False
    cfg.checkpoint_bundle_v2_period_gen = 1
    cfg.memory_action_window_instrumentation_enable = True
    cfg.memory_action_window_emit_jsonl = True
    cfg.prediction_demand_action_context_enable = True
    cfg.prediction_demand_action_context_write_events = True
    cfg.prediction_demand_action_context_sign_summary_enable = True
    return cfg


def _supported_runnercore_env_cfg() -> EnvConfig:
    cfg = _smoke_env_cfg()
    cfg.patch_world_enable = False
    cfg.patch_world_observation_enable = False
    cfg.wdl_delayed_hazard_enable = False
    cfg.wdl_delayed_hazard_observation_enable = False
    cfg.prediction_demand_enable = False
    cfg.prediction_demand_observation_enable = False
    cfg.reproduce_requires_mate = False
    cfg.jungle_predator_pressure_enable = False
    return cfg


def _supported_runnercore_sim_cfg(seed: int) -> SimConfig:
    cfg = _smoke_sim_cfg(seed)
    cfg.chemistry_functional_coupling_enable = False
    cfg.chemistry_repro_viability_enable = False
    cfg.maintenance_enable = False
    cfg.lineage_reproduction_funnel_reporting_enable = False
    cfg.memory_action_window_instrumentation_enable = False
    cfg.memory_action_window_emit_jsonl = False
    cfg.prediction_demand_action_context_enable = False
    cfg.prediction_demand_action_context_write_events = False
    cfg.prediction_demand_action_context_sign_summary_enable = False
    return cfg


def _run_smoke(
    tmp_path: Path,
    *,
    seed: int,
    profile: str | None = None,
    supported_runnercore: bool = False,
) -> dict[str, object]:
    sim_cfg = _supported_runnercore_sim_cfg(seed) if supported_runnercore else _smoke_sim_cfg(seed)
    if profile:
        apply_performance_profile(sim_cfg, profile)
    logger = MetricsLogger(str(tmp_path))
    sim = SimulationRunner(
        env_cfg=_supported_runnercore_env_cfg() if supported_runnercore else _smoke_env_cfg(),
        pop_cfg=PopConfig(population_size=6),
        sim_cfg=sim_cfg,
        logger=logger,
        run_dir=str(tmp_path),
        perf=True,
    )
    sim.run()
    alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
    ids = [int(a.agent_id) for a in alive]
    return {
        "alive_ids": ids,
        "health": [round(float(a.health), 8) for a in alive],
        "energy": [round(float(a.energy), 8) for a in alive],
        "pos_y": np.asarray(sim.env.agent_pos_y[ids], dtype=np.int32).tolist() if ids else [],
        "pos_x": np.asarray(sim.env.agent_pos_x[ids], dtype=np.int32).tolist() if ids else [],
        "final_generation": dict(getattr(logger, "last_row", {}) or {}).get("generation"),
    }


def test_fast_profile_v2_preserves_tiny_smoke_fingerprint(tmp_path: Path) -> None:
    evidence = _run_smoke(tmp_path / "evidence", seed=4321)
    fast = _run_smoke(tmp_path / "fast_v2", seed=4321, profile=FAST_EXPLORATORY_PROFILE_V2)

    assert fast == evidence
    assert (tmp_path / "fast_v2" / "metrics.csv").exists()
    assert (tmp_path / "fast_v2" / "perf_buckets.csv").exists()
    assert not (tmp_path / "fast_v2" / "behaviors.csv").exists()


def test_fast_profile_v3_preserves_tiny_smoke_fingerprint(tmp_path: Path) -> None:
    evidence = _run_smoke(tmp_path / "evidence", seed=5432)
    fast = _run_smoke(tmp_path / "fast_v3", seed=5432, profile=FAST_EXPLORATORY_PROFILE_V3)

    assert fast == evidence
    assert (tmp_path / "fast_v3" / "metrics.csv").exists()
    assert (tmp_path / "fast_v3" / "perf_buckets.csv").exists()
    assert not (tmp_path / "fast_v3" / "behaviors.csv").exists()


def test_w6_runnercore18_bridge_preserves_tiny_smoke_fingerprint(tmp_path: Path) -> None:
    evidence = _run_smoke(tmp_path / "evidence", seed=6543, supported_runnercore=True)
    fast = _run_smoke(
        tmp_path / "runnercore18_bridge",
        seed=6543,
        profile=W6_RUNNERCORE18_BRIDGE_PROFILE,
        supported_runnercore=True,
    )

    assert fast == evidence
    assert (tmp_path / "runnercore18_bridge" / "metrics.csv").exists()
    assert (tmp_path / "runnercore18_bridge" / "perf_buckets.csv").exists()
    assert (tmp_path / "runnercore18_bridge" / "behaviors.csv").exists()


def test_lro3_cpuhot_profile_preserves_tiny_smoke_fingerprint_and_outputs(tmp_path: Path) -> None:
    evidence = _run_smoke(tmp_path / "evidence", seed=7654)
    fast = _run_smoke(tmp_path / "lro3_cpuhot_fast_v1", seed=7654, profile=LRO3_CPUHOT_FAST_PROFILE)

    assert fast == evidence
    assert (tmp_path / "lro3_cpuhot_fast_v1" / "metrics.csv").exists()
    assert (tmp_path / "lro3_cpuhot_fast_v1" / "perf_buckets.csv").exists()
    assert (tmp_path / "lro3_cpuhot_fast_v1" / "behaviors.csv").exists()
