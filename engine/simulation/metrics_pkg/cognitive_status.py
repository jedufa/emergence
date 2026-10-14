from __future__ import annotations

"""Deterministic Stage 0 cognitive status metrics."""

import math
import json
from collections import OrderedDict
from typing import Any, Iterable, List, Mapping

from agents.cognition import (
    MEMORY_LADDER,
    compute_cognitive_budget_ledger,
    express_cognitive_phenotype,
)
from agents.perception import perception_channel_registry
from agents.introspection import implemented_introspection_channel_ids, introspection_channel_value
from agents.self_regulation import implemented_self_regulation_control_ids, self_regulation_runtime_summary
from agents.self_modification import implemented_self_modification_target_ids, self_modification_runtime_summary
from agents.directed_inheritance import implemented_directed_inheritance_channel_ids, directed_inheritance_payload_summary
from simulation.metrics_pkg.metrics_logger_utils import _unwrap_agent, _as_float
from simulation.metrics_pkg.population_reporting import (
    FlatPopulationReportingRecordsV1,
    validate_flat_population_reporting_records,
)
from simulation.metrics_pkg.stats import mean, percentile, std
from evolution.structural_live_semantics import (
    apply_live_structural_semantics,
    stable_json,
    structural_live_semantics_enabled,
)
from simulation.metrics_pkg.trait_capacity_saturation import empty_trait_capacity_payload, trait_capacity_saturation_payloads


def cognitive_status_metric_defaults() -> "OrderedDict[str, float]":
    keys = (
        "cog_capacity_mean",
        "cog_capacity_p50",
        "cog_capacity_p90",
        "cog_capacity_max",
        "cog_capacity_min",
        "cog_capacity_std",
        "cog_capacity_alive_mean",
        "cog_capacity_parent_mean",
        "cog_capacity_survivor_mean",
        "cog_capacity_alive_end_mean",
        "cog_capacity_memory_component_mean",
        "cog_capacity_prediction_component_mean",
        "cog_capacity_drive_component_mean",
        "cog_capacity_perception_component_mean",
        "memory_enabled_fraction",
        "memory_dim_mean",
        "memory_dim_p50",
        "memory_dim_p90",
        "memory_dim_max",
        "memory_dim_min",
        "memory_dim_std",
        "memory_capacity_fraction_mean",
        "memory_capacity_fraction_p90",
        "memory_bucket_0_fraction",
        "memory_bucket_8_fraction",
        "memory_bucket_16_fraction",
        "memory_bucket_32_fraction",
        "memory_bucket_64_fraction",
        "effective_memory_mean",
        "effective_memory_p90",
        "memory_cost_mean",
        "memory_cost_total",
        "memory_cost_hypothetical_mean",
        "memory_cost_hypothetical_total",
        "memory_cost_charged_mean",
        "memory_cost_charged_total",
        "memory_cost_fraction_of_cog_cost_mean",
        "memory_cost_zero_fraction",
        "high_memory_low_cost_fraction",
        "memory_enabled_without_cost_fraction",
        "memory_ablation_active_flag",
        "memory_force_disabled_active_flag",
        "memory_force_min_active_flag",
        "memory_force_max_active_flag",
        "memory_force_default_active_flag",
        "memory_reset_each_tick_active_flag",
        "mem_evo_population_memory_dim_histogram",
        "mem_evo_selected_parent_memory_dim_histogram",
        "mem_evo_descendant_memory_dim_histogram",
        "mem_evo_lineage_memory_dim_summary_json",
        "mem_evo_effective_memory_dim_histogram",
        "trait_capacity_saturation_summary_json",
        "trait_capacity_saturation_lineage_json",
        "trait_capacity_saturation_structural_payload_json",
        "structural_live_semantics_summary_json",
        "mem_evo_use_gru_share",
        "mem_evo_memory_cost_mean",
        "mem_evo_memory_cost_total",
        "mem_evo_memory_dim_selected_minus_population_mean",
        "mem_evo_memory_dim_descendant_minus_initial_mean",
        "mem_evo_memory_dim_wdl_minus_flat_delta",
        "mem_evo_memory_dim_shift_interpretable",
        "memory_state_abs_mean",
        "memory_state_abs_p90",
        "memory_state_delta_mean",
        "memory_state_delta_p90",
        "memory_active_slice_abs_mean",
        "memory_inactive_slice_abs_mean",
        "memory_mask_leakage_mean",
        "memory_nonzero_fraction_mean",
        "memory_effective_utilization_mean",
        "memory_saturation_proxy_mean",
        "memory_reset_count_total",
        "memory_use_measurement_available_flag",
        "memory_dim_entropy",
        "memory_dim_diversity",
        "dominant_memory_dim_fraction",
        "memory_strategy_count_approx",
        "memory_perception_strategy_entropy",
        "memory_perception_strategy_count_approx",
        "high_memory_high_perception_fraction",
        "high_memory_low_perception_fraction",
        "low_memory_high_perception_fraction",
        "memory_growth_without_pressure_flag",
        "memory_pressure_scenario_flag",
        "memory_pressure_food_regrowth_return_flag",
        "gru_enabled_fraction",
        "recurrent_policy_fraction",
        "policy_hidden_size_mean",
        "policy_hidden_size_p90",
        "policy_hidden_size_max",
        "policy_memory_active_fraction",
        "policy_memory_cost_mean",
        "prediction_enabled_fraction",
        "prediction_lr_mean",
        "prediction_lr_p50",
        "prediction_lr_p90",
        "prediction_lr_max",
        "prediction_lr_min",
        "prediction_lr_std",
        "effective_prediction_lr_mean",
        "effective_prediction_lr_p90",
        "prediction_error_mean",
        "prediction_error_p50",
        "prediction_error_p90",
        "prediction_error_max",
        "prediction_error_std",
        "prediction_error_trend_mean",
        "prediction_error_under_ablation_mean",
        "prediction_update_cost_mean",
        "prediction_cost_hypothetical_mean",
        "prediction_cost_hypothetical_total",
        "prediction_cost_charged_mean",
        "prediction_cost_charged_total",
        "prediction_cost_fraction_of_cog_cost_mean",
        "prediction_cost_zero_fraction",
        "high_prediction_low_cost_fraction",
        "prediction_ablation_active_flag",
        "prediction_force_disabled_active_flag",
        "prediction_force_frozen_active_flag",
        "prediction_force_zero_lr_active_flag",
        "prediction_force_default_active_flag",
        "prediction_reset_each_tick_active_flag",
        "prediction_direct_selection_pressure_disabled_flag",
        "prediction_state_abs_mean",
        "prediction_state_abs_p90",
        "prediction_state_delta_mean",
        "prediction_state_delta_p90",
        "prediction_error_delta_mean",
        "prediction_update_count_total",
        "prediction_nonzero_fraction_mean",
        "prediction_effective_utilization_mean",
        "prediction_saturation_proxy_mean",
        "prediction_action_differentiation_mean",
        "prediction_choice_coupling_proxy_mean",
        "prediction_capacity_mean",
        "prediction_capacity_p90",
        "prediction_capacity_max",
        "prediction_capacity_proxy_mean",
        "prediction_capacity_proxy_max",
        "parent_prediction_lr_mean",
        "alive_prediction_lr_mean",
        "dead_prediction_lr_mean",
        "reproduced_prediction_lr_mean",
        "non_reproduced_prediction_lr_mean",
        "prediction_selection_delta_survivor_vs_dead",
        "prediction_lr_entropy",
        "prediction_profile_diversity",
        "dominant_prediction_profile_fraction",
        "prediction_strategy_count_approx",
        "prediction_growth_without_pressure_flag",
        "direct_prediction_error_selection_pressure_present_flag",
        "prediction_pressure_scenario_flag",
        "prediction_pressure_delayed_consequence_flag",
        "memory_prediction_strategy_entropy",
        "perception_prediction_strategy_entropy",
        "memory_perception_prediction_strategy_count_approx",
        "high_memory_high_prediction_fraction",
        "high_perception_high_prediction_fraction",
        "high_prediction_low_memory_fraction",
        "high_prediction_low_perception_fraction",
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
        "drive_regulation_cost_mean",
        "cog_cost_mean",
        "cog_cost_p90",
        "cog_cost_max",
        "cog_cost_total",
        "cog_cost_fraction_of_energy_mean",
        "cog_cost_fraction_of_energy_p90",
        "cognition_under_starvation_fraction",
        "high_cognition_starvation_rate",
        "low_cognition_starvation_rate",
        "parent_cog_capacity_mean",
        "alive_cog_capacity_mean",
        "dead_cog_capacity_mean",
        "reproduced_cog_capacity_mean",
        "non_reproduced_cog_capacity_mean",
        "cognitive_selection_delta_parent_vs_alive",
        "cognitive_selection_delta_survivor_vs_dead",
        "cognitive_selection_delta_survivor_vs_dead_available_flag",
        "memory_selection_delta_parent_vs_alive",
        "prediction_selection_delta_parent_vs_alive",
        "drive_regulation_selection_delta_parent_vs_alive",
        "cognitive_trait_diversity",
        "cognitive_capacity_entropy",
        "memory_bucket_entropy",
        "policy_type_entropy",
        "drive_profile_diversity",
        "prediction_lr_diversity",
        "cognitive_strategy_count_approx",
        "juvenile_cog_capacity_mean",
        "adult_cog_capacity_mean",
        "elder_cog_capacity_mean",
        "cognition_by_age_slope",
        "memory_by_age_slope",
        "cognitive_cost_by_age_slope",
        "developmental_expression_active_fraction",
        "developmental_expression_factor_mean",
        "developmental_expression_factor_p50",
        "developmental_expression_factor_p90",
        "juvenile_fraction",
        "adult_fraction",
        "mature_fraction",
        "juvenile_memory_dim_mean",
        "adult_memory_dim_mean",
        "juvenile_sensor_radius_mean",
        "adult_sensor_radius_mean",
        "juvenile_sensor_channel_count_mean",
        "adult_sensor_channel_count_mean",
        "juvenile_prediction_lr_mean",
        "adult_prediction_lr_mean",
        "perception_by_age_slope",
        "prediction_by_age_slope",
        "developmental_cost_hypothetical_mean",
        "developmental_cost_charged_mean",
        "maturation_cost_hypothetical_mean",
        "maturation_cost_charged_mean",
        "developmental_ablation_active_flag",
        "developmental_force_adult_active_flag",
        "developmental_force_juvenile_active_flag",
        "developmental_force_legacy_active_flag",
        "developmental_growth_disabled_active_flag",
        "developmental_cost_charging_active_flag",
        "parent_maturity_age_mean",
        "alive_maturity_age_mean",
        "dead_maturity_age_mean",
        "reproduced_maturity_age_mean",
        "non_reproduced_maturity_age_mean",
        "maturity_age_selection_delta_parent_vs_alive",
        "developmental_expression_selection_delta_parent_vs_alive",
        "juvenile_survival_rate",
        "adult_survival_rate",
        "juvenile_reproduction_attempt_rate",
        "adult_reproduction_attempt_rate",
        "delayed_maturity_lineage_fraction",
        "developmental_stage_entropy",
        "maturity_age_entropy",
        "developmental_strategy_diversity",
        "dominant_developmental_strategy_fraction",
        "developmental_strategy_count_approx",
        "developmental_cost_zero_fraction",
        "high_adult_cognition_low_developmental_cost_fraction",
        "adult_cognition_from_birth_fraction",
        "juvenile_high_cognition_fraction",
        "developmental_growth_without_pressure_flag",
        "direct_development_reward_present_flag",
        "developmental_pressure_scenario_flag",
        "developmental_pressure_delayed_maturity_flag",
        "maturation_cost_active_flag",
        "introspection_enabled_fraction",
        "introspection_channel_active_count_mean",
        "introspection_channel_active_count_p50",
        "introspection_channel_active_count_p90",
        "introspection_channel_active_count_max",
        "introspection_channel_active_count_min",
        "introspection_channel_active_count_std",
        "introspection_channel_active_fraction_mean",
        "introspection_capacity_component_mean",
        "introspection_capacity_component_p90",
        "introspection_channel_energy_enabled_fraction",
        "introspection_channel_health_enabled_fraction",
        "introspection_channel_hunger_enabled_fraction",
        "introspection_channel_pain_enabled_fraction",
        "introspection_channel_fatigue_enabled_fraction",
        "introspection_channel_fear_enabled_fraction",
        "introspection_channel_drive_conflict_proxy_enabled_fraction",
        "introspection_channel_age_normalized_enabled_fraction",
        "introspection_channel_maturity_progress_enabled_fraction",
        "introspection_channel_developmental_expression_factor_enabled_fraction",
        "introspection_channel_memory_capacity_fraction_enabled_fraction",
        "introspection_channel_memory_saturation_proxy_enabled_fraction",
        "introspection_channel_prediction_error_enabled_fraction",
        "introspection_channel_prediction_update_activity_enabled_fraction",
        "introspection_channel_cognitive_cost_fraction_enabled_fraction",
        "internal_energy_signal_mean",
        "internal_health_signal_mean",
        "internal_drive_conflict_signal_mean",
        "internal_prediction_error_signal_mean",
        "internal_memory_saturation_signal_mean",
        "internal_cog_cost_burden_signal_mean",
        "internal_developmental_expression_signal_mean",
        "introspection_cost_hypothetical_mean",
        "introspection_cost_hypothetical_total",
        "introspection_cost_charged_mean",
        "introspection_cost_charged_total",
        "introspection_cost_fraction_of_cog_cost_mean",
        "high_introspection_low_cost_fraction",
        "introspection_enabled_without_cost_fraction",
        "introspection_channel_mask_entropy",
        "introspection_channel_mask_diversity",
        "dominant_introspection_channel_mask_fraction",
        "introspection_strategy_count_approx",
        "introspection_memory_strategy_entropy",
        "introspection_prediction_strategy_entropy",
        "introspection_perception_strategy_entropy",
        "introspection_development_strategy_entropy",
        "full_cognitive_strategy_count_approx",
        "high_introspection_high_memory_fraction",
        "high_introspection_high_prediction_fraction",
        "high_introspection_high_perception_fraction",
        "high_introspection_developmental_fraction",
        "introspection_cost_zero_fraction",
        "introspection_growth_without_pressure_flag",
        "direct_introspection_reward_present_flag",
        "direct_internal_state_reward_present_flag",
        "introspection_ablation_active_flag",
        "introspection_force_no_active_flag",
        "introspection_force_minimal_active_flag",
        "introspection_force_all_active_flag",
        "introspection_force_default_active_flag",
        "introspection_mask_to_neutral_active_flag",
        "introspection_cost_charging_active_flag",
        "introspection_pressure_scenario_flag",
        "introspection_pressure_drive_conflict_flag",
        "introspection_policy_input_active_flag",
        "self_regulation_enabled_fraction",
        "self_regulation_control_active_count_mean",
        "self_regulation_control_active_count_p50",
        "self_regulation_control_active_count_p90",
        "self_regulation_control_active_count_max",
        "self_regulation_control_active_count_min",
        "self_regulation_control_active_count_std",
        "self_regulation_control_active_fraction_mean",
        "self_regulation_capacity_component_mean",
        "self_regulation_capacity_component_p90",
        "self_regulation_control_exploration_pressure_adjustment_enabled_fraction",
        "self_regulation_control_action_temperature_adjustment_enabled_fraction",
        "self_regulation_control_drive_gain_adjustment_enabled_fraction",
        "self_regulation_control_reproduction_threshold_adjustment_enabled_fraction",
        "self_regulation_control_prediction_lr_adjustment_enabled_fraction",
        "self_regulation_control_delta_abs_mean",
        "self_regulation_control_delta_abs_p90",
        "self_regulation_control_offset_abs_mean",
        "self_regulation_control_offset_abs_p90",
        "self_regulation_control_clipped_fraction",
        "self_regulation_control_saturation_fraction",
        "self_regulation_control_application_count_total",
        "self_regulation_control_neutral_fraction",
        "self_regulation_state_reset_count_total",
        "self_regulation_decay_applied_count_total",
        "effective_exploration_pressure_mean",
        "effective_action_temperature_mean",
        "effective_drive_gain_mean",
        "effective_reproduction_threshold_mean",
        "effective_self_regulated_prediction_lr_mean",
        "self_regulation_cost_hypothetical_mean",
        "self_regulation_cost_hypothetical_total",
        "self_regulation_cost_charged_mean",
        "self_regulation_cost_charged_total",
        "self_regulation_cost_fraction_of_cog_cost_mean",
        "high_self_regulation_low_cost_fraction",
        "self_regulation_enabled_without_cost_fraction",
        "self_regulation_control_mask_entropy",
        "self_regulation_control_mask_diversity",
        "dominant_self_regulation_control_mask_fraction",
        "self_regulation_strategy_count_approx",
        "introspection_self_regulation_strategy_entropy",
        "high_introspection_high_self_regulation_fraction",
        "self_regulation_without_introspection_fraction",
        "introspection_without_self_regulation_fraction",
        "self_regulation_policy_input_independent_flag",
        "self_regulation_cost_zero_fraction",
        "self_regulation_growth_without_pressure_flag",
        "direct_self_regulation_reward_present_flag",
        "self_modification_present_flag",
        "genome_editing_present_flag",
        "self_regulation_ablation_active_flag",
        "self_regulation_force_no_active_flag",
        "self_regulation_force_all_active_flag",
        "self_regulation_force_default_active_flag",
        "self_regulation_mask_to_neutral_active_flag",
        "self_regulation_freeze_state_active_flag",
        "self_regulation_reset_each_tick_active_flag",
        "self_regulation_decay_disabled_active_flag",
        "self_regulation_cost_charging_active_flag",
        "self_regulation_pressure_scenario_flag",
        "self_regulation_pressure_hunger_fear_conflict_flag",
        "self_modification_enabled_fraction",
        "self_modification_target_active_count_mean",
        "self_modification_target_active_count_p50",
        "self_modification_target_active_count_p90",
        "self_modification_target_active_count_max",
        "self_modification_target_active_count_min",
        "self_modification_target_active_count_std",
        "self_modification_target_active_fraction_mean",
        "self_modification_capacity_component_mean",
        "self_modification_capacity_component_p90",
        "self_modification_target_exploration_bias_lifetime_offset_enabled_fraction",
        "self_modification_target_action_temperature_lifetime_offset_enabled_fraction",
        "self_modification_target_drive_gain_lifetime_offset_enabled_fraction",
        "self_modification_target_reproduction_threshold_lifetime_offset_enabled_fraction",
        "self_modification_target_prediction_lr_lifetime_offset_enabled_fraction",
        "self_modification_delta_abs_mean",
        "self_modification_delta_abs_p90",
        "self_modification_offset_abs_mean",
        "self_modification_offset_abs_p90",
        "self_modification_clipped_fraction",
        "self_modification_saturation_fraction",
        "self_modification_event_count_total",
        "self_modification_neutral_command_fraction",
        "self_modification_state_reset_count_total",
        "self_modification_decay_applied_count_total",
        "self_modification_lifetime_persistence_mean",
        "self_modification_reversal_count_total",
        "modified_exploration_bias_mean",
        "modified_action_temperature_mean",
        "modified_drive_gain_mean",
        "modified_reproduction_threshold_mean",
        "modified_prediction_lr_mean",
        "self_modification_cost_hypothetical_mean",
        "self_modification_cost_hypothetical_total",
        "self_modification_cost_charged_mean",
        "self_modification_cost_charged_total",
        "self_modification_cost_fraction_of_cog_cost_mean",
        "high_self_modification_low_cost_fraction",
        "self_modification_enabled_without_cost_fraction",
        "self_modification_target_mask_entropy",
        "self_modification_target_mask_diversity",
        "dominant_self_modification_target_mask_fraction",
        "self_modification_strategy_count_approx",
        "introspection_self_modification_strategy_entropy",
        "self_regulation_self_modification_strategy_entropy",
        "high_introspection_high_self_modification_fraction",
        "high_self_regulation_high_self_modification_fraction",
        "self_modification_without_introspection_fraction",
        "self_modification_without_self_regulation_fraction",
        "self_modification_cost_zero_fraction",
        "self_modification_growth_without_pressure_flag",
        "direct_self_modification_reward_present_flag",
        "host_access_present_flag",
        "source_code_modification_present_flag",
        "self_modification_ablation_active_flag",
        "self_modification_force_no_active_flag",
        "self_modification_force_all_active_flag",
        "self_modification_force_default_active_flag",
        "self_modification_mask_to_neutral_active_flag",
        "self_modification_freeze_state_active_flag",
        "self_modification_reset_each_tick_active_flag",
        "self_modification_reset_on_reproduction_active_flag",
        "self_modification_decay_disabled_active_flag",
        "self_modification_cost_charging_active_flag",
        "self_modification_pressure_scenario_flag",
        "self_modification_pressure_persistent_risk_flag",
        "self_modification_policy_output_active_flag",
        "directed_inheritance_enabled_fraction",
        "directed_inheritance_channel_active_count_mean",
        "directed_inheritance_channel_active_count_p50",
        "directed_inheritance_channel_active_count_p90",
        "directed_inheritance_channel_active_count_max",
        "directed_inheritance_channel_active_count_min",
        "directed_inheritance_channel_active_count_std",
        "directed_inheritance_channel_active_fraction_mean",
        "directed_inheritance_capacity_component_mean",
        "directed_inheritance_capacity_component_p90",
        "directed_inheritance_channel_offspring_developmental_expression_bias_enabled_fraction",
        "directed_inheritance_channel_offspring_juvenile_cognition_scale_bias_enabled_fraction",
        "directed_inheritance_channel_offspring_memory_expression_bias_enabled_fraction",
        "directed_inheritance_channel_offspring_prediction_lr_bias_enabled_fraction",
        "directed_inheritance_channel_offspring_exploration_bias_marker_enabled_fraction",
        "directed_inheritance_channel_offspring_reproduction_threshold_bias_enabled_fraction",
        "directed_inheritance_delta_abs_mean",
        "directed_inheritance_delta_abs_p90",
        "directed_inheritance_payload_abs_mean",
        "directed_inheritance_payload_abs_p90",
        "directed_inheritance_clipped_fraction",
        "directed_inheritance_saturation_fraction",
        "directed_inheritance_event_count_total",
        "directed_inheritance_neutral_command_fraction",
        "directed_inheritance_payload_reset_count_total",
        "directed_inheritance_payload_applied_fraction",
        "directed_inheritance_payload_frozen_fraction",
        "offspring_developmental_expression_bias_mean",
        "offspring_juvenile_cognition_scale_bias_mean",
        "offspring_memory_expression_bias_mean",
        "offspring_prediction_lr_bias_mean",
        "offspring_exploration_bias_marker_mean",
        "offspring_reproduction_threshold_bias_mean",
        "directed_inheritance_cost_hypothetical_mean",
        "directed_inheritance_cost_hypothetical_total",
        "directed_inheritance_cost_charged_mean",
        "directed_inheritance_cost_charged_total",
        "directed_inheritance_cost_fraction_of_cog_cost_mean",
        "high_directed_inheritance_low_cost_fraction",
        "directed_inheritance_enabled_without_cost_fraction",
        "directed_inheritance_channel_mask_entropy",
        "directed_inheritance_channel_mask_diversity",
        "dominant_directed_inheritance_channel_mask_fraction",
        "directed_inheritance_strategy_count_approx",
        "self_modification_directed_inheritance_strategy_entropy",
        "self_regulation_directed_inheritance_strategy_entropy",
        "high_self_modification_high_directed_inheritance_fraction",
        "high_self_regulation_high_directed_inheritance_fraction",
        "directed_inheritance_without_self_modification_fraction",
        "self_modification_without_directed_inheritance_fraction",
        "directed_inheritance_cost_zero_fraction",
        "directed_inheritance_growth_without_pressure_flag",
        "direct_directed_inheritance_reward_present_flag",
        "direct_parental_investment_reward_present_flag",
        "direct_offspring_quality_reward_present_flag",
        "mutation_rate_control_present_flag",
        "recombination_control_present_flag",
        "directed_inheritance_ablation_active_flag",
        "directed_inheritance_force_no_active_flag",
        "directed_inheritance_force_all_active_flag",
        "directed_inheritance_force_default_active_flag",
        "directed_inheritance_mask_to_neutral_active_flag",
        "directed_inheritance_freeze_payload_active_flag",
        "directed_inheritance_reset_on_birth_active_flag",
        "directed_inheritance_reset_on_maturity_active_flag",
        "directed_inheritance_cost_charging_active_flag",
        "directed_inheritance_pressure_scenario_flag",
        "directed_inheritance_pressure_developmental_bias_flag",
        "directed_inheritance_policy_output_active_flag",
        "lineage_cog_capacity_mean",
        "lineage_memory_mean",
        "lineage_prediction_lr_mean",
        "lineage_cognitive_cost_mean",
        "top_lineage_cog_capacity",
        "top_lineage_memory_dim",
        "lineage_cognitive_diversity",
        "cognition_cost_zero_fraction",
        "high_cognition_low_cost_fraction",
        "memory_enabled_without_cost_fraction",
        "prediction_enabled_without_cost_fraction",
        "cognition_growth_without_environment_pressure_flag",
        "direct_cognition_reward_present_flag",
        "direct_internal_selection_pressure_present_flag",
        "cognitive_metric_only_selection_risk_flag",
        "cognition_capacity_in_flat_world_baseline",
        "sensor_channel_active_count_mean",
        "sensor_channel_active_count_p50",
        "sensor_channel_active_count_p90",
        "sensor_channel_active_count_max",
        "sensor_channel_active_count_min",
        "sensor_channel_active_count_std",
        "sensor_channel_potential_count",
        "sensor_channel_active_fraction_mean",
        "perception_capacity_component_mean",
        "perception_capacity_component_p90",
        "perception_cost_mean",
        "perception_cost_p90",
        "perception_cost_max",
        "perception_cost_total",
        "perception_cost_charged_mean",
        "perception_cost_charged_total",
        "perception_cost_hypothetical_mean",
        "perception_cost_fraction_of_cog_cost_mean",
        "parent_sensor_channel_count_mean",
        "alive_sensor_channel_count_mean",
        "perception_selection_delta_parent_vs_alive",
        "sensor_channel_mask_diversity",
        "sensor_channel_mask_entropy",
        "dominant_sensor_channel_mask_fraction",
        "sensor_channel_strategy_count_approx",
        "perception_cost_zero_fraction",
        "high_perception_low_cost_fraction",
        "sensor_channels_enabled_without_cost_fraction",
        "perception_growth_without_pressure_flag",
        "sensor_channel_mask_ablation_active_flag",
        "sensor_channel_force_all_active_flag",
        "sensor_channel_force_minimal_active_flag",
        "sensor_radius_mean",
        "sensor_radius_p50",
        "sensor_radius_p90",
        "sensor_radius_max",
        "sensor_radius_min",
        "sensor_radius_std",
        "sensor_radius_1_fraction",
        "sensor_radius_2_fraction",
        "sensor_radius_3_fraction",
        "observed_cell_count_mean",
        "observed_cell_count_p50",
        "observed_cell_count_p90",
        "observed_cell_count_max",
        "active_cell_fraction_mean",
        "cell_channel_capacity_mean",
        "cell_channel_capacity_p90",
        "cell_channel_capacity_max",
        "spatial_perception_capacity_mean",
        "spatial_perception_capacity_p90",
        "sensor_radius_cost_hypothetical_mean",
        "sensor_radius_cost_hypothetical_total",
        "sensor_radius_cost_charged_mean",
        "sensor_radius_cost_charged_total",
        "spatial_perception_cost_mean",
        "spatial_perception_cost_p90",
        "spatial_perception_cost_max",
        "perception_cost_total_mean",
        "perception_cost_total_p90",
        "parent_sensor_radius_mean",
        "alive_sensor_radius_mean",
        "sensor_radius_selection_delta_parent_vs_alive",
        "parent_cell_channel_capacity_mean",
        "alive_cell_channel_capacity_mean",
        "cell_channel_capacity_selection_delta_parent_vs_alive",
        "sensor_radius_entropy",
        "sensor_radius_diversity",
        "dominant_sensor_radius_fraction",
        "perception_strategy_entropy",
        "perception_strategy_count_approx",
        "sensor_radius_cost_zero_fraction",
        "high_radius_low_cost_fraction",
        "high_spatial_perception_low_cost_fraction",
        "sensor_radius_enabled_without_cost_fraction",
        "radius_growth_without_pressure_flag",
        "sensor_radius_ablation_active_flag",
        "sensor_radius_force_min_active_flag",
        "sensor_radius_force_max_active_flag",
        "sensor_radius_force_default_active_flag",
    )
    json_keys = {
        "mem_evo_population_memory_dim_histogram",
        "mem_evo_selected_parent_memory_dim_histogram",
        "mem_evo_descendant_memory_dim_histogram",
        "mem_evo_lineage_memory_dim_summary_json",
        "mem_evo_effective_memory_dim_histogram",
        "trait_capacity_saturation_summary_json",
        "trait_capacity_saturation_lineage_json",
        "trait_capacity_saturation_structural_payload_json",
        "structural_live_semantics_summary_json",
    }
    out = OrderedDict((k, "{}" if k in json_keys else 0.0) for k in keys)
    for channel in perception_channel_registry():
        if channel.active_now and not channel.future_reserved:
            out[f"sensor_channel_{channel.channel_id}_enabled_fraction"] = 0.0
            if channel.patch_channel:
                out[f"patch_channel_{channel.channel_id}_enabled_fraction"] = 0.0
    return out


def _safe_agents(agents: Iterable[Any] | None) -> list[Any]:
    out = []
    for raw in list(agents or []):
        a = _unwrap_agent(raw)
        if a is None:
            continue
        out.append(a)
    out.sort(key=lambda a: int(getattr(a, "agent_id", 0)))
    return out


def _entropy(values: list[Any]) -> float:
    if not values:
        return 0.0
    counts: dict[Any, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    n = float(len(values))
    h = 0.0
    for c in counts.values():
        p = float(c) / n
        if p > 0.0:
            h -= p * math.log(p, 2)
    return float(h)


def _diversity(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(len({round(float(v), 6) for v in values}))


def _slope(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    x = xs[:n]
    y = ys[:n]
    mx = mean(x)
    my = mean(y)
    denom = sum((v - mx) ** 2 for v in x)
    if denom <= 1e-12:
        return 0.0
    return float(sum((x[i] - mx) * (y[i] - my) for i in range(n)) / denom)


def _drive_conflict(a: Any) -> float:
    d = getattr(a, "drives", None)
    try:
        if d is None or len(d) < 4:
            return 0.0
        hunger = float(d[0])
        pain = float(d[1])
        fatigue = float(d[2])
        fear = float(d[3])
        return float(abs(hunger - fatigue) + abs(pain - fear))
    except Exception:
        return 0.0


def _prediction_state_summary(st: Any) -> dict[str, float]:
    vals: list[float] = []
    try:
        vals.extend(float(x) for x in list(getattr(st, "exp_dE", []) or []))
    except Exception:
        pass
    try:
        vals.extend(float(x) for x in list(getattr(st, "exp_dInj", []) or []))
    except Exception:
        pass
    abs_vals = [abs(float(v)) for v in vals]
    n = float(len(abs_vals))
    nonzero = sum(1 for v in abs_vals if v > 1e-7)
    saturation = sum(1 for v in abs_vals if v > 0.95)
    return {
        "prediction_state_abs": mean(abs_vals),
        "prediction_state_abs_p90": percentile(abs_vals, 90),
        "prediction_nonzero_fraction": float(nonzero) / n if n > 0.0 else 0.0,
        "prediction_effective_utilization": float(nonzero) / n if n > 0.0 else 0.0,
        "prediction_saturation_proxy": float(saturation) / n if n > 0.0 else 0.0,
        "prediction_action_differentiation": std(vals),
    }


def _lineage_id(a: Any) -> int:
    st = getattr(a, "stats", None)
    for name in ("patch_habitat_lineage_id", "lineage_id", "founder_lineage_id"):
        try:
            v = getattr(st, name)
            return int(v)
        except Exception:
            pass
    return 0


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _memory_histogram_payload(values: Iterable[Any]) -> dict[str, Any]:
    vals: list[int] = []
    for value in values:
        try:
            vals.append(int(round(float(value))))
        except Exception:
            continue
    counts = {str(int(b)): 0 for b in MEMORY_LADDER}
    other = 0
    for value in vals:
        key = str(int(value))
        if key in counts:
            counts[key] += 1
        else:
            other += 1
    total = float(len(vals))
    fractions = {key: (float(count) / total if total > 0.0 else 0.0) for key, count in counts.items()}
    return {
        "buckets": counts,
        "count": int(len(vals)),
        "fractions": fractions,
        "max": float(max(vals)) if vals else 0.0,
        "mean": mean([float(v) for v in vals]) if vals else 0.0,
        "other": int(other),
        "p90": percentile([float(v) for v in vals], 90) if vals else 0.0,
    }


def _lineage_memory_payload(recs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[int, list[Mapping[str, Any]]] = {}
    for rec in recs:
        try:
            groups.setdefault(int(rec.get("lineage", 0)), []).append(rec)
        except Exception:
            groups.setdefault(0, []).append(rec)
    lineages: list[dict[str, Any]] = []
    for lineage, members in sorted(groups.items(), key=lambda kv: int(kv[0])):
        memory = _vals(members, "genetic_memory")
        active_memory = _vals(members, "memory")
        effective = _vals(members, "effective_memory")
        recurrent_share = (
            float(sum(1 for rec in members if bool(rec.get("recurrent_state")))) / float(len(members))
            if members
            else 0.0
        )
        lineages.append(
            {
                "count": int(len(members)),
                "effective_memory_dim_mean": mean(effective) if effective else 0.0,
                "lineage": int(lineage),
                "active_memory_dim_mean": mean(active_memory) if active_memory else 0.0,
                "memory_dim_histogram": _memory_histogram_payload(memory),
                "memory_dim_mean": mean(memory) if memory else 0.0,
                "use_gru_share": float(recurrent_share),
            }
        )
    return {
        "lineage_count": int(len(lineages)),
        "lineages": lineages,
        "mean_lineage_memory_dim_mean": mean([float(x["memory_dim_mean"]) for x in lineages]) if lineages else 0.0,
    }


def _agent_records(agents: list[Any], *, env_cfg: Any, sim_cfg: Any) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    for a in agents:
        g = getattr(a, "genome", None)
        phenotype = express_cognitive_phenotype(g, agent=a, env_cfg=env_cfg, sim_cfg=sim_cfg)
        ledger = compute_cognitive_budget_ledger(phenotype, agent=a, env_cfg=env_cfg, sim_cfg=sim_cfg)
        self_reg_summary = self_regulation_runtime_summary(getattr(a, "self_regulation_state", None))
        self_mod_summary = self_modification_runtime_summary(getattr(a, "self_modification_state", None))
        directed_summary = directed_inheritance_payload_summary(getattr(a, "directed_inheritance_payload", None))
        st = getattr(a, "stats", None)
        pred_n = int(getattr(st, "pred_err_count", 0) or 0) if st is not None else 0
        pred_err = 0.0
        if pred_n > 0:
            pred_err = _as_float(getattr(st, "pred_err_sum", 0.0), 0.0) / float(pred_n)
        pred_state = _prediction_state_summary(st)
        pred_update_count = int(getattr(st, "prediction_update_count", 0) or 0) if st is not None else 0
        pred_delta_sum = _as_float(getattr(st, "prediction_state_delta_abs_sum", 0.0), 0.0) if st is not None else 0.0
        pred_delta_max = _as_float(getattr(st, "prediction_state_delta_abs_max", 0.0), 0.0) if st is not None else 0.0
        recs.append(
            {
                "agent_id": int(getattr(a, "agent_id", 0)),
                "capacity": float(phenotype.cognitive_capacity),
                "genetic_memory": float(phenotype.genetic_memory_dim),
                "memory": float(phenotype.active_memory_capacity),
                "effective_memory": float(phenotype.effective_memory_capacity),
                "memory_enabled": bool(phenotype.memory_enabled),
                "policy_memory": bool(phenotype.policy_memory_enabled),
                "recurrent_state": bool(phenotype.recurrent_state_enabled),
                "memory_capacity_fraction": float(phenotype.memory_capacity_fraction),
                "memory_ladder_index": float(phenotype.memory_ladder_index),
                "memory_cost_hyp": float(ledger.memory_upkeep_cost_hypothetical),
                "memory_cost_charged": float(ledger.memory_upkeep_cost_charged),
                "memory_cost_fraction": float(ledger.memory_cost_fraction_of_cog_cost),
                "memory_ablate": bool(phenotype.memory_ablation_active),
                "memory_force_disabled": bool(phenotype.force_memory_disabled_active),
                "memory_force_min": bool(phenotype.force_memory_min_active),
                "memory_force_max": bool(phenotype.force_memory_max_active),
                "memory_force_default": bool(phenotype.force_memory_default_active),
                "memory_reset_each_tick": bool(phenotype.reset_memory_each_tick_active),
                "hidden": float(phenotype.policy_hidden_size),
                "pred_enabled": bool(phenotype.active_prediction_capability),
                "pred_lr": float(phenotype.prediction_learning_rate),
                "effective_pred_lr": float(phenotype.effective_prediction_learning_rate),
                "prediction_capacity": float(phenotype.prediction_capacity),
                "mutation_rate": float(getattr(g, "mutation_rate", 0.0)),
                "action_temperature": float(getattr(g, "action_temperature", 1.0)),
                "reproduction_threshold": float(getattr(g, "reproduction_threshold", 30.0)),
                "module_count": float(len(list(getattr(g, "modules", []) or []))),
                "connection_count": float(len(list(getattr(g, "connections", []) or []))),
                "c1_operator_noop_count": float(getattr(g, "c1_operator_noop_count", 0.0)),
                "c1_capacity_blocked_count": float(getattr(g, "c1_capacity_blocked_count", 0.0)),
                "c1_capacity_blocked_reason_counts_json": str(
                    getattr(g, "c1_capacity_blocked_reason_counts_json", "{}") or "{}"
                ),
                "c1_op_count__bounded_rewire": float(getattr(g, "c1_op_count__bounded_rewire", 0.0)),
                "c1_op_count__duplicate_with_decay": float(getattr(g, "c1_op_count__duplicate_with_decay", 0.0)),
                "c1_op_count__edge_split_insertion": float(getattr(g, "c1_op_count__edge_split_insertion", 0.0)),
                "c1_op_count__fusion_merge_compression": float(getattr(g, "c1_op_count__fusion_merge_compression", 0.0)),
                "c1_op_count__gate_drift": float(getattr(g, "c1_op_count__gate_drift", 0.0)),
                "c1_op_count__motif_duplication": float(getattr(g, "c1_op_count__motif_duplication", 0.0)),
                "c1_op_count__promoter_expression_mutation": float(getattr(g, "c1_op_count__promoter_expression_mutation", 0.0)),
                "b3_locality_dropped_connections": float(getattr(g, "b3_locality_dropped_connections", 0.0)),
                "prediction_state_available": bool(phenotype.prediction_state_available),
                "prediction_update_enabled": bool(phenotype.prediction_update_enabled),
                "pred_err": float(pred_err),
                "prediction_cost_hyp": float(ledger.prediction_update_cost_hypothetical),
                "prediction_cost_charged": float(ledger.prediction_update_cost_charged),
                "prediction_cost_fraction": float(ledger.prediction_cost_fraction_of_cog_cost),
                "prediction_ablate": bool(phenotype.prediction_ablation_active),
                "prediction_force_disabled": bool(phenotype.force_prediction_disabled_active),
                "prediction_force_frozen": bool(phenotype.force_prediction_frozen_active),
                "prediction_force_zero_lr": bool(phenotype.force_prediction_zero_lr_active),
                "prediction_force_default": bool(phenotype.force_prediction_default_active),
                "prediction_reset_each_tick": bool(phenotype.prediction_reset_each_tick_active),
                "direct_prediction_pressure": bool(phenotype.direct_prediction_selection_pressure_present),
                "prediction_state_abs": float(pred_state["prediction_state_abs"]),
                "prediction_state_abs_p90": float(pred_state["prediction_state_abs_p90"]),
                "prediction_state_delta": float(pred_delta_sum) / float(pred_update_count) if pred_update_count > 0 else 0.0,
                "prediction_state_delta_p90": float(pred_delta_max),
                "prediction_update_count": float(pred_update_count),
                "prediction_nonzero_fraction": float(pred_state["prediction_nonzero_fraction"]),
                "prediction_effective_utilization": float(pred_state["prediction_effective_utilization"]),
                "prediction_saturation_proxy": float(pred_state["prediction_saturation_proxy"]),
                "prediction_action_differentiation": float(pred_state["prediction_action_differentiation"]),
                "drive_gain": float(phenotype.drive_logit_gain),
                "drive_noise": float(phenotype.drive_noise),
                "drive_inertia": float(phenotype.drive_inertia),
                "hunger": float(phenotype.hunger_sensitivity),
                "fear": float(phenotype.fear_sensitivity),
                "pain": float(phenotype.pain_sensitivity),
                "fatigue": float(phenotype.fatigue_sensitivity),
                "cost": float(ledger.cognitive_cost_per_tick),
                "memory_cost": float(ledger.memory_upkeep_cost),
                "policy_memory_cost": float(ledger.policy_memory_cost),
                "cost_frac": float(ledger.cognitive_cost_fraction_of_energy_budget),
                "memory_component": float(phenotype.cognitive_capacity_memory_component),
                "prediction_component": float(phenotype.cognitive_capacity_prediction_component),
                "drive_component": float(phenotype.cognitive_capacity_drive_component),
                "perception_component": float(phenotype.cognitive_capacity_perception_component),
                "sensor_channel_count": float(phenotype.active_sensor_channel_count),
                "sensor_channel_potential": float(phenotype.potential_sensor_channel_count),
                "sensor_channel_mask": int(phenotype.active_sensor_channel_mask),
                "sensor_channel_ids": tuple(phenotype.active_sensor_channel_ids),
                "perception_cost": float(ledger.perception_cost_total),
                "perception_cost_hyp": float(ledger.perception_cost_hypothetical),
                "perception_cost_charged": float(ledger.perception_cost_charged),
                "sensor_radius": float(phenotype.active_sensor_radius),
                "observed_cells": float(phenotype.active_observed_cell_count),
                "active_cell_fraction": float(phenotype.active_cell_fraction),
                "cell_channel_capacity": float(phenotype.cell_channel_capacity),
                "spatial_perception_capacity": float(phenotype.spatial_perception_capacity),
                "sensor_radius_cost_hyp": float(ledger.sensor_radius_cost_hypothetical),
                "sensor_radius_cost_charged": float(ledger.sensor_radius_cost_charged),
                "spatial_perception_cost": float(ledger.spatial_perception_cost),
                "sensor_channel_ablate": bool(phenotype.sensor_channel_mask_ablation_active),
                "sensor_channel_force_all": bool(phenotype.sensor_channel_force_all_active),
                "sensor_channel_force_min": bool(phenotype.sensor_channel_force_minimal_active),
                "sensor_radius_ablate": bool(phenotype.sensor_radius_ablation_active),
                "sensor_radius_force_min": bool(phenotype.sensor_radius_force_min_active),
                "sensor_radius_force_max": bool(phenotype.sensor_radius_force_max_active),
                "sensor_radius_force_default": bool(phenotype.sensor_radius_force_default_active),
                "dev_active": bool(phenotype.developmental_expression_active),
                "age": float(phenotype.age),
                "dev_stage": str(phenotype.developmental_stage),
                "maturity_age": float(phenotype.maturity_age),
                "is_juvenile": bool(phenotype.is_juvenile),
                "is_mature": bool(phenotype.is_mature),
                "dev_factor": float(phenotype.developmental_expression_factor),
                "dev_growth_progress": float(phenotype.cognitive_growth_progress),
                "dev_cost_hyp": float(ledger.developmental_cognition_cost_hypothetical),
                "dev_cost_charged": float(ledger.developmental_cognition_cost_charged),
                "maturation_cost_hyp": float(ledger.maturation_cost_hypothetical),
                "maturation_cost_charged": float(ledger.maturation_cost_charged),
                "dev_ablate": bool(phenotype.developmental_ablation_active),
                "dev_force_adult": bool(phenotype.force_adult_cognition_active),
                "dev_force_juvenile": bool(phenotype.force_juvenile_cognition_active),
                "dev_force_legacy": bool(phenotype.force_legacy_full_cognition_active),
                "dev_growth_disabled": bool(phenotype.developmental_growth_disabled_active),
                "dev_cost_charging": bool(ledger.developmental_cost_charging_enabled),
                "adult_from_birth": bool(phenotype.force_adult_cognition_active or (phenotype.age == 0 and phenotype.developmental_expression_factor >= 0.999)),
                "memory_before_dev": float(phenotype.active_memory_dim_before_development),
                "memory_after_dev": float(phenotype.active_memory_dim_after_development),
                "sensor_radius_before_dev": float(phenotype.active_sensor_radius_before_development),
                "sensor_radius_after_dev": float(phenotype.active_sensor_radius_after_development),
                "sensor_channel_count_before_dev": float(phenotype.active_sensor_channel_count_before_development),
                "sensor_channel_count_after_dev": float(phenotype.active_sensor_channel_count_after_development),
                "prediction_lr_before_dev": float(phenotype.effective_prediction_lr_before_development),
                "prediction_lr_after_dev": float(phenotype.effective_prediction_lr_after_development),
                "introspection_enabled": bool(phenotype.introspection_enabled),
                "introspection_count": float(phenotype.active_introspection_channel_count),
                "introspection_potential": float(phenotype.potential_introspection_channel_count),
                "introspection_fraction": float(phenotype.introspection_capacity_fraction),
                "introspection_mask": int(phenotype.active_introspection_channel_mask),
                "introspection_ids": tuple(phenotype.active_introspection_channel_ids),
                "introspection_cost_hyp": float(ledger.introspection_channel_cost_hypothetical),
                "introspection_cost_charged": float(ledger.introspection_channel_cost_charged),
                "introspection_cost_fraction": float(ledger.introspection_cost_fraction_of_cog_cost),
                "introspection_ablate": bool(phenotype.introspection_ablation_active),
                "introspection_force_no": bool(phenotype.force_no_introspection_channels_active),
                "introspection_force_minimal": bool(phenotype.force_minimal_introspection_channels_active),
                "introspection_force_all": bool(phenotype.force_all_introspection_channels_active),
                "introspection_force_default": bool(phenotype.force_default_introspection_channels_active),
                "introspection_mask_neutral": bool(phenotype.mask_introspection_values_to_neutral_active),
                "introspection_cost_charging": bool(ledger.introspection_cost_charging_enabled),
                "introspection_policy_input": bool(phenotype.introspection_policy_input_enabled),
                "self_regulation_enabled": bool(phenotype.self_regulation_enabled),
                "self_regulation_surface": bool(phenotype.self_regulation_surface_enabled),
                "self_regulation_policy_output": bool(phenotype.self_regulation_policy_output_enabled),
                "self_regulation_count": float(phenotype.active_self_regulation_control_count),
                "self_regulation_potential": float(phenotype.potential_self_regulation_control_count),
                "self_regulation_fraction": float(phenotype.self_regulation_capacity_fraction),
                "self_regulation_mask": int(phenotype.active_self_regulation_control_mask),
                "self_regulation_ids": tuple(phenotype.active_self_regulation_control_ids),
                "self_regulation_cost_hyp": float(ledger.self_regulation_control_cost_hypothetical),
                "self_regulation_cost_charged": float(ledger.self_regulation_control_cost_charged),
                "self_regulation_cost_fraction": float(ledger.self_regulation_cost_fraction_of_cog_cost),
                "self_regulation_ablate": bool(phenotype.self_regulation_ablation_active),
                "self_regulation_force_no": bool(phenotype.force_no_self_regulation_active),
                "self_regulation_force_all": bool(phenotype.force_all_self_regulation_active),
                "self_regulation_force_default": bool(phenotype.force_default_self_regulation_active),
                "self_regulation_mask_neutral": bool(phenotype.mask_self_regulation_outputs_to_neutral_active),
                "self_regulation_freeze": bool(phenotype.freeze_self_regulation_state_active),
                "self_regulation_reset_each_tick": bool(phenotype.reset_self_regulation_each_tick_active),
                "self_regulation_decay_disabled": bool(phenotype.disable_self_regulation_decay_active),
                "self_regulation_cost_charging": bool(ledger.self_regulation_cost_charging_enabled),
                "self_modification_active": bool(phenotype.self_modification_active),
                "self_modification_enabled": bool(phenotype.self_modification_enabled),
                "self_modification_policy_output": bool(phenotype.self_modification_policy_output_enabled),
                "self_modification_count": float(phenotype.active_self_modification_target_count),
                "self_modification_potential": float(phenotype.potential_self_modification_target_count),
                "self_modification_fraction": float(phenotype.self_modification_capacity_fraction),
                "self_modification_mask": int(phenotype.active_self_modification_target_mask),
                "self_modification_ids": tuple(phenotype.active_self_modification_target_ids),
                "self_modification_cost_hyp": float(ledger.self_modification_command_cost_hypothetical),
                "self_modification_cost_charged": float(ledger.self_modification_command_cost_charged),
                "self_modification_cost_fraction": float(ledger.self_modification_cost_fraction_of_cog_cost),
                "self_modification_ablate": bool(phenotype.self_modification_ablation_active),
                "self_modification_force_no": bool(phenotype.force_no_self_modification_active),
                "self_modification_force_all": bool(phenotype.force_all_self_modification_active),
                "self_modification_force_default": bool(phenotype.force_default_self_modification_active),
                "self_modification_mask_neutral": bool(phenotype.mask_self_modification_commands_to_neutral_active),
                "self_modification_freeze": bool(phenotype.freeze_self_modification_state_active),
                "self_modification_reset_each_tick": bool(phenotype.reset_self_modification_each_tick_active),
                "self_modification_reset_on_reproduction": bool(phenotype.reset_self_modification_on_reproduction_active),
                "self_modification_decay_disabled": bool(phenotype.disable_self_modification_decay_active),
                "self_modification_cost_charging": bool(ledger.self_modification_cost_charging_enabled),
                "self_mod_delta_abs": float(self_mod_summary["delta_abs_mean"]),
                "self_mod_delta_abs_p90": float(self_mod_summary["delta_abs_p90"]),
                "self_mod_offset_abs": float(self_mod_summary["offset_abs_mean"]),
                "self_mod_offset_abs_p90": float(self_mod_summary["offset_abs_p90"]),
                "self_mod_clipped_fraction": float(self_mod_summary["clipped_fraction"]),
                "self_mod_saturation_fraction": float(self_mod_summary["saturation_fraction"]),
                "self_mod_event_count": float(self_mod_summary["event_count"]),
                "self_mod_neutral_fraction": float(self_mod_summary["neutral_fraction"]),
                "self_mod_reset_count": float(self_mod_summary["reset_count"]),
                "self_mod_decay_count": float(self_mod_summary["decay_count"]),
                "self_mod_lifetime_persistence": float(self_mod_summary["lifetime_persistence"]),
                "self_mod_reversal_count": float(self_mod_summary["reversal_count"]),
                "modified_exploration_bias": float(self_mod_summary["modified_exploration_bias"]),
                "modified_action_temperature": float(self_mod_summary["modified_action_temperature"]),
                "modified_drive_gain": float(self_mod_summary["modified_drive_gain"]),
                "modified_reproduction_threshold": float(self_mod_summary["modified_reproduction_threshold"]),
                "modified_prediction_lr": float(self_mod_summary["modified_prediction_lr"]),
                "genome_editing_active": bool(phenotype.genome_editing_active),
                "directed_inheritance_enabled": bool(phenotype.directed_inheritance_enabled),
                "directed_inheritance_surface": bool(phenotype.directed_inheritance_surface_enabled),
                "directed_inheritance_policy_output": bool(phenotype.directed_inheritance_policy_output_enabled),
                "directed_inheritance_count": float(phenotype.active_directed_inheritance_channel_count),
                "directed_inheritance_potential": float(phenotype.potential_directed_inheritance_channel_count),
                "directed_inheritance_fraction": float(phenotype.directed_inheritance_capacity_fraction),
                "directed_inheritance_mask": int(phenotype.active_directed_inheritance_channel_mask),
                "directed_inheritance_ids": tuple(phenotype.active_directed_inheritance_channel_ids),
                "directed_inheritance_cost_hyp": float(ledger.directed_inheritance_command_cost_hypothetical),
                "directed_inheritance_cost_charged": float(ledger.directed_inheritance_command_cost_charged),
                "directed_inheritance_cost_fraction": float(ledger.directed_inheritance_cost_fraction_of_cog_cost),
                "directed_inheritance_ablate": bool(phenotype.directed_inheritance_ablation_active),
                "directed_inheritance_force_no": bool(phenotype.force_no_directed_inheritance_active),
                "directed_inheritance_force_all": bool(phenotype.force_all_directed_inheritance_active),
                "directed_inheritance_force_default": bool(phenotype.force_default_directed_inheritance_active),
                "directed_inheritance_mask_neutral": bool(phenotype.mask_directed_inheritance_commands_to_neutral_active),
                "directed_inheritance_freeze": bool(phenotype.freeze_directed_inheritance_payload_active),
                "directed_inheritance_reset_birth": bool(phenotype.reset_directed_inheritance_payload_on_birth_active),
                "directed_inheritance_reset_maturity": bool(phenotype.reset_directed_inheritance_payload_on_maturity_active),
                "directed_inheritance_cost_charging": bool(ledger.directed_inheritance_cost_charging_enabled),
                "mutation_rate_control_active": bool(phenotype.mutation_rate_control_active),
                "recombination_control_active": bool(phenotype.recombination_control_active),
                "inherited_self_modification_state_active": bool(phenotype.inherited_self_modification_state_active),
                "directed_delta_abs": float(directed_summary["delta_abs_mean"]),
                "directed_delta_abs_p90": float(directed_summary["delta_abs_p90"]),
                "directed_payload_abs": float(directed_summary["payload_abs_mean"]),
                "directed_payload_abs_p90": float(directed_summary["payload_abs_p90"]),
                "directed_clipped_fraction": float(directed_summary["clipped_fraction"]),
                "directed_saturation_fraction": float(directed_summary["saturation_fraction"]),
                "directed_event_count": float(directed_summary["event_count"]),
                "directed_neutral_fraction": float(directed_summary["neutral_command_fraction"]),
                "directed_reset_count": float(directed_summary["payload_reset_count"]),
                "directed_payload_applied": float(directed_summary["payload_applied_fraction"]),
                "directed_payload_frozen": float(directed_summary["payload_frozen_fraction"]),
                "offspring_developmental_expression_bias": float(directed_summary["offspring_developmental_expression_bias"]),
                "offspring_juvenile_cognition_scale_bias": float(directed_summary["offspring_juvenile_cognition_scale_bias"]),
                "offspring_memory_expression_bias": float(directed_summary["offspring_memory_expression_bias"]),
                "offspring_prediction_lr_bias": float(directed_summary["offspring_prediction_lr_bias"]),
                "offspring_exploration_bias_marker": float(directed_summary["offspring_exploration_bias_marker"]),
                "offspring_reproduction_threshold_bias": float(directed_summary["offspring_reproduction_threshold_bias"]),
                "self_reg_delta_abs": float(self_reg_summary["delta_abs_mean"]),
                "self_reg_delta_abs_p90": float(self_reg_summary["delta_abs_p90"]),
                "self_reg_offset_abs": float(self_reg_summary["offset_abs_mean"]),
                "self_reg_offset_abs_p90": float(self_reg_summary["offset_abs_p90"]),
                "self_reg_clipped_fraction": float(self_reg_summary["clipped_fraction"]),
                "self_reg_saturation_fraction": float(self_reg_summary["saturation_fraction"]),
                "self_reg_application_count": float(self_reg_summary["application_count"]),
                "self_reg_neutral_fraction": float(self_reg_summary["neutral_fraction"]),
                "self_reg_reset_count": float(self_reg_summary["reset_count"]),
                "self_reg_decay_count": float(self_reg_summary["decay_count"]),
                "effective_exploration_pressure": float(self_reg_summary["effective_exploration_pressure"]),
                "effective_action_temperature": float(self_reg_summary["effective_action_temperature"]),
                "effective_drive_gain": float(self_reg_summary["effective_drive_gain"]),
                "effective_reproduction_threshold": float(self_reg_summary["effective_reproduction_threshold"]),
                "effective_prediction_lr_regulated": float(self_reg_summary["effective_prediction_lr"]),
                "internal_energy": float(introspection_channel_value("energy", agent=a, env_cfg=env_cfg, sim_cfg=sim_cfg, phenotype=phenotype, ledger=ledger)),
                "internal_health": float(introspection_channel_value("health", agent=a, env_cfg=env_cfg, sim_cfg=sim_cfg, phenotype=phenotype, ledger=ledger)),
                "internal_drive_conflict": float(introspection_channel_value("drive_conflict_proxy", agent=a, env_cfg=env_cfg, sim_cfg=sim_cfg, phenotype=phenotype, ledger=ledger)),
                "internal_prediction_error": float(introspection_channel_value("prediction_error", agent=a, env_cfg=env_cfg, sim_cfg=sim_cfg, phenotype=phenotype, ledger=ledger)),
                "internal_memory_saturation": float(introspection_channel_value("memory_saturation_proxy", agent=a, env_cfg=env_cfg, sim_cfg=sim_cfg, phenotype=phenotype, ledger=ledger)),
                "internal_cog_cost_burden": float(introspection_channel_value("cognitive_cost_fraction", agent=a, env_cfg=env_cfg, sim_cfg=sim_cfg, phenotype=phenotype, ledger=ledger)),
                "internal_developmental_expression": float(introspection_channel_value("developmental_expression_factor", agent=a, env_cfg=env_cfg, sim_cfg=sim_cfg, phenotype=phenotype, ledger=ledger)),
                "offspring": float(getattr(st, "offspring_count", 0.0) if st is not None else 0.0),
                "energy": float(getattr(a, "energy", 0.0)),
                "alive": bool(getattr(a, "alive", True)),
                "lineage": int(_lineage_id(a)),
                "drive_conflict": float(_drive_conflict(a)),
                "strategy": (
                    int(phenotype.active_memory_capacity),
                    int(phenotype.recurrent_state_enabled),
                    round(float(phenotype.effective_prediction_learning_rate), 2),
                    round(float(phenotype.drive_logit_gain), 1),
                    int(phenotype.active_sensor_channel_mask),
                    int(phenotype.active_sensor_radius),
                ),
                "memory_strategy": (
                    int(phenotype.active_memory_capacity),
                    int(phenotype.recurrent_state_enabled),
                    int(phenotype.active_sensor_channel_mask),
                    int(phenotype.active_sensor_radius),
                ),
                "memory_perception_strategy": (
                    int(phenotype.active_memory_capacity),
                    int(phenotype.active_sensor_channel_mask),
                    int(phenotype.active_sensor_radius),
                ),
                "prediction_strategy": (
                    int(phenotype.prediction_enabled),
                    round(float(phenotype.effective_prediction_learning_rate), 2),
                    int(phenotype.active_memory_capacity),
                    int(phenotype.active_sensor_channel_mask),
                    int(phenotype.active_sensor_radius),
                ),
                "developmental_strategy": (
                    int(phenotype.maturity_age),
                    str(phenotype.developmental_stage),
                    round(float(phenotype.juvenile_cognition_scale), 2),
                    int(phenotype.active_memory_capacity),
                    int(phenotype.active_sensor_radius),
                    round(float(phenotype.effective_prediction_learning_rate), 2),
                ),
                "introspection_strategy": (
                    int(phenotype.active_introspection_channel_mask),
                    int(phenotype.active_memory_capacity),
                    int(phenotype.active_sensor_radius),
                    round(float(phenotype.effective_prediction_learning_rate), 2),
                    str(phenotype.developmental_stage),
                ),
                "self_regulation_strategy": (
                    int(phenotype.active_self_regulation_control_mask),
                    int(phenotype.active_introspection_channel_mask),
                    int(phenotype.active_memory_capacity),
                    int(phenotype.active_sensor_radius),
                    round(float(phenotype.effective_prediction_learning_rate), 2),
                    str(phenotype.developmental_stage),
                ),
                "self_modification_strategy": (
                    int(phenotype.active_self_modification_target_mask),
                    int(phenotype.active_self_regulation_control_mask),
                    int(phenotype.active_introspection_channel_mask),
                    int(phenotype.active_memory_capacity),
                    int(phenotype.active_sensor_radius),
                    round(float(phenotype.effective_prediction_learning_rate), 2),
                    str(phenotype.developmental_stage),
                ),
                "directed_inheritance_strategy": (
                    int(phenotype.active_directed_inheritance_channel_mask),
                    int(phenotype.active_self_modification_target_mask),
                    int(phenotype.active_self_regulation_control_mask),
                    int(phenotype.active_introspection_channel_mask),
                    int(phenotype.active_memory_capacity),
                    int(phenotype.active_sensor_radius),
                    round(float(phenotype.effective_prediction_learning_rate), 2),
                    str(phenotype.developmental_stage),
                ),
            }
        )
    return recs


def _vals(recs: list[Mapping[str, Any]], key: str) -> list[float]:
    return [float(r.get(key, 0.0)) for r in recs]


def _escap8_sustainable_growth_reporting_context(
    agents: list[Any], *, sim_cfg: Any
) -> dict[str, Any]:
    """Aggregate ESCAP-8 reporting without widening the frozen Flat v1 rows."""

    if not bool(getattr(sim_cfg, "structural_sustainable_growth_enable", False)):
        return {}
    live_rows: list[Mapping[str, Any]] = []
    genomes: list[Any] = []
    for agent in agents:
        genome = getattr(agent, "genome", None)
        if genome is None:
            continue
        genomes.append(genome)
        row = getattr(agent, "structural_live_module_telemetry", {}) or {}
        live_rows.append(row if isinstance(row, Mapping) else {})

    def live_max(name: str) -> float:
        return float(max((float(row.get(name, 0.0) or 0.0) for row in live_rows), default=0.0))

    def genome_sum(name: str) -> float:
        return float(sum(float(getattr(genome, name, 0.0) or 0.0) for genome in genomes))

    def genome_max(name: str) -> float:
        return float(max((float(getattr(genome, name, 0.0) or 0.0) for genome in genomes), default=0.0))

    return {
        "escap8_growth_attempt_count": genome_sum("escap8_growth_attempt_count"),
        "blocked_duplication_attempts": genome_sum(
            "escap8_blocked_duplication_attempt_count"
        ),
        "blocked_prune_attempts": genome_sum("escap8_blocked_prune_attempt_count"),
        "cap_contact_count": genome_sum("escap8_cap_contact_count"),
        "cap_pressure_streak": genome_max("escap8_cap_pressure_streak"),
        "executing_module_count_max": live_max("executing_modules"),
        "dormant_module_count_max": live_max("dormant_modules"),
        "nonzero_output_module_count_max": live_max("nonzero_output_modules"),
        "neutral_duplicate_count_max": live_max("neutral_duplicate_modules"),
        "redundant_module_count_max": live_max("redundant_modules"),
        "prunable_module_count_max": live_max("prunable_modules"),
        "structural_cost_total_max": live_max("structural_cost_total"),
        "specialization_metrics_record_count": int(
            sum(
                1
                for genome in genomes
                if str(
                    getattr(genome, "escap8_specialization_metrics_json", "{}")
                    or "{}"
                )
                != "{}"
            )
        ),
    }


def flat_population_reporting_records_from_sources(
    sources: Iterable[Any] | None,
    *,
    selected_parent_sources: Iterable[Any] | None = None,
    env_cfg: Any = None,
    sim_cfg: Any = None,
) -> FlatPopulationReportingRecordsV1:
    """Adapt route-specific read-only sources to the canonical Flat contract."""

    from simulation.env_hotpath_expression_diagnostics import (
        reset_current_diagnostics,
        set_current_diagnostics,
    )

    token = set_current_diagnostics(None)
    try:
        return validate_flat_population_reporting_records(
            _agent_records(_safe_agents(sources), env_cfg=env_cfg, sim_cfg=sim_cfg),
            _agent_records(
                _safe_agents(selected_parent_sources),
                env_cfg=env_cfg,
                sim_cfg=sim_cfg,
            ),
        )
    finally:
        reset_current_diagnostics(token)


def compute_cognitive_status_metrics(
    agents: Iterable[Any] | None,
    *,
    selected_parent_agents: Iterable[Any] | None = None,
    env_cfg: Any = None,
    sim_cfg: Any = None,
    memory_runtime_telemetry: Mapping[str, Any] | None = None,
    generation: int | None = None,
) -> "OrderedDict[str, float]":
    alive_agents = _safe_agents(agents)
    parent_agents = _safe_agents(selected_parent_agents)
    structural_summary_json = None
    escap8_structural_context = _escap8_sustainable_growth_reporting_context(
        alive_agents, sim_cfg=sim_cfg
    )
    if structural_live_semantics_enabled(sim_cfg):
        structural_summary_json = stable_json(
            _structural_live_semantics_summary(alive_agents, sim_cfg=sim_cfg)
        )
    from simulation.output_cost_timing import reporting_projection_phase, reporting_reducer_phase

    with reporting_projection_phase():
        reporting_records = flat_population_reporting_records_from_sources(
            alive_agents,
            selected_parent_sources=parent_agents,
            env_cfg=env_cfg,
            sim_cfg=sim_cfg,
        )
    with reporting_reducer_phase():
        return compute_cognitive_status_metrics_from_records(
            reporting_records,
            env_cfg=env_cfg,
            sim_cfg=sim_cfg,
            memory_runtime_telemetry=memory_runtime_telemetry,
            generation=generation,
            structural_live_semantics_summary_json=structural_summary_json,
            escap8_structural_context=escap8_structural_context,
        )


def compute_cognitive_status_metrics_from_records(
    records: Iterable[Mapping[str, Any]] | FlatPopulationReportingRecordsV1 | None,
    *,
    selected_parent_records: Iterable[Mapping[str, Any]] | None = None,
    env_cfg: Any = None,
    sim_cfg: Any = None,
    memory_runtime_telemetry: Mapping[str, Any] | None = None,
    generation: int | None = None,
    structural_live_semantics_summary_json: str | None = None,
    escap8_structural_context: Mapping[str, Any] | None = None,
) -> "OrderedDict[str, float]":
    """Reduce canonical population reporting records without live Agent objects."""

    out = cognitive_status_metric_defaults()
    if sim_cfg is not None and not bool(getattr(sim_cfg, "enable_cognitive_status_metrics", True)):
        return out

    if isinstance(records, FlatPopulationReportingRecordsV1):
        if tuple(selected_parent_records or ()):
            raise ValueError(
                "selected_parent_records cannot replace rows on canonical reporting records"
            )
        reporting_records = validate_flat_population_reporting_records(records)
    else:
        reporting_records = validate_flat_population_reporting_records(
            records or (),
            selected_parent_records or (),
        )
    recs = [dict(record) for record in reporting_records.rows]
    parent_recs = [dict(record) for record in reporting_records.selected_parent_rows]

    if not recs:
        if sim_cfg is not None and bool(getattr(sim_cfg, "trait_capacity_saturation_reporting_enable", False)):
            summary_json, lineage_json, structural_json = empty_trait_capacity_payload("no_alive_records")
            out["trait_capacity_saturation_summary_json"] = summary_json
            out["trait_capacity_saturation_lineage_json"] = lineage_json
            out["trait_capacity_saturation_structural_payload_json"] = structural_json
        if structural_live_semantics_enabled(sim_cfg):
            out["structural_live_semantics_summary_json"] = (
                str(structural_live_semantics_summary_json)
                if structural_live_semantics_summary_json is not None
                else stable_json(_structural_live_semantics_summary([], sim_cfg=sim_cfg))
            )
        return out

    cap = _vals(recs, "capacity")
    genetic_mem = _vals(recs, "genetic_memory")
    mem = _vals(recs, "memory")
    eff_mem = _vals(recs, "effective_memory")
    costs = _vals(recs, "cost")
    cost_fracs = _vals(recs, "cost_frac")
    pred_lr = _vals(recs, "pred_lr")
    effective_pred_lr = _vals(recs, "effective_pred_lr")
    pred_errs = _vals(recs, "pred_err")
    prediction_capacity = _vals(recs, "prediction_capacity")
    prediction_hyp_costs = _vals(recs, "prediction_cost_hyp")
    prediction_charged_costs = _vals(recs, "prediction_cost_charged")
    drive_gain = _vals(recs, "drive_gain")
    drive_noise = _vals(recs, "drive_noise")
    drive_inertia = _vals(recs, "drive_inertia")
    drive_conflict = _vals(recs, "drive_conflict")
    sensor_counts = _vals(recs, "sensor_channel_count")
    sensor_radii = _vals(recs, "sensor_radius")
    observed_cells = _vals(recs, "observed_cells")
    cell_channel_capacity = _vals(recs, "cell_channel_capacity")
    perception_costs = _vals(recs, "perception_cost")
    perception_hyp_costs = _vals(recs, "perception_cost_hyp")
    perception_charged_costs = _vals(recs, "perception_cost_charged")
    radius_hyp_costs = _vals(recs, "sensor_radius_cost_hyp")
    radius_charged_costs = _vals(recs, "sensor_radius_cost_charged")
    spatial_costs = _vals(recs, "spatial_perception_cost")

    out["cog_capacity_mean"] = mean(cap)
    out["cog_capacity_p50"] = percentile(cap, 50)
    out["cog_capacity_p90"] = percentile(cap, 90)
    out["cog_capacity_max"] = max(cap)
    out["cog_capacity_min"] = min(cap)
    out["cog_capacity_std"] = std(cap)
    out["cog_capacity_alive_mean"] = out["cog_capacity_mean"]
    out["cog_capacity_parent_mean"] = mean(_vals(parent_recs, "capacity")) if parent_recs else 0.0
    out["cog_capacity_survivor_mean"] = out["cog_capacity_alive_mean"]
    out["cog_capacity_alive_end_mean"] = out["cog_capacity_alive_mean"]
    out["cog_capacity_memory_component_mean"] = mean(_vals(recs, "memory_component"))
    out["cog_capacity_prediction_component_mean"] = mean(_vals(recs, "prediction_component"))
    out["cog_capacity_drive_component_mean"] = mean(_vals(recs, "drive_component"))
    out["cog_capacity_perception_component_mean"] = mean(_vals(recs, "perception_component"))

    n = float(len(recs))
    out["memory_enabled_fraction"] = sum(1 for r in recs if bool(r["memory_enabled"])) / n
    out["memory_dim_mean"] = mean(mem)
    out["memory_dim_p50"] = percentile(mem, 50)
    out["memory_dim_p90"] = percentile(mem, 90)
    out["memory_dim_max"] = max(mem)
    out["memory_dim_min"] = min(mem)
    out["memory_dim_std"] = std(mem)
    out["memory_capacity_fraction_mean"] = mean(_vals(recs, "memory_capacity_fraction"))
    out["memory_capacity_fraction_p90"] = percentile(_vals(recs, "memory_capacity_fraction"), 90)
    for b in MEMORY_LADDER:
        out[f"memory_bucket_{b}_fraction"] = float(sum(1 for x in mem if int(x) == int(b))) / n
    out["effective_memory_mean"] = mean(eff_mem)
    out["effective_memory_p90"] = percentile(eff_mem, 90)
    out["memory_cost_mean"] = mean(_vals(recs, "memory_cost"))
    out["memory_cost_total"] = sum(_vals(recs, "memory_cost"))
    out["mem_evo_population_memory_dim_histogram"] = _json_dumps(_memory_histogram_payload(genetic_mem))
    out["mem_evo_descendant_memory_dim_histogram"] = out["mem_evo_population_memory_dim_histogram"]
    out["mem_evo_effective_memory_dim_histogram"] = _json_dumps(_memory_histogram_payload(eff_mem))
    out["mem_evo_memory_cost_mean"] = out["memory_cost_mean"]
    out["mem_evo_memory_cost_total"] = out["memory_cost_total"]
    out["memory_cost_hypothetical_mean"] = mean(_vals(recs, "memory_cost_hyp"))
    out["memory_cost_hypothetical_total"] = sum(_vals(recs, "memory_cost_hyp"))
    out["memory_cost_charged_mean"] = mean(_vals(recs, "memory_cost_charged"))
    out["memory_cost_charged_total"] = sum(_vals(recs, "memory_cost_charged"))
    out["memory_cost_fraction_of_cog_cost_mean"] = mean(_vals(recs, "memory_cost_fraction"))
    out["memory_cost_zero_fraction"] = sum(1 for r in recs if float(r["memory_cost_hyp"]) <= 0.0) / n
    mem_high_threshold = percentile(mem, 90)
    out["high_memory_low_cost_fraction"] = (
        sum(1 for r in recs if float(r["memory"]) >= mem_high_threshold and float(r["memory_cost_hyp"]) <= 0.0) / n
    )
    out["memory_ablation_active_flag"] = 1.0 if any(bool(r["memory_ablate"]) for r in recs) else 0.0
    out["memory_force_disabled_active_flag"] = 1.0 if any(bool(r["memory_force_disabled"]) for r in recs) else 0.0
    out["memory_force_min_active_flag"] = 1.0 if any(bool(r["memory_force_min"]) for r in recs) else 0.0
    out["memory_force_max_active_flag"] = 1.0 if any(bool(r["memory_force_max"]) for r in recs) else 0.0
    out["memory_force_default_active_flag"] = 1.0 if any(bool(r["memory_force_default"]) for r in recs) else 0.0
    out["memory_reset_each_tick_active_flag"] = 1.0 if any(bool(r["memory_reset_each_tick"]) for r in recs) else 0.0
    mem_counts: dict[int, int] = {}
    for x in mem:
        mem_counts[int(x)] = mem_counts.get(int(x), 0) + 1
    out["memory_dim_entropy"] = _entropy([int(x) for x in mem])
    out["memory_dim_diversity"] = float(len(mem_counts))
    out["dominant_memory_dim_fraction"] = float(max(mem_counts.values())) / n if mem_counts else 0.0
    out["memory_strategy_count_approx"] = float(len({r["memory_strategy"] for r in recs}))
    out["memory_perception_strategy_entropy"] = _entropy([r["memory_perception_strategy"] for r in recs])
    out["memory_perception_strategy_count_approx"] = float(len({r["memory_perception_strategy"] for r in recs}))
    if memory_runtime_telemetry:
        for key in (
            "memory_state_abs_mean",
            "memory_state_abs_p90",
            "memory_state_delta_mean",
            "memory_state_delta_p90",
            "memory_active_slice_abs_mean",
            "memory_inactive_slice_abs_mean",
            "memory_mask_leakage_mean",
            "memory_nonzero_fraction_mean",
            "memory_effective_utilization_mean",
            "memory_saturation_proxy_mean",
            "memory_reset_count_total",
            "memory_use_measurement_available_flag",
        ):
            try:
                out[key] = float(memory_runtime_telemetry.get(key, 0.0))
            except Exception:
                out[key] = 0.0
    out["memory_pressure_scenario_flag"] = 1.0 if bool(getattr(sim_cfg, "memory_pressure_scenario", False)) else 0.0
    pressure_type = str(getattr(sim_cfg, "memory_pressure_type", "") or "")
    out["memory_pressure_food_regrowth_return_flag"] = 1.0 if pressure_type == "food_regrowth_return" else 0.0

    out["gru_enabled_fraction"] = sum(1 for r in recs if bool(r["recurrent_state"])) / n
    out["recurrent_policy_fraction"] = out["gru_enabled_fraction"]
    out["mem_evo_use_gru_share"] = out["gru_enabled_fraction"]
    hidden = _vals(recs, "hidden")
    out["policy_hidden_size_mean"] = mean(hidden)
    out["policy_hidden_size_p90"] = percentile(hidden, 90)
    out["policy_hidden_size_max"] = max(hidden)
    out["policy_memory_active_fraction"] = sum(1 for r in recs if bool(r["policy_memory"])) / n
    out["policy_memory_cost_mean"] = mean(_vals(recs, "policy_memory_cost"))

    out["prediction_enabled_fraction"] = sum(1 for r in recs if bool(r["pred_enabled"])) / n
    out["prediction_lr_mean"] = mean(pred_lr)
    out["prediction_lr_p50"] = percentile(pred_lr, 50)
    out["prediction_lr_p90"] = percentile(pred_lr, 90)
    out["prediction_lr_max"] = max(pred_lr) if pred_lr else 0.0
    out["prediction_lr_min"] = min(pred_lr) if pred_lr else 0.0
    out["prediction_lr_std"] = std(pred_lr)
    out["effective_prediction_lr_mean"] = mean(effective_pred_lr)
    out["effective_prediction_lr_p90"] = percentile(effective_pred_lr, 90)
    out["prediction_error_mean"] = mean(pred_errs)
    out["prediction_error_p50"] = percentile(pred_errs, 50)
    out["prediction_error_p90"] = percentile(pred_errs, 90)
    out["prediction_error_max"] = max(pred_errs) if pred_errs else 0.0
    out["prediction_error_std"] = std(pred_errs)
    out["prediction_error_trend_mean"] = 0.0
    out["prediction_error_under_ablation_mean"] = mean([float(r["pred_err"]) for r in recs if bool(r["prediction_ablate"])])
    out["prediction_update_cost_mean"] = mean(prediction_charged_costs)
    out["prediction_cost_hypothetical_mean"] = mean(prediction_hyp_costs)
    out["prediction_cost_hypothetical_total"] = sum(prediction_hyp_costs)
    out["prediction_cost_charged_mean"] = mean(prediction_charged_costs)
    out["prediction_cost_charged_total"] = sum(prediction_charged_costs)
    out["prediction_cost_fraction_of_cog_cost_mean"] = mean(_vals(recs, "prediction_cost_fraction"))
    out["prediction_cost_zero_fraction"] = sum(1 for r in recs if float(r["prediction_cost_hyp"]) <= 0.0) / n
    pred_high_threshold = percentile(effective_pred_lr, 90)
    out["high_prediction_low_cost_fraction"] = (
        sum(1 for r in recs if float(r["effective_pred_lr"]) >= pred_high_threshold and float(r["prediction_cost_hyp"]) <= 0.0) / n
    )
    out["prediction_ablation_active_flag"] = 1.0 if any(bool(r["prediction_ablate"]) for r in recs) else 0.0
    out["prediction_force_disabled_active_flag"] = 1.0 if any(bool(r["prediction_force_disabled"]) for r in recs) else 0.0
    out["prediction_force_frozen_active_flag"] = 1.0 if any(bool(r["prediction_force_frozen"]) for r in recs) else 0.0
    out["prediction_force_zero_lr_active_flag"] = 1.0 if any(bool(r["prediction_force_zero_lr"]) for r in recs) else 0.0
    out["prediction_force_default_active_flag"] = 1.0 if any(bool(r["prediction_force_default"]) for r in recs) else 0.0
    out["prediction_reset_each_tick_active_flag"] = 1.0 if any(bool(r["prediction_reset_each_tick"]) for r in recs) else 0.0
    out["prediction_direct_selection_pressure_disabled_flag"] = 1.0 if bool(getattr(sim_cfg, "disable_prediction_error_selection_pressure", False)) else 0.0
    out["prediction_state_abs_mean"] = mean(_vals(recs, "prediction_state_abs"))
    out["prediction_state_abs_p90"] = percentile(_vals(recs, "prediction_state_abs_p90"), 90)
    out["prediction_state_delta_mean"] = mean(_vals(recs, "prediction_state_delta"))
    out["prediction_state_delta_p90"] = percentile(_vals(recs, "prediction_state_delta_p90"), 90)
    out["prediction_error_delta_mean"] = out["prediction_state_delta_mean"]
    out["prediction_update_count_total"] = sum(_vals(recs, "prediction_update_count"))
    out["prediction_nonzero_fraction_mean"] = mean(_vals(recs, "prediction_nonzero_fraction"))
    out["prediction_effective_utilization_mean"] = mean(_vals(recs, "prediction_effective_utilization"))
    out["prediction_saturation_proxy_mean"] = mean(_vals(recs, "prediction_saturation_proxy"))
    out["prediction_action_differentiation_mean"] = mean(_vals(recs, "prediction_action_differentiation"))
    out["prediction_choice_coupling_proxy_mean"] = 0.0
    out["prediction_capacity_proxy_mean"] = mean(effective_pred_lr)
    out["prediction_capacity_proxy_max"] = max(effective_pred_lr) if effective_pred_lr else 0.0
    out["prediction_capacity_mean"] = mean(prediction_capacity)
    out["prediction_capacity_p90"] = percentile(prediction_capacity, 90)
    out["prediction_capacity_max"] = max(prediction_capacity) if prediction_capacity else 0.0
    pred_profiles = [r["prediction_strategy"] for r in recs]
    out["prediction_lr_entropy"] = _entropy([round(float(x), 2) for x in effective_pred_lr])
    out["prediction_profile_diversity"] = float(len(set(pred_profiles)))
    if pred_profiles:
        counts: dict[Any, int] = {}
        for p in pred_profiles:
            counts[p] = counts.get(p, 0) + 1
        out["dominant_prediction_profile_fraction"] = float(max(counts.values())) / n
    out["prediction_strategy_count_approx"] = float(len(set(pred_profiles)))
    out["prediction_growth_without_pressure_flag"] = 0.0
    out["direct_prediction_error_selection_pressure_present_flag"] = 1.0 if any(bool(r["direct_prediction_pressure"]) for r in recs) else 0.0
    out["prediction_pressure_scenario_flag"] = 1.0 if bool(getattr(sim_cfg, "prediction_pressure_scenario", False)) else 0.0
    out["prediction_pressure_delayed_consequence_flag"] = (
        1.0 if str(getattr(sim_cfg, "prediction_pressure_type", "") or "") == "delayed_consequence" else 0.0
    )

    out["drive_gain_mean"] = mean(drive_gain)
    out["drive_gain_p90"] = percentile(drive_gain, 90)
    out["drive_noise_mean"] = mean(drive_noise)
    out["drive_inertia_mean"] = mean(drive_inertia)
    out["drive_conflict_mean"] = mean(drive_conflict)
    out["drive_conflict_p90"] = percentile(drive_conflict, 90)
    out["hunger_sensitivity_mean"] = mean(_vals(recs, "hunger"))
    out["fear_sensitivity_mean"] = mean(_vals(recs, "fear"))
    out["pain_sensitivity_mean"] = mean(_vals(recs, "pain"))
    out["fatigue_sensitivity_mean"] = mean(_vals(recs, "fatigue"))
    out["drive_regulation_cost_mean"] = 0.0

    out["cog_cost_mean"] = mean(costs)
    out["cog_cost_p90"] = percentile(costs, 90)
    out["cog_cost_max"] = max(costs)
    out["cog_cost_total"] = sum(costs)
    out["cog_cost_fraction_of_energy_mean"] = mean(cost_fracs)
    out["cog_cost_fraction_of_energy_p90"] = percentile(cost_fracs, 90)
    out["cognition_under_starvation_fraction"] = sum(1 for r in recs if float(r["energy"]) <= float(r["cost"])) / n

    cap_p50 = percentile(cap, 50)
    high = [r for r in recs if float(r["capacity"]) >= cap_p50]
    low = [r for r in recs if float(r["capacity"]) < cap_p50]
    out["high_cognition_starvation_rate"] = (
        sum(1 for r in high if float(r["energy"]) <= float(r["cost"])) / float(len(high))
        if high
        else 0.0
    )
    out["low_cognition_starvation_rate"] = (
        sum(1 for r in low if float(r["energy"]) <= float(r["cost"])) / float(len(low))
        if low
        else 0.0
    )

    parent_cap = _vals(parent_recs, "capacity")
    parent_mem = _vals(parent_recs, "memory")
    parent_pred = _vals(parent_recs, "effective_pred_lr")
    parent_drive = _vals(parent_recs, "drive_gain")
    parent_sensor_counts = _vals(parent_recs, "sensor_channel_count")
    parent_sensor_radii = _vals(parent_recs, "sensor_radius")
    parent_cell_channel = _vals(parent_recs, "cell_channel_capacity")
    out["parent_cog_capacity_mean"] = mean(parent_cap) if parent_cap else 0.0
    out["alive_cog_capacity_mean"] = mean(cap)
    out["dead_cog_capacity_mean"] = 0.0
    reproduced = [r for r in recs if float(r["offspring"]) > 0.0]
    non_reproduced = [r for r in recs if float(r["offspring"]) <= 0.0]
    out["reproduced_cog_capacity_mean"] = mean(_vals(reproduced, "capacity")) if reproduced else 0.0
    out["non_reproduced_cog_capacity_mean"] = mean(_vals(non_reproduced, "capacity")) if non_reproduced else 0.0
    out["cognitive_selection_delta_parent_vs_alive"] = out["parent_cog_capacity_mean"] - out["alive_cog_capacity_mean"]
    out["cognitive_selection_delta_survivor_vs_dead"] = 0.0
    out["cognitive_selection_delta_survivor_vs_dead_available_flag"] = 0.0
    out["memory_selection_delta_parent_vs_alive"] = (mean(parent_mem) if parent_mem else 0.0) - mean(mem)
    parent_genetic_mem = _vals(parent_recs, "genetic_memory")
    out["mem_evo_selected_parent_memory_dim_histogram"] = _json_dumps(_memory_histogram_payload(parent_genetic_mem))
    out["mem_evo_memory_dim_selected_minus_population_mean"] = (
        (mean(parent_genetic_mem) if parent_genetic_mem else 0.0) - mean(genetic_mem)
    )
    out["parent_prediction_lr_mean"] = mean(parent_pred) if parent_pred else 0.0
    out["alive_prediction_lr_mean"] = mean(effective_pred_lr)
    out["dead_prediction_lr_mean"] = 0.0
    out["reproduced_prediction_lr_mean"] = mean(_vals(reproduced, "effective_pred_lr")) if reproduced else 0.0
    out["non_reproduced_prediction_lr_mean"] = mean(_vals(non_reproduced, "effective_pred_lr")) if non_reproduced else 0.0
    out["prediction_selection_delta_parent_vs_alive"] = out["parent_prediction_lr_mean"] - out["alive_prediction_lr_mean"]
    out["prediction_selection_delta_survivor_vs_dead"] = 0.0
    out["drive_regulation_selection_delta_parent_vs_alive"] = (mean(parent_drive) if parent_drive else 0.0) - mean(drive_gain)

    out["cognitive_trait_diversity"] = _diversity(cap) + _diversity(mem) + _diversity(effective_pred_lr) + _diversity(drive_gain)
    out["cognitive_capacity_entropy"] = _entropy([round(x, 1) for x in cap])
    out["memory_bucket_entropy"] = _entropy([int(x) for x in mem])
    out["policy_type_entropy"] = _entropy([int(bool(r["policy_memory"])) for r in recs])
    out["drive_profile_diversity"] = _diversity(drive_gain)
    out["prediction_lr_diversity"] = _diversity(effective_pred_lr)
    out["cognitive_strategy_count_approx"] = float(len({r["strategy"] for r in recs}))

    ages = _vals(recs, "age")
    juvenile = [r for r in recs if float(r["age"]) < 20.0]
    adult = [r for r in recs if 20.0 <= float(r["age"]) < 100.0]
    elder = [r for r in recs if float(r["age"]) >= 100.0]
    out["juvenile_cog_capacity_mean"] = mean(_vals(juvenile, "capacity")) if juvenile else 0.0
    out["adult_cog_capacity_mean"] = mean(_vals(adult, "capacity")) if adult else 0.0
    out["elder_cog_capacity_mean"] = mean(_vals(elder, "capacity")) if elder else 0.0
    out["cognition_by_age_slope"] = _slope(ages, cap)
    out["memory_by_age_slope"] = _slope(ages, mem)
    out["cognitive_cost_by_age_slope"] = _slope(ages, costs)
    out["perception_by_age_slope"] = _slope(ages, cell_channel_capacity)
    out["prediction_by_age_slope"] = _slope(ages, effective_pred_lr)
    out["developmental_expression_active_fraction"] = sum(1 for r in recs if bool(r["dev_active"])) / n
    dev_factors = _vals(recs, "dev_factor")
    out["developmental_expression_factor_mean"] = mean(dev_factors)
    out["developmental_expression_factor_p50"] = percentile(dev_factors, 50)
    out["developmental_expression_factor_p90"] = percentile(dev_factors, 90)
    juveniles = [r for r in recs if bool(r["is_juvenile"])]
    mature = [r for r in recs if bool(r["is_mature"])]
    out["juvenile_fraction"] = float(len(juveniles)) / n
    out["adult_fraction"] = float(len(mature)) / n
    out["mature_fraction"] = out["adult_fraction"]
    out["juvenile_memory_dim_mean"] = mean(_vals(juveniles, "memory")) if juveniles else 0.0
    out["adult_memory_dim_mean"] = mean(_vals(mature, "memory")) if mature else 0.0
    out["juvenile_sensor_radius_mean"] = mean(_vals(juveniles, "sensor_radius")) if juveniles else 0.0
    out["adult_sensor_radius_mean"] = mean(_vals(mature, "sensor_radius")) if mature else 0.0
    out["juvenile_sensor_channel_count_mean"] = mean(_vals(juveniles, "sensor_channel_count")) if juveniles else 0.0
    out["adult_sensor_channel_count_mean"] = mean(_vals(mature, "sensor_channel_count")) if mature else 0.0
    out["juvenile_prediction_lr_mean"] = mean(_vals(juveniles, "effective_pred_lr")) if juveniles else 0.0
    out["adult_prediction_lr_mean"] = mean(_vals(mature, "effective_pred_lr")) if mature else 0.0
    out["developmental_cost_hypothetical_mean"] = mean(_vals(recs, "dev_cost_hyp"))
    out["developmental_cost_charged_mean"] = mean(_vals(recs, "dev_cost_charged"))
    out["maturation_cost_hypothetical_mean"] = mean(_vals(recs, "maturation_cost_hyp"))
    out["maturation_cost_charged_mean"] = mean(_vals(recs, "maturation_cost_charged"))
    out["developmental_ablation_active_flag"] = 1.0 if any(bool(r["dev_ablate"]) for r in recs) else 0.0
    out["developmental_force_adult_active_flag"] = 1.0 if any(bool(r["dev_force_adult"]) for r in recs) else 0.0
    out["developmental_force_juvenile_active_flag"] = 1.0 if any(bool(r["dev_force_juvenile"]) for r in recs) else 0.0
    out["developmental_force_legacy_active_flag"] = 1.0 if any(bool(r["dev_force_legacy"]) for r in recs) else 0.0
    out["developmental_growth_disabled_active_flag"] = 1.0 if any(bool(r["dev_growth_disabled"]) for r in recs) else 0.0
    out["developmental_cost_charging_active_flag"] = 1.0 if any(bool(r["dev_cost_charging"]) for r in recs) else 0.0
    parent_maturity = _vals(parent_recs, "maturity_age")
    out["parent_maturity_age_mean"] = mean(parent_maturity) if parent_maturity else 0.0
    out["alive_maturity_age_mean"] = mean(_vals(recs, "maturity_age"))
    out["dead_maturity_age_mean"] = 0.0
    out["reproduced_maturity_age_mean"] = mean(_vals(reproduced, "maturity_age")) if reproduced else 0.0
    out["non_reproduced_maturity_age_mean"] = mean(_vals(non_reproduced, "maturity_age")) if non_reproduced else 0.0
    out["maturity_age_selection_delta_parent_vs_alive"] = out["parent_maturity_age_mean"] - out["alive_maturity_age_mean"]
    parent_dev_factor = _vals(parent_recs, "dev_factor")
    out["developmental_expression_selection_delta_parent_vs_alive"] = (mean(parent_dev_factor) if parent_dev_factor else 0.0) - mean(dev_factors)
    out["juvenile_survival_rate"] = 1.0 if juveniles else 0.0
    out["adult_survival_rate"] = 1.0 if mature else 0.0
    out["juvenile_reproduction_attempt_rate"] = (
        sum(1 for r in juveniles if float(r["offspring"]) > 0.0) / float(len(juveniles)) if juveniles else 0.0
    )
    out["adult_reproduction_attempt_rate"] = (
        sum(1 for r in mature if float(r["offspring"]) > 0.0) / float(len(mature)) if mature else 0.0
    )
    out["delayed_maturity_lineage_fraction"] = (
        sum(1 for r in recs if float(r["maturity_age"]) > 20.0) / n
    )
    dev_stages = [str(r["dev_stage"]) for r in recs]
    out["developmental_stage_entropy"] = _entropy(dev_stages)
    out["maturity_age_entropy"] = _entropy([int(r["maturity_age"]) for r in recs])
    dev_strategies = [r["developmental_strategy"] for r in recs]
    out["developmental_strategy_diversity"] = float(len(set(dev_strategies)))
    if dev_strategies:
        dev_counts: dict[Any, int] = {}
        for s in dev_strategies:
            dev_counts[s] = dev_counts.get(s, 0) + 1
        out["dominant_developmental_strategy_fraction"] = float(max(dev_counts.values())) / n
    out["developmental_strategy_count_approx"] = float(len(set(dev_strategies)))
    out["developmental_cost_zero_fraction"] = sum(
        1 for r in recs if float(r["dev_cost_hyp"]) + float(r["maturation_cost_hyp"]) <= 0.0
    ) / n
    adult_high = percentile(_vals(mature, "capacity"), 90) if mature else 0.0
    out["high_adult_cognition_low_developmental_cost_fraction"] = (
        sum(
            1
            for r in mature
            if float(r["capacity"]) >= adult_high and float(r["dev_cost_hyp"]) + float(r["maturation_cost_hyp"]) <= 0.0
        )
        / float(len(mature))
        if mature
        else 0.0
    )
    out["adult_cognition_from_birth_fraction"] = sum(1 for r in recs if bool(r["adult_from_birth"])) / n
    juvenile_high = percentile(cap, 90)
    out["juvenile_high_cognition_fraction"] = (
        sum(1 for r in juveniles if float(r["capacity"]) >= juvenile_high) / float(len(juveniles)) if juveniles else 0.0
    )
    out["developmental_growth_without_pressure_flag"] = 0.0
    out["direct_development_reward_present_flag"] = 0.0
    out["developmental_pressure_scenario_flag"] = 1.0 if bool(getattr(sim_cfg, "developmental_pressure_scenario", False)) else 0.0
    out["developmental_pressure_delayed_maturity_flag"] = (
        1.0 if str(getattr(sim_cfg, "developmental_pressure_type", "") or "") == "delayed_maturity" else 0.0
    )
    out["maturation_cost_active_flag"] = (
        1.0 if any(float(r["maturation_cost_hyp"]) > 0.0 or float(r["maturation_cost_charged"]) > 0.0 for r in recs) else 0.0
    )
    introspection_counts = _vals(recs, "introspection_count")
    introspection_costs_hyp = _vals(recs, "introspection_cost_hyp")
    introspection_costs_charged = _vals(recs, "introspection_cost_charged")
    introspection_masks = [int(r["introspection_mask"]) for r in recs]
    introspection_strategies = [r["introspection_strategy"] for r in recs]
    out["introspection_enabled_fraction"] = sum(1 for r in recs if bool(r["introspection_enabled"])) / n
    out["introspection_channel_active_count_mean"] = mean(introspection_counts)
    out["introspection_channel_active_count_p50"] = percentile(introspection_counts, 50)
    out["introspection_channel_active_count_p90"] = percentile(introspection_counts, 90)
    out["introspection_channel_active_count_max"] = max(introspection_counts) if introspection_counts else 0.0
    out["introspection_channel_active_count_min"] = min(introspection_counts) if introspection_counts else 0.0
    out["introspection_channel_active_count_std"] = std(introspection_counts)
    out["introspection_channel_active_fraction_mean"] = mean(_vals(recs, "introspection_fraction"))
    out["introspection_capacity_component_mean"] = mean(introspection_counts)
    out["introspection_capacity_component_p90"] = percentile(introspection_counts, 90)
    for cid in implemented_introspection_channel_ids():
        out[f"introspection_channel_{cid}_enabled_fraction"] = (
            sum(1 for r in recs if str(cid) in tuple(r["introspection_ids"])) / n
        )
    out["internal_energy_signal_mean"] = mean(_vals(recs, "internal_energy"))
    out["internal_health_signal_mean"] = mean(_vals(recs, "internal_health"))
    out["internal_drive_conflict_signal_mean"] = mean(_vals(recs, "internal_drive_conflict"))
    out["internal_prediction_error_signal_mean"] = mean(_vals(recs, "internal_prediction_error"))
    out["internal_memory_saturation_signal_mean"] = mean(_vals(recs, "internal_memory_saturation"))
    out["internal_cog_cost_burden_signal_mean"] = mean(_vals(recs, "internal_cog_cost_burden"))
    out["internal_developmental_expression_signal_mean"] = mean(_vals(recs, "internal_developmental_expression"))
    out["introspection_cost_hypothetical_mean"] = mean(introspection_costs_hyp)
    out["introspection_cost_hypothetical_total"] = float(sum(introspection_costs_hyp))
    out["introspection_cost_charged_mean"] = mean(introspection_costs_charged)
    out["introspection_cost_charged_total"] = float(sum(introspection_costs_charged))
    out["introspection_cost_fraction_of_cog_cost_mean"] = mean(_vals(recs, "introspection_cost_fraction"))
    high_intro = percentile(introspection_counts, 90)
    out["high_introspection_low_cost_fraction"] = (
        sum(1 for r in recs if float(r["introspection_count"]) >= high_intro and float(r["introspection_cost_hyp"]) <= 0.0) / n
    )
    out["introspection_enabled_without_cost_fraction"] = (
        sum(1 for r in recs if bool(r["introspection_enabled"]) and float(r["introspection_cost_hyp"]) <= 0.0) / n
    )
    out["introspection_channel_mask_entropy"] = _entropy(introspection_masks)
    out["introspection_channel_mask_diversity"] = float(len(set(introspection_masks)))
    if introspection_masks:
        mask_counts: dict[int, int] = {}
        for mask in introspection_masks:
            mask_counts[mask] = mask_counts.get(mask, 0) + 1
        out["dominant_introspection_channel_mask_fraction"] = float(max(mask_counts.values())) / n
    out["introspection_strategy_count_approx"] = float(len(set(introspection_strategies)))
    out["introspection_memory_strategy_entropy"] = _entropy(
        [(int(r["introspection_mask"]), int(r["memory"])) for r in recs]
    )
    out["introspection_prediction_strategy_entropy"] = _entropy(
        [(int(r["introspection_mask"]), round(float(r["effective_pred_lr"]), 2)) for r in recs]
    )
    out["introspection_perception_strategy_entropy"] = _entropy(
        [(int(r["introspection_mask"]), int(r["sensor_channel_mask"]), int(r["sensor_radius"])) for r in recs]
    )
    out["introspection_development_strategy_entropy"] = _entropy(
        [(int(r["introspection_mask"]), str(r["dev_stage"]), round(float(r["dev_factor"]), 2)) for r in recs]
    )
    out["full_cognitive_strategy_count_approx"] = float(len(set((r["strategy"], r["introspection_strategy"]) for r in recs)))
    mem_hi = percentile(mem, 90)
    pred_hi = percentile(effective_pred_lr, 90)
    perception_hi = percentile(_vals(recs, "cell_channel_capacity"), 90)
    out["high_introspection_high_memory_fraction"] = (
        sum(1 for r in recs if float(r["introspection_count"]) >= high_intro and float(r["memory"]) >= mem_hi) / n
    )
    out["high_introspection_high_prediction_fraction"] = (
        sum(1 for r in recs if float(r["introspection_count"]) >= high_intro and float(r["effective_pred_lr"]) >= pred_hi) / n
    )
    out["high_introspection_high_perception_fraction"] = (
        sum(1 for r in recs if float(r["introspection_count"]) >= high_intro and float(r["cell_channel_capacity"]) >= perception_hi) / n
    )
    out["high_introspection_developmental_fraction"] = (
        sum(1 for r in recs if float(r["introspection_count"]) >= high_intro and bool(r["dev_active"])) / n
    )
    out["introspection_cost_zero_fraction"] = sum(1 for r in recs if float(r["introspection_cost_hyp"]) <= 0.0) / n
    out["introspection_growth_without_pressure_flag"] = 0.0
    out["direct_introspection_reward_present_flag"] = 0.0
    out["direct_internal_state_reward_present_flag"] = 0.0
    out["introspection_ablation_active_flag"] = 1.0 if any(bool(r["introspection_ablate"]) for r in recs) else 0.0
    out["introspection_force_no_active_flag"] = 1.0 if any(bool(r["introspection_force_no"]) for r in recs) else 0.0
    out["introspection_force_minimal_active_flag"] = 1.0 if any(bool(r["introspection_force_minimal"]) for r in recs) else 0.0
    out["introspection_force_all_active_flag"] = 1.0 if any(bool(r["introspection_force_all"]) for r in recs) else 0.0
    out["introspection_force_default_active_flag"] = 1.0 if any(bool(r["introspection_force_default"]) for r in recs) else 0.0
    out["introspection_mask_to_neutral_active_flag"] = 1.0 if any(bool(r["introspection_mask_neutral"]) for r in recs) else 0.0
    out["introspection_cost_charging_active_flag"] = 1.0 if any(bool(r["introspection_cost_charging"]) for r in recs) else 0.0
    out["introspection_pressure_scenario_flag"] = 1.0 if bool(getattr(sim_cfg, "introspection_pressure_scenario", False)) else 0.0
    out["introspection_pressure_drive_conflict_flag"] = (
        1.0 if str(getattr(sim_cfg, "introspection_pressure_type", "") or "") == "drive_conflict" else 0.0
    )
    out["introspection_policy_input_active_flag"] = 1.0 if any(bool(r["introspection_policy_input"]) for r in recs) else 0.0

    self_reg_counts = _vals(recs, "self_regulation_count")
    self_reg_costs_hyp = _vals(recs, "self_regulation_cost_hyp")
    self_reg_costs_charged = _vals(recs, "self_regulation_cost_charged")
    self_reg_masks = [int(r["self_regulation_mask"]) for r in recs]
    self_reg_strategies = [r["self_regulation_strategy"] for r in recs]
    out["self_regulation_enabled_fraction"] = sum(1 for r in recs if bool(r["self_regulation_enabled"])) / n
    out["self_regulation_control_active_count_mean"] = mean(self_reg_counts)
    out["self_regulation_control_active_count_p50"] = percentile(self_reg_counts, 50)
    out["self_regulation_control_active_count_p90"] = percentile(self_reg_counts, 90)
    out["self_regulation_control_active_count_max"] = max(self_reg_counts) if self_reg_counts else 0.0
    out["self_regulation_control_active_count_min"] = min(self_reg_counts) if self_reg_counts else 0.0
    out["self_regulation_control_active_count_std"] = std(self_reg_counts)
    out["self_regulation_control_active_fraction_mean"] = mean(_vals(recs, "self_regulation_fraction"))
    out["self_regulation_capacity_component_mean"] = mean(self_reg_counts)
    out["self_regulation_capacity_component_p90"] = percentile(self_reg_counts, 90)
    for cid in implemented_self_regulation_control_ids():
        out[f"self_regulation_control_{cid}_enabled_fraction"] = (
            sum(1 for r in recs if str(cid) in tuple(r["self_regulation_ids"])) / n
        )
    out["self_regulation_control_delta_abs_mean"] = mean(_vals(recs, "self_reg_delta_abs"))
    out["self_regulation_control_delta_abs_p90"] = percentile(_vals(recs, "self_reg_delta_abs_p90"), 90)
    out["self_regulation_control_offset_abs_mean"] = mean(_vals(recs, "self_reg_offset_abs"))
    out["self_regulation_control_offset_abs_p90"] = percentile(_vals(recs, "self_reg_offset_abs_p90"), 90)
    out["self_regulation_control_clipped_fraction"] = mean(_vals(recs, "self_reg_clipped_fraction"))
    out["self_regulation_control_saturation_fraction"] = mean(_vals(recs, "self_reg_saturation_fraction"))
    out["self_regulation_control_application_count_total"] = float(sum(_vals(recs, "self_reg_application_count")))
    out["self_regulation_control_neutral_fraction"] = mean(_vals(recs, "self_reg_neutral_fraction"))
    out["self_regulation_state_reset_count_total"] = float(sum(_vals(recs, "self_reg_reset_count")))
    out["self_regulation_decay_applied_count_total"] = float(sum(_vals(recs, "self_reg_decay_count")))
    out["effective_exploration_pressure_mean"] = mean(_vals(recs, "effective_exploration_pressure"))
    out["effective_action_temperature_mean"] = mean(_vals(recs, "effective_action_temperature"))
    out["effective_drive_gain_mean"] = mean(_vals(recs, "effective_drive_gain"))
    out["effective_reproduction_threshold_mean"] = mean(_vals(recs, "effective_reproduction_threshold"))
    out["effective_self_regulated_prediction_lr_mean"] = mean(_vals(recs, "effective_prediction_lr_regulated"))
    out["self_regulation_cost_hypothetical_mean"] = mean(self_reg_costs_hyp)
    out["self_regulation_cost_hypothetical_total"] = float(sum(self_reg_costs_hyp))
    out["self_regulation_cost_charged_mean"] = mean(self_reg_costs_charged)
    out["self_regulation_cost_charged_total"] = float(sum(self_reg_costs_charged))
    out["self_regulation_cost_fraction_of_cog_cost_mean"] = mean(_vals(recs, "self_regulation_cost_fraction"))
    high_self_reg = percentile(self_reg_counts, 90)
    out["high_self_regulation_low_cost_fraction"] = (
        sum(1 for r in recs if float(r["self_regulation_count"]) >= high_self_reg and float(r["self_regulation_cost_hyp"]) <= 0.0) / n
    )
    out["self_regulation_enabled_without_cost_fraction"] = (
        sum(1 for r in recs if bool(r["self_regulation_enabled"]) and float(r["self_regulation_cost_hyp"]) <= 0.0) / n
    )
    out["self_regulation_control_mask_entropy"] = _entropy(self_reg_masks)
    out["self_regulation_control_mask_diversity"] = float(len(set(self_reg_masks)))
    if self_reg_masks:
        sr_counts: dict[int, int] = {}
        for mask in self_reg_masks:
            sr_counts[mask] = sr_counts.get(mask, 0) + 1
        out["dominant_self_regulation_control_mask_fraction"] = float(max(sr_counts.values())) / n
    out["self_regulation_strategy_count_approx"] = float(len(set(self_reg_strategies)))
    out["introspection_self_regulation_strategy_entropy"] = _entropy(
        [(int(r["introspection_mask"]), int(r["self_regulation_mask"])) for r in recs]
    )
    out["high_introspection_high_self_regulation_fraction"] = (
        sum(1 for r in recs if float(r["introspection_count"]) >= high_intro and float(r["self_regulation_count"]) >= high_self_reg) / n
    )
    out["self_regulation_without_introspection_fraction"] = (
        sum(1 for r in recs if bool(r["self_regulation_enabled"]) and not bool(r["introspection_enabled"])) / n
    )
    out["introspection_without_self_regulation_fraction"] = (
        sum(1 for r in recs if bool(r["introspection_enabled"]) and not bool(r["self_regulation_enabled"])) / n
    )
    out["self_regulation_policy_input_independent_flag"] = 1.0
    out["self_regulation_cost_zero_fraction"] = sum(1 for r in recs if float(r["self_regulation_cost_hyp"]) <= 0.0) / n
    out["self_regulation_growth_without_pressure_flag"] = 0.0
    out["direct_self_regulation_reward_present_flag"] = 0.0
    out["self_modification_present_flag"] = 1.0 if any(bool(r["self_modification_active"]) for r in recs) else 0.0
    out["genome_editing_present_flag"] = 1.0 if any(bool(r["genome_editing_active"]) for r in recs) else 0.0
    out["self_regulation_ablation_active_flag"] = 1.0 if any(bool(r["self_regulation_ablate"]) for r in recs) else 0.0
    out["self_regulation_force_no_active_flag"] = 1.0 if any(bool(r["self_regulation_force_no"]) for r in recs) else 0.0
    out["self_regulation_force_all_active_flag"] = 1.0 if any(bool(r["self_regulation_force_all"]) for r in recs) else 0.0
    out["self_regulation_force_default_active_flag"] = 1.0 if any(bool(r["self_regulation_force_default"]) for r in recs) else 0.0
    out["self_regulation_mask_to_neutral_active_flag"] = 1.0 if any(bool(r["self_regulation_mask_neutral"]) for r in recs) else 0.0
    out["self_regulation_freeze_state_active_flag"] = 1.0 if any(bool(r["self_regulation_freeze"]) for r in recs) else 0.0
    out["self_regulation_reset_each_tick_active_flag"] = 1.0 if any(bool(r["self_regulation_reset_each_tick"]) for r in recs) else 0.0
    out["self_regulation_decay_disabled_active_flag"] = 1.0 if any(bool(r["self_regulation_decay_disabled"]) for r in recs) else 0.0
    out["self_regulation_cost_charging_active_flag"] = 1.0 if any(bool(r["self_regulation_cost_charging"]) for r in recs) else 0.0
    out["self_regulation_pressure_scenario_flag"] = 1.0 if bool(getattr(sim_cfg, "self_regulation_pressure_scenario", False)) else 0.0
    out["self_regulation_pressure_hunger_fear_conflict_flag"] = (
        1.0 if str(getattr(sim_cfg, "self_regulation_pressure_type", "") or "") == "hunger_fear_conflict" else 0.0
    )
    self_mod_counts = _vals(recs, "self_modification_count")
    self_mod_costs_hyp = _vals(recs, "self_modification_cost_hyp")
    self_mod_costs_charged = _vals(recs, "self_modification_cost_charged")
    self_mod_masks = [int(r["self_modification_mask"]) for r in recs]
    self_mod_strategies = [r["self_modification_strategy"] for r in recs]
    out["self_modification_enabled_fraction"] = sum(1 for r in recs if bool(r["self_modification_enabled"])) / n
    out["self_modification_target_active_count_mean"] = mean(self_mod_counts)
    out["self_modification_target_active_count_p50"] = percentile(self_mod_counts, 50)
    out["self_modification_target_active_count_p90"] = percentile(self_mod_counts, 90)
    out["self_modification_target_active_count_max"] = max(self_mod_counts) if self_mod_counts else 0.0
    out["self_modification_target_active_count_min"] = min(self_mod_counts) if self_mod_counts else 0.0
    out["self_modification_target_active_count_std"] = std(self_mod_counts)
    out["self_modification_target_active_fraction_mean"] = mean(_vals(recs, "self_modification_fraction"))
    out["self_modification_capacity_component_mean"] = mean(self_mod_counts)
    out["self_modification_capacity_component_p90"] = percentile(self_mod_counts, 90)
    for tid in implemented_self_modification_target_ids():
        out[f"self_modification_target_{tid}_enabled_fraction"] = (
            sum(1 for r in recs if str(tid) in tuple(r["self_modification_ids"])) / n
        )
    out["self_modification_delta_abs_mean"] = mean(_vals(recs, "self_mod_delta_abs"))
    out["self_modification_delta_abs_p90"] = percentile(_vals(recs, "self_mod_delta_abs_p90"), 90)
    out["self_modification_offset_abs_mean"] = mean(_vals(recs, "self_mod_offset_abs"))
    out["self_modification_offset_abs_p90"] = percentile(_vals(recs, "self_mod_offset_abs_p90"), 90)
    out["self_modification_clipped_fraction"] = mean(_vals(recs, "self_mod_clipped_fraction"))
    out["self_modification_saturation_fraction"] = mean(_vals(recs, "self_mod_saturation_fraction"))
    out["self_modification_event_count_total"] = float(sum(_vals(recs, "self_mod_event_count")))
    out["self_modification_neutral_command_fraction"] = mean(_vals(recs, "self_mod_neutral_fraction"))
    out["self_modification_state_reset_count_total"] = float(sum(_vals(recs, "self_mod_reset_count")))
    out["self_modification_decay_applied_count_total"] = float(sum(_vals(recs, "self_mod_decay_count")))
    out["self_modification_lifetime_persistence_mean"] = mean(_vals(recs, "self_mod_lifetime_persistence"))
    out["self_modification_reversal_count_total"] = float(sum(_vals(recs, "self_mod_reversal_count")))
    out["modified_exploration_bias_mean"] = mean(_vals(recs, "modified_exploration_bias"))
    out["modified_action_temperature_mean"] = mean(_vals(recs, "modified_action_temperature"))
    out["modified_drive_gain_mean"] = mean(_vals(recs, "modified_drive_gain"))
    out["modified_reproduction_threshold_mean"] = mean(_vals(recs, "modified_reproduction_threshold"))
    out["modified_prediction_lr_mean"] = mean(_vals(recs, "modified_prediction_lr"))
    out["self_modification_cost_hypothetical_mean"] = mean(self_mod_costs_hyp)
    out["self_modification_cost_hypothetical_total"] = float(sum(self_mod_costs_hyp))
    out["self_modification_cost_charged_mean"] = mean(self_mod_costs_charged)
    out["self_modification_cost_charged_total"] = float(sum(self_mod_costs_charged))
    out["self_modification_cost_fraction_of_cog_cost_mean"] = mean(_vals(recs, "self_modification_cost_fraction"))
    high_self_mod = percentile(self_mod_counts, 90)
    out["high_self_modification_low_cost_fraction"] = (
        sum(1 for r in recs if float(r["self_modification_count"]) >= high_self_mod and float(r["self_modification_cost_hyp"]) <= 0.0) / n
    )
    out["self_modification_enabled_without_cost_fraction"] = (
        sum(1 for r in recs if bool(r["self_modification_enabled"]) and float(r["self_modification_cost_hyp"]) <= 0.0) / n
    )
    out["self_modification_target_mask_entropy"] = _entropy(self_mod_masks)
    out["self_modification_target_mask_diversity"] = float(len(set(self_mod_masks)))
    if self_mod_masks:
        sm_counts: dict[int, int] = {}
        for mask in self_mod_masks:
            sm_counts[mask] = sm_counts.get(mask, 0) + 1
        out["dominant_self_modification_target_mask_fraction"] = float(max(sm_counts.values())) / n
    out["self_modification_strategy_count_approx"] = float(len(set(self_mod_strategies)))
    out["introspection_self_modification_strategy_entropy"] = _entropy(
        [(int(r["introspection_mask"]), int(r["self_modification_mask"])) for r in recs]
    )
    out["self_regulation_self_modification_strategy_entropy"] = _entropy(
        [(int(r["self_regulation_mask"]), int(r["self_modification_mask"])) for r in recs]
    )
    out["high_introspection_high_self_modification_fraction"] = (
        sum(1 for r in recs if float(r["introspection_count"]) >= high_intro and float(r["self_modification_count"]) >= high_self_mod) / n
    )
    out["high_self_regulation_high_self_modification_fraction"] = (
        sum(1 for r in recs if float(r["self_regulation_count"]) >= high_self_reg and float(r["self_modification_count"]) >= high_self_mod) / n
    )
    out["self_modification_without_introspection_fraction"] = (
        sum(1 for r in recs if bool(r["self_modification_enabled"]) and not bool(r["introspection_enabled"])) / n
    )
    out["self_modification_without_self_regulation_fraction"] = (
        sum(1 for r in recs if bool(r["self_modification_enabled"]) and not bool(r["self_regulation_enabled"])) / n
    )
    out["self_modification_cost_zero_fraction"] = sum(1 for r in recs if float(r["self_modification_cost_hyp"]) <= 0.0) / n
    out["self_modification_growth_without_pressure_flag"] = 0.0
    out["direct_self_modification_reward_present_flag"] = 0.0
    out["host_access_present_flag"] = 0.0
    out["source_code_modification_present_flag"] = 0.0
    out["self_modification_present_flag"] = 1.0 if any(bool(r["self_modification_enabled"]) for r in recs) else 0.0
    out["genome_editing_present_flag"] = 1.0 if any(bool(r["genome_editing_active"]) for r in recs) else 0.0
    out["self_modification_ablation_active_flag"] = 1.0 if any(bool(r["self_modification_ablate"]) for r in recs) else 0.0
    out["self_modification_force_no_active_flag"] = 1.0 if any(bool(r["self_modification_force_no"]) for r in recs) else 0.0
    out["self_modification_force_all_active_flag"] = 1.0 if any(bool(r["self_modification_force_all"]) for r in recs) else 0.0
    out["self_modification_force_default_active_flag"] = 1.0 if any(bool(r["self_modification_force_default"]) for r in recs) else 0.0
    out["self_modification_mask_to_neutral_active_flag"] = 1.0 if any(bool(r["self_modification_mask_neutral"]) for r in recs) else 0.0
    out["self_modification_freeze_state_active_flag"] = 1.0 if any(bool(r["self_modification_freeze"]) for r in recs) else 0.0
    out["self_modification_reset_each_tick_active_flag"] = 1.0 if any(bool(r["self_modification_reset_each_tick"]) for r in recs) else 0.0
    out["self_modification_reset_on_reproduction_active_flag"] = 1.0 if any(bool(r["self_modification_reset_on_reproduction"]) for r in recs) else 0.0
    out["self_modification_decay_disabled_active_flag"] = 1.0 if any(bool(r["self_modification_decay_disabled"]) for r in recs) else 0.0
    out["self_modification_cost_charging_active_flag"] = 1.0 if any(bool(r["self_modification_cost_charging"]) for r in recs) else 0.0
    out["self_modification_pressure_scenario_flag"] = 1.0 if bool(getattr(sim_cfg, "self_modification_pressure_scenario", False)) else 0.0
    out["self_modification_pressure_persistent_risk_flag"] = (
        1.0 if str(getattr(sim_cfg, "self_modification_pressure_type", "") or "") == "persistent_risk" else 0.0
    )
    out["self_modification_policy_output_active_flag"] = 1.0 if any(bool(r["self_modification_policy_output"]) for r in recs) else 0.0
    directed_counts = _vals(recs, "directed_inheritance_count")
    directed_costs_hyp = _vals(recs, "directed_inheritance_cost_hyp")
    directed_costs_charged = _vals(recs, "directed_inheritance_cost_charged")
    directed_masks = [int(r["directed_inheritance_mask"]) for r in recs]
    directed_strategies = [r["directed_inheritance_strategy"] for r in recs]
    out["directed_inheritance_enabled_fraction"] = sum(1 for r in recs if bool(r["directed_inheritance_enabled"])) / n
    out["directed_inheritance_channel_active_count_mean"] = mean(directed_counts)
    out["directed_inheritance_channel_active_count_p50"] = percentile(directed_counts, 50)
    out["directed_inheritance_channel_active_count_p90"] = percentile(directed_counts, 90)
    out["directed_inheritance_channel_active_count_max"] = max(directed_counts) if directed_counts else 0.0
    out["directed_inheritance_channel_active_count_min"] = min(directed_counts) if directed_counts else 0.0
    out["directed_inheritance_channel_active_count_std"] = std(directed_counts)
    out["directed_inheritance_channel_active_fraction_mean"] = mean(_vals(recs, "directed_inheritance_fraction"))
    out["directed_inheritance_capacity_component_mean"] = mean(directed_counts)
    out["directed_inheritance_capacity_component_p90"] = percentile(directed_counts, 90)
    for cid in implemented_directed_inheritance_channel_ids():
        out[f"directed_inheritance_channel_{cid}_enabled_fraction"] = (
            sum(1 for r in recs if str(cid) in tuple(r["directed_inheritance_ids"])) / n
        )
    out["directed_inheritance_delta_abs_mean"] = mean(_vals(recs, "directed_delta_abs"))
    out["directed_inheritance_delta_abs_p90"] = percentile(_vals(recs, "directed_delta_abs_p90"), 90)
    out["directed_inheritance_payload_abs_mean"] = mean(_vals(recs, "directed_payload_abs"))
    out["directed_inheritance_payload_abs_p90"] = percentile(_vals(recs, "directed_payload_abs_p90"), 90)
    out["directed_inheritance_clipped_fraction"] = mean(_vals(recs, "directed_clipped_fraction"))
    out["directed_inheritance_saturation_fraction"] = mean(_vals(recs, "directed_saturation_fraction"))
    out["directed_inheritance_event_count_total"] = float(sum(_vals(recs, "directed_event_count")))
    out["directed_inheritance_neutral_command_fraction"] = mean(_vals(recs, "directed_neutral_fraction"))
    out["directed_inheritance_payload_reset_count_total"] = float(sum(_vals(recs, "directed_reset_count")))
    out["directed_inheritance_payload_applied_fraction"] = mean(_vals(recs, "directed_payload_applied"))
    out["directed_inheritance_payload_frozen_fraction"] = mean(_vals(recs, "directed_payload_frozen"))
    out["offspring_developmental_expression_bias_mean"] = mean(_vals(recs, "offspring_developmental_expression_bias"))
    out["offspring_juvenile_cognition_scale_bias_mean"] = mean(_vals(recs, "offspring_juvenile_cognition_scale_bias"))
    out["offspring_memory_expression_bias_mean"] = mean(_vals(recs, "offspring_memory_expression_bias"))
    out["offspring_prediction_lr_bias_mean"] = mean(_vals(recs, "offspring_prediction_lr_bias"))
    out["offspring_exploration_bias_marker_mean"] = mean(_vals(recs, "offspring_exploration_bias_marker"))
    out["offspring_reproduction_threshold_bias_mean"] = mean(_vals(recs, "offspring_reproduction_threshold_bias"))
    out["directed_inheritance_cost_hypothetical_mean"] = mean(directed_costs_hyp)
    out["directed_inheritance_cost_hypothetical_total"] = float(sum(directed_costs_hyp))
    out["directed_inheritance_cost_charged_mean"] = mean(directed_costs_charged)
    out["directed_inheritance_cost_charged_total"] = float(sum(directed_costs_charged))
    out["directed_inheritance_cost_fraction_of_cog_cost_mean"] = mean(_vals(recs, "directed_inheritance_cost_fraction"))
    high_directed = percentile(directed_counts, 90)
    out["high_directed_inheritance_low_cost_fraction"] = (
        sum(1 for r in recs if float(r["directed_inheritance_count"]) >= high_directed and float(r["directed_inheritance_cost_hyp"]) <= 0.0) / n
    )
    out["directed_inheritance_enabled_without_cost_fraction"] = (
        sum(1 for r in recs if bool(r["directed_inheritance_enabled"]) and float(r["directed_inheritance_cost_hyp"]) <= 0.0) / n
    )
    out["directed_inheritance_channel_mask_entropy"] = _entropy(directed_masks)
    out["directed_inheritance_channel_mask_diversity"] = float(len(set(directed_masks)))
    if directed_masks:
        di_counts: dict[int, int] = {}
        for mask in directed_masks:
            di_counts[mask] = di_counts.get(mask, 0) + 1
        out["dominant_directed_inheritance_channel_mask_fraction"] = float(max(di_counts.values())) / n
    out["directed_inheritance_strategy_count_approx"] = float(len(set(directed_strategies)))
    out["self_modification_directed_inheritance_strategy_entropy"] = _entropy(
        [(int(r["self_modification_mask"]), int(r["directed_inheritance_mask"])) for r in recs]
    )
    out["self_regulation_directed_inheritance_strategy_entropy"] = _entropy(
        [(int(r["self_regulation_mask"]), int(r["directed_inheritance_mask"])) for r in recs]
    )
    out["high_self_modification_high_directed_inheritance_fraction"] = (
        sum(1 for r in recs if float(r["self_modification_count"]) >= high_self_mod and float(r["directed_inheritance_count"]) >= high_directed) / n
    )
    out["high_self_regulation_high_directed_inheritance_fraction"] = (
        sum(1 for r in recs if float(r["self_regulation_count"]) >= high_self_reg and float(r["directed_inheritance_count"]) >= high_directed) / n
    )
    out["directed_inheritance_without_self_modification_fraction"] = (
        sum(1 for r in recs if bool(r["directed_inheritance_enabled"]) and not bool(r["self_modification_enabled"])) / n
    )
    out["self_modification_without_directed_inheritance_fraction"] = (
        sum(1 for r in recs if bool(r["self_modification_enabled"]) and not bool(r["directed_inheritance_enabled"])) / n
    )
    out["directed_inheritance_cost_zero_fraction"] = sum(1 for r in recs if float(r["directed_inheritance_cost_hyp"]) <= 0.0) / n
    out["directed_inheritance_growth_without_pressure_flag"] = 0.0
    out["direct_directed_inheritance_reward_present_flag"] = 0.0
    out["direct_parental_investment_reward_present_flag"] = 0.0
    out["direct_offspring_quality_reward_present_flag"] = 0.0
    out["mutation_rate_control_present_flag"] = 1.0 if any(bool(r["mutation_rate_control_active"]) for r in recs) else 0.0
    out["recombination_control_present_flag"] = 1.0 if any(bool(r["recombination_control_active"]) for r in recs) else 0.0
    out["directed_inheritance_ablation_active_flag"] = 1.0 if any(bool(r["directed_inheritance_ablate"]) for r in recs) else 0.0
    out["directed_inheritance_force_no_active_flag"] = 1.0 if any(bool(r["directed_inheritance_force_no"]) for r in recs) else 0.0
    out["directed_inheritance_force_all_active_flag"] = 1.0 if any(bool(r["directed_inheritance_force_all"]) for r in recs) else 0.0
    out["directed_inheritance_force_default_active_flag"] = 1.0 if any(bool(r["directed_inheritance_force_default"]) for r in recs) else 0.0
    out["directed_inheritance_mask_to_neutral_active_flag"] = 1.0 if any(bool(r["directed_inheritance_mask_neutral"]) for r in recs) else 0.0
    out["directed_inheritance_freeze_payload_active_flag"] = 1.0 if any(bool(r["directed_inheritance_freeze"]) for r in recs) else 0.0
    out["directed_inheritance_reset_on_birth_active_flag"] = 1.0 if any(bool(r["directed_inheritance_reset_birth"]) for r in recs) else 0.0
    out["directed_inheritance_reset_on_maturity_active_flag"] = 1.0 if any(bool(r["directed_inheritance_reset_maturity"]) for r in recs) else 0.0
    out["directed_inheritance_cost_charging_active_flag"] = 1.0 if any(bool(r["directed_inheritance_cost_charging"]) for r in recs) else 0.0
    out["directed_inheritance_pressure_scenario_flag"] = 1.0 if bool(getattr(sim_cfg, "directed_inheritance_pressure_scenario", False)) else 0.0
    out["directed_inheritance_pressure_developmental_bias_flag"] = (
        1.0 if str(getattr(sim_cfg, "directed_inheritance_pressure_type", "") or "") == "developmental_bias" else 0.0
    )
    out["directed_inheritance_policy_output_active_flag"] = 1.0 if any(bool(r["directed_inheritance_policy_output"]) for r in recs) else 0.0
    out["full_cognitive_strategy_count_approx"] = float(
        len(set((r["strategy"], r["introspection_strategy"], r["self_regulation_strategy"], r["self_modification_strategy"], r["directed_inheritance_strategy"]) for r in recs))
    )

    lineages: dict[int, list[Mapping[str, Any]]] = {}
    for r in recs:
        lineages.setdefault(int(r["lineage"]), []).append(r)
    lineage_caps = [mean(_vals(v, "capacity")) for _, v in sorted(lineages.items(), key=lambda kv: int(kv[0]))]
    lineage_mems = [mean(_vals(v, "memory")) for _, v in sorted(lineages.items(), key=lambda kv: int(kv[0]))]
    lineage_preds = [mean(_vals(v, "effective_pred_lr")) for _, v in sorted(lineages.items(), key=lambda kv: int(kv[0]))]
    lineage_costs = [mean(_vals(v, "cost")) for _, v in sorted(lineages.items(), key=lambda kv: int(kv[0]))]
    out["lineage_cog_capacity_mean"] = mean(lineage_caps)
    out["lineage_memory_mean"] = mean(lineage_mems)
    out["lineage_prediction_lr_mean"] = mean(lineage_preds)
    out["lineage_cognitive_cost_mean"] = mean(lineage_costs)
    out["top_lineage_cog_capacity"] = max(lineage_caps) if lineage_caps else 0.0
    out["top_lineage_memory_dim"] = max(lineage_mems) if lineage_mems else 0.0
    out["lineage_cognitive_diversity"] = _entropy([int(r["lineage"]) for r in recs])
    out["mem_evo_lineage_memory_dim_summary_json"] = _json_dumps(_lineage_memory_payload(recs))

    out["cognition_cost_zero_fraction"] = sum(1 for r in recs if float(r["cost"]) <= 0.0) / n
    high_threshold = percentile(cap, 90)
    out["high_cognition_low_cost_fraction"] = (
        sum(1 for r in recs if float(r["capacity"]) >= high_threshold and float(r["cost"]) <= 0.0) / n
    )
    out["memory_enabled_without_cost_fraction"] = (
        sum(1 for r in recs if bool(r["memory_enabled"]) and float(r["memory_cost"]) <= 0.0) / n
    )
    out["prediction_enabled_without_cost_fraction"] = (
        sum(1 for r in recs if bool(r["pred_enabled"]) and float(r["prediction_cost_hyp"]) <= 0.0) / n
    )
    out["cognition_growth_without_environment_pressure_flag"] = 0.0
    out["direct_internal_selection_pressure_present_flag"] = 1.0 if _direct_internal_pressure_present(sim_cfg) else 0.0
    out["direct_cognition_reward_present_flag"] = out["direct_internal_selection_pressure_present_flag"]
    out["cognitive_metric_only_selection_risk_flag"] = out["direct_internal_selection_pressure_present_flag"]
    out["cognition_capacity_in_flat_world_baseline"] = mean(cap)

    out["sensor_channel_active_count_mean"] = mean(sensor_counts)
    out["sensor_channel_active_count_p50"] = percentile(sensor_counts, 50)
    out["sensor_channel_active_count_p90"] = percentile(sensor_counts, 90)
    out["sensor_channel_active_count_max"] = max(sensor_counts) if sensor_counts else 0.0
    out["sensor_channel_active_count_min"] = min(sensor_counts) if sensor_counts else 0.0
    out["sensor_channel_active_count_std"] = std(sensor_counts)
    out["sensor_channel_potential_count"] = max(_vals(recs, "sensor_channel_potential")) if recs else 0.0
    out["sensor_channel_active_fraction_mean"] = (
        mean([float(r["sensor_channel_count"]) / float(r["sensor_channel_potential"]) for r in recs if float(r["sensor_channel_potential"]) > 0.0])
    )
    out["perception_capacity_component_mean"] = mean(_vals(recs, "perception_component"))
    out["perception_capacity_component_p90"] = percentile(_vals(recs, "perception_component"), 90)

    channel_counts: dict[str, int] = {c.channel_id: 0 for c in perception_channel_registry() if c.active_now and not c.future_reserved}
    for r in recs:
        ids = set(r.get("sensor_channel_ids", ()))
        for cid in channel_counts:
            if cid in ids:
                channel_counts[cid] += 1
    for channel in perception_channel_registry():
        if channel.active_now and not channel.future_reserved:
            frac = float(channel_counts.get(channel.channel_id, 0)) / n
            out[f"sensor_channel_{channel.channel_id}_enabled_fraction"] = frac
            if channel.patch_channel:
                out[f"patch_channel_{channel.channel_id}_enabled_fraction"] = frac

    out["perception_cost_mean"] = mean(perception_costs)
    out["perception_cost_p90"] = percentile(perception_costs, 90)
    out["perception_cost_max"] = max(perception_costs) if perception_costs else 0.0
    out["perception_cost_total"] = sum(perception_costs)
    out["perception_cost_charged_mean"] = mean(perception_charged_costs)
    out["perception_cost_charged_total"] = sum(perception_charged_costs)
    out["perception_cost_hypothetical_mean"] = mean(perception_hyp_costs)
    out["perception_cost_fraction_of_cog_cost_mean"] = mean(
        [
            float(r["perception_cost"]) / float(r["cost"])
            for r in recs
            if float(r["cost"]) > 0.0 and float(r["perception_cost"]) > 0.0
        ]
    )
    out["parent_sensor_channel_count_mean"] = mean(parent_sensor_counts) if parent_sensor_counts else 0.0
    out["alive_sensor_channel_count_mean"] = mean(sensor_counts)
    out["perception_selection_delta_parent_vs_alive"] = out["parent_sensor_channel_count_mean"] - out["alive_sensor_channel_count_mean"]
    masks = [int(r["sensor_channel_mask"]) for r in recs]
    out["sensor_channel_mask_diversity"] = float(len(set(masks)))
    out["sensor_channel_mask_entropy"] = _entropy(masks)
    if masks:
        counts: dict[int, int] = {}
        for m in masks:
            counts[m] = counts.get(m, 0) + 1
        out["dominant_sensor_channel_mask_fraction"] = float(max(counts.values())) / n
    out["sensor_channel_strategy_count_approx"] = float(len(set(masks)))
    out["perception_cost_zero_fraction"] = sum(1 for r in recs if float(r["perception_cost_hyp"]) <= 0.0) / n
    high_perc_threshold = percentile(sensor_counts, 90)
    out["high_perception_low_cost_fraction"] = (
        sum(1 for r in recs if float(r["sensor_channel_count"]) >= high_perc_threshold and float(r["perception_cost_hyp"]) <= 0.0) / n
    )
    out["sensor_channels_enabled_without_cost_fraction"] = (
        sum(1 for r in recs if float(r["sensor_channel_count"]) > 0.0 and float(r["perception_cost_hyp"]) <= 0.0) / n
    )
    out["perception_growth_without_pressure_flag"] = 0.0
    out["sensor_channel_mask_ablation_active_flag"] = 1.0 if any(bool(r["sensor_channel_ablate"]) for r in recs) else 0.0
    out["sensor_channel_force_all_active_flag"] = 1.0 if any(bool(r["sensor_channel_force_all"]) for r in recs) else 0.0
    out["sensor_channel_force_minimal_active_flag"] = 1.0 if any(bool(r["sensor_channel_force_min"]) for r in recs) else 0.0

    out["sensor_radius_mean"] = mean(sensor_radii)
    out["sensor_radius_p50"] = percentile(sensor_radii, 50)
    out["sensor_radius_p90"] = percentile(sensor_radii, 90)
    out["sensor_radius_max"] = max(sensor_radii) if sensor_radii else 0.0
    out["sensor_radius_min"] = min(sensor_radii) if sensor_radii else 0.0
    out["sensor_radius_std"] = std(sensor_radii)
    for radius in (1, 2, 3):
        out[f"sensor_radius_{radius}_fraction"] = sum(1 for x in sensor_radii if int(x) == radius) / n
    out["observed_cell_count_mean"] = mean(observed_cells)
    out["observed_cell_count_p50"] = percentile(observed_cells, 50)
    out["observed_cell_count_p90"] = percentile(observed_cells, 90)
    out["observed_cell_count_max"] = max(observed_cells) if observed_cells else 0.0
    out["active_cell_fraction_mean"] = mean(_vals(recs, "active_cell_fraction"))
    out["cell_channel_capacity_mean"] = mean(cell_channel_capacity)
    out["cell_channel_capacity_p90"] = percentile(cell_channel_capacity, 90)
    out["cell_channel_capacity_max"] = max(cell_channel_capacity) if cell_channel_capacity else 0.0
    out["spatial_perception_capacity_mean"] = mean(_vals(recs, "spatial_perception_capacity"))
    out["spatial_perception_capacity_p90"] = percentile(_vals(recs, "spatial_perception_capacity"), 90)
    out["sensor_radius_cost_hypothetical_mean"] = mean(radius_hyp_costs)
    out["sensor_radius_cost_hypothetical_total"] = sum(radius_hyp_costs)
    out["sensor_radius_cost_charged_mean"] = mean(radius_charged_costs)
    out["sensor_radius_cost_charged_total"] = sum(radius_charged_costs)
    out["spatial_perception_cost_mean"] = mean(spatial_costs)
    out["spatial_perception_cost_p90"] = percentile(spatial_costs, 90)
    out["spatial_perception_cost_max"] = max(spatial_costs) if spatial_costs else 0.0
    out["perception_cost_total_mean"] = mean(perception_costs)
    out["perception_cost_total_p90"] = percentile(perception_costs, 90)
    out["parent_sensor_radius_mean"] = mean(parent_sensor_radii) if parent_sensor_radii else 0.0
    out["alive_sensor_radius_mean"] = mean(sensor_radii)
    out["sensor_radius_selection_delta_parent_vs_alive"] = out["parent_sensor_radius_mean"] - out["alive_sensor_radius_mean"]
    out["parent_cell_channel_capacity_mean"] = mean(parent_cell_channel) if parent_cell_channel else 0.0
    out["alive_cell_channel_capacity_mean"] = mean(cell_channel_capacity)
    out["cell_channel_capacity_selection_delta_parent_vs_alive"] = (
        out["parent_cell_channel_capacity_mean"] - out["alive_cell_channel_capacity_mean"]
    )
    out["sensor_radius_entropy"] = _entropy([int(x) for x in sensor_radii])
    out["sensor_radius_diversity"] = float(len({int(x) for x in sensor_radii}))
    if sensor_radii:
        radius_counts: dict[int, int] = {}
        for x in sensor_radii:
            radius_counts[int(x)] = radius_counts.get(int(x), 0) + 1
        out["dominant_sensor_radius_fraction"] = float(max(radius_counts.values())) / n
    strategies = [(int(r["sensor_channel_mask"]), int(r["sensor_radius"])) for r in recs]
    out["perception_strategy_entropy"] = _entropy(strategies)
    out["perception_strategy_count_approx"] = float(len(set(strategies)))
    out["memory_prediction_strategy_entropy"] = _entropy(
        [(int(r["memory"]), round(float(r["effective_pred_lr"]), 2)) for r in recs]
    )
    out["perception_prediction_strategy_entropy"] = _entropy(
        [(int(r["sensor_channel_mask"]), int(r["sensor_radius"]), round(float(r["effective_pred_lr"]), 2)) for r in recs]
    )
    out["memory_perception_prediction_strategy_count_approx"] = float(
        len(
            {
                (
                    int(r["memory"]),
                    int(r["sensor_channel_mask"]),
                    int(r["sensor_radius"]),
                    round(float(r["effective_pred_lr"]), 2),
                )
                for r in recs
            }
        )
    )
    mem_mid = percentile(mem, 50)
    perc_mid = percentile(cell_channel_capacity, 50)
    pred_mid = percentile(effective_pred_lr, 50)
    out["high_memory_high_perception_fraction"] = (
        sum(1 for r in recs if float(r["memory"]) >= mem_mid and float(r["cell_channel_capacity"]) >= perc_mid) / n
    )
    out["high_memory_low_perception_fraction"] = (
        sum(1 for r in recs if float(r["memory"]) >= mem_mid and float(r["cell_channel_capacity"]) < perc_mid) / n
    )
    out["low_memory_high_perception_fraction"] = (
        sum(1 for r in recs if float(r["memory"]) < mem_mid and float(r["cell_channel_capacity"]) >= perc_mid) / n
    )
    out["high_memory_high_prediction_fraction"] = (
        sum(1 for r in recs if float(r["memory"]) >= mem_mid and float(r["effective_pred_lr"]) >= pred_mid) / n
    )
    out["high_perception_high_prediction_fraction"] = (
        sum(1 for r in recs if float(r["cell_channel_capacity"]) >= perc_mid and float(r["effective_pred_lr"]) >= pred_mid) / n
    )
    out["high_prediction_low_memory_fraction"] = (
        sum(1 for r in recs if float(r["effective_pred_lr"]) >= pred_mid and float(r["memory"]) < mem_mid) / n
    )
    out["high_prediction_low_perception_fraction"] = (
        sum(1 for r in recs if float(r["effective_pred_lr"]) >= pred_mid and float(r["cell_channel_capacity"]) < perc_mid) / n
    )
    out["sensor_radius_cost_zero_fraction"] = sum(1 for r in recs if float(r["sensor_radius_cost_hyp"]) <= 0.0) / n
    radius_high = percentile(sensor_radii, 90)
    spatial_high = percentile(cell_channel_capacity, 90)
    out["high_radius_low_cost_fraction"] = (
        sum(1 for r in recs if float(r["sensor_radius"]) >= radius_high and float(r["sensor_radius_cost_hyp"]) <= 0.0) / n
    )
    out["high_spatial_perception_low_cost_fraction"] = (
        sum(1 for r in recs if float(r["cell_channel_capacity"]) >= spatial_high and float(r["perception_cost_hyp"]) <= 0.0) / n
    )
    out["sensor_radius_enabled_without_cost_fraction"] = (
        sum(1 for r in recs if float(r["sensor_radius"]) > 0.0 and float(r["sensor_radius_cost_hyp"]) <= 0.0) / n
    )
    out["radius_growth_without_pressure_flag"] = 0.0
    out["memory_growth_without_pressure_flag"] = 0.0
    out["sensor_radius_ablation_active_flag"] = 1.0 if any(bool(r["sensor_radius_ablate"]) for r in recs) else 0.0
    out["sensor_radius_force_min_active_flag"] = 1.0 if any(bool(r["sensor_radius_force_min"]) for r in recs) else 0.0
    out["sensor_radius_force_max_active_flag"] = 1.0 if any(bool(r["sensor_radius_force_max"]) for r in recs) else 0.0
    out["sensor_radius_force_default_active_flag"] = 1.0 if any(bool(r["sensor_radius_force_default"]) for r in recs) else 0.0
    if sim_cfg is not None and bool(getattr(sim_cfg, "trait_capacity_saturation_reporting_enable", False)):
        summary_json, lineage_json, structural_json = trait_capacity_saturation_payloads(
            recs,
            parent_recs,
            generation=generation,
            env_cfg=env_cfg,
            sim_cfg=sim_cfg,
            structural_context=escap8_structural_context,
        )
        out["trait_capacity_saturation_summary_json"] = summary_json
        out["trait_capacity_saturation_lineage_json"] = lineage_json
        out["trait_capacity_saturation_structural_payload_json"] = structural_json
    if structural_live_semantics_enabled(sim_cfg):
        if structural_live_semantics_summary_json is None:
            raise ValueError(
                "canonical cognitive-status records require a precomputed "
                "structural_live_semantics_summary_json when structural live semantics are enabled"
            )
        out["structural_live_semantics_summary_json"] = str(structural_live_semantics_summary_json)
    return out


def _structural_live_semantics_summary(agents: list[Any], *, sim_cfg: Any) -> dict[str, Any]:
    summaries = []
    for agent in agents:
        genome = getattr(agent, "genome", None)
        if genome is None or not hasattr(genome, "resolve_developmental_expression"):
            continue
        stats = getattr(agent, "stats", None)
        age_steps = int(getattr(stats, "survival_steps", getattr(agent, "age", 0)) or 0)
        expr = genome.resolve_developmental_expression(age_steps=age_steps)
        _expr, summary = apply_live_structural_semantics(genome=genome, expression=expr, sim_cfg=sim_cfg)
        summaries.append(summary)
    if not summaries:
        return {
            "schema_version": "structural_live_semantics_v1",
            "enabled": bool(structural_live_semantics_enabled(sim_cfg)),
            "profile": str(getattr(sim_cfg, "structural_live_semantics_profile", "none") or "none"),
            "configured_effect_scale": float(getattr(sim_cfg, "structural_live_effect_scale", 1.0)),
            "agent_count": 0,
            "graph_supported_count": 0,
            "changed_effective_agent_count": 0,
            "changed_effective_fields": [],
            "live_connection_change_count": 0,
            "live_expression_change_count": 0,
            "stored_only_change_count": 0,
            "unsupported_graph_count": 0,
            "nonfinite_failure_count": 0,
            "graph_digests": [],
            "clipping_count_by_field": {},
            "saturation_count_by_field": {},
            "residual_before_scale_sum_by_field": {},
            "residual_after_scale_sum_by_field": {},
        }
    residual_before = {}
    residual_after = {}
    clipping = {}
    saturation = {}
    for summary in summaries:
        for field, value in dict(summary.get("structural_residual_before_scale", {}) or {}).items():
            residual_before[str(field)] = float(residual_before.get(str(field), 0.0) or 0.0) + float(value or 0.0)
        for field, value in dict(summary.get("structural_residual_after_scale", {}) or {}).items():
            residual_after[str(field)] = float(residual_after.get(str(field), 0.0) or 0.0) + float(value or 0.0)
        for field in list(summary.get("clipped_fields", []) or []):
            clipping[str(field)] = int(clipping.get(str(field), 0) or 0) + 1
        for field in list(summary.get("saturated_fields", []) or []):
            saturation[str(field)] = int(saturation.get(str(field), 0) or 0) + 1
    return {
        "schema_version": "structural_live_semantics_v1",
        "enabled": True,
        "profile": str(getattr(sim_cfg, "structural_live_semantics_profile", "none") or "none"),
        "configured_effect_scale": float(getattr(sim_cfg, "structural_live_effect_scale", 1.0)),
        "agent_count": int(len(summaries)),
        "graph_supported_count": int(sum(1 for s in summaries if bool(s.get("graph_supported", False)))),
        "changed_effective_agent_count": int(sum(1 for s in summaries if list(s.get("changed_effective_fields", []) or []))),
        "changed_effective_fields": sorted({str(field) for s in summaries for field in list(s.get("changed_effective_fields", []) or [])}),
        "live_connection_change_count": int(sum(int(s.get("live_connection_change_count", 0) or 0) for s in summaries)),
        "live_expression_change_count": int(sum(int(s.get("live_expression_change_count", 0) or 0) for s in summaries)),
        "stored_only_change_count": int(sum(int(s.get("stored_only_change_count", 0) or 0) for s in summaries)),
        "unsupported_graph_count": int(sum(int(s.get("unsupported_graph_count", 0) or 0) for s in summaries)),
        "nonfinite_failure_count": int(sum(int(s.get("nonfinite_failure_count", 0) or 0) for s in summaries)),
        "graph_digests": sorted({str(s.get("graph_digest", "")) for s in summaries if s.get("graph_digest")}),
        "clipping_count_by_field": dict(sorted(clipping.items())),
        "saturation_count_by_field": dict(sorted(saturation.items())),
        "residual_before_scale_sum_by_field": dict(sorted(residual_before.items())),
        "residual_after_scale_sum_by_field": dict(sorted(residual_after.items())),
    }


def _direct_internal_pressure_present(sim_cfg: Any) -> bool:
    if sim_cfg is None:
        return True
    prediction_names = (
        "coherence_pred_err_weight",
        "coherence_episode0_pred_err_weight",
        "episode0_pred_err_fitness_weight",
        "late_pred_cost_enable",
        "late_coherence_eff_enable",
        "adult_move_penalty_enable",
    )
    names = list(prediction_names)
    if bool(getattr(sim_cfg, "disable_prediction_error_selection_pressure", False)):
        names = []
    names.append("identity_drift_enable")
    for name in names:
        try:
            v = getattr(sim_cfg, name)
        except Exception:
            continue
        if isinstance(v, bool):
            if v:
                return True
        else:
            try:
                if float(v) != 0.0:
                    return True
            except Exception:
                pass
    return False
