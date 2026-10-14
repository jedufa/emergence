"""env/config.py

Central configuration dataclass for the GridWorld environment.

This file defines environment knobs only (grid size, energy economy, physiology, etc.).
Simulation scheduling knobs (generations, shock timing) live in SimConfig.

Path A additions:
- Biomes (spatial niches): SAFE vs WILD zones on the grid.
- Biome parameters: food spawn bias, action cost multipliers, ambient hazard in WILD.

Defaults are conservative to avoid destabilizing existing runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    # -----------------------------
    # World
    # -----------------------------
    width: int = 32
    height: int = 32
    view_radius: int = 2

    food_spawn_per_step: int = 28
    max_food: int = 500

    # -----------------------------
    # Patch World / Patch-Jungle V0 Step 1-13
    # -----------------------------
    # Patch World is the future persistent ecological substrate. Step 1 added
    # the disabled feature gate and schema version placeholder; Step 2 added
    # passive substrate arrays; Step 3 activates food amount depletion/regrowth;
    # Step 4 adds deterministic terrain affordances; Step 5 adds activity traces;
    # Step 6 adds separately gated local patch observations; Step 7 adds
    # diagnostics; Step 8 adds gated waste/crowding effects on food recovery;
    # Step 9 adds separately gated Jungle predator-like pressure; Step 10 adds
    # separately gated reproduction emissions and juvenile predator context;
    # Step 11 adds separately gated terrain survival affordances; Step 12 adds
    # separately gated ecological recovery and bounded regrowth boosts; Step
    # 13 adds richer diagnostic-only lineage diversity and strategy reporting.
    patch_world_enable: bool = False
    patch_world_schema_version: int = 1
    patch_food_amount_max: float = 1.0
    patch_food_amount_initial_food_cell_value: float = 1.0
    patch_food_amount_initial_empty_cell_value: float = 0.0
    patch_food_presence_threshold: float = 0.5
    patch_food_gather_amount: float = 1.0
    patch_food_regrow_per_step: float = 0.02
    patch_food_regrow_use_fertility: bool = True
    patch_food_regrow_cap_by_fertility: bool = False
    respec0b_resource_specialization_enable: bool = False
    respec0b_resource_class_mode: str = "vertical_split_ab"
    respec0b_resource_a_handling_threshold: float = 1.50
    respec0b_resource_b_handling_threshold: float = 1.75
    respec0b_resource_a_return: float = 0.75
    respec0b_resource_b_return: float = 1.50
    respec0b_resource_equalized_sham_enable: bool = False
    respec_n1_zone_telemetry_enable: bool = False
    respec_n2_capacity_balanced_mixed_enable: bool = False
    respec_n2_capacity_balanced_formula_version: str = "energy_return_regrowth_balance_v1"
    respec_n2_capacity_balanced_a_fertility: float = 1.0
    respec_n2_capacity_balanced_b_fertility: float = 0.5
    # EED-0 reciprocal-impact overlay. Production selection is explicit and
    # these defaults keep both its state and applied consequences disabled.
    eed_resource_impact_enable: bool = False
    eed_resource_impact_apply_enable: bool = False
    eed_ecology_frozen_enable: bool = False
    eed_phenotype_fixed_enable: bool = False
    eed_phenotype_fixed_axis_profile: str = "neutral"
    eed_resource_a_initial_amount: float = 1.0
    eed_resource_b_initial_amount: float = 1.0
    eed_resource_a_capacity: float = 1.0
    eed_resource_b_capacity: float = 1.0
    eed_resource_a_renewal_per_step: float = 0.25
    eed_resource_b_renewal_per_step: float = 0.25
    eed_resource_impact_per_success: float = 0.25
    eed_resource_impact_decay_per_step: float = 0.25
    eed_resource_impact_max_load: float = 1.0
    mosaic0a_depletion_recovery_enable: bool = False
    mosaic0a_schema_version: int = 1
    mosaic0a_class_mode: str = "budget_matched_clustered_v1"
    mosaic0a_control_mode: str = "real_mosaic"
    mosaic0a_cluster_size: int = 4
    mosaic0a_budget_horizon_steps: int = 50
    mosaic0a_burst_initial_food: float = 1.0
    mosaic0a_renewal_initial_food: float = 0.4
    mosaic0a_burst_fertility: float = 0.4
    mosaic0a_renewal_fertility: float = 1.0
    mosaic0a_burst_movement_cost: float = 1.25
    mosaic0a_renewal_movement_cost: float = 0.85
    mosaic0a_burst_waste_initial: float = 0.20
    mosaic0a_renewal_waste_initial: float = 0.0
    mosaic0a_burst_recovery_activity: float = 0.0
    mosaic0a_renewal_recovery_activity: float = 0.45
    predrisk_fixed1_enable: bool = False
    predrisk_fixed1_schema_version: int = 1
    predrisk_fixed1_zone_mode: str = "clustered_quadrants_v1"
    predrisk_fixed1_control_mode: str = "correlated_tradeoff"
    predrisk_fixed1_cluster_size: int = 4
    predrisk_fixed1_budget_horizon_steps: int = 80
    predrisk_fixed1_exposed_food: float = 1.0
    predrisk_fixed1_safe_food: float = 0.45
    predrisk_fixed1_exposed_fertility: float = 0.85
    predrisk_fixed1_safe_fertility: float = 0.55
    predrisk_fixed1_exposed_cover: float = 0.05
    predrisk_fixed1_safe_cover: float = 0.85
    predrisk_fixed1_exposed_shelter: float = 0.05
    predrisk_fixed1_safe_shelter: float = 0.80
    predrisk_fixed1_exposed_movement_cost: float = 1.15
    predrisk_fixed1_safe_movement_cost: float = 0.90
    predrisk_fixed1_exposed_scent: float = 0.75
    predrisk_fixed1_safe_scent: float = 0.05
    predrisk_fixed1_exposed_waste: float = 0.25
    predrisk_fixed1_safe_waste: float = 0.05
    corridor_cp0_enable: bool = False
    corridor_cp0_schema_version: int = 1
    corridor_cp0_zone_mode: str = "left_home_mid_corridor_right_remote_v1"
    corridor_cp0_control_mode: str = "central_place_tradeoff"
    corridor_cp0_home_width: int = 0
    corridor_cp0_remote_width: int = 0
    corridor_cp0_corridor_half_width: int = 1
    corridor_cp0_home_food: float = 0.30
    corridor_cp0_corridor_food: float = 0.05
    corridor_cp0_remote_food: float = 1.00
    corridor_cp0_home_fertility: float = 0.45
    corridor_cp0_corridor_fertility: float = 0.20
    corridor_cp0_remote_fertility: float = 1.00
    corridor_cp0_home_cover: float = 0.85
    corridor_cp0_corridor_cover: float = 0.15
    corridor_cp0_remote_cover: float = 0.10
    corridor_cp0_home_shelter: float = 0.85
    corridor_cp0_corridor_shelter: float = 0.10
    corridor_cp0_remote_shelter: float = 0.05
    corridor_cp0_home_movement_cost: float = 0.75
    corridor_cp0_corridor_movement_cost: float = 1.80
    corridor_cp0_remote_movement_cost: float = 1.20
    corridor_cp0_home_scent: float = 0.05
    corridor_cp0_corridor_scent: float = 0.15
    corridor_cp0_remote_scent: float = 0.30
    corridor_cp0_home_waste: float = 0.02
    corridor_cp0_corridor_waste: float = 0.10
    corridor_cp0_remote_waste: float = 0.15
    corridor_hc1_homing_cue_enable: bool = False
    corridor_hc1_homing_cue_schema_version: int = 1
    corridor_hc1_homing_cue_field: str = "shelter"
    corridor_hc1_home_interior_value: float = 0.95
    corridor_hc1_home_edge_value: float = 0.72
    corridor_hc1_corridor_near_home_value: float = 0.58
    corridor_hc1_corridor_far_home_value: float = 0.18
    corridor_hc1_remote_value: float = 0.05
    habacclim_enable: bool = False
    habacclim_schema_version: int = 1
    habacclim_habitat_mode: str = "interleaved_bands_v1"
    habacclim_band_width: int = 4
    habacclim_control_mode: str = "acclimation_tradeoff"
    habacclim_observation_enable: bool = False
    habacclim_observation_schema_version: int = 1
    habacclim_observation_scalar_name: str = "habitat_acclimation_state_norm"
    habacclim_acclimation_n90_ticks: int = 12
    habacclim_processing_penalty_max: float = 0.35
    habacclim_a_food: float = 1.0
    habacclim_a_fertility: float = 0.35
    habacclim_a_cover: float = 0.08
    habacclim_a_shelter: float = 0.05
    habacclim_a_movement_cost: float = 1.10
    habacclim_a_scent: float = 0.70
    habacclim_a_waste: float = 0.25
    habacclim_b_food: float = 0.45
    habacclim_b_fertility: float = 1.0
    habacclim_b_cover: float = 0.80
    habacclim_b_shelter: float = 0.85
    habacclim_b_movement_cost: float = 0.90
    habacclim_b_scent: float = 0.08
    habacclim_b_waste: float = 0.05
    patch_fertility_default: float = 1.0
    patch_fertility_min: float = 0.0
    patch_fertility_max: float = 1.0
    patch_cover_default: float = 0.0
    patch_cover_min: float = 0.0
    patch_cover_max: float = 1.0
    patch_shelter_default: float = 0.0
    patch_shelter_min: float = 0.0
    patch_shelter_max: float = 1.0
    patch_shelter_from_cover_enable: bool = False
    patch_visibility_default: float = 1.0
    patch_visibility_from_cover_enable: bool = True
    patch_visibility_min: float = 0.0
    patch_visibility_max: float = 1.0
    patch_movement_cost_default: float = 1.0
    patch_movement_cost_min: float = 0.0
    patch_movement_cost_max: float = 4.0
    patch_movement_cost_apply_enable: bool = False
    patch_movement_cost_energy_scale: float = 1.0
    patch_scent_initial: float = 0.0
    patch_scent_max: float = 1.0
    patch_scent_decay_per_step: float = 0.05
    patch_scent_presence_emit: float = 0.01
    patch_scent_move_emit: float = 0.02
    patch_scent_gather_emit: float = 0.05
    patch_scent_birth_emit: float = 0.0
    patch_scent_death_emit: float = 0.0
    patch_waste_initial: float = 0.0
    patch_waste_max: float = 1.0
    patch_waste_decay_per_step: float = 0.02
    patch_waste_presence_emit: float = 0.005
    patch_waste_consume_emit: float = 0.05
    patch_waste_birth_emit: float = 0.0
    patch_waste_death_emit: float = 0.0
    patch_waste_density_emit_scale: float = 0.0
    patch_density_radius: int = 1
    patch_density_include_self: bool = True
    patch_density_normalize: bool = False
    patch_world_observation_enable: bool = False
    patch_world_observation_schema_version: int = 1
    patch_world_observation_stacked_gather_enable: bool = False
    patch_world_observation_numba_gather_enable: bool = False
    patch_world_diagnostics_enable: bool = True
    patch_world_lineage_descriptors_enable: bool = True
    patch_world_descriptor_schema_version: int = 1
    roz2a_patch_niche_enable: bool = False
    roz2a_patch_niche_mode: str = "static_three_zone_tradeoff_v1"
    roz2a_patch_niche_radius: int = 2
    roz2a_patch_niche_resource_fertility: float = 1.0
    roz2a_patch_niche_resource_movement_cost: float = 1.6
    roz2a_patch_niche_resource_waste: float = 0.20
    roz2a_patch_niche_shelter_food_multiplier: float = 0.35
    roz2a_patch_niche_shelter_fertility: float = 0.35
    roz2a_patch_niche_shelter_cover: float = 0.75
    roz2a_patch_niche_shelter_shelter: float = 0.75
    roz2a_patch_niche_shelter_movement_cost: float = 0.80
    roz2a_patch_niche_shelter_recovery: float = 0.45
    roz2a_patch_niche_transition_fertility: float = 0.65
    roz2a_patch_niche_transition_cover: float = 0.30
    roz2a_patch_niche_transition_shelter: float = 0.25
    patch_world_waste_consequence_enable: bool = False
    patch_waste_regrowth_penalty_enable: bool = True
    patch_waste_regrowth_penalty_scale: float = 0.5
    patch_waste_regrowth_penalty_max: float = 0.5
    patch_density_regrowth_penalty_enable: bool = True
    patch_density_regrowth_penalty_scale: float = 0.5
    patch_density_regrowth_penalty_max: float = 0.5
    patch_regrowth_min_multiplier: float = 0.1
    patch_waste_fertility_effect_enable: bool = False
    patch_waste_fertility_damage_rate: float = 0.0
    patch_fertility_recovery_rate: float = 0.0
    patch_fertility_recovery_target: float = 1.0
    jungle_predator_pressure_enable: bool = False
    jungle_predator_base_risk: float = 0.0
    jungle_predator_scent_weight: float = 0.02
    jungle_predator_waste_weight: float = 0.02
    jungle_predator_density_weight: float = 0.02
    jungle_predator_visibility_weight: float = 0.02
    jungle_predator_cover_protection_weight: float = 0.02
    jungle_predator_shelter_protection_weight: float = 0.02
    jungle_predator_risk_min: float = 0.0
    jungle_predator_risk_cap: float = 0.05
    jungle_predator_health_damage: float = 0.25
    jungle_reproduction_emissions_enable: bool = False
    jungle_birth_scent_emit: float = 0.05
    jungle_birth_waste_emit: float = 0.05
    jungle_birth_emission_at_child_cell_enable: bool = True
    jungle_birth_emission_clamp_enable: bool = True
    jungle_juvenile_predator_vulnerability_enable: bool = False
    jungle_juvenile_age_threshold: int = 10
    jungle_juvenile_predator_damage_multiplier: float = 1.5
    jungle_juvenile_vulnerability_cap: float = 3.0
    jungle_terrain_protection_enable: bool = False
    jungle_terrain_protection_apply_to_damage_enable: bool = True
    jungle_cover_damage_reduction_weight: float = 0.25
    jungle_shelter_damage_reduction_weight: float = 0.25
    jungle_terrain_damage_multiplier_min: float = 0.25
    jungle_ecological_recovery_enable: bool = False
    jungle_recovery_activity_initial: float = 0.0
    jungle_recovery_activity_min: float = 0.0
    jungle_recovery_activity_max: float = 1.0
    jungle_recovery_activity_decay_per_step: float = 0.02
    jungle_recovery_activity_base_growth: float = 0.0
    jungle_recovery_activity_cover_weight: float = 0.02
    jungle_recovery_activity_fertility_weight: float = 0.02
    jungle_recovery_activity_waste_penalty_weight: float = 0.05
    jungle_recovery_activity_density_penalty_weight: float = 0.05
    jungle_recovery_activity_scent_penalty_weight: float = 0.02
    jungle_recovery_regrowth_boost_enable: bool = True
    jungle_recovery_regrowth_boost_scale: float = 0.1
    jungle_recovery_regrowth_boost_max: float = 0.25
    jungle_recovery_fertility_enable: bool = False
    jungle_resource_spread_enable: bool = False
    lineage_diversity_reporting_enable: bool = True
    lineage_strategy_descriptors_enable: bool = True
    lineage_branch_candidate_reporting_enable: bool = True
    lineage_diversity_schema_version: int = 1
    lineage_dominance_top1_warning_threshold: float = 0.75
    lineage_dominance_top3_warning_threshold: float = 0.90
    lineage_entropy_warning_threshold: float = 0.50
    lineage_low_active_count_warning_threshold: int = 2
    lineage_min_alive_count_for_descriptor: int = 1
    lineage_branch_candidate_top_n: int = 5
    lineage_branch_candidate_min_alive_count: int = 1
    lineage_branch_candidate_min_age: int = 0
    wdl_delayed_hazard_enable: bool = False
    wdl_delayed_hazard_schema_version: int = 1
    wdl_delayed_hazard_cue_enable: bool = True
    wdl_delayed_hazard_observation_enable: bool = False
    wdl_delayed_hazard_observation_dropout_rate: float = 0.0
    wdl_delayed_hazard_delay_steps: int = 8
    wdl_delayed_hazard_cue_duration_steps: int = 4
    wdl_delayed_hazard_active_duration_steps: int = 6
    wdl_delayed_hazard_event_period_steps: int = 20
    wdl_delayed_hazard_radius: int = 1
    wdl_delayed_hazard_intensity: float = 0.10
    wdl_delayed_hazard_damage_enable: bool = True
    wdl_delayed_hazard_regrowth_suppression_enable: bool = False
    wdl_delayed_hazard_deterministic_schedule_enable: bool = True
    path1_drh0_reporting_enable: bool = False
    path1_drh0_schema_version: int = 1
    path1_drh0_refuge_reachability_enable: bool = False
    path1_drh0_observation_enable: bool = False
    path1_drh0_refuge_radius: int = 3
    path1_drh0_refuge_requires_shelter_or_cover: bool = True
    path1_drh0_min_refuge_cells: int = 1
    path1_drh0_consequence_enable: bool = False
    path1_drh0_refuge_damage_reduction_enable: bool = False
    path1_drh0_static_refuge_control_enable: bool = False
    path1_drh0_cue_required_for_refuge_protection: bool = True
    path1_drh0_pre_arrival_required_for_refuge_protection: bool = True
    path1_drh0_damage_reduction_fraction: float = 0.50
    path1_drh0_damage_reduction_cap: float = 1.0
    prediction_demand_enable: bool = False
    prediction_demand_schema_version: int = 1
    prediction_demand_observation_enable: bool = False
    prediction_demand_cycle_mode: str = "variable_delay_resource_wave"
    prediction_demand_delay_min: int = 4
    prediction_demand_delay_max: int = 10
    prediction_demand_cue_noise: float = 0.25
    prediction_demand_transition_strength: float = 0.35
    prediction_demand_active_duration: int = 4
    prediction_demand_event_period_steps: int = 18
    prediction_demand_radius: int = 1
    prediction_demand_resource_consequence_enable: bool = True
    prediction_demand_deterministic_schedule_enable: bool = True
    delayed_resource_depletion_enable: bool = False
    delayed_resource_depletion_schema_version: int = 1
    delayed_resource_depletion_observation_enable: bool = False
    delayed_resource_depletion_observation_dropout_rate: float = 0.0
    delayed_resource_depletion_delay_steps: int = 8
    delayed_resource_depletion_cue_duration_steps: int = 4
    delayed_resource_depletion_active_duration_steps: int = 6
    delayed_resource_depletion_event_period_steps: int = 20
    delayed_resource_depletion_radius: int = 1
    delayed_resource_depletion_food_depletion_fraction: float = 0.50
    delayed_resource_depletion_regrowth_multiplier: float = 0.25
    delayed_resource_depletion_deterministic_schedule_enable: bool = True
    delayed_crowding_waste_enable: bool = False
    delayed_crowding_waste_schema_version: int = 1
    delayed_crowding_waste_observation_enable: bool = False
    delayed_crowding_waste_observation_dropout_rate: float = 0.0
    delayed_crowding_waste_delay_steps: int = 8
    delayed_crowding_waste_cue_duration_steps: int = 4
    delayed_crowding_waste_active_duration_steps: int = 6
    delayed_crowding_waste_event_period_steps: int = 20
    delayed_crowding_waste_radius: int = 1
    delayed_crowding_waste_min_waste_trigger: float = 0.01
    delayed_crowding_waste_waste_add: float = 0.10
    delayed_crowding_waste_regrowth_multiplier: float = 0.35
    delayed_crowding_waste_deterministic_schedule_enable: bool = True
    mem_max_delayed_choice_enable: bool = False
    mem_max_delayed_choice_schema_version: int = 1
    mem_max_delayed_choice_observation_enable: bool = False
    mem_max_delayed_choice_observation_schema_version: int = 1
    mem_max_delayed_choice_global_cue_observation_enable: bool = False
    mem_max_delayed_choice_navigation_observation_enable: bool = False
    mem_max_delayed_choice_cue_enable: bool = True
    mem_max_delayed_choice_cue_steps: int = 2
    mem_max_delayed_choice_hidden_steps: int = 4
    mem_max_delayed_choice_active_steps: int = 3
    mem_max_delayed_choice_event_period_steps: int = 16
    mem_max_delayed_choice_target_geometry: str = "horizontal_symmetric"
    mem_max_delayed_choice_target_distance: int = 2
    mem_max_delayed_choice_damage: float = 0.25
    mem_max_delayed_choice_health_damage: float = 0.25
    mem_max_delayed_choice_energy_cost: float = 0.0
    mem_max_delayed_choice_correct_choice_damage_multiplier: float = 0.0
    mem_max_delayed_choice_incorrect_choice_damage_multiplier: float = 1.0
    mem_max_delayed_choice_no_choice_damage_multiplier: float = 1.0
    mem_max_delayed_choice_consequence_once_per_event: bool = True
    mem_max_delayed_choice_consequence_enable: bool = True
    mem_max_delayed_choice_shortcut_truth_enable: bool = False
    mem_max_delayed_choice_random_incorrect_cue_enable: bool = False
    mem_max_delayed_choice_deterministic_schedule_enable: bool = True

    # -----------------------------
    # Tier 1: Spatial Ecology (static heterogeneity)
    # -----------------------------
    # NOTE: These knobs are world-level only (functions of position).
    # They are generated once at run start, seeded, and logged.
    # Defaults preserve current behavior when spatial ecology is disabled.
    spatial_ecology_enable: bool = False
    spatial_ecology_seed: int = 0

    # MapSpec v1: dict objects with keys:
    #   name, version, type, params, seed
    # See env/gridworld_pkg/spatial_ecology.py for details.
    resource_regrowth_map_spec: dict = field(default_factory=lambda: {
        "name": "resource_regrowth_map",
        "version": 1,
        "type": "uniform",
        "params": {"value": 1.0},
        "seed": 0,
    })
    resource_spawn_density_map_spec: dict = field(default_factory=lambda: {
        "name": "resource_spawn_density_map",
        "version": 1,
        "type": "uniform",
        "params": {"value": 1.0},
        "seed": 0,
    })
    hazard_rate_map_spec: dict = field(default_factory=lambda: {
        "name": "hazard_rate_map",
        "version": 1,
        "type": "uniform",
        "params": {"value": 1.0},
        "seed": 0,
    })

    # Logging controls (maps/ folder inside run_dir)
    # Supported: "npy", "png", "csv" (csv is summary stats only)
    spatial_ecology_log_formats: list[str] = field(default_factory=lambda: ["npy", "png"])
    spatial_ecology_log_stats: bool = True
    spatial_ecology_log_region_id_map: bool = False

    # -----------------------------
    # Tier 2: Spatial Droughts (time-varying regional stress)
    # -----------------------------
    # Defaults preserve current behavior when disabled.
    spatial_droughts_enable: bool = False
    # Spec is a dict matching the Tier 2 schema. It is parsed from JSON via CLI or experiment registry.
    spatial_droughts_spec: dict = field(default_factory=dict)

    # Logging frequencies (steps). Only used when droughts are enabled.
    spatial_droughts_log_timeseries_every: int = 1
    spatial_droughts_log_periodic_every: int = 100





    # -----------------------------
    # Tier 3: Migration Pressure (movement + crowding + border risk)
    # -----------------------------
    # Defaults preserve current behavior when disabled.
    # === Tier 3 change: config schema ===
    migration_pressure_enable: bool = False

    # Tier 3 regions are a PRIMARY partition (each cell has exactly one region_id).
    # Supported modes (v1): "grid" (deterministic tiling), "file" (load .npy grid).
    tier3_regions_spec: dict = field(default_factory=lambda: {
        "mode": "grid",
        "grid": {"region_rows": 1, "region_cols": 1},
        "file": {"region_grid_path": None},
    })

    # Regional carrying capacity (crowding pressure)
    tier3_regional_carrying_capacity: dict = field(default_factory=lambda: {
        "enable": False,
        "capacity_by_region": [],
        "ema_gamma": 1.0,          # 1.0 = no smoothing
        "scarcity_alpha": 0.0,     # CC1: consumption yield pressure (off by default)
        "yield_floor": 0.0,
        "hazard_beta": 0.0,        # CC2: hazard-rate pressure (off by default)
        "hazard_mult_cap": None,
    })

    # Movement cost map (applies ONLY on successful moves; destination-region lookup)
    tier3_movement_cost_map: dict = field(default_factory=lambda: {
        "enable": False,
        "base_move_cost": 1.0,
        "cost_by_region": [],
        "apply_on": "successful_move",
        "lookup": "destination_region",
    })

    # Region transition risk (border crossing hazard channel)
    tier3_region_transition_risk: dict = field(default_factory=lambda: {
        "enable": False,
        "crossing_risk_base": 0.0,
        "crossing_damage": 0.1,
        "transition_risk_matrix": None,
    })

    # Validation-only: force random walk actions (plumbing check, not for real evolution runs)
    tier3_validation_force_random_walk: bool = False
    # === Tier 3 change: end config schema ===

    # -----------------------------
    # Tier 4: Juvenile spatial vulnerability (life-history pressure without hard age bans)
    # -----------------------------
    # Defaults preserve current behavior when disabled.
    tier4_juvenile_vulnerability_enable: bool = False

    # Juvenile window semantics:
    # - age_steps := AgentStats.survival_steps (steps survived since spawn)
    # - is_juvenile := (age_steps < tier4_juvenile_steps)
    tier4_juvenile_steps: int = 0

    # Optional stabilization: newborn grace window.
    # During grace, juvenile hazard amplification is NOT applied (movement penalty can still apply).
    tier4_juvenile_grace_steps: int = 0

    # Multipliers (clamped at env init; see env/gridworld_pkg/state.py)
    tier4_juvenile_hazard_multiplier: float = 1.0   # clamp [1.0, 10.0]
    tier4_juvenile_movement_penalty: float = 1.0    # clamp [1.0, 10.0]
    tier4_juvenile_harvest_multiplier: float = 1.0  # clamp [0.0, 1.0] (penalty only)

    # Survival curve bin edges in steps (used for JSON histograms in metrics).
    tier4_survival_bins_steps: list[int] = field(default_factory=lambda: [0, 10, 25, 50, 80, 120, 200])



# -----------------------------
    # Biomes (spatial niches)
    # -----------------------------
    # If enabled, the grid is partitioned into SAFE and WILD cells.
    # SAFE: cheaper to live, more food.
    # WILD: harsher (higher action costs), less food, plus ambient hazard.
    enable_biomes: bool = True

    # Fraction of cells that are SAFE (rest are WILD). (0.0..1.0)
    biome_safe_frac: float = 0.6

    # Patchiness control: number of smoothing passes to create contiguous regions.
    # 0 = pure random noise, 3-6 = recognizable patches.
    biome_smooth_passes: int = 6

    # Food spawn bias by biome (multiplies food_spawn_per_step locally).
    biome_safe_food_mult: float = 1.25
    biome_wild_food_mult: float = 0.75

    # Additional action cost multiplier applied in WILD biome (on top of injury + shock).
    biome_wild_action_cost_mult: float = 1.10

    # Ambient hazard in WILD: per-step probability of taking damage just for existing there.
    biome_wild_hazard_prob: float = 0.005
    biome_wild_hazard_damage: float = 0.35

    # If injury is enabled, ambient hazard can also add injury.
    biome_wild_hazard_injury_per_damage: float = 0.25

    # -----------------------------
    # Energy economy
    # -----------------------------
    start_energy: float = 60.0
    max_energy: float = 160.0

    move_cost: float = 0.20
    rest_cost: float = 0.05
    gather_cost: float = 0.08
    attack_cost: float = 0.60

    food_energy: float = 20.0

    # -----------------------------
    # Health / combat
    # -----------------------------
    start_health: float = 12.0

    # >>> FIX: combat was silently disabled because this flag did not exist
    enable_combat: bool = True

    attack_damage: float = 0.75

    # -----------------------------
    # Reproduction
    # -----------------------------
    reproduce_requires_mate: bool = True
    reproduce_parent_energy_share: float = 0.20
    reproduce_min_energy_buffer: float = 20.0
    reproduce_cooldown_steps: int = 35
    reproduce_try_prob: float = 0.15

    # -----------------------------
    # Injury
    # -----------------------------
    enable_injury: bool = True
    injury_per_damage: float = 0.27
    injury_max: float = 10.0

    injury_max_energy_penalty: float = 0.03
    injury_action_cost_penalty: float = 0.07

    injury_recover_per_rest_step: float = 0.02
    injury_recover_requires_streak: int = 6

    injury_carry_decay: float = 0.01
    injury_carry_decay_requires_alive: bool = True

    injury_repro_block: float = 2.75
    injury_repro_k_injury: float = 0.35
    injury_repro_k_scar: float = 0.9
    injury_birth_fail_scar: float = 0.12
    injury_birth_fail_injury: float = 0.03

    injury_penalty_curve_power: float = 0.55
    injury_min_energy_frac: float = 0.20

    # -----------------------------
    # Scar
    # -----------------------------
    enable_scar: bool = True
    scar_cost_weight: float = 1.25
    scar_start_threshold: float = 1.5
    scar_gain_per_injury: float = 0.18
    scar_gain_base: float = 0.03
    scar_max: float = 10.0

    scar_recover_requires_streak: int = 12
    scar_recover_per_rest_step: float = 0.002
    # -----------------------------
    # GRU / memory upkeep cost
    # -----------------------------
    gru_cost_base: float = 0.004
    gru_cost_per_dim: float = 0.00008


# Keep the established pickle address; the legacy modules re-export this object.
EnvConfig.__module__ = "env.env_config"
