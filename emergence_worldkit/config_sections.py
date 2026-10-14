"""Canonical data-only configuration sections for WorldKit.

Legacy modules under engine/env re-export these classes for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnvBiomeConfig:
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

@dataclass
class EnvEnergyEconomyConfig:
    # Energy economy
    # -----------------------------
    start_energy: float = 60.0
    max_energy: float = 160.0

    move_cost: float = 0.20
    rest_cost: float = 0.05
    gather_cost: float = 0.08
    attack_cost: float = 0.60

    food_energy: float = 20.0

@dataclass
class EnvGruCostConfig:
    # GRU / memory upkeep cost
    # -----------------------------
    gru_cost_base: float = 0.004
    gru_cost_per_dim: float = 0.00008

@dataclass
class EnvHealthCombatConfig:
    # Health / combat
    # -----------------------------
    start_health: float = 12.0

    # >>> FIX: combat was silently disabled because this flag did not exist
    enable_combat: bool = True

    attack_damage: float = 0.75

@dataclass
class EnvInjuryConfig:
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

@dataclass
class EnvReproductionConfig:
    # Reproduction
    # -----------------------------
    reproduce_requires_mate: bool = True
    reproduce_parent_energy_share: float = 0.20
    reproduce_min_energy_buffer: float = 20.0
    reproduce_cooldown_steps: int = 35
    reproduce_try_prob: float = 0.15

@dataclass
class EnvScarConfig:
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

@dataclass
class EnvTier1SpatialEcologyConfig:
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

@dataclass
class EnvTier2SpatialDroughtsConfig:
    # Tier 2: Spatial Droughts (time-varying regional stress)
    # -----------------------------
    # Defaults preserve current behavior when disabled.
    spatial_droughts_enable: bool = False
    # Spec is a dict matching the Tier 2 schema. It is parsed from JSON via CLI or experiment registry.
    spatial_droughts_spec: dict = field(default_factory=dict)

    # Logging frequencies (steps). Only used when droughts are enabled.
    spatial_droughts_log_timeseries_every: int = 1
    spatial_droughts_log_periodic_every: int = 100

@dataclass
class EnvTier3MigrationPressureConfig:
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

@dataclass
class EnvWorldConfig:
    # World
    # -----------------------------
    width: int = 32
    height: int = 32
    view_radius: int = 2

    food_spawn_per_step: int = 28
    max_food: int = 500

# Keep the established import addresses used by existing pickles.
EnvBiomeConfig.__module__ = "env.env_config_biomes"
EnvEnergyEconomyConfig.__module__ = "env.env_config_energy_economy"
EnvGruCostConfig.__module__ = "env.env_config_gru_cost"
EnvHealthCombatConfig.__module__ = "env.env_config_health_combat"
EnvInjuryConfig.__module__ = "env.env_config_injury"
EnvReproductionConfig.__module__ = "env.env_config_reproduction"
EnvScarConfig.__module__ = "env.env_config_scar"
EnvTier1SpatialEcologyConfig.__module__ = "env.env_config_tier1_spatial_ecology"
EnvTier2SpatialDroughtsConfig.__module__ = "env.env_config_tier2_spatial_droughts"
EnvTier3MigrationPressureConfig.__module__ = "env.env_config_tier3_migration_pressure"
EnvWorldConfig.__module__ = "env.env_config_world"

__all__ = [
    "EnvBiomeConfig",
    "EnvEnergyEconomyConfig",
    "EnvGruCostConfig",
    "EnvHealthCombatConfig",
    "EnvInjuryConfig",
    "EnvReproductionConfig",
    "EnvScarConfig",
    "EnvTier1SpatialEcologyConfig",
    "EnvTier2SpatialDroughtsConfig",
    "EnvTier3MigrationPressureConfig",
    "EnvWorldConfig",
]
