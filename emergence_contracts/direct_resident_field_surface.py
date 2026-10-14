"""Frozen dependency-free field-name surface for direct-resident persistence.

These tuples describe ordered compatibility field labels shared by Campaign and
the physical runtime.  Labels containing words such as ``generation`` are
opaque state names: this contract grants no time, checkpoint, execution, or
qualification authority.  Absolute native ticks remain the sole time model.
"""

from __future__ import annotations

from typing import Final

from .canonical import canonical_sha256
from .ids import ContentDigest


DIRECT_RESIDENT_FIELD_SURFACE_SCHEMA_VERSION: Final[str] = (
    "eiel_direct_resident_field_surface_v1"
)
DIRECT_RESIDENT_FIELD_SURFACE_TIME_AUTHORITY: Final[bool] = False
DIRECT_RESIDENT_FIELD_SURFACE_QUALIFICATION_AUTHORITY: Final[bool] = False
DIRECT_RESIDENT_FIELD_SURFACE_PERSISTENCE_LABELS_ONLY: Final[bool] = True

PATCH_P2_STATE_FIELDS_V1: Final[tuple[str, ...]] = (
    "patch_food_amount",
    "patch_fertility",
    "patch_scent",
    "patch_waste",
    "patch_scent_emit_pending",
    "patch_waste_emit_pending",
    "patch_density",
    "patch_cover",
    "patch_shelter",
    "patch_movement_cost",
)

ORDINARY_BIRTH_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "slot_to_agent_id",
    "agent_id_to_slot",
    "slot_tokens",
    "alive_mask",
    "pos_y",
    "pos_x",
    "agent_grid",
    "energy",
    "health",
    "injury",
    "scar",
    "age",
    "repro_cooldown",
    "repro_threshold",
    "offspring_count",
    "lineage_founder_id",
    "lineage_depth",
    "lineage_birth_generation",
    "lineage_birth_tick",
    "patch_habitat_lineage_id",
    "policy_energy_observation",
    "survival_steps",
    "rest_streak",
    "action_cache_token",
    "action_cache_cursor",
    "action_cache_next",
    "slot_free_queue",
    "slot_free_queue_count",
    "population_next_id",
    "last_action",
    "last_last_action",
    "action_switches",
    "action_oscillations",
    "episode0_switches",
    "episode0_oscillations",
)

ORDINARY_BIRTH_SCALAR_RNG_FIELDS: Final[tuple[str, ...]] = (
    "python_random_words",
    "python_random_index",
    "python_random_has_gauss",
    "python_random_gauss_next",
    "python_random_gauss_primary_tape",
    "python_random_gauss_secondary_tape",
    "python_random_raw_draws",
    "mutation_py_semantic_draws",
)

ORDINARY_BIRTH_GENOME_INT32_FIELDS: Final[tuple[str, ...]] = (
    "genome_hidden_size",
    "genome_memory_dim",
    "genome_sensor_channel_mask",
    "genome_sensor_radius",
    "genome_introspection_channel_mask",
    "genome_self_regulation_control_mask",
    "genome_self_modification_target_mask",
    "genome_directed_inheritance_channel_mask",
)

ORDINARY_BIRTH_GENOME_FLOAT32_FIELDS: Final[tuple[str, ...]] = (
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
    "repro_threshold",
    "gru_upkeep_cost",
    "policy_attack_bias",
    "policy_action_temperature",
    "policy_drive_logit_gain",
    "drive_inertia",
    "drive_noise",
    "pred_lr",
)

ORDINARY_BIRTH_DRIVE_FIELDS: Final[tuple[str, ...]] = (
    "drives",
    "birth_reset_shock",
    "population_pcg64_state_words",
    "population_pcg64_buffer_words",
    "mutation_np_semantic_draws",
    "chemistry",
)

ORDINARY_BIRTH_POLICY_FIELDS: Final[tuple[str, ...]] = (
    "mlp_pack",
    "hmem",
    "rng_manager_mutation_torch_cpu_state",
    "policy_birth_delta_tape",
    "policy_birth_mutation_mask_tape",
    "policy_birth_crossover_mask_packed_tape",
    "policy_birth_post_rng_state_packed_tape",
    "policy_birth_semantic_draws_tape",
    "policy_birth_tape_cursor",
    "mutation_torch_cpu_semantic_draws",
)

TURNOVER_STATE_FIELDS: Final[tuple[str, ...]] = (
    "energy_collected",
    "energy_collected_exact",
    "attacks",
    "early_survival_steps",
    "early_energy_collected",
    "early_energy_collected_exact",
    "early_injury_gained",
    "early_life_weight",
    "pred_err_sum",
    "pred_err_count",
    "first_repro_step",
    "dev_stage_initialized",
    "episode0_pred_err_sum",
    "episode0_pred_err_count",
    "late_pred_err_sum",
    "late_pred_err_count",
    "chemistry_observation_fidelity",
    "chemistry_observation_bias",
)

BASE_BATCHED_FIELDS: Final[tuple[str, ...]] = (
    "slot_to_agent_id",
    "alive_mask",
    "slot_tokens",
    "action_cache_actions",
    "action_cache_token",
    "action_cache_cursor",
    "action_cache_next",
    "pos_y",
    "pos_x",
    "agent_grid",
    "food_grid",
    "biome_grid",
    "hazard_rate_map",
    "energy",
    "health",
    "injury",
    "policy_energy_observation",
    "rest_streak",
    "pending_free_agent_ids",
    "pending_free_slots",
    "pending_free_count",
    "energy_collected",
    "energy_collected_exact",
    "last_action",
    "last_last_action",
    "action_switches",
    "action_oscillations",
    "episode0_switches",
    "episode0_oscillations",
    "survival_steps",
    "mlp_pack",
    "gru_pack",
    "drives",
    "attack_bias",
    "action_temperature",
    "drive_logit_gain",
    "age",
    "gru_upkeep_cost",
    "cost_charge_scalar",
    "cost_charge_maturation_scalar",
    "drive_inertia",
    "drive_noise",
    "prev_drive_delta",
    "drive_volatility",
    "drive_jitter",
    "episode0_drive_jitter_sum",
    "last_energy_delta",
    "last_health_delta",
    "last_injury_delta",
    "selfharm_events",
    "episode0_selfharm_events",
    "chemistry",
    "runner_pcg64_state_words",
    "runner_pcg64_buffer_words",
    "pcg64_state_words",
    "pcg64_buffer_words",
    "population_pcg64_state_words",
    "population_pcg64_buffer_words",
    "drive_noise_samples_by_tick_slot",
    "drive_noise_samples_valid_by_tick_slot",
    "slot_free_queue",
    "slot_free_queue_count",
    "lineage_founder_id",
    "lineage_depth",
)

PATCH_RESIDENT_FIELDS: Final[tuple[str, ...]] = (
    *PATCH_P2_STATE_FIELDS_V1,
    "patch_recovery_activity",
    "patch_habitat_lineage_id",
)

DIRECT_RESIDENT_FIELD_GROUPS_V1: Final[
    tuple[tuple[str, tuple[str, ...]], ...]
] = (
    ("base_batched", BASE_BATCHED_FIELDS),
    ("ordinary_birth_identity", ORDINARY_BIRTH_IDENTITY_FIELDS),
    ("ordinary_birth_scalar_rng", ORDINARY_BIRTH_SCALAR_RNG_FIELDS),
    ("ordinary_birth_genome_int32", ORDINARY_BIRTH_GENOME_INT32_FIELDS),
    ("ordinary_birth_genome_float32", ORDINARY_BIRTH_GENOME_FLOAT32_FIELDS),
    ("ordinary_birth_drive", ORDINARY_BIRTH_DRIVE_FIELDS),
    ("ordinary_birth_policy", ORDINARY_BIRTH_POLICY_FIELDS),
    ("turnover_state", TURNOVER_STATE_FIELDS),
    ("patch_p2_state", PATCH_P2_STATE_FIELDS_V1),
    ("patch_resident", PATCH_RESIDENT_FIELDS),
)

DIRECT_RESIDENT_FIELD_SURFACE_DIGEST_V1: Final[ContentDigest] = canonical_sha256(
    {
        "schema_version": DIRECT_RESIDENT_FIELD_SURFACE_SCHEMA_VERSION,
        "time_authority": DIRECT_RESIDENT_FIELD_SURFACE_TIME_AUTHORITY,
        "qualification_authority": (
            DIRECT_RESIDENT_FIELD_SURFACE_QUALIFICATION_AUTHORITY
        ),
        "persistence_labels_only": (
            DIRECT_RESIDENT_FIELD_SURFACE_PERSISTENCE_LABELS_ONLY
        ),
        "groups": [
            {"name": name, "fields": list(fields)}
            for name, fields in DIRECT_RESIDENT_FIELD_GROUPS_V1
        ],
    }
)
