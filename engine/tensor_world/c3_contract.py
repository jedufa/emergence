"""Source-bound C3 contract and audit helpers.

The C3 program is deliberately source-audited before implementation.  This
module is a small, executable inventory of the current Architecture-B route;
it does not grant C3 gate authority and it does not promote the older device
neutral manifest into a Tensor World qualification artifact.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Final

from .contracts import TensorWorldContractError


C3_SOURCE_AUDIT_SCHEMA_V1: Final[str] = "tensor_world_c3_source_audit_v1"
C3_SOURCE_MANIFEST_PATH: Final[Path] = Path(
    "engine/perf/flat_patch_wdl_device_neutral_state_manifest_v1.json"
)
C3_SOURCE_MANIFEST_SHA256: Final[str] = (
    "7d3d9ebffd6d47bf7e57465767481fb18febbd7499981ac5daf768b7e17b6ce4"
)
C3_PATCH_OBSERVATION_WIDTH_V1: Final[int] = 74
C3_COMPOSED_OBSERVATION_WIDTH_V1: Final[int] = 83
C3_POLICY_HIDDEN_WIDTH_V1: Final[int] = 64
C3_ACTION_COUNT_V1: Final[int] = 7
C3_POLICY_PACKED_WIDTH_V1: Final[int] = 9_991
C3_DRIVE_WIDTH_V1: Final[int] = 4
C3_OBSERVATION_CHANNELS_V1: Final[tuple[str, ...]] = (
    "legacy_occupancy_or_food",
    "patch_food_norm",
    "patch_cover",
    "patch_shelter",
    "patch_scent_norm",
    "patch_waste_norm",
    "patch_density_norm",
    "patch_movement_cost_norm",
    "wdl_cue",
)
C3_OBSERVATION_COLUMNS_V1: Final[tuple[str, ...]] = tuple(
    f"{channel}[dy={dy},dx={dx}]"
    for channel in C3_OBSERVATION_CHANNELS_V1
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
) + (
    "agent_energy_over_max_energy",
    "agent_health_over_start_health",
)
C3_OBSERVATION_REGISTRY_SHA256_V1: Final[str] = (
    "e8a5eac546d09f302db52503a14a9ca40597e3e6bc9e488d29d9e626035f176f"
)


# These fields are resident agent inputs/outputs for the fixed-MLP target
# route.  The names are taken from the current source header/kernel and the
# repository's source-bound state manifest.  They are intentionally explicit:
# a later implementation must either own each field or fail closed.
C3_SOURCE_FIELD_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "C3_REQUIRED": (
        "alive_mask",
        "slot_to_agent_id",
        "agent_id_to_slot_agent_ids",
        "agent_id_to_slot_slots",
        "slot_tokens",
        "pos_y",
        "pos_x",
        "agent_grid",
        "food_grid",
        "energy",
        "health",
        "injury",
        "policy_energy_observation",
        "rest_streak",
        "survival_steps",
        "energy_collected",
        "energy_collected_exact",
        "last_action",
        "last_last_action",
        "action_switches",
        "action_oscillations",
        "episode0_switches",
        "episode0_oscillations",
        "attacks",
        "age",
        "gru_upkeep_cost",
        "cost_charge_scalar",
        "cost_charge_maturation_scalar",
        "drives",
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
        "mlp_pack",
        "policy_attack_bias",
        "policy_action_temperature",
        "policy_drive_logit_gain",
        "action_cache_actions",
        "action_cache_token",
        "action_cache_cursor",
        "action_cache_next",
        "genome_hidden_size",
        "genome_memory_dim",
        "genome_sensor_channel_mask",
        "genome_sensor_radius",
        "genome_introspection_channel_mask",
        "genome_self_regulation_control_mask",
        "genome_self_modification_target_mask",
        "genome_directed_inheritance_channel_mask",
        "genome_exploration_bias",
        "genome_aggression_bias",
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
        "pred_lr",
        "repro_threshold",
        "chemistry",
        "dev_stage_initialized",
        "chemistry_observation_fidelity",
        "chemistry_observation_bias",
        "observations_out",
        "logits_out",
        "row_actions_out",
        "actions_by_slot_out",
        "death_eligibility_outputs",
    ),
    "C4_TRANSACTIONAL": (
        "scar",
        "repro_cooldown",
        "offspring_count",
        "lineage_birth_generation",
        "lineage_birth_tick",
        "lineage_founder_id",
        "lineage_depth",
        "patch_habitat_lineage_id",
        "population_next_id",
        "slot_free_queue",
        "slot_free_queue_count",
        "birth_reset_shock",
        "early_survival_steps",
        "early_energy_collected",
        "early_energy_collected_exact",
        "early_injury_gained",
        "early_life_weight",
        "first_repro_step",
        "genome_mutation_rate",
        "genome_reproduction_threshold",
        "genome_resource_handling_axis",
        "genome_upgrade_v2_enabled",
        "genome_upgrade_v2_alleles_q10",
        "genome_upgrade_v2_modifiers_q10",
        "genome_upgrade_v2_products_q10",
        "genome_upgrade_v2_phenotypes_q10",
        "genome_upgrade_v2_expression_cost_q10",
        "genome_upgrade_v2_active_product_count",
        "genome_upgrade_v2_saturated_product_count",
        "policy_birth_delta_tape",
        "policy_birth_mutation_mask_tape",
        "policy_birth_crossover_mask_packed_tape",
        "policy_birth_post_rng_state_packed_tape",
        "mutation_torch_cpu_state",
        "python_random_words",
        "python_random_index",
        "python_random_has_gauss",
        "python_random_gauss_next",
        "diploid_allele_storage",
        "inheritance_mode",
    ),
    "C5_DETERMINISM_ONLY": (
        "environment_pcg64_state",
        "policy_pcg64_state",
        "birth_spawn_pcg64_state",
        "runner_order_pcg64_state",
        "population_pcg64_state",
        "population_mt19937_state",
        "rng_manager_stream_indices",
        "runner_pcg64_state_words",
        "runner_pcg64_buffer_words",
        "shared_env_policy_pcg64_state_words",
        "shared_env_policy_pcg64_buffer_words",
        "drive_noise_samples",
        "drive_noise_samples_valid",
        "policy_np_semantic_draws",
        "metrics_np_semantic_draws",
        "mutation_np_semantic_draws",
        "mutation_py_semantic_draws",
        "mutation_torch_cpu_semantic_draws",
        "policy_birth_semantic_draws_tape",
        "policy_birth_tape_cursor",
        "policy_birth_tape_cursor_base",
        "python_random_raw_draws",
        "python_random_raw_draw_base",
    ),
    "C6_EXTERNAL_ONLY": (
        "owner_runtime_counters",
        "event_tape_actor_id",
        "event_tape_generation",
        "event_tape_target_id",
        "event_tape_tick",
        "event_tape_type",
        "event_tape_value",
        "typed_lifecycle_event_actor_agent_id",
        "typed_lifecycle_event_actor_slot",
        "typed_lifecycle_event_child_agent_id",
        "typed_lifecycle_event_event_type",
        "typed_lifecycle_event_generation",
        "typed_lifecycle_event_lineage_depth",
        "typed_lifecycle_event_lineage_founder_id",
        "typed_lifecycle_event_parent_agent_id",
        "typed_lifecycle_event_reason_code",
        "typed_lifecycle_event_sequence",
        "typed_lifecycle_event_subtype_code",
        "typed_lifecycle_event_success",
        "typed_lifecycle_event_target_agent_id",
        "typed_lifecycle_event_target_slot",
        "typed_lifecycle_event_tick",
        "typed_lifecycle_event_value",
        "behavior_reporting_accumulator",
        "immutable_config_bank",
    ),
    # The current C0 oracle binding is fixed_mlp.  GRU state remains in the
    # source ABI for explicit capability awareness, but is not silently
    # executed by this target route.
    "INACTIVE_ON_TARGET_ROUTE": (
        "gru_pack",
        "hmem",
        "policy_mem_dim",
        "policy_use_gru",
        "pred_err_sum",
        "pred_err_count",
        "episode0_pred_err_sum",
        "episode0_pred_err_count",
        "late_pred_err_sum",
        "late_pred_err_count",
    ),
}


C3_SOURCE_REFERENCES: Final[dict[str, dict[str, str]]] = {
    "persistent_generation_binding": {
        "path": "engine/perf/cuda_architecture_b_persistent_generation_binding.cpp",
        "symbols": "persistent_flat_dynamic_policy_food_birth_full_tick_v7",
        "sha256": "2bac7e3d4cf0198bdfcabb810849236ff8bf22cbf7585c5ba17937a4a405d382",
    },
    "persistent_generation_kernel": {
        "path": "engine/perf/cuda_architecture_b_persistent_generation_kernel.cu",
        "symbols": "persistent full-tick source-current execution",
        "sha256": "6a9f01583bf7126b49374e027a564aaff65ddb61ff9125c6557ff6243d4205a9",
    },
    "controller_abi": {
        "path": "engine/perf/cuda_architecture_b_flattened_controller_v1.hpp",
        "symbols": "run_flat_dynamic_policy_food_flattened_controller_v1, FlatFullTickState, FlatOrdinaryBirthState",
        "sha256": "fe275e14c9c88004cace5b15d264634c1c677e18c3af2dd3f117682afa114697",
    },
    "controller_kernel": {
        "path": "engine/perf/cuda_architecture_b_flattened_controller_v1_kernels.cu",
        "symbols": "flattened_controller_tick_core_and_birth_plan_kernel_v1, write_patch_radius_one_observation_exact",
        "sha256": "5f1e76615c1631b29dd9402a4251304d183ab4217f030e63d0ac0b0d6ca94851",
    },
    "active_mechanics": {
        "path": "engine/perf/cuda_architecture_b_persistent_active_mechanics.cuh",
        "symbols": "FlatActiveMechanicsCellState, FlatActiveMechanicsPreAction, run_flat_active_mechanics",
        "sha256": "0cd6088db1361051fa48500d127adebc49f2f6f664994faa8b9567eb59afd7c4",
    },
    "chemistry": {
        "path": "engine/perf/cuda_architecture_b_persistent_chemistry.cuh",
        "symbols": "ChemistryConfig, chemistry_update_counted",
        "sha256": "667842dc7d58d0182c2b075da341a5878ac9bfb5fda51df8bda7206e62c38d71",
    },
    "policy": {
        "path": "engine/perf/cuda_architecture_b_persistent_stochastic_policy.cuh",
        "symbols": "FixedMlpPolicyConfig, service_stable_gru_gumbel_policy_serial",
        "sha256": "0b4b69440cf6f88bcb02ce9f60f093aae39a44c20ce0e79e26160a6a05acc8b4",
    },
    "state_manifest": {
        "path": str(C3_SOURCE_MANIFEST_PATH),
        "symbols": "tensor_surfaces, non_tensor_state_surfaces",
        "sha256": C3_SOURCE_MANIFEST_SHA256,
    },
}


C3_SOURCE_ORDER: Final[tuple[str, ...]] = (
    "source_current_patch_world_update_and_food_sync",
    "ascending_identity_validation_and_runner_order",
    "genome_v2_resident_expression_when_enabled",
    "observation_materialization",
    "fixed_mlp_policy_and_action_cache_service",
    "ordered_movement_and_occupancy_commit",
    "gather_and_resource_transaction",
    "energy_cost_and_food_settlement",
    "death_eligibility_cull_boundary_without_c4_slot_transaction",
    "drive_noise_materialization_and_active_mechanics",
    "chemistry_state_update",
    "residual_death_eligibility_and_c4_boundary",
)


def _manifest_names(manifest_path: Path) -> set[str]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    tensor_names = {
        str(entry["canonical_name"]) for entry in payload["tensor_surfaces"]
    }
    non_tensor_names = {
        str(entry["canonical_name"])
        for entry in payload["non_tensor_state_surfaces"]
    }
    return tensor_names | non_tensor_names


def validate_c3_source_audit(repo_root: Path | str = ".") -> dict[str, object]:
    """Validate the source manifest binding and inventory consistency.

    The audit intentionally permits derived transaction workspaces and source
    ABI outputs that are not persistent manifest fields.  Those names are
    listed in ``C3_DERIVED_FIELDS`` below and are still required to have an
    explicit owner in the implementation contract.
    """

    root = Path(repo_root)
    manifest = root / C3_SOURCE_MANIFEST_PATH
    if not manifest.is_file():
        raise TensorWorldContractError(f"C3 source manifest is missing: {manifest}")
    digest = sha256(manifest.read_bytes()).hexdigest()
    if digest != C3_SOURCE_MANIFEST_SHA256:
        raise TensorWorldContractError(
            "C3 source manifest drifted; refresh the audit before implementation "
            f"(expected {C3_SOURCE_MANIFEST_SHA256}, got {digest})"
        )

    all_names = _manifest_names(manifest)
    groups = {name: set(values) for name, values in C3_SOURCE_FIELD_GROUPS.items()}
    flat = [item for values in groups.values() for item in values]
    duplicates = sorted({item for item in flat if flat.count(item) > 1})
    if duplicates:
        raise TensorWorldContractError(
            f"C3 source field appears in multiple classifications: {duplicates}"
        )
    missing_manifest = sorted(
        item
        for item in flat
        if item not in all_names and item not in C3_DERIVED_FIELDS
    )
    if missing_manifest:
        raise TensorWorldContractError(
            "C3 source inventory names are absent from the bound manifest or "
            f"derived-field declaration: {missing_manifest}"
        )
    return {
        "schema_version": C3_SOURCE_AUDIT_SCHEMA_V1,
        "source_manifest_sha256": digest,
        "manifest_field_count": len(all_names),
        "classified_field_count": len(flat),
        "derived_field_count": len(C3_DERIVED_FIELDS),
        "classifications": {key: len(values) for key, values in groups.items()},
        "source_order": list(C3_SOURCE_ORDER),
    }


C3_DERIVED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "active_agent_ids",
        "active_agent_slots",
        "active_count_state",
        "observations_out",
        "logits_out",
        "row_actions_out",
        "actions_by_slot_out",
        "death_eligibility_outputs",
        "chemistry_observation_fidelity",
        "chemistry_observation_bias",
        "pred_lr",
        "slot_free_queue_count",
        "runner_pcg64_state_words",
        "runner_pcg64_buffer_words",
        "shared_env_policy_pcg64_state_words",
        "shared_env_policy_pcg64_buffer_words",
        "drive_noise_samples",
        "drive_noise_samples_valid",
        "policy_np_semantic_draws",
        "metrics_np_semantic_draws",
        "mutation_np_semantic_draws",
        "mutation_py_semantic_draws",
        "mutation_torch_cpu_semantic_draws",
        "policy_birth_semantic_draws_tape",
        "policy_birth_tape_cursor",
        "policy_birth_tape_cursor_base",
        "python_random_raw_draws",
        "python_random_raw_draw_base",
        "genome_upgrade_v2_enabled",
        "genome_upgrade_v2_alleles_q10",
        "genome_upgrade_v2_modifiers_q10",
        "genome_upgrade_v2_products_q10",
        "genome_upgrade_v2_phenotypes_q10",
        "genome_upgrade_v2_expression_cost_q10",
        "genome_upgrade_v2_active_product_count",
        "genome_upgrade_v2_saturated_product_count",
        "genome_resource_handling_axis",
        "policy_birth_delta_tape",
        "policy_birth_mutation_mask_tape",
        "policy_birth_crossover_mask_packed_tape",
        "policy_birth_post_rng_state_packed_tape",
        "mutation_torch_cpu_state",
        "python_random_words",
        "python_random_index",
        "python_random_has_gauss",
        "python_random_gauss_next",
        "diploid_allele_storage",
        "inheritance_mode",
    }
)


def c3_observation_abi_manifest_v1() -> dict[str, object]:
    """Return the source-bound 83-column observation registry.

    The registry is a preparation contract shared by the C2 observation
    provider, the C3 policy oracle, and future channel-level parity reports.
    It does not imply that the C3 gate has been authorized.
    """

    return {
        "schema_version": "eiel_wdl_observation_registry_v3",
        "column_count": len(C3_OBSERVATION_COLUMNS_V1),
        "columns": list(C3_OBSERVATION_COLUMNS_V1),
        "registry_sha256": C3_OBSERVATION_REGISTRY_SHA256_V1,
        "spatial_order": (
            "channel_major_then_dy_minus1_to_plus1_then_dx_minus1_to_plus1"
        ),
        "row_mask": "alive_slots_only",
        "normalization": {
            "legacy_occupancy_or_food": "exact_-1_0_1",
            "patch_food_scent_waste_density": "safe_norm_clamped_0_1",
            "patch_cover_shelter_wdl_cue": "finite_clamped_0_1",
            "patch_movement_cost": "fixed_span_4_clamped_0_1",
            "agent_energy": "energy_over_max_energy",
            "agent_health": "health_over_start_health",
        },
    }


__all__ = [
    "C3_ACTION_COUNT_V1",
    "C3_COMPOSED_OBSERVATION_WIDTH_V1",
    "C3_DRIVE_WIDTH_V1",
    "C3_OBSERVATION_CHANNELS_V1",
    "C3_OBSERVATION_COLUMNS_V1",
    "C3_OBSERVATION_REGISTRY_SHA256_V1",
    "C3_PATCH_OBSERVATION_WIDTH_V1",
    "C3_POLICY_HIDDEN_WIDTH_V1",
    "C3_POLICY_PACKED_WIDTH_V1",
    "C3_SOURCE_AUDIT_SCHEMA_V1",
    "C3_SOURCE_FIELD_GROUPS",
    "C3_SOURCE_MANIFEST_PATH",
    "C3_SOURCE_MANIFEST_SHA256",
    "C3_SOURCE_ORDER",
    "C3_SOURCE_REFERENCES",
    "c3_observation_abi_manifest_v1",
    "validate_c3_source_audit",
]
