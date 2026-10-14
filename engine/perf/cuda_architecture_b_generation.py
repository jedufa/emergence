"""One-call deterministic or exact-PCG64 CUDA generation composition."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from hashlib import sha256
import math
import os
from pathlib import Path
from typing import Any, Final, Mapping

import torch
from torch.utils.cpp_extension import load

from perf.cuda_architecture_b_birth_transaction import (
    DRIVE_STATE_FIELDS,
    GENOME_FLOAT32_FIELDS,
    GENOME_INT32_FIELDS,
    IDENTITY_STATE_FIELDS,
    POLICY_STATE_FIELDS,
    SCALAR_RNG_STATE_FIELDS,
)
from perf.cuda_architecture_b_generation_birth_state import CudaGenerationBirthState
from perf.cuda_architecture_b_modular_generation_contract import (
    prepare_modular_generation_entry,
)
from perf.cuda_architecture_b_patch_context import PATCH_CONTEXT_P2_STAT_FIELDS
from perf.cuda_architecture_b_prebuilt import (
    load_prebuilt_cuda_architecture_b_extension,
    validate_cuda_prebuilt_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
PERF = ROOT / "engine" / "perf"
NATIVE = PERF / "native"
GENERATION_SOURCE = NATIVE / "cuda_architecture_b_generation.cpp"
GENERATION_MULTICELL_SOURCE = NATIVE / "cuda_architecture_b_generation_multicell.cpp"
GENERATION_CONTEXT_HEADER = NATIVE / "cuda_architecture_b_generation_context.hpp"
OBSERVATION_ACTION_SOURCE = NATIVE / "cxxbackbone8r_generation_island_cuda.cu"
FIXED_MLP_SOURCE = NATIVE / "cuda_architecture_b_fixed_mlp.cu"
MODULAR_TICK_SOURCE = NATIVE / "cuda_architecture_b_modular_tick.cpp"
MODULAR_TICK_HEADER = NATIVE / "cuda_architecture_b_modular_tick.hpp"
POLICY_CACHE_SOURCE = NATIVE / "cuda_architecture_b_policy_cache.cu"
PATCH_CONTEXT_SOURCE = NATIVE / "cuda_architecture_b_patch_context.cu"
FLAT_FOOD_SPAWN_SOURCE = NATIVE / "cuda_architecture_b_flat_food_spawn.cu"
ACTIVE_MECHANICS_SOURCE = NATIVE / "cuda_architecture_b_active_mechanics.cu"
LIFECYCLE_SOURCE = NATIVE / "cuda_architecture_b_lifecycle.cu"
DEATH_LIFECYCLE_EVENTS_SOURCE = NATIVE / "cuda_architecture_b_death_lifecycle_events.cu"
BIRTH_MICROCYCLE_SOURCE = NATIVE / "cuda_architecture_b_birth_microcycle.cpp"
ASEXUAL_BIRTH_PLAN_SOURCE = NATIVE / "cuda_architecture_b_asexual_birth_plan.cu"
BIRTH_TRANSACTION_SOURCE = NATIVE / "cuda_architecture_b_birth_transaction.cpp"
CHILD_DRIVES_SOURCE = NATIVE / "cuda_architecture_b_child_drives.cu"
CHILD_POLICY_SOURCE = NATIVE / "cuda_architecture_b_child_policy.cu"
TURNOVER_SOURCE = NATIVE / "cuda_architecture_b_turnover.cu"
BIRTH_LIFECYCLE_EVENTS_SOURCE = NATIVE / "cuda_architecture_b_birth_lifecycle_events.cu"
DEATH_CLASS_CONCAT_SOURCE = NATIVE / "cuda_architecture_b_death_class_concat.cu"
PCG64_HEADER = NATIVE / "cuda_architecture_b_pcg64.cuh"
TORCH_RNG_STATE_PACK_HEADER = PERF / "cuda_architecture_b_torch_rng_state_pack_v1.cuh"
COMPILE_SOURCES: Final[tuple[Path, ...]] = (
    GENERATION_SOURCE,
    GENERATION_MULTICELL_SOURCE,
    OBSERVATION_ACTION_SOURCE,
    FIXED_MLP_SOURCE,
    MODULAR_TICK_SOURCE,
    POLICY_CACHE_SOURCE,
    PATCH_CONTEXT_SOURCE,
    FLAT_FOOD_SPAWN_SOURCE,
    ACTIVE_MECHANICS_SOURCE,
    LIFECYCLE_SOURCE,
    DEATH_LIFECYCLE_EVENTS_SOURCE,
    BIRTH_MICROCYCLE_SOURCE,
    ASEXUAL_BIRTH_PLAN_SOURCE,
    BIRTH_TRANSACTION_SOURCE,
    CHILD_DRIVES_SOURCE,
    CHILD_POLICY_SOURCE,
    TURNOVER_SOURCE,
    BIRTH_LIFECYCLE_EVENTS_SOURCE,
    DEATH_CLASS_CONCAT_SOURCE,
)
HASH_SOURCES: Final[tuple[Path, ...]] = (
    *COMPILE_SOURCES,
    GENERATION_CONTEXT_HEADER,
    MODULAR_TICK_HEADER,
    PCG64_HEADER,
    TORCH_RNG_STATE_PACK_HEADER,
)

CUDA_GENERATION_CAPABILITIES: Final[dict[str, bool | str]] = {
    "schema_version": "eiel_cuda_architecture_b_generation_capabilities_v4",
    "decision": "ARCHB-CUDA-STAGE2/BOUNDED-TICK-SYNCHRONOUS-CROSS-CELL-BATCHING",
    "single_native_cpp_generation_call": True,
    "native_cpp_tick_loop": True,
    "slot_indexed_owner_observation": True,
    "fixed_mlp_refill": True,
    "generation_start_action_cache_reset": True,
    "owner_action_mutation": True,
    "fixed_capacity_active_population": True,
    "device_resident_active_count": True,
    "data_dependent_active_compaction": False,
    "host_extinction_branch": False,
    "base_patch_context": True,
    "patch_p2_waste_density_consequences": True,
    "accepted_patch_order": (
        "context_then_observation_policy_action_then_action_death_cull_then_"
        "active_mechanics_then_residual_death_cull_release_then_birth"
    ),
    "accepted_flat_patch_active_mechanics": True,
    "exact_pcg64_stochastic_fixed_mlp": True,
    "lifecycle_death_cull": True,
    "death_event_semantic_construction": "fixed_capacity_device_sidecar",
    "legacy_and_typed_event_tape_append": True,
    "death_event_data_dependent_host_shapes": False,
    "event_tape_capacity_failure": "atomic_append_failure_with_sticky_status",
    "state_cuda_resident_between_ticks": True,
    "python_tick_loop": False,
    "host_transfer_between_ticks": False,
    "generation_reporting_sidecar": "pre_turnover_immutable_device_tensors_v1",
    "generation_reporting_host_sync_inside_generation": False,
    "generation_reporting_scope": (
        "full_generation_turnover_fixed_mlp_multi_tick_no_ordinary_births"
    ),
    "generation_reporting_public_behavior_rows": "exact_flat_patch_g1",
    "lifecycle": True,
    "lifecycle_scope": "flat_patch_fixed_mlp_death_birth_turnover_exact",
    "stochastic_cpu_rng_parity": "exact_when_resident_pcg64_state_is_provided",
    "birth": True,
    "birth_scope": "ordinary_birth_focused_exact_turnover_full_generation_exact",
    "death_split_generation_integration": True,
    "ordinary_birth_generation_integration": "focused_exact",
    "checkpoint": True,
    "checkpoint_scope": "exact_flat_patch_generation_boundary_save_resume",
    "multi_cell": True,
    "multi_cell_scope": "same_shape_flat_patch_tick_synchronous_exact",
    "cross_cell_tick_synchronous_controller": True,
    "cross_cell_policy_cache_batch": True,
    "cross_cell_observation_batch": True,
    "cross_cell_fixed_mlp_batch": True,
    "cross_cell_lifecycle_batch": True,
    "cross_cell_active_mechanics_batch": "enabled_at_cell_count_16_or_greater",
    "throughput_qualified": False,
    "gru": False,
    "jungle": False,
    "wdl": False,
    "promotion_qualified": False,
}

CUDA_MODULAR_GENERATION_CAPABILITIES: Final[dict[str, bool | str]] = {
    "schema_version": "eiel_cuda_architecture_b_modular_generation_capabilities_v1",
    "lane": "single_cell_native_generation_phase_e_v1",
    "explicit_native_owner": True,
    "native_cpp_tick_loop": True,
    "native_modular_tick": True,
    "modular_residual_before_action_selection": True,
    "fixed_observation_semantics": "radius2_local25_plus_energy_health_v1",
    "fixed_action_width": "7",
    "fixed_physical_envelope": "16_modules_48_connections",
    "supported_logical_profiles": "12x32_or_16x48",
    "cache_width": "1",
    "policy_stride": "1",
    "births": False,
    "turnover": False,
    "multi_cell": False,
    "hidden_fallback": False,
    "host_transfer_between_ticks": False,
    "promotion_qualified": False,
}

CUDA_GENERATION_PREBUILT_CAPABILITY_CONTRACT: Final[dict[str, Any]] = {
    **CUDA_GENERATION_CAPABILITIES,
    "phase_e_modular_generation": dict(CUDA_MODULAR_GENERATION_CAPABILITIES),
}

_MODULES: dict[str, Any] = {}
_PREBUILT_ENV: Final[str] = "EIEL_CUDA_ARCHITECTURE_B_PREBUILT"
_REQUIRED_EXTENSION_EXPORTS: Final[tuple[str, ...]] = (
    "eiel_cuda_architecture_b_generation_v3",
    "eiel_cuda_architecture_b_generation_multicell_v2",
    "eiel_cuda_architecture_b_generation_phase_e_v1",
)


@dataclass(frozen=True)
class CudaGenerationConfig:
    steps: int
    patch_enabled: bool
    hidden_size: int
    generation: int = 0
    start_tick: int = 0
    action_count: int = 7
    cache_width: int = 1
    view_radius: int = 1
    max_energy: float = 100.0
    start_health: float = 1.0
    patch_food_amount_max: float = 1.0
    patch_food_amount_initial_food_cell_value: float = 1.0
    patch_scent_max: float = 1.0
    patch_waste_max: float = 1.0
    patch_density_radius: int = 1
    patch_density_include_self: bool = True
    patch_density_normalize: bool = False
    patch_movement_cost_min: float = 0.0
    patch_movement_cost_max: float = 4.0
    rng_seed: int = 0
    rng_counter: int = 0
    rng_semantics_version: int = 1
    world_uid: int = 0
    rng_namespace_key64: int = 0xCA05117E450A51C0
    rng_semantic_epoch: int = 1
    w_hunger_gather: float = 1.2
    w_pain_rest: float = 0.9
    w_fatigue_rest: float = 1.0
    w_pain_attack: float = 1.2
    w_fear_attack: float = 1.4
    move_cost: float = 0.2
    gather_cost: float = 0.1
    rest_cost: float = 0.05
    attack_cost: float = 0.3
    action_cost_mult: float = 1.0
    rest_cost_mult: float = 1.0
    food_energy: float = 2.0
    max_food: int = 0
    food_spawn_per_step: float = 0.0
    food_spawn_mult: float = 1.0
    enable_biomes: bool = False
    patch_food_gather_amount: float = 0.5
    patch_food_presence_threshold: float = 0.5
    patch_scent_move_emit: float = 0.0
    patch_waste_move_emit: float = 0.0
    patch_scent_gather_emit: float = 0.0
    patch_waste_consume_emit: float = 0.0
    patch_movement_cost_apply_enable: bool = False
    patch_movement_cost_energy_scale: float = 0.0
    patch_scent_decay_per_step: float = 0.0
    patch_waste_decay_per_step: float = 0.0
    patch_scent_presence_emit: float = 0.01
    patch_waste_presence_emit: float = 0.005
    patch_waste_density_emit_scale: float = 0.0
    patch_food_regrow_per_step: float = 0.0
    patch_food_regrow_use_fertility: bool = True
    patch_food_regrow_cap_by_fertility: bool = False
    patch_world_waste_consequence_enable: bool = False
    patch_waste_regrowth_penalty_enable: bool = True
    patch_waste_regrowth_penalty_scale: float = 0.5
    patch_waste_regrowth_penalty_max: float = 0.5
    patch_density_regrowth_penalty_enable: bool = True
    patch_density_regrowth_penalty_scale: float = 0.5
    patch_density_regrowth_penalty_max: float = 0.5
    patch_regrowth_min_multiplier: float = 0.1
    active_mechanics_enabled: bool = True
    resource_maintenance_energy_cost_per_tick: float = 0.0
    enable_rest_charge: bool = False
    rest_streak_free: int = 8
    rest_charge_energy: float = 0.25
    rest_charge_growth: float = 0.10
    rest_deficit_to_health: float = 1.0
    developmental_maturity_age: int = 20
    life_history_mode: int = 0
    life_history_maturity_start_ticks: int = 0
    life_history_maturity_rise_ticks: int = 0
    life_history_fertility_decline_start_ticks: int = 0
    life_history_fertility_decline_ticks: int = 0
    life_history_elder_fertility_floor: float = 1.0
    life_history_senescence_start_ticks: int = 0
    life_history_senescence_ramp_ticks: int = 0
    life_history_senescence_energy_cost: float = 0.0
    life_history_senescence_health_cost: float = 0.0
    life_history_max_age_ticks: int = 0
    structural_development_mature_start_step: int = 20
    structural_development_late_start_step: int = 40
    episode0_k: int = 20
    drive_hunger_gain: float = 0.08
    drive_hunger_decay: float = 0.02
    drive_pain_gain: float = 0.18
    drive_pain_decay: float = 0.03
    drive_fatigue_gain: float = 0.05
    drive_fatigue_rest_relief: float = 0.10
    drive_fatigue_decay: float = 0.01
    drive_fear_gain: float = 0.22
    drive_fear_decay: float = 0.03
    max_births_per_tick: int = 0
    ordinary_births_enabled: bool | None = None
    reproduce_parent_energy_share: float = 0.20
    reproduce_min_energy_buffer: float = 20.0
    reproduce_cooldown_steps: int = 35
    reproduce_try_prob: float = 1.0
    reproduce_requires_mate: bool = False
    inheritance_profile: str = "legacy_midpoint_v1"
    diploid_mutation_enable: bool = True
    recomb72_mutation_off: bool = False
    resource_handling_representation_enable: bool = False
    genome_upgrade_v2_enable: bool = False
    genome_upgrade_v2_mutation_rate_q10: int = 20
    genome_upgrade_v2_mutation_step_q10: int = 64
    genome_upgrade_v2_founder_profile_id: str = "neutral"
    genome_upgrade_v2_founder_profile_sha256: str = ""
    genome_v2_causal_capture_enable: bool = False
    injury_repro_k_injury: float = 0.35
    injury_repro_k_scar: float = 0.9
    injury_birth_fail_scar: float = 0.12
    injury_birth_fail_injury: float = 0.03
    genome_sensor_radius_max: int = 1
    generation_turnover_enabled: bool = False
    lifecycle_mode: int = 0
    population_size: int = 0
    cull_fraction: float = 0.0
    parent_fraction: float = 0.25
    turnover_selection_mode: int = 0
    turnover_selection_seed: int = 0
    start_energy: float = 50.0
    early_life_steps: int = 0
    early_life_weight: float = 1.0
    enable_combat: bool = False
    coherence_thrashing_weight: float = 0.15
    coherence_drive_volatility_weight: float = 0.08
    coherence_selfharm_weight: float = 3.0
    coherence_pred_err_weight: float = 1.25
    episode0_fitness_weight: float = 0.0
    episode0_pred_err_fitness_weight: float = 0.0
    episode0_thrashing_fitness_weight: float = 0.0
    episode0_drive_jitter_fitness_weight: float = 0.0
    episode0_selfharm_fitness_weight: float = 0.0
    coherence_drive_jitter_weight: float = 0.08
    coherence_reset_shock_weight: float = 0.35
    coherence_episode0_pred_err_weight: float = 0.75
    late_pred_cost_enable: bool = False
    late_pred_cost_weight: float = 0.0
    disable_prediction_error_selection_pressure: bool = False
    env_policy_alias: bool = True
    genome_v2_checkpoint_authority_binding: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, object],
        *,
        steps: int | None = None,
    ) -> "CudaGenerationConfig":
        """Build the resident generation controls from an owner projection."""

        world = str(metadata.get("world", "") or "").strip().lower()
        if world not in {"flat", "patch"}:
            raise RuntimeError(
                "Architecture B generation metadata requires Flat or Patch"
            )
        resolved_steps = int(
            steps
            if steps is not None
            else metadata.get("configured_steps_per_generation", 0) or 0
        )
        if resolved_steps <= 0:
            raise ValueError("generation metadata must provide positive steps")
        hidden_size = int(metadata.get("hidden", 0) or 0)
        if hidden_size <= 0:
            raise ValueError("generation metadata must provide positive hidden")

        aliases = {
            "cache_width": "policy_tick_stride",
            "start_tick": "tick",
            "w_hunger_gather": "drive_w_hunger_gather",
            "w_pain_rest": "drive_w_pain_rest",
            "w_fatigue_rest": "drive_w_fatigue_rest",
            "w_pain_attack": "drive_w_pain_attack",
            "w_fear_attack": "drive_w_fear_attack",
            "env_policy_alias": "native_runtime_env_policy_numpy_alias",
        }
        required = {"steps", "patch_enabled", "hidden_size"}
        values: dict[str, object] = {}
        for field in fields(cls):
            if field.name in required:
                continue
            key = aliases.get(field.name, field.name)
            if key not in metadata or metadata[key] is None:
                continue
            if field.default is MISSING:
                continue
            raw = metadata[key]
            default = field.default
            if field.name == "ordinary_births_enabled":
                values[field.name] = bool(raw)
            elif isinstance(default, bool):
                values[field.name] = bool(raw)
            elif isinstance(default, int):
                values[field.name] = int(raw)
            elif isinstance(default, float):
                values[field.name] = float(raw)
            else:
                values[field.name] = raw
        if "ordinary_births_enabled" not in values:
            values["ordinary_births_enabled"] = (
                int(values.get("max_births_per_tick", 0)) > 0
            )
        return cls(
            steps=resolved_steps,
            patch_enabled=world == "patch",
            hidden_size=hidden_size,
            **values,
        )


@dataclass(frozen=True)
class CudaGenerationResult:
    observations: torch.Tensor
    logits: torch.Tensor
    actions_by_tick_slot: torch.Tensor
    energy: torch.Tensor
    died_mask_by_alive_order: torch.Tensor
    food_gained_by_alive_order: torch.Tensor
    moved_count_by_alive_order: torch.Tensor
    action_metrics_float: torch.Tensor
    action_metrics_int: torch.Tensor
    slots_by_alive_order: torch.Tensor
    patch_stats_by_tick: torch.Tensor
    food_spawn_stats_by_tick: torch.Tensor
    next_rng_counter: torch.Tensor
    pcg64_state_words: torch.Tensor
    pcg64_buffer_words: torch.Tensor
    final_alive_agent_ids: torch.Tensor
    lifecycle_death_agent_ids_by_tick: torch.Tensor
    lifecycle_death_slots_by_tick: torch.Tensor
    lifecycle_death_reason_codes_by_tick: torch.Tensor
    lifecycle_death_counts_by_tick: torch.Tensor
    lifecycle_alive_counts_by_tick: torch.Tensor
    lifecycle_status_by_tick: torch.Tensor
    active_mechanics_status_by_tick: torch.Tensor
    active_mechanics_counts_by_tick: torch.Tensor
    drive_noise_rng_semantic_draws_by_tick: torch.Tensor
    drive_noise_rng_status_by_tick: torch.Tensor
    event_tape_active_count: torch.Tensor
    event_tape_status: torch.Tensor
    event_append_status_by_tick: torch.Tensor
    final_alive_count: torch.Tensor
    birth_reason_counts_by_tick: torch.Tensor
    birth_planned_counts_by_tick: torch.Tensor
    birth_decision_counts_by_tick: torch.Tensor
    birth_committed_counts_by_tick: torch.Tensor
    birth_status_by_tick: torch.Tensor
    residual_release_counts_by_tick: torch.Tensor
    residual_release_status_by_tick: torch.Tensor
    policy_pcg64_state_after_action_by_tick: torch.Tensor
    policy_pcg64_buffer_after_action_by_tick: torch.Tensor
    planner_pcg64_state_before_birth_by_tick: torch.Tensor
    planner_pcg64_buffer_before_birth_by_tick: torch.Tensor
    runner_shuffle_semantic_draws_by_tick: torch.Tensor
    runner_shuffle_status_by_tick: torch.Tensor
    chemistry_semantic_draws_by_tick: torch.Tensor
    chemistry_status_by_tick: torch.Tensor
    metrics_np_semantic_draws: torch.Tensor
    turnover_ranked_agent_ids: torch.Tensor
    turnover_culled_agent_ids: torch.Tensor
    turnover_parent_pool_agent_ids: torch.Tensor
    turnover_refill_parent_agent_ids: torch.Tensor
    turnover_refill_mate_agent_ids: torch.Tensor
    turnover_child_agent_ids: torch.Tensor
    turnover_child_slots: torch.Tensor
    turnover_counts: torch.Tensor
    turnover_status: torch.Tensor
    turnover_next_generation: torch.Tensor
    turnover_next_tick: torch.Tensor
    applied_actions: torch.Tensor
    moved: torch.Tensor
    gathered: torch.Tensor
    policy_refill_calls: torch.Tensor
    policy_refill_rows: torch.Tensor
    runtime_bookkeeping_by_tick: torch.Tensor
    reporting_status: torch.Tensor
    reporting_valid: torch.Tensor
    reporting_generation_counts: torch.Tensor
    reporting_peak_injury: torch.Tensor
    reporting_agent_ids: torch.Tensor
    reporting_cohort_codes: torch.Tensor
    reporting_action_counts: torch.Tensor
    reporting_start_injury: torch.Tensor
    reporting_survival_steps: torch.Tensor
    reporting_energy_collected_exact: torch.Tensor
    reporting_offspring_count: torch.Tensor
    reporting_attacks: torch.Tensor
    reporting_alive_end: torch.Tensor
    reporting_injury_end: torch.Tensor
    reporting_aggression_bias: torch.Tensor
    reporting_early_survival_steps: torch.Tensor
    reporting_early_energy_collected_exact: torch.Tensor
    reporting_early_injury_gained: torch.Tensor
    reporting_early_life_weight: torch.Tensor
    reporting_action_switches: torch.Tensor
    reporting_action_oscillations: torch.Tensor
    reporting_drive_jitter: torch.Tensor
    reporting_selfharm_events: torch.Tensor
    reporting_pred_err_sum: torch.Tensor
    reporting_pred_err_count: torch.Tensor
    reporting_birth_reset_shock: torch.Tensor
    reporting_episode0_pred_err_sum: torch.Tensor
    reporting_episode0_pred_err_count: torch.Tensor
    reporting_episode0_selfharm_events: torch.Tensor
    reporting_late_pred_err_sum: torch.Tensor
    reporting_late_pred_err_count: torch.Tensor


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

_TURNOVER_STATE_DTYPES: Final[tuple[torch.dtype, ...]] = (
    torch.float32,
    torch.float64,
    torch.int32,
    torch.int32,
    torch.float32,
    torch.float64,
    torch.float32,
    torch.float32,
    torch.float32,
    torch.int32,
    torch.int32,
    torch.bool,
    torch.float32,
    torch.int32,
    torch.float32,
    torch.int32,
    torch.float32,
    torch.float32,
)

_ORDINARY_POLICY_STATE_FIELDS: Final[tuple[str, ...]] = tuple(
    field
    for field in POLICY_STATE_FIELDS
    if field != "policy_birth_crossover_mask_packed_tape"
)


_OWNER_TENSOR_NAMES: Final[tuple[str, ...]] = (
    "agent_ids_by_alive_order",
    "agent_id_to_slot",
    "slot_to_agent_id",
    "alive_mask",
    "pos_y",
    "pos_x",
    "agent_grid",
    "food_grid",
    "energy",
    "health",
    "injury",
    "age",
    "policy_energy_observation",
    "rest_streak",
    "survival_steps",
    "last_action",
    "last_last_action",
    "action_switches",
    "action_oscillations",
    "episode0_switches",
    "episode0_oscillations",
    "gru_upkeep_cost",
    "cost_charge_scalar",
    "cost_charge_maturation_scalar",
    "patch_food_amount",
    "patch_fertility",
    "patch_cover",
    "patch_shelter",
    "patch_scent",
    "patch_waste",
    "patch_density",
    "patch_scent_emit_pending",
    "patch_waste_emit_pending",
    "patch_movement_cost",
    "mlp_pack",
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
    "drive_noise_samples_by_tick_slot",
    "drive_noise_samples_valid_by_tick_slot",
    "attack_bias",
    "action_temperature",
    "drive_logit_gain",
    "slot_tokens",
    "action_cache_actions",
    "action_cache_token",
    "action_cache_cursor",
    "action_cache_next",
    "pending_free_agent_ids",
    "pending_free_slots",
    "pending_free_count",
    "pcg64_state_words",
    "pcg64_buffer_words",
    "lineage_founder_id",
    "lineage_depth",
)

EVENT_TAPE_COLUMN_NAMES: Final[tuple[str, ...]] = (
    "event_tape_type",
    "event_tape_generation",
    "event_tape_tick",
    "event_tape_actor_id",
    "event_tape_target_id",
    "event_tape_value",
    "typed_lifecycle_event_event_type",
    "typed_lifecycle_event_generation",
    "typed_lifecycle_event_tick",
    "typed_lifecycle_event_sequence",
    "typed_lifecycle_event_actor_agent_id",
    "typed_lifecycle_event_target_agent_id",
    "typed_lifecycle_event_actor_slot",
    "typed_lifecycle_event_target_slot",
    "typed_lifecycle_event_parent_agent_id",
    "typed_lifecycle_event_child_agent_id",
    "typed_lifecycle_event_lineage_founder_id",
    "typed_lifecycle_event_lineage_depth",
    "typed_lifecycle_event_subtype_code",
    "typed_lifecycle_event_reason_code",
    "typed_lifecycle_event_success",
    "typed_lifecycle_event_value",
)

_EVENT_TAPE_DTYPES: Final[tuple[torch.dtype, ...]] = (
    torch.int32,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.float32,
    torch.int32,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int32,
    torch.int32,
    torch.int32,
    torch.int8,
    torch.float32,
)

_REQUIRED_TENSORS: Final[tuple[str, ...]] = (
    *_OWNER_TENSOR_NAMES,
    *EVENT_TAPE_COLUMN_NAMES,
    "event_tape_active_count",
    "event_tape_status",
)


def _source_digest() -> str:
    digest = sha256()
    for source in HASH_SOURCES:
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_cuda_generation_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("Architecture B generation composition requires CUDA")
    source_digest = _source_digest()
    prebuilt_value = str(os.environ.get(_PREBUILT_ENV, "") or "").strip()
    if prebuilt_value:
        artifact = Path(prebuilt_value).expanduser().resolve()
        identity = validate_cuda_prebuilt_manifest(
            artifact,
            required_exports=_REQUIRED_EXTENSION_EXPORTS,
        )
        if (
            identity.manifest.get("capability_contract")
            != CUDA_GENERATION_PREBUILT_CAPABILITY_CONTRACT
        ):
            raise RuntimeError(
                "CUDA Architecture B prebuilt capability contract does not match the runtime"
            )
        records = identity.manifest.get("sources")
        if not isinstance(records, list) or len(records) != len(HASH_SOURCES):
            raise RuntimeError(
                "CUDA Architecture B prebuilt source inventory does not match the runtime"
            )
        for source, record in zip(HASH_SOURCES, records, strict=True):
            if (
                not isinstance(record, dict)
                or Path(str(record.get("path", ""))).name != source.name
                or str(record.get("sha256", ""))
                != sha256(source.read_bytes()).hexdigest()
            ):
                raise RuntimeError(
                    "CUDA Architecture B prebuilt source identity mismatch: "
                    f"{source.name}"
                )
        cache_key = f"prebuilt:{identity.artifact_sha256}"
        module = _MODULES.get(cache_key)
        if module is None:
            module = load_prebuilt_cuda_architecture_b_extension(
                identity.artifact,
                required_exports=_REQUIRED_EXTENSION_EXPORTS,
            )
            _MODULES[cache_key] = module
        return module

    module = _MODULES.get(source_digest)
    if module is not None:
        return module
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    module_name = os.environ.get("EIEL_CUDA_ARCHITECTURE_B_DEV_MODULE_NAME") or (
        f"eiel_cuda_architecture_b_generation_{source_digest[:16]}"
    )
    module = load(
        name=module_name,
        sources=[str(source) for source in COMPILE_SOURCES],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_cflags=[
            "-O2",
            "-ffp-contract=off",
            "-DEIEL_CUDA_ARCHITECTURE_B_NO_PYBIND",
            "-DEIEL_CUDA_ARCHITECTURE_B_NO_BIRTH_MICROCYCLE_PYBIND",
        ],
        extra_cuda_cflags=[
            "-O2",
            "--fmad=false",
            "-DEIEL_CUDA_ARCHITECTURE_B_NO_PYBIND",
            "-DEIEL_CUDA_ARCHITECTURE_B_NO_BIRTH_MICROCYCLE_PYBIND",
        ],
        extra_include_paths=[str(NATIVE), str(PERF)],
        with_cuda=True,
        verbose=bool(verbose),
    )
    _MODULES[source_digest] = module
    return module


@dataclass(frozen=True)
class _CudaGenerationNativeCall:
    module: Any
    arguments: tuple[Any, ...]
    config: CudaGenerationConfig


def _prepare_cuda_generation_call(
    tensors: Mapping[str, torch.Tensor],
    config: CudaGenerationConfig,
    *,
    deterministic: bool = True,
    policy_family: str = "fixed_mlp",
    birth_state: CudaGenerationBirthState | None = None,
    build_directory: str | Path | None = None,
) -> _CudaGenerationNativeCall:
    if policy_family != "fixed_mlp":
        raise RuntimeError(
            "Architecture B generation composition supports fixed_mlp only; GRU is unsupported"
        )
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.life_history_mode not in {0, 1}:
        raise ValueError("life_history_mode must be legacy=0 or integrated=1")
    life_history_tick_values = (
        config.life_history_maturity_start_ticks,
        config.life_history_maturity_rise_ticks,
        config.life_history_fertility_decline_start_ticks,
        config.life_history_fertility_decline_ticks,
        config.life_history_senescence_start_ticks,
        config.life_history_senescence_ramp_ticks,
        config.life_history_max_age_ticks,
    )
    if any(int(value) < 0 for value in life_history_tick_values):
        raise ValueError("life-history tick parameters must be non-negative")
    if not (
        math.isfinite(float(config.life_history_elder_fertility_floor))
        and 0.0 <= float(config.life_history_elder_fertility_floor) <= 1.0
    ):
        raise ValueError("life_history_elder_fertility_floor must be finite in [0, 1]")
    if not (
        math.isfinite(float(config.life_history_senescence_energy_cost))
        and float(config.life_history_senescence_energy_cost) >= 0.0
        and math.isfinite(float(config.life_history_senescence_health_cost))
        and float(config.life_history_senescence_health_cost) >= 0.0
    ):
        raise ValueError("life-history senescence costs must be finite and non-negative")
    if config.max_births_per_tick < 0:
        raise ValueError("max_births_per_tick must be non-negative")
    if config.enable_biomes:
        raise RuntimeError(
            "Architecture B resident food spawn does not support biome weighting"
        )
    food_spawn_requested = (
        config.max_food > 0
        and config.food_spawn_per_step * config.food_spawn_mult > 0.0
    )
    if food_spawn_requested and not config.env_policy_alias:
        raise RuntimeError(
            "Architecture B resident food spawn requires the accepted env/policy RNG alias"
        )
    if config.generation_turnover_enabled:
        if config.population_size <= 0:
            raise ValueError("resident turnover requires a positive population_size")
        if not 0.0 <= config.cull_fraction <= 1.0:
            raise ValueError("resident turnover cull_fraction must be in [0, 1]")
        if not 0.0 <= config.parent_fraction <= 1.0:
            raise ValueError("resident turnover parent_fraction must be in [0, 1]")
        if config.lifecycle_mode not in {0, 1}:
            raise ValueError(
                "resident lifecycle_mode must be legacy=0 or hybrid-overlapping=1"
            )
        if config.turnover_selection_mode not in {0, 1}:
            raise ValueError(
                "resident turnover_selection_mode must be fitness=0 or "
                "uniform-neutral=1"
            )
    birth_enabled = config.max_births_per_tick > 0
    if (birth_enabled or config.generation_turnover_enabled) and birth_state is None:
        raise ValueError(
            "resident births or turnover require prepared generation birth state"
        )
    resident_lifecycle_state_enabled = birth_state is not None
    if food_spawn_requested and not resident_lifecycle_state_enabled:
        raise ValueError(
            "resident food spawn requires prepared policy RNG semantic state"
        )
    validation_tensors = dict(tensors)
    if birth_state is not None:
        validation_tensors.update(birth_state.identity_state)
    missing = [name for name in _REQUIRED_TENSORS if name not in validation_tensors]
    if missing:
        raise ValueError(
            f"generation tensor bank is missing required tensors: {missing}"
        )
    selected = [validation_tensors[name] for name in _REQUIRED_TENSORS]
    reference = selected[0]
    if not reference.is_cuda:
        raise ValueError("generation tensor bank must be CUDA-resident")
    for name, tensor in zip(_REQUIRED_TENSORS, selected, strict=True):
        if not tensor.is_cuda or tensor.device != reference.device:
            raise ValueError(
                f"generation tensor {name!r} must share the CUDA owner device"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"generation tensor {name!r} must be contiguous")
    event_columns = [tensors[name] for name in EVENT_TAPE_COLUMN_NAMES]
    event_capacity = event_columns[0].numel()
    for name, tensor, dtype in zip(
        EVENT_TAPE_COLUMN_NAMES,
        event_columns,
        _EVENT_TAPE_DTYPES,
        strict=True,
    ):
        if tensor.ndim != 1 or tensor.numel() != event_capacity:
            raise ValueError(
                f"generation event tape column {name!r} has an invalid capacity"
            )
        if tensor.dtype != dtype:
            raise ValueError(
                f"generation event tape column {name!r} must have dtype {dtype}"
            )
    active_count = tensors["event_tape_active_count"]
    event_status = tensors["event_tape_status"]
    if active_count.dtype != torch.int64 or tuple(active_count.shape) != (1,):
        raise ValueError("event_tape_active_count must be resident int64[1]")
    if event_status.dtype != torch.int32 or tuple(event_status.shape) != (1,):
        raise ValueError("event_tape_status must be resident int32[1]")
    if not deterministic:
        state_words = tensors["pcg64_state_words"]
        buffer_words = tensors["pcg64_buffer_words"]
        if state_words.dtype != torch.uint64 or tuple(state_words.shape) != (4,):
            raise ValueError(
                "stochastic generation requires four resident PCG64 uint64 state words"
            )
        if buffer_words.dtype != torch.int64 or tuple(buffer_words.shape) != (2,):
            raise ValueError(
                "stochastic generation requires two resident PCG64 int64 buffer words"
            )

    if birth_state is not None:
        if birth_state.receipt.steps != config.steps:
            raise ValueError(
                "birth state generation length disagrees with the generation config"
            )
        if (
            birth_enabled
            and birth_state.receipt.max_births_per_tick != config.max_births_per_tick
        ):
            raise ValueError(
                "birth state per-tick limit disagrees with the generation config"
            )
        if birth_state.receipt.target_device != str(reference.device):
            raise ValueError("birth state must share the generation CUDA device")
        execution_tensors = dict(tensors)
        execution_tensors.update(birth_state.identity_state)
        event_columns = list(birth_state.event_tape_columns)
        active_count = birth_state.event_tape_active_count
        event_status = birth_state.event_tape_status
        identity_state = [
            birth_state.identity_state[name] for name in IDENTITY_STATE_FIELDS
        ]
        scalar_rng_state = [
            birth_state.scalar_rng_state[name] for name in SCALAR_RNG_STATE_FIELDS
        ]
        genome_int32 = [birth_state.genome_state[name] for name in GENOME_INT32_FIELDS]
        genome_float32 = [
            birth_state.genome_state[name] for name in GENOME_FLOAT32_FIELDS
        ]
        drive_state = [birth_state.drive_state[name] for name in DRIVE_STATE_FIELDS]
        ordinary_policy_state = [
            birth_state.policy_state[name] for name in _ORDINARY_POLICY_STATE_FIELDS
        ]
        turnover_policy_state = [
            birth_state.turnover_policy_state[name] for name in POLICY_STATE_FIELDS
        ]
        policy_mem_dim = birth_state.policy_mem_dim
        policy_use_gru = birth_state.policy_use_gru
        planner_state = birth_state.planner_pcg64_state_words
        planner_buffer = birth_state.planner_pcg64_buffer_words
        policy_np_semantic_draws = birth_state.policy_np_semantic_draws
        metrics_np_semantic_draws = birth_state.metrics_np_semantic_draws

        alias_sources = {
            **{
                name: name
                for name in IDENTITY_STATE_FIELDS
                if name
                not in {
                    "agent_id_to_slot",
                    "slot_free_queue",
                    "slot_free_queue_count",
                    "population_next_id",
                }
            },
            **{name: name for name in GENOME_INT32_FIELDS},
            **{name: name for name in GENOME_FLOAT32_FIELDS},
            "pcg64_state_words": "population_pcg64_state_words",
            "pcg64_buffer_words": "population_pcg64_buffer_words",
            "mlp_pack": "mlp_pack",
            "hmem": "hmem",
            "policy_mem_dim": "policy_mem_dim",
            "policy_use_gru": "policy_use_gru",
        }
        prepared_aliases = {
            **birth_state.identity_state,
            **birth_state.genome_state,
            **birth_state.drive_state,
            "mlp_pack": birth_state.policy_state["mlp_pack"],
            "hmem": birth_state.policy_state["hmem"],
            "policy_mem_dim": birth_state.policy_mem_dim,
            "policy_use_gru": birth_state.policy_use_gru,
        }
        for prepared_name, source_name in alias_sources.items():
            prepared = prepared_aliases.get(prepared_name)
            source = tensors.get(source_name)
            if prepared is None or source is None:
                continue
            same_view = (
                prepared.data_ptr() == source.data_ptr()
                and prepared.storage_offset() == source.storage_offset()
                and prepared.shape == source.shape
                and prepared.stride() == source.stride()
                and prepared.dtype == source.dtype
            )
            if not same_view:
                raise ValueError(
                    "prepared birth state is not an authoritative alias for "
                    f"{source_name}"
                )
    else:
        execution_tensors = dict(tensors)
        identity_state = []
        scalar_rng_state = []
        genome_int32 = []
        genome_float32 = []
        drive_state = []
        ordinary_policy_state = []
        turnover_policy_state = []
        empty = torch.empty((0,), dtype=torch.int64, device=reference.device)
        policy_mem_dim = empty
        policy_use_gru = empty
        planner_state = empty
        planner_buffer = empty
        policy_np_semantic_draws = empty
        metrics_np_semantic_draws = torch.zeros(
            (1,), dtype=torch.int64, device=reference.device
        )

    if birth_state is not None:
        capacity = int(execution_tensors["alive_mask"].numel())
        if (
            config.generation_turnover_enabled
            and birth_state.receipt.turnover_refill_capacity < capacity
        ):
            raise ValueError(
                "turnover birth state must reserve at least the slot capacity"
            )
        optional_turnover_defaults = {
            "early_injury_gained": (torch.float32, 0.0),
            "chemistry_observation_fidelity": (torch.float32, 1.0),
            "chemistry_observation_bias": (torch.float32, 0.0),
        }
        for name, (dtype, fill_value) in optional_turnover_defaults.items():
            if name not in execution_tensors:
                execution_tensors[name] = torch.full(
                    (capacity,),
                    fill_value,
                    dtype=dtype,
                    device=reference.device,
                )
        missing_turnover = [
            name for name in TURNOVER_STATE_FIELDS if name not in execution_tensors
        ]
        if missing_turnover:
            raise ValueError(
                f"generation turnover state is missing tensors: {missing_turnover}"
            )
        for name, dtype in zip(
            TURNOVER_STATE_FIELDS,
            _TURNOVER_STATE_DTYPES,
            strict=True,
        ):
            tensor = execution_tensors[name]
            if (
                not tensor.is_cuda
                or tensor.device != reference.device
                or not tensor.is_contiguous()
                or tensor.dtype != dtype
                or tensor.ndim != 1
                or int(tensor.numel()) != capacity
            ):
                raise ValueError(
                    f"generation turnover tensor {name!r} must be contiguous "
                    f"{dtype} CUDA[{capacity}]"
                )
        turnover_state = [execution_tensors[name] for name in TURNOVER_STATE_FIELDS]
    else:
        turnover_state = []

    if config.active_mechanics_enabled:
        auxiliary_names = (
            "chemistry",
            "runner_pcg64_state_words",
            "runner_pcg64_buffer_words",
        )
        missing_auxiliary = [
            name for name in auxiliary_names if name not in execution_tensors
        ]
        if missing_auxiliary:
            raise ValueError(
                "active generation tensor bank is missing resident chemistry/RNG "
                f"state: {missing_auxiliary}"
            )
        chemistry = execution_tensors["chemistry"]
        runner_state = execution_tensors["runner_pcg64_state_words"]
        runner_buffer = execution_tensors["runner_pcg64_buffer_words"]
    else:
        chemistry = torch.empty((0, 0), dtype=torch.float32, device=reference.device)
        runner_state = torch.empty((0,), dtype=torch.uint64, device=reference.device)
        runner_buffer = torch.empty((0,), dtype=torch.int64, device=reference.device)

    module = load_cuda_generation_extension(build_directory=build_directory)
    arguments = (
        *(execution_tensors[name] for name in _OWNER_TENSOR_NAMES),
        chemistry,
        runner_state,
        runner_buffer,
        metrics_np_semantic_draws,
        event_columns,
        active_count,
        event_status,
        identity_state,
        scalar_rng_state,
        genome_int32,
        policy_mem_dim,
        policy_use_gru,
        genome_float32,
        drive_state,
        ordinary_policy_state,
        turnover_policy_state,
        turnover_state,
        planner_state,
        planner_buffer,
        policy_np_semantic_draws,
        bool(resident_lifecycle_state_enabled),
        bool(birth_enabled),
        bool(config.generation_turnover_enabled),
        int(config.steps),
        bool(config.patch_enabled),
        bool(deterministic),
        int(config.generation),
        int(config.start_tick),
        int(config.view_radius),
        float(config.max_energy),
        float(config.start_health),
        float(config.patch_food_amount_max),
        float(config.patch_scent_max),
        float(config.patch_waste_max),
        int(config.patch_density_radius),
        bool(config.patch_density_include_self),
        bool(config.patch_density_normalize),
        float(config.patch_movement_cost_min),
        float(config.patch_movement_cost_max),
        int(config.hidden_size),
        int(config.action_count),
        int(config.cache_width),
        int(config.rng_seed),
        int(config.rng_counter),
        float(config.w_hunger_gather),
        float(config.w_pain_rest),
        float(config.w_fatigue_rest),
        float(config.w_pain_attack),
        float(config.w_fear_attack),
        float(config.move_cost),
        float(config.gather_cost),
        float(config.rest_cost),
        float(config.attack_cost),
        float(config.action_cost_mult),
        float(config.rest_cost_mult),
        float(config.food_energy),
        int(config.max_food),
        float(config.food_spawn_per_step),
        float(config.food_spawn_mult),
        bool(config.enable_biomes),
        float(config.patch_food_amount_initial_food_cell_value),
        float(config.patch_food_gather_amount),
        float(config.patch_food_presence_threshold),
        float(config.patch_scent_move_emit),
        float(config.patch_waste_move_emit),
        float(config.patch_scent_gather_emit),
        float(config.patch_waste_consume_emit),
        bool(config.patch_movement_cost_apply_enable),
        float(config.patch_movement_cost_energy_scale),
        float(config.patch_scent_decay_per_step),
        float(config.patch_waste_decay_per_step),
        float(config.patch_scent_presence_emit),
        float(config.patch_waste_presence_emit),
        float(config.patch_waste_density_emit_scale),
        float(config.patch_food_regrow_per_step),
        bool(config.patch_food_regrow_use_fertility),
        bool(config.patch_food_regrow_cap_by_fertility),
        bool(config.patch_world_waste_consequence_enable),
        bool(config.patch_waste_regrowth_penalty_enable),
        float(config.patch_waste_regrowth_penalty_scale),
        float(config.patch_waste_regrowth_penalty_max),
        bool(config.patch_density_regrowth_penalty_enable),
        float(config.patch_density_regrowth_penalty_scale),
        float(config.patch_density_regrowth_penalty_max),
        float(config.patch_regrowth_min_multiplier),
        bool(config.active_mechanics_enabled),
        bool(config.enable_rest_charge),
        int(config.rest_streak_free),
        float(config.rest_charge_energy),
        float(config.rest_charge_growth),
        float(config.rest_deficit_to_health),
        int(config.developmental_maturity_age),
        int(config.life_history_mode),
        int(config.life_history_maturity_start_ticks),
        int(config.life_history_maturity_rise_ticks),
        int(config.life_history_fertility_decline_start_ticks),
        int(config.life_history_fertility_decline_ticks),
        float(config.life_history_elder_fertility_floor),
        int(config.life_history_senescence_start_ticks),
        int(config.life_history_senescence_ramp_ticks),
        float(config.life_history_senescence_energy_cost),
        float(config.life_history_senescence_health_cost),
        int(config.life_history_max_age_ticks),
        int(config.episode0_k),
        float(config.drive_hunger_gain),
        float(config.drive_hunger_decay),
        float(config.drive_pain_gain),
        float(config.drive_pain_decay),
        float(config.drive_fatigue_gain),
        float(config.drive_fatigue_rest_relief),
        float(config.drive_fatigue_decay),
        float(config.drive_fear_gain),
        float(config.drive_fear_decay),
        float(config.reproduce_parent_energy_share),
        float(config.reproduce_min_energy_buffer),
        int(config.reproduce_cooldown_steps),
        int(config.genome_sensor_radius_max),
        float(config.injury_repro_k_injury),
        float(config.injury_repro_k_scar),
        float(config.injury_birth_fail_scar),
        float(config.injury_birth_fail_injury),
        int(config.max_births_per_tick),
        bool(config.reproduce_requires_mate),
        int(config.lifecycle_mode),
        int(config.population_size),
        float(config.cull_fraction),
        float(config.parent_fraction),
        float(config.start_energy),
        int(config.early_life_steps),
        float(config.early_life_weight),
        bool(config.enable_combat),
        float(config.coherence_thrashing_weight),
        float(config.coherence_drive_volatility_weight),
        float(config.coherence_selfharm_weight),
        float(config.coherence_pred_err_weight),
        bool(config.env_policy_alias),
    )
    return _CudaGenerationNativeCall(
        module=module,
        arguments=arguments,
        config=config,
    )


def _cuda_generation_result(
    outputs: Any,
    config: CudaGenerationConfig,
) -> CudaGenerationResult:
    expected_patch_width = len(PATCH_CONTEXT_P2_STAT_FIELDS)
    if config.patch_enabled and tuple(outputs[10].shape) != (
        config.steps,
        expected_patch_width,
    ):
        raise RuntimeError(
            "resident generation returned an invalid Patch P2 statistics surface"
        )
    return CudaGenerationResult(*outputs)


def execute_cuda_generation(
    tensors: Mapping[str, torch.Tensor],
    config: CudaGenerationConfig,
    *,
    deterministic: bool = True,
    policy_family: str = "fixed_mlp",
    birth_state: CudaGenerationBirthState | None = None,
    build_directory: str | Path | None = None,
) -> CudaGenerationResult:
    call = _prepare_cuda_generation_call(
        tensors,
        config,
        deterministic=deterministic,
        policy_family=policy_family,
        birth_state=birth_state,
        build_directory=build_directory,
    )
    outputs = call.module.eiel_cuda_architecture_b_generation_v3(*call.arguments)
    return _cuda_generation_result(outputs, config)


def execute_cuda_modular_generation(
    tensors: Mapping[str, torch.Tensor],
    config: CudaGenerationConfig,
    *,
    enabled: bool = False,
    modular_tensors: Any = None,
    profile_id: Any = None,
    deterministic: bool = True,
    policy_family: str = "fixed_mlp",
    birth_state: CudaGenerationBirthState | None = None,
    build_directory: str | Path | None = None,
) -> CudaGenerationResult:
    """Run the explicit Phase E single-cell generation owner.

    Disabled mode delegates to generation-v3 before inspecting either modular
    argument.  Enabled mode fails closed unless the complete resident 57-plane
    bank and the first-live fixed policy/lifecycle contract are exact.
    """

    if enabled is False:
        return execute_cuda_generation(
            tensors,
            config,
            deterministic=deterministic,
            policy_family=policy_family,
            birth_state=birth_state,
            build_directory=build_directory,
        )
    if enabled is not True:
        raise ValueError("enabled must be an exact bool")
    if not isinstance(tensors, Mapping):
        raise TypeError("generation tensors must be a mapping")
    alive_mask = tensors.get("alive_mask")
    if not isinstance(alive_mask, torch.Tensor) or alive_mask.ndim != 1:
        raise ValueError("alive_mask must be a resident rank-one tensor")
    entry = prepare_modular_generation_entry(
        modular_tensors,
        enabled=True,
        agent_capacity=int(alive_mask.shape[0]),
        profile_id=profile_id,
        policy_family=policy_family,
        action_count=int(config.action_count),
        observation_width=27,
        cache_width=int(config.cache_width),
        policy_stride=1,
        development_mature_start_step=(config.structural_development_mature_start_step),
        development_late_start_step=(config.structural_development_late_start_step),
        max_births=int(config.max_births_per_tick),
        turnover_enabled=bool(config.generation_turnover_enabled),
        device=alive_mask.device,
        allow_cpu_validation=False,
    )
    if entry is None:
        raise RuntimeError("enabled modular generation entry disappeared")
    call = _prepare_cuda_generation_call(
        tensors,
        config,
        deterministic=deterministic,
        policy_family=policy_family,
        birth_state=birth_state,
        build_directory=build_directory,
    )
    function = getattr(
        call.module,
        "eiel_cuda_architecture_b_generation_phase_e_v1",
        None,
    )
    if function is None:
        raise RuntimeError(
            "Phase E modular generation export is unavailable; hidden v3 fallback is forbidden"
        )
    outputs = function(
        *call.arguments,
        list(entry.tensor_arguments),
        *entry.config_scalar_values,
    )
    return _cuda_generation_result(outputs, config)


__all__ = [
    "CUDA_GENERATION_CAPABILITIES",
    "CUDA_GENERATION_PREBUILT_CAPABILITY_CONTRACT",
    "CUDA_MODULAR_GENERATION_CAPABILITIES",
    "CudaGenerationConfig",
    "CudaGenerationResult",
    "EVENT_TAPE_COLUMN_NAMES",
    "TURNOVER_STATE_FIELDS",
    "execute_cuda_generation",
    "execute_cuda_modular_generation",
    "load_cuda_generation_extension",
]
