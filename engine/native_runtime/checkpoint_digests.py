from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping

import numpy as np
import torch

from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_METADATA_KEY,
    MODULAR_STORAGE_PROJECTION_PREFIX,
    build_modular_storage_layout,
    modular_storage_from_projection,
)
from perf.cuda_diploid_state_bank_v1 import (
    CHECKPOINT_TENSOR_NAMES as DIPLOID_CHECKPOINT_TENSOR_NAMES,
)

from .lifecycle_events import validated_event_storage_layout


SCHEMA_VERSION = "eiel_checkpoint_digest_layers_v1"
CANONICAL_STORAGE_EQUIVALENT_PROJECTION_SCHEMAS = frozenset(
    {
        "cxxbackbone10_checkpoint_projection_v1",
        "cxxbackbone10_checkpoint_projection_v2",
        "cxxbackbone10_resident_state_owner_v1",
    }
)
FIELD_POLICY_SCHEMA_VERSION = "eiel_checkpoint_field_digest_policy_v1"
SOURCE_PROJECTION_SCHEMA_VERSION = "eiel_checkpoint_source_projection_digest_v1"
SOURCE_ACTION_CACHE_SCHEMA_VERSION = "eiel_checkpoint_source_action_cache_v1"
SOURCE_RNG_SCHEMA_VERSION = "eiel_checkpoint_source_rng_v1"
SOURCE_AUTHORITY_SCHEMA_VERSION = "eiel_checkpoint_source_authority_v1"
SOURCE_PATCH_AUTHORITY_SCHEMA_VERSION = "eiel_checkpoint_source_patch_authority_v1"

SOURCE_ACTION_CACHE_FIELDS = (
    "action_cache_actions",
    "action_cache_token",
    "action_cache_cursor",
    "action_cache_next",
)
SOURCE_AUTHORITY_CURSOR_METADATA_FIELDS = (
    "generation",
    "tick",
    "configured_steps_per_generation",
    "agent_grid_runner_phase",
    "agent_grid_authority_epoch",
)
SOURCE_RNG_REQUIRED_METADATA_FIELDS = (
    "native_runtime_numpy_env_state",
    "native_runtime_numpy_runner_state",
    "native_runtime_numpy_policy_state",
    "native_runtime_env_policy_numpy_alias",
    "native_runtime_numpy_population_state",
    "native_runtime_python_random_state",
    "native_runtime_rng_manager_stream_indices",
)
SOURCE_RNG_OPTIONAL_METADATA_FIELDS = (
    "native_runtime_rng_state",
    "native_runtime_numpy_birth_spawn_env_state",
)
SOURCE_RNG_TENSOR_FIELDS = (
    "rng_manager_mutation_torch_cpu_state",
)
SOURCE_PROJECTION_REQUIRED_TOP_LEVEL_FIELDS = (
    "schema_version",
    "metadata",
    "tensors",
    "tensor_manifest",
    "restore_count",
    "generation_calls",
    "event_cursor",
    "runtime_event_cursor",
    "last_applied_actions",
    "last_moved",
    "last_gathered",
    "last_policy_refill_calls",
    "last_policy_refill_rows",
    "pending_generation_free_agent_ids",
    "pending_generation_free_slots",
    "last_world",
)
SOURCE_PATCH_PROFILES = frozenset({"P0", "P1", "P2"})
SOURCE_PATCH_REQUIRED_WORLD_TENSOR_FIELDS = (
    "patch_food_amount",
    "patch_fertility",
    "patch_cover",
    "patch_shelter",
    "patch_movement_cost",
    "patch_scent",
    "patch_scent_emit_pending",
    "patch_waste",
    "patch_waste_emit_pending",
)
SOURCE_PATCH_P2_REQUIRED_WORLD_TENSOR_FIELDS = ("patch_recovery_activity",)
SOURCE_PATCH_REQUIRED_ROW_TENSOR_FIELDS = ("patch_habitat_lineage_id",)
SOURCE_PATCH_DERIVED_TENSOR_FIELDS = ("patch_visibility", "patch_density")
SOURCE_WDL_STORAGE_TENSOR_DTYPES = {
    "patch_delayed_hazard_cue": torch.float32,
    "patch_delayed_hazard_timer": torch.int32,
    "patch_delayed_hazard_active": torch.float32,
    "patch_delayed_hazard_intensity": torch.float32,
    "patch_delayed_hazard_age": torch.int32,
    "patch_delayed_hazard_event_id": torch.int64,
    "patch_delayed_hazard_prior_cue": torch.int8,
}
SOURCE_PATCH_REQUIRED_METADATA_FIELDS = (
    "native_runtime_patch_profile",
    "native_runtime_patch_profile_scope",
    "native_runtime_patch_contract_schema_version",
    "patch_world_enable",
    "patch_world_schema_version",
    "patch_world_observation_enable",
    "patch_world_observation_schema_version",
    "patch_world_diagnostics_enable",
    "patch_world_lineage_descriptors_enable",
    "patch_world_descriptor_schema_version",
    "patch_movement_cost_apply_enable",
    "patch_world_waste_consequence_enable",
    "patch_array_import_phase",
    "patch_visibility_authority",
    "patch_visibility_from_cover_enable",
    "patch_visibility_default",
    "patch_visibility_min",
    "patch_visibility_max",
    "patch_density_authority",
    "patch_density_radius",
    "patch_density_include_self",
    "patch_density_normalize",
    "patch_food_presence_threshold",
    "patch_food_binary_projection",
    "patch_pending_emission_phase",
)
GENOME_V2_CHECKPOINT_TENSOR_NAMES = (
    "genome_v2_alleles_q10",
    "genome_v2_modifiers_q10",
    "genome_v2_products_q10",
    "genome_v2_phenotypes_q10",
    "genome_v2_expression_cost_q10",
    "genome_v2_active_product_count",
    "genome_v2_saturated_product_count",
)
EED_CHECKPOINT_TENSOR_NAMES = (
    "eed_resource_a_amount",
    "eed_resource_b_amount",
    "eed_resource_a_capacity",
    "eed_resource_b_capacity",
    "eed_resource_a_renewal",
    "eed_resource_b_renewal",
    "eed_resource_a_pending_impact",
    "eed_resource_b_pending_impact",
    "eed_resource_a_impact_load",
    "eed_resource_b_impact_load",
    "respec0b_resource_class",
    "respec0b_resource_active_y",
    "respec0b_resource_active_x",
    "respec0b_resource_active_class",
    "respec0b_resource_handling_progress",
)


class DigestLayer(StrEnum):
    TRAJECTORY_STATE = "trajectory_state"
    CANONICAL_CHECKPOINT_STORAGE = "canonical_checkpoint_storage"
    DIAGNOSTIC_PROJECTION = "diagnostic_projection"


@dataclass(frozen=True)
class FieldDigestPolicy:
    field: str
    primary_layer: DigestLayer
    authoritative_phase: str
    live_dead_free_behavior: str
    reconstructible: bool
    trajectory_digest: bool
    canonical_storage_digest: bool
    diagnostic_projection_digest: bool
    diagnostic_only: bool
    evidence: str
    restore_consequence: str
    public_output_consequence: str
    canonicalization: str = "none"

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = FIELD_POLICY_SCHEMA_VERSION
        payload["primary_layer"] = str(self.primary_layer)
        payload["authority_classification"] = (
            "diagnostic_only"
            if self.diagnostic_only
            else "reconstructible_derived"
            if self.reconstructible and not self.includes(self.primary_layer)
            else "authoritative"
        )
        payload["primary_source_layer"] = str(self.primary_layer)
        payload["included_digest_layers"] = [
            str(layer) for layer in DigestLayer if self.includes(layer)
        ]
        payload["canonicalizer"] = self.canonicalization
        return payload

    def includes(self, layer: DigestLayer) -> bool:
        if layer is DigestLayer.TRAJECTORY_STATE:
            return self.trajectory_digest
        if layer is DigestLayer.CANONICAL_CHECKPOINT_STORAGE:
            return self.canonical_storage_digest
        if layer is DigestLayer.DIAGNOSTIC_PROJECTION:
            return self.diagnostic_projection_digest
        raise ValueError(f"unknown digest layer: {layer}")


def _policy(
    field: str,
    *,
    primary_layer: DigestLayer,
    authoritative_phase: str,
    live_dead_free_behavior: str,
    reconstructible: bool,
    trajectory_digest: bool,
    canonical_storage_digest: bool,
    diagnostic_projection_digest: bool,
    diagnostic_only: bool,
    evidence: str,
    restore_consequence: str,
    public_output_consequence: str,
    canonicalization: str = "none",
) -> FieldDigestPolicy:
    return FieldDigestPolicy(
        field=field,
        primary_layer=primary_layer,
        authoritative_phase=authoritative_phase,
        live_dead_free_behavior=live_dead_free_behavior,
        reconstructible=reconstructible,
        trajectory_digest=trajectory_digest,
        canonical_storage_digest=canonical_storage_digest,
        diagnostic_projection_digest=diagnostic_projection_digest,
        diagnostic_only=diagnostic_only,
        evidence=evidence,
        restore_consequence=restore_consequence,
        public_output_consequence=public_output_consequence,
        canonicalization=canonicalization,
    )


def _live_row_policy(field: str, evidence: str) -> FieldDigestPolicy:
    return _policy(
        field,
        primary_layer=DigestLayer.TRAJECTORY_STATE,
        authoritative_phase="equivalent runner phase; live rows only are trajectory-authoritative",
        live_dead_free_behavior=(
            "live rows affect future ticks; dead/free row payload is storage or diagnostic "
            "unless a separate mask, slot map, or queue marks it active"
        ),
        reconstructible=False,
        trajectory_digest=True,
        canonical_storage_digest=True,
        diagnostic_projection_digest=True,
        diagnostic_only=False,
        evidence=evidence,
        restore_consequence="restore must preserve live-row values exactly",
        public_output_consequence="live-row drift can affect future public rows after resume",
        canonicalization="field-policy row masking only; never blanket-clear every dead/free tensor",
    )


def _event_tape_policy(field: str) -> FieldDigestPolicy:
    return _policy(
        field,
        primary_layer=DigestLayer.TRAJECTORY_STATE,
        authoritative_phase="event tape active prefix at the checkpoint cursor",
        live_dead_free_behavior="active prefix is authoritative; inactive tail is storage padding",
        reconstructible=False,
        trajectory_digest=True,
        canonical_storage_digest=True,
        diagnostic_projection_digest=True,
        diagnostic_only=False,
        evidence="residual inventory shows native typed event rows versus reference empty projection",
        restore_consequence="active prefix loss or duplication changes resumed event semantics",
        public_output_consequence="event rows and lineage/lifecycle summaries can duplicate or omit rows",
        canonicalization="active_event_count prefix exact; inactive tail normalized by policy only",
    )


def _exact_state_policy(field: str, evidence: str) -> FieldDigestPolicy:
    return _policy(
        field,
        primary_layer=DigestLayer.TRAJECTORY_STATE,
        authoritative_phase="exact checkpoint state at equivalent runner phase",
        live_dead_free_behavior="field is authoritative as stored and is not row-mask canonicalized",
        reconstructible=False,
        trajectory_digest=True,
        canonical_storage_digest=True,
        diagnostic_projection_digest=True,
        diagnostic_only=False,
        evidence=evidence,
        restore_consequence="restore must preserve the field exactly",
        public_output_consequence="field drift can alter later trajectory, events, or public summaries",
        canonicalization="preserve_exact_stored",
    )


def _derived_table_policy(field: str, evidence: str) -> FieldDigestPolicy:
    return _policy(
        field,
        primary_layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
        authoritative_phase="precomputed table for the current checkpoint epoch",
        live_dead_free_behavior="table is meaningful only when live rows require the derived samples",
        reconstructible=True,
        trajectory_digest=False,
        canonical_storage_digest=False,
        diagnostic_projection_digest=False,
        diagnostic_only=False,
        evidence=evidence,
        restore_consequence="restore may regenerate or normalize the table only when deterministic",
        public_output_consequence="no public-output consequence if regenerated from authoritative RNG state",
        canonicalization="absent-or-deterministically-regenerated",
    )


def _derived_world_view_policy(
    field: str,
    *,
    source: str,
) -> FieldDigestPolicy:
    return _policy(
        field,
        primary_layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
        authoritative_phase="deterministic Patch boundary view",
        live_dead_free_behavior="world-shaped view has no live/dead row authority",
        reconstructible=True,
        trajectory_digest=False,
        canonical_storage_digest=False,
        diagnostic_projection_digest=False,
        diagnostic_only=False,
        evidence=f"Patch source authority rebuilds this view from {source}",
        restore_consequence=f"restore must rebuild the view from {source}",
        public_output_consequence=(
            "no consequence when rebuilt exactly from authoritative checkpoint state"
        ),
        canonicalization=f"rebuild_exact_from_{source}",
    )


def _validate_field_digest_policy(policy: FieldDigestPolicy) -> None:
    if not isinstance(policy, FieldDigestPolicy) or not policy.field:
        raise ValueError("checkpoint digest policy requires a nonempty field")
    included = tuple(layer for layer in DigestLayer if policy.includes(layer))
    if policy.diagnostic_only:
        if policy.primary_layer is not DigestLayer.DIAGNOSTIC_PROJECTION or included:
            raise ValueError(
                f"diagnostic-only checkpoint field {policy.field!r} cannot be included in digest layers"
            )
        return
    if not included:
        if not policy.reconstructible:
            raise ValueError(
                f"checkpoint field {policy.field!r} excludes every digest layer without being reconstructible"
            )
        return
    if not policy.includes(policy.primary_layer):
        raise ValueError(
            f"checkpoint field {policy.field!r} excludes its primary digest layer {policy.primary_layer}"
        )


def register_field_digest_policy(
    registry: dict[str, FieldDigestPolicy],
    policy: FieldDigestPolicy,
) -> None:
    _validate_field_digest_policy(policy)
    if policy.field in registry:
        raise KeyError(f"duplicate checkpoint digest policy registration: {policy.field}")
    registry[policy.field] = policy


DEFAULT_FIELD_POLICIES: dict[str, FieldDigestPolicy] = {}
for _initial_policy in (
    _policy(
        "agent_grid",
        primary_layer=DigestLayer.TRAJECTORY_STATE,
        authoritative_phase="stored grid at an equivalent world/runner phase",
        live_dead_free_behavior="grid cells are authoritative as stored; not row-mask canonicalized",
        reconstructible=False,
        trajectory_digest=True,
        canonical_storage_digest=True,
        diagnostic_projection_digest=True,
        diagnostic_only=False,
        evidence="global live-row rebuild broke flat and patch partial-generation resume fingerprints",
        restore_consequence="restore must preserve the stored phase representation exactly",
        public_output_consequence="grid drift can affect movement, births, deaths, and later public rows",
        canonicalization="preserve_exact_stored_grid",
    ),
    _derived_table_policy(
        "drive_noise_samples_by_tick_slot",
        "safe repair omits the table when no live row has positive drive noise",
    ),
    _derived_table_policy(
        "drive_noise_samples_valid_by_tick_slot",
        "validity companion for the derived drive-noise sample table; authoritative RNG state can regenerate it",
    ),
    _exact_state_policy(
        "food_grid",
        "flat-world food grid is trajectory state for movement/gather continuation",
    ),
    _exact_state_policy(
        "biome_grid",
        "flat-world biome grid is trajectory state for food spawn and world continuation when enabled",
    ),
    _exact_state_policy(
        "hazard_rate_map",
        "flat-world hazard-rate map is trajectory state for biome hazard continuation when enabled",
    ),
):
    register_field_digest_policy(DEFAULT_FIELD_POLICIES, _initial_policy)

for _field in (
    "agent_ids_by_alive_order",
    "agent_id_to_slot",
    "agent_id_to_slot_agent_ids",
    "agent_id_to_slot_slots",
    "alive_mask",
    "slot_free_queue",
    "slot_to_agent_id",
    "slot_tokens",
    "rng_manager_mutation_torch_cpu_state",
):
    register_field_digest_policy(
        DEFAULT_FIELD_POLICIES,
        _exact_state_policy(
            _field,
            "slot maps, tokens, liveness, and free queue order define checkpoint identity and reuse order",
        ),
    )

for _field in (
    "diploid_allele_storage",
    *DIPLOID_CHECKPOINT_TENSOR_NAMES,
    *GENOME_V2_CHECKPOINT_TENSOR_NAMES,
):
    register_field_digest_policy(
        DEFAULT_FIELD_POLICIES,
        _exact_state_policy(
            _field,
            "diploid and Genome V2 resident arrays are exact authoritative checkpoint state",
        ),
    )

for _field, _source in (
    ("patch_visibility", "patch_cover_and_visibility_metadata"),
    ("patch_density", "agent_grid_and_density_metadata"),
):
    register_field_digest_policy(
        DEFAULT_FIELD_POLICIES,
        _derived_world_view_policy(_field, source=_source),
    )

for _field in (
    "patch_cover",
    "patch_delayed_hazard_active",
    "patch_delayed_hazard_age",
    "patch_delayed_hazard_cue",
    "patch_delayed_hazard_event_id",
    "patch_delayed_hazard_intensity",
    "patch_delayed_hazard_prior_cue",
    "patch_delayed_hazard_timer",
    "patch_fertility",
    "patch_food_amount",
    "patch_movement_cost",
    "patch_prediction_demand_active",
    "patch_prediction_demand_age",
    "patch_prediction_demand_cue",
    "patch_prediction_demand_delay",
    "patch_prediction_demand_event_id",
    "patch_prediction_demand_intensity",
    "patch_prediction_demand_prior_cue",
    "patch_prediction_demand_timer",
    "patch_prediction_demand_transition_sign",
    "patch_prediction_demand_trend",
    "patch_recovery_activity",
    "patch_scent",
    "patch_scent_emit_pending",
    "patch_shelter",
    "patch_waste",
    "patch_waste_emit_pending",
):
    register_field_digest_policy(
        DEFAULT_FIELD_POLICIES,
        _exact_state_policy(
            _field,
            "patch/WDL world arrays are authoritative world state when present",
        ),
    )

for _field in EED_CHECKPOINT_TENSOR_NAMES:
    register_field_digest_policy(
        DEFAULT_FIELD_POLICIES,
        _exact_state_policy(
            _field,
            "EED resource and slot-cursor tensors are authoritative overlay state when present",
        ),
    )

for _field in (
    "action_cache_actions",
    "action_cache_cursor",
    "action_cache_next",
    "action_cache_token",
    "action_oscillations",
    "action_switches",
    "age",
    "attacks",
    "birth_reset_shock",
    "chemistry",
    "cost_charge_maturation_scalar",
    "cost_charge_scalar",
    "dev_stage_initialized",
    "drive_inertia",
    "drive_noise",
    "drives",
    "early_energy_collected",
    "early_energy_collected_exact",
    "early_injury_gained",
    "early_life_weight",
    "early_survival_steps",
    "energy",
    "policy_energy_observation",
    "energy_collected",
    "energy_collected_exact",
    "episode0_oscillations",
    "episode0_pred_err_count",
    "episode0_pred_err_sum",
    "episode0_selfharm_events",
    "episode0_switches",
    "gru_pack",
    "gru_upkeep_cost",
    "health",
    "hmem",
    "injury",
    "first_repro_step",
    "last_action",
    "last_energy_delta",
    "last_health_delta",
    "last_injury_delta",
    "last_last_action",
    "late_pred_err_count",
    "late_pred_err_sum",
    "lineage_birth_generation",
    "lineage_birth_tick",
    "lineage_depth",
    "lineage_founder_id",
    "offspring_count",
    "patch_habitat_lineage_id",
    "policy_mem_dim",
    "policy_use_gru",
    "pos_x",
    "pos_y",
    "prev_drive_delta",
    "pred_err_count",
    "pred_err_sum",
    "repro_cooldown",
    "rest_streak",
    "scar",
    "selfharm_events",
    "survival_steps",
    "genome_action_temperature",
    "genome_aggression_bias",
    "genome_drive_birth_noise",
    "genome_drive_inertia",
    "genome_drive_logit_gain",
    "genome_drive_noise",
    "genome_drive_sp_fatigue",
    "genome_drive_sp_fear",
    "genome_drive_sp_hunger",
    "genome_drive_sp_pain",
    "genome_directed_inheritance_channel_mask",
    "genome_exp_prior_noise",
    "genome_exp_prior_strength",
    "genome_exploration_bias",
    "genome_hidden_size",
    "genome_introspection_channel_mask",
    "genome_memory_dim",
    "genome_mutation_rate",
    "genome_pred_lr",
    "genome_reproduction_threshold",
    "genome_self_modification_target_mask",
    "genome_self_regulation_control_mask",
    "genome_sensor_channel_mask",
    "genome_sensor_radius",
    "mlp_pack",
    "policy_action_temperature",
    "policy_attack_bias",
    "policy_drive_logit_gain",
    "repro_threshold",
):
    register_field_digest_policy(
        DEFAULT_FIELD_POLICIES,
        _live_row_policy(
            _field,
            "residual inventory shows live-row or dead/free-row projection mismatches; live rows remain authoritative",
        ),
    )

for _field in (
    "adult_pred_err_ema",
    "adult_pred_err_ema_exact",
    "episode0_pred_err_sum_exact",
    "exp_dE",
    "exp_dInj",
    "late_pred_err_sum_exact",
    "pred_err_sum_exact",
    "pred_lr",
    "pred_lr_exact",
    "prediction_queue_action",
    "prediction_queue_count",
    "prediction_queue_due",
    "prediction_queue_energy",
    "prediction_queue_exp_de",
    "prediction_queue_exp_dinj",
    "prediction_queue_injury",
    "prediction_reset_count",
    "prediction_state_delta_abs_max",
    "prediction_state_delta_abs_max_exact",
    "prediction_state_delta_abs_sum",
    "prediction_state_delta_abs_sum_exact",
    "prediction_update_count",
):
    register_field_digest_policy(
        DEFAULT_FIELD_POLICIES,
        _live_row_policy(
            _field,
            "prediction state is authoritative for live-row continuation and resets",
        ),
    )

for _field in (
    "event_tape_actor_id",
    "event_tape_generation",
    "event_tape_target_id",
    "event_tape_tick",
    "event_tape_type",
    "event_tape_value",
):
    register_field_digest_policy(DEFAULT_FIELD_POLICIES, _event_tape_policy(_field))

for _modular_field in build_modular_storage_layout(agent_capacity=1).fields:
    _projected_field = f"{MODULAR_STORAGE_PROJECTION_PREFIX}{_modular_field.name}"
    register_field_digest_policy(
        DEFAULT_FIELD_POLICIES,
        _exact_state_policy(
            _projected_field,
            "Phase E fixed-envelope modular plane is exact authoritative checkpoint state",
        ),
    )

for _field in (
    "drive_volatility",
    "drive_jitter",
    "episode0_drive_jitter_sum",
):
    register_field_digest_policy(
        DEFAULT_FIELD_POLICIES,
        _policy(
            _field,
            primary_layer=DigestLayer.DIAGNOSTIC_PROJECTION,
            authoritative_phase="diagnostic accumulated float summary; drives and prev_drive_delta are authoritative state",
            live_dead_free_behavior="not used as a trajectory authority in checkpoint parity comparisons",
            reconstructible=True,
            trajectory_digest=False,
            canonical_storage_digest=False,
            diagnostic_projection_digest=False,
            diagnostic_only=True,
            evidence="legacy and native paths can differ by float accumulation order while drives and prev_drive_delta match",
            restore_consequence="restore preserves the tensor payload, but parity digest does not gate on this summary counter",
            public_output_consequence="excluded from public-output parity claims",
            canonicalization="excluded from digest layers; retained in projection payload",
        ),
    )

for _field in (
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
):
    register_field_digest_policy(DEFAULT_FIELD_POLICIES, _event_tape_policy(_field))


def field_policy_manifest(
    fields: list[str] | tuple[str, ...] | set[str] | None = None,
    *,
    policies: Mapping[str, FieldDigestPolicy] | None = None,
) -> dict[str, Any]:
    resolved = dict(DEFAULT_FIELD_POLICIES if policies is None else policies)
    for name, policy in resolved.items():
        if name != policy.field:
            raise ValueError(
                f"checkpoint digest policy registry key {name!r} does not match field {policy.field!r}"
            )
        _validate_field_digest_policy(policy)
    names = sorted(resolved if fields is None else fields)
    missing = [name for name in names if name not in resolved]
    if missing:
        raise KeyError(f"unclassified checkpoint digest fields: {missing}")
    return {
        "schema_version": SCHEMA_VERSION,
        "field_policy_schema_version": FIELD_POLICY_SCHEMA_VERSION,
        "layers": [str(layer) for layer in DigestLayer],
        "fields": [resolved[name].to_manifest() for name in names],
    }


def _active_event_count(metadata: Mapping[str, Any], tensors: Mapping[str, Any]) -> int:
    layout = validated_event_storage_layout({"metadata": metadata, "tensors": tensors})
    return int(layout["active_count"])


def _canonical_tensor_for_layer(
    name: str,
    tensor: Any,
    *,
    layer: DigestLayer,
    policy: FieldDigestPolicy,
    metadata: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> torch.Tensor:
    if not hasattr(tensor, "detach"):
        raise TypeError(f"checkpoint digest field {name!r} is not tensor-like")
    out = tensor.detach().to("cpu").contiguous()
    if not name.startswith("event_tape_") and not name.startswith("typed_lifecycle_event_"):
        if policy.canonicalization != "field-policy row masking only; never blanket-clear every dead/free tensor":
            return out
        alive_mask = tensors.get("alive_mask")
        if alive_mask is None:
            raise ValueError(f"alive_mask is required to canonicalize live-row field {name!r}")
        if out.ndim == 0:
            return out
        mask = alive_mask.detach().to("cpu").to(dtype=torch.bool).contiguous()
        if int(mask.shape[0]) != int(out.shape[0]):
            raise ValueError(f"alive_mask length does not match field {name!r}")
        dead = ~mask
        if not bool(dead.any().item()):
            return out
        masked = out.clone()
        masked[dead] = torch.zeros_like(masked[dead])
        return masked.contiguous()
    active = _active_event_count(metadata, tensors)
    if layer is DigestLayer.TRAJECTORY_STATE:
        return out[:active].contiguous()
    if policy.canonicalization == "active_event_count prefix exact; inactive tail normalized by policy only":
        return out[:active].contiguous()
    return out


def _tensor_payload(tensor: torch.Tensor) -> dict[str, Any]:
    array = np.ascontiguousarray(tensor.detach().to("cpu").numpy())
    return {
        "dtype": str(array.dtype),
        "shape": [int(dim) for dim in array.shape],
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _source_projection_metadata_int(
    metadata: Mapping[str, Any],
    name: str,
    *,
    minimum: int,
) -> int:
    value = metadata.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"source projection metadata.{name} must be an int")
    resolved = int(value)
    if resolved < minimum:
        raise ValueError(
            f"source projection metadata.{name} must be >= {minimum}, got {resolved}"
        )
    return resolved


def _source_projection_tensor(
    tensors: Mapping[str, Any],
    name: str,
    *,
    dtype: torch.dtype,
    ndim: int,
) -> torch.Tensor:
    value = tensors[name]
    if not hasattr(value, "detach"):
        raise TypeError(f"source projection tensor {name!r} is not tensor-like")
    tensor = value.detach().to("cpu").contiguous()
    if tensor.dtype != dtype or tensor.ndim != ndim:
        raise ValueError(
            f"source projection tensor {name!r} must be {dtype} rank {ndim}, "
            f"got {tensor.dtype} rank {tensor.ndim}"
        )
    return tensor


def _source_projection_metadata_bool(
    metadata: Mapping[str, Any],
    name: str,
) -> bool:
    value = metadata[name]
    if type(value) is not bool:
        raise ValueError(f"source projection Patch metadata.{name} must be a bool")
    return value


def _source_projection_metadata_float(
    metadata: Mapping[str, Any],
    name: str,
) -> float:
    value = metadata[name]
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"source projection Patch metadata.{name} must be finite numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"source projection Patch metadata.{name} must be finite numeric")
    return resolved


def _source_projection_patch_tensor(
    tensors: Mapping[str, Any],
    name: str,
    *,
    dtype: torch.dtype,
    ndim: int,
) -> torch.Tensor:
    value = tensors[name]
    if not hasattr(value, "detach"):
        raise TypeError(f"source projection Patch tensor {name!r} is not tensor-like")
    tensor = value.detach()
    if tensor.device.type != "cpu":
        raise ValueError(f"source projection Patch tensor {name!r} must be CPU resident")
    if tensor.dtype != dtype or tensor.ndim != ndim:
        raise ValueError(
            f"source projection Patch tensor {name!r} must be {dtype} rank {ndim}, "
            f"got {tensor.dtype} rank {tensor.ndim}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"source projection Patch tensor {name!r} must be contiguous")
    return tensor


def _source_projection_patch_profile(
    metadata: Mapping[str, Any],
) -> str:
    missing = sorted(set(SOURCE_PATCH_REQUIRED_METADATA_FIELDS) - set(metadata))
    if missing:
        raise ValueError(f"source projection Patch metadata is missing required fields: {missing}")

    profile = metadata["native_runtime_patch_profile"]
    if type(profile) is not str:
        raise ValueError("source projection Patch profile must be an exact P0/P1/P2 string")
    if profile not in SOURCE_PATCH_PROFILES:
        raise ValueError(f"source projection Patch profile is unknown: {profile!r}")
    if metadata["native_runtime_patch_profile_scope"] != "full_profile":
        raise ValueError("source projection Patch profile scope must be 'full_profile'")
    if metadata["native_runtime_patch_contract_schema_version"] != (
        "eiel_patch_native_import_contract_v1"
    ):
        raise ValueError("source projection Patch contract schema is unsupported")

    bool_fields = (
        "patch_world_enable",
        "patch_world_observation_enable",
        "patch_world_diagnostics_enable",
        "patch_world_lineage_descriptors_enable",
        "patch_movement_cost_apply_enable",
        "patch_world_waste_consequence_enable",
        "patch_visibility_from_cover_enable",
        "patch_density_include_self",
        "patch_density_normalize",
    )
    bool_values = {
        name: _source_projection_metadata_bool(metadata, name) for name in bool_fields
    }
    if bool_values["patch_world_enable"] is not True:
        raise ValueError("source projection Patch metadata.patch_world_enable must be true")

    for name in (
        "patch_world_schema_version",
        "patch_world_observation_schema_version",
        "patch_world_descriptor_schema_version",
    ):
        if _source_projection_metadata_int(metadata, name, minimum=1) != 1:
            raise ValueError(f"source projection Patch metadata.{name} must equal 1")
    density_radius = _source_projection_metadata_int(
        metadata,
        "patch_density_radius",
        minimum=0,
    )

    visibility_default = _source_projection_metadata_float(
        metadata,
        "patch_visibility_default",
    )
    visibility_min = _source_projection_metadata_float(
        metadata,
        "patch_visibility_min",
    )
    visibility_max = _source_projection_metadata_float(
        metadata,
        "patch_visibility_max",
    )
    _source_projection_metadata_float(metadata, "patch_food_presence_threshold")
    if visibility_min > visibility_max:
        raise ValueError("source projection Patch visibility bounds are reversed")
    if (
        not bool_values["patch_visibility_from_cover_enable"]
        and not visibility_min <= visibility_default <= visibility_max
    ):
        raise ValueError("source projection Patch visibility default is outside its bounds")
    if density_radius < 0:
        raise ValueError("source projection Patch density radius must be nonnegative")

    marker_contract = {
        "patch_array_import_phase": "post_reference_setup_once",
        "patch_visibility_authority": "derived_from_patch_cover",
        "patch_density_authority": "derived_pre_action_from_agent_grid",
        "patch_food_binary_projection": (
            "patch_food_amount_strictly_greater_than_threshold"
        ),
        "patch_pending_emission_phase": "queued_until_next_tick",
    }
    for name, expected in marker_contract.items():
        if metadata[name] != expected:
            raise ValueError(f"source projection Patch contract marker mismatch: {name}")

    expected_features = {
        "P0": (False, False, False),
        "P1": (True, False, False),
        "P2": (True, True, True),
    }[profile]
    actual_features = (
        bool_values["patch_world_observation_enable"],
        bool_values["patch_movement_cost_apply_enable"],
        bool_values["patch_world_waste_consequence_enable"],
    )
    if actual_features != expected_features:
        raise ValueError("source projection Patch profile metadata contradicts feature flags")
    if profile in {"P1", "P2"} and not (
        bool_values["patch_world_diagnostics_enable"]
        and bool_values["patch_world_lineage_descriptors_enable"]
    ):
        raise ValueError("source projection Patch P1/P2 diagnostics contract is incomplete")
    return profile


def _rebuilt_patch_visibility(
    metadata: Mapping[str, Any],
    cover: torch.Tensor,
) -> torch.Tensor:
    visibility_min = _source_projection_metadata_float(
        metadata,
        "patch_visibility_min",
    )
    visibility_max = _source_projection_metadata_float(
        metadata,
        "patch_visibility_max",
    )
    cover_array = cover.numpy()
    if _source_projection_metadata_bool(metadata, "patch_visibility_from_cover_enable"):
        raw = np.float32(1.0) - cover_array
    else:
        raw = np.full(
            tuple(int(dim) for dim in cover.shape),
            _source_projection_metadata_float(metadata, "patch_visibility_default"),
            dtype=np.float32,
        )
    rebuilt = np.ascontiguousarray(
        np.clip(raw, visibility_min, visibility_max),
        dtype=np.float32,
    )
    return torch.from_numpy(rebuilt)


def _rebuilt_patch_density(
    metadata: Mapping[str, Any],
    agent_grid: torch.Tensor,
) -> torch.Tensor:
    radius = _source_projection_metadata_int(
        metadata,
        "patch_density_radius",
        minimum=0,
    )
    include_self = _source_projection_metadata_bool(
        metadata,
        "patch_density_include_self",
    )
    normalize = _source_projection_metadata_bool(
        metadata,
        "patch_density_normalize",
    )
    grid = agent_grid.numpy()
    height, width = (int(grid.shape[0]), int(grid.shape[1]))
    occupied = (grid != -1).astype(np.int32, copy=False)
    integral = np.pad(occupied, ((1, 0), (1, 0)), mode="constant").cumsum(
        axis=0
    ).cumsum(axis=1)
    ys = np.arange(height, dtype=np.int32)
    xs = np.arange(width, dtype=np.int32)
    y0 = np.maximum(0, ys - radius)
    y1 = np.minimum(height, ys + radius + 1)
    x0 = np.maximum(0, xs - radius)
    x1 = np.minimum(width, xs + radius + 1)
    density = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    ).astype(np.float32, copy=False)
    if not include_self:
        density -= occupied.astype(np.float32, copy=False)
    if normalize:
        denominator = float((2 * radius + 1) * (2 * radius + 1))
        if not include_self:
            denominator -= 1.0
        if denominator > 0.0:
            density[:, :] = density / np.float32(denominator)
    return torch.from_numpy(np.ascontiguousarray(density, dtype=np.float32))


def _canonical_patch_source_projection(
    projection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = dict(projection["metadata"])
    tensors = dict(projection["tensors"])
    profile = _source_projection_patch_profile(metadata)
    wdl_storage_enabled = metadata.get("native_runtime_wdl_storage_enable", False)
    if type(wdl_storage_enabled) is not bool:
        raise ValueError("source projection WDL storage marker must be boolean")
    wdl_fields = (
        tuple(SOURCE_WDL_STORAGE_TENSOR_DTYPES) if wdl_storage_enabled else ()
    )
    required_world_fields = (
        *SOURCE_PATCH_REQUIRED_WORLD_TENSOR_FIELDS,
        *(
            SOURCE_PATCH_P2_REQUIRED_WORLD_TENSOR_FIELDS
            if profile == "P2"
            else ()
        ),
    )
    required_fields = {
        "agent_grid",
        "food_grid",
        *SOURCE_PATCH_REQUIRED_ROW_TENSOR_FIELDS,
        *required_world_fields,
        *SOURCE_PATCH_DERIVED_TENSOR_FIELDS,
        *wdl_fields,
    }
    missing = sorted(required_fields - set(tensors))
    if missing:
        raise ValueError(f"source projection Patch tensors are missing required fields: {missing}")
    if profile != "P2" and "patch_recovery_activity" in tensors:
        raise ValueError(
            "source projection Patch profile has inactive patch_recovery_activity state"
        )

    allowed_patch_fields = {
        *SOURCE_PATCH_REQUIRED_WORLD_TENSOR_FIELDS,
        *SOURCE_PATCH_P2_REQUIRED_WORLD_TENSOR_FIELDS,
        *SOURCE_PATCH_REQUIRED_ROW_TENSOR_FIELDS,
        *SOURCE_PATCH_DERIVED_TENSOR_FIELDS,
        *wdl_fields,
    }
    unsupported = sorted(
        name
        for name in tensors
        if name.startswith("patch_") and name not in allowed_patch_fields
    )
    if unsupported:
        raise ValueError(f"source projection Patch tensors contain unsupported state: {unsupported}")

    agent_grid = _source_projection_patch_tensor(
        tensors,
        "agent_grid",
        dtype=torch.int32,
        ndim=2,
    )
    food_grid = _source_projection_patch_tensor(
        tensors,
        "food_grid",
        dtype=torch.int32,
        ndim=2,
    )
    world_shape = tuple(int(dim) for dim in agent_grid.shape)
    if tuple(int(dim) for dim in food_grid.shape) != world_shape:
        raise ValueError("source projection Patch food_grid/agent_grid shape contradiction")

    authoritative_tensors: dict[str, torch.Tensor] = {}
    for name in required_world_fields:
        tensor = _source_projection_patch_tensor(
            tensors,
            name,
            dtype=torch.float32,
            ndim=2,
        )
        if tuple(int(dim) for dim in tensor.shape) != world_shape:
            raise ValueError(
                f"source projection Patch tensor {name!r} must match agent_grid shape"
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"source projection Patch tensor {name!r} must be finite")
        authoritative_tensors[name] = tensor

    for name in wdl_fields:
        tensor = _source_projection_patch_tensor(
            tensors,
            name,
            dtype=SOURCE_WDL_STORAGE_TENSOR_DTYPES[name],
            ndim=2,
        )
        if tuple(int(dim) for dim in tensor.shape) != world_shape:
            raise ValueError(
                f"source projection WDL tensor {name!r} must match agent_grid shape"
            )
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"source projection WDL tensor {name!r} must be finite")
        authoritative_tensors[name] = tensor

    habitat_lineage = _source_projection_patch_tensor(
        tensors,
        "patch_habitat_lineage_id",
        dtype=torch.int32,
        ndim=1,
    )
    alive_mask = _source_projection_tensor(
        tensors,
        "alive_mask",
        dtype=torch.bool,
        ndim=1,
    )
    if int(habitat_lineage.shape[0]) != int(alive_mask.shape[0]):
        raise ValueError(
            "source projection Patch patch_habitat_lineage_id capacity contradiction"
        )

    for name in SOURCE_PATCH_DERIVED_TENSOR_FIELDS:
        derived = _source_projection_patch_tensor(
            tensors,
            name,
            dtype=torch.float32,
            ndim=2,
        )
        if tuple(int(dim) for dim in derived.shape) != world_shape:
            raise ValueError(
                f"source projection Patch derived tensor {name!r} must match agent_grid shape"
            )

    threshold = np.float32(
        _source_projection_metadata_float(metadata, "patch_food_presence_threshold")
    )
    expected_food_grid = torch.where(
        authoritative_tensors["patch_food_amount"] > threshold,
        torch.ones_like(food_grid),
        torch.zeros_like(food_grid),
    )
    if not torch.equal(food_grid, expected_food_grid):
        raise ValueError("source projection Patch binary food projection is inconsistent")

    rebuilt = {
        "patch_visibility": _rebuilt_patch_visibility(
            metadata,
            authoritative_tensors["patch_cover"],
        ),
        "patch_density": _rebuilt_patch_density(metadata, agent_grid),
    }
    canonical_tensors = dict(tensors)
    canonical_tensors.update(rebuilt)
    canonical = dict(projection)
    canonical["tensors"] = canonical_tensors
    tensor_manifest = projection.get("tensor_manifest")
    if not isinstance(tensor_manifest, Mapping):
        raise TypeError("source projection Patch tensor_manifest must be a mapping")
    canonical_manifest = dict(tensor_manifest)
    for name in SOURCE_PATCH_DERIVED_TENSOR_FIELDS:
        canonical_manifest.pop(name, None)
    canonical["tensor_manifest"] = canonical_manifest

    authoritative_fields = [
        "agent_grid",
        "food_grid",
        *SOURCE_PATCH_REQUIRED_ROW_TENSOR_FIELDS,
        *required_world_fields,
        *wdl_fields,
    ]
    patch_authority = {
        "schema_version": SOURCE_PATCH_AUTHORITY_SCHEMA_VERSION,
        "profile": profile,
        "profile_scope": "full_profile",
        "wdl_storage_schema_version": (
            metadata.get("native_runtime_wdl_storage_schema_version")
            if wdl_storage_enabled
            else None
        ),
        "wdl_authoritative_tensor_fields": list(wdl_fields),
        "authoritative_tensor_fields": authoritative_fields,
        "derived_tensor_fields": [
            {
                "field": "patch_visibility",
                "classification": "reconstructible_derived",
                "rebuild_source_fields": ["patch_cover"],
                **_tensor_payload(rebuilt["patch_visibility"]),
            },
            {
                "field": "patch_density",
                "classification": "reconstructible_derived",
                "rebuild_source_fields": ["agent_grid"],
                **_tensor_payload(rebuilt["patch_density"]),
            },
        ],
    }
    return canonical, patch_authority


def _source_projection_action_cache_manifest(
    projection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_value = projection.get("metadata")
    tensors_value = projection.get("tensors")
    if not isinstance(metadata_value, Mapping) or not isinstance(tensors_value, Mapping):
        raise TypeError("source projection metadata and tensors must be mappings")
    metadata = dict(metadata_value)
    tensors = dict(tensors_value)
    required_metadata = {"world", *SOURCE_AUTHORITY_CURSOR_METADATA_FIELDS}
    required_tensors = {"alive_mask", "slot_tokens", *SOURCE_ACTION_CACHE_FIELDS}
    missing_metadata = sorted(required_metadata - set(metadata))
    missing_tensors = sorted(required_tensors - set(tensors))
    if missing_metadata:
        raise ValueError(f"source projection is missing authority metadata fields: {missing_metadata}")
    if missing_tensors:
        raise ValueError(
            f"source projection is missing complete action-cache tensor fields: {missing_tensors}"
        )
    normal_world = str(metadata["world"] or "")
    if normal_world != "flat" and not (
        type(metadata["world"]) is str and metadata["world"] == "patch"
    ):
        raise ValueError("source projection digest contract requires world in {'flat','patch'}")

    generation = _source_projection_metadata_int(metadata, "generation", minimum=0)
    tick = _source_projection_metadata_int(metadata, "tick", minimum=0)
    configured_steps = _source_projection_metadata_int(
        metadata,
        "configured_steps_per_generation",
        minimum=1,
    )
    if tick > configured_steps:
        raise ValueError("source projection metadata.tick exceeds configured_steps_per_generation")
    expected_phase = "generation_boundary" if tick == 0 else "partial_generation"
    expected_epoch = f"generation:{generation}:tick:{tick}"
    if str(metadata["agent_grid_runner_phase"] or "") != expected_phase:
        raise ValueError("source projection authority cursor phase contradiction")
    if str(metadata["agent_grid_authority_epoch"] or "") != expected_epoch:
        raise ValueError("source projection authority cursor epoch contradiction")

    alive = _source_projection_tensor(tensors, "alive_mask", dtype=torch.bool, ndim=1)
    slot_tokens = _source_projection_tensor(tensors, "slot_tokens", dtype=torch.int64, ndim=1)
    actions = _source_projection_tensor(
        tensors,
        "action_cache_actions",
        dtype=torch.int32,
        ndim=2,
    )
    tokens = _source_projection_tensor(tensors, "action_cache_token", dtype=torch.int64, ndim=1)
    cursors = _source_projection_tensor(tensors, "action_cache_cursor", dtype=torch.int32, ndim=1)
    next_actions = _source_projection_tensor(tensors, "action_cache_next", dtype=torch.int32, ndim=1)
    capacity = int(alive.shape[0])
    row_counts = {
        "slot_tokens": int(slot_tokens.shape[0]),
        "action_cache_actions": int(actions.shape[0]),
        "action_cache_token": int(tokens.shape[0]),
        "action_cache_cursor": int(cursors.shape[0]),
        "action_cache_next": int(next_actions.shape[0]),
    }
    mismatched = {name: count for name, count in row_counts.items() if count != capacity}
    if mismatched:
        raise ValueError(f"source projection action-cache capacity mismatch: {mismatched}")
    if int(actions.shape[1]) <= 0:
        raise ValueError("source projection action_cache_actions must have positive width")

    token_valid = tokens.ge(0)
    cursor_valid = cursors.ge(0)
    partial_rows = torch.nonzero(alive & token_valid.ne(cursor_valid), as_tuple=False).reshape(-1)
    if int(partial_rows.numel()) > 0:
        raise ValueError("source projection action-cache token/cursor validity contradiction")
    valid = (alive & token_valid & cursor_valid).contiguous()
    invalid_tokens = torch.nonzero(valid & tokens.ne(slot_tokens), as_tuple=False).reshape(-1)
    if int(invalid_tokens.numel()) > 0:
        raise ValueError("source projection action-cache token/slot-token contradiction")
    valid_rows = torch.nonzero(valid, as_tuple=False).reshape(-1).to(torch.int64)
    for row in valid_rows.tolist():
        cursor = int(cursors[row].item())
        expected_next = int(actions[row, min(cursor, int(actions.shape[1]) - 1)].item())
        if int(next_actions[row].item()) != expected_next:
            raise ValueError("source projection action-cache cursor/next contradiction")

    authority_cursor = {
        "generation": generation,
        "tick": tick,
        "configured_steps_per_generation": configured_steps,
        "agent_grid_runner_phase": expected_phase,
        "agent_grid_authority_epoch": expected_epoch,
    }
    cache_fields = [
        {"field": name, **_tensor_payload(tensors[name])}
        for name in SOURCE_ACTION_CACHE_FIELDS
    ]
    action_cache = {
        "schema_version": SOURCE_ACTION_CACHE_SCHEMA_VERSION,
        "fields": cache_fields,
        "validity": _tensor_payload(valid),
        "valid_slots": [int(row) for row in valid_rows.tolist()],
        "valid_slot_count": int(valid_rows.numel()),
        "boundary_status": "captured",
        "sequence_length": int(actions.shape[1]),
        "authority_epoch": expected_epoch,
    }
    action_cache["digest"] = _packed_source_projection_digest(
        _pack_source_projection_value(action_cache)
    )
    return authority_cursor, action_cache


def _source_projection_rng_manifest(
    projection: Mapping[str, Any],
    *,
    require_complete: bool,
) -> dict[str, Any]:
    metadata_value = projection.get("metadata")
    tensors_value = projection.get("tensors")
    if not isinstance(metadata_value, Mapping) or not isinstance(tensors_value, Mapping):
        raise TypeError("source projection metadata and tensors must be mappings")
    metadata = dict(metadata_value)
    tensors = dict(tensors_value)
    missing = sorted(set(SOURCE_RNG_REQUIRED_METADATA_FIELDS) - set(metadata))
    if require_complete and missing:
        raise ValueError(f"source projection is missing complete RNG authority fields: {missing}")

    metadata_fields = [
        name
        for name in (*SOURCE_RNG_REQUIRED_METADATA_FIELDS, *SOURCE_RNG_OPTIONAL_METADATA_FIELDS)
        if name in metadata
    ]
    tensor_fields = [name for name in SOURCE_RNG_TENSOR_FIELDS if name in tensors]
    if "native_runtime_env_policy_numpy_alias" in metadata:
        alias = metadata["native_runtime_env_policy_numpy_alias"]
        if not isinstance(alias, (bool, np.bool_)):
            raise ValueError("source projection RNG env/policy alias marker must be a bool")
        if bool(alias):
            env_state = _pack_source_projection_value(metadata.get("native_runtime_numpy_env_state"))
            policy_state = _pack_source_projection_value(metadata.get("native_runtime_numpy_policy_state"))
            if env_state != policy_state:
                raise ValueError("source projection RNG env/policy alias state contradiction")

    packed = {
        "metadata": {
            name: _pack_source_projection_value(metadata[name]) for name in metadata_fields
        },
        "tensors": {
            name: _pack_source_projection_value(tensors[name]) for name in tensor_fields
        },
    }
    return {
        "schema_version": SOURCE_RNG_SCHEMA_VERSION,
        "digest": _packed_source_projection_digest(
            _pack_source_projection_value(packed)
        ),
        "metadata_fields": metadata_fields,
        "tensor_fields": tensor_fields,
        "env_policy_numpy_alias": bool(
            metadata.get("native_runtime_env_policy_numpy_alias", False)
        ),
        "complete": not missing,
        "missing_required_metadata_fields": missing,
    }


def _pack_source_projection_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        tensor = value.detach().to("cpu").contiguous()
        return {
            "kind": "tensor",
            **_tensor_payload(tensor),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "kind": "ndarray",
            "dtype": str(array.dtype),
            "shape": [int(dim) for dim in array.shape],
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        items = [
            (_pack_source_projection_value(key), _pack_source_projection_value(item))
            for key, item in value.items()
        ]
        items.sort(key=lambda row: json.dumps(row[0], sort_keys=True, separators=(",", ":")))
        return {"kind": "mapping", "items": items}
    if isinstance(value, (list, tuple)):
        return {
            "kind": "sequence",
            "items": [_pack_source_projection_value(item) for item in value],
        }
    if isinstance(value, np.generic):
        return _pack_source_projection_value(value.item())
    if isinstance(value, (bytes, bytearray)):
        return {"kind": "bytes", "hex": bytes(value).hex()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"kind": type(value).__name__, "value": value}
    raise TypeError(f"unsupported source projection value type: {type(value).__name__}")


def _packed_source_projection_digest(packed: Any) -> str:
    encoded = json.dumps(packed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_source_projection_digest_manifest(
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Digest the complete native checkpoint projection without layered canonicalization."""

    if not isinstance(projection, Mapping):
        raise TypeError("source projection must be a mapping")
    if any(not isinstance(name, str) for name in projection):
        raise TypeError("source projection top-level field names must be strings")
    missing = sorted(set(SOURCE_PROJECTION_REQUIRED_TOP_LEVEL_FIELDS) - set(projection))
    if missing:
        raise ValueError(f"source projection is missing top-level authority fields: {missing}")
    pending_agents = projection["pending_generation_free_agent_ids"]
    pending_slots = projection["pending_generation_free_slots"]
    if not isinstance(pending_agents, (list, tuple)) or not isinstance(pending_slots, (list, tuple)):
        raise TypeError("source projection pending-free fields must be sequences")
    if len(pending_agents) != len(pending_slots):
        raise ValueError("source projection pending-free field lengths disagree")

    authority_cursor, action_cache = _source_projection_action_cache_manifest(projection)
    rng = _source_projection_rng_manifest(projection, require_complete=False)
    canonical_projection: Mapping[str, Any] = projection
    patch_authority: dict[str, Any] | None = None
    metadata = projection["metadata"]
    if isinstance(metadata, Mapping) and type(metadata.get("world")) is str:
        if metadata["world"] == "patch":
            canonical_projection, patch_authority = _canonical_patch_source_projection(
                projection
            )
    top_level_fields = sorted(projection)
    packed_fields = {
        name: _pack_source_projection_value(canonical_projection[name])
        for name in top_level_fields
    }
    packed_projection = {
        "kind": "mapping",
        "items": [
            (_pack_source_projection_value(name), packed_fields[name])
            for name in top_level_fields
        ],
    }
    manifest = {
        "schema_version": SOURCE_PROJECTION_SCHEMA_VERSION,
        "digest": _packed_source_projection_digest(packed_projection),
        "top_level_field_count": len(top_level_fields),
        "top_level_fields": top_level_fields,
        "top_level_field_digests": {
            name: _packed_source_projection_digest(packed_fields[name]) for name in top_level_fields
        },
        "authority_cursor": authority_cursor,
        "action_cache": action_cache,
        "rng": rng,
    }
    if patch_authority is not None:
        manifest["patch_authority"] = patch_authority
    return manifest


def checkpoint_source_projection_authority_manifest(
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the one authority record copied into every native bundle view."""

    digest_manifest = checkpoint_source_projection_digest_manifest(projection)
    rng = _source_projection_rng_manifest(projection, require_complete=True)
    schema_version = projection.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError("source projection schema_version must be a non-empty string")
    metadata = dict(projection.get("metadata") or {})
    tensors = dict(projection.get("tensors") or {})
    alive_count = _source_projection_metadata_int(metadata, "alive_count", minimum=0)
    alive_mask = _source_projection_tensor(
        tensors,
        "alive_mask",
        dtype=torch.bool,
        ndim=1,
    )
    projected_alive_count = int(alive_mask.to(torch.int64).sum().item())
    if alive_count != projected_alive_count:
        raise ValueError(
            "source projection metadata.alive_count/alive_mask contradiction: "
            f"metadata={alive_count} projection={projected_alive_count}"
        )
    cursor = dict(digest_manifest["authority_cursor"])
    cache = dict(digest_manifest["action_cache"])
    return {
        "authority_record_schema_version": SOURCE_AUTHORITY_SCHEMA_VERSION,
        "source_projection_sha256": str(digest_manifest["digest"]),
        "source_projection_schema_version": schema_version,
        "source_generation": int(cursor["generation"]),
        "source_tick": int(cursor["tick"]),
        "source_phase": str(cursor["agent_grid_runner_phase"]),
        "source_authority_epoch": str(cursor["agent_grid_authority_epoch"]),
        "source_action_cache_sha256": str(cache["digest"]),
        "source_action_cache_valid_slot_count": int(cache["valid_slot_count"]),
        "source_action_cache_boundary_status": str(cache["boundary_status"]),
        "source_rng_sha256": str(rng["digest"]),
    }


def validate_checkpoint_source_projection(
    projection: Mapping[str, Any],
    *,
    require_rng_authority: bool = False,
) -> None:
    if require_rng_authority:
        checkpoint_source_projection_authority_manifest(projection)
    else:
        checkpoint_source_projection_digest_manifest(projection)


def canonicalize_checkpoint_storage_field(
    name: str,
    tensor: Any,
    *,
    alive_mask: Any | None = None,
    policy: FieldDigestPolicy | None = None,
    allow_dead_free_row_normalization: bool = False,
) -> torch.Tensor:
    resolved = DEFAULT_FIELD_POLICIES.get(name) if policy is None else policy
    if resolved is None:
        raise KeyError(f"unclassified checkpoint digest field: {name}")
    if not hasattr(tensor, "detach"):
        raise TypeError(f"checkpoint digest field {name!r} is not tensor-like")
    out = tensor.detach().to("cpu").contiguous().clone()
    if name == "agent_grid":
        return out
    if not allow_dead_free_row_normalization:
        return out
    if alive_mask is None:
        raise ValueError("alive_mask is required for dead/free row normalization")
    if out.ndim == 0:
        return out
    mask = alive_mask.detach().to("cpu").to(dtype=torch.bool).contiguous()
    if int(mask.shape[0]) != int(out.shape[0]):
        raise ValueError(f"alive_mask length does not match field {name!r}")
    if resolved.primary_layer is not DigestLayer.TRAJECTORY_STATE:
        return out
    dead = ~mask
    if bool(dead.any().item()):
        out[dead] = torch.zeros_like(out[dead])
    return out


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _first_tensor_difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any] | None:
    if list(left.shape) != list(right.shape):
        return {
            "reason": "shape_mismatch",
            "left_shape": [int(dim) for dim in left.shape],
            "right_shape": [int(dim) for dim in right.shape],
        }
    if str(left.dtype) != str(right.dtype):
        return {
            "reason": "dtype_mismatch",
            "left_dtype": str(left.dtype),
            "right_dtype": str(right.dtype),
        }
    if torch.equal(left, right):
        return None
    differing = torch.ne(left, right).reshape(-1).nonzero(as_tuple=False)
    flat_index = int(differing[0].item()) if int(differing.numel()) else 0
    index = [int(dim) for dim in np.unravel_index(flat_index, tuple(int(dim) for dim in left.shape))]
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    return {
        "reason": "value_mismatch",
        "first_differing_flat_index": flat_index,
        "first_differing_index": index,
        "left_value": _json_scalar(left_flat[flat_index]),
        "right_value": _json_scalar(right_flat[flat_index]),
    }


def compare_checkpoint_layer_first_difference(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    layer: DigestLayer | str,
    policies: Mapping[str, FieldDigestPolicy] | None = None,
) -> dict[str, Any] | None:
    resolved_layer = DigestLayer(layer)
    resolved = dict(DEFAULT_FIELD_POLICIES if policies is None else policies)
    left_metadata = dict(left.get("metadata") or {})
    right_metadata = dict(right.get("metadata") or {})
    left_tensors = dict(left.get("tensors") or {})
    right_tensors = dict(right.get("tensors") or {})
    priority = (
        "alive_mask",
        "slot_to_agent_id",
        "agent_id_to_slot_agent_ids",
        "agent_id_to_slot_slots",
        "agent_id_to_slot",
        "slot_tokens",
        "pos_y",
        "pos_x",
        "agent_grid",
    )
    all_names = set(left_tensors) | set(right_tensors)
    names = [name for name in priority if name in all_names]
    names.extend(name for name in sorted(all_names) if name not in set(priority))
    for name in names:
        policy = resolved.get(name)
        if policy is None:
            raise KeyError(f"unclassified checkpoint digest field: {name}")
        if not policy.includes(resolved_layer):
            continue
        if name not in left_tensors:
            return {
                "field": name,
                "layer": str(resolved_layer),
                "reason": "left_missing",
                "authoritative": not policy.diagnostic_only,
                "policy": policy.to_manifest(),
            }
        if name not in right_tensors:
            return {
                "field": name,
                "layer": str(resolved_layer),
                "reason": "right_missing",
                "authoritative": not policy.diagnostic_only,
                "policy": policy.to_manifest(),
            }
        left_canonical = _canonical_tensor_for_layer(
            name,
            left_tensors[name],
            layer=resolved_layer,
            policy=policy,
            metadata=left_metadata,
            tensors=left_tensors,
        )
        right_canonical = _canonical_tensor_for_layer(
            name,
            right_tensors[name],
            layer=resolved_layer,
            policy=policy,
            metadata=right_metadata,
            tensors=right_tensors,
        )
        mismatch = _first_tensor_difference(left_canonical, right_canonical)
        if mismatch is None:
            continue
        row_context: dict[str, Any] = {}
        first_index = mismatch.get("first_differing_index")
        if isinstance(first_index, list) and first_index:
            row = int(first_index[0])
            for context_name in ("alive_mask", "slot_to_agent_id"):
                left_context = left_tensors.get(context_name)
                right_context = right_tensors.get(context_name)
                if (
                    hasattr(left_context, "detach")
                    and hasattr(right_context, "detach")
                    and int(left_context.detach().to("cpu").numel()) > row
                    and int(right_context.detach().to("cpu").numel()) > row
                ):
                    row_context[f"left_{context_name}"] = _json_scalar(
                        left_context.detach().to("cpu").reshape(-1)[row]
                    )
                    row_context[f"right_{context_name}"] = _json_scalar(
                        right_context.detach().to("cpu").reshape(-1)[row]
                    )
        return {
            "field": name,
            "path": f"$.tensors.{name}",
            "layer": str(resolved_layer),
            "authoritative": not policy.diagnostic_only,
            "left_dtype": str(left_canonical.dtype),
            "right_dtype": str(right_canonical.dtype),
            "left_shape": [int(dim) for dim in left_canonical.shape],
            "right_shape": [int(dim) for dim in right_canonical.shape],
            "left_sha256": _tensor_payload(left_canonical)["sha256"],
            "right_sha256": _tensor_payload(right_canonical)["sha256"],
            "policy": policy.to_manifest(),
            **mismatch,
            **row_context,
        }
    return None


def checkpoint_digest_manifest(
    projection: Mapping[str, Any],
    *,
    fields: list[str] | tuple[str, ...] | set[str] | None = None,
    layer: DigestLayer | str,
    policies: Mapping[str, FieldDigestPolicy] | None = None,
) -> dict[str, Any]:
    resolved_layer = DigestLayer(layer)
    resolved = dict(DEFAULT_FIELD_POLICIES if policies is None else policies)
    metadata = dict(projection.get("metadata") or {})
    tensors = dict(projection.get("tensors") or {})
    modular_storage = modular_storage_from_projection(metadata, tensors)
    names = sorted(tensors if fields is None else fields)
    entries = []
    for name in names:
        policy = resolved.get(name)
        if policy is None:
            raise KeyError(f"unclassified checkpoint digest field: {name}")
        if not policy.includes(resolved_layer):
            continue
        canonical = _canonical_tensor_for_layer(
            name,
            tensors[name],
            layer=resolved_layer,
            policy=policy,
            metadata=metadata,
            tensors=tensors,
        )
        entry = {"field": name, **_tensor_payload(canonical)}
        entries.append(entry)
    projection_schema_version = str(projection.get("schema_version", ""))
    if (
        resolved_layer is DigestLayer.CANONICAL_CHECKPOINT_STORAGE
        and projection_schema_version in CANONICAL_STORAGE_EQUIVALENT_PROJECTION_SCHEMAS
    ):
        # Canonical storage compares decoded authority, not the route-specific
        # projection wrapper used to carry it into the checkpoint bundle.
        projection_schema_version = SCHEMA_VERSION
    layer_metadata = {
        "schema_version": projection_schema_version,
        "world": str(metadata.get("world", "")),
        "generation": int(metadata.get("generation", 0) or 0),
        "tick": int(metadata.get("tick", 0) or 0),
    }
    if modular_storage is not None:
        layer_metadata[MODULAR_STORAGE_METADATA_KEY] = {
            "profile_id": modular_storage.layout.profile_id,
            "sidecar_sha256": modular_storage.sidecar_sha256,
            "packed_sha256": modular_storage.packed_sha256,
        }
    layer_payload = {
        "layer": str(resolved_layer),
        "fields": entries,
        "metadata": layer_metadata,
    }
    digest = hashlib.sha256(
        json.dumps(layer_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": str(resolved_layer),
        "digest": digest,
        "field_count": len(entries),
        "fields": entries,
        "metadata": layer_payload["metadata"],
    }
