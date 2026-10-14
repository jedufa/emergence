from __future__ import annotations

"""Frozen RECOMB-1 inheritance profiles and locus/linkage registry.

The legacy profile is deliberately a no-op.  The linked profile adds one
fixed seven-draw parent-copy schedule.  Donor selections are transient runtime
context; they are not fields on :class:`Genome` and therefore are not part of
genome serialization.  The repaired checkpoint metadata freezes every repaired
locus, its allowed inheritance mode, linkage group, bounds, mutation rule, and
checkpoint schema.
"""

import hashlib
import json
from typing import Any, Mapping

from evolution.diploid_inheritance_v1 import (
    DIPLOID_EXPRESSION_PROFILE,
    DIPLOID_GENOME_SCHEMA_VERSION,
    DIPLOID_INHERITANCE_PROFILE,
    DIPLOID_MUTATION_PROFILE,
    DIPLOID_RECOMBINATION_PROFILE,
    LINKAGE_MAP_SHA256 as DIPLOID_LINKAGE_MAP_SHA256,
    LOCUS_REGISTRY_SHA256 as DIPLOID_LOCUS_REGISTRY_SHA256,
)


LEGACY_INHERITANCE_PROFILE = "legacy_midpoint_v1"
LINKED_INHERITANCE_PROFILE = "linked_parent_copy_v1"
RETIRED_INHERITANCE_PROFILES = frozenset({LINKED_INHERITANCE_PROFILE})
RETIREMENT_DECISION = "P12-A"
HISTORICAL_CHECKPOINT_DISPOSITION_SCHEMA_VERSION = (
    "eiel_historical_inheritance_checkpoint_disposition_v1"
)
DIPLOID_FLAT_LINKAGE_COMPONENT_SCHEMA_VERSION = (
    "eiel_diploid_flat_linkage_compatibility_component_v1"
)
INHERITANCE_REGISTRY_VERSION = "eiel_inheritance_registry_v1"
INHERITANCE_CHECKPOINT_SCHEMA_VERSION = "recomb1_inheritance_checkpoint_v1"
INHERITANCE_COMPATIBILITY_VERSION = "recomb1_compatibility_v1"
INHERITANCE_PARENT_ORDER_VERSION = "canonical_hereditary_payload_sha256_v1"
EQUAL_GENOME_POLICY_TIEBREAK_VERSION = "policy_controller_sha256_ascending_v1"
INHERITANCE_LINKAGE_BUDGET_BOUNDARY = "mutation.reproduction.inheritance_linkage"
MUTATION_AFTER_INHERITANCE_VERSION = "existing_mutation_schedule_after_linkage_v1"
SUPPORTED_VALIDATION_LANE = "cpu_python_reference_only_in_recomb1"
LINKED_PROFILE_DRAW_COUNT = 7

ALLOWED_INHERITANCE_MODES = (
    "parent_copy",
    "midpoint_legacy",
    "categorical_parent_copy",
    "whole_record_donor",
    "homology_record_donor",
    "nonhereditary_runtime",
)


LINKAGE_BLOCKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "behavioral_strategy_v1",
        ("exploration_bias", "aggression_bias", "action_temperature"),
    ),
    (
        "regulation_prediction_v1",
        (
            "drive_inertia",
            "drive_noise",
            "drive_logit_gain",
            "pred_lr",
            "drive_sp_hunger",
            "drive_sp_pain",
            "drive_sp_fatigue",
            "drive_sp_fear",
            "exp_prior_strength",
            "exp_prior_noise",
            "drive_birth_noise",
        ),
    ),
    ("life_history_v1", ("reproduction_threshold",)),
    ("mutation_rate_v1", ("mutation_rate",)),
    (
        "memory_controller_architecture_v1",
        ("hidden_size", "memory_dim", "use_gru", "policy_controller"),
    ),
    ("resource_handling_v1", ("resource_handling_axis",)),
    ("family_head_strategy_v1", ("family_head_residuals",)),
)
LINKAGE_BLOCK_IDS = tuple(block_id for block_id, _fields in LINKAGE_BLOCKS)


_FIELD_METADATA: dict[str, dict[str, Any]] = {
    "exploration_bias": {
        "value_type": "bounded_float",
        "bounds": [-1.0, 1.0],
        "mutation_rule": "existing_bounded_gaussian_exploration_bias",
    },
    "aggression_bias": {
        "value_type": "bounded_float",
        "bounds": [-2.0, 2.0],
        "mutation_rule": "existing_bounded_gaussian_aggression_bias",
    },
    "action_temperature": {
        "value_type": "bounded_float",
        "bounds": [0.25, 2.5],
        "mutation_rule": "existing_bounded_gaussian_action_temperature",
    },
    "drive_inertia": {
        "value_type": "bounded_float",
        "bounds": [0.0, 0.999],
        "mutation_rule": "existing_bounded_gaussian_drive_inertia",
    },
    "drive_noise": {
        "value_type": "bounded_float",
        "bounds": [0.0, 0.1],
        "mutation_rule": "existing_bounded_gaussian_drive_noise",
    },
    "drive_logit_gain": {
        "value_type": "bounded_float",
        "bounds": [0.0, 3.0],
        "mutation_rule": "existing_bounded_gaussian_drive_logit_gain",
    },
    "pred_lr": {
        "value_type": "bounded_float",
        "bounds": [0.0, 1.0],
        "mutation_rule": "existing_bounded_gaussian_pred_lr",
    },
    "drive_sp_hunger": {"value_type": "bounded_float", "bounds": [0.0, 1.0]},
    "drive_sp_pain": {"value_type": "bounded_float", "bounds": [0.0, 1.0]},
    "drive_sp_fatigue": {"value_type": "bounded_float", "bounds": [0.0, 1.0]},
    "drive_sp_fear": {"value_type": "bounded_float", "bounds": [0.0, 1.0]},
    "exp_prior_strength": {"value_type": "bounded_float", "bounds": [0.0, 1.0]},
    "exp_prior_noise": {"value_type": "bounded_float", "bounds": [0.0, 0.5]},
    "drive_birth_noise": {"value_type": "bounded_float", "bounds": [0.0, 0.2]},
    "reproduction_threshold": {
        "value_type": "bounded_float",
        "bounds": [2.0, 90.0],
        "mutation_rule": "existing_bounded_gaussian_reproduction_threshold",
    },
    "mutation_rate": {
        "value_type": "bounded_float",
        "bounds": [0.001, 0.2],
        "mutation_rule": "existing_fixed_gate_bounded_gaussian_mutation_rate",
    },
    "hidden_size": {"value_type": "categorical_integer", "bounds": None},
    "memory_dim": {
        "value_type": "categorical_integer",
        "bounds": [0, 64],
        "mutation_rule": "existing_memory_ladder_step",
    },
    "use_gru": {"value_type": "derived_boolean", "bounds": None},
    "policy_controller": {
        "value_type": "opaque_controller_record",
        "bounds": None,
        "checkpoint_schema": "weights_blob_v2",
        "mutation_rule": "existing_policy_weight_mutation_after_owned_copy",
    },
    "resource_handling_axis": {
        "value_type": "bounded_float",
        "bounds": [-1.0, 1.0],
        "mutation_rule": "existing_resource_handling_axis_mutation",
    },
    "family_head_residuals": {
        "value_type": "bounded_record",
        "bounds": "family_head_residual_schema_bounds",
        "mutation_rule": "existing_family_head_residual_mutation_when_enabled",
    },
}


_STRUCTURAL_LOCUS_SPECS: tuple[dict[str, Any], ...] = (
    {
        "semantic_locus_id": "structural.module_record",
        "field": "modules[*]",
        "value_type": "homology_aligned_module_record",
    },
    {
        "semantic_locus_id": "structural.module_live_operator_parameters",
        "field": "modules[*].params",
        "value_type": "whole_module_parameter_record",
    },
    {
        "semantic_locus_id": "structural.input_routing_record",
        "field": "modules[*].params.observation_routing",
        "value_type": "whole_routing_record",
    },
    {
        "semantic_locus_id": "structural.output_routing_record",
        "field": "modules[*].params.output_routing",
        "value_type": "whole_routing_record",
    },
    {
        "semantic_locus_id": "structural.developmental_record",
        "field": "modules[*].params.developmental_expression",
        "value_type": "whole_developmental_record",
    },
    {
        "semantic_locus_id": "structural.dormancy_record",
        "field": "modules[*].params.structural_activity.dormant",
        "value_type": "boolean_in_module_record",
    },
    {
        "semantic_locus_id": "structural.connection_record",
        "field": "connections[*]",
        "value_type": "homology_endpoint_aligned_connection_record",
    },
)


_UNCHANGED_LOCUS_SPECS: tuple[dict[str, Any], ...] = (
    {
        "semantic_locus_id": "categorical.sensor_channel_mask_bits",
        "field": "sensor_channel_mask",
        "value_type": "fixed_shape_integer_bit_loci",
        "inheritance_mode": "categorical_parent_copy",
        "linkage_group": "independent_sensor_mask_bits_v1",
        "bounds": "implemented_sensor_channel_mask",
        "mutation_rule": "existing_sensor_channel_mask_bit_toggle_when_enabled",
    },
    {
        "semantic_locus_id": "categorical.sensor_radius",
        "field": "sensor_radius",
        "value_type": "categorical_integer",
        "inheritance_mode": "categorical_parent_copy",
        "linkage_group": "sensor_radius_v1",
        "bounds": "implemented_sensor_radius_ladder_and_environment_view_radius",
        "mutation_rule": "existing_sensor_radius_step_when_enabled",
    },
    {
        "semantic_locus_id": "categorical.introspection_channel_mask_bits",
        "field": "introspection_channel_mask",
        "value_type": "fixed_shape_integer_bit_loci",
        "inheritance_mode": "categorical_parent_copy",
        "linkage_group": "independent_introspection_mask_bits_v1",
        "bounds": "implemented_introspection_channel_mask",
        "mutation_rule": "existing_introspection_mask_bit_toggle_when_enabled",
    },
    {
        "semantic_locus_id": "categorical.self_regulation_control_mask_bits",
        "field": "self_regulation_control_mask",
        "value_type": "fixed_shape_integer_bit_loci",
        "inheritance_mode": "categorical_parent_copy",
        "linkage_group": "independent_self_regulation_mask_bits_v1",
        "bounds": "implemented_self_regulation_control_mask",
        "mutation_rule": "existing_self_regulation_mask_bit_toggle_when_enabled",
    },
    {
        "semantic_locus_id": "categorical.self_modification_target_mask_bits",
        "field": "self_modification_target_mask",
        "value_type": "fixed_shape_integer_bit_loci",
        "inheritance_mode": "categorical_parent_copy",
        "linkage_group": "independent_self_modification_mask_bits_v1",
        "bounds": "implemented_self_modification_target_mask",
        "mutation_rule": "existing_self_modification_mask_bit_toggle_when_enabled",
    },
    {
        "semantic_locus_id": "categorical.directed_inheritance_channel_mask_bits",
        "field": "directed_inheritance_channel_mask",
        "value_type": "fixed_shape_integer_bit_loci",
        "inheritance_mode": "categorical_parent_copy",
        "linkage_group": "independent_directed_inheritance_mask_bits_v1",
        "bounds": "implemented_directed_inheritance_channel_mask",
        "mutation_rule": "existing_directed_inheritance_mask_bit_toggle_when_enabled",
    },
    {
        "semantic_locus_id": "developmental.top_level_program",
        "field": "developmental_program",
        "value_type": "canonical_developmental_record",
        "inheritance_mode": "whole_record_donor",
        "linkage_group": "top_level_developmental_program_v1",
        "bounds": "developmental_program_v1_schema",
        "mutation_rule": "none_in_crossover_genome",
    },
    {
        "semantic_locus_id": "runtime.offspring_prediction_priors",
        "field": "AgentStats.exp_dE_and_exp_dInj",
        "value_type": "weak_non_genomic_inherited_runtime_vectors",
        "inheritance_mode": "midpoint_legacy",
        "linkage_group": "prediction_prior_vectors_legacy_v1",
        "bounds": "runtime_stats_schema",
        "mutation_rule": "parent_midpoint_then_attenuation_and_configured_birth_noise",
        "checkpoint_schema": "agent_stats_runtime_state",
        "schema_owner": "agents.agent.AgentStats_and_evolution.population_pkg.spawn",
    },
)


def _locus_registry_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block_id, fields in LINKAGE_BLOCKS:
        for field_name in fields:
            details = dict(_FIELD_METADATA[field_name])
            mode = "whole_record_donor" if field_name in {
                "policy_controller",
                "family_head_residuals",
            } else "categorical_parent_copy" if field_name in {
                "hidden_size",
                "memory_dim",
                "use_gru",
            } else "parent_copy"
            rows.append(
                {
                    "semantic_locus_id": f"flat.{field_name}",
                    "field": field_name,
                    "inheritance_mode": mode,
                    "linkage_group": block_id,
                    "schema_owner": (
                        "agents.agent_policy.MLPPolicy_and_agents.gru_policy.GRUPolicy"
                        if field_name == "policy_controller"
                        else "agents.genome.Genome"
                    ),
                    "checkpoint_schema": details.pop(
                        "checkpoint_schema", "modular_genome_v1"
                    ),
                    "mutation_rule": details.pop(
                        "mutation_rule", "none_after_inheritance"
                    ),
                    "compatibility_version": INHERITANCE_COMPATIBILITY_VERSION,
                    **details,
                }
            )
    for spec in _STRUCTURAL_LOCUS_SPECS:
        rows.append(
            {
                **spec,
                "inheritance_mode": "homology_record_donor",
                "linkage_group": "structural_homology_connected_component_v1",
                "schema_owner": "agents.modular_genome_v1",
                "bounds": "schema_and_fixed_capacity_caps",
                "mutation_rule": "existing_frozen_c1_operator_schedule_after_inheritance",
                "checkpoint_schema": "modular_genome_v1",
                "compatibility_version": INHERITANCE_COMPATIBILITY_VERSION,
            }
        )
    for spec in _UNCHANGED_LOCUS_SPECS:
        rows.append(
            {
                **spec,
                "schema_owner": spec.get("schema_owner", "agents.genome.Genome"),
                "checkpoint_schema": spec.get(
                    "checkpoint_schema", "modular_genome_v1"
                ),
                "compatibility_version": INHERITANCE_COMPATIBILITY_VERSION,
            }
        )
    return rows


def validate_inheritance_profile(profile: Any) -> str:
    value = str(profile or LEGACY_INHERITANCE_PROFILE).strip()
    if value not in {
        LEGACY_INHERITANCE_PROFILE,
        LINKED_INHERITANCE_PROFILE,
        DIPLOID_INHERITANCE_PROFILE,
    }:
        raise ValueError(f"unsupported inheritance_profile={value!r}")
    return value


def validate_production_inheritance_profile(profile: Any) -> str:
    """Reject retired profiles before production owner construction."""

    value = validate_inheritance_profile(profile)
    if value in RETIRED_INHERITANCE_PROFILES:
        raise ValueError(
            "P12_RETIRED_INHERITANCE_PROFILE "
            f"profile={value!r} disposition=historical_analysis_only"
        )
    return value


def active_inheritance_profile(sim_cfg: Any) -> str:
    if sim_cfg is None:
        return LEGACY_INHERITANCE_PROFILE
    return validate_production_inheritance_profile(
        getattr(sim_cfg, "inheritance_profile", LEGACY_INHERITANCE_PROFILE)
    )


def validate_inheritance_execution_lane(
    sim_cfg: Any,
    *,
    policy_device: Any = None,
    bank_device: Any = None,
) -> None:
    """Reject retired or unqualified inheritance execution before dispatch.

    Genome V2 uses its separately bound package/route authority. The remaining
    diploid profile stays on its currently admitted Python CPU boundary.
    """

    profile = validate_inheritance_profile(
        getattr(sim_cfg, "inheritance_profile", LEGACY_INHERITANCE_PROFILE)
    )
    if profile in RETIRED_INHERITANCE_PROFILES:
        raise RuntimeError(
            "P12_RETIRED_INHERITANCE_PROFILE "
            f"profile={profile!r} production_execution=false"
        )
    genome_upgrade_v2_enabled = bool(
        getattr(sim_cfg, "genome_upgrade_v2_enable", False)
    )
    if genome_upgrade_v2_enabled:
        if profile != DIPLOID_INHERITANCE_PROFILE:
            raise RuntimeError(
                "GENOME_UPGRADE_V2_REQUIRES_DIPLOID_INHERITANCE "
                f"profile={profile}"
            )
        from evolution.genome_upgrade_v2_package_v1 import (
            validate_checkpoint_authority_binding_v1,
        )

        try:
            validate_checkpoint_authority_binding_v1(
                getattr(
                    sim_cfg,
                    "genome_v2_checkpoint_authority_binding",
                    (),
                )
            )
        except ValueError as exc:
            raise RuntimeError(
                "GENOME_UPGRADE_V2_UNQUALIFIED_NATIVE_LANE "
                f"profile={profile}"
            ) from exc
        # The production provider derives this immutable tuple from the exact
        # package receipt and execution route before runner construction.
        return
    if profile != DIPLOID_INHERITANCE_PROFILE:
        return
    requested_device = str(getattr(sim_cfg, "device", "auto") or "auto").lower()
    requested_bank = str(getattr(sim_cfg, "bank_device", "auto") or "auto").lower()
    if requested_device not in {"cpu", "auto", ""}:
        raise RuntimeError(
            "RECOMB1_UNSUPPORTED_EXECUTION_LANE "
            f"profile={profile} requested_device={requested_device}"
        )
    if requested_bank not in {"cpu", "auto", ""}:
        raise RuntimeError(
            "RECOMB1_UNSUPPORTED_EXECUTION_LANE "
            f"profile={profile} requested_bank_device={requested_bank}"
        )
    effective_device = str(policy_device).lower() if policy_device is not None else requested_device
    effective_bank = str(bank_device).lower() if bank_device is not None else requested_bank
    if effective_device not in {"cpu", "auto", ""}:
        raise RuntimeError(
            "RECOMB1_UNSUPPORTED_EXECUTION_LANE "
            f"profile={profile} device={effective_device}"
        )
    if effective_bank not in {"cpu", "auto", ""}:
        raise RuntimeError(
            "RECOMB1_UNSUPPORTED_EXECUTION_LANE "
            f"profile={profile} bank_device={effective_bank}"
        )
    native_authority_flags = (
        "native_runtime_enable",
        "runnercore_native_enable",
        "runnercore1_native_enable",
        "runnercore2_native_enable",
        "runnercore3_native_enable",
        "runnercore4_native_enable",
        "runnercore5_native_enable",
        "runnercore6_native_enable",
        "runnercore9_native_enable",
        "runnercore10_native_enable",
        "runnercore11_native_enable",
        "cxxrewrite_array_backend_authoritative_enable",
        "cxxrewrite_native_tick_authoritative_enable",
        "cxxrewrite_resident_frame_native_packing_authoritative_enable",
        "cxxcore13_state_backend_enable",
    )
    enabled_native = sorted(
        name for name in native_authority_flags if bool(getattr(sim_cfg, name, False))
    )
    if enabled_native:
        raise RuntimeError(
            "RECOMB1_UNSUPPORTED_EXECUTION_LANE "
            f"profile={profile} native_authority_flags={enabled_native!r}"
        )


def inheritance_registry_metadata() -> dict[str, Any]:
    blocks = []
    for block_id, fields in LINKAGE_BLOCKS:
        inheritance_mode = "parent_copy"
        if block_id in {"memory_controller_architecture_v1", "family_head_strategy_v1"}:
            inheritance_mode = "whole_record_donor"
        blocks.append(
            {
                "id": block_id,
                "fields": list(fields),
                "inheritance_mode": inheritance_mode,
            }
        )
    metadata = {
        "registry_version": INHERITANCE_REGISTRY_VERSION,
        "compatibility_version": INHERITANCE_COMPATIBILITY_VERSION,
        "legacy_profile": LEGACY_INHERITANCE_PROFILE,
        "linked_profile": LINKED_INHERITANCE_PROFILE,
        "parent_order_version": INHERITANCE_PARENT_ORDER_VERSION,
        "equal_genome_policy_tiebreak_version": EQUAL_GENOME_POLICY_TIEBREAK_VERSION,
        "budget_boundary": INHERITANCE_LINKAGE_BUDGET_BOUNDARY,
        "draws_per_offspring": LINKED_PROFILE_DRAW_COUNT,
        "migration_rng_draws": 0,
        "disabled_subsystem_policy": (
            "linked profile consumes its fixed seven flat-linkage draws even when an optional "
            "repaired field is disabled; legacy profile adds zero linkage draws"
        ),
        "mutation_after_inheritance_version": MUTATION_AFTER_INHERITANCE_VERSION,
        "supported_lane": SUPPORTED_VALIDATION_LANE,
        "exceptional_path_draw_contract": {
            "normal_linked_flat": "exactly_7_before_existing_mutation_schedule",
            "legacy_profile": "zero_added_linkage_draws_existing_schedule_byte_exact",
            "invalid_parent_payload": "rejected_at_typed_genome_boundary_before_reproduction_zero_added_linkage_draws",
            "missing_repaired_locus": (
                "required_fields_reject_at_typed_genome_boundary_with_zero_added_linkage_draws; "
                "legacy_compatible_optional_fields_receive_frozen_schema_defaults_before_reproduction"
            ),
            "malformed_individual_structural_schema": (
                "rejected_during_canonical_parent_order_before_flat_or_c2_donor_draws"
            ),
            "cross_parent_structural_incompatibility": (
                "exactly_7_flat_then_zero_c2_donor_draws_before_reject"
            ),
            "unmatched_homolog": "exactly_7_flat_then_zero_c2_donor_draws_for_unmatched_union_record",
            "cap_blocked_unmatched_structure": "exactly_7_flat_then_zero_c2_donor_draws_for_cap_loss",
            "matched_structural_component": "exactly_7_flat_plus_one_c2_draw_per_component",
            "checkpoint_migration": "zero_draws",
        },
        "allowed_inheritance_modes": list(ALLOWED_INHERITANCE_MODES),
        "loci": _locus_registry_rows(),
        "blocks": blocks,
        "structural_linkage_rule": {
            "id": "homology_connected_component_v1",
            "inheritance_mode": "homology_record_donor",
            "unit": (
                "each connected component of matched native homologs joined by matched ordinary "
                "connections or developmental dependency edges present in either parent"
            ),
            "draw_boundary": "mutation.c2.recombination_lineage",
            "draw_formula": "ctx.recomb_draws=component_count",
            "draws_per_component": 1,
            "unmatched_rule": "existing module and connection union with caps; zero donor draws",
            "pre_draw_reject_code": "DET_C2_RECOMB1_UNASSIGNED_MATCHED_CONNECTION",
            "draw_dependent_conflict_rule": (
                "consume full component schedule once then canonical validation fails closed without redraw"
            ),
        },
    }
    canonical = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    metadata["registry_metadata_sha256"] = hashlib.sha256(canonical).hexdigest()
    return metadata


def diploid_flat_linkage_component_contract_v1() -> dict[str, Any]:
    """Describe the retained flat-linkage prelude without reviving its profile."""

    payload = {
        "schema_version": DIPLOID_FLAT_LINKAGE_COMPONENT_SCHEMA_VERSION,
        "component_id": "genome.diploid_flat_linkage_compatibility.v1",
        "owner_id": "eiel_global_genome_inheritance_v1",
        "standalone_profile_id": None,
        "retired_source_profile": LINKED_INHERITANCE_PROFILE,
        "representation": {
            "linkage_blocks": list(LINKAGE_BLOCK_IDS),
            "draw_count": LINKED_PROFILE_DRAW_COUNT,
            "parent_order_version": INHERITANCE_PARENT_ORDER_VERSION,
        },
        "inheritance": "canonical_parent_order_then_whole_block_donor_copy_v1",
        "mutation": MUTATION_AFTER_INHERITANCE_VERSION,
        "execution": (
            "subordinate_diploid_and_genome_v2_python_reproduction_prelude"
        ),
        "checkpointing": (
            "owned_by_diploid_or_genome_v2_checkpoint_authority_only"
        ),
        "telemetry": "route_local_donor_context_not_public_campaign_authority",
        "native_cuda_ownership": (
            "qualified_genome_v2_package_and_execution_route_only"
        ),
        "route_qualification": "cannot_qualify_or_execute_standalone",
        "claims_boundary": (
            "mechanic compatibility only; no ecological or evolutionary claim"
        ),
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    return payload


def inspect_historical_inheritance_checkpoint_v1(
    recorded: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate old linked metadata for analysis without admitting execution."""

    if not isinstance(recorded, Mapping):
        raise ValueError("historical inheritance checkpoint metadata must be a mapping")
    payload = dict(recorded)
    if payload.get("profile") != LINKED_INHERITANCE_PROFILE:
        raise ValueError("checkpoint does not use the retired linked profile")
    expected = historical_linked_checkpoint_metadata_v1()
    if payload != expected:
        raise ValueError("retired linked checkpoint metadata contract mismatch")
    metadata_sha256 = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    return {
        "schema_version": HISTORICAL_CHECKPOINT_DISPOSITION_SCHEMA_VERSION,
        "profile": LINKED_INHERITANCE_PROFILE,
        "decision": RETIREMENT_DECISION,
        "metadata_sha256": metadata_sha256,
        "historical_analysis_readable": True,
        "production_execution_allowed": False,
        "production_resume_allowed": False,
        "migration_supported": False,
        "migration_rng_draws": 0,
        "semantic_loss": "not_applicable_no_migration",
    }


def historical_linked_checkpoint_metadata_v1() -> dict[str, Any]:
    """Return the frozen metadata envelope used only to inspect old bundles."""

    return {
        "schema_version": INHERITANCE_CHECKPOINT_SCHEMA_VERSION,
        "profile": LINKED_INHERITANCE_PROFILE,
        **inheritance_registry_metadata(),
    }


def inheritance_checkpoint_metadata(sim_cfg: Any) -> dict[str, Any] | None:
    """Return production checkpoint metadata; legacy emits nothing."""

    profile = validate_production_inheritance_profile(
        getattr(sim_cfg, "inheritance_profile", LEGACY_INHERITANCE_PROFILE)
    )
    if profile == LEGACY_INHERITANCE_PROFILE:
        return None
    metadata = {
        "schema_version": INHERITANCE_CHECKPOINT_SCHEMA_VERSION,
        "profile": profile,
        **inheritance_registry_metadata(),
    }
    if profile == DIPLOID_INHERITANCE_PROFILE:
        metadata.update(
            {
                "ploidy": 2,
                "diploid_schema_version": DIPLOID_GENOME_SCHEMA_VERSION,
                "linkage_map_hash": DIPLOID_LINKAGE_MAP_SHA256,
                "expression_profile": DIPLOID_EXPRESSION_PROFILE,
                "recombination_profile": DIPLOID_RECOMBINATION_PROFILE,
                "mutation_profile": DIPLOID_MUTATION_PROFILE,
                "admitted_locus_registry_hash": DIPLOID_LOCUS_REGISTRY_SHA256,
            }
        )
    return metadata


def canonical_hereditary_payload_sha256(genome: Any) -> str:
    from agents.modular_genome_v1 import (  # noqa: WPS433
        B2_FROZEN_CAPS,
        b2_capacity_neutral_ordering_payload,
        resolve_b2_caps,
    )

    caps = resolve_b2_caps(getattr(genome, "b2_caps", None))
    canonical_hash = getattr(genome, "canonical_hash", None)
    if caps == B2_FROZEN_CAPS and callable(canonical_hash):
        value = str(canonical_hash() or "")
        if value:
            return value
    canonical_payload = getattr(genome, "canonical_payload", None)
    if callable(canonical_payload):
        payload = canonical_payload()
    else:
        to_dict = getattr(genome, "to_dict", None)
        payload = to_dict() if callable(to_dict) else vars(genome)
    encoded = json.dumps(
        b2_capacity_neutral_ordering_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_parent_order(parent_a: Any, parent_b: Any) -> tuple[tuple[Any, Any], tuple[str, str]]:
    """Order parents by frozen hereditary-payload digest without consuming RNG."""

    rows = [
        (canonical_hereditary_payload_sha256(parent_a), 0, parent_a),
        (canonical_hereditary_payload_sha256(parent_b), 1, parent_b),
    ]
    rows.sort(key=lambda row: (row[0], row[1]))
    return (rows[0][2], rows[1][2]), (rows[0][0], rows[1][0])


def draw_linkage_parent_context(parent_a: Any, parent_b: Any, rng_manager: Any) -> dict[str, Any]:
    """Consume exactly seven mutation.py draws and return transient donor context."""

    if rng_manager is None:
        raise RuntimeError(
            "[DETERMINISM_VIOLATION]\n"
            "code=DET_FORBIDDEN_RNG_API\n"
            "context=linked inheritance requires RNGManager mutation.py stream\n"
        )
    mutation_rng = rng_manager.py_rng("mutation")
    if mutation_rng is None:
        raise RuntimeError(
            "[DETERMINISM_VIOLATION]\n"
            "code=DET_RNG_UNKNOWN_STREAM\n"
            "context=missing mutation.py stream for linked inheritance\n"
        )

    canonical_parents, canonical_digests = canonical_parent_order(parent_a, parent_b)
    expected = rng_manager.expected_delta_for_boundary(
        INHERITANCE_LINKAGE_BUDGET_BOUNDARY,
        ctx={"linkage_draws": LINKED_PROFILE_DRAW_COUNT},
    )
    with rng_manager.budget_scope(INHERITANCE_LINKAGE_BUDGET_BOUNDARY, expected_delta=expected):
        donor_ordinals = tuple(
            0 if mutation_rng.random() < 0.5 else 1
            for _ in range(LINKED_PROFILE_DRAW_COUNT)
        )

    donors = {
        block_id: canonical_parents[ordinal]
        for block_id, ordinal in zip(LINKAGE_BLOCK_IDS, donor_ordinals)
    }
    return {
        "profile": LINKED_INHERITANCE_PROFILE,
        "registry_version": INHERITANCE_REGISTRY_VERSION,
        "parent_order_version": INHERITANCE_PARENT_ORDER_VERSION,
        "canonical_parent_digests": canonical_digests,
        "donor_ordinals": donor_ordinals,
        "donor_digests": {
            block_id: canonical_digests[ordinal]
            for block_id, ordinal in zip(LINKAGE_BLOCK_IDS, donor_ordinals)
        },
        "donors": donors,
    }


__all__ = [
    "LEGACY_INHERITANCE_PROFILE",
    "LINKED_INHERITANCE_PROFILE",
    "RETIRED_INHERITANCE_PROFILES",
    "RETIREMENT_DECISION",
    "DIPLOID_INHERITANCE_PROFILE",
    "INHERITANCE_REGISTRY_VERSION",
    "INHERITANCE_CHECKPOINT_SCHEMA_VERSION",
    "ALLOWED_INHERITANCE_MODES",
    "LINKED_PROFILE_DRAW_COUNT",
    "LINKAGE_BLOCK_IDS",
    "active_inheritance_profile",
    "canonical_parent_order",
    "diploid_flat_linkage_component_contract_v1",
    "historical_linked_checkpoint_metadata_v1",
    "inspect_historical_inheritance_checkpoint_v1",
    "validate_inheritance_execution_lane",
    "validate_inheritance_profile",
    "validate_production_inheritance_profile",
    "inheritance_registry_metadata",
    "inheritance_checkpoint_metadata",
]
