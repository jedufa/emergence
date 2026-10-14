from __future__ import annotations

"""Authoritative, deterministic evolutionary-substrate capacity contract.

The contract is reporting and validation data.  It does not resize storage,
change mutation probabilities, alter selection, or introduce semantic capacity.
Values are derived from the runtime registries and schema constants that own
them.  Explicit lane entries state when an envelope is not owned by this tree.
"""

import hashlib
import json
from typing import Any, Iterable, Mapping

from agents.agent import (
    ACTION_VOCABULARY,
    ACTION_VOCABULARY_SCHEMA_VERSION,
    DRIVE_VOCABULARY,
    DRIVE_VOCABULARY_SCHEMA_VERSION,
)
from agents.chemistry import CHEMISTRY_SCHEMA_VERSION, CHEMISTRY_SLOT_NAMES
from agents.cognition import FIXED_POLICY_HIDDEN_SIZE, MEMORY_LADDER
from agents.developmental_program_v1 import DEVELOPMENTAL_PROGRAM_SCHEMA_VERSION
from agents.directed_inheritance import (
    DIRECTED_INHERITANCE_REGISTRY_SCHEMA_VERSION,
    directed_inheritance_channel_registry,
    implemented_directed_inheritance_channel_mask,
)
from agents.introspection import implemented_introspection_channel_mask
from agents.modular_genome_v1 import (
    B2_FROZEN_CAPS,
    B2_INTERFACE_SCHEMA_VERSION,
    MODULAR_GENOME_SCHEMA_VERSION,
    canonical_payload_from_genome_dict,
    resolve_b2_caps,
)
from agents.structural_identity_v1 import (
    STRUCTURAL_ANCESTRY_SCHEMA_VERSION,
    STRUCTURAL_IDENTITY_SCHEMA_VERSION,
    STRUCTURAL_LEGACY_MIGRATION_VERSION,
    TOPOLOGY_POSITION_SCHEMA_VERSION,
)
from agents.perception import (
    PERCEPTION_CHANNEL_REGISTRY_SCHEMA_VERSION,
    PERCEPTION_ROUTING_CONTRACT_VERSION,
    SENSOR_RADIUS_LADDER,
    perception_channel_registry,
    implemented_sensor_channel_mask,
)
from agents.resource_handling import RESOURCE_HANDLING_AXIS_MAX, RESOURCE_HANDLING_AXIS_MIN
from agents.self_modification import implemented_self_modification_target_mask
from agents.self_regulation import implemented_self_regulation_control_mask
from evolution.operators_c1 import (
    C1_OPERATOR_CATALOG_VERSION,
    operator_catalog,
    structural_operator_dependency_graph,
)
from evolution.recombination_c2 import (
    C2_FROZEN_ALIGNMENT_KEY,
    C2_LEGACY_UNRESOLVED_POLICY,
    C2_RECOMBINATION_VERSION,
)
from evolution.structural_live_modules_v1 import (
    LIVE_MODULE_SCHEMA_VERSION,
    LIVE_OPERATOR,
    LIVE_PROFILE,
    MAX_REST_RESIDUAL,
)
from evolution.structural_gated_recurrent_v1 import (
    GATED_EQUATION_VERSION,
    GATED_EXECUTION_OPERATION_ESTIMATE,
    GATED_NUMERIC_VERSION,
    GATED_OPERATION_BUDGET_VERSION,
    GATED_OPERATOR,
    GATED_PARAMETER_BOUNDS,
    GATED_PREACTIVATION_BOUND,
    GATED_RECORD_SCHEMA_VERSION,
    GATED_STATE_BOUND,
    GATED_STATE_WIDTH,
)
from evolution.structural_sustainable_growth_v1 import (
    ACTIVITY_RECORD_KEY,
    ACTIVITY_SCHEMA_VERSION,
    BLOAT_CLASSIFICATION_VERSION,
    COST_COEFFICIENTS,
    COST_MODE,
    COST_PROFILE,
    COST_SCHEMA_VERSION,
    SUSTAINABLE_GROWTH_PROFILE,
    classify_bloat,
)
from evolution.structural_output_routing_v1 import (
    ACTION_SEMANTIC_REGISTRY_VERSION,
    ADMITTED_ACTION_SEMANTIC_IDS,
    COMBINED_REST_COMPATIBILITY_VERSION,
    MAX_ACTION_RESIDUAL,
    MAX_OUTPUT_BINDINGS_PER_AGENT,
    MAX_OUTPUT_BINDINGS_PER_MODULE,
    OUTPUT_ROUTING_ARBITRATION_VERSION,
    OUTPUT_ROUTING_AUTHORITY_ORDER_VERSION,
    OUTPUT_ROUTING_OPERATOR_PROFILE,
    OUTPUT_ROUTING_PROFILE,
    OUTPUT_ROUTING_SCHEMA_VERSION,
)
from evolution.structural_development_v1 import (
    DEPENDENCY_IDENTITY_AUTHORITY,
    DEVELOPMENTAL_FEATURE_PROFILE,
    DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
    DEVELOPMENTAL_STAGE_IDS,
    DEVELOPMENTAL_STAGE_PROFILE_VERSION,
    INACTIVE_OUTPUT_POLICY,
    INACTIVE_STATE_POLICY,
    LATE_START_STEP,
    MATURE_START_STEP,
    MAX_DEPENDENCY_EDGES_PER_AGENT,
    MAX_PREREQUISITES_PER_MODULE,
    MAX_STAGE_GAIN,
    MIN_STAGE_GAIN,
    SUPPORTED_LANE,
)
from evolution.capacity_errors import (
    CapacityIntegrityError,
    REASON_CONNECTION_CAPACITY,
    REASON_FANIN_CAPACITY,
    REASON_FANOUT_CAPACITY,
    REASON_LOCALITY_INVALID,
    REASON_MODULE_CAPACITY,
    REASON_RUNTIME_ALLOCATION,
    REASON_STRUCTURAL_BOOTSTRAP,
    classify_structural_capacity_error,
)


CAPACITY_CONTRACT_SCHEMA_VERSION = "eiel_evolutionary_capacity_contract_v1"
DEFAULT_CAPACITY_EPOCH_ID = "capacity-epoch-0"
CAPACITY_WARNING_HEADROOM_FRACTION = 0.20

def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_contract_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(payload))).hexdigest()


def validate_active_hereditary_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate newly produced hereditary data with fail-closed semantics.

    Compatibility loaders may still request explicit clipping.  Active
    evolution must call this boundary (or an equivalent reject-policy path).
    """

    raw = dict(payload)
    raw["b2_invalid_policy"] = "reject"
    return canonical_payload_from_genome_dict(raw)


def _active_registry_count(records: Iterable[Any]) -> int:
    return sum(
        1
        for record in records
        if bool(getattr(record, "active_now", False))
        and not bool(getattr(record, "future_reserved", False))
    )


def _active_lane(sim_cfg: Any) -> str:
    if sim_cfg is None:
        return "python_reference"
    if bool(getattr(sim_cfg, "native_runtime_enable", False)):
        return "native_owner"
    device = str(getattr(sim_cfg, "device", "cpu") or "cpu").lower()
    if device.startswith("cuda"):
        return "python_reference_cuda_device"
    return "python_reference"


def _observation_width(env_cfg: Any) -> int | None:
    if env_cfg is None:
        return None
    from env.gridworld_pkg.patch_world import patch_world_total_observation_size

    return int(patch_world_total_observation_size(env_cfg))


def structural_bootstrap_reachability(sim_cfg: Any = None) -> dict[str, Any]:
    seeded = bool(getattr(sim_cfg, "structural_payload_activation_enable", False)) if sim_cfg is not None else False
    operators_enabled = bool(getattr(sim_cfg, "genome_c1_enable", True)) if sim_cfg is not None else True
    bootstrap_enabled = bool(getattr(sim_cfg, "structural_bootstrap_enable", False)) if sim_cfg is not None else False
    operators = tuple(spec.op_id for spec in operator_catalog())
    if seeded:
        classification = "reachable_only_via_explicit_seed_profile"
        reasons = ["ordinary_random_founders_are_empty", "external_seed_profile_supplies_first_structure"]
    elif operators_enabled and bootstrap_enabled:
        classification = "bootstrap_reachable_feature_gated"
        reasons = [
            "ordinary_random_founders_are_empty",
            "fixed_budget_duplicate_operator_creates_first_and_second_neutral_modules",
            "fixed_budget_rewire_operator_creates_first_zero_weight_connection",
        ]
    elif operators_enabled:
        classification = "not_bootstrap_reachable"
        reasons = ["ordinary_random_founders_are_empty", "current_structural_operators_require_existing_structure"]
    else:
        classification = "not_reachable_operators_disabled"
        reasons = ["ordinary_random_founders_are_empty", "structural_operators_disabled"]
    return {
        "classification": classification,
        "ordinary_founder_payload": "empty",
        "structural_operators_enabled": operators_enabled,
        "structural_bootstrap_enabled": bootstrap_enabled,
        "operator_catalog_version": C1_OPERATOR_CATALOG_VERSION,
        "operator_ids": list(operators),
        "reasons": reasons,
        "operator_dependency_graph": structural_operator_dependency_graph(
            bootstrap_enabled=bool(operators_enabled and bootstrap_enabled)
        ),
    }


def build_capacity_contract(
    *,
    sim_cfg: Any = None,
    env_cfg: Any = None,
    capacity_epoch_id: str | None = None,
) -> dict[str, Any]:
    from evolution.reproduction import _MUT_RATE_BOUNDS, _MUT_SPECS
    from simulation.checkpoint_bundle_v2 import SCHEMA_VERSION as CHECKPOINT_SCHEMA_VERSION

    active_b2_caps = resolve_b2_caps(
        {
            "max_modules_per_agent": getattr(
                sim_cfg, "genome_b2_max_modules_per_agent", B2_FROZEN_CAPS["max_modules_per_agent"]
            ),
            "max_connections_per_agent": getattr(
                sim_cfg, "genome_b2_max_connections_per_agent", B2_FROZEN_CAPS["max_connections_per_agent"]
            ),
            "max_module_fanin": getattr(
                sim_cfg, "genome_b2_max_module_fanin", B2_FROZEN_CAPS["max_module_fanin"]
            ),
            "max_module_fanout": getattr(
                sim_cfg, "genome_b2_max_module_fanout", B2_FROZEN_CAPS["max_module_fanout"]
            ),
        }
    )

    perception = tuple(perception_channel_registry())
    inheritance = tuple(directed_inheritance_channel_registry())
    active_perception = _active_registry_count(perception)
    routing_eligible_ids = tuple(
        sorted(str(record.channel_id) for record in perception if bool(getattr(record, "routing_eligible", False)))
    )
    active_inheritance = _active_registry_count(inheritance)
    scalar_bounds = {
        name: {"minimum": float(spec[1]), "maximum": float(spec[2]), "sigma": float(spec[0])}
        for name, spec in sorted(_MUT_SPECS.items())
    }
    scalar_bounds["mutation_rate"] = {
        "minimum": float(_MUT_RATE_BOUNDS[0]),
        "maximum": float(_MUT_RATE_BOUNDS[1]),
        "sigma": None,
    }
    scalar_locus_ranges: dict[str, Any] = {
        **{name: dict(bounds) for name, bounds in scalar_bounds.items()},
        "hidden_size": {"allowed": [int(FIXED_POLICY_HIDDEN_SIZE)]},
        "use_gru": {"allowed": [False, True]},
        "memory_dim": {"allowed": [int(v) for v in MEMORY_LADDER]},
        "sensor_channel_mask": {"minimum": 0, "maximum": int(implemented_sensor_channel_mask())},
        "sensor_radius": {"allowed": [int(v) for v in SENSOR_RADIUS_LADDER]},
        "introspection_channel_mask": {"minimum": 0, "maximum": int(implemented_introspection_channel_mask())},
        "self_regulation_control_mask": {"minimum": 0, "maximum": int(implemented_self_regulation_control_mask())},
        "self_modification_target_mask": {"minimum": 0, "maximum": int(implemented_self_modification_target_mask())},
        "directed_inheritance_channel_mask": {"minimum": 0, "maximum": int(implemented_directed_inheritance_channel_mask())},
        "drive_sp_hunger": {"minimum": 0.0, "maximum": 1.0},
        "drive_sp_pain": {"minimum": 0.0, "maximum": 1.0},
        "drive_sp_fatigue": {"minimum": 0.0, "maximum": 1.0},
        "drive_sp_fear": {"minimum": 0.0, "maximum": 1.0},
        "exp_prior_strength": {"minimum": 0.0, "maximum": 1.0},
        "exp_prior_noise": {"minimum": 0.0, "maximum": 0.5},
        "drive_birth_noise": {"minimum": 0.0, "maximum": 0.2},
        "resource_handling_axis": {
            "minimum": float(RESOURCE_HANDLING_AXIS_MIN),
            "maximum": float(RESOURCE_HANDLING_AXIS_MAX),
        },
    }
    scalar_loci = sorted(scalar_locus_ranges)
    active_lane = _active_lane(sim_cfg)
    epoch = str(
        capacity_epoch_id
        or (getattr(sim_cfg, "capacity_epoch_id", "") if sim_cfg is not None else "")
        or DEFAULT_CAPACITY_EPOCH_ID
    )
    reachability = structural_bootstrap_reachability(sim_cfg)
    contract: dict[str, Any] = {
        "schema_version": CAPACITY_CONTRACT_SCHEMA_VERSION,
        "capacity_epoch_id": epoch,
        "freeze_scope": {
            "campaign_frozen": [
                "hidden_width",
                "observation_width",
                "structural_limits",
                "runtime_live_slots",
            ],
            "platform_versioned": [
                "action_vocabulary",
                "drive_vocabulary",
                "chemistry_schema",
                "perception_channel_registry",
                "checkpoint_schema",
            ],
        },
        "genome": {
            "schema_version": MODULAR_GENOME_SCHEMA_VERSION,
            "scalar_locus_count": len(scalar_loci),
            "scalar_loci": scalar_loci,
            "scalar_locus_ranges": {name: scalar_locus_ranges[name] for name in scalar_loci},
            "directly_mutable_scalar_ranges": scalar_bounds,
            "status_dimensions": {
                "storable": True,
                "mutable": True,
                "expressed": "field_specific",
                "selectable": "world_and_cost_dependent",
                "reachable": "field_specific",
                "sustainable": "not_asserted_by_contract",
            },
        },
        "policy": {
            "architecture": "fixed_mlp_or_gru",
            "hidden_width": int(FIXED_POLICY_HIDDEN_SIZE),
            "hidden_width_ladder": [int(FIXED_POLICY_HIDDEN_SIZE)],
            "memory_ladder": [int(v) for v in MEMORY_LADDER],
            "effective_memory_max": int(max(MEMORY_LADDER)),
            "recurrent_inheritance_mode": "single_recurrent_donor_copy",
            "escap7_fixed_shape_contract": {
                "central_gru_unchanged": True,
                "central_gru_hidden_width": int(FIXED_POLICY_HIDDEN_SIZE),
                "mlp_hidden_width": int(FIXED_POLICY_HIDDEN_SIZE),
                "policy_tensor_shape_change": 0,
            },
        },
        "observations": {
            "registry_schema_version": PERCEPTION_CHANNEL_REGISTRY_SCHEMA_VERSION,
            "routing_contract_version": PERCEPTION_ROUTING_CONTRACT_VERSION,
            "declared_channel_count": len(perception),
            "implemented_active_channel_count": active_perception,
            "enabled_channel_ids": sorted(
                str(record.channel_id)
                for record in perception
                if bool(record.active_now) and not bool(record.future_reserved)
            ),
            "sensor_radius_ladder": [int(v) for v in SENSOR_RADIUS_LADDER],
            "active_observation_width": _observation_width(env_cfg),
            "escap7_observation_width_change": 0,
            "routing_eligible_semantic_ids": list(routing_eligible_ids),
            "routing_eligible_semantic_count": len(routing_eligible_ids),
        },
        "actions": {
            "vocabulary_version": ACTION_VOCABULARY_SCHEMA_VERSION,
            "count": len(ACTION_VOCABULARY),
            "vocabulary": list(ACTION_VOCABULARY),
            "semantic_capacity_auto_expandable": False,
            "escap7_action_width_change": 0,
            "output_routing_registry_version": ACTION_SEMANTIC_REGISTRY_VERSION,
            "output_routing_admitted_semantic_ids": list(ADMITTED_ACTION_SEMANTIC_IDS),
        },
        "drives": {
            "vocabulary_version": DRIVE_VOCABULARY_SCHEMA_VERSION,
            "count": len(DRIVE_VOCABULARY),
            "vocabulary": list(DRIVE_VOCABULARY),
            "semantic_capacity_auto_expandable": False,
        },
        "chemistry": {
            "schema_version": CHEMISTRY_SCHEMA_VERSION,
            "slot_count": len(CHEMISTRY_SLOT_NAMES),
            "slot_names": list(CHEMISTRY_SLOT_NAMES),
            "semantic_capacity_auto_expandable": False,
        },
        "structure": {
            "interface_schema_version": B2_INTERFACE_SCHEMA_VERSION,
            "limits": {name: int(value) for name, value in sorted(active_b2_caps.items())},
            "active_science_invalid_policy": "reject",
            "legacy_compatibility_policy": "clip_with_explicit_compatibility_context",
            "reachability": reachability,
            "identity_schema_version": STRUCTURAL_IDENTITY_SCHEMA_VERSION,
            "ancestry_schema_version": STRUCTURAL_ANCESTRY_SCHEMA_VERSION,
            "topology_position_schema_version": TOPOLOGY_POSITION_SCHEMA_VERSION,
            "legacy_migration_version": STRUCTURAL_LEGACY_MIGRATION_VERSION,
            "recombination_version": C2_RECOMBINATION_VERSION,
            "recombination_alignment_key": C2_FROZEN_ALIGNMENT_KEY,
            "legacy_homology_policy": "unresolved_modules_are_unmatched_across_parents",
            "legacy_unresolved_recombination_policy": C2_LEGACY_UNRESOLVED_POLICY,
            "live_computation": {
                "runtime_schema_version": LIVE_MODULE_SCHEMA_VERSION,
                "operator": LIVE_OPERATOR,
                "operator_registry_version": "escap7_fixed_width_scalar_recurrent_operators_v1",
                "supported_operators": [
                    {
                        "operator": LIVE_OPERATOR,
                        "state_width": 1,
                        "estimated_scalar_operations_per_evaluation": 4,
                        "execution_rng_draws": 0,
                    },
                    {
                        "operator": GATED_OPERATOR,
                        "parameter_schema_version": GATED_RECORD_SCHEMA_VERSION,
                        "equation_version": GATED_EQUATION_VERSION,
                        "numeric_policy_version": GATED_NUMERIC_VERSION,
                        "state_width": int(GATED_STATE_WIDTH),
                        "parameter_bounds": {
                            name: [float(bounds[0]), float(bounds[1])]
                            for name, bounds in GATED_PARAMETER_BOUNDS.items()
                        },
                        "preactivation_bounds": [
                            -float(GATED_PREACTIVATION_BOUND),
                            float(GATED_PREACTIVATION_BOUND),
                        ],
                        "state_bounds": [-float(GATED_STATE_BOUND), float(GATED_STATE_BOUND)],
                        "output_bounds": [-float(GATED_STATE_BOUND), float(GATED_STATE_BOUND)],
                        "estimated_scalar_operations_per_evaluation": int(
                            GATED_EXECUTION_OPERATION_ESTIMATE
                        ),
                        "operation_budget_version": GATED_OPERATION_BUDGET_VERSION,
                        "execution_rng_draws": 0,
                    },
                ],
                "profile": LIVE_PROFILE,
                "storable": True,
                "executable": True,
                "active": bool(getattr(sim_cfg, "structural_live_modules_enable", False)) if sim_cfg is not None else False,
                "gated_recurrent_enabled": bool(
                    getattr(sim_cfg, "structural_gated_recurrent_enable", False)
                ) if sim_cfg is not None else False,
                "saturated": False,
                "owned_state_dimensions_per_executable_module": 1,
                "maximum_owned_state_dimensions": int(active_b2_caps["max_modules_per_agent"]),
                "fixed_storage_contract": {
                    "state_map": "structural_live_module_state",
                    "output_map": "structural_live_module_outputs",
                    "state_width_per_module": 1,
                    "new_arrays": 0,
                    "observation_width_change": 0,
                    "action_width_change": 0,
                    "central_gru_width_change": 0,
                },
                "typed_inputs": ["existing_drive_fatigue_scalar", "previous_tick_scalar_connection_output"],
                "typed_outputs": ["scalar_connection_output", "existing_rest_logit_residual"],
                "maximum_absolute_rest_logit_residual": float(MAX_REST_RESIDUAL),
                "cost_semantics": "reporting_only",
                "runtime_telemetry": {
                    "allocated_integrator_modules": "runtime_telemetry_only",
                    "allocated_gated_recurrent_modules": "runtime_telemetry_only",
                    "executable_gated_recurrent_modules": "runtime_telemetry_only",
                    "active_gated_recurrent_modules": "runtime_telemetry_only",
                    "inactive_gated_recurrent_modules": "runtime_telemetry_only",
                    "causally_active_gated_recurrent_modules": "runtime_telemetry_only",
                    "saturated_reset_gates": "runtime_telemetry_only",
                    "saturated_update_gates": "runtime_telemetry_only",
                    "saturated_candidates": "runtime_telemetry_only",
                    "saturated_gated_states": "runtime_telemetry_only",
                    "saturated_gated_outputs": "runtime_telemetry_only",
                    "gated_execution_rng_draws": "runtime_telemetry_only_always_zero",
                    "gated_execution_operation_estimate": "reporting_only",
                    "total_live_execution_operation_estimate": "reporting_only",
                    "parameter_bound_rejections": "zero_on_success_fail_closed_before_state_commit",
                },
                "mutation_telemetry": {
                    "operator_type_mutations": "escap7_attempt_rows_operation_acquire_neutral",
                    "gated_parameter_mutations": "escap7_attempt_rows_reset_update_candidate_or_output_operation",
                    "parameter_bound_rejections": "escap7_attempt_rows_invalid_proposal",
                    "gated_duplication_events": "c1_operator_reason_counts_gated_duplicate_classification",
                },
                "lifetime_state_inheritance": "newborn_zero_not_genetic",
                "supported_lanes": ["python_reference_cpu"],
                "unsupported_lanes": ["python_reference_cuda", "native_owner", "cuda_specialization", "policy_stride"],
                "overflow_policy": "clip_declared_numeric_bounds_reject_capacity_or_identity_mismatch",
            },
            "sustainable_growth": {
                "enabled": bool(
                    getattr(sim_cfg, "structural_sustainable_growth_enable", False)
                ) if sim_cfg is not None else False,
                "profile": SUSTAINABLE_GROWTH_PROFILE,
                "activity_record_key": ACTIVITY_RECORD_KEY,
                "activity_schema_version": ACTIVITY_SCHEMA_VERSION,
                "absent_activity_record_means": "active_legacy_semantics",
                "dormancy": {
                    "hereditary": True,
                    "reversible": True,
                    "executes": False,
                    "effective_output": 0.0,
                    "state_policy": "freeze_owned_scalar_state",
                    "previous_output_policy": "force_zero_before_connections",
                    "latch_policy": "retain_without_forcing_execution",
                },
                "fixed_shape_contract": {
                    "module_cap_change": 0,
                    "connection_cap_change": 0,
                    "state_width_per_module": 1,
                    "new_runtime_arrays": 0,
                    "observation_width_change": 0,
                    "action_width_change": 0,
                    "central_gru_width_change": 0,
                },
                "reporting_dimensions": [
                    "allocated_modules",
                    "executing_modules",
                    "dormant_modules",
                    "causally_active_modules",
                    "neutral_duplicate_modules",
                    "redundant_modules",
                    "prunable_modules",
                    "blocked_duplication_attempts",
                    "blocked_prune_attempts",
                    "cap_contact",
                    "sustained_cap_pressure",
                    "structural_cost",
                    "bloat_classification",
                ],
                "cost": {
                    "schema_version": COST_SCHEMA_VERSION,
                    "profile": COST_PROFILE,
                    "mode": COST_MODE,
                    "reporting_only": True,
                    "coefficients": {
                        name: float(value)
                        for name, value in sorted(COST_COEFFICIENTS.items())
                    },
                    "organism_energy_debit": False,
                    "fitness_or_selection_term": False,
                    "serialized_genome_cap": None,
                    "serialized_genome_occupancy": "canonical_bytes_length_reporting_only",
                    "binding_count_semantics": (
                        "enabled_binding_on_executing_module_including_zero_gain_"
                        "because_the_binding_is_evaluated"
                    ),
                },
                "bloat_classification_version": BLOAT_CLASSIFICATION_VERSION,
                "bloat_classifications": [
                    "healthy_headroom",
                    "growth_without_activity",
                    "active_growth",
                    "dormant_reserve",
                    "pruning_pressure",
                    "capacity_limited",
                    "possible_bloat",
                ],
                "classification_is_selection_input": False,
                "supported_lanes": ["python_reference_cpu"],
                "execution_rng_draws": 0,
            },
            "observation_routing": {
                "schema_version": "escap4_observation_routing_v1",
                "registry_version": PERCEPTION_ROUTING_CONTRACT_VERSION,
                "profile": "escap4_scalar_self_state_parallel_v1",
                "operator_profile": "escap4_routing_binding_operator_v1",
                "operator_attempts": int(
                    getattr(sim_cfg, "structural_observation_routing_operator_attempts", 1) or 0
                ) if sim_cfg is not None else 1,
                "enabled": bool(
                    getattr(sim_cfg, "structural_observation_routing_enable", False)
                ) if sim_cfg is not None else False,
                "exposure_mode": "parallel",
                "eligible_semantic_ids": list(routing_eligible_ids),
                "eligible_semantic_count": len(routing_eligible_ids),
                "available_semantic_count": len(routing_eligible_ids),
                "maximum_bindings_per_module": 2,
                "maximum_bindings_per_agent": 8,
                "allocated_bindings": "payload_dependent",
                "executable_bindings": "payload_and_gate_dependent",
                "active_bindings": "runtime_telemetry_only",
                "causal_use": "not_implied_by_allocation_or_activity",
                "execution_rng_draws": 0,
                "supported_lanes": ["python_reference_cpu"],
                "unsupported_lanes": ["python_reference_cuda", "native_owner", "cuda_specialization", "policy_stride"],
            },
            "output_routing": {
                "schema_version": OUTPUT_ROUTING_SCHEMA_VERSION,
                "registry_version": ACTION_SEMANTIC_REGISTRY_VERSION,
                "profile": OUTPUT_ROUTING_PROFILE,
                "operator_profile": OUTPUT_ROUTING_OPERATOR_PROFILE,
                "operator_attempts": int(
                    getattr(sim_cfg, "structural_output_routing_operator_attempts", 1) or 0
                ) if sim_cfg is not None else 1,
                "enabled": bool(
                    getattr(sim_cfg, "structural_output_routing_enable", False)
                ) if sim_cfg is not None else False,
                "admitted_action_semantic_ids": list(ADMITTED_ACTION_SEMANTIC_IDS),
                "registered_action_meanings": list(ACTION_VOCABULARY),
                "maximum_bindings_per_module": int(MAX_OUTPUT_BINDINGS_PER_MODULE),
                "maximum_bindings_per_agent": int(MAX_OUTPUT_BINDINGS_PER_AGENT),
                "maximum_absolute_per_action_residual": float(MAX_ACTION_RESIDUAL),
                "allocated_bindings": "payload_dependent",
                "executable_bindings": "payload_and_gate_dependent",
                "active_nonzero_bindings": "runtime_telemetry_only",
                "clipped_bindings": "runtime_telemetry_only",
                "saturated_actions": "runtime_telemetry_only",
                "rejected_bindings": "validation_only",
                "per_action_residual_totals": "runtime_telemetry_only",
                "residual_base_logit_ratio": "reporting_only",
                "technical_causal_effect": "probe_only_not_implied_by_activity",
                "ecological_usefulness": "not_asserted",
                "arbitration_version": OUTPUT_ROUTING_ARBITRATION_VERSION,
                "authority_order_version": OUTPUT_ROUTING_AUTHORITY_ORDER_VERSION,
                "combined_rest_compatibility_version": COMBINED_REST_COMPATIBILITY_VERSION,
                "execution_rng_draws": 0,
                "cost_semantics": "reporting_only",
                "supported_lanes": ["python_reference_cpu"],
                "unsupported_lanes": ["python_reference_cuda", "native_owner", "cuda_specialization", "policy_stride"],
            },
        },
        "development": {
            "schema_version": DEVELOPMENTAL_PROGRAM_SCHEMA_VERSION,
            "storage_limits": "unbounded_python_collections_in_current_schema",
            "stage_growth_operator_reachable": False,
            "dependency_growth_operator_reachable": False,
            "structural_development": {
                "enabled": bool(
                    getattr(sim_cfg, "structural_development_enable", False)
                ) if sim_cfg is not None else False,
                "profile": DEVELOPMENTAL_FEATURE_PROFILE,
                "module_record_schema_version": DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
                "stage_profile_version": DEVELOPMENTAL_STAGE_PROFILE_VERSION,
                "stage_semantic_ids": list(DEVELOPMENTAL_STAGE_IDS),
                "mature_start_step": int(MATURE_START_STEP),
                "late_start_step": int(LATE_START_STEP),
                "stage_gain_bounds": {
                    "minimum": float(MIN_STAGE_GAIN),
                    "maximum": float(MAX_STAGE_GAIN),
                },
                "developmental_records": {
                    "maximum_per_agent": int(active_b2_caps["max_modules_per_agent"]),
                    "allocated": "payload_dependent",
                    "expressed": "runtime_stage_and_dependency_dependent",
                },
                "dependency_edges": {
                    "identity_authority": DEPENDENCY_IDENTITY_AUTHORITY,
                    "maximum_per_module": int(MAX_PREREQUISITES_PER_MODULE),
                    "maximum_per_agent": int(MAX_DEPENDENCY_EDGES_PER_AGENT),
                    "allocated": "payload_dependent",
                    "ready": "runtime_telemetry_only",
                },
                "latches": {
                    "runtime_key": "module_instance_id",
                    "maximum_per_agent": int(active_b2_caps["max_modules_per_agent"]),
                    "committed": "runtime_telemetry_only",
                    "inheritance": "newborn_empty_not_hereditary",
                },
                "expression": {
                    "active_quantity": "runtime_telemetry_only",
                    "inactive_state_policy": INACTIVE_STATE_POLICY,
                    "inactive_output_policy": INACTIVE_OUTPUT_POLICY,
                },
                "execution_rng_draws": 0,
                "cost_semantics": "reporting_only",
                "scientific_usefulness": "not_asserted",
                "supported_lanes": [SUPPORTED_LANE],
                "unsupported_lanes": [
                    "python_reference_cuda",
                    "native_owner",
                    "runnercore_authority",
                    "policy_stride_not_1",
                ],
            },
        },
        "directed_inheritance": {
            "registry_schema_version": DIRECTED_INHERITANCE_REGISTRY_SCHEMA_VERSION,
            "declared_channel_count": len(inheritance),
            "implemented_active_channel_count": active_inheritance,
        },
        "runtime_identity": {
            "biological_id_model": "monotonic_dense_historical_id",
            "runtime_live_slot_model": "recyclable_slot_pool_where_supported",
            "live_slot_limit": (
                int(getattr(sim_cfg, "population_size", 0) or 0)
                if sim_cfg is not None and hasattr(sim_cfg, "population_size")
                else None
            ),
            "biological_id_limit": None,
        },
        "checkpoint": {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "capacity_epoch_metadata_supported": True,
            "automatic_expansion_supported": False,
        },
        "execution_lanes": {
            "active_lane": active_lane,
            "profiles": {
                "python_reference": {
                    "ownership": "active_repository_implementation",
                    "capacity_source": "runtime_registries_and_config",
                },
                "native_owner": {
                    "ownership": "repository_implementation_presence_not_runtime_ownership_unless_enabled",
                    "capacity_source": "runtime_owner_tensors_and_validated_projection_metadata",
                },
                "cuda_specialization": {
                    "ownership": "external_unmerged_performance_track",
                    "capacity_source": "not_imported_into_ESCAP1",
                    "status": "explicitly_unverified_on_this_branch",
                },
            },
        },
    }
    if not bool(
        getattr(sim_cfg, "structural_sustainable_growth_enable", False)
        if sim_cfg is not None
        else False
    ):
        # Default-off means exact ESCAP-7 contract material.  In particular,
        # capacity-epoch checkpoint identity must not acquire an ESCAP-8 block.
        contract["structure"].pop("sustainable_growth", None)
    contract["contract_hash"] = canonical_contract_hash(contract)
    return contract


def _headroom(used: int, available: int) -> dict[str, Any]:
    used_i = max(0, int(used))
    available_i = max(0, int(available))
    remaining = max(0, available_i - used_i)
    fraction = float(remaining / available_i) if available_i > 0 else 0.0
    return {
        "used": used_i,
        "available": available_i,
        "remaining": remaining,
        "headroom_fraction": fraction,
        "warning_threshold": CAPACITY_WARNING_HEADROOM_FRACTION,
        "warning_threshold_crossed": bool(fraction <= CAPACITY_WARNING_HEADROOM_FRACTION),
        "over_capacity": bool(used_i > available_i),
    }


def capacity_headroom_summary(
    records: Iterable[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    structural_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = [dict(row) for row in records]
    limits = dict(contract["structure"]["limits"])
    max_modules = max((int(row.get("module_count", 0) or 0) for row in rows), default=0)
    max_connections = max((int(row.get("connection_count", 0) or 0) for row in rows), default=0)
    max_memory = max((int(row.get("effective_memory", 0) or 0) for row in rows), default=0)
    max_channels = max((int(row.get("sensor_channel_count", 0) or 0) for row in rows), default=0)
    max_observation_bindings = max(
        (int(row.get("observation_binding_count", 0) or 0) for row in rows),
        default=0,
    )
    max_output_bindings = max(
        (int(row.get("output_binding_count", 0) or 0) for row in rows),
        default=0,
    )
    max_developmental_records = max(
        (
            int(
                row.get(
                    "developmental_record_count",
                    row.get("allocated_developmental_records", 0),
                )
                or 0
            )
            for row in rows
        ),
        default=0,
    )
    max_developmental_edges = max(
        (
            int(
                row.get(
                    "developmental_dependency_edge_count",
                    row.get("dependency_edges", 0),
                )
                or 0
            )
            for row in rows
        ),
        default=0,
    )
    max_expressed_developmental_modules = max(
        (
            int(
                row.get(
                    "developmentally_expressed_module_count",
                    row.get("expressed_modules", 0),
                )
                or 0
            )
            for row in rows
        ),
        default=0,
    )
    max_developmental_latches = max(
        (
            int(
                row.get(
                    "developmental_latch_count",
                    row.get("irreversible_commitments", 0),
                )
                or 0
            )
            for row in rows
        ),
        default=0,
    )

    def _maximum_row_count(*names: str) -> int:
        return max(
            (
                int(
                    next(
                        (
                            row.get(name, 0)
                            for name in names
                            if name in row
                        ),
                        0,
                    )
                    or 0
                )
                for row in rows
            ),
            default=0,
        )

    growth_enabled = bool(
        contract["structure"].get("sustainable_growth", {}).get("enabled", False)
    )
    max_executing_modules = 0
    max_dormant_modules = 0
    max_causally_active_modules = 0
    max_neutral_duplicates = 0
    max_redundant_modules = 0
    max_prunable_modules = 0
    if growth_enabled:
        max_executing_modules = _maximum_row_count(
            "executing_module_count", "executing_modules", "executable_modules"
        )
        max_dormant_modules = _maximum_row_count(
            "dormant_module_count", "dormant_modules"
        )
        # Causal activity is probe-supplied only.  A merely nonzero output is
        # not evidence of downstream module or action-logit influence.
        max_causally_active_modules = _maximum_row_count(
            "causally_active_module_count", "causally_active_modules"
        )
        max_neutral_duplicates = _maximum_row_count(
            "neutral_duplicate_count", "neutral_duplicate_modules"
        )
        max_redundant_modules = _maximum_row_count(
            "redundant_module_count", "redundant_modules"
        )
        max_prunable_modules = _maximum_row_count(
            "prunable_module_count", "prunable_modules"
        )
    def _row_structural_cost(row: Mapping[str, Any]) -> float:
        if "structural_cost_total" in row:
            return float(row.get("structural_cost_total", 0.0) or 0.0)
        nested = row.get("structural_cost", {})
        if isinstance(nested, Mapping):
            return float(nested.get("total", 0.0) or 0.0)
        return 0.0

    max_structural_cost = (
        max((_row_structural_cost(row) for row in rows), default=0.0)
        if growth_enabled
        else 0.0
    )
    ctx = dict(structural_context or {})
    if growth_enabled:
        max_executing_modules = max(
            max_executing_modules,
            int(ctx.get("executing_module_count_max", 0) or 0),
        )
        max_dormant_modules = max(
            max_dormant_modules,
            int(ctx.get("dormant_module_count_max", 0) or 0),
        )
        max_neutral_duplicates = max(
            max_neutral_duplicates,
            int(ctx.get("neutral_duplicate_count_max", 0) or 0),
        )
        max_redundant_modules = max(
            max_redundant_modules,
            int(ctx.get("redundant_module_count_max", 0) or 0),
        )
        max_prunable_modules = max(
            max_prunable_modules,
            int(ctx.get("prunable_module_count_max", 0) or 0),
        )
        max_structural_cost = max(
            max_structural_cost,
            float(ctx.get("structural_cost_total_max", 0.0) or 0.0),
        )
    blocked = {
        REASON_MODULE_CAPACITY: int(ctx.get("module_cap_drop_count", 0) or 0),
        REASON_CONNECTION_CAPACITY: int(ctx.get("connection_cap_drop_count", 0) or 0),
        REASON_LOCALITY_INVALID: int(ctx.get("locality_drop_count", 0) or 0),
        REASON_RUNTIME_ALLOCATION: int(ctx.get("allocation_failure_count", 0) or 0),
    }
    blocked = {reason: count for reason, count in sorted(blocked.items()) if count > 0}
    blocked_duplication_attempts = (
        int(
            ctx.get(
                "blocked_duplication_attempts",
                ctx.get(
                    "cap_blocked_duplication_count",
                    sum(
                        int(
                            row.get(
                                "escap8_blocked_duplication_attempt_count", 0
                            )
                            or 0
                        )
                        for row in rows
                    ),
                ),
            )
            or 0
        )
        if growth_enabled
        else 0
    )
    blocked_prune_attempts = (
        int(
            ctx.get(
                "blocked_prune_attempts",
                ctx.get(
                    "dependency_blocked_prune_count",
                    sum(
                        int(row.get("escap8_blocked_prune_attempt_count", 0) or 0)
                        for row in rows
                    ),
                ),
            )
            or 0
        )
        if growth_enabled
        else 0
    )
    reachability = dict(contract["structure"]["reachability"])
    lane = str(contract["execution_lanes"]["active_lane"])
    lane_mismatch = lane == "cuda_specialization" or (
        lane == "python_reference_cuda_device"
        and contract["execution_lanes"]["profiles"]["cuda_specialization"]["status"]
        != "verified"
    )
    representation_limited = bool(
        any(blocked.values())
        or str(reachability.get("classification", "")).startswith("not_")
        or lane_mismatch
    )
    module_cap_contact = bool(max_modules >= int(limits["max_modules_per_agent"]))
    connection_cap_contact = bool(
        max_connections >= int(limits["max_connections_per_agent"])
    )
    cap_pressure_streak = (
        max(
            [int(row.get("escap8_cap_pressure_streak", 0) or 0) for row in rows]
            + [int(ctx.get("cap_pressure_streak", 0) or 0)]
        )
        if growth_enabled
        else 0
    )
    causal_activity_measured = bool(
        ctx.get(
            "causal_activity_measured",
            any(
                "causally_active_module_count" in row
                or "causally_active_modules" in row
                for row in rows
            ),
        )
    )
    sustained_cap_pressure = bool(growth_enabled and (
        cap_pressure_streak >= 3
        and (blocked_duplication_attempts > 0 or any(blocked.values()))
    ))
    bloat = (
        classify_bloat(
            allocated_module_count=max_modules,
            module_capacity=int(limits["max_modules_per_agent"]),
            active_module_count=max_executing_modules,
            dormant_module_count=max_dormant_modules,
            causally_active_module_count=max_causally_active_modules,
            neutral_duplicate_count=max_neutral_duplicates,
            redundant_structure_count=max_redundant_modules,
            prunable_module_count=max_prunable_modules,
            blocked_duplication_attempt_count=blocked_duplication_attempts,
            blocked_prune_attempt_count=blocked_prune_attempts,
            cap_contact_count=max(
                int(module_cap_contact or connection_cap_contact),
                int(ctx.get("cap_contact_count", 0) or 0),
            ),
            sustained_cap_pressure=sustained_cap_pressure,
            structural_cost=float(max_structural_cost),
            useful_activity_proxy=float(
                ctx.get("useful_activity_proxy", max_causally_active_modules) or 0.0
            ),
            causal_activity_measured=causal_activity_measured,
        )
        if growth_enabled
        else {}
    )
    summary = {
        "schema_version": "eiel_capacity_headroom_summary_v1",
        "capacity_contract_hash": str(contract["contract_hash"]),
        "capacity_epoch_id": str(contract["capacity_epoch_id"]),
        "allocation": {
            "modules": _headroom(max_modules, int(limits["max_modules_per_agent"])),
            "connections": _headroom(max_connections, int(limits["max_connections_per_agent"])),
            "effective_memory": _headroom(max_memory, int(contract["policy"]["effective_memory_max"])),
            "perception_channels": _headroom(
                max_channels,
                int(contract["observations"]["implemented_active_channel_count"]),
            ),
            "observation_routing_bindings": _headroom(
                max_observation_bindings,
                int(contract["structure"]["observation_routing"]["maximum_bindings_per_agent"]),
            ),
            "output_routing_bindings": _headroom(
                max_output_bindings,
                int(contract["structure"]["output_routing"]["maximum_bindings_per_agent"]),
            ),
            "developmental_records": _headroom(
                max_developmental_records,
                int(
                    contract["development"]["structural_development"]
                    ["developmental_records"]["maximum_per_agent"]
                ),
            ),
            "developmental_dependency_edges": _headroom(
                max_developmental_edges,
                int(
                    contract["development"]["structural_development"]
                    ["dependency_edges"]["maximum_per_agent"]
                ),
            ),
        },
        "functional_use": {
            "structure": (
                "live_scalar_field_compiler_only"
                if not str(reachability.get("classification", "")).startswith("not_")
                else "not_bootstrap_reachable"
            ),
            "allocation_is_not_functional_use": True,
            "developmentally_expressed_modules": {
                "maximum_observed": max_expressed_developmental_modules,
                "reporting_only": True,
                "usefulness_not_asserted": True,
            },
            "developmental_latches": {
                "maximum_observed": max_developmental_latches,
                "reporting_only": True,
                "hereditary": False,
            },
        },
        "sustainable_growth": {
            "enabled": bool(
                contract["structure"].get("sustainable_growth", {}).get("enabled", False)
            ),
            "allocated_module_count_max": max_modules,
            "executing_module_count_max": max_executing_modules,
            "dormant_module_count_max": max_dormant_modules,
            "causally_active_module_count_max": max_causally_active_modules,
            "neutral_duplicate_count_max": max_neutral_duplicates,
            "redundant_module_count_max": max_redundant_modules,
            "prunable_module_count_max": max_prunable_modules,
            "blocked_duplication_attempts": blocked_duplication_attempts,
            "blocked_prune_attempts": blocked_prune_attempts,
            "module_cap_contact": module_cap_contact,
            "connection_cap_contact": connection_cap_contact,
            "cap_contact": bool(module_cap_contact or connection_cap_contact),
            "cap_pressure_streak": cap_pressure_streak,
            "sustained_cap_pressure": sustained_cap_pressure,
            "structural_cost_max": float(max_structural_cost),
            "cost_schema_version": COST_SCHEMA_VERSION,
            "cost_mode": COST_MODE,
            "bloat_classification_version": BLOAT_CLASSIFICATION_VERSION,
            "bloat_classification": bloat,
            "serialized_genome_cap": None,
            "serialized_genome_capacity_bounded": False,
            "reporting_only": True,
            "selection_input": False,
        },
        "blocked_mutation_count_by_reason": blocked,
        "over_cap_canonicalization_attempts": int(ctx.get("over_cap_canonicalization_attempts", 0) or 0),
        "allocation_growth_failures": int(ctx.get("allocation_failure_count", 0) or 0),
        "structural_reachability_classification": str(reachability.get("classification")),
        "active_lane_capability_mismatch": bool(lane_mismatch),
        "representation_limited": representation_limited,
    }
    if not growth_enabled:
        summary.pop("sustainable_growth", None)
    return summary
