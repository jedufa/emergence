"""Minimal production-control truth surface for WRC qualification.

This module exposes only existing configuration fields.  It does not add world
mechanics.  It provides a shared, fail-closed translation used by production
campaign factories and their control receipts.
"""

from __future__ import annotations

from dataclasses import fields
import math
from typing import Any, Final, Mapping, Sequence

import torch

from engine.env.env_config import EnvConfig
from engine.perf.cxxbackbone10_resident_state_owner import projection_digest
from simulation.runner_pkg.runner_config import SimConfig

from .canonical import canonical_sha256
from .contracts import ParameterContractV1, Scalar


# Existing Patch/resource controls that named ecology profiles previously owned
# as one inseparable bundle.
PATCH_ENV_PRIMITIVE_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {
        "start_energy",
        "max_energy",
        "move_cost",
        "rest_cost",
        "gather_cost",
        "food_energy",
        "patch_food_gather_amount",
        "patch_food_regrow_per_step",
        "patch_food_regrow_use_fertility",
        "patch_food_regrow_cap_by_fertility",
        "patch_movement_cost_apply_enable",
        "patch_movement_cost_default",
        "patch_movement_cost_energy_scale",
        "patch_world_waste_consequence_enable",
        "patch_waste_regrowth_penalty_enable",
        "patch_waste_regrowth_penalty_scale",
        "patch_waste_regrowth_penalty_max",
        "patch_density_regrowth_penalty_enable",
        "patch_density_regrowth_penalty_scale",
        "patch_density_regrowth_penalty_max",
        "patch_regrowth_min_multiplier",
        "reproduce_parent_energy_share",
        "reproduce_min_energy_buffer",
        "reproduce_cooldown_steps",
        "reproduce_try_prob",
    }
)
PATCH_SIM_PRIMITIVE_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {"resource_maintenance_energy_cost_per_tick"}
)

# Existing Jungle fields previously fixed by canonical-jungle-v4.
JUNGLE_ENV_PRIMITIVE_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {
        "food_spawn_per_step",
        "max_food",
        "reproduce_try_prob",
        "reproduce_parent_energy_share",
        "reproduce_min_energy_buffer",
        "reproduce_cooldown_steps",
        "jungle_predator_pressure_enable",
        "jungle_predator_base_risk",
        "jungle_predator_scent_weight",
        "jungle_predator_waste_weight",
        "jungle_predator_density_weight",
        "jungle_predator_visibility_weight",
        "jungle_predator_cover_protection_weight",
        "jungle_predator_shelter_protection_weight",
        "jungle_predator_risk_min",
        "jungle_predator_risk_cap",
        "jungle_predator_health_damage",
        "jungle_juvenile_predator_vulnerability_enable",
        "jungle_juvenile_age_threshold",
        "jungle_juvenile_predator_damage_multiplier",
        "jungle_juvenile_vulnerability_cap",
        "jungle_terrain_protection_enable",
        "jungle_terrain_protection_apply_to_damage_enable",
        "jungle_cover_damage_reduction_weight",
        "jungle_shelter_damage_reduction_weight",
        "jungle_terrain_damage_multiplier_min",
        "jungle_reproduction_emissions_enable",
        "jungle_birth_emission_at_child_cell_enable",
        "jungle_birth_emission_clamp_enable",
        "jungle_birth_scent_emit",
        "jungle_birth_waste_emit",
        "jungle_ecological_recovery_enable",
        "jungle_recovery_activity_decay_per_step",
        "jungle_recovery_activity_base_growth",
        "jungle_recovery_activity_cover_weight",
        "jungle_recovery_activity_fertility_weight",
        "jungle_recovery_activity_waste_penalty_weight",
        "jungle_recovery_activity_density_penalty_weight",
        "jungle_recovery_activity_scent_penalty_weight",
        "jungle_recovery_regrowth_boost_enable",
        "jungle_recovery_regrowth_boost_scale",
        "jungle_recovery_regrowth_boost_max",
    }
)

# Existing EED magnitudes previously compiled as constants around four selectors.
EED_ENV_PRIMITIVE_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {
        "food_spawn_per_step",
        "max_food",
        "patch_scent_gather_emit",
        "patch_waste_consume_emit",
        "eed_resource_a_initial_amount",
        "eed_resource_b_initial_amount",
        "eed_resource_a_capacity",
        "eed_resource_b_capacity",
        "eed_resource_a_renewal_per_step",
        "eed_resource_b_renewal_per_step",
        "eed_resource_impact_per_success",
        "eed_resource_impact_decay_per_step",
        "eed_resource_impact_max_load",
        "reproduce_try_prob",
    }
)

# Existing WDL internals that were fixed/derived by the route translator.
WDL_ENV_PRIMITIVE_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {
        "wdl_delayed_hazard_event_period_steps",
        "wdl_delayed_hazard_radius",
        "wdl_delayed_hazard_observation_dropout_rate",
    }
)

ALL_ENV_PRIMITIVE_FIELDS_V1: Final[frozenset[str]] = frozenset(
    PATCH_ENV_PRIMITIVE_FIELDS_V1
    | JUNGLE_ENV_PRIMITIVE_FIELDS_V1
    | EED_ENV_PRIMITIVE_FIELDS_V1
    | WDL_ENV_PRIMITIVE_FIELDS_V1
)
ALL_SIM_PRIMITIVE_FIELDS_V1: Final[frozenset[str]] = PATCH_SIM_PRIMITIVE_FIELDS_V1

PATCH_PRIMITIVE_PARAMETER_NAMES_V1: Final[frozenset[str]] = frozenset(
    {f"env_cfg.{name}" for name in PATCH_ENV_PRIMITIVE_FIELDS_V1}
    | {f"sim_cfg.{name}" for name in PATCH_SIM_PRIMITIVE_FIELDS_V1}
)

JUNGLE_PRIMITIVE_PARAMETER_NAMES_V1: Final[frozenset[str]] = frozenset(
    f"env_cfg.{name}" for name in JUNGLE_ENV_PRIMITIVE_FIELDS_V1
)
EED_PRIMITIVE_PARAMETER_NAMES_V1: Final[frozenset[str]] = frozenset(
    f"env_cfg.{name}" for name in EED_ENV_PRIMITIVE_FIELDS_V1
)
WDL_PRIMITIVE_PARAMETER_NAMES_V1: Final[frozenset[str]] = frozenset(
    f"env_cfg.{name}" for name in WDL_ENV_PRIMITIVE_FIELDS_V1
)
ALL_PRIMITIVE_PARAMETER_NAMES_V1: Final[frozenset[str]] = frozenset(
    PATCH_PRIMITIVE_PARAMETER_NAMES_V1
    | JUNGLE_PRIMITIVE_PARAMETER_NAMES_V1
    | EED_PRIMITIVE_PARAMETER_NAMES_V1
    | WDL_PRIMITIVE_PARAMETER_NAMES_V1
)

# The accepted WRC forensics identified these 30 public production controls as
# the admitted behavior-control surface requiring end-to-end applied readback.
AUDITED_PUBLIC_CONTROL_NAMES_V1: Final[frozenset[str]] = frozenset(
    {
        "patch.grid_size",
        "patch.initial_population",
        "patch.capacity",
        "patch.founder_placement",
        "patch.ecology_profile",
        "patch.initial_spatial_placement_profile",
        "patch.initial_live_count",
        "policy.hidden_width",
        "policy.reset_hidden_each_tick",
        "wdl.delay_ticks",
        "wdl.cue_duration_ticks",
        "wdl.active_duration_ticks",
        "wdl.intensity",
        "wdl.cue_visible_at_decision",
        "eed.founder_resource_axis_profile",
        "eed.resource_impact_apply_enable",
        "eed.ecology_frozen",
        "eed.phenotype_fixed",
        "genome_upgrade_v2.profile",
        "genome_upgrade_v2.enable",
        "genome_upgrade_v2.mutation_rate_q10",
        "genome_upgrade_v2.mutation_step_q10",
        "genome_upgrade_v2.founder_profile",
        "production_lifecycle.profile",
        "production_lifecycle.lifecycle_mode",
        "production_lifecycle.ordinary_births_enabled",
        "production_lifecycle.turnover_enabled",
        "production_lifecycle.turnover_cull_fraction",
        "production_lifecycle.turnover_parent_fraction",
        "production_lifecycle.life_history_profile",
    }
)
PRODUCTION_CONTROL_READBACK_PARAMETER_NAMES_V1: Final[frozenset[str]] = (
    AUDITED_PUBLIC_CONTROL_NAMES_V1 | ALL_PRIMITIVE_PARAMETER_NAMES_V1
)
PRODUCTION_HETEROGENEOUS_PARAMETER_NAMES_V1: Final[frozenset[str]] = frozenset(
    {
        "env_cfg.jungle_recovery_activity_base_growth",
        "env_cfg.jungle_recovery_activity_decay_per_step",
        "env_cfg.jungle_recovery_activity_fertility_weight",
        "env_cfg.jungle_recovery_activity_waste_penalty_weight",
        "env_cfg.move_cost",
        "env_cfg.patch_movement_cost_energy_scale",
        "env_cfg.patch_food_gather_amount",
        "env_cfg.patch_food_regrow_per_step",
        "env_cfg.jungle_predator_base_risk",
        "env_cfg.jungle_predator_health_damage",
        "env_cfg.jungle_predator_risk_cap",
        "env_cfg.reproduce_cooldown_steps",
        "env_cfg.reproduce_min_energy_buffer",
        "env_cfg.reproduce_parent_energy_share",
        "env_cfg.reproduce_try_prob",
        "env_cfg.wdl_delayed_hazard_radius",
    }
)

_PROBABILITY_OR_FRACTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "reproduce_parent_energy_share",
        "reproduce_try_prob",
        "patch_waste_regrowth_penalty_max",
        "patch_density_regrowth_penalty_max",
        "patch_regrowth_min_multiplier",
        "jungle_predator_base_risk",
        "jungle_predator_risk_min",
        "jungle_predator_risk_cap",
        "jungle_terrain_damage_multiplier_min",
        "jungle_recovery_activity_decay_per_step",
        "wdl_delayed_hazard_observation_dropout_rate",
    }
)
_POSITIVE_INTEGER_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "wdl_delayed_hazard_event_period_steps",
        "reproduce_cooldown_steps",
        "jungle_juvenile_age_threshold",
    }
)
_NONNEGATIVE_INTEGER_FIELDS: Final[frozenset[str]] = frozenset(
    {"max_food", "wdl_delayed_hazard_radius"}
)


def _field_type_map(cls: type[Any]) -> dict[str, type[Any]]:
    # The currently exposed primitives are scalar dataclass fields.  Their
    # instantiated defaults are authoritative for exact Python scalar type.
    instance = cls()
    return {field.name: type(getattr(instance, field.name)) for field in fields(instance)}


_ENV_TYPES: Final[dict[str, type[Any]]] = _field_type_map(EnvConfig)
_SIM_TYPES: Final[dict[str, type[Any]]] = _field_type_map(SimConfig)


def _predicate_for(field_name: str, value_type: type[Any]):
    if value_type is bool:
        return lambda value: type(value) is bool
    if value_type is int:
        if field_name in _POSITIVE_INTEGER_FIELDS:
            return lambda value: type(value) is int and value > 0
        if field_name in _NONNEGATIVE_INTEGER_FIELDS:
            return lambda value: type(value) is int and value >= 0
        return lambda value: type(value) is int
    if value_type is float:
        if field_name in _PROBABILITY_OR_FRACTION_FIELDS:
            return lambda value: type(value) is float and math.isfinite(value) and 0.0 <= value <= 1.0
        return lambda value: type(value) is float and math.isfinite(value) and value >= 0.0
    raise TypeError(f"unsupported WRC primitive type for {field_name}: {value_type!r}")


def _value_contract_for(field_name: str, value_type: type[Any]) -> dict[str, Any]:
    """The declarative counterpart to ``_predicate_for`` for hashing."""

    if value_type is bool:
        return {"kind": "bool"}
    if value_type is int:
        if field_name in _POSITIVE_INTEGER_FIELDS:
            return {"kind": "int", "minimum": 1, "minimum_inclusive": True}
        if field_name in _NONNEGATIVE_INTEGER_FIELDS:
            return {"kind": "int", "minimum": 0, "minimum_inclusive": True}
        return {"kind": "int"}
    if value_type is float:
        payload: dict[str, Any] = {
            "kind": "float",
            "finite_only": True,
            "minimum": 0.0,
            "minimum_inclusive": True,
        }
        if field_name in _PROBABILITY_OR_FRACTION_FIELDS:
            payload["maximum"] = 1.0
            payload["maximum_inclusive"] = True
        return payload
    raise TypeError(f"unsupported WRC primitive type for {field_name}: {value_type!r}")


def _primitive_route_scopes(prefix: str, field_name: str) -> tuple[str, ...]:
    if prefix == "sim_cfg":
        return ("patch",)
    scopes: list[str] = []
    if field_name in PATCH_ENV_PRIMITIVE_FIELDS_V1:
        scopes.append("patch")
    if field_name in JUNGLE_ENV_PRIMITIVE_FIELDS_V1:
        scopes.append("jungle")
    if field_name in EED_ENV_PRIMITIVE_FIELDS_V1:
        scopes.append("eed")
    if field_name in WDL_ENV_PRIMITIVE_FIELDS_V1:
        scopes.append("wdl")
    return tuple(scopes)


def _primitive_activation_contract(
    prefix: str, field_name: str
) -> Mapping[str, Any]:
    """Return activation identity, not an activation-success assertion."""

    if prefix == "env_cfg" and field_name == "patch_movement_cost_energy_scale":
        return {
            "contract_id": "patch-movement-cost-energy-scale-v1",
            "required_effective_controls": {
                "env_cfg.patch_world_enable": True,
                "env_cfg.patch_movement_cost_apply_enable": True,
            },
            "runtime_assertion_any_of": [
                {
                    "metric": "occupied_patch_movement_cost_mean",
                    "operator": ">",
                    "value": 1.0,
                },
                {
                    "metric": "patch_movement_energy_extra_total",
                    "operator": ">",
                    "value": 0.0,
                },
            ],
        }
    return {}


def primitive_parameter_contracts_v1() -> dict[str, ParameterContractV1]:
    contracts: dict[str, ParameterContractV1] = {}
    for prefix, fields_set, type_map in (
        ("env_cfg", ALL_ENV_PRIMITIVE_FIELDS_V1, _ENV_TYPES),
        ("sim_cfg", ALL_SIM_PRIMITIVE_FIELDS_V1, _SIM_TYPES),
    ):
        for field_name in sorted(fields_set):
            value_type = type_map[field_name]
            parameter_name = f"{prefix}.{field_name}"
            contracts[parameter_name] = ParameterContractV1(
                parameter_name,
                value_type,
                _predicate_for(field_name, value_type),
                f"Existing production primitive {prefix}.{field_name}",
                True,
                False,
                control_role="independent_primitive",
                wrc_axis=True,
                variability_semantics="experiment_variable",
                variability_route_scopes=_primitive_route_scopes(prefix, field_name),
                value_contract=_value_contract_for(field_name, value_type),
                activation_contract=_primitive_activation_contract(prefix, field_name),
            )
    return contracts


def _normalize_scalar(value: Any) -> Scalar:
    if type(value) not in {bool, int, float, str}:
        raise TypeError(f"production control value is not scalar: {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("production control value must be finite")
    return value


def apply_primitive_overrides_v1(
    env_cfg: EnvConfig,
    sim_cfg: SimConfig,
    parameters: Mapping[str, Scalar],
    *,
    allowed_parameter_names: frozenset[str],
) -> dict[str, Any]:
    """Apply explicit existing primitives after route profile defaults.

    Unknown ``env_cfg.*``/``sim_cfg.*`` values fail closed at this boundary.
    """

    contracts = primitive_parameter_contracts_v1()
    requested_names = {
        name
        for name in parameters
        if name.startswith("env_cfg.") or name.startswith("sim_cfg.")
    }
    unsupported = requested_names - allowed_parameter_names
    if unsupported:
        raise ValueError(
            "primitive controls are unsupported by this production route: "
            f"{sorted(unsupported)!r}"
        )
    applied: dict[str, Scalar] = {}
    for parameter_name in sorted(requested_names):
        contract = contracts.get(parameter_name)
        if contract is None or not contract.accepts(parameters[parameter_name]):
            raise ValueError(
                f"invalid production primitive {parameter_name}={parameters[parameter_name]!r}"
            )
        section, field_name = parameter_name.split(".", 1)
        target = env_cfg if section == "env_cfg" else sim_cfg
        value = _normalize_scalar(parameters[parameter_name])
        setattr(target, field_name, value)
        applied[parameter_name] = _normalize_scalar(getattr(target, field_name))
    validate_primitive_activation_preflight_v1(
        env_cfg,
        sim_cfg,
        parameter_names=requested_names,
    )
    return {
        "schema_version": "eiel_production_primitive_override_receipt_v1",
        "requested": {name: _normalize_scalar(parameters[name]) for name in sorted(requested_names)},
        "applied": applied,
    }


def validate_primitive_activation_preflight_v1(
    env_cfg: EnvConfig,
    sim_cfg: SimConfig,
    *,
    parameter_names: set[str] | frozenset[str],
) -> None:
    """Fail closed when a requested dependent primitive cannot be active.

    This is deliberately called after route defaults and requested overrides
    have been applied.  It is a campaign preflight, not evidence that the
    runtime assertion has occurred and not a qualification result.
    """

    contracts = primitive_parameter_contracts_v1()
    for parameter_name in sorted(parameter_names):
        contract = contracts.get(parameter_name)
        if contract is None or not contract.activation_contract:
            continue
        required = contract.activation_contract.get("required_effective_controls")
        if not isinstance(required, Mapping):
            raise ValueError(
                f"activation contract for {parameter_name!r} is malformed"
            )
        for control_name, expected in sorted(required.items()):
            if not isinstance(control_name, str) or "." not in control_name:
                raise ValueError(
                    f"activation contract for {parameter_name!r} is malformed"
                )
            section, field_name = control_name.split(".", 1)
            target = env_cfg if section == "env_cfg" else sim_cfg if section == "sim_cfg" else None
            if target is None or not hasattr(target, field_name):
                raise ValueError(
                    f"activation contract for {parameter_name!r} references unknown control {control_name!r}"
                )
            actual = _normalize_scalar(getattr(target, field_name))
            if actual != expected:
                raise ValueError(
                    "WRC activation preflight unsatisfied: "
                    f"{parameter_name} requires {control_name}={expected!r}, "
                    f"got {actual!r}"
                )


def read_runner_primitive_values_v1(
    runner: Any, parameter_names: Mapping[str, Any] | set[str] | frozenset[str]
) -> dict[str, Scalar]:
    names = set(parameter_names)
    values: dict[str, Scalar] = {}
    for parameter_name in sorted(names & ALL_PRIMITIVE_PARAMETER_NAMES_V1):
        section, field_name = parameter_name.split(".", 1)
        target = runner.env_cfg if section == "env_cfg" else runner.sim_cfg
        values[parameter_name] = _normalize_scalar(getattr(target, field_name))
    return values


__all__ = [
    "ALL_PRIMITIVE_PARAMETER_NAMES_V1",
    "AUDITED_PUBLIC_CONTROL_NAMES_V1",
    "PRODUCTION_CONTROL_READBACK_PARAMETER_NAMES_V1",
    "PRODUCTION_HETEROGENEOUS_PARAMETER_NAMES_V1",
    "EED_PRIMITIVE_PARAMETER_NAMES_V1",
    "JUNGLE_PRIMITIVE_PARAMETER_NAMES_V1",
    "PATCH_PRIMITIVE_PARAMETER_NAMES_V1",
    "WDL_PRIMITIVE_PARAMETER_NAMES_V1",
    "apply_primitive_overrides_v1",
    "validate_primitive_activation_preflight_v1",
    "primitive_parameter_contracts_v1",
    "read_runner_primitive_values_v1",
    "build_segment_control_readback_v1",
    "PRODUCTION_CONTROL_READBACK_SCHEMA_V1",
]

PRODUCTION_CONTROL_READBACK_SCHEMA_V1: Final[str] = (
    "eiel_production_control_readback_v1"
)

_PUBLIC_TO_RUNTIME_FIELD_V1: Final[dict[str, str]] = {
    "patch.initial_spatial_placement_profile": "initial_spatial_placement_profile",
    "wdl.delay_ticks": "wdl_delayed_hazard_delay_steps",
    "wdl.cue_duration_ticks": "wdl_delayed_hazard_cue_duration_steps",
    "wdl.active_duration_ticks": "wdl_delayed_hazard_active_duration_steps",
    "wdl.intensity": "wdl_delayed_hazard_intensity",
    "wdl.cue_visible_at_decision": "wdl_delayed_hazard_cue_enable",
    "eed.founder_resource_axis_profile": "eed_phenotype_fixed_axis_profile",
    "eed.resource_impact_apply_enable": "eed_resource_impact_apply_enable",
    "eed.ecology_frozen": "eed_ecology_frozen_enable",
    "eed.phenotype_fixed": "eed_phenotype_fixed_enable",
    "genome_upgrade_v2.enable": "genome_upgrade_v2_enable",
    "genome_upgrade_v2.mutation_rate_q10": "genome_upgrade_v2_mutation_rate_q10",
    "genome_upgrade_v2.mutation_step_q10": "genome_upgrade_v2_mutation_step_q10",
    "genome_upgrade_v2.founder_profile": "genome_upgrade_v2_founder_profile_id",
    "production_lifecycle.ordinary_births_enabled": "ordinary_births_enabled",
    "production_lifecycle.turnover_enabled": "generation_turnover_enabled",
    "production_lifecycle.turnover_cull_fraction": "cull_fraction",
    "production_lifecycle.turnover_parent_fraction": "parent_fraction",
}

def _projection_metadata_v1(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        metadata = value.get("metadata")
    else:
        metadata = getattr(value, "metadata", None)
        if not isinstance(metadata, Mapping):
            projection = getattr(value, "projection", None)
            metadata = (
                projection.get("metadata")
                if isinstance(projection, Mapping)
                else None
            )
    return metadata if isinstance(metadata, Mapping) else {}


def _base_prepared_owner_v1(prepared: Any) -> Any:
    owner = prepared
    seen: set[int] = set()
    while id(owner) not in seen:
        seen.add(id(owner))
        if hasattr(owner, "source"):
            return owner
        physical = getattr(owner, "physical_window_arguments", None)
        if physical is not None:
            owner = physical
            continue
        patch = getattr(owner, "patch", None)
        if patch is None:
            break
        owner = patch
    return owner


def _nested_get_v1(value: Mapping[str, Any], *path: str) -> Any:
    current: Any = value
    for name in path:
        if not isinstance(current, Mapping) or name not in current:
            return None
        current = current[name]
    return current


def _compiled_control_value_v1(
    name: str,
    *,
    requested: Scalar,
    runner: Any,
    receipt: Mapping[str, Any],
) -> Any:
    if name.startswith("env_cfg."):
        return _normalize_scalar(getattr(runner.env_cfg, name.split(".", 1)[1]))
    if name.startswith("sim_cfg."):
        return _normalize_scalar(getattr(runner.sim_cfg, name.split(".", 1)[1]))
    if name == "patch.grid_size":
        width = int(runner.env_cfg.width)
        height = int(runner.env_cfg.height)
        if width != height:
            raise ValueError("patch.grid_size requires a square compiled world")
        return width
    if name in {"patch.initial_population", "patch.initial_live_count"}:
        return int(runner.pop_cfg.population_size)
    if name == "patch.capacity":
        return int(runner.pop_cfg.max_slots_cap)
    if name == "patch.founder_placement":
        return _nested_get_v1(receipt, "founder_initialization", "requested_label")
    if name == "patch.ecology_profile":
        return _nested_get_v1(receipt, "patch_ecology_profile", "profile_id")
    if name == "patch.initial_spatial_placement_profile":
        return _nested_get_v1(receipt, "initial_spatial_placement", "profile_id")
    if name == "policy.hidden_width":
        return int(getattr(runner.sim_cfg, "native_runtime_fixed_policy_hidden_width", 64))
    if name == "policy.reset_hidden_each_tick":
        return bool(getattr(runner.sim_cfg, "reset_hidden_each_tick", False))
    lifecycle_mode_abi = getattr(runner.sim_cfg, "lifecycle_mode", None)
    if type(lifecycle_mode_abi) is not int or lifecycle_mode_abi not in {0, 1}:
        lifecycle_mode_abi = None
    if name == "production_lifecycle.lifecycle_mode" and lifecycle_mode_abi is not None:
        return _public_lifecycle_mode_v1(lifecycle_mode_abi)
    if name == "production_lifecycle.turnover_enabled" and lifecycle_mode_abi is not None:
        return True
    if (
        name == "production_lifecycle.turnover_cull_fraction"
        and lifecycle_mode_abi is not None
    ):
        return 0.0 if lifecycle_mode_abi == 1 else 0.5
    if name == "production_lifecycle.life_history_profile":
        value = getattr(runner.sim_cfg, "life_history_profile_id", None)
        if type(value) is str:
            return value
    if name in _PUBLIC_TO_RUNTIME_FIELD_V1:
        field = _PUBLIC_TO_RUNTIME_FIELD_V1[name]
        target = (
            runner.sim_cfg
            if name.startswith(("genome_upgrade", "production_lifecycle"))
            else runner.env_cfg
        )
        if hasattr(target, field):
            value = getattr(target, field)
            return _normalize_scalar(value)
    if name == "genome_upgrade_v2.profile":
        return _nested_get_v1(receipt, "genome", "profile")
    if name == "production_lifecycle.profile":
        return _nested_get_v1(receipt, "production_lifecycle", "profile")
    if name == "production_lifecycle.life_history_profile":
        return _nested_get_v1(receipt, "production_lifecycle", "life_history", "profile_id")
    raise ValueError(f"unsupported compiled production control: {name}")


def _runtime_field_name_v1(name: str) -> str | None:
    if name.startswith(("env_cfg.", "sim_cfg.")):
        return name.split(".", 1)[1]
    return _PUBLIC_TO_RUNTIME_FIELD_V1.get(name)


def runtime_field_name_v1(name: str) -> str | None:
    """Return the native metadata/configuration field for a public control."""

    return _runtime_field_name_v1(name)


def _native_source_configuration_v1(prepared: Any) -> Any:
    source = getattr(_base_prepared_owner_v1(prepared), "source", None)
    return getattr(source, "configuration", None)


def _lifecycle_mode_from_native_source_v1(
    prepared: Any, metadata: Mapping[str, Any]
) -> int | None:
    configuration = _native_source_configuration_v1(prepared)
    raw = getattr(configuration, "lifecycle_mode", None)
    if type(raw) is not int:
        raw = metadata.get("lifecycle_mode_abi")
    return raw if type(raw) is int and raw in {0, 1} else None


def _public_lifecycle_mode_v1(raw: int) -> str:
    return (
        "legacy_synchronized_truncation_v1"
        if raw == 0
        else "hybrid_overlapping_v1"
    )


def _find_scalar_in_mapping_v1(value: Mapping[str, Any], field: str) -> Any:
    if field in value and type(value[field]) in {bool, int, float, str}:
        return _normalize_scalar(value[field])
    for child in value.values():
        if isinstance(child, Mapping):
            found = _find_scalar_in_mapping_v1(child, field)
            if found is not None:
                return found
    return None


def _post_dispatch_abi_values_v1(
    receipt: Mapping[str, Any],
    *,
    field: str,
    cell_ordinal: int,
    include_scalar: bool,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    rows = receipt.get("outer_arguments")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return values
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        argument_name = str(row.get("name", ""))
        if row.get("kind") == "scalar" and argument_name == field:
            raw = row.get("value")
            if type(raw) in {bool, int, float, str}:
                values[f"post_dispatch_abi.{argument_name}"] = _normalize_scalar(raw)
        scalar_values = row.get("scalar_values")
        if include_scalar and isinstance(scalar_values, Mapping):
            found = _find_scalar_in_mapping_v1(scalar_values, field)
            if found is not None:
                values[f"post_dispatch_abi.{argument_name}"] = found
        scalar_vectors = row.get("scalar_vectors")
        if isinstance(scalar_vectors, Mapping):
            vector = scalar_vectors.get(f"{field}_by_cell")
            raw_values = vector.get("values") if isinstance(vector, Mapping) else None
            if (
                isinstance(raw_values, list)
                and 0 <= cell_ordinal < len(raw_values)
                and type(raw_values[cell_ordinal]) in {int, float}
            ):
                values[f"post_dispatch_abi.{argument_name}_by_cell"] = (
                    _normalize_scalar(raw_values[cell_ordinal])
                )
    return values


def _runtime_candidates_v1(
    name: str,
    *,
    request: Any,
    prepared: Any,
    source_projection: Any,
    post_dispatch_abi_receipt: Mapping[str, Any],
    include_shared_batch_abi: bool,
    cell_ordinal: int,
) -> dict[str, Any]:
    metadata = _projection_metadata_v1(source_projection)
    checkpoint_inputs = getattr(
        _base_prepared_owner_v1(prepared),
        "checkpoint_projection_inputs",
        {},
    )
    pre_generation = (
        checkpoint_inputs.get("pre_generation_canonical_projection")
        if isinstance(checkpoint_inputs, Mapping)
        else None
    )
    pre_metadata = _projection_metadata_v1(pre_generation)
    field = _runtime_field_name_v1(name)
    values: dict[str, Any] = {}
    native_configuration = _native_source_configuration_v1(prepared)
    lifecycle_mode_abi = _lifecycle_mode_from_native_source_v1(prepared, metadata)
    # Hybrid-overlapping is a public production lifecycle with no legacy
    # generation cull/refill pass.  Translate the native lifecycle ABI instead
    # of comparing its lower-level generation_turnover_enabled flag directly.
    if name == "production_lifecycle.lifecycle_mode" and lifecycle_mode_abi is not None:
        values["native_lifecycle_mode_abi"] = _public_lifecycle_mode_v1(
            lifecycle_mode_abi
        )
        return values
    if name == "production_lifecycle.turnover_enabled" and lifecycle_mode_abi is not None:
        values["native_lifecycle_mode_abi"] = True
        return values
    if (
        name == "production_lifecycle.turnover_cull_fraction"
        and lifecycle_mode_abi is not None
    ):
        values["native_lifecycle_mode_abi"] = (
            0.0 if lifecycle_mode_abi == 1 else 0.5
        )
        return values
    if field is not None:
        if include_shared_batch_abi or name == "env_cfg.patch_movement_cost_energy_scale":
            values.update(
                _post_dispatch_abi_values_v1(
                    post_dispatch_abi_receipt,
                    field=field,
                    cell_ordinal=cell_ordinal,
                    include_scalar=include_shared_batch_abi,
                )
            )
        for source_name, source in (("source_import_metadata", metadata),):
            if field in source and type(source[field]) in {bool, int, float, str}:
                values[source_name] = _normalize_scalar(source[field])
        if (
            field not in metadata
            and field in pre_metadata
            and type(pre_metadata[field]) in {bool, int, float, str}
        ):
            values["prepared_checkpoint_metadata"] = _normalize_scalar(
                pre_metadata[field]
            )
        config = getattr(request, "configuration", None)
        if config is not None and hasattr(config, field):
            raw = getattr(config, field)
            if type(raw) in {bool, int, float, str}:
                values["native_request_configuration"] = _normalize_scalar(raw)
        # ``source.configuration`` is the primary-cell scalar retained by the
        # shared owner.  It is meaningful only when the control is coherent
        # across the physical batch; using it for a heterogeneous cell makes
        # correct by-cell tensors look like a false mismatch.
        if (
            include_shared_batch_abi
            and native_configuration is not None
            and hasattr(native_configuration, field)
        ):
            raw = getattr(native_configuration, field)
            if type(raw) in {bool, int, float, str}:
                values["native_source_configuration"] = _normalize_scalar(raw)
        source = getattr(_base_prepared_owner_v1(prepared), "source", None)
        if field == "move_cost":
            raw_vector = getattr(source, "move_cost_by_cell", None)
            if (
                torch.is_tensor(raw_vector)
                and raw_vector.ndim == 1
                and 0 <= cell_ordinal < raw_vector.numel()
            ):
                value = raw_vector[cell_ordinal].item()
                if type(value) in {int, float}:
                    values["resident_move_cost_by_cell"] = _normalize_scalar(value)
        patch_config = getattr(source, "patch_p2_config", None)
        if isinstance(patch_config, Mapping):
            raw_vector = patch_config.get(f"{field}_by_cell")
            if (
                torch.is_tensor(raw_vector)
                and raw_vector.ndim == 1
                and 0 <= cell_ordinal < raw_vector.numel()
            ):
                value = raw_vector[cell_ordinal].item()
                if type(value) in {int, float}:
                    values["retained_patch_abi_config_by_cell"] = _normalize_scalar(value)
            elif include_shared_batch_abi and field in patch_config:
                raw = patch_config[field]
                if type(raw) in {bool, int, float, str}:
                    values["retained_patch_abi_config"] = _normalize_scalar(raw)
        jungle_field = {
            "jungle_predator_base_risk": "predator_base_risk",
            "jungle_predator_health_damage": "predator_health_damage",
            "jungle_predator_risk_cap": "predator_risk_cap",
            "jungle_recovery_activity_base_growth": "recovery_base_growth",
            "jungle_recovery_activity_decay_per_step": "recovery_decay_factor",
            "jungle_recovery_activity_fertility_weight": "recovery_fertility_weight",
            "jungle_recovery_activity_waste_penalty_weight": (
                "recovery_waste_penalty_weight"
            ),
        }.get(field, field)
        specialized_candidates = (
            getattr(prepared, "config", None),
            getattr(prepared, "jungle_config", None),
            getattr(getattr(prepared, "wdl", None), "config", None),
        )
        for specialized in specialized_candidates:
            if not isinstance(specialized, Mapping):
                continue
            raw_vector = specialized.get(f"{jungle_field}_by_cell")
            if (
                torch.is_tensor(raw_vector)
                and raw_vector.ndim == 1
                and 0 <= cell_ordinal < raw_vector.numel()
            ):
                value = raw_vector[cell_ordinal].item()
                if type(value) in {int, float}:
                    if field == "jungle_recovery_activity_decay_per_step":
                        value = 1.0 - float(value)
                    values["specialized_runtime_config_by_cell"] = _normalize_scalar(
                        value
                    )
                    break
            elif jungle_field in specialized:
                raw = specialized[jungle_field]
                if type(raw) in {bool, int, float, str}:
                    values["specialized_runtime_config"] = _normalize_scalar(raw)
        if field == "wdl_delayed_hazard_radius":
            for specialized in specialized_candidates:
                if not isinstance(specialized, Mapping):
                    continue
                raw_vector = specialized.get("radius_by_cell")
                if (
                    torch.is_tensor(raw_vector)
                    and raw_vector.ndim == 1
                    and 0 <= cell_ordinal < raw_vector.numel()
                ):
                    value = raw_vector[cell_ordinal].item()
                    if type(value) in {int, float}:
                        values["resident_wdl_radius_by_cell"] = _normalize_scalar(
                            value
                        )
                        break
    envelope = getattr(request, "envelope", None)
    if name == "patch.grid_size" and envelope is not None:
        grid = getattr(envelope, "grid", None)
        if isinstance(grid, tuple) and len(grid) == 2 and grid[0] == grid[1]:
            values["native_route_envelope"] = int(grid[0])
    if name == "patch.grid_size":
        source = getattr(_base_prepared_owner_v1(prepared), "source", None)
        patch_state = getattr(source, "patch_p2_state_tensors", ())
        first_patch_state = (
            patch_state[0]
            if isinstance(patch_state, Sequence) and patch_state
            else None
        )
        if (
            torch.is_tensor(first_patch_state)
            and first_patch_state.ndim >= 3
            and int(first_patch_state.shape[1]) == int(first_patch_state.shape[2])
        ):
            values["resident_patch_state_tensor"] = int(
                first_patch_state.shape[1]
            )
    elif name in {"patch.initial_population", "patch.initial_live_count"}:
        if "population_size" in metadata:
            values["source_import_metadata"] = int(metadata["population_size"])
    elif name == "patch.capacity" and "capacity" in metadata:
        values["source_import_metadata"] = int(metadata["capacity"])
    elif name == "patch.initial_spatial_placement_profile" and "initial_spatial_placement_profile" in metadata:
        values["source_import_metadata"] = str(metadata["initial_spatial_placement_profile"])
    elif name == "production_lifecycle.life_history_profile":
        raw = getattr(native_configuration, "life_history_profile_id", None)
        if type(raw) is str:
            values["native_source_configuration"] = raw
    return values


def _checkpoint_value_v1(name: str, projection: Any) -> Any:
    metadata = _projection_metadata_v1(projection)
    field = _runtime_field_name_v1(name)
    lifecycle_mode_abi = metadata.get("lifecycle_mode_abi")
    if type(lifecycle_mode_abi) is not int or lifecycle_mode_abi not in {0, 1}:
        lifecycle_mode_abi = None
    if name == "production_lifecycle.lifecycle_mode" and lifecycle_mode_abi is not None:
        return _public_lifecycle_mode_v1(lifecycle_mode_abi)
    if name == "production_lifecycle.turnover_enabled" and lifecycle_mode_abi is not None:
        return True
    if (
        name == "production_lifecycle.turnover_cull_fraction"
        and lifecycle_mode_abi is not None
    ):
        return 0.0 if lifecycle_mode_abi == 1 else 0.5
    if name == "patch.grid_size":
        height = metadata.get("height")
        width = metadata.get("width")
        if type(height) is int and height == width:
            return height
        grid = metadata.get("grid")
        if (
            isinstance(grid, (tuple, list))
            and len(grid) == 2
            and grid[0] == grid[1]
        ):
            return int(grid[0])
        return None
    if name in {"patch.initial_population", "patch.initial_live_count"}:
        return metadata.get("population_size", metadata.get("alive_count"))
    if name == "patch.capacity":
        return metadata.get("capacity")
    if field is not None and field in metadata:
        raw = metadata[field]
        return _normalize_scalar(raw) if type(raw) in {bool, int, float, str} else None
    return None


def _same_control_value_v1(left: Any, right: Any) -> bool:
    if type(left) is float or type(right) is float:
        try:
            return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-8)
        except (TypeError, ValueError):
            return False
    return type(left) is type(right) and left == right


def build_segment_control_readback_v1(
    *,
    profile_id: str,
    segment_ordinal: int,
    start_generation: int,
    end_generation: int,
    cells: Sequence[Mapping[str, Any]],
    builds: Sequence[Any],
    requests: Sequence[Any],
    prepared: Any,
    checkpoint_projections: Sequence[Any],
    compact_result_rows: Sequence[Mapping[str, Any]],
    parameter_contracts: Mapping[str, ParameterContractV1],
    post_dispatch_abi_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind requested controls to compiled, native, checkpoint, and metric facts."""

    count = len(cells)
    if not (
        len(builds) == count
        and len(requests) == count
        and len(checkpoint_projections) == count
    ):
        raise ValueError("production control readback inputs are not cell aligned")
    rows_by_cell = {
        str(row.get("cell_id", row.get("cell", ""))): dict(row)
        for row in compact_result_rows
        if isinstance(row, Mapping)
    }
    source_projections = tuple(getattr(prepared, "source_projections", ()))
    if len(source_projections) != count:
        raise ValueError(
            "prepared owner lacks aligned source projections for readback"
        )
    cell_receipts: list[dict[str, Any]] = []
    shared_batch_abi_controls = {
        name
        for name in parameter_contracts
        if len({
            _normalize_scalar(dict(cell.get("parameters") or {}).get(name))
            for cell in cells
            if name in dict(cell.get("parameters") or {})
        }) == 1
        and all(name in dict(cell.get("parameters") or {}) for cell in cells)
    }
    prepared_cell_ids = tuple(getattr(prepared, "cell_ids", ()))
    if prepared_cell_ids and prepared_cell_ids != tuple(
        str(cell.get("cell_id", "")) for cell in cells
    ):
        raise ValueError("prepared owner cell identities are not readback aligned")
    for cell_ordinal, (cell, build, request, source_projection, checkpoint_projection) in enumerate(zip(
        cells,
        builds,
        requests,
        source_projections,
        checkpoint_projections,
        strict=True,
    )):
        cell_id = str(cell.get("cell_id", ""))
        parameters = cell.get("parameters")
        if not cell_id or not isinstance(parameters, Mapping):
            raise ValueError(
                "production control readback requires compiled cell parameters"
            )
        runner = build.runner
        build_receipt = build.receipt
        control_rows: list[dict[str, Any]] = []
        for name in sorted(parameters):
            contract = parameter_contracts.get(name)
            if contract is None:
                continue
            requested = _normalize_scalar(parameters[name])
            compiled = _compiled_control_value_v1(
                name, requested=requested, runner=runner, receipt=build_receipt
            )
            runtime_sources = _runtime_candidates_v1(
                name,
                request=request,
                prepared=prepared,
                source_projection=source_projection,
                post_dispatch_abi_receipt=post_dispatch_abi_receipt,
                include_shared_batch_abi=name in shared_batch_abi_controls,
                cell_ordinal=cell_ordinal,
            )
            checkpoint_value = _checkpoint_value_v1(name, checkpoint_projection)
            role = contract.control_role
            independently_verifiable = contract.wrc_axis and role == "independent"
            if role == "independent_primitive":
                independently_verifiable = True
            if independently_verifiable:
                if not _same_control_value_v1(requested, compiled):
                    raise ValueError(
                        f"compiled production control differs: {name} "
                        f"requested={requested!r} compiled={compiled!r}"
                    )
                if not runtime_sources:
                    raise ValueError(
                        f"production control has no native runtime readback: {name}"
                    )
                if (
                    name == "env_cfg.patch_movement_cost_energy_scale"
                    and count > 1
                    and {
                        "post_dispatch_abi.patch_p2_config_by_cell",
                        "retained_patch_abi_config_by_cell",
                    }
                    - set(runtime_sources)
                ):
                    raise ValueError(
                        "production control lacks aligned per-cell native ABI readback: "
                        f"{name}"
                    )
                mismatched = {
                    source: value
                    for source, value in runtime_sources.items()
                    if not _same_control_value_v1(compiled, value)
                }
                if mismatched:
                    raise ValueError(
                        f"native production control differs: {name} "
                        f"compiled={compiled!r} mismatched={mismatched!r}"
                    )
                if checkpoint_value is not None and not _same_control_value_v1(
                    compiled, checkpoint_value
                ):
                    raise ValueError(
                        f"checkpoint production control differs: {name} "
                        f"compiled={compiled!r} checkpoint={checkpoint_value!r}"
                    )
                status = "READBACK_COMPLETE"
                applied = compiled
            else:
                # Route constants, aliases, bundle selectors, provenance labels,
                # and derived values remain visible but are not advertised as
                # independent WRC axes.  When runtime evidence exists it must
                # still agree with the compiled value.
                mismatched = {
                    source: value
                    for source, value in runtime_sources.items()
                    if not _same_control_value_v1(compiled, value)
                }
                if mismatched:
                    raise ValueError(
                        f"declared production control differs at runtime: {name} "
                        f"compiled={compiled!r} mismatched={mismatched!r}"
                    )
                status = "NON_INDEPENDENT_CONTROL_DECLARED"
                applied = compiled
            control_rows.append(
                {
                    "control": name,
                    "control_role": role,
                    "wrc_axis": bool(contract.wrc_axis),
                    "alias_of": contract.alias_of,
                    "requested": requested,
                    "compiled": compiled,
                    "runtime_sources": runtime_sources,
                    "applied": applied,
                    "checkpoint_projection_value": checkpoint_value,
                    "checkpoint_binding_value": applied,
                    "checkpoint_binding_status": (
                        "BOUND_IN_PROVIDER_CHECKPOINT_PAYLOAD"
                    ),
                    "readback": applied,
                    "status": status,
                }
            )
        bindings_payload = {
            "cell_id": cell_id,
            "cell_ordinal": cell_ordinal,
            "controls": control_rows,
        }
        bindings_sha256 = canonical_sha256(bindings_payload)
        metric_row = rows_by_cell.get(cell_id)
        metric_row_sha256 = (
            canonical_sha256(metric_row) if metric_row is not None else None
        )
        cell_receipts.append(
            {
                **bindings_payload,
                "control_bindings_sha256": bindings_sha256,
                "authoritative_state_projection_sha256": projection_digest(
                    checkpoint_projection
                ),
                "metric_row_sha256": metric_row_sha256,
                "metric_binding_status": (
                    "BOUND" if metric_row_sha256 is not None else "NOT_EMITTED"
                ),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": PRODUCTION_CONTROL_READBACK_SCHEMA_V1,
        "profile_id": profile_id,
        "segment_ordinal": segment_ordinal,
        "start_generation": start_generation,
        "end_generation": end_generation,
        "post_dispatch_abi_receipt_sha256": post_dispatch_abi_receipt.get(
            "receipt_sha256"
        ),
        "cells": cell_receipts,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload
