"""Read-only Flat native-owner adapter for canonical population reporting."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any

from simulation.metrics_pkg.cognitive_status import (
    flat_population_reporting_records_from_sources,
)
from simulation.metrics_pkg.population_reporting import FlatPopulationReportingRecordsV1


_BEHAVIOR_ROW_FIELDS = (
    "generation",
    "agent_id",
    "cohort",
    "move_n",
    "move_s",
    "move_e",
    "move_w",
    "gather",
    "rest",
    "attack",
    "lifespan_steps",
    "energy_collected",
    "offspring",
    "attacks_made",
    "fitness",
    "alive_end",
    "eligible_for_selection",
    "kept_next",
    "injury_end",
    "injury_delta",
    "aggression_bias",
)

_BEHAVIOR_NONNEGATIVE_INT_FIELDS = (
    "generation",
    "agent_id",
    "move_n",
    "move_s",
    "move_e",
    "move_w",
    "gather",
    "rest",
    "attack",
    "lifespan_steps",
    "offspring",
    "attacks_made",
)

_BEHAVIOR_BINARY_FIELDS = (
    "alive_end",
    "eligible_for_selection",
    "kept_next",
)

_BEHAVIOR_FINITE_FIELDS = (
    "energy_collected",
    "fitness",
    "injury_end",
    "injury_delta",
    "aggression_bias",
)

_REQUIRED_PROJECTION_METADATA_FIELDS = (
    "world",
    "native_runtime_fixed_policy_family",
    "generation",
    "capacity",
    "alive_count",
    "developmental_maturity_age",
    "policy_energy_observation_contract_schema_version",
)

_POLICY_ENERGY_OBSERVATION_CONTRACT_SCHEMA = "eiel_policy_energy_observation_shadow_v1"
_PATCH_REPORTING_PROFILES = frozenset(("P0", "P1", "P2"))
_PATCH_REPORTING_PROFILE_SCOPE = "full_profile"
_PATCH_REPORTING_CONTRACT_SCHEMA = "eiel_patch_native_import_contract_v1"
_JUNGLE_REPORTING_FLAGS = (
    "jungle_predator_pressure_enable",
    "jungle_reproduction_emissions_enable",
    "jungle_juvenile_predator_vulnerability_enable",
    "jungle_terrain_protection_enable",
    "jungle_ecological_recovery_enable",
)

_FLOAT_GENOME_FIELDS = (
    "exploration_bias",
    "aggression_bias",
    "mutation_rate",
    "reproduction_threshold",
    "action_temperature",
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
)

_INT_GENOME_FIELDS = (
    "hidden_size",
    "memory_dim",
    "sensor_channel_mask",
    "sensor_radius",
    "introspection_channel_mask",
    "self_regulation_control_mask",
    "self_modification_target_mask",
    "directed_inheritance_channel_mask",
)

_REQUIRED_BOOL_TENSORS = (
    "alive_mask",
    "policy_use_gru",
)

_REQUIRED_INT_TENSORS = (
    "slot_to_agent_id",
    "survival_steps",
    "offspring_count",
    "pred_err_count",
    "patch_habitat_lineage_id",
    "lineage_founder_id",
    *(f"genome_{name}" for name in _INT_GENOME_FIELDS),
)

_REQUIRED_FLOAT_TENSORS = (
    "policy_energy_observation",
    "health",
    "pred_err_sum",
    *(f"genome_{name}" for name in _FLOAT_GENOME_FIELDS),
)

_UNSUPPORTED_TRUE_SIM_FEATURES = (
    "genome_c1_enable",
    "genome_c2_enable",
    "chemistry_functional_coupling_enable",
    "maintenance_enable",
    "enable_evolvable_sensor_channel_mask",
    "enable_evolvable_sensor_radius",
    "enable_developmental_cognition",
    "structural_live_semantics_enable",
    "enable_introspection_channels",
    "default_introspection_channels_enabled",
    "enable_evolvable_introspection_channels",
    "enable_introspection_policy_input",
    "enable_self_regulation",
    "enable_self_regulation_surface",
    "default_self_regulation_controls_enabled",
    "enable_evolvable_self_regulation_controls",
    "enable_self_regulation_policy_output",
    "enable_self_modification",
    "default_self_modification_targets_enabled",
    "enable_evolvable_self_modification_targets",
    "enable_self_modification_policy_output",
    "enable_directed_inheritance",
    "enable_directed_inheritance_surface",
    "default_directed_inheritance_channels_enabled",
    "enable_evolvable_directed_inheritance_channels",
    "enable_directed_inheritance_policy_output",
)

_UNSUPPORTED_OVERRIDE_FLAGS = (
    "ablate_memory",
    "force_memory_disabled",
    "force_min_memory_dim",
    "force_max_memory_dim",
    "force_default_memory_dim",
    "reset_memory_each_tick",
    "force_prediction_frozen",
    "force_prediction_zero_lr",
    "force_prediction_default",
    "reset_prediction_each_tick",
    "ablate_sensor_channel_mask",
    "force_all_sensor_channels_enabled",
    "force_minimal_sensor_channels_enabled",
    "ablate_sensor_radius",
    "force_min_sensor_radius",
    "force_max_sensor_radius",
    "force_default_sensor_radius",
    "ablate_developmental_cognition",
    "force_adult_cognition",
    "force_juvenile_cognition",
    "force_legacy_full_cognition",
    "disable_cognitive_growth",
    "ablate_introspection",
    "force_no_introspection_channels",
    "force_minimal_introspection_channels",
    "force_all_introspection_channels",
    "force_default_introspection_channels",
    "mask_introspection_values_to_neutral",
    "ablate_self_regulation",
    "force_no_self_regulation",
    "force_all_self_regulation_controls",
    "force_default_self_regulation_controls",
    "mask_self_regulation_outputs_to_neutral",
    "freeze_self_regulation_state",
    "reset_self_regulation_each_tick",
    "disable_self_regulation_decay",
    "ablate_self_modification",
    "force_no_self_modification",
    "force_all_self_modification_targets",
    "force_default_self_modification_targets",
    "mask_self_modification_commands_to_neutral",
    "freeze_self_modification_state",
    "reset_self_modification_each_tick",
    "reset_self_modification_on_reproduction",
    "disable_self_modification_decay",
    "ablate_directed_inheritance",
    "force_no_directed_inheritance",
    "force_all_directed_inheritance_channels",
    "force_default_directed_inheritance_channels",
    "mask_directed_inheritance_commands_to_neutral",
    "freeze_directed_inheritance_payload",
    "reset_directed_inheritance_payload_on_birth",
    "reset_directed_inheritance_payload_on_maturity",
)


@dataclass(frozen=True, slots=True)
class _GenomeView:
    values: MappingProxyType

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True, slots=True)
class _StatsView:
    survival_steps: int
    offspring_count: int
    pred_err_count: int
    pred_err_sum: float
    patch_habitat_lineage_id: int
    lineage_id: int
    founder_lineage_id: int


@dataclass(frozen=True, slots=True)
class _ResidentReportingView:
    agent_id: int
    genome: _GenomeView
    stats: _StatsView
    energy: float
    health: float
    drives: tuple[float, float, float, float]
    alive: bool = True
    self_regulation_state: None = None
    self_modification_state: None = None
    directed_inheritance_payload: None = None


def native_flat_population_reporting_records(
    *,
    tensors: Any,
    behavior_rows: Any,
    sim_cfg: Any,
    env_cfg: Any,
    metadata: Any = None,
    reporting_generation: Any = None,
) -> FlatPopulationReportingRecordsV1:
    """Adapt authoritative live Flat owner rows through the shared reducer."""

    _validate_supported_envelope(sim_cfg=sim_cfg, env_cfg=env_cfg, metadata=metadata)
    owner_tensors, capacity = _validated_owner_tensors(tensors)
    live_rows = _live_slot_rows(owner_tensors, capacity=capacity)
    projection_generation = _validate_projection_metadata(
        metadata,
        capacity=capacity,
        alive_count=len(live_rows),
    )
    generation = resolve_native_flat_reporting_generation(
        metadata,
        reporting_generation=reporting_generation,
        projection_generation=projection_generation,
    )
    _validate_maturity_age_authority(sim_cfg=sim_cfg, metadata=metadata)
    validated_behavior_rows = validated_native_flat_behavior_rows(
        behavior_rows,
        expected_generation=generation,
    )

    sources = tuple(
        _resident_reporting_view(owner_tensors, slot=slot, agent_id=agent_id)
        for agent_id, slot in live_rows
    )
    selected_parent_ids = _selected_parent_ids(validated_behavior_rows)
    source_by_id = {source.agent_id: source for source in sources}
    missing = [agent_id for agent_id in selected_parent_ids if agent_id not in source_by_id]
    if missing:
        raise ValueError(
            "native Flat population reporting selected parents are absent from live owner rows: "
            f"{missing}"
        )
    parents = tuple(source_by_id[agent_id] for agent_id in selected_parent_ids)
    return flat_population_reporting_records_from_sources(
        sources,
        selected_parent_sources=parents,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
    )


def resolve_native_flat_reporting_generation(
    metadata: Mapping[str, Any],
    *,
    reporting_generation: Any = None,
    projection_generation: int | None = None,
) -> int:
    """Resolve the completed generation represented by a resident projection."""

    if projection_generation is None:
        if "generation" not in metadata:
            raise ValueError(
                "scope-locked native Flat reporting is missing required metadata field generation"
            )
        projection_generation = _nonnegative_metadata_int(
            metadata["generation"],
            context="metadata.generation",
        )
    if reporting_generation is None:
        return int(projection_generation)

    completed_generation = _nonnegative_metadata_int(
        reporting_generation,
        context="reporting_generation",
    )
    if completed_generation == int(projection_generation):
        return completed_generation

    if int(projection_generation) == completed_generation + 1:
        missing = [name for name in ("tick", "agent_grid_runner_phase") if name not in metadata]
        if missing:
            raise ValueError(
                "scope-locked native Flat next-generation projection is missing boundary metadata: "
                f"{missing}"
            )
        tick = _nonnegative_metadata_int(metadata["tick"], context="metadata.tick")
        phase_value = metadata["agent_grid_runner_phase"]
        phase = phase_value.strip() if isinstance(phase_value, str) else ""
        if tick == 0 and phase == "generation_boundary":
            return completed_generation
        raise ValueError(
            "scope-locked native Flat next-generation reporting projection is not at a "
            "generation boundary: "
            f"projection={projection_generation} reporting={completed_generation} "
            f"tick={tick} phase={phase or 'missing'}"
        )

    raise ValueError(
        "scope-locked native Flat projection/metrics generation mismatch: "
        f"projection={projection_generation} metrics={completed_generation}"
    )


def _resident_reporting_view(
    tensors: Mapping[str, Any],
    *,
    slot: int,
    agent_id: int,
) -> _ResidentReportingView:
    genome_values = {
        name: round(_finite_float(tensors, f"genome_{name}", slot), 8)
        for name in _FLOAT_GENOME_FIELDS
    }
    genome_values.update(
        {
            name: _int_value(tensors, f"genome_{name}", slot)
            for name in _INT_GENOME_FIELDS
        }
    )
    use_gru = _bool_value(tensors, "policy_use_gru", slot)
    memory_dim = int(genome_values["memory_dim"])
    if use_gru or memory_dim != 0:
        raise ValueError(
            "native Flat population reporting fixed-MLP owner row is recurrent: "
            f"agent_id={agent_id} slot={slot} use_gru={use_gru} memory_dim={memory_dim}"
        )
    if "policy_mem_dim" in tensors and _int_value(tensors, "policy_mem_dim", slot) != 0:
        raise ValueError(
            "native Flat population reporting fixed-MLP owner row has nonzero policy_mem_dim: "
            f"agent_id={agent_id} slot={slot}"
        )
    genome_values.update({"use_gru": use_gru, "modules": (), "connections": ()})

    survival_steps = _nonnegative_int(tensors, "survival_steps", slot, agent_id=agent_id)
    offspring_count = _nonnegative_int(tensors, "offspring_count", slot, agent_id=agent_id)
    pred_err_count = _nonnegative_int(tensors, "pred_err_count", slot, agent_id=agent_id)
    lineage = _int_value(tensors, "patch_habitat_lineage_id", slot)
    founder_lineage = _int_value(tensors, "lineage_founder_id", slot)
    drives = _drive_values(tensors, slot=slot)
    return _ResidentReportingView(
        agent_id=agent_id,
        genome=_GenomeView(MappingProxyType(genome_values)),
        stats=_StatsView(
            survival_steps=survival_steps,
            offspring_count=offspring_count,
            pred_err_count=pred_err_count,
            pred_err_sum=_finite_float(tensors, "pred_err_sum", slot),
            patch_habitat_lineage_id=lineage,
            lineage_id=lineage,
            founder_lineage_id=founder_lineage,
        ),
        energy=_finite_float(tensors, "policy_energy_observation", slot),
        health=_finite_float(tensors, "health", slot),
        drives=drives,
    )


def validate_native_population_reporting_envelope(
    *,
    sim_cfg: Any,
    env_cfg: Any,
    metadata: Any,
) -> str:
    """Validate the exact world envelope admitted to the shared row projection."""

    if sim_cfg is None or env_cfg is None:
        raise ValueError("native Flat population reporting requires sim_cfg and env_cfg")
    if not isinstance(metadata, Mapping):
        raise ValueError("native Flat population reporting metadata must be a mapping")
    missing_metadata = [
        name for name in _REQUIRED_PROJECTION_METADATA_FIELDS if name not in metadata
    ]
    if missing_metadata:
        raise ValueError(
            "native Flat population reporting is missing required metadata fields: "
            f"{missing_metadata}"
        )

    world_value = metadata["world"]
    world = world_value.strip().lower() if isinstance(world_value, str) else ""
    if world == "flat":
        _validate_flat_supported_envelope(
            sim_cfg=sim_cfg,
            env_cfg=env_cfg,
            metadata=metadata,
        )
        return world
    if world == "patch":
        _validate_patch_supported_envelope(
            sim_cfg=sim_cfg,
            env_cfg=env_cfg,
            metadata=metadata,
        )
        return world
    raise ValueError(
        "native Flat population reporting requires an admitted world=flat or world=patch, "
        f"got {world or 'missing'}"
    )


def _validate_supported_envelope(*, sim_cfg: Any, env_cfg: Any, metadata: Any) -> None:
    """Backward-compatible private entry point for the resident adapter."""

    validate_native_population_reporting_envelope(
        sim_cfg=sim_cfg,
        env_cfg=env_cfg,
        metadata=metadata,
    )


def _validate_flat_supported_envelope(
    *,
    sim_cfg: Any,
    env_cfg: Any,
    metadata: Mapping[str, Any],
) -> None:
    if _cfg_bool(env_cfg, "patch_world_enable"):
        raise ValueError("native Flat population reporting requires world=flat, got flat")
    for name in (
        "jungle_predator_pressure_enable",
        "jungle_ecological_recovery_enable",
        "wdl_delayed_hazard_enable",
        "reproduce_requires_mate",
    ):
        if _cfg_bool(env_cfg, name):
            raise ValueError(f"native Flat population reporting does not support active env feature {name}")

    family = str(_cfg_get(sim_cfg, "native_runtime_fixed_policy_family", "") or "").strip().lower()
    metadata_family_value = metadata["native_runtime_fixed_policy_family"]
    metadata_family = (
        metadata_family_value.strip().lower()
        if isinstance(metadata_family_value, str)
        else ""
    )
    if family != "mlp" or metadata_family != "mlp":
        raise ValueError(
            "native Flat population reporting requires the fixed-MLP envelope: "
            f"sim_cfg={family or 'missing'} metadata={metadata_family or 'missing'}"
        )
    if not _cfg_bool(sim_cfg, "native_runtime_cxx26_scope_lock_enable"):
        raise ValueError(
            "native Flat population reporting requires native_runtime_cxx26_scope_lock_enable"
        )
    for name in ("device", "bank_device"):
        value = str(_cfg_get(sim_cfg, name, "cpu") or "cpu").strip().lower()
        if value != "cpu":
            raise ValueError(f"native Flat population reporting requires {name}=cpu, got {value}")

    active = [name for name in _UNSUPPORTED_TRUE_SIM_FEATURES if _cfg_bool(sim_cfg, name)]
    active.extend(name for name in _UNSUPPORTED_OVERRIDE_FLAGS if _cfg_bool(sim_cfg, name))
    if active:
        raise ValueError(
            "native Flat population reporting cannot adapt active unsupported features: "
            f"{sorted(set(active))}"
        )
    if not (
        _cfg_bool(sim_cfg, "force_prediction_disabled")
        or _cfg_bool(sim_cfg, "ablate_prediction")
    ):
        raise ValueError(
            "native Flat population reporting requires prediction bookkeeping to be disabled"
        )


def _validate_patch_supported_envelope(
    *,
    sim_cfg: Any,
    env_cfg: Any,
    metadata: Mapping[str, Any],
) -> None:
    """Admit only the finite Patch profiles to this projection."""

    if not _cfg_bool(env_cfg, "patch_world_enable"):
        raise ValueError(
            "native Patch population reporting requires world=patch; configured world=flat"
        )
    profile = metadata.get("native_runtime_patch_profile")
    if type(profile) is not str or profile not in _PATCH_REPORTING_PROFILES:
        raise ValueError(
            "native Patch population reporting requires an exact finite P0/P1/P2 profile"
        )
    if metadata.get("native_runtime_patch_profile_scope") != _PATCH_REPORTING_PROFILE_SCOPE:
        raise ValueError(
            "native Patch population reporting requires native_runtime_patch_profile_scope="
            "full_profile"
        )
    if (
        metadata.get("native_runtime_patch_contract_schema_version")
        != _PATCH_REPORTING_CONTRACT_SCHEMA
    ):
        raise ValueError("native Patch population reporting has an unsupported Patch contract schema")

    capabilities = metadata.get("capability_mask")
    if not isinstance(capabilities, Mapping):
        raise ValueError("native Patch population reporting requires a capability_mask")
    if capabilities.get("patch") is not True or capabilities.get("flat") is not False:
        raise ValueError(
            "native Patch population reporting requires capability_mask patch=true and flat=false"
        )

    from .patch_profile import classify_patch_profile

    wdl_enabled = getattr(env_cfg, "wdl_delayed_hazard_enable", False)
    wdl_observation_enabled = getattr(
        env_cfg,
        "wdl_delayed_hazard_observation_enable",
        False,
    )
    if type(wdl_enabled) is not bool or type(wdl_observation_enabled) is not bool:
        raise ValueError("native Patch reporting requires boolean WDL feature switches")
    if wdl_observation_enabled and not wdl_enabled:
        raise ValueError("native Patch reporting rejects WDL observations without WDL")
    jungle_values = {
        name: getattr(env_cfg, name, False) for name in _JUNGLE_REPORTING_FLAGS
    }
    if any(type(value) is not bool for value in jungle_values.values()):
        raise ValueError("native Patch reporting requires boolean Jungle feature switches")
    jungle_enabled = any(jungle_values.values())
    if jungle_values["jungle_juvenile_predator_vulnerability_enable"] and not jungle_values[
        "jungle_predator_pressure_enable"
    ]:
        raise ValueError("native Patch reporting rejects juvenile Jungle pressure without predators")
    if jungle_values["jungle_terrain_protection_enable"] and not jungle_values[
        "jungle_predator_pressure_enable"
    ]:
        raise ValueError("native Patch reporting rejects Jungle terrain protection without predators")
    classification_env_cfg = env_cfg
    if wdl_enabled or jungle_enabled:
        classification_env_cfg = copy.copy(env_cfg)
        if wdl_enabled:
            classification_env_cfg.wdl_delayed_hazard_enable = False
            classification_env_cfg.wdl_delayed_hazard_observation_enable = False
        if jungle_enabled:
            for name in _JUNGLE_REPORTING_FLAGS:
                setattr(classification_env_cfg, name, False)
    classification = classify_patch_profile(
        sim_cfg,
        classification_env_cfg,
        world="patch",
    )
    if not classification.supported or classification.profile != profile:
        reasons = ",".join(classification.unsupported_reasons) or "profile_metadata_mismatch"
        raise ValueError(
            "native Patch population reporting requires metadata/config agreement on "
            f"finite profile {profile}: {reasons}"
        )
    if capabilities.get("wdl", False) is not wdl_enabled:
        raise ValueError(
            "native Patch population reporting WDL capability contradicts configuration"
        )
    if capabilities.get("jungle", False) is not jungle_enabled:
        raise ValueError(
            "native Patch population reporting Jungle capability contradicts configuration"
        )
    if bool(metadata.get("native_runtime_jungle_execution_enable", False)) is not jungle_enabled:
        raise ValueError(
            "native Patch population reporting Jungle execution marker contradicts configuration"
        )


def _validated_owner_tensors(tensors: Any) -> tuple[dict[str, Any], int]:
    if not isinstance(tensors, Mapping):
        raise ValueError("native Flat population reporting tensors must be a mapping")
    required = (
        *_REQUIRED_BOOL_TENSORS,
        *_REQUIRED_INT_TENSORS,
        *_REQUIRED_FLOAT_TENSORS,
        "drives",
    )
    missing = [name for name in required if name not in tensors]
    if missing:
        raise ValueError(f"native Flat population reporting is missing required tensors: {missing}")

    owner_tensors: dict[str, Any] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    for name in required:
        tensor = tensors[name]
        shape = _tensor_shape(tensor, name=name)
        expected_rank = 2 if name == "drives" else 1
        if len(shape) != expected_rank:
            raise ValueError(
                f"native Flat population reporting tensor {name} must have rank "
                f"{expected_rank}, got {shape}"
            )
        if name == "drives" and shape[1:] != (4,):
            raise ValueError(
                "native Flat population reporting tensor drives must have shape "
                f"[capacity, 4], got {shape}"
            )
        _validate_tensor_device(tensor, name=name)
        _validate_tensor_dtype(tensor, name=name)
        owner_tensors[name] = tensor
        shapes[name] = shape

    if "policy_mem_dim" in tensors:
        policy_mem_dim = tensors["policy_mem_dim"]
        shape = _tensor_shape(policy_mem_dim, name="policy_mem_dim")
        if len(shape) != 1:
            raise ValueError(
                "native Flat population reporting tensor policy_mem_dim must have rank 1, "
                f"got {shape}"
            )
        _validate_tensor_device(policy_mem_dim, name="policy_mem_dim")
        _validate_tensor_dtype(policy_mem_dim, name="policy_mem_dim", expected="int")
        owner_tensors["policy_mem_dim"] = policy_mem_dim
        shapes["policy_mem_dim"] = shape

    capacity = shapes["alive_mask"][0]
    mismatched = [name for name, shape in shapes.items() if shape[0] != capacity]
    if mismatched:
        raise ValueError(
            "native Flat population reporting tensors disagree on slot capacity: "
            f"capacity={capacity} fields={mismatched}"
        )
    return owner_tensors, capacity


def _live_slot_rows(
    tensors: Mapping[str, Any],
    *,
    capacity: int,
) -> tuple[tuple[int, int], ...]:
    rows: list[tuple[int, int]] = []
    seen_ids: set[int] = set()
    for slot in range(capacity):
        alive = _bool_value(tensors, "alive_mask", slot)
        agent_id = _int_value(tensors, "slot_to_agent_id", slot)
        if alive != (agent_id >= 0):
            raise ValueError(
                "native Flat population reporting alive_mask and slot_to_agent_id disagree "
                f"at slot {slot}"
            )
        if not alive:
            continue
        if agent_id in seen_ids:
            raise ValueError(f"native Flat population reporting has duplicate live agent_id {agent_id}")
        seen_ids.add(agent_id)
        rows.append((agent_id, slot))
    rows.sort(key=lambda item: item[0])
    return tuple(rows)


def validated_native_flat_behavior_rows(
    behavior_rows: Any,
    *,
    expected_generation: int,
) -> tuple[dict[str, Any], ...]:
    """Return scope-locked behavior rows after exact schema validation."""

    if behavior_rows is None:
        return ()
    if isinstance(behavior_rows, (Mapping, str, bytes, bytearray)):
        raise ValueError("native Flat population reporting behavior_rows must be an iterable of rows")
    try:
        rows = list(behavior_rows)
    except TypeError as exc:
        raise ValueError("native Flat population reporting behavior_rows must be iterable") from exc

    expected_generation = _nonnegative_metadata_int(
        expected_generation,
        context="expected behavior generation",
    )
    validated: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, raw in enumerate(rows):
        row = _behavior_row(raw, index=index)
        _validate_behavior_row_fields(row, index=index)
        generation = _strict_int(
            row["generation"],
            context=f"behavior_rows[{index}].generation",
        )
        if generation != expected_generation:
            raise ValueError(
                "native Flat population reporting behavior/projection generation mismatch: "
                f"behavior_rows[{index}]={generation} projection={expected_generation}"
            )
        agent_id = _strict_int(
            row["agent_id"],
            context=f"behavior_rows[{index}].agent_id",
        )
        if agent_id in seen:
            raise ValueError(
                "native Flat population reporting behavior_rows has duplicate agent_id: "
                f"behavior_rows[{index}]={agent_id}"
            )
        seen.add(agent_id)
        validated.append(row)
    return tuple(validated)


def _selected_parent_ids(behavior_rows: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    selected = {
        _strict_int(row["agent_id"], context=f"behavior_rows[{index}].agent_id")
        for index, row in enumerate(behavior_rows)
        if _binary_int(
            row["kept_next"],
            context=f"behavior_rows[{index}].kept_next",
        )
    }
    return tuple(sorted(selected))


def _behavior_row(raw: Any, *, index: int) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        row = dict(raw)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        if len(raw) != len(_BEHAVIOR_ROW_FIELDS):
            raise ValueError(
                f"native Flat population reporting behavior_rows[{index}] must have "
                f"{len(_BEHAVIOR_ROW_FIELDS)} fields, got {len(raw)}"
            )
        row = {
            name: raw[field_index]
            for field_index, name in enumerate(_BEHAVIOR_ROW_FIELDS)
        }
    else:
        raise ValueError(f"native Flat population reporting behavior_rows[{index}] is not a row")
    missing = [name for name in _BEHAVIOR_ROW_FIELDS if name not in row]
    if missing:
        raise ValueError(
            f"native Flat population reporting behavior_rows[{index}] is missing fields: {missing}"
        )
    return row


def _validate_behavior_row_fields(row: Mapping[str, Any], *, index: int) -> None:
    for name in _BEHAVIOR_NONNEGATIVE_INT_FIELDS:
        value = _strict_int(row[name], context=f"behavior_rows[{index}].{name}")
        if value < 0:
            raise ValueError(
                f"native Flat population reporting behavior_rows[{index}].{name} "
                "must be nonnegative"
            )
    for name in _BEHAVIOR_BINARY_FIELDS:
        _binary_int(row[name], context=f"behavior_rows[{index}].{name}")
    for name in _BEHAVIOR_FINITE_FIELDS:
        _strict_finite_number(row[name], context=f"behavior_rows[{index}].{name}")
    cohort = row["cohort"]
    if not isinstance(cohort, str) or not cohort.strip():
        raise ValueError(
            f"native Flat population reporting behavior_rows[{index}].cohort "
            "must be a non-empty string"
        )


def _validate_projection_metadata(
    metadata: Mapping[str, Any],
    *,
    capacity: int,
    alive_count: int,
) -> int:
    generation = _nonnegative_metadata_int(
        metadata["generation"],
        context="metadata.generation",
    )
    metadata_capacity = _nonnegative_metadata_int(
        metadata["capacity"],
        context="metadata.capacity",
    )
    if metadata_capacity != capacity:
        raise ValueError("native Flat population reporting metadata capacity disagrees with tensors")
    metadata_alive_count = _nonnegative_metadata_int(
        metadata["alive_count"],
        context="metadata.alive_count",
    )
    if metadata_alive_count != alive_count:
        raise ValueError("native Flat population reporting metadata alive_count disagrees with tensors")
    energy_schema = str(metadata.get("policy_energy_observation_contract_schema_version", "") or "")
    if energy_schema != _POLICY_ENERGY_OBSERVATION_CONTRACT_SCHEMA:
        raise ValueError(
            "native Flat population reporting has unsupported policy energy observation schema: "
            f"{energy_schema or 'missing'}"
        )
    return generation


def _validate_maturity_age_authority(
    *,
    sim_cfg: Any,
    metadata: Mapping[str, Any],
) -> None:
    configured = _nonnegative_metadata_int(
        _cfg_get(sim_cfg, "developmental_maturity_age", 20),
        context="sim_cfg.developmental_maturity_age",
    )
    projected = _nonnegative_metadata_int(
        metadata["developmental_maturity_age"],
        context="metadata.developmental_maturity_age",
    )
    if projected != configured:
        raise ValueError(
            "native Flat population reporting maturity_age authority disagrees between "
            f"metadata={projected} and sim_cfg={configured}"
        )


def _tensor_shape(value: Any, *, name: str) -> tuple[int, ...]:
    if not hasattr(value, "detach") or not hasattr(value, "shape"):
        raise ValueError(f"native Flat population reporting field {name} is not a tensor")
    try:
        return tuple(int(dim) for dim in value.shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"native Flat population reporting tensor {name} has an invalid shape"
        ) from exc


def _validate_tensor_device(value: Any, *, name: str) -> None:
    device = str(getattr(value, "device", "") or "").strip().lower()
    if not device.startswith("cpu"):
        raise ValueError(
            f"native Flat population reporting tensor {name} must be on CPU, got {device or 'missing'}"
        )


def _validate_tensor_dtype(value: Any, *, name: str, expected: str | None = None) -> None:
    if expected is None:
        if name in _REQUIRED_BOOL_TENSORS:
            expected = "bool"
        elif name in _REQUIRED_INT_TENSORS:
            expected = "int"
        else:
            expected = "float"
    dtype = str(getattr(value, "dtype", "") or "").removeprefix("torch.").lower()
    valid = {
        "bool": dtype == "bool",
        "int": dtype.startswith("int") or dtype.startswith("uint"),
        "float": dtype.startswith("float") or dtype == "bfloat16",
    }[expected]
    if not valid:
        raise ValueError(
            f"native Flat population reporting tensor {name} must have {expected} dtype, "
            f"got {dtype or 'missing'}"
        )
    if name == "policy_energy_observation":
        if dtype != "float64":
            raise ValueError(
                "native Flat population reporting tensor policy_energy_observation must be float64"
            )
    elif (name in _REQUIRED_FLOAT_TENSORS or name == "drives") and dtype != "float32":
        raise ValueError(
            f"native Flat population reporting tensor {name} must be float32, got {dtype}"
        )


def _tensor_scalar(tensors: Mapping[str, Any], name: str, slot: int) -> Any:
    try:
        return tensors[name][slot].detach().cpu().item()
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"native Flat population reporting cannot read tensor {name} at slot {slot}"
        ) from exc


def _bool_value(tensors: Mapping[str, Any], name: str, slot: int) -> bool:
    value = _tensor_scalar(tensors, name, slot)
    if not isinstance(value, bool):
        raise ValueError(
            f"native Flat population reporting tensor {name}[{slot}] is not boolean"
        )
    return value


def _int_value(tensors: Mapping[str, Any], name: str, slot: int) -> int:
    value = _tensor_scalar(tensors, name, slot)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"native Flat population reporting tensor {name}[{slot}] is not an integer")
    return value


def _finite_float(tensors: Mapping[str, Any], name: str, slot: int) -> float:
    value = _tensor_scalar(tensors, name, slot)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"native Flat population reporting tensor {name}[{slot}] is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"native Flat population reporting tensor {name}[{slot}] is non-finite")
    return result


def _drive_values(
    tensors: Mapping[str, Any],
    *,
    slot: int,
) -> tuple[float, float, float, float]:
    try:
        raw = tensors["drives"][slot].detach().cpu().tolist()
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"native Flat population reporting cannot read tensor drives at slot {slot}"
        ) from exc
    values = tuple(float(value) for value in raw)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError(
            f"native Flat population reporting tensor drives[{slot}] must contain four finite values"
        )
    return values  # type: ignore[return-value]


def _nonnegative_int(
    tensors: Mapping[str, Any],
    name: str,
    slot: int,
    *,
    agent_id: int,
) -> int:
    value = _int_value(tensors, name, slot)
    if value < 0:
        raise ValueError(
            f"native Flat population reporting tensor {name} is negative for agent_id={agent_id}"
        )
    return value


def _strict_int(value: Any, *, context: str) -> int:
    if not isinstance(value, bool) and isinstance(value, Integral):
        return int(value)
    raise ValueError(f"native Flat population reporting {context} must be an integer")


def _nonnegative_metadata_int(value: Any, *, context: str) -> int:
    result = _strict_int(value, context=context)
    if result < 0:
        raise ValueError(
            f"native Flat population reporting {context} must be nonnegative"
        )
    return result


def _strict_finite_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"native Flat population reporting {context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"native Flat population reporting {context} must be finite")
    return result


def _binary_int(value: Any, *, context: str) -> int:
    result = _strict_int(value, context=context)
    if result not in (0, 1):
        raise ValueError(f"native Flat population reporting {context} must be 0 or 1")
    return result


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _cfg_bool(cfg: Any, name: str) -> bool:
    return bool(_cfg_get(cfg, name, False))


def _get(mapping: Any, name: str, default: Any = None) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(name, default)
    return default
