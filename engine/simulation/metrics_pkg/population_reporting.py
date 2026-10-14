"""Immutable canonical population records for Flat public reporting.

The contract is deliberately independent of live simulation objects.  Route
adapters copy primitive values into it and shared reporting reducers consume
the resulting mappings without gaining access to Agents, RNGs, or mutable
owner state.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


FLAT_POPULATION_REPORTING_SCHEMA_VERSION = "flat_population_reporting_v1"

# This is the complete primitive input surface emitted by the reference Flat
# adapter and consumed by the shared cognitive/developmental reducer.
FLAT_REPORTING_REQUIRED_FIELDS = (
    "agent_id",
    "capacity",
    "genetic_memory",
    "memory",
    "effective_memory",
    "memory_enabled",
    "policy_memory",
    "recurrent_state",
    "memory_capacity_fraction",
    "memory_ladder_index",
    "memory_cost_hyp",
    "memory_cost_charged",
    "memory_cost_fraction",
    "memory_ablate",
    "memory_force_disabled",
    "memory_force_min",
    "memory_force_max",
    "memory_force_default",
    "memory_reset_each_tick",
    "hidden",
    "pred_enabled",
    "pred_lr",
    "effective_pred_lr",
    "prediction_capacity",
    "mutation_rate",
    "action_temperature",
    "reproduction_threshold",
    "module_count",
    "connection_count",
    "c1_operator_noop_count",
    "c1_capacity_blocked_count",
    "c1_capacity_blocked_reason_counts_json",
    "c1_op_count__bounded_rewire",
    "c1_op_count__duplicate_with_decay",
    "c1_op_count__edge_split_insertion",
    "c1_op_count__fusion_merge_compression",
    "c1_op_count__gate_drift",
    "c1_op_count__motif_duplication",
    "c1_op_count__promoter_expression_mutation",
    "b3_locality_dropped_connections",
    "prediction_state_available",
    "prediction_update_enabled",
    "pred_err",
    "prediction_cost_hyp",
    "prediction_cost_charged",
    "prediction_cost_fraction",
    "prediction_ablate",
    "prediction_force_disabled",
    "prediction_force_frozen",
    "prediction_force_zero_lr",
    "prediction_force_default",
    "prediction_reset_each_tick",
    "direct_prediction_pressure",
    "prediction_state_abs",
    "prediction_state_abs_p90",
    "prediction_state_delta",
    "prediction_state_delta_p90",
    "prediction_update_count",
    "prediction_nonzero_fraction",
    "prediction_effective_utilization",
    "prediction_saturation_proxy",
    "prediction_action_differentiation",
    "drive_gain",
    "drive_noise",
    "drive_inertia",
    "hunger",
    "fear",
    "pain",
    "fatigue",
    "cost",
    "memory_cost",
    "policy_memory_cost",
    "cost_frac",
    "memory_component",
    "prediction_component",
    "drive_component",
    "perception_component",
    "sensor_channel_count",
    "sensor_channel_potential",
    "sensor_channel_mask",
    "sensor_channel_ids",
    "perception_cost",
    "perception_cost_hyp",
    "perception_cost_charged",
    "sensor_radius",
    "observed_cells",
    "active_cell_fraction",
    "cell_channel_capacity",
    "spatial_perception_capacity",
    "sensor_radius_cost_hyp",
    "sensor_radius_cost_charged",
    "spatial_perception_cost",
    "sensor_channel_ablate",
    "sensor_channel_force_all",
    "sensor_channel_force_min",
    "sensor_radius_ablate",
    "sensor_radius_force_min",
    "sensor_radius_force_max",
    "sensor_radius_force_default",
    "dev_active",
    "age",
    "dev_stage",
    "maturity_age",
    "is_juvenile",
    "is_mature",
    "dev_factor",
    "dev_growth_progress",
    "dev_cost_hyp",
    "dev_cost_charged",
    "maturation_cost_hyp",
    "maturation_cost_charged",
    "dev_ablate",
    "dev_force_adult",
    "dev_force_juvenile",
    "dev_force_legacy",
    "dev_growth_disabled",
    "dev_cost_charging",
    "adult_from_birth",
    "memory_before_dev",
    "memory_after_dev",
    "sensor_radius_before_dev",
    "sensor_radius_after_dev",
    "sensor_channel_count_before_dev",
    "sensor_channel_count_after_dev",
    "prediction_lr_before_dev",
    "prediction_lr_after_dev",
    "introspection_enabled",
    "introspection_count",
    "introspection_potential",
    "introspection_fraction",
    "introspection_mask",
    "introspection_ids",
    "introspection_cost_hyp",
    "introspection_cost_charged",
    "introspection_cost_fraction",
    "introspection_ablate",
    "introspection_force_no",
    "introspection_force_minimal",
    "introspection_force_all",
    "introspection_force_default",
    "introspection_mask_neutral",
    "introspection_cost_charging",
    "introspection_policy_input",
    "self_regulation_enabled",
    "self_regulation_surface",
    "self_regulation_policy_output",
    "self_regulation_count",
    "self_regulation_potential",
    "self_regulation_fraction",
    "self_regulation_mask",
    "self_regulation_ids",
    "self_regulation_cost_hyp",
    "self_regulation_cost_charged",
    "self_regulation_cost_fraction",
    "self_regulation_ablate",
    "self_regulation_force_no",
    "self_regulation_force_all",
    "self_regulation_force_default",
    "self_regulation_mask_neutral",
    "self_regulation_freeze",
    "self_regulation_reset_each_tick",
    "self_regulation_decay_disabled",
    "self_regulation_cost_charging",
    "self_modification_active",
    "self_modification_enabled",
    "self_modification_policy_output",
    "self_modification_count",
    "self_modification_potential",
    "self_modification_fraction",
    "self_modification_mask",
    "self_modification_ids",
    "self_modification_cost_hyp",
    "self_modification_cost_charged",
    "self_modification_cost_fraction",
    "self_modification_ablate",
    "self_modification_force_no",
    "self_modification_force_all",
    "self_modification_force_default",
    "self_modification_mask_neutral",
    "self_modification_freeze",
    "self_modification_reset_each_tick",
    "self_modification_reset_on_reproduction",
    "self_modification_decay_disabled",
    "self_modification_cost_charging",
    "self_mod_delta_abs",
    "self_mod_delta_abs_p90",
    "self_mod_offset_abs",
    "self_mod_offset_abs_p90",
    "self_mod_clipped_fraction",
    "self_mod_saturation_fraction",
    "self_mod_event_count",
    "self_mod_neutral_fraction",
    "self_mod_reset_count",
    "self_mod_decay_count",
    "self_mod_lifetime_persistence",
    "self_mod_reversal_count",
    "modified_exploration_bias",
    "modified_action_temperature",
    "modified_drive_gain",
    "modified_reproduction_threshold",
    "modified_prediction_lr",
    "genome_editing_active",
    "directed_inheritance_enabled",
    "directed_inheritance_surface",
    "directed_inheritance_policy_output",
    "directed_inheritance_count",
    "directed_inheritance_potential",
    "directed_inheritance_fraction",
    "directed_inheritance_mask",
    "directed_inheritance_ids",
    "directed_inheritance_cost_hyp",
    "directed_inheritance_cost_charged",
    "directed_inheritance_cost_fraction",
    "directed_inheritance_ablate",
    "directed_inheritance_force_no",
    "directed_inheritance_force_all",
    "directed_inheritance_force_default",
    "directed_inheritance_mask_neutral",
    "directed_inheritance_freeze",
    "directed_inheritance_reset_birth",
    "directed_inheritance_reset_maturity",
    "directed_inheritance_cost_charging",
    "mutation_rate_control_active",
    "recombination_control_active",
    "inherited_self_modification_state_active",
    "directed_delta_abs",
    "directed_delta_abs_p90",
    "directed_payload_abs",
    "directed_payload_abs_p90",
    "directed_clipped_fraction",
    "directed_saturation_fraction",
    "directed_event_count",
    "directed_neutral_fraction",
    "directed_reset_count",
    "directed_payload_applied",
    "directed_payload_frozen",
    "offspring_developmental_expression_bias",
    "offspring_juvenile_cognition_scale_bias",
    "offspring_memory_expression_bias",
    "offspring_prediction_lr_bias",
    "offspring_exploration_bias_marker",
    "offspring_reproduction_threshold_bias",
    "self_reg_delta_abs",
    "self_reg_delta_abs_p90",
    "self_reg_offset_abs",
    "self_reg_offset_abs_p90",
    "self_reg_clipped_fraction",
    "self_reg_saturation_fraction",
    "self_reg_application_count",
    "self_reg_neutral_fraction",
    "self_reg_reset_count",
    "self_reg_decay_count",
    "effective_exploration_pressure",
    "effective_action_temperature",
    "effective_drive_gain",
    "effective_reproduction_threshold",
    "effective_prediction_lr_regulated",
    "internal_energy",
    "internal_health",
    "internal_drive_conflict",
    "internal_prediction_error",
    "internal_memory_saturation",
    "internal_cog_cost_burden",
    "internal_developmental_expression",
    "offspring",
    "energy",
    "alive",
    "lineage",
    "drive_conflict",
    "strategy",
    "memory_strategy",
    "memory_perception_strategy",
    "prediction_strategy",
    "developmental_strategy",
    "introspection_strategy",
    "self_regulation_strategy",
    "self_modification_strategy",
    "directed_inheritance_strategy",
)

# Dtype names match the generated flat_population_reporting_schema_v1.json
# contract.  The wider field set follows the exact primitive classes emitted
# by the reference _agent_records adapter.
FLAT_REPORTING_FIELD_DTYPES = MappingProxyType(
    {
        "agent_id": "int64",
        "capacity": "float64",
        "genetic_memory": "float64",
        "memory": "float64",
        "effective_memory": "float64",
        "memory_enabled": "bool",
        "policy_memory": "bool",
        "recurrent_state": "bool",
        "memory_capacity_fraction": "float64",
        "memory_ladder_index": "float64",
        "memory_cost_hyp": "float64",
        "memory_cost_charged": "float64",
        "memory_cost_fraction": "float64",
        "memory_ablate": "bool",
        "memory_force_disabled": "bool",
        "memory_force_min": "bool",
        "memory_force_max": "bool",
        "memory_force_default": "bool",
        "memory_reset_each_tick": "bool",
        "hidden": "float64",
        "pred_enabled": "bool",
        "pred_lr": "float64",
        "effective_pred_lr": "float64",
        "prediction_capacity": "float64",
        "mutation_rate": "float64",
        "action_temperature": "float64",
        "reproduction_threshold": "float64",
        "module_count": "float64",
        "connection_count": "float64",
        "c1_operator_noop_count": "float64",
        "c1_capacity_blocked_count": "float64",
        "c1_capacity_blocked_reason_counts_json": "string",
        "c1_op_count__bounded_rewire": "float64",
        "c1_op_count__duplicate_with_decay": "float64",
        "c1_op_count__edge_split_insertion": "float64",
        "c1_op_count__fusion_merge_compression": "float64",
        "c1_op_count__gate_drift": "float64",
        "c1_op_count__motif_duplication": "float64",
        "c1_op_count__promoter_expression_mutation": "float64",
        "b3_locality_dropped_connections": "float64",
        "prediction_state_available": "bool",
        "prediction_update_enabled": "bool",
        "pred_err": "float64",
        "prediction_cost_hyp": "float64",
        "prediction_cost_charged": "float64",
        "prediction_cost_fraction": "float64",
        "prediction_ablate": "bool",
        "prediction_force_disabled": "bool",
        "prediction_force_frozen": "bool",
        "prediction_force_zero_lr": "bool",
        "prediction_force_default": "bool",
        "prediction_reset_each_tick": "bool",
        "direct_prediction_pressure": "bool",
        "prediction_state_abs": "float64",
        "prediction_state_abs_p90": "float64",
        "prediction_state_delta": "float64",
        "prediction_state_delta_p90": "float64",
        "prediction_update_count": "float64",
        "prediction_nonzero_fraction": "float64",
        "prediction_effective_utilization": "float64",
        "prediction_saturation_proxy": "float64",
        "prediction_action_differentiation": "float64",
        "drive_gain": "float64",
        "drive_noise": "float64",
        "drive_inertia": "float64",
        "hunger": "float64",
        "fear": "float64",
        "pain": "float64",
        "fatigue": "float64",
        "cost": "float64",
        "memory_cost": "float64",
        "policy_memory_cost": "float64",
        "cost_frac": "float64",
        "memory_component": "float64",
        "prediction_component": "float64",
        "drive_component": "float64",
        "perception_component": "float64",
        "sensor_channel_count": "float64",
        "sensor_channel_potential": "float64",
        "sensor_channel_mask": "int64",
        "sensor_channel_ids": "tuple",
        "perception_cost": "float64",
        "perception_cost_hyp": "float64",
        "perception_cost_charged": "float64",
        "sensor_radius": "float64",
        "observed_cells": "float64",
        "active_cell_fraction": "float64",
        "cell_channel_capacity": "float64",
        "spatial_perception_capacity": "float64",
        "sensor_radius_cost_hyp": "float64",
        "sensor_radius_cost_charged": "float64",
        "spatial_perception_cost": "float64",
        "sensor_channel_ablate": "bool",
        "sensor_channel_force_all": "bool",
        "sensor_channel_force_min": "bool",
        "sensor_radius_ablate": "bool",
        "sensor_radius_force_min": "bool",
        "sensor_radius_force_max": "bool",
        "sensor_radius_force_default": "bool",
        "dev_active": "bool",
        "age": "float64",
        "dev_stage": "string",
        "maturity_age": "float64",
        "is_juvenile": "bool",
        "is_mature": "bool",
        "dev_factor": "float64",
        "dev_growth_progress": "float64",
        "dev_cost_hyp": "float64",
        "dev_cost_charged": "float64",
        "maturation_cost_hyp": "float64",
        "maturation_cost_charged": "float64",
        "dev_ablate": "bool",
        "dev_force_adult": "bool",
        "dev_force_juvenile": "bool",
        "dev_force_legacy": "bool",
        "dev_growth_disabled": "bool",
        "dev_cost_charging": "bool",
        "adult_from_birth": "bool",
        "memory_before_dev": "float64",
        "memory_after_dev": "float64",
        "sensor_radius_before_dev": "float64",
        "sensor_radius_after_dev": "float64",
        "sensor_channel_count_before_dev": "float64",
        "sensor_channel_count_after_dev": "float64",
        "prediction_lr_before_dev": "float64",
        "prediction_lr_after_dev": "float64",
        "introspection_enabled": "bool",
        "introspection_count": "float64",
        "introspection_potential": "float64",
        "introspection_fraction": "float64",
        "introspection_mask": "int64",
        "introspection_ids": "tuple",
        "introspection_cost_hyp": "float64",
        "introspection_cost_charged": "float64",
        "introspection_cost_fraction": "float64",
        "introspection_ablate": "bool",
        "introspection_force_no": "bool",
        "introspection_force_minimal": "bool",
        "introspection_force_all": "bool",
        "introspection_force_default": "bool",
        "introspection_mask_neutral": "bool",
        "introspection_cost_charging": "bool",
        "introspection_policy_input": "bool",
        "self_regulation_enabled": "bool",
        "self_regulation_surface": "bool",
        "self_regulation_policy_output": "bool",
        "self_regulation_count": "float64",
        "self_regulation_potential": "float64",
        "self_regulation_fraction": "float64",
        "self_regulation_mask": "int64",
        "self_regulation_ids": "tuple",
        "self_regulation_cost_hyp": "float64",
        "self_regulation_cost_charged": "float64",
        "self_regulation_cost_fraction": "float64",
        "self_regulation_ablate": "bool",
        "self_regulation_force_no": "bool",
        "self_regulation_force_all": "bool",
        "self_regulation_force_default": "bool",
        "self_regulation_mask_neutral": "bool",
        "self_regulation_freeze": "bool",
        "self_regulation_reset_each_tick": "bool",
        "self_regulation_decay_disabled": "bool",
        "self_regulation_cost_charging": "bool",
        "self_modification_active": "bool",
        "self_modification_enabled": "bool",
        "self_modification_policy_output": "bool",
        "self_modification_count": "float64",
        "self_modification_potential": "float64",
        "self_modification_fraction": "float64",
        "self_modification_mask": "int64",
        "self_modification_ids": "tuple",
        "self_modification_cost_hyp": "float64",
        "self_modification_cost_charged": "float64",
        "self_modification_cost_fraction": "float64",
        "self_modification_ablate": "bool",
        "self_modification_force_no": "bool",
        "self_modification_force_all": "bool",
        "self_modification_force_default": "bool",
        "self_modification_mask_neutral": "bool",
        "self_modification_freeze": "bool",
        "self_modification_reset_each_tick": "bool",
        "self_modification_reset_on_reproduction": "bool",
        "self_modification_decay_disabled": "bool",
        "self_modification_cost_charging": "bool",
        "self_mod_delta_abs": "float64",
        "self_mod_delta_abs_p90": "float64",
        "self_mod_offset_abs": "float64",
        "self_mod_offset_abs_p90": "float64",
        "self_mod_clipped_fraction": "float64",
        "self_mod_saturation_fraction": "float64",
        "self_mod_event_count": "float64",
        "self_mod_neutral_fraction": "float64",
        "self_mod_reset_count": "float64",
        "self_mod_decay_count": "float64",
        "self_mod_lifetime_persistence": "float64",
        "self_mod_reversal_count": "float64",
        "modified_exploration_bias": "float64",
        "modified_action_temperature": "float64",
        "modified_drive_gain": "float64",
        "modified_reproduction_threshold": "float64",
        "modified_prediction_lr": "float64",
        "genome_editing_active": "bool",
        "directed_inheritance_enabled": "bool",
        "directed_inheritance_surface": "bool",
        "directed_inheritance_policy_output": "bool",
        "directed_inheritance_count": "float64",
        "directed_inheritance_potential": "float64",
        "directed_inheritance_fraction": "float64",
        "directed_inheritance_mask": "int64",
        "directed_inheritance_ids": "tuple",
        "directed_inheritance_cost_hyp": "float64",
        "directed_inheritance_cost_charged": "float64",
        "directed_inheritance_cost_fraction": "float64",
        "directed_inheritance_ablate": "bool",
        "directed_inheritance_force_no": "bool",
        "directed_inheritance_force_all": "bool",
        "directed_inheritance_force_default": "bool",
        "directed_inheritance_mask_neutral": "bool",
        "directed_inheritance_freeze": "bool",
        "directed_inheritance_reset_birth": "bool",
        "directed_inheritance_reset_maturity": "bool",
        "directed_inheritance_cost_charging": "bool",
        "mutation_rate_control_active": "bool",
        "recombination_control_active": "bool",
        "inherited_self_modification_state_active": "bool",
        "directed_delta_abs": "float64",
        "directed_delta_abs_p90": "float64",
        "directed_payload_abs": "float64",
        "directed_payload_abs_p90": "float64",
        "directed_clipped_fraction": "float64",
        "directed_saturation_fraction": "float64",
        "directed_event_count": "float64",
        "directed_neutral_fraction": "float64",
        "directed_reset_count": "float64",
        "directed_payload_applied": "float64",
        "directed_payload_frozen": "float64",
        "offspring_developmental_expression_bias": "float64",
        "offspring_juvenile_cognition_scale_bias": "float64",
        "offspring_memory_expression_bias": "float64",
        "offspring_prediction_lr_bias": "float64",
        "offspring_exploration_bias_marker": "float64",
        "offspring_reproduction_threshold_bias": "float64",
        "self_reg_delta_abs": "float64",
        "self_reg_delta_abs_p90": "float64",
        "self_reg_offset_abs": "float64",
        "self_reg_offset_abs_p90": "float64",
        "self_reg_clipped_fraction": "float64",
        "self_reg_saturation_fraction": "float64",
        "self_reg_application_count": "float64",
        "self_reg_neutral_fraction": "float64",
        "self_reg_reset_count": "float64",
        "self_reg_decay_count": "float64",
        "effective_exploration_pressure": "float64",
        "effective_action_temperature": "float64",
        "effective_drive_gain": "float64",
        "effective_reproduction_threshold": "float64",
        "effective_prediction_lr_regulated": "float64",
        "internal_energy": "float64",
        "internal_health": "float64",
        "internal_drive_conflict": "float64",
        "internal_prediction_error": "float64",
        "internal_memory_saturation": "float64",
        "internal_cog_cost_burden": "float64",
        "internal_developmental_expression": "float64",
        "offspring": "float64",
        "energy": "float64",
        "alive": "bool",
        "lineage": "int64",
        "drive_conflict": "float64",
        "strategy": "tuple",
        "memory_strategy": "tuple",
        "memory_perception_strategy": "tuple",
        "prediction_strategy": "tuple",
        "developmental_strategy": "tuple",
        "introspection_strategy": "tuple",
        "self_regulation_strategy": "tuple",
        "self_modification_strategy": "tuple",
        "directed_inheritance_strategy": "tuple",
    }
)

if tuple(FLAT_REPORTING_FIELD_DTYPES) != FLAT_REPORTING_REQUIRED_FIELDS:
    raise RuntimeError("Flat V1 dtype contract field order does not match required fields")

_REQUIRED_FIELD_SET = frozenset(FLAT_REPORTING_REQUIRED_FIELDS)
_FORBIDDEN_LIVE_FIELDS = ("agent", "phenotype", "ledger")
_MUTABLE_VALUE_TYPES = (bytearray, dict, list, set)


@dataclass(frozen=True, slots=True, init=False)
class FlatPopulationReportingRecordsV1:
    """Validated, deeply immutable population and selected-parent rows."""

    schema_version: str
    rows: tuple[Mapping[str, object], ...] = field(hash=False)
    selected_parent_rows: tuple[Mapping[str, object], ...] = field(hash=False)

    def __init__(
        self,
        rows: Iterable[Mapping[str, Any]] = (),
        selected_parent_rows: Iterable[Mapping[str, Any]] = (),
        *,
        schema_version: str = FLAT_POPULATION_REPORTING_SCHEMA_VERSION,
    ) -> None:
        if schema_version != FLAT_POPULATION_REPORTING_SCHEMA_VERSION:
            raise ValueError(
                "schema_version: expected "
                f"{FLAT_POPULATION_REPORTING_SCHEMA_VERSION!r}, got {schema_version!r}"
            )
        frozen_rows, frozen_parent_rows = _validate_and_freeze_rows(
            rows,
            selected_parent_rows,
        )
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "rows", frozen_rows)
        object.__setattr__(self, "selected_parent_rows", frozen_parent_rows)


def validate_flat_population_reporting_records(
    rows: Iterable[Mapping[str, Any]] | FlatPopulationReportingRecordsV1 = (),
    selected_parent_rows: Iterable[Mapping[str, Any]] = (),
    *,
    schema_version: str = FLAT_POPULATION_REPORTING_SCHEMA_VERSION,
) -> FlatPopulationReportingRecordsV1:
    """Validate and return an immutable V1 record collection.

    Passing an existing V1 collection is a no-op when no replacement
    selected-parent rows are supplied.
    """

    if isinstance(rows, FlatPopulationReportingRecordsV1):
        supplied_parent_rows = tuple(selected_parent_rows)
        if supplied_parent_rows:
            raise ValueError(
                "selected_parent_rows: cannot replace rows on an existing "
                "FlatPopulationReportingRecordsV1"
            )
        if schema_version != rows.schema_version:
            raise ValueError(
                f"schema_version: expected {rows.schema_version!r}, got {schema_version!r}"
            )
        return rows
    return FlatPopulationReportingRecordsV1(
        rows,
        selected_parent_rows,
        schema_version=schema_version,
    )


def _validate_and_freeze_rows(
    rows: Iterable[Mapping[str, Any]],
    selected_parent_rows: Iterable[Mapping[str, Any]],
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
]:
    copied_rows = _copy_row_mappings(rows, context="rows")
    copied_parent_rows = _copy_row_mappings(
        selected_parent_rows,
        context="selected_parent_rows",
    )

    frozen_rows, agent_ids = _freeze_collection(
        copied_rows,
        context="rows",
    )
    frozen_parent_rows, selected_parent_ids = _freeze_collection(
        copied_parent_rows,
        context="selected_parent_rows",
    )
    _validate_ids(agent_ids, context="rows")
    _validate_ids(selected_parent_ids, context="selected_parent_rows")

    population_rows_by_id = dict(zip(agent_ids, frozen_rows, strict=True))
    for index, (agent_id, parent_row) in enumerate(
        zip(selected_parent_ids, frozen_parent_rows, strict=True)
    ):
        population_row = population_rows_by_id.get(agent_id)
        if population_row is None:
            raise ValueError(
                f"selected_parent_rows[{index}].agent_id: {agent_id} is not present in rows"
            )
        if parent_row != population_row:
            raise ValueError(
                f"selected_parent_rows[{index}]: values differ from rows agent_id {agent_id}"
            )
    return frozen_rows, frozen_parent_rows


def _copy_row_mappings(
    source: Iterable[Mapping[str, Any]],
    *,
    context: str,
) -> tuple[dict[str, Any], ...]:
    if isinstance(source, (Mapping, str, bytes, bytearray)):
        raise ValueError(f"{context}: expected an iterable of mappings")
    try:
        raw_rows = tuple(source)
    except TypeError as exc:
        raise ValueError(f"{context}: expected an iterable of mappings") from exc

    copied: list[dict[str, Any]] = []
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"{context}[{index}]: expected a mapping")
        row = dict(raw_row)
        for field_name in row:
            if type(field_name) is not str:
                raise ValueError(
                    f"{context}[{index}]: field names must be primitive strings; "
                    f"got {type(field_name).__name__}"
                )
        copied.append(row)
    return tuple(copied)


def _freeze_collection(
    rows: tuple[dict[str, Any], ...],
    *,
    context: str,
) -> tuple[tuple[Mapping[str, object], ...], tuple[int, ...]]:
    frozen_rows: list[Mapping[str, object]] = []
    agent_ids: list[int] = []

    for index, row in enumerate(rows):
        row_context = f"{context}[{index}]"
        for forbidden in _FORBIDDEN_LIVE_FIELDS:
            if forbidden in row:
                raise ValueError(f"{row_context}.{forbidden}: live field is forbidden")
        for required in FLAT_REPORTING_REQUIRED_FIELDS:
            if required not in row:
                raise ValueError(f"{row_context}.{required}: missing required field")

        actual_fields = frozenset(row)
        if actual_fields != _REQUIRED_FIELD_SET:
            missing = next(
                (name for name in FLAT_REPORTING_REQUIRED_FIELDS if name not in actual_fields),
                None,
            )
            if missing is not None:
                raise ValueError(
                    f"{row_context}.{missing}: field set differs; field is missing"
                )
            unexpected = min(actual_fields - _REQUIRED_FIELD_SET)
            raise ValueError(
                f"{row_context}.{unexpected}: field set differs; field is unexpected"
            )

        agent_id = row["agent_id"]
        if type(agent_id) is not int:
            raise ValueError(f"{row_context}.agent_id: expected a nonnegative integer")
        if agent_id < 0:
            raise ValueError(f"{row_context}.agent_id: must be nonnegative")
        agent_ids.append(agent_id)

        frozen: dict[str, object] = {}
        for field_name in FLAT_REPORTING_REQUIRED_FIELDS:
            frozen[field_name] = _freeze_field_value(
                row[field_name],
                dtype=FLAT_REPORTING_FIELD_DTYPES[field_name],
                context=f"{row_context}.{field_name}",
            )
        frozen_rows.append(MappingProxyType(frozen))
    return tuple(frozen_rows), tuple(agent_ids)


def _freeze_field_value(value: Any, *, dtype: str, context: str) -> object:
    expected_type = {
        "bool": bool,
        "float64": float,
        "int64": int,
        "string": str,
        "tuple": tuple,
    }[dtype]
    if type(value) is not expected_type:
        raise ValueError(
            f"{context}: expected {dtype}, got {type(value).__name__}"
        )
    if expected_type is float and not math.isfinite(value):
        raise ValueError(f"{context}: numeric value must be finite")
    if expected_type is tuple:
        return tuple(
            _freeze_primitive(item, context=f"{context}[{index}]")
            for index, item in enumerate(value)
        )
    return value


def _freeze_primitive(value: Any, *, context: str) -> object:
    value_type = type(value)
    if value is None:
        raise ValueError(f"{context}: None is not a supported primitive")
    if value_type in (bool, int, str):
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{context}: numeric value must be finite")
        return value
    if value_type is tuple:
        return tuple(
            _freeze_primitive(item, context=f"{context}[{index}]")
            for index, item in enumerate(value)
        )
    if _is_rng_like(value):
        raise ValueError(f"{context}: RNG-like value is forbidden")
    if isinstance(value, _MUTABLE_VALUE_TYPES) or isinstance(value, Mapping):
        raise ValueError(
            f"{context}: mutable value type {value_type.__name__} is forbidden"
        )
    raise ValueError(
        f"{context}: value type {value_type.__name__} is not a supported primitive"
    )


def _is_rng_like(value: Any) -> bool:
    value_type = type(value)
    type_name = value_type.__name__.lower()
    module_name = value_type.__module__.lower()
    return (
        type_name in {"random", "randomstate", "bitgenerator"}
        or type_name.endswith("rng")
        or (
            module_name.startswith(("random", "numpy.random", "torch"))
            and ("random" in type_name or "generator" in type_name)
        )
    )


def _validate_ids(agent_ids: tuple[int, ...], *, context: str) -> None:
    seen: set[int] = set()
    for index, agent_id in enumerate(agent_ids):
        if agent_id in seen:
            raise ValueError(f"{context}[{index}].agent_id: duplicate value {agent_id}")
        seen.add(agent_id)
    for index in range(1, len(agent_ids)):
        if agent_ids[index] <= agent_ids[index - 1]:
            raise ValueError(
                f"{context}[{index}].agent_id: values must be in strictly increasing order"
            )


__all__ = (
    "FLAT_POPULATION_REPORTING_SCHEMA_VERSION",
    "FLAT_REPORTING_FIELD_DTYPES",
    "FLAT_REPORTING_REQUIRED_FIELDS",
    "FlatPopulationReportingRecordsV1",
    "validate_flat_population_reporting_records",
)
