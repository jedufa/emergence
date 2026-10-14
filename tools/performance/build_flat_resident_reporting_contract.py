#!/usr/bin/env python3
from __future__ import annotations

"""Build the Flat resident reporting inventory without importing engine code."""

import argparse
import ast
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COGNITIVE_STATUS_SOURCE = Path("engine/simulation/metrics_pkg/cognitive_status.py")
METRICS_LOGGER_SOURCE = Path("engine/simulation/metrics_pkg/metrics_logger.py")
METRICS_ROW_SOURCE = Path("engine/simulation/metrics_pkg/rows.py")
STATS_SOURCE = Path("engine/simulation/metrics_pkg/stats.py")
NATIVE_DISCOVERY_SOURCES = (
    Path("engine/native_runtime/public_metrics_overlay.py"),
    Path("engine/native_runtime/checkpoint_digests.py"),
    Path("engine/perf/cxxbackbone10_resident_state_owner.py"),
)

CONTRACT_FILENAME = "flat_resident_reporting_66_column_contract.json"
SCHEMA_FILENAME = "flat_population_reporting_schema_v1.json"

# This order is explicit by design. It is the current fairness mismatch union
# after removing the ten chemistry columns covered by the chemistry repair.
FAIRNESS_COLUMNS = (
    "cog_capacity_mean",
    "cog_capacity_p50",
    "cog_capacity_p90",
    "cog_capacity_max",
    "cog_capacity_min",
    "cog_capacity_std",
    "cog_capacity_alive_mean",
    "cog_capacity_parent_mean",
    "cog_capacity_survivor_mean",
    "mem_evo_population_memory_dim_histogram",
    "mem_evo_selected_parent_memory_dim_histogram",
    "mem_evo_descendant_memory_dim_histogram",
    "mem_evo_lineage_memory_dim_summary_json",
    "mem_evo_effective_memory_dim_histogram",
    "prediction_lr_mean",
    "prediction_lr_p50",
    "prediction_lr_p90",
    "prediction_lr_max",
    "prediction_lr_std",
    "drive_gain_mean",
    "drive_gain_p90",
    "drive_noise_mean",
    "drive_inertia_mean",
    "drive_conflict_mean",
    "drive_conflict_p90",
    "hunger_sensitivity_mean",
    "fear_sensitivity_mean",
    "pain_sensitivity_mean",
    "fatigue_sensitivity_mean",
    "parent_cog_capacity_mean",
    "alive_cog_capacity_mean",
    "reproduced_cog_capacity_mean",
    "non_reproduced_cog_capacity_mean",
    "cognitive_selection_delta_parent_vs_alive",
    "drive_regulation_selection_delta_parent_vs_alive",
    "cognitive_trait_diversity",
    "cognitive_capacity_entropy",
    "drive_profile_diversity",
    "cognitive_strategy_count_approx",
    "juvenile_cog_capacity_mean",
    "adult_cog_capacity_mean",
    "elder_cog_capacity_mean",
    "cognition_by_age_slope",
    "juvenile_fraction",
    "adult_fraction",
    "mature_fraction",
    "adult_sensor_radius_mean",
    "adult_sensor_channel_count_mean",
    "reproduced_maturity_age_mean",
    "adult_survival_rate",
    "adult_reproduction_attempt_rate",
    "high_adult_cognition_low_developmental_cost_fraction",
    "adult_cognition_from_birth_fraction",
    "juvenile_high_cognition_fraction",
    "internal_energy_signal_mean",
    "internal_health_signal_mean",
    "internal_drive_conflict_signal_mean",
    "full_cognitive_strategy_count_approx",
    "lineage_cog_capacity_mean",
    "top_lineage_cog_capacity",
    "lineage_cognitive_diversity",
    "high_cognition_low_cost_fraction",
    "cognition_capacity_in_flat_world_baseline",
    "cog_capacity_alive_end_mean",
    "cog_capacity_drive_component_mean",
    "sensor_radius_cost_hypothetical_total",
)

COHORT_RULES: dict[str, dict[str, Any]] = {
    "all_live": {
        "collection": "live_records",
        "definition": "all records supplied as the generation live population",
        "filter": "none",
        "ordering": "agent_id ascending",
        "membership_primitive_keys": ("agent_id",),
        "native_membership": "resident alive_mask joined to slot_to_agent_id",
    },
    "selected_parents": {
        "collection": "selected_parent_records",
        "definition": "the selected-parent snapshot supplied separately to the reducer",
        "filter": "record is present in selected_parent_records",
        "ordering": "agent_id ascending",
        "membership_primitive_keys": ("agent_id",),
        "native_membership": "native behavior row kept_next > 0 joined by agent_id",
    },
    "live_and_selected_parents": {
        "collection": "live_records and selected_parent_records",
        "definition": "all live records compared with the selected-parent snapshot",
        "filter": "none within either collection",
        "ordering": "agent_id ascending in each collection",
        "membership_primitive_keys": ("agent_id",),
        "native_membership": "resident alive rows plus behavior kept_next > 0 join",
    },
    "reproduced_live": {
        "collection": "live_records",
        "definition": "live records with offspring > 0",
        "filter": "float(record['offspring']) > 0.0",
        "ordering": "agent_id ascending",
        "membership_primitive_keys": ("agent_id", "offspring"),
        "native_membership": "resident offspring_count or behavior offspring > 0",
    },
    "non_reproduced_live": {
        "collection": "live_records",
        "definition": "live records with offspring <= 0",
        "filter": "float(record['offspring']) <= 0.0",
        "ordering": "agent_id ascending",
        "membership_primitive_keys": ("agent_id", "offspring"),
        "native_membership": "resident offspring_count or behavior offspring <= 0",
    },
    "age_lt_20": {
        "collection": "live_records",
        "definition": "fixed reporting age band below 20 ticks",
        "filter": "float(record['age']) < 20.0",
        "ordering": "agent_id ascending",
        "membership_primitive_keys": ("agent_id", "age"),
        "native_membership": "resident survival_steps < 20",
    },
    "age_20_to_99": {
        "collection": "live_records",
        "definition": "fixed reporting age band from 20 inclusive to 100 exclusive",
        "filter": "20.0 <= float(record['age']) < 100.0",
        "ordering": "agent_id ascending",
        "membership_primitive_keys": ("agent_id", "age"),
        "native_membership": "20 <= resident survival_steps < 100",
    },
    "age_ge_100": {
        "collection": "live_records",
        "definition": "fixed reporting age band at least 100 ticks",
        "filter": "float(record['age']) >= 100.0",
        "ordering": "agent_id ascending",
        "membership_primitive_keys": ("agent_id", "age"),
        "native_membership": "resident survival_steps >= 100",
    },
    "developmental_juvenile": {
        "collection": "live_records",
        "definition": "records whose shared phenotype marks is_juvenile",
        "filter": "bool(record['is_juvenile'])",
        "ordering": "agent_id ascending",
        "membership_primitive_keys": ("agent_id", "is_juvenile"),
        "native_membership": "shared phenotype projection from resident age/genome/config",
    },
    "developmental_mature": {
        "collection": "live_records",
        "definition": "records whose shared phenotype marks is_mature",
        "filter": "bool(record['is_mature'])",
        "ordering": "agent_id ascending",
        "membership_primitive_keys": ("agent_id", "is_mature"),
        "native_membership": "shared phenotype projection from resident age/genome/config",
    },
    "lineage_grouped_live": {
        "collection": "live_records",
        "definition": "all live records grouped by integer lineage",
        "filter": "none",
        "ordering": "agent_id ascending within lineage; lineage id ascending across groups",
        "membership_primitive_keys": ("agent_id", "lineage"),
        "native_membership": "resident lineage_founder_id joined to live slots",
    },
}

COLUMN_COHORT_OVERRIDES = {
    "cog_capacity_parent_mean": "selected_parents",
    "mem_evo_selected_parent_memory_dim_histogram": "selected_parents",
    "parent_cog_capacity_mean": "selected_parents",
    "cognitive_selection_delta_parent_vs_alive": "live_and_selected_parents",
    "drive_regulation_selection_delta_parent_vs_alive": "live_and_selected_parents",
    "reproduced_cog_capacity_mean": "reproduced_live",
    "reproduced_maturity_age_mean": "reproduced_live",
    "non_reproduced_cog_capacity_mean": "non_reproduced_live",
    "juvenile_cog_capacity_mean": "age_lt_20",
    "adult_cog_capacity_mean": "age_20_to_99",
    "elder_cog_capacity_mean": "age_ge_100",
    "juvenile_fraction": "developmental_juvenile",
    "juvenile_high_cognition_fraction": "developmental_juvenile",
    "adult_fraction": "developmental_mature",
    "mature_fraction": "developmental_mature",
    "adult_sensor_radius_mean": "developmental_mature",
    "adult_sensor_channel_count_mean": "developmental_mature",
    "adult_survival_rate": "developmental_mature",
    "adult_reproduction_attempt_rate": "developmental_mature",
    "high_adult_cognition_low_developmental_cost_fraction": "developmental_mature",
    "mem_evo_lineage_memory_dim_summary_json": "lineage_grouped_live",
    "lineage_cog_capacity_mean": "lineage_grouped_live",
    "top_lineage_cog_capacity": "lineage_grouped_live",
    "lineage_cognitive_diversity": "lineage_grouped_live",
}

REDUCER_GROUPS: dict[str, tuple[str, ...]] = {
    "mean": (
        "cog_capacity_mean",
        "prediction_lr_mean",
        "drive_gain_mean",
        "drive_noise_mean",
        "drive_inertia_mean",
        "drive_conflict_mean",
        "hunger_sensitivity_mean",
        "fear_sensitivity_mean",
        "pain_sensitivity_mean",
        "fatigue_sensitivity_mean",
        "alive_cog_capacity_mean",
        "juvenile_cog_capacity_mean",
        "adult_cog_capacity_mean",
        "elder_cog_capacity_mean",
        "adult_sensor_radius_mean",
        "adult_sensor_channel_count_mean",
        "internal_energy_signal_mean",
        "internal_health_signal_mean",
        "internal_drive_conflict_signal_mean",
        "lineage_cog_capacity_mean",
        "cognition_capacity_in_flat_world_baseline",
        "cog_capacity_drive_component_mean",
    ),
    "mean_or_zero": (
        "cog_capacity_parent_mean",
        "parent_cog_capacity_mean",
        "reproduced_cog_capacity_mean",
        "non_reproduced_cog_capacity_mean",
        "reproduced_maturity_age_mean",
    ),
    "linear_interpolated_percentile_p50": (
        "cog_capacity_p50",
        "prediction_lr_p50",
    ),
    "linear_interpolated_percentile_p90": (
        "cog_capacity_p90",
        "prediction_lr_p90",
        "drive_gain_p90",
        "drive_conflict_p90",
    ),
    "maximum_or_zero": (
        "cog_capacity_max",
        "prediction_lr_max",
        "top_lineage_cog_capacity",
    ),
    "minimum": ("cog_capacity_min",),
    "population_standard_deviation": (
        "cog_capacity_std",
        "prediction_lr_std",
    ),
    "alias": (
        "cog_capacity_alive_mean",
        "cog_capacity_survivor_mean",
        "mem_evo_descendant_memory_dim_histogram",
        "mature_fraction",
        "cog_capacity_alive_end_mean",
    ),
    "canonical_memory_histogram_json": (
        "mem_evo_population_memory_dim_histogram",
        "mem_evo_selected_parent_memory_dim_histogram",
        "mem_evo_effective_memory_dim_histogram",
    ),
    "canonical_lineage_memory_json": ("mem_evo_lineage_memory_dim_summary_json",),
    "difference_of_means": (
        "cognitive_selection_delta_parent_vs_alive",
        "drive_regulation_selection_delta_parent_vs_alive",
    ),
    "sum_of_rounded_distinct_counts": ("cognitive_trait_diversity",),
    "base2_entropy": (
        "cognitive_capacity_entropy",
        "lineage_cognitive_diversity",
    ),
    "rounded_distinct_count": ("drive_profile_diversity",),
    "distinct_strategy_count": (
        "cognitive_strategy_count_approx",
        "full_cognitive_strategy_count_approx",
    ),
    "least_squares_slope": ("cognition_by_age_slope",),
    "filtered_fraction": (
        "juvenile_fraction",
        "adult_fraction",
        "adult_cognition_from_birth_fraction",
        "juvenile_high_cognition_fraction",
        "high_adult_cognition_low_developmental_cost_fraction",
        "high_cognition_low_cost_fraction",
    ),
    "cohort_presence": ("adult_survival_rate",),
    "reproducer_fraction": ("adult_reproduction_attempt_rate",),
    "sum": ("sensor_radius_cost_hypothetical_total",),
}


@dataclass(frozen=True)
class NativeHint:
    kind: str
    candidate_keys: tuple[str, ...]
    note: str


NATIVE_FIELD_HINTS: dict[str, NativeHint] = {
    "agent_id": NativeHint("direct_tensor", ("slot_to_agent_id",), "stable resident slot identity"),
    "capacity": NativeHint(
        "shared_derived",
        ("genome_memory_dim", "genome_pred_lr", "genome_drive_logit_gain", "genome_sensor_channel_mask", "genome_sensor_radius"),
        "derive with the shared phenotype calculation from resident genome/config inputs",
    ),
    "genetic_memory": NativeHint("direct_tensor", ("genome_memory_dim",), "resident genetic memory dimension"),
    "memory": NativeHint("shared_derived", ("genome_memory_dim", "survival_steps"), "shared developmental phenotype projection"),
    "effective_memory": NativeHint("direct_or_shared_derived", ("policy_mem_dim", "genome_memory_dim"), "active/effective resident policy memory"),
    "recurrent_state": NativeHint("direct_tensor", ("policy_use_gru",), "resident recurrent policy flag"),
    "pred_lr": NativeHint("direct_tensor", ("genome_pred_lr",), "resident genetic prediction learning rate"),
    "effective_pred_lr": NativeHint("shared_derived", ("genome_pred_lr", "survival_steps"), "shared ablation/developmental phenotype projection"),
    "drive_gain": NativeHint("direct_tensor", ("genome_drive_logit_gain",), "resident genome drive gain"),
    "drive_noise": NativeHint("direct_tensor", ("genome_drive_noise",), "resident genome drive noise"),
    "drive_inertia": NativeHint("direct_tensor", ("genome_drive_inertia",), "resident genome drive inertia"),
    "drive_conflict": NativeHint("shared_derived", ("drives",), "abs(hunger-fatigue)+abs(pain-fear) from resident drives"),
    "hunger": NativeHint("direct_tensor", ("genome_drive_sp_hunger",), "resident hunger sensitivity"),
    "fear": NativeHint("direct_tensor", ("genome_drive_sp_fear",), "resident fear sensitivity"),
    "pain": NativeHint("direct_tensor", ("genome_drive_sp_pain",), "resident pain sensitivity"),
    "fatigue": NativeHint("direct_tensor", ("genome_drive_sp_fatigue",), "resident fatigue sensitivity"),
    "strategy": NativeHint(
        "shared_derived",
        ("genome_memory_dim", "policy_use_gru", "genome_pred_lr", "genome_drive_logit_gain", "genome_sensor_channel_mask", "genome_sensor_radius"),
        "canonical tuple from shared phenotype fields",
    ),
    "age": NativeHint("direct_tensor", ("survival_steps",), "resident survival age"),
    "is_juvenile": NativeHint("shared_derived", ("survival_steps",), "shared phenotype developmental classification"),
    "is_mature": NativeHint("shared_derived", ("survival_steps",), "shared phenotype developmental classification"),
    "sensor_radius": NativeHint("shared_derived", ("genome_sensor_radius", "survival_steps"), "active radius after shared developmental projection"),
    "sensor_channel_count": NativeHint("shared_derived", ("genome_sensor_channel_mask", "survival_steps"), "bit count after shared developmental projection"),
    "maturity_age": NativeHint(
        "shared_config_metadata",
        ("developmental_maturity_age",),
        "shared phenotype maturity_age derives from sim_cfg.developmental_maturity_age, "
        "which the resident import projection carries in metadata",
    ),
    "offspring": NativeHint("behavior_or_tensor", ("offspring", "offspring_count"), "native behavior row or resident lifetime offspring count"),
    "dev_cost_hyp": NativeHint("shared_derived", ("survival_steps",), "shared cognitive budget ledger"),
    "maturation_cost_hyp": NativeHint("shared_derived", ("survival_steps",), "shared cognitive budget ledger"),
    "adult_from_birth": NativeHint("shared_derived", ("survival_steps",), "shared developmental phenotype projection"),
    "internal_energy": NativeHint("shared_derived", ("energy",), "shared introspection channel normalization"),
    "internal_health": NativeHint("shared_derived", ("health",), "shared introspection channel normalization"),
    "internal_drive_conflict": NativeHint("shared_derived", ("drives",), "shared introspection drive-conflict channel"),
    "introspection_strategy": NativeHint(
        "shared_derived",
        ("genome_introspection_channel_mask", "genome_memory_dim", "genome_sensor_radius", "genome_pred_lr", "survival_steps"),
        "canonical tuple from shared phenotype fields",
    ),
    "self_regulation_strategy": NativeHint(
        "shared_derived",
        ("genome_self_regulation_control_mask", "genome_introspection_channel_mask", "genome_memory_dim", "genome_sensor_radius", "genome_pred_lr", "survival_steps"),
        "canonical tuple from shared phenotype fields",
    ),
    "self_modification_strategy": NativeHint(
        "shared_derived",
        ("genome_self_modification_target_mask", "genome_self_regulation_control_mask", "genome_introspection_channel_mask", "genome_memory_dim", "genome_sensor_radius", "genome_pred_lr", "survival_steps"),
        "canonical tuple from shared phenotype fields",
    ),
    "directed_inheritance_strategy": NativeHint(
        "shared_derived",
        ("genome_directed_inheritance_channel_mask", "genome_self_modification_target_mask", "genome_self_regulation_control_mask", "genome_introspection_channel_mask", "genome_memory_dim", "genome_sensor_radius", "genome_pred_lr", "survival_steps"),
        "canonical tuple from shared phenotype fields",
    ),
    "lineage": NativeHint("direct_tensor", ("lineage_founder_id",), "resident Flat founder lineage identity"),
    "cost": NativeHint(
        "shared_derived",
        ("genome_memory_dim", "genome_pred_lr", "genome_sensor_channel_mask", "genome_sensor_radius", "survival_steps"),
        "shared cognitive budget ledger",
    ),
    "drive_component": NativeHint(
        "shared_derived",
        ("genome_drive_logit_gain", "genome_drive_noise", "genome_drive_inertia"),
        "shared phenotype capacity component",
    ),
    "sensor_radius_cost_hyp": NativeHint("shared_derived", ("genome_sensor_radius", "survival_steps"), "shared cognitive budget ledger"),
}

RECORD_ITEM_NAMES = frozenset({"r", "rec", "record"})


class ContractError(RuntimeError):
    pass


def _read_source(relative_path: Path) -> tuple[str, ast.Module]:
    path = PROJECT_ROOT / relative_path
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"cannot read source {relative_path}: {exc}") from exc
    try:
        return source, ast.parse(source, filename=str(relative_path))
    except SyntaxError as exc:
        raise ContractError(f"cannot parse source {relative_path}: {exc}") from exc


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            if isinstance(node, ast.FunctionDef):
                return node
    raise ContractError(f"missing function {name}")


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise ContractError(f"missing class {name}")


def _literal_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _out_column(target: ast.AST) -> str | None:
    if not isinstance(target, ast.Subscript):
        return None
    if not isinstance(target.value, ast.Name) or target.value.id != "out":
        return None
    return _literal_string(target.slice)


def _assignment_targets(node: ast.AST) -> Iterable[tuple[ast.AST, ast.AST]]:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            yield target, node.value
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        yield node.target, node.value


def _reducer_model(tree: ast.Module) -> tuple[
    ast.FunctionDef,
    dict[str, list[tuple[int, ast.AST]]],
    dict[str, list[tuple[int, ast.AST]]],
]:
    reducer = _function(tree, "compute_cognitive_status_metrics_from_records")
    output_assignments: dict[str, list[tuple[int, ast.AST]]] = defaultdict(list)
    local_assignments: dict[str, list[tuple[int, ast.AST]]] = defaultdict(list)
    for node in ast.walk(reducer):
        for target, value in _assignment_targets(node):
            column = _out_column(target)
            if column is not None:
                output_assignments[column].append((node.lineno, value))
            elif isinstance(target, ast.Name):
                local_assignments[target.id].append((node.lineno, value))
    for assignments in output_assignments.values():
        assignments.sort(key=lambda item: item[0])
    for assignments in local_assignments.values():
        assignments.sort(key=lambda item: item[0])
    return reducer, dict(output_assignments), dict(local_assignments)


def _direct_record_keys(node: ast.AST) -> set[str]:
    keys: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name) and child.func.id == "_vals" and len(child.args) >= 2:
                key = _literal_string(child.args[1])
                if key is not None:
                    keys.add(key)
            if isinstance(child.func, ast.Attribute) and child.func.attr == "get" and child.args:
                if isinstance(child.func.value, ast.Name) and child.func.value.id in RECORD_ITEM_NAMES:
                    key = _literal_string(child.args[0])
                    if key is not None:
                        keys.add(key)
        elif isinstance(child, ast.Subscript):
            if isinstance(child.value, ast.Name) and child.value.id in RECORD_ITEM_NAMES:
                key = _literal_string(child.slice)
                if key is not None:
                    keys.add(key)
    return keys


def _name_loads(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _out_dependencies(node: ast.AST) -> set[str]:
    dependencies: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name) and child.value.id == "out":
            key = _literal_string(child.slice)
            if key is not None:
                dependencies.add(key)
    return dependencies


def _latest_before(assignments: Sequence[tuple[int, ast.AST]], line: int) -> tuple[int, ast.AST] | None:
    eligible = [assignment for assignment in assignments if assignment[0] <= line]
    return eligible[-1] if eligible else None


def _primitive_keys_for_column(
    column: str,
    *,
    output_assignments: Mapping[str, Sequence[tuple[int, ast.AST]]],
    local_assignments: Mapping[str, Sequence[tuple[int, ast.AST]]],
    helper_functions: Mapping[str, ast.FunctionDef],
    seen_columns: frozenset[str] = frozenset(),
) -> set[str]:
    if column in seen_columns:
        return set()
    assignments = output_assignments.get(column, ())
    if not assignments:
        raise ContractError(f"fairness column is not assigned by canonical reducer: {column}")
    line, expression = assignments[-1]
    seen_names: set[tuple[str, int]] = set()

    def resolve(node: ast.AST, before_line: int) -> set[str]:
        keys = _direct_record_keys(node)
        for dependency in _out_dependencies(node):
            keys.update(
                _primitive_keys_for_column(
                    dependency,
                    output_assignments=output_assignments,
                    local_assignments=local_assignments,
                    helper_functions=helper_functions,
                    seen_columns=seen_columns | {column},
                )
            )
        for child in ast.walk(node):
            if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Name):
                continue
            helper = helper_functions.get(child.func.id)
            if helper is not None:
                keys.update(_direct_record_keys(helper))
        for name in _name_loads(node):
            assignment = _latest_before(local_assignments.get(name, ()), before_line)
            if assignment is None:
                continue
            assignment_line, value = assignment
            marker = (name, assignment_line)
            if marker in seen_names:
                continue
            seen_names.add(marker)
            keys.update(resolve(value, assignment_line))
        return keys

    return resolve(expression, line)


def _record_fields(tree: ast.Module) -> dict[str, tuple[int, ast.AST]]:
    adapter = _function(tree, "_agent_records")
    dictionaries: list[ast.Dict] = []
    for node in ast.walk(adapter):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "append" and node.args and isinstance(node.args[0], ast.Dict):
            dictionaries.append(node.args[0])
    if len(dictionaries) != 1:
        raise ContractError(f"expected one canonical record dictionary, found {len(dictionaries)}")
    fields: dict[str, tuple[int, ast.AST]] = {}
    for key_node, value_node in zip(dictionaries[0].keys, dictionaries[0].values):
        key = _literal_string(key_node)
        if key is None:
            raise ContractError("canonical record dictionary contains a dynamic key")
        if key in fields:
            raise ContractError(f"canonical record dictionary repeats key: {key}")
        fields[key] = (value_node.lineno, value_node)
    return fields


def _metric_defaults(tree: ast.Module) -> dict[str, Any]:
    function = _function(tree, "cognitive_status_metric_defaults")
    values: dict[str, Any] = {}
    for node in function.body:
        for target, value in _assignment_targets(node):
            if not isinstance(target, ast.Name) or target.id not in {"keys", "json_keys"}:
                continue
            try:
                values[target.id] = ast.literal_eval(value)
            except (ValueError, TypeError) as exc:
                raise ContractError(f"cannot structurally parse metric defaults {target.id}") from exc
    if "keys" not in values or "json_keys" not in values:
        raise ContractError("metric defaults no longer expose literal keys/json_keys")
    keys = list(values["keys"])
    json_keys = set(values["json_keys"])
    return {key: "{}" if key in json_keys else 0.0 for key in keys}


def _metrics_row_fields(tree: ast.Module) -> dict[str, tuple[int, str, int]]:
    row_class = _class(tree, "MetricsRow")
    fields: dict[str, tuple[int, str, int]] = {}
    order = 0
    for node in row_class.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        fields[node.target.id] = (node.lineno, ast.unparse(node.annotation), order)
        order += 1
    return fields


def _attribute_chain(node: ast.Attribute) -> str | None:
    parts = [node.attr]
    value: ast.AST = node.value
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
        return ".".join(reversed(parts))
    return None


def _source_attributes(node: ast.AST) -> list[str]:
    roots = {"a", "g", "st", "phenotype", "ledger"}
    attributes = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute):
            chain = _attribute_chain(child)
            if chain is not None and chain.split(".", 1)[0] in roots:
                attributes.add(chain)
        elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "getattr":
            if len(child.args) >= 2 and isinstance(child.args[0], ast.Name):
                name = _literal_string(child.args[1])
                if name is not None and child.args[0].id in roots:
                    attributes.add(f"{child.args[0].id}.{name}")
    return sorted(attributes)


def _primitive_dtype(expression: ast.AST) -> str:
    if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name):
        return {
            "bool": "bool",
            "float": "float64",
            "int": "int64",
            "str": "string",
            "tuple": "tuple",
        }.get(expression.func.id, "derived")
    if isinstance(expression, ast.Tuple):
        return "tuple"
    return "derived"


def _output_dtype(annotation: str) -> str:
    return {
        "float": "float64",
        "int": "int64",
        "str": "string",
        "bool": "bool",
    }.get(annotation, annotation)


def _sha256(relative_path: Path) -> str:
    return hashlib.sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest()


def _native_literal_locations() -> dict[str, list[dict[str, Any]]]:
    first_locations: dict[str, dict[str, int]] = defaultdict(dict)
    for relative_path in NATIVE_DISCOVERY_SOURCES:
        _, tree = _read_source(relative_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                source_locations = first_locations[node.value]
                source = str(relative_path)
                source_locations[source] = min(source_locations.get(source, node.lineno), node.lineno)
    return {
        value: [
            {"source": source, "line": line}
            for source, line in sorted(source_locations.items())
        ]
        for value, source_locations in first_locations.items()
    }


def _native_mapping(key: str, literal_locations: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    hint = NATIVE_FIELD_HINTS.get(key)
    if hint is None:
        raise ContractError(f"canonical primitive lacks native mapping hint: {key}")
    discoveries = {
        candidate: list(literal_locations.get(candidate, ()))
        for candidate in hint.candidate_keys
    }
    discovered_count = sum(bool(rows) for rows in discoveries.values())
    if hint.kind == "placeholder" or not hint.candidate_keys:
        status = "placeholder"
    elif discovered_count == len(hint.candidate_keys):
        status = "discoverable"
    elif discovered_count:
        status = "partial"
    else:
        status = "placeholder"
    return {
        "status": status,
        "kind": hint.kind,
        "candidate_keys": list(hint.candidate_keys),
        "discoveries": discoveries,
        "note": hint.note,
    }


def _reducer_by_column() -> dict[str, str]:
    result: dict[str, str] = {}
    for reducer, columns in REDUCER_GROUPS.items():
        for column in columns:
            if column in result:
                raise ContractError(f"fairness column has multiple reducer labels: {column}")
            result[column] = reducer
    missing = sorted(set(FAIRNESS_COLUMNS) - set(result))
    extra = sorted(set(result) - set(FAIRNESS_COLUMNS))
    if missing or extra:
        raise ContractError(f"reducer coverage mismatch: missing={missing} extra={extra}")
    return result


def _reduction_order(reducer: str, cohort: Mapping[str, Any]) -> str:
    if "percentile" in reducer:
        return f"{cohort['ordering']}; values sorted ascending by stats.percentile before interpolation"
    if reducer == "canonical_lineage_memory_json":
        return "lineage id ascending; agent_id ascending within lineage; JSON object keys sorted"
    if reducer == "canonical_memory_histogram_json":
        return f"{cohort['ordering']}; fixed MEMORY_LADDER bucket order; JSON object keys sorted"
    if reducer == "base2_entropy":
        return f"{cohort['ordering']}; first-seen count insertion order; Python float accumulation"
    return f"{cohort['ordering']}; Python float/list reduction order preserved"


def _reference_writer_metadata(tree: ast.Module) -> dict[str, Any]:
    logger_class = _class(tree, "MetricsLogger")
    method = next(
        (
            node
            for node in logger_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "log_generation"
        ),
        None,
    )
    if method is None:
        raise ContractError("missing MetricsLogger.log_generation")
    reducer_calls = []
    row_calls = []
    for node in ast.walk(method):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "compute_cognitive_status_metrics":
            reducer_calls.append(node.lineno)
        if isinstance(node.func, ast.Name) and node.func.id == "MetricsRow":
            if any(keyword.arg is None and isinstance(keyword.value, ast.Name) and keyword.value.id == "cog_metrics" for keyword in node.keywords):
                row_calls.append(node.lineno)
    if len(reducer_calls) != 1 or len(row_calls) != 1:
        raise ContractError(
            "reference writer must have one cognitive reducer call and one MetricsRow **cog_metrics assignment"
        )
    return {
        "source": str(METRICS_LOGGER_SOURCE),
        "function": "MetricsLogger.log_generation",
        "reducer_call_line": reducer_calls[0],
        "metrics_row_assignment_line": row_calls[0],
    }


def _build_payloads() -> tuple[dict[str, Any], dict[str, Any]]:
    if len(FAIRNESS_COLUMNS) != 66 or len(set(FAIRNESS_COLUMNS)) != 66:
        raise ContractError("explicit fairness list must contain exactly 66 unique columns")

    _, cognitive_tree = _read_source(COGNITIVE_STATUS_SOURCE)
    _, logger_tree = _read_source(METRICS_LOGGER_SOURCE)
    _, row_tree = _read_source(METRICS_ROW_SOURCE)
    reducer, output_assignments, local_assignments = _reducer_model(cognitive_tree)
    helper_functions = {
        node.name: node
        for node in cognitive_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    defaults = _metric_defaults(cognitive_tree)
    row_fields = _metrics_row_fields(row_tree)
    record_fields = _record_fields(cognitive_tree)
    reducer_labels = _reducer_by_column()
    literal_locations = _native_literal_locations()

    missing_assignments = sorted(set(FAIRNESS_COLUMNS) - set(output_assignments))
    missing_defaults = sorted(set(FAIRNESS_COLUMNS) - set(defaults))
    missing_row_fields = sorted(set(FAIRNESS_COLUMNS) - set(row_fields))
    if missing_assignments or missing_defaults or missing_row_fields:
        raise ContractError(
            "fairness source coverage failure: "
            f"unassigned={missing_assignments} missing_defaults={missing_defaults} "
            f"missing_metrics_row_fields={missing_row_fields}"
        )
    row_order = [name for name, (_, _, index) in sorted(row_fields.items(), key=lambda item: item[1][2])]
    filtered_row_order = tuple(name for name in row_order if name in set(FAIRNESS_COLUMNS))
    if filtered_row_order != FAIRNESS_COLUMNS:
        raise ContractError("explicit fairness list no longer matches MetricsRow/metrics.csv order")

    primitive_keys_by_column: dict[str, list[str]] = {}
    for column in FAIRNESS_COLUMNS:
        cohort_name = COLUMN_COHORT_OVERRIDES.get(column, "all_live")
        cohort = COHORT_RULES.get(cohort_name)
        if cohort is None:
            raise ContractError(f"fairness column has no cohort rule: {column}")
        keys = _primitive_keys_for_column(
            column,
            output_assignments=output_assignments,
            local_assignments=local_assignments,
            helper_functions=helper_functions,
        )
        keys.update(cohort["membership_primitive_keys"])
        if not keys:
            raise ContractError(f"fairness column has no primitive keys: {column}")
        missing_record_keys = sorted(keys - set(record_fields))
        if missing_record_keys:
            raise ContractError(
                f"fairness column {column} uses primitive keys absent from canonical records: {missing_record_keys}"
            )
        primitive_keys_by_column[column] = sorted(keys)

    primitive_keys = sorted({key for keys in primitive_keys_by_column.values() for key in keys})
    missing_native_hints = sorted(set(primitive_keys) - set(NATIVE_FIELD_HINTS))
    if missing_native_hints:
        raise ContractError(f"canonical primitive keys lack native mapping hints: {missing_native_hints}")

    schema_fields = []
    for key in primitive_keys:
        line, expression = record_fields[key]
        schema_fields.append(
            {
                "key": key,
                "dtype": _primitive_dtype(expression),
                "required": True,
                "default": None,
                "missing_behavior": "fail closed before reduction",
                "reference_adapter": {
                    "source": str(COGNITIVE_STATUS_SOURCE),
                    "function": "_agent_records",
                    "line": line,
                    "expression": ast.unparse(expression),
                    "source_agent_attributes": _source_attributes(expression),
                },
                "native_mapping": _native_mapping(key, literal_locations),
                "required_by_columns": [
                    column for column in FAIRNESS_COLUMNS if key in primitive_keys_by_column[column]
                ],
            }
        )

    unresolved_native_primitive_keys = sorted(
        field["key"]
        for field in schema_fields
        if field["native_mapping"]["status"] != "discoverable"
    )
    if unresolved_native_primitive_keys:
        raise ContractError(
            "fairness primitive native mapping coverage failure: "
            f"unresolved={unresolved_native_primitive_keys}"
        )

    schema_by_key = {field["key"]: field for field in schema_fields}
    columns = []
    for column in FAIRNESS_COLUMNS:
        assignment_line, expression = output_assignments[column][-1]
        cohort_name = COLUMN_COHORT_OVERRIDES.get(column, "all_live")
        cohort = dict(COHORT_RULES[cohort_name])
        cohort["membership_primitive_keys"] = list(cohort["membership_primitive_keys"])
        reducer_name = reducer_labels[column]
        keys = primitive_keys_by_column[column]
        primitive_mappings = {
            key: {
                "schema_field": key,
                "status": schema_by_key[key]["native_mapping"]["status"],
                "kind": schema_by_key[key]["native_mapping"]["kind"],
                "candidate_keys": schema_by_key[key]["native_mapping"]["candidate_keys"],
            }
            for key in keys
        }
        unresolved_keys = [
            key for key, mapping in primitive_mappings.items() if mapping["status"] != "discoverable"
        ]
        columns.append(
            {
                "column": column,
                "authoritative_reducer": {
                    "source": str(COGNITIVE_STATUS_SOURCE),
                    "function": reducer.name,
                    "line": assignment_line,
                    "expression": ast.unparse(expression),
                },
                "primitive_keys": keys,
                "source_agent_attributes": sorted(
                    {
                        attribute
                        for key in keys
                        for attribute in schema_by_key[key]["reference_adapter"]["source_agent_attributes"]
                    }
                ),
                "cohort": {"name": cohort_name, **cohort},
                "reducer": reducer_name,
                "default": defaults[column],
                "missing_behavior": "return the declared default when the reducer has no usable records/values",
                "reduction_order": _reduction_order(reducer_name, cohort),
                "dtype": _output_dtype(row_fields[column][1]),
                "metrics_row_order": row_fields[column][2],
                "native_mapping": {
                    "status": "unresolved" if unresolved_keys else "discoverable",
                    "primitive_mappings": primitive_mappings,
                    "unmapped_primitive_keys": unresolved_keys,
                },
            }
        )

    sources = {
        str(path): {"sha256": _sha256(path)}
        for path in (
            COGNITIVE_STATUS_SOURCE,
            METRICS_LOGGER_SOURCE,
            METRICS_ROW_SOURCE,
            STATS_SOURCE,
            *NATIVE_DISCOVERY_SOURCES,
        )
    }
    contract = {
        "schema_version": "flat_resident_reporting_66_column_contract_v1",
        "scope": "current Flat resident generation reporting fairness columns after chemistry closure",
        "column_count": len(columns),
        "fairness_columns": list(FAIRNESS_COLUMNS),
        "reference_writer": _reference_writer_metadata(logger_tree),
        "reducer_source": {
            "source": str(COGNITIVE_STATUS_SOURCE),
            "function": reducer.name,
            "line": reducer.lineno,
        },
        "stats_contract": {
            "source": str(STATS_SOURCE),
            "mean": "sum(xs) / len(xs), default 0.0",
            "std": "population standard deviation, default 0.0",
            "percentile": "ascending sort with linear interpolation, default 0.0",
        },
        "columns": columns,
        "source_files": sources,
        "validation": {
            "explicit_unique_column_count": 66,
            "all_columns_assigned": True,
            "all_columns_have_defaults": True,
            "all_columns_have_metrics_row_dtype_and_order": True,
            "all_primitive_keys_have_reference_adapter_sources": True,
            "all_primitive_keys_have_discoverable_native_mappings": True,
        },
    }
    schema = {
        "schema_version": "flat_population_reporting_schema_v1",
        "record_name": "FlatPopulationReportingRecordV1",
        "purpose": "side-effect-free canonical records consumed by the shared cognitive/developmental reducer",
        "strict": True,
        "missing_field_behavior": "fail closed; no reducer-time field defaults",
        "record_order": "agent_id ascending before any reduction",
        "collections": [
            {
                "name": "live_records",
                "definition": "authoritative generation live population records",
                "order": "agent_id ascending",
            },
            {
                "name": "selected_parent_records",
                "definition": "authoritative selected-parent snapshot using the same record schema",
                "order": "agent_id ascending",
                "native_membership": "behavior row kept_next > 0 joined by agent_id",
            },
        ],
        "cohorts": [{"name": name, **{**rule, "membership_primitive_keys": list(rule["membership_primitive_keys"])}} for name, rule in COHORT_RULES.items()],
        "field_count": len(schema_fields),
        "fields": schema_fields,
        "fairness_column_count": len(FAIRNESS_COLUMNS),
        "fairness_columns": list(FAIRNESS_COLUMNS),
        "source_files": sources,
    }
    return contract, schema


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_outputs(output_dir: Path, outputs: Mapping[str, bytes]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in outputs.items():
        (output_dir / filename).write_bytes(data)


def _check_outputs(output_dir: Path, outputs: Mapping[str, bytes]) -> list[str]:
    errors = []
    for filename, expected in outputs.items():
        path = output_dir / filename
        try:
            observed = path.read_bytes()
        except FileNotFoundError:
            errors.append(f"missing: {path}")
            continue
        except OSError as exc:
            errors.append(f"cannot read {path}: {exc}")
            continue
        if observed != expected:
            errors.append(f"out of date: {path}")
    return errors


def _summary(output_dir: Path, outputs: Mapping[str, bytes], *, checked: bool) -> dict[str, Any]:
    return {
        "status": "checked" if checked else "written",
        "output_dir": str(output_dir),
        "column_count": 66,
        "outputs": {
            filename: {
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for filename, data in outputs.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify existing outputs are byte-for-byte current without writing",
    )
    args = parser.parse_args(argv)

    try:
        contract, schema = _build_payloads()
    except ContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    outputs = {
        CONTRACT_FILENAME: _json_bytes(contract),
        SCHEMA_FILENAME: _json_bytes(schema),
    }
    output_dir = args.output_dir.resolve()
    if args.check:
        errors = _check_outputs(output_dir, outputs)
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
    else:
        _write_outputs(output_dir, outputs)
    print(json.dumps(_summary(output_dir, outputs, checked=args.check), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
