"""Pure composition for the route-neutral EIEL science instrument.

This module expands normalized science intent into deterministic semantic,
physical, and reduction identities.  It has no filesystem, provider, build,
resource, admission, lifecycle, or execution effects.  CRCA identity and route
selection arrive as already-resolved bindings from their canonical owners.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from emergence_contracts import (
    SEMANTIC_ACQUISITION_CORE_SCHEMA_VERSION,
    CanonicalPayloadV1,
    ContentDigest,
    SchemaVersion,
    ScienceCampaignV1,
    ScienceControlDescriptorV1,
    ScienceExecutionPlanV1,
    ScienceInstrumentManifestV1,
    ScienceReducerDescriptorV1,
    ScienceReductionPlanV1,
    ScienceRouteBindingV1,
    canonical_json_bytes,
    canonical_sha256,
)

from .execution_identity_projection import (
    SHADOW_PROJECTION_AUTHORITY,
    validate_shadow_projection_v1,
)

SCIENCE_COMPILATION_BUNDLE_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_compilation_bundle_v1"
)
_REDUCTION_CONFIGURATION_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_reducer_configurations_v1"
)
_REDUCTION_EVIDENCE_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_reduction_evidence_requirements_v1"
)
_REDUCTION_DECISION_SCHEMA_VERSION = SchemaVersion("eiel_science_offline_decisions_v1")


class ScienceCompilationError(ValueError):
    """Normalized science intent cannot be compiled exactly and purely."""


@dataclass(frozen=True, slots=True)
class ScienceCRCASemanticBindingV1:
    """Validated projection of the canonical CRCA semantic identity seam."""

    semantic_run_digest: ContentDigest
    projection_sha256: ContentDigest
    absolute_start_next_tick: int
    absolute_target_next_tick: int
    trajectory_checkpoint_ticks: tuple[int, ...]
    authority: str = SHADOW_PROJECTION_AUTHORITY

    def __post_init__(self) -> None:
        if self.authority != SHADOW_PROJECTION_AUTHORITY:
            raise ScienceCompilationError("unsupported CRCA projection authority")
        if (
            type(self.absolute_start_next_tick) is not int
            or type(self.absolute_target_next_tick) is not int
            or self.absolute_start_next_tick < 0
            or self.absolute_target_next_tick <= self.absolute_start_next_tick
        ):
            raise ScienceCompilationError("CRCA absolute-tick span is invalid")
        if (
            any(type(tick) is not int for tick in self.trajectory_checkpoint_ticks)
            or tuple(sorted(set(self.trajectory_checkpoint_ticks)))
            != self.trajectory_checkpoint_ticks
        ):
            raise ScienceCompilationError(
                "CRCA trajectory checkpoint ticks are not canonical"
            )


def bind_crca_semantic_projection_v1(
    receipt: dict[str, Any],
) -> ScienceCRCASemanticBindingV1:
    """Validate a CRCA projection receipt and expose only its semantic binding."""

    try:
        validated = validate_shadow_projection_v1(receipt)
        semantic = validated["layers"]["semantic"]
        payload = semantic["payload"]
        digest = semantic["digest"]
        ticks = tuple(payload["trajectory_checkpoint_ticks"])
        return ScienceCRCASemanticBindingV1(
            semantic_run_digest=ContentDigest(digest["algorithm"], digest["hex"]),
            projection_sha256=ContentDigest("sha256", validated["projection_sha256"]),
            absolute_start_next_tick=payload["absolute_start_next_tick"],
            absolute_target_next_tick=payload["absolute_target_next_tick"],
            trajectory_checkpoint_ticks=ticks,
            authority=validated["authority"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ScienceCompilationError(
            "CRCA semantic projection receipt is invalid"
        ) from error


def science_control_catalog_sha256(
    descriptors: tuple[ScienceControlDescriptorV1, ...],
) -> ContentDigest:
    """Return the order-independent content identity of a control catalog."""

    rows = sorted(
        (descriptor.to_dict() for descriptor in descriptors),
        key=lambda row: str(row["control_id"]),
    )
    if len({row["control_id"] for row in rows}) != len(rows):
        raise ScienceCompilationError("control descriptor IDs must be unique")
    return canonical_sha256(rows)


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScienceCompilationError(f"{field} must be an object")
    if any(type(key) is not str for key in value):
        raise ScienceCompilationError(f"{field} keys must be strings")
    return dict(value)


def _list(value: object, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ScienceCompilationError(f"{field} must be an array")
    return list(value)


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ScienceCompilationError(f"{field} must be a non-empty exact string")
    return value


def _canonical_value_key(value: object) -> bytes:
    return canonical_json_bytes(value)


def _dimension_rows(campaign: ScienceCampaignV1) -> tuple[dict[str, Any], ...]:
    design = _mapping(campaign.design.to_value(), field="design")
    raw_dimensions = _list(design.get("dimensions"), field="design.dimensions")
    dimensions: list[tuple[str, tuple[Any, ...]]] = []
    for index, raw in enumerate(raw_dimensions):
        dimension = _mapping(raw, field=f"design.dimensions[{index}]")
        dimension_id = _text(
            dimension.get("dimension_id"),
            field=f"design.dimensions[{index}].dimension_id",
        )
        values = _list(
            dimension.get("canonical_values"),
            field=f"design.dimensions[{index}].canonical_values",
        )
        if not values:
            raise ScienceCompilationError("dimension values cannot be empty")
        keys = [_canonical_value_key(value) for value in values]
        if len(set(keys)) != len(keys):
            raise ScienceCompilationError(
                f"dimension {dimension_id!r} has duplicate canonical values"
            )
        if dimension.get("ordering", "declared") == "canonical_json_bytes":
            values = [value for _, value in sorted(zip(keys, values, strict=True))]
        dimensions.append((dimension_id, tuple(values)))
    dimensions.sort(key=lambda item: item[0])
    if len({item[0] for item in dimensions}) != len(dimensions):
        raise ScienceCompilationError("dimension IDs must be unique")

    if dimensions:
        rows = [
            dict(zip((item[0] for item in dimensions), combination, strict=True))
            for combination in product(*(item[1] for item in dimensions))
        ]
    else:
        rows = [{}]

    def row_pattern(raw: object, *, field: str) -> dict[str, Any]:
        pattern = _mapping(raw, field=field)
        bindings = _list(pattern.get("bindings"), field=f"{field}.bindings")
        result: dict[str, Any] = {}
        for binding_index, binding_raw in enumerate(bindings):
            binding = _mapping(binding_raw, field=f"{field}.bindings[{binding_index}]")
            dimension_id = _text(
                binding.get("dimension_id"),
                field=f"{field}.bindings[{binding_index}].dimension_id",
            )
            if dimension_id not in {item[0] for item in dimensions}:
                raise ScienceCompilationError(
                    f"{field} names unknown dimension {dimension_id!r}"
                )
            if dimension_id in result:
                raise ScienceCompilationError(
                    f"{field} repeats dimension {dimension_id!r}"
                )
            result[dimension_id] = binding.get("value")
        return result

    def matches(row: dict[str, Any], pattern: dict[str, Any]) -> bool:
        return all(
            key in row and _canonical_value_key(row[key]) == _canonical_value_key(value)
            for key, value in pattern.items()
        )

    include_patterns = tuple(
        row_pattern(raw, field=f"design.include_rows[{index}]")
        for index, raw in enumerate(
            _list(design.get("include_rows", []), field="design.include_rows")
        )
    )
    exclude_patterns = tuple(
        row_pattern(raw, field=f"design.exclude_rows[{index}]")
        for index, raw in enumerate(
            _list(design.get("exclude_rows", []), field="design.exclude_rows")
        )
    )
    if include_patterns:
        rows = [
            row for row in rows if any(matches(row, item) for item in include_patterns)
        ]
    rows = [
        row for row in rows if not any(matches(row, item) for item in exclude_patterns)
    ]
    rows.sort(key=_canonical_value_key)
    expected = design.get("expected_cell_count")
    if type(expected) is not int or expected != len(rows):
        raise ScienceCompilationError(
            "design.expected_cell_count differs from deterministic expansion"
        )
    if not rows:
        raise ScienceCompilationError("campaign expansion cannot be empty")
    return tuple(rows)


def _resolved_seed_bindings(
    campaign: ScienceCampaignV1, dimensions: dict[str, Any]
) -> dict[str, int]:
    rows = _list(campaign.seeds.to_value(), field="seeds")
    result: dict[str, int] = {}
    for index, raw in enumerate(rows):
        binding = _mapping(raw, field=f"seeds[{index}]")
        owner = _text(binding.get("owner"), field=f"seeds[{index}].owner")
        if owner in result:
            raise ScienceCompilationError(f"seed owner {owner!r} is duplicated")
        kind = binding.get("binding_kind")
        if kind == "literal":
            seed = binding.get("literal")
        elif kind == "dimension_bound":
            dimension_id = _text(
                binding.get("dimension_id"),
                field=f"seeds[{index}].dimension_id",
            )
            if dimension_id not in dimensions:
                raise ScienceCompilationError(
                    f"seed owner {owner!r} names an absent dimension"
                )
            seed = dimensions[dimension_id]
        else:
            raise ScienceCompilationError(f"seed owner {owner!r} has invalid binding")
        if type(seed) is not int or seed < 0:
            raise ScienceCompilationError(
                f"resolved seed for owner {owner!r} must be nonnegative integer"
            )
        result[owner] = seed
    if not result:
        raise ScienceCompilationError("at least one domain seed is required")
    return dict(sorted(result.items()))


def _resolved_control_bindings(
    campaign: ScienceCampaignV1,
    dimensions: dict[str, Any],
    descriptors: dict[str, ScienceControlDescriptorV1],
) -> list[dict[str, Any]]:
    rows = _list(campaign.controls.to_value(), field="controls")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        binding = _mapping(raw, field=f"controls[{index}]")
        control_id = _text(
            binding.get("control_id"), field=f"controls[{index}].control_id"
        )
        if control_id in seen:
            raise ScienceCompilationError(f"control {control_id!r} is duplicated")
        seen.add(control_id)
        descriptor = descriptors.get(control_id)
        if descriptor is None:
            raise ScienceCompilationError(
                f"control {control_id!r} is absent from the bound catalog"
            )
        schedule = _mapping(
            binding.get("schedule"), field=f"controls[{index}].schedule"
        )
        if schedule.get("kind") not in descriptor.schedule_kinds:
            raise ScienceCompilationError(
                f"control {control_id!r} uses an unsupported schedule kind"
            )
        if "dimension_binding" in binding:
            dimension_id = _text(
                binding["dimension_binding"],
                field=f"controls[{index}].dimension_binding",
            )
            if dimension_id not in dimensions:
                raise ScienceCompilationError(
                    f"control {control_id!r} names an absent dimension"
                )
            resolved_value: object = dimensions[dimension_id]
        elif "value" in binding:
            resolved_value = binding["value"]
        else:
            resolved_value = {"profile_ref": binding.get("profile_ref")}
        result.append(
            {
                "control_id": control_id,
                "resolved_value": resolved_value,
                "schedule": schedule,
                "descriptor_sha256": descriptor.descriptor_sha256().to_dict(),
            }
        )
    return sorted(result, key=lambda row: str(row["control_id"]))


def _semantic_acquisition_core(
    campaign: ScienceCampaignV1,
    descriptors: dict[str, ScienceControlDescriptorV1],
) -> CanonicalPayloadV1:
    cells: list[dict[str, Any]] = []
    world = campaign.world.to_value()
    initialization = campaign.initialization.to_value()
    for dimensions in _dimension_rows(campaign):
        seeds = _resolved_seed_bindings(campaign, dimensions)
        controls = _resolved_control_bindings(campaign, dimensions, descriptors)
        identity_core = {
            "dimension_bindings": dimensions,
            "seed_bindings": seeds,
            "control_bindings": controls,
            "world": world,
            "initialization": initialization,
        }
        cell_sha256 = canonical_sha256(identity_core)
        cells.append(
            {
                "semantic_cell_id": f"cell-{cell_sha256.hex[:24]}",
                **identity_core,
            }
        )
    cells.sort(key=lambda row: str(row["semantic_cell_id"]))
    if len({row["semantic_cell_id"] for row in cells}) != len(cells):
        raise ScienceCompilationError("semantic cell identities collide")
    for ordinal, row in enumerate(cells):
        row["logical_world_id"] = ordinal
    return CanonicalPayloadV1.from_value(
        SEMANTIC_ACQUISITION_CORE_SCHEMA_VERSION,
        {
            "cells": cells,
            "time": campaign.time.to_value(),
            "observations": campaign.observations.to_value(),
        },
    )


def _reduction_plan(
    campaign: ScienceCampaignV1,
    available: tuple[ScienceReducerDescriptorV1, ...],
) -> ScienceReductionPlanV1:
    by_id = {descriptor.reducer_id: descriptor for descriptor in available}
    if len(by_id) != len(available):
        raise ScienceCompilationError("available reducer IDs must be unique")
    selected: list[ScienceReducerDescriptorV1] = []
    configurations: list[dict[str, Any]] = []
    required_capabilities: set[str] = set()
    output_contract_ids: set[str] = set()
    for index, raw in enumerate(_list(campaign.reducers.to_value(), field="reducers")):
        binding = _mapping(raw, field=f"reducers[{index}]")
        reducer_id = _text(
            binding.get("reducer_id"), field=f"reducers[{index}].reducer_id"
        )
        descriptor = by_id.get(reducer_id)
        if descriptor is None:
            raise ScienceCompilationError(f"reducer {reducer_id!r} is not available")
        if str(descriptor.contract_version) != binding.get("contract_version"):
            raise ScienceCompilationError(
                f"reducer {reducer_id!r} contract version differs"
            )
        if descriptor.descriptor_sha256().to_dict() != binding.get("descriptor_sha256"):
            raise ScienceCompilationError(
                f"reducer {reducer_id!r} descriptor identity differs"
            )
        if list(descriptor.required_observation_capability_ids) != binding.get(
            "required_observation_capability_ids"
        ):
            raise ScienceCompilationError(
                f"reducer {reducer_id!r} observation requirements differ"
            )
        if descriptor.output_contract_id != binding.get("output_contract_id"):
            raise ScienceCompilationError(
                f"reducer {reducer_id!r} output contract differs"
            )
        selected.append(descriptor)
        configurations.append(
            {
                "reducer_id": reducer_id,
                "configuration": binding.get("configuration"),
            }
        )
        required_capabilities.update(descriptor.required_observation_capability_ids)
        output_contract_ids.add(descriptor.output_contract_id)
    if len({item.reducer_id for item in selected}) != len(selected):
        raise ScienceCompilationError("campaign reducer IDs must be unique")
    selected.sort(key=lambda item: item.reducer_id)
    configurations.sort(key=lambda item: str(item["reducer_id"]))
    decision = _mapping(campaign.decision.to_value(), field="decision")
    offline_decisions = {
        key: value for key, value in decision.items() if key != "presentation_metadata"
    }
    return ScienceReductionPlanV1(
        reducer_descriptors=tuple(selected),
        normalized_configurations=CanonicalPayloadV1.from_value(
            _REDUCTION_CONFIGURATION_SCHEMA_VERSION, configurations
        ),
        evidence_requirements=CanonicalPayloadV1.from_value(
            _REDUCTION_EVIDENCE_SCHEMA_VERSION,
            {"required_observation_capability_ids": sorted(required_capabilities)},
        ),
        offline_decisions=CanonicalPayloadV1.from_value(
            _REDUCTION_DECISION_SCHEMA_VERSION, offline_decisions
        ),
        output_contract_ids=tuple(sorted(output_contract_ids)),
    )


@dataclass(frozen=True, slots=True)
class ScienceCompilationBundleV1:
    campaign_spec_sha256: ContentDigest
    compiler_component_sha256: ContentDigest
    crca_projection_sha256: ContentDigest
    execution_plan: ScienceExecutionPlanV1
    reduction_plan: ScienceReductionPlanV1
    semantic_cells: CanonicalPayloadV1
    schema_version: SchemaVersion = SCIENCE_COMPILATION_BUNDLE_SCHEMA_VERSION

    def bundle_core_dict(self) -> dict[str, object]:
        return {
            "campaign_spec_sha256": self.campaign_spec_sha256.to_dict(),
            "compiler_component_sha256": self.compiler_component_sha256.to_dict(),
            "crca_projection_sha256": self.crca_projection_sha256.to_dict(),
            "execution_plan": self.execution_plan.to_dict(),
            "reduction_plan": self.reduction_plan.to_dict(),
            "semantic_cells": self.semantic_cells.to_dict(),
        }

    def compilation_bundle_sha256(self) -> ContentDigest:
        return canonical_sha256(
            {
                "schema_version": str(self.schema_version),
                "bundle_core": self.bundle_core_dict(),
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "bundle_core": self.bundle_core_dict(),
            "compilation_bundle_sha256": self.compilation_bundle_sha256().to_dict(),
        }


def compile_science_campaign_v1(
    campaign: ScienceCampaignV1,
    *,
    instrument_manifest: ScienceInstrumentManifestV1,
    route_binding: ScienceRouteBindingV1,
    control_descriptors: tuple[ScienceControlDescriptorV1, ...],
    reducer_descriptors: tuple[ScienceReducerDescriptorV1, ...],
    crca_semantic_binding: ScienceCRCASemanticBindingV1,
    physical_execution_core: CanonicalPayloadV1,
    compiler_component_sha256: ContentDigest,
    legacy_campaign_plan_sha256: ContentDigest | None = None,
) -> ScienceCompilationBundleV1:
    """Compile one rich campaign without selecting, building, or running a route."""

    manifest_sha256 = instrument_manifest.manifest_sha256()
    instrument_request = _mapping(campaign.instrument.to_value(), field="instrument")
    requested_manifest = _mapping(
        instrument_request.get("instrument_manifest_sha256"),
        field="instrument.instrument_manifest_sha256",
    )
    if requested_manifest != manifest_sha256.to_dict():
        raise ScienceCompilationError(
            "campaign does not name the supplied instrument manifest"
        )
    if route_binding.instrument_manifest_sha256 != manifest_sha256:
        raise ScienceCompilationError(
            "route binding does not name the supplied instrument manifest"
        )
    if route_binding.machine_profile_id != campaign.execution.to_value().get(
        "machine_profile_id"
    ):
        raise ScienceCompilationError(
            "route binding and campaign machine profiles differ"
        )
    if science_control_catalog_sha256(control_descriptors) != (
        instrument_manifest.control_catalog_sha256
    ):
        raise ScienceCompilationError(
            "control descriptors differ from the instrument manifest"
        )
    time = _mapping(campaign.time.to_value(), field="time")
    start_next_tick = time.get("start_next_tick")
    total_ticks = time.get("total_ticks")
    checkpoint_ticks = tuple(time.get("checkpoint_ticks", []))
    if (
        type(start_next_tick) is not int
        or type(total_ticks) is not int
        or crca_semantic_binding.absolute_start_next_tick != start_next_tick
        or crca_semantic_binding.absolute_target_next_tick
        != start_next_tick + total_ticks
        or crca_semantic_binding.trajectory_checkpoint_ticks
        != tuple(dict.fromkeys((start_next_tick, *checkpoint_ticks)))
    ):
        raise ScienceCompilationError(
            "CRCA semantic projection absolute-tick span differs from the campaign"
        )
    descriptor_map = {item.control_id: item for item in control_descriptors}
    semantic_core = _semantic_acquisition_core(campaign, descriptor_map)
    execution_plan = ScienceExecutionPlanV1(
        instrument_manifest_sha256=manifest_sha256,
        route_binding_sha256=route_binding.route_binding_sha256(),
        crca_semantic_run_digest=crca_semantic_binding.semantic_run_digest,
        semantic_acquisition_core=semantic_core,
        physical_execution_core=physical_execution_core,
        legacy_campaign_plan_sha256=legacy_campaign_plan_sha256,
    )
    reduction_plan = _reduction_plan(campaign, reducer_descriptors)
    return ScienceCompilationBundleV1(
        campaign_spec_sha256=campaign.campaign_spec_sha256(),
        compiler_component_sha256=compiler_component_sha256,
        crca_projection_sha256=crca_semantic_binding.projection_sha256,
        execution_plan=execution_plan,
        reduction_plan=reduction_plan,
        semantic_cells=CanonicalPayloadV1.from_value(
            SchemaVersion("eiel_science_semantic_cells_v1"),
            semantic_core.to_value()["cells"],
        ),
    )


__all__ = (
    "SCIENCE_COMPILATION_BUNDLE_SCHEMA_VERSION",
    "ScienceCompilationBundleV1",
    "ScienceCompilationError",
    "ScienceCRCASemanticBindingV1",
    "bind_crca_semantic_projection_v1",
    "compile_science_campaign_v1",
    "science_control_catalog_sha256",
)
