"""Pure projection from admitted rich science intent to the unchanged QRR form."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from emergence_contracts import (
    CanonicalPayloadV1,
    ContentDigest,
    SchemaVersion,
    ScienceCampaignV1,
    ScienceRouteBindingV1,
    canonical_json_bytes,
    canonical_sha256,
)
from engine.campaigns.artifact_lifecycle import lifecycle_profile_contract_v1

from .science_instrument import ScienceCompilationBundleV1

SCIENCE_QRR_ROUTE_ADAPTER_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_qrr_route_adapter_v1"
)
SCIENCE_QRR_EXECUTION_ADAPTER_CORE_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_qrr_execution_adapter_core_v1"
)
SCIENCE_QRR_CONTROL_ADAPTER_CORE_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_qrr_control_adapter_core_v1"
)
SCIENCE_RICH_TO_FLAT_PROJECTION_RECEIPT_SCHEMA_VERSION = (
    "eiel_science_rich_to_flat_projection_receipt_v1"
)
SCIENCE_QRR_BINDING_RECEIPT_SCHEMA_VERSION = "eiel_science_qrr_binding_receipt_v1"

_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


class ScienceQRRProjectionError(ValueError):
    """Rich semantics cannot be represented exactly by the selected QRR adapter."""


def _token(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise ScienceQRRProjectionError(f"{field} must be a path-free token")
    return value


def _sorted_mapping_pairs(
    value: tuple[tuple[str, str], ...], *, field: str
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, tuple):
        raise ScienceQRRProjectionError(f"{field} must be a tuple")
    normalized: list[tuple[str, str]] = []
    for key, target in value:
        normalized.append(
            (_token(key, field=f"{field}.key"), _token(target, field=f"{field}.target"))
        )
    if tuple(sorted(normalized)) != value or len({item[0] for item in value}) != len(
        value
    ):
        raise ScienceQRRProjectionError(f"{field} must be sorted with unique keys")
    return value


@dataclass(frozen=True, slots=True)
class ScienceQRRRouteAdapterV1:
    """Content-addressed, route-owned lowering data; never science semantics."""

    route_id: str
    machine_profile_id: str
    world_profile_id: str
    world_width: int
    maximum_batch_cells: int
    task_seed_owner: str
    control_path_by_id: tuple[tuple[str, str], ...]
    seed_path_by_owner: tuple[tuple[str, str], ...]
    fixed_flat_controls: CanonicalPayloadV1
    acs_capture_instrumentation: bool
    acs_checkpoint_slots: int | None
    living_lineage_canopy: bool
    lifecycle_total_max_bytes: int
    schema_version: SchemaVersion = SCIENCE_QRR_ROUTE_ADAPTER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("route_id", "machine_profile_id", "world_profile_id"):
            _token(getattr(self, field_name), field=field_name)
        if type(self.world_width) is not int or self.world_width <= 0:
            raise ScienceQRRProjectionError("world_width must be positive")
        if type(self.maximum_batch_cells) is not int or self.maximum_batch_cells <= 0:
            raise ScienceQRRProjectionError("maximum_batch_cells must be positive")
        _token(self.task_seed_owner, field="task_seed_owner")
        _sorted_mapping_pairs(self.control_path_by_id, field="control_path_by_id")
        _sorted_mapping_pairs(self.seed_path_by_owner, field="seed_path_by_owner")
        if set(dict(self.seed_path_by_owner)) & {self.task_seed_owner}:
            raise ScienceQRRProjectionError(
                "task seed owner cannot also be a flat control seed"
            )
        if not isinstance(
            self.fixed_flat_controls, CanonicalPayloadV1
        ) or not isinstance(self.fixed_flat_controls.to_value(), dict):
            raise ScienceQRRProjectionError(
                "fixed_flat_controls must be an object payload"
            )
        if type(self.acs_capture_instrumentation) is not bool:
            raise ScienceQRRProjectionError(
                "acs_capture_instrumentation must be boolean"
            )
        if self.acs_checkpoint_slots is not None and (
            type(self.acs_checkpoint_slots) is not int
            or self.acs_checkpoint_slots not in {0, 1, 2, 3}
        ):
            raise ScienceQRRProjectionError("acs_checkpoint_slots is invalid")
        if self.acs_checkpoint_slots and not self.acs_capture_instrumentation:
            raise ScienceQRRProjectionError("ACS slots require capture instrumentation")
        if type(self.living_lineage_canopy) is not bool:
            raise ScienceQRRProjectionError("living_lineage_canopy must be boolean")
        if (
            type(self.lifecycle_total_max_bytes) is not int
            or self.lifecycle_total_max_bytes <= 0
        ):
            raise ScienceQRRProjectionError(
                "lifecycle_total_max_bytes must be positive"
            )
        if self.schema_version != SCIENCE_QRR_ROUTE_ADAPTER_SCHEMA_VERSION:
            raise ScienceQRRProjectionError("route adapter schema is unsupported")

    def execution_adapter_core(self) -> dict[str, object]:
        return {
            "schema_version": str(SCIENCE_QRR_EXECUTION_ADAPTER_CORE_SCHEMA_VERSION),
            "route_id": self.route_id,
            "machine_profile_id": self.machine_profile_id,
            "world_profile_id": self.world_profile_id,
            "world_width": self.world_width,
            "maximum_batch_cells": self.maximum_batch_cells,
            "acs_capture_instrumentation": self.acs_capture_instrumentation,
            "acs_checkpoint_slots": self.acs_checkpoint_slots,
            "living_lineage_canopy": self.living_lineage_canopy,
            "lifecycle_total_max_bytes": self.lifecycle_total_max_bytes,
        }

    def control_adapter_core(self) -> dict[str, object]:
        return {
            "schema_version": str(SCIENCE_QRR_CONTROL_ADAPTER_CORE_SCHEMA_VERSION),
            "task_seed_owner": self.task_seed_owner,
            "control_path_by_id": [
                {"control_id": key, "flat_path": target}
                for key, target in self.control_path_by_id
            ],
            "seed_path_by_owner": [
                {"owner": key, "flat_path": target}
                for key, target in self.seed_path_by_owner
            ],
            "fixed_flat_controls": self.fixed_flat_controls.to_dict(),
        }

    def execution_adapter_sha256(self) -> ContentDigest:
        return canonical_sha256(self.execution_adapter_core())

    def control_adapter_sha256(self) -> ContentDigest:
        return canonical_sha256(self.control_adapter_core())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "execution_adapter_core": self.execution_adapter_core(),
            "control_adapter_core": self.control_adapter_core(),
            "execution_adapter_sha256": self.execution_adapter_sha256().to_dict(),
            "control_adapter_sha256": self.control_adapter_sha256().to_dict(),
        }

    def adapter_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "ScienceQRRRouteAdapterV1":
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "execution_adapter_core",
            "control_adapter_core",
            "execution_adapter_sha256",
            "control_adapter_sha256",
        }:
            raise ScienceQRRProjectionError("route adapter artifact has invalid fields")
        execution = value["execution_adapter_core"]
        control = value["control_adapter_core"]
        if not isinstance(execution, dict) or not isinstance(control, dict):
            raise ScienceQRRProjectionError("route adapter cores must be objects")
        if set(execution) != {
            "schema_version",
            "route_id",
            "machine_profile_id",
            "world_profile_id",
            "world_width",
            "maximum_batch_cells",
            "acs_capture_instrumentation",
            "acs_checkpoint_slots",
            "living_lineage_canopy",
            "lifecycle_total_max_bytes",
        } or set(control) != {
            "schema_version",
            "task_seed_owner",
            "control_path_by_id",
            "seed_path_by_owner",
            "fixed_flat_controls",
        }:
            raise ScienceQRRProjectionError("route adapter core fields are invalid")
        if execution["schema_version"] != str(
            SCIENCE_QRR_EXECUTION_ADAPTER_CORE_SCHEMA_VERSION
        ) or control["schema_version"] != str(
            SCIENCE_QRR_CONTROL_ADAPTER_CORE_SCHEMA_VERSION
        ):
            raise ScienceQRRProjectionError("route adapter core schema is unsupported")

        def pairs(
            rows: object, *, key_name: str, field: str
        ) -> tuple[tuple[str, str], ...]:
            if not isinstance(rows, list):
                raise ScienceQRRProjectionError(f"{field} must be an array")
            result: list[tuple[str, str]] = []
            for row in rows:
                if not isinstance(row, dict) or set(row) != {key_name, "flat_path"}:
                    raise ScienceQRRProjectionError(f"{field} row is invalid")
                result.append((row[key_name], row["flat_path"]))
            return tuple(result)

        try:
            fixed = CanonicalPayloadV1.from_dict(control["fixed_flat_controls"])
        except (TypeError, ValueError) as error:
            raise ScienceQRRProjectionError(
                "fixed_flat_controls payload is invalid"
            ) from error
        adapter = cls(
            route_id=execution["route_id"],
            machine_profile_id=execution["machine_profile_id"],
            world_profile_id=execution["world_profile_id"],
            world_width=execution["world_width"],
            maximum_batch_cells=execution["maximum_batch_cells"],
            task_seed_owner=control["task_seed_owner"],
            control_path_by_id=pairs(
                control["control_path_by_id"],
                key_name="control_id",
                field="control_path_by_id",
            ),
            seed_path_by_owner=pairs(
                control["seed_path_by_owner"],
                key_name="owner",
                field="seed_path_by_owner",
            ),
            fixed_flat_controls=fixed,
            acs_capture_instrumentation=execution["acs_capture_instrumentation"],
            acs_checkpoint_slots=execution["acs_checkpoint_slots"],
            living_lineage_canopy=execution["living_lineage_canopy"],
            lifecycle_total_max_bytes=execution["lifecycle_total_max_bytes"],
            schema_version=SchemaVersion(value["schema_version"]),
        )
        if (
            value["execution_adapter_sha256"]
            != adapter.execution_adapter_sha256().to_dict()
            or value["control_adapter_sha256"]
            != adapter.control_adapter_sha256().to_dict()
        ):
            raise ScienceQRRProjectionError("route adapter digest mismatch")
        return adapter


@dataclass(frozen=True, slots=True)
class ScienceRichToFlatProjectionV1:
    flat_request: dict[str, Any]
    projection_receipt: dict[str, Any]


def _campaign_task_seed_dimensions(
    campaign: ScienceCampaignV1, *, task_seed_owner: str
) -> tuple[str, ...]:
    dimensions: list[str] = []
    found = False
    for row in campaign.seeds.to_value():
        if row["owner"] != task_seed_owner:
            continue
        if found:
            raise ScienceQRRProjectionError("task seed owner is duplicated")
        found = True
        if row["binding_kind"] == "dimension_bound":
            dimensions.append(row["dimension_id"])
    if not found:
        raise ScienceQRRProjectionError("task seed owner is missing")
    return tuple(dimensions)


def _merge_flat_control(
    controls: dict[str, object], path: str, value: object, *, source: str
) -> None:
    if path in controls and canonical_json_bytes(
        controls[path]
    ) != canonical_json_bytes(value):
        raise ScienceQRRProjectionError(
            f"flat control path {path!r} receives conflicting values from {source}"
        )
    controls[path] = value


def _factor_semantic_cells(
    campaign: ScienceCampaignV1,
    bundle: ScienceCompilationBundleV1,
    adapter: ScienceQRRRouteAdapterV1,
) -> tuple[list[int], list[dict[str, object]]]:
    task_dimensions = set(
        _campaign_task_seed_dimensions(
            campaign, task_seed_owner=adapter.task_seed_owner
        )
    )
    control_paths = dict(adapter.control_path_by_id)
    seed_paths = dict(adapter.seed_path_by_owner)
    fixed_controls = dict(adapter.fixed_flat_controls.to_value())
    task_seeds: set[int] = set()
    configurations: dict[bytes, dict[str, object]] = {}
    observed_pairs: set[tuple[bytes, int]] = set()

    for cell in bundle.semantic_cells.to_value():
        seed_bindings = dict(cell["seed_bindings"])
        task_seed = seed_bindings.pop(adapter.task_seed_owner, None)
        if type(task_seed) is not int or task_seed < 0:
            raise ScienceQRRProjectionError("resolved task seed is invalid")
        task_seeds.add(task_seed)
        controls = dict(fixed_controls)
        for binding in cell["control_bindings"]:
            control_id = binding["control_id"]
            flat_path = control_paths.get(control_id)
            if flat_path is None:
                raise ScienceQRRProjectionError(
                    f"route adapter lacks control mapping for {control_id}"
                )
            if binding["schedule"].get("kind") != "static":
                raise ScienceQRRProjectionError(
                    f"QRR adapter cannot represent non-static control {control_id}"
                )
            _merge_flat_control(
                controls,
                flat_path,
                binding["resolved_value"],
                source=f"control:{control_id}",
            )
        for owner, seed in sorted(seed_bindings.items()):
            flat_path = seed_paths.get(owner)
            if flat_path is None:
                raise ScienceQRRProjectionError(
                    f"route adapter lacks seed mapping for {owner}"
                )
            _merge_flat_control(controls, flat_path, seed, source=f"seed-owner:{owner}")
        if not controls:
            raise ScienceQRRProjectionError(
                "flat QRR configuration cannot be empty; adapter needs a fixed binding"
            )
        configuration_core = {
            "dimension_bindings": {
                key: value
                for key, value in cell["dimension_bindings"].items()
                if key not in task_dimensions
            },
            "seed_bindings": seed_bindings,
            "control_bindings": cell["control_bindings"],
            "flat_controls": dict(sorted(controls.items())),
        }
        configuration_key = canonical_json_bytes(configuration_core)
        pair = (configuration_key, task_seed)
        if pair in observed_pairs:
            raise ScienceQRRProjectionError("semantic cell factorization is duplicated")
        observed_pairs.add(pair)
        configurations.setdefault(configuration_key, configuration_core)

    expected_pairs = {
        (configuration_key, task_seed)
        for configuration_key in configurations
        for task_seed in task_seeds
    }
    if observed_pairs != expected_pairs:
        raise ScienceQRRProjectionError(
            "semantic matrix cannot be factored into QRR configurations crossed with task seeds"
        )
    rows = [
        {
            "id": f"science-{canonical_sha256(core).hex[:24]}",
            "controls": core["flat_controls"],
        }
        for _, core in sorted(configurations.items())
    ]
    if len({row["id"] for row in rows}) != len(rows):
        raise ScienceQRRProjectionError("flat configuration identity collision")
    return sorted(task_seeds), rows


def project_rich_science_request_to_flat_v2(
    normalized_rich_request: dict[str, Any],
    campaign: ScienceCampaignV1,
    bundle: ScienceCompilationBundleV1,
    *,
    route_binding: ScienceRouteBindingV1,
    route_adapter: ScienceQRRRouteAdapterV1,
) -> ScienceRichToFlatProjectionV1:
    """Lower one already-compiled rich request without invoking QRR or a provider."""

    if (
        normalized_rich_request.get("science_campaign")
        != campaign.to_campaign_document()
    ):
        raise ScienceQRRProjectionError("normalized request and campaign differ")
    if bundle.campaign_spec_sha256 != campaign.campaign_spec_sha256():
        raise ScienceQRRProjectionError("compilation bundle and campaign differ")
    semantic_core = bundle.execution_plan.semantic_acquisition_core.to_value()
    if (
        not isinstance(semantic_core, dict)
        or semantic_core.get("cells") != bundle.semantic_cells.to_value()
    ):
        raise ScienceQRRProjectionError(
            "compilation semantic-cell sidecar differs from the execution plan"
        )
    if (
        bundle.execution_plan.route_binding_sha256
        != route_binding.route_binding_sha256()
    ):
        raise ScienceQRRProjectionError("compilation bundle and route binding differ")
    if route_adapter.route_id != route_binding.route_id:
        raise ScienceQRRProjectionError("route adapter and route binding IDs differ")
    if route_adapter.machine_profile_id != route_binding.machine_profile_id:
        raise ScienceQRRProjectionError("route adapter machine profile differs")
    if (
        route_adapter.execution_adapter_sha256()
        != route_binding.execution_adapter_sha256
    ):
        raise ScienceQRRProjectionError("route execution adapter identity differs")
    if route_adapter.control_adapter_sha256() != route_binding.control_adapter_sha256:
        raise ScienceQRRProjectionError("route control adapter identity differs")

    world = campaign.world.to_value()
    geometry = world.get("geometry")
    if (
        world.get("world_profile_id") != route_adapter.world_profile_id
        or not isinstance(geometry, dict)
        or geometry.get("width") != route_adapter.world_width
        or geometry.get("height") != route_adapter.world_width
    ):
        raise ScienceQRRProjectionError("campaign world differs from route authority")
    execution = campaign.execution.to_value()
    cells = bundle.semantic_cells.to_value()
    logical_world_count = execution.get("logical_world_count")
    if (
        type(logical_world_count) is not int
        or logical_world_count != len(cells)
        or logical_world_count > route_adapter.maximum_batch_cells
    ):
        raise ScienceQRRProjectionError(
            "logical world count differs from the admitted route envelope"
        )
    time = campaign.time.to_value()
    if time.get("start_next_tick") != 0 or time.get("restore") is not None:
        raise ScienceQRRProjectionError(
            "current flat QRR projection cannot represent a rich continuation"
        )
    if time.get("branches", []):
        raise ScienceQRRProjectionError(
            "scientific checkpoint forks must fail admission before QRR projection"
        )
    checkpoint_ticks = list(time.get("checkpoint_ticks", []))
    if not checkpoint_ticks or checkpoint_ticks[-1] != time["total_ticks"]:
        raise ScienceQRRProjectionError(
            "rich QRR projection requires an explicit terminal checkpoint tick"
        )
    task_seeds, configurations = _factor_semantic_cells(campaign, bundle, route_adapter)
    lifecycle = lifecycle_profile_contract_v1(
        "SCIENCE_COMPACT",
        total_max_bytes=route_adapter.lifecycle_total_max_bytes,
        checkpoint_schedule=checkpoint_ticks,
        checkpoint_authority="absolute_tick",
    )
    raw_flat: dict[str, Any] = {
        "schema_version": "eiel_production_science_request_v2",
        "request_id": normalized_rich_request["request_id"],
        "world_width": route_adapter.world_width,
        "batch_cells": logical_world_count,
        "total_ticks": time["total_ticks"],
        "checkpoint_ticks": checkpoint_ticks,
        "seeds": task_seeds,
        "configurations": configurations,
        "deliberate_restore_ticks": [],
        "observability": {"living_lineage_canopy": route_adapter.living_lineage_canopy},
        "artifact_lifecycle": lifecycle,
        "baseline": normalized_rich_request["baseline"],
        **(
            {"acs_capture_instrumentation": True}
            if route_adapter.acs_capture_instrumentation
            else {}
        ),
        **(
            {"acs_checkpoint_slots": route_adapter.acs_checkpoint_slots}
            if route_adapter.acs_checkpoint_slots is not None
            else {}
        ),
    }
    from engine.qrr.production_front_door_v1 import (  # local to avoid import cycle
        validate_production_science_request_v2,
    )

    flat_request = validate_production_science_request_v2(raw_flat)
    receipt_core = {
        "schema_version": SCIENCE_RICH_TO_FLAT_PROJECTION_RECEIPT_SCHEMA_VERSION,
        "rich_request_sha256": normalized_rich_request["request_sha256"],
        "flat_request_sha256": flat_request["request_sha256"],
        "campaign_spec_sha256": campaign.campaign_spec_sha256().to_dict(),
        "instrument_manifest_sha256": bundle.execution_plan.instrument_manifest_sha256.to_dict(),
        "route_binding_sha256": route_binding.route_binding_sha256().to_dict(),
        "route_adapter_sha256": route_adapter.adapter_sha256().to_dict(),
        "compilation_bundle_sha256": bundle.compilation_bundle_sha256().to_dict(),
        "science_semantic_acquisition_sha256": bundle.execution_plan.science_semantic_acquisition_sha256().to_dict(),
        "physical_execution_plan_sha256": bundle.execution_plan.physical_execution_plan_sha256().to_dict(),
        "science_execution_plan_sha256": bundle.execution_plan.science_execution_plan_sha256().to_dict(),
        "reduction_plan_sha256": bundle.reduction_plan.reduction_plan_sha256().to_dict(),
        "lifecycle_contract_sha256": canonical_sha256(lifecycle).to_dict(),
        "projection_rule": "semantic_cells_factorized_to_qrr_configurations_x_task_environment_seeds_v1",
        "effects_performed": False,
    }
    receipt = {
        **receipt_core,
        "receipt_sha256": canonical_sha256(receipt_core).to_dict(),
    }
    return ScienceRichToFlatProjectionV1(
        flat_request=flat_request,
        projection_receipt=receipt,
    )


def bind_science_qrr_resolution_v1(
    projection: ScienceRichToFlatProjectionV1,
    active_qrr_resolution_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Bind unchanged QRR authority to additive science sidecar identities."""

    supplied = active_qrr_resolution_receipt.get("receipt_sha256")
    core_resolution = dict(active_qrr_resolution_receipt)
    core_resolution.pop("receipt_sha256", None)
    if (
        active_qrr_resolution_receipt.get("schema_version")
        != "eiel_active_qrr_resolution_receipt_v1"
        or active_qrr_resolution_receipt.get("status") != "PASS"
        or active_qrr_resolution_receipt.get("science_request_sha256")
        != projection.flat_request.get("request_sha256")
        or supplied != canonical_sha256(core_resolution).hex
    ):
        raise ScienceQRRProjectionError("active QRR resolution receipt is invalid")
    rich_receipt = projection.projection_receipt
    receipt_core = {
        "schema_version": SCIENCE_QRR_BINDING_RECEIPT_SCHEMA_VERSION,
        "rich_to_flat_projection_receipt_sha256": rich_receipt["receipt_sha256"],
        "active_qrr_resolution_receipt_sha256": supplied,
        "active_qrr_manifest_sha256": active_qrr_resolution_receipt.get(
            "manifest_sha256"
        ),
        "rich_request_sha256": rich_receipt["rich_request_sha256"],
        "flat_request_sha256": rich_receipt["flat_request_sha256"],
        "instrument_manifest_sha256": rich_receipt["instrument_manifest_sha256"],
        "route_binding_sha256": rich_receipt["route_binding_sha256"],
        "compilation_bundle_sha256": rich_receipt["compilation_bundle_sha256"],
        "science_semantic_acquisition_sha256": rich_receipt[
            "science_semantic_acquisition_sha256"
        ],
        "physical_execution_plan_sha256": rich_receipt[
            "physical_execution_plan_sha256"
        ],
        "science_execution_plan_sha256": rich_receipt["science_execution_plan_sha256"],
        "lifecycle_contract_sha256": rich_receipt["lifecycle_contract_sha256"],
        "effects_performed": False,
    }
    return {
        **receipt_core,
        "receipt_sha256": canonical_sha256(receipt_core).to_dict(),
    }


__all__ = (
    "SCIENCE_QRR_BINDING_RECEIPT_SCHEMA_VERSION",
    "SCIENCE_QRR_CONTROL_ADAPTER_CORE_SCHEMA_VERSION",
    "SCIENCE_QRR_EXECUTION_ADAPTER_CORE_SCHEMA_VERSION",
    "SCIENCE_QRR_ROUTE_ADAPTER_SCHEMA_VERSION",
    "SCIENCE_RICH_TO_FLAT_PROJECTION_RECEIPT_SCHEMA_VERSION",
    "ScienceQRRProjectionError",
    "ScienceQRRRouteAdapterV1",
    "ScienceRichToFlatProjectionV1",
    "bind_science_qrr_resolution_v1",
    "project_rich_science_request_to_flat_v2",
)
