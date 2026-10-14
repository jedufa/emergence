"""Fail-closed route authority for the Genome V2 C1 live CUDA canary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from engine.evolution.genome_upgrade_v2_package_v1 import (
    package_identity_sha256_v1,
)
from engine.qualification_plans.v1 import EXECUTION_ROUTE_OBLIGATIONS

from .production_engine_provider import EielProductionExecutionProviderV1
from .production_engine_identity import (
    PRODUCTION_PROFILE_ID,
    PRODUCTION_PROVIDER_ID,
)


PROVIDER_ID = PRODUCTION_PROVIDER_ID
RUNTIME_PROFILE_ID = PRODUCTION_PROFILE_ID
PATCH_NATIVE_BINDING_ID = "patch_fixed_mlp_native_binding_v1"
PATCH_NATIVE_ARTIFACT_ABI = "architecture_b_patch_p2_stage3_v2"
WDL_NATIVE_BINDING_ID = "wdl_fixed_mlp_native_binding_v1"
WDL_NATIVE_ARTIFACT_ABI = "architecture_b_wdl_fixed_mlp_v2"
SINGLE_CELL_EXECUTION_ROUTE_OBLIGATIONS = EXECUTION_ROUTE_OBLIGATIONS - frozenset(
    {
        "alternate_batch_partition",
        "multicell_execution",
        "odd_physical_tail",
    }
)
SUPPORTED_NATIVE_BINDINGS = {
    PATCH_NATIVE_BINDING_ID: (
        PATCH_NATIVE_ARTIFACT_ABI,
        (),
        (EXECUTION_ROUTE_OBLIGATIONS,),
    ),
    WDL_NATIVE_BINDING_ID: (
        WDL_NATIVE_ARTIFACT_ABI,
        ("wdl_delayed_hazard",),
        (SINGLE_CELL_EXECUTION_ROUTE_OBLIGATIONS, EXECUTION_ROUTE_OBLIGATIONS),
    ),
}
NATIVE_BINDING_ID = PATCH_NATIVE_BINDING_ID
NATIVE_ARTIFACT_ABI = PATCH_NATIVE_ARTIFACT_ABI


class GenomeV2C1RouteAuthorityError(ValueError):
    """The supplied plan and receipts do not identify the C1 production route."""


@dataclass(frozen=True, slots=True)
class GenomeV2C1RouteAuthorityV1:
    execution_route_sha256: str
    production_configuration_sha256: str
    component_binding_sha256: str
    native_binding_id: str
    native_artifact_abi: str
    package_id_sha256: str
    package_receipt_sha256: str
    cell: Mapping[str, Any]

    def payload(self) -> dict[str, Any]:
        return {
            "execution_route_sha256": self.execution_route_sha256,
            "production_configuration_sha256": (
                self.production_configuration_sha256
            ),
            "component_binding_sha256": self.component_binding_sha256,
            "native_binding_id": self.native_binding_id,
            "native_artifact_abi": self.native_artifact_abi,
            "package_id_sha256": self.package_id_sha256,
            "package_receipt_sha256": self.package_receipt_sha256,
            "cell_id": self.cell["cell_id"],
        }


def _component(
    components: Mapping[str, Mapping[str, Any]], name: str
) -> Mapping[str, Any]:
    try:
        return components[name]
    except KeyError as exc:
        raise GenomeV2C1RouteAuthorityError(
            f"execution route omitted {name}"
        ) from exc


def validate_genome_v2_c1_route_authority_v1(
    *,
    plan: Mapping[str, Any],
    package_receipt: Mapping[str, Any],
    package_receipt_sha256: str,
    route_receipt: Mapping[str, Any],
    checkpoint_authority_binding: Mapping[str, str],
) -> GenomeV2C1RouteAuthorityV1:
    """Recompute and join the exact fixed-MLP Patch production route."""

    try:
        contract = EielProductionExecutionProviderV1().execution_route_contract(plan)
        route_payload = contract.payload()
        route_sha256 = contract.execution_route_sha256
    except (TypeError, ValueError, RuntimeError) as exc:
        raise GenomeV2C1RouteAuthorityError(
            "execution plan does not materialize the production route"
        ) from exc
    components = {
        str(row.get("component_id")): row
        for row in route_payload.get("components", ())
        if isinstance(row, Mapping)
    }
    configuration = _component(components, "production_engine_configuration")
    component_binding = _component(
        components, "production_engine_component_binding"
    )
    native = _component(components, "native_mechanics_binding")
    genome = _component(components, "global_genome_profile")
    package = _component(components, "genome_upgrade_v2_executable_package")
    configuration_payload = configuration.get("configuration")
    binding_payload = component_binding.get("binding")
    genome_profile = genome.get("profile")
    covered = route_receipt.get("covered_obligations")
    if (
        contract.provider_id != PROVIDER_ID
        or contract.profile_id != RUNTIME_PROFILE_ID
        or checkpoint_authority_binding.get("provider_id") != PROVIDER_ID
        or checkpoint_authority_binding.get("runtime_profile_id")
        != RUNTIME_PROFILE_ID
        or checkpoint_authority_binding.get("execution_route_sha256")
        != route_sha256
        or route_receipt.get("execution_route_sha256") != route_sha256
        or route_receipt.get("provider_id") != PROVIDER_ID
        or route_receipt.get("profile_id") != RUNTIME_PROFILE_ID
        or route_receipt.get("authority_state") != "route_qualified"
        or not isinstance(covered, list)
        or route_receipt.get("missing_obligations", []) != []
    ):
        raise GenomeV2C1RouteAuthorityError(
            "execution plan and route receipt do not join exactly"
        )
    controller_binding = (
        binding_payload.get("controller")
        if isinstance(binding_payload, Mapping)
        else None
    )
    world_configuration = (
        configuration_payload.get("world")
        if isinstance(configuration_payload, Mapping)
        else None
    )
    controller_configuration = (
        configuration_payload.get("controller")
        if isinstance(configuration_payload, Mapping)
        else None
    )
    genome_binding = (
        binding_payload.get("genome_inheritance")
        if isinstance(binding_payload, Mapping)
        else None
    )
    native_binding_id = native.get("binding_id")
    expected_native = SUPPORTED_NATIVE_BINDINGS.get(str(native_binding_id))
    if expected_native is None:
        raise GenomeV2C1RouteAuthorityError(
            "execution plan is not a fixed-MLP Patch/WDL Genome V2 route"
        )
    (
        expected_artifact_abi,
        expected_world_overlays,
        expected_obligation_sets,
    ) = expected_native
    required_obligations = frozenset(contract.required_obligations)
    if (
        set(covered) != required_obligations
        or required_obligations not in expected_obligation_sets
    ):
        raise GenomeV2C1RouteAuthorityError(
            "execution route obligations are not admitted for this binding"
        )
    if (
        not isinstance(configuration_payload, Mapping)
        or not isinstance(world_configuration, Mapping)
        or world_configuration.get("base") != "patch"
        or tuple(world_configuration.get("overlays") or ())
        != expected_world_overlays
        or not isinstance(controller_configuration, Mapping)
        or controller_configuration.get("family") != "fixed_mlp"
        or not isinstance(controller_binding, Mapping)
        or controller_binding.get("component_id")
        != "controller.fixed_mlp.v1"
        or not isinstance(genome_binding, Mapping)
        or genome_binding.get("configuration_id")
        != "genome_upgrade_v2"
        or native.get("artifact_abi") != expected_artifact_abi
        or not isinstance(genome_profile, Mapping)
        or genome_profile.get("profile_id") != "genome_upgrade_v2"
    ):
        raise GenomeV2C1RouteAuthorityError(
            "execution plan is not a fixed-MLP Patch/WDL Genome V2 route"
        )
    cells = plan.get("cells")
    execution = plan.get("execution")
    if (
        not isinstance(execution, Mapping)
        or type(execution.get("generations")) is not int
        or execution["generations"] < 3
        or execution.get("ticks_per_generation") != 80
    ):
        raise GenomeV2C1RouteAuthorityError(
            "execution plan does not admit the fixed-MLP b1g3t80 canary"
        )
    if not isinstance(cells, list) or not cells:
        raise GenomeV2C1RouteAuthorityError("execution plan has no logical cells")
    neutral_cells = [
        row
        for row in cells
        if isinstance(row, Mapping)
        and isinstance(row.get("parameters"), Mapping)
        and row["parameters"].get("genome_upgrade_v2.founder_profile")
        == "neutral"
    ]
    if not neutral_cells:
        raise GenomeV2C1RouteAuthorityError(
            "execution plan has no neutral Genome V2 founder cell"
        )
    package_id = package_receipt.get("package_id_sha256")
    try:
        computed_package_id = package_identity_sha256_v1(package_receipt)
    except ValueError as exc:
        raise GenomeV2C1RouteAuthorityError(
            "package receipt is not a valid package identity"
        ) from exc
    if (
        package_receipt.get("ok") is not True
        or package_receipt.get("authority_state")
        != "package_candidate_complete"
        or computed_package_id != package_id
        or checkpoint_authority_binding.get("package_id_sha256") != package_id
        or checkpoint_authority_binding.get("package_receipt_sha256")
        != package_receipt_sha256
        or package.get("receipt_sha256") != package_receipt_sha256
    ):
        raise GenomeV2C1RouteAuthorityError(
            "execution plan and package receipt do not join exactly"
        )
    return GenomeV2C1RouteAuthorityV1(
        execution_route_sha256=route_sha256,
        production_configuration_sha256=str(
            configuration["configuration_sha256"]
        ),
        component_binding_sha256=str(component_binding["binding_sha256"]),
        native_binding_id=str(native_binding_id),
        native_artifact_abi=str(expected_artifact_abi),
        package_id_sha256=str(package_id),
        package_receipt_sha256=package_receipt_sha256,
        cell=neutral_cells[0],
    )


__all__ = [
    "GenomeV2C1RouteAuthorityError",
    "GenomeV2C1RouteAuthorityV1",
    "NATIVE_ARTIFACT_ABI",
    "NATIVE_BINDING_ID",
    "PROVIDER_ID",
    "RUNTIME_PROFILE_ID",
    "validate_genome_v2_c1_route_authority_v1",
]
