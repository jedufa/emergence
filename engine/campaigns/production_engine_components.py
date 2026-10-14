"""Domain component identities consumed by the single production engine.

Components describe and implement mechanics within one domain. They are not
execution providers and may not own batching, lifecycle, persistence,
Chronicle, reporting, analysis, or qualification.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from .canonical import canonical_sha256
from .production_engine_configuration import ProductionEngineConfigurationV1
from .providers import CampaignProviderError


SCHEMA_VERSION: Final[str] = "eiel_production_engine_component_binding_v1"
ENGINE_LIFECYCLE_OWNER: Final[str] = "eiel_production_engine_v1"
GLOBAL_GENOME_OWNER: Final[str] = "eiel_global_genome_inheritance_v1"


@dataclass(frozen=True, slots=True)
class DomainComponentV1:
    """One immutable domain mechanic identity with no execution ownership."""

    component_id: str
    domain: str
    contract_version: str
    implementation: str
    production_ready: bool
    configuration_id: str | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "domain": self.domain,
            "contract_version": self.contract_version,
            "implementation": self.implementation,
            "production_ready": self.production_ready,
            "configuration_id": self.configuration_id,
        }


@dataclass(frozen=True, slots=True)
class ProductionEngineComponentBindingV1:
    """Canonical domain binding beneath the invariant engine identity."""

    world_base: DomainComponentV1
    world_overlays: tuple[DomainComponentV1, ...]
    controller: DomainComponentV1
    genome_inheritance: DomainComponentV1

    def payload(self) -> dict[str, Any]:
        from engine.evolution.global_genome_registry_v1 import (
            global_genome_registry_v1,
        )

        genome_registry = global_genome_registry_v1()
        return {
            "schema_version": SCHEMA_VERSION,
            "execution_owner": ENGINE_LIFECYCLE_OWNER,
            "world": {
                "base": self.world_base.payload(),
                "overlays": [row.payload() for row in self.world_overlays],
            },
            "controller": self.controller.payload(),
            "genome_inheritance": {
                **self.genome_inheritance.payload(),
                "owner": GLOBAL_GENOME_OWNER,
                "registry_schema_version": genome_registry["schema_version"],
                "registry_sha256": canonical_sha256(genome_registry),
            },
        }

    @property
    def binding_sha256(self) -> str:
        return canonical_sha256(self.payload())

    @property
    def unmigrated_components(self) -> tuple[str, ...]:
        components = (
            self.world_base,
            *self.world_overlays,
            self.controller,
            self.genome_inheritance,
        )
        return tuple(
            component.component_id
            for component in components
            if not component.production_ready
        )


_WORLD_BASES: Final[dict[str, DomainComponentV1]] = {
    "patch": DomainComponentV1(
        component_id="world.patch.v1",
        domain="world",
        contract_version="patch_world_mechanics_v1",
        implementation="engine.env.gridworld_pkg.patch_world",
        production_ready=True,
    ),
    "inheritance_cross_path_neutral": DomainComponentV1(
        component_id="world.inheritance_cross_path_neutral.v1",
        domain="world",
        contract_version="neutral_inheritance_audit_world_v1",
        implementation=(
            "engine.campaigns.inheritance_cross_path_patch_execution_v1"
        ),
        production_ready=False,
    ),
    "recomb72_neutral": DomainComponentV1(
        component_id="world.recomb72_neutral.v1",
        domain="world",
        contract_version="historical_recomb72_neutral_world_v1",
        implementation="engine.campaigns.recomb72",
        production_ready=False,
    ),
}

_WORLD_OVERLAYS: Final[dict[str, DomainComponentV1]] = {
    "eed_resource_impact": DomainComponentV1(
        component_id="world.overlay.eed_resource_impact.v1",
        domain="world",
        contract_version="eed_resource_impact_world_mechanics_v1",
        implementation="engine.env.gridworld_pkg.patch_world",
        production_ready=True,
    ),
    "wdl_delayed_hazard": DomainComponentV1(
        component_id="world.overlay.wdl_delayed_hazard.v1",
        domain="world",
        contract_version="wdl_delayed_hazard_world_mechanics_v1",
        implementation="engine.env.gridworld_pkg.patch_world",
        production_ready=True,
    ),
    "jungle": DomainComponentV1(
        component_id="world.overlay.jungle.v1",
        domain="world",
        contract_version="jungle_world_mechanics_v1",
        implementation="engine.env.gridworld_pkg.jungle",
        production_ready=True,
    ),
    "gmnw_memory_necessity": DomainComponentV1(
        component_id="world.overlay.gmnw_memory_necessity.v1",
        domain="world",
        contract_version="gmnw_memory_necessity_world_mechanics_v1",
        implementation="engine.evolution.gmnw_v1",
        production_ready=True,
    ),
}

_CONTROLLERS: Final[dict[str, DomainComponentV1]] = {
    "fixed_mlp": DomainComponentV1(
        component_id="controller.fixed_mlp.v1",
        domain="controller",
        contract_version="fixed_mlp_controller_v1",
        implementation="engine.perf.cuda_architecture_b_fixed_mlp",
        production_ready=True,
    ),
    "stable_gru": DomainComponentV1(
        component_id="controller.stable_gru.v1",
        domain="controller",
        contract_version="stable_gru_controller_v3",
        implementation="engine.perf.recurrent_native.stable_gru_cpu_artifact_v1",
        production_ready=True,
    ),
    "gmnw_policy_arms": DomainComponentV1(
        component_id="controller.gmnw_policy_arms.v1",
        domain="controller",
        contract_version="gmnw_policy_arms_v1",
        implementation="engine.perf.cuda_architecture_b_direct_resident_gmnw_gru_v1",
        production_ready=True,
    ),
    "audit_only": DomainComponentV1(
        component_id="controller.audit_only.v1",
        domain="controller",
        contract_version="no_cognition_audit_controller_v1",
        implementation="engine.campaigns.inheritance_cross_path_patch_execution_v1",
        production_ready=False,
    ),
}

_GENOME_INHERITANCE: Final[dict[str, DomainComponentV1]] = {
    "patch_default": DomainComponentV1(
        component_id="genome.patch_default.v1",
        domain="genome_inheritance",
        contract_version="patch_genome_inheritance_mutation_v1",
        implementation="engine.evolution.global_genome_registry_v1",
        production_ready=True,
        configuration_id="patch_default",
    ),
    "inheritance_cross_path": DomainComponentV1(
        component_id="genome.inheritance_cross_path.v2",
        domain="genome_inheritance",
        contract_version="inheritance_cross_path_registry_v2",
        implementation="engine.evolution.global_genome_registry_v1",
        production_ready=False,
        configuration_id="inheritance_cross_path",
    ),
    "genome_upgrade_v2": DomainComponentV1(
        component_id="genome.genome_upgrade_v2.v1",
        domain="genome_inheritance",
        contract_version="genome_upgrade_v2_fixed_q10_v1",
        implementation="engine.evolution.global_genome_registry_v1",
        production_ready=True,
        configuration_id="genome_upgrade_v2",
    ),
    "recomb72": DomainComponentV1(
        component_id="genome.recomb72_historical.v1",
        domain="genome_inheritance",
        contract_version="recomb72_frozen_inheritance_v1",
        implementation="engine.evolution.global_genome_registry_v1",
        production_ready=False,
        configuration_id="recomb72",
    ),
}


def production_domain_components_v1() -> tuple[DomainComponentV1, ...]:
    """Return every declared production-capable domain component."""

    components = (
        *_WORLD_BASES.values(),
        *_WORLD_OVERLAYS.values(),
        *_CONTROLLERS.values(),
        *_GENOME_INHERITANCE.values(),
    )
    return tuple(
        sorted(
            (
                component
                for component in components
                if component.production_ready
            ),
            key=lambda component: (component.domain, component.component_id),
        )
    )


def _require_component(
    registry: Mapping[str, DomainComponentV1],
    identity: str,
    *,
    domain: str,
) -> DomainComponentV1:
    try:
        component = registry[identity]
    except KeyError as exc:
        raise CampaignProviderError(
            f"production engine has no {domain} component for {identity!r}"
        ) from exc
    if component.domain != domain:
        raise CampaignProviderError(
            f"production engine {domain} component registry is malformed"
        )
    if not component.production_ready:
        raise CampaignProviderError(
            "P12_RETIRED_PRODUCTION_COMPONENT "
            f"domain={domain} component={component.component_id!r}"
        )
    return component


def _genome_identity(
    configuration: ProductionEngineConfigurationV1,
) -> str:
    names = set(configuration.parameter_names)
    selected = {
        identity
        for prefix, identity in (
            ("inheritance_cross_path.", "inheritance_cross_path"),
            ("genome_upgrade_v2.", "genome_upgrade_v2"),
            ("recomb72.", "recomb72"),
        )
        if any(name.startswith(prefix) for name in names)
    }
    if len(selected) > 1:
        raise CampaignProviderError(
            "production engine configuration selects multiple genome profiles"
        )
    if selected:
        return selected.pop()
    return "patch_default"


def production_engine_component_binding_v1(
    configuration: ProductionEngineConfigurationV1,
) -> ProductionEngineComponentBindingV1:
    """Resolve configuration to domain mechanics without choosing an engine."""

    world_base = _require_component(
        _WORLD_BASES, configuration.world_base, domain="world"
    )
    overlays = tuple(
        _require_component(_WORLD_OVERLAYS, identity, domain="world")
        for identity in configuration.world_overlays
    )
    controller = _require_component(
        _CONTROLLERS, configuration.controller_family, domain="controller"
    )
    genome = _require_component(
        _GENOME_INHERITANCE,
        _genome_identity(configuration),
        domain="genome_inheritance",
    )
    if genome.configuration_id is None:
        raise CampaignProviderError(
            "production engine genome component lacks a global profile identity"
        )
    from engine.evolution.global_genome_registry_v1 import (
        global_genome_profile_v1,
    )

    global_genome_profile_v1(genome.configuration_id)
    return ProductionEngineComponentBindingV1(
        world_base=world_base,
        world_overlays=overlays,
        controller=controller,
        genome_inheritance=genome,
    )


__all__ = [
    "DomainComponentV1",
    "ENGINE_LIFECYCLE_OWNER",
    "GLOBAL_GENOME_OWNER",
    "ProductionEngineComponentBindingV1",
    "production_domain_components_v1",
    "SCHEMA_VERSION",
    "production_engine_component_binding_v1",
]
