"""Descriptor-only projection of current production component ownership.

The current component registry and native route selection remain authoritative.
This module never imports or invokes native mechanics factories and exposes no
execution hooks.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from emergence_contracts import (
    ArtifactId,
    AssemblyId,
    BackendId,
    CompatibilityBindingV1,
    ComponentCompatibilityV1,
    ComponentDescriptorV1,
    ComponentId,
    ConfigurationFragmentRefV1,
    ContentDigest,
    ContractError,
    ContractErrorCategory,
    ModuleId,
    NativeBindingV1,
    RunArtifactRefV1,
    SchemaVersion,
    SystemAssemblyV1,
)


LEGACY_CONFIGURATION_SCHEMA = SchemaVersion("eiel_production_engine_configuration_v1")
LEGACY_LIFECYCLE_SCHEMA = SchemaVersion("eiel_production_engine_lifecycle_control_v2")


def _error(message: str, *, details: Mapping[str, object] | None = None) -> ContractError:
    return ContractError(
        ContractErrorCategory.CAPABILITY_UNSUPPORTED,
        owner="eiel_composition.adapters.legacy_components",
        message=message,
        details=details,
    )


def _legacy_component_api() -> tuple[object, object, object, type[Exception]]:
    import engine

    engine_root = str(Path(engine.__file__).resolve().parent)
    if engine_root not in sys.path:
        sys.path.insert(0, engine_root)
    from engine.campaigns.production_engine_components import (
        production_domain_components_v1,
        production_engine_component_binding_v1,
    )
    from engine.campaigns.production_engine_configuration import (
        production_engine_configuration_v1,
    )
    from engine.campaigns.providers import CampaignProviderError

    return (
        production_domain_components_v1,
        production_engine_component_binding_v1,
        production_engine_configuration_v1,
        CampaignProviderError,
    )


def _descriptor(component: object, *, module_id: str, native_binding_id: str | None) -> ComponentDescriptorV1:
    required = ("component_id", "domain", "contract_version", "implementation", "production_ready")
    if any(not hasattr(component, field) for field in required):
        raise _error("legacy component descriptor is malformed")
    if not component.production_ready:
        raise _error(
            "legacy component is not production ready",
            details={"component_id": str(component.component_id)},
        )
    native_ids = () if native_binding_id is None else (ArtifactId(native_binding_id),)
    return ComponentDescriptorV1(
        component_id=ComponentId(component.component_id),
        module_id=ModuleId(module_id),
        semantic_contract_version=SchemaVersion(component.contract_version),
        implementation_version=SchemaVersion(component.component_id),
        capabilities=tuple(sorted((
            "descriptor_only",
            f"legacy_domain.{component.domain}",
            f"legacy_implementation.{component.implementation}",
            "production_ready",
        ))),
        native_binding_ids=native_ids,
        compatibility=ComponentCompatibilityV1(
            semantic_contract_versions=(SchemaVersion(component.contract_version),),
        ),
    )


class LegacyWorldDescriptorAdapter:
    def describe(self, component: object, *, native_binding_id: str | None = None) -> ComponentDescriptorV1:
        if getattr(component, "domain", None) != "world":
            raise _error("world adapter requires a world component")
        return _descriptor(component, module_id="worldkit", native_binding_id=native_binding_id)


class LegacyAgentDescriptorAdapter:
    def describe(self, component: object, *, native_binding_id: str | None = None) -> ComponentDescriptorV1:
        if getattr(component, "domain", None) != "controller":
            raise _error("agent adapter requires a controller component")
        return _descriptor(component, module_id="agentkit", native_binding_id=native_binding_id)


class LegacyAdaptationDescriptorAdapter:
    def describe(self, component: object, *, native_binding_id: str | None = None) -> ComponentDescriptorV1:
        if getattr(component, "domain", None) != "genome_inheritance":
            raise _error("adaptation adapter requires a genome-inheritance component")
        return _descriptor(component, module_id="adaptation", native_binding_id=native_binding_id)


_NATIVE_SELECTIONS = {
    ("patch", (), "fixed_mlp", False): "patch_fixed_mlp_native_binding_v1",
    ("patch", ("eed_resource_impact",), "fixed_mlp", True): "eed_fixed_mlp_native_binding_v1",
    ("patch", ("wdl_delayed_hazard",), "fixed_mlp", False): "wdl_fixed_mlp_native_binding_v1",
    ("patch", ("jungle",), "fixed_mlp", False): "jungle_fixed_mlp_native_binding_v1",
    ("patch", ("jungle", "wdl_delayed_hazard"), "fixed_mlp", True): "wdl_jungle_fixed_mlp_native_binding_v1",
    ("patch", ("eed_resource_impact", "jungle", "wdl_delayed_hazard"), "fixed_mlp", True): "wdl_jungle_eed_fixed_mlp_native_binding_v1",
    ("patch", (), "stable_gru", False): "patch_stable_gru_native_binding_v1",
    ("patch", ("wdl_delayed_hazard",), "stable_gru", False): "wdl_stable_gru_native_binding_v1",
    ("patch", ("gmnw_memory_necessity",), "gmnw_policy_arms", False): "gmnw_policy_arms_native_binding_v1",
}


def _select_native_binding_id(configuration: object) -> str:
    has_genome_v2 = any(
        name.startswith("genome_upgrade_v2.") for name in configuration.parameter_names
    )
    # The two Genome V2 fused routes require the positive discriminator; the
    # other current rows are independent of that profile.
    key = (
        configuration.world_base,
        tuple(configuration.world_overlays),
        configuration.controller_family,
        has_genome_v2 if tuple(configuration.world_overlays) in {
            ("eed_resource_impact",),
            ("jungle", "wdl_delayed_hazard"),
            ("eed_resource_impact", "jungle", "wdl_delayed_hazard"),
        } else False,
    )
    try:
        return _NATIVE_SELECTIONS[key]
    except KeyError as exc:
        raise _error(
            "current production engine has no admitted native binding",
            details={"selection": repr(key)},
        ) from exc


class LegacyNativeBindingDescriptorAdapter:
    """Project declared native artifact identity without touching its bytes."""

    def describe_many(
        self,
        *,
        plan: Mapping[str, object],
        binding_id: str,
        components: tuple[object, ...],
    ) -> tuple[NativeBindingV1, ...]:
        artifacts = plan.get("artifacts")
        qualification = plan.get("qualification_key")
        if not isinstance(artifacts, Mapping) or not isinstance(qualification, Mapping):
            raise _error("compiled plan lacks native artifact declarations")
        artifact = artifacts.get("cuda_extension")
        manifest = artifacts.get("extension_manifest")
        if not isinstance(artifact, Mapping) or not isinstance(manifest, Mapping):
            raise _error("compiled plan lacks native artifact and manifest rows")
        try:
            artifact_digest = ContentDigest("sha256", artifact["sha256"])
            manifest_digest = ContentDigest("sha256", manifest["sha256"])
            abi = SchemaVersion(artifact["abi"])
            backend = BackendId(qualification["backend"])
            locator = artifact["path"]
            artifact_kind = artifact["kind"]
        except (KeyError, TypeError, ContractError) as exc:
            raise _error("compiled plan has malformed native artifact declarations") from exc
        artifact_ref = RunArtifactRefV1(
            artifact_id=ArtifactId(f"legacy.{binding_id}.artifact"),
            artifact_kind=artifact_kind,
            artifact_schema_version=abi,
            locator=locator,
            content_digest=artifact_digest,
            size_bytes=None,
        )
        flags = tuple(sorted((
            "descriptor_only",
            "fused_physical_execution",
            f"physical_binding_id.{binding_id}",
        )))
        return tuple(
            NativeBindingV1(
                binding_id=ArtifactId(binding_id),
                semantic_component_id=ComponentId(component.component_id),
                semantic_contract_version=SchemaVersion(component.contract_version),
                backend_id=backend,
                abi_version=abi,
                artifact_ref=artifact_ref,
                artifact_sha256=artifact_digest,
                manifest_sha256=manifest_digest,
                compiled_source_identity=None,
                supported_state_schema_versions=(),
                supported_parameter_contract=LEGACY_CONFIGURATION_SCHEMA,
                capability_flags=flags,
            )
            for component in components
        )


@dataclass(frozen=True, slots=True)
class LegacyAssemblyParityV1:
    configuration_sha256: str
    component_binding_sha256: str
    native_binding_id: str


class EielLegacyAssemblyFactory:
    """Build a non-executable C-070 view from an authoritative compiled plan."""

    def __init__(self) -> None:
        self._world = LegacyWorldDescriptorAdapter()
        self._agent = LegacyAgentDescriptorAdapter()
        self._adaptation = LegacyAdaptationDescriptorAdapter()
        self._native = LegacyNativeBindingDescriptorAdapter()

    @staticmethod
    def production_component_descriptors() -> tuple[ComponentDescriptorV1, ...]:
        registry, _binding, _configuration, _error_type = _legacy_component_api()
        adapters = {
            "world": LegacyWorldDescriptorAdapter(),
            "controller": LegacyAgentDescriptorAdapter(),
            "genome_inheritance": LegacyAdaptationDescriptorAdapter(),
        }
        return tuple(adapters[row.domain].describe(row) for row in registry())

    def describe(self, plan: Mapping[str, object]) -> tuple[SystemAssemblyV1, LegacyAssemblyParityV1]:
        _registry, resolve_binding, resolve_configuration, legacy_error = (
            _legacy_component_api()
        )
        try:
            configuration = resolve_configuration(plan)
            binding = resolve_binding(configuration)
        except legacy_error as exc:
            # Current failures remain the cause, while the public plane uses a
            # stable machine-readable category.
            raise _error("authoritative component resolution failed", details={"error": str(exc)}) from exc
        native_id = _select_native_binding_id(configuration)
        semantic_components = (
            binding.world_base,
            *binding.world_overlays,
            binding.controller,
            binding.genome_inheritance,
        )
        native_bindings = self._native.describe_many(
            plan=plan, binding_id=native_id, components=semantic_components
        )
        runtime_profile = ComponentDescriptorV1(
            component_id=ComponentId(str(plan.get("resolved_profile") or "eiel-production-runtime-v1")),
            module_id=ModuleId("runtime"),
            semantic_contract_version=LEGACY_LIFECYCLE_SCHEMA,
            implementation_version=SchemaVersion("eiel-production-provider-v1"),
            capabilities=("descriptor_only", "legacy_unified_production_engine"),
            compatibility=ComponentCompatibilityV1(
                semantic_contract_versions=(LEGACY_LIFECYCLE_SCHEMA,),
            ),
        )
        assembly = SystemAssemblyV1(
            assembly_id=AssemblyId(f"legacy.{binding.binding_sha256}"),
            runtime_profile=runtime_profile,
            world_components=tuple(
                self._world.describe(row, native_binding_id=native_id)
                for row in (binding.world_base, *binding.world_overlays)
            ),
            agent_component=self._agent.describe(binding.controller, native_binding_id=native_id),
            adaptation_component=self._adaptation.describe(
                binding.genome_inheritance, native_binding_id=native_id
            ),
            measurement_components=(),
            native_bindings=native_bindings,
            configuration_fragments=(
                ConfigurationFragmentRefV1(
                    owner_module_id=ModuleId("eiel"),
                    schema_version=LEGACY_CONFIGURATION_SCHEMA,
                    content_digest=ContentDigest("sha256", configuration.configuration_sha256),
                ),
            ),
            compatibility_bindings=(
                CompatibilityBindingV1(
                    consumer_module_id=ModuleId("eiel"),
                    provider_module_id=ModuleId("runtime"),
                    contract_version=LEGACY_LIFECYCLE_SCHEMA,
                ),
            ),
        )
        return assembly, LegacyAssemblyParityV1(
            configuration_sha256=configuration.configuration_sha256,
            component_binding_sha256=binding.binding_sha256,
            native_binding_id=native_id,
        )
