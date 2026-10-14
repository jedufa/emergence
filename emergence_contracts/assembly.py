"""C-070 immutable EIEL system assembly contract."""

from __future__ import annotations

from dataclasses import dataclass

from ._validation import (
    require_instance,
    require_tuple_instances,
)
from .canonical import canonical_sha256
from .components import ComponentDescriptorV1
from .errors import contract_invalid
from .ids import AssemblyId, ContentDigest, ModuleId, SchemaVersion
from .native import NativeBindingV1


CONTRACT_ID = "C-070"
CONTRACT_OWNER = "Contracts; concrete assembly owned by EIEL application"
API_VERSION = SchemaVersion("c070_v1")
SYSTEM_ASSEMBLY_SCHEMA_VERSION = SchemaVersion("system_assembly_v1")


@dataclass(frozen=True, slots=True)
class ConfigurationFragmentRefV1:
    owner_module_id: ModuleId
    schema_version: SchemaVersion
    content_digest: ContentDigest

    def __post_init__(self) -> None:
        require_instance(self.owner_module_id, ModuleId, field="owner_module_id")
        require_instance(self.schema_version, SchemaVersion, field="schema_version")
        require_instance(self.content_digest, ContentDigest, field="content_digest")

    def to_dict(self) -> dict[str, object]:
        return {
            "owner_module_id": str(self.owner_module_id),
            "schema_version": str(self.schema_version),
            "content_digest": self.content_digest.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CompatibilityBindingV1:
    consumer_module_id: ModuleId
    provider_module_id: ModuleId
    contract_version: SchemaVersion

    def __post_init__(self) -> None:
        require_instance(self.consumer_module_id, ModuleId, field="consumer_module_id")
        require_instance(self.provider_module_id, ModuleId, field="provider_module_id")
        require_instance(self.contract_version, SchemaVersion, field="contract_version")

    def to_dict(self) -> dict[str, str]:
        return {
            "consumer_module_id": str(self.consumer_module_id),
            "provider_module_id": str(self.provider_module_id),
            "contract_version": str(self.contract_version),
        }


@dataclass(frozen=True, slots=True)
class SystemAssemblyV1:
    assembly_id: AssemblyId
    runtime_profile: ComponentDescriptorV1
    world_components: tuple[ComponentDescriptorV1, ...]
    agent_component: ComponentDescriptorV1
    adaptation_component: ComponentDescriptorV1
    measurement_components: tuple[ComponentDescriptorV1, ...]
    native_bindings: tuple[NativeBindingV1, ...]
    configuration_fragments: tuple[ConfigurationFragmentRefV1, ...]
    compatibility_bindings: tuple[CompatibilityBindingV1, ...]
    schema_version: SchemaVersion = SYSTEM_ASSEMBLY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(self.assembly_id, AssemblyId, field="assembly_id")
        for field_name in (
            "runtime_profile",
            "agent_component",
            "adaptation_component",
        ):
            require_instance(
                getattr(self, field_name), ComponentDescriptorV1, field=field_name
            )
        for field_name, expected in (
            ("world_components", ComponentDescriptorV1),
            ("measurement_components", ComponentDescriptorV1),
            ("native_bindings", NativeBindingV1),
            ("configuration_fragments", ConfigurationFragmentRefV1),
            ("compatibility_bindings", CompatibilityBindingV1),
        ):
            require_tuple_instances(
                getattr(self, field_name), expected, field=field_name, unique=True
            )
        if not self.world_components:
            raise contract_invalid(
                "assembly requires at least one world component",
                field="world_components",
            )
        component_ids = [
            str(component.component_id)
            for component in (
                self.runtime_profile,
                *self.world_components,
                self.agent_component,
                self.adaptation_component,
                *self.measurement_components,
            )
        ]
        if len(component_ids) != len(set(component_ids)):
            raise contract_invalid(
                "assembly component IDs must be unique", field="components"
            )
        if self.schema_version != SYSTEM_ASSEMBLY_SCHEMA_VERSION:
            raise contract_invalid("unsupported assembly schema", field="schema_version")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "assembly_id": str(self.assembly_id),
            "runtime_profile": self.runtime_profile.to_dict(),
            "world_components": [value.to_dict() for value in self.world_components],
            "agent_component": self.agent_component.to_dict(),
            "adaptation_component": self.adaptation_component.to_dict(),
            "measurement_components": [
                value.to_dict() for value in self.measurement_components
            ],
            "native_bindings": [value.to_dict() for value in self.native_bindings],
            "configuration_fragments": [
                value.to_dict() for value in self.configuration_fragments
            ],
            "compatibility_bindings": [
                value.to_dict() for value in self.compatibility_bindings
            ],
        }

    def identity_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())
