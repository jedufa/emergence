"""C-010 Component Descriptor v1 foundation."""

from __future__ import annotations

from dataclasses import dataclass, field

from .canonical import canonical_json_bytes, canonical_sha256
from .errors import contract_invalid
from .ids import ArtifactId, ComponentId, ContentDigest, ModuleId, SchemaVersion


COMPONENT_DESCRIPTOR_SCHEMA_VERSION = SchemaVersion("component_descriptor_v1")


def _validate_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise contract_invalid(
            "value must be a non-empty, unpadded string",
            field=field_name,
            value_type=type(value).__name__,
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise contract_invalid("value contains a control character", field=field_name)
    return value


def _validate_sorted_unique(values: tuple[object, ...], *, field_name: str) -> None:
    if not isinstance(values, tuple):
        raise contract_invalid(
            "canonical descriptor collections must be tuples",
            field=field_name,
            value_type=type(values).__name__,
        )
    rendered = [str(value) for value in values]
    if rendered != sorted(set(rendered)):
        raise contract_invalid(
            "canonical descriptor collections must be sorted and unique",
            field=field_name,
        )


@dataclass(frozen=True, slots=True)
class ComponentCompatibilityV1:
    """Explicit compatibility claims; an empty tuple grants no compatibility."""

    semantic_contract_versions: tuple[SchemaVersion, ...] = ()
    state_schema_versions: tuple[SchemaVersion, ...] = ()
    artifact_schema_versions: tuple[SchemaVersion, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "semantic_contract_versions",
            "state_schema_versions",
            "artifact_schema_versions",
        ):
            values = getattr(self, field_name)
            _validate_sorted_unique(values, field_name=f"compatibility.{field_name}")
            if any(not isinstance(value, SchemaVersion) for value in values):
                raise contract_invalid(
                    "compatibility versions must be SchemaVersion values",
                    field=f"compatibility.{field_name}",
                )

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "semantic_contract_versions": [str(value) for value in self.semantic_contract_versions],
            "state_schema_versions": [str(value) for value in self.state_schema_versions],
            "artifact_schema_versions": [str(value) for value in self.artifact_schema_versions],
        }

    @classmethod
    def from_dict(cls, value: object) -> "ComponentCompatibilityV1":
        if not isinstance(value, dict) or set(value) != {
            "semantic_contract_versions",
            "state_schema_versions",
            "artifact_schema_versions",
        }:
            raise contract_invalid(
                "component compatibility has an invalid shape", field="compatibility"
            )
        return cls(
            semantic_contract_versions=tuple(
                SchemaVersion(item) for item in value["semantic_contract_versions"]
            ),
            state_schema_versions=tuple(
                SchemaVersion(item) for item in value["state_schema_versions"]
            ),
            artifact_schema_versions=tuple(
                SchemaVersion(item) for item in value["artifact_schema_versions"]
            ),
        )


@dataclass(frozen=True, slots=True)
class ComponentDescriptorV1:
    """Immutable semantic component identity; never an implementation handle."""

    component_id: ComponentId
    module_id: ModuleId
    semantic_contract_version: SchemaVersion
    implementation_version: SchemaVersion
    capabilities: tuple[str, ...] = ()
    parameter_schema_ref: str | None = None
    state_schema_refs: tuple[str, ...] = ()
    observation_schema_refs: tuple[str, ...] = ()
    action_schema_refs: tuple[str, ...] = ()
    native_binding_ids: tuple[ArtifactId, ...] = ()
    compatibility: ComponentCompatibilityV1 = field(default_factory=ComponentCompatibilityV1)
    schema_version: SchemaVersion = COMPONENT_DESCRIPTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        expected_types = {
            "component_id": ComponentId,
            "module_id": ModuleId,
            "semantic_contract_version": SchemaVersion,
            "implementation_version": SchemaVersion,
            "compatibility": ComponentCompatibilityV1,
            "schema_version": SchemaVersion,
        }
        for field_name, expected_type in expected_types.items():
            if not isinstance(getattr(self, field_name), expected_type):
                raise contract_invalid(
                    f"value must be {expected_type.__name__}", field=field_name
                )
        if self.schema_version != COMPONENT_DESCRIPTOR_SCHEMA_VERSION:
            raise contract_invalid(
                "unsupported Component Descriptor schema version", field="schema_version"
            )
        for field_name in (
            "capabilities",
            "state_schema_refs",
            "observation_schema_refs",
            "action_schema_refs",
            "native_binding_ids",
        ):
            _validate_sorted_unique(getattr(self, field_name), field_name=field_name)
        for capability in self.capabilities:
            _validate_text(capability, field_name="capabilities")
        for field_name in (
            "state_schema_refs",
            "observation_schema_refs",
            "action_schema_refs",
        ):
            for reference in getattr(self, field_name):
                _validate_text(reference, field_name=field_name)
        if any(not isinstance(value, ArtifactId) for value in self.native_binding_ids):
            raise contract_invalid(
                "native binding IDs must be ArtifactId values", field="native_binding_ids"
            )
        if self.parameter_schema_ref is not None:
            _validate_text(self.parameter_schema_ref, field_name="parameter_schema_ref")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "component_id": str(self.component_id),
            "module_id": str(self.module_id),
            "semantic_contract_version": str(self.semantic_contract_version),
            "implementation_version": str(self.implementation_version),
            "capabilities": list(self.capabilities),
            "parameter_schema_ref": self.parameter_schema_ref,
            "state_schema_refs": list(self.state_schema_refs),
            "observation_schema_refs": list(self.observation_schema_refs),
            "action_schema_refs": list(self.action_schema_refs),
            "native_binding_ids": [str(value) for value in self.native_binding_ids],
            "compatibility": self.compatibility.to_dict(),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def content_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "ComponentDescriptorV1":
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "component_id",
            "module_id",
            "semantic_contract_version",
            "implementation_version",
            "capabilities",
            "parameter_schema_ref",
            "state_schema_refs",
            "observation_schema_refs",
            "action_schema_refs",
            "native_binding_ids",
            "compatibility",
        }:
            raise contract_invalid(
                "component descriptor has an invalid shape", field="component_descriptor"
            )
        return cls(
            component_id=ComponentId(value["component_id"]),
            module_id=ModuleId(value["module_id"]),
            semantic_contract_version=SchemaVersion(value["semantic_contract_version"]),
            implementation_version=SchemaVersion(value["implementation_version"]),
            capabilities=tuple(value["capabilities"]),
            parameter_schema_ref=value["parameter_schema_ref"],
            state_schema_refs=tuple(value["state_schema_refs"]),
            observation_schema_refs=tuple(value["observation_schema_refs"]),
            action_schema_refs=tuple(value["action_schema_refs"]),
            native_binding_ids=tuple(
                ArtifactId(item) for item in value["native_binding_ids"]
            ),
            compatibility=ComponentCompatibilityV1.from_dict(value["compatibility"]),
            schema_version=SchemaVersion(value["schema_version"]),
        )
