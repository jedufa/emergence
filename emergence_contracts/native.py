"""C-170 native binding identity without loading or dispatching artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from ._validation import (
    require_instance,
    require_sorted_unique_text,
    require_tuple_instances,
)
from .artifacts import RunArtifactRefV1
from .errors import contract_invalid
from .ids import (
    ArtifactId,
    BackendId,
    ComponentId,
    ContentDigest,
    SchemaVersion,
)


CONTRACT_ID = "C-170"
CONTRACT_OWNER = "Contracts; declared by semantic owner; consumed by Runtime"
API_VERSION = SchemaVersion("c170_v1")
NATIVE_BINDING_SCHEMA_VERSION = SchemaVersion("native_binding_v1")


@dataclass(frozen=True, slots=True)
class NativeBindingV1:
    binding_id: ArtifactId
    semantic_component_id: ComponentId
    semantic_contract_version: SchemaVersion
    backend_id: BackendId
    abi_version: SchemaVersion
    artifact_ref: RunArtifactRefV1
    artifact_sha256: ContentDigest
    manifest_sha256: ContentDigest
    # Legacy manifests do not all expose a separate compiled-source digest.
    # ``None`` records that absence without fabricating an identity.
    compiled_source_identity: ContentDigest | None
    supported_state_schema_versions: tuple[SchemaVersion, ...]
    supported_parameter_contract: SchemaVersion
    capability_flags: tuple[str, ...]
    schema_version: SchemaVersion = NATIVE_BINDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name, expected in (
            ("binding_id", ArtifactId),
            ("semantic_component_id", ComponentId),
            ("semantic_contract_version", SchemaVersion),
            ("backend_id", BackendId),
            ("abi_version", SchemaVersion),
            ("artifact_ref", RunArtifactRefV1),
            ("artifact_sha256", ContentDigest),
            ("manifest_sha256", ContentDigest),
            ("supported_parameter_contract", SchemaVersion),
        ):
            require_instance(getattr(self, field_name), expected, field=field_name)
        if self.compiled_source_identity is not None:
            require_instance(
                self.compiled_source_identity,
                ContentDigest,
                field="compiled_source_identity",
            )
        require_tuple_instances(
            self.supported_state_schema_versions,
            SchemaVersion,
            field="supported_state_schema_versions",
            unique=True,
            sorted_by_string=True,
        )
        require_sorted_unique_text(self.capability_flags, field="capability_flags")
        if self.artifact_ref.content_digest != self.artifact_sha256:
            raise contract_invalid(
                "native artifact digest does not match artifact ref",
                field="artifact_sha256",
            )
        if self.schema_version != NATIVE_BINDING_SCHEMA_VERSION:
            raise contract_invalid("unsupported native binding schema", field="schema_version")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "binding_id": str(self.binding_id),
            "semantic_component_id": str(self.semantic_component_id),
            "semantic_contract_version": str(self.semantic_contract_version),
            "backend_id": str(self.backend_id),
            "abi_version": str(self.abi_version),
            "artifact_ref": self.artifact_ref.to_dict(),
            "artifact_sha256": self.artifact_sha256.to_dict(),
            "manifest_sha256": self.manifest_sha256.to_dict(),
            "compiled_source_identity": (
                None
                if self.compiled_source_identity is None
                else self.compiled_source_identity.to_dict()
            ),
            "supported_state_schema_versions": [
                str(value) for value in self.supported_state_schema_versions
            ],
            "supported_parameter_contract": str(self.supported_parameter_contract),
            "capability_flags": list(self.capability_flags),
        }
