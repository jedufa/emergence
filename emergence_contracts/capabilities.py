"""C-150 capability declarations separated from qualification authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ._validation import (
    require_instance,
    require_sorted_unique_text,
    require_tuple_instances,
)
from .artifacts import RunArtifactRefV1
from .ids import (
    ArtifactId,
    BackendId,
    ComponentId,
    ContentDigest,
    SchemaVersion,
)
from .payloads import CanonicalPayloadV1


CONTRACT_ID = "C-150"
CONTRACT_OWNER = "Contracts DTO; authority records implemented by Assurance"
API_VERSION = SchemaVersion("c150_v1")
RUNTIME_CAPABILITY_SCHEMA_VERSION = SchemaVersion("runtime_capability_profile_v1")
QUALIFIED_ROUTE_SCHEMA_VERSION = SchemaVersion("qualified_route_envelope_v1")


@dataclass(frozen=True, slots=True)
class SemanticCapabilityV1:
    component_id: ComponentId
    semantic_contract_version: SchemaVersion
    capability_flags: tuple[str, ...]

    def __post_init__(self) -> None:
        require_instance(self.component_id, ComponentId, field="component_id")
        require_instance(
            self.semantic_contract_version,
            SchemaVersion,
            field="semantic_contract_version",
        )
        require_sorted_unique_text(self.capability_flags, field="capability_flags")


@dataclass(frozen=True, slots=True)
class RuntimeCapabilityProfileV1:
    provider_id: ComponentId
    profile_id: ComponentId
    backend_id: BackendId
    semantic_capabilities: tuple[SemanticCapabilityV1, ...]
    capability_flags: tuple[str, ...]
    schema_version: SchemaVersion = RUNTIME_CAPABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name, expected in (
            ("provider_id", ComponentId),
            ("profile_id", ComponentId),
            ("backend_id", BackendId),
        ):
            require_instance(getattr(self, field_name), expected, field=field_name)
        require_tuple_instances(
            self.semantic_capabilities,
            SemanticCapabilityV1,
            field="semantic_capabilities",
            unique=True,
        )
        require_sorted_unique_text(self.capability_flags, field="capability_flags")


@dataclass(frozen=True, slots=True)
class QualifiedRouteEnvelopeV1:
    runtime_provider_id: ComponentId
    runtime_profile_id: ComponentId
    assembly_component_ids: tuple[ComponentId, ...]
    semantic_contract_versions: tuple[SchemaVersion, ...]
    native_binding_artifact_hashes: tuple[ContentDigest, ...]
    state_checkpoint_contracts: tuple[SchemaVersion, ...]
    time_contract: SchemaVersion
    fixed_invariants: CanonicalPayloadV1
    admitted_envelope: CanonicalPayloadV1
    required_obligations: tuple[str, ...]
    qualification_receipt_refs: tuple[RunArtifactRefV1, ...]
    authority_record_ref: ArtifactId
    schema_version: SchemaVersion = QUALIFIED_ROUTE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name, expected in (
            ("runtime_provider_id", ComponentId),
            ("runtime_profile_id", ComponentId),
            ("time_contract", SchemaVersion),
            ("fixed_invariants", CanonicalPayloadV1),
            ("admitted_envelope", CanonicalPayloadV1),
            ("authority_record_ref", ArtifactId),
        ):
            require_instance(getattr(self, field_name), expected, field=field_name)
        for field_name, expected in (
            ("assembly_component_ids", ComponentId),
            ("semantic_contract_versions", SchemaVersion),
            ("native_binding_artifact_hashes", ContentDigest),
            ("state_checkpoint_contracts", SchemaVersion),
            ("qualification_receipt_refs", RunArtifactRefV1),
        ):
            require_tuple_instances(
                getattr(self, field_name), expected, field=field_name, unique=True
            )
        require_sorted_unique_text(
            self.required_obligations, field="required_obligations"
        )


@runtime_checkable
class CapabilityAuthorityReaderV1(Protocol):
    """Reads bounded authority records; it cannot execute Qualification."""

    def semantic_capability(self, component_id: ComponentId) -> SemanticCapabilityV1: ...

    def runtime_capability(
        self, provider_id: ComponentId
    ) -> RuntimeCapabilityProfileV1: ...

    def qualified_route(
        self, authority_record_ref: ArtifactId
    ) -> QualifiedRouteEnvelopeV1: ...
