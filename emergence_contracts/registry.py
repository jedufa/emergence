"""Frozen public contract registry for EIEL-MODULAR-FREEZE-1.0."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from ._validation import require_sorted_unique_text, require_text
from .ids import SchemaVersion


FROZEN_ARCHITECTURE_ID = "EIEL-MODULAR-FREEZE-1.0"


@dataclass(frozen=True, slots=True)
class ContractDefinitionV1:
    contract_id: str
    name: str
    owner: str
    api_version: SchemaVersion
    public_types: tuple[str, ...]

    def __post_init__(self) -> None:
        require_text(self.contract_id, field="contract_id")
        require_text(self.name, field="name")
        require_text(self.owner, field="owner")
        require_sorted_unique_text(self.public_types, field="public_types")


CONTRACT_REGISTRY_V1 = (
    ContractDefinitionV1("C-001", "Canonical Identity", "Contracts", SchemaVersion("c001_v1"), ("ArtifactId", "ContentDigest", "SchemaVersion")),
    ContractDefinitionV1("C-010", "Component Descriptor", "Contracts", SchemaVersion("c010_v1"), ("ComponentCompatibilityV1", "ComponentDescriptorV1")),
    ContractDefinitionV1("C-020", "Runtime Services", "Contracts; provided by Runtime", SchemaVersion("c020_v1"), ("ArtifactOutputPortV1", "RuntimeServicesV1")),
    ContractDefinitionV1("C-030", "Execution Schedule and Run Cursor", "Contracts; authority implemented by Runtime", SchemaVersion("c030_v1"), ("ExecutionScheduleV1", "RunCursorV1")),
    ContractDefinitionV1("C-040", "World Component", "Contracts; implemented by WorldKit", SchemaVersion("c040_v1"), ("WorldComponentV1",)),
    ContractDefinitionV1("C-050", "Agent Component", "Contracts; implemented by AgentKit", SchemaVersion("c050_v1"), ("AgentComponentV1",)),
    ContractDefinitionV1("C-060", "Adaptation Component", "Contracts; implemented by Adaptation", SchemaVersion("c060_v1"), ("AdaptationComponentV1",)),
    ContractDefinitionV1("C-070", "System Assembly", "Contracts; concrete assembly owned by EIEL", SchemaVersion("c070_v1"), ("SystemAssemblyV1",)),
    ContractDefinitionV1("C-080", "Snapshot Participant and Composite Checkpoint", "Contracts DTO; Runtime orchestration", SchemaVersion("c080_v1"), ("CompositeCheckpointViewV1", "SnapshotParticipantPortV1")),
    ContractDefinitionV1("C-090", "RNG Authority and Lease", "Contracts; root authority implemented by Runtime", SchemaVersion("c090_v1"), ("RNGAuthorityV1", "RNGLeaseDescriptorV1", "RNGLeaseV1")),
    ContractDefinitionV1("C-100", "Runtime Event and Telemetry", "Contracts", SchemaVersion("c100_v1"), ("RuntimeEventV1", "TelemetrySinkV1")),
    ContractDefinitionV1("C-110", "Runtime Engine and Execution Provider", "Contracts; implemented by Runtime", SchemaVersion("c110_v1"), ("RuntimeExecutionRequestV1", "RuntimeProviderV1", "RuntimeResultV1")),
    ContractDefinitionV1("C-120", "Experiment Plan and Executor", "Contracts; implemented by Lab", SchemaVersion("c120_v1"), ("ExperimentExecutorV1", "ExperimentPlanV1", "ExperimentResultSetV1")),
    ContractDefinitionV1("C-130", "Run Artifact Manifest", "Contracts; written by Runtime/Lab", SchemaVersion("c130_v1"), ("RunArtifactManifestV1", "RunArtifactRefV1")),
    ContractDefinitionV1("C-140", "Evidence Evaluator", "Evidence", SchemaVersion("c140_v1"), ("EvidenceEvaluatorV1", "EvidenceProfileV1")),
    ContractDefinitionV1("C-150", "Capability and Qualification", "Contracts DTO; implemented by Assurance", SchemaVersion("c150_v1"), ("CapabilityAuthorityReaderV1", "QualifiedRouteEnvelopeV1", "RuntimeCapabilityProfileV1")),
    ContractDefinitionV1("C-160", "Control Command and Control Port", "Contracts; deterministic application owned by Runtime", SchemaVersion("c160_v1"), ("ControlCommandV1", "ControlPortV1")),
    ContractDefinitionV1("C-170", "Native Binding", "Contracts; declared by semantic owner", SchemaVersion("c170_v1"), ("NativeBindingV1",)),
)

CONTRACT_REGISTRY_BY_ID_V1 = MappingProxyType(
    {definition.contract_id: definition for definition in CONTRACT_REGISTRY_V1}
)
