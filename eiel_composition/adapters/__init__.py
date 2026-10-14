"""Public legacy adapters used while current EIEL remains authoritative."""

from __future__ import annotations

from .legacy_evidence import (
    LegacyEvidenceEvaluatorAdapter,
    LegacyRunArtifactBindingV1,
    LegacyRunArtifactManifestAdapter,
)
from .legacy_lab import (
    LegacyCampaignExecutionPortV1,
    LegacyExperimentExecutorAdapter,
    LegacyExperimentPlannerAdapter,
)
from .legacy_components import (
    EielLegacyAssemblyFactory,
    LegacyAdaptationDescriptorAdapter,
    LegacyAgentDescriptorAdapter,
    LegacyAssemblyParityV1,
    LegacyNativeBindingDescriptorAdapter,
    LegacyWorldDescriptorAdapter,
)
from .legacy_runtime import (
    LegacyControlPortAdapter,
    LegacyRuntimeAdapter,
    LegacyRuntimeServicesAdapter,
    LegacyScheduleAdapter,
    LegacyTelemetryAdapter,
)
from .legacy_state import (
    LegacyCompositeCheckpointAdapter,
    LegacyRNGAuthorityAdapter,
    LegacySnapshotParticipantView,
)

__all__ = [
    "LegacyEvidenceEvaluatorAdapter",
    "EielLegacyAssemblyFactory",
    "LegacyAdaptationDescriptorAdapter",
    "LegacyAgentDescriptorAdapter",
    "LegacyAssemblyParityV1",
    "LegacyCampaignExecutionPortV1",
    "LegacyControlPortAdapter",
    "LegacyCompositeCheckpointAdapter",
    "LegacyExperimentExecutorAdapter",
    "LegacyExperimentPlannerAdapter",
    "LegacyRunArtifactBindingV1",
    "LegacyRunArtifactManifestAdapter",
    "LegacyRuntimeAdapter",
    "LegacyRuntimeServicesAdapter",
    "LegacyRNGAuthorityAdapter",
    "LegacyScheduleAdapter",
    "LegacyTelemetryAdapter",
    "LegacySnapshotParticipantView",
    "LegacyNativeBindingDescriptorAdapter",
    "LegacyWorldDescriptorAdapter",
]
