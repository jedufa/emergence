"""Read-only schema identities and fail-closed APIs for retired authority."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final


AUTHORITY_REQUEST_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_authority_publication_request_v1"
)
PRODUCTION_BINDING_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_production_binding_manifest_v1"
)
MIXED_CAPTURE_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "eiel_mixed_mode_resident_isolation_capture_manifest_v1"
)
CROSS_STORAGE_CAPTURE_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_cross_storage_capture_manifest_v1"
)
CROSS_STORAGE_CAPTURE_ARTIFACT_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_cross_storage_capture_artifact_v1"
)
REPORTING_MODE_EVIDENCE_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_reporting_mode_evidence_v1"
)
MECHANIC_FILENAME: Final[str] = "mechanic-qualification-v2.json"
REPORTING_FILENAME: Final[str] = "reporting-qualification-v1.json"
BINDING_FILENAME: Final[str] = "production-binding-manifest-v1.json"


class InheritanceCrossPathAuthorityPublicationError(RuntimeError):
    """The retired authority cannot be published or bound to a campaign."""


@dataclass(frozen=True, slots=True)
class InheritanceCrossPathAuthorityPublicationRequestV1:
    exact_full_candidate_binding_path: Path
    compact_candidate_binding_path: Path
    mixed_gate_plan_path: Path
    mixed_capture_manifest_path: Path
    mixed_gate_result_path: Path
    cross_storage_plan_path: Path
    cross_storage_capture_manifest_path: Path
    cross_storage_result_path: Path
    reporting_enabled_evidence_path: Path
    reporting_disabled_evidence_path: Path
    execution_envelope_path: Path
    output_dir: Path
    schema_version: str = field(
        default=AUTHORITY_REQUEST_SCHEMA_VERSION,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class CampaignSpecArtifactBindingRequestV1:
    source_spec_path: Path
    production_binding_path: Path
    output_spec_path: Path
    expected_source_sha256: str
    overwrite_source: bool = False
    schema_version: str = field(
        default="eiel_campaign_spec_artifact_binding_request_v1",
        init=False,
    )


def _retired() -> InheritanceCrossPathAuthorityPublicationError:
    return InheritanceCrossPathAuthorityPublicationError(
        "P12_RETIRED_INHERITANCE_AUTHORITY standalone publication is disabled"
    )


def publish_inheritance_cross_path_authority_v1(
    request: InheritanceCrossPathAuthorityPublicationRequestV1,
) -> Path:
    """Reject before reading inputs or creating output."""

    del request
    raise _retired()


def bind_campaign_spec_artifacts_v1(
    request: CampaignSpecArtifactBindingRequestV1,
) -> Path:
    """Reject before reading a binding or writing a campaign spec."""

    del request
    raise _retired()


__all__ = [
    "AUTHORITY_REQUEST_SCHEMA_VERSION",
    "BINDING_FILENAME",
    "CROSS_STORAGE_CAPTURE_ARTIFACT_SCHEMA_VERSION",
    "CROSS_STORAGE_CAPTURE_MANIFEST_SCHEMA_VERSION",
    "CampaignSpecArtifactBindingRequestV1",
    "InheritanceCrossPathAuthorityPublicationRequestV1",
    "InheritanceCrossPathAuthorityPublicationError",
    "MECHANIC_FILENAME",
    "MIXED_CAPTURE_MANIFEST_SCHEMA_VERSION",
    "PRODUCTION_BINDING_SCHEMA_VERSION",
    "REPORTING_FILENAME",
    "REPORTING_MODE_EVIDENCE_SCHEMA_VERSION",
    "bind_campaign_spec_artifacts_v1",
    "publish_inheritance_cross_path_authority_v1",
]
