"""Versioned, fail-closed artifact catalog models."""

from .catalog_v1 import (
    ARTIFACT_CATALOG_SCHEMA_V1,
    ARTIFACT_STATUSES_V1,
    ArtifactCatalog,
    ArtifactCatalogError,
    ArtifactRecord,
    ArtifactVerificationError,
    QualificationBinding,
    load_artifact_catalog,
    parse_artifact_catalog,
    verify_artifact_sha256,
)
from .catalog_v2 import (
    ARTIFACT_CATALOG_SCHEMA_V2,
    ArtifactCatalogV2,
    ArtifactRecordV2,
    ArtifactRequirementsV2,
    load_artifact_catalog_v2,
    parse_artifact_catalog_v2,
)

__all__ = [
    "ARTIFACT_CATALOG_SCHEMA_V1",
    "ARTIFACT_STATUSES_V1",
    "ARTIFACT_CATALOG_SCHEMA_V2",
    "ArtifactCatalog",
    "ArtifactCatalogError",
    "ArtifactRecord",
    "ArtifactVerificationError",
    "ArtifactCatalogV2",
    "ArtifactRecordV2",
    "ArtifactRequirementsV2",
    "QualificationBinding",
    "load_artifact_catalog",
    "load_artifact_catalog_v2",
    "parse_artifact_catalog",
    "parse_artifact_catalog_v2",
    "verify_artifact_sha256",
]
