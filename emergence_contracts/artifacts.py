"""C-130 immutable run-artifact references and manifest handoff."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ._validation import (
    require_instance,
    require_int,
    require_sorted_unique_text,
    require_text,
    require_tuple_instances,
)
from .canonical import canonical_json_bytes, canonical_sha256
from .errors import contract_invalid
from .ids import (
    ArtifactId,
    AssemblyId,
    CellId,
    ContentDigest,
    RunId,
    SchemaVersion,
)


CONTRACT_ID = "C-130"
CONTRACT_OWNER = "Contracts; written by Runtime/Lab; consumed by Evidence/Assurance"
API_VERSION = SchemaVersion("c130_v1")
ARTIFACT_REF_SCHEMA_VERSION = SchemaVersion("run_artifact_ref_v1")
RUN_MANIFEST_SCHEMA_VERSION = SchemaVersion("run_artifact_manifest_v1")


class RunCompletionStatusV1(str, Enum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ArtifactIntegrityStatusV1(str, Enum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class RunArtifactRefV1:
    artifact_id: ArtifactId
    artifact_kind: str
    artifact_schema_version: SchemaVersion
    locator: str
    content_digest: ContentDigest
    # A declared artifact may be intentionally unmaterialized in a descriptor
    # view.  ``None`` means that no byte-length claim is being made; it is not
    # equivalent to an empty artifact.
    size_bytes: int | None
    schema_version: SchemaVersion = ARTIFACT_REF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(self.artifact_id, ArtifactId, field="artifact_id")
        require_text(self.artifact_kind, field="artifact_kind")
        require_instance(
            self.artifact_schema_version,
            SchemaVersion,
            field="artifact_schema_version",
        )
        require_text(self.locator, field="locator")
        require_instance(self.content_digest, ContentDigest, field="content_digest")
        if self.size_bytes is not None:
            require_int(self.size_bytes, field="size_bytes")
        if self.schema_version != ARTIFACT_REF_SCHEMA_VERSION:
            raise contract_invalid("unsupported artifact ref schema", field="schema_version")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "artifact_id": str(self.artifact_id),
            "artifact_kind": self.artifact_kind,
            "artifact_schema_version": str(self.artifact_schema_version),
            "locator": self.locator,
            "content_digest": self.content_digest.to_dict(),
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RunArtifactRefV1":
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "artifact_id",
            "artifact_kind",
            "artifact_schema_version",
            "locator",
            "content_digest",
            "size_bytes",
        }:
            raise contract_invalid("artifact ref has an invalid shape", field="artifact_ref")
        digest = value["content_digest"]
        if not isinstance(digest, dict):
            raise contract_invalid("digest must be an object", field="content_digest")
        return cls(
            artifact_id=ArtifactId(value["artifact_id"]),
            artifact_kind=value["artifact_kind"],
            artifact_schema_version=SchemaVersion(value["artifact_schema_version"]),
            locator=value["locator"],
            content_digest=ContentDigest(digest["algorithm"], digest["hex"]),
            size_bytes=value["size_bytes"],
            schema_version=SchemaVersion(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class NamedDigestV1:
    name: str
    digest: ContentDigest

    def __post_init__(self) -> None:
        require_text(self.name, field="name")
        require_instance(self.digest, ContentDigest, field="digest")

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "digest": self.digest.to_dict()}


@dataclass(frozen=True, slots=True)
class RunArtifactManifestV1:
    run_id: RunId
    cell_id: CellId
    assembly_id: AssemblyId
    assembly_digest: ContentDigest
    runtime_route_id: str
    source_versions: tuple[str, ...]
    config_digests: tuple[NamedDigestV1, ...]
    checkpoint_refs: tuple[RunArtifactRefV1, ...]
    metric_refs: tuple[RunArtifactRefV1, ...]
    event_refs: tuple[RunArtifactRefV1, ...]
    completion_status: RunCompletionStatusV1
    integrity_status: ArtifactIntegrityStatusV1
    output_schema_versions: tuple[SchemaVersion, ...]
    schema_version: SchemaVersion = RUN_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name, expected in (
            ("run_id", RunId),
            ("cell_id", CellId),
            ("assembly_id", AssemblyId),
            ("assembly_digest", ContentDigest),
            ("completion_status", RunCompletionStatusV1),
            ("integrity_status", ArtifactIntegrityStatusV1),
        ):
            require_instance(getattr(self, field_name), expected, field=field_name)
        require_text(self.runtime_route_id, field="runtime_route_id")
        require_sorted_unique_text(self.source_versions, field="source_versions")
        for field_name, expected in (
            ("config_digests", NamedDigestV1),
            ("checkpoint_refs", RunArtifactRefV1),
            ("metric_refs", RunArtifactRefV1),
            ("event_refs", RunArtifactRefV1),
            ("output_schema_versions", SchemaVersion),
        ):
            require_tuple_instances(
                getattr(self, field_name),
                expected,
                field=field_name,
                unique=True,
                sorted_by_string=False,
            )
        config_names = [value.name for value in self.config_digests]
        if len(config_names) != len(set(config_names)):
            raise contract_invalid(
                "config digest names must be unique", field="config_digests"
            )
        all_refs = self.checkpoint_refs + self.metric_refs + self.event_refs
        artifact_ids = [str(value.artifact_id) for value in all_refs]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise contract_invalid(
                "manifest artifact IDs must be unique", field="artifact_refs"
            )
        if self.schema_version != RUN_MANIFEST_SCHEMA_VERSION:
            raise contract_invalid("unsupported run manifest schema", field="schema_version")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "run_id": str(self.run_id),
            "cell_id": str(self.cell_id),
            "assembly_id": str(self.assembly_id),
            "assembly_digest": self.assembly_digest.to_dict(),
            "runtime_route_id": self.runtime_route_id,
            "source_versions": list(self.source_versions),
            "config_digests": [value.to_dict() for value in self.config_digests],
            "checkpoint_refs": [value.to_dict() for value in self.checkpoint_refs],
            "metric_refs": [value.to_dict() for value in self.metric_refs],
            "event_refs": [value.to_dict() for value in self.event_refs],
            "completion_status": self.completion_status.value,
            "integrity_status": self.integrity_status.value,
            "output_schema_versions": [str(value) for value in self.output_schema_versions],
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def content_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "RunArtifactManifestV1":
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "run_id",
            "cell_id",
            "assembly_id",
            "assembly_digest",
            "runtime_route_id",
            "source_versions",
            "config_digests",
            "checkpoint_refs",
            "metric_refs",
            "event_refs",
            "completion_status",
            "integrity_status",
            "output_schema_versions",
        }:
            raise contract_invalid("run manifest has an invalid shape", field="manifest")
        assembly_digest = value["assembly_digest"]
        return cls(
            run_id=RunId(value["run_id"]),
            cell_id=CellId(value["cell_id"]),
            assembly_id=AssemblyId(value["assembly_id"]),
            assembly_digest=ContentDigest(
                assembly_digest["algorithm"], assembly_digest["hex"]
            ),
            runtime_route_id=value["runtime_route_id"],
            source_versions=tuple(value["source_versions"]),
            config_digests=tuple(
                NamedDigestV1(
                    name=item["name"],
                    digest=ContentDigest(
                        item["digest"]["algorithm"], item["digest"]["hex"]
                    ),
                )
                for item in value["config_digests"]
            ),
            checkpoint_refs=tuple(
                RunArtifactRefV1.from_dict(item) for item in value["checkpoint_refs"]
            ),
            metric_refs=tuple(
                RunArtifactRefV1.from_dict(item) for item in value["metric_refs"]
            ),
            event_refs=tuple(
                RunArtifactRefV1.from_dict(item) for item in value["event_refs"]
            ),
            completion_status=RunCompletionStatusV1(value["completion_status"]),
            integrity_status=ArtifactIntegrityStatusV1(value["integrity_status"]),
            output_schema_versions=tuple(
                SchemaVersion(item) for item in value["output_schema_versions"]
            ),
            schema_version=SchemaVersion(value["schema_version"]),
        )
