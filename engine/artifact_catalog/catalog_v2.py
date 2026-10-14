"""Standalone compatibility selection for EIEL artifact catalog v2.

V2 adds explicit device requirements to the v1 artifact identity and
qualification binding. Selection is pure and deterministic. Hash verification
is opt-in and never imports, loads, routes, or executes an artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from .catalog_v1 import (
    ARTIFACT_STATUSES_V1,
    ArtifactCatalogError,
    ArtifactRecord,
    QualificationBinding,
    _LOWERCASE_SHA256_RE,
    _TOKEN_RE,
    _check_exact_fields,
    _reject_duplicate_keys,
    _reject_json_constant,
    _required_string,
    _resolve_relative_path,
    verify_artifact_sha256,
)


ARTIFACT_CATALOG_SCHEMA_V2 = "eiel_artifact_catalog_v2"

_CATALOG_FIELDS = frozenset({"schema_version", "artifacts"})
_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_id",
        "kind",
        "abi",
        "architecture",
        "build_identity",
        "status",
        "relative_path",
        "sha256",
        "requirements",
        "qualification",
    }
)
_REQUIREMENT_FIELDS = frozenset({"compute_capability", "minimum_device_memory_bytes"})
_QUALIFICATION_FIELDS = frozenset(
    {"schema_version", "envelope_id", "receipt_sha256", "capabilities"}
)


@dataclass(frozen=True)
class ArtifactRequirementsV2:
    """Physical-device requirements declared by an artifact."""

    compute_capability: str
    minimum_device_memory_bytes: int


@dataclass(frozen=True)
class ArtifactRecordV2(ArtifactRecord):
    requirements: ArtifactRequirementsV2


@dataclass(frozen=True)
class ArtifactCatalogV2:
    schema_version: str
    artifacts: tuple[ArtifactRecordV2, ...]
    artifact_root: Path

    def filter_qualified(
        self,
        *,
        abi: str,
        compute_capability: str,
        available_device_memory_bytes: int,
        mechanics_envelope_id: str,
        kind: str | None = None,
        required_capabilities: Iterable[str] = (),
    ) -> tuple[ArtifactRecordV2, ...]:
        """Return compatible qualified artifacts in stable artifact-ID order."""

        criteria = {
            "abi": abi,
            "compute_capability": compute_capability,
            "mechanics_envelope_id": mechanics_envelope_id,
        }
        for field, value in criteria.items():
            if not isinstance(value, str) or _TOKEN_RE.fullmatch(value) is None:
                raise ArtifactCatalogError(f"selection field {field!r} must be a token")
        if kind is not None and (
            not isinstance(kind, str) or _TOKEN_RE.fullmatch(kind) is None
        ):
            raise ArtifactCatalogError("selection field 'kind' must be a token")
        if (
            isinstance(available_device_memory_bytes, bool)
            or not isinstance(available_device_memory_bytes, int)
            or available_device_memory_bytes < 0
        ):
            raise ArtifactCatalogError(
                "available_device_memory_bytes must be a nonnegative integer"
            )
        try:
            capabilities = tuple(required_capabilities)
        except TypeError as exc:
            raise ArtifactCatalogError(
                "required_capabilities must be sorted unique tokens"
            ) from exc
        if (
            any(
                not isinstance(value, str) or _TOKEN_RE.fullmatch(value) is None
                for value in capabilities
            )
            or tuple(sorted(set(capabilities))) != capabilities
        ):
            raise ArtifactCatalogError(
                "required_capabilities must be sorted unique tokens"
            )

        required = frozenset(capabilities)
        return tuple(
            artifact
            for artifact in self.artifacts
            if artifact.status == "qualified"
            and artifact.abi == abi
            and artifact.requirements.compute_capability == compute_capability
            and artifact.requirements.minimum_device_memory_bytes
            <= available_device_memory_bytes
            and artifact.qualification.envelope_id == mechanics_envelope_id
            and (kind is None or artifact.kind == kind)
            and required.issubset(artifact.qualification.capabilities)
        )

    def select_one_qualified(self, **criteria: Any) -> ArtifactRecordV2:
        """Return exactly one compatible qualified artifact."""

        matches = self.filter_qualified(**criteria)
        if len(matches) != 1:
            ids = ", ".join(artifact.artifact_id for artifact in matches) or "none"
            raise ArtifactCatalogError(
                "qualified artifact selection must match exactly one record; "
                f"matched {len(matches)} ({ids})"
            )
        return matches[0]

    def select_one_verified(
        self, *, verification_chunk_size: int = 1024 * 1024, **criteria: Any
    ) -> ArtifactRecordV2:
        """Select one qualified artifact and verify its SHA-256 before return."""

        artifact = self.select_one_qualified(**criteria)
        verify_artifact_sha256(
            artifact,
            artifact_root=self.artifact_root,
            chunk_size=verification_chunk_size,
        )
        return artifact


def _required_nonnegative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactCatalogError(f"{field} must be a nonnegative integer")
    return value


def parse_artifact_catalog_v2(
    payload: Any, *, artifact_root: str | Path
) -> ArtifactCatalogV2:
    """Validate a decoded v2 catalog without touching artifact contents."""

    root_input = Path(artifact_root).expanduser()
    if not root_input.is_absolute():
        raise ArtifactCatalogError("artifact_root must be absolute")
    root = root_input.resolve(strict=False)
    catalog = _check_exact_fields(payload, expected=_CATALOG_FIELDS, location="catalog")
    if catalog["schema_version"] != ARTIFACT_CATALOG_SCHEMA_V2:
        raise ArtifactCatalogError(
            f"unsupported artifact catalog schema {catalog['schema_version']!r}"
        )
    raw_artifacts = catalog["artifacts"]
    if not isinstance(raw_artifacts, list):
        raise ArtifactCatalogError("catalog.artifacts must be a JSON array")

    records: list[ArtifactRecordV2] = []
    seen_ids: set[str] = set()
    for index, raw_artifact in enumerate(raw_artifacts):
        location = f"catalog.artifacts[{index}]"
        artifact = _check_exact_fields(
            raw_artifact, expected=_ARTIFACT_FIELDS, location=location
        )
        artifact_id = _required_string(
            artifact["artifact_id"],
            field=f"{location}.artifact_id",
            pattern=_TOKEN_RE,
        )
        if artifact_id in seen_ids:
            raise ArtifactCatalogError(f"duplicate artifact_id {artifact_id!r}")
        seen_ids.add(artifact_id)
        status = artifact["status"]
        if not isinstance(status, str) or status not in ARTIFACT_STATUSES_V1:
            raise ArtifactCatalogError(
                f"{location}.status has unsupported value {status!r}"
            )
        relative_path, resolved_path = _resolve_relative_path(
            artifact["relative_path"],
            field=f"{location}.relative_path",
            artifact_root=root,
        )
        requirements = _check_exact_fields(
            artifact["requirements"],
            expected=_REQUIREMENT_FIELDS,
            location=f"{location}.requirements",
        )
        qualification = _check_exact_fields(
            artifact["qualification"],
            expected=_QUALIFICATION_FIELDS,
            location=f"{location}.qualification",
        )
        raw_capabilities = qualification["capabilities"]
        if (
            not isinstance(raw_capabilities, list)
            or not raw_capabilities
            or any(
                not isinstance(value, str) or _TOKEN_RE.fullmatch(value) is None
                for value in raw_capabilities
            )
            or raw_capabilities != sorted(set(raw_capabilities))
        ):
            raise ArtifactCatalogError(
                f"{location}.qualification.capabilities must be a nonempty "
                "sorted list of unique tokens"
            )
        record = ArtifactRecordV2(
            artifact_id=artifact_id,
            kind=_required_string(
                artifact["kind"], field=f"{location}.kind", pattern=_TOKEN_RE
            ),
            abi=_required_string(
                artifact["abi"], field=f"{location}.abi", pattern=_TOKEN_RE
            ),
            architecture=_required_string(
                artifact["architecture"],
                field=f"{location}.architecture",
                pattern=_TOKEN_RE,
            ),
            build_identity=_required_string(
                artifact["build_identity"],
                field=f"{location}.build_identity",
                pattern=_TOKEN_RE,
            ),
            status=status,
            relative_path=relative_path,
            sha256=_required_string(
                artifact["sha256"],
                field=f"{location}.sha256",
                pattern=_LOWERCASE_SHA256_RE,
            ),
            resolved_path=resolved_path,
            qualification=QualificationBinding(
                schema_version=_required_string(
                    qualification["schema_version"],
                    field=f"{location}.qualification.schema_version",
                    pattern=_TOKEN_RE,
                ),
                envelope_id=_required_string(
                    qualification["envelope_id"],
                    field=f"{location}.qualification.envelope_id",
                    pattern=_TOKEN_RE,
                ),
                receipt_sha256=_required_string(
                    qualification["receipt_sha256"],
                    field=f"{location}.qualification.receipt_sha256",
                    pattern=_LOWERCASE_SHA256_RE,
                ),
                capabilities=tuple(raw_capabilities),
            ),
            requirements=ArtifactRequirementsV2(
                compute_capability=_required_string(
                    requirements["compute_capability"],
                    field=f"{location}.requirements.compute_capability",
                    pattern=_TOKEN_RE,
                ),
                minimum_device_memory_bytes=_required_nonnegative_integer(
                    requirements["minimum_device_memory_bytes"],
                    field=f"{location}.requirements.minimum_device_memory_bytes",
                ),
            ),
        )
        if record.architecture != record.requirements.compute_capability:
            raise ArtifactCatalogError(
                f"{location}.architecture must equal "
                f"{location}.requirements.compute_capability"
            )
        records.append(record)

    return ArtifactCatalogV2(
        schema_version=ARTIFACT_CATALOG_SCHEMA_V2,
        artifacts=tuple(sorted(records, key=lambda record: record.artifact_id)),
        artifact_root=root,
    )


def load_artifact_catalog_v2(
    path: str | Path, *, artifact_root: str | Path
) -> ArtifactCatalogV2:
    """Read and strictly validate a UTF-8 JSON v2 artifact catalog."""

    catalog_path = Path(path)
    try:
        with catalog_path.open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
    except ArtifactCatalogError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactCatalogError(
            f"could not load artifact catalog {catalog_path}: {exc}"
        ) from exc
    return parse_artifact_catalog_v2(payload, artifact_root=artifact_root)
