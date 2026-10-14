"""Strict loader for the first version of the EIEL artifact catalog.

This module describes artifacts and offers an explicit, read-only digest
verification step. It deliberately does not import, load, route, or execute
artifact contents.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Iterable, Mapping


ARTIFACT_CATALOG_SCHEMA_V1 = "eiel_artifact_catalog_v1"
ARTIFACT_STATUSES_V1 = frozenset(
    {"candidate", "qualified", "deprecated", "quarantined", "retired"}
)

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
        "qualification",
    }
)
_QUALIFICATION_FIELDS = frozenset(
    {"schema_version", "envelope_id", "receipt_sha256", "capabilities"}
)
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_LOWERCASE_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArtifactCatalogError(ValueError):
    """Raised when catalog input is invalid or selection is not unique."""


class ArtifactVerificationError(ArtifactCatalogError):
    """Raised when an opt-in artifact digest check cannot be completed safely."""


@dataclass(frozen=True)
class QualificationBinding:
    schema_version: str
    envelope_id: str
    receipt_sha256: str
    capabilities: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    kind: str
    abi: str
    architecture: str
    build_identity: str
    status: str
    relative_path: PurePosixPath
    sha256: str
    resolved_path: Path
    qualification: QualificationBinding


@dataclass(frozen=True)
class ArtifactCatalog:
    schema_version: str
    artifacts: tuple[ArtifactRecord, ...]
    artifact_root: Path

    def verify_sha256(
        self, artifact: ArtifactRecord, *, chunk_size: int = 1024 * 1024
    ) -> None:
        """Verify one catalog record without loading or executing its contents."""

        verify_artifact_sha256(
            artifact, artifact_root=self.artifact_root, chunk_size=chunk_size
        )

    def filter(
        self,
        *,
        kind: str | None = None,
        abi: str | None = None,
        architecture: str | None = None,
        build_identity: str | None = None,
        status: str | None = None,
        envelope_id: str | None = None,
        capability: str | None = None,
    ) -> tuple[ArtifactRecord, ...]:
        """Return exact matches in stable artifact-ID order."""

        criteria = {
            "kind": kind,
            "abi": abi,
            "architecture": architecture,
            "build_identity": build_identity,
            "status": status,
        }
        for field, value in criteria.items():
            if value is not None and not isinstance(value, str):
                raise ArtifactCatalogError(
                    f"selection field {field!r} must be a string"
                )
        for field, value in {
            "envelope_id": envelope_id,
            "capability": capability,
        }.items():
            if value is not None and not isinstance(value, str):
                raise ArtifactCatalogError(
                    f"selection field {field!r} must be a string"
                )
        return tuple(
            artifact
            for artifact in self.artifacts
            if all(
                value is None or getattr(artifact, field) == value
                for field, value in criteria.items()
            )
            and (
                envelope_id is None or artifact.qualification.envelope_id == envelope_id
            )
            and (
                capability is None or capability in artifact.qualification.capabilities
            )
        )

    def select_one(
        self,
        *,
        kind: str | None = None,
        abi: str | None = None,
        architecture: str | None = None,
        build_identity: str | None = None,
        status: str | None = None,
        envelope_id: str | None = None,
        capability: str | None = None,
    ) -> ArtifactRecord:
        """Return exactly one match, rejecting missing or ambiguous selection."""

        matches = self.filter(
            kind=kind,
            abi=abi,
            architecture=architecture,
            build_identity=build_identity,
            status=status,
            envelope_id=envelope_id,
            capability=capability,
        )
        if len(matches) != 1:
            ids = ", ".join(artifact.artifact_id for artifact in matches) or "none"
            raise ArtifactCatalogError(
                f"artifact selection must match exactly one record; "
                f"matched {len(matches)} ({ids})"
            )
        return matches[0]


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactCatalogError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ArtifactCatalogError(f"non-finite JSON number {value!r} is not allowed")


def _check_exact_fields(
    value: Any, *, expected: frozenset[str], location: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactCatalogError(f"{location} must be a JSON object")
    actual = frozenset(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ArtifactCatalogError(
            f"{location} has invalid fields; missing={missing!r} extra={extra!r}"
        )
    return value


def _required_string(value: Any, *, field: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ArtifactCatalogError(f"{field} has an invalid string value")
    return value


def _resolve_relative_path(
    value: Any, *, field: str, artifact_root: Path
) -> tuple[PurePosixPath, Path]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ArtifactCatalogError(f"{field} must be a non-empty POSIX relative path")
    raw_parts = value.split("/")
    relative_path = PurePosixPath(value)
    if relative_path.is_absolute() or any(
        part in {"", ".", ".."} for part in raw_parts
    ):
        raise ArtifactCatalogError(f"{field} must stay below the artifact root")

    resolved = (artifact_root / Path(*relative_path.parts)).resolve(strict=False)
    try:
        resolved.relative_to(artifact_root)
    except ValueError as exc:
        raise ArtifactCatalogError(
            f"{field} resolves outside the artifact root"
        ) from exc
    return relative_path, resolved


def parse_artifact_catalog(
    payload: Any, *, artifact_root: str | Path
) -> ArtifactCatalog:
    """Validate a decoded catalog and resolve its paths beneath ``artifact_root``."""

    root_input = Path(artifact_root).expanduser()
    if not root_input.is_absolute():
        raise ArtifactCatalogError("artifact_root must be absolute")
    root = root_input.resolve(strict=False)
    catalog = _check_exact_fields(payload, expected=_CATALOG_FIELDS, location="catalog")
    if catalog["schema_version"] != ARTIFACT_CATALOG_SCHEMA_V1:
        raise ArtifactCatalogError(
            f"unsupported artifact catalog schema {catalog['schema_version']!r}"
        )
    raw_artifacts = catalog["artifacts"]
    if not isinstance(raw_artifacts, list):
        raise ArtifactCatalogError("catalog.artifacts must be a JSON array")

    records: list[ArtifactRecord] = []
    seen_ids: set[str] = set()
    for index, raw_artifact in enumerate(raw_artifacts):
        location = f"catalog.artifacts[{index}]"
        artifact = _check_exact_fields(
            raw_artifact, expected=_ARTIFACT_FIELDS, location=location
        )
        artifact_id = _required_string(
            artifact["artifact_id"], field=f"{location}.artifact_id", pattern=_TOKEN_RE
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
        records.append(
            ArtifactRecord(
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
            )
        )

    return ArtifactCatalog(
        schema_version=ARTIFACT_CATALOG_SCHEMA_V1,
        artifacts=tuple(sorted(records, key=lambda record: record.artifact_id)),
        artifact_root=root,
    )


def load_artifact_catalog(
    path: str | Path, *, artifact_root: str | Path
) -> ArtifactCatalog:
    """Read and strictly validate a UTF-8 JSON artifact catalog."""

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
    return parse_artifact_catalog(payload, artifact_root=artifact_root)


def verify_artifact_sha256(
    artifact: ArtifactRecord,
    *,
    artifact_root: str | Path,
    chunk_size: int = 1024 * 1024,
) -> None:
    """Stream and verify one regular artifact file, failing closed.

    Verification is deliberately opt-in. Symlinks in any artifact path
    component and non-regular files are rejected; no artifact is imported,
    mapped, executed, or modified.
    """

    if not isinstance(artifact, ArtifactRecord):
        raise ArtifactVerificationError("artifact must be an ArtifactRecord")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ArtifactVerificationError("chunk_size must be a positive integer")

    root_input = Path(artifact_root).expanduser()
    if not root_input.is_absolute():
        raise ArtifactVerificationError("artifact_root must be absolute")
    root = root_input.resolve(strict=False)
    if artifact.relative_path.is_absolute() or any(
        part in {"", ".", ".."} for part in artifact.relative_path.parts
    ):
        raise ArtifactVerificationError("artifact record has an invalid relative path")
    candidate = root.joinpath(*artifact.relative_path.parts)
    current = root
    try:
        for part in artifact.relative_path.parts:
            current = current / part
            if current.is_symlink():
                raise ArtifactVerificationError(
                    f"artifact {artifact.artifact_id!r} path contains a symlink"
                )
        if artifact.resolved_path != candidate:
            raise ArtifactVerificationError(
                "artifact record path does not match artifact_root"
            )
        metadata = candidate.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ArtifactVerificationError(
                f"artifact {artifact.artifact_id!r} is not a regular file"
            )
    except ArtifactVerificationError:
        raise
    except OSError as exc:
        raise ArtifactVerificationError(
            f"could not inspect artifact {artifact.artifact_id!r}: {exc}"
        ) from exc
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_only = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_only is None:
        raise ArtifactVerificationError(
            "safe descriptor-relative artifact verification is unavailable"
        )
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow | directory_only
    )
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow
    directory_descriptor = -1
    descriptor = -1
    try:
        directory_descriptor = os.open(root, directory_flags)
        for part in artifact.relative_path.parts[:-1]:
            next_descriptor = os.open(
                part,
                directory_flags,
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        descriptor = os.open(
            artifact.relative_path.parts[-1],
            file_flags,
            dir_fd=directory_descriptor,
        )
    except OSError as exc:
        raise ArtifactVerificationError(
            f"could not open artifact {artifact.artifact_id!r}: {exc}"
        ) from exc
    finally:
        if directory_descriptor >= 0:
            os.close(directory_descriptor)

    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ArtifactVerificationError(
                f"artifact {artifact.artifact_id!r} is not a regular file"
            )
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactVerificationError(
            f"could not read artifact {artifact.artifact_id!r}: {exc}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    actual = digest.hexdigest()
    if actual != artifact.sha256:
        raise ArtifactVerificationError(
            f"artifact {artifact.artifact_id!r} sha256 mismatch: "
            f"expected {artifact.sha256}, got {actual}"
        )
