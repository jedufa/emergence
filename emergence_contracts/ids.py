"""C-001 opaque identities, versions, and content digests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from .errors import contract_invalid


_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_LOWER_HEX_RE = re.compile(r"^[0-9a-f]+$")
_SHA256_HEX_LENGTH = 64


def _validate_opaque(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise contract_invalid(
            "opaque value must be a string",
            field=field,
            value_type=type(value).__name__,
        )
    if not value:
        raise contract_invalid("opaque value must not be empty", field=field)
    if value.strip() != value:
        raise contract_invalid(
            "opaque value must not contain surrounding whitespace", field=field
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise contract_invalid("opaque value contains a control character", field=field)
    return value


@dataclass(frozen=True, slots=True)
class _OpaqueId:
    """Exact string identity: validated but never normalized or interpreted."""

    value: str
    kind: ClassVar[str] = "OpaqueId"

    def __post_init__(self) -> None:
        _validate_opaque(self.value, field=self.kind)

    def __str__(self) -> str:
        return self.value


class ModuleId(_OpaqueId):
    __slots__ = ()
    kind = "ModuleId"


class ComponentId(_OpaqueId):
    __slots__ = ()
    kind = "ComponentId"


class ComponentInstanceId(_OpaqueId):
    __slots__ = ()
    kind = "ComponentInstanceId"


class RunId(_OpaqueId):
    __slots__ = ()
    kind = "RunId"


class AttemptId(_OpaqueId):
    __slots__ = ()
    kind = "AttemptId"


class CellId(_OpaqueId):
    __slots__ = ()
    kind = "CellId"


class EntityId(_OpaqueId):
    __slots__ = ()
    kind = "EntityId"


class CandidateId(_OpaqueId):
    __slots__ = ()
    kind = "CandidateId"


class LineageId(_OpaqueId):
    __slots__ = ()
    kind = "LineageId"


class CheckpointId(_OpaqueId):
    __slots__ = ()
    kind = "CheckpointId"


class BranchId(_OpaqueId):
    __slots__ = ()
    kind = "BranchId"


class ArtifactId(_OpaqueId):
    __slots__ = ()
    kind = "ArtifactId"


class AssemblyId(_OpaqueId):
    __slots__ = ()
    kind = "AssemblyId"


class BackendId(_OpaqueId):
    __slots__ = ()
    kind = "BackendId"


class CommandId(_OpaqueId):
    __slots__ = ()
    kind = "CommandId"


class ExperimentId(_OpaqueId):
    __slots__ = ()
    kind = "ExperimentId"


class LeaseId(_OpaqueId):
    __slots__ = ()
    kind = "LeaseId"


class ParticipantId(_OpaqueId):
    __slots__ = ()
    kind = "ParticipantId"


class ResultSetId(_OpaqueId):
    __slots__ = ()
    kind = "ResultSetId"


@dataclass(frozen=True, slots=True)
class SchemaVersion:
    """An explicit schema/API version token; compatibility is not inferred."""

    value: str

    def __post_init__(self) -> None:
        value = _validate_opaque(self.value, field="SchemaVersion")
        if _TOKEN_RE.fullmatch(value) is None:
            raise contract_invalid(
                "schema version must be an ASCII token", field="SchemaVersion"
            )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ContentDigest:
    """Content/binding evidence. A digest never grants semantic authority."""

    algorithm: str
    hex: str

    def __post_init__(self) -> None:
        if self.algorithm != "sha256":
            raise contract_invalid(
                "unsupported digest algorithm", field="ContentDigest.algorithm"
            )
        if not isinstance(self.hex, str):
            raise contract_invalid(
                "digest hex must be a string",
                field="ContentDigest.hex",
                value_type=type(self.hex).__name__,
            )
        if len(self.hex) != _SHA256_HEX_LENGTH or _LOWER_HEX_RE.fullmatch(self.hex) is None:
            raise contract_invalid(
                "sha256 digest must be exactly 64 lowercase hexadecimal characters",
                field="ContentDigest.hex",
            )

    def __str__(self) -> str:
        return f"{self.algorithm}:{self.hex}"

    def to_dict(self) -> dict[str, str]:
        return {"algorithm": self.algorithm, "hex": self.hex}

    @classmethod
    def parse(cls, value: object) -> "ContentDigest":
        if not isinstance(value, str) or value.count(":") != 1:
            raise contract_invalid(
                "digest text must have the form '<algorithm>:<hex>'",
                field="ContentDigest",
                value_type=type(value).__name__,
            )
        algorithm, hex_value = value.split(":", 1)
        return cls(algorithm=algorithm, hex=hex_value)
