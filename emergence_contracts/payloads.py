"""Immutable canonical payload envelope shared by frozen public contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass

from ._validation import require_instance
from .canonical import canonical_json_bytes, canonical_sha256
from .errors import contract_invalid
from .ids import ContentDigest, SchemaVersion


CANONICAL_PAYLOAD_SCHEMA_VERSION = SchemaVersion("canonical_payload_v1")


@dataclass(frozen=True, slots=True)
class CanonicalPayloadV1:
    """Canonical JSON bytes plus the schema that gives those bytes meaning."""

    schema_version: SchemaVersion
    canonical_bytes: bytes

    def __post_init__(self) -> None:
        require_instance(self.schema_version, SchemaVersion, field="schema_version")
        if not isinstance(self.canonical_bytes, bytes):
            raise contract_invalid(
                "canonical payload must contain bytes",
                field="canonical_bytes",
                value_type=type(self.canonical_bytes).__name__,
            )
        try:
            decoded = json.loads(self.canonical_bytes.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise contract_invalid(
                "canonical payload is not valid ASCII JSON", field="canonical_bytes"
            ) from error
        if canonical_json_bytes(decoded) != self.canonical_bytes:
            raise contract_invalid(
                "payload bytes are not in canonical JSON form", field="canonical_bytes"
            )

    @classmethod
    def from_value(
        cls, schema_version: SchemaVersion, value: object
    ) -> "CanonicalPayloadV1":
        return cls(schema_version=schema_version, canonical_bytes=canonical_json_bytes(value))

    def to_value(self) -> object:
        return json.loads(self.canonical_bytes.decode("ascii"))

    def content_digest(self) -> ContentDigest:
        return canonical_sha256(
            {
                "schema_version": str(self.schema_version),
                "payload": self.to_value(),
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "payload": self.to_value(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "CanonicalPayloadV1":
        if not isinstance(value, dict) or set(value) != {"schema_version", "payload"}:
            raise contract_invalid(
                "canonical payload object has an invalid shape", field="payload"
            )
        return cls.from_value(SchemaVersion(value["schema_version"]), value["payload"])
