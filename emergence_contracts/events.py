"""C-100 immutable runtime event and telemetry envelope."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ._validation import require_instance, require_int, require_text
from .errors import contract_invalid
from .ids import CellId, ComponentId, ContentDigest, ModuleId, RunId, SchemaVersion
from .payloads import CanonicalPayloadV1
from .runtime import RunCursorV1


CONTRACT_ID = "C-100"
CONTRACT_OWNER = "Contracts; ordering and durability implemented by Runtime"
API_VERSION = SchemaVersion("c100_v1")
EVENT_SCHEMA_VERSION = SchemaVersion("runtime_event_v1")


@dataclass(frozen=True, slots=True)
class RuntimeEventV1:
    run_id: RunId
    cell_id: CellId
    cursor: RunCursorV1
    source_module_id: ModuleId
    source_component_id: ComponentId
    event_type: str
    payload: CanonicalPayloadV1
    submission_sequence: int | None = None
    sequence_digest: ContentDigest | None = None
    schema_version: SchemaVersion = EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name, expected in (
            ("run_id", RunId),
            ("cell_id", CellId),
            ("cursor", RunCursorV1),
            ("source_module_id", ModuleId),
            ("source_component_id", ComponentId),
            ("payload", CanonicalPayloadV1),
        ):
            require_instance(getattr(self, field_name), expected, field=field_name)
        require_text(self.event_type, field="event_type")
        if (self.submission_sequence is None) == (self.sequence_digest is None):
            raise contract_invalid(
                "exactly one event sequence identity is required",
                field="submission_sequence",
            )
        if self.submission_sequence is not None:
            require_int(self.submission_sequence, field="submission_sequence")
        if self.sequence_digest is not None:
            require_instance(self.sequence_digest, ContentDigest, field="sequence_digest")
        if self.schema_version != EVENT_SCHEMA_VERSION:
            raise contract_invalid("unsupported runtime event schema", field="schema_version")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "run_id": str(self.run_id),
            "cell_id": str(self.cell_id),
            "next_tick_or_event_cursor": self.cursor.to_dict(),
            "source_module_id": str(self.source_module_id),
            "source_component_id": str(self.source_component_id),
            "event_type": self.event_type,
            "payload_schema_version": str(self.payload.schema_version),
            "payload": self.payload.to_value(),
            "submission_sequence": self.submission_sequence,
            "sequence_digest": (
                None if self.sequence_digest is None else self.sequence_digest.to_dict()
            ),
        }


@runtime_checkable
class TelemetrySinkV1(Protocol):
    def emit(self, event: RuntimeEventV1) -> None: ...
