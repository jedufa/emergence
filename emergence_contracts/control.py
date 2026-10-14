"""C-160 deterministic safe-boundary control command and port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ._validation import require_instance, require_int, require_text
from .errors import contract_invalid
from .ids import CommandId, ContentDigest, RunId, SchemaVersion
from .payloads import CanonicalPayloadV1
from .runtime import RunCursorV1


CONTRACT_ID = "C-160"
CONTRACT_OWNER = "Contracts; deterministic application point owned by Runtime"
API_VERSION = SchemaVersion("c160_v1")
CONTROL_COMMAND_SCHEMA_VERSION = SchemaVersion("control_command_v1")


@dataclass(frozen=True, slots=True)
class ControlTargetV1:
    next_tick: int | None = None
    boundary_name: str | None = None

    def __post_init__(self) -> None:
        if (self.next_tick is None) == (self.boundary_name is None):
            raise contract_invalid(
                "control target requires exactly one cursor or boundary",
                field="control_target",
            )
        if self.next_tick is not None:
            require_int(self.next_tick, field="next_tick")
        if self.boundary_name is not None:
            require_text(self.boundary_name, field="boundary_name")


@dataclass(frozen=True, slots=True)
class ControlCommandV1:
    command_id: CommandId
    target_run_id: RunId
    target: ControlTargetV1
    command_type: str
    payload: CanonicalPayloadV1
    submission_sequence: int
    expectation_or_reason: str
    idempotency_key: ContentDigest
    schema_version: SchemaVersion = CONTROL_COMMAND_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(self.command_id, CommandId, field="command_id")
        require_instance(self.target_run_id, RunId, field="target_run_id")
        require_instance(self.target, ControlTargetV1, field="target")
        require_text(self.command_type, field="command_type")
        require_instance(self.payload, CanonicalPayloadV1, field="payload")
        require_int(self.submission_sequence, field="submission_sequence")
        require_text(self.expectation_or_reason, field="expectation_or_reason")
        require_instance(self.idempotency_key, ContentDigest, field="idempotency_key")
        if self.schema_version != CONTROL_COMMAND_SCHEMA_VERSION:
            raise contract_invalid("unsupported control command schema", field="schema_version")


@dataclass(frozen=True, slots=True)
class ControlResultV1:
    command_id: CommandId
    applied: bool
    applied_cursor: RunCursorV1 | None
    result: CanonicalPayloadV1

    def __post_init__(self) -> None:
        require_instance(self.command_id, CommandId, field="command_id")
        if not isinstance(self.applied, bool):
            raise contract_invalid("applied must be boolean", field="applied")
        if self.applied_cursor is not None:
            require_instance(self.applied_cursor, RunCursorV1, field="applied_cursor")
        require_instance(self.result, CanonicalPayloadV1, field="result")


@runtime_checkable
class ControlPortV1(Protocol):
    def submit(self, command: ControlCommandV1) -> ControlResultV1: ...

    def status(self, command_id: CommandId) -> ControlResultV1: ...
