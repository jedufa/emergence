"""C-110 Runtime Engine / Execution Provider contract over one lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from ._validation import (
    require_instance,
    require_text,
    require_tuple_instances,
)
from .artifacts import RunArtifactRefV1
from .assembly import SystemAssemblyV1
from .capabilities import RuntimeCapabilityProfileV1
from .control import ControlResultV1
from .errors import ContractErrorCategory, contract_invalid
from .ids import (
    AssemblyId,
    BranchId,
    CellId,
    ComponentId,
    ComponentInstanceId,
    ContentDigest,
    RunId,
    SchemaVersion,
)
from .payloads import CanonicalPayloadV1
from .runtime import ExecutionScheduleV1, RunCursorV1
from .services import ArtifactOutputPortV1


CONTRACT_ID = "C-110"
CONTRACT_OWNER = "Contracts; implemented by Runtime"
API_VERSION = SchemaVersion("c110_v1")
EXECUTION_REQUEST_SCHEMA_VERSION = SchemaVersion("runtime_execution_request_v1")


class RuntimeStatusCodeV1(str, Enum):
    PREPARED = "PREPARED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class ContractIssueV1:
    category: ContractErrorCategory
    owner: str
    message: str

    def __post_init__(self) -> None:
        require_instance(self.category, ContractErrorCategory, field="category")
        require_text(self.owner, field="owner")
        require_text(self.message, field="message")


@dataclass(frozen=True, slots=True)
class ValidationResultV1:
    valid: bool
    issues: tuple[ContractIssueV1, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise contract_invalid("valid must be boolean", field="valid")
        require_tuple_instances(self.issues, ContractIssueV1, field="issues")
        if self.valid and self.issues:
            raise contract_invalid(
                "a valid result cannot contain issues", field="issues"
            )


@dataclass(frozen=True, slots=True)
class RuntimeExecutionRequestV1:
    run_id: RunId
    cell_id: CellId
    schedule: ExecutionScheduleV1
    assembly_id: AssemblyId
    config_digest: ContentDigest
    resume_checkpoint_ref: RunArtifactRefV1 | None = None
    request_metadata: CanonicalPayloadV1 | None = None
    schema_version: SchemaVersion = EXECUTION_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name, expected in (
            ("run_id", RunId),
            ("cell_id", CellId),
            ("schedule", ExecutionScheduleV1),
            ("assembly_id", AssemblyId),
            ("config_digest", ContentDigest),
        ):
            require_instance(getattr(self, field_name), expected, field=field_name)
        if self.resume_checkpoint_ref is not None:
            require_instance(
                self.resume_checkpoint_ref,
                RunArtifactRefV1,
                field="resume_checkpoint_ref",
            )
        if self.request_metadata is not None:
            require_instance(
                self.request_metadata, CanonicalPayloadV1, field="request_metadata"
            )
        if self.schema_version != EXECUTION_REQUEST_SCHEMA_VERSION:
            raise contract_invalid("unsupported execution request schema", field="schema_version")


@dataclass(frozen=True, slots=True)
class PreparedExecutionV1:
    provider_id: ComponentId
    runtime_instance_id: ComponentInstanceId
    request_digest: ContentDigest
    assembly_digest: ContentDigest
    opaque_prepared_ref: str

    def __post_init__(self) -> None:
        for field_name, expected in (
            ("provider_id", ComponentId),
            ("runtime_instance_id", ComponentInstanceId),
            ("request_digest", ContentDigest),
            ("assembly_digest", ContentDigest),
        ):
            require_instance(getattr(self, field_name), expected, field=field_name)
        require_text(self.opaque_prepared_ref, field="opaque_prepared_ref")


@dataclass(frozen=True, slots=True)
class RuntimeResultV1:
    run_id: RunId
    status: RuntimeStatusCodeV1
    final_cursor: RunCursorV1
    run_manifest_ref: RunArtifactRefV1 | None
    result_metadata: CanonicalPayloadV1

    def __post_init__(self) -> None:
        require_instance(self.run_id, RunId, field="run_id")
        require_instance(self.status, RuntimeStatusCodeV1, field="status")
        require_instance(self.final_cursor, RunCursorV1, field="final_cursor")
        if self.run_manifest_ref is not None:
            require_instance(
                self.run_manifest_ref, RunArtifactRefV1, field="run_manifest_ref"
            )
        require_instance(self.result_metadata, CanonicalPayloadV1, field="result_metadata")


@dataclass(frozen=True, slots=True)
class BranchSpecV1:
    branch_id: BranchId
    branch_metadata: CanonicalPayloadV1

    def __post_init__(self) -> None:
        require_instance(self.branch_id, BranchId, field="branch_id")
        require_instance(self.branch_metadata, CanonicalPayloadV1, field="branch_metadata")


@runtime_checkable
class RuntimeProviderV1(Protocol):
    def capabilities(self) -> RuntimeCapabilityProfileV1: ...

    def validate_assembly(self, assembly: SystemAssemblyV1) -> ValidationResultV1: ...

    def prepare(
        self, request: RuntimeExecutionRequestV1, assembly: SystemAssemblyV1
    ) -> PreparedExecutionV1: ...

    def execute(
        self, prepared: PreparedExecutionV1, output_port: ArtifactOutputPortV1
    ) -> RuntimeResultV1: ...

    def checkpoint(
        self, runtime_instance_id: ComponentInstanceId, reason: str
    ) -> RunArtifactRefV1: ...

    def resume(
        self,
        checkpoint_ref: RunArtifactRefV1,
        request_override: RuntimeExecutionRequestV1 | None = None,
    ) -> RuntimeResultV1: ...

    def branch(
        self, checkpoint_ref: RunArtifactRefV1, branch_spec: BranchSpecV1
    ) -> RunArtifactRefV1: ...

    def cancel(self, runtime_instance_id: ComponentInstanceId) -> ControlResultV1: ...

    def status(self, runtime_instance_id: ComponentInstanceId) -> RuntimeResultV1: ...
