"""C-080 checkpoint views and C-090 RNG authority descriptors/ports."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from ._validation import (
    require_instance,
    require_int,
    require_text,
    require_tuple_instances,
)
from .artifacts import RunArtifactRefV1
from .errors import contract_invalid
from .ids import (
    ArtifactId,
    BranchId,
    CheckpointId,
    ComponentId,
    ContentDigest,
    LeaseId,
    ModuleId,
    ParticipantId,
    RunId,
    SchemaVersion,
)
from .payloads import CanonicalPayloadV1
from .runtime import RunCursorV1


C080_API_VERSION = SchemaVersion("c080_v1")
C090_API_VERSION = SchemaVersion("c090_v1")
STATE_PARTICIPANT_SCHEMA_VERSION = SchemaVersion("state_participant_descriptor_v1")
COMPOSITE_CHECKPOINT_SCHEMA_VERSION = SchemaVersion("composite_checkpoint_view_v1")
RNG_LEASE_SCHEMA_VERSION = SchemaVersion("rng_lease_descriptor_v1")


class StateMaterialRoleV1(str, Enum):
    DERIVED_REBUILDABLE = "DERIVED_REBUILDABLE"
    CONTINUATION_REQUIRED = "CONTINUATION_REQUIRED"
    AUTHORITATIVE = "AUTHORITATIVE"


@dataclass(frozen=True, slots=True)
class StateParticipantDescriptorV1:
    participant_id: ParticipantId
    module_id: ModuleId
    state_schema_version: SchemaVersion
    material_role: StateMaterialRoleV1
    stable_order: int
    schema_version: SchemaVersion = STATE_PARTICIPANT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(self.participant_id, ParticipantId, field="participant_id")
        require_instance(self.module_id, ModuleId, field="module_id")
        require_instance(
            self.state_schema_version, SchemaVersion, field="state_schema_version"
        )
        require_instance(self.material_role, StateMaterialRoleV1, field="material_role")
        require_int(self.stable_order, field="stable_order")
        if self.schema_version != STATE_PARTICIPANT_SCHEMA_VERSION:
            raise contract_invalid(
                "unsupported participant descriptor schema", field="schema_version"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "participant_id": str(self.participant_id),
            "module_id": str(self.module_id),
            "state_schema_version": str(self.state_schema_version),
            "material_role": self.material_role.value,
            "stable_order": self.stable_order,
        }


@dataclass(frozen=True, slots=True)
class SnapshotParticipantV1:
    descriptor: StateParticipantDescriptorV1
    payload_ref: RunArtifactRefV1
    payload_digest: ContentDigest

    def __post_init__(self) -> None:
        require_instance(
            self.descriptor, StateParticipantDescriptorV1, field="descriptor"
        )
        require_instance(self.payload_ref, RunArtifactRefV1, field="payload_ref")
        require_instance(self.payload_digest, ContentDigest, field="payload_digest")
        if self.payload_ref.content_digest != self.payload_digest:
            raise contract_invalid(
                "participant payload digest does not match its artifact ref",
                field="payload_digest",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "descriptor": self.descriptor.to_dict(),
            "payload_ref": self.payload_ref.to_dict(),
            "payload_digest": self.payload_digest.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RNGLeaseDescriptorV1:
    lease_id: LeaseId
    owner_component_id: ComponentId
    semantic_domain: str
    stream_identity: str
    draw_index: int
    budget_contract_ref: ArtifactId | None
    device_scope: str
    schema_version: SchemaVersion = RNG_LEASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(self.lease_id, LeaseId, field="lease_id")
        require_instance(
            self.owner_component_id, ComponentId, field="owner_component_id"
        )
        require_text(self.semantic_domain, field="semantic_domain")
        require_text(self.stream_identity, field="stream_identity")
        require_int(self.draw_index, field="draw_index")
        if self.budget_contract_ref is not None:
            require_instance(
                self.budget_contract_ref, ArtifactId, field="budget_contract_ref"
            )
        require_text(self.device_scope, field="device_scope")
        if self.schema_version != RNG_LEASE_SCHEMA_VERSION:
            raise contract_invalid("unsupported RNG lease schema", field="schema_version")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "lease_id": str(self.lease_id),
            "owner_component_id": str(self.owner_component_id),
            "semantic_domain": self.semantic_domain,
            "stream_identity": self.stream_identity,
            "draw_index": self.draw_index,
            "budget_contract_ref": (
                None if self.budget_contract_ref is None else str(self.budget_contract_ref)
            ),
            "device_scope": self.device_scope,
        }


@runtime_checkable
class RNGLeaseV1(Protocol):
    """Borrowed tracked handle; implementations remain owned by Runtime."""

    def descriptor(self) -> RNGLeaseDescriptorV1: ...

    def tracked_handle(self) -> object: ...

    def index_snapshot(self) -> int: ...

    def assert_budget(self) -> None: ...


@runtime_checkable
class RNGAuthorityV1(Protocol):
    def lease(self, request: RNGLeaseDescriptorV1) -> RNGLeaseV1: ...

    def root_snapshot_ref(self) -> RunArtifactRefV1: ...

    def restore_root_snapshot(self, snapshot_ref: RunArtifactRefV1) -> None: ...


@runtime_checkable
class SnapshotParticipantPortV1(Protocol):
    def descriptor(self) -> StateParticipantDescriptorV1: ...

    def export_state(self, checkpoint_context: CanonicalPayloadV1) -> RunArtifactRefV1: ...

    def validate_state(self, payload_ref: RunArtifactRefV1) -> None: ...

    def import_state(
        self, payload_ref: RunArtifactRefV1, restore_context: CanonicalPayloadV1
    ) -> None: ...

    def state_digest(self, payload_ref: RunArtifactRefV1) -> ContentDigest: ...

    def compatibility_check(self, payload_metadata: CanonicalPayloadV1) -> None: ...


@dataclass(frozen=True, slots=True)
class CompositeCheckpointViewV1:
    checkpoint_id: CheckpointId
    run_id: RunId
    branch_id: BranchId
    run_cursor: RunCursorV1
    runtime_state_ref: RunArtifactRefV1
    rng_root_snapshot_ref: RunArtifactRefV1
    control_position: CanonicalPayloadV1
    participants: tuple[SnapshotParticipantV1, ...]
    backend_state_ref: RunArtifactRefV1 | None
    assembly_identity: ContentDigest
    compatibility_metadata: CanonicalPayloadV1
    schema_version: SchemaVersion = COMPOSITE_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name, expected in (
            ("checkpoint_id", CheckpointId),
            ("run_id", RunId),
            ("branch_id", BranchId),
            ("run_cursor", RunCursorV1),
            ("runtime_state_ref", RunArtifactRefV1),
            ("rng_root_snapshot_ref", RunArtifactRefV1),
            ("control_position", CanonicalPayloadV1),
            ("assembly_identity", ContentDigest),
            ("compatibility_metadata", CanonicalPayloadV1),
        ):
            require_instance(getattr(self, field_name), expected, field=field_name)
        require_tuple_instances(
            self.participants,
            SnapshotParticipantV1,
            field="participants",
            unique=True,
        )
        orders = [participant.descriptor.stable_order for participant in self.participants]
        if orders != sorted(set(orders)):
            raise contract_invalid(
                "checkpoint participants must have stable unique ordering",
                field="participants",
            )
        if self.backend_state_ref is not None:
            require_instance(
                self.backend_state_ref, RunArtifactRefV1, field="backend_state_ref"
            )
        if self.schema_version != COMPOSITE_CHECKPOINT_SCHEMA_VERSION:
            raise contract_invalid(
                "unsupported composite checkpoint view schema", field="schema_version"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "checkpoint_id": str(self.checkpoint_id),
            "run_id": str(self.run_id),
            "branch_id": str(self.branch_id),
            "run_cursor": self.run_cursor.to_dict(),
            "runtime_state_ref": self.runtime_state_ref.to_dict(),
            "rng_root_snapshot_ref": self.rng_root_snapshot_ref.to_dict(),
            "control_position": self.control_position.to_dict(),
            "participants": [participant.to_dict() for participant in self.participants],
            "backend_state_ref": (
                None if self.backend_state_ref is None else self.backend_state_ref.to_dict()
            ),
            "assembly_identity": self.assembly_identity.to_dict(),
            "compatibility_metadata": self.compatibility_metadata.to_dict(),
        }
