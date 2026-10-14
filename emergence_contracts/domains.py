"""C-040/C-050/C-060 semantic component DTOs and Protocol surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ._validation import require_instance, require_text, require_tuple_instances
from .components import ComponentDescriptorV1
from .errors import contract_invalid
from .ids import (
    ArtifactId,
    CandidateId,
    ComponentInstanceId,
    EntityId,
    LineageId,
    SchemaVersion,
)
from .payloads import CanonicalPayloadV1
from .runtime import RunCursorV1
from .services import RuntimeServicesV1
from .state import RNGLeaseV1, SnapshotParticipantPortV1, StateParticipantDescriptorV1


C040_API_VERSION = SchemaVersion("c040_v1")
C050_API_VERSION = SchemaVersion("c050_v1")
C060_API_VERSION = SchemaVersion("c060_v1")


@dataclass(frozen=True, slots=True)
class InitializationContextV1:
    cursor: RunCursorV1
    payload: CanonicalPayloadV1

    def __post_init__(self) -> None:
        require_instance(self.cursor, RunCursorV1, field="cursor")
        require_instance(self.payload, CanonicalPayloadV1, field="payload")


@dataclass(frozen=True, slots=True)
class EntityBatchViewV1:
    entity_ids: tuple[EntityId, ...]
    payload: CanonicalPayloadV1

    def __post_init__(self) -> None:
        require_tuple_instances(
            self.entity_ids,
            EntityId,
            field="entity_ids",
            unique=True,
            sorted_by_string=True,
        )
        require_instance(self.payload, CanonicalPayloadV1, field="payload")


@dataclass(frozen=True, slots=True)
class SpaceDescriptorV1:
    space_kind: str
    schema_version: SchemaVersion
    payload: CanonicalPayloadV1

    def __post_init__(self) -> None:
        require_text(self.space_kind, field="space_kind")
        require_instance(self.schema_version, SchemaVersion, field="schema_version")
        require_instance(self.payload, CanonicalPayloadV1, field="payload")


@dataclass(frozen=True, slots=True)
class CursorPayloadBatchV1:
    cursor: RunCursorV1
    payload: CanonicalPayloadV1

    def __post_init__(self) -> None:
        require_instance(self.cursor, RunCursorV1, field="cursor")
        require_instance(self.payload, CanonicalPayloadV1, field="payload")


@dataclass(frozen=True, slots=True)
class ObservationBatchV1(CursorPayloadBatchV1):
    pass


@dataclass(frozen=True, slots=True)
class ActionBatchV1(CursorPayloadBatchV1):
    pass


@dataclass(frozen=True, slots=True)
class WorldActionResultV1(CursorPayloadBatchV1):
    pass


@dataclass(frozen=True, slots=True)
class WorldEventBatchV1(CursorPayloadBatchV1):
    pass


@dataclass(frozen=True, slots=True)
class OutcomeBatchV1(CursorPayloadBatchV1):
    pass


@dataclass(frozen=True, slots=True)
class AgentBlueprintV1:
    entity_id: EntityId
    candidate_id: CandidateId
    heritable_payload_ref: ArtifactId
    agent_config: CanonicalPayloadV1

    def __post_init__(self) -> None:
        require_instance(self.entity_id, EntityId, field="entity_id")
        require_instance(self.candidate_id, CandidateId, field="candidate_id")
        require_instance(
            self.heritable_payload_ref, ArtifactId, field="heritable_payload_ref"
        )
        require_instance(self.agent_config, CanonicalPayloadV1, field="agent_config")


@dataclass(frozen=True, slots=True)
class AgentRuntimeHandleV1:
    instance_id: ComponentInstanceId

    def __post_init__(self) -> None:
        require_instance(self.instance_id, ComponentInstanceId, field="instance_id")


@dataclass(frozen=True, slots=True)
class TurnoverContextV1:
    cursor: RunCursorV1
    payload: CanonicalPayloadV1

    def __post_init__(self) -> None:
        require_instance(self.cursor, RunCursorV1, field="cursor")
        require_instance(self.payload, CanonicalPayloadV1, field="payload")


@dataclass(frozen=True, slots=True)
class PopulationDeltaV1:
    activated_entities: tuple[EntityId, ...]
    deactivated_entities: tuple[EntityId, ...]
    payload: CanonicalPayloadV1

    def __post_init__(self) -> None:
        for field_name in ("activated_entities", "deactivated_entities"):
            require_tuple_instances(
                getattr(self, field_name),
                EntityId,
                field=field_name,
                unique=True,
                sorted_by_string=True,
            )
        if set(self.activated_entities) & set(self.deactivated_entities):
            raise contract_invalid(
                "an entity cannot be activated and deactivated together",
                field="population_delta",
            )
        require_instance(self.payload, CanonicalPayloadV1, field="payload")


@dataclass(frozen=True, slots=True)
class LineageViewV1:
    lineage_ids: tuple[LineageId, ...]
    payload: CanonicalPayloadV1

    def __post_init__(self) -> None:
        require_tuple_instances(
            self.lineage_ids,
            LineageId,
            field="lineage_ids",
            unique=True,
            sorted_by_string=True,
        )
        require_instance(self.payload, CanonicalPayloadV1, field="payload")


@runtime_checkable
class WorldComponentV1(Protocol):
    def descriptor(self) -> ComponentDescriptorV1: ...

    def configure(self, world_config_fragment: CanonicalPayloadV1) -> None: ...

    def initialize(
        self, context: InitializationContextV1, runtime_services: RuntimeServicesV1
    ) -> None: ...

    def observation_spec(self) -> SpaceDescriptorV1: ...

    def observe(
        self, entity_view: EntityBatchViewV1, cursor: RunCursorV1
    ) -> ObservationBatchV1: ...

    def apply_actions(
        self, action_batch: ActionBatchV1, cursor: RunCursorV1
    ) -> WorldActionResultV1: ...

    def advance_environment(self, cursor: RunCursorV1) -> WorldEventBatchV1: ...

    def state_port(self) -> SnapshotParticipantPortV1: ...

    def state_descriptor(self) -> StateParticipantDescriptorV1: ...

    def export_state(self, checkpoint_context: CanonicalPayloadV1) -> CanonicalPayloadV1: ...

    def import_state(self, payload: CanonicalPayloadV1) -> None: ...

    def validate_state(self, payload: CanonicalPayloadV1) -> None: ...


@runtime_checkable
class AgentComponentV1(Protocol):
    def descriptor(self) -> ComponentDescriptorV1: ...

    def configure(self, agent_config_fragment: CanonicalPayloadV1) -> None: ...

    def instantiate(
        self,
        agent_blueprints: tuple[AgentBlueprintV1, ...],
        runtime_services: RuntimeServicesV1,
    ) -> AgentRuntimeHandleV1: ...

    def act(
        self,
        observation_batch: ObservationBatchV1,
        entity_runtime_view: EntityBatchViewV1,
        cursor: RunCursorV1,
    ) -> ActionBatchV1: ...

    def apply_world_feedback(
        self, world_action_result: WorldActionResultV1, cursor: RunCursorV1
    ) -> None: ...

    def state_port(self) -> SnapshotParticipantPortV1: ...

    def export_runtime_state(self) -> CanonicalPayloadV1: ...

    def import_runtime_state(self, payload: CanonicalPayloadV1) -> None: ...

    def state_descriptor(self) -> StateParticipantDescriptorV1: ...


@runtime_checkable
class AdaptationComponentV1(Protocol):
    def descriptor(self) -> ComponentDescriptorV1: ...

    def configure(self, adaptation_config_fragment: CanonicalPayloadV1) -> None: ...

    def initialize_population(
        self, context: InitializationContextV1, rng_lease: RNGLeaseV1
    ) -> None: ...

    def active_entities(self) -> EntityBatchViewV1: ...

    def agent_blueprints(
        self, entity_ids: tuple[EntityId, ...]
    ) -> tuple[AgentBlueprintV1, ...]: ...

    def record_outcomes(self, outcome_batch: OutcomeBatchV1) -> None: ...

    def on_turnover(self, context: TurnoverContextV1) -> PopulationDeltaV1: ...

    def lineage_view(self) -> LineageViewV1: ...

    def state_port(self) -> SnapshotParticipantPortV1: ...

    def export_state(self) -> CanonicalPayloadV1: ...

    def import_state(self, payload: CanonicalPayloadV1) -> None: ...

    def state_descriptor(self) -> StateParticipantDescriptorV1: ...
