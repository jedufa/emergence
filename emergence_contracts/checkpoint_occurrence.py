"""CRCA checkpoint occurrence and conformance-index contracts.

The legacy ``CheckpointOccurrenceV1`` describes an already sealed
CheckpointBundleV2 transaction and its ``bundle_v2_digest`` remains exactly a
BundleV2 tree digest.  The absolute-tick DTOs below are additive and do not
reinterpret that field or format.  These contracts do not own checkpoint state,
change sealed bytes, publish a continuation, grant a lease, or make a
qualification decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath

from ._validation import require_instance, require_int, require_text
from .canonical import canonical_sha256
from .errors import contract_invalid
from .ids import (
    AttemptId,
    CheckpointId,
    ContentDigest,
    ParticipantId,
    RunId,
    SchemaVersion,
)


CHECKPOINT_PARTICIPANT_OCCURRENCE_SCHEMA_VERSION = SchemaVersion(
    "crca_checkpoint_participant_occurrence_v1"
)
CHECKPOINT_OCCURRENCE_SCHEMA_VERSION = SchemaVersion(
    "crca_checkpoint_occurrence_v1"
)
CHECKPOINT_OCCURRENCE_COMMIT_SCHEMA_VERSION = SchemaVersion(
    "crca_checkpoint_occurrence_commit_v1"
)
CHECKPOINT_OCCURRENCE_INSPECTION_SCHEMA_VERSION = SchemaVersion(
    "crca_checkpoint_occurrence_inspection_v1"
)
ABSOLUTE_TICK_CHECKPOINT_PARTICIPANT_OCCURRENCE_SCHEMA_VERSION = SchemaVersion(
    "crca_absolute_tick_checkpoint_participant_occurrence_v1"
)
ABSOLUTE_TICK_CHECKPOINT_OCCURRENCE_SCHEMA_VERSION = SchemaVersion(
    "crca_absolute_tick_checkpoint_occurrence_v1"
)
ABSOLUTE_TICK_CHECKPOINT_OCCURRENCE_COMMIT_SCHEMA_VERSION = SchemaVersion(
    "crca_absolute_tick_checkpoint_occurrence_commit_v1"
)
ABSOLUTE_TICK_CHECKPOINT_OCCURRENCE_INSPECTION_SCHEMA_VERSION = SchemaVersion(
    "crca_absolute_tick_checkpoint_occurrence_inspection_v1"
)
ABSOLUTE_TICK_CHECKPOINT_FORMAT_ID = "eiel_full_evolution_cuda_checkpoint_v2"
ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY = "absolute_tick_v1"
WORKHORSE_SEMANTIC_CHECKPOINT_MATERIAL_SCHEMA_VERSION = SchemaVersion(
    "crca_workhorse_semantic_checkpoint_material_v2"
)
WORKHORSE_SEMANTIC_CHECKPOINT_OCCURRENCE_SCHEMA_VERSION = SchemaVersion(
    "crca_workhorse_semantic_checkpoint_occurrence_v2"
)
WORKHORSE_SEMANTIC_CHECKPOINT_OCCURRENCE_COMMIT_SCHEMA_VERSION = SchemaVersion(
    "crca_workhorse_semantic_checkpoint_occurrence_commit_v2"
)
WORKHORSE_SEMANTIC_CHECKPOINT_OCCURRENCE_INSPECTION_SCHEMA_VERSION = SchemaVersion(
    "crca_workhorse_semantic_checkpoint_occurrence_inspection_v2"
)
WORKHORSE_SEMANTIC_CHECKPOINT_FORMAT_ID = (
    "eiel_workhorse_science_compatible_base_checkpoint_v2"
)
WORKHORSE_SEMANTIC_CHECKPOINT_TIME_AUTHORITY = "absolute_tick_v1"
CHECKPOINT_CONFORMANCE_SENTINEL_SCHEMA_VERSION = SchemaVersion(
    "crca_checkpoint_conformance_sentinel_v1"
)
CHECKPOINT_CONFORMANCE_INDEX_SCHEMA_VERSION = SchemaVersion(
    "crca_checkpoint_conformance_index_v1"
)
CHECKPOINT_CONFORMANCE_RESULT_SCHEMA_VERSION = SchemaVersion(
    "crca_checkpoint_conformance_result_v1"
)
CHECKPOINT_CONFORMANCE_RUN_SCHEMA_VERSION = SchemaVersion(
    "crca_checkpoint_conformance_run_v1"
)


def _digest(value: ContentDigest) -> dict[str, str]:
    require_instance(value, ContentDigest, field="digest")
    return value.to_dict()


def _optional_digest(value: ContentDigest | None) -> dict[str, str] | None:
    return None if value is None else _digest(value)


def _content_digest(value: object, *, field: str) -> ContentDigest:
    if not isinstance(value, dict) or set(value) != {"algorithm", "hex"}:
        raise contract_invalid("digest has an invalid shape", field=field)
    return ContentDigest(
        algorithm=str(value.get("algorithm") or ""),
        hex=str(value.get("hex") or ""),
    )


def _optional_content_digest(value: object, *, field: str) -> ContentDigest | None:
    return None if value is None else _content_digest(value, field=field)


def _schema(actual: SchemaVersion, expected: SchemaVersion) -> None:
    require_instance(actual, SchemaVersion, field="schema_version")
    if actual != expected:
        raise contract_invalid(
            "unsupported checkpoint occurrence schema",
            field="schema_version",
        )


def _canonical_utc(value: object, *, field: str) -> str:
    require_text(value, field=field)
    if not isinstance(value, str) or len(value) != 20 or not value.endswith("Z"):
        raise contract_invalid(
            "timestamp must use canonical second-resolution UTC",
            field=field,
        )
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise contract_invalid("timestamp is invalid", field=field) from exc
    return value


class CheckpointOccurrenceInspectionStatusV1(str, Enum):
    VISIBLE = "VISIBLE"
    ORPHAN_SEALED_BUNDLE = "ORPHAN_SEALED_BUNDLE"
    QUARANTINED_INCOMPLETE_BUNDLE = "QUARANTINED_INCOMPLETE_BUNDLE"
    QUARANTINED_PARTIAL_OCCURRENCE = "QUARANTINED_PARTIAL_OCCURRENCE"
    QUARANTINED_CONFLICT = "QUARANTINED_CONFLICT"


class WorkhorseSemanticCheckpointMaterialRoleV2(str, Enum):
    BASE_CHECKPOINT_MANIFEST = "BASE_CHECKPOINT_MANIFEST"
    BASE_CHECKPOINT_STATE = "BASE_CHECKPOINT_STATE"
    EVENT_ACKNOWLEDGEMENT = "EVENT_ACKNOWLEDGEMENT"
    SCIENCE_CHECKPOINT_COMPATIBILITY = "SCIENCE_CHECKPOINT_COMPATIBILITY"


class CheckpointConformanceLayerV1(str, Enum):
    CONTRACT = "CONTRACT"
    RUNTIME = "RUNTIME"
    BACKEND = "BACKEND"
    QUALIFICATION = "QUALIFICATION"


class CheckpointConformanceStatusV1(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class CheckpointParticipantOccurrenceV1:
    participant_id: ParticipantId
    run_id: RunId
    attempt_id: AttemptId
    transaction_id: str
    absolute_next_tick: int
    owner_id: str
    owner_digest: ContentDigest
    state_digest: ContentDigest
    schema_version: SchemaVersion = (
        CHECKPOINT_PARTICIPANT_OCCURRENCE_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        require_instance(self.participant_id, ParticipantId, field="participant_id")
        require_instance(self.run_id, RunId, field="run_id")
        require_instance(self.attempt_id, AttemptId, field="attempt_id")
        require_text(self.transaction_id, field="transaction_id")
        require_int(self.absolute_next_tick, field="absolute_next_tick")
        require_text(self.owner_id, field="owner_id")
        require_instance(self.owner_digest, ContentDigest, field="owner_digest")
        require_instance(self.state_digest, ContentDigest, field="state_digest")
        _schema(
            self.schema_version,
            CHECKPOINT_PARTICIPANT_OCCURRENCE_SCHEMA_VERSION,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "participant_id": str(self.participant_id),
            "run_id": str(self.run_id),
            "attempt_id": str(self.attempt_id),
            "transaction_id": self.transaction_id,
            "absolute_next_tick": self.absolute_next_tick,
            "owner_id": self.owner_id,
            "owner_digest": _digest(self.owner_digest),
            "state_digest": _digest(self.state_digest),
        }

    def occurrence_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(
        cls, value: object
    ) -> "CheckpointParticipantOccurrenceV1":
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "participant_id",
            "run_id",
            "attempt_id",
            "transaction_id",
            "absolute_next_tick",
            "owner_id",
            "owner_digest",
            "state_digest",
        }:
            raise contract_invalid(
                "participant occurrence has an invalid shape",
                field="participant_occurrence",
            )
        return cls(
            participant_id=ParticipantId(str(value["participant_id"])),
            run_id=RunId(str(value["run_id"])),
            attempt_id=AttemptId(str(value["attempt_id"])),
            transaction_id=str(value["transaction_id"]),
            absolute_next_tick=value["absolute_next_tick"],
            owner_id=str(value["owner_id"]),
            owner_digest=_content_digest(value["owner_digest"], field="owner_digest"),
            state_digest=_content_digest(value["state_digest"], field="state_digest"),
            schema_version=SchemaVersion(str(value["schema_version"])),
        )


@dataclass(frozen=True, slots=True)
class CheckpointOccurrenceV1:
    checkpoint_id: CheckpointId
    run_id: RunId
    attempt_id: AttemptId
    transaction_id: str
    absolute_next_tick: int
    semantic_spec_digest: ContentDigest
    implementation_digest: ContentDigest
    participant_occurrences: tuple[CheckpointParticipantOccurrenceV1, ...]
    bundle_v2_digest: ContentDigest
    completion_marker_digest: ContentDigest
    parent_checkpoint_digest: ContentDigest | None
    schema_version: SchemaVersion = CHECKPOINT_OCCURRENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(self.checkpoint_id, CheckpointId, field="checkpoint_id")
        require_instance(self.run_id, RunId, field="run_id")
        require_instance(self.attempt_id, AttemptId, field="attempt_id")
        require_text(self.transaction_id, field="transaction_id")
        require_int(self.absolute_next_tick, field="absolute_next_tick")
        for field_name in (
            "semantic_spec_digest",
            "implementation_digest",
            "bundle_v2_digest",
            "completion_marker_digest",
        ):
            require_instance(getattr(self, field_name), ContentDigest, field=field_name)
        if self.parent_checkpoint_digest is not None:
            require_instance(
                self.parent_checkpoint_digest,
                ContentDigest,
                field="parent_checkpoint_digest",
            )
        if not isinstance(self.participant_occurrences, tuple) or not (
            self.participant_occurrences
        ):
            raise contract_invalid(
                "checkpoint occurrence needs participants",
                field="participant_occurrences",
            )
        for participant in self.participant_occurrences:
            require_instance(
                participant,
                CheckpointParticipantOccurrenceV1,
                field="participant_occurrences",
            )
            if (
                participant.run_id != self.run_id
                or participant.attempt_id != self.attempt_id
                or participant.transaction_id != self.transaction_id
                or participant.absolute_next_tick != self.absolute_next_tick
            ):
                raise contract_invalid(
                    "checkpoint participant differs from its occurrence coordinates",
                    field="participant_occurrences",
                )
        identifiers = [
            str(participant.participant_id)
            for participant in self.participant_occurrences
        ]
        if identifiers != sorted(set(identifiers)):
            raise contract_invalid(
                "checkpoint participants must be sorted and unique",
                field="participant_occurrences",
            )
        _schema(self.schema_version, CHECKPOINT_OCCURRENCE_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "checkpoint_id": str(self.checkpoint_id),
            "run_id": str(self.run_id),
            "attempt_id": str(self.attempt_id),
            "transaction_id": self.transaction_id,
            "absolute_next_tick": self.absolute_next_tick,
            "semantic_spec_digest": _digest(self.semantic_spec_digest),
            "implementation_digest": _digest(self.implementation_digest),
            "participant_occurrences": [
                participant.to_dict()
                for participant in self.participant_occurrences
            ],
            "bundle_v2_digest": _digest(self.bundle_v2_digest),
            "completion_marker_digest": _digest(self.completion_marker_digest),
            "parent_checkpoint_digest": _optional_digest(
                self.parent_checkpoint_digest
            ),
        }

    def occurrence_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "CheckpointOccurrenceV1":
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "checkpoint_id",
            "run_id",
            "attempt_id",
            "transaction_id",
            "absolute_next_tick",
            "semantic_spec_digest",
            "implementation_digest",
            "participant_occurrences",
            "bundle_v2_digest",
            "completion_marker_digest",
            "parent_checkpoint_digest",
        }:
            raise contract_invalid(
                "checkpoint occurrence has an invalid shape",
                field="checkpoint_occurrence",
            )
        participants = value["participant_occurrences"]
        if not isinstance(participants, list):
            raise contract_invalid(
                "participant occurrences must be an array",
                field="participant_occurrences",
            )
        return cls(
            checkpoint_id=CheckpointId(str(value["checkpoint_id"])),
            run_id=RunId(str(value["run_id"])),
            attempt_id=AttemptId(str(value["attempt_id"])),
            transaction_id=str(value["transaction_id"]),
            absolute_next_tick=value["absolute_next_tick"],
            semantic_spec_digest=_content_digest(
                value["semantic_spec_digest"], field="semantic_spec_digest"
            ),
            implementation_digest=_content_digest(
                value["implementation_digest"], field="implementation_digest"
            ),
            participant_occurrences=tuple(
                CheckpointParticipantOccurrenceV1.from_dict(item)
                for item in participants
            ),
            bundle_v2_digest=_content_digest(
                value["bundle_v2_digest"], field="bundle_v2_digest"
            ),
            completion_marker_digest=_content_digest(
                value["completion_marker_digest"],
                field="completion_marker_digest",
            ),
            parent_checkpoint_digest=_optional_content_digest(
                value["parent_checkpoint_digest"],
                field="parent_checkpoint_digest",
            ),
            schema_version=SchemaVersion(str(value["schema_version"])),
        )


@dataclass(frozen=True, slots=True)
class CheckpointOccurrenceCommitV1:
    checkpoint_id: CheckpointId
    occurrence_digest: ContentDigest
    bundle_v2_digest: ContentDigest
    completion_marker_digest: ContentDigest
    committed_at_utc: str
    visibility: str = "CANONICAL_CRCA_VISIBLE"
    schema_version: SchemaVersion = CHECKPOINT_OCCURRENCE_COMMIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(self.checkpoint_id, CheckpointId, field="checkpoint_id")
        for field_name in (
            "occurrence_digest",
            "bundle_v2_digest",
            "completion_marker_digest",
        ):
            require_instance(getattr(self, field_name), ContentDigest, field=field_name)
        _canonical_utc(self.committed_at_utc, field="committed_at_utc")
        if self.visibility != "CANONICAL_CRCA_VISIBLE":
            raise contract_invalid(
                "checkpoint occurrence visibility is invalid",
                field="visibility",
            )
        _schema(self.schema_version, CHECKPOINT_OCCURRENCE_COMMIT_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "checkpoint_id": str(self.checkpoint_id),
            "occurrence_digest": _digest(self.occurrence_digest),
            "bundle_v2_digest": _digest(self.bundle_v2_digest),
            "completion_marker_digest": _digest(self.completion_marker_digest),
            "committed_at_utc": self.committed_at_utc,
            "visibility": self.visibility,
        }

    def commit_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "CheckpointOccurrenceCommitV1":
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "checkpoint_id",
            "occurrence_digest",
            "bundle_v2_digest",
            "completion_marker_digest",
            "committed_at_utc",
            "visibility",
        }:
            raise contract_invalid(
                "checkpoint occurrence commit has an invalid shape",
                field="checkpoint_occurrence_commit",
            )
        return cls(
            checkpoint_id=CheckpointId(str(value["checkpoint_id"])),
            occurrence_digest=_content_digest(
                value["occurrence_digest"], field="occurrence_digest"
            ),
            bundle_v2_digest=_content_digest(
                value["bundle_v2_digest"], field="bundle_v2_digest"
            ),
            completion_marker_digest=_content_digest(
                value["completion_marker_digest"],
                field="completion_marker_digest",
            ),
            committed_at_utc=str(value["committed_at_utc"]),
            visibility=str(value["visibility"]),
            schema_version=SchemaVersion(str(value["schema_version"])),
        )


@dataclass(frozen=True, slots=True)
class CheckpointOccurrenceInspectionV1:
    status: CheckpointOccurrenceInspectionStatusV1
    canonical_visible: bool
    bundle_v2_digest: ContentDigest | None
    occurrence_digest: ContentDigest | None
    commit_digest: ContentDigest | None
    problem_digest: ContentDigest | None
    schema_version: SchemaVersion = CHECKPOINT_OCCURRENCE_INSPECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.status, CheckpointOccurrenceInspectionStatusV1):
            raise contract_invalid(
                "checkpoint occurrence inspection status is invalid",
                field="status",
            )
        if type(self.canonical_visible) is not bool:
            raise contract_invalid(
                "canonical_visible must be bool",
                field="canonical_visible",
            )
        if self.canonical_visible != (
            self.status is CheckpointOccurrenceInspectionStatusV1.VISIBLE
        ):
            raise contract_invalid(
                "checkpoint visibility contradicts inspection status",
                field="canonical_visible",
            )
        for field_name in (
            "bundle_v2_digest",
            "occurrence_digest",
            "commit_digest",
            "problem_digest",
        ):
            value = getattr(self, field_name)
            if value is not None:
                require_instance(value, ContentDigest, field=field_name)
        if self.canonical_visible and any(
            value is None
            for value in (
                self.bundle_v2_digest,
                self.occurrence_digest,
                self.commit_digest,
            )
        ):
            raise contract_invalid(
                "visible occurrence lacks a complete digest chain",
                field="canonical_visible",
            )
        if self.canonical_visible and self.problem_digest is not None:
            raise contract_invalid(
                "visible occurrence cannot have a problem digest",
                field="problem_digest",
            )
        _schema(self.schema_version, CHECKPOINT_OCCURRENCE_INSPECTION_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "status": self.status.value,
            "canonical_visible": self.canonical_visible,
            "bundle_v2_digest": _optional_digest(self.bundle_v2_digest),
            "occurrence_digest": _optional_digest(self.occurrence_digest),
            "commit_digest": _optional_digest(self.commit_digest),
            "problem_digest": _optional_digest(self.problem_digest),
        }


@dataclass(frozen=True, slots=True)
class AbsoluteTickCheckpointParticipantOccurrenceV1:
    """One exact native checkpoint file bound to absolute execution coordinates."""

    participant_id: ParticipantId
    run_id: RunId
    attempt_id: AttemptId
    transaction_id: str
    absolute_next_tick: int
    owner_id: str
    owner_digest: ContentDigest
    state_digest: ContentDigest
    schema_version: SchemaVersion = (
        ABSOLUTE_TICK_CHECKPOINT_PARTICIPANT_OCCURRENCE_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        require_instance(self.participant_id, ParticipantId, field="participant_id")
        require_instance(self.run_id, RunId, field="run_id")
        require_instance(self.attempt_id, AttemptId, field="attempt_id")
        require_text(self.transaction_id, field="transaction_id")
        require_int(self.absolute_next_tick, field="absolute_next_tick")
        require_text(self.owner_id, field="owner_id")
        require_instance(self.owner_digest, ContentDigest, field="owner_digest")
        require_instance(self.state_digest, ContentDigest, field="state_digest")
        _schema(
            self.schema_version,
            ABSOLUTE_TICK_CHECKPOINT_PARTICIPANT_OCCURRENCE_SCHEMA_VERSION,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "participant_id": str(self.participant_id),
            "run_id": str(self.run_id),
            "attempt_id": str(self.attempt_id),
            "transaction_id": self.transaction_id,
            "absolute_next_tick": self.absolute_next_tick,
            "owner_id": self.owner_id,
            "owner_digest": _digest(self.owner_digest),
            "state_digest": _digest(self.state_digest),
        }

    def occurrence_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(
        cls, value: object
    ) -> "AbsoluteTickCheckpointParticipantOccurrenceV1":
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "participant_id",
            "run_id",
            "attempt_id",
            "transaction_id",
            "absolute_next_tick",
            "owner_id",
            "owner_digest",
            "state_digest",
        }:
            raise contract_invalid(
                "absolute-tick participant occurrence has an invalid shape",
                field="participant_occurrence",
            )
        return cls(
            participant_id=ParticipantId(str(value["participant_id"])),
            run_id=RunId(str(value["run_id"])),
            attempt_id=AttemptId(str(value["attempt_id"])),
            transaction_id=str(value["transaction_id"]),
            absolute_next_tick=value["absolute_next_tick"],
            owner_id=str(value["owner_id"]),
            owner_digest=_content_digest(value["owner_digest"], field="owner_digest"),
            state_digest=_content_digest(value["state_digest"], field="state_digest"),
            schema_version=SchemaVersion(str(value["schema_version"])),
        )


@dataclass(frozen=True, slots=True)
class AbsoluteTickCheckpointOccurrenceV1:
    """CRCA occurrence for the native three-file absolute-tick checkpoint."""

    checkpoint_format_id: str
    time_authority: str
    checkpoint_id: CheckpointId
    run_id: RunId
    attempt_id: AttemptId
    transaction_id: str
    absolute_next_tick: int
    semantic_spec_digest: ContentDigest
    implementation_digest: ContentDigest
    participant_occurrences: tuple[AbsoluteTickCheckpointParticipantOccurrenceV1, ...]
    checkpoint_bundle_digest: ContentDigest
    completion_marker_digest: ContentDigest
    parent_checkpoint_digest: ContentDigest | None
    schema_version: SchemaVersion = ABSOLUTE_TICK_CHECKPOINT_OCCURRENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.checkpoint_format_id != ABSOLUTE_TICK_CHECKPOINT_FORMAT_ID:
            raise contract_invalid(
                "absolute-tick checkpoint format is invalid",
                field="checkpoint_format_id",
            )
        if self.time_authority != ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY:
            raise contract_invalid(
                "absolute-tick checkpoint time authority is invalid",
                field="time_authority",
            )
        require_instance(self.checkpoint_id, CheckpointId, field="checkpoint_id")
        require_instance(self.run_id, RunId, field="run_id")
        require_instance(self.attempt_id, AttemptId, field="attempt_id")
        require_text(self.transaction_id, field="transaction_id")
        require_int(self.absolute_next_tick, field="absolute_next_tick")
        for field_name in (
            "semantic_spec_digest",
            "implementation_digest",
            "checkpoint_bundle_digest",
            "completion_marker_digest",
        ):
            require_instance(getattr(self, field_name), ContentDigest, field=field_name)
        if self.parent_checkpoint_digest is not None:
            require_instance(
                self.parent_checkpoint_digest,
                ContentDigest,
                field="parent_checkpoint_digest",
            )
        if not isinstance(self.participant_occurrences, tuple) or not (
            self.participant_occurrences
        ):
            raise contract_invalid(
                "absolute-tick checkpoint occurrence needs participants",
                field="participant_occurrences",
            )
        for participant in self.participant_occurrences:
            require_instance(
                participant,
                AbsoluteTickCheckpointParticipantOccurrenceV1,
                field="participant_occurrences",
            )
            if (
                participant.run_id != self.run_id
                or participant.attempt_id != self.attempt_id
                or participant.transaction_id != self.transaction_id
                or participant.absolute_next_tick != self.absolute_next_tick
            ):
                raise contract_invalid(
                    "absolute-tick participant differs from occurrence coordinates",
                    field="participant_occurrences",
                )
        identifiers = [
            str(participant.participant_id)
            for participant in self.participant_occurrences
        ]
        if identifiers != sorted(set(identifiers)):
            raise contract_invalid(
                "absolute-tick checkpoint participants must be sorted and unique",
                field="participant_occurrences",
            )
        _schema(
            self.schema_version,
            ABSOLUTE_TICK_CHECKPOINT_OCCURRENCE_SCHEMA_VERSION,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "checkpoint_format_id": self.checkpoint_format_id,
            "time_authority": self.time_authority,
            "checkpoint_id": str(self.checkpoint_id),
            "run_id": str(self.run_id),
            "attempt_id": str(self.attempt_id),
            "transaction_id": self.transaction_id,
            "absolute_next_tick": self.absolute_next_tick,
            "semantic_spec_digest": _digest(self.semantic_spec_digest),
            "implementation_digest": _digest(self.implementation_digest),
            "participant_occurrences": [
                participant.to_dict() for participant in self.participant_occurrences
            ],
            "checkpoint_bundle_digest": _digest(self.checkpoint_bundle_digest),
            "completion_marker_digest": _digest(self.completion_marker_digest),
            "parent_checkpoint_digest": _optional_digest(self.parent_checkpoint_digest),
        }

    def occurrence_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "AbsoluteTickCheckpointOccurrenceV1":
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "checkpoint_format_id",
            "time_authority",
            "checkpoint_id",
            "run_id",
            "attempt_id",
            "transaction_id",
            "absolute_next_tick",
            "semantic_spec_digest",
            "implementation_digest",
            "participant_occurrences",
            "checkpoint_bundle_digest",
            "completion_marker_digest",
            "parent_checkpoint_digest",
        }:
            raise contract_invalid(
                "absolute-tick checkpoint occurrence has an invalid shape",
                field="checkpoint_occurrence",
            )
        participants = value["participant_occurrences"]
        if not isinstance(participants, list):
            raise contract_invalid(
                "absolute-tick participant occurrences must be an array",
                field="participant_occurrences",
            )
        return cls(
            checkpoint_format_id=str(value["checkpoint_format_id"]),
            time_authority=str(value["time_authority"]),
            checkpoint_id=CheckpointId(str(value["checkpoint_id"])),
            run_id=RunId(str(value["run_id"])),
            attempt_id=AttemptId(str(value["attempt_id"])),
            transaction_id=str(value["transaction_id"]),
            absolute_next_tick=value["absolute_next_tick"],
            semantic_spec_digest=_content_digest(
                value["semantic_spec_digest"], field="semantic_spec_digest"
            ),
            implementation_digest=_content_digest(
                value["implementation_digest"], field="implementation_digest"
            ),
            participant_occurrences=tuple(
                AbsoluteTickCheckpointParticipantOccurrenceV1.from_dict(item)
                for item in participants
            ),
            checkpoint_bundle_digest=_content_digest(
                value["checkpoint_bundle_digest"], field="checkpoint_bundle_digest"
            ),
            completion_marker_digest=_content_digest(
                value["completion_marker_digest"],
                field="completion_marker_digest",
            ),
            parent_checkpoint_digest=_optional_content_digest(
                value["parent_checkpoint_digest"],
                field="parent_checkpoint_digest",
            ),
            schema_version=SchemaVersion(str(value["schema_version"])),
        )


@dataclass(frozen=True, slots=True)
class AbsoluteTickCheckpointOccurrenceCommitV1:
    occurrence_digest: ContentDigest
    checkpoint_bundle_digest: ContentDigest
    checkpoint_format_id: str
    committed_at_utc: str
    schema_version: SchemaVersion = (
        ABSOLUTE_TICK_CHECKPOINT_OCCURRENCE_COMMIT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        require_instance(
            self.occurrence_digest, ContentDigest, field="occurrence_digest"
        )
        require_instance(
            self.checkpoint_bundle_digest,
            ContentDigest,
            field="checkpoint_bundle_digest",
        )
        if self.checkpoint_format_id != ABSOLUTE_TICK_CHECKPOINT_FORMAT_ID:
            raise contract_invalid(
                "absolute-tick checkpoint commit format is invalid",
                field="checkpoint_format_id",
            )
        _canonical_utc(self.committed_at_utc, field="committed_at_utc")
        _schema(
            self.schema_version,
            ABSOLUTE_TICK_CHECKPOINT_OCCURRENCE_COMMIT_SCHEMA_VERSION,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "occurrence_digest": _digest(self.occurrence_digest),
            "checkpoint_bundle_digest": _digest(self.checkpoint_bundle_digest),
            "checkpoint_format_id": self.checkpoint_format_id,
            "committed_at_utc": self.committed_at_utc,
        }

    def commit_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "AbsoluteTickCheckpointOccurrenceCommitV1":
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "occurrence_digest",
            "checkpoint_bundle_digest",
            "checkpoint_format_id",
            "committed_at_utc",
        }:
            raise contract_invalid(
                "absolute-tick checkpoint occurrence commit has an invalid shape",
                field="checkpoint_occurrence_commit",
            )
        return cls(
            occurrence_digest=_content_digest(
                value["occurrence_digest"], field="occurrence_digest"
            ),
            checkpoint_bundle_digest=_content_digest(
                value["checkpoint_bundle_digest"],
                field="checkpoint_bundle_digest",
            ),
            checkpoint_format_id=str(value["checkpoint_format_id"]),
            committed_at_utc=str(value["committed_at_utc"]),
            schema_version=SchemaVersion(str(value["schema_version"])),
        )


@dataclass(frozen=True, slots=True)
class AbsoluteTickCheckpointOccurrenceInspectionV1:
    status: CheckpointOccurrenceInspectionStatusV1
    canonical_visible: bool
    checkpoint_bundle_digest: ContentDigest | None
    occurrence_digest: ContentDigest | None
    commit_digest: ContentDigest | None
    problem_digest: ContentDigest | None
    schema_version: SchemaVersion = (
        ABSOLUTE_TICK_CHECKPOINT_OCCURRENCE_INSPECTION_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(self.status, CheckpointOccurrenceInspectionStatusV1):
            raise contract_invalid(
                "absolute-tick checkpoint inspection status is invalid",
                field="status",
            )
        if type(self.canonical_visible) is not bool:
            raise contract_invalid(
                "canonical_visible must be bool",
                field="canonical_visible",
            )
        if self.canonical_visible != (
            self.status is CheckpointOccurrenceInspectionStatusV1.VISIBLE
        ):
            raise contract_invalid(
                "absolute-tick checkpoint visibility contradicts status",
                field="canonical_visible",
            )
        for field_name in (
            "checkpoint_bundle_digest",
            "occurrence_digest",
            "commit_digest",
            "problem_digest",
        ):
            value = getattr(self, field_name)
            if value is not None:
                require_instance(value, ContentDigest, field=field_name)
        if self.canonical_visible and any(
            value is None
            for value in (
                self.checkpoint_bundle_digest,
                self.occurrence_digest,
                self.commit_digest,
            )
        ):
            raise contract_invalid(
                "visible absolute-tick occurrence lacks a complete digest chain",
                field="canonical_visible",
            )
        if self.canonical_visible and self.problem_digest is not None:
            raise contract_invalid(
                "visible absolute-tick occurrence cannot have a problem digest",
                field="problem_digest",
            )
        _schema(
            self.schema_version,
            ABSOLUTE_TICK_CHECKPOINT_OCCURRENCE_INSPECTION_SCHEMA_VERSION,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "status": self.status.value,
            "canonical_visible": self.canonical_visible,
            "checkpoint_bundle_digest": _optional_digest(self.checkpoint_bundle_digest),
            "occurrence_digest": _optional_digest(self.occurrence_digest),
            "commit_digest": _optional_digest(self.commit_digest),
            "problem_digest": _optional_digest(self.problem_digest),
        }


def _workhorse_materials_digest_v2(
    materials: tuple["WorkhorseSemanticCheckpointMaterialV2", ...],
) -> ContentDigest:
    return canonical_sha256([material.to_dict() for material in materials])


@dataclass(frozen=True, slots=True)
class WorkhorseSemanticCheckpointMaterialV2:
    """One exact source file contributing to a Workhorse semantic checkpoint."""

    material_role: WorkhorseSemanticCheckpointMaterialRoleV2
    absolute_path: str
    content_digest: ContentDigest
    byte_count: int
    schema_version: SchemaVersion = WORKHORSE_SEMANTIC_CHECKPOINT_MATERIAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(
            self.material_role, WorkhorseSemanticCheckpointMaterialRoleV2
        ):
            raise contract_invalid(
                "Workhorse checkpoint material role is invalid",
                field="material_role",
            )
        require_text(self.absolute_path, field="absolute_path")
        locator = PurePosixPath(self.absolute_path)
        if (
            not locator.is_absolute()
            or str(locator) != self.absolute_path
            or any(part in {"", ".", ".."} for part in locator.parts)
        ):
            raise contract_invalid(
                "Workhorse checkpoint material path is not canonical absolute POSIX",
                field="absolute_path",
            )
        require_instance(self.content_digest, ContentDigest, field="content_digest")
        require_int(self.byte_count, field="byte_count")
        if self.byte_count < 0:
            raise contract_invalid(
                "Workhorse checkpoint material byte count is negative",
                field="byte_count",
            )
        _schema(self.schema_version, WORKHORSE_SEMANTIC_CHECKPOINT_MATERIAL_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "material_role": self.material_role.value,
            "absolute_path": self.absolute_path,
            "content_digest": _digest(self.content_digest),
            "byte_count": self.byte_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> "WorkhorseSemanticCheckpointMaterialV2":
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "material_role",
            "absolute_path",
            "content_digest",
            "byte_count",
        }:
            raise contract_invalid(
                "Workhorse checkpoint material has an invalid shape",
                field="checkpoint_material",
            )
        return cls(
            material_role=WorkhorseSemanticCheckpointMaterialRoleV2(
                str(value["material_role"])
            ),
            absolute_path=str(value["absolute_path"]),
            content_digest=_content_digest(
                value["content_digest"], field="content_digest"
            ),
            byte_count=value["byte_count"],
            schema_version=SchemaVersion(str(value["schema_version"])),
        )


@dataclass(frozen=True, slots=True)
class WorkhorseSemanticCheckpointOccurrenceV2:
    """CRCA occurrence for the Workhorse science-compatible base state."""

    checkpoint_format_id: str
    time_authority: str
    checkpoint_id: CheckpointId
    run_id: RunId
    attempt_id: AttemptId
    transaction_id: str
    absolute_next_tick: int
    semantic_spec_digest: ContentDigest
    implementation_digest: ContentDigest
    base_checkpoint_identity_digest: ContentDigest
    source_materials: tuple[WorkhorseSemanticCheckpointMaterialV2, ...]
    source_materials_digest: ContentDigest
    parent_checkpoint_digest: ContentDigest | None
    runtime_observation_mode: str = "OFF"
    qualification_authority_granted: bool = False
    science_authority_granted: bool = False
    schema_version: SchemaVersion = WORKHORSE_SEMANTIC_CHECKPOINT_OCCURRENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.checkpoint_format_id != WORKHORSE_SEMANTIC_CHECKPOINT_FORMAT_ID:
            raise contract_invalid(
                "Workhorse semantic checkpoint format is invalid",
                field="checkpoint_format_id",
            )
        if self.time_authority != WORKHORSE_SEMANTIC_CHECKPOINT_TIME_AUTHORITY:
            raise contract_invalid(
                "Workhorse semantic checkpoint time authority is invalid",
                field="time_authority",
            )
        require_instance(self.checkpoint_id, CheckpointId, field="checkpoint_id")
        require_instance(self.run_id, RunId, field="run_id")
        require_instance(self.attempt_id, AttemptId, field="attempt_id")
        require_text(self.transaction_id, field="transaction_id")
        require_int(self.absolute_next_tick, field="absolute_next_tick")
        if self.absolute_next_tick < 0:
            raise contract_invalid(
                "Workhorse semantic checkpoint tick is negative",
                field="absolute_next_tick",
            )
        for field_name in (
            "semantic_spec_digest",
            "implementation_digest",
            "base_checkpoint_identity_digest",
            "source_materials_digest",
        ):
            require_instance(getattr(self, field_name), ContentDigest, field=field_name)
        if self.parent_checkpoint_digest is not None:
            require_instance(
                self.parent_checkpoint_digest,
                ContentDigest,
                field="parent_checkpoint_digest",
            )
        if not isinstance(self.source_materials, tuple):
            raise contract_invalid(
                "Workhorse checkpoint materials must be a tuple",
                field="source_materials",
            )
        for material in self.source_materials:
            require_instance(
                material,
                WorkhorseSemanticCheckpointMaterialV2,
                field="source_materials",
            )
        roles = [material.material_role.value for material in self.source_materials]
        expected_roles = sorted(role.value for role in WorkhorseSemanticCheckpointMaterialRoleV2)
        paths = [material.absolute_path for material in self.source_materials]
        if roles != expected_roles or len(paths) != len(set(paths)):
            raise contract_invalid(
                "Workhorse checkpoint materials are not exact, sorted, and unique",
                field="source_materials",
            )
        if self.source_materials_digest != _workhorse_materials_digest_v2(
            self.source_materials
        ):
            raise contract_invalid(
                "Workhorse checkpoint material-set digest differs",
                field="source_materials_digest",
            )
        if (
            self.runtime_observation_mode != "OFF"
            or type(self.qualification_authority_granted) is not bool
            or self.qualification_authority_granted
            or type(self.science_authority_granted) is not bool
            or self.science_authority_granted
        ):
            raise contract_invalid(
                "Workhorse checkpoint cannot grant observation, science, or qualification authority",
                field="runtime_observation_mode",
            )
        _schema(
            self.schema_version,
            WORKHORSE_SEMANTIC_CHECKPOINT_OCCURRENCE_SCHEMA_VERSION,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "checkpoint_format_id": self.checkpoint_format_id,
            "time_authority": self.time_authority,
            "checkpoint_id": str(self.checkpoint_id),
            "run_id": str(self.run_id),
            "attempt_id": str(self.attempt_id),
            "transaction_id": self.transaction_id,
            "absolute_next_tick": self.absolute_next_tick,
            "semantic_spec_digest": _digest(self.semantic_spec_digest),
            "implementation_digest": _digest(self.implementation_digest),
            "base_checkpoint_identity_digest": _digest(
                self.base_checkpoint_identity_digest
            ),
            "source_materials": [
                material.to_dict() for material in self.source_materials
            ],
            "source_materials_digest": _digest(self.source_materials_digest),
            "parent_checkpoint_digest": _optional_digest(self.parent_checkpoint_digest),
            "runtime_observation_mode": self.runtime_observation_mode,
            "qualification_authority_granted": self.qualification_authority_granted,
            "science_authority_granted": self.science_authority_granted,
        }

    def occurrence_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "WorkhorseSemanticCheckpointOccurrenceV2":
        expected = {
            "schema_version",
            "checkpoint_format_id",
            "time_authority",
            "checkpoint_id",
            "run_id",
            "attempt_id",
            "transaction_id",
            "absolute_next_tick",
            "semantic_spec_digest",
            "implementation_digest",
            "base_checkpoint_identity_digest",
            "source_materials",
            "source_materials_digest",
            "parent_checkpoint_digest",
            "runtime_observation_mode",
            "qualification_authority_granted",
            "science_authority_granted",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise contract_invalid(
                "Workhorse semantic checkpoint occurrence has an invalid shape",
                field="checkpoint_occurrence",
            )
        materials = value["source_materials"]
        if not isinstance(materials, list):
            raise contract_invalid(
                "Workhorse checkpoint materials must be an array",
                field="source_materials",
            )
        return cls(
            checkpoint_format_id=str(value["checkpoint_format_id"]),
            time_authority=str(value["time_authority"]),
            checkpoint_id=CheckpointId(str(value["checkpoint_id"])),
            run_id=RunId(str(value["run_id"])),
            attempt_id=AttemptId(str(value["attempt_id"])),
            transaction_id=str(value["transaction_id"]),
            absolute_next_tick=value["absolute_next_tick"],
            semantic_spec_digest=_content_digest(
                value["semantic_spec_digest"], field="semantic_spec_digest"
            ),
            implementation_digest=_content_digest(
                value["implementation_digest"], field="implementation_digest"
            ),
            base_checkpoint_identity_digest=_content_digest(
                value["base_checkpoint_identity_digest"],
                field="base_checkpoint_identity_digest",
            ),
            source_materials=tuple(
                WorkhorseSemanticCheckpointMaterialV2.from_dict(item)
                for item in materials
            ),
            source_materials_digest=_content_digest(
                value["source_materials_digest"], field="source_materials_digest"
            ),
            parent_checkpoint_digest=_optional_content_digest(
                value["parent_checkpoint_digest"], field="parent_checkpoint_digest"
            ),
            runtime_observation_mode=str(value["runtime_observation_mode"]),
            qualification_authority_granted=value["qualification_authority_granted"],
            science_authority_granted=value["science_authority_granted"],
            schema_version=SchemaVersion(str(value["schema_version"])),
        )


@dataclass(frozen=True, slots=True)
class WorkhorseSemanticCheckpointOccurrenceCommitV2:
    occurrence_digest: ContentDigest
    source_materials_digest: ContentDigest
    checkpoint_format_id: str
    committed_at_utc: str
    schema_version: SchemaVersion = (
        WORKHORSE_SEMANTIC_CHECKPOINT_OCCURRENCE_COMMIT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        require_instance(self.occurrence_digest, ContentDigest, field="occurrence_digest")
        require_instance(
            self.source_materials_digest,
            ContentDigest,
            field="source_materials_digest",
        )
        if self.checkpoint_format_id != WORKHORSE_SEMANTIC_CHECKPOINT_FORMAT_ID:
            raise contract_invalid(
                "Workhorse checkpoint commit format is invalid",
                field="checkpoint_format_id",
            )
        _canonical_utc(self.committed_at_utc, field="committed_at_utc")
        _schema(
            self.schema_version,
            WORKHORSE_SEMANTIC_CHECKPOINT_OCCURRENCE_COMMIT_SCHEMA_VERSION,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "occurrence_digest": _digest(self.occurrence_digest),
            "source_materials_digest": _digest(self.source_materials_digest),
            "checkpoint_format_id": self.checkpoint_format_id,
            "committed_at_utc": self.committed_at_utc,
        }

    def commit_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(
        cls, value: object
    ) -> "WorkhorseSemanticCheckpointOccurrenceCommitV2":
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "occurrence_digest",
            "source_materials_digest",
            "checkpoint_format_id",
            "committed_at_utc",
        }:
            raise contract_invalid(
                "Workhorse checkpoint occurrence commit has an invalid shape",
                field="checkpoint_occurrence_commit",
            )
        return cls(
            occurrence_digest=_content_digest(
                value["occurrence_digest"], field="occurrence_digest"
            ),
            source_materials_digest=_content_digest(
                value["source_materials_digest"], field="source_materials_digest"
            ),
            checkpoint_format_id=str(value["checkpoint_format_id"]),
            committed_at_utc=str(value["committed_at_utc"]),
            schema_version=SchemaVersion(str(value["schema_version"])),
        )


@dataclass(frozen=True, slots=True)
class WorkhorseSemanticCheckpointOccurrenceInspectionV2:
    status: CheckpointOccurrenceInspectionStatusV1
    canonical_visible: bool
    source_materials_digest: ContentDigest | None
    occurrence_digest: ContentDigest | None
    commit_digest: ContentDigest | None
    problem_digest: ContentDigest | None
    schema_version: SchemaVersion = (
        WORKHORSE_SEMANTIC_CHECKPOINT_OCCURRENCE_INSPECTION_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(self.status, CheckpointOccurrenceInspectionStatusV1):
            raise contract_invalid(
                "Workhorse checkpoint inspection status is invalid",
                field="status",
            )
        if type(self.canonical_visible) is not bool or self.canonical_visible != (
            self.status is CheckpointOccurrenceInspectionStatusV1.VISIBLE
        ):
            raise contract_invalid(
                "Workhorse checkpoint visibility contradicts status",
                field="canonical_visible",
            )
        for field_name in (
            "source_materials_digest",
            "occurrence_digest",
            "commit_digest",
            "problem_digest",
        ):
            item = getattr(self, field_name)
            if item is not None:
                require_instance(item, ContentDigest, field=field_name)
        if self.canonical_visible and (
            self.source_materials_digest is None
            or self.occurrence_digest is None
            or self.commit_digest is None
            or self.problem_digest is not None
        ):
            raise contract_invalid(
                "visible Workhorse checkpoint lacks an exact digest chain",
                field="canonical_visible",
            )
        _schema(
            self.schema_version,
            WORKHORSE_SEMANTIC_CHECKPOINT_OCCURRENCE_INSPECTION_SCHEMA_VERSION,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "status": self.status.value,
            "canonical_visible": self.canonical_visible,
            "source_materials_digest": _optional_digest(
                self.source_materials_digest
            ),
            "occurrence_digest": _optional_digest(self.occurrence_digest),
            "commit_digest": _optional_digest(self.commit_digest),
            "problem_digest": _optional_digest(self.problem_digest),
        }


@dataclass(frozen=True, slots=True)
class CheckpointConformanceSentinelV1:
    sentinel_id: str
    layer: CheckpointConformanceLayerV1
    owner_id: str
    owner_source_path: str
    owner_digest: ContentDigest
    test_node_id: str
    test_source_digest: ContentDigest
    qualification_effect: bool = False
    schema_version: SchemaVersion = CHECKPOINT_CONFORMANCE_SENTINEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.sentinel_id, field="sentinel_id")
        if not isinstance(self.layer, CheckpointConformanceLayerV1):
            raise contract_invalid("conformance layer is invalid", field="layer")
        require_text(self.owner_id, field="owner_id")
        require_text(self.owner_source_path, field="owner_source_path")
        require_text(self.test_node_id, field="test_node_id")
        require_instance(self.owner_digest, ContentDigest, field="owner_digest")
        require_instance(
            self.test_source_digest, ContentDigest, field="test_source_digest"
        )
        test_path = self.test_node_id.split("::", 1)[0]
        for field, path in (
            ("owner_source_path", self.owner_source_path),
            ("test_node_id", test_path),
        ):
            candidate = PurePosixPath(path)
            if (
                candidate.is_absolute()
                or ".." in candidate.parts
                or "\\" in path
            ):
                raise contract_invalid("conformance source path is not bounded", field=field)
        if not test_path.startswith("tests/") or "::" not in self.test_node_id:
            raise contract_invalid(
                "conformance sentinel must bind one exact test node",
                field="test_node_id",
            )
        if type(self.qualification_effect) is not bool or self.qualification_effect:
            raise contract_invalid(
                "conformance sentinel cannot grant qualification authority",
                field="qualification_effect",
            )
        _schema(self.schema_version, CHECKPOINT_CONFORMANCE_SENTINEL_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "sentinel_id": self.sentinel_id,
            "layer": self.layer.value,
            "owner_id": self.owner_id,
            "owner_source_path": self.owner_source_path,
            "owner_digest": _digest(self.owner_digest),
            "test_node_id": self.test_node_id,
            "test_source_digest": _digest(self.test_source_digest),
            "qualification_effect": self.qualification_effect,
        }

    def sentinel_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class CheckpointConformanceIndexV1:
    sentinels: tuple[CheckpointConformanceSentinelV1, ...]
    schema_version: SchemaVersion = CHECKPOINT_CONFORMANCE_INDEX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.sentinels, tuple):
            raise contract_invalid(
                "conformance sentinels must be a tuple",
                field="sentinels",
            )
        layers: set[CheckpointConformanceLayerV1] = set()
        nodes: set[str] = set()
        for sentinel in self.sentinels:
            require_instance(
                sentinel,
                CheckpointConformanceSentinelV1,
                field="sentinels",
            )
            layers.add(sentinel.layer)
            if sentinel.test_node_id in nodes:
                raise contract_invalid(
                    "conformance index repeats a test node",
                    field="sentinels",
                )
            nodes.add(sentinel.test_node_id)
        identifiers = [sentinel.sentinel_id for sentinel in self.sentinels]
        if identifiers != sorted(set(identifiers)):
            raise contract_invalid(
                "conformance sentinels must be sorted and unique",
                field="sentinels",
            )
        if layers != set(CheckpointConformanceLayerV1):
            raise contract_invalid(
                "conformance index must cover contract, runtime, backend, and qualification",
                field="sentinels",
            )
        _schema(self.schema_version, CHECKPOINT_CONFORMANCE_INDEX_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "sentinels": [sentinel.to_dict() for sentinel in self.sentinels],
        }

    def index_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class CheckpointConformanceResultV1:
    sentinel_id: str
    sentinel_digest: ContentDigest
    status: CheckpointConformanceStatusV1
    evidence_digest: ContentDigest
    schema_version: SchemaVersion = CHECKPOINT_CONFORMANCE_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.sentinel_id, field="sentinel_id")
        require_instance(self.sentinel_digest, ContentDigest, field="sentinel_digest")
        require_instance(self.evidence_digest, ContentDigest, field="evidence_digest")
        if not isinstance(self.status, CheckpointConformanceStatusV1):
            raise contract_invalid(
                "conformance result status is invalid",
                field="status",
            )
        _schema(self.schema_version, CHECKPOINT_CONFORMANCE_RESULT_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "sentinel_id": self.sentinel_id,
            "sentinel_digest": _digest(self.sentinel_digest),
            "status": self.status.value,
            "evidence_digest": _digest(self.evidence_digest),
        }


@dataclass(frozen=True, slots=True)
class CheckpointConformanceRunV1:
    index_digest: ContentDigest
    results: tuple[CheckpointConformanceResultV1, ...]
    observed_at_utc: str
    qualification_authority: bool = False
    schema_version: SchemaVersion = CHECKPOINT_CONFORMANCE_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(self.index_digest, ContentDigest, field="index_digest")
        _canonical_utc(self.observed_at_utc, field="observed_at_utc")
        if not isinstance(self.results, tuple) or not self.results:
            raise contract_invalid(
                "conformance run needs results",
                field="results",
            )
        for result in self.results:
            require_instance(
                result, CheckpointConformanceResultV1, field="results"
            )
        identifiers = [result.sentinel_id for result in self.results]
        if identifiers != sorted(set(identifiers)):
            raise contract_invalid(
                "conformance results must be sorted and unique",
                field="results",
            )
        if type(self.qualification_authority) is not bool or (
            self.qualification_authority
        ):
            raise contract_invalid(
                "conformance run cannot grant qualification",
                field="qualification_authority",
            )
        _schema(self.schema_version, CHECKPOINT_CONFORMANCE_RUN_SCHEMA_VERSION)

    @property
    def status(self) -> CheckpointConformanceStatusV1:
        return (
            CheckpointConformanceStatusV1.PASS
            if all(
                result.status is CheckpointConformanceStatusV1.PASS
                for result in self.results
            )
            else CheckpointConformanceStatusV1.FAIL
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "index_digest": _digest(self.index_digest),
            "results": [result.to_dict() for result in self.results],
            "observed_at_utc": self.observed_at_utc,
            "qualification_authority": self.qualification_authority,
            "status": self.status.value,
        }

    def run_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())
