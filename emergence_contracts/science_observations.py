"""Deterministic, route-neutral construction of canonical science records."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ._validation import (
    require_instance,
    require_int,
    require_text,
    require_tuple_instances,
)
from .canonical import canonical_json_bytes, canonical_sha256
from .errors import contract_invalid
from .ids import ContentDigest, SchemaVersion
from .payloads import CanonicalPayloadV1
from .science_instrument import ScienceObservationRecordV1, ScienceRecordKindV1

SCIENCE_SEMANTIC_EXECUTION_ID_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_semantic_execution_identity_v1"
)
SCIENCE_WORLD_UID_SCHEMA_VERSION = SchemaVersion("eiel_science_world_uid_v1")
SCIENCE_PRE_SEQUENCE_KEY_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_pre_sequence_key_v1"
)
SCIENCE_EVENT_ID_SCHEMA_VERSION = SchemaVersion("eiel_science_event_identity_v1")


_RECORD_KIND_ORDINALS = {
    kind: ordinal for ordinal, kind in enumerate(ScienceRecordKindV1)
}


@dataclass(frozen=True, slots=True)
class ScienceObservationDraftV1:
    producer_component_id: str
    producer_contract_version: SchemaVersion
    producer_phase_registry_ordinal: int
    absolute_tick: int
    semantic_source_address: str
    record_kind: ScienceRecordKindV1
    record_kind_version: int
    control_binding_sha256: ContentDigest
    payload: CanonicalPayloadV1
    source_state_before_sha256: ContentDigest | None = None
    source_state_after_sha256: ContentDigest | None = None
    cause_event_ids: tuple[ContentDigest, ...] = ()
    lifecycle_epoch: int | None = None
    legacy_compatibility: CanonicalPayloadV1 | None = None

    def __post_init__(self) -> None:
        require_text(self.producer_component_id, field="producer_component_id")
        require_instance(
            self.producer_contract_version,
            SchemaVersion,
            field="producer_contract_version",
        )
        require_int(
            self.producer_phase_registry_ordinal,
            field="producer_phase_registry_ordinal",
            minimum=0,
        )
        require_int(self.absolute_tick, field="absolute_tick", minimum=0)
        require_text(self.semantic_source_address, field="semantic_source_address")
        require_instance(self.record_kind, ScienceRecordKindV1, field="record_kind")
        require_int(self.record_kind_version, field="record_kind_version", minimum=1)
        if self.record_kind_version != 1:
            raise contract_invalid(
                "unsupported science record-kind version",
                field="record_kind_version",
            )
        require_instance(
            self.control_binding_sha256,
            ContentDigest,
            field="control_binding_sha256",
        )
        require_instance(self.payload, CanonicalPayloadV1, field="payload")
        for field_name in ("source_state_before_sha256", "source_state_after_sha256"):
            value = getattr(self, field_name)
            if value is not None:
                require_instance(value, ContentDigest, field=field_name)
        require_tuple_instances(
            self.cause_event_ids,
            ContentDigest,
            field="cause_event_ids",
            unique=True,
            sorted_by_string=True,
        )
        if self.lifecycle_epoch is not None:
            require_int(self.lifecycle_epoch, field="lifecycle_epoch", minimum=0)
        if self.legacy_compatibility is not None:
            require_instance(
                self.legacy_compatibility,
                CanonicalPayloadV1,
                field="legacy_compatibility",
            )


def derive_science_semantic_execution_id_v1(
    crca_semantic_run_digest: ContentDigest,
    science_semantic_acquisition_sha256: ContentDigest,
) -> ContentDigest:
    return canonical_sha256(
        {
            "schema_version": str(SCIENCE_SEMANTIC_EXECUTION_ID_SCHEMA_VERSION),
            "crca_semantic_run_digest": crca_semantic_run_digest.to_dict(),
            "science_semantic_acquisition_sha256": (
                science_semantic_acquisition_sha256.to_dict()
            ),
        }
    )


def derive_science_world_uid_v1(
    semantic_execution_id: ContentDigest,
    semantic_world_key: CanonicalPayloadV1,
) -> ContentDigest:
    return canonical_sha256(
        {
            "schema_version": str(SCIENCE_WORLD_UID_SCHEMA_VERSION),
            "semantic_execution_id": semantic_execution_id.to_dict(),
            "semantic_world_key": semantic_world_key.to_value(),
        }
    )


def _pre_sequence_key_v1(draft: ScienceObservationDraftV1) -> ContentDigest:
    return canonical_sha256(
        {
            "schema_version": str(SCIENCE_PRE_SEQUENCE_KEY_SCHEMA_VERSION),
            "producer_component_id": draft.producer_component_id,
            "producer_contract_version": str(draft.producer_contract_version),
            "producer_phase_registry_ordinal": draft.producer_phase_registry_ordinal,
            "record_kind_registry_ordinal": _RECORD_KIND_ORDINALS[draft.record_kind],
            "semantic_source_address": draft.semantic_source_address,
            "cause_event_ids": [item.to_dict() for item in draft.cause_event_ids],
            "semantic_payload_sha256": canonical_sha256(
                draft.payload.to_value()
            ).to_dict(),
        }
    )


def _event_id_v1(
    *,
    semantic_execution_id: ContentDigest,
    world_uid: ContentDigest,
    absolute_tick: int,
    world_local_sequence: int,
    record_kind: ScienceRecordKindV1,
    record_kind_version: int,
    producer_component_id: str,
    producer_contract_version: SchemaVersion,
) -> ContentDigest:
    return canonical_sha256(
        {
            "schema_version": str(SCIENCE_EVENT_ID_SCHEMA_VERSION),
            "semantic_execution_id": semantic_execution_id.to_dict(),
            "world_uid": world_uid.to_dict(),
            "absolute_tick": absolute_tick,
            "world_local_sequence": world_local_sequence,
            "record_kind": record_kind.value,
            "record_kind_version": record_kind_version,
            "producer_component_id": producer_component_id,
            "producer_contract_version": str(producer_contract_version),
        }
    )


def finalize_science_observation_drafts_v1(
    drafts: tuple[ScienceObservationDraftV1, ...],
    *,
    instrument_manifest_sha256: ContentDigest,
    crca_semantic_run_digest: ContentDigest,
    science_semantic_acquisition_sha256: ContentDigest,
    cell_id: str,
    logical_world_id: int,
    semantic_world_key: CanonicalPayloadV1,
    start_world_local_sequence: int = 0,
) -> tuple[ScienceObservationRecordV1, ...]:
    """Assign canonical order, sequence, and event IDs to physical-free drafts."""

    if type(start_world_local_sequence) is not int or start_world_local_sequence < 0:
        raise contract_invalid(
            "start sequence must be a nonnegative integer",
            field="start_world_local_sequence",
        )
    semantic_execution_id = derive_science_semantic_execution_id_v1(
        crca_semantic_run_digest, science_semantic_acquisition_sha256
    )
    world_uid = derive_science_world_uid_v1(semantic_execution_id, semantic_world_key)
    prepared = [(draft, _pre_sequence_key_v1(draft)) for draft in drafts]
    addresses = [
        (draft.absolute_tick, draft.semantic_source_address) for draft in drafts
    ]
    if len(set(addresses)) != len(addresses):
        raise contract_invalid(
            "semantic source address is duplicated within a world and tick",
            field="drafts",
        )
    keys = [
        (
            draft.absolute_tick,
            draft.producer_phase_registry_ordinal,
            _RECORD_KIND_ORDINALS[draft.record_kind],
            pre_key.hex,
        )
        for draft, pre_key in prepared
    ]
    if len(set(keys)) != len(keys):
        raise contract_invalid("pre-sequence key collision", field="drafts")
    prepared.sort(
        key=lambda item: (
            item[0].absolute_tick,
            item[0].producer_phase_registry_ordinal,
            _RECORD_KIND_ORDINALS[item[0].record_kind],
            item[1].hex,
        )
    )
    result: list[ScienceObservationRecordV1] = []
    for offset, (draft, pre_key) in enumerate(prepared):
        sequence = start_world_local_sequence + offset
        result.append(
            ScienceObservationRecordV1(
                instrument_manifest_sha256=instrument_manifest_sha256,
                producer_component_id=draft.producer_component_id,
                producer_contract_version=draft.producer_contract_version,
                semantic_execution_id=semantic_execution_id,
                crca_semantic_run_digest=crca_semantic_run_digest,
                science_semantic_acquisition_sha256=(
                    science_semantic_acquisition_sha256
                ),
                cell_id=cell_id,
                logical_world_id=logical_world_id,
                world_uid=world_uid,
                absolute_tick=draft.absolute_tick,
                semantic_source_address=draft.semantic_source_address,
                pre_sequence_key_sha256=pre_key,
                world_local_sequence=sequence,
                deterministic_event_id=_event_id_v1(
                    semantic_execution_id=semantic_execution_id,
                    world_uid=world_uid,
                    absolute_tick=draft.absolute_tick,
                    world_local_sequence=sequence,
                    record_kind=draft.record_kind,
                    record_kind_version=draft.record_kind_version,
                    producer_component_id=draft.producer_component_id,
                    producer_contract_version=draft.producer_contract_version,
                ),
                record_kind=draft.record_kind,
                record_kind_version=draft.record_kind_version,
                control_binding_sha256=draft.control_binding_sha256,
                payload=draft.payload,
                source_state_before_sha256=draft.source_state_before_sha256,
                source_state_after_sha256=draft.source_state_after_sha256,
                cause_event_ids=draft.cause_event_ids,
                lifecycle_epoch=draft.lifecycle_epoch,
                legacy_compatibility=draft.legacy_compatibility,
            )
        )
    return tuple(result)


def validate_science_observation_stream_v1(
    rows: tuple[ScienceObservationRecordV1, ...],
    *,
    producer_phase_registry_ordinals: dict[str, int],
    expected_start_world_local_sequence: int = 0,
) -> None:
    """Recompute all ordering identities and reject gaps, tampering, or mixing."""

    if not rows:
        return
    first = rows[0]
    expected_sequences = tuple(
        range(
            expected_start_world_local_sequence,
            expected_start_world_local_sequence + len(rows),
        )
    )
    actual_sequences = tuple(row.world_local_sequence for row in rows)
    if actual_sequences != expected_sequences:
        raise contract_invalid("world-local sequence is not contiguous", field="rows")
    previous_order: tuple[int, int] | None = None
    for row in rows:
        if (
            row.semantic_execution_id != first.semantic_execution_id
            or row.world_uid != first.world_uid
            or row.logical_world_id != first.logical_world_id
        ):
            raise contract_invalid("observation stream mixes worlds", field="rows")
        phase = producer_phase_registry_ordinals.get(row.producer_component_id)
        if phase is None:
            raise contract_invalid(
                "producer phase registry entry is missing",
                field="producer_phase_registry_ordinals",
            )
        draft = ScienceObservationDraftV1(
            producer_component_id=row.producer_component_id,
            producer_contract_version=row.producer_contract_version,
            producer_phase_registry_ordinal=phase,
            absolute_tick=row.absolute_tick,
            semantic_source_address=row.semantic_source_address,
            record_kind=row.record_kind,
            record_kind_version=row.record_kind_version,
            control_binding_sha256=row.control_binding_sha256,
            payload=row.payload,
            source_state_before_sha256=row.source_state_before_sha256,
            source_state_after_sha256=row.source_state_after_sha256,
            cause_event_ids=row.cause_event_ids,
            lifecycle_epoch=row.lifecycle_epoch,
            legacy_compatibility=row.legacy_compatibility,
        )
        if _pre_sequence_key_v1(draft) != row.pre_sequence_key_sha256:
            raise contract_invalid("pre-sequence key mismatch", field="rows")
        expected_event = _event_id_v1(
            semantic_execution_id=row.semantic_execution_id,
            world_uid=row.world_uid,
            absolute_tick=row.absolute_tick,
            world_local_sequence=row.world_local_sequence,
            record_kind=row.record_kind,
            record_kind_version=row.record_kind_version,
            producer_component_id=row.producer_component_id,
            producer_contract_version=row.producer_contract_version,
        )
        if expected_event != row.deterministic_event_id:
            raise contract_invalid("deterministic event ID mismatch", field="rows")
        current_order = (row.absolute_tick, row.world_local_sequence)
        if previous_order is not None and current_order <= previous_order:
            raise contract_invalid("observation stream order is invalid", field="rows")
        previous_order = current_order


def canonical_science_record_stream_sha256_v1(
    rows: tuple[ScienceObservationRecordV1, ...],
) -> ContentDigest:
    """Hash an ordered record stream independently of physical chunk boundaries."""

    buffer = bytearray()
    for row in rows:
        encoded = canonical_json_bytes(row.semantic_dict())
        buffer.extend(len(encoded).to_bytes(8, "big"))
        buffer.extend(encoded)
    return ContentDigest("sha256", hashlib.sha256(buffer).hexdigest())


__all__ = (
    "SCIENCE_EVENT_ID_SCHEMA_VERSION",
    "SCIENCE_PRE_SEQUENCE_KEY_SCHEMA_VERSION",
    "SCIENCE_SEMANTIC_EXECUTION_ID_SCHEMA_VERSION",
    "SCIENCE_WORLD_UID_SCHEMA_VERSION",
    "ScienceObservationDraftV1",
    "canonical_science_record_stream_sha256_v1",
    "derive_science_semantic_execution_id_v1",
    "derive_science_world_uid_v1",
    "finalize_science_observation_drafts_v1",
    "validate_science_observation_stream_v1",
)
