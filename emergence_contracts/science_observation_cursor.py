"""Route-neutral observation-prefix continuity across absolute-tick checkpoints."""

from __future__ import annotations

from dataclasses import dataclass

from ._validation import require_instance, require_int, require_text
from .canonical import canonical_sha256
from .errors import contract_invalid
from .ids import ContentDigest, SchemaVersion
from .payloads import CanonicalPayloadV1
from .science_instrument import ScienceObservationRecordV1
from .science_observations import (
    ScienceObservationDraftV1,
    derive_science_semantic_execution_id_v1,
    derive_science_world_uid_v1,
    finalize_science_observation_drafts_v1,
    validate_science_observation_stream_v1,
)

SCIENCE_OBSERVATION_PREFIX_CHAIN_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_observation_prefix_chain_v1"
)
SCIENCE_OBSERVATION_PREFIX_CURSOR_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_observation_prefix_cursor_v1"
)


def _digest_from_dict(value: object, *, field: str) -> ContentDigest:
    if not isinstance(value, dict) or set(value) != {"algorithm", "hex"}:
        raise contract_invalid("digest object is malformed", field=field)
    try:
        return ContentDigest(value["algorithm"], value["hex"])
    except (TypeError, ValueError) as exc:
        raise contract_invalid("digest object is invalid", field=field) from exc


def empty_science_observation_prefix_sha256_v1() -> ContentDigest:
    """Return the domain-separated identity of an empty observation prefix."""

    return canonical_sha256(
        {
            "schema_version": str(SCIENCE_OBSERVATION_PREFIX_CHAIN_SCHEMA_VERSION),
            "prefix": "EMPTY",
        }
    )


def extend_science_observation_prefix_sha256_v1(
    prefix_sha256: ContentDigest,
    rows: tuple[ScienceObservationRecordV1, ...],
) -> ContentDigest:
    """Extend a serializable prefix chain with canonical semantic records."""

    require_instance(prefix_sha256, ContentDigest, field="prefix_sha256")
    if not isinstance(rows, tuple) or any(
        not isinstance(row, ScienceObservationRecordV1) for row in rows
    ):
        raise contract_invalid(
            "rows must be a tuple of science observation records", field="rows"
        )
    if rows:
        first = rows[0]
        if any(
            row.world_uid != first.world_uid
            or row.semantic_execution_id != first.semantic_execution_id
            for row in rows
        ):
            raise contract_invalid("prefix extension mixes worlds", field="rows")
        sequences = tuple(row.world_local_sequence for row in rows)
        if sequences != tuple(range(sequences[0], sequences[0] + len(rows))):
            raise contract_invalid(
                "prefix extension sequence is not contiguous", field="rows"
            )
    current = prefix_sha256
    for row in rows:
        current = canonical_sha256(
            {
                "schema_version": str(SCIENCE_OBSERVATION_PREFIX_CHAIN_SCHEMA_VERSION),
                "previous_prefix_sha256": current.to_dict(),
                "world_local_sequence": row.world_local_sequence,
                "semantic_record_sha256": row.semantic_record_sha256().to_dict(),
            }
        )
    return current


@dataclass(frozen=True, slots=True)
class ScienceObservationPrefixCursorV1:
    instrument_manifest_sha256: ContentDigest
    semantic_execution_id: ContentDigest
    crca_semantic_run_digest: ContentDigest
    science_semantic_acquisition_sha256: ContentDigest
    cell_id: str
    logical_world_id: int
    world_uid: ContentDigest
    checkpoint_at_tick: int
    next_absolute_tick: int
    next_world_local_sequence: int
    prefix_record_count: int
    prefix_chain_sha256: ContentDigest
    last_deterministic_event_id: ContentDigest | None
    previous_cursor_sha256: ContentDigest | None
    schema_version: SchemaVersion = SCIENCE_OBSERVATION_PREFIX_CURSOR_SCHEMA_VERSION
    effects_performed: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "instrument_manifest_sha256",
            "semantic_execution_id",
            "crca_semantic_run_digest",
            "science_semantic_acquisition_sha256",
            "world_uid",
            "prefix_chain_sha256",
        ):
            require_instance(getattr(self, field_name), ContentDigest, field=field_name)
        for field_name in (
            "last_deterministic_event_id",
            "previous_cursor_sha256",
        ):
            value = getattr(self, field_name)
            if value is not None:
                require_instance(value, ContentDigest, field=field_name)
        require_text(self.cell_id, field="cell_id")
        for field_name in (
            "logical_world_id",
            "checkpoint_at_tick",
            "next_absolute_tick",
            "next_world_local_sequence",
            "prefix_record_count",
        ):
            require_int(getattr(self, field_name), field=field_name, minimum=0)
        if self.next_absolute_tick != self.checkpoint_at_tick:
            raise contract_invalid(
                "observation cursor must resume at its absolute checkpoint boundary",
                field="next_absolute_tick",
            )
        if self.next_world_local_sequence != self.prefix_record_count:
            raise contract_invalid(
                "world sequence must equal the sealed prefix record count",
                field="next_world_local_sequence",
            )
        if self.semantic_execution_id != derive_science_semantic_execution_id_v1(
            self.crca_semantic_run_digest,
            self.science_semantic_acquisition_sha256,
        ):
            raise contract_invalid(
                "semantic execution identity differs from cursor inputs",
                field="semantic_execution_id",
            )
        if (self.last_deterministic_event_id is None) != (
            self.prefix_record_count == 0
        ):
            raise contract_invalid(
                "last event identity must match prefix emptiness",
                field="last_deterministic_event_id",
            )
        if self.prefix_record_count == 0 and (
            self.prefix_chain_sha256 != empty_science_observation_prefix_sha256_v1()
        ):
            raise contract_invalid(
                "empty observation prefix identity differs",
                field="prefix_chain_sha256",
            )
        if self.schema_version != SCIENCE_OBSERVATION_PREFIX_CURSOR_SCHEMA_VERSION:
            raise contract_invalid(
                "unsupported observation cursor schema", field="schema_version"
            )
        if self.effects_performed is not False:
            raise contract_invalid(
                "observation cursor sealing must be effect-free",
                field="effects_performed",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "instrument_manifest_sha256": self.instrument_manifest_sha256.to_dict(),
            "semantic_execution_id": self.semantic_execution_id.to_dict(),
            "crca_semantic_run_digest": self.crca_semantic_run_digest.to_dict(),
            "science_semantic_acquisition_sha256": (
                self.science_semantic_acquisition_sha256.to_dict()
            ),
            "cell_id": self.cell_id,
            "logical_world_id": self.logical_world_id,
            "world_uid": self.world_uid.to_dict(),
            "checkpoint_at_tick": self.checkpoint_at_tick,
            "next_absolute_tick": self.next_absolute_tick,
            "next_world_local_sequence": self.next_world_local_sequence,
            "prefix_record_count": self.prefix_record_count,
            "prefix_chain_sha256": self.prefix_chain_sha256.to_dict(),
            "last_deterministic_event_id": (
                None
                if self.last_deterministic_event_id is None
                else self.last_deterministic_event_id.to_dict()
            ),
            "previous_cursor_sha256": (
                None
                if self.previous_cursor_sha256 is None
                else self.previous_cursor_sha256.to_dict()
            ),
            "effects_performed": self.effects_performed,
        }

    def cursor_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "ScienceObservationPrefixCursorV1":
        required = {
            "schema_version",
            "instrument_manifest_sha256",
            "semantic_execution_id",
            "crca_semantic_run_digest",
            "science_semantic_acquisition_sha256",
            "cell_id",
            "logical_world_id",
            "world_uid",
            "checkpoint_at_tick",
            "next_absolute_tick",
            "next_world_local_sequence",
            "prefix_record_count",
            "prefix_chain_sha256",
            "last_deterministic_event_id",
            "previous_cursor_sha256",
            "effects_performed",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise contract_invalid(
                "observation cursor object has unknown or missing fields",
                field="science_observation_prefix_cursor",
            )

        def optional_digest(field_name: str) -> ContentDigest | None:
            item = value[field_name]
            return None if item is None else _digest_from_dict(item, field=field_name)

        return cls(
            instrument_manifest_sha256=_digest_from_dict(
                value["instrument_manifest_sha256"],
                field="instrument_manifest_sha256",
            ),
            semantic_execution_id=_digest_from_dict(
                value["semantic_execution_id"], field="semantic_execution_id"
            ),
            crca_semantic_run_digest=_digest_from_dict(
                value["crca_semantic_run_digest"],
                field="crca_semantic_run_digest",
            ),
            science_semantic_acquisition_sha256=_digest_from_dict(
                value["science_semantic_acquisition_sha256"],
                field="science_semantic_acquisition_sha256",
            ),
            cell_id=value["cell_id"],
            logical_world_id=value["logical_world_id"],
            world_uid=_digest_from_dict(value["world_uid"], field="world_uid"),
            checkpoint_at_tick=value["checkpoint_at_tick"],
            next_absolute_tick=value["next_absolute_tick"],
            next_world_local_sequence=value["next_world_local_sequence"],
            prefix_record_count=value["prefix_record_count"],
            prefix_chain_sha256=_digest_from_dict(
                value["prefix_chain_sha256"], field="prefix_chain_sha256"
            ),
            last_deterministic_event_id=optional_digest("last_deterministic_event_id"),
            previous_cursor_sha256=optional_digest("previous_cursor_sha256"),
            schema_version=SchemaVersion(value["schema_version"]),
            effects_performed=value["effects_performed"],
        )


def seal_science_observation_prefix_cursor_v1(
    rows: tuple[ScienceObservationRecordV1, ...],
    *,
    instrument_manifest_sha256: ContentDigest,
    crca_semantic_run_digest: ContentDigest,
    science_semantic_acquisition_sha256: ContentDigest,
    cell_id: str,
    logical_world_id: int,
    semantic_world_key: CanonicalPayloadV1,
    checkpoint_at_tick: int,
    producer_phase_registry_ordinals: dict[str, int],
    prior_cursor: ScienceObservationPrefixCursorV1 | None = None,
) -> ScienceObservationPrefixCursorV1:
    """Seal a half-open observation prefix ending at an absolute tick."""

    require_int(checkpoint_at_tick, field="checkpoint_at_tick", minimum=0)
    require_instance(semantic_world_key, CanonicalPayloadV1, field="semantic_world_key")
    if not isinstance(producer_phase_registry_ordinals, dict) or any(
        not isinstance(component_id, str)
        or not component_id
        or type(ordinal) is not int
        or ordinal < 0
        for component_id, ordinal in producer_phase_registry_ordinals.items()
    ):
        raise contract_invalid(
            "producer phase registry is invalid",
            field="producer_phase_registry_ordinals",
        )
    semantic_execution_id = derive_science_semantic_execution_id_v1(
        crca_semantic_run_digest, science_semantic_acquisition_sha256
    )
    world_uid = derive_science_world_uid_v1(semantic_execution_id, semantic_world_key)
    start_tick = 0
    start_sequence = 0
    prefix_record_count = 0
    prefix_sha256 = empty_science_observation_prefix_sha256_v1()
    last_event_id = None
    previous_cursor_sha256 = None
    if prior_cursor is not None:
        require_instance(
            prior_cursor,
            ScienceObservationPrefixCursorV1,
            field="prior_cursor",
        )
        if (
            prior_cursor.instrument_manifest_sha256 != instrument_manifest_sha256
            or prior_cursor.semantic_execution_id != semantic_execution_id
            or prior_cursor.crca_semantic_run_digest != crca_semantic_run_digest
            or prior_cursor.science_semantic_acquisition_sha256
            != science_semantic_acquisition_sha256
            or prior_cursor.cell_id != cell_id
            or prior_cursor.logical_world_id != logical_world_id
            or prior_cursor.world_uid != world_uid
        ):
            raise contract_invalid(
                "prior observation cursor identity differs",
                field="prior_cursor",
            )
        start_tick = prior_cursor.next_absolute_tick
        start_sequence = prior_cursor.next_world_local_sequence
        prefix_record_count = prior_cursor.prefix_record_count
        prefix_sha256 = prior_cursor.prefix_chain_sha256
        last_event_id = prior_cursor.last_deterministic_event_id
        previous_cursor_sha256 = prior_cursor.cursor_sha256()
    if checkpoint_at_tick < start_tick:
        raise contract_invalid(
            "checkpoint precedes the prior observation cursor",
            field="checkpoint_at_tick",
        )
    if not isinstance(rows, tuple) or any(
        not isinstance(row, ScienceObservationRecordV1) for row in rows
    ):
        raise contract_invalid(
            "rows must be a tuple of science observation records", field="rows"
        )
    if any(
        row.instrument_manifest_sha256 != instrument_manifest_sha256
        or row.semantic_execution_id != semantic_execution_id
        or row.crca_semantic_run_digest != crca_semantic_run_digest
        or row.science_semantic_acquisition_sha256
        != science_semantic_acquisition_sha256
        or row.cell_id != cell_id
        or row.logical_world_id != logical_world_id
        or row.world_uid != world_uid
        or not start_tick <= row.absolute_tick < checkpoint_at_tick
        for row in rows
    ):
        raise contract_invalid(
            "observation row identity or tick span differs from cursor boundary",
            field="rows",
        )
    validate_science_observation_stream_v1(
        rows,
        producer_phase_registry_ordinals=producer_phase_registry_ordinals,
        expected_start_world_local_sequence=start_sequence,
    )
    if rows:
        last_event_id = rows[-1].deterministic_event_id
    return ScienceObservationPrefixCursorV1(
        instrument_manifest_sha256=instrument_manifest_sha256,
        semantic_execution_id=semantic_execution_id,
        crca_semantic_run_digest=crca_semantic_run_digest,
        science_semantic_acquisition_sha256=(science_semantic_acquisition_sha256),
        cell_id=cell_id,
        logical_world_id=logical_world_id,
        world_uid=world_uid,
        checkpoint_at_tick=checkpoint_at_tick,
        next_absolute_tick=checkpoint_at_tick,
        next_world_local_sequence=start_sequence + len(rows),
        prefix_record_count=prefix_record_count + len(rows),
        prefix_chain_sha256=extend_science_observation_prefix_sha256_v1(
            prefix_sha256, rows
        ),
        last_deterministic_event_id=last_event_id,
        previous_cursor_sha256=previous_cursor_sha256,
    )


def finalize_science_observation_drafts_from_cursor_v1(
    drafts: tuple[ScienceObservationDraftV1, ...],
    *,
    cursor: ScienceObservationPrefixCursorV1,
    instrument_manifest_sha256: ContentDigest,
    crca_semantic_run_digest: ContentDigest,
    science_semantic_acquisition_sha256: ContentDigest,
    cell_id: str,
    logical_world_id: int,
    semantic_world_key: CanonicalPayloadV1,
) -> tuple[ScienceObservationRecordV1, ...]:
    """Finalize post-checkpoint drafts with the cursor's next event sequence."""

    require_instance(cursor, ScienceObservationPrefixCursorV1, field="cursor")
    semantic_execution_id = derive_science_semantic_execution_id_v1(
        crca_semantic_run_digest, science_semantic_acquisition_sha256
    )
    world_uid = derive_science_world_uid_v1(semantic_execution_id, semantic_world_key)
    if (
        cursor.instrument_manifest_sha256 != instrument_manifest_sha256
        or cursor.semantic_execution_id != semantic_execution_id
        or cursor.crca_semantic_run_digest != crca_semantic_run_digest
        or cursor.science_semantic_acquisition_sha256
        != science_semantic_acquisition_sha256
        or cursor.cell_id != cell_id
        or cursor.logical_world_id != logical_world_id
        or cursor.world_uid != world_uid
    ):
        raise contract_invalid(
            "observation cursor identity differs from continuation",
            field="cursor",
        )
    if not isinstance(drafts, tuple) or any(
        not isinstance(draft, ScienceObservationDraftV1) for draft in drafts
    ):
        raise contract_invalid(
            "drafts must be a tuple of science observation drafts", field="drafts"
        )
    if any(draft.absolute_tick < cursor.next_absolute_tick for draft in drafts):
        raise contract_invalid(
            "observation draft precedes the continuation cursor",
            field="drafts",
        )
    return finalize_science_observation_drafts_v1(
        drafts,
        instrument_manifest_sha256=instrument_manifest_sha256,
        crca_semantic_run_digest=crca_semantic_run_digest,
        science_semantic_acquisition_sha256=(science_semantic_acquisition_sha256),
        cell_id=cell_id,
        logical_world_id=logical_world_id,
        semantic_world_key=semantic_world_key,
        start_world_local_sequence=cursor.next_world_local_sequence,
    )


__all__ = (
    "SCIENCE_OBSERVATION_PREFIX_CHAIN_SCHEMA_VERSION",
    "SCIENCE_OBSERVATION_PREFIX_CURSOR_SCHEMA_VERSION",
    "ScienceObservationPrefixCursorV1",
    "empty_science_observation_prefix_sha256_v1",
    "extend_science_observation_prefix_sha256_v1",
    "finalize_science_observation_drafts_from_cursor_v1",
    "seal_science_observation_prefix_cursor_v1",
)
