"""Deterministic route-neutral publication of canonical science evidence."""

from __future__ import annotations

from dataclasses import dataclass

from emergence_contracts import (
    ContentDigest,
    ProfileResourceEnvelopeV1,
    SchemaVersion,
    ScienceEvidencePartitionV1,
    ScienceObservationProfileV1,
    ScienceObservationProfileDescriptorV1,
    ScienceObservationRecordV1,
    ScienceRecordKindV1,
    canonical_json_bytes,
    canonical_science_record_stream_sha256_v1,
    canonical_sha256,
    validate_science_observation_stream_v1,
)

from .science_reducer_protocol_v1 import science_evidence_manifest_sha256_v1

SCIENCE_OBSERVATION_PUBLICATION_PLAN_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_observation_publication_plan_v1"
)
SCIENCE_WORLD_EVIDENCE_PUBLICATION_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_world_evidence_publication_v1"
)

_KIND_ORDINAL = {kind: ordinal for ordinal, kind in enumerate(ScienceRecordKindV1)}


class SciencePublicationError(ValueError):
    """Canonical evidence cannot be sealed within its declared profile."""


def build_science_observation_publication_plan_v1(
    *,
    instrument_manifest_sha256: ContentDigest,
    science_semantic_acquisition_sha256: ContentDigest,
    profile_descriptor: ScienceObservationProfileDescriptorV1,
    resource_envelope: ProfileResourceEnvelopeV1,
    absolute_tick_start: int,
    absolute_tick_end: int,
    expected_start_world_local_sequence: int = 0,
    partition_span_ticks: int | None = None,
) -> "ScienceObservationPublicationPlanV1":
    """Build a publication plan solely from frozen profile and route capacity."""

    if not isinstance(profile_descriptor, ScienceObservationProfileDescriptorV1):
        raise SciencePublicationError("profile_descriptor is invalid")
    if not isinstance(resource_envelope, ProfileResourceEnvelopeV1):
        raise SciencePublicationError("resource_envelope is invalid")
    if resource_envelope.profile is not profile_descriptor.profile:
        raise SciencePublicationError(
            "resource envelope and observation profile differ"
        )
    span = (
        profile_descriptor.default_partition_span_ticks
        if partition_span_ticks is None
        else partition_span_ticks
    )
    return ScienceObservationPublicationPlanV1(
        instrument_manifest_sha256=instrument_manifest_sha256,
        science_semantic_acquisition_sha256=(science_semantic_acquisition_sha256),
        observation_abi_version=profile_descriptor.observation_abi_version,
        profile=profile_descriptor.profile,
        absolute_tick_start=absolute_tick_start,
        absolute_tick_end=absolute_tick_end,
        partition_span_ticks=span,
        expected_start_world_local_sequence=(expected_start_world_local_sequence),
        required_record_kinds=profile_descriptor.record_kinds,
        required_nonempty_record_kinds=(
            profile_descriptor.required_nonempty_record_kinds
        ),
        producer_phase_registry_ordinals=(
            profile_descriptor.producer_phase_registry_ordinals
        ),
        resource_envelope=resource_envelope,
    )


@dataclass(frozen=True, slots=True)
class ScienceObservationPublicationPlanV1:
    instrument_manifest_sha256: ContentDigest
    science_semantic_acquisition_sha256: ContentDigest
    observation_abi_version: SchemaVersion
    profile: ScienceObservationProfileV1
    absolute_tick_start: int
    absolute_tick_end: int
    partition_span_ticks: int
    expected_start_world_local_sequence: int
    required_record_kinds: tuple[ScienceRecordKindV1, ...]
    required_nonempty_record_kinds: tuple[ScienceRecordKindV1, ...]
    producer_phase_registry_ordinals: tuple[tuple[str, int], ...]
    resource_envelope: ProfileResourceEnvelopeV1
    schema_version: SchemaVersion = SCIENCE_OBSERVATION_PUBLICATION_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "instrument_manifest_sha256",
            "science_semantic_acquisition_sha256",
        ):
            if not isinstance(getattr(self, field_name), ContentDigest):
                raise SciencePublicationError(f"{field_name} is invalid")
        if not isinstance(self.observation_abi_version, SchemaVersion):
            raise SciencePublicationError("observation_abi_version is invalid")
        if not isinstance(self.profile, ScienceObservationProfileV1):
            raise SciencePublicationError("profile is invalid")
        if not isinstance(self.resource_envelope, ProfileResourceEnvelopeV1):
            raise SciencePublicationError("resource_envelope is invalid")
        for field_name in (
            "absolute_tick_start",
            "absolute_tick_end",
            "partition_span_ticks",
            "expected_start_world_local_sequence",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise SciencePublicationError(f"{field_name} must be nonnegative")
        if self.absolute_tick_end <= self.absolute_tick_start:
            raise SciencePublicationError("publication tick span must be nonempty")
        if self.partition_span_ticks < 1:
            raise SciencePublicationError("partition_span_ticks must be positive")
        for field_name in (
            "required_record_kinds",
            "required_nonempty_record_kinds",
        ):
            kinds = getattr(self, field_name)
            if (
                not isinstance(kinds, tuple)
                or not kinds
                or any(not isinstance(item, ScienceRecordKindV1) for item in kinds)
                or kinds
                != tuple(sorted(set(kinds), key=lambda item: _KIND_ORDINAL[item]))
            ):
                raise SciencePublicationError(
                    f"{field_name} must be unique and in registry order"
                )
        if not set(self.required_nonempty_record_kinds).issubset(
            self.required_record_kinds
        ):
            raise SciencePublicationError(
                "required nonempty kinds must be acquired record kinds"
            )
        registry = self.producer_phase_registry_ordinals
        if not isinstance(registry, tuple) or any(
            not isinstance(row, tuple) or len(row) != 2 for row in registry
        ):
            raise SciencePublicationError("producer phase registry is malformed")
        if registry != tuple(sorted(registry)) or len(
            {row[0] for row in registry}
        ) != len(registry):
            raise SciencePublicationError(
                "producer phase registry must be sorted with unique component IDs"
            )
        for component_id, ordinal in registry:
            if (
                not isinstance(component_id, str)
                or not component_id
                or type(ordinal) is not int
                or ordinal < 0
            ):
                raise SciencePublicationError("producer phase registry row is invalid")
        if self.resource_envelope.profile is not self.profile:
            raise SciencePublicationError("resource envelope profile differs")
        max_counts, max_bytes = _capacity_tables(self)
        if any(
            max_counts.get(kind, 0) < 1 or max_bytes.get(kind, 0) < 1
            for kind in self.required_record_kinds
        ):
            raise SciencePublicationError(
                "resource envelope does not cover every profile record kind"
            )
        if self.schema_version != SCIENCE_OBSERVATION_PUBLICATION_PLAN_SCHEMA_VERSION:
            raise SciencePublicationError("publication plan schema is unsupported")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "instrument_manifest_sha256": self.instrument_manifest_sha256.to_dict(),
            "science_semantic_acquisition_sha256": (
                self.science_semantic_acquisition_sha256.to_dict()
            ),
            "observation_abi_version": str(self.observation_abi_version),
            "profile_id": self.profile.value,
            "absolute_tick_start": self.absolute_tick_start,
            "absolute_tick_end": self.absolute_tick_end,
            "partition_span_ticks": self.partition_span_ticks,
            "expected_start_world_local_sequence": (
                self.expected_start_world_local_sequence
            ),
            "required_record_kinds": [
                kind.value for kind in self.required_record_kinds
            ],
            "required_nonempty_record_kinds": [
                kind.value for kind in self.required_nonempty_record_kinds
            ],
            "producer_phase_registry_ordinals": {
                component_id: ordinal
                for component_id, ordinal in self.producer_phase_registry_ordinals
            },
            "resource_envelope": self.resource_envelope.to_dict(),
        }

    def plan_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ScienceWorldEvidencePublicationV1:
    plan: ScienceObservationPublicationPlanV1
    world_uid: ContentDigest
    records: tuple[ScienceObservationRecordV1, ...]
    partitions: tuple[ScienceEvidencePartitionV1, ...]
    schema_version: SchemaVersion = SCIENCE_WORLD_EVIDENCE_PUBLICATION_SCHEMA_VERSION
    effects_performed: bool = False

    def __post_init__(self) -> None:
        if self.effects_performed is not False:
            raise SciencePublicationError("evidence publication must be effect-free")
        if self.schema_version != SCIENCE_WORLD_EVIDENCE_PUBLICATION_SCHEMA_VERSION:
            raise SciencePublicationError("publication schema is unsupported")

    def evidence_manifest_sha256(self) -> ContentDigest:
        return science_evidence_manifest_sha256_v1(self.partitions)

    def canonical_record_stream_sha256(self) -> ContentDigest:
        return canonical_science_record_stream_sha256_v1(self.records)

    def publication_core(self) -> dict[str, object]:
        partition_rows = [
            {
                "world_uid": item.world_uid.to_dict(),
                "record_kind": item.record_kind.value,
                "absolute_tick_start": item.absolute_tick_start,
                "absolute_tick_end": item.absolute_tick_end,
                "row_count": len(item.rows),
                "partition_sha256": item.partition_sha256().to_dict(),
            }
            for item in self.partitions
        ]
        return {
            "schema_version": str(self.schema_version),
            "publication_plan_sha256": self.plan.plan_sha256().to_dict(),
            "world_uid": self.world_uid.to_dict(),
            "record_count": len(self.records),
            "canonical_record_stream_sha256": (
                self.canonical_record_stream_sha256().to_dict()
            ),
            "evidence_manifest_sha256": self.evidence_manifest_sha256().to_dict(),
            "partition_count": len(self.partitions),
            "zero_row_partition_count": sum(not item.rows for item in self.partitions),
            "partitions": partition_rows,
            "effects_performed": self.effects_performed,
        }

    def to_dict(self) -> dict[str, object]:
        core = self.publication_core()
        return {
            **core,
            "publication_receipt_sha256": canonical_sha256(core).to_dict(),
        }


def _capacity_tables(
    plan: ScienceObservationPublicationPlanV1,
) -> tuple[dict[ScienceRecordKindV1, int], dict[ScienceRecordKindV1, int]]:
    return (
        dict(plan.resource_envelope.max_records_per_tick_per_world_by_kind),
        dict(plan.resource_envelope.max_record_bytes_by_kind),
    )


def _validate_record_capacity(
    records: tuple[ScienceObservationRecordV1, ...],
    plan: ScienceObservationPublicationPlanV1,
) -> None:
    max_counts, max_bytes = _capacity_tables(plan)
    observed_counts: dict[tuple[int, ScienceRecordKindV1], int] = {}
    for row in records:
        if row.record_kind not in plan.required_record_kinds:
            raise SciencePublicationError(
                f"record kind is outside publication plan: {row.record_kind.value}"
            )
        count_key = (row.absolute_tick, row.record_kind)
        observed_counts[count_key] = observed_counts.get(count_key, 0) + 1
        limit = max_counts.get(row.record_kind)
        if limit is None or observed_counts[count_key] > limit:
            raise SciencePublicationError(
                f"record count exceeds profile capacity: {row.record_kind.value}"
            )
        byte_limit = max_bytes.get(row.record_kind)
        if byte_limit is None or len(canonical_json_bytes(row.to_dict())) > byte_limit:
            raise SciencePublicationError(
                f"record bytes exceed profile capacity: {row.record_kind.value}"
            )


def publish_science_world_evidence_v1(
    physical_chunks: tuple[tuple[ScienceObservationRecordV1, ...], ...],
    plan: ScienceObservationPublicationPlanV1,
) -> ScienceWorldEvidencePublicationV1:
    """Seal one world independently of physical discovery and chunk boundaries."""

    if not isinstance(plan, ScienceObservationPublicationPlanV1):
        raise SciencePublicationError("plan is invalid")
    if not isinstance(physical_chunks, tuple) or any(
        not isinstance(chunk, tuple) for chunk in physical_chunks
    ):
        raise SciencePublicationError("physical_chunks must be tuples")
    if any(
        not isinstance(row, ScienceObservationRecordV1)
        for chunk in physical_chunks
        for row in chunk
    ):
        raise SciencePublicationError(
            "physical_chunks must contain science observation records"
        )
    records = tuple(
        sorted(
            (row for chunk in physical_chunks for row in chunk),
            key=lambda row: (row.absolute_tick, row.world_local_sequence),
        )
    )
    if not records:
        raise SciencePublicationError(
            "a world publication requires at least one canonical record"
        )
    world_uid = records[0].world_uid
    if any(
        row.instrument_manifest_sha256 != plan.instrument_manifest_sha256
        or row.science_semantic_acquisition_sha256
        != plan.science_semantic_acquisition_sha256
        or row.world_uid != world_uid
        or not plan.absolute_tick_start <= row.absolute_tick < plan.absolute_tick_end
        for row in records
    ):
        raise SciencePublicationError("record identity or tick span differs from plan")
    validate_science_observation_stream_v1(
        records,
        producer_phase_registry_ordinals=dict(plan.producer_phase_registry_ordinals),
        expected_start_world_local_sequence=(plan.expected_start_world_local_sequence),
    )
    _validate_record_capacity(records, plan)
    present_kinds = {row.record_kind for row in records}
    missing_nonempty = [
        kind.value
        for kind in plan.required_nonempty_record_kinds
        if kind not in present_kinds
    ]
    if missing_nonempty:
        raise SciencePublicationError(
            f"required nonempty record kinds are missing: {missing_nonempty}"
        )

    partitions: list[ScienceEvidencePartitionV1] = []
    for start in range(
        plan.absolute_tick_start,
        plan.absolute_tick_end,
        plan.partition_span_ticks,
    ):
        end = min(start + plan.partition_span_ticks, plan.absolute_tick_end)
        for kind in plan.required_record_kinds:
            partition = ScienceEvidencePartitionV1(
                instrument_manifest_sha256=plan.instrument_manifest_sha256,
                science_semantic_acquisition_sha256=(
                    plan.science_semantic_acquisition_sha256
                ),
                observation_abi_version=plan.observation_abi_version,
                profile=plan.profile,
                world_uid=world_uid,
                record_kind=kind,
                absolute_tick_start=start,
                absolute_tick_end=end,
                rows=tuple(
                    row
                    for row in records
                    if row.record_kind is kind and start <= row.absolute_tick < end
                ),
            )
            if len(partition.rows) > plan.resource_envelope.chunk_max_records:
                raise SciencePublicationError("partition record capacity exceeded")
            if (
                len(canonical_json_bytes(partition.to_dict()))
                > plan.resource_envelope.chunk_max_bytes
            ):
                raise SciencePublicationError("partition byte capacity exceeded")
            partitions.append(partition)
    partitions.sort(
        key=lambda item: (
            _KIND_ORDINAL[item.record_kind],
            item.absolute_tick_start,
            item.absolute_tick_end,
        )
    )
    return ScienceWorldEvidencePublicationV1(
        plan=plan,
        world_uid=world_uid,
        records=records,
        partitions=tuple(partitions),
    )


__all__ = (
    "SCIENCE_OBSERVATION_PUBLICATION_PLAN_SCHEMA_VERSION",
    "SCIENCE_WORLD_EVIDENCE_PUBLICATION_SCHEMA_VERSION",
    "ScienceObservationPublicationPlanV1",
    "SciencePublicationError",
    "ScienceWorldEvidencePublicationV1",
    "build_science_observation_publication_plan_v1",
    "publish_science_world_evidence_v1",
)
