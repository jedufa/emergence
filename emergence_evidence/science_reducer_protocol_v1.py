"""Pure offline reducer protocol over canonical science evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from emergence_contracts import (
    CanonicalPayloadV1,
    ContentDigest,
    SchemaVersion,
    ScienceEvidencePartitionV1,
    ScienceRecordKindV1,
    ScienceReducerDescriptorV1,
    ScienceResultReceiptV1,
    canonical_sha256,
)

SCIENCE_REDUCER_CHECKPOINT_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_reducer_checkpoint_v1"
)
_RECORD_KIND_ORDINALS = {
    kind: ordinal for ordinal, kind in enumerate(ScienceRecordKindV1)
}


class ScienceReducerError(ValueError):
    """Canonical evidence is insufficient or reducer identity differs."""


class ScienceReducerImplementationV1(Protocol):
    reducer_id: str
    implementation_sha256: ContentDigest

    def initial_state(
        self, configuration: CanonicalPayloadV1
    ) -> CanonicalPayloadV1: ...

    def consume_partition(
        self,
        state: CanonicalPayloadV1,
        partition: ScienceEvidencePartitionV1,
    ) -> CanonicalPayloadV1: ...

    def finalize(
        self,
        state: CanonicalPayloadV1,
        configuration: CanonicalPayloadV1,
        *,
        output_contract_id: str,
    ) -> CanonicalPayloadV1: ...


def _ordered_partitions(
    partitions: tuple[ScienceEvidencePartitionV1, ...],
) -> tuple[ScienceEvidencePartitionV1, ...]:
    return tuple(
        sorted(
            partitions,
            key=lambda item: (
                item.world_uid.hex,
                _RECORD_KIND_ORDINALS[item.record_kind],
                item.absolute_tick_start,
                -1 if item.first_sequence is None else item.first_sequence,
                item.absolute_tick_end,
            ),
        )
    )


def _partition_identity(partition: ScienceEvidencePartitionV1) -> dict[str, object]:
    return {
        "instrument_manifest_sha256": partition.instrument_manifest_sha256.to_dict(),
        "science_semantic_acquisition_sha256": (
            partition.science_semantic_acquisition_sha256.to_dict()
        ),
        "observation_abi_version": str(partition.observation_abi_version),
        "profile_id": partition.profile.value,
        "world_uid": partition.world_uid.to_dict(),
        "record_kind": partition.record_kind.value,
        "absolute_tick_start": partition.absolute_tick_start,
        "absolute_tick_end": partition.absolute_tick_end,
        "first_sequence": partition.first_sequence,
        "last_sequence": partition.last_sequence,
        "row_count": len(partition.rows),
        "rows_sha256": partition.rows_sha256().to_dict(),
    }


def science_evidence_manifest_sha256_v1(
    partitions: tuple[ScienceEvidencePartitionV1, ...],
) -> ContentDigest:
    return canonical_sha256(
        [_partition_identity(item) for item in _ordered_partitions(partitions)]
    )


@dataclass(frozen=True, slots=True)
class ScienceReducerCheckpointV1:
    reducer_descriptor_sha256: ContentDigest
    reducer_implementation_sha256: ContentDigest
    configuration_sha256: ContentDigest
    evidence_manifest_sha256: ContentDigest
    next_partition_index: int
    processed_prefix_sha256: ContentDigest
    accumulator: CanonicalPayloadV1
    schema_version: SchemaVersion = SCIENCE_REDUCER_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "reducer_descriptor_sha256",
            "reducer_implementation_sha256",
            "configuration_sha256",
            "evidence_manifest_sha256",
            "processed_prefix_sha256",
        ):
            if not isinstance(getattr(self, field_name), ContentDigest):
                raise ScienceReducerError(f"{field_name} is invalid")
        if type(self.next_partition_index) is not int or self.next_partition_index < 0:
            raise ScienceReducerError("next_partition_index is invalid")
        if not isinstance(self.accumulator, CanonicalPayloadV1):
            raise ScienceReducerError("accumulator is invalid")
        if self.schema_version != SCIENCE_REDUCER_CHECKPOINT_SCHEMA_VERSION:
            raise ScienceReducerError("unsupported reducer checkpoint schema")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "reducer_descriptor_sha256": self.reducer_descriptor_sha256.to_dict(),
            "reducer_implementation_sha256": self.reducer_implementation_sha256.to_dict(),
            "configuration_sha256": self.configuration_sha256.to_dict(),
            "evidence_manifest_sha256": self.evidence_manifest_sha256.to_dict(),
            "next_partition_index": self.next_partition_index,
            "processed_prefix_sha256": self.processed_prefix_sha256.to_dict(),
            "accumulator": self.accumulator.to_dict(),
        }

    def checkpoint_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ScienceReducerProgressV1:
    complete: bool
    checkpoint: ScienceReducerCheckpointV1 | None
    result_payload: CanonicalPayloadV1 | None
    result_receipt: ScienceResultReceiptV1 | None

    def __post_init__(self) -> None:
        if self.complete:
            if (
                self.checkpoint is not None
                or self.result_payload is None
                or self.result_receipt is None
            ):
                raise ScienceReducerError("complete reducer progress has invalid shape")
        elif (
            self.checkpoint is None
            or self.result_payload is not None
            or self.result_receipt is not None
        ):
            raise ScienceReducerError("incomplete reducer progress has invalid shape")


def execute_science_reducer_v1(
    descriptor: ScienceReducerDescriptorV1,
    implementation: ScienceReducerImplementationV1,
    configuration: CanonicalPayloadV1,
    partitions: tuple[ScienceEvidencePartitionV1, ...],
    *,
    evidence_capability_ids: tuple[str, ...],
    campaign_spec_sha256: ContentDigest,
    instrument_manifest_sha256: ContentDigest,
    route_binding_sha256: ContentDigest,
    science_semantic_acquisition_sha256: ContentDigest,
    physical_execution_plan_sha256: ContentDigest,
    science_execution_plan_sha256: ContentDigest,
    reduction_plan_sha256: ContentDigest,
    resume: ScienceReducerCheckpointV1 | None = None,
    stop_after_partitions: int | None = None,
) -> ScienceReducerProgressV1:
    """Reduce canonical evidence in stable order, with exact-prefix resume."""

    if implementation.reducer_id != descriptor.reducer_id:
        raise ScienceReducerError("reducer implementation ID differs")
    if implementation.implementation_sha256 != descriptor.implementation_sha256:
        raise ScienceReducerError("reducer implementation identity differs")
    missing = sorted(
        set(descriptor.required_observation_capability_ids)
        - set(evidence_capability_ids)
    )
    if missing:
        raise ScienceReducerError(
            f"required observation capabilities are missing: {missing}"
        )
    ordered = _ordered_partitions(partitions)
    evidence_manifest_sha256 = science_evidence_manifest_sha256_v1(ordered)
    for partition in ordered:
        if partition.instrument_manifest_sha256 != instrument_manifest_sha256:
            raise ScienceReducerError("evidence instrument identity differs")
        if (
            partition.science_semantic_acquisition_sha256
            != science_semantic_acquisition_sha256
        ):
            raise ScienceReducerError("evidence acquisition identity differs")
    descriptor_sha256 = descriptor.descriptor_sha256()
    configuration_sha256 = canonical_sha256(configuration.to_dict())
    start = 0
    resume_chain_sha256 = None
    if resume is None:
        state = implementation.initial_state(configuration)
    else:
        expected_prefix = canonical_sha256(
            [
                _partition_identity(item)
                for item in ordered[: resume.next_partition_index]
            ]
        )
        if (
            resume.reducer_descriptor_sha256 != descriptor_sha256
            or resume.reducer_implementation_sha256
            != implementation.implementation_sha256
            or resume.configuration_sha256 != configuration_sha256
            or resume.evidence_manifest_sha256 != evidence_manifest_sha256
            or resume.processed_prefix_sha256 != expected_prefix
            or resume.next_partition_index > len(ordered)
        ):
            raise ScienceReducerError("reducer checkpoint identity or prefix differs")
        start = resume.next_partition_index
        state = resume.accumulator
        resume_chain_sha256 = resume.checkpoint_sha256()
    end = len(ordered)
    if stop_after_partitions is not None:
        if type(stop_after_partitions) is not int or stop_after_partitions < 0:
            raise ScienceReducerError("stop_after_partitions is invalid")
        end = min(end, start + stop_after_partitions)
    for partition in ordered[start:end]:
        state = implementation.consume_partition(state, partition)
    if end < len(ordered):
        checkpoint = ScienceReducerCheckpointV1(
            reducer_descriptor_sha256=descriptor_sha256,
            reducer_implementation_sha256=implementation.implementation_sha256,
            configuration_sha256=configuration_sha256,
            evidence_manifest_sha256=evidence_manifest_sha256,
            next_partition_index=end,
            processed_prefix_sha256=canonical_sha256(
                [_partition_identity(item) for item in ordered[:end]]
            ),
            accumulator=state,
        )
        return ScienceReducerProgressV1(
            complete=False,
            checkpoint=checkpoint,
            result_payload=None,
            result_receipt=None,
        )

    result_payload = implementation.finalize(
        state, configuration, output_contract_id=descriptor.output_contract_id
    )
    result_sha256 = canonical_sha256(result_payload.to_value())
    receipt = ScienceResultReceiptV1(
        campaign_spec_sha256=campaign_spec_sha256,
        instrument_manifest_sha256=instrument_manifest_sha256,
        route_binding_sha256=route_binding_sha256,
        science_semantic_acquisition_sha256=science_semantic_acquisition_sha256,
        physical_execution_plan_sha256=physical_execution_plan_sha256,
        science_execution_plan_sha256=science_execution_plan_sha256,
        evidence_manifest_sha256=evidence_manifest_sha256,
        reduction_plan_sha256=reduction_plan_sha256,
        reducer_descriptor_sha256=descriptor_sha256,
        reducer_implementation_sha256=implementation.implementation_sha256,
        reducer_configuration_sha256=configuration_sha256,
        output_contract_id=descriptor.output_contract_id,
        result_sha256=result_sha256,
        resume_chain_sha256=resume_chain_sha256,
    )
    return ScienceReducerProgressV1(
        complete=True,
        checkpoint=None,
        result_payload=result_payload,
        result_receipt=receipt,
    )


@dataclass(frozen=True, slots=True)
class CountingScienceReducerV1:
    """Generic deterministic reducer used as a protocol and empty-world sentinel."""

    implementation_sha256: ContentDigest
    reducer_id: str = "science.counting"

    def initial_state(self, configuration: CanonicalPayloadV1) -> CanonicalPayloadV1:
        return CanonicalPayloadV1.from_value(
            SchemaVersion("eiel_science_counting_state_v1"),
            {
                "row_count": 0,
                "counts_by_kind": {},
                "world_uids": [],
                "completion_classes": {},
            },
        )

    def consume_partition(
        self,
        state: CanonicalPayloadV1,
        partition: ScienceEvidencePartitionV1,
    ) -> CanonicalPayloadV1:
        value = dict(state.to_value())
        counts = dict(value["counts_by_kind"])
        counts[partition.record_kind.value] = counts.get(
            partition.record_kind.value, 0
        ) + len(partition.rows)
        worlds = set(value["world_uids"])
        worlds.add(partition.world_uid.hex)
        completions = dict(value["completion_classes"])
        for row in partition.rows:
            if row.record_kind.value == "world_completion":
                completion_class = row.payload.to_value()["completion_class"]
                completions[completion_class] = completions.get(completion_class, 0) + 1
        return CanonicalPayloadV1.from_value(
            SchemaVersion("eiel_science_counting_state_v1"),
            {
                "row_count": value["row_count"] + len(partition.rows),
                "counts_by_kind": dict(sorted(counts.items())),
                "world_uids": sorted(worlds),
                "completion_classes": dict(sorted(completions.items())),
            },
        )

    def finalize(
        self,
        state: CanonicalPayloadV1,
        configuration: CanonicalPayloadV1,
        *,
        output_contract_id: str,
    ) -> CanonicalPayloadV1:
        value = dict(state.to_value())
        return CanonicalPayloadV1.from_value(
            SchemaVersion(output_contract_id),
            {
                "row_count": value["row_count"],
                "world_count": len(value["world_uids"]),
                "counts_by_kind": value["counts_by_kind"],
                "completion_classes": value["completion_classes"],
                "configuration_sha256": canonical_sha256(
                    configuration.to_dict()
                ).to_dict(),
            },
        )


__all__ = (
    "SCIENCE_REDUCER_CHECKPOINT_SCHEMA_VERSION",
    "CountingScienceReducerV1",
    "ScienceReducerCheckpointV1",
    "ScienceReducerError",
    "ScienceReducerImplementationV1",
    "ScienceReducerProgressV1",
    "execute_science_reducer_v1",
    "science_evidence_manifest_sha256_v1",
)
