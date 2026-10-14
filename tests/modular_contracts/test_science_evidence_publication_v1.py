from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest
from referencing import Registry, Resource

from emergence_contracts import (
    CanonicalPayloadV1,
    ContentDigest,
    ProfileResourceEnvelopeV1,
    SchemaVersion,
    ScienceObservationDraftV1,
    ScienceObservationProfileV1,
    ScienceObservationProfileDescriptorV1,
    ScienceRecordKindV1,
    finalize_science_observation_drafts_v1,
)
from emergence_evidence import (
    ScienceObservationPublicationPlanV1,
    SciencePublicationError,
    build_science_observation_publication_plan_v1,
    publish_science_world_evidence_v1,
)


def _digest(character: str) -> ContentDigest:
    return ContentDigest("sha256", character * 64)


def _draft(*, tick: int, address: str) -> ScienceObservationDraftV1:
    return ScienceObservationDraftV1(
        producer_component_id="portable-science-producer-v1",
        producer_contract_version=SchemaVersion("portable_science_producer_v1"),
        producer_phase_registry_ordinal=2,
        absolute_tick=tick,
        semantic_source_address=address,
        record_kind=ScienceRecordKindV1.CONTROL_BINDING,
        record_kind_version=1,
        control_binding_sha256=_digest("4"),
        payload=CanonicalPayloadV1.from_value(
            SchemaVersion("science_record_payload_v1"),
            {
                "binding_id": f"binding-{tick}",
                "semantic_cell_id": "cell-001",
                "seed_bindings": {"task_environment": 7},
                "control_values": {"selection.pressure": 0.1},
                "control_schedules": {"selection.pressure": {"kind": "static"}},
                "registered_profile_ids": ["world-v1"],
                "semantic_envelope_resolution_sha256": "5" * 64,
                "semantic_applied_readback_sha256": "6" * 64,
                "continuation_parent_execution_identity": None,
            },
        ),
    )


def _records():
    return finalize_science_observation_drafts_v1(
        (
            _draft(tick=12, address="tick:12/phase:2/control:selection.pressure"),
            _draft(tick=2, address="tick:2/phase:2/control:selection.pressure"),
        ),
        instrument_manifest_sha256=_digest("1"),
        crca_semantic_run_digest=_digest("2"),
        science_semantic_acquisition_sha256=_digest("3"),
        cell_id="cell-001",
        logical_world_id=0,
        semantic_world_key=CanonicalPayloadV1.from_value(
            SchemaVersion("science_semantic_world_key_v1"),
            {"semantic_cell_id": "cell-001", "repeat_ordinal": 0},
        ),
    )


def _resource_envelope() -> ProfileResourceEnvelopeV1:
    chunk_max_bytes = 20_000
    max_in_flight_chunks = 2
    fixed_buffer_bytes = 100
    per_world_cursor_bytes = 16
    per_agent_bytes = 1
    logical_world_count = 1
    resident_capacity = 10
    return ProfileResourceEnvelopeV1(
        profile=ScienceObservationProfileV1.STANDARD,
        max_records_per_tick_per_world_by_kind=(
            (ScienceRecordKindV1.CONTROL_BINDING, 2),
            (ScienceRecordKindV1.WORLD_STATE_SNAPSHOT, 1),
        ),
        max_record_bytes_by_kind=(
            (ScienceRecordKindV1.CONTROL_BINDING, 4_096),
            (ScienceRecordKindV1.WORLD_STATE_SNAPSHOT, 4_096),
        ),
        chunk_max_records=16,
        chunk_max_bytes=chunk_max_bytes,
        max_in_flight_chunks=max_in_flight_chunks,
        fixed_buffer_bytes=fixed_buffer_bytes,
        per_world_cursor_bytes=per_world_cursor_bytes,
        per_resident_agent_observation_state_bytes=per_agent_bytes,
        logical_world_count=logical_world_count,
        resident_capacity=resident_capacity,
        max_resident_observation_bytes=(
            fixed_buffer_bytes
            + max_in_flight_chunks * chunk_max_bytes
            + logical_world_count * per_world_cursor_bytes
            + logical_world_count * resident_capacity * per_agent_bytes
        ),
    )


def _plan() -> ScienceObservationPublicationPlanV1:
    return ScienceObservationPublicationPlanV1(
        instrument_manifest_sha256=_digest("1"),
        science_semantic_acquisition_sha256=_digest("3"),
        observation_abi_version=SchemaVersion("eiel_science_observation_abi_v1"),
        profile=ScienceObservationProfileV1.STANDARD,
        absolute_tick_start=0,
        absolute_tick_end=20,
        partition_span_ticks=10,
        expected_start_world_local_sequence=0,
        required_record_kinds=(
            ScienceRecordKindV1.CONTROL_BINDING,
            ScienceRecordKindV1.WORLD_STATE_SNAPSHOT,
        ),
        required_nonempty_record_kinds=(ScienceRecordKindV1.CONTROL_BINDING,),
        producer_phase_registry_ordinals=(("portable-science-producer-v1", 2),),
        resource_envelope=_resource_envelope(),
    )


def _profile_descriptor() -> ScienceObservationProfileDescriptorV1:
    return ScienceObservationProfileDescriptorV1(
        profile=ScienceObservationProfileV1.STANDARD,
        observation_abi_version=SchemaVersion("eiel_science_observation_abi_v1"),
        record_kinds=(
            ScienceRecordKindV1.CONTROL_BINDING,
            ScienceRecordKindV1.WORLD_STATE_SNAPSHOT,
        ),
        required_nonempty_record_kinds=(ScienceRecordKindV1.CONTROL_BINDING,),
        capability_ids=("causal_lifecycle",),
        producer_phase_registry_ordinals=(("portable-science-producer-v1", 2),),
        default_partition_span_ticks=10,
    )


def test_publication_is_independent_of_physical_chunks_and_emits_typed_zero_rows() -> (
    None
):
    records = _records()
    contiguous = publish_science_world_evidence_v1((records,), _plan())
    permuted_chunks = publish_science_world_evidence_v1(
        ((records[1],), (records[0],)), _plan()
    )

    assert contiguous.to_dict() == permuted_chunks.to_dict()
    assert contiguous.partitions == permuted_chunks.partitions
    assert contiguous.to_dict()["partition_count"] == 4
    assert contiguous.to_dict()["zero_row_partition_count"] == 2
    assert all(
        partition.to_dict()["rows_sha256"]["hex"] for partition in contiguous.partitions
    )
    assert contiguous.effects_performed is False


def test_publication_fails_closed_on_missing_completeness_or_profile_overflow() -> None:
    records = _records()
    missing_required = replace(
        _plan(),
        required_nonempty_record_kinds=(
            ScienceRecordKindV1.CONTROL_BINDING,
            ScienceRecordKindV1.WORLD_STATE_SNAPSHOT,
        ),
    )
    with pytest.raises(SciencePublicationError, match="nonempty record kinds"):
        publish_science_world_evidence_v1((records,), missing_required)

    resource = replace(
        _resource_envelope(),
        max_record_bytes_by_kind=(
            (ScienceRecordKindV1.CONTROL_BINDING, 1),
            (ScienceRecordKindV1.WORLD_STATE_SNAPSHOT, 4_096),
        ),
    )
    with pytest.raises(SciencePublicationError, match="record bytes"):
        publish_science_world_evidence_v1(
            (records,), replace(_plan(), resource_envelope=resource)
        )


def test_resource_envelope_is_horizon_independent() -> None:
    short = _plan()
    long = replace(short, absolute_tick_end=5_000)

    assert short.resource_envelope == long.resource_envelope
    assert short.resource_envelope.computed_resident_bytes() == (
        long.resource_envelope.computed_resident_bytes()
    )


def test_publication_plan_is_derived_from_frozen_profile_and_route_capacity() -> None:
    plan = build_science_observation_publication_plan_v1(
        instrument_manifest_sha256=_digest("1"),
        science_semantic_acquisition_sha256=_digest("3"),
        profile_descriptor=_profile_descriptor(),
        resource_envelope=_resource_envelope(),
        absolute_tick_start=0,
        absolute_tick_end=20,
    )

    assert plan.to_dict() == _plan().to_dict()
    with pytest.raises(SciencePublicationError, match="does not cover"):
        build_science_observation_publication_plan_v1(
            instrument_manifest_sha256=_digest("1"),
            science_semantic_acquisition_sha256=_digest("3"),
            profile_descriptor=_profile_descriptor(),
            resource_envelope=replace(
                _resource_envelope(),
                max_record_bytes_by_kind=(
                    (ScienceRecordKindV1.CONTROL_BINDING, 4_096),
                ),
            ),
            absolute_tick_start=0,
            absolute_tick_end=20,
        )


def test_publication_plan_and_receipt_match_portable_wire_schemas() -> None:
    plan = _plan()
    publication = publish_science_world_evidence_v1((_records(),), plan)
    schema_root = Path(__file__).parents[2] / "_spec/science_instrument/v1"
    names = (
        "science_profile_resource_envelope.schema.json",
        "science_observation_publication_plan.schema.json",
        "science_world_evidence_publication.schema.json",
    )
    schemas = {
        name: json.loads((schema_root / name).read_text(encoding="utf-8"))
        for name in names
    }
    registry = Registry()
    for schema in schemas.values():
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))

    for schema in schemas.values():
        jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(
        schemas["science_observation_publication_plan.schema.json"],
        registry=registry,
    ).validate(plan.to_dict())
    jsonschema.validate(
        publication.to_dict(),
        schemas["science_world_evidence_publication.schema.json"],
    )
