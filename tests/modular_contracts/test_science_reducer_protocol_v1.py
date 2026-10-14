from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from emergence_evidence import (
    CountingScienceReducerV1,
    ScienceReducerError,
    execute_science_reducer_v1,
)
from emergence_contracts import (
    CanonicalPayloadV1,
    ContentDigest,
    SchemaVersion,
    ScienceEvidencePartitionV1,
    ScienceObservationDraftV1,
    ScienceObservationProfileV1,
    ScienceRecordKindV1,
    ScienceReducerDescriptorV1,
    finalize_science_observation_drafts_v1,
)


def _digest(character: str) -> ContentDigest:
    return ContentDigest("sha256", character * 64)


def _payload(binding_id: str) -> CanonicalPayloadV1:
    return CanonicalPayloadV1.from_value(
        SchemaVersion("science_record_payload_v1"),
        {
            "binding_id": binding_id,
            "semantic_cell_id": "cell-001",
            "seed_bindings": {"ecology": 11},
            "control_values": {"selection.pressure": 0.1},
            "control_schedules": {"selection.pressure": {"kind": "static"}},
            "registered_profile_ids": ["world-v1"],
            "semantic_envelope_resolution_sha256": "9" * 64,
            "semantic_applied_readback_sha256": "a" * 64,
            "continuation_parent_execution_identity": None,
        },
    )


def _partitions() -> tuple[ScienceEvidencePartitionV1, ...]:
    drafts = tuple(
        ScienceObservationDraftV1(
            producer_component_id="science-control-producer-v1",
            producer_contract_version=SchemaVersion("science_control_producer_v1"),
            producer_phase_registry_ordinal=2,
            absolute_tick=tick,
            semantic_source_address=f"tick:{tick}/phase:2/entity:1/op:0",
            record_kind=ScienceRecordKindV1.CONTROL_BINDING,
            record_kind_version=1,
            control_binding_sha256=_digest("8"),
            payload=_payload(f"binding-{tick}"),
        )
        for tick in (7, 9)
    )
    rows = finalize_science_observation_drafts_v1(
        drafts,
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
    return (
        ScienceEvidencePartitionV1(
            instrument_manifest_sha256=_digest("1"),
            science_semantic_acquisition_sha256=_digest("3"),
            observation_abi_version=SchemaVersion("eiel_science_observation_abi_v1"),
            profile=ScienceObservationProfileV1.STANDARD,
            world_uid=rows[0].world_uid,
            record_kind=ScienceRecordKindV1.CONTROL_BINDING,
            absolute_tick_start=0,
            absolute_tick_end=8,
            rows=(rows[0],),
        ),
        ScienceEvidencePartitionV1(
            instrument_manifest_sha256=_digest("1"),
            science_semantic_acquisition_sha256=_digest("3"),
            observation_abi_version=SchemaVersion("eiel_science_observation_abi_v1"),
            profile=ScienceObservationProfileV1.STANDARD,
            world_uid=rows[0].world_uid,
            record_kind=ScienceRecordKindV1.CONTROL_BINDING,
            absolute_tick_start=8,
            absolute_tick_end=10,
            rows=(rows[1],),
        ),
    )


def _descriptor() -> ScienceReducerDescriptorV1:
    return ScienceReducerDescriptorV1(
        reducer_id="science.counting",
        contract_version=SchemaVersion("science_counting_reducer_v1"),
        implementation_sha256=_digest("4"),
        configuration_schema_sha256=_digest("5"),
        required_observation_capability_ids=("causal_lifecycle",),
        output_contract_id="eiel_science_count_result_v1",
        purity_declarations=(
            "DETERMINISTIC_CANONICAL_OUTPUT",
            "NO_SIMULATOR_IMPORT",
            "NO_TRAJECTORY_MUTATION",
            "OFFLINE_ONLY",
        ),
    )


def _execute(
    partitions: tuple[ScienceEvidencePartitionV1, ...],
    *,
    route_character: str = "6",
    resume=None,
    stop_after_partitions: int | None = None,
    capabilities: tuple[str, ...] = ("causal_lifecycle",),
):
    return execute_science_reducer_v1(
        _descriptor(),
        CountingScienceReducerV1(implementation_sha256=_digest("4")),
        CanonicalPayloadV1.from_value(
            SchemaVersion("science_counting_config_v1"), {"group_by": "kind"}
        ),
        partitions,
        evidence_capability_ids=capabilities,
        campaign_spec_sha256=_digest("0"),
        instrument_manifest_sha256=_digest("1"),
        route_binding_sha256=_digest(route_character),
        science_semantic_acquisition_sha256=_digest("3"),
        physical_execution_plan_sha256=_digest(route_character),
        science_execution_plan_sha256=_digest(route_character),
        reduction_plan_sha256=_digest("7"),
        resume=resume,
        stop_after_partitions=stop_after_partitions,
    )


def test_reducer_is_partition_order_independent_and_route_result_neutral() -> None:
    partitions = _partitions()
    b136 = _execute(partitions)
    permuted = _execute(tuple(reversed(partitions)))
    next_engine = _execute(partitions, route_character="8")

    assert b136.complete and permuted.complete and next_engine.complete
    assert b136.result_payload == permuted.result_payload == next_engine.result_payload
    assert b136.result_receipt.result_sha256 == next_engine.result_receipt.result_sha256
    assert b136.result_receipt.receipt_sha256() != (
        next_engine.result_receipt.receipt_sha256()
    )
    assert b136.result_receipt.trajectory_effects_performed is False
    schema = json.loads(
        (
            Path(__file__).parents[2]
            / "_spec/science_instrument/v1/science_result_receipt.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(b136.result_receipt.to_dict(), schema)


def test_reducer_resume_preserves_result_payload_and_hash() -> None:
    partitions = _partitions()
    uninterrupted = _execute(partitions)
    interrupted = _execute(partitions, stop_after_partitions=1)
    assert not interrupted.complete and interrupted.checkpoint is not None
    resumed = _execute(partitions, resume=interrupted.checkpoint)

    assert resumed.complete
    assert resumed.result_payload == uninterrupted.result_payload
    assert resumed.result_receipt.result_sha256 == (
        uninterrupted.result_receipt.result_sha256
    )
    assert resumed.result_receipt.resume_chain_sha256 == (
        interrupted.checkpoint.checkpoint_sha256()
    )


def test_reducer_fails_closed_on_missing_capability_or_changed_prefix() -> None:
    partitions = _partitions()
    with pytest.raises(ScienceReducerError, match="capabilities are missing"):
        _execute(partitions, capabilities=())

    interrupted = _execute(partitions, stop_after_partitions=1)
    with pytest.raises(ScienceReducerError, match="identity or prefix differs"):
        _execute((partitions[1],), resume=interrupted.checkpoint)


def test_zero_row_partition_produces_explicit_zero_result() -> None:
    template = _partitions()[0]
    empty = ScienceEvidencePartitionV1(
        instrument_manifest_sha256=template.instrument_manifest_sha256,
        science_semantic_acquisition_sha256=(
            template.science_semantic_acquisition_sha256
        ),
        observation_abi_version=template.observation_abi_version,
        profile=template.profile,
        world_uid=template.world_uid,
        record_kind=template.record_kind,
        absolute_tick_start=0,
        absolute_tick_end=10,
        rows=(),
    )
    result = _execute((empty,))
    assert result.complete
    assert result.result_payload.to_value()["row_count"] == 0
    assert result.result_payload.to_value()["world_count"] == 1
