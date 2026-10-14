from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

from emergence_contracts import (
    CanonicalPayloadV1,
    ContentDigest,
    ContractError,
    SchemaVersion,
    ScienceObservationDraftV1,
    ScienceObservationPrefixCursorV1,
    ScienceRecordKindV1,
    empty_science_observation_prefix_sha256_v1,
    extend_science_observation_prefix_sha256_v1,
    finalize_science_observation_drafts_from_cursor_v1,
    finalize_science_observation_drafts_v1,
    seal_science_observation_prefix_cursor_v1,
)

ROOT = Path(__file__).resolve().parents[2]


def _digest(character: str) -> ContentDigest:
    return ContentDigest("sha256", character * 64)


def _world_key() -> CanonicalPayloadV1:
    return CanonicalPayloadV1.from_value(
        SchemaVersion("science_semantic_world_key_v1"),
        {"semantic_cell_id": "cell-001", "repeat_ordinal": 0},
    )


def _draft(tick: int) -> ScienceObservationDraftV1:
    return ScienceObservationDraftV1(
        producer_component_id="portable-science-producer-v1",
        producer_contract_version=SchemaVersion("portable_science_producer_v1"),
        producer_phase_registry_ordinal=2,
        absolute_tick=tick,
        semantic_source_address=f"tick:{tick}/phase:2/control:selection.pressure",
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


def _finalize(drafts, *, start_sequence: int = 0):
    return finalize_science_observation_drafts_v1(
        drafts,
        instrument_manifest_sha256=_digest("1"),
        crca_semantic_run_digest=_digest("2"),
        science_semantic_acquisition_sha256=_digest("3"),
        cell_id="cell-001",
        logical_world_id=0,
        semantic_world_key=_world_key(),
        start_world_local_sequence=start_sequence,
    )


def _seal(rows, *, checkpoint_at_tick: int, prior_cursor=None):
    return seal_science_observation_prefix_cursor_v1(
        rows,
        instrument_manifest_sha256=_digest("1"),
        crca_semantic_run_digest=_digest("2"),
        science_semantic_acquisition_sha256=_digest("3"),
        cell_id="cell-001",
        logical_world_id=0,
        semantic_world_key=_world_key(),
        checkpoint_at_tick=checkpoint_at_tick,
        producer_phase_registry_ordinals={"portable-science-producer-v1": 2},
        prior_cursor=prior_cursor,
    )


def test_resume_preserves_uninterrupted_future_event_ids_and_prefix_chain() -> None:
    drafts = (_draft(2), _draft(12))
    uninterrupted = _finalize(drafts)
    prefix = _finalize((drafts[0],))
    cursor = _seal(prefix, checkpoint_at_tick=10)
    resumed = finalize_science_observation_drafts_from_cursor_v1(
        (drafts[1],),
        cursor=cursor,
        instrument_manifest_sha256=_digest("1"),
        crca_semantic_run_digest=_digest("2"),
        science_semantic_acquisition_sha256=_digest("3"),
        cell_id="cell-001",
        logical_world_id=0,
        semantic_world_key=_world_key(),
    )

    assert prefix + resumed == uninterrupted
    terminal = _seal(resumed, checkpoint_at_tick=20, prior_cursor=cursor)
    assert terminal.prefix_record_count == 2
    assert terminal.next_world_local_sequence == 2
    assert terminal.previous_cursor_sha256 == cursor.cursor_sha256()
    assert terminal.prefix_chain_sha256 == (
        extend_science_observation_prefix_sha256_v1(
            empty_science_observation_prefix_sha256_v1(), uninterrupted
        )
    )


def test_cursor_round_trip_matches_portable_schema_and_excludes_route_identity() -> (
    None
):
    cursor = _seal(_finalize((_draft(2),)), checkpoint_at_tick=10)
    payload = cursor.to_dict()
    schema = json.loads(
        (
            ROOT
            / "_spec/science_instrument/v1/science_observation_prefix_cursor.schema.json"
        ).read_text()
    )

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(payload, schema)
    assert ScienceObservationPrefixCursorV1.from_dict(payload) == cursor
    assert not ({"route_id", "physical_partition", "provider_id"} & set(payload))


def test_cursor_fails_closed_on_identity_or_absolute_tick_mismatch() -> None:
    cursor = _seal(_finalize((_draft(2),)), checkpoint_at_tick=10)
    with pytest.raises(ContractError, match="identity differs"):
        finalize_science_observation_drafts_from_cursor_v1(
            (_draft(12),),
            cursor=cursor,
            instrument_manifest_sha256=_digest("9"),
            crca_semantic_run_digest=_digest("2"),
            science_semantic_acquisition_sha256=_digest("3"),
            cell_id="cell-001",
            logical_world_id=0,
            semantic_world_key=_world_key(),
        )
    with pytest.raises(ContractError, match="precedes"):
        finalize_science_observation_drafts_from_cursor_v1(
            (_draft(9),),
            cursor=cursor,
            instrument_manifest_sha256=_digest("1"),
            crca_semantic_run_digest=_digest("2"),
            science_semantic_acquisition_sha256=_digest("3"),
            cell_id="cell-001",
            logical_world_id=0,
            semantic_world_key=_world_key(),
        )
    with pytest.raises(ContractError, match="prefix record count"):
        replace(cursor, next_world_local_sequence=4)
