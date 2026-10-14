from __future__ import annotations

from dataclasses import replace

import pytest

from emergence_contracts import (
    CanonicalPayloadV1,
    ContentDigest,
    ContractError,
    SchemaVersion,
    ScienceObservationDraftV1,
    ScienceRecordKindV1,
    canonical_science_record_stream_sha256_v1,
    finalize_science_observation_drafts_v1,
    validate_science_observation_stream_v1,
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


def _draft(address: str, binding_id: str) -> ScienceObservationDraftV1:
    return ScienceObservationDraftV1(
        producer_component_id="science-control-producer-v1",
        producer_contract_version=SchemaVersion("science_control_producer_v1"),
        producer_phase_registry_ordinal=2,
        absolute_tick=7,
        semantic_source_address=address,
        record_kind=ScienceRecordKindV1.CONTROL_BINDING,
        record_kind_version=1,
        control_binding_sha256=_digest("8"),
        payload=_payload(binding_id),
    )


def _finalize(drafts: tuple[ScienceObservationDraftV1, ...]):
    return finalize_science_observation_drafts_v1(
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


def test_physical_discovery_permutation_preserves_records_and_stream_hash() -> None:
    left = _draft("tick:7/phase:2/entity:1/op:0", "binding-a")
    right = _draft("tick:7/phase:2/entity:2/op:0", "binding-b")
    first = _finalize((left, right))
    permuted = _finalize((right, left))

    assert first == permuted
    assert [row.world_local_sequence for row in first] == [0, 1]
    assert canonical_science_record_stream_sha256_v1(first) == (
        canonical_science_record_stream_sha256_v1(permuted)
    )
    validate_science_observation_stream_v1(
        first,
        producer_phase_registry_ordinals={"science-control-producer-v1": 2},
    )


def test_event_and_sequence_tampering_fail_recomputation() -> None:
    rows = _finalize(
        (
            _draft("tick:7/phase:2/entity:1/op:0", "binding-a"),
            _draft("tick:7/phase:2/entity:2/op:0", "binding-b"),
        )
    )
    with pytest.raises(ContractError, match="event ID mismatch"):
        validate_science_observation_stream_v1(
            (replace(rows[0], deterministic_event_id=_digest("f")), rows[1]),
            producer_phase_registry_ordinals={"science-control-producer-v1": 2},
        )
    with pytest.raises(ContractError, match="not contiguous"):
        validate_science_observation_stream_v1(
            (rows[0], replace(rows[1], world_local_sequence=3)),
            producer_phase_registry_ordinals={"science-control-producer-v1": 2},
        )


def test_legacy_display_metadata_cannot_change_semantic_stream_identity() -> None:
    rows = _finalize((_draft("tick:7/phase:2/entity:1/op:0", "binding-a"),))
    with_legacy = (
        replace(
            rows[0],
            legacy_compatibility=CanonicalPayloadV1.from_value(
                SchemaVersion("science_legacy_display_metadata_v1"),
                {"generation_label": "G-display-only"},
            ),
        ),
    )
    assert rows[0].semantic_record_sha256() == with_legacy[0].semantic_record_sha256()
    assert canonical_science_record_stream_sha256_v1(rows) == (
        canonical_science_record_stream_sha256_v1(with_legacy)
    )


def test_duplicate_physical_free_source_address_fails_before_sequence_assignment() -> (
    None
):
    duplicate = "tick:7/phase:2/entity:1/op:0"
    with pytest.raises(ContractError, match="source address is duplicated"):
        _finalize((_draft(duplicate, "binding-a"), _draft(duplicate, "binding-b")))


def test_draft_rejects_invalid_registry_and_record_versions() -> None:
    with pytest.raises(ContractError, match="integer must be"):
        replace(
            _draft("tick:7/phase:2/entity:1/op:0", "binding-a"),
            producer_phase_registry_ordinal=-1,
        )
    with pytest.raises(ContractError, match="record-kind version"):
        replace(
            _draft("tick:7/phase:2/entity:1/op:0", "binding-a"), record_kind_version=2
        )
