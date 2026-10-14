from __future__ import annotations

import copy
import hashlib
import json

import pytest

from engine.perf.recurrent_native.modular_recurrent_mutation_receipt_v1 import (
    float64_to_bits,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_sequence_v1 import (
    CAPACITY,
    OVERFLOW_SCHEMA_VERSION,
    SCHEMA_VERSION,
    ModularRecurrentMutationSequenceOverflowError,
    ModularRecurrentMutationSequenceValidationError,
    append_modular_recurrent_mutation_sequence_v1,
    build_modular_recurrent_mutation_sequence_v1,
    canonical_modular_recurrent_mutation_sequence_bytes_v1,
    modular_recurrent_mutation_sequence_overflow_contract_payload_v1,
    mutation_sequence_entry_sha256_v1,
    mutation_sequence_sha256_v1,
    validate_modular_recurrent_mutation_sequence_v1,
)


SOURCE_SHA = "11" * 32


def _entry(
    sequence_index: int,
    *,
    generation: int | None = None,
    child_agent_id: int | None = None,
    child_slot: int | None = None,
    child_slot_token: int | None = None,
) -> dict[str, object]:
    return {
        "sequence_index": sequence_index,
        "generation": sequence_index + 1 if generation is None else generation,
        "tick": 0,
        "parent_agent_id": 100 + sequence_index,
        "parent_slot": 2 + sequence_index,
        "parent_slot_token": 12 + sequence_index,
        "child_agent_id": (
            200 + sequence_index if child_agent_id is None else child_agent_id
        ),
        "child_slot": 4 + sequence_index if child_slot is None else child_slot,
        "child_slot_token": (
            18 + sequence_index if child_slot_token is None else child_slot_token
        ),
        "draw0_float64_bits": float64_to_bits(0.6),
        "draw1_float64_bits": float64_to_bits(0.6796875),
        "operation": "refine_descendant_parameter",
        "status": "accepted",
        "reason": "proposal_valid",
        "selected_instance_id_index": 7,
        "selected_homology_id_index": 3,
        "changed_dimension": "live_input_gain",
        "changed": True,
        "pre_genome_sha256": f"{30 + sequence_index:02x}" * 32,
        "post_genome_sha256": f"{40 + sequence_index:02x}" * 32,
        "committed": True,
    }


def _sequence(count: int = 2):
    return build_modular_recurrent_mutation_sequence_v1(
        profile_id="b2-16x48-v1",
        source_packed_sha256=SOURCE_SHA,
        entries=[_entry(index) for index in range(count)],
    )


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _redigest_entry(wire: dict[str, object], index: int) -> None:
    entry = wire["entries"][index]  # type: ignore[index]
    body = {key: value for key, value in entry.items() if key != "receipt_sha256"}
    entry["receipt_sha256"] = _canonical_digest(body)


def _redigest_sequence(wire: dict[str, object]) -> dict[str, object]:
    body = {key: value for key, value in wire.items() if key != "sequence_sha256"}
    wire["sequence_sha256"] = _canonical_digest(body)
    return wire


def test_build_validate_and_canonicalize_fixed_capacity_sequence() -> None:
    sequence = _sequence()
    assert sequence.schema_version == SCHEMA_VERSION
    assert sequence.capacity == CAPACITY == 3
    assert sequence.count == 2
    assert len(sequence.entries) == 3
    assert sequence.entries[2] is None
    assert sequence.entries[0] is not None
    assert sequence.entries[0].receipt_sha256 == mutation_sequence_entry_sha256_v1(
        sequence.entries[0]
    )
    assert sequence.sequence_sha256 == mutation_sequence_sha256_v1(sequence)
    wire = json.loads(canonical_modular_recurrent_mutation_sequence_bytes_v1(sequence))
    assert validate_modular_recurrent_mutation_sequence_v1(wire) == sequence


def test_canonical_bytes_ignore_mapping_insertion_order() -> None:
    wire = _sequence(1).to_dict()
    reversed_wire = dict(reversed(tuple(wire.items())))
    reversed_wire["entries"] = [
        dict(reversed(tuple(wire["entries"][0].items()))),  # type: ignore[index,union-attr]
        None,
        None,
    ]
    assert canonical_modular_recurrent_mutation_sequence_bytes_v1(
        reversed_wire
    ) == canonical_modular_recurrent_mutation_sequence_bytes_v1(wire)


@pytest.mark.parametrize("scope", ["top", "entry"])
def test_unknown_fields_fail_closed_even_when_digests_are_recomputed(
    scope: str,
) -> None:
    wire = _sequence(1).to_dict()
    if scope == "top":
        wire["unexpected"] = 1
    else:
        wire["entries"][0]["unexpected"] = 1  # type: ignore[index]
        _redigest_entry(wire, 0)
    with pytest.raises(
        ModularRecurrentMutationSequenceValidationError, match="unknown fields"
    ):
        validate_modular_recurrent_mutation_sequence_v1(_redigest_sequence(wire))


@pytest.mark.parametrize("scope", ["top", "entry"])
def test_missing_fields_fail_closed(scope: str) -> None:
    wire = _sequence(1).to_dict()
    if scope == "top":
        del wire["profile_id"]
    else:
        del wire["entries"][0]["tick"]  # type: ignore[index]
    with pytest.raises(
        ModularRecurrentMutationSequenceValidationError, match="missing required"
    ):
        validate_modular_recurrent_mutation_sequence_v1(wire)


@pytest.mark.parametrize(
    ("capacity", "entries"),
    [
        (2, [None, None, None]),
        (4, [None, None, None]),
        (3, [None, None]),
        (3, [None, None, None, None]),
    ],
)
def test_capacity_and_array_length_are_exact(
    capacity: int, entries: list[None]
) -> None:
    wire = _sequence(0).to_dict()
    wire["capacity"] = capacity
    wire["entries"] = entries
    with pytest.raises(ModularRecurrentMutationSequenceValidationError):
        validate_modular_recurrent_mutation_sequence_v1(_redigest_sequence(wire))


@pytest.mark.parametrize(
    ("count", "entries", "match"),
    [
        (1, [None, None, None], "active"),
        (0, ["entry", None, None], "inactive suffix"),
        (1, ["entry", None, "entry"], "inactive suffix"),
    ],
)
def test_count_requires_exact_active_prefix_and_inactive_none_suffix(
    count: int, entries: list[object], match: str
) -> None:
    valid_entry = _sequence(1).to_dict()["entries"][0]  # type: ignore[index]
    wire = _sequence(0).to_dict()
    wire["count"] = count
    wire["entries"] = [valid_entry if item == "entry" else item for item in entries]
    with pytest.raises(ModularRecurrentMutationSequenceValidationError, match=match):
        validate_modular_recurrent_mutation_sequence_v1(_redigest_sequence(wire))


def test_entry_and_sequence_digest_tampering_are_detected_separately() -> None:
    wire = _sequence(1).to_dict()
    wire["entries"][0]["reason"] = "tampered"  # type: ignore[index]
    with pytest.raises(
        ModularRecurrentMutationSequenceValidationError, match="receipt_sha256"
    ):
        validate_modular_recurrent_mutation_sequence_v1(wire)

    _redigest_entry(wire, 0)
    with pytest.raises(
        ModularRecurrentMutationSequenceValidationError, match="sequence_sha256"
    ):
        validate_modular_recurrent_mutation_sequence_v1(wire)


@pytest.mark.parametrize("indexes", [(1,), (0, 2), (1, 0)])
def test_sequence_indexes_are_contiguous(indexes: tuple[int, ...]) -> None:
    with pytest.raises(
        ModularRecurrentMutationSequenceValidationError, match="canonical contiguous"
    ):
        build_modular_recurrent_mutation_sequence_v1(
            profile_id="b2-16x48-v1",
            source_packed_sha256=SOURCE_SHA,
            entries=[_entry(index) for index in indexes],
        )


def test_generation_must_be_contiguous() -> None:
    with pytest.raises(
        ModularRecurrentMutationSequenceValidationError, match="contiguous"
    ):
        build_modular_recurrent_mutation_sequence_v1(
            profile_id="b2-16x48-v1",
            source_packed_sha256=SOURCE_SHA,
            entries=[_entry(0, generation=3), _entry(1, generation=2)],
        )


@pytest.mark.parametrize("duplicate", ["agent_id", "owner"])
def test_child_id_and_slot_token_owner_are_unique(duplicate: str) -> None:
    second = _entry(1)
    if duplicate == "agent_id":
        second["child_agent_id"] = 200
    else:
        second["child_slot"] = 4
        second["child_slot_token"] = 18
    with pytest.raises(
        ModularRecurrentMutationSequenceValidationError, match="duplicate"
    ):
        build_modular_recurrent_mutation_sequence_v1(
            profile_id="b2-16x48-v1",
            source_packed_sha256=SOURCE_SHA,
            entries=[_entry(0), second],
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("committed", False, "committed"),
        ("draw0_float64_bits", "3FE3333333333333", "draw0"),
        ("changed", 1, "changed"),
        ("parent_slot", -1, "parent_slot"),
        ("pre_genome_sha256", "AA" * 32, "pre_genome"),
    ],
)
def test_active_entry_types_and_canonical_values_fail_closed(
    field: str, value: object, match: str
) -> None:
    entry = _entry(0)
    entry[field] = value
    with pytest.raises(ModularRecurrentMutationSequenceValidationError, match=match):
        build_modular_recurrent_mutation_sequence_v1(
            profile_id="b2-16x48-v1",
            source_packed_sha256=SOURCE_SHA,
            entries=[entry],
        )


def test_changed_flag_requires_an_exact_genome_digest_transition() -> None:
    entry = _entry(0)
    entry["post_genome_sha256"] = entry["pre_genome_sha256"]
    with pytest.raises(
        ModularRecurrentMutationSequenceValidationError, match="digest transition"
    ):
        build_modular_recurrent_mutation_sequence_v1(
            profile_id="b2-16x48-v1",
            source_packed_sha256=SOURCE_SHA,
            entries=[entry],
        )


def test_builder_accepts_exact_padded_form_and_rejects_bad_count() -> None:
    compact = [_entry(0), None, None]
    sequence = build_modular_recurrent_mutation_sequence_v1(
        profile_id="b2-16x48-v1",
        source_packed_sha256=SOURCE_SHA,
        entries=compact,
        count=1,
        capacity=3,
    )
    assert sequence.count == 1
    assert sequence.entries[1:] == (None, None)
    with pytest.raises(ModularRecurrentMutationSequenceValidationError, match="count"):
        build_modular_recurrent_mutation_sequence_v1(
            profile_id="b2-16x48-v1",
            source_packed_sha256=SOURCE_SHA,
            entries=compact,
            count=2,
        )


def test_append_returns_a_new_sequence_without_mutating_inputs() -> None:
    sequence_wire = _sequence(1).to_dict()
    entry = _entry(1)
    original_sequence = copy.deepcopy(sequence_wire)
    original_entry = copy.deepcopy(entry)
    appended = append_modular_recurrent_mutation_sequence_v1(sequence_wire, entry)
    assert appended.count == 2
    assert appended.entries[1] is not None
    assert sequence_wire == original_sequence
    assert entry == original_entry


def test_overflow_fails_closed_and_reports_the_unchanged_exact_prefix() -> None:
    sequence_wire = _sequence(3).to_dict()
    attempted = _entry(3)
    original_sequence = copy.deepcopy(sequence_wire)
    original_attempted = copy.deepcopy(attempted)
    expected_payload = modular_recurrent_mutation_sequence_overflow_contract_payload_v1(
        sequence_wire
    )

    with pytest.raises(ModularRecurrentMutationSequenceOverflowError) as captured:
        append_modular_recurrent_mutation_sequence_v1(sequence_wire, attempted)

    assert sequence_wire == original_sequence
    assert attempted == original_attempted
    assert captured.value.payload == expected_payload
    assert expected_payload["schema_version"] == OVERFLOW_SCHEMA_VERSION
    assert expected_payload["attempted_sequence_index"] == CAPACITY
    assert expected_payload["status"] == "rejected"
    assert expected_payload["reason"] == "capacity_overflow"
    assert expected_payload["entries"] == original_sequence["entries"]
    assert expected_payload["sequence_sha256"] == original_sequence["sequence_sha256"]


def test_overflow_payload_requires_an_already_full_valid_sequence() -> None:
    with pytest.raises(
        ModularRecurrentMutationSequenceValidationError, match="equal capacity"
    ):
        modular_recurrent_mutation_sequence_overflow_contract_payload_v1(_sequence(2))


def test_empty_sequence_has_three_inactive_slots() -> None:
    sequence = _sequence(0)
    assert sequence.count == 0
    assert sequence.entries == (None, None, None)
    assert (
        validate_modular_recurrent_mutation_sequence_v1(sequence.to_dict()) == sequence
    )
