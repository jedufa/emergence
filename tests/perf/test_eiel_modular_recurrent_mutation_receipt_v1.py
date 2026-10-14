from __future__ import annotations

import copy
import hashlib
import json
import math

import pytest

from engine.perf.recurrent_native.modular_recurrent_mutation_receipt_v1 import (
    CONTRACT_ID,
    SCHEMA_VERSION,
    ModularRecurrentMutationReceiptValidationError,
    build_modular_recurrent_mutation_receipt_v1,
    canonical_modular_recurrent_mutation_receipt_bytes_v1,
    float64_from_bits,
    float64_to_bits,
    mutation_receipt_sha256_v1,
    turnover_mutation_ancestry_sha256_v1,
    validate_modular_recurrent_mutation_receipt_v1,
)


SOURCE_SHA = "11" * 32
ANCESTRY_SHA = "22" * 32


def _transaction(
    ordinal: int = 0,
    *,
    child_agent_id: int = 200,
    child_slot: int = 4,
    child_slot_token: int = 18,
) -> dict[str, object]:
    return {
        "transaction_kind": "turnover",
        "transaction_ordinal": ordinal,
        "parent_agent_id": 100,
        "parent_slot": 2,
        "parent_slot_token": 12,
        "child_agent_id": child_agent_id,
        "child_slot": child_slot,
        "child_slot_token": child_slot_token,
        "attempt_ordinal": 0,
        "draw0_float64_bits": float64_to_bits(0.6),
        "draw1_float64_bits": float64_to_bits(0.6796875),
        "operation": "refine_descendant_parameter",
        "status": "accepted",
        "reason": "proposal_valid",
        "selected_instance_id_index": 7,
        "selected_homology_id_index": 3,
        "changed_dimension": "live_input_gain",
        "changed": True,
    }


def _receipt(*transactions: dict[str, object]):
    return build_modular_recurrent_mutation_receipt_v1(
        profile_id="b2-16x48-v1",
        source_packed_sha256=SOURCE_SHA,
        ancestry_tape_sha256=ANCESTRY_SHA,
        transactions=transactions or (_transaction(),),
    )


def _redigest(wire: dict[str, object]) -> dict[str, object]:
    body = {key: value for key, value in wire.items() if key != "canonical_sha256"}
    wire["canonical_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return wire


def test_float64_bits_preserve_exact_ieee_754_values() -> None:
    for value in (0.0, -0.0, 0.6, 0.6796875, math.nextafter(1.0, 0.0)):
        bits = float64_to_bits(value)
        assert len(bits) == 16
        assert float64_to_bits(float64_from_bits(bits)) == bits
    assert float64_to_bits(0.0) != float64_to_bits(-0.0)


def test_turnover_ancestry_digest_preserves_original_canonical_formula() -> None:
    payload = {
        "schema_version": "eiel_modular_turnover_mutation_ancestry_v1",
        "parent_slot": 28,
        "child_slot": 64,
        "parent_agent_id": 29,
        "child_agent_id": 65,
        "child_slot_token": 1,
    }
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    assert turnover_mutation_ancestry_sha256_v1(
        parent_slot=28,
        child_slot=64,
        parent_agent_id=29,
        child_agent_id=65,
        child_slot_token=1,
    ) == expected


@pytest.mark.parametrize("value", [0, math.nan, math.inf, -math.inf])
def test_float64_encoding_rejects_non_float_or_non_finite_values(value: object) -> None:
    with pytest.raises(ModularRecurrentMutationReceiptValidationError):
        float64_to_bits(value)  # type: ignore[arg-type]


def test_build_validate_and_canonicalize_ordered_receipt() -> None:
    receipt = _receipt(
        _transaction(),
        _transaction(1, child_agent_id=201, child_slot=5, child_slot_token=19),
    )
    assert receipt.schema_version == SCHEMA_VERSION
    assert receipt.contract_id == CONTRACT_ID
    assert receipt.canonical_sha256 == mutation_receipt_sha256_v1(receipt)
    wire = json.loads(canonical_modular_recurrent_mutation_receipt_bytes_v1(receipt))
    assert validate_modular_recurrent_mutation_receipt_v1(wire) == receipt
    assert [row["transaction_ordinal"] for row in wire["transactions"]] == [0, 1]


def test_canonical_digest_is_independent_of_mapping_insertion_order() -> None:
    receipt = _receipt()
    wire = receipt.to_dict()
    reversed_wire = dict(reversed(tuple(wire.items())))
    reversed_wire["transactions"] = [
        dict(reversed(tuple(wire["transactions"][0].items())))  # type: ignore[index,union-attr]
    ]
    assert canonical_modular_recurrent_mutation_receipt_bytes_v1(
        reversed_wire
    ) == canonical_modular_recurrent_mutation_receipt_bytes_v1(wire)


@pytest.mark.parametrize("scope", ["top", "transaction"])
def test_unknown_fields_fail_closed_even_with_recomputed_digest(scope: str) -> None:
    wire = _receipt().to_dict()
    if scope == "top":
        wire["unexpected"] = 1
    else:
        wire["transactions"][0]["unexpected"] = 1  # type: ignore[index]
    with pytest.raises(
        ModularRecurrentMutationReceiptValidationError, match="unknown fields"
    ):
        validate_modular_recurrent_mutation_receipt_v1(_redigest(wire))


@pytest.mark.parametrize("field", ["schema_version", "contract_id"])
def test_unknown_schema_or_contract_fails_closed(field: str) -> None:
    wire = _receipt().to_dict()
    wire[field] = "unknown_v99"
    with pytest.raises(ModularRecurrentMutationReceiptValidationError, match=field):
        validate_modular_recurrent_mutation_receipt_v1(_redigest(wire))


def test_tampering_is_detected_by_canonical_digest() -> None:
    wire = _receipt().to_dict()
    wire["transactions"][0]["reason"] = "tampered"  # type: ignore[index]
    with pytest.raises(
        ModularRecurrentMutationReceiptValidationError, match="canonical_sha256"
    ):
        validate_modular_recurrent_mutation_receipt_v1(wire)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parent_agent_id", -1),
        ("parent_slot", 1 << 31),
        ("child_slot_token", 1 << 63),
        ("attempt_ordinal", 1),
        ("selected_instance_id_index", -2),
        ("changed", 1),
    ],
)
def test_transaction_ranges_and_exact_types_fail_closed(
    field: str, value: object
) -> None:
    wire = _receipt().to_dict()
    wire["transactions"][0][field] = value  # type: ignore[index]
    with pytest.raises(ModularRecurrentMutationReceiptValidationError, match=field):
        validate_modular_recurrent_mutation_receipt_v1(_redigest(wire))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("draw0_float64_bits", "3fe333333333333"),
        ("draw0_float64_bits", "3FE3333333333333"),
        ("draw0_float64_bits", "fff8000000000000"),
        ("draw1_float64_bits", "3ff0000000000000"),
    ],
)
def test_draw_bits_require_canonical_finite_unit_float64(
    field: str, value: str
) -> None:
    wire = _receipt().to_dict()
    wire["transactions"][0][field] = value  # type: ignore[index]
    with pytest.raises(ModularRecurrentMutationReceiptValidationError, match=field):
        validate_modular_recurrent_mutation_receipt_v1(_redigest(wire))


def test_operation_must_match_draw0_bin() -> None:
    wire = _receipt().to_dict()
    wire["transactions"][0]["operation"] = "toggle_dormancy"  # type: ignore[index]
    with pytest.raises(
        ModularRecurrentMutationReceiptValidationError, match="does not match draw0"
    ):
        validate_modular_recurrent_mutation_receipt_v1(_redigest(wire))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("selected_instance_id_index", -1, "selected module"),
        ("selected_homology_id_index", -1, "selected module"),
        ("changed_dimension", "dormant", "incompatible"),
        ("parent_agent_id", 200, "identities must differ"),
    ],
)
def test_accepted_transaction_requires_coherent_target_and_identity(
    field: str, value: object, match: str
) -> None:
    wire = _receipt().to_dict()
    wire["transactions"][0][field] = value  # type: ignore[index]
    with pytest.raises(ModularRecurrentMutationReceiptValidationError, match=match):
        validate_modular_recurrent_mutation_receipt_v1(_redigest(wire))


def test_unsupported_profile_fails_closed() -> None:
    wire = _receipt().to_dict()
    wire["profile_id"] = "b2-99x99-v1"
    with pytest.raises(
        ModularRecurrentMutationReceiptValidationError, match="profile_id"
    ):
        validate_modular_recurrent_mutation_receipt_v1(_redigest(wire))


@pytest.mark.parametrize(
    ("status", "changed", "dimension"),
    [
        ("accepted", False, "none"),
        ("noop", True, "live_input_gain"),
        ("noop", False, "live_input_gain"),
        ("accepted", True, "none"),
    ],
)
def test_changed_status_and_dimension_must_be_consistent(
    status: str, changed: bool, dimension: str
) -> None:
    wire = _receipt().to_dict()
    transaction = wire["transactions"][0]  # type: ignore[index]
    transaction.update(status=status, changed=changed, changed_dimension=dimension)
    with pytest.raises(ModularRecurrentMutationReceiptValidationError):
        validate_modular_recurrent_mutation_receipt_v1(_redigest(wire))


@pytest.mark.parametrize("ordinals", [(1,), (0, 2), (1, 0)])
def test_transactions_must_have_contiguous_canonical_order(
    ordinals: tuple[int, ...],
) -> None:
    transactions = [
        _transaction(
            ordinal,
            child_agent_id=200 + index,
            child_slot=4 + index,
            child_slot_token=18 + index,
        )
        for index, ordinal in enumerate(ordinals)
    ]
    with pytest.raises(
        ModularRecurrentMutationReceiptValidationError, match="canonical contiguous"
    ):
        _receipt(*transactions)


@pytest.mark.parametrize("duplicate", ["agent", "owner"])
def test_duplicate_child_authority_fails_closed(duplicate: str) -> None:
    first = _transaction()
    second = _transaction(1, child_agent_id=201, child_slot=5, child_slot_token=19)
    if duplicate == "agent":
        second["child_agent_id"] = first["child_agent_id"]
    else:
        second["child_slot"] = first["child_slot"]
        second["child_slot_token"] = first["child_slot_token"]
    with pytest.raises(
        ModularRecurrentMutationReceiptValidationError, match="duplicate"
    ):
        _receipt(first, second)


def test_input_mapping_is_not_mutated() -> None:
    transaction = _transaction()
    original = copy.deepcopy(transaction)
    _receipt(transaction)
    assert transaction == original


def test_empty_ordered_receipt_is_valid_and_canonical() -> None:
    receipt = build_modular_recurrent_mutation_receipt_v1(
        profile_id="b2-16x48-v1",
        source_packed_sha256=SOURCE_SHA,
        ancestry_tape_sha256=ANCESTRY_SHA,
        transactions=(),
    )
    assert receipt.transactions == ()
    assert validate_modular_recurrent_mutation_receipt_v1(receipt.to_dict()) == receipt
