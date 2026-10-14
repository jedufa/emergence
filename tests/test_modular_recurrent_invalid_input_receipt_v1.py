from __future__ import annotations

import pytest

from engine.perf.recurrent_native.modular_recurrent_invalid_input_receipt_v1 import (
    CONTRACT_ID,
    SCHEMA_VERSION,
    ModularRecurrentInvalidInputReceiptValidationError,
    build_modular_recurrent_invalid_input_receipt_v1,
    canonical_modular_recurrent_invalid_input_receipt_bytes_v1,
    invalid_input_digest_sha256_v1,
    validate_modular_recurrent_invalid_input_receipt_v1,
)


def _receipt():
    return build_modular_recurrent_invalid_input_receipt_v1(
        profile_id="b2-16x48-v1",
        input_kind="native_core_ordinary_birth",
        transaction_kind="ordinary_birth",
        parent_agent_id=100,
        parent_slot=2,
        parent_slot_token=12,
        child_agent_id=(1 << 40) + 7,
        child_slot=4,
        child_slot_token=18,
        attempt_ordinal=0,
        draw0_float64_bits="3ff0000000000000",
        draw1_float64_bits="3fe0000000000000",
        operation="unknown",
        reason="invalid_draw",
        native_transaction_status=4,
        native_status=7,
        input_digest_sha256=invalid_input_digest_sha256_v1(
            {"draw0_float64_bits": "3ff0000000000000"}
        ),
    )


def _canonical_replay_receipt(
    *,
    reason: str,
    operation: str = "unknown",
    draw0_present: bool = True,
    draw0_float64_bits: str = "3fe0000000000000",
    draw1_present: bool = True,
    draw1_float64_bits: str = "3fd0000000000000",
    payload: object | None = None,
):
    if payload is None:
        payload = {
            "schema_version": "malformed_modular_two_draws_v1",
            "reason": reason,
            "draws": [
                draw0_float64_bits if draw0_present else None,
                draw1_float64_bits if draw1_present else None,
            ],
        }
    return build_modular_recurrent_invalid_input_receipt_v1(
        profile_id="b2-16x48-v1",
        input_kind="canonical_replay_receipt",
        transaction_kind="ordinary_birth",
        parent_agent_id=100,
        parent_slot=2,
        parent_slot_token=12,
        child_agent_id=(1 << 40) + 7,
        child_slot=4,
        child_slot_token=18,
        attempt_ordinal=0,
        draw0_present=draw0_present,
        draw0_float64_bits=draw0_float64_bits if draw0_present else "0000000000000000",
        draw1_present=draw1_present,
        draw1_float64_bits=draw1_float64_bits if draw1_present else "0000000000000000",
        operation=operation,
        reason=reason,
        input_digest_sha256=invalid_input_digest_sha256_v1(payload),
    )


def test_invalid_input_receipt_is_canonical_and_bounded() -> None:
    receipt = _receipt()
    payload = receipt.to_dict()

    assert receipt.schema_version == SCHEMA_VERSION
    assert receipt.contract_id == CONTRACT_ID
    assert receipt.status == "invalid"
    assert receipt.reason == "invalid_draw"
    assert receipt.changed is False
    assert receipt.child_state_changed is False
    assert receipt.canonical_sha256
    assert validate_modular_recurrent_invalid_input_receipt_v1(payload) == receipt
    assert canonical_modular_recurrent_invalid_input_receipt_bytes_v1(receipt)


def test_invalid_input_receipt_rejects_state_change_claims() -> None:
    payload = _receipt().to_dict()
    payload["changed"] = True
    payload["canonical_sha256"] = "0" * 64

    with pytest.raises(
        ModularRecurrentInvalidInputReceiptValidationError,
        match="must not claim state change",
    ):
        validate_modular_recurrent_invalid_input_receipt_v1(payload)


def test_invalid_input_receipt_rejects_unknown_fields() -> None:
    payload = _receipt().to_dict()
    payload["unexpected"] = "field"

    with pytest.raises(
        ModularRecurrentInvalidInputReceiptValidationError,
        match="unknown fields",
    ):
        validate_modular_recurrent_invalid_input_receipt_v1(payload)


def test_invalid_input_receipt_records_absent_draws_without_fake_bits() -> None:
    receipt = _canonical_replay_receipt(
        reason="missing_draw",
        draw1_present=False,
        payload={
            "schema_version": "malformed_modular_two_draws_v1",
            "draw0_float64_bits": "3fe0000000000000",
        },
    )
    payload = receipt.to_dict()

    assert receipt.draw0_present is True
    assert receipt.draw1_present is False
    assert receipt.draw1_float64_bits == "0000000000000000"
    assert validate_modular_recurrent_invalid_input_receipt_v1(payload) == receipt


def test_invalid_input_receipt_rejects_absent_draw_nonzero_bits() -> None:
    payload = _canonical_replay_receipt(reason="missing_draw", draw1_present=False).to_dict()
    payload["draw1_float64_bits"] = "3fe0000000000000"
    payload["canonical_sha256"] = "0" * 64

    with pytest.raises(
        ModularRecurrentInvalidInputReceiptValidationError,
        match="must be zero when draw1 is absent",
    ):
        validate_modular_recurrent_invalid_input_receipt_v1(payload)


@pytest.mark.parametrize(
    ("case_id", "operation", "payload"),
    [
        (
            "extra_draw",
            "unknown",
            {
                "schema_version": "malformed_modular_two_draws_v1",
                "draws": [
                    "3fe0000000000000",
                    "3fd0000000000000",
                    "3fc0000000000000",
                ],
            },
        ),
        (
            "operation_draw_mismatch",
            "prune_eligible_module",
            {
                "schema_version": "modular_two_draws_v1",
                "operation": "prune_eligible_module",
                "draw0_float64_bits": "3ff0000000000000",
                "draw1_float64_bits": "3fd0000000000000",
            },
        ),
        (
            "selected_index_out_of_range",
            "refine_descendant_parameter",
            {
                "schema_version": "modular_two_draws_v1",
                "selected_instance_id_index": 99,
                "identity_table_size": 2,
            },
        ),
        (
            "malformed_replay_receipt",
            "unknown",
            {"schema_version": "modular_two_draws_v1", "transactions": "not-a-list"},
        ),
        (
            "changed_replay_digest",
            "refine_descendant_parameter",
            {
                "schema_version": "modular_two_draws_v1",
                "canonical_sha256": "0" * 64,
                "body_changed_after_digest": True,
            },
        ),
        (
            "duplicate_replay_field",
            "unknown",
            {
                "raw_json_pairs": [
                    ["schema_version", "modular_two_draws_v1"],
                    ["schema_version", "modular_two_draws_v1"],
                ],
            },
        ),
        (
            "unsupported_operator_id",
            "unknown",
            {"schema_version": "modular_two_draws_v1", "operation": "unsupported"},
        ),
    ],
)
def test_invalid_input_receipt_binds_malformed_replay_cases(
    case_id: str, operation: str, payload: object
) -> None:
    receipt = _canonical_replay_receipt(
        reason=case_id,
        operation=operation,
        payload=payload,
    )

    assert receipt.input_kind == "canonical_replay_receipt"
    assert receipt.reason == case_id
    assert receipt.input_digest_sha256 == invalid_input_digest_sha256_v1(payload)
    assert receipt.native_transaction_status == -1
    assert receipt.native_status == -1
    assert validate_modular_recurrent_invalid_input_receipt_v1(receipt.to_dict()) == receipt
