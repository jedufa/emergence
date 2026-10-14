"""Fail-closed CPU replay for modular recurrent mutation receipts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from evolution.structural_sustainable_growth_v1 import mutate_sustainable_growth
from engine.perf.recurrent_native.modular_recurrent_mutation_receipt_v1 import (
    ModularRecurrentMutationReceiptV1,
    ModularRecurrentMutationReceiptValidationError,
    MutationTransactionReceiptV1,
    float64_from_bits,
    validate_modular_recurrent_mutation_receipt_v1,
)


class ModularRecurrentMutationReplayError(RuntimeError):
    """A cold-record mutation could not be replayed exactly."""


def _validate_identity_table(
    identity_table: Sequence[str],
    parent_payload: Mapping[str, Any],
) -> tuple[str, ...]:
    if isinstance(identity_table, (str, bytes)) or not isinstance(
        identity_table, Sequence
    ):
        raise ModularRecurrentMutationReplayError(
            "identity_table must be a sequence of identity strings"
        )
    table = tuple(identity_table)
    if any(type(value) is not str or not value for value in table):
        raise ModularRecurrentMutationReplayError(
            "identity_table entries must be nonempty strings"
        )
    if tuple(sorted(set(table))) != table:
        raise ModularRecurrentMutationReplayError(
            "identity_table must be sorted and contain no duplicates"
        )

    modules = parent_payload.get("modules", [])
    if not isinstance(modules, (list, tuple)):
        raise ModularRecurrentMutationReplayError(
            "parent payload modules must be a sequence"
        )
    table_values = set(table)
    instance_ids: set[str] = set()
    for module_index, module in enumerate(modules):
        if not isinstance(module, Mapping):
            raise ModularRecurrentMutationReplayError(
                f"parent module {module_index} must be an object"
            )
        instance_id = module.get("module_instance_id")
        homology_id = module.get("homology_id")
        for field, value in (
            ("module_instance_id", instance_id),
            ("homology_id", homology_id),
        ):
            if type(value) is not str or not value:
                raise ModularRecurrentMutationReplayError(
                    f"parent module {module_index} has an invalid {field}"
                )
            if value not in table_values:
                raise ModularRecurrentMutationReplayError(
                    f"parent module {module_index} {field} is absent from identity_table"
                )
        if instance_id in instance_ids:
            raise ModularRecurrentMutationReplayError(
                "parent module_instance_id values must be unique"
            )
        instance_ids.add(instance_id)
    return table


def _validate_selected_indexes(
    transaction: MutationTransactionReceiptV1,
    identity_table: tuple[str, ...],
) -> None:
    for field, index in (
        ("selected_instance_id_index", transaction.selected_instance_id_index),
        ("selected_homology_id_index", transaction.selected_homology_id_index),
    ):
        if index != -1 and not 0 <= index < len(identity_table):
            raise ModularRecurrentMutationReplayError(
                f"receipt {field} is outside identity_table"
            )


def _identity_index(
    identity: object,
    identity_table: tuple[str, ...],
    identity_index: Mapping[str, int],
    *,
    field: str,
) -> int:
    if type(identity) is not str:
        raise ModularRecurrentMutationReplayError(
            f"authoritative mutation returned a non-string {field}"
        )
    if not identity:
        return -1
    try:
        return identity_index[identity]
    except KeyError as exc:
        raise ModularRecurrentMutationReplayError(
            f"authoritative mutation {field} is absent from identity_table"
        ) from exc


def _outcome(
    telemetry: Mapping[str, Any],
    identity_table: tuple[str, ...],
) -> dict[str, object]:
    identity_index = {value: index for index, value in enumerate(identity_table)}
    operation = telemetry.get("operation")
    status = telemetry.get("status")
    reason = telemetry.get("reason")
    changed_dimension = telemetry.get("selected_field") or "none"
    parent_preserved = telemetry.get("parent_preserved")
    if not all(type(value) is str for value in (operation, status, reason)):
        raise ModularRecurrentMutationReplayError(
            "authoritative mutation returned malformed outcome strings"
        )
    if type(changed_dimension) is not str or type(parent_preserved) is not bool:
        raise ModularRecurrentMutationReplayError(
            "authoritative mutation returned malformed change metadata"
        )
    return {
        "operation": operation,
        "status": status,
        "reason": reason,
        "selected_instance_id_index": _identity_index(
            telemetry.get("selected_module_instance_id"),
            identity_table,
            identity_index,
            field="selected_module_instance_id",
        ),
        "selected_homology_id_index": _identity_index(
            telemetry.get("selected_homology_id"),
            identity_table,
            identity_index,
            field="selected_homology_id",
        ),
        "changed_dimension": changed_dimension,
        "changed": not parent_preserved,
    }


def replay_modular_recurrent_mutation_v1(
    parent_payload: Mapping[str, Any],
    receipt: Mapping[str, object] | ModularRecurrentMutationReceiptV1,
    *,
    identity_table: Sequence[str],
    transaction_ordinal: int = 0,
) -> dict[str, Any]:
    """Replay one receipt transaction and return an independent child payload."""

    if not isinstance(parent_payload, Mapping):
        raise ModularRecurrentMutationReplayError("parent_payload must be an object")
    try:
        to_dict = getattr(receipt, "to_dict", None)
        validated_receipt = validate_modular_recurrent_mutation_receipt_v1(
            to_dict() if callable(to_dict) else receipt
        )
    except ModularRecurrentMutationReceiptValidationError as exc:
        raise ModularRecurrentMutationReplayError(
            f"mutation receipt is invalid: {exc}"
        ) from exc
    if type(transaction_ordinal) is not int or not (
        0 <= transaction_ordinal < len(validated_receipt.transactions)
    ):
        raise ModularRecurrentMutationReplayError(
            "transaction_ordinal does not select a receipt transaction"
        )

    table = _validate_identity_table(identity_table, parent_payload)
    transaction = validated_receipt.transactions[transaction_ordinal]
    _validate_selected_indexes(transaction, table)
    draw0 = float64_from_bits(transaction.draw0_float64_bits)
    draw1 = float64_from_bits(transaction.draw1_float64_bits)

    result = mutate_sustainable_growth(
        deepcopy(dict(parent_payload)),
        draw0,
        draw1,
        enabled=transaction.status != "disabled",
    )
    actual = _outcome(result.telemetry, table)
    expected = {
        "operation": transaction.operation,
        "status": transaction.status,
        "reason": transaction.reason,
        "selected_instance_id_index": transaction.selected_instance_id_index,
        "selected_homology_id_index": transaction.selected_homology_id_index,
        "changed_dimension": transaction.changed_dimension,
        "changed": transaction.changed,
    }
    for field, expected_value in expected.items():
        actual_value = actual[field]
        if actual_value != expected_value:
            raise ModularRecurrentMutationReplayError(
                f"mutation outcome mismatch for {field}: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )
    return deepcopy(result.payload)
