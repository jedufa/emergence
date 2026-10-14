"""Bridge native-core ordinary-birth receipts to canonical recurrent receipts.

This adapter is intentionally narrow. It translates the bounded native-core C
receipt shape into the existing `modular_two_draws_v1` receipt ABI, but it does
not prove structural payload parity, replay parity, CUDA parity, or route
qualification.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Final

from engine.perf.recurrent_native.modular_recurrent_invalid_input_receipt_v1 import (
    ModularRecurrentInvalidInputReceiptV1,
    build_modular_recurrent_invalid_input_receipt_v1,
    invalid_input_digest_sha256_v1,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_receipt_v1 import (
    ModularRecurrentMutationReceiptV1,
    build_modular_recurrent_mutation_receipt_v1,
)


class NativeCoreOrdinaryBirthBridgeError(ValueError):
    """A native-core receipt could not be translated without ambiguity."""


OPERATION_BY_CODE: Final[dict[int, str]] = {
    0: "toggle_dormancy",
    1: "prune_eligible_module",
    2: "refine_descendant_parameter",
    3: "refine_descendant_routing_or_expression",
}

STATUS_OK: Final[int] = 0
STATUS_NO_ELIGIBLE_MODULE: Final[int] = 9
STATUS_DEPENDENCY_BLOCKED: Final[int] = 10

TRANSACTION_COMMITTED: Final[int] = 0
TRANSACTION_CAPACITY_UNAVAILABLE: Final[int] = 1
TRANSACTION_PUBLICATION_REJECTED: Final[int] = 2
TRANSACTION_OWNER_MISMATCH: Final[int] = 3
TRANSACTION_INVALID_DRAW: Final[int] = 4
TRANSACTION_NATIVE_MUTATION_REJECTED: Final[int] = 5
TRANSACTION_MUTATION_DISABLED: Final[int] = 6

DISABLED_REASON_BY_CODE: Final[dict[int, str]] = {
    1: "operator_disabled",
    2: "structural_mutation_disabled",
    3: "recurrent_mutation_disabled",
    4: "operator_family_disabled",
    5: "zero_mutation_rate",
    6: "mutation_attempt_budget_zero",
}

DISABLED_REASON_CODE_ORDER: Final[tuple[int, ...]] = tuple(DISABLED_REASON_BY_CODE)

DISABLED_REASON_CODE_BY_REASON: Final[dict[str, int]] = {
    reason: code for code, reason in DISABLED_REASON_BY_CODE.items()
}

ORDINARY_BIRTH_TERMINAL_PRECEDENCE_V1: Final[tuple[dict[str, str], ...]] = (
    {
        "order": "0",
        "condition": "malformed_or_missing_input",
        "terminal_surface": "invalid_input_envelope",
        "reason": "invalid_input_digest_bound",
        "disabled_reason_code": "absent",
        "proposal_digest": "absent",
    },
    {
        "order": "1",
        "condition": "invalid_draw_bits",
        "terminal_surface": "rejected_receipt",
        "reason": "invalid_draw",
        "disabled_reason_code": "0",
        "proposal_digest": "0",
    },
    {
        "order": "2",
        "condition": "owner_or_slot_token_mismatch",
        "terminal_surface": "rejected_receipt",
        "reason": "stale_slot_token",
        "disabled_reason_code": "0",
        "proposal_digest": "0",
    },
    {
        "order": "3",
        "condition": "capacity_unavailable",
        "terminal_surface": "rejected_receipt",
        "reason": "capacity_unavailable",
        "disabled_reason_code": "0",
        "proposal_digest": "0",
    },
    {
        "order": "4",
        "condition": "mutation_disabled",
        "terminal_surface": "disabled_receipt",
        "reason": "canonical_disabled_reason",
        "disabled_reason_code": "1..6",
        "proposal_digest": "0",
    },
    {
        "order": "5",
        "condition": "proposal_generation",
        "terminal_surface": "candidate_proposal",
        "reason": "native_core_committed_or_ineligible",
        "disabled_reason_code": "0_or_absent",
        "proposal_digest": "candidate_digest_when_constructed",
    },
    {
        "order": "6",
        "condition": "publication_rejected",
        "terminal_surface": "rejected_receipt",
        "reason": "child_publication_failed",
        "disabled_reason_code": "0_or_absent",
        "proposal_digest": "candidate_digest",
    },
)

REASON_BY_TRANSACTION: Final[dict[int, str]] = {
    TRANSACTION_CAPACITY_UNAVAILABLE: "capacity_unavailable",
    TRANSACTION_PUBLICATION_REJECTED: "child_publication_failed",
    TRANSACTION_OWNER_MISMATCH: "stale_slot_token",
    TRANSACTION_INVALID_DRAW: "invalid_draw",
    TRANSACTION_NATIVE_MUTATION_REJECTED: "native_mutation_rejected",
}

DIMENSION_BY_OPERATION_AND_MASK: Final[dict[tuple[int, int], str]] = {
    (0, 1): "dormant",
    (1, 0): "module",
    (2, 2): "expression_scale",
    (2, 4): "live_retention",
    (2, 8): "live_input_gain",
    (2, 16): "live_output_gain",
    (2, 1 << 8): "reset_bias",
    (2, 1 << 9): "reset_input_gain",
    (2, 1 << 10): "reset_state_gain",
    (2, 1 << 11): "update_bias",
    (2, 1 << 12): "update_input_gain",
    (2, 1 << 13): "update_state_gain",
    (2, 1 << 14): "candidate_bias",
    (2, 1 << 15): "candidate_input_gain",
    (2, 1 << 16): "candidate_state_gain",
    (2, 1 << 17): "output_gain",
    (3, 1 << 5): "expression_scale",
    (3, 1 << 18): "input_binding_gain",
    (3, 1 << 19): "input_binding_gain",
    (3, 1 << 20): "output_binding_gain",
    (3, 1 << 21): "output_binding_gain",
}


def _int_field(receipt: Mapping[str, Any], field: str, *, lower: int = 0) -> int:
    value = receipt.get(field)
    if type(value) is not int or value < lower:
        raise NativeCoreOrdinaryBirthBridgeError(f"{field} must be an int >= {lower}")
    return value


def _bits_field(receipt: Mapping[str, Any], field: str) -> str:
    value = _int_field(receipt, field)
    if value > (1 << 64) - 1:
        raise NativeCoreOrdinaryBirthBridgeError(f"{field} must fit uint64")
    return f"{value:016x}"


def _operation(receipt: Mapping[str, Any]) -> tuple[int, str]:
    code = _int_field(receipt, "operation")
    try:
        return code, OPERATION_BY_CODE[code]
    except KeyError as exc:
        raise NativeCoreOrdinaryBirthBridgeError("unknown native operation") from exc


def _status_reason_changed_dimension(
    receipt: Mapping[str, Any],
    *,
    operation_code: int,
    accepted_reason: str = "native_core_committed",
) -> tuple[str, str, str, bool]:
    transaction_status = _int_field(receipt, "transaction_status")
    native_status = _int_field(receipt, "native_status")
    changed = bool(_int_field(receipt, "changed") == 1)
    changed_mask = _int_field(receipt, "changed_dimension")

    if transaction_status != TRANSACTION_COMMITTED:
        if changed:
            raise NativeCoreOrdinaryBirthBridgeError("rejected transaction changed")
        if transaction_status == TRANSACTION_MUTATION_DISABLED:
            disabled_reason_code = native_receipt_reason_code(receipt)
            try:
                return (
                    "disabled",
                    DISABLED_REASON_BY_CODE[disabled_reason_code],
                    "none",
                    False,
                )
            except KeyError as exc:
                raise NativeCoreOrdinaryBirthBridgeError(
                    "unknown native disabled reason code"
                ) from exc
        try:
            return (
                "rejected",
                REASON_BY_TRANSACTION[transaction_status],
                "none",
                False,
            )
        except KeyError as exc:
            raise NativeCoreOrdinaryBirthBridgeError(
                "unknown native transaction status"
            ) from exc

    if native_status == STATUS_NO_ELIGIBLE_MODULE:
        return "ineligible", "native_core_no_eligible_module", "none", False
    if native_status == STATUS_DEPENDENCY_BLOCKED:
        return "dependency_blocked", "native_core_dependency_blocked", "none", False
    if native_status != STATUS_OK:
        raise NativeCoreOrdinaryBirthBridgeError("committed receipt has bad native status")
    if not changed:
        return "noop", "native_core_noop", "none", False
    try:
        dimension = DIMENSION_BY_OPERATION_AND_MASK[(operation_code, changed_mask)]
    except KeyError as exc:
        raise NativeCoreOrdinaryBirthBridgeError(
            "unknown native changed_dimension"
        ) from exc
    return "accepted", accepted_reason, dimension, True


def native_receipt_reason_code(receipt: Mapping[str, Any]) -> int:
    """Return explicit native disabled reason code, defaulting legacy v3."""

    value = receipt.get("disabled_reason_code", 1)
    if type(value) is not int or value < 0:
        raise NativeCoreOrdinaryBirthBridgeError(
            "disabled_reason_code must be a nonnegative int"
        )
    return value


def _identity_table_index(identity_table: Sequence[str]) -> dict[str, int]:
    if isinstance(identity_table, (str, bytes)):
        raise NativeCoreOrdinaryBirthBridgeError("identity_table must be a sequence")
    table = tuple(identity_table)
    if any(type(value) is not str or not value for value in table):
        raise NativeCoreOrdinaryBirthBridgeError(
            "identity_table entries must be nonempty strings"
        )
    if tuple(sorted(set(table))) != table:
        raise NativeCoreOrdinaryBirthBridgeError(
            "identity_table must be sorted and unique"
        )
    return {value: index for index, value in enumerate(table)}


def _mapped_identity_index(
    *,
    native_id: int,
    native_to_python_identity: Mapping[int, str],
    identity_index: Mapping[str, int],
    field: str,
) -> int:
    if type(native_id) is not int or native_id <= 0:
        raise NativeCoreOrdinaryBirthBridgeError(f"{field} native id must be positive")
    if not isinstance(native_to_python_identity, Mapping):
        raise NativeCoreOrdinaryBirthBridgeError(f"{field} map must be a mapping")
    mapped = native_to_python_identity.get(native_id)
    if type(mapped) is not str or not mapped:
        raise NativeCoreOrdinaryBirthBridgeError(f"{field} map is missing native id")
    try:
        return identity_index[mapped]
    except KeyError as exc:
        raise NativeCoreOrdinaryBirthBridgeError(
            f"{field} mapped identity is absent from identity_table"
        ) from exc


def build_recurrent_receipt_from_native_core_ordinary_birth_v1(
    native_receipt: Mapping[str, Any],
    *,
    profile_id: str,
    source_packed_sha256: str,
    ancestry_tape_sha256: str,
    selected_instance_id_index: int,
    selected_homology_id_index: int,
    accepted_reason: str = "native_core_committed",
) -> ModularRecurrentMutationReceiptV1:
    """Translate one native-core ordinary-birth transaction into canonical ABI."""

    operation_code, operation = _operation(native_receipt)
    status, reason, changed_dimension, changed = _status_reason_changed_dimension(
        native_receipt,
        operation_code=operation_code,
        accepted_reason=accepted_reason,
    )
    if changed and (selected_instance_id_index < 0 or selected_homology_id_index < 0):
        raise NativeCoreOrdinaryBirthBridgeError(
            "accepted transaction requires selected identity indexes"
        )
    if not changed:
        selected_instance_id_index = -1
        selected_homology_id_index = -1

    return build_modular_recurrent_mutation_receipt_v1(
        profile_id=profile_id,
        source_packed_sha256=source_packed_sha256,
        ancestry_tape_sha256=ancestry_tape_sha256,
        transactions=[
            {
                "transaction_kind": "ordinary_birth",
                "transaction_ordinal": 0,
                "parent_agent_id": _int_field(native_receipt, "parent_agent_id"),
                "parent_slot": _int_field(native_receipt, "parent_slot"),
                "parent_slot_token": _int_field(native_receipt, "parent_slot_token"),
                "child_agent_id": _int_field(native_receipt, "child_agent_id"),
                "child_slot": _int_field(native_receipt, "child_slot"),
                "child_slot_token": _int_field(
                    native_receipt, "child_publication_token"
                ),
                "attempt_ordinal": 0,
                "draw0_float64_bits": _bits_field(native_receipt, "draw0_float64_bits"),
                "draw1_float64_bits": _bits_field(native_receipt, "draw1_float64_bits"),
                "operation": operation,
                "status": status,
                "reason": reason,
                "selected_instance_id_index": selected_instance_id_index,
                "selected_homology_id_index": selected_homology_id_index,
                "changed_dimension": changed_dimension,
                "changed": changed,
            }
        ],
    )


def build_invalid_input_receipt_from_native_core_ordinary_birth_v1(
    native_receipt: Mapping[str, Any],
    *,
    profile_id: str,
) -> ModularRecurrentInvalidInputReceiptV1:
    """Translate malformed native-core ordinary-birth input into invalid ABI."""

    transaction_status = _int_field(native_receipt, "transaction_status")
    if transaction_status != TRANSACTION_INVALID_DRAW:
        raise NativeCoreOrdinaryBirthBridgeError("native receipt is not invalid draw")
    if _int_field(native_receipt, "changed") != 0:
        raise NativeCoreOrdinaryBirthBridgeError("invalid input transaction changed")
    if _int_field(native_receipt, "proposal_digest") != 0:
        raise NativeCoreOrdinaryBirthBridgeError("invalid input has proposal digest")
    if _int_field(native_receipt, "child_digest_before") != _int_field(
        native_receipt, "child_digest_after"
    ):
        raise NativeCoreOrdinaryBirthBridgeError("invalid input changed child state")

    return build_modular_recurrent_invalid_input_receipt_v1(
        profile_id=profile_id,
        input_kind="native_core_ordinary_birth",
        transaction_kind="ordinary_birth",
        parent_agent_id=_int_field(native_receipt, "parent_agent_id"),
        parent_slot=_int_field(native_receipt, "parent_slot"),
        parent_slot_token=_int_field(native_receipt, "parent_slot_token"),
        child_agent_id=_int_field(native_receipt, "child_agent_id"),
        child_slot=_int_field(native_receipt, "child_slot"),
        child_slot_token=_int_field(native_receipt, "child_publication_token"),
        attempt_ordinal=0,
        draw0_float64_bits=_bits_field(native_receipt, "draw0_float64_bits"),
        draw1_float64_bits=_bits_field(native_receipt, "draw1_float64_bits"),
        operation="unknown",
        reason="invalid_draw",
        native_transaction_status=transaction_status,
        native_status=_int_field(native_receipt, "native_status"),
        input_digest_sha256=invalid_input_digest_sha256_v1(dict(native_receipt)),
    )


def build_recurrent_receipt_from_mapped_native_core_ordinary_birth_v1(
    native_receipt: Mapping[str, Any],
    *,
    profile_id: str,
    source_packed_sha256: str,
    ancestry_tape_sha256: str,
    identity_table: Sequence[str],
    native_module_instance_id_to_identity: Mapping[int, str],
    native_homology_id_to_identity: Mapping[int, str],
) -> ModularRecurrentMutationReceiptV1:
    """Translate using explicit native numeric ID to Python identity mappings."""

    operation_code, _ = _operation(native_receipt)
    _, _, _, changed = _status_reason_changed_dimension(
        native_receipt,
        operation_code=operation_code,
    )
    if changed:
        identity_index = _identity_table_index(identity_table)
        selected_instance = _mapped_identity_index(
            native_id=_int_field(native_receipt, "selected_module_instance_id"),
            native_to_python_identity=native_module_instance_id_to_identity,
            identity_index=identity_index,
            field="selected_module_instance_id",
        )
        selected_homology = _mapped_identity_index(
            native_id=_int_field(native_receipt, "selected_homology_id"),
            native_to_python_identity=native_homology_id_to_identity,
            identity_index=identity_index,
            field="selected_homology_id",
        )
    else:
        selected_instance = -1
        selected_homology = -1

    return build_recurrent_receipt_from_native_core_ordinary_birth_v1(
        native_receipt,
        profile_id=profile_id,
        source_packed_sha256=source_packed_sha256,
        ancestry_tape_sha256=ancestry_tape_sha256,
        selected_instance_id_index=selected_instance,
        selected_homology_id_index=selected_homology,
    )


def build_python_oracle_aligned_recurrent_receipt_from_mapped_native_core_ordinary_birth_v1(
    native_receipt: Mapping[str, Any],
    *,
    profile_id: str,
    source_packed_sha256: str,
    ancestry_tape_sha256: str,
    identity_table: Sequence[str],
    native_module_instance_id_to_identity: Mapping[int, str],
    native_homology_id_to_identity: Mapping[int, str],
) -> ModularRecurrentMutationReceiptV1:
    """Translate one native receipt into the Python oracle replay receipt ABI."""

    operation_code, _ = _operation(native_receipt)
    _, _, _, changed = _status_reason_changed_dimension(
        native_receipt,
        operation_code=operation_code,
        accepted_reason="proposal_valid",
    )
    if changed:
        identity_index = _identity_table_index(identity_table)
        selected_instance = _mapped_identity_index(
            native_id=_int_field(native_receipt, "selected_module_instance_id"),
            native_to_python_identity=native_module_instance_id_to_identity,
            identity_index=identity_index,
            field="selected_module_instance_id",
        )
        selected_homology = _mapped_identity_index(
            native_id=_int_field(native_receipt, "selected_homology_id"),
            native_to_python_identity=native_homology_id_to_identity,
            identity_index=identity_index,
            field="selected_homology_id",
        )
    else:
        selected_instance = -1
        selected_homology = -1

    return build_recurrent_receipt_from_native_core_ordinary_birth_v1(
        native_receipt,
        profile_id=profile_id,
        source_packed_sha256=source_packed_sha256,
        ancestry_tape_sha256=ancestry_tape_sha256,
        selected_instance_id_index=selected_instance,
        selected_homology_id_index=selected_homology,
        accepted_reason="proposal_valid",
    )


def project_python_payload_from_mapped_native_core_ordinary_birth_v1(
    parent_payload: Mapping[str, Any],
    native_receipt: Mapping[str, Any],
    *,
    identity_table: Sequence[str],
    native_module_instance_id_to_identity: Mapping[int, str],
    native_homology_id_to_identity: Mapping[int, str],
) -> dict[str, Any]:
    """Project one mapped accepted native-core scalar mutation onto a payload."""

    if not isinstance(parent_payload, Mapping):
        raise NativeCoreOrdinaryBirthBridgeError("parent_payload must be a mapping")
    operation_code, _operation_name = _operation(native_receipt)
    status, _reason, changed_dimension, changed = _status_reason_changed_dimension(
        native_receipt,
        operation_code=operation_code,
    )
    projected = deepcopy(dict(parent_payload))
    if status != "accepted" or not changed:
        return projected
    if changed_dimension not in {
        "expression_scale",
        "live_retention",
        "live_input_gain",
        "live_output_gain",
    }:
        raise NativeCoreOrdinaryBirthBridgeError(
            "payload projection only supports direct scalar integrator fields"
        )
    identity_index = _identity_table_index(identity_table)
    selected_instance_index = _mapped_identity_index(
        native_id=_int_field(native_receipt, "selected_module_instance_id"),
        native_to_python_identity=native_module_instance_id_to_identity,
        identity_index=identity_index,
        field="selected_module_instance_id",
    )
    _mapped_identity_index(
        native_id=_int_field(native_receipt, "selected_homology_id"),
        native_to_python_identity=native_homology_id_to_identity,
        identity_index=identity_index,
        field="selected_homology_id",
    )
    selected_instance = tuple(identity_table)[selected_instance_index]
    modules = projected.get("modules")
    if not isinstance(modules, list):
        raise NativeCoreOrdinaryBirthBridgeError("parent_payload modules must be a list")
    matches = [
        module
        for module in modules
        if isinstance(module, Mapping)
        and module.get("module_instance_id") == selected_instance
    ]
    if len(matches) != 1:
        raise NativeCoreOrdinaryBirthBridgeError(
            "selected mapped module must identify exactly one payload module"
        )
    module = deepcopy(dict(matches[0]))
    params = deepcopy(dict(module.get("params", {})))
    value_after = native_receipt.get("value_after_float32")
    if type(value_after) not in {float, int}:
        raise NativeCoreOrdinaryBirthBridgeError("value_after_float32 must be numeric")
    params[changed_dimension] = float(value_after)
    module["params"] = params
    projected["modules"] = [
        module if item is matches[0] else item
        for item in modules
    ]
    return projected


__all__ = [
    "NativeCoreOrdinaryBirthBridgeError",
    "build_invalid_input_receipt_from_native_core_ordinary_birth_v1",
    "build_python_oracle_aligned_recurrent_receipt_from_mapped_native_core_ordinary_birth_v1",
    "build_recurrent_receipt_from_mapped_native_core_ordinary_birth_v1",
    "build_recurrent_receipt_from_native_core_ordinary_birth_v1",
    "project_python_payload_from_mapped_native_core_ordinary_birth_v1",
]
