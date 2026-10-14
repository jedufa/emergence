"""Bounded canonical receipts for modular recurrent mutation transactions.

This module validates CPU-side receipt data only. It does not replay mutations,
publish checkpoints, route production execution, or invoke CUDA.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import re
import struct
from typing import Any, Final


SCHEMA_VERSION: Final[str] = "eiel_modular_recurrent_mutation_receipt_v1"
CONTRACT_ID: Final[str] = "modular_two_draws_v1"
SUPPORTED_PROFILE_IDS: Final[frozenset[str]] = frozenset({"b2-12x32-v1", "b2-16x48-v1"})

TRANSACTION_KINDS: Final[frozenset[str]] = frozenset({"ordinary_birth", "turnover"})
OPERATIONS: Final[tuple[str, ...]] = (
    "toggle_dormancy",
    "prune_eligible_module",
    "refine_descendant_parameter",
    "refine_descendant_routing_or_expression",
)
STATUSES: Final[frozenset[str]] = frozenset(
    {"accepted", "dependency_blocked", "disabled", "ineligible", "noop", "rejected"}
)
CHANGED_DIMENSIONS: Final[frozenset[str]] = frozenset(
    {
        "none",
        "module",
        "dormant",
        "expression_scale",
        "live_retention",
        "live_input_gain",
        "live_output_gain",
        "reset_bias",
        "reset_input_gain",
        "reset_state_gain",
        "update_bias",
        "update_input_gain",
        "update_state_gain",
        "candidate_bias",
        "candidate_input_gain",
        "candidate_state_gain",
        "output_gain",
        "input_binding_gain",
        "output_binding_gain",
    }
)
_DIMENSIONS_BY_OPERATION: Final[dict[str, frozenset[str]]] = {
    "toggle_dormancy": frozenset({"dormant"}),
    "prune_eligible_module": frozenset({"module"}),
    "refine_descendant_parameter": frozenset(
        CHANGED_DIMENSIONS
        - {
            "none",
            "module",
            "dormant",
            "input_binding_gain",
            "output_binding_gain",
        }
    ),
    "refine_descendant_routing_or_expression": frozenset(
        {"expression_scale", "input_binding_gain", "output_binding_gain"}
    ),
}

_TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "contract_id",
    "profile_id",
    "source_packed_sha256",
    "ancestry_tape_sha256",
    "transactions",
    "canonical_sha256",
)
_TRANSACTION_FIELDS: Final[tuple[str, ...]] = (
    "transaction_kind",
    "transaction_ordinal",
    "parent_agent_id",
    "parent_slot",
    "parent_slot_token",
    "child_agent_id",
    "child_slot",
    "child_slot_token",
    "attempt_ordinal",
    "draw0_float64_bits",
    "draw1_float64_bits",
    "operation",
    "status",
    "reason",
    "selected_instance_id_index",
    "selected_homology_id_index",
    "changed_dimension",
    "changed",
)
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FLOAT64_BITS = re.compile(r"[0-9a-f]{16}\Z")
_REASON = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,255}\Z")
_PROFILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_INT32_MAX: Final[int] = (1 << 31) - 1
_INT64_MAX: Final[int] = (1 << 63) - 1


class ModularRecurrentMutationReceiptValidationError(ValueError):
    """A mutation receipt failed the bounded v1 contract."""


@dataclass(frozen=True, slots=True)
class MutationTransactionReceiptV1:
    transaction_kind: str
    transaction_ordinal: int
    parent_agent_id: int
    parent_slot: int
    parent_slot_token: int
    child_agent_id: int
    child_slot: int
    child_slot_token: int
    attempt_ordinal: int
    draw0_float64_bits: str
    draw1_float64_bits: str
    operation: str
    status: str
    reason: str
    selected_instance_id_index: int
    selected_homology_id_index: int
    changed_dimension: str
    changed: bool

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in _TRANSACTION_FIELDS}


@dataclass(frozen=True, slots=True)
class ModularRecurrentMutationReceiptV1:
    schema_version: str
    contract_id: str
    profile_id: str
    source_packed_sha256: str
    ancestry_tape_sha256: str
    transactions: tuple[MutationTransactionReceiptV1, ...]
    canonical_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "profile_id": self.profile_id,
            "source_packed_sha256": self.source_packed_sha256,
            "ancestry_tape_sha256": self.ancestry_tape_sha256,
            "transactions": [row.to_dict() for row in self.transactions],
            "canonical_sha256": self.canonical_sha256,
        }


def float64_to_bits(value: float) -> str:
    """Return exact IEEE-754 binary64 bits as canonical lowercase hex."""

    if type(value) is not float or not math.isfinite(value):
        raise ModularRecurrentMutationReceiptValidationError(
            "float64 value must be a finite Python float"
        )
    return struct.pack(">d", value).hex()


def float64_from_bits(bits: str) -> float:
    """Decode canonical lowercase binary64 hex without decimal conversion."""

    if type(bits) is not str or _FLOAT64_BITS.fullmatch(bits) is None:
        raise ModularRecurrentMutationReceiptValidationError(
            "float64 bits must be exactly 16 lowercase hexadecimal characters"
        )
    return struct.unpack(">d", bytes.fromhex(bits))[0]


def _fail(path: str, message: str) -> None:
    raise ModularRecurrentMutationReceiptValidationError(f"{path} {message}")


def _object(
    value: object, path: str, expected_fields: tuple[str, ...]
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    keys = set(value.keys())
    if not all(type(key) is str for key in keys):
        _fail(path, "contains a non-string field name")
    expected = set(expected_fields)
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing:
        _fail(path, f"missing required fields: {missing}")
    if unknown:
        _fail(path, f"contains unknown fields: {unknown}")
    return value


def _literal(value: object, path: str, expected: str) -> str:
    if type(value) is not str or value != expected:
        _fail(path, f"must be {expected!r}")
    return value


def _matched_string(value: object, path: str, pattern: re.Pattern[str]) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        _fail(path, "has an invalid canonical string value")
    return value


def _choice(value: object, path: str, choices: frozenset[str] | tuple[str, ...]) -> str:
    if type(value) is not str or value not in choices:
        _fail(path, f"must be one of {sorted(choices)}")
    return value


def _integer(value: object, path: str, lower: int, upper: int) -> int:
    if type(value) is not int or value < lower or value > upper:
        _fail(path, f"must be an integer in [{lower}, {upper}]")
    return value


def _boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        _fail(path, "must be a boolean")
    return value


def _operation_for_draw0(draw0: float) -> str:
    if draw0 < 0.25:
        return OPERATIONS[0]
    if draw0 < 0.5:
        return OPERATIONS[1]
    if draw0 < 0.75:
        return OPERATIONS[2]
    return OPERATIONS[3]


def _unit_draw(bits: object, path: str) -> tuple[str, float]:
    canonical_bits = _matched_string(bits, path, _FLOAT64_BITS)
    value = float64_from_bits(canonical_bits)
    if not math.isfinite(value) or value < 0.0 or value >= 1.0:
        _fail(path, "must encode a finite value in [0, 1)")
    return canonical_bits, value


def _validate_transaction(value: object, *, index: int) -> MutationTransactionReceiptV1:
    path = f"$.transactions[{index}]"
    row = _object(value, path, _TRANSACTION_FIELDS)
    draw0_bits, draw0 = _unit_draw(
        row["draw0_float64_bits"], f"{path}.draw0_float64_bits"
    )
    draw1_bits, _ = _unit_draw(row["draw1_float64_bits"], f"{path}.draw1_float64_bits")
    operation = _choice(row["operation"], f"{path}.operation", OPERATIONS)
    expected_operation = _operation_for_draw0(draw0)
    if operation != expected_operation:
        _fail(
            f"{path}.operation",
            f"does not match draw0; expected {expected_operation!r}",
        )

    changed = _boolean(row["changed"], f"{path}.changed")
    status = _choice(row["status"], f"{path}.status", STATUSES)
    dimension = _choice(
        row["changed_dimension"], f"{path}.changed_dimension", CHANGED_DIMENSIONS
    )
    if changed != (status == "accepted"):
        _fail(path, "must set changed exactly when status is 'accepted'")
    if changed == (dimension == "none"):
        _fail(path, "must use a non-'none' changed_dimension exactly when changed")

    selected_instance = _integer(
        row["selected_instance_id_index"],
        f"{path}.selected_instance_id_index",
        -1,
        _INT32_MAX,
    )
    selected_homology = _integer(
        row["selected_homology_id_index"],
        f"{path}.selected_homology_id_index",
        -1,
        _INT32_MAX,
    )
    if changed and (selected_instance < 0 or selected_homology < 0):
        _fail(path, "accepted mutation must identify its selected module")
    if changed and dimension not in _DIMENSIONS_BY_OPERATION[operation]:
        _fail(path, "changed_dimension is incompatible with operation")

    parent_agent_id = _integer(
        row["parent_agent_id"], f"{path}.parent_agent_id", 0, _INT64_MAX
    )
    child_agent_id = _integer(
        row["child_agent_id"], f"{path}.child_agent_id", 0, _INT64_MAX
    )
    if parent_agent_id == child_agent_id:
        _fail(path, "parent and child agent identities must differ")

    return MutationTransactionReceiptV1(
        transaction_kind=_choice(
            row["transaction_kind"], f"{path}.transaction_kind", TRANSACTION_KINDS
        ),
        transaction_ordinal=_integer(
            row["transaction_ordinal"], f"{path}.transaction_ordinal", 0, _INT64_MAX
        ),
        parent_agent_id=parent_agent_id,
        parent_slot=_integer(row["parent_slot"], f"{path}.parent_slot", 0, _INT32_MAX),
        parent_slot_token=_integer(
            row["parent_slot_token"], f"{path}.parent_slot_token", 0, _INT64_MAX
        ),
        child_agent_id=child_agent_id,
        child_slot=_integer(row["child_slot"], f"{path}.child_slot", 0, _INT32_MAX),
        child_slot_token=_integer(
            row["child_slot_token"], f"{path}.child_slot_token", 0, _INT64_MAX
        ),
        attempt_ordinal=_integer(
            row["attempt_ordinal"], f"{path}.attempt_ordinal", 0, 0
        ),
        draw0_float64_bits=draw0_bits,
        draw1_float64_bits=draw1_bits,
        operation=operation,
        status=status,
        reason=_matched_string(row["reason"], f"{path}.reason", _REASON),
        selected_instance_id_index=selected_instance,
        selected_homology_id_index=selected_homology,
        changed_dimension=dimension,
        changed=changed,
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _body_dict(
    *,
    profile_id: str,
    source_packed_sha256: str,
    ancestry_tape_sha256: str,
    transactions: Sequence[MutationTransactionReceiptV1],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "profile_id": profile_id,
        "source_packed_sha256": source_packed_sha256,
        "ancestry_tape_sha256": ancestry_tape_sha256,
        "transactions": [row.to_dict() for row in transactions],
    }


def mutation_receipt_sha256_v1(value: ModularRecurrentMutationReceiptV1) -> str:
    """Compute the canonical digest over the validated receipt body."""

    body = _body_dict(
        profile_id=value.profile_id,
        source_packed_sha256=value.source_packed_sha256,
        ancestry_tape_sha256=value.ancestry_tape_sha256,
        transactions=value.transactions,
    )
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def mutation_ancestry_sha256_v1(
    *,
    transaction_kind: str,
    parent_slot: int,
    child_slot: int,
    parent_agent_id: int,
    child_agent_id: int,
    child_slot_token: int,
) -> str:
    """Bind one bounded lifecycle receipt to its exact parent/child ownership."""

    if transaction_kind not in TRANSACTION_KINDS:
        raise ModularRecurrentMutationReceiptValidationError(
            "transaction_kind must be ordinary_birth or turnover"
        )
    values: dict[str, object] = {
        "parent_slot": parent_slot,
        "child_slot": child_slot,
        "parent_agent_id": parent_agent_id,
        "child_agent_id": child_agent_id,
        "child_slot_token": child_slot_token,
    }
    for name, value in values.items():
        if type(value) is not int or not 0 <= value <= _INT64_MAX:
            raise ModularRecurrentMutationReceiptValidationError(
                f"{name} must be an exact nonnegative int64 value"
            )
    payload = (
        {
            "schema_version": "eiel_modular_turnover_mutation_ancestry_v1",
            **values,
        }
        if transaction_kind == "turnover"
        else {
            "schema_version": "eiel_modular_ordinary_birth_mutation_ancestry_v1",
            "transaction_kind": transaction_kind,
            **values,
        }
    )
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def turnover_mutation_ancestry_sha256_v1(
    *,
    parent_slot: int,
    child_slot: int,
    parent_agent_id: int,
    child_agent_id: int,
    child_slot_token: int,
) -> str:
    """Compatibility wrapper for the accepted turnover ancestry contract."""

    return mutation_ancestry_sha256_v1(
        transaction_kind="turnover",
        parent_slot=parent_slot,
        child_slot=child_slot,
        parent_agent_id=parent_agent_id,
        child_agent_id=child_agent_id,
        child_slot_token=child_slot_token,
    )


def validate_modular_recurrent_mutation_receipt_v1(
    value: object,
) -> ModularRecurrentMutationReceiptV1:
    """Validate and normalize an exact bounded mutation receipt."""

    row = _object(value, "$", _TOP_LEVEL_FIELDS)
    transactions_value = row["transactions"]
    if type(transactions_value) is not list:
        _fail("$.transactions", "must be an array")
    transactions = tuple(
        _validate_transaction(transaction, index=index)
        for index, transaction in enumerate(transactions_value)
    )
    for index, transaction in enumerate(transactions):
        if transaction.transaction_ordinal != index:
            _fail(
                f"$.transactions[{index}].transaction_ordinal",
                f"must be the canonical contiguous ordinal {index}",
            )

    child_ids: set[int] = set()
    child_owners: set[tuple[int, int]] = set()
    for index, transaction in enumerate(transactions):
        if transaction.child_agent_id in child_ids:
            _fail(f"$.transactions[{index}].child_agent_id", "is duplicated")
        child_owner = (transaction.child_slot, transaction.child_slot_token)
        if child_owner in child_owners:
            _fail(f"$.transactions[{index}]", "duplicates a child slot/token owner")
        child_ids.add(transaction.child_agent_id)
        child_owners.add(child_owner)

    receipt = ModularRecurrentMutationReceiptV1(
        schema_version=_literal(
            row["schema_version"], "$.schema_version", SCHEMA_VERSION
        ),
        contract_id=_literal(row["contract_id"], "$.contract_id", CONTRACT_ID),
        profile_id=_choice(
            _matched_string(row["profile_id"], "$.profile_id", _PROFILE),
            "$.profile_id",
            SUPPORTED_PROFILE_IDS,
        ),
        source_packed_sha256=_matched_string(
            row["source_packed_sha256"], "$.source_packed_sha256", _LOWER_SHA256
        ),
        ancestry_tape_sha256=_matched_string(
            row["ancestry_tape_sha256"], "$.ancestry_tape_sha256", _LOWER_SHA256
        ),
        transactions=transactions,
        canonical_sha256=_matched_string(
            row["canonical_sha256"], "$.canonical_sha256", _LOWER_SHA256
        ),
    )
    expected_digest = mutation_receipt_sha256_v1(receipt)
    if not hmac.compare_digest(receipt.canonical_sha256, expected_digest):
        _fail("$.canonical_sha256", "does not match the canonical receipt body")
    return receipt


def build_modular_recurrent_mutation_receipt_v1(
    *,
    profile_id: str,
    source_packed_sha256: str,
    ancestry_tape_sha256: str,
    transactions: Sequence[Mapping[str, object] | MutationTransactionReceiptV1],
) -> ModularRecurrentMutationReceiptV1:
    """Build a receipt and validate it through the same strict wire contract."""

    transaction_rows = [
        transaction.to_dict()
        if isinstance(transaction, MutationTransactionReceiptV1)
        else dict(transaction)
        for transaction in transactions
    ]
    body: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "profile_id": profile_id,
        "source_packed_sha256": source_packed_sha256,
        "ancestry_tape_sha256": ancestry_tape_sha256,
        "transactions": transaction_rows,
    }
    body["canonical_sha256"] = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
    return validate_modular_recurrent_mutation_receipt_v1(body)


def canonical_modular_recurrent_mutation_receipt_bytes_v1(
    value: Mapping[str, object] | ModularRecurrentMutationReceiptV1,
) -> bytes:
    """Return canonical JSON bytes only after complete receipt validation."""

    wire = (
        value.to_dict()
        if isinstance(value, ModularRecurrentMutationReceiptV1)
        else value
    )
    receipt = validate_modular_recurrent_mutation_receipt_v1(wire)
    return _canonical_json_bytes(receipt.to_dict())
