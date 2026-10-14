"""Canonical invalid-input receipts for modular recurrent mutation attempts.

This receipt binds malformed inputs that cannot enter the replay receipt ABI.
It does not replay mutations, publish state, or authorize route qualification.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import hmac
import json
import re
from typing import Any, Final


SCHEMA_VERSION: Final[str] = "eiel_modular_recurrent_invalid_input_receipt_v1"
CONTRACT_ID: Final[str] = "modular_recurrent_invalid_input_v1"
SUPPORTED_PROFILE_IDS: Final[frozenset[str]] = frozenset({"b2-12x32-v1", "b2-16x48-v1"})
TRANSACTION_KINDS: Final[frozenset[str]] = frozenset({"ordinary_birth", "turnover"})
INPUT_KINDS: Final[frozenset[str]] = frozenset(
    {"native_core_ordinary_birth", "canonical_replay_receipt"}
)
INVALID_STATUSES: Final[frozenset[str]] = frozenset({"invalid"})
OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        "toggle_dormancy",
        "prune_eligible_module",
        "refine_descendant_parameter",
        "refine_descendant_routing_or_expression",
        "unknown",
    }
)

_TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "contract_id",
    "profile_id",
    "input_kind",
    "transaction_kind",
    "parent_agent_id",
    "parent_slot",
    "parent_slot_token",
    "child_agent_id",
    "child_slot",
    "child_slot_token",
    "attempt_ordinal",
    "draw0_present",
    "draw0_float64_bits",
    "draw1_present",
    "draw1_float64_bits",
    "operation",
    "status",
    "reason",
    "changed",
    "child_state_changed",
    "native_transaction_status",
    "native_status",
    "input_digest_sha256",
    "canonical_sha256",
)
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FLOAT64_BITS = re.compile(r"[0-9a-f]{16}\Z")
_REASON = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,255}\Z")
_PROFILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_INT64_MAX: Final[int] = (1 << 63) - 1


class ModularRecurrentInvalidInputReceiptValidationError(ValueError):
    """An invalid-input receipt failed the bounded contract."""


@dataclass(frozen=True, slots=True)
class ModularRecurrentInvalidInputReceiptV1:
    schema_version: str
    contract_id: str
    profile_id: str
    input_kind: str
    transaction_kind: str
    parent_agent_id: int
    parent_slot: int
    parent_slot_token: int
    child_agent_id: int
    child_slot: int
    child_slot_token: int
    attempt_ordinal: int
    draw0_present: bool
    draw0_float64_bits: str
    draw1_present: bool
    draw1_float64_bits: str
    operation: str
    status: str
    reason: str
    changed: bool
    child_state_changed: bool
    native_transaction_status: int
    native_status: int
    input_digest_sha256: str
    canonical_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in _TOP_LEVEL_FIELDS}


def _fail(path: str, message: str) -> None:
    raise ModularRecurrentInvalidInputReceiptValidationError(f"{path} {message}")


def _object(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    keys = set(value.keys())
    if not all(type(key) is str for key in keys):
        _fail(path, "contains a non-string field name")
    expected = set(_TOP_LEVEL_FIELDS)
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


def _choice(value: object, path: str, choices: frozenset[str]) -> str:
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


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _body_dict(value: ModularRecurrentInvalidInputReceiptV1) -> dict[str, object]:
    return {
        field: getattr(value, field)
        for field in _TOP_LEVEL_FIELDS
        if field != "canonical_sha256"
    }


def invalid_input_receipt_sha256_v1(
    value: ModularRecurrentInvalidInputReceiptV1,
) -> str:
    """Compute the canonical digest over the validated invalid-input receipt."""

    return hashlib.sha256(_canonical_json_bytes(_body_dict(value))).hexdigest()


def invalid_input_digest_sha256_v1(value: object) -> str:
    """Digest a malformed input payload without admitting it to replay."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def validate_modular_recurrent_invalid_input_receipt_v1(
    value: object,
) -> ModularRecurrentInvalidInputReceiptV1:
    """Validate and normalize an invalid-input receipt."""

    row = _object(value, "$")
    receipt = ModularRecurrentInvalidInputReceiptV1(
        schema_version=_literal(
            row["schema_version"], "$.schema_version", SCHEMA_VERSION
        ),
        contract_id=_literal(row["contract_id"], "$.contract_id", CONTRACT_ID),
        profile_id=_choice(
            _matched_string(row["profile_id"], "$.profile_id", _PROFILE),
            "$.profile_id",
            SUPPORTED_PROFILE_IDS,
        ),
        input_kind=_choice(row["input_kind"], "$.input_kind", INPUT_KINDS),
        transaction_kind=_choice(
            row["transaction_kind"], "$.transaction_kind", TRANSACTION_KINDS
        ),
        parent_agent_id=_integer(
            row["parent_agent_id"], "$.parent_agent_id", 0, _INT64_MAX
        ),
        parent_slot=_integer(row["parent_slot"], "$.parent_slot", 0, _INT64_MAX),
        parent_slot_token=_integer(
            row["parent_slot_token"], "$.parent_slot_token", 0, _INT64_MAX
        ),
        child_agent_id=_integer(
            row["child_agent_id"], "$.child_agent_id", 0, _INT64_MAX
        ),
        child_slot=_integer(row["child_slot"], "$.child_slot", 0, _INT64_MAX),
        child_slot_token=_integer(
            row["child_slot_token"], "$.child_slot_token", 0, _INT64_MAX
        ),
        attempt_ordinal=_integer(
            row["attempt_ordinal"], "$.attempt_ordinal", 0, 0
        ),
        draw0_present=_boolean(row["draw0_present"], "$.draw0_present"),
        draw0_float64_bits=_matched_string(
            row["draw0_float64_bits"], "$.draw0_float64_bits", _FLOAT64_BITS
        ),
        draw1_present=_boolean(row["draw1_present"], "$.draw1_present"),
        draw1_float64_bits=_matched_string(
            row["draw1_float64_bits"], "$.draw1_float64_bits", _FLOAT64_BITS
        ),
        operation=_choice(row["operation"], "$.operation", OPERATIONS),
        status=_choice(row["status"], "$.status", INVALID_STATUSES),
        reason=_matched_string(row["reason"], "$.reason", _REASON),
        changed=_boolean(row["changed"], "$.changed"),
        child_state_changed=_boolean(row["child_state_changed"], "$.child_state_changed"),
        native_transaction_status=_integer(
            row["native_transaction_status"],
            "$.native_transaction_status",
            -1,
            _INT64_MAX,
        ),
        native_status=_integer(row["native_status"], "$.native_status", -1, _INT64_MAX),
        input_digest_sha256=_matched_string(
            row["input_digest_sha256"], "$.input_digest_sha256", _LOWER_SHA256
        ),
        canonical_sha256=_matched_string(
            row["canonical_sha256"], "$.canonical_sha256", _LOWER_SHA256
        ),
    )
    if receipt.parent_agent_id == receipt.child_agent_id:
        _fail("$", "parent and child agent identities must differ")
    if receipt.changed or receipt.child_state_changed:
        _fail("$", "invalid input receipts must not claim state change")
    if not receipt.draw0_present and receipt.draw0_float64_bits != "0000000000000000":
        _fail("$.draw0_float64_bits", "must be zero when draw0 is absent")
    if not receipt.draw1_present and receipt.draw1_float64_bits != "0000000000000000":
        _fail("$.draw1_float64_bits", "must be zero when draw1 is absent")
    expected_digest = invalid_input_receipt_sha256_v1(receipt)
    if not hmac.compare_digest(receipt.canonical_sha256, expected_digest):
        _fail("$.canonical_sha256", "does not match the canonical receipt body")
    return receipt


def build_modular_recurrent_invalid_input_receipt_v1(
    *,
    profile_id: str,
    input_kind: str,
    transaction_kind: str,
    parent_agent_id: int,
    parent_slot: int,
    parent_slot_token: int,
    child_agent_id: int,
    child_slot: int,
    child_slot_token: int,
    attempt_ordinal: int,
    draw0_float64_bits: str,
    draw1_float64_bits: str,
    operation: str,
    reason: str,
    native_transaction_status: int = -1,
    native_status: int = -1,
    input_digest_sha256: str,
    draw0_present: bool = True,
    draw1_present: bool = True,
) -> ModularRecurrentInvalidInputReceiptV1:
    """Build and validate a canonical invalid-input receipt."""

    body: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "profile_id": profile_id,
        "input_kind": input_kind,
        "transaction_kind": transaction_kind,
        "parent_agent_id": parent_agent_id,
        "parent_slot": parent_slot,
        "parent_slot_token": parent_slot_token,
        "child_agent_id": child_agent_id,
        "child_slot": child_slot,
        "child_slot_token": child_slot_token,
        "attempt_ordinal": attempt_ordinal,
        "draw0_present": draw0_present,
        "draw0_float64_bits": draw0_float64_bits,
        "draw1_present": draw1_present,
        "draw1_float64_bits": draw1_float64_bits,
        "operation": operation,
        "status": "invalid",
        "reason": reason,
        "changed": False,
        "child_state_changed": False,
        "native_transaction_status": native_transaction_status,
        "native_status": native_status,
        "input_digest_sha256": input_digest_sha256,
    }
    body["canonical_sha256"] = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
    return validate_modular_recurrent_invalid_input_receipt_v1(body)


def canonical_modular_recurrent_invalid_input_receipt_bytes_v1(
    value: ModularRecurrentInvalidInputReceiptV1,
) -> bytes:
    """Return strict canonical JSON bytes for a validated invalid-input receipt."""

    return _canonical_json_bytes(value.to_dict())


__all__ = [
    "CONTRACT_ID",
    "SCHEMA_VERSION",
    "ModularRecurrentInvalidInputReceiptV1",
    "ModularRecurrentInvalidInputReceiptValidationError",
    "build_modular_recurrent_invalid_input_receipt_v1",
    "canonical_modular_recurrent_invalid_input_receipt_bytes_v1",
    "invalid_input_digest_sha256_v1",
    "invalid_input_receipt_sha256_v1",
    "validate_modular_recurrent_invalid_input_receipt_v1",
]
