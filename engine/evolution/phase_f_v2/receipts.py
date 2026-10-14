"""Registry-only, self-digesting receipts for the Phase F v2 matrix."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .canonical import canonical_sha256
from .matrix import (
    CASE_COUNT,
    CHECKPOINT_CASE_COUNT,
    EXPECTED_ACCELERATED_INVOCATIONS,
    EXPECTED_CPU_ORACLE_EVALUATIONS,
    EXPECTED_TOTAL_REQUIRED_EVALUATIONS,
    UNINTERRUPTED_REPEATS,
)


MATRIX_RECEIPT_SCHEMA_VERSION = "eiel_phase_f_matrix_compilation_receipt_v2"
MATRIX_RECEIPT_STATUS = "compiled_verified"
MATRIX_RECEIPT_AUTHORITY = "registry_only"
MATRIX_RECEIPT_OFFLINE_SCOPE = "self_consistency_only"
EXPECTED_LOGICAL_CASE_COUNT = CASE_COUNT
EXPECTED_CHECKPOINT_CASE_COUNT = CHECKPOINT_CASE_COUNT
EXPECTED_REPEAT_COUNT = UNINTERRUPTED_REPEATS
EXPECTED_UNINTERRUPTED_INVOCATION_COUNT = CASE_COUNT * UNINTERRUPTED_REPEATS
EXPECTED_CHECKPOINT_RESUME_INVOCATION_COUNT = (
    CHECKPOINT_CASE_COUNT * UNINTERRUPTED_REPEATS
)
EXPECTED_ACCELERATED_INVOCATION_COUNT = EXPECTED_ACCELERATED_INVOCATIONS
EXPECTED_CPU_EXPECTATION_COUNT = EXPECTED_CPU_ORACLE_EVALUATIONS
EXPECTED_TOTAL_REQUIRED_EVALUATION_COUNT = EXPECTED_TOTAL_REQUIRED_EVALUATIONS

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "authority",
        "offline_verification_scope",
        "no_launch",
        "dispatch_started",
        "execution_started",
        "source_commit",
        "preregistration_sha256",
        "phase_e_authorization_sha256",
        "matrix_spec_sha256",
        "fixture_manifest_sha256",
        "cell_registry_sha256",
        "logical_case_count",
        "checkpoint_case_count",
        "repeat_count",
        "uninterrupted_invocation_count",
        "checkpoint_resume_invocation_count",
        "accelerated_invocation_count",
        "cpu_expectation_count",
        "total_required_evaluation_count",
        "receipt_sha256",
    }
)


class PhaseFReceiptError(ValueError):
    """A Phase F receipt is malformed, over-authoritative, or inconsistent."""


def _exact_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise PhaseFReceiptError("receipt must be a string-keyed mapping")
    receipt = dict(value)
    missing = sorted(_RECEIPT_FIELDS - set(receipt))
    extra = sorted(set(receipt) - _RECEIPT_FIELDS)
    if missing or extra:
        raise PhaseFReceiptError(
            f"receipt fields are not exact; missing={missing!r} extra={extra!r}"
        )
    return receipt


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PhaseFReceiptError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _source_commit(value: Any) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise PhaseFReceiptError("source_commit must be a full lowercase Git commit")
    return value


def _exact_integer(value: Any, *, field: str, expected: int) -> int:
    if type(value) is not int or value != expected:
        raise PhaseFReceiptError(f"{field} must equal {expected}")
    return value


def create_matrix_registry_receipt(
    *,
    source_commit: str,
    preregistration_sha256: str,
    phase_e_authorization_sha256: str,
    matrix_spec_sha256: str,
    fixture_manifest_sha256: str,
    cell_registry_sha256: str,
    logical_case_count: int = EXPECTED_LOGICAL_CASE_COUNT,
    checkpoint_case_count: int = EXPECTED_CHECKPOINT_CASE_COUNT,
    repeat_count: int = EXPECTED_REPEAT_COUNT,
) -> dict[str, Any]:
    """Create a verified receipt that grants registry authority only.

    The receipt describes deterministic matrix materialization.  It cannot
    authorize, start, route, or dispatch a simulation.
    """

    checked_source = _source_commit(source_commit)
    hashes = {
        "preregistration_sha256": _sha256(
            preregistration_sha256, field="preregistration_sha256"
        ),
        "phase_e_authorization_sha256": _sha256(
            phase_e_authorization_sha256,
            field="phase_e_authorization_sha256",
        ),
        "matrix_spec_sha256": _sha256(
            matrix_spec_sha256, field="matrix_spec_sha256"
        ),
        "fixture_manifest_sha256": _sha256(
            fixture_manifest_sha256, field="fixture_manifest_sha256"
        ),
        "cell_registry_sha256": _sha256(
            cell_registry_sha256, field="cell_registry_sha256"
        ),
    }
    checked_cases = _exact_integer(
        logical_case_count,
        field="logical_case_count",
        expected=EXPECTED_LOGICAL_CASE_COUNT,
    )
    checked_repeats = _exact_integer(
        repeat_count,
        field="repeat_count",
        expected=EXPECTED_REPEAT_COUNT,
    )
    checked_checkpoint_cases = _exact_integer(
        checkpoint_case_count,
        field="checkpoint_case_count",
        expected=EXPECTED_CHECKPOINT_CASE_COUNT,
    )

    receipt: dict[str, Any] = {
        "schema_version": MATRIX_RECEIPT_SCHEMA_VERSION,
        "status": MATRIX_RECEIPT_STATUS,
        "authority": MATRIX_RECEIPT_AUTHORITY,
        "offline_verification_scope": MATRIX_RECEIPT_OFFLINE_SCOPE,
        "no_launch": True,
        "dispatch_started": False,
        "execution_started": False,
        "source_commit": checked_source,
        **hashes,
        "logical_case_count": checked_cases,
        "checkpoint_case_count": checked_checkpoint_cases,
        "repeat_count": checked_repeats,
        "uninterrupted_invocation_count": EXPECTED_UNINTERRUPTED_INVOCATION_COUNT,
        "checkpoint_resume_invocation_count": EXPECTED_CHECKPOINT_RESUME_INVOCATION_COUNT,
        "accelerated_invocation_count": EXPECTED_ACCELERATED_INVOCATION_COUNT,
        "cpu_expectation_count": EXPECTED_CPU_EXPECTATION_COUNT,
        "total_required_evaluation_count": EXPECTED_TOTAL_REQUIRED_EVALUATION_COUNT,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    validate_matrix_registry_receipt(receipt)
    return receipt


def validate_matrix_registry_receipt(value: Any) -> dict[str, Any]:
    """Return a checked copy or fail closed on any field or digest mismatch."""

    receipt = _exact_object(value)
    if receipt["schema_version"] != MATRIX_RECEIPT_SCHEMA_VERSION:
        raise PhaseFReceiptError("receipt schema_version is unsupported")
    if receipt["status"] != MATRIX_RECEIPT_STATUS:
        raise PhaseFReceiptError("receipt status is not compiled_verified")
    if receipt["authority"] != MATRIX_RECEIPT_AUTHORITY:
        raise PhaseFReceiptError("receipt authority must remain registry_only")
    if receipt["offline_verification_scope"] != MATRIX_RECEIPT_OFFLINE_SCOPE:
        raise PhaseFReceiptError("receipt offline verification scope is invalid")
    for field, expected in (
        ("no_launch", True),
        ("dispatch_started", False),
        ("execution_started", False),
    ):
        if type(receipt[field]) is not bool or receipt[field] is not expected:
            raise PhaseFReceiptError(f"receipt {field} violates registry-only authority")

    _source_commit(receipt["source_commit"])
    for field in (
        "preregistration_sha256",
        "phase_e_authorization_sha256",
        "matrix_spec_sha256",
        "fixture_manifest_sha256",
        "cell_registry_sha256",
        "receipt_sha256",
    ):
        _sha256(receipt[field], field=field)

    cases = _exact_integer(
        receipt["logical_case_count"],
        field="logical_case_count",
        expected=EXPECTED_LOGICAL_CASE_COUNT,
    )
    repeats = _exact_integer(
        receipt["repeat_count"],
        field="repeat_count",
        expected=EXPECTED_REPEAT_COUNT,
    )
    checkpoint_cases = _exact_integer(
        receipt["checkpoint_case_count"],
        field="checkpoint_case_count",
        expected=EXPECTED_CHECKPOINT_CASE_COUNT,
    )
    uninterrupted = _exact_integer(
        receipt["uninterrupted_invocation_count"],
        field="uninterrupted_invocation_count",
        expected=EXPECTED_UNINTERRUPTED_INVOCATION_COUNT,
    )
    resumed = _exact_integer(
        receipt["checkpoint_resume_invocation_count"],
        field="checkpoint_resume_invocation_count",
        expected=EXPECTED_CHECKPOINT_RESUME_INVOCATION_COUNT,
    )
    accelerated = _exact_integer(
        receipt["accelerated_invocation_count"],
        field="accelerated_invocation_count",
        expected=EXPECTED_ACCELERATED_INVOCATION_COUNT,
    )
    cpu_expectations = _exact_integer(
        receipt["cpu_expectation_count"],
        field="cpu_expectation_count",
        expected=EXPECTED_CPU_EXPECTATION_COUNT,
    )
    total = _exact_integer(
        receipt["total_required_evaluation_count"],
        field="total_required_evaluation_count",
        expected=EXPECTED_TOTAL_REQUIRED_EVALUATION_COUNT,
    )
    if uninterrupted != cases * repeats:
        raise PhaseFReceiptError("uninterrupted invocation count is inconsistent")
    if resumed != checkpoint_cases * repeats:
        raise PhaseFReceiptError("checkpoint invocation count is inconsistent")
    if accelerated != uninterrupted + resumed:
        raise PhaseFReceiptError("accelerated invocation count is inconsistent")
    if cpu_expectations != cases + checkpoint_cases:
        raise PhaseFReceiptError("CPU expectation count is inconsistent")
    if total != accelerated + cpu_expectations:
        raise PhaseFReceiptError("total evaluation count is inconsistent")

    observed = receipt["receipt_sha256"]
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256")
    if canonical_sha256(unsigned) != observed:
        raise PhaseFReceiptError("receipt self digest mismatch")
    return receipt


def verify_matrix_registry_receipt(value: Any) -> None:
    """Verify a receipt without granting execution or routing authority."""

    validate_matrix_registry_receipt(value)


# Short aliases for callers that already live inside the Phase F namespace.
create_matrix_receipt = create_matrix_registry_receipt
validate_matrix_receipt = validate_matrix_registry_receipt
verify_matrix_receipt = verify_matrix_registry_receipt


__all__ = [
    "EXPECTED_ACCELERATED_INVOCATION_COUNT",
    "EXPECTED_CHECKPOINT_CASE_COUNT",
    "EXPECTED_CHECKPOINT_RESUME_INVOCATION_COUNT",
    "EXPECTED_CPU_EXPECTATION_COUNT",
    "EXPECTED_LOGICAL_CASE_COUNT",
    "EXPECTED_REPEAT_COUNT",
    "EXPECTED_TOTAL_REQUIRED_EVALUATION_COUNT",
    "EXPECTED_UNINTERRUPTED_INVOCATION_COUNT",
    "MATRIX_RECEIPT_AUTHORITY",
    "MATRIX_RECEIPT_OFFLINE_SCOPE",
    "MATRIX_RECEIPT_SCHEMA_VERSION",
    "MATRIX_RECEIPT_STATUS",
    "PhaseFReceiptError",
    "create_matrix_receipt",
    "create_matrix_registry_receipt",
    "validate_matrix_receipt",
    "validate_matrix_registry_receipt",
    "verify_matrix_receipt",
    "verify_matrix_registry_receipt",
]
