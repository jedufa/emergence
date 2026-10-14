"""Data-only first-difference records for Phase F v2 checks."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .canonical import canonical_bytes, canonical_sha256


FIRST_DIFFERENCE_SCHEMA_VERSION = "eiel_phase_f_first_difference_receipt_v2"
NO_MISMATCH_SCHEMA_VERSION = "eiel_phase_f_no_mismatch_receipt_v2"
EVIDENCE_ONLY_AUTHORITY = "evidence_only"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,255}\Z")
_IDENTITY_FIELDS = frozenset(
    {
        "authority",
        "execution_authorized",
        "case_id",
        "execution_id",
        "source_commit",
        "matrix_receipt_sha256",
        "owner_id",
        "profile_id",
        "lane_id",
        "repeat_index",
        "boundary_index",
        "ordered_check_names_sha256",
    }
)
_FIRST_DIFFERENCE_FIELDS = frozenset(
    {
        "schema_version",
        "outcome",
        *_IDENTITY_FIELDS,
        "check_ordinal",
        "check_name",
        "comparison",
        "receipt_sha256",
    }
)
_NO_MISMATCH_FIELDS = frozenset(
    {
        "schema_version",
        "outcome",
        *_IDENTITY_FIELDS,
        "completed_check_count",
        "receipt_sha256",
    }
)


class PhaseFFirstDifferenceError(ValueError):
    """A first-difference input or record violates the frozen data contract."""


def _token(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _TOKEN_RE.fullmatch(value) is None:
        raise PhaseFFirstDifferenceError(f"{field} must be a stable token")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PhaseFFirstDifferenceError(
            f"{field} must be a lowercase SHA-256 digest"
        )
    return value


def _commit(value: Any) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise PhaseFFirstDifferenceError(
            "source_commit must be a full lowercase Git commit"
        )
    return value


def _index(value: Any, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise PhaseFFirstDifferenceError(f"{field} must be a nonnegative integer")
    return value


def _exact_mapping(
    value: Any, *, fields: frozenset[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise PhaseFFirstDifferenceError(f"{label} must be a string-keyed mapping")
    result = dict(value)
    missing = sorted(fields - set(result))
    extra = sorted(set(result) - fields)
    if missing or extra:
        raise PhaseFFirstDifferenceError(
            f"{label} fields are not exact; missing={missing!r} extra={extra!r}"
        )
    return result


def _canonical_scalar(value: Any, *, field: str) -> bool | int | float | str | None:
    if value is not None and type(value) not in {bool, int, float, str}:
        raise PhaseFFirstDifferenceError(
            f"{field} must be a null, boolean, integer, finite float, or string"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise PhaseFFirstDifferenceError(f"{field} must be finite")
    canonical_bytes(value)
    return value


def _comparison(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PhaseFFirstDifferenceError("comparison must be a mapping")
    kind = value.get("kind")
    if kind == "digest":
        result = _exact_mapping(
            value,
            fields=frozenset({"kind", "expected_sha256", "observed_sha256"}),
            label="comparison",
        )
        expected = _sha256(result["expected_sha256"], field="expected_sha256")
        observed = _sha256(result["observed_sha256"], field="observed_sha256")
        if expected == observed:
            raise PhaseFFirstDifferenceError(
                "a digest mismatch requires different expected and observed digests"
            )
        return {
            "kind": "digest",
            "expected_sha256": expected,
            "observed_sha256": observed,
        }
    if kind == "canonical_scalar":
        result = _exact_mapping(
            value,
            fields=frozenset({"kind", "expected", "observed"}),
            label="comparison",
        )
        expected = _canonical_scalar(result["expected"], field="expected")
        observed = _canonical_scalar(result["observed"], field="observed")
        if canonical_bytes(expected) == canonical_bytes(observed):
            raise PhaseFFirstDifferenceError(
                "a scalar mismatch requires different expected and observed values"
            )
        return {
            "kind": "canonical_scalar",
            "expected": expected,
            "observed": observed,
        }
    raise PhaseFFirstDifferenceError(
        "comparison.kind must be 'digest' or 'canonical_scalar'"
    )


def _ordered_checks(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise PhaseFFirstDifferenceError("checks must be a nonempty ordered sequence")
    checks: list[dict[str, Any]] = []
    names: set[str] = set()
    for ordinal, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise PhaseFFirstDifferenceError(f"checks[{ordinal}] must be a mapping")
        matches = raw.get("matches")
        if type(matches) is not bool:
            raise PhaseFFirstDifferenceError(
                f"checks[{ordinal}].matches must be an exact boolean"
            )
        fields = (
            frozenset({"check_name", "matches"})
            if matches
            else frozenset({"check_name", "matches", "comparison"})
        )
        row = _exact_mapping(raw, fields=fields, label=f"checks[{ordinal}]")
        name = _token(row["check_name"], field=f"checks[{ordinal}].check_name")
        if name in names:
            raise PhaseFFirstDifferenceError(f"duplicate check_name: {name!r}")
        names.add(name)
        canonical: dict[str, Any] = {"check_name": name, "matches": matches}
        if not matches:
            canonical["comparison"] = _comparison(row["comparison"])
        checks.append(canonical)
    return checks


def select_first_difference(checks: Any) -> tuple[int, dict[str, Any]] | None:
    """Select the first mismatch using the caller's frozen sequence order."""

    canonical = _ordered_checks(checks)
    for ordinal, row in enumerate(canonical):
        if row["matches"] is False:
            return ordinal, dict(row)
    return None


def _identity(
    *,
    case_id: str,
    execution_id: str,
    source_commit: str,
    matrix_receipt_sha256: str,
    owner_id: str,
    profile_id: str,
    lane_id: str,
    repeat_index: int,
    boundary_index: int,
    ordered_check_names_sha256: str,
) -> dict[str, Any]:
    return {
        "authority": EVIDENCE_ONLY_AUTHORITY,
        "execution_authorized": False,
        "case_id": _token(case_id, field="case_id"),
        "execution_id": _token(execution_id, field="execution_id"),
        "source_commit": _commit(source_commit),
        "matrix_receipt_sha256": _sha256(
            matrix_receipt_sha256, field="matrix_receipt_sha256"
        ),
        "owner_id": _token(owner_id, field="owner_id"),
        "profile_id": _token(profile_id, field="profile_id"),
        "lane_id": _token(lane_id, field="lane_id"),
        "repeat_index": _index(repeat_index, field="repeat_index"),
        "boundary_index": _index(boundary_index, field="boundary_index"),
        "ordered_check_names_sha256": _sha256(
            ordered_check_names_sha256, field="ordered_check_names_sha256"
        ),
    }


def build_first_difference_record(
    *,
    case_id: str,
    execution_id: str,
    source_commit: str,
    matrix_receipt_sha256: str,
    owner_id: str,
    profile_id: str,
    lane_id: str,
    repeat_index: int,
    boundary_index: int,
    checks: Any,
) -> dict[str, Any]:
    """Build either one first-difference receipt or a distinct no-mismatch receipt."""

    canonical_checks = _ordered_checks(checks)
    names = [row["check_name"] for row in canonical_checks]
    identity = _identity(
        case_id=case_id,
        execution_id=execution_id,
        source_commit=source_commit,
        matrix_receipt_sha256=matrix_receipt_sha256,
        owner_id=owner_id,
        profile_id=profile_id,
        lane_id=lane_id,
        repeat_index=repeat_index,
        boundary_index=boundary_index,
        ordered_check_names_sha256=canonical_sha256(names),
    )
    selected = next(
        (
            (ordinal, row)
            for ordinal, row in enumerate(canonical_checks)
            if row["matches"] is False
        ),
        None,
    )
    if selected is None:
        receipt: dict[str, Any] = {
            "schema_version": NO_MISMATCH_SCHEMA_VERSION,
            "outcome": "no_mismatch",
            **identity,
            "completed_check_count": len(canonical_checks),
        }
    else:
        ordinal, row = selected
        receipt = {
            "schema_version": FIRST_DIFFERENCE_SCHEMA_VERSION,
            "outcome": "first_difference",
            **identity,
            "check_ordinal": ordinal,
            "check_name": row["check_name"],
            "comparison": row["comparison"],
        }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return validate_first_difference_record(receipt)


def _validate_identity(receipt: Mapping[str, Any]) -> None:
    if receipt["authority"] != EVIDENCE_ONLY_AUTHORITY:
        raise PhaseFFirstDifferenceError("authority must remain evidence_only")
    if (
        type(receipt["execution_authorized"]) is not bool
        or receipt["execution_authorized"] is not False
    ):
        raise PhaseFFirstDifferenceError("execution_authorized must remain false")
    for field in ("case_id", "execution_id", "owner_id", "profile_id", "lane_id"):
        _token(receipt[field], field=field)
    _commit(receipt["source_commit"])
    _sha256(receipt["matrix_receipt_sha256"], field="matrix_receipt_sha256")
    _sha256(
        receipt["ordered_check_names_sha256"], field="ordered_check_names_sha256"
    )
    _index(receipt["repeat_index"], field="repeat_index")
    _index(receipt["boundary_index"], field="boundary_index")


def validate_first_difference_record(value: Any) -> dict[str, Any]:
    """Return a checked record or fail closed on shape, authority, or digest drift."""

    if not isinstance(value, Mapping):
        raise PhaseFFirstDifferenceError("record must be a mapping")
    schema = value.get("schema_version")
    if schema == FIRST_DIFFERENCE_SCHEMA_VERSION:
        receipt = _exact_mapping(
            value, fields=_FIRST_DIFFERENCE_FIELDS, label="first-difference record"
        )
        if receipt["outcome"] != "first_difference":
            raise PhaseFFirstDifferenceError("first-difference outcome is invalid")
        _index(receipt["check_ordinal"], field="check_ordinal")
        _token(receipt["check_name"], field="check_name")
        receipt["comparison"] = _comparison(receipt["comparison"])
    elif schema == NO_MISMATCH_SCHEMA_VERSION:
        receipt = _exact_mapping(
            value, fields=_NO_MISMATCH_FIELDS, label="no-mismatch record"
        )
        if receipt["outcome"] != "no_mismatch":
            raise PhaseFFirstDifferenceError("no-mismatch outcome is invalid")
        count = receipt["completed_check_count"]
        if type(count) is not int or count < 1:
            raise PhaseFFirstDifferenceError(
                "completed_check_count must be a positive integer"
            )
    else:
        raise PhaseFFirstDifferenceError("record schema_version is unsupported")

    _validate_identity(receipt)
    observed = _sha256(receipt["receipt_sha256"], field="receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256")
    if canonical_sha256(unsigned) != observed:
        raise PhaseFFirstDifferenceError("record self digest mismatch")
    return receipt


def verify_first_difference_record(value: Any) -> None:
    """Verify one record without granting execution or dispatch authority."""

    validate_first_difference_record(value)


__all__ = [
    "EVIDENCE_ONLY_AUTHORITY",
    "FIRST_DIFFERENCE_SCHEMA_VERSION",
    "NO_MISMATCH_SCHEMA_VERSION",
    "PhaseFFirstDifferenceError",
    "build_first_difference_record",
    "select_first_difference",
    "validate_first_difference_record",
    "verify_first_difference_record",
]
