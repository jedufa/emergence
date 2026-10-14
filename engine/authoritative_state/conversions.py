"""Checked scalar conversions used at authoritative-state boundaries."""

from __future__ import annotations

import math
import numbers
import re
import struct
from typing import Any, Final

from .errors import (
    FLOAT32_RANGE_ERROR,
    FLOAT_FINITE_ERROR,
    FLOAT_POSITIVE_ERROR,
    INTEGER_RANGE_ERROR,
    LINEAGE_FOUNDER_RANGE_ERROR,
    TYPE_ERROR,
    UINT64_JSON_ERROR,
    fail,
)


INT32_MIN: Final[int] = -(2**31)
INT32_MAX: Final[int] = 2**31 - 1
UINT64_MAX: Final[int] = 2**64 - 1
_CANONICAL_UINT_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)")


def _exact_integral(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise fail(TYPE_ERROR, path, "must be an integer, not bool")
    return int(value)


def checked_integer(
    value: Any,
    *,
    bits: int,
    signed: bool,
    path: str,
) -> int:
    """Return an integer only when it fits the declared fixed-width type."""

    if isinstance(bits, bool) or not isinstance(bits, int) or bits < 1:
        raise ValueError("bits must be a positive integer")
    integer = _exact_integral(value, path=path)
    minimum = -(2 ** (bits - 1)) if signed else 0
    maximum = 2 ** (bits - (1 if signed else 0)) - 1
    if integer < minimum or integer > maximum:
        kind = "signed" if signed else "unsigned"
        raise fail(
            INTEGER_RANGE_ERROR,
            path,
            f"must fit {kind} int{bits} in [{minimum}, {maximum}]",
        )
    return integer


def checked_int32(value: Any, *, path: str) -> int:
    return checked_integer(value, bits=32, signed=True, path=path)


def checked_uint64(value: Any, *, path: str) -> int:
    return checked_integer(value, bits=64, signed=False, path=path)


def checked_lineage_founder_id(value: Any, *, path: str) -> int:
    """Admit the absent sentinel or a nonnegative resident int32 founder ID."""

    integer = _exact_integral(value, path=path)
    if integer < -1 or integer > INT32_MAX:
        raise fail(
            LINEAGE_FOUNDER_RANGE_ERROR,
            path,
            "must be -1 or an integer in [0, 2147483647]",
        )
    return integer


def uint64_to_json_decimal(value: Any, *, path: str) -> str:
    """Encode uint64 as canonical decimal text, avoiding JSON number loss."""

    return str(checked_uint64(value, path=path))


def uint64_from_json_decimal(value: Any, *, path: str) -> int:
    """Decode only the canonical decimal-string representation of uint64."""

    if not isinstance(value, str) or _CANONICAL_UINT_DECIMAL.fullmatch(value) is None:
        raise fail(
            UINT64_JSON_ERROR,
            path,
            "must be a canonical unsigned decimal string",
        )
    integer = int(value)
    if integer > UINT64_MAX:
        raise fail(
            UINT64_JSON_ERROR,
            path,
            f"must be in [0, {UINT64_MAX}]",
        )
    return integer


def checked_finite_float32(value: Any, *, path: str) -> float:
    """Narrow a real scalar to finite IEEE-754 binary32 without zero underflow."""

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise fail(TYPE_ERROR, path, "must be a real number, not bool")
    source = float(value)
    if not math.isfinite(source):
        raise fail(FLOAT_FINITE_ERROR, path, "must be finite")
    try:
        narrowed = struct.unpack("!f", struct.pack("!f", source))[0]
    except OverflowError as exc:
        raise fail(
            FLOAT32_RANGE_ERROR,
            path,
            "must be representable as finite float32",
        ) from exc
    if not math.isfinite(narrowed) or (source != 0.0 and narrowed == 0.0):
        raise fail(
            FLOAT32_RANGE_ERROR,
            path,
            "must be representable as finite non-underflowing float32",
        )
    return narrowed


def checked_finite_positive_float32(value: Any, *, path: str) -> float:
    """Narrow a finite positive scalar while rejecting overflow and underflow."""

    narrowed = checked_finite_float32(value, path=path)
    if narrowed <= 0.0:
        raise fail(FLOAT_POSITIVE_ERROR, path, "must be positive")
    return narrowed
