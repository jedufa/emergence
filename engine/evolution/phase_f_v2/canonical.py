"""Small, execution-free canonical JSON helpers for Phase F v2."""

from __future__ import annotations

import hashlib
import json
from typing import Any


class PhaseFCanonicalJsonError(ValueError):
    """A value cannot be represented by the Phase F canonical JSON contract."""


def canonical_bytes(value: Any) -> bytes:
    """Return sorted, compact, finite, ASCII JSON bytes.

    This function intentionally performs no float normalization.  Phase F
    identity material must either serialize exactly or fail closed.
    """

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PhaseFCanonicalJsonError(
            "value is not finite canonical JSON data"
        ) from exc


def canonical_sha256(value: Any) -> str:
    """Return the lowercase SHA-256 digest of :func:`canonical_bytes`."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def pretty_json(value: Any) -> str:
    """Return sorted, finite, ASCII JSON text with one trailing newline."""

    try:
        return json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PhaseFCanonicalJsonError(
            "value is not finite canonical JSON data"
        ) from exc


# Descriptive aliases keep call sites explicit without creating another JSON
# contract.
canonical_json_bytes = canonical_bytes
canonical_json_sha256 = canonical_sha256


__all__ = [
    "PhaseFCanonicalJsonError",
    "canonical_bytes",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "canonical_sha256",
    "pretty_json",
]
