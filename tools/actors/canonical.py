"""Canonical JSON helpers owned by the deterministic operator tooling.

The operator layer must remain independently transferable.  This module keeps
its small hashing dependency out of the scientific campaign package while
preserving the exact canonical byte contract used previously.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


# This identifier is part of the durable operator-tooling ABI.  The bytes
# produced by ``canonical_bytes`` must never change in place.  A future
# canonicalization policy gets a new function and identifier rather than
# retroactively turning intact historical material into apparent corruption.
CANONICAL_JSON_ALGORITHM = "canonical_json_ascii_sorted_compact_v1"
CANONICAL_HASH_ALGORITHM = "sha256"
CANONICAL_SHA256_KIND = f"{CANONICAL_JSON_ALGORITHM}+{CANONICAL_HASH_ALGORITHM}"


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON data") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def canonical_digest_descriptor() -> dict[str, str]:
    """Return the named algorithm contract for new digest-bearing records."""

    return {
        "canonicalization": CANONICAL_JSON_ALGORITHM,
        "hash_algorithm": CANONICAL_HASH_ALGORITHM,
        "digest_kind": CANONICAL_SHA256_KIND,
    }


def pretty_json(value: Any) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


__all__ = [
    "CANONICAL_HASH_ALGORITHM",
    "CANONICAL_JSON_ALGORITHM",
    "CANONICAL_SHA256_KIND",
    "canonical_bytes",
    "canonical_digest_descriptor",
    "canonical_sha256",
    "pretty_json",
]
