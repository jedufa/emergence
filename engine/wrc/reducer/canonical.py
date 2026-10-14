"""Small canonical JSON helpers local to the standalone WRC reducer.

The reducer intentionally does not import ``engine.campaigns``.  That keeps
post-run reduction usable from a transferred repository snapshot even when the
full simulation import path is not configured, and it makes the package easier
to move between repository versions.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


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


def pretty_json(value: Any) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


__all__ = ["canonical_bytes", "canonical_sha256", "pretty_json"]
