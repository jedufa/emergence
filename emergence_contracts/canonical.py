"""Deterministic canonical JSON and SHA-256 support for public contracts."""

from __future__ import annotations

import hashlib
import json
import math

from .errors import contract_invalid
from .ids import ContentDigest


def _validate_json_value(value: object, *, path: str, active: set[int]) -> None:
    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        return
    if value_type is float:
        if not math.isfinite(value):
            raise contract_invalid(
                "canonical JSON rejects NaN and infinity", field=path
            )
        return
    if value_type is dict:
        identity = id(value)
        if identity in active:
            raise contract_invalid("canonical JSON rejects cycles", field=path)
        active.add(identity)
        try:
            for key, item in value.items():
                if type(key) is not str:
                    raise contract_invalid(
                        "canonical JSON object keys must be strings",
                        field=path,
                        value_type=type(key).__name__,
                    )
                _validate_json_value(item, path=f"{path}.{key}", active=active)
        finally:
            active.remove(identity)
        return
    if value_type in {list, tuple}:
        identity = id(value)
        if identity in active:
            raise contract_invalid("canonical JSON rejects cycles", field=path)
        active.add(identity)
        try:
            for index, item in enumerate(value):
                _validate_json_value(item, path=f"{path}[{index}]", active=active)
        finally:
            active.remove(identity)
        return
    raise contract_invalid(
        "unsupported canonical JSON value",
        field=path,
        value_type=value_type.__name__,
    )


def canonical_json_bytes(value: object) -> bytes:
    """Return stable compact ASCII JSON bytes for the explicitly supported domain."""

    _validate_json_value(value, path="$", active=set())
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def canonical_json_text(value: object) -> str:
    return canonical_json_bytes(value).decode("ascii")


def canonical_sha256(value: object) -> ContentDigest:
    return ContentDigest(
        algorithm="sha256",
        hex=hashlib.sha256(canonical_json_bytes(value)).hexdigest(),
    )
