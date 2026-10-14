"""Atomic, execution-free persistence for Phase F v2 JSON artifacts."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

from .canonical import canonical_bytes, pretty_json


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    """Replace ``path`` atomically using a same-directory temporary file."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return target


def atomic_write_canonical_json(path: str | Path, value: Any) -> Path:
    """Atomically write compact canonical JSON followed by one newline."""

    payload = canonical_bytes(value) + b"\n"
    return atomic_write_bytes(path, payload)


def atomic_write_pretty_json(path: str | Path, value: Any) -> Path:
    """Atomically write sorted, finite, human-readable ASCII JSON."""

    return atomic_write_bytes(path, pretty_json(value).encode("ascii"))


__all__ = [
    "atomic_write_bytes",
    "atomic_write_canonical_json",
    "atomic_write_pretty_json",
]
