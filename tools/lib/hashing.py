"""Emergence-local deterministic hashing helpers."""

from __future__ import annotations

import hashlib
from typing import Any

from tools.lib.canonical_json import canonical_json


def sha256_utf8(text: str) -> str:
    """Return the SHA-256 digest of a UTF-8 string."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha256(obj: Any) -> str:
    """Return the SHA-256 digest of canonical JSON for ``obj``."""

    return sha256_utf8(canonical_json(obj))


def job_spec_hash(job_spec: Any) -> str:
    """Hash a job spec while ignoring any placeholder self-hash field."""

    if isinstance(job_spec, dict):
        payload = dict(job_spec)
        payload["job_spec_hash"] = ""
        return canonical_sha256(payload)
    return canonical_sha256(job_spec)


__all__ = ["sha256_utf8", "canonical_sha256", "job_spec_hash"]
