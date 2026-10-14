"""Exact content identity for resident inheritance checkpoint sidecars."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any, Final

import torch


INHERITANCE_SIDECAR_IDENTITY_SCHEMA_V1: Final[str] = (
    "eiel_inheritance_sidecar_identity_v1"
)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "__tensor__": True,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_canonical_value(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(
        f"unsupported inheritance sidecar payload type: {type(value).__name__}"
    )


def inheritance_sidecar_payload_sha256_v1(payload: Mapping[str, Any]) -> str:
    canonical = _canonical_value(payload)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "INHERITANCE_SIDECAR_IDENTITY_SCHEMA_V1",
    "inheritance_sidecar_payload_sha256_v1",
]
