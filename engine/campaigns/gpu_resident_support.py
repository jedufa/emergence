"""Science-agnostic integrity helpers for GPU-resident campaign providers."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch

from engine.campaigns.canonical import canonical_bytes, canonical_sha256


SUPPORT_SCHEMA_VERSION = "gpu_resident_support_v1"
ARTIFACT_BINDING_SCHEMA_VERSION = "verified_prebuilt_artifact_v1"
_SAFE_CELL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class GpuResidentSupportError(ValueError):
    """A generic GPU-resident identity or integrity contract failed."""


def _sha256(value: Any, *, name: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise GpuResidentSupportError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return digest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_safe_cell_id(cell_id: Any) -> str:
    """Return a cell ID that is safe to use as one filesystem component."""

    if (
        not isinstance(cell_id, str)
        or not _SAFE_CELL_ID.fullmatch(cell_id)
        or cell_id in {".", ".."}
        or Path(cell_id).name != cell_id
    ):
        raise GpuResidentSupportError("cell_id is not a safe path component")
    return cell_id


def safe_cell_path(
    runtime_root: str | Path,
    cell_id: Any,
    *relative_parts: str | Path,
) -> Path:
    """Resolve a per-cell path and prove it remains below ``runtime_root``."""

    root = Path(runtime_root).expanduser().resolve()
    safe_id = validate_safe_cell_id(cell_id)
    if any(Path(part).is_absolute() for part in relative_parts):
        raise GpuResidentSupportError("cell path suffix must be relative")
    cell_root = (root / "cells" / safe_id).resolve()
    candidate = cell_root.joinpath(*relative_parts).resolve()
    try:
        candidate.relative_to(cell_root)
    except ValueError as exc:
        raise GpuResidentSupportError("cell path escapes its cell root") from exc
    return candidate


def canonical_tensor_sha256(name: str, value: Any) -> str:
    """Hash one tensor from canonical metadata and contiguous CPU bytes."""

    if not isinstance(name, str) or not name:
        raise GpuResidentSupportError("tensor name must be non-empty")
    if not isinstance(value, torch.Tensor):
        raise GpuResidentSupportError(f"tensor {name!r} is not a torch.Tensor")
    try:
        tensor = value.detach().cpu().contiguous()
        byte_source = tensor.reshape(1) if tensor.ndim == 0 else tensor
        raw = byte_source.view(torch.uint8).numpy().tobytes()
    except (RuntimeError, TypeError, ValueError) as exc:
        raise GpuResidentSupportError(
            f"tensor {name!r} cannot be canonically serialized"
        ) from exc
    descriptor = canonical_bytes(
        {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
    )
    return hashlib.sha256(descriptor + b"\0" + raw).hexdigest()


def tensor_checkpoint_binding(
    payload: Mapping[str, Any],
    *,
    tensor_field: str = "tensors",
) -> dict[str, Any]:
    """Bind all canonical JSON metadata and tensors in a checkpoint payload."""

    if not isinstance(payload, Mapping):
        raise GpuResidentSupportError("checkpoint payload must be an object")
    tensors = payload.get(tensor_field)
    if not isinstance(tensors, Mapping) or not tensors:
        raise GpuResidentSupportError("checkpoint payload lacks tensor mapping")
    if any(not isinstance(name, str) or not name for name in tensors):
        raise GpuResidentSupportError("checkpoint tensor names must be non-empty strings")
    tensor_sha256 = {
        name: canonical_tensor_sha256(name, tensors[name])
        for name in sorted(tensors)
    }
    metadata = {key: value for key, value in payload.items() if key != tensor_field}
    try:
        state_sha256 = canonical_sha256(
            {
                "metadata": metadata,
                "tensor_field": tensor_field,
                "tensor_sha256": tensor_sha256,
            }
        )
    except ValueError as exc:
        raise GpuResidentSupportError(
            "checkpoint metadata is not canonical JSON"
        ) from exc
    return {
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "tensor_field": tensor_field,
        "tensor_sha256": tensor_sha256,
        "state_sha256": state_sha256,
    }


def load_verified_tensor_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_schema_version: str,
    expected_tensor_sha256: Mapping[str, str],
    expected_state_sha256: str,
    tensor_field: str = "tensors",
) -> Mapping[str, Any]:
    """Load a weights-only tensor payload and verify its complete binding."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise GpuResidentSupportError("checkpoint path is not a file")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise GpuResidentSupportError("checkpoint payload cannot be loaded") from exc
    if not isinstance(payload, Mapping):
        raise GpuResidentSupportError("checkpoint payload must be an object")
    if payload.get("schema_version") != expected_schema_version:
        raise GpuResidentSupportError("checkpoint payload schema mismatch")
    binding = tensor_checkpoint_binding(payload, tensor_field=tensor_field)
    if not isinstance(expected_tensor_sha256, Mapping):
        raise GpuResidentSupportError("expected tensor digests must be an object")
    expected_names = set(expected_tensor_sha256)
    if expected_names != set(binding["tensor_sha256"]):
        raise GpuResidentSupportError("checkpoint tensor identity mismatch")
    for name in expected_names:
        if _sha256(
            expected_tensor_sha256[name],
            name=f"expected_tensor_sha256.{name}",
        ) != binding["tensor_sha256"][name]:
            raise GpuResidentSupportError(
                f"checkpoint tensor digest mismatch: {name}"
            )
    if _sha256(
        expected_state_sha256, name="expected_state_sha256"
    ) != binding["state_sha256"]:
        raise GpuResidentSupportError("checkpoint full-state digest mismatch")
    return payload


def validate_exact_batch_identity(
    expected: Sequence[Mapping[str, Any]],
    observed: Sequence[Mapping[str, Any]],
    *,
    identity_fields: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    """Validate exact ordered batch membership and caller-selected identity fields."""

    fields = tuple(identity_fields)
    if not fields or len(set(fields)) != len(fields) or any(
        not isinstance(field, str) or not field for field in fields
    ):
        raise GpuResidentSupportError("identity_fields must be unique names")
    if len(expected) != len(observed) or not expected:
        raise GpuResidentSupportError("batch identity cardinality mismatch")

    def project(
        rows: Sequence[Mapping[str, Any]], *, side: str
    ) -> tuple[dict[str, Any], ...]:
        result: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or any(field not in row for field in fields):
                raise GpuResidentSupportError(
                    f"{side} batch identity {index} is incomplete"
                )
            result.append({field: row[field] for field in fields})
        try:
            digests = [canonical_sha256(row) for row in result]
        except ValueError as exc:
            raise GpuResidentSupportError(
                f"{side} batch identity is not canonical JSON"
            ) from exc
        if len(set(digests)) != len(digests):
            raise GpuResidentSupportError(f"{side} batch identity contains duplicates")
        return tuple(result)

    expected_projection = project(expected, side="expected")
    observed_projection = project(observed, side="observed")
    if observed_projection != expected_projection:
        raise GpuResidentSupportError("exact ordered batch identity mismatch")
    return observed_projection


def verified_prebuilt_artifact_binding(
    artifact_path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Verify a prebuilt artifact and return immutable path/hash metadata."""

    expected = _sha256(expected_sha256, name="expected_sha256")
    path = Path(artifact_path).expanduser().resolve()
    if not path.is_file():
        raise GpuResidentSupportError("prebuilt artifact path is not a file")
    actual = _file_sha256(path)
    if actual != expected:
        raise GpuResidentSupportError("prebuilt artifact SHA-256 mismatch")
    return {
        "schema_version": ARTIFACT_BINDING_SCHEMA_VERSION,
        "artifact_path": str(path),
        "sha256": actual,
        "size_bytes": path.stat().st_size,
        "verified": True,
    }


__all__ = [
    "ARTIFACT_BINDING_SCHEMA_VERSION",
    "GpuResidentSupportError",
    "SUPPORT_SCHEMA_VERSION",
    "canonical_tensor_sha256",
    "load_verified_tensor_checkpoint",
    "safe_cell_path",
    "tensor_checkpoint_binding",
    "validate_exact_batch_identity",
    "validate_safe_cell_id",
    "verified_prebuilt_artifact_binding",
]
