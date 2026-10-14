from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from engine.campaigns.gpu_resident_support import (
    ARTIFACT_BINDING_SCHEMA_VERSION,
    GpuResidentSupportError,
    load_verified_tensor_checkpoint,
    safe_cell_path,
    tensor_checkpoint_binding,
    validate_exact_batch_identity,
    validate_safe_cell_id,
    verified_prebuilt_artifact_binding,
)


def _payload(cell_id: str, value: int) -> dict:
    return {
        "schema_version": "example_gpu_state_v1",
        "identity": {"cell_id": cell_id, "generation": 5},
        "tensors": {
            "scalar": torch.tensor(value, dtype=torch.int64),
            "state": torch.arange(6, dtype=torch.float32).reshape(2, 3) + value,
        },
    }


@pytest.mark.parametrize(
    "cell_id",
    ("", ".", "..", "../cell", "cell/other", "/absolute", "cell name"),
)
def test_safe_cell_paths_reject_traversal_and_unsafe_components(
    tmp_path: Path,
    cell_id: str,
) -> None:
    with pytest.raises(GpuResidentSupportError, match="safe path component"):
        validate_safe_cell_id(cell_id)

    assert safe_cell_path(tmp_path, "cell-001", "checkpoints", "g005.pt") == (
        tmp_path / "cells" / "cell-001" / "checkpoints" / "g005.pt"
    ).resolve()
    with pytest.raises(GpuResidentSupportError, match="escapes"):
        safe_cell_path(tmp_path, "cell-001", "..", "..", "..", "escaped.pt")
    with pytest.raises(GpuResidentSupportError, match="cell root"):
        safe_cell_path(tmp_path, "cell-001", "..", "cell-002", "checkpoint.pt")


def test_weights_only_checkpoint_load_binds_metadata_and_all_tensors(
    tmp_path: Path,
) -> None:
    payload = _payload("cell-a", 7)
    binding = tensor_checkpoint_binding(payload)
    path = tmp_path / "cell-a.pt"
    torch.save(payload, path)

    loaded = load_verified_tensor_checkpoint(
        path,
        expected_schema_version="example_gpu_state_v1",
        expected_tensor_sha256=binding["tensor_sha256"],
        expected_state_sha256=binding["state_sha256"],
    )

    assert loaded["identity"] == payload["identity"]
    assert loaded["tensors"]["scalar"].item() == 7


def test_swapped_or_tampered_checkpoint_is_rejected(tmp_path: Path) -> None:
    expected = _payload("cell-a", 7)
    expected_binding = tensor_checkpoint_binding(expected)
    swapped_path = tmp_path / "swapped.pt"
    torch.save(_payload("cell-b", 7), swapped_path)

    with pytest.raises(GpuResidentSupportError, match="full-state digest mismatch"):
        load_verified_tensor_checkpoint(
            swapped_path,
            expected_schema_version="example_gpu_state_v1",
            expected_tensor_sha256=expected_binding["tensor_sha256"],
            expected_state_sha256=expected_binding["state_sha256"],
        )

    tampered = _payload("cell-a", 8)
    torch.save(tampered, swapped_path)
    with pytest.raises(GpuResidentSupportError, match="tensor digest mismatch"):
        load_verified_tensor_checkpoint(
            swapped_path,
            expected_schema_version="example_gpu_state_v1",
            expected_tensor_sha256=expected_binding["tensor_sha256"],
            expected_state_sha256=expected_binding["state_sha256"],
        )


def test_exact_batch_identity_rejects_swaps_missing_rows_and_duplicates() -> None:
    expected = [
        {"cell_id": "cell-a", "seed": 101},
        {"cell_id": "cell-b", "seed": 202},
    ]
    fields = ("cell_id", "seed")

    assert validate_exact_batch_identity(
        expected, list(expected), identity_fields=fields
    ) == tuple(expected)
    with pytest.raises(GpuResidentSupportError, match="ordered batch"):
        validate_exact_batch_identity(
            expected, list(reversed(expected)), identity_fields=fields
        )
    with pytest.raises(GpuResidentSupportError, match="cardinality"):
        validate_exact_batch_identity(expected, expected[:1], identity_fields=fields)
    with pytest.raises(GpuResidentSupportError, match="duplicates"):
        validate_exact_batch_identity(
            [expected[0], expected[0]],
            [expected[0], expected[0]],
            identity_fields=fields,
        )


def test_prebuilt_artifact_binding_requires_existing_exact_hash(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "provider.so"
    artifact.write_bytes(b"verified prebuilt")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    binding = verified_prebuilt_artifact_binding(
        artifact, expected_sha256=digest
    )
    assert binding == {
        "schema_version": ARTIFACT_BINDING_SCHEMA_VERSION,
        "artifact_path": str(artifact.resolve()),
        "sha256": digest,
        "size_bytes": len(b"verified prebuilt"),
        "verified": True,
    }
    with pytest.raises(GpuResidentSupportError, match="not a file"):
        verified_prebuilt_artifact_binding(
            tmp_path / "missing.so", expected_sha256=digest
        )
    with pytest.raises(GpuResidentSupportError, match="SHA-256 mismatch"):
        verified_prebuilt_artifact_binding(
            artifact, expected_sha256="0" * 64
        )
