"""CPU-only checks for the source-bound C4 identity staging ABI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import engine.tensor_world.c4_native as c4_native
from engine.tensor_world.c4_native import (
    C4_IDENTITY_NATIVE_STATUS_V1,
    C4_IDENTITY_STAGING_FIELDS_V1,
    C4_IDENTITY_STAGING_SCHEMA_VERSION_V1,
    c4_identity_native_source_manifest_v1,
    validate_c4_identity_preflight_staging_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[1]


def _cpu_staging(*, batch: int = 2, capacity: int = 4, status: int = 0) -> dict[str, torch.Tensor]:
    return {
        "staged_parent_slots": torch.full((batch, capacity), -1, dtype=torch.int64),
        "staged_child_slots": torch.full((batch, capacity), -1, dtype=torch.int64),
        "staged_child_agent_ids": torch.full((batch, capacity), -1, dtype=torch.int64),
        "staged_child_energy": torch.zeros((batch, capacity), dtype=torch.float64),
        "staged_count": torch.zeros((batch,), dtype=torch.int64),
        "queue_head": torch.zeros((batch,), dtype=torch.int64),
        "queue_count": torch.zeros((batch,), dtype=torch.int64),
        "status": torch.full((batch,), status, dtype=torch.int32),
    }


def test_c4_identity_staging_manifest_and_artifact_bind_exact_abi() -> None:
    manifest = c4_identity_native_source_manifest_v1()
    assert manifest["staging_schema_version"] == C4_IDENTITY_STAGING_SCHEMA_VERSION_V1
    assert tuple(manifest["staging_fields"]) == C4_IDENTITY_STAGING_FIELDS_V1
    assert manifest["staging_device"] == "same_cuda_device_as_inputs"
    assert "fail_closed" in str(manifest["status_policy"])

    artifact = ROOT / (
        "_artifacts/tensor_world_c0_c7_v3/c4/"
        "tensor_world_C4_identity_staging_contract.json"
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["schema_version"] == C4_IDENTITY_STAGING_SCHEMA_VERSION_V1
    assert tuple(payload["fields"]) == C4_IDENTITY_STAGING_FIELDS_V1
    assert payload["fields"] == manifest["staging_fields"]
    assert payload["authoritative_inputs_untouched"] is True
    assert payload["gate_claim"] is False
    assert payload["status_codes"] == {
        str(code): name for code, name in C4_IDENTITY_NATIVE_STATUS_V1.items()
    }


def test_c4_identity_staging_rejects_cpu_storage_without_fallback() -> None:
    with pytest.raises(TensorWorldContractError, match="requires CUDA"):
        validate_c4_identity_preflight_staging_v1(
            _cpu_staging(),
            batch=2,
            capacity=4,
            device=torch.device("cuda"),
        )


def test_c4_identity_staging_rejects_unknown_fields_before_status_admission() -> None:
    outputs = _cpu_staging()
    outputs["unexpected"] = torch.zeros((2,), dtype=torch.int32)
    with pytest.raises(TensorWorldContractError, match="fields are not exact"):
        validate_c4_identity_preflight_staging_v1(
            outputs,
            batch=2,
            capacity=4,
            device=torch.device("cuda"),
        )


def test_c4_identity_staging_nonzero_status_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise status admission with CPU tensors while bypassing only the
    # device assertion. The preceding test proves that the real validator
    # rejects CPU staging; this test keeps status policy unit-testable without
    # loading or executing a CUDA extension.
    monkeypatch.setattr(c4_native, "_require_cuda_tensor", lambda *args, **kwargs: None)
    with pytest.raises(TensorWorldContractError, match="CPU fallback and partial commit"):
        validate_c4_identity_preflight_staging_v1(
            _cpu_staging(status=13),
            batch=2,
            capacity=4,
            device=torch.device("cuda"),
            require_success=True,
        )


def test_c4_identity_staging_unknown_status_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(c4_native, "_require_cuda_tensor", lambda *args, **kwargs: None)
    with pytest.raises(TensorWorldContractError, match="unknown status code"):
        validate_c4_identity_preflight_staging_v1(
            _cpu_staging(status=99),
            batch=2,
            capacity=4,
            device=torch.device("cuda"),
        )
