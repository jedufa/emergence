from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import pytest
import torch

from perf import cuda_architecture_b_state_bank as state_bank
from perf.cuda_architecture_b_state_bank import (
    ACCEPTED_MANIFEST_SHA256,
    ACCEPTED_TENSOR_FIELD_COUNT,
    CudaResidentStateField,
    DEFAULT_ACCEPTED_MANIFEST,
    DEFAULT_WDL_MANIFEST,
    StateBankManifestError,
    WDL_MANIFEST_SHA256,
    WDL_TENSOR_FIELD_COUNT,
    load_accepted_state_bank_manifest,
    load_wdl_state_bank_manifest,
    validate_cuda_resident_state_bank,
    verify_cuda_state_bank_checkpoint_transport,
)


TEST_SHAPE_BINDINGS = {
    "cell": 2,
    "capacity": 3,
    "policy_tick_stride": 2,
    "height": 2,
    "width": 3,
    "drive_channels": 4,
    "chemistry_channels": 6,
    "gru_pack_width": 4,
    "gru_hidden": 3,
    "mlp_pack_width": 5,
    "mutation_rng_state_bytes": 8,
    "event_capacity": 4,
}

WDL_FIELDS = {
    "patch_delayed_hazard_cue": "float32",
    "patch_delayed_hazard_timer": "int32",
    "patch_delayed_hazard_active": "float32",
    "patch_delayed_hazard_intensity": "float32",
    "patch_delayed_hazard_age": "int32",
    "patch_delayed_hazard_event_id": "int64",
    "patch_delayed_hazard_prior_cue": "int8",
}


def _cuda_wdl_bank() -> dict[str, CudaResidentStateField]:
    manifest = load_wdl_state_bank_manifest()
    bank = {}
    for field in manifest.fields:
        shape = tuple(
            TEST_SHAPE_BINDINGS[symbol]
            for symbol in field.device_neutral_shape_variables
        )
        tensor = torch.zeros(shape, dtype=field.torch_dtype, device="cuda")
        if field.name == "patch_delayed_hazard_timer":
            tensor.fill_(-1)
        bank[field.name] = CudaResidentStateField(
            tensor=tensor,
            authority_class=field.authority_class,
        )
    return bank


def test_wdl_manifest_is_a_strict_additive_overlay() -> None:
    flat_patch = load_accepted_state_bank_manifest()
    wdl = load_wdl_state_bank_manifest()

    assert ACCEPTED_TENSOR_FIELD_COUNT == 139
    assert WDL_TENSOR_FIELD_COUNT == 146
    assert ACCEPTED_MANIFEST_SHA256 == sha256(
        DEFAULT_ACCEPTED_MANIFEST.read_bytes()
    ).hexdigest()
    assert WDL_MANIFEST_SHA256 == sha256(DEFAULT_WDL_MANIFEST.read_bytes()).hexdigest()
    assert tuple(asdict(field) for field in wdl.fields[:139]) == tuple(
        asdict(field) for field in flat_patch.fields
    )
    assert {field.name: field.dtype_name for field in wdl.fields[-7:]} == WDL_FIELDS
    assert all(
        field.device_neutral_shape_variables == ("cell", "height", "width")
        for field in wdl.fields[-7:]
    )
    assert all(
        field.checkpoint_mode == "included_exact_tensor_clone"
        and field.authority_class == "immutable_snapshot_clone"
        for field in wdl.fields[-7:]
    )
    assert "wdl_base_seed" in wdl.non_tensor_field_names
    assert "wdl_next_absolute_step" in wdl.non_tensor_field_names
    assert "wdl_base_seed" not in flat_patch.non_tensor_field_names


def test_wdl_manifest_fails_closed_on_digest_and_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.loads(DEFAULT_WDL_MANIFEST.read_text(encoding="utf-8"))
    payload["schema_version"] = "unexpected_wdl_state_bank_schema"
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    candidate = tmp_path / "wdl_state_manifest.json"
    candidate.write_bytes(encoded)

    with pytest.raises(StateBankManifestError, match="digest mismatch"):
        load_wdl_state_bank_manifest(candidate)

    monkeypatch.setattr(state_bank, "WDL_MANIFEST_SHA256", sha256(encoded).hexdigest())
    with pytest.raises(StateBankManifestError, match="schema mismatch"):
        load_wdl_state_bank_manifest(candidate)


def test_wdl_manifest_can_drive_validation_without_changing_default_scope() -> None:
    default_receipt = validate_cuda_resident_state_bank({})
    wdl_manifest = load_wdl_state_bank_manifest()
    wdl_receipt = validate_cuda_resident_state_bank(
        {},
        accepted_manifest=wdl_manifest,
    )

    assert default_receipt.accepted_field_count == 139
    assert wdl_receipt.accepted_field_count == 146
    assert len(default_receipt.missing_accepted_fields) == 139
    assert len(wdl_receipt.missing_accepted_fields) == 146
    assert set(WDL_FIELDS) <= set(wdl_receipt.missing_accepted_fields)
    assert set(WDL_FIELDS).isdisjoint(default_receipt.missing_accepted_fields)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_wdl_bank_mechanically_roundtrips_all_146_tensors() -> None:
    manifest = load_wdl_state_bank_manifest()
    bank = _cuda_wdl_bank()

    validation = validate_cuda_resident_state_bank(
        bank,
        expected_shape_bindings=TEST_SHAPE_BINDINGS,
        accepted_manifest=manifest,
    )
    transport = verify_cuda_state_bank_checkpoint_transport(
        bank,
        validation_receipt=validation,
        expected_shape_bindings=TEST_SHAPE_BINDINGS,
        accepted_manifest=manifest,
    )

    assert validation.resident_qualified is True
    assert validation.accepted_field_count == 146
    assert len(validation.exact_tensor_checkpoint_fields) == 113
    assert transport.mechanical_roundtrip_qualified is True
    assert len(transport.mechanically_roundtripped_fields) == 146
    assert transport.mismatched_fields == ()
    assert transport.simulation_parity_claimed is False
