from __future__ import annotations

import json
from collections import defaultdict
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
    StateBankManifestError,
    load_accepted_state_bank_manifest,
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


def _cuda_bank() -> tuple[
    dict[str, CudaResidentStateField],
    dict[str, tuple[int, ...]],
]:
    manifest = load_accepted_state_bank_manifest()
    bank = {}
    shapes = {}
    for field in manifest.fields:
        shape = tuple(
            TEST_SHAPE_BINDINGS[symbol]
            for symbol in field.device_neutral_shape_variables
        )
        shapes[field.name] = shape
        bank[field.name] = CudaResidentStateField(
            tensor=torch.zeros(shape, dtype=field.torch_dtype, device="cuda"),
            authority_class=field.authority_class,
        )
    return bank, shapes


def test_accepted_manifest_exposes_exact_checkpoint_transport_classes() -> None:
    manifest = load_accepted_state_bank_manifest()

    assert manifest.source_path == str(DEFAULT_ACCEPTED_MANIFEST)
    assert manifest.source_sha256 == ACCEPTED_MANIFEST_SHA256
    assert len(manifest.fields) == ACCEPTED_TENSOR_FIELD_COUNT
    assert len({field.name for field in manifest.fields}) == ACCEPTED_TENSOR_FIELD_COUNT
    by_mode = defaultdict(list)
    for field in manifest.fields:
        by_mode[field.checkpoint_mode].append(field.name)
        assert field.device_neutral_shape_variables[0] == "cell"
        assert field.device_neutral_rank == len(
            field.device_neutral_shape_variables
        )

    assert len(by_mode["included_exact_tensor_clone"]) == 106
    assert len(by_mode["included_canonical_disabled_tensor"]) == 5
    assert len(by_mode["included_refreshed_derived_tensor"]) == 5
    assert len(by_mode["included_active_prefix"]) == 22
    assert by_mode["included_cpu_generator_byte_state"] == [
        "rng_manager_mutation_torch_cpu_state"
    ]
    assert len(manifest.non_tensor_field_names) == 21


def test_accepted_manifest_is_repository_bundled_with_truthful_provenance() -> None:
    manifest_path = DEFAULT_ACCEPTED_MANIFEST.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = payload["provenance"]

    assert manifest_path.parent == Path(state_bank.__file__).resolve().parent
    assert "/data/" not in str(manifest_path)
    assert provenance["mechanics_change"] is False
    assert provenance["runtime_external_dependency"] is False
    assert (
        provenance["superseded_external_manifest"]["sha256"]
        == "e09abac8fe69b0730f204e35384ed9dcee395bcadcef634f668763d9000c2051"
    )
    assert provenance["derivation"]["descriptor_backed_field_count"] == 127
    assert provenance["derivation"]["patch_field_count"] == 12


def test_accepted_manifest_fails_closed_on_digest_and_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.loads(DEFAULT_ACCEPTED_MANIFEST.read_text(encoding="utf-8"))
    payload["schema_version"] = "unexpected_state_bank_schema"
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    candidate = tmp_path / "state_manifest.json"
    candidate.write_bytes(encoded)

    with pytest.raises(StateBankManifestError, match="digest mismatch"):
        load_accepted_state_bank_manifest(candidate)

    monkeypatch.setattr(
        state_bank,
        "ACCEPTED_MANIFEST_SHA256",
        sha256(encoded).hexdigest(),
    )
    with pytest.raises(StateBankManifestError, match="schema mismatch"):
        load_accepted_state_bank_manifest(candidate)


def test_empty_bank_receipt_deterministically_identifies_all_missing_fields() -> None:
    first = validate_cuda_resident_state_bank({})
    second = validate_cuda_resident_state_bank({})
    accepted_names = tuple(
        sorted(field.name for field in load_accepted_state_bank_manifest().fields)
    )

    assert first.resident_qualified is False
    assert first.resident_field_count == 0
    assert first.missing_accepted_fields == accepted_names
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.to_dict() == second.to_dict()
    assert first.simulation_parity_claimed is False
    json.dumps(first.to_dict(), sort_keys=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_complete_bank_validates_residency_layout_authority_and_coverage() -> None:
    bank, _ = _cuda_bank()

    receipt = validate_cuda_resident_state_bank(
        bank,
        expected_shape_bindings=TEST_SHAPE_BINDINGS,
    )

    assert receipt.resident_qualified is True
    assert receipt.accepted_field_count == ACCEPTED_TENSOR_FIELD_COUNT
    assert receipt.provided_field_count == ACCEPTED_TENSOR_FIELD_COUNT
    assert receipt.present_accepted_field_count == ACCEPTED_TENSOR_FIELD_COUNT
    assert receipt.resident_field_count == ACCEPTED_TENSOR_FIELD_COUNT
    assert receipt.missing_accepted_fields == ()
    assert receipt.unexpected_fields == ()
    assert receipt.invalid_accepted_fields == ()
    assert receipt.issues == ()
    assert receipt.device is not None and receipt.device.startswith("cuda")
    assert dict(receipt.shape_bindings) == TEST_SHAPE_BINDINGS
    assert len(receipt.exact_tensor_checkpoint_fields) == 106
    assert len(receipt.canonical_disabled_checkpoint_fields) == 5
    assert len(receipt.refresh_required_checkpoint_fields) == 6
    assert len(receipt.active_prefix_checkpoint_fields) == 22
    assert receipt.blocked_checkpoint_fields == (
        "rng_manager_mutation_torch_cpu_state",
    )
    assert receipt.complete_semantic_checkpoint_transport_qualified is False
    assert receipt.simulation_parity_claimed is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_validator_rejects_dtype_shape_residency_contiguity_authority_and_aliases() -> None:
    bank, shapes = _cuda_bank()
    manifest = load_accepted_state_bank_manifest()
    contracts = manifest.fields_by_name

    energy = bank["energy"]
    bank["energy"] = CudaResidentStateField(
        torch.zeros(shapes["energy"], dtype=torch.float64, device="cuda"),
        energy.authority_class,
    )
    age = bank["age"]
    bank["age"] = CudaResidentStateField(
        torch.zeros(shapes["age"], dtype=contracts["age"].torch_dtype),
        age.authority_class,
    )
    health = bank["health"]
    health_backing = torch.zeros(
        (*shapes["health"], 2),
        dtype=contracts["health"].torch_dtype,
        device="cuda",
    )
    bank["health"] = CudaResidentStateField(
        health_backing[..., 0], health.authority_class
    )
    grid = bank["agent_grid"]
    bank["agent_grid"] = CudaResidentStateField(
        grid.tensor.flatten(), grid.authority_class
    )
    pos_x = bank["pos_x"]
    bank["pos_x"] = CudaResidentStateField(
        torch.zeros(
            (TEST_SHAPE_BINDINGS["cell"], TEST_SHAPE_BINDINGS["capacity"] + 1),
            dtype=contracts["pos_x"].torch_dtype,
            device="cuda",
        ),
        pos_x.authority_class,
    )
    alive = bank["alive_mask"]
    bank["alive_mask"] = CudaResidentStateField(
        alive.tensor, "derived_recomputed"
    )

    by_contract = defaultdict(list)
    for name, field in bank.items():
        if name in {"energy", "age", "health", "agent_grid", "alive_mask"}:
            continue
        tensor = field.tensor
        by_contract[(tensor.dtype, tuple(tensor.shape), field.authority_class)].append(
            name
        )
    pair = next(names[:2] for names in by_contract.values() if len(names) >= 2)
    bank[pair[1]] = CudaResidentStateField(
        bank[pair[0]].tensor, bank[pair[1]].authority_class
    )
    bank["not_in_the_accepted_manifest"] = CudaResidentStateField(
        torch.zeros((2, 3), device="cuda"), "immutable_snapshot_clone"
    )

    receipt = validate_cuda_resident_state_bank(
        bank,
        expected_shape_bindings=TEST_SHAPE_BINDINGS,
    )
    issue_codes = {issue.code for issue in receipt.issues}

    assert receipt.resident_qualified is False
    assert {
        "dtype_mismatch",
        "not_cuda_resident",
        "not_contiguous",
        "not_dedicated_named_bank_storage",
        "rank_mismatch",
        "shape_symbol_mismatch",
        "authority_class_mismatch",
        "aliased_named_bank_storage",
        "unexpected_field",
    } <= issue_codes
    assert set(pair) <= set(receipt.invalid_accepted_fields)
    assert receipt.unexpected_fields == ("not_in_the_accepted_manifest",)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_checkpoint_api_roundtrip_is_exact_but_semantically_bounded() -> None:
    bank, _ = _cuda_bank()
    validation = validate_cuda_resident_state_bank(
        bank,
        expected_shape_bindings=TEST_SHAPE_BINDINGS,
    )

    transport = verify_cuda_state_bank_checkpoint_transport(
        bank,
        validation_receipt=validation,
    )

    assert transport.mechanical_roundtrip_qualified is True
    assert len(transport.mechanically_roundtripped_fields) == 139
    assert transport.mismatched_fields == ()
    assert transport.device_to_host_operations == 139
    assert transport.host_to_device_operations == 139
    assert transport.device_to_host_bytes == transport.host_to_device_bytes
    assert transport.checkpoint_api_device_synchronizations == 2
    assert len(transport.exact_tensor_checkpoint_fields) == 106
    assert len(transport.canonical_value_required_checkpoint_fields) == 5
    assert len(transport.refresh_required_checkpoint_fields) == 6
    assert len(transport.active_prefix_projection_required_fields) == 22
    assert transport.blocked_semantic_checkpoint_fields == (
        "rng_manager_mutation_torch_cpu_state",
    )
    assert transport.complete_semantic_checkpoint_transport_qualified is False
    assert transport.simulation_parity_claimed is False
    json.dumps(transport.to_dict(), sort_keys=True)

    changed_bank = dict(bank)
    changed_bank.pop("energy")
    with pytest.raises(RuntimeError, match="receipt does not match"):
        verify_cuda_state_bank_checkpoint_transport(
            changed_bank,
            validation_receipt=validation,
        )
