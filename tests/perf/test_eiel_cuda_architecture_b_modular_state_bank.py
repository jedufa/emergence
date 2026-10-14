from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from agents.modular_genome_v1 import B2_CAPACITY_PROFILES
from perf.cuda_architecture_b_boundary import CudaBoundaryImportReceipt
from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_METADATA_KEY,
    MODULAR_STORAGE_PROJECTION_PREFIX,
    build_modular_storage_layout,
    modular_storage_projection_payload,
    pack_modular_storage,
)
from perf.cuda_architecture_b_modular_state_bank import (
    MODULAR_STATE_BANK_FIELD_COUNT,
    MODULAR_STATE_BANK_MANIFEST_SCHEMA_VERSION,
    ModularStateBankError,
    build_modular_state_bank_manifest,
    validate_modular_cpu_boundary_projection,
    validate_modular_resident_state_bank,
)
from perf.cuda_architecture_b_state_bank import (
    ACCEPTED_MANIFEST_SHA256,
    ACCEPTED_TENSOR_FIELD_COUNT,
)
from perf.cxxbackbone10_resident_state_owner import projection_digest


def _record(*, agent_id: int = 7, slot_token: int = 70) -> dict[str, object]:
    return {
        "agent_id": agent_id,
        "agent_slot_token": slot_token,
        "genome": {
            "schema_version": "modular_genome_v1",
            "genes": {},
            "modules": [],
            "connections": [],
            "b2_caps": deepcopy(B2_CAPACITY_PROFILES["b2-12x32-v1"]),
            "b2_invalid_policy": "reject",
        },
        "gated_recurrent_enabled": False,
    }


def _projection(*, agent_id: int = 7, slot_token: int = 70) -> dict[str, object]:
    packed = pack_modular_storage(
        [_record(agent_id=agent_id, slot_token=slot_token), None],
        agent_capacity=2,
    )
    payload = modular_storage_projection_payload(packed)
    return {
        "metadata": payload["metadata"],
        "tensors": {
            "alive_mask": torch.tensor([True, False]),
            "slot_to_agent_id": torch.tensor([7, -1], dtype=torch.int64),
            "slot_tokens": torch.tensor([70, 99], dtype=torch.int64),
            **payload["tensors"],
        },
    }


def test_accepted_manifest_contract_matches_repository_bundle() -> None:
    assert ACCEPTED_TENSOR_FIELD_COUNT == 139
    assert ACCEPTED_MANIFEST_SHA256 == (
        "e7981dc8d602f3a25b627881690ccdb231d82b7aaa98e4fdcd6afda6d715dfce"
    )


@pytest.mark.parametrize(
    ("profile_id", "logical_modules", "logical_connections", "expected_hash"),
    [
        (
            "b2-12x32-v1",
            12,
            32,
            "6d04b414d4d03d6fbce66498d1d13cb374997c9f61dcef6491954185651226e4",
        ),
        (
            "b2-16x48-v1",
            16,
            48,
            "71f0fb80ef21f9fd21c39794fe4fb6398b115b3fefa1569a7241315d9810c2f6",
        ),
    ],
)
def test_modular_manifest_is_exact_deterministic_and_profile_bound(
    profile_id: str,
    logical_modules: int,
    logical_connections: int,
    expected_hash: str,
) -> None:
    first = build_modular_state_bank_manifest(2, profile_id)
    second = build_modular_state_bank_manifest(2, profile_id)
    abi = build_modular_storage_layout(2, profile_id)

    assert first == second
    assert first.manifest_sha256 == expected_hash
    assert first.schema_version == MODULAR_STATE_BANK_MANIFEST_SCHEMA_VERSION
    assert first.agent_capacity == 2
    assert first.logical_module_slots == logical_modules
    assert first.logical_connection_slots == logical_connections
    assert first.physical_module_slots == 16
    assert first.physical_connection_slots == 48
    assert len(first.fields) == MODULAR_STATE_BANK_FIELD_COUNT == 57
    for contract, field in zip(first.fields, abi.fields, strict=True):
        assert contract.logical_name == field.name
        assert contract.projected_name == f"{MODULAR_STORAGE_PROJECTION_PREFIX}{field.name}"
        assert contract.to_dict()["name"] == contract.projected_name
        assert contract.dtype_name == field.dtype_name
        assert contract.shape == field.shape
        assert contract.fill_value == field.fill_value


def test_absent_overlay_is_a_noop_and_legacy_digest_is_frozen() -> None:
    metadata = {"world": "flat", "generation": 0, "tick": 0}
    tensors = {
        "alive_mask": torch.tensor([True, False]),
        "slot_to_agent_id": torch.tensor([7, -1], dtype=torch.int64),
    }
    before_metadata = deepcopy(metadata)
    before_tensors = {name: tensor.clone() for name, tensor in tensors.items()}

    assert validate_modular_cpu_boundary_projection(
        metadata,
        tensors,
        agent_capacity=2,
    ) is None
    assert metadata == before_metadata
    assert all(torch.equal(tensors[name], value) for name, value in before_tensors.items())
    assert projection_digest({"metadata": metadata, "tensors": tensors}) == (
        "7f7e8b9ca586da87d219f81cbe4d318f100e16f6e7e3f40a848b00407c0a09b2"
    )

    receipt = CudaBoundaryImportReceipt(
        schema_version="eiel_cuda_architecture_b_boundary_import_v1",
        source_schema_version="source",
        source_projection_digest="0" * 64,
        target_device="cuda:0",
        source_tensor_count=2,
        preserved_source_tensor_count=2,
        source_event_column_capacity=0,
        resident_event_column_capacity=0,
        active_event_count=0,
        patch_tensor_mode="flat_neutral_device_synthesized",
        transferred_tensor_names=(),
        added_tensor_names=(),
        host_to_device_operations=0,
        host_to_device_bytes=0,
        device_synchronizations=0,
    ).to_dict()
    assert not any("modular" in name for name in receipt)
    assert len(receipt) == 19


@pytest.mark.parametrize("damage", ["tamper", "missing", "extra", "stray", "mixed"])
def test_cpu_validator_rejects_invalid_overlay(damage: str) -> None:
    projection = _projection()
    first_name = next(
        name for name in projection["tensors"] if name.startswith(MODULAR_STORAGE_PROJECTION_PREFIX)
    )
    if damage == "tamper":
        projection["tensors"][first_name][0] = False
    elif damage == "missing":
        del projection["tensors"][first_name]
    elif damage == "extra":
        projection["tensors"][f"{MODULAR_STORAGE_PROJECTION_PREFIX}future"] = torch.tensor([0])
    elif damage == "stray":
        del projection["metadata"][MODULAR_STORAGE_METADATA_KEY]
    else:
        projection["tensors"]["phase_e_modular_v2__agent_id"] = torch.tensor([7, -1])

    with pytest.raises(ModularStateBankError, match="invalid modular CPU projection"):
        validate_modular_cpu_boundary_projection(
            projection["metadata"],
            projection["tensors"],
            agent_capacity=2,
        )


@pytest.mark.parametrize(
    ("agent_id", "slot_token", "alive", "error"),
    [
        (8, 70, True, "agent_id"),
        (7, 71, True, "agent_slot_token"),
        (7, 70, False, "occupancy"),
    ],
)
def test_cpu_validator_rejects_owner_binding_mismatch(
    agent_id: int,
    slot_token: int,
    alive: bool,
    error: str,
) -> None:
    projection = _projection(agent_id=agent_id, slot_token=slot_token)
    projection["tensors"]["alive_mask"][0] = alive

    with pytest.raises(ModularStateBankError, match=error):
        validate_modular_cpu_boundary_projection(
            projection["metadata"],
            projection["tensors"],
            agent_capacity=2,
        )


def test_cpu_device_resident_validator_covers_all_planes_without_d2h_receipt() -> None:
    projection = _projection()
    overlay = validate_modular_cpu_boundary_projection(
        projection["metadata"],
        projection["tensors"],
        agent_capacity=2,
    )
    assert overlay is not None
    resident = {
        name: tensor.clone() for name, tensor in projection["tensors"].items()
    }

    receipt = validate_modular_resident_state_bank(
        overlay,
        resident,
        device="cpu",
    )

    assert receipt.manifest_sha256 == overlay.manifest.manifest_sha256
    assert receipt.field_count == 57
    assert receipt.profile_id == "b2-12x32-v1"
    assert receipt.agent_capacity == 2
    assert receipt.device == "cpu"
    assert receipt.resident_qualified is True
    assert receipt.device_to_host_synchronizations == 0
    assert receipt.simulation_executed is False
    assert receipt.checkpoint_closure_claimed is False


def test_resident_validator_rejects_shape_dtype_field_set_and_binding_errors() -> None:
    projection = _projection()
    overlay = validate_modular_cpu_boundary_projection(
        projection["metadata"],
        projection["tensors"],
        agent_capacity=2,
    )
    assert overlay is not None

    mutations = []
    wrong_dtype = {name: tensor.clone() for name, tensor in projection["tensors"].items()}
    wrong_dtype[f"{MODULAR_STORAGE_PROJECTION_PREFIX}agent_occupied"] = torch.zeros(
        (2,), dtype=torch.int8
    )
    mutations.append(wrong_dtype)
    extra = {name: tensor.clone() for name, tensor in projection["tensors"].items()}
    extra[f"{MODULAR_STORAGE_PROJECTION_PREFIX}future"] = torch.tensor([0])
    mutations.append(extra)
    wrong_binding = {
        name: tensor.clone() for name, tensor in projection["tensors"].items()
    }
    wrong_binding["slot_tokens"][0] += 1
    mutations.append(wrong_binding)

    for resident in mutations:
        with pytest.raises(ModularStateBankError):
            validate_modular_resident_state_bank(
                overlay,
                resident,
                device="cpu",
            )
