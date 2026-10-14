from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from agents.modular_genome_v1 import B2_CAPACITY_PROFILES
from perf.cuda_architecture_b_modular_capacity_epoch import (
    MODULAR_CAPACITY_EPOCH_AUTHORIZATION_SCHEMA_VERSION,
    MODULAR_CAPACITY_EPOCH_RECEIPT_SCHEMA_VERSION,
    ModularCapacityEpochError,
    capacity_epoch_authorization_receipt_sha256,
    migrate_modular_capacity_epoch_projection,
    validate_legacy_unused_tail_empty,
    validate_modular_capacity_epoch_transition,
)
from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_METADATA_KEY,
    MODULAR_STORAGE_PROJECTION_PREFIX,
    modular_storage_from_projection,
    modular_storage_projection_payload,
    pack_modular_storage,
)
from perf.eiel_phase_e_modular_tick_fixture_v1 import (
    _connection as _fixture_connection,
    _development as _fixture_development,
    _module as _fixture_module,
)
from simulation.capacity_epoch import SAFE_BOUNDARY_KIND


def _record(profile_id: str = "b2-12x32-v1") -> dict:
    first = _fixture_module(
        0,
        0,
        "capacity-epoch-active",
        input_semantic="self_energy",
        input_gain=0.5,
        output_semantic="rest",
        output_gain=0.75,
        development=_fixture_development((0.25, 0.5, 1.0), latch=True),
    )
    second = _fixture_module(
        0,
        1,
        "capacity-epoch-dormant",
        dormant=True,
    )
    first_id = str(first["module_instance_id"])
    second_id = str(second["module_instance_id"])
    return {
        "agent_id": 7,
        "agent_slot_token": 70,
        "genome": {
            "schema_version": "modular_genome_v1",
            "genes": {"escap8_growth_last_reason": "capacity-epoch-fixture"},
            "modules": [first, second],
            "connections": [_fixture_connection(0, first, second)],
            "b2_caps": deepcopy(B2_CAPACITY_PROFILES[profile_id]),
            "b2_invalid_policy": "reject",
        },
        "structural_live_module_state": {first_id: 0.375, second_id: -0.25},
        "structural_live_module_outputs": {first_id: -0.125, second_id: 0.0},
        "structural_development_latch_by_instance_id": {first_id: True},
        "structural_development_effective_gain_by_instance_id": {first_id: 0.5},
        "gated_recurrent_enabled": False,
        "saturated_or_rejected_activity_count": 3,
    }


def _projection(profile_id: str = "b2-12x32-v1") -> tuple[dict, dict]:
    packed = pack_modular_storage(
        [_record(profile_id), None],
        agent_capacity=2,
        profile_id=profile_id,
    )
    payload = modular_storage_projection_payload(packed)
    return (
        {"world": "flat", "generation": 4, **payload["metadata"]},
        {
            "alive_mask": torch.tensor([True, False], dtype=torch.bool),
            "slot_to_agent_id": torch.tensor([7, -1], dtype=torch.int64),
            "slot_tokens": torch.tensor([70, 99], dtype=torch.int64),
            **payload["tensors"],
        },
    )


def _authorization(**overrides) -> dict:
    payload = {
        "schema_version": MODULAR_CAPACITY_EPOCH_AUTHORIZATION_SCHEMA_VERSION,
        "authorization_receipt_id": "phase-e-capacity-epoch-auth-0004",
        "authorization_receipt_sha256": "",
        "authorized": True,
        "authorized_generation": 4,
        "next_generation": 5,
        "source_profile_id": "b2-12x32-v1",
        "target_profile_id": "b2-16x48-v1",
        "safe_boundary": SAFE_BOUNDARY_KIND,
        "rollback_checkpoint_digest": "a" * 64,
        "rollback_checkpoint_verified": True,
    }
    payload.update(overrides)
    payload["authorization_receipt_sha256"] = (
        capacity_epoch_authorization_receipt_sha256(payload)
    )
    return payload


def _rng() -> tuple[dict, dict]:
    return (
        {
            "python": {"version": 3, "state": [1, 2, 3], "gauss": None},
            "numpy": {"bit_generator": "PCG64", "state": {"state": 17, "inc": 9}},
        },
        {
            "mutation_torch_cpu": torch.arange(16, dtype=torch.uint8),
            "policy_torch_cpu": torch.tensor([3, 1, 4, 1], dtype=torch.int64),
        },
    )


def _migrate():
    metadata, tensors = _projection()
    rng_state, rng_tensors = _rng()
    migration = migrate_modular_capacity_epoch_projection(
        source_metadata=metadata,
        source_tensors=tensors,
        authorization=_authorization(),
        rng_state_before=deepcopy(rng_state),
        rng_state_after=deepcopy(rng_state),
        rng_tensors_before={name: value.clone() for name, value in rng_tensors.items()},
        rng_tensors_after={name: value.clone() for name, value in rng_tensors.items()},
    )
    return metadata, tensors, rng_state, rng_tensors, migration


def test_authorized_12_32_to_16_48_migration_is_exact_and_rng_neutral() -> None:
    global_torch_rng = torch.get_rng_state().clone()
    metadata, tensors, rng_state, rng_tensors, migration = _migrate()
    assert torch.equal(torch.get_rng_state(), global_torch_rng)
    source = modular_storage_from_projection(metadata, tensors)
    target = modular_storage_from_projection(migration.metadata, migration.tensors)
    assert source is not None and target is not None

    assert source.layout.profile_id == "b2-12x32-v1"
    assert target.layout.profile_id == "b2-16x48-v1"
    assert (source.layout.logical_module_slots, target.layout.logical_module_slots) == (
        12,
        16,
    )
    assert (
        source.layout.logical_connection_slots,
        target.layout.logical_connection_slots,
    ) == (32, 48)
    assert source.layout.physical_module_slots == target.layout.physical_module_slots == 16
    assert (
        source.layout.physical_connection_slots
        == target.layout.physical_connection_slots
        == 48
    )
    assert [(field.name, field.dtype_name, field.shape) for field in source.layout.fields] == [
        (field.name, field.dtype_name, field.shape) for field in target.layout.fields
    ]
    assert all(
        torch.equal(source.tensors[field.name], target.tensors[field.name])
        for field in source.layout.fields
    )
    assert target.records_sidecar[0] is not None
    assert target.records_sidecar[0]["genome"]["b2_caps"] == B2_CAPACITY_PROFILES[
        "b2-16x48-v1"
    ]
    assert migration.metadata["world"] == metadata["world"]
    assert migration.metadata[MODULAR_STORAGE_METADATA_KEY] != metadata[
        MODULAR_STORAGE_METADATA_KEY
    ]

    receipt = migration.receipt.to_dict()
    assert receipt["schema_version"] == MODULAR_CAPACITY_EPOCH_RECEIPT_SCHEMA_VERSION
    assert receipt["source_profile_id"] == "b2-12x32-v1"
    assert receipt["target_profile_id"] == "b2-16x48-v1"
    assert receipt["physical_tensor_resize_count"] == 0
    assert receipt["migration_rng_draws"] == receipt["rng_byte_delta"] == 0
    assert receipt["rng_state_before_sha256"] == receipt["rng_state_after_sha256"]
    assert receipt["rng_tensors_before_sha256"] == receipt["rng_tensors_after_sha256"]
    assert receipt["rollback_checkpoint_digest"] == "a" * 64
    assert receipt["old_occupied_subspace_exact"] is True
    assert receipt["newly_legal_tail_canonical_empty"] is True
    assert receipt["identity_topology_order_state_cost_exact"] is True
    assert receipt["simulation_executed"] is False
    assert receipt["campaign_executed"] is False
    assert receipt["native_execution_performed"] is False
    assert receipt["intelligence_or_emergence_claimed"] is False
    assert receipt["scope"] == "mechanical_storage_validation_only"
    assert rng_state == _rng()[0]
    assert all(torch.equal(rng_tensors[name], _rng()[1][name]) for name in rng_tensors)


def test_manifest_and_profile_switch_are_exact_and_deterministic() -> None:
    _metadata, _tensors, _state, _rng_tensors, first = _migrate()
    _metadata, _tensors, _state, _rng_tensors, second = _migrate()
    assert first.metadata == second.metadata
    assert first.receipt.source_manifest_sha256 == (
        "6d04b414d4d03d6fbce66498d1d13cb374997c9f61dcef6491954185651226e4"
    )
    assert first.receipt.target_manifest_sha256 == (
        "71f0fb80ef21f9fd21c39794fe4fb6398b115b3fefa1569a7241315d9810c2f6"
    )
    assert first.receipt.target_manifest_sha256 != first.receipt.source_manifest_sha256
    assert first.receipt.to_dict() == second.receipt.to_dict()
    assert all(
        torch.equal(first.tensors[name], second.tensors[name])
        for name in first.tensors
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"authorized": False},
        {"source_profile_id": "b2-16x48-v1", "target_profile_id": "b2-12x32-v1"},
        {"target_profile_id": "b2-99x99-v1"},
        {"safe_boundary": "mid_generation"},
        {"rollback_checkpoint_verified": False},
        {"rollback_checkpoint_digest": "not-a-digest"},
    ],
)
def test_missing_reverse_unsupported_or_unsafe_authorization_fails_closed(
    overrides: dict,
) -> None:
    metadata, tensors = _projection()
    rng_state, rng_tensors = _rng()
    with pytest.raises(ModularCapacityEpochError):
        migrate_modular_capacity_epoch_projection(
            source_metadata=metadata,
            source_tensors=tensors,
            authorization=_authorization(**overrides),
            rng_state_before=rng_state,
            rng_state_after=deepcopy(rng_state),
            rng_tensors_before=rng_tensors,
            rng_tensors_after={name: value.clone() for name, value in rng_tensors.items()},
        )


def test_missing_or_tampered_authorization_receipt_fails_closed() -> None:
    metadata, tensors = _projection()
    rng_state, rng_tensors = _rng()
    with pytest.raises(ModularCapacityEpochError, match="authorization"):
        migrate_modular_capacity_epoch_projection(
            source_metadata=metadata,
            source_tensors=tensors,
            authorization={},
            rng_state_before=rng_state,
            rng_state_after=rng_state,
            rng_tensors_before=rng_tensors,
            rng_tensors_after=rng_tensors,
        )
    receipt = _authorization()
    receipt["next_generation"] = 6
    with pytest.raises(ModularCapacityEpochError, match="digest"):
        migrate_modular_capacity_epoch_projection(
            source_metadata=metadata,
            source_tensors=tensors,
            authorization=receipt,
            rng_state_before=rng_state,
            rng_state_after=rng_state,
            rng_tensors_before=rng_tensors,
            rng_tensors_after=rng_tensors,
        )


@pytest.mark.parametrize("kind", ["state", "tensor"])
def test_rng_state_or_tensor_tamper_fails_closed(kind: str) -> None:
    metadata, tensors = _projection()
    rng_state, rng_tensors = _rng()
    post_state = deepcopy(rng_state)
    post_tensors = {name: value.clone() for name, value in rng_tensors.items()}
    if kind == "state":
        post_state["numpy"]["state"]["state"] += 1
    else:
        post_tensors["mutation_torch_cpu"][0] += 1
    with pytest.raises(ModularCapacityEpochError, match="RNG state changed"):
        migrate_modular_capacity_epoch_projection(
            source_metadata=metadata,
            source_tensors=tensors,
            authorization=_authorization(),
            rng_state_before=rng_state,
            rng_state_after=post_state,
            rng_tensors_before=rng_tensors,
            rng_tensors_after=post_tensors,
        )


@pytest.mark.parametrize("kind", ["identity", "state", "shape"])
def test_target_identity_state_or_tensor_shape_tamper_fails_closed(kind: str) -> None:
    metadata, tensors, rng_state, rng_tensors, migration = _migrate()
    target_metadata = deepcopy(migration.metadata)
    target_tensors = {name: value.clone() for name, value in migration.tensors.items()}
    if kind in {"identity", "state"}:
        target = modular_storage_from_projection(target_metadata, target_tensors)
        assert target is not None
        records = deepcopy(list(target.records_sidecar))
        assert records[0] is not None
        if kind == "identity":
            records[0]["agent_id"] = 8
        else:
            state_key = next(iter(records[0]["structural_live_module_state"]))
            records[0]["structural_live_module_state"][state_key] = 0.5
        repacked = pack_modular_storage(
            records,
            agent_capacity=2,
            profile_id="b2-16x48-v1",
        )
        payload = modular_storage_projection_payload(repacked)
        target_metadata[MODULAR_STORAGE_METADATA_KEY] = payload["metadata"][
            MODULAR_STORAGE_METADATA_KEY
        ]
        target_tensors.update(payload["tensors"])
    else:
        name = f"{MODULAR_STORAGE_PROJECTION_PREFIX}agent_id"
        target_tensors[name] = target_tensors[name][:1].clone()

    with pytest.raises(ModularCapacityEpochError):
        validate_modular_capacity_epoch_transition(
            source_metadata=metadata,
            source_tensors=tensors,
            target_metadata=target_metadata,
            target_tensors=target_tensors,
            authorization=_authorization(),
            rng_state_before=rng_state,
            rng_state_after=deepcopy(rng_state),
            rng_tensors_before=rng_tensors,
            rng_tensors_after={name: value.clone() for name, value in rng_tensors.items()},
        )


@pytest.mark.parametrize(
    ("field", "index", "match"),
    [
        ("module_occupied", 12, "module tail"),
        ("connection_occupied", 32, "connection tail"),
    ],
)
def test_occupied_forbidden_tail_is_rejected_before_migration(
    field: str,
    index: int,
    match: str,
) -> None:
    metadata, tensors = _projection()
    packed = modular_storage_from_projection(metadata, tensors)
    assert packed is not None
    packed.tensors[field][0, index] = True
    with pytest.raises(ModularCapacityEpochError, match=match):
        validate_legacy_unused_tail_empty(packed)


def test_expanded_or_tampered_source_projection_fails_closed() -> None:
    metadata, tensors = _projection("b2-16x48-v1")
    rng_state, rng_tensors = _rng()
    with pytest.raises(ModularCapacityEpochError, match="profile mismatch"):
        migrate_modular_capacity_epoch_projection(
            source_metadata=metadata,
            source_tensors=tensors,
            authorization=_authorization(),
            rng_state_before=rng_state,
            rng_state_after=deepcopy(rng_state),
            rng_tensors_before=rng_tensors,
            rng_tensors_after={name: value.clone() for name, value in rng_tensors.items()},
        )

    metadata, tensors = _projection()
    tensors[f"{MODULAR_STORAGE_PROJECTION_PREFIX}module_state"][0, 0, 0] += 0.25
    with pytest.raises(ModularCapacityEpochError, match="invalid modular projection"):
        migrate_modular_capacity_epoch_projection(
            source_metadata=metadata,
            source_tensors=tensors,
            authorization=_authorization(),
            rng_state_before=rng_state,
            rng_state_after=deepcopy(rng_state),
            rng_tensors_before=rng_tensors,
            rng_tensors_after={name: value.clone() for name, value in rng_tensors.items()},
        )


@pytest.mark.skip(reason="optional tiny CUDA validation intentionally not run in this CPU-only task")
def test_optional_tiny_cuda_validation_not_run() -> None:
    assert torch.cuda.is_available()
