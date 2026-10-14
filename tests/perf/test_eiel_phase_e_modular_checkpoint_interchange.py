from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from agents.modular_genome_v1 import B2_CAPACITY_PROFILES
from agents.structural_identity_v1 import new_native_identity
from native_runtime.checkpoint_digests import DigestLayer, checkpoint_digest_manifest
from native_runtime.interchange_projection import (
    InterchangeProjectionError,
    interchange_projection_from_native_checkpoint,
    native_checkpoint_projection_from_interchange,
    validate_native_checkpoint_projection,
)
from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_METADATA_KEY,
    MODULAR_STORAGE_PROJECTION_PREFIX,
    ModularStorageError,
    modular_storage_projection_payload,
    pack_modular_storage,
)


def _legacy_projection() -> dict[str, object]:
    tensors = {
        "agent_grid": torch.tensor([[-1]], dtype=torch.int64),
        "alive_mask": torch.tensor([False]),
        "slot_to_agent_id": torch.tensor([-1], dtype=torch.int64),
        "agent_id_to_slot": torch.tensor([-1], dtype=torch.int32),
        "agent_ids_by_alive_order": torch.empty((0,), dtype=torch.int64),
        "agent_id_to_slot_agent_ids": torch.empty((0,), dtype=torch.int64),
        "agent_id_to_slot_slots": torch.empty((0,), dtype=torch.int32),
        "slot_tokens": torch.tensor([0], dtype=torch.int64),
        "slot_free_queue": torch.tensor([0], dtype=torch.int64),
        "pos_y": torch.tensor([0], dtype=torch.int64),
        "pos_x": torch.tensor([0], dtype=torch.int64),
        "energy": torch.tensor([0.0], dtype=torch.float32),
        "health": torch.tensor([0.0], dtype=torch.float32),
        "drives": torch.zeros((1, 4), dtype=torch.float32),
        "hmem": torch.empty((1, 0), dtype=torch.float32),
        "mlp_pack": torch.zeros((1, 1), dtype=torch.float32),
        "action_cache_token": torch.tensor([-1], dtype=torch.int64),
        "action_cache_cursor": torch.tensor([-1], dtype=torch.int32),
        "action_cache_next": torch.tensor([-1], dtype=torch.int32),
        "action_cache_actions": torch.empty((1, 0), dtype=torch.int32),
        "event_tape_type": torch.empty((0,), dtype=torch.int32),
        "event_tape_actor_id": torch.empty((0,), dtype=torch.int64),
        "event_tape_target_id": torch.empty((0,), dtype=torch.int64),
        "event_tape_value": torch.empty((0,), dtype=torch.float32),
        "lineage_founder_id": torch.tensor([-1], dtype=torch.int64),
        "lineage_depth": torch.tensor([0], dtype=torch.int32),
        "offspring_count": torch.tensor([0], dtype=torch.int64),
    }
    return {
        "schema_version": "cxxbackbone10_checkpoint_projection_v1",
        "metadata": {
            "generation": 3,
            "tick": 0,
            "world": "flat",
            "agent_grid_phase_contract_schema_version": "eiel_agent_grid_phase_contract_v1",
            "agent_grid_runner_phase": "generation_boundary",
            "agent_grid_authority_epoch": "generation:3:tick:0",
            "agent_grid_preserve_exact_stored": True,
            "native_runtime_numpy_env_state": {},
            "native_runtime_numpy_runner_state": {},
            "native_runtime_numpy_population_state": {},
            "native_runtime_python_random_state": {},
            "event_tape_active_count": 0,
            "event_tape_cursor": 0,
        },
        "tensors": tensors,
        "event_cursor": 0,
        "runtime_event_cursor": 0,
    }


def _native_module(*, ancestry_generation: int) -> dict[str, object]:
    identity = new_native_identity(
        context={
            "campaign_id": "phase-e-interchange",
            "branch_id": "test",
            "generation": 1,
            "tick": 0,
            "child_agent_id": 7,
            "parent_agent_ids": [],
            "lineage_token": "phase-e-interchange-lineage",
        },
        operator_id="fixture",
        attempt_index=0,
        operator_role="module-0",
    )
    identity["ancestry"]["origin_generation"] = ancestry_generation
    return {
        "module_id": "m000",
        "module_type": "generic",
        "params": {
            "live_operator": "bounded_scalar_integrator_v1",
            "expression_scale": 0.75,
            "live_retention": 0.5,
            "live_input_gain": 0.25,
            "live_output_gain": -0.5,
        },
        "io": {
            "inputs": [{"slot_id": "in", "value_type": "scalar"}],
            "outputs": [{"slot_id": "out", "value_type": "scalar"}],
            "caps": {"max_fanin": 6, "max_fanout": 6},
        },
        **identity,
        "latent_pos": list(identity["topology_pos"]),
    }


def _packed_with_ancestry(ancestry_generation: int):
    return pack_modular_storage(
        [
            {
                "agent_id": 7,
                "agent_slot_token": 70,
                "genome": {
                    "schema_version": "modular_genome_v1",
                    "genes": {},
                    "modules": [_native_module(ancestry_generation=ancestry_generation)],
                    "connections": [],
                    "b2_caps": deepcopy(B2_CAPACITY_PROFILES["b2-12x32-v1"]),
                    "b2_invalid_policy": "reject",
                },
                "gated_recurrent_enabled": False,
            }
        ],
        agent_capacity=1,
    )


def _with_modular_payload(projection: dict[str, object], packed=None) -> dict[str, object]:
    out = deepcopy(projection)
    payload = modular_storage_projection_payload(
        pack_modular_storage([], agent_capacity=1) if packed is None else packed
    )
    out["metadata"].update(payload["metadata"])
    out["tensors"].update(payload["tensors"])
    return out


def test_modular_cold_sidecar_change_alters_trajectory_and_storage_digests() -> None:
    before_packed = _packed_with_ancestry(1)
    after_packed = _packed_with_ancestry(2)
    assert all(
        torch.equal(before_packed.tensors[name], after_packed.tensors[name])
        for name in before_packed.tensors
    )
    before = _with_modular_payload(_legacy_projection(), before_packed)
    after = _with_modular_payload(_legacy_projection(), after_packed)

    for layer in (
        DigestLayer.TRAJECTORY_STATE,
        DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
    ):
        before_manifest = checkpoint_digest_manifest(before, layer=layer)
        after_manifest = checkpoint_digest_manifest(after, layer=layer)
        assert before_manifest["digest"] != after_manifest["digest"]
        assert before_manifest["metadata"][MODULAR_STORAGE_METADATA_KEY] == {
            "profile_id": before_packed.layout.profile_id,
            "sidecar_sha256": before_packed.sidecar_sha256,
            "packed_sha256": before_packed.packed_sha256,
        }


@pytest.mark.parametrize("damage", ["tamper", "missing", "extra", "stray", "mixed"])
def test_interchange_validation_rejects_invalid_modular_projection(damage: str) -> None:
    projection = _with_modular_payload(_legacy_projection())
    first_name = next(
        name for name in projection["tensors"] if name.startswith(MODULAR_STORAGE_PROJECTION_PREFIX)
    )
    if damage == "tamper":
        projection["tensors"][first_name][0] = True
    elif damage == "missing":
        del projection["tensors"][first_name]
    elif damage == "extra":
        projection["tensors"][f"{MODULAR_STORAGE_PROJECTION_PREFIX}future_plane"] = torch.tensor([0])
    elif damage == "stray":
        del projection["metadata"][MODULAR_STORAGE_METADATA_KEY]
    else:
        projection["tensors"]["phase_e_modular_v2__agent_occupied"] = torch.tensor([False])

    with pytest.raises(ModularStorageError):
        checkpoint_digest_manifest(projection, layer=DigestLayer.TRAJECTORY_STATE)
    with pytest.raises(InterchangeProjectionError, match="invalid Phase E modular storage"):
        validate_native_checkpoint_projection(projection)


@pytest.mark.parametrize(
    ("runtime_cursor", "event_cursor", "message"),
    [
        (None, 0, "runtime_event_cursor must be a non-negative int"),
        (-1, 0, "runtime_event_cursor must be a non-negative int"),
        (0, 1, "runtime_event_cursor cannot precede event_cursor"),
        (1.0, 0, "runtime_event_cursor must be a non-negative int"),
    ],
)
def test_interchange_rejects_missing_stale_or_incompatible_runtime_event_cursor(
    runtime_cursor: object,
    event_cursor: int,
    message: str,
) -> None:
    projection = _with_modular_payload(_legacy_projection())
    projection["event_cursor"] = event_cursor
    if runtime_cursor is None:
        del projection["runtime_event_cursor"]
    else:
        projection["runtime_event_cursor"] = runtime_cursor

    with pytest.raises(InterchangeProjectionError, match=message):
        validate_native_checkpoint_projection(projection)


def test_valid_interchange_roundtrip_retains_exact_modular_envelope_and_tensors() -> None:
    projection = _with_modular_payload(_legacy_projection(), _packed_with_ancestry(4))

    interchange = interchange_projection_from_native_checkpoint(
        projection,
        source_runtime="test",
        checkpoint_boundary="generation_boundary",
    )
    restored = native_checkpoint_projection_from_interchange(interchange)

    assert restored["metadata"][MODULAR_STORAGE_METADATA_KEY] == (
        projection["metadata"][MODULAR_STORAGE_METADATA_KEY]
    )
    names = {
        name for name in projection["tensors"] if name.startswith(MODULAR_STORAGE_PROJECTION_PREFIX)
    }
    assert names
    for name in names:
        assert torch.equal(restored["tensors"][name], projection["tensors"][name])
        assert restored["tensors"][name].data_ptr() != projection["tensors"][name].data_ptr()
