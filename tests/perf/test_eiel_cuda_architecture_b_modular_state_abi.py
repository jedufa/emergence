from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import random

import numpy as np
import pytest
import torch

from agents.modular_genome_v1 import B2_CAPACITY_PROFILES
from agents.structural_identity_v1 import new_native_identity
from evolution.structural_development_v1 import (
    DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
    DEVELOPMENTAL_STAGE_PROFILE_VERSION,
)
from evolution.structural_gated_recurrent_v1 import (
    GATED_OPERATOR,
    neutral_gated_recurrent_record,
)
from evolution.structural_sustainable_growth_v1 import (
    ACTIVITY_RECORD_KEY,
    ACTIVITY_SCHEMA_VERSION,
)
from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_METADATA_KEY,
    MODULAR_STORAGE_PROJECTION_PREFIX,
    PHYSICAL_CONNECTION_SLOTS,
    PHYSICAL_MODULE_SLOTS,
    ModularStorageError,
    build_modular_storage_layout,
    maybe_modular_storage_projection_payload,
    maybe_pack_modular_storage,
    modular_storage_from_projection,
    modular_storage_projection_payload,
    pack_modular_storage,
    unpack_modular_storage,
    validate_modular_storage,
)


def _identity(index: int) -> dict:
    identity = new_native_identity(
        context={
            "campaign_id": "phase-e-storage-abi",
            "branch_id": "test",
            "generation": 2,
            "tick": 0,
            "child_agent_id": index + 100,
            "parent_agent_ids": [1, 2],
            "lineage_token": "storage-abi-lineage",
        },
        operator_id="fixture",
        attempt_index=index,
        operator_role=f"module-{index}",
    )
    position = [0.10 + index * 0.005, 0.20 + index * 0.005]
    identity["topology_pos"] = position
    return identity


def _module(
    index: int,
    *,
    gated: bool = False,
    dormant: bool = False,
    routed: bool = False,
    prerequisites: tuple[str, ...] = (),
    latch: bool = False,
) -> dict:
    identity = _identity(index)
    params: dict = {
        "live_operator": "bounded_scalar_integrator_v1",
        "expression_scale": 0.75,
        "live_retention": 0.5,
        "live_input_gain": 0.25,
        "live_output_gain": -0.5,
    }
    if gated:
        gated_record = neutral_gated_recurrent_record()
        gated_record.update(
            reset_bias=0.1,
            update_input_gain=-0.2,
            candidate_state_gain=0.3,
            output_gain=0.4,
        )
        params = {
            "live_operator": GATED_OPERATOR,
            "expression_scale": 0.8,
            "gated_recurrent": gated_record,
        }
    if dormant:
        params[ACTIVITY_RECORD_KEY] = {
            "schema_version": ACTIVITY_SCHEMA_VERSION,
            "dormant": True,
        }
    if routed:
        params["observation_routing"] = {
            "bindings": [
                {
                    "semantic_id": "self_energy",
                    "target_slot": "in",
                    "gain": 0.5,
                    "enabled": True,
                },
                {
                    "semantic_id": "self_health",
                    "target_slot": "in",
                    "gain": -0.25,
                    "enabled": False,
                },
            ]
        }
        params["output_routing"] = {
            "bindings": [
                {
                    "semantic_id": "move_north",
                    "source_slot": "out",
                    "gain": -0.5,
                    "enabled": False,
                },
                {
                    "semantic_id": "rest",
                    "source_slot": "out",
                    "gain": 0.75,
                    "enabled": True,
                },
            ]
        }
    if prerequisites or latch:
        params["developmental_expression"] = {
            "schema_version": DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
            "stage_profile_version": DEVELOPMENTAL_STAGE_PROFILE_VERSION,
            "stage_gains": [0.25, 0.5, 1.0],
            "prerequisite_homology_ids": list(prerequisites),
            "latch_dependency_readiness": bool(latch),
        }
    return {
        "module_id": f"m{index:03d}",
        "module_type": "generic",
        "params": params,
        "io": {
            "inputs": [{"slot_id": "in", "value_type": "scalar"}],
            "outputs": [{"slot_id": "out", "value_type": "scalar"}],
            "caps": {"max_fanin": 6, "max_fanout": 6},
        },
        **identity,
        "latent_pos": list(identity["topology_pos"]),
    }


def _connection(source: dict, target: dict, index: int) -> dict:
    return {
        "connection_id": f"c{index:03d}",
        "src_module_id": source["module_id"],
        "dst_module_id": target["module_id"],
        "src_slot": "out",
        "dst_slot": "in",
        "value_type": "scalar",
        "channel": "typed",
        "weight": np.float32(0.1 + index * 0.001).item(),
    }


def _record(
    modules: list[dict],
    connections: list[dict] | None = None,
    *,
    profile_id: str = "b2-12x32-v1",
    with_runtime: bool = False,
    agent_id: int = 7,
    agent_slot_token: int = 70,
) -> dict:
    record = {
        "agent_id": agent_id,
        "agent_slot_token": agent_slot_token,
        "genome": {
            "schema_version": "modular_genome_v1",
            "genes": {},
            "modules": deepcopy(modules),
            "connections": deepcopy(connections or []),
            "b2_caps": deepcopy(B2_CAPACITY_PROFILES[profile_id]),
            "b2_invalid_policy": "reject",
        },
        "gated_recurrent_enabled": any(
            str(module.get("params", {}).get("live_operator", "")) == GATED_OPERATOR
            for module in modules
        ),
    }
    if with_runtime:
        ids = [str(module["module_instance_id"]) for module in modules]
        record.update(
            structural_live_module_state={
                identity: np.float32(0.1 + index * 0.1).item()
                for index, identity in enumerate(ids)
            },
            structural_live_module_outputs={
                identity: np.float32(-0.05 - index * 0.05).item()
                for index, identity in enumerate(ids)
            },
            structural_development_latch_by_instance_id={ids[-1]: True},
            structural_development_effective_gain_by_instance_id={ids[-1]: 0.5},
            saturated_or_rejected_activity_count=3,
        )
    return record


def _packed(record: dict, profile_id: str = "b2-12x32-v1"):
    return pack_modular_storage(
        [record, None], agent_capacity=2, profile_id=profile_id
    )


def _legacy_projection(world: str) -> dict:
    return {
        "metadata": {"world": world, "capacity": 2, "feature": False},
        "tensors": {
            "slot_to_agent_id": torch.tensor([4, -1], dtype=torch.int64),
            "alive_mask": torch.tensor([True, False], dtype=torch.bool),
            "energy": torch.tensor([1.25, 0.0], dtype=torch.float32),
        },
    }


def _projection_digest(projection: dict) -> str:
    payload = {
        "metadata": projection["metadata"],
        "tensors": {
            name: {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "bytes": tensor.numpy().tobytes().hex(),
            }
            for name, tensor in sorted(projection["tensors"].items())
        },
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_both_profiles_have_identical_16x48_physical_shapes() -> None:
    legacy = build_modular_storage_layout(3, "b2-12x32-v1")
    expanded = build_modular_storage_layout(3, "b2-16x48-v1")
    assert legacy.physical_module_slots == expanded.physical_module_slots == 16
    assert legacy.physical_connection_slots == expanded.physical_connection_slots == 48
    assert [(f.name, f.dtype_name, f.shape) for f in legacy.fields] == [
        (f.name, f.dtype_name, f.shape) for f in expanded.fields
    ]
    assert all(field.dtype_name != "int16" for field in legacy.fields)
    assert all(field.dtype_name != "int16" for field in expanded.fields)
    assert (legacy.logical_module_slots, legacy.logical_connection_slots) == (12, 32)
    assert (expanded.logical_module_slots, expanded.logical_connection_slots) == (16, 48)


def test_legacy_profile_rejects_module_12_and_connection_32_occupancy() -> None:
    modules = [_module(index) for index in range(13)]
    with pytest.raises(ModularStorageError, match="logical module capacity"):
        _packed(_record(modules))
    modules = [_module(index) for index in range(12)]
    connections = [
        _connection(modules[index % 12], modules[(index + 1) % 12], index)
        for index in range(33)
    ]
    with pytest.raises(ModularStorageError, match="logical connection capacity"):
        _packed(_record(modules, connections))


def test_expanded_profile_accepts_16_modules_and_48_connections() -> None:
    modules = [_module(index) for index in range(16)]
    connections = []
    for offset in (1, 2, 3):
        for index in range(16):
            connections.append(
                _connection(modules[index], modules[(index + offset) % 16], len(connections))
            )
    packed = _packed(
        _record(modules, connections, profile_id="b2-16x48-v1"),
        "b2-16x48-v1",
    )
    assert int(packed.tensors["module_count"][0]) == 16
    assert int(packed.tensors["connection_count"][0]) == 48
    assert bool(packed.tensors["module_occupied"][0].all())
    assert bool(packed.tensors["connection_occupied"][0].all())


def test_above_physical_envelope_fails_closed() -> None:
    modules = [_module(index) for index in range(PHYSICAL_MODULE_SLOTS + 1)]
    with pytest.raises(ModularStorageError, match="logical module capacity"):
        _packed(
            _record(modules, profile_id="b2-16x48-v1"),
            "b2-16x48-v1",
        )


def test_integrator_and_gated_payloads_roundtrip_exactly() -> None:
    modules = [_module(0), _module(1, gated=True)]
    source = _record(modules, [_connection(modules[0], modules[1], 0)])
    packed = _packed(source)
    restored = unpack_modular_storage(packed)
    assert restored[0] == packed.records_sidecar[0]
    assert restored[1] is None
    assert set(packed.tensors["operator_code"][0, :2].tolist()) == {1, 2}


def test_routing_development_dormancy_and_runtime_state_roundtrip_exactly() -> None:
    root = _module(0, routed=True)
    child = _module(
        1,
        gated=True,
        dormant=True,
        routed=True,
        prerequisites=(root["homology_id"],),
        latch=True,
    )
    packed = _packed(
        _record([root, child], [_connection(root, child, 0)], with_runtime=True)
    )
    restored = unpack_modular_storage(packed)[0]
    assert restored is not None
    assert restored["structural_live_module_state"]
    assert restored["structural_development_latch_by_instance_id"] == {
        child["module_instance_id"]: True
    }
    assert int(packed.tensors["input_binding_count"][0].sum()) == 4
    assert int(packed.tensors["output_binding_count"][0].sum()) == 4
    assert int(packed.tensors["development_prerequisite_count"][0].sum()) == 1
    assert int(packed.tensors["module_dormant"][0].sum()) == 1
    assert float(packed.tensors["structural_cost_total"][0]) > 0.0


def test_output_meaning_code_is_separate_from_authoritative_action_index() -> None:
    packed = _packed(_record([_module(0, routed=True)]))
    # The compact semantic registry has five entries, so ``rest`` has semantic
    # code 4.  The established seven-action vocabulary has GATHER at 4 and REST
    # at 5.  The hot ABI must never confuse those two namespaces.
    assert packed.tensors["output_semantic_code"][0, 0].tolist() == [0, 4]
    assert packed.tensors["output_action_index"][0, 0].tolist() == [0, 5]


def test_identity_sidecar_and_slot_maps_roundtrip_exactly() -> None:
    modules = [_module(2), _module(0), _module(1)]
    packed = _packed(_record(modules))
    table = packed.identity_table
    indices = packed.tensors["module_instance_id_index"][0, :3].tolist()
    observed = {table[index] for index in indices}
    assert observed == {module["module_instance_id"] for module in modules}
    assert packed.tensors["agent_id"].dtype == torch.int64
    assert packed.tensors["agent_slot_token"].dtype == torch.int64
    assert packed.tensors["agent_id"].tolist() == [7, -1]
    assert packed.tensors["agent_slot_token"].tolist() == [70, -1]
    restored = unpack_modular_storage(packed)[0]
    assert restored is not None
    assert (restored["agent_id"], restored["agent_slot_token"]) == (7, 70)
    assert packed.sidecar_sha256 == sha256(
        json.dumps(
            {
                "identity_table": list(packed.identity_table),
                "records": list(packed.records_sidecar),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def test_local_module_and_connection_repack_preserves_hot_semantic_planes() -> None:
    modules = [_module(0, gated=True), _module(1, gated=True)]
    connection = _connection(modules[0], modules[1], 0)
    first = _packed(_record(modules, [connection]))
    repacked_modules = deepcopy(modules)
    repacked_modules[0]["module_id"] = "m900"
    repacked_modules[1]["module_id"] = "m100"
    repacked_connection = deepcopy(connection)
    repacked_connection.update(
        connection_id="c777",
        src_module_id="m900",
        dst_module_id="m100",
    )
    second = _packed(_record(repacked_modules, [repacked_connection]))
    stable_planes = (
        "module_instance_id_index",
        "module_homology_id_index",
        "topology_position",
        "module_execution_order",
        "operator_code",
        "integrator_parameters",
        "gated_parameters",
        "connection_src_module_slot",
        "connection_dst_module_slot",
        "connection_weight",
        "connection_execution_order",
    )
    for name in stable_planes:
        if name in {"module_instance_id_index", "module_homology_id_index"}:
            left = [first.identity_table[index] for index in first.tensors[name][0, :2]]
            right = [second.identity_table[index] for index in second.tensors[name][0, :2]]
            assert left == right
        else:
            assert torch.equal(first.tensors[name], second.tensors[name]), name


def test_unused_physical_tail_has_canonical_bytes() -> None:
    packed = _packed(_record([_module(0)]))
    for field in packed.layout.fields:
        tensor = packed.tensors[field.name]
        tail = None
        if len(field.shape) >= 2 and field.shape[1] == PHYSICAL_MODULE_SLOTS:
            tail = tensor[0, 1:]
        elif len(field.shape) >= 2 and field.shape[1] == PHYSICAL_CONNECTION_SLOTS:
            tail = tensor[0, 0:]
        if tail is not None:
            assert torch.equal(tail, torch.full_like(tail, field.fill_value)), field.name
    for field in packed.layout.fields:
        assert torch.equal(
            packed.tensors[field.name][1],
            torch.full_like(packed.tensors[field.name][1], field.fill_value),
        ), field.name


def test_duplicate_dangling_and_dependency_states_fail_closed() -> None:
    first = _module(0)
    duplicate = _module(1)
    duplicate["module_instance_id"] = first["module_instance_id"]
    with pytest.raises(ModularStorageError, match="duplicate module instance"):
        _packed(_record([first, duplicate]))
    dangling = _connection(first, _module(1), 0)
    with pytest.raises(ModularStorageError, match="canonicalization failed|dropped"):
        _packed(_record([first], [dangling]))
    missing = _module(
        2,
        prerequisites=("mh_missing_exact_identity",),
        latch=True,
    )
    with pytest.raises(ModularStorageError, match="canonicalization failed"):
        _packed(_record([missing]))
    valid = _record([_module(3)], agent_id=8, agent_slot_token=80)
    with pytest.raises(ModularStorageError, match="duplicate occupied agent_id"):
        pack_modular_storage(
            [valid, _record([_module(4)], agent_id=8, agent_slot_token=81)],
            agent_capacity=2,
        )
    shared_token = pack_modular_storage(
        [valid, _record([_module(4)], agent_id=9, agent_slot_token=80)],
        agent_capacity=2,
    )
    assert shared_token.tensors["agent_slot_token"].tolist() == [80, 80]
    for field, value in (("agent_id", -1), ("agent_slot_token", True)):
        invalid = _record([_module(5)])
        invalid[field] = value
        with pytest.raises(ModularStorageError, match=field):
            _packed(invalid)


def test_pack_unpack_consumes_zero_python_numpy_and_torch_rng() -> None:
    source = _record([_module(0), _module(1, gated=True)], with_runtime=True)
    python_before = random.getstate()
    numpy_before = deepcopy(np.random.get_state())
    torch_before = torch.random.get_rng_state().clone()
    packed = _packed(source)
    unpack_modular_storage(packed)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_disabled_overlay_does_not_read_inputs_or_mutate_projection() -> None:
    class ExplodesOnRead:
        def __iter__(self):
            raise AssertionError("disabled overlay inspected records")

        def __getattribute__(self, name):
            if name == "__class__":
                return ExplodesOnRead
            raise AssertionError("disabled overlay inspected input")

    projection = _legacy_projection("flat")
    before = deepcopy(projection)
    assert maybe_pack_modular_storage(
        False,
        ExplodesOnRead(),
        agent_capacity=object(),
        profile_id=object(),
    ) is None
    assert projection["metadata"] == before["metadata"]
    for name in projection["tensors"]:
        assert torch.equal(projection["tensors"][name], before["tensors"][name])


def test_enabled_overlay_rejects_an_empty_profile_id() -> None:
    with pytest.raises(ModularStorageError, match="unsupported modular capacity profile"):
        maybe_pack_modular_storage(True, [], agent_capacity=1, profile_id="")


def test_disabled_flat_projection_digest_and_tensor_bytes_are_exact() -> None:
    projection = _legacy_projection("flat")
    digest = _projection_digest(projection)
    tensor_bytes = {name: value.numpy().tobytes() for name, value in projection["tensors"].items()}
    assert maybe_pack_modular_storage(False, (_ for _ in () if False)) is None
    assert _projection_digest(projection) == digest
    assert {name: value.numpy().tobytes() for name, value in projection["tensors"].items()} == tensor_bytes


def test_disabled_patch_projection_digest_and_tensor_bytes_are_exact() -> None:
    projection = _legacy_projection("patch")
    digest = _projection_digest(projection)
    tensor_bytes = {name: value.numpy().tobytes() for name, value in projection["tensors"].items()}
    assert maybe_pack_modular_storage(False, None) is None
    assert _projection_digest(projection) == digest
    assert {name: value.numpy().tobytes() for name, value in projection["tensors"].items()} == tensor_bytes


def test_layout_json_and_packed_bytes_are_repeatable() -> None:
    modules = [_module(0, routed=True), _module(1, gated=True)]
    source = _record(modules, [_connection(modules[0], modules[1], 0)], with_runtime=True)
    first = _packed(source)
    reordered = deepcopy(source)
    reordered["genome"] = dict(reversed(list(reordered["genome"].items())))
    second = _packed(reordered)
    validate_modular_storage(first)
    validate_modular_storage(second)
    assert json.dumps(first.layout.to_dict(), sort_keys=True) == json.dumps(
        second.layout.to_dict(), sort_keys=True
    )
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.packed_sha256 == second.packed_sha256


def test_projection_encode_decode_is_exact_and_retains_profile_slot_bindings() -> None:
    source = _record(
        [_module(0), _module(1, gated=True)],
        profile_id="b2-16x48-v1",
        with_runtime=True,
        agent_id=41,
        agent_slot_token=3,
    )
    packed = pack_modular_storage(
        [source, None], agent_capacity=2, profile_id="b2-16x48-v1"
    )
    projection = modular_storage_projection_payload(packed)
    assert set(projection) == {"metadata", "tensors"}
    assert set(projection["metadata"]) == {MODULAR_STORAGE_METADATA_KEY}
    assert all(
        name.startswith(MODULAR_STORAGE_PROJECTION_PREFIX)
        for name in projection["tensors"]
    )
    restored = modular_storage_from_projection(
        projection["metadata"], projection["tensors"]
    )
    assert restored is not None
    assert restored.canonical_bytes() == packed.canonical_bytes()
    assert restored.layout.profile_id == "b2-16x48-v1"
    assert restored.layout.physical_module_slots == 16
    assert restored.layout.physical_connection_slots == 48
    assert restored.tensors["agent_id"].tolist() == [41, -1]
    assert restored.tensors["agent_slot_token"].tolist() == [3, -1]


def test_projection_tensors_are_clones_and_decode_does_not_alias_inputs() -> None:
    packed = _packed(_record([_module(0)], agent_id=12, agent_slot_token=2))
    projection = modular_storage_projection_payload(packed)
    projected_agent_id = projection["tensors"][
        f"{MODULAR_STORAGE_PROJECTION_PREFIX}agent_id"
    ]
    assert projected_agent_id.data_ptr() != packed.tensors["agent_id"].data_ptr()
    restored = modular_storage_from_projection(
        projection["metadata"], projection["tensors"]
    )
    assert restored is not None
    assert restored.tensors["agent_id"].data_ptr() != projected_agent_id.data_ptr()
    restored.tensors["agent_id"][0] = 999
    assert projected_agent_id.tolist() == [12, -1]
    assert packed.tensors["agent_id"].tolist() == [12, -1]


def test_default_off_projection_returns_empty_before_reads_and_consumes_zero_rng() -> None:
    class ExplodesOnRead:
        def __iter__(self):
            raise AssertionError("disabled projection inspected records")

        def __getattribute__(self, name):
            if name == "__class__":
                return ExplodesOnRead
            raise AssertionError("disabled projection inspected input")

    python_before = random.getstate()
    numpy_before = deepcopy(np.random.get_state())
    torch_before = torch.random.get_rng_state().clone()
    assert maybe_modular_storage_projection_payload(
        False,
        ExplodesOnRead(),
        agent_capacity=object(),
        profile_id=object(),
    ) == {"metadata": {}, "tensors": {}}
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_projection_absence_is_none_and_stray_prefixed_tensor_is_rejected() -> None:
    legacy = _legacy_projection("flat")
    assert modular_storage_from_projection(legacy["metadata"], legacy["tensors"]) is None
    stray = dict(legacy["tensors"])
    stray[f"{MODULAR_STORAGE_PROJECTION_PREFIX}agent_id"] = torch.tensor(
        [-1], dtype=torch.int64
    )
    with pytest.raises(ModularStorageError, match="without metadata"):
        modular_storage_from_projection(legacy["metadata"], stray)


def test_projection_missing_extra_and_mixed_version_fields_fail_closed() -> None:
    packed = _packed(_record([_module(0)]))
    projection = modular_storage_projection_payload(packed)
    missing = dict(projection["tensors"])
    missing.pop(f"{MODULAR_STORAGE_PROJECTION_PREFIX}agent_id")
    with pytest.raises(ModularStorageError, match="tensor field set mismatch"):
        modular_storage_from_projection(projection["metadata"], missing)
    extra = dict(projection["tensors"])
    extra[f"{MODULAR_STORAGE_PROJECTION_PREFIX}unexpected"] = torch.zeros(
        (1,), dtype=torch.int32
    )
    with pytest.raises(ModularStorageError, match="tensor field set mismatch"):
        modular_storage_from_projection(projection["metadata"], extra)
    mixed = dict(projection["tensors"])
    mixed["phase_e_modular_v2__agent_id"] = torch.zeros((1,), dtype=torch.int64)
    with pytest.raises(ModularStorageError, match="mixed modular projection version"):
        modular_storage_from_projection(projection["metadata"], mixed)


def test_projection_tampered_metadata_tensor_manifest_and_digests_fail_closed() -> None:
    packed = _packed(_record([_module(0)]))
    projection = modular_storage_projection_payload(packed)
    tampered_metadata = deepcopy(projection["metadata"])
    tampered_metadata[MODULAR_STORAGE_METADATA_KEY]["profile_id"] = "b2-16x48-v1"
    with pytest.raises(ModularStorageError, match="envelope digest mismatch"):
        modular_storage_from_projection(tampered_metadata, projection["tensors"])

    tampered_tensor = {name: tensor.clone() for name, tensor in projection["tensors"].items()}
    tampered_tensor[f"{MODULAR_STORAGE_PROJECTION_PREFIX}agent_id"][0] += 1
    with pytest.raises(ModularStorageError, match="plane digest mismatch"):
        modular_storage_from_projection(projection["metadata"], tampered_tensor)

    tampered_manifest = deepcopy(projection["metadata"])
    envelope = tampered_manifest[MODULAR_STORAGE_METADATA_KEY]
    envelope["tensor_manifest"][0]["sha256"] = "0" * 64
    without_digest = dict(envelope)
    without_digest.pop("envelope_sha256")
    envelope["envelope_sha256"] = sha256(
        json.dumps(
            without_digest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    with pytest.raises(ModularStorageError, match="plane digest mismatch"):
        modular_storage_from_projection(tampered_manifest, projection["tensors"])

    tampered_packed_digest = deepcopy(projection["metadata"])
    envelope = tampered_packed_digest[MODULAR_STORAGE_METADATA_KEY]
    envelope["packed_sha256"] = "f" * 64
    without_digest = dict(envelope)
    without_digest.pop("envelope_sha256")
    envelope["envelope_sha256"] = sha256(
        json.dumps(
            without_digest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    with pytest.raises(ModularStorageError, match="packed modular storage digest mismatch"):
        modular_storage_from_projection(tampered_packed_digest, projection["tensors"])


def test_projection_payload_is_repeatable_and_does_not_mutate_legacy_inputs() -> None:
    source = _record([_module(0, routed=True)], agent_id=55, agent_slot_token=4)
    source_before = deepcopy(source)
    first = maybe_modular_storage_projection_payload(
        True, [source], agent_capacity=1, profile_id="b2-12x32-v1"
    )
    second = maybe_modular_storage_projection_payload(
        True, [deepcopy(source)], agent_capacity=1, profile_id="b2-12x32-v1"
    )
    assert source == source_before
    assert first["metadata"] == second["metadata"]
    assert set(first["tensors"]) == set(second["tensors"])
    for name in first["tensors"]:
        assert torch.equal(first["tensors"][name], second["tensors"][name])
        assert first["tensors"][name].data_ptr() != second["tensors"][name].data_ptr()
