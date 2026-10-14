from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from agents.modular_genome_v1 import B2_CAPACITY_PROFILES
from perf.cuda_architecture_b_modular_lifecycle import (
    MODULAR_LIFECYCLE_RECEIPT_SCHEMA_VERSION,
    REPLACEMENT_FIELD_NAMES,
    ModularLifecycleError,
    prune_and_reuse_modular_slots,
)
from perf.cuda_architecture_b_modular_state_abi import pack_modular_storage
from perf.eiel_phase_e_modular_tick_fixture_v1 import (
    _connection as _fixture_connection,
    _development as _fixture_development,
    _module as _fixture_module,
)


MODULE = (
    Path(__file__).resolve().parents[2]
    / "engine/perf/cuda_architecture_b_modular_lifecycle.py"
)


def _modules(*, dependency_blocked: bool = False):
    first = _fixture_module(0, 0, "lifecycle-first")
    pruned = _fixture_module(0, 1, "lifecycle-pruned", dormant=True)
    development = (
        _fixture_development(
            prerequisites=(str(pruned["homology_id"]),),
            latch=True,
        )
        if dependency_blocked
        else None
    )
    third = _fixture_module(
        0,
        2,
        "lifecycle-third",
        development=development,
    )
    replacement = _fixture_module(0, 3, "lifecycle-replacement")
    return first, pruned, third, replacement


def _record(
    *,
    profile_id: str,
    replacement: bool = False,
    dependency_blocked: bool = False,
) -> tuple[dict, str]:
    first, pruned, third, new_module = _modules(
        dependency_blocked=dependency_blocked
    )
    if replacement:
        modules = [first, new_module, third]
        connections = [_fixture_connection(2, first, third)]
        state_modules = [first, third]
        selected_id = str(new_module["module_instance_id"])
    else:
        modules = [first, pruned, third]
        connections = [
            _fixture_connection(0, first, pruned),
            _fixture_connection(1, pruned, third),
            _fixture_connection(2, first, third),
        ]
        state_modules = modules
        selected_id = str(pruned["module_instance_id"])
    state = {
        str(module["module_instance_id"]): 0.125 * (index + 1)
        for index, module in enumerate(state_modules)
    }
    outputs = {
        str(module["module_instance_id"]): -0.0625 * (index + 1)
        for index, module in enumerate(state_modules)
    }
    return (
        {
            "agent_id": 7,
            "agent_slot_token": 70,
            "genome": {
                "schema_version": "modular_genome_v1",
                "genes": {},
                "modules": modules,
                "connections": connections,
                "b2_caps": deepcopy(B2_CAPACITY_PROFILES[profile_id]),
                "b2_invalid_policy": "reject",
            },
            "structural_live_module_state": state,
            "structural_live_module_outputs": outputs,
            "structural_development_latch_by_instance_id": {},
            "structural_development_effective_gain_by_instance_id": {},
            "gated_recurrent_enabled": False,
            "saturated_or_rejected_activity_count": 0,
        },
        selected_id,
    )


def _physical_slot(packed, instance_id: str) -> int:
    identity_index = packed.identity_table.index(instance_id)
    matches = torch.nonzero(
        packed.tensors["module_instance_id_index"][0] == identity_index,
        as_tuple=False,
    )
    return int(matches[0, 0].item())


def _fixture(
    profile_id: str = "b2-12x32-v1",
    *,
    dependency_blocked: bool = False,
):
    source_record, pruned_id = _record(
        profile_id=profile_id,
        dependency_blocked=dependency_blocked,
    )
    target_record, replacement_id = _record(
        profile_id=profile_id,
        replacement=True,
    )
    source = pack_modular_storage(
        [source_record, None], agent_capacity=2, profile_id=profile_id
    )
    target = pack_modular_storage(
        [target_record, None], agent_capacity=2, profile_id=profile_id
    )
    prune_slot = _physical_slot(source, pruned_id)
    replacement_slot = _physical_slot(target, replacement_id)
    fields = source.layout.fields_by_name
    replacement = {}
    for name in REPLACEMENT_FIELD_NAMES:
        if name.startswith("structural_cost_"):
            replacement[name] = target.tensors[name][0:1].clone()
        else:
            replacement[name] = target.tensors[name][0, replacement_slot].unsqueeze(0).clone()
    # The caller's cold identity table has already admitted new identities.
    # Use deliberately distinct indices so resident duplicate checks are exact.
    replacement["module_local_id_index"].fill_(10_001)
    replacement["module_instance_id_index"].fill_(10_002)
    replacement["module_homology_id_index"].fill_(10_003)
    replacement["module_type_index"].fill_(10_004)
    planes = {name: value.clone() for name, value in source.tensors.items()}
    owner = {
        "slot_to_agent_id": torch.tensor([7, -1], dtype=torch.int64),
        "slot_tokens": torch.tensor([70, 99], dtype=torch.int64),
        "alive_mask": torch.tensor([True, False], dtype=torch.bool),
    }
    return source, planes, owner, replacement, prune_slot


def _run(profile_id: str = "b2-12x32-v1"):
    source, planes, owner, replacement, prune_slot = _fixture(profile_id)
    before = {name: value.clone() for name, value in planes.items()}
    result = prune_and_reuse_modular_slots(
        planes,
        owner,
        profile_id=profile_id,
        agent_rows=torch.tensor([0], dtype=torch.int64),
        module_slots=torch.tensor([prune_slot], dtype=torch.int64),
        replacement=replacement,
        cpu_validation_mode=True,
    )
    return source, planes, before, owner, replacement, prune_slot, result


@pytest.mark.parametrize("profile_id", ["b2-12x32-v1", "b2-16x48-v1"])
def test_prune_reuses_same_slot_clears_incident_state_and_preserves_untouched_bytes(
    profile_id: str,
) -> None:
    source, planes, before, owner, replacement, prune_slot, result = _run(profile_id)
    output = result.planes

    assert all(torch.equal(planes[name], before[name]) for name in planes)
    assert torch.equal(output["agent_id"], before["agent_id"])
    assert torch.equal(output["agent_slot_token"], before["agent_slot_token"])
    assert torch.equal(output["module_count"], before["module_count"])
    assert output["module_occupied"][0, prune_slot].item() is True
    assert output["module_instance_id_index"][0, prune_slot].item() == 10_002
    assert output["module_state_present"][0, prune_slot].item() is False
    assert output["module_state"][0, prune_slot].tolist() == [0.0]
    assert output["module_output_present"][0, prune_slot].item() is False
    assert output["module_previous_output"][0, prune_slot].tolist() == [0.0]
    assert output["development_prerequisite_count"][0, prune_slot].item() == 0
    assert torch.equal(output["module_execution_order"], before["module_execution_order"])
    assert torch.equal(
        output["dependency_topological_order"],
        before["dependency_topological_order"],
    )

    incident = result.audit_tensors["cleared_connection_mask"]
    assert int(incident[0].sum().item()) == 2
    assert output["connection_count"][0].item() == 1
    assert output["connection_execution_order"][0].tolist()[:3] == [-1, -1, 0]
    untouched_modules = torch.ones_like(output["module_occupied"])
    untouched_modules[0, prune_slot] = False
    for field in source.layout.fields:
        if field.name.startswith("structural_cost_"):
            assert torch.equal(output[field.name][0:1], replacement[field.name])
        elif field.name in {"connection_count", "connection_execution_order"}:
            continue
        elif field.name.startswith("connection_"):
            tensor = output[field.name]
            fill = torch.full_like(tensor, field.fill_value)
            assert torch.equal(tensor[incident], fill[incident])
            assert torch.equal(tensor[~incident], before[field.name][~incident])
        elif field.name in replacement:
            assert torch.equal(
                output[field.name][0, prune_slot],
                replacement[field.name][0],
            )
            assert torch.equal(
                output[field.name][untouched_modules],
                before[field.name][untouched_modules],
            )
        else:
            assert torch.equal(output[field.name], before[field.name])
    assert all(torch.equal(owner[name], value) for name, value in {
        "slot_to_agent_id": torch.tensor([7, -1], dtype=torch.int64),
        "slot_tokens": torch.tensor([70, 99], dtype=torch.int64),
        "alive_mask": torch.tensor([True, False], dtype=torch.bool),
    }.items())

    receipt = result.receipt.to_dict()
    assert receipt["schema_version"] == MODULAR_LIFECYCLE_RECEIPT_SCHEMA_VERSION
    assert receipt["profile_id"] == profile_id
    assert receipt["decision_count"] == 1
    assert receipt["source_mutated"] is False
    assert receipt["reused_same_physical_slot"] is True
    assert receipt["rng_draws"] == 0
    assert receipt["device_to_host_synchronizations"] == 0
    assert receipt["reproduction_claimed"] is False
    assert receipt["campaign_claimed"] is False
    assert receipt["intelligence_or_emergence_claimed"] is False


def test_active_developmental_dependency_blocks_prune_before_mutation() -> None:
    _source, planes, owner, replacement, prune_slot = _fixture(
        dependency_blocked=True
    )
    before = {name: value.clone() for name, value in planes.items()}
    with pytest.raises(ModularLifecycleError, match="active developmental dependencies"):
        prune_and_reuse_modular_slots(
            planes,
            owner,
            profile_id="b2-12x32-v1",
            agent_rows=torch.tensor([0], dtype=torch.int64),
            module_slots=torch.tensor([prune_slot], dtype=torch.int64),
            replacement=replacement,
            cpu_validation_mode=True,
        )
    assert all(torch.equal(planes[name], before[name]) for name in planes)


@pytest.mark.parametrize("damage", ["duplicate", "out_of_range", "logical", "profile"])
def test_duplicate_out_of_range_or_profile_decisions_fail_closed(damage: str) -> None:
    _source, planes, owner, replacement, prune_slot = _fixture()
    rows = torch.tensor([0], dtype=torch.int64)
    slots = torch.tensor([prune_slot], dtype=torch.int64)
    profile_id = "b2-12x32-v1"
    if damage == "duplicate":
        rows = torch.tensor([0, 0], dtype=torch.int64)
        slots = torch.tensor([prune_slot, prune_slot], dtype=torch.int64)
        replacement = {
            name: torch.cat((value, value), dim=0) for name, value in replacement.items()
        }
    elif damage == "out_of_range":
        rows = torch.tensor([2], dtype=torch.int64)
    elif damage == "logical":
        slots = torch.tensor([12], dtype=torch.int64)
    else:
        profile_id = "b2-99x99-v1"
    with pytest.raises(ModularLifecycleError):
        prune_and_reuse_modular_slots(
            planes,
            owner,
            profile_id=profile_id,
            agent_rows=rows,
            module_slots=slots,
            replacement=replacement,
            cpu_validation_mode=True,
        )


@pytest.mark.parametrize("identity", ["module_local_id_index", "module_instance_id_index", "module_homology_id_index"])
def test_duplicate_replacement_identity_is_rejected(identity: str) -> None:
    _source, planes, owner, replacement, prune_slot = _fixture()
    other_slot = next(
        slot
        for slot in range(16)
        if slot != prune_slot and bool(planes["module_occupied"][0, slot].item())
    )
    replacement[identity][0] = planes[identity][0, other_slot]
    with pytest.raises(ModularLifecycleError, match="duplicate occupied module ID"):
        prune_and_reuse_modular_slots(
            planes,
            owner,
            profile_id="b2-12x32-v1",
            agent_rows=torch.tensor([0], dtype=torch.int64),
            module_slots=torch.tensor([prune_slot], dtype=torch.int64),
            replacement=replacement,
            cpu_validation_mode=True,
        )


@pytest.mark.parametrize("damage", ["state", "dependency", "cost", "shape", "dtype"])
def test_replacement_lifetime_cost_shape_or_dtype_tamper_fails_closed(damage: str) -> None:
    _source, planes, owner, replacement, prune_slot = _fixture()
    if damage == "state":
        replacement["module_state_present"][0] = True
    elif damage == "dependency":
        replacement["development_prerequisite_count"][0] = 1
    elif damage == "cost":
        replacement["structural_cost_total"][0] += 1.0
    elif damage == "shape":
        replacement["topology_position"] = torch.zeros((1, 3), dtype=torch.float32)
    else:
        replacement["module_type_code"] = replacement["module_type_code"].to(
            dtype=torch.int32
        )
    with pytest.raises(ModularLifecycleError):
        prune_and_reuse_modular_slots(
            planes,
            owner,
            profile_id="b2-12x32-v1",
            agent_rows=torch.tensor([0], dtype=torch.int64),
            module_slots=torch.tensor([prune_slot], dtype=torch.int64),
            replacement=replacement,
            cpu_validation_mode=True,
        )


@pytest.mark.parametrize("damage", ["owner", "plane_shape", "plane_dtype", "namespace"])
def test_owner_or_resident_plane_contract_tamper_fails_closed(damage: str) -> None:
    _source, planes, owner, replacement, prune_slot = _fixture()
    if damage == "owner":
        owner["slot_tokens"][0] += 1
    elif damage == "plane_shape":
        planes["module_state"] = planes["module_state"][:, :, :0]
    elif damage == "plane_dtype":
        planes["agent_id"] = planes["agent_id"].to(dtype=torch.int32)
    else:
        planes["phase_e_modular_v2__agent_id"] = planes["agent_id"]
    with pytest.raises(ModularLifecycleError):
        prune_and_reuse_modular_slots(
            planes,
            owner,
            profile_id="b2-12x32-v1",
            agent_rows=torch.tensor([0], dtype=torch.int64),
            module_slots=torch.tensor([prune_slot], dtype=torch.int64),
            replacement=replacement,
            cpu_validation_mode=True,
        )


def test_duplicate_agent_and_invalid_connection_endpoint_fail_closed() -> None:
    _source, planes, owner, replacement, prune_slot = _fixture()
    for name, tensor in planes.items():
        tensor[1] = tensor[0]
    owner["alive_mask"][1] = True
    owner["slot_to_agent_id"][1] = owner["slot_to_agent_id"][0]
    owner["slot_tokens"][1] = planes["agent_slot_token"][1]
    with pytest.raises(ModularLifecycleError, match="duplicate occupied agent ID"):
        prune_and_reuse_modular_slots(
            planes,
            owner,
            profile_id="b2-12x32-v1",
            agent_rows=torch.tensor([0], dtype=torch.int64),
            module_slots=torch.tensor([prune_slot], dtype=torch.int64),
            replacement=replacement,
            cpu_validation_mode=True,
        )

    _source, planes, owner, replacement, prune_slot = _fixture()
    occupied_connection = int(torch.nonzero(planes["connection_occupied"][0])[0, 0].item())
    planes["connection_src_module_slot"][0, occupied_connection] = 16
    with pytest.raises(ModularLifecycleError, match="endpoint"):
        prune_and_reuse_modular_slots(
            planes,
            owner,
            profile_id="b2-12x32-v1",
            agent_rows=torch.tensor([0], dtype=torch.int64),
            module_slots=torch.tensor([prune_slot], dtype=torch.int64),
            replacement=replacement,
            cpu_validation_mode=True,
        )


def test_occupied_agent_id_must_be_nonnegative() -> None:
    _source, planes, owner, replacement, prune_slot = _fixture()
    planes["agent_id"][0] = -1
    owner["slot_to_agent_id"][0] = -1
    with pytest.raises(ModularLifecycleError, match="occupied agent ID is out of range"):
        prune_and_reuse_modular_slots(
            planes,
            owner,
            profile_id="b2-12x32-v1",
            agent_rows=torch.tensor([0], dtype=torch.int64),
            module_slots=torch.tensor([prune_slot], dtype=torch.int64),
            replacement=replacement,
            cpu_validation_mode=True,
        )


@pytest.mark.parametrize("tail_kind", ["module", "connection"])
def test_legacy_profile_rejects_occupied_physical_tail(tail_kind: str) -> None:
    source, planes, owner, replacement, prune_slot = _fixture()
    if tail_kind == "module":
        for name in REPLACEMENT_FIELD_NAMES:
            if name.startswith("structural_cost_"):
                continue
            field = source.layout.fields_by_name[name]
            planes[name][0, 12].copy_(planes[name][0, prune_slot])
            planes[name][0, prune_slot].fill_(field.fill_value)
        for name in (
            "connection_src_module_slot",
            "connection_dst_module_slot",
            "module_execution_order",
            "dependency_topological_order",
        ):
            row = planes[name][0]
            row[row == prune_slot] = 12
    else:
        moved_slot = int(
            torch.nonzero(planes["connection_occupied"][0], as_tuple=False)[-1, 0].item()
        )
        for field in source.layout.fields:
            if not field.name.startswith("connection_") or field.name in {
                "connection_count",
                "connection_execution_order",
            }:
                continue
            planes[field.name][0, 32].copy_(planes[field.name][0, moved_slot])
            planes[field.name][0, moved_slot].fill_(field.fill_value)
        order = planes["connection_execution_order"][0]
        order[order == moved_slot] = 32
    with pytest.raises(
        ModularLifecycleError,
        match=f"{tail_kind} occupancy is outside the logical profile",
    ):
        prune_and_reuse_modular_slots(
            planes,
            owner,
            profile_id="b2-12x32-v1",
            agent_rows=torch.tensor([0], dtype=torch.int64),
            module_slots=torch.tensor([prune_slot], dtype=torch.int64),
            replacement=replacement,
            cpu_validation_mode=True,
        )


def test_cuda_path_source_has_no_host_read_or_sync_primitives() -> None:
    source = MODULE.read_text(encoding="utf-8")
    for forbidden in (".cpu(", ".item(", ".tolist(", ".numpy(", "import numpy"):
        assert forbidden not in source
    assert "torch._assert_async" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tiny_cuda_prune_and_reuse_preserves_resident_order() -> None:
    _source, planes, owner, replacement, prune_slot = _fixture()
    device_planes = {name: value.cuda() for name, value in planes.items()}
    device_owner = {name: value.cuda() for name, value in owner.items()}
    device_replacement = {
        name: value.cuda() for name, value in replacement.items()
    }

    result = prune_and_reuse_modular_slots(
        device_planes,
        device_owner,
        profile_id="b2-12x32-v1",
        agent_rows=torch.tensor([0], dtype=torch.int64, device="cuda"),
        module_slots=torch.tensor(
            [prune_slot], dtype=torch.int64, device="cuda"
        ),
        replacement=device_replacement,
    )

    assert all(value.is_cuda for value in result.planes.values())
    assert torch.equal(
        result.planes["connection_execution_order"][0, :3],
        torch.tensor([-1, -1, 0], dtype=torch.int32, device="cuda"),
    )
    assert result.receipt.device_to_host_synchronizations == 0
