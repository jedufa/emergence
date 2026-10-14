"""Manifest completeness and opt-in CUDA proof for R11 transport R13."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import os
from pathlib import Path

import pytest
import torch

from engine.tensor_world.c4_event_ring_r11 import make_c4_event_ring_r11_v1
from engine.tensor_world.c4_fixture import make_c4_fixture_v1
from engine.tensor_world.c4_resident_composed_r11 import (
    _R11_GUARDED_SHARED_TENSOR_PATHS_V1,
    _clone_resident_value_v1,
)
from engine.tensor_world.c4_resident_transport_r13 import (
    C4_RESIDENT_TRANSPORT_R13_DECISION_V1,
    C4_RESIDENT_TRANSPORT_R13_SCHEMA_VERSION_V1,
    c4_resident_transport_r13_source_receipt_v1,
    inventory_c4_resident_transport_r13_v1,
    load_c4_resident_transport_r13_extension,
    materialize_c4_resident_transport_plan_r13_v1,
)
from engine.tensor_world.c5_determinism import make_c5_pcg64_context_v1
from engine.tensor_world.contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[1]
PROBE_ENV = "EIEL_C4_R13_TRANSPORT_PROBE"
BUILD_DIR_ENV = "EIEL_C4_R13_TRANSPORT_BUILD_DIR"


def _tensor_leaves(value: object, path: str) -> dict[str, torch.Tensor]:
    leaves: dict[str, torch.Tensor] = {}
    if isinstance(value, torch.Tensor):
        leaves[path] = value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            leaves.update(_tensor_leaves(item, f"{path}[{index}]"))
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            leaves.update(
                _tensor_leaves(
                    getattr(value, field.name),
                    f"{path}.{field.name}" if path else field.name,
                )
            )
    return leaves


def _fixture(*, batch: int, capacity: int, device: torch.device):
    bundle, _ = make_c4_fixture_v1(
        batch=batch,
        capacity=capacity,
        active=min(3, capacity),
        width=8 if capacity <= 32 else 32,
    )
    bundle.event_ring = make_c4_event_ring_r11_v1(
        batch=batch,
        capacity=capacity,
        event_capacity=bundle.events.capacity,
        device="cpu",
    )
    replay = make_c5_pcg64_context_v1(
        bundle,
        streams=5,
        seed=701,
        typed=True,
    )
    if device.type == "cuda":
        bundle = _clone_to(bundle, device)
        replay = _clone_to(replay, device)
    shadow_bundle = _clone_resident_value_v1(bundle)
    shadow_replay = _clone_resident_value_v1(replay)
    return bundle, replay, shadow_bundle, shadow_replay


def _clone_to(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, copy=True).contiguous()
    if isinstance(value, tuple):
        return tuple(_clone_to(item, device) for item in value)
    if isinstance(value, list):
        return [_clone_to(item, device) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                field.name: _clone_to(getattr(value, field.name), device)
                for field in fields(value)
            }
        )
    return value


def _assert_bytes_equal(left: torch.Tensor, right: torch.Tensor, path: str) -> None:
    assert torch.equal(left.view(torch.uint8), right.view(torch.uint8)), path


def test_r13_manifest_is_complete_deduplicated_and_excludes_only_policy() -> None:
    bundle, replay, shadow_bundle, shadow_replay = _fixture(
        batch=2,
        capacity=8,
        device=torch.device("cpu"),
    )
    inventory = inventory_c4_resident_transport_r13_v1(
        bundle, shadow_bundle, replay, shadow_replay
    )
    source_paths = set(_tensor_leaves(bundle, "bundle")) | set(
        _tensor_leaves(replay, "replay")
    )
    expected_shared = {
        f"bundle.{path}" for path in _R11_GUARDED_SHARED_TENSOR_PATHS_V1
    }
    assert set(inventory.covered_paths) | set(inventory.shared_alias_paths) == source_paths
    assert set(inventory.shared_alias_paths) == expected_shared
    assert set(inventory.covered_paths).isdisjoint(expected_shared)
    assert inventory.unique_field_count == 322
    assert len(source_paths) == 324
    assert inventory.total_bytes > 0
    assert inventory.chunk_count >= inventory.unique_field_count

    receipt = c4_resident_transport_r13_source_receipt_v1()
    assert receipt["schema_version"] == C4_RESIDENT_TRANSPORT_R13_SCHEMA_VERSION_V1
    assert receipt["decision"] == C4_RESIDENT_TRANSPORT_R13_DECISION_V1
    assert receipt["copy_launches_per_tick"] == 1
    assert receipt["guarded_publish_launches_per_tick"] == 1
    assert receipt["hardware_fault_atomicity_claim"] is False
    assert receipt["production_bound"] is True
    assert receipt["integration_surface"] == "TensorWorldC4ResidentComposedR11OwnerV1"
    assert receipt["gate_claim"] is False
    owner_source = (
        ROOT / "engine/tensor_world/c4_resident_composed_r11.py"
    ).read_text(encoding="utf-8")
    assert "c4_resident_transport_r13" in owner_source
    assert "transport_extension.copy_to_shadow" in owner_source
    assert "transport_extension.publish_if_all_ok" in owner_source
    assert "for destination, source in copy_pairs:" not in owner_source
    assert "for destination, source in commit_pairs:" not in owner_source


def test_r13_manifest_fails_closed_on_unapproved_or_partial_aliasing() -> None:
    bundle, replay, shadow_bundle, shadow_replay = _fixture(
        batch=2,
        capacity=8,
        device=torch.device("cpu"),
    )
    shadow_bundle.base.base.core.agents.energy = (
        bundle.base.base.core.agents.energy
    )
    with pytest.raises(TensorWorldContractError, match="unapproved.*alias"):
        inventory_c4_resident_transport_r13_v1(
            bundle, shadow_bundle, replay, shadow_replay
        )

    _, _, shadow_bundle, shadow_replay = _fixture(
        batch=2,
        capacity=8,
        device=torch.device("cpu"),
    )
    shadow_bundle.base.base.core.agents.energy = (
        bundle.base.base.core.agents.energy[:, :]
    )
    with pytest.raises(TensorWorldContractError, match="unapproved.*alias"):
        inventory_c4_resident_transport_r13_v1(
            bundle, shadow_bundle, replay, shadow_replay
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    os.environ.get(PROBE_ENV) != "1",
    reason="R13 transport probe is explicitly opt-in",
)
def test_r13_cuda_exact_copy_rollback_and_two_node_graph(tmp_path: Path) -> None:
    bundle, replay, shadow_bundle, shadow_replay = _fixture(
        batch=48,
        capacity=32,
        device=torch.device("cuda:0"),
    )
    inventory = inventory_c4_resident_transport_r13_v1(
        bundle, shadow_bundle, replay, shadow_replay
    )
    plan = materialize_c4_resident_transport_plan_r13_v1(inventory)
    extension = load_c4_resident_transport_r13_extension(
        build_directory=os.environ.get(BUILD_DIR_ENV, tmp_path / "build")
    )
    status = torch.zeros(48, dtype=torch.int32, device="cuda:0")
    source = _tensor_leaves(bundle, "bundle") | _tensor_leaves(replay, "replay")
    shadow = _tensor_leaves(shadow_bundle, "bundle") | _tensor_leaves(
        shadow_replay, "replay"
    )

    for path in inventory.covered_paths:
        shadow[path].view(torch.uint8).fill_(0xA5)
    extension.copy_to_shadow(
        plan.copy_source_addresses,
        plan.copy_destination_addresses,
        plan.copy_byte_counts,
    )
    torch.cuda.synchronize()
    for path in inventory.covered_paths:
        _assert_bytes_equal(shadow[path], source[path], path)

    for path in inventory.covered_paths:
        shadow[path].view(torch.uint8).bitwise_xor_(0x5A)
    expected = {path: shadow[path].clone() for path in inventory.covered_paths}
    extension.publish_if_all_ok(
        plan.publish_source_addresses,
        plan.publish_destination_addresses,
        plan.publish_byte_counts,
        status,
    )
    torch.cuda.synchronize()
    for path in inventory.covered_paths:
        _assert_bytes_equal(source[path], expected[path], path)

    before_failure = {
        path: source[path].clone() for path in inventory.covered_paths
    }
    for path in inventory.covered_paths:
        shadow[path].view(torch.uint8).bitwise_xor_(0x3C)
    status[17] = 65
    extension.publish_if_all_ok(
        plan.publish_source_addresses,
        plan.publish_destination_addresses,
        plan.publish_byte_counts,
        status,
    )
    torch.cuda.synchronize()
    for path in inventory.covered_paths:
        _assert_bytes_equal(source[path], before_failure[path], path)

    status.zero_()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        extension.copy_to_shadow(
            plan.copy_source_addresses,
            plan.copy_destination_addresses,
            plan.copy_byte_counts,
        )
        extension.publish_if_all_ok(
            plan.publish_source_addresses,
            plan.publish_destination_addresses,
            plan.publish_byte_counts,
            status,
        )
    graph.instantiate()
    assert extension.captured_graph_node_count(graph.raw_cuda_graph()) == 2
    assert inventory.unique_field_count > 2
