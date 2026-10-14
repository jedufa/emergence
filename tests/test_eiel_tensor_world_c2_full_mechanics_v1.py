from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from engine.tensor_world.c2_backend import (
    TensorWorldC2ConfigV1,
    _config_tensors_v1,
    _native_state_v1,
    clone_c2_bundle_to_device_v1,
)
from engine.tensor_world.c2_fixture import make_c2_fixture_v1
from engine.tensor_world.c2_mechanics import C2_MECHANICS_V1
from engine.tensor_world.c2_native_loader import load_tensor_world_c2_extension
from engine.tensor_world.c2_state import (
    C2_OBSERVATION_WIDTH_V1,
    C2_POLICY_PACKED_WIDTH_V1,
    c2_state_tensor_names_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[1]


def test_c2_dependency_graph_is_frozen_before_implementation() -> None:
    graph = json.loads(
        (ROOT / "_artifacts/tensor_world_c0_c7_v3/c2/tensor_world_C2_dependency_graph.json").read_text()
    )
    assert graph["status"] == "FROZEN_BEFORE_IMPLEMENTATION"
    assert [row["id"] for row in graph["ordered_phases"]] == [
        "wdl.advance", "eed.settle_resident_resources",
        "patch.decay_density_recovery_regrowth", "jungle.predator_pressure",
        "effects.consume_pre_policy", "observation.assemble", "policy_and_action",
        "wdl.action_context", "world_events.commit", "agent.energy_settlement",
        "wdl.damage", "effects.consume_post_action",
    ]
    assert graph["eed_role"].startswith("SIMULATION_RESOURCE_MECHANICS_ONLY")


def test_c2_mechanics_are_static_owned_modules() -> None:
    assert tuple(mechanic.capability_id for mechanic in C2_MECHANICS_V1) == (
        "wdl_tensor_mechanic", "eed_tensor_mechanic",
        "patch_tensor_mechanic", "jungle_tensor_mechanic",
    )
    assert all(mechanic.capability_version == 1 for mechanic in C2_MECHANICS_V1)
    assert "effects.pre_policy_health_delta" in C2_MECHANICS_V1[-1].mutated_fields
    assert "effects.post_action_health_delta" in C2_MECHANICS_V1[0].mutated_fields


def test_c2_state_schema_and_fixture_are_fail_closed() -> None:
    state, _ = make_c2_fixture_v1(batch=1, capacity=8, active=4, width=8)
    state.validate()
    assert state.transactions.observations.shape[-1] == C2_OBSERVATION_WIDTH_V1
    assert state.policy.packed_weights.shape[-1] == C2_POLICY_PACKED_WIDTH_V1
    assert "WdlTensorStateV1.prior_cue" in c2_state_tensor_names_v1()
    assert "AgentEffectBufferV1.pre_policy_health_delta" in c2_state_tensor_names_v1()
    assert "WorldEventBufferV1.scent_emission" in c2_state_tensor_names_v1()


def test_c2_wdl_temporal_config_rejects_impossible_period() -> None:
    with pytest.raises(TensorWorldContractError, match="event period"):
        TensorWorldC2ConfigV1(
            wdl_delay_steps=12, wdl_active_steps=2, wdl_event_period_steps=14
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("TENSOR_WORLD_C2_PREBUILT"),
    reason="receipt-bound C2 CUDA extension is not configured",
)
def test_c2_modular_and_native_source_order_are_multitick_exact() -> None:
    source, config = make_c2_fixture_v1(batch=1, capacity=8, active=4, width=32)
    left = clone_c2_bundle_to_device_v1(source, torch.device("cuda:0"))
    right = clone_c2_bundle_to_device_v1(source, torch.device("cuda:0"))
    left_tensors, right_tensors = _native_state_v1(left), _native_state_v1(right)
    left_ci, left_cf = _config_tensors_v1(config, left)
    right_ci, right_cf = _config_tensors_v1(config, right)
    native = load_tensor_world_c2_extension()
    phases: list[tuple[int, int, int]] = []
    for tick in range(21):
        native.wdl_step(left_tensors, left_ci, left_cf)
        native.eed_step(left_tensors, left_ci, left_cf)
        native.patch_step(left_tensors, left_ci, left_cf)
        native.jungle_step(left_tensors, left_ci, left_cf)
        native.observe(left_tensors, left_ci, left_cf)
        native.policy_logits(left_tensors, left_ci, left_cf)
        native.select_actions(left_tensors, left_ci, left_cf)
        native.settle(left_tensors, left_ci, left_cf)
        native.oracle_tick(right_tensors, right_ci, right_cf)
        torch.cuda.synchronize()
        phases.append((
            int((left.wdl.cue > 0).sum()),
            int(((left.wdl.timer >= 0) & (left.wdl.cue <= 0) & (left.wdl.active <= 0)).sum()),
            int((left.wdl.active > 0).sum()),
        ))
        for index, (candidate, oracle) in enumerate(
            zip(left_tensors, right_tensors, strict=True)
        ):
            assert torch.equal(candidate, oracle), f"C2 tensor ABI index {index} diverged at tick {tick}"
    assert any(cue > 0 for cue, _, _ in phases)
    assert any(hidden > 0 for _, hidden, _ in phases)
    assert any(active > 0 for _, _, active in phases)
    assert int(left.wdl.next_absolute_step[0]) == 21
    assert bool(torch.isfinite(left.transactions.logits).all())
