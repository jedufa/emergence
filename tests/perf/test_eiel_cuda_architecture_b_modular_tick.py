from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from evolution.structural_live_modules_v1 import execute_live_modules_batch
from evolution.structural_observation_routing_v1 import route_observations_batch
from evolution.structural_output_routing_v1 import compile_output_routing_batch
from perf.cuda_architecture_b_modular_state_abi import (
    COST_COUNT_FIELDS,
    MODULAR_STORAGE_PROJECTION_PREFIX,
    pack_modular_storage,
)
from perf.cuda_architecture_b_modular_tick import (
    ACTION_WIDTH,
    MODULAR_TICK_LANE,
    ModularTickError,
    execute_modular_tick,
)
from perf.eiel_phase_e_modular_tick_fixture_v1 import (
    AGENT_COUNT,
    EXPANDED_B2_CAPS,
    LOCAL_SIZE,
    NEUTRAL_AGENT_COUNT,
    STRUCTURED_AGENT_COUNT,
    TICK_AGES,
    _build_population,
    _config,
    _fixed_inputs,
)


_MUTABLE_PLANES = {
    "module_state_present",
    "module_state",
    "module_output_present",
    "module_previous_output",
    "development_latch_present",
    "development_latch_state",
    "development_effective_gain_present",
    "development_effective_gain",
}


def _packed_fixture():
    alive, modules = _build_population()
    records = []
    for index, agent in enumerate(alive):
        records.append(
            {
                "agent_id": int(agent.agent_id),
                "agent_slot_token": 10_000 + index,
                "genome": {
                    "schema_version": "modular_genome_v1",
                    "genes": {},
                    "modules": deepcopy(agent.genome.modules),
                    "connections": deepcopy(agent.genome.connections),
                    "b2_caps": deepcopy(EXPANDED_B2_CAPS),
                    "b2_invalid_policy": "reject",
                },
                "structural_live_module_state": deepcopy(
                    agent.structural_live_module_state
                ),
                "structural_live_module_outputs": deepcopy(
                    agent.structural_live_module_outputs
                ),
                "structural_development_latch_by_instance_id": {},
                "structural_development_effective_gain_by_instance_id": {},
                "gated_recurrent_enabled": True,
            }
        )
    packed = pack_modular_storage(
        records,
        agent_capacity=AGENT_COUNT,
        profile_id="b2-16x48-v1",
    )
    return packed, alive, modules


def _cpu_fixture_tick(alive, observations: np.ndarray, *, age: int):
    cfg = _config()
    for agent in alive:
        agent.stats.survival_steps = int(age)
    routed = route_observations_batch(
        alive,
        observations,
        local_size=LOCAL_SIZE,
        sim_cfg=cfg,
    )
    rest = execute_live_modules_batch(
        alive,
        observations[:, LOCAL_SIZE],
        sim_cfg=cfg,
        device="cpu",
        routing_batches=routed,
    )
    output = compile_output_routing_batch(
        alive,
        sim_cfg=cfg,
        legacy_rest_residuals=rest,
    )
    return routed, output


def _module_instance_id(packed, agent_slot: int, module_slot: int) -> str:
    identity_index = int(
        packed.tensors["module_instance_id_index"][agent_slot, module_slot]
    )
    return packed.identity_table[identity_index]


def _assert_f32_exact(actual: torch.Tensor, expected: float) -> None:
    expected_tensor = torch.tensor(np.float32(expected), dtype=torch.float32)
    assert torch.equal(actual.detach().reshape(()), expected_tensor)


def test_three_ticks_match_committed_cpu_fixture_exactly() -> None:
    packed, alive, _ = _packed_fixture()
    observations, _, _, _ = _fixed_inputs()
    static_before = {
        name: tensor.clone()
        for name, tensor in packed.tensors.items()
        if name not in _MUTABLE_PLANES
    }
    initial_cost_counts = packed.tensors["structural_cost_counts"].clone()
    initial_cost_components = packed.tensors["structural_cost_components"].clone()
    initial_cost_total = packed.tensors["structural_cost_total"].clone()

    for tick, age in enumerate(TICK_AGES):
        routed, output = _cpu_fixture_tick(alive, observations[tick], age=age)
        result = execute_modular_tick(
            packed.tensors,
            torch.from_numpy(observations[tick].copy()),
            torch.from_numpy(observations[tick, :, LOCAL_SIZE].copy()),
            torch.full((AGENT_COUNT,), tick, dtype=torch.int64),
        )

        assert torch.equal(
            result.action_residuals,
            torch.from_numpy(output.residuals.copy()),
        )
        assert int(result.audit_tensors["structured_agent_count"]) == STRUCTURED_AGENT_COUNT
        assert int(result.audit_tensors["neutral_agent_count"]) == NEUTRAL_AGENT_COUNT
        for agent_slot in range(STRUCTURED_AGENT_COUNT):
            module_count = int(packed.tensors["module_count"][agent_slot])
            telemetry = alive[agent_slot].structural_live_module_telemetry
            cost_counts = telemetry["structural_cost"]["counts"]
            assert int(result.audit_tensors["allocated_module_count"][agent_slot]) == (
                cost_counts["allocated_module"]
            )
            assert int(result.audit_tensors["executing_integrator_count"][agent_slot]) == (
                cost_counts["executing_integrator"]
            )
            assert int(result.audit_tensors["executing_gated_count"][agent_slot]) == (
                cost_counts["executing_gated"]
            )
            assert int(result.audit_tensors["evaluated_connection_count"][agent_slot]) == (
                cost_counts["evaluated_connection"]
            )
            assert int(result.audit_tensors["active_input_binding_count"][agent_slot]) == (
                cost_counts["active_input_binding"]
            )
            assert int(result.audit_tensors["active_output_binding_count"][agent_slot]) == (
                cost_counts["active_output_binding"]
            )
            assert int(result.audit_tensors["active_dependency_edge_count"][agent_slot]) == (
                cost_counts["active_dependency_edge"]
            )
            assert int(result.audit_tensors["dormant_module_count"][agent_slot]) == (
                cost_counts["dormant_module_maintenance"]
            )
            for module_slot in range(module_count):
                instance_id = _module_instance_id(packed, agent_slot, module_slot)
                _assert_f32_exact(
                    packed.tensors["module_state"][agent_slot, module_slot, 0],
                    alive[agent_slot].structural_live_module_state[instance_id],
                )
                _assert_f32_exact(
                    packed.tensors["module_previous_output"][agent_slot, module_slot, 0],
                    alive[agent_slot].structural_live_module_outputs[instance_id],
                )
                _assert_f32_exact(
                    result.audit_tensors["routed_observation"][agent_slot, module_slot],
                    routed[agent_slot].aggregate_by_instance_id.get(instance_id, 0.0),
                )
                _assert_f32_exact(
                    packed.tensors["development_effective_gain"][agent_slot, module_slot],
                    alive[agent_slot].structural_development_effective_gain_by_instance_id[
                        instance_id
                    ],
                )
                expected_latch = bool(
                    alive[agent_slot].structural_development_latch_by_instance_id.get(
                        instance_id, False
                    )
                )
                assert bool(
                    packed.tensors["development_latch_state"][agent_slot, module_slot]
                ) is expected_latch
        assert torch.equal(
            result.audit_tensors["structural_cost_counts"], initial_cost_counts
        )
        assert torch.equal(
            result.audit_tensors["structural_cost_components"],
            initial_cost_components,
        )
        assert torch.equal(
            result.audit_tensors["structural_cost_total"], initial_cost_total
        )

    for name, expected in static_before.items():
        assert torch.equal(packed.tensors[name], expected), name


def test_dependency_latch_dormancy_and_previous_tick_connection_are_explicit() -> None:
    packed, alive, modules = _packed_fixture()
    observations, _, _, _ = _fixed_inputs()
    root_id = str(modules["prerequisite"]["module_instance_id"])
    dependent_id = str(modules["dependent"]["module_instance_id"])
    dormant_id = str(modules["dormant"]["module_instance_id"])
    dependent_slot = next(
        slot
        for slot in range(int(packed.tensors["module_count"][2]))
        if _module_instance_id(packed, 2, slot) == dependent_id
    )
    root_slot = next(
        slot
        for slot in range(int(packed.tensors["module_count"][2]))
        if _module_instance_id(packed, 2, slot) == root_id
    )
    dormant_slot = next(
        slot
        for slot in range(int(packed.tensors["module_count"][3]))
        if _module_instance_id(packed, 3, slot) == dormant_id
    )

    for tick, age in enumerate(TICK_AGES):
        _cpu_fixture_tick(alive, observations[tick], age=age)
        result = execute_modular_tick(
            packed.tensors,
            torch.from_numpy(observations[tick].copy()),
            torch.from_numpy(observations[tick, :, LOCAL_SIZE].copy()),
            torch.full((AGENT_COUNT,), tick, dtype=torch.int64),
        )
        assert bool(result.audit_tensors["development_dependency_ready"][2, root_slot])
        if tick == 2:
            assert packed.tensors["development_effective_gain"][2, root_slot] == 0.0
            assert packed.tensors["development_effective_gain"][2, dependent_slot] == 1.0
            assert bool(packed.tensors["development_latch_state"][2, dependent_slot])
            assert result.audit_tensors["previous_tick_connection_input"][2, dependent_slot] == 0.0
            assert int(result.audit_tensors["evaluated_connection_count"][2]) == 1
        assert packed.tensors["module_state"][3, dormant_slot, 0] == np.float32(0.75)
        assert packed.tensors["module_previous_output"][3, dormant_slot, 0] == 0.0


def test_output_action_index_is_authoritative_and_rest_is_combined_once() -> None:
    baseline, _, _ = _packed_fixture()
    changed, _, _ = _packed_fixture()
    observations, _, _, _ = _fixed_inputs()
    active = changed.tensors["output_enabled"]
    changed.tensors["output_semantic_code"][active] = 127

    baseline_result = execute_modular_tick(
        baseline.tensors,
        torch.from_numpy(observations[0].copy()),
        torch.from_numpy(observations[0, :, LOCAL_SIZE].copy()),
        torch.zeros((AGENT_COUNT,), dtype=torch.int64),
    )
    changed_result = execute_modular_tick(
        changed.tensors,
        torch.from_numpy(observations[0].copy()),
        torch.from_numpy(observations[0, :, LOCAL_SIZE].copy()),
        torch.zeros((AGENT_COUNT,), dtype=torch.int64),
    )

    assert torch.equal(baseline_result.action_residuals, changed_result.action_residuals)
    audit = baseline_result.audit_tensors
    routed_rest = audit["routed_action_residual_before_rest_combine"][:, 5]
    expected_rest = torch.where(
        routed_rest == 0.0,
        audit["legacy_rest_residual"],
        torch.clamp(audit["legacy_rest_residual"] + routed_rest, -0.25, 0.25),
    )
    assert torch.equal(baseline_result.action_residuals[:, 5], expected_rest)


def test_prefixed_resident_plane_set_executes_without_resize() -> None:
    packed, _, _ = _packed_fixture()
    observations, _, _, _ = _fixed_inputs()
    resident = {
        f"{MODULAR_STORAGE_PROJECTION_PREFIX}{name}": tensor.clone()
        for name, tensor in packed.tensors.items()
    }
    resident["alive_mask"] = torch.ones((AGENT_COUNT,), dtype=torch.bool)
    shapes_before = {name: tuple(tensor.shape) for name, tensor in resident.items()}

    result = execute_modular_tick(
        resident,
        torch.from_numpy(observations[0].copy()),
        torch.from_numpy(observations[0, :, LOCAL_SIZE].copy()),
        torch.zeros((AGENT_COUNT,), dtype=torch.int64),
    )

    assert result.planes is resident
    assert result.receipt.field_namespace == MODULAR_STORAGE_PROJECTION_PREFIX
    assert result.action_residuals.shape == (AGENT_COUNT, ACTION_WIDTH)
    assert {name: tuple(tensor.shape) for name, tensor in resident.items()} == shapes_before


def test_padding_and_static_planes_are_preserved() -> None:
    packed, _, _ = _packed_fixture()
    observations, _, _, _ = _fixed_inputs()
    packed.tensors["module_state"][0, 15, 0] = 0.8125
    packed.tensors["module_previous_output"][0, 15, 0] = -0.625
    static_before = {
        name: tensor.clone()
        for name, tensor in packed.tensors.items()
        if name not in _MUTABLE_PLANES
    }

    execute_modular_tick(
        packed.tensors,
        torch.from_numpy(observations[0].copy()),
        torch.from_numpy(observations[0, :, LOCAL_SIZE].copy()),
        torch.zeros((AGENT_COUNT,), dtype=torch.int64),
    )

    assert packed.tensors["module_state"][0, 15, 0] == np.float32(0.8125)
    assert packed.tensors["module_previous_output"][0, 15, 0] == np.float32(-0.625)
    for name, expected in static_before.items():
        assert torch.equal(packed.tensors[name], expected), name


@pytest.mark.parametrize(
    "damage",
    [
        "partial",
        "mixed",
        "plane_dtype",
        "plane_shape",
        "observation_dtype",
        "observation_shape",
        "fatigue_dtype",
        "fatigue_shape",
        "stage_dtype",
        "stage_shape",
        "action_width",
    ],
)
def test_malformed_plane_namespace_shape_and_dtype_fail_closed(damage: str) -> None:
    packed, _, _ = _packed_fixture()
    observations, _, _, _ = _fixed_inputs()
    planes = {name: tensor.clone() for name, tensor in packed.tensors.items()}
    obs = torch.from_numpy(observations[0].copy())
    fatigue = torch.from_numpy(observations[0, :, LOCAL_SIZE].copy())
    stages = torch.zeros((AGENT_COUNT,), dtype=torch.int64)
    action_width = ACTION_WIDTH
    if damage == "partial":
        del planes["module_state"]
    elif damage == "mixed":
        planes[f"{MODULAR_STORAGE_PROJECTION_PREFIX}module_state"] = planes[
            "module_state"
        ].clone()
    elif damage == "plane_dtype":
        planes["module_state"] = planes["module_state"].to(dtype=torch.float64)
    elif damage == "plane_shape":
        planes["module_state"] = planes["module_state"][:, :-1].contiguous()
    elif damage == "observation_dtype":
        obs = obs.to(dtype=torch.float64)
    elif damage == "observation_shape":
        obs = obs[:, :-1].contiguous()
    elif damage == "fatigue_dtype":
        fatigue = fatigue.to(dtype=torch.float64)
    elif damage == "fatigue_shape":
        fatigue = fatigue[:-1].contiguous()
    elif damage == "stage_dtype":
        stages = stages.to(dtype=torch.int32)
    elif damage == "stage_shape":
        stages = stages[:-1].contiguous()
    else:
        action_width = ACTION_WIDTH - 1

    with pytest.raises(ModularTickError):
        execute_modular_tick(
            planes,
            obs,
            fatigue,
            stages,
            action_width=action_width,
        )


def test_receipt_and_audit_are_tensor_only_zero_rng_nonclaims() -> None:
    packed, _, _ = _packed_fixture()
    observations, _, _, _ = _fixed_inputs()
    result = execute_modular_tick(
        packed.tensors,
        torch.from_numpy(observations[0].copy()),
        torch.from_numpy(observations[0, :, LOCAL_SIZE].copy()),
        torch.zeros((AGENT_COUNT,), dtype=torch.int64),
    )

    assert result.receipt.lane == MODULAR_TICK_LANE
    assert result.receipt.rng_draws == 0
    assert result.receipt.device_to_host_synchronizations == 0
    assert result.receipt.hidden_fallback_count == 0
    assert result.receipt.simulation_executed is False
    assert result.receipt.checkpoint_closure_claimed is False
    assert result.receipt.lifecycle_claimed is False
    assert result.receipt.campaign_claimed is False
    assert result.audit_tensors
    assert all(isinstance(value, torch.Tensor) for value in result.audit_tensors.values())
    assert result.audit_tensors["rng_draw_count"].dtype == torch.int64
    assert int(result.audit_tensors["rng_draw_count"]) == 0
    assert tuple(COST_COUNT_FIELDS) == (
        "allocated_module",
        "executing_integrator",
        "executing_gated",
        "evaluated_connection",
        "active_input_binding",
        "active_output_binding",
        "active_dependency_edge",
        "dormant_module_maintenance",
        "serialized_genome_byte",
        "saturated_or_rejected_activity",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tiny_cuda_tick_matches_cpu_tensor_results() -> None:
    cpu_packed, _, _ = _packed_fixture()
    gpu_packed, _, _ = _packed_fixture()
    observations, _, _, _ = _fixed_inputs()
    cpu_result = execute_modular_tick(
        cpu_packed.tensors,
        torch.from_numpy(observations[0].copy()),
        torch.from_numpy(observations[0, :, LOCAL_SIZE].copy()),
        torch.zeros((AGENT_COUNT,), dtype=torch.int64),
    )
    gpu_planes = {name: tensor.cuda() for name, tensor in gpu_packed.tensors.items()}
    gpu_result = execute_modular_tick(
        gpu_planes,
        torch.from_numpy(observations[0].copy()).cuda(),
        torch.from_numpy(observations[0, :, LOCAL_SIZE].copy()).cuda(),
        torch.zeros((AGENT_COUNT,), dtype=torch.int64, device="cuda"),
    )

    torch.testing.assert_close(
        gpu_result.action_residuals.cpu(),
        cpu_result.action_residuals,
        rtol=0.0,
        atol=2.0e-7,
    )
    torch.testing.assert_close(
        gpu_planes["module_state"].cpu(),
        cpu_packed.tensors["module_state"],
        rtol=0.0,
        atol=2.0e-7,
    )
