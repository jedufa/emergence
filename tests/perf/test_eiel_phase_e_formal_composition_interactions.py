from __future__ import annotations

from copy import deepcopy
import random

import numpy as np
import torch

from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_RUNTIME_MUTABLE_FIELD_NAMES,
)
from perf.cuda_architecture_b_modular_tick import execute_modular_tick
from perf.eiel_phase_e_modular_tick_fixture_v1 import (
    AGENT_COUNT,
    LOCAL_SIZE,
    _fixed_inputs,
)
from tests.perf.test_eiel_cuda_architecture_b_modular_tick import (
    _packed_fixture,
)


_MUTABLE_PLANES = frozenset(MODULAR_RUNTIME_MUTABLE_FIELD_NAMES)
_COST_PLANES = frozenset(
    {
        "structural_cost_counts",
        "structural_cost_components",
        "structural_cost_total",
    }
)


def _clone_planes() -> dict[str, torch.Tensor]:
    packed, _, _ = _packed_fixture()
    return {name: tensor.clone() for name, tensor in packed.tensors.items()}


def _tick_inputs(tick: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    observations, _, _, _ = _fixed_inputs()
    return (
        torch.from_numpy(observations[tick].copy()),
        torch.from_numpy(observations[tick, :, LOCAL_SIZE].copy()),
        torch.full((AGENT_COUNT,), tick, dtype=torch.int64),
    )


def _rng_snapshot() -> tuple[torch.Tensor, tuple, tuple]:
    return (
        torch.get_rng_state().clone(),
        deepcopy(np.random.get_state()),
        deepcopy(random.getstate()),
    )


def _assert_rng_unchanged(before: tuple[torch.Tensor, tuple, tuple]) -> None:
    torch_before, numpy_before, python_before = before
    numpy_after = np.random.get_state()
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert random.getstate() == python_before


def _execute_with_guards(
    planes: dict[str, torch.Tensor],
    *,
    tick: int = 0,
):
    before = {name: tensor.clone() for name, tensor in planes.items()}
    rng_before = _rng_snapshot()
    result = execute_modular_tick(planes, *_tick_inputs(tick))
    for name, expected in before.items():
        if name not in _MUTABLE_PLANES:
            assert torch.equal(planes[name], expected), name
    _assert_rng_unchanged(rng_before)
    assert result.receipt.rng_draws == 0
    return before, result


def _copy_first_input_binding(
    planes: dict[str, torch.Tensor],
    *,
    source: tuple[int, int],
    target: tuple[int, int],
) -> None:
    source_agent, source_module = source
    target_agent, target_module = target
    planes["input_binding_count"][target_agent, target_module] = 1
    for name in (
        "input_semantic_code",
        "input_semantic_id_index",
        "input_gain",
        "input_enabled",
    ):
        planes[name][target_agent, target_module, 0] = planes[name][
            source_agent, source_module, 0
        ]


def _copy_first_output_binding(
    planes: dict[str, torch.Tensor],
    *,
    source: tuple[int, int],
    target: tuple[int, int],
) -> None:
    source_agent, source_module = source
    target_agent, target_module = target
    planes["output_binding_count"][target_agent, target_module] = 1
    for name in (
        "output_semantic_code",
        "output_semantic_id_index",
        "output_action_index",
        "output_gain",
        "output_enabled",
    ):
        planes[name][target_agent, target_module, 0] = planes[name][
            source_agent, source_module, 0
        ]


def test_development_zero_gain_disables_a_nonvacuous_routed_gated_module() -> None:
    control = _clone_planes()
    disabled = _clone_planes()
    agent_slot, module_slot = 1, 0
    disabled["development_stage_gains"][agent_slot, module_slot].zero_()

    _, control_result = _execute_with_guards(control)
    disabled_before, disabled_result = _execute_with_guards(disabled)

    assert int(disabled["input_binding_count"][agent_slot, module_slot]) == 1
    assert bool(disabled["input_enabled"][agent_slot, module_slot, 0])
    assert int(disabled["output_binding_count"][agent_slot, module_slot]) == 1
    assert bool(disabled["output_enabled"][agent_slot, module_slot, 0])
    assert (
        disabled_result.audit_tensors["routed_observation"][agent_slot, module_slot]
        != 0.0
    )
    assert (
        control_result.audit_tensors["module_output"][agent_slot, module_slot]
        != 0.0
    )
    assert torch.count_nonzero(control_result.action_residuals[agent_slot]) > 0

    assert disabled["development_effective_gain"][agent_slot, module_slot] == 0.0
    assert torch.equal(
        disabled["module_state"][agent_slot, module_slot],
        disabled_before["module_state"][agent_slot, module_slot],
    )
    assert (
        disabled_result.audit_tensors["module_output"][agent_slot, module_slot]
        == 0.0
    )
    assert (
        int(
            disabled_result.audit_tensors["active_input_binding_count"][agent_slot]
        )
        == 0
    )
    assert (
        int(
            disabled_result.audit_tensors["active_output_binding_count"][agent_slot]
        )
        == 0
    )
    assert torch.count_nonzero(disabled_result.action_residuals[agent_slot]) == 0


def test_dependency_block_zeroes_an_enabled_output_route() -> None:
    control = _clone_planes()
    blocked = _clone_planes()
    agent_slot, root_slot, dependent_slot = 2, 0, 1
    for planes in (control, blocked):
        _copy_first_output_binding(
            planes,
            source=(1, 0),
            target=(agent_slot, dependent_slot),
        )
    blocked["development_stage_gains"][agent_slot, root_slot].zero_()

    _, control_result = _execute_with_guards(control)
    blocked_before, blocked_result = _execute_with_guards(blocked)

    assert bool(blocked["output_enabled"][agent_slot, dependent_slot, 0])
    assert int(blocked["output_binding_count"][agent_slot, dependent_slot]) == 1
    assert (
        control_result.audit_tensors["module_output"][agent_slot, dependent_slot]
        != 0.0
    )
    assert torch.count_nonzero(control_result.action_residuals[agent_slot]) > 0
    assert not bool(
        blocked_result.audit_tensors["development_dependency_ready"][
            agent_slot, dependent_slot
        ]
    )
    assert (
        blocked["development_effective_gain"][agent_slot, dependent_slot] == 0.0
    )
    assert torch.equal(
        blocked["module_state"][agent_slot, dependent_slot],
        blocked_before["module_state"][agent_slot, dependent_slot],
    )
    assert (
        blocked_result.audit_tensors["module_output"][agent_slot, dependent_slot]
        == 0.0
    )
    assert (
        int(blocked_result.audit_tensors["active_output_binding_count"][agent_slot])
        == 0
    )
    assert torch.count_nonzero(blocked_result.action_residuals[agent_slot]) == 0


def test_dormancy_zeroes_output_despite_an_active_nonzero_record() -> None:
    active = _clone_planes()
    dormant = _clone_planes()
    agent_slot, module_slot = 3, 0
    active["module_dormant"][agent_slot, module_slot] = False

    _, active_result = _execute_with_guards(active)
    dormant_before, dormant_result = _execute_with_guards(dormant)

    assert int(dormant["output_binding_count"][agent_slot, module_slot]) == 1
    assert bool(dormant["output_enabled"][agent_slot, module_slot, 0])
    assert (
        dormant_before["module_previous_output"][agent_slot, module_slot, 0]
        != 0.0
    )
    assert active_result.audit_tensors["module_output"][agent_slot, module_slot] != 0.0
    assert torch.count_nonzero(active_result.action_residuals[agent_slot]) > 0

    assert torch.equal(
        dormant["module_state"][agent_slot, module_slot],
        dormant_before["module_state"][agent_slot, module_slot],
    )
    assert dormant_result.audit_tensors["module_output"][agent_slot, module_slot] == 0.0
    assert (
        int(dormant_result.audit_tensors["active_output_binding_count"][agent_slot])
        == 0
    )
    assert torch.count_nonzero(dormant_result.action_residuals[agent_slot]) == 0


def test_input_routing_and_previous_connection_contribute_once_together() -> None:
    planes = _clone_planes()
    agent_slot, source_slot, target_slot = 2, 0, 1
    _copy_first_input_binding(
        planes,
        source=(0, 0),
        target=(agent_slot, target_slot),
    )

    before, result = _execute_with_guards(planes)
    routed_observation = result.audit_tensors["routed_observation"][
        agent_slot, target_slot
    ]
    routed_previous = result.audit_tensors["previous_tick_connection_input"][
        agent_slot, target_slot
    ]
    inputs = _tick_inputs()
    semantic_code = int(
        before["input_semantic_code"][agent_slot, target_slot, 0]
    )
    expected_routed_observation = (
        before["input_gain"][agent_slot, target_slot, 0]
        * inputs[0][agent_slot, LOCAL_SIZE + semantic_code]
    )
    expected_routed_previous = (
        before["connection_weight"][agent_slot, 0]
        * before["module_previous_output"][agent_slot, source_slot, 0]
    )
    assert routed_observation != 0.0
    assert routed_previous != 0.0
    assert torch.equal(routed_observation, expected_routed_observation)
    assert torch.equal(routed_previous, expected_routed_previous)

    parameters = before["integrator_parameters"][agent_slot, target_slot]
    combined_external = inputs[1][agent_slot] + routed_observation
    expected_state = torch.clamp(
        parameters[1] * before["module_state"][agent_slot, target_slot, 0]
        + parameters[2] * combined_external
        + routed_previous,
        -1.0,
        1.0,
    )
    assert torch.equal(
        result.audit_tensors["state_after"][agent_slot, target_slot],
        expected_state,
    )


def test_neutral_rows_are_exact_zero_on_every_dynamic_and_routing_surface() -> None:
    planes = _clone_planes()
    before, result = _execute_with_guards(planes)
    neutral = torch.logical_and(
        before["agent_occupied"],
        torch.logical_not(torch.any(before["module_occupied"], dim=1)),
    )
    assert int(neutral.sum()) == 60

    for name in (
        "state_before",
        "state_after",
        "module_output",
        "development_effective_gain",
        "routed_observation",
        "previous_tick_connection_input",
        "legacy_rest_residual",
        "routed_action_residual_before_rest_combine",
        "action_residuals",
    ):
        assert torch.count_nonzero(result.audit_tensors[name][neutral]) == 0, name
    for name in _MUTABLE_PLANES:
        assert torch.count_nonzero(planes[name][neutral]) == 0, name
    assert torch.count_nonzero(result.action_residuals[neutral]) == 0


def test_structural_cost_planes_are_reporting_only_under_counterfactual_values() -> None:
    baseline = _clone_planes()
    altered = _clone_planes()
    altered["structural_cost_counts"].add_(17)
    altered["structural_cost_components"].add_(0.75)
    altered["structural_cost_total"].add_(11.0)
    altered_cost_before = {
        name: altered[name].clone() for name in _COST_PLANES
    }
    rng_before = _rng_snapshot()

    baseline_result = execute_modular_tick(baseline, *_tick_inputs())
    altered_result = execute_modular_tick(altered, *_tick_inputs())

    assert torch.equal(altered_result.action_residuals, baseline_result.action_residuals)
    for name in baseline:
        if name not in _COST_PLANES:
            assert torch.equal(altered[name], baseline[name]), name
    for name in baseline_result.audit_tensors:
        if name not in _COST_PLANES:
            assert torch.equal(
                altered_result.audit_tensors[name],
                baseline_result.audit_tensors[name],
            ), name
    for name, expected in altered_cost_before.items():
        assert torch.equal(altered[name], expected), name
        assert torch.equal(altered_result.audit_tensors[name], expected), name
    _assert_rng_unchanged(rng_before)


def test_valid_noncanonical_module_order_has_the_canonical_tick_result() -> None:
    canonical = _clone_planes()
    permuted = _clone_planes()
    agent_slot = 2
    canonical_order = canonical["module_execution_order"][agent_slot, :2].clone()
    assert canonical_order[0] != canonical_order[1]
    permuted["module_execution_order"][agent_slot, :2] = canonical_order.flip(0)
    permuted_before = permuted["module_execution_order"].clone()
    rng_before = _rng_snapshot()

    canonical_result = execute_modular_tick(canonical, *_tick_inputs())
    permuted_result = execute_modular_tick(permuted, *_tick_inputs())

    assert torch.equal(permuted["module_execution_order"], permuted_before)
    assert torch.equal(
        permuted_result.action_residuals,
        canonical_result.action_residuals,
    )
    for name in canonical:
        if name != "module_execution_order":
            assert torch.equal(permuted[name], canonical[name]), name
    for name in canonical_result.audit_tensors:
        assert torch.equal(
            permuted_result.audit_tensors[name],
            canonical_result.audit_tensors[name],
        ), name
    _assert_rng_unchanged(rng_before)
