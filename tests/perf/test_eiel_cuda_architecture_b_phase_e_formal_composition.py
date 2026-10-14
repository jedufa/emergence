from __future__ import annotations

from dataclasses import fields
import inspect
from typing import Mapping

import torch

from agents.batched_policy import forward_batched_mlp
from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_RUNTIME_MUTABLE_FIELD_NAMES,
    PHYSICAL_CONNECTION_SLOTS,
    PHYSICAL_MODULE_SLOTS,
    build_modular_storage_layout,
)
from perf.cuda_architecture_b_phase_e_formal_composition import (
    ACTION_WIDTH,
    MASKED_LOGIT_FLOOR,
    PHASE_E_FORMAL_COMPOSITION_EXCLUDED_OPERATIONS,
    PHASE_E_FORMAL_COMPOSITION_SCOPE,
    PhaseEFormalCompositionRaw,
    build_phase_e_formal_composition_fixture,
    execute_phase_e_cpu_formal_composition,
)


REST_ACTION = 5
MOVE_EAST_ACTION = 2
LOGICAL_MODULE_SLOTS = 12
LOGICAL_CONNECTION_SLOTS = 32


def _planes(
    *,
    module_count: int = 1,
    execution_slots: tuple[int, ...] | None = None,
) -> dict[str, torch.Tensor]:
    layout = build_modular_storage_layout(1)
    planes = {
        field.name: torch.full(
            field.shape,
            field.fill_value,
            dtype=field.torch_dtype,
        ).contiguous()
        for field in layout.fields
    }
    planes["agent_occupied"][0] = True
    planes["agent_id"][0] = 41
    planes["agent_slot_token"][0] = 9001
    planes["module_count"][0] = module_count
    planes["connection_count"][0] = 0
    if execution_slots is None:
        execution_slots = tuple(range(module_count))
    assert sorted(execution_slots) == list(range(module_count))
    for order, module_slot in enumerate(execution_slots):
        planes["module_execution_order"][0, order] = module_slot
        planes["dependency_topological_order"][0, order] = module_slot
    for module_slot in range(module_count):
        planes["module_occupied"][0, module_slot] = True
        planes["module_local_id_index"][0, module_slot] = 100 + module_slot
        planes["module_instance_id_index"][0, module_slot] = 200 + module_slot
        planes["module_homology_id_index"][0, module_slot] = 300 + module_slot
        planes["module_type_index"][0, module_slot] = 400 + module_slot
        planes["module_type_code"][0, module_slot] = 1
        planes["identity_origin_code"][0, module_slot] = 2
        planes["operator_code"][0, module_slot] = 1
        planes["integrator_parameters"][0, module_slot] = torch.tensor(
            [0.75, 0.5, 0.75, 0.5], dtype=torch.float32
        )
        planes["module_state_present"][0, module_slot] = True
        planes["module_state"][0, module_slot, 0] = 0.2 - 0.1 * module_slot
        planes["module_output_present"][0, module_slot] = True
        planes["module_previous_output"][0, module_slot, 0] = (
            0.1 - 0.05 * module_slot
        )
        planes["input_binding_count"][0, module_slot] = 1
        planes["input_semantic_code"][0, module_slot, 0] = module_slot % 2
        planes["input_semantic_id_index"][0, module_slot, 0] = 500 + module_slot
        planes["input_gain"][0, module_slot, 0] = 1.0 - 0.25 * module_slot
        planes["input_enabled"][0, module_slot, 0] = True
        planes["output_binding_count"][0, module_slot] = 1
        planes["output_semantic_code"][0, module_slot, 0] = module_slot
        planes["output_semantic_id_index"][0, module_slot, 0] = 600 + module_slot
        planes["output_action_index"][0, module_slot, 0] = (
            REST_ACTION if module_slot == 0 else MOVE_EAST_ACTION
        )
        planes["output_gain"][0, module_slot, 0] = 1.0
        planes["output_enabled"][0, module_slot, 0] = True
        planes["development_explicit"][0, module_slot] = True
        planes["development_stage_gains"][0, module_slot] = 1.0
    return planes


def _arguments(
    planes: dict[str, torch.Tensor],
) -> dict[str, object]:
    observations = torch.zeros((1, 27), dtype=torch.float32)
    observations[0, 25] = 0.8
    observations[0, 26] = 0.6
    hidden_size = 3
    policy_observations = torch.linspace(
        -0.5,
        0.5,
        27,
        dtype=torch.float32,
    ).reshape(1, 27)
    validity_mask = torch.tensor(
        [[True, True, True, True, True, True, False]],
        dtype=torch.bool,
    )
    authority_mask = torch.tensor(
        [[False, True, True, True, True, True, True]],
        dtype=torch.bool,
    )
    return {
        "planes": planes,
        "modular_observations": observations,
        "fatigue": torch.tensor([0.1], dtype=torch.float32),
        "stage_indices": torch.tensor([0], dtype=torch.int64),
        "policy_observations": policy_observations,
        "w1": torch.zeros((1, hidden_size, 27), dtype=torch.float32),
        "b1": torch.zeros((1, hidden_size), dtype=torch.float32),
        "w2": torch.zeros((1, hidden_size, hidden_size), dtype=torch.float32),
        "b2": torch.zeros((1, hidden_size), dtype=torch.float32),
        "w3": torch.zeros((1, ACTION_WIDTH, hidden_size), dtype=torch.float32),
        "b3": torch.tensor(
            [[0.2, 0.05, 0.04, 0.03, 0.02, 0.0, 0.5]],
            dtype=torch.float32,
        ),
        "attack_bias": torch.zeros((1,), dtype=torch.float32),
        "action_temperature": torch.ones((1,), dtype=torch.float32),
        "validity_mask": validity_mask,
        "authority_mask": authority_mask,
        "rng_stream_indices": {"environment": 23, "policy": 17},
    }


def _assert_exact(left: object, right: object, *, name: str = "root") -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), name
        assert left.dtype == right.dtype, name
        assert tuple(left.shape) == tuple(right.shape), name
        assert torch.equal(left, right), name
        return
    if isinstance(left, Mapping):
        assert isinstance(right, Mapping), name
        assert tuple(left) == tuple(right), name
        for key in left:
            _assert_exact(left[key], right[key], name=f"{name}.{key}")
        return
    if isinstance(left, (tuple, list)):
        assert isinstance(right, type(left)), name
        assert len(left) == len(right), name
        for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
            _assert_exact(
                left_value,
                right_value,
                name=f"{name}[{index}]",
            )
        return
    assert left == right, name


def _assert_raw_repeat_exact(
    left: PhaseEFormalCompositionRaw,
    right: PhaseEFormalCompositionRaw,
) -> None:
    for field in fields(PhaseEFormalCompositionRaw):
        _assert_exact(
            getattr(left, field.name),
            getattr(right, field.name),
            name=field.name,
        )


def test_canonical_receipt_fixture_matches_the_explicit_reference() -> None:
    expected = execute_phase_e_cpu_formal_composition(**_arguments(_planes()))
    actual = build_phase_e_formal_composition_fixture()

    _assert_raw_repeat_exact(actual, expected)


def test_cpu_one_tick_raw_composition_is_independently_recomputable() -> None:
    arguments = _arguments(_planes())

    result = execute_phase_e_cpu_formal_composition(**arguments)

    policy = result.policy_forward_inputs
    recomputed_base = forward_batched_mlp(
        policy["policy_observations"],
        policy["w1"],
        policy["b1"],
        policy["w2"],
        policy["b2"],
        policy["w3"],
        policy["b3"],
        attack_bias=policy["attack_bias"],
        action_temperature=policy["action_temperature"],
    )
    assert torch.equal(result.base_logits, recomputed_base)
    assert torch.equal(
        result.modular_action_residuals,
        result.modular_audit_tensors["action_residuals"],
    )
    assert torch.count_nonzero(result.modular_action_residuals) > 0
    assert torch.equal(
        result.composed_logits,
        result.base_logits + result.modular_action_residuals,
    )
    recomputed_mask = torch.logical_and(
        result.validity_mask_before,
        result.authority_mask_before,
    )
    assert torch.equal(result.effective_action_mask, recomputed_mask)
    recomputed_masked = torch.where(
        recomputed_mask,
        result.composed_logits,
        torch.full_like(result.composed_logits, MASKED_LOGIT_FLOOR),
    )
    assert torch.equal(result.masked_logits, recomputed_masked)
    assert torch.equal(
        result.deterministic_actions,
        recomputed_masked.argmax(dim=1).to(torch.int32),
    )
    assert result.deterministic_actions.tolist() == [REST_ACTION]
    assert torch.equal(result.validity_mask_before, result.validity_mask_after)
    assert torch.equal(result.authority_mask_before, result.authority_mask_after)


def test_cpu_one_tick_changes_only_owned_planes_and_preserves_inactive_reserved() -> None:
    source = _planes()
    source_before = {name: tensor.clone() for name, tensor in source.items()}

    result = execute_phase_e_cpu_formal_composition(**_arguments(source))

    assert len(result.plane_field_names) == 57
    assert set(result.runtime_mutable_plane_fields) == set(
        MODULAR_RUNTIME_MUTABLE_FIELD_NAMES
    )
    changed = {
        name
        for name in result.plane_field_names
        if not torch.equal(result.planes_before[name], result.planes_after[name])
    }
    assert {"module_state", "module_previous_output"} <= changed
    assert changed <= set(MODULAR_RUNTIME_MUTABLE_FIELD_NAMES)
    for name in result.plane_field_names:
        assert torch.equal(source[name], source_before[name]), name
        before = result.planes_before[name]
        after = result.planes_after[name]
        if before.ndim >= 2 and before.shape[1] == PHYSICAL_MODULE_SLOTS:
            assert torch.equal(before[:, 1:], after[:, 1:]), name
            assert torch.equal(
                before[:, LOGICAL_MODULE_SLOTS:],
                after[:, LOGICAL_MODULE_SLOTS:],
            ), name
        if before.ndim >= 2 and before.shape[1] == PHYSICAL_CONNECTION_SLOTS:
            assert torch.equal(before, after), name
            assert torch.equal(
                before[:, LOGICAL_CONNECTION_SLOTS:],
                after[:, LOGICAL_CONNECTION_SLOTS:],
            ), name
    for identity_name in (
        "agent_id",
        "agent_slot_token",
        "module_local_id_index",
        "module_instance_id_index",
        "module_homology_id_index",
    ):
        assert torch.equal(
            result.planes_before[identity_name],
            result.planes_after[identity_name],
        ), identity_name


def test_cpu_one_tick_is_repeat_exact_and_rng_neutral() -> None:
    first = execute_phase_e_cpu_formal_composition(**_arguments(_planes()))
    second = execute_phase_e_cpu_formal_composition(**_arguments(_planes()))

    _assert_raw_repeat_exact(first, second)
    assert first.rng_stream_indices_before == first.rng_stream_indices_after
    assert torch.equal(first.torch_rng_state_before, first.torch_rng_state_after)
    _assert_exact(first.numpy_rng_state_before, first.numpy_rng_state_after)
    assert first.python_rng_state_before == first.python_rng_state_after
    assert first.modular_receipt["rng_draws"] == 0
    assert int(first.modular_audit_tensors["rng_draw_count"]) == 0


def test_independent_module_order_is_adversarially_invariant() -> None:
    forward = execute_phase_e_cpu_formal_composition(
        **_arguments(_planes(module_count=2, execution_slots=(0, 1)))
    )
    reversed_order = execute_phase_e_cpu_formal_composition(
        **_arguments(_planes(module_count=2, execution_slots=(1, 0)))
    )

    for name in (
        "module_state",
        "module_previous_output",
        "development_effective_gain",
    ):
        assert torch.equal(forward.planes_after[name], reversed_order.planes_after[name])
    for name in (
        "base_logits",
        "modular_action_residuals",
        "composed_logits",
        "effective_action_mask",
        "masked_logits",
        "deterministic_actions",
    ):
        assert torch.equal(getattr(forward, name), getattr(reversed_order, name)), name
    assert not torch.equal(
        forward.planes_before["module_execution_order"],
        reversed_order.planes_before["module_execution_order"],
    )


def test_neutral_bank_composes_zero_once_and_changes_no_plane() -> None:
    result = execute_phase_e_cpu_formal_composition(
        **_arguments(_planes(module_count=0))
    )

    assert torch.equal(
        result.modular_action_residuals,
        torch.zeros_like(result.modular_action_residuals),
    )
    assert torch.equal(result.composed_logits, result.base_logits)
    for name in result.plane_field_names:
        assert torch.equal(result.planes_before[name], result.planes_after[name]), name
    assert result.modular_receipt["hidden_fallback_count"] == 0
    assert result.modular_receipt["simulation_executed"] is False
    assert result.modular_receipt["lifecycle_claimed"] is False


def test_api_excludes_every_out_of_scope_lifecycle_operation() -> None:
    assert PHASE_E_FORMAL_COMPOSITION_SCOPE == (
        "one_cell_one_tick_fixed_mlp_fixed_identity_no_lifecycle_v1"
    )
    assert PHASE_E_FORMAL_COMPOSITION_EXCLUDED_OPERATIONS == (
        "birth",
        "death",
        "cull",
        "refill",
        "turnover",
        "prune",
        "reuse",
        "migration",
    )
    parameters = inspect.signature(execute_phase_e_cpu_formal_composition).parameters
    assert not set(PHASE_E_FORMAL_COMPOSITION_EXCLUDED_OPERATIONS) & set(parameters)
