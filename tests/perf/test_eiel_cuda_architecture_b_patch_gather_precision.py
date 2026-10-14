from __future__ import annotations

import numpy as np
import pytest
import torch

from env.gridworld_pkg.constants import ACT_GATHER, EMPTY, FOOD
from perf.cuda_architecture_b import CudaActionConfig, ResidentCudaActionState


FOOD_AMOUNT_BITS = 0x3F002699
EXPECTED_SCENT_PENDING_BITS = 0x3CCD0A8E


def _float32_from_bits(bits: int) -> np.float32:
    return np.asarray([bits], dtype=np.uint32).view(np.float32)[0]


def _float32_bits(value: object) -> int:
    return int(np.asarray([value], dtype=np.float32).view(np.uint32)[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_patch_gather_uses_cpu_double_boundary_for_pending_scent() -> None:
    food_amount = _float32_from_bits(FOOD_AMOUNT_BITS)
    expected_scent_pending = np.float32(float(food_amount) * 0.05)
    float32_only_scent_pending = np.float32(np.float32(0.05) * food_amount)
    assert _float32_bits(expected_scent_pending) == EXPECTED_SCENT_PENDING_BITS
    assert _float32_bits(float32_only_scent_pending) != EXPECTED_SCENT_PENDING_BITS

    owner = ResidentCudaActionState.from_arrays(
        agent_ids=np.asarray([0], dtype=np.int32),
        agent_pos_y=np.asarray([0], dtype=np.int32),
        agent_pos_x=np.asarray([0], dtype=np.int32),
        agent_grid=np.asarray([[0]], dtype=np.int32),
        food_grid=np.asarray([[FOOD]], dtype=np.int32),
        energies=np.asarray([1.0], dtype=np.float32),
        healths=np.asarray([1.0], dtype=np.float32),
        patch_enabled=True,
        patch_food_amount=np.asarray([[food_amount]], dtype=np.float32),
        patch_scent_emit_pending=np.zeros((1, 1), dtype=np.float32),
        patch_waste_emit_pending=np.zeros((1, 1), dtype=np.float32),
        patch_movement_cost=np.ones((1, 1), dtype=np.float32),
    )
    config = CudaActionConfig(
        move_cost=0.0,
        gather_cost=0.0,
        rest_cost=0.0,
        attack_cost=0.0,
        max_energy=10.0,
        food_energy=1.0,
        patch_food_gather_amount=1.0,
        patch_food_presence_threshold=0.0,
        patch_scent_gather_emit=0.05,
    )

    receipt = owner.execute_scripted_generation(
        np.asarray([[ACT_GATHER]], dtype=np.int32),
        config,
    )
    actual = owner.export_boundary()

    actual_pending = actual["patch_scent_emit_pending"][0, 0]
    assert _float32_bits(actual_pending) == EXPECTED_SCENT_PENDING_BITS
    assert actual_pending == expected_scent_pending
    assert (
        _float32_bits(actual["patch_scent_gather_queued_total"])
        == EXPECTED_SCENT_PENDING_BITS
    )
    assert _float32_bits(actual["patch_food_harvest_total"]) == FOOD_AMOUNT_BITS
    assert _float32_bits(actual["food_gained"][0]) == FOOD_AMOUNT_BITS
    assert actual["patch_food_amount"][0, 0] == np.float32(0.0)
    assert actual["food_grid"][0, 0] == EMPTY
    assert actual["patch_waste_emit_pending"][0, 0] == np.float32(0.0)
    assert actual["energy"][0] == np.float32(np.float32(1.0) + food_amount)
    assert not bool(actual["died_mask"][0])
    assert receipt["hidden_fallback"] is False
    assert receipt["native_generation_kernel_launches"] == 1
    assert receipt["python_tick_loops"] == 0
