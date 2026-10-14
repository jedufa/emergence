from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from perf.cuda_architecture_b_generation import execute_cuda_generation
from tests.perf.test_eiel_cuda_architecture_b_generation import (
    CAPACITY,
    REST_ACTION,
    RIGHT_ACTION,
    SLOTS,
    _fixture,
    _generation_config,
)


@pytest.fixture(scope="module")
def generation_build(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("archb-cuda-generation-action-history-build")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_generation_updates_exact_action_history_before_action_time_death(
    generation_build: Path,
) -> None:
    tensors, _ = _fixture(patch=False, policy_action=REST_ACTION)
    episode0_k = 20
    first_slot, repeat_slot, oscillation_death_slot, switch_slot = map(int, SLOTS)
    inactive_slots = sorted(set(range(CAPACITY)) - set(map(int, SLOTS)))
    assert inactive_slots == [2, 5]

    seeded = {
        "last_action": [-1, RIGHT_ACTION, 1, REST_ACTION, RIGHT_ACTION, 2],
        "last_last_action": [-1, RIGHT_ACTION, 4, RIGHT_ACTION, REST_ACTION, 6],
        "action_switches": [10, 11, 12, 13, 14, 15],
        "action_oscillations": [20, 21, 22, 23, 24, 25],
        "episode0_switches": [30, 31, 32, 33, 34, 35],
        "episode0_oscillations": [40, 41, 42, 43, 44, 45],
    }
    for name, values in seeded.items():
        tensors[name] = torch.tensor(values, device="cuda", dtype=torch.int32)

    tensors["survival_steps"][oscillation_death_slot] = episode0_k - 1
    tensors["survival_steps"][switch_slot] = episode0_k
    tensors["energy"][oscillation_death_slot] = 0.01
    tensors["policy_energy_observation"][oscillation_death_slot] = 0.01
    config = replace(
        _generation_config(patch=False),
        steps=1,
        episode0_k=episode0_k,
        active_mechanics_enabled=False,
    )

    result = execute_cuda_generation(
        tensors,
        config,
        build_directory=generation_build,
    )

    assert result.actions_by_tick_slot[0, list(map(int, SLOTS))].cpu().tolist() == [
        REST_ACTION,
        REST_ACTION,
        REST_ACTION,
        REST_ACTION,
    ]
    assert result.lifecycle_death_counts_by_tick.cpu().tolist() == [1]
    assert result.lifecycle_death_slots_by_tick[0, 0].item() == oscillation_death_slot
    assert not tensors["alive_mask"][oscillation_death_slot].item()

    expected = {
        "last_action": [REST_ACTION, REST_ACTION, 1, REST_ACTION, REST_ACTION, 2],
        "last_last_action": [-1, RIGHT_ACTION, 4, REST_ACTION, RIGHT_ACTION, 6],
        "action_switches": [10, 12, 12, 13, 15, 15],
        "action_oscillations": [20, 21, 22, 23, 25, 25],
        "episode0_switches": [30, 31, 32, 33, 35, 35],
        "episode0_oscillations": [40, 41, 42, 43, 45, 45],
    }
    for name, values in expected.items():
        assert tensors[name].cpu().tolist() == values, name

    assert tensors["last_last_action"][first_slot].item() == -1
    assert tensors["action_switches"][repeat_slot].item() == seeded[
        "action_switches"
    ][repeat_slot]
    for slot in inactive_slots:
        for name, values in seeded.items():
            assert tensors[name][slot].item() == values[slot], (name, slot)
