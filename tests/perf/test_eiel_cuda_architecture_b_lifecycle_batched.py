from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[2]
CUDA_SOURCE = ROOT / "engine/perf/native/cuda_architecture_b_lifecycle.cu"
INPUT_NAMES = (
    "slot_to_agent_id",
    "agent_id_to_slot",
    "alive_mask",
    "pos_y",
    "pos_x",
    "agent_grid",
    "energy",
    "health",
    "policy_energy_observation",
    "rest_streak",
    "action_cache_token",
    "action_cache_cursor",
    "action_cache_next",
    "pending_free_agent_ids",
    "pending_free_slots",
    "pending_free_count",
)


@pytest.fixture(scope="module")
def lifecycle_module(tmp_path_factory: pytest.TempPathFactory):
    cuda_source = CUDA_SOURCE.read_text(encoding="utf-8")
    cpp_source = r"""
    #include <torch/extension.h>

    std::vector<torch::Tensor> eiel_cuda_architecture_b_lifecycle_death_cull_v1(
        torch::Tensor slot_to_agent_id,
        torch::Tensor agent_id_to_slot,
        torch::Tensor alive_mask,
        torch::Tensor pos_y,
        torch::Tensor pos_x,
        torch::Tensor agent_grid,
        torch::Tensor energy,
        torch::Tensor health,
        torch::Tensor policy_energy_observation,
        torch::Tensor rest_streak,
        torch::Tensor action_cache_token,
        torch::Tensor action_cache_cursor,
        torch::Tensor action_cache_next,
        torch::Tensor pending_free_agent_ids,
        torch::Tensor pending_free_slots,
        torch::Tensor pending_free_count
    );

    std::vector<torch::Tensor>
    eiel_cuda_architecture_b_lifecycle_death_cull_batched_counted_v1(
        torch::Tensor slot_to_agent_id,
        torch::Tensor agent_id_to_slot,
        torch::Tensor alive_mask,
        torch::Tensor pos_y,
        torch::Tensor pos_x,
        torch::Tensor agent_grid,
        torch::Tensor energy,
        torch::Tensor health,
        torch::Tensor policy_energy_observation,
        torch::Tensor rest_streak,
        torch::Tensor action_cache_token,
        torch::Tensor action_cache_cursor,
        torch::Tensor action_cache_next,
        torch::Tensor pending_free_agent_ids,
        torch::Tensor pending_free_slots,
        torch::Tensor pending_free_count
    );
    """
    return load_inline(
        name=(
            "eiel_cuda_architecture_b_lifecycle_batched_stage2_test_"
            f"{sha256(cuda_source.encode()).hexdigest()[:12]}"
        ),
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=[
            "eiel_cuda_architecture_b_lifecycle_death_cull_v1",
            "eiel_cuda_architecture_b_lifecycle_death_cull_batched_counted_v1",
        ],
        build_directory=str(tmp_path_factory.mktemp("stage2-lifecycle-batched-build")),
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
    )


def _cell(kind: str) -> dict[str, torch.Tensor]:
    definitions = {
        "deaths": ([20, 5, 30, -1, 7, 50], [10, 0, 10, 0, 10, 0], [1, 1, 0, 0, 1, 0], 1),
        "none": ([60, 10, -1, 40, 80, 20], [10, 10, 0, 10, 10, 10], [1, 1, 0, 1, 1, 1], 2),
        "invalid": ([11, 22, 33, 44, -1, 55], [10, 0, 10, 10, 0, 10], [1, 1, 1, 1, 0, 1], 7),
        "overflow": ([12, 24, 36, 48, 72, -1], [10, 0, 10, 10, 10, 0], [1, 1, 0, 1, 1, 0], 5),
    }
    agent_ids, energy_values, health_values, pending_count = definitions[kind]
    slot_to_agent_id = torch.tensor(agent_ids, dtype=torch.int64)
    alive_mask = slot_to_agent_id.ge(0)
    pos_y = torch.full((6,), -1, dtype=torch.int32)
    pos_x = torch.full((6,), -1, dtype=torch.int32)
    agent_grid = torch.full((3, 3), -1, dtype=torch.int32)
    agent_id_to_slot = torch.full((128,), -1, dtype=torch.int32)
    for slot, agent_id in enumerate(agent_ids):
        if agent_id < 0:
            continue
        y, x = divmod(slot, 3)
        pos_y[slot] = y
        pos_x[slot] = x
        agent_grid[y, x] = agent_id
        agent_id_to_slot[agent_id] = slot

    pending_free_agent_ids = torch.arange(900, 906, dtype=torch.int64)
    pending_free_slots = torch.arange(100, 106, dtype=torch.int64)
    state = {
        "slot_to_agent_id": slot_to_agent_id,
        "agent_id_to_slot": agent_id_to_slot,
        "alive_mask": alive_mask,
        "pos_y": pos_y,
        "pos_x": pos_x,
        "agent_grid": agent_grid,
        "energy": torch.tensor(energy_values, dtype=torch.float32),
        "health": torch.tensor(health_values, dtype=torch.float32),
        "policy_energy_observation": torch.tensor(
            [float(value) if alive else torch.nan for value, alive in zip(energy_values, alive_mask)],
            dtype=torch.float64,
        ),
        "rest_streak": torch.arange(1, 7, dtype=torch.int32),
        "action_cache_token": torch.arange(101, 107, dtype=torch.int64),
        "action_cache_cursor": torch.arange(11, 17, dtype=torch.int32),
        "action_cache_next": torch.arange(21, 27, dtype=torch.int32),
        "pending_free_agent_ids": pending_free_agent_ids,
        "pending_free_slots": pending_free_slots,
        "pending_free_count": torch.tensor([pending_count], dtype=torch.int64),
    }
    return {name: tensor.cuda().contiguous() for name, tensor in state.items()}


def _invoke(function, state: dict[str, torch.Tensor]):
    return function(*(state[name] for name in INPUT_NAMES))


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.is_floating_point():
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    else:
        assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_batched_death_cull_matches_isolated_cells_and_preserves_isolation(
    lifecycle_module,
) -> None:
    source = [_cell(kind) for kind in ("deaths", "none", "invalid", "overflow")]
    isolated_states = [
        {name: tensor.clone() for name, tensor in cell.items()} for cell in source
    ]
    isolated_outputs = [
        _invoke(
            lifecycle_module.eiel_cuda_architecture_b_lifecycle_death_cull_v1,
            state,
        )
        for state in isolated_states
    ]
    batched_state = {
        name: torch.stack([cell[name] for cell in source]).contiguous()
        for name in INPUT_NAMES
    }
    batched_outputs = _invoke(
        lifecycle_module.eiel_cuda_architecture_b_lifecycle_death_cull_batched_counted_v1,
        batched_state,
    )
    torch.cuda.synchronize()

    for cell_index, expected_outputs in enumerate(isolated_outputs):
        for output_index, expected in enumerate(expected_outputs):
            _assert_exact(batched_outputs[output_index][cell_index], expected)
        for name in INPUT_NAMES:
            _assert_exact(batched_state[name][cell_index], isolated_states[cell_index][name])

    assert batched_outputs[6].flatten().tolist() == [0, 0, 1, 2]
    assert batched_outputs[3].flatten().tolist() == [3, 0, 0, 0]
    assert batched_outputs[0][0, :3].tolist() == [5, 30, 50]
    assert batched_outputs[1][0, :3].tolist() == [1, 2, 5]
    assert batched_outputs[2][0, :3].tolist() == [1, 2, 2]
    assert batched_outputs[4][0, :2].tolist() == [7, 20]
