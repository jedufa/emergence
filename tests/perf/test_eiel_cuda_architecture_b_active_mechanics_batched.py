from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.cpp_extension import load_inline


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Architecture B batched active mechanics requires CUDA",
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_CUDA = (
    ROOT / "engine" / "perf" / "native" / "cuda_architecture_b_active_mechanics.cu"
)
TENSOR_FIELDS = (
    "slot_to_agent_id",
    "alive_mask",
    "actions_by_slot",
    "energy",
    "health",
    "injury",
    "age",
    "policy_energy_observation",
    "rest_streak",
    "survival_steps",
    "gru_upkeep_cost",
    "cost_charge_scalar",
    "cost_charge_maturation_scalar",
    "drives",
    "drive_inertia",
    "drive_noise",
    "prev_drive_delta",
    "drive_volatility",
    "drive_jitter",
    "episode0_drive_jitter_sum",
    "last_energy_delta",
    "last_health_delta",
    "last_injury_delta",
    "selfharm_events",
    "episode0_selfharm_events",
    "pre_energy",
    "pre_health",
    "pre_injury",
    "drive_noise_samples",
    "drive_noise_sample_valid",
)
MUTABLE_FIELDS = (
    "energy",
    "health",
    "policy_energy_observation",
    "rest_streak",
    "survival_steps",
    "drives",
    "prev_drive_delta",
    "drive_volatility",
    "drive_jitter",
    "episode0_drive_jitter_sum",
    "last_energy_delta",
    "last_health_delta",
    "last_injury_delta",
    "selfharm_events",
    "episode0_selfharm_events",
)
CONTROL_ARGS = (
    True,
    1,
    0.25,
    0.10,
    1.0,
    2,
    4,
    20.0,
    1.0,
    0.08,
    0.02,
    0.18,
    0.03,
    0.05,
    0.10,
    0.01,
    0.22,
    0.03,
)


@pytest.fixture(scope="module")
def cuda_module(tmp_path_factory: pytest.TempPathFactory) -> Any:
    cpp_source = r"""
    #include <torch/extension.h>

    #define EIEL_ACTIVE_MECHANICS_ARGS \
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, \
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, \
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, \
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, \
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, \
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, \
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, \
        torch::Tensor, torch::Tensor, bool, int64_t, double, double, \
        double, int64_t, int64_t, double, double, double, \
        double, double, double, double, double, double, double, double

    std::vector<torch::Tensor> eiel_cuda_architecture_b_active_mechanics_v1(
        EIEL_ACTIVE_MECHANICS_ARGS
    );
    std::vector<torch::Tensor>
    eiel_cuda_architecture_b_active_mechanics_batched_counted_v1(
        EIEL_ACTIVE_MECHANICS_ARGS
    );
    """
    build_directory = tmp_path_factory.mktemp("archb-active-mechanics-batched-build")
    return load_inline(
        name="eiel_cuda_architecture_b_active_mechanics_batched_test",
        cpp_sources=[cpp_source],
        cuda_sources=[SOURCE_CUDA.read_text(encoding="utf-8")],
        functions=[
            "eiel_cuda_architecture_b_active_mechanics_v1",
            "eiel_cuda_architecture_b_active_mechanics_batched_counted_v1",
        ],
        build_directory=str(build_directory),
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
    )


def _cell_state(cell: int) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    slot_to_agent_id = (
        torch.tensor([11, 12, -1, -1], dtype=torch.int64, device=device)
        if cell == 0
        else torch.tensor([21, -1, 23, -1], dtype=torch.int64, device=device)
    )
    alive_mask = slot_to_agent_id >= 0
    actions = torch.tensor([5, 4, 6, 5], dtype=torch.int32, device=device)
    energy = torch.tensor(
        [4.0 + cell, 3.0, 0.0, 0.0], dtype=torch.float32, device=device
    )
    if cell == 1:
        energy[2] = 2.5
    health = torch.where(
        alive_mask,
        torch.tensor(1.0, dtype=torch.float32, device=device),
        torch.tensor(0.0, dtype=torch.float32, device=device),
    )
    zeros_f = torch.zeros(4, dtype=torch.float32, device=device)
    zeros_i = torch.zeros(4, dtype=torch.int32, device=device)
    return {
        "slot_to_agent_id": slot_to_agent_id,
        "alive_mask": alive_mask,
        "actions_by_slot": actions,
        "energy": energy,
        "health": health,
        "injury": zeros_f.clone(),
        "age": zeros_i.clone(),
        "policy_energy_observation": energy.to(torch.float64),
        "rest_streak": torch.tensor([1, 0, 0, 0], dtype=torch.int32, device=device),
        "survival_steps": torch.tensor([1, 2, 0, 0], dtype=torch.int32, device=device),
        "gru_upkeep_cost": torch.tensor(
            [0.05, 0.0, 0.02, 0.0], dtype=torch.float32, device=device
        ),
        "cost_charge_scalar": torch.tensor(
            [0.10, 0.20, 0.05, 0.0], dtype=torch.float32, device=device
        ),
        "cost_charge_maturation_scalar": torch.tensor(
            [0.03, 0.04, 0.02, 0.0], dtype=torch.float32, device=device
        ),
        "drives": torch.tensor(
            [[0.2, 0.1, 0.3, 0.0]] * 4, dtype=torch.float32, device=device
        ),
        "drive_inertia": torch.full((4,), 0.85, dtype=torch.float32, device=device),
        "drive_noise": zeros_f.clone(),
        "prev_drive_delta": torch.zeros((4, 4), dtype=torch.float32, device=device),
        "drive_volatility": zeros_f.clone(),
        "drive_jitter": zeros_f.clone(),
        "episode0_drive_jitter_sum": zeros_f.clone(),
        "last_energy_delta": zeros_f.clone(),
        "last_health_delta": zeros_f.clone(),
        "last_injury_delta": zeros_f.clone(),
        "selfharm_events": zeros_i.clone(),
        "episode0_selfharm_events": zeros_i.clone(),
        "pre_energy": energy + 0.25,
        "pre_health": health.clone(),
        "pre_injury": zeros_f.clone(),
        "drive_noise_samples": torch.zeros(
            (4, 4), dtype=torch.float32, device=device
        ),
        "drive_noise_sample_valid": torch.zeros(
            4, dtype=torch.bool, device=device
        ),
    }


def _args(state: dict[str, torch.Tensor]) -> list[object]:
    return [*(state[name] for name in TENSOR_FIELDS), *CONTROL_ARGS]


def _stack(cells: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        name: torch.stack([cell[name] for cell in cells]).contiguous()
        for name in TENSOR_FIELDS
    }


def test_batched_active_mechanics_matches_isolated_cells_exactly(
    cuda_module: Any,
) -> None:
    source_cells = [_cell_state(0), _cell_state(1)]
    expected_cells = [
        {name: tensor.clone() for name, tensor in cell.items()}
        for cell in source_cells
    ]
    expected_results = [
        cuda_module.eiel_cuda_architecture_b_active_mechanics_v1(*_args(cell))
        for cell in expected_cells
    ]

    batched = _stack(source_cells)
    status, counts = (
        cuda_module.eiel_cuda_architecture_b_active_mechanics_batched_counted_v1(
            *_args(batched)
        )
    )
    torch.cuda.synchronize()

    assert torch.equal(status, torch.cat([result[0] for result in expected_results]))
    assert torch.equal(counts, torch.stack([result[1] for result in expected_results]))
    for cell_index, expected in enumerate(expected_cells):
        for name in MUTABLE_FIELDS:
            torch.testing.assert_close(
                batched[name][cell_index],
                expected[name],
                rtol=0,
                atol=0,
                equal_nan=True,
            )


def test_batched_active_mechanics_failure_is_cell_local(cuda_module: Any) -> None:
    source_cells = [_cell_state(0), _cell_state(1)]
    expected_good = {
        name: tensor.clone() for name, tensor in source_cells[0].items()
    }
    expected_status, expected_counts = (
        cuda_module.eiel_cuda_architecture_b_active_mechanics_v1(
            *_args(expected_good)
        )
    )
    batched = _stack(source_cells)
    batched["alive_mask"][1, 1] = True
    failed_before = {
        name: batched[name][1].clone() for name in MUTABLE_FIELDS
    }

    status, counts = (
        cuda_module.eiel_cuda_architecture_b_active_mechanics_batched_counted_v1(
            *_args(batched)
        )
    )
    torch.cuda.synchronize()

    assert status.tolist() == [int(expected_status.item()), 1]
    assert torch.equal(counts[0], expected_counts)
    assert torch.count_nonzero(counts[1]).item() == 0
    for name in MUTABLE_FIELDS:
        torch.testing.assert_close(
            batched[name][0],
            expected_good[name],
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        torch.testing.assert_close(
            batched[name][1],
            failed_before[name],
            rtol=0,
            atol=0,
            equal_nan=True,
        )
