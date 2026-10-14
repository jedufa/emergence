from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[2]
CUDA_SOURCE = ROOT / "engine/perf/native/cuda_architecture_b_policy_cache.cu"
CAPACITY = 6
CACHE_CAPACITY = 3
CACHE_WIDTH = 2


@pytest.fixture(scope="module")
def policy_cache_module(tmp_path_factory: pytest.TempPathFactory):
    cpp_source = r"""
    #include <torch/extension.h>

    std::vector<torch::Tensor> eiel_cuda_architecture_b_policy_cache_service_v1(
        torch::Tensor active_agent_ids,
        torch::Tensor active_count,
        torch::Tensor slots,
        torch::Tensor slot_tokens,
        torch::Tensor action_cache_actions,
        torch::Tensor action_cache_token,
        torch::Tensor action_cache_cursor,
        torch::Tensor action_cache_next,
        int64_t cache_width
    );

    std::vector<torch::Tensor>
    eiel_cuda_architecture_b_policy_cache_service_batched_v1(
        torch::Tensor active_agent_ids,
        torch::Tensor active_count,
        torch::Tensor slots,
        torch::Tensor slot_tokens,
        torch::Tensor action_cache_actions,
        torch::Tensor action_cache_token,
        torch::Tensor action_cache_cursor,
        torch::Tensor action_cache_next,
        int64_t cache_width
    );
    """
    return load_inline(
        name="eiel_cuda_architecture_b_policy_cache_stage2_test_ext",
        cpp_sources=[cpp_source],
        cuda_sources=[CUDA_SOURCE.read_text(encoding="utf-8")],
        functions=[
            "eiel_cuda_architecture_b_policy_cache_service_v1",
            "eiel_cuda_architecture_b_policy_cache_service_batched_v1",
        ],
        build_directory=str(tmp_path_factory.mktemp("stage2-policy-cache-build")),
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
    )


def _cell(mode: str) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    slots = torch.tensor((4, 1, 3, 0, 0, 0), device=device, dtype=torch.int32)
    state = {
        "active_agent_ids": torch.tensor(
            (2, 0, 1, -1, -1, -1), device=device, dtype=torch.int64
        ),
        "active_count": torch.tensor([3], device=device, dtype=torch.int64),
        "slots": slots,
        "slot_tokens": torch.arange(100, 106, device=device, dtype=torch.int64),
        "action_cache_actions": torch.full(
            (CAPACITY, CACHE_CAPACITY), -1, device=device, dtype=torch.int32
        ),
        "action_cache_token": torch.full(
            (CAPACITY,), -1, device=device, dtype=torch.int64
        ),
        "action_cache_cursor": torch.full(
            (CAPACITY,), -1, device=device, dtype=torch.int32
        ),
        "action_cache_next": torch.full(
            (CAPACITY,), -1, device=device, dtype=torch.int32
        ),
    }
    hit_rows = 0 if mode == "miss" else (3 if mode == "hit" else 1)
    for row in range(hit_rows):
        slot = int(slots[row])
        state["action_cache_actions"][slot, :CACHE_WIDTH] = torch.tensor(
            (row + 1, row + 2), device=device, dtype=torch.int32
        )
        state["action_cache_token"][slot] = state["slot_tokens"][slot]
        state["action_cache_cursor"][slot] = 0
        state["action_cache_next"][slot] = row + 1
    return state


def _clone(cell: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in cell.items()}


def _invoke(function, state: dict[str, torch.Tensor]):
    return function(
        state["active_agent_ids"],
        state["active_count"],
        state["slots"],
        state["slot_tokens"],
        state["action_cache_actions"],
        state["action_cache_token"],
        state["action_cache_cursor"],
        state["action_cache_next"],
        CACHE_WIDTH,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_batched_policy_cache_matches_isolated_cells_and_preserves_isolation(
    policy_cache_module,
) -> None:
    source = [_cell("miss"), _cell("hit"), _cell("partial")]
    isolated_states = [_clone(cell) for cell in source]
    isolated_outputs = [
        _invoke(
            policy_cache_module.eiel_cuda_architecture_b_policy_cache_service_v1,
            state,
        )
        for state in isolated_states
    ]
    batched_state = {
        name: torch.stack([cell[name] for cell in source]).contiguous()
        for name in source[0]
    }
    batched_outputs = _invoke(
        policy_cache_module.eiel_cuda_architecture_b_policy_cache_service_batched_v1,
        batched_state,
    )

    for cell_index, expected in enumerate(isolated_outputs):
        for output_index, expected_tensor in enumerate(expected):
            assert torch.equal(
                batched_outputs[output_index][cell_index],
                expected_tensor,
            )
        for name in (
            "action_cache_actions",
            "action_cache_token",
            "action_cache_cursor",
            "action_cache_next",
        ):
            assert torch.equal(batched_state[name][cell_index], isolated_states[cell_index][name])

    assert batched_outputs[4].flatten().tolist() == [3, 0, 2]
    assert batched_outputs[3].flatten().tolist() == [0, 3, 1]
