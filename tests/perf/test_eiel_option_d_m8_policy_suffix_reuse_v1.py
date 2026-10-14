from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "engine/perf/cuda_architecture_b_resident_window_refill_v1.hpp"


@pytest.fixture(scope="session")
def m8_module() -> Any:
    digest = sha256(HEADER.read_bytes()).hexdigest()[:12]
    source = r"""
    #include <torch/extension.h>
    #include "cuda_architecture_b_resident_window_refill_v1.hpp"

    pybind11::dict ensure_policy_window(
        torch::Tensor source_state,
        torch::Tensor delta,
        torch::Tensor mutation_mask,
        torch::Tensor crossover_mask_packed,
        torch::Tensor post_state_packed,
        torch::Tensor semantic_tape,
        torch::Tensor absolute_cursor,
        torch::Tensor semantic_counter,
        torch::Tensor window_base,
        int64_t policy_row_width,
        int64_t required_future_episodes
    ) {
        return eiel::cuda_architecture_b::resident_window_refill_v1::
            ensure_fixed_mlp_policy_window_v1(
                source_state, delta, mutation_mask, crossover_mask_packed,
                post_state_packed, semantic_tape, absolute_cursor,
                semantic_counter, window_base, policy_row_width, false,
                "dense", 0.0, 0.0, required_future_episodes
            );
    }

    pybind11::dict refill_policy_window(
        torch::Tensor source_state,
        torch::Tensor delta,
        torch::Tensor mutation_mask,
        torch::Tensor crossover_mask_packed,
        torch::Tensor post_state_packed,
        torch::Tensor semantic_tape,
        torch::Tensor absolute_cursor,
        torch::Tensor semantic_counter,
        torch::Tensor window_base,
        int64_t policy_row_width
    ) {
        return eiel::cuda_architecture_b::resident_window_refill_v1::
            refill_fixed_mlp_policy_window_v1(
                source_state, delta, mutation_mask, crossover_mask_packed,
                post_state_packed, semantic_tape, absolute_cursor,
                semantic_counter, window_base, policy_row_width, false,
                "dense", 0.0, 0.0
            );
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
        module.def("ensure_policy_window", &ensure_policy_window);
        module.def("refill_policy_window", &refill_policy_window);
    }
    """
    numpy_random_lib = Path(np.__file__).resolve().parent / "random/lib/libnpyrandom.a"
    return load_inline(
        name=f"eiel_option_d_m8_suffix_reuse_{digest}",
        cpp_sources=[source],
        extra_include_paths=[str(ROOT / "engine/perf"), np.get_include()],
        extra_cflags=[
            "-O0",
            "-DEIEL_OPTION_D_M0_TELEMETRY_ENABLED=1",
            "-DEIEL_OPTION_D_M2_POLICY_SHADOW_ENABLED=0",
            "-DEIEL_OPTION_D_M4_DEVICE_POLICY_REFILL_ENABLED=0",
            "-DEIEL_OPTION_D_M7_BATCHED_CPU_POLICY_REFILL_ENABLED=1",
            "-DEIEL_OPTION_D_M8_RESIDENT_POLICY_SUFFIX_REUSE_ENABLED=1",
        ],
        extra_ldflags=[str(numpy_random_lib)],
        with_cuda=False,
        verbose=False,
    )


def _inputs(cursors: list[int], *, capacity: int = 8, width: int = 17) -> dict[str, torch.Tensor]:
    cells = len(cursors)
    states = []
    for cell in range(cells):
        generator = torch.Generator(device="cpu").manual_seed(1701 + cell)
        states.append(generator.get_state())
    return {
        "source": torch.stack(states).contiguous(),
        "delta": torch.full((cells, capacity, 1), 7.0),
        "mask": torch.full((cells, capacity, 1), True, dtype=torch.bool),
        "crossover": torch.full(
            (cells, capacity, (width + 7) // 8), 0xA5, dtype=torch.uint8
        ),
        "post": torch.empty((cells, capacity, 0), dtype=torch.uint8),
        "semantic": torch.full((cells, capacity), -1, dtype=torch.int64),
        "cursor": torch.tensor(cursors, dtype=torch.int64),
        "counter": torch.tensor([31 + i for i in range(cells)], dtype=torch.int64),
        "base": torch.zeros(cells, dtype=torch.int64),
    }


def _call(module: Any, tensors: dict[str, torch.Tensor], required: int, width: int = 17) -> dict[str, Any]:
    return dict(
        module.ensure_policy_window(
            tensors["source"], tensors["delta"], tensors["mask"],
            tensors["crossover"], tensors["post"], tensors["semantic"],
            tensors["cursor"], tensors["counter"], tensors["base"],
            width, required,
        )
    )


def test_m8_reuses_exact_capacity_boundary_without_mutating_resident_state(
    m8_module: Any,
) -> None:
    tensors = _inputs([5, 4, 0], capacity=8)
    before = {name: value.clone() for name, value in tensors.items()}
    receipt = _call(m8_module, tensors, required=3)
    assert receipt["physical_refill_performed"] is False
    assert receipt["resident_window_reused"] is True
    assert receipt["minimum_remaining_episodes"] == 3
    assert receipt["remaining_episodes_by_cell"] == [3, 4, 8]
    assert receipt["refill_trigger_cell_count"] == 0
    assert receipt["option_d_m8_resident_policy_suffix_reuse_used"] is True
    assert receipt["option_d_m7_batched_cpu_policy_refill_used"] is False
    assert receipt["host_to_device_operations_lower_bound"] == 0
    for name, value in tensors.items():
        assert torch.equal(value, before[name]), name


def test_m8_b48_skew_refills_atomically_when_one_cell_is_one_row_short(
    m8_module: Any,
) -> None:
    tensors = _inputs([0] * 47 + [6], capacity=8)
    receipt = _call(m8_module, tensors, required=3)
    assert receipt["physical_refill_performed"] is True
    assert receipt["resident_window_reused"] is False
    assert receipt["minimum_remaining_episodes"] == 2
    assert receipt["remaining_episodes_by_cell"] == [8] * 47 + [2]
    assert receipt["refill_trigger_cell_count"] == 1
    assert receipt["option_d_m8_resident_policy_suffix_reuse_used"] is False
    assert receipt["option_d_m7_batched_cpu_policy_refill_used"] is True
    assert torch.equal(tensors["base"], tensors["cursor"])
    assert bool(torch.all(tensors["semantic"] == 17))


def test_m8_forced_refill_is_exact_m7_fallback_including_rng_and_packed_bytes(
    m8_module: Any,
) -> None:
    candidate = _inputs([0] * 47 + [6], capacity=8)
    direct = {name: value.clone() for name, value in candidate.items()}

    candidate_receipt = _call(m8_module, candidate, required=3)
    direct_receipt = dict(
        m8_module.refill_policy_window(
            direct["source"],
            direct["delta"],
            direct["mask"],
            direct["crossover"],
            direct["post"],
            direct["semantic"],
            direct["cursor"],
            direct["counter"],
            direct["base"],
            17,
        )
    )

    assert candidate_receipt["physical_refill_performed"] is True
    assert candidate_receipt["option_d_m7_batched_cpu_policy_refill_used"] is True
    assert direct_receipt["option_d_m7_batched_cpu_policy_refill_used"] is True
    for name in candidate:
        assert torch.equal(candidate[name], direct[name]), name


def test_m8_longer_cursor_sequence_reuses_then_forces_physical_refill(
    m8_module: Any,
) -> None:
    tensors = _inputs([0, 0], capacity=8)
    initial_state = tensors["source"].clone()
    for cursors in ([0, 0], [2, 1], [5, 4]):
        tensors["cursor"].copy_(torch.tensor(cursors, dtype=torch.int64))
        receipt = _call(m8_module, tensors, required=3)
        assert receipt["resident_window_reused"] is True
        assert torch.equal(tensors["source"], initial_state)
        assert torch.equal(tensors["base"], torch.zeros(2, dtype=torch.int64))

    tensors["cursor"].copy_(torch.tensor([6, 5], dtype=torch.int64))
    receipt = _call(m8_module, tensors, required=3)
    assert receipt["physical_refill_performed"] is True
    assert receipt["refill_trigger_cell_count"] == 1
    assert torch.equal(tensors["source"], initial_state)
    assert torch.equal(tensors["base"], tensors["cursor"])


@pytest.mark.parametrize("required", [0, 3])
def test_m8_zero_and_max_birth_coverage_checks(
    m8_module: Any, required: int
) -> None:
    tensors = _inputs([8 if required == 0 else 5], capacity=8)
    receipt = _call(m8_module, tensors, required=required)
    assert receipt["resident_window_reused"] is True
    assert receipt["minimum_remaining_episodes"] == (0 if required == 0 else 3)


def test_m8_fails_closed_for_cursor_outside_resident_window(m8_module: Any) -> None:
    tensors = _inputs([9], capacity=8)
    with pytest.raises(RuntimeError, match="outside the resident physical tape"):
        _call(m8_module, tensors, required=3)
