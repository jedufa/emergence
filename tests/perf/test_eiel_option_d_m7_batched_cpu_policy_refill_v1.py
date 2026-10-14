from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.cpp_extension import load_inline

from engine.perf.cuda_architecture_b_child_policy import (
    pack_crossover_mask_lsb0_v1,
)


ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "engine/perf/cuda_architecture_b_resident_window_refill_v1.hpp"
M2_HEADER = ROOT / "engine/perf/cuda_architecture_b_option_d_m2_policy_shadow_v1.cuh"
M4_HEADER = ROOT / "engine/perf/cuda_architecture_b_option_d_m4_policy_refill_v1.cuh"


def _load_refill_module(*, m7_enabled: bool, m0_enabled: bool = False) -> Any:
    enabled = 1 if m7_enabled else 0
    telemetry = 1 if m0_enabled else 0
    digest = sha256(
        HEADER.read_bytes()
        + M2_HEADER.read_bytes()
        + M4_HEADER.read_bytes()
        + str(enabled).encode("ascii")
        + str(telemetry).encode("ascii")
    ).hexdigest()[:12]
    cpp_source = r"""
    #include <torch/extension.h>
    #include "cuda_architecture_b_resident_window_refill_v1.hpp"

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
                source_state,
                delta,
                mutation_mask,
                crossover_mask_packed,
                post_state_packed,
                semantic_tape,
                absolute_cursor,
                semantic_counter,
                window_base,
                policy_row_width,
                false,
                "dense",
                0.0,
                0.0
            );
    }

    torch::Tensor pack_batched(torch::Tensor mask) {
        return eiel::cuda_architecture_b::resident_window_refill_v1::
            pack_crossover_mask_lsb0_batched_v1(mask);
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
        module.def("refill_policy_window", &refill_policy_window);
        module.def("pack_batched", &pack_batched);
    }
    """
    numpy_random_lib = (
        Path(np.__file__).resolve().parent / "random/lib/libnpyrandom.a"
    )
    return load_inline(
        name=f"eiel_option_d_m7_refill_{enabled}_{telemetry}_{digest}",
        cpp_sources=[cpp_source],
        extra_include_paths=[str(ROOT / "engine/perf"), np.get_include()],
        extra_cflags=[
            "-O2",
            "-ffp-contract=off",
            f"-DEIEL_OPTION_D_M0_TELEMETRY_ENABLED={telemetry}",
            "-DEIEL_OPTION_D_M2_POLICY_SHADOW_ENABLED=0",
            "-DEIEL_OPTION_D_M4_DEVICE_POLICY_REFILL_ENABLED=0",
            f"-DEIEL_OPTION_D_M7_BATCHED_CPU_POLICY_REFILL_ENABLED={enabled}",
        ],
        extra_ldflags=[str(numpy_random_lib)],
        with_cuda=False,
        verbose=False,
    )


@pytest.fixture(scope="session")
def control_module() -> Any:
    return _load_refill_module(m7_enabled=False)


@pytest.fixture(scope="session")
def candidate_module() -> Any:
    return _load_refill_module(m7_enabled=True)


def _inputs(
    *,
    cells: int,
    episodes: int,
    width: int,
    cursor: int = 0,
    semantic_counter: int = 0,
    base: int = 0,
) -> dict[str, torch.Tensor]:
    states = []
    for cell in range(cells):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0xE1E1_0000 + cell)
        states.append(generator.get_state())
    return {
        "source": torch.stack(states).contiguous(),
        "delta": torch.full((cells, episodes, 1), 7.0, dtype=torch.float32),
        "mask": torch.full((cells, episodes, 1), True, dtype=torch.bool),
        "crossover": torch.full(
            (cells, episodes, (width + 7) // 8),
            0xA5,
            dtype=torch.uint8,
        ),
        "post": torch.empty((cells, episodes, 0), dtype=torch.uint8),
        "semantic": torch.full((cells, episodes), -1, dtype=torch.int64),
        "cursor": torch.full((cells,), cursor, dtype=torch.int64),
        "counter": torch.full(
            (cells,), semantic_counter, dtype=torch.int64
        ),
        "base": torch.full((cells,), base, dtype=torch.int64),
    }


def _call(module: Any, tensors: dict[str, torch.Tensor], width: int) -> dict[str, Any]:
    return dict(
        module.refill_policy_window(
            tensors["source"],
            tensors["delta"],
            tensors["mask"],
            tensors["crossover"],
            tensors["post"],
            tensors["semantic"],
            tensors["cursor"],
            tensors["counter"],
            tensors["base"],
            width,
        )
    )


def _assert_equal_outputs(
    control: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]
) -> None:
    for name in (
        "source",
        "delta",
        "mask",
        "crossover",
        "post",
        "semantic",
        "cursor",
        "counter",
        "base",
    ):
        assert torch.equal(control[name], candidate[name]), name


@pytest.mark.parametrize("width", [1, 7, 8, 9, 13, 63, 64, 65, 9_991])
def test_policy_lsb0_001_exact_bytes_and_zero_tail(
    candidate_module: Any, width: int
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(719 + width)
    mask = torch.rand((5, width), generator=generator).lt(0.5)
    actual = candidate_module.pack_batched(mask.contiguous())
    expected = torch.stack(
        [pack_crossover_mask_lsb0_v1(row.contiguous()) for row in mask]
    )
    assert torch.equal(actual, expected)
    if width % 8:
        unused_mask = ((1 << (8 - (width % 8))) - 1) << (width % 8)
        assert bool(torch.all(actual[:, -1].bitwise_and(unused_mask) == 0))


def test_policy_rng_xover_001_exact_batched_values_and_terminal_state() -> None:
    episodes = 124
    width = 9_991
    source = torch.Generator(device="cpu").manual_seed(733).get_state()
    row_generator = torch.Generator(device="cpu")
    row_generator.set_state(source)
    row_values = torch.stack(
        [
            torch.rand((width,), generator=row_generator, dtype=torch.float32)
            for _ in range(episodes)
        ]
    )
    batch_generator = torch.Generator(device="cpu")
    batch_generator.set_state(source)
    batch_values = torch.rand(
        (episodes, width), generator=batch_generator, dtype=torch.float32
    )
    assert torch.equal(batch_values, row_values)
    assert torch.equal(batch_generator.get_state(), row_generator.get_state())
    assert torch.equal(batch_values.lt(0.5), row_values.lt(0.5))


def test_policy_batch_001_physical_b48_exact_digest(
    control_module: Any, candidate_module: Any
) -> None:
    control = _inputs(cells=48, episodes=124, width=9_991)
    candidate = {name: value.clone() for name, value in control.items()}
    control_receipt = _call(control_module, control, 9_991)
    candidate_receipt = _call(candidate_module, candidate, 9_991)
    _assert_equal_outputs(control, candidate)
    assert control_receipt["option_d_m7_batched_cpu_policy_refill_enabled"] is False
    assert candidate_receipt["option_d_m7_batched_cpu_policy_refill_enabled"] is True
    assert candidate_receipt["option_d_m7_batched_cpu_policy_refill_used"] is True
    assert candidate_receipt["source_rng_state_preserved"] is True
    assert candidate_receipt["absolute_cursor_preserved"] is True
    assert candidate_receipt["semantic_draws_preserved"] is True


def test_policy_boundary_ckpt_replay_001_consecutive_refills_are_exact(
    control_module: Any, candidate_module: Any
) -> None:
    control = _inputs(cells=3, episodes=5, width=13, cursor=17, base=0)
    candidate = {name: value.clone() for name, value in control.items()}
    _call(control_module, control, 13)
    _call(candidate_module, candidate, 13)
    _assert_equal_outputs(control, candidate)
    checkpoint = {name: value.clone() for name, value in candidate.items()}

    for tensors in (control, candidate):
        tensors["delta"].fill_(91.0)
        tensors["mask"].fill_(True)
        tensors["crossover"].fill_(0xFF)
        tensors["semantic"].fill_(-91)
        tensors["cursor"].fill_(22)
        tensors["counter"].fill_(65)
    _call(control_module, control, 13)
    _call(candidate_module, candidate, 13)
    _assert_equal_outputs(control, candidate)

    resumed = {name: value.clone() for name, value in checkpoint.items()}
    resumed["cursor"].fill_(22)
    resumed["counter"].fill_(65)
    resumed["delta"].fill_(91.0)
    resumed["mask"].fill_(True)
    resumed["crossover"].fill_(0xFF)
    resumed["semantic"].fill_(-91)
    _call(candidate_module, resumed, 13)
    _assert_equal_outputs(candidate, resumed)
    assert bool(torch.all(candidate["crossover"][:, :, -1] < 32))


def test_m7_fails_closed_to_control_for_noncompact_mutation_surface(
    candidate_module: Any
) -> None:
    tensors = _inputs(cells=1, episodes=2, width=13)
    wrong_delta = torch.empty((1, 2, 13), dtype=torch.float32)
    wrong_mask = torch.empty((1, 2, 13), dtype=torch.bool)
    wrong_post = torch.empty((1, 2, 2_560), dtype=torch.uint8)
    receipt = dict(
        candidate_module.refill_policy_window(
            tensors["source"],
            wrong_delta,
            wrong_mask,
            tensors["crossover"],
            wrong_post,
            tensors["semantic"],
            tensors["cursor"],
            tensors["counter"],
            tensors["base"],
            13,
        )
    )
    assert receipt["option_d_m7_batched_cpu_policy_refill_enabled"] is True
    assert receipt["option_d_m7_batched_cpu_policy_refill_used"] is False
    assert receipt["option_d_m7_batched_cpu_policy_refill_authority"] == (
        "compiled_candidate_not_used_for_surface"
    )
