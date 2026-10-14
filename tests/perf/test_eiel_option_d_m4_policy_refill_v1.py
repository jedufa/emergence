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
    packed_crossover_row_bytes_v1,
)


ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "engine/perf/cuda_architecture_b_option_d_m4_policy_refill_v1.cuh"
CUDA_SOURCE = ROOT / "engine/perf/cuda_architecture_b_option_d_m4_policy_refill_v1.cu"
M2_HEADER = ROOT / "engine/perf/cuda_architecture_b_option_d_m2_policy_shadow_v1.cuh"
M2_CUDA_SOURCE = ROOT / "engine/perf/cuda_architecture_b_option_d_m2_policy_shadow_v1.cu"
REPLAY_HEADER = ROOT / "engine/perf/cuda_architecture_b_torch_mt19937_replay_v1.cuh"
STATE_HEADER = ROOT / "engine/perf/cuda_architecture_b_torch_rng_state_pack_v1.cuh"
REFILL_HEADER = ROOT / "engine/perf/cuda_architecture_b_resident_window_refill_v1.hpp"


@pytest.fixture(scope="session")
def m4_module() -> Any:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the M4 policy refill generator")

    digest = sha256(
        HEADER.read_bytes()
        + CUDA_SOURCE.read_bytes()
        + REPLAY_HEADER.read_bytes()
        + STATE_HEADER.read_bytes()
    ).hexdigest()[:12]
    cpp_source = r"""
    #include <torch/extension.h>
    #include "cuda_architecture_b_option_d_m4_policy_refill_v1.cuh"

    std::vector<int64_t> generate_compact_policy_window(
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
        auto result =
            eiel::cuda_architecture_b::option_d_m4_policy_refill_v1::
                generate_compact_policy_window_v1(
                    source_state,
                    delta,
                    mutation_mask,
                    crossover_mask_packed,
                    post_state_packed,
                    semantic_tape,
                    absolute_cursor,
                    semantic_counter,
                    window_base,
                    policy_row_width
                );
        return {
            result.generated_rows,
            result.generated_bytes,
            result.invalid_state_count,
            result.invalid_counter_count,
        };
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
        module.def(
            "generate_compact_policy_window",
            &generate_compact_policy_window
        );
    }
    """
    return load_inline(
        name=f"eiel_option_d_m4_policy_refill_v1_{digest}",
        cpp_sources=[cpp_source],
        cuda_sources=[CUDA_SOURCE.read_text(encoding="utf-8")],
        extra_include_paths=[str(ROOT / "engine/perf")],
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=[
            "-O2",
            "--expt-relaxed-constexpr",
            "--fmad=false",
            "--prec-div=true",
            "--prec-sqrt=true",
        ],
        with_cuda=True,
        verbose=False,
    )


def _load_refill_module(*, device_refill_enabled: bool) -> Any:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the M4 refill integration")

    enabled_value = 1 if device_refill_enabled else 0
    digest = sha256(
        REFILL_HEADER.read_bytes()
        + HEADER.read_bytes()
        + CUDA_SOURCE.read_bytes()
        + M2_HEADER.read_bytes()
        + M2_CUDA_SOURCE.read_bytes()
        + REPLAY_HEADER.read_bytes()
        + STATE_HEADER.read_bytes()
        + str(enabled_value).encode("ascii")
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

    pybind11::dict refill_mutation_policy_window(
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
                true,
                "dense",
                1.0,
                0.125
            );
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
        module.def("refill_policy_window", &refill_policy_window);
        module.def(
            "refill_mutation_policy_window",
            &refill_mutation_policy_window
        );
    }
    """
    cuda_sources = [CUDA_SOURCE.read_text(encoding="utf-8")]
    return load_inline(
        name=f"eiel_option_d_m4_refill_{enabled_value}_{digest}",
        cpp_sources=[cpp_source],
        cuda_sources=cuda_sources,
        extra_include_paths=[str(ROOT / "engine/perf"), np.get_include()],
        extra_cflags=[
            "-O2",
            "-ffp-contract=off",
            "-DEIEL_OPTION_D_M2_POLICY_SHADOW_ENABLED=0",
            f"-DEIEL_OPTION_D_M4_DEVICE_POLICY_REFILL_ENABLED={enabled_value}",
        ],
        extra_cuda_cflags=[
            "-O2",
            "--expt-relaxed-constexpr",
            "--fmad=false",
            "--prec-div=true",
            "--prec-sqrt=true",
            "-DEIEL_OPTION_D_M2_POLICY_SHADOW_ENABLED=0",
            f"-DEIEL_OPTION_D_M4_DEVICE_POLICY_REFILL_ENABLED={enabled_value}",
        ],
        with_cuda=True,
        verbose=False,
    )


@pytest.fixture(scope="session")
def refill_m4_on_module() -> Any:
    return _load_refill_module(device_refill_enabled=True)


@pytest.fixture(scope="session")
def refill_m4_off_module() -> Any:
    return _load_refill_module(device_refill_enabled=False)


def _seed_state(seed: int) -> torch.Tensor:
    return torch.Generator(device="cpu").manual_seed(seed).get_state().clone()


def _expected_policy_segment(
    *,
    seeds: list[int],
    episodes: int,
    width: int,
) -> torch.Tensor:
    crossover_rows: list[torch.Tensor] = []
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        cell_rows: list[torch.Tensor] = []
        for _episode in range(episodes):
            crossover = torch.rand(
                (width,),
                generator=generator,
                dtype=torch.float32,
            ).lt(0.5)
            cell_rows.append(pack_crossover_mask_lsb0_v1(crossover).cpu())
        crossover_rows.append(torch.stack(cell_rows).contiguous())
    return torch.stack(crossover_rows).contiguous()


def _empty_refill_inputs(
    *,
    seeds: list[int],
    episodes: int,
    width: int,
    cursor: int = 0,
    semantic_counter: int = 0,
    base: int = 0,
) -> dict[str, torch.Tensor]:
    cells = len(seeds)
    row_bytes = packed_crossover_row_bytes_v1(width)
    device = torch.device("cuda")
    return {
        "source": torch.stack([_seed_state(seed) for seed in seeds])
        .contiguous()
        .to(device),
        "delta": torch.empty((cells, episodes, 1), dtype=torch.float32, device=device),
        "mask": torch.empty((cells, episodes, 1), dtype=torch.bool, device=device),
        "crossover": torch.empty(
            (cells, episodes, row_bytes),
            dtype=torch.uint8,
            device=device,
        ),
        "post": torch.empty((cells, episodes, 0), dtype=torch.uint8, device=device),
        "semantic": torch.empty((cells, episodes), dtype=torch.int64, device=device),
        "cursor": torch.full((cells,), cursor, dtype=torch.int64, device=device),
        "counter": torch.full(
            (cells,),
            semantic_counter,
            dtype=torch.int64,
            device=device,
        ),
        "base": torch.full((cells,), base, dtype=torch.int64, device=device),
    }


def _empty_mutation_refill_inputs(
    *,
    seeds: list[int],
    episodes: int,
    width: int,
) -> dict[str, torch.Tensor]:
    cells = len(seeds)
    row_bytes = packed_crossover_row_bytes_v1(width)
    device = torch.device("cuda")
    return {
        "source": torch.stack([_seed_state(seed) for seed in seeds])
        .contiguous()
        .to(device),
        "delta": torch.empty(
            (cells, episodes, width),
            dtype=torch.float32,
            device=device,
        ),
        "mask": torch.empty((cells, episodes, width), dtype=torch.bool, device=device),
        "crossover": torch.empty(
            (cells, episodes, row_bytes),
            dtype=torch.uint8,
            device=device,
        ),
        "post": torch.empty((cells, episodes, 2560), dtype=torch.uint8, device=device),
        "semantic": torch.empty((cells, episodes), dtype=torch.int64, device=device),
        "cursor": torch.zeros((cells,), dtype=torch.int64, device=device),
        "counter": torch.zeros((cells,), dtype=torch.int64, device=device),
        "base": torch.zeros((cells,), dtype=torch.int64, device=device),
    }


def _call_generate(module: Any, inputs: dict[str, torch.Tensor], width: int) -> list[int]:
    return module.generate_compact_policy_window(
        inputs["source"],
        inputs["delta"],
        inputs["mask"],
        inputs["crossover"],
        inputs["post"],
        inputs["semantic"],
        inputs["cursor"],
        inputs["counter"],
        inputs["base"],
        width,
    )


@pytest.mark.parametrize("width", [625, 1249, 9415])
def test_m4_device_policy_refill_matches_cpu_prepared_compact_policy_segment(
    m4_module: Any,
    width: int,
) -> None:
    seeds = [42, 0xFFFF_FFFF_FFFF_FFFF, 123456789]
    episodes = 3
    inputs = _empty_refill_inputs(
        seeds=seeds,
        episodes=episodes,
        width=width,
        cursor=17,
        semantic_counter=99,
        base=7,
    )
    source_before = inputs["source"].clone()
    cursor_before = inputs["cursor"].clone()
    counter_before = inputs["counter"].clone()
    expected = _expected_policy_segment(seeds=seeds, episodes=episodes, width=width)

    result = _call_generate(m4_module, inputs, width)
    torch.cuda.synchronize()

    assert result[0] == len(seeds) * episodes
    assert result[2:] == [0, 0]
    assert torch.equal(inputs["crossover"].cpu(), expected)
    assert torch.equal(inputs["delta"].cpu(), torch.zeros_like(inputs["delta"].cpu()))
    assert not inputs["mask"].cpu().any()
    assert torch.equal(
        inputs["semantic"].cpu(),
        torch.full((len(seeds), episodes), width, dtype=torch.int64),
    )
    assert torch.equal(inputs["base"].cpu(), cursor_before.cpu())
    assert torch.equal(inputs["source"], source_before)
    assert torch.equal(inputs["cursor"], cursor_before)
    assert torch.equal(inputs["counter"], counter_before)


def test_m4_device_policy_refill_rejects_noncanonical_state(m4_module: Any) -> None:
    width = 625
    inputs = _empty_refill_inputs(seeds=[42], episodes=1, width=width)
    inputs["source"][0, 24 + 4] = 1

    result = _call_generate(m4_module, inputs, width)
    torch.cuda.synchronize()

    assert result == [0, 0, 1, 0]


def test_m4_device_policy_refill_rejects_invalid_cursor(m4_module: Any) -> None:
    width = 625
    inputs = _empty_refill_inputs(seeds=[42], episodes=1, width=width, cursor=3, base=4)

    result = _call_generate(m4_module, inputs, width)
    torch.cuda.synchronize()

    assert result == [0, 0, 0, 1]


def test_refill_policy_window_m4_is_default_off_and_cpu_path_receipted(
    refill_m4_off_module: Any,
) -> None:
    width = 1249
    episodes = 2
    inputs = _empty_refill_inputs(seeds=[11, 12], episodes=episodes, width=width)

    receipt = refill_m4_off_module.refill_policy_window(
        inputs["source"],
        inputs["delta"],
        inputs["mask"],
        inputs["crossover"],
        inputs["post"],
        inputs["semantic"],
        inputs["cursor"],
        inputs["counter"],
        inputs["base"],
        width,
    )

    assert receipt["option_d_m4_device_policy_refill_enabled"] is False
    assert receipt["option_d_m4_device_policy_refill_authority"] == "disabled"
    assert "option_d_m4_device_policy_refill_generated_rows" not in receipt


def test_refill_policy_window_m4_candidate_generates_device_compact_segment(
    refill_m4_on_module: Any,
) -> None:
    width = 9415
    episodes = 2
    seeds = [42, 43, 44]
    inputs = _empty_refill_inputs(seeds=seeds, episodes=episodes, width=width)
    expected = _expected_policy_segment(seeds=seeds, episodes=episodes, width=width)

    receipt = refill_m4_on_module.refill_policy_window(
        inputs["source"],
        inputs["delta"],
        inputs["mask"],
        inputs["crossover"],
        inputs["post"],
        inputs["semantic"],
        inputs["cursor"],
        inputs["counter"],
        inputs["base"],
        width,
    )
    torch.cuda.synchronize()

    assert receipt["option_d_m4_device_policy_refill_enabled"] is True
    assert receipt["option_d_m4_device_policy_refill_authority"] == (
        "candidate_device_generated_compact_refill"
    )
    assert receipt["option_d_m4_device_policy_refill_generated_rows"] == (
        len(seeds) * episodes
    )
    assert receipt["option_d_m4_device_policy_refill_invalid_state_count"] == 0
    assert receipt["option_d_m4_device_policy_refill_invalid_counter_count"] == 0
    assert receipt["option_d_m4_device_policy_refill_materialized_float32_crossover"] is False
    assert receipt["option_d_m4_device_policy_refill_persistent_rng_state_written"] is False
    assert receipt["host_to_device_bytes_lower_bound"] == 0
    assert receipt["device_to_host_bytes_lower_bound"] == 32
    assert torch.equal(inputs["crossover"].cpu(), expected)
    assert not inputs["mask"].cpu().any()
    assert torch.equal(
        inputs["semantic"].cpu(),
        torch.full((len(seeds), episodes), width, dtype=torch.int64),
    )


def test_refill_policy_window_m4_on_falls_back_for_mutation_surface(
    refill_m4_on_module: Any,
) -> None:
    width = 7
    episodes = 2
    inputs = _empty_mutation_refill_inputs(
        seeds=[101],
        episodes=episodes,
        width=width,
    )

    receipt = refill_m4_on_module.refill_mutation_policy_window(
        inputs["source"],
        inputs["delta"],
        inputs["mask"],
        inputs["crossover"],
        inputs["post"],
        inputs["semantic"],
        inputs["cursor"],
        inputs["counter"],
        inputs["base"],
        width,
    )
    torch.cuda.synchronize()

    assert receipt["option_d_m4_device_policy_refill_enabled"] is True
    assert receipt["option_d_m4_device_policy_refill_authority"] == (
        "compiled_candidate_not_used_for_surface"
    )
    assert "option_d_m4_device_policy_refill_generated_rows" not in receipt
    assert receipt["policy_birth_post_rng_state_resident_rows_materialized"] is True
