from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
import torch
import numpy as np
from torch.utils.cpp_extension import load_inline

from engine.perf.cuda_architecture_b_child_policy import (
    pack_crossover_mask_lsb0_v1,
    packed_crossover_row_bytes_v1,
)


ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "engine/perf/cuda_architecture_b_option_d_m2_policy_shadow_v1.cuh"
CUDA_SOURCE = ROOT / "engine/perf/cuda_architecture_b_option_d_m2_policy_shadow_v1.cu"
REPLAY_HEADER = ROOT / "engine/perf/cuda_architecture_b_torch_mt19937_replay_v1.cuh"
STATE_HEADER = ROOT / "engine/perf/cuda_architecture_b_torch_rng_state_pack_v1.cuh"
REFILL_HEADER = ROOT / "engine/perf/cuda_architecture_b_resident_window_refill_v1.hpp"
FULL_STATE_BYTES = 5056


@pytest.fixture(scope="session")
def shadow_module() -> Any:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the M2 policy shadow verifier")

    digest = sha256(
        HEADER.read_bytes()
        + CUDA_SOURCE.read_bytes()
        + REPLAY_HEADER.read_bytes()
        + STATE_HEADER.read_bytes()
    ).hexdigest()[:12]
    cpp_source = r"""
    #include <torch/extension.h>
    #include "cuda_architecture_b_option_d_m2_policy_shadow_v1.cuh"

    std::vector<int64_t> verify_policy_segment_shadow(
        torch::Tensor source_state,
        torch::Tensor expected_crossover_mask_packed,
        torch::Tensor semantic_tape,
        int64_t policy_row_width
    ) {
        auto result =
            eiel::cuda_architecture_b::option_d_m2_policy_shadow_v1::
                verify_policy_segment_shadow_v1(
                    source_state,
                    expected_crossover_mask_packed,
                    semantic_tape,
                    policy_row_width
                );
        return {
            result.checked_rows,
            result.checked_bytes,
            result.mismatch_count,
            result.invalid_state_count,
            result.invalid_semantic_count,
        };
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
        module.def(
            "verify_policy_segment_shadow",
            &verify_policy_segment_shadow
        );
    }
    """
    return load_inline(
        name=f"eiel_option_d_m2_policy_shadow_v1_{digest}",
        cpp_sources=[cpp_source],
        cuda_sources=[CUDA_SOURCE.read_text(encoding="utf-8")],
        extra_include_paths=[str(ROOT / "engine/perf"), np.get_include()],
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


def _load_refill_module(*, shadow_enabled: bool) -> Any:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the M2 policy shadow refill integration")

    enabled_value = 1 if shadow_enabled else 0
    digest = sha256(
        REFILL_HEADER.read_bytes()
        + HEADER.read_bytes()
        + CUDA_SOURCE.read_bytes()
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

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
        module.def("refill_policy_window", &refill_policy_window);
    }
    """
    return load_inline(
        name=f"eiel_option_d_m2_refill_shadow_{enabled_value}_{digest}",
        cpp_sources=[cpp_source],
        cuda_sources=[CUDA_SOURCE.read_text(encoding="utf-8")],
        extra_include_paths=[str(ROOT / "engine/perf"), np.get_include()],
        extra_cflags=[
            "-O2",
            "-ffp-contract=off",
            f"-DEIEL_OPTION_D_M2_POLICY_SHADOW_ENABLED={enabled_value}",
        ],
        extra_cuda_cflags=[
            "-O2",
            "--expt-relaxed-constexpr",
            "--fmad=false",
            "--prec-div=true",
            "--prec-sqrt=true",
            f"-DEIEL_OPTION_D_M2_POLICY_SHADOW_ENABLED={enabled_value}",
        ],
        with_cuda=True,
        verbose=False,
    )


@pytest.fixture(scope="session")
def refill_shadow_on_module() -> Any:
    return _load_refill_module(shadow_enabled=True)


@pytest.fixture(scope="session")
def refill_shadow_off_module() -> Any:
    return _load_refill_module(shadow_enabled=False)


def _seed_state(seed: int) -> torch.Tensor:
    return torch.Generator(device="cpu").manual_seed(seed).get_state().clone()


def _policy_segment_rows(
    *,
    seeds: list[int],
    episodes: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_rows: list[torch.Tensor] = []
    crossover_rows: list[torch.Tensor] = []
    semantic_rows: list[torch.Tensor] = []
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        source_rows.append(generator.get_state().clone())
        cell_rows: list[torch.Tensor] = []
        cell_semantic: list[int] = []
        for _episode in range(episodes):
            crossover = torch.rand(
                (width,),
                generator=generator,
                dtype=torch.float32,
            ).lt(0.5)
            cell_rows.append(pack_crossover_mask_lsb0_v1(crossover).cpu())
            cell_semantic.append(width)
        crossover_rows.append(torch.stack(cell_rows).contiguous())
        semantic_rows.append(torch.tensor(cell_semantic, dtype=torch.int64))
    return (
        torch.stack(source_rows).contiguous(),
        torch.stack(crossover_rows).contiguous(),
        torch.stack(semantic_rows).contiguous(),
    )


def test_m2_policy_shadow_matches_cpu_prepared_packed_crossover(
    shadow_module: Any,
) -> None:
    width = 9415
    episodes = 3
    seeds = [42, 0xFFFF_FFFF_FFFF_FFFF, 123456789]
    source, crossover, semantic = _policy_segment_rows(
        seeds=seeds,
        episodes=episodes,
        width=width,
    )

    result = shadow_module.verify_policy_segment_shadow(
        source.cuda(),
        crossover.cuda(),
        semantic.cuda(),
        width,
    )

    assert result == [
        len(seeds) * episodes,
        len(seeds) * episodes * packed_crossover_row_bytes_v1(width),
        0,
        0,
        0,
    ]


def test_m2_policy_shadow_detects_packed_crossover_mismatch(
    shadow_module: Any,
) -> None:
    width = 1249
    source, crossover, semantic = _policy_segment_rows(
        seeds=[7, 8],
        episodes=2,
        width=width,
    )
    corrupted = crossover.clone()
    corrupted[1, 0, 0] ^= 0x01

    result = shadow_module.verify_policy_segment_shadow(
        source.cuda(),
        corrupted.cuda(),
        semantic.cuda(),
        width,
    )

    assert result[0] == 4
    assert result[2] == 1
    assert result[3:] == [0, 0]


def test_m2_policy_shadow_rejects_noncanonical_state(
    shadow_module: Any,
) -> None:
    width = 625
    source, crossover, semantic = _policy_segment_rows(
        seeds=[42],
        episodes=1,
        width=width,
    )
    source[0, 24 + 4] = 1

    result = shadow_module.verify_policy_segment_shadow(
        source.cuda(),
        crossover.cuda(),
        semantic.cuda(),
        width,
    )

    assert result == [0, 0, 0, 1, 0]


def test_m2_policy_shadow_rejects_non_width_semantic_rows(
    shadow_module: Any,
) -> None:
    width = 625
    source, crossover, semantic = _policy_segment_rows(
        seeds=[42],
        episodes=2,
        width=width,
    )
    semantic[0, 1] = width + 1

    result = shadow_module.verify_policy_segment_shadow(
        source.cuda(),
        crossover.cuda(),
        semantic.cuda(),
        width,
    )

    assert result == [1, packed_crossover_row_bytes_v1(width), 0, 0, 1]


def test_m2_policy_shadow_rejects_wrong_tensor_surfaces(
    shadow_module: Any,
) -> None:
    width = 625
    source, crossover, semantic = _policy_segment_rows(
        seeds=[42],
        episodes=1,
        width=width,
    )

    with pytest.raises(RuntimeError, match="source state must be CUDA resident"):
        shadow_module.verify_policy_segment_shadow(source, crossover.cuda(), semantic.cuda(), width)
    with pytest.raises(RuntimeError, match="crossover row width mismatch"):
        shadow_module.verify_policy_segment_shadow(
            source.cuda(),
            torch.zeros((1, 1, 1), dtype=torch.uint8, device="cuda"),
            semantic.cuda(),
            width,
        )


def _empty_refill_inputs(
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
        "delta": torch.empty((cells, episodes, 1), dtype=torch.float32, device=device),
        "mask": torch.empty((cells, episodes, 1), dtype=torch.bool, device=device),
        "crossover": torch.empty((cells, episodes, row_bytes), dtype=torch.uint8, device=device),
        "post": torch.empty((cells, episodes, 0), dtype=torch.uint8, device=device),
        "semantic": torch.empty((cells, episodes), dtype=torch.int64, device=device),
        "cursor": torch.zeros((cells,), dtype=torch.int64, device=device),
        "counter": torch.zeros((cells,), dtype=torch.int64, device=device),
        "base": torch.zeros((cells,), dtype=torch.int64, device=device),
    }


def test_refill_policy_window_m2_shadow_is_default_off_and_receipted(
    refill_shadow_off_module: Any,
) -> None:
    width = 1249
    episodes = 2
    inputs = _empty_refill_inputs(seeds=[11, 12], episodes=episodes, width=width)

    receipt = refill_shadow_off_module.refill_policy_window(
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

    assert receipt["option_d_m2_policy_shadow_enabled"] is False
    assert receipt["option_d_m2_policy_shadow_authority"] == "disabled"
    assert "option_d_m2_policy_shadow_checked_rows" not in receipt


def test_refill_policy_window_m2_shadow_runs_compare_only(
    refill_shadow_on_module: Any,
) -> None:
    width = 9415
    episodes = 2
    seeds = [42, 43, 44]
    inputs = _empty_refill_inputs(seeds=seeds, episodes=episodes, width=width)

    receipt = refill_shadow_on_module.refill_policy_window(
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

    assert receipt["option_d_m2_policy_shadow_enabled"] is True
    assert receipt["option_d_m2_policy_shadow_authority"] == (
        "non_authoritative_compare_only"
    )
    assert receipt["option_d_m2_policy_shadow_checked_rows"] == len(seeds) * episodes
    assert receipt["option_d_m2_policy_shadow_checked_bytes"] == (
        len(seeds) * episodes * packed_crossover_row_bytes_v1(width)
    )
    assert receipt["option_d_m2_policy_shadow_mismatch_count"] == 0
    assert receipt["option_d_m2_policy_shadow_invalid_state_count"] == 0
    assert receipt["option_d_m2_policy_shadow_invalid_semantic_count"] == 0
    assert receipt["option_d_m2_policy_shadow_materialized_float32_crossover"] is False
    assert receipt["option_d_m2_policy_shadow_persistent_rng_state_written"] is False
