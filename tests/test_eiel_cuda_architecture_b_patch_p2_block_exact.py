from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch.utils.cpp_extension import load_inline


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Patch P2 block exactness requires CUDA",
)

ROOT = Path(__file__).resolve().parents[1]
ENGINE_PERF = ROOT / "engine" / "perf"
HEADER = ENGINE_PERF / "cuda_architecture_b_persistent_patch_p2.cuh"
HEIGHT = 16
WIDTH = 16
CELL_COUNT = HEIGHT * WIDTH
FLOAT_PLANE_COUNT = 11
INT_PLANE_COUNT = 2
REQUIRED_STAT_PREFIX_COUNT = 19
PATCH_STAT_COUNT = 25
PATTERNS = ("sparse_edges", "checkerboard", "full")
DENSITY_VARIANTS = (
    (False, False),
    (False, True),
    (True, False),
    (True, True),
)

CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "cuda_architecture_b_persistent_patch_p2.cuh"

namespace patch_p2 =
    eiel::cuda_architecture_b::persistent_patch_p2;

namespace {

constexpr int64_t kHeight = 16;
constexpr int64_t kWidth = 16;
constexpr int64_t kCellCount = kHeight * kWidth;
constexpr int64_t kFloatPlaneCount = 11;
constexpr int64_t kIntPlaneCount = 2;

patch_p2::PatchP2State make_state(
    torch::Tensor float_planes,
    torch::Tensor int_planes
) {
    float* floats = float_planes.data_ptr<float>();
    int32_t* integers = int_planes.data_ptr<int32_t>();
    return patch_p2::PatchP2State{
        floats + 0 * kCellCount,
        floats + 1 * kCellCount,
        floats + 2 * kCellCount,
        floats + 3 * kCellCount,
        floats + 4 * kCellCount,
        floats + 5 * kCellCount,
        integers + 0 * kCellCount,
        floats + 6 * kCellCount,
        integers + 1 * kCellCount,
    };
}

patch_p2::PatchP2Scratch make_scratch(torch::Tensor float_planes) {
    float* floats = float_planes.data_ptr<float>();
    return patch_p2::PatchP2Scratch{
        floats + 7 * kCellCount,
        floats + 8 * kCellCount,
        floats + 9 * kCellCount,
        floats + 10 * kCellCount,
    };
}

patch_p2::PatchP2Config make_config(
    const bool density_include_self,
    const bool density_normalize
) {
    return patch_p2::PatchP2Config{
        kHeight,
        kWidth,
        0.875f,
        0.75f,
        0.125,
        0.0625,
        0.03125f,
        4.0f,
        3.0f,
        5.0f,
        0.0625f,
        true,
        true,
        0.5f,
        2,
        density_include_self,
        density_normalize,
        true,
        true,
        0.5f,
        0.45f,
        true,
        0.375f,
        0.4f,
        0.1f,
    };
}

void validate_inputs(
    const torch::Tensor float_planes,
    const torch::Tensor int_planes,
    const torch::Tensor stats
) {
    TORCH_CHECK(float_planes.is_cuda(), "float_planes must be CUDA");
    TORCH_CHECK(int_planes.is_cuda(), "int_planes must be CUDA");
    TORCH_CHECK(stats.is_cuda(), "stats must be CUDA");
    TORCH_CHECK(float_planes.is_contiguous(), "float_planes must be contiguous");
    TORCH_CHECK(int_planes.is_contiguous(), "int_planes must be contiguous");
    TORCH_CHECK(stats.is_contiguous(), "stats must be contiguous");
    TORCH_CHECK(
        float_planes.scalar_type() == c10::ScalarType::Float,
        "float_planes must be float32"
    );
    TORCH_CHECK(
        int_planes.scalar_type() == c10::ScalarType::Int,
        "int_planes must be int32"
    );
    TORCH_CHECK(
        stats.scalar_type() == c10::ScalarType::Double,
        "stats must be float64"
    );
    TORCH_CHECK(
        float_planes.sizes() == torch::IntArrayRef({kFloatPlaneCount, kCellCount}),
        "float_planes has the wrong shape"
    );
    TORCH_CHECK(
        int_planes.sizes() == torch::IntArrayRef({kIntPlaneCount, kCellCount}),
        "int_planes has the wrong shape"
    );
    TORCH_CHECK(
        stats.numel() == patch_p2::kStatCount,
        "stats has the wrong size"
    );
    TORCH_CHECK(
        float_planes.get_device() == int_planes.get_device()
            && float_planes.get_device() == stats.get_device(),
        "all tensors must be on the same CUDA device"
    );
}

__global__ void serial_tick_kernel(
    const patch_p2::PatchP2State state,
    const patch_p2::PatchP2Scratch scratch,
    const patch_p2::PatchP2Outputs outputs,
    const patch_p2::PatchP2Config config
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    patch_p2::patch_p2_single_tick_exact(state, scratch, outputs, config);
}

__global__ void block_tick_kernel(
    const patch_p2::PatchP2State state,
    const patch_p2::PatchP2Scratch scratch,
    const patch_p2::PatchP2Outputs outputs,
    const patch_p2::PatchP2Config config
) {
    if (blockIdx.x != 0) {
        return;
    }
    patch_p2::patch_p2_single_tick_exact_block_256(
        state,
        scratch,
        outputs,
        config
    );
}

template <bool UseBlockHelper>
torch::Tensor run_tick(
    torch::Tensor float_planes,
    torch::Tensor int_planes,
    torch::Tensor stats,
    const bool density_include_self,
    const bool density_normalize
) {
    validate_inputs(float_planes, int_planes, stats);
    c10::cuda::CUDAGuard device_guard(float_planes.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const patch_p2::PatchP2State state = make_state(float_planes, int_planes);
    const patch_p2::PatchP2Scratch scratch = make_scratch(float_planes);
    const patch_p2::PatchP2Outputs outputs{stats.data_ptr<double>()};
    const patch_p2::PatchP2Config config = make_config(
        density_include_self,
        density_normalize
    );
    if constexpr (UseBlockHelper) {
        block_tick_kernel<<<1, 128, 0, stream>>>(state, scratch, outputs, config);
    } else {
        serial_tick_kernel<<<1, 1, 0, stream>>>(state, scratch, outputs, config);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return float_planes;
}

}  // namespace

torch::Tensor eiel_patch_p2_serial_tick_exact_test(
    torch::Tensor float_planes,
    torch::Tensor int_planes,
    torch::Tensor stats,
    const bool density_include_self,
    const bool density_normalize
) {
    return run_tick<false>(
        float_planes,
        int_planes,
        stats,
        density_include_self,
        density_normalize
    );
}

torch::Tensor eiel_patch_p2_block_tick_exact_test(
    torch::Tensor float_planes,
    torch::Tensor int_planes,
    torch::Tensor stats,
    const bool density_include_self,
    const bool density_normalize
) {
    return run_tick<true>(
        float_planes,
        int_planes,
        stats,
        density_include_self,
        density_normalize
    );
}
"""


@pytest.fixture(scope="module")
def block_exact_module(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    digest = sha256(HEADER.read_bytes() + CUDA_SOURCE.encode()).hexdigest()[:12]
    cpp_source = r"""
    #include <torch/extension.h>

    torch::Tensor eiel_patch_p2_serial_tick_exact_test(
        torch::Tensor float_planes,
        torch::Tensor int_planes,
        torch::Tensor stats,
        bool density_include_self,
        bool density_normalize
    );
    torch::Tensor eiel_patch_p2_block_tick_exact_test(
        torch::Tensor float_planes,
        torch::Tensor int_planes,
        torch::Tensor stats,
        bool density_include_self,
        bool density_normalize
    );
    """
    return load_inline(
        name=f"eiel_patch_p2_block_exact_test_{digest}",
        cpp_sources=[cpp_source],
        cuda_sources=[CUDA_SOURCE],
        functions=[
            "eiel_patch_p2_serial_tick_exact_test",
            "eiel_patch_p2_block_tick_exact_test",
        ],
        extra_include_paths=[str(ENGINE_PERF)],
        build_directory=str(tmp_path_factory.mktemp("patch-p2-block-exact-build")),
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
    )


def _occupied_mask(pattern: str) -> torch.Tensor:
    indices = torch.arange(CELL_COUNT, dtype=torch.int64)
    y = torch.div(indices, WIDTH, rounding_mode="floor")
    x = indices % WIDTH
    if pattern == "sparse_edges":
        occupied_indices = torch.tensor(
            (0, WIDTH - 1, CELL_COUNT // 2, CELL_COUNT - WIDTH, CELL_COUNT - 1),
            dtype=torch.int64,
        )
        mask = torch.zeros(CELL_COUNT, dtype=torch.bool)
        mask[occupied_indices] = True
        return mask
    if pattern == "checkerboard":
        return (x + y) % 2 == 0
    if pattern == "full":
        return torch.ones(CELL_COUNT, dtype=torch.bool)
    raise AssertionError(f"unknown pattern: {pattern}")


def _inputs(pattern: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = torch.arange(CELL_COUNT, dtype=torch.float32)
    float_planes = torch.empty(
        (FLOAT_PLANE_COUNT, CELL_COUNT),
        dtype=torch.float32,
    )
    float_planes[0] = torch.remainder(index * 13.0, 127.0) * 0.03125
    float_planes[1] = 0.5 + torch.remainder(index * 5.0, 17.0) * 0.03125
    float_planes[2] = torch.remainder(index * 3.0, 53.0) * 0.015625
    float_planes[3] = torch.remainder(index * 7.0, 43.0) * 0.03125
    float_planes[4] = torch.where(index % 5 == 0, 0.125, 0.0)
    float_planes[5] = torch.where(index % 7 == 0, 0.0625, 0.0)
    float_planes[6] = -701.25
    float_planes[7] = -702.25
    float_planes[8] = -703.25
    float_planes[9] = -704.25
    float_planes[10] = -705.25

    occupied = _occupied_mask(pattern)
    int_planes = torch.empty((INT_PLANE_COUNT, CELL_COUNT), dtype=torch.int32)
    agent_ids = torch.arange(CELL_COUNT, dtype=torch.int32)
    int_planes[0] = torch.where(occupied, agent_ids, -1)
    int_planes[1] = -17

    stats = torch.full((PATCH_STAT_COUNT,), -9.87654321e200, dtype=torch.float64)
    assert bool(torch.isfinite(float_planes).all())
    assert bool(torch.isfinite(stats).all())
    return float_planes, int_planes, stats


def _assert_outputs_bitwise_equal(
    serial_float: torch.Tensor,
    serial_int: torch.Tensor,
    serial_stats: torch.Tensor,
    block_float: torch.Tensor,
    block_int: torch.Tensor,
    block_stats: torch.Tensor,
) -> None:
    for plane in range(FLOAT_PLANE_COUNT):
        assert torch.equal(
            serial_float[plane].view(torch.int32),
            block_float[plane].view(torch.int32),
        ), f"float plane {plane} differs"
    for plane in range(INT_PLANE_COUNT):
        assert torch.equal(serial_int[plane], block_int[plane]), (
            f"int plane {plane} differs"
        )
    assert torch.equal(
        serial_stats[:REQUIRED_STAT_PREFIX_COUNT].view(torch.int64),
        block_stats[:REQUIRED_STAT_PREFIX_COUNT].view(torch.int64),
    ), "required 19-stat prefix differs"
    assert torch.equal(
        serial_stats.view(torch.int64),
        block_stats.view(torch.int64),
    ), "full 25-stat Patch P2 contract differs"


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize(
    ("density_include_self", "density_normalize"),
    DENSITY_VARIANTS,
)
def test_block_256_tick_matches_serial_tick_exactly(
    block_exact_module: ModuleType,
    pattern: str,
    density_include_self: bool,
    density_normalize: bool,
) -> None:
    initial_float, initial_int, initial_stats = _inputs(pattern)
    serial_float = initial_float.cuda()
    serial_int = initial_int.cuda()
    serial_stats = initial_stats.cuda()
    block_float = initial_float.cuda()
    block_int = initial_int.cuda()
    block_stats = initial_stats.cuda()
    repeat_float = initial_float.cuda()
    repeat_int = initial_int.cuda()
    repeat_stats = initial_stats.cuda()

    block_exact_module.eiel_patch_p2_serial_tick_exact_test(
        serial_float,
        serial_int,
        serial_stats,
        density_include_self,
        density_normalize,
    )
    block_exact_module.eiel_patch_p2_block_tick_exact_test(
        block_float,
        block_int,
        block_stats,
        density_include_self,
        density_normalize,
    )
    block_exact_module.eiel_patch_p2_block_tick_exact_test(
        repeat_float,
        repeat_int,
        repeat_stats,
        density_include_self,
        density_normalize,
    )

    serial_float = serial_float.cpu()
    serial_int = serial_int.cpu()
    serial_stats = serial_stats.cpu()
    block_float = block_float.cpu()
    block_int = block_int.cpu()
    block_stats = block_stats.cpu()
    repeat_float = repeat_float.cpu()
    repeat_int = repeat_int.cpu()
    repeat_stats = repeat_stats.cpu()

    _assert_outputs_bitwise_equal(
        serial_float,
        serial_int,
        serial_stats,
        block_float,
        block_int,
        block_stats,
    )
    _assert_outputs_bitwise_equal(
        block_float,
        block_int,
        block_stats,
        repeat_float,
        repeat_int,
        repeat_stats,
    )
    assert torch.equal(serial_float[1].view(torch.int32), initial_float[1].view(torch.int32))
    assert torch.equal(serial_int[0], initial_int[0])
    assert bool(torch.isfinite(serial_float).all())
    assert bool(torch.isfinite(serial_stats).all())
    assert not torch.equal(serial_float[6], initial_float[6])
    assert not torch.equal(serial_float[7:11], initial_float[7:11])
    assert not torch.equal(serial_stats, initial_stats)
