from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch.utils.cpp_extension import load_inline


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Patch P2 exact sort requires CUDA",
)

ROOT = Path(__file__).resolve().parents[1]
ENGINE_PERF = ROOT / "engine" / "perf"
HEADER = ENGINE_PERF / "cuda_architecture_b_persistent_patch_p2.cuh"
COUNTS = (1, 2, 3, 4, 8, 255, 256)
PATTERNS = ("descending", "shuffled", "duplicates")
LEFT_CANARY = -1048575.5
RIGHT_CANARY = 1048575.25

CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "cuda_architecture_b_persistent_patch_p2.cuh"

namespace {

__global__ void sort_guarded_kernel(float* guarded_values, const int64_t count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    eiel::cuda_architecture_b::persistent_patch_p2::detail::
        sort_f32_ascending_exact(guarded_values + 1, count);
}

}  // namespace

torch::Tensor eiel_patch_p2_sort_f32_ascending_exact_test(
    torch::Tensor guarded_values,
    const int64_t count
) {
    TORCH_CHECK(guarded_values.is_cuda(), "guarded_values must be CUDA");
    TORCH_CHECK(guarded_values.is_contiguous(), "guarded_values must be contiguous");
    TORCH_CHECK(
        guarded_values.scalar_type() == c10::ScalarType::Float,
        "guarded_values must be float32"
    );
    TORCH_CHECK(guarded_values.dim() == 1, "guarded_values must be one-dimensional");
    TORCH_CHECK(count > 0 && count <= 256, "count must be in [1, 256]");
    TORCH_CHECK(
        guarded_values.numel() == count + 2,
        "guarded_values must contain two canaries"
    );

    c10::cuda::CUDAGuard device_guard(guarded_values.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sort_guarded_kernel<<<1, 1, 0, stream>>>(
        guarded_values.data_ptr<float>(),
        count
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return guarded_values;
}
"""


@pytest.fixture(scope="module")
def sort_module(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    digest = sha256(HEADER.read_bytes() + CUDA_SOURCE.encode()).hexdigest()[:12]
    cpp_source = r"""
    #include <torch/extension.h>

    torch::Tensor eiel_patch_p2_sort_f32_ascending_exact_test(
        torch::Tensor guarded_values,
        int64_t count
    );
    """
    return load_inline(
        name=f"eiel_patch_p2_sort_f32_ascending_exact_test_{digest}",
        cpp_sources=[cpp_source],
        cuda_sources=[CUDA_SOURCE],
        functions=["eiel_patch_p2_sort_f32_ascending_exact_test"],
        extra_include_paths=[str(ENGINE_PERF)],
        build_directory=str(tmp_path_factory.mktemp("patch-p2-sort-exact-build")),
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
    )


def _payload(count: int, pattern: str) -> torch.Tensor:
    if pattern == "duplicates":
        repeated = torch.tensor(
            (3.5, -2.25, 3.5, 0.0, -2.25, 11.0, 0.0),
            dtype=torch.float32,
        )
        return repeated.repeat((count + repeated.numel() - 1) // repeated.numel())[:count]

    ascending = (torch.arange(count, dtype=torch.float32) - count // 2) * 0.25
    if pattern == "descending":
        return ascending.flip(0)
    if pattern == "shuffled":
        permutation = (torch.arange(count, dtype=torch.int64) * 73 + 19) % count
        return ascending.index_select(0, permutation)
    raise AssertionError(f"unknown pattern: {pattern}")


def _stable_cpu_insertion_sort(values: torch.Tensor) -> torch.Tensor:
    result = values.clone()
    for index in range(1, result.numel()):
        value = result[index].clone()
        insert = index
        while insert > 0 and bool(result[insert - 1] > value):
            result[insert] = result[insert - 1]
            insert -= 1
        result[insert] = value
    return result


@pytest.mark.parametrize("count", COUNTS)
@pytest.mark.parametrize("pattern", PATTERNS)
def test_sort_f32_ascending_exact_matches_stable_insertion_bitwise_and_guards(
    sort_module: ModuleType,
    count: int,
    pattern: str,
) -> None:
    payload = _payload(count, pattern)
    assert payload.dtype == torch.float32
    assert bool(torch.isfinite(payload).all())
    expected = _stable_cpu_insertion_sort(payload)
    guarded = torch.cat(
        (
            torch.tensor((LEFT_CANARY,), dtype=torch.float32),
            payload,
            torch.tensor((RIGHT_CANARY,), dtype=torch.float32),
        )
    )
    expected_canary_bits = guarded[[0, -1]].view(torch.int32)

    runs = []
    for _ in range(3):
        actual = sort_module.eiel_patch_p2_sort_f32_ascending_exact_test(
            guarded.cuda(),
            count,
        ).cpu()
        assert torch.equal(actual[[0, -1]].view(torch.int32), expected_canary_bits)
        assert torch.equal(actual[1:-1].view(torch.int32), expected.view(torch.int32))
        runs.append(actual.view(torch.int32))

    assert torch.equal(runs[1], runs[0])
    assert torch.equal(runs[2], runs[0])
