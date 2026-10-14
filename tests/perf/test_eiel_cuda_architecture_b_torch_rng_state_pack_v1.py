from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.cpp_extension import load_inline


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "engine/perf/cuda_architecture_b_torch_rng_state_pack_v1.cuh"
FULL_STATE_BYTES = 5056
PACKED_STATE_BYTES = 2560
LEADING_FULL_BYTES = 3 * 8
MIDDLE_LANES = 624
TRAILING_FULL_OFFSET = LEADING_FULL_BYTES + MIDDLE_LANES * 8
TRAILING_PACKED_OFFSET = LEADING_FULL_BYTES + MIDDLE_LANES * 4
SENTINEL = 0xA5


@pytest.fixture(scope="module")
def decoder_module(tmp_path_factory: pytest.TempPathFactory) -> Any:
    cpp_source = r"""
    #include <torch/extension.h>

    std::vector<torch::Tensor> decode_rows_cuda(torch::Tensor packed);
    std::vector<torch::Tensor> probe_copy_extents_cuda(
        torch::Tensor packed,
        int64_t full_state_bytes,
        int64_t packed_state_bytes
    );
    torch::Tensor packed_byte_indices_cuda(torch::Tensor full_indices);

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
        module.def("decode_rows", &decode_rows_cuda);
        module.def("probe_copy_extents", &probe_copy_extents_cuda);
        module.def("packed_byte_indices", &packed_byte_indices_cuda);
    }
    """
    cuda_source = r"""
    #include <torch/extension.h>

    #include <ATen/cuda/CUDAContext.h>
    #include <c10/cuda/CUDAGuard.h>
    #include <c10/cuda/CUDAException.h>
    #include <cuda.h>
    #include <cuda_runtime.h>

    #include <cstdint>
    #include <vector>

    #include "cuda_architecture_b_torch_rng_state_pack_v1.cuh"

    namespace pack =
        eiel::cuda_architecture_b::torch_rng_state_pack_v1;

    namespace {

    void check_packed_rows(const torch::Tensor& packed) {
        TORCH_CHECK(packed.is_cuda(), "packed must be a CUDA tensor");
        TORCH_CHECK(packed.is_contiguous(), "packed must be contiguous");
        TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "packed must be uint8");
        TORCH_CHECK(
            packed.dim() == 2 && packed.size(1) == pack::kPackedStateBytes,
            "packed must be uint8[rows, 2560]"
        );
    }

    __global__ void copy_rows_kernel(
        const std::uint8_t* const packed,
        std::uint8_t* const full,
        bool* const status,
        const std::int64_t rows,
        const std::int64_t full_state_bytes,
        const std::int64_t packed_state_bytes
    ) {
        const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
        if (row >= rows || threadIdx.x != 0) {
            return;
        }
        status[row] = pack::copy_unpacked_state_v1(
            packed + row * pack::kPackedStateBytes,
            packed_state_bytes,
            full + row * pack::kFullStateBytes,
            full_state_bytes
        );
    }

    __global__ void packed_byte_indices_kernel(
        const std::int64_t* const full_indices,
        std::int64_t* const packed_indices,
        const std::int64_t count
    ) {
        const std::int64_t index = static_cast<std::int64_t>(blockIdx.x)
            * blockDim.x + threadIdx.x;
        if (index < count) {
            packed_indices[index] = pack::packed_byte_index_v1(full_indices[index]);
        }
    }

    std::vector<torch::Tensor> launch_copy_rows(
        const torch::Tensor& packed,
        const std::int64_t full_state_bytes,
        const std::int64_t packed_state_bytes
    ) {
        check_packed_rows(packed);
        const c10::cuda::CUDAGuard device_guard(packed.device());
        const std::int64_t rows = packed.size(0);
        auto full = torch::full(
            {rows, pack::kFullStateBytes},
            0xA5,
            packed.options()
        );
        auto status = torch::zeros(
            {rows}, packed.options().dtype(torch::kBool)
        );
        if (rows > 0) {
            copy_rows_kernel<<<
                static_cast<unsigned int>(rows),
                1,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                packed.data_ptr<std::uint8_t>(),
                full.data_ptr<std::uint8_t>(),
                status.data_ptr<bool>(),
                rows,
                full_state_bytes,
                packed_state_bytes
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return {full, status};
    }

    }  // namespace

    std::vector<torch::Tensor> decode_rows_cuda(torch::Tensor packed) {
        return launch_copy_rows(
            packed,
            pack::kFullStateBytes,
            pack::kPackedStateBytes
        );
    }

    std::vector<torch::Tensor> probe_copy_extents_cuda(
        torch::Tensor packed,
        const std::int64_t full_state_bytes,
        const std::int64_t packed_state_bytes
    ) {
        return launch_copy_rows(packed, full_state_bytes, packed_state_bytes);
    }

    torch::Tensor packed_byte_indices_cuda(torch::Tensor full_indices) {
        TORCH_CHECK(full_indices.is_cuda(), "full_indices must be a CUDA tensor");
        TORCH_CHECK(full_indices.is_contiguous(), "full_indices must be contiguous");
        TORCH_CHECK(
            full_indices.scalar_type() == torch::kInt64,
            "full_indices must be int64"
        );
        TORCH_CHECK(full_indices.dim() == 1, "full_indices must be rank 1");
        const c10::cuda::CUDAGuard device_guard(full_indices.device());
        auto packed_indices = torch::empty_like(full_indices);
        const std::int64_t count = full_indices.numel();
        if (count > 0) {
            constexpr std::int64_t threads = 128;
            const std::int64_t blocks = (count + threads - 1) / threads;
            packed_byte_indices_kernel<<<
                static_cast<unsigned int>(blocks),
                static_cast<unsigned int>(threads),
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                full_indices.data_ptr<std::int64_t>(),
                packed_indices.data_ptr<std::int64_t>(),
                count
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return packed_indices;
    }
    """
    digest = sha256(
        HEADER.read_bytes() + cpp_source.encode() + cuda_source.encode()
    ).hexdigest()[:16]
    return load_inline(
        name=f"eiel_torch_rng_state_pack_v1_{digest}",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        build_directory=str(tmp_path_factory.mktemp("torch-rng-state-pack-build")),
        extra_include_paths=[str(HEADER.parent)],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
    )


def _torch_rng_state(seed: int, draws: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.rand(draws, generator=generator)
    state = generator.get_state()
    assert state.dtype == torch.uint8
    assert state.numel() == FULL_STATE_BYTES
    return state


def _full_rows() -> torch.Tensor:
    pattern = torch.arange(FULL_STATE_BYTES, dtype=torch.int64).remainder(251)
    pattern = pattern.to(torch.uint8)
    return torch.stack(
        (
            _torch_rng_state(0, 0),
            _torch_rng_state(1, 17),
            _torch_rng_state(0x0123_4567_89AB_CDEF, 625),
            pattern,
        )
    ).contiguous()


def _pack_rows_local(full: torch.Tensor) -> torch.Tensor:
    assert full.dtype == torch.uint8
    assert tuple(full.shape[1:]) == (FULL_STATE_BYTES,)
    middle = full[:, LEADING_FULL_BYTES:TRAILING_FULL_OFFSET]
    middle_low_halves = middle.view(full.size(0), MIDDLE_LANES, 8)[:, :, :4]
    return torch.cat(
        (
            full[:, :LEADING_FULL_BYTES],
            middle_low_halves.reshape(full.size(0), -1),
            full[:, TRAILING_FULL_OFFSET:],
        ),
        dim=1,
    ).contiguous()


def _canonical_full_rows(full: torch.Tensor) -> torch.Tensor:
    canonical = full.clone()
    middle = canonical[:, LEADING_FULL_BYTES:TRAILING_FULL_OFFSET]
    middle.view(full.size(0), MIDDLE_LANES, 8)[:, :, 4:].zero_()
    return canonical


def _expected_packed_index(full_index: int) -> int:
    if full_index < 0 or full_index >= FULL_STATE_BYTES:
        return -1
    if full_index < LEADING_FULL_BYTES:
        return full_index
    if full_index < TRAILING_FULL_OFFSET:
        middle_index = full_index - LEADING_FULL_BYTES
        byte_in_lane = middle_index % 8
        if byte_in_lane >= 4:
            return -1
        return LEADING_FULL_BYTES + (middle_index // 8) * 4 + byte_in_lane
    return TRAILING_PACKED_OFFSET + (full_index - TRAILING_FULL_OFFSET)


def test_device_decoder_reconstructs_multiple_canonical_rows_exactly(
    decoder_module: Any,
) -> None:
    full = _full_rows()
    packed = _pack_rows_local(full)
    assert tuple(packed.shape) == (full.size(0), PACKED_STATE_BYTES)

    decoded, status = decoder_module.decode_rows(packed.cuda())
    torch.cuda.synchronize()

    assert status.cpu().tolist() == [True] * full.size(0)
    assert torch.equal(decoded.cpu(), _canonical_full_rows(full))
    pattern_middle = decoded[-1, LEADING_FULL_BYTES:TRAILING_FULL_OFFSET]
    assert torch.count_nonzero(pattern_middle.view(MIDDLE_LANES, 8)[:, 4:]) == 0


def test_device_mapping_covers_boundaries_and_stays_inside_packed_extent(
    decoder_module: Any,
) -> None:
    boundary_indices = (
        -1,
        0,
        LEADING_FULL_BYTES - 1,
        LEADING_FULL_BYTES,
        LEADING_FULL_BYTES + 3,
        LEADING_FULL_BYTES + 4,
        TRAILING_FULL_OFFSET - 8,
        TRAILING_FULL_OFFSET - 5,
        TRAILING_FULL_OFFSET - 4,
        TRAILING_FULL_OFFSET - 1,
        TRAILING_FULL_OFFSET,
        FULL_STATE_BYTES - 1,
        FULL_STATE_BYTES,
    )
    indices = torch.tensor(boundary_indices, dtype=torch.int64, device="cuda")
    actual = decoder_module.packed_byte_indices(indices).cpu().tolist()
    expected = [_expected_packed_index(index) for index in boundary_indices]
    assert actual == expected

    all_indices = torch.arange(FULL_STATE_BYTES, dtype=torch.int64, device="cuda")
    all_mappings = decoder_module.packed_byte_indices(all_indices).cpu()
    retained = all_mappings[all_mappings >= 0]
    assert retained.numel() == PACKED_STATE_BYTES
    assert int(retained.min()) == 0
    assert int(retained.max()) == PACKED_STATE_BYTES - 1
    assert torch.equal(retained, torch.arange(PACKED_STATE_BYTES))


@pytest.mark.parametrize(
    ("full_state_bytes", "packed_state_bytes"),
    (
        (FULL_STATE_BYTES - 1, PACKED_STATE_BYTES),
        (FULL_STATE_BYTES + 1, PACKED_STATE_BYTES),
        (FULL_STATE_BYTES, PACKED_STATE_BYTES - 1),
        (FULL_STATE_BYTES, PACKED_STATE_BYTES + 1),
    ),
)
def test_device_copy_rejects_incompatible_extents_without_writing(
    decoder_module: Any,
    full_state_bytes: int,
    packed_state_bytes: int,
) -> None:
    packed = _pack_rows_local(_full_rows()[:2]).cuda()

    decoded, status = decoder_module.probe_copy_extents(
        packed, full_state_bytes, packed_state_bytes
    )
    torch.cuda.synchronize()

    assert status.cpu().tolist() == [False, False]
    assert bool(torch.all(decoded == SENTINEL).cpu())


def test_extension_binding_rejects_malformed_packed_row_extent(
    decoder_module: Any,
) -> None:
    packed = _pack_rows_local(_full_rows()[:1])[:, :-1].contiguous().cuda()

    with pytest.raises(RuntimeError, match=r"uint8\[rows, 2560\]"):
        decoder_module.decode_rows(packed)
