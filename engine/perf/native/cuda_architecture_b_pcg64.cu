#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "cuda_architecture_b_pcg64.cuh"

namespace {

using eiel::cuda_architecture_b::Pcg64State;
using eiel::cuda_architecture_b::pcg64_numpy_float32;

__global__ void pcg64_uniform_float32_kernel(
    uint64_t* __restrict__ state_words,
    int64_t* __restrict__ buffer_words,
    float* __restrict__ output,
    const int64_t count
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    Pcg64State rng{
        state_words[0],
        state_words[1],
        state_words[2],
        state_words[3],
        static_cast<uint32_t>(buffer_words[0]),
        static_cast<uint32_t>(buffer_words[1]),
    };
    for (int64_t index = 0; index < count; ++index) {
        output[index] = pcg64_numpy_float32(rng);
    }
    state_words[0] = rng.state_lo;
    state_words[1] = rng.state_hi;
    state_words[2] = rng.inc_lo;
    state_words[3] = rng.inc_hi;
    buffer_words[0] = static_cast<int64_t>(rng.has_uint32);
    buffer_words[1] = static_cast<int64_t>(rng.uinteger);
}

}  // namespace

std::vector<torch::Tensor> eiel_cuda_pcg64_uniform_float32_v1(
    torch::Tensor state_words,
    torch::Tensor buffer_words,
    const int64_t count
) {
    TORCH_CHECK(state_words.is_cuda(), "state_words must be a CUDA tensor");
    TORCH_CHECK(buffer_words.is_cuda(), "buffer_words must be a CUDA tensor");
    TORCH_CHECK(state_words.is_contiguous(), "state_words must be contiguous");
    TORCH_CHECK(buffer_words.is_contiguous(), "buffer_words must be contiguous");
    TORCH_CHECK(state_words.scalar_type() == torch::kUInt64, "state_words must be uint64");
    TORCH_CHECK(buffer_words.scalar_type() == torch::kInt64, "buffer_words must be int64");
    TORCH_CHECK(state_words.dim() == 1 && state_words.numel() == 4, "state_words must have four words");
    TORCH_CHECK(buffer_words.dim() == 1 && buffer_words.numel() == 2, "buffer_words must have two words");
    TORCH_CHECK(state_words.device() == buffer_words.device(), "PCG64 tensors must share one device");
    TORCH_CHECK(count >= 0, "count must be non-negative");
    auto output = torch::empty({count}, state_words.options().dtype(torch::kFloat32));
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        pcg64_uniform_float32_kernel<<<1, 1, 0, stream>>>(
            state_words.data_ptr<uint64_t>(),
            buffer_words.data_ptr<int64_t>(),
            output.data_ptr<float>(),
            count
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {output, state_words, buffer_words};
}
