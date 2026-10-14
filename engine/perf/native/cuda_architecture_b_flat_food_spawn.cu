#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdint>

#include "cuda_architecture_b_pcg64.cuh"

namespace {

constexpr int32_t FOOD = 1;

__global__ void flat_food_spawn_kernel(
    int32_t* __restrict__ food_grid,
    float* __restrict__ patch_food_amount,
    uint64_t* __restrict__ pcg64_state_words,
    int64_t* __restrict__ pcg64_buffer_words,
    int64_t* __restrict__ policy_np_semantic_draws,
    int64_t* __restrict__ stats,
    const int64_t cell_count,
    const int64_t max_food,
    const int64_t effective_spawn_count,
    const bool patch_enabled,
    const float patch_food_presence_threshold,
    const float patch_food_initial_value
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int64_t current_food = 0;
    for (int64_t cell = 0; cell < cell_count; ++cell) {
        current_food += food_grid[cell] == FOOD ? 1 : 0;
    }
    if (max_food <= 0 || effective_spawn_count <= 0 || current_food >= max_food) {
        return;
    }

    const int64_t target = min(effective_spawn_count, max_food - current_food);
    const int64_t max_tries = max(static_cast<int64_t>(80), target * 10);
    eiel::cuda_architecture_b::Pcg64State rng{
        pcg64_state_words[0],
        pcg64_state_words[1],
        pcg64_state_words[2],
        pcg64_state_words[3],
        static_cast<uint32_t>(pcg64_buffer_words[0]),
        static_cast<uint32_t>(pcg64_buffer_words[1]),
    };

    int64_t placed = 0;
    int64_t tries = 0;
    while (placed < target && tries < max_tries) {
        ++tries;
        int64_t cell = static_cast<int64_t>(
            eiel::cuda_architecture_b::pcg64_double01(rng)
            * static_cast<double>(cell_count)
        );
        cell = max(static_cast<int64_t>(0), min(cell, cell_count - 1));
        if (food_grid[cell] != FOOD) {
            food_grid[cell] = FOOD;
            ++placed;
        }
    }

    int64_t patch_sync_rows = 0;
    if (patch_enabled) {
        for (int64_t cell = 0; cell < cell_count; ++cell) {
            if (food_grid[cell] == FOOD
                && patch_food_amount[cell] <= patch_food_presence_threshold) {
                patch_food_amount[cell] = patch_food_initial_value;
                ++patch_sync_rows;
            }
        }
        for (int64_t cell = 0; cell < cell_count; ++cell) {
            food_grid[cell] = patch_food_amount[cell] > patch_food_presence_threshold
                ? FOOD
                : 0;
        }
    }

    pcg64_state_words[0] = rng.state_lo;
    pcg64_state_words[1] = rng.state_hi;
    pcg64_state_words[2] = rng.inc_lo;
    pcg64_state_words[3] = rng.inc_hi;
    pcg64_buffer_words[0] = static_cast<int64_t>(rng.has_uint32);
    pcg64_buffer_words[1] = static_cast<int64_t>(rng.uinteger);
    policy_np_semantic_draws[0] += tries;
    stats[0] = tries;
    stats[1] = placed;
    stats[2] = patch_sync_rows;
}

}  // namespace

torch::Tensor eiel_cuda_architecture_b_flat_food_spawn_v1(
    torch::Tensor food_grid,
    torch::Tensor patch_food_amount,
    torch::Tensor pcg64_state_words,
    torch::Tensor pcg64_buffer_words,
    torch::Tensor policy_np_semantic_draws,
    int64_t max_food,
    int64_t effective_spawn_count,
    bool patch_enabled,
    double patch_food_presence_threshold,
    double patch_food_initial_value
) {
    TORCH_CHECK(food_grid.is_cuda(), "food_grid must be CUDA-resident");
    TORCH_CHECK(food_grid.scalar_type() == torch::kInt32 && food_grid.dim() == 2,
        "food_grid must be CUDA int32[height,width]");
    TORCH_CHECK(food_grid.is_contiguous(), "food_grid must be contiguous");
    TORCH_CHECK(food_grid.numel() > 0, "food_grid must not be empty");
    const auto device = food_grid.device();
    TORCH_CHECK(
        pcg64_state_words.is_cuda()
            && pcg64_state_words.scalar_type() == torch::kUInt64
            && pcg64_state_words.numel() == 4
            && pcg64_state_words.is_contiguous()
            && pcg64_state_words.device() == device,
        "food spawn PCG64 state must be CUDA uint64[4]"
    );
    TORCH_CHECK(
        pcg64_buffer_words.is_cuda()
            && pcg64_buffer_words.scalar_type() == torch::kInt64
            && pcg64_buffer_words.numel() == 2
            && pcg64_buffer_words.is_contiguous()
            && pcg64_buffer_words.device() == device,
        "food spawn PCG64 buffer must be CUDA int64[2]"
    );
    TORCH_CHECK(
        policy_np_semantic_draws.is_cuda()
            && policy_np_semantic_draws.scalar_type() == torch::kInt64
            && policy_np_semantic_draws.numel() == 1
            && policy_np_semantic_draws.is_contiguous()
            && policy_np_semantic_draws.device() == device,
        "policy_np_semantic_draws must be CUDA int64[1]"
    );
    if (patch_enabled) {
        TORCH_CHECK(
            patch_food_amount.is_cuda()
                && patch_food_amount.scalar_type() == torch::kFloat32
                && patch_food_amount.sizes() == food_grid.sizes()
                && patch_food_amount.is_contiguous()
                && patch_food_amount.device() == device,
            "Patch food spawn requires contiguous CUDA float32 patch_food_amount"
        );
    }

    auto stats = torch::zeros(
        {3}, food_grid.options().dtype(torch::kInt64)
    );
    c10::cuda::CUDAGuard device_guard(device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    flat_food_spawn_kernel<<<1, 1, 0, stream>>>(
        food_grid.data_ptr<int32_t>(),
        patch_enabled ? patch_food_amount.data_ptr<float>() : nullptr,
        pcg64_state_words.data_ptr<uint64_t>(),
        pcg64_buffer_words.data_ptr<int64_t>(),
        policy_np_semantic_draws.data_ptr<int64_t>(),
        stats.data_ptr<int64_t>(),
        food_grid.numel(),
        max_food,
        effective_spawn_count,
        patch_enabled,
        static_cast<float>(patch_food_presence_threshold),
        static_cast<float>(patch_food_initial_value)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return stats;
}
