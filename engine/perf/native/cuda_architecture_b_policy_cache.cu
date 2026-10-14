#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int32_t ACT_REST = 5;
constexpr int32_t ACT_ATTACK = 6;

constexpr int32_t STATUS_OK = 0;
constexpr int32_t STATUS_INVALID_COUNT = 1;
constexpr int32_t STATUS_INVALID_SLOT = 2;
constexpr int32_t STATUS_INVALID_CACHE_WIDTH = 3;

void check_cuda_contiguous(
    const torch::Tensor& tensor,
    const char* name,
    const c10::Device& device
) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == device, name, " must share one CUDA device");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_vector(
    const torch::Tensor& tensor,
    const char* name,
    const c10::ScalarType dtype,
    const int64_t capacity,
    const c10::Device& device
) {
    check_cuda_contiguous(tensor, name, device);
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
    TORCH_CHECK(
        tensor.dim() == 1 && tensor.size(0) == capacity,
        name,
        " must be [capacity]"
    );
}

__global__ void policy_cache_service_kernel(
    const int64_t* __restrict__ active_agent_ids,
    const int64_t* __restrict__ active_count,
    const int32_t* __restrict__ slots,
    const int64_t* __restrict__ slot_tokens,
    const int32_t* __restrict__ action_cache_actions,
    const int64_t* __restrict__ action_cache_token,
    int32_t* __restrict__ action_cache_cursor,
    int32_t* __restrict__ action_cache_next,
    int64_t* __restrict__ miss_agent_ids,
    int32_t* __restrict__ miss_slots,
    int64_t* __restrict__ requested,
    int64_t* __restrict__ hits,
    int64_t* __restrict__ misses,
    int64_t* __restrict__ full_refill_tick,
    int64_t* __restrict__ partial_refill_tick,
    int64_t* __restrict__ full_hit_tick,
    int32_t* __restrict__ status,
    const int64_t capacity,
    const int64_t cache_capacity,
    const int64_t cache_width
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    if (cache_width <= 0 || cache_width > cache_capacity) {
        status[0] = STATUS_INVALID_CACHE_WIDTH;
        return;
    }

    const int64_t row_count = active_count[0];
    if (row_count < 0 || row_count > capacity) {
        status[0] = STATUS_INVALID_COUNT;
        return;
    }

    // Validate all active rows before mutating cache state.
    for (int64_t row = 0; row < row_count; ++row) {
        const int32_t slot = slots[row];
        if (slot < 0 || static_cast<int64_t>(slot) >= capacity) {
            status[0] = STATUS_INVALID_SLOT;
            return;
        }
    }

    int64_t hit_count = 0;
    int64_t miss_count = 0;
    for (int64_t row = 0; row < row_count; ++row) {
        const int32_t slot = slots[row];
        const int32_t cursor = action_cache_cursor[slot];
        const bool hit = action_cache_token[slot] == slot_tokens[slot]
            && cursor >= 0
            && static_cast<int64_t>(cursor) < cache_width;
        if (!hit) {
            miss_agent_ids[miss_count] = active_agent_ids[row];
            miss_slots[miss_count] = slot;
            ++miss_count;
            continue;
        }

        int32_t action = action_cache_actions[
            static_cast<int64_t>(slot) * cache_capacity + cursor
        ];
        if (action < 0 || action > ACT_ATTACK) {
            action = action_cache_next[slot];
        }
        if (action < 0 || action > ACT_ATTACK) {
            action = ACT_REST;
        }
        action_cache_next[slot] = action;
        action_cache_cursor[slot] = cursor + 1;
        ++hit_count;
    }

    requested[0] = row_count;
    hits[0] = hit_count;
    misses[0] = miss_count;
    full_refill_tick[0] = miss_count > 0 && hit_count == 0 ? 1 : 0;
    partial_refill_tick[0] = miss_count > 0 && hit_count > 0 ? 1 : 0;
    full_hit_tick[0] = hit_count > 0 && miss_count == 0 ? 1 : 0;
    status[0] = STATUS_OK;
}

__global__ void policy_cache_service_batched_kernel(
    const int64_t* __restrict__ active_agent_ids,
    const int64_t* __restrict__ active_count,
    const int32_t* __restrict__ slots,
    const int64_t* __restrict__ slot_tokens,
    const int32_t* __restrict__ action_cache_actions,
    const int64_t* __restrict__ action_cache_token,
    int32_t* __restrict__ action_cache_cursor,
    int32_t* __restrict__ action_cache_next,
    int64_t* __restrict__ miss_agent_ids,
    int32_t* __restrict__ miss_slots,
    int64_t* __restrict__ requested,
    int64_t* __restrict__ hits,
    int64_t* __restrict__ misses,
    int64_t* __restrict__ full_refill_tick,
    int64_t* __restrict__ partial_refill_tick,
    int64_t* __restrict__ full_hit_tick,
    int32_t* __restrict__ status,
    const int64_t cell_count,
    const int64_t capacity,
    const int64_t cache_capacity,
    const int64_t cache_width
) {
    const int64_t cell = static_cast<int64_t>(blockIdx.x);
    if (cell >= cell_count || threadIdx.x != 0) {
        return;
    }

    const int64_t row_offset = cell * capacity;
    const int64_t count_offset = cell;
    const int64_t cache_offset = row_offset * cache_capacity;
    if (cache_width <= 0 || cache_width > cache_capacity) {
        status[count_offset] = STATUS_INVALID_CACHE_WIDTH;
        return;
    }

    const int64_t row_count = active_count[count_offset];
    if (row_count < 0 || row_count > capacity) {
        status[count_offset] = STATUS_INVALID_COUNT;
        return;
    }

    for (int64_t row = 0; row < row_count; ++row) {
        const int32_t slot = slots[row_offset + row];
        if (slot < 0 || static_cast<int64_t>(slot) >= capacity) {
            status[count_offset] = STATUS_INVALID_SLOT;
            return;
        }
    }

    int64_t hit_count = 0;
    int64_t miss_count = 0;
    for (int64_t row = 0; row < row_count; ++row) {
        const int64_t row_index = row_offset + row;
        const int32_t slot = slots[row_index];
        const int64_t slot_index = row_offset + static_cast<int64_t>(slot);
        const int32_t cursor = action_cache_cursor[slot_index];
        const bool hit = action_cache_token[slot_index] == slot_tokens[slot_index]
            && cursor >= 0
            && static_cast<int64_t>(cursor) < cache_width;
        if (!hit) {
            miss_agent_ids[row_offset + miss_count] = active_agent_ids[row_index];
            miss_slots[row_offset + miss_count] = slot;
            ++miss_count;
            continue;
        }

        int32_t action = action_cache_actions[
            cache_offset + static_cast<int64_t>(slot) * cache_capacity + cursor
        ];
        if (action < 0 || action > ACT_ATTACK) {
            action = action_cache_next[slot_index];
        }
        if (action < 0 || action > ACT_ATTACK) {
            action = ACT_REST;
        }
        action_cache_next[slot_index] = action;
        action_cache_cursor[slot_index] = cursor + 1;
        ++hit_count;
    }

    requested[count_offset] = row_count;
    hits[count_offset] = hit_count;
    misses[count_offset] = miss_count;
    full_refill_tick[count_offset] = miss_count > 0 && hit_count == 0 ? 1 : 0;
    partial_refill_tick[count_offset] = miss_count > 0 && hit_count > 0 ? 1 : 0;
    full_hit_tick[count_offset] = hit_count > 0 && miss_count == 0 ? 1 : 0;
    status[count_offset] = STATUS_OK;
}

}  // namespace

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
) {
    TORCH_CHECK(active_agent_ids.is_cuda(), "active_agent_ids must be a CUDA tensor");
    const auto device = active_agent_ids.device();
    c10::cuda::CUDAGuard device_guard(device);

    check_cuda_contiguous(active_agent_ids, "active_agent_ids", device);
    TORCH_CHECK(
        active_agent_ids.scalar_type() == torch::kInt64,
        "active_agent_ids must be int64"
    );
    TORCH_CHECK(active_agent_ids.dim() == 1, "active_agent_ids must be [capacity]");
    const int64_t capacity = active_agent_ids.size(0);

    check_vector(slots, "slots", torch::kInt32, capacity, device);
    check_vector(slot_tokens, "slot_tokens", torch::kInt64, capacity, device);
    check_vector(
        action_cache_token,
        "action_cache_token",
        torch::kInt64,
        capacity,
        device
    );
    check_vector(
        action_cache_cursor,
        "action_cache_cursor",
        torch::kInt32,
        capacity,
        device
    );
    check_vector(
        action_cache_next,
        "action_cache_next",
        torch::kInt32,
        capacity,
        device
    );

    check_cuda_contiguous(active_count, "active_count", device);
    TORCH_CHECK(
        active_count.scalar_type() == torch::kInt64
            && active_count.dim() == 1
            && active_count.numel() == 1,
        "active_count must be int64[1]"
    );

    check_cuda_contiguous(action_cache_actions, "action_cache_actions", device);
    TORCH_CHECK(
        action_cache_actions.scalar_type() == torch::kInt32,
        "action_cache_actions must be int32"
    );
    TORCH_CHECK(
        action_cache_actions.dim() == 2 && action_cache_actions.size(0) == capacity,
        "action_cache_actions must be [capacity,cache_capacity]"
    );
    const int64_t cache_capacity = action_cache_actions.size(1);

    const auto int64_options = active_agent_ids.options().dtype(torch::kInt64);
    const auto int32_options = active_agent_ids.options().dtype(torch::kInt32);
    auto miss_agent_ids = torch::zeros({capacity}, int64_options);
    auto miss_slots = torch::zeros({capacity}, int32_options);
    auto requested = torch::zeros({1}, int64_options);
    auto hits = torch::zeros({1}, int64_options);
    auto misses = torch::zeros({1}, int64_options);
    auto full_refill_tick = torch::zeros({1}, int64_options);
    auto partial_refill_tick = torch::zeros({1}, int64_options);
    auto full_hit_tick = torch::zeros({1}, int64_options);
    auto status = torch::zeros({1}, int32_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    policy_cache_service_kernel<<<1, 1, 0, stream>>>(
        active_agent_ids.data_ptr<int64_t>(),
        active_count.data_ptr<int64_t>(),
        slots.data_ptr<int32_t>(),
        slot_tokens.data_ptr<int64_t>(),
        action_cache_actions.data_ptr<int32_t>(),
        action_cache_token.data_ptr<int64_t>(),
        action_cache_cursor.data_ptr<int32_t>(),
        action_cache_next.data_ptr<int32_t>(),
        miss_agent_ids.data_ptr<int64_t>(),
        miss_slots.data_ptr<int32_t>(),
        requested.data_ptr<int64_t>(),
        hits.data_ptr<int64_t>(),
        misses.data_ptr<int64_t>(),
        full_refill_tick.data_ptr<int64_t>(),
        partial_refill_tick.data_ptr<int64_t>(),
        full_hit_tick.data_ptr<int64_t>(),
        status.data_ptr<int32_t>(),
        capacity,
        cache_capacity,
        cache_width
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        miss_agent_ids,
        miss_slots,
        requested,
        hits,
        misses,
        full_refill_tick,
        partial_refill_tick,
        full_hit_tick,
        status,
    };
}

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
) {
    TORCH_CHECK(
        active_agent_ids.is_cuda(),
        "active_agent_ids must be a CUDA tensor"
    );
    const auto device = active_agent_ids.device();
    c10::cuda::CUDAGuard device_guard(device);
    check_cuda_contiguous(active_agent_ids, "active_agent_ids", device);
    TORCH_CHECK(
        active_agent_ids.scalar_type() == torch::kInt64
            && active_agent_ids.dim() == 2,
        "batched active_agent_ids must be int64[cells,capacity]"
    );
    const int64_t cell_count = active_agent_ids.size(0);
    const int64_t capacity = active_agent_ids.size(1);
    TORCH_CHECK(cell_count > 0, "batched policy cache requires at least one cell");

    const auto check_batched_vector = [&](const torch::Tensor& tensor,
                                          const char* name,
                                          c10::ScalarType dtype) {
        check_cuda_contiguous(tensor, name, device);
        TORCH_CHECK(
            tensor.scalar_type() == dtype
                && tensor.dim() == 2
                && tensor.size(0) == cell_count
                && tensor.size(1) == capacity,
            name,
            " must be [cells,capacity]"
        );
    };
    check_batched_vector(slots, "slots", torch::kInt32);
    check_batched_vector(slot_tokens, "slot_tokens", torch::kInt64);
    check_batched_vector(action_cache_token, "action_cache_token", torch::kInt64);
    check_batched_vector(action_cache_cursor, "action_cache_cursor", torch::kInt32);
    check_batched_vector(action_cache_next, "action_cache_next", torch::kInt32);

    check_cuda_contiguous(active_count, "active_count", device);
    TORCH_CHECK(
        active_count.scalar_type() == torch::kInt64
            && active_count.dim() == 2
            && active_count.size(0) == cell_count
            && active_count.size(1) == 1,
        "batched active_count must be int64[cells,1]"
    );
    check_cuda_contiguous(action_cache_actions, "action_cache_actions", device);
    TORCH_CHECK(
        action_cache_actions.scalar_type() == torch::kInt32
            && action_cache_actions.dim() == 3
            && action_cache_actions.size(0) == cell_count
            && action_cache_actions.size(1) == capacity,
        "batched action_cache_actions must be int32[cells,capacity,cache_capacity]"
    );
    const int64_t cache_capacity = action_cache_actions.size(2);

    const auto int64_options = active_agent_ids.options().dtype(torch::kInt64);
    const auto int32_options = active_agent_ids.options().dtype(torch::kInt32);
    auto miss_agent_ids = torch::zeros({cell_count, capacity}, int64_options);
    auto miss_slots = torch::zeros({cell_count, capacity}, int32_options);
    auto requested = torch::zeros({cell_count, 1}, int64_options);
    auto hits = torch::zeros({cell_count, 1}, int64_options);
    auto misses = torch::zeros({cell_count, 1}, int64_options);
    auto full_refill_tick = torch::zeros({cell_count, 1}, int64_options);
    auto partial_refill_tick = torch::zeros({cell_count, 1}, int64_options);
    auto full_hit_tick = torch::zeros({cell_count, 1}, int64_options);
    auto status = torch::zeros({cell_count, 1}, int32_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    policy_cache_service_batched_kernel<<<
        static_cast<unsigned int>(cell_count),
        1,
        0,
        stream
    >>>(
        active_agent_ids.data_ptr<int64_t>(),
        active_count.data_ptr<int64_t>(),
        slots.data_ptr<int32_t>(),
        slot_tokens.data_ptr<int64_t>(),
        action_cache_actions.data_ptr<int32_t>(),
        action_cache_token.data_ptr<int64_t>(),
        action_cache_cursor.data_ptr<int32_t>(),
        action_cache_next.data_ptr<int32_t>(),
        miss_agent_ids.data_ptr<int64_t>(),
        miss_slots.data_ptr<int32_t>(),
        requested.data_ptr<int64_t>(),
        hits.data_ptr<int64_t>(),
        misses.data_ptr<int64_t>(),
        full_refill_tick.data_ptr<int64_t>(),
        partial_refill_tick.data_ptr<int64_t>(),
        full_hit_tick.data_ptr<int64_t>(),
        status.data_ptr<int32_t>(),
        cell_count,
        capacity,
        cache_capacity,
        cache_width
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        miss_agent_ids,
        miss_slots,
        requested,
        hits,
        misses,
        full_refill_tick,
        partial_refill_tick,
        full_hit_tick,
        status,
    };
}
