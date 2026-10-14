#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "cuda_architecture_b_pcg64.cuh"

namespace {

constexpr int64_t ACT_GATHER = 4;
constexpr int64_t ACT_REST = 5;
constexpr int64_t ACT_ATTACK = 6;
constexpr float GUMBEL_EPS = 1.0e-6f;

void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(
    const torch::Tensor& reference,
    const torch::Tensor& tensor,
    const char* name
) {
    TORCH_CHECK(tensor.device() == reference.device(), name, " must share the observation device");
}

__device__ __forceinline__ uint64_t splitmix64(const uint64_t value) {
    uint64_t z = value + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}

__device__ __forceinline__ float counter_uniform_float32(
    const uint64_t seed,
    const uint64_t draw_index
) {
    const uint64_t raw = splitmix64(seed + draw_index);
    const uint32_t mantissa = static_cast<uint32_t>(raw >> 40U);
    float value = (static_cast<float>(mantissa) + 0.5f) * (1.0f / 16777216.0f);
    if (value < GUMBEL_EPS) {
        value = GUMBEL_EPS;
    } else if (value > (1.0f - GUMBEL_EPS)) {
        value = 1.0f - GUMBEL_EPS;
    }
    return value;
}

__device__ __forceinline__ int64_t bounded_active_count_cuda(
    const int64_t* __restrict__ active_count,
    const int64_t capacity
) {
    if (active_count == nullptr) {
        return capacity;
    }
    const int64_t count = active_count[0];
    if (count <= 0) {
        return 0;
    }
    return count < capacity ? count : capacity;
}

__global__ void fill_fixed_mlp_cache_kernel(
    const float* __restrict__ logits,
    const int32_t* __restrict__ slots,
    const int64_t* __restrict__ slot_tokens,
    int32_t* __restrict__ action_cache_actions,
    int64_t* __restrict__ action_cache_token,
    int32_t* __restrict__ action_cache_cursor,
    int32_t* __restrict__ action_cache_next,
    int32_t* __restrict__ row_actions,
    const int64_t row_count,
    const int64_t* __restrict__ active_count,
    const int64_t capacity,
    const int64_t action_count,
    const int64_t cache_capacity,
    const int64_t cache_width,
    const bool deterministic,
    const uint64_t rng_seed,
    const uint64_t rng_counter
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= bounded_active_count_cuda(active_count, row_count)) {
        return;
    }

    const int32_t slot = slots[row];
    if (slot < 0 || static_cast<int64_t>(slot) >= capacity) {
        row_actions[row] = -1;
        return;
    }
    for (int64_t cache_index = 0; cache_index < cache_width; ++cache_index) {
        int32_t best_action = 0;
        float best_score = logits[row * action_count];
        if (!deterministic) {
            const uint64_t first_draw = rng_counter
                + static_cast<uint64_t>((row * cache_width + cache_index) * action_count);
            const float uniform = counter_uniform_float32(rng_seed, first_draw);
            best_score += -logf(-logf(uniform));
        }
        for (int64_t action = 1; action < action_count; ++action) {
            float score = logits[row * action_count + action];
            if (!deterministic) {
                const uint64_t draw_index = rng_counter
                    + static_cast<uint64_t>(
                        (row * cache_width + cache_index) * action_count + action
                    );
                const float uniform = counter_uniform_float32(rng_seed, draw_index);
                score += -logf(-logf(uniform));
            }
            if (score > best_score) {
                best_score = score;
                best_action = static_cast<int32_t>(action);
            }
        }
        action_cache_actions[static_cast<int64_t>(slot) * cache_capacity + cache_index] = best_action;
        if (cache_index == 0) {
            row_actions[row] = best_action;
            action_cache_next[slot] = best_action;
        }
    }
    action_cache_token[slot] = slot_tokens[slot];
    action_cache_cursor[slot] = 1;
}

__global__ void fill_fixed_mlp_cache_pcg64_kernel(
    const float* __restrict__ logits,
    const int32_t* __restrict__ slots,
    const int64_t* __restrict__ slot_tokens,
    int32_t* __restrict__ action_cache_actions,
    int64_t* __restrict__ action_cache_token,
    int32_t* __restrict__ action_cache_cursor,
    int32_t* __restrict__ action_cache_next,
    int32_t* __restrict__ row_actions,
    uint64_t* __restrict__ state_words,
    int64_t* __restrict__ buffer_words,
    const int64_t row_count,
    const int64_t* __restrict__ active_count,
    const int64_t capacity,
    const int64_t action_count,
    const int64_t cache_capacity,
    const int64_t cache_width
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    eiel::cuda_architecture_b::Pcg64State rng{
        state_words[0],
        state_words[1],
        state_words[2],
        state_words[3],
        static_cast<uint32_t>(buffer_words[0]),
        static_cast<uint32_t>(buffer_words[1]),
    };
    const int64_t active_rows = bounded_active_count_cuda(active_count, row_count);
    for (int64_t row = 0; row < active_rows; ++row) {
        const int32_t slot = slots[row];
        if (slot < 0 || static_cast<int64_t>(slot) >= capacity) {
            row_actions[row] = -1;
            return;
        }
        for (int64_t cache_index = 0; cache_index < cache_width; ++cache_index) {
            int32_t best_action = 0;
            float first_uniform = eiel::cuda_architecture_b::pcg64_numpy_float32(rng);
            if (first_uniform < GUMBEL_EPS) {
                first_uniform = GUMBEL_EPS;
            } else if (first_uniform > (1.0f - GUMBEL_EPS)) {
                first_uniform = 1.0f - GUMBEL_EPS;
            }
            float best_score = logits[row * action_count]
                - logf(-logf(first_uniform));
            for (int64_t action = 1; action < action_count; ++action) {
                float uniform = eiel::cuda_architecture_b::pcg64_numpy_float32(rng);
                if (uniform < GUMBEL_EPS) {
                    uniform = GUMBEL_EPS;
                } else if (uniform > (1.0f - GUMBEL_EPS)) {
                    uniform = 1.0f - GUMBEL_EPS;
                }
                const float score = logits[row * action_count + action]
                    - logf(-logf(uniform));
                if (score > best_score) {
                    best_score = score;
                    best_action = static_cast<int32_t>(action);
                }
            }
            action_cache_actions[static_cast<int64_t>(slot) * cache_capacity + cache_index] = best_action;
            if (cache_index == 0) {
                row_actions[row] = best_action;
                action_cache_next[slot] = best_action;
            }
        }
        action_cache_token[slot] = slot_tokens[slot];
        action_cache_cursor[slot] = 1;
    }
    state_words[0] = rng.state_lo;
    state_words[1] = rng.state_hi;
    state_words[2] = rng.inc_lo;
    state_words[3] = rng.inc_hi;
    buffer_words[0] = static_cast<int64_t>(rng.has_uint32);
    buffer_words[1] = static_cast<int64_t>(rng.uinteger);
}

__global__ void fill_fixed_mlp_cache_batched_kernel(
    const float* __restrict__ logits,
    const int32_t* __restrict__ slots,
    const int64_t* __restrict__ active_count,
    const int64_t* __restrict__ slot_tokens,
    int32_t* __restrict__ action_cache_actions,
    int64_t* __restrict__ action_cache_token,
    int32_t* __restrict__ action_cache_cursor,
    int32_t* __restrict__ action_cache_next,
    int32_t* __restrict__ row_actions,
    const int64_t cell_count,
    const int64_t row_count,
    const int64_t capacity,
    const int64_t action_count,
    const int64_t cache_capacity,
    const int64_t cache_width,
    const bool deterministic,
    const uint64_t rng_seed,
    const uint64_t rng_counter
) {
    const int64_t flat_row = static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (flat_row >= cell_count * row_count) {
        return;
    }
    const int64_t cell = flat_row / row_count;
    const int64_t row = flat_row - cell * row_count;
    if (row >= bounded_active_count_cuda(active_count + cell, row_count)) {
        return;
    }
    const int32_t slot = slots[flat_row];
    if (slot < 0 || static_cast<int64_t>(slot) >= capacity) {
        row_actions[flat_row] = -1;
        return;
    }
    const int64_t slot_index = cell * capacity + static_cast<int64_t>(slot);
    const int64_t cache_offset = slot_index * cache_capacity;
    for (int64_t cache_index = 0; cache_index < cache_width; ++cache_index) {
        int32_t best_action = 0;
        float best_score = logits[flat_row * action_count];
        if (!deterministic) {
            const uint64_t first_draw = rng_counter
                + static_cast<uint64_t>((row * cache_width + cache_index) * action_count);
            best_score += -logf(-logf(counter_uniform_float32(rng_seed, first_draw)));
        }
        for (int64_t action = 1; action < action_count; ++action) {
            float score = logits[flat_row * action_count + action];
            if (!deterministic) {
                const uint64_t draw_index = rng_counter
                    + static_cast<uint64_t>(
                        (row * cache_width + cache_index) * action_count + action
                    );
                score += -logf(-logf(counter_uniform_float32(rng_seed, draw_index)));
            }
            if (score > best_score) {
                best_score = score;
                best_action = static_cast<int32_t>(action);
            }
        }
        action_cache_actions[cache_offset + cache_index] = best_action;
        if (cache_index == 0) {
            row_actions[flat_row] = best_action;
            action_cache_next[slot_index] = best_action;
        }
    }
    action_cache_token[slot_index] = slot_tokens[slot_index];
    action_cache_cursor[slot_index] = 1;
}

__global__ void fill_fixed_mlp_cache_pcg64_batched_kernel(
    const float* __restrict__ logits,
    const int32_t* __restrict__ slots,
    const int64_t* __restrict__ active_count,
    const int64_t* __restrict__ slot_tokens,
    int32_t* __restrict__ action_cache_actions,
    int64_t* __restrict__ action_cache_token,
    int32_t* __restrict__ action_cache_cursor,
    int32_t* __restrict__ action_cache_next,
    int32_t* __restrict__ row_actions,
    uint64_t* __restrict__ state_words,
    int64_t* __restrict__ buffer_words,
    const int64_t cell_count,
    const int64_t row_count,
    const int64_t capacity,
    const int64_t action_count,
    const int64_t cache_capacity,
    const int64_t cache_width
) {
    const int64_t cell = static_cast<int64_t>(blockIdx.x);
    if (cell >= cell_count || threadIdx.x != 0) {
        return;
    }
    uint64_t* cell_state = state_words + cell * 4;
    int64_t* cell_buffer = buffer_words + cell * 2;
    eiel::cuda_architecture_b::Pcg64State rng{
        cell_state[0],
        cell_state[1],
        cell_state[2],
        cell_state[3],
        static_cast<uint32_t>(cell_buffer[0]),
        static_cast<uint32_t>(cell_buffer[1]),
    };
    const int64_t active_rows = bounded_active_count_cuda(
        active_count + cell,
        row_count
    );
    const int64_t row_offset = cell * row_count;
    const int64_t slot_offset = cell * capacity;
    for (int64_t row = 0; row < active_rows; ++row) {
        const int64_t flat_row = row_offset + row;
        const int32_t slot = slots[flat_row];
        if (slot < 0 || static_cast<int64_t>(slot) >= capacity) {
            row_actions[flat_row] = -1;
            return;
        }
        const int64_t slot_index = slot_offset + static_cast<int64_t>(slot);
        const int64_t cache_offset = slot_index * cache_capacity;
        for (int64_t cache_index = 0; cache_index < cache_width; ++cache_index) {
            int32_t best_action = 0;
            float first_uniform = eiel::cuda_architecture_b::pcg64_numpy_float32(rng);
            if (first_uniform < GUMBEL_EPS) {
                first_uniform = GUMBEL_EPS;
            } else if (first_uniform > (1.0f - GUMBEL_EPS)) {
                first_uniform = 1.0f - GUMBEL_EPS;
            }
            float best_score = logits[flat_row * action_count]
                - logf(-logf(first_uniform));
            for (int64_t action = 1; action < action_count; ++action) {
                float uniform = eiel::cuda_architecture_b::pcg64_numpy_float32(rng);
                if (uniform < GUMBEL_EPS) {
                    uniform = GUMBEL_EPS;
                } else if (uniform > (1.0f - GUMBEL_EPS)) {
                    uniform = 1.0f - GUMBEL_EPS;
                }
                const float score = logits[flat_row * action_count + action]
                    - logf(-logf(uniform));
                if (score > best_score) {
                    best_score = score;
                    best_action = static_cast<int32_t>(action);
                }
            }
            action_cache_actions[cache_offset + cache_index] = best_action;
            if (cache_index == 0) {
                row_actions[flat_row] = best_action;
                action_cache_next[slot_index] = best_action;
            }
        }
        action_cache_token[slot_index] = slot_tokens[slot_index];
        action_cache_cursor[slot_index] = 1;
    }
    cell_state[0] = rng.state_lo;
    cell_state[1] = rng.state_hi;
    cell_state[2] = rng.inc_lo;
    cell_state[3] = rng.inc_hi;
    cell_buffer[0] = static_cast<int64_t>(rng.has_uint32);
    cell_buffer[1] = static_cast<int64_t>(rng.uinteger);
}

std::vector<torch::Tensor> fixed_mlp_refill_impl(
    torch::Tensor observations,
    torch::Tensor slots,
    torch::Tensor active_count,
    torch::Tensor logit_residuals,
    torch::Tensor mlp_pack,
    torch::Tensor drives,
    torch::Tensor attack_bias,
    torch::Tensor action_temperature,
    torch::Tensor drive_logit_gain,
    torch::Tensor slot_tokens,
    torch::Tensor action_cache_actions,
    torch::Tensor action_cache_token,
    torch::Tensor action_cache_cursor,
    torch::Tensor action_cache_next,
    torch::Tensor pcg64_state_words,
    torch::Tensor pcg64_buffer_words,
    int64_t hidden_size,
    int64_t action_count,
    int64_t cache_width,
    bool deterministic,
    int64_t rng_seed,
    int64_t rng_counter,
    double w_hunger_gather,
    double w_pain_rest,
    double w_fatigue_rest,
    double w_pain_attack,
    double w_fear_attack,
    bool counted_rows
);

}  // namespace

std::vector<torch::Tensor> eiel_cuda_fixed_mlp_refill_batched_counted_v1(
    torch::Tensor observations,
    torch::Tensor slots,
    torch::Tensor active_count,
    torch::Tensor mlp_pack,
    torch::Tensor drives,
    torch::Tensor attack_bias,
    torch::Tensor action_temperature,
    torch::Tensor drive_logit_gain,
    torch::Tensor slot_tokens,
    torch::Tensor action_cache_actions,
    torch::Tensor action_cache_token,
    torch::Tensor action_cache_cursor,
    torch::Tensor action_cache_next,
    torch::Tensor pcg64_state_words,
    torch::Tensor pcg64_buffer_words,
    const int64_t hidden_size,
    const int64_t action_count,
    const int64_t cache_width,
    const bool deterministic,
    const int64_t rng_seed,
    const int64_t rng_counter,
    const double w_hunger_gather,
    const double w_pain_rest,
    const double w_fatigue_rest,
    const double w_pain_attack,
    const double w_fear_attack
) {
    check_cuda_tensor(observations, "observations");
    const auto device = observations.device();
    c10::cuda::CUDAGuard device_guard(device);
    const auto check = [&](const torch::Tensor& tensor, const char* name) {
        check_cuda_tensor(tensor, name);
        check_same_device(observations, tensor, name);
    };
    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"slots", slots},
             {"active_count", active_count},
             {"mlp_pack", mlp_pack},
             {"drives", drives},
             {"attack_bias", attack_bias},
             {"action_temperature", action_temperature},
             {"drive_logit_gain", drive_logit_gain},
             {"slot_tokens", slot_tokens},
             {"action_cache_actions", action_cache_actions},
             {"action_cache_token", action_cache_token},
             {"action_cache_cursor", action_cache_cursor},
             {"action_cache_next", action_cache_next},
             {"pcg64_state_words", pcg64_state_words},
             {"pcg64_buffer_words", pcg64_buffer_words},
         }) {
        check(named.second, named.first);
    }

    TORCH_CHECK(
        observations.scalar_type() == torch::kFloat32
            && observations.dim() == 3,
        "batched observations must be float32[cells,rows,features]"
    );
    const int64_t cell_count = observations.size(0);
    const int64_t row_count = observations.size(1);
    const int64_t observation_size = observations.size(2);
    TORCH_CHECK(cell_count > 0, "batched fixed MLP requires at least one cell");
    TORCH_CHECK(
        slots.scalar_type() == torch::kInt32
            && slots.sizes() == torch::IntArrayRef({cell_count, row_count}),
        "batched slots must be int32[cells,rows]"
    );
    TORCH_CHECK(
        active_count.scalar_type() == torch::kInt64
            && active_count.dim() == 2
            && active_count.size(0) == cell_count
            && active_count.size(1) == 1,
        "batched active_count must be int64[cells,1]"
    );
    TORCH_CHECK(
        mlp_pack.scalar_type() == torch::kFloat32 && mlp_pack.dim() == 3,
        "batched mlp_pack must be float32[cells,capacity,packed_width]"
    );
    const int64_t capacity = mlp_pack.size(1);
    TORCH_CHECK(mlp_pack.size(0) == cell_count, "mlp_pack cell count mismatch");
    const auto check_cell_capacity = [&](const torch::Tensor& tensor,
                                         const char* name,
                                         c10::ScalarType dtype,
                                         int64_t rank) {
        TORCH_CHECK(
            tensor.scalar_type() == dtype
                && tensor.dim() == rank
                && tensor.size(0) == cell_count
                && tensor.size(1) == capacity,
            name,
            " has an invalid batched cell/capacity shape"
        );
    };
    check_cell_capacity(drives, "drives", torch::kFloat32, 3);
    TORCH_CHECK(drives.size(2) >= 4, "drives must expose at least four channels");
    check_cell_capacity(attack_bias, "attack_bias", torch::kFloat32, 2);
    check_cell_capacity(
        action_temperature,
        "action_temperature",
        torch::kFloat32,
        2
    );
    check_cell_capacity(
        drive_logit_gain,
        "drive_logit_gain",
        torch::kFloat32,
        2
    );
    check_cell_capacity(slot_tokens, "slot_tokens", torch::kInt64, 2);
    check_cell_capacity(
        action_cache_actions,
        "action_cache_actions",
        torch::kInt32,
        3
    );
    check_cell_capacity(
        action_cache_token,
        "action_cache_token",
        torch::kInt64,
        2
    );
    check_cell_capacity(
        action_cache_cursor,
        "action_cache_cursor",
        torch::kInt32,
        2
    );
    check_cell_capacity(
        action_cache_next,
        "action_cache_next",
        torch::kInt32,
        2
    );
    TORCH_CHECK(hidden_size > 0, "hidden_size must be positive");
    TORCH_CHECK(action_count >= 7, "action_count must include the fixed action set");
    TORCH_CHECK(cache_width > 0, "cache_width must be positive");
    const int64_t cache_capacity = action_cache_actions.size(2);
    TORCH_CHECK(cache_width <= cache_capacity, "cache_width exceeds cache capacity");
    const int64_t packed_width = hidden_size * observation_size
        + hidden_size
        + hidden_size * hidden_size
        + hidden_size
        + action_count * hidden_size
        + action_count;
    TORCH_CHECK(mlp_pack.size(2) == packed_width, "fixed MLP packed width mismatch");

    const bool exact_pcg64 = pcg64_state_words.numel() != 0
        || pcg64_buffer_words.numel() != 0;
    TORCH_CHECK(
        !exact_pcg64
            || (pcg64_state_words.scalar_type() == torch::kUInt64
                && pcg64_state_words.dim() == 2
                && pcg64_state_words.size(0) == cell_count
                && pcg64_state_words.size(1) == 4),
        "batched exact PCG64 state must be uint64[cells,4]"
    );
    TORCH_CHECK(
        !exact_pcg64
            || (pcg64_buffer_words.scalar_type() == torch::kInt64
                && pcg64_buffer_words.dim() == 2
                && pcg64_buffer_words.size(0) == cell_count
                && pcg64_buffer_words.size(1) == 2),
        "batched exact PCG64 buffer must be int64[cells,2]"
    );

    auto row_indices = torch::arange(
        row_count,
        slots.options().dtype(torch::kInt64)
    ).reshape({1, row_count});
    auto active_mask = row_indices < torch::clamp(active_count, 0, row_count);
    auto safe_slots = torch::where(
        active_mask,
        slots,
        torch::zeros_like(slots)
    ).contiguous();
    auto cell_offsets = torch::arange(
        cell_count,
        slots.options().dtype(torch::kInt64)
    ).reshape({cell_count, 1}) * capacity;
    auto global_slots = (
        safe_slots.to(torch::kInt64) + cell_offsets
    ).reshape({cell_count * row_count});
    auto flat_pack = mlp_pack.reshape({cell_count * capacity, packed_width});
    auto pack = flat_pack.index_select(0, global_slots).contiguous();
    auto x = observations.reshape({cell_count * row_count, observation_size});
    auto flat_attack_bias = attack_bias.reshape({cell_count * capacity});
    auto flat_temperature = action_temperature.reshape({cell_count * capacity});
    auto flat_gain = drive_logit_gain.reshape({cell_count * capacity});
    auto flat_drives = drives.reshape({cell_count * capacity, drives.size(2)});
    auto aggr = flat_attack_bias.index_select(0, global_slots).contiguous();
    auto temp = flat_temperature.index_select(0, global_slots).contiguous();
    auto selected_drives = flat_drives.index_select(0, global_slots).contiguous();
    auto gain = torch::clamp(
        flat_gain.index_select(0, global_slots).contiguous(),
        0.0
    );

    int64_t offset = 0;
    auto w1 = pack.slice(1, offset, offset + hidden_size * observation_size)
        .view({-1, hidden_size, observation_size});
    offset += hidden_size * observation_size;
    auto b1 = pack.slice(1, offset, offset + hidden_size).view({-1, hidden_size});
    offset += hidden_size;
    auto w2 = pack.slice(1, offset, offset + hidden_size * hidden_size)
        .view({-1, hidden_size, hidden_size});
    offset += hidden_size * hidden_size;
    auto b2 = pack.slice(1, offset, offset + hidden_size).view({-1, hidden_size});
    offset += hidden_size;
    auto w3 = pack.slice(1, offset, offset + action_count * hidden_size)
        .view({-1, action_count, hidden_size});
    offset += action_count * hidden_size;
    auto b3 = pack.slice(1, offset, offset + action_count)
        .view({-1, action_count});
    auto h1 = torch::tanh(torch::bmm(w1, x.unsqueeze(-1)).squeeze(-1) + b1);
    auto h2 = torch::tanh(torch::bmm(w2, h1.unsqueeze(-1)).squeeze(-1) + b2);
    auto logits_policy = torch::bmm(w3, h2.unsqueeze(-1)).squeeze(-1) + b3;
    namespace idx = torch::indexing;
    logits_policy.index_put_(
        {idx::Slice(), ACT_ATTACK},
        logits_policy.index({idx::Slice(), ACT_ATTACK}) + aggr
    );
    logits_policy = logits_policy / torch::clamp(temp, 0.05).unsqueeze(-1);
    auto logits = logits_policy.clone();
    logits.index_put_(
        {idx::Slice(), ACT_GATHER},
        logits.index({idx::Slice(), ACT_GATHER})
            + gain * (w_hunger_gather * selected_drives.index({idx::Slice(), 0}))
    );
    logits.index_put_(
        {idx::Slice(), ACT_REST},
        logits.index({idx::Slice(), ACT_REST})
            + gain * (
                w_pain_rest * selected_drives.index({idx::Slice(), 1})
                + w_fatigue_rest * selected_drives.index({idx::Slice(), 2})
            )
    );
    logits.index_put_(
        {idx::Slice(), ACT_ATTACK},
        logits.index({idx::Slice(), ACT_ATTACK})
            - gain * (
                w_pain_attack * selected_drives.index({idx::Slice(), 1})
                + w_fear_attack * selected_drives.index({idx::Slice(), 3})
            )
    );
    logits = torch::where(
        active_mask.reshape({cell_count * row_count, 1}),
        logits,
        torch::zeros_like(logits)
    );
    auto row_actions = torch::full(
        {cell_count, row_count},
        -1,
        observations.options().dtype(torch::kInt32)
    );
    if (cell_count * row_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        if (!deterministic && exact_pcg64) {
            fill_fixed_mlp_cache_pcg64_batched_kernel<<<
                static_cast<unsigned int>(cell_count),
                1,
                0,
                stream
            >>>(
                logits.data_ptr<float>(),
                safe_slots.data_ptr<int32_t>(),
                active_count.data_ptr<int64_t>(),
                slot_tokens.data_ptr<int64_t>(),
                action_cache_actions.data_ptr<int32_t>(),
                action_cache_token.data_ptr<int64_t>(),
                action_cache_cursor.data_ptr<int32_t>(),
                action_cache_next.data_ptr<int32_t>(),
                row_actions.data_ptr<int32_t>(),
                pcg64_state_words.data_ptr<uint64_t>(),
                pcg64_buffer_words.data_ptr<int64_t>(),
                cell_count,
                row_count,
                capacity,
                action_count,
                cache_capacity,
                cache_width
            );
        } else {
            constexpr int threads = 128;
            const int64_t total_rows = cell_count * row_count;
            const int blocks = static_cast<int>((total_rows + threads - 1) / threads);
            fill_fixed_mlp_cache_batched_kernel<<<blocks, threads, 0, stream>>>(
                logits.data_ptr<float>(),
                safe_slots.data_ptr<int32_t>(),
                active_count.data_ptr<int64_t>(),
                slot_tokens.data_ptr<int64_t>(),
                action_cache_actions.data_ptr<int32_t>(),
                action_cache_token.data_ptr<int64_t>(),
                action_cache_cursor.data_ptr<int32_t>(),
                action_cache_next.data_ptr<int32_t>(),
                row_actions.data_ptr<int32_t>(),
                cell_count,
                row_count,
                capacity,
                action_count,
                cache_capacity,
                cache_width,
                deterministic,
                static_cast<uint64_t>(rng_seed),
                static_cast<uint64_t>(rng_counter)
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    torch::Tensor next_rng_counter;
    if (!deterministic && !exact_pcg64) {
        next_rng_counter = torch::clamp(active_count, 0, row_count)
            * (cache_width * action_count)
            + rng_counter;
    } else {
        next_rng_counter = torch::full(
            {cell_count, 1},
            rng_counter,
            observations.options().dtype(torch::kInt64)
        );
    }
    return {
        logits.reshape({cell_count, row_count, action_count}),
        row_actions,
        action_cache_actions,
        action_cache_token,
        action_cache_cursor,
        action_cache_next,
        next_rng_counter,
        pcg64_state_words,
        pcg64_buffer_words,
    };
}

std::vector<torch::Tensor> eiel_cuda_fixed_mlp_refill_v1(
    torch::Tensor observations,
    torch::Tensor slots,
    torch::Tensor mlp_pack,
    torch::Tensor drives,
    torch::Tensor attack_bias,
    torch::Tensor action_temperature,
    torch::Tensor drive_logit_gain,
    torch::Tensor slot_tokens,
    torch::Tensor action_cache_actions,
    torch::Tensor action_cache_token,
    torch::Tensor action_cache_cursor,
    torch::Tensor action_cache_next,
    torch::Tensor pcg64_state_words,
    torch::Tensor pcg64_buffer_words,
    const int64_t hidden_size,
    const int64_t action_count,
    const int64_t cache_width,
    const bool deterministic,
    const int64_t rng_seed,
    const int64_t rng_counter,
    const double w_hunger_gather,
    const double w_pain_rest,
    const double w_fatigue_rest,
    const double w_pain_attack,
    const double w_fear_attack
) {
    return fixed_mlp_refill_impl(
        observations,
        slots,
        torch::Tensor(),
        torch::Tensor(),
        mlp_pack,
        drives,
        attack_bias,
        action_temperature,
        drive_logit_gain,
        slot_tokens,
        action_cache_actions,
        action_cache_token,
        action_cache_cursor,
        action_cache_next,
        pcg64_state_words,
        pcg64_buffer_words,
        hidden_size,
        action_count,
        cache_width,
        deterministic,
        rng_seed,
        rng_counter,
        w_hunger_gather,
        w_pain_rest,
        w_fatigue_rest,
        w_pain_attack,
        w_fear_attack,
        false
    );
}

std::vector<torch::Tensor> eiel_cuda_fixed_mlp_refill_counted_v1(
    torch::Tensor observations,
    torch::Tensor slots,
    torch::Tensor active_count,
    torch::Tensor mlp_pack,
    torch::Tensor drives,
    torch::Tensor attack_bias,
    torch::Tensor action_temperature,
    torch::Tensor drive_logit_gain,
    torch::Tensor slot_tokens,
    torch::Tensor action_cache_actions,
    torch::Tensor action_cache_token,
    torch::Tensor action_cache_cursor,
    torch::Tensor action_cache_next,
    torch::Tensor pcg64_state_words,
    torch::Tensor pcg64_buffer_words,
    const int64_t hidden_size,
    const int64_t action_count,
    const int64_t cache_width,
    const bool deterministic,
    const int64_t rng_seed,
    const int64_t rng_counter,
    const double w_hunger_gather,
    const double w_pain_rest,
    const double w_fatigue_rest,
    const double w_pain_attack,
    const double w_fear_attack
) {
    return fixed_mlp_refill_impl(
        observations,
        slots,
        active_count,
        torch::Tensor(),
        mlp_pack,
        drives,
        attack_bias,
        action_temperature,
        drive_logit_gain,
        slot_tokens,
        action_cache_actions,
        action_cache_token,
        action_cache_cursor,
        action_cache_next,
        pcg64_state_words,
        pcg64_buffer_words,
        hidden_size,
        action_count,
        cache_width,
        deterministic,
        rng_seed,
        rng_counter,
        w_hunger_gather,
        w_pain_rest,
        w_fatigue_rest,
        w_pain_attack,
        w_fear_attack,
        true
    );
}

std::vector<torch::Tensor> eiel_cuda_fixed_mlp_refill_counted_with_residual_v1(
    torch::Tensor observations,
    torch::Tensor slots,
    torch::Tensor active_count,
    torch::Tensor logit_residuals,
    torch::Tensor mlp_pack,
    torch::Tensor drives,
    torch::Tensor attack_bias,
    torch::Tensor action_temperature,
    torch::Tensor drive_logit_gain,
    torch::Tensor slot_tokens,
    torch::Tensor action_cache_actions,
    torch::Tensor action_cache_token,
    torch::Tensor action_cache_cursor,
    torch::Tensor action_cache_next,
    torch::Tensor pcg64_state_words,
    torch::Tensor pcg64_buffer_words,
    const int64_t hidden_size,
    const int64_t action_count,
    const int64_t cache_width,
    const bool deterministic,
    const int64_t rng_seed,
    const int64_t rng_counter,
    const double w_hunger_gather,
    const double w_pain_rest,
    const double w_fatigue_rest,
    const double w_pain_attack,
    const double w_fear_attack
) {
    return fixed_mlp_refill_impl(
        observations,
        slots,
        active_count,
        logit_residuals,
        mlp_pack,
        drives,
        attack_bias,
        action_temperature,
        drive_logit_gain,
        slot_tokens,
        action_cache_actions,
        action_cache_token,
        action_cache_cursor,
        action_cache_next,
        pcg64_state_words,
        pcg64_buffer_words,
        hidden_size,
        action_count,
        cache_width,
        deterministic,
        rng_seed,
        rng_counter,
        w_hunger_gather,
        w_pain_rest,
        w_fatigue_rest,
        w_pain_attack,
        w_fear_attack,
        true
    );
}

namespace {

std::vector<torch::Tensor> fixed_mlp_refill_impl(
    torch::Tensor observations,
    torch::Tensor slots,
    torch::Tensor active_count,
    torch::Tensor logit_residuals,
    torch::Tensor mlp_pack,
    torch::Tensor drives,
    torch::Tensor attack_bias,
    torch::Tensor action_temperature,
    torch::Tensor drive_logit_gain,
    torch::Tensor slot_tokens,
    torch::Tensor action_cache_actions,
    torch::Tensor action_cache_token,
    torch::Tensor action_cache_cursor,
    torch::Tensor action_cache_next,
    torch::Tensor pcg64_state_words,
    torch::Tensor pcg64_buffer_words,
    const int64_t hidden_size,
    const int64_t action_count,
    const int64_t cache_width,
    const bool deterministic,
    const int64_t rng_seed,
    const int64_t rng_counter,
    const double w_hunger_gather,
    const double w_pain_rest,
    const double w_fatigue_rest,
    const double w_pain_attack,
    const double w_fear_attack,
    const bool counted_rows
) {
    check_cuda_tensor(observations, "observations");
    check_cuda_tensor(slots, "slots");
    check_cuda_tensor(mlp_pack, "mlp_pack");
    check_cuda_tensor(drives, "drives");
    check_cuda_tensor(attack_bias, "attack_bias");
    check_cuda_tensor(action_temperature, "action_temperature");
    check_cuda_tensor(drive_logit_gain, "drive_logit_gain");
    check_cuda_tensor(slot_tokens, "slot_tokens");
    check_cuda_tensor(action_cache_actions, "action_cache_actions");
    check_cuda_tensor(action_cache_token, "action_cache_token");
    check_cuda_tensor(action_cache_cursor, "action_cache_cursor");
    check_cuda_tensor(action_cache_next, "action_cache_next");
    check_cuda_tensor(pcg64_state_words, "pcg64_state_words");
    check_cuda_tensor(pcg64_buffer_words, "pcg64_buffer_words");
    if (counted_rows) {
        check_cuda_tensor(active_count, "active_count");
    }
    const bool residuals_enabled = logit_residuals.defined();
    if (residuals_enabled) {
        check_cuda_tensor(logit_residuals, "logit_residuals");
    }

    check_same_device(observations, slots, "slots");
    check_same_device(observations, mlp_pack, "mlp_pack");
    check_same_device(observations, drives, "drives");
    check_same_device(observations, attack_bias, "attack_bias");
    check_same_device(observations, action_temperature, "action_temperature");
    check_same_device(observations, drive_logit_gain, "drive_logit_gain");
    check_same_device(observations, slot_tokens, "slot_tokens");
    check_same_device(observations, action_cache_actions, "action_cache_actions");
    check_same_device(observations, action_cache_token, "action_cache_token");
    check_same_device(observations, action_cache_cursor, "action_cache_cursor");
    check_same_device(observations, action_cache_next, "action_cache_next");
    check_same_device(observations, pcg64_state_words, "pcg64_state_words");
    check_same_device(observations, pcg64_buffer_words, "pcg64_buffer_words");
    if (counted_rows) {
        check_same_device(observations, active_count, "active_count");
    }
    if (residuals_enabled) {
        check_same_device(observations, logit_residuals, "logit_residuals");
    }

    TORCH_CHECK(observations.scalar_type() == torch::kFloat32, "observations must be float32");
    TORCH_CHECK(
        !residuals_enabled || logit_residuals.scalar_type() == torch::kFloat32,
        "logit_residuals must be float32"
    );
    TORCH_CHECK(slots.scalar_type() == torch::kInt32, "slots must be int32");
    TORCH_CHECK(mlp_pack.scalar_type() == torch::kFloat32, "mlp_pack must be float32");
    TORCH_CHECK(drives.scalar_type() == torch::kFloat32, "drives must be float32");
    TORCH_CHECK(attack_bias.scalar_type() == torch::kFloat32, "attack_bias must be float32");
    TORCH_CHECK(action_temperature.scalar_type() == torch::kFloat32, "action_temperature must be float32");
    TORCH_CHECK(drive_logit_gain.scalar_type() == torch::kFloat32, "drive_logit_gain must be float32");
    TORCH_CHECK(slot_tokens.scalar_type() == torch::kInt64, "slot_tokens must be int64");
    TORCH_CHECK(action_cache_actions.scalar_type() == torch::kInt32, "action_cache_actions must be int32");
    TORCH_CHECK(action_cache_token.scalar_type() == torch::kInt64, "action_cache_token must be int64");
    TORCH_CHECK(action_cache_cursor.scalar_type() == torch::kInt32, "action_cache_cursor must be int32");
    TORCH_CHECK(action_cache_next.scalar_type() == torch::kInt32, "action_cache_next must be int32");
    TORCH_CHECK(pcg64_state_words.scalar_type() == torch::kUInt64, "pcg64_state_words must be uint64");
    TORCH_CHECK(pcg64_buffer_words.scalar_type() == torch::kInt64, "pcg64_buffer_words must be int64");
    TORCH_CHECK(
        !counted_rows || (
            active_count.scalar_type() == torch::kInt64
            && active_count.dim() == 1
            && active_count.numel() == 1
        ),
        "active_count must be CUDA int64[1]"
    );

    TORCH_CHECK(observations.dim() == 2, "observations must be [rows, observation_size]");
    TORCH_CHECK(
        !residuals_enabled
            || (logit_residuals.dim() == 2
                && logit_residuals.size(0) == observations.size(0)
                && logit_residuals.size(1) == action_count),
        "logit_residuals must be [rows, action_count]"
    );
    TORCH_CHECK(
        !residuals_enabled || logit_residuals.is_contiguous(),
        "logit_residuals must be contiguous"
    );
    TORCH_CHECK(slots.dim() == 1 && slots.size(0) == observations.size(0), "slots must match observation rows");
    TORCH_CHECK(mlp_pack.dim() == 2, "mlp_pack must be [capacity, packed_policy_width]");
    TORCH_CHECK(drives.dim() == 2 && drives.size(1) >= 4, "drives must be [capacity, at_least_4]");
    TORCH_CHECK(attack_bias.dim() == 1, "attack_bias must be [capacity]");
    TORCH_CHECK(action_temperature.dim() == 1, "action_temperature must be [capacity]");
    TORCH_CHECK(drive_logit_gain.dim() == 1, "drive_logit_gain must be [capacity]");
    TORCH_CHECK(slot_tokens.dim() == 1, "slot_tokens must be [capacity]");
    TORCH_CHECK(action_cache_actions.dim() == 2, "action_cache_actions must be [capacity, cache_capacity]");
    TORCH_CHECK(action_cache_token.dim() == 1, "action_cache_token must be [capacity]");
    TORCH_CHECK(action_cache_cursor.dim() == 1, "action_cache_cursor must be [capacity]");
    TORCH_CHECK(action_cache_next.dim() == 1, "action_cache_next must be [capacity]");
    const bool exact_pcg64 = pcg64_state_words.numel() != 0 || pcg64_buffer_words.numel() != 0;
    TORCH_CHECK(
        !exact_pcg64 || (pcg64_state_words.dim() == 1 && pcg64_state_words.numel() == 4),
        "exact PCG64 state must have four uint64 words"
    );
    TORCH_CHECK(
        !exact_pcg64 || (pcg64_buffer_words.dim() == 1 && pcg64_buffer_words.numel() == 2),
        "exact PCG64 buffer must have two int64 words"
    );
    TORCH_CHECK(hidden_size > 0, "hidden_size must be positive");
    TORCH_CHECK(action_count >= 7, "fixed MLP action_count must include gather, rest, and attack actions");
    TORCH_CHECK(cache_width > 0, "cache_width must be positive");
    TORCH_CHECK(cache_width <= action_cache_actions.size(1), "cache_width exceeds action cache capacity");
    TORCH_CHECK(rng_counter >= 0, "rng_counter must be non-negative");

    const int64_t row_count = observations.size(0);
    const int64_t observation_size = observations.size(1);
    const int64_t capacity = mlp_pack.size(0);
    const int64_t packed_width = hidden_size * observation_size
        + hidden_size
        + hidden_size * hidden_size
        + hidden_size
        + action_count * hidden_size
        + action_count;
    TORCH_CHECK(mlp_pack.size(1) == packed_width, "mlp_pack width does not match the fixed MLP layout");
    TORCH_CHECK(drives.size(0) == capacity, "drives capacity mismatch");
    TORCH_CHECK(attack_bias.size(0) == capacity, "attack_bias capacity mismatch");
    TORCH_CHECK(action_temperature.size(0) == capacity, "action_temperature capacity mismatch");
    TORCH_CHECK(drive_logit_gain.size(0) == capacity, "drive_logit_gain capacity mismatch");
    TORCH_CHECK(slot_tokens.size(0) == capacity, "slot_tokens capacity mismatch");
    TORCH_CHECK(action_cache_actions.size(0) == capacity, "action_cache_actions capacity mismatch");
    TORCH_CHECK(action_cache_token.size(0) == capacity, "action_cache_token capacity mismatch");
    TORCH_CHECK(action_cache_cursor.size(0) == capacity, "action_cache_cursor capacity mismatch");
    TORCH_CHECK(action_cache_next.size(0) == capacity, "action_cache_next capacity mismatch");

    torch::Tensor active_mask;
    auto safe_slots = slots;
    if (counted_rows) {
        auto row_indices = torch::arange(row_count, slots.options().dtype(torch::kInt64));
        active_mask = row_indices < torch::clamp(active_count, 0, row_count);
        safe_slots = torch::where(active_mask, slots, torch::zeros_like(slots)).contiguous();
    }
    auto row_slots = safe_slots.to(torch::kLong);
    auto pack = mlp_pack.index_select(0, row_slots).contiguous();
    auto x = observations.contiguous();
    auto aggr = attack_bias.index_select(0, row_slots).contiguous();
    auto temp = action_temperature.index_select(0, row_slots).contiguous();
    auto selected_drives = drives.index_select(0, row_slots).contiguous();
    auto gain = torch::clamp(drive_logit_gain.index_select(0, row_slots).contiguous(), 0.0);

    int64_t offset = 0;
    auto w1 = pack.slice(1, offset, offset + hidden_size * observation_size)
        .view({-1, hidden_size, observation_size});
    offset += hidden_size * observation_size;
    auto b1 = pack.slice(1, offset, offset + hidden_size).view({-1, hidden_size});
    offset += hidden_size;
    auto w2 = pack.slice(1, offset, offset + hidden_size * hidden_size)
        .view({-1, hidden_size, hidden_size});
    offset += hidden_size * hidden_size;
    auto b2 = pack.slice(1, offset, offset + hidden_size).view({-1, hidden_size});
    offset += hidden_size;
    auto w3 = pack.slice(1, offset, offset + action_count * hidden_size)
        .view({-1, action_count, hidden_size});
    offset += action_count * hidden_size;
    auto b3 = pack.slice(1, offset, offset + action_count).view({-1, action_count});

    auto h1 = torch::tanh(torch::bmm(w1, x.unsqueeze(-1)).squeeze(-1) + b1);
    auto h2 = torch::tanh(torch::bmm(w2, h1.unsqueeze(-1)).squeeze(-1) + b2);
    auto logits_policy = torch::bmm(w3, h2.unsqueeze(-1)).squeeze(-1) + b3;
    namespace idx = torch::indexing;
    logits_policy.index_put_(
        {idx::Slice(), ACT_ATTACK},
        logits_policy.index({idx::Slice(), ACT_ATTACK}) + aggr
    );
    logits_policy = logits_policy / torch::clamp(temp, 0.05).unsqueeze(-1);
    auto logits = logits_policy.clone();
    logits.index_put_(
        {idx::Slice(), ACT_GATHER},
        logits.index({idx::Slice(), ACT_GATHER})
            + gain * (w_hunger_gather * selected_drives.index({idx::Slice(), 0}))
    );
    logits.index_put_(
        {idx::Slice(), ACT_REST},
        logits.index({idx::Slice(), ACT_REST})
            + gain * (
                w_pain_rest * selected_drives.index({idx::Slice(), 1})
                + w_fatigue_rest * selected_drives.index({idx::Slice(), 2})
            )
    );
    logits.index_put_(
        {idx::Slice(), ACT_ATTACK},
        logits.index({idx::Slice(), ACT_ATTACK})
            - gain * (
                w_pain_attack * selected_drives.index({idx::Slice(), 1})
                + w_fear_attack * selected_drives.index({idx::Slice(), 3})
            )
    );
    if (residuals_enabled) {
        logits.add_(logit_residuals);
    }
    if (counted_rows) {
        logits = torch::where(
            active_mask.unsqueeze(-1),
            logits,
            torch::zeros_like(logits)
        );
    }

    auto row_actions = counted_rows
        ? torch::full({row_count}, -1, observations.options().dtype(torch::kInt32))
        : torch::empty({row_count}, observations.options().dtype(torch::kInt32));
    if (row_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        const int64_t* active_count_ptr = counted_rows
            ? active_count.data_ptr<int64_t>()
            : nullptr;
        if (!deterministic && exact_pcg64) {
            fill_fixed_mlp_cache_pcg64_kernel<<<1, 1, 0, stream>>>(
                logits.data_ptr<float>(),
                safe_slots.data_ptr<int32_t>(),
                slot_tokens.data_ptr<int64_t>(),
                action_cache_actions.data_ptr<int32_t>(),
                action_cache_token.data_ptr<int64_t>(),
                action_cache_cursor.data_ptr<int32_t>(),
                action_cache_next.data_ptr<int32_t>(),
                row_actions.data_ptr<int32_t>(),
                pcg64_state_words.data_ptr<uint64_t>(),
                pcg64_buffer_words.data_ptr<int64_t>(),
                row_count,
                active_count_ptr,
                capacity,
                action_count,
                action_cache_actions.size(1),
                cache_width
            );
        } else {
            constexpr int threads = 128;
            const int blocks = static_cast<int>((row_count + threads - 1) / threads);
            fill_fixed_mlp_cache_kernel<<<blocks, threads, 0, stream>>>(
                logits.data_ptr<float>(),
                safe_slots.data_ptr<int32_t>(),
                slot_tokens.data_ptr<int64_t>(),
                action_cache_actions.data_ptr<int32_t>(),
                action_cache_token.data_ptr<int64_t>(),
                action_cache_cursor.data_ptr<int32_t>(),
                action_cache_next.data_ptr<int32_t>(),
                row_actions.data_ptr<int32_t>(),
                row_count,
                active_count_ptr,
                capacity,
                action_count,
                action_cache_actions.size(1),
                cache_width,
                deterministic,
                static_cast<uint64_t>(rng_seed),
                static_cast<uint64_t>(rng_counter)
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    torch::Tensor next_rng_counter;
    if (counted_rows && !deterministic && !exact_pcg64) {
        next_rng_counter = torch::clamp(active_count, 0, row_count)
            * (cache_width * action_count)
            + rng_counter;
    } else {
        const int64_t draws_consumed = deterministic
            ? 0
            : row_count * cache_width * action_count;
        next_rng_counter = torch::full(
            {1},
            rng_counter + (exact_pcg64 ? 0 : draws_consumed),
            observations.options().dtype(torch::kInt64)
        );
    }
    return {
        logits,
        row_actions,
        action_cache_actions,
        action_cache_token,
        action_cache_cursor,
        action_cache_next,
        next_rng_counter,
        pcg64_state_words,
        pcg64_buffer_words,
    };
}

}  // namespace
