#pragma once

#include <torch/extension.h>

#include <cstdint>

namespace eiel::cuda_architecture_b::option_d_m4_policy_refill_v1 {

struct PolicyWindowRefillResultV1 {
    std::int64_t generated_rows = 0;
    std::int64_t generated_bytes = 0;
    std::int64_t invalid_state_count = 0;
    std::int64_t invalid_counter_count = 0;
};

PolicyWindowRefillResultV1 generate_compact_policy_window_v1(
    const torch::Tensor& source_state,
    torch::Tensor delta,
    torch::Tensor mutation_mask,
    torch::Tensor crossover_mask_packed,
    torch::Tensor post_state_packed,
    torch::Tensor semantic_tape,
    const torch::Tensor& absolute_cursor,
    const torch::Tensor& semantic_counter,
    torch::Tensor window_base,
    std::int64_t policy_row_width
);

}  // namespace eiel::cuda_architecture_b::option_d_m4_policy_refill_v1
