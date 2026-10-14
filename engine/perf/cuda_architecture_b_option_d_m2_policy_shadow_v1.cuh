#pragma once

#include <torch/extension.h>

#include <cstdint>

namespace eiel::cuda_architecture_b::option_d_m2_policy_shadow_v1 {

struct PolicySegmentShadowResultV1 {
    std::int64_t checked_rows = 0;
    std::int64_t checked_bytes = 0;
    std::int64_t mismatch_count = 0;
    std::int64_t invalid_state_count = 0;
    std::int64_t invalid_semantic_count = 0;
};

PolicySegmentShadowResultV1 verify_policy_segment_shadow_v1(
    const torch::Tensor& source_state,
    const torch::Tensor& expected_crossover_mask_packed,
    const torch::Tensor& semantic_tape,
    std::int64_t policy_row_width
);

}  // namespace eiel::cuda_architecture_b::option_d_m2_policy_shadow_v1
