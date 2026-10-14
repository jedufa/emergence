#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace eiel {
namespace cuda_architecture_b {

constexpr int64_t kModularTickPlaneCount = 57;
constexpr int64_t kModularTickAuditCount = 27;

// Validate the complete fixed-shape resident-plane envelope without mutating
// any plane. Content checks cover every invariant consumed by modular_tick_v1.
void validate_modular_planes_v1(
    const std::vector<torch::Tensor>& planes
);

// Reorder compact active rows into slot order with exact copies. The work and
// storage are O(capacity * row_width), never O(capacity * capacity).
torch::Tensor scatter_active_rows_by_slot_v1(
    torch::Tensor rows,
    torch::Tensor slots,
    torch::Tensor active_mask,
    int64_t capacity
);

// Clear lifecycle-dead rows to the committed Phase E ABI fill values. The
// death list is fixed-capacity and only its first dead_count entries are live.
void clear_modular_rows_v1(
    const std::vector<torch::Tensor>& planes,
    torch::Tensor dead_slots,
    torch::Tensor dead_count
);

// The planes follow the committed Phase E ABI order.  The call mutates the
// eight runtime planes in place and returns action residuals followed by the
// fixed audit tensor order exposed by the Python wrapper.
std::vector<torch::Tensor> modular_tick_v1(
    const std::vector<torch::Tensor>& planes,
    torch::Tensor observations,
    torch::Tensor fatigue,
    torch::Tensor stage_indices,
    int64_t action_width
);

}  // namespace cuda_architecture_b
}  // namespace eiel
