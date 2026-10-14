#ifndef EIEL_EED_RESOURCE_IMPACT_NATIVE_V1_HPP_
#define EIEL_EED_RESOURCE_IMPACT_NATIVE_V1_HPP_

#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace eiel::eed_resource_impact_native_v1 {

inline constexpr std::int32_t kSchemaVersion = 1;
inline constexpr std::int32_t kResidentSchemaVersion = 2;
inline constexpr std::int64_t kStateTensorCount = 10;
inline constexpr std::int32_t kResourceA = 1;
inline constexpr std::int32_t kResourceB = 2;

struct Config {
    float impact_per_success;
    float decay_per_step;
    float maximum_load;
    bool impact_applied;
    bool ecology_frozen;
    bool phenotype_fixed;
    std::int32_t fixed_axis_mode;
};

struct ResidentConfigV1 {
    std::int32_t schema_version;
    float food_presence_threshold;
    float gather_amount;
    float resource_a_handling_threshold;
    float resource_b_handling_threshold;
    float resource_a_return;
    float resource_b_return;
    Config impact;
    bool equalized_sham;
    bool enabled;
};

struct ResidentContextV1 {
    bool enabled;
    ResidentConfigV1 config;
    const std::int32_t* fixed_axis_mode_by_cell;
    const double* gather_amount_by_cell;
    const double* regrow_amount_by_cell;
    float* resource_a_amount;
    float* resource_b_amount;
    float* resource_a_capacity;
    float* resource_b_capacity;
    float* resource_a_renewal;
    float* resource_b_renewal;
    float* resource_a_pending_impact;
    float* resource_b_pending_impact;
    float* resource_a_impact_load;
    float* resource_b_impact_load;
    float* resource_class;
    std::int32_t* active_y;
    std::int32_t* active_x;
    std::int32_t* active_class;
    float* handling_progress;
    std::int64_t cells;
    std::int64_t grid_height;
    std::int64_t grid_width;
    std::int64_t capacity;
};

void install_resident_context_v1(const ResidentContextV1& context);
void clear_resident_context_v1() noexcept;
const ResidentContextV1& current_resident_context_v1() noexcept;
ResidentContextV1 validated_resident_context_v1(
    const pybind11::list& state_tensors,
    const pybind11::dict& config
);

void transition_cuda_v1(
    const std::vector<torch::Tensor>& state,
    const torch::Tensor& resource_classes,
    const torch::Tensor& flat_indices,
    const torch::Tensor& handling_axes,
    Config config
);

void bind_eed_resource_impact_native_v1(pybind11::module_& module);

}  // namespace eiel::eed_resource_impact_native_v1

#endif
