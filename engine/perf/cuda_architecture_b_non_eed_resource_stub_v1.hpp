#ifndef EIEL_CUDA_ARCHITECTURE_B_NON_EED_RESOURCE_STUB_V1_HPP_
#define EIEL_CUDA_ARCHITECTURE_B_NON_EED_RESOURCE_STUB_V1_HPP_

// The shared flattened kernel retains a context slot for its optional EED
// overlay.  The dedicated WDL/Jungle artifact must neither link that overlay
// nor expose its state.  This ABI-compatible disabled context lets the shared
// kernel compile while every EED branch is permanently false for this module.

#include <cstdint>

namespace eiel::eed_resource_impact_native_v1 {

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

inline const ResidentContextV1& current_resident_context_v1() noexcept {
    static const ResidentContextV1 context{
        false,
        {1, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
         {0.0F, 0.0F, 0.0F, false, false, false, 0}, false, false},
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, 0, 0, 0, 0,
    };
    return context;
}

}  // namespace eiel::eed_resource_impact_native_v1

#endif
