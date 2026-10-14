#ifndef EIEL_CUDA_ARCHITECTURE_B_FLATTENED_GRU_CONTROLLER_V3_HPP_
#define EIEL_CUDA_ARCHITECTURE_B_FLATTENED_GRU_CONTROLLER_V3_HPP_

#include <cstdint>

namespace eiel::cuda_architecture_b::flattened_gru_controller_v3 {

inline constexpr std::int32_t kAbiVersionV3 = 3;
inline constexpr std::int64_t kStableHiddenWidthV3 = 64;
inline constexpr std::int64_t kStableActionCountV3 = 7;
inline constexpr std::int64_t kStableScratchWidthV3 = 519;

// The v3 binding installs these device authorities for one native trajectory.
// The existing Patch/WDL ABI remains unchanged and sees a disabled context.
struct FlatGruContextV3 {
    bool enabled;
    bool immutable_shared_policy;
    float* gru_pack;
    std::int64_t gru_packed_width;
    float* scratch;
    std::int32_t* status;
    std::int64_t cells;
    std::int64_t capacity;
    double upkeep_cost_base;
    double upkeep_cost_per_dim;
};

void install_flat_gru_context_v3(const FlatGruContextV3& context);
void clear_flat_gru_context_v3() noexcept;
const FlatGruContextV3& current_flat_gru_context_v3() noexcept;

}  // namespace eiel::cuda_architecture_b::flattened_gru_controller_v3

#endif
