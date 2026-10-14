#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace eiel::cuda_architecture_b::patch_p2_cell_params_v1 {

inline constexpr char kContractVersionV1[] =
    "eiel_cuda_architecture_b_patch_p2_cell_params_v1";
inline constexpr char kDigestSchemaV1[] =
    "sha256_canonical_little_endian_patch_p2_cell_params_v1";
inline constexpr std::uint32_t kAbiVersionV1 = 1U;
inline constexpr std::size_t kByteWidthV1 = 160U;
inline constexpr std::size_t kAlignmentV1 = 16U;

enum PatchP2CellParamFlagsV1 : std::uint32_t {
    kFoodRegrowUseFertilityV1 = 1U << 0U,
    kFoodRegrowCapByFertilityV1 = 1U << 1U,
    kDensityIncludeSelfV1 = 1U << 2U,
    kDensityNormalizeV1 = 1U << 3U,
    kWasteConsequenceEnableV1 = 1U << 4U,
    kWastePenaltyEnableV1 = 1U << 5U,
    kDensityPenaltyEnableV1 = 1U << 6U,
    kMovementCostApplyEnableV1 = 1U << 7U,
};

inline constexpr std::uint32_t kKnownFlagsV1 =
    kFoodRegrowUseFertilityV1
    | kFoodRegrowCapByFertilityV1
    | kDensityIncludeSelfV1
    | kDensityNormalizeV1
    | kWasteConsequenceEnableV1
    | kWastePenaltyEnableV1
    | kDensityPenaltyEnableV1
    | kMovementCostApplyEnableV1;
inline constexpr std::uint32_t kRequiredP2ProfileFlagsV1 =
    kWasteConsequenceEnableV1 | kMovementCostApplyEnableV1;

// This value object is copied once per cell. Floating-point members contain
// host-normalized values, not the raw user configuration values.
struct alignas(16) PatchP2CellParamsV1 {
    std::uint32_t abi_version;
    std::uint32_t flags;
    std::int64_t patch_density_radius;

    float patch_scent_decay_factor;
    float patch_waste_decay_factor;
    float patch_waste_density_emit_scale;
    float patch_scent_max;
    float patch_waste_max;
    float patch_food_amount_max;
    float patch_food_regrow_per_step;
    float patch_food_presence_threshold;
    float patch_waste_regrowth_penalty_scale;
    float patch_waste_regrowth_penalty_max;
    float patch_density_regrowth_penalty_scale;
    float patch_density_regrowth_penalty_max;
    float patch_regrowth_min_multiplier;
    std::uint32_t reserved_zero;

    double patch_scent_presence_emit;
    double patch_waste_presence_emit;
    double patch_food_amount_initial_food_cell_value;
    double patch_food_gather_amount;
    double patch_scent_move_emit;
    double patch_waste_move_emit;
    double patch_scent_gather_emit;
    double patch_waste_consume_emit;
    double patch_movement_cost_energy_scale;
    double patch_movement_cost_min;
    double patch_movement_cost_max;
};

inline bool valid_patch_p2_cell_params_v1(
    const PatchP2CellParamsV1& params
) noexcept {
    if (
        params.abi_version != kAbiVersionV1
        || (params.flags & ~kKnownFlagsV1) != 0U
        || (params.flags & kRequiredP2ProfileFlagsV1)
            != kRequiredP2ProfileFlagsV1
        || params.patch_density_radius < 0
        || params.reserved_zero != 0U
    ) {
        return false;
    }

    const double float64_values[] = {
        params.patch_scent_presence_emit,
        params.patch_waste_presence_emit,
        params.patch_food_amount_initial_food_cell_value,
        params.patch_food_gather_amount,
        params.patch_scent_move_emit,
        params.patch_waste_move_emit,
        params.patch_scent_gather_emit,
        params.patch_waste_consume_emit,
        params.patch_movement_cost_energy_scale,
        params.patch_movement_cost_min,
        params.patch_movement_cost_max,
    };
    for (const double value : float64_values) {
        if (!std::isfinite(value) || (value == 0.0 && std::signbit(value))) {
            return false;
        }
    }

    const float float32_values[] = {
        params.patch_scent_decay_factor,
        params.patch_waste_decay_factor,
        params.patch_waste_density_emit_scale,
        params.patch_scent_max,
        params.patch_waste_max,
        params.patch_food_amount_max,
        params.patch_food_regrow_per_step,
        params.patch_food_presence_threshold,
        params.patch_waste_regrowth_penalty_scale,
        params.patch_waste_regrowth_penalty_max,
        params.patch_density_regrowth_penalty_scale,
        params.patch_density_regrowth_penalty_max,
        params.patch_regrowth_min_multiplier,
    };
    for (const float value : float32_values) {
        if (!std::isfinite(value) || (value == 0.0F && std::signbit(value))) {
            return false;
        }
    }

    return params.patch_scent_presence_emit >= 0.0
        && params.patch_waste_presence_emit >= 0.0
        && params.patch_food_amount_initial_food_cell_value
            > static_cast<double>(params.patch_food_presence_threshold)
        && params.patch_food_amount_initial_food_cell_value
            <= static_cast<double>(params.patch_food_amount_max)
        && params.patch_food_gather_amount > 0.0
        && params.patch_scent_move_emit >= 0.0
        && params.patch_waste_move_emit >= 0.0
        && params.patch_scent_gather_emit >= 0.0
        && params.patch_waste_consume_emit >= 0.0
        && params.patch_movement_cost_energy_scale >= 0.0
        && params.patch_movement_cost_min >= 0.0
        && params.patch_movement_cost_max >= 0.0
        && params.patch_movement_cost_max >= params.patch_movement_cost_min
        && params.patch_scent_decay_factor >= 0.0F
        && params.patch_scent_decay_factor <= 1.0F
        && params.patch_waste_decay_factor >= 0.0F
        && params.patch_waste_decay_factor <= 1.0F
        && params.patch_waste_density_emit_scale >= 0.0F
        && params.patch_scent_max >= 0.0F
        && params.patch_waste_max >= 0.0F
        && params.patch_food_amount_max >= 0.0F
        && params.patch_food_regrow_per_step >= 0.0F
        && params.patch_food_presence_threshold >= 0.0F
        && params.patch_food_presence_threshold <= params.patch_food_amount_max
        && params.patch_waste_regrowth_penalty_scale >= 0.0F
        && params.patch_waste_regrowth_penalty_max >= 0.0F
        && params.patch_waste_regrowth_penalty_max <= 1.0F
        && params.patch_density_regrowth_penalty_scale >= 0.0F
        && params.patch_density_regrowth_penalty_max >= 0.0F
        && params.patch_density_regrowth_penalty_max <= 1.0F
        && params.patch_regrowth_min_multiplier >= 0.0F
        && params.patch_regrowth_min_multiplier <= 1.0F;
}

static_assert(std::is_standard_layout_v<PatchP2CellParamsV1>);
static_assert(std::is_trivially_copyable_v<PatchP2CellParamsV1>);
static_assert(alignof(PatchP2CellParamsV1) == kAlignmentV1);
static_assert(sizeof(PatchP2CellParamsV1) == kByteWidthV1);
static_assert(offsetof(PatchP2CellParamsV1, abi_version) == 0);
static_assert(offsetof(PatchP2CellParamsV1, flags) == 4);
static_assert(offsetof(PatchP2CellParamsV1, patch_density_radius) == 8);
static_assert(offsetof(PatchP2CellParamsV1, patch_scent_decay_factor) == 16);
static_assert(offsetof(PatchP2CellParamsV1, patch_waste_decay_factor) == 20);
static_assert(
    offsetof(PatchP2CellParamsV1, patch_waste_density_emit_scale) == 24
);
static_assert(offsetof(PatchP2CellParamsV1, patch_scent_max) == 28);
static_assert(offsetof(PatchP2CellParamsV1, patch_waste_max) == 32);
static_assert(offsetof(PatchP2CellParamsV1, patch_food_amount_max) == 36);
static_assert(offsetof(PatchP2CellParamsV1, patch_food_regrow_per_step) == 40);
static_assert(
    offsetof(PatchP2CellParamsV1, patch_food_presence_threshold) == 44
);
static_assert(
    offsetof(PatchP2CellParamsV1, patch_waste_regrowth_penalty_scale) == 48
);
static_assert(
    offsetof(PatchP2CellParamsV1, patch_waste_regrowth_penalty_max) == 52
);
static_assert(
    offsetof(PatchP2CellParamsV1, patch_density_regrowth_penalty_scale) == 56
);
static_assert(
    offsetof(PatchP2CellParamsV1, patch_density_regrowth_penalty_max) == 60
);
static_assert(
    offsetof(PatchP2CellParamsV1, patch_regrowth_min_multiplier) == 64
);
static_assert(offsetof(PatchP2CellParamsV1, reserved_zero) == 68);
static_assert(offsetof(PatchP2CellParamsV1, patch_scent_presence_emit) == 72);
static_assert(offsetof(PatchP2CellParamsV1, patch_waste_presence_emit) == 80);
static_assert(
    offsetof(PatchP2CellParamsV1, patch_food_amount_initial_food_cell_value)
    == 88
);
static_assert(offsetof(PatchP2CellParamsV1, patch_food_gather_amount) == 96);
static_assert(offsetof(PatchP2CellParamsV1, patch_scent_move_emit) == 104);
static_assert(offsetof(PatchP2CellParamsV1, patch_waste_move_emit) == 112);
static_assert(offsetof(PatchP2CellParamsV1, patch_scent_gather_emit) == 120);
static_assert(offsetof(PatchP2CellParamsV1, patch_waste_consume_emit) == 128);
static_assert(
    offsetof(PatchP2CellParamsV1, patch_movement_cost_energy_scale) == 136
);
static_assert(offsetof(PatchP2CellParamsV1, patch_movement_cost_min) == 144);
static_assert(offsetof(PatchP2CellParamsV1, patch_movement_cost_max) == 152);

}  // namespace eiel::cuda_architecture_b::patch_p2_cell_params_v1
