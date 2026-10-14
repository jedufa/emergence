#ifndef EIEL_CUDA_ARCHITECTURE_B_FLATTENED_JUNGLE_CONTROLLER_V4_HPP_
#define EIEL_CUDA_ARCHITECTURE_B_FLATTENED_JUNGLE_CONTROLLER_V4_HPP_

#include <cstdint>

namespace eiel::cuda_architecture_b::flattened_jungle_controller_v4 {

inline constexpr std::int32_t kAbiVersionV4 = 4;
inline constexpr std::int64_t kTickIntegerMetricCountV4 = 9;
inline constexpr std::int64_t kTickDoubleMetricCountV4 = 7;

enum JungleTickIntegerMetricV4 : std::int64_t {
    kPredatorExposureCountV4 = 0,
    kPredatorEventCountV4,
    kPredatorDeathCountV4,
    kJuvenilePredatorEventCountV4,
    kJuvenilePredatorDeathCountV4,
    kAdultPredatorEventCountV4,
    kAdultPredatorDeathCountV4,
    kBirthEmissionCountV4,
    kEnvironmentRngDrawCountV4,
};

enum JungleTickDoubleMetricV4 : std::int64_t {
    kPredatorRawDamageTotalV4 = 0,
    kPredatorEffectiveDamageTotalV4,
    kPredatorPreventedDamageTotalV4,
    kJuvenilePredatorDamageTotalV4,
    kAdultPredatorDamageTotalV4,
    kBirthScentEmittedTotalV4,
    kBirthWasteEmittedTotalV4,
};

struct FlatJungleConfigV4 {
    bool enabled;
    bool predator_pressure_enabled;
    bool juvenile_vulnerability_enabled;
    bool terrain_protection_enabled;
    bool terrain_protection_apply_to_damage_enabled;
    bool birth_emissions_enabled;
    bool birth_emission_clamp_enabled;
    bool ecological_recovery_enabled;
    bool visibility_from_cover_enabled;
    bool recovery_regrowth_boost_enabled;
    std::int32_t juvenile_age_threshold;
    float scent_max;
    float waste_max;
    float density_denominator;
    float visibility_min;
    float visibility_max;
    float visibility_default;
    float predator_base_risk;
    float predator_scent_weight;
    float predator_waste_weight;
    float predator_density_weight;
    float predator_visibility_weight;
    float predator_cover_protection_weight;
    float predator_shelter_protection_weight;
    float predator_risk_min;
    float predator_risk_cap;
    double predator_health_damage;
    double juvenile_damage_multiplier;
    float cover_damage_reduction_weight;
    float shelter_damage_reduction_weight;
    float terrain_damage_multiplier_min;
    double birth_scent_emit;
    double birth_waste_emit;
    float recovery_activity_min;
    float recovery_activity_max;
    float fertility_min;
    float fertility_max;
    float recovery_decay_factor;
    float recovery_base_growth;
    float recovery_cover_weight;
    float recovery_fertility_weight;
    float recovery_waste_penalty_weight;
    float recovery_density_penalty_weight;
    float recovery_scent_penalty_weight;
    float recovery_regrowth_boost_scale;
    float recovery_regrowth_boost_max;
};

// State remains device-resident for the whole trajectory. Scratch and reports
// cover only the reusable physical window, so memory does not grow with the
// logical generation count.
struct FlatJungleContextV4 {
    bool enabled;
    FlatJungleConfigV4 config;
    float* patch_recovery_activity;
    float* patch_visibility;
    float* recovery_scratch;
    double* recovery_stats;
    std::int64_t* tick_integer_metrics;
    double* tick_double_metrics;
    float* predator_damage_multiplier_tape;
    std::int64_t ticks_per_generation;
    std::int64_t generation_rows_per_cell;
    std::int64_t generation_base;
    std::int64_t capacity;
    const double* predator_base_risk_by_cell;
    const double* predator_health_damage_by_cell;
    const double* predator_risk_cap_by_cell;
    const double* recovery_base_growth_by_cell;
    const double* recovery_decay_factor_by_cell;
    const double* recovery_fertility_weight_by_cell;
    const double* recovery_waste_penalty_weight_by_cell;
};

void install_flat_jungle_context_v4(const FlatJungleContextV4& context);
void clear_flat_jungle_context_v4() noexcept;
const FlatJungleContextV4& current_flat_jungle_context_v4() noexcept;

}  // namespace eiel::cuda_architecture_b::flattened_jungle_controller_v4

#endif
