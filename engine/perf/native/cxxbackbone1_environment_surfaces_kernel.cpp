#include <algorithm>
#include <cstdint>

extern "C" int cxxbackbone1_environment_surfaces_v1(
    const float* patch_food,
    const float* patch_fertility,
    const float* patch_cover,
    const float* patch_shelter,
    const float* jungle_predator_risk,
    const float* jungle_terrain_protection,
    const float* wdl_cue,
    const float* wdl_active,
    const std::int32_t* wdl_timer,
    const float* occupant_density,
    const std::int64_t cells,
    const float patch_regrow_rate,
    const float patch_food_cap,
    const float jungle_risk_decay,
    const float wdl_cue_to_active_threshold,
    const std::int32_t wdl_delay_ticks,
    float* out_patch_food,
    float* out_patch_visibility,
    float* out_jungle_predator_risk,
    float* out_jungle_damage_pressure,
    float* out_wdl_cue,
    float* out_wdl_active,
    std::int32_t* out_wdl_timer,
    double* out_summary
) {
    if (
        !patch_food || !patch_fertility || !patch_cover || !patch_shelter || !jungle_predator_risk
        || !jungle_terrain_protection || !wdl_cue || !wdl_active || !wdl_timer || !occupant_density
        || !out_patch_food || !out_patch_visibility || !out_jungle_predator_risk || !out_jungle_damage_pressure
        || !out_wdl_cue || !out_wdl_active || !out_wdl_timer || !out_summary
    ) {
        return 1;
    }
    if (cells < 0) {
        return 2;
    }
    double patch_food_sum = 0.0;
    double jungle_pressure_sum = 0.0;
    double wdl_active_sum = 0.0;
    double visibility_sum = 0.0;
    for (std::int64_t i = 0; i < cells; ++i) {
        const float regrown = patch_food[i] + patch_regrow_rate * patch_fertility[i] * (1.0f - occupant_density[i]);
        const float food = std::max(0.0f, std::min(patch_food_cap, regrown));
        const float visibility = std::max(0.0f, std::min(1.0f, 1.0f - 0.5f * patch_cover[i] - 0.25f * patch_shelter[i]));
        const float risk = std::max(0.0f, jungle_predator_risk[i] * jungle_risk_decay + occupant_density[i] * 0.05f);
        const float pressure = std::max(0.0f, risk * (1.0f - jungle_terrain_protection[i]));
        float cue = wdl_cue[i];
        float active = wdl_active[i];
        std::int32_t timer = wdl_timer[i];
        if (cue >= wdl_cue_to_active_threshold && active <= 0.0f && timer < 0) {
            timer = wdl_delay_ticks;
        } else if (timer > 0) {
            timer -= 1;
        } else if (timer == 0) {
            active = 1.0f;
            timer = -1;
        }
        if (active > 0.0f && cue < 0.01f) {
            active = std::max(0.0f, active - 0.1f);
        }
        cue = std::max(0.0f, cue * 0.95f);
        out_patch_food[i] = food;
        out_patch_visibility[i] = visibility;
        out_jungle_predator_risk[i] = risk;
        out_jungle_damage_pressure[i] = pressure;
        out_wdl_cue[i] = cue;
        out_wdl_active[i] = active;
        out_wdl_timer[i] = timer;
        patch_food_sum += static_cast<double>(food);
        jungle_pressure_sum += static_cast<double>(pressure);
        wdl_active_sum += static_cast<double>(active);
        visibility_sum += static_cast<double>(visibility);
    }
    out_summary[0] = patch_food_sum;
    out_summary[1] = visibility_sum;
    out_summary[2] = jungle_pressure_sum;
    out_summary[3] = wdl_active_sum;
    return 0;
}
