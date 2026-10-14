#include "cxxcore5_tick_island_kernel.cpp"

#include <cstdint>
#include <vector>

extern "C" int cxxcore7_rng_precompute_run(
    const std::int32_t* agent_ids,
    const float* initial_energy,
    const float* initial_health,
    const float* initial_injury,
    const float* initial_scar,
    const std::int32_t* initial_pos_y_by_id,
    const std::int32_t* initial_pos_x_by_id,
    const std::int64_t pos_len,
    const std::int32_t* initial_agent_grid,
    const std::int8_t* initial_grid,
    const float* move_cost_mults,
    const float* juvenile_harvest_mults,
    const std::int32_t* actions_by_tick,
    const std::int32_t validation_action_override_enable,
    const std::int32_t* validation_action_override_ints,
    const std::int64_t validation_action_override_len,
    const double* cost_layout_rows,
    const std::int64_t cost_layout_cols,
    const std::int32_t charge_channels,
    const std::int32_t charge_radius,
    const std::int32_t ledger_enabled,
    const std::int64_t ticks,
    const std::int64_t n,
    const std::int64_t height,
    const std::int64_t width,
    const float move_cost,
    const float gather_cost,
    const float rest_cost,
    const float attack_cost,
    const float action_cost_mult,
    const float rest_cost_mult,
    const float food_energy,
    const float max_energy,
    const std::int32_t export_final,
    float* out_energy,
    float* out_health,
    float* out_injury,
    float* out_scar,
    std::uint8_t* out_died,
    std::int32_t* out_pos_y,
    std::int32_t* out_pos_x,
    std::int8_t* out_grid,
    float* out_cost_total_by_agent,
    double* out_layout_cost_scalar_by_agent,
    double* out_summary
) {
    if (ticks < 0 || n < 0) {
        return 1;
    }
    const std::int64_t expected_action_len = ticks * n;
    const std::int32_t* actions = actions_by_tick;
    std::vector<std::int32_t> overridden_actions;
    if (validation_action_override_enable != 0) {
        if (validation_action_override_ints == nullptr || validation_action_override_len != expected_action_len) {
            return 7;
        }
        overridden_actions.assign(validation_action_override_ints, validation_action_override_ints + validation_action_override_len);
        for (std::int64_t i = 0; i < expected_action_len; ++i) {
            const std::int32_t action = overridden_actions[static_cast<std::size_t>(i)];
            if (action < 0 || action > 3) {
                return 8;
            }
        }
        actions = overridden_actions.data();
    }
    return cxxcore5_tick_island_run(
        agent_ids,
        initial_energy,
        initial_health,
        initial_injury,
        initial_scar,
        initial_pos_y_by_id,
        initial_pos_x_by_id,
        pos_len,
        initial_agent_grid,
        initial_grid,
        move_cost_mults,
        juvenile_harvest_mults,
        actions,
        cost_layout_rows,
        cost_layout_cols,
        charge_channels,
        charge_radius,
        ledger_enabled,
        ticks,
        n,
        height,
        width,
        move_cost,
        gather_cost,
        rest_cost,
        attack_cost,
        action_cost_mult,
        rest_cost_mult,
        food_energy,
        max_energy,
        export_final,
        out_energy,
        out_health,
        out_injury,
        out_scar,
        out_died,
        out_pos_y,
        out_pos_x,
        out_grid,
        out_cost_total_by_agent,
        out_layout_cost_scalar_by_agent,
        out_summary
    );
}
