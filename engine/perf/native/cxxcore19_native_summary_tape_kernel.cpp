#include <cstdint>

extern "C" int cxxcore19_native_summary_tape_run(
    const std::int64_t n,
    const std::int32_t generation,
    const std::int32_t tick,
    const float cost_scalar_total,
    const std::uint8_t validation_random_walk_override,
    const std::int32_t* agent_ids,
    const std::int32_t* actions,
    const std::int32_t* pre_y,
    const std::int32_t* pre_x,
    const std::int32_t* post_y,
    const std::int32_t* post_x,
    const float* pre_energy,
    const float* post_energy,
    const float* pre_health,
    const float* post_health,
    const float* pre_injury,
    const float* post_injury,
    const float* pre_scar,
    const float* post_scar,
    const std::uint8_t* pre_alive,
    const std::uint8_t* post_alive,
    const std::int32_t* phase_codes,
    const std::int32_t* post_phase_codes,
    const float* effective_memory,
    const float* memory_abs,
    std::int32_t* out_generation_tick,
    float* out_cost_flags,
    std::int32_t* out_agent_ids,
    std::int32_t* out_actions,
    std::int32_t* out_pre_y,
    std::int32_t* out_pre_x,
    std::int32_t* out_post_y,
    std::int32_t* out_post_x,
    float* out_pre_energy,
    float* out_post_energy,
    float* out_pre_health,
    float* out_post_health,
    float* out_pre_injury,
    float* out_post_injury,
    float* out_pre_scar,
    float* out_post_scar,
    std::uint8_t* out_pre_alive,
    std::uint8_t* out_post_alive,
    std::int32_t* out_phase_codes,
    std::int32_t* out_post_phase_codes,
    float* out_effective_memory,
    float* out_memory_abs,
    std::int32_t* out_phase_action_counts,
    float* out_phase_damage_total,
    std::int32_t* out_summary_counts
) {
    if (n < 0) {
        return 1;
    }
    if (!agent_ids || !actions || !pre_y || !pre_x || !post_y || !post_x ||
        !pre_energy || !post_energy || !pre_health || !post_health ||
        !pre_injury || !post_injury || !pre_scar || !post_scar ||
        !pre_alive || !post_alive || !phase_codes || !post_phase_codes ||
        !effective_memory || !memory_abs || !out_generation_tick || !out_cost_flags ||
        !out_agent_ids || !out_actions || !out_pre_y || !out_pre_x ||
        !out_post_y || !out_post_x || !out_pre_energy || !out_post_energy ||
        !out_pre_health || !out_post_health || !out_pre_injury || !out_post_injury ||
        !out_pre_scar || !out_post_scar || !out_pre_alive || !out_post_alive ||
        !out_phase_codes || !out_post_phase_codes || !out_effective_memory ||
        !out_memory_abs || !out_phase_action_counts || !out_phase_damage_total ||
        !out_summary_counts) {
        return 2;
    }

    out_generation_tick[0] = generation;
    out_generation_tick[1] = tick;
    out_cost_flags[0] = cost_scalar_total;
    out_cost_flags[1] = validation_random_walk_override ? 1.0f : 0.0f;

    for (int i = 0; i < 32; ++i) {
        out_phase_action_counts[i] = 0;
    }
    for (int i = 0; i < 4; ++i) {
        out_phase_damage_total[i] = 0.0f;
    }
    for (int i = 0; i < 8; ++i) {
        out_summary_counts[i] = 0;
    }

    for (std::int64_t i = 0; i < n; ++i) {
        out_agent_ids[i] = agent_ids[i];
        out_actions[i] = actions[i];
        out_pre_y[i] = pre_y[i];
        out_pre_x[i] = pre_x[i];
        out_post_y[i] = post_y[i];
        out_post_x[i] = post_x[i];
        out_pre_energy[i] = pre_energy[i];
        out_post_energy[i] = post_energy[i];
        out_pre_health[i] = pre_health[i];
        out_post_health[i] = post_health[i];
        out_pre_injury[i] = pre_injury[i];
        out_post_injury[i] = post_injury[i];
        out_pre_scar[i] = pre_scar[i];
        out_post_scar[i] = post_scar[i];
        out_pre_alive[i] = pre_alive[i] ? 1 : 0;
        out_post_alive[i] = post_alive[i] ? 1 : 0;
        out_phase_codes[i] = phase_codes[i];
        out_post_phase_codes[i] = post_phase_codes[i];
        out_effective_memory[i] = effective_memory[i];
        out_memory_abs[i] = memory_abs[i];

        const int phase = phase_codes[i];
        const int action = actions[i];
        if (phase >= 0 && phase < 4 && action >= 0 && action < 8) {
            out_phase_action_counts[phase * 8 + action] += 1;
        }
        if (pre_alive[i] && !post_alive[i]) {
            out_summary_counts[0] += 1;
        }
        if (!pre_alive[i] && post_alive[i]) {
            out_summary_counts[1] += 1;
        }
        if (post_y[i] != pre_y[i] || post_x[i] != pre_x[i]) {
            out_summary_counts[2] += 1;
        }
        const float damage = pre_health[i] - post_health[i];
        if (damage > 0.0f) {
            out_summary_counts[3] += 1;
            if (phase >= 0 && phase < 4) {
                out_phase_damage_total[phase] += damage;
            }
        }
        if (phase != 3 && post_phase_codes[i] == 3) {
            out_summary_counts[4] += 1;
        }
        if (phase == 3 && post_phase_codes[i] != 3) {
            out_summary_counts[5] += 1;
        }
        if ((phase == 1 || phase == 2) && post_phase_codes[i] == 0) {
            out_summary_counts[6] += 1;
        }
    }
    out_summary_counts[7] = static_cast<std::int32_t>(n);
    return 0;
}
