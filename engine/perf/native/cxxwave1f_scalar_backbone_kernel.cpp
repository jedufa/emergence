#include <algorithm>
#include <cmath>
#include <cstdint>

static inline float clamp01f(float x) {
    if (x <= 0.0f) {
        return 0.0f;
    }
    if (x >= 1.0f) {
        return 1.0f;
    }
    return x;
}

extern "C" int cxxwave1f_apply_scalar_backbone_v1(
    const std::int32_t* slots,
    const std::int32_t* actions,
    const float* chemistry_noise,
    const float* drive_noise_samples,
    const float* maintenance_opportunity,
    std::int64_t rows,
    std::int64_t capacity,
    std::uint8_t* alive,
    float* energy,
    float* health,
    float* injury,
    const float* last_energy_delta,
    const float* last_health_delta,
    const float* last_injury_delta,
    float* drives,
    float* prev_drive_delta,
    const float* drive_inertia,
    const float* drive_noise_scale,
    std::int32_t* survival_steps,
    float* drive_volatility,
    float* drive_jitter,
    float* episode0_drive_jitter_sum,
    std::int32_t* episode0_selfharm_events,
    float* chemistry,
    float* maintenance,
    std::int32_t* maintenance_requests,
    std::int32_t* maintenance_applied,
    float* maintenance_energy_spent_total,
    float* maintenance_opportunity_cost_total,
    float* maintenance_applied_effect_total,
    float max_energy,
    float start_health,
    float hunger_gain,
    float hunger_decay,
    float pain_gain,
    float pain_decay,
    float fatigue_gain,
    float fatigue_rest_relief,
    float fatigue_decay,
    float fear_gain,
    float fear_decay,
    std::int32_t episode0_k,
    std::int32_t maintenance_enable,
    std::int32_t maintenance_delay_ticks,
    float maintenance_base_effect,
    float maintenance_diminishing_factor,
    float maintenance_recent_decay,
    float maintenance_energy_cost,
    float maintenance_opportunity_cost_coeff,
    std::int32_t tick_index,
    double* out_summary
) {
    if (
        !slots || !actions || !chemistry_noise || !drive_noise_samples || !maintenance_opportunity ||
        !alive || !energy || !health || !injury || !last_energy_delta || !last_health_delta ||
        !last_injury_delta || !drives || !prev_drive_delta || !drive_inertia || !drive_noise_scale ||
        !survival_steps || !drive_volatility || !drive_jitter || !episode0_drive_jitter_sum ||
        !episode0_selfharm_events || !chemistry || !maintenance || !maintenance_requests ||
        !maintenance_applied || !maintenance_energy_spent_total || !maintenance_opportunity_cost_total ||
        !maintenance_applied_effect_total || !out_summary
    ) {
        return 1;
    }
    if (rows < 0 || capacity < 0) {
        return 2;
    }

    double drive_rows = 0.0;
    double chemistry_rows = 0.0;
    double maintenance_rows = 0.0;
    double schedule_requests = 0.0;
    double schedule_energy_cost_sum = 0.0;
    double schedule_opportunity_cost_sum = 0.0;
    double apply_count = 0.0;
    double applied_effect_sum = 0.0;
    double episode0_selfharm_count = 0.0;

    const float e_norm = std::max(max_energy, 1.0e-6f);
    const float h_norm = std::max(start_health, 1.0e-6f);
    const float base_effect = clamp01f(maintenance_base_effect);
    const float decay = clamp01f(maintenance_recent_decay);

    for (std::int64_t r = 0; r < rows; ++r) {
        const std::int32_t slot = slots[r];
        if (slot < 0 || static_cast<std::int64_t>(slot) >= capacity || !alive[slot]) {
            continue;
        }
        const std::int32_t action = actions[r];
        const std::int64_t d = static_cast<std::int64_t>(slot) * 4;
        const float e_v = clamp01f(energy[slot] / e_norm);
        const float h_v = clamp01f(health[slot] / h_norm);
        const float de = last_energy_delta[slot];
        const float dh = last_health_delta[slot];
        const float dinj = last_injury_delta[slot];
        const float inertia = drive_inertia[slot];
        const float inv = 1.0f - inertia;
        const float prev0 = drives[d + 0];
        const float prev1 = drives[d + 1];
        const float prev2 = drives[d + 2];
        const float prev3 = drives[d + 3];

        float target_hunger = prev0 + hunger_gain * (1.0f - e_v);
        target_hunger = target_hunger + 0.04f * (-de / e_norm) - hunger_decay;
        if (action == 4 && de > 0.0f) {
            target_hunger -= 0.20f * (de / e_norm);
        }
        target_hunger = clamp01f(target_hunger);

        float target_pain = prev1;
        if (dinj > 0.0f) {
            target_pain += pain_gain * std::min(1.0f, dinj);
        }
        if (dh < 0.0f) {
            target_pain += 0.10f * std::min(1.0f, -dh / h_norm);
        }
        target_pain = clamp01f(target_pain - pain_decay);

        float target_fatigue = prev2;
        if (action == 5) {
            target_fatigue -= fatigue_rest_relief;
        } else {
            target_fatigue += fatigue_gain;
        }
        target_fatigue = clamp01f(target_fatigue - fatigue_decay);

        float target_fear = prev3;
        if (dinj > 0.0f) {
            target_fear += fear_gain * std::min(1.0f, dinj);
        }
        target_fear = clamp01f(target_fear + 0.05f * (1.0f - h_v) - fear_decay);

        float next0 = inertia * prev0 + inv * target_hunger;
        float next1 = inertia * prev1 + inv * target_pain;
        float next2 = inertia * prev2 + inv * target_fatigue;
        float next3 = inertia * prev3 + inv * target_fear;
        const float noise_scale = drive_noise_scale[slot];
        if (noise_scale > 0.0f) {
            next0 += noise_scale * drive_noise_samples[r * 4 + 0];
            next1 += noise_scale * drive_noise_samples[r * 4 + 1];
            next2 += noise_scale * drive_noise_samples[r * 4 + 2];
            next3 += noise_scale * drive_noise_samples[r * 4 + 3];
        }
        next0 = clamp01f(next0);
        next1 = clamp01f(next1);
        next2 = clamp01f(next2);
        next3 = clamp01f(next3);

        const float delta0 = next0 - prev0;
        const float delta1 = next1 - prev1;
        const float delta2 = next2 - prev2;
        const float delta3 = next3 - prev3;
        const float jitter =
            std::fabs(delta0 - prev_drive_delta[d + 0]) +
            std::fabs(delta1 - prev_drive_delta[d + 1]) +
            std::fabs(delta2 - prev_drive_delta[d + 2]) +
            std::fabs(delta3 - prev_drive_delta[d + 3]);

        drives[d + 0] = next0;
        drives[d + 1] = next1;
        drives[d + 2] = next2;
        drives[d + 3] = next3;
        prev_drive_delta[d + 0] = delta0;
        prev_drive_delta[d + 1] = delta1;
        prev_drive_delta[d + 2] = delta2;
        prev_drive_delta[d + 3] = delta3;
        drive_volatility[slot] += std::fabs(delta0) + std::fabs(delta1) + std::fabs(delta2) + std::fabs(delta3);
        drive_jitter[slot] += jitter;
        if (survival_steps[slot] <= episode0_k) {
            episode0_drive_jitter_sum[slot] += jitter;
            if (action == 6 && dinj > 0.0f) {
                episode0_selfharm_events[slot] += 1;
                episode0_selfharm_count += 1.0;
            }
        }
        drive_rows += 1.0;

        const std::int64_t c = static_cast<std::int64_t>(slot) * 6;
        const float energy_n = clamp01f(energy[slot] / e_norm);
        const float health_n = clamp01f(health[slot] / h_norm);
        const float injury_n = clamp01f(injury[slot] / h_norm);
        const float stress = clamp01f(0.55f * injury_n + 0.30f * drives[d + 1] + 0.15f * (1.0f - energy_n));
        float energy_store = chemistry[c + 0];
        float repair = chemistry[c + 1];
        float acute = chemistry[c + 2];
        float chronic = chemistry[c + 3];
        float reg1 = chemistry[c + 4];
        float reg2 = chemistry[c + 5];
        const float acute_target = clamp01f(1.25f * injury_n + 0.45f * stress + 0.20f * (1.0f - health_n));
        acute = clamp01f(acute + 0.50f * (acute_target - acute) + 0.03f * reg1);
        const float chronic_target = clamp01f(0.75f * chronic + 0.25f * acute);
        chronic = clamp01f(chronic + 0.07f * (chronic_target - chronic));
        energy_store = clamp01f(energy_store + 0.03f * energy_n - 0.02f * stress - 0.01f * acute);
        repair = clamp01f(repair + 0.04f * energy_n - 0.03f * acute - 0.01f * chronic);
        reg1 = clamp01f(0.92f * reg1 + 0.08f * (1.0f - acute) + 0.01f * chemistry_noise[r * 2 + 0]);
        reg2 = clamp01f(0.97f * reg2 + 0.03f * chronic + 0.01f * chemistry_noise[r * 2 + 1]);
        const float acute_resolve = std::min(acute, 0.10f * repair + 0.04f * energy_store);
        acute = clamp01f(acute - acute_resolve);
        chronic = clamp01f(chronic + 0.25f * acute - 0.002f * repair);
        chemistry[c + 0] = energy_store;
        chemistry[c + 1] = repair;
        chemistry[c + 2] = acute;
        chemistry[c + 3] = chronic;
        chemistry[c + 4] = reg1;
        chemistry[c + 5] = reg2;
        chemistry_rows += 1.0;

        if (maintenance_enable) {
            const std::int64_t m = static_cast<std::int64_t>(slot) * 4;
            maintenance[m + 2] = clamp01f(maintenance[m + 2] * decay);
            if (action == 5) {
                schedule_requests += 1.0;
                const float opp_cost = std::max(0.0f, maintenance_opportunity_cost_coeff) * maintenance_opportunity[r];
                const float total_cost = std::max(0.0f, maintenance_energy_cost) + opp_cost;
                if (total_cost > 0.0f) {
                    energy[slot] = std::max(0.0f, energy[slot] - total_cost);
                    schedule_energy_cost_sum += total_cost;
                    schedule_opportunity_cost_sum += opp_cost;
                    maintenance_energy_spent_total[slot] += total_cost;
                    maintenance_opportunity_cost_total[slot] += opp_cost;
                }
                maintenance[m + 0] = clamp01f(maintenance[m + 0] + base_effect);
                maintenance[m + 1] = static_cast<float>(tick_index + maintenance_delay_ticks);
                maintenance[m + 3] = static_cast<float>(tick_index);
                maintenance_requests[slot] += 1;
            }
            const float pending = maintenance[m + 0];
            const std::int32_t due_tick = static_cast<std::int32_t>(std::llround(maintenance[m + 1]));
            if (pending > 0.0f && due_tick >= 0 && tick_index >= due_tick) {
                const float recent_load = maintenance[m + 2];
                const float eff_mult = 1.0f / (1.0f + std::max(0.0f, maintenance_diminishing_factor) * recent_load);
                const float applied = clamp01f(pending * eff_mult);
                chemistry[c + 1] = clamp01f(chemistry[c + 1] + 0.65f * applied);
                chemistry[c + 2] = clamp01f(chemistry[c + 2] - 0.55f * applied);
                chemistry[c + 3] = clamp01f(chemistry[c + 3] - 0.25f * applied);
                maintenance[m + 0] = 0.0f;
                maintenance[m + 1] = -1.0f;
                maintenance[m + 2] = clamp01f(recent_load + applied);
                maintenance_applied[slot] += 1;
                maintenance_applied_effect_total[slot] += applied;
                apply_count += 1.0;
                applied_effect_sum += applied;
            }
            maintenance_rows += 1.0;
        }
    }

    out_summary[0] = drive_rows;
    out_summary[1] = chemistry_rows;
    out_summary[2] = maintenance_rows;
    out_summary[3] = schedule_requests;
    out_summary[4] = schedule_energy_cost_sum;
    out_summary[5] = schedule_opportunity_cost_sum;
    out_summary[6] = apply_count;
    out_summary[7] = applied_effect_sum;
    out_summary[8] = episode0_selfharm_count;
    return 0;
}

extern "C" int cxxwave1f_emit_scalar_event_tape_v1(
    const std::int32_t* slots,
    const std::int32_t* actions,
    std::int64_t rows,
    std::int64_t capacity,
    const std::uint8_t* alive,
    const float* energy,
    const float* health,
    const float* injury,
    const float* drives,
    const float* chemistry,
    const std::int32_t* maintenance_requests,
    const std::int32_t* maintenance_applied,
    std::int32_t tick_index,
    std::int32_t* event_type,
    std::int32_t* event_tick,
    std::int32_t* event_slot,
    std::int32_t* event_action,
    float* value0,
    float* value1,
    float* value2,
    float* value3,
    double* out_summary
) {
    if (
        !slots || !actions || !alive || !energy || !health || !injury || !drives || !chemistry ||
        !maintenance_requests || !maintenance_applied || !event_type || !event_tick || !event_slot ||
        !event_action || !value0 || !value1 || !value2 || !value3 || !out_summary
    ) {
        return 1;
    }
    if (rows < 0 || capacity < 0) {
        return 2;
    }
    double emitted = 0.0;
    double skipped = 0.0;
    double energy_sum = 0.0;
    double health_sum = 0.0;
    for (std::int64_t r = 0; r < rows; ++r) {
        const std::int32_t slot = slots[r];
        if (slot < 0 || static_cast<std::int64_t>(slot) >= capacity || !alive[slot]) {
            event_type[r] = 0;
            event_tick[r] = tick_index;
            event_slot[r] = slot;
            event_action[r] = actions[r];
            value0[r] = 0.0f;
            value1[r] = 0.0f;
            value2[r] = 0.0f;
            value3[r] = 0.0f;
            skipped += 1.0;
            continue;
        }
        const std::int64_t d = static_cast<std::int64_t>(slot) * 4;
        const std::int64_t c = static_cast<std::int64_t>(slot) * 6;
        const float drive_sum = drives[d + 0] + drives[d + 1] + drives[d + 2] + drives[d + 3];
        const float chem_damage = chemistry[c + 2] + chemistry[c + 3];
        event_type[r] = 4100 + actions[r];
        event_tick[r] = tick_index;
        event_slot[r] = slot;
        event_action[r] = actions[r];
        value0[r] = energy[slot];
        value1[r] = health[slot];
        value2[r] = drive_sum;
        value3[r] = chem_damage + static_cast<float>(maintenance_requests[slot] + maintenance_applied[slot]) * 0.001f;
        emitted += 1.0;
        energy_sum += static_cast<double>(energy[slot]);
        health_sum += static_cast<double>(health[slot]);
    }
    out_summary[0] = emitted;
    out_summary[1] = skipped;
    out_summary[2] = energy_sum;
    out_summary[3] = health_sum;
    return 0;
}

extern "C" int cxxwave1f_pack_scalar_checkpoint_v1(
    std::int64_t capacity,
    const std::uint8_t* alive,
    const float* energy,
    const float* health,
    const float* injury,
    const float* drives,
    const float* prev_drive_delta,
    const std::int32_t* survival_steps,
    const float* drive_volatility,
    const float* drive_jitter,
    const float* episode0_drive_jitter_sum,
    const std::int32_t* episode0_selfharm_events,
    const float* chemistry,
    const float* maintenance,
    const std::int32_t* maintenance_requests,
    const std::int32_t* maintenance_applied,
    const float* maintenance_energy_spent_total,
    const float* maintenance_opportunity_cost_total,
    const float* maintenance_applied_effect_total,
    float* packed_f32,
    std::int32_t* packed_i32,
    double* out_summary
) {
    if (
        !alive || !energy || !health || !injury || !drives || !prev_drive_delta || !survival_steps ||
        !drive_volatility || !drive_jitter || !episode0_drive_jitter_sum || !episode0_selfharm_events ||
        !chemistry || !maintenance || !maintenance_requests || !maintenance_applied ||
        !maintenance_energy_spent_total || !maintenance_opportunity_cost_total ||
        !maintenance_applied_effect_total || !packed_f32 || !packed_i32 || !out_summary
    ) {
        return 1;
    }
    if (capacity < 0) {
        return 2;
    }
    double f_sum = 0.0;
    double i_sum = 0.0;
    for (std::int64_t slot = 0; slot < capacity; ++slot) {
        const std::int64_t f = slot * 27;
        const std::int64_t i = slot * 5;
        packed_f32[f + 0] = energy[slot];
        packed_f32[f + 1] = health[slot];
        packed_f32[f + 2] = injury[slot];
        for (std::int64_t k = 0; k < 4; ++k) {
            packed_f32[f + 3 + k] = drives[slot * 4 + k];
            packed_f32[f + 7 + k] = prev_drive_delta[slot * 4 + k];
        }
        packed_f32[f + 11] = drive_volatility[slot];
        packed_f32[f + 12] = drive_jitter[slot];
        packed_f32[f + 13] = episode0_drive_jitter_sum[slot];
        for (std::int64_t k = 0; k < 6; ++k) {
            packed_f32[f + 14 + k] = chemistry[slot * 6 + k];
        }
        for (std::int64_t k = 0; k < 4; ++k) {
            packed_f32[f + 20 + k] = maintenance[slot * 4 + k];
        }
        packed_f32[f + 24] = maintenance_energy_spent_total[slot];
        packed_f32[f + 25] = maintenance_opportunity_cost_total[slot];
        packed_f32[f + 26] = maintenance_applied_effect_total[slot];
        packed_i32[i + 0] = static_cast<std::int32_t>(alive[slot] ? 1 : 0);
        packed_i32[i + 1] = survival_steps[slot];
        packed_i32[i + 2] = episode0_selfharm_events[slot];
        packed_i32[i + 3] = maintenance_requests[slot];
        packed_i32[i + 4] = maintenance_applied[slot];
        for (std::int64_t k = 0; k < 27; ++k) {
            f_sum += static_cast<double>(packed_f32[f + k]) * static_cast<double>(k + 1);
        }
        for (std::int64_t k = 0; k < 5; ++k) {
            i_sum += static_cast<double>(packed_i32[i + k]) * static_cast<double>(k + 1);
        }
    }
    out_summary[0] = static_cast<double>(capacity);
    out_summary[1] = f_sum;
    out_summary[2] = i_sum;
    return 0;
}

extern "C" int cxxwave1f_restore_scalar_checkpoint_v1(
    std::int64_t capacity,
    const float* packed_f32,
    const std::int32_t* packed_i32,
    std::uint8_t* alive,
    float* energy,
    float* health,
    float* injury,
    float* drives,
    float* prev_drive_delta,
    std::int32_t* survival_steps,
    float* drive_volatility,
    float* drive_jitter,
    float* episode0_drive_jitter_sum,
    std::int32_t* episode0_selfharm_events,
    float* chemistry,
    float* maintenance,
    std::int32_t* maintenance_requests,
    std::int32_t* maintenance_applied,
    float* maintenance_energy_spent_total,
    float* maintenance_opportunity_cost_total,
    float* maintenance_applied_effect_total,
    double* out_summary
) {
    if (
        !packed_f32 || !packed_i32 || !alive || !energy || !health || !injury || !drives ||
        !prev_drive_delta || !survival_steps || !drive_volatility || !drive_jitter ||
        !episode0_drive_jitter_sum || !episode0_selfharm_events || !chemistry || !maintenance ||
        !maintenance_requests || !maintenance_applied || !maintenance_energy_spent_total ||
        !maintenance_opportunity_cost_total || !maintenance_applied_effect_total || !out_summary
    ) {
        return 1;
    }
    if (capacity < 0) {
        return 2;
    }
    double alive_count = 0.0;
    for (std::int64_t slot = 0; slot < capacity; ++slot) {
        const std::int64_t f = slot * 27;
        const std::int64_t i = slot * 5;
        alive[slot] = packed_i32[i + 0] ? 1 : 0;
        energy[slot] = packed_f32[f + 0];
        health[slot] = packed_f32[f + 1];
        injury[slot] = packed_f32[f + 2];
        for (std::int64_t k = 0; k < 4; ++k) {
            drives[slot * 4 + k] = packed_f32[f + 3 + k];
            prev_drive_delta[slot * 4 + k] = packed_f32[f + 7 + k];
        }
        drive_volatility[slot] = packed_f32[f + 11];
        drive_jitter[slot] = packed_f32[f + 12];
        episode0_drive_jitter_sum[slot] = packed_f32[f + 13];
        for (std::int64_t k = 0; k < 6; ++k) {
            chemistry[slot * 6 + k] = packed_f32[f + 14 + k];
        }
        for (std::int64_t k = 0; k < 4; ++k) {
            maintenance[slot * 4 + k] = packed_f32[f + 20 + k];
        }
        maintenance_energy_spent_total[slot] = packed_f32[f + 24];
        maintenance_opportunity_cost_total[slot] = packed_f32[f + 25];
        maintenance_applied_effect_total[slot] = packed_f32[f + 26];
        survival_steps[slot] = packed_i32[i + 1];
        episode0_selfharm_events[slot] = packed_i32[i + 2];
        maintenance_requests[slot] = packed_i32[i + 3];
        maintenance_applied[slot] = packed_i32[i + 4];
        if (alive[slot]) {
            alive_count += 1.0;
        }
    }
    out_summary[0] = static_cast<double>(capacity);
    out_summary[1] = alive_count;
    return 0;
}
