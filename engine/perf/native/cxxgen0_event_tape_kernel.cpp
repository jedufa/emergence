#include <cstdint>

extern "C" int cxxgen0_build_event_tape_v1(
    const int32_t* actions_by_tick,
    const float* pre_energy,
    const float* post_energy,
    const uint8_t* died_mask,
    const uint8_t* moved_mask,
    const float* food_gained,
    int64_t ticks,
    int64_t agents,
    int32_t* out_event_type,
    int32_t* out_tick,
    int32_t* out_agent_index,
    float* out_value0,
    float* out_value1,
    int64_t max_rows,
    int64_t* out_rows,
    double* out_summary
) {
    if (!actions_by_tick || !pre_energy || !post_energy || !died_mask || !moved_mask || !food_gained ||
        !out_event_type || !out_tick || !out_agent_index || !out_value0 || !out_value1 || !out_rows || !out_summary) {
        return 1;
    }
    if (ticks < 0 || agents < 0 || max_rows <= 0) {
        return 2;
    }

    double energy_delta = 0.0;
    double death_count = 0.0;
    double moved_count = 0.0;
    double food_total = 0.0;
    double action_counts[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (int64_t t = 0; t < ticks; ++t) {
        for (int64_t a = 0; a < agents; ++a) {
            const int32_t action = actions_by_tick[t * agents + a];
            if (action >= 0 && action < 7) {
                action_counts[action] += 1.0;
            }
        }
    }

    for (int64_t a = 0; a < agents; ++a) {
        energy_delta += static_cast<double>(post_energy[a] - pre_energy[a]);
        if (died_mask[a]) {
            death_count += 1.0;
        }
        if (moved_mask[a]) {
            moved_count += 1.0;
        }
        food_total += static_cast<double>(food_gained[a]);
    }

    int64_t cursor = 0;
    out_event_type[cursor] = 1;
    out_tick[cursor] = static_cast<int32_t>(ticks);
    out_agent_index[cursor] = -1;
    out_value0[cursor] = static_cast<float>(energy_delta);
    out_value1[cursor] = static_cast<float>(death_count);
    ++cursor;

    for (int64_t a = 0; a < agents; ++a) {
        if (moved_mask[a]) {
            if (cursor >= max_rows) {
                return 3;
            }
            out_event_type[cursor] = 2;
            out_tick[cursor] = -1;
            out_agent_index[cursor] = static_cast<int32_t>(a);
            out_value0[cursor] = 1.0f;
            out_value1[cursor] = 0.0f;
            ++cursor;
        }
        if (food_gained[a] != 0.0f) {
            if (cursor >= max_rows) {
                return 3;
            }
            out_event_type[cursor] = 3;
            out_tick[cursor] = -1;
            out_agent_index[cursor] = static_cast<int32_t>(a);
            out_value0[cursor] = food_gained[a];
            out_value1[cursor] = 0.0f;
            ++cursor;
        }
        if (died_mask[a]) {
            if (cursor >= max_rows) {
                return 3;
            }
            out_event_type[cursor] = 4;
            out_tick[cursor] = -1;
            out_agent_index[cursor] = static_cast<int32_t>(a);
            out_value0[cursor] = post_energy[a];
            out_value1[cursor] = 0.0f;
            ++cursor;
        }
    }

    out_rows[0] = cursor;
    out_summary[0] = energy_delta;
    out_summary[1] = death_count;
    out_summary[2] = moved_count;
    out_summary[3] = food_total;
    for (int i = 0; i < 7; ++i) {
        out_summary[4 + i] = action_counts[i];
    }
    return 0;
}
