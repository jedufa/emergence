#include <cstdint>

namespace {
constexpr int kPhases = 4;
constexpr int kActionBuckets = 8;
constexpr int kMemoryScalars = 10;
constexpr int kPredictionScalars = 11;
constexpr int kActiveHazard = 3;
constexpr int kCue = 1;
constexpr int kHiddenGap = 2;
constexpr int kBaseline = 0;
constexpr int kWindowActiveConsequence = 3;

int action_label_code(const int action) {
    if (action >= 0 && action <= 6) {
        return action;
    }
    return 7;
}

int movement_scalar_index(
    const int action,
    const int y,
    const int x,
    const int cy,
    const int cx,
    const int height,
    const int width
) {
    int dy = 0;
    int dx = 0;
    bool move = true;
    if (action == 0) {
        dy = -1;
    } else if (action == 1) {
        dy = 1;
    } else if (action == 2) {
        dx = 1;
    } else if (action == 3) {
        dx = -1;
    } else {
        move = false;
    }
    if (!move) {
        return 6;
    }
    const int pre_y = y > cy ? y - cy : cy - y;
    const int pre_x = x > cx ? x - cx : cx - x;
    const int pre = pre_y > pre_x ? pre_y : pre_x;
    int yy = y + dy;
    int xx = x + dx;
    if (yy < 0) yy = 0;
    if (xx < 0) xx = 0;
    if (yy >= height) yy = height - 1;
    if (xx >= width) xx = width - 1;
    const int post_y = yy > cy ? yy - cy : cy - yy;
    const int post_x = xx > cx ? xx - cx : cx - xx;
    const int post = post_y > post_x ? post_y : post_x;
    if (post < pre) return 3;
    if (post > pre) return 4;
    return 5;
}

int find_or_add_group(
    const int event_id,
    const int window_code,
    const int center_y,
    const int center_x,
    const int trend_sign,
    const int configured_delay,
    const int time_to_consequence,
    const float cue_ambiguity,
    const int max_groups,
    int* group_count,
    int* out_event_id,
    int* out_window_code,
    int* out_center_y,
    int* out_center_x,
    int* out_trend_sign,
    int* out_configured_delay,
    int* out_time_to_consequence,
    float* out_cue_ambiguity
) {
    for (int g = 0; g < *group_count; ++g) {
        if (out_event_id[g] == event_id && out_window_code[g] == window_code) {
            return g;
        }
    }
    if (*group_count >= max_groups) {
        return -1;
    }
    const int g = *group_count;
    *group_count += 1;
    out_event_id[g] = event_id;
    out_window_code[g] = window_code;
    out_center_y[g] = center_y;
    out_center_x[g] = center_x;
    out_trend_sign[g] = trend_sign;
    out_configured_delay[g] = configured_delay;
    out_time_to_consequence[g] = time_to_consequence;
    out_cue_ambiguity[g] = cue_ambiguity;
    return g;
}

void unique_top_for_phase(
    const int phase,
    const std::int64_t n,
    const int* phase_codes,
    const int* codes,
    int* unique_count,
    int* top_code,
    int* top_count
) {
    *unique_count = 0;
    *top_code = 0;
    *top_count = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        const int code = codes[i];
        if (phase_codes[i] != phase || code <= 0) {
            continue;
        }
        bool seen = false;
        for (std::int64_t j = 0; j < i; ++j) {
            if (phase_codes[j] == phase && codes[j] == code) {
                seen = true;
                break;
            }
        }
        if (seen) {
            continue;
        }
        int count = 0;
        for (std::int64_t k = i; k < n; ++k) {
            if (phase_codes[k] == phase && codes[k] == code) {
                count += 1;
            }
        }
        *unique_count += 1;
        if (count > *top_count || (count == *top_count && (*top_code == 0 || code < *top_code))) {
            *top_count = count;
            *top_code = code;
        }
    }
}

void unique_top_for_group(
    const int event_id,
    const int window_code,
    const std::int64_t n,
    const std::uint8_t* has_prediction,
    const int* prediction_event_id,
    const int* prediction_window_code,
    const int* codes,
    int* unique_count,
    int* top_code,
    int* top_count
) {
    *unique_count = 0;
    *top_code = 0;
    *top_count = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        const int code = codes[i];
        if (!has_prediction[i] || prediction_event_id[i] != event_id || prediction_window_code[i] != window_code || code <= 0) {
            continue;
        }
        bool seen = false;
        for (std::int64_t j = 0; j < i; ++j) {
            if (has_prediction[j] && prediction_event_id[j] == event_id && prediction_window_code[j] == window_code && codes[j] == code) {
                seen = true;
                break;
            }
        }
        if (seen) {
            continue;
        }
        int count = 0;
        for (std::int64_t k = i; k < n; ++k) {
            if (has_prediction[k] && prediction_event_id[k] == event_id && prediction_window_code[k] == window_code && codes[k] == code) {
                count += 1;
            }
        }
        *unique_count += 1;
        if (count > *top_count || (count == *top_count && (*top_code == 0 || code < *top_code))) {
            *top_count = count;
            *top_code = code;
        }
    }
}
}

extern "C" int cxxcore20_compact_summary_run(
    const std::int64_t n,
    const int grid_height,
    const int grid_width,
    const int* actions,
    const int* pre_y,
    const int* pre_x,
    const int* pre_alive,
    const int* post_alive,
    const float* pre_health,
    const float* post_health,
    const int* phase_codes,
    const int* post_phase_codes,
    const float* effective_memory,
    const float* memory_abs,
    const int* lineage_codes,
    const int* founder_codes,
    const std::uint8_t* has_prediction,
    const int* prediction_event_id,
    const int* prediction_window_code,
    const int* prediction_center_y,
    const int* prediction_center_x,
    const int* prediction_trend_sign,
    const int* prediction_configured_delay,
    const int* prediction_time_to_consequence,
    const float* prediction_cue_ambiguity,
    const std::uint8_t* prediction_prior_cue,
    const int max_groups,
    std::uint8_t* out_memory_phase_present,
    int* out_memory_agent_count,
    int* out_memory_action_counts,
    int* out_memory_scalar_counts,
    float* out_memory_damage_total,
    float* out_memory_effective_memory_sum,
    float* out_memory_abs_sum,
    int* out_memory_lineage_unique_count,
    int* out_memory_top_lineage_code,
    int* out_memory_top_lineage_count,
    int* out_memory_founder_unique_count,
    int* out_prediction_group_count,
    int* out_prediction_event_id,
    int* out_prediction_window_code,
    int* out_prediction_center_y,
    int* out_prediction_center_x,
    int* out_prediction_trend_sign,
    int* out_prediction_configured_delay,
    int* out_prediction_time_to_consequence,
    float* out_prediction_cue_ambiguity,
    int* out_prediction_action_counts,
    int* out_prediction_scalar_counts,
    int* out_prediction_top_lineage_code
) {
    if (n < 0 || max_groups <= 0) {
        return 1;
    }
    *out_prediction_group_count = 0;
    for (int p = 0; p < kPhases; ++p) {
        out_memory_phase_present[p] = 0;
        out_memory_agent_count[p] = 0;
        out_memory_damage_total[p] = 0.0f;
        out_memory_effective_memory_sum[p] = 0.0f;
        out_memory_abs_sum[p] = 0.0f;
        out_memory_lineage_unique_count[p] = 0;
        out_memory_top_lineage_code[p] = 0;
        out_memory_top_lineage_count[p] = 0;
        out_memory_founder_unique_count[p] = 0;
        for (int a = 0; a < kActionBuckets; ++a) {
            out_memory_action_counts[p * kActionBuckets + a] = 0;
        }
        for (int s = 0; s < kMemoryScalars; ++s) {
            out_memory_scalar_counts[p * kMemoryScalars + s] = 0;
        }
    }
    for (int g = 0; g < max_groups; ++g) {
        out_prediction_event_id[g] = 0;
        out_prediction_window_code[g] = -1;
        out_prediction_center_y[g] = 0;
        out_prediction_center_x[g] = 0;
        out_prediction_trend_sign[g] = 0;
        out_prediction_configured_delay[g] = 0;
        out_prediction_time_to_consequence[g] = 0;
        out_prediction_cue_ambiguity[g] = 0.0f;
        out_prediction_top_lineage_code[g] = 0;
        for (int a = 0; a < kActionBuckets; ++a) {
            out_prediction_action_counts[g * kActionBuckets + a] = 0;
        }
        for (int s = 0; s < kPredictionScalars; ++s) {
            out_prediction_scalar_counts[g * kPredictionScalars + s] = 0;
        }
    }

    for (std::int64_t i = 0; i < n; ++i) {
        int phase = phase_codes[i];
        if (phase < 0 || phase >= kPhases) {
            phase = 0;
        }
        const int action = actions[i];
        out_memory_phase_present[phase] = 1;
        out_memory_agent_count[phase] += 1;
        const int action_bucket = action >= 0 && action < kActionBuckets ? action : 7;
        out_memory_action_counts[phase * kActionBuckets + action_bucket] += 1;
        if (action >= 0 && action <= 3) {
            out_memory_scalar_counts[phase * kMemoryScalars + 0] += 1;
        } else if (action == 4) {
            out_memory_scalar_counts[phase * kMemoryScalars + 1] += 1;
        } else if (action == 5) {
            out_memory_scalar_counts[phase * kMemoryScalars + 2] += 1;
        } else if (action < 0 || action >= 7) {
            out_memory_scalar_counts[phase * kMemoryScalars + 3] += 1;
        }
        const float damage = pre_health[i] - post_health[i];
        if (damage > 0.0f) {
            out_memory_scalar_counts[phase * kMemoryScalars + 4] += 1;
            out_memory_damage_total[phase] += damage;
            if (phase == kCue || phase == kHiddenGap || phase == kActiveHazard) {
                out_memory_scalar_counts[phase * kMemoryScalars + 5] += 1;
            }
        }
        if (phase == kActiveHazard) {
            out_memory_scalar_counts[phase * kMemoryScalars + 6] += 1;
        }
        if (post_alive[i]) {
            const int post_phase = post_phase_codes[i];
            if (phase == kCue || phase == kHiddenGap) {
                if (post_phase == kBaseline) {
                    out_memory_scalar_counts[phase * kMemoryScalars + 7] += 1;
                }
            }
            if (phase != kActiveHazard && post_phase == kActiveHazard) {
                out_memory_scalar_counts[phase * kMemoryScalars + 8] += 1;
            }
            if (phase == kActiveHazard && post_phase != kActiveHazard) {
                out_memory_scalar_counts[phase * kMemoryScalars + 9] += 1;
            }
        }
        out_memory_effective_memory_sum[phase] += effective_memory[i];
        out_memory_abs_sum[phase] += memory_abs[i];

        if (has_prediction[i]) {
            const int group = find_or_add_group(
                prediction_event_id[i],
                prediction_window_code[i],
                prediction_center_y[i],
                prediction_center_x[i],
                prediction_trend_sign[i],
                prediction_configured_delay[i],
                prediction_time_to_consequence[i],
                prediction_cue_ambiguity[i],
                max_groups,
                out_prediction_group_count,
                out_prediction_event_id,
                out_prediction_window_code,
                out_prediction_center_y,
                out_prediction_center_x,
                out_prediction_trend_sign,
                out_prediction_configured_delay,
                out_prediction_time_to_consequence,
                out_prediction_cue_ambiguity
            );
            if (group < 0) {
                return 2;
            }
            out_prediction_action_counts[group * kActionBuckets + action_label_code(action)] += 1;
            out_prediction_scalar_counts[group * kPredictionScalars + 0] += 1;
            out_prediction_scalar_counts[group * kPredictionScalars + 1] += 1;
            if (action == 4) {
                out_prediction_scalar_counts[group * kPredictionScalars + 2] += 1;
            }
            const int move_idx = movement_scalar_index(
                action,
                pre_y[i],
                pre_x[i],
                prediction_center_y[i],
                prediction_center_x[i],
                grid_height,
                grid_width
            );
            out_prediction_scalar_counts[group * kPredictionScalars + move_idx] += 1;
            if (prediction_window_code[i] == kWindowActiveConsequence) {
                if (prediction_prior_cue[i]) {
                    out_prediction_scalar_counts[group * kPredictionScalars + 9] += 1;
                } else {
                    out_prediction_scalar_counts[group * kPredictionScalars + 10] += 1;
                }
            }
        }
    }

    for (int phase = 0; phase < kPhases; ++phase) {
        int unique = 0;
        int top_code = 0;
        int top_count = 0;
        unique_top_for_phase(phase, n, phase_codes, lineage_codes, &unique, &top_code, &top_count);
        out_memory_lineage_unique_count[phase] = unique;
        out_memory_top_lineage_code[phase] = top_code;
        out_memory_top_lineage_count[phase] = top_count;
        int founder_unique = 0;
        int founder_top = 0;
        int founder_top_count = 0;
        unique_top_for_phase(phase, n, phase_codes, founder_codes, &founder_unique, &founder_top, &founder_top_count);
        out_memory_founder_unique_count[phase] = founder_unique;
    }

    for (int group = 0; group < *out_prediction_group_count; ++group) {
        int unique = 0;
        int top_code = 0;
        int top_count = 0;
        unique_top_for_group(
            out_prediction_event_id[group],
            out_prediction_window_code[group],
            n,
            has_prediction,
            prediction_event_id,
            prediction_window_code,
            lineage_codes,
            &unique,
            &top_code,
            &top_count
        );
        out_prediction_scalar_counts[group * kPredictionScalars + 7] = unique;
        out_prediction_scalar_counts[group * kPredictionScalars + 8] = top_count;
        out_prediction_top_lineage_code[group] = top_code;
    }
    return 0;
}
