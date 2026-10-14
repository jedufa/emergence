#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" int cxxbackbone1_lineage_lifecycle_v1(
    const std::int64_t* parent_a,
    const std::int64_t* parent_b,
    const std::int64_t* founder_a,
    const std::int64_t* founder_b,
    const std::int32_t* depth_a,
    const std::int32_t* depth_b,
    const std::int32_t* mutation_delta_q,
    const std::int64_t child_count,
    std::int64_t* out_parent_a,
    std::int64_t* out_parent_b,
    std::int64_t* out_founder,
    std::int32_t* out_depth,
    std::int32_t* out_mutation_delta_q,
    double* out_summary
) {
    if (!parent_a || !parent_b || !founder_a || !founder_b || !depth_a || !depth_b || !mutation_delta_q
        || !out_parent_a || !out_parent_b || !out_founder || !out_depth || !out_mutation_delta_q || !out_summary) {
        return 1;
    }
    if (child_count < 0) {
        return 2;
    }
    double depth_sum = 0.0;
    double founder_switches = 0.0;
    double mutation_abs_sum = 0.0;
    for (std::int64_t i = 0; i < child_count; ++i) {
        const bool choose_b = mutation_delta_q[i] < 0;
        const std::int64_t founder = choose_b ? founder_b[i] : founder_a[i];
        const std::int32_t depth = std::max(depth_a[i], depth_b[i]) + 1;
        out_parent_a[i] = parent_a[i];
        out_parent_b[i] = parent_b[i];
        out_founder[i] = founder;
        out_depth[i] = depth;
        out_mutation_delta_q[i] = mutation_delta_q[i];
        depth_sum += static_cast<double>(depth);
        founder_switches += choose_b ? 1.0 : 0.0;
        mutation_abs_sum += static_cast<double>(std::abs(mutation_delta_q[i]));
    }
    out_summary[0] = static_cast<double>(child_count);
    out_summary[1] = depth_sum;
    out_summary[2] = founder_switches;
    out_summary[3] = mutation_abs_sum;
    return 0;
}

extern "C" int cxxbackbone1_lifecycle_transactions_v1(
    const std::int32_t* alive,
    const std::int32_t* energy_q,
    const std::int32_t* maturity_q,
    const std::int32_t* genome_q,
    const std::int32_t* rng_q,
    const std::int64_t slot_count,
    std::int32_t* out_alive,
    std::int64_t* out_parent_slot,
    std::int32_t* out_genome_q,
    std::int32_t* out_event_type,
    double* out_summary
) {
    if (!alive || !energy_q || !maturity_q || !genome_q || !rng_q || !out_alive || !out_parent_slot
        || !out_genome_q || !out_event_type || !out_summary) {
        return 1;
    }
    if (slot_count < 0) {
        return 2;
    }
    double births = 0.0;
    double deaths = 0.0;
    double genome_sum = 0.0;
    for (std::int64_t i = 0; i < slot_count; ++i) {
        out_alive[i] = alive[i];
        out_parent_slot[i] = -1;
        out_genome_q[i] = genome_q[i];
        out_event_type[i] = 0;
        if (alive[i] != 0 && energy_q[i] <= 200) {
            out_alive[i] = 0;
            out_event_type[i] = 1;
            deaths += 1.0;
        }
    }
    for (std::int64_t i = 0; i < slot_count; ++i) {
        if (out_alive[i] != 0 || rng_q[i] >= 375) {
            continue;
        }
        std::int64_t parent = -1;
        for (std::int64_t step = 1; step <= slot_count; ++step) {
            const std::int64_t candidate = (i + step) % slot_count;
            if (out_alive[candidate] != 0 && maturity_q[candidate] >= 500) {
                parent = candidate;
                break;
            }
        }
        if (parent >= 0) {
            const std::int32_t mutation = (rng_q[i] % 21) - 10;
            out_alive[i] = 1;
            out_parent_slot[i] = parent;
            out_genome_q[i] = genome_q[parent] + mutation;
            out_event_type[i] = 2;
            births += 1.0;
        }
    }
    for (std::int64_t i = 0; i < slot_count; ++i) {
        genome_sum += static_cast<double>(out_genome_q[i]);
    }
    out_summary[0] = births;
    out_summary[1] = deaths;
    out_summary[2] = genome_sum;
    out_summary[3] = static_cast<double>(slot_count);
    return 0;
}

extern "C" int cxxbackbone1_checkpoint_restore_v1(
    const std::int64_t* slot_token,
    const std::int32_t* policy_row,
    const std::int32_t* hmem_q,
    const std::int64_t* founder,
    const std::uint64_t* rng_state,
    const std::int64_t count,
    std::int64_t* out_slot_token,
    std::int32_t* out_policy_row,
    std::int32_t* out_hmem_q,
    std::int64_t* out_founder,
    std::uint64_t* out_rng_state,
    std::int32_t* out_resumed_action,
    double* out_summary
) {
    if (!slot_token || !policy_row || !hmem_q || !founder || !rng_state || !out_slot_token || !out_policy_row
        || !out_hmem_q || !out_founder || !out_rng_state || !out_resumed_action || !out_summary) {
        return 1;
    }
    if (count < 0) {
        return 2;
    }
    double checksum = 0.0;
    double action_sum = 0.0;
    for (std::int64_t i = 0; i < count; ++i) {
        out_slot_token[i] = slot_token[i];
        out_policy_row[i] = policy_row[i];
        out_hmem_q[i] = hmem_q[i];
        out_founder[i] = founder[i];
        out_rng_state[i] = rng_state[i];
        const std::uint64_t action = (static_cast<std::uint64_t>(slot_token[i])
            + static_cast<std::uint64_t>(policy_row[i] * 17)
            + static_cast<std::uint64_t>(hmem_q[i] * 31)
            + rng_state[i]) % 7ULL;
        out_resumed_action[i] = static_cast<std::int32_t>(action);
        checksum += static_cast<double>((rng_state[i] ^ static_cast<std::uint64_t>(slot_token[i])) & 0xFFFFULL);
        action_sum += static_cast<double>(out_resumed_action[i]);
    }
    out_summary[0] = static_cast<double>(count);
    out_summary[1] = checksum;
    out_summary[2] = action_sum;
    out_summary[3] = 0.0;
    return 0;
}

extern "C" int cxxbackbone1_public_output_v1(
    const std::int64_t* event_id,
    const std::int64_t* slot_id,
    const std::int32_t* event_type,
    const std::int64_t row_count,
    char* out_jsonl,
    const std::int64_t out_jsonl_capacity,
    char* out_csv,
    const std::int64_t out_csv_capacity,
    std::int64_t* out_jsonl_size,
    std::int64_t* out_csv_size
) {
    if (!event_id || !slot_id || !event_type || !out_jsonl || !out_csv || !out_jsonl_size || !out_csv_size) {
        return 1;
    }
    if (row_count < 0 || out_jsonl_capacity <= 0 || out_csv_capacity <= 0) {
        return 2;
    }
    std::int64_t jp = 0;
    std::int64_t cp = std::snprintf(out_csv, static_cast<std::size_t>(out_csv_capacity), "event_id,slot_id,event_type\n");
    if (cp < 0 || cp >= out_csv_capacity) {
        return 3;
    }
    for (std::int64_t i = 0; i < row_count; ++i) {
        const int jw = std::snprintf(
            out_jsonl + jp,
            static_cast<std::size_t>(out_jsonl_capacity - jp),
            "{\"event_id\":%lld,\"slot_id\":%lld,\"event_type\":%d}\n",
            static_cast<long long>(event_id[i]),
            static_cast<long long>(slot_id[i]),
            static_cast<int>(event_type[i])
        );
        if (jw < 0 || jp + jw >= out_jsonl_capacity) {
            return 4;
        }
        jp += jw;
        const int cw = std::snprintf(
            out_csv + cp,
            static_cast<std::size_t>(out_csv_capacity - cp),
            "%lld,%lld,%d\n",
            static_cast<long long>(event_id[i]),
            static_cast<long long>(slot_id[i]),
            static_cast<int>(event_type[i])
        );
        if (cw < 0 || cp + cw >= out_csv_capacity) {
            return 5;
        }
        cp += cw;
    }
    *out_jsonl_size = jp;
    *out_csv_size = cp;
    return 0;
}

extern "C" int cxxbackbone1_multicell_batch_v1(
    const float* observations,
    const float* weights,
    const float* bias,
    const std::int64_t* cell_offsets,
    const std::int64_t cell_count,
    const std::int64_t obs_dim,
    const std::int64_t action_dim,
    std::int32_t* out_actions,
    double* out_cell_action_sums
) {
    if (!observations || !weights || !bias || !cell_offsets || !out_actions || !out_cell_action_sums) {
        return 1;
    }
    if (cell_count < 0 || obs_dim <= 0 || action_dim <= 0) {
        return 2;
    }
    for (std::int64_t c = 0; c < cell_count; ++c) {
        const std::int64_t start = cell_offsets[c];
        const std::int64_t end = cell_offsets[c + 1];
        if (end < start) {
            return 3;
        }
        double sum = 0.0;
        for (std::int64_t row = start; row < end; ++row) {
            std::int64_t best = 0;
            float best_logit = bias[0];
            for (std::int64_t k = 0; k < obs_dim; ++k) {
                best_logit += observations[row * obs_dim + k] * weights[k * action_dim];
            }
            for (std::int64_t action = 1; action < action_dim; ++action) {
                float logit = bias[action];
                for (std::int64_t k = 0; k < obs_dim; ++k) {
                    logit += observations[row * obs_dim + k] * weights[k * action_dim + action];
                }
                if (logit > best_logit) {
                    best_logit = logit;
                    best = action;
                }
            }
            out_actions[row] = static_cast<std::int32_t>(best);
            sum += static_cast<double>(best);
        }
        out_cell_action_sums[c] = sum;
    }
    return 0;
}
