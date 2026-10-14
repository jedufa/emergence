#include <algorithm>
#include <cmath>
#include <cstdint>

static inline float bb1_sigmoid(float x) {
    const double xd = static_cast<double>(x);
    return static_cast<float>(1.0 / (1.0 + std::exp(-xd)));
}

static inline float bb1_tanh(float x) {
    return std::tanh(x);
}

extern "C" int cxxbackbone1_policy_authority_v1(
    const std::int32_t* policy_kind,
    const std::int32_t* row_slots,
    const float* observations,
    const float* drives,
    const float* gain,
    const float* aggression,
    const float* temperature,
    const std::int32_t* effective_memory_dim,
    const float* mlp_pack,
    const std::int64_t mlp_pack_stride,
    const float* gru_pack,
    const std::int64_t gru_pack_stride,
    const float* hmem,
    const std::int32_t* action_cache_cursor,
    const std::int64_t cache_stride,
    const std::int64_t rows,
    const std::int64_t slots,
    const std::int64_t obs_cols,
    const std::int64_t mlp_hidden,
    const std::int64_t gru_hidden,
    const std::int64_t actions,
    const float w_hunger_gather,
    const float w_pain_rest,
    const float w_fatigue_rest,
    const float w_pain_attack,
    const float w_fear_attack,
    std::int32_t* out_actions,
    float* out_logits,
    float* out_hmem,
    std::int32_t* out_action_cache_cursor
) {
    if (
        !policy_kind || !row_slots || !observations || !drives || !gain || !aggression || !temperature
        || !effective_memory_dim || !mlp_pack || !gru_pack || !hmem || !action_cache_cursor
        || !out_actions || !out_logits || !out_hmem || !out_action_cache_cursor
    ) {
        return 1;
    }
    if (rows < 0 || slots < 0 || obs_cols <= 0 || mlp_hidden <= 0 || gru_hidden <= 0 || actions != 7) {
        return 2;
    }
    const std::int64_t expected_mlp_stride = mlp_hidden * obs_cols + mlp_hidden + mlp_hidden * mlp_hidden + mlp_hidden + actions * mlp_hidden + actions;
    const std::int64_t expected_gru_stride = gru_hidden * obs_cols + gru_hidden + 3 * gru_hidden * gru_hidden + 3 * gru_hidden + 3 * gru_hidden * gru_hidden + 3 * gru_hidden + actions * gru_hidden + actions;
    if (mlp_pack_stride != expected_mlp_stride || gru_pack_stride != expected_gru_stride) {
        return 3;
    }
    for (std::int64_t slot = 0; slot < slots; ++slot) {
        for (std::int64_t h = 0; h < gru_hidden; ++h) {
            out_hmem[slot * gru_hidden + h] = hmem[slot * gru_hidden + h];
        }
        out_action_cache_cursor[slot] = action_cache_cursor[slot];
    }

    for (std::int64_t row = 0; row < rows; ++row) {
        const std::int32_t slot32 = row_slots[row];
        if (slot32 < 0 || static_cast<std::int64_t>(slot32) >= slots) {
            return 4;
        }
        const std::int64_t slot = static_cast<std::int64_t>(slot32);
        float logits[7] = {0, 0, 0, 0, 0, 0, 0};
        if (policy_kind[row] == 0) {
            const float* pack = mlp_pack + slot * mlp_pack_stride;
            const std::int64_t w1_off = 0;
            const std::int64_t b1_off = w1_off + mlp_hidden * obs_cols;
            const std::int64_t w2_off = b1_off + mlp_hidden;
            const std::int64_t b2_off = w2_off + mlp_hidden * mlp_hidden;
            const std::int64_t w3_off = b2_off + mlp_hidden;
            const std::int64_t b3_off = w3_off + actions * mlp_hidden;
            float h1[128];
            float h2[128];
            if (mlp_hidden > 128) {
                return 5;
            }
            for (std::int64_t h = 0; h < mlp_hidden; ++h) {
                float acc = pack[b1_off + h];
                for (std::int64_t c = 0; c < obs_cols; ++c) {
                    acc = static_cast<float>(acc + static_cast<float>(pack[w1_off + h * obs_cols + c] * observations[row * obs_cols + c]));
                }
                h1[h] = bb1_tanh(acc);
            }
            for (std::int64_t h = 0; h < mlp_hidden; ++h) {
                float acc = pack[b2_off + h];
                for (std::int64_t c = 0; c < mlp_hidden; ++c) {
                    acc = static_cast<float>(acc + static_cast<float>(pack[w2_off + h * mlp_hidden + c] * h1[c]));
                }
                h2[h] = bb1_tanh(acc);
            }
            for (std::int64_t a = 0; a < actions; ++a) {
                float acc = pack[b3_off + a];
                for (std::int64_t h = 0; h < mlp_hidden; ++h) {
                    acc = static_cast<float>(acc + static_cast<float>(pack[w3_off + a * mlp_hidden + h] * h2[h]));
                }
                logits[a] = acc;
            }
        } else if (policy_kind[row] == 1) {
            const float* pack = gru_pack + slot * gru_pack_stride;
            const std::int64_t encw_off = 0;
            const std::int64_t encb_off = encw_off + gru_hidden * obs_cols;
            const std::int64_t wih_off = encb_off + gru_hidden;
            const std::int64_t bih_off = wih_off + 3 * gru_hidden * gru_hidden;
            const std::int64_t whh_off = bih_off + 3 * gru_hidden;
            const std::int64_t bhh_off = whh_off + 3 * gru_hidden * gru_hidden;
            const std::int64_t headw_off = bhh_off + 3 * gru_hidden;
            const std::int64_t headb_off = headw_off + actions * gru_hidden;
            float z_in[128];
            float h_prev[128];
            float h_next[128];
            if (gru_hidden > 128) {
                return 6;
            }
            const std::int32_t mem_dim = std::max(0, std::min(static_cast<std::int32_t>(gru_hidden), effective_memory_dim[row]));
            for (std::int64_t h = 0; h < gru_hidden; ++h) {
                h_prev[h] = h < mem_dim ? hmem[slot * gru_hidden + h] : 0.0f;
            }
            for (std::int64_t h = 0; h < gru_hidden; ++h) {
                float acc = pack[encb_off + h];
                for (std::int64_t c = 0; c < obs_cols; ++c) {
                    acc = static_cast<float>(acc + static_cast<float>(pack[encw_off + h * obs_cols + c] * observations[row * obs_cols + c]));
                }
                z_in[h] = bb1_tanh(acc);
            }
            for (std::int64_t h = 0; h < gru_hidden; ++h) {
                float i_r = pack[bih_off + h];
                float i_z = pack[bih_off + gru_hidden + h];
                float i_n = pack[bih_off + 2 * gru_hidden + h];
                float h_r = pack[bhh_off + h];
                float h_z = pack[bhh_off + gru_hidden + h];
                float h_n = pack[bhh_off + 2 * gru_hidden + h];
                for (std::int64_t c = 0; c < gru_hidden; ++c) {
                    i_r = static_cast<float>(i_r + static_cast<float>(pack[wih_off + h * gru_hidden + c] * z_in[c]));
                    i_z = static_cast<float>(i_z + static_cast<float>(pack[wih_off + (gru_hidden + h) * gru_hidden + c] * z_in[c]));
                    i_n = static_cast<float>(i_n + static_cast<float>(pack[wih_off + (2 * gru_hidden + h) * gru_hidden + c] * z_in[c]));
                    h_r = static_cast<float>(h_r + static_cast<float>(pack[whh_off + h * gru_hidden + c] * h_prev[c]));
                    h_z = static_cast<float>(h_z + static_cast<float>(pack[whh_off + (gru_hidden + h) * gru_hidden + c] * h_prev[c]));
                    h_n = static_cast<float>(h_n + static_cast<float>(pack[whh_off + (2 * gru_hidden + h) * gru_hidden + c] * h_prev[c]));
                }
                const float r = bb1_sigmoid(static_cast<float>(i_r + h_r));
                const float z = bb1_sigmoid(static_cast<float>(i_z + h_z));
                const float n_gate = bb1_tanh(static_cast<float>(i_n + static_cast<float>(r * h_n)));
                h_next[h] = h < mem_dim ? static_cast<float>(static_cast<float>(1.0f - z) * n_gate + static_cast<float>(z * h_prev[h])) : 0.0f;
            }
            for (std::int64_t a = 0; a < actions; ++a) {
                float acc = pack[headb_off + a];
                for (std::int64_t h = 0; h < gru_hidden; ++h) {
                    acc = static_cast<float>(acc + static_cast<float>(pack[headw_off + a * gru_hidden + h] * h_next[h]));
                }
                logits[a] = acc;
            }
            for (std::int64_t h = 0; h < gru_hidden; ++h) {
                out_hmem[slot * gru_hidden + h] = h_next[h];
            }
        } else {
            return 7;
        }

        logits[6] = static_cast<float>(logits[6] + aggression[row]);
        const float temp = std::max(temperature[row], 0.05f);
        for (std::int64_t a = 0; a < actions; ++a) {
            logits[a] = static_cast<float>(logits[a] / temp);
        }
        const float* d = drives + row * 4;
        logits[4] = static_cast<float>(logits[4] + gain[row] * (w_hunger_gather * d[0]));
        logits[5] = static_cast<float>(logits[5] + gain[row] * (w_pain_rest * d[1] + w_fatigue_rest * d[2]));
        logits[6] = static_cast<float>(logits[6] - gain[row] * (w_pain_attack * d[1] + w_fear_attack * d[3]));
        std::int32_t best = 0;
        float best_value = logits[0];
        for (std::int64_t a = 0; a < actions; ++a) {
            out_logits[row * actions + a] = logits[a];
            if (logits[a] > best_value) {
                best_value = logits[a];
                best = static_cast<std::int32_t>(a);
            }
        }
        out_actions[row] = best;
        if (cache_stride > 0) {
            out_action_cache_cursor[slot] = static_cast<std::int32_t>((static_cast<std::int64_t>(out_action_cache_cursor[slot]) + 1) % cache_stride);
        } else {
            out_action_cache_cursor[slot] += 1;
        }
    }
    return 0;
}
