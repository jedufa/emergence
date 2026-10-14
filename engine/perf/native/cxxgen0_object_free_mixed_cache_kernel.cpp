#include <cmath>
#include <cstdint>
#include <vector>

static inline float sigmoid_f(double x) {
    return static_cast<float>(1.0 / (1.0 + std::exp(-x)));
}

static inline double gumbel_from_uniform(float raw) {
    const double u = std::fmin(1.0 - 1.0e-6, std::fmax(1.0e-6, static_cast<double>(raw)));
    return -std::log(-std::log(u));
}

extern "C" int cxxgen0_object_free_mixed_cache_execute_v1(
    const int32_t* slot_tokens,
    const uint8_t* is_gru,
    const float* obs,
    const float* aggr,
    const float* temps,
    const float* drives,
    const float* gain,
    const float* mlp_pack,
    const float* gru_pack,
    const int32_t* slots,
    const int64_t* mem_dims,
    const float* hmem,
    const int32_t* cache_tokens,
    const int32_t* cache_cursors,
    const int32_t* cache_actions,
    const float* mlp_uniforms,
    const float* gru_uniforms,
    int64_t agents,
    int64_t obs_cols,
    int64_t hidden,
    int64_t gru_hidden,
    int64_t actions,
    int64_t stride,
    int64_t hmem_rows,
    int32_t deterministic_actions,
    float w_hunger_gather,
    float w_pain_rest,
    float w_fatigue_rest,
    float w_pain_attack,
    float w_fear_attack,
    int32_t* out_actions,
    int32_t* out_cache_tokens,
    int32_t* out_cache_cursors,
    int32_t* out_cache_actions,
    float* out_hmem,
    float* out_logits
) {
    if (!slot_tokens || !is_gru || !obs || !aggr || !temps || !drives || !gain || !mlp_pack || !gru_pack ||
        !slots || !mem_dims || !hmem || !cache_tokens || !cache_cursors || !cache_actions || !mlp_uniforms ||
        !gru_uniforms || !out_actions || !out_cache_tokens || !out_cache_cursors || !out_cache_actions ||
        !out_hmem || !out_logits) {
        return 1;
    }
    if (agents < 0 || obs_cols < 0 || hidden <= 0 || gru_hidden <= 0 || actions <= 0 || stride <= 0 || hmem_rows < 0) {
        return 2;
    }
    if (actions < 7) {
        return 3;
    }

    const int64_t mlp_width = hidden * obs_cols + hidden + hidden * hidden + hidden + actions * hidden + actions;
    const int64_t gru_width =
        gru_hidden * obs_cols + gru_hidden +
        3 * gru_hidden * gru_hidden + 3 * gru_hidden +
        3 * gru_hidden * gru_hidden + 3 * gru_hidden +
        actions * gru_hidden + actions;

    std::vector<float> h1(static_cast<size_t>(hidden), 0.0f);
    std::vector<float> h2(static_cast<size_t>(hidden), 0.0f);
    std::vector<float> z_in(static_cast<size_t>(gru_hidden), 0.0f);
    std::vector<float> i_proj(static_cast<size_t>(3 * gru_hidden), 0.0f);
    std::vector<float> h_proj(static_cast<size_t>(3 * gru_hidden), 0.0f);
    std::vector<float> h_next(static_cast<size_t>(gru_hidden), 0.0f);
    std::vector<float> logits(static_cast<size_t>(actions), 0.0f);

    for (int64_t row = 0; row < hmem_rows; ++row) {
        for (int64_t h = 0; h < gru_hidden; ++h) {
            out_hmem[row * gru_hidden + h] = hmem[row * gru_hidden + h];
        }
    }

    for (int64_t i = 0; i < agents; ++i) {
        for (int64_t k = 0; k < stride; ++k) {
            out_cache_actions[i * stride + k] = cache_actions[i * stride + k];
        }
        out_cache_tokens[i] = cache_tokens[i];
        out_cache_cursors[i] = cache_cursors[i];
        for (int64_t a = 0; a < actions; ++a) {
            out_logits[i * actions + a] = 0.0f;
        }

        const bool cache_hit =
            cache_tokens[i] == slot_tokens[i] && cache_cursors[i] >= 0 && static_cast<int64_t>(cache_cursors[i]) < stride;
        if (cache_hit) {
            out_actions[i] = cache_actions[i * stride + cache_cursors[i]];
            out_cache_cursors[i] = cache_cursors[i] + 1;
            continue;
        }

        const int32_t slot = slots[i];
        if (slot < 0) {
            return 10;
        }

        if (!is_gru[i]) {
            const float* pack = mlp_pack + static_cast<int64_t>(slot) * mlp_width;
            int64_t off = 0;
            const float* W1 = pack + off;
            off += hidden * obs_cols;
            const float* b1 = pack + off;
            off += hidden;
            const float* W2 = pack + off;
            off += hidden * hidden;
            const float* b2 = pack + off;
            off += hidden;
            const float* W3 = pack + off;
            off += actions * hidden;
            const float* b3 = pack + off;

            for (int64_t h = 0; h < hidden; ++h) {
                double v = static_cast<double>(b1[h]);
                for (int64_t o = 0; o < obs_cols; ++o) {
                    v += static_cast<double>(W1[h * obs_cols + o]) * static_cast<double>(obs[i * obs_cols + o]);
                }
                h1[static_cast<size_t>(h)] = static_cast<float>(std::tanh(v));
            }
            for (int64_t h = 0; h < hidden; ++h) {
                double v = static_cast<double>(b2[h]);
                for (int64_t p = 0; p < hidden; ++p) {
                    v += static_cast<double>(W2[h * hidden + p]) * static_cast<double>(h1[static_cast<size_t>(p)]);
                }
                h2[static_cast<size_t>(h)] = static_cast<float>(std::tanh(v));
            }
            for (int64_t a = 0; a < actions; ++a) {
                double v = static_cast<double>(b3[a]);
                for (int64_t h = 0; h < hidden; ++h) {
                    v += static_cast<double>(W3[a * hidden + h]) * static_cast<double>(h2[static_cast<size_t>(h)]);
                }
                logits[static_cast<size_t>(a)] = static_cast<float>(v);
            }
            logits[6] += aggr[i];
            const float temp = temps[i] < 0.05f ? 0.05f : temps[i];
            for (int64_t a = 0; a < actions; ++a) {
                logits[static_cast<size_t>(a)] = logits[static_cast<size_t>(a)] / temp;
            }
            const float g = gain[i] < 0.0f ? 0.0f : gain[i];
            logits[4] += g * (w_hunger_gather * drives[i * 4 + 0]);
            logits[5] += g * (w_pain_rest * drives[i * 4 + 1] + w_fatigue_rest * drives[i * 4 + 2]);
            logits[6] -= g * (w_pain_attack * drives[i * 4 + 1] + w_fear_attack * drives[i * 4 + 3]);

            for (int64_t a = 0; a < actions; ++a) {
                out_logits[i * actions + a] = logits[static_cast<size_t>(a)];
            }
            for (int64_t k = 0; k < stride; ++k) {
                int32_t best = 0;
                double best_value = static_cast<double>(logits[0]);
                if (!deterministic_actions) {
                    best_value += gumbel_from_uniform(mlp_uniforms[(i * stride + k) * actions + 0]);
                }
                for (int64_t a = 1; a < actions; ++a) {
                    double value = static_cast<double>(logits[static_cast<size_t>(a)]);
                    if (!deterministic_actions) {
                        value += gumbel_from_uniform(mlp_uniforms[(i * stride + k) * actions + a]);
                    }
                    if (value > best_value) {
                        best_value = value;
                        best = static_cast<int32_t>(a);
                    }
                }
                out_cache_actions[i * stride + k] = best;
                if (k == 0) {
                    out_actions[i] = best;
                }
            }
        } else {
            if (static_cast<int64_t>(slot) >= hmem_rows) {
                return 11;
            }
            const float* pack = gru_pack + static_cast<int64_t>(slot) * gru_width;
            int64_t off = 0;
            const float* encW = pack + off;
            off += gru_hidden * obs_cols;
            const float* encb = pack + off;
            off += gru_hidden;
            const float* Wih = pack + off;
            off += 3 * gru_hidden * gru_hidden;
            const float* bih = pack + off;
            off += 3 * gru_hidden;
            const float* Whh = pack + off;
            off += 3 * gru_hidden * gru_hidden;
            const float* bhh = pack + off;
            off += 3 * gru_hidden;
            const float* headW = pack + off;
            off += actions * gru_hidden;
            const float* headb = pack + off;

            for (int64_t h = 0; h < gru_hidden; ++h) {
                double value = static_cast<double>(encb[h]);
                for (int64_t o = 0; o < obs_cols; ++o) {
                    value += static_cast<double>(encW[h * obs_cols + o]) * static_cast<double>(obs[i * obs_cols + o]);
                }
                z_in[static_cast<size_t>(h)] = static_cast<float>(std::tanh(value));
            }
            for (int64_t row = 0; row < 3 * gru_hidden; ++row) {
                double iv = static_cast<double>(bih[row]);
                double hv = static_cast<double>(bhh[row]);
                for (int64_t col = 0; col < gru_hidden; ++col) {
                    const float hp = col < mem_dims[i] ? hmem[static_cast<int64_t>(slot) * gru_hidden + col] : 0.0f;
                    iv += static_cast<double>(Wih[row * gru_hidden + col]) * static_cast<double>(z_in[static_cast<size_t>(col)]);
                    hv += static_cast<double>(Whh[row * gru_hidden + col]) * static_cast<double>(hp);
                }
                i_proj[static_cast<size_t>(row)] = static_cast<float>(iv);
                h_proj[static_cast<size_t>(row)] = static_cast<float>(hv);
            }
            for (int64_t h = 0; h < gru_hidden; ++h) {
                const float hp = h < mem_dims[i] ? hmem[static_cast<int64_t>(slot) * gru_hidden + h] : 0.0f;
                const float r = sigmoid_f(static_cast<double>(i_proj[static_cast<size_t>(h)]) + static_cast<double>(h_proj[static_cast<size_t>(h)]));
                const float z = sigmoid_f(static_cast<double>(i_proj[static_cast<size_t>(gru_hidden + h)]) + static_cast<double>(h_proj[static_cast<size_t>(gru_hidden + h)]));
                const float n = static_cast<float>(std::tanh(
                    static_cast<double>(i_proj[static_cast<size_t>(2 * gru_hidden + h)]) +
                    static_cast<double>(r) * static_cast<double>(h_proj[static_cast<size_t>(2 * gru_hidden + h)])
                ));
                const float hn = (1.0f - z) * n + z * hp;
                h_next[static_cast<size_t>(h)] = h < mem_dims[i] ? hn : 0.0f;
                out_hmem[static_cast<int64_t>(slot) * gru_hidden + h] = h_next[static_cast<size_t>(h)];
            }
            for (int64_t a = 0; a < actions; ++a) {
                double value = static_cast<double>(headb[a]);
                for (int64_t h = 0; h < gru_hidden; ++h) {
                    value += static_cast<double>(headW[a * gru_hidden + h]) * static_cast<double>(h_next[static_cast<size_t>(h)]);
                }
                logits[static_cast<size_t>(a)] = static_cast<float>(value);
            }
            logits[6] += aggr[i];
            const float temp = temps[i] < 0.05f ? 0.05f : temps[i];
            for (int64_t a = 0; a < actions; ++a) {
                logits[static_cast<size_t>(a)] = logits[static_cast<size_t>(a)] / temp;
            }
            const float g = gain[i] < 0.0f ? 0.0f : gain[i];
            logits[4] += g * (w_hunger_gather * drives[i * 4 + 0]);
            logits[5] += g * (w_pain_rest * drives[i * 4 + 1] + w_fatigue_rest * drives[i * 4 + 2]);
            logits[6] -= g * (w_pain_attack * drives[i * 4 + 1] + w_fear_attack * drives[i * 4 + 3]);

            int32_t best = 0;
            double best_value = static_cast<double>(logits[0]);
            if (!deterministic_actions) {
                best_value += gumbel_from_uniform(gru_uniforms[i * actions + 0]);
            }
            for (int64_t a = 1; a < actions; ++a) {
                double value = static_cast<double>(logits[static_cast<size_t>(a)]);
                if (!deterministic_actions) {
                    value += gumbel_from_uniform(gru_uniforms[i * actions + a]);
                }
                if (value > best_value) {
                    best_value = value;
                    best = static_cast<int32_t>(a);
                }
            }
            for (int64_t a = 0; a < actions; ++a) {
                out_logits[i * actions + a] = logits[static_cast<size_t>(a)];
            }
            out_actions[i] = best;
            for (int64_t k = 0; k < stride; ++k) {
                out_cache_actions[i * stride + k] = best;
            }
        }
        out_cache_tokens[i] = slot_tokens[i];
        out_cache_cursors[i] = 1;
    }
    return 0;
}
