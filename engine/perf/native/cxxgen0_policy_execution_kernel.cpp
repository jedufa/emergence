#include <cstdint>

extern "C" int cxxgen0_policy_execute_v1(
    const float* observations,
    const float* drives,
    const float* gain,
    const float* hmem,
    const int32_t* action_cache_cursor,
    int64_t agents,
    int64_t obs_cols,
    int64_t drive_cols,
    int64_t hmem_cols,
    int64_t cache_stride,
    int32_t* out_actions,
    float* out_logits,
    float* out_hmem,
    int32_t* out_action_cache_cursor
) {
    if (!observations || !drives || !gain || !hmem || !action_cache_cursor ||
        !out_actions || !out_logits || !out_hmem || !out_action_cache_cursor) {
        return 1;
    }
    if (agents < 0 || obs_cols < 0 || drive_cols < 0 || hmem_cols < 0) {
        return 2;
    }

    for (int64_t i = 0; i < agents; ++i) {
        const float o0 = obs_cols > 0 ? observations[i * obs_cols + 0] : 0.0f;
        const float o1 = obs_cols > 1 ? observations[i * obs_cols + 1] : 0.0f;
        const float o2 = obs_cols > 2 ? observations[i * obs_cols + 2] : 0.0f;
        const float o3 = obs_cols > 3 ? observations[i * obs_cols + 3] : 0.0f;
        const float d0 = drive_cols > 0 ? drives[i * drive_cols + 0] : 0.0f;
        const float d1 = drive_cols > 1 ? drives[i * drive_cols + 1] : 0.0f;
        const float d2 = drive_cols > 2 ? drives[i * drive_cols + 2] : 0.0f;
        const float d3 = drive_cols > 3 ? drives[i * drive_cols + 3] : 0.0f;
        const float h0 = hmem_cols > 0 ? hmem[i * hmem_cols + 0] : 0.0f;
        const float h1 = hmem_cols > 1 ? hmem[i * hmem_cols + 1] : 0.0f;
        const float g = gain[i];
        float logits[7];
        logits[0] = o0 - d2;
        logits[1] = o1 - d2;
        logits[2] = o2 - d1;
        logits[3] = o3 - d1;
        logits[4] = (o0 + o1 + d0) * g;
        logits[5] = d2 + h0;
        logits[6] = d3 + h1;

        int32_t best = 0;
        float best_value = logits[0];
        for (int32_t j = 0; j < 7; ++j) {
            out_logits[i * 7 + j] = logits[j];
            if (logits[j] > best_value) {
                best_value = logits[j];
                best = j;
            }
        }
        out_actions[i] = best;
        for (int64_t j = 0; j < hmem_cols; ++j) {
            double value = 0.75 * static_cast<double>(hmem[i * hmem_cols + j]);
            if (j == 0) {
                value += 0.05 * static_cast<double>(o0) + 0.01 * static_cast<double>(best);
            } else if (j == 1) {
                value += 0.05 * static_cast<double>(best_value);
            }
            out_hmem[i * hmem_cols + j] = static_cast<float>(value);
        }
        if (cache_stride > 0) {
            out_action_cache_cursor[i] = static_cast<int32_t>((static_cast<int64_t>(action_cache_cursor[i]) + 1) % cache_stride);
        } else {
            out_action_cache_cursor[i] = action_cache_cursor[i] + 1;
        }
    }
    return 0;
}
