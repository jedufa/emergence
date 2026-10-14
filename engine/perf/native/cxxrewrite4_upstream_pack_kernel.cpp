#include <cstdint>

extern "C" int cxxrewrite4_pack_policy_heads(
    const int32_t* agent_ids,
    const int32_t n,
    const int32_t capacity,
    const int32_t drive_capacity,
    const float energy_norm_denom,
    const float health_norm_denom,
    const float* energy_by_id,
    const float* health_by_id,
    const float* injury_by_id,
    const float* scar_by_id,
    const int32_t* pos_y_by_id,
    const int32_t* pos_x_by_id,
    const int32_t* action_by_id,
    const int32_t* agent_id_to_slot,
    const float* drive_hunger,
    const float* drive_pain,
    const float* drive_fatigue,
    const float* drive_fear,
    float* out_packed,
    double* out_checksum
) {
    if (!agent_ids || !energy_by_id || !health_by_id || !injury_by_id || !scar_by_id ||
        !pos_y_by_id || !pos_x_by_id || !action_by_id || !agent_id_to_slot ||
        !drive_hunger || !drive_pain || !drive_fatigue || !drive_fear ||
        !out_packed || !out_checksum) {
        return 1;
    }
    if (n < 0 || capacity <= 0 || drive_capacity < 0 || energy_norm_denom == 0.0f || health_norm_denom == 0.0f) {
        return 2;
    }

    double checksum = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        const int32_t aid = agent_ids[i];
        if (aid < 0 || aid >= capacity) {
            return 3;
        }
        const int32_t slot = agent_id_to_slot[aid];
        float dh = 0.0f;
        float dp = 0.0f;
        float df = 0.0f;
        float dfr = 0.0f;
        if (slot >= 0 && slot < drive_capacity) {
            dh = drive_hunger[slot];
            dp = drive_pain[slot];
            df = drive_fatigue[slot];
            dfr = drive_fear[slot];
        }

        const float e = energy_by_id[aid] / energy_norm_denom;
        const float h = health_by_id[aid] / health_norm_denom;
        const float inj = injury_by_id[aid];
        const float scar = scar_by_id[aid];
        const float py = static_cast<float>(pos_y_by_id[aid]);
        const float px = static_cast<float>(pos_x_by_id[aid]);
        const float act = static_cast<float>(action_by_id[aid]);
        const int32_t base = i * 11;

        out_packed[base + 0] = e;
        out_packed[base + 1] = h;
        out_packed[base + 2] = inj;
        out_packed[base + 3] = scar;
        out_packed[base + 4] = py;
        out_packed[base + 5] = px;
        out_packed[base + 6] = dh;
        out_packed[base + 7] = dp;
        out_packed[base + 8] = df;
        out_packed[base + 9] = dfr;
        out_packed[base + 10] = act;

        checksum += static_cast<double>(e) * 1.0;
        checksum += static_cast<double>(h) * 3.0;
        checksum += static_cast<double>(inj) * 5.0;
        checksum += static_cast<double>(scar) * 7.0;
        checksum += static_cast<double>(py) * 11.0;
        checksum += static_cast<double>(px) * 13.0;
        checksum += static_cast<double>(dh) * 17.0;
        checksum += static_cast<double>(dp) * 19.0;
        checksum += static_cast<double>(df) * 23.0;
        checksum += static_cast<double>(dfr) * 29.0;
        checksum += static_cast<double>(act) * 31.0;
    }
    *out_checksum = checksum;
    return 0;
}
