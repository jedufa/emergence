#include <cstdint>

extern "C" int cxxrewrite3_pack_native_entry(
    const int32_t* agent_ids,
    const int32_t n,
    const int32_t capacity,
    const float* energy_by_id,
    const float* health_by_id,
    const float* injury_by_id,
    const float* scar_by_id,
    const int32_t* pos_y_by_id,
    const int32_t* pos_x_by_id,
    const int32_t* actions_by_id,
    float* out_packed,
    double* out_checksum
) {
    if (!agent_ids || !energy_by_id || !health_by_id || !injury_by_id || !scar_by_id ||
        !pos_y_by_id || !pos_x_by_id || !actions_by_id || !out_packed || !out_checksum) {
        return 1;
    }
    if (n < 0 || capacity <= 0) {
        return 2;
    }

    double checksum = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        const int32_t aid = agent_ids[i];
        if (aid < 0 || aid >= capacity) {
            return 3;
        }
        const int32_t base = i * 7;
        const float e = energy_by_id[aid];
        const float h = health_by_id[aid];
        const float inj = injury_by_id[aid];
        const float scar = scar_by_id[aid];
        const float py = static_cast<float>(pos_y_by_id[aid]);
        const float px = static_cast<float>(pos_x_by_id[aid]);
        const float act = static_cast<float>(actions_by_id[aid]);
        out_packed[base + 0] = e;
        out_packed[base + 1] = h;
        out_packed[base + 2] = inj;
        out_packed[base + 3] = scar;
        out_packed[base + 4] = py;
        out_packed[base + 5] = px;
        out_packed[base + 6] = act;
        checksum += static_cast<double>(e) * 1.0;
        checksum += static_cast<double>(h) * 3.0;
        checksum += static_cast<double>(inj) * 5.0;
        checksum += static_cast<double>(scar) * 7.0;
        checksum += static_cast<double>(py) * 11.0;
        checksum += static_cast<double>(px) * 13.0;
        checksum += static_cast<double>(act) * 17.0;
    }
    *out_checksum = checksum;
    return 0;
}
