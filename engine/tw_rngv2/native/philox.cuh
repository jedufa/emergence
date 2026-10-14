#pragma once

#include <cstdint>

namespace tw_rngv2 {

static constexpr uint32_t PHILOX_M0 = 0xD2511F53u;
static constexpr uint32_t PHILOX_M1 = 0xCD9E8D57u;
static constexpr uint32_t PHILOX_W0 = 0x9E3779B9u;
static constexpr uint32_t PHILOX_W1 = 0xBB67AE85u;

struct PhiloxWords {
    uint32_t w0;
    uint32_t w1;
    uint32_t w2;
    uint32_t w3;
};

#if defined(__CUDACC__)
#define TWRNGV2_HD __host__ __device__
#else
#define TWRNGV2_HD
#endif

TWRNGV2_HD inline void mulhilo32(uint32_t multiplier, uint32_t ctr, uint32_t* lo, uint32_t* hi) {
    const uint64_t product = static_cast<uint64_t>(multiplier) * static_cast<uint64_t>(ctr);
    *lo = static_cast<uint32_t>(product);
    *hi = static_cast<uint32_t>(product >> 32);
}

TWRNGV2_HD inline PhiloxWords philox4x32_rounds(
    uint32_t c0,
    uint32_t c1,
    uint32_t c2,
    uint32_t c3,
    uint32_t k0,
    uint32_t k1,
    int rounds
) {
    PhiloxWords ctr{c0, c1, c2, c3};
    if (rounds <= 0) {
        return ctr;
    }
    for (int round = 0; round < rounds; ++round) {
        if (round > 0) {
            k0 += PHILOX_W0;
            k1 += PHILOX_W1;
        }
        uint32_t lo0 = 0;
        uint32_t hi0 = 0;
        uint32_t lo1 = 0;
        uint32_t hi1 = 0;
        mulhilo32(PHILOX_M0, ctr.w0, &lo0, &hi0);
        mulhilo32(PHILOX_M1, ctr.w2, &lo1, &hi1);
        PhiloxWords out;
        out.w0 = hi1 ^ ctr.w1 ^ k0;
        out.w1 = lo1;
        out.w2 = hi0 ^ ctr.w3 ^ k1;
        out.w3 = lo0;
        ctr = out;
    }
    return ctr;
}

TWRNGV2_HD inline PhiloxWords philox4x32_10(
    uint32_t c0,
    uint32_t c1,
    uint32_t c2,
    uint32_t c3,
    uint32_t k0,
    uint32_t k1
) {
    return philox4x32_rounds(c0, c1, c2, c3, k0, k1, 10);
}

}  // namespace tw_rngv2
