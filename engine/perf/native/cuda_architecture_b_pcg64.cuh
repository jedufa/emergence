#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace eiel::cuda_architecture_b {

struct Pcg64State {
    uint64_t state_lo;
    uint64_t state_hi;
    uint64_t inc_lo;
    uint64_t inc_hi;
    uint32_t has_uint32;
    uint32_t uinteger;
};

__device__ __forceinline__ uint64_t pcg64_raw64(Pcg64State& rng) {
    constexpr uint64_t multiplier_lo = 4865540595714422341ULL;
    constexpr uint64_t multiplier_hi = 2549297995355413924ULL;

    const uint64_t old_lo = rng.state_lo;
    const uint64_t old_hi = rng.state_hi;
    uint64_t next_lo = old_lo * multiplier_lo;
    uint64_t next_hi = __umul64hi(old_lo, multiplier_lo)
        + old_lo * multiplier_hi
        + old_hi * multiplier_lo;
    const uint64_t before_inc = next_lo;
    next_lo += rng.inc_lo;
    next_hi += rng.inc_hi + static_cast<uint64_t>(next_lo < before_inc);
    rng.state_lo = next_lo;
    rng.state_hi = next_hi;

    const uint64_t xorshifted = next_hi ^ next_lo;
    const uint32_t rotation = static_cast<uint32_t>(next_hi >> 58U);
    if (rotation == 0U) {
        return xorshifted;
    }
    return (xorshifted >> rotation) | (xorshifted << ((64U - rotation) & 63U));
}

__device__ __forceinline__ uint32_t pcg64_raw32(Pcg64State& rng) {
    if (rng.has_uint32 != 0U) {
        rng.has_uint32 = 0U;
        return rng.uinteger;
    }
    const uint64_t raw = pcg64_raw64(rng);
    rng.uinteger = static_cast<uint32_t>(raw >> 32U);
    rng.has_uint32 = 1U;
    return static_cast<uint32_t>(raw);
}

__device__ __forceinline__ double pcg64_double01(Pcg64State& rng) {
    return static_cast<double>(pcg64_raw64(rng) >> 11U) * 0x1.0p-53;
}

__device__ __forceinline__ float pcg64_numpy_float32(Pcg64State& rng) {
    return static_cast<float>(pcg64_raw32(rng) >> 8U) * 0x1.0p-24f;
}

}  // namespace eiel::cuda_architecture_b
