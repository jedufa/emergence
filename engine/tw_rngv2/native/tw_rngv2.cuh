#pragma once

#include "philox.cuh"
#include "twdmath.cuh"

#include <cstdint>

namespace tw_rngv2 {

enum Status : int32_t {
    RNG_OK = 0,
    RNG_UNKNOWN_DOMAIN = 1,
    RNG_FIELD_OVERFLOW = 2,
    RNG_ADDRESS_EXHAUSTED = 3,
    RNG_INVALID_N = 4,
    RNG_NONFINITE = 5,
    RNG_DUPLICATE_ADDRESS = 6,
    RNG_TRANSACTION_ABORTED = 7
};

static constexpr uint64_t MASK48 = (1ull << 48) - 1ull;
static constexpr uint32_t MASK16 = 0xFFFFu;
static constexpr uint32_t MASK32 = 0xFFFFFFFFu;
static constexpr int32_t MAX_WORD_ORDINAL = (0xFFFF << 2) | 3;
static constexpr uint64_t NAMESPACE_DEFAULT = 0xCA05117E450A51C0ull;
static constexpr uint32_t SEMANTIC_EPOCH_V1 = 1u;
static constexpr uint32_t DOMAIN_POLICY_BIRTH_DRAW = 0x0370u;
static constexpr uint32_t DOMAIN_LIFECYCLE_TRANSITION = 0x0400u;

TWRNGV2_HD inline bool known_domain(uint32_t domain_id) {
    switch (domain_id) {
        case 0x0100u:
        case 0x0200u:
        case 0x0210u:
        case 0x0300u:
        case 0x0310u:
        case 0x0320u:
        case 0x0330u:
        case 0x0331u:
        case 0x0332u:
        case 0x0340u:
        case 0x0341u:
        case 0x0342u:
        case 0x0360u:
        case 0x0370u:
        case 0x0400u:
        case 0x0500u:
        case 0xF000u:
            return true;
        default:
            return false;
    }
}

TWRNGV2_HD inline int32_t validate_fields(
    uint32_t world_uid,
    uint32_t domain_id,
    uint32_t semantic_epoch,
    uint64_t subject_uid,
    uint64_t absolute_tick,
    uint32_t operation_id,
    uint32_t draw_block
) {
    if (world_uid > MASK32) {
        return RNG_FIELD_OVERFLOW;
    }
    if (domain_id > MASK16 || semantic_epoch > MASK16 || operation_id > MASK16 || draw_block > MASK16) {
        return RNG_FIELD_OVERFLOW;
    }
    if (subject_uid > MASK48 || absolute_tick > MASK48) {
        return RNG_FIELD_OVERFLOW;
    }
    if (!known_domain(domain_id)) {
        return RNG_UNKNOWN_DOMAIN;
    }
    return RNG_OK;
}

TWRNGV2_HD inline void encode_key(
    uint64_t namespace_key,
    uint32_t world_uid,
    uint32_t domain_id,
    uint32_t semantic_epoch,
    uint32_t* k0,
    uint32_t* k1
) {
    const uint64_t packed =
        (static_cast<uint64_t>(world_uid)) |
        (static_cast<uint64_t>(domain_id & MASK16) << 32) |
        (static_cast<uint64_t>(semantic_epoch & MASK16) << 48);
    const uint64_t key = namespace_key ^ packed;
    *k0 = static_cast<uint32_t>(key);
    *k1 = static_cast<uint32_t>(key >> 32);
}

TWRNGV2_HD inline void encode_counter(
    uint64_t subject_uid,
    uint64_t absolute_tick,
    uint32_t operation_id,
    uint32_t draw_block,
    uint32_t* c0,
    uint32_t* c1,
    uint32_t* c2,
    uint32_t* c3
) {
    const uint64_t subject = subject_uid & MASK48;
    const uint64_t tick = absolute_tick & MASK48;
    *c0 = static_cast<uint32_t>(subject);
    *c1 = static_cast<uint32_t>((subject >> 32) & 0xFFFFull) |
          (static_cast<uint32_t>(tick & 0xFFFFull) << 16);
    *c2 = static_cast<uint32_t>(tick >> 16);
    *c3 = (operation_id & MASK16) | ((draw_block & MASK16) << 16);
}

TWRNGV2_HD inline float u32_to_f32_half_open(uint32_t u32) {
    const uint32_t q24 = u32 >> 8;
    return static_cast<float>(q24) * 0x1p-24f;
}

TWRNGV2_HD inline double u64_to_f64_half_open(uint64_t u64) {
    const uint64_t q53 = u64 >> 11;
    return static_cast<double>(q53) * 0x1p-53;
}

TWRNGV2_HD inline double u64_to_f64_open(uint64_t u64) {
    const uint64_t q52 = u64 >> 12;
    return (static_cast<double>(q52) + 0.5) * 0x1p-52;
}

TWRNGV2_HD inline uint32_t word_at(
    uint64_t namespace_key,
    uint32_t world_uid,
    uint32_t domain_id,
    uint32_t semantic_epoch,
    uint64_t subject_uid,
    uint64_t absolute_tick,
    uint32_t operation_id,
    int32_t ordinal
) {
    const int32_t block = ordinal >> 2;
    const int32_t lane = ordinal & 3;
    uint32_t k0 = 0;
    uint32_t k1 = 0;
    uint32_t c0 = 0;
    uint32_t c1 = 0;
    uint32_t c2 = 0;
    uint32_t c3 = 0;
    encode_key(namespace_key, world_uid, domain_id, semantic_epoch, &k0, &k1);
    encode_counter(subject_uid, absolute_tick, operation_id, static_cast<uint32_t>(block), &c0, &c1, &c2, &c3);
    const PhiloxWords words = philox4x32_10(c0, c1, c2, c3, k0, k1);
    if (lane == 0) return words.w0;
    if (lane == 1) return words.w1;
    if (lane == 2) return words.w2;
    return words.w3;
}

TWRNGV2_HD inline void bounded_uint32(
    uint64_t n,
    uint64_t namespace_key,
    uint32_t world_uid,
    uint32_t domain_id,
    uint32_t semantic_epoch,
    uint64_t subject_uid,
    uint64_t absolute_tick,
    uint32_t operation_id,
    uint32_t* value,
    int32_t* accepted_ordinal,
    int32_t* status
) {
    if (n == 1ull) {
        *value = 0;
        *accepted_ordinal = 0;
        *status = RNG_OK;
        return;
    }
    if (n == (1ull << 32)) {
        *value = word_at(
            namespace_key,
            world_uid,
            domain_id,
            semantic_epoch,
            subject_uid,
            absolute_tick,
            operation_id,
            0
        );
        *accepted_ordinal = 0;
        *status = RNG_OK;
        return;
    }
    if (n < 1ull || n > (1ull << 32)) {
        *value = 0;
        *accepted_ordinal = 0;
        *status = RNG_INVALID_N;
        return;
    }
    const uint32_t nn = static_cast<uint32_t>(n);
    const uint32_t threshold = static_cast<uint32_t>(0u - nn) % nn;
    int32_t ordinal = 0;
    while (true) {
        if (ordinal > MAX_WORD_ORDINAL) {
            *value = 0;
            *accepted_ordinal = ordinal;
            *status = RNG_ADDRESS_EXHAUSTED;
            return;
        }
        const uint32_t sample = word_at(
            namespace_key,
            world_uid,
            domain_id,
            semantic_epoch,
            subject_uid,
            absolute_tick,
            operation_id,
            ordinal
        );
        const uint64_t product = static_cast<uint64_t>(sample) * static_cast<uint64_t>(nn);
        const uint32_t low = static_cast<uint32_t>(product);
        if (low < nn && low < threshold) {
            ++ordinal;
            continue;
        }
        *value = static_cast<uint32_t>(product >> 32);
        *accepted_ordinal = ordinal;
        *status = RNG_OK;
        return;
    }
}

TWRNGV2_HD inline void box_muller(
    const PhiloxWords& words,
    uint64_t* z0_bits,
    uint64_t* z1_bits,
    int32_t* status
) {
    const uint64_t u_angle_bits = static_cast<uint64_t>(words.w0) | (static_cast<uint64_t>(words.w1) << 32);
    const uint64_t u_radius_bits = static_cast<uint64_t>(words.w2) | (static_cast<uint64_t>(words.w3) << 32);
    const double u_angle = u64_to_f64_half_open(u_angle_bits);
    const double u_radius = u64_to_f64_open(u_radius_bits);
    if (!(u_angle >= 0.0 && u_angle < 1.0) || !(u_radius > 0.0 && u_radius < 1.0)) {
        *z0_bits = 0;
        *z1_bits = 0;
        *status = RNG_NONFINITE;
        return;
    }
    const double log_r = twdmath_log(u_radius);
    const double radius = twdmath_sqrt(-2.0 * log_r);
    const double theta = twdmath_tau() * u_angle;
    const double z0 = radius * twdmath_cos(theta);
    const double z1 = radius * twdmath_sin(theta);
    *z0_bits = f64_bits(z0);
    *z1_bits = f64_bits(z1);
#if defined(__CUDA_ARCH__)
    const bool finite = isfinite(z0) && isfinite(z1);
#else
    const bool finite = std::isfinite(z0) && std::isfinite(z1);
#endif
    *status = finite ? RNG_OK : RNG_NONFINITE;
}

TWRNGV2_HD inline uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

}  // namespace tw_rngv2
