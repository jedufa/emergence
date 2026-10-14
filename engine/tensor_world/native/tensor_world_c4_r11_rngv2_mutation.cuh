#pragma once

#include "tw_rngv2.cuh"

#include <cstdint>

namespace r4 {
namespace rngv2 {

constexpr uint32_t kBernoulli = 0x0330u;
constexpr uint32_t kNormal = 0x0331u;
constexpr uint32_t kDiscrete = 0x0332u;
constexpr uint64_t kNamespace = 0xCA05117E450A51C0ull;
constexpr uint32_t kEpoch = 1u;

__device__ __forceinline__ double clamp_d(double value, double lo, double hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

__device__ int64_t birth_subject(
    const int64_t* agent_id,
    const int64_t* plan_parent,
    const int64_t* plan_mate,
    int64_t offset,
    int64_t planned,
    int64_t self
) {
    int64_t parent_uids[64];
    int64_t mate_uids[64];
    int64_t occ[64];
    if (planned < 0 || planned > 64) return -1;
    for (int64_t i = 0; i < planned; ++i) {
        const int64_t parent = plan_parent[offset + i];
        const int64_t mate = plan_mate[offset + i];
        parent_uids[i] = agent_id[offset + parent];
        mate_uids[i] = agent_id[offset + mate];
        occ[i] = 0;
        for (int64_t j = 0; j < i; ++j) {
            if (parent_uids[j] == parent_uids[i] && mate_uids[j] == mate_uids[i]) {
                ++occ[i];
            }
        }
    }
    int64_t rank = 0;
    const int64_t sp = parent_uids[self];
    const int64_t sm = mate_uids[self];
    const int64_t so = occ[self];
    for (int64_t i = 0; i < planned; ++i) {
        const int64_t p = parent_uids[i];
        const int64_t m = mate_uids[i];
        const int64_t o = occ[i];
        if (p < sp || (p == sp && m < sm) || (p == sp && m == sm && o < so)) {
            ++rank;
        }
    }
    return rank;
}

__device__ bool mutate_planned_birth_v2(
    uint32_t world_uid,
    uint64_t absolute_tick,
    uint64_t subject_uid,
    const double parent_genes[8],
    const double mate_genes[8],
    double parent_rate,
    double mate_rate,
    double* child_genes,
    double* child_rate
) {
    const double specs[8][3] = {
        {0.10, -1.0, 1.0}, {0.15, -2.0, 2.0}, {2.0, 2.0, 90.0},
        {0.10, 0.25, 2.5}, {0.05, 0.0, 0.999}, {0.01, 0.0, 0.10},
        {0.20, 0.0, 3.0}, {0.03, 0.0, 1.0},
    };
    const double rate = clamp_d(0.5 * (parent_rate + mate_rate), 0.0, 1.0);
    for (int field = 0; field < 8; ++field) {
        const int32_t field_status = tw_rngv2::validate_fields(
            world_uid, kBernoulli, kEpoch, subject_uid, absolute_tick,
            static_cast<uint32_t>(field), 0
        );
        if (field_status != tw_rngv2::RNG_OK) return false;
        uint32_t k0 = 0, k1 = 0, c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        tw_rngv2::encode_key(kNamespace, world_uid, kBernoulli, kEpoch, &k0, &k1);
        tw_rngv2::encode_counter(
            subject_uid, absolute_tick, static_cast<uint32_t>(field), 0, &c0, &c1, &c2, &c3
        );
        const tw_rngv2::PhiloxWords bernoulli = tw_rngv2::philox4x32_10(c0, c1, c2, c3, k0, k1);
        const uint64_t u_bits =
            static_cast<uint64_t>(bernoulli.w0) | (static_cast<uint64_t>(bernoulli.w1) << 32);
        const double mid = 0.5 * (parent_genes[field] + mate_genes[field]);
        double value = mid;
        if (tw_rngv2::u64_to_f64_half_open(u_bits) < rate) {
            tw_rngv2::encode_key(kNamespace, world_uid, kNormal, kEpoch, &k0, &k1);
            tw_rngv2::encode_counter(
                subject_uid, absolute_tick, static_cast<uint32_t>(field), 0, &c0, &c1, &c2, &c3
            );
            const tw_rngv2::PhiloxWords normal = tw_rngv2::philox4x32_10(c0, c1, c2, c3, k0, k1);
            uint64_t z0_bits = 0, z1_bits = 0;
            int32_t nstatus = tw_rngv2::RNG_OK;
            tw_rngv2::box_muller(normal, &z0_bits, &z1_bits, &nstatus);
            if (nstatus != tw_rngv2::RNG_OK) return false;
            value = mid + specs[field][0] * tw_rngv2::f64_from_bits(z0_bits);
        }
        child_genes[field] = clamp_d(value, specs[field][1], specs[field][2]);
        uint32_t bounded = 0;
        int32_t accepted = 0;
        int32_t bstatus = tw_rngv2::RNG_OK;
        tw_rngv2::bounded_uint32(
            7ull, kNamespace, world_uid, kDiscrete, kEpoch,
            subject_uid, absolute_tick, static_cast<uint32_t>(field),
            &bounded, &accepted, &bstatus
        );
        if (bstatus != tw_rngv2::RNG_OK) return false;
    }
    uint32_t k0 = 0, k1 = 0, c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    tw_rngv2::encode_key(kNamespace, world_uid, kBernoulli, kEpoch, &k0, &k1);
    tw_rngv2::encode_counter(subject_uid, absolute_tick, 8u, 0, &c0, &c1, &c2, &c3);
    const tw_rngv2::PhiloxWords rate_words = tw_rngv2::philox4x32_10(c0, c1, c2, c3, k0, k1);
    const uint64_t rate_bits =
        static_cast<uint64_t>(rate_words.w0) | (static_cast<uint64_t>(rate_words.w1) << 32);
    double out_rate = rate;
    if (tw_rngv2::u64_to_f64_half_open(rate_bits) < 0.2) {
        tw_rngv2::encode_key(kNamespace, world_uid, kNormal, kEpoch, &k0, &k1);
        tw_rngv2::encode_counter(subject_uid, absolute_tick, 8u, 0, &c0, &c1, &c2, &c3);
        const tw_rngv2::PhiloxWords normal = tw_rngv2::philox4x32_10(c0, c1, c2, c3, k0, k1);
        uint64_t z0_bits = 0, z1_bits = 0;
        int32_t nstatus = tw_rngv2::RNG_OK;
        tw_rngv2::box_muller(normal, &z0_bits, &z1_bits, &nstatus);
        if (nstatus != tw_rngv2::RNG_OK) return false;
        out_rate = clamp_d(rate + 0.005 * tw_rngv2::f64_from_bits(z0_bits), 0.0, 1.0);
    }
    *child_rate = out_rate;
    return true;
}

}  // namespace rngv2
}  // namespace r4
