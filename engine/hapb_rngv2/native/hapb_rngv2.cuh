#pragma once

#include "../../tw_rngv2/native/tw_rngv2.cuh"

#include <cstdint>

namespace hapb_rngv2 {

static constexpr uint32_t DOMAIN_POLICY_WINDOW_CROSSOVER = 0x0600u;
static constexpr uint32_t DOMAIN_POLICY_WINDOW_MUTATION = 0x0601u;
static constexpr uint32_t DOMAIN_POLICY_WINDOW_MASK = 0x0602u;

TWRNGV2_HD inline bool known_domain(uint32_t domain_id) {
    if (tw_rngv2::known_domain(domain_id)) {
        return true;
    }
    switch (domain_id) {
        case DOMAIN_POLICY_WINDOW_CROSSOVER:
        case DOMAIN_POLICY_WINDOW_MUTATION:
        case DOMAIN_POLICY_WINDOW_MASK:
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
    if (world_uid > tw_rngv2::MASK32) {
        return tw_rngv2::RNG_FIELD_OVERFLOW;
    }
    if (domain_id > tw_rngv2::MASK16 || semantic_epoch > tw_rngv2::MASK16
        || operation_id > tw_rngv2::MASK16 || draw_block > tw_rngv2::MASK16) {
        return tw_rngv2::RNG_FIELD_OVERFLOW;
    }
    if (subject_uid > tw_rngv2::MASK48 || absolute_tick > tw_rngv2::MASK48) {
        return tw_rngv2::RNG_FIELD_OVERFLOW;
    }
    if (!known_domain(domain_id)) {
        return tw_rngv2::RNG_UNKNOWN_DOMAIN;
    }
    return tw_rngv2::RNG_OK;
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
    return tw_rngv2::word_at(
        namespace_key,
        world_uid,
        domain_id,
        semantic_epoch,
        subject_uid,
        absolute_tick,
        operation_id,
        ordinal
    );
}

}  // namespace hapb_rngv2
