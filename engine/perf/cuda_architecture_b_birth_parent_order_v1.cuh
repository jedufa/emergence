#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define EIEL_BIRTH_PARENT_ORDER_HD __host__ __device__ __forceinline__
#else
#define EIEL_BIRTH_PARENT_ORDER_HD inline
#endif

namespace eiel::cuda_architecture_b::birth_parent_order_v1 {

enum class BirthParentOrderModeV1 : std::int32_t {
    ascending_agent_id_v1 = 0,
    authoritative_plan_v1 = 1,
};

EIEL_BIRTH_PARENT_ORDER_HD bool valid_mode_v1(
    const BirthParentOrderModeV1 mode
) noexcept {
    return mode == BirthParentOrderModeV1::ascending_agent_id_v1
        || mode == BirthParentOrderModeV1::authoritative_plan_v1;
}

// Legacy ordinary-birth plans are ordered by ascending parent agent ID.
// Selection systems with their own deterministic order may instead declare
// the plan order authoritative. That mode preserves the upstream row order
// while still rejecting duplicate parent identities.
EIEL_BIRTH_PARENT_ORDER_HD bool valid_parent_ordinal_v1(
    const BirthParentOrderModeV1 mode,
    const std::int64_t birth,
    const std::int64_t parent_id,
    const std::int64_t previous_parent_id,
    const std::int64_t* const planned_parent_slots,
    const std::int64_t* const slot_to_agent_id
) noexcept {
    if (
        !valid_mode_v1(mode) || birth < 0 || parent_id < 0
        || planned_parent_slots == nullptr || slot_to_agent_id == nullptr
    ) {
        return false;
    }
    if (mode == BirthParentOrderModeV1::ascending_agent_id_v1) {
        return birth == 0 || parent_id > previous_parent_id;
    }
    for (std::int64_t prior = 0; prior < birth; ++prior) {
        const std::int64_t prior_slot = planned_parent_slots[prior];
        if (prior_slot < 0 || slot_to_agent_id[prior_slot] == parent_id) {
            return false;
        }
    }
    return true;
}

}  // namespace eiel::cuda_architecture_b::birth_parent_order_v1

#undef EIEL_BIRTH_PARENT_ORDER_HD
