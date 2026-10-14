#ifndef EIEL_CUDA_ARCHITECTURE_B_ACTIVE_IDENTITY_PAIRS_V1_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_ACTIVE_IDENTITY_PAIRS_V1_CUH_

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define EIEL_ACTIVE_IDENTITY_HOST_DEVICE __host__ __device__
#else
#define EIEL_ACTIVE_IDENTITY_HOST_DEVICE
#endif

namespace eiel::cuda_architecture_b::active_identity_pairs_v1 {

constexpr int64_t kMaximumGridAgentId =
    static_cast<int64_t>(std::numeric_limits<int32_t>::max());

enum ResolveStatusV1 : int32_t {
    kResolveFound = 0,
    kResolveAbsent = 1,
    kResolveInvalid = 2,
};

struct ActiveIdentityPairsV1 {
    const int64_t* active_agent_ids;
    const int32_t* active_agent_slots;
    const int64_t* slot_to_agent_id;
    const bool* alive_mask;
    int64_t active_count;
    int64_t active_capacity;
    int64_t slot_count;
};

struct ResolveResultV1 {
    int64_t slot;
    int32_t status;
};

EIEL_ACTIVE_IDENTITY_HOST_DEVICE constexpr bool agent_id_fits_grid_v1(
    const int64_t agent_id
) {
    return agent_id >= 0 && agent_id <= kMaximumGridAgentId;
}

EIEL_ACTIVE_IDENTITY_HOST_DEVICE constexpr bool birth_id_range_fits_grid_v1(
    const int64_t next_id,
    const int64_t birth_count
) {
    if (birth_count < 0) {
        return false;
    }
    if (birth_count == 0) {
        return next_id >= 0 && next_id <= kMaximumGridAgentId + 1;
    }
    return agent_id_fits_grid_v1(next_id)
        && birth_count - 1 <= kMaximumGridAgentId - next_id;
}

EIEL_ACTIVE_IDENTITY_HOST_DEVICE __forceinline__ bool valid_active_identity_pairs_v1(
    const ActiveIdentityPairsV1& pairs
) {
    if (
        pairs.active_count < 0
        || pairs.active_capacity < pairs.active_count
        || pairs.active_capacity > pairs.slot_count
        || pairs.slot_count < 0
        || (
            pairs.active_count > 0
            && (
                pairs.active_agent_ids == nullptr
                || pairs.active_agent_slots == nullptr
                || pairs.slot_to_agent_id == nullptr
                || pairs.alive_mask == nullptr
            )
        )
    ) {
        return false;
    }

    int64_t previous_id = -1;
    for (int64_t row = 0; row < pairs.active_count; ++row) {
        const int64_t agent_id = pairs.active_agent_ids[row];
        const int64_t slot =
            static_cast<int64_t>(pairs.active_agent_slots[row]);
        if (
            !agent_id_fits_grid_v1(agent_id)
            || (row > 0 && agent_id <= previous_id)
            || slot < 0 || slot >= pairs.slot_count
            || !pairs.alive_mask[slot]
            || pairs.slot_to_agent_id[slot] != agent_id
        ) {
            return false;
        }
        previous_id = agent_id;
    }
    return true;
}

// The active rows are sorted by agent ID. The full bounded pair table is
// validated before binary search so duplicates and corrupt roundtrips cannot
// be mistaken for an ordinary absence.
EIEL_ACTIVE_IDENTITY_HOST_DEVICE __forceinline__ ResolveResultV1 resolve_active_agent_slot_v1(
    const ActiveIdentityPairsV1& pairs,
    const int64_t agent_id
) {
    if (
        !agent_id_fits_grid_v1(agent_id)
        || !valid_active_identity_pairs_v1(pairs)
    ) {
        return {-1, kResolveInvalid};
    }
    int64_t lower = 0;
    int64_t upper = pairs.active_count;
    while (lower < upper) {
        const int64_t middle = lower + (upper - lower) / 2;
        const int64_t middle_id = pairs.active_agent_ids[middle];
        if (middle_id < agent_id) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    if (
        lower >= pairs.active_count
        || pairs.active_agent_ids[lower] != agent_id
    ) {
        return {-1, kResolveAbsent};
    }
    return {
        static_cast<int64_t>(pairs.active_agent_slots[lower]),
        kResolveFound,
    };
}

// Rebuild support for a sorted ID list after turnover. This scan is bounded by
// resident capacity and rejects duplicate slot-to-ID rows.
EIEL_ACTIVE_IDENTITY_HOST_DEVICE __forceinline__ ResolveResultV1
resolve_resident_agent_slot_v1(
    const int64_t* const slot_to_agent_id,
    const bool* const alive_mask,
    const int64_t slot_count,
    const int64_t agent_id
) {
    if (
        slot_to_agent_id == nullptr || alive_mask == nullptr || slot_count < 0
        || !agent_id_fits_grid_v1(agent_id)
    ) {
        return {-1, kResolveInvalid};
    }
    int64_t found_slot = -1;
    for (int64_t slot = 0; slot < slot_count; ++slot) {
        if (!alive_mask[slot] || slot_to_agent_id[slot] != agent_id) {
            continue;
        }
        if (found_slot >= 0) {
            return {-1, kResolveInvalid};
        }
        found_slot = slot;
    }
    return found_slot < 0
        ? ResolveResultV1{-1, kResolveAbsent}
        : ResolveResultV1{found_slot, kResolveFound};
}

}  // namespace eiel::cuda_architecture_b::active_identity_pairs_v1

#undef EIEL_ACTIVE_IDENTITY_HOST_DEVICE

#endif  // EIEL_CUDA_ARCHITECTURE_B_ACTIVE_IDENTITY_PAIRS_V1_CUH_
