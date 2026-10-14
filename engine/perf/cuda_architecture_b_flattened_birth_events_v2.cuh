#pragma once

#include "cuda_architecture_b_persistent_birth_events.cuh"

#include <cstdint>
#include <limits>

namespace eiel::cuda_architecture_b::flattened_birth_events_v2 {

constexpr int32_t kStatusOk = persistent_birth_events::kStatusOk;
constexpr int32_t kStatusCapacityExceeded =
    persistent_birth_events::kStatusCapacityExceeded;
constexpr int32_t kStatusInvalidSemantics =
    persistent_birth_events::kStatusInvalidSemantics;
static_assert(kStatusOk == 0);
static_assert(kStatusCapacityExceeded == 1);
static_assert(kStatusInvalidSemantics == 2);

// All pointers are offset to one simulation cell. Authoritative identity is
// read-only and describes the state before birth identity commit.
struct FlatOrdinaryBirthEventIdentityStateV2 {
    const int64_t* slot_to_agent_id;
    const int32_t* lineage_founder_id;
    const int32_t* lineage_depth;
    int64_t slot_count;
};

struct FlatOrdinaryBirthDecisionRowsV2 {
    const int64_t* parent_slots;
    const int32_t* reason_codes;
    int64_t count;
    int64_t capacity;
};

// Birth-ordered rows have commits.capacity elements. parent_agent_ids and child lineage are
// staged values that the identity commit will publish. The lineage values must
// match persistent_birth_identity's founder fallback and depth increment.
struct FlatOrdinaryBirthCommitRowsV2 {
    const int64_t* parent_slots;
    const int64_t* parent_agent_ids;
    const int64_t* child_slots;
    const int64_t* child_agent_ids;
    const int32_t* child_lineage_founder_id;
    const int32_t* child_lineage_depth;
    const bool* mutation_occurred;
    const float* mutation_delta_total;
    int64_t planned_count;
    int64_t committed_count;
    int64_t capacity;
};

struct FlatOrdinaryBirthEventCursorV2 {
    int64_t active_count;
    int64_t capacity;
    int64_t sequence_base;
};

// Allocation-free, no-authoritative-write preflight for the flattened B<=2
// transaction. The checks reproduce persistent_birth_events' decision order,
// successful-row matching, lineage validity, mutation row expansion, and
// capacity result while applying the flattened absolute-sequence contract.
//
// The caller owns all pointer/stride validity. On every return this function
// writes only scratch_status and scratch_required_rows. Invalid row semantics
// leave required rows at zero; event-state and capacity failures report the
// exact required row count. The scratch scalars must not alias any input or
// authoritative event state. Mutation deltas intentionally have no numeric
// restriction: the persistent append helper records their float bit patterns
// verbatim when the corresponding mutation flag is true.
__device__ __forceinline__ void
preflight_ordinary_birth_events_external_scratch_v2(
    const FlatOrdinaryBirthEventIdentityStateV2 identity,
    const FlatOrdinaryBirthDecisionRowsV2 decisions,
    const FlatOrdinaryBirthCommitRowsV2 commits,
    const FlatOrdinaryBirthEventCursorV2 event_cursor,
    int32_t& scratch_status,
    int64_t& scratch_required_rows
) {
    scratch_status = kStatusOk;
    scratch_required_rows = 0;

    const int64_t decision_count = decisions.count;
    const int64_t planned_birth_count = commits.planned_count;
    const int64_t committed_birth_count = commits.committed_count;
    if (
        identity.slot_count < 0
        || decisions.capacity < 0
        || commits.capacity < 0
        || commits.capacity > identity.slot_count
        || planned_birth_count < 0
        || planned_birth_count > commits.capacity
        || committed_birth_count < 0
        || committed_birth_count > commits.capacity
        || committed_birth_count != planned_birth_count
        || decision_count < 0
        || decision_count > decisions.capacity
    ) {
        scratch_status = kStatusInvalidSemantics;
        return;
    }

    int64_t success_ordinal = 0;
    int64_t required_rows = 0;
    for (int64_t decision = 0; decision < decision_count; ++decision) {
        const int64_t parent_slot = decisions.parent_slots[decision];
        const int32_t reason = decisions.reason_codes[decision];
        if (
            parent_slot < 0 || parent_slot >= identity.slot_count
            || identity.slot_to_agent_id[parent_slot] < 0
            || identity.lineage_depth[parent_slot] < 0
        ) {
            scratch_status = kStatusInvalidSemantics;
            return;
        }

        if (reason == 0) {
            if (success_ordinal >= committed_birth_count) {
                scratch_status = kStatusInvalidSemantics;
                return;
            }

            const int64_t committed_parent_slot =
                commits.parent_slots[success_ordinal];
            const int64_t committed_child_slot =
                commits.child_slots[success_ordinal];
            const int64_t committed_parent_id =
                commits.parent_agent_ids[success_ordinal];
            const int64_t committed_child_id =
                commits.child_agent_ids[success_ordinal];
            if (
                committed_parent_slot != parent_slot
                || committed_parent_slot < 0
                || committed_parent_slot >= identity.slot_count
                || committed_child_slot < 0
                || committed_child_slot >= identity.slot_count
                || committed_parent_id < 0
                || committed_child_id < 0
                || identity.slot_to_agent_id[committed_parent_slot]
                    != committed_parent_id
            ) {
                scratch_status = kStatusInvalidSemantics;
                return;
            }

            const int32_t parent_founder =
                identity.lineage_founder_id[committed_parent_slot];
            const int32_t parent_depth =
                identity.lineage_depth[committed_parent_slot];
            const int64_t expected_child_founder = parent_founder >= 0
                ? static_cast<int64_t>(parent_founder)
                : committed_parent_id;
            if (
                parent_depth == std::numeric_limits<int32_t>::max()
                || expected_child_founder
                    > std::numeric_limits<int32_t>::max()
                || commits.child_lineage_founder_id[success_ordinal]
                    != static_cast<int32_t>(expected_child_founder)
                || commits.child_lineage_depth[success_ordinal]
                    != parent_depth + 1
            ) {
                scratch_status = kStatusInvalidSemantics;
                return;
            }
            for (int64_t prior = 0; prior < success_ordinal; ++prior) {
                if (
                    commits.child_slots[prior] == committed_child_slot
                    || commits.child_agent_ids[prior] == committed_child_id
                ) {
                    scratch_status = kStatusInvalidSemantics;
                    return;
                }
            }

            const bool mutation_occurred =
                commits.mutation_occurred[success_ordinal];
            if (mutation_occurred) {
                // Validate that the staged row supplying the mutation event is
                // addressable; its value domain is intentionally unrestricted.
                static_cast<void>(
                    commits.mutation_delta_total[success_ordinal]
                );
            }
            required_rows += mutation_occurred ? 2 : 1;
            ++success_ordinal;
            continue;
        }
        if (
            (reason >= 1 && reason <= 3)
            || (reason >= 5 && reason <= 6)
        ) {
            ++required_rows;
            continue;
        }
        if (reason == 4 || (reason >= 7 && reason <= 11)) {
            continue;
        }
        scratch_status = kStatusInvalidSemantics;
        return;
    }

    if (success_ordinal != committed_birth_count) {
        scratch_status = kStatusInvalidSemantics;
        return;
    }

    scratch_required_rows = required_rows;
    constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();
    if (
        event_cursor.capacity < 0
        || event_cursor.active_count < 0
        || event_cursor.active_count > event_cursor.capacity
    ) {
        scratch_status = kStatusCapacityExceeded;
        return;
    }
    if (
        event_cursor.sequence_base < 0
        || event_cursor.sequence_base > kInt64Max - event_cursor.capacity
    ) {
        scratch_status = kStatusInvalidSemantics;
        return;
    }
    if (
        required_rows
            > event_cursor.capacity - event_cursor.active_count
    ) {
        scratch_status = kStatusCapacityExceeded;
        return;
    }
}

}  // namespace eiel::cuda_architecture_b::flattened_birth_events_v2
