#pragma once

#include "cuda_architecture_b_persistent_birth_identity.cuh"

#include <cmath>
#include <cstdint>

namespace eiel::cuda_architecture_b::flattened_birth_identity_v2 {

using BirthParentOrderModeV1 =
    persistent_birth_identity::parent_order::BirthParentOrderModeV1;

// Every pointer is already offset to the authoritative simulation cell owned
// by the caller. The view is read-only: this preflight never commits identity,
// lineage, energy, grid, or free-queue state.
struct FlatOrdinaryBirthIdentityPreflightState {
    const int64_t* slot_to_agent_id;
    const bool* alive_mask;
    const int32_t* agent_grid;
    const float* energy;
    const float* health;
    // This is the effective commit-time view. If planner cooldown settlement
    // is still staged, pass that read-only planner row here and publish it only
    // after this preflight and the remaining transaction preflights succeed.
    const int32_t* repro_cooldown;
    const float* repro_threshold;
    const double* policy_energy_observation;
    const int64_t* slot_free_queue;
    const int64_t* slot_free_queue_count;
    const int64_t* population_next_id;
};

// Birth-ordered arrays have config.plan_capacity elements. Scalars have one
// element. child_energy is double because the persistent identity owner
// computes parent_energy * parent_energy_share in double before authoritative
// float/double stores. queue_head is the consumed source prefix; queue_count is
// the valid tail count after excluding all staged child slots.
struct FlatOrdinaryBirthIdentityPreflightScratch {
    int64_t* parent_slots;
    int64_t* child_slots;
    int64_t* child_agent_ids;
    double* child_energy;
    int64_t* staged_count;
    int64_t* queue_head;
    int64_t* queue_count;
};

// Allocation-free configuration-sized reservation transcription of
// persistent_birth_identity::commit_ordinary_birth_identity_serial. Pointer
// validity, scalar finiteness/ranges, and physical extents remain launch-boundary
// obligations, matching the persistent helper. On failure no scratch or
// authoritative state is changed. On success only caller-owned scratch changes.
__device__ __forceinline__
persistent_birth_identity::OrdinaryBirthIdentityStatus
preflight_ordinary_birth_identity_external_scratch_v2(
    const FlatOrdinaryBirthIdentityPreflightState state,
    const persistent_birth_identity::FlatOrdinaryBirthPlan plan,
    const FlatOrdinaryBirthIdentityPreflightScratch scratch,
    const persistent_birth_identity::OrdinaryBirthIdentityConfig config
) {
    using namespace persistent_birth_identity;

    const int64_t birth_count = plan.planned_count[0];
    if (
        config.plan_capacity < 0
        || config.plan_capacity > config.slot_count
        || birth_count < 0
        || birth_count > config.plan_capacity
        || birth_count > config.slot_count
    ) {
        return kStatusInvalidBirthPlan;
    }

    const int64_t free_count = state.slot_free_queue_count[0];
    if (free_count < 0 || free_count > config.queue_capacity) {
        return kStatusInvalidFreeQueueCount;
    }

    int64_t next_id = state.population_next_id[0] < 1
        ? 1
        : state.population_next_id[0];
    for (int64_t slot = 0; slot < config.slot_count; ++slot) {
        const int64_t resident_id = state.slot_to_agent_id[slot];
        if (
            resident_id >= 0
            && !active_identity::agent_id_fits_grid_v1(resident_id)
        ) {
            return kStatusAgentIdCapacity;
        }
        if (resident_id >= next_id) {
            next_id = resident_id + 1;
        }
    }
    if (!active_identity::birth_id_range_fits_grid_v1(next_id, birth_count)) {
        return kStatusAgentIdCapacity;
    }

    for (int64_t birth = 0; birth < config.plan_capacity; ++birth) {
        scratch.parent_slots[birth] = -1;
        scratch.child_slots[birth] = -1;
        scratch.child_agent_ids[birth] = -1;
        scratch.child_energy[birth] = 0.0;
    }
    int64_t previous_parent_id = -1;
    int64_t queue_head = 0;

    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = plan.parent_slots[birth];
        if (parent_slot < 0 || parent_slot >= config.slot_count) {
            return kStatusInvalidBirthPlan;
        }

        const int64_t parent_id = state.slot_to_agent_id[parent_slot];
        if (!parent_order::valid_parent_ordinal_v1(
                config.parent_order_mode,
                birth,
                parent_id,
                previous_parent_id,
                plan.parent_slots,
                state.slot_to_agent_id
            )) {
            return kStatusInvalidParentOrder;
        }
        previous_parent_id = parent_id;

        const double parent_energy = ::isfinite(
            state.policy_energy_observation[parent_slot]
        )
            ? state.policy_energy_observation[parent_slot]
            : static_cast<double>(state.energy[parent_slot]);
        const double staged_child_energy =
            parent_energy * config.parent_energy_share;
        if (
            !state.alive_mask[parent_slot]
            || state.health[parent_slot] <= 0.0f
            || (
                state.repro_cooldown[parent_slot] != 0
                && state.repro_cooldown[parent_slot]
                    != config.cooldown_steps
            )
            || parent_energy
                < static_cast<double>(state.repro_threshold[parent_slot])
            || parent_energy - staged_child_energy
                < config.minimum_energy_buffer
        ) {
            return kStatusParentNotViable;
        }

        const int32_t child_y = plan.spawn_positions[birth * 2];
        const int32_t child_x = plan.spawn_positions[birth * 2 + 1];
        if (
            child_y < 0 || child_y >= config.grid_height
            || child_x < 0 || child_x >= config.grid_width
            || state.agent_grid[
                static_cast<int64_t>(child_y) * config.grid_width + child_x
            ] != -1
        ) {
            return kStatusSpawnReservation;
        }
        for (int64_t prior = 0; prior < birth; ++prior) {
            if (
                plan.spawn_positions[prior * 2] == child_y
                && plan.spawn_positions[prior * 2 + 1] == child_x
            ) {
                return kStatusSpawnReservation;
            }
        }

        int64_t child_slot = -1;
        while (queue_head < free_count) {
            const int64_t candidate = state.slot_free_queue[queue_head++];
            if (
                candidate < 0 || candidate >= config.slot_count
                || state.alive_mask[candidate]
                || state.slot_to_agent_id[candidate] >= 0
            ) {
                continue;
            }
            bool already_reserved = false;
            for (int64_t prior = 0; prior < birth; ++prior) {
                already_reserved = already_reserved
                    || scratch.child_slots[prior] == candidate;
            }
            if (already_reserved) {
                continue;
            }
            child_slot = candidate;
            break;
        }
        if (child_slot < 0) {
            return kStatusNoFreeSlot;
        }

        scratch.parent_slots[birth] = parent_slot;
        scratch.child_slots[birth] = child_slot;
        scratch.child_agent_ids[birth] = next_id + birth;
        scratch.child_energy[birth] = staged_child_energy;
    }

    int64_t remaining_queue_count = 0;
    for (
        int64_t queue_index = queue_head;
        queue_index < free_count;
        ++queue_index
    ) {
        const int64_t slot = state.slot_free_queue[queue_index];
        if (
            slot >= 0 && slot < config.slot_count
            && !state.alive_mask[slot]
            && state.slot_to_agent_id[slot] < 0
        ) {
            bool reserved_child = false;
            for (int64_t birth = 0; birth < birth_count; ++birth) {
                reserved_child = reserved_child
                    || scratch.child_slots[birth] == slot;
            }
            if (!reserved_child) {
                ++remaining_queue_count;
            }
        }
    }

    scratch.staged_count[0] = birth_count;
    scratch.queue_head[0] = queue_head;
    scratch.queue_count[0] = remaining_queue_count;
    return kStatusOk;
}

}  // namespace eiel::cuda_architecture_b::flattened_birth_identity_v2
