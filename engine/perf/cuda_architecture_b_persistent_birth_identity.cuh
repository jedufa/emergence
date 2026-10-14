#ifndef EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_BIRTH_IDENTITY_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_BIRTH_IDENTITY_CUH_

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "cuda_architecture_b_active_identity_pairs_v1.cuh"
#include "cuda_architecture_b_birth_parent_order_v1.cuh"

namespace eiel::cuda_architecture_b::persistent_birth_identity {

namespace parent_order =
    eiel::cuda_architecture_b::birth_parent_order_v1;
namespace active_identity =
    eiel::cuda_architecture_b::active_identity_pairs_v1;
using BirthParentOrderModeV1 = parent_order::BirthParentOrderModeV1;

enum OrdinaryBirthIdentityStatus : int32_t {
    kStatusOk = 0,
    kStatusInvalidBirthPlan = 10,
    kStatusInvalidParentOrder = 11,
    kStatusParentNotViable = 12,
    kStatusNoFreeSlot = 13,
    kStatusAgentIdCapacity = 14,
    kStatusSpawnReservation = 15,
    kStatusInvalidFreeQueueCount = 16,
};

// Every pointer is already offset to the simulation cell owned by the caller.
// Array extents are supplied by OrdinaryBirthIdentityConfig.
struct FlatOrdinaryBirthIdentityState {
    int64_t* slot_to_agent_id;
    int32_t* agent_id_to_slot;
    int64_t* slot_tokens;
    bool* alive_mask;
    int32_t* pos_y;
    int32_t* pos_x;
    int32_t* agent_grid;
    float* energy;
    float* health;
    float* injury;
    float* scar;
    int32_t* age;
    int32_t* repro_cooldown;
    float* repro_threshold;
    int32_t* offspring_count;
    int32_t* lineage_founder_id;
    int32_t* lineage_depth;
    int32_t* lineage_birth_generation;
    int32_t* lineage_birth_tick;
    int32_t* patch_habitat_lineage_id;
    double* policy_energy_observation;
    int32_t* survival_steps;
    int32_t* rest_streak;
    int64_t* action_cache_token;
    int32_t* action_cache_cursor;
    int32_t* action_cache_next;
    // The active queue is slot_free_queue[0:slot_free_queue_count[0]].
    // Stage2 has no persisted head: each transaction starts at zero and a
    // successful commit compacts the unread tail back to a zero-based prefix.
    int64_t* slot_free_queue;
    int64_t* slot_free_queue_count;
    int64_t* population_next_id;
};

struct FlatOrdinaryBirthPlan {
    const int64_t* parent_slots;
    const int32_t* spawn_positions;
    const int64_t* planned_count;
};

// The committed arrays have plan_capacity rows. alive_agent_ids_by_order has
// slot_count rows. Scalar count and status pointers each have one row.
struct FlatOrdinaryBirthIdentityOutputs {
    int64_t* committed_parent_agent_ids;
    int64_t* committed_child_agent_ids;
    int64_t* committed_child_slots;
    int64_t* committed_count;
    int64_t* alive_agent_ids_by_order;
    int32_t* alive_agent_slots_by_order;
    int64_t* alive_count;
    int32_t* status;
};

// Pointer validity, scalar finiteness/ranges, and physical array extents are
// launch-boundary obligations, matching the host checks around the source
// kernel. This device helper validates the kernel's dynamic plan/state gates.
struct OrdinaryBirthIdentityConfig {
    int64_t plan_capacity;
    int64_t slot_count;
    int64_t agent_id_capacity;
    int64_t grid_height;
    int64_t grid_width;
    int64_t queue_capacity;
    double parent_energy_share;
    double minimum_energy_buffer;
    float start_health;
    int32_t cooldown_steps;
    int32_t generation;
    int32_t tick;
    parent_order::BirthParentOrderModeV1 parent_order_mode{
        parent_order::BirthParentOrderModeV1::ascending_agent_id_v1
    };
};

// Exact ordinary (generation_turnover_refill=false) transcription of
// native/cuda_architecture_b_lifecycle.cu's controlled identity transaction.
// Exactly one device thread invokes this allocation-free function for one
// simulation cell. No synchronization is performed.
__device__ __forceinline__ void commit_ordinary_birth_identity_serial(
    const FlatOrdinaryBirthIdentityState state,
    const FlatOrdinaryBirthPlan plan,
    const FlatOrdinaryBirthIdentityOutputs outputs,
    const OrdinaryBirthIdentityConfig config
) {
    for (int64_t birth = 0; birth < config.plan_capacity; ++birth) {
        outputs.committed_parent_agent_ids[birth] = -1;
        outputs.committed_child_agent_ids[birth] = -1;
        outputs.committed_child_slots[birth] = -1;
    }
    for (int64_t slot = 0; slot < config.slot_count; ++slot) {
        outputs.alive_agent_ids_by_order[slot] = -1;
        outputs.alive_agent_slots_by_order[slot] = -1;
    }
    outputs.committed_count[0] = 0;
    outputs.alive_count[0] = 0;
    outputs.status[0] = kStatusOk;

    const int64_t birth_count = plan.planned_count[0];
    if (
        birth_count < 0 || birth_count > config.plan_capacity
        || birth_count > config.slot_count
    ) {
        outputs.status[0] = kStatusInvalidBirthPlan;
        return;
    }
    const int64_t free_count = state.slot_free_queue_count[0];
    if (free_count < 0 || free_count > config.queue_capacity) {
        outputs.status[0] = kStatusInvalidFreeQueueCount;
        return;
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
            outputs.status[0] = kStatusAgentIdCapacity;
            return;
        }
        if (resident_id >= next_id) {
            next_id = resident_id + 1;
        }
    }
    if (!active_identity::birth_id_range_fits_grid_v1(next_id, birth_count)) {
        outputs.status[0] = kStatusAgentIdCapacity;
        return;
    }

    int64_t previous_parent_id = -1;
    // This head is deliberately transaction-local. A failed reservation must
    // not consume queue entries from authoritative state.
    int64_t queue_head = 0;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = plan.parent_slots[birth];
        if (parent_slot < 0 || parent_slot >= config.slot_count) {
            outputs.status[0] = kStatusInvalidBirthPlan;
            return;
        }
        const int64_t parent_id = state.slot_to_agent_id[parent_slot];
        if (
            !active_identity::agent_id_fits_grid_v1(parent_id)
            || (birth > 0 && parent_id <= previous_parent_id)
        ) {
            outputs.status[0] = kStatusInvalidParentOrder;
            return;
        }
        previous_parent_id = parent_id;

        const double parent_energy = isfinite(
            state.policy_energy_observation[parent_slot]
        )
            ? state.policy_energy_observation[parent_slot]
            : static_cast<double>(state.energy[parent_slot]);
        const double parent_share = parent_energy * config.parent_energy_share;
        if (
            !state.alive_mask[parent_slot]
            || state.health[parent_slot] <= 0.0f
            || state.repro_cooldown[parent_slot] != 0
            || parent_energy
                < static_cast<double>(state.repro_threshold[parent_slot])
            || parent_energy - parent_share < config.minimum_energy_buffer
        ) {
            outputs.status[0] = kStatusParentNotViable;
            return;
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
            outputs.status[0] = kStatusSpawnReservation;
            return;
        }
        for (int64_t previous = 0; previous < birth; ++previous) {
            if (
                plan.spawn_positions[previous * 2] == child_y
                && plan.spawn_positions[previous * 2 + 1] == child_x
            ) {
                outputs.status[0] = kStatusSpawnReservation;
                return;
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
            bool already_selected = false;
            for (int64_t previous = 0; previous < birth; ++previous) {
                already_selected = already_selected
                    || outputs.committed_child_slots[previous] == candidate;
            }
            if (!already_selected) {
                child_slot = candidate;
                break;
            }
        }
        if (child_slot < 0) {
            outputs.status[0] = kStatusNoFreeSlot;
            return;
        }
        outputs.committed_parent_agent_ids[birth] = parent_id;
        outputs.committed_child_agent_ids[birth] = next_id + birth;
        outputs.committed_child_slots[birth] = child_slot;
    }

    // Authoritative state remains unchanged until every birth is reserved.
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = plan.parent_slots[birth];
        const int64_t parent_id = outputs.committed_parent_agent_ids[birth];
        const int64_t child_id = outputs.committed_child_agent_ids[birth];
        const int64_t child_slot = outputs.committed_child_slots[birth];
        const int32_t child_y = plan.spawn_positions[birth * 2];
        const int32_t child_x = plan.spawn_positions[birth * 2 + 1];
        const double parent_energy = isfinite(
            state.policy_energy_observation[parent_slot]
        )
            ? state.policy_energy_observation[parent_slot]
            : static_cast<double>(state.energy[parent_slot]);
        const double child_energy = parent_energy * config.parent_energy_share;
        const double parent_energy_after = parent_energy - child_energy;

        state.energy[parent_slot] = static_cast<float>(parent_energy_after);
        state.policy_energy_observation[parent_slot] = parent_energy_after;
        state.repro_cooldown[parent_slot] = config.cooldown_steps;
        state.offspring_count[parent_slot] += 1;

        if (
            state.agent_id_to_slot != nullptr
            && child_id < config.agent_id_capacity
        ) {
            state.agent_id_to_slot[child_id] =
                static_cast<int32_t>(child_slot);
        }
        state.slot_to_agent_id[child_slot] = child_id;
        state.slot_tokens[child_slot] = state.slot_tokens[child_slot] < 0
            ? 1
            : state.slot_tokens[child_slot] + 1;
        state.alive_mask[child_slot] = true;
        state.pos_y[child_slot] = child_y;
        state.pos_x[child_slot] = child_x;
        state.agent_grid[
            static_cast<int64_t>(child_y) * config.grid_width + child_x
        ] = static_cast<int32_t>(child_id);
        state.energy[child_slot] = static_cast<float>(child_energy);
        state.policy_energy_observation[child_slot] = child_energy;
        state.health[child_slot] = config.start_health;
        state.injury[child_slot] = 0.0f;
        state.scar[child_slot] = 0.0f;
        state.age[child_slot] = 0;
        state.survival_steps[child_slot] = 0;
        state.rest_streak[child_slot] = 0;
        state.repro_cooldown[child_slot] = 0;
        state.repro_threshold[child_slot] = state.repro_threshold[parent_slot];
        state.offspring_count[child_slot] = 0;
        const int32_t parent_founder = state.lineage_founder_id[parent_slot] >= 0
            ? state.lineage_founder_id[parent_slot]
            : static_cast<int32_t>(parent_id);
        state.lineage_founder_id[child_slot] = parent_founder;
        state.lineage_depth[child_slot] = state.lineage_depth[parent_slot] + 1;
        state.lineage_birth_generation[child_slot] = config.generation;
        state.lineage_birth_tick[child_slot] = config.tick;
        const int32_t parent_patch_lineage =
            state.patch_habitat_lineage_id[parent_slot] != 0
            ? state.patch_habitat_lineage_id[parent_slot]
            : static_cast<int32_t>(parent_id);
        state.patch_habitat_lineage_id[child_slot] = parent_patch_lineage;
        state.action_cache_token[child_slot] = -1;
        state.action_cache_cursor[child_slot] = -1;
        state.action_cache_next[child_slot] = -1;
    }
    state.population_next_id[0] = next_id + birth_count;
    outputs.committed_count[0] = birth_count;

    // Match Stage2's successful-commit queue normalization: discard the
    // consumed prefix and any now-invalid entries, then reset the implicit
    // head by compacting the remaining entries to index zero.
    int64_t remaining_queue_count = 0;
    for (
        int64_t queue_index = queue_head;
        queue_index < free_count;
        ++queue_index
    ) {
        const int64_t slot = state.slot_free_queue[queue_index];
        if (
            slot >= 0 && slot < config.slot_count
            && !state.alive_mask[slot] && state.slot_to_agent_id[slot] < 0
        ) {
            state.slot_free_queue[remaining_queue_count++] = slot;
        }
    }
    for (
        int64_t queue_index = remaining_queue_count;
        queue_index < free_count;
        ++queue_index
    ) {
        state.slot_free_queue[queue_index] = -1;
    }
    state.slot_free_queue_count[0] = remaining_queue_count;

    int64_t remaining_alive = 0;
    int64_t previous_agent_id = INT64_MIN;
    int64_t previous_slot = -1;
    while (true) {
        int64_t selected_slot = -1;
        int64_t selected_agent_id = INT64_MAX;
        for (int64_t slot = 0; slot < config.slot_count; ++slot) {
            if (!state.alive_mask[slot] || state.slot_to_agent_id[slot] < 0) {
                continue;
            }
            const int64_t agent_id = state.slot_to_agent_id[slot];
            const bool follows_previous = agent_id > previous_agent_id
                || (agent_id == previous_agent_id && slot > previous_slot);
            if (
                follows_previous
                && (
                    agent_id < selected_agent_id
                    || (agent_id == selected_agent_id && slot < selected_slot)
                )
            ) {
                selected_agent_id = agent_id;
                selected_slot = slot;
            }
        }
        if (selected_slot < 0) {
            break;
        }
        outputs.alive_agent_ids_by_order[remaining_alive] = selected_agent_id;
        outputs.alive_agent_slots_by_order[remaining_alive] =
            static_cast<int32_t>(selected_slot);
        ++remaining_alive;
        previous_agent_id = selected_agent_id;
        previous_slot = selected_slot;
    }
    outputs.alive_count[0] = remaining_alive;
    outputs.status[0] = kStatusOk;
}

}  // namespace eiel::cuda_architecture_b::persistent_birth_identity

#endif  // EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_BIRTH_IDENTITY_CUH_
