#ifndef EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_TURNOVER_IDENTITY_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_TURNOVER_IDENTITY_CUH_

#include <cuda_runtime.h>

#include <cstdint>

#include "cuda_architecture_b_active_identity_pairs_v1.cuh"

namespace eiel::cuda_architecture_b::persistent_turnover_identity {

namespace active_identity =
    eiel::cuda_architecture_b::active_identity_pairs_v1;

enum TurnoverCullReleaseStatus : int32_t {
    kCullStatusOk = 0,
    kCullStatusPlanFailed = 1,
    kCullStatusInvalidResidentState = 2,
    kCullStatusQueueCapacityExceeded = 3,
    kCullStatusReleaseContradiction = 4,
    kCullStatusReleaseActiveIdentityContradiction = 41,
    kCullStatusReleaseFreeQueueContradiction = 42,
    kCullStatusReleaseFreeQueueDuplicate = 43,
    kCullStatusReleaseInvalidRow = 44,
    kCullStatusReleasePendingContradiction = 45,
    kCullStatusReleaseCulledContradiction = 46,
    kCullStatusReleaseAlreadyFree = 47,
    kCullStatusReleaseDuplicateRow = 48,
    kCullStatusReleaseCulledDeadSlot = 461,
    kCullStatusReleaseCulledSlotAgentMismatch = 462,
    kCullStatusReleaseCulledResolveMissing = 463,
    kCullStatusReleaseCulledResolveSlotMismatch = 464,
    kCullStatusReleaseCulledPositionOutOfBounds = 465,
    kCullStatusReleaseCulledGridMismatch = 466,
};

enum TurnoverRefillIdentityStatus : int32_t {
    kIdentityStatusOk = 0,
    kIdentityStatusInvalidBirthPlan = 10,
    kIdentityStatusInvalidParentOrder = 11,
    kIdentityStatusParentNotViable = 12,
    kIdentityStatusNoFreeSlot = 13,
    kIdentityStatusAgentIdCapacity = 14,
    kIdentityStatusSpawnReservation = 15,
    kIdentityStatusInvalidFreeQueueCount = 16,
    kIdentityStatusParentageRebindFailed = 17,
};

constexpr int32_t kTurnoverPlanStatusOk = 0;

// Every pointer is already offset to the simulation cell owned by the caller.
struct FlatTurnoverCullReleaseState {
    int64_t* slot_to_agent_id;
    int32_t* agent_id_to_slot;
    bool* alive_mask;
    int32_t* pos_y;
    int32_t* pos_x;
    int32_t* agent_grid;
    float* energy;
    float* health;
    double* policy_energy_observation;
    int64_t* action_cache_token;
    int32_t* action_cache_cursor;
    int32_t* action_cache_next;
    int32_t* action_cache_actions;
    int64_t* slot_free_queue;
    int64_t* slot_free_queue_count;
    const int64_t* pending_free_agent_ids;
    const int64_t* pending_free_slots;
    int64_t* pending_free_count;
    const int64_t* active_agent_ids;
    const int32_t* active_agent_slots;
    const int64_t* active_count;
};

// Ranking and cull selection are upstream obligations. This helper consumes
// only the immutable Stage2 cull result and its status.
struct FlatTurnoverCullPlan {
    const int64_t* culled_agent_ids;
    const int64_t* culled_slots;
    const int64_t* culled_count;
    const int32_t* turnover_plan_status;
};

struct FlatTurnoverCullReleaseOutputs {
    int64_t* released_agent_ids;
    int64_t* released_slots;
    int64_t* released_count;
    int64_t* pending_released_count;
    int64_t* culled_released_count;
    int32_t* status;
};

struct TurnoverCullReleaseConfig {
    int64_t slot_count;
    int64_t agent_id_capacity;
    int64_t grid_height;
    int64_t grid_width;
    int64_t action_cache_width;
    int64_t free_queue_capacity;
    int64_t pending_capacity;
    int64_t cull_capacity;
    int64_t qualification_fail_after_culled_rows;
    int32_t qualification_failure_status;
    int64_t active_capacity;
};

__device__ __forceinline__ double canonical_quiet_nan() {
    return __longlong_as_double(
        static_cast<long long>(0x7ff8000000000000ULL)
    );
}

// Exact serial transcription of immutable Stage2
// native/cuda_architecture_b_turnover.cu:533-704. Exactly one device thread
// invokes this allocation-free cull/release for one simulation cell.
// Authoritative resident state is unchanged on failure.
__device__ __forceinline__ void commit_turnover_cull_release_serial(
    const FlatTurnoverCullReleaseState state,
    const FlatTurnoverCullPlan plan,
    const FlatTurnoverCullReleaseOutputs outputs,
    const TurnoverCullReleaseConfig config
) {
    const int64_t release_capacity =
        config.pending_capacity + config.cull_capacity;
    for (int64_t index = 0; index < release_capacity; ++index) {
        outputs.released_agent_ids[index] = -1;
        outputs.released_slots[index] = -1;
    }
    outputs.released_count[0] = 0;
    outputs.pending_released_count[0] = 0;
    outputs.culled_released_count[0] = 0;
    outputs.status[0] = kCullStatusOk;

    if (plan.turnover_plan_status[0] != kTurnoverPlanStatusOk) {
        outputs.status[0] = kCullStatusPlanFailed;
        return;
    }
    const int64_t queue_count = state.slot_free_queue_count[0];
    const int64_t pending_count = state.pending_free_count[0];
    const int64_t cull_rows = plan.culled_count[0];
    if (
        queue_count < 0 || queue_count > config.free_queue_capacity
        || pending_count < 0 || pending_count > config.pending_capacity
        || cull_rows < 0 || cull_rows > config.cull_capacity
    ) {
        outputs.status[0] = kCullStatusInvalidResidentState;
        return;
    }
    const int64_t release_rows = pending_count + cull_rows;
    if (release_rows > config.free_queue_capacity - queue_count) {
        outputs.status[0] = kCullStatusQueueCapacityExceeded;
        return;
    }
    if (state.active_count == nullptr) {
        outputs.status[0] = kCullStatusInvalidResidentState;
        return;
    }
    const active_identity::ActiveIdentityPairsV1 active_pairs{
        state.active_agent_ids,
        state.active_agent_slots,
        state.slot_to_agent_id,
        state.alive_mask,
        state.active_count[0],
        config.active_capacity,
        config.slot_count,
    };
    if (!active_identity::valid_active_identity_pairs_v1(active_pairs)) {
        outputs.status[0] = kCullStatusReleaseActiveIdentityContradiction;
        return;
    }

    for (int64_t queue_index = 0; queue_index < queue_count; ++queue_index) {
        const int64_t slot = state.slot_free_queue[queue_index];
        if (
            slot < 0 || slot >= config.slot_count || state.alive_mask[slot]
            || state.slot_to_agent_id[slot] >= 0
        ) {
            outputs.status[0] = kCullStatusReleaseFreeQueueContradiction;
            return;
        }
        for (int64_t previous = 0; previous < queue_index; ++previous) {
            if (state.slot_free_queue[previous] == slot) {
                outputs.status[0] = kCullStatusReleaseFreeQueueDuplicate;
                return;
            }
        }
    }

    for (int64_t index = 0; index < release_rows; ++index) {
        const bool pending = index < pending_count;
        const int64_t source_index = pending ? index : index - pending_count;
        const int64_t agent_id = pending
            ? state.pending_free_agent_ids[source_index]
            : plan.culled_agent_ids[source_index];
        const int64_t slot = pending
            ? state.pending_free_slots[source_index]
            : plan.culled_slots[source_index];
        if (
            !active_identity::agent_id_fits_grid_v1(agent_id)
            || slot < 0 || slot >= config.slot_count
        ) {
            outputs.status[0] = kCullStatusReleaseInvalidRow;
            return;
        }
        const active_identity::ResolveResultV1 resolved =
            active_identity::resolve_active_agent_slot_v1(
                active_pairs, agent_id
            );
        if (pending) {
            if (
                state.alive_mask[slot] || state.slot_to_agent_id[slot] >= 0
                || resolved.status != active_identity::kResolveAbsent
            ) {
                outputs.status[0] = kCullStatusReleasePendingContradiction;
                return;
            }
        } else {
            const int32_t y = state.pos_y[slot];
            const int32_t x = state.pos_x[slot];
            if (!state.alive_mask[slot]) {
                outputs.status[0] = kCullStatusReleaseCulledDeadSlot;
                return;
            }
            if (state.slot_to_agent_id[slot] != agent_id) {
                outputs.status[0] = kCullStatusReleaseCulledSlotAgentMismatch;
                return;
            }
            if (resolved.status != active_identity::kResolveFound) {
                outputs.status[0] = kCullStatusReleaseCulledResolveMissing;
                return;
            }
            if (resolved.slot != slot) {
                outputs.status[0] =
                    kCullStatusReleaseCulledResolveSlotMismatch;
                return;
            }
            if (
                y < 0 || y >= config.grid_height
                || x < 0 || x >= config.grid_width
            ) {
                outputs.status[0] =
                    kCullStatusReleaseCulledPositionOutOfBounds;
                return;
            }
            if (
                state.agent_grid[
                    static_cast<int64_t>(y) * config.grid_width + x
                ] != static_cast<int32_t>(agent_id)
            ) {
                outputs.status[0] = kCullStatusReleaseCulledGridMismatch;
                return;
            }
        }
        for (int64_t queue_index = 0; queue_index < queue_count; ++queue_index) {
            if (state.slot_free_queue[queue_index] == slot) {
                outputs.status[0] = kCullStatusReleaseAlreadyFree;
                return;
            }
        }
        for (int64_t previous = 0; previous < index; ++previous) {
            if (
                outputs.released_agent_ids[previous] == agent_id
                || outputs.released_slots[previous] == slot
            ) {
                outputs.status[0] = kCullStatusReleaseDuplicateRow;
                return;
            }
        }

        int64_t insert_at = index;
        while (insert_at > 0) {
            const int64_t previous_agent_id =
                outputs.released_agent_ids[insert_at - 1];
            const int64_t previous_slot = outputs.released_slots[insert_at - 1];
            if (
                previous_agent_id < agent_id
                || (previous_agent_id == agent_id && previous_slot < slot)
            ) {
                break;
            }
            outputs.released_agent_ids[insert_at] = previous_agent_id;
            outputs.released_slots[insert_at] = previous_slot;
            --insert_at;
        }
        outputs.released_agent_ids[insert_at] = agent_id;
        outputs.released_slots[insert_at] = slot;
    }

    for (int64_t index = 0; index < cull_rows; ++index) {
        const int64_t agent_id = plan.culled_agent_ids[index];
        const int64_t slot = plan.culled_slots[index];
        const int32_t y = state.pos_y[slot];
        const int32_t x = state.pos_x[slot];
        state.agent_grid[
            static_cast<int64_t>(y) * config.grid_width + x
        ] = -1;
        if (
            state.agent_id_to_slot != nullptr
            && agent_id < config.agent_id_capacity
        ) {
            state.agent_id_to_slot[agent_id] = -1;
        }
        state.alive_mask[slot] = false;
        state.slot_to_agent_id[slot] = -1;
        state.pos_y[slot] = -1;
        state.pos_x[slot] = -1;
        state.energy[slot] = 0.0f;
        state.health[slot] = 0.0f;
        state.policy_energy_observation[slot] = canonical_quiet_nan();
        state.action_cache_token[slot] = -1;
        state.action_cache_cursor[slot] = -1;
        state.action_cache_next[slot] = -1;
        for (
            int64_t column = 0;
            column < config.action_cache_width;
            ++column
        ) {
            state.action_cache_actions[
                slot * config.action_cache_width + column
            ] = -1;
        }
        if (
            config.qualification_fail_after_culled_rows > 0
            && index + 1 == config.qualification_fail_after_culled_rows
        ) {
            outputs.status[0] = config.qualification_failure_status;
            return;
        }
    }
    for (int64_t index = 0; index < release_rows; ++index) {
        const int64_t slot = outputs.released_slots[index];
        state.policy_energy_observation[slot] = canonical_quiet_nan();
        state.slot_free_queue[queue_count + index] = slot;
    }
    state.slot_free_queue_count[0] = queue_count + release_rows;
    state.pending_free_count[0] = 0;
    outputs.released_count[0] = release_rows;
    outputs.pending_released_count[0] = pending_count;
    outputs.culled_released_count[0] = cull_rows;
    outputs.status[0] = kCullStatusOk;
}

struct FlatTurnoverRefillIdentityState {
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
    int64_t* slot_free_queue;
    int64_t* slot_free_queue_count;
    int64_t* population_next_id;
};

// Parent/mate selection and spawn planning remain upstream Stage2 work.
struct FlatTurnoverRefillIdentityPlan {
    const int64_t* parent_slots;
    const int64_t* mate_slots;
    const int32_t* spawn_positions;
    const int64_t* planned_count;
};

struct FlatTurnoverRefillIdentityOutputs {
    int64_t* committed_parent_agent_ids;
    int64_t* committed_mate_agent_ids;
    int64_t* committed_child_agent_ids;
    int64_t* committed_child_slots;
    int64_t* committed_count;
    int64_t* alive_agent_ids_by_order;
    int64_t* alive_count;
    int32_t* status;
};

struct TurnoverRefillIdentityConfig {
    int64_t plan_capacity;
    int64_t slot_count;
    int64_t agent_id_capacity;
    int64_t grid_height;
    int64_t grid_width;
    int64_t queue_capacity;
    double start_energy;
    float start_health;
    int32_t generation;
};

// Exact turnover-only (generation_turnover_refill=true) extraction of immutable
// Stage2 native/cuda_architecture_b_lifecycle.cu:610-908, with the fixed
// turnover wrapper arguments at lines 2059-2106. It commits preplanned rows
// without parent energy or cooldown debit and does not perform genome or
// policy inheritance.
__device__ __forceinline__ void commit_turnover_refill_identity_serial(
    const FlatTurnoverRefillIdentityState state,
    const FlatTurnoverRefillIdentityPlan plan,
    const FlatTurnoverRefillIdentityOutputs outputs,
    const TurnoverRefillIdentityConfig config
) {
    for (int64_t birth = 0; birth < config.plan_capacity; ++birth) {
        outputs.committed_parent_agent_ids[birth] = -1;
        outputs.committed_mate_agent_ids[birth] = -1;
        outputs.committed_child_agent_ids[birth] = -1;
        outputs.committed_child_slots[birth] = -1;
    }
    for (int64_t slot = 0; slot < config.slot_count; ++slot) {
        outputs.alive_agent_ids_by_order[slot] = -1;
    }
    outputs.committed_count[0] = 0;
    outputs.alive_count[0] = 0;
    outputs.status[0] = kIdentityStatusOk;

    const int64_t birth_count = plan.planned_count[0];
    if (
        birth_count < 0 || birth_count > config.plan_capacity
        || birth_count > config.slot_count
    ) {
        outputs.status[0] = kIdentityStatusInvalidBirthPlan;
        return;
    }
    const int64_t free_count = state.slot_free_queue_count[0];
    if (free_count < 0 || free_count > config.queue_capacity) {
        outputs.status[0] = kIdentityStatusInvalidFreeQueueCount;
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
            outputs.status[0] = kIdentityStatusAgentIdCapacity;
            return;
        }
        if (resident_id >= next_id) {
            next_id = resident_id + 1;
        }
    }
    if (!active_identity::birth_id_range_fits_grid_v1(next_id, birth_count)) {
        outputs.status[0] = kIdentityStatusAgentIdCapacity;
        return;
    }

    // The queue head is transaction-local. Failed validation does not consume
    // authoritative queue entries.
    int64_t queue_head = 0;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = plan.parent_slots[birth];
        const int64_t mate_slot = plan.mate_slots[birth];
        if (
            parent_slot < 0 || parent_slot >= config.slot_count
            || mate_slot < 0 || mate_slot >= config.slot_count
        ) {
            outputs.status[0] = kIdentityStatusInvalidBirthPlan;
            return;
        }
        const int64_t parent_id = state.slot_to_agent_id[parent_slot];
        const int64_t mate_id = state.slot_to_agent_id[mate_slot];
        if (
            !active_identity::agent_id_fits_grid_v1(parent_id)
            || !active_identity::agent_id_fits_grid_v1(mate_id)
            || parent_slot == mate_slot
        ) {
            outputs.status[0] = kIdentityStatusInvalidParentOrder;
            return;
        }
        if (
            !state.alive_mask[parent_slot] || state.health[parent_slot] <= 0.0f
            || !state.alive_mask[mate_slot]
        ) {
            outputs.status[0] = kIdentityStatusParentNotViable;
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
            outputs.status[0] = kIdentityStatusSpawnReservation;
            return;
        }
        for (int64_t previous = 0; previous < birth; ++previous) {
            if (
                plan.spawn_positions[previous * 2] == child_y
                && plan.spawn_positions[previous * 2 + 1] == child_x
            ) {
                outputs.status[0] = kIdentityStatusSpawnReservation;
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
            outputs.status[0] = kIdentityStatusNoFreeSlot;
            return;
        }
        outputs.committed_parent_agent_ids[birth] = parent_id;
        outputs.committed_mate_agent_ids[birth] = mate_id;
        outputs.committed_child_agent_ids[birth] = next_id + birth;
        outputs.committed_child_slots[birth] = child_slot;
    }

    // All dynamic checks are complete; authoritative writes begin here.
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = plan.parent_slots[birth];
        const int64_t mate_slot = plan.mate_slots[birth];
        const int64_t parent_id = outputs.committed_parent_agent_ids[birth];
        const int64_t mate_id = outputs.committed_mate_agent_ids[birth];
        const int64_t child_id = outputs.committed_child_agent_ids[birth];
        const int64_t child_slot = outputs.committed_child_slots[birth];
        const int32_t child_y = plan.spawn_positions[birth * 2];
        const int32_t child_x = plan.spawn_positions[birth * 2 + 1];

        // Stage2 turnover increments both parents' offspring counts but does
        // not change parent energy, policy observation, or cooldown.
        state.offspring_count[parent_slot] += 1;
        state.offspring_count[mate_slot] += 1;

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
        state.energy[child_slot] = static_cast<float>(config.start_energy);
        state.policy_energy_observation[child_slot] = config.start_energy;
        state.health[child_slot] = config.start_health;
        state.injury[child_slot] = 0.0f;
        state.scar[child_slot] = 0.0f;
        state.age[child_slot] = 0;
        state.survival_steps[child_slot] = 0;
        state.rest_streak[child_slot] = 0;
        state.repro_cooldown[child_slot] = 0;
        state.repro_threshold[child_slot] = state.repro_threshold[parent_slot];
        state.offspring_count[child_slot] = 0;

        const int32_t parent_founder =
            state.lineage_founder_id[parent_slot] >= 0
            ? state.lineage_founder_id[parent_slot]
            : static_cast<int32_t>(parent_id);
        const int32_t mate_founder = state.lineage_founder_id[mate_slot] >= 0
            ? state.lineage_founder_id[mate_slot]
            : static_cast<int32_t>(mate_id);
        state.lineage_founder_id[child_slot] = parent_founder < mate_founder
            ? parent_founder
            : mate_founder;
        const int32_t parent_depth = state.lineage_depth[parent_slot];
        const int32_t mate_depth = state.lineage_depth[mate_slot];
        state.lineage_depth[child_slot] = (
            parent_depth > mate_depth ? parent_depth : mate_depth
        ) + 1;
        state.lineage_birth_generation[child_slot] = config.generation;
        state.lineage_birth_tick[child_slot] = 0;

        const int32_t parent_patch_lineage =
            state.patch_habitat_lineage_id[parent_slot] != 0
            ? state.patch_habitat_lineage_id[parent_slot]
            : static_cast<int32_t>(parent_id);
        const int32_t mate_patch_lineage =
            state.patch_habitat_lineage_id[mate_slot] != 0
            ? state.patch_habitat_lineage_id[mate_slot]
            : static_cast<int32_t>(mate_id);
        state.patch_habitat_lineage_id[child_slot] =
            parent_patch_lineage < mate_patch_lineage
            ? parent_patch_lineage
            : mate_patch_lineage;
        state.action_cache_token[child_slot] = -1;
        state.action_cache_cursor[child_slot] = -1;
        state.action_cache_next[child_slot] = -1;
    }
    state.population_next_id[0] = next_id + birth_count;
    outputs.committed_count[0] = birth_count;

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
        outputs.alive_agent_ids_by_order[remaining_alive++] =
            selected_agent_id;
        previous_agent_id = selected_agent_id;
        previous_slot = selected_slot;
    }
    outputs.alive_count[0] = remaining_alive;
    outputs.status[0] = kIdentityStatusOk;
}

// The immutable Stage2 transaction performs an allocation-only identity pass
// before its interleaved scalar pair redraw, then commits identity again with
// the selected genetic parents. The flattened controller uses one shadow state
// instead of two tensor clones, so it first commits the provisional allocation
// and then applies this exact parent-dependent correction. Child IDs, slots,
// positions, queue state, and population_next_id are pair-independent.
__device__ __forceinline__ bool rebind_turnover_refill_parentage_serial(
    FlatTurnoverRefillIdentityState state,
    int64_t* provisional_parent_slots,
    int64_t* provisional_mate_slots,
    const int64_t* selected_parent_slots,
    const int64_t* selected_mate_slots,
    const int64_t* child_slots,
    const int64_t* committed_count,
    int64_t* committed_parent_agent_ids,
    int64_t* committed_mate_agent_ids,
    const TurnoverRefillIdentityConfig config
) {
    const int64_t count = committed_count[0];
    if (count < 0 || count > config.plan_capacity || count > config.slot_count) {
        return false;
    }

    for (int64_t birth = 0; birth < count; ++birth) {
        const int64_t provisional_parent = provisional_parent_slots[birth];
        const int64_t provisional_mate = provisional_mate_slots[birth];
        const int64_t selected_parent = selected_parent_slots[birth];
        const int64_t selected_mate = selected_mate_slots[birth];
        const int64_t child_slot = child_slots[birth];
        if (
            provisional_parent < 0 || provisional_parent >= config.slot_count
            || provisional_mate < 0 || provisional_mate >= config.slot_count
            || selected_parent < 0 || selected_parent >= config.slot_count
            || selected_mate < 0 || selected_mate >= config.slot_count
            || selected_parent == selected_mate
            || child_slot < 0 || child_slot >= config.slot_count
            || !state.alive_mask[selected_parent]
            || !state.alive_mask[selected_mate]
            || !state.alive_mask[child_slot]
            || state.slot_to_agent_id[selected_parent] < 0
            || state.slot_to_agent_id[selected_mate] < 0
            || state.slot_to_agent_id[child_slot] < 0
            || state.offspring_count[provisional_parent] <= 0
            || state.offspring_count[provisional_mate] <= 0
        ) {
            return false;
        }
        state.offspring_count[provisional_parent] -= 1;
        state.offspring_count[provisional_mate] -= 1;
    }

    for (int64_t birth = 0; birth < count; ++birth) {
        const int64_t parent_slot = selected_parent_slots[birth];
        const int64_t mate_slot = selected_mate_slots[birth];
        const int64_t child_slot = child_slots[birth];
        const int64_t parent_id = state.slot_to_agent_id[parent_slot];
        const int64_t mate_id = state.slot_to_agent_id[mate_slot];

        state.offspring_count[parent_slot] += 1;
        state.offspring_count[mate_slot] += 1;
        const int32_t parent_founder = state.lineage_founder_id[parent_slot] >= 0
            ? state.lineage_founder_id[parent_slot]
            : static_cast<int32_t>(parent_id);
        const int32_t mate_founder = state.lineage_founder_id[mate_slot] >= 0
            ? state.lineage_founder_id[mate_slot]
            : static_cast<int32_t>(mate_id);
        state.lineage_founder_id[child_slot] = parent_founder < mate_founder
            ? parent_founder
            : mate_founder;
        const int32_t parent_depth = state.lineage_depth[parent_slot];
        const int32_t mate_depth = state.lineage_depth[mate_slot];
        state.lineage_depth[child_slot] = (
            parent_depth > mate_depth ? parent_depth : mate_depth
        ) + 1;
        const int32_t parent_patch_lineage =
            state.patch_habitat_lineage_id[parent_slot] != 0
            ? state.patch_habitat_lineage_id[parent_slot]
            : static_cast<int32_t>(parent_id);
        const int32_t mate_patch_lineage =
            state.patch_habitat_lineage_id[mate_slot] != 0
            ? state.patch_habitat_lineage_id[mate_slot]
            : static_cast<int32_t>(mate_id);
        state.patch_habitat_lineage_id[child_slot] =
            parent_patch_lineage < mate_patch_lineage
            ? parent_patch_lineage
            : mate_patch_lineage;

        provisional_parent_slots[birth] = parent_slot;
        provisional_mate_slots[birth] = mate_slot;
        committed_parent_agent_ids[birth] = parent_id;
        committed_mate_agent_ids[birth] = mate_id;
    }
    return true;
}

}  // namespace eiel::cuda_architecture_b::persistent_turnover_identity

#endif  // EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_TURNOVER_IDENTITY_CUH_
