#include "tensor_world_c4_resident_mutation_r4.cu"

namespace r5 {

constexpr int32_t kPeerFailed = 40;
constexpr int32_t kInvalidQueue = 41;
constexpr int32_t kCapacity = 42;
constexpr int32_t kIdentityCapacity = 43;
constexpr int32_t kIdentityOccupancy = 44;
constexpr int32_t kDeathEvent = 4;

struct DeathState {
    bool* candidate;
    int32_t* reason;
    int64_t* pending_ids;
    int64_t* pending_slots;
    int64_t* pending_count;
    int64_t* output_ids;
    int64_t* output_slots;
    int32_t* output_reasons;
    int64_t* output_count;
    int32_t* rest_streak;
    int64_t* cache_token;
    int32_t* cache_cursor;
    int32_t* cache_next;
};

struct TurnStage {
    int64_t* death_ids;
    int64_t* death_slots;
    int32_t* death_reasons;
    int64_t* death_count;
    int64_t* virtual_queue;
    int64_t* virtual_count;
    int32_t* status;
    bool* alive;
    int64_t* agent_id;
    int64_t* pos_y;
    int64_t* pos_x;
    float* energy;
    float* health;
    int64_t* occupancy_agent;
    int64_t* occupancy_slot;
    int32_t* patch_agent;
    int32_t* id_to_slot;
    int32_t* repro_cooldown;
    double* policy_energy;
    int64_t* event_count;
    int32_t* event_status;
    int64_t* replay_event_sequence;
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    r4::PhaseTiming timing;
#endif
};

__device__ __forceinline__ int64_t sorted_identity_slot(
    State state, int64_t world, int64_t agent_id) {
    const int64_t offset = world * state.capacity;
    int64_t low = 0, high = state.active_count[world];
    if (high < 0 || high > state.capacity) return -2;
    while (low < high) {
        const int64_t middle = low + (high - low) / 2;
        if (state.active_agent_ids[offset + middle] < agent_id) low = middle + 1;
        else high = middle;
    }
    if (low >= state.active_count[world]
        || state.active_agent_ids[offset + low] != agent_id) return -1;
    return state.active_agent_slots[offset + low];
}

__device__ __forceinline__ bool death_candidate(
    State state, DeathState death, int64_t offset, int64_t slot) {
    return state.alive[offset + slot] && state.agent_id[offset + slot] >= 0
        && death.candidate[offset + slot];
}

__global__ void prepare_post_death_shadow(
    State live, DeathState death, r4::Replay replay, TurnStage turn, Config config) {
    if (threadIdx.x != 0) return;
    const int64_t world = blockIdx.x;
    if (world >= live.batch) return;
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    r4::phase_stamp(turn.timing, world, 8);
#endif
    const int64_t offset = world * live.capacity;
    const int64_t grid_offset = world * live.grid_size * live.grid_size;
    turn.status[world] = 0;
    turn.death_count[world] = 0;
    turn.virtual_count[world] = 0;
    turn.event_status[world] = live.event_status[world];
    turn.event_count[world] = live.event_active_count[world];
    turn.replay_event_sequence[world] = replay.event_sequence[world];
    for (int64_t row = 0; row < live.capacity; ++row) {
        turn.death_ids[offset + row] = -1;
        turn.death_slots[offset + row] = -1;
        turn.death_reasons[offset + row] = 0;
        turn.virtual_queue[offset + row] = -1;
        turn.alive[offset + row] = live.alive[offset + row];
        turn.agent_id[offset + row] = live.agent_id[offset + row];
        turn.pos_y[offset + row] = live.pos_y[offset + row];
        turn.pos_x[offset + row] = live.pos_x[offset + row];
        turn.energy[offset + row] = live.energy[offset + row];
        turn.health[offset + row] = live.health[offset + row];
        turn.id_to_slot[offset + row] = live.agent_id_to_slot[offset + row];
        turn.repro_cooldown[offset + row] = live.repro_cooldown[offset + row];
        turn.policy_energy[offset + row] = live.policy_energy_observation[offset + row];
    }
    for (int64_t cell = 0; cell < live.grid_size * live.grid_size; ++cell) {
        turn.occupancy_agent[grid_offset + cell] = live.occupancy_agent[grid_offset + cell];
        turn.occupancy_slot[grid_offset + cell] = live.occupancy_slot[grid_offset + cell];
        turn.patch_agent[grid_offset + cell] = live.patch_agent[grid_offset + cell];
    }
    const int64_t pending = death.pending_count[world];
    const int64_t free_count = live.free_count[world];
    const int64_t cursor = live.event_active_count[world];
    if (pending < 0 || pending > live.capacity || free_count < 0
        || free_count > live.capacity || pending + free_count > live.capacity) {
        turn.status[world] = kInvalidQueue;
        return;
    }
    if (replay.event_sequence[world] != cursor || live.event_status[world] != 0
        || cursor < 0 || cursor > live.event_capacity) {
        turn.status[world] = kIdentityOccupancy;
        return;
    }
    int64_t virtual_count = 0;
    for (int64_t row = 0; row < free_count + pending; ++row) {
        const int64_t slot = row < free_count ? live.free_queue[offset + row]
            : death.pending_slots[offset + row - free_count];
        if (slot < 0 || slot >= live.capacity || live.alive[offset + slot]
            || live.agent_id[offset + slot] >= 0) {
            turn.status[world] = kInvalidQueue;
            return;
        }
        for (int64_t prior = 0; prior < row; ++prior) {
            if (turn.virtual_queue[offset + prior] == slot) {
                turn.status[world] = kInvalidQueue;
                return;
            }
        }
        turn.virtual_queue[offset + virtual_count++] = slot;
    }
    int64_t candidates = 0;
    for (int64_t slot = 0; slot < live.capacity; ++slot) {
        candidates += death_candidate(live, death, offset, slot) ? 1 : 0;
    }
    if (virtual_count + candidates > live.capacity
        || candidates > live.event_capacity - cursor) {
        turn.status[world] = kCapacity;
        return;
    }
    int64_t previous_id = INT64_MIN;
    int64_t previous_slot = -1;
    for (int64_t row = 0; row < candidates; ++row) {
        int64_t selected_id = INT64_MAX;
        int64_t selected_slot = -1;
        for (int64_t slot = 0; slot < live.capacity; ++slot) {
            if (!death_candidate(live, death, offset, slot)) continue;
            const int64_t id = live.agent_id[offset + slot];
            const bool follows = id > previous_id || (id == previous_id && slot > previous_slot);
            if (follows && (id < selected_id || (id == selected_id && slot < selected_slot))) {
                selected_id = id;
                selected_slot = slot;
            }
        }
        if (selected_slot < 0 || (config.identity_mode == kIdentityLegacyDense
            && selected_id >= live.capacity)) {
            turn.status[world] = kIdentityCapacity;
            return;
        }
        const int64_t resolved_slot = config.identity_mode == kIdentitySortedMonotonic
            ? sorted_identity_slot(live, world, selected_id)
            : live.agent_id_to_slot[offset + selected_id];
        if (resolved_slot != selected_slot
            || live.lineage_founder[offset + selected_slot] < 0
            || live.lineage_depth[offset + selected_slot] < 0) {
            turn.status[world] = kIdentityOccupancy;
            return;
        }
        for (int64_t prior = 0; prior < virtual_count; ++prior) {
            if (turn.virtual_queue[offset + prior] == selected_slot) {
                turn.status[world] = kInvalidQueue;
                return;
            }
        }
        const int64_t y = live.pos_y[offset + selected_slot];
        const int64_t x = live.pos_x[offset + selected_slot];
        if (y < 0 || y >= live.grid_size || x < 0 || x >= live.grid_size) {
            turn.status[world] = kIdentityOccupancy;
            return;
        }
        const int64_t cell = grid_offset + y * live.grid_size + x;
        if (turn.occupancy_agent[cell] != selected_id
            || turn.occupancy_slot[cell] != selected_slot
            || turn.patch_agent[cell] != static_cast<int32_t>(selected_id)) {
            turn.status[world] = kIdentityOccupancy;
            return;
        }
        turn.death_ids[offset + row] = selected_id;
        turn.death_slots[offset + row] = selected_slot;
        turn.death_reasons[offset + row] = live.health[offset + selected_slot] <= 0.0f ? 2 : 1;
        turn.virtual_queue[offset + virtual_count++] = selected_slot;
        turn.occupancy_agent[cell] = -1;
        turn.occupancy_slot[cell] = -1;
        turn.patch_agent[cell] = -1;
        if (selected_id >= 0 && selected_id < live.capacity)
            turn.id_to_slot[offset + selected_id] = -1;
        turn.alive[offset + selected_slot] = false;
        turn.agent_id[offset + selected_slot] = -1;
        turn.pos_y[offset + selected_slot] = -1;
        turn.pos_x[offset + selected_slot] = -1;
        turn.energy[offset + selected_slot] = 0.0f;
        turn.health[offset + selected_slot] = 0.0f;
        turn.repro_cooldown[offset + selected_slot] = 0;
        turn.policy_energy[offset + selected_slot] = NAN;
        previous_id = selected_id;
        previous_slot = selected_slot;
    }
    turn.death_count[world] = candidates;
    turn.virtual_count[world] = virtual_count;
    turn.event_count[world] = cursor + candidates;
    turn.event_status[world] = 0;
    turn.replay_event_sequence[world] = cursor + candidates;
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    r4::phase_stamp(turn.timing, world, 9);
#endif
}

#ifdef EIEL_C4_PHASE_C_SINGLE_SHADOW
// R11 owns a disposable outer shadow.  This variant validates the same death
// ledger/order/queue contract, but applies the post-death state directly to
// that disposable shadow instead of copying thirteen capacity/grid fields into
// TurnStage.  The event cursor remains staged until commit_deaths_outer so the
// ordered death-before-birth event boundary is unchanged.
__global__ void prepare_post_death_outer(
    State live, DeathState death, r4::Replay replay, TurnStage turn, Config config) {
    if (threadIdx.x != 0) return;
    const int64_t world = blockIdx.x;
    if (world >= live.batch) return;
    const int64_t offset = world * live.capacity;
    const int64_t grid_offset = world * live.grid_size * live.grid_size;
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    r4::phase_stamp(turn.timing, world, 8);
#endif
    turn.status[world] = 0;
    turn.death_count[world] = 0;
    turn.event_status[world] = live.event_status[world];
    turn.event_count[world] = live.event_active_count[world];
    turn.replay_event_sequence[world] = replay.event_sequence[world];
    for (int64_t row = 0; row < live.capacity; ++row) {
        turn.death_ids[offset + row] = -1;
        turn.death_slots[offset + row] = -1;
        turn.death_reasons[offset + row] = 0;
    }
    const int64_t pending = death.pending_count[world];
    const int64_t free_count = live.free_count[world];
    const int64_t cursor = live.event_active_count[world];
    if (pending < 0 || pending > live.capacity || free_count < 0
        || free_count > live.capacity || pending + free_count > live.capacity) {
        turn.status[world] = kInvalidQueue;
        return;
    }
    if (replay.event_sequence[world] != cursor || live.event_status[world] != 0
        || cursor < 0 || cursor > live.event_capacity) {
        turn.status[world] = kIdentityOccupancy;
        return;
    }
    const int64_t existing = free_count + pending;
    for (int64_t row = 0; row < existing; ++row) {
        const int64_t slot = row < free_count ? live.free_queue[offset + row]
            : death.pending_slots[offset + row - free_count];
        if (slot < 0 || slot >= live.capacity || live.alive[offset + slot]
            || live.agent_id[offset + slot] >= 0) {
            turn.status[world] = kInvalidQueue;
            return;
        }
        for (int64_t prior = 0; prior < row; ++prior) {
            const int64_t prior_slot = prior < free_count
                ? live.free_queue[offset + prior]
                : death.pending_slots[offset + prior - free_count];
            if (prior_slot == slot) {
                turn.status[world] = kInvalidQueue;
                return;
            }
        }
    }
    for (int64_t row = 0; row < pending; ++row)
        live.free_queue[offset + free_count + row] = death.pending_slots[offset + row];
    int64_t candidates = 0;
    for (int64_t slot = 0; slot < live.capacity; ++slot)
        candidates += death_candidate(live, death, offset, slot) ? 1 : 0;
    if (existing + candidates > live.capacity
        || candidates > live.event_capacity - cursor) {
        turn.status[world] = kCapacity;
        return;
    }
    int64_t previous_id = INT64_MIN;
    int64_t previous_slot = -1;
    for (int64_t row = 0; row < candidates; ++row) {
        int64_t selected_id = INT64_MAX;
        int64_t selected_slot = -1;
        for (int64_t slot = 0; slot < live.capacity; ++slot) {
            if (!death_candidate(live, death, offset, slot)) continue;
            const int64_t id = live.agent_id[offset + slot];
            const bool follows = id > previous_id
                || (id == previous_id && slot > previous_slot);
            if (follows && (id < selected_id
                || (id == selected_id && slot < selected_slot))) {
                selected_id = id;
                selected_slot = slot;
            }
        }
        if (selected_slot < 0 || (config.identity_mode == kIdentityLegacyDense
            && selected_id >= live.capacity)) {
            turn.status[world] = kIdentityCapacity;
            return;
        }
        const int64_t resolved_slot = config.identity_mode == kIdentitySortedMonotonic
            ? sorted_identity_slot(live, world, selected_id)
            : live.agent_id_to_slot[offset + selected_id];
        if (resolved_slot != selected_slot
            || live.lineage_founder[offset + selected_slot] < 0
            || live.lineage_depth[offset + selected_slot] < 0) {
            turn.status[world] = kIdentityOccupancy;
            return;
        }
        for (int64_t prior = 0; prior < existing + row; ++prior) {
            const int64_t prior_slot = live.free_queue[offset + prior];
            if (prior_slot == selected_slot) {
                turn.status[world] = kInvalidQueue;
                return;
            }
        }
        const int64_t y = live.pos_y[offset + selected_slot];
        const int64_t x = live.pos_x[offset + selected_slot];
        if (y < 0 || y >= live.grid_size || x < 0 || x >= live.grid_size) {
            turn.status[world] = kIdentityOccupancy;
            return;
        }
        const int64_t cell = grid_offset + y * live.grid_size + x;
        if (live.occupancy_agent[cell] != selected_id
            || live.occupancy_slot[cell] != selected_slot
            || live.patch_agent[cell] != static_cast<int32_t>(selected_id)) {
            turn.status[world] = kIdentityOccupancy;
            return;
        }
        turn.death_ids[offset + row] = selected_id;
        turn.death_slots[offset + row] = selected_slot;
        turn.death_reasons[offset + row] = live.health[offset + selected_slot] <= 0.0f ? 2 : 1;
        previous_id = selected_id;
        previous_slot = selected_slot;
    }
    int64_t virtual_count = existing;
    for (int64_t row = 0; row < candidates; ++row) {
        const int64_t slot = turn.death_slots[offset + row];
        const int64_t id = turn.death_ids[offset + row];
        const int64_t y = live.pos_y[offset + slot];
        const int64_t x = live.pos_x[offset + slot];
        const int64_t cell = grid_offset + y * live.grid_size + x;
        live.free_queue[offset + virtual_count++] = slot;
        live.occupancy_agent[cell] = -1;
        live.occupancy_slot[cell] = -1;
        live.patch_agent[cell] = -1;
        if (id >= 0 && id < live.capacity)
            live.agent_id_to_slot[offset + id] = -1;
        live.alive[offset + slot] = false;
        live.agent_id[offset + slot] = -1;
        live.pos_y[offset + slot] = -1;
        live.pos_x[offset + slot] = -1;
        live.energy[offset + slot] = 0.0f;
        live.health[offset + slot] = 0.0f;
        live.policy_energy_observation[offset + slot] = NAN;
        live.repro_cooldown[offset + slot] = 0;
        live.planned_repro_cooldown[offset + slot] = 0;
    }
    for (int64_t row = virtual_count; row < live.capacity; ++row)
        live.free_queue[offset + row] = -1;
    live.free_count[world] = virtual_count;
    turn.death_count[world] = candidates;
    turn.virtual_count[world] = virtual_count;
    turn.event_count[world] = cursor + candidates;
    turn.event_status[world] = 0;
    turn.replay_event_sequence[world] = cursor + candidates;
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    r4::phase_stamp(turn.timing, world, 9);
#endif
}
#endif

__global__ void merge_all_status(
    r4::R4Stage stage, TurnStage turn, State live, Config config) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int64_t world = 0; world < live.batch; ++world) {
        const int64_t planned = stage.base.planned_count[world];
        const int64_t next_id = live.next_agent_id[world];
        if (config.identity_mode == kIdentityLegacyDense
            && stage.base.status[world] == r4::kCapacity
            && (next_id < 0 || planned > live.capacity - next_id)) {
            turn.status[world] = kIdentityCapacity;
        }
    }
    bool failed = false;
    for (int64_t world = 0; world < live.batch; ++world) {
        failed |= turn.status[world] != 0 || stage.base.status[world] != 0;
    }
    if (!failed) return;
    for (int64_t world = 0; world < live.batch; ++world) {
        if (turn.status[world] == 0) turn.status[world] = kPeerFailed;
        if (stage.base.status[world] == 0) stage.base.status[world] = kPeerFailed;
    }
}

__device__ void rebuild_active(State state, int64_t world) {
    const int64_t offset = world * state.capacity;
    for (int64_t row = 0; row < state.capacity; ++row) {
        state.active_agent_ids[offset + row] = -1;
        state.active_agent_slots[offset + row] = -1;
        state.survivor_agent_ids[offset + row] = -1;
    }
    int64_t active = 0, previous_id = INT64_MIN, previous_slot = -1;
    while (true) {
        int64_t selected_id = INT64_MAX, selected_slot = -1;
        for (int64_t slot = 0; slot < state.capacity; ++slot) {
            if (!state.alive[offset + slot] || state.agent_id[offset + slot] < 0) continue;
            const int64_t id = state.agent_id[offset + slot];
            const bool follows = id > previous_id || (id == previous_id && slot > previous_slot);
            if (follows && (id < selected_id || (id == selected_id && slot < selected_slot))) {
                selected_id = id;
                selected_slot = slot;
            }
        }
        if (selected_slot < 0) break;
        state.active_agent_ids[offset + active] = selected_id;
        state.active_agent_slots[offset + active] = static_cast<int32_t>(selected_slot);
        state.survivor_agent_ids[offset + active] = selected_id;
        ++active;
        previous_id = selected_id;
        previous_slot = selected_slot;
    }
    state.active_count[world] = active;
}

__global__ void commit_deaths(
    State state, DeathState death, r4::R4Stage stage, TurnStage turn, Config config) {
    if (threadIdx.x != 0) return;
    for (int64_t other = 0; other < state.batch; ++other) {
        if (turn.status[other] != 0 || stage.base.status[other] != 0) return;
    }
    const int64_t world = blockIdx.x;
    if (world >= state.batch) return;
    const int64_t offset = world * state.capacity;
    const int64_t grid_offset = world * state.grid_size * state.grid_size;
    const int64_t event_before = state.event_active_count[world];
    const int64_t deaths = turn.death_count[world];
    Config event_config = config;
    // Historical R9/R10 preserve the frozen death-generation projection when
    // no device time owner is supplied.  The R11 graph route supplies typed
    // device time and must publish the same nonzero generation/tick on death
    // and birth events.
    if (config.generation_device != nullptr) {
        event_config.generation = config_generation(config, world);
        event_config.tick = config_tick(config, world);
    } else {
        event_config.generation = 0;
    }
    for (int64_t row = 0; row < deaths; ++row) {
        const int64_t slot = turn.death_slots[offset + row];
        const int64_t id = turn.death_ids[offset + row];
        set_event(state, world, event_before + row, kDeathEvent, id, -1, slot, -1,
            -1, -1, state.lineage_founder[offset + slot], state.lineage_depth[offset + slot],
            turn.death_reasons[offset + row], 1,
            static_cast<float>(turn.death_reasons[offset + row]), event_config);
    }
    const int64_t old_pending = death.pending_count[world];
    for (int64_t row = 0; row < state.capacity; ++row) {
        death.output_ids[offset + row] = -1;
        death.output_slots[offset + row] = -1;
        death.output_reasons[offset + row] = 0;
    }
    for (int64_t row = 0; row < turn.virtual_count[world]; ++row) {
        state.free_queue[offset + row] = turn.virtual_queue[offset + row];
    }
    for (int64_t row = turn.virtual_count[world]; row < state.capacity; ++row) {
        state.free_queue[offset + row] = -1;
    }
    state.free_count[world] = turn.virtual_count[world];
    for (int64_t row = 0; row < deaths; ++row) {
        const int64_t slot = turn.death_slots[offset + row];
        const int64_t id = turn.death_ids[offset + row];
        death.pending_ids[offset + old_pending + row] = id;
        death.pending_slots[offset + old_pending + row] = slot;
        death.output_ids[offset + row] = id;
        death.output_slots[offset + row] = slot;
        death.output_reasons[offset + row] = turn.death_reasons[offset + row];
        const int64_t y = state.pos_y[offset + slot];
        const int64_t x = state.pos_x[offset + slot];
        const int64_t cell = grid_offset + y * state.grid_size + x;
        state.occupancy_agent[cell] = -1;
        state.occupancy_slot[cell] = -1;
        state.patch_agent[cell] = -1;
        if (id >= 0 && id < state.capacity)
            state.agent_id_to_slot[offset + id] = -1;
        state.alive[offset + slot] = false;
        state.agent_id[offset + slot] = -1;
        state.pos_y[offset + slot] = -1;
        state.pos_x[offset + slot] = -1;
        state.energy[offset + slot] = 0.0f;
        state.health[offset + slot] = 0.0f;
        state.policy_energy_observation[offset + slot] = NAN;
        state.repro_cooldown[offset + slot] = 0;
        state.planned_repro_cooldown[offset + slot] = 0;
        death.rest_streak[offset + slot] = 0;
        death.cache_token[offset + slot] = -1;
        death.cache_cursor[offset + slot] = -1;
        death.cache_next[offset + slot] = -1;
        death.candidate[offset + slot] = false;
        death.reason[offset + slot] = 0;
    }
    death.output_count[world] = deaths;
    death.pending_count[world] = 0;
#ifndef EIEL_C4_R11_PARALLEL_ACTIVE_REBUILD
    rebuild_active(state, world);
#endif
    state.event_active_count[world] = event_before + deaths;
    state.event_status[world] = 0;
}

#ifdef EIEL_C4_PHASE_C_SINGLE_SHADOW
// Death state has already been applied to the disposable outer shadow by
// prepare_post_death_outer.  This commit retains only the ordered event,
// pending/output, and cursor publication that must occur immediately before
// the birth commit; no canonical/shared surface is touched on a failed batch.
__global__ void commit_deaths_outer(
    State state, DeathState death, r4::R4Stage stage, TurnStage turn, Config config) {
    if (threadIdx.x != 0) return;
    for (int64_t other = 0; other < state.batch; ++other) {
        if (turn.status[other] != 0 || stage.base.status[other] != 0) return;
    }
    const int64_t world = blockIdx.x;
    if (world >= state.batch) return;
    const int64_t offset = world * state.capacity;
    const int64_t event_before = turn.event_count[world] - turn.death_count[world];
    const int64_t deaths = turn.death_count[world];
    Config event_config = config;
    if (config.generation_device != nullptr) {
        event_config.generation = config_generation(config, world);
        event_config.tick = config_tick(config, world);
    } else {
        event_config.generation = 0;
    }
    for (int64_t row = 0; row < deaths; ++row) {
        const int64_t slot = turn.death_slots[offset + row];
        const int64_t id = turn.death_ids[offset + row];
        set_event(state, world, event_before + row, kDeathEvent, id, -1, slot, -1,
            -1, -1, state.lineage_founder[offset + slot], state.lineage_depth[offset + slot],
            turn.death_reasons[offset + row], 1,
            static_cast<float>(turn.death_reasons[offset + row]), event_config);
    }
    const int64_t old_pending = death.pending_count[world];
    for (int64_t row = 0; row < state.capacity; ++row) {
        death.output_ids[offset + row] = -1;
        death.output_slots[offset + row] = -1;
        death.output_reasons[offset + row] = 0;
    }
    for (int64_t row = 0; row < deaths; ++row) {
        const int64_t slot = turn.death_slots[offset + row];
        const int64_t id = turn.death_ids[offset + row];
        death.pending_ids[offset + old_pending + row] = id;
        death.pending_slots[offset + old_pending + row] = slot;
        death.output_ids[offset + row] = id;
        death.output_slots[offset + row] = slot;
        death.output_reasons[offset + row] = turn.death_reasons[offset + row];
        death.rest_streak[offset + slot] = 0;
        death.cache_token[offset + slot] = -1;
        death.cache_cursor[offset + slot] = -1;
        death.cache_next[offset + slot] = -1;
        death.candidate[offset + slot] = false;
        death.reason[offset + slot] = 0;
    }
    death.output_count[world] = deaths;
    death.pending_count[world] = 0;
    state.event_active_count[world] = turn.event_count[world];
    state.event_status[world] = 0;
}
#endif

TurnStage make_turn(const std::vector<torch::Tensor>& t) {
    return TurnStage{
        t[0].data_ptr<int64_t>(), t[1].data_ptr<int64_t>(), t[2].data_ptr<int32_t>(),
        t[3].data_ptr<int64_t>(), t[4].data_ptr<int64_t>(), t[5].data_ptr<int64_t>(),
        t[6].data_ptr<int32_t>(), t[7].data_ptr<bool>(), t[8].data_ptr<int64_t>(),
        t[9].data_ptr<int64_t>(), t[10].data_ptr<int64_t>(), t[11].data_ptr<float>(),
        t[12].data_ptr<float>(), t[13].data_ptr<int64_t>(), t[14].data_ptr<int64_t>(),
        t[15].data_ptr<int32_t>(), t[16].data_ptr<int32_t>(), t[17].data_ptr<int32_t>(),
        t[18].data_ptr<double>(), t[19].data_ptr<int64_t>(), t[20].data_ptr<int32_t>(),
        t[21].data_ptr<int64_t>()
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
        , r4::PhaseTiming{nullptr}
#endif
    };
}

void exact(
    const torch::Tensor& tensor, const torch::Device& device,
    torch::ScalarType dtype, const std::vector<int64_t>& shape, const char* group) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == device && tensor.is_contiguous(),
        "C4 R5 ", group, " tensors must be contiguous on one CUDA device");
    TORCH_CHECK(tensor.scalar_type() == dtype, "C4 R5 ", group, " dtype mismatch");
    TORCH_CHECK(tensor.sizes().vec() == shape, "C4 R5 ", group, " shape mismatch");
}

void validate_abi(
    const std::vector<torch::Tensor>& p, const std::vector<torch::Tensor>& death,
    const std::vector<torch::Tensor>& inherit, const std::vector<torch::Tensor>& zero,
    const std::vector<torch::Tensor>& minus_one, const std::vector<torch::Tensor>& half,
    const std::vector<torch::Tensor>& one, const std::vector<torch::Tensor>& scalar_rng,
    const std::vector<torch::Tensor>& replay, const std::vector<torch::Tensor>& stage,
    const std::vector<torch::Tensor>& turn, const torch::Device& device,
    int64_t batch, int64_t capacity, int64_t grid_size, int64_t event_capacity,
    int64_t streams, int64_t tape_capacity) {
    const std::vector<int64_t> slots{batch, capacity}, batch_shape{batch};
    const std::vector<int64_t> grid{batch, grid_size, grid_size};
    const std::vector<int64_t> events{batch, event_capacity};
    const torch::ScalarType p_types[101] = {
        torch::kBool, torch::kInt64, torch::kInt64, torch::kInt64, torch::kFloat32,
        torch::kFloat32, torch::kInt64, torch::kInt64, torch::kFloat64,
        torch::kInt64, torch::kInt64, torch::kInt32, torch::kFloat32,
        torch::kFloat32, torch::kFloat32, torch::kFloat32, torch::kInt32,
        torch::kInt64, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kInt32, torch::kInt32, torch::kInt32, torch::kInt32, torch::kFloat32,
        torch::kInt32, torch::kInt64, torch::kInt32, torch::kFloat32,
        torch::kInt32, torch::kInt32, torch::kInt32, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kInt32, torch::kInt64, torch::kInt32, torch::kInt64,
        torch::kInt64, torch::kInt32, torch::kInt32, torch::kInt64, torch::kInt32,
        torch::kInt64, torch::kInt64, torch::kInt64, torch::kBool, torch::kFloat32,
        torch::kInt64, torch::kInt64, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kInt32, torch::kInt64, torch::kInt64, torch::kInt32, torch::kInt32,
        torch::kFloat64, torch::kFloat32, torch::kFloat32, torch::kFloat32,
        torch::kFloat32, torch::kFloat32, torch::kFloat32, torch::kFloat32,
        torch::kFloat32, torch::kFloat32, torch::kFloat32, torch::kFloat32,
        torch::kFloat32, torch::kFloat32, torch::kFloat32, torch::kFloat32,
        torch::kInt32, torch::kInt32, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kFloat32, torch::kInt32, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kInt64, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kInt64, torch::kInt64, torch::kInt32, torch::kInt32,
        torch::kInt32, torch::kInt8, torch::kFloat32, torch::kInt64, torch::kInt32};
    std::vector<std::vector<int64_t>> p_shapes(101, slots);
    for (int index : {18, 20, 27, 28, 43, 47, 50, 51, 52, 53, 56, 99, 100})
        p_shapes[index] = batch_shape;
    p_shapes[9] = grid; p_shapes[10] = grid; p_shapes[11] = grid;
    p_shapes[12] = {batch, capacity, 4}; p_shapes[61] = {batch, capacity, 4};
    p_shapes[65] = {batch, capacity, 6}; p_shapes[72] = {batch, capacity, 4};
    for (int index = 77; index <= 98; ++index) p_shapes[index] = events;
    for (int index = 0; index < 101; ++index)
        exact(p[index], device, p_types[index], p_shapes[index], "primary");

    const torch::ScalarType death_types[13] = {
        torch::kBool, torch::kInt32, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kInt64, torch::kInt32, torch::kInt64, torch::kInt32,
        torch::kInt64, torch::kInt32, torch::kInt32};
    for (int index = 0; index < 13; ++index)
        exact(death[index], device, death_types[index],
            index == 4 || index == 8 ? batch_shape : slots, "death");

    const torch::ScalarType inherit_types[29] = {
        torch::kFloat32, torch::kFloat32, torch::kInt32, torch::kInt32, torch::kInt32,
        torch::kInt32, torch::kInt32, torch::kInt32, torch::kInt32, torch::kInt32,
        torch::kInt64, torch::kBool, torch::kFloat32, torch::kFloat32, torch::kFloat32,
        torch::kFloat32, torch::kFloat32, torch::kFloat32, torch::kFloat32,
        torch::kFloat32, torch::kFloat32, torch::kFloat32, torch::kFloat32,
        torch::kFloat32, torch::kFloat32, torch::kFloat32, torch::kFloat32,
        torch::kFloat32, torch::kFloat32};
    for (int index = 0; index < 29; ++index) {
        std::vector<int64_t> shape = slots;
        if (index == 0) shape = {batch, capacity, 9415};
        if (index == 1) shape = {batch, capacity, 9991};
        if (index == 21) shape = {batch, capacity, 4};
        exact(inherit[index], device, inherit_types[index], shape, "inherit");
    }

    const torch::ScalarType zero_types[55] = {
        torch::kBool, torch::kBool, torch::kFloat32, torch::kFloat32, torch::kFloat32,
        torch::kInt32, torch::kFloat32, torch::kFloat32, torch::kFloat32,
        torch::kFloat32, torch::kUInt8, torch::kFloat32, torch::kInt32,
        torch::kFloat32, torch::kFloat32, torch::kFloat32, torch::kFloat32,
        torch::kFloat64, torch::kInt32, torch::kInt32, torch::kInt32, torch::kInt32,
        torch::kInt32, torch::kInt32, torch::kFloat32, torch::kFloat32,
        torch::kFloat32, torch::kFloat32, torch::kFloat32, torch::kFloat32,
        torch::kFloat32, torch::kFloat32, torch::kFloat32, torch::kFloat32,
        torch::kInt32, torch::kInt32, torch::kFloat32, torch::kInt32,
        torch::kFloat32, torch::kInt32, torch::kFloat32, torch::kInt32,
        torch::kBool, torch::kFloat32, torch::kBool, torch::kInt32,
        torch::kFloat32, torch::kFloat32, torch::kFloat32, torch::kFloat32,
        torch::kFloat32, torch::kFloat32, torch::kBool, torch::kFloat32, torch::kBool};
    const int zero_widths[55] = {0,0,0,74,7,0,0,0,0,0,0,0,0,83,7,0,0,0,0,0,
        0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,83,7,2,0,4,0};
    for (int index = 0; index < 55; ++index)
        exact(zero[index], device, zero_types[index], zero_widths[index] == 0 ? slots
            : std::vector<int64_t>{batch, capacity, zero_widths[index]}, "zero reset");
    const torch::ScalarType minus_types[11] = {
        torch::kInt64, torch::kInt32, torch::kInt32, torch::kInt32, torch::kInt32,
        torch::kInt32, torch::kInt64, torch::kInt32, torch::kInt32,
        torch::kInt32, torch::kInt32};
    for (int index = 0; index < 11; ++index)
        exact(minus_one[index], device, minus_types[index], index == 5
            ? std::vector<int64_t>{batch, capacity, 2} : slots, "minus-one reset");
    exact(half[0], device, torch::kFloat64, {batch, capacity, 7}, "half reset");
    exact(one[0], device, torch::kFloat32, slots, "one reset");

    const torch::ScalarType scalar_types[8] = {torch::kUInt32, torch::kInt64,
        torch::kBool, torch::kFloat64, torch::kFloat64, torch::kFloat64,
        torch::kInt64, torch::kInt64};
    for (int index = 0; index < 8; ++index) {
        std::vector<int64_t> shape = batch_shape;
        if (index == 0) shape = {batch, 624};
        if (index == 4 || index == 5) shape = {batch, tape_capacity};
        exact(scalar_rng[index], device, scalar_types[index], shape, "scalar RNG");
    }
    exact(replay[0], device, torch::kInt64, {batch, streams, 4}, "replay");
    exact(replay[1], device, torch::kInt64, {batch, streams, 2}, "replay");
    exact(replay[2], device, torch::kInt64, {batch, streams}, "replay");
    exact(replay[3], device, torch::kInt64, {batch, streams}, "replay");
    exact(replay[4], device, torch::kInt64, batch_shape, "replay");
    exact(replay[5], device, torch::kInt64, slots, "replay");

    const torch::ScalarType stage_types[49] = {
        torch::kFloat64, torch::kInt32, torch::kInt32, torch::kInt32, torch::kInt32,
        torch::kFloat32, torch::kInt64, torch::kInt64, torch::kInt32, torch::kInt32,
        torch::kInt64, torch::kInt32, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kInt64, torch::kInt32, torch::kInt32, torch::kInt64,
        torch::kInt64, torch::kInt64, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kUInt32, torch::kInt64, torch::kBool, torch::kFloat64,
        torch::kInt64, torch::kInt64, torch::kFloat32, torch::kBool, torch::kFloat32,
        torch::kInt32, torch::kInt32, torch::kFloat64, torch::kFloat64,
        torch::kFloat64, torch::kFloat64, torch::kInt32, torch::kInt32,
        torch::kInt64, torch::kInt64, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kFloat64};
    for (int index = 0; index < 49; ++index) {
        std::vector<int64_t> shape = slots;
        if (index == 0) shape = {batch, capacity, 3};
        if (index == 6 || index == 10 || index == 12 || index == 13
            || (index >= 15 && index <= 17)
            || (index >= 20 && index <= 21) || (index >= 27 && index <= 31)
            || index == 44) shape = batch_shape;
        if (index == 22) shape = {batch, streams, 4};
        if (index == 23) shape = {batch, streams, 2};
        if (index == 24 || index == 25) shape = {batch, streams};
        if (index == 26) shape = {batch, 624};
        if (index == 32) shape = {batch, capacity, 16};
        if (index == 41) shape = {batch, capacity, 2};
        if (index == 43) shape = {batch, 12};
        exact(stage[index], device, stage_types[index], shape, "R4 staging");
    }
    const torch::ScalarType turn_types[22] = {
        torch::kInt64, torch::kInt64, torch::kInt32, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kInt32, torch::kBool, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kFloat32, torch::kFloat32, torch::kInt64,
        torch::kInt64, torch::kInt32, torch::kInt32, torch::kInt32,
        torch::kFloat64, torch::kInt64, torch::kInt32, torch::kInt64};
    for (int index = 0; index < 22; ++index) {
        std::vector<int64_t> shape = slots;
        if (index == 3 || index == 5 || index == 6 || index >= 19) shape = batch_shape;
        if (index >= 13 && index <= 15) shape = grid;
        exact(turn[index], device, turn_types[index], shape, "turnover staging");
    }
}

}  // namespace r5

void tensor_world_c4_resident_population_r5_cuda(
    std::vector<torch::Tensor> primary, std::vector<torch::Tensor> death_tensors,
    std::vector<torch::Tensor> inherit, std::vector<torch::Tensor> zero,
    std::vector<torch::Tensor> minus_one, std::vector<torch::Tensor> half,
    std::vector<torch::Tensor> one, std::vector<torch::Tensor> scalar_rng,
    std::vector<torch::Tensor> replay, std::vector<torch::Tensor> staging,
    std::vector<torch::Tensor> turnover, int64_t max_births,
    double parent_energy_share, double minimum_energy_buffer, int64_t cooldown_steps,
    double start_health, double max_energy, int64_t life_history_mode,
    int64_t maturity_start_ticks, int64_t maturity_rise_ticks,
    int64_t fertility_decline_start_ticks, int64_t fertility_decline_ticks,
    double elder_fertility_floor, int64_t generation, int64_t tick,
    int64_t environment_stream, int64_t runner_stream) {
    TORCH_CHECK(primary.size() == 101 && death_tensors.size() == 13,
        "C4 R5 authoritative ABI is incomplete");
    TORCH_CHECK(inherit.size() == 29 && zero.size() == 55 && minus_one.size() == 11
        && half.size() == 1 && one.size() == 1, "C4 R5 reset ABI is incomplete");
    TORCH_CHECK(scalar_rng.size() == 8 && replay.size() == 6
        && staging.size() == 49 && turnover.size() == 22,
        "C4 R5 continuation/staging ABI is incomplete");
    const int64_t batch = primary[0].size(0), capacity = primary[0].size(1);
    TORCH_CHECK(batch == 4 && capacity > 0, "C4 R5 is bounded to positive B4");
    const int64_t grid_size = primary[9].size(1), event_capacity = primary[77].size(1);
    const int64_t streams = replay[0].size(1), tape_capacity = scalar_rng[4].size(1);
    const auto device = primary[0].device();
    TORCH_CHECK(device.is_cuda(), "C4 R5 requires CUDA state");
    c10::cuda::CUDAGuard guard(device);
    r5::validate_abi(primary, death_tensors, inherit, zero, minus_one, half, one,
        scalar_rng, replay, staging, turnover, device, batch, capacity, grid_size,
        event_capacity, streams, tape_capacity);
    State live = make_state(primary, batch, capacity, grid_size, event_capacity);
    live.policy_use_gru = inherit[11].data_ptr<bool>();
    r4::Genome genome{
        inherit[2].data_ptr<int32_t>(), inherit[3].data_ptr<int32_t>(),
        inherit[10].data_ptr<int64_t>(), inherit[11].data_ptr<bool>(),
        inherit[12].data_ptr<float>(), inherit[13].data_ptr<float>(),
        inherit[14].data_ptr<float>(), inherit[15].data_ptr<float>(),
        inherit[16].data_ptr<float>(), inherit[17].data_ptr<float>(),
        inherit[18].data_ptr<float>(), inherit[19].data_ptr<float>(),
        inherit[20].data_ptr<float>(), inherit[21].data_ptr<float>(),
        inherit[26].data_ptr<float>(), inherit[27].data_ptr<float>(),
        inherit[28].data_ptr<float>()};
    r4::RandomState python{
        scalar_rng[0].data_ptr<uint32_t>(), scalar_rng[1].data_ptr<int64_t>(),
        scalar_rng[2].data_ptr<bool>(), scalar_rng[3].data_ptr<double>(),
        scalar_rng[4].data_ptr<double>(), scalar_rng[5].data_ptr<double>(),
        scalar_rng[6].data_ptr<int64_t>(), scalar_rng[7].data_ptr<int64_t>(), tape_capacity};
    r4::Replay replay_state{
        replay[0].data_ptr<int64_t>(), replay[1].data_ptr<int64_t>(),
        replay[2].data_ptr<int64_t>(), replay[3].data_ptr<int64_t>(),
        replay[4].data_ptr<int64_t>(), streams, environment_stream, runner_stream};
    r4::R4State live_r4{live, genome, python, replay_state, zero[15].data_ptr<float>()};
    r4::R4Stage stage = r4::make_stage(staging);
    r5::DeathState death{
        death_tensors[0].data_ptr<bool>(), death_tensors[1].data_ptr<int32_t>(),
        death_tensors[2].data_ptr<int64_t>(), death_tensors[3].data_ptr<int64_t>(),
        death_tensors[4].data_ptr<int64_t>(), death_tensors[5].data_ptr<int64_t>(),
        death_tensors[6].data_ptr<int64_t>(), death_tensors[7].data_ptr<int32_t>(),
        death_tensors[8].data_ptr<int64_t>(), death_tensors[9].data_ptr<int32_t>(),
        death_tensors[10].data_ptr<int64_t>(), death_tensors[11].data_ptr<int32_t>(),
        death_tensors[12].data_ptr<int32_t>()};
    r5::TurnStage turn = r5::make_turn(turnover);
    Config config{max_births, parent_energy_share, minimum_energy_buffer,
        static_cast<int32_t>(cooldown_steps), static_cast<float>(start_health),
        static_cast<float>(max_energy), static_cast<int32_t>(life_history_mode),
        static_cast<int32_t>(maturity_start_ticks), static_cast<int32_t>(maturity_rise_ticks),
        static_cast<int32_t>(fertility_decline_start_ticks),
        static_cast<int32_t>(fertility_decline_ticks), elder_fertility_floor,
        generation, tick};
    const auto stream = at::cuda::getCurrentCUDAStream(device.index());
    r5::prepare_post_death_shadow<<<4, 1, 0, stream>>>(
        live, death, replay_state, turn, config);
    State shadow = live;
    shadow.alive = turn.alive; shadow.agent_id = turn.agent_id;
    shadow.pos_y = turn.pos_y; shadow.pos_x = turn.pos_x;
    shadow.energy = turn.energy; shadow.health = turn.health;
    shadow.occupancy_agent = turn.occupancy_agent;
    shadow.occupancy_slot = turn.occupancy_slot; shadow.patch_agent = turn.patch_agent;
    shadow.agent_id_to_slot = turn.id_to_slot; shadow.free_queue = turn.virtual_queue;
    shadow.free_count = turn.virtual_count; shadow.repro_cooldown = turn.repro_cooldown;
    shadow.policy_energy_observation = turn.policy_energy;
    shadow.event_active_count = turn.event_count; shadow.event_status = turn.event_status;
    r4::Replay shadow_replay = replay_state;
    shadow_replay.event_sequence = turn.replay_event_sequence;
    r4::R4State shadow_r4{shadow, genome, python, shadow_replay, zero[15].data_ptr<float>()};
    r4::preflight_kernel<<<4, 1, 0, stream>>>(shadow_r4, stage, config);
    r5::merge_all_status<<<1, 1, 0, stream>>>(stage, turn, live, config);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    for (auto& tensor : inherit) dispatch_inherit(tensor, stage.base, batch, capacity, stream);
    for (auto& tensor : zero) dispatch_fill(tensor, 0.0, stage.base, batch, capacity, stream);
    for (auto& tensor : minus_one) dispatch_fill(tensor, -1.0, stage.base, batch, capacity, stream);
    for (auto& tensor : half) dispatch_fill(tensor, 0.5, stage.base, batch, capacity, stream);
    for (auto& tensor : one) dispatch_fill(tensor, 1.0, stage.base, batch, capacity, stream);
    r5::commit_deaths<<<4, 1, 0, stream>>>(live, death, stage, turn, config);
    r4::commit_kernel<<<4, 1, 0, stream>>>(live_r4, stage, config);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
