#include "tensor_world_c4_resident_birth_r2.cu"

namespace {

constexpr int32_t kR3InvalidQueue = 20;
constexpr int32_t kR3Capacity = 21;
constexpr int32_t kR3Duplicate = 22;
constexpr int32_t kR3Event = 23;
constexpr int32_t kR3Birth = 24;
constexpr int32_t kR3Admission = 25;
constexpr int32_t kR3DeathEvent = 4;

struct DeathStateR3 {
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

struct TurnStageR3 {
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
};

__device__ __forceinline__ bool death_candidate_r3(
    State state, DeathStateR3 death, int64_t offset, int64_t slot) {
    return state.alive[offset + slot] && state.agent_id[offset + slot] >= 0
        && death.candidate[offset + slot];
}

__global__ void prepare_shadow_r3(
    State live, DeathStateR3 death, Stage birth, TurnStageR3 turn) {
    if (threadIdx.x != 0) return;
    const int64_t world = static_cast<int64_t>(blockIdx.x);
    if (world >= live.batch) return;
    const int64_t offset = world * live.capacity;
    const int64_t grid_offset = world * live.grid_size * live.grid_size;
    turn.status[world] = 0;
    turn.death_count[world] = 0;
    turn.virtual_count[world] = 0;
    turn.event_status[world] = live.event_status[world];
    turn.event_count[world] = live.event_active_count[world];
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
        turn.status[world] = kR3InvalidQueue;
        return;
    }
    if (birth.event_sequence[world] != cursor || live.event_status[world] != 0
        || cursor < 0 || cursor > live.event_capacity) {
        turn.status[world] = kR3Event;
        return;
    }
    int64_t virtual_count = 0;
    for (int64_t row = 0; row < free_count + pending; ++row) {
        const int64_t slot = row < free_count ? live.free_queue[offset + row]
            : death.pending_slots[offset + row - free_count];
        if (slot < 0 || slot >= live.capacity || live.alive[offset + slot]
            || live.agent_id[offset + slot] >= 0) {
            turn.status[world] = kR3InvalidQueue;
            return;
        }
        for (int64_t prior = 0; prior < row; ++prior) {
            if (turn.virtual_queue[offset + prior] == slot) {
                turn.status[world] = kR3Duplicate;
                return;
            }
        }
        turn.virtual_queue[offset + virtual_count++] = slot;
    }

    int64_t candidates = 0;
    for (int64_t slot = 0; slot < live.capacity; ++slot) {
        candidates += death_candidate_r3(live, death, offset, slot) ? 1 : 0;
    }
    if (virtual_count + candidates > live.capacity
        || candidates > live.event_capacity - cursor) {
        turn.status[world] = kR3Capacity;
        return;
    }
    int64_t previous_id = INT64_MIN;
    int64_t previous_slot = -1;
    for (int64_t row = 0; row < candidates; ++row) {
        int64_t selected_id = INT64_MAX;
        int64_t selected_slot = -1;
        for (int64_t slot = 0; slot < live.capacity; ++slot) {
            if (!death_candidate_r3(live, death, offset, slot)) continue;
            const int64_t id = live.agent_id[offset + slot];
            const bool follows = id > previous_id || (id == previous_id && slot > previous_slot);
            if (follows && (id < selected_id || (id == selected_id && slot < selected_slot))) {
                selected_id = id;
                selected_slot = slot;
            }
        }
        if (selected_slot < 0 || (row > 0 && turn.death_ids[offset + row - 1] == selected_id)
            || live.lineage_founder[offset + selected_slot] < 0
            || live.lineage_depth[offset + selected_slot] < 0
            || selected_id >= live.capacity
            || live.agent_id_to_slot[offset + selected_id] != selected_slot) {
            turn.status[world] = kR3Event;
            return;
        }
        for (int64_t prior = 0; prior < virtual_count; ++prior) {
            if (turn.virtual_queue[offset + prior] == selected_slot) {
                turn.status[world] = kR3Duplicate;
                return;
            }
        }
        turn.death_ids[offset + row] = selected_id;
        turn.death_slots[offset + row] = selected_slot;
        turn.death_reasons[offset + row] = live.health[offset + selected_slot] <= 0.0f ? 2 : 1;
        turn.virtual_queue[offset + virtual_count++] = selected_slot;

        const int64_t y = live.pos_y[offset + selected_slot];
        const int64_t x = live.pos_x[offset + selected_slot];
        if (y < 0 || y >= live.grid_size || x < 0 || x >= live.grid_size) {
            turn.status[world] = kR3Event;
            return;
        }
        const int64_t cell = grid_offset + y * live.grid_size + x;
        if (turn.occupancy_agent[cell] != selected_id
            || turn.occupancy_slot[cell] != selected_slot
            || turn.patch_agent[cell] != static_cast<int32_t>(selected_id)) {
            turn.status[world] = kR3Event;
            return;
        }
        turn.occupancy_agent[cell] = -1;
        turn.occupancy_slot[cell] = -1;
        turn.patch_agent[cell] = -1;
        turn.id_to_slot[offset + selected_id] = -1;
        turn.alive[offset + selected_slot] = false;
        turn.agent_id[offset + selected_slot] = -1;
        turn.pos_y[offset + selected_slot] = -1;
        turn.pos_x[offset + selected_slot] = -1;
        turn.energy[offset + selected_slot] = 0.0f;
        turn.health[offset + selected_slot] = 0.0f;
        turn.repro_cooldown[offset + selected_slot] = 0;
        turn.policy_energy[offset + selected_slot] = __longlong_as_double(
            static_cast<long long>(0x7ff8000000000000ULL));
        previous_id = selected_id;
        previous_slot = selected_slot;
    }
    turn.death_count[world] = candidates;
    turn.virtual_count[world] = virtual_count;
    turn.event_count[world] = cursor + candidates;
    turn.event_status[world] = 0;
    birth.event_sequence[world] = cursor + candidates;
}

__global__ void merge_status_r3(Stage birth, TurnStageR3 turn, int64_t batch) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    bool failed = false;
    for (int64_t world = 0; world < batch; ++world) {
        failed |= turn.status[world] != 0 || birth.status[world] != 0;
    }
    if (failed) {
        for (int64_t world = 0; world < batch; ++world) {
            if (birth.status[world] == 0) birth.status[world] = kR3Birth;
        }
    }
}

__global__ void enforce_admission_r3(Stage birth, TurnStageR3 turn, int64_t capacity) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (turn.status[0] != 0 || birth.status[0] != 0) return;
    bool reuses_new_death = false;
    for (int64_t child = 0; child < birth.planned_count[0]; ++child) {
        for (int64_t death = 0; death < turn.death_count[0]; ++death) {
            reuses_new_death |= birth.child_slots[child] == turn.death_slots[death];
        }
    }
    if (turn.death_count[0] < 1 || birth.planned_count[0] < 1 || !reuses_new_death
        || turn.death_count[0] > capacity || birth.planned_count[0] > capacity) {
        turn.status[0] = kR3Admission;
    }
}

__global__ void combined_commit_r3(
    State state, DeathStateR3 death, Stage stage, TurnStageR3 turn, Config config) {
    if (threadIdx.x != 0) return;
    for (int64_t other = 0; other < state.batch; ++other) {
        if (turn.status[other] != 0 || stage.status[other] != 0) return;
    }
    const int64_t world = static_cast<int64_t>(blockIdx.x);
    if (world >= state.batch) return;
    const int64_t offset = world * state.capacity;
    const int64_t grid_offset = world * state.grid_size * state.grid_size;
    const int64_t event_before = state.event_active_count[world];
    const int64_t deaths = turn.death_count[world];
    Config death_event_config = config;
    death_event_config.generation = 0;  // Frozen CPU death-settlement semantics.

    // Seal all death events while deceased lineage and identity are authoritative.
    for (int64_t row = 0; row < deaths; ++row) {
        const int64_t slot = turn.death_slots[offset + row];
        const int64_t id = turn.death_ids[offset + row];
        set_event(state, world, event_before + row, kR3DeathEvent, id, -1,
            slot, -1, -1, -1, state.lineage_founder[offset + slot],
            state.lineage_depth[offset + slot], turn.death_reasons[offset + row], 1,
            static_cast<float>(turn.death_reasons[offset + row]), death_event_config);
    }

    const int64_t old_pending = death.pending_count[world];
    for (int64_t row = 0; row < state.capacity; ++row) {
        death.output_ids[offset + row] = -1;
        death.output_slots[offset + row] = -1;
        death.output_reasons[offset + row] = 0;
        state.parent_slots[offset + row] = stage.parent_slots[offset + row];
        state.child_slots[offset + row] = stage.child_slots[offset + row];
        state.child_y[offset + row] = stage.child_y[offset + row];
        state.child_x[offset + row] = stage.child_x[offset + row];
        state.child_energy[offset + row] = stage.child_energy[offset + row];
        state.mate_slots[offset + row] = -1;
        state.decision_parent_slots[offset + row] = stage.decision_parent_slots[offset + row];
        state.decision_mate_slots[offset + row] = -1;
        state.decision_adjacent_counts[offset + row] = stage.decision_adjacent_counts[offset + row];
        state.decision_reason_codes[offset + row] = stage.decision_reason_codes[offset + row];
        state.planned_repro_cooldown[offset + row] = stage.planned_repro_cooldown[offset + row];
        state.birth_parent_ids[offset + row] = -1;
        state.birth_child_ids[offset + row] = -1;
        state.mutation_occurred[offset + row] = false;
        state.mutation_delta[offset + row] = 0.0f;
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
        state.agent_id_to_slot[offset + id] = -1;
        state.alive[offset + slot] = false;
        state.agent_id[offset + slot] = -1;
        state.pos_y[offset + slot] = -1;
        state.pos_x[offset + slot] = -1;
        state.energy[offset + slot] = 0.0f;
        state.health[offset + slot] = 0.0f;
        state.policy_energy_observation[offset + slot] = __longlong_as_double(
            static_cast<long long>(0x7ff8000000000000ULL));
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
    death.pending_count[world] = 0;  // payload rows are explicitly count-masked.

    const int64_t planned = stage.planned_count[world];
    state.planned_count[world] = planned;
    state.plan_status[world] = 0;
    state.decision_count[world] = stage.decision_count[world];
    state.birth_count[world] = planned;
    state.spawn_draws[world] = stage.spawn_draws[world];
    state.spawn_semantic_draws[world] = stage.spawn_semantic_draws[world];
    state.runner_draws[world] = 0;
    state.runner_semantic_draws[world] = 0;
    for (int64_t birth = 0; birth < planned; ++birth) {
        const int64_t parent = stage.parent_slots[offset + birth];
        const int64_t child = stage.child_slots[offset + birth];
        const int64_t parent_id = state.agent_id[offset + parent];
        const int64_t child_id = stage.child_agent_ids[offset + birth];
        double parent_energy = state.policy_energy_observation[offset + parent];
        if (!isfinite(parent_energy)) parent_energy = state.energy[offset + parent];
        const float child_energy = stage.child_energy[offset + birth];
        const double remaining = parent_energy - child_energy;
        state.energy[offset + parent] = static_cast<float>(remaining);
        state.policy_energy_observation[offset + parent] = remaining;
        state.repro_cooldown[offset + parent] = config.cooldown_steps;
        ++state.offspring_count[offset + parent];
        if (state.first_repro_step[offset + parent] < 0) {
            state.first_repro_step[offset + parent] =
                static_cast<int32_t>(state.survival_steps[offset + parent]);
        }
        state.birth_parent_ids[offset + birth] = parent_id;
        state.birth_child_ids[offset + birth] = child_id;
        ++state.slot_tokens[offset + child];
        state.alive[offset + child] = true;
        state.agent_id[offset + child] = child_id;
        state.pos_y[offset + child] = stage.child_y[offset + birth];
        state.pos_x[offset + child] = stage.child_x[offset + birth];
        const int64_t cell = grid_offset + stage.child_y[offset + birth] * state.grid_size
            + stage.child_x[offset + birth];
        state.occupancy_agent[cell] = child_id;
        state.occupancy_slot[cell] = child;
        state.patch_agent[cell] = static_cast<int32_t>(child_id);
        state.energy[offset + child] = child_energy;
        state.health[offset + child] = config.start_health;
        state.age[offset + child] = 0;
        state.survival_steps[offset + child] = 0;
        state.energy_collected[offset + child] = 0.0;
        state.agent_id_to_slot[offset + child_id] = static_cast<int32_t>(child);
        state.agent_age[offset + child] = 0;
        state.agent_survival_steps[offset + child] = 0;
        state.policy_energy_observation[offset + child] = child_energy;
        state.repro_threshold[offset + child] = state.genome_repro_threshold[offset + child];
        state.drive_inertia[offset + child] = state.genome_drive_inertia[offset + child];
        state.drive_noise[offset + child] = state.genome_drive_noise[offset + child];
        const double energy_n = newborn_energy_n_cpu_oracle(
            child_energy, config.max_energy);
        state.drives[(offset + child) * 4] = newborn_hunger_drive_cpu_oracle(
            state.genome_drive_setpoints[(offset + child) * 4], energy_n);
        for (int32_t drive = 1; drive < 4; ++drive) {
            state.drives[(offset + child) * 4 + drive] = fmaxf(0.0f, fminf(1.0f,
                state.genome_drive_setpoints[(offset + child) * 4 + drive]));
        }
        for (int32_t drive = 0; drive < 4; ++drive) {
            state.policy_drives[(offset + child) * 4 + drive] =
                state.drives[(offset + child) * 4 + drive];
        }
        state.policy_attack_bias[offset + child] = state.genome_aggression[offset + child];
        state.policy_action_temperature[offset + child] =
            state.genome_action_temperature[offset + child];
        state.policy_drive_logit_gain[offset + child] =
            state.genome_drive_logit_gain[offset + child];
        float* chemistry = state.chemistry + (offset + child) * 6;
        chemistry[0] = newborn_energy_store_cpu_oracle(energy_n);
        chemistry[1] = 0.55f;
        chemistry[2] = 0.0f;
        chemistry[3] = 0.0f;
        chemistry[4] = 0.50f;
        chemistry[5] = 0.25f;
        state.pre_energy[offset + child] = child_energy;
        state.pre_health[offset + child] = config.start_health;
        state.pre_injury[offset + child] = 0.0f;
        state.pre_survival_steps[offset + child] = 0;
        const int64_t founder = state.lineage_founder[offset + parent];
        state.lineage_founder[offset + child] = founder < 0 ? parent_id : founder;
        state.lineage_depth[offset + child] = state.lineage_depth[offset + parent] + 1;
        state.lineage_birth_generation[offset + child] = config.generation;
        state.lineage_birth_tick[offset + child] = config.tick;
        state.lineage_parent[offset + child] = parent_id;
        const int32_t parent_patch = state.patch_lineage[offset + parent];
        state.patch_lineage[offset + child] = parent_patch == 0
            ? static_cast<int32_t>(parent_id) : parent_patch;
        state.scar[offset + child] = 0.0f;
        state.repro_cooldown[offset + child] = 0;
        state.planned_repro_cooldown[offset + child] = 0;
        state.offspring_count[offset + child] = 0;
        state.first_repro_step[offset + child] = -1;
    }

    int64_t remaining = 0;
    for (int64_t row = stage.queue_cursor[world]; row < turn.virtual_count[world]; ++row) {
        const int64_t slot = state.free_queue[offset + row];
        if (slot >= 0 && slot < state.capacity && !state.alive[offset + slot]
            && state.agent_id[offset + slot] < 0) state.free_queue[offset + remaining++] = slot;
    }
    for (int64_t row = remaining; row < state.capacity; ++row) state.free_queue[offset + row] = -1;
    state.free_count[world] = remaining;
    if (planned > 0) state.next_agent_id[world] = stage.child_agent_ids[offset + planned - 1] + 1;

    for (int64_t row = 0; row < state.capacity; ++row) {
        state.active_agent_ids[offset + row] = -1;
        state.active_agent_slots[offset + row] = -1;
        state.survivor_agent_ids[offset + row] = -1;
    }
    int64_t active = 0;
    int64_t previous_id = INT64_MIN;
    int64_t previous_slot = -1;
    while (true) {
        int64_t selected_id = INT64_MAX;
        int64_t selected_slot = -1;
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

    int64_t output = event_before + deaths;
    int64_t success = 0;
    for (int64_t decision = 0; decision < stage.decision_count[world]; ++decision) {
        const int64_t parent_slot = stage.decision_parent_slots[offset + decision];
        const int64_t parent_id = state.agent_id[offset + parent_slot];
        const int32_t reason = stage.decision_reason_codes[offset + decision];
        if ((reason >= 1 && reason <= 3) || (reason >= 5 && reason <= 6)) {
            int64_t founder = state.lineage_founder[offset + parent_slot];
            if (founder < 0) founder = parent_id;
            set_event(state, world, output++, kBirthFailedEvent, parent_id, -1,
                parent_slot, -1, -1, -1, founder, state.lineage_depth[offset + parent_slot],
                reason, 0, static_cast<float>(reason), config);
        } else if (reason == kAccepted) {
            const int64_t child_slot = stage.child_slots[offset + success];
            const int64_t child_id = stage.child_agent_ids[offset + success];
            set_event(state, world, output++, kBirthCommittedEvent, parent_id, child_id,
                parent_slot, child_slot, parent_id, child_id,
                state.lineage_founder[offset + child_slot], state.lineage_depth[offset + child_slot],
                -1, 1, stage.child_energy[offset + success], config);
            ++success;
        }
    }
    state.event_active_count[world] = output;
    stage.event_sequence[world] = output;
    state.event_status[world] = 0;
}

void require_exact_r3(
    const torch::Tensor& tensor, const torch::Device& device,
    torch::ScalarType dtype, const std::vector<int64_t>& shape, const char* group) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == device && tensor.is_contiguous(),
        "C4 R3 ", group, " tensors must be contiguous on one CUDA device");
    TORCH_CHECK(tensor.scalar_type() == dtype, "C4 R3 ", group, " dtype mismatch");
    TORCH_CHECK(tensor.sizes().vec() == shape, "C4 R3 ", group, " shape mismatch");
}

void validate_abi_r3(
    const std::vector<torch::Tensor>& p, const std::vector<torch::Tensor>& death,
    const std::vector<torch::Tensor>& inherit, const std::vector<torch::Tensor>& zero,
    const std::vector<torch::Tensor>& minus_one, const std::vector<torch::Tensor>& half,
    const std::vector<torch::Tensor>& one, const std::vector<torch::Tensor>& birth,
    const std::vector<torch::Tensor>& turn, const torch::Device& device,
    int64_t batch, int64_t capacity, int64_t grid_size, int64_t event_capacity) {
    const std::vector<int64_t> slots{batch, capacity};
    const std::vector<int64_t> batch_shape{batch};
    const std::vector<int64_t> grid{batch, grid_size, grid_size};
    const std::vector<int64_t> events{batch, event_capacity};
    const std::vector<torch::ScalarType> p_types{
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
    TORCH_CHECK(p_types.size() == p.size(), "C4 R3 primary dtype inventory mismatch");
    std::vector<std::vector<int64_t>> p_shapes(101, slots);
    for (int index : {18, 20, 27, 28, 43, 47, 50, 51, 52, 53, 56, 99, 100})
        p_shapes[index] = batch_shape;
    p_shapes[9] = grid; p_shapes[10] = grid; p_shapes[11] = grid;
    p_shapes[12] = {batch, capacity, 4};
    p_shapes[61] = {batch, capacity, 4};
    p_shapes[65] = {batch, capacity, 6};
    p_shapes[72] = {batch, capacity, 4};
    for (int index = 77; index <= 98; ++index) p_shapes[index] = events;
    for (size_t index = 0; index < p.size(); ++index)
        require_exact_r3(p[index], device, p_types[index], p_shapes[index], "primary");

    const torch::ScalarType death_types[13] = {
        torch::kBool, torch::kInt32, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kInt64, torch::kInt32, torch::kInt64, torch::kInt32,
        torch::kInt64, torch::kInt32, torch::kInt32};
    for (int index = 0; index < 13; ++index) {
        require_exact_r3(death[index], device, death_types[index],
            index == 4 || index == 8 ? batch_shape : slots, "death");
    }

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
        require_exact_r3(inherit[index], device, inherit_types[index], shape, "inherit");
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
    for (int index = 0; index < 55; ++index) {
        const std::vector<int64_t> shape = zero_widths[index] == 0 ? slots
            : std::vector<int64_t>{batch, capacity, zero_widths[index]};
        require_exact_r3(zero[index], device, zero_types[index], shape, "zero reset");
    }
    const torch::ScalarType minus_types[11] = {
        torch::kInt64, torch::kInt32, torch::kInt32, torch::kInt32, torch::kInt32,
        torch::kInt32, torch::kInt64, torch::kInt32, torch::kInt32,
        torch::kInt32, torch::kInt32};
    for (int index = 0; index < 11; ++index) {
        const std::vector<int64_t> shape = index == 5
            ? std::vector<int64_t>{batch, capacity, 2} : slots;
        require_exact_r3(minus_one[index], device, minus_types[index], shape, "minus-one reset");
    }
    require_exact_r3(half[0], device, torch::kFloat64, {batch, capacity, 7}, "half reset");
    require_exact_r3(one[0], device, torch::kFloat32, slots, "one reset");

    const torch::ScalarType birth_types[18] = {
        torch::kFloat64, torch::kInt32, torch::kInt32, torch::kInt32, torch::kInt32,
        torch::kFloat32, torch::kInt64, torch::kInt64, torch::kInt32, torch::kInt32,
        torch::kInt64, torch::kInt32, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kInt64, torch::kInt32};
    for (int index = 0; index < 18; ++index) {
        std::vector<int64_t> shape = slots;
        if (index == 0) shape = {batch, capacity, 3};
        if (index == 6 || (index >= 10 && index <= 13) || index >= 15) shape = batch_shape;
        if (index == 11 || index == 14) shape = slots;
        require_exact_r3(birth[index], device, birth_types[index], shape, "birth staging");
    }
    const torch::ScalarType turn_types[21] = {
        torch::kInt64, torch::kInt64, torch::kInt32, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kInt32, torch::kBool, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kFloat32, torch::kFloat32, torch::kInt64,
        torch::kInt64, torch::kInt32, torch::kInt32, torch::kInt32,
        torch::kFloat64, torch::kInt64, torch::kInt32};
    for (int index = 0; index < 21; ++index) {
        std::vector<int64_t> shape = slots;
        if (index == 3 || index == 5 || index == 6 || index == 19 || index == 20)
            shape = batch_shape;
        if (index >= 13 && index <= 15) shape = grid;
        require_exact_r3(turn[index], device, turn_types[index], shape, "turnover staging");
    }
}

}  // namespace

void tensor_world_c4_resident_turnover_r3_cuda(
    std::vector<torch::Tensor> primary,
    std::vector<torch::Tensor> death_authoritative,
    std::vector<torch::Tensor> inherit,
    std::vector<torch::Tensor> zero,
    std::vector<torch::Tensor> minus_one,
    std::vector<torch::Tensor> half,
    std::vector<torch::Tensor> one,
    std::vector<torch::Tensor> birth_staging,
    std::vector<torch::Tensor> turnover_staging,
    int64_t max_births, double parent_energy_share, double minimum_energy_buffer,
    int64_t cooldown_steps, double start_health, double max_energy,
    int64_t life_history_mode, int64_t maturity_start_ticks, int64_t maturity_rise_ticks,
    int64_t fertility_decline_start_ticks, int64_t fertility_decline_ticks,
    double elder_fertility_floor, int64_t generation, int64_t tick) {
    TORCH_CHECK(primary.size() == 101 && death_authoritative.size() == 13,
        "C4 R3 authoritative ABI is incomplete");
    TORCH_CHECK(birth_staging.size() == 18 && turnover_staging.size() == 21,
        "C4 R3 staging ABI is incomplete");
    TORCH_CHECK(primary[0].dim() == 2 && primary[0].size(0) == 1,
        "C4 R3 is bounded to B1");
    const int64_t batch = 1;
    const int64_t capacity = primary[0].size(1);
    TORCH_CHECK(primary[9].dim() == 3 && primary[9].size(0) == batch
        && primary[9].size(1) == primary[9].size(2), "C4 R3 grid ABI is invalid");
    TORCH_CHECK(primary[77].dim() == 2 && primary[77].size(0) == batch,
        "C4 R3 event ABI is invalid");
    const int64_t grid_size = primary[9].size(1);
    const int64_t event_capacity = primary[77].size(1);
    const auto device = primary[0].device();
    TORCH_CHECK(device.is_cuda() && capacity > 0, "C4 R3 requires CUDA state");
    c10::cuda::CUDAGuard guard(device);
    TORCH_CHECK(inherit.size() == 29 && zero.size() == 55 && minus_one.size() == 11
        && half.size() == 1 && one.size() == 1, "C4 R3 reset ABI is incomplete");
    validate_abi_r3(primary, death_authoritative, inherit, zero, minus_one, half, one,
        birth_staging, turnover_staging, device, batch, capacity, grid_size, event_capacity);
    TORCH_CHECK(cooldown_steps >= 0 && generation >= 0 && tick >= 0,
        "C4 R3 scalar configuration is invalid");
    State live = make_state(primary, batch, capacity, grid_size, event_capacity);
    live.policy_use_gru = inherit[11].data_ptr<bool>();
    Stage birth{
        birth_staging[0].data_ptr<double>(), birth_staging[1].data_ptr<int32_t>(),
        birth_staging[2].data_ptr<int32_t>(), birth_staging[3].data_ptr<int32_t>(),
        birth_staging[4].data_ptr<int32_t>(), birth_staging[5].data_ptr<float>(),
        birth_staging[6].data_ptr<int64_t>(), birth_staging[7].data_ptr<int64_t>(),
        birth_staging[8].data_ptr<int32_t>(), birth_staging[9].data_ptr<int32_t>(),
        birth_staging[10].data_ptr<int64_t>(), birth_staging[11].data_ptr<int32_t>(),
        birth_staging[12].data_ptr<int64_t>(), birth_staging[13].data_ptr<int64_t>(),
        birth_staging[14].data_ptr<int64_t>(), birth_staging[15].data_ptr<int64_t>(),
        birth_staging[16].data_ptr<int64_t>(), birth_staging[17].data_ptr<int32_t>()};
    DeathStateR3 death{
        death_authoritative[0].data_ptr<bool>(), death_authoritative[1].data_ptr<int32_t>(),
        death_authoritative[2].data_ptr<int64_t>(), death_authoritative[3].data_ptr<int64_t>(),
        death_authoritative[4].data_ptr<int64_t>(), death_authoritative[5].data_ptr<int64_t>(),
        death_authoritative[6].data_ptr<int64_t>(), death_authoritative[7].data_ptr<int32_t>(),
        death_authoritative[8].data_ptr<int64_t>(), death_authoritative[9].data_ptr<int32_t>(),
        death_authoritative[10].data_ptr<int64_t>(), death_authoritative[11].data_ptr<int32_t>(),
        death_authoritative[12].data_ptr<int32_t>()};
    TurnStageR3 turn{
        turnover_staging[0].data_ptr<int64_t>(), turnover_staging[1].data_ptr<int64_t>(),
        turnover_staging[2].data_ptr<int32_t>(), turnover_staging[3].data_ptr<int64_t>(),
        turnover_staging[4].data_ptr<int64_t>(), turnover_staging[5].data_ptr<int64_t>(),
        turnover_staging[6].data_ptr<int32_t>(), turnover_staging[7].data_ptr<bool>(),
        turnover_staging[8].data_ptr<int64_t>(), turnover_staging[9].data_ptr<int64_t>(),
        turnover_staging[10].data_ptr<int64_t>(), turnover_staging[11].data_ptr<float>(),
        turnover_staging[12].data_ptr<float>(), turnover_staging[13].data_ptr<int64_t>(),
        turnover_staging[14].data_ptr<int64_t>(), turnover_staging[15].data_ptr<int32_t>(),
        turnover_staging[16].data_ptr<int32_t>(), turnover_staging[17].data_ptr<int32_t>(),
        turnover_staging[18].data_ptr<double>(), turnover_staging[19].data_ptr<int64_t>(),
        turnover_staging[20].data_ptr<int32_t>()};
    Config config{max_births, parent_energy_share, minimum_energy_buffer,
        static_cast<int32_t>(cooldown_steps), static_cast<float>(start_health),
        static_cast<float>(max_energy), static_cast<int32_t>(life_history_mode),
        static_cast<int32_t>(maturity_start_ticks), static_cast<int32_t>(maturity_rise_ticks),
        static_cast<int32_t>(fertility_decline_start_ticks),
        static_cast<int32_t>(fertility_decline_ticks), elder_fertility_floor, generation, tick};

    const auto stream = at::cuda::getCurrentCUDAStream(device.index());
    prepare_shadow_r3<<<1, 1, 0, stream>>>(live, death, birth, turn);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    State shadow = live;
    shadow.alive = turn.alive;
    shadow.agent_id = turn.agent_id;
    shadow.pos_y = turn.pos_y;
    shadow.pos_x = turn.pos_x;
    shadow.energy = turn.energy;
    shadow.health = turn.health;
    shadow.occupancy_agent = turn.occupancy_agent;
    shadow.occupancy_slot = turn.occupancy_slot;
    shadow.patch_agent = turn.patch_agent;
    shadow.agent_id_to_slot = turn.id_to_slot;
    shadow.free_queue = turn.virtual_queue;
    shadow.free_count = turn.virtual_count;
    shadow.repro_cooldown = turn.repro_cooldown;
    shadow.policy_energy_observation = turn.policy_energy;
    shadow.event_active_count = turn.event_count;
    shadow.event_status = turn.event_status;
    preflight_plan_kernel<<<1, 1, 0, stream>>>(shadow, birth, config);
    enforce_admission_r3<<<1, 1, 0, stream>>>(birth, turn, capacity);
    merge_status_r3<<<1, 1, 0, stream>>>(birth, turn, batch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    for (auto& tensor : inherit) dispatch_inherit(tensor, birth, batch, capacity, stream);
    for (auto& tensor : zero) dispatch_fill(tensor, 0.0, birth, batch, capacity, stream);
    for (auto& tensor : minus_one) dispatch_fill(tensor, -1.0, birth, batch, capacity, stream);
    for (auto& tensor : half) dispatch_fill(tensor, 0.5, birth, batch, capacity, stream);
    for (auto& tensor : one) dispatch_fill(tensor, 1.0, birth, batch, capacity, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    combined_commit_r3<<<1, 1, 0, stream>>>(live, death, birth, turn, config);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
