#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int32_t kOk = 0;
constexpr int32_t kInvalidPlan = 1;
constexpr int32_t kInvalidQueue = 2;
constexpr int32_t kInvalidParent = 3;
constexpr int32_t kInvalidChild = 4;
constexpr int32_t kIdentityCapacity = 5;
constexpr int32_t kEventCapacity = 6;
constexpr int32_t kEventSemantics = 7;
constexpr int32_t kUnsupportedGru = 8;
constexpr int32_t kSpawnTapeInvalid = 9;

constexpr int32_t kAccepted = 0;
constexpr int32_t kNoSpace = 1;
constexpr int32_t kBuffer = 2;
constexpr int32_t kNoFreeSlot = 3;
constexpr int32_t kOpportunity = 4;
constexpr int32_t kHealth = 7;
constexpr int32_t kCooldown = 8;
constexpr int32_t kNoEnergy = 9;
constexpr int32_t kBirthCommittedEvent = 1;
constexpr int32_t kBirthFailedEvent = 3;
constexpr int64_t kSentinel = -1;

struct Config {
    int64_t max_births;
    double parent_energy_share;
    double minimum_energy_buffer;
    int32_t cooldown_steps;
    float start_health;
    float max_energy;
    int32_t life_history_mode;
    int32_t maturity_start_ticks;
    int32_t maturity_rise_ticks;
    int32_t fertility_decline_start_ticks;
    int32_t fertility_decline_ticks;
    double elder_fertility_floor;
    int64_t generation;
    int64_t tick;
    const int64_t* generation_device;
    const int64_t* tick_device;
    int32_t identity_mode;
};

constexpr int32_t kIdentityLegacyDense = 0;
constexpr int32_t kIdentitySortedMonotonic = 1;

__device__ __forceinline__ int64_t config_generation(
    const Config& config, int64_t world) {
    return config.generation_device == nullptr
        ? config.generation : config.generation_device[world];
}

__device__ __forceinline__ int64_t config_tick(
    const Config& config, int64_t world) {
    return config.tick_device == nullptr ? config.tick : config.tick_device[world];
}

// Match the scalar CPU newborn oracle exactly.  The oracle computes the
// energy term in Python float64, casts that scalar once when adding it to the
// float32 setpoint term, and materializes chemistry from float64 scalars.
// Explicit round-to-nearest float operations prevent nvcc from fusing the
// setpoint multiply and final add into a different result.
__device__ __forceinline__ double newborn_energy_n_cpu_oracle(
    float child_energy, float max_energy) {
    return fmax(0.0, fmin(1.0,
        static_cast<double>(child_energy) / static_cast<double>(max_energy)));
}

__device__ __forceinline__ float newborn_hunger_drive_cpu_oracle(
    float setpoint, double energy_n) {
    const float setpoint_term = __fmul_rn(0.60f, setpoint);
    const float energy_term = static_cast<float>(0.40 * (1.0 - energy_n));
    return fmaxf(0.0f, fminf(1.0f, __fadd_rn(setpoint_term, energy_term)));
}

__device__ __forceinline__ float newborn_energy_store_cpu_oracle(
    double energy_n) {
    return static_cast<float>(0.40 + 0.40 * energy_n);
}

struct Stage {
    double* spawn_uniforms;
    int32_t* parent_slots;
    int32_t* child_slots;
    int32_t* child_y;
    int32_t* child_x;
    float* child_energy;
    int64_t* planned_count;
    int64_t* decision_parent_slots;
    int32_t* decision_adjacent_counts;
    int32_t* decision_reason_codes;
    int64_t* decision_count;
    int32_t* planned_repro_cooldown;
    int64_t* spawn_draws;
    int64_t* spawn_semantic_draws;
    int64_t* child_agent_ids;
    int64_t* queue_cursor;
    int64_t* event_sequence;
    int32_t* status;
};

struct State {
    bool* alive;
    int64_t* agent_id;
    int64_t* pos_y;
    int64_t* pos_x;
    float* energy;
    float* health;
    int64_t* age;
    int64_t* survival_steps;
    double* energy_collected;
    int64_t* occupancy_agent;
    int64_t* occupancy_slot;
    int32_t* patch_agent;
    float* policy_drives;
    float* policy_attack_bias;
    float* policy_action_temperature;
    float* policy_drive_logit_gain;
    int32_t* agent_id_to_slot;
    int64_t* slot_tokens;
    int64_t* next_agent_id;
    int64_t* free_queue;
    int64_t* free_count;
    int32_t* parent_slots;
    int32_t* child_slots;
    int32_t* child_y;
    int32_t* child_x;
    float* child_energy;
    int32_t* mate_slots;
    int64_t* planned_count;
    int32_t* plan_status;
    float* scar;
    int32_t* repro_cooldown;
    int32_t* offspring_count;
    int32_t* first_repro_step;
    int64_t* lineage_birth_generation;
    int64_t* lineage_birth_tick;
    int64_t* lineage_founder;
    int32_t* lineage_depth;
    int64_t* lineage_parent;
    int32_t* patch_lineage;
    int64_t* decision_parent_slots;
    int64_t* decision_mate_slots;
    int32_t* decision_adjacent_counts;
    int32_t* decision_reason_codes;
    int64_t* decision_count;
    int32_t* planned_repro_cooldown;
    int64_t* birth_parent_ids;
    int64_t* birth_child_ids;
    int64_t* birth_count;
    bool* mutation_occurred;
    float* mutation_delta;
    int64_t* spawn_draws;
    int64_t* spawn_semantic_draws;
    int64_t* runner_draws;
    int64_t* runner_semantic_draws;
    int64_t* active_agent_ids;
    int32_t* active_agent_slots;
    int64_t* active_count;
    int64_t* survivor_agent_ids;
    int32_t* agent_age;
    int32_t* agent_survival_steps;
    double* policy_energy_observation;
    float* drives;
    float* drive_inertia;
    float* drive_noise;
    float* repro_threshold;
    float* chemistry;
    float* genome_repro_threshold;
    float* genome_aggression;
    float* genome_action_temperature;
    float* genome_drive_inertia;
    float* genome_drive_noise;
    float* genome_drive_logit_gain;
    float* genome_drive_setpoints;
    float* pre_energy;
    float* pre_health;
    float* pre_injury;
    int32_t* pre_survival_steps;
    int32_t* event_type;
    int64_t* event_generation;
    int64_t* event_tick;
    int64_t* event_actor;
    int64_t* event_target;
    float* event_value;
    int32_t* typed_event_type;
    int64_t* typed_generation;
    int64_t* typed_tick;
    int64_t* typed_sequence;
    int64_t* typed_actor;
    int64_t* typed_target;
    int64_t* typed_actor_slot;
    int64_t* typed_target_slot;
    int64_t* typed_parent;
    int64_t* typed_child;
    int64_t* typed_founder;
    int32_t* typed_depth;
    int32_t* typed_subtype;
    int32_t* typed_reason;
    int8_t* typed_success;
    float* typed_value;
    int64_t* event_active_count;
    int32_t* event_status;
    bool* policy_use_gru;
    int64_t batch;
    int64_t capacity;
    int64_t grid_size;
    int64_t event_capacity;
};

__device__ __forceinline__ bool all_ok(const Stage& stage, const int64_t batch) {
    for (int64_t world = 0; world < batch; ++world) {
        if (stage.status[world] != kOk) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ double fertility_scale(
    const int64_t age_ticks, const Config& config) {
    if (config.life_history_mode == 0) {
        return 1.0;
    }
    if (age_ticks < config.maturity_start_ticks) {
        return 0.0;
    }
    double scale = 1.0;
    if (config.maturity_rise_ticks > 0) {
        scale = fmin(1.0, static_cast<double>(
            age_ticks - config.maturity_start_ticks + 1) / config.maturity_rise_ticks);
    }
    if (config.fertility_decline_start_ticks > 0
        && age_ticks >= config.fertility_decline_start_ticks) {
        if (config.fertility_decline_ticks <= 0) {
            scale = fmin(scale, config.elder_fertility_floor);
        } else {
            const double decline = fmin(1.0, static_cast<double>(
                age_ticks - config.fertility_decline_start_ticks + 1)
                / config.fertility_decline_ticks);
            scale = fmin(scale, 1.0 - decline * (1.0 - config.elder_fertility_floor));
        }
    }
    return fmax(0.0, fmin(1.0, scale));
}

__global__ void preflight_plan_kernel(State state, Stage stage, Config config) {
    if (threadIdx.x != 0) {
        return;
    }
    const int64_t world = static_cast<int64_t>(blockIdx.x);
    if (world >= state.batch) {
        return;
    }
    const int64_t offset = world * state.capacity;
    for (int64_t row = 0; row < state.capacity; ++row) {
        stage.parent_slots[offset + row] = -1;
        stage.child_slots[offset + row] = -1;
        stage.child_y[offset + row] = -1;
        stage.child_x[offset + row] = -1;
        stage.child_energy[offset + row] = 0.0f;
        stage.decision_parent_slots[offset + row] = -1;
        stage.decision_adjacent_counts[offset + row] = 0;
        stage.decision_reason_codes[offset + row] = -1;
        stage.planned_repro_cooldown[offset + row] = state.repro_cooldown[offset + row];
        stage.child_agent_ids[offset + row] = -1;
    }
    stage.planned_count[world] = 0;
    stage.decision_count[world] = 0;
    stage.spawn_draws[world] = 0;
    stage.spawn_semantic_draws[world] = 0;
    stage.queue_cursor[world] = 0;
    stage.status[world] = kOk;

    if (config.max_births < 0 || config.max_births > state.capacity
        || config.parent_energy_share < 0.0 || config.parent_energy_share > 1.0
        || config.minimum_energy_buffer < 0.0 || config.start_health < 0.0f
        || config.life_history_mode < 0
        || config.life_history_mode > 1) {
        stage.status[world] = kInvalidPlan;
        return;
    }
    for (int64_t index = 0; index < state.capacity * 3; ++index) {
        const double value = stage.spawn_uniforms[world * state.capacity * 3 + index];
        if (!isfinite(value) || value < 0.0 || value > 1.0) {
            stage.status[world] = kSpawnTapeInvalid;
            return;
        }
    }
    const int64_t queue_count = state.free_count[world];
    if (queue_count < 0 || queue_count > state.capacity) {
        stage.status[world] = kInvalidQueue;
        return;
    }
    const int64_t grid_offset = world * state.grid_size * state.grid_size;
    for (int64_t cell = 0; cell < state.grid_size * state.grid_size; ++cell) {
        const int64_t occupant = state.occupancy_agent[grid_offset + cell];
        const int64_t slot = state.occupancy_slot[grid_offset + cell];
        const int32_t patch = state.patch_agent[grid_offset + cell];
        if (occupant < 0) {
            if (slot != -1 || patch != -1) {
                stage.status[world] = kInvalidChild;
                return;
            }
        } else if (slot < 0 || slot >= state.capacity
            || !state.alive[offset + slot] || state.agent_id[offset + slot] != occupant
            || patch != static_cast<int32_t>(occupant)) {
            stage.status[world] = kInvalidChild;
            return;
        }
    }

    int64_t active_rows = 0;
    for (int64_t slot = 0; slot < state.capacity; ++slot) {
        active_rows += state.alive[offset + slot] && state.agent_id[offset + slot] >= 0;
    }
    int64_t planned = 0;
    int64_t decisions = 0;
    int64_t queue_cursor = 0;
    int64_t previous_agent_id = INT64_MIN;
    int64_t previous_slot = -1;
    for (int64_t decision = 0; decision < active_rows && planned < config.max_births;
         ++decision) {
        int64_t parent = -1;
        int64_t parent_id = INT64_MAX;
        for (int64_t slot = 0; slot < state.capacity; ++slot) {
            if (!state.alive[offset + slot] || state.agent_id[offset + slot] < 0) {
                continue;
            }
            const int64_t candidate_id = state.agent_id[offset + slot];
            const bool follows = candidate_id > previous_agent_id
                || (candidate_id == previous_agent_id && slot > previous_slot);
            if (follows && (candidate_id < parent_id
                    || (candidate_id == parent_id && slot < parent))) {
                parent = slot;
                parent_id = candidate_id;
            }
        }
        if (parent < 0) {
            stage.status[world] = kInvalidParent;
            return;
        }
        previous_agent_id = parent_id;
        previous_slot = parent;
        stage.decision_parent_slots[offset + decisions] = parent;
        ++decisions;
        int32_t reason = kAccepted;
        const float health = state.health[offset + parent];
        double parent_energy = state.policy_energy_observation[offset + parent];
        if (!isfinite(parent_energy)) {
            parent_energy = state.energy[offset + parent];
        }
        if (health <= 0.0f) {
            reason = kHealth;
        } else {
            const double fertility = fertility_scale(state.survival_steps[offset + parent], config);
            if (fertility <= 0.0) {
                reason = kOpportunity;
            } else if (stage.planned_repro_cooldown[offset + parent] > 0) {
                --stage.planned_repro_cooldown[offset + parent];
            }
            if (reason == kAccepted && stage.planned_repro_cooldown[offset + parent] != 0) {
                reason = kCooldown;
            } else if (reason == kAccepted
                && parent_energy < state.repro_threshold[offset + parent] / fertility) {
                reason = kNoEnergy;
            }
        }

        int32_t candidate_y[4];
        int32_t candidate_x[4];
        int32_t candidate_count = 0;
        if (reason == kAccepted) {
            const int64_t y = state.pos_y[offset + parent];
            const int64_t x = state.pos_x[offset + parent];
            constexpr int32_t dy[4] = {-1, 1, 0, 0};
            constexpr int32_t dx[4] = {0, 0, -1, 1};
            const int64_t grid_offset = world * state.grid_size * state.grid_size;
            for (int32_t direction = 0; direction < 4; ++direction) {
                const int64_t ny = y + dy[direction];
                const int64_t nx = x + dx[direction];
                const int64_t cell = grid_offset + ny * state.grid_size + nx;
                if (ny >= 0 && ny < state.grid_size && nx >= 0 && nx < state.grid_size
                    && state.occupancy_agent[cell] == -1
                    && state.occupancy_slot[cell] == -1
                    && state.patch_agent[cell] == -1) {
                    candidate_y[candidate_count] = static_cast<int32_t>(ny);
                    candidate_x[candidate_count] = static_cast<int32_t>(nx);
                    ++candidate_count;
                }
            }
            stage.spawn_semantic_draws[world] += candidate_count;
            for (int32_t index = candidate_count - 1; index > 0; --index) {
                const int64_t tape_index = world * state.capacity * 3
                    + (decisions - 1) * 3 + (candidate_count - 1 - index);
                const int32_t selected = min(
                    index, static_cast<int32_t>(floor(stage.spawn_uniforms[tape_index]
                        * static_cast<double>(index + 1))));
                const int32_t swap_y = candidate_y[index];
                const int32_t swap_x = candidate_x[index];
                candidate_y[index] = candidate_y[selected];
                candidate_x[index] = candidate_x[selected];
                candidate_y[selected] = swap_y;
                candidate_x[selected] = swap_x;
                ++stage.spawn_draws[world];
            }
            if (candidate_count == 0) {
                reason = kNoSpace;
            } else {
                for (int64_t prior = 0; prior < planned; ++prior) {
                    if (stage.child_y[offset + prior] == candidate_y[0]
                        && stage.child_x[offset + prior] == candidate_x[0]) {
                        reason = kNoSpace;
                    }
                }
            }
        }
        int64_t child_slot = -1;
        if (reason == kAccepted) {
            const double child_energy = parent_energy * config.parent_energy_share;
            if (parent_energy - child_energy < config.minimum_energy_buffer) {
                reason = kBuffer;
            } else {
                while (queue_cursor < queue_count) {
                    const int64_t candidate_slot = state.free_queue[offset + queue_cursor++];
                    bool used = false;
                    for (int64_t prior = 0; prior < planned; ++prior) {
                        used |= stage.child_slots[offset + prior] == candidate_slot;
                    }
                    if (candidate_slot >= 0 && candidate_slot < state.capacity && !used
                        && !state.alive[offset + candidate_slot]
                        && state.agent_id[offset + candidate_slot] < 0) {
                        child_slot = candidate_slot;
                        break;
                    }
                }
                if (child_slot < 0) {
                    reason = kNoFreeSlot;
                }
            }
        }
        stage.decision_reason_codes[offset + decisions - 1] = reason;
        if (reason == kAccepted) {
            if (state.policy_use_gru == nullptr) {
                stage.status[world] = kInvalidParent;
                return;
            }
            stage.parent_slots[offset + planned] = static_cast<int32_t>(parent);
            stage.child_slots[offset + planned] = static_cast<int32_t>(child_slot);
            stage.child_y[offset + planned] = candidate_y[0];
            stage.child_x[offset + planned] = candidate_x[0];
            stage.child_energy[offset + planned] = static_cast<float>(
                parent_energy * config.parent_energy_share);
            ++planned;
        }
        if (reason == kNoFreeSlot) {
            break;
        }
    }
    stage.planned_count[world] = planned;
    stage.decision_count[world] = decisions;
    stage.queue_cursor[world] = queue_cursor;
    if (planned > 0 && config.max_energy <= 0.0f) {
        stage.status[world] = kInvalidPlan;
        return;
    }

    int64_t next_id = state.next_agent_id[world];
    for (int64_t slot = 0; slot < state.capacity; ++slot) {
        next_id = max(next_id, state.agent_id[offset + slot] + 1);
    }
    const bool identity_invalid = config.identity_mode == kIdentitySortedMonotonic
        ? (next_id < 0 || planned > INT64_MAX - next_id)
        : (next_id < 0 || next_id + planned > state.capacity);
    if (identity_invalid) {
        stage.status[world] = kIdentityCapacity;
        return;
    }
    for (int64_t birth = 0; birth < planned; ++birth) {
        const int64_t parent = stage.parent_slots[offset + birth];
        if (parent < 0 || parent >= state.capacity || !state.alive[offset + parent]
            || state.agent_id[offset + parent] < 0
            || state.policy_use_gru[offset + parent]) {
            stage.status[world] = state.policy_use_gru[offset + parent]
                ? kUnsupportedGru : kInvalidParent;
            return;
        }
        if (birth > 0 && state.agent_id[offset + parent]
            <= state.agent_id[offset + stage.parent_slots[offset + birth - 1]]) {
            stage.status[world] = kInvalidParent;
            return;
        }
        const int64_t child = stage.child_slots[offset + birth];
        if (child < 0 || child >= state.capacity || state.alive[offset + child]
            || state.agent_id[offset + child] >= 0) {
            stage.status[world] = kInvalidChild;
            return;
        }
        stage.child_agent_ids[offset + birth] = next_id + birth;
        if (config.identity_mode == kIdentityLegacyDense
            && state.agent_id_to_slot[offset + next_id + birth] != -1) {
            stage.status[world] = kIdentityCapacity;
            return;
        }
    }
    int64_t event_rows = 0;
    int64_t accepted = 0;
    for (int64_t decision = 0; decision < decisions; ++decision) {
        const int32_t reason = stage.decision_reason_codes[offset + decision];
        if (reason == kAccepted) {
            ++accepted;
            ++event_rows;
        } else if ((reason >= 1 && reason <= 3) || (reason >= 5 && reason <= 6)) {
            ++event_rows;
        } else if (!(reason == 4 || (reason >= 7 && reason <= 11))) {
            stage.status[world] = kEventSemantics;
            return;
        }
    }
    if (accepted != planned) {
        stage.status[world] = kEventSemantics;
        return;
    }
    const int64_t cursor = state.event_active_count[world];
    if (stage.event_sequence[world] != cursor) {
        stage.status[world] = kEventSemantics;
        return;
    }
    if (state.event_status[world] == 1 || cursor < 0 || cursor > state.event_capacity
        || event_rows > state.event_capacity - cursor) {
        stage.status[world] = kEventCapacity;
        return;
    }
    if (state.event_status[world] == 2) {
        stage.status[world] = kEventSemantics;
    }
}

template <typename T>
__global__ void inherit_rows_kernel(
    T* values, int64_t row_width, Stage stage, int64_t batch, int64_t capacity) {
    if (threadIdx.x != 0 || !all_ok(stage, batch)) {
        return;
    }
    const int64_t world = static_cast<int64_t>(blockIdx.x);
    if (world >= batch) return;
    const int64_t offset = world * capacity;
    for (int64_t birth = 0; birth < stage.planned_count[world]; ++birth) {
        const int64_t parent = stage.parent_slots[offset + birth];
        const int64_t child = stage.child_slots[offset + birth];
        for (int64_t element = 0; element < row_width; ++element) {
            values[(offset + child) * row_width + element] =
                values[(offset + parent) * row_width + element];
        }
    }
}

template <typename T>
__global__ void fill_rows_kernel(
    T* values, int64_t row_width, T fill, Stage stage, int64_t batch, int64_t capacity) {
    if (threadIdx.x != 0 || !all_ok(stage, batch)) {
        return;
    }
    const int64_t world = static_cast<int64_t>(blockIdx.x);
    if (world >= batch) return;
    const int64_t offset = world * capacity;
    for (int64_t birth = 0; birth < stage.planned_count[world]; ++birth) {
        const int64_t child = stage.child_slots[offset + birth];
        for (int64_t element = 0; element < row_width; ++element) {
            values[(offset + child) * row_width + element] = fill;
        }
    }
}

__device__ void set_event(
    State state, int64_t world, int64_t row, int32_t event_type,
    int64_t actor, int64_t target, int64_t actor_slot, int64_t target_slot,
    int64_t parent, int64_t child, int64_t founder, int32_t depth,
    int32_t reason, int8_t success, float value, Config config) {
    const int64_t index = world * state.event_capacity + row;
    state.event_type[index] = event_type;
    state.event_generation[index] = config_generation(config, world);
    state.event_tick[index] = config_tick(config, world);
    state.event_actor[index] = actor;
    state.event_target[index] = target;
    state.event_value[index] = value;
    state.typed_event_type[index] = event_type;
    state.typed_generation[index] = config_generation(config, world);
    state.typed_tick[index] = config_tick(config, world);
    state.typed_sequence[index] = row;
    state.typed_actor[index] = actor;
    state.typed_target[index] = target;
    state.typed_actor_slot[index] = actor_slot;
    state.typed_target_slot[index] = target_slot;
    state.typed_parent[index] = parent;
    state.typed_child[index] = child;
    state.typed_founder[index] = founder;
    state.typed_depth[index] = depth;
    state.typed_subtype[index] = event_type;
    state.typed_reason[index] = reason;
    state.typed_success[index] = success;
    state.typed_value[index] = value;
}

__global__ void commit_kernel(State state, Stage stage, Config config) {
    if (threadIdx.x != 0 || !all_ok(stage, state.batch)) {
        return;
    }
    const int64_t world = static_cast<int64_t>(blockIdx.x);
    if (world >= state.batch) return;
    const int64_t offset = world * state.capacity;
    const int64_t planned = stage.planned_count[world];
    for (int64_t row = 0; row < state.capacity; ++row) {
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
    state.planned_count[world] = planned;
    state.plan_status[world] = 0;
    state.decision_count[world] = stage.decision_count[world];
    state.birth_count[world] = planned;
    state.spawn_draws[world] = stage.spawn_draws[world];
    state.spawn_semantic_draws[world] = stage.spawn_semantic_draws[world];
    state.runner_draws[world] = 0;
    state.runner_semantic_draws[world] = 0;

    const int64_t grid_offset = world * state.grid_size * state.grid_size;
    for (int64_t birth = 0; birth < planned; ++birth) {
        const int64_t parent = stage.parent_slots[offset + birth];
        const int64_t child = stage.child_slots[offset + birth];
        const int64_t parent_id = state.agent_id[offset + parent];
        const int64_t child_id = stage.child_agent_ids[offset + birth];
        double parent_energy = state.policy_energy_observation[offset + parent];
        if (!isfinite(parent_energy)) parent_energy = state.energy[offset + parent];
        const float child_energy = stage.child_energy[offset + birth];
        const double remaining_energy = parent_energy - child_energy;
        state.energy[offset + parent] = static_cast<float>(remaining_energy);
        state.policy_energy_observation[offset + parent] = remaining_energy;
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
        if (child_id >= 0 && child_id < state.capacity) {
            state.agent_id_to_slot[offset + child_id] = static_cast<int32_t>(child);
        }
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
        const int64_t parent_founder = state.lineage_founder[offset + parent];
        state.lineage_founder[offset + child] = parent_founder < 0 ? parent_id : parent_founder;
        state.lineage_depth[offset + child] = state.lineage_depth[offset + parent] + 1;
        state.lineage_birth_generation[offset + child] =
            config_generation(config, world);
        state.lineage_birth_tick[offset + child] = config_tick(config, world);
        state.lineage_parent[offset + child] = parent_id;
        const int32_t parent_patch = state.patch_lineage[offset + parent];
        state.patch_lineage[offset + child] = parent_patch == 0
            ? static_cast<int32_t>(parent_id) : parent_patch;
        state.scar[offset + child] = 0.0f;
        state.repro_cooldown[offset + child] = 0;
        state.offspring_count[offset + child] = 0;
        state.first_repro_step[offset + child] = -1;
    }
    if (planned > 0) {
        int64_t remaining = 0;
        const int64_t old_count = state.free_count[world];
        for (int64_t row = stage.queue_cursor[world]; row < old_count; ++row) {
            const int64_t slot = state.free_queue[offset + row];
            if (slot >= 0 && slot < state.capacity && !state.alive[offset + slot]
                && state.agent_id[offset + slot] < 0) {
                state.free_queue[offset + remaining++] = slot;
            }
        }
        for (int64_t row = remaining; row < state.capacity; ++row) {
            state.free_queue[offset + row] = -1;
        }
        state.free_count[world] = remaining;
        state.next_agent_id[world] = stage.child_agent_ids[offset + planned - 1] + 1;
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
    }

    int64_t output = state.event_active_count[world];
    int64_t success = 0;
    for (int64_t decision = 0; decision < stage.decision_count[world]; ++decision) {
        const int64_t parent_slot = stage.decision_parent_slots[offset + decision];
        const int64_t parent_id = state.agent_id[offset + parent_slot];
        const int32_t reason = stage.decision_reason_codes[offset + decision];
        if ((reason >= 1 && reason <= 3) || (reason >= 5 && reason <= 6)) {
            int64_t founder = state.lineage_founder[offset + parent_slot];
            if (founder < 0) founder = parent_id;
            set_event(state, world, output++, kBirthFailedEvent, parent_id, kSentinel,
                parent_slot, kSentinel, kSentinel, kSentinel, founder,
                state.lineage_depth[offset + parent_slot], reason, 0,
                static_cast<float>(reason), config);
        } else if (reason == kAccepted) {
            const int64_t child_slot = stage.child_slots[offset + success];
            const int64_t child_id = stage.child_agent_ids[offset + success];
            set_event(state, world, output++, kBirthCommittedEvent, parent_id, child_id,
                parent_slot, child_slot, parent_id, child_id,
                state.lineage_founder[offset + child_slot],
                state.lineage_depth[offset + child_slot], -1, 1,
                stage.child_energy[offset + success], config);
            ++success;
        }
    }
    state.event_active_count[world] = output;
    stage.event_sequence[world] = output;
    state.event_status[world] = 0;
}

template <typename T>
void launch_inherit(torch::Tensor tensor, Stage stage, int64_t batch, int64_t capacity,
                    cudaStream_t stream) {
    const int64_t width = tensor.numel() / (batch * capacity);
    inherit_rows_kernel<<<static_cast<unsigned int>(batch), 1, 0, stream>>>(
        tensor.data_ptr<T>(), width, stage, batch, capacity);
}

template <typename T>
void launch_fill(torch::Tensor tensor, T value, Stage stage, int64_t batch, int64_t capacity,
                 cudaStream_t stream) {
    const int64_t width = tensor.numel() / (batch * capacity);
    fill_rows_kernel<<<static_cast<unsigned int>(batch), 1, 0, stream>>>(
        tensor.data_ptr<T>(), width, value, stage, batch, capacity);
}

void dispatch_inherit(torch::Tensor tensor, Stage stage, int64_t batch, int64_t capacity,
                      cudaStream_t stream) {
    switch (tensor.scalar_type()) {
        case torch::kBool: launch_inherit<bool>(tensor, stage, batch, capacity, stream); break;
        case torch::kInt32: launch_inherit<int32_t>(tensor, stage, batch, capacity, stream); break;
        case torch::kInt64: launch_inherit<int64_t>(tensor, stage, batch, capacity, stream); break;
        case torch::kFloat32: launch_inherit<float>(tensor, stage, batch, capacity, stream); break;
        case torch::kFloat64: launch_inherit<double>(tensor, stage, batch, capacity, stream); break;
        default: TORCH_CHECK(false, "unsupported C4 R2 inherit dtype");
    }
}

void dispatch_fill(torch::Tensor tensor, double value, Stage stage, int64_t batch,
                   int64_t capacity, cudaStream_t stream) {
    switch (tensor.scalar_type()) {
        case torch::kBool: launch_fill<bool>(tensor, value != 0.0, stage, batch, capacity, stream); break;
        case torch::kUInt8: launch_fill<uint8_t>(tensor, static_cast<uint8_t>(value), stage, batch, capacity, stream); break;
        case torch::kInt32: launch_fill<int32_t>(tensor, static_cast<int32_t>(value), stage, batch, capacity, stream); break;
        case torch::kInt64: launch_fill<int64_t>(tensor, static_cast<int64_t>(value), stage, batch, capacity, stream); break;
        case torch::kFloat32: launch_fill<float>(tensor, static_cast<float>(value), stage, batch, capacity, stream); break;
        case torch::kFloat64: launch_fill<double>(tensor, value, stage, batch, capacity, stream); break;
        default: TORCH_CHECK(false, "unsupported C4 R2 reset dtype");
    }
}

State make_state(const std::vector<torch::Tensor>& p, int64_t batch, int64_t capacity,
                 int64_t grid_size, int64_t event_capacity) {
    return State{
        p[0].data_ptr<bool>(), p[1].data_ptr<int64_t>(), p[2].data_ptr<int64_t>(),
        p[3].data_ptr<int64_t>(), p[4].data_ptr<float>(), p[5].data_ptr<float>(),
        p[6].data_ptr<int64_t>(), p[7].data_ptr<int64_t>(), p[8].data_ptr<double>(),
        p[9].data_ptr<int64_t>(), p[10].data_ptr<int64_t>(), p[11].data_ptr<int32_t>(),
        p[12].data_ptr<float>(), p[13].data_ptr<float>(), p[14].data_ptr<float>(),
        p[15].data_ptr<float>(), p[16].data_ptr<int32_t>(), p[17].data_ptr<int64_t>(),
        p[18].data_ptr<int64_t>(), p[19].data_ptr<int64_t>(), p[20].data_ptr<int64_t>(),
        p[21].data_ptr<int32_t>(), p[22].data_ptr<int32_t>(), p[23].data_ptr<int32_t>(),
        p[24].data_ptr<int32_t>(), p[25].data_ptr<float>(), p[26].data_ptr<int32_t>(),
        p[27].data_ptr<int64_t>(), p[28].data_ptr<int32_t>(), p[29].data_ptr<float>(),
        p[30].data_ptr<int32_t>(), p[31].data_ptr<int32_t>(), p[32].data_ptr<int32_t>(),
        p[33].data_ptr<int64_t>(), p[34].data_ptr<int64_t>(), p[35].data_ptr<int64_t>(),
        p[36].data_ptr<int32_t>(), p[37].data_ptr<int64_t>(), p[38].data_ptr<int32_t>(),
        p[39].data_ptr<int64_t>(), p[40].data_ptr<int64_t>(), p[41].data_ptr<int32_t>(),
        p[42].data_ptr<int32_t>(), p[43].data_ptr<int64_t>(), p[44].data_ptr<int32_t>(),
        p[45].data_ptr<int64_t>(), p[46].data_ptr<int64_t>(), p[47].data_ptr<int64_t>(),
        p[48].data_ptr<bool>(), p[49].data_ptr<float>(), p[50].data_ptr<int64_t>(),
        p[51].data_ptr<int64_t>(), p[52].data_ptr<int64_t>(), p[53].data_ptr<int64_t>(),
        p[54].data_ptr<int64_t>(), p[55].data_ptr<int32_t>(), p[56].data_ptr<int64_t>(),
        p[57].data_ptr<int64_t>(), p[58].data_ptr<int32_t>(), p[59].data_ptr<int32_t>(),
        p[60].data_ptr<double>(), p[61].data_ptr<float>(), p[62].data_ptr<float>(),
        p[63].data_ptr<float>(), p[64].data_ptr<float>(), p[65].data_ptr<float>(),
        p[66].data_ptr<float>(), p[67].data_ptr<float>(), p[68].data_ptr<float>(),
        p[69].data_ptr<float>(), p[70].data_ptr<float>(), p[71].data_ptr<float>(),
        p[72].data_ptr<float>(), p[73].data_ptr<float>(), p[74].data_ptr<float>(),
        p[75].data_ptr<float>(), p[76].data_ptr<int32_t>(), p[77].data_ptr<int32_t>(),
        p[78].data_ptr<int64_t>(), p[79].data_ptr<int64_t>(), p[80].data_ptr<int64_t>(),
        p[81].data_ptr<int64_t>(), p[82].data_ptr<float>(), p[83].data_ptr<int32_t>(),
        p[84].data_ptr<int64_t>(), p[85].data_ptr<int64_t>(), p[86].data_ptr<int64_t>(),
        p[87].data_ptr<int64_t>(), p[88].data_ptr<int64_t>(), p[89].data_ptr<int64_t>(),
        p[90].data_ptr<int64_t>(), p[91].data_ptr<int64_t>(), p[92].data_ptr<int64_t>(),
        p[93].data_ptr<int64_t>(), p[94].data_ptr<int32_t>(), p[95].data_ptr<int32_t>(),
        p[96].data_ptr<int32_t>(), p[97].data_ptr<int8_t>(), p[98].data_ptr<float>(),
        p[99].data_ptr<int64_t>(), p[100].data_ptr<int32_t>(), nullptr,
        batch, capacity, grid_size, event_capacity};
}

}  // namespace

void tensor_world_c4_resident_birth_r2_cuda(
    std::vector<torch::Tensor> primary,
    std::vector<torch::Tensor> inherit,
    std::vector<torch::Tensor> zero,
    std::vector<torch::Tensor> minus_one,
    std::vector<torch::Tensor> half,
    std::vector<torch::Tensor> one,
    std::vector<torch::Tensor> staging,
    int64_t max_births,
    double parent_energy_share,
    double minimum_energy_buffer,
    int64_t cooldown_steps,
    double start_health,
    double max_energy,
    int64_t life_history_mode,
    int64_t maturity_start_ticks,
    int64_t maturity_rise_ticks,
    int64_t fertility_decline_start_ticks,
    int64_t fertility_decline_ticks,
    double elder_fertility_floor,
    int64_t generation,
    int64_t tick) {
    TORCH_CHECK(primary.size() == 101, "C4 R2 primary ABI is incomplete");
    TORCH_CHECK(staging.size() == 18, "C4 R2 staging ABI is incomplete");
    TORCH_CHECK(primary[0].dim() == 2, "C4 R2 alive state must be rank two");
    const int64_t batch = primary[0].size(0);
    const int64_t capacity = primary[0].size(1);
    TORCH_CHECK(batch > 0 && capacity > 0, "C4 R2 dimensions must be positive");
    TORCH_CHECK(primary[9].dim() == 3 && primary[9].size(0) == batch
        && primary[9].size(1) == primary[9].size(2), "C4 R2 grid is invalid");
    const int64_t grid_size = primary[9].size(1);
    TORCH_CHECK(primary[77].dim() == 2 && primary[77].size(0) == batch,
        "C4 R2 event tape is invalid");
    const int64_t event_capacity = primary[77].size(1);
    const auto device = primary[0].device();
    TORCH_CHECK(device.is_cuda(), "C4 R2 requires CUDA tensors");
    c10::cuda::CUDAGuard guard(device);
    for (const auto& group : {primary, inherit, zero, minus_one, half, one, staging}) {
        for (const auto& tensor : group) {
            TORCH_CHECK(tensor.is_cuda() && tensor.device() == device && tensor.is_contiguous(),
                "C4 R2 tensors must be contiguous on one CUDA device");
        }
    }
    for (const auto& group : {inherit, zero, minus_one, half, one}) {
        for (const auto& tensor : group) {
            TORCH_CHECK(tensor.dim() >= 2 && tensor.size(0) == batch
                && tensor.size(1) == capacity, "C4 R2 reset tensor shape is invalid");
        }
    }
    TORCH_CHECK(staging[0].sizes() == torch::IntArrayRef({batch, capacity, 3}),
        "C4 R2 spawn tape shape is invalid");
    TORCH_CHECK(cooldown_steps >= 0 && generation >= 0 && tick >= 0,
        "C4 R2 scalar configuration is invalid");

    Stage stage{
        staging[0].data_ptr<double>(), staging[1].data_ptr<int32_t>(),
        staging[2].data_ptr<int32_t>(), staging[3].data_ptr<int32_t>(),
        staging[4].data_ptr<int32_t>(), staging[5].data_ptr<float>(),
        staging[6].data_ptr<int64_t>(), staging[7].data_ptr<int64_t>(),
        staging[8].data_ptr<int32_t>(), staging[9].data_ptr<int32_t>(),
        staging[10].data_ptr<int64_t>(), staging[11].data_ptr<int32_t>(),
        staging[12].data_ptr<int64_t>(), staging[13].data_ptr<int64_t>(),
        staging[14].data_ptr<int64_t>(), staging[15].data_ptr<int64_t>(),
        staging[16].data_ptr<int64_t>(), staging[17].data_ptr<int32_t>()};
    State state = make_state(primary, batch, capacity, grid_size, event_capacity);
    // policy_use_gru is inherited tensor index 11 by the fixed source inventory.
    state.policy_use_gru = inherit[11].data_ptr<bool>();
    Config config{
        max_births, parent_energy_share, minimum_energy_buffer,
        static_cast<int32_t>(cooldown_steps), static_cast<float>(start_health),
        static_cast<float>(max_energy), static_cast<int32_t>(life_history_mode),
        static_cast<int32_t>(maturity_start_ticks), static_cast<int32_t>(maturity_rise_ticks),
        static_cast<int32_t>(fertility_decline_start_ticks),
        static_cast<int32_t>(fertility_decline_ticks), elder_fertility_floor,
        generation, tick};
    const auto stream = at::cuda::getCurrentCUDAStream(device.index());
    preflight_plan_kernel<<<static_cast<unsigned int>(batch), 1, 0, stream>>>(
        state, stage, config);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    for (auto& tensor : inherit) dispatch_inherit(tensor, stage, batch, capacity, stream);
    for (auto& tensor : zero) dispatch_fill(tensor, 0.0, stage, batch, capacity, stream);
    for (auto& tensor : minus_one) dispatch_fill(tensor, -1.0, stage, batch, capacity, stream);
    for (auto& tensor : half) dispatch_fill(tensor, 0.5, stage, batch, capacity, stream);
    for (auto& tensor : one) dispatch_fill(tensor, 1.0, stage, batch, capacity, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    commit_kernel<<<static_cast<unsigned int>(batch), 1, 0, stream>>>(state, stage, config);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
