#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

constexpr int32_t STATUS_OK = 0;
constexpr int32_t STATUS_INVALID_PENDING_COUNT = 1;
constexpr int32_t STATUS_PENDING_CAPACITY_EXCEEDED = 2;
constexpr int32_t STATUS_INVALID_BIRTH_PLAN = 10;
constexpr int32_t STATUS_INVALID_PARENT_ORDER = 11;
constexpr int32_t STATUS_PARENT_NOT_VIABLE = 12;
constexpr int32_t STATUS_NO_FREE_SLOT = 13;
constexpr int32_t STATUS_AGENT_ID_CAPACITY = 14;
constexpr int32_t STATUS_SPAWN_RESERVATION = 15;
constexpr int32_t STATUS_INVALID_FREE_QUEUE_COUNT = 16;
constexpr int32_t STATUS_INVALID_SCALAR_BIRTH_PLAN = 20;
constexpr int32_t STATUS_INVALID_PYTHON_RANDOM_STATE = 21;
constexpr int32_t STATUS_FRESH_GAUSS_UNSUPPORTED = 22;
constexpr int32_t STATUS_INVALID_FREE_RELEASE_STATE = 30;
constexpr int32_t STATUS_FREE_RELEASE_CAPACITY = 31;
constexpr int32_t STATUS_FREE_RELEASE_CONTRADICTION = 32;

struct DeviceCpythonRandom {
    uint32_t* state;
    int64_t index;
    bool has_gauss_next;
    double gauss_next;
    const double* gauss_primary_tape;
    const double* gauss_secondary_tape;
    int64_t gauss_tape_capacity;
    int64_t raw_draws;

    __device__ __forceinline__ void twist() {
        constexpr uint32_t MATRIX_A = 0x9908b0dfU;
        constexpr uint32_t UPPER_MASK = 0x80000000U;
        constexpr uint32_t LOWER_MASK = 0x7fffffffU;
        for (int64_t kk = 0; kk < 624 - 397; ++kk) {
            const uint32_t y = (state[kk] & UPPER_MASK) | (state[kk + 1] & LOWER_MASK);
            state[kk] = state[kk + 397] ^ (y >> 1U) ^ ((y & 1U) ? MATRIX_A : 0U);
        }
        for (int64_t kk = 624 - 397; kk < 623; ++kk) {
            const uint32_t y = (state[kk] & UPPER_MASK) | (state[kk + 1] & LOWER_MASK);
            state[kk] = state[kk + (397 - 624)] ^ (y >> 1U) ^ ((y & 1U) ? MATRIX_A : 0U);
        }
        const uint32_t y = (state[623] & UPPER_MASK) | (state[0] & LOWER_MASK);
        state[623] = state[396] ^ (y >> 1U) ^ ((y & 1U) ? MATRIX_A : 0U);
        index = 0;
    }

    __device__ __forceinline__ uint32_t raw32() {
        if (index >= 624) {
            twist();
        }
        uint32_t y = state[index++];
        y ^= y >> 11U;
        y ^= (y << 7U) & 0x9d2c5680U;
        y ^= (y << 15U) & 0xefc60000U;
        y ^= y >> 18U;
        raw_draws += 1;
        return y;
    }

    __device__ __forceinline__ double random_double() {
        const uint32_t a = raw32() >> 5U;
        const uint32_t b = raw32() >> 6U;
        return (static_cast<double>(a) * 67108864.0 + static_cast<double>(b))
            * (1.0 / 9007199254740992.0);
    }

    __device__ __forceinline__ bool consume_gauss(
        const double sigma,
        double* value
    ) {
        if (has_gauss_next) {
            *value = gauss_next * sigma;
            has_gauss_next = false;
            gauss_next = 0.0;
            return true;
        }
        const int64_t tape_index = raw_draws;
        if (
            tape_index < 0 || tape_index >= gauss_tape_capacity
            || gauss_primary_tape == nullptr || gauss_secondary_tape == nullptr
        ) {
            return false;
        }
        (void)random_double();
        (void)random_double();
        *value = gauss_primary_tape[tape_index] * sigma;
        gauss_next = gauss_secondary_tape[tape_index];
        has_gauss_next = true;
        return true;
    }

    __device__ __forceinline__ uint64_t getrandbits(const int64_t bit_count) {
        if (bit_count == 0) {
            return 0;
        }
        if (bit_count <= 32) {
            return static_cast<uint64_t>(raw32() >> (32 - bit_count));
        }
        uint64_t value = 0;
        int64_t bits_filled = 0;
        int64_t remaining = bit_count;
        while (remaining > 0) {
            const int64_t take = remaining < 32 ? remaining : 32;
            uint32_t word = raw32();
            if (take < 32) {
                word >>= 32 - take;
            }
            value |= static_cast<uint64_t>(word) << bits_filled;
            bits_filled += take;
            remaining -= take;
        }
        return value;
    }

    __device__ __forceinline__ uint64_t randbelow(const uint64_t n) {
        int64_t bit_count = 0;
        for (uint64_t value = n; value != 0; value >>= 1U) {
            ++bit_count;
        }
        uint64_t value = getrandbits(bit_count);
        while (value >= n) {
            value = getrandbits(bit_count);
        }
        return value;
    }
};

__device__ __forceinline__ double clamp_double(
    const double value,
    const double lower,
    const double upper
) {
    return value < lower ? lower : (value > upper ? upper : value);
}

__device__ __forceinline__ double canon_genome_double_device(const double value) {
    if (!isfinite(value)) {
        return 0.0;
    }
    const float value32 = static_cast<float>(value);
    return nearbyint(static_cast<double>(value32) * 100000000.0) / 100000000.0;
}

__device__ __forceinline__ float canon_genome_float_device(const double value) {
    return static_cast<float>(canon_genome_double_device(value));
}

__device__ __forceinline__ int32_t snap_memory_dim_device(const int32_t value) {
    constexpr int32_t ladder[5] = {0, 8, 16, 32, 64};
    int32_t best = ladder[0];
    int32_t best_delta = abs(value - ladder[0]);
    for (int32_t index = 1; index < 5; ++index) {
        const int32_t delta = abs(value - ladder[index]);
        if (delta < best_delta) {
            best = ladder[index];
            best_delta = delta;
        }
    }
    return best;
}

__device__ __forceinline__ int32_t memory_index_device(const int32_t value) {
    if (value <= 0) return 0;
    if (value <= 8) return 1;
    if (value <= 16) return 2;
    if (value <= 32) return 3;
    return 4;
}

__device__ __forceinline__ int32_t snap_sensor_radius_device(
    const int32_t value,
    const int32_t sensor_radius_max
) {
    constexpr int32_t ladder[3] = {1, 2, 3};
    int32_t best = ladder[0];
    int32_t best_delta = abs(value - ladder[0]);
    for (int32_t index = 1; index < 3; ++index) {
        const int32_t delta = abs(value - ladder[index]);
        if (delta < best_delta) {
            best = ladder[index];
            best_delta = delta;
        }
    }
    if (sensor_radius_max > 0 && best > sensor_radius_max) {
        best = sensor_radius_max > 1 ? sensor_radius_max : 1;
    }
    return best > 1 ? best : 1;
}

__device__ bool scalar_birth_rng_exceeds_gauss_tape(
    DeviceCpythonRandom& rng,
    const float parent_mutation_rate,
    const float mate_mutation_rate
) {
    (void)rng.random_double();  // hidden-size parent choice
    const double base_mutation_rate = 0.5 * (
        canon_genome_double_device(parent_mutation_rate)
        + canon_genome_double_device(mate_mutation_rate)
    );
    const double scalar_mutation_rate = clamp_double(base_mutation_rate, 0.0, 1.0);
    for (int32_t gene = 0; gene < 8; ++gene) {
        if (rng.random_double() < scalar_mutation_rate) {
            double ignored = 0.0;
            if (!rng.consume_gauss(1.0, &ignored)) {
                return true;
            }
        }
    }
    double child_mutation_rate = base_mutation_rate;
    if (rng.random_double() < 0.2) {
        double delta = 0.0;
        if (!rng.consume_gauss(0.005, &delta)) {
            return true;
        }
        child_mutation_rate += delta;
    }
    child_mutation_rate = clamp_double(child_mutation_rate, 0.001, 0.20);
    const double memory_probability = clamp_double(0.25 * child_mutation_rate, 0.002, 0.05);
    if (rng.random_double() < memory_probability) {
        (void)rng.randbelow(2);
    }
    return false;
}

__device__ __forceinline__ double mutate_scalar_gene_cached_only(
    DeviceCpythonRandom& rng,
    double value,
    const double rate,
    const double sigma,
    const double lower,
    const double upper,
    int64_t* semantic_draws
) {
    *semantic_draws += 1;
    if (rng.random_double() < rate) {
        double delta = 0.0;
        const bool consumed = rng.consume_gauss(sigma, &delta);
        if (consumed) {
            value += delta;
            *semantic_draws += 1;
        }
    }
    return clamp_double(value, lower, upper);
}

__device__ __forceinline__ bool is_death_candidate(
    const int64_t slot,
    const int64_t* __restrict__ slot_to_agent_id,
    const bool* __restrict__ alive_mask,
    const float* __restrict__ energy,
    const float* __restrict__ health
) {
    return alive_mask[slot]
        && slot_to_agent_id[slot] >= 0
        && (health[slot] <= 0.0f || energy[slot] <= 0.0f);
}

__device__ __forceinline__ void lifecycle_death_cull_cell(
    int64_t* __restrict__ slot_to_agent_id,
    int32_t* __restrict__ agent_id_to_slot,
    bool* __restrict__ alive_mask,
    int32_t* __restrict__ pos_y,
    int32_t* __restrict__ pos_x,
    int32_t* __restrict__ agent_grid,
    float* __restrict__ energy,
    float* __restrict__ health,
    double* __restrict__ policy_energy_observation,
    int32_t* __restrict__ rest_streak,
    int64_t* __restrict__ action_cache_token,
    int32_t* __restrict__ action_cache_cursor,
    int32_t* __restrict__ action_cache_next,
    int64_t* __restrict__ pending_free_agent_ids,
    int64_t* __restrict__ pending_free_slots,
    int64_t* __restrict__ pending_free_count,
    int64_t* __restrict__ death_agent_ids,
    int64_t* __restrict__ death_slots,
    int32_t* __restrict__ death_reason_codes,
    int64_t* __restrict__ death_count,
    int64_t* __restrict__ alive_agent_ids_by_order,
    int64_t* __restrict__ alive_count,
    int32_t* __restrict__ status,
    const int64_t slot_count,
    const int64_t agent_id_capacity,
    const int64_t grid_height,
    const int64_t grid_width,
    const int64_t pending_capacity
) {
    const int64_t pending_before = pending_free_count[0];
    if (pending_before < 0 || pending_before > pending_capacity) {
        status[0] = STATUS_INVALID_PENDING_COUNT;
        return;
    }

    int64_t candidate_count = 0;
    for (int64_t slot = 0; slot < slot_count; ++slot) {
        candidate_count += is_death_candidate(
            slot,
            slot_to_agent_id,
            alive_mask,
            energy,
            health
        ) ? 1 : 0;
    }
    if (pending_before + candidate_count > pending_capacity) {
        status[0] = STATUS_PENDING_CAPACITY_EXCEEDED;
        return;
    }

    // The accepted owner applies action-time deaths in stable agent-ID order.
    // This conservative kernel preserves that scalar order before parallelizing.
    for (int64_t death_index = 0; death_index < candidate_count; ++death_index) {
        int64_t selected_slot = -1;
        int64_t selected_agent_id = INT64_MAX;
        for (int64_t slot = 0; slot < slot_count; ++slot) {
            if (!is_death_candidate(slot, slot_to_agent_id, alive_mask, energy, health)) {
                continue;
            }
            const int64_t agent_id = slot_to_agent_id[slot];
            if (
                agent_id < selected_agent_id
                || (agent_id == selected_agent_id && slot < selected_slot)
            ) {
                selected_agent_id = agent_id;
                selected_slot = slot;
            }
        }

        const int32_t reason_code = health[selected_slot] <= 0.0f ? 2 : 1;
        const int32_t y = pos_y[selected_slot];
        const int32_t x = pos_x[selected_slot];
        if (
            y >= 0 && y < grid_height
            && x >= 0 && x < grid_width
            && agent_grid[static_cast<int64_t>(y) * grid_width + x]
                == static_cast<int32_t>(selected_agent_id)
        ) {
            agent_grid[static_cast<int64_t>(y) * grid_width + x] = -1;
        }
        if (selected_agent_id >= 0 && selected_agent_id < agent_id_capacity) {
            agent_id_to_slot[selected_agent_id] = -1;
        }

        alive_mask[selected_slot] = false;
        slot_to_agent_id[selected_slot] = -1;
        pos_y[selected_slot] = -1;
        pos_x[selected_slot] = -1;
        energy[selected_slot] = 0.0f;
        health[selected_slot] = 0.0f;
        policy_energy_observation[selected_slot] = __longlong_as_double(
            static_cast<long long>(0x7ff8000000000000ULL)
        );
        rest_streak[selected_slot] = 0;
        action_cache_token[selected_slot] = -1;
        action_cache_cursor[selected_slot] = -1;
        action_cache_next[selected_slot] = -1;

        const int64_t pending_index = pending_before + death_index;
        pending_free_agent_ids[pending_index] = selected_agent_id;
        pending_free_slots[pending_index] = selected_slot;
        death_agent_ids[death_index] = selected_agent_id;
        death_slots[death_index] = selected_slot;
        death_reason_codes[death_index] = reason_code;
    }
    pending_free_count[0] = pending_before + candidate_count;
    death_count[0] = candidate_count;

    int64_t remaining = 0;
    int64_t previous_agent_id = INT64_MIN;
    int64_t previous_slot = -1;
    while (true) {
        int64_t selected_slot = -1;
        int64_t selected_agent_id = INT64_MAX;
        for (int64_t slot = 0; slot < slot_count; ++slot) {
            if (!alive_mask[slot] || slot_to_agent_id[slot] < 0) {
                continue;
            }
            const int64_t agent_id = slot_to_agent_id[slot];
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
        alive_agent_ids_by_order[remaining] = selected_agent_id;
        previous_agent_id = selected_agent_id;
        previous_slot = selected_slot;
        ++remaining;
    }
    alive_count[0] = remaining;
    status[0] = STATUS_OK;
}

__global__ void lifecycle_death_cull_kernel(
    int64_t* __restrict__ slot_to_agent_id,
    int32_t* __restrict__ agent_id_to_slot,
    bool* __restrict__ alive_mask,
    int32_t* __restrict__ pos_y,
    int32_t* __restrict__ pos_x,
    int32_t* __restrict__ agent_grid,
    float* __restrict__ energy,
    float* __restrict__ health,
    double* __restrict__ policy_energy_observation,
    int32_t* __restrict__ rest_streak,
    int64_t* __restrict__ action_cache_token,
    int32_t* __restrict__ action_cache_cursor,
    int32_t* __restrict__ action_cache_next,
    int64_t* __restrict__ pending_free_agent_ids,
    int64_t* __restrict__ pending_free_slots,
    int64_t* __restrict__ pending_free_count,
    int64_t* __restrict__ death_agent_ids,
    int64_t* __restrict__ death_slots,
    int32_t* __restrict__ death_reason_codes,
    int64_t* __restrict__ death_count,
    int64_t* __restrict__ alive_agent_ids_by_order,
    int64_t* __restrict__ alive_count,
    int32_t* __restrict__ status,
    const int64_t slot_count,
    const int64_t agent_id_capacity,
    const int64_t grid_height,
    const int64_t grid_width,
    const int64_t pending_capacity
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    lifecycle_death_cull_cell(
        slot_to_agent_id,
        agent_id_to_slot,
        alive_mask,
        pos_y,
        pos_x,
        agent_grid,
        energy,
        health,
        policy_energy_observation,
        rest_streak,
        action_cache_token,
        action_cache_cursor,
        action_cache_next,
        pending_free_agent_ids,
        pending_free_slots,
        pending_free_count,
        death_agent_ids,
        death_slots,
        death_reason_codes,
        death_count,
        alive_agent_ids_by_order,
        alive_count,
        status,
        slot_count,
        agent_id_capacity,
        grid_height,
        grid_width,
        pending_capacity
    );
}

__global__ void lifecycle_death_cull_batched_counted_kernel(
    int64_t* __restrict__ slot_to_agent_id,
    int32_t* __restrict__ agent_id_to_slot,
    bool* __restrict__ alive_mask,
    int32_t* __restrict__ pos_y,
    int32_t* __restrict__ pos_x,
    int32_t* __restrict__ agent_grid,
    float* __restrict__ energy,
    float* __restrict__ health,
    double* __restrict__ policy_energy_observation,
    int32_t* __restrict__ rest_streak,
    int64_t* __restrict__ action_cache_token,
    int32_t* __restrict__ action_cache_cursor,
    int32_t* __restrict__ action_cache_next,
    int64_t* __restrict__ pending_free_agent_ids,
    int64_t* __restrict__ pending_free_slots,
    int64_t* __restrict__ pending_free_count,
    int64_t* __restrict__ death_agent_ids,
    int64_t* __restrict__ death_slots,
    int32_t* __restrict__ death_reason_codes,
    int64_t* __restrict__ death_count,
    int64_t* __restrict__ alive_agent_ids_by_order,
    int64_t* __restrict__ alive_count,
    int32_t* __restrict__ status,
    const int64_t cell_count,
    const int64_t slot_count,
    const int64_t agent_id_capacity,
    const int64_t grid_height,
    const int64_t grid_width,
    const int64_t pending_capacity
) {
    const int64_t cell = static_cast<int64_t>(blockIdx.x);
    if (cell >= cell_count || threadIdx.x != 0) {
        return;
    }

    const int64_t slot_offset = cell * slot_count;
    const int64_t agent_id_offset = cell * agent_id_capacity;
    const int64_t grid_offset = cell * grid_height * grid_width;
    const int64_t pending_offset = cell * pending_capacity;
    lifecycle_death_cull_cell(
        slot_to_agent_id + slot_offset,
        agent_id_to_slot + agent_id_offset,
        alive_mask + slot_offset,
        pos_y + slot_offset,
        pos_x + slot_offset,
        agent_grid + grid_offset,
        energy + slot_offset,
        health + slot_offset,
        policy_energy_observation + slot_offset,
        rest_streak + slot_offset,
        action_cache_token + slot_offset,
        action_cache_cursor + slot_offset,
        action_cache_next + slot_offset,
        pending_free_agent_ids + pending_offset,
        pending_free_slots + pending_offset,
        pending_free_count + cell,
        death_agent_ids + slot_offset,
        death_slots + slot_offset,
        death_reason_codes + slot_offset,
        death_count + cell,
        alive_agent_ids_by_order + slot_offset,
        alive_count + cell,
        status + cell,
        slot_count,
        agent_id_capacity,
        grid_height,
        grid_width,
        pending_capacity
    );
}

__global__ void release_pending_free_slots_kernel(
    const int64_t* __restrict__ slot_to_agent_id,
    const bool* __restrict__ alive_mask,
    int64_t* __restrict__ slot_free_queue,
    int64_t* __restrict__ slot_free_queue_count,
    const int64_t* __restrict__ pending_free_slots,
    const int64_t* __restrict__ pending_free_count,
    int64_t* __restrict__ pending_release_cursor,
    int64_t* __restrict__ appended_count,
    int32_t* __restrict__ status,
    const int64_t slot_count,
    const int64_t queue_capacity,
    const int64_t pending_capacity
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const int64_t queue_before = slot_free_queue_count[0];
    const int64_t pending_rows = pending_free_count[0];
    const int64_t cursor = pending_release_cursor[0];
    if (queue_before < 0 || queue_before > queue_capacity
        || pending_rows < 0 || pending_rows > pending_capacity
        || cursor < 0 || cursor > pending_rows) {
        status[0] = STATUS_INVALID_FREE_RELEASE_STATE;
        return;
    }
    const int64_t rows = pending_rows - cursor;
    if (queue_before + rows > queue_capacity) {
        status[0] = STATUS_FREE_RELEASE_CAPACITY;
        return;
    }
    for (int64_t pending = cursor; pending < pending_rows; ++pending) {
        const int64_t slot = pending_free_slots[pending];
        if (slot < 0 || slot >= slot_count || alive_mask[slot]
            || slot_to_agent_id[slot] >= 0) {
            status[0] = STATUS_FREE_RELEASE_CONTRADICTION;
            return;
        }
        for (int64_t queued = 0; queued < queue_before; ++queued) {
            if (slot_free_queue[queued] == slot) {
                status[0] = STATUS_FREE_RELEASE_CONTRADICTION;
                return;
            }
        }
        for (int64_t prior = cursor; prior < pending; ++prior) {
            if (pending_free_slots[prior] == slot) {
                status[0] = STATUS_FREE_RELEASE_CONTRADICTION;
                return;
            }
        }
    }
    for (int64_t row = 0; row < rows; ++row) {
        slot_free_queue[queue_before + row] = pending_free_slots[cursor + row];
    }
    slot_free_queue_count[0] = queue_before + rows;
    pending_release_cursor[0] = pending_rows;
    appended_count[0] = rows;
    status[0] = STATUS_OK;
}

__global__ void controlled_birth_identity_commit_kernel(
    int64_t* __restrict__ slot_to_agent_id,
    int32_t* __restrict__ agent_id_to_slot,
    int64_t* __restrict__ slot_tokens,
    bool* __restrict__ alive_mask,
    int32_t* __restrict__ pos_y,
    int32_t* __restrict__ pos_x,
    int32_t* __restrict__ agent_grid,
    float* __restrict__ energy,
    float* __restrict__ health,
    float* __restrict__ injury,
    float* __restrict__ scar,
    int32_t* __restrict__ age,
    int32_t* __restrict__ repro_cooldown,
    float* __restrict__ repro_threshold,
    int32_t* __restrict__ offspring_count,
    int32_t* __restrict__ lineage_founder_id,
    int32_t* __restrict__ lineage_depth,
    int32_t* __restrict__ lineage_birth_generation,
    int32_t* __restrict__ lineage_birth_tick,
    int32_t* __restrict__ patch_habitat_lineage_id,
    double* __restrict__ policy_energy_observation,
    int32_t* __restrict__ survival_steps,
    int32_t* __restrict__ rest_streak,
    int64_t* __restrict__ action_cache_token,
    int32_t* __restrict__ action_cache_cursor,
    int32_t* __restrict__ action_cache_next,
    int64_t* __restrict__ slot_free_queue,
    int64_t* __restrict__ slot_free_queue_count,
    int64_t* __restrict__ population_next_id,
    const int64_t* __restrict__ parent_slots,
    const int64_t* __restrict__ mate_slots,
    const int32_t* __restrict__ spawn_positions,
    int64_t* __restrict__ committed_parent_agent_ids,
    int64_t* __restrict__ committed_mate_agent_ids,
    int64_t* __restrict__ committed_child_agent_ids,
    int64_t* __restrict__ committed_child_slots,
    int64_t* __restrict__ committed_count,
    int64_t* __restrict__ alive_agent_ids_by_order,
    int64_t* __restrict__ alive_count,
    int32_t* __restrict__ status,
    const int64_t* __restrict__ planned_count,
    const int64_t plan_capacity,
    const int64_t slot_count,
    const int64_t agent_id_capacity,
    const int64_t grid_height,
    const int64_t grid_width,
    const int64_t queue_capacity,
    const double parent_energy_share,
    const double minimum_energy_buffer,
    const double turnover_start_energy,
    const float start_health,
    const int32_t cooldown_steps,
    const int32_t generation,
    const int32_t tick,
    const bool generation_turnover_refill
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const int64_t birth_count = planned_count[0];
    if (
        birth_count < 0 || birth_count > plan_capacity
        || birth_count > slot_count
    ) {
        status[0] = STATUS_INVALID_BIRTH_PLAN;
        return;
    }
    const int64_t free_count = slot_free_queue_count[0];
    if (free_count < 0 || free_count > queue_capacity) {
        status[0] = STATUS_INVALID_FREE_QUEUE_COUNT;
        return;
    }

    int64_t next_id = population_next_id[0] < 1 ? 1 : population_next_id[0];
    for (int64_t slot = 0; slot < slot_count; ++slot) {
        if (slot_to_agent_id[slot] >= next_id) {
            next_id = slot_to_agent_id[slot] + 1;
        }
    }
    if (
        next_id < 0
        || (
            birth_count > 0
            && (
                next_id > INT32_MAX
                || birth_count - 1 > INT32_MAX - next_id
            )
        )
    ) {
        status[0] = STATUS_AGENT_ID_CAPACITY;
        return;
    }

    int64_t previous_parent_id = -1;
    int64_t queue_head = 0;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = parent_slots[birth];
        const int64_t mate_slot = mate_slots[birth];
        if (parent_slot < 0 || parent_slot >= slot_count
            || mate_slot < 0 || mate_slot >= slot_count) {
            status[0] = STATUS_INVALID_BIRTH_PLAN;
            return;
        }
        const int64_t parent_id = slot_to_agent_id[parent_slot];
        const int64_t mate_id = slot_to_agent_id[mate_slot];
        if (parent_id < 0 || mate_id < 0
            || (generation_turnover_refill && parent_slot == mate_slot)
            || (!generation_turnover_refill
                && birth > 0 && parent_id <= previous_parent_id)) {
            status[0] = STATUS_INVALID_PARENT_ORDER;
            return;
        }
        previous_parent_id = parent_id;
        const double parent_energy = isfinite(policy_energy_observation[parent_slot])
            ? policy_energy_observation[parent_slot]
            : static_cast<double>(energy[parent_slot]);
        const double parent_share = parent_energy * parent_energy_share;
        const bool parent_invalid = !alive_mask[parent_slot]
            || health[parent_slot] <= 0.0f;
        const bool ordinary_parent_invalid = !generation_turnover_refill
            && (repro_cooldown[parent_slot] != 0
                || parent_energy < static_cast<double>(repro_threshold[parent_slot])
                || parent_energy - parent_share < minimum_energy_buffer);
        const bool turnover_mate_invalid = generation_turnover_refill
            && !alive_mask[mate_slot];
        if (parent_invalid || ordinary_parent_invalid || turnover_mate_invalid) {
            status[0] = STATUS_PARENT_NOT_VIABLE;
            return;
        }

        const int32_t child_y = spawn_positions[birth * 2];
        const int32_t child_x = spawn_positions[birth * 2 + 1];
        if (
            child_y < 0 || child_y >= grid_height
            || child_x < 0 || child_x >= grid_width
            || agent_grid[static_cast<int64_t>(child_y) * grid_width + child_x] != -1
        ) {
            status[0] = STATUS_SPAWN_RESERVATION;
            return;
        }
        for (int64_t previous = 0; previous < birth; ++previous) {
            if (
                spawn_positions[previous * 2] == child_y
                && spawn_positions[previous * 2 + 1] == child_x
            ) {
                status[0] = STATUS_SPAWN_RESERVATION;
                return;
            }
        }

        int64_t child_slot = -1;
        while (queue_head < free_count) {
            const int64_t candidate = slot_free_queue[queue_head++];
            if (
                candidate < 0 || candidate >= slot_count
                || alive_mask[candidate] || slot_to_agent_id[candidate] >= 0
            ) {
                continue;
            }
            bool already_selected = false;
            for (int64_t previous = 0; previous < birth; ++previous) {
                already_selected = already_selected || committed_child_slots[previous] == candidate;
            }
            if (!already_selected) {
                child_slot = candidate;
                break;
            }
        }
        if (child_slot < 0) {
            status[0] = STATUS_NO_FREE_SLOT;
            return;
        }
        committed_parent_agent_ids[birth] = parent_id;
        committed_mate_agent_ids[birth] = mate_id;
        committed_child_agent_ids[birth] = next_id + birth;
        committed_child_slots[birth] = child_slot;
    }

    // All capacity, parent, queue, and reservation checks are complete. From
    // here the controlled identity transaction cannot leave a partial commit.
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = parent_slots[birth];
        const int64_t mate_slot = mate_slots[birth];
        const int64_t parent_id = committed_parent_agent_ids[birth];
        const int64_t mate_id = committed_mate_agent_ids[birth];
        const int64_t child_id = committed_child_agent_ids[birth];
        const int64_t child_slot = committed_child_slots[birth];
        const int32_t child_y = spawn_positions[birth * 2];
        const int32_t child_x = spawn_positions[birth * 2 + 1];
        const double parent_energy = isfinite(policy_energy_observation[parent_slot])
            ? policy_energy_observation[parent_slot]
            : static_cast<double>(energy[parent_slot]);
        const double child_energy = generation_turnover_refill
            ? turnover_start_energy
            : parent_energy * parent_energy_share;
        const double parent_energy_after = parent_energy - child_energy;

        if (generation_turnover_refill) {
            offspring_count[parent_slot] += 1;
            offspring_count[mate_slot] += 1;
        } else {
            energy[parent_slot] = static_cast<float>(parent_energy_after);
            policy_energy_observation[parent_slot] = parent_energy_after;
            repro_cooldown[parent_slot] = cooldown_steps;
            offspring_count[parent_slot] += 1;
        }

        if (child_id >= 0 && child_id < agent_id_capacity) {
            agent_id_to_slot[child_id] = static_cast<int32_t>(child_slot);
        }
        slot_to_agent_id[child_slot] = child_id;
        slot_tokens[child_slot] = slot_tokens[child_slot] < 0
            ? 1
            : slot_tokens[child_slot] + 1;
        alive_mask[child_slot] = true;
        pos_y[child_slot] = child_y;
        pos_x[child_slot] = child_x;
        agent_grid[static_cast<int64_t>(child_y) * grid_width + child_x]
            = static_cast<int32_t>(child_id);
        energy[child_slot] = static_cast<float>(child_energy);
        policy_energy_observation[child_slot] = child_energy;
        health[child_slot] = start_health;
        injury[child_slot] = 0.0f;
        scar[child_slot] = 0.0f;
        age[child_slot] = 0;
        survival_steps[child_slot] = 0;
        rest_streak[child_slot] = 0;
        repro_cooldown[child_slot] = 0;
        repro_threshold[child_slot] = repro_threshold[parent_slot];
        offspring_count[child_slot] = 0;
        const int32_t parent_founder = lineage_founder_id[parent_slot] >= 0
            ? lineage_founder_id[parent_slot]
            : static_cast<int32_t>(parent_id);
        const int32_t mate_founder = lineage_founder_id[mate_slot] >= 0
            ? lineage_founder_id[mate_slot]
            : static_cast<int32_t>(mate_id);
        lineage_founder_id[child_slot] = generation_turnover_refill
            ? min(parent_founder, mate_founder)
            : parent_founder;
        lineage_depth[child_slot] = generation_turnover_refill
            ? max(lineage_depth[parent_slot], lineage_depth[mate_slot]) + 1
            : lineage_depth[parent_slot] + 1;
        lineage_birth_generation[child_slot] = generation;
        lineage_birth_tick[child_slot] = tick;
        const int32_t parent_patch_lineage = patch_habitat_lineage_id[parent_slot] != 0
            ? patch_habitat_lineage_id[parent_slot]
            : static_cast<int32_t>(parent_id);
        const int32_t mate_patch_lineage = patch_habitat_lineage_id[mate_slot] != 0
            ? patch_habitat_lineage_id[mate_slot]
            : static_cast<int32_t>(mate_id);
        patch_habitat_lineage_id[child_slot] = generation_turnover_refill
            ? min(parent_patch_lineage, mate_patch_lineage)
            : parent_patch_lineage;
        action_cache_token[child_slot] = -1;
        action_cache_cursor[child_slot] = -1;
        action_cache_next[child_slot] = -1;
    }
    population_next_id[0] = next_id + birth_count;
    committed_count[0] = birth_count;

    int64_t remaining_queue_count = 0;
    for (int64_t queue_index = queue_head; queue_index < free_count; ++queue_index) {
        const int64_t slot = slot_free_queue[queue_index];
        if (
            slot >= 0 && slot < slot_count
            && !alive_mask[slot] && slot_to_agent_id[slot] < 0
        ) {
            slot_free_queue[remaining_queue_count++] = slot;
        }
    }
    for (int64_t queue_index = remaining_queue_count; queue_index < free_count; ++queue_index) {
        slot_free_queue[queue_index] = -1;
    }
    slot_free_queue_count[0] = remaining_queue_count;

    int64_t remaining_alive = 0;
    int64_t previous_agent_id = INT64_MIN;
    int64_t previous_slot = -1;
    while (true) {
        int64_t selected_slot = -1;
        int64_t selected_agent_id = INT64_MAX;
        for (int64_t slot = 0; slot < slot_count; ++slot) {
            if (!alive_mask[slot] || slot_to_agent_id[slot] < 0) {
                continue;
            }
            const int64_t agent_id = slot_to_agent_id[slot];
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
        alive_agent_ids_by_order[remaining_alive++] = selected_agent_id;
        previous_agent_id = selected_agent_id;
        previous_slot = selected_slot;
    }
    alive_count[0] = remaining_alive;
    status[0] = STATUS_OK;
}

__global__ void fixed_mlp_child_scalar_genome_kernel(
    const int64_t* __restrict__ slot_to_agent_id,
    const bool* __restrict__ alive_mask,
    const int64_t* __restrict__ parent_slots,
    const int64_t* __restrict__ mate_slots,
    const int64_t* __restrict__ turnover_parent_pool_slots,
    const int64_t* __restrict__ turnover_parent_pool_count,
    const int64_t* __restrict__ child_slots,
    uint32_t* __restrict__ python_random_words,
    int64_t* __restrict__ python_random_index,
    bool* __restrict__ python_random_has_gauss,
    double* __restrict__ python_random_gauss_next,
    const double* __restrict__ python_random_gauss_primary_tape,
    const double* __restrict__ python_random_gauss_secondary_tape,
    int64_t* __restrict__ python_random_raw_draws,
    int64_t* __restrict__ mutation_py_semantic_draws,
    int32_t* __restrict__ genome_hidden_size,
    int32_t* __restrict__ genome_memory_dim,
    int32_t* __restrict__ genome_sensor_channel_mask,
    int32_t* __restrict__ genome_sensor_radius,
    int32_t* __restrict__ genome_introspection_channel_mask,
    int32_t* __restrict__ genome_self_regulation_control_mask,
    int32_t* __restrict__ genome_self_modification_target_mask,
    int32_t* __restrict__ genome_directed_inheritance_channel_mask,
    int64_t* __restrict__ policy_mem_dim,
    bool* __restrict__ policy_use_gru,
    float* __restrict__ genome_exploration_bias,
    float* __restrict__ genome_aggression_bias,
    float* __restrict__ genome_mutation_rate,
    float* __restrict__ genome_reproduction_threshold,
    float* __restrict__ genome_action_temperature,
    float* __restrict__ genome_drive_inertia,
    float* __restrict__ genome_drive_noise,
    float* __restrict__ genome_drive_logit_gain,
    float* __restrict__ genome_pred_lr,
    float* __restrict__ genome_drive_sp_hunger,
    float* __restrict__ genome_drive_sp_pain,
    float* __restrict__ genome_drive_sp_fatigue,
    float* __restrict__ genome_drive_sp_fear,
    float* __restrict__ genome_exp_prior_strength,
    float* __restrict__ genome_exp_prior_noise,
    float* __restrict__ genome_drive_birth_noise,
    float* __restrict__ repro_threshold,
    float* __restrict__ gru_upkeep_cost,
    float* __restrict__ policy_attack_bias,
    float* __restrict__ policy_action_temperature,
    float* __restrict__ policy_drive_logit_gain,
    float* __restrict__ drive_inertia,
    float* __restrict__ drive_noise,
    float* __restrict__ pred_lr,
    int64_t* __restrict__ committed_parent_slots,
    int64_t* __restrict__ committed_mate_slots,
    int64_t* __restrict__ committed_count,
    int64_t* __restrict__ semantic_draws_added,
    int32_t* __restrict__ status,
    const int64_t* __restrict__ planned_count,
    const int64_t plan_capacity,
    const int64_t slot_count,
    const int64_t gauss_tape_capacity,
    const int32_t sensor_radius_max,
    const bool select_turnover_pairs,
    const bool generation_turnover_refill
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const int64_t birth_count = planned_count[0];
    if (
        birth_count < 0 || birth_count > plan_capacity
        || birth_count > slot_count
    ) {
        status[0] = STATUS_INVALID_SCALAR_BIRTH_PLAN;
        return;
    }
    if (
        python_random_index[0] < 0 || python_random_index[0] > 624
        || python_random_raw_draws[0] < 0 || mutation_py_semantic_draws[0] < 0
    ) {
        status[0] = STATUS_INVALID_PYTHON_RANDOM_STATE;
        return;
    }

    const int64_t turnover_pool_count = select_turnover_pairs
        ? turnover_parent_pool_count[0]
        : 0;
    if (
        select_turnover_pairs
        && (
            turnover_pool_count < (birth_count > 0 ? 2 : 0)
            || turnover_pool_count > slot_count
        )
    ) {
        status[0] = STATUS_INVALID_SCALAR_BIRTH_PLAN;
        return;
    }
    for (int64_t pool_row = 0; pool_row < turnover_pool_count; ++pool_row) {
        const int64_t pool_slot = turnover_parent_pool_slots[pool_row];
        if (
            pool_slot < 0 || pool_slot >= slot_count
            || !alive_mask[pool_slot]
            || slot_to_agent_id[pool_slot] < 0
            || policy_use_gru[pool_slot]
        ) {
            status[0] = STATUS_INVALID_SCALAR_BIRTH_PLAN;
            return;
        }
        for (int64_t previous = 0; previous < pool_row; ++previous) {
            if (turnover_parent_pool_slots[previous] == pool_slot) {
                status[0] = STATUS_INVALID_SCALAR_BIRTH_PLAN;
                return;
            }
        }
    }

    int64_t previous_parent_id = -1;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t child_slot = child_slots[birth];
        if (
            child_slot < 0 || child_slot >= slot_count
            || !alive_mask[child_slot]
            || slot_to_agent_id[child_slot] < 0
        ) {
            status[0] = STATUS_INVALID_SCALAR_BIRTH_PLAN;
            return;
        }
        if (!select_turnover_pairs) {
            const int64_t parent_slot = parent_slots[birth];
            const int64_t mate_slot = mate_slots[birth];
            if (
                parent_slot < 0 || parent_slot >= slot_count
                || mate_slot < 0 || mate_slot >= slot_count
                || parent_slot == child_slot || mate_slot == child_slot
                || (generation_turnover_refill && parent_slot == mate_slot)
                || !alive_mask[parent_slot] || !alive_mask[mate_slot]
                || slot_to_agent_id[parent_slot] < 0
                || slot_to_agent_id[mate_slot] < 0
                || policy_use_gru[parent_slot] || policy_use_gru[mate_slot]
            ) {
                status[0] = STATUS_INVALID_SCALAR_BIRTH_PLAN;
                return;
            }
            const int64_t parent_id = slot_to_agent_id[parent_slot];
            if (!generation_turnover_refill
                && birth > 0 && parent_id <= previous_parent_id) {
                status[0] = STATUS_INVALID_SCALAR_BIRTH_PLAN;
                return;
            }
            previous_parent_id = parent_id;
        }
        for (int64_t previous = 0; previous < birth; ++previous) {
            if (child_slots[previous] == child_slot) {
                status[0] = STATUS_INVALID_SCALAR_BIRTH_PLAN;
                return;
            }
        }
    }

    uint32_t trial_words[624];
    for (int64_t word = 0; word < 624; ++word) {
        trial_words[word] = python_random_words[word];
    }
    DeviceCpythonRandom trial_rng{
        trial_words,
        python_random_index[0],
        python_random_has_gauss[0],
        python_random_gauss_next[0],
        python_random_gauss_primary_tape,
        python_random_gauss_secondary_tape,
        gauss_tape_capacity,
        python_random_raw_draws[0],
    };
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        int64_t parent_slot = parent_slots != nullptr ? parent_slots[birth] : -1;
        int64_t mate_slot = mate_slots != nullptr ? mate_slots[birth] : -1;
        if (select_turnover_pairs) {
            do {
                parent_slot = turnover_parent_pool_slots[trial_rng.randbelow(
                    static_cast<uint64_t>(turnover_pool_count)
                )];
                mate_slot = turnover_parent_pool_slots[trial_rng.randbelow(
                    static_cast<uint64_t>(turnover_pool_count)
                )];
            } while (mate_slot == parent_slot);
        }
        if (scalar_birth_rng_exceeds_gauss_tape(
                trial_rng,
                genome_mutation_rate[parent_slot],
                genome_mutation_rate[mate_slot]
            )) {
            status[0] = STATUS_FRESH_GAUSS_UNSUPPORTED;
            return;
        }
    }

    DeviceCpythonRandom rng{
        python_random_words,
        python_random_index[0],
        python_random_has_gauss[0],
        python_random_gauss_next[0],
        python_random_gauss_primary_tape,
        python_random_gauss_secondary_tape,
        gauss_tape_capacity,
        python_random_raw_draws[0],
    };
    int64_t draws_added = 0;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        int64_t parent_slot = parent_slots != nullptr ? parent_slots[birth] : -1;
        int64_t mate_slot = mate_slots != nullptr ? mate_slots[birth] : -1;
        if (select_turnover_pairs) {
            do {
                parent_slot = turnover_parent_pool_slots[rng.randbelow(
                    static_cast<uint64_t>(turnover_pool_count)
                )];
                mate_slot = turnover_parent_pool_slots[rng.randbelow(
                    static_cast<uint64_t>(turnover_pool_count)
                )];
                draws_added += 2;
            } while (mate_slot == parent_slot);
        }
        const int64_t child_slot = child_slots[birth];
        if (committed_parent_slots != nullptr && committed_mate_slots != nullptr) {
            committed_parent_slots[birth] = parent_slot;
            committed_mate_slots[birth] = mate_slot;
        }

        const double hidden_size_decision = rng.random_double();
        draws_added += 1;
        const int32_t child_hidden_size = hidden_size_decision < 0.5
            ? genome_hidden_size[parent_slot]
            : genome_hidden_size[mate_slot];
        const double parent_mutation_rate = canon_genome_double_device(
            static_cast<double>(genome_mutation_rate[parent_slot])
        );
        const double mate_mutation_rate = canon_genome_double_device(
            static_cast<double>(genome_mutation_rate[mate_slot])
        );
        const double base_mutation_rate = 0.5 * (
            parent_mutation_rate + mate_mutation_rate
        );
        const double scalar_mutation_rate = clamp_double(base_mutation_rate, 0.0, 1.0);
        const double exploration_base = 0.5 * (
            canon_genome_double_device(genome_exploration_bias[parent_slot])
            + canon_genome_double_device(genome_exploration_bias[mate_slot])
        );
        const float child_exploration_bias = canon_genome_float_device(
            mutate_scalar_gene_cached_only(
                rng, exploration_base, scalar_mutation_rate, 0.10, -1.0, 1.0, &draws_added
            )
        );
        const double aggression_base = 0.5 * (
            canon_genome_double_device(genome_aggression_bias[parent_slot])
            + canon_genome_double_device(genome_aggression_bias[mate_slot])
        );
        const float child_aggression_bias = canon_genome_float_device(
            mutate_scalar_gene_cached_only(
                rng, aggression_base, scalar_mutation_rate, 0.15, -2.0, 2.0, &draws_added
            )
        );
        const double threshold_base = 0.5 * (
            canon_genome_double_device(genome_reproduction_threshold[parent_slot])
            + canon_genome_double_device(genome_reproduction_threshold[mate_slot])
        );
        const float child_reproduction_threshold = canon_genome_float_device(
            mutate_scalar_gene_cached_only(
                rng, threshold_base, scalar_mutation_rate, 2.0, 2.0, 90.0, &draws_added
            )
        );
        const double temperature_base = 0.5 * (
            canon_genome_double_device(genome_action_temperature[parent_slot])
            + canon_genome_double_device(genome_action_temperature[mate_slot])
        );
        const float child_action_temperature = canon_genome_float_device(
            mutate_scalar_gene_cached_only(
                rng, temperature_base, scalar_mutation_rate, 0.10, 0.25, 2.5, &draws_added
            )
        );
        const double inertia_base = 0.5 * (
            canon_genome_double_device(genome_drive_inertia[parent_slot])
            + canon_genome_double_device(genome_drive_inertia[mate_slot])
        );
        const float child_drive_inertia = canon_genome_float_device(
            mutate_scalar_gene_cached_only(
                rng, inertia_base, scalar_mutation_rate, 0.05, 0.0, 0.999, &draws_added
            )
        );
        const double noise_base = 0.5 * (
            canon_genome_double_device(genome_drive_noise[parent_slot])
            + canon_genome_double_device(genome_drive_noise[mate_slot])
        );
        const float child_drive_noise = canon_genome_float_device(
            mutate_scalar_gene_cached_only(
                rng, noise_base, scalar_mutation_rate, 0.01, 0.0, 0.10, &draws_added
            )
        );
        const double gain_base = 0.5 * (
            canon_genome_double_device(genome_drive_logit_gain[parent_slot])
            + canon_genome_double_device(genome_drive_logit_gain[mate_slot])
        );
        const float child_drive_logit_gain = canon_genome_float_device(
            mutate_scalar_gene_cached_only(
                rng, gain_base, scalar_mutation_rate, 0.20, 0.0, 3.0, &draws_added
            )
        );
        const double pred_lr_base = 0.5 * (
            canon_genome_double_device(genome_pred_lr[parent_slot])
            + canon_genome_double_device(genome_pred_lr[mate_slot])
        );
        const float child_pred_lr = canon_genome_float_device(
            mutate_scalar_gene_cached_only(
                rng, pred_lr_base, scalar_mutation_rate, 0.03, 0.0, 1.0, &draws_added
            )
        );

        double child_mutation_rate_raw = base_mutation_rate;
        const double mutation_rate_decision = rng.random_double();
        draws_added += 1;
        if (mutation_rate_decision < 0.2) {
            double mutation_rate_delta = 0.0;
            (void)rng.consume_gauss(0.005, &mutation_rate_delta);
            child_mutation_rate_raw += mutation_rate_delta;
            draws_added += 1;
        }
        child_mutation_rate_raw = clamp_double(child_mutation_rate_raw, 0.001, 0.20);
        const float child_mutation_rate = canon_genome_float_device(child_mutation_rate_raw);

        const int32_t parent_memory_dim = genome_memory_dim[parent_slot];
        const int32_t mate_memory_dim = genome_memory_dim[mate_slot];
        const int32_t inherited_base = snap_memory_dim_device(static_cast<int32_t>(
            llround(0.5 * static_cast<double>(parent_memory_dim + mate_memory_dim))
        ));
        const double memory_probability = clamp_double(0.25 * child_mutation_rate_raw, 0.002, 0.05);
        const double memory_decision = rng.random_double();
        draws_added += 1;
        if (memory_decision < memory_probability) {
            int32_t memory_index = memory_index_device(inherited_base);
            const int32_t step = rng.randbelow(2) == 0 ? -1 : 1;
            draws_added += 1;
            memory_index += step;
            if (memory_index < 0) memory_index = 0;
            if (memory_index > 4) memory_index = 4;
        }

        const int32_t child_sensor_mask = (genome_sensor_channel_mask[parent_slot] & 1023) | 1;
        const int32_t child_sensor_radius = snap_sensor_radius_device(
            genome_sensor_radius[parent_slot], sensor_radius_max
        );
        const double child_sp_hunger_scalar = canon_genome_double_device(clamp_double(
            0.5 * (
                canon_genome_double_device(genome_drive_sp_hunger[parent_slot])
                + canon_genome_double_device(genome_drive_sp_hunger[mate_slot])
            ), 0.0, 1.0
        ));
        const float child_sp_hunger = static_cast<float>(child_sp_hunger_scalar);
        const float child_sp_pain = canon_genome_float_device(clamp_double(
            0.5 * (
                canon_genome_double_device(genome_drive_sp_pain[parent_slot])
                + canon_genome_double_device(genome_drive_sp_pain[mate_slot])
            ), 0.0, 1.0
        ));
        const float child_sp_fatigue = canon_genome_float_device(clamp_double(
            0.5 * (
                canon_genome_double_device(genome_drive_sp_fatigue[parent_slot])
                + canon_genome_double_device(genome_drive_sp_fatigue[mate_slot])
            ), 0.0, 1.0
        ));
        const float child_sp_fear = canon_genome_float_device(clamp_double(
            0.5 * (
                canon_genome_double_device(genome_drive_sp_fear[parent_slot])
                + canon_genome_double_device(genome_drive_sp_fear[mate_slot])
            ), 0.0, 1.0
        ));
        const float child_exp_prior_strength = canon_genome_float_device(clamp_double(
            0.5 * (
                canon_genome_double_device(genome_exp_prior_strength[parent_slot])
                + canon_genome_double_device(genome_exp_prior_strength[mate_slot])
            ), 0.0, 1.0
        ));
        const float child_exp_prior_noise = canon_genome_float_device(clamp_double(
            0.5 * (
                canon_genome_double_device(genome_exp_prior_noise[parent_slot])
                + canon_genome_double_device(genome_exp_prior_noise[mate_slot])
            ), 0.0, 0.5
        ));
        const float child_drive_birth_noise = canon_genome_float_device(clamp_double(
            0.5 * (
                canon_genome_double_device(genome_drive_birth_noise[parent_slot])
                + canon_genome_double_device(genome_drive_birth_noise[mate_slot])
            ), 0.0, 0.2
        ));

        genome_hidden_size[child_slot] = child_hidden_size;
        genome_memory_dim[child_slot] = 0;
        policy_mem_dim[child_slot] = 0;
        policy_use_gru[child_slot] = false;
        gru_upkeep_cost[child_slot] = 0.0f;
        genome_sensor_channel_mask[child_slot] = child_sensor_mask;
        genome_sensor_radius[child_slot] = child_sensor_radius;
        genome_introspection_channel_mask[child_slot] = genome_introspection_channel_mask[parent_slot] & 32767;
        genome_self_regulation_control_mask[child_slot] = genome_self_regulation_control_mask[parent_slot] & 31;
        genome_self_modification_target_mask[child_slot] = genome_self_modification_target_mask[parent_slot] & 31;
        genome_directed_inheritance_channel_mask[child_slot] = genome_directed_inheritance_channel_mask[parent_slot] & 63;
        genome_exploration_bias[child_slot] = child_exploration_bias;
        genome_aggression_bias[child_slot] = child_aggression_bias;
        genome_mutation_rate[child_slot] = child_mutation_rate;
        genome_reproduction_threshold[child_slot] = child_reproduction_threshold;
        genome_action_temperature[child_slot] = child_action_temperature;
        genome_drive_inertia[child_slot] = child_drive_inertia;
        genome_drive_noise[child_slot] = child_drive_noise;
        genome_drive_logit_gain[child_slot] = child_drive_logit_gain;
        genome_pred_lr[child_slot] = child_pred_lr;
        genome_drive_sp_hunger[child_slot] = child_sp_hunger;
        genome_drive_sp_pain[child_slot] = child_sp_pain;
        genome_drive_sp_fatigue[child_slot] = child_sp_fatigue;
        genome_drive_sp_fear[child_slot] = child_sp_fear;
        genome_exp_prior_strength[child_slot] = child_exp_prior_strength;
        genome_exp_prior_noise[child_slot] = child_exp_prior_noise;
        genome_drive_birth_noise[child_slot] = child_drive_birth_noise;
        repro_threshold[child_slot] = child_reproduction_threshold;
        policy_attack_bias[child_slot] = child_aggression_bias;
        policy_action_temperature[child_slot] = child_action_temperature;
        policy_drive_logit_gain[child_slot] = child_drive_logit_gain;
        drive_inertia[child_slot] = child_drive_inertia;
        drive_noise[child_slot] = child_drive_noise;
        pred_lr[child_slot] = child_pred_lr;
    }

    python_random_index[0] = rng.index;
    python_random_has_gauss[0] = rng.has_gauss_next;
    python_random_gauss_next[0] = rng.gauss_next;
    python_random_raw_draws[0] = rng.raw_draws;
    mutation_py_semantic_draws[0] += draws_added;
    committed_count[0] = birth_count;
    semantic_draws_added[0] = draws_added;
    status[0] = STATUS_OK;
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA-resident");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_slot_tensor(
    const torch::Tensor& tensor,
    const torch::ScalarType dtype,
    const int64_t slot_count,
    const char* name,
    const torch::Device& device
) {
    check_cuda_contiguous(tensor, name);
    TORCH_CHECK(tensor.device() == device, name, " must share the lifecycle CUDA device");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == slot_count, name, " must match slot capacity");
}

}  // namespace

std::vector<torch::Tensor> eiel_cuda_architecture_b_lifecycle_death_cull_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor agent_id_to_slot,
    torch::Tensor alive_mask,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor energy,
    torch::Tensor health,
    torch::Tensor policy_energy_observation,
    torch::Tensor rest_streak,
    torch::Tensor action_cache_token,
    torch::Tensor action_cache_cursor,
    torch::Tensor action_cache_next,
    torch::Tensor pending_free_agent_ids,
    torch::Tensor pending_free_slots,
    torch::Tensor pending_free_count
) {
    check_cuda_contiguous(slot_to_agent_id, "slot_to_agent_id");
    TORCH_CHECK(slot_to_agent_id.scalar_type() == torch::kInt64, "slot_to_agent_id must be int64");
    TORCH_CHECK(slot_to_agent_id.dim() == 1, "slot_to_agent_id must be rank 1");
    const int64_t slot_count = slot_to_agent_id.size(0);
    TORCH_CHECK(slot_count > 0, "slot capacity must be positive");
    const auto device = slot_to_agent_id.device();

    check_cuda_contiguous(agent_id_to_slot, "agent_id_to_slot");
    TORCH_CHECK(agent_id_to_slot.device() == device, "agent_id_to_slot must share the lifecycle CUDA device");
    TORCH_CHECK(agent_id_to_slot.scalar_type() == torch::kInt32, "agent_id_to_slot must be int32");
    TORCH_CHECK(agent_id_to_slot.dim() == 1, "agent_id_to_slot must be rank 1");

    check_slot_tensor(alive_mask, torch::kBool, slot_count, "alive_mask", device);
    check_slot_tensor(pos_y, torch::kInt32, slot_count, "pos_y", device);
    check_slot_tensor(pos_x, torch::kInt32, slot_count, "pos_x", device);
    check_slot_tensor(energy, torch::kFloat32, slot_count, "energy", device);
    check_slot_tensor(health, torch::kFloat32, slot_count, "health", device);
    check_slot_tensor(
        policy_energy_observation,
        torch::kFloat64,
        slot_count,
        "policy_energy_observation",
        device
    );
    check_slot_tensor(rest_streak, torch::kInt32, slot_count, "rest_streak", device);
    check_slot_tensor(action_cache_token, torch::kInt64, slot_count, "action_cache_token", device);
    check_slot_tensor(action_cache_cursor, torch::kInt32, slot_count, "action_cache_cursor", device);
    check_slot_tensor(action_cache_next, torch::kInt32, slot_count, "action_cache_next", device);

    check_cuda_contiguous(agent_grid, "agent_grid");
    TORCH_CHECK(agent_grid.device() == device, "agent_grid must share the lifecycle CUDA device");
    TORCH_CHECK(agent_grid.scalar_type() == torch::kInt32, "agent_grid must be int32");
    TORCH_CHECK(agent_grid.dim() == 2, "agent_grid must be a single-cell rank-2 grid");

    check_cuda_contiguous(pending_free_agent_ids, "pending_free_agent_ids");
    check_cuda_contiguous(pending_free_slots, "pending_free_slots");
    check_cuda_contiguous(pending_free_count, "pending_free_count");
    TORCH_CHECK(
        pending_free_agent_ids.device() == device
        && pending_free_slots.device() == device
        && pending_free_count.device() == device,
        "pending-free state must share the lifecycle CUDA device"
    );
    TORCH_CHECK(
        pending_free_agent_ids.scalar_type() == torch::kInt64
        && pending_free_slots.scalar_type() == torch::kInt64
        && pending_free_count.scalar_type() == torch::kInt64,
        "pending-free state must be int64"
    );
    TORCH_CHECK(
        pending_free_agent_ids.dim() == 1
        && pending_free_slots.dim() == 1
        && pending_free_agent_ids.size(0) == pending_free_slots.size(0),
        "pending-free banks must be aligned rank-1 tensors"
    );
    TORCH_CHECK(
        pending_free_agent_ids.size(0) >= slot_count,
        "pending-free capacity must cover the slot capacity"
    );
    TORCH_CHECK(
        pending_free_count.dim() == 1 && pending_free_count.numel() == 1,
        "pending_free_count must be a one-element rank-1 tensor"
    );

    c10::cuda::CUDAGuard device_guard(device);
    auto int64_options = slot_to_agent_id.options().dtype(torch::kInt64);
    auto int32_options = slot_to_agent_id.options().dtype(torch::kInt32);
    auto death_agent_ids = torch::full({slot_count}, -1, int64_options);
    auto death_slots = torch::full({slot_count}, -1, int64_options);
    auto death_reason_codes = torch::zeros({slot_count}, int32_options);
    auto death_count = torch::zeros({1}, int64_options);
    auto alive_agent_ids_by_order = torch::full({slot_count}, -1, int64_options);
    auto alive_count = torch::zeros({1}, int64_options);
    auto status = torch::zeros({1}, int32_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    lifecycle_death_cull_kernel<<<1, 1, 0, stream>>>(
        slot_to_agent_id.data_ptr<int64_t>(),
        agent_id_to_slot.data_ptr<int32_t>(),
        alive_mask.data_ptr<bool>(),
        pos_y.data_ptr<int32_t>(),
        pos_x.data_ptr<int32_t>(),
        agent_grid.data_ptr<int32_t>(),
        energy.data_ptr<float>(),
        health.data_ptr<float>(),
        policy_energy_observation.data_ptr<double>(),
        rest_streak.data_ptr<int32_t>(),
        action_cache_token.data_ptr<int64_t>(),
        action_cache_cursor.data_ptr<int32_t>(),
        action_cache_next.data_ptr<int32_t>(),
        pending_free_agent_ids.data_ptr<int64_t>(),
        pending_free_slots.data_ptr<int64_t>(),
        pending_free_count.data_ptr<int64_t>(),
        death_agent_ids.data_ptr<int64_t>(),
        death_slots.data_ptr<int64_t>(),
        death_reason_codes.data_ptr<int32_t>(),
        death_count.data_ptr<int64_t>(),
        alive_agent_ids_by_order.data_ptr<int64_t>(),
        alive_count.data_ptr<int64_t>(),
        status.data_ptr<int32_t>(),
        slot_count,
        agent_id_to_slot.size(0),
        agent_grid.size(0),
        agent_grid.size(1),
        pending_free_agent_ids.size(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        death_agent_ids,
        death_slots,
        death_reason_codes,
        death_count,
        alive_agent_ids_by_order,
        alive_count,
        status,
    };
}

std::vector<torch::Tensor>
eiel_cuda_architecture_b_lifecycle_death_cull_batched_counted_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor agent_id_to_slot,
    torch::Tensor alive_mask,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor energy,
    torch::Tensor health,
    torch::Tensor policy_energy_observation,
    torch::Tensor rest_streak,
    torch::Tensor action_cache_token,
    torch::Tensor action_cache_cursor,
    torch::Tensor action_cache_next,
    torch::Tensor pending_free_agent_ids,
    torch::Tensor pending_free_slots,
    torch::Tensor pending_free_count
) {
    check_cuda_contiguous(slot_to_agent_id, "slot_to_agent_id");
    TORCH_CHECK(
        slot_to_agent_id.scalar_type() == torch::kInt64
            && slot_to_agent_id.dim() == 2,
        "batched slot_to_agent_id must be int64[cells,slots]"
    );
    const int64_t cell_count = slot_to_agent_id.size(0);
    const int64_t slot_count = slot_to_agent_id.size(1);
    TORCH_CHECK(cell_count > 0, "batched death cull requires at least one cell");
    TORCH_CHECK(slot_count > 0, "slot capacity must be positive");
    TORCH_CHECK(cell_count <= 2147483647, "batched death cull cell count exceeds the CUDA grid limit");
    const auto device = slot_to_agent_id.device();

    const auto check_batched_slot_bank = [&](const torch::Tensor& tensor,
                                             const char* name,
                                             c10::ScalarType dtype) {
        check_cuda_contiguous(tensor, name);
        TORCH_CHECK(tensor.device() == device, name, " must share the lifecycle CUDA device");
        TORCH_CHECK(
            tensor.scalar_type() == dtype
                && tensor.dim() == 2
                && tensor.size(0) == cell_count
                && tensor.size(1) == slot_count,
            name,
            " must be [cells,slots]"
        );
    };
    check_batched_slot_bank(alive_mask, "alive_mask", torch::kBool);
    check_batched_slot_bank(pos_y, "pos_y", torch::kInt32);
    check_batched_slot_bank(pos_x, "pos_x", torch::kInt32);
    check_batched_slot_bank(energy, "energy", torch::kFloat32);
    check_batched_slot_bank(health, "health", torch::kFloat32);
    check_batched_slot_bank(
        policy_energy_observation,
        "policy_energy_observation",
        torch::kFloat64
    );
    check_batched_slot_bank(rest_streak, "rest_streak", torch::kInt32);
    check_batched_slot_bank(action_cache_token, "action_cache_token", torch::kInt64);
    check_batched_slot_bank(action_cache_cursor, "action_cache_cursor", torch::kInt32);
    check_batched_slot_bank(action_cache_next, "action_cache_next", torch::kInt32);

    check_cuda_contiguous(agent_id_to_slot, "agent_id_to_slot");
    TORCH_CHECK(
        agent_id_to_slot.device() == device
            && agent_id_to_slot.scalar_type() == torch::kInt32
            && agent_id_to_slot.dim() == 2
            && agent_id_to_slot.size(0) == cell_count,
        "batched agent_id_to_slot must be int32[cells,agent_ids] on the lifecycle device"
    );
    const int64_t agent_id_capacity = agent_id_to_slot.size(1);

    check_cuda_contiguous(agent_grid, "agent_grid");
    TORCH_CHECK(
        agent_grid.device() == device
            && agent_grid.scalar_type() == torch::kInt32
            && agent_grid.dim() == 3
            && agent_grid.size(0) == cell_count,
        "batched agent_grid must be int32[cells,height,width] on the lifecycle device"
    );
    const int64_t grid_height = agent_grid.size(1);
    const int64_t grid_width = agent_grid.size(2);

    check_cuda_contiguous(pending_free_agent_ids, "pending_free_agent_ids");
    check_cuda_contiguous(pending_free_slots, "pending_free_slots");
    TORCH_CHECK(
        pending_free_agent_ids.device() == device
            && pending_free_slots.device() == device
            && pending_free_agent_ids.scalar_type() == torch::kInt64
            && pending_free_slots.scalar_type() == torch::kInt64
            && pending_free_agent_ids.dim() == 2
            && pending_free_slots.sizes() == pending_free_agent_ids.sizes()
            && pending_free_agent_ids.size(0) == cell_count,
        "batched pending-free banks must be aligned int64[cells,pending_capacity] tensors"
    );
    const int64_t pending_capacity = pending_free_agent_ids.size(1);
    TORCH_CHECK(
        pending_capacity >= slot_count,
        "pending-free capacity must cover the slot capacity"
    );
    check_cuda_contiguous(pending_free_count, "pending_free_count");
    TORCH_CHECK(
        pending_free_count.device() == device
            && pending_free_count.scalar_type() == torch::kInt64
            && pending_free_count.dim() == 2
            && pending_free_count.size(0) == cell_count
            && pending_free_count.size(1) == 1,
        "batched pending_free_count must be int64[cells,1] on the lifecycle device"
    );

    c10::cuda::CUDAGuard device_guard(device);
    auto int64_options = slot_to_agent_id.options().dtype(torch::kInt64);
    auto int32_options = slot_to_agent_id.options().dtype(torch::kInt32);
    auto death_agent_ids = torch::full({cell_count, slot_count}, -1, int64_options);
    auto death_slots = torch::full({cell_count, slot_count}, -1, int64_options);
    auto death_reason_codes = torch::zeros({cell_count, slot_count}, int32_options);
    auto death_count = torch::zeros({cell_count, 1}, int64_options);
    auto alive_agent_ids_by_order = torch::full(
        {cell_count, slot_count},
        -1,
        int64_options
    );
    auto alive_count = torch::zeros({cell_count, 1}, int64_options);
    auto status = torch::zeros({cell_count, 1}, int32_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    lifecycle_death_cull_batched_counted_kernel<<<
        static_cast<unsigned int>(cell_count),
        1,
        0,
        stream
    >>>(
        slot_to_agent_id.data_ptr<int64_t>(),
        agent_id_to_slot.data_ptr<int32_t>(),
        alive_mask.data_ptr<bool>(),
        pos_y.data_ptr<int32_t>(),
        pos_x.data_ptr<int32_t>(),
        agent_grid.data_ptr<int32_t>(),
        energy.data_ptr<float>(),
        health.data_ptr<float>(),
        policy_energy_observation.data_ptr<double>(),
        rest_streak.data_ptr<int32_t>(),
        action_cache_token.data_ptr<int64_t>(),
        action_cache_cursor.data_ptr<int32_t>(),
        action_cache_next.data_ptr<int32_t>(),
        pending_free_agent_ids.data_ptr<int64_t>(),
        pending_free_slots.data_ptr<int64_t>(),
        pending_free_count.data_ptr<int64_t>(),
        death_agent_ids.data_ptr<int64_t>(),
        death_slots.data_ptr<int64_t>(),
        death_reason_codes.data_ptr<int32_t>(),
        death_count.data_ptr<int64_t>(),
        alive_agent_ids_by_order.data_ptr<int64_t>(),
        alive_count.data_ptr<int64_t>(),
        status.data_ptr<int32_t>(),
        cell_count,
        slot_count,
        agent_id_capacity,
        grid_height,
        grid_width,
        pending_capacity
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        death_agent_ids,
        death_slots,
        death_reason_codes,
        death_count,
        alive_agent_ids_by_order,
        alive_count,
        status,
    };
}

std::vector<torch::Tensor> eiel_cuda_architecture_b_release_pending_free_slots_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor alive_mask,
    torch::Tensor slot_free_queue,
    torch::Tensor slot_free_queue_count,
    torch::Tensor pending_free_slots,
    torch::Tensor pending_free_count,
    torch::Tensor pending_release_cursor
) {
    check_cuda_contiguous(slot_to_agent_id, "slot_to_agent_id");
    TORCH_CHECK(slot_to_agent_id.scalar_type() == torch::kInt64, "slot_to_agent_id must be int64");
    TORCH_CHECK(slot_to_agent_id.dim() == 1, "slot_to_agent_id must be rank 1");
    const int64_t slot_count = slot_to_agent_id.size(0);
    const auto device = slot_to_agent_id.device();
    check_slot_tensor(alive_mask, torch::kBool, slot_count, "alive_mask", device);
    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"slot_free_queue", slot_free_queue},
             {"slot_free_queue_count", slot_free_queue_count},
             {"pending_free_slots", pending_free_slots},
             {"pending_free_count", pending_free_count},
             {"pending_release_cursor", pending_release_cursor},
         }) {
        check_cuda_contiguous(named.second, named.first);
        TORCH_CHECK(named.second.device() == device, named.first, " must share the lifecycle CUDA device");
        TORCH_CHECK(named.second.scalar_type() == torch::kInt64, named.first, " must be int64");
        TORCH_CHECK(named.second.dim() == 1, named.first, " must be rank 1");
    }
    TORCH_CHECK(slot_free_queue.size(0) >= slot_count, "slot_free_queue must cover slot capacity");
    TORCH_CHECK(pending_free_slots.size(0) >= slot_count, "pending_free_slots must cover slot capacity");
    TORCH_CHECK(slot_free_queue_count.numel() == 1, "slot_free_queue_count must have one value");
    TORCH_CHECK(pending_free_count.numel() == 1, "pending_free_count must have one value");
    TORCH_CHECK(pending_release_cursor.numel() == 1, "pending_release_cursor must have one value");

    c10::cuda::CUDAGuard device_guard(device);
    auto appended_count = torch::zeros({1}, slot_free_queue.options());
    auto status = torch::zeros({1}, slot_free_queue.options().dtype(torch::kInt32));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    release_pending_free_slots_kernel<<<1, 1, 0, stream>>>(
        slot_to_agent_id.data_ptr<int64_t>(),
        alive_mask.data_ptr<bool>(),
        slot_free_queue.data_ptr<int64_t>(),
        slot_free_queue_count.data_ptr<int64_t>(),
        pending_free_slots.data_ptr<int64_t>(),
        pending_free_count.data_ptr<int64_t>(),
        pending_release_cursor.data_ptr<int64_t>(),
        appended_count.data_ptr<int64_t>(),
        status.data_ptr<int32_t>(),
        slot_count,
        slot_free_queue.size(0),
        pending_free_slots.size(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {slot_free_queue_count, pending_release_cursor, appended_count, status};
}

static std::vector<torch::Tensor> controlled_birth_identity_commit_impl(
    torch::Tensor slot_to_agent_id,
    torch::Tensor agent_id_to_slot,
    torch::Tensor slot_tokens,
    torch::Tensor alive_mask,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor energy,
    torch::Tensor health,
    torch::Tensor injury,
    torch::Tensor scar,
    torch::Tensor age,
    torch::Tensor repro_cooldown,
    torch::Tensor repro_threshold,
    torch::Tensor offspring_count,
    torch::Tensor lineage_founder_id,
    torch::Tensor lineage_depth,
    torch::Tensor lineage_birth_generation,
    torch::Tensor lineage_birth_tick,
    torch::Tensor patch_habitat_lineage_id,
    torch::Tensor policy_energy_observation,
    torch::Tensor survival_steps,
    torch::Tensor rest_streak,
    torch::Tensor action_cache_token,
    torch::Tensor action_cache_cursor,
    torch::Tensor action_cache_next,
    torch::Tensor slot_free_queue,
    torch::Tensor slot_free_queue_count,
    torch::Tensor population_next_id,
    torch::Tensor parent_slots,
    torch::Tensor mate_slots,
    torch::Tensor spawn_positions,
    torch::Tensor planned_count,
    const double parent_energy_share,
    const double minimum_energy_buffer,
    const double turnover_start_energy,
    const double start_health,
    const int64_t cooldown_steps,
    const int64_t generation,
    const int64_t tick,
    const bool generation_turnover_refill
) {
    check_cuda_contiguous(slot_to_agent_id, "slot_to_agent_id");
    TORCH_CHECK(slot_to_agent_id.scalar_type() == torch::kInt64, "slot_to_agent_id must be int64");
    TORCH_CHECK(slot_to_agent_id.dim() == 1, "slot_to_agent_id must be rank 1");
    const int64_t slot_count = slot_to_agent_id.size(0);
    TORCH_CHECK(slot_count > 0, "slot capacity must be positive");
    const auto device = slot_to_agent_id.device();

    check_cuda_contiguous(agent_id_to_slot, "agent_id_to_slot");
    TORCH_CHECK(agent_id_to_slot.device() == device, "agent_id_to_slot must share the lifecycle CUDA device");
    TORCH_CHECK(agent_id_to_slot.scalar_type() == torch::kInt32, "agent_id_to_slot must be int32");
    TORCH_CHECK(agent_id_to_slot.dim() == 1, "agent_id_to_slot must be rank 1");
    check_slot_tensor(slot_tokens, torch::kInt64, slot_count, "slot_tokens", device);
    check_slot_tensor(alive_mask, torch::kBool, slot_count, "alive_mask", device);
    check_slot_tensor(pos_y, torch::kInt32, slot_count, "pos_y", device);
    check_slot_tensor(pos_x, torch::kInt32, slot_count, "pos_x", device);
    check_slot_tensor(energy, torch::kFloat32, slot_count, "energy", device);
    check_slot_tensor(health, torch::kFloat32, slot_count, "health", device);
    check_slot_tensor(injury, torch::kFloat32, slot_count, "injury", device);
    check_slot_tensor(scar, torch::kFloat32, slot_count, "scar", device);
    check_slot_tensor(age, torch::kInt32, slot_count, "age", device);
    check_slot_tensor(repro_cooldown, torch::kInt32, slot_count, "repro_cooldown", device);
    check_slot_tensor(repro_threshold, torch::kFloat32, slot_count, "repro_threshold", device);
    check_slot_tensor(offspring_count, torch::kInt32, slot_count, "offspring_count", device);
    check_slot_tensor(
        lineage_founder_id,
        torch::kInt32,
        slot_count,
        "lineage_founder_id",
        device
    );
    check_slot_tensor(lineage_depth, torch::kInt32, slot_count, "lineage_depth", device);
    check_slot_tensor(
        lineage_birth_generation,
        torch::kInt32,
        slot_count,
        "lineage_birth_generation",
        device
    );
    check_slot_tensor(
        lineage_birth_tick,
        torch::kInt32,
        slot_count,
        "lineage_birth_tick",
        device
    );
    check_slot_tensor(
        patch_habitat_lineage_id,
        torch::kInt32,
        slot_count,
        "patch_habitat_lineage_id",
        device
    );
    check_slot_tensor(
        policy_energy_observation,
        torch::kFloat64,
        slot_count,
        "policy_energy_observation",
        device
    );
    check_slot_tensor(survival_steps, torch::kInt32, slot_count, "survival_steps", device);
    check_slot_tensor(rest_streak, torch::kInt32, slot_count, "rest_streak", device);
    check_slot_tensor(action_cache_token, torch::kInt64, slot_count, "action_cache_token", device);
    check_slot_tensor(action_cache_cursor, torch::kInt32, slot_count, "action_cache_cursor", device);
    check_slot_tensor(action_cache_next, torch::kInt32, slot_count, "action_cache_next", device);

    check_cuda_contiguous(agent_grid, "agent_grid");
    TORCH_CHECK(agent_grid.device() == device, "agent_grid must share the lifecycle CUDA device");
    TORCH_CHECK(agent_grid.scalar_type() == torch::kInt32, "agent_grid must be int32");
    TORCH_CHECK(agent_grid.dim() == 2, "agent_grid must be a single-cell rank-2 grid");

    check_cuda_contiguous(slot_free_queue, "slot_free_queue");
    check_cuda_contiguous(slot_free_queue_count, "slot_free_queue_count");
    check_cuda_contiguous(population_next_id, "population_next_id");
    TORCH_CHECK(
        slot_free_queue.device() == device
        && slot_free_queue_count.device() == device
        && population_next_id.device() == device,
        "allocation state must share the lifecycle CUDA device"
    );
    TORCH_CHECK(
        slot_free_queue.scalar_type() == torch::kInt64
        && slot_free_queue_count.scalar_type() == torch::kInt64
        && population_next_id.scalar_type() == torch::kInt64,
        "allocation state must be int64"
    );
    TORCH_CHECK(slot_free_queue.dim() == 1, "slot_free_queue must be rank 1");
    TORCH_CHECK(slot_free_queue.size(0) >= slot_count, "slot_free_queue must cover slot capacity");
    TORCH_CHECK(
        slot_free_queue_count.dim() == 1 && slot_free_queue_count.numel() == 1,
        "slot_free_queue_count must be a one-element rank-1 tensor"
    );
    TORCH_CHECK(
        population_next_id.dim() == 1 && population_next_id.numel() == 1,
        "population_next_id must be a one-element rank-1 tensor"
    );

    check_cuda_contiguous(parent_slots, "parent_slots");
    check_cuda_contiguous(mate_slots, "mate_slots");
    check_cuda_contiguous(spawn_positions, "spawn_positions");
    check_cuda_contiguous(planned_count, "planned_count");
    TORCH_CHECK(
        parent_slots.device() == device && mate_slots.device() == device
            && spawn_positions.device() == device
            && planned_count.device() == device,
        "controlled birth plan must share the lifecycle CUDA device"
    );
    TORCH_CHECK(parent_slots.scalar_type() == torch::kInt64, "parent_slots must be int64");
    TORCH_CHECK(mate_slots.scalar_type() == torch::kInt64, "mate_slots must be int64");
    TORCH_CHECK(spawn_positions.scalar_type() == torch::kInt32, "spawn_positions must be int32");
    TORCH_CHECK(parent_slots.dim() == 1, "parent_slots must be rank 1");
    TORCH_CHECK(mate_slots.dim() == 1 && mate_slots.sizes() == parent_slots.sizes(), "mate_slots must align with parent_slots");
    TORCH_CHECK(
        spawn_positions.dim() == 2
        && spawn_positions.size(0) == parent_slots.size(0)
        && spawn_positions.size(1) == 2,
        "spawn_positions must be [births,2]"
    );
    TORCH_CHECK(
        planned_count.scalar_type() == torch::kInt64
            && planned_count.dim() == 1
            && planned_count.numel() == 1,
        "planned_count must be int64[1]"
    );
    TORCH_CHECK(
        std::isfinite(parent_energy_share)
        && parent_energy_share >= 0.0
        && parent_energy_share <= 1.0,
        "parent_energy_share must be finite and in [0,1]"
    );
    TORCH_CHECK(
        std::isfinite(minimum_energy_buffer) && minimum_energy_buffer >= 0.0,
        "minimum_energy_buffer must be finite and nonnegative"
    );
    TORCH_CHECK(
        !generation_turnover_refill
            || (std::isfinite(turnover_start_energy) && turnover_start_energy > 0.0),
        "turnover_start_energy must be positive for generation turnover"
    );
    TORCH_CHECK(std::isfinite(start_health) && start_health > 0.0, "start_health must be positive");
    TORCH_CHECK(cooldown_steps >= 0 && cooldown_steps <= INT32_MAX, "cooldown_steps is out of range");
    TORCH_CHECK(generation >= 0 && generation <= INT32_MAX, "generation is out of range");
    TORCH_CHECK(tick >= 0 && tick <= INT32_MAX, "tick is out of range");

    const int64_t plan_capacity = parent_slots.size(0);
    c10::cuda::CUDAGuard device_guard(device);
    auto int64_options = slot_to_agent_id.options().dtype(torch::kInt64);
    auto int32_options = slot_to_agent_id.options().dtype(torch::kInt32);
    auto committed_parent_agent_ids = torch::full({plan_capacity}, -1, int64_options);
    auto committed_mate_agent_ids = torch::full({plan_capacity}, -1, int64_options);
    auto committed_child_agent_ids = torch::full({plan_capacity}, -1, int64_options);
    auto committed_child_slots = torch::full({plan_capacity}, -1, int64_options);
    auto committed_count = torch::zeros({1}, int64_options);
    auto alive_agent_ids_by_order = torch::full({slot_count}, -1, int64_options);
    auto alive_count = torch::zeros({1}, int64_options);
    auto status = torch::zeros({1}, int32_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    controlled_birth_identity_commit_kernel<<<1, 1, 0, stream>>>(
        slot_to_agent_id.data_ptr<int64_t>(),
        agent_id_to_slot.data_ptr<int32_t>(),
        slot_tokens.data_ptr<int64_t>(),
        alive_mask.data_ptr<bool>(),
        pos_y.data_ptr<int32_t>(),
        pos_x.data_ptr<int32_t>(),
        agent_grid.data_ptr<int32_t>(),
        energy.data_ptr<float>(),
        health.data_ptr<float>(),
        injury.data_ptr<float>(),
        scar.data_ptr<float>(),
        age.data_ptr<int32_t>(),
        repro_cooldown.data_ptr<int32_t>(),
        repro_threshold.data_ptr<float>(),
        offspring_count.data_ptr<int32_t>(),
        lineage_founder_id.data_ptr<int32_t>(),
        lineage_depth.data_ptr<int32_t>(),
        lineage_birth_generation.data_ptr<int32_t>(),
        lineage_birth_tick.data_ptr<int32_t>(),
        patch_habitat_lineage_id.data_ptr<int32_t>(),
        policy_energy_observation.data_ptr<double>(),
        survival_steps.data_ptr<int32_t>(),
        rest_streak.data_ptr<int32_t>(),
        action_cache_token.data_ptr<int64_t>(),
        action_cache_cursor.data_ptr<int32_t>(),
        action_cache_next.data_ptr<int32_t>(),
        slot_free_queue.data_ptr<int64_t>(),
        slot_free_queue_count.data_ptr<int64_t>(),
        population_next_id.data_ptr<int64_t>(),
        parent_slots.data_ptr<int64_t>(),
        mate_slots.data_ptr<int64_t>(),
        spawn_positions.data_ptr<int32_t>(),
        committed_parent_agent_ids.data_ptr<int64_t>(),
        committed_mate_agent_ids.data_ptr<int64_t>(),
        committed_child_agent_ids.data_ptr<int64_t>(),
        committed_child_slots.data_ptr<int64_t>(),
        committed_count.data_ptr<int64_t>(),
        alive_agent_ids_by_order.data_ptr<int64_t>(),
        alive_count.data_ptr<int64_t>(),
        status.data_ptr<int32_t>(),
        planned_count.data_ptr<int64_t>(),
        plan_capacity,
        slot_count,
        agent_id_to_slot.size(0),
        agent_grid.size(0),
        agent_grid.size(1),
        slot_free_queue.size(0),
        parent_energy_share,
        minimum_energy_buffer,
        turnover_start_energy,
        static_cast<float>(start_health),
        static_cast<int32_t>(cooldown_steps),
        static_cast<int32_t>(generation),
        static_cast<int32_t>(tick),
        generation_turnover_refill
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        committed_parent_agent_ids,
        committed_mate_agent_ids,
        committed_child_agent_ids,
        committed_child_slots,
        committed_count,
        alive_agent_ids_by_order,
        alive_count,
        status,
    };
}

std::vector<torch::Tensor> eiel_cuda_architecture_b_controlled_birth_identity_commit_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor agent_id_to_slot,
    torch::Tensor slot_tokens,
    torch::Tensor alive_mask,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor energy,
    torch::Tensor health,
    torch::Tensor injury,
    torch::Tensor scar,
    torch::Tensor age,
    torch::Tensor repro_cooldown,
    torch::Tensor repro_threshold,
    torch::Tensor offspring_count,
    torch::Tensor lineage_founder_id,
    torch::Tensor lineage_depth,
    torch::Tensor lineage_birth_generation,
    torch::Tensor lineage_birth_tick,
    torch::Tensor patch_habitat_lineage_id,
    torch::Tensor policy_energy_observation,
    torch::Tensor survival_steps,
    torch::Tensor rest_streak,
    torch::Tensor action_cache_token,
    torch::Tensor action_cache_cursor,
    torch::Tensor action_cache_next,
    torch::Tensor slot_free_queue,
    torch::Tensor slot_free_queue_count,
    torch::Tensor population_next_id,
    torch::Tensor parent_slots,
    torch::Tensor spawn_positions,
    torch::Tensor planned_count,
    const double parent_energy_share,
    const double minimum_energy_buffer,
    const double start_health,
    const int64_t cooldown_steps,
    const int64_t generation,
    const int64_t tick
) {
    auto result = controlled_birth_identity_commit_impl(
        slot_to_agent_id, agent_id_to_slot, slot_tokens, alive_mask, pos_y, pos_x,
        agent_grid, energy, health, injury, scar, age, repro_cooldown,
        repro_threshold, offspring_count, lineage_founder_id, lineage_depth,
        lineage_birth_generation, lineage_birth_tick, patch_habitat_lineage_id,
        policy_energy_observation, survival_steps, rest_streak, action_cache_token,
        action_cache_cursor, action_cache_next, slot_free_queue,
        slot_free_queue_count, population_next_id, parent_slots, parent_slots,
        spawn_positions, planned_count, parent_energy_share, minimum_energy_buffer,
        0.0, start_health, cooldown_steps, generation, tick, false
    );
    return {
        result[0], result[2], result[3], result[4],
        result[5], result[6], result[7],
    };
}

std::vector<torch::Tensor> eiel_cuda_architecture_b_turnover_refill_identity_commit_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor agent_id_to_slot,
    torch::Tensor slot_tokens,
    torch::Tensor alive_mask,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor energy,
    torch::Tensor health,
    torch::Tensor injury,
    torch::Tensor scar,
    torch::Tensor age,
    torch::Tensor repro_cooldown,
    torch::Tensor repro_threshold,
    torch::Tensor offspring_count,
    torch::Tensor lineage_founder_id,
    torch::Tensor lineage_depth,
    torch::Tensor lineage_birth_generation,
    torch::Tensor lineage_birth_tick,
    torch::Tensor patch_habitat_lineage_id,
    torch::Tensor policy_energy_observation,
    torch::Tensor survival_steps,
    torch::Tensor rest_streak,
    torch::Tensor action_cache_token,
    torch::Tensor action_cache_cursor,
    torch::Tensor action_cache_next,
    torch::Tensor slot_free_queue,
    torch::Tensor slot_free_queue_count,
    torch::Tensor population_next_id,
    torch::Tensor parent_slots,
    torch::Tensor mate_slots,
    torch::Tensor spawn_positions,
    torch::Tensor planned_count,
    const double start_energy,
    const double start_health,
    const int64_t generation
) {
    return controlled_birth_identity_commit_impl(
        slot_to_agent_id, agent_id_to_slot, slot_tokens, alive_mask, pos_y, pos_x,
        agent_grid, energy, health, injury, scar, age, repro_cooldown,
        repro_threshold, offspring_count, lineage_founder_id, lineage_depth,
        lineage_birth_generation, lineage_birth_tick, patch_habitat_lineage_id,
        policy_energy_observation, survival_steps, rest_streak, action_cache_token,
        action_cache_cursor, action_cache_next, slot_free_queue,
        slot_free_queue_count, population_next_id, parent_slots, mate_slots,
        spawn_positions, planned_count, 0.0, 0.0, start_energy, start_health, 0,
        generation, 0, true
    );
}

static std::vector<torch::Tensor> fixed_mlp_child_scalar_genome_impl(
    torch::Tensor slot_to_agent_id,
    torch::Tensor alive_mask,
    torch::Tensor parent_slots_or_pool,
    torch::Tensor mate_slots_or_pool_count,
    torch::Tensor child_slots,
    torch::Tensor planned_count,
    torch::Tensor python_random_words,
    torch::Tensor python_random_index,
    torch::Tensor python_random_has_gauss,
    torch::Tensor python_random_gauss_next,
    torch::Tensor python_random_gauss_primary_tape,
    torch::Tensor python_random_gauss_secondary_tape,
    torch::Tensor python_random_raw_draws,
    torch::Tensor mutation_py_semantic_draws,
    std::vector<torch::Tensor> genome_int32_fields,
    torch::Tensor policy_mem_dim,
    torch::Tensor policy_use_gru,
    std::vector<torch::Tensor> genome_float32_fields,
    const int64_t sensor_radius_max,
    const bool generation_turnover_refill,
    const bool select_turnover_pairs
) {
    check_cuda_contiguous(slot_to_agent_id, "slot_to_agent_id");
    TORCH_CHECK(slot_to_agent_id.scalar_type() == torch::kInt64, "slot_to_agent_id must be int64");
    TORCH_CHECK(slot_to_agent_id.dim() == 1, "slot_to_agent_id must be rank 1");
    const int64_t slot_count = slot_to_agent_id.size(0);
    TORCH_CHECK(slot_count > 0, "slot capacity must be positive");
    const auto device = slot_to_agent_id.device();
    check_slot_tensor(alive_mask, torch::kBool, slot_count, "alive_mask", device);

    check_cuda_contiguous(
        parent_slots_or_pool,
        select_turnover_pairs ? "parent_pool_slots" : "parent_slots"
    );
    check_cuda_contiguous(
        mate_slots_or_pool_count,
        select_turnover_pairs ? "parent_pool_count" : "mate_slots"
    );
    check_cuda_contiguous(child_slots, "child_slots");
    TORCH_CHECK(
        parent_slots_or_pool.device() == device
            && mate_slots_or_pool_count.device() == device
            && child_slots.device() == device,
        "scalar birth plan must share the lifecycle CUDA device"
    );
    TORCH_CHECK(
        parent_slots_or_pool.scalar_type() == torch::kInt64
            && mate_slots_or_pool_count.scalar_type() == torch::kInt64
            && child_slots.scalar_type() == torch::kInt64,
        "scalar birth plan must be int64"
    );
    if (select_turnover_pairs) {
        TORCH_CHECK(
            parent_slots_or_pool.dim() == 1
                && parent_slots_or_pool.numel() == slot_count,
            "parent_pool_slots must be int64[slots]"
        );
        TORCH_CHECK(
            mate_slots_or_pool_count.dim() == 1
                && mate_slots_or_pool_count.numel() == 1,
            "parent_pool_count must be int64[1]"
        );
        TORCH_CHECK(
            child_slots.dim() == 1,
            "turnover child_slots must be rank 1"
        );
    } else {
        TORCH_CHECK(
            parent_slots_or_pool.dim() == 1
                && mate_slots_or_pool_count.dim() == 1
                && child_slots.dim() == 1
                && parent_slots_or_pool.size(0) == mate_slots_or_pool_count.size(0)
                && parent_slots_or_pool.size(0) == child_slots.size(0),
            "parent_slots, mate_slots, and child_slots must be aligned rank-1 tensors"
        );
    }
    check_cuda_contiguous(planned_count, "planned_count");
    TORCH_CHECK(
        planned_count.device() == device
            && planned_count.scalar_type() == torch::kInt64
            && planned_count.dim() == 1
            && planned_count.numel() == 1,
        "planned_count must be CUDA int64[1] on the lifecycle device"
    );

    check_cuda_contiguous(python_random_words, "python_random_words");
    check_cuda_contiguous(python_random_index, "python_random_index");
    check_cuda_contiguous(python_random_has_gauss, "python_random_has_gauss");
    check_cuda_contiguous(python_random_gauss_next, "python_random_gauss_next");
    check_cuda_contiguous(python_random_gauss_primary_tape, "python_random_gauss_primary_tape");
    check_cuda_contiguous(python_random_gauss_secondary_tape, "python_random_gauss_secondary_tape");
    check_cuda_contiguous(python_random_raw_draws, "python_random_raw_draws");
    check_cuda_contiguous(mutation_py_semantic_draws, "mutation_py_semantic_draws");
    TORCH_CHECK(
        python_random_words.device() == device
        && python_random_index.device() == device
        && python_random_has_gauss.device() == device
        && python_random_gauss_next.device() == device
        && python_random_gauss_primary_tape.device() == device
        && python_random_gauss_secondary_tape.device() == device
        && python_random_raw_draws.device() == device
        && mutation_py_semantic_draws.device() == device,
        "Python random state must share the lifecycle CUDA device"
    );
    TORCH_CHECK(
        python_random_words.scalar_type() == c10::ScalarType::UInt32
        && python_random_words.dim() == 1 && python_random_words.numel() == 624,
        "python_random_words must be uint32[624]"
    );
    TORCH_CHECK(
        python_random_index.scalar_type() == torch::kInt64
        && python_random_index.dim() == 1 && python_random_index.numel() == 1,
        "python_random_index must be int64[1]"
    );
    TORCH_CHECK(
        python_random_has_gauss.scalar_type() == torch::kBool
        && python_random_has_gauss.dim() == 1 && python_random_has_gauss.numel() == 1,
        "python_random_has_gauss must be bool[1]"
    );
    TORCH_CHECK(
        python_random_gauss_next.scalar_type() == torch::kFloat64
        && python_random_gauss_next.dim() == 1 && python_random_gauss_next.numel() == 1,
        "python_random_gauss_next must be float64[1]"
    );
    TORCH_CHECK(
        python_random_gauss_primary_tape.scalar_type() == torch::kFloat64
        && python_random_gauss_primary_tape.dim() == 1,
        "python_random_gauss_primary_tape must be rank-1 float64"
    );
    TORCH_CHECK(
        python_random_gauss_secondary_tape.scalar_type() == torch::kFloat64
        && python_random_gauss_secondary_tape.dim() == 1
        && python_random_gauss_secondary_tape.numel() == python_random_gauss_primary_tape.numel(),
        "Python Gaussian tapes must be aligned rank-1 float64 tensors"
    );
    TORCH_CHECK(
        python_random_raw_draws.scalar_type() == torch::kInt64
        && python_random_raw_draws.dim() == 1 && python_random_raw_draws.numel() == 1,
        "python_random_raw_draws must be int64[1]"
    );
    TORCH_CHECK(
        mutation_py_semantic_draws.scalar_type() == torch::kInt64
        && mutation_py_semantic_draws.dim() == 1 && mutation_py_semantic_draws.numel() == 1,
        "mutation_py_semantic_draws must be int64[1]"
    );

    TORCH_CHECK(genome_int32_fields.size() == 8, "exactly eight int32 genome fields are required");
    for (size_t index = 0; index < genome_int32_fields.size(); ++index) {
        check_slot_tensor(
            genome_int32_fields[index],
            torch::kInt32,
            slot_count,
            "genome_int32_field",
            device
        );
    }
    check_slot_tensor(policy_mem_dim, torch::kInt64, slot_count, "policy_mem_dim", device);
    check_slot_tensor(policy_use_gru, torch::kBool, slot_count, "policy_use_gru", device);
    TORCH_CHECK(genome_float32_fields.size() == 24, "exactly 24 float32 genome/alias fields are required");
    for (size_t index = 0; index < genome_float32_fields.size(); ++index) {
        check_slot_tensor(
            genome_float32_fields[index],
            torch::kFloat32,
            slot_count,
            "genome_float32_field",
            device
        );
    }
    TORCH_CHECK(
        sensor_radius_max >= 0 && sensor_radius_max <= INT32_MAX,
        "sensor_radius_max is out of range"
    );

    const int64_t plan_capacity = child_slots.size(0);
    c10::cuda::CUDAGuard device_guard(device);
    auto int64_options = slot_to_agent_id.options().dtype(torch::kInt64);
    auto int32_options = slot_to_agent_id.options().dtype(torch::kInt32);
    auto committed_count = torch::zeros({1}, int64_options);
    auto semantic_draws_added = torch::zeros({1}, int64_options);
    auto status = torch::zeros({1}, int32_options);
    auto committed_parent_slots = select_turnover_pairs
        ? torch::full({plan_capacity}, -1, int64_options)
        : torch::Tensor();
    auto committed_mate_slots = select_turnover_pairs
        ? torch::full({plan_capacity}, -1, int64_options)
        : torch::Tensor();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fixed_mlp_child_scalar_genome_kernel<<<1, 1, 0, stream>>>(
        slot_to_agent_id.data_ptr<int64_t>(),
        alive_mask.data_ptr<bool>(),
        select_turnover_pairs
            ? nullptr
            : parent_slots_or_pool.data_ptr<int64_t>(),
        select_turnover_pairs
            ? nullptr
            : mate_slots_or_pool_count.data_ptr<int64_t>(),
        select_turnover_pairs
            ? parent_slots_or_pool.data_ptr<int64_t>()
            : nullptr,
        select_turnover_pairs
            ? mate_slots_or_pool_count.data_ptr<int64_t>()
            : nullptr,
        child_slots.data_ptr<int64_t>(),
        python_random_words.data_ptr<uint32_t>(),
        python_random_index.data_ptr<int64_t>(),
        python_random_has_gauss.data_ptr<bool>(),
        python_random_gauss_next.data_ptr<double>(),
        python_random_gauss_primary_tape.numel() > 0
            ? python_random_gauss_primary_tape.data_ptr<double>()
            : nullptr,
        python_random_gauss_secondary_tape.numel() > 0
            ? python_random_gauss_secondary_tape.data_ptr<double>()
            : nullptr,
        python_random_raw_draws.data_ptr<int64_t>(),
        mutation_py_semantic_draws.data_ptr<int64_t>(),
        genome_int32_fields[0].data_ptr<int32_t>(),
        genome_int32_fields[1].data_ptr<int32_t>(),
        genome_int32_fields[2].data_ptr<int32_t>(),
        genome_int32_fields[3].data_ptr<int32_t>(),
        genome_int32_fields[4].data_ptr<int32_t>(),
        genome_int32_fields[5].data_ptr<int32_t>(),
        genome_int32_fields[6].data_ptr<int32_t>(),
        genome_int32_fields[7].data_ptr<int32_t>(),
        policy_mem_dim.data_ptr<int64_t>(),
        policy_use_gru.data_ptr<bool>(),
        genome_float32_fields[0].data_ptr<float>(),
        genome_float32_fields[1].data_ptr<float>(),
        genome_float32_fields[2].data_ptr<float>(),
        genome_float32_fields[3].data_ptr<float>(),
        genome_float32_fields[4].data_ptr<float>(),
        genome_float32_fields[5].data_ptr<float>(),
        genome_float32_fields[6].data_ptr<float>(),
        genome_float32_fields[7].data_ptr<float>(),
        genome_float32_fields[8].data_ptr<float>(),
        genome_float32_fields[9].data_ptr<float>(),
        genome_float32_fields[10].data_ptr<float>(),
        genome_float32_fields[11].data_ptr<float>(),
        genome_float32_fields[12].data_ptr<float>(),
        genome_float32_fields[13].data_ptr<float>(),
        genome_float32_fields[14].data_ptr<float>(),
        genome_float32_fields[15].data_ptr<float>(),
        genome_float32_fields[16].data_ptr<float>(),
        genome_float32_fields[17].data_ptr<float>(),
        genome_float32_fields[18].data_ptr<float>(),
        genome_float32_fields[19].data_ptr<float>(),
        genome_float32_fields[20].data_ptr<float>(),
        genome_float32_fields[21].data_ptr<float>(),
        genome_float32_fields[22].data_ptr<float>(),
        genome_float32_fields[23].data_ptr<float>(),
        select_turnover_pairs
            ? committed_parent_slots.data_ptr<int64_t>()
            : nullptr,
        select_turnover_pairs
            ? committed_mate_slots.data_ptr<int64_t>()
            : nullptr,
        committed_count.data_ptr<int64_t>(),
        semantic_draws_added.data_ptr<int64_t>(),
        status.data_ptr<int32_t>(),
        planned_count.data_ptr<int64_t>(),
        plan_capacity,
        slot_count,
        python_random_gauss_primary_tape.numel(),
        static_cast<int32_t>(sensor_radius_max),
        select_turnover_pairs,
        generation_turnover_refill
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (!select_turnover_pairs) {
        return {committed_count, semantic_draws_added, status};
    }
    return {
        committed_parent_slots,
        committed_mate_slots,
        committed_count,
        semantic_draws_added,
        status,
    };
}

std::vector<torch::Tensor> eiel_cuda_architecture_b_fixed_mlp_child_scalar_genome_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor alive_mask,
    torch::Tensor parent_slots,
    torch::Tensor child_slots,
    torch::Tensor planned_count,
    torch::Tensor python_random_words,
    torch::Tensor python_random_index,
    torch::Tensor python_random_has_gauss,
    torch::Tensor python_random_gauss_next,
    torch::Tensor python_random_gauss_primary_tape,
    torch::Tensor python_random_gauss_secondary_tape,
    torch::Tensor python_random_raw_draws,
    torch::Tensor mutation_py_semantic_draws,
    std::vector<torch::Tensor> genome_int32_fields,
    torch::Tensor policy_mem_dim,
    torch::Tensor policy_use_gru,
    std::vector<torch::Tensor> genome_float32_fields,
    const int64_t sensor_radius_max
) {
    return fixed_mlp_child_scalar_genome_impl(
        slot_to_agent_id, alive_mask, parent_slots, parent_slots, child_slots,
        planned_count, python_random_words, python_random_index,
        python_random_has_gauss, python_random_gauss_next,
        python_random_gauss_primary_tape, python_random_gauss_secondary_tape,
        python_random_raw_draws, mutation_py_semantic_draws,
        std::move(genome_int32_fields), policy_mem_dim, policy_use_gru,
        std::move(genome_float32_fields), sensor_radius_max, false, false
    );
}

std::vector<torch::Tensor> eiel_cuda_architecture_b_turnover_fixed_mlp_child_scalar_genome_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor alive_mask,
    torch::Tensor parent_slots,
    torch::Tensor mate_slots,
    torch::Tensor child_slots,
    torch::Tensor planned_count,
    torch::Tensor python_random_words,
    torch::Tensor python_random_index,
    torch::Tensor python_random_has_gauss,
    torch::Tensor python_random_gauss_next,
    torch::Tensor python_random_gauss_primary_tape,
    torch::Tensor python_random_gauss_secondary_tape,
    torch::Tensor python_random_raw_draws,
    torch::Tensor mutation_py_semantic_draws,
    std::vector<torch::Tensor> genome_int32_fields,
    torch::Tensor policy_mem_dim,
    torch::Tensor policy_use_gru,
    std::vector<torch::Tensor> genome_float32_fields,
    const int64_t sensor_radius_max
) {
    return fixed_mlp_child_scalar_genome_impl(
        slot_to_agent_id, alive_mask, parent_slots, mate_slots, child_slots,
        planned_count, python_random_words, python_random_index,
        python_random_has_gauss, python_random_gauss_next,
        python_random_gauss_primary_tape, python_random_gauss_secondary_tape,
        python_random_raw_draws, mutation_py_semantic_draws,
        std::move(genome_int32_fields), policy_mem_dim, policy_use_gru,
        std::move(genome_float32_fields), sensor_radius_max, true, false
    );
}

std::vector<torch::Tensor>
eiel_cuda_architecture_b_turnover_interleaved_fixed_mlp_child_scalar_genome_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor alive_mask,
    torch::Tensor parent_pool_slots,
    torch::Tensor parent_pool_count,
    torch::Tensor child_slots,
    torch::Tensor planned_count,
    torch::Tensor python_random_words,
    torch::Tensor python_random_index,
    torch::Tensor python_random_has_gauss,
    torch::Tensor python_random_gauss_next,
    torch::Tensor python_random_gauss_primary_tape,
    torch::Tensor python_random_gauss_secondary_tape,
    torch::Tensor python_random_raw_draws,
    torch::Tensor mutation_py_semantic_draws,
    std::vector<torch::Tensor> genome_int32_fields,
    torch::Tensor policy_mem_dim,
    torch::Tensor policy_use_gru,
    std::vector<torch::Tensor> genome_float32_fields,
    const int64_t sensor_radius_max
) {
    return fixed_mlp_child_scalar_genome_impl(
        slot_to_agent_id, alive_mask, parent_pool_slots, parent_pool_count,
        child_slots, planned_count, python_random_words, python_random_index,
        python_random_has_gauss, python_random_gauss_next,
        python_random_gauss_primary_tape, python_random_gauss_secondary_tape,
        python_random_raw_draws, mutation_py_semantic_draws,
        std::move(genome_int32_fields), policy_mem_dim, policy_use_gru,
        std::move(genome_float32_fields), sensor_radius_max, true, true
    );
}
