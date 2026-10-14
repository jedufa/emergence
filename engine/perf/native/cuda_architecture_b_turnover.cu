#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "cuda_architecture_b_pcg64.cuh"

namespace {

constexpr int32_t STATUS_OK = 0;
constexpr int32_t STATUS_INVALID_CONFIG = 1;
constexpr int32_t STATUS_INVALID_RESIDENT_STATE = 2;
constexpr int32_t STATUS_NONUNIFORM_EARLY_LIFE_WEIGHT = 3;
constexpr int32_t STATUS_LOW_POPULATION_UNSUPPORTED = 4;
constexpr int32_t STATUS_POST_CULL_LOW_POPULATION_UNSUPPORTED = 5;
constexpr int32_t STATUS_INSUFFICIENT_PARENT_POOL = 6;

constexpr int32_t CULL_STATUS_OK = 0;
constexpr int32_t CULL_STATUS_PLAN_FAILED = 1;
constexpr int32_t CULL_STATUS_INVALID_RESIDENT_STATE = 2;
constexpr int32_t CULL_STATUS_QUEUE_CAPACITY_EXCEEDED = 3;
constexpr int32_t CULL_STATUS_RELEASE_CONTRADICTION = 4;

constexpr int32_t REFILL_PLAN_STATUS_OK = 0;
constexpr int32_t REFILL_PLAN_STATUS_TURNOVER_FAILED = 1;
constexpr int32_t REFILL_PLAN_STATUS_INVALID_STATE = 2;
constexpr int32_t REFILL_PLAN_STATUS_INVALID_PARENT_POOL = 3;
constexpr int32_t REFILL_PLAN_STATUS_NO_SPAWN_SPACE = 4;

struct DeviceCpythonChoiceRng {
    uint32_t words[624];
    int64_t index;
    int64_t raw_draws;

    __device__ __forceinline__ void twist() {
        constexpr uint32_t matrix_a = 0x9908b0dfU;
        constexpr uint32_t upper_mask = 0x80000000U;
        constexpr uint32_t lower_mask = 0x7fffffffU;
        for (int64_t row = 0; row < 624 - 397; ++row) {
            const uint32_t y = (words[row] & upper_mask)
                | (words[row + 1] & lower_mask);
            words[row] = words[row + 397] ^ (y >> 1U)
                ^ ((y & 1U) ? matrix_a : 0U);
        }
        for (int64_t row = 624 - 397; row < 623; ++row) {
            const uint32_t y = (words[row] & upper_mask)
                | (words[row + 1] & lower_mask);
            words[row] = words[row + (397 - 624)] ^ (y >> 1U)
                ^ ((y & 1U) ? matrix_a : 0U);
        }
        const uint32_t y = (words[623] & upper_mask) | (words[0] & lower_mask);
        words[623] = words[396] ^ (y >> 1U) ^ ((y & 1U) ? matrix_a : 0U);
        index = 0;
    }

    __device__ __forceinline__ uint32_t raw32() {
        if (index >= 624) {
            twist();
        }
        uint32_t value = words[index++];
        value ^= value >> 11U;
        value ^= (value << 7U) & 0x9d2c5680U;
        value ^= (value << 15U) & 0xefc60000U;
        value ^= value >> 18U;
        ++raw_draws;
        return value;
    }

    __device__ __forceinline__ uint64_t getrandbits(const int64_t bit_count) {
        if (bit_count == 0) {
            return 0;
        }
        if (bit_count <= 32) {
            return static_cast<uint64_t>(raw32() >> (32 - bit_count));
        }
        uint64_t value = 0;
        int64_t filled = 0;
        int64_t remaining = bit_count;
        while (remaining > 0) {
            const int64_t take = remaining < 32 ? remaining : 32;
            uint32_t word = raw32();
            if (take < 32) {
                word >>= 32 - take;
            }
            value |= static_cast<uint64_t>(word) << filled;
            filled += take;
            remaining -= take;
        }
        return value;
    }

    __device__ __forceinline__ uint64_t randbelow(const uint64_t bound) {
        int64_t bit_count = 0;
        for (uint64_t value = bound; value != 0; value >>= 1U) {
            ++bit_count;
        }
        uint64_t value = getrandbits(bit_count);
        while (value >= bound) {
            value = getrandbits(bit_count);
        }
        return value;
    }
};

__device__ __forceinline__ int64_t pcg64_numpy_int64_integer(
    eiel::cuda_architecture_b::Pcg64State& rng,
    const int64_t high_exclusive
) {
    if (high_exclusive == 1) {
        return 0;
    }
    const uint32_t range = static_cast<uint32_t>(high_exclusive);
    const uint32_t threshold = static_cast<uint32_t>((0U - range) % range);
    while (true) {
        const uint32_t raw = eiel::cuda_architecture_b::pcg64_raw32(rng);
        const uint64_t product = static_cast<uint64_t>(raw)
            * static_cast<uint64_t>(range);
        if (static_cast<uint32_t>(product) >= threshold) {
            return static_cast<int64_t>(product >> 32U);
        }
    }
}

__global__ void turnover_refill_plan_kernel(
    const int64_t* __restrict__ slot_to_agent_id,
    const bool* __restrict__ alive_mask,
    const int32_t* __restrict__ agent_grid,
    const int64_t* __restrict__ parent_pool_slots,
    const int64_t* __restrict__ parent_pool_count,
    const int64_t* __restrict__ refill_count,
    const int32_t* __restrict__ turnover_status,
    uint32_t* __restrict__ python_random_words,
    int64_t* __restrict__ python_random_index,
    int64_t* __restrict__ python_random_raw_draws,
    int64_t* __restrict__ mutation_py_semantic_draws,
    uint64_t* __restrict__ env_pcg64_state_words,
    int64_t* __restrict__ env_pcg64_buffer_words,
    int64_t* __restrict__ policy_np_semantic_draws,
    int64_t* __restrict__ planned_parent_slots,
    int64_t* __restrict__ planned_mate_slots,
    int32_t* __restrict__ planned_spawn_positions,
    int64_t* __restrict__ spawn_attempts_by_birth,
    int64_t* __restrict__ planned_count,
    int64_t* __restrict__ python_raw_draws_added,
    int64_t* __restrict__ python_semantic_draws_added,
    int64_t* __restrict__ spawn_raw_draws_added,
    int64_t* __restrict__ spawn_semantic_draws_added,
    int32_t* __restrict__ status,
    const int64_t slot_count,
    const int64_t grid_height,
    const int64_t grid_width,
    const bool env_policy_alias
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (turnover_status[0] != STATUS_OK) {
        status[0] = REFILL_PLAN_STATUS_TURNOVER_FAILED;
        return;
    }
    const int64_t pool_count = parent_pool_count[0];
    const int64_t births = refill_count[0];
    if (births < 0 || births > slot_count || pool_count < (births > 0 ? 2 : 0)
        || pool_count > slot_count || python_random_index[0] < 0
        || python_random_index[0] > 624 || python_random_raw_draws[0] < 0
        || mutation_py_semantic_draws[0] < 0 || policy_np_semantic_draws[0] < 0
        || grid_height <= 0 || grid_width <= 0
        || grid_height > static_cast<int64_t>(UINT32_MAX)
        || grid_width > static_cast<int64_t>(UINT32_MAX)) {
        status[0] = REFILL_PLAN_STATUS_INVALID_STATE;
        return;
    }
    if (births == 0) {
        planned_count[0] = 0;
        status[0] = REFILL_PLAN_STATUS_OK;
        return;
    }
    for (int64_t row = 0; row < pool_count; ++row) {
        const int64_t slot = parent_pool_slots[row];
        if (slot < 0 || slot >= slot_count || !alive_mask[slot]
            || slot_to_agent_id[slot] < 0) {
            status[0] = REFILL_PLAN_STATUS_INVALID_PARENT_POOL;
            return;
        }
        for (int64_t previous = 0; previous < row; ++previous) {
            if (parent_pool_slots[previous] == slot) {
                status[0] = REFILL_PLAN_STATUS_INVALID_PARENT_POOL;
                return;
            }
        }
    }

    DeviceCpythonChoiceRng py_rng{};
    for (int64_t row = 0; row < 624; ++row) {
        py_rng.words[row] = python_random_words[row];
    }
    py_rng.index = python_random_index[0];
    py_rng.raw_draws = python_random_raw_draws[0];
    const int64_t initial_python_raw_draws = py_rng.raw_draws;
    int64_t python_semantic_added = 0;

    eiel::cuda_architecture_b::Pcg64State env_rng{
        env_pcg64_state_words[0],
        env_pcg64_state_words[1],
        env_pcg64_state_words[2],
        env_pcg64_state_words[3],
        static_cast<uint32_t>(env_pcg64_buffer_words[0]),
        static_cast<uint32_t>(env_pcg64_buffer_words[1]),
    };
    int64_t spawn_draws = 0;
    int64_t spawn_semantic = 0;

    for (int64_t birth = 0; birth < births; ++birth) {
        int64_t parent_slot = -1;
        int64_t mate_slot = -1;
        int64_t pair_attempts = 0;
        do {
            parent_slot = parent_pool_slots[py_rng.randbelow(
                static_cast<uint64_t>(pool_count)
            )];
            mate_slot = parent_pool_slots[py_rng.randbelow(
                static_cast<uint64_t>(pool_count)
            )];
            python_semantic_added += 2;
            ++pair_attempts;
            if (pair_attempts > 1000000) {
                status[0] = REFILL_PLAN_STATUS_INVALID_PARENT_POOL;
                return;
            }
        } while (mate_slot == parent_slot);

        int32_t selected_y = -1;
        int32_t selected_x = -1;
        int64_t attempts = 0;
        for (; attempts < 200; ++attempts) {
            const int32_t y = static_cast<int32_t>(pcg64_numpy_int64_integer(
                env_rng, grid_height
            ));
            const int32_t x = static_cast<int32_t>(pcg64_numpy_int64_integer(
                env_rng, grid_width
            ));
            spawn_draws += 2;
            spawn_semantic += 2;
            if (agent_grid[y * grid_width + x] != -1) {
                continue;
            }
            bool reserved = false;
            for (int64_t previous = 0; previous < birth; ++previous) {
                if (planned_spawn_positions[previous * 2] == y
                    && planned_spawn_positions[previous * 2 + 1] == x) {
                    reserved = true;
                    break;
                }
            }
            if (reserved) {
                continue;
            }
            selected_y = y;
            selected_x = x;
            break;
        }
        if (selected_y < 0 || selected_x < 0) {
            status[0] = REFILL_PLAN_STATUS_NO_SPAWN_SPACE;
            return;
        }
        planned_parent_slots[birth] = parent_slot;
        planned_mate_slots[birth] = mate_slot;
        planned_spawn_positions[birth * 2] = selected_y;
        planned_spawn_positions[birth * 2 + 1] = selected_x;
        spawn_attempts_by_birth[birth] = attempts + 1;
    }

    for (int64_t row = 0; row < 624; ++row) {
        python_random_words[row] = py_rng.words[row];
    }
    python_random_index[0] = py_rng.index;
    python_random_raw_draws[0] = py_rng.raw_draws;
    mutation_py_semantic_draws[0] += python_semantic_added;
    env_pcg64_state_words[0] = env_rng.state_lo;
    env_pcg64_state_words[1] = env_rng.state_hi;
    env_pcg64_state_words[2] = env_rng.inc_lo;
    env_pcg64_state_words[3] = env_rng.inc_hi;
    env_pcg64_buffer_words[0] = static_cast<int64_t>(env_rng.has_uint32);
    env_pcg64_buffer_words[1] = static_cast<int64_t>(env_rng.uinteger);
    if (env_policy_alias) {
        policy_np_semantic_draws[0] += spawn_semantic;
    }
    planned_count[0] = births;
    python_raw_draws_added[0] = py_rng.raw_draws - initial_python_raw_draws;
    python_semantic_draws_added[0] = python_semantic_added;
    spawn_raw_draws_added[0] = spawn_draws;
    spawn_semantic_draws_added[0] = spawn_semantic;
    status[0] = REFILL_PLAN_STATUS_OK;
}

__device__ __forceinline__ double canonical_quiet_nan() {
    return __longlong_as_double(static_cast<long long>(0x7ff8000000000000ULL));
}

__device__ __forceinline__ double maximum_one(const double value) {
    return value > 1.0 ? value : 1.0;
}

__device__ __forceinline__ double effective_fitness(
    const int64_t slot,
    const int32_t* __restrict__ survival_steps,
    const double* __restrict__ energy_collected_exact,
    const int32_t* __restrict__ offspring_count,
    const int32_t* __restrict__ attacks,
    const int32_t* __restrict__ early_survival_steps,
    const double* __restrict__ early_energy_collected_exact,
    const float* __restrict__ early_injury_gained,
    const int32_t* __restrict__ action_switches,
    const int32_t* __restrict__ action_oscillations,
    const float* __restrict__ drive_volatility,
    const int32_t* __restrict__ selfharm_events,
    const float* __restrict__ pred_err_sum,
    const int32_t* __restrict__ pred_err_count,
    const double coherence_thrashing_weight,
    const double coherence_drive_volatility_weight,
    const double coherence_selfharm_weight,
    const double coherence_pred_err_weight
) {
    const double survival = static_cast<double>(survival_steps[slot]);
    const double offspring = static_cast<double>(offspring_count[slot]);
    double score = (survival < 120.0 ? survival : 120.0) * 0.05;
    if (offspring > 0.0) {
        score += 180.0;
    }
    if (offspring > 1.0) {
        score += 120.0;
    }
    if (offspring > 2.0) {
        score += 60.0 * (offspring - 2.0);
    }
    score += (energy_collected_exact[slot] / maximum_one(survival)) * 80.0;
    score -= static_cast<double>(attacks[slot]) * 2.5;

    const double early_survival = static_cast<double>(early_survival_steps[slot]);
    const double early_component = 0.10 * early_survival
        + 80.0 * (early_energy_collected_exact[slot] / maximum_one(early_survival))
        - 10.0 * static_cast<double>(early_injury_gained[slot]);
    score += early_component;
    score += 10.0;

    const double switches = static_cast<double>(action_switches[slot]);
    const double oscillations = static_cast<double>(action_oscillations[slot]);
    const double thrash = 0.5 * switches + oscillations;
    const double pred_mean = static_cast<double>(pred_err_sum[slot])
        / maximum_one(static_cast<double>(pred_err_count[slot]));
    return score
        - coherence_thrashing_weight * thrash
        - coherence_drive_volatility_weight * static_cast<double>(drive_volatility[slot])
        - coherence_selfharm_weight * static_cast<double>(selfharm_events[slot])
        - coherence_pred_err_weight * pred_mean;
}

__global__ void turnover_plan_kernel(
    const int64_t* __restrict__ slot_to_agent_id,
    const bool* __restrict__ alive_mask,
    const int32_t* __restrict__ survival_steps,
    const double* __restrict__ energy_collected_exact,
    const int32_t* __restrict__ offspring_count,
    const int32_t* __restrict__ attacks,
    const int32_t* __restrict__ early_survival_steps,
    const double* __restrict__ early_energy_collected_exact,
    const float* __restrict__ early_injury_gained,
    const float* __restrict__ early_life_weight,
    const int32_t* __restrict__ action_switches,
    const int32_t* __restrict__ action_oscillations,
    const float* __restrict__ drive_volatility,
    const int32_t* __restrict__ selfharm_events,
    const float* __restrict__ pred_err_sum,
    const int32_t* __restrict__ pred_err_count,
    int64_t* __restrict__ ranked_slots,
    int64_t* __restrict__ ranked_agent_ids,
    double* __restrict__ ranked_fitness,
    int64_t* __restrict__ ranked_count,
    int64_t* __restrict__ culled_slots,
    int64_t* __restrict__ culled_agent_ids,
    int64_t* __restrict__ culled_count,
    int64_t* __restrict__ parent_pool_slots,
    int64_t* __restrict__ parent_pool_agent_ids,
    int64_t* __restrict__ parent_pool_count,
    int64_t* __restrict__ survivor_count,
    int64_t* __restrict__ refill_count,
    bool* __restrict__ hardcap,
    int32_t* __restrict__ status,
    const int64_t slot_count,
    const int64_t population_size,
    const double cull_fraction,
    const double parent_fraction,
    const double coherence_thrashing_weight,
    const double coherence_drive_volatility_weight,
    const double coherence_selfharm_weight,
    const double coherence_pred_err_weight
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (population_size <= 0 || !::isfinite(cull_fraction)
        || cull_fraction < 0.0 || cull_fraction > 1.0
        || !::isfinite(parent_fraction) || parent_fraction < 0.0
        || parent_fraction > 1.0
        || !::isfinite(coherence_thrashing_weight)
        || !::isfinite(coherence_drive_volatility_weight)
        || !::isfinite(coherence_selfharm_weight)
        || !::isfinite(coherence_pred_err_weight)) {
        status[0] = STATUS_INVALID_CONFIG;
        return;
    }

    int64_t live_count = 0;
    for (int64_t slot = 0; slot < slot_count; ++slot) {
        if (!alive_mask[slot]) {
            if (slot_to_agent_id[slot] >= 0) {
                status[0] = STATUS_INVALID_RESIDENT_STATE;
                return;
            }
            continue;
        }
        if (slot_to_agent_id[slot] < 0 || !::isfinite(early_life_weight[slot])) {
            status[0] = STATUS_INVALID_RESIDENT_STATE;
            return;
        }
        if (early_life_weight[slot] != 1.0f) {
            status[0] = STATUS_NONUNIFORM_EARLY_LIFE_WEIGHT;
            return;
        }
        const double fitness = effective_fitness(
            slot,
            survival_steps,
            energy_collected_exact,
            offspring_count,
            attacks,
            early_survival_steps,
            early_energy_collected_exact,
            early_injury_gained,
            action_switches,
            action_oscillations,
            drive_volatility,
            selfharm_events,
            pred_err_sum,
            pred_err_count,
            coherence_thrashing_weight,
            coherence_drive_volatility_weight,
            coherence_selfharm_weight,
            coherence_pred_err_weight
        );
        if (!::isfinite(fitness)) {
            status[0] = STATUS_INVALID_RESIDENT_STATE;
            return;
        }

        int64_t insert_at = live_count;
        while (insert_at > 0) {
            const double previous_fitness = ranked_fitness[insert_at - 1];
            const int64_t previous_agent_id = ranked_agent_ids[insert_at - 1];
            if (previous_fitness > fitness
                || (previous_fitness == fitness
                    && previous_agent_id < slot_to_agent_id[slot])) {
                break;
            }
            ranked_fitness[insert_at] = previous_fitness;
            ranked_agent_ids[insert_at] = previous_agent_id;
            ranked_slots[insert_at] = ranked_slots[insert_at - 1];
            --insert_at;
        }
        ranked_fitness[insert_at] = fitness;
        ranked_agent_ids[insert_at] = slot_to_agent_id[slot];
        ranked_slots[insert_at] = slot;
        ++live_count;
    }
    ranked_count[0] = live_count;
    if (live_count < 2) {
        status[0] = STATUS_LOW_POPULATION_UNSUPPORTED;
        return;
    }

    const bool hardcap_value = live_count > population_size;
    hardcap[0] = hardcap_value;
    int64_t survivors = population_size;
    int64_t cull_rows = live_count - population_size;
    int64_t refill_rows = 0;
    int64_t selected_parents = 0;
    if (!hardcap_value) {
        cull_rows = static_cast<int64_t>(static_cast<double>(live_count) * cull_fraction);
        survivors = live_count - cull_rows;
        if (survivors < 2) {
            status[0] = STATUS_POST_CULL_LOW_POPULATION_UNSUPPORTED;
            return;
        }
        refill_rows = population_size - survivors;
        if (refill_rows < 0 || population_size > slot_count) {
            status[0] = STATUS_INVALID_CONFIG;
            return;
        }
        if (refill_rows > 0) {
            const int64_t fraction_parents = static_cast<int64_t>(
                static_cast<double>(survivors) * parent_fraction
            );
            const int64_t requested_parents = fraction_parents > 2 ? fraction_parents : 2;
            selected_parents = requested_parents < survivors ? requested_parents : survivors;
            if (selected_parents < 2) {
                status[0] = STATUS_INSUFFICIENT_PARENT_POOL;
                return;
            }
        }
    }

    const int64_t cull_start = live_count - cull_rows;
    for (int64_t index = 0; index < cull_rows; ++index) {
        culled_slots[index] = ranked_slots[cull_start + index];
        culled_agent_ids[index] = ranked_agent_ids[cull_start + index];
    }
    for (int64_t index = 0; index < selected_parents; ++index) {
        parent_pool_slots[index] = ranked_slots[index];
        parent_pool_agent_ids[index] = ranked_agent_ids[index];
    }
    culled_count[0] = cull_rows;
    parent_pool_count[0] = selected_parents;
    survivor_count[0] = survivors;
    refill_count[0] = refill_rows;
    status[0] = STATUS_OK;
}

__global__ void turnover_cull_release_kernel(
    int64_t* __restrict__ slot_to_agent_id,
    int32_t* __restrict__ agent_id_to_slot,
    bool* __restrict__ alive_mask,
    int32_t* __restrict__ pos_y,
    int32_t* __restrict__ pos_x,
    int32_t* __restrict__ agent_grid,
    float* __restrict__ energy,
    float* __restrict__ health,
    double* __restrict__ policy_energy_observation,
    int64_t* __restrict__ action_cache_token,
    int32_t* __restrict__ action_cache_cursor,
    int32_t* __restrict__ action_cache_next,
    int32_t* __restrict__ action_cache_actions,
    int64_t* __restrict__ slot_free_queue,
    int64_t* __restrict__ slot_free_queue_count,
    const int64_t* __restrict__ pending_free_agent_ids,
    const int64_t* __restrict__ pending_free_slots,
    int64_t* __restrict__ pending_free_count,
    const int64_t* __restrict__ culled_agent_ids,
    const int64_t* __restrict__ culled_slots,
    const int64_t* __restrict__ culled_count,
    const int32_t* __restrict__ turnover_plan_status,
    int64_t* __restrict__ released_agent_ids,
    int64_t* __restrict__ released_slots,
    int64_t* __restrict__ released_count,
    int64_t* __restrict__ pending_released_count,
    int64_t* __restrict__ culled_released_count,
    int32_t* __restrict__ status,
    const int64_t slot_count,
    const int64_t agent_id_capacity,
    const int64_t grid_height,
    const int64_t grid_width,
    const int64_t action_cache_width,
    const int64_t free_queue_capacity,
    const int64_t pending_capacity,
    const int64_t cull_capacity
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (turnover_plan_status[0] != STATUS_OK) {
        status[0] = CULL_STATUS_PLAN_FAILED;
        return;
    }
    const int64_t queue_count = slot_free_queue_count[0];
    const int64_t pending_count = pending_free_count[0];
    const int64_t cull_rows = culled_count[0];
    if (queue_count < 0 || queue_count > free_queue_capacity
        || pending_count < 0 || pending_count > pending_capacity
        || cull_rows < 0 || cull_rows > cull_capacity) {
        status[0] = CULL_STATUS_INVALID_RESIDENT_STATE;
        return;
    }
    const int64_t release_rows = pending_count + cull_rows;
    if (release_rows > free_queue_capacity - queue_count) {
        status[0] = CULL_STATUS_QUEUE_CAPACITY_EXCEEDED;
        return;
    }

    for (int64_t queue_index = 0; queue_index < queue_count; ++queue_index) {
        const int64_t slot = slot_free_queue[queue_index];
        if (slot < 0 || slot >= slot_count || alive_mask[slot]
            || slot_to_agent_id[slot] >= 0) {
            status[0] = CULL_STATUS_RELEASE_CONTRADICTION;
            return;
        }
        for (int64_t previous = 0; previous < queue_index; ++previous) {
            if (slot_free_queue[previous] == slot) {
                status[0] = CULL_STATUS_RELEASE_CONTRADICTION;
                return;
            }
        }
    }

    for (int64_t index = 0; index < release_rows; ++index) {
        const bool pending = index < pending_count;
        const int64_t source_index = pending ? index : index - pending_count;
        const int64_t agent_id = pending
            ? pending_free_agent_ids[source_index]
            : culled_agent_ids[source_index];
        const int64_t slot = pending
            ? pending_free_slots[source_index]
            : culled_slots[source_index];
        if (agent_id < 0 || slot < 0 || slot >= slot_count) {
            status[0] = CULL_STATUS_RELEASE_CONTRADICTION;
            return;
        }
        if (pending) {
            if (alive_mask[slot] || slot_to_agent_id[slot] >= 0
                || (
                    agent_id < agent_id_capacity
                    && agent_id_to_slot[agent_id] >= 0
                )) {
                status[0] = CULL_STATUS_RELEASE_CONTRADICTION;
                return;
            }
        } else {
            const int32_t y = pos_y[slot];
            const int32_t x = pos_x[slot];
            if (!alive_mask[slot] || slot_to_agent_id[slot] != agent_id
                || y < 0 || y >= grid_height || x < 0 || x >= grid_width
                || agent_grid[y * grid_width + x]
                    != static_cast<int32_t>(agent_id)) {
                status[0] = CULL_STATUS_RELEASE_CONTRADICTION;
                return;
            }
        }
        for (int64_t queue_index = 0; queue_index < queue_count; ++queue_index) {
            if (slot_free_queue[queue_index] == slot) {
                status[0] = CULL_STATUS_RELEASE_CONTRADICTION;
                return;
            }
        }
        for (int64_t previous = 0; previous < index; ++previous) {
            if (released_agent_ids[previous] == agent_id
                || released_slots[previous] == slot) {
                status[0] = CULL_STATUS_RELEASE_CONTRADICTION;
                return;
            }
        }
        int64_t insert_at = index;
        while (insert_at > 0) {
            const int64_t previous_agent_id = released_agent_ids[insert_at - 1];
            const int64_t previous_slot = released_slots[insert_at - 1];
            if (previous_agent_id < agent_id
                || (previous_agent_id == agent_id && previous_slot < slot)) {
                break;
            }
            released_agent_ids[insert_at] = previous_agent_id;
            released_slots[insert_at] = previous_slot;
            --insert_at;
        }
        released_agent_ids[insert_at] = agent_id;
        released_slots[insert_at] = slot;
    }

    for (int64_t index = 0; index < cull_rows; ++index) {
        const int64_t agent_id = culled_agent_ids[index];
        const int64_t slot = culled_slots[index];
        const int32_t y = pos_y[slot];
        const int32_t x = pos_x[slot];
        agent_grid[y * grid_width + x] = -1;
        if (agent_id < agent_id_capacity) {
            agent_id_to_slot[agent_id] = -1;
        }
        alive_mask[slot] = false;
        slot_to_agent_id[slot] = -1;
        pos_y[slot] = -1;
        pos_x[slot] = -1;
        energy[slot] = 0.0f;
        health[slot] = 0.0f;
        policy_energy_observation[slot] = canonical_quiet_nan();
        action_cache_token[slot] = -1;
        action_cache_cursor[slot] = -1;
        action_cache_next[slot] = -1;
        for (int64_t column = 0; column < action_cache_width; ++column) {
            action_cache_actions[slot * action_cache_width + column] = -1;
        }
    }
    for (int64_t index = 0; index < release_rows; ++index) {
        const int64_t slot = released_slots[index];
        policy_energy_observation[slot] = canonical_quiet_nan();
        slot_free_queue[queue_count + index] = slot;
    }
    slot_free_queue_count[0] = queue_count + release_rows;
    pending_free_count[0] = 0;
    released_count[0] = release_rows;
    pending_released_count[0] = pending_count;
    culled_released_count[0] = cull_rows;
    status[0] = CULL_STATUS_OK;
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA-resident");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::vector<torch::Tensor> eiel_cuda_architecture_b_turnover_plan_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor alive_mask,
    torch::Tensor survival_steps,
    torch::Tensor energy_collected_exact,
    torch::Tensor offspring_count,
    torch::Tensor attacks,
    torch::Tensor early_survival_steps,
    torch::Tensor early_energy_collected_exact,
    torch::Tensor early_injury_gained,
    torch::Tensor early_life_weight,
    torch::Tensor action_switches,
    torch::Tensor action_oscillations,
    torch::Tensor drive_volatility,
    torch::Tensor selfharm_events,
    torch::Tensor pred_err_sum,
    torch::Tensor pred_err_count,
    const int64_t population_size,
    const double cull_fraction,
    const double parent_fraction,
    const double coherence_thrashing_weight,
    const double coherence_drive_volatility_weight,
    const double coherence_selfharm_weight,
    const double coherence_pred_err_weight
) {
    const std::vector<std::pair<const char*, torch::Tensor>> tensors = {
        {"slot_to_agent_id", slot_to_agent_id},
        {"alive_mask", alive_mask},
        {"survival_steps", survival_steps},
        {"energy_collected_exact", energy_collected_exact},
        {"offspring_count", offspring_count},
        {"attacks", attacks},
        {"early_survival_steps", early_survival_steps},
        {"early_energy_collected_exact", early_energy_collected_exact},
        {"early_injury_gained", early_injury_gained},
        {"early_life_weight", early_life_weight},
        {"action_switches", action_switches},
        {"action_oscillations", action_oscillations},
        {"drive_volatility", drive_volatility},
        {"selfharm_events", selfharm_events},
        {"pred_err_sum", pred_err_sum},
        {"pred_err_count", pred_err_count},
    };
    for (const auto& item : tensors) {
        check_cuda_contiguous(item.second, item.first);
    }
    const auto device = slot_to_agent_id.device();
    for (const auto& item : tensors) {
        TORCH_CHECK(item.second.device() == device, item.first, " must share one CUDA device");
        TORCH_CHECK(item.second.dim() == 1, item.first, " must be rank 1");
        TORCH_CHECK(item.second.numel() == slot_to_agent_id.numel(), item.first, " must share slot capacity");
    }
    TORCH_CHECK(slot_to_agent_id.scalar_type() == torch::kInt64, "slot_to_agent_id must be int64");
    TORCH_CHECK(alive_mask.scalar_type() == torch::kBool, "alive_mask must be bool");
    TORCH_CHECK(survival_steps.scalar_type() == torch::kInt32, "survival_steps must be int32");
    TORCH_CHECK(energy_collected_exact.scalar_type() == torch::kFloat64, "energy_collected_exact must be float64");
    TORCH_CHECK(offspring_count.scalar_type() == torch::kInt32, "offspring_count must be int32");
    TORCH_CHECK(attacks.scalar_type() == torch::kInt32, "attacks must be int32");
    TORCH_CHECK(early_survival_steps.scalar_type() == torch::kInt32, "early_survival_steps must be int32");
    TORCH_CHECK(early_energy_collected_exact.scalar_type() == torch::kFloat64, "early_energy_collected_exact must be float64");
    TORCH_CHECK(early_injury_gained.scalar_type() == torch::kFloat32, "early_injury_gained must be float32");
    TORCH_CHECK(early_life_weight.scalar_type() == torch::kFloat32, "early_life_weight must be float32");
    TORCH_CHECK(action_switches.scalar_type() == torch::kInt32, "action_switches must be int32");
    TORCH_CHECK(action_oscillations.scalar_type() == torch::kInt32, "action_oscillations must be int32");
    TORCH_CHECK(drive_volatility.scalar_type() == torch::kFloat32, "drive_volatility must be float32");
    TORCH_CHECK(selfharm_events.scalar_type() == torch::kInt32, "selfharm_events must be int32");
    TORCH_CHECK(pred_err_sum.scalar_type() == torch::kFloat32, "pred_err_sum must be float32");
    TORCH_CHECK(pred_err_count.scalar_type() == torch::kInt32, "pred_err_count must be int32");

    c10::cuda::CUDAGuard guard(device);
    const int64_t capacity = slot_to_agent_id.numel();
    auto int64_options = slot_to_agent_id.options().dtype(torch::kInt64);
    auto float64_options = slot_to_agent_id.options().dtype(torch::kFloat64);
    auto int32_options = slot_to_agent_id.options().dtype(torch::kInt32);
    auto ranked_slots = torch::full({capacity}, -1, int64_options);
    auto ranked_agent_ids = torch::full({capacity}, -1, int64_options);
    auto ranked_fitness = torch::full(
        {capacity},
        std::numeric_limits<double>::quiet_NaN(),
        float64_options
    );
    auto ranked_count = torch::zeros({1}, int64_options);
    auto culled_slots = torch::full({capacity}, -1, int64_options);
    auto culled_agent_ids = torch::full({capacity}, -1, int64_options);
    auto culled_count = torch::zeros({1}, int64_options);
    auto parent_pool_slots = torch::full({capacity}, -1, int64_options);
    auto parent_pool_agent_ids = torch::full({capacity}, -1, int64_options);
    auto parent_pool_count = torch::zeros({1}, int64_options);
    auto survivor_count = torch::zeros({1}, int64_options);
    auto refill_count = torch::zeros({1}, int64_options);
    auto hardcap = torch::zeros({1}, slot_to_agent_id.options().dtype(torch::kBool));
    auto status = torch::zeros({1}, int32_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    turnover_plan_kernel<<<1, 1, 0, stream>>>(
        slot_to_agent_id.data_ptr<int64_t>(),
        alive_mask.data_ptr<bool>(),
        survival_steps.data_ptr<int32_t>(),
        energy_collected_exact.data_ptr<double>(),
        offspring_count.data_ptr<int32_t>(),
        attacks.data_ptr<int32_t>(),
        early_survival_steps.data_ptr<int32_t>(),
        early_energy_collected_exact.data_ptr<double>(),
        early_injury_gained.data_ptr<float>(),
        early_life_weight.data_ptr<float>(),
        action_switches.data_ptr<int32_t>(),
        action_oscillations.data_ptr<int32_t>(),
        drive_volatility.data_ptr<float>(),
        selfharm_events.data_ptr<int32_t>(),
        pred_err_sum.data_ptr<float>(),
        pred_err_count.data_ptr<int32_t>(),
        ranked_slots.data_ptr<int64_t>(),
        ranked_agent_ids.data_ptr<int64_t>(),
        ranked_fitness.data_ptr<double>(),
        ranked_count.data_ptr<int64_t>(),
        culled_slots.data_ptr<int64_t>(),
        culled_agent_ids.data_ptr<int64_t>(),
        culled_count.data_ptr<int64_t>(),
        parent_pool_slots.data_ptr<int64_t>(),
        parent_pool_agent_ids.data_ptr<int64_t>(),
        parent_pool_count.data_ptr<int64_t>(),
        survivor_count.data_ptr<int64_t>(),
        refill_count.data_ptr<int64_t>(),
        hardcap.data_ptr<bool>(),
        status.data_ptr<int32_t>(),
        capacity,
        population_size,
        cull_fraction,
        parent_fraction,
        coherence_thrashing_weight,
        coherence_drive_volatility_weight,
        coherence_selfharm_weight,
        coherence_pred_err_weight
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        ranked_slots,
        ranked_agent_ids,
        ranked_fitness,
        ranked_count,
        culled_slots,
        culled_agent_ids,
        culled_count,
        parent_pool_slots,
        parent_pool_agent_ids,
        parent_pool_count,
        survivor_count,
        refill_count,
        hardcap,
        status,
    };
}

std::vector<torch::Tensor> eiel_cuda_architecture_b_turnover_cull_release_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor agent_id_to_slot,
    torch::Tensor alive_mask,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor energy,
    torch::Tensor health,
    torch::Tensor policy_energy_observation,
    torch::Tensor action_cache_token,
    torch::Tensor action_cache_cursor,
    torch::Tensor action_cache_next,
    torch::Tensor action_cache_actions,
    torch::Tensor slot_free_queue,
    torch::Tensor slot_free_queue_count,
    torch::Tensor pending_free_agent_ids,
    torch::Tensor pending_free_slots,
    torch::Tensor pending_free_count,
    torch::Tensor culled_agent_ids,
    torch::Tensor culled_slots,
    torch::Tensor culled_count,
    torch::Tensor turnover_plan_status
) {
    const std::vector<std::pair<const char*, torch::Tensor>> tensors = {
        {"slot_to_agent_id", slot_to_agent_id},
        {"agent_id_to_slot", agent_id_to_slot},
        {"alive_mask", alive_mask},
        {"pos_y", pos_y},
        {"pos_x", pos_x},
        {"agent_grid", agent_grid},
        {"energy", energy},
        {"health", health},
        {"policy_energy_observation", policy_energy_observation},
        {"action_cache_token", action_cache_token},
        {"action_cache_cursor", action_cache_cursor},
        {"action_cache_next", action_cache_next},
        {"action_cache_actions", action_cache_actions},
        {"slot_free_queue", slot_free_queue},
        {"slot_free_queue_count", slot_free_queue_count},
        {"pending_free_agent_ids", pending_free_agent_ids},
        {"pending_free_slots", pending_free_slots},
        {"pending_free_count", pending_free_count},
        {"culled_agent_ids", culled_agent_ids},
        {"culled_slots", culled_slots},
        {"culled_count", culled_count},
        {"turnover_plan_status", turnover_plan_status},
    };
    for (const auto& item : tensors) {
        check_cuda_contiguous(item.second, item.first);
    }
    const auto device = slot_to_agent_id.device();
    for (const auto& item : tensors) {
        TORCH_CHECK(item.second.device() == device, item.first, " must share one CUDA device");
    }
    const int64_t slot_count = slot_to_agent_id.numel();
    TORCH_CHECK(slot_count > 0 && slot_to_agent_id.scalar_type() == torch::kInt64 && slot_to_agent_id.dim() == 1, "slot_to_agent_id must be nonempty int64[slots]");
    TORCH_CHECK(agent_id_to_slot.scalar_type() == torch::kInt32 && agent_id_to_slot.dim() == 1, "agent_id_to_slot must be int32[ids]");
    TORCH_CHECK(alive_mask.scalar_type() == torch::kBool && alive_mask.numel() == slot_count, "alive_mask must be bool[slots]");
    TORCH_CHECK(pos_y.scalar_type() == torch::kInt32 && pos_y.numel() == slot_count, "pos_y must be int32[slots]");
    TORCH_CHECK(pos_x.scalar_type() == torch::kInt32 && pos_x.numel() == slot_count, "pos_x must be int32[slots]");
    TORCH_CHECK(agent_grid.scalar_type() == torch::kInt32 && agent_grid.dim() == 2, "agent_grid must be int32[height,width]");
    TORCH_CHECK(energy.scalar_type() == torch::kFloat32 && energy.numel() == slot_count, "energy must be float32[slots]");
    TORCH_CHECK(health.scalar_type() == torch::kFloat32 && health.numel() == slot_count, "health must be float32[slots]");
    TORCH_CHECK(policy_energy_observation.scalar_type() == torch::kFloat64 && policy_energy_observation.numel() == slot_count, "policy_energy_observation must be float64[slots]");
    TORCH_CHECK(action_cache_token.scalar_type() == torch::kInt64 && action_cache_token.numel() == slot_count, "action_cache_token must be int64[slots]");
    TORCH_CHECK(action_cache_cursor.scalar_type() == torch::kInt32 && action_cache_cursor.numel() == slot_count, "action_cache_cursor must be int32[slots]");
    TORCH_CHECK(action_cache_next.scalar_type() == torch::kInt32 && action_cache_next.numel() == slot_count, "action_cache_next must be int32[slots]");
    TORCH_CHECK(action_cache_actions.scalar_type() == torch::kInt32 && action_cache_actions.dim() == 2 && action_cache_actions.size(0) == slot_count, "action_cache_actions must be int32[slots,width]");
    TORCH_CHECK(slot_free_queue.scalar_type() == torch::kInt64 && slot_free_queue.dim() == 1, "slot_free_queue must be int64[queue]");
    TORCH_CHECK(slot_free_queue_count.scalar_type() == torch::kInt64 && slot_free_queue_count.numel() == 1, "slot_free_queue_count must be int64[1]");
    TORCH_CHECK(pending_free_agent_ids.scalar_type() == torch::kInt64 && pending_free_agent_ids.dim() == 1, "pending_free_agent_ids must be int64[capacity]");
    TORCH_CHECK(pending_free_slots.scalar_type() == torch::kInt64 && pending_free_slots.sizes() == pending_free_agent_ids.sizes(), "pending_free_slots must align with pending ids");
    TORCH_CHECK(pending_free_count.scalar_type() == torch::kInt64 && pending_free_count.numel() == 1, "pending_free_count must be int64[1]");
    TORCH_CHECK(culled_agent_ids.scalar_type() == torch::kInt64 && culled_agent_ids.dim() == 1, "culled_agent_ids must be int64[capacity]");
    TORCH_CHECK(culled_slots.scalar_type() == torch::kInt64 && culled_slots.sizes() == culled_agent_ids.sizes(), "culled_slots must align with culled ids");
    TORCH_CHECK(culled_count.scalar_type() == torch::kInt64 && culled_count.numel() == 1, "culled_count must be int64[1]");
    TORCH_CHECK(turnover_plan_status.scalar_type() == torch::kInt32 && turnover_plan_status.numel() == 1, "turnover_plan_status must be int32[1]");

    c10::cuda::CUDAGuard guard(device);
    auto int64_options = slot_to_agent_id.options().dtype(torch::kInt64);
    auto int32_options = slot_to_agent_id.options().dtype(torch::kInt32);
    const int64_t release_capacity = pending_free_agent_ids.numel() + culled_agent_ids.numel();
    auto released_agent_ids = torch::full({release_capacity}, -1, int64_options);
    auto released_slots = torch::full({release_capacity}, -1, int64_options);
    auto released_count = torch::zeros({1}, int64_options);
    auto pending_released_count = torch::zeros({1}, int64_options);
    auto culled_released_count = torch::zeros({1}, int64_options);
    auto status = torch::zeros({1}, int32_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    turnover_cull_release_kernel<<<1, 1, 0, stream>>>(
        slot_to_agent_id.data_ptr<int64_t>(),
        agent_id_to_slot.data_ptr<int32_t>(),
        alive_mask.data_ptr<bool>(),
        pos_y.data_ptr<int32_t>(),
        pos_x.data_ptr<int32_t>(),
        agent_grid.data_ptr<int32_t>(),
        energy.data_ptr<float>(),
        health.data_ptr<float>(),
        policy_energy_observation.data_ptr<double>(),
        action_cache_token.data_ptr<int64_t>(),
        action_cache_cursor.data_ptr<int32_t>(),
        action_cache_next.data_ptr<int32_t>(),
        action_cache_actions.data_ptr<int32_t>(),
        slot_free_queue.data_ptr<int64_t>(),
        slot_free_queue_count.data_ptr<int64_t>(),
        pending_free_agent_ids.data_ptr<int64_t>(),
        pending_free_slots.data_ptr<int64_t>(),
        pending_free_count.data_ptr<int64_t>(),
        culled_agent_ids.data_ptr<int64_t>(),
        culled_slots.data_ptr<int64_t>(),
        culled_count.data_ptr<int64_t>(),
        turnover_plan_status.data_ptr<int32_t>(),
        released_agent_ids.data_ptr<int64_t>(),
        released_slots.data_ptr<int64_t>(),
        released_count.data_ptr<int64_t>(),
        pending_released_count.data_ptr<int64_t>(),
        culled_released_count.data_ptr<int64_t>(),
        status.data_ptr<int32_t>(),
        slot_count,
        agent_id_to_slot.numel(),
        agent_grid.size(0),
        agent_grid.size(1),
        action_cache_actions.size(1),
        slot_free_queue.numel(),
        pending_free_agent_ids.numel(),
        culled_agent_ids.numel()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        released_agent_ids,
        released_slots,
        released_count,
        pending_released_count,
        culled_released_count,
        slot_free_queue_count,
        status,
    };
}

std::vector<torch::Tensor> eiel_cuda_architecture_b_turnover_refill_plan_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor alive_mask,
    torch::Tensor agent_grid,
    torch::Tensor parent_pool_slots,
    torch::Tensor parent_pool_count,
    torch::Tensor refill_count,
    torch::Tensor turnover_plan_status,
    torch::Tensor python_random_words,
    torch::Tensor python_random_index,
    torch::Tensor python_random_raw_draws,
    torch::Tensor mutation_py_semantic_draws,
    torch::Tensor env_pcg64_state_words,
    torch::Tensor env_pcg64_buffer_words,
    torch::Tensor policy_np_semantic_draws,
    const bool env_policy_alias
) {
    const std::vector<std::pair<const char*, torch::Tensor>> tensors = {
        {"slot_to_agent_id", slot_to_agent_id},
        {"alive_mask", alive_mask},
        {"agent_grid", agent_grid},
        {"parent_pool_slots", parent_pool_slots},
        {"parent_pool_count", parent_pool_count},
        {"refill_count", refill_count},
        {"turnover_plan_status", turnover_plan_status},
        {"python_random_words", python_random_words},
        {"python_random_index", python_random_index},
        {"python_random_raw_draws", python_random_raw_draws},
        {"mutation_py_semantic_draws", mutation_py_semantic_draws},
        {"env_pcg64_state_words", env_pcg64_state_words},
        {"env_pcg64_buffer_words", env_pcg64_buffer_words},
        {"policy_np_semantic_draws", policy_np_semantic_draws},
    };
    for (const auto& item : tensors) {
        check_cuda_contiguous(item.second, item.first);
    }
    const auto device = slot_to_agent_id.device();
    for (const auto& item : tensors) {
        TORCH_CHECK(item.second.device() == device, item.first, " must share one CUDA device");
    }
    const int64_t slot_count = slot_to_agent_id.numel();
    TORCH_CHECK(slot_count > 0 && slot_to_agent_id.scalar_type() == torch::kInt64 && slot_to_agent_id.dim() == 1, "slot_to_agent_id must be nonempty int64[slots]");
    TORCH_CHECK(alive_mask.scalar_type() == torch::kBool && alive_mask.sizes() == slot_to_agent_id.sizes(), "alive_mask must be bool[slots]");
    TORCH_CHECK(agent_grid.scalar_type() == torch::kInt32 && agent_grid.dim() == 2 && agent_grid.numel() > 0, "agent_grid must be nonempty int32[height,width]");
    TORCH_CHECK(parent_pool_slots.scalar_type() == torch::kInt64 && parent_pool_slots.dim() == 1 && parent_pool_slots.numel() == slot_count, "parent_pool_slots must be int64[slots]");
    TORCH_CHECK(parent_pool_count.scalar_type() == torch::kInt64 && parent_pool_count.numel() == 1, "parent_pool_count must be int64[1]");
    TORCH_CHECK(refill_count.scalar_type() == torch::kInt64 && refill_count.numel() == 1, "refill_count must be int64[1]");
    TORCH_CHECK(turnover_plan_status.scalar_type() == torch::kInt32 && turnover_plan_status.numel() == 1, "turnover_plan_status must be int32[1]");
    TORCH_CHECK(python_random_words.scalar_type() == torch::kUInt32 && python_random_words.dim() == 1 && python_random_words.numel() == 624, "python_random_words must be uint32[624]");
    TORCH_CHECK(python_random_index.scalar_type() == torch::kInt64 && python_random_index.numel() == 1, "python_random_index must be int64[1]");
    TORCH_CHECK(python_random_raw_draws.scalar_type() == torch::kInt64 && python_random_raw_draws.numel() == 1, "python_random_raw_draws must be int64[1]");
    TORCH_CHECK(mutation_py_semantic_draws.scalar_type() == torch::kInt64 && mutation_py_semantic_draws.numel() == 1, "mutation_py_semantic_draws must be int64[1]");
    TORCH_CHECK(env_pcg64_state_words.scalar_type() == torch::kUInt64 && env_pcg64_state_words.dim() == 1 && env_pcg64_state_words.numel() == 4, "env_pcg64_state_words must be uint64[4]");
    TORCH_CHECK(env_pcg64_buffer_words.scalar_type() == torch::kInt64 && env_pcg64_buffer_words.dim() == 1 && env_pcg64_buffer_words.numel() == 2, "env_pcg64_buffer_words must be int64[2]");
    TORCH_CHECK(policy_np_semantic_draws.scalar_type() == torch::kInt64 && policy_np_semantic_draws.numel() == 1, "policy_np_semantic_draws must be int64[1]");

    c10::cuda::CUDAGuard guard(device);
    const auto int64_options = slot_to_agent_id.options().dtype(torch::kInt64);
    const auto int32_options = slot_to_agent_id.options().dtype(torch::kInt32);
    auto planned_parent_slots = torch::full({slot_count}, -1, int64_options);
    auto planned_mate_slots = torch::full({slot_count}, -1, int64_options);
    auto planned_spawn_positions = torch::full({slot_count, 2}, -1, int32_options);
    auto spawn_attempts_by_birth = torch::zeros({slot_count}, int64_options);
    auto planned_count = torch::zeros({1}, int64_options);
    auto python_raw_draws_added = torch::zeros({1}, int64_options);
    auto python_semantic_draws_added = torch::zeros({1}, int64_options);
    auto spawn_raw_draws_added = torch::zeros({1}, int64_options);
    auto spawn_semantic_draws_added = torch::zeros({1}, int64_options);
    auto status = torch::zeros({1}, int32_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    turnover_refill_plan_kernel<<<1, 1, 0, stream>>>(
        slot_to_agent_id.data_ptr<int64_t>(),
        alive_mask.data_ptr<bool>(),
        agent_grid.data_ptr<int32_t>(),
        parent_pool_slots.data_ptr<int64_t>(),
        parent_pool_count.data_ptr<int64_t>(),
        refill_count.data_ptr<int64_t>(),
        turnover_plan_status.data_ptr<int32_t>(),
        python_random_words.data_ptr<uint32_t>(),
        python_random_index.data_ptr<int64_t>(),
        python_random_raw_draws.data_ptr<int64_t>(),
        mutation_py_semantic_draws.data_ptr<int64_t>(),
        env_pcg64_state_words.data_ptr<uint64_t>(),
        env_pcg64_buffer_words.data_ptr<int64_t>(),
        policy_np_semantic_draws.data_ptr<int64_t>(),
        planned_parent_slots.data_ptr<int64_t>(),
        planned_mate_slots.data_ptr<int64_t>(),
        planned_spawn_positions.data_ptr<int32_t>(),
        spawn_attempts_by_birth.data_ptr<int64_t>(),
        planned_count.data_ptr<int64_t>(),
        python_raw_draws_added.data_ptr<int64_t>(),
        python_semantic_draws_added.data_ptr<int64_t>(),
        spawn_raw_draws_added.data_ptr<int64_t>(),
        spawn_semantic_draws_added.data_ptr<int64_t>(),
        status.data_ptr<int32_t>(),
        slot_count,
        agent_grid.size(0),
        agent_grid.size(1),
        env_policy_alias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        planned_parent_slots,
        planned_mate_slots,
        planned_spawn_positions,
        spawn_attempts_by_birth,
        planned_count,
        python_raw_draws_added,
        python_semantic_draws_added,
        spawn_raw_draws_added,
        spawn_semantic_draws_added,
        status,
    };
}
