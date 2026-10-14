#ifndef EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_TURNOVER_PLAN_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_TURNOVER_PLAN_CUH_

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "native/cuda_architecture_b_pcg64.cuh"

namespace eiel::cuda_architecture_b::persistent_turnover_plan {

constexpr int64_t kCpythonStateWordCount = 624;
constexpr int64_t kCpythonTwistOffset = 397;
constexpr int64_t kMaximumParentPairAttempts = 1000000;
constexpr int64_t kMaximumSpawnAttempts = 200;
// Spawn coordinates and the grid index are represented by signed int32 rows.
// Keeping both dimensions within this bound also keeps y * width in int64.
constexpr int64_t kMaximumGridDimension = 0x7fffffffLL;

enum GenerationTurnoverPlanStatus : int32_t {
    kTurnoverStatusOk = 0,
    kTurnoverStatusInvalidConfig = 1,
    kTurnoverStatusInvalidResidentState = 2,
    kTurnoverStatusNonuniformEarlyLifeWeight = 3,
    kTurnoverStatusLowPopulationUnsupported = 4,
    kTurnoverStatusPostCullLowPopulationUnsupported = 5,
    kTurnoverStatusInsufficientParentPool = 6,
};

enum GenerationTurnoverRefillPlanStatus : int32_t {
    kRefillStatusOk = 0,
    kRefillStatusTurnoverFailed = 1,
    kRefillStatusInvalidState = 2,
    kRefillStatusInvalidParentPool = 3,
    kRefillStatusNoSpawnSpace = 4,
};

// Every pointer is already offset to the simulation cell owned by the caller.
// Each input has exactly config.slot_count rows and remains read-only.
struct FlatGenerationTurnoverState {
    const int64_t* slot_to_agent_id;
    const bool* alive_mask;
    const int32_t* survival_steps;
    const double* energy_collected_exact;
    const int32_t* offspring_count;
    const int32_t* attacks;
    const int32_t* early_survival_steps;
    const double* early_energy_collected_exact;
    const float* early_injury_gained;
    const float* early_life_weight;
    const int32_t* action_switches;
    const int32_t* action_oscillations;
    const float* drive_volatility;
    const int32_t* selfharm_events;
    const float* pred_err_sum;
    const int32_t* pred_err_count;
};

// Ranked, culled, and parent-pool arrays each have config.slot_count rows.
// Every count, hardcap, and status pointer has exactly one row. Storage is
// fixed-capacity caller-owned device memory and is initialized by the helper.
struct FlatGenerationTurnoverBuffers {
    int64_t* ranked_slots;
    int64_t* ranked_agent_ids;
    double* ranked_fitness;
    int64_t* ranked_count;
    int64_t* culled_slots;
    int64_t* culled_agent_ids;
    int64_t* culled_count;
    int64_t* parent_pool_slots;
    int64_t* parent_pool_agent_ids;
    int64_t* parent_pool_count;
    int64_t* survivor_count;
    int64_t* refill_count;
    bool* hardcap;
    int32_t* status;
};

struct GenerationTurnoverConfig {
    int64_t slot_count;
    int32_t lifecycle_mode;
    int64_t population_size;
    double cull_fraction;
    double parent_fraction;
    double coherence_thrashing_weight;
    double coherence_drive_volatility_weight;
    double coherence_selfharm_weight;
    double coherence_pred_err_weight;
    bool terminal_low_population_no_reinit;
};

enum GenerationTurnoverSelectionModeV1 : int32_t {
    kTurnoverSelectionFitnessV1 = 0,
    kTurnoverSelectionUniformNeutralV1 = 1,
};

// Versioned, cell-local selection control. The neutral mode ranks the
// authoritative living set from identity-independent pseudorandom keys. It
// does not inspect genotype, lineage, policy, strategy, or fitness fields and
// does not advance any resident RNG cursor.
struct GenerationTurnoverSelectionConfigV1 {
    int32_t version;
    int32_t mode;
    uint64_t cell_seed;
    int64_t generation;
};

// The first seven pointers are read-only outputs/state from turnover and the
// resident grid. The remaining pointers are mutable RNG cursors and scratch.
// CPython state and caller-owned trial scratch each have 624 uint32 words;
// index/raw-draw cursors have one row. PCG64 state has four uint64 words plus
// two int64 buffer words. Every counter has one row.
struct FlatGenerationTurnoverRefillState {
    const int64_t* slot_to_agent_id;
    const bool* alive_mask;
    const int32_t* agent_grid;
    const int64_t* parent_pool_slots;
    const int64_t* parent_pool_count;
    const int64_t* refill_count;
    const int32_t* turnover_status;
    uint32_t* python_random_words;
    uint32_t* python_random_trial_words;
    int64_t* python_random_index;
    int64_t* python_random_raw_draws;
    int64_t* mutation_py_semantic_draws;
    uint64_t* env_pcg64_state_words;
    int64_t* env_pcg64_buffer_words;
    int64_t* policy_np_semantic_draws;
};

// Plan arrays have config.slot_count rows; spawn positions are interleaved
// [y, x] pairs with 2 * config.slot_count int32 values. Scalar outputs have
// one row. A successful plan does not commit child identity, genome, policy,
// slot ownership, or grid state.
struct FlatGenerationTurnoverRefillBuffers {
    int64_t* planned_parent_slots;
    int64_t* planned_mate_slots;
    int32_t* planned_spawn_positions;
    int64_t* spawn_attempts_by_birth;
    int64_t* planned_count;
    int64_t* python_raw_draws_added;
    int64_t* python_semantic_draws_added;
    int64_t* spawn_raw_draws_added;
    int64_t* spawn_semantic_draws_added;
    int32_t* status;
};

struct GenerationTurnoverRefillConfig {
    int64_t slot_count;
    int64_t grid_height;
    int64_t grid_width;
    bool env_policy_alias;
};

__device__ __forceinline__ double canonical_quiet_nan() {
    return __longlong_as_double(
        static_cast<long long>(0x7ff8000000000000ULL)
    );
}

__device__ __forceinline__ double maximum_one(const double value) {
    return value > 1.0 ? value : 1.0;
}

__device__ __forceinline__ double effective_fitness_exact(
    const FlatGenerationTurnoverState state,
    const GenerationTurnoverConfig config,
    const int64_t slot
) {
    const double survival = static_cast<double>(state.survival_steps[slot]);
    const double offspring = static_cast<double>(state.offspring_count[slot]);
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
    score += (
        state.energy_collected_exact[slot] / maximum_one(survival)
    ) * 80.0;
    score -= static_cast<double>(state.attacks[slot]) * 2.5;

    const double early_survival = static_cast<double>(
        state.early_survival_steps[slot]
    );
    const double early_component = 0.10 * early_survival
        + 80.0 * (
            state.early_energy_collected_exact[slot]
            / maximum_one(early_survival)
        )
        - 10.0 * static_cast<double>(state.early_injury_gained[slot]);
    score += early_component;
    score += 10.0;

    const double switches = static_cast<double>(state.action_switches[slot]);
    const double oscillations = static_cast<double>(
        state.action_oscillations[slot]
    );
    const double thrash = 0.5 * switches + oscillations;
    const double pred_mean = static_cast<double>(state.pred_err_sum[slot])
        / maximum_one(static_cast<double>(state.pred_err_count[slot]));
    return score
        - config.coherence_thrashing_weight * thrash
        - config.coherence_drive_volatility_weight
            * static_cast<double>(state.drive_volatility[slot])
        - config.coherence_selfharm_weight
            * static_cast<double>(state.selfharm_events[slot])
        - config.coherence_pred_err_weight * pred_mean;
}

__device__ __forceinline__ uint64_t neutral_selection_mix_v1(
    uint64_t value
) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

__device__ __forceinline__ double neutral_selection_rank_v1(
    const GenerationTurnoverSelectionConfigV1 selection,
    const int64_t agent_id
) {
    uint64_t key = selection.cell_seed;
    key ^= neutral_selection_mix_v1(
        static_cast<uint64_t>(selection.generation)
        ^ 0xd1b54a32d192ed03ULL
    );
    key ^= neutral_selection_mix_v1(
        static_cast<uint64_t>(agent_id) ^ 0x8cb92baa7f3d8dd7ULL
    );
    // Every integer through 2^53 is represented exactly as a double. Exact
    // key ties retain the existing ascending-agent-id deterministic tiebreak.
    return static_cast<double>(neutral_selection_mix_v1(key) >> 11U);
}

__device__ __forceinline__ void initialize_turnover_outputs(
    const FlatGenerationTurnoverBuffers buffers,
    const int64_t slot_count
) {
    for (int64_t row = 0; row < slot_count; ++row) {
        buffers.ranked_slots[row] = -1;
        buffers.ranked_agent_ids[row] = -1;
        buffers.ranked_fitness[row] = canonical_quiet_nan();
        buffers.culled_slots[row] = -1;
        buffers.culled_agent_ids[row] = -1;
        buffers.parent_pool_slots[row] = -1;
        buffers.parent_pool_agent_ids[row] = -1;
    }
    buffers.ranked_count[0] = 0;
    buffers.culled_count[0] = 0;
    buffers.parent_pool_count[0] = 0;
    buffers.survivor_count[0] = 0;
    buffers.refill_count[0] = 0;
    buffers.hardcap[0] = false;
    buffers.status[0] = kTurnoverStatusOk;
}

// Exact allocation-free serial extraction of Stage2
// native/cuda_architecture_b_turnover.cu:310-362 and 364-531. One device
// thread calls this helper for one cell. Ranking is fitness-descending; exact
// fitness ties use ascending agent_id, matching the Stage2 insertion order.
// The result is a plan only: it does not release culled rows or refill state.
__device__ __forceinline__ void plan_generation_turnover_serial(
    const FlatGenerationTurnoverState state,
    const FlatGenerationTurnoverBuffers buffers,
    const GenerationTurnoverConfig config,
    const GenerationTurnoverSelectionConfigV1 selection
) {
    initialize_turnover_outputs(buffers, config.slot_count);

    if (
        config.slot_count <= 0
        || config.population_size <= 0
        || config.population_size > config.slot_count
        || (config.lifecycle_mode != 0 && config.lifecycle_mode != 1)
        || !::isfinite(config.cull_fraction)
        || config.cull_fraction < 0.0
        || config.cull_fraction > 1.0
        || !::isfinite(config.parent_fraction)
        || config.parent_fraction < 0.0
        || config.parent_fraction > 1.0
        || !::isfinite(config.coherence_thrashing_weight)
        || !::isfinite(config.coherence_drive_volatility_weight)
        || !::isfinite(config.coherence_selfharm_weight)
        || !::isfinite(config.coherence_pred_err_weight)
        || selection.version != 1
        || (
            selection.mode != kTurnoverSelectionFitnessV1
            && selection.mode != kTurnoverSelectionUniformNeutralV1
        )
    ) {
        buffers.status[0] = kTurnoverStatusInvalidConfig;
        return;
    }

    int64_t live_count = 0;
    for (int64_t slot = 0; slot < config.slot_count; ++slot) {
        if (!state.alive_mask[slot]) {
            if (state.slot_to_agent_id[slot] >= 0) {
                buffers.status[0] = kTurnoverStatusInvalidResidentState;
                return;
            }
            continue;
        }
        if (state.slot_to_agent_id[slot] < 0) {
            buffers.status[0] = kTurnoverStatusInvalidResidentState;
            return;
        }
        double fitness = 0.0;
        if (selection.mode == kTurnoverSelectionFitnessV1) {
            if (!::isfinite(state.early_life_weight[slot])) {
                buffers.status[0] = kTurnoverStatusInvalidResidentState;
                return;
            }
            if (state.early_life_weight[slot] != 1.0f) {
                buffers.status[0] =
                    kTurnoverStatusNonuniformEarlyLifeWeight;
                return;
            }
            fitness = effective_fitness_exact(state, config, slot);
        } else {
            fitness = neutral_selection_rank_v1(
                selection,
                state.slot_to_agent_id[slot]
            );
        }
        if (!::isfinite(fitness)) {
            buffers.status[0] = kTurnoverStatusInvalidResidentState;
            return;
        }

        int64_t insert_at = live_count;
        while (insert_at > 0) {
            const double previous_fitness =
                buffers.ranked_fitness[insert_at - 1];
            const int64_t previous_agent_id =
                buffers.ranked_agent_ids[insert_at - 1];
            if (
                previous_fitness > fitness
                || (
                    previous_fitness == fitness
                    && previous_agent_id < state.slot_to_agent_id[slot]
                )
            ) {
                break;
            }
            buffers.ranked_fitness[insert_at] = previous_fitness;
            buffers.ranked_agent_ids[insert_at] = previous_agent_id;
            buffers.ranked_slots[insert_at] =
                buffers.ranked_slots[insert_at - 1];
            --insert_at;
        }
        buffers.ranked_fitness[insert_at] = fitness;
        buffers.ranked_agent_ids[insert_at] = state.slot_to_agent_id[slot];
        buffers.ranked_slots[insert_at] = slot;
        ++live_count;
    }
    buffers.ranked_count[0] = live_count;
    if (live_count == 0) {
        // Extinction is an authoritative terminal state. Do not synthesize
        // founders or consume refill RNG when no living parent remains.
        return;
    }
    if (live_count < 2 && !config.terminal_low_population_no_reinit) {
        buffers.status[0] = kTurnoverStatusLowPopulationUnsupported;
        return;
    }
    if (live_count < 2) {
        buffers.survivor_count[0] = live_count;
        return;
    }

    const bool hardcap_value = live_count > config.population_size;
    buffers.hardcap[0] = hardcap_value;
    int64_t survivors = config.population_size;
    int64_t cull_rows = live_count - config.population_size;
    int64_t refill_rows = 0;
    int64_t selected_parents = 0;
    const bool hybrid_overlapping =
        config.lifecycle_mode == 1;
    if (hybrid_overlapping && !hardcap_value) {
        survivors = live_count;
        cull_rows = 0;
        refill_rows = 0;
        selected_parents = 0;
    } else if (!hardcap_value) {
        cull_rows = static_cast<int64_t>(
            static_cast<double>(live_count) * config.cull_fraction
        );
        survivors = live_count - cull_rows;
        if (survivors < 2 && !config.terminal_low_population_no_reinit) {
            buffers.status[0] =
                kTurnoverStatusPostCullLowPopulationUnsupported;
            return;
        }
        refill_rows = survivors < 2 ? 0 : config.population_size - survivors;
        if (refill_rows < 0 || config.population_size > config.slot_count) {
            buffers.status[0] = kTurnoverStatusInvalidConfig;
            return;
        }
        if (refill_rows > 0) {
            const int64_t fraction_parents = static_cast<int64_t>(
                static_cast<double>(survivors) * config.parent_fraction
            );
            const int64_t requested_parents = fraction_parents > 2
                ? fraction_parents
                : 2;
            selected_parents = requested_parents < survivors
                ? requested_parents
                : survivors;
            if (selected_parents < 2) {
                buffers.status[0] =
                    kTurnoverStatusInsufficientParentPool;
                return;
            }
        }
    }

    const int64_t cull_start = live_count - cull_rows;
    for (int64_t index = 0; index < cull_rows; ++index) {
        buffers.culled_slots[index] =
            buffers.ranked_slots[cull_start + index];
        buffers.culled_agent_ids[index] =
            buffers.ranked_agent_ids[cull_start + index];
    }
    for (int64_t index = 0; index < selected_parents; ++index) {
        buffers.parent_pool_slots[index] = buffers.ranked_slots[index];
        buffers.parent_pool_agent_ids[index] =
            buffers.ranked_agent_ids[index];
    }
    buffers.culled_count[0] = cull_rows;
    buffers.parent_pool_count[0] = selected_parents;
    buffers.survivor_count[0] = survivors;
    buffers.refill_count[0] = refill_rows;
    buffers.status[0] = kTurnoverStatusOk;
}

// Source-compatible legacy entry point. This wrapper preserves the original
// fitness-ranked behavior byte-for-byte for callers that do not opt in.
__device__ __forceinline__ void plan_generation_turnover_serial(
    const FlatGenerationTurnoverState state,
    const FlatGenerationTurnoverBuffers buffers,
    const GenerationTurnoverConfig config
) {
    plan_generation_turnover_serial(
        state,
        buffers,
        config,
        GenerationTurnoverSelectionConfigV1{
            1,
            kTurnoverSelectionFitnessV1,
            0ULL,
            0,
        }
    );
}

struct DeviceCpythonChoiceRng {
    uint32_t* words;
    int64_t index;
    int64_t raw_draws;

    __device__ __forceinline__ void twist() {
        constexpr uint32_t matrix_a = 0x9908b0dfU;
        constexpr uint32_t upper_mask = 0x80000000U;
        constexpr uint32_t lower_mask = 0x7fffffffU;
        for (
            int64_t row = 0;
            row < kCpythonStateWordCount - kCpythonTwistOffset;
            ++row
        ) {
            const uint32_t y = (words[row] & upper_mask)
                | (words[row + 1] & lower_mask);
            words[row] = words[row + kCpythonTwistOffset] ^ (y >> 1U)
                ^ ((y & 1U) ? matrix_a : 0U);
        }
        for (
            int64_t row = kCpythonStateWordCount - kCpythonTwistOffset;
            row < kCpythonStateWordCount - 1;
            ++row
        ) {
            const uint32_t y = (words[row] & upper_mask)
                | (words[row + 1] & lower_mask);
            words[row] = words[
                row + (kCpythonTwistOffset - kCpythonStateWordCount)
            ] ^ (y >> 1U) ^ ((y & 1U) ? matrix_a : 0U);
        }
        const uint32_t y = (words[kCpythonStateWordCount - 1] & upper_mask)
            | (words[0] & lower_mask);
        words[kCpythonStateWordCount - 1] =
            words[kCpythonTwistOffset - 1] ^ (y >> 1U)
            ^ ((y & 1U) ? matrix_a : 0U);
        index = 0;
    }

    __device__ __forceinline__ uint32_t raw32() {
        if (index >= kCpythonStateWordCount) {
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

    __device__ __forceinline__ uint64_t getrandbits(
        const int64_t bit_count
    ) {
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

__device__ __forceinline__ int64_t pcg64_numpy_int64_integer_exact(
    Pcg64State& rng,
    const int64_t high_exclusive,
    int64_t* const raw32_draws = nullptr
) {
    if (high_exclusive == 1) {
        return 0;
    }
    const uint32_t range = static_cast<uint32_t>(high_exclusive);
    const uint32_t threshold = static_cast<uint32_t>((0U - range) % range);
    int64_t raw32_count = 0;
    while (true) {
        const uint32_t raw = pcg64_raw32(rng);
        ++raw32_count;
        const uint64_t product = static_cast<uint64_t>(raw)
            * static_cast<uint64_t>(range);
        if (static_cast<uint32_t>(product) >= threshold) {
            if (raw32_draws != nullptr) {
                *raw32_draws += raw32_count;
            }
            return static_cast<int64_t>(product >> 32U);
        }
    }
}

__device__ __forceinline__ void initialize_refill_outputs(
    const FlatGenerationTurnoverRefillBuffers buffers,
    const int64_t slot_count
) {
    for (int64_t row = 0; row < slot_count; ++row) {
        buffers.planned_parent_slots[row] = -1;
        buffers.planned_mate_slots[row] = -1;
        buffers.planned_spawn_positions[row * 2] = -1;
        buffers.planned_spawn_positions[row * 2 + 1] = -1;
        buffers.spawn_attempts_by_birth[row] = 0;
    }
    buffers.planned_count[0] = 0;
    buffers.python_raw_draws_added[0] = 0;
    buffers.python_semantic_draws_added[0] = 0;
    buffers.spawn_raw_draws_added[0] = 0;
    buffers.spawn_semantic_draws_added[0] = 0;
    buffers.status[0] = kRefillStatusOk;
}

// Exact allocation-free serial extraction of Stage2
// native/cuda_architecture_b_turnover.cu:39-128 and 130-300. For each birth,
// the helper completes CPython parent/mate selection before consuming NumPy
// PCG64 y/x spawn draws. RNG state and semantic counters commit only after the
// complete plan succeeds; plan rows may be partial on a later failure while
// planned_count stays zero, matching the Stage2 wrapper contract.
__device__ __forceinline__ void plan_generation_turnover_refill_serial(
    const FlatGenerationTurnoverRefillState state,
    const FlatGenerationTurnoverRefillBuffers buffers,
    const GenerationTurnoverRefillConfig config
) {
    initialize_refill_outputs(buffers, config.slot_count);

    if (state.turnover_status[0] != kTurnoverStatusOk) {
        buffers.status[0] = kRefillStatusTurnoverFailed;
        return;
    }
    const int64_t pool_count = state.parent_pool_count[0];
    const int64_t births = state.refill_count[0];
    if (
        births < 0
        || births > config.slot_count
        || config.slot_count <= 0
        || pool_count < (births > 0 ? 2 : 0)
        || pool_count > config.slot_count
        || state.python_random_index[0] < 0
        || state.python_random_index[0] > kCpythonStateWordCount
        || state.python_random_raw_draws[0] < 0
        || state.mutation_py_semantic_draws[0] < 0
        || state.policy_np_semantic_draws[0] < 0
        || config.grid_height <= 0
        || config.grid_width <= 0
        || config.grid_height > kMaximumGridDimension
        || config.grid_width > kMaximumGridDimension
        || state.env_pcg64_buffer_words[0] < 0
        || state.env_pcg64_buffer_words[0] > 1
        || state.env_pcg64_buffer_words[1] < 0
        || static_cast<uint64_t>(state.env_pcg64_buffer_words[1])
            > 0xffffffffULL
        || (state.env_pcg64_state_words[2] & 1ULL) == 0ULL
        || (births > 0 && state.python_random_trial_words == nullptr)
    ) {
        buffers.status[0] = kRefillStatusInvalidState;
        return;
    }
    if (births == 0) {
        buffers.planned_count[0] = 0;
        buffers.status[0] = kRefillStatusOk;
        return;
    }
    for (int64_t row = 0; row < pool_count; ++row) {
        const int64_t slot = state.parent_pool_slots[row];
        if (
            slot < 0
            || slot >= config.slot_count
            || !state.alive_mask[slot]
            || state.slot_to_agent_id[slot] < 0
        ) {
            buffers.status[0] = kRefillStatusInvalidParentPool;
            return;
        }
        for (int64_t previous = 0; previous < row; ++previous) {
            if (state.parent_pool_slots[previous] == slot) {
                buffers.status[0] = kRefillStatusInvalidParentPool;
                return;
            }
        }
    }

    DeviceCpythonChoiceRng py_rng{
        state.python_random_trial_words,
        state.python_random_index[0],
        state.python_random_raw_draws[0],
    };
    for (int64_t row = 0; row < kCpythonStateWordCount; ++row) {
        py_rng.words[row] = state.python_random_words[row];
    }
    const int64_t initial_python_raw_draws = py_rng.raw_draws;
    int64_t python_semantic_added = 0;

    Pcg64State env_rng{
        state.env_pcg64_state_words[0],
        state.env_pcg64_state_words[1],
        state.env_pcg64_state_words[2],
        state.env_pcg64_state_words[3],
        static_cast<uint32_t>(state.env_pcg64_buffer_words[0]),
        static_cast<uint32_t>(state.env_pcg64_buffer_words[1]),
    };
    int64_t spawn_draws = 0;
    int64_t spawn_semantic = 0;

    for (int64_t birth = 0; birth < births; ++birth) {
        int64_t parent_slot = -1;
        int64_t mate_slot = -1;
        int64_t pair_attempts = 0;
        do {
            parent_slot = state.parent_pool_slots[py_rng.randbelow(
                static_cast<uint64_t>(pool_count)
            )];
            mate_slot = state.parent_pool_slots[py_rng.randbelow(
                static_cast<uint64_t>(pool_count)
            )];
            python_semantic_added += 2;
            ++pair_attempts;
            if (pair_attempts > kMaximumParentPairAttempts) {
                buffers.status[0] = kRefillStatusInvalidParentPool;
                return;
            }
        } while (mate_slot == parent_slot);

        int32_t selected_y = -1;
        int32_t selected_x = -1;
        int64_t attempts = 0;
        for (; attempts < kMaximumSpawnAttempts; ++attempts) {
            int64_t y_raw32_draws = 0;
            int64_t x_raw32_draws = 0;
            const int32_t y = static_cast<int32_t>(
                pcg64_numpy_int64_integer_exact(
                    env_rng,
                    config.grid_height,
                    &y_raw32_draws
                )
            );
            const int32_t x = static_cast<int32_t>(
                pcg64_numpy_int64_integer_exact(
                    env_rng,
                    config.grid_width,
                    &x_raw32_draws
                )
            );
            spawn_draws += y_raw32_draws + x_raw32_draws;
            spawn_semantic += 2;
            if (state.agent_grid[y * config.grid_width + x] != -1) {
                continue;
            }
            bool reserved = false;
            for (int64_t previous = 0; previous < birth; ++previous) {
                if (
                    buffers.planned_spawn_positions[previous * 2] == y
                    && buffers.planned_spawn_positions[previous * 2 + 1] == x
                ) {
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
            buffers.status[0] = kRefillStatusNoSpawnSpace;
            return;
        }
        buffers.planned_parent_slots[birth] = parent_slot;
        buffers.planned_mate_slots[birth] = mate_slot;
        buffers.planned_spawn_positions[birth * 2] = selected_y;
        buffers.planned_spawn_positions[birth * 2 + 1] = selected_x;
        buffers.spawn_attempts_by_birth[birth] = attempts + 1;
    }

    for (int64_t row = 0; row < kCpythonStateWordCount; ++row) {
        state.python_random_words[row] = py_rng.words[row];
    }
    state.python_random_index[0] = py_rng.index;
    state.python_random_raw_draws[0] = py_rng.raw_draws;
    state.mutation_py_semantic_draws[0] += python_semantic_added;
    state.env_pcg64_state_words[0] = env_rng.state_lo;
    state.env_pcg64_state_words[1] = env_rng.state_hi;
    state.env_pcg64_state_words[2] = env_rng.inc_lo;
    state.env_pcg64_state_words[3] = env_rng.inc_hi;
    state.env_pcg64_buffer_words[0] =
        static_cast<int64_t>(env_rng.has_uint32);
    state.env_pcg64_buffer_words[1] =
        static_cast<int64_t>(env_rng.uinteger);
    if (config.env_policy_alias) {
        state.policy_np_semantic_draws[0] += spawn_semantic;
    }
    buffers.planned_count[0] = births;
    buffers.python_raw_draws_added[0] =
        py_rng.raw_draws - initial_python_raw_draws;
    buffers.python_semantic_draws_added[0] = python_semantic_added;
    buffers.spawn_raw_draws_added[0] = spawn_draws;
    buffers.spawn_semantic_draws_added[0] = spawn_semantic;
    buffers.status[0] = kRefillStatusOk;
}

}  // namespace eiel::cuda_architecture_b::persistent_turnover_plan

#endif  // EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_TURNOVER_PLAN_CUH_
