#ifndef EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_BIRTH_SCALAR_GENOME_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_BIRTH_SCALAR_GENOME_CUH_

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "cuda_architecture_b_birth_parent_order_v1.cuh"

namespace eiel::cuda_architecture_b::persistent_birth_scalar_genome {

namespace parent_order =
    eiel::cuda_architecture_b::birth_parent_order_v1;

constexpr int64_t kCpythonRandomWordCount = 624;

enum OrdinaryScalarGenomeStatus : int32_t {
    kStatusOk = 0,
    kStatusInvalidScalarBirthPlan = 20,
    kStatusInvalidPythonRandomState = 21,
    kStatusFreshGaussUnsupported = 22,
};

// Every pointer is already offset to the simulation cell owned by the caller.
// The Gaussian tapes are indexed by raw MT19937 draw count, exactly as in the
// immutable Stage2 generation-boundary tape contract.
struct FlatCpythonRandomState {
    uint32_t* python_random_words;
    int64_t* python_random_index;
    bool* python_random_has_gauss;
    double* python_random_gauss_next;
    const double* python_random_gauss_primary_tape;
    const double* python_random_gauss_secondary_tape;
    int64_t* python_random_raw_draws;
    int64_t* mutation_py_semantic_draws;
};

struct FlatOrdinaryScalarGenomePlan {
    const int64_t* parent_slots;
    const int64_t* child_slots;
    const int64_t* planned_count;
};

// Fields preserve the immutable Stage2 int32, policy, and float32 argument
// order. Ordinary births are asexual: the parent is also the mate input to the
// scalar recombination arithmetic, and no turnover pair selection is present.
struct FlatFixedMlpScalarGenomeState {
    const int64_t* slot_to_agent_id;
    const bool* alive_mask;

    int32_t* genome_hidden_size;
    int32_t* genome_memory_dim;
    int32_t* genome_sensor_channel_mask;
    int32_t* genome_sensor_radius;
    int32_t* genome_introspection_channel_mask;
    int32_t* genome_self_regulation_control_mask;
    int32_t* genome_self_modification_target_mask;
    int32_t* genome_directed_inheritance_channel_mask;
    int64_t* policy_mem_dim;
    bool* policy_use_gru;

    float* genome_exploration_bias;
    float* genome_aggression_bias;
    float* genome_mutation_rate;
    float* genome_reproduction_threshold;
    float* genome_action_temperature;
    float* genome_drive_inertia;
    float* genome_drive_noise;
    float* genome_drive_logit_gain;
    float* genome_pred_lr;
    float* genome_drive_sp_hunger;
    float* genome_drive_sp_pain;
    float* genome_drive_sp_fatigue;
    float* genome_drive_sp_fear;
    float* genome_exp_prior_strength;
    float* genome_exp_prior_noise;
    float* genome_drive_birth_noise;
    float* repro_threshold;
    float* gru_upkeep_cost;
    float* policy_attack_bias;
    float* policy_action_temperature;
    float* policy_drive_logit_gain;
    float* drive_inertia;
    float* drive_noise;
    float* pred_lr;
};

struct FlatOrdinaryScalarGenomeOutputs {
    int64_t* committed_count;
    int64_t* semantic_draws_added;
    int32_t* status;
};

struct OrdinaryScalarGenomeConfig {
    int64_t plan_capacity;
    int64_t slot_count;
    int64_t gauss_tape_capacity;
    int32_t sensor_radius_max;
    parent_order::BirthParentOrderModeV1 parent_order_mode{
        parent_order::BirthParentOrderModeV1::ascending_agent_id_v1
    };
};

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
        constexpr uint32_t kMatrixA = 0x9908b0dfU;
        constexpr uint32_t kUpperMask = 0x80000000U;
        constexpr uint32_t kLowerMask = 0x7fffffffU;
        for (int64_t kk = 0; kk < 624 - 397; ++kk) {
            const uint32_t y =
                (state[kk] & kUpperMask) | (state[kk + 1] & kLowerMask);
            state[kk] = state[kk + 397]
                ^ (y >> 1U)
                ^ ((y & 1U) ? kMatrixA : 0U);
        }
        for (int64_t kk = 624 - 397; kk < 623; ++kk) {
            const uint32_t y =
                (state[kk] & kUpperMask) | (state[kk + 1] & kLowerMask);
            state[kk] = state[kk + (397 - 624)]
                ^ (y >> 1U)
                ^ ((y & 1U) ? kMatrixA : 0U);
        }
        const uint32_t y =
            (state[623] & kUpperMask) | (state[0] & kLowerMask);
        state[623] = state[396]
            ^ (y >> 1U)
            ^ ((y & 1U) ? kMatrixA : 0U);
        index = 0;
    }

    __device__ __forceinline__ uint32_t raw32() {
        if (index >= kCpythonRandomWordCount) {
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
        return (
            static_cast<double>(a) * 67108864.0 + static_cast<double>(b)
        ) * (1.0 / 9007199254740992.0);
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
            || gauss_primary_tape == nullptr
            || gauss_secondary_tape == nullptr
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

__device__ __forceinline__ double canon_genome_double(
    const double value
) {
    if (!isfinite(value)) {
        return 0.0;
    }
    const float value32 = static_cast<float>(value);
    return nearbyint(static_cast<double>(value32) * 100000000.0)
        / 100000000.0;
}

__device__ __forceinline__ float canon_genome_float(const double value) {
    return static_cast<float>(canon_genome_double(value));
}

__device__ __forceinline__ int32_t snap_memory_dim(const int32_t value) {
    constexpr int32_t kLadder[5] = {0, 8, 16, 32, 64};
    int32_t best = kLadder[0];
    int32_t best_delta = abs(value - kLadder[0]);
    for (int32_t index = 1; index < 5; ++index) {
        const int32_t delta = abs(value - kLadder[index]);
        if (delta < best_delta) {
            best = kLadder[index];
            best_delta = delta;
        }
    }
    return best;
}

__device__ __forceinline__ int32_t memory_index(const int32_t value) {
    if (value <= 0) return 0;
    if (value <= 8) return 1;
    if (value <= 16) return 2;
    if (value <= 32) return 3;
    return 4;
}

__device__ __forceinline__ int32_t snap_sensor_radius(
    const int32_t value,
    const int32_t sensor_radius_max
) {
    constexpr int32_t kLadder[3] = {1, 2, 3};
    int32_t best = kLadder[0];
    int32_t best_delta = abs(value - kLadder[0]);
    for (int32_t index = 1; index < 3; ++index) {
        const int32_t delta = abs(value - kLadder[index]);
        if (delta < best_delta) {
            best = kLadder[index];
            best_delta = delta;
        }
    }
    if (sensor_radius_max > 0 && best > sensor_radius_max) {
        best = sensor_radius_max > 1 ? sensor_radius_max : 1;
    }
    return best > 1 ? best : 1;
}

__device__ __forceinline__ bool scalar_birth_rng_exceeds_gauss_tape(
    DeviceCpythonRandom& rng,
    const float parent_mutation_rate
) {
    (void)rng.random_double();
    const double base_mutation_rate = 0.5 * (
        canon_genome_double(parent_mutation_rate)
        + canon_genome_double(parent_mutation_rate)
    );
    const double scalar_mutation_rate = clamp_double(
        base_mutation_rate,
        0.0,
        1.0
    );
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
    const double memory_probability = clamp_double(
        0.25 * child_mutation_rate,
        0.002,
        0.05
    );
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

// Exact ordinary fixed-MLP transcription of immutable Stage2 commit
// e40633bc, native/cuda_architecture_b_lifecycle.cu. Exactly one device thread
// invokes this helper for one simulation cell. The fixed-size trial MT state is
// thread-local; the function performs no dynamic allocation or synchronization.
// Pointer validity, extents, dtypes, and config ranges remain launch-boundary
// obligations. Dynamic plan, RNG, and Gaussian capacity failures leave all
// authoritative genome and RNG state unchanged.
__device__ __forceinline__ void
commit_ordinary_fixed_mlp_child_scalar_genome_with_trial_scratch_serial(
    const FlatFixedMlpScalarGenomeState state,
    const FlatOrdinaryScalarGenomePlan plan,
    const FlatCpythonRandomState python_random,
    const FlatOrdinaryScalarGenomeOutputs outputs,
    const OrdinaryScalarGenomeConfig config,
    uint32_t* trial_words
) {
    outputs.committed_count[0] = 0;
    outputs.semantic_draws_added[0] = 0;
    outputs.status[0] = kStatusOk;

    const int64_t birth_count = plan.planned_count[0];
    if (
        birth_count < 0 || birth_count > config.plan_capacity
        || birth_count > config.slot_count
    ) {
        outputs.status[0] = kStatusInvalidScalarBirthPlan;
        return;
    }
    if (
        python_random.python_random_index[0] < 0
        || python_random.python_random_index[0] > kCpythonRandomWordCount
        || python_random.python_random_raw_draws[0] < 0
        || python_random.mutation_py_semantic_draws[0] < 0
    ) {
        outputs.status[0] = kStatusInvalidPythonRandomState;
        return;
    }

    int64_t previous_parent_id = -1;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = plan.parent_slots[birth];
        const int64_t child_slot = plan.child_slots[birth];
        if (
            child_slot < 0 || child_slot >= config.slot_count
            || !state.alive_mask[child_slot]
            || state.slot_to_agent_id[child_slot] < 0
            || parent_slot < 0 || parent_slot >= config.slot_count
            || parent_slot == child_slot
            || !state.alive_mask[parent_slot]
            || state.slot_to_agent_id[parent_slot] < 0
        ) {
            outputs.status[0] = kStatusInvalidScalarBirthPlan;
            return;
        }
        const int64_t parent_id = state.slot_to_agent_id[parent_slot];
        if (birth > 0 && parent_id <= previous_parent_id) {
            outputs.status[0] = kStatusInvalidScalarBirthPlan;
            return;
        }
        previous_parent_id = parent_id;
        for (int64_t previous = 0; previous < birth; ++previous) {
            if (plan.child_slots[previous] == child_slot) {
                outputs.status[0] = kStatusInvalidScalarBirthPlan;
                return;
            }
        }
    }

    for (int64_t word = 0; word < kCpythonRandomWordCount; ++word) {
        trial_words[word] = python_random.python_random_words[word];
    }
    DeviceCpythonRandom trial_rng{
        trial_words,
        python_random.python_random_index[0],
        python_random.python_random_has_gauss[0],
        python_random.python_random_gauss_next[0],
        python_random.python_random_gauss_primary_tape,
        python_random.python_random_gauss_secondary_tape,
        config.gauss_tape_capacity,
        python_random.python_random_raw_draws[0],
    };
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = plan.parent_slots[birth];
        if (scalar_birth_rng_exceeds_gauss_tape(
                trial_rng,
                state.genome_mutation_rate[parent_slot]
            )) {
            outputs.status[0] = kStatusFreshGaussUnsupported;
            return;
        }
    }

    DeviceCpythonRandom rng{
        python_random.python_random_words,
        python_random.python_random_index[0],
        python_random.python_random_has_gauss[0],
        python_random.python_random_gauss_next[0],
        python_random.python_random_gauss_primary_tape,
        python_random.python_random_gauss_secondary_tape,
        config.gauss_tape_capacity,
        python_random.python_random_raw_draws[0],
    };
    int64_t draws_added = 0;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = plan.parent_slots[birth];
        const int64_t child_slot = plan.child_slots[birth];

        const double hidden_size_decision = rng.random_double();
        draws_added += 1;
        const int32_t child_hidden_size = hidden_size_decision < 0.5
            ? state.genome_hidden_size[parent_slot]
            : state.genome_hidden_size[parent_slot];
        const double parent_mutation_rate = canon_genome_double(
            static_cast<double>(state.genome_mutation_rate[parent_slot])
        );
        const double mate_mutation_rate = canon_genome_double(
            static_cast<double>(state.genome_mutation_rate[parent_slot])
        );
        const double base_mutation_rate = 0.5 * (
            parent_mutation_rate + mate_mutation_rate
        );
        const double scalar_mutation_rate = clamp_double(
            base_mutation_rate,
            0.0,
            1.0
        );
        const double exploration_base = 0.5 * (
            canon_genome_double(state.genome_exploration_bias[parent_slot])
            + canon_genome_double(state.genome_exploration_bias[parent_slot])
        );
        const float child_exploration_bias = canon_genome_float(
            mutate_scalar_gene_cached_only(
                rng,
                exploration_base,
                scalar_mutation_rate,
                0.10,
                -1.0,
                1.0,
                &draws_added
            )
        );
        const double aggression_base = 0.5 * (
            canon_genome_double(state.genome_aggression_bias[parent_slot])
            + canon_genome_double(state.genome_aggression_bias[parent_slot])
        );
        const float child_aggression_bias = canon_genome_float(
            mutate_scalar_gene_cached_only(
                rng,
                aggression_base,
                scalar_mutation_rate,
                0.15,
                -2.0,
                2.0,
                &draws_added
            )
        );
        const double threshold_base = 0.5 * (
            canon_genome_double(
                state.genome_reproduction_threshold[parent_slot]
            )
            + canon_genome_double(
                state.genome_reproduction_threshold[parent_slot]
            )
        );
        const float child_reproduction_threshold = canon_genome_float(
            mutate_scalar_gene_cached_only(
                rng,
                threshold_base,
                scalar_mutation_rate,
                2.0,
                2.0,
                90.0,
                &draws_added
            )
        );
        const double temperature_base = 0.5 * (
            canon_genome_double(state.genome_action_temperature[parent_slot])
            + canon_genome_double(state.genome_action_temperature[parent_slot])
        );
        const float child_action_temperature = canon_genome_float(
            mutate_scalar_gene_cached_only(
                rng,
                temperature_base,
                scalar_mutation_rate,
                0.10,
                0.25,
                2.5,
                &draws_added
            )
        );
        const double inertia_base = 0.5 * (
            canon_genome_double(state.genome_drive_inertia[parent_slot])
            + canon_genome_double(state.genome_drive_inertia[parent_slot])
        );
        const float child_drive_inertia = canon_genome_float(
            mutate_scalar_gene_cached_only(
                rng,
                inertia_base,
                scalar_mutation_rate,
                0.05,
                0.0,
                0.999,
                &draws_added
            )
        );
        const double noise_base = 0.5 * (
            canon_genome_double(state.genome_drive_noise[parent_slot])
            + canon_genome_double(state.genome_drive_noise[parent_slot])
        );
        const float child_drive_noise = canon_genome_float(
            mutate_scalar_gene_cached_only(
                rng,
                noise_base,
                scalar_mutation_rate,
                0.01,
                0.0,
                0.10,
                &draws_added
            )
        );
        const double gain_base = 0.5 * (
            canon_genome_double(state.genome_drive_logit_gain[parent_slot])
            + canon_genome_double(state.genome_drive_logit_gain[parent_slot])
        );
        const float child_drive_logit_gain = canon_genome_float(
            mutate_scalar_gene_cached_only(
                rng,
                gain_base,
                scalar_mutation_rate,
                0.20,
                0.0,
                3.0,
                &draws_added
            )
        );
        const double pred_lr_base = 0.5 * (
            canon_genome_double(state.genome_pred_lr[parent_slot])
            + canon_genome_double(state.genome_pred_lr[parent_slot])
        );
        const float child_pred_lr = canon_genome_float(
            mutate_scalar_gene_cached_only(
                rng,
                pred_lr_base,
                scalar_mutation_rate,
                0.03,
                0.0,
                1.0,
                &draws_added
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
        child_mutation_rate_raw = clamp_double(
            child_mutation_rate_raw,
            0.001,
            0.20
        );
        const float child_mutation_rate = canon_genome_float(
            child_mutation_rate_raw
        );

        const int32_t parent_memory_dim =
            state.genome_memory_dim[parent_slot];
        const int32_t mate_memory_dim =
            state.genome_memory_dim[parent_slot];
        const int32_t inherited_base = snap_memory_dim(static_cast<int32_t>(
            llround(0.5 * static_cast<double>(
                parent_memory_dim + mate_memory_dim
            ))
        ));
        const double memory_probability = clamp_double(
            0.25 * child_mutation_rate_raw,
            0.002,
            0.05
        );
        const double memory_decision = rng.random_double();
        draws_added += 1;
        int32_t child_memory_dim = inherited_base;
        if (memory_decision < memory_probability) {
            int32_t inherited_memory_index = memory_index(inherited_base);
            const int32_t step = rng.randbelow(2) == 0 ? -1 : 1;
            draws_added += 1;
            inherited_memory_index += step;
            if (inherited_memory_index < 0) inherited_memory_index = 0;
            if (inherited_memory_index > 4) inherited_memory_index = 4;
            constexpr int32_t memory_ladder[5] = {0, 8, 16, 32, 64};
            child_memory_dim = memory_ladder[inherited_memory_index];
        }

        const int32_t child_sensor_mask =
            (state.genome_sensor_channel_mask[parent_slot] & 1023) | 1;
        const int32_t child_sensor_radius = snap_sensor_radius(
            state.genome_sensor_radius[parent_slot],
            config.sensor_radius_max
        );
        const double child_sp_hunger_scalar = canon_genome_double(clamp_double(
            0.5 * (
                canon_genome_double(state.genome_drive_sp_hunger[parent_slot])
                + canon_genome_double(
                    state.genome_drive_sp_hunger[parent_slot]
                )
            ),
            0.0,
            1.0
        ));
        const float child_sp_hunger =
            static_cast<float>(child_sp_hunger_scalar);
        const float child_sp_pain = canon_genome_float(clamp_double(
            0.5 * (
                canon_genome_double(state.genome_drive_sp_pain[parent_slot])
                + canon_genome_double(state.genome_drive_sp_pain[parent_slot])
            ),
            0.0,
            1.0
        ));
        const float child_sp_fatigue = canon_genome_float(clamp_double(
            0.5 * (
                canon_genome_double(state.genome_drive_sp_fatigue[parent_slot])
                + canon_genome_double(
                    state.genome_drive_sp_fatigue[parent_slot]
                )
            ),
            0.0,
            1.0
        ));
        const float child_sp_fear = canon_genome_float(clamp_double(
            0.5 * (
                canon_genome_double(state.genome_drive_sp_fear[parent_slot])
                + canon_genome_double(state.genome_drive_sp_fear[parent_slot])
            ),
            0.0,
            1.0
        ));
        const float child_exp_prior_strength = canon_genome_float(clamp_double(
            0.5 * (
                canon_genome_double(
                    state.genome_exp_prior_strength[parent_slot]
                )
                + canon_genome_double(
                    state.genome_exp_prior_strength[parent_slot]
                )
            ),
            0.0,
            1.0
        ));
        const float child_exp_prior_noise = canon_genome_float(clamp_double(
            0.5 * (
                canon_genome_double(state.genome_exp_prior_noise[parent_slot])
                + canon_genome_double(
                    state.genome_exp_prior_noise[parent_slot]
                )
            ),
            0.0,
            0.5
        ));
        const float child_drive_birth_noise = canon_genome_float(clamp_double(
            0.5 * (
                canon_genome_double(state.genome_drive_birth_noise[parent_slot])
                + canon_genome_double(
                    state.genome_drive_birth_noise[parent_slot]
                )
            ),
            0.0,
            0.2
        ));

        const bool child_use_gru = state.policy_use_gru[parent_slot];
        state.genome_hidden_size[child_slot] = child_hidden_size;
        state.genome_memory_dim[child_slot] = child_use_gru
            ? child_memory_dim
            : 0;
        state.policy_mem_dim[child_slot] = child_use_gru
            ? static_cast<int64_t>(child_memory_dim)
            : 0;
        state.policy_use_gru[child_slot] = child_use_gru;
        state.gru_upkeep_cost[child_slot] = child_use_gru
            ? state.gru_upkeep_cost[parent_slot]
            : 0.0f;
        state.genome_sensor_channel_mask[child_slot] = child_sensor_mask;
        state.genome_sensor_radius[child_slot] = child_sensor_radius;
        state.genome_introspection_channel_mask[child_slot] =
            state.genome_introspection_channel_mask[parent_slot] & 32767;
        state.genome_self_regulation_control_mask[child_slot] =
            state.genome_self_regulation_control_mask[parent_slot] & 31;
        state.genome_self_modification_target_mask[child_slot] =
            state.genome_self_modification_target_mask[parent_slot] & 31;
        state.genome_directed_inheritance_channel_mask[child_slot] =
            state.genome_directed_inheritance_channel_mask[parent_slot] & 63;
        state.genome_exploration_bias[child_slot] = child_exploration_bias;
        state.genome_aggression_bias[child_slot] = child_aggression_bias;
        state.genome_mutation_rate[child_slot] = child_mutation_rate;
        state.genome_reproduction_threshold[child_slot] =
            child_reproduction_threshold;
        state.genome_action_temperature[child_slot] = child_action_temperature;
        state.genome_drive_inertia[child_slot] = child_drive_inertia;
        state.genome_drive_noise[child_slot] = child_drive_noise;
        state.genome_drive_logit_gain[child_slot] = child_drive_logit_gain;
        state.genome_pred_lr[child_slot] = child_pred_lr;
        state.genome_drive_sp_hunger[child_slot] = child_sp_hunger;
        state.genome_drive_sp_pain[child_slot] = child_sp_pain;
        state.genome_drive_sp_fatigue[child_slot] = child_sp_fatigue;
        state.genome_drive_sp_fear[child_slot] = child_sp_fear;
        state.genome_exp_prior_strength[child_slot] = child_exp_prior_strength;
        state.genome_exp_prior_noise[child_slot] = child_exp_prior_noise;
        state.genome_drive_birth_noise[child_slot] = child_drive_birth_noise;
        state.repro_threshold[child_slot] = child_reproduction_threshold;
        state.policy_attack_bias[child_slot] = child_aggression_bias;
        state.policy_action_temperature[child_slot] = child_action_temperature;
        state.policy_drive_logit_gain[child_slot] = child_drive_logit_gain;
        state.drive_inertia[child_slot] = child_drive_inertia;
        state.drive_noise[child_slot] = child_drive_noise;
        state.pred_lr[child_slot] = child_pred_lr;
        if (child_use_gru) {
            (void)rng.random_double();
            draws_added += 1;
        }
    }

    python_random.python_random_index[0] = rng.index;
    python_random.python_random_has_gauss[0] = rng.has_gauss_next;
    python_random.python_random_gauss_next[0] = rng.gauss_next;
    python_random.python_random_raw_draws[0] = rng.raw_draws;
    python_random.mutation_py_semantic_draws[0] += draws_added;
    outputs.committed_count[0] = birth_count;
    outputs.semantic_draws_added[0] = draws_added;
    outputs.status[0] = kStatusOk;
}

// Compatibility entry point for the closed v7 experiment. New resident
// challengers must call the external-scratch overload above; this wrapper is
// intentionally the only remaining owner of the legacy thread-local MT copy.
__device__ __forceinline__ void
commit_ordinary_fixed_mlp_child_scalar_genome_serial(
    const FlatFixedMlpScalarGenomeState state,
    const FlatOrdinaryScalarGenomePlan plan,
    const FlatCpythonRandomState python_random,
    const FlatOrdinaryScalarGenomeOutputs outputs,
    const OrdinaryScalarGenomeConfig config
) {
    uint32_t trial_words[kCpythonRandomWordCount];
    commit_ordinary_fixed_mlp_child_scalar_genome_with_trial_scratch_serial(
        state,
        plan,
        python_random,
        outputs,
        config,
        trial_words
    );
}

}  // namespace eiel::cuda_architecture_b::persistent_birth_scalar_genome

#endif  // EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_BIRTH_SCALAR_GENOME_CUH_
