#ifndef EIEL_CUDA_ARCHITECTURE_B_FLATTENED_BIRTH_SCALAR_GENOME_V2_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_FLATTENED_BIRTH_SCALAR_GENOME_V2_CUH_

#include <cuda_runtime.h>

#include <cstdint>

#include "cuda_architecture_b_persistent_birth_scalar_genome.cuh"

namespace eiel::cuda_architecture_b::flattened_birth_scalar_genome_v2 {

namespace scalar = persistent_birth_scalar_genome;

constexpr int64_t kCpythonRandomWordCount =
    scalar::kCpythonRandomWordCount;

enum OrdinaryScalarGenomeStatus : int32_t {
    kStatusOk = scalar::kStatusOk,
    kStatusInvalidScalarBirthPlan =
        scalar::kStatusInvalidScalarBirthPlan,
    kStatusInvalidPythonRandomState =
        scalar::kStatusInvalidPythonRandomState,
    kStatusFreshGaussUnsupported =
        scalar::kStatusFreshGaussUnsupported,
};

static_assert(kCpythonRandomWordCount == 624);
static_assert(kStatusOk == 0);
static_assert(kStatusInvalidScalarBirthPlan == 20);
static_assert(kStatusInvalidPythonRandomState == 21);
static_assert(kStatusFreshGaussUnsupported == 22);

using FlatOrdinaryScalarGenomePlan =
    scalar::FlatOrdinaryScalarGenomePlan;
using OrdinaryScalarGenomeConfig =
    scalar::OrdinaryScalarGenomeConfig;

// All arrays are authoritative, cell-offset inputs. Child-slot scalar values
// are never read; child identity and liveness are validated through the first
// two fields exactly as in the persistent helper.
struct FlatAuthoritativeFixedMlpScalarGenomeState {
    const int64_t* slot_to_agent_id;
    const bool* alive_mask;

    const int32_t* genome_hidden_size;
    const int32_t* genome_memory_dim;
    const int32_t* genome_sensor_channel_mask;
    const int32_t* genome_sensor_radius;
    const int32_t* genome_introspection_channel_mask;
    const int32_t* genome_self_regulation_control_mask;
    const int32_t* genome_self_modification_target_mask;
    const int32_t* genome_directed_inheritance_channel_mask;
    const int64_t* policy_mem_dim;
    const bool* policy_use_gru;

    const float* genome_exploration_bias;
    const float* genome_aggression_bias;
    const float* genome_mutation_rate;
    const float* genome_reproduction_threshold;
    const float* genome_action_temperature;
    const float* genome_drive_inertia;
    const float* genome_drive_noise;
    const float* genome_drive_logit_gain;
    const float* genome_pred_lr;
    const float* genome_drive_sp_hunger;
    const float* genome_drive_sp_pain;
    const float* genome_drive_sp_fatigue;
    const float* genome_drive_sp_fear;
    const float* genome_exp_prior_strength;
    const float* genome_exp_prior_noise;
    const float* genome_drive_birth_noise;
    const float* repro_threshold;
    const float* gru_upkeep_cost;
    const float* policy_attack_bias;
    const float* policy_action_temperature;
    const float* policy_drive_logit_gain;
    const float* drive_inertia;
    const float* drive_noise;
    const float* pred_lr;
};

struct FlatAuthoritativeCpythonRandomState {
    const uint32_t* python_random_words;
    const int64_t* python_random_index;
    const bool* python_random_has_gauss;
    const double* python_random_gauss_next;
    const double* python_random_gauss_primary_tape;
    const double* python_random_gauss_secondary_tape;
    const int64_t* python_random_raw_draws;
    const int64_t* mutation_py_semantic_draws;
};

// Every child field has config.plan_capacity elements and is indexed by birth
// ordinal, not child
// slot. python_random_words and python_random_trial_words each have extent 624.
// The remaining RNG and result pointers each have extent 1. Scratch must not
// alias authoritative input or overlap another scratch field.
struct FlatBirthOrderedScalarGenomeScratch {
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

    uint32_t* python_random_words;
    uint32_t* python_random_trial_words;
    int64_t* python_random_index;
    bool* python_random_has_gauss;
    double* python_random_gauss_next;
    int64_t* python_random_raw_draws;
    int64_t* mutation_py_semantic_draws;

    int64_t* committed_count;
    int64_t* semantic_draws_added;
    int32_t* status;
};

// Allocation-free, single-thread scalar-genome transaction for zero through
// two ordinary births. This is an exact birth-ordered staging transcription of
// commit_ordinary_fixed_mlp_child_scalar_genome_with_trial_scratch_serial.
// It preserves that helper's validation, draw order, canonicalization,
// statuses, and Gaussian-tape preflight. Authoritative arrays and RNG state are
// read-only. On rejection, child rows and final RNG scratch remain unchanged;
// only result metadata and trial words may have been written. Pointer validity,
// extents, dtypes, non-aliasing, and config ranges remain caller obligations.
__device__ __forceinline__ void
stage_ordinary_fixed_mlp_child_scalar_genomes_external_scratch_v2(
    const FlatAuthoritativeFixedMlpScalarGenomeState state,
    const FlatOrdinaryScalarGenomePlan plan,
    const FlatAuthoritativeCpythonRandomState python_random,
    const FlatBirthOrderedScalarGenomeScratch scratch,
    const OrdinaryScalarGenomeConfig config
) {
    scratch.committed_count[0] = 0;
    scratch.semantic_draws_added[0] = 0;
    scratch.status[0] = kStatusOk;

    const int64_t birth_count = plan.planned_count[0];
    if (
        birth_count < 0 || birth_count > config.plan_capacity
        || birth_count > config.slot_count
    ) {
        scratch.status[0] = kStatusInvalidScalarBirthPlan;
        return;
    }
    if (
        python_random.python_random_index[0] < 0
        || python_random.python_random_index[0] > kCpythonRandomWordCount
        || python_random.python_random_raw_draws[0] < 0
        || python_random.mutation_py_semantic_draws[0] < 0
    ) {
        scratch.status[0] = kStatusInvalidPythonRandomState;
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
            || state.policy_use_gru[parent_slot]
        ) {
            scratch.status[0] = kStatusInvalidScalarBirthPlan;
            return;
        }
        const int64_t parent_id = state.slot_to_agent_id[parent_slot];
        if (birth > 0 && parent_id <= previous_parent_id) {
            scratch.status[0] = kStatusInvalidScalarBirthPlan;
            return;
        }
        previous_parent_id = parent_id;
        for (int64_t previous = 0; previous < birth; ++previous) {
            if (plan.child_slots[previous] == child_slot) {
                scratch.status[0] = kStatusInvalidScalarBirthPlan;
                return;
            }
        }
    }

    for (int64_t word = 0; word < kCpythonRandomWordCount; ++word) {
        scratch.python_random_trial_words[word] =
            python_random.python_random_words[word];
    }
    scalar::DeviceCpythonRandom trial_rng{
        scratch.python_random_trial_words,
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
        if (scalar::scalar_birth_rng_exceeds_gauss_tape(
                trial_rng,
                state.genome_mutation_rate[parent_slot]
            )) {
            scratch.status[0] = kStatusFreshGaussUnsupported;
            return;
        }
    }

    for (int64_t word = 0; word < kCpythonRandomWordCount; ++word) {
        scratch.python_random_words[word] =
            python_random.python_random_words[word];
    }
    scalar::DeviceCpythonRandom rng{
        scratch.python_random_words,
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

        const double hidden_size_decision = rng.random_double();
        draws_added += 1;
        const int32_t child_hidden_size = hidden_size_decision < 0.5
            ? state.genome_hidden_size[parent_slot]
            : state.genome_hidden_size[parent_slot];
        const double parent_mutation_rate = scalar::canon_genome_double(
            static_cast<double>(state.genome_mutation_rate[parent_slot])
        );
        const double mate_mutation_rate = scalar::canon_genome_double(
            static_cast<double>(state.genome_mutation_rate[parent_slot])
        );
        const double base_mutation_rate = 0.5 * (
            parent_mutation_rate + mate_mutation_rate
        );
        const double scalar_mutation_rate = scalar::clamp_double(
            base_mutation_rate,
            0.0,
            1.0
        );
        const double exploration_base = 0.5 * (
            scalar::canon_genome_double(
                state.genome_exploration_bias[parent_slot]
            )
            + scalar::canon_genome_double(
                state.genome_exploration_bias[parent_slot]
            )
        );
        const float child_exploration_bias = scalar::canon_genome_float(
            scalar::mutate_scalar_gene_cached_only(
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
            scalar::canon_genome_double(
                state.genome_aggression_bias[parent_slot]
            )
            + scalar::canon_genome_double(
                state.genome_aggression_bias[parent_slot]
            )
        );
        const float child_aggression_bias = scalar::canon_genome_float(
            scalar::mutate_scalar_gene_cached_only(
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
            scalar::canon_genome_double(
                state.genome_reproduction_threshold[parent_slot]
            )
            + scalar::canon_genome_double(
                state.genome_reproduction_threshold[parent_slot]
            )
        );
        const float child_reproduction_threshold =
            scalar::canon_genome_float(
                scalar::mutate_scalar_gene_cached_only(
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
            scalar::canon_genome_double(
                state.genome_action_temperature[parent_slot]
            )
            + scalar::canon_genome_double(
                state.genome_action_temperature[parent_slot]
            )
        );
        const float child_action_temperature = scalar::canon_genome_float(
            scalar::mutate_scalar_gene_cached_only(
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
            scalar::canon_genome_double(
                state.genome_drive_inertia[parent_slot]
            )
            + scalar::canon_genome_double(
                state.genome_drive_inertia[parent_slot]
            )
        );
        const float child_drive_inertia = scalar::canon_genome_float(
            scalar::mutate_scalar_gene_cached_only(
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
            scalar::canon_genome_double(
                state.genome_drive_noise[parent_slot]
            )
            + scalar::canon_genome_double(
                state.genome_drive_noise[parent_slot]
            )
        );
        const float child_drive_noise = scalar::canon_genome_float(
            scalar::mutate_scalar_gene_cached_only(
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
            scalar::canon_genome_double(
                state.genome_drive_logit_gain[parent_slot]
            )
            + scalar::canon_genome_double(
                state.genome_drive_logit_gain[parent_slot]
            )
        );
        const float child_drive_logit_gain = scalar::canon_genome_float(
            scalar::mutate_scalar_gene_cached_only(
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
            scalar::canon_genome_double(state.genome_pred_lr[parent_slot])
            + scalar::canon_genome_double(
                state.genome_pred_lr[parent_slot]
            )
        );
        const float child_pred_lr = scalar::canon_genome_float(
            scalar::mutate_scalar_gene_cached_only(
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
        child_mutation_rate_raw = scalar::clamp_double(
            child_mutation_rate_raw,
            0.001,
            0.20
        );
        const float child_mutation_rate = scalar::canon_genome_float(
            child_mutation_rate_raw
        );

        const int32_t parent_memory_dim =
            state.genome_memory_dim[parent_slot];
        const int32_t mate_memory_dim =
            state.genome_memory_dim[parent_slot];
        const int32_t inherited_base = scalar::snap_memory_dim(
            static_cast<int32_t>(::llround(0.5 * static_cast<double>(
                parent_memory_dim + mate_memory_dim
            )))
        );
        const double memory_probability = scalar::clamp_double(
            0.25 * child_mutation_rate_raw,
            0.002,
            0.05
        );
        const double memory_decision = rng.random_double();
        draws_added += 1;
        if (memory_decision < memory_probability) {
            int32_t inherited_memory_index =
                scalar::memory_index(inherited_base);
            const int32_t step = rng.randbelow(2) == 0 ? -1 : 1;
            draws_added += 1;
            inherited_memory_index += step;
            if (inherited_memory_index < 0) inherited_memory_index = 0;
            if (inherited_memory_index > 4) inherited_memory_index = 4;
        }

        const int32_t child_sensor_mask =
            (state.genome_sensor_channel_mask[parent_slot] & 1023) | 1;
        const int32_t child_sensor_radius = scalar::snap_sensor_radius(
            state.genome_sensor_radius[parent_slot],
            config.sensor_radius_max
        );
        const double child_sp_hunger_scalar = scalar::canon_genome_double(
            scalar::clamp_double(
                0.5 * (
                    scalar::canon_genome_double(
                        state.genome_drive_sp_hunger[parent_slot]
                    )
                    + scalar::canon_genome_double(
                        state.genome_drive_sp_hunger[parent_slot]
                    )
                ),
                0.0,
                1.0
            )
        );
        const float child_sp_hunger =
            static_cast<float>(child_sp_hunger_scalar);
        const float child_sp_pain = scalar::canon_genome_float(
            scalar::clamp_double(
                0.5 * (
                    scalar::canon_genome_double(
                        state.genome_drive_sp_pain[parent_slot]
                    )
                    + scalar::canon_genome_double(
                        state.genome_drive_sp_pain[parent_slot]
                    )
                ),
                0.0,
                1.0
            )
        );
        const float child_sp_fatigue = scalar::canon_genome_float(
            scalar::clamp_double(
                0.5 * (
                    scalar::canon_genome_double(
                        state.genome_drive_sp_fatigue[parent_slot]
                    )
                    + scalar::canon_genome_double(
                        state.genome_drive_sp_fatigue[parent_slot]
                    )
                ),
                0.0,
                1.0
            )
        );
        const float child_sp_fear = scalar::canon_genome_float(
            scalar::clamp_double(
                0.5 * (
                    scalar::canon_genome_double(
                        state.genome_drive_sp_fear[parent_slot]
                    )
                    + scalar::canon_genome_double(
                        state.genome_drive_sp_fear[parent_slot]
                    )
                ),
                0.0,
                1.0
            )
        );
        const float child_exp_prior_strength = scalar::canon_genome_float(
            scalar::clamp_double(
                0.5 * (
                    scalar::canon_genome_double(
                        state.genome_exp_prior_strength[parent_slot]
                    )
                    + scalar::canon_genome_double(
                        state.genome_exp_prior_strength[parent_slot]
                    )
                ),
                0.0,
                1.0
            )
        );
        const float child_exp_prior_noise = scalar::canon_genome_float(
            scalar::clamp_double(
                0.5 * (
                    scalar::canon_genome_double(
                        state.genome_exp_prior_noise[parent_slot]
                    )
                    + scalar::canon_genome_double(
                        state.genome_exp_prior_noise[parent_slot]
                    )
                ),
                0.0,
                0.5
            )
        );
        const float child_drive_birth_noise = scalar::canon_genome_float(
            scalar::clamp_double(
                0.5 * (
                    scalar::canon_genome_double(
                        state.genome_drive_birth_noise[parent_slot]
                    )
                    + scalar::canon_genome_double(
                        state.genome_drive_birth_noise[parent_slot]
                    )
                ),
                0.0,
                0.2
            )
        );

        scratch.genome_hidden_size[birth] = child_hidden_size;
        scratch.genome_memory_dim[birth] = 0;
        scratch.policy_mem_dim[birth] = 0;
        scratch.policy_use_gru[birth] = false;
        scratch.gru_upkeep_cost[birth] = 0.0f;
        scratch.genome_sensor_channel_mask[birth] = child_sensor_mask;
        scratch.genome_sensor_radius[birth] = child_sensor_radius;
        scratch.genome_introspection_channel_mask[birth] =
            state.genome_introspection_channel_mask[parent_slot] & 32767;
        scratch.genome_self_regulation_control_mask[birth] =
            state.genome_self_regulation_control_mask[parent_slot] & 31;
        scratch.genome_self_modification_target_mask[birth] =
            state.genome_self_modification_target_mask[parent_slot] & 31;
        scratch.genome_directed_inheritance_channel_mask[birth] =
            state.genome_directed_inheritance_channel_mask[parent_slot] & 63;
        scratch.genome_exploration_bias[birth] = child_exploration_bias;
        scratch.genome_aggression_bias[birth] = child_aggression_bias;
        scratch.genome_mutation_rate[birth] = child_mutation_rate;
        scratch.genome_reproduction_threshold[birth] =
            child_reproduction_threshold;
        scratch.genome_action_temperature[birth] =
            child_action_temperature;
        scratch.genome_drive_inertia[birth] = child_drive_inertia;
        scratch.genome_drive_noise[birth] = child_drive_noise;
        scratch.genome_drive_logit_gain[birth] = child_drive_logit_gain;
        scratch.genome_pred_lr[birth] = child_pred_lr;
        scratch.genome_drive_sp_hunger[birth] = child_sp_hunger;
        scratch.genome_drive_sp_pain[birth] = child_sp_pain;
        scratch.genome_drive_sp_fatigue[birth] = child_sp_fatigue;
        scratch.genome_drive_sp_fear[birth] = child_sp_fear;
        scratch.genome_exp_prior_strength[birth] =
            child_exp_prior_strength;
        scratch.genome_exp_prior_noise[birth] = child_exp_prior_noise;
        scratch.genome_drive_birth_noise[birth] = child_drive_birth_noise;
        scratch.repro_threshold[birth] = child_reproduction_threshold;
        scratch.policy_attack_bias[birth] = child_aggression_bias;
        scratch.policy_action_temperature[birth] =
            child_action_temperature;
        scratch.policy_drive_logit_gain[birth] = child_drive_logit_gain;
        scratch.drive_inertia[birth] = child_drive_inertia;
        scratch.drive_noise[birth] = child_drive_noise;
        scratch.pred_lr[birth] = child_pred_lr;
    }

    scratch.python_random_index[0] = rng.index;
    scratch.python_random_has_gauss[0] = rng.has_gauss_next;
    scratch.python_random_gauss_next[0] = rng.gauss_next;
    scratch.python_random_raw_draws[0] = rng.raw_draws;
    scratch.mutation_py_semantic_draws[0] =
        python_random.mutation_py_semantic_draws[0] + draws_added;
    scratch.committed_count[0] = birth_count;
    scratch.semantic_draws_added[0] = draws_added;
    scratch.status[0] = kStatusOk;
}

}  // namespace eiel::cuda_architecture_b::flattened_birth_scalar_genome_v2

#endif  // EIEL_CUDA_ARCHITECTURE_B_FLATTENED_BIRTH_SCALAR_GENOME_V2_CUH_
