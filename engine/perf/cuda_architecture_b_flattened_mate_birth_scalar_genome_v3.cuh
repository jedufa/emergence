#ifndef EIEL_CUDA_ARCHITECTURE_B_FLATTENED_MATE_BIRTH_SCALAR_GENOME_V3_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_FLATTENED_MATE_BIRTH_SCALAR_GENOME_V3_CUH_

#include <cuda_runtime.h>

#include <cstdint>

#include "cuda_architecture_b_flattened_birth_scalar_genome_v2.cuh"
#include "cuda_architecture_b_persistent_turnover_scalar_genome.cuh"
#include "inheritance_linkage_selection_v1.cuh"

namespace eiel::cuda_architecture_b::flattened_mate_birth_scalar_genome_v3 {

namespace asexual = flattened_birth_scalar_genome_v2;
namespace scalar = persistent_birth_scalar_genome;
namespace turnover = persistent_turnover_scalar_genome;
namespace linkage = eiel::inheritance_linkage_v1;

constexpr int64_t kCpythonRandomWordCount = scalar::kCpythonRandomWordCount;

enum OrdinaryScalarGenomeStatus : int32_t {
    kStatusOk = scalar::kStatusOk,
    kStatusInvalidScalarBirthPlan = scalar::kStatusInvalidScalarBirthPlan,
    kStatusInvalidPythonRandomState = scalar::kStatusInvalidPythonRandomState,
    kStatusFreshGaussUnsupported = scalar::kStatusFreshGaussUnsupported,
    kStatusInvalidInheritancePlan = 23,
};

static_assert(kCpythonRandomWordCount == 624);
static_assert(kStatusOk == 0);
static_assert(kStatusInvalidScalarBirthPlan == 20);
static_assert(kStatusInvalidPythonRandomState == 21);
static_assert(kStatusFreshGaussUnsupported == 22);
static_assert(kStatusInvalidInheritancePlan == 23);

using FlatAuthoritativeFixedMlpScalarGenomeState =
    asexual::FlatAuthoritativeFixedMlpScalarGenomeState;
using FlatAuthoritativeCpythonRandomState =
    asexual::FlatAuthoritativeCpythonRandomState;
using FlatBirthOrderedScalarGenomeScratch =
    asexual::FlatBirthOrderedScalarGenomeScratch;
using OrdinaryScalarGenomeConfig = scalar::OrdinaryScalarGenomeConfig;

// Every array is birth ordered. A mate slot of -1 marks an asexual row and
// selects the parent as the genetic mate. A nonnegative mate slot is the
// already-selected genetic mate from the runtime B ordinary-birth plan.
struct FlatMateOrdinaryScalarGenomePlan {
    const int64_t* parent_slots;
    const int64_t* mate_slots;
    const int64_t* child_slots;
    const int64_t* planned_count;
};

// Optional versioned inheritance policy for the same ordinary-birth plan.
// Canonical parent slots are birth ordered and must be an exact permutation of
// each row's parent and mate. Donor ordinals are a caller-owned
// [plan_capacity, 7] row-major output. Legacy mode accepts null canonical/output
// pointers and adds no donor draws. Linked mode requires every pointer and
// consumes exactly seven CPython mutation-stream draws per committed child
// before the unchanged post-inheritance mutation schedule.
struct FlatMateOrdinaryInheritancePlanV1 {
    linkage::InheritanceModeV1 mode;
    // true preserves the accepted post-inheritance mutation schedule; false
    // disables that schedule and every mutation draw without changing donor
    // selection. The legacy v3 wrapper always supplies true.
    bool mutation_enabled;
    const int64_t* canonical_parent_0_slots;
    const int64_t* canonical_parent_1_slots;
    uint8_t* donor_ordinals;
};

__device__ __forceinline__ bool linked_mode_v1(
    const FlatMateOrdinaryInheritancePlanV1* inheritance
) {
    return inheritance != nullptr
        && inheritance->mode
            == linkage::InheritanceModeV1::linked_parent_copy_v1;
}

__device__ __forceinline__ double inherited_scalar_gene_double_v1(
    const float* values,
    const int64_t direct_parent_slot,
    const int64_t direct_mate_slot,
    const int64_t canonical_parent_0_slot,
    const int64_t canonical_parent_1_slot,
    const bool linked,
    const uint8_t* donor_ordinals,
    const uint32_t linkage_group
) {
    if (!linked) {
        return 0.5 * (
            scalar::canon_genome_double(
                static_cast<double>(values[direct_parent_slot])
            )
            + scalar::canon_genome_double(
                static_cast<double>(values[direct_mate_slot])
            )
        );
    }
    const auto selection = linkage::select_linkage_v1(
        linkage::InheritanceModeV1::linked_parent_copy_v1,
        linkage_group,
        donor_ordinals,
        linkage::kLinkedDonorDrawCount
    );
    const int64_t donor_slot = selection.donor_ordinal == 0
        ? canonical_parent_0_slot
        : canonical_parent_1_slot;
    return scalar::canon_genome_double(
        static_cast<double>(values[donor_slot])
    );
}

__device__ __forceinline__ float inherited_scalar_gene_v1(
    const float* values,
    const int64_t direct_parent_slot,
    const int64_t direct_mate_slot,
    const int64_t canonical_parent_0_slot,
    const int64_t canonical_parent_1_slot,
    const bool linked,
    const uint8_t* donor_ordinals,
    const uint32_t linkage_group
) {
    return scalar::canon_genome_float(inherited_scalar_gene_double_v1(
        values,
        direct_parent_slot,
        direct_mate_slot,
        canonical_parent_0_slot,
        canonical_parent_1_slot,
        linked,
        donor_ordinals,
        linkage_group
    ));
}

__device__ __forceinline__ bool
linked_scalar_birth_rng_exceeds_gauss_tape_v1(
    scalar::DeviceCpythonRandom& rng,
    const float selected_mutation_rate
) {
    const double scalar_mutation_rate = scalar::clamp_double(
        scalar::canon_genome_double(
            static_cast<double>(selected_mutation_rate)
        ),
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
    double child_mutation_rate = scalar_mutation_rate;
    if (rng.random_double() < 0.2) {
        double delta = 0.0;
        if (!rng.consume_gauss(0.005, &delta)) {
            return true;
        }
        child_mutation_rate += delta;
    }
    child_mutation_rate = scalar::clamp_double(
        child_mutation_rate,
        0.001,
        0.20
    );
    const double memory_probability = scalar::clamp_double(
        0.25 * child_mutation_rate,
        0.002,
        0.05
    );
    if (rng.random_double() < memory_probability) {
        (void)rng.randbelow(2);
    }
    return false;
}

__device__ __forceinline__ double maybe_mutate_scalar_gene_v1(
    scalar::DeviceCpythonRandom& rng,
    const double inherited_value,
    const bool mutation_enabled,
    const double mutation_rate,
    const double sigma,
    const double lower,
    const double upper,
    int64_t* draws_added
) {
    if (!mutation_enabled) {
        return scalar::clamp_double(inherited_value, lower, upper);
    }
    return scalar::mutate_scalar_gene_cached_only(
        rng,
        inherited_value,
        mutation_rate,
        sigma,
        lower,
        upper,
        draws_added
    );
}

__device__ __forceinline__ double midpoint_gene(
    const float* values,
    const int64_t parent_slot,
    const int64_t genetic_mate_slot
) {
    return 0.5 * (
        scalar::canon_genome_double(static_cast<double>(values[parent_slot]))
        + scalar::canon_genome_double(
            static_cast<double>(values[genetic_mate_slot])
        )
    );
}

__device__ __forceinline__ float midpoint_gene_clamped(
    const float* values,
    const int64_t parent_slot,
    const int64_t genetic_mate_slot,
    const double lower,
    const double upper
) {
    return scalar::canon_genome_float(scalar::clamp_double(
        midpoint_gene(values, parent_slot, genetic_mate_slot),
        lower,
        upper
    ));
}

__device__ __forceinline__ float inherited_scalar_gene_clamped_v1(
    const float* values,
    const int64_t direct_parent_slot,
    const int64_t direct_mate_slot,
    const int64_t canonical_parent_0_slot,
    const int64_t canonical_parent_1_slot,
    const bool linked,
    const uint8_t* donor_ordinals,
    const uint32_t linkage_group,
    const double lower,
    const double upper
) {
    return scalar::canon_genome_float(scalar::clamp_double(
        inherited_scalar_gene_double_v1(
            values,
            direct_parent_slot,
            direct_mate_slot,
            canonical_parent_0_slot,
            canonical_parent_1_slot,
            linked,
            donor_ordinals,
            linkage_group
        ),
        lower,
        upper
    ));
}

// Allocation-free, single-thread, birth-ordered staging for the runtime B
// ordinary-birth plan. This preserves the accepted C++ owner/Python scalar
// recombination equations and CPython mutation-stream draw order. The plan
// supplies genetic pairs, so unlike turnover this helper performs no pair
// redraw. Authoritative arrays and RNG state are read-only. On rejection,
// staged child rows and final RNG scratch remain unchanged; result metadata and
// trial words may have been written. Pointer validity, extents, dtypes,
// non-aliasing, and config ranges remain caller obligations.
__device__ __forceinline__ void
stage_ordinary_fixed_mlp_child_scalar_genomes_with_mates_external_scratch_impl_v4(
    const FlatAuthoritativeFixedMlpScalarGenomeState state,
    const FlatMateOrdinaryScalarGenomePlan plan,
    const FlatAuthoritativeCpythonRandomState python_random,
    const FlatBirthOrderedScalarGenomeScratch scratch,
    const OrdinaryScalarGenomeConfig config,
    const FlatMateOrdinaryInheritancePlanV1* inheritance
) {
    scratch.committed_count[0] = 0;
    scratch.semantic_draws_added[0] = 0;
    scratch.status[0] = kStatusOk;

    const int64_t birth_count = plan.planned_count[0];
    const bool linked = linked_mode_v1(inheritance);
    if (
        inheritance != nullptr
        && inheritance->mode != linkage::InheritanceModeV1::legacy_midpoint_v1
        && inheritance->mode
            != linkage::InheritanceModeV1::linked_parent_copy_v1
    ) {
        scratch.status[0] = kStatusInvalidInheritancePlan;
        return;
    }
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
        const int64_t mate_slot = plan.mate_slots[birth];
        const int64_t child_slot = plan.child_slots[birth];
        if (
            child_slot < 0 || child_slot >= config.slot_count
            || !state.alive_mask[child_slot]
            || state.slot_to_agent_id[child_slot] < 0
            || parent_slot < 0 || parent_slot >= config.slot_count
            || parent_slot == child_slot || !state.alive_mask[parent_slot]
            || state.slot_to_agent_id[parent_slot] < 0
            || state.policy_use_gru[parent_slot]
        ) {
            scratch.status[0] = kStatusInvalidScalarBirthPlan;
            return;
        }
        if (
            mate_slot < -1 || mate_slot >= config.slot_count
            || (mate_slot >= 0 && (
                mate_slot == parent_slot || mate_slot == child_slot
                || !state.alive_mask[mate_slot]
                || state.slot_to_agent_id[mate_slot] < 0
                || state.policy_use_gru[mate_slot]
            ))
        ) {
            scratch.status[0] = kStatusInvalidScalarBirthPlan;
            return;
        }
        if (linked) {
            if (
                mate_slot < 0
                || inheritance->canonical_parent_0_slots == nullptr
                || inheritance->canonical_parent_1_slots == nullptr
                || inheritance->donor_ordinals == nullptr
            ) {
                scratch.status[0] = kStatusInvalidInheritancePlan;
                return;
            }
            const int64_t canonical_0 =
                inheritance->canonical_parent_0_slots[birth];
            const int64_t canonical_1 =
                inheritance->canonical_parent_1_slots[birth];
            if (
                canonical_0 == canonical_1
                || !(
                    (canonical_0 == parent_slot && canonical_1 == mate_slot)
                    || (
                        canonical_0 == mate_slot
                        && canonical_1 == parent_slot
                    )
                )
            ) {
                scratch.status[0] = kStatusInvalidInheritancePlan;
                return;
            }
        }
        const int64_t parent_id = state.slot_to_agent_id[parent_slot];
        if (!scalar::parent_order::valid_parent_ordinal_v1(
                config.parent_order_mode,
                birth,
                parent_id,
                previous_parent_id,
                plan.parent_slots,
                state.slot_to_agent_id
            )) {
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
        if (inheritance != nullptr && !inheritance->mutation_enabled) {
            continue;
        }
        const int64_t parent_slot = plan.parent_slots[birth];
        const int64_t mate_slot = plan.mate_slots[birth];
        const int64_t genetic_mate_slot =
            mate_slot >= 0 ? mate_slot : parent_slot;
        bool gauss_tape_exceeded = false;
        if (linked) {
            uint8_t trial_donors[linkage::kLinkedDonorDrawCount];
            for (
                uint32_t group = 0;
                group < linkage::kLinkedDonorDrawCount;
                ++group
            ) {
                trial_donors[group] =
                    trial_rng.random_double() < 0.5 ? 0 : 1;
            }
            const int64_t canonical_0 =
                inheritance->canonical_parent_0_slots[birth];
            const int64_t canonical_1 =
                inheritance->canonical_parent_1_slots[birth];
            const float selected_mutation_rate = inherited_scalar_gene_double_v1(
                state.genome_mutation_rate,
                parent_slot,
                genetic_mate_slot,
                canonical_0,
                canonical_1,
                true,
                trial_donors,
                3
            );
            gauss_tape_exceeded =
                linked_scalar_birth_rng_exceeds_gauss_tape_v1(
                    trial_rng,
                    selected_mutation_rate
                );
        } else {
            gauss_tape_exceeded =
                turnover::turnover_scalar_birth_rng_exceeds_gauss_tape(
                    trial_rng,
                    state.genome_mutation_rate[parent_slot],
                    state.genome_mutation_rate[genetic_mate_slot]
                );
        }
        if (gauss_tape_exceeded) {
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
        const int64_t mate_slot = plan.mate_slots[birth];
        const int64_t genetic_mate_slot =
            mate_slot >= 0 ? mate_slot : parent_slot;
        const int64_t canonical_0 = linked
            ? inheritance->canonical_parent_0_slots[birth]
            : parent_slot;
        const int64_t canonical_1 = linked
            ? inheritance->canonical_parent_1_slots[birth]
            : genetic_mate_slot;
        uint8_t local_donors[linkage::kLinkedDonorDrawCount] = {};
        if (linked) {
            for (
                uint32_t group = 0;
                group < linkage::kLinkedDonorDrawCount;
                ++group
            ) {
                const uint8_t donor =
                    rng.random_double() < 0.5 ? 0 : 1;
                local_donors[group] = donor;
                inheritance->donor_ordinals[
                    birth * linkage::kLinkedDonorDrawCount + group
                ] = donor;
                draws_added += 1;
            }
        }

        int32_t child_hidden_size = 0;
        if (linked) {
            child_hidden_size =
                local_donors[4] == 0
                    ? state.genome_hidden_size[canonical_0]
                    : state.genome_hidden_size[canonical_1];
        } else {
            const double hidden_size_decision = rng.random_double();
            draws_added += 1;
            child_hidden_size = hidden_size_decision < 0.5
                ? state.genome_hidden_size[parent_slot]
                : state.genome_hidden_size[genetic_mate_slot];
        }
        const double base_mutation_rate = inherited_scalar_gene_double_v1(
            state.genome_mutation_rate,
            parent_slot,
            genetic_mate_slot,
            canonical_0,
            canonical_1,
            linked,
            local_donors,
            3
        );
        const double scalar_mutation_rate = scalar::clamp_double(
            base_mutation_rate,
            0.0,
            1.0
        );

        const float child_exploration_bias = scalar::canon_genome_float(
            maybe_mutate_scalar_gene_v1(
                rng,
                inherited_scalar_gene_double_v1(
                    state.genome_exploration_bias,
                    parent_slot,
                    genetic_mate_slot,
                    canonical_0,
                    canonical_1,
                    linked,
                    local_donors,
                    0
                ),
                inheritance == nullptr || inheritance->mutation_enabled,
                scalar_mutation_rate,
                0.10,
                -1.0,
                1.0,
                &draws_added
            )
        );
        const float child_aggression_bias = scalar::canon_genome_float(
            maybe_mutate_scalar_gene_v1(
                rng,
                inherited_scalar_gene_double_v1(
                    state.genome_aggression_bias,
                    parent_slot,
                    genetic_mate_slot,
                    canonical_0,
                    canonical_1,
                    linked,
                    local_donors,
                    0
                ),
                inheritance == nullptr || inheritance->mutation_enabled,
                scalar_mutation_rate,
                0.15,
                -2.0,
                2.0,
                &draws_added
            )
        );
        const float child_reproduction_threshold = scalar::canon_genome_float(
            maybe_mutate_scalar_gene_v1(
                rng,
                inherited_scalar_gene_double_v1(
                    state.genome_reproduction_threshold,
                    parent_slot,
                    genetic_mate_slot,
                    canonical_0,
                    canonical_1,
                    linked,
                    local_donors,
                    2
                ),
                inheritance == nullptr || inheritance->mutation_enabled,
                scalar_mutation_rate,
                2.0,
                2.0,
                90.0,
                &draws_added
            )
        );
        const float child_action_temperature = scalar::canon_genome_float(
            maybe_mutate_scalar_gene_v1(
                rng,
                inherited_scalar_gene_double_v1(
                    state.genome_action_temperature,
                    parent_slot,
                    genetic_mate_slot,
                    canonical_0,
                    canonical_1,
                    linked,
                    local_donors,
                    0
                ),
                inheritance == nullptr || inheritance->mutation_enabled,
                scalar_mutation_rate,
                0.10,
                0.25,
                2.5,
                &draws_added
            )
        );
        const float child_drive_inertia = scalar::canon_genome_float(
            maybe_mutate_scalar_gene_v1(
                rng,
                inherited_scalar_gene_double_v1(
                    state.genome_drive_inertia,
                    parent_slot,
                    genetic_mate_slot,
                    canonical_0,
                    canonical_1,
                    linked,
                    local_donors,
                    1
                ),
                inheritance == nullptr || inheritance->mutation_enabled,
                scalar_mutation_rate,
                0.05,
                0.0,
                0.999,
                &draws_added
            )
        );
        const float child_drive_noise = scalar::canon_genome_float(
            maybe_mutate_scalar_gene_v1(
                rng,
                inherited_scalar_gene_double_v1(
                    state.genome_drive_noise,
                    parent_slot,
                    genetic_mate_slot,
                    canonical_0,
                    canonical_1,
                    linked,
                    local_donors,
                    1
                ),
                inheritance == nullptr || inheritance->mutation_enabled,
                scalar_mutation_rate,
                0.01,
                0.0,
                0.10,
                &draws_added
            )
        );
        const float child_drive_logit_gain = scalar::canon_genome_float(
            maybe_mutate_scalar_gene_v1(
                rng,
                inherited_scalar_gene_double_v1(
                    state.genome_drive_logit_gain,
                    parent_slot,
                    genetic_mate_slot,
                    canonical_0,
                    canonical_1,
                    linked,
                    local_donors,
                    1
                ),
                inheritance == nullptr || inheritance->mutation_enabled,
                scalar_mutation_rate,
                0.20,
                0.0,
                3.0,
                &draws_added
            )
        );
        const float child_pred_lr = scalar::canon_genome_float(
            maybe_mutate_scalar_gene_v1(
                rng,
                inherited_scalar_gene_double_v1(
                    state.genome_pred_lr,
                    parent_slot,
                    genetic_mate_slot,
                    canonical_0,
                    canonical_1,
                    linked,
                    local_donors,
                    1
                ),
                inheritance == nullptr || inheritance->mutation_enabled,
                scalar_mutation_rate,
                0.03,
                0.0,
                1.0,
                &draws_added
            )
        );

        double child_mutation_rate_raw = base_mutation_rate;
        const bool mutation_enabled =
            inheritance == nullptr || inheritance->mutation_enabled;
        if (mutation_enabled) {
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
        }
        const float child_mutation_rate = scalar::canon_genome_float(
            child_mutation_rate_raw
        );

        int32_t child_memory_dim = scalar::snap_memory_dim(
            linked
                ? (
                    local_donors[4] == 0
                        ? state.genome_memory_dim[canonical_0]
                        : state.genome_memory_dim[canonical_1]
                )
                : static_cast<int32_t>(::llround(
                    0.5 * static_cast<double>(
                        state.genome_memory_dim[parent_slot]
                        + state.genome_memory_dim[genetic_mate_slot]
                    )
                ))
        );
        const double memory_probability = scalar::clamp_double(
            0.25 * child_mutation_rate_raw,
            0.002,
            0.05
        );
        if (mutation_enabled) {
            if (rng.random_double() < memory_probability) {
                int32_t inherited_memory_index = scalar::memory_index(
                    child_memory_dim
                );
                const int32_t step = rng.randbelow(2) == 0 ? -1 : 1;
                draws_added += 1;
                inherited_memory_index += step;
                if (inherited_memory_index < 0) inherited_memory_index = 0;
                if (inherited_memory_index > 4) inherited_memory_index = 4;
                static constexpr int32_t kMemoryLadder[5] = {
                    0, 8, 16, 32, 64
                };
                child_memory_dim = kMemoryLadder[inherited_memory_index];
            }
            draws_added += 1;
        }

        const int32_t child_sensor_mask =
            (state.genome_sensor_channel_mask[parent_slot] & 1023) | 1;
        const int32_t child_sensor_radius = scalar::snap_sensor_radius(
            state.genome_sensor_radius[parent_slot],
            config.sensor_radius_max
        );
        const float child_sp_hunger = inherited_scalar_gene_clamped_v1(
            state.genome_drive_sp_hunger,
            parent_slot,
            genetic_mate_slot,
            canonical_0,
            canonical_1,
            linked,
            local_donors,
            1,
            0.0,
            1.0
        );
        const float child_sp_pain = inherited_scalar_gene_clamped_v1(
            state.genome_drive_sp_pain,
            parent_slot,
            genetic_mate_slot,
            canonical_0,
            canonical_1,
            linked,
            local_donors,
            1,
            0.0,
            1.0
        );
        const float child_sp_fatigue = inherited_scalar_gene_clamped_v1(
            state.genome_drive_sp_fatigue,
            parent_slot,
            genetic_mate_slot,
            canonical_0,
            canonical_1,
            linked,
            local_donors,
            1,
            0.0,
            1.0
        );
        const float child_sp_fear = inherited_scalar_gene_clamped_v1(
            state.genome_drive_sp_fear,
            parent_slot,
            genetic_mate_slot,
            canonical_0,
            canonical_1,
            linked,
            local_donors,
            1,
            0.0,
            1.0
        );
        const float child_exp_prior_strength = inherited_scalar_gene_clamped_v1(
            state.genome_exp_prior_strength,
            parent_slot,
            genetic_mate_slot,
            canonical_0,
            canonical_1,
            linked,
            local_donors,
            1,
            0.0,
            1.0
        );
        const float child_exp_prior_noise = inherited_scalar_gene_clamped_v1(
            state.genome_exp_prior_noise,
            parent_slot,
            genetic_mate_slot,
            canonical_0,
            canonical_1,
            linked,
            local_donors,
            1,
            0.0,
            0.5
        );
        const float child_drive_birth_noise = inherited_scalar_gene_clamped_v1(
            state.genome_drive_birth_noise,
            parent_slot,
            genetic_mate_slot,
            canonical_0,
            canonical_1,
            linked,
            local_donors,
            1,
            0.0,
            0.2
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
        scratch.genome_action_temperature[birth] = child_action_temperature;
        scratch.genome_drive_inertia[birth] = child_drive_inertia;
        scratch.genome_drive_noise[birth] = child_drive_noise;
        scratch.genome_drive_logit_gain[birth] = child_drive_logit_gain;
        scratch.genome_pred_lr[birth] = child_pred_lr;
        scratch.genome_drive_sp_hunger[birth] = child_sp_hunger;
        scratch.genome_drive_sp_pain[birth] = child_sp_pain;
        scratch.genome_drive_sp_fatigue[birth] = child_sp_fatigue;
        scratch.genome_drive_sp_fear[birth] = child_sp_fear;
        scratch.genome_exp_prior_strength[birth] = child_exp_prior_strength;
        scratch.genome_exp_prior_noise[birth] = child_exp_prior_noise;
        scratch.genome_drive_birth_noise[birth] = child_drive_birth_noise;
        scratch.repro_threshold[birth] = child_reproduction_threshold;
        scratch.policy_attack_bias[birth] = child_aggression_bias;
        scratch.policy_action_temperature[birth] = child_action_temperature;
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

// Backward-compatible accepted legacy entry point. This wrapper deliberately
// provides no inheritance plan, preserving the exact v3 midpoint/RNG behavior.
__device__ __forceinline__ void
stage_ordinary_fixed_mlp_child_scalar_genomes_with_mates_external_scratch_v3(
    const FlatAuthoritativeFixedMlpScalarGenomeState state,
    const FlatMateOrdinaryScalarGenomePlan plan,
    const FlatAuthoritativeCpythonRandomState python_random,
    const FlatBirthOrderedScalarGenomeScratch scratch,
    const OrdinaryScalarGenomeConfig config
) {
    stage_ordinary_fixed_mlp_child_scalar_genomes_with_mates_external_scratch_impl_v4(
        state,
        plan,
        python_random,
        scratch,
        config,
        nullptr
    );
}

// Versioned entry point for legacy midpoint or linked seven-block donor copy.
// The caller owns and validates canonical hereditary parent ordering; this
// transaction validates only that each supplied canonical pair is an exact
// permutation of the already-selected reproductive pair.
__device__ __forceinline__ void
stage_ordinary_fixed_mlp_child_scalar_genomes_with_mates_external_scratch_v4(
    const FlatAuthoritativeFixedMlpScalarGenomeState state,
    const FlatMateOrdinaryScalarGenomePlan plan,
    const FlatAuthoritativeCpythonRandomState python_random,
    const FlatBirthOrderedScalarGenomeScratch scratch,
    const OrdinaryScalarGenomeConfig config,
    const FlatMateOrdinaryInheritancePlanV1 inheritance
) {
    stage_ordinary_fixed_mlp_child_scalar_genomes_with_mates_external_scratch_impl_v4(
        state,
        plan,
        python_random,
        scratch,
        config,
        &inheritance
    );
}

}  // namespace eiel::cuda_architecture_b::flattened_mate_birth_scalar_genome_v3

#endif  // EIEL_CUDA_ARCHITECTURE_B_FLATTENED_MATE_BIRTH_SCALAR_GENOME_V3_CUH_
