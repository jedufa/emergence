#ifndef EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_TURNOVER_SCALAR_GENOME_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_TURNOVER_SCALAR_GENOME_CUH_

#include <cuda_runtime.h>

#include <cstdint>

#include "cuda_architecture_b_persistent_birth_scalar_genome.cuh"
#include "inheritance_linkage_selection_v1.cuh"

namespace eiel::cuda_architecture_b::persistent_turnover_scalar_genome {

namespace birth = persistent_birth_scalar_genome;
namespace inheritance = eiel::inheritance_linkage_v1;

constexpr int64_t kCpythonRandomWordCount = 624;

enum TurnoverScalarGenomeStatus : int32_t {
    kStatusOk = 0,
    kStatusInvalidScalarBirthPlan = 20,
    kStatusInvalidPythonRandomState = 21,
    kStatusFreshGaussUnsupported = 22,
    kStatusInvalidInheritanceMode = 23,
    kStatusInvalidInheritanceDonorBudget = 24,
    kStatusMissingInheritanceSidecar = 25,
    kStatusInvalidCanonicalParentOrder = 26,
    kStatusMissingInheritanceOutputs = 27,
    kStatusInvalidCanonicalParentRankAuthority = 28,
    kStatusInheritanceSidecarCapacityExceeded = 29,
};

using FlatCpythonRandomState = birth::FlatCpythonRandomState;
using FlatFixedMlpScalarGenomeState = birth::FlatFixedMlpScalarGenomeState;
using DeviceCpythonRandom = birth::DeviceCpythonRandom;

// Stage2 consumes provisional parent/mate choices while planning spawn rows,
// then deliberately redraws the genetic pair inside the scalar transaction.
// The child slots come from the provisional identity allocation; selected
// parent/mate slots are published for the authoritative identity rebind.
struct FlatTurnoverScalarGenomePlan {
    const int64_t* parent_pool_slots;
    const int64_t* parent_pool_count;
    const int64_t* child_slots;
    const int64_t* planned_count;
};

struct FlatTurnoverScalarGenomeOutputs {
    int64_t* selected_parent_slots;
    int64_t* selected_mate_slots;
    int64_t* committed_count;
    int64_t* semantic_draws_added;
    int32_t* status;
};

struct TurnoverScalarGenomeConfig {
    int64_t plan_capacity;
    int64_t slot_count;
    int64_t gauss_tape_capacity;
    int32_t sensor_radius_max;
};

// Canonical ordering is deliberately caller-owned. The transaction cannot
// reconstruct canonical_hereditary_payload_sha256_v1 from its reduced resident
// scalar state. In linked mode each row must therefore provide the already
// canonicalized slots as an exact permutation of the transaction-selected
// parent/mate pair.
struct TurnoverInheritanceConfigV1 {
    inheritance::InheritanceModeV1 mode;
    std::uint32_t donor_draw_budget;
    // false disables every post-inheritance mutation draw while retaining
    // the mode-owned donor schedule. The exact legacy wrapper supplies true.
    bool mutation_enabled;
    const int64_t* canonical_parent_0_slots;
    const int64_t* canonical_parent_1_slots;
};

// Turnover selects its authoritative genetic pair inside the scalar RNG
// transaction. A caller that owns resident hereditary-payload ranks must bind
// that authority to the selected pair, rather than to an earlier provisional
// spawn pair. The output rows are transaction-owned publication sidecars.
struct TurnoverCanonicalParentRankAuthorityV1 {
    const int64_t* genotype_codes_by_slot;
    const int64_t* digest_rank_by_code;
    int64_t genotype_count;
    int64_t* canonical_parent_0_slots_out;
    int64_t* canonical_parent_1_slots_out;
};

struct TurnoverRankedInheritanceConfigV2 {
    inheritance::InheritanceModeV1 mode;
    std::uint32_t donor_draw_budget;
    bool mutation_enabled;
    TurnoverCanonicalParentRankAuthorityV1 canonical_parent_rank_authority;
};

// donor_ordinals and donor_statuses are row-major [plan_capacity, 7].
// status and donor_draws_added are one-element transaction aggregates.
struct FlatTurnoverInheritanceOutputsV1 {
    std::uint8_t* donor_ordinals;
    std::uint8_t* donor_statuses;
    int64_t* donor_draws_added;
    int32_t* status;
};

__device__ __forceinline__ bool resolve_turnover_canonical_parent_order_v2(
    const TurnoverInheritanceConfigV1 inheritance_config,
    const TurnoverCanonicalParentRankAuthorityV1* const rank_authority,
    const int64_t parent_slot,
    const int64_t mate_slot,
    const int64_t slot_count,
    const int64_t birth,
    int64_t* const canonical_parent_0_slot,
    int64_t* const canonical_parent_1_slot
) {
    if (
        canonical_parent_0_slot == nullptr
        || canonical_parent_1_slot == nullptr
        || parent_slot < 0 || parent_slot >= slot_count
        || mate_slot < 0 || mate_slot >= slot_count
        || parent_slot == mate_slot
    ) {
        return false;
    }
    if (rank_authority == nullptr) {
        if (
            inheritance_config.canonical_parent_0_slots == nullptr
            || inheritance_config.canonical_parent_1_slots == nullptr
        ) {
            return false;
        }
        const int64_t supplied_parent_0 =
            inheritance_config.canonical_parent_0_slots[birth];
        const int64_t supplied_parent_1 =
            inheritance_config.canonical_parent_1_slots[birth];
        const bool exact_permutation =
            (
                supplied_parent_0 == parent_slot
                && supplied_parent_1 == mate_slot
            )
            || (
                supplied_parent_0 == mate_slot
                && supplied_parent_1 == parent_slot
            );
        if (!exact_permutation) {
            return false;
        }
        canonical_parent_0_slot[0] = supplied_parent_0;
        canonical_parent_1_slot[0] = supplied_parent_1;
        return true;
    }
    if (
        rank_authority->genotype_codes_by_slot == nullptr
        || rank_authority->digest_rank_by_code == nullptr
        || rank_authority->genotype_count <= 0
        || rank_authority->canonical_parent_0_slots_out == nullptr
        || rank_authority->canonical_parent_1_slots_out == nullptr
    ) {
        return false;
    }
    const int64_t parent_code =
        rank_authority->genotype_codes_by_slot[parent_slot];
    const int64_t mate_code =
        rank_authority->genotype_codes_by_slot[mate_slot];
    if (
        parent_code < 0 || parent_code >= rank_authority->genotype_count
        || mate_code < 0 || mate_code >= rank_authority->genotype_count
    ) {
        return false;
    }
    const int64_t parent_rank =
        rank_authority->digest_rank_by_code[parent_code];
    const int64_t mate_rank =
        rank_authority->digest_rank_by_code[mate_code];
    if (parent_rank < 0 || mate_rank < 0) {
        return false;
    }
    const bool parent_first = parent_rank <= mate_rank;
    canonical_parent_0_slot[0] = parent_first ? parent_slot : mate_slot;
    canonical_parent_1_slot[0] = parent_first ? mate_slot : parent_slot;
    return true;
}

__device__ __forceinline__ double select_turnover_scalar_base_v1(
    const bool linked_inheritance,
    const TurnoverInheritanceConfigV1 inheritance_config,
    const std::uint32_t linkage_group,
    const std::uint8_t* donor_ordinals,
    const std::uint32_t donor_draw_count,
    const double legacy_parent_value,
    const double legacy_mate_value,
    const double canonical_parent_0_value,
    const double canonical_parent_1_value
) {
    if (!linked_inheritance) {
        return 0.5 * (legacy_parent_value + legacy_mate_value);
    }
    return inheritance::select_scalar_v1<double>(
        inheritance_config.mode,
        linkage_group,
        donor_ordinals,
        donor_draw_count,
        canonical_parent_0_value,
        canonical_parent_1_value
    ).value;
}

__device__ __forceinline__ bool
turnover_scalar_birth_rng_exceeds_gauss_tape_for_inheritance_v1(
    DeviceCpythonRandom& rng,
    const double base_mutation_rate,
    const bool consume_legacy_hidden_size_draw
) {
    if (consume_legacy_hidden_size_draw) {
        (void)rng.random_double();
    }
    const double scalar_mutation_rate = birth::clamp_double(
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
    child_mutation_rate = birth::clamp_double(
        child_mutation_rate,
        0.001,
        0.20
    );
    const double memory_probability = birth::clamp_double(
        0.25 * child_mutation_rate,
        0.002,
        0.05
    );
    if (rng.random_double() < memory_probability) {
        (void)rng.randbelow(2);
    }
    return false;
}

__device__ __forceinline__ double maybe_mutate_turnover_scalar_gene_v1(
    DeviceCpythonRandom& rng,
    const double base,
    const bool mutation_enabled,
    const double probability,
    const double sigma,
    const double lower,
    const double upper,
    int64_t* const draws_added
) {
    if (!mutation_enabled) {
        return base;
    }
    return birth::mutate_scalar_gene_cached_only(
        rng,
        base,
        probability,
        sigma,
        lower,
        upper,
        draws_added
    );
}

__device__ __forceinline__ bool turnover_scalar_birth_rng_exceeds_gauss_tape(
    DeviceCpythonRandom& rng,
    const float parent_mutation_rate,
    const float mate_mutation_rate
) {
    const double base_mutation_rate = 0.5 * (
        birth::canon_genome_double(static_cast<double>(parent_mutation_rate))
        + birth::canon_genome_double(static_cast<double>(mate_mutation_rate))
    );
    return turnover_scalar_birth_rng_exceeds_gauss_tape_for_inheritance_v1(
        rng,
        base_mutation_rate,
        true
    );
}

// Exact turnover-only fixed-MLP transcription of immutable Stage2
// native/cuda_architecture_b_lifecycle.cu. The interleaved pair redraw is part
// of the scalar RNG transaction. The helper uses one device thread per cell,
// performs no allocation or synchronization, uses caller-owned 624-word trial
// scratch, and commits no authoritative genome or RNG state until all dynamic
// Gaussian-tape checks have passed.
__device__ __forceinline__ void
commit_turnover_fixed_mlp_child_scalar_genome_with_inheritance_impl_v2_serial(
    const FlatFixedMlpScalarGenomeState state,
    const FlatTurnoverScalarGenomePlan plan,
    const FlatCpythonRandomState python_random,
    const FlatTurnoverScalarGenomeOutputs outputs,
    const TurnoverScalarGenomeConfig config,
    uint32_t* python_random_trial_words,
    const TurnoverInheritanceConfigV1 inheritance_config,
    const TurnoverCanonicalParentRankAuthorityV1* const rank_authority,
    const FlatTurnoverInheritanceOutputsV1 inheritance_outputs
) {
    outputs.committed_count[0] = 0;
    outputs.semantic_draws_added[0] = 0;
    outputs.status[0] = kStatusOk;
    if (inheritance_outputs.status != nullptr) {
        inheritance_outputs.status[0] = kStatusOk;
    }
    if (inheritance_outputs.donor_draws_added != nullptr) {
        inheritance_outputs.donor_draws_added[0] = 0;
    }

    const auto required_donor_draws =
        inheritance::required_donor_draws_v1(inheritance_config.mode);
    if (required_donor_draws == static_cast<std::uint32_t>(-1)) {
        outputs.status[0] = kStatusInvalidInheritanceMode;
        if (inheritance_outputs.status != nullptr) {
            inheritance_outputs.status[0] = kStatusInvalidInheritanceMode;
        }
        return;
    }
    if (inheritance_config.donor_draw_budget != required_donor_draws) {
        outputs.status[0] = kStatusInvalidInheritanceDonorBudget;
        if (inheritance_outputs.status != nullptr) {
            inheritance_outputs.status[0] =
                kStatusInvalidInheritanceDonorBudget;
        }
        return;
    }
    const bool linked_inheritance =
        inheritance_config.mode ==
        inheritance::InheritanceModeV1::linked_parent_copy_v1;
    if (
        linked_inheritance && rank_authority == nullptr
        && (
            inheritance_config.canonical_parent_0_slots == nullptr
            || inheritance_config.canonical_parent_1_slots == nullptr
        )
    ) {
        outputs.status[0] = kStatusMissingInheritanceSidecar;
        if (inheritance_outputs.status != nullptr) {
            inheritance_outputs.status[0] =
                kStatusMissingInheritanceSidecar;
        }
        return;
    }
    if (
        linked_inheritance && rank_authority != nullptr
        && (
            rank_authority->genotype_codes_by_slot == nullptr
            || rank_authority->digest_rank_by_code == nullptr
            || rank_authority->genotype_count <= 0
            || rank_authority->canonical_parent_0_slots_out == nullptr
            || rank_authority->canonical_parent_1_slots_out == nullptr
        )
    ) {
        outputs.status[0] = kStatusInvalidCanonicalParentRankAuthority;
        if (inheritance_outputs.status != nullptr) {
            inheritance_outputs.status[0] =
                kStatusInvalidCanonicalParentRankAuthority;
        }
        return;
    }
    if (
        linked_inheritance
        && (
            inheritance_outputs.donor_ordinals == nullptr
            || inheritance_outputs.donor_statuses == nullptr
            || inheritance_outputs.donor_draws_added == nullptr
            || inheritance_outputs.status == nullptr
        )
    ) {
        outputs.status[0] = kStatusMissingInheritanceOutputs;
        if (inheritance_outputs.status != nullptr) {
            inheritance_outputs.status[0] =
                kStatusMissingInheritanceOutputs;
        }
        return;
    }

    const int64_t birth_count = plan.planned_count[0];
    const int64_t parent_pool_count = plan.parent_pool_count[0];
    if (
        birth_count < 0 || birth_count > config.plan_capacity
        || birth_count > config.slot_count || config.plan_capacity < 0
        || config.slot_count <= 0 || config.gauss_tape_capacity < 0
        || config.sensor_radius_max < 0
        || parent_pool_count < (birth_count > 0 ? 2 : 0)
        || parent_pool_count > config.slot_count
    ) {
        outputs.status[0] = kStatusInvalidScalarBirthPlan;
        return;
    }
    if (
        python_random.python_random_index[0] < 0
        || python_random.python_random_index[0] > kCpythonRandomWordCount
        || python_random.python_random_raw_draws[0] < 0
        || python_random.mutation_py_semantic_draws[0] < 0
        || python_random_trial_words == nullptr
    ) {
        outputs.status[0] = kStatusInvalidPythonRandomState;
        return;
    }

    for (int64_t row = 0; row < config.plan_capacity; ++row) {
        outputs.selected_parent_slots[row] = -1;
        outputs.selected_mate_slots[row] = -1;
        if (linked_inheritance && rank_authority != nullptr) {
            rank_authority->canonical_parent_0_slots_out[row] = -1;
            rank_authority->canonical_parent_1_slots_out[row] = -1;
        }
        if (
            inheritance_outputs.donor_ordinals != nullptr
            && inheritance_outputs.donor_statuses != nullptr
        ) {
            for (
                std::uint32_t group = 0;
                group < inheritance::kLinkageGroupCount;
                ++group
            ) {
                const int64_t offset =
                    row * inheritance::kLinkageGroupCount + group;
                inheritance_outputs.donor_ordinals[offset] =
                    inheritance::kNoDonorOrdinal;
                inheritance_outputs.donor_statuses[offset] =
                    static_cast<std::uint8_t>(
                        inheritance::SelectionStatusV1::ok
                    );
            }
        }
    }
    for (int64_t row = 0; row < parent_pool_count; ++row) {
        const int64_t slot = plan.parent_pool_slots[row];
        if (
            slot < 0 || slot >= config.slot_count
            || !state.alive_mask[slot]
            || state.slot_to_agent_id[slot] < 0
        ) {
            outputs.status[0] = kStatusInvalidScalarBirthPlan;
            return;
        }
        for (int64_t previous = 0; previous < row; ++previous) {
            if (plan.parent_pool_slots[previous] == slot) {
                outputs.status[0] = kStatusInvalidScalarBirthPlan;
                return;
            }
        }
    }
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t child_slot = plan.child_slots[birth];
        if (
            child_slot < 0 || child_slot >= config.slot_count
            || !state.alive_mask[child_slot]
            || state.slot_to_agent_id[child_slot] < 0
        ) {
            outputs.status[0] = kStatusInvalidScalarBirthPlan;
            return;
        }
        for (int64_t previous = 0; previous < birth; ++previous) {
            if (plan.child_slots[previous] == child_slot) {
                outputs.status[0] = kStatusInvalidScalarBirthPlan;
                return;
            }
        }
    }

    for (int64_t word = 0; word < kCpythonRandomWordCount; ++word) {
        python_random_trial_words[word] =
            python_random.python_random_words[word];
    }
    DeviceCpythonRandom trial_rng{
        python_random_trial_words,
        python_random.python_random_index[0],
        python_random.python_random_has_gauss[0],
        python_random.python_random_gauss_next[0],
        python_random.python_random_gauss_primary_tape,
        python_random.python_random_gauss_secondary_tape,
        config.gauss_tape_capacity,
        python_random.python_random_raw_draws[0],
    };
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        int64_t parent_slot = -1;
        int64_t mate_slot = -1;
        do {
            parent_slot = plan.parent_pool_slots[trial_rng.randbelow(
                static_cast<uint64_t>(parent_pool_count)
            )];
            mate_slot = plan.parent_pool_slots[trial_rng.randbelow(
                static_cast<uint64_t>(parent_pool_count)
            )];
        } while (mate_slot == parent_slot);

        int64_t canonical_parent_0_slot = parent_slot;
        int64_t canonical_parent_1_slot = mate_slot;
        std::uint8_t donor_ordinals[inheritance::kLinkageGroupCount] = {};
        if (linked_inheritance) {
            if (!resolve_turnover_canonical_parent_order_v2(
                    inheritance_config,
                    rank_authority,
                    parent_slot,
                    mate_slot,
                    config.slot_count,
                    birth,
                    &canonical_parent_0_slot,
                    &canonical_parent_1_slot
                )) {
                outputs.status[0] = rank_authority == nullptr
                    ? kStatusInvalidCanonicalParentOrder
                    : kStatusInvalidCanonicalParentRankAuthority;
                inheritance_outputs.status[0] =
                    outputs.status[0];
                return;
            }
            for (
                std::uint32_t group = 0;
                group < required_donor_draws;
                ++group
            ) {
                donor_ordinals[group] =
                    trial_rng.random_double() < 0.5 ? 0 : 1;
            }
        }
        double trial_base_mutation_rate = 0.5 * (
            birth::canon_genome_double(static_cast<double>(
                state.genome_mutation_rate[parent_slot]
            ))
            + birth::canon_genome_double(static_cast<double>(
                state.genome_mutation_rate[mate_slot]
            ))
        );
        if (linked_inheritance) {
            const auto selected_mutation_rate =
                inheritance::select_scalar_v1<double>(
                    inheritance_config.mode,
                    3,
                    donor_ordinals,
                    required_donor_draws,
                    birth::canon_genome_double(static_cast<double>(
                        state.genome_mutation_rate[
                            canonical_parent_0_slot
                        ]
                    )),
                    birth::canon_genome_double(static_cast<double>(
                        state.genome_mutation_rate[
                            canonical_parent_1_slot
                        ]
                    ))
                );
            if (
                selected_mutation_rate.status !=
                inheritance::SelectionStatusV1::ok
            ) {
                outputs.status[0] = kStatusInvalidInheritanceDonorBudget;
                inheritance_outputs.status[0] =
                    kStatusInvalidInheritanceDonorBudget;
                return;
            }
            trial_base_mutation_rate = selected_mutation_rate.value;
        }
        if (!inheritance_config.mutation_enabled) {
            continue;
        }
        if (turnover_scalar_birth_rng_exceeds_gauss_tape_for_inheritance_v1(
                trial_rng,
                trial_base_mutation_rate,
                !linked_inheritance
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
        int64_t parent_slot = -1;
        int64_t mate_slot = -1;
        do {
            parent_slot = plan.parent_pool_slots[rng.randbelow(
                static_cast<uint64_t>(parent_pool_count)
            )];
            mate_slot = plan.parent_pool_slots[rng.randbelow(
                static_cast<uint64_t>(parent_pool_count)
            )];
            draws_added += 2;
        } while (mate_slot == parent_slot);
        const int64_t child_slot = plan.child_slots[birth];
        outputs.selected_parent_slots[birth] = parent_slot;
        outputs.selected_mate_slots[birth] = mate_slot;

        int64_t canonical_parent_0_slot = parent_slot;
        int64_t canonical_parent_1_slot = mate_slot;
        std::uint8_t donor_ordinals[inheritance::kLinkageGroupCount] = {};
        if (linked_inheritance) {
            if (!resolve_turnover_canonical_parent_order_v2(
                    inheritance_config,
                    rank_authority,
                    parent_slot,
                    mate_slot,
                    config.slot_count,
                    birth,
                    &canonical_parent_0_slot,
                    &canonical_parent_1_slot
                )) {
                outputs.status[0] = rank_authority == nullptr
                    ? kStatusInvalidCanonicalParentOrder
                    : kStatusInvalidCanonicalParentRankAuthority;
                inheritance_outputs.status[0] = outputs.status[0];
                return;
            }
            if (rank_authority != nullptr) {
                rank_authority->canonical_parent_0_slots_out[birth] =
                    canonical_parent_0_slot;
                rank_authority->canonical_parent_1_slots_out[birth] =
                    canonical_parent_1_slot;
            }
            for (
                std::uint32_t group = 0;
                group < required_donor_draws;
                ++group
            ) {
                donor_ordinals[group] =
                    rng.random_double() < 0.5 ? 0 : 1;
                draws_added += 1;
                const auto selection = inheritance::select_linkage_v1(
                    inheritance_config.mode,
                    group,
                    donor_ordinals,
                    required_donor_draws
                );
                const int64_t offset =
                    birth * inheritance::kLinkageGroupCount + group;
                inheritance_outputs.donor_ordinals[offset] =
                    donor_ordinals[group];
                inheritance_outputs.donor_statuses[offset] =
                    static_cast<std::uint8_t>(selection.status);
            }
        }

        int32_t child_hidden_size = 0;
        if (linked_inheritance) {
            const auto memory_selection = inheritance::select_linkage_v1(
                inheritance_config.mode,
                4,
                donor_ordinals,
                required_donor_draws
            );
            const int64_t memory_parent_slot =
                memory_selection.donor_ordinal == 0
                ? canonical_parent_0_slot
                : canonical_parent_1_slot;
            child_hidden_size =
                state.genome_hidden_size[memory_parent_slot];
        } else {
            const double hidden_size_decision = rng.random_double();
            draws_added += 1;
            child_hidden_size = hidden_size_decision < 0.5
                ? state.genome_hidden_size[parent_slot]
                : state.genome_hidden_size[mate_slot];
        }
        const double parent_mutation_rate = birth::canon_genome_double(
            static_cast<double>(state.genome_mutation_rate[parent_slot])
        );
        const double mate_mutation_rate = birth::canon_genome_double(
            static_cast<double>(state.genome_mutation_rate[mate_slot])
        );
        double base_mutation_rate = 0.5 * (
            parent_mutation_rate + mate_mutation_rate
        );
        if (linked_inheritance) {
            base_mutation_rate = inheritance::select_scalar_v1<double>(
                inheritance_config.mode,
                3,
                donor_ordinals,
                required_donor_draws,
                birth::canon_genome_double(static_cast<double>(
                    state.genome_mutation_rate[canonical_parent_0_slot]
                )),
                birth::canon_genome_double(static_cast<double>(
                    state.genome_mutation_rate[canonical_parent_1_slot]
                ))
            ).value;
        }
        const double scalar_mutation_rate = birth::clamp_double(
            base_mutation_rate,
            0.0,
            1.0
        );

        const double exploration_base = linked_inheritance
            ? inheritance::select_scalar_v1<double>(
                inheritance_config.mode,
                0,
                donor_ordinals,
                required_donor_draws,
                birth::canon_genome_double(
                    state.genome_exploration_bias[
                        canonical_parent_0_slot
                    ]
                ),
                birth::canon_genome_double(
                    state.genome_exploration_bias[
                        canonical_parent_1_slot
                    ]
                )
            ).value
            : 0.5 * (
                birth::canon_genome_double(
                    state.genome_exploration_bias[parent_slot]
                )
                + birth::canon_genome_double(
                    state.genome_exploration_bias[mate_slot]
                )
            );
        const float child_exploration_bias = birth::canon_genome_float(
            maybe_mutate_turnover_scalar_gene_v1(
                rng, exploration_base, inheritance_config.mutation_enabled,
                scalar_mutation_rate, 0.10,
                -1.0, 1.0, &draws_added
            )
        );
        const double aggression_base = linked_inheritance
            ? inheritance::select_scalar_v1<double>(
                inheritance_config.mode,
                0,
                donor_ordinals,
                required_donor_draws,
                birth::canon_genome_double(
                    state.genome_aggression_bias[
                        canonical_parent_0_slot
                    ]
                ),
                birth::canon_genome_double(
                    state.genome_aggression_bias[
                        canonical_parent_1_slot
                    ]
                )
            ).value
            : 0.5 * (
                birth::canon_genome_double(
                    state.genome_aggression_bias[parent_slot]
                )
                + birth::canon_genome_double(
                    state.genome_aggression_bias[mate_slot]
                )
            );
        const float child_aggression_bias = birth::canon_genome_float(
            maybe_mutate_turnover_scalar_gene_v1(
                rng, aggression_base, inheritance_config.mutation_enabled,
                scalar_mutation_rate, 0.15,
                -2.0, 2.0, &draws_added
            )
        );
        const double threshold_base = linked_inheritance
            ? inheritance::select_scalar_v1<double>(
                inheritance_config.mode,
                2,
                donor_ordinals,
                required_donor_draws,
                birth::canon_genome_double(
                    state.genome_reproduction_threshold[
                        canonical_parent_0_slot
                    ]
                ),
                birth::canon_genome_double(
                    state.genome_reproduction_threshold[
                        canonical_parent_1_slot
                    ]
                )
            ).value
            : 0.5 * (
                birth::canon_genome_double(
                    state.genome_reproduction_threshold[parent_slot]
                )
                + birth::canon_genome_double(
                    state.genome_reproduction_threshold[mate_slot]
                )
            );
        const float child_reproduction_threshold = birth::canon_genome_float(
            maybe_mutate_turnover_scalar_gene_v1(
                rng, threshold_base, inheritance_config.mutation_enabled,
                scalar_mutation_rate, 2.0,
                2.0, 90.0, &draws_added
            )
        );
        const double temperature_base = linked_inheritance
            ? inheritance::select_scalar_v1<double>(
                inheritance_config.mode,
                0,
                donor_ordinals,
                required_donor_draws,
                birth::canon_genome_double(
                    state.genome_action_temperature[
                        canonical_parent_0_slot
                    ]
                ),
                birth::canon_genome_double(
                    state.genome_action_temperature[
                        canonical_parent_1_slot
                    ]
                )
            ).value
            : 0.5 * (
                birth::canon_genome_double(
                    state.genome_action_temperature[parent_slot]
                )
                + birth::canon_genome_double(
                    state.genome_action_temperature[mate_slot]
                )
            );
        const float child_action_temperature = birth::canon_genome_float(
            maybe_mutate_turnover_scalar_gene_v1(
                rng, temperature_base, inheritance_config.mutation_enabled,
                scalar_mutation_rate, 0.10,
                0.25, 2.5, &draws_added
            )
        );
        const double inertia_base = select_turnover_scalar_base_v1(
            linked_inheritance,
            inheritance_config,
            1,
            donor_ordinals,
            required_donor_draws,
            birth::canon_genome_double(
                state.genome_drive_inertia[parent_slot]
            ),
            birth::canon_genome_double(
                state.genome_drive_inertia[mate_slot]
            ),
            birth::canon_genome_double(
                state.genome_drive_inertia[canonical_parent_0_slot]
            ),
            birth::canon_genome_double(
                state.genome_drive_inertia[canonical_parent_1_slot]
            )
        );
        const float child_drive_inertia = birth::canon_genome_float(
            maybe_mutate_turnover_scalar_gene_v1(
                rng, inertia_base, inheritance_config.mutation_enabled,
                scalar_mutation_rate, 0.05,
                0.0, 0.999, &draws_added
            )
        );
        const double noise_base = select_turnover_scalar_base_v1(
            linked_inheritance,
            inheritance_config,
            1,
            donor_ordinals,
            required_donor_draws,
            birth::canon_genome_double(
                state.genome_drive_noise[parent_slot]
            ),
            birth::canon_genome_double(
                state.genome_drive_noise[mate_slot]
            ),
            birth::canon_genome_double(
                state.genome_drive_noise[canonical_parent_0_slot]
            ),
            birth::canon_genome_double(
                state.genome_drive_noise[canonical_parent_1_slot]
            )
        );
        const float child_drive_noise = birth::canon_genome_float(
            maybe_mutate_turnover_scalar_gene_v1(
                rng, noise_base, inheritance_config.mutation_enabled,
                scalar_mutation_rate, 0.01,
                0.0, 0.10, &draws_added
            )
        );
        const double gain_base = select_turnover_scalar_base_v1(
            linked_inheritance,
            inheritance_config,
            1,
            donor_ordinals,
            required_donor_draws,
            birth::canon_genome_double(
                state.genome_drive_logit_gain[parent_slot]
            ),
            birth::canon_genome_double(
                state.genome_drive_logit_gain[mate_slot]
            ),
            birth::canon_genome_double(
                state.genome_drive_logit_gain[canonical_parent_0_slot]
            ),
            birth::canon_genome_double(
                state.genome_drive_logit_gain[canonical_parent_1_slot]
            )
        );
        const float child_drive_logit_gain = birth::canon_genome_float(
            maybe_mutate_turnover_scalar_gene_v1(
                rng, gain_base, inheritance_config.mutation_enabled,
                scalar_mutation_rate, 0.20,
                0.0, 3.0, &draws_added
            )
        );
        const double pred_lr_base = select_turnover_scalar_base_v1(
            linked_inheritance,
            inheritance_config,
            1,
            donor_ordinals,
            required_donor_draws,
            birth::canon_genome_double(state.genome_pred_lr[parent_slot]),
            birth::canon_genome_double(state.genome_pred_lr[mate_slot]),
            birth::canon_genome_double(
                state.genome_pred_lr[canonical_parent_0_slot]
            ),
            birth::canon_genome_double(
                state.genome_pred_lr[canonical_parent_1_slot]
            )
        );
        const float child_pred_lr = birth::canon_genome_float(
            maybe_mutate_turnover_scalar_gene_v1(
                rng, pred_lr_base, inheritance_config.mutation_enabled,
                scalar_mutation_rate, 0.03,
                0.0, 1.0, &draws_added
            )
        );

        double child_mutation_rate_raw = base_mutation_rate;
        if (inheritance_config.mutation_enabled) {
            const double mutation_rate_decision = rng.random_double();
            draws_added += 1;
            if (mutation_rate_decision < 0.2) {
                double mutation_rate_delta = 0.0;
                (void)rng.consume_gauss(0.005, &mutation_rate_delta);
                child_mutation_rate_raw += mutation_rate_delta;
                draws_added += 1;
            }
            child_mutation_rate_raw = birth::clamp_double(
                child_mutation_rate_raw,
                0.001,
                0.20
            );
        }
        const float child_mutation_rate = birth::canon_genome_float(
            child_mutation_rate_raw
        );

        int64_t memory_parent_slot = parent_slot;
        int32_t inherited_base = 0;
        if (linked_inheritance) {
            const auto memory_selection = inheritance::select_linkage_v1(
                inheritance_config.mode,
                4,
                donor_ordinals,
                required_donor_draws
            );
            memory_parent_slot = memory_selection.donor_ordinal == 0
                ? canonical_parent_0_slot
                : canonical_parent_1_slot;
            inherited_base = birth::snap_memory_dim(
                state.genome_memory_dim[memory_parent_slot]
            );
        } else {
            const int32_t parent_memory_dim =
                state.genome_memory_dim[parent_slot];
            const int32_t mate_memory_dim =
                state.genome_memory_dim[mate_slot];
            inherited_base = birth::snap_memory_dim(
                static_cast<int32_t>(llround(0.5 * static_cast<double>(
                    parent_memory_dim + mate_memory_dim
                )))
            );
        }
        int32_t child_memory_dim = inherited_base;
        if (inheritance_config.mutation_enabled) {
            const double memory_probability = birth::clamp_double(
                0.25 * child_mutation_rate_raw,
                0.002,
                0.05
            );
            const double memory_decision = rng.random_double();
            draws_added += 1;
            if (memory_decision < memory_probability) {
                int32_t inherited_memory_index = birth::memory_index(
                    inherited_base
                );
                const int32_t step = rng.randbelow(2) == 0 ? -1 : 1;
                draws_added += 1;
                inherited_memory_index += step;
                if (inherited_memory_index < 0) inherited_memory_index = 0;
                if (inherited_memory_index > 4) inherited_memory_index = 4;
                constexpr int32_t memory_ladder[5] = {0, 8, 16, 32, 64};
                child_memory_dim = memory_ladder[inherited_memory_index];
            }
        }

        // Stage2 keeps these structural channels parent-directed while scalar
        // drives and policy aliases use the two-parent recombination above.
        const int32_t child_sensor_mask =
            (state.genome_sensor_channel_mask[parent_slot] & 1023) | 1;
        const int32_t child_sensor_radius = birth::snap_sensor_radius(
            state.genome_sensor_radius[parent_slot],
            config.sensor_radius_max
        );
        const float child_sp_hunger = birth::canon_genome_float(
            birth::clamp_double(select_turnover_scalar_base_v1(
                linked_inheritance,
                inheritance_config,
                1,
                donor_ordinals,
                required_donor_draws,
                birth::canon_genome_double(
                    state.genome_drive_sp_hunger[parent_slot]
                ),
                birth::canon_genome_double(
                    state.genome_drive_sp_hunger[mate_slot]
                ),
                birth::canon_genome_double(
                    state.genome_drive_sp_hunger[
                        canonical_parent_0_slot
                    ]
                ),
                birth::canon_genome_double(
                    state.genome_drive_sp_hunger[
                        canonical_parent_1_slot
                    ]
                )
            ), 0.0, 1.0)
        );
        const float child_sp_pain = birth::canon_genome_float(
            birth::clamp_double(select_turnover_scalar_base_v1(
                linked_inheritance,
                inheritance_config,
                1,
                donor_ordinals,
                required_donor_draws,
                birth::canon_genome_double(
                    state.genome_drive_sp_pain[parent_slot]
                ),
                birth::canon_genome_double(
                    state.genome_drive_sp_pain[mate_slot]
                ),
                birth::canon_genome_double(
                    state.genome_drive_sp_pain[canonical_parent_0_slot]
                ),
                birth::canon_genome_double(
                    state.genome_drive_sp_pain[canonical_parent_1_slot]
                )
            ), 0.0, 1.0)
        );
        const float child_sp_fatigue = birth::canon_genome_float(
            birth::clamp_double(select_turnover_scalar_base_v1(
                linked_inheritance,
                inheritance_config,
                1,
                donor_ordinals,
                required_donor_draws,
                birth::canon_genome_double(
                    state.genome_drive_sp_fatigue[parent_slot]
                ),
                birth::canon_genome_double(
                    state.genome_drive_sp_fatigue[mate_slot]
                ),
                birth::canon_genome_double(
                    state.genome_drive_sp_fatigue[
                        canonical_parent_0_slot
                    ]
                ),
                birth::canon_genome_double(
                    state.genome_drive_sp_fatigue[
                        canonical_parent_1_slot
                    ]
                )
            ), 0.0, 1.0)
        );
        const float child_sp_fear = birth::canon_genome_float(
            birth::clamp_double(select_turnover_scalar_base_v1(
                linked_inheritance,
                inheritance_config,
                1,
                donor_ordinals,
                required_donor_draws,
                birth::canon_genome_double(
                    state.genome_drive_sp_fear[parent_slot]
                ),
                birth::canon_genome_double(
                    state.genome_drive_sp_fear[mate_slot]
                ),
                birth::canon_genome_double(
                    state.genome_drive_sp_fear[canonical_parent_0_slot]
                ),
                birth::canon_genome_double(
                    state.genome_drive_sp_fear[canonical_parent_1_slot]
                )
            ), 0.0, 1.0)
        );
        const float child_exp_prior_strength = birth::canon_genome_float(
            birth::clamp_double(select_turnover_scalar_base_v1(
                linked_inheritance,
                inheritance_config,
                1,
                donor_ordinals,
                required_donor_draws,
                birth::canon_genome_double(
                    state.genome_exp_prior_strength[parent_slot]
                ),
                birth::canon_genome_double(
                    state.genome_exp_prior_strength[mate_slot]
                ),
                birth::canon_genome_double(
                    state.genome_exp_prior_strength[
                        canonical_parent_0_slot
                    ]
                ),
                birth::canon_genome_double(
                    state.genome_exp_prior_strength[
                        canonical_parent_1_slot
                    ]
                )
            ), 0.0, 1.0)
        );
        const float child_exp_prior_noise = birth::canon_genome_float(
            birth::clamp_double(select_turnover_scalar_base_v1(
                linked_inheritance,
                inheritance_config,
                1,
                donor_ordinals,
                required_donor_draws,
                birth::canon_genome_double(
                    state.genome_exp_prior_noise[parent_slot]
                ),
                birth::canon_genome_double(
                    state.genome_exp_prior_noise[mate_slot]
                ),
                birth::canon_genome_double(
                    state.genome_exp_prior_noise[
                        canonical_parent_0_slot
                    ]
                ),
                birth::canon_genome_double(
                    state.genome_exp_prior_noise[
                        canonical_parent_1_slot
                    ]
                )
            ), 0.0, 0.5)
        );
        const float child_drive_birth_noise = birth::canon_genome_float(
            birth::clamp_double(select_turnover_scalar_base_v1(
                linked_inheritance,
                inheritance_config,
                1,
                donor_ordinals,
                required_donor_draws,
                birth::canon_genome_double(
                    state.genome_drive_birth_noise[parent_slot]
                ),
                birth::canon_genome_double(
                    state.genome_drive_birth_noise[mate_slot]
                ),
                birth::canon_genome_double(
                    state.genome_drive_birth_noise[
                        canonical_parent_0_slot
                    ]
                ),
                birth::canon_genome_double(
                    state.genome_drive_birth_noise[
                        canonical_parent_1_slot
                    ]
                )
            ), 0.0, 0.2)
        );

        // Legacy retains selected-parent policy-family ownership. Linked mode
        // binds the complete represented memory/controller block to its one
        // canonical donor; the downstream policy transaction consumes the
        // published group-4 donor ordinal for controller-row ownership.
        const int64_t policy_family_parent_slot = linked_inheritance
            ? memory_parent_slot
            : parent_slot;
        const bool child_use_gru =
            state.policy_use_gru[policy_family_parent_slot];
        state.genome_hidden_size[child_slot] = child_hidden_size;
        state.genome_memory_dim[child_slot] = child_use_gru
            ? child_memory_dim
            : 0;
        state.policy_mem_dim[child_slot] = child_use_gru
            ? static_cast<int64_t>(child_memory_dim)
            : 0;
        state.policy_use_gru[child_slot] = child_use_gru;
        state.gru_upkeep_cost[child_slot] = child_use_gru
            ? state.gru_upkeep_cost[policy_family_parent_slot]
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
    }

    python_random.python_random_index[0] = rng.index;
    python_random.python_random_has_gauss[0] = rng.has_gauss_next;
    python_random.python_random_gauss_next[0] = rng.gauss_next;
    python_random.python_random_raw_draws[0] = rng.raw_draws;
    python_random.mutation_py_semantic_draws[0] += draws_added;
    outputs.committed_count[0] = birth_count;
    outputs.semantic_draws_added[0] = draws_added;
    outputs.status[0] = kStatusOk;
    if (inheritance_outputs.donor_draws_added != nullptr) {
        inheritance_outputs.donor_draws_added[0] =
            static_cast<int64_t>(required_donor_draws) * birth_count;
    }
    if (inheritance_outputs.status != nullptr) {
        inheritance_outputs.status[0] = kStatusOk;
    }
}

// V1 preserves the caller-supplied exact-pair contract.
__device__ __forceinline__ void
commit_turnover_fixed_mlp_child_scalar_genome_with_inheritance_v1_serial(
    const FlatFixedMlpScalarGenomeState state,
    const FlatTurnoverScalarGenomePlan plan,
    const FlatCpythonRandomState python_random,
    const FlatTurnoverScalarGenomeOutputs outputs,
    const TurnoverScalarGenomeConfig config,
    uint32_t* python_random_trial_words,
    const TurnoverInheritanceConfigV1 inheritance_config,
    const FlatTurnoverInheritanceOutputsV1 inheritance_outputs
) {
    commit_turnover_fixed_mlp_child_scalar_genome_with_inheritance_impl_v2_serial(
        state,
        plan,
        python_random,
        outputs,
        config,
        python_random_trial_words,
        inheritance_config,
        nullptr,
        inheritance_outputs
    );
}

// V2 binds canonical digest-rank authority to the pair selected inside the
// scalar transaction and publishes the resolved pair without additional RNG.
__device__ __forceinline__ void
commit_turnover_fixed_mlp_child_scalar_genome_with_ranked_inheritance_v2_serial(
    const FlatFixedMlpScalarGenomeState state,
    const FlatTurnoverScalarGenomePlan plan,
    const FlatCpythonRandomState python_random,
    const FlatTurnoverScalarGenomeOutputs outputs,
    const TurnoverScalarGenomeConfig config,
    uint32_t* python_random_trial_words,
    const TurnoverRankedInheritanceConfigV2 inheritance_config,
    const FlatTurnoverInheritanceOutputsV1 inheritance_outputs
) {
    const TurnoverInheritanceConfigV1 base_config{
        inheritance_config.mode,
        inheritance_config.donor_draw_budget,
        inheritance_config.mutation_enabled,
        nullptr,
        nullptr,
    };
    commit_turnover_fixed_mlp_child_scalar_genome_with_inheritance_impl_v2_serial(
        state,
        plan,
        python_random,
        outputs,
        config,
        python_random_trial_words,
        base_config,
        &inheritance_config.canonical_parent_rank_authority,
        inheritance_outputs
    );
}

// Exact legacy entry point. Existing callers retain their byte-for-byte scalar
// behavior and consume no linkage donor draws or telemetry storage.
__device__ __forceinline__ void
commit_turnover_fixed_mlp_child_scalar_genome_serial(
    const FlatFixedMlpScalarGenomeState state,
    const FlatTurnoverScalarGenomePlan plan,
    const FlatCpythonRandomState python_random,
    const FlatTurnoverScalarGenomeOutputs outputs,
    const TurnoverScalarGenomeConfig config,
    uint32_t* python_random_trial_words
) {
    commit_turnover_fixed_mlp_child_scalar_genome_with_inheritance_v1_serial(
        state,
        plan,
        python_random,
        outputs,
        config,
        python_random_trial_words,
        {
            inheritance::InheritanceModeV1::legacy_midpoint_v1,
            inheritance::kLegacyAddedDonorDrawCount,
            true,
            nullptr,
            nullptr,
        },
        {
            nullptr,
            nullptr,
            nullptr,
            nullptr,
        }
    );
}

}  // namespace eiel::cuda_architecture_b::persistent_turnover_scalar_genome

#endif  // EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_TURNOVER_SCALAR_GENOME_CUH_
