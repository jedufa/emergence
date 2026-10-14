#include "../../engine/perf/cuda_architecture_b_persistent_turnover_scalar_genome.cuh"

#include <type_traits>

namespace scalar_genome =
    eiel::cuda_architecture_b::persistent_turnover_scalar_genome;

using GenomeState = scalar_genome::FlatFixedMlpScalarGenomeState;
using RandomState = scalar_genome::FlatCpythonRandomState;
using TurnoverPlan = scalar_genome::FlatTurnoverScalarGenomePlan;
using TurnoverOutputs = scalar_genome::FlatTurnoverScalarGenomeOutputs;
using InheritanceConfig = scalar_genome::TurnoverInheritanceConfigV1;
using InheritanceOutputs =
    scalar_genome::FlatTurnoverInheritanceOutputsV1;
using CanonicalParentRankAuthority =
    scalar_genome::TurnoverCanonicalParentRankAuthorityV1;
using RankedInheritanceConfig =
    scalar_genome::TurnoverRankedInheritanceConfigV2;

static_assert(scalar_genome::kCpythonRandomWordCount == 624);
static_assert(scalar_genome::kStatusOk == 0);
static_assert(scalar_genome::kStatusInvalidScalarBirthPlan == 20);
static_assert(scalar_genome::kStatusInvalidPythonRandomState == 21);
static_assert(scalar_genome::kStatusFreshGaussUnsupported == 22);
static_assert(scalar_genome::kStatusInvalidInheritanceMode == 23);
static_assert(scalar_genome::kStatusInvalidInheritanceDonorBudget == 24);
static_assert(scalar_genome::kStatusMissingInheritanceSidecar == 25);
static_assert(scalar_genome::kStatusInvalidCanonicalParentOrder == 26);
static_assert(scalar_genome::kStatusMissingInheritanceOutputs == 27);
static_assert(
    scalar_genome::kStatusInvalidCanonicalParentRankAuthority == 28
);
static_assert(
    scalar_genome::kStatusInheritanceSidecarCapacityExceeded == 29
);

static_assert(std::is_trivially_copyable_v<GenomeState>);
static_assert(std::is_trivially_copyable_v<RandomState>);
static_assert(std::is_trivially_copyable_v<TurnoverPlan>);
static_assert(std::is_trivially_copyable_v<TurnoverOutputs>);
static_assert(std::is_trivially_copyable_v<InheritanceConfig>);
static_assert(std::is_trivially_copyable_v<InheritanceOutputs>);
static_assert(std::is_trivially_copyable_v<CanonicalParentRankAuthority>);
static_assert(std::is_trivially_copyable_v<RankedInheritanceConfig>);
static_assert(
    std::is_same_v<decltype(RandomState::python_random_words), uint32_t*>
);
static_assert(
    std::is_same_v<decltype(RandomState::python_random_has_gauss), bool*>
);
static_assert(
    std::is_same_v<
        decltype(RandomState::python_random_gauss_primary_tape),
        const double*
    >
);
static_assert(
    std::is_same_v<decltype(RandomState::mutation_py_semantic_draws), int64_t*>
);
static_assert(
    std::is_same_v<decltype(TurnoverPlan::parent_pool_slots), const int64_t*>
);
static_assert(
    std::is_same_v<decltype(TurnoverPlan::parent_pool_count), const int64_t*>
);
static_assert(
    std::is_same_v<decltype(TurnoverPlan::child_slots), const int64_t*>
);
static_assert(
    std::is_same_v<decltype(TurnoverOutputs::selected_parent_slots), int64_t*>
);
static_assert(
    std::is_same_v<decltype(TurnoverOutputs::selected_mate_slots), int64_t*>
);
static_assert(
    std::is_same_v<
        decltype(InheritanceConfig::canonical_parent_0_slots),
        const int64_t*
    >
);
static_assert(
    std::is_same_v<
        decltype(InheritanceConfig::canonical_parent_1_slots),
        const int64_t*
    >
);
static_assert(
    std::is_same_v<decltype(InheritanceOutputs::donor_ordinals), uint8_t*>
);
static_assert(
    std::is_same_v<decltype(InheritanceOutputs::donor_statuses), uint8_t*>
);
static_assert(
    std::is_same_v<
        decltype(CanonicalParentRankAuthority::genotype_codes_by_slot),
        const int64_t*
    >
);
static_assert(
    std::is_same_v<
        decltype(CanonicalParentRankAuthority::digest_rank_by_code),
        const int64_t*
    >
);
static_assert(
    std::is_same_v<
        decltype(CanonicalParentRankAuthority::genotype_count),
        int64_t
    >
);
static_assert(
    std::is_same_v<
        decltype(CanonicalParentRankAuthority::canonical_parent_0_slots_out),
        int64_t*
    >
);
static_assert(
    std::is_same_v<
        decltype(CanonicalParentRankAuthority::canonical_parent_1_slots_out),
        int64_t*
    >
);
static_assert(
    std::is_same_v<
        decltype(RankedInheritanceConfig::canonical_parent_rank_authority),
        CanonicalParentRankAuthority
    >
);
static_assert(
    std::is_same_v<
        decltype(RankedInheritanceConfig::mode),
        eiel::inheritance_linkage_v1::InheritanceModeV1
    >
);
static_assert(
    std::is_same_v<
        decltype(RankedInheritanceConfig::donor_draw_budget),
        std::uint32_t
    >
);
static_assert(
    std::is_same_v<decltype(RankedInheritanceConfig::mutation_enabled), bool>
);
static_assert(
    std::is_same_v<decltype(GenomeState::genome_hidden_size), int32_t*>
);
static_assert(
    std::is_same_v<decltype(GenomeState::genome_mutation_rate), float*>
);
static_assert(std::is_same_v<decltype(GenomeState::policy_mem_dim), int64_t*>);
static_assert(std::is_same_v<decltype(GenomeState::policy_use_gru), bool*>);
static_assert(std::is_same_v<decltype(GenomeState::policy_attack_bias), float*>);
static_assert(std::is_same_v<decltype(GenomeState::pred_lr), float*>);

__global__ void compile_persistent_turnover_scalar_genome_probe(
    const scalar_genome::FlatFixedMlpScalarGenomeState state,
    const scalar_genome::FlatTurnoverScalarGenomePlan plan,
    const scalar_genome::FlatCpythonRandomState python_random,
    const scalar_genome::FlatTurnoverScalarGenomeOutputs outputs,
    const scalar_genome::TurnoverScalarGenomeConfig config,
    uint32_t* python_random_trial_words
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    scalar_genome::commit_turnover_fixed_mlp_child_scalar_genome_serial(
        state,
        plan,
        python_random,
        outputs,
        config,
        python_random_trial_words
    );
}

__global__ void compile_persistent_turnover_linked_scalar_genome_probe(
    const scalar_genome::FlatFixedMlpScalarGenomeState state,
    const scalar_genome::FlatTurnoverScalarGenomePlan plan,
    const scalar_genome::FlatCpythonRandomState python_random,
    const scalar_genome::FlatTurnoverScalarGenomeOutputs outputs,
    const scalar_genome::TurnoverScalarGenomeConfig config,
    uint32_t* python_random_trial_words,
    const scalar_genome::TurnoverInheritanceConfigV1 inheritance_config,
    const scalar_genome::FlatTurnoverInheritanceOutputsV1 inheritance_outputs
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    scalar_genome::
        commit_turnover_fixed_mlp_child_scalar_genome_with_inheritance_v1_serial(
            state,
            plan,
            python_random,
            outputs,
            config,
            python_random_trial_words,
            inheritance_config,
            inheritance_outputs
        );
}

__global__ void compile_persistent_turnover_ranked_scalar_genome_probe(
    const scalar_genome::FlatFixedMlpScalarGenomeState state,
    const scalar_genome::FlatTurnoverScalarGenomePlan plan,
    const scalar_genome::FlatCpythonRandomState python_random,
    const scalar_genome::FlatTurnoverScalarGenomeOutputs outputs,
    const scalar_genome::TurnoverScalarGenomeConfig config,
    uint32_t* python_random_trial_words,
    const scalar_genome::TurnoverRankedInheritanceConfigV2 inheritance_config,
    const scalar_genome::FlatTurnoverInheritanceOutputsV1 inheritance_outputs
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    scalar_genome::
        commit_turnover_fixed_mlp_child_scalar_genome_with_ranked_inheritance_v2_serial(
            state,
            plan,
            python_random,
            outputs,
            config,
            python_random_trial_words,
            inheritance_config,
            inheritance_outputs
        );
}
