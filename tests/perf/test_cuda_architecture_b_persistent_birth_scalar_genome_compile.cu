#include "../../engine/perf/cuda_architecture_b_persistent_birth_scalar_genome.cuh"

#include <type_traits>

namespace scalar_genome =
    eiel::cuda_architecture_b::persistent_birth_scalar_genome;

using GenomeState = scalar_genome::FlatFixedMlpScalarGenomeState;
using RandomState = scalar_genome::FlatCpythonRandomState;

static_assert(scalar_genome::kCpythonRandomWordCount == 624);
static_assert(scalar_genome::kStatusOk == 0);
static_assert(scalar_genome::kStatusInvalidScalarBirthPlan == 20);
static_assert(scalar_genome::kStatusInvalidPythonRandomState == 21);
static_assert(scalar_genome::kStatusFreshGaussUnsupported == 22);

static_assert(std::is_trivially_copyable_v<GenomeState>);
static_assert(std::is_trivially_copyable_v<RandomState>);
static_assert(
    std::is_same_v<decltype(RandomState::python_random_words), uint32_t*>
);
static_assert(
    std::is_same_v<
        decltype(RandomState::python_random_gauss_primary_tape),
        const double*
    >
);
static_assert(
    std::is_same_v<
        decltype(RandomState::mutation_py_semantic_draws),
        int64_t*
    >
);
static_assert(
    std::is_same_v<decltype(GenomeState::genome_hidden_size), int32_t*>
);
static_assert(
    std::is_same_v<decltype(GenomeState::policy_mem_dim), int64_t*>
);
static_assert(
    std::is_same_v<decltype(GenomeState::policy_use_gru), bool*>
);
static_assert(
    std::is_same_v<decltype(GenomeState::genome_mutation_rate), float*>
);
static_assert(
    std::is_same_v<decltype(GenomeState::policy_attack_bias), float*>
);
static_assert(
    std::is_same_v<decltype(GenomeState::pred_lr), float*>
);

__global__ void compile_persistent_birth_scalar_genome_probe(
    const scalar_genome::FlatFixedMlpScalarGenomeState state,
    const scalar_genome::FlatOrdinaryScalarGenomePlan plan,
    const scalar_genome::FlatCpythonRandomState python_random,
    const scalar_genome::FlatOrdinaryScalarGenomeOutputs outputs,
    const scalar_genome::OrdinaryScalarGenomeConfig config
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    scalar_genome::commit_ordinary_fixed_mlp_child_scalar_genome_serial(
        state,
        plan,
        python_random,
        outputs,
        config
    );
}
