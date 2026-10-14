#include "../../engine/perf/cuda_architecture_b_persistent_birth_scalar_genome.cuh"

namespace scalar =
    eiel::cuda_architecture_b::persistent_birth_scalar_genome;

__global__ void compile_stackless_birth_scalar_probe(
    const scalar::FlatFixedMlpScalarGenomeState state,
    const scalar::FlatOrdinaryScalarGenomePlan plan,
    const scalar::FlatCpythonRandomState python_random,
    const scalar::FlatOrdinaryScalarGenomeOutputs outputs,
    const scalar::OrdinaryScalarGenomeConfig config,
    uint32_t* trial_words
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    scalar::commit_ordinary_fixed_mlp_child_scalar_genome_with_trial_scratch_serial(
        state,
        plan,
        python_random,
        outputs,
        config,
        trial_words
    );
}
