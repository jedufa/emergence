#include "../../engine/perf/cuda_architecture_b_flattened_mate_birth_scalar_genome_v3.cuh"

#include <type_traits>

namespace mate_scalar =
    eiel::cuda_architecture_b::flattened_mate_birth_scalar_genome_v3;

using Plan = mate_scalar::FlatMateOrdinaryScalarGenomePlan;
using State = mate_scalar::FlatAuthoritativeFixedMlpScalarGenomeState;
using RandomState = mate_scalar::FlatAuthoritativeCpythonRandomState;
using Scratch = mate_scalar::FlatBirthOrderedScalarGenomeScratch;
using Inheritance = mate_scalar::FlatMateOrdinaryInheritancePlanV1;

static_assert(mate_scalar::kCpythonRandomWordCount == 624);
static_assert(std::is_trivially_copyable_v<Plan>);
static_assert(std::is_trivially_copyable_v<State>);
static_assert(std::is_trivially_copyable_v<RandomState>);
static_assert(std::is_trivially_copyable_v<Scratch>);
static_assert(std::is_trivially_copyable_v<Inheritance>);
static_assert(std::is_same_v<decltype(Plan::parent_slots), const int64_t*>);
static_assert(std::is_same_v<decltype(Plan::mate_slots), const int64_t*>);
static_assert(std::is_same_v<decltype(Plan::child_slots), const int64_t*>);

__global__ void compile_flattened_mate_birth_scalar_genome_v3_probe(
    const State state,
    const Plan plan,
    const RandomState python_random,
    const Scratch scratch,
    const mate_scalar::OrdinaryScalarGenomeConfig config
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    mate_scalar::
        stage_ordinary_fixed_mlp_child_scalar_genomes_with_mates_external_scratch_v3(
            state,
            plan,
            python_random,
            scratch,
            config
        );
}

__global__ void compile_flattened_mate_birth_scalar_genome_v4_probe(
    const State state,
    const Plan plan,
    const RandomState python_random,
    const Scratch scratch,
    const mate_scalar::OrdinaryScalarGenomeConfig config,
    const Inheritance inheritance
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    mate_scalar::
        stage_ordinary_fixed_mlp_child_scalar_genomes_with_mates_external_scratch_v4(
            state,
            plan,
            python_random,
            scratch,
            config,
            inheritance
        );
}
