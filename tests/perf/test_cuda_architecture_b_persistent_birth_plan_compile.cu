#include "../../engine/perf/cuda_architecture_b_persistent_birth_plan.cuh"

namespace birth_plan = eiel::cuda_architecture_b::persistent_birth_plan;

static_assert(birth_plan::kReasonAccepted == 0);
static_assert(birth_plan::kReasonNoSpace == 1);
static_assert(birth_plan::kReasonNoFreeSlot == 3);
static_assert(birth_plan::kReasonConception == 5);
static_assert(birth_plan::kReasonNoEnergy == 9);
static_assert(birth_plan::kReasonCount == 10);

__global__ void compile_persistent_birth_plan_probe(
    const birth_plan::FlatOrdinaryBirthPlanState state,
    const birth_plan::FlatOrdinaryBirthPlanBuffers buffers,
    const birth_plan::OrdinaryBirthPlanConfig config
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    birth_plan::plan_ordinary_births_serial(state, buffers, config);
}
