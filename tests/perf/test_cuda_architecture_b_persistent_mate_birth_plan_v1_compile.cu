#include "../../engine/perf/cuda_architecture_b_persistent_mate_birth_plan_v1.cuh"

namespace mate_plan =
    eiel::cuda_architecture_b::persistent_mate_birth_plan_v1;

static_assert(mate_plan::kMateSpawnCandidateCount == 8);
static_assert(mate_plan::kReasonNoMate == 4);
static_assert(mate_plan::kReasonCount == 12);

__global__ void compile_persistent_mate_birth_plan_v1_probe(
    const mate_plan::FlatMateBirthPlanStateV1 state,
    const mate_plan::FlatMateBirthPlanBuffersV1 buffers,
    const mate_plan::MateBirthPlanConfigV1 config
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        mate_plan::plan_births_serial_v1(state, buffers, config);
    }
}
