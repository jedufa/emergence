#include "../../engine/perf/cuda_architecture_b_persistent_mate_birth_plan_v1.cuh"

namespace mate =
    eiel::cuda_architecture_b::persistent_mate_birth_plan_v1;

static_assert(mate::kMateBirthSelectionLegacyAgentIdV1 == 0);
static_assert(mate::kMateBirthSelectionUniformNeutralV1 == 1);

__global__ void compile_neutral_mate_selection_v1(
    const mate::FlatMateBirthPlanStateV1 state,
    const mate::FlatMateBirthPlanBuffersV1 buffers,
    const mate::MateBirthPlanConfigV1 config,
    const mate::MateBirthSelectionConfigV1 selection
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        mate::plan_births_serial_v1(state, buffers, config, selection);
    }
}
