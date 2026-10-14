#include "../../engine/perf/cuda_architecture_b_compact_summary_reductions_v1.cuh"
#include "../../engine/perf/cuda_architecture_b_flattened_controller_v1.hpp"
#include "../../engine/perf/cuda_architecture_b_persistent_birth_plan.cuh"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace reductions =
    eiel::cuda_architecture_b::compact_summary_reductions_v1;
namespace controller =
    eiel::cuda_architecture_b::flattened_controller_v1;
namespace birth_plan =
    eiel::cuda_architecture_b::persistent_birth_plan;

static_assert(reductions::kEventTypeCountV1 == 4);
static_assert(reductions::kBirthCommittedEventTypeV1 == 1);
static_assert(reductions::kMutationEventTypeV1 == 2);
static_assert(reductions::kBirthFailedEventTypeV1 == 3);
static_assert(reductions::kDeathCommittedEventTypeV1 == 4);
static_assert(
    reductions::kBirthPlanAcceptedReasonCodeV1 == birth_plan::kReasonAccepted
);
static_assert(
    reductions::kBirthPlanNoSpaceReasonCodeV1 == birth_plan::kReasonNoSpace
);
static_assert(
    reductions::kBirthPlanBufferReasonCodeV1 == birth_plan::kReasonBuffer
);
static_assert(
    reductions::kBirthPlanNoFreeSlotReasonCodeV1 == birth_plan::kReasonNoFreeSlot
);
static_assert(
    reductions::kBirthPlanConceptionReasonCodeV1 == birth_plan::kReasonConception
);
static_assert(
    reductions::kBirthPlanGestationReasonCodeV1 == birth_plan::kReasonGestation
);
static_assert(
    reductions::kBirthPlanHealthReasonCodeV1 == birth_plan::kReasonHealth
);
static_assert(
    reductions::kBirthPlanCooldownReasonCodeV1 == birth_plan::kReasonCooldown
);
static_assert(
    reductions::kBirthPlanNoEnergyReasonCodeV1 == birth_plan::kReasonNoEnergy
);
static_assert(
    reductions::kGenerationSummaryGenerationColumnV1
    == controller::kGenerationSummaryGeneration
);
static_assert(
    reductions::kGenerationSummaryCursorStartColumnV1
    == controller::kGenerationSummaryEventCursorStart
);
static_assert(
    reductions::kGenerationSummaryCursorPreTurnoverColumnV1
    == controller::kGenerationSummaryEventCursorPreTurnover
);
static_assert(
    reductions::kGenerationSummaryCursorEndColumnV1
    == controller::kGenerationSummaryEventCursorEnd
);
static_assert(
    reductions::kGenerationSummaryScalarWidthV1
    == controller::kGenerationSummaryScalarWidthV1
);
static_assert(std::is_standard_layout_v<reductions::ReductionInputsV1>);
static_assert(std::is_trivially_copyable_v<reductions::ReductionInputsV1>);
static_assert(std::is_standard_layout_v<reductions::ReductionOutputViewV1>);
static_assert(std::is_trivially_copyable_v<reductions::ReductionOutputViewV1>);
static_assert(std::is_standard_layout_v<reductions::ReductionResultV1>);
static_assert(std::is_trivially_copyable_v<reductions::ReductionResultV1>);
static_assert(sizeof(reductions::ReductionResultV1) == 456);

__global__ void compile_compact_summary_reductions_v1(
    const reductions::ReductionInputsV1 inputs,
    reductions::ReductionResultV1* output
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    output[0] = reductions::reduce_cell(inputs, 0);
}
