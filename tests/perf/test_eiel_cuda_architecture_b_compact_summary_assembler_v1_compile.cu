#include "../../engine/perf/cuda_architecture_b_compact_summary_assembler_v1.cuh"

#include <cstdint>
#include <type_traits>

namespace assembler =
    eiel::cuda_architecture_b::compact_summary_assembler_v1;
namespace abi = eiel::cuda_architecture_b::compact_summary_v2;

static_assert(assembler::kBoundarySourceVersionV1 == 1);
static_assert(assembler::kAssembleSuccess == 0);
static_assert(assembler::kAssembleTextInvalid == 14);
static_assert(
    std::is_standard_layout_v<assembler::CompactSummaryBoundarySourceV1>
);
static_assert(
    std::is_trivially_copyable_v<assembler::CompactSummaryBoundarySourceV1>
);
static_assert(
    std::is_standard_layout_v<assembler::CompactSummaryScratchRowV1>
);
static_assert(
    std::is_trivially_copyable_v<assembler::CompactSummaryScratchRowV1>
);

__global__ void compile_compact_summary_assembler_v1(
    const assembler::CompactSummaryBoundarySourceV1 source,
    const assembler::CompactSummaryScratchRowV1 scratch,
    std::int32_t* status
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        status[0] = static_cast<std::int32_t>(
            assembler::assemble_compact_summary_scratch_row(source, scratch)
        );
    }
}
