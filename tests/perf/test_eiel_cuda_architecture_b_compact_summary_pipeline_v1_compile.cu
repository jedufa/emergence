#include "cuda_architecture_b_compact_summary_assembler_v1.cuh"
#include "cuda_architecture_b_compact_summary_finalizer_v2.cuh"

#include <type_traits>

namespace assembler =
    eiel::cuda_architecture_b::compact_summary_assembler_v1;
namespace finalizer =
    eiel::cuda_architecture_b::compact_summary_finalizer_v2;

static_assert(std::is_trivially_copyable_v<
              assembler::CompactSummaryBoundarySourceV1>);
static_assert(std::is_trivially_copyable_v<
              assembler::CompactSummaryScratchRowV1>);
static_assert(finalizer::kProducerStatusSuccess == 0);
static_assert(finalizer::kProducerStatusTerminalStickyTail == 16);

int main() {
    return 0;
}
