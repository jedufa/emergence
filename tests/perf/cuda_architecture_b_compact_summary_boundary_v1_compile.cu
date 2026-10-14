#include "../../engine/perf/cuda_architecture_b_compact_summary_boundary_v1.cu"

#include <type_traits>

namespace boundary =
    eiel::cuda_architecture_b::compact_summary_boundary_v1;

static_assert(boundary::kRequiredHostCanonicalDigestMaskV1 == 0x7FU);
static_assert(boundary::kBoundaryLaunchSuccessV1 == 0);
static_assert(boundary::kBoundaryHostDigestFinalizationRequiredV1 == 1);
static_assert(boundary::kBoundaryProducerReductionFailureV1 > 16);
static_assert(boundary::kBoundaryProducerCheckpointHashFailureV1 > 16);
static_assert(std::is_trivially_copyable_v<
              boundary::CompactSummaryBoundaryRequestV1>);
static_assert(std::is_trivially_copyable_v<
              boundary::BoundaryLaunchReceiptV1>);

int main() {
    return 0;
}
