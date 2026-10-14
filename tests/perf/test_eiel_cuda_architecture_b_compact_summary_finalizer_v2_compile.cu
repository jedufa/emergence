#include "../../engine/perf/cuda_architecture_b_compact_summary_finalizer_v2.cuh"

#include <cstdint>
#include <type_traits>

namespace finalizer =
    eiel::cuda_architecture_b::compact_summary_finalizer_v2;
namespace summary_abi = eiel::cuda_architecture_b::compact_summary_v2;

static_assert(finalizer::kProducerStatusSuccess == 0);
static_assert(finalizer::kProducerStatusScratchNotReady == 1);
static_assert(finalizer::kProducerStatusReservedDefinedMask == 2);
static_assert(finalizer::kProducerStatusStatusCodeRange == 3);
static_assert(finalizer::kProducerStatusNullableStatusCode == 4);
static_assert(finalizer::kProducerStatusPatchMask == 5);
static_assert(finalizer::kProducerStatusIntegerRange == 6);
static_assert(finalizer::kProducerStatusFloatNotFinite == 7);
static_assert(finalizer::kProducerStatusFloatRange == 8);
static_assert(finalizer::kProducerStatusDigestBytes == 9);
static_assert(finalizer::kProducerStatusTextUndefinedRecord == 10);
static_assert(finalizer::kProducerStatusTextLength == 11);
static_assert(finalizer::kProducerStatusTextUtf8 == 12);
static_assert(finalizer::kProducerStatusTextTail == 13);
static_assert(finalizer::kProducerStatusTerminalRecoverable == 14);
static_assert(finalizer::kProducerStatusTerminalFailure == 15);
static_assert(finalizer::kProducerStatusTerminalStickyTail == 16);
static_assert(finalizer::kThreadsPerBlockV2 == 128);
static_assert(finalizer::kPatchChildDefinedMaskV2 == 0x00000000FFFC0000ULL);
static_assert(std::is_standard_layout_v<finalizer::KernelResourceReceiptV2>);
static_assert(std::is_trivially_copyable_v<finalizer::KernelResourceReceiptV2>);

__global__ void compile_compact_summary_finalizer_v2_probe(
    const summary_abi::CompactSummaryViewV2 scratch,
    const summary_abi::CompactSummaryViewV2 output,
    std::int32_t* const producer_status,
    const std::int64_t generation
) {
    const std::int64_t cell =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cell >= scratch.cell_count) {
        return;
    }
    finalizer::finalize_cell(
        scratch,
        output,
        producer_status,
        generation,
        cell
    );
}
