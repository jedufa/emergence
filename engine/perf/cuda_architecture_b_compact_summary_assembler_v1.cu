#include "cuda_architecture_b_compact_summary_assembler_v1.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace assembler =
    eiel::cuda_architecture_b::compact_summary_assembler_v1;
namespace abi = eiel::cuda_architecture_b::compact_summary_v2;

namespace {

__global__ void compact_summary_assembler_v1_kernel(
    const abi::DefinedMaskV1* source_defined_masks,
    const std::int32_t* source_status_codes,
    const std::int64_t* source_integer_values,
    const double* source_float_values,
    const std::uint8_t* source_digest_values,
    const std::uint8_t* source_text_values,
    std::uint8_t* scratch_ready,
    std::uint8_t* scratch_validity,
    abi::DefinedMaskV1* scratch_defined_masks,
    std::int32_t* scratch_status_codes,
    std::int64_t* scratch_integer_values,
    double* scratch_float_values,
    std::uint8_t* scratch_digest_values,
    std::uint8_t* scratch_text_values,
    std::int32_t* assemble_status,
    const std::int64_t row_count
) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (row >= row_count) {
        return;
    }
    const std::int32_t* const status =
        source_status_codes + row * abi::kStatusFieldCountV2;
    const assembler::CompactSummaryBoundarySourceV1 source = {
        assembler::kBoundarySourceVersionV1,
        source_defined_masks[row],
        status[abi::kStatusSummaryStatus],
        status[abi::kStatusWorld],
        status[abi::kStatusTurnoverStatus],
        status[abi::kStatusRngAuthorityComplete],
        status[abi::kStatusRngExportNonConsuming],
        status[abi::kStatusTerminalStatus],
        status[abi::kStatusTerminalRecoverableErrorStatus],
        status[abi::kStatusTerminalStickyFailure],
        status[abi::kStatusCheckpointDue],
        status[abi::kStatusLineageMetricsLineageCollapseWarning],
        status[abi::kStatusTerminalRecoverableErrorCode],
        status[abi::kStatusTerminalFailureStage],
        source_integer_values + row * abi::kIntegerFieldCountV2,
        source_float_values + row * abi::kFloatFieldCountV2,
        source_digest_values
            + row * abi::kDigestFieldCountV2 * abi::kDigestBytesV2,
        source_text_values + row * abi::kTextRecordBytesV2,
    };
    const assembler::CompactSummaryScratchRowV1 scratch = {
        scratch_ready + row,
        scratch_validity + row,
        scratch_defined_masks + row,
        scratch_status_codes + row * abi::kStatusFieldCountV2,
        scratch_integer_values + row * abi::kIntegerFieldCountV2,
        scratch_float_values + row * abi::kFloatFieldCountV2,
        scratch_digest_values
            + row * abi::kDigestFieldCountV2 * abi::kDigestBytesV2,
        scratch_text_values + row * abi::kTextRecordBytesV2,
    };
    assemble_status[row] = static_cast<std::int32_t>(
        assembler::assemble_compact_summary_scratch_row(source, scratch)
    );
}

}  // namespace

void eiel_cuda_architecture_b_compact_summary_assembler_v1_launch(
    const abi::DefinedMaskV1* source_defined_masks,
    const std::int32_t* source_status_codes,
    const std::int64_t* source_integer_values,
    const double* source_float_values,
    const std::uint8_t* source_digest_values,
    const std::uint8_t* source_text_values,
    std::uint8_t* scratch_ready,
    std::uint8_t* scratch_validity,
    abi::DefinedMaskV1* scratch_defined_masks,
    std::int32_t* scratch_status_codes,
    std::int64_t* scratch_integer_values,
    double* scratch_float_values,
    std::uint8_t* scratch_digest_values,
    std::uint8_t* scratch_text_values,
    std::int32_t* assemble_status,
    const std::int64_t row_count,
    cudaStream_t stream
) {
    constexpr int threads = 128;
    const int blocks = static_cast<int>((row_count + threads - 1) / threads);
    compact_summary_assembler_v1_kernel<<<blocks, threads, 0, stream>>>(
        source_defined_masks,
        source_status_codes,
        source_integer_values,
        source_float_values,
        source_digest_values,
        source_text_values,
        scratch_ready,
        scratch_validity,
        scratch_defined_masks,
        scratch_status_codes,
        scratch_integer_values,
        scratch_float_values,
        scratch_digest_values,
        scratch_text_values,
        assemble_status,
        row_count
    );
}

cudaError_t eiel_cuda_architecture_b_compact_summary_assembler_v1_attributes(
    cudaFuncAttributes* attributes
) {
    return cudaFuncGetAttributes(
        attributes, compact_summary_assembler_v1_kernel
    );
}
