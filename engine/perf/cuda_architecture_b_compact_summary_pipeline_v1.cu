#include "cuda_architecture_b_compact_summary_assembler_v1.cuh"
#include "cuda_architecture_b_compact_summary_finalizer_v2.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace abi = eiel::cuda_architecture_b::compact_summary_v2;
namespace finalizer =
    eiel::cuda_architecture_b::compact_summary_finalizer_v2;

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
    std::int64_t row_count,
    cudaStream_t stream
);

namespace {

__global__ void compact_summary_pipeline_status_gate_v1(
    const std::int32_t* const assemble_status,
    std::int32_t* const producer_status,
    const std::int64_t row_count
) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (row >= row_count) {
        return;
    }
    if (producer_status[row] == finalizer::kProducerStatusSuccess) {
        producer_status[row] = assemble_status[row];
    }
}

}  // namespace

void eiel_cuda_architecture_b_compact_summary_pipeline_v1_launch(
    const abi::DefinedMaskV1* const source_defined_masks,
    const std::int32_t* const source_status_codes,
    const std::int64_t* const source_integer_values,
    const double* const source_float_values,
    const std::uint8_t* const source_digest_values,
    const std::uint8_t* const source_text_values,
    std::uint8_t* const scratch_ready,
    std::uint8_t* const scratch_validity,
    abi::DefinedMaskV1* const scratch_defined_masks,
    std::int32_t* const scratch_status_codes,
    std::int64_t* const scratch_integer_values,
    double* const scratch_float_values,
    std::uint8_t* const scratch_digest_values,
    std::uint8_t* const scratch_text_values,
    const abi::CompactSummaryViewV2 scratch_view,
    const abi::CompactSummaryViewV2 output_view,
    std::int32_t* const assemble_status,
    std::int32_t* const producer_status,
    const std::int64_t row_count,
    const std::int64_t generation,
    cudaStream_t stream
) {
    eiel_cuda_architecture_b_compact_summary_assembler_v1_launch(
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
        row_count,
        stream
    );

    constexpr std::int32_t threads = 128;
    const auto blocks = static_cast<std::int32_t>(
        (row_count + threads - 1) / threads
    );
    compact_summary_pipeline_status_gate_v1<<<blocks, threads, 0, stream>>>(
        assemble_status,
        producer_status,
        row_count
    );

    finalizer::launch_compact_summary_finalizer_v2(
        scratch_view,
        output_view,
        producer_status,
        generation,
        stream
    );
}

cudaError_t
eiel_cuda_architecture_b_compact_summary_pipeline_v1_gate_attributes(
    cudaFuncAttributes* const attributes
) {
    return cudaFuncGetAttributes(
        attributes,
        compact_summary_pipeline_status_gate_v1
    );
}
