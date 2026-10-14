#include "cuda_architecture_b_compact_summary_reductions_v1.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace eiel::cuda_architecture_b::compact_summary_reductions_v1 {

namespace {

__global__ void compact_summary_reductions_kernel_v1(
    const ReductionInputsV1 inputs,
    const ReductionOutputViewV1 output
) {
    const std::int64_t cell =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cell >= inputs.cell_count) {
        return;
    }
    const std::int64_t output_row = generation_major_row_index_v1(
        inputs.generation_index, inputs.cell_count, cell
    );
    ReductionResultV1 result = reduce_cell(inputs, cell);
    const bool valid = result.status == kReductionSuccessV1;
    result.valid = 0;
    output.rows[output_row] = result;
    if (valid) {
        // Validity is the publication fence for the completed POD row.
        __threadfence();
        output.rows[output_row].valid = 1;
    }
}

}  // namespace

ReductionLaunchStatusV1 launch_compact_summary_reductions_v1(
    const ReductionInputsV1 inputs,
    const ReductionOutputViewV1 output,
    cudaStream_t stream
) {
    if (
        inputs.generation_summary_scalar.width
            != kGenerationSummaryScalarWidthV1
        || inputs.generation_summary_scalar.cell_count != inputs.cell_count
        || !valid_generation_shape_v1(
            inputs.generation_index,
            inputs.generation_summary_scalar.generation_count,
            inputs.cell_count
        )
        || inputs.generation_summary_scalar.generation_count
                * inputs.cell_count
            > kInt64MaximumV1 / kGenerationSummaryScalarWidthV1
    ) {
        return kReductionLaunchInvalidRequestShapeV1;
    }
    if (
        output.rows == nullptr
        || output.generation_count
            != inputs.generation_summary_scalar.generation_count
        || output.cell_count != inputs.cell_count
        || !valid_generation_shape_v1(
            inputs.generation_index,
            output.generation_count,
            output.cell_count
        )
    ) {
        return kReductionLaunchInvalidOutputShapeV1;
    }
    const unsigned int blocks = static_cast<unsigned int>(
        (inputs.cell_count + kThreadsPerBlockV1 - 1) / kThreadsPerBlockV1
    );
    compact_summary_reductions_kernel_v1<<<
        blocks,
        kThreadsPerBlockV1,
        0,
        stream
    >>>(inputs, output);
    return kReductionLaunchSuccessV1;
}

KernelResourceReceiptV1 query_compact_summary_reductions_v1_resources() {
    cudaFuncAttributes attributes{};
    cudaError_t error = cudaFuncGetAttributes(
        &attributes,
        compact_summary_reductions_kernel_v1
    );
    if (error != cudaSuccess) {
        return {-1, -1, -1, -1, -1, -1, -1, -1};
    }
    std::int32_t occupancy = -1;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy,
        compact_summary_reductions_kernel_v1,
        kThreadsPerBlockV1,
        0
    );
    if (error != cudaSuccess) {
        occupancy = -1;
    }
    return {
        attributes.binaryVersion,
        attributes.ptxVersion,
        attributes.maxThreadsPerBlock,
        attributes.numRegs,
        occupancy,
        static_cast<std::int64_t>(attributes.constSizeBytes),
        static_cast<std::int64_t>(attributes.localSizeBytes),
        static_cast<std::int64_t>(attributes.sharedSizeBytes),
    };
}

}  // namespace eiel::cuda_architecture_b::compact_summary_reductions_v1
