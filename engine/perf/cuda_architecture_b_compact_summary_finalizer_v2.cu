#include "cuda_architecture_b_compact_summary_finalizer_v2.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace eiel::cuda_architecture_b::compact_summary_finalizer_v2 {

__global__ void compact_summary_finalizer_kernel_v2(
    const abi::CompactSummaryViewV2 scratch,
    const abi::CompactSummaryViewV2 output,
    std::int32_t* const producer_status,
    const std::int64_t generation
) {
    const std::int64_t cell =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cell >= scratch.cell_count) {
        return;
    }
    finalize_cell(scratch, output, producer_status, generation, cell);
}

void launch_compact_summary_finalizer_v2(
    const abi::CompactSummaryViewV2 scratch,
    const abi::CompactSummaryViewV2 output,
    std::int32_t* const producer_status,
    const std::int64_t generation,
    cudaStream_t stream
) {
    const unsigned int blocks = static_cast<unsigned int>(
        (scratch.cell_count + kThreadsPerBlockV2 - 1) / kThreadsPerBlockV2
    );
    compact_summary_finalizer_kernel_v2<<<
        blocks,
        kThreadsPerBlockV2,
        0,
        stream
    >>>(scratch, output, producer_status, generation);
}

KernelResourceReceiptV2 query_compact_summary_finalizer_v2_resources() {
    cudaFuncAttributes attributes{};
    cudaError_t error = cudaFuncGetAttributes(
        &attributes,
        compact_summary_finalizer_kernel_v2
    );
    if (error != cudaSuccess) {
        return {-1, -1, -1, -1, -1, -1, -1, -1};
    }
    std::int32_t occupancy = -1;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy,
        compact_summary_finalizer_kernel_v2,
        kThreadsPerBlockV2,
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

}  // namespace eiel::cuda_architecture_b::compact_summary_finalizer_v2
