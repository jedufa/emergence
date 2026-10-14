#include "cuda_architecture_b_compact_summary_checkpoint_hash_v1.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace eiel::cuda_architecture_b::compact_summary_checkpoint_hash_v1 {

namespace {

__global__ void checkpoint_tensor_hash_snapshot_kernel_v1(
    const TensorDigestPointerBatchV1 pointer_batch,
    const SnapshotOutputsV1 outputs
) {
    const std::int64_t linear =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total =
        outputs.cell_count * static_cast<std::int64_t>(pointer_batch.count);
    if (linear >= total || pointer_batch.count <= 0) {
        return;
    }
    const std::int64_t cell = linear / pointer_batch.count;
    const std::int32_t batch_field =
        static_cast<std::int32_t>(linear % pointer_batch.count);
    const TensorDigestPointerEntryV1 entry = pointer_batch.entries[batch_field];
    if (
        entry.descriptor_index < 0
        || entry.descriptor_index >= outputs.descriptor_field_count
    ) {
        return;
    }

    const std::int64_t status_offset =
        (outputs.generation_index * outputs.cell_count + cell)
            * outputs.descriptor_field_count
        + entry.descriptor_index;
    outputs.valid[status_offset] = 0U;

    std::uint8_t raw_digest[kDigestBytesV1];
    std::uint8_t canonical_digest[kDigestBytesV1];
#pragma unroll
    for (std::int32_t index = 0; index < kDigestBytesV1; ++index) {
        raw_digest[index] = 0U;
        canonical_digest[index] = 0U;
    }
    const SnapshotStatusV1 status = hash_entry_for_cell(
        entry,
        outputs,
        cell,
        raw_digest,
        canonical_digest
    );
    outputs.status[status_offset] = static_cast<std::int32_t>(status);

    if (
        entry.raw_output_index >= 0
        && entry.raw_output_index < outputs.output_field_count
    ) {
        const std::int64_t raw_offset =
            ((outputs.generation_index * outputs.cell_count + cell)
                 * outputs.output_field_count
             + entry.raw_output_index)
            * kDigestBytesV1;
        detail::zero_digest(outputs.raw_digests + raw_offset);
        if (status == kSnapshotSuccessV1) {
            detail::copy_digest(raw_digest, outputs.raw_digests + raw_offset);
        }
    }
    if (
        entry.canonical_output_index >= 0
        && entry.canonical_output_index < outputs.output_field_count
    ) {
        const std::int64_t canonical_offset =
            ((outputs.generation_index * outputs.cell_count + cell)
                 * outputs.output_field_count
             + entry.canonical_output_index)
            * kDigestBytesV1;
        detail::zero_digest(outputs.canonical_digests + canonical_offset);
        if (status == kSnapshotSuccessV1) {
            detail::copy_digest(
                canonical_digest,
                outputs.canonical_digests + canonical_offset
            );
        }
    }
    if (status == kSnapshotSuccessV1) {
        // Validity is written last and publishes both completed digest slots.
        outputs.valid[status_offset] = 1U;
    }
}

}  // namespace

void launch_checkpoint_tensor_hash_snapshot_v1(
    const TensorDigestPointerBatchV1 pointer_batch,
    const SnapshotOutputsV1 outputs,
    cudaStream_t stream
) {
    const std::int64_t total =
        outputs.cell_count * static_cast<std::int64_t>(pointer_batch.count);
    const unsigned int blocks = static_cast<unsigned int>(
        (total + kThreadsPerBlockV1 - 1) / kThreadsPerBlockV1
    );
    checkpoint_tensor_hash_snapshot_kernel_v1<<<
        blocks,
        kThreadsPerBlockV1,
        0,
        stream
    >>>(pointer_batch, outputs);
}

KernelResourceReceiptV1 query_checkpoint_tensor_hash_snapshot_v1_resources() {
    cudaFuncAttributes attributes{};
    cudaError_t error = cudaFuncGetAttributes(
        &attributes,
        checkpoint_tensor_hash_snapshot_kernel_v1
    );
    if (error != cudaSuccess) {
        return {-1, -1, -1, -1, -1, -1, -1, -1};
    }
    std::int32_t occupancy = -1;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy,
        checkpoint_tensor_hash_snapshot_kernel_v1,
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

}  // namespace eiel::cuda_architecture_b::compact_summary_checkpoint_hash_v1
