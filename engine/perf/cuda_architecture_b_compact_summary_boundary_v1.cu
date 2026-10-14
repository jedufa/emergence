#include "cuda_architecture_b_compact_summary_boundary_v1.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace boundary_abi =
    eiel::cuda_architecture_b::compact_summary_v2;

void eiel_cuda_architecture_b_compact_summary_pipeline_v1_launch(
    const boundary_abi::DefinedMaskV1* source_defined_masks,
    const std::int32_t* source_status_codes,
    const std::int64_t* source_integer_values,
    const double* source_float_values,
    const std::uint8_t* source_digest_values,
    const std::uint8_t* source_text_values,
    std::uint8_t* scratch_ready,
    std::uint8_t* scratch_validity,
    boundary_abi::DefinedMaskV1* scratch_defined_masks,
    std::int32_t* scratch_status_codes,
    std::int64_t* scratch_integer_values,
    double* scratch_float_values,
    std::uint8_t* scratch_digest_values,
    std::uint8_t* scratch_text_values,
    boundary_abi::CompactSummaryViewV2 scratch_view,
    boundary_abi::CompactSummaryViewV2 output_view,
    std::int32_t* assemble_status,
    std::int32_t* producer_status,
    std::int64_t row_count,
    std::int64_t generation,
    cudaStream_t stream
);

namespace eiel::cuda_architecture_b::compact_summary_boundary_v1 {

namespace {

__global__ void compact_summary_boundary_validity_reset_v1(
    const abi::CompactSummaryViewV2 output,
    std::int32_t* const producer_status,
    const std::int64_t generation,
    const std::int32_t forced_producer_status
) {
    const std::int64_t cell = static_cast<std::int64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (cell >= output.cell_count) {
        return;
    }
    abi::row_pointer(output.validity, generation, cell)[0] = 0U;
    if (
        forced_producer_status != finalizer::kProducerStatusSuccess
        && producer_status != nullptr
        && producer_status[cell] == finalizer::kProducerStatusSuccess
    ) {
        producer_status[cell] = forced_producer_status;
    }
}

__global__ void compact_summary_boundary_prerequisite_gate_v1(
    const reductions::ReductionResultV1* const reduction_output,
    const checkpoint_hash::SnapshotOutputsV1 checkpoint_outputs,
    std::int32_t* const producer_status,
    const bool host_digests_complete
) {
    const std::int64_t cell = static_cast<std::int64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (cell >= checkpoint_outputs.cell_count) {
        return;
    }
    if (producer_status[cell] != finalizer::kProducerStatusSuccess) {
        return;
    }
    if (
        reduction_output[cell].valid != 1U
        || reduction_output[cell].status != reductions::kReductionSuccessV1
    ) {
        producer_status[cell] = kBoundaryProducerReductionFailureV1;
        return;
    }
    const std::int64_t status_base =
        (checkpoint_outputs.generation_index * checkpoint_outputs.cell_count
         + cell)
        * checkpoint_outputs.descriptor_field_count;
    for (
        std::int64_t field = 0;
        field < checkpoint_outputs.descriptor_field_count;
        ++field
    ) {
        const std::int64_t offset = status_base + field;
        if (
            checkpoint_outputs.valid[offset] != 1U
            || checkpoint_outputs.status[offset]
                != checkpoint_hash::kSnapshotSuccessV1
        ) {
            producer_status[cell] =
                kBoundaryProducerCheckpointHashFailureV1;
            return;
        }
    }
    if (!host_digests_complete) {
        producer_status[cell] =
            kBoundaryProducerHostDigestFinalizationRequiredV1;
    }
}

template <typename T>
bool valid_plane(
    const abi::StridedPlaneViewV2<T>& plane,
    const std::int64_t row_width,
    const std::int64_t generation_count,
    const std::int64_t cell_count
) noexcept {
    if (
        plane.data == nullptr
        || row_width <= 0
        || plane.cell_stride < row_width
        || plane.generation_stride < 0
    ) {
        return false;
    }
    if (
        generation_count > 1
        && plane.generation_stride < cell_count * plane.cell_stride
    ) {
        return false;
    }
    return true;
}

bool valid_output_target(
    const CompactSummaryBoundaryRequestV1& request
) noexcept {
    const abi::CompactSummaryViewV2& output = request.pipeline.output_view;
    return request.cell_count > 0
        && request.generation_index >= 0
        && output.generation_count > request.generation_index
        && output.cell_count == request.cell_count
        && valid_plane(
            output.validity,
            abi::kValidityFieldCountV2,
            output.generation_count,
            request.cell_count
        );
}

bool valid_compact_view(
    const abi::CompactSummaryViewV2& view,
    const std::int64_t minimum_generation_count,
    const std::int64_t cell_count
) noexcept {
    return view.generation_count >= minimum_generation_count
        && view.cell_count == cell_count
        && valid_plane(
            view.validity,
            abi::kValidityFieldCountV2,
            view.generation_count,
            cell_count
        )
        && valid_plane(
            view.defined_mask,
            abi::kDefinedMaskFieldCountV2,
            view.generation_count,
            cell_count
        )
        && valid_plane(
            view.status_codes,
            abi::kStatusFieldCountV2,
            view.generation_count,
            cell_count
        )
        && valid_plane(
            view.integer_values,
            abi::kIntegerFieldCountV2,
            view.generation_count,
            cell_count
        )
        && valid_plane(
            view.float_values,
            abi::kFloatFieldCountV2,
            view.generation_count,
            cell_count
        )
        && valid_plane(
            view.digest_values,
            abi::kDigestFieldCountV2 * abi::kDigestBytesV2,
            view.generation_count,
            cell_count
        )
        && valid_plane(
            view.text_values,
            abi::kTextFieldCountV2 * abi::kTextRecordBytesV2,
            view.generation_count,
            cell_count
        );
}

bool duplicate_output_index(
    const checkpoint_hash::TensorDigestPointerBatchV1* const batches,
    const std::int32_t batch,
    const std::int32_t entry,
    const bool raw
) noexcept {
    const auto& candidate = batches[batch].entries[entry];
    const std::int32_t candidate_index = raw
        ? candidate.raw_output_index
        : candidate.canonical_output_index;
    for (std::int32_t prior_batch = 0; prior_batch <= batch; ++prior_batch) {
        const std::int32_t end = prior_batch == batch
            ? entry
            : batches[prior_batch].count;
        for (std::int32_t prior_entry = 0; prior_entry < end; ++prior_entry) {
            const auto& prior = batches[prior_batch].entries[prior_entry];
            const std::int32_t prior_index = raw
                ? prior.raw_output_index
                : prior.canonical_output_index;
            if (prior_index == candidate_index) {
                return true;
            }
        }
    }
    return false;
}

BoundaryLaunchStatusV1 validate_checkpoint_batches(
    const CompactSummaryBoundaryRequestV1& request
) noexcept {
    const auto& outputs = request.checkpoint_outputs;
    if (
        request.checkpoint_batch_count < 0
        || outputs.generation_count <= request.generation_index
        || outputs.cell_count != request.cell_count
        || outputs.generation_index != request.generation_index
        || outputs.descriptor_field_count < 0
        || outputs.output_field_count < 0
    ) {
        return kBoundaryInvalidCheckpointBatchLayoutV1;
    }
    if (request.checkpoint_batch_count == 0) {
        return outputs.descriptor_field_count == 0
            ? kBoundaryLaunchSuccessV1
            : kBoundaryInvalidCheckpointBatchLayoutV1;
    }
    if (
        request.checkpoint_batches == nullptr
        || outputs.raw_digests == nullptr
        || outputs.canonical_digests == nullptr
        || outputs.status == nullptr
        || outputs.valid == nullptr
        || outputs.output_field_count <= 0
        || outputs.descriptor_field_count <= 0
    ) {
        return kBoundaryInvalidCheckpointBatchLayoutV1;
    }

    std::int64_t descriptor_index = 0;
    for (
        std::int32_t batch_index = 0;
        batch_index < request.checkpoint_batch_count;
        ++batch_index
    ) {
        const auto& batch = request.checkpoint_batches[batch_index];
        if (
            batch.count <= 0
            || batch.count > checkpoint_hash::kPointerBatchCapacityV1
        ) {
            return kBoundaryInvalidCheckpointBatchLayoutV1;
        }
        for (std::int32_t entry_index = 0; entry_index < batch.count; ++entry_index) {
            const auto& entry = batch.entries[entry_index];
            if (
                entry.descriptor_index != descriptor_index
                || entry.raw_output_index < 0
                || entry.raw_output_index >= outputs.output_field_count
                || entry.canonical_output_index < 0
                || entry.canonical_output_index >= outputs.output_field_count
                || duplicate_output_index(
                    request.checkpoint_batches,
                    batch_index,
                    entry_index,
                    true
                )
                || duplicate_output_index(
                    request.checkpoint_batches,
                    batch_index,
                    entry_index,
                    false
                )
            ) {
                return kBoundaryInvalidCheckpointBatchLayoutV1;
            }
            ++descriptor_index;
        }
    }
    return descriptor_index == outputs.descriptor_field_count
        ? kBoundaryLaunchSuccessV1
        : kBoundaryInvalidCheckpointBatchLayoutV1;
}

bool valid_pipeline_buffers(
    const CompactSummaryBoundaryRequestV1& request,
    const bool host_digests_complete
) noexcept {
    const CompactPipelineBuffersV1& pipeline = request.pipeline;
    if (
        request.reduction_output == nullptr
        || request.reduction_inputs.cell_count != request.cell_count
        || pipeline.producer_status == nullptr
    ) {
        return false;
    }
    if (!host_digests_complete) {
        return true;
    }
    return pipeline.source_defined_masks != nullptr
        && pipeline.source_status_codes != nullptr
        && pipeline.source_integer_values != nullptr
        && pipeline.source_float_values != nullptr
        && pipeline.source_digest_values != nullptr
        && pipeline.source_text_values != nullptr
        && pipeline.scratch_ready != nullptr
        && pipeline.scratch_validity != nullptr
        && pipeline.scratch_defined_masks != nullptr
        && pipeline.scratch_status_codes != nullptr
        && pipeline.scratch_integer_values != nullptr
        && pipeline.scratch_float_values != nullptr
        && pipeline.scratch_digest_values != nullptr
        && pipeline.scratch_text_values != nullptr
        && pipeline.assemble_status != nullptr
        && valid_compact_view(pipeline.scratch_view, 1, request.cell_count)
        && valid_compact_view(
            pipeline.output_view,
            request.generation_index + 1,
            request.cell_count
        )
        && pipeline.scratch_view.validity.data == pipeline.scratch_validity
        && pipeline.scratch_view.defined_mask.data
            == pipeline.scratch_defined_masks
        && pipeline.scratch_view.status_codes.data
            == pipeline.scratch_status_codes
        && pipeline.scratch_view.integer_values.data
            == pipeline.scratch_integer_values
        && pipeline.scratch_view.float_values.data
            == pipeline.scratch_float_values
        && pipeline.scratch_view.digest_values.data
            == pipeline.scratch_digest_values
        && pipeline.scratch_view.text_values.data
            == pipeline.scratch_text_values;
}

RetainedHostCanonicalizationOutputsV1 retained_outputs(
    const CompactSummaryBoundaryRequestV1& request
) noexcept {
    return {
        request.reduction_output,
        request.checkpoint_batches,
        request.checkpoint_outputs,
        request.reporting,
        request.pipeline.source_digest_values,
        request.generation_index,
        request.cell_count,
    };
}

BoundaryLaunchReceiptV1 base_receipt(
    const CompactSummaryBoundaryRequestV1& request
) noexcept {
    const std::uint32_t provided =
        request.provided_host_canonical_digest_mask
        & kRequiredHostCanonicalDigestMaskV1;
    return {
        kBoundaryInvalidArgumentsV1,
        provided,
        kRequiredHostCanonicalDigestMaskV1 & ~provided,
        kUnresolvedLiveMetadataMaskV1,
        0U,
        0U,
        0U,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        retained_outputs(request),
    };
}

template <typename Kernel>
KernelResourceReceiptV1 query_kernel_resources(const Kernel kernel) {
    cudaFuncAttributes attributes{};
    cudaError_t error = cudaFuncGetAttributes(&attributes, kernel);
    if (error != cudaSuccess) {
        return {-1, -1, -1, -1, -1, -1, -1, -1};
    }
    std::int32_t occupancy = -1;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy,
        kernel,
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

}  // namespace

BoundaryLaunchReceiptV1 launch_compact_summary_boundary_v1(
    const CompactSummaryBoundaryRequestV1 request,
    cudaStream_t stream
) {
    BoundaryLaunchReceiptV1 receipt = base_receipt(request);
    if (!valid_output_target(request)) {
        return receipt;
    }

    const std::uint32_t unexpected_digest_bits =
        request.provided_host_canonical_digest_mask
        & ~kRequiredHostCanonicalDigestMaskV1;
    const bool host_digests_complete =
        receipt.missing_host_canonical_digest_mask == 0U;
    const BoundaryLaunchStatusV1 checkpoint_status =
        validate_checkpoint_batches(request);
    const bool arguments_valid = unexpected_digest_bits == 0U
        && checkpoint_status == kBoundaryLaunchSuccessV1
        && valid_pipeline_buffers(request, host_digests_complete);

    const unsigned int blocks = static_cast<unsigned int>(
        (request.cell_count + kThreadsPerBlockV1 - 1) / kThreadsPerBlockV1
    );
    compact_summary_boundary_validity_reset_v1<<<
        blocks,
        kThreadsPerBlockV1,
        0,
        stream
    >>>(
        request.pipeline.output_view,
        request.pipeline.producer_status,
        request.generation_index,
        arguments_valid
            ? static_cast<std::int32_t>(finalizer::kProducerStatusSuccess)
            : static_cast<std::int32_t>(kBoundaryProducerInvalidArgumentsV1)
    );
    receipt.validity_reset_kernel_launches = 1;
    receipt.total_kernel_launches = 1;
    if (!arguments_valid) {
        receipt.status = checkpoint_status == kBoundaryLaunchSuccessV1
            ? kBoundaryInvalidArgumentsV1
            : checkpoint_status;
        return receipt;
    }

    reductions::launch_compact_summary_reductions_v1(
        request.reduction_inputs,
        request.reduction_output,
        stream
    );
    receipt.reduction_kernel_launches = 1;
    ++receipt.total_kernel_launches;

    for (
        std::int32_t batch = 0;
        batch < request.checkpoint_batch_count;
        ++batch
    ) {
        checkpoint_hash::launch_checkpoint_tensor_hash_snapshot_v1(
            request.checkpoint_batches[batch],
            request.checkpoint_outputs,
            stream
        );
        ++receipt.checkpoint_hash_kernel_launches;
        ++receipt.total_kernel_launches;
    }

    compact_summary_boundary_prerequisite_gate_v1<<<
        blocks,
        kThreadsPerBlockV1,
        0,
        stream
    >>>(
        request.reduction_output,
        request.checkpoint_outputs,
        request.pipeline.producer_status,
        host_digests_complete
    );
    receipt.prerequisite_gate_kernel_launches = 1;
    ++receipt.total_kernel_launches;

    if (!host_digests_complete) {
        receipt.status = kBoundaryHostDigestFinalizationRequiredV1;
        return receipt;
    }

    eiel_cuda_architecture_b_compact_summary_pipeline_v1_launch(
        request.pipeline.source_defined_masks,
        request.pipeline.source_status_codes,
        request.pipeline.source_integer_values,
        request.pipeline.source_float_values,
        request.pipeline.source_digest_values,
        request.pipeline.source_text_values,
        request.pipeline.scratch_ready,
        request.pipeline.scratch_validity,
        request.pipeline.scratch_defined_masks,
        request.pipeline.scratch_status_codes,
        request.pipeline.scratch_integer_values,
        request.pipeline.scratch_float_values,
        request.pipeline.scratch_digest_values,
        request.pipeline.scratch_text_values,
        request.pipeline.scratch_view,
        request.pipeline.output_view,
        request.pipeline.assemble_status,
        request.pipeline.producer_status,
        request.cell_count,
        request.generation_index,
        stream
    );
    receipt.accepted_pipeline_kernel_launches = 3;
    receipt.total_kernel_launches += 3;
    receipt.status = kBoundaryLaunchSuccessV1;
    receipt.canonical_row_submission_capable = 1U;
    return receipt;
}

BoundaryKernelResourceReceiptV1
query_compact_summary_boundary_v1_resources() {
    return {
        query_kernel_resources(compact_summary_boundary_validity_reset_v1),
        query_kernel_resources(compact_summary_boundary_prerequisite_gate_v1),
    };
}

}  // namespace eiel::cuda_architecture_b::compact_summary_boundary_v1
