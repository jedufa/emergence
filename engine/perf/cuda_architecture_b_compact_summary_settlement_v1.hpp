#pragma once

#include "cuda_architecture_b_compact_summary_source_projection_v1.cuh"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>

namespace eiel::cuda_architecture_b::compact_summary_settlement_v1 {

namespace projection = compact_summary_source_projection_v1;

inline constexpr char kContractVersionV1[] =
    "eiel_cuda_architecture_b_compact_summary_settlement_v1";

enum SettlementStatusV1 : std::int32_t {
    kSettlementSuccessV1 = 0,
    kSettlementInvalidArgumentsV1 = 1,
    kSettlementPartialAuthorityCopyFailedV1 = 2,
    kSettlementReportingCopyFailedV1 = 3,
    kSettlementCheckpointCopyFailedV1 = 4,
    kSettlementSummaryCopyFailedV1 = 5,
    kSettlementLifecycleCopyFailedV1 = 6,
    kSettlementReductionCopyFailedV1 = 7,
    kSettlementMetadataCopyFailedV1 = 8,
    kSettlementHostCallbackFailedV1 = 9,
    kSettlementAuthorityCopyFailedV1 = 10,
    kSettlementProjectionLaunchFailedV1 = 11,
    kSettlementPatchMetricsCopyFailedV1 = 12,
    kSettlementCompletionCopyFailedV1 = 13,
};

struct ByteTransferV1 {
    const void* device_source;
    void* host_destination;
    std::size_t bytes;
};

struct SettlementRequestV1 {
    ByteTransferV1 partial_authority;
    ByteTransferV1 reporting;
    ByteTransferV1 checkpoint_raw_digests;
    ByteTransferV1 checkpoint_canonical_digests;
    ByteTransferV1 checkpoint_status;
    ByteTransferV1 checkpoint_validity;
    ByteTransferV1 generation_summary;
    // Flat leaves this transfer empty. Patch P2 supplies one compact row per
    // generation/cell without copying the full per-tick Patch stats tape.
    ByteTransferV1 patch_metrics;
    ByteTransferV1 lifecycle_digest;
    ByteTransferV1 lifecycle_status;
    ByteTransferV1 lifecycle_validity;
    ByteTransferV1 reduction;
    ByteTransferV1 metadata_retention;
    ByteTransferV1 completed_generations;
    void* host_callback_context;
    cudaHostFn_t host_callback;
    const projection::CompactSummaryGenerationCellAuthorityV1* host_authority;
    projection::CompactSummaryGenerationCellAuthorityV1* device_authority;
    std::int64_t authority_row_count;
    compact_summary_v2::CompactSummaryViewV2 output;
    std::int32_t* producer_status;
};

struct SettlementReceiptV1 {
    std::int32_t status;
    std::int32_t d2h_operations;
    std::int32_t host_callback_launches;
    std::int32_t h2d_operations;
    std::int32_t projection_launches;
    std::int32_t synchronizations;
    std::int32_t allocations;
};

inline bool valid_transfer_v1(const ByteTransferV1& transfer) noexcept {
    return transfer.device_source != nullptr
        && transfer.host_destination != nullptr
        && transfer.bytes > 0;
}

inline SettlementReceiptV1 enqueue_settlement_v1(
    const SettlementRequestV1& request,
    cudaStream_t stream
) noexcept {
    SettlementReceiptV1 receipt{
        kSettlementInvalidArgumentsV1,
        0,
        0,
        0,
        0,
        0,
        0,
    };
    if (
        !valid_transfer_v1(request.partial_authority)
        || !valid_transfer_v1(request.reporting)
        || !valid_transfer_v1(request.checkpoint_raw_digests)
        || !valid_transfer_v1(request.checkpoint_canonical_digests)
        || !valid_transfer_v1(request.checkpoint_status)
        || !valid_transfer_v1(request.checkpoint_validity)
        || !valid_transfer_v1(request.generation_summary)
        || !valid_transfer_v1(request.lifecycle_digest)
        || !valid_transfer_v1(request.lifecycle_status)
        || !valid_transfer_v1(request.lifecycle_validity)
        || !valid_transfer_v1(request.reduction)
        || !valid_transfer_v1(request.metadata_retention)
        || !valid_transfer_v1(request.completed_generations)
        || request.host_callback_context == nullptr
        || request.host_callback == nullptr
        || request.host_authority == nullptr
        || request.device_authority == nullptr
        || request.authority_row_count <= 0
        || request.producer_status == nullptr
    ) {
        return receipt;
    }

    const bool patch_metrics_enabled =
        valid_transfer_v1(request.patch_metrics);
    const bool patch_metrics_disabled =
        request.patch_metrics.device_source == nullptr
        && request.patch_metrics.host_destination == nullptr
        && request.patch_metrics.bytes == 0;
    if (!patch_metrics_enabled && !patch_metrics_disabled) {
        return receipt;
    }

    const ByteTransferV1 transfers[] = {
        request.partial_authority,
        request.reporting,
        request.checkpoint_raw_digests,
        request.checkpoint_canonical_digests,
        request.checkpoint_status,
        request.checkpoint_validity,
        request.generation_summary,
        request.lifecycle_digest,
        request.lifecycle_status,
        request.lifecycle_validity,
        request.reduction,
        request.metadata_retention,
        request.completed_generations,
    };
    for (std::size_t index = 0; index < std::size(transfers); ++index) {
        const cudaError_t status = cudaMemcpyAsync(
            transfers[index].host_destination,
            transfers[index].device_source,
            transfers[index].bytes,
            cudaMemcpyDeviceToHost,
            stream
        );
        if (status != cudaSuccess) {
            if (index == 0) {
                receipt.status = kSettlementPartialAuthorityCopyFailedV1;
            } else if (index == 1) {
                receipt.status = kSettlementReportingCopyFailedV1;
            } else if (index <= 5) {
                receipt.status = kSettlementCheckpointCopyFailedV1;
            } else if (index == 6) {
                receipt.status = kSettlementSummaryCopyFailedV1;
            } else if (index <= 9) {
                receipt.status = kSettlementLifecycleCopyFailedV1;
            } else if (index == 10) {
                receipt.status = kSettlementReductionCopyFailedV1;
            } else if (index == 11) {
                receipt.status = kSettlementMetadataCopyFailedV1;
            } else {
                receipt.status = kSettlementCompletionCopyFailedV1;
            }
            return receipt;
        }
        ++receipt.d2h_operations;
    }
    if (patch_metrics_enabled) {
        const cudaError_t status = cudaMemcpyAsync(
            request.patch_metrics.host_destination,
            request.patch_metrics.device_source,
            request.patch_metrics.bytes,
            cudaMemcpyDeviceToHost,
            stream
        );
        if (status != cudaSuccess) {
            receipt.status = kSettlementPatchMetricsCopyFailedV1;
            return receipt;
        }
        ++receipt.d2h_operations;
    }

    if (cudaLaunchHostFunc(
            stream,
            request.host_callback,
            request.host_callback_context
        ) != cudaSuccess) {
        receipt.status = kSettlementHostCallbackFailedV1;
        return receipt;
    }
    receipt.host_callback_launches = 1;

    const std::size_t authority_bytes = static_cast<std::size_t>(
        request.authority_row_count
    ) * sizeof(projection::CompactSummaryGenerationCellAuthorityV1);
    if (cudaMemcpyAsync(
            request.device_authority,
            request.host_authority,
            authority_bytes,
            cudaMemcpyHostToDevice,
            stream
        ) != cudaSuccess) {
        receipt.status = kSettlementAuthorityCopyFailedV1;
        return receipt;
    }
    receipt.h2d_operations = 1;

    if (projection::launch_compact_summary_source_projection_v1(
            request.device_authority,
            request.authority_row_count,
            request.output,
            request.producer_status,
            stream
        ) != cudaSuccess) {
        receipt.status = kSettlementProjectionLaunchFailedV1;
        return receipt;
    }
    receipt.projection_launches = 1;
    receipt.status = kSettlementSuccessV1;
    return receipt;
}

static_assert(std::is_standard_layout_v<ByteTransferV1>);
static_assert(std::is_trivially_copyable_v<ByteTransferV1>);
static_assert(std::is_standard_layout_v<SettlementRequestV1>);
static_assert(std::is_trivially_copyable_v<SettlementRequestV1>);
static_assert(std::is_standard_layout_v<SettlementReceiptV1>);
static_assert(std::is_trivially_copyable_v<SettlementReceiptV1>);

}  // namespace eiel::cuda_architecture_b::compact_summary_settlement_v1
