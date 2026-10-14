#pragma once

#include "cuda_architecture_b_compact_summary_checkpoint_hash_v1.cuh"
#include "cuda_architecture_b_compact_summary_finalizer_v2.cuh"
#include "cuda_architecture_b_compact_summary_reductions_v1.cuh"
#include "cuda_architecture_b_compact_summary_reporting_authority_v1.cuh"
#include "cuda_architecture_b_compact_summary_v2.cuh"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

namespace eiel::cuda_architecture_b::compact_summary_boundary_v1 {

namespace abi = compact_summary_v2;
namespace checkpoint_hash = compact_summary_checkpoint_hash_v1;
namespace finalizer = compact_summary_finalizer_v2;
namespace reductions = compact_summary_reductions_v1;
namespace reporting = compact_summary_reporting_authority_v1;

inline constexpr char kContractVersionV1[] =
    "eiel_cuda_architecture_b_compact_summary_boundary_v1";

// These seven slots have canonical-JSON or manifest-envelope preimages. This
// boundary never synthesizes their CPython-compatible digest bytes on device.
enum HostCanonicalDigestSlotV1 : std::uint32_t {
    kHostDigestRngAuthorityV1 = 1U << 0,
    kHostDigestRngPolicyMutationTapeV1 = 1U << 1,
    kHostDigestAuthoritativeStateV1 = 1U << 2,
    kHostDigestCanonicalCheckpointStorageV1 = 1U << 3,
    kHostDigestNextGenerationStartStateV1 = 1U << 4,
    kHostDigestBehaviorRowsV1 = 1U << 5,
    kHostDigestLineageRecordsV1 = 1U << 6,
};

inline constexpr std::uint32_t kRequiredHostCanonicalDigestMaskV1 =
    kHostDigestRngAuthorityV1
    | kHostDigestRngPolicyMutationTapeV1
    | kHostDigestAuthoritativeStateV1
    | kHostDigestCanonicalCheckpointStorageV1
    | kHostDigestNextGenerationStartStateV1
    | kHostDigestBehaviorRowsV1
    | kHostDigestLineageRecordsV1;

// These scalar manifest authorities remain outside this standalone boundary.
// Reporting retained-layout v2 already closes cohort and lineage row metadata.
enum UnresolvedLiveMetadataInputV1 : std::uint32_t {
    kUnresolvedCheckpointManifestMetadataV1 = 1U << 0,
    kUnresolvedRngManifestMetadataV1 = 1U << 1,
};

inline constexpr std::uint32_t kUnresolvedLiveMetadataMaskV1 =
    kUnresolvedCheckpointManifestMetadataV1
    | kUnresolvedRngManifestMetadataV1;

enum BoundaryLaunchStatusV1 : std::int32_t {
    kBoundaryLaunchSuccessV1 = 0,
    kBoundaryHostDigestFinalizationRequiredV1 = 1,
    kBoundaryInvalidArgumentsV1 = 2,
    kBoundaryInvalidCheckpointBatchLayoutV1 = 3,
};

// These values live in the caller's producer_status plane and are deliberately
// outside the accepted finalizer status range. Any nonzero value makes the
// accepted finalizer a no-op for that cell.
enum BoundaryProducerStatusV1 : std::int32_t {
    kBoundaryProducerReductionFailureV1 = 100,
    kBoundaryProducerCheckpointHashFailureV1 = 101,
    kBoundaryProducerHostDigestFinalizationRequiredV1 = 102,
    kBoundaryProducerInvalidArgumentsV1 = 103,
};

struct CompactPipelineBuffersV1 {
    const abi::DefinedMaskV1* source_defined_masks;
    const std::int32_t* source_status_codes;
    const std::int64_t* source_integer_values;
    const double* source_float_values;
    const std::uint8_t* source_digest_values;
    const std::uint8_t* source_text_values;
    std::uint8_t* scratch_ready;
    std::uint8_t* scratch_validity;
    abi::DefinedMaskV1* scratch_defined_masks;
    std::int32_t* scratch_status_codes;
    std::int64_t* scratch_integer_values;
    double* scratch_float_values;
    std::uint8_t* scratch_digest_values;
    std::uint8_t* scratch_text_values;
    abi::CompactSummaryViewV2 scratch_view;
    abi::CompactSummaryViewV2 output_view;
    std::int32_t* assemble_status;
    std::int32_t* producer_status;
};

// Pre-turnover reporting rows are produced elsewhere and retained here only as
// typed inputs for the one final host canonicalization after the final sync.
struct RetainedReportingAuthorityV1 {
    const reporting::GenerationAuthorityHeaderV1* headers;
    const reporting::BehaviorAuthorityRowV1* behavior_rows;
    const reporting::LineageAuthorityRowV1* lineage_rows;
    std::int64_t header_cell_stride;
    std::int64_t behavior_cell_stride;
    std::int64_t lineage_cell_stride;
    std::int32_t row_capacity;
};

struct CompactSummaryBoundaryRequestV1 {
    std::int64_t generation_index;
    std::int64_t cell_count;
    reductions::ReductionInputsV1 reduction_inputs;
    reductions::ReductionResultV1* reduction_output;
    const checkpoint_hash::TensorDigestPointerBatchV1* checkpoint_batches;
    std::int32_t checkpoint_batch_count;
    checkpoint_hash::SnapshotOutputsV1 checkpoint_outputs;
    CompactPipelineBuffersV1 pipeline;
    RetainedReportingAuthorityV1 reporting;
    std::uint32_t provided_host_canonical_digest_mask;
};

struct RetainedHostCanonicalizationOutputsV1 {
    const reductions::ReductionResultV1* lifecycle_identity;
    const checkpoint_hash::TensorDigestPointerBatchV1* checkpoint_batches;
    checkpoint_hash::SnapshotOutputsV1 checkpoint_outputs;
    RetainedReportingAuthorityV1 reporting;
    const std::uint8_t* completed_host_digest_values;
    std::int64_t generation_index;
    std::int64_t cell_count;
};

struct BoundaryLaunchReceiptV1 {
    std::int32_t status;
    std::uint32_t provided_host_canonical_digest_mask;
    std::uint32_t missing_host_canonical_digest_mask;
    std::uint32_t unresolved_live_metadata_mask;
    std::uint8_t canonical_row_submission_capable;
    std::uint8_t runtime_capability_promoted;
    std::uint16_t reserved;
    std::int32_t validity_reset_kernel_launches;
    std::int32_t reduction_kernel_launches;
    std::int32_t checkpoint_hash_kernel_launches;
    std::int32_t prerequisite_gate_kernel_launches;
    std::int32_t accepted_pipeline_kernel_launches;
    std::int32_t total_kernel_launches;
    std::int32_t intermediate_synchronizations;
    std::int32_t device_allocations;
    std::int32_t host_allocations;
    RetainedHostCanonicalizationOutputsV1 retained;
};

struct KernelResourceReceiptV1 {
    std::int32_t binary_version;
    std::int32_t ptx_version;
    std::int32_t max_threads_per_block;
    std::int32_t registers_per_thread;
    std::int32_t occupancy_blocks_per_multiprocessor;
    std::int64_t constant_bytes;
    std::int64_t local_bytes_per_thread;
    std::int64_t shared_bytes_per_block;
};

struct BoundaryKernelResourceReceiptV1 {
    KernelResourceReceiptV1 validity_reset;
    KernelResourceReceiptV1 prerequisite_gate;
};

inline constexpr std::int32_t kThreadsPerBlockV1 = 128;

// Contract:
//   validity reset -> lifecycle/identity reduction -> checkpoint batches in
//   caller order -> prerequisite gate -> accepted assembler/status-gate/
//   finalizer pipeline. Every operation is queued on `stream`.
//
// The caller owns every pointed-to buffer and the host-resident immutable batch
// descriptor array. This function performs zero allocation, zero transfer, and
// zero synchronization. CUDA submission errors remain observable through the
// normal CUDA last-error mechanism after return.
BoundaryLaunchReceiptV1 launch_compact_summary_boundary_v1(
    CompactSummaryBoundaryRequestV1 request,
    cudaStream_t stream
);

BoundaryKernelResourceReceiptV1
query_compact_summary_boundary_v1_resources();

static_assert(std::is_standard_layout_v<CompactPipelineBuffersV1>);
static_assert(std::is_trivially_copyable_v<CompactPipelineBuffersV1>);
static_assert(std::is_standard_layout_v<RetainedReportingAuthorityV1>);
static_assert(std::is_trivially_copyable_v<RetainedReportingAuthorityV1>);
static_assert(std::is_standard_layout_v<CompactSummaryBoundaryRequestV1>);
static_assert(std::is_trivially_copyable_v<CompactSummaryBoundaryRequestV1>);
static_assert(std::is_standard_layout_v<BoundaryLaunchReceiptV1>);
static_assert(std::is_trivially_copyable_v<BoundaryLaunchReceiptV1>);
static_assert(kRequiredHostCanonicalDigestMaskV1 == 0x7FU);

}  // namespace eiel::cuda_architecture_b::compact_summary_boundary_v1
