#pragma once

#include "cuda_architecture_b_compact_summary_host_canonical_v1.hpp"
#include "cuda_architecture_b_compact_summary_host_token_resolver_v1.hpp"
#include "cuda_architecture_b_compact_summary_metadata_retention_v1.cuh"
#include "cuda_architecture_b_compact_summary_reductions_v1.cuh"
#include "cuda_architecture_b_compact_summary_reporting_authority_v1.cuh"
#include "cuda_architecture_b_compact_summary_source_projection_v1.cuh"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace eiel::cuda_architecture_b::compact_summary_host_settlement_v1 {

namespace canonical = compact_summary_host_canonical_v1;
namespace token_resolver = compact_summary_host_token_resolver_v1;
namespace metadata = compact_summary_metadata_retention_v1;
namespace projection = compact_summary_source_projection_v1;
namespace reductions = compact_summary_reductions_v1;
namespace reporting = compact_summary_reporting_authority_v1;

inline constexpr char kContractVersionV1[] =
    "eiel_cuda_architecture_b_compact_summary_host_settlement_v1";
inline constexpr std::int64_t kPatchMetricWidthV1 = 19;

enum HostSettlementStatusV1 : std::int32_t {
    kHostSettlementSuccessV1 = 0,
    kHostSettlementInvalidArgumentsV1 = 1,
    kHostSettlementLifecycleInvalidV1 = 2,
    kHostSettlementReportingInvalidV1 = 3,
    kHostSettlementOuterAuthorityPendingV1 = 4,
    kHostSettlementCheckpointInvalidV1 = 5,
    kHostSettlementReductionInvalidV1 = 6,
    kHostSettlementMetadataInvalidV1 = 7,
    kHostSettlementTokenResolutionInvalidV1 = 8,
    kHostSettlementTemplateRenderInvalidV1 = 9,
    kHostSettlementPartialPrefixV1 = 10,
};

struct HostSettlementContextV1 {
    projection::CompactSummaryGenerationCellAuthorityV1* authority_rows;
    const std::uint8_t* reporting_retained_bytes;
    const double* generation_summary_values;
    const double* patch_metrics_values;
    const std::uint8_t* lifecycle_digest_values;
    const std::int32_t* lifecycle_status_values;
    const std::uint8_t* lifecycle_validity_values;
    const std::uint8_t* checkpoint_raw_digests;
    const std::uint8_t* checkpoint_canonical_digests;
    const std::int32_t* checkpoint_status_values;
    const std::uint8_t* checkpoint_validity_values;
    const reductions::ReductionResultV1* reduction_rows;
    const metadata::CompactSummaryMetadataRetentionV1* metadata_rows;
    const bool* checkpoint_due_values;
    const std::uint8_t* import_digest_values;
    token_resolver::runtime::HostTemplateRuntimeV1* host_template_runtime;
    // Caller-owned continuation state, one independent entry per cell. The
    // null/zero form is retained only for the legacy single-cell path.
    canonical::LineageMetricsStateV1* lineage_state_values;
    const std::int64_t* completed_generation_values;
    const std::uint64_t* host_token_resolver_rows;
    const std::int64_t* checkpoint_descriptor_specs;
    std::uint64_t host_token_resolver_row_count;
    token_resolver::HostTokenResolverFailureV1 token_resolver_failure;
    std::int64_t initial_generation;
    std::int64_t generation_count;
    std::int64_t cell_count;
    std::int64_t checkpoint_field_count;
    std::int64_t checkpoint_descriptor_count;
    std::int64_t reporting_retained_row_bytes;
    std::int64_t generation_summary_width;
    std::int64_t patch_metrics_width;
    std::int64_t lineage_state_count;
    std::int32_t world_code;
    std::int32_t reporting_row_capacity;
    std::int32_t completed_rows;
    std::int32_t first_failed_row;
    std::int32_t status;
};

void settle_compact_summary_rows_callback_v1(void* user_data) noexcept;

static_assert(std::is_standard_layout_v<HostSettlementContextV1>);
static_assert(std::is_trivially_copyable_v<HostSettlementContextV1>);

}  // namespace eiel::cuda_architecture_b::compact_summary_host_settlement_v1
