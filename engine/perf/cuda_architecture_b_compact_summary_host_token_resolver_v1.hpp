#pragma once

#include "cuda_architecture_b_compact_summary_checkpoint_runtime_v1.hpp"
#include "cuda_architecture_b_compact_summary_host_template_runtime_v1.hpp"
#include "cuda_architecture_b_compact_summary_metadata_retention_v1.cuh"

#include <cstdint>
#include <type_traits>

namespace eiel::cuda_architecture_b::compact_summary_host_token_resolver_v1 {

namespace checkpoint = compact_summary_checkpoint_runtime_v1;
namespace metadata = compact_summary_metadata_retention_v1;
namespace runtime = compact_summary_host_template_runtime_v1;

inline constexpr char kContractVersionV1[] =
    "eiel_cuda_architecture_b_compact_summary_host_token_resolver_v1";
inline constexpr std::uint64_t kResolverAbiVersionV1 = 1U;
inline constexpr std::uint32_t kResolverColumnCountV1 = 6U;
inline constexpr std::uint64_t kNoResolverRowV1 = UINT64_MAX;
inline constexpr std::uint32_t kNoResolverCoordinateV1 = UINT32_MAX;
inline constexpr std::int64_t kNoDescriptorIndexV1 = -1;

enum ResolverColumnV1 : std::uint32_t {
    kResolverAbiVersionColumnV1 = 0,
    kResolverSourceKindColumnV1,
    kResolverPrimaryIndexColumnV1,
    kResolverSecondaryCodeColumnV1,
    kResolverRankColumnV1,
    kResolverReservedZeroColumnV1,
};

enum class HostTokenResolverStatusV1 : std::uint32_t {
    kSuccess = 0,
    kNullArgument = 1,
    kInvalidDimensions = 2,
    kArithmeticOverflow = 3,
    kRuntimeNotReady = 4,
    kResolverRowCountMismatch = 5,
    kResolverAbiMismatch = 6,
    kResolverReservedNonzero = 7,
    kUnsupportedSourceKind = 8,
    kResolverContradiction = 9,
    kDescriptorContradiction = 10,
    kCheckpointInvalid = 11,
    kMetadataInvalid = 12,
    kTokenStorageContradiction = 13,
};

enum class HostTokenResolverFailureComponentV1 : std::uint32_t {
    kNone = 0,
    kRequest = 1,
    kRuntime = 2,
    kDescriptorSpecs = 3,
    kCheckpointPlanes = 4,
    kMetadata = 5,
    kResolverPlan = 6,
    kTokenStorage = 7,
};

struct HostTokenResolverFailureV1 {
    std::uint32_t status;
    std::uint32_t component;
    std::uint64_t resolver_row;
    std::uint32_t generation;
    std::uint32_t cell;
    std::uint32_t surface;
    std::uint32_t token_index;
    std::int64_t descriptor_index;
};

// Every pointer is a contiguous pinned-host view owned by the caller. Digest
// values are [G,C,D,32], status/validity are [G,C,D], metadata is [G,C], and
// import digests are [C,32]. D must be the exact Flat or Patch P2 descriptor
// count. Resolver rows are [T,6] in exact program and token order.
// resolve_compact_summary_host_tokens_v1 performs no allocation.
struct HostTokenResolverRequestV1 {
    const std::uint64_t* resolver_rows;
    std::uint64_t resolver_row_count;
    runtime::HostTemplateRuntimeV1* host_template_runtime;
    const std::int64_t* descriptor_specs;
    const std::uint8_t* checkpoint_raw_digests;
    const std::uint8_t* checkpoint_canonical_digests;
    const std::int32_t* checkpoint_status;
    const std::uint8_t* checkpoint_validity;
    const metadata::CompactSummaryMetadataRetentionV1* metadata_rows;
    const std::uint8_t* import_digests;
    // generation_count is the allocated/runtime extent. active_generation_count
    // is the completed prefix that may be settled after a sticky failure.
    std::int64_t generation_count;
    std::int64_t active_generation_count;
    std::int64_t cell_count;
    std::int64_t descriptor_count;
    HostTokenResolverFailureV1* failure;
};

HostTokenResolverStatusV1 resolve_compact_summary_host_tokens_v1(
    const HostTokenResolverRequestV1* request
) noexcept;

const char* host_token_resolver_status_name_v1(
    HostTokenResolverStatusV1 status
) noexcept;

static_assert(kResolverReservedZeroColumnV1 + 1U == kResolverColumnCountV1);
static_assert(std::is_standard_layout_v<HostTokenResolverFailureV1>);
static_assert(std::is_trivially_copyable_v<HostTokenResolverFailureV1>);
static_assert(std::is_standard_layout_v<HostTokenResolverRequestV1>);
static_assert(std::is_trivially_copyable_v<HostTokenResolverRequestV1>);

}  // namespace eiel::cuda_architecture_b::compact_summary_host_token_resolver_v1
