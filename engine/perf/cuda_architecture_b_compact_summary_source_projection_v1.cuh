#pragma once

#include "cuda_architecture_b_compact_summary_v2.cuh"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace eiel::cuda_architecture_b::compact_summary_source_projection_v1 {

namespace abi = compact_summary_v2;

inline constexpr std::uint32_t kAuthoritySourceVersionV1 = 1;
inline constexpr char kAuthoritySourceSchemaV1[] =
    "eiel_cuda_architecture_b_compact_summary_source_authority_v1";

enum AuthorityRowDispositionV1 : std::uint32_t {
    kAuthorityRowComplete = 1,
    kAuthorityRowFailedGeneration = 2,
    kAuthorityRowStickyNoop = 3,
    kAuthorityRowUnexecuted = 4,
};

enum ProjectionProducerStatusV1 : std::int32_t {
    kProjectionSuccess = 0,
    kProjectionFailedGenerationNoop = 1,
    kProjectionStickyNoop = 2,
    kProjectionUnexecutedNoop = 3,
    kProjectionFailClosed = 4,
    kProjectionSourceVersionMismatch = 5,
    kProjectionDispositionInvalid = 6,
    kProjectionIndexInvalid = 7,
    kProjectionValidityInvalid = 8,
    kProjectionReservedBytesNonzero = 9,
    kProjectionReservedDefinedMask = 10,
    kProjectionStatusCodeInvalid = 11,
    kProjectionUndefinedStorageNonzero = 12,
    kProjectionIntegerRangeInvalid = 13,
    kProjectionFloatNonFinite = 14,
    kProjectionFloatRangeInvalid = 15,
    kProjectionDigestUndefinedStorageNonzero = 16,
    kProjectionTextLengthInvalid = 17,
    kProjectionTextUtf8Invalid = 18,
    kProjectionTextTailNonzero = 19,
    kProjectionPatchMaskInvalid = 20,
};

inline constexpr std::int64_t kAuthorityStatusValueCountV1 =
    abi::kStatusFieldCountV2;
inline constexpr std::int64_t kAuthorityIntegerValueCountV1 =
    abi::kIntegerFieldCountV2;
inline constexpr std::int64_t kAuthorityFloatValueCountV1 =
    abi::kFloatFieldCountV2;
inline constexpr std::int64_t kAuthorityDigestValueByteCountV1 =
    abi::kDigestFieldCountV2 * abi::kDigestBytesV2;
inline constexpr std::int64_t kAuthorityTextValueByteCountV1 =
    abi::kTextFieldCountV2 * abi::kTextRecordBytesV2;
inline constexpr std::size_t kAuthoritySourcePlaneLogicalBytesV1 =
    sizeof(std::uint8_t)
    + sizeof(abi::DefinedMaskV1)
    + sizeof(std::int32_t) * kAuthorityStatusValueCountV1
    + sizeof(std::int64_t) * kAuthorityIntegerValueCountV1
    + sizeof(double) * kAuthorityFloatValueCountV1
    + sizeof(std::uint8_t) * kAuthorityDigestValueByteCountV1
    + sizeof(std::uint8_t) * kAuthorityTextValueByteCountV1;

// The semantic extraction boundary supplies this complete value object. Arrays
// deliberately use frozen ABI indices until each controller authority has a
// separately qualified named extraction contract.
struct CompactSummaryGenerationCellAuthorityV1 {
    std::uint32_t source_version;
    std::uint32_t disposition;
    std::int64_t generation_index;
    std::int64_t cell_index;
    std::uint8_t validity;
    std::uint8_t header_reserved[7];
    abi::DefinedMaskV1 defined_mask;
    std::int32_t status_codes[kAuthorityStatusValueCountV1];
    std::int64_t integer_values[kAuthorityIntegerValueCountV1];
    double float_values[kAuthorityFloatValueCountV1];
    std::uint8_t digest_values[kAuthorityDigestValueByteCountV1];
    std::uint8_t text_values[kAuthorityTextValueByteCountV1];
    std::uint8_t tail_reserved[7];
};

inline constexpr std::size_t kAuthorityByteWidthV1 = 1256;
inline constexpr std::int32_t kThreadsPerBlockV1 = 128;

cudaError_t launch_compact_summary_source_projection_v1(
    const CompactSummaryGenerationCellAuthorityV1* authority_rows,
    std::int64_t authority_row_count,
    abi::CompactSummaryViewV2 output,
    std::int32_t* producer_status,
    cudaStream_t stream
) noexcept;

cudaError_t query_compact_summary_source_projection_v1_attributes(
    cudaFuncAttributes* attributes
) noexcept;

static_assert(abi::kNativeAbiVersionV2 == 2);
static_assert(abi::kPlaneCountV2 == 7);
static_assert(kAuthorityStatusValueCountV1 == 12);
static_assert(kAuthorityIntegerValueCountV1 == 76);
static_assert(kAuthorityFloatValueCountV1 == 21);
static_assert(kAuthorityDigestValueByteCountV1 == 320);
static_assert(kAuthorityTextValueByteCountV1 == 65);
static_assert(
    kAuthoritySourcePlaneLogicalBytesV1
    == abi::kLogicalBytesPerGenerationCellV2
);
static_assert(
    std::extent_v<decltype(CompactSummaryGenerationCellAuthorityV1::status_codes)>
    == abi::kStatusFieldCountV2
);
static_assert(
    std::extent_v<decltype(CompactSummaryGenerationCellAuthorityV1::integer_values)>
    == abi::kIntegerFieldCountV2
);
static_assert(
    std::extent_v<decltype(CompactSummaryGenerationCellAuthorityV1::float_values)>
    == abi::kFloatFieldCountV2
);
static_assert(
    std::extent_v<decltype(CompactSummaryGenerationCellAuthorityV1::digest_values)>
    == abi::kDigestFieldCountV2 * abi::kDigestBytesV2
);
static_assert(
    std::extent_v<decltype(CompactSummaryGenerationCellAuthorityV1::text_values)>
    == abi::kTextFieldCountV2 * abi::kTextRecordBytesV2
);
static_assert(std::is_standard_layout_v<CompactSummaryGenerationCellAuthorityV1>);
static_assert(std::is_trivially_copyable_v<CompactSummaryGenerationCellAuthorityV1>);
static_assert(alignof(CompactSummaryGenerationCellAuthorityV1) == 8);
static_assert(sizeof(CompactSummaryGenerationCellAuthorityV1) == kAuthorityByteWidthV1);
static_assert(offsetof(CompactSummaryGenerationCellAuthorityV1, source_version) == 0);
static_assert(offsetof(CompactSummaryGenerationCellAuthorityV1, disposition) == 4);
static_assert(offsetof(CompactSummaryGenerationCellAuthorityV1, generation_index) == 8);
static_assert(offsetof(CompactSummaryGenerationCellAuthorityV1, cell_index) == 16);
static_assert(offsetof(CompactSummaryGenerationCellAuthorityV1, validity) == 24);
static_assert(offsetof(CompactSummaryGenerationCellAuthorityV1, defined_mask) == 32);
static_assert(offsetof(CompactSummaryGenerationCellAuthorityV1, status_codes) == 40);
static_assert(offsetof(CompactSummaryGenerationCellAuthorityV1, integer_values) == 88);
static_assert(offsetof(CompactSummaryGenerationCellAuthorityV1, float_values) == 696);
static_assert(offsetof(CompactSummaryGenerationCellAuthorityV1, digest_values) == 864);
static_assert(offsetof(CompactSummaryGenerationCellAuthorityV1, text_values) == 1184);
static_assert(offsetof(CompactSummaryGenerationCellAuthorityV1, tail_reserved) == 1249);

}  // namespace eiel::cuda_architecture_b::compact_summary_source_projection_v1
