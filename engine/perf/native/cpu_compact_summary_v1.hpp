#pragma once

#include "../cuda_architecture_b_compact_summary_v2.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace eiel::native::cpu_compact_summary_v1 {

namespace abi = eiel::cuda_architecture_b::compact_summary_v2;

inline constexpr std::uint32_t kSourceAuthorityVersionV1 = 1;

using ValidityPlaneRowV1 =
    std::array<std::uint8_t, abi::kValidityFieldCountV2>;
using DefinedMaskPlaneRowV1 =
    std::array<abi::DefinedMaskV1, abi::kDefinedMaskFieldCountV2>;
using StatusPlaneRowV1 =
    std::array<std::int32_t, abi::kStatusFieldCountV2>;
using IntegerPlaneRowV1 =
    std::array<std::int64_t, abi::kIntegerFieldCountV2>;
using FloatPlaneRowV1 =
    std::array<double, abi::kFloatFieldCountV2>;
using DigestPlaneRowV1 = std::array<
    std::uint8_t,
    abi::kDigestFieldCountV2 * abi::kDigestBytesV2>;
using TextPlaneRowV1 = std::array<
    std::uint8_t,
    abi::kTextFieldCountV2 * abi::kTextRecordBytesV2>;

struct RawDigestV1 {
    // Canonical wire projection renders these fixed raw bytes as lowercase hex.
    std::array<std::uint8_t, abi::kDigestBytesV2> bytes;
};

struct Utf8TextValueV1 {
    std::uint8_t length;
    std::array<std::uint8_t, abi::kTextMaximumPayloadBytesV2> payload;
};

using SourceStatusValuesV1 = StatusPlaneRowV1;
using SourceIntegerValuesV1 = IntegerPlaneRowV1;
using SourceFloatValuesV1 = FloatPlaneRowV1;
using SourceDigestValuesV1 =
    std::array<RawDigestV1, abi::kDigestFieldCountV2>;
using SourceTextValuesV1 =
    std::array<Utf8TextValueV1, abi::kTextFieldCountV2>;

// This is the complete semantic authority accepted by the encoder. Callers
// populate it by the frozen ABI enum indices; no mapping or string lookup occurs.
struct CpuCompactSummarySourceAuthorityV1 {
    std::uint32_t authority_version;
    abi::DefinedMaskV1 defined_mask;
    SourceStatusValuesV1 status_codes;
    SourceIntegerValuesV1 integer_values;
    SourceFloatValuesV1 float_values;
    SourceDigestValuesV1 digest_values;
    SourceTextValuesV1 text_values;
};

// Each member is one exact ABI-v2 plane row. The struct itself is not a packed
// wire record; consumers should use the named planes or compact_summary_view().
struct CpuCompactSummaryRowV1 {
    ValidityPlaneRowV1 validity;
    DefinedMaskPlaneRowV1 defined_mask;
    StatusPlaneRowV1 status_codes;
    IntegerPlaneRowV1 integer_values;
    FloatPlaneRowV1 float_values;
    DigestPlaneRowV1 digest_values;
    TextPlaneRowV1 text_values;
};

enum class EncodeStatusV1 : std::uint32_t {
    kSuccess = 0,
    kSourceAuthorityVersionMismatch = 1,
    kReservedDefinedBit = 2,
    kPatchChildWithoutObject = 3,
    kStatusCodeInvalid = 4,
    kWorldPatchPresenceMismatch = 5,
    kIntegerRangeInvalid = 6,
    kFloatNonFinite = 7,
    kFloatRangeInvalid = 8,
    kUndefinedStorageNonzero = 9,
    kFlatPatchStorageNonzero = 10,
    kRecoverableStateInconsistent = 11,
    kTerminalStateInconsistent = 12,
    kTextLengthInvalid = 13,
    kTextUtf8Invalid = 14,
    kTextTailNonzero = 15,
    kRequiredDigestMissing = 16,
};

// Establishes the exact invalid-row contract: every plane is bit-zero.
void initialize_invalid_row(CpuCompactSummaryRowV1& row) noexcept;

[[nodiscard]] EncodeStatusV1 validate_completed_source(
    const CpuCompactSummarySourceAuthorityV1& source
) noexcept;

// Clears row first. On success validity becomes one only after all other planes
// are written; on any error row remains the exact invalid-row representation.
[[nodiscard]] EncodeStatusV1 encode_completed_row(
    const CpuCompactSummarySourceAuthorityV1& source,
    CpuCompactSummaryRowV1& row
) noexcept;

[[nodiscard]] abi::CompactSummaryViewV2 compact_summary_view(
    CpuCompactSummaryRowV1& row
) noexcept;

[[nodiscard]] const char* encode_status_name(EncodeStatusV1 status) noexcept;

static_assert(sizeof(RawDigestV1) == abi::kDigestBytesV2);
static_assert(sizeof(Utf8TextValueV1) == abi::kTextRecordBytesV2);
static_assert(sizeof(ValidityPlaneRowV1) == abi::kValidityLogicalBytesPerRowV2);
static_assert(
    sizeof(DefinedMaskPlaneRowV1) == abi::kDefinedMaskLogicalBytesPerRowV2
);
static_assert(sizeof(StatusPlaneRowV1) == abi::kStatusLogicalBytesPerRowV2);
static_assert(sizeof(IntegerPlaneRowV1) == abi::kIntegerLogicalBytesPerRowV2);
static_assert(sizeof(FloatPlaneRowV1) == abi::kFloatLogicalBytesPerRowV2);
static_assert(sizeof(DigestPlaneRowV1) == abi::kDigestLogicalBytesPerRowV2);
static_assert(sizeof(TextPlaneRowV1) == abi::kTextLogicalBytesPerRowV2);
static_assert(std::is_standard_layout_v<CpuCompactSummarySourceAuthorityV1>);
static_assert(std::is_trivially_copyable_v<CpuCompactSummarySourceAuthorityV1>);
static_assert(std::is_standard_layout_v<CpuCompactSummaryRowV1>);
static_assert(std::is_trivially_copyable_v<CpuCompactSummaryRowV1>);

}  // namespace eiel::native::cpu_compact_summary_v1
