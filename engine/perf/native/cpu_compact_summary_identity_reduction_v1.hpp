#pragma once

#include "cpu_compact_summary_digests_v1.hpp"

#include <array>
#include <cstdint>
#include <type_traits>

namespace eiel::native::cpu_compact_summary_identity_reduction_v1 {

namespace digests = cpu_compact_summary_digests_v1;

inline constexpr char kContractVersionV1[] =
    "eiel_cpu_compact_summary_identity_reduction_v1";

// Adapters with a wider integer source can preserve overflow as an invalid
// scalar instead of narrowing before this authority boundary.
struct CheckedInt64V1 {
    std::int64_t value;
    bool in_range;
};

enum class IdentityReductionStatusV1 : std::uint32_t {
    kSuccess = 0,
    kNonLittleEndianHost = 1,
    kDtypeMismatch = 2,
    kRankMismatch = 3,
    kNegativeExtent = 4,
    kNonContiguous = 5,
    kStorageSizeMismatch = 6,
    kNullData = 7,
    kSizeOverflow = 8,
    kActiveAgentIdInvalid = 9,
    kCapacityOutOfRange = 10,
    kPopulationNextIdOutOfRange = 11,
    kActiveCountOutOfRange = 12,
    kActiveCountExceedsCapacity = 13,
    kSlotConservationInvalid = 14,
    kDefinedFlagInvalid = 15,
    kDefinedStateInvalid = 16,
    kActiveAgentIdBoundsInvalid = 17,
    kPopulationNextIdNotAboveMaximum = 18,
    kUnexpectedDigestFailure = 19,
};

struct IdentityReductionResultV1 {
    std::int64_t active_agent_count;
    std::int64_t active_agent_id_min;
    std::int64_t active_agent_id_max;
    std::int64_t free_slot_count;
    std::int64_t population_next_id;
    digests::DigestV1 active_agent_ids_digest;
    std::uint8_t active_agent_id_min_defined;
    std::uint8_t active_agent_id_max_defined;
    std::array<std::uint8_t, 6> reserved;
};

// Establishes the fail-closed representation for every result field.
void clear_identity_reduction_result(
    IdentityReductionResultV1& output
) noexcept;

// Validates scalar ranges, slot conservation, and optional-bound state. The
// candidate is cleared on every failure.
[[nodiscard]] IdentityReductionStatusV1 validate_identity_reduction_result(
    std::int64_t capacity,
    IdentityReductionResultV1& candidate
) noexcept;

// Clears output first, validates and digests the accepted TensorViewV1, then
// publishes a complete candidate only after every identity invariant passes.
[[nodiscard]] IdentityReductionStatusV1 reduce_identity(
    const digests::TensorViewV1& active_agent_ids,
    CheckedInt64V1 capacity,
    CheckedInt64V1 population_next_id,
    IdentityReductionResultV1& output
) noexcept;

[[nodiscard]] IdentityReductionStatusV1 reduce_identity(
    const digests::TensorViewV1& active_agent_ids,
    std::int64_t capacity,
    std::int64_t population_next_id,
    IdentityReductionResultV1& output
) noexcept;

[[nodiscard]] const char* identity_reduction_status_name(
    IdentityReductionStatusV1 status
) noexcept;

static_assert(std::is_standard_layout_v<CheckedInt64V1>);
static_assert(std::is_trivially_copyable_v<CheckedInt64V1>);
static_assert(std::is_standard_layout_v<IdentityReductionResultV1>);
static_assert(std::is_trivially_copyable_v<IdentityReductionResultV1>);
static_assert(sizeof(IdentityReductionResultV1) == 80U);

}  // namespace eiel::native::cpu_compact_summary_identity_reduction_v1
