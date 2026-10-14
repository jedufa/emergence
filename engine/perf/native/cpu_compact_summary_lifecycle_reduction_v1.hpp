#pragma once

#include "cpu_compact_summary_digests_v1.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace eiel::native::cpu_compact_summary_lifecycle_reduction_v1 {

namespace digests = cpu_compact_summary_digests_v1;

inline constexpr std::size_t kLifecycleEventTypeCountV1 = 4U;

using EventTypeCountsV1 =
    std::array<std::int64_t, kLifecycleEventTypeCountV1>;

struct LifecycleReductionV1 {
    std::int64_t event_row_count;
    std::int64_t generation_event_count;
    std::int64_t turnover_event_count;
    // Indices 0..3 represent event type codes 1..4 respectively.
    EventTypeCountsV1 generation_counts_by_type;
    EventTypeCountsV1 turnover_counts_by_type;
    std::int64_t deaths_energy_depleted;
    std::int64_t deaths_health_depleted;
    digests::DigestV1 active_prefix_digest;
};

enum class ReductionStatusV1 : std::uint32_t {
    kSuccess = 0,
    kCursorOutOfRange = 1,
    kCursorOrderInvalid = 2,
    kDigestValidationFailed = 3,
    kUnknownEventType = 4,
    kUnknownGenerationDeathReason = 5,
};

// events.active_count is the authoritative active-prefix length and must equal
// event_cursor_end. Output is bit-zero on every non-success return.
[[nodiscard]] ReductionStatusV1 reduce_typed_lifecycle_active_prefix(
    const digests::TypedLifecycleEventsViewV1& events,
    std::int64_t event_cursor_start,
    std::int64_t event_cursor_pre_turnover,
    std::int64_t event_cursor_end,
    LifecycleReductionV1& output
) noexcept;

[[nodiscard]] const char* reduction_status_name(
    ReductionStatusV1 status
) noexcept;

static_assert(std::is_standard_layout_v<LifecycleReductionV1>);
static_assert(std::is_trivially_copyable_v<LifecycleReductionV1>);

}  // namespace eiel::native::cpu_compact_summary_lifecycle_reduction_v1
