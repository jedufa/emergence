#include "cpu_compact_summary_lifecycle_reduction_v1.hpp"

#include <cstddef>
#include <cstdint>

namespace eiel::native::cpu_compact_summary_lifecycle_reduction_v1 {
namespace {

inline constexpr std::int32_t kBirthCommittedV1 = 1;
inline constexpr std::int32_t kMutationV1 = 2;
inline constexpr std::int32_t kBirthFailedV1 = 3;
inline constexpr std::int32_t kDeathCommittedV1 = 4;
inline constexpr std::int32_t kEnergyDepletedV1 = 1;
inline constexpr std::int32_t kHealthDepletedV1 = 2;

void clear_output(LifecycleReductionV1& output) noexcept {
    output = {};
}

bool increment_event_count(
    const std::int32_t event_type,
    EventTypeCountsV1& counts
) noexcept {
    if (event_type < kBirthCommittedV1 || event_type > kDeathCommittedV1) {
        return false;
    }
    const auto index = static_cast<std::size_t>(event_type - 1);
    ++counts[index];
    return true;
}

}  // namespace

ReductionStatusV1 reduce_typed_lifecycle_active_prefix(
    const digests::TypedLifecycleEventsViewV1& events,
    const std::int64_t event_cursor_start,
    const std::int64_t event_cursor_pre_turnover,
    const std::int64_t event_cursor_end,
    LifecycleReductionV1& output
) noexcept {
    clear_output(output);
    if (event_cursor_start < 0
        || event_cursor_pre_turnover < 0
        || event_cursor_end < 0
        || event_cursor_end != events.active_count) {
        return ReductionStatusV1::kCursorOutOfRange;
    }
    if (event_cursor_start > event_cursor_pre_turnover
        || event_cursor_pre_turnover > event_cursor_end) {
        return ReductionStatusV1::kCursorOrderInvalid;
    }

    LifecycleReductionV1 candidate{};
    const digests::DigestStatusV1 digest_status =
        digests::typed_lifecycle_active_prefix_digest(
            events, candidate.active_prefix_digest
        );
    if (digest_status != digests::DigestStatusV1::kSuccess) {
        return digest_status == digests::DigestStatusV1::kCursorOutOfRange
            ? ReductionStatusV1::kCursorOutOfRange
            : ReductionStatusV1::kDigestValidationFailed;
    }

    candidate.event_row_count = event_cursor_end - event_cursor_start;
    candidate.generation_event_count =
        event_cursor_pre_turnover - event_cursor_start;
    candidate.turnover_event_count =
        event_cursor_end - event_cursor_pre_turnover;

    const auto* const event_types =
        static_cast<const std::int32_t*>(events.event_type.data);
    const auto* const reason_codes =
        static_cast<const std::int32_t*>(events.reason_code.data);

    for (std::int64_t row = event_cursor_start;
         row < event_cursor_pre_turnover;
         ++row) {
        const std::int32_t event_type = event_types[row];
        if (!increment_event_count(
                event_type, candidate.generation_counts_by_type
            )) {
            return ReductionStatusV1::kUnknownEventType;
        }
        if (event_type == kDeathCommittedV1) {
            const std::int32_t reason_code = reason_codes[row];
            if (reason_code == kEnergyDepletedV1) {
                ++candidate.deaths_energy_depleted;
            } else if (reason_code == kHealthDepletedV1) {
                ++candidate.deaths_health_depleted;
            } else {
                return ReductionStatusV1::kUnknownGenerationDeathReason;
            }
        }
    }
    for (std::int64_t row = event_cursor_pre_turnover;
         row < event_cursor_end;
         ++row) {
        if (!increment_event_count(
                event_types[row], candidate.turnover_counts_by_type
            )) {
            return ReductionStatusV1::kUnknownEventType;
        }
    }

    output = candidate;
    return ReductionStatusV1::kSuccess;
}

const char* reduction_status_name(const ReductionStatusV1 status) noexcept {
    switch (status) {
        case ReductionStatusV1::kSuccess:
            return "success";
        case ReductionStatusV1::kCursorOutOfRange:
            return "cursor_out_of_range";
        case ReductionStatusV1::kCursorOrderInvalid:
            return "cursor_order_invalid";
        case ReductionStatusV1::kDigestValidationFailed:
            return "digest_validation_failed";
        case ReductionStatusV1::kUnknownEventType:
            return "unknown_event_type";
        case ReductionStatusV1::kUnknownGenerationDeathReason:
            return "unknown_generation_death_reason";
    }
    return "unknown";
}

}  // namespace eiel::native::cpu_compact_summary_lifecycle_reduction_v1
