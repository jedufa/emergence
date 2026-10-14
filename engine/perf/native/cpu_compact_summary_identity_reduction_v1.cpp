#include "cpu_compact_summary_identity_reduction_v1.hpp"

#include <cstdint>
#include <limits>

namespace eiel::native::cpu_compact_summary_identity_reduction_v1 {
namespace {

IdentityReductionStatusV1 map_digest_status(
    const digests::DigestStatusV1 status
) noexcept {
    switch (status) {
        case digests::DigestStatusV1::kSuccess:
            return IdentityReductionStatusV1::kSuccess;
        case digests::DigestStatusV1::kNonLittleEndianHost:
            return IdentityReductionStatusV1::kNonLittleEndianHost;
        case digests::DigestStatusV1::kDtypeMismatch:
            return IdentityReductionStatusV1::kDtypeMismatch;
        case digests::DigestStatusV1::kRankMismatch:
            return IdentityReductionStatusV1::kRankMismatch;
        case digests::DigestStatusV1::kNegativeExtent:
            return IdentityReductionStatusV1::kNegativeExtent;
        case digests::DigestStatusV1::kNonContiguous:
            return IdentityReductionStatusV1::kNonContiguous;
        case digests::DigestStatusV1::kStorageSizeMismatch:
            return IdentityReductionStatusV1::kStorageSizeMismatch;
        case digests::DigestStatusV1::kNullData:
            return IdentityReductionStatusV1::kNullData;
        case digests::DigestStatusV1::kSizeOverflow:
            return IdentityReductionStatusV1::kSizeOverflow;
        case digests::DigestStatusV1::kActiveAgentIdInvalid:
            return IdentityReductionStatusV1::kActiveAgentIdInvalid;
        case digests::DigestStatusV1::kColumnLengthMismatch:
        case digests::DigestStatusV1::kCursorOutOfRange:
        case digests::DigestStatusV1::kSequenceMismatch:
        case digests::DigestStatusV1::kRowOrderInvalid:
        case digests::DigestStatusV1::kNonFiniteValue:
        case digests::DigestStatusV1::kPolicyMaskShapeMismatch:
        case digests::DigestStatusV1::kBooleanValueInvalid:
        case digests::DigestStatusV1::kInvalidMode:
        case digests::DigestStatusV1::kInvalidRowLayout:
        case digests::DigestStatusV1::kNullActiveSource:
        case digests::DigestStatusV1::kActiveLengthOutOfRange:
            return IdentityReductionStatusV1::kUnexpectedDigestFailure;
    }
    return IdentityReductionStatusV1::kUnexpectedDigestFailure;
}

IdentityReductionStatusV1 fail_candidate(
    IdentityReductionResultV1& candidate,
    const IdentityReductionStatusV1 status
) noexcept {
    clear_identity_reduction_result(candidate);
    return status;
}

}  // namespace

void clear_identity_reduction_result(
    IdentityReductionResultV1& output
) noexcept {
    output = IdentityReductionResultV1{};
}

IdentityReductionStatusV1 validate_identity_reduction_result(
    const std::int64_t capacity,
    IdentityReductionResultV1& candidate
) noexcept {
    if (capacity < 0) {
        return fail_candidate(
            candidate, IdentityReductionStatusV1::kCapacityOutOfRange
        );
    }
    if (candidate.population_next_id < 0) {
        return fail_candidate(
            candidate,
            IdentityReductionStatusV1::kPopulationNextIdOutOfRange
        );
    }
    if (candidate.active_agent_count < 0) {
        return fail_candidate(
            candidate, IdentityReductionStatusV1::kActiveCountOutOfRange
        );
    }
    if (candidate.active_agent_count > capacity) {
        return fail_candidate(
            candidate,
            IdentityReductionStatusV1::kActiveCountExceedsCapacity
        );
    }
    if (
        candidate.free_slot_count < 0
        || candidate.free_slot_count > capacity
        || candidate.active_agent_count
            > std::numeric_limits<std::int64_t>::max()
                - candidate.free_slot_count
        || candidate.active_agent_count + candidate.free_slot_count != capacity
    ) {
        return fail_candidate(
            candidate, IdentityReductionStatusV1::kSlotConservationInvalid
        );
    }
    if (
        candidate.active_agent_id_min_defined > 1U
        || candidate.active_agent_id_max_defined > 1U
    ) {
        return fail_candidate(
            candidate, IdentityReductionStatusV1::kDefinedFlagInvalid
        );
    }

    const bool minimum_defined =
        candidate.active_agent_id_min_defined == 1U;
    const bool maximum_defined =
        candidate.active_agent_id_max_defined == 1U;
    const bool nonempty = candidate.active_agent_count != 0;
    if (
        minimum_defined != maximum_defined
        || minimum_defined != nonempty
        || (!nonempty
            && (candidate.active_agent_id_min != 0
                || candidate.active_agent_id_max != 0))
    ) {
        return fail_candidate(
            candidate, IdentityReductionStatusV1::kDefinedStateInvalid
        );
    }
    if (!nonempty) {
        return IdentityReductionStatusV1::kSuccess;
    }
    if (
        candidate.active_agent_id_min < 0
        || candidate.active_agent_id_max < candidate.active_agent_id_min
        || (candidate.active_agent_count == 1
            && candidate.active_agent_id_min != candidate.active_agent_id_max)
        || (candidate.active_agent_count > 1
            && candidate.active_agent_id_min >= candidate.active_agent_id_max)
    ) {
        return fail_candidate(
            candidate, IdentityReductionStatusV1::kActiveAgentIdBoundsInvalid
        );
    }
    if (candidate.population_next_id <= candidate.active_agent_id_max) {
        return fail_candidate(
            candidate,
            IdentityReductionStatusV1::kPopulationNextIdNotAboveMaximum
        );
    }
    return IdentityReductionStatusV1::kSuccess;
}

IdentityReductionStatusV1 reduce_identity(
    const digests::TensorViewV1& active_agent_ids,
    const CheckedInt64V1 capacity,
    const CheckedInt64V1 population_next_id,
    IdentityReductionResultV1& output
) noexcept {
    clear_identity_reduction_result(output);
    if (!capacity.in_range || capacity.value < 0) {
        return IdentityReductionStatusV1::kCapacityOutOfRange;
    }
    if (!population_next_id.in_range || population_next_id.value < 0) {
        return IdentityReductionStatusV1::kPopulationNextIdOutOfRange;
    }

    IdentityReductionResultV1 candidate{};
    const digests::DigestStatusV1 digest_status =
        digests::active_agent_ids_digest(
            active_agent_ids, candidate.active_agent_ids_digest
        );
    if (digest_status != digests::DigestStatusV1::kSuccess) {
        return map_digest_status(digest_status);
    }

    candidate.active_agent_count = active_agent_ids.extent0;
    if (candidate.active_agent_count > capacity.value) {
        clear_identity_reduction_result(candidate);
        return IdentityReductionStatusV1::kActiveCountExceedsCapacity;
    }
    candidate.free_slot_count = capacity.value - candidate.active_agent_count;
    candidate.population_next_id = population_next_id.value;
    if (candidate.active_agent_count > 0) {
        const auto* const ids =
            static_cast<const std::int64_t*>(active_agent_ids.data);
        candidate.active_agent_id_min = ids[0];
        candidate.active_agent_id_max = ids[candidate.active_agent_count - 1];
        candidate.active_agent_id_min_defined = 1U;
        candidate.active_agent_id_max_defined = 1U;
    }

    const IdentityReductionStatusV1 validation =
        validate_identity_reduction_result(capacity.value, candidate);
    if (validation != IdentityReductionStatusV1::kSuccess) {
        return validation;
    }
    output = candidate;
    return IdentityReductionStatusV1::kSuccess;
}

IdentityReductionStatusV1 reduce_identity(
    const digests::TensorViewV1& active_agent_ids,
    const std::int64_t capacity,
    const std::int64_t population_next_id,
    IdentityReductionResultV1& output
) noexcept {
    return reduce_identity(
        active_agent_ids,
        CheckedInt64V1{capacity, true},
        CheckedInt64V1{population_next_id, true},
        output
    );
}

const char* identity_reduction_status_name(
    const IdentityReductionStatusV1 status
) noexcept {
    switch (status) {
        case IdentityReductionStatusV1::kSuccess: return "success";
        case IdentityReductionStatusV1::kNonLittleEndianHost:
            return "non_little_endian_host";
        case IdentityReductionStatusV1::kDtypeMismatch:
            return "dtype_mismatch";
        case IdentityReductionStatusV1::kRankMismatch:
            return "rank_mismatch";
        case IdentityReductionStatusV1::kNegativeExtent:
            return "negative_extent";
        case IdentityReductionStatusV1::kNonContiguous:
            return "non_contiguous";
        case IdentityReductionStatusV1::kStorageSizeMismatch:
            return "storage_size_mismatch";
        case IdentityReductionStatusV1::kNullData: return "null_data";
        case IdentityReductionStatusV1::kSizeOverflow: return "size_overflow";
        case IdentityReductionStatusV1::kActiveAgentIdInvalid:
            return "active_agent_id_invalid";
        case IdentityReductionStatusV1::kCapacityOutOfRange:
            return "capacity_out_of_range";
        case IdentityReductionStatusV1::kPopulationNextIdOutOfRange:
            return "population_next_id_out_of_range";
        case IdentityReductionStatusV1::kActiveCountOutOfRange:
            return "active_count_out_of_range";
        case IdentityReductionStatusV1::kActiveCountExceedsCapacity:
            return "active_count_exceeds_capacity";
        case IdentityReductionStatusV1::kSlotConservationInvalid:
            return "slot_conservation_invalid";
        case IdentityReductionStatusV1::kDefinedFlagInvalid:
            return "defined_flag_invalid";
        case IdentityReductionStatusV1::kDefinedStateInvalid:
            return "defined_state_invalid";
        case IdentityReductionStatusV1::kActiveAgentIdBoundsInvalid:
            return "active_agent_id_bounds_invalid";
        case IdentityReductionStatusV1::kPopulationNextIdNotAboveMaximum:
            return "population_next_id_not_above_maximum";
        case IdentityReductionStatusV1::kUnexpectedDigestFailure:
            return "unexpected_digest_failure";
    }
    return "unknown";
}

}  // namespace eiel::native::cpu_compact_summary_identity_reduction_v1
