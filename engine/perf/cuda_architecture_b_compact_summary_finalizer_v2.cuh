#pragma once

#include "cuda_architecture_b_compact_summary_v2.cuh"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

namespace eiel::cuda_architecture_b::compact_summary_finalizer_v2 {

namespace abi = compact_summary_v2;

enum ProducerStatusCodeV2 : std::int32_t {
    kProducerStatusSuccess = 0,
    kProducerStatusScratchNotReady = 1,
    kProducerStatusReservedDefinedMask = 2,
    kProducerStatusStatusCodeRange = 3,
    kProducerStatusNullableStatusCode = 4,
    kProducerStatusPatchMask = 5,
    kProducerStatusIntegerRange = 6,
    kProducerStatusFloatNotFinite = 7,
    kProducerStatusFloatRange = 8,
    kProducerStatusDigestBytes = 9,
    kProducerStatusTextUndefinedRecord = 10,
    kProducerStatusTextLength = 11,
    kProducerStatusTextUtf8 = 12,
    kProducerStatusTextTail = 13,
    kProducerStatusTerminalRecoverable = 14,
    kProducerStatusTerminalFailure = 15,
    kProducerStatusTerminalStickyTail = 16,
};

inline constexpr std::int32_t kThreadsPerBlockV2 = 128;
inline constexpr std::uint64_t kPatchChildDefinedMaskV2 =
    0x00000000FFFC0000ULL;

struct KernelResourceReceiptV2 {
    std::int32_t binary_version;
    std::int32_t ptx_version;
    std::int32_t max_threads_per_block;
    std::int32_t registers_per_thread;
    std::int32_t occupancy_blocks_per_multiprocessor;
    std::int64_t constant_bytes;
    std::int64_t local_bytes_per_thread;
    std::int64_t shared_bytes_per_block;
};

void launch_compact_summary_finalizer_v2(
    abi::CompactSummaryViewV2 scratch,
    abi::CompactSummaryViewV2 output,
    std::int32_t* producer_status,
    std::int64_t generation,
    cudaStream_t stream
);

KernelResourceReceiptV2 query_compact_summary_finalizer_v2_resources();

#if defined(__CUDACC__)

#define EIEL_FINALIZER_DEVICE __device__ __forceinline__

EIEL_FINALIZER_DEVICE bool bit_is_set(
    const abi::DefinedMaskV1 mask,
    const abi::DefinedBitV1 bit
) noexcept {
    return abi::test_defined_bit(mask, bit);
}

EIEL_FINALIZER_DEVICE abi::DefinedBitV1 nullable_status_bit(
    const abi::StatusFieldV2 field
) noexcept {
    switch (field) {
        case abi::kStatusLineageMetricsLineageCollapseWarning:
            return abi::kDefinedLineageMetricsLineageCollapseWarning;
        case abi::kStatusTerminalRecoverableErrorCode:
            return abi::kDefinedTerminalRecoverableErrorCode;
        case abi::kStatusTerminalFailureStage:
            return abi::kDefinedTerminalFailureStage;
        default:
            return abi::kDefinedBitCountV1;
    }
}

EIEL_FINALIZER_DEVICE bool is_nullable_status(
    const abi::StatusFieldV2 field
) noexcept {
    return field == abi::kStatusLineageMetricsLineageCollapseWarning
        || field == abi::kStatusTerminalRecoverableErrorCode
        || field == abi::kStatusTerminalFailureStage;
}

EIEL_FINALIZER_DEVICE std::int32_t validate_status_codes(
    const abi::DefinedMaskV1 mask,
    const std::int32_t* const status_codes
) noexcept {
    for (std::int64_t index = 0; index < abi::kStatusFieldCountV2; ++index) {
        const auto field = static_cast<abi::StatusFieldV2>(index);
        const std::int32_t code = status_codes[index];
        if (!is_nullable_status(field)) {
            if (!abi::is_known_status_code(field, code)) {
                return kProducerStatusStatusCodeRange;
            }
            continue;
        }

        const bool defined = bit_is_set(mask, nullable_status_bit(field));
        if (!defined) {
            if (code != abi::kReservedInvalidStatusCodeV2) {
                return kProducerStatusNullableStatusCode;
            }
        } else if (code == abi::kReservedInvalidStatusCodeV2) {
            return kProducerStatusNullableStatusCode;
        } else if (!abi::is_known_status_code(field, code)) {
            return kProducerStatusStatusCodeRange;
        }
    }
    return kProducerStatusSuccess;
}

EIEL_FINALIZER_DEVICE std::int32_t validate_patch_mask(
    const abi::DefinedMaskV1 mask,
    const std::int32_t world_code
) noexcept {
    const bool patch_defined = bit_is_set(mask, abi::kDefinedPatchMetrics);
    const bool child_defined = (mask & kPatchChildDefinedMaskV2) != 0;
    if (!patch_defined && child_defined) {
        return kProducerStatusPatchMask;
    }
    if (world_code == abi::kWorldFlat) {
        return (patch_defined || child_defined)
            ? kProducerStatusPatchMask
            : kProducerStatusSuccess;
    }
    if (world_code == abi::kWorldPatch && !patch_defined) {
        return kProducerStatusPatchMask;
    }
    return kProducerStatusSuccess;
}

EIEL_FINALIZER_DEVICE bool integer_field_is_defined(
    const abi::DefinedMaskV1 mask,
    const abi::IntegerFieldV2 field
) noexcept {
    switch (field) {
        case abi::kIntegerIdentityActiveAgentIdMin:
            return bit_is_set(mask, abi::kDefinedIdentityActiveAgentIdMin);
        case abi::kIntegerIdentityActiveAgentIdMax:
            return bit_is_set(mask, abi::kDefinedIdentityActiveAgentIdMax);
        case abi::kIntegerTerminalFailureGeneration:
            return bit_is_set(mask, abi::kDefinedTerminalFailureGeneration);
        case abi::kIntegerTerminalFailureTick:
            return bit_is_set(mask, abi::kDefinedTerminalFailureTick);
        case abi::kIntegerLineageMetricsActiveLineages:
            return bit_is_set(mask, abi::kDefinedLineageMetricsActiveLineages);
        case abi::kIntegerLineageMetricsRarePersistentLineages:
            return bit_is_set(
                mask,
                abi::kDefinedLineageMetricsRarePersistentLineages
            );
        case abi::kIntegerLineageMetricsLineageCollapseStreak:
            return bit_is_set(
                mask,
                abi::kDefinedLineageMetricsLineageCollapseStreak
            );
        case abi::kIntegerPatchMetricsPatchMovementEnergyEventCount:
            return bit_is_set(mask, abi::kDefinedPatchMetrics);
        case abi::kIntegerPatchMetricsPatchSpawnSyncRows:
            return bit_is_set(
                mask,
                abi::kDefinedPatchMetricsPatchSpawnSyncRows
            );
        case abi::kIntegerPatchMetricsTypedDeathEventRows:
            return bit_is_set(
                mask,
                abi::kDefinedPatchMetricsTypedDeathEventRows
            );
        case abi::kIntegerPatchMetricsEnvRngPaddingDraws:
            return bit_is_set(
                mask,
                abi::kDefinedPatchMetricsEnvRngPaddingDraws
            );
        default:
            return true;
    }
}

EIEL_FINALIZER_DEVICE bool integer_field_is_positive(
    const abi::IntegerFieldV2 field
) noexcept {
    return field == abi::kIntegerGenerationEndTickExclusive
        || field == abi::kIntegerNextGenerationStartGeneration
        || field == abi::kIntegerIdentityPopulationNextId
        || field == abi::kIntegerGenerationMetricsCapacity;
}

EIEL_FINALIZER_DEVICE std::int32_t validate_integer_values(
    const abi::DefinedMaskV1 mask,
    const std::int64_t* const integer_values
) noexcept {
    for (std::int64_t index = 0; index < abi::kIntegerFieldCountV2; ++index) {
        const auto field = static_cast<abi::IntegerFieldV2>(index);
        if (!integer_field_is_defined(mask, field)) {
            continue;
        }
        const std::int64_t value = integer_values[index];
        if (field == abi::kIntegerGenerationStartTick
            || field == abi::kIntegerNextGenerationStartTick) {
            if (value != 0) {
                return kProducerStatusIntegerRange;
            }
        } else if (field == abi::kIntegerTurnoverCount) {
            if (value != 1) {
                return kProducerStatusIntegerRange;
            }
        } else if (integer_field_is_positive(field)) {
            if (value <= 0) {
                return kProducerStatusIntegerRange;
            }
        } else if (value < 0) {
            return kProducerStatusIntegerRange;
        }
    }
    if (integer_values[abi::kIntegerGenerationEndGeneration]
        == 0x7FFFFFFFFFFFFFFFLL) {
        return kProducerStatusIntegerRange;
    }
    return kProducerStatusSuccess;
}

EIEL_FINALIZER_DEVICE bool float_field_is_defined(
    const abi::DefinedMaskV1 mask,
    const abi::FloatFieldV2 field
) noexcept {
    switch (field) {
        case abi::kFloatLineageMetricsDominantLineagePopulationShare:
            return bit_is_set(
                mask,
                abi::kDefinedLineageMetricsDominantLineagePopulationShare
            );
        case abi::kFloatLineageMetricsTop3LineagePopulationShare:
            return bit_is_set(
                mask,
                abi::kDefinedLineageMetricsTop3LineagePopulationShare
            );
        case abi::kFloatLineageMetricsLineagePopulationEntropy:
            return bit_is_set(
                mask,
                abi::kDefinedLineageMetricsLineagePopulationEntropy
            );
        case abi::kFloatLineageMetricsLineageReproductionEntropy:
            return bit_is_set(
                mask,
                abi::kDefinedLineageMetricsLineageReproductionEntropy
            );
        case abi::kFloatLineageMetricsDominantLineageReproductionShare:
            return bit_is_set(
                mask,
                abi::kDefinedLineageMetricsDominantLineageReproductionShare
            );
        case abi::kFloatPatchMetricsPatchFoodHarvestTotal:
        case abi::kFloatPatchMetricsPatchScentEmitTotal:
        case abi::kFloatPatchMetricsPatchWasteEmitTotal:
        case abi::kFloatPatchMetricsPatchMovementEnergyTotal:
            return bit_is_set(mask, abi::kDefinedPatchMetrics);
        case abi::kFloatPatchMetricsEnergyCollectedDelta:
            return bit_is_set(
                mask,
                abi::kDefinedPatchMetricsEnergyCollectedDelta
            );
        case abi::kFloatPatchMetricsPatchScentMoveQueuedTotal:
            return bit_is_set(
                mask,
                abi::kDefinedPatchMetricsPatchScentMoveQueuedTotal
            );
        case abi::kFloatPatchMetricsPatchWasteMoveQueuedTotal:
            return bit_is_set(
                mask,
                abi::kDefinedPatchMetricsPatchWasteMoveQueuedTotal
            );
        case abi::kFloatPatchMetricsPatchScentGatherQueuedTotal:
            return bit_is_set(
                mask,
                abi::kDefinedPatchMetricsPatchScentGatherQueuedTotal
            );
        case abi::kFloatPatchMetricsPatchWasteGatherQueuedTotal:
            return bit_is_set(
                mask,
                abi::kDefinedPatchMetricsPatchWasteGatherQueuedTotal
            );
        case abi::kFloatPatchMetricsPatchMovementEnergyExtraTotal:
            return bit_is_set(
                mask,
                abi::kDefinedPatchMetricsPatchMovementEnergyExtraTotal
            );
        case abi::kFloatPatchMetricsPatchScentEmittedTotal:
            return bit_is_set(
                mask,
                abi::kDefinedPatchMetricsPatchScentEmittedTotal
            );
        case abi::kFloatPatchMetricsPatchWasteEmittedTotal:
            return bit_is_set(
                mask,
                abi::kDefinedPatchMetricsPatchWasteEmittedTotal
            );
        case abi::kFloatPatchMetricsPatchFoodDepletedFrac:
            return bit_is_set(
                mask,
                abi::kDefinedPatchMetricsPatchFoodDepletedFrac
            );
        case abi::kFloatPatchMetricsPatchFoodFullFrac:
            return bit_is_set(
                mask,
                abi::kDefinedPatchMetricsPatchFoodFullFrac
            );
        case abi::kFloatPatchMetricsPatchDensityHotspotFrac:
            return bit_is_set(
                mask,
                abi::kDefinedPatchMetricsPatchDensityHotspotFrac
            );
        default:
            return true;
    }
}

EIEL_FINALIZER_DEVICE bool double_is_finite(const double value) noexcept {
    const std::uint64_t bits =
        static_cast<std::uint64_t>(__double_as_longlong(value));
    return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
}

EIEL_FINALIZER_DEVICE bool float_field_is_ratio(
    const abi::FloatFieldV2 field
) noexcept {
    return field == abi::kFloatLineageMetricsDominantLineagePopulationShare
        || field == abi::kFloatLineageMetricsTop3LineagePopulationShare
        || field == abi::kFloatLineageMetricsDominantLineageReproductionShare
        || field == abi::kFloatPatchMetricsPatchFoodDepletedFrac
        || field == abi::kFloatPatchMetricsPatchFoodFullFrac
        || field == abi::kFloatPatchMetricsPatchDensityHotspotFrac;
}

EIEL_FINALIZER_DEVICE bool float_field_is_nonnegative(
    const abi::FloatFieldV2 field
) noexcept {
    return field == abi::kFloatGenerationMetricsPeakInjury
        || field == abi::kFloatPatchMetricsPatchFoodHarvestTotal
        || field == abi::kFloatPatchMetricsPatchScentEmitTotal
        || field == abi::kFloatPatchMetricsPatchWasteEmitTotal;
}

EIEL_FINALIZER_DEVICE std::int32_t validate_float_values(
    const abi::DefinedMaskV1 mask,
    const double* const float_values
) noexcept {
    for (std::int64_t index = 0; index < abi::kFloatFieldCountV2; ++index) {
        const double value = float_values[index];
        if (!double_is_finite(value)) {
            return kProducerStatusFloatNotFinite;
        }
        const auto field = static_cast<abi::FloatFieldV2>(index);
        if (!float_field_is_defined(mask, field)) {
            continue;
        }
        if (float_field_is_ratio(field) && (value < 0.0 || value > 1.0)) {
            return kProducerStatusFloatRange;
        }
        if (float_field_is_nonnegative(field) && value < 0.0) {
            return kProducerStatusFloatRange;
        }
    }
    return kProducerStatusSuccess;
}

EIEL_FINALIZER_DEVICE bool all_bytes_zero(
    const std::uint8_t* const values,
    const std::int64_t count
) noexcept {
    for (std::int64_t index = 0; index < count; ++index) {
        if (values[index] != 0) {
            return false;
        }
    }
    return true;
}

EIEL_FINALIZER_DEVICE std::int32_t validate_digest_bytes(
    const abi::DefinedMaskV1 mask,
    const std::uint8_t* const digest_values
) noexcept {
    if (!bit_is_set(mask, abi::kDefinedTerminalRecoverableErrorDigestSha256)) {
        const std::int64_t offset = abi::digest_field_offset(
            abi::kDigestTerminalRecoverableErrorDigestSha256
        );
        if (!all_bytes_zero(digest_values + offset, abi::kDigestBytesV2)) {
            return kProducerStatusDigestBytes;
        }
    }
    return kProducerStatusSuccess;
}

EIEL_FINALIZER_DEVICE bool is_utf8_continuation(
    const std::uint8_t value
) noexcept {
    return value >= 0x80 && value <= 0xBF;
}

EIEL_FINALIZER_DEVICE bool strict_utf8(
    const std::uint8_t* const payload,
    const std::int64_t length
) noexcept {
    std::int64_t index = 0;
    while (index < length) {
        const std::uint8_t first = payload[index];
        if (first <= 0x7F) {
            ++index;
            continue;
        }
        if (first >= 0xC2 && first <= 0xDF) {
            if (index + 1 >= length || !is_utf8_continuation(payload[index + 1])) {
                return false;
            }
            index += 2;
            continue;
        }
        if (first == 0xE0) {
            if (index + 2 >= length
                || payload[index + 1] < 0xA0
                || payload[index + 1] > 0xBF
                || !is_utf8_continuation(payload[index + 2])) {
                return false;
            }
            index += 3;
            continue;
        }
        if ((first >= 0xE1 && first <= 0xEC)
            || (first >= 0xEE && first <= 0xEF)) {
            if (index + 2 >= length
                || !is_utf8_continuation(payload[index + 1])
                || !is_utf8_continuation(payload[index + 2])) {
                return false;
            }
            index += 3;
            continue;
        }
        if (first == 0xED) {
            if (index + 2 >= length
                || payload[index + 1] < 0x80
                || payload[index + 1] > 0x9F
                || !is_utf8_continuation(payload[index + 2])) {
                return false;
            }
            index += 3;
            continue;
        }
        if (first == 0xF0) {
            if (index + 3 >= length
                || payload[index + 1] < 0x90
                || payload[index + 1] > 0xBF
                || !is_utf8_continuation(payload[index + 2])
                || !is_utf8_continuation(payload[index + 3])) {
                return false;
            }
            index += 4;
            continue;
        }
        if (first >= 0xF1 && first <= 0xF3) {
            if (index + 3 >= length
                || !is_utf8_continuation(payload[index + 1])
                || !is_utf8_continuation(payload[index + 2])
                || !is_utf8_continuation(payload[index + 3])) {
                return false;
            }
            index += 4;
            continue;
        }
        if (first == 0xF4) {
            if (index + 3 >= length
                || payload[index + 1] < 0x80
                || payload[index + 1] > 0x8F
                || !is_utf8_continuation(payload[index + 2])
                || !is_utf8_continuation(payload[index + 3])) {
                return false;
            }
            index += 4;
            continue;
        }
        return false;
    }
    return true;
}

EIEL_FINALIZER_DEVICE std::int32_t validate_text_record(
    const abi::DefinedMaskV1 mask,
    const std::uint8_t* const text_values
) noexcept {
    const bool defined = bit_is_set(
        mask,
        abi::kDefinedLineageMetricsDominantLineageId
    );
    if (!defined) {
        return all_bytes_zero(text_values, abi::kTextRecordBytesV2)
            ? kProducerStatusSuccess
            : kProducerStatusTextUndefinedRecord;
    }
    const std::int64_t length = text_values[abi::kTextLengthByteIndexV2];
    if (length <= 0 || length > abi::kTextMaximumPayloadBytesV2) {
        return kProducerStatusTextLength;
    }
    if (!strict_utf8(text_values + 1, length)) {
        return kProducerStatusTextUtf8;
    }
    for (std::int64_t index = length + 1;
         index < abi::kTextRecordBytesV2;
         ++index) {
        if (text_values[index] != 0) {
            return kProducerStatusTextTail;
        }
    }
    return kProducerStatusSuccess;
}

EIEL_FINALIZER_DEVICE std::int32_t validate_terminal_invariants(
    const abi::DefinedMaskV1 mask,
    const std::int32_t* const status_codes
) noexcept {
    const bool error_code_defined = bit_is_set(
        mask,
        abi::kDefinedTerminalRecoverableErrorCode
    );
    const bool error_digest_defined = bit_is_set(
        mask,
        abi::kDefinedTerminalRecoverableErrorDigestSha256
    );
    const std::int32_t recoverable =
        status_codes[abi::kStatusTerminalRecoverableErrorStatus];
    if ((recoverable == abi::kRecoverableErrorStatusNone
         && (error_code_defined || error_digest_defined))
        || (recoverable == abi::kRecoverableErrorStatusPresent
            && (!error_code_defined || !error_digest_defined))) {
        return kProducerStatusTerminalRecoverable;
    }

    const bool failure_generation_defined = bit_is_set(
        mask,
        abi::kDefinedTerminalFailureGeneration
    );
    const bool failure_tick_defined = bit_is_set(
        mask,
        abi::kDefinedTerminalFailureTick
    );
    const bool failure_stage_defined = bit_is_set(
        mask,
        abi::kDefinedTerminalFailureStage
    );
    const bool terminal_failed =
        status_codes[abi::kStatusTerminalStatus] == abi::kTerminalStatusFailed;
    const bool sticky_failure =
        status_codes[abi::kStatusTerminalStickyFailure]
        == abi::kBooleanStatusTrue;
    if (terminal_failed) {
        if (!sticky_failure
            || !failure_generation_defined
            || !failure_tick_defined
            || !failure_stage_defined) {
            return kProducerStatusTerminalFailure;
        }
    } else if (sticky_failure
               || failure_generation_defined
               || failure_tick_defined
               || failure_stage_defined) {
        return kProducerStatusTerminalFailure;
    }
    return kProducerStatusSuccess;
}

template <typename T>
EIEL_FINALIZER_DEVICE void copy_values(
    T* const destination,
    const T* const source,
    const std::int64_t count
) noexcept {
    for (std::int64_t index = 0; index < count; ++index) {
        destination[index] = source[index];
    }
}

EIEL_FINALIZER_DEVICE std::int32_t validate_scratch_row(
    const abi::DefinedMaskV1 mask,
    const std::int32_t* const status_codes,
    const std::int64_t* const integer_values,
    const double* const float_values,
    const std::uint8_t* const digest_values,
    const std::uint8_t* const text_values
) noexcept {
    if (!abi::defined_mask_has_only_frozen_bits(mask)) {
        return kProducerStatusReservedDefinedMask;
    }
    std::int32_t status = validate_status_codes(mask, status_codes);
    if (status != kProducerStatusSuccess) {
        return status;
    }
    status = validate_patch_mask(mask, status_codes[abi::kStatusWorld]);
    if (status != kProducerStatusSuccess) {
        return status;
    }
    status = validate_integer_values(mask, integer_values);
    if (status != kProducerStatusSuccess) {
        return status;
    }
    status = validate_float_values(mask, float_values);
    if (status != kProducerStatusSuccess) {
        return status;
    }
    status = validate_digest_bytes(mask, digest_values);
    if (status != kProducerStatusSuccess) {
        return status;
    }
    status = validate_text_record(mask, text_values);
    if (status != kProducerStatusSuccess) {
        return status;
    }
    return validate_terminal_invariants(mask, status_codes);
}

EIEL_FINALIZER_DEVICE void finalize_cell(
    const abi::CompactSummaryViewV2 scratch,
    const abi::CompactSummaryViewV2 output,
    std::int32_t* const producer_status,
    const std::int64_t generation,
    const std::int64_t cell
) noexcept {
    if (producer_status[cell] != kProducerStatusSuccess) {
        return;
    }

    const std::uint8_t* const scratch_validity =
        abi::row_pointer(scratch.validity, 0, cell);
    if (scratch_validity[0] != 1) {
        producer_status[cell] = kProducerStatusScratchNotReady;
        return;
    }

    const abi::DefinedMaskV1* const scratch_mask =
        abi::row_pointer(scratch.defined_mask, 0, cell);
    const std::int32_t* const scratch_status =
        abi::row_pointer(scratch.status_codes, 0, cell);
    const std::int64_t* const scratch_integer =
        abi::row_pointer(scratch.integer_values, 0, cell);
    const double* const scratch_float =
        abi::row_pointer(scratch.float_values, 0, cell);
    const std::uint8_t* const scratch_digest =
        abi::row_pointer(scratch.digest_values, 0, cell);
    const std::uint8_t* const scratch_text =
        abi::row_pointer(scratch.text_values, 0, cell);

    const std::int32_t validation_status = validate_scratch_row(
        scratch_mask[0],
        scratch_status,
        scratch_integer,
        scratch_float,
        scratch_digest,
        scratch_text
    );
    if (validation_status != kProducerStatusSuccess) {
        producer_status[cell] = validation_status;
        return;
    }

    abi::DefinedMaskV1* const output_mask =
        abi::row_pointer(output.defined_mask, generation, cell);
    std::int32_t* const output_status =
        abi::row_pointer(output.status_codes, generation, cell);
    std::int64_t* const output_integer =
        abi::row_pointer(output.integer_values, generation, cell);
    double* const output_float =
        abi::row_pointer(output.float_values, generation, cell);
    std::uint8_t* const output_digest =
        abi::row_pointer(output.digest_values, generation, cell);
    std::uint8_t* const output_text =
        abi::row_pointer(output.text_values, generation, cell);

    output_mask[0] = scratch_mask[0];
    copy_values(output_status, scratch_status, abi::kStatusFieldCountV2);
    copy_values(output_integer, scratch_integer, abi::kIntegerFieldCountV2);
    copy_values(output_float, scratch_float, abi::kFloatFieldCountV2);
    copy_values(
        output_digest,
        scratch_digest,
        abi::kDigestFieldCountV2 * abi::kDigestBytesV2
    );
    copy_values(
        output_text,
        scratch_text,
        abi::kTextFieldCountV2 * abi::kTextRecordBytesV2
    );
    if (scratch_status[abi::kStatusTerminalStatus]
            == abi::kTerminalStatusFailed
        && scratch_status[abi::kStatusTerminalStickyFailure]
            == abi::kBooleanStatusTrue) {
        producer_status[cell] = kProducerStatusTerminalStickyTail;
    }
    __threadfence();
    abi::row_pointer(output.validity, generation, cell)[0] = 1;
}

#undef EIEL_FINALIZER_DEVICE

#endif  // defined(__CUDACC__)

static_assert(kPatchChildDefinedMaskV2 == 0x00000000FFFC0000ULL);
static_assert(std::is_standard_layout_v<KernelResourceReceiptV2>);
static_assert(std::is_trivially_copyable_v<KernelResourceReceiptV2>);

}  // namespace eiel::cuda_architecture_b::compact_summary_finalizer_v2
