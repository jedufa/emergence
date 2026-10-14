#include "cuda_architecture_b_compact_summary_source_projection_v1.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace projection =
    eiel::cuda_architecture_b::compact_summary_source_projection_v1;
namespace abi = eiel::cuda_architecture_b::compact_summary_v2;

namespace {

using Authority = projection::CompactSummaryGenerationCellAuthorityV1;

__device__ __forceinline__ bool bytes_are_zero(
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

__device__ __forceinline__ std::uint64_t double_bits(
    const double value
) noexcept {
    return static_cast<std::uint64_t>(__double_as_longlong(value));
}

__device__ __forceinline__ bool double_is_finite(
    const double value
) noexcept {
    return (double_bits(value) & 0x7FF0000000000000ULL)
        != 0x7FF0000000000000ULL;
}

__device__ __forceinline__ bool bit_is_set(
    const abi::DefinedMaskV1 mask,
    const abi::DefinedBitV1 bit
) noexcept {
    return abi::test_defined_bit(mask, bit);
}

__device__ __forceinline__ bool is_complete_disposition(
    const std::uint32_t disposition
) noexcept {
    return disposition == projection::kAuthorityRowComplete;
}

__device__ __forceinline__ bool is_noop_disposition(
    const std::uint32_t disposition
) noexcept {
    return disposition == projection::kAuthorityRowFailedGeneration
        || disposition == projection::kAuthorityRowStickyNoop
        || disposition == projection::kAuthorityRowUnexecuted;
}

__device__ __forceinline__ std::int32_t noop_status(
    const std::uint32_t disposition
) noexcept {
    if (disposition == projection::kAuthorityRowFailedGeneration) {
        return projection::kProjectionFailedGenerationNoop;
    }
    if (disposition == projection::kAuthorityRowStickyNoop) {
        return projection::kProjectionStickyNoop;
    }
    return projection::kProjectionUnexecutedNoop;
}

__device__ __forceinline__ abi::DefinedBitV1 nullable_status_bit(
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

__device__ __forceinline__ std::int32_t validate_status_codes(
    const Authority& source
) noexcept {
    for (std::int64_t index = 0; index < abi::kStatusFieldCountV2; ++index) {
        const auto field = static_cast<abi::StatusFieldV2>(index);
        const abi::DefinedBitV1 nullable_bit = nullable_status_bit(field);
        const std::int32_t code = source.status_codes[index];
        if (nullable_bit == abi::kDefinedBitCountV1) {
            if (!abi::is_known_status_code(field, code)) {
                return projection::kProjectionStatusCodeInvalid;
            }
            continue;
        }
        const bool defined = bit_is_set(source.defined_mask, nullable_bit);
        if ((!defined && code != abi::kReservedInvalidStatusCodeV2)
            || (defined && !abi::is_known_status_code(field, code))) {
            return projection::kProjectionStatusCodeInvalid;
        }
    }
    return projection::kProjectionSuccess;
}

__device__ __forceinline__ bool integer_field_is_defined(
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
            return bit_is_set(mask, abi::kDefinedLineageMetricsRarePersistentLineages);
        case abi::kIntegerLineageMetricsLineageCollapseStreak:
            return bit_is_set(mask, abi::kDefinedLineageMetricsLineageCollapseStreak);
        case abi::kIntegerPatchMetricsPatchMovementEnergyEventCount:
            return bit_is_set(mask, abi::kDefinedPatchMetrics);
        case abi::kIntegerPatchMetricsPatchSpawnSyncRows:
            return bit_is_set(mask, abi::kDefinedPatchMetricsPatchSpawnSyncRows);
        case abi::kIntegerPatchMetricsTypedDeathEventRows:
            return bit_is_set(mask, abi::kDefinedPatchMetricsTypedDeathEventRows);
        case abi::kIntegerPatchMetricsEnvRngPaddingDraws:
            return bit_is_set(mask, abi::kDefinedPatchMetricsEnvRngPaddingDraws);
        default:
            return true;
    }
}

__device__ __forceinline__ bool integer_field_is_positive(
    const abi::IntegerFieldV2 field
) noexcept {
    return field == abi::kIntegerGenerationEndTickExclusive
        || field == abi::kIntegerNextGenerationStartGeneration
        || field == abi::kIntegerIdentityPopulationNextId
        || field == abi::kIntegerGenerationMetricsCapacity;
}

__device__ __forceinline__ std::int32_t validate_integer_values(
    const Authority& source
) noexcept {
    for (std::int64_t index = 0; index < abi::kIntegerFieldCountV2; ++index) {
        const auto field = static_cast<abi::IntegerFieldV2>(index);
        const std::int64_t value = source.integer_values[index];
        if (!integer_field_is_defined(source.defined_mask, field)) {
            if (value != 0) {
                return projection::kProjectionUndefinedStorageNonzero;
            }
            continue;
        }
        if (field == abi::kIntegerGenerationStartTick
            || field == abi::kIntegerNextGenerationStartTick) {
            if (value != 0) {
                return projection::kProjectionIntegerRangeInvalid;
            }
        } else if (field == abi::kIntegerTurnoverCount) {
            if (value != 1) {
                return projection::kProjectionIntegerRangeInvalid;
            }
        } else if (integer_field_is_positive(field)) {
            if (value <= 0) {
                return projection::kProjectionIntegerRangeInvalid;
            }
        } else if (value < 0) {
            return projection::kProjectionIntegerRangeInvalid;
        }
    }
    if (source.integer_values[abi::kIntegerGenerationEndGeneration]
        == 0x7FFFFFFFFFFFFFFFLL) {
        return projection::kProjectionIntegerRangeInvalid;
    }
    return projection::kProjectionSuccess;
}

__device__ __forceinline__ bool float_field_is_defined(
    const abi::DefinedMaskV1 mask,
    const abi::FloatFieldV2 field
) noexcept {
    switch (field) {
        case abi::kFloatLineageMetricsDominantLineagePopulationShare:
            return bit_is_set(mask, abi::kDefinedLineageMetricsDominantLineagePopulationShare);
        case abi::kFloatLineageMetricsTop3LineagePopulationShare:
            return bit_is_set(mask, abi::kDefinedLineageMetricsTop3LineagePopulationShare);
        case abi::kFloatLineageMetricsLineagePopulationEntropy:
            return bit_is_set(mask, abi::kDefinedLineageMetricsLineagePopulationEntropy);
        case abi::kFloatLineageMetricsLineageReproductionEntropy:
            return bit_is_set(mask, abi::kDefinedLineageMetricsLineageReproductionEntropy);
        case abi::kFloatLineageMetricsDominantLineageReproductionShare:
            return bit_is_set(mask, abi::kDefinedLineageMetricsDominantLineageReproductionShare);
        case abi::kFloatPatchMetricsPatchFoodHarvestTotal:
        case abi::kFloatPatchMetricsPatchScentEmitTotal:
        case abi::kFloatPatchMetricsPatchWasteEmitTotal:
        case abi::kFloatPatchMetricsPatchMovementEnergyTotal:
            return bit_is_set(mask, abi::kDefinedPatchMetrics);
        case abi::kFloatPatchMetricsEnergyCollectedDelta:
            return bit_is_set(mask, abi::kDefinedPatchMetricsEnergyCollectedDelta);
        case abi::kFloatPatchMetricsPatchScentMoveQueuedTotal:
            return bit_is_set(mask, abi::kDefinedPatchMetricsPatchScentMoveQueuedTotal);
        case abi::kFloatPatchMetricsPatchWasteMoveQueuedTotal:
            return bit_is_set(mask, abi::kDefinedPatchMetricsPatchWasteMoveQueuedTotal);
        case abi::kFloatPatchMetricsPatchScentGatherQueuedTotal:
            return bit_is_set(mask, abi::kDefinedPatchMetricsPatchScentGatherQueuedTotal);
        case abi::kFloatPatchMetricsPatchWasteGatherQueuedTotal:
            return bit_is_set(mask, abi::kDefinedPatchMetricsPatchWasteGatherQueuedTotal);
        case abi::kFloatPatchMetricsPatchMovementEnergyExtraTotal:
            return bit_is_set(mask, abi::kDefinedPatchMetricsPatchMovementEnergyExtraTotal);
        case abi::kFloatPatchMetricsPatchScentEmittedTotal:
            return bit_is_set(mask, abi::kDefinedPatchMetricsPatchScentEmittedTotal);
        case abi::kFloatPatchMetricsPatchWasteEmittedTotal:
            return bit_is_set(mask, abi::kDefinedPatchMetricsPatchWasteEmittedTotal);
        case abi::kFloatPatchMetricsPatchFoodDepletedFrac:
            return bit_is_set(mask, abi::kDefinedPatchMetricsPatchFoodDepletedFrac);
        case abi::kFloatPatchMetricsPatchFoodFullFrac:
            return bit_is_set(mask, abi::kDefinedPatchMetricsPatchFoodFullFrac);
        case abi::kFloatPatchMetricsPatchDensityHotspotFrac:
            return bit_is_set(mask, abi::kDefinedPatchMetricsPatchDensityHotspotFrac);
        default:
            return true;
    }
}

__device__ __forceinline__ bool float_field_is_ratio(
    const abi::FloatFieldV2 field
) noexcept {
    return field == abi::kFloatLineageMetricsDominantLineagePopulationShare
        || field == abi::kFloatLineageMetricsTop3LineagePopulationShare
        || field == abi::kFloatLineageMetricsDominantLineageReproductionShare
        || field == abi::kFloatPatchMetricsPatchFoodDepletedFrac
        || field == abi::kFloatPatchMetricsPatchFoodFullFrac
        || field == abi::kFloatPatchMetricsPatchDensityHotspotFrac;
}

__device__ __forceinline__ bool float_field_is_nonnegative(
    const abi::FloatFieldV2 field
) noexcept {
    return field == abi::kFloatGenerationMetricsPeakInjury
        || field == abi::kFloatPatchMetricsPatchFoodHarvestTotal
        || field == abi::kFloatPatchMetricsPatchScentEmitTotal
        || field == abi::kFloatPatchMetricsPatchWasteEmitTotal
        || field == abi::kFloatPatchMetricsPatchMovementEnergyTotal;
}

__device__ __forceinline__ std::int32_t validate_float_values(
    const Authority& source
) noexcept {
    for (std::int64_t index = 0; index < abi::kFloatFieldCountV2; ++index) {
        const double value = source.float_values[index];
        if (!double_is_finite(value)) {
            return projection::kProjectionFloatNonFinite;
        }
        const auto field = static_cast<abi::FloatFieldV2>(index);
        if (!float_field_is_defined(source.defined_mask, field)) {
            if (double_bits(value) != 0) {
                return projection::kProjectionUndefinedStorageNonzero;
            }
            continue;
        }
        if (float_field_is_ratio(field) && (value < 0.0 || value > 1.0)) {
            return projection::kProjectionFloatRangeInvalid;
        }
        if (float_field_is_nonnegative(field) && value < 0.0) {
            return projection::kProjectionFloatRangeInvalid;
        }
    }
    return projection::kProjectionSuccess;
}

__device__ __forceinline__ bool utf8_continuation(
    const std::uint8_t value
) noexcept {
    return value >= 0x80 && value <= 0xBF;
}

__device__ __forceinline__ bool strict_utf8(
    const std::uint8_t* const payload,
    const std::int64_t length
) noexcept {
    std::int64_t index = 0;
    while (index < length) {
        const std::uint8_t first = payload[index];
        if (first <= 0x7F) {
            ++index;
        } else if (first >= 0xC2 && first <= 0xDF) {
            if (index + 1 >= length || !utf8_continuation(payload[index + 1])) {
                return false;
            }
            index += 2;
        } else if (first == 0xE0) {
            if (index + 2 >= length || payload[index + 1] < 0xA0
                || payload[index + 1] > 0xBF
                || !utf8_continuation(payload[index + 2])) {
                return false;
            }
            index += 3;
        } else if ((first >= 0xE1 && first <= 0xEC)
                   || (first >= 0xEE && first <= 0xEF)) {
            if (index + 2 >= length
                || !utf8_continuation(payload[index + 1])
                || !utf8_continuation(payload[index + 2])) {
                return false;
            }
            index += 3;
        } else if (first == 0xED) {
            if (index + 2 >= length || payload[index + 1] < 0x80
                || payload[index + 1] > 0x9F
                || !utf8_continuation(payload[index + 2])) {
                return false;
            }
            index += 3;
        } else if (first == 0xF0) {
            if (index + 3 >= length || payload[index + 1] < 0x90
                || payload[index + 1] > 0xBF
                || !utf8_continuation(payload[index + 2])
                || !utf8_continuation(payload[index + 3])) {
                return false;
            }
            index += 4;
        } else if (first >= 0xF1 && first <= 0xF3) {
            if (index + 3 >= length
                || !utf8_continuation(payload[index + 1])
                || !utf8_continuation(payload[index + 2])
                || !utf8_continuation(payload[index + 3])) {
                return false;
            }
            index += 4;
        } else if (first == 0xF4) {
            if (index + 3 >= length || payload[index + 1] < 0x80
                || payload[index + 1] > 0x8F
                || !utf8_continuation(payload[index + 2])
                || !utf8_continuation(payload[index + 3])) {
                return false;
            }
            index += 4;
        } else {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ std::int32_t validate_digest_and_text(
    const Authority& source
) noexcept {
    const std::int64_t recoverable_digest_offset = abi::digest_field_offset(
        abi::kDigestTerminalRecoverableErrorDigestSha256
    );
    if (!bit_is_set(
            source.defined_mask,
            abi::kDefinedTerminalRecoverableErrorDigestSha256
        )
        && !bytes_are_zero(
            source.digest_values + recoverable_digest_offset,
            abi::kDigestBytesV2
        )) {
        return projection::kProjectionDigestUndefinedStorageNonzero;
    }

    const bool text_defined = bit_is_set(
        source.defined_mask,
        abi::kDefinedLineageMetricsDominantLineageId
    );
    if (!text_defined) {
        return bytes_are_zero(source.text_values, abi::kTextRecordBytesV2)
            ? projection::kProjectionSuccess
            : projection::kProjectionUndefinedStorageNonzero;
    }
    const std::int64_t length =
        source.text_values[abi::kTextLengthByteIndexV2];
    if (length <= 0 || length > abi::kTextMaximumPayloadBytesV2) {
        return projection::kProjectionTextLengthInvalid;
    }
    if (!strict_utf8(source.text_values + 1, length)) {
        return projection::kProjectionTextUtf8Invalid;
    }
    if (!bytes_are_zero(
            source.text_values + length + 1,
            abi::kTextRecordBytesV2 - length - 1
        )) {
        return projection::kProjectionTextTailNonzero;
    }
    return projection::kProjectionSuccess;
}

__device__ __forceinline__ std::int32_t validate_patch_mask(
    const Authority& source
) noexcept {
    constexpr abi::DefinedMaskV1 kPatchChildMask = 0x00000000FFFC0000ULL;
    const bool patch_defined = bit_is_set(
        source.defined_mask,
        abi::kDefinedPatchMetrics
    );
    const bool child_defined = (source.defined_mask & kPatchChildMask) != 0;
    const std::int32_t world = source.status_codes[abi::kStatusWorld];
    if ((!patch_defined && child_defined)
        || (world == abi::kWorldFlat && (patch_defined || child_defined))
        || (world == abi::kWorldPatch && !patch_defined)) {
        return projection::kProjectionPatchMaskInvalid;
    }
    return projection::kProjectionSuccess;
}

__device__ __forceinline__ void clear_output_row(
    const abi::CompactSummaryViewV2& output,
    const std::int64_t generation,
    const std::int64_t cell
) noexcept {
    *abi::row_pointer(output.validity, generation, cell) = 0;
    *abi::row_pointer(output.defined_mask, generation, cell) = 0;
    std::int32_t* const statuses =
        abi::row_pointer(output.status_codes, generation, cell);
    std::int64_t* const integers =
        abi::row_pointer(output.integer_values, generation, cell);
    double* const floats = abi::row_pointer(output.float_values, generation, cell);
    std::uint8_t* const digests =
        abi::row_pointer(output.digest_values, generation, cell);
    std::uint8_t* const text = abi::row_pointer(output.text_values, generation, cell);
    for (std::int64_t index = 0; index < abi::kStatusFieldCountV2; ++index) {
        statuses[index] = 0;
    }
    for (std::int64_t index = 0; index < abi::kIntegerFieldCountV2; ++index) {
        integers[index] = 0;
    }
    for (std::int64_t index = 0; index < abi::kFloatFieldCountV2; ++index) {
        floats[index] = 0.0;
    }
    for (std::int64_t index = 0;
         index < abi::kDigestFieldCountV2 * abi::kDigestBytesV2;
         ++index) {
        digests[index] = 0;
    }
    for (std::int64_t index = 0;
         index < abi::kTextFieldCountV2 * abi::kTextRecordBytesV2;
         ++index) {
        text[index] = 0;
    }
}

__device__ __forceinline__ void copy_output_row(
    const Authority& source,
    const abi::CompactSummaryViewV2& output,
    const std::int64_t generation,
    const std::int64_t cell
) noexcept {
    *abi::row_pointer(output.defined_mask, generation, cell) = source.defined_mask;
    std::int32_t* const statuses =
        abi::row_pointer(output.status_codes, generation, cell);
    std::int64_t* const integers =
        abi::row_pointer(output.integer_values, generation, cell);
    double* const floats = abi::row_pointer(output.float_values, generation, cell);
    std::uint8_t* const digests =
        abi::row_pointer(output.digest_values, generation, cell);
    std::uint8_t* const text = abi::row_pointer(output.text_values, generation, cell);
    for (std::int64_t index = 0; index < abi::kStatusFieldCountV2; ++index) {
        statuses[index] = source.status_codes[index];
    }
    for (std::int64_t index = 0; index < abi::kIntegerFieldCountV2; ++index) {
        integers[index] = source.integer_values[index];
    }
    for (std::int64_t index = 0; index < abi::kFloatFieldCountV2; ++index) {
        floats[index] = source.float_values[index];
    }
    for (std::int64_t index = 0;
         index < abi::kDigestFieldCountV2 * abi::kDigestBytesV2;
         ++index) {
        digests[index] = source.digest_values[index];
    }
    for (std::int64_t index = 0;
         index < abi::kTextFieldCountV2 * abi::kTextRecordBytesV2;
         ++index) {
        text[index] = source.text_values[index];
    }
    *abi::row_pointer(output.validity, generation, cell) = source.validity;
}

__global__ void compact_summary_source_projection_v1_kernel(
    const Authority* const authority_rows,
    const std::int64_t authority_row_count,
    const abi::CompactSummaryViewV2 output,
    std::int32_t* const producer_status
) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (row >= authority_row_count) {
        return;
    }
    const std::int64_t generation = row / output.cell_count;
    const std::int64_t cell = row - generation * output.cell_count;
    producer_status[row] = projection::kProjectionFailClosed;
    clear_output_row(output, generation, cell);

    const Authority& source = authority_rows[row];
    if (source.source_version != projection::kAuthoritySourceVersionV1) {
        producer_status[row] = projection::kProjectionSourceVersionMismatch;
        return;
    }
    if (!is_complete_disposition(source.disposition)
        && !is_noop_disposition(source.disposition)) {
        producer_status[row] = projection::kProjectionDispositionInvalid;
        return;
    }
    if (source.generation_index != generation || source.cell_index != cell) {
        producer_status[row] = projection::kProjectionIndexInvalid;
        return;
    }
    if (!bytes_are_zero(source.header_reserved, 7)
        || !bytes_are_zero(source.tail_reserved, 7)) {
        producer_status[row] = projection::kProjectionReservedBytesNonzero;
        return;
    }
    if (is_noop_disposition(source.disposition)) {
        producer_status[row] = source.validity == 0
            ? noop_status(source.disposition)
            : projection::kProjectionValidityInvalid;
        return;
    }
    if (source.validity != 1) {
        producer_status[row] = projection::kProjectionValidityInvalid;
        return;
    }
    if (!abi::defined_mask_has_only_frozen_bits(source.defined_mask)) {
        producer_status[row] = projection::kProjectionReservedDefinedMask;
        return;
    }

    std::int32_t status = validate_status_codes(source);
    if (status == projection::kProjectionSuccess) {
        status = validate_patch_mask(source);
    }
    if (status == projection::kProjectionSuccess) {
        status = validate_integer_values(source);
    }
    if (status == projection::kProjectionSuccess) {
        status = validate_float_values(source);
    }
    if (status == projection::kProjectionSuccess) {
        status = validate_digest_and_text(source);
    }
    if (status != projection::kProjectionSuccess) {
        producer_status[row] = status;
        return;
    }

    copy_output_row(source, output, generation, cell);
    producer_status[row] = projection::kProjectionSuccess;
}

template <typename T>
bool plane_layout_is_valid(
    const abi::StridedPlaneViewV2<T>& plane,
    const std::int64_t generation_count,
    const std::int64_t cell_count,
    const std::int64_t row_width
) noexcept {
    if (plane.data == nullptr || plane.cell_stride < row_width
        || plane.generation_stride <= 0 || cell_count <= 0
        || generation_count <= 0) {
        return false;
    }
    if (cell_count > std::numeric_limits<std::int64_t>::max()
                         / plane.cell_stride) {
        return false;
    }
    return plane.generation_stride >= cell_count * plane.cell_stride;
}

bool output_layout_is_valid(
    const abi::CompactSummaryViewV2& output
) noexcept {
    return plane_layout_is_valid(
               output.validity,
               output.generation_count,
               output.cell_count,
               abi::kValidityFieldCountV2
           )
        && plane_layout_is_valid(
               output.defined_mask,
               output.generation_count,
               output.cell_count,
               abi::kDefinedMaskFieldCountV2
           )
        && plane_layout_is_valid(
               output.status_codes,
               output.generation_count,
               output.cell_count,
               abi::kStatusFieldCountV2
           )
        && plane_layout_is_valid(
               output.integer_values,
               output.generation_count,
               output.cell_count,
               abi::kIntegerFieldCountV2
           )
        && plane_layout_is_valid(
               output.float_values,
               output.generation_count,
               output.cell_count,
               abi::kFloatFieldCountV2
           )
        && plane_layout_is_valid(
               output.digest_values,
               output.generation_count,
               output.cell_count,
               abi::kDigestFieldCountV2 * abi::kDigestBytesV2
           )
        && plane_layout_is_valid(
               output.text_values,
               output.generation_count,
               output.cell_count,
               abi::kTextFieldCountV2 * abi::kTextRecordBytesV2
           );
}

}  // namespace

namespace eiel::cuda_architecture_b::compact_summary_source_projection_v1 {

cudaError_t launch_compact_summary_source_projection_v1(
    const CompactSummaryGenerationCellAuthorityV1* const authority_rows,
    const std::int64_t authority_row_count,
    const abi::CompactSummaryViewV2 output,
    std::int32_t* const producer_status,
    const cudaStream_t stream
) noexcept {
    if (authority_rows == nullptr || producer_status == nullptr
        || !output_layout_is_valid(output)
        || output.generation_count
               > std::numeric_limits<std::int64_t>::max() / output.cell_count
        || authority_row_count
               != output.generation_count * output.cell_count) {
        return cudaErrorInvalidValue;
    }
    const std::int64_t block_count =
        (authority_row_count + kThreadsPerBlockV1 - 1) / kThreadsPerBlockV1;
    if (block_count <= 0
        || block_count > std::numeric_limits<std::int32_t>::max()) {
        return cudaErrorInvalidConfiguration;
    }
    compact_summary_source_projection_v1_kernel<<<
        static_cast<std::uint32_t>(block_count),
        kThreadsPerBlockV1,
        0,
        stream
    >>>(authority_rows, authority_row_count, output, producer_status);
    return cudaGetLastError();
}

cudaError_t query_compact_summary_source_projection_v1_attributes(
    cudaFuncAttributes* const attributes
) noexcept {
    if (attributes == nullptr) {
        return cudaErrorInvalidValue;
    }
    return cudaFuncGetAttributes(
        attributes,
        compact_summary_source_projection_v1_kernel
    );
}

}  // namespace eiel::cuda_architecture_b::compact_summary_source_projection_v1
