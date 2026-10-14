#include "cpu_compact_summary_v1.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace eiel::native::cpu_compact_summary_v1 {
namespace {

template <std::size_t Size>
bool bytes_are_zero(const std::array<std::uint8_t, Size>& values) noexcept {
    for (const std::uint8_t value : values) {
        if (value != 0) {
            return false;
        }
    }
    return true;
}

std::uint64_t double_bits(const double value) noexcept {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool raw_double_is_zero(const double value) noexcept {
    return double_bits(value) == 0;
}

bool valid_utf8(const std::uint8_t* const data, const std::uint8_t count) noexcept {
    std::uint32_t index = 0;
    while (index < count) {
        const std::uint8_t lead = data[index++];
        if (lead <= 0x7FU) {
            continue;
        }

        std::uint32_t continuation_count = 0;
        std::uint32_t codepoint = 0;
        std::uint32_t minimum = 0;
        if (lead >= 0xC2U && lead <= 0xDFU) {
            continuation_count = 1;
            codepoint = lead & 0x1FU;
            minimum = 0x80U;
        } else if (lead >= 0xE0U && lead <= 0xEFU) {
            continuation_count = 2;
            codepoint = lead & 0x0FU;
            minimum = 0x800U;
        } else if (lead >= 0xF0U && lead <= 0xF4U) {
            continuation_count = 3;
            codepoint = lead & 0x07U;
            minimum = 0x10000U;
        } else {
            return false;
        }
        if (index + continuation_count > count) {
            return false;
        }
        for (std::uint32_t offset = 0; offset < continuation_count; ++offset) {
            const std::uint8_t next = data[index++];
            if ((next & 0xC0U) != 0x80U) {
                return false;
            }
            codepoint = (codepoint << 6U) | (next & 0x3FU);
        }
        if (codepoint < minimum || codepoint > 0x10FFFFU
            || (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) {
            return false;
        }
    }
    return true;
}

EncodeStatusV1 validate_status_codes(
    const CpuCompactSummarySourceAuthorityV1& source
) noexcept {
    for (
        std::int64_t index = 0;
        index < abi::kStatusFieldCountV2;
        ++index
    ) {
        const auto field = static_cast<abi::StatusFieldV2>(index);
        const std::int32_t code = source.status_codes[index];
        bool valid = false;
        switch (field) {
            case abi::kStatusLineageMetricsLineageCollapseWarning:
                valid = abi::test_defined_bit(
                    source.defined_mask,
                    abi::kDefinedLineageMetricsLineageCollapseWarning
                ) ? abi::is_known_status_code(field, code)
                  : code == abi::kReservedInvalidStatusCodeV2;
                break;
            case abi::kStatusTerminalRecoverableErrorCode:
                valid = abi::test_defined_bit(
                    source.defined_mask,
                    abi::kDefinedTerminalRecoverableErrorCode
                ) ? abi::is_known_status_code(field, code)
                  : code == abi::kReservedInvalidStatusCodeV2;
                break;
            case abi::kStatusTerminalFailureStage:
                valid = abi::test_defined_bit(
                    source.defined_mask,
                    abi::kDefinedTerminalFailureStage
                ) ? abi::is_known_status_code(field, code)
                  : code == abi::kReservedInvalidStatusCodeV2;
                break;
            default:
                valid = abi::is_known_status_code(field, code);
                break;
        }
        if (!valid) {
            return EncodeStatusV1::kStatusCodeInvalid;
        }
    }
    return EncodeStatusV1::kSuccess;
}

EncodeStatusV1 validate_integer_values(
    const CpuCompactSummarySourceAuthorityV1& source
) noexcept {
    for (const std::int64_t value : source.integer_values) {
        if (value < 0) {
            return EncodeStatusV1::kIntegerRangeInvalid;
        }
    }
    constexpr std::array<abi::IntegerFieldV2, 5> kPositiveFields = {
        abi::kIntegerGenerationEndTickExclusive,
        abi::kIntegerNextGenerationStartGeneration,
        abi::kIntegerTurnoverCount,
        abi::kIntegerIdentityPopulationNextId,
        abi::kIntegerGenerationMetricsCapacity,
    };
    for (const abi::IntegerFieldV2 field : kPositiveFields) {
        if (source.integer_values[field] <= 0) {
            return EncodeStatusV1::kIntegerRangeInvalid;
        }
    }
    if (source.integer_values[abi::kIntegerGenerationStartGeneration]
        > 0x7FFFFFFFFFFFFFFELL) {
        return EncodeStatusV1::kIntegerRangeInvalid;
    }
    return EncodeStatusV1::kSuccess;
}

EncodeStatusV1 validate_float_values(
    const CpuCompactSummarySourceAuthorityV1& source
) noexcept {
    for (const double value : source.float_values) {
        if (!std::isfinite(value)) {
            return EncodeStatusV1::kFloatNonFinite;
        }
    }
    constexpr std::array<abi::FloatFieldV2, 4> kNonnegativeFields = {
        abi::kFloatGenerationMetricsPeakInjury,
        abi::kFloatPatchMetricsPatchFoodHarvestTotal,
        abi::kFloatPatchMetricsPatchScentEmitTotal,
        abi::kFloatPatchMetricsPatchWasteEmitTotal,
    };
    for (const abi::FloatFieldV2 field : kNonnegativeFields) {
        if (source.float_values[field] < 0.0) {
            return EncodeStatusV1::kFloatRangeInvalid;
        }
    }

    struct RatioField {
        abi::FloatFieldV2 field;
        abi::DefinedBitV1 bit;
    };
    constexpr std::array<RatioField, 6> kRatioFields = {{
        {abi::kFloatLineageMetricsDominantLineagePopulationShare,
         abi::kDefinedLineageMetricsDominantLineagePopulationShare},
        {abi::kFloatLineageMetricsTop3LineagePopulationShare,
         abi::kDefinedLineageMetricsTop3LineagePopulationShare},
        {abi::kFloatLineageMetricsDominantLineageReproductionShare,
         abi::kDefinedLineageMetricsDominantLineageReproductionShare},
        {abi::kFloatPatchMetricsPatchFoodDepletedFrac,
         abi::kDefinedPatchMetricsPatchFoodDepletedFrac},
        {abi::kFloatPatchMetricsPatchFoodFullFrac,
         abi::kDefinedPatchMetricsPatchFoodFullFrac},
        {abi::kFloatPatchMetricsPatchDensityHotspotFrac,
         abi::kDefinedPatchMetricsPatchDensityHotspotFrac},
    }};
    for (const RatioField& entry : kRatioFields) {
        if (abi::test_defined_bit(source.defined_mask, entry.bit)) {
            const double value = source.float_values[entry.field];
            if (value < 0.0 || value > 1.0) {
                return EncodeStatusV1::kFloatRangeInvalid;
            }
        }
    }
    return EncodeStatusV1::kSuccess;
}

EncodeStatusV1 validate_nullable_storage(
    const CpuCompactSummarySourceAuthorityV1& source
) noexcept {
    struct NullableInteger {
        abi::DefinedBitV1 bit;
        abi::IntegerFieldV2 field;
    };
    constexpr std::array<NullableInteger, 10> kNullableIntegers = {{
        {abi::kDefinedIdentityActiveAgentIdMin,
         abi::kIntegerIdentityActiveAgentIdMin},
        {abi::kDefinedIdentityActiveAgentIdMax,
         abi::kIntegerIdentityActiveAgentIdMax},
        {abi::kDefinedTerminalFailureGeneration,
         abi::kIntegerTerminalFailureGeneration},
        {abi::kDefinedTerminalFailureTick, abi::kIntegerTerminalFailureTick},
        {abi::kDefinedLineageMetricsActiveLineages,
         abi::kIntegerLineageMetricsActiveLineages},
        {abi::kDefinedLineageMetricsRarePersistentLineages,
         abi::kIntegerLineageMetricsRarePersistentLineages},
        {abi::kDefinedLineageMetricsLineageCollapseStreak,
         abi::kIntegerLineageMetricsLineageCollapseStreak},
        {abi::kDefinedPatchMetricsPatchSpawnSyncRows,
         abi::kIntegerPatchMetricsPatchSpawnSyncRows},
        {abi::kDefinedPatchMetricsTypedDeathEventRows,
         abi::kIntegerPatchMetricsTypedDeathEventRows},
        {abi::kDefinedPatchMetricsEnvRngPaddingDraws,
         abi::kIntegerPatchMetricsEnvRngPaddingDraws},
    }};
    for (const NullableInteger& entry : kNullableIntegers) {
        if (!abi::test_defined_bit(source.defined_mask, entry.bit)
            && source.integer_values[entry.field] != 0) {
            return EncodeStatusV1::kUndefinedStorageNonzero;
        }
    }

    struct NullableFloat {
        abi::DefinedBitV1 bit;
        abi::FloatFieldV2 field;
    };
    constexpr std::array<NullableFloat, 16> kNullableFloats = {{
        {abi::kDefinedLineageMetricsDominantLineagePopulationShare,
         abi::kFloatLineageMetricsDominantLineagePopulationShare},
        {abi::kDefinedLineageMetricsTop3LineagePopulationShare,
         abi::kFloatLineageMetricsTop3LineagePopulationShare},
        {abi::kDefinedLineageMetricsLineagePopulationEntropy,
         abi::kFloatLineageMetricsLineagePopulationEntropy},
        {abi::kDefinedLineageMetricsLineageReproductionEntropy,
         abi::kFloatLineageMetricsLineageReproductionEntropy},
        {abi::kDefinedLineageMetricsDominantLineageReproductionShare,
         abi::kFloatLineageMetricsDominantLineageReproductionShare},
        {abi::kDefinedPatchMetricsEnergyCollectedDelta,
         abi::kFloatPatchMetricsEnergyCollectedDelta},
        {abi::kDefinedPatchMetricsPatchScentMoveQueuedTotal,
         abi::kFloatPatchMetricsPatchScentMoveQueuedTotal},
        {abi::kDefinedPatchMetricsPatchWasteMoveQueuedTotal,
         abi::kFloatPatchMetricsPatchWasteMoveQueuedTotal},
        {abi::kDefinedPatchMetricsPatchScentGatherQueuedTotal,
         abi::kFloatPatchMetricsPatchScentGatherQueuedTotal},
        {abi::kDefinedPatchMetricsPatchWasteGatherQueuedTotal,
         abi::kFloatPatchMetricsPatchWasteGatherQueuedTotal},
        {abi::kDefinedPatchMetricsPatchMovementEnergyExtraTotal,
         abi::kFloatPatchMetricsPatchMovementEnergyExtraTotal},
        {abi::kDefinedPatchMetricsPatchScentEmittedTotal,
         abi::kFloatPatchMetricsPatchScentEmittedTotal},
        {abi::kDefinedPatchMetricsPatchWasteEmittedTotal,
         abi::kFloatPatchMetricsPatchWasteEmittedTotal},
        {abi::kDefinedPatchMetricsPatchFoodDepletedFrac,
         abi::kFloatPatchMetricsPatchFoodDepletedFrac},
        {abi::kDefinedPatchMetricsPatchFoodFullFrac,
         abi::kFloatPatchMetricsPatchFoodFullFrac},
        {abi::kDefinedPatchMetricsPatchDensityHotspotFrac,
         abi::kFloatPatchMetricsPatchDensityHotspotFrac},
    }};
    for (const NullableFloat& entry : kNullableFloats) {
        if (!abi::test_defined_bit(source.defined_mask, entry.bit)
            && !raw_double_is_zero(source.float_values[entry.field])) {
            return EncodeStatusV1::kUndefinedStorageNonzero;
        }
    }

    if (!abi::test_defined_bit(
            source.defined_mask,
            abi::kDefinedTerminalRecoverableErrorDigestSha256
        )
        && !bytes_are_zero(
            source.digest_values[
                abi::kDigestTerminalRecoverableErrorDigestSha256
            ].bytes
        )) {
        return EncodeStatusV1::kUndefinedStorageNonzero;
    }
    return EncodeStatusV1::kSuccess;
}

EncodeStatusV1 validate_flat_storage(
    const CpuCompactSummarySourceAuthorityV1& source,
    const bool patch_present
) noexcept {
    if (patch_present) {
        return EncodeStatusV1::kSuccess;
    }
    for (
        std::int64_t index =
            abi::kIntegerPatchMetricsPatchMovementEnergyEventCount;
        index < abi::kIntegerFieldCountV2;
        ++index
    ) {
        if (source.integer_values[index] != 0) {
            return EncodeStatusV1::kFlatPatchStorageNonzero;
        }
    }
    for (
        std::int64_t index = abi::kFloatPatchMetricsPatchFoodHarvestTotal;
        index < abi::kFloatFieldCountV2;
        ++index
    ) {
        if (!raw_double_is_zero(source.float_values[index])) {
            return EncodeStatusV1::kFlatPatchStorageNonzero;
        }
    }
    return EncodeStatusV1::kSuccess;
}

EncodeStatusV1 validate_terminal_state(
    const CpuCompactSummarySourceAuthorityV1& source
) noexcept {
    const bool recoverable_code = abi::test_defined_bit(
        source.defined_mask, abi::kDefinedTerminalRecoverableErrorCode
    );
    const bool recoverable_digest = abi::test_defined_bit(
        source.defined_mask,
        abi::kDefinedTerminalRecoverableErrorDigestSha256
    );
    const bool recoverable_present = source.status_codes[
        abi::kStatusTerminalRecoverableErrorStatus
    ] == abi::kRecoverableErrorStatusPresent;
    if (recoverable_code != recoverable_digest
        || recoverable_present != recoverable_code) {
        return EncodeStatusV1::kRecoverableStateInconsistent;
    }

    const bool failure_generation = abi::test_defined_bit(
        source.defined_mask, abi::kDefinedTerminalFailureGeneration
    );
    const bool failure_tick = abi::test_defined_bit(
        source.defined_mask, abi::kDefinedTerminalFailureTick
    );
    const bool failure_stage = abi::test_defined_bit(
        source.defined_mask, abi::kDefinedTerminalFailureStage
    );
    const bool failed = source.status_codes[abi::kStatusTerminalStatus]
        == abi::kTerminalStatusFailed;
    const bool sticky = source.status_codes[abi::kStatusTerminalStickyFailure]
        == abi::kBooleanStatusTrue;
    if (failed) {
        if (!sticky || !failure_generation || !failure_tick || !failure_stage) {
            return EncodeStatusV1::kTerminalStateInconsistent;
        }
    } else if (sticky || failure_generation || failure_tick || failure_stage) {
        return EncodeStatusV1::kTerminalStateInconsistent;
    }
    return EncodeStatusV1::kSuccess;
}

EncodeStatusV1 validate_digest_values(
    const CpuCompactSummarySourceAuthorityV1& source
) noexcept {
    const bool recoverable_error_defined = abi::test_defined_bit(
        source.defined_mask,
        abi::kDefinedTerminalRecoverableErrorDigestSha256
    );
    for (
        std::int64_t index = 0;
        index < abi::kDigestFieldCountV2;
        ++index
    ) {
        if (
            index == abi::kDigestTerminalRecoverableErrorDigestSha256
            && !recoverable_error_defined
        ) {
            continue;
        }
        if (bytes_are_zero(source.digest_values[index].bytes)) {
            return EncodeStatusV1::kRequiredDigestMissing;
        }
    }
    return EncodeStatusV1::kSuccess;
}

EncodeStatusV1 validate_text_values(
    const CpuCompactSummarySourceAuthorityV1& source
) noexcept {
    for (
        std::int64_t index = 0;
        index < abi::kTextFieldCountV2;
        ++index
    ) {
        const Utf8TextValueV1& text = source.text_values[index];
        const bool defined = abi::test_defined_bit(
            source.defined_mask,
            abi::kDefinedLineageMetricsDominantLineageId
        );
        if (!defined) {
            if (text.length != 0 || !bytes_are_zero(text.payload)) {
                return EncodeStatusV1::kUndefinedStorageNonzero;
            }
            continue;
        }
        if (text.length == 0
            || text.length > abi::kTextMaximumPayloadBytesV2) {
            return EncodeStatusV1::kTextLengthInvalid;
        }
        for (
            std::size_t offset = text.length;
            offset < text.payload.size();
            ++offset
        ) {
            if (text.payload[offset] != 0) {
                return EncodeStatusV1::kTextTailNonzero;
            }
        }
        if (!valid_utf8(text.payload.data(), text.length)) {
            return EncodeStatusV1::kTextUtf8Invalid;
        }
    }
    return EncodeStatusV1::kSuccess;
}

}  // namespace

void initialize_invalid_row(CpuCompactSummaryRowV1& row) noexcept {
    row.validity.fill(0);
    row.defined_mask.fill(0);
    row.status_codes.fill(0);
    row.integer_values.fill(0);
    row.float_values.fill(0.0);
    row.digest_values.fill(0);
    row.text_values.fill(0);
}

EncodeStatusV1 validate_completed_source(
    const CpuCompactSummarySourceAuthorityV1& source
) noexcept {
    if (source.authority_version != kSourceAuthorityVersionV1) {
        return EncodeStatusV1::kSourceAuthorityVersionMismatch;
    }
    if (!abi::defined_mask_has_only_frozen_bits(source.defined_mask)) {
        return EncodeStatusV1::kReservedDefinedBit;
    }
    constexpr std::uint32_t kPatchObjectBit =
        static_cast<std::uint32_t>(abi::kDefinedPatchMetrics);
    constexpr abi::DefinedMaskV1 kPatchChildMask =
        abi::kUsedDefinedMaskV1
        & ~((abi::DefinedMaskV1{1} << (kPatchObjectBit + 1U)) - 1U);
    const bool patch_present = abi::test_defined_bit(
        source.defined_mask, abi::kDefinedPatchMetrics
    );
    if (!patch_present && (source.defined_mask & kPatchChildMask) != 0) {
        return EncodeStatusV1::kPatchChildWithoutObject;
    }

    EncodeStatusV1 status = validate_status_codes(source);
    if (status != EncodeStatusV1::kSuccess) {
        return status;
    }
    const bool patch_world = source.status_codes[abi::kStatusWorld]
        == abi::kWorldPatch;
    if (patch_world != patch_present) {
        return EncodeStatusV1::kWorldPatchPresenceMismatch;
    }
    status = validate_integer_values(source);
    if (status != EncodeStatusV1::kSuccess) {
        return status;
    }
    status = validate_float_values(source);
    if (status != EncodeStatusV1::kSuccess) {
        return status;
    }
    status = validate_nullable_storage(source);
    if (status != EncodeStatusV1::kSuccess) {
        return status;
    }
    status = validate_flat_storage(source, patch_present);
    if (status != EncodeStatusV1::kSuccess) {
        return status;
    }
    status = validate_terminal_state(source);
    if (status != EncodeStatusV1::kSuccess) {
        return status;
    }
    status = validate_digest_values(source);
    if (status != EncodeStatusV1::kSuccess) {
        return status;
    }
    return validate_text_values(source);
}

EncodeStatusV1 encode_completed_row(
    const CpuCompactSummarySourceAuthorityV1& source,
    CpuCompactSummaryRowV1& row
) noexcept {
    initialize_invalid_row(row);
    const EncodeStatusV1 status = validate_completed_source(source);
    if (status != EncodeStatusV1::kSuccess) {
        return status;
    }

    row.defined_mask[0] = source.defined_mask;
    row.status_codes = source.status_codes;
    row.integer_values = source.integer_values;
    row.float_values = source.float_values;
    for (
        std::int64_t field = 0;
        field < abi::kDigestFieldCountV2;
        ++field
    ) {
        const std::int64_t output_offset = field * abi::kDigestBytesV2;
        for (
            std::int64_t byte = 0;
            byte < abi::kDigestBytesV2;
            ++byte
        ) {
            row.digest_values[output_offset + byte] =
                source.digest_values[field].bytes[byte];
        }
    }
    for (
        std::int64_t field = 0;
        field < abi::kTextFieldCountV2;
        ++field
    ) {
        const std::int64_t output_offset = field * abi::kTextRecordBytesV2;
        const Utf8TextValueV1& text = source.text_values[field];
        row.text_values[output_offset] = text.length;
        for (
            std::int64_t byte = 0;
            byte < abi::kTextMaximumPayloadBytesV2;
            ++byte
        ) {
            row.text_values[output_offset + 1 + byte] = text.payload[byte];
        }
    }

    row.validity[0] = 1;
    return EncodeStatusV1::kSuccess;
}

abi::CompactSummaryViewV2 compact_summary_view(
    CpuCompactSummaryRowV1& row
) noexcept {
    return {
        1,
        1,
        {row.validity.data(), 1, 1},
        {row.defined_mask.data(), 1, 1},
        {row.status_codes.data(), abi::kStatusFieldCountV2,
         abi::kStatusFieldCountV2},
        {row.integer_values.data(), abi::kIntegerFieldCountV2,
         abi::kIntegerFieldCountV2},
        {row.float_values.data(), abi::kFloatFieldCountV2,
         abi::kFloatFieldCountV2},
        {row.digest_values.data(),
         abi::kDigestFieldCountV2 * abi::kDigestBytesV2,
         abi::kDigestFieldCountV2 * abi::kDigestBytesV2},
        {row.text_values.data(),
         abi::kTextFieldCountV2 * abi::kTextRecordBytesV2,
         abi::kTextFieldCountV2 * abi::kTextRecordBytesV2},
    };
}

const char* encode_status_name(const EncodeStatusV1 status) noexcept {
    switch (status) {
        case EncodeStatusV1::kSuccess: return "success";
        case EncodeStatusV1::kSourceAuthorityVersionMismatch:
            return "source_authority_version_mismatch";
        case EncodeStatusV1::kReservedDefinedBit:
            return "reserved_defined_bit";
        case EncodeStatusV1::kPatchChildWithoutObject:
            return "patch_child_without_object";
        case EncodeStatusV1::kStatusCodeInvalid: return "status_code_invalid";
        case EncodeStatusV1::kWorldPatchPresenceMismatch:
            return "world_patch_presence_mismatch";
        case EncodeStatusV1::kIntegerRangeInvalid:
            return "integer_range_invalid";
        case EncodeStatusV1::kFloatNonFinite: return "float_non_finite";
        case EncodeStatusV1::kFloatRangeInvalid:
            return "float_range_invalid";
        case EncodeStatusV1::kUndefinedStorageNonzero:
            return "undefined_storage_nonzero";
        case EncodeStatusV1::kFlatPatchStorageNonzero:
            return "flat_patch_storage_nonzero";
        case EncodeStatusV1::kRecoverableStateInconsistent:
            return "recoverable_state_inconsistent";
        case EncodeStatusV1::kTerminalStateInconsistent:
            return "terminal_state_inconsistent";
        case EncodeStatusV1::kTextLengthInvalid:
            return "text_length_invalid";
        case EncodeStatusV1::kTextUtf8Invalid: return "text_utf8_invalid";
        case EncodeStatusV1::kTextTailNonzero:
            return "text_tail_nonzero";
        case EncodeStatusV1::kRequiredDigestMissing:
            return "required_digest_missing";
    }
    return "unknown";
}

}  // namespace eiel::native::cpu_compact_summary_v1
