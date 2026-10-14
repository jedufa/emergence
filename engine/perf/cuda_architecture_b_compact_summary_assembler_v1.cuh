#pragma once

#include "cuda_architecture_b_compact_summary_v2.cuh"

#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__CUDACC__)
#define EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE __host__ __device__
#else
#define EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE
#endif

namespace eiel::cuda_architecture_b::compact_summary_assembler_v1 {

namespace abi = eiel::cuda_architecture_b::compact_summary_v2;

inline constexpr std::uint32_t kBoundarySourceVersionV1 = 1;

enum AssembleStatusV1 : std::int32_t {
    kAssembleSuccess = 0,
    kAssembleNullPointer = 1,
    kAssembleSourceVersionMismatch = 2,
    kAssembleReservedDefinedBit = 3,
    kAssemblePatchChildWithoutObject = 4,
    kAssembleStatusCodeInvalid = 5,
    kAssembleWorldPatchPresenceMismatch = 6,
    kAssembleIntegerRangeInvalid = 7,
    kAssembleFloatNonFinite = 8,
    kAssembleFloatRangeInvalid = 9,
    kAssembleUndefinedStorageNonzero = 10,
    kAssembleFlatPatchStorageNonzero = 11,
    kAssembleRecoverableStateInconsistent = 12,
    kAssembleTerminalStateInconsistent = 13,
    kAssembleTextInvalid = 14,
};

// The source is a non-owning, device-neutral semantic view. Status members use
// the frozen ABI-v2 codebooks; numeric and digest pointers use the frozen enum
// indices from cuda_architecture_b_compact_summary_v2.cuh.
struct CompactSummaryBoundarySourceV1 {
    std::uint32_t source_version;
    abi::DefinedMaskV1 defined_mask;
    std::int32_t summary_status_code;
    std::int32_t world_code;
    std::int32_t turnover_status_code;
    std::int32_t rng_authority_complete_code;
    std::int32_t rng_export_non_consuming_code;
    std::int32_t terminal_status_code;
    std::int32_t terminal_recoverable_error_status_code;
    std::int32_t terminal_sticky_failure_code;
    std::int32_t checkpoint_due_code;
    std::int32_t lineage_collapse_warning_code;
    std::int32_t terminal_recoverable_error_code;
    std::int32_t terminal_failure_stage_code;
    const std::int64_t* integer_values;
    const double* float_values;
    const std::uint8_t* digest_values;
    const std::uint8_t* dominant_lineage_id_record;
};

// This points at one row of each ABI-v2 scratch plane. `ready` is separate from
// ABI validity so the eventual output finalizer can publish validity last.
struct CompactSummaryScratchRowV1 {
    std::uint8_t* ready;
    std::uint8_t* validity;
    abi::DefinedMaskV1* defined_mask;
    std::int32_t* status_codes;
    std::int64_t* integer_values;
    double* float_values;
    std::uint8_t* digest_values;
    std::uint8_t* text_values;
};

EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE inline std::uint64_t double_bits(
    const double value
) noexcept {
#if defined(__CUDA_ARCH__)
    return static_cast<std::uint64_t>(__double_as_longlong(value));
#else
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
#endif
}

EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE inline bool is_finite_double(
    const double value
) noexcept {
    return (double_bits(value) & 0x7FF0000000000000ULL)
        != 0x7FF0000000000000ULL;
}

EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE inline bool raw_double_is_zero(
    const double value
) noexcept {
    return double_bits(value) == 0;
}

EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE inline bool bytes_are_zero(
    const std::uint8_t* data,
    const std::int64_t count
) noexcept {
    if (data == nullptr) {
        return false;
    }
    for (std::int64_t index = 0; index < count; ++index) {
        if (data[index] != 0) {
            return false;
        }
    }
    return true;
}

EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE inline bool is_valid_utf8(
    const std::uint8_t* data,
    const std::uint8_t count
) noexcept {
    std::uint32_t index = 0;
    while (index < count) {
        const std::uint8_t lead = data[index++];
        if (lead <= 0x7F) {
            continue;
        }
        std::uint32_t continuation_count = 0;
        std::uint32_t codepoint = 0;
        std::uint32_t minimum = 0;
        if (lead >= 0xC2 && lead <= 0xDF) {
            continuation_count = 1;
            codepoint = lead & 0x1F;
            minimum = 0x80;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            continuation_count = 2;
            codepoint = lead & 0x0F;
            minimum = 0x800;
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            continuation_count = 3;
            codepoint = lead & 0x07;
            minimum = 0x10000;
        } else {
            return false;
        }
        if (index + continuation_count > count) {
            return false;
        }
        for (std::uint32_t offset = 0; offset < continuation_count; ++offset) {
            const std::uint8_t next = data[index++];
            if ((next & 0xC0) != 0x80) {
                return false;
            }
            codepoint = (codepoint << 6) | (next & 0x3F);
        }
        if (codepoint < minimum || codepoint > 0x10FFFF
            || (codepoint >= 0xD800 && codepoint <= 0xDFFF)) {
            return false;
        }
    }
    return true;
}

EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE inline bool scratch_pointers_valid(
    const CompactSummaryScratchRowV1& scratch
) noexcept {
    return scratch.ready != nullptr
        && scratch.validity != nullptr
        && scratch.defined_mask != nullptr
        && scratch.status_codes != nullptr
        && scratch.integer_values != nullptr
        && scratch.float_values != nullptr
        && scratch.digest_values != nullptr
        && scratch.text_values != nullptr;
}

EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE inline void clear_scratch_row(
    const CompactSummaryScratchRowV1& scratch
) noexcept {
    *scratch.ready = 0;
    *scratch.validity = 0;
    *scratch.defined_mask = 0;
    for (std::int64_t index = 0; index < abi::kStatusFieldCountV2; ++index) {
        scratch.status_codes[index] = 0;
    }
    for (std::int64_t index = 0; index < abi::kIntegerFieldCountV2; ++index) {
        scratch.integer_values[index] = 0;
    }
    for (std::int64_t index = 0; index < abi::kFloatFieldCountV2; ++index) {
        scratch.float_values[index] = 0.0;
    }
    for (
        std::int64_t index = 0;
        index < abi::kDigestFieldCountV2 * abi::kDigestBytesV2;
        ++index
    ) {
        scratch.digest_values[index] = 0;
    }
    for (
        std::int64_t index = 0;
        index < abi::kTextFieldCountV2 * abi::kTextRecordBytesV2;
        ++index
    ) {
        scratch.text_values[index] = 0;
    }
}

EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE inline bool source_pointers_valid(
    const CompactSummaryBoundarySourceV1& source
) noexcept {
    return source.integer_values != nullptr
        && source.float_values != nullptr
        && source.digest_values != nullptr
        && source.dominant_lineage_id_record != nullptr;
}

EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE inline bool nullable_code_valid(
    const abi::DefinedMaskV1 mask,
    const abi::DefinedBitV1 bit,
    const abi::StatusFieldV2 field,
    const std::int32_t code
) noexcept {
    return abi::test_defined_bit(mask, bit)
        ? abi::is_known_status_code(field, code)
        : code == abi::kReservedInvalidStatusCodeV2;
}

EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE inline AssembleStatusV1
validate_statuses(const CompactSummaryBoundarySourceV1& source) noexcept {
    if (source.summary_status_code != abi::kSummaryStatusComplete
        || source.turnover_status_code != abi::kTurnoverStatusCommitted
        || !abi::is_known_status_code(
            abi::kStatusWorld, source.world_code
        )
        || !abi::is_known_status_code(
            abi::kStatusRngAuthorityComplete,
            source.rng_authority_complete_code
        )
        || !abi::is_known_status_code(
            abi::kStatusRngExportNonConsuming,
            source.rng_export_non_consuming_code
        )
        || !abi::is_known_status_code(
            abi::kStatusTerminalStatus, source.terminal_status_code
        )
        || !abi::is_known_status_code(
            abi::kStatusTerminalRecoverableErrorStatus,
            source.terminal_recoverable_error_status_code
        )
        || !abi::is_known_status_code(
            abi::kStatusTerminalStickyFailure,
            source.terminal_sticky_failure_code
        )
        || !abi::is_known_status_code(
            abi::kStatusCheckpointDue, source.checkpoint_due_code
        )) {
        return kAssembleStatusCodeInvalid;
    }
    if (!nullable_code_valid(
            source.defined_mask,
            abi::kDefinedLineageMetricsLineageCollapseWarning,
            abi::kStatusLineageMetricsLineageCollapseWarning,
            source.lineage_collapse_warning_code
        )
        || !nullable_code_valid(
            source.defined_mask,
            abi::kDefinedTerminalRecoverableErrorCode,
            abi::kStatusTerminalRecoverableErrorCode,
            source.terminal_recoverable_error_code
        )
        || !nullable_code_valid(
            source.defined_mask,
            abi::kDefinedTerminalFailureStage,
            abi::kStatusTerminalFailureStage,
            source.terminal_failure_stage_code
        )) {
        return kAssembleStatusCodeInvalid;
    }
    return kAssembleSuccess;
}

EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE inline AssembleStatusV1
validate_integers(const CompactSummaryBoundarySourceV1& source) noexcept {
    for (std::int64_t index = 0; index < abi::kIntegerFieldCountV2; ++index) {
        if (source.integer_values[index] < 0) {
            return kAssembleIntegerRangeInvalid;
        }
    }
    constexpr abi::IntegerFieldV2 positive_fields[] = {
        abi::kIntegerGenerationEndTickExclusive,
        abi::kIntegerNextGenerationStartGeneration,
        abi::kIntegerTurnoverCount,
        abi::kIntegerIdentityPopulationNextId,
        abi::kIntegerGenerationMetricsCapacity,
    };
    for (const abi::IntegerFieldV2 field : positive_fields) {
        if (source.integer_values[field] <= 0) {
            return kAssembleIntegerRangeInvalid;
        }
    }
    if (source.integer_values[abi::kIntegerGenerationStartGeneration]
        > 0x7FFFFFFFFFFFFFFELL) {
        return kAssembleIntegerRangeInvalid;
    }
    return kAssembleSuccess;
}

EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE inline bool add_nonnegative_exact(
    const std::int64_t left,
    const std::int64_t right,
    std::int64_t* const result
) noexcept {
    if (left < 0 || right < 0 || result == nullptr
        || left > 0x7FFFFFFFFFFFFFFFLL - right) {
        return false;
    }
    *result = left + right;
    return true;
}

EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE inline bool sum_four_nonnegative_exact(
    const std::int64_t first,
    const std::int64_t second,
    const std::int64_t third,
    const std::int64_t fourth,
    std::int64_t* const result
) noexcept {
    std::int64_t left = 0;
    std::int64_t right = 0;
    return add_nonnegative_exact(first, second, &left)
        && add_nonnegative_exact(third, fourth, &right)
        && add_nonnegative_exact(left, right, result);
}

EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE inline AssembleStatusV1
validate_cross_field_invariants(
    const CompactSummaryBoundarySourceV1& source
) noexcept {
    const std::int64_t* const value = source.integer_values;
    const std::int64_t generation =
        value[abi::kIntegerGenerationStartGeneration];
    if (generation != value[abi::kIntegerGenerationEndGeneration]
        || value[abi::kIntegerGenerationStartTick] != 0
        || value[abi::kIntegerNextGenerationStartTick] != 0
        || value[abi::kIntegerGenerationEndGeneration]
            == 0x7FFFFFFFFFFFFFFFLL
        || value[abi::kIntegerNextGenerationStartGeneration]
            != value[abi::kIntegerGenerationEndGeneration] + 1
        || value[abi::kIntegerGenerationEndTickExclusive]
            != value[abi::kIntegerGenerationMetricsPolicyLoopTickCalls]
        || value[abi::kIntegerTurnoverCount] != 1) {
        return kAssembleIntegerRangeInvalid;
    }

    const std::int64_t capacity =
        value[abi::kIntegerGenerationMetricsCapacity];
    const std::int64_t free_slots =
        value[abi::kIntegerGenerationMetricsFreeSlots];
    if (free_slots > capacity
        || value[abi::kIntegerAlive] != capacity - free_slots) {
        return kAssembleIntegerRangeInvalid;
    }
    std::int64_t sum = 0;
    if (!add_nonnegative_exact(
            value[abi::kIntegerDeathsByClassEnergyDepleted],
            value[abi::kIntegerDeathsByClassHealthDepleted],
            &sum
        )
        || value[abi::kIntegerDeaths] != sum) {
        return kAssembleIntegerRangeInvalid;
    }

    const std::int64_t cursor_start =
        value[abi::kIntegerEventsCursorStart];
    const std::int64_t cursor_pre =
        value[abi::kIntegerEventsCursorPreTurnover];
    const std::int64_t cursor_end = value[abi::kIntegerEventsCursorEnd];
    if (cursor_pre < cursor_start || cursor_end < cursor_pre
        || value[abi::kIntegerEventsAuthoritativeCursor] != cursor_end
        || cursor_pre - cursor_start
            != value[abi::kIntegerEventsGenerationEventCount]
        || cursor_end - cursor_pre
            != value[abi::kIntegerEventsTurnoverEventCount]
        || cursor_end - cursor_start
            != value[abi::kIntegerEventsEventRowCount]) {
        return kAssembleIntegerRangeInvalid;
    }
    if (!sum_four_nonnegative_exact(
            value[abi::kIntegerEventsGenerationCountsByTypeBirthCommitted],
            value[abi::kIntegerEventsGenerationCountsByTypeMutation],
            value[abi::kIntegerEventsGenerationCountsByTypeBirthFailed],
            value[abi::kIntegerEventsGenerationCountsByTypeDeathCommitted],
            &sum
        )
        || sum != value[abi::kIntegerEventsGenerationEventCount]
        || value[abi::kIntegerBirths]
            != value[
                abi::kIntegerEventsGenerationCountsByTypeBirthCommitted
            ]
        || value[abi::kIntegerFailedBirths]
            != value[abi::kIntegerEventsGenerationCountsByTypeBirthFailed]
        || value[abi::kIntegerDeaths]
            != value[abi::kIntegerEventsGenerationCountsByTypeDeathCommitted]) {
        return kAssembleIntegerRangeInvalid;
    }
    if (!sum_four_nonnegative_exact(
            value[abi::kIntegerEventsTurnoverCountsByTypeBirthCommitted],
            value[abi::kIntegerEventsTurnoverCountsByTypeMutation],
            value[abi::kIntegerEventsTurnoverCountsByTypeBirthFailed],
            value[abi::kIntegerEventsTurnoverCountsByTypeDeathCommitted],
            &sum
        )
        || sum != value[abi::kIntegerEventsTurnoverEventCount]) {
        return kAssembleIntegerRangeInvalid;
    }

    if (!add_nonnegative_exact(
            value[abi::kIntegerIdentityActiveAgentCount],
            value[abi::kIntegerIdentityFreeSlotCount],
            &sum
        )
        || sum != capacity) {
        return kAssembleIntegerRangeInvalid;
    }
    const bool minimum_defined = abi::test_defined_bit(
        source.defined_mask, abi::kDefinedIdentityActiveAgentIdMin
    );
    const bool maximum_defined = abi::test_defined_bit(
        source.defined_mask, abi::kDefinedIdentityActiveAgentIdMax
    );
    const std::int64_t active =
        value[abi::kIntegerIdentityActiveAgentCount];
    if (active == 0) {
        if (minimum_defined || maximum_defined) {
            return kAssembleIntegerRangeInvalid;
        }
    } else if (!minimum_defined || !maximum_defined
        || value[abi::kIntegerIdentityActiveAgentIdMin]
            > value[abi::kIntegerIdentityActiveAgentIdMax]
        || value[abi::kIntegerIdentityActiveAgentIdMax]
            >= value[abi::kIntegerIdentityPopulationNextId]) {
        return kAssembleIntegerRangeInvalid;
    }

    if (!sum_four_nonnegative_exact(
            value[abi::kIntegerBehaviorMetricsMoveN],
            value[abi::kIntegerBehaviorMetricsMoveS],
            value[abi::kIntegerBehaviorMetricsMoveE],
            value[abi::kIntegerBehaviorMetricsMoveW],
            &sum
        )
        || sum != value[abi::kIntegerBehaviorMetricsMoveTotal]
        || !sum_four_nonnegative_exact(
            value[abi::kIntegerBehaviorMetricsMoveTotal],
            value[abi::kIntegerBehaviorMetricsGather],
            value[abi::kIntegerBehaviorMetricsRest],
            value[abi::kIntegerBehaviorMetricsAttack],
            &sum
        )
        || sum != value[abi::kIntegerBehaviorMetricsActionTotal]
        || value[abi::kIntegerBehaviorMetricsCohortSurvivors]
            > value[abi::kIntegerBehaviorMetricsCohortSize]
        || value[abi::kIntegerBehaviorMetricsCohortSize]
            > value[abi::kIntegerBehaviorMetricsRowCount]) {
        return kAssembleIntegerRangeInvalid;
    }

    if (!add_nonnegative_exact(
            value[abi::kIntegerGenerationMetricsPolicyCacheRowsHit],
            value[abi::kIntegerGenerationMetricsPolicyCacheRowsMissed],
            &sum
        )
        || sum
            != value[
                abi::kIntegerGenerationMetricsPolicyCacheRowsRequested
            ]
        || value[abi::kIntegerGenerationMetricsPolicyCacheRowsMissed]
            != value[abi::kIntegerGenerationMetricsPolicyCacheRowsRefilled]) {
        return kAssembleIntegerRangeInvalid;
    }
    if (source.rng_authority_complete_code != abi::kBooleanStatusTrue
        || source.rng_export_non_consuming_code != abi::kBooleanStatusTrue) {
        return kAssembleStatusCodeInvalid;
    }
    return kAssembleSuccess;
}

EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE inline AssembleStatusV1
validate_floats(const CompactSummaryBoundarySourceV1& source) noexcept {
    for (std::int64_t index = 0; index < abi::kFloatFieldCountV2; ++index) {
        if (!is_finite_double(source.float_values[index])) {
            return kAssembleFloatNonFinite;
        }
    }
    constexpr abi::FloatFieldV2 nonnegative_fields[] = {
        abi::kFloatGenerationMetricsPeakInjury,
        abi::kFloatPatchMetricsPatchFoodHarvestTotal,
        abi::kFloatPatchMetricsPatchScentEmitTotal,
        abi::kFloatPatchMetricsPatchWasteEmitTotal,
        abi::kFloatPatchMetricsPatchMovementEnergyTotal,
    };
    for (const abi::FloatFieldV2 field : nonnegative_fields) {
        if (source.float_values[field] < 0.0) {
            return kAssembleFloatRangeInvalid;
        }
    }
    constexpr struct {
        abi::FloatFieldV2 field;
        abi::DefinedBitV1 bit;
    } ratio_fields[] = {
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
    };
    for (const auto& entry : ratio_fields) {
        if (abi::test_defined_bit(source.defined_mask, entry.bit)) {
            const double value = source.float_values[entry.field];
            if (value < 0.0 || value > 1.0) {
                return kAssembleFloatRangeInvalid;
            }
        }
    }
    return kAssembleSuccess;
}

EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE inline AssembleStatusV1
validate_nullable_storage(const CompactSummaryBoundarySourceV1& source) noexcept {
    constexpr struct {
        abi::DefinedBitV1 bit;
        abi::IntegerFieldV2 field;
    } nullable_integers[] = {
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
    };
    for (const auto& entry : nullable_integers) {
        if (!abi::test_defined_bit(source.defined_mask, entry.bit)
            && source.integer_values[entry.field] != 0) {
            return kAssembleUndefinedStorageNonzero;
        }
    }
    constexpr struct {
        abi::DefinedBitV1 bit;
        abi::FloatFieldV2 field;
    } nullable_floats[] = {
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
    };
    for (const auto& entry : nullable_floats) {
        if (!abi::test_defined_bit(source.defined_mask, entry.bit)
            && !raw_double_is_zero(source.float_values[entry.field])) {
            return kAssembleUndefinedStorageNonzero;
        }
    }
    if (!abi::test_defined_bit(
            source.defined_mask,
            abi::kDefinedTerminalRecoverableErrorDigestSha256
        )
        && !bytes_are_zero(
            source.digest_values
                + abi::digest_field_offset(
                    abi::kDigestTerminalRecoverableErrorDigestSha256
                ),
            abi::kDigestBytesV2
        )) {
        return kAssembleUndefinedStorageNonzero;
    }
    return kAssembleSuccess;
}

EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE inline AssembleStatusV1
validate_text(const CompactSummaryBoundarySourceV1& source) noexcept {
    const std::uint8_t* const record = source.dominant_lineage_id_record;
    const bool defined = abi::test_defined_bit(
        source.defined_mask,
        abi::kDefinedLineageMetricsDominantLineageId
    );
    if (!defined) {
        return bytes_are_zero(record, abi::kTextRecordBytesV2)
            ? kAssembleSuccess
            : kAssembleUndefinedStorageNonzero;
    }
    const std::uint8_t length = record[abi::kTextLengthByteIndexV2];
    if (length == 0 || length > abi::kTextMaximumPayloadBytesV2
        || !bytes_are_zero(
            record + 1 + length,
            abi::kTextMaximumPayloadBytesV2 - length
        )
        || !is_valid_utf8(record + 1, length)) {
        return kAssembleTextInvalid;
    }
    return kAssembleSuccess;
}

EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE inline AssembleStatusV1
validate_flat_patch_storage(
    const CompactSummaryBoundarySourceV1& source,
    const bool patch_present
) noexcept {
    if (patch_present) {
        return kAssembleSuccess;
    }
    for (
        std::int64_t index = abi::kIntegerPatchMetricsPatchMovementEnergyEventCount;
        index < abi::kIntegerFieldCountV2;
        ++index
    ) {
        if (source.integer_values[index] != 0) {
            return kAssembleFlatPatchStorageNonzero;
        }
    }
    for (
        std::int64_t index = abi::kFloatPatchMetricsPatchFoodHarvestTotal;
        index < abi::kFloatFieldCountV2;
        ++index
    ) {
        if (!raw_double_is_zero(source.float_values[index])) {
            return kAssembleFlatPatchStorageNonzero;
        }
    }
    return kAssembleSuccess;
}

EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE inline AssembleStatusV1
validate_terminal_consistency(
    const CompactSummaryBoundarySourceV1& source
) noexcept {
    const bool recoverable_code = abi::test_defined_bit(
        source.defined_mask, abi::kDefinedTerminalRecoverableErrorCode
    );
    const bool recoverable_digest = abi::test_defined_bit(
        source.defined_mask,
        abi::kDefinedTerminalRecoverableErrorDigestSha256
    );
    const bool recoverable_present =
        source.terminal_recoverable_error_status_code
        == abi::kRecoverableErrorStatusPresent;
    if (recoverable_present != (recoverable_code && recoverable_digest)
        || recoverable_code != recoverable_digest) {
        return kAssembleRecoverableStateInconsistent;
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
    const bool failed = source.terminal_status_code == abi::kTerminalStatusFailed;
    const bool sticky = source.terminal_sticky_failure_code
        == abi::kBooleanStatusTrue;
    if (failed) {
        if (!sticky || !failure_generation || !failure_tick || !failure_stage) {
            return kAssembleTerminalStateInconsistent;
        }
    } else if (sticky || failure_generation || failure_tick || failure_stage) {
        return kAssembleTerminalStateInconsistent;
    }
    return kAssembleSuccess;
}

EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE inline AssembleStatusV1
validate_source(const CompactSummaryBoundarySourceV1& source) noexcept {
    if (!source_pointers_valid(source)) {
        return kAssembleNullPointer;
    }
    if (source.source_version != kBoundarySourceVersionV1) {
        return kAssembleSourceVersionMismatch;
    }
    if (!abi::defined_mask_has_only_frozen_bits(source.defined_mask)) {
        return kAssembleReservedDefinedBit;
    }
    constexpr abi::DefinedMaskV1 patch_children =
        abi::kUsedDefinedMaskV1
        & ~((abi::DefinedMaskV1{1}
             << static_cast<std::uint32_t>(abi::kDefinedPatchMetrics))
            - 1);
    const bool patch_present = abi::test_defined_bit(
        source.defined_mask, abi::kDefinedPatchMetrics
    );
    if (!patch_present && (source.defined_mask & patch_children) != 0) {
        return kAssemblePatchChildWithoutObject;
    }
    AssembleStatusV1 status = validate_statuses(source);
    if (status != kAssembleSuccess) {
        return status;
    }
    if ((source.world_code == abi::kWorldPatch) != patch_present) {
        return kAssembleWorldPatchPresenceMismatch;
    }
    status = validate_integers(source);
    if (status != kAssembleSuccess) {
        return status;
    }
    status = validate_floats(source);
    if (status != kAssembleSuccess) {
        return status;
    }
    status = validate_cross_field_invariants(source);
    if (status != kAssembleSuccess) {
        return status;
    }
    status = validate_nullable_storage(source);
    if (status != kAssembleSuccess) {
        return status;
    }
    status = validate_flat_patch_storage(source, patch_present);
    if (status != kAssembleSuccess) {
        return status;
    }
    status = validate_terminal_consistency(source);
    if (status != kAssembleSuccess) {
        return status;
    }
    return validate_text(source);
}

EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE inline void write_statuses(
    const CompactSummaryBoundarySourceV1& source,
    std::int32_t* output
) noexcept {
    output[abi::kStatusSummaryStatus] = source.summary_status_code;
    output[abi::kStatusWorld] = source.world_code;
    output[abi::kStatusTurnoverStatus] = source.turnover_status_code;
    output[abi::kStatusRngAuthorityComplete] =
        source.rng_authority_complete_code;
    output[abi::kStatusRngExportNonConsuming] =
        source.rng_export_non_consuming_code;
    output[abi::kStatusTerminalStatus] = source.terminal_status_code;
    output[abi::kStatusTerminalRecoverableErrorStatus] =
        source.terminal_recoverable_error_status_code;
    output[abi::kStatusTerminalStickyFailure] =
        source.terminal_sticky_failure_code;
    output[abi::kStatusCheckpointDue] = source.checkpoint_due_code;
    output[abi::kStatusLineageMetricsLineageCollapseWarning] =
        source.lineage_collapse_warning_code;
    output[abi::kStatusTerminalRecoverableErrorCode] =
        source.terminal_recoverable_error_code;
    output[abi::kStatusTerminalFailureStage] =
        source.terminal_failure_stage_code;
}

EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE inline AssembleStatusV1
assemble_compact_summary_scratch_row(
    const CompactSummaryBoundarySourceV1& source,
    const CompactSummaryScratchRowV1& scratch
) noexcept {
    if (!scratch_pointers_valid(scratch)) {
        return kAssembleNullPointer;
    }
    const AssembleStatusV1 status = validate_source(source);
    if (status != kAssembleSuccess) {
        clear_scratch_row(scratch);
        return status;
    }

    *scratch.ready = 0;
    *scratch.validity = 0;
    *scratch.defined_mask = source.defined_mask;
    write_statuses(source, scratch.status_codes);
    for (std::int64_t index = 0; index < abi::kIntegerFieldCountV2; ++index) {
        scratch.integer_values[index] = source.integer_values[index];
    }
    for (std::int64_t index = 0; index < abi::kFloatFieldCountV2; ++index) {
        scratch.float_values[index] = source.float_values[index];
    }
    for (
        std::int64_t index = 0;
        index < abi::kDigestFieldCountV2 * abi::kDigestBytesV2;
        ++index
    ) {
        scratch.digest_values[index] = source.digest_values[index];
    }
    for (std::int64_t index = 0; index < abi::kTextRecordBytesV2; ++index) {
        scratch.text_values[index] = source.dominant_lineage_id_record[index];
    }
    *scratch.validity = 1;
    *scratch.ready = 1;
    return kAssembleSuccess;
}

static_assert(std::is_standard_layout_v<CompactSummaryBoundarySourceV1>);
static_assert(std::is_trivially_copyable_v<CompactSummaryBoundarySourceV1>);
static_assert(std::is_standard_layout_v<CompactSummaryScratchRowV1>);
static_assert(std::is_trivially_copyable_v<CompactSummaryScratchRowV1>);

}  // namespace eiel::cuda_architecture_b::compact_summary_assembler_v1

#undef EIEL_COMPACT_ASSEMBLER_V1_HOST_DEVICE
