#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__CUDACC__)
#define EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE __host__ __device__
#else
#define EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE
#endif

namespace eiel::cuda_architecture_b::compact_summary_v2 {

inline constexpr std::uint32_t kNativeAbiVersionV2 = 2;
inline constexpr std::uint32_t kWireSchemaVersionV1 = 1;
inline constexpr std::uint32_t kCodebookVersionV1 = 1;
inline constexpr std::uint32_t kDefinedMaskVersionV1 = 1;
inline constexpr std::uint32_t kPlaneLayoutVersionV2 = 2;
inline constexpr std::uint32_t kProjectionContractVersionV2 = 2;
inline constexpr std::uint32_t kInvalidRowContractVersionV2 = 2;

inline constexpr char kNativeAbiSchemaVersionV2[] =
    "eiel_compact_summary_native_abi_v2";
inline constexpr char kWireSchemaVersionNameV1[] =
    "eiel_compact_generation_summary_v1";
inline constexpr char kCodebookSchemaVersionV1[] =
    "eiel_compact_summary_codebooks_v1";
inline constexpr char kDefinedMaskSchemaVersionV1[] =
    "eiel_compact_summary_defined_mask_v1";
inline constexpr char kPlaneLayoutSchemaVersionV2[] =
    "eiel_compact_summary_plane_layout_v2";
inline constexpr char kProjectionContractSchemaVersionV2[] =
    "eiel_compact_summary_projection_contract_v2";
inline constexpr char kInvalidRowContractSchemaVersionV2[] =
    "eiel_compact_summary_invalid_row_contract_v2";

enum CompactSummaryPlaneV2 : std::int64_t {
    kPlaneValidity = 0,
    kPlaneDefinedMask,
    kPlaneStatusCodes,
    kPlaneIntegerValues,
    kPlaneFloatValues,
    kPlaneDigestValues,
    kPlaneTextValues,
    kPlaneCountV2,
};

enum StatusFieldV2 : std::int64_t {
    kStatusSummaryStatus = 0,
    kStatusWorld,
    kStatusTurnoverStatus,
    kStatusRngAuthorityComplete,
    kStatusRngExportNonConsuming,
    kStatusTerminalStatus,
    kStatusTerminalRecoverableErrorStatus,
    kStatusTerminalStickyFailure,
    kStatusCheckpointDue,
    kStatusLineageMetricsLineageCollapseWarning,
    kStatusTerminalRecoverableErrorCode,
    kStatusTerminalFailureStage,
    kStatusFieldCountV2,
};

enum SummaryStatusCodeV1 : std::int32_t {
    kSummaryStatusComplete = 1,
};

enum WorldCodeV1 : std::int32_t {
    kWorldFlat = 1,
    kWorldPatch = 2,
};

enum TurnoverStatusCodeV1 : std::int32_t {
    kTurnoverStatusCommitted = 1,
};

enum BooleanStatusCodeV1 : std::int32_t {
    kBooleanStatusFalse = 1,
    kBooleanStatusTrue = 2,
};

enum TerminalStatusCodeV1 : std::int32_t {
    kTerminalStatusNotTerminal = 1,
    kTerminalStatusRunComplete = 2,
    kTerminalStatusCancelled = 3,
    kTerminalStatusFailed = 4,
};

enum RecoverableErrorStatusCodeV1 : std::int32_t {
    kRecoverableErrorStatusNone = 1,
    kRecoverableErrorStatusPresent = 2,
};

enum RecoverableErrorCodeV1 : std::int32_t {
    kRecoverableErrorExactReportingFailed = 1,
    kRecoverableErrorCheckpointPersistenceFailed = 2,
    kRecoverableErrorOutputFlushFailed = 3,
};

enum FailureStageCodeV1 : std::int32_t {
    kFailureStageTickCoreAndBirthPlan = 1,
    kFailureStageBirthPrepare = 2,
    kFailureStageBirthTransactionAndSettlement = 3,
    kFailureStageBirthEventAppend = 4,
    kFailureStageImportValidation = 5,
    kFailureStageImportCopy = 6,
    kFailureStageRunRequestValidation = 7,
    kFailureStageGeneration = 8,
    kFailureStageTurnover = 9,
    kFailureStageSnapshot = 10,
    kFailureStageCancellation = 11,
    kFailureStageFinalization = 12,
    kFailureStageApiState = 13,
    kFailureStageTypedLifecyclePublicOutput = 14,
    kFailureStageBehaviorPublicOutput = 15,
    kFailureStageGenerationResultValidation = 16,
};

enum IntegerFieldV2 : std::int64_t {
    kIntegerGenerationStartGeneration = 0,
    kIntegerGenerationStartTick,
    kIntegerGenerationEndGeneration,
    kIntegerGenerationEndTickExclusive,
    kIntegerNextGenerationStartGeneration,
    kIntegerNextGenerationStartTick,
    kIntegerAlive,
    kIntegerBirths,
    kIntegerFailedBirths,
    kIntegerDeaths,
    kIntegerDeathsByClassEnergyDepleted,
    kIntegerDeathsByClassHealthDepleted,
    kIntegerTurnoverCount,
    kIntegerEventsCursorStart,
    kIntegerEventsCursorPreTurnover,
    kIntegerEventsCursorEnd,
    kIntegerEventsAuthoritativeCursor,
    kIntegerEventsEventRowCount,
    kIntegerEventsGenerationEventCount,
    kIntegerEventsTurnoverEventCount,
    kIntegerEventsGenerationCountsByTypeBirthCommitted,
    kIntegerEventsGenerationCountsByTypeMutation,
    kIntegerEventsGenerationCountsByTypeBirthFailed,
    kIntegerEventsGenerationCountsByTypeDeathCommitted,
    kIntegerEventsTurnoverCountsByTypeBirthCommitted,
    kIntegerEventsTurnoverCountsByTypeMutation,
    kIntegerEventsTurnoverCountsByTypeBirthFailed,
    kIntegerEventsTurnoverCountsByTypeDeathCommitted,
    kIntegerIdentityPopulationNextId,
    kIntegerIdentityActiveAgentCount,
    kIntegerIdentityActiveAgentIdMin,
    kIntegerIdentityActiveAgentIdMax,
    kIntegerIdentityFreeSlotCount,
    kIntegerRngPolicyMutationTapeCursor,
    kIntegerTerminalFailureGeneration,
    kIntegerTerminalFailureTick,
    kIntegerGenerationMetricsCapacity,
    kIntegerGenerationMetricsFreeSlots,
    kIntegerGenerationMetricsBirthAttempts,
    kIntegerGenerationMetricsNativeFailedBirthEventRows,
    kIntegerGenerationMetricsInjuryBlockedBirths,
    kIntegerGenerationMetricsMoved,
    kIntegerGenerationMetricsGathered,
    kIntegerGenerationMetricsMovementAttempts,
    kIntegerGenerationMetricsMovementBlockedOccupied,
    kIntegerGenerationMetricsMovementBlockedBounds,
    kIntegerGenerationMetricsPolicyLoopTickCalls,
    kIntegerGenerationMetricsPolicyRefillCalls,
    kIntegerGenerationMetricsPolicyRefillRows,
    kIntegerGenerationMetricsPolicyCacheHitCalls,
    kIntegerGenerationMetricsPolicyCacheRowsRequested,
    kIntegerGenerationMetricsPolicyCacheRowsHit,
    kIntegerGenerationMetricsPolicyCacheRowsMissed,
    kIntegerGenerationMetricsPolicyCacheRowsRefilled,
    kIntegerGenerationMetricsPolicyCacheFullRefillTicks,
    kIntegerGenerationMetricsPolicyCachePartialRefillTicks,
    kIntegerBehaviorMetricsRowCount,
    kIntegerBehaviorMetricsCohortSize,
    kIntegerBehaviorMetricsCohortSurvivors,
    kIntegerBehaviorMetricsActionTotal,
    kIntegerBehaviorMetricsMoveTotal,
    kIntegerBehaviorMetricsMoveN,
    kIntegerBehaviorMetricsMoveS,
    kIntegerBehaviorMetricsMoveE,
    kIntegerBehaviorMetricsMoveW,
    kIntegerBehaviorMetricsGather,
    kIntegerBehaviorMetricsRest,
    kIntegerBehaviorMetricsAttack,
    kIntegerLineageMetricsRecordCount,
    kIntegerLineageMetricsActiveLineages,
    kIntegerLineageMetricsRarePersistentLineages,
    kIntegerLineageMetricsLineageCollapseStreak,
    kIntegerPatchMetricsPatchMovementEnergyEventCount,
    kIntegerPatchMetricsPatchSpawnSyncRows,
    kIntegerPatchMetricsTypedDeathEventRows,
    kIntegerPatchMetricsEnvRngPaddingDraws,
    kIntegerFieldCountV2,
};

enum FloatFieldV2 : std::int64_t {
    kFloatGenerationMetricsPeakInjury = 0,
    kFloatLineageMetricsDominantLineagePopulationShare,
    kFloatLineageMetricsTop3LineagePopulationShare,
    kFloatLineageMetricsLineagePopulationEntropy,
    kFloatLineageMetricsLineageReproductionEntropy,
    kFloatLineageMetricsDominantLineageReproductionShare,
    kFloatPatchMetricsPatchFoodHarvestTotal,
    kFloatPatchMetricsPatchScentEmitTotal,
    kFloatPatchMetricsPatchWasteEmitTotal,
    kFloatPatchMetricsPatchMovementEnergyTotal,
    kFloatPatchMetricsEnergyCollectedDelta,
    kFloatPatchMetricsPatchScentMoveQueuedTotal,
    kFloatPatchMetricsPatchWasteMoveQueuedTotal,
    kFloatPatchMetricsPatchScentGatherQueuedTotal,
    kFloatPatchMetricsPatchWasteGatherQueuedTotal,
    kFloatPatchMetricsPatchMovementEnergyExtraTotal,
    kFloatPatchMetricsPatchScentEmittedTotal,
    kFloatPatchMetricsPatchWasteEmittedTotal,
    kFloatPatchMetricsPatchFoodDepletedFrac,
    kFloatPatchMetricsPatchFoodFullFrac,
    kFloatPatchMetricsPatchDensityHotspotFrac,
    kFloatFieldCountV2,
};

enum DigestFieldV2 : std::int64_t {
    kDigestEventsActivePrefixDigestSha256 = 0,
    kDigestIdentityActiveAgentIdsDigestSha256,
    kDigestRngAuthorityDigestSha256,
    kDigestRngPolicyMutationTapeDigestSha256,
    kDigestTerminalRecoverableErrorDigestSha256,
    kDigestStateDigestsAuthoritativeStateDigestSha256,
    kDigestStateDigestsCanonicalCheckpointStorageDigestSha256,
    kDigestStateDigestsNextGenerationStartStateDigestSha256,
    kDigestBehaviorMetricsRowsDigestSha256,
    kDigestLineageMetricsRecordsDigestSha256,
    kDigestFieldCountV2,
};

enum TextFieldV2 : std::int64_t {
    kTextLineageMetricsDominantLineageId = 0,
    kTextFieldCountV2,
};

enum DefinedBitV1 : std::uint32_t {
    kDefinedIdentityActiveAgentIdMin = 0,
    kDefinedIdentityActiveAgentIdMax,
    kDefinedTerminalRecoverableErrorCode,
    kDefinedTerminalRecoverableErrorDigestSha256,
    kDefinedTerminalFailureGeneration,
    kDefinedTerminalFailureTick,
    kDefinedTerminalFailureStage,
    kDefinedLineageMetricsActiveLineages,
    kDefinedLineageMetricsDominantLineageId,
    kDefinedLineageMetricsDominantLineagePopulationShare,
    kDefinedLineageMetricsTop3LineagePopulationShare,
    kDefinedLineageMetricsLineagePopulationEntropy,
    kDefinedLineageMetricsLineageReproductionEntropy,
    kDefinedLineageMetricsDominantLineageReproductionShare,
    kDefinedLineageMetricsRarePersistentLineages,
    kDefinedLineageMetricsLineageCollapseWarning,
    kDefinedLineageMetricsLineageCollapseStreak,
    kDefinedPatchMetrics,
    kDefinedPatchMetricsPatchSpawnSyncRows,
    kDefinedPatchMetricsTypedDeathEventRows,
    kDefinedPatchMetricsEnvRngPaddingDraws,
    kDefinedPatchMetricsEnergyCollectedDelta,
    kDefinedPatchMetricsPatchScentMoveQueuedTotal,
    kDefinedPatchMetricsPatchWasteMoveQueuedTotal,
    kDefinedPatchMetricsPatchScentGatherQueuedTotal,
    kDefinedPatchMetricsPatchWasteGatherQueuedTotal,
    kDefinedPatchMetricsPatchMovementEnergyExtraTotal,
    kDefinedPatchMetricsPatchScentEmittedTotal,
    kDefinedPatchMetricsPatchWasteEmittedTotal,
    kDefinedPatchMetricsPatchFoodDepletedFrac,
    kDefinedPatchMetricsPatchFoodFullFrac,
    kDefinedPatchMetricsPatchDensityHotspotFrac,
    kDefinedBitCountV1,
};

using DefinedMaskV1 = std::uint64_t;

inline constexpr std::int64_t kValidityFieldCountV2 = 1;
inline constexpr std::int64_t kDefinedMaskFieldCountV2 = 1;
inline constexpr std::int64_t kDigestBytesV2 = 32;
inline constexpr std::int64_t kTextFieldCount = kTextFieldCountV2;
inline constexpr std::int64_t kTextRecordBytesV2 = 65;
inline constexpr std::int64_t kTextLengthByteIndexV2 = 0;
inline constexpr std::int64_t kTextMaximumPayloadBytesV2 = 64;
inline constexpr std::int64_t kUsedDefinedBitCountV1 = 32;
inline constexpr std::int64_t kReservedDefinedBitBeginV1 = 32;
inline constexpr std::int64_t kReservedDefinedBitEndV1 = 63;

inline constexpr std::size_t kValidityLogicalBytesPerRowV2 =
    sizeof(std::uint8_t) * kValidityFieldCountV2;
inline constexpr std::size_t kDefinedMaskLogicalBytesPerRowV2 =
    sizeof(DefinedMaskV1) * kDefinedMaskFieldCountV2;
inline constexpr std::size_t kStatusLogicalBytesPerRowV2 =
    sizeof(std::int32_t) * kStatusFieldCountV2;
inline constexpr std::size_t kIntegerLogicalBytesPerRowV2 =
    sizeof(std::int64_t) * kIntegerFieldCountV2;
inline constexpr std::size_t kFloatLogicalBytesPerRowV2 =
    sizeof(double) * kFloatFieldCountV2;
inline constexpr std::size_t kDigestLogicalBytesPerRowV2 =
    sizeof(std::uint8_t) * kDigestFieldCountV2 * kDigestBytesV2;
inline constexpr std::size_t kTextLogicalBytesPerRowV2 =
    sizeof(std::uint8_t) * kTextFieldCountV2 * kTextRecordBytesV2;
inline constexpr std::size_t kLogicalBytesPerGenerationCellV2 =
    kValidityLogicalBytesPerRowV2
    + kDefinedMaskLogicalBytesPerRowV2
    + kStatusLogicalBytesPerRowV2
    + kIntegerLogicalBytesPerRowV2
    + kFloatLogicalBytesPerRowV2
    + kDigestLogicalBytesPerRowV2
    + kTextLogicalBytesPerRowV2;

inline constexpr DefinedMaskV1 kUsedDefinedMaskV1 = 0x00000000FFFFFFFFULL;
inline constexpr DefinedMaskV1 kReservedDefinedMaskV1 =
    0xFFFFFFFF00000000ULL;
inline constexpr std::int32_t kReservedInvalidStatusCodeV2 = 0;

enum InvalidRowKindV2 : std::uint32_t {
    kInvalidRowFailedGenerationSlot = 1U << 0,
    kInvalidRowStickyNoopTail = 1U << 1,
    kInvalidRowUnexecutedSlot = 1U << 2,
    kInvalidRowProducerValidationFailure = 1U << 3,
};

inline constexpr std::uint32_t kAllInvalidRowKindsV2 =
    kInvalidRowFailedGenerationSlot
    | kInvalidRowStickyNoopTail
    | kInvalidRowUnexecutedSlot
    | kInvalidRowProducerValidationFailure;

enum RowWritePhaseV2 : std::int64_t {
    kWriteDefinedMask = 0,
    kWriteStatusCodes,
    kWriteIntegerValues,
    kWriteFloatValues,
    kWriteDigestValues,
    kWriteTextValues,
    kWriteValidityLast,
    kRowWritePhaseCountV2,
};

struct InvalidRowContractMetadataV2 {
    std::uint32_t contract_version;
    std::uint32_t applies_to;
    std::uint8_t validity;
    std::uint8_t digest_byte;
    std::uint8_t text_length;
    std::uint8_t text_tail_byte;
    DefinedMaskV1 defined_mask;
    std::int32_t status_code;
    std::int32_t reserved;
    std::int64_t integer_value;
    double float_value;
    bool float_is_positive_zero;
    bool validity_written_last;
    bool canonical_projection_permitted;
    bool completed_generation_count_includes_row;
    bool completed_rows_before_failure_remain_valid_and_immutable;
    bool later_simulation_mutation_after_sticky_failure;
    bool default_values_imply_definedness;
    bool phase_a_nan_telemetry_is_separate_authority;
};

inline constexpr InvalidRowContractMetadataV2 kInvalidRowContractV2 = {
    kInvalidRowContractVersionV2,
    kAllInvalidRowKindsV2,
    0,
    0,
    0,
    0,
    0,
    kReservedInvalidStatusCodeV2,
    0,
    0,
    0.0,
    true,
    true,
    false,
    false,
    true,
    false,
    false,
    true,
};

template <typename T>
struct StridedPlaneViewV2 {
    T* data;
    // Strides are measured in T elements. Digest/text plane elements are bytes.
    std::int64_t generation_stride;
    std::int64_t cell_stride;
};

// Inner field, digest-byte, and text-byte axes are frozen contiguous row data.
struct CompactSummaryViewV2 {
    std::int64_t generation_count;
    std::int64_t cell_count;
    StridedPlaneViewV2<std::uint8_t> validity;
    StridedPlaneViewV2<DefinedMaskV1> defined_mask;
    StridedPlaneViewV2<std::int32_t> status_codes;
    StridedPlaneViewV2<std::int64_t> integer_values;
    StridedPlaneViewV2<double> float_values;
    StridedPlaneViewV2<std::uint8_t> digest_values;
    StridedPlaneViewV2<std::uint8_t> text_values;
};

EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE constexpr std::int64_t row_offset(
    const std::int64_t generation,
    const std::int64_t cell,
    const std::int64_t generation_stride,
    const std::int64_t cell_stride
) noexcept {
    return generation * generation_stride + cell * cell_stride;
}

template <typename T>
EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE constexpr std::int64_t row_offset(
    const StridedPlaneViewV2<T>& plane,
    const std::int64_t generation,
    const std::int64_t cell
) noexcept {
    return row_offset(
        generation,
        cell,
        plane.generation_stride,
        plane.cell_stride
    );
}

template <typename T>
EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE constexpr T* row_pointer(
    const StridedPlaneViewV2<T>& plane,
    const std::int64_t generation,
    const std::int64_t cell
) noexcept {
    return plane.data + row_offset(plane, generation, cell);
}

EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE constexpr std::int64_t
contiguous_row_index(
    const std::int64_t generation,
    const std::int64_t cell,
    const std::int64_t cell_count
) noexcept {
    return generation * cell_count + cell;
}

EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE constexpr std::int64_t
validity_row_offset(
    const std::int64_t generation,
    const std::int64_t cell,
    const std::int64_t cell_count
) noexcept {
    return contiguous_row_index(generation, cell, cell_count);
}

EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE constexpr std::int64_t
defined_mask_row_offset(
    const std::int64_t generation,
    const std::int64_t cell,
    const std::int64_t cell_count
) noexcept {
    return contiguous_row_index(generation, cell, cell_count);
}

EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE constexpr std::int64_t status_row_offset(
    const std::int64_t generation,
    const std::int64_t cell,
    const std::int64_t cell_count
) noexcept {
    return contiguous_row_index(generation, cell, cell_count)
        * kStatusFieldCountV2;
}

EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE constexpr std::int64_t integer_row_offset(
    const std::int64_t generation,
    const std::int64_t cell,
    const std::int64_t cell_count
) noexcept {
    return contiguous_row_index(generation, cell, cell_count)
        * kIntegerFieldCountV2;
}

EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE constexpr std::int64_t float_row_offset(
    const std::int64_t generation,
    const std::int64_t cell,
    const std::int64_t cell_count
) noexcept {
    return contiguous_row_index(generation, cell, cell_count)
        * kFloatFieldCountV2;
}

EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE constexpr std::int64_t digest_row_offset(
    const std::int64_t generation,
    const std::int64_t cell,
    const std::int64_t cell_count
) noexcept {
    return contiguous_row_index(generation, cell, cell_count)
        * kDigestFieldCountV2 * kDigestBytesV2;
}

EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE constexpr std::int64_t text_row_offset(
    const std::int64_t generation,
    const std::int64_t cell,
    const std::int64_t cell_count
) noexcept {
    return contiguous_row_index(generation, cell, cell_count)
        * kTextFieldCountV2 * kTextRecordBytesV2;
}

EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE constexpr std::int64_t digest_field_offset(
    const DigestFieldV2 field
) noexcept {
    return static_cast<std::int64_t>(field) * kDigestBytesV2;
}

EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE constexpr std::int64_t text_field_offset(
    const TextFieldV2 field
) noexcept {
    return static_cast<std::int64_t>(field) * kTextRecordBytesV2;
}

EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE constexpr bool is_defined_bit(
    const std::uint32_t bit
) noexcept {
    return bit < static_cast<std::uint32_t>(kDefinedBitCountV1);
}

EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE constexpr DefinedMaskV1 defined_bit_mask(
    const DefinedBitV1 bit
) noexcept {
    const std::uint32_t index = static_cast<std::uint32_t>(bit);
    return is_defined_bit(index) ? (DefinedMaskV1{1} << index) : 0;
}

EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE constexpr DefinedMaskV1
set_defined_bit(
    const DefinedMaskV1 mask,
    const DefinedBitV1 bit
) noexcept {
    return mask | defined_bit_mask(bit);
}

EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE constexpr bool test_defined_bit(
    const DefinedMaskV1 mask,
    const DefinedBitV1 bit
) noexcept {
    return (mask & defined_bit_mask(bit)) != 0;
}

EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE constexpr bool
defined_mask_has_only_frozen_bits(const DefinedMaskV1 mask) noexcept {
    return (mask & kReservedDefinedMaskV1) == 0;
}

EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE constexpr std::int32_t
reserved_invalid_status_code(const StatusFieldV2) noexcept {
    return kReservedInvalidStatusCodeV2;
}

EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE constexpr std::int32_t
maximum_known_status_code(const StatusFieldV2 field) noexcept {
    switch (field) {
        case kStatusSummaryStatus:
        case kStatusTurnoverStatus:
            return 1;
        case kStatusWorld:
        case kStatusRngAuthorityComplete:
        case kStatusRngExportNonConsuming:
        case kStatusTerminalRecoverableErrorStatus:
        case kStatusTerminalStickyFailure:
        case kStatusCheckpointDue:
        case kStatusLineageMetricsLineageCollapseWarning:
            return 2;
        case kStatusTerminalStatus:
            return 4;
        case kStatusTerminalRecoverableErrorCode:
            return 3;
        case kStatusTerminalFailureStage:
            return 16;
        default:
            return 0;
    }
}

EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE constexpr bool is_known_status_code(
    const StatusFieldV2 field,
    const std::int32_t code
) noexcept {
    return code > 0 && code <= maximum_known_status_code(field);
}

EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE constexpr bool
is_reserved_or_known_status_code(
    const StatusFieldV2 field,
    const std::int32_t code
) noexcept {
    return code == kReservedInvalidStatusCodeV2
        || is_known_status_code(field, code);
}

static_assert(sizeof(std::uint8_t) == 1);
static_assert(sizeof(std::int32_t) == 4);
static_assert(sizeof(std::int64_t) == 8);
static_assert(sizeof(DefinedMaskV1) == 8);
static_assert(sizeof(double) == 8);
static_assert(std::numeric_limits<double>::is_iec559);
static_assert(kPlaneCountV2 == 7);
static_assert(kStatusFieldCountV2 == 12);
static_assert(kIntegerFieldCountV2 == 76);
static_assert(kFloatFieldCountV2 == 21);
static_assert(kDigestFieldCountV2 == 10);
static_assert(kTextFieldCountV2 == 1);
static_assert(kDefinedBitCountV1 == 32);
static_assert(kValidityLogicalBytesPerRowV2 == 1);
static_assert(kDefinedMaskLogicalBytesPerRowV2 == 8);
static_assert(kStatusLogicalBytesPerRowV2 == 48);
static_assert(kIntegerLogicalBytesPerRowV2 == 608);
static_assert(kFloatLogicalBytesPerRowV2 == 168);
static_assert(kDigestLogicalBytesPerRowV2 == 320);
static_assert(kTextLogicalBytesPerRowV2 == 65);
static_assert(kLogicalBytesPerGenerationCellV2 == 1218);
static_assert(std::is_standard_layout_v<InvalidRowContractMetadataV2>);
static_assert(std::is_trivially_copyable_v<InvalidRowContractMetadataV2>);
static_assert(std::is_standard_layout_v<StridedPlaneViewV2<std::uint8_t>>);
static_assert(std::is_trivially_copyable_v<StridedPlaneViewV2<std::uint8_t>>);
static_assert(std::is_standard_layout_v<CompactSummaryViewV2>);
static_assert(std::is_trivially_copyable_v<CompactSummaryViewV2>);

}  // namespace eiel::cuda_architecture_b::compact_summary_v2

#undef EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE
