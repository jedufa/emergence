#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include "cuda_architecture_b_cross_path_resident_control_v1.cuh"

namespace eiel::cuda_architecture_b::genome_v2_causal_output_v1 {

constexpr int64_t kOrdinaryBirthEventKind = 1;
constexpr int64_t kTurnoverBirthEventKind = 2;
constexpr int64_t kUnavailableValue = INT64_MIN;
constexpr int64_t kRecordCapacity = 1024;

enum RecordColumnV1 : int64_t {
    kRecordOrdinal = 0,
    kEventKind,
    kCellOrdinal,
    kGeneration,
    kTick,
    kEventSequence,
    kEventSequenceValid,
    kChildAgentId,
    kParentAgentId,
    kDestinationSlot,
    kLineageFounderId,
    kLineageDepth,
    kAllelesQ10,
    kModifiersQ10 = kAllelesQ10 + 2 * 12 * 4,
    kProductsQ10 = kModifiersQ10 + 2 * 12,
    kRegulatorValuesValid = kProductsQ10 + 12,
    kPhenotypesQ10,
    kContextValid = kPhenotypesQ10 + 10,
    kContextQ10,
    kExpressionCostQ10 = kContextQ10 + 8,
    kRegulatoryCostValid,
    kRegulatoryCostQ10,
    kRuntimeProjectionQ10,
    // Append-only V2 field. Columns 0..178 retain the V1 interpretation.
    kMateParentAgentId = kRuntimeProjectionQ10 + 12,
    // Append-only V3 fields. Columns 0..179 retain the V2 interpretation.
    kParentSlot,
    kMateParentSlot,
    kRecordWidth,
};

static_assert(kRecordWidth == 182);

}  // namespace eiel::cuda_architecture_b::genome_v2_causal_output_v1

namespace eiel::cuda_architecture_b::flattened_turnover_v1 {

constexpr int64_t kMaximumCapacity = 768;
constexpr int64_t kMaximumPending = kMaximumCapacity;
constexpr int64_t kMaximumReleaseRows = kMaximumCapacity + kMaximumPending;
constexpr int64_t kCpythonWordCount = 624;
constexpr int64_t kMaximumScratchBytesPerCell = 163840;
constexpr int64_t kScratchRowAlignment = 16;
constexpr int64_t kScratchCellAlignment = 256;

enum TurnoverMutableTensorIndex : int32_t {
    kActiveAgentIds = 0,
    kActiveAgentSlots,
    kActiveCountState,
    kSlotToAgentId,
    kAgentIdToSlot,
    kSlotTokens,
    kAliveMask,
    kPosY,
    kPosX,
    kAgentGrid,
    kEnergy,
    kHealth,
    kInjury,
    kPolicyEnergyObservation,
    kRestStreak,
    kPendingFreeAgentIds,
    kPendingFreeSlots,
    kPendingFreeCount,
    kActionCacheActions,
    kActionCacheToken,
    kActionCacheCursor,
    kActionCacheNext,
    kLastAction,
    kLastLastAction,
    kActionSwitches,
    kActionOscillations,
    kEpisode0Switches,
    kEpisode0Oscillations,
    kSurvivalSteps,
    kSlotFreeQueue,
    kSlotFreeQueueCount,
    kPopulationNextId,
    kScar,
    kAge,
    kReproCooldown,
    kReproThreshold,
    kOffspringCount,
    kLineageFounderId,
    kLineageDepth,
    kLineageBirthGeneration,
    kLineageBirthTick,
    kPatchHabitatLineageId,
    kPythonRandomWords,
    kPythonRandomIndex,
    kPythonRandomHasGauss,
    kPythonRandomGaussNext,
    kPythonRandomRawDraws,
    kMutationPySemanticDraws,
    kGenomeHiddenSize,
    kGenomeMemoryDim,
    kGenomeSensorChannelMask,
    kGenomeSensorRadius,
    kGenomeIntrospectionChannelMask,
    kGenomeSelfRegulationControlMask,
    kGenomeSelfModificationTargetMask,
    kGenomeDirectedInheritanceChannelMask,
    kPolicyMemDim,
    kPolicyUseGru,
    kGenomeExplorationBias,
    kGenomeAggressionBias,
    kGenomeMutationRate,
    kGenomeReproductionThreshold,
    kGenomeActionTemperature,
    kGenomeDriveInertia,
    kGenomeDriveNoise,
    kGenomeDriveLogitGain,
    kGenomePredLr,
    kGenomeDriveSpHunger,
    kGenomeDriveSpPain,
    kGenomeDriveSpFatigue,
    kGenomeDriveSpFear,
    kGenomeExpPriorStrength,
    kGenomeExpPriorNoise,
    kGenomeDriveBirthNoise,
    kGruUpkeepCost,
    kPolicyAttackBias,
    kPolicyActionTemperature,
    kPolicyDriveLogitGain,
    kDriveInertia,
    kDriveNoise,
    kPredLr,
    kDrives,
    kBirthResetShock,
    kPopulationPcg64StateWords,
    kPopulationPcg64BufferWords,
    kMutationNpSemanticDraws,
    kChemistry,
    kMlpPack,
    kHmem,
    kMutationTorchCpuState,
    kPolicyBirthTapeCursor,
    kMutationTorchCpuSemanticDraws,
    kEnergyCollected,
    kEnergyCollectedExact,
    kAttacks,
    kEarlySurvivalSteps,
    kEarlyEnergyCollected,
    kEarlyEnergyCollectedExact,
    kEarlyInjuryGained,
    kEarlyLifeWeight,
    kPredErrSum,
    kPredErrCount,
    kFirstReproStep,
    kDevStageInitialized,
    kEpisode0PredErrSum,
    kEpisode0PredErrCount,
    kLatePredErrSum,
    kLatePredErrCount,
    kChemistryObservationFidelity,
    kChemistryObservationBias,
    kPrevDriveDelta,
    kDriveVolatility,
    kDriveJitter,
    kEpisode0DriveJitterSum,
    kLastEnergyDelta,
    kLastHealthDelta,
    kLastInjuryDelta,
    kSelfharmEvents,
    kEpisode0SelfharmEvents,
    kSharedEnvPolicyPcg64StateWords,
    kSharedEnvPolicyPcg64BufferWords,
    kPolicyNpSemanticDraws,
    kTurnoverMutableTensorCount,
};

// Each base address spans every cell. Source and transactional-shadow rows may
// use different cell strides: the MLP shadow only retains the bounded newborn
// rows, while every other tensor keeps the legacy matching layout.
struct TurnoverMutableTensorTriplet {
    uint8_t* source_address;
    uint8_t* shadow_address;
    int64_t source_bytes_per_cell;
    int64_t shadow_bytes_per_cell;
};

// The triplet array is caller-owned device memory with exactly
// kTurnoverMutableTensorCount entries. Copy helpers are lane-strided so either
// one thread or a whole block can move a cell without allocation.
struct FlatTurnoverMutableTensorTable {
    const TurnoverMutableTensorTriplet* tensors;

    __device__ __forceinline__ const TurnoverMutableTensorTriplet& entry(
        const TurnoverMutableTensorIndex tensor
    ) const {
        return tensors[static_cast<int32_t>(tensor)];
    }

    __device__ __forceinline__ uint8_t* source_bytes(
        const TurnoverMutableTensorIndex tensor,
        const int64_t cell
    ) const {
        const TurnoverMutableTensorTriplet& value = entry(tensor);
        return value.source_address + cell * value.source_bytes_per_cell;
    }

    __device__ __forceinline__ uint8_t* shadow_bytes(
        const TurnoverMutableTensorIndex tensor,
        const int64_t cell
    ) const {
        const TurnoverMutableTensorTriplet& value = entry(tensor);
        return value.shadow_address + cell * value.shadow_bytes_per_cell;
    }

    template <typename T>
    __device__ __forceinline__ T* source(
        const TurnoverMutableTensorIndex tensor,
        const int64_t cell
    ) const {
        return reinterpret_cast<T*>(source_bytes(tensor, cell));
    }

    template <typename T>
    __device__ __forceinline__ T* shadow(
        const TurnoverMutableTensorIndex tensor,
        const int64_t cell
    ) const {
        return reinterpret_cast<T*>(shadow_bytes(tensor, cell));
    }

    __device__ __forceinline__ void copy_source_to_shadow(
        const TurnoverMutableTensorIndex tensor,
        const int64_t cell,
        const int64_t first_byte,
        const int64_t byte_stride
    ) const {
        if (first_byte < 0 || byte_stride <= 0) {
            return;
        }
        const TurnoverMutableTensorTriplet& value = entry(tensor);
        uint8_t* const destination = shadow_bytes(tensor, cell);
        const uint8_t* const source = source_bytes(tensor, cell);
        for (
            int64_t byte = first_byte;
            byte < value.source_bytes_per_cell;
            byte += byte_stride
        ) {
            destination[byte] = source[byte];
        }
    }

    __device__ __forceinline__ void copy_shadow_to_source(
        const TurnoverMutableTensorIndex tensor,
        const int64_t cell,
        const int64_t first_byte,
        const int64_t byte_stride
    ) const {
        if (first_byte < 0 || byte_stride <= 0) {
            return;
        }
        const TurnoverMutableTensorTriplet& value = entry(tensor);
        uint8_t* const destination = source_bytes(tensor, cell);
        const uint8_t* const source = shadow_bytes(tensor, cell);
        for (
            int64_t byte = first_byte;
            byte < value.source_bytes_per_cell;
            byte += byte_stride
        ) {
            destination[byte] = source[byte];
        }
    }
};

enum TurnoverScratchCounterIndex : int32_t {
    kPlanRankedCount = 0,
    kPlanCulledCount,
    kPlanParentPoolCount,
    kPlanSurvivorCount,
    kPlanRefillCount,
    kCullReleasedCount,
    kCullPendingReleasedCount,
    kCullCulledReleasedCount,
    kRefillPlannedCount,
    kRefillPythonRawDrawsAdded,
    kRefillPythonSemanticDrawsAdded,
    kRefillSpawnRawDrawsAdded,
    kRefillSpawnSemanticDrawsAdded,
    kIdentityCommittedCount,
    kScalarCommittedCount,
    kScalarSemanticDrawsAdded,
    kDriveCommittedCount,
    kDriveBirthNoiseRows,
    kDriveExpPriorRows,
    kDriveStandardNormalRows,
    kDriveSemanticDrawsAdded,
    kChemistryCommittedCount,
    kPolicyCommittedCount,
    kPolicySemanticDrawsAdded,
    kPolicyMutationEvents,
    kPolicyMutatedElements,
    kAggregateCommittedCount,
    kAggregateFinalLiveCount,
    kTurnoverScratchCounterCount,
};

enum TurnoverScratchStatusIndex : int32_t {
    kPlanStatus = 0,
    kCullStatus,
    kRefillStatus,
    kIdentityStatus,
    kScalarStatus,
    kDriveStatus,
    kChemistryStatus,
    kPolicyStatus,
    kAggregateStatus,
    kTurnoverScratchStatusCount,
};

// Public failure diagnostics preserve zero as success while encoding the first
// failing internal phase and its typed substatus in one existing int32 output.
// This avoids widening the resident ABI solely to make fail-closed errors
// actionable. Phase ordinals are one-based scratch-status indices; phase 10 is
// reserved for aggregate-only failures such as parentage rebind.
constexpr int32_t kTurnoverFailureDiagnosticStrideV1 = 1000;
constexpr int32_t kTurnoverAggregateOnlyDiagnosticPhaseV1 = 10;

constexpr int32_t encode_turnover_failure_diagnostic_v1(
    const int32_t phase,
    const int32_t substatus
) {
    return phase * kTurnoverFailureDiagnosticStrideV1 + substatus;
}

constexpr int32_t turnover_failure_diagnostic_phase_v1(const int32_t status) {
    return status / kTurnoverFailureDiagnosticStrideV1;
}

constexpr int32_t turnover_failure_diagnostic_substatus_v1(
    const int32_t status
) {
    return status % kTurnoverFailureDiagnosticStrideV1;
}

constexpr int64_t align_up(
    const int64_t value,
    const int64_t alignment
) {
    return (value + alignment - 1) / alignment * alignment;
}

constexpr int64_t next_scratch_row(
    const int64_t offset,
    const int64_t bytes
) {
    return align_up(offset + bytes, kScratchRowAlignment);
}

struct FlatTurnoverWorkLayout {
    static constexpr int64_t kCpythonWordsA = 0;
    static constexpr int64_t kCpythonWordsB = next_scratch_row(
        kCpythonWordsA,
        kCpythonWordCount * static_cast<int64_t>(sizeof(uint32_t))
    );
    static constexpr int64_t kRankedSlots = next_scratch_row(
        kCpythonWordsB,
        kCpythonWordCount * static_cast<int64_t>(sizeof(uint32_t))
    );
    static constexpr int64_t kRankedAgentIds = next_scratch_row(
        kRankedSlots,
        kMaximumCapacity * static_cast<int64_t>(sizeof(int64_t))
    );
    static constexpr int64_t kRankedFitness = next_scratch_row(
        kRankedAgentIds,
        kMaximumCapacity * static_cast<int64_t>(sizeof(int64_t))
    );
    static constexpr int64_t kCulledSlots = next_scratch_row(
        kRankedFitness,
        kMaximumCapacity * static_cast<int64_t>(sizeof(double))
    );
    static constexpr int64_t kCulledAgentIds = next_scratch_row(
        kCulledSlots,
        kMaximumCapacity * static_cast<int64_t>(sizeof(int64_t))
    );
    static constexpr int64_t kParentPoolSlots = next_scratch_row(
        kCulledAgentIds,
        kMaximumCapacity * static_cast<int64_t>(sizeof(int64_t))
    );
    static constexpr int64_t kParentPoolAgentIds = next_scratch_row(
        kParentPoolSlots,
        kMaximumCapacity * static_cast<int64_t>(sizeof(int64_t))
    );
    static constexpr int64_t kRefillParentSlots = next_scratch_row(
        kParentPoolAgentIds,
        kMaximumCapacity * static_cast<int64_t>(sizeof(int64_t))
    );
    static constexpr int64_t kRefillMateSlots = next_scratch_row(
        kRefillParentSlots,
        kMaximumCapacity * static_cast<int64_t>(sizeof(int64_t))
    );
    static constexpr int64_t kRefillSpawnPositions = next_scratch_row(
        kRefillMateSlots,
        kMaximumCapacity * static_cast<int64_t>(sizeof(int64_t))
    );
    static constexpr int64_t kRefillSpawnAttempts = next_scratch_row(
        kRefillSpawnPositions,
        2 * kMaximumCapacity * static_cast<int64_t>(sizeof(int32_t))
    );
    static constexpr int64_t kReleaseAgentIds = next_scratch_row(
        kRefillSpawnAttempts,
        kMaximumCapacity * static_cast<int64_t>(sizeof(int64_t))
    );
    static constexpr int64_t kReleaseSlots = next_scratch_row(
        kReleaseAgentIds,
        kMaximumReleaseRows * static_cast<int64_t>(sizeof(int64_t))
    );
    static constexpr int64_t kCommittedParentAgentIds = next_scratch_row(
        kReleaseSlots,
        kMaximumReleaseRows * static_cast<int64_t>(sizeof(int64_t))
    );
    static constexpr int64_t kCommittedMateAgentIds = next_scratch_row(
        kCommittedParentAgentIds,
        kMaximumCapacity * static_cast<int64_t>(sizeof(int64_t))
    );
    static constexpr int64_t kCommittedChildAgentIds = next_scratch_row(
        kCommittedMateAgentIds,
        kMaximumCapacity * static_cast<int64_t>(sizeof(int64_t))
    );
    static constexpr int64_t kCommittedChildSlots = next_scratch_row(
        kCommittedChildAgentIds,
        kMaximumCapacity * static_cast<int64_t>(sizeof(int64_t))
    );
    static constexpr int64_t kSortedAliveAgentIds = next_scratch_row(
        kCommittedChildSlots,
        kMaximumCapacity * static_cast<int64_t>(sizeof(int64_t))
    );
    static constexpr int64_t kPolicyMutationOccurred = next_scratch_row(
        kSortedAliveAgentIds,
        kMaximumCapacity * static_cast<int64_t>(sizeof(int64_t))
    );
    static constexpr int64_t kPolicyMutationDelta = next_scratch_row(
        kPolicyMutationOccurred,
        kMaximumCapacity * static_cast<int64_t>(sizeof(bool))
    );
    static constexpr int64_t kScalarCounters = next_scratch_row(
        kPolicyMutationDelta,
        kMaximumCapacity * static_cast<int64_t>(sizeof(float))
    );
    static constexpr int64_t kPhaseStatuses = next_scratch_row(
        kScalarCounters,
        kTurnoverScratchCounterCount
            * static_cast<int64_t>(sizeof(int64_t))
    );
    static constexpr int64_t kPlanHardcap = next_scratch_row(
        kPhaseStatuses,
        kTurnoverScratchStatusCount
            * static_cast<int64_t>(sizeof(int32_t))
    );
    static constexpr int64_t kPolicyMutationDeltaTotal = next_scratch_row(
        kPlanHardcap,
        static_cast<int64_t>(sizeof(bool))
    );
    static constexpr int64_t kAggregateNextGeneration = next_scratch_row(
        kPolicyMutationDeltaTotal,
        static_cast<int64_t>(sizeof(double))
    );
    static constexpr int64_t kAggregateNextTick = next_scratch_row(
        kAggregateNextGeneration,
        static_cast<int64_t>(sizeof(int64_t))
    );
    static constexpr int64_t kRequiredBytesPerCell = align_up(
        kAggregateNextTick + static_cast<int64_t>(sizeof(int64_t)),
        kScratchCellAlignment
    );
};

struct FlatTurnoverScratch {
    uint8_t* storage;
    int64_t bytes_per_cell;

    template <typename T>
    __device__ __forceinline__ T* at(
        const int64_t cell,
        const int64_t byte_offset
    ) const {
        return reinterpret_cast<T*>(
            storage + cell * bytes_per_cell + byte_offset
        );
    }
};

template <typename T>
__device__ __forceinline__ T* at(
    const FlatTurnoverScratch scratch,
    const int64_t cell,
    const int64_t byte_offset
) {
    return scratch.at<T>(cell, byte_offset);
}

enum TurnoverOutputCountIndex : int32_t {
    kOutputRankedCount = 0,
    kOutputCulledCount,
    kOutputParentPoolCount,
    kOutputRefillCount,
    kOutputReleasedCount,
    kOutputRefillPlannedCount,
    kOutputChildCommittedCount,
    kOutputFinalLiveCount,
    kTurnoverOutputCount,
};

// ID/slot arrays have kMaximumCapacity rows per cell. counts has exactly
// kTurnoverOutputCount rows; status, next_generation, and next_tick have one.
struct FlatTurnoverOutputs {
    int64_t* ranked_agent_ids;
    int64_t* culled_agent_ids;
    int64_t* parent_pool_agent_ids;
    int64_t* refill_parent_agent_ids;
    int64_t* refill_mate_agent_ids;
    int64_t* child_agent_ids;
    int64_t* child_slots;
    int64_t* counts;
    int32_t* status;
    int64_t* next_generation;
    int64_t* next_tick;
    int64_t* genome_v2_causal_record_rows;
    bool* genome_v2_causal_record_valid;
    int64_t* genome_v2_causal_record_count;
    int64_t* genome_v2_causal_record_dropped_count;
    bool* genome_v2_causal_capture_enabled;
    int64_t genome_v2_causal_record_capacity;
};

struct FlatTurnoverConfig {
    bool enabled;
    int32_t lifecycle_mode;
    int64_t population_size;
    double cull_fraction;
    double parent_fraction;
    double coherence_thrashing_weight;
    double coherence_drive_volatility_weight;
    double coherence_selfharm_weight;
    double coherence_pred_err_weight;
    const int32_t* prerequisite_status;
    const int32_t* genome_upgrade_v2_qualification_fault_stage;
    int64_t genome_upgrade_v2_qualification_fault_stride_bytes;
    cross_path_resident_control_v1::FlatCrossPathResidentControlV1
        cross_path_control;
    int32_t selection_mode;
    const uint64_t* selection_seed_by_cell;
    // HAPB keeps the full mutable policy bank on the host.  Native turnover
    // still owns identity/RNG/genome state, but its policy-row bytes are
    // committed by the guarded host transaction between plan and publish.
    bool hapb_host_authoritative_policy;
};

static_assert(kTurnoverMutableTensorCount == 122);
static_assert(kPolicyNpSemanticDraws == 121);
static_assert(kMaximumReleaseRows == 1536);
static_assert(kTurnoverOutputCount == 8);
static_assert(kTurnoverScratchStatusCount == 9);
static_assert(sizeof(bool) == 1);
static_assert(std::is_standard_layout<TurnoverMutableTensorTriplet>::value);
static_assert(std::is_trivially_copyable<TurnoverMutableTensorTriplet>::value);
static_assert(sizeof(TurnoverMutableTensorTriplet) == 4 * sizeof(int64_t));
static_assert(
    alignof(TurnoverMutableTensorTriplet) >= alignof(uint8_t*)
);
static_assert(std::is_standard_layout<FlatTurnoverMutableTensorTable>::value);
static_assert(std::is_trivially_copyable<FlatTurnoverMutableTensorTable>::value);
static_assert(std::is_standard_layout<FlatTurnoverScratch>::value);
static_assert(std::is_trivially_copyable<FlatTurnoverScratch>::value);
static_assert(std::is_standard_layout<FlatTurnoverOutputs>::value);
static_assert(std::is_trivially_copyable<FlatTurnoverOutputs>::value);
static_assert(std::is_standard_layout<FlatTurnoverConfig>::value);
static_assert(std::is_trivially_copyable<FlatTurnoverConfig>::value);

static_assert(
    FlatTurnoverWorkLayout::kCpythonWordsA % alignof(uint32_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kCpythonWordsB % alignof(uint32_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kRankedSlots % alignof(int64_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kRankedAgentIds % alignof(int64_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kRankedFitness % alignof(double) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kCulledSlots % alignof(int64_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kCulledAgentIds % alignof(int64_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kParentPoolSlots % alignof(int64_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kParentPoolAgentIds % alignof(int64_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kRefillParentSlots % alignof(int64_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kRefillMateSlots % alignof(int64_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kRefillSpawnPositions % alignof(int32_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kRefillSpawnAttempts % alignof(int64_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kReleaseAgentIds % alignof(int64_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kReleaseSlots % alignof(int64_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kCommittedParentAgentIds
        % alignof(int64_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kCommittedMateAgentIds
        % alignof(int64_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kCommittedChildAgentIds
        % alignof(int64_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kCommittedChildSlots % alignof(int64_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kSortedAliveAgentIds % alignof(int64_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kPolicyMutationOccurred % alignof(bool) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kPolicyMutationDelta % alignof(float) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kScalarCounters % alignof(int64_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kPhaseStatuses % alignof(int32_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kPolicyMutationDeltaTotal % alignof(double) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kAggregateNextGeneration
        % alignof(int64_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kAggregateNextTick % alignof(int64_t) == 0
);
static_assert(
    FlatTurnoverWorkLayout::kRequiredBytesPerCell % kScratchCellAlignment == 0
);
static_assert(FlatTurnoverWorkLayout::kRequiredBytesPerCell == 132096);
static_assert(
    FlatTurnoverWorkLayout::kRequiredBytesPerCell
        <= kMaximumScratchBytesPerCell
);

}  // namespace eiel::cuda_architecture_b::flattened_turnover_v1
