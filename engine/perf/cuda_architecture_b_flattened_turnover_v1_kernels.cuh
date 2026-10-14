#pragma once

#include "cuda_architecture_b_flattened_controller_v1.hpp"
#include "cuda_architecture_b_active_identity_pairs_v1.cuh"
#include "cuda_architecture_b_flattened_birth_scratch_v2.cuh"
#include "cuda_architecture_b_flattened_modular_controller_v1.cuh"
#include "cuda_architecture_b_persistent_turnover_drives.cuh"
#include "cuda_architecture_b_persistent_turnover_identity.cuh"
#include "cuda_architecture_b_persistent_turnover_newborn_state.cuh"
#include "cuda_architecture_b_persistent_turnover_plan.cuh"
#include "cuda_architecture_b_persistent_turnover_policy.cuh"
#include "cuda_architecture_b_persistent_turnover_scalar_genome.cuh"
#include "cuda_diploid_scalar_genome_v1.cuh"
#include "../tw_rngv2/native/cuda_diploid_scalar_genome_rngv2_v1.cuh"
#include "../tw_rngv2/native/genome_upgrade_v2_q10_rngv2_v1.cuh"
#include "../tw_rngv2/native/tw_rngv2.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>

namespace eiel::cuda_architecture_b::flattened_turnover_v1 {

namespace controller = flattened_controller_v1;
namespace drives = persistent_turnover_drives;
namespace identity = persistent_turnover_identity;
namespace active_identity = active_identity_pairs_v1;
namespace newborn = persistent_turnover_newborn_state;
namespace plan = persistent_turnover_plan;
namespace policy = persistent_turnover_policy;
namespace scalar = persistent_turnover_scalar_genome;
namespace diploid = cuda_diploid_scalar_genome_v1;
namespace modular = flattened_modular_controller_v1;
namespace cross_path = cross_path_resident_control_v1;
namespace torch_rng_pack =
    eiel::cuda_architecture_b::torch_rng_state_pack_v1;

constexpr int32_t kTurnoverThreadsPerCell = 64;

__host__ __device__ __forceinline__ bool turnover_cell_enabled_v1(
    const FlatTurnoverConfig config,
    const int64_t cell
) {
    return config.enabled
        && !resident_reproduction_exposure_v1::
            attempt_capacity_qualification_faulted_v1(
                config.cross_path_control.reproduction_exposure,
                cell
            )
        && (
            !cross_path::has_cross_path_authority_v1(
                config.cross_path_control
            )
            || (
                cell >= 0
                && config.cross_path_control.reproduction_path_by_cell[cell]
                    == cross_path::kReproductionPathGenerationBoundaryRefill
            )
        );
}

struct FlatTurnoverExecutionConfig {
    int64_t generation;
    int64_t start_tick;
    int64_t ticks;
    int64_t capacity;
    int64_t agent_id_capacity;
    int64_t pending_capacity;
    int64_t grid_height;
    int64_t grid_width;
    double start_energy;
    float start_health;
    float max_energy;
    int32_t sensor_radius_max;
    int64_t action_cache_width;
    int64_t drive_width;
    int64_t chemistry_width;
    int64_t mlp_row_width;
    int64_t hmem_width;
    int64_t gauss_tape_capacity;
    int64_t policy_tape_capacity;
    int64_t policy_mutation_tape_width;
    int64_t torch_state_bytes;
    bool env_policy_alias;
    bool immutable_shared_gru_policy;
    bool terminal_low_population_no_reinit;
    float* gru_pack;
    int64_t gru_row_width;
    int32_t rng_semantics_version;
    uint32_t world_uid;
    const int64_t* world_uid_by_cell;
    int64_t ticks_per_generation;
    uint64_t rng_namespace_key;
    uint32_t rng_semantic_epoch;
};

__device__ __forceinline__ uint32_t turnover_world_uid_for_cell_v1(
    const FlatTurnoverExecutionConfig& execution,
    const int64_t cell
) {
    return execution.world_uid_by_cell == nullptr
        ? execution.world_uid
        : static_cast<uint32_t>(execution.world_uid_by_cell[cell]);
}

struct TurnoverLaunchReceipt {
    int64_t shadow_plan_launches;
    int64_t cull_refill_plan_launches;
    int64_t commit_launches;
    int64_t publish_launches;
    int32_t cuda_status;
};

using KernelResourceReceipt = controller::KernelResourceReceipt;
using FlatOrdinaryBirthState = controller::FlatOrdinaryBirthState;
using FlatOrdinaryBirthConfig = controller::FlatOrdinaryBirthConfig;

static_assert(std::is_standard_layout_v<FlatTurnoverExecutionConfig>);
static_assert(std::is_trivially_copyable_v<FlatTurnoverExecutionConfig>);
static_assert(std::is_standard_layout_v<TurnoverLaunchReceipt>);
static_assert(std::is_trivially_copyable_v<TurnoverLaunchReceipt>);

namespace detail {

constexpr int32_t kInternalDisabled = -1;
constexpr int32_t kGenomeUpgradeV2TurnoverQualificationFailureBase = 260;
namespace causal_output =
    eiel::cuda_architecture_b::genome_v2_causal_output_v1;

__device__ __forceinline__ void write_turnover_causal_record_v1(
    const FlatTurnoverOutputs outputs,
    const int64_t row_index,
    const int64_t cell,
    const FlatTurnoverExecutionConfig execution,
    const int64_t child_id,
    const int64_t parent_id,
    const int64_t mate_parent_id,
    const int64_t child_slot,
    const int64_t parent_slot,
    const int64_t mate_parent_slot,
    const int64_t founder_id,
    const int64_t lineage_depth,
    const eiel::genome_upgrade_v2_q10_v1::ResidentPlanesV1& planes
) {
    namespace genome_v2 = eiel::genome_upgrade_v2_q10_v1;
    if (
        outputs.genome_v2_causal_record_rows == nullptr
        || outputs.genome_v2_causal_record_valid == nullptr
        || row_index < 0
        || row_index >= outputs.genome_v2_causal_record_capacity
    ) {
        return;
    }
    const int64_t physical_row =
        cell * outputs.genome_v2_causal_record_capacity + row_index;
    int64_t* const row = outputs.genome_v2_causal_record_rows
        + physical_row * causal_output::kRecordWidth;
    row[causal_output::kRecordOrdinal] = row_index;
    row[causal_output::kEventKind] =
        causal_output::kTurnoverBirthEventKind;
    row[causal_output::kCellOrdinal] = cell;
    row[causal_output::kGeneration] = execution.generation;
    row[causal_output::kTick] = execution.ticks;
    // Turnover does not publish a typed lifecycle-event sequence. The bounded
    // causal-output ordinal is its deterministic publication sequence.
    row[causal_output::kEventSequence] = row_index;
    row[causal_output::kEventSequenceValid] = 1;
    row[causal_output::kChildAgentId] = child_id;
    row[causal_output::kParentAgentId] = parent_id;
    row[causal_output::kMateParentAgentId] = mate_parent_id;
    row[causal_output::kDestinationSlot] = child_slot;
    row[causal_output::kParentSlot] = parent_slot;
    row[causal_output::kMateParentSlot] = mate_parent_slot;
    row[causal_output::kLineageFounderId] = founder_id;
    row[causal_output::kLineageDepth] = lineage_depth;

    genome_v2::GenotypeV1 genotype{};
    genome_v2::load_genotype_v1(planes, child_slot, &genotype);
    int64_t column = causal_output::kAllelesQ10;
    for (int32_t homolog = 0; homolog < genome_v2::kHomologCount; ++homolog) {
        for (int32_t product = 0; product < genome_v2::kProductCount; ++product) {
            for (
                int32_t parameter = 0;
                parameter < genome_v2::kParameterCount;
                ++parameter
            ) {
                row[column++] =
                    genotype.alleles[homolog][product][parameter];
            }
        }
    }
    column = causal_output::kModifiersQ10;
    for (int32_t homolog = 0; homolog < genome_v2::kHomologCount; ++homolog) {
        for (int32_t product = 0; product < genome_v2::kProductCount; ++product) {
            row[column++] = genotype.modifiers[homolog][product];
        }
    }
    genome_v2::ExpressionV1 expression{};
    for (int32_t product = 0; product < genome_v2::kProductCount; ++product) {
        expression.products[product] = planes.products[
            static_cast<int64_t>(product) * planes.slot_count + child_slot
        ];
        row[causal_output::kProductsQ10 + product] =
            expression.products[product];
    }
    row[causal_output::kRegulatorValuesValid] = 0;
    for (
        int32_t phenotype = 0;
        phenotype < genome_v2::kPhenotypeCount;
        ++phenotype
    ) {
        expression.phenotypes[phenotype] = planes.phenotypes[
            static_cast<int64_t>(phenotype) * planes.slot_count + child_slot
        ];
        row[causal_output::kPhenotypesQ10 + phenotype] =
            expression.phenotypes[phenotype];
    }
    // Turnover children are committed under the frozen neutral context.
    row[causal_output::kContextValid] = 1;
    for (int32_t index = 0; index < genome_v2::kContextCount; ++index) {
        row[causal_output::kContextQ10 + index] = 0;
    }
    expression.expression_cost = planes.expression_cost[child_slot];
    row[causal_output::kExpressionCostQ10] = expression.expression_cost;
    row[causal_output::kRegulatoryCostValid] = 0;
    row[causal_output::kRegulatoryCostQ10] =
        causal_output::kUnavailableValue;
    const genome_v2::RuntimeProjectionV1 runtime =
        genome_v2::project_runtime_traits_v1(expression);
    const int64_t runtime_values[12] = {
        runtime.body_size_factor_q10,
        runtime.movement_capacity_factor_q10,
        runtime.damage_received_factor_q10,
        runtime.metabolic_efficiency_factor_q10,
        runtime.recovery_factor_q10,
        runtime.sensory_fidelity_factor_q10,
        runtime.reproduction_threshold_factor_q10,
        runtime.offspring_energy_factor_q10,
        runtime.threat_response_q10,
        runtime.memory_allocation_q10,
        runtime.memory_dimension,
        runtime.expression_energy_debit_q10,
    };
    for (int32_t index = 0; index < 12; ++index) {
        row[causal_output::kRuntimeProjectionQ10 + index] =
            runtime_values[index];
    }
    outputs.genome_v2_causal_record_valid[physical_row] = true;
}

__device__ __forceinline__ int32_t qualification_fault_stage_v1(
    const FlatTurnoverConfig config,
    const int64_t cell
) {
    if (
        config.genome_upgrade_v2_qualification_fault_stage == nullptr
        || config.genome_upgrade_v2_qualification_fault_stride_bytes
            < static_cast<int64_t>(sizeof(int32_t))
    ) {
        return 0;
    }
    const uint8_t* const base = reinterpret_cast<const uint8_t*>(
        config.genome_upgrade_v2_qualification_fault_stage
    );
    return reinterpret_cast<const int32_t*>(
        base
        + cell * config.genome_upgrade_v2_qualification_fault_stride_bytes
    )[0];
}

__device__ __forceinline__ int32_t turnover_fault_status_v1(
    const int32_t stage
) {
    return kGenomeUpgradeV2TurnoverQualificationFailureBase
        + stage - flattened_birth_scratch_v2::
            kGenomeUpgradeV2FaultTurnoverMidCull + 1;
}

__device__ __forceinline__ int32_t maximum_status(
    const int32_t lhs,
    const int32_t rhs
) {
    return lhs > rhs ? lhs : rhs;
}

template <typename T>
__device__ __forceinline__ const T* offset_or_null(
    const T* const base,
    const int64_t offset
) {
    return base == nullptr ? nullptr : base + offset;
}

__device__ __forceinline__ void initialize_cell_scratch(
    const FlatTurnoverScratch scratch,
    const int64_t cell,
    const FlatTurnoverExecutionConfig execution
) {
    int64_t* const ranked_slots = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kRankedSlots
    );
    int64_t* const ranked_ids = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kRankedAgentIds
    );
    double* const ranked_fitness = at<double>(
        scratch, cell, FlatTurnoverWorkLayout::kRankedFitness
    );
    int64_t* const culled_slots = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kCulledSlots
    );
    int64_t* const culled_ids = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kCulledAgentIds
    );
    int64_t* const parent_slots = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kParentPoolSlots
    );
    int64_t* const parent_ids = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kParentPoolAgentIds
    );
    int64_t* const refill_parent_slots = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kRefillParentSlots
    );
    int64_t* const refill_mate_slots = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kRefillMateSlots
    );
    int32_t* const spawn_positions = at<int32_t>(
        scratch, cell, FlatTurnoverWorkLayout::kRefillSpawnPositions
    );
    int64_t* const spawn_attempts = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kRefillSpawnAttempts
    );
    int64_t* const committed_parent_ids = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kCommittedParentAgentIds
    );
    int64_t* const committed_mate_ids = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kCommittedMateAgentIds
    );
    int64_t* const committed_child_ids = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kCommittedChildAgentIds
    );
    int64_t* const committed_child_slots = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kCommittedChildSlots
    );
    int64_t* const sorted_alive_ids = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kSortedAliveAgentIds
    );
    bool* const policy_mutated = at<bool>(
        scratch, cell, FlatTurnoverWorkLayout::kPolicyMutationOccurred
    );
    float* const policy_delta = at<float>(
        scratch, cell, FlatTurnoverWorkLayout::kPolicyMutationDelta
    );

    for (int64_t row = 0; row < kMaximumCapacity; ++row) {
        ranked_slots[row] = -1;
        ranked_ids[row] = -1;
        ranked_fitness[row] = plan::canonical_quiet_nan();
        culled_slots[row] = -1;
        culled_ids[row] = -1;
        parent_slots[row] = -1;
        parent_ids[row] = -1;
        refill_parent_slots[row] = -1;
        refill_mate_slots[row] = -1;
        spawn_positions[row * 2] = -1;
        spawn_positions[row * 2 + 1] = -1;
        spawn_attempts[row] = 0;
        committed_parent_ids[row] = -1;
        committed_mate_ids[row] = -1;
        committed_child_ids[row] = -1;
        committed_child_slots[row] = -1;
        sorted_alive_ids[row] = -1;
        policy_mutated[row] = false;
        policy_delta[row] = 0.0f;
    }

    int64_t* const release_ids = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kReleaseAgentIds
    );
    int64_t* const release_slots = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kReleaseSlots
    );
    for (int64_t row = 0; row < kMaximumReleaseRows; ++row) {
        release_ids[row] = -1;
        release_slots[row] = -1;
    }

    int64_t* const counters = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kScalarCounters
    );
    for (int32_t index = 0; index < kTurnoverScratchCounterCount; ++index) {
        counters[index] = 0;
    }
    int32_t* const statuses = at<int32_t>(
        scratch, cell, FlatTurnoverWorkLayout::kPhaseStatuses
    );
    for (int32_t index = 0; index < kTurnoverScratchStatusCount; ++index) {
        statuses[index] = 0;
    }
    at<bool>(scratch, cell, FlatTurnoverWorkLayout::kPlanHardcap)[0] = false;
    at<double>(
        scratch, cell, FlatTurnoverWorkLayout::kPolicyMutationDeltaTotal
    )[0] = 0.0;
    at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kAggregateNextGeneration
    )[0] = execution.generation;
    at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kAggregateNextTick
    )[0] = execution.start_tick + execution.ticks;
}

__device__ __forceinline__ void reset_newborn_rows(
    const FlatTurnoverMutableTensorTable tensors,
    const FlatTurnoverScratch scratch,
    const int64_t cell,
    const FlatTurnoverExecutionConfig execution
) {
    const newborn::FlatTurnoverNewbornState state{
        {tensors.shadow<float>(kEnergyCollected, cell), 1},
        {tensors.shadow<double>(kEnergyCollectedExact, cell), 1},
        {tensors.shadow<int32_t>(kAttacks, cell), 1},
        {tensors.shadow<int32_t>(kEarlySurvivalSteps, cell), 1},
        {tensors.shadow<float>(kEarlyEnergyCollected, cell), 1},
        {tensors.shadow<double>(kEarlyEnergyCollectedExact, cell), 1},
        {tensors.shadow<float>(kEarlyInjuryGained, cell), 1},
        {tensors.shadow<float>(kEarlyLifeWeight, cell), 1},
        {tensors.shadow<float>(kPredErrSum, cell), 1},
        {tensors.shadow<int32_t>(kPredErrCount, cell), 1},
        {tensors.shadow<int32_t>(kFirstReproStep, cell), 1},
        {tensors.shadow<bool>(kDevStageInitialized, cell), 1},
        {tensors.shadow<float>(kEpisode0PredErrSum, cell), 1},
        {tensors.shadow<int32_t>(kEpisode0PredErrCount, cell), 1},
        {tensors.shadow<float>(kLatePredErrSum, cell), 1},
        {tensors.shadow<int32_t>(kLatePredErrCount, cell), 1},
        {tensors.shadow<float>(kChemistryObservationFidelity, cell), 1},
        {tensors.shadow<float>(kChemistryObservationBias, cell), 1},
        {tensors.shadow<int32_t>(kLastAction, cell), 1},
        {tensors.shadow<int32_t>(kLastLastAction, cell), 1},
        {tensors.shadow<int32_t>(kActionSwitches, cell), 1},
        {tensors.shadow<int32_t>(kActionOscillations, cell), 1},
        {tensors.shadow<int32_t>(kEpisode0Switches, cell), 1},
        {tensors.shadow<int32_t>(kEpisode0Oscillations, cell), 1},
        {
            tensors.shadow<int32_t>(kActionCacheActions, cell),
            execution.action_cache_width,
            1,
        },
        {
            tensors.shadow<float>(kPrevDriveDelta, cell),
            execution.drive_width,
            1,
        },
        {tensors.shadow<float>(kDriveVolatility, cell), 1},
        {tensors.shadow<float>(kDriveJitter, cell), 1},
        {tensors.shadow<float>(kEpisode0DriveJitterSum, cell), 1},
        {tensors.shadow<float>(kLastEnergyDelta, cell), 1},
        {tensors.shadow<float>(kLastHealthDelta, cell), 1},
        {tensors.shadow<float>(kLastInjuryDelta, cell), 1},
        {tensors.shadow<int32_t>(kSelfharmEvents, cell), 1},
        {tensors.shadow<int32_t>(kEpisode0SelfharmEvents, cell), 1},
        {tensors.shadow<int32_t>(kOffspringCount, cell), 1},
    };
    const int64_t count = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kScalarCounters
    )[kIdentityCommittedCount];
    const int64_t* const child_slots = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kCommittedChildSlots
    );
    for (int64_t birth = 0; birth < count; ++birth) {
        const int64_t child_slot = child_slots[birth];
        newborn::reset_turnover_newborn_state_serial(
            state,
            child_slot,
            {execution.action_cache_width, execution.drive_width}
        );
        tensors.shadow<int64_t>(kActionCacheToken, cell)[child_slot] = -1;
        tensors.shadow<int32_t>(kActionCacheCursor, cell)[child_slot] = -1;
        tensors.shadow<int32_t>(kActionCacheNext, cell)[child_slot] = -1;
    }
}

__device__ __forceinline__ void normalize_inactive_rows(
    const FlatTurnoverMutableTensorTable tensors,
    const int64_t cell,
    const FlatTurnoverExecutionConfig execution
) {
    const bool* const alive = tensors.shadow<bool>(kAliveMask, cell);
    for (int64_t slot = 0; slot < execution.capacity; ++slot) {
        if (alive[slot]) {
            continue;
        }

        tensors.shadow<float>(kEnergy, cell)[slot] = 0.0f;
        tensors.shadow<float>(kHealth, cell)[slot] = 0.0f;
        tensors.shadow<float>(kInjury, cell)[slot] = 0.0f;
        tensors.shadow<float>(kScar, cell)[slot] = 0.0f;
        tensors.shadow<int32_t>(kAge, cell)[slot] = 0;
        tensors.shadow<int32_t>(kReproCooldown, cell)[slot] = 0;
        tensors.shadow<float>(kReproThreshold, cell)[slot] = 30.0f;
        tensors.shadow<int32_t>(kOffspringCount, cell)[slot] = 0;
        tensors.shadow<int32_t>(kLineageFounderId, cell)[slot] = -1;
        tensors.shadow<int32_t>(kLineageDepth, cell)[slot] = 0;
        tensors.shadow<int32_t>(kLineageBirthGeneration, cell)[slot] = 0;
        tensors.shadow<int32_t>(kLineageBirthTick, cell)[slot] = 0;
        tensors.shadow<int32_t>(kPatchHabitatLineageId, cell)[slot] = 0;
        tensors.shadow<int32_t>(kSurvivalSteps, cell)[slot] = 0;
        tensors.shadow<int32_t>(kRestStreak, cell)[slot] = 0;
        tensors.shadow<int64_t>(kActionCacheToken, cell)[slot] = -1;
        tensors.shadow<int32_t>(kActionCacheCursor, cell)[slot] = -1;
        tensors.shadow<int32_t>(kActionCacheNext, cell)[slot] = -1;
        tensors.shadow<int32_t>(kLastAction, cell)[slot] = 0;
        tensors.shadow<int32_t>(kLastLastAction, cell)[slot] = 0;
        tensors.shadow<int32_t>(kActionSwitches, cell)[slot] = 0;
        tensors.shadow<int32_t>(kActionOscillations, cell)[slot] = 0;
        tensors.shadow<int32_t>(kEpisode0Switches, cell)[slot] = 0;
        tensors.shadow<int32_t>(kEpisode0Oscillations, cell)[slot] = 0;

        tensors.shadow<int32_t>(kGenomeHiddenSize, cell)[slot] = 64;
        tensors.shadow<int32_t>(kGenomeMemoryDim, cell)[slot] = 0;
        tensors.shadow<int32_t>(kGenomeSensorChannelMask, cell)[slot] = 0;
        tensors.shadow<int32_t>(kGenomeSensorRadius, cell)[slot] = 1;
        tensors.shadow<int32_t>(kGenomeIntrospectionChannelMask, cell)[slot] = 0;
        tensors.shadow<int32_t>(kGenomeSelfRegulationControlMask, cell)[slot] = 0;
        tensors.shadow<int32_t>(kGenomeSelfModificationTargetMask, cell)[slot] = 0;
        tensors.shadow<int32_t>(kGenomeDirectedInheritanceChannelMask, cell)[slot] = 0;
        tensors.shadow<int64_t>(kPolicyMemDim, cell)[slot] = 0;
        tensors.shadow<bool>(kPolicyUseGru, cell)[slot] = false;

        tensors.shadow<float>(kGenomeExplorationBias, cell)[slot] = 0.0f;
        tensors.shadow<float>(kGenomeAggressionBias, cell)[slot] = 0.0f;
        tensors.shadow<float>(kGenomeMutationRate, cell)[slot] = 0.0f;
        tensors.shadow<float>(kGenomeReproductionThreshold, cell)[slot] = 30.0f;
        tensors.shadow<float>(kGenomeActionTemperature, cell)[slot] = 1.0f;
        tensors.shadow<float>(kGenomeDriveInertia, cell)[slot] = 0.85f;
        tensors.shadow<float>(kGenomeDriveNoise, cell)[slot] = 0.02f;
        tensors.shadow<float>(kGenomeDriveLogitGain, cell)[slot] = 1.0f;
        tensors.shadow<float>(kGenomePredLr, cell)[slot] = 0.08f;
        tensors.shadow<float>(kGenomeDriveSpHunger, cell)[slot] = 0.55f;
        tensors.shadow<float>(kGenomeDriveSpPain, cell)[slot] = 0.05f;
        tensors.shadow<float>(kGenomeDriveSpFatigue, cell)[slot] = 0.15f;
        tensors.shadow<float>(kGenomeDriveSpFear, cell)[slot] = 0.10f;
        tensors.shadow<float>(kGenomeExpPriorStrength, cell)[slot] = 0.35f;
        tensors.shadow<float>(kGenomeExpPriorNoise, cell)[slot] = 0.05f;
        tensors.shadow<float>(kGenomeDriveBirthNoise, cell)[slot] = 0.03f;
        tensors.shadow<float>(kPolicyAttackBias, cell)[slot] = 0.0f;
        tensors.shadow<float>(kPolicyActionTemperature, cell)[slot] = 1.0f;
        tensors.shadow<float>(kPolicyDriveLogitGain, cell)[slot] = 1.0f;
        tensors.shadow<float>(kDriveInertia, cell)[slot] = 0.85f;
        tensors.shadow<float>(kDriveNoise, cell)[slot] = 0.02f;
        tensors.shadow<float>(kPredLr, cell)[slot] = 0.08f;

        for (int64_t drive = 0; drive < execution.drive_width; ++drive) {
            tensors.shadow<float>(kDrives, cell)[
                slot * execution.drive_width + drive
            ] = 0.0f;
            tensors.shadow<float>(kPrevDriveDelta, cell)[
                slot * execution.drive_width + drive
            ] = 0.0f;
        }
        tensors.shadow<float>(kBirthResetShock, cell)[slot] = 0.0f;
        for (
            int64_t chemistry = 0;
            chemistry < execution.chemistry_width;
            ++chemistry
        ) {
            tensors.shadow<float>(kChemistry, cell)[
                slot * execution.chemistry_width + chemistry
            ] = 0.0f;
        }
        for (int64_t memory = 0; memory < execution.hmem_width; ++memory) {
            tensors.shadow<float>(kHmem, cell)[
                slot * execution.hmem_width + memory
            ] = 0.0f;
        }

        tensors.shadow<float>(kEnergyCollected, cell)[slot] = 0.0f;
        tensors.shadow<double>(kEnergyCollectedExact, cell)[slot] = 0.0;
        tensors.shadow<int32_t>(kAttacks, cell)[slot] = 0;
        tensors.shadow<int32_t>(kEarlySurvivalSteps, cell)[slot] = 0;
        tensors.shadow<float>(kEarlyEnergyCollected, cell)[slot] = 0.0f;
        tensors.shadow<double>(kEarlyEnergyCollectedExact, cell)[slot] = 0.0;
        tensors.shadow<float>(kEarlyInjuryGained, cell)[slot] = 0.0f;
        tensors.shadow<float>(kEarlyLifeWeight, cell)[slot] = 1.0f;
        tensors.shadow<float>(kPredErrSum, cell)[slot] = 0.0f;
        tensors.shadow<int32_t>(kPredErrCount, cell)[slot] = 0;
        tensors.shadow<int32_t>(kFirstReproStep, cell)[slot] = -1;
        tensors.shadow<bool>(kDevStageInitialized, cell)[slot] = false;
        tensors.shadow<float>(kEpisode0PredErrSum, cell)[slot] = 0.0f;
        tensors.shadow<int32_t>(kEpisode0PredErrCount, cell)[slot] = 0;
        tensors.shadow<float>(kLatePredErrSum, cell)[slot] = 0.0f;
        tensors.shadow<int32_t>(kLatePredErrCount, cell)[slot] = 0;
        tensors.shadow<float>(kChemistryObservationFidelity, cell)[slot] = 0.0f;
        tensors.shadow<float>(kChemistryObservationBias, cell)[slot] = 0.0f;

        for (
            int64_t action = 0;
            action < execution.action_cache_width;
            ++action
        ) {
            tensors.shadow<int32_t>(kActionCacheActions, cell)[
                slot * execution.action_cache_width + action
            ] = -1;
        }
        tensors.shadow<float>(kDriveVolatility, cell)[slot] = 0.0f;
        tensors.shadow<float>(kDriveJitter, cell)[slot] = 0.0f;
        tensors.shadow<float>(kEpisode0DriveJitterSum, cell)[slot] = 0.0f;
        tensors.shadow<float>(kLastEnergyDelta, cell)[slot] = 0.0f;
        tensors.shadow<float>(kLastHealthDelta, cell)[slot] = 0.0f;
        tensors.shadow<float>(kLastInjuryDelta, cell)[slot] = 0.0f;
        tensors.shadow<int32_t>(kSelfharmEvents, cell)[slot] = 0;
        tensors.shadow<int32_t>(kEpisode0SelfharmEvents, cell)[slot] = 0;
    }
}

__device__ __forceinline__ bool canonicalize_shadow_agent_grid_v1(
    const FlatTurnoverMutableTensorTable tensors,
    const int64_t cell,
    const FlatTurnoverExecutionConfig execution
) {
    int32_t* const agent_grid = tensors.shadow<int32_t>(kAgentGrid, cell);
    const int64_t* const slot_to_agent_id =
        tensors.shadow<int64_t>(kSlotToAgentId, cell);
    const bool* const alive_mask = tensors.shadow<bool>(kAliveMask, cell);
    const int32_t* const pos_y = tensors.shadow<int32_t>(kPosY, cell);
    const int32_t* const pos_x = tensors.shadow<int32_t>(kPosX, cell);
    const int64_t grid_cells = execution.grid_height * execution.grid_width;
    for (int64_t index = 0; index < grid_cells; ++index) {
        agent_grid[index] = -1;
    }
    for (int64_t slot = 0; slot < execution.capacity; ++slot) {
        if (!alive_mask[slot]) {
            continue;
        }
        const int64_t agent_id = slot_to_agent_id[slot];
        const int32_t y = pos_y[slot];
        const int32_t x = pos_x[slot];
        if (
            !active_identity::agent_id_fits_grid_v1(agent_id)
            || y < 0 || y >= execution.grid_height
            || x < 0 || x >= execution.grid_width
        ) {
            return false;
        }
        const int64_t grid_index =
            static_cast<int64_t>(y) * execution.grid_width + x;
        if (agent_grid[grid_index] != -1) {
            return false;
        }
        agent_grid[grid_index] = static_cast<int32_t>(agent_id);
    }
    return true;
}

inline bool valid_launch_boundary(
    const int64_t cells,
    const FlatTurnoverMutableTensorTable tensors,
    const FlatTurnoverScratch scratch,
    const FlatTurnoverOutputs outputs,
    const FlatTurnoverConfig config,
    const FlatTurnoverExecutionConfig execution,
    const FlatOrdinaryBirthState birth_state,
    const FlatOrdinaryBirthConfig birth_config
) {
    if (
        cells <= 0 || cells > 0xffffffffLL || tensors.tensors == nullptr
        || scratch.storage == nullptr
        || scratch.bytes_per_cell < FlatTurnoverWorkLayout::kRequiredBytesPerCell
        || execution.capacity <= 0 || execution.capacity > kMaximumCapacity
        || execution.agent_id_capacity <= 0
        || execution.pending_capacity < 0
        || execution.pending_capacity > kMaximumPending
        || execution.grid_height <= 0 || execution.grid_width <= 0
        || execution.action_cache_width < 0 || execution.drive_width < 4
        || execution.chemistry_width < 6 || execution.mlp_row_width < 0
        || execution.hmem_width < 0 || execution.gauss_tape_capacity < 0
        || execution.policy_tape_capacity < 0
        || (
            execution.policy_mutation_tape_width != 1
            && execution.policy_mutation_tape_width
                != execution.mlp_row_width
        )
        || execution.torch_state_bytes < 0 || execution.ticks < 0
        || execution.generation < 0 || execution.generation >= INT32_MAX
        || execution.rng_semantics_version < 1
        || execution.rng_semantics_version > 2
        || execution.start_tick < 0
        || execution.start_tick > INT64_MAX - execution.ticks
        || outputs.ranked_agent_ids == nullptr
        || outputs.culled_agent_ids == nullptr
        || outputs.parent_pool_agent_ids == nullptr
        || outputs.refill_parent_agent_ids == nullptr
        || outputs.refill_mate_agent_ids == nullptr
        || outputs.child_agent_ids == nullptr || outputs.child_slots == nullptr
        || outputs.counts == nullptr || outputs.status == nullptr
        || outputs.next_generation == nullptr || outputs.next_tick == nullptr
        || (config.enabled && config.prerequisite_status == nullptr)
    ) {
        return false;
    }
    if (
        execution.rng_semantics_version == 2
        && (
            execution.ticks_per_generation <= 0
            || execution.gauss_tape_capacity != 0
            || execution.policy_tape_capacity != 0
            || execution.mlp_row_width > 65536
            || execution.rng_semantic_epoch != tw_rngv2::SEMANTIC_EPOCH_V1
            || execution.generation
                > (
                    INT64_MAX - execution.start_tick - execution.ticks
                ) / execution.ticks_per_generation
        )
    ) {
        return false;
    }
    if (
        execution.sensor_radius_max != birth_config.sensor_radius_max
        || execution.hmem_width != birth_config.hmem_width
        || execution.gauss_tape_capacity != birth_config.gauss_tape_capacity
        || execution.policy_tape_capacity != birth_config.policy_tape_capacity
        || execution.policy_mutation_tape_width
            != birth_config.policy_mutation_tape_width
        || execution.torch_state_bytes != birth_config.torch_state_bytes
        || execution.env_policy_alias != birth_config.env_policy_alias
    ) {
        return false;
    }
    if (
        execution.gauss_tape_capacity > 0
        && (
            birth_state.python_random_gauss_primary_tape == nullptr
            || birth_state.python_random_gauss_secondary_tape == nullptr
        )
    ) {
        return false;
    }
    return execution.policy_tape_capacity == 0
        || (
            birth_state.policy_birth_delta_tape != nullptr
            && birth_state.policy_birth_mutation_mask_tape != nullptr
            && birth_state.policy_birth_crossover_mask_packed_tape != nullptr
            && (
                birth_state.policy_birth_post_rng_state_packed_tape != nullptr
                || execution.policy_mutation_tape_width == 1
            )
            && birth_state.policy_birth_semantic_draws_tape != nullptr
        );
}

template <typename Kernel>
inline KernelResourceReceipt query_kernel_resources(const Kernel kernel) {
    cudaFuncAttributes attributes{};
    cudaError_t error = cudaFuncGetAttributes(&attributes, kernel);
    if (error != cudaSuccess) {
        return {-1, -1, -1, -1, -1, -1, -1, -1};
    }
    int32_t occupancy = -1;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, kTurnoverThreadsPerCell, 0
    );
    if (error != cudaSuccess) {
        occupancy = -1;
    }
    return {
        attributes.binaryVersion,
        occupancy,
        attributes.maxThreadsPerBlock,
        attributes.numRegs,
        attributes.ptxVersion,
        static_cast<int64_t>(attributes.constSizeBytes),
        static_cast<int64_t>(attributes.localSizeBytes),
        static_cast<int64_t>(attributes.sharedSizeBytes),
    };
}

inline void retain_first_error(int32_t* status) {
    const cudaError_t error = cudaGetLastError();
    if (*status == static_cast<int32_t>(cudaSuccess) && error != cudaSuccess) {
        *status = static_cast<int32_t>(error);
    }
}

}  // namespace detail

static __global__ void flattened_turnover_shadow_plan_kernel_v1(
    const int64_t cells,
    const FlatTurnoverMutableTensorTable tensors,
    const FlatTurnoverScratch scratch,
    const FlatTurnoverConfig config,
    const FlatTurnoverExecutionConfig execution,
    const modular::FlatModularContextV1 modular_context
) {
    const int64_t cell = static_cast<int64_t>(blockIdx.x);
    if (
        cell >= cells
        || (
            config.prerequisite_status != nullptr
            && config.prerequisite_status[cell] != 0
        )
    ) {
        return;
    }
    const int64_t lane = static_cast<int64_t>(threadIdx.x);
    if (!turnover_cell_enabled_v1(config, cell)) {
        return;
    }
    const int64_t lanes = static_cast<int64_t>(blockDim.x);
    for (int32_t index = 0; index < kTurnoverMutableTensorCount; ++index) {
        if (index == kMlpPack) {
            continue;
        }
        tensors.copy_source_to_shadow(
            static_cast<TurnoverMutableTensorIndex>(index), cell, lane, lanes
        );
    }
    if (modular_context.enabled && modular_context.lifecycle_enabled) {
        modular::device::copy_lifecycle_cell_source_to_shadow_v2(
            modular_context, cell, lane, lanes
        );
    }
    __syncthreads();
    if (lane != 0) {
        return;
    }

    detail::initialize_cell_scratch(scratch, cell, execution);
    int32_t* const statuses = at<int32_t>(
        scratch, cell, FlatTurnoverWorkLayout::kPhaseStatuses
    );
    int64_t* const counters = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kScalarCounters
    );
    const int32_t qualification_fault_stage =
        detail::qualification_fault_stage_v1(config, cell);
    if (
        qualification_fault_stage < 0
        || (
            qualification_fault_stage
                > flattened_birth_scratch_v2::
                    kGenomeUpgradeV2FaultImmediatelyAfterCommitBeforeReporting
            && (
                qualification_fault_stage
                    < flattened_birth_scratch_v2::
                        kGenomeUpgradeV2FaultTurnoverMidCull
                || qualification_fault_stage
                    > flattened_birth_scratch_v2::
                        kGenomeUpgradeV2FaultTurnoverAfterCommitBeforeCheckpoint
            )
        )
    ) {
        statuses[kPlanStatus] =
            detail::kGenomeUpgradeV2TurnoverQualificationFailureBase;
        statuses[kAggregateStatus] =
            detail::kGenomeUpgradeV2TurnoverQualificationFailureBase;
        return;
    }
    if (
        !detail::canonicalize_shadow_agent_grid_v1(tensors, cell, execution)
    ) {
        statuses[kPlanStatus] = plan::kTurnoverStatusInvalidResidentState;
        statuses[kAggregateStatus] = plan::kTurnoverStatusInvalidResidentState;
        return;
    }
    plan::plan_generation_turnover_serial(
        {
            tensors.shadow<int64_t>(kSlotToAgentId, cell),
            tensors.shadow<bool>(kAliveMask, cell),
            tensors.shadow<int32_t>(kSurvivalSteps, cell),
            tensors.shadow<double>(kEnergyCollectedExact, cell),
            tensors.shadow<int32_t>(kOffspringCount, cell),
            tensors.shadow<int32_t>(kAttacks, cell),
            tensors.shadow<int32_t>(kEarlySurvivalSteps, cell),
            tensors.shadow<double>(kEarlyEnergyCollectedExact, cell),
            tensors.shadow<float>(kEarlyInjuryGained, cell),
            tensors.shadow<float>(kEarlyLifeWeight, cell),
            tensors.shadow<int32_t>(kActionSwitches, cell),
            tensors.shadow<int32_t>(kActionOscillations, cell),
            tensors.shadow<float>(kDriveVolatility, cell),
            tensors.shadow<int32_t>(kSelfharmEvents, cell),
            tensors.shadow<float>(kPredErrSum, cell),
            tensors.shadow<int32_t>(kPredErrCount, cell),
        },
        {
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kRankedSlots),
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kRankedAgentIds),
            at<double>(scratch, cell, FlatTurnoverWorkLayout::kRankedFitness),
            counters + kPlanRankedCount,
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kCulledSlots),
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kCulledAgentIds),
            counters + kPlanCulledCount,
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kParentPoolSlots),
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kParentPoolAgentIds),
            counters + kPlanParentPoolCount,
            counters + kPlanSurvivorCount,
            counters + kPlanRefillCount,
            at<bool>(scratch, cell, FlatTurnoverWorkLayout::kPlanHardcap),
            statuses + kPlanStatus,
        },
        {
            execution.capacity,
            config.lifecycle_mode,
            config.population_size,
            config.cull_fraction,
            config.parent_fraction,
            config.coherence_thrashing_weight,
            config.coherence_drive_volatility_weight,
            config.coherence_selfharm_weight,
            config.coherence_pred_err_weight,
            execution.terminal_low_population_no_reinit,
        },
        {
            1,
            config.selection_mode == plan::kTurnoverSelectionFitnessV1
                ? plan::kTurnoverSelectionFitnessV1
                : (
                    config.selection_mode
                            == plan::kTurnoverSelectionUniformNeutralV1
                        && config.selection_seed_by_cell != nullptr
                            ? plan::kTurnoverSelectionUniformNeutralV1
                            : -1
                ),
            config.selection_seed_by_cell != nullptr
                ? config.selection_seed_by_cell[cell]
                : 0ULL,
            execution.generation,
        }
    );
}

static __global__ void flattened_turnover_cull_refill_plan_kernel_v1(
    const int64_t cells,
    const FlatTurnoverMutableTensorTable tensors,
    const FlatTurnoverScratch scratch,
    const FlatTurnoverConfig config,
    const FlatTurnoverExecutionConfig execution,
    const modular::FlatModularContextV1 modular_context
) {
    const int64_t cell = static_cast<int64_t>(blockIdx.x);
    if (
        cell >= cells || threadIdx.x != 0
        || !turnover_cell_enabled_v1(config, cell)
        || (
            config.prerequisite_status != nullptr
            && config.prerequisite_status[cell] != 0
        )
    ) {
        return;
    }
    int64_t* const counters = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kScalarCounters
    );
    int32_t* const statuses = at<int32_t>(
        scratch, cell, FlatTurnoverWorkLayout::kPhaseStatuses
    );

    identity::commit_turnover_cull_release_serial(
        {
            tensors.shadow<int64_t>(kSlotToAgentId, cell),
            tensors.shadow<int32_t>(kAgentIdToSlot, cell),
            tensors.shadow<bool>(kAliveMask, cell),
            tensors.shadow<int32_t>(kPosY, cell),
            tensors.shadow<int32_t>(kPosX, cell),
            tensors.shadow<int32_t>(kAgentGrid, cell),
            tensors.shadow<float>(kEnergy, cell),
            tensors.shadow<float>(kHealth, cell),
            tensors.shadow<double>(kPolicyEnergyObservation, cell),
            tensors.shadow<int64_t>(kActionCacheToken, cell),
            tensors.shadow<int32_t>(kActionCacheCursor, cell),
            tensors.shadow<int32_t>(kActionCacheNext, cell),
            tensors.shadow<int32_t>(kActionCacheActions, cell),
            tensors.shadow<int64_t>(kSlotFreeQueue, cell),
            tensors.shadow<int64_t>(kSlotFreeQueueCount, cell),
            tensors.shadow<int64_t>(kPendingFreeAgentIds, cell),
            tensors.shadow<int64_t>(kPendingFreeSlots, cell),
            tensors.shadow<int64_t>(kPendingFreeCount, cell),
            tensors.shadow<int64_t>(kActiveAgentIds, cell),
            tensors.shadow<int32_t>(kActiveAgentSlots, cell),
            tensors.shadow<int64_t>(kActiveCountState, cell),
        },
        {
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kCulledAgentIds),
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kCulledSlots),
            counters + kPlanCulledCount,
            statuses + kPlanStatus,
        },
        {
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kReleaseAgentIds),
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kReleaseSlots),
            counters + kCullReleasedCount,
            counters + kCullPendingReleasedCount,
            counters + kCullCulledReleasedCount,
            statuses + kCullStatus,
        },
        {
            execution.capacity,
            execution.agent_id_capacity,
            execution.grid_height,
            execution.grid_width,
            execution.action_cache_width,
            execution.capacity,
            execution.pending_capacity,
            execution.capacity,
            detail::qualification_fault_stage_v1(config, cell)
                    == flattened_birth_scratch_v2::
                        kGenomeUpgradeV2FaultTurnoverMidCull
                ? detail::maximum_status(
                    1,
                    counters[kPlanCulledCount] / 2
                )
                : 0,
            detail::turnover_fault_status_v1(
                flattened_birth_scratch_v2::
                    kGenomeUpgradeV2FaultTurnoverMidCull
            ),
            execution.capacity,
        }
    );
    if (
        detail::qualification_fault_stage_v1(config, cell)
            == flattened_birth_scratch_v2::
                kGenomeUpgradeV2FaultTurnoverMidCull
        && statuses[kCullStatus] == identity::kCullStatusOk
    ) {
        statuses[kCullStatus] = detail::turnover_fault_status_v1(
            flattened_birth_scratch_v2::
                kGenomeUpgradeV2FaultTurnoverMidCull
        );
    }
    if (statuses[kCullStatus] != identity::kCullStatusOk) {
        return;
    }
    if (
        modular_context.enabled && modular_context.lifecycle_enabled
        && statuses[kCullStatus] == identity::kCullStatusOk
    ) {
        const int64_t released_count = counters[kCullReleasedCount];
        const int64_t* const released_slots = at<int64_t>(
            scratch, cell, FlatTurnoverWorkLayout::kReleaseSlots
        );
        for (int64_t row = 0; row < released_count; ++row) {
            const int64_t released_slot = released_slots[row];
            if (released_slot < 0 || released_slot >= execution.capacity) {
                statuses[kAggregateStatus] = 8;
                return;
            }
            modular::device::fill_lifecycle_row_v2(
                modular_context, true, cell, released_slot
            );
        }
    }

    uint32_t* const refill_words = at<uint32_t>(
        scratch, cell, FlatTurnoverWorkLayout::kCpythonWordsA
    );
    const uint32_t* const scalar_words = tensors.shadow<uint32_t>(
        kPythonRandomWords, cell
    );
    for (int64_t word = 0; word < kCpythonWordCount; ++word) {
        refill_words[word] = scalar_words[word];
    }
    counters[kScalarCommittedCount] =
        tensors.shadow<int64_t>(kPythonRandomIndex, cell)[0];
    counters[kDriveCommittedCount] =
        tensors.shadow<int64_t>(kPythonRandomRawDraws, cell)[0];
    counters[kChemistryCommittedCount] =
        tensors.shadow<int64_t>(kMutationPySemanticDraws, cell)[0];

    plan::plan_generation_turnover_refill_serial(
        {
            tensors.shadow<int64_t>(kSlotToAgentId, cell),
            tensors.shadow<bool>(kAliveMask, cell),
            tensors.shadow<int32_t>(kAgentGrid, cell),
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kParentPoolSlots),
            counters + kPlanParentPoolCount,
            counters + kPlanRefillCount,
            statuses + kCullStatus,
            refill_words,
            at<uint32_t>(scratch, cell, FlatTurnoverWorkLayout::kCpythonWordsB),
            counters + kScalarCommittedCount,
            counters + kDriveCommittedCount,
            counters + kChemistryCommittedCount,
            tensors.shadow<uint64_t>(kSharedEnvPolicyPcg64StateWords, cell),
            tensors.shadow<int64_t>(kSharedEnvPolicyPcg64BufferWords, cell),
            tensors.shadow<int64_t>(kPolicyNpSemanticDraws, cell),
        },
        {
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kRefillParentSlots),
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kRefillMateSlots),
            at<int32_t>(scratch, cell, FlatTurnoverWorkLayout::kRefillSpawnPositions),
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kRefillSpawnAttempts),
            counters + kRefillPlannedCount,
            counters + kRefillPythonRawDrawsAdded,
            counters + kRefillPythonSemanticDrawsAdded,
            counters + kRefillSpawnRawDrawsAdded,
            counters + kRefillSpawnSemanticDrawsAdded,
            statuses + kRefillStatus,
        },
        {
            execution.capacity,
            execution.grid_height,
            execution.grid_width,
            execution.env_policy_alias,
        }
    );
    const int64_t exposure_attempt_count = counters[kRefillPlannedCount];
    if (
        statuses[kRefillStatus] == plan::kRefillStatusOk
        && exposure_attempt_count > 0
        && !resident_reproduction_exposure_v1::
            append_shared_set_successes_v1(
                config.cross_path_control.reproduction_exposure,
                cell,
                config.cross_path_control.reproduction_path_by_cell == nullptr
                    ? static_cast<int8_t>(0)
                    : config.cross_path_control.reproduction_path_by_cell[cell],
                execution.generation,
                execution.ticks,
                at<int64_t>(
                    scratch,
                    cell,
                    FlatTurnoverWorkLayout::kParentPoolAgentIds
                ),
                counters[kPlanParentPoolCount],
                at<int64_t>(
                    scratch,
                    cell,
                    FlatTurnoverWorkLayout::kRefillParentSlots
                ),
                at<int64_t>(
                    scratch,
                    cell,
                    FlatTurnoverWorkLayout::kRefillMateSlots
                ),
                tensors.shadow<int64_t>(kSlotToAgentId, cell),
                exposure_attempt_count
            )
    ) {
        statuses[kRefillStatus] = plan::kRefillStatusInvalidState;
    }
    if (
        detail::qualification_fault_stage_v1(config, cell)
        == flattened_birth_scratch_v2::
            kGenomeUpgradeV2FaultTurnoverAfterDestinationAllocation
    ) {
        statuses[kRefillStatus] = detail::turnover_fault_status_v1(
            flattened_birth_scratch_v2::
                kGenomeUpgradeV2FaultTurnoverAfterDestinationAllocation
        );
    }
}

static __global__ void flattened_turnover_commit_kernel_v1(
    const int64_t cells,
    const FlatTurnoverMutableTensorTable tensors,
    const FlatTurnoverScratch scratch,
    const FlatTurnoverConfig config,
    const FlatTurnoverExecutionConfig execution,
    const FlatOrdinaryBirthState birth_state,
    const modular::FlatModularContextV1 modular_context
) {
    const int64_t cell = static_cast<int64_t>(blockIdx.x);
    if (
        cell >= cells || threadIdx.x != 0
        || !turnover_cell_enabled_v1(config, cell)
        || (
            config.prerequisite_status != nullptr
            && config.prerequisite_status[cell] != 0
        )
    ) {
        return;
    }
    int64_t* const counters = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kScalarCounters
    );
    int32_t* const statuses = at<int32_t>(
        scratch, cell, FlatTurnoverWorkLayout::kPhaseStatuses
    );
    int32_t pretransaction_status = detail::maximum_status(
        statuses[kPlanStatus], statuses[kCullStatus]
    );
    pretransaction_status = detail::maximum_status(
        pretransaction_status, statuses[kRefillStatus]
    );
    if (pretransaction_status != 0) {
        statuses[kAggregateStatus] = pretransaction_status;
        counters[kAggregateFinalLiveCount] = counters[kPlanSurvivorCount];
        return;
    }
    const auto& cross_control = config.cross_path_control;
    const bool cross_authority =
        cross_path::has_cross_path_authority_v1(cross_control);
    const bool diploid_authority =
        birth_state.inheritance_mode != nullptr
        && birth_state.inheritance_mode[cell] == 2;
    const bool diploid_mode =
        diploid_authority
        && birth_state.inheritance_mutation_enabled != nullptr
        && birth_state.resource_handling_enabled != nullptr
        && birth_state.mutation_sigma_multiplier != nullptr
        && birth_state.diploid_allele_storage != nullptr
        && birth_state.diploid_allele_shadow_storage != nullptr
        && birth_state.genome_resource_handling_axis != nullptr
        && birth_state.genome_resource_handling_axis_shadow != nullptr;
    const bool genome_upgrade_v2_authority =
        birth_state.genome_upgrade_v2_enabled != nullptr
        && birth_state.genome_upgrade_v2_enabled[cell];
    const bool genome_upgrade_v2_mode =
        genome_upgrade_v2_authority
        && diploid_mode
        && birth_state.genome_upgrade_v2_mutation_rate_q10 != nullptr
        && birth_state.genome_upgrade_v2_mutation_step_q10 != nullptr
        && birth_state.genome_upgrade_v2_alleles_q10 != nullptr
        && birth_state.genome_upgrade_v2_modifiers_q10 != nullptr
        && birth_state.genome_upgrade_v2_products_q10 != nullptr
        && birth_state.genome_upgrade_v2_phenotypes_q10 != nullptr
        && birth_state.genome_upgrade_v2_expression_cost_q10 != nullptr
        && birth_state.genome_upgrade_v2_active_product_count != nullptr
        && birth_state.genome_upgrade_v2_saturated_product_count != nullptr
        && birth_state.genome_upgrade_v2_alleles_q10_shadow != nullptr
        && birth_state.genome_upgrade_v2_modifiers_q10_shadow != nullptr
        && birth_state.genome_upgrade_v2_products_q10_shadow != nullptr
        && birth_state.genome_upgrade_v2_phenotypes_q10_shadow != nullptr
        && birth_state.genome_upgrade_v2_expression_cost_q10_shadow != nullptr
        && birth_state.genome_upgrade_v2_active_product_count_shadow != nullptr
        && birth_state.genome_upgrade_v2_saturated_product_count_shadow
            != nullptr;
    const int32_t qualification_fault_stage =
        detail::qualification_fault_stage_v1(config, cell);
    if (
        (cross_authority && diploid_authority)
        || (diploid_authority && !diploid_mode)
        || (genome_upgrade_v2_authority && !genome_upgrade_v2_mode)
    ) {
        // Two mutation authorities, or an incomplete selected authority, are
        // never admissible for one cell.
        statuses[kScalarStatus] = scalar::kStatusMissingInheritanceSidecar;
        statuses[kAggregateStatus] = 2;
        counters[kAggregateFinalLiveCount] = counters[kPlanSurvivorCount];
        return;
    }
    if (
        qualification_fault_stage
            >= flattened_birth_scratch_v2::
                kGenomeUpgradeV2FaultTurnoverMidCull
        && qualification_fault_stage
            <= flattened_birth_scratch_v2::
                kGenomeUpgradeV2FaultTurnoverAfterCommitBeforeCheckpoint
        && !genome_upgrade_v2_mode
    ) {
        statuses[kScalarStatus] = scalar::kStatusMissingInheritanceSidecar;
        statuses[kAggregateStatus] = 2;
        counters[kAggregateFinalLiveCount] = counters[kPlanSurvivorCount];
        return;
    }
    if (diploid_mode) {
        const int64_t allele_cell_stride =
            diploid::kAllelePlaneCount * execution.capacity;
        const int64_t allele_cell_offset = cell * allele_cell_stride;
        for (int64_t index = 0; index < allele_cell_stride; ++index) {
            birth_state.diploid_allele_shadow_storage[
                allele_cell_offset + index
            ] = birth_state.diploid_allele_storage[
                allele_cell_offset + index
            ];
        }
        const int64_t cell_capacity_offset = cell * execution.capacity;
        for (int64_t slot = 0; slot < execution.capacity; ++slot) {
            birth_state.genome_resource_handling_axis_shadow[
                cell_capacity_offset + slot
            ] = birth_state.genome_resource_handling_axis[
                cell_capacity_offset + slot
            ];
        }
    }
    if (genome_upgrade_v2_mode) {
        namespace genome_v2 = eiel::genome_upgrade_v2_q10_v1;
        const int64_t cell_capacity_offset = cell * execution.capacity;
        const int64_t allele_cell_stride =
            2 * genome_v2::kProductCount * genome_v2::kParameterCount
            * execution.capacity;
        const int64_t modifier_cell_stride =
            2 * genome_v2::kProductCount * execution.capacity;
        const int64_t product_cell_stride =
            genome_v2::kProductCount * execution.capacity;
        const int64_t phenotype_cell_stride =
            genome_v2::kPhenotypeCount * execution.capacity;
        const int64_t allele_offset = cell * allele_cell_stride;
        const int64_t modifier_offset = cell * modifier_cell_stride;
        const int64_t product_offset = cell * product_cell_stride;
        const int64_t phenotype_offset = cell * phenotype_cell_stride;
        for (int64_t index = 0; index < allele_cell_stride; ++index) {
            birth_state.genome_upgrade_v2_alleles_q10_shadow[
                allele_offset + index
            ] = birth_state.genome_upgrade_v2_alleles_q10[
                allele_offset + index
            ];
        }
        for (int64_t index = 0; index < modifier_cell_stride; ++index) {
            birth_state.genome_upgrade_v2_modifiers_q10_shadow[
                modifier_offset + index
            ] = birth_state.genome_upgrade_v2_modifiers_q10[
                modifier_offset + index
            ];
        }
        for (int64_t index = 0; index < product_cell_stride; ++index) {
            birth_state.genome_upgrade_v2_products_q10_shadow[
                product_offset + index
            ] = birth_state.genome_upgrade_v2_products_q10[
                product_offset + index
            ];
        }
        for (int64_t index = 0; index < phenotype_cell_stride; ++index) {
            birth_state.genome_upgrade_v2_phenotypes_q10_shadow[
                phenotype_offset + index
            ] = birth_state.genome_upgrade_v2_phenotypes_q10[
                phenotype_offset + index
            ];
        }
        for (int64_t slot = 0; slot < execution.capacity; ++slot) {
            const int64_t index = cell_capacity_offset + slot;
            birth_state.genome_upgrade_v2_expression_cost_q10_shadow[index] =
                birth_state.genome_upgrade_v2_expression_cost_q10[index];
            birth_state.genome_upgrade_v2_active_product_count_shadow[index] =
                birth_state.genome_upgrade_v2_active_product_count[index];
            birth_state.genome_upgrade_v2_saturated_product_count_shadow[index] =
                birth_state.genome_upgrade_v2_saturated_product_count[index];
        }
    }

    // Stage2 marks every agent that completed the generation as developmentally
    // initialized before turnover. Newborn and inactive rows are reset below.
    bool* const dev_stage_initialized = tensors.shadow<bool>(
        kDevStageInitialized, cell
    );
    const bool* const alive_before_refill = tensors.shadow<bool>(kAliveMask, cell);
    for (int64_t slot = 0; slot < execution.capacity; ++slot) {
        if (alive_before_refill[slot]) {
            dev_stage_initialized[slot] = true;
        }
    }

    identity::commit_turnover_refill_identity_serial(
        {
            tensors.shadow<int64_t>(kSlotToAgentId, cell),
            tensors.shadow<int32_t>(kAgentIdToSlot, cell),
            tensors.shadow<int64_t>(kSlotTokens, cell),
            tensors.shadow<bool>(kAliveMask, cell),
            tensors.shadow<int32_t>(kPosY, cell),
            tensors.shadow<int32_t>(kPosX, cell),
            tensors.shadow<int32_t>(kAgentGrid, cell),
            tensors.shadow<float>(kEnergy, cell),
            tensors.shadow<float>(kHealth, cell),
            tensors.shadow<float>(kInjury, cell),
            tensors.shadow<float>(kScar, cell),
            tensors.shadow<int32_t>(kAge, cell),
            tensors.shadow<int32_t>(kReproCooldown, cell),
            tensors.shadow<float>(kReproThreshold, cell),
            tensors.shadow<int32_t>(kOffspringCount, cell),
            tensors.shadow<int32_t>(kLineageFounderId, cell),
            tensors.shadow<int32_t>(kLineageDepth, cell),
            tensors.shadow<int32_t>(kLineageBirthGeneration, cell),
            tensors.shadow<int32_t>(kLineageBirthTick, cell),
            tensors.shadow<int32_t>(kPatchHabitatLineageId, cell),
            tensors.shadow<double>(kPolicyEnergyObservation, cell),
            tensors.shadow<int32_t>(kSurvivalSteps, cell),
            tensors.shadow<int32_t>(kRestStreak, cell),
            tensors.shadow<int64_t>(kActionCacheToken, cell),
            tensors.shadow<int32_t>(kActionCacheCursor, cell),
            tensors.shadow<int32_t>(kActionCacheNext, cell),
            tensors.shadow<int64_t>(kSlotFreeQueue, cell),
            tensors.shadow<int64_t>(kSlotFreeQueueCount, cell),
            tensors.shadow<int64_t>(kPopulationNextId, cell),
        },
        {
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kRefillParentSlots),
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kRefillMateSlots),
            at<int32_t>(scratch, cell, FlatTurnoverWorkLayout::kRefillSpawnPositions),
            counters + kRefillPlannedCount,
        },
        {
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kCommittedParentAgentIds),
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kCommittedMateAgentIds),
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kCommittedChildAgentIds),
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kCommittedChildSlots),
            counters + kIdentityCommittedCount,
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kSortedAliveAgentIds),
            counters + kAggregateFinalLiveCount,
            statuses + kIdentityStatus,
        },
        {
            execution.capacity,
            execution.capacity,
            execution.agent_id_capacity,
            execution.grid_height,
            execution.grid_width,
            execution.capacity,
            execution.start_energy,
            execution.start_health,
            static_cast<int32_t>(execution.generation),
        }
    );
    if (statuses[kIdentityStatus] != identity::kIdentityStatusOk) {
        statuses[kAggregateStatus] = 1;
        counters[kAggregateFinalLiveCount] = counters[kPlanSurvivorCount];
        return;
    }
    if (
        qualification_fault_stage
        == flattened_birth_scratch_v2::
            kGenomeUpgradeV2FaultTurnoverAfterChildStaging
    ) {
        statuses[kIdentityStatus] = detail::turnover_fault_status_v1(
            qualification_fault_stage
        );
        statuses[kAggregateStatus] = 1;
        counters[kAggregateFinalLiveCount] = counters[kPlanSurvivorCount];
        return;
    }

    int64_t* const turnover_python_raw_draws = tensors.shadow<int64_t>(
        kPythonRandomRawDraws, cell
    );
    const int64_t python_raw_draw_base =
        birth_state.python_random_raw_draw_base[cell];
    if (execution.rng_semantics_version == 2) {
        // Leftover CPython choice-RNG draws are not a V1 Gaussian-tape cursor.
        if (turnover_python_raw_draws[0] < 0) {
            statuses[kScalarStatus] = scalar::kStatusInvalidPythonRandomState;
        }
    } else if (
        python_raw_draw_base < 0
        || turnover_python_raw_draws[0] < python_raw_draw_base
        || turnover_python_raw_draws[0] - python_raw_draw_base
            > execution.gauss_tape_capacity
    ) {
        statuses[kScalarStatus] = scalar::kStatusInvalidPythonRandomState;
    } else {
        turnover_python_raw_draws[0] -= python_raw_draw_base;
    }
    int64_t* diploid_selected_parent_slots = nullptr;
    int64_t* diploid_selected_mate_slots = nullptr;
    const bool valid_inheritance_mode =
        !cross_authority
        || (
            cross_control.inheritance_mode_by_cell != nullptr
            && (
                cross_control.inheritance_mode_by_cell[cell]
                    == cross_path::kInheritanceModeLegacyMidpoint
                || cross_control.inheritance_mode_by_cell[cell]
                    == cross_path::kInheritanceModeLinkedParentCopy
            )
        );
    const bool linked_inheritance =
        cross_authority
        && cross_control.inheritance_mode_by_cell != nullptr
        && cross_control.inheritance_mode_by_cell[cell]
            == cross_path::kInheritanceModeLinkedParentCopy;
    const int64_t inheritance_row_base =
        cell * cross_control.inheritance_rows_per_cell;
    const int64_t planned_birth_count = counters[kRefillPlannedCount];
    if (statuses[kScalarStatus] == scalar::kStatusOk && !diploid_mode) {
    if (
        cross_authority
        && (
            !cross_path::has_complete_ordinary_authority_v1(cross_control)
            || !valid_inheritance_mode
        )
    ) {
        statuses[kScalarStatus] = scalar::kStatusMissingInheritanceSidecar;
    }
    if (
        cross_authority
        && statuses[kScalarStatus] == scalar::kStatusOk
        && (
            planned_birth_count < 0
            || cross_control.inheritance_rows_per_cell < planned_birth_count
        )
    ) {
        statuses[kScalarStatus] =
            scalar::kStatusInheritanceSidecarCapacityExceeded;
    }
    if (cross_authority && statuses[kScalarStatus] == scalar::kStatusOk) {
        const int64_t* const child_slots = at<int64_t>(
            scratch, cell, FlatTurnoverWorkLayout::kCommittedChildSlots
        );
        // The resident sidecar is sized to the admitted transaction bound,
        // which can be smaller than the slot capacity. Turnover may touch only
        // its planned refill rows and clears the complete admitted row window.
        for (
            int64_t birth = 0;
            birth < cross_control.inheritance_rows_per_cell;
            ++birth
        ) {
            const int64_t row = inheritance_row_base + birth;
            cross_control.canonical_parent_0_slots[row] = -1;
            cross_control.canonical_parent_1_slots[row] = -1;
            cross_control.child_slots[row] = -1;
            for (
                int64_t group = 0;
                group < inheritance_linkage_v1::kLinkageGroupCount;
                ++group
            ) {
                const int64_t donor_row =
                    row * inheritance_linkage_v1::kLinkageGroupCount + group;
                cross_control.donor_ordinals[donor_row] =
                    cross_path::kNoDonorOrdinal;
                cross_control.donor_statuses[donor_row] =
                    cross_path::kDonorStatusOk;
            }
            if (birth >= planned_birth_count || !linked_inheritance) {
                continue;
            }
            cross_control.child_slots[row] = child_slots[birth];
        }
        cross_control.donor_draws_added[cell] = 0;
        cross_control.staged_counts[cell] =
            linked_inheritance ? planned_birth_count : 0;
        cross_control.committed_counts[cell] = 0;
        cross_control.inheritance_status[cell] =
            linked_inheritance
            ? cross_path::kInheritanceStatusNotCommitted
            : cross_path::kInheritanceStatusOk;
    }
    if (statuses[kScalarStatus] == scalar::kStatusOk) {
    scalar::
        commit_turnover_fixed_mlp_child_scalar_genome_with_ranked_inheritance_v2_serial(
        {
            tensors.shadow<int64_t>(kSlotToAgentId, cell),
            tensors.shadow<bool>(kAliveMask, cell),
            tensors.shadow<int32_t>(kGenomeHiddenSize, cell),
            tensors.shadow<int32_t>(kGenomeMemoryDim, cell),
            tensors.shadow<int32_t>(kGenomeSensorChannelMask, cell),
            tensors.shadow<int32_t>(kGenomeSensorRadius, cell),
            tensors.shadow<int32_t>(kGenomeIntrospectionChannelMask, cell),
            tensors.shadow<int32_t>(kGenomeSelfRegulationControlMask, cell),
            tensors.shadow<int32_t>(kGenomeSelfModificationTargetMask, cell),
            tensors.shadow<int32_t>(kGenomeDirectedInheritanceChannelMask, cell),
            tensors.shadow<int64_t>(kPolicyMemDim, cell),
            tensors.shadow<bool>(kPolicyUseGru, cell),
            tensors.shadow<float>(kGenomeExplorationBias, cell),
            tensors.shadow<float>(kGenomeAggressionBias, cell),
            tensors.shadow<float>(kGenomeMutationRate, cell),
            tensors.shadow<float>(kGenomeReproductionThreshold, cell),
            tensors.shadow<float>(kGenomeActionTemperature, cell),
            tensors.shadow<float>(kGenomeDriveInertia, cell),
            tensors.shadow<float>(kGenomeDriveNoise, cell),
            tensors.shadow<float>(kGenomeDriveLogitGain, cell),
            tensors.shadow<float>(kGenomePredLr, cell),
            tensors.shadow<float>(kGenomeDriveSpHunger, cell),
            tensors.shadow<float>(kGenomeDriveSpPain, cell),
            tensors.shadow<float>(kGenomeDriveSpFatigue, cell),
            tensors.shadow<float>(kGenomeDriveSpFear, cell),
            tensors.shadow<float>(kGenomeExpPriorStrength, cell),
            tensors.shadow<float>(kGenomeExpPriorNoise, cell),
            tensors.shadow<float>(kGenomeDriveBirthNoise, cell),
            tensors.shadow<float>(kReproThreshold, cell),
            tensors.shadow<float>(kGruUpkeepCost, cell),
            tensors.shadow<float>(kPolicyAttackBias, cell),
            tensors.shadow<float>(kPolicyActionTemperature, cell),
            tensors.shadow<float>(kPolicyDriveLogitGain, cell),
            tensors.shadow<float>(kDriveInertia, cell),
            tensors.shadow<float>(kDriveNoise, cell),
            tensors.shadow<float>(kPredLr, cell),
        },
        {
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kParentPoolSlots),
            counters + kPlanParentPoolCount,
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kCommittedChildSlots),
            counters + kIdentityCommittedCount,
        },
        {
            tensors.shadow<uint32_t>(kPythonRandomWords, cell),
            tensors.shadow<int64_t>(kPythonRandomIndex, cell),
            tensors.shadow<bool>(kPythonRandomHasGauss, cell),
            tensors.shadow<double>(kPythonRandomGaussNext, cell),
            detail::offset_or_null(
                birth_state.python_random_gauss_primary_tape,
                cell * execution.gauss_tape_capacity
            ),
            detail::offset_or_null(
                birth_state.python_random_gauss_secondary_tape,
                cell * execution.gauss_tape_capacity
            ),
            turnover_python_raw_draws,
            tensors.shadow<int64_t>(kMutationPySemanticDraws, cell),
        },
        {
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kCommittedParentAgentIds),
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kCommittedMateAgentIds),
            counters + kScalarCommittedCount,
            counters + kScalarSemanticDrawsAdded,
            statuses + kScalarStatus,
        },
        {
            planned_birth_count,
            execution.capacity,
            execution.gauss_tape_capacity,
            execution.sensor_radius_max,
        },
        at<uint32_t>(scratch, cell, FlatTurnoverWorkLayout::kCpythonWordsB),
        {
            linked_inheritance
                ? inheritance_linkage_v1::InheritanceModeV1::
                    linked_parent_copy_v1
                : inheritance_linkage_v1::InheritanceModeV1::
                    legacy_midpoint_v1,
            linked_inheritance
                ? inheritance_linkage_v1::kLinkedDonorDrawCount
                : inheritance_linkage_v1::kLegacyAddedDonorDrawCount,
            !cross_authority
                || cross_control.mutation_enabled_by_cell[cell],
            {
                cross_authority
                    ? cross_control.genotype_codes
                        + cell * execution.capacity
                    : nullptr,
                cross_authority
                    ? cross_control.digest_rank_by_code
                        + cell * cross_control.genotype_count
                    : nullptr,
                cross_authority ? cross_control.genotype_count : 0,
                cross_authority
                    ? cross_control.canonical_parent_0_slots
                        + inheritance_row_base
                    : nullptr,
                cross_authority
                    ? cross_control.canonical_parent_1_slots
                        + inheritance_row_base
                    : nullptr,
            },
        },
        {
            cross_authority
                ? cross_control.donor_ordinals
                    + inheritance_row_base
                        * inheritance_linkage_v1::kLinkageGroupCount
                : nullptr,
            cross_authority
                ? cross_control.donor_statuses
                    + inheritance_row_base
                        * inheritance_linkage_v1::kLinkageGroupCount
                : nullptr,
            cross_authority
                ? cross_control.donor_draws_added + cell
                : nullptr,
            cross_authority
                ? cross_control.inheritance_status + cell
                : nullptr,
        }
    );
    }
    }
    if (statuses[kScalarStatus] == scalar::kStatusOk && diploid_mode) {
        const int64_t birth_count = counters[kIdentityCommittedCount];
        const int64_t parent_pool_count = counters[kPlanParentPoolCount];
        int64_t* const parent_pool_slots = at<int64_t>(
            scratch, cell, FlatTurnoverWorkLayout::kParentPoolSlots
        );
        int64_t* const child_slots = at<int64_t>(
            scratch, cell, FlatTurnoverWorkLayout::kCommittedChildSlots
        );
        int64_t* const selected_parent_slots = at<int64_t>(
            scratch, cell, FlatTurnoverWorkLayout::kReleaseAgentIds
        );
        int64_t* const selected_mate_slots =
            selected_parent_slots + execution.capacity;
        diploid_selected_parent_slots = selected_parent_slots;
        diploid_selected_mate_slots = selected_mate_slots;
        int64_t* const selected_parent_ids = at<int64_t>(
            scratch, cell, FlatTurnoverWorkLayout::kCommittedParentAgentIds
        );
        int64_t* const selected_mate_ids = at<int64_t>(
            scratch, cell, FlatTurnoverWorkLayout::kCommittedMateAgentIds
        );
        if (
            parent_pool_count < (birth_count > 0 ? 2 : 0)
            || parent_pool_count > execution.capacity
        ) {
            statuses[kScalarStatus] = scalar::kStatusInvalidScalarBirthPlan;
        } else {
            uint32_t* const python_words =
                tensors.shadow<uint32_t>(kPythonRandomWords, cell);
            scalar::DeviceCpythonRandom rng{
                python_words,
                tensors.shadow<int64_t>(kPythonRandomIndex, cell)[0],
                tensors.shadow<bool>(kPythonRandomHasGauss, cell)[0],
                tensors.shadow<double>(kPythonRandomGaussNext, cell)[0],
                detail::offset_or_null(
                    birth_state.python_random_gauss_primary_tape,
                    cell * execution.gauss_tape_capacity
                ),
                detail::offset_or_null(
                    birth_state.python_random_gauss_secondary_tape,
                    cell * execution.gauss_tape_capacity
                ),
                execution.gauss_tape_capacity,
                turnover_python_raw_draws[0],
            };
            const int64_t allele_cell_stride =
                diploid::kAllelePlaneCount * execution.capacity;
            float* const allele_cell_base =
                birth_state.diploid_allele_shadow_storage
                + cell * allele_cell_stride;
            const diploid::AllelePlanesV1 allele_planes{
                allele_cell_base,
                allele_cell_base
                    + diploid::kLocusCount * execution.capacity,
                execution.capacity,
                execution.capacity,
            };
            const int64_t cell_capacity_offset =
                cell * execution.capacity;
            const diploid::PhenotypeArraysV1 phenotype_arrays{{
                tensors.shadow<float>(kGenomeExplorationBias, cell),
                tensors.shadow<float>(kGenomeAggressionBias, cell),
                tensors.shadow<float>(kGenomeActionTemperature, cell),
                tensors.shadow<float>(kGenomeDriveInertia, cell),
                tensors.shadow<float>(kGenomeDriveNoise, cell),
                tensors.shadow<float>(kGenomeDriveLogitGain, cell),
                tensors.shadow<float>(kGenomePredLr, cell),
                tensors.shadow<float>(kGenomeDriveSpHunger, cell),
                tensors.shadow<float>(kGenomeDriveSpPain, cell),
                tensors.shadow<float>(kGenomeDriveSpFatigue, cell),
                tensors.shadow<float>(kGenomeDriveSpFear, cell),
                tensors.shadow<float>(kGenomeExpPriorStrength, cell),
                tensors.shadow<float>(kGenomeExpPriorNoise, cell),
                tensors.shadow<float>(kGenomeDriveBirthNoise, cell),
                tensors.shadow<float>(kGenomeReproductionThreshold, cell),
                tensors.shadow<float>(kGenomeMutationRate, cell),
                birth_state.genome_resource_handling_axis_shadow
                    + cell_capacity_offset,
            }};
            namespace genome_v2 = eiel::genome_upgrade_v2_q10_v1;
            const int64_t gv2_allele_cell_stride =
                2 * genome_v2::kProductCount * genome_v2::kParameterCount
                * execution.capacity;
            const int64_t gv2_modifier_cell_stride =
                2 * genome_v2::kProductCount * execution.capacity;
            const int64_t gv2_product_cell_stride =
                genome_v2::kProductCount * execution.capacity;
            const int64_t gv2_phenotype_cell_stride =
                genome_v2::kPhenotypeCount * execution.capacity;
            const genome_v2::ResidentPlanesV1 genome_v2_planes{
                genome_upgrade_v2_mode
                    ? birth_state.genome_upgrade_v2_alleles_q10_shadow
                        + cell * gv2_allele_cell_stride
                    : nullptr,
                genome_upgrade_v2_mode
                    ? birth_state.genome_upgrade_v2_modifiers_q10_shadow
                        + cell * gv2_modifier_cell_stride
                    : nullptr,
                genome_upgrade_v2_mode
                    ? birth_state.genome_upgrade_v2_products_q10_shadow
                        + cell * gv2_product_cell_stride
                    : nullptr,
                genome_upgrade_v2_mode
                    ? birth_state.genome_upgrade_v2_phenotypes_q10_shadow
                        + cell * gv2_phenotype_cell_stride
                    : nullptr,
                genome_upgrade_v2_mode
                    ? birth_state.genome_upgrade_v2_expression_cost_q10_shadow
                        + cell_capacity_offset
                    : nullptr,
                genome_upgrade_v2_mode
                    ? birth_state.genome_upgrade_v2_active_product_count_shadow
                        + cell_capacity_offset
                    : nullptr,
                genome_upgrade_v2_mode
                    ? birth_state
                        .genome_upgrade_v2_saturated_product_count_shadow
                        + cell_capacity_offset
                    : nullptr,
                execution.capacity,
            };
            int64_t semantic_draws_added = 0;
            for (int64_t birth = 0; birth < birth_count; ++birth) {
                int64_t parent_slot = -1;
                int64_t mate_slot = -1;
                do {
                    parent_slot = parent_pool_slots[rng.randbelow(
                        static_cast<uint64_t>(parent_pool_count)
                    )];
                    mate_slot = parent_pool_slots[rng.randbelow(
                        static_cast<uint64_t>(parent_pool_count)
                    )];
                    semantic_draws_added += 2;
                } while (mate_slot == parent_slot);
                const int64_t child_slot = child_slots[birth];
                if (
                    child_slot < 0 || child_slot >= execution.capacity
                    || parent_slot < 0 || parent_slot >= execution.capacity
                    || mate_slot < 0 || mate_slot >= execution.capacity
                ) {
                    statuses[kScalarStatus] =
                        scalar::kStatusInvalidScalarBirthPlan;
                    break;
                }
                selected_parent_slots[birth] = parent_slot;
                selected_mate_slots[birth] = mate_slot;
                selected_parent_ids[birth] =
                    tensors.shadow<int64_t>(kSlotToAgentId, cell)[parent_slot];
                selected_mate_ids[birth] =
                    tensors.shadow<int64_t>(kSlotToAgentId, cell)[mate_slot];

                uint8_t linked_donors[7];
                for (int32_t group = 0; group < 7; ++group) {
                    linked_donors[group] =
                        rng.random_double() < 0.5 ? 0 : 1;
                    semantic_draws_added += 1;
                }
                const int64_t structural_donor =
                    linked_donors[4] == 0 ? parent_slot : mate_slot;
                tensors.shadow<int32_t>(kGenomeHiddenSize, cell)[child_slot] =
                    tensors.shadow<int32_t>(
                        kGenomeHiddenSize, cell
                    )[structural_donor];
                tensors.shadow<int32_t>(kGenomeMemoryDim, cell)[child_slot] = 0;
                tensors.shadow<int64_t>(kPolicyMemDim, cell)[child_slot] = 0;
                tensors.shadow<bool>(kPolicyUseGru, cell)[child_slot] = false;
                tensors.shadow<float>(kGruUpkeepCost, cell)[child_slot] = 0.0f;
                tensors.shadow<int32_t>(
                    kGenomeSensorChannelMask, cell
                )[child_slot] = (
                    tensors.shadow<int32_t>(
                        kGenomeSensorChannelMask, cell
                    )[parent_slot] & 1023
                ) | 1;
                int32_t sensor_radius = tensors.shadow<int32_t>(
                    kGenomeSensorRadius, cell
                )[parent_slot];
                sensor_radius = sensor_radius < 0
                    ? 0
                    : (
                        sensor_radius > execution.sensor_radius_max
                            ? execution.sensor_radius_max
                            : sensor_radius
                    );
                tensors.shadow<int32_t>(
                    kGenomeSensorRadius, cell
                )[child_slot] = sensor_radius;
                tensors.shadow<int32_t>(
                    kGenomeIntrospectionChannelMask, cell
                )[child_slot] = tensors.shadow<int32_t>(
                    kGenomeIntrospectionChannelMask, cell
                )[parent_slot] & 32767;
                tensors.shadow<int32_t>(
                    kGenomeSelfRegulationControlMask, cell
                )[child_slot] = tensors.shadow<int32_t>(
                    kGenomeSelfRegulationControlMask, cell
                )[parent_slot] & 31;
                tensors.shadow<int32_t>(
                    kGenomeSelfModificationTargetMask, cell
                )[child_slot] = tensors.shadow<int32_t>(
                    kGenomeSelfModificationTargetMask, cell
                )[parent_slot] & 31;
                tensors.shadow<int32_t>(
                    kGenomeDirectedInheritanceChannelMask, cell
                )[child_slot] = tensors.shadow<int32_t>(
                    kGenomeDirectedInheritanceChannelMask, cell
                )[parent_slot] & 63;

                diploid::StagedChildV1 staged{};
                const int64_t parent_id = selected_parent_ids[birth];
                const int64_t mate_id = selected_mate_ids[birth];
                const int64_t canonical_parent_0_slot =
                    parent_id <= mate_id ? parent_slot : mate_slot;
                const int64_t canonical_parent_1_slot =
                    parent_id <= mate_id ? mate_slot : parent_slot;
                const int64_t child_id = at<int64_t>(
                    scratch, cell, FlatTurnoverWorkLayout::kCommittedChildAgentIds
                )[birth];
                const bool rngv2_birth = execution.rng_semantics_version == 2;
                const uint64_t rngv2_absolute_tick = static_cast<uint64_t>(
                    execution.generation * execution.ticks_per_generation
                    + execution.start_tick + execution.ticks
                );
                const diploid::StatusV1 diploid_status = rngv2_birth
                    ? diploid::stage_child_v2(
                        allele_planes,
                        canonical_parent_0_slot,
                        canonical_parent_1_slot,
                        turnover_world_uid_for_cell_v1(execution, cell),
                        rngv2_absolute_tick,
                        static_cast<uint64_t>(child_id),
                        birth_state.inheritance_mutation_enabled[cell],
                        birth_state.resource_handling_enabled[cell],
                        birth_state.mutation_sigma_multiplier[cell],
                        &staged
                    )
                    : diploid::stage_child_v1(
                        allele_planes,
                        parent_slot,
                        mate_slot,
                        rng,
                        birth_state.inheritance_mutation_enabled[cell],
                        birth_state.resource_handling_enabled[cell],
                        birth_state.mutation_sigma_multiplier[cell],
                        &staged
                    );
                if (diploid_status != diploid::StatusV1::ok) {
                    statuses[kScalarStatus] =
                        scalar::kStatusFreshGaussUnsupported
                        + static_cast<int32_t>(diploid_status);
                    break;
                }
                if (
                    diploid::commit_staged_child_v1(
                        allele_planes,
                        phenotype_arrays,
                        child_slot,
                        staged
                    ) != diploid::StatusV1::ok
                ) {
                    statuses[kScalarStatus] =
                        scalar::kStatusInvalidScalarBirthPlan;
                    break;
                }
                semantic_draws_added += staged.semantic_draws;
                if (genome_upgrade_v2_mode) {
                    genome_v2::StagedChildV1 genome_v2_staged{};
                    const bool staged_ok = rngv2_birth
                        ? genome_v2::stage_child_v2(
                            genome_v2_planes,
                            canonical_parent_0_slot,
                            canonical_parent_1_slot,
                            turnover_world_uid_for_cell_v1(execution, cell),
                            rngv2_absolute_tick,
                            static_cast<uint64_t>(child_id),
                            birth_state
                                .genome_upgrade_v2_mutation_rate_q10[cell],
                            birth_state
                                .genome_upgrade_v2_mutation_step_q10[cell],
                            &genome_v2_staged
                        )
                        : genome_v2::stage_child_v1(
                            genome_v2_planes,
                            parent_slot,
                            mate_slot,
                            rng,
                            birth_state
                                .genome_upgrade_v2_mutation_rate_q10[cell],
                            birth_state
                                .genome_upgrade_v2_mutation_step_q10[cell],
                            &genome_v2_staged
                        );
                    if (!staged_ok) {
                        statuses[kScalarStatus] =
                            scalar::kStatusInvalidScalarBirthPlan;
                        break;
                    }
                    const int64_t neutral_context[
                        genome_v2::kContextCount
                    ] = {};
                    if (
                        !genome_v2::commit_staged_child_v1(
                            genome_v2_planes,
                            child_slot,
                            neutral_context,
                            genome_v2_staged
                        )
                    ) {
                        statuses[kScalarStatus] =
                            scalar::kStatusInvalidScalarBirthPlan;
                        break;
                    }
                    semantic_draws_added +=
                        genome_v2_staged.semantic_draws;
                }
                const float* const phenotype = staged.phenotype;
                tensors.shadow<float>(kReproThreshold, cell)[child_slot] =
                    phenotype[diploid::kReproductionThreshold];
                tensors.shadow<float>(kPolicyAttackBias, cell)[child_slot] =
                    phenotype[diploid::kAggressionBias];
                tensors.shadow<float>(
                    kPolicyActionTemperature, cell
                )[child_slot] = phenotype[diploid::kActionTemperature];
                tensors.shadow<float>(
                    kPolicyDriveLogitGain, cell
                )[child_slot] = phenotype[diploid::kDriveLogitGain];
                tensors.shadow<float>(kDriveInertia, cell)[child_slot] =
                    phenotype[diploid::kDriveInertia];
                tensors.shadow<float>(kDriveNoise, cell)[child_slot] =
                    phenotype[diploid::kDriveNoise];
                tensors.shadow<float>(kPredLr, cell)[child_slot] =
                    phenotype[diploid::kPredLr];
            }
            if (statuses[kScalarStatus] == scalar::kStatusOk) {
                tensors.shadow<int64_t>(kPythonRandomIndex, cell)[0] =
                    rng.index;
                tensors.shadow<bool>(kPythonRandomHasGauss, cell)[0] =
                    rng.has_gauss_next;
                tensors.shadow<double>(kPythonRandomGaussNext, cell)[0] =
                    rng.gauss_next;
                turnover_python_raw_draws[0] = rng.raw_draws;
                tensors.shadow<int64_t>(
                    kMutationPySemanticDraws, cell
                )[0] += semantic_draws_added;
                counters[kScalarCommittedCount] = birth_count;
                counters[kScalarSemanticDrawsAdded] = semantic_draws_added;
            }
        }
    }
    if (statuses[kScalarStatus] == scalar::kStatusOk) {
        if (execution.rng_semantics_version == 2) {
            if (turnover_python_raw_draws[0] < 0) {
                statuses[kScalarStatus] =
                    scalar::kStatusInvalidPythonRandomState;
            }
        } else if (
            turnover_python_raw_draws[0] < 0
            || turnover_python_raw_draws[0]
                > execution.gauss_tape_capacity
            || python_raw_draw_base
                > std::numeric_limits<int64_t>::max()
                    - turnover_python_raw_draws[0]
        ) {
            statuses[kScalarStatus] =
                scalar::kStatusInvalidPythonRandomState;
        } else {
            turnover_python_raw_draws[0] += python_raw_draw_base;
        }
    }
    if (
        cross_authority
        && statuses[kScalarStatus] == scalar::kStatusOk
        && linked_inheritance
    ) {
        cross_control.committed_counts[cell] = planned_birth_count;
        const int64_t slot_base = cell * execution.capacity;
        const int64_t* const child_slots = at<int64_t>(
            scratch, cell, FlatTurnoverWorkLayout::kCommittedChildSlots
        );
        for (int64_t birth = 0; birth < planned_birth_count; ++birth) {
            const int64_t row = inheritance_row_base + birth;
            const int64_t parent_0_slot =
                cross_control.canonical_parent_0_slots[row];
            const int64_t parent_1_slot =
                cross_control.canonical_parent_1_slots[row];
            const int64_t child_slot = child_slots[birth];
            int64_t child_code = 0;
            for (
                int64_t group = 0;
                group < inheritance_linkage_v1::kLinkageGroupCount;
                ++group
            ) {
                const uint8_t donor = cross_control.donor_ordinals[
                    row * inheritance_linkage_v1::kLinkageGroupCount + group
                ];
                const int64_t donor_slot =
                    donor == 0 ? parent_0_slot : parent_1_slot;
                child_code |= (
                    (
                        cross_control.genotype_codes[
                            slot_base + donor_slot
                        ] >> group
                    ) & 1LL
                ) << group;
            }
            cross_control.genotype_codes[slot_base + child_slot] = child_code;
        }
    }

    if (statuses[kScalarStatus] != scalar::kStatusOk) {
        statuses[kAggregateStatus] = 2;
        counters[kAggregateFinalLiveCount] = counters[kPlanSurvivorCount];
        return;
    }
    const bool parentage_rebound = identity::rebind_turnover_refill_parentage_serial(
        {
            tensors.shadow<int64_t>(kSlotToAgentId, cell),
            tensors.shadow<int32_t>(kAgentIdToSlot, cell),
            tensors.shadow<int64_t>(kSlotTokens, cell),
            tensors.shadow<bool>(kAliveMask, cell),
            tensors.shadow<int32_t>(kPosY, cell),
            tensors.shadow<int32_t>(kPosX, cell),
            tensors.shadow<int32_t>(kAgentGrid, cell),
            tensors.shadow<float>(kEnergy, cell),
            tensors.shadow<float>(kHealth, cell),
            tensors.shadow<float>(kInjury, cell),
            tensors.shadow<float>(kScar, cell),
            tensors.shadow<int32_t>(kAge, cell),
            tensors.shadow<int32_t>(kReproCooldown, cell),
            tensors.shadow<float>(kReproThreshold, cell),
            tensors.shadow<int32_t>(kOffspringCount, cell),
            tensors.shadow<int32_t>(kLineageFounderId, cell),
            tensors.shadow<int32_t>(kLineageDepth, cell),
            tensors.shadow<int32_t>(kLineageBirthGeneration, cell),
            tensors.shadow<int32_t>(kLineageBirthTick, cell),
            tensors.shadow<int32_t>(kPatchHabitatLineageId, cell),
            tensors.shadow<double>(kPolicyEnergyObservation, cell),
            tensors.shadow<int32_t>(kSurvivalSteps, cell),
            tensors.shadow<int32_t>(kRestStreak, cell),
            tensors.shadow<int64_t>(kActionCacheToken, cell),
            tensors.shadow<int32_t>(kActionCacheCursor, cell),
            tensors.shadow<int32_t>(kActionCacheNext, cell),
            tensors.shadow<int64_t>(kSlotFreeQueue, cell),
            tensors.shadow<int64_t>(kSlotFreeQueueCount, cell),
            tensors.shadow<int64_t>(kPopulationNextId, cell),
        },
        at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kRefillParentSlots),
        at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kRefillMateSlots),
        diploid_mode
            ? diploid_selected_parent_slots
            : at<int64_t>(
                scratch,
                cell,
                FlatTurnoverWorkLayout::kCommittedParentAgentIds
            ),
        diploid_mode
            ? diploid_selected_mate_slots
            : at<int64_t>(
                scratch,
                cell,
                FlatTurnoverWorkLayout::kCommittedMateAgentIds
            ),
        at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kCommittedChildSlots),
        counters + kIdentityCommittedCount,
        at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kCommittedParentAgentIds),
        at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kCommittedMateAgentIds),
        {
            execution.capacity,
            execution.capacity,
            execution.agent_id_capacity,
            execution.grid_height,
            execution.grid_width,
            execution.capacity,
            execution.start_energy,
            execution.start_health,
            static_cast<int32_t>(execution.generation),
        }
    );
    if (!parentage_rebound) {
        statuses[kIdentityStatus] =
            identity::kIdentityStatusParentageRebindFailed;
        statuses[kAggregateStatus] = 2;
        counters[kAggregateFinalLiveCount] = counters[kPlanSurvivorCount];
        return;
    }

    if (modular_context.enabled && modular_context.lifecycle_enabled) {
        if (!modular::device::lifecycle_table_valid_v2(modular_context)) {
            statuses[kAggregateStatus] = 8;
            counters[kAggregateFinalLiveCount] = counters[kPlanSurvivorCount];
            return;
        }
        const int64_t committed = counters[kIdentityCommittedCount];
        const int64_t* const parent_ids = at<int64_t>(
            scratch, cell, FlatTurnoverWorkLayout::kCommittedParentAgentIds
        );
        const int64_t* const child_ids = at<int64_t>(
            scratch, cell, FlatTurnoverWorkLayout::kCommittedChildAgentIds
        );
        const int64_t* const child_slots = at<int64_t>(
            scratch, cell, FlatTurnoverWorkLayout::kCommittedChildSlots
        );
        const int64_t* const parent_slots = at<int64_t>(
            scratch, cell, FlatTurnoverWorkLayout::kRefillParentSlots
        );
        const int64_t* const mate_slots = at<int64_t>(
            scratch, cell, FlatTurnoverWorkLayout::kRefillMateSlots
        );
        const int64_t* const slot_tokens =
            tensors.shadow<int64_t>(kSlotTokens, cell);
        for (int64_t birth = 0; birth < committed; ++birth) {
            const int64_t parent_id = parent_ids[birth];
            const int64_t child_slot = child_slots[birth];
            const int64_t parent_slot = parent_slots[birth];
            const int64_t mate_slot = mate_slots[birth];
            if (
                child_slot < 0 || child_slot >= execution.capacity
                || !modular::device::inherit_lifecycle_newborn_v2(
                    modular_context,
                    true,
                    cell,
                    parent_slot,
                    mate_slot,
                    child_slot,
                    child_ids[birth],
                    slot_tokens[child_slot]
                )
                || !modular::device::mutate_turnover_descendant_parameter_v1(
                    modular_context,
                    cell,
                    birth,
                    parent_slot,
                    child_slot,
                    parent_id,
                    child_ids[birth],
                    slot_tokens[child_slot],
                    execution.generation + 1,
                    0
                )
            ) {
                statuses[kAggregateStatus] = 8;
                counters[kAggregateFinalLiveCount] =
                    counters[kPlanSurvivorCount];
                return;
            }
        }
    }
    if (
        qualification_fault_stage
        == flattened_birth_scratch_v2::
            kGenomeUpgradeV2FaultTurnoverAfterMutation
    ) {
        statuses[kScalarStatus] = detail::turnover_fault_status_v1(
            qualification_fault_stage
        );
        statuses[kAggregateStatus] = 2;
        counters[kAggregateFinalLiveCount] = counters[kPlanSurvivorCount];
        return;
    }

    drives::initialize_turnover_newborn_drives_serial(
        {
            tensors.shadow<int64_t>(kSlotToAgentId, cell),
            tensors.shadow<bool>(kAliveMask, cell),
            tensors.shadow<double>(kPolicyEnergyObservation, cell),
            tensors.shadow<float>(kGenomeDriveSpHunger, cell),
            tensors.shadow<float>(kGenomeDriveSpPain, cell),
            tensors.shadow<float>(kGenomeDriveSpFatigue, cell),
            tensors.shadow<float>(kGenomeDriveSpFear, cell),
            tensors.shadow<float>(kGenomeExpPriorNoise, cell),
            tensors.shadow<float>(kGenomeDriveBirthNoise, cell),
        },
        {
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kCommittedChildSlots),
            counters + kIdentityCommittedCount,
        },
        {
            tensors.shadow<uint64_t>(kPopulationPcg64StateWords, cell),
            tensors.shadow<int64_t>(kPopulationPcg64BufferWords, cell),
            tensors.shadow<int64_t>(kMutationNpSemanticDraws, cell),
        },
        {
            tensors.shadow<float>(kDrives, cell),
            tensors.shadow<float>(kBirthResetShock, cell),
            counters + kDriveCommittedCount,
            counters + kDriveBirthNoiseRows,
            counters + kDriveExpPriorRows,
            counters + kDriveStandardNormalRows,
            counters + kDriveSemanticDrawsAdded,
            statuses + kDriveStatus,
        },
        {
            execution.capacity,
            execution.capacity,
            execution.drive_width,
            execution.max_energy,
        }
    );

    drives::initialize_turnover_newborn_chemistry_serial(
        {
            tensors.shadow<int64_t>(kSlotToAgentId, cell),
            tensors.shadow<bool>(kAliveMask, cell),
            tensors.shadow<double>(kPolicyEnergyObservation, cell),
            tensors.shadow<float>(kChemistry, cell),
        },
        {
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kCommittedChildSlots),
            counters + kIdentityCommittedCount,
        },
        {statuses + kChemistryStatus},
        {
            execution.capacity,
            execution.capacity,
            execution.chemistry_width,
            execution.max_energy,
        }
    );
    counters[kChemistryCommittedCount] =
        statuses[kChemistryStatus] == drives::kStatusOk
        ? counters[kIdentityCommittedCount]
        : 0;

    bool fixed_gru_turnover = counters[kIdentityCommittedCount] > 0;
    bool mixed_turnover = false;
    bool saw_gru_turnover = false;
    bool saw_mlp_turnover = false;
    int64_t* const policy_parent_slots = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kRefillParentSlots
    );
    int64_t* const policy_mate_slots = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kRefillMateSlots
    );
    const int64_t* const committed_child_slots = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kCommittedChildSlots
    );
    const bool* const rebound_policy_use_gru = tensors.shadow<bool>(
        kPolicyUseGru, cell
    );
    const bool defer_mlp_policy_rows =
        tensors.entry(kMlpPack).shadow_bytes_per_cell == 0;
    for (int64_t birth = 0; birth < counters[kIdentityCommittedCount]; ++birth) {
        const int64_t parent_slot = policy_parent_slots[birth];
        const int64_t mate_slot = policy_mate_slots[birth];
        const int64_t child_slot = committed_child_slots[birth];
        const bool row_gru = parent_slot >= 0 && mate_slot >= 0
            && child_slot >= 0
            && rebound_policy_use_gru[parent_slot]
            && rebound_policy_use_gru[mate_slot]
            && rebound_policy_use_gru[child_slot];
        const bool row_mlp = parent_slot >= 0 && mate_slot >= 0
            && child_slot >= 0
            && !rebound_policy_use_gru[parent_slot]
            && !rebound_policy_use_gru[mate_slot]
            && !rebound_policy_use_gru[child_slot];
        saw_gru_turnover = saw_gru_turnover || row_gru;
        saw_mlp_turnover = saw_mlp_turnover || row_mlp;
        fixed_gru_turnover = fixed_gru_turnover && row_gru;
        mixed_turnover = mixed_turnover || (!row_gru && !row_mlp)
            || (saw_gru_turnover && saw_mlp_turnover);
    }

    if (mixed_turnover) {
        // Mixed-family turnover is admitted only for the frozen shared-policy
        // lane. The scalar stage has already assigned the selected parent's
        // family to the child. No policy crossover, mutation tape, or donor
        // draw is consumed in this lane.
        statuses[kPolicyStatus] = policy::kStatusOk;
        counters[kPolicyCommittedCount] = 0;
        counters[kPolicySemanticDrawsAdded] = 0;
        counters[kPolicyMutationEvents] = 0;
        counters[kPolicyMutatedElements] = 0;
        at<double>(
            scratch,
            cell,
            FlatTurnoverWorkLayout::kPolicyMutationDeltaTotal
        )[0] = 0.0;
        const float* const source_mlp = tensors.source<float>(kMlpPack, cell);
        float* const shadow_mlp = defer_mlp_policy_rows
            ? nullptr
            : tensors.shadow<float>(kMlpPack, cell);
        bool* const mutation_occurred = at<bool>(
            scratch,
            cell,
            FlatTurnoverWorkLayout::kPolicyMutationOccurred
        );
        float* const mutation_delta = at<float>(
            scratch,
            cell,
            FlatTurnoverWorkLayout::kPolicyMutationDelta
        );
        if (!execution.immutable_shared_gru_policy || defer_mlp_policy_rows) {
            statuses[kPolicyStatus] = policy::kStatusInvalidPlan;
        }
        for (
            int64_t birth = 0;
            birth < counters[kIdentityCommittedCount]
                && statuses[kPolicyStatus] == policy::kStatusOk;
            ++birth
        ) {
            const int64_t parent_slot = policy_parent_slots[birth];
            const int64_t child_slot = committed_child_slots[birth];
            if (
                parent_slot < 0 || child_slot < 0
                || rebound_policy_use_gru[parent_slot]
                    != rebound_policy_use_gru[child_slot]
            ) {
                statuses[kPolicyStatus] = policy::kStatusInvalidPlan;
                break;
            }
            for (int64_t column = 0; column < execution.mlp_row_width; ++column) {
                shadow_mlp[birth * execution.mlp_row_width + column] =
                    source_mlp[parent_slot * execution.mlp_row_width + column];
            }
            for (int64_t column = 0; column < execution.hmem_width; ++column) {
                tensors.shadow<float>(kHmem, cell)[
                    child_slot * execution.hmem_width + column
                ] = 0.0F;
            }
            mutation_occurred[birth] = false;
            mutation_delta[birth] = 0.0F;
            counters[kPolicyCommittedCount] += 1;
        }
    } else if (fixed_gru_turnover) {
        if (
            defer_mlp_policy_rows
            || execution.gru_pack == nullptr || execution.gru_row_width <= 0
        ) {
            statuses[kPolicyStatus] = policy::kStatusInvalidPlan;
        } else {
            const float* const source_mlp = tensors.source<float>(kMlpPack, cell);
            float* const shadow_mlp = tensors.shadow<float>(kMlpPack, cell);
            for (int64_t birth = 0; birth < counters[kIdentityCommittedCount]; ++birth) {
                const int64_t parent_slot = policy_parent_slots[birth];
                for (int64_t column = 0; column < execution.mlp_row_width; ++column) {
                    shadow_mlp[birth * execution.mlp_row_width + column] =
                        source_mlp[parent_slot * execution.mlp_row_width + column];
                }
            }
            policy::prepare_stable_gru_turnover_child_policy_serial(
                {
                    tensors.shadow<int64_t>(kSlotToAgentId, cell),
                    tensors.shadow<bool>(kAliveMask, cell),
                    tensors.shadow<float>(kMlpPack, cell),
                    tensors.shadow<float>(kHmem, cell),
                    tensors.shadow<bool>(kPolicyUseGru, cell),
                    tensors.shadow<uint8_t>(kMutationTorchCpuState, cell),
                    tensors.shadow<int64_t>(kMutationTorchCpuSemanticDraws, cell),
                },
                {
                    policy_parent_slots,
                    policy_mate_slots,
                    committed_child_slots,
                    counters + kIdentityCommittedCount,
                },
                {
                    tensors.shadow<uint32_t>(kPythonRandomWords, cell),
                    tensors.shadow<int64_t>(kPythonRandomIndex, cell),
                    tensors.shadow<bool>(kPythonRandomHasGauss, cell),
                    tensors.shadow<double>(kPythonRandomGaussNext, cell),
                    nullptr,
                    nullptr,
                    tensors.shadow<int64_t>(kPythonRandomRawDraws, cell),
                    tensors.shadow<int64_t>(kMutationPySemanticDraws, cell),
                },
                at<int64_t>(
                    scratch, cell, FlatTurnoverWorkLayout::kRankedSlots
                ),
                {
                    counters + kPolicyCommittedCount,
                    counters + kPolicySemanticDrawsAdded,
                    counters + kPolicyMutationEvents,
                    counters + kPolicyMutatedElements,
                    at<double>(
                        scratch,
                        cell,
                        FlatTurnoverWorkLayout::kPolicyMutationDeltaTotal
                    ),
                    at<bool>(
                        scratch,
                        cell,
                        FlatTurnoverWorkLayout::kPolicyMutationOccurred
                    ),
                    at<float>(
                        scratch,
                        cell,
                        FlatTurnoverWorkLayout::kPolicyMutationDelta
                    ),
                    statuses + kPolicyStatus,
                },
                {
                    execution.capacity,
                    execution.capacity,
                    execution.gru_row_width,
                    execution.hmem_width,
                }
            );
        }
    } else if (execution.rng_semantics_version == 2) {
        if (config.hapb_host_authoritative_policy) {
            statuses[kPolicyStatus] = policy::kStatusInvalidPlan;
        } else {
            const uint64_t absolute_tick = static_cast<uint64_t>(
                execution.generation * execution.ticks_per_generation
                + execution.start_tick + execution.ticks
            );
            policy::inherit_fixed_mlp_turnover_child_policy_rngv2_serial(
                {
                    tensors.shadow<int64_t>(kSlotToAgentId, cell),
                    tensors.shadow<bool>(kAliveMask, cell),
                    defer_mlp_policy_rows
                        ? tensors.source<float>(kMlpPack, cell)
                        : tensors.shadow<float>(kMlpPack, cell),
                    tensors.shadow<float>(kHmem, cell),
                    tensors.shadow<bool>(kPolicyUseGru, cell),
                    tensors.shadow<uint8_t>(kMutationTorchCpuState, cell),
                    tensors.shadow<int64_t>(
                        kMutationTorchCpuSemanticDraws, cell
                    ),
                },
                {
                    policy_parent_slots,
                    policy_mate_slots,
                    committed_child_slots,
                    counters + kIdentityCommittedCount,
                },
                {
                    counters + kPolicyCommittedCount,
                    counters + kPolicySemanticDrawsAdded,
                    counters + kPolicyMutationEvents,
                    counters + kPolicyMutatedElements,
                    at<double>(
                        scratch,
                        cell,
                        FlatTurnoverWorkLayout::kPolicyMutationDeltaTotal
                    ),
                    at<bool>(
                        scratch,
                        cell,
                        FlatTurnoverWorkLayout::kPolicyMutationOccurred
                    ),
                    at<float>(
                        scratch,
                        cell,
                        FlatTurnoverWorkLayout::kPolicyMutationDelta
                    ),
                    statuses + kPolicyStatus,
                },
                {
                    execution.capacity,
                    execution.capacity,
                    execution.mlp_row_width,
                    execution.hmem_width,
                    execution.torch_state_bytes,
                    execution.policy_tape_capacity,
                    execution.policy_mutation_tape_width,
                },
                execution.rng_namespace_key,
                turnover_world_uid_for_cell_v1(execution, cell),
                execution.rng_semantic_epoch,
                absolute_tick,
                tensors.source<float>(kMlpPack, cell),
                !defer_mlp_policy_rows,
                defer_mlp_policy_rows
            );
        }
    } else {
        const int64_t policy_mutation_cell_stride =
            execution.policy_tape_capacity
            * execution.policy_mutation_tape_width;
        const int64_t policy_crossover_cell_stride =
            execution.policy_tape_capacity * ((execution.mlp_row_width + 7) / 8);
        int64_t* const turnover_policy_cursor = tensors.shadow<int64_t>(
            kPolicyBirthTapeCursor, cell
        );
        const int64_t policy_cursor_base =
            birth_state.policy_birth_tape_cursor_base[cell];
        if (
            policy_cursor_base < 0
            || turnover_policy_cursor[0] < policy_cursor_base
            || turnover_policy_cursor[0] - policy_cursor_base
                > execution.policy_tape_capacity
        ) {
            statuses[kPolicyStatus] = policy::kStatusTapeExhausted;
        } else {
            turnover_policy_cursor[0] -= policy_cursor_base;
        }
        if (statuses[kPolicyStatus] == policy::kStatusOk) {
        policy::inherit_fixed_mlp_turnover_child_policy_serial(
        {
            tensors.shadow<int64_t>(kSlotToAgentId, cell),
            tensors.shadow<bool>(kAliveMask, cell),
            defer_mlp_policy_rows
                ? tensors.source<float>(kMlpPack, cell)
                : tensors.shadow<float>(kMlpPack, cell),
            tensors.shadow<float>(kHmem, cell),
            tensors.shadow<bool>(kPolicyUseGru, cell),
            tensors.shadow<uint8_t>(kMutationTorchCpuState, cell),
            tensors.shadow<int64_t>(kMutationTorchCpuSemanticDraws, cell),
        },
        {
            policy_parent_slots,
            policy_mate_slots,
            at<int64_t>(scratch, cell, FlatTurnoverWorkLayout::kCommittedChildSlots),
            counters + kIdentityCommittedCount,
        },
        {
            detail::offset_or_null(
                birth_state.policy_birth_delta_tape,
                cell * policy_mutation_cell_stride
            ),
            detail::offset_or_null(
                birth_state.policy_birth_mutation_mask_tape,
                cell * policy_mutation_cell_stride
            ),
            detail::offset_or_null(
                birth_state.policy_birth_crossover_mask_packed_tape,
                cell * policy_crossover_cell_stride
            ),
            detail::offset_or_null(
                birth_state.policy_birth_post_rng_state_packed_tape,
                cell * execution.policy_tape_capacity
                    * torch_rng_pack::kPackedStateBytes
            ),
            detail::offset_or_null(
                birth_state.policy_birth_semantic_draws_tape,
                cell * execution.policy_tape_capacity
            ),
            turnover_policy_cursor,
        },
        {
            counters + kPolicyCommittedCount,
            counters + kPolicySemanticDrawsAdded,
            counters + kPolicyMutationEvents,
            counters + kPolicyMutatedElements,
            at<double>(
                scratch,
                cell,
                FlatTurnoverWorkLayout::kPolicyMutationDeltaTotal
            ),
            at<bool>(
                scratch,
                cell,
                FlatTurnoverWorkLayout::kPolicyMutationOccurred
            ),
            at<float>(
                scratch,
                cell,
                FlatTurnoverWorkLayout::kPolicyMutationDelta
            ),
            statuses + kPolicyStatus,
        },
        {
            execution.capacity,
            execution.capacity,
            execution.mlp_row_width,
            execution.hmem_width,
            execution.torch_state_bytes,
            execution.policy_tape_capacity,
            execution.policy_mutation_tape_width,
        },
        tensors.source<float>(kMlpPack, cell),
        !defer_mlp_policy_rows,
        defer_mlp_policy_rows,
        config.hapb_host_authoritative_policy
    );
    if (
        statuses[kPolicyStatus] == policy::kStatusOk
        && (
            turnover_policy_cursor[0] < 0
            || turnover_policy_cursor[0] > execution.policy_tape_capacity
            || policy_cursor_base
                > std::numeric_limits<int64_t>::max()
                    - turnover_policy_cursor[0]
        )
    ) {
        statuses[kPolicyStatus] = policy::kStatusTapeExhausted;
        }
    }
    if (statuses[kPolicyStatus] == policy::kStatusOk) {
        turnover_policy_cursor[0] += policy_cursor_base;
    }
    }

    if (
        statuses[kDriveStatus] != drives::kStatusOk
        || statuses[kChemistryStatus] != drives::kStatusOk
    ) {
        statuses[kAggregateStatus] = 3;
        counters[kAggregateFinalLiveCount] = counters[kPlanSurvivorCount];
        return;
    }
    if (statuses[kPolicyStatus] != policy::kStatusOk) {
        statuses[kAggregateStatus] = 4;
        counters[kAggregateFinalLiveCount] = counters[kPlanSurvivorCount];
        return;
    }
    if (
        qualification_fault_stage
        == flattened_birth_scratch_v2::
            kGenomeUpgradeV2FaultTurnoverAfterPhenotypeProjection
    ) {
        statuses[kAggregateStatus] = detail::turnover_fault_status_v1(
            qualification_fault_stage
        );
        counters[kAggregateFinalLiveCount] = counters[kPlanSurvivorCount];
        return;
    }

    detail::reset_newborn_rows(tensors, scratch, cell, execution);
    detail::normalize_inactive_rows(tensors, cell, execution);
    const int64_t committed = counters[kIdentityCommittedCount];
    const int64_t final_live = counters[kAggregateFinalLiveCount];
    const bool terminal_extinction =
        counters[kPlanRankedCount] == 0
        && counters[kPlanSurvivorCount] == 0
        && counters[kPlanParentPoolCount] == 0
        && counters[kPlanRefillCount] == 0
        && committed == 0
        && final_live == 0;
    const bool terminal_low_population =
        execution.terminal_low_population_no_reinit
        && counters[kPlanSurvivorCount] > 0
        && counters[kPlanSurvivorCount] < 2
        && counters[kPlanRefillCount] == 0
        && committed == 0
        && final_live == counters[kPlanSurvivorCount];
    if (
        committed != counters[kPlanRefillCount]
        || (
            final_live != config.population_size
            && !terminal_extinction
            && !terminal_low_population
        )
    ) {
        statuses[kAggregateStatus] = 7;
        counters[kAggregateCommittedCount] = 0;
        return;
    }

    int64_t* const active_ids = tensors.shadow<int64_t>(
        kActiveAgentIds, cell
    );
    int32_t* const active_slots = tensors.shadow<int32_t>(
        kActiveAgentSlots, cell
    );
    const int64_t* const slot_to_agent_id = tensors.shadow<int64_t>(
        kSlotToAgentId, cell
    );
    const bool* const alive_mask = tensors.shadow<bool>(kAliveMask, cell);
    const int64_t* const sorted_ids = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kSortedAliveAgentIds
    );
    for (int64_t row = 0; row < execution.capacity; ++row) {
        const int64_t agent_id = sorted_ids[row];
        active_ids[row] = agent_id;
        active_slots[row] = -1;
        if (row >= final_live) {
            continue;
        }
        if (!active_identity::agent_id_fits_grid_v1(agent_id)) {
            statuses[kAggregateStatus] = 8;
            counters[kAggregateCommittedCount] = 0;
            return;
        }
        const active_identity::ResolveResultV1 resolved =
            active_identity::resolve_resident_agent_slot_v1(
                slot_to_agent_id,
                alive_mask,
                execution.capacity,
                agent_id
            );
        if (resolved.status != active_identity::kResolveFound) {
            statuses[kAggregateStatus] = 8;
            counters[kAggregateCommittedCount] = 0;
            return;
        }
        active_slots[row] = static_cast<int32_t>(resolved.slot);
    }
    const active_identity::ActiveIdentityPairsV1 final_active_pairs{
        active_ids,
        active_slots,
        slot_to_agent_id,
        alive_mask,
        final_live,
        execution.capacity,
        execution.capacity,
    };
    if (!active_identity::valid_active_identity_pairs_v1(final_active_pairs)) {
        statuses[kAggregateStatus] = 8;
        counters[kAggregateCommittedCount] = 0;
        return;
    }
    tensors.shadow<int64_t>(kActiveCountState, cell)[0] = final_live;
    counters[kAggregateCommittedCount] = committed;
    if (
        qualification_fault_stage
        == flattened_birth_scratch_v2::
            kGenomeUpgradeV2FaultTurnoverBeforeCommit
    ) {
        statuses[kAggregateStatus] = detail::turnover_fault_status_v1(
            qualification_fault_stage
        );
        counters[kAggregateCommittedCount] = 0;
        return;
    }
    statuses[kAggregateStatus] = 0;
    at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kAggregateNextGeneration
    )[0] = execution.generation + 1;
    at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kAggregateNextTick
    )[0] = 0;
}

static __global__ void flattened_turnover_publish_kernel_v1(
    const int64_t cells,
    const FlatTurnoverMutableTensorTable tensors,
    const FlatTurnoverScratch scratch,
    const FlatTurnoverOutputs outputs,
    const FlatTurnoverConfig config,
    const FlatTurnoverExecutionConfig execution,
    const FlatOrdinaryBirthState birth_state,
    const modular::FlatModularContextV1 modular_context
) {
    const int64_t cell = static_cast<int64_t>(blockIdx.x);
    if (
        cell >= cells
        || (
            config.prerequisite_status != nullptr
            && config.prerequisite_status[cell] != 0
        )
    ) {
        return;
    }
    const int64_t lane = static_cast<int64_t>(threadIdx.x);
    if (!turnover_cell_enabled_v1(config, cell)) {
        if (lane == 0) {
            // The dormant transaction performs no mechanic, event, RNG, or
            // inheritance-sidecar writes. Its settlement output still mirrors
            // authoritative resident identity so mixed-mode checkpoint
            // projection does not mistake "no turnover" for "no live agents".
            int64_t* const output_counts =
                outputs.counts + cell * kTurnoverOutputCount;
            for (
                int32_t index = 0;
                index < kTurnoverOutputCount;
                ++index
            ) {
                output_counts[index] = 0;
            }
            output_counts[kOutputFinalLiveCount] =
                tensors.source<int64_t>(kActiveCountState, cell)[0];
            outputs.status[cell] = 0;
            outputs.next_generation[cell] = execution.generation + 1;
            outputs.next_tick[cell] = 0;
        }
        return;
    }
    const int64_t lanes = static_cast<int64_t>(blockDim.x);
    const int32_t aggregate_status = at<int32_t>(
        scratch, cell, FlatTurnoverWorkLayout::kPhaseStatuses
    )[kAggregateStatus];
    const bool success = aggregate_status == 0;
    if (success) {
        for (int32_t index = 0; index < kTurnoverMutableTensorCount; ++index) {
            if (index == kMlpPack) {
                continue;
            }
            tensors.copy_shadow_to_source(
                static_cast<TurnoverMutableTensorIndex>(index), cell, lane, lanes
            );
        }
        __syncthreads();
        if (
            birth_state.inheritance_mode != nullptr
            && birth_state.inheritance_mode[cell] == 2
        ) {
            const int64_t allele_cell_stride =
                diploid::kAllelePlaneCount * execution.capacity;
            const int64_t allele_cell_offset = cell * allele_cell_stride;
            for (
                int64_t index = lane;
                index < allele_cell_stride;
                index += lanes
            ) {
                birth_state.diploid_allele_storage[
                    allele_cell_offset + index
                ] = birth_state.diploid_allele_shadow_storage[
                    allele_cell_offset + index
                ];
            }
            const int64_t cell_capacity_offset =
                cell * execution.capacity;
            for (
                int64_t slot = lane;
                slot < execution.capacity;
                slot += lanes
            ) {
                birth_state.genome_resource_handling_axis[
                    cell_capacity_offset + slot
                ] = birth_state.genome_resource_handling_axis_shadow[
                    cell_capacity_offset + slot
                ];
            }
        }
        if (
            birth_state.genome_upgrade_v2_enabled != nullptr
            && birth_state.genome_upgrade_v2_enabled[cell]
        ) {
            namespace genome_v2 = eiel::genome_upgrade_v2_q10_v1;
            const int64_t allele_cell_stride =
                2 * genome_v2::kProductCount * genome_v2::kParameterCount
                * execution.capacity;
            const int64_t modifier_cell_stride =
                2 * genome_v2::kProductCount * execution.capacity;
            const int64_t product_cell_stride =
                genome_v2::kProductCount * execution.capacity;
            const int64_t phenotype_cell_stride =
                genome_v2::kPhenotypeCount * execution.capacity;
            const int64_t allele_offset = cell * allele_cell_stride;
            const int64_t modifier_offset = cell * modifier_cell_stride;
            const int64_t product_offset = cell * product_cell_stride;
            const int64_t phenotype_offset = cell * phenotype_cell_stride;
            for (
                int64_t index = lane;
                index < allele_cell_stride;
                index += lanes
            ) {
                birth_state.genome_upgrade_v2_alleles_q10[
                    allele_offset + index
                ] = birth_state.genome_upgrade_v2_alleles_q10_shadow[
                    allele_offset + index
                ];
            }
            for (
                int64_t index = lane;
                index < modifier_cell_stride;
                index += lanes
            ) {
                birth_state.genome_upgrade_v2_modifiers_q10[
                    modifier_offset + index
                ] = birth_state.genome_upgrade_v2_modifiers_q10_shadow[
                    modifier_offset + index
                ];
            }
            for (
                int64_t index = lane;
                index < product_cell_stride;
                index += lanes
            ) {
                birth_state.genome_upgrade_v2_products_q10[
                    product_offset + index
                ] = birth_state.genome_upgrade_v2_products_q10_shadow[
                    product_offset + index
                ];
            }
            for (
                int64_t index = lane;
                index < phenotype_cell_stride;
                index += lanes
            ) {
                birth_state.genome_upgrade_v2_phenotypes_q10[
                    phenotype_offset + index
                ] = birth_state.genome_upgrade_v2_phenotypes_q10_shadow[
                    phenotype_offset + index
                ];
            }
            const int64_t cell_capacity_offset =
                cell * execution.capacity;
            for (
                int64_t slot = lane;
                slot < execution.capacity;
                slot += lanes
            ) {
                const int64_t index = cell_capacity_offset + slot;
                birth_state.genome_upgrade_v2_expression_cost_q10[index] =
                    birth_state
                        .genome_upgrade_v2_expression_cost_q10_shadow[index];
                birth_state.genome_upgrade_v2_active_product_count[index] =
                    birth_state
                        .genome_upgrade_v2_active_product_count_shadow[index];
                birth_state.genome_upgrade_v2_saturated_product_count[index] =
                    birth_state
                        .genome_upgrade_v2_saturated_product_count_shadow[index];
            }
        }
        if (modular_context.enabled && modular_context.lifecycle_enabled) {
            modular::device::copy_lifecycle_cell_shadow_to_source_v2(
                modular_context, cell, lane, lanes
            );
        }
        const int64_t committed = at<int64_t>(
            scratch, cell, FlatTurnoverWorkLayout::kScalarCounters
        )[kAggregateCommittedCount];
        const int64_t* const child_slots = at<int64_t>(
            scratch, cell, FlatTurnoverWorkLayout::kCommittedChildSlots
        );
        const bool defer_mlp_policy_rows =
            tensors.entry(kMlpPack).shadow_bytes_per_cell == 0;
        const int64_t mlp_row_bytes = execution.mlp_row_width
            * static_cast<int64_t>(sizeof(float));
        uint8_t* const source_mlp = tensors.source_bytes(kMlpPack, cell);
        const uint8_t* const shadow_mlp = defer_mlp_policy_rows
            ? nullptr
            : tensors.shadow_bytes(kMlpPack, cell);
        for (int64_t birth = 0; birth < committed; ++birth) {
            const int64_t source_row_offset =
                child_slots[birth] * mlp_row_bytes;
            const int64_t shadow_row_offset = birth * mlp_row_bytes;
            if (!defer_mlp_policy_rows) {
                for (
                    int64_t byte = lane;
                    byte < mlp_row_bytes;
                    byte += lanes
                ) {
                    source_mlp[source_row_offset + byte] =
                        shadow_mlp[shadow_row_offset + byte];
                }
            }
            if (
                !execution.immutable_shared_gru_policy
                && execution.gru_pack != nullptr
                && execution.gru_row_width > 0
            ) {
                const int64_t donor_slot = at<int64_t>(
                    scratch, cell, FlatTurnoverWorkLayout::kRankedSlots
                )[birth];
                if (donor_slot >= 0) {
                    const int64_t source_row = (
                        cell * execution.capacity + donor_slot
                    ) * execution.gru_row_width;
                    const int64_t destination_row = (
                        cell * execution.capacity + child_slots[birth]
                    ) * execution.gru_row_width;
                    for (
                        int64_t column = lane;
                        column < execution.gru_row_width;
                        column += lanes
                    ) {
                        execution.gru_pack[destination_row + column] =
                            execution.gru_pack[source_row + column];
                    }
                }
            }
        }
        if (
            defer_mlp_policy_rows
            && !config.hapb_host_authoritative_policy
            && committed > 0
        ) {
            bool materialized = false;
            if (execution.rng_semantics_version == 2) {
                const uint64_t absolute_tick = static_cast<uint64_t>(
                    execution.generation * execution.ticks_per_generation
                    + execution.start_tick + execution.ticks
                );
                materialized =
                    policy::materialize_fixed_mlp_turnover_policy_rows_rngv2_v1(
                        tensors.source<float>(kMlpPack, cell),
                        tensors.source<float>(kMlpPack, cell),
                        tensors.shadow<int64_t>(kSlotToAgentId, cell),
                        {
                            at<int64_t>(
                                scratch,
                                cell,
                                FlatTurnoverWorkLayout::kRefillParentSlots
                            ),
                            at<int64_t>(
                                scratch,
                                cell,
                                FlatTurnoverWorkLayout::kRefillMateSlots
                            ),
                            at<int64_t>(
                                scratch,
                                cell,
                                FlatTurnoverWorkLayout::kCommittedChildSlots
                            ),
                            at<int64_t>(
                                scratch,
                                cell,
                                FlatTurnoverWorkLayout::kScalarCounters
                            ) + kAggregateCommittedCount,
                        },
                        {
                            execution.capacity,
                            execution.capacity,
                            execution.mlp_row_width,
                            execution.hmem_width,
                            execution.torch_state_bytes,
                            execution.policy_tape_capacity,
                            execution.policy_mutation_tape_width,
                        },
                        execution.rng_namespace_key,
                        turnover_world_uid_for_cell_v1(execution, cell),
                        execution.rng_semantic_epoch,
                        absolute_tick,
                        lane,
                        lanes
                    );
            } else {
                const int64_t policy_mutation_cell_stride =
                    execution.policy_tape_capacity
                    * execution.policy_mutation_tape_width;
                const int64_t policy_crossover_cell_stride =
                    execution.policy_tape_capacity
                    * ((execution.mlp_row_width + 7) / 8);
                const int64_t policy_cursor_base =
                    birth_state.policy_birth_tape_cursor_base[cell];
                const int64_t local_after_cursor =
                    tensors.shadow<int64_t>(kPolicyBirthTapeCursor, cell)[0]
                    - policy_cursor_base;
                materialized = policy::materialize_fixed_mlp_turnover_policy_rows_v1(
                    tensors.source<float>(kMlpPack, cell),
                    tensors.source<float>(kMlpPack, cell),
                    {
                        at<int64_t>(
                            scratch, cell, FlatTurnoverWorkLayout::kRefillParentSlots
                        ),
                        at<int64_t>(
                            scratch, cell, FlatTurnoverWorkLayout::kRefillMateSlots
                        ),
                        at<int64_t>(
                            scratch,
                            cell,
                            FlatTurnoverWorkLayout::kCommittedChildSlots
                        ),
                        at<int64_t>(
                            scratch,
                            cell,
                            FlatTurnoverWorkLayout::kScalarCounters
                        ) + kAggregateCommittedCount,
                    },
                    {
                        detail::offset_or_null(
                            birth_state.policy_birth_delta_tape,
                            cell * policy_mutation_cell_stride
                        ),
                        detail::offset_or_null(
                            birth_state.policy_birth_mutation_mask_tape,
                            cell * policy_mutation_cell_stride
                        ),
                        detail::offset_or_null(
                            birth_state.policy_birth_crossover_mask_packed_tape,
                            cell * policy_crossover_cell_stride
                        ),
                        nullptr,
                        nullptr,
                        nullptr,
                    },
                    {
                        execution.capacity,
                        execution.capacity,
                        execution.mlp_row_width,
                        execution.hmem_width,
                        execution.torch_state_bytes,
                        execution.policy_tape_capacity,
                        execution.policy_mutation_tape_width,
                    },
                    local_after_cursor - committed,
                    lane,
                    lanes,
                    false
                );
            }
            if (!materialized && lane == 0) {
                outputs.status[cell] = 92;
            }
        }
    }
    __syncthreads();
    if (lane != 0) {
        return;
    }

    const int64_t output_offset = cell * execution.capacity;
    const int64_t* const ranked = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kRankedAgentIds
    );
    const int64_t* const culled = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kCulledAgentIds
    );
    const int64_t* const parents = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kParentPoolAgentIds
    );
    const int64_t* const refill_parents = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kCommittedParentAgentIds
    );
    const int64_t* const refill_mates = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kCommittedMateAgentIds
    );
    const int64_t* const refill_parent_slots = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kRefillParentSlots
    );
    const int64_t* const refill_mate_slots = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kRefillMateSlots
    );
    const int64_t* const child_ids = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kCommittedChildAgentIds
    );
    const int64_t* const child_slots = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kCommittedChildSlots
    );
    for (int64_t row = 0; row < execution.capacity; ++row) {
        outputs.ranked_agent_ids[output_offset + row] = ranked[row];
        outputs.culled_agent_ids[output_offset + row] = culled[row];
        outputs.parent_pool_agent_ids[output_offset + row] = parents[row];
        outputs.refill_parent_agent_ids[output_offset + row] =
            refill_parents[row];
        outputs.refill_mate_agent_ids[output_offset + row] = refill_mates[row];
        outputs.child_agent_ids[output_offset + row] = child_ids[row];
        outputs.child_slots[output_offset + row] = child_slots[row];
    }

    const int64_t* const counters = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kScalarCounters
    );
    const bool causal_capture_enabled =
        success
        && outputs.genome_v2_causal_capture_enabled != nullptr
        && outputs.genome_v2_causal_capture_enabled[cell]
        && birth_state.genome_upgrade_v2_enabled != nullptr
        && birth_state.genome_upgrade_v2_enabled[cell];
    if (causal_capture_enabled) {
        const int64_t record_start =
            outputs.genome_v2_causal_record_count != nullptr
            ? outputs.genome_v2_causal_record_count[cell]
            : 0;
        const bool storage_valid =
            outputs.genome_v2_causal_record_rows != nullptr
            && outputs.genome_v2_causal_record_valid != nullptr
            && outputs.genome_v2_causal_record_count != nullptr
            && outputs.genome_v2_causal_record_dropped_count != nullptr
            && record_start >= 0
            && record_start <= outputs.genome_v2_causal_record_capacity;
        const int64_t available = storage_valid
            ? outputs.genome_v2_causal_record_capacity - record_start
            : 0;
        const int64_t committed = counters[kAggregateCommittedCount];
        const int64_t captured =
            committed < available ? committed : available;
        namespace genome_v2 = eiel::genome_upgrade_v2_q10_v1;
        const int64_t allele_stride =
            genome_v2::kHomologCount * genome_v2::kProductCount
            * genome_v2::kParameterCount * execution.capacity;
        const int64_t modifier_stride =
            genome_v2::kHomologCount * genome_v2::kProductCount
            * execution.capacity;
        const int64_t product_stride =
            genome_v2::kProductCount * execution.capacity;
        const int64_t phenotype_stride =
            genome_v2::kPhenotypeCount * execution.capacity;
        const int64_t cell_capacity_offset = cell * execution.capacity;
        const genome_v2::ResidentPlanesV1 source_planes{
            birth_state.genome_upgrade_v2_alleles_q10
                + cell * allele_stride,
            birth_state.genome_upgrade_v2_modifiers_q10
                + cell * modifier_stride,
            birth_state.genome_upgrade_v2_products_q10
                + cell * product_stride,
            birth_state.genome_upgrade_v2_phenotypes_q10
                + cell * phenotype_stride,
            birth_state.genome_upgrade_v2_expression_cost_q10
                + cell_capacity_offset,
            birth_state.genome_upgrade_v2_active_product_count
                + cell_capacity_offset,
            birth_state.genome_upgrade_v2_saturated_product_count
                + cell_capacity_offset,
            execution.capacity,
        };
        for (int64_t birth = 0; birth < captured; ++birth) {
            const int64_t child_slot = child_slots[birth];
            detail::write_turnover_causal_record_v1(
                outputs,
                record_start + birth,
                cell,
                execution,
                child_ids[birth],
                refill_parents[birth],
                refill_mates[birth],
                child_slot,
                refill_parent_slots[birth],
                refill_mate_slots[birth],
                tensors.source<int32_t>(
                    kLineageFounderId, cell
                )[child_slot],
                tensors.source<int32_t>(
                    kLineageDepth, cell
                )[child_slot],
                source_planes
            );
        }
        if (storage_valid) {
            outputs.genome_v2_causal_record_count[cell] =
                record_start + captured;
            outputs.genome_v2_causal_record_dropped_count[cell] +=
                committed - captured;
        }
    }
    int64_t* const output_counts = outputs.counts
        + cell * kTurnoverOutputCount;
    output_counts[kOutputRankedCount] = counters[kPlanRankedCount];
    output_counts[kOutputCulledCount] = counters[kPlanCulledCount];
    output_counts[kOutputParentPoolCount] = counters[kPlanParentPoolCount];
    output_counts[kOutputRefillCount] = counters[kPlanRefillCount];
    output_counts[kOutputReleasedCount] = counters[kCullReleasedCount];
    output_counts[kOutputRefillPlannedCount] = counters[kRefillPlannedCount];
    output_counts[kOutputChildCommittedCount] =
        counters[kAggregateCommittedCount];
    output_counts[kOutputFinalLiveCount] = counters[kAggregateFinalLiveCount];
    int32_t publish_status = aggregate_status;
    if (!success) {
        const int32_t* const phase_statuses = at<int32_t>(
            scratch, cell, FlatTurnoverWorkLayout::kPhaseStatuses
        );
        publish_status = encode_turnover_failure_diagnostic_v1(
            kTurnoverAggregateOnlyDiagnosticPhaseV1,
            aggregate_status
        );
        for (int32_t phase = kPlanStatus; phase < kAggregateStatus; ++phase) {
            if (phase_statuses[phase] != 0) {
                publish_status = encode_turnover_failure_diagnostic_v1(
                    phase + 1,
                    phase_statuses[phase]
                );
                break;
            }
        }
    }
    const auto& exposure =
        config.cross_path_control.reproduction_exposure;
    const int8_t exposure_path =
        config.cross_path_control.reproduction_path_by_cell != nullptr
        ? config.cross_path_control.reproduction_path_by_cell[cell]
        : resident_reproduction_exposure_v1::kNoPath;
    if (success) {
        const int64_t committed = counters[kAggregateCommittedCount];
        const int64_t staged =
            resident_reproduction_exposure_v1::staged_attempt_count_v1(
                exposure,
                cell,
                exposure_path,
                execution.generation,
                execution.ticks
            );
        if (
            resident_reproduction_exposure_v1::enabled_v1(exposure)
            && staged != committed
        ) {
            publish_status =
                resident_reproduction_exposure_v1::
                    kStatusAuthoritativeCommitMismatch;
            (void)resident_reproduction_exposure_v1::
                abort_staged_attempts_v1(
                    exposure,
                    cell,
                    exposure_path,
                    execution.generation,
                    execution.ticks
                );
        }
        const bool linked =
            config.cross_path_control.inheritance_mode_by_cell != nullptr
            && config.cross_path_control.inheritance_mode_by_cell[cell]
                == cross_path::kInheritanceModeLinkedParentCopy;
        const int64_t inheritance_base =
            cell * config.cross_path_control.inheritance_rows_per_cell;
        for (
            int64_t birth = 0;
            publish_status == aggregate_status && birth < committed;
            ++birth
        ) {
            if (
                !resident_reproduction_exposure_v1::
                    rebind_staged_attempt_selected_parents_v1(
                        exposure,
                        cell,
                        exposure_path,
                        execution.generation,
                        execution.ticks,
                        birth,
                        refill_parents[birth],
                        refill_mates[birth]
                    )
                || !resident_reproduction_exposure_v1::
                    finalize_staged_attempt_v1(
                        exposure,
                        cell,
                        exposure_path,
                        execution.generation,
                        execution.ticks,
                        birth,
                        refill_parents[birth],
                        refill_mates[birth],
                        child_ids[birth],
                        linked
                            ? config.cross_path_control.donor_ordinals
                                + (inheritance_base + birth)
                                    * inheritance_linkage_v1::
                                        kLinkageGroupCount
                            : nullptr,
                        linked
                            ? inheritance_linkage_v1::kLinkageGroupCount
                            : 0
                    )
            ) {
                publish_status =
                    resident_reproduction_exposure_v1::
                        kStatusAuthoritativeCommitMismatch;
                break;
            }
        }
    } else {
        (void)resident_reproduction_exposure_v1::abort_staged_attempts_v1(
            exposure,
            cell,
            exposure_path,
            execution.generation,
            execution.ticks
        );
    }
    if (
        success
        && publish_status == aggregate_status
        && !resident_reproduction_exposure_v1::record_path_rng_commit_v1(
            exposure,
            cell,
            exposure_path,
            counters[kAggregateCommittedCount]
        )
    ) {
        publish_status =
            resident_reproduction_exposure_v1::
                kStatusAuthoritativeCommitMismatch;
    }
    if (
        success
        && publish_status == aggregate_status
        && detail::qualification_fault_stage_v1(config, cell)
            == flattened_birth_scratch_v2::
                kGenomeUpgradeV2FaultTurnoverAfterCommitBeforeCheckpoint
    ) {
        publish_status = detail::turnover_fault_status_v1(
            flattened_birth_scratch_v2::
                kGenomeUpgradeV2FaultTurnoverAfterCommitBeforeCheckpoint
        );
    }
    const int32_t prior_status = outputs.status[cell];
    outputs.status[cell] = prior_status != 0
        ? prior_status
        : (
            publish_status == detail::kInternalDisabled
                ? 0
                : publish_status
        );
    outputs.next_generation[cell] = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kAggregateNextGeneration
    )[0];
    outputs.next_tick[cell] = at<int64_t>(
        scratch, cell, FlatTurnoverWorkLayout::kAggregateNextTick
    )[0];
}

inline TurnoverLaunchReceipt run_flattened_turnover_v1(
    const cudaStream_t stream,
    const int64_t cells,
    const FlatTurnoverMutableTensorTable tensors,
    const FlatTurnoverScratch scratch,
    const FlatTurnoverOutputs outputs,
    const FlatTurnoverConfig config,
    const FlatTurnoverExecutionConfig execution,
    const FlatOrdinaryBirthState birth_state,
    const FlatOrdinaryBirthConfig birth_config,
    const modular::FlatModularContextV1 modular_context,
    const std::function<void()>& hapb_host_policy_commit = {},
    int64_t* const host_synchronizations_inside_loop = nullptr
) {
    if (!detail::valid_launch_boundary(
            cells,
            tensors,
            scratch,
            outputs,
            config,
            execution,
            birth_state,
            birth_config
        )) {
        return {0, 0, 0, 0, static_cast<int32_t>(cudaErrorInvalidValue)};
    }

    TurnoverLaunchReceipt receipt{1, 1, 1, 1, 0};
    flattened_turnover_shadow_plan_kernel_v1<<<
        static_cast<unsigned int>(cells), kTurnoverThreadsPerCell, 0, stream
    >>>(cells, tensors, scratch, config, execution, modular_context);
    detail::retain_first_error(&receipt.cuda_status);

    flattened_turnover_cull_refill_plan_kernel_v1<<<
        static_cast<unsigned int>(cells), kTurnoverThreadsPerCell, 0, stream
    >>>(cells, tensors, scratch, config, execution, modular_context);
    detail::retain_first_error(&receipt.cuda_status);

    flattened_turnover_commit_kernel_v1<<<
        static_cast<unsigned int>(cells), kTurnoverThreadsPerCell, 0, stream
    >>>(
        cells, tensors, scratch, config, execution, birth_state, modular_context
    );
    detail::retain_first_error(&receipt.cuda_status);

    // HAPB must observe the committed child slots and the post-turnover policy
    // cursor.  The cull/refill plan does not assign child slots; that happens
    // inside the native commit kernel above.  Synchronize at this exact
    // transaction boundary, before publication, so host policy bytes can be
    // committed without letting any later native publication read a stale
    // policy authority.
    if (hapb_host_policy_commit && receipt.cuda_status == cudaSuccess) {
        const cudaError_t host_boundary_status = cudaStreamSynchronize(stream);
        if (host_boundary_status != cudaSuccess) {
            receipt.cuda_status = static_cast<int32_t>(host_boundary_status);
            return receipt;
        }
        if (host_synchronizations_inside_loop != nullptr) {
            *host_synchronizations_inside_loop += 1;
        }
        hapb_host_policy_commit();
        if (host_synchronizations_inside_loop != nullptr) {
            *host_synchronizations_inside_loop += 1;
        }
    }

    flattened_turnover_publish_kernel_v1<<<
        static_cast<unsigned int>(cells), kTurnoverThreadsPerCell, 0, stream
    >>>(
        cells,
        tensors,
        scratch,
        outputs,
        config,
        execution,
        birth_state,
        modular_context
    );
    detail::retain_first_error(&receipt.cuda_status);
    return receipt;
}

KernelResourceReceipt query_flattened_turnover_shadow_plan_resources_v1() {
    return detail::query_kernel_resources(
        flattened_turnover_shadow_plan_kernel_v1
    );
}

KernelResourceReceipt
query_flattened_turnover_cull_refill_plan_resources_v1() {
    return detail::query_kernel_resources(
        flattened_turnover_cull_refill_plan_kernel_v1
    );
}

KernelResourceReceipt query_flattened_turnover_commit_resources_v1() {
    return detail::query_kernel_resources(flattened_turnover_commit_kernel_v1);
}

KernelResourceReceipt query_flattened_turnover_publish_resources_v1() {
    return detail::query_kernel_resources(flattened_turnover_publish_kernel_v1);
}

}  // namespace eiel::cuda_architecture_b::flattened_turnover_v1
