#include "cuda_architecture_b_flattened_controller_v1.hpp"
#include "cuda_architecture_b_flattened_birth_drives_v1.cuh"
#include "cuda_architecture_b_flattened_birth_events_v2.cuh"
#include "cuda_architecture_b_flattened_birth_identity_v2.cuh"
#include "cuda_architecture_b_flattened_mate_birth_scalar_genome_v3.cuh"
#include "cuda_architecture_b_flattened_birth_policy_v2.cuh"
#include "cuda_architecture_b_flattened_birth_scratch_v2.cuh"
#include "cuda_architecture_b_flattened_birth_scalar_genome_v2.cuh"
#include "cuda_architecture_b_flattened_turnover_v1_kernels.cuh"
#include "cuda_architecture_b_active_identity_pairs_v1.cuh"
#include "cuda_architecture_b_persistent_active_mechanics.cuh"
#include "cuda_architecture_b_persistent_birth_action_history.cuh"
#include "cuda_architecture_b_persistent_birth_drives.cuh"
#include "cuda_architecture_b_persistent_birth_events.cuh"
#include "cuda_architecture_b_persistent_birth_identity.cuh"
#include "cuda_architecture_b_persistent_birth_plan.cuh"
#include "cuda_architecture_b_persistent_mate_birth_plan_v1.cuh"
#include "cuda_architecture_b_persistent_birth_policy.cuh"
#include "cuda_architecture_b_persistent_birth_scalar_genome.cuh"
#include "cuda_architecture_b_persistent_chemistry.cuh"
#include "cuda_architecture_b_persistent_food_spawn.cuh"
#include "cuda_architecture_b_persistent_lifecycle_events.cuh"
#include "cuda_architecture_b_persistent_newborn_state.cuh"
#include "cuda_architecture_b_persistent_stochastic_policy.cuh"
#include "cuda_diploid_scalar_genome_v1.cuh"
#include "cuda_canonical_hereditary_digest_v1.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef EIEL_CUDA_STAGE3_PARALLEL_PATCH_POLICY_ROWS
#define EIEL_CUDA_STAGE3_PARALLEL_PATCH_POLICY_ROWS 0
#endif

#ifndef EIEL_OPTION_D_M9_BIRTH_PREPARE_COMMIT_FUSED_ENABLED
#define EIEL_OPTION_D_M9_BIRTH_PREPARE_COMMIT_FUSED_ENABLED 0
#endif

namespace eiel::cuda_architecture_b::flattened_controller_v1 {
namespace {

namespace torch_rng_pack =
    eiel::cuda_architecture_b::torch_rng_state_pack_v1;

constexpr int64_t kMaxHiddenSize = 64;
constexpr int64_t kMaxObservationSize = 74;
constexpr int64_t kPatchObservationChannels = 8;
constexpr int64_t kPatchRadiusOneObservationSize = 74;
constexpr int64_t kMaxActionCount = 16;
constexpr int32_t kEmpty = 0;
constexpr int32_t kFood = 1;
constexpr int32_t kGatherAction = 4;
constexpr int32_t kRestAction = 5;
constexpr int32_t kAttackAction = 6;
constexpr int32_t kLifecycleOk = 0;
constexpr int32_t kLifecycleInvalidPendingCount = 1;
constexpr int32_t kLifecyclePendingCapacityExceeded = 2;

namespace reporting = compact_summary_reporting_authority_v1;
namespace causal_output = genome_v2_causal_output_v1;
namespace active_identity = active_identity_pairs_v1;

__device__ __forceinline__ void write_genome_v2_causal_record_v1(
    int64_t* const rows,
    bool* const valid,
    const int64_t capacity,
    const int64_t row_index,
    const int64_t event_kind,
    const int64_t cell,
    const int64_t generation,
    const int64_t tick,
    const int64_t event_sequence,
    const bool event_sequence_valid,
    const int64_t child_id,
    const int64_t parent_id,
    const int64_t mate_parent_id,
    const int64_t destination_slot,
    const int64_t parent_slot,
    const int64_t mate_parent_slot,
    const int64_t lineage_founder_id,
    const int64_t lineage_depth,
    const eiel::genome_upgrade_v2_q10_v1::ResidentPlanesV1& planes
) {
    namespace genome_v2 = eiel::genome_upgrade_v2_q10_v1;
    if (
        rows == nullptr || valid == nullptr || row_index < 0
        || row_index >= capacity
    ) {
        return;
    }
    const int64_t physical_row = cell * capacity + row_index;
    int64_t* const row =
        rows + physical_row * causal_output::kRecordWidth;
    row[causal_output::kRecordOrdinal] = row_index;
    row[causal_output::kEventKind] = event_kind;
    row[causal_output::kCellOrdinal] = cell;
    row[causal_output::kGeneration] = generation;
    row[causal_output::kTick] = tick;
    row[causal_output::kEventSequence] = event_sequence_valid
        ? event_sequence : causal_output::kUnavailableValue;
    row[causal_output::kEventSequenceValid] =
        event_sequence_valid ? 1 : 0;
    row[causal_output::kChildAgentId] = child_id;
    row[causal_output::kParentAgentId] = parent_id;
    row[causal_output::kMateParentAgentId] = mate_parent_id;
    row[causal_output::kDestinationSlot] = destination_slot;
    row[causal_output::kParentSlot] = parent_slot;
    row[causal_output::kMateParentSlot] = mate_parent_slot;
    row[causal_output::kLineageFounderId] = lineage_founder_id;
    row[causal_output::kLineageDepth] = lineage_depth;
    genome_v2::GenotypeV1 genotype{};
    genome_v2::load_genotype_v1(planes, destination_slot, &genotype);
    int64_t allele_column = causal_output::kAllelesQ10;
    for (int32_t homolog = 0; homolog < genome_v2::kHomologCount; ++homolog) {
        for (int32_t product = 0; product < genome_v2::kProductCount; ++product) {
            for (
                int32_t parameter = 0;
                parameter < genome_v2::kParameterCount;
                ++parameter
            ) {
                row[allele_column++] =
                    genotype.alleles[homolog][product][parameter];
            }
        }
    }
    int64_t modifier_column = causal_output::kModifiersQ10;
    for (int32_t homolog = 0; homolog < genome_v2::kHomologCount; ++homolog) {
        for (int32_t product = 0; product < genome_v2::kProductCount; ++product) {
            row[modifier_column++] = genotype.modifiers[homolog][product];
        }
    }
    genome_v2::ExpressionV1 expression{};
    for (int32_t product = 0; product < genome_v2::kProductCount; ++product) {
        expression.products[product] = planes.products[
            static_cast<int64_t>(product) * planes.slot_count
            + destination_slot
        ];
        row[causal_output::kProductsQ10 + product] =
            expression.products[product];
    }
    // The fixed regulator executes inside express_v1 but does not expose its
    // four node values or its isolated cost. Keep both unavailable rather than
    // inventing values from the aggregate expression result.
    row[causal_output::kRegulatorValuesValid] = 0;
    for (
        int32_t phenotype = 0;
        phenotype < genome_v2::kPhenotypeCount;
        ++phenotype
    ) {
        expression.phenotypes[phenotype] = planes.phenotypes[
            static_cast<int64_t>(phenotype) * planes.slot_count
            + destination_slot
        ];
        row[causal_output::kPhenotypesQ10 + phenotype] =
            expression.phenotypes[phenotype];
    }
    // Ordinary children are committed under the frozen neutral context.
    row[causal_output::kContextValid] = 1;
    for (int32_t context = 0; context < genome_v2::kContextCount; ++context) {
        row[causal_output::kContextQ10 + context] = 0;
    }
    expression.expression_cost =
        planes.expression_cost[destination_slot];
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
    valid[physical_row] = true;
}

__device__ __forceinline__ bool physical_cursor_from_absolute_v1(
    const int64_t absolute_cursor,
    const int64_t sequence_base,
    const int64_t physical_capacity,
    int64_t* const physical_cursor
) {
    if (
        physical_cursor == nullptr
        || sequence_base < 0
        || physical_capacity < 0
        || absolute_cursor < sequence_base
    ) {
        return false;
    }
    const int64_t candidate = absolute_cursor - sequence_base;
    if (candidate > physical_capacity) {
        return false;
    }
    physical_cursor[0] = candidate;
    return true;
}

__device__ __forceinline__ bool absolute_cursor_from_physical_v1(
    const int64_t physical_cursor,
    const int64_t sequence_base,
    const int64_t physical_capacity,
    int64_t* const absolute_cursor
) {
    if (
        absolute_cursor == nullptr
        || sequence_base < 0
        || physical_cursor < 0
        || physical_cursor > physical_capacity
        || sequence_base > INT64_MAX - physical_cursor
    ) {
        return false;
    }
    absolute_cursor[0] = sequence_base + physical_cursor;
    return true;
}

__device__ __forceinline__ bool event_sequence_contract_valid_v1(
    const FlatEventState event_state,
    const int64_t cell
) {
    if (
        cell < 0
        || event_state.active_count == nullptr
        || event_state.sequence_base == nullptr
        || event_state.typed_sequence == nullptr
        || event_state.capacity < 0
    ) {
        return false;
    }
    const int64_t sequence_base = event_state.sequence_base[cell];
    const int64_t physical_cursor = event_state.active_count[cell];
    return sequence_base >= 0
        && physical_cursor >= 0
        && physical_cursor <= event_state.capacity
        && sequence_base <= INT64_MAX - event_state.capacity;
}

__device__ __forceinline__ bool absolute_event_cursor_v1(
    const FlatEventState event_state,
    const int64_t cell,
    int64_t* const absolute_cursor
) {
    return event_sequence_contract_valid_v1(event_state, cell)
        && absolute_cursor_from_physical_v1(
            event_state.active_count[cell],
            event_state.sequence_base[cell],
            event_state.capacity,
            absolute_cursor
        );
}

__device__ __forceinline__ void fail_event_sequence_contract_v1(
    const FlatEventState event_state,
    const int64_t cell
) {
    if (cell >= 0 && event_state.status != nullptr) {
        event_state.status[cell] =
            persistent_lifecycle_events::kStatusInvalidSemantics;
    }
}

__device__ __forceinline__ bool publish_absolute_event_sequences_v1(
    const FlatEventState event_state,
    const int64_t cell,
    const int64_t physical_begin,
    const int64_t physical_end
) {
    if (
        !event_sequence_contract_valid_v1(event_state, cell)
        || physical_begin < 0
        || physical_begin > physical_end
        || physical_end != event_state.active_count[cell]
    ) {
        fail_event_sequence_contract_v1(event_state, cell);
        return false;
    }
    const int64_t cell_offset = cell * event_state.capacity;
    const int64_t sequence_base = event_state.sequence_base[cell];
    for (int64_t row = physical_begin; row < physical_end; ++row) {
        event_state.typed_sequence[cell_offset + row] = sequence_base + row;
    }
    return true;
}

template <typename T>
__device__ __forceinline__ const T* reporting_offset_or_null_v1(
    const T* const pointer,
    const int64_t offset
) {
    return pointer == nullptr ? nullptr : pointer + offset;
}

__device__ __forceinline__ bool compact_summary_context_addressable_v1(
    const FlatCompactSummaryContextV1 context,
    const int64_t cell
) {
    return context.enabled
        && context.reporting_retained_bytes != nullptr
        && context.reporting_slot_to_behavior_row != nullptr
        && context.reporting_effective_fitness_by_row != nullptr
        && context.generation_index >= 0
        && context.generation_index < context.generation_count
        && context.cell_count > 0
        && cell >= 0
        && cell < context.cell_count
        && context.reporting_row_capacity > 0
        && context.reporting_row_capacity
            <= reporting::kMaximumDynamicRowCapacityV1
        && context.slot_capacity > 0;
}

__device__ __forceinline__ int64_t compact_summary_row_index_v1(
    const FlatCompactSummaryContextV1 context,
    const int64_t cell
) {
    return context.generation_index * context.cell_count + cell;
}

__device__ __forceinline__ bool compact_summary_context_matches_controller_v1(
    const FlatCompactSummaryContextV1 context,
    const int64_t capacity
) {
    return context.slot_capacity == capacity
        && context.event_cursor_start != nullptr
        && context.event_cursor_pre_turnover != nullptr
        && context.event_cursor_end != nullptr;
}

__device__ __forceinline__ reporting::ResidentGenerationSourcesV1
compact_summary_resident_sources_v1(
    const int64_t* const slot_to_agent_id,
    const double* const energy_collected_exact,
    const float* const injury,
    const int32_t* const action_switches,
    const int32_t* const action_oscillations,
    const int32_t* const survival_steps,
    const double* const policy_energy_observation,
    const float* const health,
    const float* const drives,
    const int32_t* const pos_y,
    const int32_t* const pos_x,
    const FlatFullTickState full_tick_state,
    const FlatEventState event_state,
    const FlatOrdinaryBirthState birth_state,
    const int64_t cell,
    const int64_t capacity,
    const int64_t drive_width
) {
    const int64_t cell_capacity_offset = cell * capacity;
    const bool* const policy_use_gru = reporting_offset_or_null_v1(
        birth_state.policy_use_gru, cell_capacity_offset
    );
    return {
        reporting_offset_or_null_v1(slot_to_agent_id, cell_capacity_offset),
        reporting_offset_or_null_v1(
            energy_collected_exact, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(
            birth_state.early_energy_collected_exact, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(injury, cell_capacity_offset),
        reporting_offset_or_null_v1(
            birth_state.genome_aggression_bias, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(
            birth_state.early_injury_gained, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(
            birth_state.early_life_weight, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(
            full_tick_state.drive_jitter, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(
            birth_state.pred_err_sum, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(
            birth_state.birth_reset_shock, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(
            birth_state.episode0_pred_err_sum, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(
            birth_state.late_pred_err_sum, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(survival_steps, cell_capacity_offset),
        reporting_offset_or_null_v1(
            birth_state.offspring_count, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(birth_state.attacks, cell_capacity_offset),
        reporting_offset_or_null_v1(
            birth_state.early_survival_steps, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(action_switches, cell_capacity_offset),
        reporting_offset_or_null_v1(action_oscillations, cell_capacity_offset),
        reporting_offset_or_null_v1(
            full_tick_state.selfharm_events, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(
            birth_state.pred_err_count, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(
            birth_state.episode0_pred_err_count, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(
            full_tick_state.episode0_selfharm_events, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(
            birth_state.late_pred_err_count, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(
            policy_energy_observation, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(health, cell_capacity_offset),
        reporting_offset_or_null_v1(
            drives, cell_capacity_offset * drive_width
        ),
        nullptr,  // identity_drift_ema is unavailable in controller v1.
        reporting_offset_or_null_v1(
            birth_state.genome_exploration_bias, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(pos_y, cell_capacity_offset),
        reporting_offset_or_null_v1(pos_x, cell_capacity_offset),
        reporting_offset_or_null_v1(
            event_state.lineage_founder_id, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(
            birth_state.lineage_birth_generation, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(
            birth_state.lineage_birth_tick, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(
            event_state.lineage_depth, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(
            birth_state.first_repro_step, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(
            birth_state.patch_habitat_lineage_id, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(
            birth_state.genome_hidden_size, cell_capacity_offset
        ),
        reporting_offset_or_null_v1(
            birth_state.genome_memory_dim, cell_capacity_offset
        ),
        reinterpret_cast<const uint8_t*>(policy_use_gru),
        0,
        static_cast<int32_t>(capacity),
    };
}

__device__ __forceinline__ float clamp01_patch_observation(
    const float value
) {
    if (value < 0.0f) {
        return 0.0f;
    }
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

__device__ __forceinline__ float safe_norm_patch_observation(
    const float value,
    const float denominator
) {
    return denominator <= 0.0f
        ? 0.0f
        : clamp01_patch_observation(value / denominator);
}

constexpr int64_t kScratchPlannerPcgState = 0;
constexpr int64_t kScratchPlannerPcgBuffer = 32;
constexpr int64_t kScratchPlannerPolicyDraws = 48;
constexpr int64_t kScratchRunnerPcgState = 56;
constexpr int64_t kScratchRunnerPcgBuffer = 88;
constexpr int64_t kScratchRunnerMetricDraws = 104;
constexpr int64_t kScratchPythonPostWords = 128;
constexpr int64_t kScratchPythonTrialWords = 2624;
constexpr int64_t kScratchPythonIndex = 5120;
constexpr int64_t kScratchPythonHasGauss = 5128;
constexpr int64_t kScratchPythonGaussNext = 5136;
constexpr int64_t kScratchPythonRawDraws = 5144;
constexpr int64_t kScratchPythonSemanticDraws = 5152;
constexpr int64_t kScratchPopulationPcgState = 5160;
constexpr int64_t kScratchPopulationPcgBuffer = 5192;
constexpr int64_t kScratchPopulationSemanticDraws = 5208;
constexpr int64_t kScratchMiniSlotIds = 5216;
constexpr int64_t kScratchMiniAlive = 5232;
constexpr int64_t kScratchMiniGenomeInt32 = 5240;
constexpr int64_t kScratchMiniPolicyMemDim = 5304;
constexpr int64_t kScratchMiniPolicyUseGru = 5320;
constexpr int64_t kScratchMiniGenomeFloat32 = 5328;
constexpr int64_t kScratchMiniPlanParent = 5520;
constexpr int64_t kScratchMiniPlanChild = 5528;
constexpr int64_t kScratchMiniPlanCount = 5536;
constexpr int64_t kScratchMiniScalarOutputs = 5544;
constexpr int64_t kScratchMiniEnergy = 5568;
constexpr int64_t kScratchMiniDrives = 5584;
constexpr int64_t kScratchMiniChemistry = 5648;
constexpr int64_t kScratchMiniBirthShock = 5712;
constexpr int64_t kScratchMiniDriveOutputs = 5720;
constexpr int64_t kScratchTransactionStatus = 5792;

template <typename T>
__device__ __forceinline__ T* birth_scratch_at(
    const FlatFlattenedBirthScratch scratch,
    const int64_t cell,
    const int64_t byte_offset
) {
    return reinterpret_cast<T*>(
        scratch.storage + cell * scratch.bytes_per_cell + byte_offset
    );
}

__device__ __forceinline__ void record_status(
    int32_t* status,
    int64_t* first_failure_records,
    const int64_t cell,
    const int64_t generation,
    const int64_t tick,
    const ControllerFailureStageV1 stage,
    const int32_t code,
    const int64_t failure_code = kFirstFailureNoFailureSentinelV1,
    const int64_t attempted_index = kFirstFailureNoFailureSentinelV1,
    const int64_t required_capacity = kFirstFailureNoFailureSentinelV1,
    const int64_t available_capacity = kFirstFailureNoFailureSentinelV1
) {
    if (atomicCAS(status + cell, 0, code) != 0) {
        return;
    }
    int64_t* const record = first_failure_records
        + cell * kFirstFailureRecordWidthV1;
    record[kFirstFailureGeneration] = generation;
    record[kFirstFailureTick] = tick;
    record[kFirstFailureStage] = static_cast<int64_t>(stage);
    record[kFirstFailureStatus] = static_cast<int64_t>(code);
    record[kFirstFailureCode] = failure_code;
    record[kFirstFailureCell] = cell;
    record[kFirstFailureAttemptedIndex] = attempted_index;
    record[kFirstFailureRequiredCapacity] = required_capacity;
    record[kFirstFailureAvailableCapacity] = available_capacity;
    record[kFirstFailureSequence] = 1;
}

__device__ __forceinline__ void materialize_missing_drive_noise_v1(
    const int32_t* __restrict__ ordered_slots,
    const int64_t count,
    const int64_t* __restrict__ slot_to_agent_id,
    const bool* __restrict__ alive_mask,
    const float* __restrict__ drive_noise,
    const float* __restrict__ samples,
    const bool* __restrict__ samples_valid,
    float* __restrict__ effective_samples,
    bool* __restrict__ effective_samples_valid,
    uint64_t* __restrict__ state_words,
    int64_t* __restrict__ buffer_words,
    int64_t* __restrict__ semantic_draws,
    int64_t* __restrict__ raw64_draws,
    int32_t* __restrict__ status,
    const int64_t capacity
) {
    namespace chemistry_detail = persistent_chemistry::detail;
    semantic_draws[0] = 0;
    raw64_draws[0] = 0;
    status[0] = 0;
    if (count < 0 || count > capacity || (state_words[2] & 1ULL) == 0ULL) {
        status[0] = 1;
        return;
    }
    bool any_drive_noise = false;
    int64_t live_count = 0;
    for (int64_t row = 0; row < count; ++row) {
        const int32_t slot = ordered_slots[row];
        if (slot < 0 || static_cast<int64_t>(slot) >= capacity) {
            status[0] = 2;
            return;
        }
        if (!alive_mask[slot] || slot_to_agent_id[slot] < 0) {
            continue;
        }
        ++live_count;
        any_drive_noise = any_drive_noise || drive_noise[slot] > 0.0f;
        effective_samples_valid[slot] = false;
    }
    if (!any_drive_noise) {
        return;
    }

    chemistry_detail::Pcg64State rng = chemistry_detail::load_rng(
        state_words, buffer_words
    );
    int64_t raw64_count = 0;
    for (int64_t row = 0; row < count; ++row) {
        const int64_t slot = static_cast<int64_t>(ordered_slots[row]);
        if (!alive_mask[slot] || slot_to_agent_id[slot] < 0) {
            continue;
        }
        const bool use_noise = drive_noise[slot] > 0.0f;
        const bool use_tape = use_noise && samples_valid[slot];
        for (int64_t drive = 0; drive < 4; ++drive) {
            const float value = static_cast<float>(
                chemistry_detail::pcg64_numpy_standard_normal(
                    rng,
                    &raw64_count
                )
            );
            if (use_noise) {
                effective_samples[slot * 4 + drive] = use_tape
                    ? samples[slot * 4 + drive]
                    : value;
            }
        }
        effective_samples_valid[slot] = use_noise;
    }
    chemistry_detail::store_rng(state_words, buffer_words, rng);
    semantic_draws[0] = live_count * 4;
    raw64_draws[0] = raw64_count;
}

__device__ __forceinline__ float action_cost(
    const int32_t action,
    const double move_cost,
    const double gather_cost,
    const double rest_cost,
    const double attack_cost,
    const double action_cost_mult,
    const double rest_cost_mult
) {
    float cost = static_cast<float>(move_cost);
    if (action == kGatherAction) {
        cost = static_cast<float>(gather_cost);
    } else if (action == kRestAction) {
        cost = static_cast<float>(rest_cost);
    } else if (action == kAttackAction) {
        cost = static_cast<float>(attack_cost);
    }
    if (action_cost_mult != 1.0) {
        cost = static_cast<float>(static_cast<double>(cost) * action_cost_mult);
    }
    if (rest_cost_mult != 1.0 && action == kRestAction) {
        cost = static_cast<float>(static_cast<double>(cost) * rest_cost_mult);
    }
    return cost;
}

__device__ __forceinline__ bool is_death_candidate_v3(
    const int64_t slot,
    const int64_t* __restrict__ slot_to_agent_id,
    const bool* __restrict__ alive_mask,
    const float* __restrict__ energy,
    const float* __restrict__ health
) {
    return alive_mask[slot]
        && slot_to_agent_id[slot] >= 0
        && (health[slot] <= 0.0f || energy[slot] <= 0.0f);
}

__device__ __forceinline__ int32_t lifecycle_death_cull_cell_v3(
    int64_t* __restrict__ active_agent_ids,
    int32_t* __restrict__ active_agent_slots,
    int64_t* __restrict__ active_count_state,
    int64_t* __restrict__ slot_to_agent_id,
    int32_t* __restrict__ agent_id_to_slot,
    bool* __restrict__ alive_mask,
    int32_t* __restrict__ pos_y,
    int32_t* __restrict__ pos_x,
    int32_t* __restrict__ agent_grid,
    float* __restrict__ energy,
    float* __restrict__ health,
    double* __restrict__ policy_energy_observation,
    int32_t* __restrict__ rest_streak,
    int64_t* __restrict__ action_cache_token,
    int32_t* __restrict__ action_cache_cursor,
    int32_t* __restrict__ action_cache_next,
    int64_t* __restrict__ pending_free_agent_ids,
    int64_t* __restrict__ pending_free_slots,
    int64_t* __restrict__ pending_free_count,
    int64_t* __restrict__ death_agent_ids,
    int64_t* __restrict__ death_slots,
    int32_t* __restrict__ death_reason_codes,
    int64_t* __restrict__ death_count,
    int64_t* __restrict__ survivor_agent_ids,
    int64_t* __restrict__ survivor_count,
    const bool reporting_enabled,
    const bool reporting_include_current_tick_survival,
    const reporting::GenerationAuthorityAccessV1 reporting_access,
    const reporting::ResidentGenerationSourcesV1 reporting_sources,
    const int64_t slot_count,
    const int64_t active_capacity,
    const int64_t agent_id_capacity,
    const int64_t grid_height,
    const int64_t grid_width,
    const int64_t pending_capacity
) {
    const int64_t pending_before = pending_free_count[0];
    if (pending_before < 0 || pending_before > pending_capacity) {
        return kLifecycleInvalidPendingCount;
    }

    int64_t candidate_count = 0;
    for (int64_t slot = 0; slot < slot_count; ++slot) {
        candidate_count += is_death_candidate_v3(
            slot,
            slot_to_agent_id,
            alive_mask,
            energy,
            health
        ) ? 1 : 0;
    }
    if (pending_before + candidate_count > pending_capacity) {
        return kLifecyclePendingCapacityExceeded;
    }

    for (int64_t death_index = 0; death_index < candidate_count; ++death_index) {
        int64_t selected_slot = -1;
        int64_t selected_agent_id = INT64_MAX;
        for (int64_t slot = 0; slot < slot_count; ++slot) {
            if (!is_death_candidate_v3(
                    slot,
                    slot_to_agent_id,
                    alive_mask,
                    energy,
                    health
                )) {
                continue;
            }
            const int64_t agent_id = slot_to_agent_id[slot];
            if (
                agent_id < selected_agent_id
                || (agent_id == selected_agent_id && slot < selected_slot)
            ) {
                selected_agent_id = agent_id;
                selected_slot = slot;
            }
        }

        const int32_t reason_code = health[selected_slot] <= 0.0f ? 2 : 1;
        if (reporting_enabled) {
            reporting::BehaviorTerminalValuesV1 terminal =
                reporting::terminal_at_v1(reporting_sources, selected_slot);
            if (reporting_include_current_tick_survival) {
                ++terminal.lifespan_steps;
            }
            reporting::seal_death_before_slot_reuse_v1(
                reporting_access,
                static_cast<int32_t>(selected_slot),
                selected_agent_id,
                terminal
            );
        }
        const int32_t y = pos_y[selected_slot];
        const int32_t x = pos_x[selected_slot];
        if (
            y >= 0 && y < grid_height
            && x >= 0 && x < grid_width
            && agent_grid[static_cast<int64_t>(y) * grid_width + x]
                == static_cast<int32_t>(selected_agent_id)
        ) {
            agent_grid[static_cast<int64_t>(y) * grid_width + x] = -1;
        }
        if (selected_agent_id >= 0 && selected_agent_id < agent_id_capacity) {
            agent_id_to_slot[selected_agent_id] = -1;
        }

        alive_mask[selected_slot] = false;
        slot_to_agent_id[selected_slot] = -1;
        pos_y[selected_slot] = -1;
        pos_x[selected_slot] = -1;
        energy[selected_slot] = 0.0f;
        health[selected_slot] = 0.0f;
        policy_energy_observation[selected_slot] = __longlong_as_double(
            static_cast<long long>(0x7ff8000000000000ULL)
        );
        rest_streak[selected_slot] = 0;
        action_cache_token[selected_slot] = -1;
        action_cache_cursor[selected_slot] = -1;
        action_cache_next[selected_slot] = -1;

        const int64_t pending_index = pending_before + death_index;
        pending_free_agent_ids[pending_index] = selected_agent_id;
        pending_free_slots[pending_index] = selected_slot;
        death_agent_ids[death_index] = selected_agent_id;
        death_slots[death_index] = selected_slot;
        death_reason_codes[death_index] = reason_code;
    }
    pending_free_count[0] = pending_before + candidate_count;
    death_count[0] = candidate_count;

    for (int64_t index = 0; index < active_capacity; ++index) {
        active_agent_ids[index] = -1;
        active_agent_slots[index] = -1;
    }
    int64_t remaining = 0;
    int64_t previous_agent_id = INT64_MIN;
    int64_t previous_slot = -1;
    while (true) {
        int64_t selected_slot = -1;
        int64_t selected_agent_id = INT64_MAX;
        for (int64_t slot = 0; slot < slot_count; ++slot) {
            if (!alive_mask[slot] || slot_to_agent_id[slot] < 0) {
                continue;
            }
            const int64_t agent_id = slot_to_agent_id[slot];
            const bool follows_previous = agent_id > previous_agent_id
                || (agent_id == previous_agent_id && slot > previous_slot);
            if (
                follows_previous
                && (
                    agent_id < selected_agent_id
                    || (agent_id == selected_agent_id && slot < selected_slot)
                )
            ) {
                selected_agent_id = agent_id;
                selected_slot = slot;
            }
        }
        if (selected_slot < 0) {
            break;
        }
        if (remaining >= active_capacity) {
            return kLifecyclePendingCapacityExceeded;
        }
        active_agent_ids[remaining] = selected_agent_id;
        active_agent_slots[remaining] = static_cast<int32_t>(selected_slot);
        survivor_agent_ids[remaining] = selected_agent_id;
        previous_agent_id = selected_agent_id;
        previous_slot = selected_slot;
        ++remaining;
    }
    active_count_state[0] = remaining;
    survivor_count[0] = remaining;
    return kLifecycleOk;
}

__device__ __forceinline__ int32_t release_residual_free_slots_v4(
    const int64_t* __restrict__ slot_to_agent_id,
    const bool* __restrict__ alive_mask,
    int64_t* __restrict__ slot_free_queue,
    int64_t* __restrict__ slot_free_queue_count,
    const int64_t* __restrict__ residual_slots,
    const int64_t residual_count,
    int64_t* __restrict__ appended_count,
    const int64_t slot_count,
    const int64_t queue_capacity
) {
    constexpr int32_t kInvalidState = 30;
    constexpr int32_t kCapacityExceeded = 31;
    constexpr int32_t kContradiction = 32;
    const int64_t queue_before = slot_free_queue_count[0];
    if (
        queue_before < 0 || queue_before > queue_capacity
        || residual_count < 0 || residual_count > slot_count
    ) {
        return kInvalidState;
    }
    if (queue_before + residual_count > queue_capacity) {
        return kCapacityExceeded;
    }
    for (int64_t row = 0; row < residual_count; ++row) {
        const int64_t slot = residual_slots[row];
        if (
            slot < 0 || slot >= slot_count || alive_mask[slot]
            || slot_to_agent_id[slot] >= 0
        ) {
            return kContradiction;
        }
        for (int64_t queued = 0; queued < queue_before; ++queued) {
            if (slot_free_queue[queued] == slot) {
                return kContradiction;
            }
        }
        for (int64_t prior = 0; prior < row; ++prior) {
            if (residual_slots[prior] == slot) {
                return kContradiction;
            }
        }
    }
    for (int64_t row = 0; row < residual_count; ++row) {
        slot_free_queue[queue_before + row] = residual_slots[row];
    }
    slot_free_queue_count[0] = queue_before + residual_count;
    appended_count[0] = residual_count;
    return kLifecycleOk;
}

__device__ __forceinline__ void append_death_event_class_v4(
    const FlatEventState& event_state,
    const int64_t* __restrict__ death_agent_ids,
    const int64_t* __restrict__ death_slots,
    const int32_t* __restrict__ death_reason_codes,
    const int64_t* __restrict__ death_count,
    const int64_t ticks,
    const int64_t capacity,
    const int64_t cell,
    const int64_t tick,
    const int64_t generation,
    const int64_t absolute_tick
) {
    using namespace persistent_lifecycle_events;
    if (!event_sequence_contract_valid_v1(event_state, cell)) {
        fail_event_sequence_contract_v1(event_state, cell);
        return;
    }
    const int64_t physical_sequence_begin = event_state.active_count[cell];
    const FlatEventTapeColumns columns{
        {event_state.event_type, event_state.capacity, 1},
        {event_state.event_generation, event_state.capacity, 1},
        {event_state.event_tick, event_state.capacity, 1},
        {event_state.event_actor_id, event_state.capacity, 1},
        {event_state.event_target_id, event_state.capacity, 1},
        {event_state.event_value, event_state.capacity, 1},
        {event_state.typed_event_type, event_state.capacity, 1},
        {event_state.typed_generation, event_state.capacity, 1},
        {event_state.typed_tick, event_state.capacity, 1},
        {event_state.typed_sequence, event_state.capacity, 1},
        {event_state.typed_actor_id, event_state.capacity, 1},
        {event_state.typed_target_id, event_state.capacity, 1},
        {event_state.typed_actor_slot, event_state.capacity, 1},
        {event_state.typed_target_slot, event_state.capacity, 1},
        {event_state.typed_parent_id, event_state.capacity, 1},
        {event_state.typed_child_id, event_state.capacity, 1},
        {event_state.typed_founder_id, event_state.capacity, 1},
        {event_state.typed_lineage_depth, event_state.capacity, 1},
        {event_state.typed_subtype, event_state.capacity, 1},
        {event_state.typed_reason, event_state.capacity, 1},
        {event_state.typed_success, event_state.capacity, 1},
        {event_state.typed_value, event_state.capacity, 1},
    };
    const FlatEventTapeState tape{
        {event_state.active_count, 1},
        {event_state.status, 1},
        event_state.capacity,
    };
    const FlatDeathRows deaths{
        {death_agent_ids + tick * capacity, ticks * capacity, 1},
        {death_slots + tick * capacity, ticks * capacity, 1},
        {death_reason_codes + tick * capacity, ticks * capacity, 1},
        {death_count + tick, ticks},
        capacity,
    };
    const FlatLineageRows lineage{
        {event_state.lineage_founder_id, capacity, 1},
        {event_state.lineage_depth, capacity, 1},
        capacity,
    };
    append_death_lifecycle_events(
        columns,
        tape,
        deaths,
        lineage,
        cell,
        generation,
        absolute_tick
    );
    if (event_state.status[cell] == kStatusOk) {
        publish_absolute_event_sequences_v1(
            event_state,
            cell,
            physical_sequence_begin,
            event_state.active_count[cell]
        );
    }
}

__device__ __forceinline__ void accumulate_action_terminal_v4(
    const FlatEventState& event_state,
    const int64_t* __restrict__ death_agent_ids,
    const int64_t* __restrict__ death_slots,
    const int32_t* __restrict__ death_reason_codes,
    const int64_t* __restrict__ death_count,
    const int32_t* __restrict__ survival_steps,
    const int64_t ticks,
    const int64_t capacity,
    const int64_t /* agent_id_capacity */,
    const int64_t cell,
    const int64_t tick
) {
    using namespace persistent_lifecycle_events;
    const FlatDeathRows deaths{
        {death_agent_ids + tick * capacity, ticks * capacity, 1},
        {death_slots + tick * capacity, ticks * capacity, 1},
        {death_reason_codes + tick * capacity, ticks * capacity, 1},
        {death_count + tick, ticks},
        capacity,
    };
    if (
        event_state.initial_agent_ids_by_row == nullptr
        || event_state.terminal_survival == nullptr
        || event_state.terminal_rows == nullptr
        || event_state.reporting_rows < 0
    ) {
        return;
    }
    const int64_t count = deaths.count.at(cell);
    if (count <= 0 || count > deaths.capacity) {
        return;
    }
    const int64_t* const initial_agent_ids_by_row =
        event_state.initial_agent_ids_by_row + cell * event_state.reporting_rows;
    for (int64_t death = 0; death < count; ++death) {
        const int64_t agent_id = deaths.agent_ids.at(cell, death);
        const int64_t slot = deaths.slots.at(cell, death);
        if (
            !active_identity::agent_id_fits_grid_v1(agent_id)
            || slot < 0 || slot >= capacity
        ) {
            continue;
        }
        int64_t lower = 0;
        int64_t upper = event_state.reporting_rows;
        while (lower < upper) {
            const int64_t middle = lower + (upper - lower) / 2;
            if (initial_agent_ids_by_row[middle] < agent_id) {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }
        if (
            lower >= event_state.reporting_rows
            || initial_agent_ids_by_row[lower] != agent_id
        ) {
            continue;
        }
        const int64_t row = cell * event_state.reporting_rows + lower;
        event_state.terminal_survival[row] +=
            survival_steps[cell * capacity + slot] + 1;
        event_state.terminal_rows[row] += 1;
    }
}

struct FlatMatePlannerScratchV1 {
    double* planned_energy;
    int64_t* planned_parent_slots;
    int64_t* planned_mate_slots;
    int64_t* planned_child_slots;
    int32_t* planned_spawn_positions;
    double* planned_child_energy;
    double* planned_parent_energy_after;
    double* planned_mate_energy_after;
    int64_t* decision_mate_slots;
    int32_t* decision_adjacent_counts;
    int32_t* decision_spawn_candidate_counts;
    int64_t* reason_counts;
};

__device__ __forceinline__ bool make_flat_mate_planner_scratch_v1(
    const FlatFlattenedBirthScratch scratch,
    const int64_t cell,
    const int64_t birth_capacity,
    const int64_t capacity,
    const int64_t packed_width,
    const int64_t hmem_width,
    const int64_t torch_state_bytes,
    FlatMatePlannerScratchV1* const output
) {
    namespace legacy = flattened_birth_scratch_v1;
    namespace dynamic = flattened_birth_scratch_v2;
    if (
        output == nullptr || scratch.storage == nullptr || cell < 0
        || birth_capacity < 0 || birth_capacity > capacity
        || scratch.bytes_per_cell < legacy::kDynamicOffset
    ) {
        return false;
    }
    const dynamic::FlatBirthScratchLayoutV2 layout = dynamic::make_layout(
        birth_capacity,
        capacity,
        packed_width,
        hmem_width,
        torch_state_bytes
    );
    if (
        !layout.ok()
        || layout.required_bytes
            > scratch.bytes_per_cell - legacy::kDynamicOffset
    ) {
        return false;
    }
    const dynamic::FlatBirthScratchArenaV2 arena{
        scratch.storage + legacy::kDynamicOffset,
        scratch.bytes_per_cell,
    };
    if (!arena.holds(layout)) {
        return false;
    }

    uint8_t* const decision_storage = arena.at<uint8_t>(
        cell, layout, dynamic::kEventStagingRows
    );
    const int64_t decision_storage_bytes =
        layout.segment(dynamic::kEventStagingRows).bytes;
    const int64_t row64_bytes =
        capacity * static_cast<int64_t>(sizeof(int64_t));
    const int64_t row32_bytes =
        capacity * static_cast<int64_t>(sizeof(int32_t));
    const int64_t planned_energy_offset = 0;
    const int64_t planned_parent_offset = planned_energy_offset + row64_bytes;
    const int64_t planned_mate_offset = planned_parent_offset + row64_bytes;
    const int64_t planned_child_offset = planned_mate_offset + row64_bytes;
    const int64_t planned_spawn_offset = planned_child_offset + row64_bytes;
    const int64_t planned_child_energy_offset =
        planned_spawn_offset + 2 * row32_bytes;
    const int64_t planned_parent_energy_after_offset =
        planned_child_energy_offset + row64_bytes;
    const int64_t planned_mate_energy_after_offset =
        planned_parent_energy_after_offset + row64_bytes;
    const int64_t decision_mate_offset =
        planned_mate_energy_after_offset + row64_bytes;
    const int64_t decision_adjacent_offset =
        decision_mate_offset + row64_bytes;
    const int64_t decision_spawn_offset =
        decision_adjacent_offset + row32_bytes;
    const int64_t reason_offset = decision_spawn_offset + row32_bytes;
    constexpr int64_t kMateReasonCount =
        persistent_mate_birth_plan_v1::kReasonCount;
    const int64_t required_decision_bytes = reason_offset
        + kMateReasonCount * static_cast<int64_t>(sizeof(int64_t));
    if (
        decision_storage == nullptr
        || required_decision_bytes > decision_storage_bytes
    ) {
        return false;
    }

    output[0] = {
        reinterpret_cast<double*>(
            decision_storage + planned_energy_offset
        ),
        reinterpret_cast<int64_t*>(
            decision_storage + planned_parent_offset
        ),
        reinterpret_cast<int64_t*>(
            decision_storage + planned_mate_offset
        ),
        reinterpret_cast<int64_t*>(
            decision_storage + planned_child_offset
        ),
        reinterpret_cast<int32_t*>(
            decision_storage + planned_spawn_offset
        ),
        reinterpret_cast<double*>(
            decision_storage + planned_child_energy_offset
        ),
        reinterpret_cast<double*>(
            decision_storage + planned_parent_energy_after_offset
        ),
        reinterpret_cast<double*>(
            decision_storage + planned_mate_energy_after_offset
        ),
        reinterpret_cast<int64_t*>(
            decision_storage + decision_mate_offset
        ),
        reinterpret_cast<int32_t*>(
            decision_storage + decision_adjacent_offset
        ),
        reinterpret_cast<int32_t*>(
            decision_storage + decision_spawn_offset
        ),
        reinterpret_cast<int64_t*>(decision_storage + reason_offset),
    };
    return output->planned_energy != nullptr
        && output->planned_parent_slots != nullptr
        && output->planned_mate_slots != nullptr
        && output->planned_child_slots != nullptr
        && output->planned_spawn_positions != nullptr
        && output->planned_child_energy != nullptr
        && output->planned_parent_energy_after != nullptr
        && output->planned_mate_energy_after != nullptr;
}

template <int kPhase>
__device__ __forceinline__ int32_t execute_ordinary_birth_phase_v1(
    int64_t* __restrict__ active_agent_ids,
    int32_t* __restrict__ active_agent_slots,
    int64_t* __restrict__ active_count_state,
    int64_t* __restrict__ slot_to_agent_id,
    int32_t* __restrict__ agent_id_to_slot,
    int64_t* __restrict__ slot_tokens,
    bool* __restrict__ alive_mask,
    int32_t* __restrict__ pos_y,
    int32_t* __restrict__ pos_x,
    int32_t* __restrict__ agent_grid,
    float* __restrict__ energy,
    float* __restrict__ health,
    float* __restrict__ injury,
    double* __restrict__ policy_energy_observation,
    int32_t* __restrict__ rest_streak,
    int32_t* __restrict__ action_cache_actions,
    int64_t* __restrict__ action_cache_token,
    int32_t* __restrict__ action_cache_cursor,
    int32_t* __restrict__ action_cache_next,
    int32_t* __restrict__ last_action,
    int32_t* __restrict__ last_last_action,
    int32_t* __restrict__ action_switches,
    int32_t* __restrict__ action_oscillations,
    int32_t* __restrict__ episode0_switches,
    int32_t* __restrict__ episode0_oscillations,
    int32_t* __restrict__ survival_steps,
    float* __restrict__ mlp_pack,
    float* __restrict__ drives,
    float* __restrict__ attack_bias,
    float* __restrict__ action_temperature,
    float* __restrict__ drive_logit_gain,
    float* __restrict__ energy_collected,
    double* __restrict__ energy_collected_exact,
    const FlatFullTickState full_tick_state,
    const FlatEventState event_state,
    const FlatOrdinaryBirthState birth_state,
    const FlatOrdinaryBirthOutputs birth_outputs,
    const FlatFlattenedBirthScratch birth_scratch,
    const FlatOrdinaryBirthConfig birth_config,
    const FlatMechanicsConfig mechanics_config,
    const int64_t cell,
    const int64_t tick,
    const int64_t output_ticks,
    const int64_t capacity,
    const int64_t active_capacity,
    const int64_t agent_id_capacity,
    const int64_t height,
    const int64_t width,
    const int64_t cache_capacity,
    const int64_t drive_width,
    const int64_t packed_width,
    const int64_t chemistry_width,
    const float max_energy,
    const float start_health,
    const int32_t external_gate_status
) {
    static_assert(kPhase == 1 || kPhase == 2, "flattened birth has two phases");
    using namespace persistent_birth_action_history;
    using namespace persistent_birth_drives;
    using namespace persistent_birth_events;
    using namespace persistent_birth_identity;
    using namespace persistent_birth_plan;
    using namespace persistent_birth_policy;
    using namespace persistent_birth_scalar_genome;
    using namespace persistent_newborn_state;

    const int64_t cell_capacity_offset = cell * capacity;
    const int64_t cell_active_offset = cell * active_capacity;
    const int64_t cell_id_map_offset = cell * agent_id_capacity;
    const int64_t cell_grid_offset = cell * height * width;
    const int64_t output_tick = output_ticks == 1 ? 0 : tick;
    const int64_t cell_tick = cell * output_ticks + output_tick;
    const int64_t tick_capacity_offset = cell_tick * capacity;
    const int64_t birth_capacity = birth_config.max_births_per_tick;
    const int64_t tick_birth_offset = cell_tick * birth_capacity;
    const int64_t absolute_tick = mechanics_config.start_tick + tick;

    if constexpr (kPhase == 1) {
        for (int64_t birth = 0; birth < birth_capacity; ++birth) {
            birth_outputs.planned_mate_slots[tick_birth_offset + birth] = -1;
            birth_outputs.committed_mate_agent_ids[
                tick_birth_offset + birth
            ] = -1;
        }
        birth_outputs.combined_status[cell_tick] = 0;
        if (external_gate_status != 0) {
            birth_outputs.combined_status[cell_tick] = 4;
            return 4;
        }

        birth_outputs.slot_free_queue_head[cell] = 0;
        for (int64_t word = 0; word < 4; ++word) {
            birth_outputs.planner_pcg64_state_before[cell_tick * 4 + word] =
                full_tick_state.shared_env_policy_pcg64_state_words[cell * 4 + word];
        }
        for (int64_t word = 0; word < 2; ++word) {
            birth_outputs.planner_pcg64_buffer_before[cell_tick * 2 + word] =
                full_tick_state.shared_env_policy_pcg64_buffer_words[cell * 2 + word];
        }

        uint64_t* planner_pcg_state = birth_scratch_at<uint64_t>(
            birth_scratch,
            cell,
            kScratchPlannerPcgState
        );
        int64_t* planner_pcg_buffer = birth_scratch_at<int64_t>(
            birth_scratch,
            cell,
            kScratchPlannerPcgBuffer
        );
        int64_t* planner_policy_draws = birth_scratch_at<int64_t>(
            birth_scratch,
            cell,
            kScratchPlannerPolicyDraws
        );
        uint64_t* runner_pcg_state = birth_scratch_at<uint64_t>(
            birth_scratch,
            cell,
            kScratchRunnerPcgState
        );
        int64_t* runner_pcg_buffer = birth_scratch_at<int64_t>(
            birth_scratch,
            cell,
            kScratchRunnerPcgBuffer
        );
        int64_t* runner_metric_draws = birth_scratch_at<int64_t>(
            birth_scratch,
            cell,
            kScratchRunnerMetricDraws
        );
        int32_t* transaction_status = birth_scratch_at<int32_t>(
            birth_scratch,
            cell,
            kScratchTransactionStatus
        );
        transaction_status[0] = -1;
        for (int64_t slot = 0; slot < capacity; ++slot) {
            birth_outputs.planned_repro_cooldown[cell_capacity_offset + slot] =
                birth_state.repro_cooldown[cell_capacity_offset + slot];
        }
        for (int64_t word = 0; word < 4; ++word) {
            planner_pcg_state[word] =
                full_tick_state.shared_env_policy_pcg64_state_words[cell * 4 + word];
            runner_pcg_state[word] =
                full_tick_state.runner_pcg64_state_words[cell * 4 + word];
        }
        for (int64_t word = 0; word < 2; ++word) {
            planner_pcg_buffer[word] =
                full_tick_state.shared_env_policy_pcg64_buffer_words[cell * 2 + word];
            runner_pcg_buffer[word] =
                full_tick_state.runner_pcg64_buffer_words[cell * 2 + word];
        }
        planner_policy_draws[0] = full_tick_state.policy_np_semantic_draws[cell];
        runner_metric_draws[0] = full_tick_state.metrics_np_semantic_draws[cell];

        const FlatOrdinaryBirthPlanState plan_state{
            slot_to_agent_id + cell_capacity_offset,
            alive_mask + cell_capacity_offset,
            birth_state.policy_use_gru + cell_capacity_offset,
            pos_y + cell_capacity_offset,
            pos_x + cell_capacity_offset,
            agent_grid + cell_grid_offset,
            energy + cell_capacity_offset,
            policy_energy_observation + cell_capacity_offset,
            health + cell_capacity_offset,
            injury + cell_capacity_offset,
            birth_state.scar + cell_capacity_offset,
            birth_outputs.planned_repro_cooldown + cell_capacity_offset,
            birth_state.repro_threshold + cell_capacity_offset,
            full_tick_state.slot_free_queue + cell_capacity_offset,
            full_tick_state.slot_free_queue_count + cell,
            birth_outputs.slot_free_queue_head + cell,
            planner_pcg_state,
            planner_pcg_buffer,
            planner_policy_draws,
            runner_pcg_state,
            runner_pcg_buffer,
            runner_metric_draws,
            nullptr,
            survival_steps + cell_capacity_offset,
        };
        const FlatOrdinaryBirthPlanBuffers plan_buffers{
        birth_outputs.planned_repro_cooldown + cell_capacity_offset,
        birth_outputs.planned_parent_slots + tick_birth_offset,
        birth_outputs.planned_child_slots + tick_birth_offset,
        birth_outputs.planned_spawn_positions + tick_birth_offset * 2,
        birth_outputs.planned_child_energy + tick_birth_offset,
        birth_outputs.decision_parent_slots + tick_capacity_offset,
        birth_outputs.decision_reason_codes + tick_capacity_offset,
        birth_outputs.decision_candidate_counts + tick_capacity_offset,
        birth_outputs.decision_draw_starts + tick_capacity_offset,
        birth_outputs.decision_draw_ends + tick_capacity_offset,
        birth_outputs.decision_selected_positions + tick_capacity_offset * 2,
        birth_outputs.decision_selected_reserved + tick_capacity_offset,
        birth_outputs.reason_counts + cell_tick * kReasonCount,
        birth_outputs.planned_count + cell_tick,
        birth_outputs.decision_count + cell_tick,
        birth_outputs.spawn_rng_draws + cell_tick,
        birth_outputs.spawn_rng_semantic_draws + cell_tick,
        birth_outputs.runner_rng_draws + cell_tick,
        birth_outputs.runner_rng_semantic_draws + cell_tick,
        birth_outputs.plan_status + cell_tick,
    };
        if (!birth_config.requires_mate) {
            plan_ordinary_births_serial(
                plan_state,
                plan_buffers,
                {
                    capacity,
                    height,
                    width,
                    capacity,
                    birth_capacity,
                    birth_parent_energy_share_for_cell_v1(birth_config, cell),
                    static_cast<float>(
                        birth_minimum_energy_buffer_for_cell_v1(
                            birth_config, cell
                        )
                    ),
                    birth_config.injury_repro_k_injury,
                    birth_config.injury_repro_k_scar,
                    birth_config.injury_birth_fail_scar,
                    birth_config.injury_birth_fail_injury,
                    birth_config.env_policy_alias,
                    false,
                    false,
                    birth_config.injury_repro_k_injury > 0.0
                        || birth_config.injury_repro_k_scar > 0.0,
                    birth_config.injury_birth_fail_scar > 0.0
                        || birth_config.injury_birth_fail_injury > 0.0,
                    false,
                    mechanics_config.life_history_mode,
                    mechanics_config.life_history_maturity_start_ticks,
                    mechanics_config.life_history_maturity_rise_ticks,
                    mechanics_config.life_history_fertility_decline_start_ticks,
                    mechanics_config.life_history_fertility_decline_ticks,
                    mechanics_config.life_history_elder_fertility_floor,
                }
            );
        } else {
            namespace mate_plan = persistent_mate_birth_plan_v1;
            FlatMatePlannerScratchV1 mate_scratch{};
            if (!make_flat_mate_planner_scratch_v1(
                    birth_scratch,
                    cell,
                    birth_capacity,
                    capacity,
                    packed_width,
                    birth_config.hmem_width,
                    birth_config.torch_state_bytes,
                    &mate_scratch
                )) {
                birth_outputs.plan_status[cell_tick] =
                    mate_plan::kStatusInvalidConfig;
            } else {
                for (int64_t slot = 0; slot < capacity; ++slot) {
                    const int64_t slot_index = cell_capacity_offset + slot;
                    const double observed =
                        policy_energy_observation[slot_index];
                    mate_scratch.planned_energy[slot] =
                        ::isfinite(observed)
                        ? observed
                        : static_cast<double>(energy[slot_index]);
                }
                mate_plan::plan_births_serial_v1(
                    {
                        slot_to_agent_id + cell_capacity_offset,
                        alive_mask + cell_capacity_offset,
                        pos_y + cell_capacity_offset,
                        pos_x + cell_capacity_offset,
                        agent_grid + cell_grid_offset,
                        mate_scratch.planned_energy,
                        health + cell_capacity_offset,
                        injury + cell_capacity_offset,
                        birth_state.scar + cell_capacity_offset,
                        survival_steps + cell_capacity_offset,
                        nullptr,
                        nullptr,
                        birth_outputs.planned_repro_cooldown
                            + cell_capacity_offset,
                        birth_state.repro_threshold + cell_capacity_offset,
                        full_tick_state.slot_free_queue + cell_capacity_offset,
                        full_tick_state.slot_free_queue_count + cell,
                        birth_outputs.slot_free_queue_head + cell,
                        planner_pcg_state,
                        planner_pcg_buffer,
                        planner_policy_draws,
                        runner_pcg_state,
                        runner_pcg_buffer,
                        runner_metric_draws,
                    },
                    {
                        birth_outputs.planned_repro_cooldown
                            + cell_capacity_offset,
                        mate_scratch.planned_energy,
                        mate_scratch.planned_parent_slots,
                        mate_scratch.planned_mate_slots,
                        mate_scratch.planned_child_slots,
                        mate_scratch.planned_spawn_positions,
                        mate_scratch.planned_child_energy,
                        mate_scratch.planned_parent_energy_after,
                        mate_scratch.planned_mate_energy_after,
                        birth_outputs.decision_parent_slots
                            + tick_capacity_offset,
                        mate_scratch.decision_mate_slots,
                        birth_outputs.decision_reason_codes
                            + tick_capacity_offset,
                        mate_scratch.decision_adjacent_counts,
                        mate_scratch.decision_spawn_candidate_counts,
                        mate_scratch.reason_counts,
                        birth_outputs.planned_count + cell_tick,
                        birth_outputs.decision_count + cell_tick,
                        birth_outputs.spawn_rng_draws + cell_tick,
                        birth_outputs.spawn_rng_semantic_draws + cell_tick,
                        birth_outputs.runner_rng_draws + cell_tick,
                        birth_outputs.runner_rng_semantic_draws + cell_tick,
                        birth_outputs.plan_status + cell_tick,
                    },
                    {
                        capacity,
                        height,
                        width,
                        capacity,
                        birth_capacity,
                        birth_parent_energy_share_for_cell_v1(birth_config, cell),
                        static_cast<float>(
                            birth_minimum_energy_buffer_for_cell_v1(
                                birth_config, cell
                            )
                        ),
                        birth_cooldown_steps_for_cell_v1(birth_config, cell),
                        birth_try_probability_for_cell_v1(birth_config, cell),
                        0.0,
                        0.0,
                        birth_config.injury_repro_k_injury,
                        birth_config.injury_repro_k_scar,
                        birth_config.injury_birth_fail_scar,
                        birth_config.injury_birth_fail_injury,
                        birth_config.env_policy_alias,
                        true,
                        false,
                        false,
                        false,
                        mechanics_config.life_history_mode,
                        mechanics_config.life_history_maturity_start_ticks,
                        mechanics_config.life_history_maturity_rise_ticks,
                        mechanics_config.life_history_fertility_decline_start_ticks,
                        mechanics_config.life_history_fertility_decline_ticks,
                        mechanics_config.life_history_elder_fertility_floor,
                    },
                    {
                        1,
                        cross_path_resident_control_v1::
                            has_cross_path_authority_v1(
                                birth_config.cross_path_control
                            )
                            ? (
                                birth_config.cross_path_control
                                    .selection_mode_by_cell[cell]
                                    == cross_path_resident_control_v1::
                                        kSelectionModeGenotypeBlindNeutral
                                ? mate_plan::
                                    kMateBirthSelectionUniformNeutralV1
                                : -1
                            )
                            : mate_plan::kMateBirthSelectionLegacyAgentIdV1,
                        cross_path_resident_control_v1::
                            has_cross_path_authority_v1(
                                birth_config.cross_path_control
                            )
                            ? birth_config.cross_path_control
                                .selection_seed_by_cell[cell]
                            : 0ULL,
                        mechanics_config.generation,
                        absolute_tick,
                        !cross_path_resident_control_v1::
                            has_cross_path_authority_v1(
                                birth_config.cross_path_control
                            )
                            || birth_config.cross_path_control
                                .trait_authority_enabled_by_cell[cell],
                        cross_path_resident_control_v1::
                            has_cross_path_authority_v1(
                                birth_config.cross_path_control
                            )
	                            ? birth_config.cross_path_control
	                                .neutral_repro_threshold_by_cell[cell]
	                            : 0.0,
	                        birth_config.cross_path_control.reproduction_exposure,
	                        cell,
	                        cross_path_resident_control_v1::
	                            has_cross_path_authority_v1(
	                                birth_config.cross_path_control
	                            )
	                            ? birth_config.cross_path_control
	                                .reproduction_path_by_cell[cell]
	                            : static_cast<int8_t>(-1),
	                    }
	                );
                if (birth_outputs.plan_status[cell_tick] == mate_plan::kStatusOk) {
                    const int64_t mate_planned_count =
                        birth_outputs.planned_count[cell_tick];
                    const int64_t mate_decision_count =
                        birth_outputs.decision_count[cell_tick];
                    for (int64_t birth = 0;
                         birth < birth_capacity;
                         ++birth) {
                        const int64_t output_row = tick_birth_offset + birth;
                        birth_outputs.planned_parent_slots[output_row] =
                            mate_scratch.planned_parent_slots[birth];
                        birth_outputs.planned_mate_slots[output_row] =
                            mate_scratch.planned_mate_slots[birth];
                        birth_outputs.planned_child_slots[output_row] =
                            mate_scratch.planned_child_slots[birth];
                        birth_outputs.planned_spawn_positions[
                            output_row * 2
                        ] = mate_scratch.planned_spawn_positions[birth * 2];
                        birth_outputs.planned_spawn_positions[
                            output_row * 2 + 1
                        ] = mate_scratch.planned_spawn_positions[
                            birth * 2 + 1
                        ];
                        birth_outputs.planned_child_energy[output_row] =
                            static_cast<float>(
                                mate_scratch.planned_child_energy[birth]
                            );
                    }
                    for (int64_t birth = 0;
                         birth < mate_planned_count;
                         ++birth) {
                        const int64_t output_row = tick_birth_offset + birth;
                        const int64_t parent_slot =
                            birth_outputs.planned_parent_slots[output_row];
                        const int64_t mate_slot =
                            birth_outputs.planned_mate_slots[output_row];
                        // Identity preflight consumes the eligible post-aging
                        // cooldown view; successful cooldown settlement occurs
                        // atomically in the later birth commit.
                        birth_outputs.planned_repro_cooldown[
                            cell_capacity_offset + parent_slot
                        ] = 0;
                        birth_outputs.planned_repro_cooldown[
                            cell_capacity_offset + mate_slot
                        ] = 0;
                    }
                    for (int64_t decision = 0;
                         decision < mate_decision_count;
                         ++decision) {
                        const int64_t output_row =
                            tick_capacity_offset + decision;
                        birth_outputs.decision_candidate_counts[output_row] =
                            mate_scratch
                                .decision_spawn_candidate_counts[decision];
                        birth_outputs.decision_draw_starts[output_row] = 0;
                        birth_outputs.decision_draw_ends[output_row] = 0;
                        birth_outputs.decision_selected_positions[
                            output_row * 2
                        ] = -1;
                        birth_outputs.decision_selected_positions[
                            output_row * 2 + 1
                        ] = -1;
                        birth_outputs.decision_selected_reserved[output_row] =
                            false;
                    }
                    for (int32_t reason = 0;
                         reason < persistent_birth_plan::kReasonCount;
                         ++reason) {
                        birth_outputs.reason_counts[
                            cell_tick * persistent_birth_plan::kReasonCount
                                + reason
                        ] = mate_scratch.reason_counts[reason];
                    }
                    // The accepted 10-row public reason plane predates mating.
                    // Preserve its extent while aggregating the two new
                    // non-birth opportunities into its opportunity row.
                    birth_outputs.reason_counts[
                        cell_tick * persistent_birth_plan::kReasonCount
                            + persistent_birth_plan::kReasonOpportunity
                    ] += mate_scratch.reason_counts[
                            mate_plan::kReasonAlreadyMated
                        ]
                        + mate_scratch.reason_counts[
                            mate_plan::kReasonTryProbability
                        ];
                }
            }
        }
        if (birth_outputs.plan_status[cell_tick] != 0) {
            birth_outputs.combined_status[cell_tick] = 1;
            transaction_status[0] = 1;
            return 1;
        }
        transaction_status[0] = 0;
        return 0;
    }

    if (external_gate_status != 0) {
        birth_outputs.combined_status[cell_tick] = 4;
        return 4;
    }
    if (birth_outputs.plan_status[cell_tick] != 0) {
        birth_outputs.combined_status[cell_tick] = 1;
        return 1;
    }

    int32_t* transaction_status = birth_scratch_at<int32_t>(
        birth_scratch,
        cell,
        kScratchTransactionStatus
    );
    const int64_t planned_count = birth_outputs.planned_count[cell_tick];
    const int64_t decision_count = birth_outputs.decision_count[cell_tick];
    int64_t physical_policy_cursor = 0;
    int64_t physical_python_raw_draws = 0;
    if (
        transaction_status[0] != 0
        || planned_count < 0 || planned_count > birth_capacity
        || decision_count < 0 || decision_count > capacity
        || birth_scratch.bytes_per_cell < kBirthScratchBytesPerCell
        || birth_state.policy_birth_tape_cursor_base[cell]
            > INT64_MAX - birth_config.policy_tape_capacity
        || !physical_cursor_from_absolute_v1(
            birth_state.policy_birth_tape_cursor[cell],
            birth_state.policy_birth_tape_cursor_base[cell],
            birth_config.policy_tape_capacity,
            &physical_policy_cursor
        )
        || planned_count
            > birth_config.policy_tape_capacity - physical_policy_cursor
        || birth_state.python_random_raw_draw_base[cell]
            > INT64_MAX - birth_config.gauss_tape_capacity
        || !physical_cursor_from_absolute_v1(
            birth_state.python_random_raw_draws[cell],
            birth_state.python_random_raw_draw_base[cell],
            birth_config.gauss_tape_capacity,
            &physical_python_raw_draws
        )
    ) {
        transaction_status[0] = 2;
        birth_outputs.combined_status[cell_tick] = 2;
        return 2;
    }
    birth_scratch_at<int64_t>(
        birth_scratch,
        cell,
        kScratchPythonRawDraws
    )[0] = physical_python_raw_draws;
    birth_scratch_at<int64_t>(
        birth_scratch,
        cell,
        flattened_birth_scratch_v1::kMiniPolicyCursor
    )[0] = physical_policy_cursor;

    int64_t required_event_rows = 0;
    int64_t success_ordinal = 0;
    const int64_t policy_cursor = physical_policy_cursor;
    for (int64_t decision = 0; decision < decision_count; ++decision) {
        const int32_t reason = birth_outputs.decision_reason_codes[
            tick_capacity_offset + decision
        ];
        if (reason == 0) {
            if (success_ordinal >= planned_count) {
                transaction_status[0] = 2;
                birth_outputs.combined_status[cell_tick] = 2;
                return 2;
            }
            bool mutation_occurred = false;
            const int64_t tape_row = policy_cursor + success_ordinal;
            for (int64_t element = 0; element < packed_width; ++element) {
                const int64_t mutation_element =
                    birth_config.policy_mutation_tape_width == 1
                    ? 0
                    : element;
                mutation_occurred = mutation_occurred
                    || birth_state.policy_birth_mutation_mask_tape[
                        (cell * birth_config.policy_tape_capacity + tape_row)
                            * birth_config.policy_mutation_tape_width
                            + mutation_element
                    ];
            }
            required_event_rows += mutation_occurred ? 2 : 1;
            ++success_ordinal;
        } else if (
            (reason >= 1 && reason <= 3)
            || (
                birth_config.requires_mate
                && reason >= 5 && reason <= 6
            )
        ) {
            ++required_event_rows;
        } else if (
            (birth_config.requires_mate && reason == 4)
            || (reason >= 7 && reason <= 9)
            || (
                birth_config.requires_mate
                && reason >= 10 && reason <= 11
            )
        ) {
            continue;
        } else {
            transaction_status[0] = 2;
            birth_outputs.combined_status[cell_tick] = 2;
            return 2;
        }
    }
    const int64_t event_cursor = event_state.active_count[cell];
    birth_scratch_at<int64_t>(
        birth_scratch,
        cell,
        flattened_birth_scratch_v1::kRequiredEventRows
    )[0] = required_event_rows;
    if (
        success_ordinal != planned_count
        || !event_sequence_contract_valid_v1(event_state, cell)
        || event_cursor < 0 || event_cursor > event_state.capacity
        || required_event_rows > event_state.capacity - event_cursor
    ) {
        if (!event_sequence_contract_valid_v1(event_state, cell)) {
            fail_event_sequence_contract_v1(event_state, cell);
        }
        transaction_status[0] = 25;
        birth_outputs.combined_status[cell_tick] = 3;
        return 3;
    }

    for (int64_t slot = 0; slot < capacity; ++slot) {
        birth_state.repro_cooldown[cell_capacity_offset + slot] =
            birth_outputs.planned_repro_cooldown[cell_capacity_offset + slot];
    }
    commit_ordinary_birth_identity_serial(
        {
            slot_to_agent_id + cell_capacity_offset,
            agent_id_to_slot + cell_id_map_offset,
            slot_tokens + cell_capacity_offset,
            alive_mask + cell_capacity_offset,
            pos_y + cell_capacity_offset,
            pos_x + cell_capacity_offset,
            agent_grid + cell_grid_offset,
            energy + cell_capacity_offset,
            health + cell_capacity_offset,
            injury + cell_capacity_offset,
            birth_state.scar + cell_capacity_offset,
            full_tick_state.age + cell_capacity_offset,
            birth_state.repro_cooldown + cell_capacity_offset,
            birth_state.repro_threshold + cell_capacity_offset,
            birth_state.offspring_count + cell_capacity_offset,
            event_state.lineage_founder_id + cell_capacity_offset,
            event_state.lineage_depth + cell_capacity_offset,
            birth_state.lineage_birth_generation + cell_capacity_offset,
            birth_state.lineage_birth_tick + cell_capacity_offset,
            birth_state.patch_habitat_lineage_id + cell_capacity_offset,
            policy_energy_observation + cell_capacity_offset,
            survival_steps + cell_capacity_offset,
            rest_streak + cell_capacity_offset,
            action_cache_token + cell_capacity_offset,
            action_cache_cursor + cell_capacity_offset,
            action_cache_next + cell_capacity_offset,
            full_tick_state.slot_free_queue + cell_capacity_offset,
            full_tick_state.slot_free_queue_count + cell,
            birth_state.population_next_id + cell,
        },
        {
            birth_outputs.planned_parent_slots + tick_birth_offset,
            birth_outputs.planned_spawn_positions + tick_birth_offset * 2,
            birth_outputs.planned_count + cell_tick,
        },
        {
            birth_outputs.committed_parent_agent_ids + tick_birth_offset,
            birth_outputs.committed_child_agent_ids + tick_birth_offset,
            birth_outputs.committed_child_slots + tick_birth_offset,
            birth_outputs.identity_committed_count + cell_tick,
            active_agent_ids + cell_active_offset,
            active_agent_slots + cell_active_offset,
            active_count_state + cell,
            birth_outputs.identity_status + cell_tick,
        },
        {
            birth_capacity,
            capacity,
            agent_id_capacity,
            height,
            width,
            capacity,
            birth_parent_energy_share_for_cell_v1(birth_config, cell),
            static_cast<double>(birth_minimum_energy_buffer_for_cell_v1(birth_config, cell)),
            start_health,
            birth_cooldown_steps_for_cell_v1(birth_config, cell),
            static_cast<int32_t>(mechanics_config.generation),
            static_cast<int32_t>(absolute_tick),
        }
    );

    int32_t action_history_status = 0;
    reset_ordinary_newborn_action_history_serial(
        {
            last_action + cell_capacity_offset,
            last_last_action + cell_capacity_offset,
            action_switches + cell_capacity_offset,
            action_oscillations + cell_capacity_offset,
            episode0_switches + cell_capacity_offset,
            episode0_oscillations + cell_capacity_offset,
        },
        {
            birth_outputs.committed_child_slots + tick_birth_offset,
            birth_outputs.identity_committed_count + cell_tick,
            birth_outputs.identity_status + cell_tick,
        },
        {&action_history_status},
        {birth_capacity, capacity}
    );
    if (
        birth_outputs.identity_status[cell_tick] != 0
        || action_history_status != 0
    ) {
        birth_outputs.combined_status[cell_tick] = 2;
        return 2;
    }

    commit_ordinary_fixed_mlp_child_scalar_genome_with_trial_scratch_serial(
        {
            slot_to_agent_id + cell_capacity_offset,
            alive_mask + cell_capacity_offset,
            birth_state.genome_hidden_size + cell_capacity_offset,
            birth_state.genome_memory_dim + cell_capacity_offset,
            birth_state.genome_sensor_channel_mask + cell_capacity_offset,
            birth_state.genome_sensor_radius + cell_capacity_offset,
            birth_state.genome_introspection_channel_mask + cell_capacity_offset,
            birth_state.genome_self_regulation_control_mask + cell_capacity_offset,
            birth_state.genome_self_modification_target_mask + cell_capacity_offset,
            birth_state.genome_directed_inheritance_channel_mask + cell_capacity_offset,
            birth_state.policy_mem_dim + cell_capacity_offset,
            birth_state.policy_use_gru + cell_capacity_offset,
            birth_state.genome_exploration_bias + cell_capacity_offset,
            birth_state.genome_aggression_bias + cell_capacity_offset,
            birth_state.genome_mutation_rate + cell_capacity_offset,
            birth_state.genome_reproduction_threshold + cell_capacity_offset,
            birth_state.genome_action_temperature + cell_capacity_offset,
            birth_state.genome_drive_inertia + cell_capacity_offset,
            birth_state.genome_drive_noise + cell_capacity_offset,
            birth_state.genome_drive_logit_gain + cell_capacity_offset,
            birth_state.genome_pred_lr + cell_capacity_offset,
            birth_state.genome_drive_sp_hunger + cell_capacity_offset,
            birth_state.genome_drive_sp_pain + cell_capacity_offset,
            birth_state.genome_drive_sp_fatigue + cell_capacity_offset,
            birth_state.genome_drive_sp_fear + cell_capacity_offset,
            birth_state.genome_exp_prior_strength + cell_capacity_offset,
            birth_state.genome_exp_prior_noise + cell_capacity_offset,
            birth_state.genome_drive_birth_noise + cell_capacity_offset,
            birth_state.repro_threshold + cell_capacity_offset,
            full_tick_state.gru_upkeep_cost + cell_capacity_offset,
            attack_bias + cell_capacity_offset,
            action_temperature + cell_capacity_offset,
            drive_logit_gain + cell_capacity_offset,
            full_tick_state.drive_inertia + cell_capacity_offset,
            full_tick_state.drive_noise + cell_capacity_offset,
            birth_state.pred_lr + cell_capacity_offset,
        },
        {
            birth_outputs.planned_parent_slots + tick_birth_offset,
            birth_outputs.committed_child_slots + tick_birth_offset,
            birth_outputs.identity_committed_count + cell_tick,
        },
        {
            birth_state.python_random_words + cell * 624,
            birth_state.python_random_index + cell,
            birth_state.python_random_has_gauss + cell,
            birth_state.python_random_gauss_next + cell,
            birth_state.python_random_gauss_primary_tape
                + cell * birth_config.gauss_tape_capacity,
            birth_state.python_random_gauss_secondary_tape
                + cell * birth_config.gauss_tape_capacity,
            birth_scratch_at<int64_t>(
                birth_scratch,
                cell,
                kScratchPythonRawDraws
            ),
            birth_state.mutation_py_semantic_draws + cell,
        },
        {
            birth_outputs.scalar_committed_count + cell_tick,
            birth_outputs.scalar_semantic_draws + cell_tick,
            birth_outputs.scalar_status + cell_tick,
        },
        {
            birth_capacity,
            capacity,
            birth_config.gauss_tape_capacity,
            birth_config.sensor_radius_max,
        },
        birth_scratch_at<uint32_t>(
            birth_scratch,
            cell,
            kScratchPythonTrialWords
        )
    );
    if (birth_outputs.scalar_status[cell_tick] != 0) {
        birth_outputs.combined_status[cell_tick] = 2;
        return 2;
    }
    int64_t absolute_python_raw_draws = 0;
    if (!absolute_cursor_from_physical_v1(
            birth_scratch_at<int64_t>(
                birth_scratch,
                cell,
                kScratchPythonRawDraws
            )[0],
            birth_state.python_random_raw_draw_base[cell],
            birth_config.gauss_tape_capacity,
            &absolute_python_raw_draws
        )) {
        birth_outputs.combined_status[cell_tick] = 2;
        return 2;
    }
    birth_state.python_random_raw_draws[cell] = absolute_python_raw_draws;

    uint64_t* drive_population_state = birth_scratch_at<uint64_t>(
        birth_scratch,
        cell,
        flattened_birth_scratch_v1::kPopulationPcgState
    );
    int64_t* drive_population_buffer = birth_scratch_at<int64_t>(
        birth_scratch,
        cell,
        flattened_birth_scratch_v1::kPopulationPcgBuffer
    );
    int64_t* drive_population_semantic = birth_scratch_at<int64_t>(
        birth_scratch,
        cell,
        flattened_birth_scratch_v1::kPopulationSemanticDraws
    );
    for (int64_t word = 0; word < 4; ++word) {
        drive_population_state[word] =
            birth_state.population_pcg64_state_words[cell * 4 + word];
    }
    for (int64_t word = 0; word < 2; ++word) {
        drive_population_buffer[word] =
            birth_state.population_pcg64_buffer_words[cell * 2 + word];
    }
    drive_population_semantic[0] =
        birth_state.mutation_np_semantic_draws[cell];

    float* scratch_drives = birth_scratch_at<float>(
        birth_scratch,
        cell,
        flattened_birth_scratch_v1::kMiniDrives
    );
    float* scratch_birth_shock = birth_scratch_at<float>(
        birth_scratch,
        cell,
        flattened_birth_scratch_v1::kMiniBirthShock
    );
    int64_t* scratch_drive_committed = birth_scratch_at<int64_t>(
        birth_scratch,
        cell,
        flattened_birth_scratch_v1::kMiniDriveCommitted
    );
    int64_t* scratch_birth_noise_rows = birth_scratch_at<int64_t>(
        birth_scratch,
        cell,
        flattened_birth_scratch_v1::kMiniBirthNoiseRows
    );
    int64_t* scratch_exp_prior_rows = birth_scratch_at<int64_t>(
        birth_scratch,
        cell,
        flattened_birth_scratch_v1::kMiniExpPriorRows
    );
    int64_t* scratch_standard_normal_rows = birth_scratch_at<int64_t>(
        birth_scratch,
        cell,
        flattened_birth_scratch_v1::kMiniStandardNormalRows
    );
    int64_t* scratch_drive_semantic = birth_scratch_at<int64_t>(
        birth_scratch,
        cell,
        flattened_birth_scratch_v1::kMiniDriveSemanticDraws
    );
    int32_t* scratch_drive_status = birth_scratch_at<int32_t>(
        birth_scratch,
        cell,
        flattened_birth_scratch_v1::kMiniDriveStatus
    );
    scratch_drive_committed[0] = 0;
    scratch_birth_noise_rows[0] = 0;
    scratch_exp_prior_rows[0] = 0;
    scratch_standard_normal_rows[0] = 0;
    scratch_drive_semantic[0] = 0;
    scratch_drive_status[0] = 0;
    flattened_birth_drives_v1::
        prepare_ordinary_newborn_drives_external_scratch_v2(
            slot_to_agent_id + cell_capacity_offset,
            alive_mask + cell_capacity_offset,
            birth_state.policy_use_gru + cell_capacity_offset,
            policy_energy_observation + cell_capacity_offset,
            birth_state.genome_drive_sp_hunger + cell_capacity_offset,
            birth_state.genome_drive_sp_pain + cell_capacity_offset,
            birth_state.genome_drive_sp_fatigue + cell_capacity_offset,
            birth_state.genome_drive_sp_fear + cell_capacity_offset,
            birth_state.genome_exp_prior_noise + cell_capacity_offset,
            birth_state.genome_drive_birth_noise + cell_capacity_offset,
            birth_outputs.planned_parent_slots + tick_birth_offset,
            birth_outputs.committed_child_slots + tick_birth_offset,
            birth_outputs.identity_committed_count[cell_tick],
            capacity,
            max_energy,
            drive_population_state,
            drive_population_buffer,
            drive_population_semantic[0],
            scratch_drives,
            scratch_birth_shock,
            scratch_drive_committed[0],
            scratch_birth_noise_rows[0],
            scratch_exp_prior_rows[0],
            scratch_standard_normal_rows[0],
            scratch_drive_semantic[0],
            scratch_drive_status[0]
        );
    birth_outputs.drive_status[cell_tick] = scratch_drive_status[0];
    if (scratch_drive_status[0] == 0) {
        const int64_t committed_births = scratch_drive_committed[0];
        for (int64_t birth = 0; birth < committed_births; ++birth) {
            const int64_t child_slot =
                birth_outputs.committed_child_slots[tick_birth_offset + birth];
            for (int64_t drive = 0; drive < 4; ++drive) {
                drives[(cell_capacity_offset + child_slot) * drive_width + drive] =
                    scratch_drives[birth * 4 + drive];
            }
            birth_state.birth_reset_shock[cell_capacity_offset + child_slot] =
                scratch_birth_shock[birth];
        }
        for (int64_t word = 0; word < 4; ++word) {
            birth_state.population_pcg64_state_words[cell * 4 + word] =
                drive_population_state[word];
        }
        for (int64_t word = 0; word < 2; ++word) {
            birth_state.population_pcg64_buffer_words[cell * 2 + word] =
                drive_population_buffer[word];
        }
        birth_state.mutation_np_semantic_draws[cell] =
            drive_population_semantic[0];
        birth_outputs.drive_committed_count[cell_tick] =
            scratch_drive_committed[0];
        birth_outputs.birth_noise_rows_consumed[cell_tick] =
            scratch_birth_noise_rows[0];
        birth_outputs.exp_prior_rows_consumed[cell_tick] =
            scratch_exp_prior_rows[0];
        birth_outputs.standard_normal_rows[cell_tick] =
            scratch_standard_normal_rows[0];
        birth_outputs.drive_semantic_draws[cell_tick] =
            scratch_drive_semantic[0];
    }
    birth_outputs.chemistry_status[cell_tick] = 0;
    const int64_t chemistry_birth_count =
        birth_outputs.identity_committed_count[cell_tick];
    if (
        chemistry_birth_count < 0 || chemistry_birth_count > birth_capacity
        || chemistry_birth_count > capacity || !::isfinite(max_energy)
        || max_energy <= 0.0f
    ) {
        birth_outputs.chemistry_status[cell_tick] =
            persistent_birth_drives::kStatusInvalidBirthPlan;
    }
    for (
        int64_t birth = 0;
        birth < chemistry_birth_count
            && birth_outputs.chemistry_status[cell_tick] == 0;
        ++birth
    ) {
        const int64_t child_slot =
            birth_outputs.committed_child_slots[tick_birth_offset + birth];
        const int64_t child_index = cell_capacity_offset + child_slot;
        if (
            child_slot < 0 || child_slot >= capacity || !alive_mask[child_index]
            || slot_to_agent_id[child_index] < 0
            || !::isfinite(policy_energy_observation[child_index])
        ) {
            birth_outputs.chemistry_status[cell_tick] =
                persistent_birth_drives::kStatusInvalidChildScalarState;
        }
    }
    if (birth_outputs.chemistry_status[cell_tick] == 0) {
        const double max_energy_f64 = static_cast<double>(max_energy);
        for (int64_t birth = 0; birth < chemistry_birth_count; ++birth) {
            const int64_t child_slot =
                birth_outputs.committed_child_slots[tick_birth_offset + birth];
            const int64_t child_index = cell_capacity_offset + child_slot;
            const double energy_n = persistent_chemistry::detail::clamp_double(
                policy_energy_observation[child_index] / max_energy_f64,
                0.0,
                1.0
            );
            const int64_t chemistry_offset = child_index * chemistry_width;
            full_tick_state.chemistry[chemistry_offset] =
                static_cast<float>(0.40 + 0.40 * energy_n);
            full_tick_state.chemistry[chemistry_offset + 1] = 0.55f;
            full_tick_state.chemistry[chemistry_offset + 2] = 0.0f;
            full_tick_state.chemistry[chemistry_offset + 3] = 0.0f;
            full_tick_state.chemistry[chemistry_offset + 4] = 0.50f;
            full_tick_state.chemistry[chemistry_offset + 5] = 0.25f;
        }
    }
    if (
        birth_outputs.drive_status[cell_tick] != 0
        || birth_outputs.chemistry_status[cell_tick] != 0
    ) {
        birth_outputs.combined_status[cell_tick] = 2;
        return 2;
    }

    inherit_ordinary_fixed_mlp_child_policy_serial(
        {
            slot_to_agent_id + cell_capacity_offset,
            alive_mask + cell_capacity_offset,
            mlp_pack + cell_capacity_offset * packed_width,
            birth_state.hmem + cell_capacity_offset * birth_config.hmem_width,
            birth_state.policy_use_gru + cell_capacity_offset,
            birth_state.mutation_torch_cpu_state
                + cell * birth_config.torch_state_bytes,
            birth_state.mutation_torch_cpu_semantic_draws + cell,
        },
        {
            birth_outputs.planned_parent_slots + tick_birth_offset,
            birth_outputs.committed_child_slots + tick_birth_offset,
            birth_outputs.identity_committed_count + cell_tick,
        },
        {
            birth_state.policy_birth_delta_tape
                + cell * birth_config.policy_tape_capacity
                    * birth_config.policy_mutation_tape_width,
            birth_state.policy_birth_mutation_mask_tape
                + cell * birth_config.policy_tape_capacity
                    * birth_config.policy_mutation_tape_width,
            birth_state.policy_birth_post_rng_state_packed_tape == nullptr
                ? nullptr
                : birth_state.policy_birth_post_rng_state_packed_tape
                    + cell * birth_config.policy_tape_capacity
                        * torch_rng_pack::kPackedStateBytes,
            birth_state.policy_birth_semantic_draws_tape
                + cell * birth_config.policy_tape_capacity,
            birth_scratch_at<int64_t>(
                birth_scratch,
                cell,
                flattened_birth_scratch_v1::kMiniPolicyCursor
            ),
        },
        {
            birth_outputs.policy_committed_count + cell_tick,
            birth_outputs.policy_semantic_draws + cell_tick,
            birth_outputs.policy_mutation_events + cell_tick,
            birth_outputs.policy_mutated_elements + cell_tick,
            birth_outputs.policy_mutation_delta_total + cell_tick,
            birth_outputs.policy_mutation_occurred + tick_birth_offset,
            birth_outputs.policy_mutation_delta + tick_birth_offset,
            birth_outputs.policy_status + cell_tick,
        },
        {
            birth_capacity,
            capacity,
            packed_width,
            birth_config.hmem_width,
            birth_config.torch_state_bytes,
            birth_config.policy_tape_capacity,
            birth_config.policy_mutation_tape_width,
        }
    );
    if (birth_outputs.policy_status[cell_tick] != 0) {
        birth_outputs.combined_status[cell_tick] = 2;
        transaction_status[0] = 2;
        return 2;
    }
    int64_t absolute_policy_cursor = 0;
    if (!absolute_cursor_from_physical_v1(
            birth_scratch_at<int64_t>(
                birth_scratch,
                cell,
                flattened_birth_scratch_v1::kMiniPolicyCursor
            )[0],
            birth_state.policy_birth_tape_cursor_base[cell],
            birth_config.policy_tape_capacity,
            &absolute_policy_cursor
        )) {
        birth_outputs.combined_status[cell_tick] = 2;
        transaction_status[0] = 2;
        return 2;
    }
    birth_state.policy_birth_tape_cursor[cell] = absolute_policy_cursor;

    uint64_t* planner_pcg_state = birth_scratch_at<uint64_t>(
        birth_scratch,
        cell,
        kScratchPlannerPcgState
    );
    int64_t* planner_pcg_buffer = birth_scratch_at<int64_t>(
        birth_scratch,
        cell,
        kScratchPlannerPcgBuffer
    );
    uint64_t* runner_pcg_state = birth_scratch_at<uint64_t>(
        birth_scratch,
        cell,
        kScratchRunnerPcgState
    );
    int64_t* runner_pcg_buffer = birth_scratch_at<int64_t>(
        birth_scratch,
        cell,
        kScratchRunnerPcgBuffer
    );
    for (int64_t word = 0; word < 4; ++word) {
        full_tick_state.shared_env_policy_pcg64_state_words[cell * 4 + word] =
            planner_pcg_state[word];
        full_tick_state.runner_pcg64_state_words[cell * 4 + word] =
            runner_pcg_state[word];
    }
    for (int64_t word = 0; word < 2; ++word) {
        full_tick_state.shared_env_policy_pcg64_buffer_words[cell * 2 + word] =
            planner_pcg_buffer[word];
        full_tick_state.runner_pcg64_buffer_words[cell * 2 + word] =
            runner_pcg_buffer[word];
    }
    full_tick_state.policy_np_semantic_draws[cell] =
        birth_scratch_at<int64_t>(
            birth_scratch,
            cell,
            kScratchPlannerPolicyDraws
        )[0];
    full_tick_state.metrics_np_semantic_draws[cell] =
        birth_scratch_at<int64_t>(
            birth_scratch,
            cell,
            kScratchRunnerMetricDraws
        )[0];

    const persistent_lifecycle_events::FlatEventTapeColumns columns{
        {event_state.event_type, event_state.capacity, 1},
        {event_state.event_generation, event_state.capacity, 1},
        {event_state.event_tick, event_state.capacity, 1},
        {event_state.event_actor_id, event_state.capacity, 1},
        {event_state.event_target_id, event_state.capacity, 1},
        {event_state.event_value, event_state.capacity, 1},
        {event_state.typed_event_type, event_state.capacity, 1},
        {event_state.typed_generation, event_state.capacity, 1},
        {event_state.typed_tick, event_state.capacity, 1},
        {event_state.typed_sequence, event_state.capacity, 1},
        {event_state.typed_actor_id, event_state.capacity, 1},
        {event_state.typed_target_id, event_state.capacity, 1},
        {event_state.typed_actor_slot, event_state.capacity, 1},
        {event_state.typed_target_slot, event_state.capacity, 1},
        {event_state.typed_parent_id, event_state.capacity, 1},
        {event_state.typed_child_id, event_state.capacity, 1},
        {event_state.typed_founder_id, event_state.capacity, 1},
        {event_state.typed_lineage_depth, event_state.capacity, 1},
        {event_state.typed_subtype, event_state.capacity, 1},
        {event_state.typed_reason, event_state.capacity, 1},
        {event_state.typed_success, event_state.capacity, 1},
        {event_state.typed_value, event_state.capacity, 1},
    };
    const persistent_lifecycle_events::FlatEventTapeState tape{
        {event_state.active_count, 1},
        {event_state.status, 1},
        event_state.capacity,
    };
    event_state.status[cell] = persistent_birth_events::kStatusOk;
    append_preflighted_ordinary_birth_events_serial(
        columns,
        tape,
        {
            {slot_to_agent_id, capacity, 1},
            {event_state.lineage_founder_id, capacity, 1},
            {event_state.lineage_depth, capacity, 1},
            capacity,
        },
        {
            {birth_outputs.decision_parent_slots + output_tick * capacity,
             output_ticks * capacity, 1},
            {birth_outputs.decision_reason_codes + output_tick * capacity,
             output_ticks * capacity, 1},
            {birth_outputs.decision_count + output_tick, output_ticks},
            capacity,
        },
        {
            {birth_outputs.committed_parent_agent_ids
                 + output_tick * birth_capacity,
             output_ticks * birth_capacity, 1},
            {birth_outputs.committed_child_agent_ids
                 + output_tick * birth_capacity,
             output_ticks * birth_capacity, 1},
            {birth_outputs.planned_parent_slots
                 + output_tick * birth_capacity,
             output_ticks * birth_capacity, 1},
            {birth_outputs.committed_child_slots
                 + output_tick * birth_capacity,
             output_ticks * birth_capacity, 1},
            {birth_outputs.policy_mutation_occurred
                 + output_tick * birth_capacity,
             output_ticks * birth_capacity, 1},
            {birth_outputs.policy_mutation_delta
                 + output_tick * birth_capacity,
             output_ticks * birth_capacity, 1},
            {birth_outputs.planned_child_energy
                 + output_tick * birth_capacity,
             output_ticks * birth_capacity, 1},
            {birth_outputs.identity_committed_count + output_tick,
             output_ticks},
            birth_capacity,
        },
        cell,
        mechanics_config.generation,
        absolute_tick
    );
    if (
        event_state.status[cell] == persistent_birth_events::kStatusOk
        && !publish_absolute_event_sequences_v1(
            event_state,
            cell,
            event_cursor,
            event_state.active_count[cell]
        )
    ) {
        birth_outputs.combined_status[cell_tick] = 3;
        return 3;
    }
    if (event_state.status[cell] != 0) {
        birth_outputs.combined_status[cell_tick] = 3;
        return 3;
    }

    const FlatFreshNewbornSelectionState newborn_state{
        {energy_collected + cell_capacity_offset, 1},
        {energy_collected_exact + cell_capacity_offset, 1},
        {birth_state.attacks + cell_capacity_offset, 1},
        {birth_state.early_survival_steps + cell_capacity_offset, 1},
        {birth_state.early_energy_collected + cell_capacity_offset, 1},
        {birth_state.early_energy_collected_exact + cell_capacity_offset, 1},
        {birth_state.early_injury_gained + cell_capacity_offset, 1},
        {birth_state.early_life_weight + cell_capacity_offset, 1},
        {birth_state.pred_err_sum + cell_capacity_offset, 1},
        {birth_state.pred_err_count + cell_capacity_offset, 1},
        {birth_state.first_repro_step + cell_capacity_offset, 1},
        {birth_state.dev_stage_initialized + cell_capacity_offset, 1},
        {birth_state.episode0_pred_err_sum + cell_capacity_offset, 1},
        {birth_state.episode0_pred_err_count + cell_capacity_offset, 1},
        {birth_state.late_pred_err_sum + cell_capacity_offset, 1},
        {birth_state.late_pred_err_count + cell_capacity_offset, 1},
        {birth_state.chemistry_observation_fidelity + cell_capacity_offset, 1},
        {birth_state.chemistry_observation_bias + cell_capacity_offset, 1},
        {action_cache_actions + cell * capacity * cache_capacity,
         cache_capacity, 1},
        {full_tick_state.prev_drive_delta
             + cell_capacity_offset * drive_width,
         drive_width, 1},
        {full_tick_state.drive_volatility + cell_capacity_offset, 1},
        {full_tick_state.drive_jitter + cell_capacity_offset, 1},
        {full_tick_state.episode0_drive_jitter_sum + cell_capacity_offset, 1},
        {full_tick_state.last_energy_delta + cell_capacity_offset, 1},
        {full_tick_state.last_health_delta + cell_capacity_offset, 1},
        {full_tick_state.last_injury_delta + cell_capacity_offset, 1},
        {full_tick_state.selfharm_events + cell_capacity_offset, 1},
        {full_tick_state.episode0_selfharm_events + cell_capacity_offset, 1},
    };
    const int64_t committed_count =
        birth_outputs.identity_committed_count[cell_tick];
    for (int64_t birth = 0; birth < committed_count; ++birth) {
        const int64_t child_slot =
            birth_outputs.committed_child_slots[tick_birth_offset + birth];
        reset_ordinary_newborn_selection_state(
            newborn_state,
            child_slot,
            {cache_capacity, drive_width}
        );
        const int64_t parent_slot =
            birth_outputs.planned_parent_slots[tick_birth_offset + birth];
        if (birth_state.first_repro_step[cell_capacity_offset + parent_slot] < 0) {
            birth_state.first_repro_step[cell_capacity_offset + parent_slot] =
                survival_steps[cell_capacity_offset + parent_slot];
        }
    }
    transaction_status[0] = 0;
    return 0;
}

struct Phase2KernelArguments {
    int64_t* active_agent_ids;
    int32_t* active_agent_slots;
    int64_t* active_count_state;
    int64_t* slot_to_agent_id;
    int32_t* agent_id_to_slot;
    int64_t* slot_tokens;
    bool* alive_mask;
    int32_t* pos_y;
    int32_t* pos_x;
    int32_t* agent_grid;
    float* energy;
    float* health;
    float* injury;
    double* policy_energy_observation;
    int32_t* rest_streak;
    int32_t* action_cache_actions;
    int64_t* action_cache_token;
    int32_t* action_cache_cursor;
    int32_t* action_cache_next;
    int32_t* last_action;
    int32_t* last_last_action;
    int32_t* action_switches;
    int32_t* action_oscillations;
    int32_t* episode0_switches;
    int32_t* episode0_oscillations;
    int32_t* survival_steps;
    float* mlp_pack;
    float* drives;
    float* attack_bias;
    float* action_temperature;
    float* drive_logit_gain;
    float* energy_collected;
    double* energy_collected_exact;
    int64_t* survivor_agent_ids;
    int64_t* survivor_count;
    int32_t* lifecycle_status;
    int32_t* status;
    int64_t* first_failure_records;
    double* generation_summary_scalar_out;
    FlatFullTickState full_tick_state;
    FlatFullTickOutputs full_tick_outputs;
    FlatEventState event_state;
    FlatOrdinaryBirthState birth_state;
    FlatOrdinaryBirthOutputs birth_outputs;
    FlatFlattenedBirthScratch birth_scratch;
    FlatOrdinaryBirthConfig birth_config;
    FlatMechanicsConfig mechanics_config;
    FlatCompactSummaryContextV1 compact_summary_context;
    FlatPatchP2Context patch_p2_context;
    int64_t ticks;
    int64_t output_ticks_per_cell;
    int64_t active_capacity;
    int64_t agent_id_capacity;
    int64_t capacity;
    int64_t height;
    int64_t width;
    int64_t cache_capacity;
    int64_t cache_width;
    int64_t drive_width;
    int64_t packed_width;
    float max_energy;
    float start_health;
};

constexpr int64_t kFlattenedPreparedStamp = 0x464c415442495254LL;
constexpr int64_t kRuntimeBirthScratchPrefixBytesV2 = kBirthScratchBytesPerCell;

static_assert(
    kRuntimeBirthScratchPrefixBytesV2 == kBirthScratchBytesPerCell
);

__device__ __forceinline__ bool make_runtime_birth_scratch_access_v2(
    const Phase2KernelArguments arguments,
    flattened_birth_scratch_v2::FlatBirthScratchLayoutV2* layout,
    flattened_birth_scratch_v2::FlatBirthScratchArenaV2* arena
) {
    namespace scratch = flattened_birth_scratch_v2;
    *layout = scratch::make_layout(
        arguments.birth_config.max_births_per_tick,
        arguments.capacity,
        arguments.packed_width,
        arguments.birth_config.hmem_width,
        arguments.birth_config.torch_state_bytes
    );
    if (
        arguments.birth_scratch.storage == nullptr
        || arguments.birth_scratch.bytes_per_cell
            < kRuntimeBirthScratchPrefixBytesV2
        || !layout->ok()
        || layout->required_bytes
            > arguments.birth_scratch.bytes_per_cell
                - kRuntimeBirthScratchPrefixBytesV2
    ) {
        return false;
    }
    *arena = {
        arguments.birth_scratch.storage + kRuntimeBirthScratchPrefixBytesV2,
        arguments.birth_scratch.bytes_per_cell,
    };
    return arena->holds(*layout);
}

__device__ __forceinline__ double effective_birth_energy_v1(
    const Phase2KernelArguments arguments,
    const int64_t cell_capacity_offset,
    const int64_t slot
) {
    const double observed =
        arguments.policy_energy_observation[cell_capacity_offset + slot];
    return ::isfinite(observed)
        ? observed
        : static_cast<double>(
            arguments.energy[cell_capacity_offset + slot]
        );
}

// The established identity preflight owns child slots and IDs. This additional
// read-only gate validates the selected second parent and replaces only the
// staged child-energy row with the two-parent ordinary-birth sum.
__device__ __forceinline__ int32_t
validate_mate_identity_and_stage_energy_v1(
    const Phase2KernelArguments arguments,
    const int64_t cell,
    const int64_t cell_tick,
    const int64_t birth_count,
    double* const staged_child_energy
) {
    if (!arguments.birth_config.requires_mate) {
        return 0;
    }
    if (
        staged_child_energy == nullptr || birth_count < 0
        || birth_count > arguments.birth_config.max_births_per_tick
        || birth_count > arguments.capacity
    ) {
        return 210;
    }
    const int64_t cell_capacity_offset = cell * arguments.capacity;
    const int64_t tick_birth_offset =
        cell_tick * arguments.birth_config.max_births_per_tick;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t output_row = tick_birth_offset + birth;
        const int64_t parent_slot =
            arguments.birth_outputs.planned_parent_slots[output_row];
        const int64_t mate_slot =
            arguments.birth_outputs.planned_mate_slots[output_row];
        if (
            parent_slot < 0 || parent_slot >= arguments.capacity
            || mate_slot < 0 || mate_slot >= arguments.capacity
            || mate_slot == parent_slot
        ) {
            return 210;
        }
        const int64_t parent_index = cell_capacity_offset + parent_slot;
        const int64_t mate_index = cell_capacity_offset + mate_slot;
        const int32_t distance =
            abs(arguments.pos_y[parent_index] - arguments.pos_y[mate_index])
            + abs(arguments.pos_x[parent_index] - arguments.pos_x[mate_index]);
        const double parent_energy = effective_birth_energy_v1(
            arguments, cell_capacity_offset, parent_slot
        );
        const double mate_energy = effective_birth_energy_v1(
            arguments, cell_capacity_offset, mate_slot
        );
        const double parent_share =
            parent_energy * birth_parent_energy_share_for_cell_v1(arguments.birth_config, cell);
        const double mate_share =
            mate_energy * birth_parent_energy_share_for_cell_v1(arguments.birth_config, cell);
        const double expected_child_energy = parent_share + mate_share;
        if (
            distance != 1
            || !arguments.alive_mask[parent_index]
            || !arguments.alive_mask[mate_index]
            || arguments.slot_to_agent_id[parent_index] < 0
            || arguments.slot_to_agent_id[mate_index] < 0
            || arguments.health[parent_index] <= 0.0f
            || arguments.health[mate_index] <= 0.0f
            || arguments.birth_outputs.planned_repro_cooldown[parent_index]
                != 0
            || arguments.birth_outputs.planned_repro_cooldown[mate_index] != 0
            || !::isfinite(parent_energy) || !::isfinite(mate_energy)
            || parent_energy < static_cast<double>(
                arguments.birth_state.repro_threshold[parent_index]
            )
            || mate_energy < static_cast<double>(
                arguments.birth_state.repro_threshold[mate_index]
            )
            || parent_energy - parent_share
                < static_cast<double>(
                    birth_minimum_energy_buffer_for_cell_v1(arguments.birth_config, cell)
                )
            || mate_energy - mate_share
                < static_cast<double>(
                    birth_minimum_energy_buffer_for_cell_v1(arguments.birth_config, cell)
                )
            || static_cast<float>(expected_child_energy)
                != arguments.birth_outputs.planned_child_energy[output_row]
        ) {
            return 211;
        }
        for (int64_t prior = 0; prior < birth; ++prior) {
            const int64_t prior_row = tick_birth_offset + prior;
            const int64_t prior_parent =
                arguments.birth_outputs.planned_parent_slots[prior_row];
            const int64_t prior_mate =
                arguments.birth_outputs.planned_mate_slots[prior_row];
            if (
                parent_slot == prior_parent || parent_slot == prior_mate
                || mate_slot == prior_parent || mate_slot == prior_mate
            ) {
                return 212;
            }
        }
        staged_child_energy[birth] = expected_child_energy;
    }
    return 0;
}

__device__ __forceinline__ int32_t prepare_flattened_b2_transaction_v2(
    const Phase2KernelArguments arguments,
    const int64_t cell,
    const int64_t controller_tick,
    const int64_t physical_policy_cursor,
    const int64_t physical_python_raw_draws
) {
    namespace scratch = flattened_birth_scratch_v1;
    namespace identity_v2 = flattened_birth_identity_v2;
    namespace scalar_v2 = flattened_birth_scalar_genome_v2;
    namespace policy_v2 = flattened_birth_policy_v2;
    namespace events_v2 = flattened_birth_events_v2;
    namespace chemistry_detail = persistent_chemistry::detail;

    const int64_t output_tick = arguments.output_ticks_per_cell == 1
        ? 0
        : controller_tick;
    const int64_t cell_tick =
        cell * arguments.output_ticks_per_cell + output_tick;
    const int64_t cell_capacity_offset = cell * arguments.capacity;
    const int64_t cell_grid_offset = cell * arguments.height * arguments.width;
    const int64_t tick_capacity_offset = cell_tick * arguments.capacity;
    const int64_t tick_birth_offset = cell_tick * 2;
    const int64_t birth_count = arguments.birth_outputs.planned_count[cell_tick];
    const int64_t decision_count = arguments.birth_outputs.decision_count[cell_tick];
    const auto layout = scratch::make_b2_dynamic_layout(
        arguments.packed_width,
        arguments.birth_config.hmem_width,
        arguments.birth_config.torch_state_bytes
    );
    if (
        layout.required_bytes > arguments.birth_scratch.bytes_per_cell
        || layout.required_bytes > scratch::kBytesPerCell
    ) {
        return 108;
    }

    int64_t* identity_parent_slots = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kB2IdentityParentSlots
    );
    int64_t* identity_child_slots = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kB2IdentityChildSlots
    );
    int64_t* identity_child_ids = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kB2IdentityChildIds
    );
    double* identity_child_energy = scratch::at<double>(
        arguments.birth_scratch, cell, scratch::kB2IdentityChildEnergy
    );
    int64_t* identity_count = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kB2IdentityCount
    );
    int64_t* queue_head = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kB2QueueHead
    );
    int64_t* queue_count = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kB2QueueCount
    );
    const auto identity_status =
        identity_v2::preflight_ordinary_birth_identity_external_scratch_v2(
            {
                arguments.slot_to_agent_id + cell_capacity_offset,
                arguments.alive_mask + cell_capacity_offset,
                arguments.agent_grid + cell_grid_offset,
                arguments.energy + cell_capacity_offset,
                arguments.health + cell_capacity_offset,
                arguments.birth_outputs.planned_repro_cooldown
                    + cell_capacity_offset,
                arguments.birth_state.repro_threshold + cell_capacity_offset,
                arguments.policy_energy_observation + cell_capacity_offset,
                arguments.full_tick_state.slot_free_queue + cell_capacity_offset,
                arguments.full_tick_state.slot_free_queue_count + cell,
                arguments.birth_state.population_next_id + cell,
            },
            {
                arguments.birth_outputs.planned_parent_slots + tick_birth_offset,
                arguments.birth_outputs.planned_spawn_positions
                    + tick_birth_offset * 2,
                arguments.birth_outputs.planned_count + cell_tick,
            },
            {
                identity_parent_slots,
                identity_child_slots,
                identity_child_ids,
                identity_child_energy,
                identity_count,
                queue_head,
                queue_count,
            },
            {
                2,
                arguments.capacity,
                arguments.agent_id_capacity,
                arguments.height,
                arguments.width,
                arguments.capacity,
                birth_parent_energy_share_for_cell_v1(arguments.birth_config, cell),
                static_cast<double>(birth_minimum_energy_buffer_for_cell_v1(arguments.birth_config, cell)),
                arguments.start_health,
                birth_cooldown_steps_for_cell_v1(arguments.birth_config, cell),
                static_cast<int32_t>(arguments.mechanics_config.generation),
                static_cast<int32_t>(
                    arguments.mechanics_config.start_tick + controller_tick
                ),
            }
        );
    if (identity_status != persistent_birth_identity::kStatusOk) {
        return identity_status == persistent_birth_identity::kStatusAgentIdCapacity
            ? 20
            : 200 + static_cast<int32_t>(identity_status);
    }
    const int32_t mate_identity_status =
        validate_mate_identity_and_stage_energy_v1(
            arguments,
            cell,
            cell_tick,
            birth_count,
            identity_child_energy
        );
    if (mate_identity_status != 0) {
        return mate_identity_status;
    }

    int64_t* mini_slot_ids = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kB2MiniSlotIds
    );
    bool* mini_alive = scratch::at<bool>(
        arguments.birth_scratch, cell, scratch::kB2MiniAlive
    );
    int32_t* mini_i32 = scratch::at<int32_t>(
        arguments.birth_scratch, cell, scratch::kB2MiniGenomeInt32
    );
    int64_t* mini_mem_dim = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kB2MiniPolicyMemDim
    );
    bool* mini_use_gru = scratch::at<bool>(
        arguments.birth_scratch, cell, scratch::kB2MiniPolicyUseGru
    );
    float* mini_f32 = scratch::at<float>(
        arguments.birth_scratch, cell, scratch::kB2MiniGenomeFloat32
    );
    double* mini_energy = scratch::at<double>(
        arguments.birth_scratch, cell, scratch::kB2MiniEnergy
    );
    int64_t* mini_plan_parent = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kB2MiniPlanParent
    );
    int64_t* mini_plan_child = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kB2MiniPlanChild
    );
    int64_t* mini_plan_count = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kB2MiniPlanCount
    );
    for (int64_t slot = 0; slot < 4; ++slot) {
        mini_slot_ids[slot] = -1;
        mini_alive[slot] = false;
        mini_mem_dim[slot] = 0;
        mini_use_gru[slot] = false;
        mini_energy[slot] = 0.0;
    }
    for (int64_t index = 0; index < 8 * 4; ++index) {
        mini_i32[index] = 0;
    }
    for (int64_t index = 0; index < 24 * 4; ++index) {
        mini_f32[index] = 0.0f;
    }
    mini_plan_parent[0] = 0;
    mini_plan_parent[1] = 2;
    mini_plan_child[0] = 1;
    mini_plan_child[1] = 3;
    mini_plan_count[0] = birth_count;

    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t mini_parent = birth * 2;
        const int64_t mini_child = mini_parent + 1;
        const int64_t parent_slot = identity_parent_slots[birth];
        const int64_t parent_index = cell_capacity_offset + parent_slot;
        mini_slot_ids[mini_parent] = arguments.slot_to_agent_id[parent_index];
        mini_slot_ids[mini_child] = identity_child_ids[birth];
        mini_alive[mini_parent] = true;
        mini_alive[mini_child] = true;
        mini_mem_dim[mini_parent] = arguments.birth_state.policy_mem_dim[parent_index];
        mini_use_gru[mini_parent] = arguments.birth_state.policy_use_gru[parent_index];
        mini_energy[mini_parent] = ::isfinite(
            arguments.policy_energy_observation[parent_index]
        )
            ? arguments.policy_energy_observation[parent_index]
            : static_cast<double>(arguments.energy[parent_index]);
        mini_energy[mini_child] = identity_child_energy[birth];

#define EIEL_B2_COPY_PARENT_I32(index, source)                                   \
        mini_i32[(index) * 4 + mini_parent] = (source)[parent_index]
        EIEL_B2_COPY_PARENT_I32(0, arguments.birth_state.genome_hidden_size);
        EIEL_B2_COPY_PARENT_I32(1, arguments.birth_state.genome_memory_dim);
        EIEL_B2_COPY_PARENT_I32(2, arguments.birth_state.genome_sensor_channel_mask);
        EIEL_B2_COPY_PARENT_I32(3, arguments.birth_state.genome_sensor_radius);
        EIEL_B2_COPY_PARENT_I32(4, arguments.birth_state.genome_introspection_channel_mask);
        EIEL_B2_COPY_PARENT_I32(5, arguments.birth_state.genome_self_regulation_control_mask);
        EIEL_B2_COPY_PARENT_I32(6, arguments.birth_state.genome_self_modification_target_mask);
        EIEL_B2_COPY_PARENT_I32(7, arguments.birth_state.genome_directed_inheritance_channel_mask);
#undef EIEL_B2_COPY_PARENT_I32

#define EIEL_B2_COPY_PARENT_F32(index, source)                                   \
        mini_f32[(index) * 4 + mini_parent] = (source)[parent_index]
        EIEL_B2_COPY_PARENT_F32(0, arguments.birth_state.genome_exploration_bias);
        EIEL_B2_COPY_PARENT_F32(1, arguments.birth_state.genome_aggression_bias);
        EIEL_B2_COPY_PARENT_F32(2, arguments.birth_state.genome_mutation_rate);
        EIEL_B2_COPY_PARENT_F32(3, arguments.birth_state.genome_reproduction_threshold);
        EIEL_B2_COPY_PARENT_F32(4, arguments.birth_state.genome_action_temperature);
        EIEL_B2_COPY_PARENT_F32(5, arguments.birth_state.genome_drive_inertia);
        EIEL_B2_COPY_PARENT_F32(6, arguments.birth_state.genome_drive_noise);
        EIEL_B2_COPY_PARENT_F32(7, arguments.birth_state.genome_drive_logit_gain);
        EIEL_B2_COPY_PARENT_F32(8, arguments.birth_state.genome_pred_lr);
        EIEL_B2_COPY_PARENT_F32(9, arguments.birth_state.genome_drive_sp_hunger);
        EIEL_B2_COPY_PARENT_F32(10, arguments.birth_state.genome_drive_sp_pain);
        EIEL_B2_COPY_PARENT_F32(11, arguments.birth_state.genome_drive_sp_fatigue);
        EIEL_B2_COPY_PARENT_F32(12, arguments.birth_state.genome_drive_sp_fear);
        EIEL_B2_COPY_PARENT_F32(13, arguments.birth_state.genome_exp_prior_strength);
        EIEL_B2_COPY_PARENT_F32(14, arguments.birth_state.genome_exp_prior_noise);
        EIEL_B2_COPY_PARENT_F32(15, arguments.birth_state.genome_drive_birth_noise);
        EIEL_B2_COPY_PARENT_F32(16, arguments.birth_state.repro_threshold);
        EIEL_B2_COPY_PARENT_F32(17, arguments.full_tick_state.gru_upkeep_cost);
        EIEL_B2_COPY_PARENT_F32(18, arguments.attack_bias);
        EIEL_B2_COPY_PARENT_F32(19, arguments.action_temperature);
        EIEL_B2_COPY_PARENT_F32(20, arguments.drive_logit_gain);
        EIEL_B2_COPY_PARENT_F32(21, arguments.full_tick_state.drive_inertia);
        EIEL_B2_COPY_PARENT_F32(22, arguments.full_tick_state.drive_noise);
        EIEL_B2_COPY_PARENT_F32(23, arguments.birth_state.pred_lr);
#undef EIEL_B2_COPY_PARENT_F32
    }

    int32_t* scalar_i32 = scratch::at<int32_t>(
        arguments.birth_scratch, cell, scratch::kB2ScalarGenomeInt32
    );
    int64_t* scalar_mem_dim = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kB2ScalarPolicyMemDim
    );
    bool* scalar_use_gru = scratch::at<bool>(
        arguments.birth_scratch, cell, scratch::kB2ScalarPolicyUseGru
    );
    float* scalar_f32 = scratch::at<float>(
        arguments.birth_scratch, cell, scratch::kB2ScalarGenomeFloat32
    );
    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPythonRawDraws
    )[0] = physical_python_raw_draws;
    scalar_v2::stage_ordinary_fixed_mlp_child_scalar_genomes_external_scratch_v2(
        {
            mini_slot_ids,
            mini_alive,
            mini_i32 + 0 * 4,
            mini_i32 + 1 * 4,
            mini_i32 + 2 * 4,
            mini_i32 + 3 * 4,
            mini_i32 + 4 * 4,
            mini_i32 + 5 * 4,
            mini_i32 + 6 * 4,
            mini_i32 + 7 * 4,
            mini_mem_dim,
            mini_use_gru,
            mini_f32 + 0 * 4,
            mini_f32 + 1 * 4,
            mini_f32 + 2 * 4,
            mini_f32 + 3 * 4,
            mini_f32 + 4 * 4,
            mini_f32 + 5 * 4,
            mini_f32 + 6 * 4,
            mini_f32 + 7 * 4,
            mini_f32 + 8 * 4,
            mini_f32 + 9 * 4,
            mini_f32 + 10 * 4,
            mini_f32 + 11 * 4,
            mini_f32 + 12 * 4,
            mini_f32 + 13 * 4,
            mini_f32 + 14 * 4,
            mini_f32 + 15 * 4,
            mini_f32 + 16 * 4,
            mini_f32 + 17 * 4,
            mini_f32 + 18 * 4,
            mini_f32 + 19 * 4,
            mini_f32 + 20 * 4,
            mini_f32 + 21 * 4,
            mini_f32 + 22 * 4,
            mini_f32 + 23 * 4,
        },
        {mini_plan_parent, mini_plan_child, mini_plan_count},
        {
            arguments.birth_state.python_random_words + cell * 624,
            arguments.birth_state.python_random_index + cell,
            arguments.birth_state.python_random_has_gauss + cell,
            arguments.birth_state.python_random_gauss_next + cell,
            arguments.birth_state.python_random_gauss_primary_tape
                + cell * arguments.birth_config.gauss_tape_capacity,
            arguments.birth_state.python_random_gauss_secondary_tape
                + cell * arguments.birth_config.gauss_tape_capacity,
            scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kPythonRawDraws
            ),
            arguments.birth_state.mutation_py_semantic_draws + cell,
        },
        {
            scalar_i32 + 0 * 2,
            scalar_i32 + 1 * 2,
            scalar_i32 + 2 * 2,
            scalar_i32 + 3 * 2,
            scalar_i32 + 4 * 2,
            scalar_i32 + 5 * 2,
            scalar_i32 + 6 * 2,
            scalar_i32 + 7 * 2,
            scalar_mem_dim,
            scalar_use_gru,
            scalar_f32 + 0 * 2,
            scalar_f32 + 1 * 2,
            scalar_f32 + 2 * 2,
            scalar_f32 + 3 * 2,
            scalar_f32 + 4 * 2,
            scalar_f32 + 5 * 2,
            scalar_f32 + 6 * 2,
            scalar_f32 + 7 * 2,
            scalar_f32 + 8 * 2,
            scalar_f32 + 9 * 2,
            scalar_f32 + 10 * 2,
            scalar_f32 + 11 * 2,
            scalar_f32 + 12 * 2,
            scalar_f32 + 13 * 2,
            scalar_f32 + 14 * 2,
            scalar_f32 + 15 * 2,
            scalar_f32 + 16 * 2,
            scalar_f32 + 17 * 2,
            scalar_f32 + 18 * 2,
            scalar_f32 + 19 * 2,
            scalar_f32 + 20 * 2,
            scalar_f32 + 21 * 2,
            scalar_f32 + 22 * 2,
            scalar_f32 + 23 * 2,
            scratch::at<uint32_t>(
                arguments.birth_scratch, cell, scratch::kPythonPostWords
            ),
            scratch::at<uint32_t>(
                arguments.birth_scratch, cell, scratch::kPythonTrialWords
            ),
            scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kPythonIndex
            ),
            scratch::at<bool>(
                arguments.birth_scratch, cell, scratch::kPythonHasGauss
            ),
            scratch::at<double>(
                arguments.birth_scratch, cell, scratch::kPythonGaussNext
            ),
            scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kPythonRawDraws
            ),
            scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kPythonSemanticDraws
            ),
            scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kMiniScalarCommitted
            ),
            scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kMiniScalarSemanticDraws
            ),
            scratch::at<int32_t>(
                arguments.birth_scratch, cell, scratch::kMiniScalarStatus
            ),
        },
        {2, 4, arguments.birth_config.gauss_tape_capacity,
         arguments.birth_config.sensor_radius_max}
    );
    const int32_t scalar_status = scratch::at<int32_t>(
        arguments.birth_scratch, cell, scratch::kMiniScalarStatus
    )[0];
    if (scalar_status != 0) {
        return 120 + scalar_status;
    }

    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t child = birth * 2 + 1;
        for (int64_t field = 0; field < 8; ++field) {
            mini_i32[field * 4 + child] = scalar_i32[field * 2 + birth];
        }
        mini_mem_dim[child] = scalar_mem_dim[birth];
        mini_use_gru[child] = scalar_use_gru[birth];
        for (int64_t field = 0; field < 24; ++field) {
            mini_f32[field * 4 + child] = scalar_f32[field * 2 + birth];
        }
    }

    uint64_t* population_state = scratch::at<uint64_t>(
        arguments.birth_scratch, cell, scratch::kPopulationPcgState
    );
    int64_t* population_buffer = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPopulationPcgBuffer
    );
    int64_t* population_semantic = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPopulationSemanticDraws
    );
    for (int64_t word = 0; word < 4; ++word) {
        population_state[word] =
            arguments.birth_state.population_pcg64_state_words[cell * 4 + word];
    }
    for (int64_t word = 0; word < 2; ++word) {
        population_buffer[word] =
            arguments.birth_state.population_pcg64_buffer_words[cell * 2 + word];
    }
    population_semantic[0] = arguments.birth_state.mutation_np_semantic_draws[cell];
    flattened_birth_drives_v1::
        prepare_ordinary_newborn_drives_external_scratch_v2(
            mini_slot_ids,
            mini_alive,
            mini_use_gru,
            mini_energy,
            mini_f32 + 9 * 4,
            mini_f32 + 10 * 4,
            mini_f32 + 11 * 4,
            mini_f32 + 12 * 4,
            mini_f32 + 14 * 4,
            mini_f32 + 15 * 4,
            mini_plan_parent,
            mini_plan_child,
            birth_count,
            4,
            arguments.max_energy,
            population_state,
            population_buffer,
            population_semantic[0],
            scratch::at<float>(
                arguments.birth_scratch, cell, scratch::kMiniDrives
            ),
            scratch::at<float>(
                arguments.birth_scratch, cell, scratch::kMiniBirthShock
            ),
            scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kMiniDriveCommitted
            )[0],
            scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kMiniBirthNoiseRows
            )[0],
            scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kMiniExpPriorRows
            )[0],
            scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kMiniStandardNormalRows
            )[0],
            scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kMiniDriveSemanticDraws
            )[0],
            scratch::at<int32_t>(
                arguments.birth_scratch, cell, scratch::kMiniDriveStatus
            )[0]
        );
    const int32_t drive_status = scratch::at<int32_t>(
        arguments.birth_scratch, cell, scratch::kMiniDriveStatus
    )[0];
    if (drive_status != 0) {
        return 140 + drive_status;
    }

    float* chemistry = scratch::at<float>(
        arguments.birth_scratch, cell, scratch::kMiniChemistry
    );
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const double energy_n = chemistry_detail::clamp_double(
            identity_child_energy[birth]
                / static_cast<double>(arguments.max_energy),
            0.0,
            1.0
        );
        const int64_t offset = birth * 6;
        chemistry[offset] = static_cast<float>(0.40 + 0.40 * energy_n);
        chemistry[offset + 1] = 0.55f;
        chemistry[offset + 2] = 0.0f;
        chemistry[offset + 3] = 0.0f;
        chemistry[offset + 4] = 0.50f;
        chemistry[offset + 5] = 0.25f;
    }
    scratch::at<int32_t>(
        arguments.birth_scratch, cell, scratch::kMiniChemistryStatus
    )[0] = 0;

    int64_t* policy_cursor = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kMiniPolicyCursor
    );
    policy_cursor[0] = physical_policy_cursor;
    policy_v2::prepare_ordinary_fixed_mlp_child_policy_external_scratch_v2(
        mini_slot_ids,
        mini_alive,
        arguments.mlp_pack + cell_capacity_offset * arguments.packed_width,
        false,
        mini_use_gru,
        arguments.birth_state.mutation_torch_cpu_state
            + cell * arguments.birth_config.torch_state_bytes,
        arguments.birth_state.mutation_torch_cpu_semantic_draws[cell],
        arguments.birth_state.policy_birth_delta_tape
            + cell * arguments.birth_config.policy_tape_capacity
                * arguments.birth_config.policy_mutation_tape_width,
        arguments.birth_state.policy_birth_mutation_mask_tape
            + cell * arguments.birth_config.policy_tape_capacity
                * arguments.birth_config.policy_mutation_tape_width,
        arguments.birth_state.policy_birth_post_rng_state_packed_tape == nullptr
            ? nullptr
            : arguments.birth_state.policy_birth_post_rng_state_packed_tape
                + cell * arguments.birth_config.policy_tape_capacity
                    * torch_rng_pack::kPackedStateBytes,
        arguments.birth_state.policy_birth_semantic_draws_tape
            + cell * arguments.birth_config.policy_tape_capacity,
        physical_policy_cursor,
        mini_plan_parent,
        mini_plan_child,
        identity_parent_slots,
        birth_count,
        4,
        arguments.capacity,
        arguments.packed_width,
        arguments.birth_config.hmem_width,
        arguments.birth_config.torch_state_bytes,
        arguments.birth_config.policy_tape_capacity,
        arguments.birth_config.policy_mutation_tape_width,
        scratch::at<float>(
            arguments.birth_scratch, cell, layout.child_mlp_rows
        ),
        scratch::at<float>(
            arguments.birth_scratch, cell, layout.child_hmem_rows
        ),
        scratch::at<uint8_t>(
            arguments.birth_scratch, cell, layout.torch_state
        ),
        policy_cursor[0],
        scratch::at<int64_t>(
            arguments.birth_scratch, cell, scratch::kMiniPolicyStateSemanticDraws
        )[0],
        scratch::at<int64_t>(
            arguments.birth_scratch, cell, scratch::kMiniPolicyCommitted
        )[0],
        scratch::at<int64_t>(
            arguments.birth_scratch, cell, scratch::kMiniPolicySemanticDraws
        )[0],
        scratch::at<int64_t>(
            arguments.birth_scratch, cell, scratch::kMiniPolicyMutationEvents
        )[0],
        scratch::at<int64_t>(
            arguments.birth_scratch, cell, scratch::kMiniPolicyMutatedElements
        )[0],
        scratch::at<double>(
            arguments.birth_scratch, cell, scratch::kMiniPolicyMutationDeltaTotal
        )[0],
        scratch::at<bool>(
            arguments.birth_scratch, cell, scratch::kB2PolicyMutationOccurred
        ),
        scratch::at<float>(
            arguments.birth_scratch, cell, scratch::kB2PolicyMutationDelta
        ),
        scratch::at<int32_t>(
            arguments.birth_scratch, cell, scratch::kMiniPolicyStatus
        )[0]
    );
    const int32_t policy_status = scratch::at<int32_t>(
        arguments.birth_scratch, cell, scratch::kMiniPolicyStatus
    )[0];
    if (policy_status != 0) {
        return 160 + policy_status;
    }
    int64_t absolute_python_raw_draws = 0;
    int64_t absolute_policy_cursor = 0;
    if (
        !absolute_cursor_from_physical_v1(
            scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kPythonRawDraws
            )[0],
            arguments.birth_state.python_random_raw_draw_base[cell],
            arguments.birth_config.gauss_tape_capacity,
            &absolute_python_raw_draws
        )
        || !absolute_cursor_from_physical_v1(
            policy_cursor[0],
            arguments.birth_state.policy_birth_tape_cursor_base[cell],
            arguments.birth_config.policy_tape_capacity,
            &absolute_policy_cursor
        )
    ) {
        return 26;
    }

    int64_t* parent_agent_ids = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kB2ParentAgentIds
    );
    int32_t* child_founders = scratch::at<int32_t>(
        arguments.birth_scratch, cell, scratch::kB2ChildLineageFounder
    );
    int32_t* child_depths = scratch::at<int32_t>(
        arguments.birth_scratch, cell, scratch::kB2ChildLineageDepth
    );
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = identity_parent_slots[birth];
        const int64_t parent_index = cell_capacity_offset + parent_slot;
        const int64_t parent_id = arguments.slot_to_agent_id[parent_index];
        const int32_t parent_founder =
            arguments.event_state.lineage_founder_id[parent_index];
        const int32_t parent_depth = arguments.event_state.lineage_depth[parent_index];
        const int64_t founder = parent_founder >= 0 ? parent_founder : parent_id;
        if (
            founder < 0 || founder > INT32_MAX
            || parent_depth < 0 || parent_depth == INT32_MAX
        ) {
            return 182;
        }
        parent_agent_ids[birth] = parent_id;
        child_founders[birth] = static_cast<int32_t>(founder);
        child_depths[birth] = parent_depth + 1;
    }
    int32_t* event_status = scratch::at<int32_t>(
        arguments.birth_scratch, cell, scratch::kB2EventStatus
    );
    int64_t* required_event_rows = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kRequiredEventRows
    );
    int32_t staged_event_status = events_v2::kStatusOk;
    int64_t staged_required_event_rows = 0;
    events_v2::preflight_ordinary_birth_events_external_scratch_v2(
        {
            arguments.slot_to_agent_id + cell_capacity_offset,
            arguments.event_state.lineage_founder_id + cell_capacity_offset,
            arguments.event_state.lineage_depth + cell_capacity_offset,
            arguments.capacity,
        },
        {
            arguments.birth_outputs.decision_parent_slots + tick_capacity_offset,
            arguments.birth_outputs.decision_reason_codes + tick_capacity_offset,
            decision_count,
            arguments.capacity,
        },
        {
            identity_parent_slots,
            parent_agent_ids,
            identity_child_slots,
            identity_child_ids,
            child_founders,
            child_depths,
            scratch::at<bool>(
                arguments.birth_scratch, cell, scratch::kB2PolicyMutationOccurred
            ),
            scratch::at<float>(
                arguments.birth_scratch, cell, scratch::kB2PolicyMutationDelta
            ),
            birth_count,
            identity_count[0],
            2,
        },
        {
            arguments.event_state.active_count[cell],
            arguments.event_state.capacity,
            arguments.event_state.sequence_base[cell],
        },
        staged_event_status,
        staged_required_event_rows
    );
    event_status[0] = staged_event_status;
    required_event_rows[0] = staged_required_event_rows;
    if (event_status[0] != events_v2::kStatusOk) {
        return event_status[0] == events_v2::kStatusCapacityExceeded ? 25 : 205;
    }
    return 0;
}

__device__ __forceinline__ int32_t
stage_ordinary_cross_path_parent_authority_v1(
    const Phase2KernelArguments arguments,
    const int64_t cell,
    const int64_t cell_tick,
    const int64_t birth_count
) {
    namespace control_v1 = cross_path_resident_control_v1;
    const auto& control = arguments.birth_config.cross_path_control;
    if (!control_v1::has_cross_path_authority_v1(control)) {
        return control_v1::kInheritanceStatusOk;
    }
    if (!control_v1::ordinary_live_birth_cell_v1(control, cell)) {
        return control_v1::kInheritanceStatusOk;
    }
    if (
        !control_v1::has_complete_ordinary_authority_v1(control)
        || !arguments.birth_config.requires_mate
        || birth_count < 0
        || birth_count > arguments.birth_config.max_births_per_tick
        || control.inheritance_rows_per_cell
            < arguments.birth_config.max_births_per_tick
        || arguments.capacity <= 0
    ) {
        if (control.inheritance_status != nullptr) {
            control.inheritance_status[cell] =
                control_v1::kInheritanceStatusInvalidControl;
        }
        return control_v1::kInheritanceStatusInvalidControl;
    }
    const int8_t mode = control.inheritance_mode_by_cell[cell];
    if (
        mode != control_v1::kInheritanceModeLegacyMidpoint
        && mode != control_v1::kInheritanceModeLinkedParentCopy
    ) {
        control.inheritance_status[cell] =
            control_v1::kInheritanceStatusInvalidMode;
        return control_v1::kInheritanceStatusInvalidMode;
    }
    const bool linked =
        mode == control_v1::kInheritanceModeLinkedParentCopy;

    const int64_t birth_capacity =
        arguments.birth_config.max_births_per_tick;
    const int64_t birth_base =
        cell * control.inheritance_rows_per_cell;
    const int64_t slot_base = cell * arguments.capacity;
    const int64_t rank_base = cell * control.genotype_count;
    const int64_t tick_birth_base = cell_tick * birth_capacity;
    for (int64_t birth = 0; birth < birth_capacity; ++birth) {
        const int64_t row = birth_base + birth;
        control.canonical_parent_0_slots[row] = -1;
        control.canonical_parent_1_slots[row] = -1;
        control.child_slots[row] = -1;
        for (
            int64_t group = 0;
            group
                < flattened_mate_birth_scalar_genome_v3::linkage::
                    kLinkedDonorDrawCount;
            ++group
        ) {
            const int64_t group_row =
                row
                    * flattened_mate_birth_scalar_genome_v3::linkage::
                        kLinkedDonorDrawCount
                    + group;
            control.donor_ordinals[group_row] =
                control_v1::kNoDonorOrdinal;
            control.donor_statuses[group_row] =
                control_v1::kDonorStatusOk;
        }
        if (birth >= birth_count || !linked) {
            continue;
        }
        const int64_t source_row = tick_birth_base + birth;
        const int64_t parent_slot =
            arguments.birth_outputs.planned_parent_slots[source_row];
        const int64_t mate_slot =
            arguments.birth_outputs.planned_mate_slots[source_row];
        control.child_slots[row] =
            arguments.birth_outputs.planned_child_slots[source_row];
        if (
            parent_slot < 0 || parent_slot >= arguments.capacity
            || mate_slot < 0 || mate_slot >= arguments.capacity
            || parent_slot == mate_slot
        ) {
            control.inheritance_status[cell] =
                control_v1::kInheritanceStatusInvalidParentIdentity;
            return control_v1::kInheritanceStatusInvalidParentIdentity;
        }
        const int64_t parent_code =
            control.genotype_codes[slot_base + parent_slot];
        const int64_t mate_code =
            control.genotype_codes[slot_base + mate_slot];
        if (
            parent_code < 0 || parent_code >= control.genotype_count
            || mate_code < 0 || mate_code >= control.genotype_count
        ) {
            control.inheritance_status[cell] =
                control_v1::kInheritanceStatusInvalidParentIdentity;
            return control_v1::kInheritanceStatusInvalidParentIdentity;
        }
        const int64_t parent_rank =
            control.digest_rank_by_code[rank_base + parent_code];
        const int64_t mate_rank =
            control.digest_rank_by_code[rank_base + mate_code];
        const bool parent_first = parent_rank <= mate_rank;
        control.canonical_parent_0_slots[birth_base + birth] =
            parent_first ? parent_slot : mate_slot;
        control.canonical_parent_1_slots[birth_base + birth] =
            parent_first ? mate_slot : parent_slot;
    }
    control.donor_draws_added[cell] = 0;
    control.staged_counts[cell] = linked ? birth_count : 0;
    control.committed_counts[cell] = 0;
    control.inheritance_status[cell] =
        linked
        ? control_v1::kInheritanceStatusNotCommitted
        : control_v1::kInheritanceStatusOk;
    return control_v1::kInheritanceStatusOk;
}

__device__ __forceinline__ int32_t
finalize_ordinary_cross_path_transaction_v1(
    const Phase2KernelArguments arguments,
    const int64_t cell,
    const int64_t cell_tick,
    const int64_t controller_tick,
    const int32_t transaction_status
) {
    namespace control_v1 = cross_path_resident_control_v1;
    namespace mate_scalar = flattened_mate_birth_scalar_genome_v3;
    const auto& control = arguments.birth_config.cross_path_control;
    const auto& exposure = control.reproduction_exposure;
    const int8_t exposure_path =
        control.reproduction_path_by_cell != nullptr
        ? control.reproduction_path_by_cell[cell]
        : resident_reproduction_exposure_v1::kNoPath;
    const int64_t exposure_generation = arguments.mechanics_config.generation;
    const int64_t exposure_tick =
        arguments.mechanics_config.start_tick + controller_tick;
    if (
        !control_v1::has_cross_path_authority_v1(control)
        || !control_v1::ordinary_live_birth_cell_v1(control, cell)
    ) {
        return transaction_status;
    }
    if (!control_v1::has_complete_ordinary_authority_v1(control)) {
        if (control.inheritance_status != nullptr) {
            control.inheritance_status[cell] =
                control_v1::kInheritanceStatusInvalidControl;
        }
        return transaction_status;
    }
    if (transaction_status != 0) {
        (void)resident_reproduction_exposure_v1::abort_staged_attempts_v1(
            exposure,
            cell,
            exposure_path,
            exposure_generation,
            exposure_tick
        );
        control.donor_draws_added[cell] = 0;
        control.committed_counts[cell] = 0;
        control.inheritance_status[cell] =
            control_v1::kInheritanceStatusControllerBase
            + transaction_status;
        return transaction_status;
    }
    const int64_t birth_count =
        arguments.birth_outputs.planned_count[cell_tick];
    if (
        resident_reproduction_exposure_v1::enabled_v1(exposure)
        && resident_reproduction_exposure_v1::staged_attempt_count_v1(
            exposure,
            cell,
            exposure_path,
            exposure_generation,
            exposure_tick
        ) != birth_count
    ) {
        (void)resident_reproduction_exposure_v1::abort_staged_attempts_v1(
            exposure,
            cell,
            exposure_path,
            exposure_generation,
            exposure_tick
        );
        control.inheritance_status[cell] =
            control_v1::kInheritanceStatusInvalidParentIdentity;
        return control_v1::kInheritanceStatusInvalidParentIdentity;
    }
    const int64_t birth_capacity =
        arguments.birth_config.max_births_per_tick;
    const int64_t birth_base =
        cell * control.inheritance_rows_per_cell;
    const int64_t slot_base = cell * arguments.capacity;
    const bool linked =
        control.inheritance_mode_by_cell[cell]
            == control_v1::kInheritanceModeLinkedParentCopy;
    if (!linked) {
        for (int64_t birth = 0; birth < birth_count; ++birth) {
            const int64_t row = cell_tick * birth_capacity + birth;
            if (!resident_reproduction_exposure_v1::finalize_staged_attempt_v1(
                    exposure,
                    cell,
                    exposure_path,
                    exposure_generation,
                    exposure_tick,
                    birth,
                    arguments.birth_outputs.committed_parent_agent_ids[row],
                    arguments.birth_outputs.committed_mate_agent_ids[row],
                    arguments.birth_outputs.committed_child_agent_ids[row],
                    nullptr,
                    0
                )) {
                control.inheritance_status[cell] =
                    control_v1::kInheritanceStatusInvalidParentIdentity;
                return control_v1::kInheritanceStatusInvalidParentIdentity;
            }
        }
        control.donor_draws_added[cell] = 0;
        control.committed_counts[cell] = 0;
        control.inheritance_status[cell] =
            control_v1::kInheritanceStatusOk;
        if (!resident_reproduction_exposure_v1::record_path_rng_commit_v1(
                exposure, cell, exposure_path, birth_count
            )) {
            control.inheritance_status[cell] =
                control_v1::kInheritanceStatusInvalidParentIdentity;
            return control_v1::kInheritanceStatusInvalidParentIdentity;
        }
        return transaction_status;
    }
    control.donor_draws_added[cell] = linked
        ? birth_count * mate_scalar::linkage::kLinkedDonorDrawCount
        : 0;
    control.committed_counts[cell] = birth_count;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        if (linked) {
            const int64_t parent_0_slot =
                control.canonical_parent_0_slots[birth_base + birth];
            const int64_t parent_1_slot =
                control.canonical_parent_1_slots[birth_base + birth];
            const int64_t child_slot =
                arguments.birth_outputs.planned_child_slots[
                    cell_tick * birth_capacity + birth
                ];
            if (
                parent_0_slot < 0 || parent_0_slot >= arguments.capacity
                || parent_1_slot < 0 || parent_1_slot >= arguments.capacity
                || child_slot < 0 || child_slot >= arguments.capacity
            ) {
                (void)resident_reproduction_exposure_v1::
                    abort_staged_attempts_v1(
                        exposure,
                        cell,
                        exposure_path,
                        exposure_generation,
                        exposure_tick
                    );
                control.inheritance_status[cell] =
                    control_v1::kInheritanceStatusInvalidParentIdentity;
                return transaction_status;
            }
            const int64_t parent_0_code =
                control.genotype_codes[slot_base + parent_0_slot];
            const int64_t parent_1_code =
                control.genotype_codes[slot_base + parent_1_slot];
            int64_t child_code = 0;
            for (
                int64_t group = 0;
                group < mate_scalar::linkage::kLinkedDonorDrawCount;
                ++group
            ) {
                const uint8_t donor = control.donor_ordinals[
                    (birth_base + birth)
                        * mate_scalar::linkage::kLinkedDonorDrawCount
                    + group
                ];
                if (donor > 1) {
                    (void)resident_reproduction_exposure_v1::
                        abort_staged_attempts_v1(
                            exposure,
                            cell,
                            exposure_path,
                            exposure_generation,
                            exposure_tick
                        );
                    control.inheritance_status[cell] =
                        control_v1::kInheritanceStatusInvalidParentIdentity;
                    return transaction_status;
                }
                const int64_t donor_code =
                    donor == 0 ? parent_0_code : parent_1_code;
                child_code |= ((donor_code >> group) & 1LL) << group;
            }
            control.genotype_codes[slot_base + child_slot] = child_code;
        }
        const int64_t committed_row = cell_tick * birth_capacity + birth;
        if (!resident_reproduction_exposure_v1::finalize_staged_attempt_v1(
                exposure,
                cell,
                exposure_path,
                exposure_generation,
                exposure_tick,
                birth,
                arguments.birth_outputs.committed_parent_agent_ids[
                    committed_row
                ],
                arguments.birth_outputs.committed_mate_agent_ids[
                    committed_row
                ],
                arguments.birth_outputs.committed_child_agent_ids[
                    committed_row
                ],
                control.donor_ordinals
                    + (birth_base + birth)
                        * mate_scalar::linkage::kLinkedDonorDrawCount,
                mate_scalar::linkage::kLinkedDonorDrawCount
            )) {
            control.inheritance_status[cell] =
                control_v1::kInheritanceStatusInvalidParentIdentity;
            return control_v1::kInheritanceStatusInvalidParentIdentity;
        }
        for (
            int64_t group = 0;
            group < mate_scalar::linkage::kLinkedDonorDrawCount;
            ++group
        ) {
            control.donor_statuses[
                (birth_base + birth)
                    * mate_scalar::linkage::kLinkedDonorDrawCount
                + group
            ] = control_v1::kDonorStatusOk;
        }
    }
    control.inheritance_status[cell] =
        control_v1::kInheritanceStatusOk;
    if (!resident_reproduction_exposure_v1::record_path_rng_commit_v1(
            exposure, cell, exposure_path, birth_count
        )) {
        control.inheritance_status[cell] =
            control_v1::kInheritanceStatusInvalidParentIdentity;
        return control_v1::kInheritanceStatusInvalidParentIdentity;
    }
    return transaction_status;
}

__device__ __forceinline__ int32_t
prepare_flattened_runtime_birth_transaction_v2(
    const Phase2KernelArguments arguments,
    const int64_t cell,
    const int64_t controller_tick
) {
    namespace scratch = flattened_birth_scratch_v2;
    namespace identity_v2 = flattened_birth_identity_v2;
    namespace scalar_v2 = flattened_birth_scalar_genome_v2;
    namespace mate_scalar_v3 = flattened_mate_birth_scalar_genome_v3;
    namespace diploid = cuda_diploid_scalar_genome_v1;
    namespace policy_v2 = flattened_birth_policy_v2;
    namespace events_v2 = flattened_birth_events_v2;
    namespace chemistry_detail = persistent_chemistry::detail;
    namespace parent_order = birth_parent_order_v1;

    scratch::FlatBirthScratchLayoutV2 layout{};
    scratch::FlatBirthScratchArenaV2 arena{};
    if (!make_runtime_birth_scratch_access_v2(arguments, &layout, &arena)) {
        return 108;
    }
    scratch::BirthScalarStatusesV2* statuses =
        arena.at<scratch::BirthScalarStatusesV2>(
            cell, layout, scratch::kScalarStatuses
        );
    statuses[0] = {};
    statuses[0].transaction_status = -1;
    const int32_t qualification_fault_stage = arena.at<int32_t>(
        cell,
        layout,
        scratch::kGenomeUpgradeV2QualificationFaultStage
    )[0];

#define EIEL_RUNTIME_BIRTH_PREPARE_FAIL(code)                                   \
    do {                                                                        \
        statuses[0].transaction_status = (code);                                \
        return (code);                                                          \
    } while (false)

    const int64_t birth_capacity = arguments.birth_config.max_births_per_tick;
    const int64_t output_tick = arguments.output_ticks_per_cell == 1
        ? 0
        : controller_tick;
    const int64_t cell_tick =
        cell * arguments.output_ticks_per_cell + output_tick;
    const int64_t cell_capacity_offset = cell * arguments.capacity;
    const int64_t cell_grid_offset = cell * arguments.height * arguments.width;
    const int64_t tick_capacity_offset = cell_tick * arguments.capacity;
    const int64_t tick_birth_offset = cell_tick * birth_capacity;
    const int64_t birth_count = arguments.birth_outputs.planned_count[cell_tick];
    const int64_t decision_count = arguments.birth_outputs.decision_count[cell_tick];
    const auto parent_order_mode =
        cross_path_resident_control_v1::has_complete_ordinary_authority_v1(
            arguments.birth_config.cross_path_control
        )
        && arguments.birth_config.cross_path_control
                .selection_mode_by_cell[cell]
            == cross_path_resident_control_v1::
                kSelectionModeGenotypeBlindNeutral
        ? parent_order::BirthParentOrderModeV1::authoritative_plan_v1
        : parent_order::BirthParentOrderModeV1::ascending_agent_id_v1;
    statuses[0].planned_count = birth_count;
    statuses[0].plan_status = arguments.birth_outputs.plan_status[cell_tick];
    if (arguments.full_tick_outputs.residual_release_status[cell_tick] != 0) {
        EIEL_RUNTIME_BIRTH_PREPARE_FAIL(4);
    }
    if (statuses[0].plan_status != 0) {
        EIEL_RUNTIME_BIRTH_PREPARE_FAIL(1);
    }
    if (
        birth_capacity <= 0 || birth_capacity > arguments.capacity
        || birth_count < 0 || birth_count > birth_capacity
        || decision_count < 0 || decision_count > arguments.capacity
        || arguments.drive_width < 4 || arguments.drive_width > 8
        || arguments.full_tick_state.chemistry_width < 6
        || arguments.full_tick_state.chemistry_width > 8
        || !::isfinite(arguments.max_energy) || arguments.max_energy <= 0.0f
    ) {
        EIEL_RUNTIME_BIRTH_PREPARE_FAIL(112);
    }
    const bool diploid_mode =
        arguments.birth_state.inheritance_mode != nullptr
        && arguments.birth_state.inheritance_mode[cell] == 2
        && arguments.birth_state.inheritance_mutation_enabled != nullptr
        && arguments.birth_state.resource_handling_enabled != nullptr
        && arguments.birth_state.mutation_sigma_multiplier != nullptr
        && arguments.birth_state.diploid_allele_storage != nullptr
        && arguments.birth_state.genome_resource_handling_axis != nullptr;
    const bool genome_upgrade_v2_requested =
        arguments.birth_state.genome_upgrade_v2_enabled != nullptr
        && arguments.birth_state.genome_upgrade_v2_enabled[cell];
    const bool genome_upgrade_v2_mode =
        genome_upgrade_v2_requested
        && diploid_mode
        && arguments.birth_state.genome_upgrade_v2_mutation_rate_q10 != nullptr
        && arguments.birth_state.genome_upgrade_v2_mutation_step_q10 != nullptr
        && arguments.birth_state.genome_upgrade_v2_alleles_q10 != nullptr
        && arguments.birth_state.genome_upgrade_v2_modifiers_q10 != nullptr
        && arguments.birth_state.genome_upgrade_v2_products_q10 != nullptr
        && arguments.birth_state.genome_upgrade_v2_phenotypes_q10 != nullptr
        && arguments.birth_state.genome_upgrade_v2_expression_cost_q10 != nullptr
        && arguments.birth_state.genome_upgrade_v2_active_product_count != nullptr
        && arguments.birth_state.genome_upgrade_v2_saturated_product_count
            != nullptr;
    if (genome_upgrade_v2_requested && !genome_upgrade_v2_mode) {
        EIEL_RUNTIME_BIRTH_PREPARE_FAIL(235);
    }
    const auto staged_parent_order_mode = diploid_mode
        ? parent_order::BirthParentOrderModeV1::authoritative_plan_v1
        : parent_order_mode;
    constexpr int64_t kDigestPackBytes = 8448;
    const uint8_t* digest_pack = diploid_mode
        ? arguments.birth_scratch.storage
            + (cell + 1) * arguments.birth_scratch.bytes_per_cell
            - kDigestPackBytes
        : nullptr;
    uint8_t* canonical_hereditary_digest =
        diploid_mode
        ? arguments.birth_state.canonical_hereditary_digest
        : nullptr;
    if (diploid_mode && canonical_hereditary_digest == nullptr) {
        EIEL_RUNTIME_BIRTH_PREPARE_FAIL(236);
    }
    const bool cross_path_mode =
        cross_path_resident_control_v1::has_cross_path_authority_v1(
            arguments.birth_config.cross_path_control
        );
    if (diploid_mode && cross_path_mode) {
        EIEL_RUNTIME_BIRTH_PREPARE_FAIL(239);
    }
    const int32_t cross_path_parent_status =
        stage_ordinary_cross_path_parent_authority_v1(
            arguments, cell, cell_tick, birth_count
        );
    if (cross_path_parent_status != 0) {
        EIEL_RUNTIME_BIRTH_PREPARE_FAIL(
            240 + cross_path_parent_status
        );
    }
    int64_t physical_policy_cursor = 0;
    if (
        arguments.birth_state.policy_birth_tape_cursor_base[cell]
            > INT64_MAX - arguments.birth_config.policy_tape_capacity
        || !physical_cursor_from_absolute_v1(
            arguments.birth_state.policy_birth_tape_cursor[cell],
            arguments.birth_state.policy_birth_tape_cursor_base[cell],
            arguments.birth_config.policy_tape_capacity,
            &physical_policy_cursor
        )
    ) {
        EIEL_RUNTIME_BIRTH_PREPARE_FAIL(116);
    }
    if (
        birth_count
        > arguments.birth_config.policy_tape_capacity - physical_policy_cursor
    ) {
        EIEL_RUNTIME_BIRTH_PREPARE_FAIL(117);
    }
    int64_t physical_python_raw_draws = 0;
    if (
        arguments.birth_state.python_random_raw_draw_base[cell]
            > INT64_MAX - arguments.birth_config.gauss_tape_capacity
        || !physical_cursor_from_absolute_v1(
            arguments.birth_state.python_random_raw_draws[cell],
            arguments.birth_state.python_random_raw_draw_base[cell],
            arguments.birth_config.gauss_tape_capacity,
            &physical_python_raw_draws
        )
    ) {
        EIEL_RUNTIME_BIRTH_PREPARE_FAIL(118);
    }

    scratch::Pcg64SnapshotV2* planner_rng =
        arena.at<scratch::Pcg64SnapshotV2>(
            cell, layout, scratch::kPlannerPcg64Snapshot
        );
    scratch::Pcg64SnapshotV2* runner_rng =
        arena.at<scratch::Pcg64SnapshotV2>(
            cell, layout, scratch::kRunnerPcg64Snapshot
        );
    namespace legacy_scratch = flattened_birth_scratch_v1;
    const uint64_t* legacy_planner_state = legacy_scratch::at<uint64_t>(
        arguments.birth_scratch, cell, legacy_scratch::kPlannerPcgState
    );
    const int64_t* legacy_planner_buffer = legacy_scratch::at<int64_t>(
        arguments.birth_scratch, cell, legacy_scratch::kPlannerPcgBuffer
    );
    const uint64_t* legacy_runner_state = legacy_scratch::at<uint64_t>(
        arguments.birth_scratch, cell, legacy_scratch::kRunnerPcgState
    );
    const int64_t* legacy_runner_buffer = legacy_scratch::at<int64_t>(
        arguments.birth_scratch, cell, legacy_scratch::kRunnerPcgBuffer
    );
    for (int64_t word = 0; word < scratch::kPcg64StateWordCount; ++word) {
        planner_rng[0].state_words[word] = legacy_planner_state[word];
        runner_rng[0].state_words[word] = legacy_runner_state[word];
    }
    for (int64_t word = 0; word < scratch::kPcg64BufferWordCount; ++word) {
        planner_rng[0].buffer_words[word] = legacy_planner_buffer[word];
        runner_rng[0].buffer_words[word] = legacy_runner_buffer[word];
    }

    int64_t* identity_parent_slots = arena.at<int64_t>(
        cell, layout, scratch::kIdentityParentSlots
    );
    int64_t* identity_child_slots = arena.at<int64_t>(
        cell, layout, scratch::kIdentityChildSlots
    );
    int64_t* identity_child_ids = arena.at<int64_t>(
        cell, layout, scratch::kIdentityChildIds
    );
    double* identity_child_energy = arena.at<double>(
        cell, layout, scratch::kIdentityChildEnergy
    );
    int64_t* identity_count = arena.at<int64_t>(
        cell, layout, scratch::kIdentityCount
    );
    int64_t* queue_head = arena.at<int64_t>(
        cell, layout, scratch::kQueueHead
    );
    int64_t* queue_count = arena.at<int64_t>(
        cell, layout, scratch::kQueueCount
    );
    const auto identity_status =
        identity_v2::preflight_ordinary_birth_identity_external_scratch_v2(
            {
                arguments.slot_to_agent_id + cell_capacity_offset,
                arguments.alive_mask + cell_capacity_offset,
                arguments.agent_grid + cell_grid_offset,
                arguments.energy + cell_capacity_offset,
                arguments.health + cell_capacity_offset,
                arguments.birth_outputs.planned_repro_cooldown
                    + cell_capacity_offset,
                arguments.birth_state.repro_threshold + cell_capacity_offset,
                arguments.policy_energy_observation + cell_capacity_offset,
                arguments.full_tick_state.slot_free_queue + cell_capacity_offset,
                arguments.full_tick_state.slot_free_queue_count + cell,
                arguments.birth_state.population_next_id + cell,
            },
            {
                arguments.birth_outputs.planned_parent_slots + tick_birth_offset,
                arguments.birth_outputs.planned_spawn_positions
                    + tick_birth_offset * 2,
                arguments.birth_outputs.planned_count + cell_tick,
            },
            {
                identity_parent_slots,
                identity_child_slots,
                identity_child_ids,
                identity_child_energy,
                identity_count,
                queue_head,
                queue_count,
            },
            {
                birth_capacity,
                arguments.capacity,
                arguments.agent_id_capacity,
                arguments.height,
                arguments.width,
                arguments.capacity,
                birth_parent_energy_share_for_cell_v1(arguments.birth_config, cell),
                static_cast<double>(birth_minimum_energy_buffer_for_cell_v1(arguments.birth_config, cell)),
                arguments.start_health,
                birth_cooldown_steps_for_cell_v1(arguments.birth_config, cell),
                static_cast<int32_t>(arguments.mechanics_config.generation),
                static_cast<int32_t>(
                    arguments.mechanics_config.start_tick + controller_tick
                ),
                parent_order_mode,
            }
        );
    statuses[0].identity_status = static_cast<int32_t>(identity_status);
    statuses[0].identity_committed_count = identity_count[0];
    if (identity_status != persistent_birth_identity::kStatusOk) {
        EIEL_RUNTIME_BIRTH_PREPARE_FAIL(
            identity_status == persistent_birth_identity::kStatusAgentIdCapacity
                ? 20
                : 200 + static_cast<int32_t>(identity_status)
        );
    }
    const int32_t mate_identity_status =
        validate_mate_identity_and_stage_energy_v1(
            arguments,
            cell,
            cell_tick,
            birth_count,
            identity_child_energy
        );
    if (mate_identity_status != 0) {
        EIEL_RUNTIME_BIRTH_PREPARE_FAIL(mate_identity_status);
    }

    const int64_t mini_rows = layout.mini_state_rows;
    int64_t* mini_slot_ids = arena.at<int64_t>(
        cell, layout, scratch::kMiniSlotRows
    );
    bool* mini_alive = arena.at<bool>(
        cell, layout, scratch::kMiniAliveRows
    );
    int32_t* mini_i32 = arena.at<int32_t>(
        cell, layout, scratch::kMiniGenomeInt32Planes
    );
    int64_t* mini_mem_dim = arena.at<int64_t>(
        cell, layout, scratch::kMiniPolicyMemoryDimensionRows
    );
    bool* mini_use_gru = arena.at<bool>(
        cell, layout, scratch::kMiniPolicyUseGruRows
    );
    float* mini_f32 = arena.at<float>(
        cell, layout, scratch::kMiniGenomeFloatPlanes
    );
    double* mini_energy = arena.at<double>(
        cell, layout, scratch::kMiniEnergyRows
    );
    int64_t* mini_plan_parent = arena.at<int64_t>(
        cell, layout, scratch::kParentPlanRows
    );
    int64_t* mini_plan_mate = arena.at<int64_t>(
        cell, layout, scratch::kMatePlanRows
    );
    int64_t* mini_plan_child = arena.at<int64_t>(
        cell, layout, scratch::kChildPlanRows
    );
    for (int64_t row = 0; row < mini_rows; ++row) {
        mini_slot_ids[row] = -1;
        mini_alive[row] = false;
        mini_mem_dim[row] = 0;
        mini_use_gru[row] = false;
        mini_energy[row] = 0.0;
    }
    for (int64_t index = 0; index < 8 * mini_rows; ++index) {
        mini_i32[index] = 0;
    }
    for (int64_t index = 0; index < 24 * mini_rows; ++index) {
        mini_f32[index] = 0.0f;
    }
    for (int64_t birth = 0; birth < birth_capacity; ++birth) {
        const int64_t mini_parent = layout.mini_row(scratch::kMiniParent, birth);
        const int64_t mini_mate = layout.mini_row(scratch::kMiniMate, birth);
        const int64_t mini_child = layout.mini_row(scratch::kMiniChild, birth);
        mini_plan_parent[birth] = mini_parent;
        mini_plan_mate[birth] = mini_mate;
        mini_plan_child[birth] = mini_child;
        if (birth >= birth_count) {
            continue;
        }
        const int64_t parent_slot = identity_parent_slots[birth];
        const int64_t mate_slot = arguments.birth_config.requires_mate
            ? arguments.birth_outputs.planned_mate_slots[
                  tick_birth_offset + birth
              ]
            : parent_slot;
        const int64_t parent_index = cell_capacity_offset + parent_slot;
        const int64_t mate_index = cell_capacity_offset + mate_slot;
        const int64_t parent_id = arguments.slot_to_agent_id[parent_index];
        mini_slot_ids[mini_parent] = parent_id;
        mini_slot_ids[mini_mate] =
            arguments.slot_to_agent_id[mate_index];
        mini_slot_ids[mini_child] = identity_child_ids[birth];
        mini_alive[mini_parent] = true;
        mini_alive[mini_mate] = true;
        mini_alive[mini_child] = true;
        mini_mem_dim[mini_parent] =
            arguments.birth_state.policy_mem_dim[parent_index];
        mini_mem_dim[mini_mate] =
            arguments.birth_state.policy_mem_dim[mate_index];
        mini_use_gru[mini_parent] =
            arguments.birth_state.policy_use_gru[parent_index];
        mini_use_gru[mini_mate] =
            arguments.birth_state.policy_use_gru[mate_index];
        const double parent_energy = ::isfinite(
            arguments.policy_energy_observation[parent_index]
        )
            ? arguments.policy_energy_observation[parent_index]
            : static_cast<double>(arguments.energy[parent_index]);
        mini_energy[mini_parent] = parent_energy;
        mini_energy[mini_mate] = effective_birth_energy_v1(
            arguments, cell_capacity_offset, mate_slot
        );
        mini_energy[mini_child] = identity_child_energy[birth];

#define EIEL_RUNTIME_COPY_PARENT_I32(field, source)                              \
        mini_i32[(field) * mini_rows + mini_parent] = (source)[parent_index];    \
        mini_i32[(field) * mini_rows + mini_mate] = (source)[mate_index]
        EIEL_RUNTIME_COPY_PARENT_I32(0, arguments.birth_state.genome_hidden_size);
        EIEL_RUNTIME_COPY_PARENT_I32(1, arguments.birth_state.genome_memory_dim);
        EIEL_RUNTIME_COPY_PARENT_I32(2, arguments.birth_state.genome_sensor_channel_mask);
        EIEL_RUNTIME_COPY_PARENT_I32(3, arguments.birth_state.genome_sensor_radius);
        EIEL_RUNTIME_COPY_PARENT_I32(4, arguments.birth_state.genome_introspection_channel_mask);
        EIEL_RUNTIME_COPY_PARENT_I32(5, arguments.birth_state.genome_self_regulation_control_mask);
        EIEL_RUNTIME_COPY_PARENT_I32(6, arguments.birth_state.genome_self_modification_target_mask);
        EIEL_RUNTIME_COPY_PARENT_I32(7, arguments.birth_state.genome_directed_inheritance_channel_mask);
#undef EIEL_RUNTIME_COPY_PARENT_I32

#define EIEL_RUNTIME_COPY_PARENT_F32(field, source)                              \
        mini_f32[(field) * mini_rows + mini_parent] = (source)[parent_index];    \
        mini_f32[(field) * mini_rows + mini_mate] = (source)[mate_index]
        EIEL_RUNTIME_COPY_PARENT_F32(0, arguments.birth_state.genome_exploration_bias);
        EIEL_RUNTIME_COPY_PARENT_F32(1, arguments.birth_state.genome_aggression_bias);
        EIEL_RUNTIME_COPY_PARENT_F32(2, arguments.birth_state.genome_mutation_rate);
        EIEL_RUNTIME_COPY_PARENT_F32(3, arguments.birth_state.genome_reproduction_threshold);
        EIEL_RUNTIME_COPY_PARENT_F32(4, arguments.birth_state.genome_action_temperature);
        EIEL_RUNTIME_COPY_PARENT_F32(5, arguments.birth_state.genome_drive_inertia);
        EIEL_RUNTIME_COPY_PARENT_F32(6, arguments.birth_state.genome_drive_noise);
        EIEL_RUNTIME_COPY_PARENT_F32(7, arguments.birth_state.genome_drive_logit_gain);
        EIEL_RUNTIME_COPY_PARENT_F32(8, arguments.birth_state.genome_pred_lr);
        EIEL_RUNTIME_COPY_PARENT_F32(9, arguments.birth_state.genome_drive_sp_hunger);
        EIEL_RUNTIME_COPY_PARENT_F32(10, arguments.birth_state.genome_drive_sp_pain);
        EIEL_RUNTIME_COPY_PARENT_F32(11, arguments.birth_state.genome_drive_sp_fatigue);
        EIEL_RUNTIME_COPY_PARENT_F32(12, arguments.birth_state.genome_drive_sp_fear);
        EIEL_RUNTIME_COPY_PARENT_F32(13, arguments.birth_state.genome_exp_prior_strength);
        EIEL_RUNTIME_COPY_PARENT_F32(14, arguments.birth_state.genome_exp_prior_noise);
        EIEL_RUNTIME_COPY_PARENT_F32(15, arguments.birth_state.genome_drive_birth_noise);
        EIEL_RUNTIME_COPY_PARENT_F32(16, arguments.birth_state.repro_threshold);
        EIEL_RUNTIME_COPY_PARENT_F32(17, arguments.full_tick_state.gru_upkeep_cost);
        EIEL_RUNTIME_COPY_PARENT_F32(18, arguments.attack_bias);
        EIEL_RUNTIME_COPY_PARENT_F32(19, arguments.action_temperature);
        EIEL_RUNTIME_COPY_PARENT_F32(20, arguments.drive_logit_gain);
        EIEL_RUNTIME_COPY_PARENT_F32(21, arguments.full_tick_state.drive_inertia);
        EIEL_RUNTIME_COPY_PARENT_F32(22, arguments.full_tick_state.drive_noise);
        EIEL_RUNTIME_COPY_PARENT_F32(23, arguments.birth_state.pred_lr);
#undef EIEL_RUNTIME_COPY_PARENT_F32
    }

    int32_t* scalar_i32 = arena.at<int32_t>(
        cell, layout, scratch::kGenomeInt32Planes
    );
    int64_t* scalar_mem_dim = arena.at<int64_t>(
        cell, layout, scratch::kPolicyMemoryDimensionRows
    );
    bool* scalar_use_gru = arena.at<bool>(
        cell, layout, scratch::kPolicyUseGruRows
    );
    float* scalar_f32 = arena.at<float>(
        cell, layout, scratch::kGenomeFloatPlanes
    );
    uint32_t* python_post_words = arena.at<uint32_t>(
        cell, layout, scratch::kPythonRandomPostWords
    );
    uint32_t* python_trial_words = arena.at<uint32_t>(
        cell, layout, scratch::kPythonRandomTrialWords
    );
    scratch::PythonRandomSnapshotV2* python_snapshots =
        arena.at<scratch::PythonRandomSnapshotV2>(
            cell, layout, scratch::kPythonRandomSnapshots
        );
    python_snapshots[0] = {
        arguments.birth_state.python_random_index[cell],
        physical_python_raw_draws,
        arguments.birth_state.mutation_py_semantic_draws[cell],
        arguments.birth_state.python_random_gauss_next[cell],
        arguments.birth_state.python_random_has_gauss[cell],
        {},
    };
    python_snapshots[1] = python_snapshots[0];
    const scalar_v2::FlatAuthoritativeFixedMlpScalarGenomeState scalar_state{
            mini_slot_ids,
            mini_alive,
            mini_i32 + 0 * mini_rows,
            mini_i32 + 1 * mini_rows,
            mini_i32 + 2 * mini_rows,
            mini_i32 + 3 * mini_rows,
            mini_i32 + 4 * mini_rows,
            mini_i32 + 5 * mini_rows,
            mini_i32 + 6 * mini_rows,
            mini_i32 + 7 * mini_rows,
            mini_mem_dim,
            mini_use_gru,
            mini_f32 + 0 * mini_rows,
            mini_f32 + 1 * mini_rows,
            mini_f32 + 2 * mini_rows,
            mini_f32 + 3 * mini_rows,
            mini_f32 + 4 * mini_rows,
            mini_f32 + 5 * mini_rows,
            mini_f32 + 6 * mini_rows,
            mini_f32 + 7 * mini_rows,
            mini_f32 + 8 * mini_rows,
            mini_f32 + 9 * mini_rows,
            mini_f32 + 10 * mini_rows,
            mini_f32 + 11 * mini_rows,
            mini_f32 + 12 * mini_rows,
            mini_f32 + 13 * mini_rows,
            mini_f32 + 14 * mini_rows,
            mini_f32 + 15 * mini_rows,
            mini_f32 + 16 * mini_rows,
            mini_f32 + 17 * mini_rows,
            mini_f32 + 18 * mini_rows,
            mini_f32 + 19 * mini_rows,
            mini_f32 + 20 * mini_rows,
            mini_f32 + 21 * mini_rows,
            mini_f32 + 22 * mini_rows,
            mini_f32 + 23 * mini_rows,
    };
    const scalar_v2::FlatAuthoritativeCpythonRandomState scalar_rng{
            arguments.birth_state.python_random_words + cell * 624,
            arguments.birth_state.python_random_index + cell,
            arguments.birth_state.python_random_has_gauss + cell,
            arguments.birth_state.python_random_gauss_next + cell,
            arguments.birth_state.python_random_gauss_primary_tape
                + cell * arguments.birth_config.gauss_tape_capacity,
            arguments.birth_state.python_random_gauss_secondary_tape
                + cell * arguments.birth_config.gauss_tape_capacity,
            &physical_python_raw_draws,
            arguments.birth_state.mutation_py_semantic_draws + cell,
    };
    const scalar_v2::FlatBirthOrderedScalarGenomeScratch scalar_scratch{
            scalar_i32 + 0 * birth_capacity,
            scalar_i32 + 1 * birth_capacity,
            scalar_i32 + 2 * birth_capacity,
            scalar_i32 + 3 * birth_capacity,
            scalar_i32 + 4 * birth_capacity,
            scalar_i32 + 5 * birth_capacity,
            scalar_i32 + 6 * birth_capacity,
            scalar_i32 + 7 * birth_capacity,
            scalar_mem_dim,
            scalar_use_gru,
            scalar_f32 + 0 * birth_capacity,
            scalar_f32 + 1 * birth_capacity,
            scalar_f32 + 2 * birth_capacity,
            scalar_f32 + 3 * birth_capacity,
            scalar_f32 + 4 * birth_capacity,
            scalar_f32 + 5 * birth_capacity,
            scalar_f32 + 6 * birth_capacity,
            scalar_f32 + 7 * birth_capacity,
            scalar_f32 + 8 * birth_capacity,
            scalar_f32 + 9 * birth_capacity,
            scalar_f32 + 10 * birth_capacity,
            scalar_f32 + 11 * birth_capacity,
            scalar_f32 + 12 * birth_capacity,
            scalar_f32 + 13 * birth_capacity,
            scalar_f32 + 14 * birth_capacity,
            scalar_f32 + 15 * birth_capacity,
            scalar_f32 + 16 * birth_capacity,
            scalar_f32 + 17 * birth_capacity,
            scalar_f32 + 18 * birth_capacity,
            scalar_f32 + 19 * birth_capacity,
            scalar_f32 + 20 * birth_capacity,
            scalar_f32 + 21 * birth_capacity,
            scalar_f32 + 22 * birth_capacity,
            scalar_f32 + 23 * birth_capacity,
            python_post_words,
            python_trial_words,
            &python_snapshots[1].index,
            &python_snapshots[1].has_gauss,
            &python_snapshots[1].gauss_next,
            &python_snapshots[1].raw_draws,
            &python_snapshots[1].semantic_draws,
            &statuses[0].scalar_committed_count,
            &statuses[0].scalar_semantic_draws,
            &statuses[0].scalar_status,
    };
    const scalar_v2::OrdinaryScalarGenomeConfig scalar_config{
        birth_capacity,
        mini_rows,
        arguments.birth_config.gauss_tape_capacity,
        arguments.birth_config.sensor_radius_max,
        staged_parent_order_mode,
    };
    if (diploid_mode) {
        uint8_t* linked_donor_ordinals = arena.at<uint8_t>(
            cell, layout, scratch::kLinkedDonorOrdinals
        );
        const uint8_t* const canonical_digest =
            canonical_hereditary_digest + cell * arguments.capacity * 32;
        for (int64_t birth = 0; birth < birth_count; ++birth) {
            const int64_t output_row = tick_birth_offset + birth;
            const int64_t parent_slot =
                arguments.birth_outputs.planned_parent_slots[output_row];
            const int64_t mate_slot =
                arguments.birth_outputs.planned_mate_slots[output_row];
            bool mate_first = false;
            for (int32_t byte = 0; byte < 32; ++byte) {
                const uint8_t parent_value =
                    canonical_digest[parent_slot * 32 + byte];
                const uint8_t mate_value =
                    canonical_digest[mate_slot * 32 + byte];
                if (parent_value != mate_value) {
                    mate_first = mate_value < parent_value;
                    break;
                }
            }
            if (mate_first) {
                const int64_t mini_parent = mini_plan_parent[birth];
                mini_plan_parent[birth] = mini_plan_mate[birth];
                mini_plan_mate[birth] = mini_parent;
            }
        }
        mate_scalar_v3::
            stage_ordinary_fixed_mlp_child_scalar_genomes_with_mates_external_scratch_v4(
                scalar_state,
                {
                    mini_plan_parent,
                    mini_plan_mate,
                    mini_plan_child,
                    &birth_count,
                },
                scalar_rng,
                scalar_scratch,
                scalar_config,
                {
                    eiel::inheritance_linkage_v1::InheritanceModeV1::
                        linked_parent_copy_v1,
                    false,
                    mini_plan_parent,
                    mini_plan_mate,
                    linked_donor_ordinals,
                }
            );
    } else if (arguments.birth_config.requires_mate) {
        namespace mate_scalar =
            flattened_mate_birth_scalar_genome_v3;
        namespace control_v1 = cross_path_resident_control_v1;
        const auto& control = arguments.birth_config.cross_path_control;
        if (control_v1::has_cross_path_authority_v1(control)) {
            const int64_t control_birth_base =
                cell * control.inheritance_rows_per_cell;
            // v4 validates compact mini-state slots. Preserve external
            // canonical rows in authoritative slot coordinates, translating
            // them only while the scalar transaction executes.
            for (int64_t birth = 0; birth < birth_count; ++birth) {
                const int64_t output_row = tick_birth_offset + birth;
                const int64_t parent_slot =
                    arguments.birth_outputs.planned_parent_slots[output_row];
                const int64_t canonical_0 =
                    control.canonical_parent_0_slots[
                        control_birth_base + birth
                    ];
                const bool parent_first = canonical_0 == parent_slot;
                control.canonical_parent_0_slots[
                    control_birth_base + birth
                ] = parent_first
                    ? mini_plan_parent[birth]
                    : mini_plan_mate[birth];
                control.canonical_parent_1_slots[
                    control_birth_base + birth
                ] = parent_first
                    ? mini_plan_mate[birth]
                    : mini_plan_parent[birth];
            }
            const int8_t mode_code =
                control.inheritance_mode_by_cell[cell];
            const auto mode =
                mode_code == control_v1::kInheritanceModeLinkedParentCopy
                ? mate_scalar::linkage::InheritanceModeV1::
                    linked_parent_copy_v1
                : mate_scalar::linkage::InheritanceModeV1::
                    legacy_midpoint_v1;
            mate_scalar::
                stage_ordinary_fixed_mlp_child_scalar_genomes_with_mates_external_scratch_v4(
                    scalar_state,
                    {
                        mini_plan_parent,
                        mini_plan_mate,
                        mini_plan_child,
                        &birth_count,
                    },
                    scalar_rng,
                    scalar_scratch,
                    scalar_config,
                    {
                        mode,
                        control.mutation_enabled_by_cell[cell],
                        control.canonical_parent_0_slots
                            + control_birth_base,
                        control.canonical_parent_1_slots
                            + control_birth_base,
                        control.donor_ordinals
                            + control_birth_base
                                * mate_scalar::linkage::
                                    kLinkedDonorDrawCount,
                    }
                );
            for (int64_t birth = 0; birth < birth_count; ++birth) {
                const int64_t output_row = tick_birth_offset + birth;
                const int64_t parent_slot =
                    arguments.birth_outputs.planned_parent_slots[output_row];
                const int64_t mate_slot =
                    arguments.birth_outputs.planned_mate_slots[output_row];
                const bool parent_first =
                    control.canonical_parent_0_slots[
                        control_birth_base + birth
                    ] == mini_plan_parent[birth];
                control.canonical_parent_0_slots[
                    control_birth_base + birth
                ] = parent_first ? parent_slot : mate_slot;
                control.canonical_parent_1_slots[
                    control_birth_base + birth
                ] = parent_first ? mate_slot : parent_slot;
            }
            if (statuses[0].scalar_status != 0) {
                control.inheritance_status[cell] =
                    control_v1::kInheritanceStatusScalarBase
                    + statuses[0].scalar_status;
            }
        } else {
            mate_scalar::
                stage_ordinary_fixed_mlp_child_scalar_genomes_with_mates_external_scratch_v3(
                    scalar_state,
                    {
                        mini_plan_parent,
                        mini_plan_mate,
                        mini_plan_child,
                        &birth_count,
                    },
                    scalar_rng,
                    scalar_scratch,
                    scalar_config
                );
        }
    } else {
        scalar_v2::
            stage_ordinary_fixed_mlp_child_scalar_genomes_external_scratch_v2(
                scalar_state,
                {mini_plan_parent, mini_plan_child, &birth_count},
                scalar_rng,
                scalar_scratch,
                scalar_config
            );
    }
    if (statuses[0].scalar_status != 0) {
        EIEL_RUNTIME_BIRTH_PREPARE_FAIL(120 + statuses[0].scalar_status);
    }
    if (diploid_mode) {
        const int64_t allele_cell_stride =
            diploid::kAllelePlaneCount * arguments.capacity;
        float* const allele_cell_base =
            arguments.birth_state.diploid_allele_storage
            + cell * allele_cell_stride;
        const diploid::AllelePlanesV1 allele_planes{
            allele_cell_base,
            allele_cell_base
                + diploid::kLocusCount * arguments.capacity,
            arguments.capacity,
            arguments.capacity,
        };
        diploid::StagedChildV1* staged_children =
            arena.at<diploid::StagedChildV1>(
                cell, layout, scratch::kDiploidStagedChildren
            );
        persistent_birth_scalar_genome::DeviceCpythonRandom diploid_rng{
            python_post_words,
            python_snapshots[1].index,
            python_snapshots[1].has_gauss,
            python_snapshots[1].gauss_next,
            arguments.birth_state.python_random_gauss_primary_tape
                + cell * arguments.birth_config.gauss_tape_capacity,
            arguments.birth_state.python_random_gauss_secondary_tape
                + cell * arguments.birth_config.gauss_tape_capacity,
            arguments.birth_config.gauss_tape_capacity,
            python_snapshots[1].raw_draws,
        };
        namespace genome_v2 = eiel::genome_upgrade_v2_q10_v1;
        genome_v2::StagedChildV1* genome_v2_staged_children =
            genome_upgrade_v2_mode
            ? arena.at<genome_v2::StagedChildV1>(
                cell, layout, scratch::kGenomeUpgradeV2StagedChildren
            )
            : nullptr;
        const int64_t gv2_allele_cell_stride =
            2 * genome_v2::kProductCount * genome_v2::kParameterCount
            * arguments.capacity;
        const int64_t gv2_modifier_cell_stride =
            2 * genome_v2::kProductCount * arguments.capacity;
        const int64_t gv2_product_cell_stride =
            genome_v2::kProductCount * arguments.capacity;
        const int64_t gv2_phenotype_cell_stride =
            genome_v2::kPhenotypeCount * arguments.capacity;
        const genome_v2::ResidentPlanesV1 genome_v2_planes{
            genome_upgrade_v2_mode
                ? arguments.birth_state.genome_upgrade_v2_alleles_q10
                    + cell * gv2_allele_cell_stride
                : nullptr,
            genome_upgrade_v2_mode
                ? arguments.birth_state.genome_upgrade_v2_modifiers_q10
                    + cell * gv2_modifier_cell_stride
                : nullptr,
            genome_upgrade_v2_mode
                ? arguments.birth_state.genome_upgrade_v2_products_q10
                    + cell * gv2_product_cell_stride
                : nullptr,
            genome_upgrade_v2_mode
                ? arguments.birth_state.genome_upgrade_v2_phenotypes_q10
                    + cell * gv2_phenotype_cell_stride
                : nullptr,
            genome_upgrade_v2_mode
                ? arguments.birth_state.genome_upgrade_v2_expression_cost_q10
                    + cell * arguments.capacity
                : nullptr,
            genome_upgrade_v2_mode
                ? arguments.birth_state.genome_upgrade_v2_active_product_count
                    + cell * arguments.capacity
                : nullptr,
            genome_upgrade_v2_mode
                ? arguments.birth_state.genome_upgrade_v2_saturated_product_count
                    + cell * arguments.capacity
                : nullptr,
            arguments.capacity,
        };
        for (int64_t birth = 0; birth < birth_count; ++birth) {
            const int64_t parent_slot = identity_parent_slots[birth];
            const int64_t mate_slot =
                arguments.birth_outputs.planned_mate_slots[
                    tick_birth_offset + birth
                ];
            const uint8_t* const canonical_digest =
                canonical_hereditary_digest + cell * arguments.capacity * 32;
            bool parent_first = true;
            for (int32_t byte = 0; byte < 32; ++byte) {
                const uint8_t parent_value =
                    canonical_digest[parent_slot * 32 + byte];
                const uint8_t mate_value =
                    canonical_digest[mate_slot * 32 + byte];
                if (parent_value != mate_value) {
                    parent_first = parent_value < mate_value;
                    break;
                }
            }
            const int64_t canonical_parent_0_slot =
                parent_first ? parent_slot : mate_slot;
            const int64_t canonical_parent_1_slot =
                parent_first ? mate_slot : parent_slot;
            const diploid::StatusV1 diploid_status = diploid::stage_child_v1(
                allele_planes,
                canonical_parent_0_slot,
                canonical_parent_1_slot,
                diploid_rng,
                arguments.birth_state.inheritance_mutation_enabled[cell],
                arguments.birth_state.resource_handling_enabled[cell],
                arguments.birth_state.mutation_sigma_multiplier[cell],
                &staged_children[birth]
            );
            if (diploid_status != diploid::StatusV1::ok) {
                EIEL_RUNTIME_BIRTH_PREPARE_FAIL(
                    190 + static_cast<int32_t>(diploid_status)
                );
            }
            const float* phenotype = staged_children[birth].phenotype;
            scalar_f32[0 * birth_capacity + birth] =
                phenotype[diploid::kExplorationBias];
            scalar_f32[1 * birth_capacity + birth] =
                phenotype[diploid::kAggressionBias];
            scalar_f32[2 * birth_capacity + birth] =
                phenotype[diploid::kMutationRate];
            scalar_f32[3 * birth_capacity + birth] =
                phenotype[diploid::kReproductionThreshold];
            scalar_f32[4 * birth_capacity + birth] =
                phenotype[diploid::kActionTemperature];
            scalar_f32[5 * birth_capacity + birth] =
                phenotype[diploid::kDriveInertia];
            scalar_f32[6 * birth_capacity + birth] =
                phenotype[diploid::kDriveNoise];
            scalar_f32[7 * birth_capacity + birth] =
                phenotype[diploid::kDriveLogitGain];
            scalar_f32[8 * birth_capacity + birth] =
                phenotype[diploid::kPredLr];
            scalar_f32[9 * birth_capacity + birth] =
                phenotype[diploid::kDriveSpHunger];
            scalar_f32[10 * birth_capacity + birth] =
                phenotype[diploid::kDriveSpPain];
            scalar_f32[11 * birth_capacity + birth] =
                phenotype[diploid::kDriveSpFatigue];
            scalar_f32[12 * birth_capacity + birth] =
                phenotype[diploid::kDriveSpFear];
            scalar_f32[13 * birth_capacity + birth] =
                phenotype[diploid::kExpPriorStrength];
            scalar_f32[14 * birth_capacity + birth] =
                phenotype[diploid::kExpPriorNoise];
            scalar_f32[15 * birth_capacity + birth] =
                phenotype[diploid::kDriveBirthNoise];
            scalar_f32[16 * birth_capacity + birth] =
                phenotype[diploid::kReproductionThreshold];
            // Diploid RNG V2 bypasses the fixed-MLP scalar stager above. Keep
            // its newborn GRU-disabled invariant authoritative instead of
            // committing a reused scratch word from the prior transaction.
            scalar_f32[17 * birth_capacity + birth] = 0.0f;
            scalar_f32[18 * birth_capacity + birth] =
                phenotype[diploid::kAggressionBias];
            scalar_f32[19 * birth_capacity + birth] =
                phenotype[diploid::kActionTemperature];
            scalar_f32[20 * birth_capacity + birth] =
                phenotype[diploid::kDriveLogitGain];
            scalar_f32[21 * birth_capacity + birth] =
                phenotype[diploid::kDriveInertia];
            scalar_f32[22 * birth_capacity + birth] =
                phenotype[diploid::kDriveNoise];
            scalar_f32[23 * birth_capacity + birth] =
                phenotype[diploid::kPredLr];
            int32_t child_memory_dim =
                scalar_scratch.genome_memory_dim[birth];
            double memory_probability =
                0.25
                    * static_cast<double>(
                        phenotype[diploid::kMutationRate]
                    );
            if (memory_probability < 0.002) memory_probability = 0.002;
            if (memory_probability > 0.05) memory_probability = 0.05;
            int64_t memory_draws = 1;
            if (diploid_rng.random_double() < memory_probability) {
                static constexpr int32_t kMemoryLadder[5] = {
                    0, 8, 16, 32, 64
                };
                int32_t memory_index = 0;
                int32_t best_delta =
                    child_memory_dim >= 0
                    ? child_memory_dim
                    : -child_memory_dim;
                for (int32_t index = 1; index < 5; ++index) {
                    const int32_t raw_delta =
                        child_memory_dim - kMemoryLadder[index];
                    const int32_t delta =
                        raw_delta >= 0 ? raw_delta : -raw_delta;
                    if (delta < best_delta) {
                        best_delta = delta;
                        memory_index = index;
                    }
                }
                memory_index += diploid_rng.randbelow(2) == 0 ? -1 : 1;
                if (memory_index < 0) memory_index = 0;
                if (memory_index > 4) memory_index = 4;
                child_memory_dim = kMemoryLadder[memory_index];
                memory_draws += 1;
            }
            // This controller stages fixed-MLP policy rows only. Preserve the
            // accepted mutation draw schedule, but do not publish a GRU child
            // that this execution route cannot subsequently reproduce.
            child_memory_dim = 0;
            scalar_scratch.genome_memory_dim[birth] = child_memory_dim;
            scalar_scratch.policy_mem_dim[birth] = child_memory_dim;
            scalar_scratch.policy_use_gru[birth] = false;
            const int64_t child_slot =
                arguments.birth_outputs.planned_child_slots[
                    tick_birth_offset + birth
                ];
            const uint8_t* template_pack = digest_pack + 8;
            const int64_t canonical_template_bytes =
                static_cast<int64_t>(template_pack[0])
                | (static_cast<int64_t>(template_pack[1]) << 8)
                | (static_cast<int64_t>(template_pack[2]) << 16)
                | (static_cast<int64_t>(template_pack[3]) << 24);
            const int64_t diploid_payload_template_bytes =
                static_cast<int64_t>(template_pack[4])
                | (static_cast<int64_t>(template_pack[5]) << 8)
                | (static_cast<int64_t>(template_pack[6]) << 16)
                | (static_cast<int64_t>(template_pack[7]) << 24);
            const int64_t diploid_genotype_template_bytes =
                static_cast<int64_t>(template_pack[8])
                | (static_cast<int64_t>(template_pack[9]) << 8)
                | (static_cast<int64_t>(template_pack[10]) << 16)
                | (static_cast<int64_t>(template_pack[11]) << 24);
            const uint8_t* canonical_template = template_pack + 12;
            const uint8_t* diploid_payload_template =
                canonical_template + canonical_template_bytes;
            const uint8_t* diploid_genotype_template =
                diploid_payload_template + diploid_payload_template_bytes;
            canonical_hereditary_digest_v1::compute_v1(
                canonical_template,
                canonical_template_bytes,
                diploid_payload_template,
                diploid_payload_template_bytes,
                diploid_genotype_template,
                diploid_genotype_template_bytes,
                {
                    staged_children[birth].allele_0,
                    staged_children[birth].allele_1,
                    staged_children[birth].phenotype,
                    scalar_scratch.genome_hidden_size[birth],
                    child_memory_dim,
                    scalar_scratch.genome_sensor_channel_mask[birth],
                    scalar_scratch.genome_sensor_radius[birth],
                    scalar_scratch.genome_introspection_channel_mask[birth],
                    scalar_scratch.genome_self_regulation_control_mask[birth],
                    scalar_scratch.genome_self_modification_target_mask[birth],
                    scalar_scratch
                        .genome_directed_inheritance_channel_mask[birth],
                    child_memory_dim > 0,
                },
                canonical_hereditary_digest
                    + (
                        cell * arguments.capacity + child_slot
                    ) * 32
            );
            if (
                genome_upgrade_v2_mode
                && !genome_v2::stage_child_v1(
                    genome_v2_planes,
                    canonical_parent_0_slot,
                    canonical_parent_1_slot,
                    diploid_rng,
                    arguments.birth_state
                        .genome_upgrade_v2_mutation_rate_q10[cell],
                    arguments.birth_state
                        .genome_upgrade_v2_mutation_step_q10[cell],
                    &genome_v2_staged_children[birth]
                )
            ) {
                EIEL_RUNTIME_BIRTH_PREPARE_FAIL(236);
            }
            if (genome_upgrade_v2_mode) {
                genome_v2::ExpressionV1 child_expression{};
                const int64_t neutral_context[genome_v2::kContextCount] = {};
                genome_v2::express_v1(
                    genome_v2_staged_children[birth].genotype,
                    neutral_context,
                    &child_expression
                );
                const genome_v2::RuntimeProjectionV1 child_projection =
                    genome_v2::project_runtime_traits_v1(child_expression);
                const double parent_energy = effective_birth_energy_v1(
                    arguments, cell_capacity_offset, parent_slot
                );
                const double mate_energy = arguments.birth_config.requires_mate
                    ? effective_birth_energy_v1(
                          arguments, cell_capacity_offset, mate_slot
                      )
                    : 0.0;
                const double parent_base_transfer =
                    arguments.birth_config.requires_mate
                    ? parent_energy * birth_parent_energy_share_for_cell_v1(arguments.birth_config, cell)
                    : identity_child_energy[birth];
                const double mate_base_transfer =
                    arguments.birth_config.requires_mate
                    ? mate_energy * birth_parent_energy_share_for_cell_v1(arguments.birth_config, cell)
                    : 0.0;
                double parent_transfer = 0.0;
                double mate_transfer = 0.0;
                if (
                    !genome_v2::scale_offspring_transfer_v1(
                        parent_base_transfer,
                        child_projection.offspring_energy_factor_q10,
                        &parent_transfer
                    )
                    || (
                        arguments.birth_config.requires_mate
                        && !genome_v2::scale_offspring_transfer_v1(
                            mate_base_transfer,
                            child_projection.offspring_energy_factor_q10,
                            &mate_transfer
                        )
                    )
                    || parent_energy - parent_transfer
                        < static_cast<double>(
                            birth_minimum_energy_buffer_for_cell_v1(arguments.birth_config, cell)
                        )
                    || (
                        arguments.birth_config.requires_mate
                        && mate_energy - mate_transfer
                            < static_cast<double>(
                                birth_minimum_energy_buffer_for_cell_v1(arguments.birth_config, cell)
                            )
                    )
                ) {
                    EIEL_RUNTIME_BIRTH_PREPARE_FAIL(237);
                }
                identity_child_energy[birth] =
                    parent_transfer + mate_transfer;
                arguments.birth_outputs.planned_child_energy[
                    tick_birth_offset + birth
                ] = static_cast<float>(identity_child_energy[birth]);
            }
            statuses[0].scalar_semantic_draws +=
                staged_children[birth].semantic_draws + memory_draws;
            python_snapshots[1].semantic_draws +=
                staged_children[birth].semantic_draws + memory_draws;
            if (genome_upgrade_v2_mode) {
                statuses[0].scalar_semantic_draws +=
                    genome_v2_staged_children[birth].semantic_draws;
                python_snapshots[1].semantic_draws +=
                    genome_v2_staged_children[birth].semantic_draws;
            }
        }
        python_snapshots[1].index = diploid_rng.index;
        python_snapshots[1].has_gauss = diploid_rng.has_gauss_next;
        python_snapshots[1].gauss_next = diploid_rng.gauss_next;
        python_snapshots[1].raw_draws = diploid_rng.raw_draws;
    }
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t child = layout.mini_row(scratch::kMiniChild, birth);
        for (int64_t field = 0; field < 8; ++field) {
            mini_i32[field * mini_rows + child] =
                scalar_i32[field * birth_capacity + birth];
        }
        mini_mem_dim[child] = scalar_mem_dim[birth];
        mini_use_gru[child] = scalar_use_gru[birth];
        for (int64_t field = 0; field < 24; ++field) {
            mini_f32[field * mini_rows + child] =
                scalar_f32[field * birth_capacity + birth];
        }
    }
    scratch::Pcg64SnapshotV2* population_rng =
        arena.at<scratch::Pcg64SnapshotV2>(
            cell, layout, scratch::kPopulationPcg64Snapshot
        );
    for (int64_t word = 0; word < scratch::kPcg64StateWordCount; ++word) {
        population_rng[0].state_words[word] =
            arguments.birth_state.population_pcg64_state_words[cell * 4 + word];
    }
    for (int64_t word = 0; word < scratch::kPcg64BufferWordCount; ++word) {
        population_rng[0].buffer_words[word] =
            arguments.birth_state.population_pcg64_buffer_words[cell * 2 + word];
    }
    statuses[0].population_np_semantic_draws =
        arguments.birth_state.mutation_np_semantic_draws[cell];
    flattened_birth_drives_v1::
        prepare_ordinary_newborn_drives_external_scratch_v2(
            mini_slot_ids,
            mini_alive,
            mini_use_gru,
            mini_energy,
            mini_f32 + 9 * mini_rows,
            mini_f32 + 10 * mini_rows,
            mini_f32 + 11 * mini_rows,
            mini_f32 + 12 * mini_rows,
            mini_f32 + 14 * mini_rows,
            mini_f32 + 15 * mini_rows,
            mini_plan_parent,
            mini_plan_child,
            birth_count,
            mini_rows,
            arguments.max_energy,
            population_rng[0].state_words,
            population_rng[0].buffer_words,
            statuses[0].population_np_semantic_draws,
            arena.at<float>(cell, layout, scratch::kDriveRows),
            arena.at<float>(cell, layout, scratch::kBirthResetShockRows),
            statuses[0].drive_committed_count,
            statuses[0].birth_noise_rows,
            statuses[0].exp_prior_rows,
            statuses[0].standard_normal_rows,
            statuses[0].drive_semantic_draws,
            statuses[0].drive_status,
            staged_parent_order_mode
        );
    if (statuses[0].drive_status != 0) {
        EIEL_RUNTIME_BIRTH_PREPARE_FAIL(140 + statuses[0].drive_status);
    }

    float* staged_chemistry = arena.at<float>(
        cell, layout, scratch::kChemistryRows
    );
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const double energy_n = chemistry_detail::clamp_double(
            identity_child_energy[birth]
                / static_cast<double>(arguments.max_energy),
            0.0,
            1.0
        );
        const int64_t offset = birth * scratch::kChemistryWidth;
        staged_chemistry[offset] = static_cast<float>(0.40 + 0.40 * energy_n);
        staged_chemistry[offset + 1] = 0.55f;
        staged_chemistry[offset + 2] = 0.0f;
        staged_chemistry[offset + 3] = 0.0f;
        staged_chemistry[offset + 4] = 0.50f;
        staged_chemistry[offset + 5] = 0.25f;
    }
    statuses[0].chemistry_status = 0;

    bool* mutation_flags = arena.at<bool>(
        cell, layout, scratch::kMutationFlags
    );
    float* mutation_deltas = arena.at<float>(
        cell, layout, scratch::kMutationDeltas
    );
    for (int64_t birth = 0; birth < birth_capacity; ++birth) {
        mutation_flags[birth] = false;
        mutation_deltas[birth] = 0.0f;
    }
    statuses[0].policy_birth_tape_cursor = physical_policy_cursor;
    policy_v2::prepare_ordinary_fixed_mlp_child_policy_external_scratch_v2(
        mini_slot_ids,
        mini_alive,
        arguments.mlp_pack + cell_capacity_offset * arguments.packed_width,
        false,
        mini_use_gru,
        arguments.birth_state.mutation_torch_cpu_state
            + cell * arguments.birth_config.torch_state_bytes,
        arguments.birth_state.mutation_torch_cpu_semantic_draws[cell],
        arguments.birth_state.policy_birth_delta_tape
            + cell * arguments.birth_config.policy_tape_capacity
                * arguments.birth_config.policy_mutation_tape_width,
        arguments.birth_state.policy_birth_mutation_mask_tape
            + cell * arguments.birth_config.policy_tape_capacity
                * arguments.birth_config.policy_mutation_tape_width,
        arguments.birth_state.policy_birth_post_rng_state_packed_tape == nullptr
            ? nullptr
            : arguments.birth_state.policy_birth_post_rng_state_packed_tape
                + cell * arguments.birth_config.policy_tape_capacity
                    * torch_rng_pack::kPackedStateBytes,
        arguments.birth_state.policy_birth_semantic_draws_tape
            + cell * arguments.birth_config.policy_tape_capacity,
        physical_policy_cursor,
        mini_plan_parent,
        mini_plan_child,
        identity_parent_slots,
        birth_count,
        mini_rows,
        arguments.capacity,
        arguments.packed_width,
        arguments.birth_config.hmem_width,
        arguments.birth_config.torch_state_bytes,
        arguments.birth_config.policy_tape_capacity,
        arguments.birth_config.policy_mutation_tape_width,
        arena.at<float>(cell, layout, scratch::kPolicyMlpRows),
        arena.at<float>(cell, layout, scratch::kHmemRows),
        arena.at<uint8_t>(cell, layout, scratch::kTorchState),
        statuses[0].policy_birth_tape_cursor,
        statuses[0].policy_state_semantic_draws,
        statuses[0].policy_committed_count,
        statuses[0].policy_semantic_draws,
        statuses[0].policy_mutation_events,
        statuses[0].policy_mutated_elements,
        statuses[0].policy_mutation_delta_total,
        mutation_flags,
        mutation_deltas,
        statuses[0].policy_status,
        staged_parent_order_mode
    );
    if (statuses[0].policy_status != 0) {
        EIEL_RUNTIME_BIRTH_PREPARE_FAIL(160 + statuses[0].policy_status);
    }

    int64_t absolute_python_raw_draws = 0;
    int64_t absolute_policy_cursor = 0;
    if (
        !absolute_cursor_from_physical_v1(
            python_snapshots[1].raw_draws,
            arguments.birth_state.python_random_raw_draw_base[cell],
            arguments.birth_config.gauss_tape_capacity,
            &absolute_python_raw_draws
        )
        || !absolute_cursor_from_physical_v1(
            statuses[0].policy_birth_tape_cursor,
            arguments.birth_state.policy_birth_tape_cursor_base[cell],
            arguments.birth_config.policy_tape_capacity,
            &absolute_policy_cursor
        )
    ) {
        EIEL_RUNTIME_BIRTH_PREPARE_FAIL(26);
    }

    int64_t* parent_agent_ids = arena.at<int64_t>(
        cell, layout, scratch::kParentAgentIds
    );
    int32_t* child_founders = arena.at<int32_t>(
        cell, layout, scratch::kChildLineageFounderIds
    );
    int32_t* child_depths = arena.at<int32_t>(
        cell, layout, scratch::kChildLineageDepths
    );
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = identity_parent_slots[birth];
        const int64_t parent_index = cell_capacity_offset + parent_slot;
        const int64_t parent_id = arguments.slot_to_agent_id[parent_index];
        const int32_t parent_founder =
            arguments.event_state.lineage_founder_id[parent_index];
        const int32_t parent_depth =
            arguments.event_state.lineage_depth[parent_index];
        const int64_t founder = parent_founder >= 0
            ? parent_founder
            : parent_id;
        if (
            founder < 0 || founder > INT32_MAX
            || parent_depth < 0 || parent_depth == INT32_MAX
        ) {
            EIEL_RUNTIME_BIRTH_PREPARE_FAIL(182);
        }
        parent_agent_ids[birth] = parent_id;
        child_founders[birth] = static_cast<int32_t>(founder);
        child_depths[birth] = parent_depth + 1;
    }

    events_v2::preflight_ordinary_birth_events_external_scratch_v2(
        {
            arguments.slot_to_agent_id + cell_capacity_offset,
            arguments.event_state.lineage_founder_id + cell_capacity_offset,
            arguments.event_state.lineage_depth + cell_capacity_offset,
            arguments.capacity,
        },
        {
            arguments.birth_outputs.decision_parent_slots + tick_capacity_offset,
            arguments.birth_outputs.decision_reason_codes + tick_capacity_offset,
            decision_count,
            arguments.capacity,
        },
        {
            identity_parent_slots,
            parent_agent_ids,
            identity_child_slots,
            identity_child_ids,
            child_founders,
            child_depths,
            mutation_flags,
            mutation_deltas,
            birth_count,
            identity_count[0],
            birth_capacity,
        },
        {
            arguments.event_state.active_count[cell],
            arguments.event_state.capacity,
            arguments.event_state.sequence_base[cell],
        },
        statuses[0].event_status,
        statuses[0].required_event_rows
    );
    if (statuses[0].event_status != events_v2::kStatusOk) {
        EIEL_RUNTIME_BIRTH_PREPARE_FAIL(
            statuses[0].event_status == events_v2::kStatusCapacityExceeded
                ? 25
                : 205
        );
    }
    if (
        statuses[0].required_event_rows < 0
        || statuses[0].required_event_rows
            > layout.segment(scratch::kEventStagingRows).bytes
                / static_cast<int64_t>(sizeof(scratch::BirthEventStagingRowV2))
    ) {
        EIEL_RUNTIME_BIRTH_PREPARE_FAIL(205);
    }

    scratch::BirthEventStagingRowV2* staged_events =
        arena.at<scratch::BirthEventStagingRowV2>(
            cell, layout, scratch::kEventStagingRows
        );
    int64_t event_ordinal = 0;
    int64_t success_ordinal = 0;
    const int64_t event_cursor = arguments.event_state.active_count[cell];
    const int64_t event_sequence_base =
        arguments.event_state.sequence_base[cell];
    const int64_t generation = arguments.mechanics_config.generation;
    const int64_t absolute_tick =
        arguments.mechanics_config.start_tick + controller_tick;
    for (int64_t decision = 0; decision < decision_count; ++decision) {
        const int64_t parent_slot = arguments.birth_outputs.decision_parent_slots[
            tick_capacity_offset + decision
        ];
        const int64_t parent_index = cell_capacity_offset + parent_slot;
        const int64_t parent_id = arguments.slot_to_agent_id[parent_index];
        const int32_t reason = arguments.birth_outputs.decision_reason_codes[
            tick_capacity_offset + decision
        ];
        if (
            (reason >= 1 && reason <= 3)
            || (
                arguments.birth_config.requires_mate
                && reason >= 5 && reason <= 6
            )
        ) {
            const int32_t raw_founder =
                arguments.event_state.lineage_founder_id[parent_index];
            staged_events[event_ordinal] = {
                generation,
                absolute_tick,
                event_sequence_base + event_cursor + event_ordinal,
                parent_id,
                persistent_birth_events::kSentinelId,
                parent_slot,
                persistent_birth_events::kSentinelId,
                persistent_birth_events::kSentinelId,
                persistent_birth_events::kSentinelId,
                persistent_birth_events::kFailedBirthEventType,
                raw_founder,
                arguments.event_state.lineage_depth[parent_index],
                persistent_birth_events::kFailedBirthEventType,
                reason,
                static_cast<float>(reason),
                false,
                {},
            };
            ++event_ordinal;
            continue;
        }
        if (reason != 0) {
            continue;
        }
        const int64_t child_slot = identity_child_slots[success_ordinal];
        const int64_t child_id = identity_child_ids[success_ordinal];
        if (mutation_flags[success_ordinal]) {
            staged_events[event_ordinal] = {
                generation,
                absolute_tick,
                event_sequence_base + event_cursor + event_ordinal,
                child_id,
                child_id,
                child_slot,
                child_slot,
                parent_agent_ids[success_ordinal],
                child_id,
                persistent_birth_events::kMutationEventType,
                child_founders[success_ordinal],
                child_depths[success_ordinal],
                persistent_birth_events::kMutationEventType,
                -1,
                mutation_deltas[success_ordinal],
                true,
                {},
            };
            ++event_ordinal;
        }
        staged_events[event_ordinal] = {
            generation,
            absolute_tick,
            event_sequence_base + event_cursor + event_ordinal,
            parent_agent_ids[success_ordinal],
            child_id,
            identity_parent_slots[success_ordinal],
            child_slot,
            parent_agent_ids[success_ordinal],
            child_id,
            persistent_birth_events::kBirthCommittedEventType,
            child_founders[success_ordinal],
            child_depths[success_ordinal],
            persistent_birth_events::kBirthCommittedEventType,
            -1,
            arguments.birth_outputs.planned_child_energy[
                tick_birth_offset + success_ordinal
            ],
            true,
            {},
        };
        ++event_ordinal;
        ++success_ordinal;
    }
    if (
        event_ordinal != statuses[0].required_event_rows
        || success_ordinal != birth_count
    ) {
        EIEL_RUNTIME_BIRTH_PREPARE_FAIL(205);
    }
    if (
        qualification_fault_stage < scratch::kGenomeUpgradeV2FaultDisabled
        || (
            qualification_fault_stage
                > scratch::
                    kGenomeUpgradeV2FaultImmediatelyAfterCommitBeforeReporting
            && (
                qualification_fault_stage
                    < scratch::kGenomeUpgradeV2FaultTurnoverMidCull
                || qualification_fault_stage
                    > scratch::
                        kGenomeUpgradeV2FaultTurnoverAfterCommitBeforeCheckpoint
            )
        )
    ) {
        EIEL_RUNTIME_BIRTH_PREPARE_FAIL(240);
    }
    if (
        qualification_fault_stage
            >= scratch::kGenomeUpgradeV2FaultAfterDestinationReservation
        && qualification_fault_stage
            <= scratch::kGenomeUpgradeV2FaultAfterShadowWrite
    ) {
        EIEL_RUNTIME_BIRTH_PREPARE_FAIL(240 + qualification_fault_stage);
    }
    statuses[0].transaction_status = 0;
#undef EIEL_RUNTIME_BIRTH_PREPARE_FAIL
    return 0;
}

__global__ void flattened_controller_birth_prepare_kernel_v1(
    const Phase2KernelArguments arguments,
    const int64_t controller_tick
) {
    if (threadIdx.x != 0) {
        return;
    }
    const int64_t cell = static_cast<int64_t>(blockIdx.x);
    if (
        !arguments.birth_config.enabled
        || !cross_path_resident_control_v1::ordinary_live_birth_cell_v1(
            arguments.birth_config.cross_path_control, cell
        )
    ) {
        return;
    }
    namespace scratch = flattened_birth_scratch_v1;
    namespace flattened_drives = flattened_birth_drives_v1;
    namespace scalar = persistent_birth_scalar_genome;
    namespace policy = persistent_birth_policy;
    namespace chemistry_detail = persistent_chemistry::detail;

    if (arguments.status[cell] != 0) {
        return;
    }
    const int64_t output_tick = arguments.output_ticks_per_cell == 1
        ? 0
        : controller_tick;
    const int64_t cell_tick =
        cell * arguments.output_ticks_per_cell + output_tick;
    const int64_t cell_capacity_offset = cell * arguments.capacity;
    const int64_t birth_capacity = arguments.birth_config.max_births_per_tick;
    const int64_t tick_birth_offset = cell_tick * birth_capacity;
    const int32_t runtime_status =
        prepare_flattened_runtime_birth_transaction_v2(
            arguments, cell, controller_tick
        );
    arguments.birth_outputs.combined_status[cell_tick] = runtime_status == 0
        ? 0
        : (runtime_status == 1
               ? 1
               : (runtime_status == 4
                      ? 4
                      : (runtime_status == 25 ? 3 : 2)));
    return;
    int32_t* transaction_status = scratch::at<int32_t>(
        arguments.birth_scratch, cell, scratch::kTransactionStatus
    );
    int64_t* prepared_stamp = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPreparedStamp
    );
    transaction_status[0] = 0;
    prepared_stamp[0] = 0;

    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kMiniScalarCommitted
    )[0] = 0;
    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kMiniScalarSemanticDraws
    )[0] = 0;
    scratch::at<int32_t>(
        arguments.birth_scratch, cell, scratch::kMiniScalarStatus
    )[0] = 0;
    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kMiniDriveCommitted
    )[0] = 0;
    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kMiniBirthNoiseRows
    )[0] = 0;
    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kMiniExpPriorRows
    )[0] = 0;
    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kMiniStandardNormalRows
    )[0] = 0;
    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kMiniDriveSemanticDraws
    )[0] = 0;
    scratch::at<int32_t>(
        arguments.birth_scratch, cell, scratch::kMiniDriveStatus
    )[0] = 0;
    scratch::at<int32_t>(
        arguments.birth_scratch, cell, scratch::kMiniChemistryStatus
    )[0] = 0;
    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kMiniPolicyCommitted
    )[0] = 0;
    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kMiniPolicySemanticDraws
    )[0] = 0;
    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kMiniPolicyMutationEvents
    )[0] = 0;
    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kMiniPolicyMutatedElements
    )[0] = 0;
    scratch::at<double>(
        arguments.birth_scratch, cell, scratch::kMiniPolicyMutationDeltaTotal
    )[0] = 0.0;
    scratch::at<bool>(
        arguments.birth_scratch, cell, scratch::kMiniPolicyMutationOccurred
    )[0] = false;
    scratch::at<float>(
        arguments.birth_scratch, cell, scratch::kMiniPolicyMutationDelta
    )[0] = 0.0f;
    scratch::at<int32_t>(
        arguments.birth_scratch, cell, scratch::kMiniPolicyStatus
    )[0] = 0;
    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kRequiredEventRows
    )[0] = 0;
    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPreparedParentSlot
    )[0] = -1;
    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPreparedChildSlot
    )[0] = -1;
    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPreparedChildId
    )[0] = -1;

    if (arguments.full_tick_outputs.residual_release_status[cell_tick] != 0) {
        transaction_status[0] = 4;
        return;
    }
    if (arguments.birth_outputs.plan_status[cell_tick] != 0) {
        transaction_status[0] = 1;
        return;
    }
    const int64_t planned_count = arguments.birth_outputs.planned_count[cell_tick];
    const int64_t decision_count = arguments.birth_outputs.decision_count[cell_tick];
    if (planned_count < 0 || planned_count > birth_capacity) {
        transaction_status[0] = 110;
        return;
    }
    if (decision_count < 0 || decision_count > arguments.capacity) {
        transaction_status[0] = 111;
        return;
    }
    if (birth_capacity < 0 || birth_capacity > 2) {
        transaction_status[0] = 112;
        return;
    }
    if (arguments.drive_width < 4 || arguments.drive_width > 8) {
        transaction_status[0] = 113;
        return;
    }
    if (
        arguments.full_tick_state.chemistry_width < 6
        || arguments.full_tick_state.chemistry_width > 8
    ) {
        transaction_status[0] = 114;
        return;
    }
    if (!::isfinite(arguments.max_energy) || arguments.max_energy <= 0.0f) {
        transaction_status[0] = 115;
        return;
    }
    int64_t physical_policy_cursor = 0;
    if (
        arguments.birth_state.policy_birth_tape_cursor_base[cell]
            > INT64_MAX - arguments.birth_config.policy_tape_capacity
        || !physical_cursor_from_absolute_v1(
            arguments.birth_state.policy_birth_tape_cursor[cell],
            arguments.birth_state.policy_birth_tape_cursor_base[cell],
            arguments.birth_config.policy_tape_capacity,
            &physical_policy_cursor
        )
    ) {
        transaction_status[0] = 116;
        return;
    }
    if (
        planned_count
        > arguments.birth_config.policy_tape_capacity - physical_policy_cursor
    ) {
        transaction_status[0] = 117;
        return;
    }
    int64_t physical_python_raw_draws = 0;
    if (
        arguments.birth_state.python_random_raw_draw_base[cell]
            > INT64_MAX - arguments.birth_config.gauss_tape_capacity
        || !physical_cursor_from_absolute_v1(
            arguments.birth_state.python_random_raw_draws[cell],
            arguments.birth_state.python_random_raw_draw_base[cell],
            arguments.birth_config.gauss_tape_capacity,
            &physical_python_raw_draws
        )
    ) {
        transaction_status[0] = 118;
        return;
    }
    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPythonRawDraws
    )[0] = physical_python_raw_draws;
    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kMiniPolicyCursor
    )[0] = physical_policy_cursor;

    int64_t required_event_rows = 0;
    int64_t success_ordinal = 0;
    const int64_t tick_capacity_offset = cell_tick * arguments.capacity;
    const int64_t policy_cursor = physical_policy_cursor;
    for (int64_t decision = 0; decision < decision_count; ++decision) {
        const int64_t decision_parent_slot =
            arguments.birth_outputs.decision_parent_slots[
                tick_capacity_offset + decision
            ];
        if (
            decision_parent_slot < 0
            || decision_parent_slot >= arguments.capacity
            || arguments.slot_to_agent_id[
                cell_capacity_offset + decision_parent_slot
            ] < 0
            || arguments.event_state.lineage_depth[
                cell_capacity_offset + decision_parent_slot
            ] < 0
        ) {
            transaction_status[0] = 102;
            return;
        }
        const int32_t reason = arguments.birth_outputs.decision_reason_codes[
            tick_capacity_offset + decision
        ];
        if (reason == 0) {
            if (
                success_ordinal >= planned_count
                || decision_parent_slot
                    != arguments.birth_outputs.planned_parent_slots[
                        tick_birth_offset + success_ordinal
                    ]
            ) {
                transaction_status[0] = 102;
                return;
            }
            bool mutation_occurred = false;
            const int64_t tape_row = policy_cursor + success_ordinal;
            for (int64_t element = 0; element < arguments.packed_width; ++element) {
                const int64_t mutation_element =
                    arguments.birth_config.policy_mutation_tape_width == 1
                    ? 0
                    : element;
                mutation_occurred = mutation_occurred
                    || arguments.birth_state.policy_birth_mutation_mask_tape[
                        (cell * arguments.birth_config.policy_tape_capacity + tape_row)
                            * arguments.birth_config.policy_mutation_tape_width
                            + mutation_element
                    ];
            }
            required_event_rows += mutation_occurred ? 2 : 1;
            ++success_ordinal;
        } else if (
            (reason >= 1 && reason <= 3)
            || (
                arguments.birth_config.requires_mate
                && reason >= 5 && reason <= 6
            )
        ) {
            ++required_event_rows;
        } else if (
            !(
                (arguments.birth_config.requires_mate && reason == 4)
                || (reason >= 7 && reason <= 9)
                || (
                    arguments.birth_config.requires_mate
                    && reason >= 10 && reason <= 11
                )
            )
        ) {
            transaction_status[0] = 103;
            return;
        }
    }
    const int64_t event_cursor = arguments.event_state.active_count[cell];
    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kRequiredEventRows
    )[0] = required_event_rows;
    if (
        success_ordinal != planned_count
        || !event_sequence_contract_valid_v1(arguments.event_state, cell)
        || event_cursor < 0 || event_cursor > arguments.event_state.capacity
        || required_event_rows > arguments.event_state.capacity - event_cursor
    ) {
        if (!event_sequence_contract_valid_v1(arguments.event_state, cell)) {
            fail_event_sequence_contract_v1(arguments.event_state, cell);
        }
        transaction_status[0] = 25;
        return;
    }
    // The runtime-sized three-role scratch path below is the qualified
    // mate-bearing transaction route. Fixed B1/B2 scratch has only
    // parent/child rows, so fail closed instead of silently constructing an
    // asexual child when a mate is required.
    if (arguments.birth_config.requires_mate && birth_capacity <= 2) {
        transaction_status[0] = 213;
        return;
    }
    if (birth_capacity > 1) {
        transaction_status[0] = prepare_flattened_b2_transaction_v2(
            arguments,
            cell,
            controller_tick,
            physical_policy_cursor,
            physical_python_raw_draws
        );
        if (transaction_status[0] == 0) {
            prepared_stamp[0] = kFlattenedPreparedStamp;
        }
        return;
    }
    if (planned_count == 0) {
        transaction_status[0] = 0;
        prepared_stamp[0] = kFlattenedPreparedStamp;
        return;
    }

    const int64_t parent_slot =
        arguments.birth_outputs.planned_parent_slots[tick_birth_offset];
    const int64_t free_count =
        arguments.full_tick_state.slot_free_queue_count[cell];
    if (
        parent_slot < 0 || parent_slot >= arguments.capacity
        || free_count < 0 || free_count > arguments.capacity
    ) {
        transaction_status[0] = 104;
        return;
    }
    const int64_t parent_id =
        arguments.slot_to_agent_id[cell_capacity_offset + parent_slot];
    const double parent_energy = ::isfinite(
        arguments.policy_energy_observation[cell_capacity_offset + parent_slot]
    )
        ? arguments.policy_energy_observation[cell_capacity_offset + parent_slot]
        : static_cast<double>(
            arguments.energy[cell_capacity_offset + parent_slot]
        );
    const double child_energy =
        parent_energy * birth_parent_energy_share_for_cell_v1(arguments.birth_config, cell);
    if (
        parent_id < 0
        || !arguments.alive_mask[cell_capacity_offset + parent_slot]
        || arguments.health[cell_capacity_offset + parent_slot] <= 0.0f
        || arguments.birth_outputs.planned_repro_cooldown[
            cell_capacity_offset + parent_slot
        ] != 0
        || parent_energy < static_cast<double>(
            arguments.birth_state.repro_threshold[
                cell_capacity_offset + parent_slot
            ]
        )
        || parent_energy - child_energy
            < static_cast<double>(birth_minimum_energy_buffer_for_cell_v1(arguments.birth_config, cell))
    ) {
        transaction_status[0] = 105;
        return;
    }
    const int32_t child_y =
        arguments.birth_outputs.planned_spawn_positions[tick_birth_offset * 2];
    const int32_t child_x =
        arguments.birth_outputs.planned_spawn_positions[tick_birth_offset * 2 + 1];
    const int64_t cell_grid_offset = cell * arguments.height * arguments.width;
    if (
        child_y < 0 || child_y >= arguments.height
        || child_x < 0 || child_x >= arguments.width
        || arguments.agent_grid[
            cell_grid_offset + static_cast<int64_t>(child_y) * arguments.width
                + child_x
        ] != -1
    ) {
        transaction_status[0] = 106;
        return;
    }
    int64_t child_slot = -1;
    for (int64_t queue_index = 0; queue_index < free_count; ++queue_index) {
        const int64_t candidate = arguments.full_tick_state.slot_free_queue[
            cell_capacity_offset + queue_index
        ];
        if (
            candidate >= 0 && candidate < arguments.capacity
            && !arguments.alive_mask[cell_capacity_offset + candidate]
            && arguments.slot_to_agent_id[cell_capacity_offset + candidate] < 0
        ) {
            child_slot = candidate;
            break;
        }
    }
    int64_t child_id = arguments.birth_state.population_next_id[cell] < 1
        ? 1
        : arguments.birth_state.population_next_id[cell];
    for (int64_t slot = 0; slot < arguments.capacity; ++slot) {
        const int64_t agent_id =
            arguments.slot_to_agent_id[cell_capacity_offset + slot];
        if (agent_id >= child_id) {
            child_id = agent_id + 1;
        }
    }
    if (
        child_slot < 0 || child_slot == parent_slot
        || !active_identity::birth_id_range_fits_grid_v1(
            child_id, planned_count
        )
    ) {
        transaction_status[0] = 20;
        return;
    }

    const auto layout = scratch::make_dynamic_layout(
        arguments.packed_width,
        arguments.birth_config.hmem_width,
        arguments.birth_config.torch_state_bytes
    );
    if (
        layout.required_bytes > arguments.birth_scratch.bytes_per_cell
        || layout.required_bytes > scratch::kBytesPerCell
    ) {
        transaction_status[0] = 108;
        return;
    }

    int64_t* mini_slot_ids = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kMiniSlotIds
    );
    bool* mini_alive = scratch::at<bool>(
        arguments.birth_scratch, cell, scratch::kMiniAlive
    );
    int32_t* mini_genome_i32 = scratch::at<int32_t>(
        arguments.birth_scratch, cell, scratch::kMiniGenomeInt32
    );
    int64_t* mini_policy_mem_dim = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kMiniPolicyMemDim
    );
    bool* mini_policy_use_gru = scratch::at<bool>(
        arguments.birth_scratch, cell, scratch::kMiniPolicyUseGru
    );
    float* mini_genome_f32 = scratch::at<float>(
        arguments.birth_scratch, cell, scratch::kMiniGenomeFloat32
    );
    int64_t* mini_plan_parent = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kMiniPlanParent
    );
    int64_t* mini_plan_child = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kMiniPlanChild
    );
    int64_t* mini_plan_count = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kMiniPlanCount
    );
    double* mini_energy = scratch::at<double>(
        arguments.birth_scratch, cell, scratch::kMiniEnergy
    );

    mini_slot_ids[0] = parent_id;
    mini_slot_ids[1] = child_id;
    mini_alive[0] = true;
    mini_alive[1] = true;
    mini_plan_parent[0] = 0;
    mini_plan_child[0] = 1;
    mini_plan_count[0] = 1;
    mini_energy[0] = parent_energy;
    mini_energy[1] = child_energy;

#define EIEL_COPY_MINI_I32(index, source)                                      \
    do {                                                                       \
        mini_genome_i32[(index) * 2] =                                         \
            (source)[cell_capacity_offset + parent_slot];                      \
        mini_genome_i32[(index) * 2 + 1] = 0;                                  \
    } while (false)
    EIEL_COPY_MINI_I32(0, arguments.birth_state.genome_hidden_size);
    EIEL_COPY_MINI_I32(1, arguments.birth_state.genome_memory_dim);
    EIEL_COPY_MINI_I32(2, arguments.birth_state.genome_sensor_channel_mask);
    EIEL_COPY_MINI_I32(3, arguments.birth_state.genome_sensor_radius);
    EIEL_COPY_MINI_I32(4, arguments.birth_state.genome_introspection_channel_mask);
    EIEL_COPY_MINI_I32(5, arguments.birth_state.genome_self_regulation_control_mask);
    EIEL_COPY_MINI_I32(6, arguments.birth_state.genome_self_modification_target_mask);
    EIEL_COPY_MINI_I32(7, arguments.birth_state.genome_directed_inheritance_channel_mask);
#undef EIEL_COPY_MINI_I32

    mini_policy_mem_dim[0] = arguments.birth_state.policy_mem_dim[
        cell_capacity_offset + parent_slot
    ];
    mini_policy_mem_dim[1] = 0;
    mini_policy_use_gru[0] = arguments.birth_state.policy_use_gru[
        cell_capacity_offset + parent_slot
    ];
    mini_policy_use_gru[1] = false;

#define EIEL_COPY_MINI_F32(index, source)                                      \
    do {                                                                       \
        mini_genome_f32[(index) * 2] =                                         \
            (source)[cell_capacity_offset + parent_slot];                      \
        mini_genome_f32[(index) * 2 + 1] = 0.0f;                               \
    } while (false)
    EIEL_COPY_MINI_F32(0, arguments.birth_state.genome_exploration_bias);
    EIEL_COPY_MINI_F32(1, arguments.birth_state.genome_aggression_bias);
    EIEL_COPY_MINI_F32(2, arguments.birth_state.genome_mutation_rate);
    EIEL_COPY_MINI_F32(3, arguments.birth_state.genome_reproduction_threshold);
    EIEL_COPY_MINI_F32(4, arguments.birth_state.genome_action_temperature);
    EIEL_COPY_MINI_F32(5, arguments.birth_state.genome_drive_inertia);
    EIEL_COPY_MINI_F32(6, arguments.birth_state.genome_drive_noise);
    EIEL_COPY_MINI_F32(7, arguments.birth_state.genome_drive_logit_gain);
    EIEL_COPY_MINI_F32(8, arguments.birth_state.genome_pred_lr);
    EIEL_COPY_MINI_F32(9, arguments.birth_state.genome_drive_sp_hunger);
    EIEL_COPY_MINI_F32(10, arguments.birth_state.genome_drive_sp_pain);
    EIEL_COPY_MINI_F32(11, arguments.birth_state.genome_drive_sp_fatigue);
    EIEL_COPY_MINI_F32(12, arguments.birth_state.genome_drive_sp_fear);
    EIEL_COPY_MINI_F32(13, arguments.birth_state.genome_exp_prior_strength);
    EIEL_COPY_MINI_F32(14, arguments.birth_state.genome_exp_prior_noise);
    EIEL_COPY_MINI_F32(15, arguments.birth_state.genome_drive_birth_noise);
    EIEL_COPY_MINI_F32(16, arguments.birth_state.repro_threshold);
    EIEL_COPY_MINI_F32(17, arguments.full_tick_state.gru_upkeep_cost);
    EIEL_COPY_MINI_F32(18, arguments.attack_bias);
    EIEL_COPY_MINI_F32(19, arguments.action_temperature);
    EIEL_COPY_MINI_F32(20, arguments.drive_logit_gain);
    EIEL_COPY_MINI_F32(21, arguments.full_tick_state.drive_inertia);
    EIEL_COPY_MINI_F32(22, arguments.full_tick_state.drive_noise);
    EIEL_COPY_MINI_F32(23, arguments.birth_state.pred_lr);
#undef EIEL_COPY_MINI_F32

    uint32_t* python_post_words = scratch::at<uint32_t>(
        arguments.birth_scratch, cell, scratch::kPythonPostWords
    );
    for (int64_t word = 0; word < 624; ++word) {
        python_post_words[word] =
            arguments.birth_state.python_random_words[cell * 624 + word];
    }
    scratch::at<int64_t>(arguments.birth_scratch, cell, scratch::kPythonIndex)[0] =
        arguments.birth_state.python_random_index[cell];
    scratch::at<bool>(arguments.birth_scratch, cell, scratch::kPythonHasGauss)[0] =
        arguments.birth_state.python_random_has_gauss[cell];
    scratch::at<double>(arguments.birth_scratch, cell, scratch::kPythonGaussNext)[0] =
        arguments.birth_state.python_random_gauss_next[cell];
    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPythonRawDraws
    )[0] = physical_python_raw_draws;
    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPythonSemanticDraws
    )[0] = arguments.birth_state.mutation_py_semantic_draws[cell];

    scalar::commit_ordinary_fixed_mlp_child_scalar_genome_with_trial_scratch_serial(
        {
            mini_slot_ids,
            mini_alive,
            mini_genome_i32 + 0 * 2,
            mini_genome_i32 + 1 * 2,
            mini_genome_i32 + 2 * 2,
            mini_genome_i32 + 3 * 2,
            mini_genome_i32 + 4 * 2,
            mini_genome_i32 + 5 * 2,
            mini_genome_i32 + 6 * 2,
            mini_genome_i32 + 7 * 2,
            mini_policy_mem_dim,
            mini_policy_use_gru,
            mini_genome_f32 + 0 * 2,
            mini_genome_f32 + 1 * 2,
            mini_genome_f32 + 2 * 2,
            mini_genome_f32 + 3 * 2,
            mini_genome_f32 + 4 * 2,
            mini_genome_f32 + 5 * 2,
            mini_genome_f32 + 6 * 2,
            mini_genome_f32 + 7 * 2,
            mini_genome_f32 + 8 * 2,
            mini_genome_f32 + 9 * 2,
            mini_genome_f32 + 10 * 2,
            mini_genome_f32 + 11 * 2,
            mini_genome_f32 + 12 * 2,
            mini_genome_f32 + 13 * 2,
            mini_genome_f32 + 14 * 2,
            mini_genome_f32 + 15 * 2,
            mini_genome_f32 + 16 * 2,
            mini_genome_f32 + 17 * 2,
            mini_genome_f32 + 18 * 2,
            mini_genome_f32 + 19 * 2,
            mini_genome_f32 + 20 * 2,
            mini_genome_f32 + 21 * 2,
            mini_genome_f32 + 22 * 2,
            mini_genome_f32 + 23 * 2,
        },
        {mini_plan_parent, mini_plan_child, mini_plan_count},
        {
            python_post_words,
            scratch::at<int64_t>(arguments.birth_scratch, cell, scratch::kPythonIndex),
            scratch::at<bool>(arguments.birth_scratch, cell, scratch::kPythonHasGauss),
            scratch::at<double>(arguments.birth_scratch, cell, scratch::kPythonGaussNext),
            arguments.birth_state.python_random_gauss_primary_tape
                + cell * arguments.birth_config.gauss_tape_capacity,
            arguments.birth_state.python_random_gauss_secondary_tape
                + cell * arguments.birth_config.gauss_tape_capacity,
            scratch::at<int64_t>(arguments.birth_scratch, cell, scratch::kPythonRawDraws),
            scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kPythonSemanticDraws
            ),
        },
        {
            scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kMiniScalarCommitted
            ),
            scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kMiniScalarSemanticDraws
            ),
            scratch::at<int32_t>(
                arguments.birth_scratch, cell, scratch::kMiniScalarStatus
            ),
        },
        {1, 2, arguments.birth_config.gauss_tape_capacity,
         arguments.birth_config.sensor_radius_max},
        scratch::at<uint32_t>(
            arguments.birth_scratch, cell, scratch::kPythonTrialWords
        )
    );
    if (
        scratch::at<int32_t>(
            arguments.birth_scratch, cell, scratch::kMiniScalarStatus
        )[0] != 0
    ) {
        transaction_status[0] = 120 + scratch::at<int32_t>(
            arguments.birth_scratch, cell, scratch::kMiniScalarStatus
        )[0];
        return;
    }

    uint64_t* population_state = scratch::at<uint64_t>(
        arguments.birth_scratch, cell, scratch::kPopulationPcgState
    );
    int64_t* population_buffer = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPopulationPcgBuffer
    );
    int64_t* population_semantic = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPopulationSemanticDraws
    );
    for (int64_t word = 0; word < 4; ++word) {
        population_state[word] =
            arguments.birth_state.population_pcg64_state_words[cell * 4 + word];
    }
    for (int64_t word = 0; word < 2; ++word) {
        population_buffer[word] =
            arguments.birth_state.population_pcg64_buffer_words[cell * 2 + word];
    }
    population_semantic[0] = arguments.birth_state.mutation_np_semantic_draws[cell];
    float* mini_drives = scratch::at<float>(
        arguments.birth_scratch, cell, scratch::kMiniDrives
    );
    float* mini_birth_shock = scratch::at<float>(
        arguments.birth_scratch, cell, scratch::kMiniBirthShock
    );
    flattened_drives::prepare_ordinary_newborn_drives_scratch_v1(
        mini_slot_ids,
        mini_alive,
        mini_policy_use_gru,
        mini_energy,
        mini_genome_f32 + 9 * 2,
        mini_genome_f32 + 10 * 2,
        mini_genome_f32 + 11 * 2,
        mini_genome_f32 + 12 * 2,
        mini_genome_f32 + 14 * 2,
        mini_genome_f32 + 15 * 2,
        1,
        arguments.max_energy,
        population_state,
        population_buffer,
        population_semantic[0],
        mini_drives,
        mini_birth_shock[0],
        scratch::at<int64_t>(
            arguments.birth_scratch, cell, scratch::kMiniDriveCommitted
        )[0],
        scratch::at<int64_t>(
            arguments.birth_scratch, cell, scratch::kMiniBirthNoiseRows
        )[0],
        scratch::at<int64_t>(
            arguments.birth_scratch, cell, scratch::kMiniExpPriorRows
        )[0],
        scratch::at<int64_t>(
            arguments.birth_scratch, cell, scratch::kMiniStandardNormalRows
        )[0],
        scratch::at<int64_t>(
            arguments.birth_scratch, cell, scratch::kMiniDriveSemanticDraws
        )[0],
        scratch::at<int32_t>(
            arguments.birth_scratch, cell, scratch::kMiniDriveStatus
        )[0]
    );
    if (
        scratch::at<int32_t>(
            arguments.birth_scratch, cell, scratch::kMiniDriveStatus
        )[0] != 0
    ) {
        transaction_status[0] = 140 + scratch::at<int32_t>(
            arguments.birth_scratch, cell, scratch::kMiniDriveStatus
        )[0];
        return;
    }

    int32_t* chemistry_status = scratch::at<int32_t>(
        arguments.birth_scratch, cell, scratch::kMiniChemistryStatus
    );
    chemistry_status[0] = 0;
    float* mini_chemistry = scratch::at<float>(
        arguments.birth_scratch, cell, scratch::kMiniChemistry
    );
    const double energy_n = chemistry_detail::clamp_double(
        mini_energy[1] / static_cast<double>(arguments.max_energy), 0.0, 1.0
    );
    mini_chemistry[0] = static_cast<float>(0.40 + 0.40 * energy_n);
    mini_chemistry[1] = 0.55f;
    mini_chemistry[2] = 0.0f;
    mini_chemistry[3] = 0.0f;
    mini_chemistry[4] = 0.50f;
    mini_chemistry[5] = 0.25f;

    float* mini_mlp = scratch::at<float>(
        arguments.birth_scratch, cell, layout.mlp_rows
    );
    float* mini_hmem = scratch::at<float>(
        arguments.birth_scratch, cell, layout.hmem_rows
    );
    uint8_t* mini_torch_state = scratch::at<uint8_t>(
        arguments.birth_scratch, cell, layout.torch_state
    );
    for (int64_t column = 0; column < 2 * arguments.birth_config.hmem_width; ++column) {
        mini_hmem[column] = 0.0f;
    }
    for (int64_t index = 0; index < arguments.birth_config.torch_state_bytes; ++index) {
        mini_torch_state[index] = arguments.birth_state.mutation_torch_cpu_state[
            cell * arguments.birth_config.torch_state_bytes + index
        ];
    }
    int64_t* policy_state_semantic = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kMiniPolicyStateSemanticDraws
    );
    policy_state_semantic[0] =
        arguments.birth_state.mutation_torch_cpu_semantic_draws[cell];
    int64_t* mini_policy_cursor = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kMiniPolicyCursor
    );
    mini_policy_cursor[0] = physical_policy_cursor;
    policy::inherit_ordinary_fixed_mlp_child_policy_serial(
        {
            mini_slot_ids,
            mini_alive,
            mini_mlp,
            mini_hmem,
            mini_policy_use_gru,
            mini_torch_state,
            policy_state_semantic,
        },
        {mini_plan_parent, mini_plan_child, mini_plan_count},
        {
            arguments.birth_state.policy_birth_delta_tape
                + cell * arguments.birth_config.policy_tape_capacity
                    * arguments.birth_config.policy_mutation_tape_width,
            arguments.birth_state.policy_birth_mutation_mask_tape
                + cell * arguments.birth_config.policy_tape_capacity
                    * arguments.birth_config.policy_mutation_tape_width,
            arguments.birth_state.policy_birth_post_rng_state_packed_tape == nullptr
                ? nullptr
                : arguments.birth_state.policy_birth_post_rng_state_packed_tape
                    + cell * arguments.birth_config.policy_tape_capacity
                        * torch_rng_pack::kPackedStateBytes,
            arguments.birth_state.policy_birth_semantic_draws_tape
                + cell * arguments.birth_config.policy_tape_capacity,
            mini_policy_cursor,
        },
        {
            scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kMiniPolicyCommitted
            ),
            scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kMiniPolicySemanticDraws
            ),
            scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kMiniPolicyMutationEvents
            ),
            scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kMiniPolicyMutatedElements
            ),
            scratch::at<double>(
                arguments.birth_scratch, cell, scratch::kMiniPolicyMutationDeltaTotal
            ),
            scratch::at<bool>(
                arguments.birth_scratch, cell, scratch::kMiniPolicyMutationOccurred
            ),
            scratch::at<float>(
                arguments.birth_scratch, cell, scratch::kMiniPolicyMutationDelta
            ),
            scratch::at<int32_t>(
                arguments.birth_scratch, cell, scratch::kMiniPolicyStatus
            ),
        },
        {
            1,
            2,
            arguments.packed_width,
            arguments.birth_config.hmem_width,
            arguments.birth_config.torch_state_bytes,
            arguments.birth_config.policy_tape_capacity,
            arguments.birth_config.policy_mutation_tape_width,
        },
        arguments.mlp_pack
            + (cell_capacity_offset + parent_slot) * arguments.packed_width
    );
    if (
        scratch::at<int32_t>(
            arguments.birth_scratch, cell, scratch::kMiniPolicyStatus
        )[0] != 0
    ) {
        transaction_status[0] = 160 + scratch::at<int32_t>(
            arguments.birth_scratch, cell, scratch::kMiniPolicyStatus
        )[0];
        return;
    }

    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPreparedParentSlot
    )[0] = parent_slot;
    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPreparedChildSlot
    )[0] = child_slot;
    scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPreparedChildId
    )[0] = child_id;
    transaction_status[0] = 0;
    prepared_stamp[0] = kFlattenedPreparedStamp;
}

__device__ __forceinline__ void publish_flattened_birth_component_outputs_v1(
    const Phase2KernelArguments arguments,
    const int64_t cell,
    const int64_t cell_tick,
    const int64_t tick_birth_offset
) {
    namespace scratch = flattened_birth_scratch_v1;
#define EIEL_PUBLISH_I64(output, offset)                                        \
    (output)[cell_tick] = scratch::at<int64_t>(                                \
        arguments.birth_scratch, cell, (offset)                                \
    )[0]
#define EIEL_PUBLISH_I32(output, offset)                                        \
    (output)[cell_tick] = scratch::at<int32_t>(                                \
        arguments.birth_scratch, cell, (offset)                                \
    )[0]
    EIEL_PUBLISH_I64(
        arguments.birth_outputs.scalar_committed_count,
        scratch::kMiniScalarCommitted
    );
    EIEL_PUBLISH_I64(
        arguments.birth_outputs.scalar_semantic_draws,
        scratch::kMiniScalarSemanticDraws
    );
    EIEL_PUBLISH_I32(
        arguments.birth_outputs.scalar_status,
        scratch::kMiniScalarStatus
    );
    EIEL_PUBLISH_I64(
        arguments.birth_outputs.drive_committed_count,
        scratch::kMiniDriveCommitted
    );
    EIEL_PUBLISH_I64(
        arguments.birth_outputs.birth_noise_rows_consumed,
        scratch::kMiniBirthNoiseRows
    );
    EIEL_PUBLISH_I64(
        arguments.birth_outputs.exp_prior_rows_consumed,
        scratch::kMiniExpPriorRows
    );
    EIEL_PUBLISH_I64(
        arguments.birth_outputs.standard_normal_rows,
        scratch::kMiniStandardNormalRows
    );
    EIEL_PUBLISH_I64(
        arguments.birth_outputs.drive_semantic_draws,
        scratch::kMiniDriveSemanticDraws
    );
    EIEL_PUBLISH_I32(
        arguments.birth_outputs.drive_status,
        scratch::kMiniDriveStatus
    );
    EIEL_PUBLISH_I32(
        arguments.birth_outputs.chemistry_status,
        scratch::kMiniChemistryStatus
    );
    EIEL_PUBLISH_I64(
        arguments.birth_outputs.policy_committed_count,
        scratch::kMiniPolicyCommitted
    );
    EIEL_PUBLISH_I64(
        arguments.birth_outputs.policy_semantic_draws,
        scratch::kMiniPolicySemanticDraws
    );
    EIEL_PUBLISH_I64(
        arguments.birth_outputs.policy_mutation_events,
        scratch::kMiniPolicyMutationEvents
    );
    EIEL_PUBLISH_I64(
        arguments.birth_outputs.policy_mutated_elements,
        scratch::kMiniPolicyMutatedElements
    );
    arguments.birth_outputs.policy_mutation_delta_total[cell_tick] =
        scratch::at<double>(
            arguments.birth_scratch,
            cell,
            scratch::kMiniPolicyMutationDeltaTotal
        )[0];
    const int64_t birth_capacity = arguments.birth_config.max_births_per_tick;
    const bool* mutation_occurred = scratch::at<bool>(
        arguments.birth_scratch,
        cell,
        birth_capacity > 1
            ? scratch::kB2PolicyMutationOccurred
            : scratch::kMiniPolicyMutationOccurred
    );
    const float* mutation_delta = scratch::at<float>(
        arguments.birth_scratch,
        cell,
        birth_capacity > 1
            ? scratch::kB2PolicyMutationDelta
            : scratch::kMiniPolicyMutationDelta
    );
    for (int64_t birth = 0; birth < birth_capacity; ++birth) {
        arguments.birth_outputs.policy_mutation_occurred[
            tick_birth_offset + birth
        ] = mutation_occurred[birth];
        arguments.birth_outputs.policy_mutation_delta[
            tick_birth_offset + birth
        ] = mutation_delta[birth];
    }
    EIEL_PUBLISH_I32(
        arguments.birth_outputs.policy_status,
        scratch::kMiniPolicyStatus
    );
#undef EIEL_PUBLISH_I32
#undef EIEL_PUBLISH_I64
}

__device__ __forceinline__ void commit_flattened_birth_payload_v1(
    const Phase2KernelArguments arguments,
    const int64_t cell,
    const int64_t child_slot,
    const int64_t absolute_python_raw_draws,
    const int64_t absolute_policy_cursor
) {
    namespace scratch = flattened_birth_scratch_v1;
    const int64_t cell_capacity_offset = cell * arguments.capacity;
    const int64_t child_index = cell_capacity_offset + child_slot;
    const int32_t* mini_i32 = scratch::at<int32_t>(
        arguments.birth_scratch, cell, scratch::kMiniGenomeInt32
    );
    const int64_t* mini_mem_dim = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kMiniPolicyMemDim
    );
    const bool* mini_use_gru = scratch::at<bool>(
        arguments.birth_scratch, cell, scratch::kMiniPolicyUseGru
    );
    const float* mini_f32 = scratch::at<float>(
        arguments.birth_scratch, cell, scratch::kMiniGenomeFloat32
    );

#define EIEL_COMMIT_CHILD_I32(target, index)                                    \
    (target)[child_index] = mini_i32[(index) * 2 + 1]
    EIEL_COMMIT_CHILD_I32(arguments.birth_state.genome_hidden_size, 0);
    EIEL_COMMIT_CHILD_I32(arguments.birth_state.genome_memory_dim, 1);
    EIEL_COMMIT_CHILD_I32(arguments.birth_state.genome_sensor_channel_mask, 2);
    EIEL_COMMIT_CHILD_I32(arguments.birth_state.genome_sensor_radius, 3);
    EIEL_COMMIT_CHILD_I32(
        arguments.birth_state.genome_introspection_channel_mask, 4
    );
    EIEL_COMMIT_CHILD_I32(
        arguments.birth_state.genome_self_regulation_control_mask, 5
    );
    EIEL_COMMIT_CHILD_I32(
        arguments.birth_state.genome_self_modification_target_mask, 6
    );
    EIEL_COMMIT_CHILD_I32(
        arguments.birth_state.genome_directed_inheritance_channel_mask, 7
    );
#undef EIEL_COMMIT_CHILD_I32
    arguments.birth_state.policy_mem_dim[child_index] = mini_mem_dim[1];
    arguments.birth_state.policy_use_gru[child_index] = mini_use_gru[1];

#define EIEL_COMMIT_CHILD_F32(target, index)                                    \
    (target)[child_index] = mini_f32[(index) * 2 + 1]
    EIEL_COMMIT_CHILD_F32(arguments.birth_state.genome_exploration_bias, 0);
    EIEL_COMMIT_CHILD_F32(arguments.birth_state.genome_aggression_bias, 1);
    EIEL_COMMIT_CHILD_F32(arguments.birth_state.genome_mutation_rate, 2);
    EIEL_COMMIT_CHILD_F32(
        arguments.birth_state.genome_reproduction_threshold, 3
    );
    EIEL_COMMIT_CHILD_F32(arguments.birth_state.genome_action_temperature, 4);
    EIEL_COMMIT_CHILD_F32(arguments.birth_state.genome_drive_inertia, 5);
    EIEL_COMMIT_CHILD_F32(arguments.birth_state.genome_drive_noise, 6);
    EIEL_COMMIT_CHILD_F32(arguments.birth_state.genome_drive_logit_gain, 7);
    EIEL_COMMIT_CHILD_F32(arguments.birth_state.genome_pred_lr, 8);
    EIEL_COMMIT_CHILD_F32(arguments.birth_state.genome_drive_sp_hunger, 9);
    EIEL_COMMIT_CHILD_F32(arguments.birth_state.genome_drive_sp_pain, 10);
    EIEL_COMMIT_CHILD_F32(arguments.birth_state.genome_drive_sp_fatigue, 11);
    EIEL_COMMIT_CHILD_F32(arguments.birth_state.genome_drive_sp_fear, 12);
    EIEL_COMMIT_CHILD_F32(arguments.birth_state.genome_exp_prior_strength, 13);
    EIEL_COMMIT_CHILD_F32(arguments.birth_state.genome_exp_prior_noise, 14);
    EIEL_COMMIT_CHILD_F32(arguments.birth_state.genome_drive_birth_noise, 15);
    EIEL_COMMIT_CHILD_F32(arguments.birth_state.repro_threshold, 16);
    EIEL_COMMIT_CHILD_F32(arguments.full_tick_state.gru_upkeep_cost, 17);
    EIEL_COMMIT_CHILD_F32(arguments.attack_bias, 18);
    EIEL_COMMIT_CHILD_F32(arguments.action_temperature, 19);
    EIEL_COMMIT_CHILD_F32(arguments.drive_logit_gain, 20);
    EIEL_COMMIT_CHILD_F32(arguments.full_tick_state.drive_inertia, 21);
    EIEL_COMMIT_CHILD_F32(arguments.full_tick_state.drive_noise, 22);
    EIEL_COMMIT_CHILD_F32(arguments.birth_state.pred_lr, 23);
#undef EIEL_COMMIT_CHILD_F32

    const uint32_t* python_words = scratch::at<uint32_t>(
        arguments.birth_scratch, cell, scratch::kPythonPostWords
    );
    for (int64_t word = 0; word < 624; ++word) {
        arguments.birth_state.python_random_words[cell * 624 + word] =
            python_words[word];
    }
    arguments.birth_state.python_random_index[cell] = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPythonIndex
    )[0];
    arguments.birth_state.python_random_has_gauss[cell] = scratch::at<bool>(
        arguments.birth_scratch, cell, scratch::kPythonHasGauss
    )[0];
    arguments.birth_state.python_random_gauss_next[cell] = scratch::at<double>(
        arguments.birth_scratch, cell, scratch::kPythonGaussNext
    )[0];
    arguments.birth_state.python_random_raw_draws[cell] =
        absolute_python_raw_draws;
    arguments.birth_state.mutation_py_semantic_draws[cell] = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPythonSemanticDraws
    )[0];

    const uint64_t* population_state = scratch::at<uint64_t>(
        arguments.birth_scratch, cell, scratch::kPopulationPcgState
    );
    const int64_t* population_buffer = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPopulationPcgBuffer
    );
    for (int64_t word = 0; word < 4; ++word) {
        arguments.birth_state.population_pcg64_state_words[cell * 4 + word] =
            population_state[word];
    }
    for (int64_t word = 0; word < 2; ++word) {
        arguments.birth_state.population_pcg64_buffer_words[cell * 2 + word] =
            population_buffer[word];
    }
    arguments.birth_state.mutation_np_semantic_draws[cell] = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPopulationSemanticDraws
    )[0];
    const float* mini_drives = scratch::at<float>(
        arguments.birth_scratch, cell, scratch::kMiniDrives
    );
    for (int64_t drive = 0; drive < 4; ++drive) {
        arguments.drives[child_index * arguments.drive_width + drive] =
            mini_drives[drive];
    }
    arguments.birth_state.birth_reset_shock[child_index] = scratch::at<float>(
        arguments.birth_scratch, cell, scratch::kMiniBirthShock
    )[0];
    const float* mini_chemistry = scratch::at<float>(
        arguments.birth_scratch, cell, scratch::kMiniChemistry
    );
    for (int64_t column = 0; column < 6; ++column) {
        arguments.full_tick_state.chemistry[
            child_index * arguments.full_tick_state.chemistry_width + column
        ] = mini_chemistry[column];
    }

    const auto layout = scratch::make_dynamic_layout(
        arguments.packed_width,
        arguments.birth_config.hmem_width,
        arguments.birth_config.torch_state_bytes
    );
    const float* mini_mlp = scratch::at<float>(
        arguments.birth_scratch, cell, layout.mlp_rows
    );
    for (int64_t column = 0; column < arguments.packed_width; ++column) {
        arguments.mlp_pack[child_index * arguments.packed_width + column] =
            mini_mlp[arguments.packed_width + column];
    }
    const float* mini_hmem = scratch::at<float>(
        arguments.birth_scratch, cell, layout.hmem_rows
    );
    for (int64_t column = 0; column < arguments.birth_config.hmem_width; ++column) {
        arguments.birth_state.hmem[
            child_index * arguments.birth_config.hmem_width + column
        ] = mini_hmem[arguments.birth_config.hmem_width + column];
    }
    const uint8_t* mini_torch_state = scratch::at<uint8_t>(
        arguments.birth_scratch, cell, layout.torch_state
    );
    for (int64_t index = 0; index < arguments.birth_config.torch_state_bytes; ++index) {
        arguments.birth_state.mutation_torch_cpu_state[
            cell * arguments.birth_config.torch_state_bytes + index
        ] = mini_torch_state[index];
    }
    arguments.birth_state.policy_birth_tape_cursor[cell] =
        absolute_policy_cursor;
    arguments.birth_state.mutation_torch_cpu_semantic_draws[cell] =
        scratch::at<int64_t>(
            arguments.birth_scratch,
            cell,
            scratch::kMiniPolicyStateSemanticDraws
        )[0];
}

__device__ __forceinline__ void commit_flattened_b2_birth_payload_v2(
    const Phase2KernelArguments arguments,
    const int64_t cell,
    const int64_t birth,
    const int64_t child_slot,
    const int64_t absolute_python_raw_draws,
    const int64_t absolute_policy_cursor
) {
    namespace scratch = flattened_birth_scratch_v1;
    const int64_t cell_capacity_offset = cell * arguments.capacity;
    const int64_t child_index = cell_capacity_offset + child_slot;
    const int32_t* scalar_i32 = scratch::at<int32_t>(
        arguments.birth_scratch, cell, scratch::kB2ScalarGenomeInt32
    );
    const int64_t* scalar_mem_dim = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kB2ScalarPolicyMemDim
    );
    const bool* scalar_use_gru = scratch::at<bool>(
        arguments.birth_scratch, cell, scratch::kB2ScalarPolicyUseGru
    );
    const float* scalar_f32 = scratch::at<float>(
        arguments.birth_scratch, cell, scratch::kB2ScalarGenomeFloat32
    );

#define EIEL_COMMIT_B2_CHILD_I32(target, field)                                  \
    (target)[child_index] = scalar_i32[(field) * 2 + birth]
    EIEL_COMMIT_B2_CHILD_I32(arguments.birth_state.genome_hidden_size, 0);
    EIEL_COMMIT_B2_CHILD_I32(arguments.birth_state.genome_memory_dim, 1);
    EIEL_COMMIT_B2_CHILD_I32(arguments.birth_state.genome_sensor_channel_mask, 2);
    EIEL_COMMIT_B2_CHILD_I32(arguments.birth_state.genome_sensor_radius, 3);
    EIEL_COMMIT_B2_CHILD_I32(
        arguments.birth_state.genome_introspection_channel_mask, 4
    );
    EIEL_COMMIT_B2_CHILD_I32(
        arguments.birth_state.genome_self_regulation_control_mask, 5
    );
    EIEL_COMMIT_B2_CHILD_I32(
        arguments.birth_state.genome_self_modification_target_mask, 6
    );
    EIEL_COMMIT_B2_CHILD_I32(
        arguments.birth_state.genome_directed_inheritance_channel_mask, 7
    );
#undef EIEL_COMMIT_B2_CHILD_I32
    arguments.birth_state.policy_mem_dim[child_index] = scalar_mem_dim[birth];
    arguments.birth_state.policy_use_gru[child_index] = scalar_use_gru[birth];

#define EIEL_COMMIT_B2_CHILD_F32(target, field)                                  \
    (target)[child_index] = scalar_f32[(field) * 2 + birth]
    EIEL_COMMIT_B2_CHILD_F32(arguments.birth_state.genome_exploration_bias, 0);
    EIEL_COMMIT_B2_CHILD_F32(arguments.birth_state.genome_aggression_bias, 1);
    EIEL_COMMIT_B2_CHILD_F32(arguments.birth_state.genome_mutation_rate, 2);
    EIEL_COMMIT_B2_CHILD_F32(
        arguments.birth_state.genome_reproduction_threshold, 3
    );
    EIEL_COMMIT_B2_CHILD_F32(arguments.birth_state.genome_action_temperature, 4);
    EIEL_COMMIT_B2_CHILD_F32(arguments.birth_state.genome_drive_inertia, 5);
    EIEL_COMMIT_B2_CHILD_F32(arguments.birth_state.genome_drive_noise, 6);
    EIEL_COMMIT_B2_CHILD_F32(arguments.birth_state.genome_drive_logit_gain, 7);
    EIEL_COMMIT_B2_CHILD_F32(arguments.birth_state.genome_pred_lr, 8);
    EIEL_COMMIT_B2_CHILD_F32(arguments.birth_state.genome_drive_sp_hunger, 9);
    EIEL_COMMIT_B2_CHILD_F32(arguments.birth_state.genome_drive_sp_pain, 10);
    EIEL_COMMIT_B2_CHILD_F32(arguments.birth_state.genome_drive_sp_fatigue, 11);
    EIEL_COMMIT_B2_CHILD_F32(arguments.birth_state.genome_drive_sp_fear, 12);
    EIEL_COMMIT_B2_CHILD_F32(arguments.birth_state.genome_exp_prior_strength, 13);
    EIEL_COMMIT_B2_CHILD_F32(arguments.birth_state.genome_exp_prior_noise, 14);
    EIEL_COMMIT_B2_CHILD_F32(arguments.birth_state.genome_drive_birth_noise, 15);
    EIEL_COMMIT_B2_CHILD_F32(arguments.birth_state.repro_threshold, 16);
    EIEL_COMMIT_B2_CHILD_F32(arguments.full_tick_state.gru_upkeep_cost, 17);
    EIEL_COMMIT_B2_CHILD_F32(arguments.attack_bias, 18);
    EIEL_COMMIT_B2_CHILD_F32(arguments.action_temperature, 19);
    EIEL_COMMIT_B2_CHILD_F32(arguments.drive_logit_gain, 20);
    EIEL_COMMIT_B2_CHILD_F32(arguments.full_tick_state.drive_inertia, 21);
    EIEL_COMMIT_B2_CHILD_F32(arguments.full_tick_state.drive_noise, 22);
    EIEL_COMMIT_B2_CHILD_F32(arguments.birth_state.pred_lr, 23);
#undef EIEL_COMMIT_B2_CHILD_F32

    const float* staged_drives = scratch::at<float>(
        arguments.birth_scratch, cell, scratch::kMiniDrives
    );
    for (int64_t drive = 0; drive < 4; ++drive) {
        arguments.drives[child_index * arguments.drive_width + drive] =
            staged_drives[birth * 4 + drive];
    }
    arguments.birth_state.birth_reset_shock[child_index] = scratch::at<float>(
        arguments.birth_scratch, cell, scratch::kMiniBirthShock
    )[birth];
    const float* staged_chemistry = scratch::at<float>(
        arguments.birth_scratch, cell, scratch::kMiniChemistry
    );
    for (int64_t column = 0; column < 6; ++column) {
        arguments.full_tick_state.chemistry[
            child_index * arguments.full_tick_state.chemistry_width + column
        ] = staged_chemistry[birth * 6 + column];
    }

    const auto layout = scratch::make_b2_dynamic_layout(
        arguments.packed_width,
        arguments.birth_config.hmem_width,
        arguments.birth_config.torch_state_bytes
    );
    const float* staged_mlp = scratch::at<float>(
        arguments.birth_scratch, cell, layout.child_mlp_rows
    );
    for (int64_t column = 0; column < arguments.packed_width; ++column) {
        arguments.mlp_pack[child_index * arguments.packed_width + column] =
            staged_mlp[birth * arguments.packed_width + column];
    }
    const float* staged_hmem = scratch::at<float>(
        arguments.birth_scratch, cell, layout.child_hmem_rows
    );
    for (int64_t column = 0; column < arguments.birth_config.hmem_width; ++column) {
        arguments.birth_state.hmem[
            child_index * arguments.birth_config.hmem_width + column
        ] = staged_hmem[birth * arguments.birth_config.hmem_width + column];
    }

    const uint32_t* python_words = scratch::at<uint32_t>(
        arguments.birth_scratch, cell, scratch::kPythonPostWords
    );
    for (int64_t word = 0; word < 624; ++word) {
        arguments.birth_state.python_random_words[cell * 624 + word] =
            python_words[word];
    }
    arguments.birth_state.python_random_index[cell] = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPythonIndex
    )[0];
    arguments.birth_state.python_random_has_gauss[cell] = scratch::at<bool>(
        arguments.birth_scratch, cell, scratch::kPythonHasGauss
    )[0];
    arguments.birth_state.python_random_gauss_next[cell] = scratch::at<double>(
        arguments.birth_scratch, cell, scratch::kPythonGaussNext
    )[0];
    arguments.birth_state.python_random_raw_draws[cell] =
        absolute_python_raw_draws;
    arguments.birth_state.mutation_py_semantic_draws[cell] = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPythonSemanticDraws
    )[0];

    const uint64_t* population_state = scratch::at<uint64_t>(
        arguments.birth_scratch, cell, scratch::kPopulationPcgState
    );
    const int64_t* population_buffer = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPopulationPcgBuffer
    );
    for (int64_t word = 0; word < 4; ++word) {
        arguments.birth_state.population_pcg64_state_words[cell * 4 + word] =
            population_state[word];
    }
    for (int64_t word = 0; word < 2; ++word) {
        arguments.birth_state.population_pcg64_buffer_words[cell * 2 + word] =
            population_buffer[word];
    }
    arguments.birth_state.mutation_np_semantic_draws[cell] = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPopulationSemanticDraws
    )[0];

    const uint8_t* staged_torch_state = scratch::at<uint8_t>(
        arguments.birth_scratch, cell, layout.torch_state
    );
    for (int64_t index = 0; index < arguments.birth_config.torch_state_bytes; ++index) {
        arguments.birth_state.mutation_torch_cpu_state[
            cell * arguments.birth_config.torch_state_bytes + index
        ] = staged_torch_state[index];
    }
    arguments.birth_state.policy_birth_tape_cursor[cell] = absolute_policy_cursor;
    arguments.birth_state.mutation_torch_cpu_semantic_draws[cell] =
        scratch::at<int64_t>(
            arguments.birth_scratch,
            cell,
            scratch::kMiniPolicyStateSemanticDraws
        )[0];
}

__device__ __forceinline__ void commit_flattened_birth_plan_rng_v1(
    const Phase2KernelArguments arguments,
    const int64_t cell
) {
    namespace scratch = flattened_birth_scratch_v1;
    const int64_t cell_capacity_offset = cell * arguments.capacity;
    for (int64_t slot = 0; slot < arguments.capacity; ++slot) {
        arguments.birth_state.repro_cooldown[cell_capacity_offset + slot] =
            arguments.birth_outputs.planned_repro_cooldown[
                cell_capacity_offset + slot
            ];
    }
    const uint64_t* planner_state = scratch::at<uint64_t>(
        arguments.birth_scratch, cell, scratch::kPlannerPcgState
    );
    const int64_t* planner_buffer = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPlannerPcgBuffer
    );
    const uint64_t* runner_state = scratch::at<uint64_t>(
        arguments.birth_scratch, cell, scratch::kRunnerPcgState
    );
    const int64_t* runner_buffer = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kRunnerPcgBuffer
    );
    for (int64_t word = 0; word < 4; ++word) {
        arguments.full_tick_state.shared_env_policy_pcg64_state_words[
            cell * 4 + word
        ] = planner_state[word];
        arguments.full_tick_state.runner_pcg64_state_words[cell * 4 + word] =
            runner_state[word];
    }
    for (int64_t word = 0; word < 2; ++word) {
        arguments.full_tick_state.shared_env_policy_pcg64_buffer_words[
            cell * 2 + word
        ] = planner_buffer[word];
        arguments.full_tick_state.runner_pcg64_buffer_words[cell * 2 + word] =
            runner_buffer[word];
    }
    arguments.full_tick_state.policy_np_semantic_draws[cell] =
        scratch::at<int64_t>(
            arguments.birth_scratch, cell, scratch::kPlannerPolicyDraws
        )[0];
    arguments.full_tick_state.metrics_np_semantic_draws[cell] =
        scratch::at<int64_t>(
            arguments.birth_scratch, cell, scratch::kRunnerMetricDraws
        )[0];
}

__device__ __forceinline__ int32_t commit_flattened_birth_transaction_v1(
    const Phase2KernelArguments arguments,
    const int64_t cell,
    const int64_t controller_tick
) {
    namespace scratch = flattened_birth_scratch_v1;
    using namespace persistent_birth_events;
    using namespace persistent_newborn_state;

    const int64_t output_tick = arguments.output_ticks_per_cell == 1
        ? 0
        : controller_tick;
    const int64_t cell_tick =
        cell * arguments.output_ticks_per_cell + output_tick;
    const int64_t cell_capacity_offset = cell * arguments.capacity;
    const int64_t cell_active_offset = cell * arguments.active_capacity;
    const int64_t cell_id_map_offset = cell * arguments.agent_id_capacity;
    const int64_t cell_grid_offset = cell * arguments.height * arguments.width;
    const int64_t birth_capacity = arguments.birth_config.max_births_per_tick;
    const int64_t tick_capacity_offset = cell_tick * arguments.capacity;
    const int64_t tick_birth_offset = cell_tick * birth_capacity;
    const int64_t absolute_tick =
        arguments.mechanics_config.start_tick + controller_tick;
    arguments.birth_outputs.combined_status[cell_tick] = 0;

    if (arguments.full_tick_outputs.residual_release_status[cell_tick] != 0) {
        arguments.birth_outputs.combined_status[cell_tick] = 4;
        return 4;
    }
    if (arguments.birth_outputs.plan_status[cell_tick] != 0) {
        arguments.birth_outputs.combined_status[cell_tick] = 1;
        return 1;
    }

    publish_flattened_birth_component_outputs_v1(
        arguments, cell, cell_tick, tick_birth_offset
    );
    const int64_t planned_count =
        arguments.birth_outputs.planned_count[cell_tick];
    const int32_t transaction_status = scratch::at<int32_t>(
        arguments.birth_scratch, cell, scratch::kTransactionStatus
    )[0];
    const int64_t prepared_stamp = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kPreparedStamp
    )[0];
    if (transaction_status != 0 || prepared_stamp != kFlattenedPreparedStamp) {
        const int32_t status = transaction_status == 25 ? 3 : 2;
        arguments.birth_outputs.combined_status[cell_tick] = status;
        return transaction_status != 0 && transaction_status != 25
            ? transaction_status
            : status;
    }
    const bool component_contract_ok =
        planned_count >= 0 && planned_count <= birth_capacity
        && birth_capacity >= 1 && birth_capacity <= 2
        && arguments.birth_outputs.scalar_committed_count[cell_tick]
            == planned_count
        && arguments.birth_outputs.scalar_status[cell_tick] == 0
        && arguments.birth_outputs.drive_committed_count[cell_tick]
            == planned_count
        && arguments.birth_outputs.drive_status[cell_tick] == 0
        && arguments.birth_outputs.chemistry_status[cell_tick] == 0
        && arguments.birth_outputs.policy_committed_count[cell_tick]
            == planned_count
        && arguments.birth_outputs.policy_status[cell_tick] == 0;
    if (!component_contract_ok) {
        arguments.birth_outputs.combined_status[cell_tick] = 2;
        return 22;
    }

    int64_t absolute_python_raw_draws = 0;
    int64_t absolute_policy_cursor = 0;
    if (
        !absolute_cursor_from_physical_v1(
            scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kPythonRawDraws
            )[0],
            arguments.birth_state.python_random_raw_draw_base[cell],
            arguments.birth_config.gauss_tape_capacity,
            &absolute_python_raw_draws
        )
        || !absolute_cursor_from_physical_v1(
            scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kMiniPolicyCursor
            )[0],
            arguments.birth_state.policy_birth_tape_cursor_base[cell],
            arguments.birth_config.policy_tape_capacity,
            &absolute_policy_cursor
        )
    ) {
        arguments.birth_outputs.combined_status[cell_tick] = 2;
        return 26;
    }

    const int64_t required_event_rows = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kRequiredEventRows
    )[0];
    const int64_t event_cursor = arguments.event_state.active_count[cell];
    if (
        required_event_rows < 0
        || !event_sequence_contract_valid_v1(arguments.event_state, cell)
        || event_cursor < 0 || event_cursor > arguments.event_state.capacity
        || required_event_rows > arguments.event_state.capacity - event_cursor
    ) {
        if (!event_sequence_contract_valid_v1(arguments.event_state, cell)) {
            fail_event_sequence_contract_v1(arguments.event_state, cell);
        }
        arguments.birth_outputs.combined_status[cell_tick] = 3;
        return 3;
    }

    const int64_t free_count =
        arguments.full_tick_state.slot_free_queue_count[cell];
    if (free_count < 0 || free_count > arguments.capacity) {
        arguments.birth_outputs.combined_status[cell_tick] = 2;
        return 23;
    }
    int64_t queue_head = 0;
    int64_t prepared_parent_slot = -1;
    int64_t prepared_child_slot = -1;
    int64_t prepared_child_id = -1;
    const bool b2_transaction = birth_capacity == 2;
    const int64_t* b2_parent_slots = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kB2IdentityParentSlots
    );
    const int64_t* b2_child_slots = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kB2IdentityChildSlots
    );
    const int64_t* b2_child_ids = scratch::at<int64_t>(
        arguments.birth_scratch, cell, scratch::kB2IdentityChildIds
    );
    if (b2_transaction) {
        queue_head = scratch::at<int64_t>(
            arguments.birth_scratch, cell, scratch::kB2QueueHead
        )[0];
        if (
            queue_head < 0 || queue_head > free_count
            || scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kB2IdentityCount
            )[0] != planned_count
        ) {
            arguments.birth_outputs.combined_status[cell_tick] = 2;
            return 24;
        }
        int64_t expected_child_id = arguments.birth_state.population_next_id[cell] < 1
            ? 1
            : arguments.birth_state.population_next_id[cell];
        for (int64_t slot = 0; slot < arguments.capacity; ++slot) {
            const int64_t agent_id =
                arguments.slot_to_agent_id[cell_capacity_offset + slot];
            if (agent_id >= expected_child_id) {
                expected_child_id = agent_id + 1;
            }
        }
        for (int64_t birth = 0; birth < planned_count; ++birth) {
            const int64_t parent_slot = b2_parent_slots[birth];
            const int64_t child_slot = b2_child_slots[birth];
            const int64_t child_id = b2_child_ids[birth];
            const int32_t child_y = arguments.birth_outputs.planned_spawn_positions[
                (tick_birth_offset + birth) * 2
            ];
            const int32_t child_x = arguments.birth_outputs.planned_spawn_positions[
                (tick_birth_offset + birth) * 2 + 1
            ];
            if (
                parent_slot
                    != arguments.birth_outputs.planned_parent_slots[
                        tick_birth_offset + birth
                    ]
                || parent_slot < 0 || parent_slot >= arguments.capacity
                || child_slot < 0 || child_slot >= arguments.capacity
                || child_id != expected_child_id + birth
                || !active_identity::agent_id_fits_grid_v1(child_id)
                || arguments.alive_mask[cell_capacity_offset + child_slot]
                || arguments.slot_to_agent_id[cell_capacity_offset + child_slot] >= 0
                || child_y < 0 || child_y >= arguments.height
                || child_x < 0 || child_x >= arguments.width
                || arguments.agent_grid[
                    cell_grid_offset
                        + static_cast<int64_t>(child_y) * arguments.width + child_x
                ] != -1
                || (birth == 1 && child_slot == b2_child_slots[0])
            ) {
                arguments.birth_outputs.combined_status[cell_tick] = 2;
                return 24;
            }
        }
        int64_t staged_remaining_queue_count = 0;
        for (int64_t queue_index = queue_head; queue_index < free_count; ++queue_index) {
            const int64_t slot = arguments.full_tick_state.slot_free_queue[
                cell_capacity_offset + queue_index
            ];
            bool reserved_child = false;
            for (int64_t birth = 0; birth < planned_count; ++birth) {
                reserved_child = reserved_child || slot == b2_child_slots[birth];
            }
            if (
                slot >= 0 && slot < arguments.capacity
                && !arguments.alive_mask[cell_capacity_offset + slot]
                && arguments.slot_to_agent_id[cell_capacity_offset + slot] < 0
                && !reserved_child
            ) {
                ++staged_remaining_queue_count;
            }
        }
        if (
            staged_remaining_queue_count != scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kB2QueueCount
            )[0]
        ) {
            arguments.birth_outputs.combined_status[cell_tick] = 2;
            return 24;
        }
        if (planned_count > 0) {
            prepared_parent_slot = b2_parent_slots[planned_count - 1];
            prepared_child_slot = b2_child_slots[planned_count - 1];
            prepared_child_id = b2_child_ids[planned_count - 1];
        }
    } else if (planned_count == 1) {
        prepared_parent_slot = scratch::at<int64_t>(
            arguments.birth_scratch, cell, scratch::kPreparedParentSlot
        )[0];
        prepared_child_slot = scratch::at<int64_t>(
            arguments.birth_scratch, cell, scratch::kPreparedChildSlot
        )[0];
        prepared_child_id = scratch::at<int64_t>(
            arguments.birth_scratch, cell, scratch::kPreparedChildId
        )[0];
        const int64_t planned_parent =
            arguments.birth_outputs.planned_parent_slots[tick_birth_offset];
        int64_t first_free_slot = -1;
        while (queue_head < free_count) {
            const int64_t candidate = arguments.full_tick_state.slot_free_queue[
                cell_capacity_offset + queue_head++
            ];
            if (
                candidate >= 0 && candidate < arguments.capacity
                && !arguments.alive_mask[cell_capacity_offset + candidate]
                && arguments.slot_to_agent_id[cell_capacity_offset + candidate] < 0
            ) {
                first_free_slot = candidate;
                break;
            }
        }
        int64_t expected_child_id = arguments.birth_state.population_next_id[cell] < 1
            ? 1
            : arguments.birth_state.population_next_id[cell];
        for (int64_t slot = 0; slot < arguments.capacity; ++slot) {
            const int64_t agent_id =
                arguments.slot_to_agent_id[cell_capacity_offset + slot];
            if (agent_id >= expected_child_id) {
                expected_child_id = agent_id + 1;
            }
        }
        if (
            prepared_parent_slot != planned_parent
            || prepared_parent_slot < 0
            || prepared_parent_slot >= arguments.capacity
            || prepared_child_slot != first_free_slot
            || prepared_child_slot < 0
            || prepared_child_id != expected_child_id
            || !active_identity::agent_id_fits_grid_v1(prepared_child_id)
        ) {
            arguments.birth_outputs.combined_status[cell_tick] = 2;
            return 24;
        }
    }

    const int64_t prior_active_count = arguments.active_count_state[cell];
    bool reuse_sorted_active_ids =
        prior_active_count >= 0 && prior_active_count <= arguments.capacity;
    int64_t prior_alive_count = 0;
    const active_identity::ActiveIdentityPairsV1 active_pairs{
        arguments.active_agent_ids + cell_active_offset,
        arguments.active_agent_slots + cell_active_offset,
        arguments.slot_to_agent_id + cell_capacity_offset,
        arguments.alive_mask + cell_capacity_offset,
        prior_active_count,
        arguments.active_capacity,
        arguments.capacity,
    };
    if (
        !reuse_sorted_active_ids
        || !active_identity::valid_active_identity_pairs_v1(active_pairs)
    ) {
        arguments.birth_outputs.combined_status[cell_tick] = 2;
        return 24;
    }
    const int64_t previous_active_id = prior_active_count > 0
        ? arguments.active_agent_ids[
            cell_active_offset + prior_active_count - 1
        ]
        : -1;
    for (int64_t slot = 0; slot < arguments.capacity; ++slot) {
        const int64_t slot_index = cell_capacity_offset + slot;
        prior_alive_count += arguments.alive_mask[slot_index]
            && arguments.slot_to_agent_id[slot_index] >= 0
            ? 1
            : 0;
    }
    reuse_sorted_active_ids = reuse_sorted_active_ids
        && prior_alive_count == prior_active_count
        && (
            planned_count == 0
            || prepared_child_id > previous_active_id
        );

    arguments.event_state.status[cell] = 0;
    commit_flattened_birth_plan_rng_v1(arguments, cell);

    for (int64_t birth = 0; birth < birth_capacity; ++birth) {
        arguments.birth_outputs.committed_parent_agent_ids[
            tick_birth_offset + birth
        ] = -1;
        arguments.birth_outputs.committed_mate_agent_ids[
            tick_birth_offset + birth
        ] = -1;
        arguments.birth_outputs.committed_child_agent_ids[
            tick_birth_offset + birth
        ] = -1;
        arguments.birth_outputs.committed_child_slots[
            tick_birth_offset + birth
        ] = -1;
        arguments.birth_outputs.committed_mate_agent_ids[
            tick_birth_offset + birth
        ] = -1;
    }
    arguments.birth_outputs.identity_committed_count[cell_tick] = 0;
    arguments.birth_outputs.identity_status[cell_tick] = 0;

    for (int64_t birth = 0; birth < planned_count; ++birth) {
        const int64_t birth_parent_slot = b2_transaction
            ? b2_parent_slots[birth]
            : prepared_parent_slot;
        const int64_t birth_child_slot = b2_transaction
            ? b2_child_slots[birth]
            : prepared_child_slot;
        const int64_t birth_child_id = b2_transaction
            ? b2_child_ids[birth]
            : prepared_child_id;
        const int64_t parent_index =
            cell_capacity_offset + birth_parent_slot;
        const int64_t child_index =
            cell_capacity_offset + birth_child_slot;
        const int64_t parent_id = b2_transaction
            ? scratch::at<int64_t>(
                arguments.birth_scratch, cell, scratch::kB2ParentAgentIds
            )[birth]
            : arguments.slot_to_agent_id[parent_index];
        const int32_t child_y = arguments.birth_outputs.planned_spawn_positions[
            (tick_birth_offset + birth) * 2
        ];
        const int32_t child_x = arguments.birth_outputs.planned_spawn_positions[
            (tick_birth_offset + birth) * 2 + 1
        ];
        const double parent_energy = ::isfinite(
            arguments.policy_energy_observation[parent_index]
        )
            ? arguments.policy_energy_observation[parent_index]
            : static_cast<double>(arguments.energy[parent_index]);
        const double child_energy = b2_transaction
            ? scratch::at<double>(
                arguments.birth_scratch, cell, scratch::kB2IdentityChildEnergy
            )[birth]
            : parent_energy * birth_parent_energy_share_for_cell_v1(arguments.birth_config, cell);
        const double parent_energy_after = parent_energy - child_energy;

        arguments.birth_outputs.committed_parent_agent_ids[
            tick_birth_offset + birth
        ] = parent_id;
        arguments.birth_outputs.committed_child_agent_ids[
            tick_birth_offset + birth
        ] = birth_child_id;
        arguments.birth_outputs.committed_child_slots[
            tick_birth_offset + birth
        ] = birth_child_slot;
        arguments.energy[parent_index] = static_cast<float>(parent_energy_after);
        arguments.policy_energy_observation[parent_index] = parent_energy_after;
        arguments.birth_state.repro_cooldown[parent_index] =
            birth_cooldown_steps_for_cell_v1(arguments.birth_config, cell);
        arguments.birth_state.offspring_count[parent_index] += 1;

        if (birth_child_id < arguments.agent_id_capacity) {
            arguments.agent_id_to_slot[
                cell_id_map_offset + birth_child_id
            ] = static_cast<int32_t>(birth_child_slot);
        }
        arguments.slot_to_agent_id[child_index] = birth_child_id;
        arguments.slot_tokens[child_index] = arguments.slot_tokens[child_index] < 0
            ? 1
            : arguments.slot_tokens[child_index] + 1;
        arguments.alive_mask[child_index] = true;
        arguments.pos_y[child_index] = child_y;
        arguments.pos_x[child_index] = child_x;
        arguments.agent_grid[
            cell_grid_offset + static_cast<int64_t>(child_y) * arguments.width
                + child_x
        ] = static_cast<int32_t>(birth_child_id);
        arguments.energy[child_index] = static_cast<float>(child_energy);
        arguments.policy_energy_observation[child_index] = child_energy;
        arguments.health[child_index] = arguments.start_health;
        arguments.injury[child_index] = 0.0f;
        arguments.birth_state.scar[child_index] = 0.0f;
        arguments.full_tick_state.age[child_index] = 0;
        arguments.survival_steps[child_index] = 0;
        arguments.rest_streak[child_index] = 0;
        arguments.birth_state.repro_cooldown[child_index] = 0;
        arguments.birth_state.repro_threshold[child_index] =
            arguments.birth_state.repro_threshold[parent_index];
        arguments.birth_state.offspring_count[child_index] = 0;
        arguments.event_state.lineage_founder_id[child_index] = b2_transaction
            ? scratch::at<int32_t>(
                arguments.birth_scratch,
                cell,
                scratch::kB2ChildLineageFounder
            )[birth]
            : (
                arguments.event_state.lineage_founder_id[parent_index] >= 0
                ? arguments.event_state.lineage_founder_id[parent_index]
                : static_cast<int32_t>(parent_id)
            );
        arguments.event_state.lineage_depth[child_index] = b2_transaction
            ? scratch::at<int32_t>(
                arguments.birth_scratch,
                cell,
                scratch::kB2ChildLineageDepth
            )[birth]
            : arguments.event_state.lineage_depth[parent_index] + 1;
        arguments.birth_state.lineage_birth_generation[child_index] =
            static_cast<int32_t>(arguments.mechanics_config.generation);
        arguments.birth_state.lineage_birth_tick[child_index] =
            static_cast<int32_t>(absolute_tick);
        const int32_t parent_patch_lineage =
            arguments.birth_state.patch_habitat_lineage_id[parent_index] != 0
            ? arguments.birth_state.patch_habitat_lineage_id[parent_index]
            : static_cast<int32_t>(parent_id);
        arguments.birth_state.patch_habitat_lineage_id[child_index] =
            parent_patch_lineage;
        arguments.action_cache_token[child_index] = -1;
        arguments.action_cache_cursor[child_index] = -1;
        arguments.action_cache_next[child_index] = -1;
        arguments.birth_state.population_next_id[cell] = birth_child_id + 1;
        arguments.birth_outputs.identity_committed_count[cell_tick] = birth + 1;

        arguments.last_action[child_index] = -1;
        arguments.last_last_action[child_index] = -1;
        arguments.action_switches[child_index] = 0;
        arguments.action_oscillations[child_index] = 0;
        arguments.episode0_switches[child_index] = 0;
        arguments.episode0_oscillations[child_index] = 0;
        if (b2_transaction) {
            commit_flattened_b2_birth_payload_v2(
                arguments,
                cell,
                birth,
                birth_child_slot,
                absolute_python_raw_draws,
                absolute_policy_cursor
            );
        } else {
            commit_flattened_birth_payload_v1(
                arguments,
                cell,
                birth_child_slot,
                absolute_python_raw_draws,
                absolute_policy_cursor
            );
        }
    }

    int64_t remaining_queue_count = 0;
    for (int64_t queue_index = queue_head; queue_index < free_count; ++queue_index) {
        const int64_t slot = arguments.full_tick_state.slot_free_queue[
            cell_capacity_offset + queue_index
        ];
        if (
            slot >= 0 && slot < arguments.capacity
            && !arguments.alive_mask[cell_capacity_offset + slot]
            && arguments.slot_to_agent_id[cell_capacity_offset + slot] < 0
        ) {
            arguments.full_tick_state.slot_free_queue[
                cell_capacity_offset + remaining_queue_count++
            ] = slot;
        }
    }
    for (
        int64_t queue_index = remaining_queue_count;
        queue_index < free_count;
        ++queue_index
    ) {
        arguments.full_tick_state.slot_free_queue[
            cell_capacity_offset + queue_index
        ] = -1;
    }
    arguments.full_tick_state.slot_free_queue_count[cell] = remaining_queue_count;

    int64_t remaining_alive = 0;
    if (reuse_sorted_active_ids) {
        remaining_alive = prior_active_count;
        for (int64_t row = 0; row < prior_active_count; ++row) {
            arguments.survivor_agent_ids[tick_capacity_offset + row] =
                arguments.active_agent_ids[cell_active_offset + row];
        }
        for (int64_t birth = 0; birth < planned_count; ++birth) {
            const int64_t child_id = b2_transaction
                ? b2_child_ids[birth]
                : prepared_child_id;
            const int32_t child_slot = static_cast<int32_t>(
                b2_transaction
                    ? b2_child_slots[birth]
                    : prepared_child_slot
            );
            arguments.active_agent_ids[cell_active_offset + remaining_alive] =
                child_id;
            arguments.active_agent_slots[
                cell_active_offset + remaining_alive
            ] = child_slot;
            arguments.survivor_agent_ids[tick_capacity_offset + remaining_alive] =
                child_id;
            ++remaining_alive;
        }
        for (int64_t row = remaining_alive; row < arguments.capacity; ++row) {
            arguments.active_agent_ids[cell_active_offset + row] = -1;
            arguments.active_agent_slots[cell_active_offset + row] = -1;
            arguments.survivor_agent_ids[tick_capacity_offset + row] = -1;
        }
    } else {
        for (int64_t row = 0; row < arguments.capacity; ++row) {
            arguments.active_agent_ids[cell_active_offset + row] = -1;
            arguments.active_agent_slots[cell_active_offset + row] = -1;
            arguments.survivor_agent_ids[tick_capacity_offset + row] = -1;
        }
        int64_t previous_agent_id = INT64_MIN;
        int64_t previous_slot = -1;
        while (true) {
            int64_t selected_slot = -1;
            int64_t selected_agent_id = INT64_MAX;
            for (int64_t slot = 0; slot < arguments.capacity; ++slot) {
                const int64_t slot_index = cell_capacity_offset + slot;
                if (!arguments.alive_mask[slot_index]) {
                    continue;
                }
                const int64_t agent_id = arguments.slot_to_agent_id[slot_index];
                const bool follows_previous = agent_id > previous_agent_id
                    || (agent_id == previous_agent_id && slot > previous_slot);
                if (
                    agent_id >= 0 && follows_previous
                    && (
                        agent_id < selected_agent_id
                        || (agent_id == selected_agent_id && slot < selected_slot)
                    )
                ) {
                    selected_agent_id = agent_id;
                    selected_slot = slot;
                }
            }
            if (selected_slot < 0) {
                break;
            }
            arguments.active_agent_ids[cell_active_offset + remaining_alive] =
                selected_agent_id;
            arguments.active_agent_slots[
                cell_active_offset + remaining_alive
            ] = static_cast<int32_t>(selected_slot);
            arguments.survivor_agent_ids[tick_capacity_offset + remaining_alive] =
                selected_agent_id;
            ++remaining_alive;
            previous_agent_id = selected_agent_id;
            previous_slot = selected_slot;
        }
    }
    arguments.active_count_state[cell] = remaining_alive;
    arguments.survivor_count[cell_tick] = remaining_alive;

    const persistent_lifecycle_events::FlatEventTapeColumns columns{
        {arguments.event_state.event_type, arguments.event_state.capacity, 1},
        {arguments.event_state.event_generation, arguments.event_state.capacity, 1},
        {arguments.event_state.event_tick, arguments.event_state.capacity, 1},
        {arguments.event_state.event_actor_id, arguments.event_state.capacity, 1},
        {arguments.event_state.event_target_id, arguments.event_state.capacity, 1},
        {arguments.event_state.event_value, arguments.event_state.capacity, 1},
        {arguments.event_state.typed_event_type, arguments.event_state.capacity, 1},
        {arguments.event_state.typed_generation, arguments.event_state.capacity, 1},
        {arguments.event_state.typed_tick, arguments.event_state.capacity, 1},
        {arguments.event_state.typed_sequence, arguments.event_state.capacity, 1},
        {arguments.event_state.typed_actor_id, arguments.event_state.capacity, 1},
        {arguments.event_state.typed_target_id, arguments.event_state.capacity, 1},
        {arguments.event_state.typed_actor_slot, arguments.event_state.capacity, 1},
        {arguments.event_state.typed_target_slot, arguments.event_state.capacity, 1},
        {arguments.event_state.typed_parent_id, arguments.event_state.capacity, 1},
        {arguments.event_state.typed_child_id, arguments.event_state.capacity, 1},
        {arguments.event_state.typed_founder_id, arguments.event_state.capacity, 1},
        {arguments.event_state.typed_lineage_depth, arguments.event_state.capacity, 1},
        {arguments.event_state.typed_subtype, arguments.event_state.capacity, 1},
        {arguments.event_state.typed_reason, arguments.event_state.capacity, 1},
        {arguments.event_state.typed_success, arguments.event_state.capacity, 1},
        {arguments.event_state.typed_value, arguments.event_state.capacity, 1},
    };
    const persistent_lifecycle_events::FlatEventTapeState tape{
        {arguments.event_state.active_count, 1},
        {arguments.event_state.status, 1},
        arguments.event_state.capacity,
    };
    append_preflighted_ordinary_birth_events_serial(
        columns,
        tape,
        {
            {arguments.slot_to_agent_id, arguments.capacity, 1},
            {arguments.event_state.lineage_founder_id, arguments.capacity, 1},
            {arguments.event_state.lineage_depth, arguments.capacity, 1},
            arguments.capacity,
        },
        {
            {arguments.birth_outputs.decision_parent_slots
                 + output_tick * arguments.capacity,
             arguments.output_ticks_per_cell * arguments.capacity, 1},
            {arguments.birth_outputs.decision_reason_codes
                 + output_tick * arguments.capacity,
             arguments.output_ticks_per_cell * arguments.capacity, 1},
            {arguments.birth_outputs.decision_count + output_tick,
             arguments.output_ticks_per_cell},
            arguments.capacity,
        },
        {
            {arguments.birth_outputs.committed_parent_agent_ids
                 + output_tick * birth_capacity,
             arguments.output_ticks_per_cell * birth_capacity, 1},
            {arguments.birth_outputs.committed_child_agent_ids
                 + output_tick * birth_capacity,
             arguments.output_ticks_per_cell * birth_capacity, 1},
            {arguments.birth_outputs.planned_parent_slots
                 + output_tick * birth_capacity,
             arguments.output_ticks_per_cell * birth_capacity, 1},
            {arguments.birth_outputs.committed_child_slots
                 + output_tick * birth_capacity,
             arguments.output_ticks_per_cell * birth_capacity, 1},
            {arguments.birth_outputs.policy_mutation_occurred
                 + output_tick * birth_capacity,
             arguments.output_ticks_per_cell * birth_capacity, 1},
            {arguments.birth_outputs.policy_mutation_delta
                 + output_tick * birth_capacity,
             arguments.output_ticks_per_cell * birth_capacity, 1},
            {arguments.birth_outputs.planned_child_energy
                 + output_tick * birth_capacity,
             arguments.output_ticks_per_cell * birth_capacity, 1},
            {arguments.birth_outputs.identity_committed_count + output_tick,
             arguments.output_ticks_per_cell},
            birth_capacity,
        },
        cell,
        arguments.mechanics_config.generation,
        absolute_tick
    );
    const int64_t event_end = arguments.event_state.active_count[cell];
    const int64_t event_cell_offset = cell * arguments.event_state.capacity;
    const int64_t event_sequence_base = arguments.event_state.sequence_base[cell];
    for (int64_t row = event_cursor; row < event_end; ++row) {
        arguments.event_state.typed_sequence[event_cell_offset + row] =
            event_sequence_base + row;
    }
    arguments.event_state.status[cell] = persistent_birth_events::kStatusOk;
    if (planned_count > 0) {
        const FlatFreshNewbornSelectionState newborn_state{
            {arguments.energy_collected + cell_capacity_offset, 1},
            {arguments.energy_collected_exact + cell_capacity_offset, 1},
            {arguments.birth_state.attacks + cell_capacity_offset, 1},
            {arguments.birth_state.early_survival_steps + cell_capacity_offset, 1},
            {arguments.birth_state.early_energy_collected + cell_capacity_offset, 1},
            {arguments.birth_state.early_energy_collected_exact + cell_capacity_offset, 1},
            {arguments.birth_state.early_injury_gained + cell_capacity_offset, 1},
            {arguments.birth_state.early_life_weight + cell_capacity_offset, 1},
            {arguments.birth_state.pred_err_sum + cell_capacity_offset, 1},
            {arguments.birth_state.pred_err_count + cell_capacity_offset, 1},
            {arguments.birth_state.first_repro_step + cell_capacity_offset, 1},
            {arguments.birth_state.dev_stage_initialized + cell_capacity_offset, 1},
            {arguments.birth_state.episode0_pred_err_sum + cell_capacity_offset, 1},
            {arguments.birth_state.episode0_pred_err_count + cell_capacity_offset, 1},
            {arguments.birth_state.late_pred_err_sum + cell_capacity_offset, 1},
            {arguments.birth_state.late_pred_err_count + cell_capacity_offset, 1},
            {arguments.birth_state.chemistry_observation_fidelity + cell_capacity_offset, 1},
            {arguments.birth_state.chemistry_observation_bias + cell_capacity_offset, 1},
            {arguments.action_cache_actions
                 + cell * arguments.capacity * arguments.cache_capacity,
             arguments.cache_capacity, 1},
            {arguments.full_tick_state.prev_drive_delta
                 + cell_capacity_offset * arguments.drive_width,
             arguments.drive_width, 1},
            {arguments.full_tick_state.drive_volatility + cell_capacity_offset, 1},
            {arguments.full_tick_state.drive_jitter + cell_capacity_offset, 1},
            {arguments.full_tick_state.episode0_drive_jitter_sum + cell_capacity_offset, 1},
            {arguments.full_tick_state.last_energy_delta + cell_capacity_offset, 1},
            {arguments.full_tick_state.last_health_delta + cell_capacity_offset, 1},
            {arguments.full_tick_state.last_injury_delta + cell_capacity_offset, 1},
            {arguments.full_tick_state.selfharm_events + cell_capacity_offset, 1},
            {arguments.full_tick_state.episode0_selfharm_events + cell_capacity_offset, 1},
        };
        for (int64_t birth = 0; birth < planned_count; ++birth) {
            const int64_t parent_slot = b2_transaction
                ? b2_parent_slots[birth]
                : prepared_parent_slot;
            const int64_t child_slot = b2_transaction
                ? b2_child_slots[birth]
                : prepared_child_slot;
            reset_ordinary_newborn_selection_state(
                newborn_state,
                child_slot,
                {arguments.cache_capacity, arguments.drive_width}
            );
            if (
                arguments.birth_state.first_repro_step[
                    cell_capacity_offset + parent_slot
                ] < 0
            ) {
                arguments.birth_state.first_repro_step[
                    cell_capacity_offset + parent_slot
                ] = arguments.survival_steps[
                    cell_capacity_offset + parent_slot
                ];
            }
        }
    }
    scratch::at<int32_t>(
        arguments.birth_scratch, cell, scratch::kTransactionStatus
    )[0] = 0;
    arguments.birth_outputs.combined_status[cell_tick] = 0;
    return 0;
}

__device__ __forceinline__ void
publish_flattened_runtime_birth_component_outputs_v2(
    const Phase2KernelArguments arguments,
    const int64_t cell,
    const int64_t cell_tick,
    const int64_t tick_birth_offset,
    const flattened_birth_scratch_v2::FlatBirthScratchLayoutV2 layout,
    const flattened_birth_scratch_v2::FlatBirthScratchArenaV2 arena
) {
    namespace scratch = flattened_birth_scratch_v2;
    const scratch::BirthScalarStatusesV2* statuses =
        arena.at<scratch::BirthScalarStatusesV2>(
            cell, layout, scratch::kScalarStatuses
        );
    arguments.birth_outputs.scalar_committed_count[cell_tick] =
        statuses[0].scalar_committed_count;
    arguments.birth_outputs.scalar_semantic_draws[cell_tick] =
        statuses[0].scalar_semantic_draws;
    arguments.birth_outputs.scalar_status[cell_tick] = statuses[0].scalar_status;
    arguments.birth_outputs.drive_committed_count[cell_tick] =
        statuses[0].drive_committed_count;
    arguments.birth_outputs.birth_noise_rows_consumed[cell_tick] =
        statuses[0].birth_noise_rows;
    arguments.birth_outputs.exp_prior_rows_consumed[cell_tick] =
        statuses[0].exp_prior_rows;
    arguments.birth_outputs.standard_normal_rows[cell_tick] =
        statuses[0].standard_normal_rows;
    arguments.birth_outputs.drive_semantic_draws[cell_tick] =
        statuses[0].drive_semantic_draws;
    arguments.birth_outputs.drive_status[cell_tick] = statuses[0].drive_status;
    arguments.birth_outputs.chemistry_status[cell_tick] =
        statuses[0].chemistry_status;
    arguments.birth_outputs.policy_committed_count[cell_tick] =
        statuses[0].policy_committed_count;
    arguments.birth_outputs.policy_semantic_draws[cell_tick] =
        statuses[0].policy_semantic_draws;
    arguments.birth_outputs.policy_mutation_events[cell_tick] =
        statuses[0].policy_mutation_events;
    arguments.birth_outputs.policy_mutated_elements[cell_tick] =
        statuses[0].policy_mutated_elements;
    arguments.birth_outputs.policy_mutation_delta_total[cell_tick] =
        statuses[0].policy_mutation_delta_total;
    arguments.birth_outputs.policy_status[cell_tick] = statuses[0].policy_status;
    const bool* mutation_flags = arena.at<bool>(
        cell, layout, scratch::kMutationFlags
    );
    const float* mutation_deltas = arena.at<float>(
        cell, layout, scratch::kMutationDeltas
    );
    for (
        int64_t birth = 0;
        birth < arguments.birth_config.max_births_per_tick;
        ++birth
    ) {
        arguments.birth_outputs.policy_mutation_occurred[
            tick_birth_offset + birth
        ] = mutation_flags[birth];
        arguments.birth_outputs.policy_mutation_delta[
            tick_birth_offset + birth
        ] = mutation_deltas[birth];
    }
}

__device__ __forceinline__ void commit_flattened_runtime_birth_payload_v2(
    const Phase2KernelArguments arguments,
    const int64_t cell,
    const int64_t birth,
    const int64_t child_slot,
    const flattened_birth_scratch_v2::FlatBirthScratchLayoutV2 layout,
    const flattened_birth_scratch_v2::FlatBirthScratchArenaV2 arena
) {
    namespace scratch = flattened_birth_scratch_v2;
    const int64_t birth_capacity = arguments.birth_config.max_births_per_tick;
    const int64_t child_index = cell * arguments.capacity + child_slot;
    const int32_t* scalar_i32 = arena.at<int32_t>(
        cell, layout, scratch::kGenomeInt32Planes
    );
    const int64_t* scalar_mem_dim = arena.at<int64_t>(
        cell, layout, scratch::kPolicyMemoryDimensionRows
    );
    const bool* scalar_use_gru = arena.at<bool>(
        cell, layout, scratch::kPolicyUseGruRows
    );
    const float* scalar_f32 = arena.at<float>(
        cell, layout, scratch::kGenomeFloatPlanes
    );

#define EIEL_COMMIT_RUNTIME_CHILD_I32(target, field)                             \
    (target)[child_index] = scalar_i32[(field) * birth_capacity + birth]
    EIEL_COMMIT_RUNTIME_CHILD_I32(arguments.birth_state.genome_hidden_size, 0);
    EIEL_COMMIT_RUNTIME_CHILD_I32(arguments.birth_state.genome_memory_dim, 1);
    EIEL_COMMIT_RUNTIME_CHILD_I32(arguments.birth_state.genome_sensor_channel_mask, 2);
    EIEL_COMMIT_RUNTIME_CHILD_I32(arguments.birth_state.genome_sensor_radius, 3);
    EIEL_COMMIT_RUNTIME_CHILD_I32(arguments.birth_state.genome_introspection_channel_mask, 4);
    EIEL_COMMIT_RUNTIME_CHILD_I32(arguments.birth_state.genome_self_regulation_control_mask, 5);
    EIEL_COMMIT_RUNTIME_CHILD_I32(arguments.birth_state.genome_self_modification_target_mask, 6);
    EIEL_COMMIT_RUNTIME_CHILD_I32(arguments.birth_state.genome_directed_inheritance_channel_mask, 7);
#undef EIEL_COMMIT_RUNTIME_CHILD_I32
    arguments.birth_state.policy_mem_dim[child_index] = scalar_mem_dim[birth];
    arguments.birth_state.policy_use_gru[child_index] = scalar_use_gru[birth];

#define EIEL_COMMIT_RUNTIME_CHILD_F32(target, field)                             \
    (target)[child_index] = scalar_f32[(field) * birth_capacity + birth]
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.birth_state.genome_exploration_bias, 0);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.birth_state.genome_aggression_bias, 1);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.birth_state.genome_mutation_rate, 2);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.birth_state.genome_reproduction_threshold, 3);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.birth_state.genome_action_temperature, 4);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.birth_state.genome_drive_inertia, 5);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.birth_state.genome_drive_noise, 6);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.birth_state.genome_drive_logit_gain, 7);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.birth_state.genome_pred_lr, 8);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.birth_state.genome_drive_sp_hunger, 9);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.birth_state.genome_drive_sp_pain, 10);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.birth_state.genome_drive_sp_fatigue, 11);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.birth_state.genome_drive_sp_fear, 12);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.birth_state.genome_exp_prior_strength, 13);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.birth_state.genome_exp_prior_noise, 14);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.birth_state.genome_drive_birth_noise, 15);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.birth_state.repro_threshold, 16);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.full_tick_state.gru_upkeep_cost, 17);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.attack_bias, 18);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.action_temperature, 19);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.drive_logit_gain, 20);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.full_tick_state.drive_inertia, 21);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.full_tick_state.drive_noise, 22);
    EIEL_COMMIT_RUNTIME_CHILD_F32(arguments.birth_state.pred_lr, 23);
#undef EIEL_COMMIT_RUNTIME_CHILD_F32

    if (
        arguments.birth_state.inheritance_mode != nullptr
        && arguments.birth_state.inheritance_mode[cell] == 2
    ) {
        namespace diploid = cuda_diploid_scalar_genome_v1;
        const int64_t cell_capacity_offset = cell * arguments.capacity;
        const int64_t allele_cell_stride =
            diploid::kAllelePlaneCount * arguments.capacity;
        float* const allele_cell_base =
            arguments.birth_state.diploid_allele_storage
            + cell * allele_cell_stride;
        const diploid::AllelePlanesV1 allele_planes{
            allele_cell_base,
            allele_cell_base
                + diploid::kLocusCount * arguments.capacity,
            arguments.capacity,
            arguments.capacity,
        };
        const diploid::PhenotypeArraysV1 phenotype_arrays{{
            arguments.birth_state.genome_exploration_bias
                + cell_capacity_offset,
            arguments.birth_state.genome_aggression_bias
                + cell_capacity_offset,
            arguments.birth_state.genome_action_temperature
                + cell_capacity_offset,
            arguments.birth_state.genome_drive_inertia
                + cell_capacity_offset,
            arguments.birth_state.genome_drive_noise
                + cell_capacity_offset,
            arguments.birth_state.genome_drive_logit_gain
                + cell_capacity_offset,
            arguments.birth_state.genome_pred_lr
                + cell_capacity_offset,
            arguments.birth_state.genome_drive_sp_hunger
                + cell_capacity_offset,
            arguments.birth_state.genome_drive_sp_pain
                + cell_capacity_offset,
            arguments.birth_state.genome_drive_sp_fatigue
                + cell_capacity_offset,
            arguments.birth_state.genome_drive_sp_fear
                + cell_capacity_offset,
            arguments.birth_state.genome_exp_prior_strength
                + cell_capacity_offset,
            arguments.birth_state.genome_exp_prior_noise
                + cell_capacity_offset,
            arguments.birth_state.genome_drive_birth_noise
                + cell_capacity_offset,
            arguments.birth_state.genome_reproduction_threshold
                + cell_capacity_offset,
            arguments.birth_state.genome_mutation_rate
                + cell_capacity_offset,
            arguments.birth_state.genome_resource_handling_axis
                + cell_capacity_offset,
        }};
        const diploid::StagedChildV1* staged_children =
            arena.at<diploid::StagedChildV1>(
                cell, layout, scratch::kDiploidStagedChildren
            );
        (void)diploid::commit_staged_child_v1(
            allele_planes,
            phenotype_arrays,
            child_slot,
            staged_children[birth]
        );
    }
    if (
        arguments.birth_state.genome_upgrade_v2_enabled != nullptr
        && arguments.birth_state.genome_upgrade_v2_enabled[cell]
    ) {
        namespace genome_v2 = eiel::genome_upgrade_v2_q10_v1;
        const int64_t allele_cell_stride =
            2 * genome_v2::kProductCount * genome_v2::kParameterCount
            * arguments.capacity;
        const int64_t modifier_cell_stride =
            2 * genome_v2::kProductCount * arguments.capacity;
        const int64_t product_cell_stride =
            genome_v2::kProductCount * arguments.capacity;
        const int64_t phenotype_cell_stride =
            genome_v2::kPhenotypeCount * arguments.capacity;
        const genome_v2::ResidentPlanesV1 planes{
            arguments.birth_state.genome_upgrade_v2_alleles_q10
                + cell * allele_cell_stride,
            arguments.birth_state.genome_upgrade_v2_modifiers_q10
                + cell * modifier_cell_stride,
            arguments.birth_state.genome_upgrade_v2_products_q10
                + cell * product_cell_stride,
            arguments.birth_state.genome_upgrade_v2_phenotypes_q10
                + cell * phenotype_cell_stride,
            arguments.birth_state.genome_upgrade_v2_expression_cost_q10
                + cell * arguments.capacity,
            arguments.birth_state.genome_upgrade_v2_active_product_count
                + cell * arguments.capacity,
            arguments.birth_state.genome_upgrade_v2_saturated_product_count
                + cell * arguments.capacity,
            arguments.capacity,
        };
        const genome_v2::StagedChildV1* staged_children =
            arena.at<genome_v2::StagedChildV1>(
                cell, layout, scratch::kGenomeUpgradeV2StagedChildren
            );
        const int64_t neutral_context[genome_v2::kContextCount] = {};
        (void)genome_v2::commit_staged_child_v1(
            planes,
            child_slot,
            neutral_context,
            staged_children[birth]
        );
    }

    const float* staged_drives = arena.at<float>(
        cell, layout, scratch::kDriveRows
    );
    for (int64_t drive = 0; drive < scratch::kDriveWidth; ++drive) {
        arguments.drives[child_index * arguments.drive_width + drive] =
            staged_drives[birth * scratch::kDriveWidth + drive];
    }
    arguments.birth_state.birth_reset_shock[child_index] = arena.at<float>(
        cell, layout, scratch::kBirthResetShockRows
    )[birth];
    const float* staged_chemistry = arena.at<float>(
        cell, layout, scratch::kChemistryRows
    );
    for (int64_t column = 0; column < scratch::kChemistryWidth; ++column) {
        arguments.full_tick_state.chemistry[
            child_index * arguments.full_tick_state.chemistry_width + column
        ] = staged_chemistry[birth * scratch::kChemistryWidth + column];
    }
    const float* staged_mlp = arena.at<float>(
        cell, layout, scratch::kPolicyMlpRows
    );
    for (int64_t column = 0; column < arguments.packed_width; ++column) {
        arguments.mlp_pack[child_index * arguments.packed_width + column] =
            staged_mlp[birth * arguments.packed_width + column];
    }
    const float* staged_hmem = arena.at<float>(
        cell, layout, scratch::kHmemRows
    );
    for (int64_t column = 0; column < arguments.birth_config.hmem_width; ++column) {
        arguments.birth_state.hmem[
            child_index * arguments.birth_config.hmem_width + column
        ] = staged_hmem[birth * arguments.birth_config.hmem_width + column];
    }
}

__device__ __forceinline__ void commit_flattened_runtime_birth_rng_v2(
    const Phase2KernelArguments arguments,
    const int64_t cell,
    const int64_t absolute_python_raw_draws,
    const int64_t absolute_policy_cursor,
    const flattened_birth_scratch_v2::FlatBirthScratchLayoutV2 layout,
    const flattened_birth_scratch_v2::FlatBirthScratchArenaV2 arena
) {
    namespace scratch = flattened_birth_scratch_v2;
    namespace legacy_scratch = flattened_birth_scratch_v1;
    const int64_t cell_capacity_offset = cell * arguments.capacity;
    for (int64_t slot = 0; slot < arguments.capacity; ++slot) {
        arguments.birth_state.repro_cooldown[cell_capacity_offset + slot] =
            arguments.birth_outputs.planned_repro_cooldown[
                cell_capacity_offset + slot
            ];
    }
    const scratch::Pcg64SnapshotV2* planner_rng =
        arena.at<scratch::Pcg64SnapshotV2>(
            cell, layout, scratch::kPlannerPcg64Snapshot
        );
    const scratch::Pcg64SnapshotV2* runner_rng =
        arena.at<scratch::Pcg64SnapshotV2>(
            cell, layout, scratch::kRunnerPcg64Snapshot
        );
    for (int64_t word = 0; word < scratch::kPcg64StateWordCount; ++word) {
        arguments.full_tick_state.shared_env_policy_pcg64_state_words[
            cell * 4 + word
        ] = planner_rng[0].state_words[word];
        arguments.full_tick_state.runner_pcg64_state_words[cell * 4 + word] =
            runner_rng[0].state_words[word];
    }
    for (int64_t word = 0; word < scratch::kPcg64BufferWordCount; ++word) {
        arguments.full_tick_state.shared_env_policy_pcg64_buffer_words[
            cell * 2 + word
        ] = planner_rng[0].buffer_words[word];
        arguments.full_tick_state.runner_pcg64_buffer_words[cell * 2 + word] =
            runner_rng[0].buffer_words[word];
    }
    arguments.full_tick_state.policy_np_semantic_draws[cell] =
        legacy_scratch::at<int64_t>(
            arguments.birth_scratch,
            cell,
            legacy_scratch::kPlannerPolicyDraws
        )[0];
    arguments.full_tick_state.metrics_np_semantic_draws[cell] =
        legacy_scratch::at<int64_t>(
            arguments.birth_scratch,
            cell,
            legacy_scratch::kRunnerMetricDraws
        )[0];

    const uint32_t* python_words = arena.at<uint32_t>(
        cell, layout, scratch::kPythonRandomPostWords
    );
    const scratch::PythonRandomSnapshotV2* python_snapshots =
        arena.at<scratch::PythonRandomSnapshotV2>(
            cell, layout, scratch::kPythonRandomSnapshots
        );
    for (int64_t word = 0; word < scratch::kPythonRandomWordCount; ++word) {
        arguments.birth_state.python_random_words[
            cell * scratch::kPythonRandomWordCount + word
        ] = python_words[word];
    }
    arguments.birth_state.python_random_index[cell] = python_snapshots[1].index;
    arguments.birth_state.python_random_has_gauss[cell] =
        python_snapshots[1].has_gauss;
    arguments.birth_state.python_random_gauss_next[cell] =
        python_snapshots[1].gauss_next;
    arguments.birth_state.python_random_raw_draws[cell] =
        absolute_python_raw_draws;
    arguments.birth_state.mutation_py_semantic_draws[cell] =
        python_snapshots[1].semantic_draws;

    const scratch::Pcg64SnapshotV2* population_rng =
        arena.at<scratch::Pcg64SnapshotV2>(
            cell, layout, scratch::kPopulationPcg64Snapshot
        );
    for (int64_t word = 0; word < scratch::kPcg64StateWordCount; ++word) {
        arguments.birth_state.population_pcg64_state_words[cell * 4 + word] =
            population_rng[0].state_words[word];
    }
    for (int64_t word = 0; word < scratch::kPcg64BufferWordCount; ++word) {
        arguments.birth_state.population_pcg64_buffer_words[cell * 2 + word] =
            population_rng[0].buffer_words[word];
    }
    const scratch::BirthScalarStatusesV2* statuses =
        arena.at<scratch::BirthScalarStatusesV2>(
            cell, layout, scratch::kScalarStatuses
        );
    arguments.birth_state.mutation_np_semantic_draws[cell] =
        statuses[0].population_np_semantic_draws;
    const uint8_t* torch_state = arena.at<uint8_t>(
        cell, layout, scratch::kTorchState
    );
    for (int64_t index = 0; index < arguments.birth_config.torch_state_bytes; ++index) {
        arguments.birth_state.mutation_torch_cpu_state[
            cell * arguments.birth_config.torch_state_bytes + index
        ] = torch_state[index];
    }
    arguments.birth_state.policy_birth_tape_cursor[cell] = absolute_policy_cursor;
    arguments.birth_state.mutation_torch_cpu_semantic_draws[cell] =
        statuses[0].policy_state_semantic_draws;
}

__device__ __forceinline__ int32_t
commit_flattened_runtime_birth_transaction_v2(
    const Phase2KernelArguments arguments,
    const int64_t cell,
    const int64_t controller_tick
) {
    namespace scratch = flattened_birth_scratch_v2;
    using namespace persistent_newborn_state;

    const int64_t birth_capacity = arguments.birth_config.max_births_per_tick;
    const int64_t output_tick = arguments.output_ticks_per_cell == 1
        ? 0
        : controller_tick;
    const int64_t cell_tick =
        cell * arguments.output_ticks_per_cell + output_tick;
    const int64_t cell_capacity_offset = cell * arguments.capacity;
    const int64_t cell_active_offset = cell * arguments.active_capacity;
    const int64_t cell_id_map_offset = cell * arguments.agent_id_capacity;
    const int64_t cell_grid_offset = cell * arguments.height * arguments.width;
    const int64_t tick_capacity_offset = cell_tick * arguments.capacity;
    const int64_t tick_birth_offset = cell_tick * birth_capacity;
    const int64_t absolute_tick =
        arguments.mechanics_config.start_tick + controller_tick;
    arguments.birth_outputs.combined_status[cell_tick] = 0;
    if (arguments.full_tick_outputs.residual_release_status[cell_tick] != 0) {
        arguments.birth_outputs.combined_status[cell_tick] = 4;
        return 4;
    }
    if (arguments.birth_outputs.plan_status[cell_tick] != 0) {
        arguments.birth_outputs.combined_status[cell_tick] = 1;
        return 1;
    }

    scratch::FlatBirthScratchLayoutV2 layout{};
    scratch::FlatBirthScratchArenaV2 arena{};
    if (!make_runtime_birth_scratch_access_v2(arguments, &layout, &arena)) {
        arguments.birth_outputs.combined_status[cell_tick] = 2;
        return 108;
    }
    const scratch::BirthScalarStatusesV2* statuses =
        arena.at<scratch::BirthScalarStatusesV2>(
            cell, layout, scratch::kScalarStatuses
        );
    publish_flattened_runtime_birth_component_outputs_v2(
        arguments, cell, cell_tick, tick_birth_offset, layout, arena
    );
    const int64_t planned_count = arguments.birth_outputs.planned_count[cell_tick];
    if (statuses[0].transaction_status != 0) {
        const int32_t status = statuses[0].transaction_status == 25 ? 3 : 2;
        arguments.birth_outputs.combined_status[cell_tick] = status;
        return statuses[0].transaction_status == 25
            ? status
            : statuses[0].transaction_status;
    }
    const bool component_contract_ok =
        birth_capacity >= 1 && birth_capacity <= arguments.capacity
        && planned_count >= 0 && planned_count <= birth_capacity
        && statuses[0].planned_count == planned_count
        && statuses[0].identity_committed_count == planned_count
        && statuses[0].identity_status == 0
        && arguments.birth_outputs.scalar_committed_count[cell_tick]
            == planned_count
        && arguments.birth_outputs.scalar_status[cell_tick] == 0
        && arguments.birth_outputs.drive_committed_count[cell_tick]
            == planned_count
        && arguments.birth_outputs.drive_status[cell_tick] == 0
        && arguments.birth_outputs.chemistry_status[cell_tick] == 0
        && arguments.birth_outputs.policy_committed_count[cell_tick]
            == planned_count
        && arguments.birth_outputs.policy_status[cell_tick] == 0
        && statuses[0].event_status == 0;
    if (!component_contract_ok) {
        arguments.birth_outputs.combined_status[cell_tick] = 2;
        return 22;
    }

    const scratch::PythonRandomSnapshotV2* python_snapshots =
        arena.at<scratch::PythonRandomSnapshotV2>(
            cell, layout, scratch::kPythonRandomSnapshots
        );
    int64_t absolute_python_raw_draws = 0;
    int64_t absolute_policy_cursor = 0;
    if (
        !absolute_cursor_from_physical_v1(
            python_snapshots[1].raw_draws,
            arguments.birth_state.python_random_raw_draw_base[cell],
            arguments.birth_config.gauss_tape_capacity,
            &absolute_python_raw_draws
        )
        || !absolute_cursor_from_physical_v1(
            statuses[0].policy_birth_tape_cursor,
            arguments.birth_state.policy_birth_tape_cursor_base[cell],
            arguments.birth_config.policy_tape_capacity,
            &absolute_policy_cursor
        )
    ) {
        arguments.birth_outputs.combined_status[cell_tick] = 2;
        return 26;
    }

    const int64_t required_event_rows = statuses[0].required_event_rows;
    const int64_t event_cursor = arguments.event_state.active_count[cell];
    const bool genome_v2_causal_enabled =
        arguments.birth_outputs.genome_v2_causal_capture_enabled != nullptr
        && arguments.birth_outputs
            .genome_v2_causal_capture_enabled[cell];
    const int64_t causal_record_start = genome_v2_causal_enabled
        && arguments.birth_outputs.genome_v2_causal_record_count != nullptr
        ? arguments.birth_outputs.genome_v2_causal_record_count[cell]
        : 0;
    const bool genome_v2_causal_storage_valid =
        genome_v2_causal_enabled
        && arguments.birth_outputs.genome_v2_causal_record_rows != nullptr
        && arguments.birth_outputs.genome_v2_causal_record_valid != nullptr
        && arguments.birth_outputs.genome_v2_causal_record_count != nullptr
        && arguments.birth_outputs.genome_v2_causal_record_dropped_count
            != nullptr
        && causal_record_start >= 0
        && causal_record_start
            <= arguments.birth_outputs.genome_v2_causal_record_capacity;
    const int64_t causal_record_available =
        genome_v2_causal_storage_valid
        ? arguments.birth_outputs.genome_v2_causal_record_capacity
            - causal_record_start
        : 0;
    const int64_t causal_capture_count =
        planned_count < causal_record_available
        ? planned_count
        : causal_record_available;
    if (
        required_event_rows < 0
        || required_event_rows
            > layout.segment(scratch::kEventStagingRows).bytes
                / static_cast<int64_t>(sizeof(scratch::BirthEventStagingRowV2))
        || !event_sequence_contract_valid_v1(arguments.event_state, cell)
        || event_cursor < 0 || event_cursor > arguments.event_state.capacity
        || required_event_rows > arguments.event_state.capacity - event_cursor
    ) {
        if (!event_sequence_contract_valid_v1(arguments.event_state, cell)) {
            fail_event_sequence_contract_v1(arguments.event_state, cell);
        }
        arguments.birth_outputs.combined_status[cell_tick] = 3;
        return 3;
    }
    const int64_t free_count =
        arguments.full_tick_state.slot_free_queue_count[cell];
    const int64_t queue_head = arena.at<int64_t>(
        cell, layout, scratch::kQueueHead
    )[0];
    const int64_t staged_queue_count = arena.at<int64_t>(
        cell, layout, scratch::kQueueCount
    )[0];
    const int64_t* parent_slots = arena.at<int64_t>(
        cell, layout, scratch::kIdentityParentSlots
    );
    const int64_t* child_slots = arena.at<int64_t>(
        cell, layout, scratch::kIdentityChildSlots
    );
    const int64_t* child_ids = arena.at<int64_t>(
        cell, layout, scratch::kIdentityChildIds
    );
    const double* child_energy = arena.at<double>(
        cell, layout, scratch::kIdentityChildEnergy
    );
    const int64_t* parent_agent_ids = arena.at<int64_t>(
        cell, layout, scratch::kParentAgentIds
    );
    const eiel::genome_upgrade_v2_q10_v1::StagedChildV1*
        genome_v2_staged_children =
            arguments.birth_state.genome_upgrade_v2_enabled != nullptr
            && arguments.birth_state.genome_upgrade_v2_enabled[cell]
        ? arena.at<eiel::genome_upgrade_v2_q10_v1::StagedChildV1>(
              cell, layout, scratch::kGenomeUpgradeV2StagedChildren
          )
        : nullptr;
    const int32_t* child_founders = arena.at<int32_t>(
        cell, layout, scratch::kChildLineageFounderIds
    );
    const int32_t* child_depths = arena.at<int32_t>(
        cell, layout, scratch::kChildLineageDepths
    );
    if (
        free_count < 0 || free_count > arguments.capacity
        || queue_head < 0 || queue_head > free_count
        || staged_queue_count < 0 || staged_queue_count > free_count
        || arena.at<int64_t>(cell, layout, scratch::kIdentityCount)[0]
            != planned_count
    ) {
        arguments.birth_outputs.combined_status[cell_tick] = 2;
        return 24;
    }
    int64_t expected_child_id = arguments.birth_state.population_next_id[cell] < 1
        ? 1
        : arguments.birth_state.population_next_id[cell];
    for (int64_t slot = 0; slot < arguments.capacity; ++slot) {
        const int64_t agent_id =
            arguments.slot_to_agent_id[cell_capacity_offset + slot];
        if (agent_id >= expected_child_id) {
            expected_child_id = agent_id + 1;
        }
    }
    for (int64_t birth = 0; birth < planned_count; ++birth) {
        const int64_t parent_slot = parent_slots[birth];
        const int64_t child_slot = child_slots[birth];
        const int64_t child_id = child_ids[birth];
        const int32_t child_y = arguments.birth_outputs.planned_spawn_positions[
            (tick_birth_offset + birth) * 2
        ];
        const int32_t child_x = arguments.birth_outputs.planned_spawn_positions[
            (tick_birth_offset + birth) * 2 + 1
        ];
        if (
            parent_slot != arguments.birth_outputs.planned_parent_slots[
                tick_birth_offset + birth
            ]
            || parent_slot < 0 || parent_slot >= arguments.capacity
            || child_slot < 0 || child_slot >= arguments.capacity
            || parent_agent_ids[birth]
                != arguments.slot_to_agent_id[
                    cell_capacity_offset + parent_slot
                ]
            || child_id != expected_child_id + birth
            || !active_identity::agent_id_fits_grid_v1(child_id)
            || arguments.alive_mask[cell_capacity_offset + child_slot]
            || arguments.slot_to_agent_id[cell_capacity_offset + child_slot] >= 0
            || child_y < 0 || child_y >= arguments.height
            || child_x < 0 || child_x >= arguments.width
            || arguments.agent_grid[
                cell_grid_offset
                    + static_cast<int64_t>(child_y) * arguments.width + child_x
            ] != -1
        ) {
            arguments.birth_outputs.combined_status[cell_tick] = 2;
            return 24;
        }
        for (int64_t prior = 0; prior < birth; ++prior) {
            if (
                child_slots[prior] == child_slot
                || child_ids[prior] == child_id
                || (
                    arguments.birth_outputs.planned_spawn_positions[
                        (tick_birth_offset + prior) * 2
                    ] == child_y
                    && arguments.birth_outputs.planned_spawn_positions[
                        (tick_birth_offset + prior) * 2 + 1
                    ] == child_x
                )
            ) {
                arguments.birth_outputs.combined_status[cell_tick] = 2;
                return 24;
            }
        }
    }
    int64_t remaining_queue_count = 0;
    for (int64_t queue_index = queue_head; queue_index < free_count; ++queue_index) {
        const int64_t slot = arguments.full_tick_state.slot_free_queue[
            cell_capacity_offset + queue_index
        ];
        bool reserved_child = false;
        for (int64_t birth = 0; birth < planned_count; ++birth) {
            reserved_child = reserved_child || slot == child_slots[birth];
        }
        if (
            slot >= 0 && slot < arguments.capacity
            && !arguments.alive_mask[cell_capacity_offset + slot]
            && arguments.slot_to_agent_id[cell_capacity_offset + slot] < 0
            && !reserved_child
        ) {
            ++remaining_queue_count;
        }
    }
    if (remaining_queue_count != staged_queue_count) {
        arguments.birth_outputs.combined_status[cell_tick] = 2;
        return 24;
    }

    const int64_t prior_active_count = arguments.active_count_state[cell];
    if (
        prior_active_count < 0 || prior_active_count > arguments.active_capacity
        || planned_count > arguments.active_capacity - prior_active_count
    ) {
        arguments.birth_outputs.combined_status[cell_tick] = 2;
        return 24;
    }
    bool reuse_sorted_active_ids = prior_active_count <= arguments.capacity;
    int64_t prior_alive_count = 0;
    const active_identity::ActiveIdentityPairsV1 active_pairs{
        arguments.active_agent_ids + cell_active_offset,
        arguments.active_agent_slots + cell_active_offset,
        arguments.slot_to_agent_id + cell_capacity_offset,
        arguments.alive_mask + cell_capacity_offset,
        prior_active_count,
        arguments.active_capacity,
        arguments.capacity,
    };
    if (
        !reuse_sorted_active_ids
        || !active_identity::valid_active_identity_pairs_v1(active_pairs)
    ) {
        arguments.birth_outputs.combined_status[cell_tick] = 2;
        return 24;
    }
    const int64_t previous_active_id = prior_active_count > 0
        ? arguments.active_agent_ids[
            cell_active_offset + prior_active_count - 1
        ]
        : -1;
    for (int64_t slot = 0; slot < arguments.capacity; ++slot) {
        const int64_t slot_index = cell_capacity_offset + slot;
        prior_alive_count += arguments.alive_mask[slot_index]
            && arguments.slot_to_agent_id[slot_index] >= 0
            ? 1
            : 0;
    }
    if (
        prior_alive_count > arguments.capacity - planned_count
        || prior_alive_count + planned_count > arguments.active_capacity
    ) {
        arguments.birth_outputs.combined_status[cell_tick] = 2;
        return 24;
    }
    reuse_sorted_active_ids = reuse_sorted_active_ids
        && prior_alive_count == prior_active_count
        && (planned_count == 0 || child_ids[planned_count - 1] > previous_active_id);

    const scratch::BirthEventStagingRowV2* staged_events =
        arena.at<scratch::BirthEventStagingRowV2>(
            cell, layout, scratch::kEventStagingRows
        );
    for (int64_t row = 0; row < required_event_rows; ++row) {
        if (
            staged_events[row].generation != arguments.mechanics_config.generation
            || staged_events[row].tick != absolute_tick
            || staged_events[row].sequence
                != arguments.event_state.sequence_base[cell] + event_cursor + row
        ) {
            arguments.birth_outputs.combined_status[cell_tick] = 2;
            return 205;
        }
    }

    const int32_t qualification_fault_stage = arena.at<int32_t>(
        cell,
        layout,
        scratch::kGenomeUpgradeV2QualificationFaultStage
    )[0];
    if (
        qualification_fault_stage
        == scratch::kGenomeUpgradeV2FaultImmediatelyBeforeCommit
    ) {
        arguments.birth_outputs.combined_status[cell_tick] = 2;
        return 249;
    }

    arguments.event_state.status[cell] = 0;
    commit_flattened_runtime_birth_rng_v2(
        arguments,
        cell,
        absolute_python_raw_draws,
        absolute_policy_cursor,
        layout,
        arena
    );
    for (int64_t birth = 0; birth < birth_capacity; ++birth) {
        arguments.birth_outputs.committed_parent_agent_ids[
            tick_birth_offset + birth
        ] = -1;
        arguments.birth_outputs.committed_mate_agent_ids[
            tick_birth_offset + birth
        ] = -1;
        arguments.birth_outputs.committed_child_agent_ids[
            tick_birth_offset + birth
        ] = -1;
        arguments.birth_outputs.committed_child_slots[
            tick_birth_offset + birth
        ] = -1;
        arguments.birth_outputs.committed_mate_agent_ids[
            tick_birth_offset + birth
        ] = -1;
    }
    arguments.birth_outputs.identity_committed_count[cell_tick] = 0;
    arguments.birth_outputs.identity_status[cell_tick] = 0;

    for (int64_t birth = 0; birth < planned_count; ++birth) {
        const int64_t parent_slot = parent_slots[birth];
        const int64_t mate_slot = arguments.birth_config.requires_mate
            ? arguments.birth_outputs.planned_mate_slots[
                  tick_birth_offset + birth
              ]
            : parent_slot;
        const int64_t child_slot = child_slots[birth];
        const int64_t child_id = child_ids[birth];
        const int64_t parent_index = cell_capacity_offset + parent_slot;
        const int64_t mate_index = cell_capacity_offset + mate_slot;
        const int64_t child_index = cell_capacity_offset + child_slot;
        const int64_t parent_id = parent_agent_ids[birth];
        const int64_t mate_id = arguments.birth_config.requires_mate
            ? arguments.slot_to_agent_id[mate_index]
            : -1;
        const int32_t child_y = arguments.birth_outputs.planned_spawn_positions[
            (tick_birth_offset + birth) * 2
        ];
        const int32_t child_x = arguments.birth_outputs.planned_spawn_positions[
            (tick_birth_offset + birth) * 2 + 1
        ];
        const double parent_energy = ::isfinite(
            arguments.policy_energy_observation[parent_index]
        )
            ? arguments.policy_energy_observation[parent_index]
            : static_cast<double>(arguments.energy[parent_index]);
        const double parent_share = arguments.birth_config.requires_mate
            ? parent_energy * birth_parent_energy_share_for_cell_v1(arguments.birth_config, cell)
            : child_energy[birth];
        const double mate_energy = arguments.birth_config.requires_mate
            ? effective_birth_energy_v1(
                  arguments, cell_capacity_offset, mate_slot
              )
            : 0.0;
        const double mate_share = arguments.birth_config.requires_mate
            ? mate_energy * birth_parent_energy_share_for_cell_v1(arguments.birth_config, cell)
            : 0.0;
        double committed_parent_share = parent_share;
        double committed_mate_share = mate_share;
        if (genome_v2_staged_children != nullptr) {
            namespace genome_v2 = eiel::genome_upgrade_v2_q10_v1;
            genome_v2::ExpressionV1 child_expression{};
            const int64_t neutral_context[genome_v2::kContextCount] = {};
            genome_v2::express_v1(
                genome_v2_staged_children[birth].genotype,
                neutral_context,
                &child_expression
            );
            const genome_v2::RuntimeProjectionV1 child_projection =
                genome_v2::project_runtime_traits_v1(child_expression);
            if (
                !genome_v2::scale_offspring_transfer_v1(
                    parent_share,
                    child_projection.offspring_energy_factor_q10,
                    &committed_parent_share
                )
                || (
                    arguments.birth_config.requires_mate
                    && !genome_v2::scale_offspring_transfer_v1(
                        mate_share,
                        child_projection.offspring_energy_factor_q10,
                        &committed_mate_share
                    )
                )
                || committed_parent_share + committed_mate_share
                    != child_energy[birth]
            ) {
                arguments.birth_outputs.combined_status[cell_tick] = 2;
                return 237;
            }
        }
        const double parent_energy_after =
            parent_energy - committed_parent_share;
        const double mate_energy_after = mate_energy - committed_mate_share;

        arguments.birth_outputs.committed_parent_agent_ids[
            tick_birth_offset + birth
        ] = parent_id;
        arguments.birth_outputs.committed_mate_agent_ids[
            tick_birth_offset + birth
        ] = mate_id;
        arguments.birth_outputs.committed_child_agent_ids[
            tick_birth_offset + birth
        ] = child_id;
        arguments.birth_outputs.committed_child_slots[
            tick_birth_offset + birth
        ] = child_slot;
        arguments.energy[parent_index] = static_cast<float>(parent_energy_after);
        arguments.policy_energy_observation[parent_index] = parent_energy_after;
        arguments.birth_state.repro_cooldown[parent_index] =
            birth_cooldown_steps_for_cell_v1(arguments.birth_config, cell);
        arguments.birth_state.offspring_count[parent_index] += 1;
        if (arguments.birth_config.requires_mate) {
            arguments.energy[mate_index] =
                static_cast<float>(mate_energy_after);
            arguments.policy_energy_observation[mate_index] =
                mate_energy_after;
            arguments.birth_state.repro_cooldown[mate_index] =
                birth_cooldown_steps_for_cell_v1(arguments.birth_config, cell);
            arguments.birth_state.offspring_count[mate_index] += 1;
        }

        if (child_id < arguments.agent_id_capacity) {
            arguments.agent_id_to_slot[cell_id_map_offset + child_id] =
                static_cast<int32_t>(child_slot);
        }
        arguments.slot_to_agent_id[child_index] = child_id;
        arguments.slot_tokens[child_index] = arguments.slot_tokens[child_index] < 0
            ? 1
            : arguments.slot_tokens[child_index] + 1;
        arguments.alive_mask[child_index] = true;
        arguments.pos_y[child_index] = child_y;
        arguments.pos_x[child_index] = child_x;
        arguments.agent_grid[
            cell_grid_offset + static_cast<int64_t>(child_y) * arguments.width
                + child_x
        ] = static_cast<int32_t>(child_id);
        arguments.energy[child_index] = static_cast<float>(child_energy[birth]);
        arguments.policy_energy_observation[child_index] = child_energy[birth];
        arguments.health[child_index] = arguments.start_health;
        arguments.injury[child_index] = 0.0f;
        arguments.birth_state.scar[child_index] = 0.0f;
        arguments.full_tick_state.age[child_index] = 0;
        arguments.survival_steps[child_index] = 0;
        arguments.rest_streak[child_index] = 0;
        arguments.birth_state.repro_cooldown[child_index] = 0;
        arguments.birth_state.repro_threshold[child_index] =
            arguments.birth_state.repro_threshold[parent_index];
        arguments.birth_state.offspring_count[child_index] = 0;
        arguments.event_state.lineage_founder_id[child_index] =
            child_founders[birth];
        arguments.event_state.lineage_depth[child_index] = child_depths[birth];
        arguments.birth_state.lineage_birth_generation[child_index] =
            static_cast<int32_t>(arguments.mechanics_config.generation);
        arguments.birth_state.lineage_birth_tick[child_index] =
            static_cast<int32_t>(absolute_tick);
        const int32_t parent_patch_lineage =
            arguments.birth_state.patch_habitat_lineage_id[parent_index] != 0
            ? arguments.birth_state.patch_habitat_lineage_id[parent_index]
            : static_cast<int32_t>(parent_id);
        arguments.birth_state.patch_habitat_lineage_id[child_index] =
            parent_patch_lineage;
        arguments.action_cache_token[child_index] = -1;
        arguments.action_cache_cursor[child_index] = -1;
        arguments.action_cache_next[child_index] = -1;
        arguments.birth_state.population_next_id[cell] = child_id + 1;
        arguments.birth_outputs.identity_committed_count[cell_tick] = birth + 1;
        arguments.last_action[child_index] = -1;
        arguments.last_last_action[child_index] = -1;
        arguments.action_switches[child_index] = 0;
        arguments.action_oscillations[child_index] = 0;
        arguments.episode0_switches[child_index] = 0;
        arguments.episode0_oscillations[child_index] = 0;
        commit_flattened_runtime_birth_payload_v2(
            arguments, cell, birth, child_slot, layout, arena
        );
    }

    remaining_queue_count = 0;
    for (int64_t queue_index = queue_head; queue_index < free_count; ++queue_index) {
        const int64_t slot = arguments.full_tick_state.slot_free_queue[
            cell_capacity_offset + queue_index
        ];
        if (
            slot >= 0 && slot < arguments.capacity
            && !arguments.alive_mask[cell_capacity_offset + slot]
            && arguments.slot_to_agent_id[cell_capacity_offset + slot] < 0
        ) {
            arguments.full_tick_state.slot_free_queue[
                cell_capacity_offset + remaining_queue_count++
            ] = slot;
        }
    }
    for (
        int64_t queue_index = remaining_queue_count;
        queue_index < free_count;
        ++queue_index
    ) {
        arguments.full_tick_state.slot_free_queue[
            cell_capacity_offset + queue_index
        ] = -1;
    }
    arguments.full_tick_state.slot_free_queue_count[cell] = remaining_queue_count;

    int64_t remaining_alive = 0;
    if (reuse_sorted_active_ids) {
        remaining_alive = prior_active_count;
        for (int64_t row = 0; row < prior_active_count; ++row) {
            arguments.survivor_agent_ids[tick_capacity_offset + row] =
                arguments.active_agent_ids[cell_active_offset + row];
        }
        for (int64_t birth = 0; birth < planned_count; ++birth) {
            arguments.active_agent_ids[cell_active_offset + remaining_alive] =
                child_ids[birth];
            arguments.active_agent_slots[
                cell_active_offset + remaining_alive
            ] = static_cast<int32_t>(child_slots[birth]);
            arguments.survivor_agent_ids[tick_capacity_offset + remaining_alive] =
                child_ids[birth];
            ++remaining_alive;
        }
        for (int64_t row = remaining_alive; row < arguments.capacity; ++row) {
            arguments.active_agent_ids[cell_active_offset + row] = -1;
            arguments.active_agent_slots[cell_active_offset + row] = -1;
            arguments.survivor_agent_ids[tick_capacity_offset + row] = -1;
        }
    } else {
        for (int64_t row = 0; row < arguments.capacity; ++row) {
            arguments.active_agent_ids[cell_active_offset + row] = -1;
            arguments.active_agent_slots[cell_active_offset + row] = -1;
            arguments.survivor_agent_ids[tick_capacity_offset + row] = -1;
        }
        int64_t previous_agent_id = INT64_MIN;
        int64_t previous_slot = -1;
        while (true) {
            int64_t selected_slot = -1;
            int64_t selected_agent_id = INT64_MAX;
            for (int64_t slot = 0; slot < arguments.capacity; ++slot) {
                const int64_t slot_index = cell_capacity_offset + slot;
                if (!arguments.alive_mask[slot_index]) {
                    continue;
                }
                const int64_t agent_id = arguments.slot_to_agent_id[slot_index];
                const bool follows_previous = agent_id > previous_agent_id
                    || (agent_id == previous_agent_id && slot > previous_slot);
                if (
                    agent_id >= 0 && follows_previous
                    && (
                        agent_id < selected_agent_id
                        || (agent_id == selected_agent_id && slot < selected_slot)
                    )
                ) {
                    selected_agent_id = agent_id;
                    selected_slot = slot;
                }
            }
            if (selected_slot < 0) {
                break;
            }
            arguments.active_agent_ids[cell_active_offset + remaining_alive] =
                selected_agent_id;
            arguments.active_agent_slots[
                cell_active_offset + remaining_alive
            ] = static_cast<int32_t>(selected_slot);
            arguments.survivor_agent_ids[tick_capacity_offset + remaining_alive] =
                selected_agent_id;
            ++remaining_alive;
            previous_agent_id = selected_agent_id;
            previous_slot = selected_slot;
        }
    }
    arguments.active_count_state[cell] = remaining_alive;
    arguments.survivor_count[cell_tick] = remaining_alive;

    const int64_t event_cell_offset = cell * arguments.event_state.capacity;
    for (int64_t row = 0; row < required_event_rows; ++row) {
        const int64_t event_index = event_cell_offset + event_cursor + row;
        const scratch::BirthEventStagingRowV2 staged = staged_events[row];
        const int64_t founder =
            staged.event_type == persistent_birth_events::kFailedBirthEventType
                && staged.lineage_founder_id < 0
            ? staged.actor_id
            : static_cast<int64_t>(staged.lineage_founder_id);
        arguments.event_state.event_type[event_index] = staged.event_type;
        arguments.event_state.event_generation[event_index] = staged.generation;
        arguments.event_state.event_tick[event_index] = staged.tick;
        arguments.event_state.event_actor_id[event_index] = staged.actor_id;
        arguments.event_state.event_target_id[event_index] = staged.target_id;
        arguments.event_state.event_value[event_index] = staged.value;
        arguments.event_state.typed_event_type[event_index] = staged.event_type;
        arguments.event_state.typed_generation[event_index] = staged.generation;
        arguments.event_state.typed_tick[event_index] = staged.tick;
        arguments.event_state.typed_sequence[event_index] = staged.sequence;
        arguments.event_state.typed_actor_id[event_index] = staged.actor_id;
        arguments.event_state.typed_target_id[event_index] = staged.target_id;
        arguments.event_state.typed_actor_slot[event_index] = staged.actor_slot;
        arguments.event_state.typed_target_slot[event_index] = staged.target_slot;
        arguments.event_state.typed_parent_id[event_index] = staged.parent_id;
        arguments.event_state.typed_child_id[event_index] = staged.child_id;
        arguments.event_state.typed_founder_id[event_index] = founder;
        arguments.event_state.typed_lineage_depth[event_index] =
            staged.lineage_depth;
        arguments.event_state.typed_subtype[event_index] = staged.subtype;
        arguments.event_state.typed_reason[event_index] = staged.reason;
        arguments.event_state.typed_success[event_index] = staged.success;
        arguments.event_state.typed_value[event_index] = staged.value;
    }
    if (genome_v2_staged_children != nullptr && genome_v2_causal_enabled) {
        namespace genome_v2 = eiel::genome_upgrade_v2_q10_v1;
        const int64_t gv2_allele_stride =
            genome_v2::kHomologCount * genome_v2::kProductCount
            * genome_v2::kParameterCount * arguments.capacity;
        const int64_t gv2_modifier_stride =
            genome_v2::kHomologCount * genome_v2::kProductCount
            * arguments.capacity;
        const int64_t gv2_product_stride =
            genome_v2::kProductCount * arguments.capacity;
        const int64_t gv2_phenotype_stride =
            genome_v2::kPhenotypeCount * arguments.capacity;
        const genome_v2::ResidentPlanesV1 committed_planes{
            arguments.birth_state.genome_upgrade_v2_alleles_q10
                + cell * gv2_allele_stride,
            arguments.birth_state.genome_upgrade_v2_modifiers_q10
                + cell * gv2_modifier_stride,
            arguments.birth_state.genome_upgrade_v2_products_q10
                + cell * gv2_product_stride,
            arguments.birth_state.genome_upgrade_v2_phenotypes_q10
                + cell * gv2_phenotype_stride,
            arguments.birth_state.genome_upgrade_v2_expression_cost_q10
                + cell_capacity_offset,
            arguments.birth_state.genome_upgrade_v2_active_product_count
                + cell_capacity_offset,
            arguments.birth_state.genome_upgrade_v2_saturated_product_count
                + cell_capacity_offset,
            arguments.capacity,
        };
        for (int64_t birth = 0; birth < causal_capture_count; ++birth) {
            int64_t event_sequence = causal_output::kUnavailableValue;
            bool event_sequence_valid = false;
            for (int64_t row = 0; row < required_event_rows; ++row) {
                const scratch::BirthEventStagingRowV2 staged =
                    staged_events[row];
                if (
                    staged.event_type
                        == persistent_birth_events::kBirthCommittedEventType
                    && staged.success
                    && staged.child_id == child_ids[birth]
                ) {
                    event_sequence = staged.sequence;
                    event_sequence_valid = true;
                    break;
                }
            }
            write_genome_v2_causal_record_v1(
                arguments.birth_outputs.genome_v2_causal_record_rows,
                arguments.birth_outputs.genome_v2_causal_record_valid,
                arguments.birth_outputs.genome_v2_causal_record_capacity,
                causal_record_start + birth,
                causal_output::kOrdinaryBirthEventKind,
                cell,
                arguments.mechanics_config.generation,
                absolute_tick,
                event_sequence,
                event_sequence_valid,
                child_ids[birth],
                parent_agent_ids[birth],
                arguments.birth_outputs.committed_mate_agent_ids[
                    tick_birth_offset + birth
                ],
                child_slots[birth],
                arguments.birth_outputs.planned_parent_slots[
                    tick_birth_offset + birth
                ],
                arguments.birth_outputs.planned_mate_slots[
                    tick_birth_offset + birth
                ],
                child_founders[birth],
                child_depths[birth],
                committed_planes
            );
        }
        if (genome_v2_causal_storage_valid) {
            arguments.birth_outputs.genome_v2_causal_record_count[cell] =
                causal_record_start + causal_capture_count;
            arguments.birth_outputs
                .genome_v2_causal_record_dropped_count[cell] +=
                    planned_count - causal_capture_count;
        }
    }
    arguments.event_state.active_count[cell] = event_cursor + required_event_rows;
    arguments.event_state.status[cell] = persistent_birth_events::kStatusOk;

    if (planned_count > 0) {
        const FlatFreshNewbornSelectionState newborn_state{
            {arguments.energy_collected + cell_capacity_offset, 1},
            {arguments.energy_collected_exact + cell_capacity_offset, 1},
            {arguments.birth_state.attacks + cell_capacity_offset, 1},
            {arguments.birth_state.early_survival_steps + cell_capacity_offset, 1},
            {arguments.birth_state.early_energy_collected + cell_capacity_offset, 1},
            {arguments.birth_state.early_energy_collected_exact + cell_capacity_offset, 1},
            {arguments.birth_state.early_injury_gained + cell_capacity_offset, 1},
            {arguments.birth_state.early_life_weight + cell_capacity_offset, 1},
            {arguments.birth_state.pred_err_sum + cell_capacity_offset, 1},
            {arguments.birth_state.pred_err_count + cell_capacity_offset, 1},
            {arguments.birth_state.first_repro_step + cell_capacity_offset, 1},
            {arguments.birth_state.dev_stage_initialized + cell_capacity_offset, 1},
            {arguments.birth_state.episode0_pred_err_sum + cell_capacity_offset, 1},
            {arguments.birth_state.episode0_pred_err_count + cell_capacity_offset, 1},
            {arguments.birth_state.late_pred_err_sum + cell_capacity_offset, 1},
            {arguments.birth_state.late_pred_err_count + cell_capacity_offset, 1},
            {arguments.birth_state.chemistry_observation_fidelity + cell_capacity_offset, 1},
            {arguments.birth_state.chemistry_observation_bias + cell_capacity_offset, 1},
            {arguments.action_cache_actions
                 + cell * arguments.capacity * arguments.cache_capacity,
             arguments.cache_capacity, 1},
            {arguments.full_tick_state.prev_drive_delta
                 + cell_capacity_offset * arguments.drive_width,
             arguments.drive_width, 1},
            {arguments.full_tick_state.drive_volatility + cell_capacity_offset, 1},
            {arguments.full_tick_state.drive_jitter + cell_capacity_offset, 1},
            {arguments.full_tick_state.episode0_drive_jitter_sum + cell_capacity_offset, 1},
            {arguments.full_tick_state.last_energy_delta + cell_capacity_offset, 1},
            {arguments.full_tick_state.last_health_delta + cell_capacity_offset, 1},
            {arguments.full_tick_state.last_injury_delta + cell_capacity_offset, 1},
            {arguments.full_tick_state.selfharm_events + cell_capacity_offset, 1},
            {arguments.full_tick_state.episode0_selfharm_events + cell_capacity_offset, 1},
        };
        for (int64_t birth = 0; birth < planned_count; ++birth) {
            reset_ordinary_newborn_selection_state(
                newborn_state,
                child_slots[birth],
                {arguments.cache_capacity, arguments.drive_width}
            );
            if (
                arguments.birth_state.first_repro_step[
                    cell_capacity_offset + parent_slots[birth]
                ] < 0
            ) {
                arguments.birth_state.first_repro_step[
                    cell_capacity_offset + parent_slots[birth]
                ] = arguments.survival_steps[
                    cell_capacity_offset + parent_slots[birth]
                ];
            }
            if (arguments.birth_config.requires_mate) {
                const int64_t mate_slot =
                    arguments.birth_outputs.planned_mate_slots[
                        tick_birth_offset + birth
                    ];
                if (
                    arguments.birth_state.first_repro_step[
                        cell_capacity_offset + mate_slot
                    ] < 0
                ) {
                    arguments.birth_state.first_repro_step[
                        cell_capacity_offset + mate_slot
                    ] = arguments.survival_steps[
                        cell_capacity_offset + mate_slot
                    ];
                }
            }
        }
    }
    if (
        qualification_fault_stage
        == scratch::
            kGenomeUpgradeV2FaultImmediatelyAfterCommitBeforeReporting
    ) {
        arguments.birth_outputs.combined_status[cell_tick] = 2;
        return 250;
    }
    arguments.birth_outputs.combined_status[cell_tick] = 0;
    return 0;
}

__device__ __forceinline__ void
flattened_controller_birth_transaction_and_settlement_body_v1(
    const Phase2KernelArguments arguments,
    const int64_t controller_tick
) {
    if (threadIdx.x != 0) {
        return;
    }
    const int64_t cell = static_cast<int64_t>(blockIdx.x);
    if (arguments.status[cell] != 0) {
        return;
    }
    const int64_t output_tick = arguments.output_ticks_per_cell == 1
        ? 0
        : controller_tick;
    const int64_t cell_tick =
        cell * arguments.output_ticks_per_cell + output_tick;
    const int64_t cell_capacity_offset = cell * arguments.capacity;
    const int64_t cell_cache_offset =
        cell_capacity_offset * arguments.cache_capacity;

    if (
        arguments.birth_config.enabled
        && cross_path_resident_control_v1::ordinary_live_birth_cell_v1(
            arguments.birth_config.cross_path_control, cell
        )
    ) {
        // Runtime v2 is the accepted main route for legacy and diploid births,
        // and also carries the cross-path overlay when that authority is
        // selected. Keeping one transaction implementation prevents prepare
        // and commit from disagreeing at small birth capacities.
        const bool runtime_birth = true;
        flattened_birth_scratch_v2::FlatBirthScratchLayoutV2 runtime_layout{};
        flattened_birth_scratch_v2::FlatBirthScratchArenaV2 runtime_arena{};
        const bool runtime_scratch_valid = runtime_birth
            && make_runtime_birth_scratch_access_v2(
                arguments, &runtime_layout, &runtime_arena
            );
        const int32_t birth_status = runtime_birth
            ? commit_flattened_runtime_birth_transaction_v2(
                arguments,
                cell,
                controller_tick
            )
            : (arguments.birth_scratch.bytes_per_cell < kBirthScratchBytesPerCell
            ? 2
            : commit_flattened_birth_transaction_v1(
                arguments,
                cell,
                controller_tick
            ));
        (void)finalize_ordinary_cross_path_transaction_v1(
            arguments, cell, cell_tick, controller_tick, birth_status
        );
        if (
            !runtime_birth
            && arguments.birth_scratch.bytes_per_cell < kBirthScratchBytesPerCell
        ) {
            arguments.birth_outputs.combined_status[cell_tick] = 2;
        }
        if (birth_status != 0) {
            namespace scratch = flattened_birth_scratch_v1;
            const flattened_birth_scratch_v2::BirthScalarStatusesV2*
                runtime_statuses = runtime_scratch_valid
                ? runtime_arena.at<
                      flattened_birth_scratch_v2::BirthScalarStatusesV2
                  >(
                      cell,
                      runtime_layout,
                      flattened_birth_scratch_v2::kScalarStatuses
                  )
                : nullptr;
            const int32_t prepare_status = runtime_birth
                ? (runtime_statuses != nullptr
                       ? runtime_statuses[0].transaction_status
                       : 0)
                : (arguments.birth_scratch.bytes_per_cell
                        >= kBirthScratchBytesPerCell
                ? scratch::at<int32_t>(
                      arguments.birth_scratch,
                      cell,
                      scratch::kTransactionStatus
                  )[0]
                : 0);
            const int64_t event_cursor = arguments.event_state.active_count[cell];
            const int64_t required_event_rows = runtime_birth
                ? (runtime_statuses != nullptr
                       ? runtime_statuses[0].required_event_rows
                       : kFirstFailureNoFailureSentinelV1)
                : (arguments.birth_scratch.bytes_per_cell
                        >= kBirthScratchBytesPerCell
                ? scratch::at<int64_t>(
                      arguments.birth_scratch,
                      cell,
                      scratch::kRequiredEventRows
                  )[0]
                : kFirstFailureNoFailureSentinelV1);
            const bool event_capacity_failure =
                birth_status == 3
                && event_cursor >= 0
                && event_cursor <= arguments.event_state.capacity
                && required_event_rows > 0
                && required_event_rows
                    > arguments.event_state.capacity - event_cursor;
            const ControllerFailureStageV1 stage = birth_status == 3
                ? kControllerFailureStageBirthEventAppend
                : (prepare_status != 0
                       ? kControllerFailureStageBirthPrepare
                       : kControllerFailureStageBirthTransactionAndSettlement);
            record_status(
                arguments.status,
                arguments.first_failure_records,
                cell,
                arguments.mechanics_config.generation,
                arguments.mechanics_config.start_tick + controller_tick,
                stage,
                90 + birth_status,
                event_capacity_failure
                    ? kFirstFailureEventCapacityExceededV1
                    : kFirstFailureNoFailureSentinelV1,
                event_capacity_failure
                    ? event_cursor
                    : kFirstFailureNoFailureSentinelV1,
                event_capacity_failure
                    ? event_cursor + 1
                    : kFirstFailureNoFailureSentinelV1,
                event_capacity_failure
                    ? arguments.event_state.capacity
                    : kFirstFailureNoFailureSentinelV1
            );
        }
        if (
            birth_status == 0
            && compact_summary_context_addressable_v1(
                arguments.compact_summary_context, cell
            )
        ) {
            const auto access = compact_summary_reporting_access_v1(
                arguments.compact_summary_context, cell
            );
            if (arguments.compact_summary_context.slot_capacity
                != arguments.capacity) {
                reporting::record_failure_v1(
                    access.header, reporting::kLedgerInvalidStorageV1
                );
            } else {
                const int64_t committed_count =
                    arguments.birth_outputs.identity_committed_count[cell_tick];
                const int64_t birth_capacity =
                    arguments.birth_config.max_births_per_tick;
                const int64_t tick_birth_offset = cell_tick * birth_capacity;
                if (committed_count < 0 || committed_count > birth_capacity) {
                    reporting::record_failure_v1(
                        access.header, reporting::kLedgerInvalidInputV1
                    );
                }
                for (
                    int64_t birth = 0;
                    birth < committed_count && birth < birth_capacity;
                    ++birth
                ) {
                    const int64_t child_slot =
                        arguments.birth_outputs.committed_child_slots[
                            tick_birth_offset + birth
                        ];
                    const int64_t child_agent_id =
                        arguments.birth_outputs.committed_child_agent_ids[
                            tick_birth_offset + birth
                        ];
                    if (child_slot < 0 || child_slot >= arguments.capacity) {
                        reporting::record_failure_v1(
                            access.header, reporting::kLedgerInvalidInputV1
                        );
                        continue;
                    }
                    reporting::append_ordinary_birth_v1(
                        access,
                        child_agent_id,
                        static_cast<int32_t>(child_slot),
                        arguments.injury[cell_capacity_offset + child_slot]
                    );
                }
            }
        }
    }

    // Birth settlement changes occupancy after the tick core has already
    // refreshed Patch state. Match the CPU owner by publishing density from
    // the post-birth agent grid before the next tick or checkpoint observes it.
    if (
        arguments.status[cell] == 0
        && arguments.patch_p2_context.enabled
        && arguments.patch_p2_context.mode == FlatPatchP2Mode::kFullPatchP2
    ) {
        const int64_t cell_grid_offset =
            cell * arguments.height * arguments.width;
        persistent_patch_p2::refresh_patch_density_exact(
            arguments.agent_grid + cell_grid_offset,
            arguments.patch_p2_context.state.patch_density + cell_grid_offset,
            arguments.patch_p2_context.config
        );
    }

    // Stage 2's public checkpoint value is the post-service cached action.
    for (int64_t slot = 0; slot < arguments.capacity; ++slot) {
        const int64_t slot_index = cell_capacity_offset + slot;
        const int32_t cursor = arguments.action_cache_cursor[slot_index];
        if (
            !arguments.alive_mask[slot_index]
            || arguments.action_cache_token[slot_index] < 0
            || cursor < 0
        ) {
            arguments.action_cache_next[slot_index] = -1;
            continue;
        }
        const int64_t projected_index = cursor < arguments.cache_width
            ? static_cast<int64_t>(cursor)
            : arguments.cache_width - 1;
        arguments.action_cache_next[slot_index] = arguments.action_cache_actions[
            cell_cache_offset
                + slot * arguments.cache_capacity
                + projected_index
        ];
    }
    arguments.event_state.status_by_tick[cell_tick] =
        arguments.event_state.status[cell];
    if (arguments.event_state.status[cell] != 0) {
        const bool event_capacity_failure =
            arguments.event_state.status[cell]
            == persistent_birth_events::kStatusCapacityExceeded;
        const int64_t event_cursor = arguments.event_state.active_count[cell];
        record_status(
            arguments.status,
            arguments.first_failure_records,
            cell,
            arguments.mechanics_config.generation,
            arguments.mechanics_config.start_tick + controller_tick,
            kControllerFailureStageBirthEventAppend,
            80 + arguments.event_state.status[cell],
            event_capacity_failure
                ? kFirstFailureEventCapacityExceededV1
                : kFirstFailureNoFailureSentinelV1,
            event_capacity_failure
                ? event_cursor
                : kFirstFailureNoFailureSentinelV1,
            event_capacity_failure
                ? event_cursor + 1
                : kFirstFailureNoFailureSentinelV1,
            event_capacity_failure
                ? arguments.event_state.capacity
                : kFirstFailureNoFailureSentinelV1
        );
    }
    arguments.lifecycle_status[cell_tick] = arguments.status[cell];
    if (
        controller_tick + 1 == arguments.ticks
        && compact_summary_context_addressable_v1(
            arguments.compact_summary_context, cell
        )
    ) {
        const auto access = compact_summary_reporting_access_v1(
            arguments.compact_summary_context, cell
        );
        if (
            !compact_summary_context_matches_controller_v1(
                arguments.compact_summary_context, arguments.capacity
            )
            || arguments.event_state.active_count == nullptr
        ) {
            reporting::record_failure_v1(
                access.header, reporting::kLedgerInvalidStorageV1
            );
        } else {
            const int64_t reporting_row = compact_summary_row_index_v1(
                arguments.compact_summary_context, cell
            );
            int64_t absolute_event_cursor = 0;
            if (!absolute_event_cursor_v1(
                    arguments.event_state, cell, &absolute_event_cursor
                )) {
                reporting::record_failure_v1(
                    access.header, reporting::kLedgerInvalidStorageV1
                );
            } else {
                arguments.compact_summary_context.event_cursor_pre_turnover[
                    reporting_row
                ] = absolute_event_cursor;
            }
            const auto sources = compact_summary_resident_sources_v1(
                arguments.slot_to_agent_id,
                arguments.energy_collected_exact,
                arguments.injury,
                arguments.action_switches,
                arguments.action_oscillations,
                arguments.survival_steps,
                arguments.policy_energy_observation,
                arguments.health,
                arguments.drives,
                arguments.pos_y,
                arguments.pos_x,
                arguments.full_tick_state,
                arguments.event_state,
                arguments.birth_state,
                cell,
                arguments.capacity,
                arguments.drive_width
            );
            if (arguments.status[cell] == 0) {
                if (
                    arguments.drive_width != 4
                    || !reporting::resident_sources_valid_v1(
                        sources,
                        arguments.compact_summary_context.slot_capacity
                    )
                ) {
                    reporting::record_failure_v1(
                        access.header, reporting::kLedgerInvalidStorageV1
                    );
                } else {
                    reporting::finalize_pre_turnover_v1(
                        access,
                        sources,
                        arguments.compact_summary_context.reporting_fitness_config
                    );
                }
            }
        }
    }
    if (
        arguments.generation_summary_scalar_out != nullptr
        && controller_tick + 1 == arguments.ticks
    ) {
        double* const summary = arguments.generation_summary_scalar_out
            + cell * kGenerationSummaryScalarWidthV1;
        int64_t absolute_event_cursor = 0;
        summary[kGenerationSummaryEventCursorPreTurnover] =
            absolute_event_cursor_v1(
                arguments.event_state, cell, &absolute_event_cursor
            )
            ? static_cast<double>(absolute_event_cursor)
            : NAN;
    }
}

__global__ void flattened_controller_birth_transaction_and_settlement_kernel_v1(
    const Phase2KernelArguments arguments,
    const int64_t controller_tick
) {
    flattened_controller_birth_transaction_and_settlement_body_v1(
        arguments,
        controller_tick
    );
}

__global__ void flattened_controller_birth_prepare_and_transaction_kernel_v1(
    const Phase2KernelArguments arguments,
    const int64_t controller_tick
) {
    if (threadIdx.x != 0) {
        return;
    }
    const int64_t cell = static_cast<int64_t>(blockIdx.x);
    if (
        arguments.status[cell] == 0
        && arguments.birth_config.enabled
        && cross_path_resident_control_v1::ordinary_live_birth_cell_v1(
            arguments.birth_config.cross_path_control, cell
        )
    ) {
        const int64_t output_tick = arguments.output_ticks_per_cell == 1
            ? 0
            : controller_tick;
        const int64_t cell_tick =
            cell * arguments.output_ticks_per_cell + output_tick;
        const int32_t runtime_status =
            prepare_flattened_runtime_birth_transaction_v2(
                arguments,
                cell,
                controller_tick
            );
        arguments.birth_outputs.combined_status[cell_tick] = runtime_status == 0
            ? 0
            : (runtime_status == 1
                   ? 1
                   : (runtime_status == 4
                          ? 4
                          : (runtime_status == 25 ? 3 : 2)));
    }
    flattened_controller_birth_transaction_and_settlement_body_v1(
        arguments,
        controller_tick
    );
}

struct Phase1KernelArguments {
    int64_t* active_agent_ids;
    int32_t* active_agent_slots;
    int64_t* active_count_state;
    int64_t* slot_to_agent_id;
    int32_t* agent_id_to_slot;
    bool* alive_mask;
    int64_t* slot_tokens;
    int32_t* action_cache_actions;
    int64_t* action_cache_token;
    int32_t* action_cache_cursor;
    int32_t* action_cache_next;
    int32_t* pos_y;
    int32_t* pos_x;
    int32_t* agent_grid;
    int32_t* food_grid;
    float* energy;
    float* health;
    float* injury;
    double* policy_energy_observation;
    int32_t* rest_streak;
    int64_t* pending_free_agent_ids;
    int64_t* pending_free_slots;
    int64_t* pending_free_count;
    float* energy_collected;
    double* energy_collected_exact;
    int32_t* last_action;
    int32_t* last_last_action;
    int32_t* action_switches;
    int32_t* action_oscillations;
    int32_t* episode0_switches;
    int32_t* episode0_oscillations;
    int32_t* survival_steps;
    float* mlp_pack;
    float* drives;
    float* attack_bias;
    float* action_temperature;
    float* drive_logit_gain;
    float* food_gained_scratch;
    int64_t* ordered_agent_ids_out;
    int32_t* ordered_slots_out;
    float* observations_out;
    float* logits_out;
    int32_t* row_actions_out;
    int32_t* actions_by_slot_out;
    float* pre_energy_out;
    float* pre_health_out;
    float* pre_injury_out;
    int32_t* pre_survival_steps_out;
    int64_t* active_count_out;
    int64_t* policy_refill_rows_out;
    int64_t* policy_cache_hit_rows_out;
    int64_t* runtime_bookkeeping_by_tick_out;
    int64_t* death_agent_ids_out;
    int64_t* death_slots_out;
    int32_t* death_reason_codes_out;
    int64_t* death_count_out;
    int64_t* survivor_agent_ids_out;
    int64_t* survivor_count_out;
    int32_t* lifecycle_status_out;
    int32_t* status_out;
    int64_t* first_failure_records;
    double* generation_summary_scalar_out;
    FlatFullTickState full_tick_state;
    FlatFullTickOutputs full_tick_outputs;
    FlatEventState event_state;
    FlatOrdinaryBirthState ordinary_birth_state;
    FlatOrdinaryBirthOutputs ordinary_birth_outputs;
    FlatFlattenedBirthScratch ordinary_birth_scratch;
    FlatOrdinaryBirthConfig ordinary_birth_config;
    FlatMechanicsConfig mechanics_config;
    FlatPatchP2Context patch_p2_context;
    FlatCompactSummaryContextV1 compact_summary_context;
    bool generation_turnover_enabled;
    int64_t ticks;
    int64_t output_ticks_per_cell;
    int64_t resident_ticks_per_cell;
    int64_t active_capacity;
    int64_t agent_id_capacity;
    int64_t capacity;
    int64_t pending_capacity;
    int64_t height;
    int64_t width;
    int64_t view_radius;
    int64_t observation_size;
    int64_t hidden_size;
    int64_t action_count;
    int64_t packed_width;
    int64_t drive_width;
    int64_t cache_capacity;
    int64_t cache_width;
    bool deterministic_actions;
    int64_t max_food;
    int64_t effective_food_spawn_count;
    double max_energy;
    double start_health;
    const double* move_cost_by_cell;
    double gather_cost;
    double rest_cost;
    double attack_cost;
    double action_cost_mult;
    double rest_cost_mult;
    double food_energy;
    int64_t episode0_k;
    float w_hunger_gather;
    float w_pain_rest;
    float w_fatigue_rest;
    float w_pain_attack;
    float w_fear_attack;
    TickCoreExecutionModeV1 execution_mode;
};

__device__ __noinline__ void patch_p2_single_tick_exact_from_flat_controller(
    const FlatPatchP2Context context,
    const int64_t cell,
    const int64_t cell_grid_offset,
    const int64_t scratch_offset,
    const int64_t cell_tick
) {
    auto cell_config = context.config;
    if (context.patch_food_regrow_per_step_by_cell != nullptr) {
        cell_config.food_regrow_per_step = static_cast<float>(
            context.patch_food_regrow_per_step_by_cell[cell]
        );
    }
    persistent_patch_p2::patch_p2_single_tick_exact(
        {
            context.state.patch_food_amount + cell_grid_offset,
            context.state.patch_fertility + cell_grid_offset,
            context.state.patch_scent + cell_grid_offset,
            context.state.patch_waste + cell_grid_offset,
            context.state.patch_scent_emit_pending + cell_grid_offset,
            context.state.patch_waste_emit_pending + cell_grid_offset,
            context.state.agent_grid + cell_grid_offset,
            context.state.patch_density + cell_grid_offset,
            context.state.food_grid + cell_grid_offset,
        },
        {
            context.scratch.multiplier + scratch_offset,
            context.scratch.waste_penalty + scratch_offset,
            context.scratch.density_penalty + scratch_offset,
            context.scratch.sorted_multiplier + scratch_offset,
        },
        {
            context.outputs.stats
                + cell_tick * persistent_patch_p2::kStatCount,
        },
        cell_config
    );
}

__device__ __noinline__ void
patch_p2_single_tick_exact_block_256_from_flat_controller(
    const FlatPatchP2Context context,
    const int64_t cell,
    const int64_t cell_grid_offset,
    const int64_t scratch_offset,
    const int64_t cell_tick
) {
    auto cell_config = context.config;
    if (context.patch_food_regrow_per_step_by_cell != nullptr) {
        cell_config.food_regrow_per_step = static_cast<float>(
            context.patch_food_regrow_per_step_by_cell[cell]
        );
    }
    persistent_patch_p2::patch_p2_single_tick_exact_block_256(
        {
            context.state.patch_food_amount + cell_grid_offset,
            context.state.patch_fertility + cell_grid_offset,
            context.state.patch_scent + cell_grid_offset,
            context.state.patch_waste + cell_grid_offset,
            context.state.patch_scent_emit_pending + cell_grid_offset,
            context.state.patch_waste_emit_pending + cell_grid_offset,
            context.state.agent_grid + cell_grid_offset,
            context.state.patch_density + cell_grid_offset,
            context.state.food_grid + cell_grid_offset,
        },
        {
            context.scratch.multiplier + scratch_offset,
            context.scratch.waste_penalty + scratch_offset,
            context.scratch.density_penalty + scratch_offset,
            context.scratch.sorted_multiplier + scratch_offset,
        },
        {
            context.outputs.stats
                + cell_tick * persistent_patch_p2::kStatCount,
        },
        cell_config
    );
}

__device__ __forceinline__ bool patch_p2_dimensions_match(
    const FlatPatchP2Context context,
    const int64_t grid_height,
    const int64_t grid_width
) {
    return context.config.height == grid_height
        && context.config.width == grid_width;
}

__device__ __forceinline__ int64_t sync_patch_spawned_food_exact(
    int32_t* __restrict__ food_grid,
    float* __restrict__ patch_food_amount,
    const int64_t cell_count,
    const float presence_threshold,
    const float initial_food_value
) {
    int64_t synced = 0;
    for (int64_t cell = 0; cell < cell_count; ++cell) {
        if (
            food_grid[cell] == kFood
            && patch_food_amount[cell] <= presence_threshold
        ) {
            patch_food_amount[cell] = initial_food_value;
            ++synced;
        }
    }
    for (int64_t cell = 0; cell < cell_count; ++cell) {
        food_grid[cell] = patch_food_amount[cell] > presence_threshold
            ? kFood
            : kEmpty;
    }
    return synced;
}

__device__ __noinline__ void write_patch_radius_one_observation_exact(
    const FlatPatchP2Context context,
    const int32_t* __restrict__ agent_grid,
    const int32_t* __restrict__ food_grid,
    const float* __restrict__ energy,
    const float* __restrict__ health,
    const int64_t grid_offset,
    const int64_t slot_index,
    const int32_t center_y,
    const int32_t center_x,
    const int64_t grid_height,
    const int64_t grid_width,
    const float max_energy,
    const float start_health,
    float* __restrict__ observation
) {
    constexpr int64_t local_size = 9;
    float density_denom = static_cast<float>(
        (2 * context.config.density_radius + 1)
        * (2 * context.config.density_radius + 1)
    );
    if (!context.config.density_include_self) {
        density_denom -= 1.0f;
    }
    if (density_denom <= 0.0f) {
        density_denom = 1.0f;
    }
    const float movement_min = static_cast<float>(
        context.patch_movement_cost_min
    );
    const float movement_max = static_cast<float>(
        context.patch_movement_cost_max
    );
    const float movement_span = movement_max - movement_min;

    int64_t offset = 0;
    for (int64_t dy = -1; dy <= 1; ++dy) {
        for (int64_t dx = -1; dx <= 1; ++dx) {
            const int64_t y = static_cast<int64_t>(center_y) + dy;
            const int64_t x = static_cast<int64_t>(center_x) + dx;
            float legacy = 0.0f;
            float food_norm = 0.0f;
            float cover = 0.0f;
            float shelter = 0.0f;
            float scent_norm = 0.0f;
            float waste_norm = 0.0f;
            float density_norm = 0.0f;
            float movement_norm = 0.0f;
            if (y >= 0 && y < grid_height && x >= 0 && x < grid_width) {
                const int64_t index = grid_offset + y * grid_width + x;
                if (agent_grid[index] != -1) {
                    legacy = -1.0f;
                } else if (food_grid[index] == kFood) {
                    legacy = 1.0f;
                }
                food_norm = safe_norm_patch_observation(
                    context.state.patch_food_amount[index],
                    context.config.food_max
                );
                cover = clamp01_patch_observation(context.patch_cover[index]);
                shelter = clamp01_patch_observation(
                    context.patch_shelter[index]
                );
                scent_norm = safe_norm_patch_observation(
                    context.state.patch_scent[index], context.config.scent_max
                );
                waste_norm = safe_norm_patch_observation(
                    context.state.patch_waste[index], context.config.waste_max
                );
                density_norm = context.config.density_normalize
                    ? clamp01_patch_observation(
                        context.state.patch_density[index]
                    )
                    : safe_norm_patch_observation(
                        context.state.patch_density[index], density_denom
                    );
                if (movement_span > 0.0f) {
                    movement_norm = clamp01_patch_observation(
                        (context.patch_movement_cost[index] - movement_min)
                            / movement_span
                    );
                }
            }
            observation[offset] = legacy;
            observation[local_size + offset] = food_norm;
            observation[2 * local_size + offset] = cover;
            observation[3 * local_size + offset] = shelter;
            observation[4 * local_size + offset] = scent_norm;
            observation[5 * local_size + offset] = waste_norm;
            observation[6 * local_size + offset] = density_norm;
            observation[7 * local_size + offset] = movement_norm;
            ++offset;
        }
    }
    observation[kPatchObservationChannels * local_size] =
        energy[slot_index] / max_energy;
    observation[kPatchObservationChannels * local_size + 1] =
        health[slot_index] / start_health;
}

// Patch observations are held in the caller's preallocated output tensor. This
// helper retains only the two established 64-wide hidden layers on its stack.
__device__ __noinline__ void service_patch_mlp_gumbel_policy_serial(
    const int64_t* __restrict__ ordered_agent_ids,
    const int32_t* __restrict__ ordered_slots,
    const float* __restrict__ observations,
    const float* __restrict__ mlp_pack,
    const float* __restrict__ drives,
    const float* __restrict__ attack_bias,
    const float* __restrict__ action_temperature,
    const float* __restrict__ drive_logit_gain,
    const int64_t* __restrict__ slot_tokens,
    int32_t* __restrict__ action_cache_actions,
    int64_t* __restrict__ action_cache_token,
    int32_t* __restrict__ action_cache_cursor,
    int32_t* __restrict__ action_cache_next,
    float* __restrict__ logits_out,
    int32_t* __restrict__ row_actions_out,
    int64_t* __restrict__ miss_rows,
    uint64_t* __restrict__ pcg64_state_words,
    int64_t* __restrict__ pcg64_buffer_words,
    persistent_stochastic_policy::PolicyCacheCounters* __restrict__ counters_out,
    const persistent_stochastic_policy::FixedMlpPolicyConfig config
) {
    using persistent_stochastic_policy::PolicyCacheCounters;
    PolicyCacheCounters counters{};
    const int64_t active_rows = config.active_rows < 0 ? 0 : config.active_rows;
    if (
        active_rows > config.capacity
        || config.observation_size != kPatchRadiusOneObservationSize
        || config.hidden_size <= 0
        || config.hidden_size > kMaxHiddenSize
        || config.action_count <= 0
        || config.action_count > kMaxActionCount
        || config.drive_width < 4
        || config.cache_width <= 0
        || config.cache_width > config.cache_capacity
    ) {
        *counters_out = counters;
        return;
    }

    for (int64_t row = 0; row < active_rows; ++row) {
        ++counters.requested_rows;
        const int32_t slot = ordered_slots[row];
        if (slot < 0 || static_cast<int64_t>(slot) >= config.capacity) {
            miss_rows[counters.missed_rows++] = row;
            continue;
        }
        const int32_t cursor = action_cache_cursor[slot];
        const bool cache_hit = action_cache_token[slot] == slot_tokens[slot]
            && cursor >= 0
            && static_cast<int64_t>(cursor) < config.cache_width;
        if (!cache_hit) {
            miss_rows[counters.missed_rows++] = row;
            continue;
        }

        int32_t action = action_cache_actions[
            static_cast<int64_t>(slot) * config.cache_capacity + cursor
        ];
        if (action < 0 || action > kAttackAction) {
            action = action_cache_next[slot];
        }
        if (action < 0 || action > kAttackAction) {
            action = kRestAction;
        }
        action_cache_next[slot] = action;
        action_cache_cursor[slot] = cursor + 1;
        row_actions_out[row] = action;
        ++counters.hit_rows;
    }

    if (config.cache_width > 1) {
        for (int64_t left = 0; left < counters.missed_rows; ++left) {
            int64_t selected = left;
            for (
                int64_t candidate = left + 1;
                candidate < counters.missed_rows;
                ++candidate
            ) {
                const int64_t selected_row = miss_rows[selected];
                const int64_t candidate_row = miss_rows[candidate];
                const int64_t selected_id = ordered_agent_ids[selected_row];
                const int64_t candidate_id = ordered_agent_ids[candidate_row];
                if (
                    candidate_id < selected_id
                    || (
                        candidate_id == selected_id
                        && candidate_row < selected_row
                    )
                ) {
                    selected = candidate;
                }
            }
            const int64_t tmp = miss_rows[left];
            miss_rows[left] = miss_rows[selected];
            miss_rows[selected] = tmp;
        }
    }

    Pcg64State rng{
        pcg64_state_words[0],
        pcg64_state_words[1],
        pcg64_state_words[2],
        pcg64_state_words[3],
        static_cast<uint32_t>(pcg64_buffer_words[0]),
        static_cast<uint32_t>(pcg64_buffer_words[1]),
    };
    for (int64_t miss_index = 0; miss_index < counters.missed_rows; ++miss_index) {
        const int64_t row = miss_rows[miss_index];
        const int32_t slot = ordered_slots[row];
        if (slot < 0 || static_cast<int64_t>(slot) >= config.capacity) {
            row_actions_out[row] = -1;
            continue;
        }

        const float* observation = observations
            + row * kPatchRadiusOneObservationSize;
        const float* pack = mlp_pack
            + static_cast<int64_t>(slot) * config.packed_width;
        float h1[kMaxHiddenSize];
        float h2[kMaxHiddenSize];
        int64_t pack_offset = 0;
        for (int64_t hidden = 0; hidden < config.hidden_size; ++hidden) {
            float value = pack[
                config.hidden_size * config.observation_size + hidden
            ];
            const float* weights = pack + hidden * config.observation_size;
            for (
                int64_t feature = 0;
                feature < config.observation_size;
                ++feature
            ) {
                value += weights[feature] * observation[feature];
            }
            h1[hidden] = tanhf(value);
        }
        pack_offset += config.hidden_size * config.observation_size
            + config.hidden_size;
        for (int64_t hidden = 0; hidden < config.hidden_size; ++hidden) {
            float value = pack[
                pack_offset + config.hidden_size * config.hidden_size + hidden
            ];
            const float* weights = pack
                + pack_offset + hidden * config.hidden_size;
            for (int64_t input = 0; input < config.hidden_size; ++input) {
                value += weights[input] * h1[input];
            }
            h2[hidden] = tanhf(value);
        }
        pack_offset += config.hidden_size * config.hidden_size
            + config.hidden_size;

        const float temperature = action_temperature[slot] < 0.05f
            ? 0.05f
            : action_temperature[slot];
        const float gain = drive_logit_gain[slot] < 0.0f
            ? 0.0f
            : drive_logit_gain[slot];
        const float* slot_drives = drives
            + static_cast<int64_t>(slot) * config.drive_width;
        float* row_logits = logits_out + row * config.action_count;
        for (int64_t action = 0; action < config.action_count; ++action) {
            const float* weights = pack
                + pack_offset + action * config.hidden_size;
            float score = pack[
                pack_offset + config.action_count * config.hidden_size + action
            ];
            for (int64_t hidden = 0; hidden < config.hidden_size; ++hidden) {
                score += weights[hidden] * h2[hidden];
            }
            if (action == kAttackAction) {
                score += attack_bias[slot];
            }
            score /= temperature;
            if (action == kGatherAction) {
                score += gain * (config.w_hunger_gather * slot_drives[0]);
            } else if (action == kRestAction) {
                score += gain * (
                    config.w_pain_rest * slot_drives[1]
                    + config.w_fatigue_rest * slot_drives[2]
                );
            } else if (action == kAttackAction) {
                score -= gain * (
                    config.w_pain_attack * slot_drives[1]
                    + config.w_fear_attack * slot_drives[3]
                );
            }
            row_logits[action] = score;
        }

        const int64_t cache_offset = static_cast<int64_t>(slot)
            * config.cache_capacity;
        for (
            int64_t cache_index = 0;
            cache_index < config.cache_width;
            ++cache_index
        ) {
            int32_t best_action = 0;
            float uniform = pcg64_numpy_float32(rng);
            uniform = uniform < persistent_stochastic_policy::kGumbelEps
                ? persistent_stochastic_policy::kGumbelEps
                : (
                    uniform
                        > 1.0f - persistent_stochastic_policy::kGumbelEps
                    ? 1.0f - persistent_stochastic_policy::kGumbelEps
                    : uniform
                );
            float best_score = row_logits[0] + -logf(-logf(uniform));
            for (int64_t action = 1; action < config.action_count; ++action) {
                uniform = pcg64_numpy_float32(rng);
                uniform = uniform < persistent_stochastic_policy::kGumbelEps
                    ? persistent_stochastic_policy::kGumbelEps
                    : (
                        uniform
                            > 1.0f - persistent_stochastic_policy::kGumbelEps
                        ? 1.0f - persistent_stochastic_policy::kGumbelEps
                        : uniform
                    );
                const float score = row_logits[action]
                    + -logf(-logf(uniform));
                if (score > best_score) {
                    best_score = score;
                    best_action = static_cast<int32_t>(action);
                }
            }
            action_cache_actions[cache_offset + cache_index] = best_action;
            if (cache_index == 0) {
                row_actions_out[row] = best_action;
                action_cache_next[slot] = best_action;
            }
        }
        action_cache_token[slot] = slot_tokens[slot];
        action_cache_cursor[slot] = 1;
        ++counters.refilled_rows;
    }

    counters.refill_calls = counters.refilled_rows > 0 ? 1 : 0;
    counters.full_refill_ticks = counters.refilled_rows > 0
            && counters.hit_rows == 0
        ? 1
        : 0;
    counters.partial_refill_ticks = counters.refilled_rows > 0
            && counters.hit_rows > 0
        ? 1
        : 0;
    counters.gumbel_draws = counters.refilled_rows
        * config.cache_width * config.action_count;
    pcg64_state_words[0] = rng.state_lo;
    pcg64_state_words[1] = rng.state_hi;
    pcg64_state_words[2] = rng.inc_lo;
    pcg64_state_words[3] = rng.inc_hi;
    pcg64_buffer_words[0] = static_cast<int64_t>(rng.has_uint32);
    pcg64_buffer_words[1] = static_cast<int64_t>(rng.uinteger);
    *counters_out = counters;
}

__device__ __forceinline__ bool patch_mlp_policy_config_valid(
    const persistent_stochastic_policy::FixedMlpPolicyConfig config
) {
    const int64_t active_rows = config.active_rows < 0 ? 0 : config.active_rows;
    return !(
        active_rows > config.capacity
        || config.observation_size != kPatchRadiusOneObservationSize
        || config.hidden_size <= 0
        || config.hidden_size > kMaxHiddenSize
        || config.action_count <= 0
        || config.action_count > kMaxActionCount
        || config.drive_width < 4
        || config.cache_width <= 0
        || config.cache_width > config.cache_capacity
    );
}

__device__ __forceinline__ void classify_patch_policy_rows_serial(
    const int64_t* __restrict__ ordered_agent_ids,
    const int32_t* __restrict__ ordered_slots,
    const int64_t* __restrict__ slot_tokens,
    int32_t* __restrict__ action_cache_actions,
    int64_t* __restrict__ action_cache_token,
    int32_t* __restrict__ action_cache_cursor,
    int32_t* __restrict__ action_cache_next,
    int32_t* __restrict__ row_actions_out,
    int64_t* __restrict__ miss_rows,
    persistent_stochastic_policy::PolicyCacheCounters* __restrict__ counters_out,
    const persistent_stochastic_policy::FixedMlpPolicyConfig config
) {
    using persistent_stochastic_policy::PolicyCacheCounters;
    PolicyCacheCounters counters{};
    if (!patch_mlp_policy_config_valid(config)) {
        *counters_out = counters;
        return;
    }

    const int64_t active_rows = config.active_rows < 0 ? 0 : config.active_rows;
    for (int64_t row = 0; row < active_rows; ++row) {
        ++counters.requested_rows;
        const int32_t slot = ordered_slots[row];
        if (slot < 0 || static_cast<int64_t>(slot) >= config.capacity) {
            miss_rows[counters.missed_rows++] = row;
            continue;
        }
        const int32_t cursor = action_cache_cursor[slot];
        const bool cache_hit = action_cache_token[slot] == slot_tokens[slot]
            && cursor >= 0
            && static_cast<int64_t>(cursor) < config.cache_width;
        if (!cache_hit) {
            miss_rows[counters.missed_rows++] = row;
            continue;
        }

        int32_t action = action_cache_actions[
            static_cast<int64_t>(slot) * config.cache_capacity + cursor
        ];
        if (action < 0 || action > kAttackAction) {
            action = action_cache_next[slot];
        }
        if (action < 0 || action > kAttackAction) {
            action = kRestAction;
        }
        action_cache_next[slot] = action;
        action_cache_cursor[slot] = cursor + 1;
        row_actions_out[row] = action;
        ++counters.hit_rows;
    }

    if (config.cache_width > 1) {
        for (int64_t left = 0; left < counters.missed_rows; ++left) {
            int64_t selected = left;
            for (
                int64_t candidate = left + 1;
                candidate < counters.missed_rows;
                ++candidate
            ) {
                const int64_t selected_row = miss_rows[selected];
                const int64_t candidate_row = miss_rows[candidate];
                const int64_t selected_id = ordered_agent_ids[selected_row];
                const int64_t candidate_id = ordered_agent_ids[candidate_row];
                if (
                    candidate_id < selected_id
                    || (
                        candidate_id == selected_id
                        && candidate_row < selected_row
                    )
                ) {
                    selected = candidate;
                }
            }
            const int64_t tmp = miss_rows[left];
            miss_rows[left] = miss_rows[selected];
            miss_rows[selected] = tmp;
        }
    }
    *counters_out = counters;
}

#if EIEL_CUDA_STAGE3_PARALLEL_PATCH_POLICY_ROWS
__device__ __forceinline__ void compute_patch_policy_row_logits_exact(
    const int64_t row,
    const int32_t* __restrict__ ordered_slots,
    const float* __restrict__ observations,
    const float* __restrict__ mlp_pack,
    const float* __restrict__ drives,
    const float* __restrict__ attack_bias,
    const float* __restrict__ action_temperature,
    const float* __restrict__ drive_logit_gain,
    float* __restrict__ logits_out,
    const persistent_stochastic_policy::FixedMlpPolicyConfig config
) {
    const int32_t slot = ordered_slots[row];
    if (slot < 0 || static_cast<int64_t>(slot) >= config.capacity) {
        return;
    }

    const float* observation = observations
        + row * kPatchRadiusOneObservationSize;
    const float* pack = mlp_pack
        + static_cast<int64_t>(slot) * config.packed_width;
    float h1[kMaxHiddenSize];
    float h2[kMaxHiddenSize];
    int64_t pack_offset = 0;
    for (int64_t hidden = 0; hidden < config.hidden_size; ++hidden) {
        float value = pack[
            config.hidden_size * config.observation_size + hidden
        ];
        const float* weights = pack + hidden * config.observation_size;
        for (
            int64_t feature = 0;
            feature < config.observation_size;
            ++feature
        ) {
            value += weights[feature] * observation[feature];
        }
        h1[hidden] = tanhf(value);
    }
    pack_offset += config.hidden_size * config.observation_size
        + config.hidden_size;
    for (int64_t hidden = 0; hidden < config.hidden_size; ++hidden) {
        float value = pack[
            pack_offset + config.hidden_size * config.hidden_size + hidden
        ];
        const float* weights = pack
            + pack_offset + hidden * config.hidden_size;
        for (int64_t input = 0; input < config.hidden_size; ++input) {
            value += weights[input] * h1[input];
        }
        h2[hidden] = tanhf(value);
    }
    pack_offset += config.hidden_size * config.hidden_size
        + config.hidden_size;

    const float temperature = action_temperature[slot] < 0.05f
        ? 0.05f
        : action_temperature[slot];
    const float gain = drive_logit_gain[slot] < 0.0f
        ? 0.0f
        : drive_logit_gain[slot];
    const float* slot_drives = drives
        + static_cast<int64_t>(slot) * config.drive_width;
    float* row_logits = logits_out + row * config.action_count;
    for (int64_t action = 0; action < config.action_count; ++action) {
        const float* weights = pack
            + pack_offset + action * config.hidden_size;
        float score = pack[
            pack_offset + config.action_count * config.hidden_size + action
        ];
        for (int64_t hidden = 0; hidden < config.hidden_size; ++hidden) {
            score += weights[hidden] * h2[hidden];
        }
        if (action == kAttackAction) {
            score += attack_bias[slot];
        }
        score /= temperature;
        if (action == kGatherAction) {
            score += gain * (config.w_hunger_gather * slot_drives[0]);
        } else if (action == kRestAction) {
            score += gain * (
                config.w_pain_rest * slot_drives[1]
                + config.w_fatigue_rest * slot_drives[2]
            );
        } else if (action == kAttackAction) {
            score -= gain * (
                config.w_pain_attack * slot_drives[1]
                + config.w_fear_attack * slot_drives[3]
            );
        }
        row_logits[action] = score;
    }
}

__device__ __forceinline__ void consume_patch_policy_rows_serial(
    const int32_t* __restrict__ ordered_slots,
    const int64_t* __restrict__ slot_tokens,
    int32_t* __restrict__ action_cache_actions,
    int64_t* __restrict__ action_cache_token,
    int32_t* __restrict__ action_cache_cursor,
    int32_t* __restrict__ action_cache_next,
    const float* __restrict__ logits_out,
    int32_t* __restrict__ row_actions_out,
    const int64_t* __restrict__ miss_rows,
    uint64_t* __restrict__ pcg64_state_words,
    int64_t* __restrict__ pcg64_buffer_words,
    persistent_stochastic_policy::PolicyCacheCounters* __restrict__ counters_io,
    const persistent_stochastic_policy::FixedMlpPolicyConfig config
) {
    using persistent_stochastic_policy::PolicyCacheCounters;
    if (!patch_mlp_policy_config_valid(config)) {
        return;
    }

    PolicyCacheCounters counters = *counters_io;
    Pcg64State rng{
        pcg64_state_words[0],
        pcg64_state_words[1],
        pcg64_state_words[2],
        pcg64_state_words[3],
        static_cast<uint32_t>(pcg64_buffer_words[0]),
        static_cast<uint32_t>(pcg64_buffer_words[1]),
    };
    for (int64_t miss_index = 0; miss_index < counters.missed_rows; ++miss_index) {
        const int64_t row = miss_rows[miss_index];
        const int32_t slot = ordered_slots[row];
        if (slot < 0 || static_cast<int64_t>(slot) >= config.capacity) {
            row_actions_out[row] = -1;
            continue;
        }

        const float* row_logits = logits_out + row * config.action_count;
        const int64_t cache_offset = static_cast<int64_t>(slot)
            * config.cache_capacity;
        for (
            int64_t cache_index = 0;
            cache_index < config.cache_width;
            ++cache_index
        ) {
            int32_t best_action = 0;
            float uniform = pcg64_numpy_float32(rng);
            uniform = uniform < persistent_stochastic_policy::kGumbelEps
                ? persistent_stochastic_policy::kGumbelEps
                : (
                    uniform
                        > 1.0f - persistent_stochastic_policy::kGumbelEps
                    ? 1.0f - persistent_stochastic_policy::kGumbelEps
                    : uniform
                );
            float best_score = row_logits[0] + -logf(-logf(uniform));
            for (int64_t action = 1; action < config.action_count; ++action) {
                uniform = pcg64_numpy_float32(rng);
                uniform = uniform < persistent_stochastic_policy::kGumbelEps
                    ? persistent_stochastic_policy::kGumbelEps
                    : (
                        uniform
                            > 1.0f - persistent_stochastic_policy::kGumbelEps
                        ? 1.0f - persistent_stochastic_policy::kGumbelEps
                        : uniform
                    );
                const float score = row_logits[action]
                    + -logf(-logf(uniform));
                if (score > best_score) {
                    best_score = score;
                    best_action = static_cast<int32_t>(action);
                }
            }
            action_cache_actions[cache_offset + cache_index] = best_action;
            if (cache_index == 0) {
                row_actions_out[row] = best_action;
                action_cache_next[slot] = best_action;
            }
        }
        action_cache_token[slot] = slot_tokens[slot];
        action_cache_cursor[slot] = 1;
        ++counters.refilled_rows;
    }

    counters.refill_calls = counters.refilled_rows > 0 ? 1 : 0;
    counters.full_refill_ticks = counters.refilled_rows > 0
            && counters.hit_rows == 0
        ? 1
        : 0;
    counters.partial_refill_ticks = counters.refilled_rows > 0
            && counters.hit_rows > 0
        ? 1
        : 0;
    counters.gumbel_draws = counters.refilled_rows
        * config.cache_width * config.action_count;
    pcg64_state_words[0] = rng.state_lo;
    pcg64_state_words[1] = rng.state_hi;
    pcg64_state_words[2] = rng.inc_lo;
    pcg64_state_words[3] = rng.inc_hi;
    pcg64_buffer_words[0] = static_cast<int64_t>(rng.has_uint32);
    pcg64_buffer_words[1] = static_cast<int64_t>(rng.uinteger);
    *counters_io = counters;
}
#endif

__device__ __forceinline__
eiel::genome_upgrade_v2_q10_v1::ResidentPlanesV1
genome_v2_resident_planes_v1(
    const FlatOrdinaryBirthState& state,
    const int64_t cell,
    const int64_t capacity
) {
    namespace genome_v2 = eiel::genome_upgrade_v2_q10_v1;
    return {
        state.genome_upgrade_v2_alleles_q10
            + cell * genome_v2::kHomologCount
                * genome_v2::kProductCount
                * genome_v2::kParameterCount
                * capacity,
        state.genome_upgrade_v2_modifiers_q10
            + cell * genome_v2::kHomologCount
                * genome_v2::kProductCount
                * capacity,
        state.genome_upgrade_v2_products_q10
            + cell * genome_v2::kProductCount * capacity,
        state.genome_upgrade_v2_phenotypes_q10
            + cell * genome_v2::kPhenotypeCount * capacity,
        state.genome_upgrade_v2_expression_cost_q10 + cell * capacity,
        state.genome_upgrade_v2_active_product_count + cell * capacity,
        state.genome_upgrade_v2_saturated_product_count + cell * capacity,
        capacity,
    };
}

__global__ void flattened_controller_tick_core_and_birth_plan_kernel_v1(
    const Phase1KernelArguments arguments,
    const int64_t controller_tick
) {
#define active_agent_ids arguments.active_agent_ids
#define active_agent_slots arguments.active_agent_slots
#define active_count_state arguments.active_count_state
#define slot_to_agent_id arguments.slot_to_agent_id
#define agent_id_to_slot arguments.agent_id_to_slot
#define alive_mask arguments.alive_mask
#define slot_tokens arguments.slot_tokens
#define action_cache_actions arguments.action_cache_actions
#define action_cache_token arguments.action_cache_token
#define action_cache_cursor arguments.action_cache_cursor
#define action_cache_next arguments.action_cache_next
#define pos_y arguments.pos_y
#define pos_x arguments.pos_x
#define agent_grid arguments.agent_grid
#define food_grid arguments.food_grid
#define energy arguments.energy
#define health arguments.health
#define injury arguments.injury
#define policy_energy_observation arguments.policy_energy_observation
#define rest_streak arguments.rest_streak
#define pending_free_agent_ids arguments.pending_free_agent_ids
#define pending_free_slots arguments.pending_free_slots
#define pending_free_count arguments.pending_free_count
#define energy_collected arguments.energy_collected
#define energy_collected_exact arguments.energy_collected_exact
#define last_action arguments.last_action
#define last_last_action arguments.last_last_action
#define action_switches arguments.action_switches
#define action_oscillations arguments.action_oscillations
#define episode0_switches arguments.episode0_switches
#define episode0_oscillations arguments.episode0_oscillations
#define survival_steps arguments.survival_steps
#define mlp_pack arguments.mlp_pack
#define drives arguments.drives
#define attack_bias arguments.attack_bias
#define action_temperature arguments.action_temperature
#define drive_logit_gain arguments.drive_logit_gain
#define food_gained_scratch arguments.food_gained_scratch
#define ordered_agent_ids_out arguments.ordered_agent_ids_out
#define ordered_slots_out arguments.ordered_slots_out
#define observations_out arguments.observations_out
#define logits_out arguments.logits_out
#define row_actions_out arguments.row_actions_out
#define actions_by_slot_out arguments.actions_by_slot_out
#define pre_energy_out arguments.pre_energy_out
#define pre_health_out arguments.pre_health_out
#define pre_injury_out arguments.pre_injury_out
#define pre_survival_steps_out arguments.pre_survival_steps_out
#define active_count_out arguments.active_count_out
#define policy_refill_rows_out arguments.policy_refill_rows_out
#define policy_cache_hit_rows_out arguments.policy_cache_hit_rows_out
#define runtime_bookkeeping_by_tick_out arguments.runtime_bookkeeping_by_tick_out
#define death_agent_ids_out arguments.death_agent_ids_out
#define death_slots_out arguments.death_slots_out
#define death_reason_codes_out arguments.death_reason_codes_out
#define death_count_out arguments.death_count_out
#define survivor_agent_ids_out arguments.survivor_agent_ids_out
#define survivor_count_out arguments.survivor_count_out
#define lifecycle_status_out arguments.lifecycle_status_out
#define status_out arguments.status_out
#define first_failure_records arguments.first_failure_records
#define generation_summary_scalar_out arguments.generation_summary_scalar_out
#define full_tick_state arguments.full_tick_state
#define full_tick_outputs arguments.full_tick_outputs
#define event_state arguments.event_state
#define ordinary_birth_state arguments.ordinary_birth_state
#define ordinary_birth_outputs arguments.ordinary_birth_outputs
#define ordinary_birth_scratch arguments.ordinary_birth_scratch
#define ordinary_birth_config arguments.ordinary_birth_config
#define mechanics_config arguments.mechanics_config
#define patch_p2_context arguments.patch_p2_context
#define compact_summary_context arguments.compact_summary_context
#define ticks arguments.ticks
#define active_capacity arguments.active_capacity
#define agent_id_capacity arguments.agent_id_capacity
#define pending_capacity arguments.pending_capacity
#define height arguments.height
#define width arguments.width
#define view_radius arguments.view_radius
#define observation_size arguments.observation_size
#define hidden_size arguments.hidden_size
#define action_count arguments.action_count
#define packed_width arguments.packed_width
#define drive_width arguments.drive_width
#define cache_capacity arguments.cache_capacity
#define cache_width arguments.cache_width
#define deterministic_actions arguments.deterministic_actions
#define max_food arguments.max_food
#define effective_food_spawn_count arguments.effective_food_spawn_count
#define max_energy arguments.max_energy
#define start_health arguments.start_health
#define move_cost arguments.move_cost_by_cell[cell]
#define gather_cost arguments.gather_cost
#define rest_cost arguments.rest_cost
#define attack_cost arguments.attack_cost
#define action_cost_mult arguments.action_cost_mult
#define rest_cost_mult arguments.rest_cost_mult
#define food_energy arguments.food_energy
#define episode0_k arguments.episode0_k
#define w_hunger_gather arguments.w_hunger_gather
#define w_pain_rest arguments.w_pain_rest
#define w_fatigue_rest arguments.w_fatigue_rest
#define w_pain_attack arguments.w_pain_attack
#define w_fear_attack arguments.w_fear_attack
#define RECORD_TICK_CORE_STATUS(code) record_status( \
    status_out, \
    first_failure_records, \
    cell, \
    mechanics_config.generation, \
    mechanics_config.start_tick + controller_tick, \
    kControllerFailureStageTickCoreAndBirthPlan, \
    code \
)
    const int64_t capacity = arguments.capacity;
    const int64_t cell = static_cast<int64_t>(blockIdx.x);
    const int64_t thread = static_cast<int64_t>(threadIdx.x);
    const int64_t side = 2 * view_radius + 1;
    const int64_t local_size = side * side;
    const bool full_patch_p2 = patch_p2_context.enabled
        && patch_p2_context.mode == FlatPatchP2Mode::kFullPatchP2;
    const int64_t expected_observation_size = full_patch_p2
        ? kPatchObservationChannels * local_size + 2
        : local_size + 2;
    const int64_t cell_capacity_offset = cell * capacity;
    const int64_t cell_grid_offset = cell * height * width;
    const int64_t cell_id_map_offset = cell * agent_id_capacity;
    const int64_t cell_active_offset = cell * active_capacity;
    const int64_t cell_cache_offset = cell_capacity_offset * cache_capacity;
    const int64_t cell_pending_offset = cell * pending_capacity;
    const int64_t output_tick = arguments.output_ticks_per_cell == 1
        ? 0
        : controller_tick;
    const int64_t cell_tick =
        cell * arguments.output_ticks_per_cell + output_tick;
    const int64_t resident_cell_tick =
        cell * arguments.resident_ticks_per_cell + controller_tick;
    const int64_t runtime_bookkeeping_offset = resident_cell_tick * 5;
    const int64_t tick = controller_tick;
    int64_t pre_action_rows = 0;
    __shared__ int64_t current_active_rows;
    __shared__ double patch_tick_metrics[kPatchCompactMetricCount];
    __shared__ persistent_stochastic_policy::PolicyCacheCounters
        patch_policy_counters;
    __shared__ bool attempt_capacity_qualification_faulted;
    __shared__ int32_t genome_v2_expression_status;

    if (
        (
            cross_path_resident_control_v1::has_cross_path_authority_v1(
                ordinary_birth_config.cross_path_control
            )
            && !cross_path_resident_control_v1::
                has_complete_ordinary_authority_v1(
                    ordinary_birth_config.cross_path_control
                )
        )
        ||
        !cross_path_resident_control_v1::valid_reproduction_path_v1(
            ordinary_birth_config.cross_path_control, cell
        )
    ) {
        if (thread == 0) {
            RECORD_TICK_CORE_STATUS(240);
        }
        return;
    }
    if (
        cross_path_resident_control_v1::has_cross_path_authority_v1(
            ordinary_birth_config.cross_path_control
        )
        && ordinary_birth_state.inheritance_mode != nullptr
        && ordinary_birth_state.inheritance_mode[cell] == 2
    ) {
        if (thread == 0) {
            RECORD_TICK_CORE_STATUS(241);
        }
        return;
    }
    const bool neutral_fixed_action =
        cross_path_resident_control_v1::has_cross_path_authority_v1(
            ordinary_birth_config.cross_path_control
        )
        && ordinary_birth_config.cross_path_control
            .neutral_fixture_enabled_by_cell[cell];
    if (thread == 0) {
        attempt_capacity_qualification_faulted =
            resident_reproduction_exposure_v1::
                apply_attempt_capacity_qualification_v1(
                    ordinary_birth_config.cross_path_control
                        .reproduction_exposure,
                    cell
                );
    }
    __syncthreads();
    if (attempt_capacity_qualification_faulted) {
        if (thread == 0) {
            const int64_t active_rows = active_count_state[cell];
            active_count_out[cell_tick] = active_rows;
            policy_refill_rows_out[cell_tick] = 0;
            policy_cache_hit_rows_out[cell_tick] = 0;
            runtime_bookkeeping_by_tick_out[runtime_bookkeeping_offset] =
                active_rows;
        }
        for (int64_t slot = thread; slot < capacity; slot += blockDim.x) {
            const int64_t source = cell_capacity_offset + slot;
            const int64_t target = cell_tick * capacity + slot;
            actions_by_slot_out[target] = kRestAction;
            pre_energy_out[target] = energy[source];
            pre_health_out[target] = health[source];
            pre_injury_out[target] = injury[source];
            pre_survival_steps_out[target] = survival_steps[source];
        }
        for (int64_t word = thread; word < 4; word += blockDim.x) {
            full_tick_outputs.policy_pcg64_state_after_action[
                cell_tick * 4 + word
            ] = full_tick_state.shared_env_policy_pcg64_state_words[
                cell * 4 + word
            ];
        }
        for (int64_t word = thread; word < 2; word += blockDim.x) {
            full_tick_outputs.policy_pcg64_buffer_after_action[
                cell_tick * 2 + word
            ] = full_tick_state.shared_env_policy_pcg64_buffer_words[
                cell * 2 + word
            ];
        }
        __syncthreads();
        return;
    }
    const bool genome_v2_requested =
        ordinary_birth_state.genome_upgrade_v2_enabled != nullptr
        && ordinary_birth_state.genome_upgrade_v2_enabled[cell];
    const bool genome_v2_mode =
        genome_v2_requested
        && ordinary_birth_state.genome_upgrade_v2_alleles_q10 != nullptr
        && ordinary_birth_state.genome_upgrade_v2_modifiers_q10 != nullptr
        && ordinary_birth_state.genome_upgrade_v2_products_q10 != nullptr
        && ordinary_birth_state.genome_upgrade_v2_phenotypes_q10 != nullptr
        && ordinary_birth_state.genome_upgrade_v2_expression_cost_q10 != nullptr
        && ordinary_birth_state.genome_upgrade_v2_active_product_count != nullptr
        && ordinary_birth_state.genome_upgrade_v2_saturated_product_count
            != nullptr;
    if (genome_v2_requested && (!genome_v2_mode || !full_patch_p2)) {
        if (thread == 0) {
            RECORD_TICK_CORE_STATUS(242);
        }
        return;
    }

    const bool exact_cpu_policy_prepare =
        arguments.execution_mode
        == TickCoreExecutionModeV1::kPrepareExactCpuPolicy;
    const bool exact_cpu_policy_resume =
        arguments.execution_mode
        == TickCoreExecutionModeV1::kResumeAfterExactCpuPolicy;
    if (exact_cpu_policy_resume) {
        if (thread == 0) {
            current_active_rows = active_count_out[cell_tick];
            for (
                int64_t metric = 0;
                metric < kPatchCompactMetricCount;
                ++metric
            ) {
                patch_tick_metrics[metric] = 0.0;
            }
        }
        __syncthreads();
        if (
            current_active_rows < 0
            || current_active_rows > active_capacity
        ) {
            if (thread == 0) {
                RECORD_TICK_CORE_STATUS(4);
            }
            return;
        }
        pre_action_rows = current_active_rows;
    } else {
    if (
        thread == 0
        && controller_tick == 0
        && compact_summary_context_addressable_v1(compact_summary_context, cell)
    ) {
        const auto access = compact_summary_reporting_access_v1(
            compact_summary_context, cell
        );
        const int64_t active_count = active_count_state[cell];
        int64_t absolute_event_cursor = 0;
        if (
            !compact_summary_context_matches_controller_v1(
                compact_summary_context, capacity
            )
            || ordinary_birth_state.population_next_id == nullptr
            || event_state.active_count == nullptr
            || !absolute_event_cursor_v1(
                event_state, cell, &absolute_event_cursor
            )
            || active_count < 0
            || active_count > compact_summary_context.reporting_row_capacity
            || agent_id_capacity != capacity
        ) {
            reporting::record_failure_v1(
                access.header, reporting::kLedgerInvalidStorageV1
            );
        } else {
            reporting::initialize_generation_from_active_slot_state_v1(
                access,
                ordinary_birth_state.population_next_id[cell],
                active_agent_ids + cell_active_offset,
                active_agent_slots + cell_active_offset,
                injury + cell_capacity_offset,
                static_cast<int32_t>(active_count)
            );
            const int64_t reporting_row = compact_summary_row_index_v1(
                compact_summary_context, cell
            );
            compact_summary_context.event_cursor_start[reporting_row] =
                absolute_event_cursor;
            compact_summary_context.event_cursor_pre_turnover[reporting_row] =
                absolute_event_cursor;
            compact_summary_context.event_cursor_end[reporting_row] =
                absolute_event_cursor;
        }
    }

    if (
        thread == 0
        && controller_tick == 0
        && generation_summary_scalar_out != nullptr
    ) {
        double* const summary = generation_summary_scalar_out
            + cell * kGenerationSummaryScalarWidthV1;
        for (
            int64_t column = 0;
            column < kGenerationSummaryScalarWidthV1;
            ++column
        ) {
            summary[column] = 0.0;
        }
        summary[kGenerationSummaryGeneration] =
            static_cast<double>(mechanics_config.generation);
        int64_t absolute_event_cursor = 0;
        if (absolute_event_cursor_v1(
                event_state, cell, &absolute_event_cursor
            )) {
            const double event_cursor = static_cast<double>(
                absolute_event_cursor
            );
            summary[kGenerationSummaryEventCursorStart] = event_cursor;
            summary[kGenerationSummaryEventCursorPreTurnover] = event_cursor;
            summary[kGenerationSummaryEventCursorEnd] = event_cursor;
        } else {
            fail_event_sequence_contract_v1(event_state, cell);
            summary[kGenerationSummaryEventCursorStart] = NAN;
            summary[kGenerationSummaryEventCursorPreTurnover] = NAN;
            summary[kGenerationSummaryEventCursorEnd] = NAN;
        }
    }
    __syncthreads();

    // ABI columns: applied_actions, moved, gathered, policy_refill_calls,
    // policy_refill_rows.
    if (thread == 0) {
        for (int64_t column = 0; column < 5; ++column) {
            runtime_bookkeeping_by_tick_out[
                runtime_bookkeeping_offset + column
            ] = 0;
        }
    }

    const auto runtime_birth_layout =
        flattened_birth_scratch_v2::make_layout(
            ordinary_birth_config.max_births_per_tick,
            capacity,
            packed_width,
            ordinary_birth_config.hmem_width,
            ordinary_birth_config.torch_state_bytes
        );
    const bool runtime_birth_layout_valid =
        ordinary_birth_config.max_births_per_tick <= 2
        || (
            ordinary_birth_scratch.storage != nullptr
            && ordinary_birth_scratch.bytes_per_cell
                >= kRuntimeBirthScratchPrefixBytesV2
            && runtime_birth_layout.ok()
            && runtime_birth_layout.required_bytes
                <= ordinary_birth_scratch.bytes_per_cell
                    - kRuntimeBirthScratchPrefixBytesV2
        );

    if (
        hidden_size > kMaxHiddenSize
        || observation_size > kMaxObservationSize
        || action_count > kMaxActionCount
        || observation_size != expected_observation_size
        || (full_patch_p2 && view_radius != 1)
        || (
            full_patch_p2
            && (
                observation_size != kPatchRadiusOneObservationSize
                || height > persistent_patch_p2::kMaximumExactSerialCells
                    / width
                || !patch_p2_dimensions_match(
                    patch_p2_context, height, width
                )
                || patch_p2_context.state.patch_food_amount == nullptr
                || patch_p2_context.state.patch_scent == nullptr
                || patch_p2_context.state.patch_waste == nullptr
                || patch_p2_context.state.patch_scent_emit_pending == nullptr
                || patch_p2_context.state.patch_waste_emit_pending == nullptr
                || patch_p2_context.state.patch_density == nullptr
                || patch_p2_context.patch_cover == nullptr
                || patch_p2_context.patch_shelter == nullptr
                || patch_p2_context.patch_movement_cost == nullptr
                || ordinary_birth_state.dev_stage_initialized == nullptr
            )
        )
        || drive_width < 4
        || cache_width <= 0
        || cache_width > cache_capacity
        || pending_capacity < capacity
        || full_tick_state.chemistry_width < 6
        || mechanics_config.start_tick < 0
        || mechanics_config.start_tick + ticks
            > full_tick_state.drive_noise_sample_ticks
        || (
            arguments.output_ticks_per_cell != 1
            && arguments.output_ticks_per_cell != ticks
        )
        || arguments.resident_ticks_per_cell < ticks
        || (
            patch_p2_context.enabled
            && patch_p2_context.stats_ticks_per_cell < ticks
        )
        || event_state.capacity < 0
        || event_state.reporting_rows < 0
        || (
            ordinary_birth_config.enabled
            && ordinary_birth_config.max_births_per_tick <= 0
        )
        || ordinary_birth_config.max_births_per_tick < 0
        || !runtime_birth_layout_valid
        || ordinary_birth_config.max_births_per_tick > capacity
        || (
            ordinary_birth_config.enabled
            && (
                ordinary_birth_config.gauss_tape_capacity <= 0
                || ordinary_birth_config.policy_tape_capacity <= 0
                || ordinary_birth_config.torch_state_bytes <= 0
                || ordinary_birth_config.hmem_width < 0
                || ordinary_birth_config.sensor_radius_max < 0
                || !::isfinite(ordinary_birth_config.parent_energy_share)
                || ordinary_birth_config.parent_energy_share < 0.0
                || ordinary_birth_config.parent_energy_share > 1.0
                || !::isfinite(ordinary_birth_config.minimum_energy_buffer)
                || ordinary_birth_config.minimum_energy_buffer < 0.0f
            )
        )
    ) {
        if (thread == 0) {
            RECORD_TICK_CORE_STATUS(1);
        }
        return;
    }

    if (thread == 0) {
        current_active_rows = active_count_state[cell];
        if (current_active_rows < 0 || current_active_rows > active_capacity) {
            RECORD_TICK_CORE_STATUS(4);
        }
    }
    __syncthreads();
    if (status_out[cell] != 0) {
        return;
    }

    // Stage 2 resets generation-local action cache once, not once per launch.
    if (controller_tick == 0) {
        for (int64_t slot = thread; slot < capacity; slot += blockDim.x) {
            const int64_t slot_index = cell_capacity_offset + slot;
            action_cache_token[slot_index] = -1;
            action_cache_cursor[slot_index] = -1;
            action_cache_next[slot_index] = -1;
            if (alive_mask[slot_index]) {
                action_switches[slot_index] = 0;
                action_oscillations[slot_index] = 0;
                full_tick_state.drive_volatility[slot_index] = 0.0f;
                full_tick_state.last_energy_delta[slot_index] = 0.0f;
                full_tick_state.last_health_delta[slot_index] = 0.0f;
                full_tick_state.last_injury_delta[slot_index] = 0.0f;
                if (ordinary_birth_state.pred_err_sum != nullptr) {
                    ordinary_birth_state.pred_err_sum[slot_index] = 0.0f;
                    ordinary_birth_state.pred_err_count[slot_index] = 0;
                }
                full_tick_state.selfharm_events[slot_index] = 0;
            }
        }
        for (
            int64_t index = thread;
            index < capacity * cache_capacity;
            index += blockDim.x
        ) {
            action_cache_actions[cell_cache_offset + index] = -1;
        }
    }
    __syncthreads();

    // One block owns one Flat tick. The native C++ controller owns the loop.
        pre_action_rows = current_active_rows;
        if (thread == 0) {
            active_count_out[cell_tick] = pre_action_rows;
            policy_refill_rows_out[cell_tick] = 0;
            policy_cache_hit_rows_out[cell_tick] = 0;
            runtime_bookkeeping_by_tick_out[runtime_bookkeeping_offset] =
                pre_action_rows;
            full_tick_outputs.runner_shuffle_draws[cell_tick] = 0;
            full_tick_outputs.runner_shuffle_status[cell_tick] = 0;
            full_tick_outputs.drive_noise_draws[cell_tick] = 0;
            full_tick_outputs.drive_noise_raw64_draws[cell_tick] = 0;
            full_tick_outputs.drive_noise_status[cell_tick] = 0;
            full_tick_outputs.chemistry_draws[cell_tick] = 0;
            full_tick_outputs.chemistry_status[cell_tick] = 0;
            full_tick_outputs.active_mechanics_status[cell_tick] = 0;
            full_tick_outputs.residual_death_count[cell_tick] = 0;
            full_tick_outputs.residual_release_count[cell_tick] = 0;
            full_tick_outputs.residual_release_status[cell_tick] = 0;
            for (int64_t stat = 0; stat < 3; ++stat) {
                full_tick_outputs.food_spawn_stats[cell_tick * 3 + stat] = 0;
            }
            for (int64_t count = 0; count < 5; ++count) {
                full_tick_outputs.active_mechanics_counts[
                    cell_tick * 5 + count
                ] = 0;
            }
            for (
                int64_t metric = 0;
                metric < kPatchCompactMetricCount;
                ++metric
            ) {
                patch_tick_metrics[metric] = 0.0;
            }
        }
        for (int64_t slot = thread; slot < capacity; slot += blockDim.x) {
            actions_by_slot_out[cell_tick * capacity + slot] = kRestAction;
            food_gained_scratch[cell_capacity_offset + slot] = 0.0f;
            full_tick_outputs.chemistry_order_slots[
                cell_tick * capacity + slot
            ] = -1;
            full_tick_outputs.residual_death_agent_ids[
                cell_tick * capacity + slot
            ] = -1;
            full_tick_outputs.residual_death_slots[
                cell_tick * capacity + slot
            ] = -1;
            full_tick_outputs.residual_death_reason_codes[
                cell_tick * capacity + slot
            ] = 0;
            full_tick_state.residual_pending_free_agent_ids[
                cell_capacity_offset + slot
            ] = -1;
            full_tick_state.residual_pending_free_slots[
                cell_capacity_offset + slot
            ] = -1;
        }
        if (thread == 0) {
            full_tick_state.residual_pending_free_count[cell] = 0;
        }
        __syncthreads();

        if (patch_p2_context.enabled) {
            const int64_t patch_p2_scratch_offset =
                cell * 4 * height * width;
            const int64_t patch_stats_tick =
                cell * patch_p2_context.stats_ticks_per_cell
                + controller_tick;
            const bool cooperative_patch_p2 =
                full_patch_p2
                && height == 16
                && width == 16
                && patch_p2_dimensions_match(
                    patch_p2_context,
                    height,
                    width
                )
                && blockDim.x == kThreadsPerCell;
            if (cooperative_patch_p2) {
                patch_p2_single_tick_exact_block_256_from_flat_controller(
                    patch_p2_context,
                    cell,
                    cell_grid_offset,
                    patch_p2_scratch_offset,
                    patch_stats_tick
                );
            } else if (thread == 0) {
                patch_p2_single_tick_exact_from_flat_controller(
                    patch_p2_context,
                    cell,
                    cell_grid_offset,
                    patch_p2_scratch_offset,
                    patch_stats_tick
                );
            }
            if (
                thread == 0
                && full_patch_p2
                && patch_p2_context.compact_metrics != nullptr
            ) {
                double* const metrics = patch_p2_context.compact_metrics
                    + cell * kPatchCompactMetricCount;
                const double* const stats = patch_p2_context.outputs.stats
                    + patch_stats_tick * persistent_patch_p2::kStatCount;
                metrics[kPatchCompactScentEmittedTotal] +=
                    stats[persistent_patch_p2::kScentEmittedTotal];
                metrics[kPatchCompactWasteEmittedTotal] +=
                    stats[persistent_patch_p2::kWasteEmittedTotal];
                metrics[kPatchCompactFoodDepletedFrac] =
                    stats[persistent_patch_p2::kFoodDepletedFrac];
                metrics[kPatchCompactFoodFullFrac] =
                    stats[persistent_patch_p2::kFoodFullFrac];
                metrics[kPatchCompactDensityHotspotFrac] =
                    stats[persistent_patch_p2::kDensityHotspotFrac];
            }
        }
        __syncthreads();

        if (thread == 0) {
            using namespace persistent_food_spawn;
            spawn_flat_food_exact(
                {
                    food_grid + cell_grid_offset,
                    nullptr,
                },
                {
                    full_tick_state.shared_env_policy_pcg64_state_words + cell * 4,
                    full_tick_state.shared_env_policy_pcg64_buffer_words + cell * 2,
                    full_tick_state.policy_np_semantic_draws + cell,
                },
                {
                    full_tick_outputs.food_spawn_stats + cell_tick * 3,
                },
                {
                    height * width,
                    max_food,
                    effective_food_spawn_count,
                    false,
                    0.0f,
                    0.0f,
                }
            );
            if (full_patch_p2) {
                const int64_t patch_sync_rows = sync_patch_spawned_food_exact(
                    food_grid + cell_grid_offset,
                    patch_p2_context.state.patch_food_amount
                        + cell_grid_offset,
                    height * width,
                    patch_p2_context.config.food_presence_threshold,
                    static_cast<float>(
                        patch_p2_context.patch_food_initial_value
                    )
                );
                full_tick_outputs.food_spawn_stats[
                    cell_tick * 3
                        + persistent_food_spawn::kPatchSyncRows
                ] = patch_sync_rows;
                if (patch_p2_context.compact_metrics != nullptr) {
                    patch_p2_context.compact_metrics[
                        cell * kPatchCompactMetricCount
                            + kPatchCompactSpawnSyncRows
                    ] += static_cast<double>(patch_sync_rows);
                }
            }
        }
        __syncthreads();

        // Validate in ascending row order so distinct invalid-row statuses
        // cannot race through the first-failure latch.
        if (thread == 0) {
            for (int64_t row = 0; row < pre_action_rows; ++row) {
                const int64_t agent_id = active_agent_ids[
                    cell_active_offset + row
                ];
                if (!active_identity::agent_id_fits_grid_v1(agent_id)) {
                    RECORD_TICK_CORE_STATUS(2);
                    break;
                }
                const int32_t slot =
                    active_agent_slots[cell_active_offset + row];
                if (
                    slot < 0 || static_cast<int64_t>(slot) >= capacity
                    || !alive_mask[cell_capacity_offset + slot]
                    || slot_to_agent_id[cell_capacity_offset + slot] != agent_id
                ) {
                    RECORD_TICK_CORE_STATUS(3);
                    break;
                }
            }
        }
        __syncthreads();
        if (status_out[cell] != 0) {
            return;
        }

        // The Patch CPU owner initializes developmental state, then consumes
        // the runner shuffle before policy service and observation/action work.
        if (thread == 0 && full_patch_p2) {
            int32_t* patch_order = full_tick_outputs.chemistry_order_slots
                + cell_tick * capacity;
            for (int64_t row = 0; row < pre_action_rows; ++row) {
                const int32_t slot =
                    active_agent_slots[cell_active_offset + row];
                patch_order[row] = slot;
                if (!ordinary_birth_state.dev_stage_initialized[
                    cell_capacity_offset + slot
                ]) {
                    ordinary_birth_state.dev_stage_initialized[
                        cell_capacity_offset + slot
                    ] = true;
                }
            }
            if (status_out[cell] == 0) {
                persistent_chemistry::runner_shuffle_counted(
                    patch_order,
                    pre_action_rows,
                    full_tick_state.runner_pcg64_state_words + cell * 4,
                    full_tick_state.runner_pcg64_buffer_words + cell * 2,
                    full_tick_outputs.runner_shuffle_draws + cell_tick,
                    full_tick_outputs.runner_shuffle_status + cell_tick,
                    capacity
                );
                if (full_tick_outputs.runner_shuffle_status[cell_tick] == 0) {
                    full_tick_state.metrics_np_semantic_draws[cell] +=
                        full_tick_outputs.runner_shuffle_draws[cell_tick];
                } else {
                    RECORD_TICK_CORE_STATUS(20);
                }
            }
        }
        __syncthreads();
        if (status_out[cell] != 0) {
            return;
        }

        if (thread == 0) {
            genome_v2_expression_status = 0;
        }
        __syncthreads();
        if (genome_v2_mode) {
            namespace genome_v2 = eiel::genome_upgrade_v2_q10_v1;
            const genome_v2::ResidentPlanesV1 planes =
                genome_v2_resident_planes_v1(
                    ordinary_birth_state, cell, capacity
                );
            float density_denom = static_cast<float>(
                (2 * patch_p2_context.config.density_radius + 1)
                * (2 * patch_p2_context.config.density_radius + 1)
            );
            if (!patch_p2_context.config.density_include_self) {
                density_denom -= 1.0f;
            }
            if (density_denom <= 0.0f) {
                density_denom = 1.0f;
            }
            for (
                int64_t row = thread;
                row < pre_action_rows;
                row += blockDim.x
            ) {
                const int32_t slot =
                    active_agent_slots[cell_active_offset + row];
                if (slot < 0 || static_cast<int64_t>(slot) >= capacity) {
                    atomicExch(&genome_v2_expression_status, 1);
                    continue;
                }
                const int64_t slot_index = cell_capacity_offset + slot;
                const int64_t grid_index = cell_grid_offset
                    + static_cast<int64_t>(pos_y[slot_index]) * width
                    + pos_x[slot_index];
                const double density =
                    patch_p2_context.config.density_normalize
                    ? patch_p2_context.state.patch_density[grid_index]
                    : patch_p2_context.state.patch_density[grid_index]
                        / density_denom;
                const genome_v2::RuntimeContextInputsV1 context_inputs{
                    static_cast<double>(energy[slot_index]),
                    max_energy,
                    static_cast<double>(injury[slot_index]),
                    start_health,
                    density,
                    static_cast<double>(
                        patch_p2_context.state.patch_food_amount[grid_index]
                    ) / patch_p2_context.config.food_max,
                    static_cast<double>(
                        patch_p2_context.state.patch_fertility[grid_index]
                    ),
                    static_cast<double>(
                        patch_p2_context.state.patch_waste[grid_index]
                    ) / patch_p2_context.config.waste_max,
                    full_tick_state.age[slot_index],
                    mechanics_config.developmental_maturity_age,
                };
                int64_t context_q10[genome_v2::kContextCount]{};
                if (
                    !genome_v2::build_runtime_context_q10_v1(
                        context_inputs, context_q10
                    )
                    || !genome_v2::express_resident_slot_v1(
                        planes, slot, context_q10
                    )
                ) {
                    atomicExch(&genome_v2_expression_status, 1);
                    continue;
                }
                action_cache_token[slot_index] = -1;
                action_cache_cursor[slot_index] = -1;
                action_cache_next[slot_index] = -1;
                const genome_v2::RuntimeProjectionV1 runtime =
                    genome_v2::load_runtime_projection_v1(planes, slot);
                double reproduction_threshold = 0.0;
                if (
                    !genome_v2::reproduction_threshold_v1(
                        ordinary_birth_state
                            .genome_reproduction_threshold[slot_index],
                        runtime,
                        &reproduction_threshold
                    )
                ) {
                    atomicExch(&genome_v2_expression_status, 1);
                    continue;
                }
                ordinary_birth_state.repro_threshold[slot_index] =
                    static_cast<float>(reproduction_threshold);
                const int64_t admitted_memory = runtime.memory_dimension
                    < ordinary_birth_config.hmem_width
                    ? runtime.memory_dimension
                    : ordinary_birth_config.hmem_width;
                const int64_t hereditary_memory =
                    ordinary_birth_state.genome_memory_dim[slot_index];
                ordinary_birth_state.policy_mem_dim[slot_index] =
                    hereditary_memory < admitted_memory
                    ? hereditary_memory
                    : admitted_memory;
                attack_bias[slot_index] =
                    ordinary_birth_state.genome_aggression_bias[slot_index]
                    + static_cast<float>(runtime.threat_response_q10)
                        / genome_v2::kScale;
            }
        }
        __syncthreads();
        if (genome_v2_expression_status != 0) {
            if (thread == 0) {
                RECORD_TICK_CORE_STATUS(243);
            }
            return;
        }

        for (int64_t row = thread; row < pre_action_rows; row += blockDim.x) {
            const int64_t agent_id =
                active_agent_ids[cell_active_offset + row];
            const int32_t slot =
                active_agent_slots[cell_active_offset + row];

            const int64_t row_out = cell_tick * capacity + row;
            const int64_t slot_index = cell_capacity_offset + slot;
            ordered_agent_ids_out[row_out] = agent_id;
            ordered_slots_out[row_out] = slot;

            if (neutral_fixed_action) {
                row_actions_out[row_out] = kRestAction;
                actions_by_slot_out[cell_tick * capacity + slot] =
                    kRestAction;
                continue;
            }

            const int32_t cursor = action_cache_cursor[slot_index];
            const bool cache_hit = action_cache_token[slot_index] == slot_tokens[slot_index]
                && cursor >= 0
                && static_cast<int64_t>(cursor) < cache_width;
            if (cache_hit) {
                if (deterministic_actions || neutral_fixed_action) {
                    int32_t action = action_cache_actions[
                        cell_cache_offset
                        + static_cast<int64_t>(slot) * cache_capacity
                        + cursor
                    ];
                    if (action < 0 || action > kAttackAction) {
                        action = action_cache_next[slot_index];
                    }
                    if (action < 0 || action > kAttackAction) {
                        action = kRestAction;
                    }
                    action_cache_next[slot_index] = action;
                    action_cache_cursor[slot_index] = cursor + 1;
                    row_actions_out[row_out] = action;
                    actions_by_slot_out[cell_tick * capacity + slot] = action;
                    atomicAdd(
                        reinterpret_cast<unsigned long long*>(
                            policy_cache_hit_rows_out + cell_tick
                        ),
                        1ULL
                    );
                }
                continue;
            }

            const int32_t cy = pos_y[slot_index];
            const int32_t cx = pos_x[slot_index];
            float* observation = observations_out + row_out * observation_size;
            if (full_patch_p2) {
                write_patch_radius_one_observation_exact(
                    patch_p2_context,
                    agent_grid,
                    food_grid,
                    energy,
                    health,
                    cell_grid_offset,
                    slot_index,
                    cy,
                    cx,
                    height,
                    width,
                    static_cast<float>(max_energy),
                    static_cast<float>(start_health),
                    observation
                );
            } else {
                int64_t observation_offset = 0;
                for (int64_t dy = -view_radius; dy <= view_radius; ++dy) {
                    for (int64_t dx = -view_radius; dx <= view_radius; ++dx) {
                        const int64_t y = static_cast<int64_t>(cy) + dy;
                        const int64_t x = static_cast<int64_t>(cx) + dx;
                        float value = 0.0f;
                        if (y >= 0 && y < height && x >= 0 && x < width) {
                            const int64_t grid_index = cell_grid_offset + y * width + x;
                            if (agent_grid[grid_index] != -1) {
                                value = -1.0f;
                            } else if (food_grid[grid_index] == kFood) {
                                value = 1.0f;
                            }
                        }
                        observation[observation_offset++] = value;
                    }
                }
                observation[local_size] = energy[slot_index]
                    / static_cast<float>(max_energy);
                observation[local_size + 1] = health[slot_index]
                    / static_cast<float>(start_health);
            }
            if (genome_v2_mode) {
                namespace genome_v2 = eiel::genome_upgrade_v2_q10_v1;
                const auto planes = genome_v2_resident_planes_v1(
                    ordinary_birth_state, cell, capacity
                );
                const auto runtime = genome_v2::load_runtime_projection_v1(
                    planes, slot
                );
                const float fidelity =
                    static_cast<float>(runtime.sensory_fidelity_factor_q10)
                    / genome_v2::kScale;
                for (
                    int64_t feature = 0;
                    feature < observation_size - 2;
                    ++feature
                ) {
                    observation[feature] = fmaxf(
                        -1.0f,
                        fminf(1.0f, observation[feature] * fidelity)
                    );
                }
            }

            // Stochastic cache service must consume Gumbel draws serially in
            // compact miss order. Thread zero performs its MLP/refill after
            // every missing observation has been materialized.
            if (!deterministic_actions && !neutral_fixed_action) {
                continue;
            }

            const float* pack = mlp_pack + slot_index * packed_width;
            float h1[kMaxHiddenSize];
            float h2[kMaxHiddenSize];
            int64_t pack_offset = 0;
            for (int64_t hidden = 0; hidden < hidden_size; ++hidden) {
                float value = pack[hidden_size * observation_size + hidden];
                const float* weights = pack + hidden * observation_size;
                for (int64_t feature = 0; feature < observation_size; ++feature) {
                    value += weights[feature] * observation[feature];
                }
                h1[hidden] = tanhf(value);
            }
            pack_offset += hidden_size * observation_size + hidden_size;
            for (int64_t hidden = 0; hidden < hidden_size; ++hidden) {
                float value = pack[pack_offset + hidden_size * hidden_size + hidden];
                const float* weights = pack + pack_offset + hidden * hidden_size;
                for (int64_t input = 0; input < hidden_size; ++input) {
                    value += weights[input] * h1[input];
                }
                h2[hidden] = tanhf(value);
            }
            pack_offset += hidden_size * hidden_size + hidden_size;

            const float temperature = action_temperature[slot_index] < 0.05f
                ? 0.05f
                : action_temperature[slot_index];
            const float gain = drive_logit_gain[slot_index] < 0.0f
                ? 0.0f
                : drive_logit_gain[slot_index];
            const float* slot_drives = drives + slot_index * drive_width;
            int32_t best_action = 0;
            float best_score = 0.0f;
            for (int64_t action = 0; action < action_count; ++action) {
                const float* weights = pack + pack_offset + action * hidden_size;
                float score = pack[
                    pack_offset + action_count * hidden_size + action
                ];
                for (int64_t hidden = 0; hidden < hidden_size; ++hidden) {
                    score += weights[hidden] * h2[hidden];
                }
                if (action == kAttackAction) {
                    score += attack_bias[slot_index];
                }
                score /= temperature;
                if (action == kGatherAction) {
                    score += gain * (w_hunger_gather * slot_drives[0]);
                } else if (action == kRestAction) {
                    score += gain * (
                        w_pain_rest * slot_drives[1]
                        + w_fatigue_rest * slot_drives[2]
                    );
                } else if (action == kAttackAction) {
                    score -= gain * (
                        w_pain_attack * slot_drives[1]
                        + w_fear_attack * slot_drives[3]
                    );
                }
                logits_out[row_out * action_count + action] = score;
                if (action == 0 || score > best_score) {
                    best_score = score;
                    best_action = static_cast<int32_t>(action);
                }
            }

            const int64_t cache_offset = cell_cache_offset
                + static_cast<int64_t>(slot) * cache_capacity;
            for (int64_t cache_index = 0; cache_index < cache_width; ++cache_index) {
                action_cache_actions[cache_offset + cache_index] = best_action;
            }
            action_cache_token[slot_index] = slot_tokens[slot_index];
            action_cache_cursor[slot_index] = 1;
            action_cache_next[slot_index] = best_action;
            row_actions_out[row_out] = best_action;
            actions_by_slot_out[cell_tick * capacity + slot] = best_action;
            atomicAdd(
                reinterpret_cast<unsigned long long*>(policy_refill_rows_out + cell_tick),
                1ULL
            );
        }
        __syncthreads();

#if EIEL_CUDA_STAGE3_PARALLEL_PATCH_POLICY_ROWS
        if (
            !deterministic_actions
            && !neutral_fixed_action
            && status_out[cell] == 0
            && full_patch_p2
        ) {
            using namespace persistent_stochastic_policy;
            const int64_t tick_capacity_offset = cell_tick * capacity;
            const FixedMlpPolicyConfig policy_config{
                pre_action_rows,
                capacity,
                observation_size,
                hidden_size,
                action_count,
                packed_width,
                drive_width,
                cache_capacity,
                cache_width,
                w_hunger_gather,
                w_pain_rest,
                w_fatigue_rest,
                w_pain_attack,
                w_fear_attack,
                false,
            };
            if (thread == 0) {
                classify_patch_policy_rows_serial(
                    ordered_agent_ids_out + tick_capacity_offset,
                    ordered_slots_out + tick_capacity_offset,
                    slot_tokens + cell_capacity_offset,
                    action_cache_actions + cell_cache_offset,
                    action_cache_token + cell_capacity_offset,
                    action_cache_cursor + cell_capacity_offset,
                    action_cache_next + cell_capacity_offset,
                    row_actions_out + tick_capacity_offset,
                    death_slots_out + tick_capacity_offset,
                    &patch_policy_counters,
                    policy_config
                );
            }
            __syncthreads();

            if (!exact_cpu_policy_prepare) {
                // Each thread owns complete rows; only row-level concurrency
                // changes.
                for (
                    int64_t miss_index = thread;
                    miss_index < patch_policy_counters.missed_rows;
                    miss_index += blockDim.x
                ) {
                    const int64_t row = death_slots_out[
                        tick_capacity_offset + miss_index
                    ];
                    compute_patch_policy_row_logits_exact(
                        row,
                        ordered_slots_out + tick_capacity_offset,
                        observations_out
                            + tick_capacity_offset * observation_size,
                        mlp_pack + cell_capacity_offset * packed_width,
                        drives + cell_capacity_offset * drive_width,
                        attack_bias + cell_capacity_offset,
                        action_temperature + cell_capacity_offset,
                        drive_logit_gain + cell_capacity_offset,
                        logits_out + tick_capacity_offset * action_count,
                        policy_config
                    );
                }
            }
            __syncthreads();

            if (thread == 0 && !exact_cpu_policy_prepare) {
                consume_patch_policy_rows_serial(
                    ordered_slots_out + tick_capacity_offset,
                    slot_tokens + cell_capacity_offset,
                    action_cache_actions + cell_cache_offset,
                    action_cache_token + cell_capacity_offset,
                    action_cache_cursor + cell_capacity_offset,
                    action_cache_next + cell_capacity_offset,
                    logits_out + tick_capacity_offset * action_count,
                    row_actions_out + tick_capacity_offset,
                    death_slots_out + tick_capacity_offset,
                    full_tick_state.shared_env_policy_pcg64_state_words
                        + cell * 4,
                    full_tick_state.shared_env_policy_pcg64_buffer_words
                        + cell * 2,
                    &patch_policy_counters,
                    policy_config
                );
                const PolicyCacheCounters counters = patch_policy_counters;
                if (
                    counters.requested_rows != pre_action_rows
                    || counters.missed_rows != counters.refilled_rows
                ) {
                    RECORD_TICK_CORE_STATUS(33);
                } else {
                    policy_refill_rows_out[cell_tick] = counters.refilled_rows;
                    policy_cache_hit_rows_out[cell_tick] = counters.hit_rows;
                    full_tick_state.policy_np_semantic_draws[cell] +=
                        counters.gumbel_draws;
                    for (int64_t row = 0; row < pre_action_rows; ++row) {
                        const int32_t slot = ordered_slots_out[
                            tick_capacity_offset + row
                        ];
                        actions_by_slot_out[tick_capacity_offset + slot] =
                            row_actions_out[tick_capacity_offset + row];
                    }
                }
            }
            if (thread == 0 && exact_cpu_policy_prepare) {
                policy_refill_rows_out[cell_tick] =
                    patch_policy_counters.missed_rows;
                policy_cache_hit_rows_out[cell_tick] =
                    patch_policy_counters.hit_rows;
            }
        }
#endif
        __syncthreads();
#if !EIEL_CUDA_STAGE3_PARALLEL_PATCH_POLICY_ROWS
        if (
            !deterministic_actions
            && !neutral_fixed_action
            && status_out[cell] == 0
            && full_patch_p2
            && exact_cpu_policy_prepare
        ) {
            using namespace persistent_stochastic_policy;
            const int64_t tick_capacity_offset = cell_tick * capacity;
            const FixedMlpPolicyConfig policy_config{
                pre_action_rows,
                capacity,
                observation_size,
                hidden_size,
                action_count,
                packed_width,
                drive_width,
                cache_capacity,
                cache_width,
                w_hunger_gather,
                w_pain_rest,
                w_fatigue_rest,
                w_pain_attack,
                w_fear_attack,
                false,
            };
            if (thread == 0) {
                classify_patch_policy_rows_serial(
                    ordered_agent_ids_out + tick_capacity_offset,
                    ordered_slots_out + tick_capacity_offset,
                    slot_tokens + cell_capacity_offset,
                    action_cache_actions + cell_cache_offset,
                    action_cache_token + cell_capacity_offset,
                    action_cache_cursor + cell_capacity_offset,
                    action_cache_next + cell_capacity_offset,
                    row_actions_out + tick_capacity_offset,
                    death_slots_out + tick_capacity_offset,
                    &patch_policy_counters,
                    policy_config
                );
                policy_refill_rows_out[cell_tick] =
                    patch_policy_counters.missed_rows;
                policy_cache_hit_rows_out[cell_tick] =
                    patch_policy_counters.hit_rows;
            }
        }
        __syncthreads();
#endif
        if (exact_cpu_policy_prepare) {
            return;
        }

        if (
            !deterministic_actions
            && !neutral_fixed_action
            && status_out[cell] == 0
            && thread == 0
#if EIEL_CUDA_STAGE3_PARALLEL_PATCH_POLICY_ROWS
            && !full_patch_p2
#endif
        ) {
            using namespace persistent_stochastic_policy;
            PolicyCacheCounters counters{};
            const int64_t tick_capacity_offset = cell_tick * capacity;
            const FixedMlpPolicyConfig policy_config{
                pre_action_rows,
                capacity,
                observation_size,
                hidden_size,
                action_count,
                packed_width,
                drive_width,
                cache_capacity,
                cache_width,
                w_hunger_gather,
                w_pain_rest,
                w_fatigue_rest,
                w_pain_attack,
                w_fear_attack,
                false,
            };
#define EIEL_PATCH_POLICY_ARGUMENTS                                         \
                ordered_agent_ids_out + tick_capacity_offset,              \
                ordered_slots_out + tick_capacity_offset,                  \
                observations_out + tick_capacity_offset * observation_size,\
                mlp_pack + cell_capacity_offset * packed_width,            \
                drives + cell_capacity_offset * drive_width,               \
                attack_bias + cell_capacity_offset,                        \
                action_temperature + cell_capacity_offset,                 \
                drive_logit_gain + cell_capacity_offset,                   \
                slot_tokens + cell_capacity_offset,                        \
                action_cache_actions + cell_cache_offset,                  \
                action_cache_token + cell_capacity_offset,                 \
                action_cache_cursor + cell_capacity_offset,                \
                action_cache_next + cell_capacity_offset,                  \
                logits_out + tick_capacity_offset * action_count,          \
                row_actions_out + tick_capacity_offset,                    \
                death_slots_out + tick_capacity_offset,                    \
                full_tick_state.shared_env_policy_pcg64_state_words        \
                    + cell * 4,                                            \
                full_tick_state.shared_env_policy_pcg64_buffer_words       \
                    + cell * 2,                                            \
                &counters,                                                 \
                policy_config
#if EIEL_CUDA_STAGE3_PARALLEL_PATCH_POLICY_ROWS
            service_fixed_mlp_gumbel_policy_serial(
                EIEL_PATCH_POLICY_ARGUMENTS
            );
#else
            if (full_patch_p2) {
                service_patch_mlp_gumbel_policy_serial(
                    EIEL_PATCH_POLICY_ARGUMENTS
                );
            } else {
                service_fixed_mlp_gumbel_policy_serial(
                    EIEL_PATCH_POLICY_ARGUMENTS
                );
            }
#endif
#undef EIEL_PATCH_POLICY_ARGUMENTS
            if (
                counters.requested_rows != pre_action_rows
                || counters.missed_rows != counters.refilled_rows
            ) {
                RECORD_TICK_CORE_STATUS(33);
            } else {
                policy_refill_rows_out[cell_tick] = counters.refilled_rows;
                policy_cache_hit_rows_out[cell_tick] = counters.hit_rows;
                full_tick_state.policy_np_semantic_draws[cell] +=
                    counters.gumbel_draws;
                for (int64_t row = 0; row < pre_action_rows; ++row) {
                    const int32_t slot = ordered_slots_out[
                        tick_capacity_offset + row
                    ];
                    actions_by_slot_out[tick_capacity_offset + slot] =
                        row_actions_out[tick_capacity_offset + row];
                }
            }
        }
        __syncthreads();
    }
        if (thread == 0) {
            const int64_t refill_rows = policy_refill_rows_out[cell_tick];
            runtime_bookkeeping_by_tick_out[runtime_bookkeeping_offset + 3] =
                refill_rows > 0 ? 1 : 0;
            runtime_bookkeeping_by_tick_out[runtime_bookkeeping_offset + 4] =
                refill_rows;
        }
        if (status_out[cell] != 0) {
            return;
        }
        if (!deterministic_actions && !neutral_fixed_action) {
            const int64_t tick_capacity_offset = cell_tick * capacity;
            for (int64_t row = thread; row < capacity; row += blockDim.x) {
                death_slots_out[tick_capacity_offset + row] = -1;
            }
        }
        __syncthreads();

        if (thread == 0) {
            int32_t* chemistry_order = full_tick_outputs.chemistry_order_slots
                + cell_tick * capacity;
            if (!full_patch_p2) {
                for (int64_t row = 0; row < pre_action_rows; ++row) {
                    chemistry_order[row] = ordered_slots_out[
                        cell_tick * capacity + row
                    ];
                }
                persistent_chemistry::runner_shuffle_counted(
                    chemistry_order,
                    pre_action_rows,
                    full_tick_state.runner_pcg64_state_words + cell * 4,
                    full_tick_state.runner_pcg64_buffer_words + cell * 2,
                    full_tick_outputs.runner_shuffle_draws + cell_tick,
                    full_tick_outputs.runner_shuffle_status + cell_tick,
                    capacity
                );
                if (full_tick_outputs.runner_shuffle_status[cell_tick] == 0) {
                    full_tick_state.metrics_np_semantic_draws[cell] +=
                        full_tick_outputs.runner_shuffle_draws[cell_tick];
                } else {
                    RECORD_TICK_CORE_STATUS(20);
                }
            }
            for (int64_t word = 0; word < 4; ++word) {
                full_tick_outputs.policy_pcg64_state_after_action[
                    cell_tick * 4 + word
                ] = full_tick_state.shared_env_policy_pcg64_state_words[
                    cell * 4 + word
                ];
            }
            for (int64_t word = 0; word < 2; ++word) {
                full_tick_outputs.policy_pcg64_buffer_after_action[
                    cell_tick * 2 + word
                ] = full_tick_state.shared_env_policy_pcg64_buffer_words[
                    cell * 2 + word
                ];
            }
        }
        __syncthreads();
        if (status_out[cell] != 0) {
            return;
        }

        // Stage 2 commits coherence counters before cloning pre-action state.
        if (thread == 0) {
            reporting::GenerationAuthorityAccessV1 reporting_access{};
            bool count_reporting_actions = false;
            if (
                compact_summary_context_addressable_v1(
                    compact_summary_context, cell
                )
            ) {
                reporting_access = compact_summary_reporting_access_v1(
                    compact_summary_context, cell
                );
                if (compact_summary_context.slot_capacity != capacity) {
                    reporting::record_failure_v1(
                        reporting_access.header,
                        reporting::kLedgerInvalidStorageV1
                    );
                } else {
                    count_reporting_actions = true;
                }
            }
            for (int64_t row = 0; row < pre_action_rows; ++row) {
                const int64_t agent_id = active_agent_ids[cell_active_offset + row];
                if (!active_identity::agent_id_fits_grid_v1(agent_id)) {
                    continue;
                }
                const int32_t slot =
                    active_agent_slots[cell_active_offset + row];
                if (slot < 0 || static_cast<int64_t>(slot) >= capacity) {
                    continue;
                }
                const int64_t slot_index = cell_capacity_offset + slot;
                const int32_t action = full_patch_p2
                    ? actions_by_slot_out[cell_tick * capacity + slot]
                    : row_actions_out[cell_tick * capacity + row];
                if (count_reporting_actions) {
                    reporting::count_action_v1(
                        reporting_access, slot, agent_id, action
                    );
                }
                const int32_t old_last = last_action[slot_index];
                const int32_t old_last_last = last_last_action[slot_index];
                if (old_last != -1 && action != old_last) {
                    action_switches[slot_index] += 1;
                    if (survival_steps[slot_index] < episode0_k) {
                        episode0_switches[slot_index] += 1;
                    }
                }
                if (old_last_last != -1 && action == old_last_last && action != old_last) {
                    action_oscillations[slot_index] += 1;
                    if (survival_steps[slot_index] < episode0_k) {
                        episode0_oscillations[slot_index] += 1;
                    }
                }
                last_last_action[slot_index] = old_last;
                last_action[slot_index] = action;
            }
        }
        __syncthreads();

        for (int64_t slot = thread; slot < capacity; slot += blockDim.x) {
            const int64_t source = cell_capacity_offset + slot;
            const int64_t target = cell_tick * capacity + slot;
            pre_energy_out[target] = energy[source];
            pre_health_out[target] = health[source];
            pre_injury_out[target] = injury[source];
            pre_survival_steps_out[target] = survival_steps[source];
        }
        __syncthreads();

        // Occupancy changes are order-dependent and must follow ascending agent ID.
        if (thread == 0) {
            for (int64_t row = 0; row < pre_action_rows; ++row) {
                const int64_t agent_id = active_agent_ids[cell_active_offset + row];
                if (!active_identity::agent_id_fits_grid_v1(agent_id)) {
                    continue;
                }
                const int32_t slot =
                    active_agent_slots[cell_active_offset + row];
                if (slot < 0 || static_cast<int64_t>(slot) >= capacity) {
                    continue;
                }
                const int32_t action = full_patch_p2
                    ? actions_by_slot_out[cell_tick * capacity + slot]
                    : row_actions_out[cell_tick * capacity + row];
                if (action < 0 || action > 3) {
                    continue;
                }
                double* const summary = generation_summary_scalar_out == nullptr
                    ? nullptr
                    : generation_summary_scalar_out
                        + cell * kGenerationSummaryScalarWidthV1;
                if (summary != nullptr) {
                    summary[kGenerationSummaryMovementAttempts] += 1.0;
                }
                const int64_t slot_index = cell_capacity_offset + slot;
                const int32_t y = pos_y[slot_index];
                const int32_t x = pos_x[slot_index];
                int32_t ny = y;
                int32_t nx = x;
                if (action == 0) {
                    --ny;
                } else if (action == 1) {
                    ++ny;
                } else if (action == 2) {
                    ++nx;
                } else {
                    --nx;
                }
                if (ny < 0 || ny >= height || nx < 0 || nx >= width) {
                    if (summary != nullptr) {
                        summary[kGenerationSummaryMovementBlockedBounds] += 1.0;
                    }
                    continue;
                }
                const int64_t target = cell_grid_offset
                    + static_cast<int64_t>(ny) * width + nx;
                if (agent_grid[target] != -1) {
                    if (summary != nullptr) {
                        summary[kGenerationSummaryMovementBlockedOccupied] += 1.0;
                    }
                    continue;
                }
                const int64_t source = cell_grid_offset
                    + static_cast<int64_t>(y) * width + x;
                if (agent_grid[source] == static_cast<int32_t>(agent_id)) {
                    agent_grid[source] = -1;
                }
                agent_grid[target] = static_cast<int32_t>(agent_id);
                pos_y[slot_index] = ny;
                pos_x[slot_index] = nx;
                if (full_patch_p2) {
                    // Move and gather are mutually exclusive, so the negative
                    // sentinel carries successful-move state without another
                    // per-tick allocation.
                    food_gained_scratch[slot_index] = -1.0f;
                    if (patch_p2_context.patch_scent_move_emit > 0.0) {
                        const float delta = static_cast<float>(
                            patch_p2_context.patch_scent_move_emit
                        );
                        patch_p2_context.state.patch_scent_emit_pending[
                            target
                        ] += delta;
                        if (patch_p2_context.compact_metrics != nullptr) {
                            patch_tick_metrics[
                                kPatchCompactScentMoveQueuedTotal
                            ] += static_cast<double>(delta);
                        }
                    }
                    if (patch_p2_context.patch_waste_move_emit > 0.0) {
                        const float delta = static_cast<float>(
                            patch_p2_context.patch_waste_move_emit
                        );
                        patch_p2_context.state.patch_waste_emit_pending[
                            target
                        ] += delta;
                        if (patch_p2_context.compact_metrics != nullptr) {
                            patch_tick_metrics[
                                kPatchCompactWasteMoveQueuedTotal
                            ] += static_cast<double>(delta);
                        }
                    }
                }
                runtime_bookkeeping_by_tick_out[runtime_bookkeeping_offset + 1]
                    += 1;
            }
        }
        __syncthreads();

        if (thread == 0) {
            for (int64_t row = 0; row < pre_action_rows; ++row) {
                const int64_t agent_id = active_agent_ids[cell_active_offset + row];
                if (!active_identity::agent_id_fits_grid_v1(agent_id)) {
                    continue;
                }
                const int32_t slot =
                    active_agent_slots[cell_active_offset + row];
                if (slot < 0 || static_cast<int64_t>(slot) >= capacity) {
                    continue;
                }
                const int32_t action = full_patch_p2
                    ? actions_by_slot_out[cell_tick * capacity + slot]
                    : row_actions_out[cell_tick * capacity + row];
                if (action != kGatherAction) {
                    continue;
                }
                const int64_t slot_index = cell_capacity_offset + slot;
                const int64_t grid_index = cell_grid_offset
                    + static_cast<int64_t>(pos_y[slot_index]) * width
                    + pos_x[slot_index];
                if (full_patch_p2) {
                    double gather_amount =
                        patch_p2_context.patch_food_gather_amount_by_cell == nullptr
                        ? patch_p2_context.patch_food_gather_amount
                        : patch_p2_context.patch_food_gather_amount_by_cell[cell];
                    if (gather_amount <= 0.0) {
                        gather_amount = 1.0;
                    }
                    const double threshold = static_cast<double>(
                        patch_p2_context.config.food_presence_threshold
                    );
                    const double amount = static_cast<double>(
                        patch_p2_context.state.patch_food_amount[grid_index]
                    );
                    if (amount <= threshold) {
                        food_grid[grid_index] = kEmpty;
                        continue;
                    }
                    const double harvested = amount > gather_amount
                        ? gather_amount
                        : amount;
                    double post = amount - harvested;
                    if (post < 0.0) {
                        post = 0.0;
                    }
                    patch_p2_context.state.patch_food_amount[grid_index] =
                        static_cast<float>(post);
                    food_grid[grid_index] = post > threshold ? kFood : kEmpty;
                    const double harvest_scale = harvested / gather_amount;
                    food_gained_scratch[slot_index] = static_cast<float>(
                        food_energy * harvest_scale
                    );
                    if (patch_p2_context.compact_metrics != nullptr) {
                        patch_tick_metrics[kPatchCompactFoodHarvestTotal] +=
                            harvested;
                        patch_tick_metrics[kPatchCompactEnergyCollectedDelta] +=
                            static_cast<double>(food_gained_scratch[slot_index]);
                    }
                    if (patch_p2_context.patch_scent_gather_emit > 0.0) {
                        const double delta =
                            patch_p2_context.patch_scent_gather_emit
                            * harvest_scale;
                        patch_p2_context.state.patch_scent_emit_pending[
                            grid_index
                        ] = static_cast<float>(
                            static_cast<double>(
                                patch_p2_context.state.patch_scent_emit_pending[
                                    grid_index
                                ]
                            ) + delta
                        );
                        if (patch_p2_context.compact_metrics != nullptr) {
                            patch_tick_metrics[
                                kPatchCompactScentGatherQueuedTotal
                            ] += delta;
                        }
                    }
                    if (patch_p2_context.patch_waste_consume_emit > 0.0) {
                        const double delta =
                            patch_p2_context.patch_waste_consume_emit
                            * harvest_scale;
                        patch_p2_context.state.patch_waste_emit_pending[
                            grid_index
                        ] = static_cast<float>(
                            static_cast<double>(
                                patch_p2_context.state.patch_waste_emit_pending[
                                    grid_index
                                ]
                            ) + delta
                        );
                        if (patch_p2_context.compact_metrics != nullptr) {
                            patch_tick_metrics[
                                kPatchCompactWasteGatherQueuedTotal
                            ] += delta;
                        }
                    }
                    runtime_bookkeeping_by_tick_out[
                        runtime_bookkeeping_offset + 2
                    ] += 1;
                } else if (food_grid[grid_index] == kFood) {
                    food_grid[grid_index] = kEmpty;
                    food_gained_scratch[slot_index] = static_cast<float>(food_energy);
                    runtime_bookkeeping_by_tick_out[
                        runtime_bookkeeping_offset + 2
                    ] += 1;
                }
            }
        }
        __syncthreads();

        if (thread == 0) {
            for (int64_t row = 0; row < pre_action_rows; ++row) {
                const int64_t agent_id = active_agent_ids[cell_active_offset + row];
                if (!active_identity::agent_id_fits_grid_v1(agent_id)) {
                    continue;
                }
                const int32_t slot =
                    active_agent_slots[cell_active_offset + row];
                if (slot < 0 || static_cast<int64_t>(slot) >= capacity) {
                    continue;
                }
                const int64_t slot_index = cell_capacity_offset + slot;
                const int32_t action = full_patch_p2
                    ? actions_by_slot_out[cell_tick * capacity + slot]
                    : row_actions_out[cell_tick * capacity + row];
                float cost = action_cost(
                    action,
                    move_cost,
                    gather_cost,
                    rest_cost,
                    attack_cost,
                    action_cost_mult,
                    rest_cost_mult
                );
                const bool moved = full_patch_p2
                    && food_gained_scratch[slot_index] < 0.0f;
                if (genome_v2_mode) {
                    namespace genome_v2 =
                        eiel::genome_upgrade_v2_q10_v1;
                    const auto planes = genome_v2_resident_planes_v1(
                        ordinary_birth_state, cell, capacity
                    );
                    const auto runtime =
                        genome_v2::load_runtime_projection_v1(planes, slot);
                    double adjusted_cost = static_cast<double>(cost);
                    if (moved) {
                        (void)genome_v2::successful_movement_cost_v1(
                            cost, runtime, &adjusted_cost
                        );
                    }
                    double expression_debit = 0.0;
                    (void)genome_v2::expression_energy_debit_v1(
                        runtime, &expression_debit
                    );
                    cost = static_cast<float>(
                        adjusted_cost + expression_debit
                    );
                }
                if (
                    moved
                    && patch_p2_context.patch_movement_cost_apply_enable
                ) {
                    const float old_cost = cost;
                    const int64_t grid_index = cell_grid_offset
                        + static_cast<int64_t>(pos_y[slot_index]) * width
                        + pos_x[slot_index];
                    const float terrain_cost =
                        patch_p2_context.patch_movement_cost[grid_index];
                    const float multiplier = fmaxf(
                        0.0f,
                        1.0f
                            + static_cast<float>(
                                patch_p2_context
                                    .patch_movement_cost_energy_scale_by_cell[cell]
                            ) * (terrain_cost - 1.0f)
                    );
                    cost = fmaxf(0.0f, cost * multiplier);
                    if (patch_p2_context.compact_metrics != nullptr) {
                        patch_tick_metrics[
                            kPatchCompactMovementEnergyExtraTotal
                        ] +=
                            static_cast<double>(cost - old_cost);
                        patch_tick_metrics[
                            kPatchCompactMovementEnergyEventCount
                        ] += 1.0;
                    }
                }
                const float food_gained = food_gained_scratch[slot_index] > 0.0f
                    ? food_gained_scratch[slot_index]
                    : 0.0f;
                float updated = energy[slot_index]
                    - cost
                    + food_gained;
                if (updated > static_cast<float>(max_energy)) {
                    updated = static_cast<float>(max_energy);
                }
                if (updated < 0.0f) {
                    updated = 0.0f;
                }
                energy[slot_index] = updated;
                policy_energy_observation[slot_index] = static_cast<double>(updated);
                if (food_gained > 0.0f) {
                    energy_collected_exact[slot_index] += static_cast<double>(
                        food_gained
                    );
                    energy_collected[slot_index] = static_cast<float>(
                        energy_collected_exact[slot_index]
                    );
                }
            }
        }
        __syncthreads();

        if (thread == 0) {
            const int64_t tick_capacity_offset = cell_tick * capacity;
            reporting::GenerationAuthorityAccessV1 reporting_access{};
            reporting::ResidentGenerationSourcesV1 reporting_sources{};
            bool seal_reporting_deaths = false;
            if (
                compact_summary_context_addressable_v1(
                    compact_summary_context, cell
                )
            ) {
                reporting_access = compact_summary_reporting_access_v1(
                    compact_summary_context, cell
                );
                reporting_sources = compact_summary_resident_sources_v1(
                    slot_to_agent_id,
                    energy_collected_exact,
                    injury,
                    action_switches,
                    action_oscillations,
                    survival_steps,
                    policy_energy_observation,
                    health,
                    drives,
                    pos_y,
                    pos_x,
                    full_tick_state,
                    event_state,
                    ordinary_birth_state,
                    cell,
                    capacity,
                    drive_width
                );
                if (
                    compact_summary_context.slot_capacity != capacity
                    || drive_width != 4
                    || !reporting::resident_sources_valid_v1(
                        reporting_sources,
                        compact_summary_context.slot_capacity
                    )
                ) {
                    reporting::record_failure_v1(
                        reporting_access.header,
                        reporting::kLedgerInvalidStorageV1
                    );
                } else {
                    seal_reporting_deaths = true;
                }
            }
            const int32_t lifecycle_status = lifecycle_death_cull_cell_v3(
                active_agent_ids + cell_active_offset,
                active_agent_slots + cell_active_offset,
                active_count_state + cell,
                slot_to_agent_id + cell_capacity_offset,
                agent_id_to_slot + cell_id_map_offset,
                alive_mask + cell_capacity_offset,
                pos_y + cell_capacity_offset,
                pos_x + cell_capacity_offset,
                agent_grid + cell_grid_offset,
                energy + cell_capacity_offset,
                health + cell_capacity_offset,
                policy_energy_observation + cell_capacity_offset,
                rest_streak + cell_capacity_offset,
                action_cache_token + cell_capacity_offset,
                action_cache_cursor + cell_capacity_offset,
                action_cache_next + cell_capacity_offset,
                pending_free_agent_ids + cell_pending_offset,
                pending_free_slots + cell_pending_offset,
                pending_free_count + cell,
                death_agent_ids_out + tick_capacity_offset,
                death_slots_out + tick_capacity_offset,
                death_reason_codes_out + tick_capacity_offset,
                death_count_out + cell_tick,
                survivor_agent_ids_out + tick_capacity_offset,
                survivor_count_out + cell_tick,
                seal_reporting_deaths,
                true,
                reporting_access,
                reporting_sources,
                capacity,
                active_capacity,
                agent_id_capacity,
                height,
                width,
                pending_capacity
            );
            lifecycle_status_out[cell_tick] = lifecycle_status;
            if (lifecycle_status != kLifecycleOk) {
                RECORD_TICK_CORE_STATUS(4 + lifecycle_status);
            } else {
                current_active_rows = active_count_state[cell];
                accumulate_action_terminal_v4(
                    event_state,
                    death_agent_ids_out,
                    death_slots_out,
                    death_reason_codes_out,
                    death_count_out,
                    survival_steps,
                    arguments.output_ticks_per_cell,
                    capacity,
                    agent_id_capacity,
                    cell,
                    output_tick
                );
                append_death_event_class_v4(
                    event_state,
                    death_agent_ids_out,
                    death_slots_out,
                    death_reason_codes_out,
                    death_count_out,
                    arguments.output_ticks_per_cell,
                    capacity,
                    cell,
                    output_tick,
                    mechanics_config.generation,
                    mechanics_config.start_tick + tick
                );
                if (!arguments.generation_turnover_enabled) {
                    int64_t primary_release_count = 0;
                    const int32_t primary_release_status =
                        release_residual_free_slots_v4(
                            slot_to_agent_id + cell_capacity_offset,
                            alive_mask + cell_capacity_offset,
                            full_tick_state.slot_free_queue + cell_capacity_offset,
                            full_tick_state.slot_free_queue_count + cell,
                            death_slots_out + tick_capacity_offset,
                            death_count_out[cell_tick],
                            &primary_release_count,
                            capacity,
                            capacity
                        );
                    if (primary_release_status != kLifecycleOk) {
                        RECORD_TICK_CORE_STATUS(80 + primary_release_status);
                    } else {
                        pending_free_count[cell] = 0;
                    }
                }
            }
        }
        __syncthreads();
        if (status_out[cell] != 0) {
            return;
        }

        if (thread == 0) {
            using namespace persistent_active_mechanics;
            namespace scratch = flattened_birth_scratch_v1;
            const int64_t global_tick = mechanics_config.start_tick + tick;
            const int64_t sample_cell_offset =
                (cell * full_tick_state.drive_noise_sample_ticks + global_tick)
                * capacity;
            const auto drive_noise_layout =
                scratch::make_resident_drive_noise_layout(capacity);
            float* effective_drive_noise_samples = scratch::at<float>(
                ordinary_birth_scratch,
                cell,
                drive_noise_layout.samples
            );
            bool* effective_drive_noise_samples_valid = scratch::at<bool>(
                ordinary_birth_scratch,
                cell,
                drive_noise_layout.valid
            );
            materialize_missing_drive_noise_v1(
                full_tick_outputs.chemistry_order_slots + cell_tick * capacity,
                pre_action_rows,
                slot_to_agent_id + cell_capacity_offset,
                alive_mask + cell_capacity_offset,
                full_tick_state.drive_noise + cell_capacity_offset,
                full_tick_state.drive_noise_samples
                    + sample_cell_offset * drive_width,
                full_tick_state.drive_noise_samples_valid + sample_cell_offset,
                effective_drive_noise_samples,
                effective_drive_noise_samples_valid,
                full_tick_state.runner_pcg64_state_words + cell * 4,
                full_tick_state.runner_pcg64_buffer_words + cell * 2,
                full_tick_outputs.drive_noise_draws + cell_tick,
                full_tick_outputs.drive_noise_raw64_draws + cell_tick,
                full_tick_outputs.drive_noise_status + cell_tick,
                capacity
            );
            if (full_tick_outputs.drive_noise_status[cell_tick] == 0) {
                full_tick_state.metrics_np_semantic_draws[cell] +=
                    full_tick_outputs.drive_noise_draws[cell_tick];
            } else {
                RECORD_TICK_CORE_STATUS(40);
            }
            const FlatActiveMechanicsCellState mechanics_state{
                {slot_to_agent_id + cell_capacity_offset, 1},
                {alive_mask + cell_capacity_offset, 1},
                {energy + cell_capacity_offset, 1},
                {health + cell_capacity_offset, 1},
                {injury + cell_capacity_offset, 1},
                {full_tick_state.age + cell_capacity_offset, 1},
                {policy_energy_observation + cell_capacity_offset, 1},
                {rest_streak + cell_capacity_offset, 1},
                {survival_steps + cell_capacity_offset, 1},
                {full_tick_state.gru_upkeep_cost + cell_capacity_offset, 1},
                {full_tick_state.cost_charge_scalar + cell_capacity_offset, 1},
                {
                    full_tick_state.cost_charge_maturation_scalar
                        + cell_capacity_offset,
                    1
                },
                {drives + cell_capacity_offset * drive_width, drive_width, 1},
                {full_tick_state.drive_inertia + cell_capacity_offset, 1},
                {full_tick_state.drive_noise + cell_capacity_offset, 1},
                {
                    full_tick_state.prev_drive_delta
                        + cell_capacity_offset * drive_width,
                    drive_width,
                    1
                },
                {full_tick_state.drive_volatility + cell_capacity_offset, 1},
                {full_tick_state.drive_jitter + cell_capacity_offset, 1},
                {
                    full_tick_state.episode0_drive_jitter_sum
                        + cell_capacity_offset,
                    1
                },
                {full_tick_state.last_energy_delta + cell_capacity_offset, 1},
                {full_tick_state.last_health_delta + cell_capacity_offset, 1},
                {full_tick_state.last_injury_delta + cell_capacity_offset, 1},
                {full_tick_state.selfharm_events + cell_capacity_offset, 1},
                {
                    full_tick_state.episode0_selfharm_events
                        + cell_capacity_offset,
                    1
                },
            };
            const FlatActiveMechanicsPreAction pre_action{
                {actions_by_slot_out + cell_tick * capacity, 1},
                {pre_energy_out + cell_tick * capacity, 1},
                {pre_health_out + cell_tick * capacity, 1},
                {pre_injury_out + cell_tick * capacity, 1},
                {
                    effective_drive_noise_samples,
                    drive_width,
                    1
                },
                {
                    effective_drive_noise_samples_valid,
                    1
                },
            };
            run_flat_active_mechanics(
                mechanics_state,
                pre_action,
                full_tick_outputs.active_mechanics_status + cell_tick,
                full_tick_outputs.active_mechanics_counts + cell_tick * 5,
                1,
                capacity,
                mechanics_config.enable_rest_charge,
                mechanics_config.rest_streak_free,
                mechanics_config.rest_charge_energy,
                mechanics_config.rest_charge_growth,
                mechanics_config.rest_deficit_to_health,
                mechanics_config.developmental_maturity_age,
                mechanics_config.life_history_mode,
                mechanics_config.life_history_maturity_start_ticks,
                mechanics_config.life_history_maturity_rise_ticks,
                mechanics_config.life_history_fertility_decline_start_ticks,
                mechanics_config.life_history_fertility_decline_ticks,
                mechanics_config.life_history_elder_fertility_floor,
                mechanics_config.life_history_senescence_start_ticks,
                mechanics_config.life_history_senescence_ramp_ticks,
                mechanics_config.life_history_senescence_energy_cost,
                mechanics_config.life_history_senescence_health_cost,
                mechanics_config.life_history_max_age_ticks,
                static_cast<int32_t>(episode0_k),
                static_cast<float>(max_energy),
                static_cast<float>(start_health),
                mechanics_config.drive_hunger_gain,
                mechanics_config.drive_hunger_decay,
                mechanics_config.drive_pain_gain,
                mechanics_config.drive_pain_decay,
                mechanics_config.drive_fatigue_gain,
                mechanics_config.drive_fatigue_rest_relief,
                mechanics_config.drive_fatigue_decay,
                mechanics_config.drive_fear_gain,
                mechanics_config.drive_fear_decay
            );
            if (full_tick_outputs.active_mechanics_status[cell_tick] != 0) {
                RECORD_TICK_CORE_STATUS(30);
            }
            if (generation_summary_scalar_out != nullptr) {
                double* const summary = generation_summary_scalar_out
                    + cell * kGenerationSummaryScalarWidthV1;
                double peak_injury = summary[kGenerationSummaryPeakInjury];
                const bool rolling_outputs =
                    arguments.output_ticks_per_cell == 1;
                for (int64_t row = 0; row < pre_action_rows; ++row) {
                    const int32_t slot = ordered_slots_out[
                        cell_tick * capacity + row
                    ];
                    if (slot >= 0 && static_cast<int64_t>(slot) < capacity) {
                        const int64_t output_slot =
                            cell_tick * capacity + slot;
                        const double pre_tick_injury = static_cast<double>(
                            pre_injury_out[output_slot]
                        );
                        const double post_tick_injury = static_cast<double>(
                            injury[cell_capacity_offset + slot]
                        );
                        if (
                            !isfinite(pre_tick_injury)
                            || !isfinite(post_tick_injury)
                        ) {
                            peak_injury = NAN;
                            break;
                        }
                        peak_injury = fmax(
                            peak_injury,
                            fmax(pre_tick_injury, post_tick_injury)
                        );
                        if (rolling_outputs) {
                            const int32_t action =
                                actions_by_slot_out[output_slot];
                            if (action >= 0 && action <= kAttackAction) {
                                summary[kGenerationSummaryActionTotal] += 1.0;
                            }
                            if (action >= 0 && action < kGatherAction) {
                                summary[kGenerationSummaryMoveTotal] += 1.0;
                            } else if (action == kGatherAction) {
                                summary[kGenerationSummaryGatherTotal] += 1.0;
                            }
                        }
                    }
                }
                summary[kGenerationSummaryPeakInjury] = peak_injury;
                if (rolling_outputs) {
                    const int64_t refill_rows =
                        policy_refill_rows_out[cell_tick];
                    summary[kGenerationSummaryPolicyRefillRows] +=
                        static_cast<double>(refill_rows);
                    summary[kGenerationSummaryPolicyCacheRowsHit] +=
                        static_cast<double>(
                            policy_cache_hit_rows_out[cell_tick]
                        );
                    if (refill_rows > 0) {
                        summary[kGenerationSummaryPolicyRefillCalls] += 1.0;
                        const int64_t refill_column =
                            refill_rows == pre_action_rows
                            ? kGenerationSummaryPolicyCacheFullRefillTicks
                            : kGenerationSummaryPolicyCachePartialRefillTicks;
                        summary[refill_column] += 1.0;
                    }
                }
            }

            int32_t* chemistry_order = full_tick_outputs.chemistry_order_slots
                + cell_tick * capacity;
            persistent_chemistry::chemistry_update_counted(
                full_tick_state.chemistry
                    + cell_capacity_offset * full_tick_state.chemistry_width,
                energy + cell_capacity_offset,
                health + cell_capacity_offset,
                injury + cell_capacity_offset,
                drives + cell_capacity_offset * drive_width,
                alive_mask + cell_capacity_offset,
                chemistry_order,
                pre_action_rows,
                full_tick_state.shared_env_policy_pcg64_state_words + cell * 4,
                full_tick_state.shared_env_policy_pcg64_buffer_words + cell * 2,
                full_tick_state.policy_np_semantic_draws + cell,
                full_tick_outputs.chemistry_draws + cell_tick,
                full_tick_outputs.chemistry_status + cell_tick,
                capacity,
                {
                    full_tick_state.chemistry_width,
                    drive_width,
                    static_cast<float>(max_energy),
                    static_cast<float>(start_health),
                }
            );
            if (full_patch_p2 && patch_p2_context.compact_metrics != nullptr) {
                patch_tick_metrics[kPatchCompactEnvRngPaddingDraws] +=
                    static_cast<double>(
                    full_tick_outputs.chemistry_draws[cell_tick]
                );
                patch_tick_metrics[kPatchCompactScentEmitTotal] =
                    patch_tick_metrics[kPatchCompactScentMoveQueuedTotal]
                    + patch_tick_metrics[kPatchCompactScentGatherQueuedTotal];
                patch_tick_metrics[kPatchCompactWasteEmitTotal] =
                    patch_tick_metrics[kPatchCompactWasteMoveQueuedTotal]
                    + patch_tick_metrics[kPatchCompactWasteGatherQueuedTotal];
                double* const metrics = patch_p2_context.compact_metrics
                    + cell * kPatchCompactMetricCount;
                for (const int64_t field : {
                         kPatchCompactMovementEnergyEventCount,
                         kPatchCompactEnvRngPaddingDraws,
                         kPatchCompactFoodHarvestTotal,
                         kPatchCompactScentEmitTotal,
                         kPatchCompactWasteEmitTotal,
                         kPatchCompactEnergyCollectedDelta,
                         kPatchCompactScentMoveQueuedTotal,
                         kPatchCompactWasteMoveQueuedTotal,
                         kPatchCompactScentGatherQueuedTotal,
                         kPatchCompactWasteGatherQueuedTotal,
                         kPatchCompactMovementEnergyExtraTotal,
                     }) {
                    metrics[field] += patch_tick_metrics[field];
                }
            }
            if (full_tick_outputs.chemistry_status[cell_tick] != 0) {
                RECORD_TICK_CORE_STATUS(50);
            }
            if (genome_v2_mode && status_out[cell] == 0) {
                namespace genome_v2 = eiel::genome_upgrade_v2_q10_v1;
                const auto planes = genome_v2_resident_planes_v1(
                    ordinary_birth_state, cell, capacity
                );
                for (int64_t row = 0; row < pre_action_rows; ++row) {
                    const int32_t slot = ordered_slots_out[
                        cell_tick * capacity + row
                    ];
                    if (
                        slot < 0
                        || static_cast<int64_t>(slot) >= capacity
                        || !alive_mask[cell_capacity_offset + slot]
                    ) {
                        continue;
                    }
                    const int64_t slot_index = cell_capacity_offset + slot;
                    const int64_t output_slot = cell_tick * capacity + slot;
                    const float before = pre_health_out[output_slot];
                    const float after = health[slot_index];
                    const auto runtime =
                        genome_v2::load_runtime_projection_v1(planes, slot);
                    double adjusted_health = before;
                    (void)genome_v2::health_after_delta_v1(
                        before,
                        after,
                        start_health,
                        runtime,
                        &adjusted_health
                    );
                    health[slot_index] = static_cast<float>(adjusted_health);
                    full_tick_state.last_health_delta[slot_index] =
                        health[slot_index] - before;
                }
            }

            const int64_t tick_capacity_offset = cell_tick * capacity;
            reporting::GenerationAuthorityAccessV1 reporting_access{};
            reporting::ResidentGenerationSourcesV1 reporting_sources{};
            bool seal_reporting_deaths = false;
            if (
                compact_summary_context_addressable_v1(
                    compact_summary_context, cell
                )
            ) {
                reporting_access = compact_summary_reporting_access_v1(
                    compact_summary_context, cell
                );
                reporting_sources = compact_summary_resident_sources_v1(
                    slot_to_agent_id,
                    energy_collected_exact,
                    injury,
                    action_switches,
                    action_oscillations,
                    survival_steps,
                    policy_energy_observation,
                    health,
                    drives,
                    pos_y,
                    pos_x,
                    full_tick_state,
                    event_state,
                    ordinary_birth_state,
                    cell,
                    capacity,
                    drive_width
                );
                if (
                    compact_summary_context.slot_capacity != capacity
                    || drive_width != 4
                    || !reporting::resident_sources_valid_v1(
                        reporting_sources,
                        compact_summary_context.slot_capacity
                    )
                ) {
                    reporting::record_failure_v1(
                        reporting_access.header,
                        reporting::kLedgerInvalidStorageV1
                    );
                } else {
                    seal_reporting_deaths = true;
                }
            }
            int64_t* const residual_pending_agent_ids =
                arguments.generation_turnover_enabled
                ? pending_free_agent_ids + cell_pending_offset
                : full_tick_state.residual_pending_free_agent_ids
                    + cell_capacity_offset;
            int64_t* const residual_pending_slots =
                arguments.generation_turnover_enabled
                ? pending_free_slots + cell_pending_offset
                : full_tick_state.residual_pending_free_slots
                    + cell_capacity_offset;
            int64_t* const residual_pending_count =
                arguments.generation_turnover_enabled
                ? pending_free_count + cell
                : full_tick_state.residual_pending_free_count + cell;
            const int32_t residual_status = lifecycle_death_cull_cell_v3(
                active_agent_ids + cell_active_offset,
                active_agent_slots + cell_active_offset,
                active_count_state + cell,
                slot_to_agent_id + cell_capacity_offset,
                agent_id_to_slot + cell_id_map_offset,
                alive_mask + cell_capacity_offset,
                pos_y + cell_capacity_offset,
                pos_x + cell_capacity_offset,
                agent_grid + cell_grid_offset,
                energy + cell_capacity_offset,
                health + cell_capacity_offset,
                policy_energy_observation + cell_capacity_offset,
                rest_streak + cell_capacity_offset,
                action_cache_token + cell_capacity_offset,
                action_cache_cursor + cell_capacity_offset,
                action_cache_next + cell_capacity_offset,
                residual_pending_agent_ids,
                residual_pending_slots,
                residual_pending_count,
                full_tick_outputs.residual_death_agent_ids
                    + tick_capacity_offset,
                full_tick_outputs.residual_death_slots + tick_capacity_offset,
                full_tick_outputs.residual_death_reason_codes
                    + tick_capacity_offset,
                full_tick_outputs.residual_death_count + cell_tick,
                survivor_agent_ids_out + tick_capacity_offset,
                survivor_count_out + cell_tick,
                seal_reporting_deaths,
                false,
                reporting_access,
                reporting_sources,
                capacity,
                active_capacity,
                agent_id_capacity,
                height,
                width,
                capacity
            );
            if (residual_status != kLifecycleOk) {
                RECORD_TICK_CORE_STATUS(60 + residual_status);
            } else {
                current_active_rows = active_count_state[cell];
                append_death_event_class_v4(
                    event_state,
                    full_tick_outputs.residual_death_agent_ids,
                    full_tick_outputs.residual_death_slots,
                    full_tick_outputs.residual_death_reason_codes,
                    full_tick_outputs.residual_death_count,
                    arguments.output_ticks_per_cell,
                    capacity,
                    cell,
                    output_tick,
                    mechanics_config.generation,
                    global_tick
                );
            }

            int32_t residual_release_status_for_birth = kLifecycleOk;
            if (arguments.generation_turnover_enabled) {
                full_tick_outputs.residual_release_count[cell_tick] = 0;
                full_tick_outputs.residual_release_status[cell_tick] =
                    kLifecycleOk;
            } else {
                const int32_t release_status = release_residual_free_slots_v4(
                    slot_to_agent_id + cell_capacity_offset,
                    alive_mask + cell_capacity_offset,
                    full_tick_state.slot_free_queue + cell_capacity_offset,
                    full_tick_state.slot_free_queue_count + cell,
                    full_tick_outputs.residual_death_slots + tick_capacity_offset,
                    full_tick_outputs.residual_death_count[cell_tick],
                    full_tick_outputs.residual_release_count + cell_tick,
                    capacity,
                    capacity
                );
                full_tick_outputs.residual_release_status[cell_tick] =
                    release_status;
                residual_release_status_for_birth = release_status;
                if (release_status != kLifecycleOk) {
                    RECORD_TICK_CORE_STATUS(70 + release_status);
                }
            }

            if (
                ordinary_birth_config.enabled
                && cross_path_resident_control_v1::
                    ordinary_live_birth_cell_v1(
                        ordinary_birth_config.cross_path_control, cell
                    )
            ) {
                const int32_t birth_status = execute_ordinary_birth_phase_v1<1>(
                    active_agent_ids,
                    active_agent_slots,
                    active_count_state,
                    slot_to_agent_id,
                    agent_id_to_slot,
                    slot_tokens,
                    alive_mask,
                    pos_y,
                    pos_x,
                    agent_grid,
                    energy,
                    health,
                    injury,
                    policy_energy_observation,
                    rest_streak,
                    action_cache_actions,
                    action_cache_token,
                    action_cache_cursor,
                    action_cache_next,
                    last_action,
                    last_last_action,
                    action_switches,
                    action_oscillations,
                    episode0_switches,
                    episode0_oscillations,
                    survival_steps,
                    mlp_pack,
                    drives,
                    attack_bias,
                    action_temperature,
                    drive_logit_gain,
                    energy_collected,
                    energy_collected_exact,
                    full_tick_state,
                    event_state,
                    ordinary_birth_state,
                    ordinary_birth_outputs,
                    ordinary_birth_scratch,
                    ordinary_birth_config,
                    mechanics_config,
                    cell,
                    tick,
                    arguments.output_ticks_per_cell,
                    capacity,
                    active_capacity,
                    agent_id_capacity,
                    height,
                    width,
                    cache_capacity,
                    drive_width,
                    packed_width,
                    full_tick_state.chemistry_width,
                    static_cast<float>(max_energy),
                    static_cast<float>(start_health),
                    residual_release_status_for_birth
                );
                if (birth_status != 0) {
                    if (birth_status == 3) {
                        namespace scratch = flattened_birth_scratch_v1;
                        const int32_t transaction_status = scratch::at<int32_t>(
                            ordinary_birth_scratch,
                            cell,
                            scratch::kTransactionStatus
                        )[0];
                        const int64_t event_cursor =
                            event_state.active_count[cell];
                        const int64_t required_event_rows = scratch::at<int64_t>(
                            ordinary_birth_scratch,
                            cell,
                            scratch::kRequiredEventRows
                        )[0];
                        const bool event_capacity_failure =
                            event_state.status[cell]
                                == persistent_birth_events::kStatusCapacityExceeded
                            || (
                                transaction_status == 25
                                && event_cursor >= 0
                                && event_cursor <= event_state.capacity
                                && required_event_rows > 0
                                && required_event_rows
                                    > event_state.capacity - event_cursor
                            );
                        record_status(
                            status_out,
                            first_failure_records,
                            cell,
                            mechanics_config.generation,
                            mechanics_config.start_tick + controller_tick,
                            kControllerFailureStageBirthEventAppend,
                            90 + birth_status,
                            event_capacity_failure
                                ? kFirstFailureEventCapacityExceededV1
                                : kFirstFailureNoFailureSentinelV1,
                            event_capacity_failure
                                ? event_cursor
                                : kFirstFailureNoFailureSentinelV1,
                            event_capacity_failure
                                ? event_cursor + 1
                                : kFirstFailureNoFailureSentinelV1,
                            event_capacity_failure
                                ? event_state.capacity
                                : kFirstFailureNoFailureSentinelV1
                        );
                    } else {
                        RECORD_TICK_CORE_STATUS(90 + birth_status);
                    }
                }
            }
        }
        __syncthreads();

        // The checkpoint contract rebuilds derived density from final
        // occupancy. Policy rows remain in ascending agent-ID order; the
        // separately shuffled order is consumed only by active mechanics.
        if (
            thread == 0
            && full_patch_p2
            && controller_tick + 1 == ticks
            && status_out[cell] == 0
        ) {
            persistent_patch_p2::refresh_patch_density_exact(
                agent_grid + cell_grid_offset,
                patch_p2_context.state.patch_density + cell_grid_offset,
                patch_p2_context.config
            );
        }
        __syncthreads();
#undef active_agent_ids
#undef active_agent_slots
#undef active_count_state
#undef slot_to_agent_id
#undef agent_id_to_slot
#undef alive_mask
#undef slot_tokens
#undef action_cache_actions
#undef action_cache_token
#undef action_cache_cursor
#undef action_cache_next
#undef pos_y
#undef pos_x
#undef agent_grid
#undef food_grid
#undef energy
#undef health
#undef injury
#undef policy_energy_observation
#undef rest_streak
#undef pending_free_agent_ids
#undef pending_free_slots
#undef pending_free_count
#undef energy_collected
#undef energy_collected_exact
#undef last_action
#undef last_last_action
#undef action_switches
#undef action_oscillations
#undef episode0_switches
#undef episode0_oscillations
#undef survival_steps
#undef mlp_pack
#undef drives
#undef attack_bias
#undef action_temperature
#undef drive_logit_gain
#undef food_gained_scratch
#undef ordered_agent_ids_out
#undef ordered_slots_out
#undef observations_out
#undef logits_out
#undef row_actions_out
#undef actions_by_slot_out
#undef pre_energy_out
#undef pre_health_out
#undef pre_injury_out
#undef pre_survival_steps_out
#undef active_count_out
#undef policy_refill_rows_out
#undef policy_cache_hit_rows_out
#undef runtime_bookkeeping_by_tick_out
#undef death_agent_ids_out
#undef death_slots_out
#undef death_reason_codes_out
#undef death_count_out
#undef survivor_agent_ids_out
#undef survivor_count_out
#undef lifecycle_status_out
#undef status_out
#undef first_failure_records
#undef generation_summary_scalar_out
#undef full_tick_state
#undef full_tick_outputs
#undef event_state
#undef ordinary_birth_state
#undef ordinary_birth_outputs
#undef ordinary_birth_scratch
#undef ordinary_birth_config
#undef mechanics_config
#undef patch_p2_context
#undef compact_summary_context
#undef RECORD_TICK_CORE_STATUS
#undef ticks
#undef active_capacity
#undef agent_id_capacity
#undef capacity
#undef pending_capacity
#undef height
#undef width
#undef view_radius
#undef observation_size
#undef hidden_size
#undef action_count
#undef packed_width
#undef drive_width
#undef cache_capacity
#undef cache_width
#undef deterministic_actions
#undef max_food
#undef effective_food_spawn_count
#undef max_energy
#undef start_health
#undef move_cost
#undef gather_cost
#undef rest_cost
#undef attack_cost
#undef action_cost_mult
#undef rest_cost_mult
#undef food_energy
#undef episode0_k
#undef w_hunger_gather
#undef w_pain_rest
#undef w_fatigue_rest
#undef w_pain_attack
#undef w_fear_attack
}

struct GenerationSummaryFinalizeArgumentsV1 {
    double* summary;
    const int64_t* active_count_state;
    const bool* alive_mask;
    const float* energy;
    const double* policy_energy_observation;
    const double* energy_collected_exact;
    const int32_t* ordered_slots;
    const int32_t* actions_by_slot;
    const float* pre_injury;
    const int64_t* active_count;
    const int64_t* policy_refill_rows;
    const int64_t* policy_cache_hit_rows;
    const int64_t* runtime_bookkeeping_by_tick;
    const int32_t* controller_status;
    const int32_t* turnover_status;
    bool turnover_enabled;
    cross_path_resident_control_v1::FlatCrossPathResidentControlV1
        cross_path_control;
    FlatEventState event_state;
    FlatCompactSummaryContextV1 compact_summary_context;
    FlatPatchP2Context patch_p2_context;
    int64_t ticks;
    int64_t output_ticks_per_cell;
    int64_t runtime_bookkeeping_ticks_per_cell;
    int64_t capacity;
    int64_t height;
    int64_t width;
};

__global__ void finalize_generation_summary_scalar_v1(
    const GenerationSummaryFinalizeArgumentsV1 arguments
) {
    if (threadIdx.x != 0) {
        return;
    }
    const int64_t cell = static_cast<int64_t>(blockIdx.x);
    if (
        compact_summary_context_addressable_v1(
            arguments.compact_summary_context, cell
        )
    ) {
        const auto access = compact_summary_reporting_access_v1(
            arguments.compact_summary_context, cell
        );
        if (
            arguments.compact_summary_context.event_cursor_end == nullptr
            || arguments.event_state.active_count == nullptr
            || arguments.compact_summary_context.slot_capacity
                != arguments.capacity
        ) {
            reporting::record_failure_v1(
                access.header, reporting::kLedgerInvalidStorageV1
            );
        } else {
            int64_t absolute_event_cursor = 0;
            if (!absolute_event_cursor_v1(
                    arguments.event_state, cell, &absolute_event_cursor
                )) {
                reporting::record_failure_v1(
                    access.header, reporting::kLedgerInvalidStorageV1
                );
            } else {
                arguments.compact_summary_context.event_cursor_end[
                    compact_summary_row_index_v1(
                        arguments.compact_summary_context, cell
                    )
                ] = absolute_event_cursor;
            }
        }
    }
    if (arguments.summary == nullptr) {
        return;
    }
    double* const summary = arguments.summary
        + cell * kGenerationSummaryScalarWidthV1;
    const auto invalidate = [&]() {
        for (
            int64_t column = 0;
            column < kGenerationSummaryScalarWidthV1;
            ++column
        ) {
            summary[column] = NAN;
        }
    };
    if (
        arguments.controller_status == nullptr
        || arguments.controller_status[cell] != 0
        || arguments.alive_mask == nullptr
        || arguments.policy_energy_observation == nullptr
        || arguments.energy_collected_exact == nullptr
        || (
            arguments.turnover_enabled
            && (
                !cross_path_resident_control_v1::
                    has_cross_path_authority_v1(
                        arguments.cross_path_control
                    )
                || !cross_path_resident_control_v1::
                    ordinary_live_birth_cell_v1(
                        arguments.cross_path_control, cell
                    )
            )
            && (
                arguments.turnover_status == nullptr
                || arguments.turnover_status[cell] != 0
            )
        )
        || arguments.active_count_state[cell] < 0
        || arguments.active_count_state[cell] > arguments.capacity
    ) {
        invalidate();
        return;
    }

    const bool rolling_outputs = arguments.output_ticks_per_cell == 1;
    int64_t action_total = rolling_outputs
        ? static_cast<int64_t>(summary[kGenerationSummaryActionTotal])
        : 0;
    int64_t move_total = rolling_outputs
        ? static_cast<int64_t>(summary[kGenerationSummaryMoveTotal])
        : 0;
    int64_t gather_total = rolling_outputs
        ? static_cast<int64_t>(summary[kGenerationSummaryGatherTotal])
        : 0;
    int64_t moved_total = 0;
    int64_t gathered_total = 0;
    int64_t policy_refill_calls = rolling_outputs
        ? static_cast<int64_t>(summary[kGenerationSummaryPolicyRefillCalls])
        : 0;
    int64_t policy_refill_rows = rolling_outputs
        ? static_cast<int64_t>(summary[kGenerationSummaryPolicyRefillRows])
        : 0;
    int64_t policy_cache_rows_hit = rolling_outputs
        ? static_cast<int64_t>(summary[kGenerationSummaryPolicyCacheRowsHit])
        : 0;
    int64_t policy_cache_full_refill_ticks = rolling_outputs
        ? static_cast<int64_t>(
            summary[kGenerationSummaryPolicyCacheFullRefillTicks]
        )
        : 0;
    int64_t policy_cache_partial_refill_ticks = rolling_outputs
        ? static_cast<int64_t>(
            summary[kGenerationSummaryPolicyCachePartialRefillTicks]
        )
        : 0;
    float peak_injury = rolling_outputs
        ? static_cast<float>(summary[kGenerationSummaryPeakInjury])
        : 0.0f;
    if (
        rolling_outputs
        && (
            !isfinite(summary[kGenerationSummaryActionTotal])
            || !isfinite(summary[kGenerationSummaryMoveTotal])
            || !isfinite(summary[kGenerationSummaryGatherTotal])
            || !isfinite(summary[kGenerationSummaryPeakInjury])
            || !isfinite(summary[kGenerationSummaryPolicyRefillCalls])
            || !isfinite(summary[kGenerationSummaryPolicyRefillRows])
            || !isfinite(summary[kGenerationSummaryPolicyCacheRowsHit])
            || !isfinite(
                summary[kGenerationSummaryPolicyCacheFullRefillTicks]
            )
            || !isfinite(
                summary[kGenerationSummaryPolicyCachePartialRefillTicks]
            )
        )
    ) {
        invalidate();
        return;
    }

    for (int64_t tick = 0; tick < arguments.ticks; ++tick) {
        const int64_t cell_tick = cell * arguments.output_ticks_per_cell + tick;
        int64_t rows = 0;
        if (!rolling_outputs) {
            rows = arguments.active_count[cell_tick];
            if (rows < 0 || rows > arguments.capacity) {
                invalidate();
                return;
            }
            for (int64_t row = 0; row < rows; ++row) {
                const int64_t row_offset =
                    cell_tick * arguments.capacity + row;
                const int32_t slot = arguments.ordered_slots[row_offset];
                if (
                    slot < 0
                    || static_cast<int64_t>(slot) >= arguments.capacity
                ) {
                    continue;
                }
                const int64_t slot_offset =
                    cell_tick * arguments.capacity + slot;
                const int32_t action = arguments.actions_by_slot[slot_offset];
                if (action >= 0 && action <= kAttackAction) {
                    ++action_total;
                }
                if (action >= 0 && action < kGatherAction) {
                    ++move_total;
                } else if (action == kGatherAction) {
                    ++gather_total;
                }
                const float observed_injury =
                    arguments.pre_injury[slot_offset];
                if (!isfinite(observed_injury)) {
                    invalidate();
                    return;
                }
                if (observed_injury > peak_injury) {
                    peak_injury = observed_injury;
                }
            }
        }

        const int64_t bookkeeping_cell_tick =
            cell * arguments.runtime_bookkeeping_ticks_per_cell + tick;
        const int64_t bookkeeping_offset = bookkeeping_cell_tick * 5;
        moved_total += arguments.runtime_bookkeeping_by_tick[
            bookkeeping_offset + 1
        ];
        gathered_total += arguments.runtime_bookkeeping_by_tick[
            bookkeeping_offset + 2
        ];
        if (!rolling_outputs) {
            const int64_t refill_rows =
                arguments.policy_refill_rows[cell_tick];
            policy_refill_rows += refill_rows;
            policy_cache_rows_hit +=
                arguments.policy_cache_hit_rows[cell_tick];
            if (refill_rows > 0) {
                ++policy_refill_calls;
                if (refill_rows == rows) {
                    ++policy_cache_full_refill_ticks;
                } else {
                    ++policy_cache_partial_refill_ticks;
                }
            }
        }
    }

    if (
        !isfinite(summary[kGenerationSummaryEventCursorStart])
        || !isfinite(summary[kGenerationSummaryEventCursorPreTurnover])
    ) {
        invalidate();
        return;
    }
    const int64_t absolute_cursor_start = static_cast<int64_t>(
        summary[kGenerationSummaryEventCursorStart]
    );
    const int64_t absolute_cursor_pre_turnover = static_cast<int64_t>(
        summary[kGenerationSummaryEventCursorPreTurnover]
    );
    int64_t absolute_cursor_end = 0;
    int64_t cursor_start = 0;
    int64_t cursor_pre_turnover = 0;
    const int64_t cursor_end = arguments.event_state.active_count[cell];
    if (
        !absolute_event_cursor_v1(
            arguments.event_state, cell, &absolute_cursor_end
        )
        || !physical_cursor_from_absolute_v1(
            absolute_cursor_start,
            arguments.event_state.sequence_base[cell],
            arguments.event_state.capacity,
            &cursor_start
        )
        || !physical_cursor_from_absolute_v1(
            absolute_cursor_pre_turnover,
            arguments.event_state.sequence_base[cell],
            arguments.event_state.capacity,
            &cursor_pre_turnover
        )
        || cursor_start < 0
        || cursor_pre_turnover < cursor_start
        || cursor_end < cursor_pre_turnover
        || cursor_end > arguments.event_state.capacity
    ) {
        invalidate();
        return;
    }
    int64_t births = 0;
    int64_t deaths = 0;
    int64_t failed_births = 0;
    const int64_t event_offset = cell * arguments.event_state.capacity;
    for (int64_t row = cursor_start; row < cursor_pre_turnover; ++row) {
        const int32_t event_type =
            arguments.event_state.typed_event_type[event_offset + row];
        if (
            event_type
            == persistent_birth_events::kBirthCommittedEventType
        ) {
            ++births;
        } else if (
            event_type == persistent_birth_events::kFailedBirthEventType
        ) {
            ++failed_births;
        } else if (
            event_type
            == persistent_lifecycle_events::kDeathCommittedEventType
        ) {
            ++deaths;
        }
    }

    double energy_sum = 0.0;
    double policy_energy_live_sum = 0.0;
    double energy_collected_exact_sum = 0.0;
    const int64_t cell_capacity_offset = cell * arguments.capacity;
    for (int64_t slot = 0; slot < arguments.capacity; ++slot) {
        const double slot_energy = static_cast<double>(
            arguments.energy[cell_capacity_offset + slot]
        );
        if (!isfinite(slot_energy)) {
            invalidate();
            return;
        }
        energy_sum += slot_energy;
        const double collected = arguments.energy_collected_exact[
            cell_capacity_offset + slot
        ];
        if (!isfinite(collected)) {
            invalidate();
            return;
        }
        energy_collected_exact_sum += collected;
        if (arguments.alive_mask[cell_capacity_offset + slot]) {
            const double policy_energy = arguments.policy_energy_observation[
                cell_capacity_offset + slot
            ];
            if (!isfinite(policy_energy)) {
                invalidate();
                return;
            }
            policy_energy_live_sum += policy_energy;
        }
    }

    double patch_food_stock = 0.0;
    if (
        arguments.patch_p2_context.state.patch_food_amount == nullptr
        || arguments.height <= 0
        || arguments.width <= 0
    ) {
        invalidate();
        return;
    }
    const int64_t grid_cells = arguments.height * arguments.width;
    const int64_t cell_grid_offset = cell * grid_cells;
    for (int64_t grid_cell = 0; grid_cell < grid_cells; ++grid_cell) {
        const double stock = static_cast<double>(
            arguments.patch_p2_context.state.patch_food_amount[
                cell_grid_offset + grid_cell
            ]
        );
        if (!isfinite(stock) || stock < 0.0) {
            invalidate();
            return;
        }
        patch_food_stock += stock;
    }

    summary[kGenerationSummaryEventCursorPreTurnover] =
        static_cast<double>(absolute_cursor_pre_turnover);
    summary[kGenerationSummaryEventCursorEnd] =
        static_cast<double>(absolute_cursor_end);
    summary[kGenerationSummaryAliveAfterTurnover] = static_cast<double>(
        arguments.active_count_state[cell]
    );
    summary[kGenerationSummaryEnergySumAfterTurnover] = energy_sum;
    summary[kGenerationSummaryBirths] = static_cast<double>(births);
    summary[kGenerationSummaryDeaths] = static_cast<double>(deaths);
    summary[kGenerationSummaryFailedBirths] =
        static_cast<double>(failed_births);
    summary[kGenerationSummaryActionTotal] = static_cast<double>(action_total);
    summary[kGenerationSummaryMoveTotal] = static_cast<double>(move_total);
    summary[kGenerationSummaryGatherTotal] = static_cast<double>(gather_total);
    summary[kGenerationSummaryMovedTotal] = static_cast<double>(moved_total);
    summary[kGenerationSummaryGatheredTotal] =
        static_cast<double>(gathered_total);
    summary[kGenerationSummaryPeakInjury] = fmax(
        summary[kGenerationSummaryPeakInjury],
        static_cast<double>(peak_injury)
    );
    summary[kGenerationSummaryPolicyLoopTickCalls] =
        static_cast<double>(arguments.ticks);
    summary[kGenerationSummaryPolicyRefillCalls] =
        static_cast<double>(policy_refill_calls);
    summary[kGenerationSummaryPolicyRefillRows] =
        static_cast<double>(policy_refill_rows);
    summary[kGenerationSummaryPolicyCacheHitCalls] =
        static_cast<double>(arguments.ticks - policy_refill_calls);
    summary[kGenerationSummaryPolicyCacheRowsRequested] =
        static_cast<double>(action_total);
    summary[kGenerationSummaryPolicyCacheRowsHit] =
        static_cast<double>(policy_cache_rows_hit);
    summary[kGenerationSummaryPolicyCacheRowsMissed] =
        static_cast<double>(action_total - policy_cache_rows_hit);
    summary[kGenerationSummaryPolicyCacheRowsRefilled] =
        static_cast<double>(policy_refill_rows);
    summary[kGenerationSummaryPolicyCacheFullRefillTicks] =
        static_cast<double>(policy_cache_full_refill_ticks);
    summary[kGenerationSummaryPolicyCachePartialRefillTicks] =
        static_cast<double>(policy_cache_partial_refill_ticks);
    summary[kGenerationSummaryPatchFoodStockAfterTurnover] =
        patch_food_stock;
    summary[kGenerationSummaryEnergyCollectedExactSumAfterTurnover] =
        energy_collected_exact_sum;
    summary[kGenerationSummaryPolicyEnergyLiveSumAfterTurnover] =
        policy_energy_live_sum;
}

__global__ void compact_checkpoint_active_slots_projection_kernel_v1(
    const int64_t* const active_agent_ids,
    const int32_t* const active_agent_slots,
    const int64_t* const active_count_state,
    const int64_t* const slot_to_agent_id,
    const bool* const alive_mask,
    int32_t* const active_agent_slots_out,
    int64_t* const agent_id_to_slot_prefix_length_out,
    const int64_t active_capacity,
    const int64_t slot_count
) {
    const int64_t cell = static_cast<int64_t>(blockIdx.x);
    if (threadIdx.x != 0) {
        return;
    }
    const int64_t active_count = active_count_state[cell];
    const int64_t active_offset = cell * active_capacity;
    const int64_t slot_offset = cell * slot_count;
    for (int64_t row = 0; row < active_capacity; ++row) {
        active_agent_slots_out[active_offset + row] = -1;
    }
    const active_identity::ActiveIdentityPairsV1 pairs{
        active_agent_ids + active_offset,
        active_agent_slots + active_offset,
        slot_to_agent_id + slot_offset,
        alive_mask + slot_offset,
        active_count,
        active_capacity,
        slot_count,
    };
    if (!active_identity::valid_active_identity_pairs_v1(pairs)) {
        agent_id_to_slot_prefix_length_out[cell] = -1;
        return;
    }
    for (int64_t row = 0; row < active_count; ++row) {
        active_agent_slots_out[active_offset + row] =
            active_agent_slots[active_offset + row];
    }
    agent_id_to_slot_prefix_length_out[cell] = active_count;
}

__global__ void refresh_patch_density_after_turnover_kernel_v1(
    const int32_t* const agent_grid,
    const int32_t* const controller_status,
    const int32_t* const turnover_status,
    const FlatPatchP2Context patch_p2_context,
    const cross_path_resident_control_v1::FlatCrossPathResidentControlV1
        cross_path_control,
    const int64_t height,
    const int64_t width
) {
    const int64_t cell = static_cast<int64_t>(blockIdx.x);
    if (
        threadIdx.x != 0
        || (
            cross_path_resident_control_v1::has_cross_path_authority_v1(
                cross_path_control
            )
            && cross_path_resident_control_v1::ordinary_live_birth_cell_v1(
                cross_path_control, cell
            )
        )
        || controller_status[cell] != 0
        || turnover_status[cell] != 0
    ) {
        return;
    }
    const int64_t cell_grid_offset = cell * height * width;
    persistent_patch_p2::refresh_patch_density_exact(
        agent_grid + cell_grid_offset,
        patch_p2_context.state.patch_density + cell_grid_offset,
        patch_p2_context.config
    );
}

}  // namespace

void launch_compact_checkpoint_active_slots_projection_v1(
    const int64_t* const active_agent_ids,
    const int32_t* const active_agent_slots,
    const int64_t* const active_count_state,
    const int64_t* const slot_to_agent_id,
    const bool* const alive_mask,
    int32_t* const active_agent_slots_out,
    int64_t* const agent_id_to_slot_prefix_length_out,
    const int64_t cells,
    const int64_t active_capacity,
    const int64_t slot_count,
    cudaStream_t stream
) {
    compact_checkpoint_active_slots_projection_kernel_v1<<<
        static_cast<unsigned int>(cells), 128, 0, stream
    >>>(
        active_agent_ids,
        active_agent_slots,
        active_count_state,
        slot_to_agent_id,
        alive_mask,
        active_agent_slots_out,
        agent_id_to_slot_prefix_length_out,
        active_capacity,
        slot_count
    );
}

void maybe_open_inter_tick_control_v2(
    const std::function<void(int64_t, int64_t, int64_t)>& inter_tick_control,
    const int64_t generation,
    const int64_t start_tick,
    const int64_t tick,
    const int64_t ticks,
    cudaStream_t stream
) {
    if (tick != 0 || tick + 1 >= ticks || !inter_tick_control) {
        return;
    }
    const cudaError_t synchronization_status = cudaStreamSynchronize(stream);
    if (synchronization_status != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA inter-tick control synchronization failed: ")
            + cudaGetErrorString(synchronization_status)
        );
    }
    (void)start_tick;
    const int64_t completed_tick = tick;
    inter_tick_control(generation, completed_tick, completed_tick + 1);
}

cudaError_t maybe_refill_gaussian_segment_v1(
    const std::function<void(int64_t, int64_t, int64_t)>& gaussian_segment_refill,
    const int64_t generation,
    const int64_t completed_tick,
    const int64_t next_tick,
    cudaStream_t stream,
    int64_t* host_synchronizations_inside_loop
#if EIEL_OPTION_D_M0_TELEMETRY_ENABLED == 1
    , std::vector<int64_t>* pre_refill_synchronization_time_ns,
    std::vector<int64_t>* post_copy_synchronization_time_ns
#endif
) {
    if (!gaussian_segment_refill) {
        return cudaSuccess;
    }
#if EIEL_OPTION_D_M0_TELEMETRY_ENABLED == 1
    const auto pre_refill_synchronization_started =
        std::chrono::steady_clock::now();
#endif
    cudaError_t synchronization_status = cudaStreamSynchronize(stream);
#if EIEL_OPTION_D_M0_TELEMETRY_ENABLED == 1
    pre_refill_synchronization_time_ns->push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()
                - pre_refill_synchronization_started
        ).count()
    );
#endif
    if (synchronization_status != cudaSuccess) {
        return synchronization_status;
    }
    *host_synchronizations_inside_loop += 1;
    gaussian_segment_refill(generation, completed_tick, next_tick);
#if EIEL_OPTION_D_M0_TELEMETRY_ENABLED == 1
    const auto post_copy_synchronization_started =
        std::chrono::steady_clock::now();
#endif
    synchronization_status = cudaStreamSynchronize(stream);
#if EIEL_OPTION_D_M0_TELEMETRY_ENABLED == 1
    post_copy_synchronization_time_ns->push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()
                - post_copy_synchronization_started
        ).count()
    );
#endif
    if (synchronization_status != cudaSuccess) {
        return synchronization_status;
    }
    *host_synchronizations_inside_loop += 1;
    return cudaSuccess;
}

cudaError_t maybe_refill_policy_birth_segment_v1(
    const std::function<void(int64_t, int64_t, int64_t)>&
        policy_birth_segment_refill,
    const int64_t generation,
    const int64_t completed_tick,
    const int64_t next_tick,
    cudaStream_t stream,
    int64_t* host_synchronizations_inside_loop
) {
    if (!policy_birth_segment_refill) {
        return cudaSuccess;
    }
    cudaError_t synchronization_status = cudaStreamSynchronize(stream);
    if (synchronization_status != cudaSuccess) {
        return synchronization_status;
    }
    *host_synchronizations_inside_loop += 1;
    policy_birth_segment_refill(generation, completed_tick, next_tick);
    synchronization_status = cudaStreamSynchronize(stream);
    if (synchronization_status != cudaSuccess) {
        return synchronization_status;
    }
    *host_synchronizations_inside_loop += 1;
    return cudaSuccess;
}

ControllerRunReceipt run_flat_dynamic_policy_food_flattened_controller_v1(
    int64_t* active_agent_ids,
    int32_t* active_agent_slots,
    int64_t* active_count_state,
    int64_t* slot_to_agent_id,
    int32_t* agent_id_to_slot,
    bool* alive_mask,
    int64_t* slot_tokens,
    int32_t* action_cache_actions,
    int64_t* action_cache_token,
    int32_t* action_cache_cursor,
    int32_t* action_cache_next,
    int32_t* pos_y,
    int32_t* pos_x,
    int32_t* agent_grid,
    int32_t* food_grid,
    float* energy,
    float* health,
    float* injury,
    double* policy_energy_observation,
    int32_t* rest_streak,
    int64_t* pending_free_agent_ids,
    int64_t* pending_free_slots,
    int64_t* pending_free_count,
    float* energy_collected,
    double* energy_collected_exact,
    int32_t* last_action,
    int32_t* last_last_action,
    int32_t* action_switches,
    int32_t* action_oscillations,
    int32_t* episode0_switches,
    int32_t* episode0_oscillations,
    int32_t* survival_steps,
    float* mlp_pack,
    float* drives,
    float* attack_bias,
    float* action_temperature,
    float* drive_logit_gain,
    float* food_gained_scratch,
    int64_t* ordered_agent_ids_out,
    int32_t* ordered_slots_out,
    float* observations_out,
    float* logits_out,
    int32_t* row_actions_out,
    int32_t* actions_by_slot_out,
    float* pre_energy_out,
    float* pre_health_out,
    float* pre_injury_out,
    int32_t* pre_survival_steps_out,
    int64_t* active_count_out,
    int64_t* policy_refill_rows_out,
    int64_t* policy_cache_hit_rows_out,
    int64_t* runtime_bookkeeping_by_tick_out,
    int64_t* death_agent_ids_out,
    int64_t* death_slots_out,
    int32_t* death_reason_codes_out,
    int64_t* death_count_out,
    int64_t* survivor_agent_ids_out,
    int64_t* survivor_count_out,
    int32_t* lifecycle_status_out,
    int32_t* status_out,
    int64_t* first_failure_records,
    double* generation_summary_scalar_out,
    const FlatFullTickState full_tick_state,
    const FlatFullTickOutputs full_tick_outputs,
    const FlatEventState event_state,
    const FlatOrdinaryBirthState ordinary_birth_state,
    const FlatOrdinaryBirthOutputs ordinary_birth_outputs,
    const FlatFlattenedBirthScratch ordinary_birth_scratch,
    const FlatOrdinaryBirthConfig ordinary_birth_config,
    const flattened_turnover_v1::FlatTurnoverMutableTensorTable turnover_tensors,
    const flattened_turnover_v1::FlatTurnoverScratch turnover_scratch,
    const flattened_turnover_v1::FlatTurnoverOutputs turnover_outputs,
    const flattened_turnover_v1::FlatTurnoverConfig turnover_config,
    const FlatMechanicsConfig mechanics_config,
    const FlatPatchP2Context patch_p2_context,
    const FlatCompactSummaryContextV1 compact_summary_context,
    const int64_t cells,
    const int64_t ticks,
    const int64_t output_ticks_per_cell,
    const int64_t resident_ticks_per_cell,
    const int64_t active_capacity,
    const int64_t agent_id_capacity,
    const int64_t capacity,
    const int64_t pending_capacity,
    const int64_t height,
    const int64_t width,
    const int64_t view_radius,
    const int64_t observation_size,
    const int64_t hidden_size,
    const int64_t action_count,
    const int64_t packed_width,
    const int64_t drive_width,
    const int64_t cache_capacity,
    const int64_t cache_width,
    const bool deterministic_actions,
    const int64_t max_food,
    const int64_t effective_food_spawn_count,
    const double max_energy,
    const double start_energy,
    const double start_health,
    const double* move_cost_by_cell,
    const double gather_cost,
    const double rest_cost,
    const double attack_cost,
    const double action_cost_mult,
    const double rest_cost_mult,
    const double food_energy,
    const int64_t episode0_k,
    const float w_hunger_gather,
    const float w_pain_rest,
    const float w_fatigue_rest,
    const float w_pain_attack,
    const float w_fear_attack,
    const std::function<void(int64_t, int64_t)>& exact_cpu_policy_refill,
    const std::function<void(int64_t, int64_t)>& hapb_ordinary_birth_commit,
    const std::function<void(int64_t, int64_t, int64_t)>&
        policy_birth_segment_refill,
    const std::function<void(int64_t, int64_t, int64_t)>& gaussian_segment_refill,
    const std::function<void(int64_t, int64_t, int64_t)>& inter_tick_control,
    const std::function<void()>& hapb_turnover_commit,
    cudaStream_t stream,
    const bool synchronize_at_end
) {
    const Phase2KernelArguments phase_2_arguments{
        active_agent_ids,
        active_agent_slots,
        active_count_state,
        slot_to_agent_id,
        agent_id_to_slot,
        slot_tokens,
        alive_mask,
        pos_y,
        pos_x,
        agent_grid,
        energy,
        health,
        injury,
        policy_energy_observation,
        rest_streak,
        action_cache_actions,
        action_cache_token,
        action_cache_cursor,
        action_cache_next,
        last_action,
        last_last_action,
        action_switches,
        action_oscillations,
        episode0_switches,
        episode0_oscillations,
        survival_steps,
        mlp_pack,
        drives,
        attack_bias,
        action_temperature,
        drive_logit_gain,
        energy_collected,
        energy_collected_exact,
        survivor_agent_ids_out,
        survivor_count_out,
        lifecycle_status_out,
        status_out,
        first_failure_records,
        generation_summary_scalar_out,
        full_tick_state,
        full_tick_outputs,
        event_state,
        ordinary_birth_state,
        ordinary_birth_outputs,
        ordinary_birth_scratch,
        ordinary_birth_config,
        mechanics_config,
        compact_summary_context,
        patch_p2_context,
        ticks,
        output_ticks_per_cell,
        active_capacity,
        agent_id_capacity,
        capacity,
        height,
        width,
        cache_capacity,
        cache_width,
        drive_width,
        packed_width,
        static_cast<float>(max_energy),
        static_cast<float>(start_health),
    };
    Phase1KernelArguments phase_1_arguments{
        active_agent_ids,
        active_agent_slots,
        active_count_state,
        slot_to_agent_id,
        agent_id_to_slot,
        alive_mask,
        slot_tokens,
        action_cache_actions,
        action_cache_token,
        action_cache_cursor,
        action_cache_next,
        pos_y,
        pos_x,
        agent_grid,
        food_grid,
        energy,
        health,
        injury,
        policy_energy_observation,
        rest_streak,
        pending_free_agent_ids,
        pending_free_slots,
        pending_free_count,
        energy_collected,
        energy_collected_exact,
        last_action,
        last_last_action,
        action_switches,
        action_oscillations,
        episode0_switches,
        episode0_oscillations,
        survival_steps,
        mlp_pack,
        drives,
        attack_bias,
        action_temperature,
        drive_logit_gain,
        food_gained_scratch,
        ordered_agent_ids_out,
        ordered_slots_out,
        observations_out,
        logits_out,
        row_actions_out,
        actions_by_slot_out,
        pre_energy_out,
        pre_health_out,
        pre_injury_out,
        pre_survival_steps_out,
        active_count_out,
        policy_refill_rows_out,
        policy_cache_hit_rows_out,
        runtime_bookkeeping_by_tick_out,
        death_agent_ids_out,
        death_slots_out,
        death_reason_codes_out,
        death_count_out,
        survivor_agent_ids_out,
        survivor_count_out,
        lifecycle_status_out,
        status_out,
        first_failure_records,
        generation_summary_scalar_out,
        full_tick_state,
        full_tick_outputs,
        event_state,
        ordinary_birth_state,
        ordinary_birth_outputs,
        ordinary_birth_scratch,
        ordinary_birth_config,
        mechanics_config,
        patch_p2_context,
        compact_summary_context,
        turnover_config.enabled,
        ticks,
        output_ticks_per_cell,
        resident_ticks_per_cell,
        active_capacity,
        agent_id_capacity,
        capacity,
        pending_capacity,
        height,
        width,
        view_radius,
        observation_size,
        hidden_size,
        action_count,
        packed_width,
        drive_width,
        cache_capacity,
        cache_width,
        deterministic_actions,
        max_food,
        effective_food_spawn_count,
        max_energy,
        start_health,
        move_cost_by_cell,
        gather_cost,
        rest_cost,
        attack_cost,
        action_cost_mult,
        rest_cost_mult,
        food_energy,
        episode0_k,
        w_hunger_gather,
        w_pain_rest,
        w_fatigue_rest,
        w_pain_attack,
        w_fear_attack,
        TickCoreExecutionModeV1::kFull,
    };

    int64_t host_synchronizations_inside_loop = 0;
#if EIEL_OPTION_D_M0_TELEMETRY_ENABLED == 1
    std::vector<int64_t> gaussian_pre_refill_synchronization_time_ns;
    std::vector<int64_t> gaussian_post_copy_synchronization_time_ns;
    gaussian_pre_refill_synchronization_time_ns.reserve(
        static_cast<std::size_t>(ticks)
    );
    gaussian_post_copy_synchronization_time_ns.reserve(
        static_cast<std::size_t>(ticks)
    );
#endif
    int64_t phase_1_launches = 0;
    int64_t birth_prepare_launches = 0;
    int64_t birth_commit_launches = 0;
    int64_t birth_prepare_commit_fused_launches = 0;
    for (int64_t tick = 0; tick < ticks; ++tick) {
        const bool use_exact_cpu_policy =
            static_cast<bool>(exact_cpu_policy_refill);
        phase_1_arguments.execution_mode = use_exact_cpu_policy
            ? TickCoreExecutionModeV1::kPrepareExactCpuPolicy
            : TickCoreExecutionModeV1::kFull;
        flattened_controller_tick_core_and_birth_plan_kernel_v1<<<
            static_cast<unsigned int>(cells),
            kThreadsPerCell,
            0,
            stream
        >>>(
            phase_1_arguments,
            tick
        );
        ++phase_1_launches;
        if (use_exact_cpu_policy) {
            const cudaError_t policy_boundary_status =
                cudaStreamSynchronize(stream);
            if (policy_boundary_status != cudaSuccess) {
                return {
                    1,
                    tick,
                    phase_1_launches,
                    birth_prepare_launches,
                    birth_commit_launches,
                    birth_prepare_commit_fused_launches,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    host_synchronizations_inside_loop + 1,
                    0,
                    static_cast<int32_t>(policy_boundary_status),
                };
            }
            ++host_synchronizations_inside_loop;
            exact_cpu_policy_refill(
                tick,
                output_ticks_per_cell == 1 ? 0 : tick
            );
            ++host_synchronizations_inside_loop;
            phase_1_arguments.execution_mode =
                TickCoreExecutionModeV1::kResumeAfterExactCpuPolicy;
            flattened_controller_tick_core_and_birth_plan_kernel_v1<<<
                static_cast<unsigned int>(cells),
                kThreadsPerCell,
                0,
                stream
            >>>(
                phase_1_arguments,
                tick
            );
            ++phase_1_launches;
        }
        if constexpr (EIEL_OPTION_D_M9_BIRTH_PREPARE_COMMIT_FUSED_ENABLED) {
            flattened_controller_birth_prepare_and_transaction_kernel_v1<<<
                static_cast<unsigned int>(cells),
                kBirthThreadsPerCell,
                0,
                stream
            >>>(
                phase_2_arguments,
                tick
            );
            ++birth_prepare_commit_fused_launches;
        } else {
            flattened_controller_birth_prepare_kernel_v1<<<
                static_cast<unsigned int>(cells),
                kBirthThreadsPerCell,
                0,
                stream
            >>>(
                phase_2_arguments,
                tick
            );
            ++birth_prepare_launches;
            if (hapb_ordinary_birth_commit) {
                const cudaError_t birth_boundary_status = cudaStreamSynchronize(stream);
                if (birth_boundary_status != cudaSuccess) {
                    return {
                        1,
                        tick,
                        phase_1_launches,
                        birth_prepare_launches,
                        birth_commit_launches,
                        birth_prepare_commit_fused_launches,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        host_synchronizations_inside_loop + 1,
                        0,
                        static_cast<int32_t>(birth_boundary_status),
                    };
                }
                ++host_synchronizations_inside_loop;
                hapb_ordinary_birth_commit(
                    tick,
                    output_ticks_per_cell == 1 ? 0 : tick
                );
                ++host_synchronizations_inside_loop;
            }
            flattened_controller_birth_transaction_and_settlement_kernel_v1<<<
                static_cast<unsigned int>(cells),
                kBirthThreadsPerCell,
                0,
                stream
            >>>(
                phase_2_arguments,
                tick
            );
            ++birth_commit_launches;
        }
        if (tick + 1 < ticks) {
            const cudaError_t policy_birth_refill_status =
                maybe_refill_policy_birth_segment_v1(
                    policy_birth_segment_refill,
                    mechanics_config.generation,
                    tick,
                    tick + 1,
                    stream,
                    &host_synchronizations_inside_loop
                );
            if (policy_birth_refill_status != cudaSuccess) {
                return {
                    1,
                    tick + 1,
                    phase_1_launches,
                    birth_prepare_launches,
                    birth_commit_launches,
                    birth_prepare_commit_fused_launches,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    host_synchronizations_inside_loop,
                    0,
                    static_cast<int32_t>(policy_birth_refill_status),
                };
            }
                const cudaError_t gaussian_refill_status =
                    maybe_refill_gaussian_segment_v1(
                    gaussian_segment_refill,
                    mechanics_config.generation,
                    tick,
                    tick + 1,
                    stream,
                    &host_synchronizations_inside_loop
#if EIEL_OPTION_D_M0_TELEMETRY_ENABLED == 1
                    , &gaussian_pre_refill_synchronization_time_ns,
                    &gaussian_post_copy_synchronization_time_ns
#endif
                );
            if (gaussian_refill_status != cudaSuccess) {
                return {
                    1,
                    tick + 1,
                    phase_1_launches,
                    birth_prepare_launches,
                    birth_commit_launches,
                    birth_prepare_commit_fused_launches,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    host_synchronizations_inside_loop,
                    0,
                    static_cast<int32_t>(gaussian_refill_status),
                };
            }
        }
        maybe_open_inter_tick_control_v2(
            inter_tick_control,
            mechanics_config.generation,
            mechanics_config.start_tick,
            tick,
            ticks,
            stream
        );
    }

    flattened_turnover_v1::TurnoverLaunchReceipt turnover_receipt{};
    if (turnover_config.enabled) {
        const cudaError_t policy_birth_turnover_refill_status =
            maybe_refill_policy_birth_segment_v1(
                policy_birth_segment_refill,
                mechanics_config.generation,
                ticks - 1,
                ticks,
                stream,
                &host_synchronizations_inside_loop
            );
        if (policy_birth_turnover_refill_status != cudaSuccess) {
            return {
                1,
                ticks,
                phase_1_launches,
                birth_prepare_launches,
                birth_commit_launches,
                birth_prepare_commit_fused_launches,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                host_synchronizations_inside_loop,
                0,
                static_cast<int32_t>(policy_birth_turnover_refill_status),
            };
        }
        const cudaError_t gaussian_turnover_refill_status =
            maybe_refill_gaussian_segment_v1(
                gaussian_segment_refill,
                mechanics_config.generation,
                ticks - 1,
                ticks,
                stream,
                &host_synchronizations_inside_loop
#if EIEL_OPTION_D_M0_TELEMETRY_ENABLED == 1
                , &gaussian_pre_refill_synchronization_time_ns,
                &gaussian_post_copy_synchronization_time_ns
#endif
            );
        if (gaussian_turnover_refill_status != cudaSuccess) {
            return {
                1,
                ticks,
                phase_1_launches,
                birth_prepare_launches,
                birth_commit_launches,
                birth_prepare_commit_fused_launches,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                host_synchronizations_inside_loop,
                0,
                static_cast<int32_t>(gaussian_turnover_refill_status),
            };
        }
        const flattened_turnover_v1::FlatTurnoverExecutionConfig
            turnover_execution{
                mechanics_config.generation,
                mechanics_config.start_tick,
                ticks,
                capacity,
                agent_id_capacity,
                pending_capacity < flattened_turnover_v1::kMaximumPending
                    ? pending_capacity
                    : flattened_turnover_v1::kMaximumPending,
                height,
                width,
                start_energy,
                static_cast<float>(start_health),
                static_cast<float>(max_energy),
                ordinary_birth_config.sensor_radius_max,
                cache_width,
                drive_width,
                full_tick_state.chemistry_width,
                packed_width,
                ordinary_birth_config.hmem_width,
                ordinary_birth_config.gauss_tape_capacity,
                ordinary_birth_config.policy_tape_capacity,
                ordinary_birth_config.policy_mutation_tape_width,
                ordinary_birth_config.torch_state_bytes,
                ordinary_birth_config.env_policy_alias,
                false,
                false,
                nullptr,
                0,
                ordinary_birth_config.rng_semantics_version,
                ordinary_birth_config.world_uid,
                ordinary_birth_config.world_uid_by_cell,
                160,
                ordinary_birth_config.rng_namespace_key,
                ordinary_birth_config.rng_semantic_epoch,
            };
        turnover_receipt = flattened_turnover_v1::run_flattened_turnover_v1(
            stream,
            cells,
            turnover_tensors,
            turnover_scratch,
            turnover_outputs,
            turnover_config,
            turnover_execution,
            ordinary_birth_state,
            ordinary_birth_config,
            {},
            hapb_turnover_commit,
            &host_synchronizations_inside_loop
        );
    }

    cudaError_t patch_density_refresh_launch_status = cudaSuccess;
    const bool patch_density_refresh_enabled =
        turnover_config.enabled
        && patch_p2_context.enabled
        && patch_p2_context.mode == FlatPatchP2Mode::kFullPatchP2;
    if (patch_density_refresh_enabled) {
        refresh_patch_density_after_turnover_kernel_v1<<<
            static_cast<unsigned int>(cells), 1, 0, stream
        >>>(
            agent_grid,
            status_out,
            turnover_outputs.status,
            patch_p2_context,
            turnover_config.cross_path_control,
            height,
            width
        );
        patch_density_refresh_launch_status = cudaPeekAtLastError();
    }

    cudaError_t generation_summary_launch_status = cudaSuccess;
    if (
        generation_summary_scalar_out != nullptr
        || compact_summary_context.enabled
    ) {
        GenerationSummaryFinalizeArgumentsV1 generation_summary_arguments{
            generation_summary_scalar_out,
            active_count_state,
            alive_mask,
            energy,
            policy_energy_observation,
            energy_collected_exact,
            ordered_slots_out,
            actions_by_slot_out,
            pre_injury_out,
            active_count_out,
            policy_refill_rows_out,
            policy_cache_hit_rows_out,
            runtime_bookkeeping_by_tick_out,
            status_out,
            turnover_outputs.status,
            turnover_config.enabled,
            turnover_config.cross_path_control,
            event_state,
	        compact_summary_context,
            patch_p2_context,
	        ticks,
	        output_ticks_per_cell,
	        resident_ticks_per_cell,
            capacity,
            height,
            width,
        };
        void* generation_summary_kernel_arguments[] = {
            &generation_summary_arguments,
        };
        generation_summary_launch_status = cudaLaunchKernel(
            reinterpret_cast<const void*>(
                finalize_generation_summary_scalar_v1
            ),
            dim3(static_cast<unsigned int>(cells)),
            dim3(1),
            generation_summary_kernel_arguments,
            0,
            stream
        );
    }

    const cudaError_t synchronization_status = synchronize_at_end
        ? cudaStreamSynchronize(stream)
        : cudaGetLastError();
    const int32_t final_status = turnover_receipt.cuda_status
            != static_cast<int32_t>(cudaSuccess)
        ? turnover_receipt.cuda_status
        : (patch_density_refresh_launch_status != cudaSuccess
               ? static_cast<int32_t>(patch_density_refresh_launch_status)
               : (generation_summary_launch_status != cudaSuccess
               ? static_cast<int32_t>(generation_summary_launch_status)
               : static_cast<int32_t>(synchronization_status)));
    return {
        synchronize_at_end ? 1 : 0,
        ticks,
        phase_1_launches,
        birth_prepare_launches,
        birth_commit_launches,
        birth_prepare_commit_fused_launches,
        turnover_receipt.shadow_plan_launches,
        turnover_receipt.cull_refill_plan_launches,
        turnover_receipt.commit_launches,
        turnover_receipt.publish_launches,
        patch_density_refresh_enabled ? 1 : 0,
        0,
        generation_summary_scalar_out != nullptr
                || compact_summary_context.enabled
            ? 1
            : 0,
        host_synchronizations_inside_loop,
        synchronize_at_end ? 1 : 0,
        final_status,
#if EIEL_OPTION_D_M0_TELEMETRY_ENABLED == 1
        std::move(gaussian_pre_refill_synchronization_time_ns),
        std::move(gaussian_post_copy_synchronization_time_ns),
#endif
    };
}

KernelResourceReceipt query_flattened_controller_tick_core_resources_v1() {
    cudaFuncAttributes attributes{};
    cudaError_t error = cudaFuncGetAttributes(
        &attributes,
        flattened_controller_tick_core_and_birth_plan_kernel_v1
    );
    if (error != cudaSuccess) {
        return {-1, -1, -1, -1, -1, -1, -1, -1};
    }
    int32_t occupancy = -1;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy,
        flattened_controller_tick_core_and_birth_plan_kernel_v1,
        kThreadsPerCell,
        0
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

KernelResourceReceipt query_flattened_controller_birth_prepare_resources_v1() {
    cudaFuncAttributes attributes{};
    cudaError_t error = cudaFuncGetAttributes(
        &attributes,
        flattened_controller_birth_prepare_kernel_v1
    );
    if (error != cudaSuccess) {
        return {-1, -1, -1, -1, -1, -1, -1, -1};
    }
    int32_t occupancy = -1;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy,
        flattened_controller_birth_prepare_kernel_v1,
        kBirthThreadsPerCell,
        0
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

KernelResourceReceipt query_flattened_controller_birth_commit_resources_v1() {
    cudaFuncAttributes attributes{};
    cudaError_t error = cudaFuncGetAttributes(
        &attributes,
        flattened_controller_birth_transaction_and_settlement_kernel_v1
    );
    if (error != cudaSuccess) {
        return {-1, -1, -1, -1, -1, -1, -1, -1};
    }
    int32_t occupancy = -1;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy,
        flattened_controller_birth_transaction_and_settlement_kernel_v1,
        kBirthThreadsPerCell,
        0
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

KernelResourceReceipt
query_flattened_controller_birth_prepare_commit_resources_v1() {
    cudaFuncAttributes attributes{};
    cudaError_t error = cudaFuncGetAttributes(
        &attributes,
        flattened_controller_birth_prepare_and_transaction_kernel_v1
    );
    if (error != cudaSuccess) {
        return {-1, -1, -1, -1, -1, -1, -1, -1};
    }
    int32_t occupancy = -1;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy,
        flattened_controller_birth_prepare_and_transaction_kernel_v1,
        kBirthThreadsPerCell,
        0
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

KernelResourceReceipt query_generation_summary_finalize_resources_v1() {
    cudaFuncAttributes attributes{};
    cudaError_t error = cudaFuncGetAttributes(
        &attributes,
        finalize_generation_summary_scalar_v1
    );
    if (error != cudaSuccess) {
        return {-1, -1, -1, -1, -1, -1, -1, -1};
    }
    int32_t occupancy = -1;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy,
        finalize_generation_summary_scalar_v1,
        1,
        0
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

}  // namespace eiel::cuda_architecture_b::flattened_controller_v1
