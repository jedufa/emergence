#pragma once

#include <cuda_runtime.h>

#include <torch/types.h>

#include <cstdint>
#include <functional>
#if EIEL_OPTION_D_M0_TELEMETRY_ENABLED == 1
#include <vector>
#endif

#include "cuda_architecture_b_cross_path_resident_control_v1.cuh"
#include "cuda_architecture_b_flattened_birth_scratch_v1.cuh"
#include "cuda_architecture_b_flattened_turnover_v1.cuh"
#include "cuda_architecture_b_compact_summary_reporting_authority_v1.cuh"
#include "cuda_architecture_b_persistent_patch_p2.cuh"
#include "genome_upgrade_v2_q10_v1.cuh"

namespace eiel::cuda_architecture_b::flattened_controller_v1 {

// HAPB prepare/resume policy transaction: classifier and mapped-tile logits
// have already completed.  This reuses the resident serial Gumbel settlement
// on the original per-cell row coordinates and never accepts an MLP bank.
torch::Tensor hapb_settle_prepared_policy_rows_v1(
    const torch::Tensor& ordered_slots,
    const torch::Tensor& slot_tokens,
    const torch::Tensor& action_cache_actions,
    const torch::Tensor& action_cache_token,
    const torch::Tensor& action_cache_cursor,
    const torch::Tensor& action_cache_next,
    const torch::Tensor& logits,
    const torch::Tensor& row_actions,
    const torch::Tensor& stable_miss_rows,
    const torch::Tensor& active_row_counts,
    const torch::Tensor& miss_row_counts,
    const torch::Tensor& policy_pcg64_state_words,
    const torch::Tensor& policy_pcg64_buffer_words,
    float w_hunger_gather,
    float w_pain_rest,
    float w_fatigue_rest,
    float w_pain_attack,
    float w_fear_attack
);

constexpr int kThreadsPerCell = 128;
constexpr int kBirthThreadsPerCell = 1;
constexpr int64_t kBirthScratchBytesPerCell =
    flattened_birth_scratch_v1::kBytesPerCell;

constexpr int64_t kFirstFailureNoFailureSentinelV1 = -1;

enum FirstFailureRecordColumnV1 : int64_t {
    kFirstFailureGeneration = 0,
    kFirstFailureTick,
    kFirstFailureStage,
    kFirstFailureStatus,
    kFirstFailureCode,
    kFirstFailureCell,
    kFirstFailureAttemptedIndex,
    kFirstFailureRequiredCapacity,
    kFirstFailureAvailableCapacity,
    kFirstFailureSequence,
    kFirstFailureRecordWidthV1,
};
static_assert(kFirstFailureRecordWidthV1 == 10);

enum ControllerFailureStageV1 : int64_t {
    kControllerFailureStageTickCoreAndBirthPlan = 1,
    kControllerFailureStageBirthPrepare = 2,
    kControllerFailureStageBirthTransactionAndSettlement = 3,
    kControllerFailureStageBirthEventAppend = 4,
};

constexpr int64_t kFirstFailureEventCapacityExceededV1 = 1;

// The tick-core preflight keeps its public status code at 1.  These bits are
// diagnostic-only detail carried in the existing first-failure `failure_code`
// column; they do not add fields to, or change the shape of, the native ABI.
enum TickCorePredicateBitV1 : int64_t {
    kTickCorePredicateHiddenSizeExceedsMax = 0,
    kTickCorePredicateObservationSizeExceedsMax,
    kTickCorePredicateActionCountExceedsMax,
    kTickCorePredicateObservationSizeMismatch,
    kTickCorePredicateGmnwContextInvalid,
    kTickCorePredicateGruContextInvalid,
    kTickCorePredicateGmnwControlsInvalid,
    kTickCorePredicateGmnwTraceInvalid,
    kTickCorePredicateGmnwDecisionTapeInvalid,
    kTickCorePredicateGmnwSelectionMetricsInvalid,
    kTickCorePredicatePatchViewRadiusInvalid,
    kTickCorePredicateModularContextInvalid,
    kTickCorePredicateWdlRequiresPatch,
    kTickCorePredicateEedRequiresPatch,
    kTickCorePredicateEedContextInvalid,
    kTickCorePredicateWdlTickWindowInvalid,
    kTickCorePredicateWdlContextInvalid,
    kTickCorePredicatePatchContextInvalid,
    kTickCorePredicateDriveWidthInvalid,
    kTickCorePredicateCacheWidthNonpositive,
    kTickCorePredicateCacheWidthExceedsCapacity,
    kTickCorePredicatePendingCapacityInvalid,
    kTickCorePredicateChemistryWidthInvalid,
    kTickCorePredicateNegativeStartTick,
    kTickCorePredicateDriveNoiseWindowInvalid,
    kTickCorePredicateRngSemanticsVersionInvalid,
    kTickCorePredicateRngV2ContextInvalid,
    kTickCorePredicateOutputTickShapeInvalid,
    kTickCorePredicateResidentTickWindowInvalid,
    kTickCorePredicatePatchStatsTickWindowInvalid,
    kTickCorePredicateEventCapacityNegative,
    kTickCorePredicateEventReportingRowsNegative,
    kTickCorePredicateMaxBirthsNegative,
    kTickCorePredicateBirthLayoutInvalid,
    kTickCorePredicateMaxBirthsExceedsCapacity,
    kTickCorePredicateOrdinaryBirthConfigInvalid,
    kTickCorePredicateCountV1,
};
static_assert(kTickCorePredicateCountV1 < 63);

constexpr const char* kTickCorePredicateDiagnosticSchemaVersionV1 =
    "eiel_cuda_tick_core_predicate_diagnostic_v1";

inline constexpr const char* tick_core_predicate_name_v1(
    const int64_t bit
) {
    switch (bit) {
        case kTickCorePredicateHiddenSizeExceedsMax:
            return "hidden_size_exceeds_max";
        case kTickCorePredicateObservationSizeExceedsMax:
            return "observation_size_exceeds_max";
        case kTickCorePredicateActionCountExceedsMax:
            return "action_count_exceeds_max";
        case kTickCorePredicateObservationSizeMismatch:
            return "observation_size_mismatch";
        case kTickCorePredicateGmnwContextInvalid:
            return "gmnw_context_invalid";
        case kTickCorePredicateGruContextInvalid:
            return "gru_context_invalid";
        case kTickCorePredicateGmnwControlsInvalid:
            return "gmnw_controls_invalid";
        case kTickCorePredicateGmnwTraceInvalid:
            return "gmnw_trace_invalid";
        case kTickCorePredicateGmnwDecisionTapeInvalid:
            return "gmnw_decision_tape_invalid";
        case kTickCorePredicateGmnwSelectionMetricsInvalid:
            return "gmnw_selection_metrics_invalid";
        case kTickCorePredicatePatchViewRadiusInvalid:
            return "patch_view_radius_invalid";
        case kTickCorePredicateModularContextInvalid:
            return "modular_context_invalid";
        case kTickCorePredicateWdlRequiresPatch:
            return "wdl_requires_patch";
        case kTickCorePredicateEedRequiresPatch:
            return "eed_requires_patch";
        case kTickCorePredicateEedContextInvalid:
            return "eed_context_invalid";
        case kTickCorePredicateWdlTickWindowInvalid:
            return "wdl_tick_window_invalid";
        case kTickCorePredicateWdlContextInvalid:
            return "wdl_context_invalid";
        case kTickCorePredicatePatchContextInvalid:
            return "patch_context_invalid";
        case kTickCorePredicateDriveWidthInvalid:
            return "drive_width_invalid";
        case kTickCorePredicateCacheWidthNonpositive:
            return "cache_width_nonpositive";
        case kTickCorePredicateCacheWidthExceedsCapacity:
            return "cache_width_exceeds_capacity";
        case kTickCorePredicatePendingCapacityInvalid:
            return "pending_capacity_invalid";
        case kTickCorePredicateChemistryWidthInvalid:
            return "chemistry_width_invalid";
        case kTickCorePredicateNegativeStartTick:
            return "negative_start_tick";
        case kTickCorePredicateDriveNoiseWindowInvalid:
            return "drive_noise_window_invalid";
        case kTickCorePredicateRngSemanticsVersionInvalid:
            return "rng_semantics_version_invalid";
        case kTickCorePredicateRngV2ContextInvalid:
            return "rng_v2_context_invalid";
        case kTickCorePredicateOutputTickShapeInvalid:
            return "output_tick_shape_invalid";
        case kTickCorePredicateResidentTickWindowInvalid:
            return "resident_tick_window_invalid";
        case kTickCorePredicatePatchStatsTickWindowInvalid:
            return "patch_stats_tick_window_invalid";
        case kTickCorePredicateEventCapacityNegative:
            return "event_capacity_negative";
        case kTickCorePredicateEventReportingRowsNegative:
            return "event_reporting_rows_negative";
        case kTickCorePredicateMaxBirthsNegative:
            return "max_births_negative";
        case kTickCorePredicateBirthLayoutInvalid:
            return "birth_layout_invalid";
        case kTickCorePredicateMaxBirthsExceedsCapacity:
            return "max_births_exceeds_capacity";
        case kTickCorePredicateOrdinaryBirthConfigInvalid:
            return "ordinary_birth_config_invalid";
        default:
            return "unknown";
    }
}

inline constexpr const char* controller_failure_stage_name_v1(
    const int64_t stage
) {
    switch (stage) {
        case kFirstFailureNoFailureSentinelV1:
            return "no_failure";
        case kControllerFailureStageTickCoreAndBirthPlan:
            return "tick_core_and_birth_plan";
        case kControllerFailureStageBirthPrepare:
            return "birth_prepare";
        case kControllerFailureStageBirthTransactionAndSettlement:
            return "birth_transaction_and_settlement";
        case kControllerFailureStageBirthEventAppend:
            return "birth_event_append";
        default:
            return "unknown";
    }
}

inline constexpr const char* kGenerationSummaryScalarLayoutVersionV1 =
    "eiel_cuda_architecture_b_generation_summary_scalar_v1";

enum GenerationSummaryScalarColumnV1 : int64_t {
    kGenerationSummaryGeneration = 0,
    kGenerationSummaryEventCursorStart,
    kGenerationSummaryEventCursorPreTurnover,
    kGenerationSummaryEventCursorEnd,
    kGenerationSummaryAliveAfterTurnover,
    kGenerationSummaryEnergySumAfterTurnover,
    kGenerationSummaryBirths,
    kGenerationSummaryDeaths,
    kGenerationSummaryFailedBirths,
    kGenerationSummaryActionTotal,
    kGenerationSummaryMoveTotal,
    kGenerationSummaryGatherTotal,
    kGenerationSummaryMovedTotal,
    kGenerationSummaryGatheredTotal,
    kGenerationSummaryMovementAttempts,
    kGenerationSummaryMovementBlockedOccupied,
    kGenerationSummaryMovementBlockedBounds,
    kGenerationSummaryPeakInjury,
    kGenerationSummaryPolicyLoopTickCalls,
    kGenerationSummaryPolicyRefillCalls,
    kGenerationSummaryPolicyRefillRows,
    kGenerationSummaryPolicyCacheHitCalls,
    kGenerationSummaryPolicyCacheRowsRequested,
    kGenerationSummaryPolicyCacheRowsHit,
    kGenerationSummaryPolicyCacheRowsMissed,
    kGenerationSummaryPolicyCacheRowsRefilled,
    kGenerationSummaryPolicyCacheFullRefillTicks,
    kGenerationSummaryPolicyCachePartialRefillTicks,
    kGenerationSummaryPatchFoodStockAfterTurnover,
    kGenerationSummaryEnergyCollectedExactSumAfterTurnover,
    kGenerationSummaryPolicyEnergyLiveSumAfterTurnover,
    kGenerationSummaryScalarWidthV1,
};

inline constexpr const char* kGenerationSummaryScalarColumnNamesV1[] = {
    "generation",
    "event_cursor_start",
    "event_cursor_pre_turnover",
    "event_cursor_end",
    "alive_after_turnover",
    "energy_sum_after_turnover",
    "births",
    "deaths",
    "failed_births",
    "action_total",
    "move_total",
    "gather_total",
    "moved_total",
    "gathered_total",
    "movement_attempts",
    "movement_blocked_occupied",
    "movement_blocked_bounds",
    "peak_injury",
    "policy_loop_tick_calls",
    "policy_refill_calls",
    "policy_refill_rows",
    "policy_cache_hit_calls",
    "policy_cache_rows_requested",
    "policy_cache_rows_hit",
    "policy_cache_rows_missed",
    "policy_cache_rows_refilled",
    "policy_cache_full_refill_ticks",
    "policy_cache_partial_refill_ticks",
    "patch_food_stock_after_turnover",
    "energy_collected_exact_sum_after_turnover",
    "policy_energy_live_sum_after_turnover",
};

void launch_genome_upgrade_v2_c1_batch_v1(
    const int64_t* genotype,
    const int64_t* context,
    int64_t* genotype_out,
    int64_t* context_out,
    int64_t* product_out,
    int64_t* regulatory_out,
    int64_t* phenotype_out,
    int64_t* runtime_out,
    int64_t* cost_out,
    int32_t* status_out,
    int64_t batch,
    cudaStream_t stream
);

#if defined(__CUDACC__)
namespace genome_v2_c1_detail {

__global__ void batch_kernel_v1(
    const int64_t* const genotype,
    const int64_t* const context,
    int64_t* const genotype_out,
    int64_t* const context_out,
    int64_t* const product_out,
    int64_t* const regulatory_out,
    int64_t* const phenotype_out,
    int64_t* const runtime_out,
    int64_t* const cost_out,
    int32_t* const status_out,
    const int64_t batch
) {
    namespace genome_v2 = eiel::genome_upgrade_v2_q10_v1;
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= batch) {
        return;
    }
    const int64_t* const genotype_row =
        genotype + row * genome_v2::kC1GenotypeWidth;
    const int64_t* const context_row =
        context + row * genome_v2::kContextCount;
    genome_v2::GenotypeV1 value{};
    int32_t genotype_index = 0;
    for (int32_t homolog = 0; homolog < genome_v2::kHomologCount; ++homolog) {
        for (
            int32_t product = 0;
            product < genome_v2::kProductCount;
            ++product
        ) {
            for (
                int32_t parameter = 0;
                parameter < genome_v2::kParameterCount;
                ++parameter
            ) {
                value.alleles[homolog][product][parameter] =
                    genotype_row[genotype_index++];
            }
        }
    }
    for (int32_t homolog = 0; homolog < genome_v2::kHomologCount; ++homolog) {
        for (
            int32_t product = 0;
            product < genome_v2::kProductCount;
            ++product
        ) {
            value.modifiers[homolog][product] =
                genotype_row[genotype_index++];
        }
    }
    const int32_t input_status =
        genome_v2::c1_input_status_v1(value, context_row);
    status_out[row] = input_status;
    if (input_status != 0) {
        return;
    }
    genome_v2::ExpressionV1 expression{};
    genome_v2::RegulatoryDiagnosticsV1 diagnostics{};
    genome_v2::express_v1(
        value, context_row, &expression, &diagnostics
    );
    const genome_v2::RuntimeProjectionV1 runtime =
        genome_v2::project_runtime_traits_v1(expression);
    genome_v2::write_c1_dimensions_v1(
        value,
        context_row,
        expression,
        diagnostics,
        runtime,
        genotype_out + row * genome_v2::kC1GenotypeWidth,
        context_out + row * genome_v2::kContextCount,
        product_out + row * genome_v2::kC1ProductWidth,
        regulatory_out + row * genome_v2::kC1RegulatoryWidth,
        phenotype_out + row * genome_v2::kPhenotypeCount,
        runtime_out + row * genome_v2::kC1RuntimeWidth,
        cost_out + row * genome_v2::kC1CostWidth
    );
}

}  // namespace genome_v2_c1_detail

void launch_genome_upgrade_v2_c1_batch_v1(
    const int64_t* const genotype,
    const int64_t* const context,
    int64_t* const genotype_out,
    int64_t* const context_out,
    int64_t* const product_out,
    int64_t* const regulatory_out,
    int64_t* const phenotype_out,
    int64_t* const runtime_out,
    int64_t* const cost_out,
    int32_t* const status_out,
    const int64_t batch,
    cudaStream_t stream
) {
    constexpr int32_t kThreads = 128;
    const unsigned int blocks = static_cast<unsigned int>(
        (batch + kThreads - 1) / kThreads
    );
    genome_v2_c1_detail::batch_kernel_v1<<<blocks, kThreads, 0, stream>>>(
        genotype,
        context,
        genotype_out,
        context_out,
        product_out,
        regulatory_out,
        phenotype_out,
        runtime_out,
        cost_out,
        status_out,
        batch
    );
}
#endif

static_assert(
    sizeof(kGenerationSummaryScalarColumnNamesV1)
            / sizeof(kGenerationSummaryScalarColumnNamesV1[0])
        == kGenerationSummaryScalarWidthV1
);

struct KernelResourceReceipt {
    int32_t binary_version;
    int32_t max_blocks_per_multiprocessor;
    int32_t max_threads_per_block;
    int32_t num_registers;
    int32_t ptx_version;
    int64_t constant_size_bytes;
    int64_t local_size_bytes;
    int64_t shared_size_bytes;
};

struct ControllerRunReceipt {
    int64_t native_dispatches;
    int64_t native_tick_iterations;
    int64_t phase_1_launches;
    int64_t birth_prepare_launches;
    int64_t birth_commit_launches;
    int64_t birth_prepare_commit_fused_launches;
    int64_t turnover_shadow_plan_launches;
    int64_t turnover_cull_refill_plan_launches;
    int64_t turnover_commit_launches;
    int64_t turnover_publish_launches;
    int64_t patch_density_refresh_launches;
    int64_t gru_cleanup_launches;
    int64_t generation_summary_finalize_launches;
    int64_t host_synchronizations_inside_loop;
    int64_t final_host_synchronizations;
    int32_t cuda_status;
#if EIEL_OPTION_D_M0_TELEMETRY_ENABLED == 1
    std::vector<int64_t> gaussian_pre_refill_synchronization_time_ns;
    std::vector<int64_t> gaussian_post_copy_synchronization_time_ns;
#endif
};

enum class TickCoreExecutionModeV1 : int32_t {
    kFull = 0,
    kPrepareExactCpuPolicy = 1,
    kResumeAfterExactCpuPolicy = 2,
};

struct FlatFullTickState {
    int32_t* age;
    float* gru_upkeep_cost;
    const float* cost_charge_scalar;
    const float* cost_charge_maturation_scalar;
    float* drive_inertia;
    float* drive_noise;
    float* prev_drive_delta;
    float* drive_volatility;
    float* drive_jitter;
    float* episode0_drive_jitter_sum;
    float* last_energy_delta;
    float* last_health_delta;
    float* last_injury_delta;
    int32_t* selfharm_events;
    int32_t* episode0_selfharm_events;
    float* chemistry;
    uint64_t* runner_pcg64_state_words;
    int64_t* runner_pcg64_buffer_words;
    uint64_t* shared_env_policy_pcg64_state_words;
    int64_t* shared_env_policy_pcg64_buffer_words;
    float* drive_noise_samples;
    bool* drive_noise_samples_valid;
    int64_t* slot_free_queue;
    int64_t* slot_free_queue_count;
    int64_t* residual_pending_free_agent_ids;
    int64_t* residual_pending_free_slots;
    int64_t* residual_pending_free_count;
    int64_t* policy_np_semantic_draws;
    int64_t* metrics_np_semantic_draws;
    int64_t drive_noise_sample_ticks;
    int64_t chemistry_width;
};

struct FlatFullTickOutputs {
    int32_t* chemistry_order_slots;
    int64_t* runner_shuffle_draws;
    int32_t* runner_shuffle_status;
    int64_t* drive_noise_draws;
    int64_t* drive_noise_raw64_draws;
    int32_t* drive_noise_status;
    int64_t* chemistry_draws;
    int32_t* chemistry_status;
    int32_t* active_mechanics_status;
    int64_t* active_mechanics_counts;
    int64_t* residual_death_agent_ids;
    int64_t* residual_death_slots;
    int32_t* residual_death_reason_codes;
    int64_t* residual_death_count;
    int64_t* residual_release_count;
    int32_t* residual_release_status;
    uint64_t* policy_pcg64_state_after_action;
    int64_t* policy_pcg64_buffer_after_action;
    int64_t* food_spawn_stats;
};

struct FlatEventState {
    int32_t* event_type;
    int64_t* event_generation;
    int64_t* event_tick;
    int64_t* event_actor_id;
    int64_t* event_target_id;
    float* event_value;
    int32_t* typed_event_type;
    int64_t* typed_generation;
    int64_t* typed_tick;
    int64_t* typed_sequence;
    int64_t* typed_actor_id;
    int64_t* typed_target_id;
    int64_t* typed_actor_slot;
    int64_t* typed_target_slot;
    int64_t* typed_parent_id;
    int64_t* typed_child_id;
    int64_t* typed_founder_id;
    int32_t* typed_lineage_depth;
    int32_t* typed_subtype;
    int32_t* typed_reason;
    int8_t* typed_success;
    float* typed_value;
    int64_t* active_count;
    int64_t* sequence_base;
    int32_t* status;
    int32_t* lineage_founder_id;
    int32_t* lineage_depth;
    const int64_t* initial_agent_ids_by_row;
    int32_t* terminal_survival;
    int32_t* terminal_rows;
    int32_t* status_by_tick;
    int64_t capacity;
    int64_t reporting_rows;
};

struct FlatOrdinaryBirthState {
    float* scar;
    int32_t* repro_cooldown;
    float* repro_threshold;
    int32_t* offspring_count;
    int32_t* lineage_birth_generation;
    int32_t* lineage_birth_tick;
    int32_t* patch_habitat_lineage_id;
    int64_t* population_next_id;

    uint32_t* python_random_words;
    int64_t* python_random_index;
    bool* python_random_has_gauss;
    double* python_random_gauss_next;
    const double* python_random_gauss_primary_tape;
    const double* python_random_gauss_secondary_tape;
    int64_t* python_random_raw_draws;
    int64_t* mutation_py_semantic_draws;

    int32_t* genome_hidden_size;
    int32_t* genome_memory_dim;
    int32_t* genome_sensor_channel_mask;
    int32_t* genome_sensor_radius;
    int32_t* genome_introspection_channel_mask;
    int32_t* genome_self_regulation_control_mask;
    int32_t* genome_self_modification_target_mask;
    int32_t* genome_directed_inheritance_channel_mask;
    int64_t* policy_mem_dim;
    bool* policy_use_gru;

    float* genome_exploration_bias;
    float* genome_aggression_bias;
    float* genome_mutation_rate;
    float* genome_reproduction_threshold;
    float* genome_action_temperature;
    float* genome_drive_inertia;
    float* genome_drive_noise;
    float* genome_drive_logit_gain;
    float* genome_pred_lr;
    float* genome_drive_sp_hunger;
    float* genome_drive_sp_pain;
    float* genome_drive_sp_fatigue;
    float* genome_drive_sp_fear;
    float* genome_exp_prior_strength;
    float* genome_exp_prior_noise;
    float* genome_drive_birth_noise;
    float* pred_lr;

    float* birth_reset_shock;
    uint64_t* population_pcg64_state_words;
    int64_t* population_pcg64_buffer_words;
    int64_t* mutation_np_semantic_draws;

    float* hmem;
    uint8_t* mutation_torch_cpu_state;
    const float* policy_birth_delta_tape;
    const bool* policy_birth_mutation_mask_tape;
    const uint8_t* policy_birth_crossover_mask_packed_tape;
    const uint8_t* policy_birth_post_rng_state_packed_tape;
    const int64_t* policy_birth_semantic_draws_tape;
    int64_t* policy_birth_tape_cursor;
    int64_t* mutation_torch_cpu_semantic_draws;

    int32_t* attacks;
    int32_t* early_survival_steps;
    float* early_energy_collected;
    double* early_energy_collected_exact;
    float* early_injury_gained;
    float* early_life_weight;
    float* pred_err_sum;
    int32_t* pred_err_count;
    int32_t* first_repro_step;
    bool* dev_stage_initialized;
    float* episode0_pred_err_sum;
    int32_t* episode0_pred_err_count;
    float* late_pred_err_sum;
    int32_t* late_pred_err_count;
    float* chemistry_observation_fidelity;
    float* chemistry_observation_bias;
    int64_t* policy_birth_tape_cursor_base;
    int64_t* python_random_raw_draw_base;

    // Present only for the versioned diploid route. The packed storage is
    // [cell, homolog*locus, slot], with 34 independent contiguous SoA planes.
    int32_t* inheritance_mode;
    bool* inheritance_mutation_enabled;
    bool* resource_handling_enabled;
    double* mutation_sigma_multiplier;
    float* diploid_allele_storage;
    float* genome_resource_handling_axis;
    float* diploid_allele_shadow_storage;
    float* genome_resource_handling_axis_shadow;

    // Present only for the fixed Q10 GENOME-UPGRADE-V2 route. All resident
    // planes are exact int64 SoA arrays with slot as the final dimension.
    bool* genome_upgrade_v2_enabled;
    int64_t* genome_upgrade_v2_mutation_rate_q10;
    int64_t* genome_upgrade_v2_mutation_step_q10;
    int64_t* genome_upgrade_v2_alleles_q10;
    int64_t* genome_upgrade_v2_modifiers_q10;
    int64_t* genome_upgrade_v2_products_q10;
    int64_t* genome_upgrade_v2_phenotypes_q10;
    int64_t* genome_upgrade_v2_expression_cost_q10;
    int64_t* genome_upgrade_v2_active_product_count;
    int64_t* genome_upgrade_v2_saturated_product_count;
    int64_t* genome_upgrade_v2_alleles_q10_shadow;
    int64_t* genome_upgrade_v2_modifiers_q10_shadow;
    int64_t* genome_upgrade_v2_products_q10_shadow;
    int64_t* genome_upgrade_v2_phenotypes_q10_shadow;
    int64_t* genome_upgrade_v2_expression_cost_q10_shadow;
    int64_t* genome_upgrade_v2_active_product_count_shadow;
    int64_t* genome_upgrade_v2_saturated_product_count_shadow;

    // The canonical digest authority is a live resident plane, not metadata
    // encoded into the scratch tail.  Keep this additive at the end of the
    // ABI so the existing positional fields retain their layout.
    uint8_t* canonical_hereditary_digest;
};

struct FlatOrdinaryBirthOutputs {
    int64_t* slot_free_queue_head;
    int32_t* planned_repro_cooldown;
    int64_t* planned_parent_slots;
    int64_t* planned_child_slots;
    int32_t* planned_spawn_positions;
    float* planned_child_energy;
    int64_t* decision_parent_slots;
    int32_t* decision_reason_codes;
    int32_t* decision_candidate_counts;
    int64_t* decision_draw_starts;
    int64_t* decision_draw_ends;
    int32_t* decision_selected_positions;
    bool* decision_selected_reserved;
    int64_t* reason_counts;
    int64_t* planned_count;
    int64_t* decision_count;
    int64_t* spawn_rng_draws;
    int64_t* spawn_rng_semantic_draws;
    int64_t* runner_rng_draws;
    int64_t* runner_rng_semantic_draws;
    int32_t* plan_status;
    int64_t* committed_parent_agent_ids;
    int64_t* committed_child_agent_ids;
    int64_t* committed_child_slots;
    int64_t* identity_committed_count;
    int32_t* identity_status;
    int64_t* scalar_committed_count;
    int64_t* scalar_semantic_draws;
    int32_t* scalar_status;
    int64_t* drive_committed_count;
    int64_t* birth_noise_rows_consumed;
    int64_t* exp_prior_rows_consumed;
    int64_t* standard_normal_rows;
    int64_t* drive_semantic_draws;
    int32_t* drive_status;
    int32_t* chemistry_status;
    int64_t* policy_committed_count;
    int64_t* policy_semantic_draws;
    int64_t* policy_mutation_events;
    int64_t* policy_mutated_elements;
    double* policy_mutation_delta_total;
    bool* policy_mutation_occurred;
    float* policy_mutation_delta;
    int32_t* policy_status;
    int32_t* combined_status;
    uint64_t* planner_pcg64_state_before;
    int64_t* planner_pcg64_buffer_before;
    int64_t* planned_mate_slots;
    int64_t* committed_mate_agent_ids;
    double* mate_planned_energy;
    double* mate_planned_child_energy_exact;
    double* mate_planned_parent_energy_after;
    double* mate_planned_mate_energy_after;
    int64_t* decision_mate_slots;
    int32_t* decision_adjacent_counts;
    int32_t* decision_spawn_candidate_counts;
    int64_t* mate_reason_counts;
    int32_t* mate_decision_reason_codes;
    int64_t* genome_v2_causal_record_rows;
    bool* genome_v2_causal_record_valid;
    int64_t* genome_v2_causal_record_count;
    int64_t* genome_v2_causal_record_dropped_count;
    bool* genome_v2_causal_capture_enabled;
    int64_t genome_v2_causal_record_capacity;
};

struct FlatFlattenedBirthScratch {
    uint8_t* storage;
    int64_t bytes_per_cell;
};

struct FlatOrdinaryBirthConfig {
    bool enabled;
    int64_t max_births_per_tick;
    double parent_energy_share;
    float minimum_energy_buffer;
    int32_t cooldown_steps;
    int32_t sensor_radius_max;
    double injury_repro_k_injury;
    double injury_repro_k_scar;
    double injury_birth_fail_scar;
    double injury_birth_fail_injury;
    bool env_policy_alias;
    bool requires_mate;
    int64_t gauss_tape_capacity;
    int64_t policy_tape_capacity;
    int64_t policy_mutation_tape_width;
    int64_t torch_state_bytes;
    int64_t hmem_width;
    cross_path_resident_control_v1::FlatCrossPathResidentControlV1
        cross_path_control;
    const double* parent_energy_share_by_cell;
    const double* minimum_energy_buffer_by_cell;
    const std::int32_t* cooldown_steps_by_cell;
    const double* reproduce_try_probability_by_cell;
    int32_t rng_semantics_version;
    uint32_t world_uid;
    const int64_t* world_uid_by_cell;
    uint64_t rng_namespace_key;
    uint32_t rng_semantic_epoch;
};

#if defined(__CUDACC__)
__device__ __forceinline__ double birth_parent_energy_share_for_cell_v1(
    const FlatOrdinaryBirthConfig& config,
    const std::int64_t cell
) {
    return config.parent_energy_share_by_cell == nullptr
        ? config.parent_energy_share
        : config.parent_energy_share_by_cell[cell];
}

__device__ __forceinline__ uint32_t rng_world_uid_for_cell_v1(
    const FlatOrdinaryBirthConfig& config,
    const std::int64_t cell
) {
    return config.world_uid_by_cell == nullptr
        ? config.world_uid
        : static_cast<uint32_t>(config.world_uid_by_cell[cell]);
}

__device__ __forceinline__ double birth_minimum_energy_buffer_for_cell_v1(
    const FlatOrdinaryBirthConfig& config,
    const std::int64_t cell
) {
    return config.minimum_energy_buffer_by_cell == nullptr
        ? static_cast<double>(config.minimum_energy_buffer)
        : config.minimum_energy_buffer_by_cell[cell];
}

__device__ __forceinline__ std::int32_t birth_cooldown_steps_for_cell_v1(
    const FlatOrdinaryBirthConfig& config,
    const std::int64_t cell
) {
    return config.cooldown_steps_by_cell == nullptr
        ? config.cooldown_steps
        : config.cooldown_steps_by_cell[cell];
}

__device__ __forceinline__ double birth_try_probability_for_cell_v1(
    const FlatOrdinaryBirthConfig& config,
    const std::int64_t cell
) {
    return config.reproduce_try_probability_by_cell == nullptr
        ? 1.0
        : config.reproduce_try_probability_by_cell[cell];
}
#endif

struct FlatMechanicsConfig {
    bool enable_rest_charge;
    int32_t rest_streak_free;
    float rest_charge_energy;
    float rest_charge_growth;
    float rest_deficit_to_health;
    int32_t developmental_maturity_age;
    int32_t life_history_mode;
    int32_t life_history_maturity_start_ticks;
    int32_t life_history_maturity_rise_ticks;
    int32_t life_history_fertility_decline_start_ticks;
    int32_t life_history_fertility_decline_ticks;
    float life_history_elder_fertility_floor;
    int32_t life_history_senescence_start_ticks;
    int32_t life_history_senescence_ramp_ticks;
    float life_history_senescence_energy_cost;
    float life_history_senescence_health_cost;
    int32_t life_history_max_age_ticks;
    float drive_hunger_gain;
    float drive_hunger_decay;
    float drive_pain_gain;
    float drive_pain_decay;
    float drive_fatigue_gain;
    float drive_fatigue_rest_relief;
    float drive_fatigue_decay;
    float drive_fear_gain;
    float drive_fear_decay;
    int64_t generation;
    int64_t start_tick;
    int64_t resident_start_tick;
};

enum class FlatPatchP2Mode : int32_t {
    kDisabled = 0,
    kContextProbeB2T1 = 1,
    kFullPatchP2 = 2,
};

enum FlatPatchP2CompactMetric : int64_t {
    kPatchCompactMovementEnergyEventCount = 0,
    kPatchCompactSpawnSyncRows,
    kPatchCompactTypedDeathEventRows,
    kPatchCompactEnvRngPaddingDraws,
    kPatchCompactFoodHarvestTotal,
    kPatchCompactScentEmitTotal,
    kPatchCompactWasteEmitTotal,
    kPatchCompactMovementEnergyTotal,
    kPatchCompactEnergyCollectedDelta,
    kPatchCompactScentMoveQueuedTotal,
    kPatchCompactWasteMoveQueuedTotal,
    kPatchCompactScentGatherQueuedTotal,
    kPatchCompactWasteGatherQueuedTotal,
    kPatchCompactMovementEnergyExtraTotal,
    kPatchCompactScentEmittedTotal,
    kPatchCompactWasteEmittedTotal,
    kPatchCompactFoodDepletedFrac,
    kPatchCompactFoodFullFrac,
    kPatchCompactDensityHotspotFrac,
    kPatchCompactMetricCount,
};

struct FlatPatchP2Context {
    bool enabled;
    FlatPatchP2Mode mode;
    int64_t stats_ticks_per_cell;
    persistent_patch_p2::PatchP2State state;
    persistent_patch_p2::PatchP2Scratch scratch;
    persistent_patch_p2::PatchP2Outputs outputs;
    persistent_patch_p2::PatchP2Config config;
    double* compact_metrics;
    float* patch_cover;
    float* patch_shelter;
    float* patch_movement_cost;
    double patch_food_initial_value;
    double patch_food_gather_amount;
    double patch_scent_move_emit;
    double patch_waste_move_emit;
    double patch_scent_gather_emit;
    double patch_waste_consume_emit;
    bool patch_movement_cost_apply_enable;
    double patch_movement_cost_energy_scale;
    const double* patch_movement_cost_energy_scale_by_cell;
    double patch_movement_cost_min;
    double patch_movement_cost_max;
    const double* patch_food_gather_amount_by_cell;
    const double* patch_food_regrow_per_step_by_cell;
};

struct FlatCompactSummaryContextV1 {
    bool enabled;
    std::uint8_t* reporting_retained_bytes;
    std::int32_t* reporting_slot_to_behavior_row;
    double* reporting_effective_fitness_by_row;
    std::int64_t* event_cursor_start;
    std::int64_t* event_cursor_pre_turnover;
    std::int64_t* event_cursor_end;
    compact_summary_reporting_authority_v1::ReportingFitnessConfigV1
        reporting_fitness_config;
    std::int64_t generation_index;
    std::int64_t generation_count;
    std::int64_t cell_count;
    std::int32_t reporting_row_capacity;
    std::int32_t slot_capacity;
};

#if defined(__CUDACC__)
__device__ __forceinline__
compact_summary_reporting_authority_v1::GenerationAuthorityAccessV1
compact_summary_reporting_access_v1(
    const FlatCompactSummaryContextV1& context,
    const std::int64_t cell
) noexcept {
    const std::int64_t row_index =
        context.generation_index * context.cell_count + cell;
    return compact_summary_reporting_authority_v1::access_at_row_v1(
        context.reporting_retained_bytes,
        context.reporting_slot_to_behavior_row,
        context.reporting_effective_fitness_by_row,
        row_index,
        context.reporting_row_capacity,
        context.slot_capacity
    );
}
#endif

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
    FlatFullTickState full_tick_state,
    FlatFullTickOutputs full_tick_outputs,
    FlatEventState event_state,
    FlatOrdinaryBirthState ordinary_birth_state,
    FlatOrdinaryBirthOutputs ordinary_birth_outputs,
    FlatFlattenedBirthScratch ordinary_birth_scratch,
    FlatOrdinaryBirthConfig ordinary_birth_config,
    flattened_turnover_v1::FlatTurnoverMutableTensorTable turnover_tensors,
    flattened_turnover_v1::FlatTurnoverScratch turnover_scratch,
    flattened_turnover_v1::FlatTurnoverOutputs turnover_outputs,
    flattened_turnover_v1::FlatTurnoverConfig turnover_config,
    FlatMechanicsConfig mechanics_config,
    FlatPatchP2Context patch_p2_context,
    FlatCompactSummaryContextV1 compact_summary_context,
    int64_t cells,
    int64_t ticks,
    int64_t output_ticks_per_cell,
    int64_t resident_ticks_per_cell,
    int64_t active_capacity,
    int64_t agent_id_capacity,
    int64_t capacity,
    int64_t pending_capacity,
    int64_t height,
    int64_t width,
    int64_t view_radius,
    int64_t observation_size,
    int64_t hidden_size,
    int64_t action_count,
    int64_t packed_width,
    int64_t drive_width,
    int64_t cache_capacity,
    int64_t cache_width,
    bool deterministic_actions,
    int64_t max_food,
    int64_t effective_food_spawn_count,
    double max_energy,
    double start_energy,
    double start_health,
    const double* move_cost_by_cell,
    double gather_cost,
    double rest_cost,
    double attack_cost,
    double action_cost_mult,
    double rest_cost_mult,
    double food_energy,
    int64_t episode0_k,
    float w_hunger_gather,
    float w_pain_rest,
    float w_fatigue_rest,
    float w_pain_attack,
    float w_fear_attack,
    const std::function<void(int64_t, int64_t)>& exact_cpu_policy_refill,
    const std::function<void(int64_t, int64_t)>& hapb_ordinary_birth_commit,
    const std::function<void(int64_t, int64_t, int64_t)>&
        policy_birth_segment_refill,
    const std::function<void(int64_t, int64_t, int64_t)>& gaussian_segment_refill,
    const std::function<void(int64_t, int64_t, int64_t)>& inter_tick_control,
    const std::function<void()>& hapb_turnover_commit,
    cudaStream_t stream,
    bool synchronize_at_end
);

void launch_compact_checkpoint_active_slots_projection_v1(
    const int64_t* active_agent_ids,
    const int32_t* active_agent_slots,
    const int64_t* active_count_state,
    const int64_t* slot_to_agent_id,
    const bool* alive_mask,
    int32_t* active_agent_slots_out,
    int64_t* agent_id_to_slot_prefix_length_out,
    int64_t cells,
    int64_t active_capacity,
    int64_t slot_count,
    cudaStream_t stream
);

KernelResourceReceipt query_flattened_controller_tick_core_resources_v1();
KernelResourceReceipt query_flattened_controller_birth_prepare_resources_v1();
KernelResourceReceipt query_flattened_controller_birth_commit_resources_v1();
KernelResourceReceipt
query_flattened_controller_birth_prepare_commit_resources_v1();
KernelResourceReceipt query_generation_summary_finalize_resources_v1();

}  // namespace eiel::cuda_architecture_b::flattened_controller_v1

namespace eiel::cuda_architecture_b::flattened_turnover_v1 {

flattened_controller_v1::KernelResourceReceipt
query_flattened_turnover_shadow_plan_resources_v1();
flattened_controller_v1::KernelResourceReceipt
query_flattened_turnover_cull_refill_plan_resources_v1();
flattened_controller_v1::KernelResourceReceipt
query_flattened_turnover_commit_resources_v1();
flattened_controller_v1::KernelResourceReceipt
query_flattened_turnover_publish_resources_v1();

}  // namespace eiel::cuda_architecture_b::flattened_turnover_v1
