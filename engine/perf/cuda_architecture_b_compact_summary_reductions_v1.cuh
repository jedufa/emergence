#ifndef EIEL_CUDA_ARCHITECTURE_B_COMPACT_SUMMARY_REDUCTIONS_V1_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_COMPACT_SUMMARY_REDUCTIONS_V1_CUH_

#include <cstddef>
#include <cstdint>
#include <type_traits>

// Keep the POD ABI usable from an ordinary C++ translation unit. CUDA defines
// cudaStream_t as this opaque pointer type.
struct CUstream_st;

namespace eiel::cuda_architecture_b::compact_summary_reductions_v1 {

inline constexpr char kContractVersionV1[] =
    "eiel_cuda_architecture_b_compact_summary_reductions_v1";
inline constexpr std::int64_t kEventTypeCountV1 = 4;
inline constexpr std::int32_t kBirthCommittedEventTypeV1 = 1;
inline constexpr std::int32_t kMutationEventTypeV1 = 2;
inline constexpr std::int32_t kBirthFailedEventTypeV1 = 3;
inline constexpr std::int32_t kDeathCommittedEventTypeV1 = 4;
inline constexpr std::int32_t kEnergyDepletedReasonCodeV1 = 1;
inline constexpr std::int32_t kHealthDepletedReasonCodeV1 = 2;
inline constexpr std::int32_t kMinimumBirthFailureReasonCodeV1 = 1;
inline constexpr std::int32_t kMaximumBirthFailureReasonCodeV1 = 9;
inline constexpr std::int64_t kBirthFailureReasonCountV1 = 9;
inline constexpr std::int64_t kSentinelIntegerV1 = -1;
inline constexpr std::int64_t kInt64MaximumV1 = 9223372036854775807LL;

// GenerationSummaryScalarColumnV1 coordinates used by this reduction.
inline constexpr std::int64_t kGenerationSummaryGenerationColumnV1 = 0;
inline constexpr std::int64_t kGenerationSummaryCursorStartColumnV1 = 1;
inline constexpr std::int64_t kGenerationSummaryCursorPreTurnoverColumnV1 = 2;
inline constexpr std::int64_t kGenerationSummaryCursorEndColumnV1 = 3;
inline constexpr std::int64_t kGenerationSummaryEnergySumColumnV1 = 5;
inline constexpr std::int64_t kGenerationSummaryBirthsColumnV1 = 6;
inline constexpr std::int64_t kGenerationSummaryDeathsColumnV1 = 7;
inline constexpr std::int64_t kGenerationSummaryFailedBirthsColumnV1 = 8;
inline constexpr std::int64_t kGenerationSummaryPeakInjuryColumnV1 = 17;
inline constexpr std::int64_t kGenerationSummaryPatchFoodStockColumnV1 = 28;
inline constexpr std::int64_t kGenerationSummaryEnergyCollectedExactSumColumnV1 = 29;
inline constexpr std::int64_t kGenerationSummaryPolicyEnergyLiveSumColumnV1 = 30;
inline constexpr std::int64_t kGenerationSummaryScalarWidthV1 = 31;
inline constexpr std::int64_t kBirthPlanReasonCountV1 = 10;
inline constexpr std::int32_t kBirthPlanAcceptedReasonCodeV1 = 0;
inline constexpr std::int32_t kBirthPlanNoSpaceReasonCodeV1 = 1;
inline constexpr std::int32_t kBirthPlanBufferReasonCodeV1 = 2;
inline constexpr std::int32_t kBirthPlanNoFreeSlotReasonCodeV1 = 3;
inline constexpr std::int32_t kBirthPlanUnusedReasonCodeV1 = 4;
inline constexpr std::int32_t kBirthPlanConceptionReasonCodeV1 = 5;
inline constexpr std::int32_t kBirthPlanGestationReasonCodeV1 = 6;
inline constexpr std::int32_t kBirthPlanHealthReasonCodeV1 = 7;
inline constexpr std::int32_t kBirthPlanCooldownReasonCodeV1 = 8;
inline constexpr std::int32_t kBirthPlanNoEnergyReasonCodeV1 = 9;
inline constexpr bool kFlatHardInjuryGateActiveV1 = false;

enum ReductionStatusV1 : std::int32_t {
    kReductionSuccessV1 = 0,
    kReductionInvalidStorageV1 = 1,
    kReductionCursorContradictionV1 = 2,
    kReductionEventSemanticContradictionV1 = 3,
    kReductionIdentityContradictionV1 = 4,
    kReductionSlotConservationContradictionV1 = 5,
    kReductionBirthMetricsContradictionV1 = 6,
};

enum ReductionLaunchStatusV1 : std::int32_t {
    kReductionLaunchSuccessV1 = 0,
    kReductionLaunchInvalidRequestShapeV1 = 1,
    kReductionLaunchInvalidOutputShapeV1 = 2,
};

struct TypedLifecycleColumnsV1 {
    const std::int32_t* event_type;
    const std::int64_t* generation;
    const std::int64_t* tick;
    const std::int64_t* sequence;
    const std::int64_t* actor_agent_id;
    const std::int64_t* target_agent_id;
    const std::int64_t* actor_slot;
    const std::int64_t* target_slot;
    const std::int64_t* parent_agent_id;
    const std::int64_t* child_agent_id;
    const std::int64_t* lineage_founder_id;
    const std::int32_t* lineage_depth;
    const std::int32_t* subtype_code;
    const std::int32_t* reason_code;
    const std::int8_t* success;
    const float* value;
    std::int64_t cell_stride;
    std::int64_t capacity;
};

struct LifecycleSourcesV1 {
    TypedLifecycleColumnsV1 typed;
    // FlatEventState::active_count is authoritative for typed/tape/cursor state.
    const std::int64_t* active_count;
};

struct GenerationSummaryScalarViewV1 {
    const double* values;
    std::int64_t generation_count;
    std::int64_t cell_count;
    std::int64_t width;
};

struct IdentitySourcesV1 {
    const std::int64_t* active_agent_ids;
    const std::int64_t* active_count_state;
    const std::int64_t* population_next_id;
    const std::int64_t* slot_free_queue_count;
    std::int64_t active_id_cell_stride;
    std::int64_t capacity;
};

struct OrdinaryBirthMetricsSourcesV1 {
    // ordinary_birth_output_tensors[15], contiguous int64 [C,ticks].
    const std::int64_t* birth_decision_count;
    // ordinary_birth_output_tensors[13], contiguous int64 [C,ticks,10].
    const std::int64_t* birth_reason_counts;
    std::int64_t tick_count;
    std::int64_t decision_cell_stride;
    std::int64_t reason_cell_stride;
    std::int64_t reason_tick_stride;
    std::int64_t reason_count;
};

struct ReductionInputsV1 {
    LifecycleSourcesV1 lifecycle;
    GenerationSummaryScalarViewV1 generation_summary_scalar;
    IdentitySourcesV1 identity;
    OrdinaryBirthMetricsSourcesV1 ordinary_birth_metrics;
    std::int64_t generation_index;
    std::int64_t cell_count;
};

struct alignas(8) ReductionResultV1 {
    std::int32_t status;
    std::uint8_t valid;
    std::uint8_t active_id_bounds_defined;
    std::uint16_t reserved;
    std::int64_t generation_event_count;
    std::int64_t turnover_event_count;
    std::int64_t generation_counts_by_type[kEventTypeCountV1];
    std::int64_t turnover_counts_by_type[kEventTypeCountV1];
    std::int64_t generation_failed_birth_counts_by_reason[
        kBirthFailureReasonCountV1
    ];
    std::int64_t turnover_failed_birth_counts_by_reason[
        kBirthFailureReasonCountV1
    ];
    std::int64_t birth_failure_counts_by_reason[kBirthFailureReasonCountV1];
    std::int64_t native_failed_birth_event_rows;
    std::int64_t injury_blocked_births;
    std::int64_t birth_attempt_rows;
    std::int64_t birth_failure_rows;
    std::int64_t deaths_energy_depleted;
    std::int64_t deaths_health_depleted;
    std::int64_t cursor_start;
    std::int64_t cursor_pre_turnover;
    std::int64_t cursor_end;
    std::int64_t authoritative_cursor;
    std::int64_t event_row_count;
    std::int64_t active_event_count;
    // Compatibility aliases: the resident typed columns and tape share active_count.
    std::int64_t typed_event_row_count;
    std::int64_t tape_event_row_count;
    std::int64_t population_next_id;
    std::int64_t active_agent_count;
    std::int64_t active_agent_id_min;
    std::int64_t active_agent_id_max;
    std::int64_t free_slot_count;
};

struct ReductionOutputViewV1 {
    ReductionResultV1* rows;
    std::int64_t generation_count;
    std::int64_t cell_count;
};

struct KernelResourceReceiptV1 {
    std::int32_t binary_version;
    std::int32_t ptx_version;
    std::int32_t max_threads_per_block;
    std::int32_t registers_per_thread;
    std::int32_t occupancy_blocks_per_multiprocessor;
    std::int64_t constant_bytes;
    std::int64_t local_bytes_per_thread;
    std::int64_t shared_bytes_per_block;
};

inline constexpr std::int32_t kThreadsPerBlockV1 = 128;

#if defined(__CUDACC__)
#define EIEL_REDUCTIONS_HOST_DEVICE_V1 __host__ __device__
#else
#define EIEL_REDUCTIONS_HOST_DEVICE_V1
#endif

EIEL_REDUCTIONS_HOST_DEVICE_V1 inline constexpr bool counts_as_birth_attempt_v1(
    const std::int64_t reason
) noexcept {
    return reason == kBirthPlanAcceptedReasonCodeV1
        || reason == kBirthPlanNoFreeSlotReasonCodeV1
        || reason == kBirthPlanConceptionReasonCodeV1
        || reason == kBirthPlanGestationReasonCodeV1;
}

EIEL_REDUCTIONS_HOST_DEVICE_V1 inline constexpr bool
counts_as_failed_birth_row_v1(const std::int64_t reason) noexcept {
    return reason == kBirthPlanNoSpaceReasonCodeV1
        || reason == kBirthPlanBufferReasonCodeV1
        || reason == kBirthPlanNoFreeSlotReasonCodeV1
        || reason == kBirthPlanConceptionReasonCodeV1
        || reason == kBirthPlanGestationReasonCodeV1;
}

EIEL_REDUCTIONS_HOST_DEVICE_V1 inline constexpr bool positive_product_fits_i64(
    const std::int64_t left,
    const std::int64_t right
) noexcept {
    return left > 0
        && right > 0
        && left <= kInt64MaximumV1 / right;
}

EIEL_REDUCTIONS_HOST_DEVICE_V1 inline constexpr bool valid_generation_shape_v1(
    const std::int64_t generation_index,
    const std::int64_t generation_count,
    const std::int64_t cell_count
) noexcept {
    return generation_index >= 0
        && generation_index < generation_count
        && positive_product_fits_i64(generation_count, cell_count);
}

EIEL_REDUCTIONS_HOST_DEVICE_V1 inline constexpr std::int64_t
generation_major_row_index_v1(
    const std::int64_t generation_index,
    const std::int64_t cell_count,
    const std::int64_t cell
) noexcept {
    return generation_index * cell_count + cell;
}

#undef EIEL_REDUCTIONS_HOST_DEVICE_V1

#if defined(__CUDACC__)

__device__ __forceinline__ ReductionResultV1 invalid_result(
    const ReductionStatusV1 status
) noexcept {
    ReductionResultV1 result{};
    result.status = static_cast<std::int32_t>(status);
    return result;
}

__device__ __forceinline__ bool valid_event_storage(
    const LifecycleSourcesV1& lifecycle,
    const std::int64_t cell_count
) noexcept {
    const TypedLifecycleColumnsV1& events = lifecycle.typed;
    return events.event_type != nullptr
        && events.generation != nullptr
        && events.tick != nullptr
        && events.sequence != nullptr
        && events.actor_agent_id != nullptr
        && events.target_agent_id != nullptr
        && events.actor_slot != nullptr
        && events.target_slot != nullptr
        && events.parent_agent_id != nullptr
        && events.child_agent_id != nullptr
        && events.lineage_founder_id != nullptr
        && events.lineage_depth != nullptr
        && events.subtype_code != nullptr
        && events.reason_code != nullptr
        && events.success != nullptr
        && events.value != nullptr
        && lifecycle.active_count != nullptr
        && events.capacity >= 0
        && events.cell_stride >= events.capacity
        && (
            cell_count <= 1
            || events.cell_stride
                <= kInt64MaximumV1 / cell_count
        );
}

__device__ __forceinline__ bool valid_generation_summary_storage(
    const ReductionInputsV1& inputs
) noexcept {
    const GenerationSummaryScalarViewV1& summary =
        inputs.generation_summary_scalar;
    return summary.values != nullptr
        && summary.width == kGenerationSummaryScalarWidthV1
        && summary.cell_count == inputs.cell_count
        && valid_generation_shape_v1(
            inputs.generation_index,
            summary.generation_count,
            summary.cell_count
        )
        && summary.generation_count * summary.cell_count
            <= kInt64MaximumV1 / summary.width;
}

__device__ __forceinline__ bool valid_identity_storage(
    const IdentitySourcesV1& identity,
    const std::int64_t cell_count
) noexcept {
    return identity.active_agent_ids != nullptr
        && identity.active_count_state != nullptr
        && identity.population_next_id != nullptr
        && identity.slot_free_queue_count != nullptr
        && identity.capacity > 0
        && identity.active_id_cell_stride >= identity.capacity
        && (
            cell_count <= 1
            || identity.active_id_cell_stride
                <= kInt64MaximumV1 / cell_count
        );
}

__device__ __forceinline__ bool valid_birth_metrics_storage(
    const OrdinaryBirthMetricsSourcesV1& metrics,
    const std::int64_t cell_count
) noexcept {
    return metrics.birth_decision_count != nullptr
        && metrics.birth_reason_counts != nullptr
        && metrics.tick_count > 0
        && metrics.reason_count == kBirthPlanReasonCountV1
        && metrics.decision_cell_stride >= metrics.tick_count
        && metrics.reason_tick_stride >= metrics.reason_count
        && positive_product_fits_i64(
            metrics.tick_count, metrics.reason_tick_stride
        )
        && metrics.reason_cell_stride
            >= metrics.tick_count * metrics.reason_tick_stride
        && (
            cell_count <= 1
            || metrics.decision_cell_stride
                <= kInt64MaximumV1 / cell_count
        )
        && (
            cell_count <= 1
            || metrics.reason_cell_stride
                <= kInt64MaximumV1 / cell_count
        );
}

__device__ __forceinline__ bool exact_nonnegative_i64(
    const double value,
    std::int64_t* const output
) noexcept {
    constexpr double kExclusiveInt64UpperBound = 9223372036854775808.0;
    if (!isfinite(value) || value < 0.0 || value >= kExclusiveInt64UpperBound) {
        return false;
    }
    const std::int64_t converted = static_cast<std::int64_t>(value);
    if (static_cast<double>(converted) != value) {
        return false;
    }
    *output = converted;
    return true;
}

__device__ __forceinline__ bool valid_summary_row(
    const double* const summary,
    std::int64_t* const generation,
    std::int64_t* const cursor_start,
    std::int64_t* const cursor_pre_turnover,
    std::int64_t* const cursor_end
) noexcept {
    for (std::int64_t column = 0; column < kGenerationSummaryScalarWidthV1;
         ++column) {
        const double value = summary[column];
        if (!isfinite(value)) {
            return false;
        }
        if (
            column != kGenerationSummaryEnergySumColumnV1
            && column != kGenerationSummaryPeakInjuryColumnV1
            && column != kGenerationSummaryPatchFoodStockColumnV1
            && column != kGenerationSummaryEnergyCollectedExactSumColumnV1
            && column != kGenerationSummaryPolicyEnergyLiveSumColumnV1
        ) {
            std::int64_t ignored = 0;
            if (!exact_nonnegative_i64(value, &ignored)) {
                return false;
            }
        }
    }
    return exact_nonnegative_i64(
               summary[kGenerationSummaryGenerationColumnV1], generation
           )
        && exact_nonnegative_i64(
            summary[kGenerationSummaryCursorStartColumnV1], cursor_start
        )
        && exact_nonnegative_i64(
            summary[kGenerationSummaryCursorPreTurnoverColumnV1],
            cursor_pre_turnover
        )
        && exact_nonnegative_i64(
            summary[kGenerationSummaryCursorEndColumnV1], cursor_end
        );
}

__device__ __forceinline__ bool exact_reason_value(
    const float value,
    const std::int32_t reason
) noexcept {
    return isfinite(value) && value == static_cast<float>(reason);
}

__device__ __forceinline__ bool valid_typed_event_row(
    const TypedLifecycleColumnsV1& columns,
    const std::int64_t offset,
    const std::int64_t row
) noexcept {
    const std::int32_t event_type = columns.event_type[offset];
    const std::int64_t actor_id = columns.actor_agent_id[offset];
    const std::int64_t target_id = columns.target_agent_id[offset];
    const std::int64_t actor_slot = columns.actor_slot[offset];
    const std::int64_t target_slot = columns.target_slot[offset];
    const std::int64_t parent_id = columns.parent_agent_id[offset];
    const std::int64_t child_id = columns.child_agent_id[offset];
    const std::int32_t reason = columns.reason_code[offset];
    const std::int8_t success = columns.success[offset];
    const float value = columns.value[offset];
    if (
        event_type < kBirthCommittedEventTypeV1
        || event_type > kDeathCommittedEventTypeV1
        || columns.generation[offset] < 0
        || columns.tick[offset] < 0
        || columns.sequence[offset] != row
        || columns.subtype_code[offset] != event_type
        || actor_id < 0
        || actor_slot < 0
        || columns.lineage_founder_id[offset] < 0
        || columns.lineage_depth[offset] < 0
        || !isfinite(value)
    ) {
        return false;
    }
    if (event_type == kBirthCommittedEventTypeV1) {
        return success == 1
            && reason == kSentinelIntegerV1
            && target_id >= 0
            && target_slot >= 0
            && parent_id == actor_id
            && child_id == target_id;
    }
    if (event_type == kMutationEventTypeV1) {
        return success == 1
            && reason == kSentinelIntegerV1
            && target_id >= 0
            && target_slot >= 0
            && parent_id >= 0
            && child_id == target_id
            && actor_id == target_id;
    }
    if (event_type == kBirthFailedEventTypeV1) {
        return success == 0
            && reason >= kMinimumBirthFailureReasonCodeV1
            && reason <= kMaximumBirthFailureReasonCodeV1
            && exact_reason_value(value, reason)
            && target_id == kSentinelIntegerV1
            && target_slot == kSentinelIntegerV1
            && parent_id == kSentinelIntegerV1
            && child_id == kSentinelIntegerV1;
    }
    return success == 1
        && (reason == kEnergyDepletedReasonCodeV1
            || reason == kHealthDepletedReasonCodeV1)
        && exact_reason_value(value, reason)
        && target_id == kSentinelIntegerV1
        && target_slot == kSentinelIntegerV1
        && parent_id == kSentinelIntegerV1
        && child_id == kSentinelIntegerV1;
}

__device__ __forceinline__ ReductionResultV1 reduce_cell(
    const ReductionInputsV1& inputs,
    const std::int64_t cell
) noexcept {
    if (
        cell < 0
        || cell >= inputs.cell_count
        || inputs.cell_count <= 0
        || !valid_event_storage(inputs.lifecycle, inputs.cell_count)
        || !valid_generation_summary_storage(inputs)
        || !valid_identity_storage(inputs.identity, inputs.cell_count)
        || !valid_birth_metrics_storage(
            inputs.ordinary_birth_metrics, inputs.cell_count
        )
    ) {
        return invalid_result(kReductionInvalidStorageV1);
    }

    const std::int64_t summary_row_index = generation_major_row_index_v1(
        inputs.generation_index, inputs.cell_count, cell
    );
    const double* const summary = inputs.generation_summary_scalar.values
        + summary_row_index * kGenerationSummaryScalarWidthV1;
    std::int64_t generation = 0;
    std::int64_t cursor_start = 0;
    std::int64_t cursor_pre_turnover = 0;
    std::int64_t cursor_end = 0;
    if (!valid_summary_row(
            summary,
            &generation,
            &cursor_start,
            &cursor_pre_turnover,
            &cursor_end
        )) {
        return invalid_result(kReductionCursorContradictionV1);
    }

    const std::int64_t active_event_count =
        inputs.lifecycle.active_count[cell];
    if (
        cursor_start > cursor_pre_turnover
        || cursor_pre_turnover > cursor_end
        || cursor_end > inputs.lifecycle.typed.capacity
        || active_event_count != cursor_end
    ) {
        return invalid_result(kReductionCursorContradictionV1);
    }

    ReductionResultV1 result{};
    result.status = kReductionSuccessV1;
    result.cursor_start = cursor_start;
    result.cursor_pre_turnover = cursor_pre_turnover;
    result.cursor_end = cursor_end;
    result.authoritative_cursor = cursor_end;
    result.event_row_count = cursor_end - cursor_start;
    result.active_event_count = cursor_end;
    result.typed_event_row_count = cursor_end;
    result.tape_event_row_count = cursor_end;

    const OrdinaryBirthMetricsSourcesV1& birth_metrics =
        inputs.ordinary_birth_metrics;
    const std::int64_t decision_base =
        cell * birth_metrics.decision_cell_stride;
    const std::int64_t reason_base = cell * birth_metrics.reason_cell_stride;
    std::int64_t accepted_birth_decisions = 0;
    for (std::int64_t tick = 0; tick < birth_metrics.tick_count; ++tick) {
        const std::int64_t decision_count =
            birth_metrics.birth_decision_count[decision_base + tick];
        if (decision_count < 0) {
            return invalid_result(kReductionBirthMetricsContradictionV1);
        }
        std::int64_t tick_reason_total = 0;
        for (
            std::int64_t reason = 0;
            reason < kBirthPlanReasonCountV1;
            ++reason
        ) {
            const std::int64_t count = birth_metrics.birth_reason_counts[
                reason_base + tick * birth_metrics.reason_tick_stride + reason
            ];
            if (count < 0) {
                return invalid_result(kReductionBirthMetricsContradictionV1);
            }
            if (
                tick_reason_total
                > kInt64MaximumV1 - count
            ) {
                return invalid_result(kReductionBirthMetricsContradictionV1);
            }
            tick_reason_total += count;
            if (reason == kBirthPlanAcceptedReasonCodeV1) {
                if (
                    accepted_birth_decisions
                    > kInt64MaximumV1 - count
                ) {
                    return invalid_result(kReductionBirthMetricsContradictionV1);
                }
                accepted_birth_decisions += count;
            } else {
                if (
                    result.birth_failure_counts_by_reason[reason - 1]
                    > kInt64MaximumV1 - count
                ) {
                    return invalid_result(kReductionBirthMetricsContradictionV1);
                }
                result.birth_failure_counts_by_reason[reason - 1] += count;
            }
            if (counts_as_birth_attempt_v1(reason)) {
                if (
                    result.birth_attempt_rows
                    > kInt64MaximumV1 - count
                ) {
                    return invalid_result(kReductionBirthMetricsContradictionV1);
                }
                result.birth_attempt_rows += count;
            }
            if (counts_as_failed_birth_row_v1(reason)) {
                if (
                    result.birth_failure_rows
                    > kInt64MaximumV1 - count
                ) {
                    return invalid_result(kReductionBirthMetricsContradictionV1);
                }
                result.birth_failure_rows += count;
            }
        }
        if (
            tick_reason_total != decision_count
            || birth_metrics.birth_reason_counts[
                reason_base
                + tick * birth_metrics.reason_tick_stride
                + kBirthPlanUnusedReasonCodeV1
            ] != 0
        ) {
            return invalid_result(kReductionBirthMetricsContradictionV1);
        }
    }
    if (
        accepted_birth_decisions
            + result.birth_failure_counts_by_reason[
                kBirthPlanNoFreeSlotReasonCodeV1 - 1
            ]
            + result.birth_failure_counts_by_reason[
                kBirthPlanConceptionReasonCodeV1 - 1
            ]
            + result.birth_failure_counts_by_reason[
                kBirthPlanGestationReasonCodeV1 - 1
            ]
            != result.birth_attempt_rows
    ) {
        return invalid_result(kReductionBirthMetricsContradictionV1);
    }
    // The current Flat envelope rejects hard injury_gate_active. Reason 7 in
    // birth_reason_counts is health, so retain the gate metric as explicit zero.
    result.injury_blocked_births = 0;

    const TypedLifecycleColumnsV1& events = inputs.lifecycle.typed;
    const std::int64_t event_cell_offset = cell * events.cell_stride;
    for (std::int64_t row = 0; row < cursor_end; ++row) {
        const std::int64_t offset = event_cell_offset + row;
        if (!valid_typed_event_row(events, offset, row)) {
            return invalid_result(kReductionEventSemanticContradictionV1);
        }
        if (row < cursor_start) {
            continue;
        }
        if (events.generation[offset] != generation) {
            return invalid_result(kReductionEventSemanticContradictionV1);
        }
        const std::int32_t event_type = events.event_type[offset];
        const std::int64_t type_index = event_type - 1;
        if (row < cursor_pre_turnover) {
            ++result.generation_counts_by_type[type_index];
            ++result.generation_event_count;
            if (event_type == kBirthFailedEventTypeV1) {
                const std::int32_t reason = events.reason_code[offset];
                ++result.generation_failed_birth_counts_by_reason[reason - 1];
                ++result.native_failed_birth_event_rows;
            }
            if (event_type == kDeathCommittedEventTypeV1) {
                const std::int32_t reason = events.reason_code[offset];
                if (reason == kEnergyDepletedReasonCodeV1) {
                    ++result.deaths_energy_depleted;
                } else {
                    ++result.deaths_health_depleted;
                }
            }
        } else {
            ++result.turnover_counts_by_type[type_index];
            ++result.turnover_event_count;
            if (event_type == kBirthFailedEventTypeV1) {
                const std::int32_t reason = events.reason_code[offset];
                ++result.turnover_failed_birth_counts_by_reason[reason - 1];
            }
        }
    }
    std::int64_t generation_failed_birth_reason_total = 0;
    std::int64_t turnover_failed_birth_reason_total = 0;
    for (
        std::int64_t reason = 0;
        reason < kBirthFailureReasonCountV1;
        ++reason
    ) {
        generation_failed_birth_reason_total +=
            result.generation_failed_birth_counts_by_reason[reason];
        turnover_failed_birth_reason_total +=
            result.turnover_failed_birth_counts_by_reason[reason];
    }
    if (
        result.generation_event_count != cursor_pre_turnover - cursor_start
        || result.turnover_event_count != cursor_end - cursor_pre_turnover
        || generation_failed_birth_reason_total
            != result.generation_counts_by_type[
                kBirthFailedEventTypeV1 - 1
            ]
        || turnover_failed_birth_reason_total
            != result.turnover_counts_by_type[
                kBirthFailedEventTypeV1 - 1
            ]
        || result.native_failed_birth_event_rows
            != generation_failed_birth_reason_total
        || result.native_failed_birth_event_rows
            != result.birth_failure_counts_by_reason[0]
                + result.birth_failure_counts_by_reason[1]
                + result.birth_failure_counts_by_reason[2]
        || result.generation_failed_birth_counts_by_reason[0]
            != result.birth_failure_counts_by_reason[0]
        || result.generation_failed_birth_counts_by_reason[1]
            != result.birth_failure_counts_by_reason[1]
        || result.generation_failed_birth_counts_by_reason[2]
            != result.birth_failure_counts_by_reason[2]
        || accepted_birth_decisions
            != result.generation_counts_by_type[
                kBirthCommittedEventTypeV1 - 1
            ]
        || result.deaths_energy_depleted + result.deaths_health_depleted
            != result.generation_counts_by_type[
                kDeathCommittedEventTypeV1 - 1
            ]
    ) {
        return invalid_result(kReductionEventSemanticContradictionV1);
    }
    for (
        std::int64_t reason = 3;
        reason < kBirthFailureReasonCountV1;
        ++reason
    ) {
        if (result.generation_failed_birth_counts_by_reason[reason] != 0) {
            return invalid_result(kReductionEventSemanticContradictionV1);
        }
    }
    const std::int64_t summary_births = static_cast<std::int64_t>(
        summary[kGenerationSummaryBirthsColumnV1]
    );
    const std::int64_t summary_deaths = static_cast<std::int64_t>(
        summary[kGenerationSummaryDeathsColumnV1]
    );
    const std::int64_t summary_failed_births = static_cast<std::int64_t>(
        summary[kGenerationSummaryFailedBirthsColumnV1]
    );
    if (
        summary_births
            != result.generation_counts_by_type[
                kBirthCommittedEventTypeV1 - 1
            ]
        || summary_deaths
            != result.generation_counts_by_type[
                kDeathCommittedEventTypeV1 - 1
            ]
        || summary_failed_births != result.native_failed_birth_event_rows
    ) {
        return invalid_result(kReductionEventSemanticContradictionV1);
    }

    const std::int64_t active_count =
        inputs.identity.active_count_state[cell];
    const std::int64_t population_next_id =
        inputs.identity.population_next_id[cell];
    const std::int64_t free_slot_count =
        inputs.identity.slot_free_queue_count[cell];
    if (
        active_count < 0
        || active_count > inputs.identity.capacity
        || population_next_id < 1
    ) {
        return invalid_result(kReductionIdentityContradictionV1);
    }
    const std::int64_t identity_cell_offset =
        cell * inputs.identity.active_id_cell_stride;
    std::int64_t previous_id = kSentinelIntegerV1;
    for (std::int64_t row = 0; row < active_count; ++row) {
        const std::int64_t agent_id =
            inputs.identity.active_agent_ids[identity_cell_offset + row];
        if (agent_id < 0 || (row > 0 && agent_id <= previous_id)) {
            return invalid_result(kReductionIdentityContradictionV1);
        }
        previous_id = agent_id;
    }
    if (active_count > 0 && previous_id >= population_next_id) {
        return invalid_result(kReductionIdentityContradictionV1);
    }
    if (
        free_slot_count < 0
        || free_slot_count > inputs.identity.capacity
        || active_count + free_slot_count != inputs.identity.capacity
    ) {
        return invalid_result(kReductionSlotConservationContradictionV1);
    }

    result.population_next_id = population_next_id;
    result.active_agent_count = active_count;
    result.free_slot_count = free_slot_count;
    if (active_count > 0) {
        result.active_id_bounds_defined = 1;
        result.active_agent_id_min =
            inputs.identity.active_agent_ids[identity_cell_offset];
        result.active_agent_id_max = previous_id;
    }
    return result;
}

#endif  // defined(__CUDACC__)

// Queues exactly one kernel and performs no allocation, transfer, or
// synchronization. Invalid host-visible shapes return without touching output.
ReductionLaunchStatusV1 launch_compact_summary_reductions_v1(
    ReductionInputsV1 inputs,
    ReductionOutputViewV1 output,
    CUstream_st* stream
);

KernelResourceReceiptV1 query_compact_summary_reductions_v1_resources();

static_assert(std::is_standard_layout_v<ReductionInputsV1>);
static_assert(std::is_trivially_copyable_v<ReductionInputsV1>);
static_assert(std::is_standard_layout_v<ReductionOutputViewV1>);
static_assert(std::is_trivially_copyable_v<ReductionOutputViewV1>);
static_assert(std::is_standard_layout_v<ReductionResultV1>);
static_assert(std::is_trivially_copyable_v<ReductionResultV1>);
static_assert(sizeof(ReductionResultV1) == 456);
static_assert(offsetof(ReductionResultV1, generation_event_count) == 8);
static_assert(offsetof(ReductionResultV1, native_failed_birth_event_rows) == 304);
static_assert(offsetof(ReductionResultV1, birth_attempt_rows) == 320);
static_assert(offsetof(ReductionResultV1, free_slot_count) == 448);

}  // namespace eiel::cuda_architecture_b::compact_summary_reductions_v1

#endif  // EIEL_CUDA_ARCHITECTURE_B_COMPACT_SUMMARY_REDUCTIONS_V1_CUH_
