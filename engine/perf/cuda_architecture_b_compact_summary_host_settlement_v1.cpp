#include "cuda_architecture_b_compact_summary_host_settlement_v1.hpp"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <utility>

namespace eiel::cuda_architecture_b::compact_summary_host_settlement_v1 {

namespace {

namespace abi = eiel::cuda_architecture_b::compact_summary_v2;
namespace renderer =
    eiel::cuda_architecture_b::compact_summary_host_template_renderer_v1;
namespace runtime =
    eiel::cuda_architecture_b::compact_summary_host_template_runtime_v1;

constexpr std::size_t kLifecycleDigestOffsetV1 = 0;
constexpr std::size_t kBehaviorDigestOffsetV1 = 8 * 32;
constexpr std::size_t kLineageDigestOffsetV1 = 9 * 32;

enum GenerationSummaryColumnV1 : std::int64_t {
    kSummaryGenerationV1 = 0,
    kSummaryCursorStartV1,
    kSummaryCursorPreTurnoverV1,
    kSummaryCursorEndV1,
    kSummaryAliveV1,
    kSummaryEnergySumV1,
    kSummaryBirthsV1,
    kSummaryDeathsV1,
    kSummaryFailedBirthsV1,
    kSummaryActionTotalV1,
    kSummaryMoveTotalV1,
    kSummaryGatherTotalV1,
    kSummaryMovedV1,
    kSummaryGatheredV1,
    kSummaryMovementAttemptsV1,
    kSummaryMovementBlockedOccupiedV1,
    kSummaryMovementBlockedBoundsV1,
    kSummaryPeakInjuryV1,
    kSummaryPolicyLoopTickCallsV1,
    kSummaryPolicyRefillCallsV1,
    kSummaryPolicyRefillRowsV1,
    kSummaryPolicyCacheHitCallsV1,
    kSummaryPolicyCacheRowsRequestedV1,
    kSummaryPolicyCacheRowsHitV1,
    kSummaryPolicyCacheRowsMissedV1,
    kSummaryPolicyCacheRowsRefilledV1,
    kSummaryPolicyCacheFullRefillTicksV1,
    kSummaryPolicyCachePartialRefillTicksV1,
    kSummaryPatchFoodStockAfterTurnoverV1,
    kSummaryEnergyCollectedExactSumAfterTurnoverV1,
    kSummaryPolicyEnergyLiveSumAfterTurnoverV1,
    kSummaryWidthV1,
};

enum PatchMetricColumnV1 : std::int64_t {
    kPatchMovementEnergyEventCountV1 = 0,
    kPatchSpawnSyncRowsV1,
    kPatchTypedDeathEventRowsV1,
    kPatchEnvRngPaddingDrawsV1,
    kPatchFoodHarvestTotalV1,
    kPatchScentEmitTotalV1,
    kPatchWasteEmitTotalV1,
    kPatchMovementEnergyTotalV1,
    kPatchEnergyCollectedDeltaV1,
    kPatchScentMoveQueuedTotalV1,
    kPatchWasteMoveQueuedTotalV1,
    kPatchScentGatherQueuedTotalV1,
    kPatchWasteGatherQueuedTotalV1,
    kPatchMovementEnergyExtraTotalV1,
    kPatchScentEmittedTotalV1,
    kPatchWasteEmittedTotalV1,
    kPatchFoodDepletedFracV1,
    kPatchFoodFullFracV1,
    kPatchDensityHotspotFracV1,
    kPatchMetricWidthInternalV1,
};

static_assert(kPatchMetricWidthInternalV1 == kPatchMetricWidthV1);

bool exact_nonnegative_integer(
    const double value,
    std::int64_t* const output
) noexcept {
    constexpr double exclusive_upper = 9223372036854775808.0;
    if (
        output == nullptr
        || !std::isfinite(value)
        || value < 0.0
        || value >= exclusive_upper
    ) {
        return false;
    }
    const std::int64_t converted = static_cast<std::int64_t>(value);
    if (static_cast<double>(converted) != value) {
        return false;
    }
    *output = converted;
    return true;
}

bool copy_generation_summary_integers(
    const double* const summary,
    std::int64_t output[kSummaryWidthV1]
) noexcept {
    if (summary == nullptr || output == nullptr) {
        return false;
    }
    for (std::int64_t column = 0; column < kSummaryWidthV1; ++column) {
        if (
            column == kSummaryEnergySumV1
            || column == kSummaryPeakInjuryV1
            || column == kSummaryPatchFoodStockAfterTurnoverV1
            || column == kSummaryEnergyCollectedExactSumAfterTurnoverV1
            || column == kSummaryPolicyEnergyLiveSumAfterTurnoverV1
        ) {
            if (!std::isfinite(summary[column])) {
                return false;
            }
            output[column] = 0;
            continue;
        }
        if (!exact_nonnegative_integer(summary[column], output + column)) {
            return false;
        }
    }
    return true;
}

bool lineage_state_is_valid(
    const canonical::LineageMetricsStateV1& state
) noexcept {
    return state.collapse_streak >= 0
        && state.population_dominance_streak >= 0
        && state.reproduction_dominance_streak >= 0;
}

bool lineage_state_storage_is_valid(
    const HostSettlementContextV1& context
) noexcept {
    if (context.lineage_state_values == nullptr) {
        return context.cell_count == 1 && context.lineage_state_count == 0;
    }
    if (context.lineage_state_count != context.cell_count) {
        return false;
    }
    for (std::int64_t cell = 0; cell < context.cell_count; ++cell) {
        if (!lineage_state_is_valid(context.lineage_state_values[cell])) {
            return false;
        }
    }
    return true;
}

bool settlement_extent_is_safe_to_clear(
    const HostSettlementContextV1& context
) noexcept {
    return context.authority_rows != nullptr
        && context.generation_count > 0
        && context.cell_count > 0
        && context.generation_count
            <= std::numeric_limits<std::int32_t>::max() / context.cell_count
        && context.generation_count
            <= static_cast<std::int64_t>(
                   std::numeric_limits<std::size_t>::max()
                   / sizeof(
                       projection::CompactSummaryGenerationCellAuthorityV1
                   )
               )
                / context.cell_count;
}

bool render_outer_authority_digests(
    HostSettlementContextV1& context,
    const std::uint32_t generation,
    const std::uint32_t cell,
    projection::CompactSummaryGenerationCellAuthorityV1& authority
) noexcept {
    const std::pair<renderer::HostTemplateSurfaceV1, abi::DigestFieldV2>
        surfaces[] = {
            {
                renderer::HostTemplateSurfaceV1::kCheckpointTrajectory,
                abi::kDigestStateDigestsAuthoritativeStateDigestSha256,
            },
            {
                renderer::HostTemplateSurfaceV1::kCanonicalCheckpointStorage,
                abi::kDigestStateDigestsCanonicalCheckpointStorageDigestSha256,
            },
            {
                renderer::HostTemplateSurfaceV1::kNextGenerationSource,
                abi::kDigestStateDigestsNextGenerationStartStateDigestSha256,
            },
            {
                renderer::HostTemplateSurfaceV1::kRng,
                abi::kDigestRngAuthorityDigestSha256,
            },
            {
                renderer::HostTemplateSurfaceV1::kActiveAgentIds,
                abi::kDigestIdentityActiveAgentIdsDigestSha256,
            },
            {
                renderer::HostTemplateSurfaceV1::kPolicyMutationTape,
                abi::kDigestRngPolicyMutationTapeDigestSha256,
            },
        };
    for (const auto& [surface, digest_field] : surfaces) {
        const auto binding = runtime::lookup_host_template_program_v1(
            context.host_template_runtime,
            generation,
            cell,
            static_cast<std::uint32_t>(surface)
        );
        if (binding.program == nullptr || binding.token_values == nullptr) {
            return false;
        }
        renderer::HostTemplateResultV1 result{};
        const renderer::HostTemplateRequestV1 request = {
            renderer::kHostTemplateRendererAbiVersionV1,
            static_cast<std::uint32_t>(
                renderer::HostTemplateRenderModeV1::kHashOnly
            ),
            binding.program,
            binding.token_values,
            binding.token_value_count,
            0,
            nullptr,
            0,
            &result,
            0,
        };
        const auto status = renderer::render_and_hash_host_template_v1(&request);
        if (status != renderer::HostTemplateStatusV1::kSuccess
            || result.status != static_cast<std::uint32_t>(
                renderer::HostTemplateStatusV1::kSuccess
            )) {
            return false;
        }
        std::memcpy(
            authority.digest_values
                + static_cast<std::int64_t>(digest_field)
                    * abi::kDigestBytesV2,
            result.digest_sha256,
            abi::kDigestBytesV2
        );
    }
    return true;
}

bool context_is_valid(const HostSettlementContextV1& context) noexcept {
    const bool reporting_capacity_valid = context.reporting_row_capacity > 0
        && context.reporting_row_capacity
            <= reporting::kMaximumDynamicRowCapacityV1;
    const std::int64_t retained_row_bytes = reporting_capacity_valid
        ? static_cast<std::int64_t>(
            reporting::retained_bytes_per_generation_cell_v1(
                static_cast<std::int32_t>(context.reporting_row_capacity)
            )
        )
        : 0;
    return context.authority_rows != nullptr
        && context.reporting_retained_bytes != nullptr
        && context.generation_summary_values != nullptr
        && context.lifecycle_digest_values != nullptr
        && context.lifecycle_status_values != nullptr
        && context.lifecycle_validity_values != nullptr
        && context.checkpoint_raw_digests != nullptr
        && context.checkpoint_canonical_digests != nullptr
        && context.checkpoint_status_values != nullptr
        && context.checkpoint_validity_values != nullptr
        && context.reduction_rows != nullptr
        && context.metadata_rows != nullptr
        && context.checkpoint_due_values != nullptr
        && context.import_digest_values != nullptr
        && context.host_template_runtime != nullptr
        && context.completed_generation_values != nullptr
        && context.host_token_resolver_rows != nullptr
        && context.checkpoint_descriptor_specs != nullptr
        && context.host_token_resolver_row_count > 0
        && context.generation_count > 0
        && context.generation_count <= UINT32_MAX
        && context.cell_count > 0
        && context.cell_count <= UINT32_MAX
        && context.generation_count
            <= std::numeric_limits<std::int64_t>::max() / context.cell_count
        && context.generation_count
            <= std::numeric_limits<std::int32_t>::max() / context.cell_count
        && lineage_state_storage_is_valid(context)
        && context.checkpoint_field_count > 0
        && context.checkpoint_descriptor_count == context.checkpoint_field_count
        && context.generation_count * context.cell_count
            <= std::numeric_limits<std::int64_t>::max()
                / context.checkpoint_field_count
        && reporting_capacity_valid
        && context.reporting_retained_row_bytes == retained_row_bytes
        && context.generation_summary_width == kSummaryWidthV1
        && (
            (context.world_code == abi::kWorldFlat
                && context.patch_metrics_values == nullptr
                && context.patch_metrics_width == 0)
            || (context.world_code == abi::kWorldPatch
                && context.patch_metrics_values != nullptr
                && context.patch_metrics_width == kPatchMetricWidthV1)
        )
        && context.reporting_row_capacity <= UINT16_MAX;
}

}  // namespace

void settle_compact_summary_rows_callback_v1(void* user_data) noexcept {
    if (user_data == nullptr) {
        return;
    }
    auto* const context = static_cast<HostSettlementContextV1*>(user_data);
    context->completed_rows = 0;
    context->first_failed_row = -1;
    context->status = kHostSettlementInvalidArgumentsV1;
    if (settlement_extent_is_safe_to_clear(*context)) {
        const std::int64_t requested_row_count =
            context->generation_count * context->cell_count;
        std::memset(
            context->authority_rows,
            0,
            static_cast<std::size_t>(requested_row_count)
                * sizeof(projection::CompactSummaryGenerationCellAuthorityV1)
        );
    }
    if (!context_is_valid(*context)) {
        return;
    }

    std::int64_t completed_generation_count = context->generation_count;
    for (std::int64_t cell = 0; cell < context->cell_count; ++cell) {
        const std::int64_t next_generation =
            context->completed_generation_values[cell];
        if (next_generation < context->initial_generation
            || next_generation
                > context->initial_generation + context->generation_count) {
            return;
        }
        completed_generation_count = std::min(
            completed_generation_count,
            next_generation - context->initial_generation
        );
    }
    if (completed_generation_count == 0) {
        context->first_failed_row = 0;
        context->status = kHostSettlementPartialPrefixV1;
        return;
    }

    const token_resolver::HostTokenResolverRequestV1 resolver_request = {
        context->host_token_resolver_rows,
        context->host_token_resolver_row_count,
        context->host_template_runtime,
        context->checkpoint_descriptor_specs,
        context->checkpoint_raw_digests,
        context->checkpoint_canonical_digests,
        context->checkpoint_status_values,
        context->checkpoint_validity_values,
        context->metadata_rows,
        context->import_digest_values,
        context->generation_count,
        completed_generation_count,
        context->cell_count,
        context->checkpoint_descriptor_count,
        &context->token_resolver_failure,
    };
    if (token_resolver::resolve_compact_summary_host_tokens_v1(
            &resolver_request
        ) != token_resolver::HostTokenResolverStatusV1::kSuccess) {
        context->status = kHostSettlementTokenResolutionInvalidV1;
        if (
            context->token_resolver_failure.generation
                != token_resolver::kNoResolverCoordinateV1
            && context->token_resolver_failure.cell
                != token_resolver::kNoResolverCoordinateV1
        ) {
            context->first_failed_row = static_cast<std::int32_t>(
                static_cast<std::int64_t>(
                    context->token_resolver_failure.generation
                ) * context->cell_count
                + context->token_resolver_failure.cell
            );
        }
        return;
    }

    const std::int64_t row_count =
        completed_generation_count * context->cell_count;
    canonical::LineageMetricsStateV1 legacy_single_cell_lineage_state =
        canonical::kDefaultLineageMetricsInputV1.initial_state;
    for (std::int64_t row_index = 0; row_index < row_count; ++row_index) {
        auto& authority = context->authority_rows[row_index];
        const std::int64_t generation = row_index / context->cell_count;
        const std::int64_t cell = row_index % context->cell_count;
        authority.source_version = projection::kAuthoritySourceVersionV1;
        authority.disposition = projection::kAuthorityRowComplete;
        authority.generation_index = generation;
        authority.cell_index = cell;
        authority.validity = 0;

        const std::int64_t checkpoint_row_offset =
            row_index * context->checkpoint_field_count;
        for (
            std::int64_t field = 0;
            field < context->checkpoint_field_count;
            ++field
        ) {
            const std::int64_t index = checkpoint_row_offset + field;
            if (context->checkpoint_status_values[index] != 0
                || context->checkpoint_validity_values[index] != 1U) {
                context->status = kHostSettlementCheckpointInvalidV1;
                context->first_failed_row = static_cast<std::int32_t>(row_index);
                return;
            }
        }
        const auto& reduction = context->reduction_rows[row_index];
        if (reduction.status != reductions::kReductionSuccessV1
            || reduction.valid != 1U) {
            context->status = kHostSettlementReductionInvalidV1;
            context->first_failed_row = static_cast<std::int32_t>(row_index);
            return;
        }
        const auto& metadata_row = context->metadata_rows[row_index];
        if (metadata_row.status != metadata::kRetentionSuccessV1
            || metadata_row.validity != 1U
            || (metadata_row.flags & metadata::kRequiredRetentionFlagsV1)
                != metadata::kRequiredRetentionFlagsV1) {
            context->status = kHostSettlementMetadataInvalidV1;
            context->first_failed_row = static_cast<std::int32_t>(row_index);
            return;
        }

        const double* const generation_summary =
            context->generation_summary_values
            + row_index * context->generation_summary_width;
        std::int64_t summary_integers[kSummaryWidthV1]{};
        if (!copy_generation_summary_integers(
                generation_summary,
                summary_integers
            )) {
            context->status = kHostSettlementReductionInvalidV1;
            context->first_failed_row = static_cast<std::int32_t>(row_index);
            return;
        }

        if (context->lifecycle_status_values[row_index] != 0
            || context->lifecycle_validity_values[row_index] != 1U) {
            context->status = kHostSettlementLifecycleInvalidV1;
            context->first_failed_row = static_cast<std::int32_t>(row_index);
            return;
        }
        std::memcpy(
            authority.digest_values + kLifecycleDigestOffsetV1,
            context->lifecycle_digest_values + row_index * 32,
            32
        );

        const std::uint8_t* const retained =
            context->reporting_retained_bytes
            + row_index * context->reporting_retained_row_bytes;
        const auto* const header =
            reinterpret_cast<const reporting::GenerationAuthorityHeaderV1*>(
                retained
            );
        const auto* const behavior =
            reinterpret_cast<const reporting::BehaviorAuthorityRowV1*>(
                retained + sizeof(reporting::GenerationAuthorityHeaderV1)
            );
        const auto* const lineage =
            reinterpret_cast<const reporting::LineageAuthorityRowV1*>(
                retained + sizeof(reporting::GenerationAuthorityHeaderV1)
                + static_cast<std::size_t>(context->reporting_row_capacity)
                    * sizeof(reporting::BehaviorAuthorityRowV1)
            );
        if (
            static_cast<std::int64_t>(header->lineage_record_count)
            > metadata_row.capacity
        ) {
            context->status = kHostSettlementReportingInvalidV1;
            context->first_failed_row = static_cast<std::int32_t>(row_index);
            return;
        }
        canonical::HostCanonicalResultV1 canonical_result{};
        canonical::LineageMetricsInputV1 lineage_metrics_input =
            canonical::kDefaultLineageMetricsInputV1;
        canonical::LineageMetricsStateV1* const lineage_state =
            context->lineage_state_values != nullptr
                ? context->lineage_state_values + cell
                : &legacy_single_cell_lineage_state;
        lineage_metrics_input.initial_state = *lineage_state;
        if (canonical::canonicalize_retained_summary_v1(
                summary_integers[kSummaryGenerationV1],
                header,
                behavior,
                lineage,
                context->reporting_row_capacity,
                &lineage_metrics_input,
                &canonical_result
            ) != canonical::HostCanonicalStatusV1::kSuccess) {
            context->status = kHostSettlementReportingInvalidV1;
            context->first_failed_row = static_cast<std::int32_t>(row_index);
            return;
        }
        authority.status_codes[abi::kStatusSummaryStatus] =
            abi::kSummaryStatusComplete;
        authority.status_codes[abi::kStatusWorld] = context->world_code;
        authority.status_codes[abi::kStatusTurnoverStatus] =
            abi::kTurnoverStatusCommitted;
        authority.status_codes[abi::kStatusRngAuthorityComplete] =
            abi::kBooleanStatusTrue;
        authority.status_codes[abi::kStatusRngExportNonConsuming] =
            abi::kBooleanStatusTrue;
        authority.status_codes[abi::kStatusTerminalStatus] =
            generation == context->generation_count - 1
                ? abi::kTerminalStatusRunComplete
                : abi::kTerminalStatusNotTerminal;
        authority.status_codes[abi::kStatusTerminalRecoverableErrorStatus] =
            abi::kRecoverableErrorStatusNone;
        authority.status_codes[abi::kStatusTerminalStickyFailure] =
            abi::kBooleanStatusFalse;
        authority.status_codes[abi::kStatusCheckpointDue] =
            context->checkpoint_due_values[row_index]
                ? abi::kBooleanStatusTrue
                : abi::kBooleanStatusFalse;

        auto* const integers = authority.integer_values;
        integers[abi::kIntegerGenerationStartGeneration] =
            summary_integers[kSummaryGenerationV1];
        integers[abi::kIntegerGenerationStartTick] = 0;
        integers[abi::kIntegerGenerationEndGeneration] =
            summary_integers[kSummaryGenerationV1];
        integers[abi::kIntegerGenerationEndTickExclusive] =
            metadata_row.configured_steps_per_generation;
        integers[abi::kIntegerNextGenerationStartGeneration] =
            metadata_row.generation;
        integers[abi::kIntegerNextGenerationStartTick] = metadata_row.tick;
        integers[abi::kIntegerAlive] = header->lineage_record_count;
        integers[abi::kIntegerBirths] = summary_integers[kSummaryBirthsV1];
        integers[abi::kIntegerFailedBirths] =
            summary_integers[kSummaryFailedBirthsV1];
        integers[abi::kIntegerDeaths] = summary_integers[kSummaryDeathsV1];
        integers[abi::kIntegerDeathsByClassEnergyDepleted] =
            reduction.deaths_energy_depleted;
        integers[abi::kIntegerDeathsByClassHealthDepleted] =
            reduction.deaths_health_depleted;
        integers[abi::kIntegerTurnoverCount] = 1;
        integers[abi::kIntegerEventsCursorStart] = reduction.cursor_start;
        integers[abi::kIntegerEventsCursorPreTurnover] =
            reduction.cursor_pre_turnover;
        integers[abi::kIntegerEventsCursorEnd] = reduction.cursor_end;
        integers[abi::kIntegerEventsAuthoritativeCursor] =
            reduction.authoritative_cursor;
        integers[abi::kIntegerEventsEventRowCount] = reduction.event_row_count;
        integers[abi::kIntegerEventsGenerationEventCount] =
            reduction.generation_event_count;
        integers[abi::kIntegerEventsTurnoverEventCount] =
            reduction.turnover_event_count;
        for (std::int64_t type = 0; type < reductions::kEventTypeCountV1; ++type) {
            integers[
                abi::kIntegerEventsGenerationCountsByTypeBirthCommitted + type
            ] = reduction.generation_counts_by_type[type];
            integers[
                abi::kIntegerEventsTurnoverCountsByTypeBirthCommitted + type
            ] = reduction.turnover_counts_by_type[type];
        }
        integers[abi::kIntegerIdentityPopulationNextId] =
            reduction.population_next_id;
        integers[abi::kIntegerIdentityActiveAgentCount] =
            reduction.active_agent_count;
        integers[abi::kIntegerIdentityFreeSlotCount] = reduction.free_slot_count;
        if (reduction.active_id_bounds_defined == 1U) {
            authority.defined_mask |=
                (abi::DefinedMaskV1{1} << abi::kDefinedIdentityActiveAgentIdMin)
                | (abi::DefinedMaskV1{1}
                   << abi::kDefinedIdentityActiveAgentIdMax);
            integers[abi::kIntegerIdentityActiveAgentIdMin] =
                reduction.active_agent_id_min;
            integers[abi::kIntegerIdentityActiveAgentIdMax] =
                reduction.active_agent_id_max;
        }
        integers[abi::kIntegerRngPolicyMutationTapeCursor] =
            metadata_row.policy_birth_tape_cursor;
        integers[abi::kIntegerGenerationMetricsCapacity] = metadata_row.capacity;
        integers[abi::kIntegerGenerationMetricsFreeSlots] =
            metadata_row.capacity
            - static_cast<std::int64_t>(header->lineage_record_count);
        integers[abi::kIntegerGenerationMetricsBirthAttempts] =
            reduction.birth_attempt_rows;
        integers[abi::kIntegerGenerationMetricsNativeFailedBirthEventRows] =
            reduction.native_failed_birth_event_rows;
        integers[abi::kIntegerGenerationMetricsInjuryBlockedBirths] =
            reduction.injury_blocked_births;
        integers[abi::kIntegerGenerationMetricsMoved] =
            summary_integers[kSummaryMovedV1];
        integers[abi::kIntegerGenerationMetricsGathered] =
            summary_integers[kSummaryGatheredV1];
        integers[abi::kIntegerGenerationMetricsMovementAttempts] =
            summary_integers[kSummaryMovementAttemptsV1];
        integers[abi::kIntegerGenerationMetricsMovementBlockedOccupied] =
            summary_integers[kSummaryMovementBlockedOccupiedV1];
        integers[abi::kIntegerGenerationMetricsMovementBlockedBounds] =
            summary_integers[kSummaryMovementBlockedBoundsV1];
        integers[abi::kIntegerGenerationMetricsPolicyLoopTickCalls] =
            summary_integers[kSummaryPolicyLoopTickCallsV1];
        integers[abi::kIntegerGenerationMetricsPolicyRefillCalls] =
            summary_integers[kSummaryPolicyRefillCallsV1];
        integers[abi::kIntegerGenerationMetricsPolicyRefillRows] =
            summary_integers[kSummaryPolicyRefillRowsV1];
        integers[abi::kIntegerGenerationMetricsPolicyCacheHitCalls] =
            summary_integers[kSummaryPolicyCacheHitCallsV1];
        integers[abi::kIntegerGenerationMetricsPolicyCacheRowsRequested] =
            summary_integers[kSummaryPolicyCacheRowsRequestedV1];
        integers[abi::kIntegerGenerationMetricsPolicyCacheRowsHit] =
            summary_integers[kSummaryPolicyCacheRowsHitV1];
        integers[abi::kIntegerGenerationMetricsPolicyCacheRowsMissed] =
            summary_integers[kSummaryPolicyCacheRowsMissedV1];
        integers[abi::kIntegerGenerationMetricsPolicyCacheRowsRefilled] =
            summary_integers[kSummaryPolicyCacheRowsRefilledV1];
        integers[abi::kIntegerGenerationMetricsPolicyCacheFullRefillTicks] =
            summary_integers[kSummaryPolicyCacheFullRefillTicksV1];
        integers[abi::kIntegerGenerationMetricsPolicyCachePartialRefillTicks] =
            summary_integers[kSummaryPolicyCachePartialRefillTicksV1];

        if (context->world_code == abi::kWorldPatch) {
            const double* const patch = context->patch_metrics_values
                + row_index * context->patch_metrics_width;
            std::int64_t movement_events = 0;
            std::int64_t spawn_sync_rows = 0;
            std::int64_t env_rng_padding_draws = 0;
            if (!exact_nonnegative_integer(
                    patch[kPatchMovementEnergyEventCountV1],
                    &movement_events
                )
                || !exact_nonnegative_integer(
                    patch[kPatchSpawnSyncRowsV1],
                    &spawn_sync_rows
                )
                || !exact_nonnegative_integer(
                    patch[kPatchEnvRngPaddingDrawsV1],
                    &env_rng_padding_draws
                )
                || patch[kPatchTypedDeathEventRowsV1] != 0.0) {
                context->status = kHostSettlementReductionInvalidV1;
                context->first_failed_row = static_cast<std::int32_t>(row_index);
                return;
            }
            for (
                std::int64_t field = kPatchFoodHarvestTotalV1;
                field < kPatchMetricWidthV1;
                ++field
            ) {
                if (!std::isfinite(patch[field])) {
                    context->status = kHostSettlementReductionInvalidV1;
                    context->first_failed_row =
                        static_cast<std::int32_t>(row_index);
                    return;
                }
            }
            for (const std::int64_t field : {
                     kPatchFoodHarvestTotalV1,
                     kPatchScentEmitTotalV1,
                     kPatchWasteEmitTotalV1,
                     kPatchEnergyCollectedDeltaV1,
                     kPatchScentMoveQueuedTotalV1,
                     kPatchWasteMoveQueuedTotalV1,
                     kPatchScentGatherQueuedTotalV1,
                     kPatchWasteGatherQueuedTotalV1,
                     kPatchScentEmittedTotalV1,
                     kPatchWasteEmittedTotalV1,
                 }) {
                if (patch[field] < 0.0) {
                    context->status = kHostSettlementReductionInvalidV1;
                    context->first_failed_row =
                        static_cast<std::int32_t>(row_index);
                    return;
                }
            }
            for (const std::int64_t field : {
                     kPatchFoodDepletedFracV1,
                     kPatchFoodFullFracV1,
                     kPatchDensityHotspotFracV1,
                 }) {
                if (patch[field] < 0.0 || patch[field] > 1.0) {
                    context->status = kHostSettlementReductionInvalidV1;
                    context->first_failed_row =
                        static_cast<std::int32_t>(row_index);
                    return;
                }
            }
            const std::int64_t generation_deaths =
                reduction.generation_counts_by_type[
                    reductions::kDeathCommittedEventTypeV1 - 1
                ];
            if (generation_deaths < 0) {
                context->status = kHostSettlementReductionInvalidV1;
                context->first_failed_row = static_cast<std::int32_t>(row_index);
                return;
            }

            authority.defined_mask |=
                (abi::DefinedMaskV1{1} << abi::kDefinedPatchMetrics);
            for (
                std::uint32_t bit =
                    abi::kDefinedPatchMetricsPatchSpawnSyncRows;
                bit <= abi::kDefinedPatchMetricsPatchDensityHotspotFrac;
                ++bit
            ) {
                authority.defined_mask |= abi::DefinedMaskV1{1} << bit;
            }
            integers[abi::kIntegerPatchMetricsPatchMovementEnergyEventCount] =
                movement_events;
            integers[abi::kIntegerPatchMetricsPatchSpawnSyncRows] =
                spawn_sync_rows;
            integers[abi::kIntegerPatchMetricsTypedDeathEventRows] =
                generation_deaths;
            integers[abi::kIntegerPatchMetricsEnvRngPaddingDraws] =
                env_rng_padding_draws;

            auto* const floats = authority.float_values;
            floats[abi::kFloatPatchMetricsPatchFoodHarvestTotal] =
                patch[kPatchFoodHarvestTotalV1];
            floats[abi::kFloatPatchMetricsPatchScentMoveQueuedTotal] =
                patch[kPatchScentMoveQueuedTotalV1];
            floats[abi::kFloatPatchMetricsPatchWasteMoveQueuedTotal] =
                patch[kPatchWasteMoveQueuedTotalV1];
            floats[abi::kFloatPatchMetricsPatchScentGatherQueuedTotal] =
                patch[kPatchScentGatherQueuedTotalV1];
            floats[abi::kFloatPatchMetricsPatchWasteGatherQueuedTotal] =
                patch[kPatchWasteGatherQueuedTotalV1];
            floats[abi::kFloatPatchMetricsPatchScentEmitTotal] =
                patch[kPatchScentEmitTotalV1];
            floats[abi::kFloatPatchMetricsPatchWasteEmitTotal] =
                patch[kPatchWasteEmitTotalV1];
            floats[abi::kFloatPatchMetricsPatchMovementEnergyTotal] =
                patch[kPatchMovementEnergyExtraTotalV1];
            floats[abi::kFloatPatchMetricsEnergyCollectedDelta] =
                patch[kPatchEnergyCollectedDeltaV1];
            floats[abi::kFloatPatchMetricsPatchMovementEnergyExtraTotal] =
                patch[kPatchMovementEnergyExtraTotalV1];
            floats[abi::kFloatPatchMetricsPatchScentEmittedTotal] =
                patch[kPatchScentEmittedTotalV1];
            floats[abi::kFloatPatchMetricsPatchWasteEmittedTotal] =
                patch[kPatchWasteEmittedTotalV1];
            floats[abi::kFloatPatchMetricsPatchFoodDepletedFrac] =
                patch[kPatchFoodDepletedFracV1];
            floats[abi::kFloatPatchMetricsPatchFoodFullFrac] =
                patch[kPatchFoodFullFracV1];
            floats[abi::kFloatPatchMetricsPatchDensityHotspotFrac] =
                patch[kPatchDensityHotspotFracV1];
        }

        const auto& behavior_aggregate = canonical_result.behavior;
        integers[abi::kIntegerBehaviorMetricsRowCount] =
            behavior_aggregate.row_count;
        integers[abi::kIntegerBehaviorMetricsCohortSize] =
            behavior_aggregate.cohort_size;
        integers[abi::kIntegerBehaviorMetricsCohortSurvivors] =
            behavior_aggregate.cohort_survivors;
        integers[abi::kIntegerBehaviorMetricsActionTotal] =
            behavior_aggregate.action_total;
        integers[abi::kIntegerBehaviorMetricsMoveTotal] =
            behavior_aggregate.move_total;
        integers[abi::kIntegerBehaviorMetricsMoveN] = behavior_aggregate.move_n;
        integers[abi::kIntegerBehaviorMetricsMoveS] = behavior_aggregate.move_s;
        integers[abi::kIntegerBehaviorMetricsMoveE] = behavior_aggregate.move_e;
        integers[abi::kIntegerBehaviorMetricsMoveW] = behavior_aggregate.move_w;
        integers[abi::kIntegerBehaviorMetricsGather] = behavior_aggregate.gather;
        integers[abi::kIntegerBehaviorMetricsRest] = behavior_aggregate.rest;
        integers[abi::kIntegerBehaviorMetricsAttack] = behavior_aggregate.attack;

        const auto& lineage_summary = canonical_result.lineage;
        authority.defined_mask |= lineage_summary.compact_abi_v2_defined_mask;
        integers[abi::kIntegerLineageMetricsRecordCount] =
            lineage_summary.record_count;
        integers[abi::kIntegerLineageMetricsActiveLineages] =
            lineage_summary.active_lineages;
        integers[abi::kIntegerLineageMetricsRarePersistentLineages] =
            lineage_summary.rare_persistent_lineages;
        integers[abi::kIntegerLineageMetricsLineageCollapseStreak] =
            lineage_summary.collapse_streak;
        if (
            authority.defined_mask
            & (abi::DefinedMaskV1{1}
               << abi::kDefinedLineageMetricsLineageCollapseWarning)
        ) {
            authority.status_codes[
                abi::kStatusLineageMetricsLineageCollapseWarning
            ] = lineage_summary.collapse_warning_code;
        }

        authority.float_values[abi::kFloatGenerationMetricsPeakInjury] =
            generation_summary[kSummaryPeakInjuryV1];
        authority.float_values[
            abi::kFloatLineageMetricsDominantLineagePopulationShare
        ] = lineage_summary.dominant_population_share;
        authority.float_values[
            abi::kFloatLineageMetricsTop3LineagePopulationShare
        ] = lineage_summary.top3_population_share;
        authority.float_values[
            abi::kFloatLineageMetricsLineagePopulationEntropy
        ] = lineage_summary.population_entropy;
        authority.float_values[
            abi::kFloatLineageMetricsLineageReproductionEntropy
        ] = lineage_summary.reproduction_entropy;
        authority.float_values[
            abi::kFloatLineageMetricsDominantLineageReproductionShare
        ] = lineage_summary.dominant_reproduction_share;

        std::memcpy(
            authority.text_values,
            lineage_summary.dominant_lineage_text_record,
            canonical::kDominantLineageTextRecordBytesV1
        );
        std::memcpy(
            authority.digest_values + kBehaviorDigestOffsetV1,
            canonical_result.behavior_rows_digest_sha256,
            canonical::kDigestBytesV1
        );
        std::memcpy(
            authority.digest_values + kLineageDigestOffsetV1,
            canonical_result.lineage_records_digest_sha256,
            canonical::kDigestBytesV1
        );
        if (!render_outer_authority_digests(
                *context,
                static_cast<std::uint32_t>(generation),
                static_cast<std::uint32_t>(cell),
                authority
            )) {
            context->status = kHostSettlementTemplateRenderInvalidV1;
            context->first_failed_row = static_cast<std::int32_t>(row_index);
            return;
        }
        authority.validity = 1;
        *lineage_state = canonical_result.lineage_state_after;
        ++context->completed_rows;
    }
    if (completed_generation_count == context->generation_count) {
        context->status = kHostSettlementSuccessV1;
    } else {
        context->first_failed_row = static_cast<std::int32_t>(row_count);
        context->status = kHostSettlementPartialPrefixV1;
    }
}

}  // namespace eiel::cuda_architecture_b::compact_summary_host_settlement_v1
