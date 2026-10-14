#include "cpu_compact_summary_source_composer_v1.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace eiel::native::cpu_compact_summary_source_composer_v1 {
namespace {

static_assert(
    canonical::kCompactAbiV2DefinedLineageActiveV1
    == abi::defined_bit_mask(abi::kDefinedLineageMetricsActiveLineages)
);
static_assert(
    canonical::kCompactAbiV2DefinedLineageDominantTextV1
    == abi::defined_bit_mask(abi::kDefinedLineageMetricsDominantLineageId)
);
static_assert(
    canonical::kCompactAbiV2DefinedLineageDominantPopulationShareV1
    == abi::defined_bit_mask(
        abi::kDefinedLineageMetricsDominantLineagePopulationShare
    )
);
static_assert(
    canonical::kCompactAbiV2DefinedLineageTop3ShareV1
    == abi::defined_bit_mask(
        abi::kDefinedLineageMetricsTop3LineagePopulationShare
    )
);
static_assert(
    canonical::kCompactAbiV2DefinedLineagePopulationEntropyV1
    == abi::defined_bit_mask(
        abi::kDefinedLineageMetricsLineagePopulationEntropy
    )
);
static_assert(
    canonical::kCompactAbiV2DefinedLineageReproductionEntropyV1
    == abi::defined_bit_mask(
        abi::kDefinedLineageMetricsLineageReproductionEntropy
    )
);
static_assert(
    canonical::kCompactAbiV2DefinedLineageDominantReproductionShareV1
    == abi::defined_bit_mask(
        abi::kDefinedLineageMetricsDominantLineageReproductionShare
    )
);
static_assert(
    canonical::kCompactAbiV2DefinedLineageRarePersistentV1
    == abi::defined_bit_mask(
        abi::kDefinedLineageMetricsRarePersistentLineages
    )
);
static_assert(
    canonical::kCompactAbiV2DefinedLineageCollapseWarningV1
    == abi::defined_bit_mask(
        abi::kDefinedLineageMetricsLineageCollapseWarning
    )
);
static_assert(
    canonical::kCompactAbiV2DefinedLineageCollapseStreakV1
    == abi::defined_bit_mask(
        abi::kDefinedLineageMetricsLineageCollapseStreak
    )
);

template <std::size_t Size>
bool bytes_are_zero(const std::array<std::uint8_t, Size>& bytes) noexcept {
    for (const std::uint8_t byte : bytes) {
        if (byte != 0U) {
            return false;
        }
    }
    return true;
}

bool bytes_are_zero(const std::uint8_t* const bytes, const std::size_t size)
    noexcept {
    for (std::size_t index = 0; index < size; ++index) {
        if (bytes[index] != 0U) {
            return false;
        }
    }
    return true;
}

bool raw_double_is_zero(const double value) noexcept {
    std::uint64_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits == 0U;
}

bool valid_defined_flag(const std::uint8_t value) noexcept {
    return value <= 1U;
}

bool add_nonnegative_exact(
    const std::int64_t left,
    const std::int64_t right,
    std::int64_t& output
) noexcept {
    if (left < 0 || right < 0
        || left > std::numeric_limits<std::int64_t>::max() - right) {
        return false;
    }
    output = left + right;
    return true;
}

bool sum_four_nonnegative_exact(
    const std::int64_t first,
    const std::int64_t second,
    const std::int64_t third,
    const std::int64_t fourth,
    std::int64_t& output
) noexcept {
    std::int64_t left = 0;
    std::int64_t right = 0;
    return add_nonnegative_exact(first, second, left)
        && add_nonnegative_exact(third, fourth, right)
        && add_nonnegative_exact(left, right, output);
}

void copy_digest(
    const digests::DigestV1& input,
    encoder::RawDigestV1& output
) noexcept {
    output.bytes = input.bytes;
}

void set_defined(
    abi::DefinedMaskV1& mask,
    const abi::DefinedBitV1 bit
) noexcept {
    mask = abi::set_defined_bit(mask, bit);
}

bool optional_int_valid(const OptionalInt64V1& value) noexcept {
    return valid_defined_flag(value.defined)
        && (value.defined == 1U || value.value == 0);
}

bool optional_status_valid(const OptionalStatusCodeV1& value) noexcept {
    return valid_defined_flag(value.defined)
        && (value.defined == 1U
            || value.value == abi::kReservedInvalidStatusCodeV2);
}

bool optional_double_valid(const OptionalDoubleV1& value) noexcept {
    return valid_defined_flag(value.defined)
        && (value.defined == 1U || raw_double_is_zero(value.value));
}

bool optional_digest_valid(const OptionalDigestV1& value) noexcept {
    return valid_defined_flag(value.defined)
        && (value.defined == 1U || bytes_are_zero(value.value.bytes));
}

ComposeStatusV1 validate_generation(
    const CpuCompactSummarySourceComposerInputV1& input
) noexcept {
    const GenerationBoundaryScalarsV1& generation = input.generation;
    const GenerationMetricsV1& metrics = input.generation_metrics;
    if (generation.summary_status_code != abi::kSummaryStatusComplete
        || (generation.world_code != abi::kWorldFlat
            && generation.world_code != abi::kWorldPatch)
        || input.turnover.status_code != abi::kTurnoverStatusCommitted
        || input.turnover.count != 1
        || generation.generation_start_generation < 0
        || generation.generation_start_generation
            != generation.generation_end_generation
        || generation.generation_start_tick != 0
        || generation.generation_end_generation
            == std::numeric_limits<std::int64_t>::max()
        || generation.next_generation_start_generation
            != generation.generation_end_generation + 1
        || generation.next_generation_start_tick != 0
        || generation.generation_end_tick_exclusive
            != metrics.policy_loop_tick_calls
        || metrics.capacity <= 0
        || metrics.free_slots < 0
        || metrics.free_slots > metrics.capacity
        || generation.alive != metrics.capacity - metrics.free_slots) {
        return ComposeStatusV1::kGenerationAuthorityMismatch;
    }

    std::int64_t cache_rows = 0;
    if (!add_nonnegative_exact(
            metrics.policy_cache_rows_hit,
            metrics.policy_cache_rows_missed,
            cache_rows
        )
        || cache_rows != metrics.policy_cache_rows_requested
        || metrics.policy_cache_rows_missed
            != metrics.policy_cache_rows_refilled) {
        return ComposeStatusV1::kGenerationAuthorityMismatch;
    }
    return ComposeStatusV1::kSuccess;
}

ComposeStatusV1 validate_lifecycle(
    const CpuCompactSummarySourceComposerInputV1& input
) noexcept {
    const LifecycleAuthorityV1& authority = input.lifecycle_authority;
    const lifecycle::LifecycleReductionV1& reduction = authority.reduction;
    if (authority.event_cursor_start < 0
        || authority.event_cursor_pre_turnover < authority.event_cursor_start
        || authority.event_cursor_end < authority.event_cursor_pre_turnover
        || reduction.event_row_count
            != authority.event_cursor_end - authority.event_cursor_start
        || reduction.generation_event_count
            != authority.event_cursor_pre_turnover
                - authority.event_cursor_start
        || reduction.turnover_event_count
            != authority.event_cursor_end
                - authority.event_cursor_pre_turnover) {
        return ComposeStatusV1::kLifecycleAuthorityMismatch;
    }

    std::int64_t count = 0;
    if (!sum_four_nonnegative_exact(
            reduction.generation_counts_by_type[0],
            reduction.generation_counts_by_type[1],
            reduction.generation_counts_by_type[2],
            reduction.generation_counts_by_type[3],
            count
        )
        || count != reduction.generation_event_count
        || input.generation.births != reduction.generation_counts_by_type[0]
        || input.generation.failed_births
            != reduction.generation_counts_by_type[2]
        || input.generation.deaths != reduction.generation_counts_by_type[3]
        || !sum_four_nonnegative_exact(
            reduction.turnover_counts_by_type[0],
            reduction.turnover_counts_by_type[1],
            reduction.turnover_counts_by_type[2],
            reduction.turnover_counts_by_type[3],
            count
        )
        || count != reduction.turnover_event_count
        || !add_nonnegative_exact(
            reduction.deaths_energy_depleted,
            reduction.deaths_health_depleted,
            count
        )
        || count != input.generation.deaths) {
        return ComposeStatusV1::kLifecycleAuthorityMismatch;
    }
    return ComposeStatusV1::kSuccess;
}

ComposeStatusV1 validate_identity(
    const CpuCompactSummarySourceComposerInputV1& input
) noexcept {
    const identity::IdentityReductionResultV1& value =
        input.identity_authority;
    if (!valid_defined_flag(value.active_agent_id_min_defined)
        || !valid_defined_flag(value.active_agent_id_max_defined)) {
        return ComposeStatusV1::kDefinedFlagInvalid;
    }
    if (!bytes_are_zero(value.reserved.data(), value.reserved.size())) {
        return ComposeStatusV1::kIdentityAuthorityMismatch;
    }

    const bool minimum_defined = value.active_agent_id_min_defined == 1U;
    const bool maximum_defined = value.active_agent_id_max_defined == 1U;
    const bool nonempty = value.active_agent_count != 0;
    std::int64_t slots = 0;
    if (value.active_agent_count < 0
        || value.free_slot_count < 0
        || !add_nonnegative_exact(
            value.active_agent_count, value.free_slot_count, slots
        )
        || slots != input.generation_metrics.capacity
        || minimum_defined != maximum_defined
        || minimum_defined != nonempty
        || (!nonempty
            && (value.active_agent_id_min != 0
                || value.active_agent_id_max != 0))
        || (nonempty
            && (value.active_agent_id_min < 0
                || value.active_agent_id_max < value.active_agent_id_min
                || (value.active_agent_count == 1
                    && value.active_agent_id_min
                        != value.active_agent_id_max)
                || (value.active_agent_count > 1
                    && value.active_agent_id_min
                        >= value.active_agent_id_max)
                || value.population_next_id
                    <= value.active_agent_id_max))) {
        return ComposeStatusV1::kIdentityAuthorityMismatch;
    }
    return ComposeStatusV1::kSuccess;
}

bool patch_optional_flags_and_storage_valid(const PatchMetricsV1& patch)
    noexcept {
    return optional_int_valid(patch.patch_spawn_sync_rows)
        && optional_int_valid(patch.typed_death_event_rows)
        && optional_int_valid(patch.env_rng_padding_draws)
        && optional_double_valid(patch.energy_collected_delta)
        && optional_double_valid(patch.patch_scent_move_queued_total)
        && optional_double_valid(patch.patch_waste_move_queued_total)
        && optional_double_valid(patch.patch_scent_gather_queued_total)
        && optional_double_valid(patch.patch_waste_gather_queued_total)
        && optional_double_valid(patch.patch_movement_energy_extra_total)
        && optional_double_valid(patch.patch_scent_emitted_total)
        && optional_double_valid(patch.patch_waste_emitted_total)
        && optional_double_valid(patch.patch_food_depleted_frac)
        && optional_double_valid(patch.patch_food_full_frac)
        && optional_double_valid(patch.patch_density_hotspot_frac);
}

bool flat_patch_storage_is_zero(const PatchMetricsV1& patch) noexcept {
    return patch.patch_movement_energy_event_count == 0
        && patch.patch_spawn_sync_rows.defined == 0U
        && patch.typed_death_event_rows.defined == 0U
        && patch.env_rng_padding_draws.defined == 0U
        && raw_double_is_zero(patch.patch_food_harvest_total)
        && raw_double_is_zero(patch.patch_scent_emit_total)
        && raw_double_is_zero(patch.patch_waste_emit_total)
        && raw_double_is_zero(patch.patch_movement_energy_total)
        && patch.energy_collected_delta.defined == 0U
        && patch.patch_scent_move_queued_total.defined == 0U
        && patch.patch_waste_move_queued_total.defined == 0U
        && patch.patch_scent_gather_queued_total.defined == 0U
        && patch.patch_waste_gather_queued_total.defined == 0U
        && patch.patch_movement_energy_extra_total.defined == 0U
        && patch.patch_scent_emitted_total.defined == 0U
        && patch.patch_waste_emitted_total.defined == 0U
        && patch.patch_food_depleted_frac.defined == 0U
        && patch.patch_food_full_frac.defined == 0U
        && patch.patch_density_hotspot_frac.defined == 0U;
}

ComposeStatusV1 validate_patch(
    const CpuCompactSummarySourceComposerInputV1& input
) noexcept {
    const PatchMetricsV1& patch = input.patch_metrics;
    if (!valid_defined_flag(patch.present)
        || !patch_optional_flags_and_storage_valid(patch)) {
        return ComposeStatusV1::kDefinedFlagInvalid;
    }
    const bool patch_world = input.generation.world_code == abi::kWorldPatch;
    if ((patch.present == 1U) != patch_world
        || (patch.present == 0U && !flat_patch_storage_is_zero(patch))) {
        return ComposeStatusV1::kPatchAuthorityMismatch;
    }
    return ComposeStatusV1::kSuccess;
}

ComposeStatusV1 validate_host_canonical(
    const canonical::HostCanonicalResultV1& value
) noexcept {
    constexpr abi::DefinedMaskV1 kLineageMask =
        abi::defined_bit_mask(abi::kDefinedLineageMetricsActiveLineages)
        | abi::defined_bit_mask(abi::kDefinedLineageMetricsDominantLineageId)
        | abi::defined_bit_mask(
            abi::kDefinedLineageMetricsDominantLineagePopulationShare
        )
        | abi::defined_bit_mask(
            abi::kDefinedLineageMetricsTop3LineagePopulationShare
        )
        | abi::defined_bit_mask(
            abi::kDefinedLineageMetricsLineagePopulationEntropy
        )
        | abi::defined_bit_mask(
            abi::kDefinedLineageMetricsLineageReproductionEntropy
        )
        | abi::defined_bit_mask(
            abi::kDefinedLineageMetricsDominantLineageReproductionShare
        )
        | abi::defined_bit_mask(
            abi::kDefinedLineageMetricsRarePersistentLineages
        )
        | abi::defined_bit_mask(
            abi::kDefinedLineageMetricsLineageCollapseWarning
        )
        | abi::defined_bit_mask(
            abi::kDefinedLineageMetricsLineageCollapseStreak
        );
    constexpr abi::DefinedMaskV1 kRequiredLineageMask =
        abi::defined_bit_mask(abi::kDefinedLineageMetricsActiveLineages)
        | abi::defined_bit_mask(
            abi::kDefinedLineageMetricsDominantLineagePopulationShare
        )
        | abi::defined_bit_mask(
            abi::kDefinedLineageMetricsTop3LineagePopulationShare
        )
        | abi::defined_bit_mask(
            abi::kDefinedLineageMetricsLineagePopulationEntropy
        )
        | abi::defined_bit_mask(
            abi::kDefinedLineageMetricsRarePersistentLineages
        )
        | abi::defined_bit_mask(
            abi::kDefinedLineageMetricsLineageCollapseWarning
        )
        | abi::defined_bit_mask(
            abi::kDefinedLineageMetricsLineageCollapseStreak
        );
    constexpr abi::DefinedMaskV1 kReproductionMask =
        abi::defined_bit_mask(
            abi::kDefinedLineageMetricsLineageReproductionEntropy
        )
        | abi::defined_bit_mask(
            abi::kDefinedLineageMetricsDominantLineageReproductionShare
        );

    const canonical::BehaviorAggregateOutputV1& behavior = value.behavior;
    const canonical::LineageSummaryOutputV1& lineage = value.lineage;
    const abi::DefinedMaskV1 mask = lineage.compact_abi_v2_defined_mask;
    std::int64_t total = 0;
    if (value.status
            != static_cast<std::uint32_t>(
                canonical::HostCanonicalStatusV1::kSuccess
            )
        || value.failed_field
            != static_cast<std::uint32_t>(canonical::HostCanonicalFieldV1::kNone)
        || value.failed_row != canonical::kNoFailedRowV1
        || value.reserved != 0U
        || lineage.reserved != 0U
        || !bytes_are_zero(lineage.reserved_tail, sizeof(lineage.reserved_tail))
        || (mask & ~kLineageMask) != 0U
        || behavior.row_count < 0
        || behavior.cohort_size < 0
        || behavior.cohort_survivors < 0
        || behavior.cohort_survivors > behavior.cohort_size
        || behavior.cohort_size > behavior.row_count
        || !sum_four_nonnegative_exact(
            behavior.move_n,
            behavior.move_s,
            behavior.move_e,
            behavior.move_w,
            total
        )
        || total != behavior.move_total
        || !sum_four_nonnegative_exact(
            behavior.move_total,
            behavior.gather,
            behavior.rest,
            behavior.attack,
            total
        )
        || total != behavior.action_total
        || lineage.record_count < 0) {
        return ComposeStatusV1::kHostCanonicalAuthorityMismatch;
    }

    if (mask == 0U) {
        if (lineage.active_lineages != 0
            || lineage.rare_persistent_lineages != 0
            || lineage.collapse_streak != 0
            || !raw_double_is_zero(lineage.dominant_population_share)
            || !raw_double_is_zero(lineage.top3_population_share)
            || !raw_double_is_zero(lineage.population_entropy)
            || !raw_double_is_zero(lineage.reproduction_entropy)
            || !raw_double_is_zero(lineage.dominant_reproduction_share)
            || lineage.collapse_warning_code
                != abi::kReservedInvalidStatusCodeV2
            || !bytes_are_zero(
                lineage.dominant_lineage_text_record,
                sizeof(lineage.dominant_lineage_text_record)
            )) {
            return ComposeStatusV1::kHostCanonicalAuthorityMismatch;
        }
        return ComposeStatusV1::kSuccess;
    }

    const bool text_defined = (mask & abi::defined_bit_mask(
        abi::kDefinedLineageMetricsDominantLineageId
    )) != 0U;
    const abi::DefinedMaskV1 reproduction = mask & kReproductionMask;
    if ((mask & kRequiredLineageMask) != kRequiredLineageMask
        || (reproduction != 0U && reproduction != kReproductionMask)
        || lineage.active_lineages < 0
        || lineage.active_lineages > lineage.record_count
        || text_defined != (lineage.active_lineages > 0)) {
        return ComposeStatusV1::kHostCanonicalAuthorityMismatch;
    }
    return ComposeStatusV1::kSuccess;
}

ComposeStatusV1 validate_terminal(
    const TerminalCheckpointAuthorityV1& terminal
) noexcept {
    if (!optional_status_valid(terminal.recoverable_error_code)
        || !optional_digest_valid(terminal.recoverable_error_digest)
        || !optional_int_valid(terminal.failure_generation)
        || !optional_int_valid(terminal.failure_tick)
        || !optional_status_valid(terminal.failure_stage)) {
        return ComposeStatusV1::kDefinedFlagInvalid;
    }

    const bool recoverable_present = terminal.recoverable_error_status_code
        == abi::kRecoverableErrorStatusPresent;
    const bool recoverable_defined =
        terminal.recoverable_error_code.defined == 1U
        && terminal.recoverable_error_digest.defined == 1U;
    const bool failed = terminal.terminal_status_code
        == abi::kTerminalStatusFailed;
    const bool sticky = terminal.sticky_failure_code
        == abi::kBooleanStatusTrue;
    const bool failure_defined = terminal.failure_generation.defined == 1U
        && terminal.failure_tick.defined == 1U
        && terminal.failure_stage.defined == 1U;
    if (recoverable_present != recoverable_defined
        || terminal.recoverable_error_code.defined
            != terminal.recoverable_error_digest.defined
        || failed != sticky
        || failed != failure_defined) {
        return ComposeStatusV1::kTerminalAuthorityMismatch;
    }
    return ComposeStatusV1::kSuccess;
}

void map_generation(
    const CpuCompactSummarySourceComposerInputV1& input,
    encoder::CpuCompactSummarySourceAuthorityV1& output
) noexcept {
    const GenerationBoundaryScalarsV1& generation = input.generation;
    output.status_codes[abi::kStatusSummaryStatus] =
        generation.summary_status_code;
    output.status_codes[abi::kStatusWorld] = generation.world_code;
    output.status_codes[abi::kStatusTurnoverStatus] =
        input.turnover.status_code;

    output.integer_values[abi::kIntegerGenerationStartGeneration] =
        generation.generation_start_generation;
    output.integer_values[abi::kIntegerGenerationStartTick] =
        generation.generation_start_tick;
    output.integer_values[abi::kIntegerGenerationEndGeneration] =
        generation.generation_end_generation;
    output.integer_values[abi::kIntegerGenerationEndTickExclusive] =
        generation.generation_end_tick_exclusive;
    output.integer_values[abi::kIntegerNextGenerationStartGeneration] =
        generation.next_generation_start_generation;
    output.integer_values[abi::kIntegerNextGenerationStartTick] =
        generation.next_generation_start_tick;
    output.integer_values[abi::kIntegerAlive] = generation.alive;
    output.integer_values[abi::kIntegerBirths] = generation.births;
    output.integer_values[abi::kIntegerFailedBirths] =
        generation.failed_births;
    output.integer_values[abi::kIntegerDeaths] = generation.deaths;
    output.integer_values[abi::kIntegerTurnoverCount] = input.turnover.count;
}

void map_lifecycle(
    const LifecycleAuthorityV1& authority,
    encoder::CpuCompactSummarySourceAuthorityV1& output
) noexcept {
    const lifecycle::LifecycleReductionV1& value = authority.reduction;
    output.integer_values[abi::kIntegerDeathsByClassEnergyDepleted] =
        value.deaths_energy_depleted;
    output.integer_values[abi::kIntegerDeathsByClassHealthDepleted] =
        value.deaths_health_depleted;
    output.integer_values[abi::kIntegerEventsCursorStart] =
        authority.event_cursor_start;
    output.integer_values[abi::kIntegerEventsCursorPreTurnover] =
        authority.event_cursor_pre_turnover;
    output.integer_values[abi::kIntegerEventsCursorEnd] =
        authority.event_cursor_end;
    output.integer_values[abi::kIntegerEventsAuthoritativeCursor] =
        authority.event_cursor_end;
    output.integer_values[abi::kIntegerEventsEventRowCount] =
        value.event_row_count;
    output.integer_values[abi::kIntegerEventsGenerationEventCount] =
        value.generation_event_count;
    output.integer_values[abi::kIntegerEventsTurnoverEventCount] =
        value.turnover_event_count;
    output.integer_values[
        abi::kIntegerEventsGenerationCountsByTypeBirthCommitted
    ] = value.generation_counts_by_type[0];
    output.integer_values[
        abi::kIntegerEventsGenerationCountsByTypeMutation
    ] = value.generation_counts_by_type[1];
    output.integer_values[
        abi::kIntegerEventsGenerationCountsByTypeBirthFailed
    ] = value.generation_counts_by_type[2];
    output.integer_values[
        abi::kIntegerEventsGenerationCountsByTypeDeathCommitted
    ] = value.generation_counts_by_type[3];
    output.integer_values[
        abi::kIntegerEventsTurnoverCountsByTypeBirthCommitted
    ] = value.turnover_counts_by_type[0];
    output.integer_values[
        abi::kIntegerEventsTurnoverCountsByTypeMutation
    ] = value.turnover_counts_by_type[1];
    output.integer_values[
        abi::kIntegerEventsTurnoverCountsByTypeBirthFailed
    ] = value.turnover_counts_by_type[2];
    output.integer_values[
        abi::kIntegerEventsTurnoverCountsByTypeDeathCommitted
    ] = value.turnover_counts_by_type[3];
    copy_digest(
        value.active_prefix_digest,
        output.digest_values[abi::kDigestEventsActivePrefixDigestSha256]
    );
}

void map_identity(
    const identity::IdentityReductionResultV1& value,
    encoder::CpuCompactSummarySourceAuthorityV1& output
) noexcept {
    output.integer_values[abi::kIntegerIdentityPopulationNextId] =
        value.population_next_id;
    output.integer_values[abi::kIntegerIdentityActiveAgentCount] =
        value.active_agent_count;
    output.integer_values[abi::kIntegerIdentityActiveAgentIdMin] =
        value.active_agent_id_min;
    output.integer_values[abi::kIntegerIdentityActiveAgentIdMax] =
        value.active_agent_id_max;
    output.integer_values[abi::kIntegerIdentityFreeSlotCount] =
        value.free_slot_count;
    if (value.active_agent_id_min_defined == 1U) {
        set_defined(output.defined_mask, abi::kDefinedIdentityActiveAgentIdMin);
    }
    if (value.active_agent_id_max_defined == 1U) {
        set_defined(output.defined_mask, abi::kDefinedIdentityActiveAgentIdMax);
    }
    copy_digest(
        value.active_agent_ids_digest,
        output.digest_values[abi::kDigestIdentityActiveAgentIdsDigestSha256]
    );
}

void map_rng_state(
    const RngStateDigestAuthorityV1& value,
    encoder::CpuCompactSummarySourceAuthorityV1& output
) noexcept {
    output.status_codes[abi::kStatusRngAuthorityComplete] =
        value.authority_complete_code;
    output.status_codes[abi::kStatusRngExportNonConsuming] =
        value.export_non_consuming_code;
    output.integer_values[abi::kIntegerRngPolicyMutationTapeCursor] =
        value.policy_mutation_tape_cursor;
    copy_digest(
        value.authority_digest,
        output.digest_values[abi::kDigestRngAuthorityDigestSha256]
    );
    copy_digest(
        value.policy_mutation_tape_digest,
        output.digest_values[abi::kDigestRngPolicyMutationTapeDigestSha256]
    );
    copy_digest(
        value.authoritative_state_digest,
        output.digest_values[
            abi::kDigestStateDigestsAuthoritativeStateDigestSha256
        ]
    );
    copy_digest(
        value.canonical_checkpoint_storage_digest,
        output.digest_values[
            abi::kDigestStateDigestsCanonicalCheckpointStorageDigestSha256
        ]
    );
    copy_digest(
        value.next_generation_start_state_digest,
        output.digest_values[
            abi::kDigestStateDigestsNextGenerationStartStateDigestSha256
        ]
    );
}

void map_generation_metrics(
    const GenerationMetricsV1& value,
    encoder::CpuCompactSummarySourceAuthorityV1& output
) noexcept {
    output.integer_values[abi::kIntegerGenerationMetricsCapacity] =
        value.capacity;
    output.integer_values[abi::kIntegerGenerationMetricsFreeSlots] =
        value.free_slots;
    output.integer_values[abi::kIntegerGenerationMetricsBirthAttempts] =
        value.birth_attempts;
    output.integer_values[
        abi::kIntegerGenerationMetricsNativeFailedBirthEventRows
    ] = value.native_failed_birth_event_rows;
    output.integer_values[
        abi::kIntegerGenerationMetricsInjuryBlockedBirths
    ] = value.injury_blocked_births;
    output.integer_values[abi::kIntegerGenerationMetricsMoved] = value.moved;
    output.integer_values[abi::kIntegerGenerationMetricsGathered] =
        value.gathered;
    output.integer_values[abi::kIntegerGenerationMetricsMovementAttempts] =
        value.movement_attempts;
    output.integer_values[
        abi::kIntegerGenerationMetricsMovementBlockedOccupied
    ] = value.movement_blocked_occupied;
    output.integer_values[
        abi::kIntegerGenerationMetricsMovementBlockedBounds
    ] = value.movement_blocked_bounds;
    output.integer_values[
        abi::kIntegerGenerationMetricsPolicyLoopTickCalls
    ] = value.policy_loop_tick_calls;
    output.integer_values[
        abi::kIntegerGenerationMetricsPolicyRefillCalls
    ] = value.policy_refill_calls;
    output.integer_values[
        abi::kIntegerGenerationMetricsPolicyRefillRows
    ] = value.policy_refill_rows;
    output.integer_values[
        abi::kIntegerGenerationMetricsPolicyCacheHitCalls
    ] = value.policy_cache_hit_calls;
    output.integer_values[
        abi::kIntegerGenerationMetricsPolicyCacheRowsRequested
    ] = value.policy_cache_rows_requested;
    output.integer_values[
        abi::kIntegerGenerationMetricsPolicyCacheRowsHit
    ] = value.policy_cache_rows_hit;
    output.integer_values[
        abi::kIntegerGenerationMetricsPolicyCacheRowsMissed
    ] = value.policy_cache_rows_missed;
    output.integer_values[
        abi::kIntegerGenerationMetricsPolicyCacheRowsRefilled
    ] = value.policy_cache_rows_refilled;
    output.integer_values[
        abi::kIntegerGenerationMetricsPolicyCacheFullRefillTicks
    ] = value.policy_cache_full_refill_ticks;
    output.integer_values[
        abi::kIntegerGenerationMetricsPolicyCachePartialRefillTicks
    ] = value.policy_cache_partial_refill_ticks;
    output.float_values[abi::kFloatGenerationMetricsPeakInjury] =
        value.peak_injury;
}

void map_optional_integer(
    const OptionalInt64V1& value,
    const abi::DefinedBitV1 bit,
    const abi::IntegerFieldV2 field,
    encoder::CpuCompactSummarySourceAuthorityV1& output
) noexcept {
    output.integer_values[field] = value.value;
    if (value.defined == 1U) {
        set_defined(output.defined_mask, bit);
    }
}

void map_optional_double(
    const OptionalDoubleV1& value,
    const abi::DefinedBitV1 bit,
    const abi::FloatFieldV2 field,
    encoder::CpuCompactSummarySourceAuthorityV1& output
) noexcept {
    output.float_values[field] = value.value;
    if (value.defined == 1U) {
        set_defined(output.defined_mask, bit);
    }
}

void map_patch(
    const PatchMetricsV1& value,
    encoder::CpuCompactSummarySourceAuthorityV1& output
) noexcept {
    if (value.present == 0U) {
        return;
    }
    set_defined(output.defined_mask, abi::kDefinedPatchMetrics);
    output.integer_values[
        abi::kIntegerPatchMetricsPatchMovementEnergyEventCount
    ] = value.patch_movement_energy_event_count;
    map_optional_integer(
        value.patch_spawn_sync_rows,
        abi::kDefinedPatchMetricsPatchSpawnSyncRows,
        abi::kIntegerPatchMetricsPatchSpawnSyncRows,
        output
    );
    map_optional_integer(
        value.typed_death_event_rows,
        abi::kDefinedPatchMetricsTypedDeathEventRows,
        abi::kIntegerPatchMetricsTypedDeathEventRows,
        output
    );
    map_optional_integer(
        value.env_rng_padding_draws,
        abi::kDefinedPatchMetricsEnvRngPaddingDraws,
        abi::kIntegerPatchMetricsEnvRngPaddingDraws,
        output
    );
    output.float_values[abi::kFloatPatchMetricsPatchFoodHarvestTotal] =
        value.patch_food_harvest_total;
    output.float_values[abi::kFloatPatchMetricsPatchScentEmitTotal] =
        value.patch_scent_emit_total;
    output.float_values[abi::kFloatPatchMetricsPatchWasteEmitTotal] =
        value.patch_waste_emit_total;
    output.float_values[abi::kFloatPatchMetricsPatchMovementEnergyTotal] =
        value.patch_movement_energy_total;
    map_optional_double(
        value.energy_collected_delta,
        abi::kDefinedPatchMetricsEnergyCollectedDelta,
        abi::kFloatPatchMetricsEnergyCollectedDelta,
        output
    );
    map_optional_double(
        value.patch_scent_move_queued_total,
        abi::kDefinedPatchMetricsPatchScentMoveQueuedTotal,
        abi::kFloatPatchMetricsPatchScentMoveQueuedTotal,
        output
    );
    map_optional_double(
        value.patch_waste_move_queued_total,
        abi::kDefinedPatchMetricsPatchWasteMoveQueuedTotal,
        abi::kFloatPatchMetricsPatchWasteMoveQueuedTotal,
        output
    );
    map_optional_double(
        value.patch_scent_gather_queued_total,
        abi::kDefinedPatchMetricsPatchScentGatherQueuedTotal,
        abi::kFloatPatchMetricsPatchScentGatherQueuedTotal,
        output
    );
    map_optional_double(
        value.patch_waste_gather_queued_total,
        abi::kDefinedPatchMetricsPatchWasteGatherQueuedTotal,
        abi::kFloatPatchMetricsPatchWasteGatherQueuedTotal,
        output
    );
    map_optional_double(
        value.patch_movement_energy_extra_total,
        abi::kDefinedPatchMetricsPatchMovementEnergyExtraTotal,
        abi::kFloatPatchMetricsPatchMovementEnergyExtraTotal,
        output
    );
    map_optional_double(
        value.patch_scent_emitted_total,
        abi::kDefinedPatchMetricsPatchScentEmittedTotal,
        abi::kFloatPatchMetricsPatchScentEmittedTotal,
        output
    );
    map_optional_double(
        value.patch_waste_emitted_total,
        abi::kDefinedPatchMetricsPatchWasteEmittedTotal,
        abi::kFloatPatchMetricsPatchWasteEmittedTotal,
        output
    );
    map_optional_double(
        value.patch_food_depleted_frac,
        abi::kDefinedPatchMetricsPatchFoodDepletedFrac,
        abi::kFloatPatchMetricsPatchFoodDepletedFrac,
        output
    );
    map_optional_double(
        value.patch_food_full_frac,
        abi::kDefinedPatchMetricsPatchFoodFullFrac,
        abi::kFloatPatchMetricsPatchFoodFullFrac,
        output
    );
    map_optional_double(
        value.patch_density_hotspot_frac,
        abi::kDefinedPatchMetricsPatchDensityHotspotFrac,
        abi::kFloatPatchMetricsPatchDensityHotspotFrac,
        output
    );
}

void map_host_canonical(
    const canonical::HostCanonicalResultV1& value,
    encoder::CpuCompactSummarySourceAuthorityV1& output
) noexcept {
    const canonical::BehaviorAggregateOutputV1& behavior = value.behavior;
    const canonical::LineageSummaryOutputV1& lineage = value.lineage;
    output.defined_mask |= lineage.compact_abi_v2_defined_mask;
    output.integer_values[abi::kIntegerBehaviorMetricsRowCount] =
        behavior.row_count;
    output.integer_values[abi::kIntegerBehaviorMetricsCohortSize] =
        behavior.cohort_size;
    output.integer_values[abi::kIntegerBehaviorMetricsCohortSurvivors] =
        behavior.cohort_survivors;
    output.integer_values[abi::kIntegerBehaviorMetricsActionTotal] =
        behavior.action_total;
    output.integer_values[abi::kIntegerBehaviorMetricsMoveTotal] =
        behavior.move_total;
    output.integer_values[abi::kIntegerBehaviorMetricsMoveN] = behavior.move_n;
    output.integer_values[abi::kIntegerBehaviorMetricsMoveS] = behavior.move_s;
    output.integer_values[abi::kIntegerBehaviorMetricsMoveE] = behavior.move_e;
    output.integer_values[abi::kIntegerBehaviorMetricsMoveW] = behavior.move_w;
    output.integer_values[abi::kIntegerBehaviorMetricsGather] = behavior.gather;
    output.integer_values[abi::kIntegerBehaviorMetricsRest] = behavior.rest;
    output.integer_values[abi::kIntegerBehaviorMetricsAttack] = behavior.attack;

    output.integer_values[abi::kIntegerLineageMetricsRecordCount] =
        lineage.record_count;
    output.integer_values[abi::kIntegerLineageMetricsActiveLineages] =
        lineage.active_lineages;
    output.integer_values[abi::kIntegerLineageMetricsRarePersistentLineages] =
        lineage.rare_persistent_lineages;
    output.integer_values[abi::kIntegerLineageMetricsLineageCollapseStreak] =
        lineage.collapse_streak;
    output.float_values[
        abi::kFloatLineageMetricsDominantLineagePopulationShare
    ] = lineage.dominant_population_share;
    output.float_values[abi::kFloatLineageMetricsTop3LineagePopulationShare] =
        lineage.top3_population_share;
    output.float_values[abi::kFloatLineageMetricsLineagePopulationEntropy] =
        lineage.population_entropy;
    output.float_values[
        abi::kFloatLineageMetricsLineageReproductionEntropy
    ] = lineage.reproduction_entropy;
    output.float_values[
        abi::kFloatLineageMetricsDominantLineageReproductionShare
    ] = lineage.dominant_reproduction_share;
    output.status_codes[
        abi::kStatusLineageMetricsLineageCollapseWarning
    ] = lineage.collapse_warning_code;
    std::memcpy(
        &output.text_values[abi::kTextLineageMetricsDominantLineageId],
        lineage.dominant_lineage_text_record,
        sizeof(lineage.dominant_lineage_text_record)
    );
    std::memcpy(
        output.digest_values[abi::kDigestBehaviorMetricsRowsDigestSha256]
            .bytes.data(),
        value.behavior_rows_digest_sha256,
        canonical::kDigestBytesV1
    );
    std::memcpy(
        output.digest_values[abi::kDigestLineageMetricsRecordsDigestSha256]
            .bytes.data(),
        value.lineage_records_digest_sha256,
        canonical::kDigestBytesV1
    );
}

void map_terminal(
    const TerminalCheckpointAuthorityV1& value,
    encoder::CpuCompactSummarySourceAuthorityV1& output
) noexcept {
    output.status_codes[abi::kStatusTerminalStatus] =
        value.terminal_status_code;
    output.status_codes[abi::kStatusTerminalRecoverableErrorStatus] =
        value.recoverable_error_status_code;
    output.status_codes[abi::kStatusTerminalStickyFailure] =
        value.sticky_failure_code;
    output.status_codes[abi::kStatusCheckpointDue] =
        value.checkpoint_due_code;
    output.status_codes[abi::kStatusTerminalRecoverableErrorCode] =
        value.recoverable_error_code.value;
    output.status_codes[abi::kStatusTerminalFailureStage] =
        value.failure_stage.value;
    output.integer_values[abi::kIntegerTerminalFailureGeneration] =
        value.failure_generation.value;
    output.integer_values[abi::kIntegerTerminalFailureTick] =
        value.failure_tick.value;
    if (value.recoverable_error_code.defined == 1U) {
        set_defined(
            output.defined_mask, abi::kDefinedTerminalRecoverableErrorCode
        );
    }
    if (value.recoverable_error_digest.defined == 1U) {
        set_defined(
            output.defined_mask,
            abi::kDefinedTerminalRecoverableErrorDigestSha256
        );
        copy_digest(
            value.recoverable_error_digest.value,
            output.digest_values[
                abi::kDigestTerminalRecoverableErrorDigestSha256
            ]
        );
    }
    if (value.failure_generation.defined == 1U) {
        set_defined(output.defined_mask, abi::kDefinedTerminalFailureGeneration);
    }
    if (value.failure_tick.defined == 1U) {
        set_defined(output.defined_mask, abi::kDefinedTerminalFailureTick);
    }
    if (value.failure_stage.defined == 1U) {
        set_defined(output.defined_mask, abi::kDefinedTerminalFailureStage);
    }
}

}  // namespace

ComposeStatusV1 compose_source_authority(
    const CpuCompactSummarySourceComposerInputV1& input,
    encoder::CpuCompactSummarySourceAuthorityV1& output
) noexcept {
    output = {};
    if (input.input_version != kComposerInputVersionV1) {
        return ComposeStatusV1::kInputVersionMismatch;
    }

    ComposeStatusV1 status = validate_generation(input);
    if (status != ComposeStatusV1::kSuccess) {
        return status;
    }
    status = validate_lifecycle(input);
    if (status != ComposeStatusV1::kSuccess) {
        return status;
    }
    status = validate_identity(input);
    if (status != ComposeStatusV1::kSuccess) {
        return status;
    }
    if (input.rng_state_authority.authority_complete_code
            != abi::kBooleanStatusTrue
        || input.rng_state_authority.export_non_consuming_code
            != abi::kBooleanStatusTrue) {
        return ComposeStatusV1::kRngAuthorityMismatch;
    }
    status = validate_patch(input);
    if (status != ComposeStatusV1::kSuccess) {
        return status;
    }
    status = validate_host_canonical(input.host_canonical_authority);
    if (status != ComposeStatusV1::kSuccess) {
        return status;
    }
    status = validate_terminal(input.terminal_checkpoint_authority);
    if (status != ComposeStatusV1::kSuccess) {
        return status;
    }

    encoder::CpuCompactSummarySourceAuthorityV1 candidate{};
    candidate.authority_version = encoder::kSourceAuthorityVersionV1;
    map_generation(input, candidate);
    map_lifecycle(input.lifecycle_authority, candidate);
    map_identity(input.identity_authority, candidate);
    map_rng_state(input.rng_state_authority, candidate);
    map_generation_metrics(input.generation_metrics, candidate);
    map_patch(input.patch_metrics, candidate);
    map_host_canonical(input.host_canonical_authority, candidate);
    map_terminal(input.terminal_checkpoint_authority, candidate);

    if (encoder::validate_completed_source(candidate)
        != encoder::EncodeStatusV1::kSuccess) {
        return ComposeStatusV1::kEncoderRejected;
    }
    output = candidate;
    return ComposeStatusV1::kSuccess;
}

ComposeStatusV1 compose_and_encode_completed_row(
    const CpuCompactSummarySourceComposerInputV1& input,
    encoder::CpuCompactSummaryRowV1& row
) noexcept {
    encoder::initialize_invalid_row(row);
    encoder::CpuCompactSummarySourceAuthorityV1 source{};
    const ComposeStatusV1 compose_status = compose_source_authority(
        input, source
    );
    if (compose_status != ComposeStatusV1::kSuccess) {
        return compose_status;
    }
    const encoder::EncodeStatusV1 encode_status =
        encoder::encode_completed_row(source, row);
    return encode_status == encoder::EncodeStatusV1::kSuccess
        ? ComposeStatusV1::kSuccess
        : ComposeStatusV1::kEncoderRejected;
}

const char* compose_status_name(const ComposeStatusV1 status) noexcept {
    switch (status) {
        case ComposeStatusV1::kSuccess: return "success";
        case ComposeStatusV1::kInputVersionMismatch:
            return "input_version_mismatch";
        case ComposeStatusV1::kDefinedFlagInvalid:
            return "defined_flag_invalid";
        case ComposeStatusV1::kGenerationAuthorityMismatch:
            return "generation_authority_mismatch";
        case ComposeStatusV1::kLifecycleAuthorityMismatch:
            return "lifecycle_authority_mismatch";
        case ComposeStatusV1::kIdentityAuthorityMismatch:
            return "identity_authority_mismatch";
        case ComposeStatusV1::kRngAuthorityMismatch:
            return "rng_authority_mismatch";
        case ComposeStatusV1::kPatchAuthorityMismatch:
            return "patch_authority_mismatch";
        case ComposeStatusV1::kHostCanonicalAuthorityMismatch:
            return "host_canonical_authority_mismatch";
        case ComposeStatusV1::kTerminalAuthorityMismatch:
            return "terminal_authority_mismatch";
        case ComposeStatusV1::kEncoderRejected:
            return "encoder_rejected";
    }
    return "unknown";
}

}  // namespace eiel::native::cpu_compact_summary_source_composer_v1
