#pragma once

#include "cpu_compact_summary_identity_reduction_v1.hpp"
#include "cpu_compact_summary_lifecycle_reduction_v1.hpp"
#include "cpu_compact_summary_v1.hpp"
#include "../cuda_architecture_b_compact_summary_host_canonical_v1.hpp"

#include <cstdint>
#include <type_traits>

namespace eiel::native::cpu_compact_summary_source_composer_v1 {

namespace abi = eiel::cuda_architecture_b::compact_summary_v2;
namespace canonical =
    eiel::cuda_architecture_b::compact_summary_host_canonical_v1;
namespace encoder = eiel::native::cpu_compact_summary_v1;
namespace identity = eiel::native::cpu_compact_summary_identity_reduction_v1;
namespace lifecycle = eiel::native::cpu_compact_summary_lifecycle_reduction_v1;
namespace digests = eiel::native::cpu_compact_summary_digests_v1;

inline constexpr std::uint32_t kComposerInputVersionV1 = 1U;

struct OptionalInt64V1 {
    std::int64_t value;
    std::uint8_t defined;
};

struct OptionalStatusCodeV1 {
    std::int32_t value;
    std::uint8_t defined;
};

struct OptionalDoubleV1 {
    double value;
    std::uint8_t defined;
};

struct OptionalDigestV1 {
    digests::DigestV1 value;
    std::uint8_t defined;
};

struct GenerationBoundaryScalarsV1 {
    std::int32_t summary_status_code;
    std::int32_t world_code;
    std::int64_t generation_start_generation;
    std::int64_t generation_start_tick;
    std::int64_t generation_end_generation;
    std::int64_t generation_end_tick_exclusive;
    std::int64_t next_generation_start_generation;
    std::int64_t next_generation_start_tick;
    std::int64_t alive;
    std::int64_t births;
    std::int64_t failed_births;
    std::int64_t deaths;
};

struct TurnoverScalarsV1 {
    std::int64_t count;
    std::int32_t status_code;
};

struct LifecycleAuthorityV1 {
    std::int64_t event_cursor_start;
    std::int64_t event_cursor_pre_turnover;
    std::int64_t event_cursor_end;
    lifecycle::LifecycleReductionV1 reduction;
};

struct RngStateDigestAuthorityV1 {
    std::int32_t authority_complete_code;
    std::int32_t export_non_consuming_code;
    std::int64_t policy_mutation_tape_cursor;
    digests::DigestV1 authority_digest;
    digests::DigestV1 policy_mutation_tape_digest;
    digests::DigestV1 authoritative_state_digest;
    digests::DigestV1 canonical_checkpoint_storage_digest;
    digests::DigestV1 next_generation_start_state_digest;
};

struct GenerationMetricsV1 {
    std::int64_t capacity;
    std::int64_t free_slots;
    std::int64_t birth_attempts;
    std::int64_t native_failed_birth_event_rows;
    std::int64_t injury_blocked_births;
    std::int64_t moved;
    std::int64_t gathered;
    std::int64_t movement_attempts;
    std::int64_t movement_blocked_occupied;
    std::int64_t movement_blocked_bounds;
    std::int64_t policy_loop_tick_calls;
    std::int64_t policy_refill_calls;
    std::int64_t policy_refill_rows;
    std::int64_t policy_cache_hit_calls;
    std::int64_t policy_cache_rows_requested;
    std::int64_t policy_cache_rows_hit;
    std::int64_t policy_cache_rows_missed;
    std::int64_t policy_cache_rows_refilled;
    std::int64_t policy_cache_full_refill_ticks;
    std::int64_t policy_cache_partial_refill_ticks;
    double peak_injury;
};

struct PatchMetricsV1 {
    std::uint8_t present;
    std::int64_t patch_movement_energy_event_count;
    OptionalInt64V1 patch_spawn_sync_rows;
    OptionalInt64V1 typed_death_event_rows;
    OptionalInt64V1 env_rng_padding_draws;
    double patch_food_harvest_total;
    double patch_scent_emit_total;
    double patch_waste_emit_total;
    double patch_movement_energy_total;
    OptionalDoubleV1 energy_collected_delta;
    OptionalDoubleV1 patch_scent_move_queued_total;
    OptionalDoubleV1 patch_waste_move_queued_total;
    OptionalDoubleV1 patch_scent_gather_queued_total;
    OptionalDoubleV1 patch_waste_gather_queued_total;
    OptionalDoubleV1 patch_movement_energy_extra_total;
    OptionalDoubleV1 patch_scent_emitted_total;
    OptionalDoubleV1 patch_waste_emitted_total;
    OptionalDoubleV1 patch_food_depleted_frac;
    OptionalDoubleV1 patch_food_full_frac;
    OptionalDoubleV1 patch_density_hotspot_frac;
};

struct TerminalCheckpointAuthorityV1 {
    std::int32_t terminal_status_code;
    std::int32_t recoverable_error_status_code;
    std::int32_t sticky_failure_code;
    std::int32_t checkpoint_due_code;
    OptionalStatusCodeV1 recoverable_error_code;
    OptionalDigestV1 recoverable_error_digest;
    OptionalInt64V1 failure_generation;
    OptionalInt64V1 failure_tick;
    OptionalStatusCodeV1 failure_stage;
};

struct CpuCompactSummarySourceComposerInputV1 {
    std::uint32_t input_version;
    GenerationBoundaryScalarsV1 generation;
    TurnoverScalarsV1 turnover;
    LifecycleAuthorityV1 lifecycle_authority;
    identity::IdentityReductionResultV1 identity_authority;
    RngStateDigestAuthorityV1 rng_state_authority;
    GenerationMetricsV1 generation_metrics;
    PatchMetricsV1 patch_metrics;
    canonical::HostCanonicalResultV1 host_canonical_authority;
    TerminalCheckpointAuthorityV1 terminal_checkpoint_authority;
};

using ComposerInputV1 = CpuCompactSummarySourceComposerInputV1;

enum class ComposeStatusV1 : std::uint32_t {
    kSuccess = 0,
    kInputVersionMismatch = 1,
    kDefinedFlagInvalid = 2,
    kGenerationAuthorityMismatch = 3,
    kLifecycleAuthorityMismatch = 4,
    kIdentityAuthorityMismatch = 5,
    kRngAuthorityMismatch = 6,
    kPatchAuthorityMismatch = 7,
    kHostCanonicalAuthorityMismatch = 8,
    kTerminalAuthorityMismatch = 9,
    kEncoderRejected = 10,
};

// Output is bit-zero on every non-success return.
[[nodiscard]] ComposeStatusV1 compose_source_authority(
    const CpuCompactSummarySourceComposerInputV1& input,
    encoder::CpuCompactSummarySourceAuthorityV1& output
) noexcept;

// Clears row before inspecting input and delegates final acceptance and
// validity-last publication to the frozen native encoder.
[[nodiscard]] ComposeStatusV1 compose_and_encode_completed_row(
    const CpuCompactSummarySourceComposerInputV1& input,
    encoder::CpuCompactSummaryRowV1& row
) noexcept;

[[nodiscard]] const char* compose_status_name(ComposeStatusV1 status) noexcept;

static_assert(std::is_standard_layout_v<OptionalInt64V1>);
static_assert(std::is_trivially_copyable_v<OptionalInt64V1>);
static_assert(std::is_standard_layout_v<OptionalStatusCodeV1>);
static_assert(std::is_trivially_copyable_v<OptionalStatusCodeV1>);
static_assert(std::is_standard_layout_v<OptionalDoubleV1>);
static_assert(std::is_trivially_copyable_v<OptionalDoubleV1>);
static_assert(std::is_standard_layout_v<OptionalDigestV1>);
static_assert(std::is_trivially_copyable_v<OptionalDigestV1>);
static_assert(std::is_standard_layout_v<CpuCompactSummarySourceComposerInputV1>);
static_assert(
    std::is_trivially_copyable_v<CpuCompactSummarySourceComposerInputV1>
);

}  // namespace eiel::native::cpu_compact_summary_source_composer_v1
