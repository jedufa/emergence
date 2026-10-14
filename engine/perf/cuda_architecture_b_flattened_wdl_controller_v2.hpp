#ifndef EIEL_CUDA_ARCHITECTURE_B_FLATTENED_WDL_CONTROLLER_V2_HPP_
#define EIEL_CUDA_ARCHITECTURE_B_FLATTENED_WDL_CONTROLLER_V2_HPP_

#include <cstdint>

#include <torch/types.h>

namespace eiel::cuda_architecture_b::flattened_wdl_controller_v2 {

inline constexpr std::int32_t kAbiVersionV3 = 3;
inline constexpr std::int64_t kPatchWdlRadiusOneObservationSizeV2 = 83;
inline constexpr std::uint8_t kWdlAgentFlagCuedV3 = 1U;
inline constexpr std::uint8_t kWdlAgentFlagDamagedV3 = 2U;
inline constexpr std::uint8_t kWdlAgentFlagMaskV3 =
    kWdlAgentFlagCuedV3 | kWdlAgentFlagDamagedV3;
inline constexpr std::uint64_t kWdlHistoryHashMultiplierV3 =
    11400714819323198485ULL;
inline constexpr std::int32_t kWdlAgentHistoryFullStatusV3 = 250;
inline constexpr std::int32_t kWdlLineageHistoryFullStatusV3 = 251;

enum WdlGenerationMetricV2 : std::int64_t {
    kWdlTickCountV2 = 0,
    kWdlEventCountV2,
    kWdlCueCellCountSumV2,
    kWdlActiveCellCountSumV2,
    kWdlCueWithoutActiveCountV2,
    kWdlActiveWithoutPriorCueCountV2,
    kWdlCueToActiveDelaySumV2,
    kWdlCueToActiveDelayCountV2,
    kWdlAgentCueExposureCountV2,
    kWdlAgentActiveExposureCountV2,
    kWdlLethalExposureCountV2,
    kWdlCueMoveAwayCountV2,
    kWdlCueMoveTowardCountV2,
    kWdlCueStayedCountV2,
    kWdlPreHazardExitCountV2,
    kWdlHazardEnterCountV2,
    kWdlHazardExposureAfterCueCountV2,
    kWdlDamageTotalV2,
    kWdlDamageMaxV2,
    kWdlGenerationMetricCountV2,
};

inline constexpr std::int64_t kWdlTickIntegerMetricCountV2 = 6;
inline constexpr std::int64_t kWdlScheduledEventIntegerCountV2 = 5;

struct FlatWdlConfigV2 {
    std::int32_t schema_version;
    std::int32_t delay_steps;
    std::int32_t cue_duration_steps;
    std::int32_t active_duration_steps;
    std::int32_t event_period_steps;
    std::int32_t radius;
    float intensity;
    float observation_dropout_rate;
    std::uint8_t enabled;
    std::uint8_t cue_enabled;
    std::uint8_t observation_enabled;
    std::uint8_t damage_enabled;
    std::uint8_t deterministic_schedule_enabled;
};

// All pointers are device-resident, cell-major authorities. The seven WDL
// state arrays and schedule arrays remain live across every tick and generation
// in one native trajectory. Reporting arrays are bounded generation windows.
struct FlatWdlContextV2 {
    bool enabled;
    FlatWdlConfigV2 config;
    const std::int32_t* radius_by_cell;
    float* cue;
    std::int32_t* timer;
    float* active;
    float* intensity;
    std::int32_t* age;
    std::int64_t* event_id;
    std::int8_t* prior_cue;
    std::int64_t* base_seed;
    std::int64_t* next_absolute_step;
    std::int64_t* tick_integer_metrics;
    double* tick_delay_sum;
    std::int64_t* scheduled_event_integers;
    float* scheduled_event_intensity;
    std::int32_t* status_by_tick;
    double* generation_metrics;
    std::int64_t* agent_history_keys;
    std::uint8_t* agent_history_flags;
    std::int64_t* lineage_history_keys;
    double* lineage_history_damage;
    std::int64_t stats_ticks_per_cell;
    std::int64_t generation_rows_per_cell;
    std::int64_t generation_base;
    std::int64_t history_capacity;
};

// The v3 mechanic installs one context for the duration of a single native
// trajectory call. Calls without an installed context execute the unchanged
// Patch path.
void install_flat_wdl_context_v2(const FlatWdlContextV2& context);
void clear_flat_wdl_context_v2() noexcept;
const FlatWdlContextV2& current_flat_wdl_context_v2() noexcept;

// HAPB is deliberately a separate compact-policy ABI.  It consumes only one
// mapped policy tile; callers cannot pass a resident [B,C512,9991] CUDA bank.
// One CUDA thread owns each row so the fixed-MLP scalar operation order is the
// same as the resident scalar policy path.
torch::Tensor hapb_compact_policy_logits_v1(
    const torch::Tensor& mapped_policy_rows,
    const torch::Tensor& observations,
    const torch::Tensor& global_row_slots,
    const torch::Tensor& drives,
    const torch::Tensor& attack_bias,
    const torch::Tensor& action_temperature,
    const torch::Tensor& drive_logit_gain,
    std::int64_t hidden_size,
    std::int64_t action_count,
    float w_hunger_gather,
    float w_pain_rest,
    float w_fatigue_rest,
    float w_pain_attack,
    float w_fear_attack
);

// Selected P256 prefix path.  It reads indexed rows from the noncanonical
// cache directly; no gathered policy-row tensor is materialized.
torch::Tensor hapb_hot_prefix_policy_logits_v1(
    const torch::Tensor& hot_prefix_cache,
    const torch::Tensor& observations,
    const torch::Tensor& global_row_slots,
    const torch::Tensor& drives,
    const torch::Tensor& attack_bias,
    const torch::Tensor& action_temperature,
    const torch::Tensor& drive_logit_gain,
    std::int64_t hidden_size,
    std::int64_t action_count,
    float w_hunger_gather,
    float w_pain_rest,
    float w_fatigue_rest,
    float w_pain_attack,
    float w_fear_attack
);

// Compose a policy refill from the installed HAPB transaction context.  The
// canonical bank remains pageable host authority: overflow rows are copied
// through its one pinned/CUDA tile while P256 rows are indexed in-place from
// the device cache.  The result is dense action logits only, never a CUDA
// policy bank.  The active-row arrays are flattened B48/C512 ordered-row
// coordinates where the result is scattered for exact serial settlement.
torch::Tensor hapb_refill_policy_logits_from_context_v1(
    const torch::Tensor& observations,
    const torch::Tensor& stable_hot_prefix_rows,
    const torch::Tensor& stable_hot_prefix_active_rows,
    const torch::Tensor& drives,
    const torch::Tensor& attack_bias,
    const torch::Tensor& action_temperature,
    const torch::Tensor& drive_logit_gain,
    std::int64_t hidden_size,
    std::int64_t action_count,
    float w_hunger_gather,
    float w_pain_rest,
    float w_fatigue_rest,
    float w_pain_attack,
    float w_fear_attack
);

// Exact ordinary-birth fixed-MLP arithmetic over bounded CUDA work rows.
// Callers own all three non-overlapping row views in their HAPB tile arena.
// This is deliberately an in-place workbench ABI rather than a tensor-return
// allocation, so it cannot construct a CUDA C512 policy bank.
void hapb_ordinary_birth_mutate_rows_v1(
    const torch::Tensor& parent_rows,
    const torch::Tensor& taped_delta_rows,
    const torch::Tensor& child_rows
);

// B1-only resident-layout oracle.  It has no authority semantics and retains
// no bank: callers provide a bounded row slice solely to compare the current
// resident scalar helper with HAPB's mapped-tile invocation.
torch::Tensor hapb_resident_policy_logits_oracle_v1(
    const torch::Tensor& resident_policy_rows,
    const torch::Tensor& observations,
    const torch::Tensor& global_row_slots,
    const torch::Tensor& drives,
    const torch::Tensor& attack_bias,
    const torch::Tensor& action_temperature,
    const torch::Tensor& drive_logit_gain,
    std::int64_t hidden_size,
    std::int64_t action_count,
    float w_hunger_gather,
    float w_pain_rest,
    float w_fatigue_rest,
    float w_pain_attack,
    float w_fear_attack
);

}  // namespace eiel::cuda_architecture_b::flattened_wdl_controller_v2

#endif
