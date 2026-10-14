#ifndef EIEL_CUDA_ARCHITECTURE_B_FLATTENED_GMNW_CONTROLLER_V1_HPP_
#define EIEL_CUDA_ARCHITECTURE_B_FLATTENED_GMNW_CONTROLLER_V1_HPP_

#include <cstdint>

#include "cuda_architecture_b_gmnw_control_matrix_v1.hpp"

namespace eiel::cuda_architecture_b::flattened_gmnw_controller_v1 {

inline constexpr std::int32_t kAbiVersionV1 = 1;
inline constexpr std::int32_t kAbiVersionV2 = 2;
inline constexpr std::int64_t kObservationColumnsV1 = 12;
inline constexpr std::int64_t kCounterColumnsV1 = 6;
inline constexpr std::int64_t kQualificationTraceTaskStateColumnsV1 = 9;
inline constexpr std::int64_t kQualificationTraceHiddenWidthV1 = 64;
inline constexpr std::int64_t kQualificationTraceVersionV1 = 1;
inline constexpr std::int64_t kQualificationInitialStreamCountV1 = 64;
inline constexpr std::int64_t kQualificationEpisodesPerStreamV1 = 48;
inline constexpr std::int64_t kQualificationDecisionRowsV1 =
    kQualificationInitialStreamCountV1 * kQualificationEpisodesPerStreamV1;
inline constexpr std::int64_t kQualificationDecisionColumnsV1 = 6;
inline constexpr std::int64_t kQualificationDecisionTapeVersionV1 = 1;
inline constexpr std::int64_t kSelectionMetricsVersionV1 = 1;
inline constexpr std::int64_t kSelectionPolicyFamilyCountV1 = 2;
inline constexpr std::int64_t kSelectionCountMetricCountV1 = 7;

enum VenvConsequenceProfileV1 : std::int32_t {
    kVenvConsequenceLegacyV1 = 0,
    kVenvConsequenceP1V1 = 1,
    kVenvConsequenceP2V1 = 2,
    kVenvConsequenceP3V1 = 3,
};

enum SelectionCountMetricV1 : std::int64_t {
    kSelectionAlivePreTurnoverV1 = 0,
    kSelectionAlivePostTurnoverV1,
    kSelectionDecisionCountV1,
    kSelectionCorrectCountV1,
    kSelectionTaskDeathCountV1,
    kSelectionParentPoolCountV1,
    kSelectionTurnoverOffspringCountV1,
};

struct GmnwSelectionMetricsContextV1 {
    bool enabled;
    std::int64_t* counts;
    double* recurrent_upkeep_charge_total;
    std::int32_t* status;
};

struct GmnwQualificationTraceContextV1 {
    bool enabled;
    std::int64_t output_ticks_per_cell;
    std::int64_t* active_count;
    std::int64_t* ordered_agent_ids;
    std::int32_t* ordered_slots;
    float* hidden_before;
    float* hidden_after;
    std::int64_t* task_state_before;
    std::int64_t* task_state_after;
};

struct GmnwQualificationDecisionTapeContextV1 {
    bool enabled;
    std::int8_t* present;
    std::int64_t* rows;
    std::int32_t* status;
};

struct FlatGmnwContextV1 {
    bool enabled;
    std::int64_t* stream_id;
    std::int64_t* episode_index;
    std::int8_t* cue_identity;
    std::int8_t* cue_phase;
    std::int32_t* phase_tick;
    std::int32_t* delay_remaining;
    bool* decision_due;
    std::int8_t* correct_action;
    std::int8_t* last_response;
    std::int8_t* correctness;
    std::int8_t* consequence_status;
    std::int64_t* task_rng_state;
    std::int64_t* task_rng_cursor;
    float* reproduction_scale;
    std::int64_t* counters;
    double* damage_total;
    std::int32_t* status;
    const std::int32_t* cue_alphabet;
    const std::int32_t* delay_ticks;
    const bool* distractors_enabled;
    const bool* reset_hidden_each_tick;
    const std::int64_t* task_seed;
    const std::int32_t* venv_consequence_profile;
    std::int64_t cells;
    std::int64_t capacity;
    std::int64_t generation_base;
    std::int64_t generation_rows_per_cell;
    std::int32_t cue_ticks;
    std::int32_t consequence_ticks;
    std::int32_t grace_generation_count;
    std::int32_t severe_generation_count;
    double grace_damage_fraction;
    double severe_damage_fraction;
    double lethal_damage_fraction;
    float grace_reproduction_scale;
    bool terminal_low_population_no_reinit;
    GmnwQualificationTraceContextV1 qualification_trace;
    GmnwQualificationDecisionTapeContextV1 qualification_decision_tape;
    gmnw_control_matrix_v1::GmnwControlMatrixContextV1 control_matrix;
    GmnwSelectionMetricsContextV1 selection_metrics;
};

void install_flat_gmnw_context_v1(const FlatGmnwContextV1& context);
void clear_flat_gmnw_context_v1() noexcept;
const FlatGmnwContextV1& current_flat_gmnw_context_v1() noexcept;

}  // namespace eiel::cuda_architecture_b::flattened_gmnw_controller_v1

#endif
