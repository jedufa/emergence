#pragma once

#include "recurrent_common_v1.hpp"

#include <cstdint>

#if defined(__CUDACC__)
#define EIEL_RECURRENT_HOST_DEVICE __host__ __device__
#else
#define EIEL_RECURRENT_HOST_DEVICE
#endif

namespace eiel::native::recurrent_v1 {

struct StableGruStepInputV1 {
    StableGruShapeV1 shape;
    const float* observation;
    std::int64_t observation_count;
    const float* hidden_before;
    std::int64_t hidden_before_count;
    std::int64_t active_memory_size;
    const float* packed_parameters;
    std::int64_t packed_parameter_count;
    float aggression;
    float temperature;
    const float* drives;
    std::int64_t drive_count;
    float gain;
    float hunger_gather_weight;
    float pain_rest_weight;
    float fatigue_rest_weight;
    float pain_attack_weight;
    float fear_attack_weight;
};

struct StableGruStepScratchV1 {
    float* encoder;
    std::int64_t encoder_count;
    float* input_projection;
    std::int64_t input_projection_count;
    float* hidden_projection;
    std::int64_t hidden_projection_count;
    float* hidden_after;
    std::int64_t hidden_after_count;
    float* logits;
    std::int64_t logits_count;
};

struct StableGruStepOutputV1 {
    float* hidden_after;
    std::int64_t hidden_after_count;
    float* logits;
    std::int64_t logits_count;
    std::int32_t* action;
};

struct MutationDrawTapeV1 {
    std::uint32_t contract_version;
    const float* additive_draws;
    std::int64_t draw_count;
    const std::uint8_t* mutation_mask;
    std::int64_t mask_count;
    float scale;
};

struct CheckpointRowV1 {
    std::uint32_t contract_version;
    std::int64_t cell_index;
    std::int64_t agent_index;
    std::int64_t hidden_size;
    std::uint64_t token;
    float* hidden;
    std::int64_t hidden_count;
};

struct ConstCheckpointRowV1 {
    std::uint32_t contract_version;
    std::int64_t cell_index;
    std::int64_t agent_index;
    std::int64_t hidden_size;
    std::uint64_t token;
    const float* hidden;
    std::int64_t hidden_count;
};

struct StableGruCostConfigV1 {
    std::uint32_t contract_version;
    float base_cost;
    float per_active_hidden_cost;
};

struct StableGruCostResultV1 {
    float energy_cost;
    std::uint64_t active_hidden_elements;
    std::uint64_t parameter_elements;
};

[[nodiscard]] EIEL_RECURRENT_HOST_DEVICE StatusV1 stable_gru_packed_layout(
    const StableGruShapeV1& shape,
    StableGruPackedLayoutV1& layout
) noexcept;

[[nodiscard]] EIEL_RECURRENT_HOST_DEVICE StatusV1 validate_recurrent_state_view(
    const ConstRecurrentStateViewV1& state,
    std::int64_t hidden_size
) noexcept;

[[nodiscard]] EIEL_RECURRENT_HOST_DEVICE StatusV1 recurrent_row_address(
    const ConstRecurrentStateViewV1& state,
    std::int64_t hidden_size,
    std::int64_t cell_index,
    std::int64_t agent_index,
    std::uint64_t expected_token,
    RecurrentRowAddressV1& address
) noexcept;

// All validation and math finish in scratch before any output or counter write.
[[nodiscard]] EIEL_RECURRENT_HOST_DEVICE StatusV1 stable_gru_forward_one(
    const StableGruStepInputV1& input,
    StableGruStepScratchV1& scratch,
    StableGruStepOutputV1& output,
    StableGruCompactCountersV1* counters
) noexcept;

[[nodiscard]] EIEL_RECURRENT_HOST_DEVICE StatusV1 apply_recurrent_lifecycle(
    RecurrentStateViewV1& state,
    std::int64_t hidden_size,
    std::int64_t cell_index,
    std::int64_t agent_index,
    std::uint64_t expected_token,
    std::uint64_t publication_token,
    LifecycleOperationV1 operation,
    StableGruCompactCountersV1* counters
) noexcept;

[[nodiscard]] EIEL_RECURRENT_HOST_DEVICE StatusV1 reset_recurrent_row(
    RecurrentStateViewV1& state,
    std::int64_t hidden_size,
    std::int64_t cell_index,
    std::int64_t agent_index,
    std::uint64_t token,
    StableGruCompactCountersV1* counters
) noexcept;

[[nodiscard]] EIEL_RECURRENT_HOST_DEVICE StatusV1 initialize_birth_row(
    RecurrentStateViewV1& state,
    std::int64_t hidden_size,
    std::int64_t cell_index,
    std::int64_t agent_index,
    std::uint64_t newborn_token,
    StableGruCompactCountersV1* counters
) noexcept;

[[nodiscard]] EIEL_RECURRENT_HOST_DEVICE StatusV1 cleanup_death_row(
    RecurrentStateViewV1& state,
    std::int64_t hidden_size,
    std::int64_t cell_index,
    std::int64_t agent_index,
    std::uint64_t deceased_token,
    StableGruCompactCountersV1* counters
) noexcept;

[[nodiscard]] EIEL_RECURRENT_HOST_DEVICE StatusV1 reset_turnover_row(
    RecurrentStateViewV1& state,
    std::int64_t hidden_size,
    std::int64_t cell_index,
    std::int64_t agent_index,
    std::uint64_t replaced_token,
    std::uint64_t newborn_token,
    StableGruCompactCountersV1* counters
) noexcept;

// One draw and one mask entry correspond to every parameter, preserving draw order.
[[nodiscard]] EIEL_RECURRENT_HOST_DEVICE StatusV1 inherit_and_mutate_parameters(
    const float* parent,
    std::int64_t parameter_count,
    const MutationDrawTapeV1& tape,
    float* scratch,
    std::int64_t scratch_count,
    float* child,
    std::int64_t child_count,
    StableGruCompactCountersV1* counters
) noexcept;

[[nodiscard]] EIEL_RECURRENT_HOST_DEVICE StatusV1 validate_checkpoint_row(
    const ConstCheckpointRowV1& checkpoint
) noexcept;

[[nodiscard]] EIEL_RECURRENT_HOST_DEVICE StatusV1 project_checkpoint_row(
    const ConstRecurrentStateViewV1& state,
    std::int64_t hidden_size,
    std::int64_t cell_index,
    std::int64_t agent_index,
    std::uint64_t expected_token,
    CheckpointRowV1& destination,
    StableGruCompactCountersV1* counters
) noexcept;

[[nodiscard]] EIEL_RECURRENT_HOST_DEVICE StatusV1 restore_checkpoint_row(
    const ConstCheckpointRowV1& checkpoint,
    RecurrentStateViewV1& destination,
    std::uint64_t expected_destination_token,
    StableGruCompactCountersV1* counters
) noexcept;

[[nodiscard]] EIEL_RECURRENT_HOST_DEVICE StatusV1 stable_gru_cost(
    const StableGruShapeV1& shape,
    std::int64_t active_memory_size,
    const StableGruCostConfigV1& config,
    StableGruCostResultV1& result
) noexcept;

}  // namespace eiel::native::recurrent_v1

#undef EIEL_RECURRENT_HOST_DEVICE
