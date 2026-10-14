#include "stable_gru_cpu_oracle_c_api_v1.h"

#include "stable_gru_v1.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace stable = eiel::native::recurrent_v1;

namespace {

static_assert(std::is_standard_layout_v<eiel_stable_gru_step_config_v1>);
static_assert(std::is_trivially_copyable_v<eiel_stable_gru_step_config_v1>);
static_assert(sizeof(eiel_stable_gru_step_config_v1) == 72U);
static_assert(offsetof(eiel_stable_gru_step_config_v1, observation_size) == 8U);
static_assert(offsetof(eiel_stable_gru_step_config_v1, drives) == 32U);
static_assert(offsetof(eiel_stable_gru_step_config_v1, fear_attack_weight) == 68U);
static_assert(std::is_standard_layout_v<eiel_stable_gru_checkpoint_row_v1>);
static_assert(std::is_trivially_copyable_v<eiel_stable_gru_checkpoint_row_v1>);
static_assert(sizeof(eiel_stable_gru_checkpoint_row_v1) == 296U);
static_assert(offsetof(eiel_stable_gru_checkpoint_row_v1, token) == 32U);
static_assert(offsetof(eiel_stable_gru_checkpoint_row_v1, hidden) == 40U);

int32_t code(const stable::StatusV1 status) noexcept {
    return static_cast<int32_t>(status);
}

stable::RecurrentStateViewV1 state_view(
    float* const hidden,
    const int64_t hidden_element_count,
    uint64_t* const tokens,
    const int64_t token_count,
    const int64_t row_count
) noexcept {
    return stable::RecurrentStateViewV1{
        hidden, hidden_element_count, tokens, token_count, 1, row_count,
        hidden_element_count,
        EIEL_STABLE_GRU_HIDDEN_WIDTH_V1, row_count, 1};
}

stable::ConstRecurrentStateViewV1 const_state_view(
    const float* const hidden,
    const int64_t hidden_element_count,
    const uint64_t* const tokens,
    const int64_t token_count,
    const int64_t row_count
) noexcept {
    return stable::ConstRecurrentStateViewV1{
        hidden, hidden_element_count, tokens, token_count, 1, row_count,
        hidden_element_count,
        EIEL_STABLE_GRU_HIDDEN_WIDTH_V1, row_count, 1};
}

}  // namespace

extern "C" {

uint32_t eiel_stable_gru_cpu_oracle_abi_version_v1(void) {
    return EIEL_STABLE_GRU_CPU_ORACLE_ABI_VERSION_V1;
}

int32_t eiel_stable_gru_packed_parameter_count_v1(
    const int64_t observation_size,
    int64_t* const parameter_count_out
) {
    if (parameter_count_out == nullptr) return code(stable::StatusV1::kNullPointer);
    stable::StableGruPackedLayoutV1 layout{};
    const stable::StatusV1 status = stable::stable_gru_packed_layout(
        {stable::kRecurrentContractVersionV1, observation_size,
         EIEL_STABLE_GRU_HIDDEN_WIDTH_V1, EIEL_STABLE_GRU_ACTION_WIDTH_V1},
        layout);
    if (!stable::ok(status)) return code(status);
    *parameter_count_out = layout.total_elements;
    return code(stable::StatusV1::kSuccess);
}

int32_t eiel_stable_gru_forward_v1(
    const eiel_stable_gru_step_config_v1* const config,
    const float* const observation,
    const int64_t observation_count,
    const float* const hidden_before,
    const int64_t hidden_before_count,
    const float* const packed_parameters,
    const int64_t packed_parameter_count,
    float* const hidden_after,
    const int64_t hidden_after_count,
    float* const logits,
    const int64_t logits_count,
    int32_t* const action_out
) {
    if (config == nullptr) return code(stable::StatusV1::kNullPointer);
    std::array<float, EIEL_STABLE_GRU_HIDDEN_WIDTH_V1> encoder{};
    std::array<float, 3 * EIEL_STABLE_GRU_HIDDEN_WIDTH_V1> input_projection{};
    std::array<float, 3 * EIEL_STABLE_GRU_HIDDEN_WIDTH_V1> hidden_projection{};
    std::array<float, EIEL_STABLE_GRU_HIDDEN_WIDTH_V1> scratch_hidden{};
    std::array<float, EIEL_STABLE_GRU_ACTION_WIDTH_V1> scratch_logits{};
    stable::StableGruStepInputV1 input{
        {config->contract_version, config->observation_size,
         EIEL_STABLE_GRU_HIDDEN_WIDTH_V1, EIEL_STABLE_GRU_ACTION_WIDTH_V1},
        observation, observation_count, hidden_before, hidden_before_count,
        config->active_memory_size, packed_parameters, packed_parameter_count,
        config->aggression, config->temperature, config->drives, 4,
        config->gain < 0.0F ? 0.0F : config->gain,
        config->hunger_gather_weight, config->pain_rest_weight,
        config->fatigue_rest_weight, config->pain_attack_weight,
        config->fear_attack_weight};
    stable::StableGruStepScratchV1 scratch{
        encoder.data(), static_cast<int64_t>(encoder.size()),
        input_projection.data(), static_cast<int64_t>(input_projection.size()),
        hidden_projection.data(), static_cast<int64_t>(hidden_projection.size()),
        scratch_hidden.data(), static_cast<int64_t>(scratch_hidden.size()),
        scratch_logits.data(), static_cast<int64_t>(scratch_logits.size())};
    stable::StableGruStepOutputV1 output{
        hidden_after, hidden_after_count, logits, logits_count, action_out};
    return code(stable::stable_gru_forward_one(input, scratch, output, nullptr));
}

int32_t eiel_stable_gru_forward_batch_v1(
    const eiel_stable_gru_step_config_v1* const configs,
    const int64_t config_count,
    const float* const observations,
    const int64_t observation_element_count,
    const float* const hidden_before,
    const int64_t hidden_before_element_count,
    const float* const packed_parameters,
    const int64_t packed_parameter_element_count,
    const int64_t row_count,
    float* const scratch_hidden,
    const int64_t scratch_hidden_element_count,
    float* const scratch_logits,
    const int64_t scratch_logits_element_count,
    int32_t* const scratch_actions,
    const int64_t scratch_action_count,
    float* const hidden_after,
    const int64_t hidden_after_element_count,
    float* const logits,
    const int64_t logits_element_count,
    int32_t* const actions,
    const int64_t action_count
) {
    if (configs == nullptr || observations == nullptr || hidden_before == nullptr
        || packed_parameters == nullptr || scratch_hidden == nullptr
        || scratch_logits == nullptr || scratch_actions == nullptr
        || hidden_after == nullptr || logits == nullptr || actions == nullptr) {
        return code(stable::StatusV1::kNullPointer);
    }
    if (row_count <= 0 || config_count != row_count) {
        return code(stable::StatusV1::kInvalidDimension);
    }
    const int64_t observation_size = configs[0].observation_size;
    stable::StableGruPackedLayoutV1 layout{};
    stable::StatusV1 status = stable::stable_gru_packed_layout(
        {stable::kRecurrentContractVersionV1, observation_size,
         EIEL_STABLE_GRU_HIDDEN_WIDTH_V1, EIEL_STABLE_GRU_ACTION_WIDTH_V1}, layout);
    if (!stable::ok(status)) return code(status);
    int64_t expected_observations = 0;
    int64_t expected_hidden = 0;
    int64_t expected_logits = 0;
    int64_t expected_parameters = 0;
    if (!stable::checked_mul_i64(row_count, observation_size, expected_observations)
        || !stable::checked_mul_i64(row_count, EIEL_STABLE_GRU_HIDDEN_WIDTH_V1, expected_hidden)
        || !stable::checked_mul_i64(row_count, EIEL_STABLE_GRU_ACTION_WIDTH_V1, expected_logits)
        || !stable::checked_mul_i64(row_count, layout.total_elements, expected_parameters)) {
        return code(stable::StatusV1::kIntegerOverflow);
    }
    if (observation_element_count != expected_observations
        || hidden_before_element_count != expected_hidden
        || packed_parameter_element_count != expected_parameters
        || scratch_hidden_element_count < expected_hidden
        || scratch_logits_element_count < expected_logits
        || scratch_action_count < row_count
        || hidden_after_element_count < expected_hidden
        || logits_element_count < expected_logits || action_count < row_count) {
        return code(stable::StatusV1::kExtentMismatch);
    }
    for (int64_t row = 0; row < row_count; ++row) {
        if (configs[row].observation_size != observation_size) {
            return code(stable::StatusV1::kExtentMismatch);
        }
        const int32_t row_status = eiel_stable_gru_forward_v1(
            &configs[row], observations + row * observation_size, observation_size,
            hidden_before + row * EIEL_STABLE_GRU_HIDDEN_WIDTH_V1,
            EIEL_STABLE_GRU_HIDDEN_WIDTH_V1,
            packed_parameters + row * layout.total_elements, layout.total_elements,
            scratch_hidden + row * EIEL_STABLE_GRU_HIDDEN_WIDTH_V1,
            EIEL_STABLE_GRU_HIDDEN_WIDTH_V1,
            scratch_logits + row * EIEL_STABLE_GRU_ACTION_WIDTH_V1,
            EIEL_STABLE_GRU_ACTION_WIDTH_V1, scratch_actions + row);
        if (row_status != code(stable::StatusV1::kSuccess)) return row_status;
    }
    for (int64_t index = 0; index < expected_hidden; ++index) {
        hidden_after[index] = scratch_hidden[index];
    }
    for (int64_t index = 0; index < expected_logits; ++index) {
        logits[index] = scratch_logits[index];
    }
    for (int64_t row = 0; row < row_count; ++row) {
        actions[row] = scratch_actions[row];
    }
    return code(stable::StatusV1::kSuccess);
}

int32_t eiel_stable_gru_lifecycle_v1(
    float* const hidden,
    const int64_t hidden_element_count,
    uint64_t* const tokens,
    const int64_t token_count,
    const int64_t row_count,
    const int64_t row_index,
    const uint64_t expected_token,
    const uint64_t publication_token,
    const uint32_t operation
) {
    stable::RecurrentStateViewV1 view = state_view(
        hidden, hidden_element_count, tokens, token_count, row_count);
    return code(stable::apply_recurrent_lifecycle(
        view, EIEL_STABLE_GRU_HIDDEN_WIDTH_V1, 0, row_index, expected_token,
        publication_token, static_cast<stable::LifecycleOperationV1>(operation), nullptr));
}

int32_t eiel_stable_gru_mutate_v1(
    const float* const parent,
    const int64_t parameter_count,
    const float* const additive_draws,
    const uint8_t* const mutation_mask,
    const float scale,
    float* const scratch,
    const int64_t scratch_count,
    float* const child,
    const int64_t child_count
) {
    const stable::MutationDrawTapeV1 tape{
        stable::kRecurrentContractVersionV1, additive_draws, parameter_count,
        mutation_mask, parameter_count, scale};
    return code(stable::inherit_and_mutate_parameters(
        parent, parameter_count, tape, scratch, scratch_count, child, child_count, nullptr));
}

int32_t eiel_stable_gru_checkpoint_project_v1(
    const float* const hidden,
    const int64_t hidden_element_count,
    const uint64_t* const tokens,
    const int64_t token_count,
    const int64_t row_count,
    const int64_t row_index,
    const uint64_t expected_token,
    eiel_stable_gru_checkpoint_row_v1* const checkpoint_out
) {
    if (checkpoint_out == nullptr) return code(stable::StatusV1::kNullPointer);
    stable::CheckpointRowV1 checkpoint{
        0, 0, 0, EIEL_STABLE_GRU_HIDDEN_WIDTH_V1, 0,
        checkpoint_out->hidden, EIEL_STABLE_GRU_HIDDEN_WIDTH_V1};
    const stable::StatusV1 status = stable::project_checkpoint_row(
        const_state_view(hidden, hidden_element_count, tokens, token_count, row_count),
        EIEL_STABLE_GRU_HIDDEN_WIDTH_V1, 0, row_index, expected_token, checkpoint, nullptr);
    if (!stable::ok(status)) return code(status);
    checkpoint_out->contract_version = checkpoint.contract_version;
    checkpoint_out->cell_index = checkpoint.cell_index;
    checkpoint_out->agent_index = checkpoint.agent_index;
    checkpoint_out->hidden_size = checkpoint.hidden_size;
    checkpoint_out->token = checkpoint.token;
    return code(stable::StatusV1::kSuccess);
}

int32_t eiel_stable_gru_checkpoint_restore_v1(
    const eiel_stable_gru_checkpoint_row_v1* const checkpoint,
    float* const hidden,
    const int64_t hidden_element_count,
    uint64_t* const tokens,
    const int64_t token_count,
    const int64_t row_count,
    const uint64_t expected_destination_token
) {
    if (checkpoint == nullptr) return code(stable::StatusV1::kNullPointer);
    const stable::ConstCheckpointRowV1 source{
        checkpoint->contract_version, checkpoint->cell_index, checkpoint->agent_index,
        checkpoint->hidden_size, checkpoint->token, checkpoint->hidden,
        EIEL_STABLE_GRU_HIDDEN_WIDTH_V1};
    stable::RecurrentStateViewV1 destination = state_view(
        hidden, hidden_element_count, tokens, token_count, row_count);
    return code(stable::restore_checkpoint_row(
        source, destination, expected_destination_token, nullptr));
}

}  // extern "C"
