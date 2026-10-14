#include "stable_gru_cpu_oracle_c_api_v1.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

int main() {
    std::int64_t parameter_count = 0;
    if (eiel_stable_gru_packed_parameter_count_v1(3, &parameter_count) != 0
        || parameter_count <= 0) {
        return 1;
    }
    std::array<eiel_stable_gru_step_config_v1, 2> configs{};
    for (auto& config : configs) {
        config.contract_version = 1;
        config.observation_size = 3;
        config.active_memory_size = 64;
        config.temperature = 1.0F;
        config.gain = 1.0F;
    }
    configs[1].active_memory_size = 8;
    std::array<float, 6> observations{};
    std::array<float, 128> hidden_before{};
    std::vector<float> parameters(static_cast<std::size_t>(2 * parameter_count), 0.0F);
    std::array<float, 128> scratch_hidden{};
    std::array<float, 14> scratch_logits{};
    std::array<std::int32_t, 2> scratch_actions{};
    std::array<float, 128> hidden_after{};
    std::array<float, 14> logits{};
    std::array<std::int32_t, 2> actions{};
    if (eiel_stable_gru_forward_batch_v1(
            configs.data(), 2, observations.data(), observations.size(),
            hidden_before.data(), hidden_before.size(), parameters.data(), parameters.size(), 2,
            scratch_hidden.data(), scratch_hidden.size(), scratch_logits.data(),
            scratch_logits.size(), scratch_actions.data(), scratch_actions.size(),
            hidden_after.data(), hidden_after.size(), logits.data(), logits.size(),
            actions.data(), actions.size()) != 0) {
        return 2;
    }
    if (actions[0] != 0 || actions[1] != 0) return 3;

    observations[3] = std::numeric_limits<float>::quiet_NaN();
    hidden_after.fill(91.0F);
    logits.fill(92.0F);
    actions.fill(93);
    if (eiel_stable_gru_forward_batch_v1(
            configs.data(), 2, observations.data(), observations.size(),
            hidden_before.data(), hidden_before.size(), parameters.data(), parameters.size(), 2,
            scratch_hidden.data(), scratch_hidden.size(), scratch_logits.data(),
            scratch_logits.size(), scratch_actions.data(), scratch_actions.size(),
            hidden_after.data(), hidden_after.size(), logits.data(), logits.size(),
            actions.data(), actions.size()) != 11) {
        return 4;
    }
    if (!std::all_of(hidden_after.begin(), hidden_after.end(), [](float v) { return v == 91.0F; })
        || !std::all_of(logits.begin(), logits.end(), [](float v) { return v == 92.0F; })
        || actions[0] != 93 || actions[1] != 93) {
        return 5;
    }

    std::array<float, 128> lifecycle_hidden{};
    std::array<std::uint64_t, 2> lifecycle_tokens{101, 202};
    lifecycle_hidden.fill(4.0F);
    if (eiel_stable_gru_lifecycle_v1(
            lifecycle_hidden.data(), lifecycle_hidden.size(), lifecycle_tokens.data(),
            lifecycle_tokens.size(), 2, 0, 101, 101, 99) != 3
        || lifecycle_hidden[0] != 4.0F || lifecycle_tokens[0] != 101) {
        return 6;
    }

    eiel_stable_gru_checkpoint_row_v1 checkpoint{};
    if (eiel_stable_gru_checkpoint_project_v1(
            lifecycle_hidden.data(), lifecycle_hidden.size(), lifecycle_tokens.data(),
            lifecycle_tokens.size(), 2, 1, 202, &checkpoint) != 0) {
        return 7;
    }
    std::array<float, 128> restored_hidden{};
    std::array<std::uint64_t, 2> restored_tokens{1, 999};
    restored_hidden.fill(-5.0F);
    if (eiel_stable_gru_checkpoint_restore_v1(
            &checkpoint, restored_hidden.data(), restored_hidden.size(), restored_tokens.data(),
            restored_tokens.size(), 2, 999) != 0
        || restored_hidden[64] != 4.0F || restored_tokens[1] != 202) {
        return 8;
    }
    return 0;
}
