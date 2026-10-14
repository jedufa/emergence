#include "engine/native_runtime/wdl/wdl_cpu_oracle_c_api.h"

#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    eiel_wdl_cpu_oracle_config_v1 config{};
    config.schema_version = 1;
    config.height = 6;
    config.width = 6;
    config.delay_steps = 4;
    config.cue_duration_steps = 2;
    config.active_duration_steps = 3;
    config.event_period_steps = 12;
    config.intensity = 0.25F;
    config.observation_dropout_rate = 0.35F;
    config.enabled = 1;
    config.cue_enabled = 1;
    config.observation_enabled = 1;
    config.damage_enabled = 1;
    config.deterministic_schedule_enabled = 1;
    config.view_radius = 1;
    config.max_scheduled_events = 32;
    config.max_damage_events = 64;

    std::int32_t status = -1;
    auto* oracle = eiel_wdl_cpu_oracle_create_v1(&config, 101, 1, &status);
    if (oracle == nullptr || status != 0) return 1;
    const std::uint8_t alive[1] = {1};
    const std::int32_t y[1] = {0};
    const std::int32_t x[1] = {0};
    const float health[1] = {1.0F};
    const float scale[1] = {1.0F};
    if (eiel_wdl_cpu_oracle_set_agents_v1(oracle, alive, y, x, health, scale, 1) != 0) return 2;
    for (std::int64_t step = 0; step < 5; ++step) {
        eiel_wdl_tick_receipt_v1 receipt{};
        if (eiel_wdl_cpu_oracle_step_v1(oracle, step, nullptr, nullptr, 0, 0, &receipt) != 0) return 3;
    }
    const std::uint64_t size = eiel_wdl_cpu_oracle_checkpoint_size_v1(oracle);
    std::vector<std::uint8_t> checkpoint(size);
    std::uint64_t written = 0;
    if (eiel_wdl_cpu_oracle_write_checkpoint_v1(
            oracle, checkpoint.data(), checkpoint.size(), &written) != 0
        || written != size) return 4;

    auto* restored = eiel_wdl_cpu_oracle_create_v1(&config, -1, 1, &status);
    if (restored == nullptr || status != 0) return 5;
    if (eiel_wdl_cpu_oracle_restore_checkpoint_v1(restored, checkpoint.data(), checkpoint.size()) != 0) return 6;
    eiel_wdl_tick_receipt_v1 receipt{};
    if (eiel_wdl_cpu_oracle_step_v1(restored, 5, nullptr, nullptr, 0, 0, &receipt) != 0) return 7;
    const std::int32_t invalid_y[1] = {6};
    if (eiel_wdl_cpu_oracle_step_v1(restored, 6, invalid_y, x, 1, 0, &receipt) == 0) return 8;

    eiel_wdl_cpu_oracle_destroy_v1(restored);
    eiel_wdl_cpu_oracle_destroy_v1(oracle);
    std::cout << "wdl_cpu_oracle_sanitizer_probe=ok\n";
    return 0;
}
