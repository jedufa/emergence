#pragma once

#include "wdl_native_contract.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace eiel::wdl::cpu_oracle_v1 {

using native_v1::Config;
using native_v1::DamageMetrics;
using native_v1::ScheduledEventRow;
using native_v1::Status;
using native_v1::TickMetrics;

inline constexpr std::int32_t kOracleSchemaVersion = 1;

struct OracleConfig {
    Config wdl{};
    std::int32_t view_radius = 1;
    std::size_t max_scheduled_events = 4096;
    std::size_t max_damage_events = 65536;
};

struct DamageEventRow {
    std::int64_t abs_step = 0;
    std::int32_t agent_id = -1;
    std::int32_t y = -1;
    std::int32_t x = -1;
    float raw_intensity = 0.0F;
    float damage_scale = 1.0F;
    float effective_damage = 0.0F;
    float health_before = 0.0F;
    float health_after = 0.0F;
    std::uint8_t died = 0;
};

struct GenerationSummary {
    std::int64_t event_count = 0;
    double cue_cell_count_mean = 0.0;
    double active_cell_count_mean = 0.0;
    std::int64_t agent_cue_exposure_count = 0;
    std::int64_t agent_active_exposure_count = 0;
    double damage_total = 0.0;
    float damage_max = 0.0F;
    std::int64_t agents_damaged = 0;
    double cue_to_active_delay_mean = 0.0;
    std::int64_t cue_without_active_count = 0;
    std::int64_t active_without_prior_cue_count = 0;
    std::int64_t enabled = 0;
    std::int64_t tick_count = 0;
};

struct TickReceipt {
    std::int64_t abs_step = -1;
    TickMetrics tick{};
    DamageMetrics damage{};
    ScheduledEventRow scheduled_event{};
    std::size_t observation_rows = 0;
    std::size_t observation_width = 0;
};

class CpuOracle {
public:
    CpuOracle() = default;

    Status initialize(
        const OracleConfig& config,
        std::int64_t base_seed,
        std::size_t agent_count
    );

    Status set_agents(
        const std::uint8_t* alive,
        const std::int32_t* pos_y,
        const std::int32_t* pos_x,
        const float* health,
        const float* damage_scale,
        std::size_t agent_count
    );

    // One atomic WDL tick: phase transition -> pre-action cue observation ->
    // post-action positions -> delayed damage -> event/summary publication.
    Status step(
        std::int64_t abs_step,
        const std::int32_t* post_action_y,
        const std::int32_t* post_action_x,
        std::size_t position_count,
        bool cancel_requested,
        TickReceipt* receipt
    );

    // Reset generation-scoped reductions without disturbing world state,
    // agent state, absolute-step continuity, or deterministic scheduling.
    Status begin_generation(bool clear_event_journals) noexcept;

    const OracleConfig& config() const noexcept { return config_; }
    native_v1::ScheduleState schedule() const noexcept { return schedule_; }
    std::size_t agent_count() const noexcept { return alive_.size(); }
    std::size_t observation_width() const noexcept;
    const std::vector<float>& last_observation() const noexcept { return last_observation_; }
    const std::vector<std::int32_t>& last_observation_agent_ids() const noexcept {
        return last_observation_agent_ids_;
    }
    const std::vector<ScheduledEventRow>& scheduled_events() const noexcept { return scheduled_events_; }
    const std::vector<DamageEventRow>& damage_events() const noexcept { return damage_events_; }
    GenerationSummary generation_summary() const noexcept;

    native_v1::ConstStateView state_view() const noexcept;
    const std::vector<std::uint8_t>& agent_alive() const noexcept { return alive_; }
    const std::vector<std::int32_t>& agent_pos_y() const noexcept { return pos_y_; }
    const std::vector<std::int32_t>& agent_pos_x() const noexcept { return pos_x_; }
    const std::vector<float>& agent_health() const noexcept { return health_; }
    const std::vector<float>& agent_damage_scale() const noexcept { return damage_scale_; }

    std::size_t checkpoint_size_bytes() const noexcept;
    Status write_checkpoint(
        std::uint8_t* output,
        std::size_t capacity,
        std::size_t* bytes_written
    ) const noexcept;
    Status restore_checkpoint(const std::uint8_t* input, std::size_t size) noexcept;

private:
    struct SummaryAccumulator {
        std::int64_t event_count = 0;
        double cue_cell_count_sum = 0.0;
        double active_cell_count_sum = 0.0;
        std::int64_t agent_cue_exposure_count = 0;
        std::int64_t agent_active_exposure_count = 0;
        double damage_total = 0.0;
        float damage_max = 0.0F;
        double cue_to_active_delay_sum = 0.0;
        std::int64_t cue_to_active_delay_count = 0;
        std::int64_t cue_without_active_count = 0;
        std::int64_t active_without_prior_cue_count = 0;
        std::int64_t tick_count = 0;
    };

    native_v1::StateView mutable_state_view() noexcept;
    native_v1::AgentStateView mutable_agent_view() noexcept;
    Status validate_agents() const noexcept;
    void build_observations();
    std::int64_t count_cue_exposures() const noexcept;

    OracleConfig config_{};
    native_v1::ScheduleState schedule_{};
    std::vector<float> cue_;
    std::vector<std::int32_t> timer_;
    std::vector<float> active_;
    std::vector<float> intensity_;
    std::vector<std::int32_t> age_;
    std::vector<std::int64_t> event_id_;
    std::vector<std::int8_t> prior_cue_;
    std::vector<std::uint8_t> alive_;
    std::vector<std::int32_t> pos_y_;
    std::vector<std::int32_t> pos_x_;
    std::vector<float> health_;
    std::vector<float> damage_scale_;
    std::vector<std::uint8_t> damaged_ever_;
    std::vector<std::int32_t> last_observation_agent_ids_;
    std::vector<float> last_observation_;
    std::vector<ScheduledEventRow> scheduled_events_;
    std::vector<DamageEventRow> damage_events_;
    SummaryAccumulator summary_{};
    bool initialized_ = false;
};

}  // namespace eiel::wdl::cpu_oracle_v1
