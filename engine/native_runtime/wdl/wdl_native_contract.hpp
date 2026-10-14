#pragma once

#include <cstddef>
#include <cstdint>

namespace eiel::wdl::native_v1 {

inline constexpr std::int32_t kSchemaVersion = 1;
inline constexpr std::size_t kStateDescriptorCount = 7;
inline constexpr std::size_t kObservationChannelCount = 1;
inline constexpr std::size_t kScheduledEventFieldCount = 6;
inline constexpr std::size_t kTickMetricFieldCount = 7;
inline constexpr std::size_t kGenerationSummaryFieldCount = 13;

enum class Status : std::int32_t {
    kOk = 0,
    kCancelled = 1,
    kInvalidConfig = 2,
    kInvalidState = 3,
    kInvalidCheckpoint = 4,
    kInsufficientCapacity = 5,
};

enum class Phase : std::int8_t {
    kBaseline = 0,
    kCue = 1,
    kHiddenGap = 2,
    kActiveHazard = 3,
};

struct Config {
    std::int32_t schema_version = kSchemaVersion;
    std::int32_t height = 0;
    std::int32_t width = 0;
    std::int32_t delay_steps = 8;
    std::int32_t cue_duration_steps = 4;
    std::int32_t active_duration_steps = 6;
    std::int32_t event_period_steps = 20;
    std::int32_t radius = 1;
    float intensity = 0.10F;
    float observation_dropout_rate = 0.0F;
    std::uint8_t enabled = 0;
    std::uint8_t cue_enabled = 1;
    std::uint8_t observation_enabled = 0;
    std::uint8_t damage_enabled = 1;
    std::uint8_t deterministic_schedule_enabled = 1;
};

// This is a non-owning, contiguous structure-of-arrays view. The same field
// order and fixed-width types are suitable for a host oracle or CUDA kernel.
struct StateView {
    float* cue = nullptr;
    std::int32_t* timer = nullptr;
    float* active = nullptr;
    float* intensity = nullptr;
    std::int32_t* age = nullptr;
    std::int64_t* event_id = nullptr;
    std::int8_t* prior_cue = nullptr;
    std::size_t cell_count = 0;
};

struct ConstStateView {
    const float* cue = nullptr;
    const std::int32_t* timer = nullptr;
    const float* active = nullptr;
    const float* intensity = nullptr;
    const std::int32_t* age = nullptr;
    const std::int64_t* event_id = nullptr;
    const std::int8_t* prior_cue = nullptr;
    std::size_t cell_count = 0;
};

struct ScheduleState {
    // Live WDL-1 consumes no mutable RNG draws. Scheduling and observation
    // dropout are pure functions of this seed and their coordinates.
    std::int64_t base_seed = 0;
    std::int64_t next_abs_step = 0;
};

struct EventCoordinate {
    std::int32_t y = 0;
    std::int32_t x = 0;
    std::int64_t event_id = 0;
};

struct TickMetrics {
    std::int64_t cue_cell_count = 0;
    std::int64_t active_cell_count = 0;
    std::int64_t event_count = 0;
    std::int64_t cue_without_active_count = 0;
    std::int64_t active_without_prior_cue_count = 0;
    double cue_to_active_delay_sum = 0.0;
    std::int64_t cue_to_active_delay_count = 0;
};

struct ScheduledEventRow {
    std::int64_t event_id = 0;
    std::int64_t abs_step = 0;
    std::int32_t center_y = 0;
    std::int32_t center_x = 0;
    std::int64_t scheduled_cell_count = 0;
    float intensity = 0.0F;
};

struct AgentStateView {
    std::uint8_t* alive = nullptr;
    std::int32_t* pos_y = nullptr;
    std::int32_t* pos_x = nullptr;
    float* health = nullptr;
    // Optional per-agent multiplier supplied by shared refuge/path logic.
    const float* damage_scale = nullptr;
    std::size_t agent_count = 0;
};

struct DamageMetrics {
    std::int64_t active_exposure_count = 0;
    std::int64_t agents_damaged = 0;
    std::int64_t deaths = 0;
    double damage_total = 0.0;
    float damage_max = 0.0F;
};

struct Descriptor {
    const char* name;
    const char* dtype;
    std::size_t element_bytes;
};

extern const Descriptor kStateDescriptors[kStateDescriptorCount];
extern const char* const kObservationChannels[kObservationChannelCount];
extern const char* const kScheduledEventFields[kScheduledEventFieldCount];
extern const char* const kTickMetricFields[kTickMetricFieldCount];
extern const char* const kGenerationSummaryFields[kGenerationSummaryFieldCount];

Status validate_config(const Config& config) noexcept;
Status validate_state(const Config& config, ConstStateView state) noexcept;
Status initialize_state(const Config& config, StateView state) noexcept;

Phase phase_at(const Config& config, ConstStateView state, std::size_t cell) noexcept;
EventCoordinate stable_event_coordinate(
    std::int64_t base_seed,
    std::int64_t event_index,
    std::int32_t height,
    std::int32_t width
) noexcept;
float cue_observation_value(
    float cue,
    float dropout_rate,
    std::int64_t base_seed,
    std::int32_t y,
    std::int32_t x
) noexcept;

// Validation and cancellation happen before the first authoritative write.
Status advance_tick(
    const Config& config,
    StateView state,
    ScheduleState* schedule,
    std::int64_t abs_step,
    bool cancel_requested,
    TickMetrics* metrics,
    ScheduledEventRow* event = nullptr
) noexcept;

// Damage is likewise all-or-nothing for malformed input or cancellation.
// Grid occupancy removal and lifecycle event publication remain shared-engine
// responsibilities after this helper returns the death count.
Status apply_delayed_damage(
    const Config& config,
    ConstStateView state,
    AgentStateView agents,
    bool cancel_requested,
    DamageMetrics* metrics
) noexcept;

std::size_t checkpoint_size_bytes(const Config& config) noexcept;
Status write_checkpoint(
    const Config& config,
    ConstStateView state,
    const ScheduleState& schedule,
    std::uint8_t* output,
    std::size_t output_capacity,
    std::size_t* bytes_written
) noexcept;
Status restore_checkpoint(
    const Config& config,
    const std::uint8_t* input,
    std::size_t input_size,
    StateView state,
    ScheduleState* schedule
) noexcept;

}  // namespace eiel::wdl::native_v1
