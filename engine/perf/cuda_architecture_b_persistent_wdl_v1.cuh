#ifndef EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_WDL_V1_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_WDL_V1_CUH_

#include "cuda_architecture_b_sha256_v1.cuh"

#include <cstdint>
#include <math.h>

#if !defined(__CUDACC__)
#error "cuda_architecture_b_persistent_wdl_v1.cuh must be compiled by nvcc"
#endif

namespace eiel::cuda_architecture_b::persistent_wdl_v1 {

inline constexpr std::int32_t kSchemaVersion = 1;

enum class Status : std::int32_t {
    kOk = 0,
    kCancelled = 1,
    kInvalidConfig = 2,
    kInvalidState = 3,
};

struct Config {
    std::int32_t schema_version;
    std::int32_t height;
    std::int32_t width;
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

struct StateView {
    float* cue;
    std::int32_t* timer;
    float* active;
    float* intensity;
    std::int32_t* age;
    std::int64_t* event_id;
    std::int8_t* prior_cue;
    std::int64_t cell_count;
};

struct ConstStateView {
    const float* cue;
    const std::int32_t* timer;
    const float* active;
    const float* intensity;
    const std::int32_t* age;
    const std::int64_t* event_id;
    const std::int8_t* prior_cue;
    std::int64_t cell_count;
};

struct ScheduleState {
    std::int64_t base_seed;
    std::int64_t next_abs_step;
};

struct EventCoordinate {
    std::int32_t y;
    std::int32_t x;
    std::int64_t event_id;
};

struct TickMetrics {
    std::int64_t cue_cell_count;
    std::int64_t active_cell_count;
    std::int64_t event_count;
    std::int64_t cue_without_active_count;
    std::int64_t active_without_prior_cue_count;
    double cue_to_active_delay_sum;
    std::int64_t cue_to_active_delay_count;
};

struct ScheduledEventRow {
    std::int64_t event_id;
    std::int64_t abs_step;
    std::int32_t center_y;
    std::int32_t center_x;
    std::int64_t scheduled_cell_count;
    float intensity;
};

struct DamageResult {
    bool exposed;
    bool damaged;
    bool lethal;
    float damage;
};

namespace detail {

__device__ __forceinline__ bool valid_bool(const std::uint8_t value) {
    return value <= 1U;
}

__device__ __forceinline__ std::int32_t clamp_int32(
    const std::int32_t value,
    const std::int32_t lower,
    const std::int32_t upper
) {
    return value < lower ? lower : (value > upper ? upper : value);
}

__device__ __forceinline__ float clamp01(const float value) {
    return value < 0.0F ? 0.0F : (value > 1.0F ? 1.0F : value);
}

__device__ __forceinline__ int append_unsigned_decimal(
    char* const output,
    int cursor,
    std::uint64_t value
) {
    char reversed[20];
    int count = 0;
    do {
        reversed[count++] = static_cast<char>('0' + value % 10U);
        value /= 10U;
    } while (value != 0U);
    while (count > 0) {
        output[cursor++] = reversed[--count];
    }
    return cursor;
}

__device__ __forceinline__ int append_signed_decimal(
    char* const output,
    int cursor,
    const std::int64_t value
) {
    std::uint64_t magnitude = static_cast<std::uint64_t>(value);
    if (value < 0) {
        output[cursor++] = '-';
        magnitude = static_cast<std::uint64_t>(-(value + 1)) + 1U;
    }
    return append_unsigned_decimal(output, cursor, magnitude);
}

__device__ __forceinline__ std::uint64_t load_big_endian_u64(
    const std::uint8_t* const bytes
) {
    std::uint64_t value = 0;
#pragma unroll
    for (int index = 0; index < 8; ++index) {
        value = (value << 8U) | bytes[index];
    }
    return value;
}

}  // namespace detail

__device__ __forceinline__ bool config_valid(const Config& config) {
    return config.schema_version == kSchemaVersion
        && config.height > 0
        && config.width > 0
        && config.delay_steps >= 1
        && config.cue_duration_steps >= 1
        && config.cue_duration_steps <= config.delay_steps
        && config.active_duration_steps >= 1
        && static_cast<std::int64_t>(config.event_period_steps)
            >= static_cast<std::int64_t>(config.delay_steps)
                + static_cast<std::int64_t>(config.active_duration_steps) + 1
        && config.radius >= 0
        && isfinite(config.intensity)
        && config.intensity >= 0.0F
        && config.intensity <= 1.0F
        && isfinite(config.observation_dropout_rate)
        && config.observation_dropout_rate >= 0.0F
        && config.observation_dropout_rate <= 1.0F
        && detail::valid_bool(config.enabled)
        && detail::valid_bool(config.cue_enabled)
        && detail::valid_bool(config.observation_enabled)
        && detail::valid_bool(config.damage_enabled)
        && detail::valid_bool(config.deterministic_schedule_enabled);
}

__device__ __forceinline__ ConstStateView as_const(const StateView state) {
    return {
        state.cue,
        state.timer,
        state.active,
        state.intensity,
        state.age,
        state.event_id,
        state.prior_cue,
        state.cell_count,
    };
}

__device__ __forceinline__ bool state_storage_valid(
    const Config& config,
    const ConstStateView state
) {
    return config_valid(config)
        && state.cell_count
            == static_cast<std::int64_t>(config.height) * config.width
        && state.cue != nullptr
        && state.timer != nullptr
        && state.active != nullptr
        && state.intensity != nullptr
        && state.age != nullptr
        && state.event_id != nullptr
        && state.prior_cue != nullptr;
}

__device__ __forceinline__ bool state_values_valid(
    const Config& config,
    const ConstStateView state
) {
    if (!state_storage_valid(config, state)) {
        return false;
    }
    for (std::int64_t index = 0; index < state.cell_count; ++index) {
        if (
            !isfinite(state.cue[index])
            || state.cue[index] < 0.0F
            || state.cue[index] > 1.0F
            || !isfinite(state.active[index])
            || state.active[index] < 0.0F
            || state.active[index] > 1.0F
            || !isfinite(state.intensity[index])
            || state.intensity[index] < 0.0F
            || state.intensity[index] > 1.0F
            || state.timer[index] < -1
            || state.age[index] < 0
            || state.event_id[index] < 0
            || (
                state.prior_cue[index] != 0
                && state.prior_cue[index] != 1
            )
        ) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ EventCoordinate stable_event_coordinate_exact(
    const std::int64_t base_seed,
    const std::int64_t event_index,
    const std::int32_t height,
    const std::int32_t width
) {
    char material[160];
    int cursor = 0;
    material[cursor++] = 'w';
    material[cursor++] = 'd';
    material[cursor++] = 'l';
    material[cursor++] = '1';
    material[cursor++] = ':';
    cursor = detail::append_signed_decimal(material, cursor, base_seed);
    material[cursor++] = ':';
    cursor = detail::append_signed_decimal(material, cursor, event_index);
    material[cursor++] = ':';
    cursor = detail::append_signed_decimal(material, cursor, height);
    material[cursor++] = ':';
    cursor = detail::append_signed_decimal(material, cursor, width);

    std::uint8_t digest[sha256_v1::kDigestBytesV1];
    sha256_v1::sha256_digest_v1(
        reinterpret_cast<const std::uint8_t*>(material),
        static_cast<std::uint64_t>(cursor),
        digest
    );
    const std::uint64_t y_value = detail::load_big_endian_u64(digest);
    const std::uint64_t x_value = detail::load_big_endian_u64(digest + 8);
    const std::uint64_t event_value = detail::load_big_endian_u64(digest + 16);
    return {
        static_cast<std::int32_t>(
            y_value % static_cast<std::uint64_t>(height > 0 ? height : 1)
        ),
        static_cast<std::int32_t>(
            x_value % static_cast<std::uint64_t>(width > 0 ? width : 1)
        ),
        static_cast<std::int64_t>(event_value & UINT64_C(0x7FFFFFFFFFFFFFFF)),
    };
}

__device__ __forceinline__ float cue_observation_value_exact(
    float cue,
    float dropout_rate,
    const std::int64_t base_seed,
    const std::int32_t y,
    const std::int32_t x
) {
    cue = isfinite(cue) ? detail::clamp01(cue) : 0.0F;
    dropout_rate = isfinite(dropout_rate)
        ? detail::clamp01(dropout_rate)
        : 0.0F;
    if (dropout_rate <= 0.0F) {
        return cue;
    }
    if (dropout_rate >= 1.0F) {
        return 0.0F;
    }
    const std::uint32_t mixed = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(static_cast<std::int64_t>(y) * 1103515245LL)
        + static_cast<std::uint64_t>(static_cast<std::int64_t>(x) * 12345LL)
        + static_cast<std::uint32_t>(base_seed)
        + UINT64_C(0x9E3779B9)
    );
    const double unit = static_cast<double>(mixed) / 4294967296.0;
    return unit >= static_cast<double>(dropout_rate) ? cue : 0.0F;
}

// One CUDA block calls this from thread zero before Patch update and policy
// service. Validation completes before the first authoritative state write.
__device__ __forceinline__ Status advance_tick_exact(
    const Config& config,
    const StateView state,
    ScheduleState* const schedule,
    const std::int64_t absolute_step,
    const bool cancel_requested,
    TickMetrics* const metrics,
    ScheduledEventRow* const scheduled_event
) {
    if (schedule == nullptr || metrics == nullptr) {
        return Status::kInvalidState;
    }
    if (cancel_requested) {
        return Status::kCancelled;
    }
    if (
        schedule->next_abs_step != absolute_step
        || absolute_step < 0
        || absolute_step == INT64_MAX
    ) {
        return Status::kInvalidState;
    }
    if (!config_valid(config)) {
        return Status::kInvalidConfig;
    }
    if (!state_values_valid(config, as_const(state))) {
        return Status::kInvalidState;
    }

    TickMetrics pending{};
    ScheduledEventRow pending_event{};
    if (!config.enabled) {
        schedule->next_abs_step = absolute_step + 1;
        *metrics = pending;
        if (scheduled_event != nullptr) {
            *scheduled_event = pending_event;
        }
        return Status::kOk;
    }

    for (std::int64_t index = 0; index < state.cell_count; ++index) {
        const bool occupied = state.timer[index] >= 0;
        if (occupied) {
            ++state.age[index];
            --state.timer[index];
        }
        state.cue[index] = 0.0F;
        state.active[index] = 0.0F;
        const bool cue_phase = occupied
            && state.age[index] < config.cue_duration_steps;
        const bool hidden_phase = occupied
            && state.age[index] >= config.cue_duration_steps
            && state.age[index] < config.delay_steps;
        const bool active_phase = occupied
            && state.age[index] >= config.delay_steps
            && state.age[index]
                < config.delay_steps + config.active_duration_steps;
        const bool done = occupied
            && state.age[index]
                >= config.delay_steps + config.active_duration_steps;
        if (cue_phase && config.cue_enabled != 0) {
            state.cue[index] = 1.0F;
        }
        if (cue_phase || hidden_phase || active_phase) {
            state.prior_cue[index] = 1;
        }
        if (active_phase) {
            state.active[index] = 1.0F;
            state.intensity[index] = config.intensity;
            pending.cue_to_active_delay_sum += config.delay_steps;
            ++pending.cue_to_active_delay_count;
        }
        if (!active_phase && !cue_phase && !hidden_phase) {
            state.intensity[index] = 0.0F;
        }
        if (done) {
            state.cue[index] = 0.0F;
            state.active[index] = 0.0F;
            state.intensity[index] = 0.0F;
            state.age[index] = 0;
            state.timer[index] = -1;
            state.event_id[index] = 0;
            state.prior_cue[index] = 0;
        }
    }

    if (
        config.deterministic_schedule_enabled
        && absolute_step % config.event_period_steps == 0
    ) {
        const EventCoordinate coordinate = stable_event_coordinate_exact(
            schedule->base_seed,
            absolute_step / config.event_period_steps,
            config.height,
            config.width
        );
        pending_event.event_id = coordinate.event_id;
        pending_event.abs_step = absolute_step;
        pending_event.center_y = coordinate.y;
        pending_event.center_x = coordinate.x;
        pending_event.intensity = config.intensity;
        const std::int32_t y0 = detail::clamp_int32(
            coordinate.y - config.radius, 0, config.height
        );
        const std::int32_t y1 = detail::clamp_int32(
            coordinate.y + config.radius + 1, 0, config.height
        );
        const std::int32_t x0 = detail::clamp_int32(
            coordinate.x - config.radius, 0, config.width
        );
        const std::int32_t x1 = detail::clamp_int32(
            coordinate.x + config.radius + 1, 0, config.width
        );
        for (std::int32_t y = y0; y < y1; ++y) {
            for (std::int32_t x = x0; x < x1; ++x) {
                const std::int64_t index =
                    static_cast<std::int64_t>(y) * config.width + x;
                if (state.timer[index] >= 0) {
                    continue;
                }
                state.cue[index] = config.cue_enabled != 0 ? 1.0F : 0.0F;
                state.timer[index] =
                    config.delay_steps + config.active_duration_steps;
                state.active[index] = 0.0F;
                state.intensity[index] = config.intensity;
                state.age[index] = 0;
                state.event_id[index] = coordinate.event_id;
                state.prior_cue[index] = 1;
                ++pending.event_count;
                ++pending_event.scheduled_cell_count;
            }
        }
        if (pending_event.scheduled_cell_count == 0) {
            pending_event = {};
        }
    }

    for (std::int64_t index = 0; index < state.cell_count; ++index) {
        pending.cue_cell_count += state.cue[index] > 0.0F;
        pending.active_cell_count += state.active[index] > 0.0F;
        pending.cue_without_active_count +=
            state.prior_cue[index] > 0 && state.active[index] <= 0.0F;
        pending.active_without_prior_cue_count +=
            state.active[index] > 0.0F && state.prior_cue[index] <= 0;
    }
    schedule->next_abs_step = absolute_step + 1;
    *metrics = pending;
    if (scheduled_event != nullptr) {
        *scheduled_event = pending_event;
    }
    return Status::kOk;
}

// The caller visits live slots in ascending agent-ID order. Lethal rows are
// returned to the shared lifecycle owner, which publishes the death and frees
// the slot exactly once after this helper settles health.
__device__ __forceinline__ DamageResult apply_damage_to_slot_exact(
    const Config& config,
    const ConstStateView state,
    const bool alive,
    const std::int32_t y,
    const std::int32_t x,
    float* const health,
    float* const last_health_delta,
    const float damage_scale = 1.0F
) {
    DamageResult result{};
    if (
        !config.enabled
        || !config.damage_enabled
        || !alive
        || health == nullptr
        || y < 0
        || y >= config.height
        || x < 0
        || x >= config.width
        || !isfinite(*health)
        || *health < 0.0F
        || !isfinite(damage_scale)
        || damage_scale < 0.0F
    ) {
        return result;
    }
    const std::int64_t cell = static_cast<std::int64_t>(y) * config.width + x;
    if (state.active[cell] <= 0.0F) {
        return result;
    }
    result.exposed = true;
    result.damage = fmaxf(0.0F, state.intensity[cell] * damage_scale);
    if (result.damage <= 0.0F) {
        return result;
    }
    result.damaged = true;
    *health = fmaxf(0.0F, *health - result.damage);
    if (last_health_delta != nullptr) {
        *last_health_delta -= result.damage;
    }
    result.lethal = *health <= 0.0F;
    return result;
}

}  // namespace eiel::cuda_architecture_b::persistent_wdl_v1

#endif
