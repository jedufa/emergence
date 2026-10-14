#include "wdl_cpu_oracle_c_api.h"

#include "wdl_cpu_oracle.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>

namespace oracle = eiel::wdl::cpu_oracle_v1;
namespace native = eiel::wdl::native_v1;

static_assert(sizeof(float) == 4 && sizeof(double) == 8, "WDL C ABI requires IEEE-width floats");
static_assert(std::is_standard_layout_v<eiel_wdl_cpu_oracle_config_v1>);
static_assert(std::is_standard_layout_v<eiel_wdl_tick_receipt_v1>);
static_assert(sizeof(eiel_wdl_cpu_oracle_config_v1) == 72);
static_assert(offsetof(eiel_wdl_cpu_oracle_config_v1, schema_version) == 0);
static_assert(offsetof(eiel_wdl_cpu_oracle_config_v1, intensity) == 32);
static_assert(offsetof(eiel_wdl_cpu_oracle_config_v1, enabled) == 40);
static_assert(offsetof(eiel_wdl_cpu_oracle_config_v1, deterministic_schedule_enabled) == 44);
static_assert(offsetof(eiel_wdl_cpu_oracle_config_v1, view_radius) == 48);
static_assert(offsetof(eiel_wdl_cpu_oracle_config_v1, max_scheduled_events) == 56);
static_assert(offsetof(eiel_wdl_cpu_oracle_config_v1, max_damage_events) == 64);
static_assert(sizeof(eiel_wdl_tick_metrics_v1) == 56);
static_assert(offsetof(eiel_wdl_tick_metrics_v1, cue_to_active_delay_sum) == 40);
static_assert(offsetof(eiel_wdl_tick_metrics_v1, cue_to_active_delay_count) == 48);
static_assert(sizeof(eiel_wdl_damage_metrics_v1) == 40);
static_assert(offsetof(eiel_wdl_damage_metrics_v1, damage_total) == 24);
static_assert(offsetof(eiel_wdl_damage_metrics_v1, damage_max) == 32);
static_assert(sizeof(eiel_wdl_scheduled_event_v1) == 40);
static_assert(offsetof(eiel_wdl_scheduled_event_v1, scheduled_cell_count) == 24);
static_assert(offsetof(eiel_wdl_scheduled_event_v1, intensity) == 32);
static_assert(sizeof(eiel_wdl_damage_event_v1) == 48);
static_assert(offsetof(eiel_wdl_damage_event_v1, raw_intensity) == 20);
static_assert(offsetof(eiel_wdl_damage_event_v1, died) == 40);
static_assert(sizeof(eiel_wdl_tick_receipt_v1) == 160);
static_assert(offsetof(eiel_wdl_tick_receipt_v1, tick) == 8);
static_assert(offsetof(eiel_wdl_tick_receipt_v1, damage) == 64);
static_assert(offsetof(eiel_wdl_tick_receipt_v1, scheduled_event) == 104);
static_assert(offsetof(eiel_wdl_tick_receipt_v1, observation_rows) == 144);
static_assert(offsetof(eiel_wdl_tick_receipt_v1, observation_width) == 152);
static_assert(sizeof(eiel_wdl_generation_summary_v1) == 104);
static_assert(offsetof(eiel_wdl_generation_summary_v1, damage_max) == 48);
static_assert(offsetof(eiel_wdl_generation_summary_v1, agents_damaged) == 56);
static_assert(offsetof(eiel_wdl_generation_summary_v1, tick_count) == 96);

struct eiel_wdl_cpu_oracle_handle_v1 {
    oracle::CpuOracle oracle;
};

namespace {

std::int32_t code(native::Status status) noexcept {
    return static_cast<std::int32_t>(status);
}

oracle::OracleConfig convert_config(const eiel_wdl_cpu_oracle_config_v1& input) {
    oracle::OracleConfig output{};
    auto& c = output.wdl;
    c.schema_version = input.schema_version;
    c.height = input.height; c.width = input.width; c.delay_steps = input.delay_steps;
    c.cue_duration_steps = input.cue_duration_steps; c.active_duration_steps = input.active_duration_steps;
    c.event_period_steps = input.event_period_steps; c.radius = input.radius;
    c.intensity = input.intensity; c.observation_dropout_rate = input.observation_dropout_rate;
    c.enabled = input.enabled; c.cue_enabled = input.cue_enabled;
    c.observation_enabled = input.observation_enabled; c.damage_enabled = input.damage_enabled;
    c.deterministic_schedule_enabled = input.deterministic_schedule_enabled;
    output.view_radius = input.view_radius;
    output.max_scheduled_events = static_cast<std::size_t>(input.max_scheduled_events);
    output.max_damage_events = static_cast<std::size_t>(input.max_damage_events);
    return output;
}

eiel_wdl_tick_metrics_v1 convert_tick(const native::TickMetrics& value) noexcept {
    return {value.cue_cell_count, value.active_cell_count, value.event_count,
            value.cue_without_active_count, value.active_without_prior_cue_count,
            value.cue_to_active_delay_sum, value.cue_to_active_delay_count};
}

eiel_wdl_damage_metrics_v1 convert_damage(const native::DamageMetrics& value) noexcept {
    return {value.active_exposure_count, value.agents_damaged, value.deaths,
            value.damage_total, value.damage_max};
}

eiel_wdl_scheduled_event_v1 convert_event(const native::ScheduledEventRow& value) noexcept {
    return {value.event_id, value.abs_step, value.center_y, value.center_x,
            value.scheduled_cell_count, value.intensity};
}

eiel_wdl_damage_event_v1 convert_damage_event(const oracle::DamageEventRow& value) noexcept {
    return {value.abs_step, value.agent_id, value.y, value.x, value.raw_intensity,
            value.damage_scale, value.effective_damage, value.health_before,
            value.health_after, value.died};
}

std::int32_t invalid_state() noexcept { return code(native::Status::kInvalidState); }
std::int32_t insufficient() noexcept { return code(native::Status::kInsufficientCapacity); }

template <typename T>
std::int32_t copy_vector(const std::vector<T>& source, T* output, std::uint64_t capacity) noexcept {
    if (source.size() > capacity || (!source.empty() && output == nullptr)) return insufficient();
    if (!source.empty()) std::memcpy(output, source.data(), source.size() * sizeof(T));
    return code(native::Status::kOk);
}

}  // namespace

extern "C" std::int32_t eiel_wdl_cpu_oracle_abi_version_v1(void) {
    return EIEL_WDL_CPU_ORACLE_ABI_VERSION_V1;
}

extern "C" eiel_wdl_cpu_oracle_handle_v1* eiel_wdl_cpu_oracle_create_v1(
    const eiel_wdl_cpu_oracle_config_v1* config,
    std::int64_t base_seed,
    std::uint64_t agent_count,
    std::int32_t* status_out
) {
    if (status_out != nullptr) *status_out = invalid_state();
    if (config == nullptr || status_out == nullptr
        || agent_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())
        || config->max_scheduled_events
            > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())
        || config->max_damage_events
            > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) return nullptr;
    try {
        auto* handle = new eiel_wdl_cpu_oracle_handle_v1();
        const native::Status status = handle->oracle.initialize(
            convert_config(*config), base_seed, static_cast<std::size_t>(agent_count));
        *status_out = code(status);
        if (status != native::Status::kOk) { delete handle; return nullptr; }
        return handle;
    } catch (...) {
        *status_out = insufficient();
        return nullptr;
    }
}

extern "C" void eiel_wdl_cpu_oracle_destroy_v1(eiel_wdl_cpu_oracle_handle_v1* handle) {
    delete handle;
}

extern "C" std::int32_t eiel_wdl_cpu_oracle_set_agents_v1(
    eiel_wdl_cpu_oracle_handle_v1* handle,
    const std::uint8_t* alive,
    const std::int32_t* pos_y,
    const std::int32_t* pos_x,
    const float* health,
    const float* damage_scale,
    std::uint64_t agent_count
) {
    if (handle == nullptr || agent_count > std::numeric_limits<std::size_t>::max()) return invalid_state();
    try {
        return code(handle->oracle.set_agents(alive, pos_y, pos_x, health, damage_scale,
            static_cast<std::size_t>(agent_count)));
    } catch (...) { return insufficient(); }
}

extern "C" std::int32_t eiel_wdl_cpu_oracle_step_v1(
    eiel_wdl_cpu_oracle_handle_v1* handle,
    std::int64_t abs_step,
    const std::int32_t* post_action_y,
    const std::int32_t* post_action_x,
    std::uint64_t position_count,
    std::uint8_t cancel_requested,
    eiel_wdl_tick_receipt_v1* receipt
) {
    if (handle == nullptr || receipt == nullptr || cancel_requested > 1
        || position_count > std::numeric_limits<std::size_t>::max()) return invalid_state();
    try {
        oracle::TickReceipt native_receipt{};
        const native::Status status = handle->oracle.step(
            abs_step, post_action_y, post_action_x, static_cast<std::size_t>(position_count),
            cancel_requested != 0, &native_receipt);
        if (status != native::Status::kOk) return code(status);
        receipt->abs_step = native_receipt.abs_step;
        receipt->tick = convert_tick(native_receipt.tick);
        receipt->damage = convert_damage(native_receipt.damage);
        receipt->scheduled_event = convert_event(native_receipt.scheduled_event);
        receipt->observation_rows = native_receipt.observation_rows;
        receipt->observation_width = native_receipt.observation_width;
        return code(status);
    } catch (...) { return insufficient(); }
}

extern "C" std::int32_t eiel_wdl_cpu_oracle_begin_generation_v1(
    eiel_wdl_cpu_oracle_handle_v1* handle,
    std::uint8_t clear_event_journals
) {
    if (handle == nullptr || clear_event_journals > 1) return invalid_state();
    return code(handle->oracle.begin_generation(clear_event_journals != 0));
}

extern "C" std::uint64_t eiel_wdl_cpu_oracle_observation_count_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle
) {
    return handle == nullptr ? 0 : handle->oracle.last_observation().size();
}

extern "C" std::uint64_t eiel_wdl_cpu_oracle_observation_row_count_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle
) {
    return handle == nullptr ? 0 : handle->oracle.last_observation_agent_ids().size();
}

extern "C" std::int32_t eiel_wdl_cpu_oracle_copy_observation_agent_ids_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle,
    std::int32_t* output,
    std::uint64_t capacity
) {
    if (handle == nullptr) return invalid_state();
    return copy_vector(handle->oracle.last_observation_agent_ids(), output, capacity);
}

extern "C" std::int32_t eiel_wdl_cpu_oracle_copy_observation_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle,
    float* output,
    std::uint64_t capacity
) {
    if (handle == nullptr) return invalid_state();
    return copy_vector(handle->oracle.last_observation(), output, capacity);
}

extern "C" std::uint64_t eiel_wdl_cpu_oracle_state_field_bytes_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle,
    std::int32_t field
) {
    if (handle == nullptr || field < EIEL_WDL_STATE_CUE_V1 || field > EIEL_WDL_STATE_PRIOR_CUE_V1) return 0;
    const std::size_t cells = handle->oracle.state_view().cell_count;
    static constexpr std::size_t widths[7] = {4, 4, 4, 4, 4, 8, 1};
    return cells * widths[field];
}

extern "C" std::int32_t eiel_wdl_cpu_oracle_copy_state_field_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle,
    std::int32_t field,
    void* output,
    std::uint64_t capacity_bytes
) {
    const std::uint64_t required = eiel_wdl_cpu_oracle_state_field_bytes_v1(handle, field);
    if (handle == nullptr || required == 0 || output == nullptr || capacity_bytes < required) return insufficient();
    const native::ConstStateView state = handle->oracle.state_view();
    const void* source = nullptr;
    switch (field) {
        case EIEL_WDL_STATE_CUE_V1: source = state.cue; break;
        case EIEL_WDL_STATE_TIMER_V1: source = state.timer; break;
        case EIEL_WDL_STATE_ACTIVE_V1: source = state.active; break;
        case EIEL_WDL_STATE_INTENSITY_V1: source = state.intensity; break;
        case EIEL_WDL_STATE_AGE_V1: source = state.age; break;
        case EIEL_WDL_STATE_EVENT_ID_V1: source = state.event_id; break;
        case EIEL_WDL_STATE_PRIOR_CUE_V1: source = state.prior_cue; break;
        default: return invalid_state();
    }
    std::memcpy(output, source, required);
    return code(native::Status::kOk);
}

extern "C" std::int32_t eiel_wdl_cpu_oracle_copy_agents_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle,
    std::uint8_t* alive,
    std::int32_t* pos_y,
    std::int32_t* pos_x,
    float* health,
    std::uint64_t capacity
) {
    if (handle == nullptr || capacity < handle->oracle.agent_count()
        || (handle->oracle.agent_count() > 0 && (alive == nullptr || pos_y == nullptr
            || pos_x == nullptr || health == nullptr))) return insufficient();
    const std::size_t count = handle->oracle.agent_count();
    if (count > 0) {
        std::memcpy(alive, handle->oracle.agent_alive().data(), count);
        std::memcpy(pos_y, handle->oracle.agent_pos_y().data(), count * sizeof(std::int32_t));
        std::memcpy(pos_x, handle->oracle.agent_pos_x().data(), count * sizeof(std::int32_t));
        std::memcpy(health, handle->oracle.agent_health().data(), count * sizeof(float));
    }
    return code(native::Status::kOk);
}

extern "C" std::int32_t eiel_wdl_cpu_oracle_generation_summary_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle,
    eiel_wdl_generation_summary_v1* output
) {
    if (handle == nullptr || output == nullptr) return invalid_state();
    const oracle::GenerationSummary value = handle->oracle.generation_summary();
    *output = {value.event_count, value.cue_cell_count_mean, value.active_cell_count_mean,
        value.agent_cue_exposure_count, value.agent_active_exposure_count, value.damage_total,
        value.damage_max, value.agents_damaged, value.cue_to_active_delay_mean,
        value.cue_without_active_count, value.active_without_prior_cue_count,
        value.enabled, value.tick_count};
    return code(native::Status::kOk);
}

extern "C" std::uint64_t eiel_wdl_cpu_oracle_scheduled_event_count_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle
) { return handle == nullptr ? 0 : handle->oracle.scheduled_events().size(); }

extern "C" std::int32_t eiel_wdl_cpu_oracle_copy_scheduled_events_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle,
    eiel_wdl_scheduled_event_v1* output,
    std::uint64_t capacity
) {
    if (handle == nullptr || capacity < handle->oracle.scheduled_events().size()
        || (!handle->oracle.scheduled_events().empty() && output == nullptr)) return insufficient();
    for (std::size_t i = 0; i < handle->oracle.scheduled_events().size(); ++i) {
        output[i] = convert_event(handle->oracle.scheduled_events()[i]);
    }
    return code(native::Status::kOk);
}

extern "C" std::uint64_t eiel_wdl_cpu_oracle_damage_event_count_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle
) { return handle == nullptr ? 0 : handle->oracle.damage_events().size(); }

extern "C" std::int32_t eiel_wdl_cpu_oracle_copy_damage_events_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle,
    eiel_wdl_damage_event_v1* output,
    std::uint64_t capacity
) {
    if (handle == nullptr || capacity < handle->oracle.damage_events().size()
        || (!handle->oracle.damage_events().empty() && output == nullptr)) return insufficient();
    for (std::size_t i = 0; i < handle->oracle.damage_events().size(); ++i) {
        output[i] = convert_damage_event(handle->oracle.damage_events()[i]);
    }
    return code(native::Status::kOk);
}

extern "C" std::uint64_t eiel_wdl_cpu_oracle_checkpoint_size_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle
) { return handle == nullptr ? 0 : handle->oracle.checkpoint_size_bytes(); }

extern "C" std::int32_t eiel_wdl_cpu_oracle_write_checkpoint_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle,
    std::uint8_t* output,
    std::uint64_t capacity,
    std::uint64_t* bytes_written
) {
    if (bytes_written != nullptr) *bytes_written = 0;
    if (handle == nullptr || bytes_written == nullptr || capacity > std::numeric_limits<std::size_t>::max()) {
        return invalid_state();
    }
    std::size_t written = 0;
    const native::Status status = handle->oracle.write_checkpoint(
        output, static_cast<std::size_t>(capacity), &written);
    if (status == native::Status::kOk) *bytes_written = written;
    return code(status);
}

extern "C" std::int32_t eiel_wdl_cpu_oracle_restore_checkpoint_v1(
    eiel_wdl_cpu_oracle_handle_v1* handle,
    const std::uint8_t* input,
    std::uint64_t size
) {
    if (handle == nullptr || size > std::numeric_limits<std::size_t>::max()) return invalid_state();
    return code(handle->oracle.restore_checkpoint(input, static_cast<std::size_t>(size)));
}
