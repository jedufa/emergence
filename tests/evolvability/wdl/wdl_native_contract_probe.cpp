#include "engine/native_runtime/wdl/wdl_native_contract.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace wdl = eiel::wdl::native_v1;

namespace {

struct OwnedState {
    explicit OwnedState(std::size_t count)
        : cue(count), timer(count), active(count), intensity(count), age(count),
          event_id(count), prior_cue(count) {}

    std::vector<float> cue;
    std::vector<std::int32_t> timer;
    std::vector<float> active;
    std::vector<float> intensity;
    std::vector<std::int32_t> age;
    std::vector<std::int64_t> event_id;
    std::vector<std::int8_t> prior_cue;

    wdl::StateView view() {
        return {cue.data(), timer.data(), active.data(), intensity.data(), age.data(),
                event_id.data(), prior_cue.data(), cue.size()};
    }

    wdl::ConstStateView const_view() const {
        return {cue.data(), timer.data(), active.data(), intensity.data(), age.data(),
                event_id.data(), prior_cue.data(), cue.size()};
    }

    bool operator==(const OwnedState& other) const {
        return cue == other.cue && timer == other.timer && active == other.active
            && intensity == other.intensity && age == other.age
            && event_id == other.event_id && prior_cue == other.prior_cue;
    }
};

wdl::Config config(std::int32_t active_duration = 3) {
    wdl::Config value{};
    value.height = 6;
    value.width = 6;
    value.delay_steps = 4;
    value.cue_duration_steps = 2;
    value.active_duration_steps = active_duration;
    value.event_period_steps = 12;
    value.radius = 0;
    value.intensity = 0.25F;
    value.observation_dropout_rate = 0.0F;
    value.enabled = 1;
    value.cue_enabled = 1;
    value.observation_enabled = 1;
    value.damage_enabled = 1;
    value.deterministic_schedule_enabled = 1;
    return value;
}

bool schedule_equal(const wdl::ScheduleState& left, const wdl::ScheduleState& right) {
    return left.base_seed == right.base_seed && left.next_abs_step == right.next_abs_step;
}

bool metric_sentinel(const wdl::TickMetrics& value) {
    return value.cue_cell_count == 1 && value.active_cell_count == 2
        && value.event_count == 3 && value.cue_without_active_count == 4
        && value.active_without_prior_cue_count == 5
        && value.cue_to_active_delay_sum == 6.0 && value.cue_to_active_delay_count == 7;
}

bool damage_metric_zero(const wdl::DamageMetrics& value) {
    return value.active_exposure_count == 0 && value.agents_damaged == 0
        && value.deaths == 0 && value.damage_total == 0.0 && value.damage_max == 0.0F;
}

bool damage_metric_sentinel(const wdl::DamageMetrics& value) {
    return value.active_exposure_count == 1 && value.agents_damaged == 2
        && value.deaths == 3 && value.damage_total == 4.0 && value.damage_max == 5.0F;
}

bool advance(const wdl::Config& cfg, OwnedState& state, wdl::ScheduleState& schedule,
             std::int64_t step, wdl::TickMetrics* metrics = nullptr) {
    wdl::TickMetrics local{};
    return wdl::advance_tick(cfg, state.view(), &schedule, step, false,
                             metrics == nullptr ? &local : metrics) == wdl::Status::kOk;
}

bool initialize(const wdl::Config& cfg, OwnedState& state) {
    return wdl::initialize_state(cfg, state.view()) == wdl::Status::kOk;
}

bool phase_schedule_test() {
    const auto cfg = config();
    OwnedState state(36);
    wdl::ScheduleState schedule{101, 0};
    if (!initialize(cfg, state)) return false;
    const auto event = wdl::stable_event_coordinate(101, 0, cfg.height, cfg.width);
    const std::size_t cell = static_cast<std::size_t>(event.y) * cfg.width + event.x;
    const std::array<wdl::Phase, 8> expected{
        wdl::Phase::kCue, wdl::Phase::kCue,
        wdl::Phase::kHiddenGap, wdl::Phase::kHiddenGap,
        wdl::Phase::kActiveHazard, wdl::Phase::kActiveHazard,
        wdl::Phase::kActiveHazard, wdl::Phase::kBaseline,
    };
    for (std::int64_t step = 0; step < static_cast<std::int64_t>(expected.size()); ++step) {
        wdl::TickMetrics metrics{};
        if (!advance(cfg, state, schedule, step, &metrics)) return false;
        if (wdl::phase_at(cfg, state.const_view(), cell) != expected[static_cast<std::size_t>(step)]) {
            std::cerr << "phase mismatch step=" << step << " actual="
                      << static_cast<int>(wdl::phase_at(cfg, state.const_view(), cell))
                      << " expected=" << static_cast<int>(expected[static_cast<std::size_t>(step)]) << '\n';
            return false;
        }
        const bool cue = step < 2;
        const bool hidden = step >= 2 && step < 4;
        const bool active = step >= 4 && step < 7;
        if ((state.cue[cell] > 0.0F) != cue || (state.active[cell] > 0.0F) != active) {
            std::cerr << "surface mismatch step=" << step << " cue=" << state.cue[cell]
                      << " active=" << state.active[cell] << '\n';
            return false;
        }
        if (hidden && (state.cue[cell] != 0.0F || state.active[cell] != 0.0F)) return false;
        if ((step == 0) != (metrics.event_count == 1)) {
            std::cerr << "event mismatch step=" << step << " count=" << metrics.event_count << '\n';
            return false;
        }
    }
    return schedule.next_abs_step == 8;
}

bool configured_active_duration_test() {
    const auto cfg = config(2);
    OwnedState state(36);
    wdl::ScheduleState schedule{202, 0};
    if (!initialize(cfg, state)) return false;
    const auto event = wdl::stable_event_coordinate(202, 0, cfg.height, cfg.width);
    const std::size_t cell = static_cast<std::size_t>(event.y) * cfg.width + event.x;
    std::int32_t active_ticks = 0;
    for (std::int64_t step = 0; step < 7; ++step) {
        if (!advance(cfg, state, schedule, step)) return false;
        active_ticks += state.active[cell] > 0.0F ? 1 : 0;
    }
    return active_ticks == 2 && wdl::phase_at(cfg, state.const_view(), cell) == wdl::Phase::kBaseline;
}

bool same_seed_and_dropout_test() {
    const auto cfg = config();
    OwnedState first(36), second(36), different(36);
    wdl::ScheduleState first_schedule{101, 0}, second_schedule{101, 0}, different_schedule{202, 0};
    if (!initialize(cfg, first) || !initialize(cfg, second) || !initialize(cfg, different)) return false;
    bool different_seed_diverged = false;
    for (std::int64_t step = 0; step < 8; ++step) {
        if (!advance(cfg, first, first_schedule, step) || !advance(cfg, second, second_schedule, step)
            || !advance(cfg, different, different_schedule, step)) return false;
        different_seed_diverged = different_seed_diverged || !(first == different);
    }
    if (!(first == second) || !schedule_equal(first_schedule, second_schedule) || !different_seed_diverged) return false;
    const float cue = 0.75F;
    if (wdl::cue_observation_value(cue, 0.0F, 101, 2, 3) != cue) return false;
    if (wdl::cue_observation_value(cue, 1.0F, 101, 2, 3) != 0.0F) return false;
    const float sampled = wdl::cue_observation_value(cue, 0.4F, 101, 2, 3);
    return sampled == wdl::cue_observation_value(cue, 0.4F, 101, 2, 3)
        && (sampled == 0.0F || sampled == cue);
}

bool checkpoint_continuation_at(std::int64_t checkpoint_step) {
    const auto cfg = config();
    OwnedState uninterrupted(36), prefix(36), restored(36);
    wdl::ScheduleState uninterrupted_schedule{7070, 0}, prefix_schedule{7070, 0}, restored_schedule{-1, -1};
    if (!initialize(cfg, uninterrupted) || !initialize(cfg, prefix) || !initialize(cfg, restored)) return false;
    for (std::int64_t step = 0; step < 8; ++step) {
        if (!advance(cfg, uninterrupted, uninterrupted_schedule, step)) return false;
    }
    for (std::int64_t step = 0; step <= checkpoint_step; ++step) {
        if (!advance(cfg, prefix, prefix_schedule, step)) return false;
    }
    std::vector<std::uint8_t> bytes(wdl::checkpoint_size_bytes(cfg));
    std::size_t written = 0;
    if (wdl::write_checkpoint(cfg, prefix.const_view(), prefix_schedule, bytes.data(), bytes.size(), &written)
        != wdl::Status::kOk || written != bytes.size()) return false;
    if (wdl::restore_checkpoint(cfg, bytes.data(), bytes.size(), restored.view(), &restored_schedule)
        != wdl::Status::kOk) return false;
    if (!(prefix == restored) || !schedule_equal(prefix_schedule, restored_schedule)) return false;
    for (std::int64_t step = checkpoint_step + 1; step < 8; ++step) {
        if (!advance(cfg, restored, restored_schedule, step)) return false;
    }
    return uninterrupted == restored && schedule_equal(uninterrupted_schedule, restored_schedule);
}

bool checkpoint_and_schedule_test() {
    // Cue start/end, hidden start/end, all configured active ticks, and the
    // first post-active state are exact continuation boundaries.
    for (std::int64_t step = 0; step <= 7; ++step) {
        if (!checkpoint_continuation_at(step)) return false;
    }
    return true;
}

bool damage_test() {
    const auto cfg = config();
    OwnedState state(36);
    wdl::ScheduleState schedule{101, 0};
    if (!initialize(cfg, state)) return false;
    const auto event = wdl::stable_event_coordinate(101, 0, cfg.height, cfg.width);
    std::array<std::uint8_t, 2> alive{1, 1};
    std::array<std::int32_t, 2> pos_y{event.y, 0};
    std::array<std::int32_t, 2> pos_x{event.x, 0};
    std::array<float, 2> health{1.0F, 1.0F};
    std::array<float, 2> scale{1.0F, 0.5F};
    wdl::AgentStateView agents{alive.data(), pos_y.data(), pos_x.data(), health.data(), scale.data(), 2};
    for (std::int64_t step = 0; step < 4; ++step) {
        if (!advance(cfg, state, schedule, step)) return false;
        wdl::DamageMetrics metrics{};
        if (wdl::apply_delayed_damage(cfg, state.const_view(), agents, false, &metrics) != wdl::Status::kOk
            || health[0] != 1.0F || !damage_metric_zero(metrics)) return false;
    }
    if (!advance(cfg, state, schedule, 4)) return false;
    wdl::DamageMetrics metrics{};
    if (wdl::apply_delayed_damage(cfg, state.const_view(), agents, false, &metrics) != wdl::Status::kOk) return false;
    return health[0] == 0.75F && health[1] == 1.0F && metrics.active_exposure_count == 1
        && metrics.agents_damaged == 1 && metrics.damage_total == 0.25 && metrics.deaths == 0;
}

bool cancellation_test() {
    const auto cfg = config();
    OwnedState state(36);
    wdl::ScheduleState schedule{101, 0};
    if (!initialize(cfg, state) || !advance(cfg, state, schedule, 0)) return false;
    const OwnedState before = state;
    const wdl::ScheduleState schedule_before = schedule;
    wdl::TickMetrics tick_metrics{1, 2, 3, 4, 5, 6.0, 7};
    const auto tick_status = wdl::advance_tick(cfg, state.view(), &schedule, 1, true, &tick_metrics);
    if (tick_status != wdl::Status::kCancelled || !(state == before)
        || !schedule_equal(schedule, schedule_before) || !metric_sentinel(tick_metrics)) {
        std::cerr << "cancel tick status=" << static_cast<int>(tick_status)
                  << " state=" << (state == before) << " schedule="
                  << schedule_equal(schedule, schedule_before) << " metric=" << metric_sentinel(tick_metrics)
                  << " values=" << tick_metrics.cue_cell_count << ',' << tick_metrics.active_cell_count
                  << ',' << tick_metrics.event_count << ',' << tick_metrics.cue_without_active_count
                  << ',' << tick_metrics.active_without_prior_cue_count << ','
                  << tick_metrics.cue_to_active_delay_sum << ',' << tick_metrics.cue_to_active_delay_count << '\n';
        return false;
    }

    std::array<std::uint8_t, 1> alive{1};
    std::array<std::int32_t, 1> pos_y{0}, pos_x{0};
    std::array<float, 1> health{1.0F};
    wdl::AgentStateView agents{alive.data(), pos_y.data(), pos_x.data(), health.data(), nullptr, 1};
    wdl::DamageMetrics damage_metrics{1, 2, 3, 4.0, 5.0F};
    const auto damage_status = wdl::apply_delayed_damage(cfg, state.const_view(), agents, true, &damage_metrics);
    if (damage_status != wdl::Status::kCancelled || health[0] != 1.0F || alive[0] != 1
        || !damage_metric_sentinel(damage_metrics)) {
        std::cerr << "cancel damage status=" << static_cast<int>(damage_status)
                  << " health=" << health[0] << " alive=" << static_cast<int>(alive[0])
                  << " metric=" << damage_metric_sentinel(damage_metrics) << '\n';
        return false;
    }
    return true;
}

bool malformed_atomicity_test() {
    const auto cfg = config();
    OwnedState state(36);
    wdl::ScheduleState schedule{101, 0};
    if (!initialize(cfg, state) || !advance(cfg, state, schedule, 0)) return false;
    state.timer[7] = -2;
    const OwnedState malformed_before = state;
    const auto schedule_before = schedule;
    wdl::TickMetrics tick_metrics{1, 2, 3, 4, 5, 6.0, 7};
    if (wdl::advance_tick(cfg, state.view(), &schedule, 1, false, &tick_metrics) == wdl::Status::kOk
        || !(state == malformed_before) || !schedule_equal(schedule, schedule_before)
        || !metric_sentinel(tick_metrics)) return false;

    state.timer[7] = -1;
    std::array<std::uint8_t, 2> alive{1, 1};
    std::array<std::int32_t, 2> pos_y{0, 0}, pos_x{0, cfg.width};
    std::array<float, 2> health{1.0F, 1.0F};
    wdl::AgentStateView agents{alive.data(), pos_y.data(), pos_x.data(), health.data(), nullptr, 2};
    wdl::DamageMetrics damage_metrics{1, 2, 3, 4.0, 5.0F};
    if (wdl::apply_delayed_damage(cfg, state.const_view(), agents, false, &damage_metrics)
        == wdl::Status::kOk) return false;
    return health == std::array<float, 2>{1.0F, 1.0F} && alive == std::array<std::uint8_t, 2>{1, 1}
        && damage_metric_sentinel(damage_metrics);
}

bool checkpoint_fail_closed_test() {
    const auto cfg = config();
    OwnedState source(36), destination(36);
    wdl::ScheduleState source_schedule{101, 0}, destination_schedule{999, 777};
    if (!initialize(cfg, source) || !initialize(cfg, destination) || !advance(cfg, source, source_schedule, 0)) {
        return false;
    }
    std::fill(destination.cue.begin(), destination.cue.end(), 0.5F);
    std::vector<std::uint8_t> bytes(wdl::checkpoint_size_bytes(cfg));
    std::size_t written = 0;
    if (wdl::write_checkpoint(cfg, source.const_view(), source_schedule, bytes.data(), bytes.size(), &written)
        != wdl::Status::kOk) return false;
    const OwnedState before = destination;
    const auto schedule_before = destination_schedule;
    if (wdl::restore_checkpoint(cfg, bytes.data(), bytes.size() - 1, destination.view(), &destination_schedule)
        != wdl::Status::kInvalidCheckpoint || !(destination == before)
        || !schedule_equal(destination_schedule, schedule_before)) return false;
    bytes.push_back(0);
    if (wdl::restore_checkpoint(cfg, bytes.data(), bytes.size(), destination.view(), &destination_schedule)
        != wdl::Status::kInvalidCheckpoint || !(destination == before)
        || !schedule_equal(destination_schedule, schedule_before)) return false;
    bytes.resize(written);
    bytes[0] ^= 0xFFU;
    return wdl::restore_checkpoint(cfg, bytes.data(), bytes.size(), destination.view(), &destination_schedule)
            == wdl::Status::kInvalidCheckpoint
        && destination == before && schedule_equal(destination_schedule, schedule_before);
}

bool scheduled_event_and_dead_cue_flag_test() {
    auto cfg = config();
    cfg.cue_enabled = 0;
    OwnedState state(36);
    wdl::ScheduleState schedule{101, 0};
    if (!initialize(cfg, state)) return false;
    const auto expected = wdl::stable_event_coordinate(101, 0, cfg.height, cfg.width);
    wdl::TickMetrics metrics{};
    wdl::ScheduledEventRow event{};
    if (wdl::advance_tick(cfg, state.view(), &schedule, 0, false, &metrics, &event)
        != wdl::Status::kOk) return false;
    const std::size_t cell = static_cast<std::size_t>(expected.y) * cfg.width + expected.x;
    if (event.event_id != expected.event_id || event.abs_step != 0
        || event.center_y != expected.y || event.center_x != expected.x
        || event.scheduled_cell_count != 1 || event.intensity != cfg.intensity
        || state.cue[cell] != 0.0F) return false;

    const OwnedState before = state;
    const auto schedule_before = schedule;
    const wdl::ScheduledEventRow sentinel{11, 12, 13, 14, 15, 0.5F};
    event = sentinel;
    if (wdl::advance_tick(cfg, state.view(), &schedule, 1, true, &metrics, &event)
        != wdl::Status::kCancelled || !(state == before)
        || !schedule_equal(schedule, schedule_before)) return false;
    return event.event_id == sentinel.event_id && event.abs_step == sentinel.abs_step
        && event.center_y == sentinel.center_y && event.center_x == sentinel.center_x
        && event.scheduled_cell_count == sentinel.scheduled_cell_count
        && event.intensity == sentinel.intensity;
}

bool descriptor_test() {
    static_assert(wdl::kSchemaVersion == 1);
    static_assert(wdl::kStateDescriptorCount == 7);
    static_assert(wdl::kObservationChannelCount == 1);
    static_assert(wdl::kScheduledEventFieldCount == 6);
    static_assert(wdl::kTickMetricFieldCount == 7);
    static_assert(wdl::kGenerationSummaryFieldCount == 13);
    const std::array<std::string, 7> names{
        "patch_delayed_hazard_cue", "patch_delayed_hazard_timer",
        "patch_delayed_hazard_active", "patch_delayed_hazard_intensity",
        "patch_delayed_hazard_age", "patch_delayed_hazard_event_id",
        "patch_delayed_hazard_prior_cue",
    };
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (wdl::kStateDescriptors[i].name != names[i] || wdl::kStateDescriptors[i].element_bytes == 0) return false;
    }
    return std::string(wdl::kObservationChannels[0]) == "wdl_delayed_hazard_cue_norm"
        && std::string(wdl::kScheduledEventFields[5]) == "intensity"
        && std::string(wdl::kTickMetricFields[0]) == "wdl_delayed_hazard_cue_cell_count"
        && std::string(wdl::kGenerationSummaryFields[12]) == "wdl_delayed_hazard_tick_count";
}

bool abi_layout_test() {
    static_assert(sizeof(void*) == 8, "qualified CPU oracle ABI is LP64");
    static_assert(std::is_standard_layout_v<wdl::Config> && std::is_trivially_copyable_v<wdl::Config>);
    static_assert(std::is_standard_layout_v<wdl::StateView> && std::is_trivially_copyable_v<wdl::StateView>);
    static_assert(std::is_standard_layout_v<wdl::ConstStateView> && std::is_trivially_copyable_v<wdl::ConstStateView>);
    static_assert(std::is_standard_layout_v<wdl::ScheduleState> && std::is_trivially_copyable_v<wdl::ScheduleState>);
    static_assert(std::is_standard_layout_v<wdl::ScheduledEventRow> && std::is_trivially_copyable_v<wdl::ScheduledEventRow>);
    static_assert(sizeof(wdl::Config) == 48 && offsetof(wdl::Config, intensity) == 32
        && offsetof(wdl::Config, enabled) == 40);
    static_assert(sizeof(wdl::StateView) == 64 && offsetof(wdl::StateView, cell_count) == 56);
    static_assert(sizeof(wdl::ConstStateView) == 64 && offsetof(wdl::ConstStateView, cell_count) == 56);
    static_assert(sizeof(wdl::ScheduleState) == 16 && offsetof(wdl::ScheduleState, next_abs_step) == 8);
    static_assert(sizeof(wdl::EventCoordinate) == 16 && offsetof(wdl::EventCoordinate, event_id) == 8);
    static_assert(sizeof(wdl::TickMetrics) == 56 && offsetof(wdl::TickMetrics, cue_to_active_delay_sum) == 40);
    static_assert(sizeof(wdl::ScheduledEventRow) == 40 && offsetof(wdl::ScheduledEventRow, scheduled_cell_count) == 24);
    static_assert(sizeof(wdl::AgentStateView) == 48 && offsetof(wdl::AgentStateView, agent_count) == 40);
    static_assert(sizeof(wdl::DamageMetrics) == 40 && offsetof(wdl::DamageMetrics, damage_total) == 24);
    static_assert(sizeof(wdl::Descriptor) == 24 && offsetof(wdl::Descriptor, element_bytes) == 16);
    return true;
}

std::uint32_t float_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::uint64_t double_bits(double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

template <typename Values, typename Convert>
void emit_values(const Values& values, Convert convert) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) std::cout << ',';
        std::cout << convert(values[i]);
    }
}

void emit_case_trace_line(
    std::int64_t step,
    const wdl::Config& cfg,
    const OwnedState& state,
    const wdl::ScheduleState& schedule,
    const wdl::TickMetrics& metrics
) {
    std::cout << step << '|';
    emit_values(state.cue, [](float value) { return float_bits(value); });
    std::cout << '|';
    emit_values(state.timer, [](std::int32_t value) { return value; });
    std::cout << '|';
    emit_values(state.active, [](float value) { return float_bits(value); });
    std::cout << '|';
    emit_values(state.intensity, [](float value) { return float_bits(value); });
    std::cout << '|';
    emit_values(state.age, [](std::int32_t value) { return value; });
    std::cout << '|';
    emit_values(state.event_id, [](std::int64_t value) { return value; });
    std::cout << '|';
    emit_values(state.prior_cue, [](std::int8_t value) { return static_cast<int>(value); });
    std::cout << '|';
    for (std::size_t i = 0; i < state.cue.size(); ++i) {
        if (i != 0) std::cout << ',';
        const auto y = static_cast<std::int32_t>(i / static_cast<std::size_t>(cfg.width));
        const auto x = static_cast<std::int32_t>(i % static_cast<std::size_t>(cfg.width));
        std::cout << float_bits(wdl::cue_observation_value(
            state.cue[i], cfg.observation_dropout_rate, schedule.base_seed, y, x));
    }
    std::cout << '|' << metrics.cue_cell_count << ',' << metrics.active_cell_count << ','
              << metrics.event_count << ',' << metrics.cue_without_active_count << ','
              << metrics.active_without_prior_cue_count << ','
              << double_bits(metrics.cue_to_active_delay_sum) << ','
              << metrics.cue_to_active_delay_count << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 6 && std::string(argv[1]) == "coordinate") {
        const auto coordinate = wdl::stable_event_coordinate(
            std::strtoll(argv[2], nullptr, 10), std::strtoll(argv[3], nullptr, 10),
            static_cast<std::int32_t>(std::strtol(argv[4], nullptr, 10)),
            static_cast<std::int32_t>(std::strtol(argv[5], nullptr, 10)));
        std::cout << coordinate.y << ' ' << coordinate.x << ' ' << coordinate.event_id << '\n';
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "trace") {
        const auto cfg = config();
        OwnedState state(36);
        wdl::ScheduleState schedule{101, 0};
        if (!initialize(cfg, state)) return 1;
        const auto coordinate = wdl::stable_event_coordinate(101, 0, cfg.height, cfg.width);
        const std::size_t cell = static_cast<std::size_t>(coordinate.y) * cfg.width + coordinate.x;
        for (std::int64_t step = 0; step < 8; ++step) {
            wdl::TickMetrics metrics{};
            if (!advance(cfg, state, schedule, step, &metrics)) return 2;
            std::cout << step << ' ' << static_cast<int>(state.cue[cell]) << ' '
                      << state.timer[cell] << ' ' << static_cast<int>(state.active[cell]) << ' ';
            std::cout.setf(std::ios::fixed);
            std::cout.precision(2);
            std::cout << state.intensity[cell] << ' ' << state.age[cell] << ' '
                      << state.event_id[cell] << ' ' << static_cast<int>(state.prior_cue[cell]) << ' '
                      << metrics.event_count << ' ' << metrics.cue_cell_count << ' '
                      << metrics.active_cell_count << '\n';
        }
        return 0;
    }
    if (argc == 13 && std::string(argv[1]) == "trace-case") {
        wdl::Config cfg{};
        const auto seed = std::strtoll(argv[2], nullptr, 10);
        cfg.height = static_cast<std::int32_t>(std::strtol(argv[3], nullptr, 10));
        cfg.width = static_cast<std::int32_t>(std::strtol(argv[4], nullptr, 10));
        cfg.delay_steps = static_cast<std::int32_t>(std::strtol(argv[5], nullptr, 10));
        cfg.cue_duration_steps = static_cast<std::int32_t>(std::strtol(argv[6], nullptr, 10));
        cfg.active_duration_steps = static_cast<std::int32_t>(std::strtol(argv[7], nullptr, 10));
        cfg.event_period_steps = static_cast<std::int32_t>(std::strtol(argv[8], nullptr, 10));
        cfg.radius = static_cast<std::int32_t>(std::strtol(argv[9], nullptr, 10));
        cfg.intensity = std::strtof(argv[10], nullptr);
        cfg.observation_dropout_rate = std::strtof(argv[11], nullptr);
        const auto steps = std::strtoll(argv[12], nullptr, 10);
        cfg.enabled = 1;
        cfg.cue_enabled = 1;
        cfg.observation_enabled = 1;
        cfg.damage_enabled = 1;
        cfg.deterministic_schedule_enabled = 1;
        const std::size_t cell_count = static_cast<std::size_t>(cfg.height) * cfg.width;
        OwnedState state(cell_count);
        wdl::ScheduleState schedule{seed, 0};
        if (!initialize(cfg, state)) return 3;
        for (std::int64_t step = 0; step < steps; ++step) {
            wdl::TickMetrics metrics{};
            if (!advance(cfg, state, schedule, step, &metrics)) return 4;
            emit_case_trace_line(step, cfg, state, schedule, metrics);
        }
        return 0;
    }
    const std::array<bool, 10> results{
        descriptor_test() && abi_layout_test(), phase_schedule_test(), configured_active_duration_test(),
        same_seed_and_dropout_test(), checkpoint_and_schedule_test(), damage_test(),
        cancellation_test(), malformed_atomicity_test(), checkpoint_fail_closed_test(),
        scheduled_event_and_dead_cue_flag_test(),
    };
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (!results[i]) {
            std::cerr << "wdl_native_contract_probe failed case=" << i << '\n';
            return static_cast<int>(i + 1);
        }
    }
    std::cout << "wdl_native_contract_probe=ok cases=" << results.size() << '\n';
    return 0;
}
