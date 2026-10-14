#include "wdl_cpu_oracle.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace eiel::wdl::cpu_oracle_v1 {
namespace {

constexpr std::uint64_t kCheckpointMagic = 0x31454C434F4C4457ULL;  // WDLOCLE1

template <typename T>
void write_little(std::uint8_t*& cursor, T value) noexcept {
    std::uint8_t bytes[sizeof(T)];
    std::memcpy(bytes, &value, sizeof(T));
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    std::reverse(bytes, bytes + sizeof(T));
#endif
    std::memcpy(cursor, bytes, sizeof(T));
    cursor += sizeof(T);
}

template <typename T>
bool read_little(const std::uint8_t*& cursor, const std::uint8_t* end, T* value) noexcept {
    if (value == nullptr || static_cast<std::size_t>(end - cursor) < sizeof(T)) return false;
    std::uint8_t bytes[sizeof(T)];
    std::memcpy(bytes, cursor, sizeof(T));
    cursor += sizeof(T);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    std::reverse(bytes, bytes + sizeof(T));
#endif
    std::memcpy(value, bytes, sizeof(T));
    return true;
}

bool config_equal(const OracleConfig& a, const OracleConfig& b) noexcept {
    const Config& x = a.wdl;
    const Config& y = b.wdl;
    return x.schema_version == y.schema_version && x.height == y.height && x.width == y.width
        && x.delay_steps == y.delay_steps && x.cue_duration_steps == y.cue_duration_steps
        && x.active_duration_steps == y.active_duration_steps
        && x.event_period_steps == y.event_period_steps && x.radius == y.radius
        && x.intensity == y.intensity && x.observation_dropout_rate == y.observation_dropout_rate
        && x.enabled == y.enabled && x.cue_enabled == y.cue_enabled
        && x.observation_enabled == y.observation_enabled && x.damage_enabled == y.damage_enabled
        && x.deterministic_schedule_enabled == y.deterministic_schedule_enabled
        && a.view_radius == b.view_radius && a.max_scheduled_events == b.max_scheduled_events
        && a.max_damage_events == b.max_damage_events;
}

bool checked_add(std::size_t* total, std::size_t value) noexcept {
    if (total == nullptr || value > std::numeric_limits<std::size_t>::max() - *total) return false;
    *total += value;
    return true;
}

bool checked_product(std::size_t a, std::size_t b, std::size_t* result) noexcept {
    if (result == nullptr || (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)) return false;
    *result = a * b;
    return true;
}

}  // namespace

native_v1::StateView CpuOracle::mutable_state_view() noexcept {
    return {cue_.data(), timer_.data(), active_.data(), intensity_.data(), age_.data(),
            event_id_.data(), prior_cue_.data(), cue_.size()};
}

native_v1::ConstStateView CpuOracle::state_view() const noexcept {
    return {cue_.data(), timer_.data(), active_.data(), intensity_.data(), age_.data(),
            event_id_.data(), prior_cue_.data(), cue_.size()};
}

native_v1::AgentStateView CpuOracle::mutable_agent_view() noexcept {
    return {alive_.data(), pos_y_.data(), pos_x_.data(), health_.data(),
            damage_scale_.data(), alive_.size()};
}

Status CpuOracle::initialize(
    const OracleConfig& config,
    std::int64_t base_seed,
    std::size_t agent_count
) {
    if (native_v1::validate_config(config.wdl) != Status::kOk || config.view_radius < 0
        || config.view_radius > (std::numeric_limits<std::int32_t>::max() - 1) / 2
        || agent_count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        return Status::kInvalidConfig;
    }
    const std::size_t side = static_cast<std::size_t>(2 * config.view_radius + 1);
    if (side > std::numeric_limits<std::size_t>::max() / side) {
        return Status::kInvalidConfig;
    }
    const std::size_t observation_width = side * side;
    if (agent_count > 0 && observation_width > std::numeric_limits<std::size_t>::max() / agent_count) {
        return Status::kInsufficientCapacity;
    }
    try {
        CpuOracle candidate;
        candidate.config_ = config;
        candidate.schedule_ = {base_seed, 0};
        const std::size_t cells = static_cast<std::size_t>(config.wdl.height) * config.wdl.width;
        candidate.cue_.resize(cells);
        candidate.timer_.resize(cells);
        candidate.active_.resize(cells);
        candidate.intensity_.resize(cells);
        candidate.age_.resize(cells);
        candidate.event_id_.resize(cells);
        candidate.prior_cue_.resize(cells);
        candidate.alive_.assign(agent_count, 0);
        candidate.pos_y_.assign(agent_count, -1);
        candidate.pos_x_.assign(agent_count, -1);
        candidate.health_.assign(agent_count, 0.0F);
        candidate.damage_scale_.assign(agent_count, 1.0F);
        candidate.damaged_ever_.assign(agent_count, 0);
        const Status status = native_v1::initialize_state(config.wdl, candidate.mutable_state_view());
        if (status != Status::kOk) return status;
        candidate.initialized_ = true;
        *this = std::move(candidate);
        return Status::kOk;
    } catch (const std::bad_alloc&) {
        return Status::kInsufficientCapacity;
    }
}

Status CpuOracle::validate_agents() const noexcept {
    const std::size_t count = alive_.size();
    if (pos_y_.size() != count || pos_x_.size() != count || health_.size() != count
        || damage_scale_.size() != count || damaged_ever_.size() != count) {
        return Status::kInvalidState;
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (alive_[i] > 1 || !std::isfinite(health_[i]) || health_[i] < 0.0F
            || !std::isfinite(damage_scale_[i]) || damage_scale_[i] < 0.0F) {
            return Status::kInvalidState;
        }
        if (alive_[i] && (pos_y_[i] < 0 || pos_y_[i] >= config_.wdl.height
            || pos_x_[i] < 0 || pos_x_[i] >= config_.wdl.width)) {
            return Status::kInvalidState;
        }
    }
    return Status::kOk;
}

Status CpuOracle::set_agents(
    const std::uint8_t* alive,
    const std::int32_t* pos_y,
    const std::int32_t* pos_x,
    const float* health,
    const float* damage_scale,
    std::size_t agent_count
) {
    if (!initialized_ || agent_count != alive_.size()
        || (agent_count > 0 && (alive == nullptr || pos_y == nullptr || pos_x == nullptr
            || health == nullptr))) {
        return Status::kInvalidState;
    }
    try {
        CpuOracle candidate = *this;
        for (std::size_t i = 0; i < agent_count; ++i) {
            candidate.alive_[i] = alive[i];
            candidate.pos_y_[i] = pos_y[i];
            candidate.pos_x_[i] = pos_x[i];
            candidate.health_[i] = health[i];
            candidate.damage_scale_[i] = damage_scale == nullptr ? 1.0F : damage_scale[i];
        }
        const Status status = candidate.validate_agents();
        if (status != Status::kOk) return status;
        candidate.build_observations();
        *this = std::move(candidate);
        return Status::kOk;
    } catch (const std::bad_alloc&) {
        return Status::kInsufficientCapacity;
    }
}

std::size_t CpuOracle::observation_width() const noexcept {
    if (!config_.wdl.enabled || !config_.wdl.observation_enabled || config_.view_radius < 0) return 0;
    const std::size_t side = static_cast<std::size_t>(2 * config_.view_radius + 1);
    return side * side;
}

void CpuOracle::build_observations() {
    const std::size_t width = observation_width();
    last_observation_agent_ids_.clear();
    last_observation_.clear();
    last_observation_agent_ids_.reserve(alive_.size());
    if (width > 0) last_observation_.reserve(alive_.size() * width);
    const std::int32_t r = config_.view_radius;
    const std::int32_t side = 2 * r + 1;
    for (std::size_t agent = 0; agent < alive_.size(); ++agent) {
        if (!alive_[agent]) continue;
        last_observation_agent_ids_.push_back(static_cast<std::int32_t>(agent));
        if (width == 0) continue;
        const std::size_t row_start = last_observation_.size();
        last_observation_.resize(row_start + width, 0.0F);
        for (std::int32_t dy = -r; dy <= r; ++dy) {
            for (std::int32_t dx = -r; dx <= r; ++dx) {
                const std::int32_t y = pos_y_[agent] + dy;
                const std::int32_t x = pos_x_[agent] + dx;
                if (y < 0 || y >= config_.wdl.height || x < 0 || x >= config_.wdl.width) continue;
                const std::size_t cell = static_cast<std::size_t>(y) * config_.wdl.width + x;
                const std::size_t column = static_cast<std::size_t>(dy + r) * side + (dx + r);
                last_observation_[row_start + column] = native_v1::cue_observation_value(
                    cue_[cell], config_.wdl.observation_dropout_rate, schedule_.base_seed, y, x);
            }
        }
    }
}

std::int64_t CpuOracle::count_cue_exposures() const noexcept {
    std::int64_t count = 0;
    for (std::size_t i = 0; i < alive_.size(); ++i) {
        if (!alive_[i]) continue;
        const std::size_t cell = static_cast<std::size_t>(pos_y_[i]) * config_.wdl.width + pos_x_[i];
        count += cue_[cell] > 0.0F;
    }
    return count;
}

Status CpuOracle::step(
    std::int64_t abs_step,
    const std::int32_t* post_action_y,
    const std::int32_t* post_action_x,
    std::size_t position_count,
    bool cancel_requested,
    TickReceipt* receipt
) {
    if (!initialized_ || receipt == nullptr) return Status::kInvalidState;
    if (cancel_requested) return Status::kCancelled;
    if (abs_step != schedule_.next_abs_step) return Status::kInvalidState;
    if ((post_action_y == nullptr) != (post_action_x == nullptr)
        || (post_action_y != nullptr && position_count != alive_.size())
        || (post_action_y == nullptr && position_count != 0)) {
        return Status::kInvalidState;
    }
    try {
        CpuOracle candidate = *this;
        TickReceipt pending{};
        pending.abs_step = abs_step;
        const Status transition = native_v1::advance_tick(
            candidate.config_.wdl,
            candidate.mutable_state_view(),
            &candidate.schedule_,
            abs_step,
            false,
            &pending.tick,
            &pending.scheduled_event);
        if (transition != Status::kOk) return transition;

        candidate.build_observations();
        pending.observation_rows = candidate.last_observation_agent_ids_.size();
        pending.observation_width = candidate.observation_width();
        const std::int64_t cue_exposures = candidate.count_cue_exposures();

        if (post_action_y != nullptr) {
            for (std::size_t i = 0; i < candidate.alive_.size(); ++i) {
                candidate.pos_y_[i] = post_action_y[i];
                candidate.pos_x_[i] = post_action_x[i];
            }
            const Status agent_status = candidate.validate_agents();
            if (agent_status != Status::kOk) return agent_status;
        }

        const std::vector<std::uint8_t> alive_before = candidate.alive_;
        const std::vector<float> health_before = candidate.health_;
        const Status damage = native_v1::apply_delayed_damage(
            candidate.config_.wdl,
            candidate.state_view(),
            candidate.mutable_agent_view(),
            false,
            &pending.damage);
        if (damage != Status::kOk) return damage;

        if (pending.scheduled_event.event_id > 0) {
            if (candidate.scheduled_events_.size() >= candidate.config_.max_scheduled_events) {
                return Status::kInsufficientCapacity;
            }
            candidate.scheduled_events_.push_back(pending.scheduled_event);
        }
        for (std::size_t i = 0; i < candidate.alive_.size(); ++i) {
            if (candidate.health_[i] >= health_before[i]) continue;
            if (candidate.damage_events_.size() >= candidate.config_.max_damage_events) {
                return Status::kInsufficientCapacity;
            }
            const std::size_t cell = static_cast<std::size_t>(candidate.pos_y_[i])
                * candidate.config_.wdl.width + candidate.pos_x_[i];
            candidate.damage_events_.push_back({
                abs_step,
                static_cast<std::int32_t>(i),
                candidate.pos_y_[i],
                candidate.pos_x_[i],
                candidate.intensity_[cell],
                candidate.damage_scale_[i],
                candidate.intensity_[cell] * candidate.damage_scale_[i],
                health_before[i],
                candidate.health_[i],
                static_cast<std::uint8_t>(alive_before[i] && !candidate.alive_[i]),
            });
            candidate.damaged_ever_[i] = 1;
        }

        candidate.summary_.event_count += pending.tick.event_count;
        candidate.summary_.cue_cell_count_sum += pending.tick.cue_cell_count;
        candidate.summary_.active_cell_count_sum += pending.tick.active_cell_count;
        candidate.summary_.agent_cue_exposure_count += cue_exposures;
        candidate.summary_.agent_active_exposure_count += pending.damage.active_exposure_count;
        candidate.summary_.damage_total += pending.damage.damage_total;
        candidate.summary_.damage_max = std::max(candidate.summary_.damage_max, pending.damage.damage_max);
        candidate.summary_.cue_to_active_delay_sum += pending.tick.cue_to_active_delay_sum;
        candidate.summary_.cue_to_active_delay_count += pending.tick.cue_to_active_delay_count;
        candidate.summary_.cue_without_active_count += pending.tick.cue_without_active_count;
        candidate.summary_.active_without_prior_cue_count += pending.tick.active_without_prior_cue_count;
        ++candidate.summary_.tick_count;

        *this = std::move(candidate);
        *receipt = pending;
        return Status::kOk;
    } catch (const std::bad_alloc&) {
        return Status::kInsufficientCapacity;
    }
}

Status CpuOracle::begin_generation(bool clear_event_journals) noexcept {
    if (!initialized_) return Status::kInvalidState;
    summary_ = {};
    std::fill(damaged_ever_.begin(), damaged_ever_.end(), 0);
    if (clear_event_journals) {
        scheduled_events_.clear();
        damage_events_.clear();
    }
    return Status::kOk;
}

GenerationSummary CpuOracle::generation_summary() const noexcept {
    GenerationSummary result{};
    result.event_count = summary_.event_count;
    if (summary_.tick_count > 0) {
        result.cue_cell_count_mean = summary_.cue_cell_count_sum / summary_.tick_count;
        result.active_cell_count_mean = summary_.active_cell_count_sum / summary_.tick_count;
    }
    result.agent_cue_exposure_count = summary_.agent_cue_exposure_count;
    result.agent_active_exposure_count = summary_.agent_active_exposure_count;
    result.damage_total = summary_.damage_total;
    result.damage_max = summary_.damage_max;
    result.agents_damaged = std::count(damaged_ever_.begin(), damaged_ever_.end(), 1);
    if (summary_.cue_to_active_delay_count > 0) {
        result.cue_to_active_delay_mean = summary_.cue_to_active_delay_sum
            / summary_.cue_to_active_delay_count;
    }
    result.cue_without_active_count = summary_.cue_without_active_count;
    result.active_without_prior_cue_count = summary_.active_without_prior_cue_count;
    result.enabled = config_.wdl.enabled ? 1 : 0;
    result.tick_count = summary_.tick_count;
    return result;
}

std::size_t CpuOracle::checkpoint_size_bytes() const noexcept {
    if (!initialized_) return 0;
    const std::size_t state_bytes = native_v1::checkpoint_size_bytes(config_.wdl);
    const std::size_t agents = alive_.size();
    std::size_t total = 8 + 4 + 45 + 4 + 8 + 8 + 8 + 8;
    std::size_t bytes = 0;
    if (!checked_add(&total, state_bytes)
        || !checked_product(agents, 18, &bytes) || !checked_add(&total, bytes)
        || !checked_add(&total, 92) || !checked_add(&total, 8)
        || !checked_product(scheduled_events_.size(), 36, &bytes) || !checked_add(&total, bytes)
        || !checked_add(&total, 8)
        || !checked_product(damage_events_.size(), 41, &bytes) || !checked_add(&total, bytes)) {
        return 0;
    }
    return total;
}

Status CpuOracle::write_checkpoint(
    std::uint8_t* output,
    std::size_t capacity,
    std::size_t* bytes_written
) const noexcept {
    const std::size_t required = checkpoint_size_bytes();
    if (required == 0 || output == nullptr || capacity < required || bytes_written == nullptr) {
        return Status::kInsufficientCapacity;
    }
    std::vector<std::uint8_t> state_bytes;
    try {
        state_bytes.resize(native_v1::checkpoint_size_bytes(config_.wdl));
    } catch (const std::bad_alloc&) {
        return Status::kInsufficientCapacity;
    }
    std::size_t state_written = 0;
    const Status state_status = native_v1::write_checkpoint(
        config_.wdl, state_view(), schedule_, state_bytes.data(), state_bytes.size(), &state_written);
    if (state_status != Status::kOk) return state_status;

    std::uint8_t* cursor = output;
    write_little(cursor, kCheckpointMagic);
    write_little(cursor, kOracleSchemaVersion);
    const Config& c = config_.wdl;
    write_little(cursor, c.schema_version); write_little(cursor, c.height); write_little(cursor, c.width);
    write_little(cursor, c.delay_steps); write_little(cursor, c.cue_duration_steps);
    write_little(cursor, c.active_duration_steps); write_little(cursor, c.event_period_steps);
    write_little(cursor, c.radius); write_little(cursor, c.intensity);
    write_little(cursor, c.observation_dropout_rate); write_little(cursor, c.enabled);
    write_little(cursor, c.cue_enabled); write_little(cursor, c.observation_enabled);
    write_little(cursor, c.damage_enabled); write_little(cursor, c.deterministic_schedule_enabled);
    write_little(cursor, config_.view_radius);
    write_little(cursor, static_cast<std::uint64_t>(config_.max_scheduled_events));
    write_little(cursor, static_cast<std::uint64_t>(config_.max_damage_events));
    write_little(cursor, static_cast<std::uint64_t>(alive_.size()));
    write_little(cursor, static_cast<std::uint64_t>(state_written));
    std::memcpy(cursor, state_bytes.data(), state_written); cursor += state_written;
    for (std::size_t i = 0; i < alive_.size(); ++i) {
        write_little(cursor, alive_[i]); write_little(cursor, pos_y_[i]); write_little(cursor, pos_x_[i]);
        write_little(cursor, health_[i]); write_little(cursor, damage_scale_[i]); write_little(cursor, damaged_ever_[i]);
    }
    write_little(cursor, summary_.event_count); write_little(cursor, summary_.cue_cell_count_sum);
    write_little(cursor, summary_.active_cell_count_sum); write_little(cursor, summary_.agent_cue_exposure_count);
    write_little(cursor, summary_.agent_active_exposure_count); write_little(cursor, summary_.damage_total);
    write_little(cursor, summary_.damage_max); write_little(cursor, summary_.cue_to_active_delay_sum);
    write_little(cursor, summary_.cue_to_active_delay_count); write_little(cursor, summary_.cue_without_active_count);
    write_little(cursor, summary_.active_without_prior_cue_count); write_little(cursor, summary_.tick_count);
    write_little(cursor, static_cast<std::uint64_t>(scheduled_events_.size()));
    for (const auto& event : scheduled_events_) {
        write_little(cursor, event.event_id); write_little(cursor, event.abs_step);
        write_little(cursor, event.center_y); write_little(cursor, event.center_x);
        write_little(cursor, event.scheduled_cell_count); write_little(cursor, event.intensity);
    }
    write_little(cursor, static_cast<std::uint64_t>(damage_events_.size()));
    for (const auto& event : damage_events_) {
        write_little(cursor, event.abs_step); write_little(cursor, event.agent_id);
        write_little(cursor, event.y); write_little(cursor, event.x); write_little(cursor, event.raw_intensity);
        write_little(cursor, event.damage_scale); write_little(cursor, event.effective_damage);
        write_little(cursor, event.health_before); write_little(cursor, event.health_after); write_little(cursor, event.died);
    }
    *bytes_written = required;
    return static_cast<std::size_t>(cursor - output) == required ? Status::kOk : Status::kInvalidCheckpoint;
}

Status CpuOracle::restore_checkpoint(const std::uint8_t* input, std::size_t size) noexcept {
    if (input == nullptr || size == 0) return Status::kInvalidCheckpoint;
    try {
        const std::uint8_t* cursor = input;
        const std::uint8_t* end = input + size;
        std::uint64_t magic = 0; std::int32_t schema = 0;
        if (!read_little(cursor, end, &magic) || !read_little(cursor, end, &schema)
            || magic != kCheckpointMagic || schema != kOracleSchemaVersion) return Status::kInvalidCheckpoint;
        OracleConfig stored{};
        Config& c = stored.wdl;
        if (!read_little(cursor, end, &c.schema_version) || !read_little(cursor, end, &c.height)
            || !read_little(cursor, end, &c.width) || !read_little(cursor, end, &c.delay_steps)
            || !read_little(cursor, end, &c.cue_duration_steps) || !read_little(cursor, end, &c.active_duration_steps)
            || !read_little(cursor, end, &c.event_period_steps) || !read_little(cursor, end, &c.radius)
            || !read_little(cursor, end, &c.intensity) || !read_little(cursor, end, &c.observation_dropout_rate)
            || !read_little(cursor, end, &c.enabled) || !read_little(cursor, end, &c.cue_enabled)
            || !read_little(cursor, end, &c.observation_enabled) || !read_little(cursor, end, &c.damage_enabled)
            || !read_little(cursor, end, &c.deterministic_schedule_enabled)
            || !read_little(cursor, end, &stored.view_radius)) return Status::kInvalidCheckpoint;
        std::uint64_t max_scheduled = 0, max_damage = 0, agent_count = 0, state_size = 0;
        if (!read_little(cursor, end, &max_scheduled) || !read_little(cursor, end, &max_damage)
            || !read_little(cursor, end, &agent_count) || !read_little(cursor, end, &state_size)) {
            return Status::kInvalidCheckpoint;
        }
        stored.max_scheduled_events = static_cast<std::size_t>(max_scheduled);
        stored.max_damage_events = static_cast<std::size_t>(max_damage);
        if (agent_count > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())
            || state_size > static_cast<std::uint64_t>(end - cursor)) return Status::kInvalidCheckpoint;
        if (!initialized_ || !config_equal(config_, stored) || agent_count != alive_.size()) {
            return Status::kInvalidCheckpoint;
        }
        CpuOracle candidate;
        Status status = candidate.initialize(stored, 0, static_cast<std::size_t>(agent_count));
        if (status != Status::kOk) return Status::kInvalidCheckpoint;
        status = native_v1::restore_checkpoint(
            stored.wdl, cursor, static_cast<std::size_t>(state_size), candidate.mutable_state_view(), &candidate.schedule_);
        if (status != Status::kOk) return Status::kInvalidCheckpoint;
        cursor += state_size;
        for (std::size_t i = 0; i < candidate.alive_.size(); ++i) {
            if (!read_little(cursor, end, &candidate.alive_[i]) || !read_little(cursor, end, &candidate.pos_y_[i])
                || !read_little(cursor, end, &candidate.pos_x_[i]) || !read_little(cursor, end, &candidate.health_[i])
                || !read_little(cursor, end, &candidate.damage_scale_[i])
                || !read_little(cursor, end, &candidate.damaged_ever_[i])) return Status::kInvalidCheckpoint;
        }
        auto& s = candidate.summary_;
        if (!read_little(cursor, end, &s.event_count) || !read_little(cursor, end, &s.cue_cell_count_sum)
            || !read_little(cursor, end, &s.active_cell_count_sum) || !read_little(cursor, end, &s.agent_cue_exposure_count)
            || !read_little(cursor, end, &s.agent_active_exposure_count) || !read_little(cursor, end, &s.damage_total)
            || !read_little(cursor, end, &s.damage_max) || !read_little(cursor, end, &s.cue_to_active_delay_sum)
            || !read_little(cursor, end, &s.cue_to_active_delay_count) || !read_little(cursor, end, &s.cue_without_active_count)
            || !read_little(cursor, end, &s.active_without_prior_cue_count) || !read_little(cursor, end, &s.tick_count)) {
            return Status::kInvalidCheckpoint;
        }
        std::uint64_t scheduled_count = 0;
        if (!read_little(cursor, end, &scheduled_count) || scheduled_count > stored.max_scheduled_events
            || scheduled_count > static_cast<std::uint64_t>(end - cursor) / 36U) {
            return Status::kInvalidCheckpoint;
        }
        candidate.scheduled_events_.resize(static_cast<std::size_t>(scheduled_count));
        for (auto& event : candidate.scheduled_events_) {
            if (!read_little(cursor, end, &event.event_id) || !read_little(cursor, end, &event.abs_step)
                || !read_little(cursor, end, &event.center_y) || !read_little(cursor, end, &event.center_x)
                || !read_little(cursor, end, &event.scheduled_cell_count) || !read_little(cursor, end, &event.intensity)) {
                return Status::kInvalidCheckpoint;
            }
        }
        std::uint64_t damage_count = 0;
        if (!read_little(cursor, end, &damage_count) || damage_count > stored.max_damage_events
            || damage_count > static_cast<std::uint64_t>(end - cursor) / 41U) {
            return Status::kInvalidCheckpoint;
        }
        candidate.damage_events_.resize(static_cast<std::size_t>(damage_count));
        for (auto& event : candidate.damage_events_) {
            if (!read_little(cursor, end, &event.abs_step) || !read_little(cursor, end, &event.agent_id)
                || !read_little(cursor, end, &event.y) || !read_little(cursor, end, &event.x)
                || !read_little(cursor, end, &event.raw_intensity) || !read_little(cursor, end, &event.damage_scale)
                || !read_little(cursor, end, &event.effective_damage) || !read_little(cursor, end, &event.health_before)
                || !read_little(cursor, end, &event.health_after) || !read_little(cursor, end, &event.died)) {
                return Status::kInvalidCheckpoint;
            }
        }
        if (cursor != end || candidate.validate_agents() != Status::kOk
            || native_v1::validate_state(candidate.config_.wdl, candidate.state_view()) != Status::kOk
            || !config_equal(candidate.config_, stored)) return Status::kInvalidCheckpoint;
        for (std::uint8_t value : candidate.damaged_ever_) {
            if (value > 1) return Status::kInvalidCheckpoint;
        }
        if (s.event_count < 0 || s.agent_cue_exposure_count < 0 || s.agent_active_exposure_count < 0
            || s.cue_to_active_delay_count < 0 || s.cue_without_active_count < 0
            || s.active_without_prior_cue_count < 0 || s.tick_count < 0
            || !std::isfinite(s.cue_cell_count_sum) || s.cue_cell_count_sum < 0.0
            || !std::isfinite(s.active_cell_count_sum) || s.active_cell_count_sum < 0.0
            || !std::isfinite(s.damage_total) || s.damage_total < 0.0
            || !std::isfinite(s.damage_max) || s.damage_max < 0.0F
            || !std::isfinite(s.cue_to_active_delay_sum) || s.cue_to_active_delay_sum < 0.0
            || s.tick_count > candidate.schedule_.next_abs_step) return Status::kInvalidCheckpoint;
        for (const auto& event : candidate.scheduled_events_) {
            if (event.event_id <= 0 || event.abs_step < 0 || event.abs_step >= candidate.schedule_.next_abs_step
                || event.center_y < 0 || event.center_y >= stored.wdl.height
                || event.center_x < 0 || event.center_x >= stored.wdl.width
                || event.scheduled_cell_count <= 0
                || event.scheduled_cell_count > static_cast<std::int64_t>(candidate.cue_.size())
                || !std::isfinite(event.intensity) || event.intensity < 0.0F || event.intensity > 1.0F) {
                return Status::kInvalidCheckpoint;
            }
        }
        for (const auto& event : candidate.damage_events_) {
            if (event.abs_step < 0 || event.abs_step >= candidate.schedule_.next_abs_step
                || event.agent_id < 0 || static_cast<std::size_t>(event.agent_id) >= candidate.alive_.size()
                || event.y < 0 || event.y >= stored.wdl.height || event.x < 0 || event.x >= stored.wdl.width
                || event.died > 1 || !std::isfinite(event.raw_intensity) || event.raw_intensity < 0.0F
                || event.raw_intensity > 1.0F || !std::isfinite(event.damage_scale) || event.damage_scale < 0.0F
                || !std::isfinite(event.effective_damage) || event.effective_damage < 0.0F
                || !std::isfinite(event.health_before) || event.health_before < 0.0F
                || !std::isfinite(event.health_after) || event.health_after < 0.0F
                || event.health_after > event.health_before) return Status::kInvalidCheckpoint;
        }
        candidate.build_observations();
        *this = std::move(candidate);
        return Status::kOk;
    } catch (...) {
        return Status::kInsufficientCapacity;
    }
}

}  // namespace eiel::wdl::cpu_oracle_v1
