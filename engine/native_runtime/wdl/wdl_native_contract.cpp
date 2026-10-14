#include "wdl_native_contract.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace eiel::wdl::native_v1 {

const Descriptor kStateDescriptors[kStateDescriptorCount] = {
    {"patch_delayed_hazard_cue", "float32", 4},
    {"patch_delayed_hazard_timer", "int32", 4},
    {"patch_delayed_hazard_active", "float32", 4},
    {"patch_delayed_hazard_intensity", "float32", 4},
    {"patch_delayed_hazard_age", "int32", 4},
    {"patch_delayed_hazard_event_id", "int64", 8},
    {"patch_delayed_hazard_prior_cue", "int8", 1},
};

const char* const kObservationChannels[kObservationChannelCount] = {
    "wdl_delayed_hazard_cue_norm",
};

const char* const kScheduledEventFields[kScheduledEventFieldCount] = {
    "event_id",
    "abs_step",
    "center_y",
    "center_x",
    "scheduled_cell_count",
    "intensity",
};

const char* const kTickMetricFields[kTickMetricFieldCount] = {
    "wdl_delayed_hazard_cue_cell_count",
    "wdl_delayed_hazard_active_cell_count",
    "wdl_delayed_hazard_event_count",
    "wdl_delayed_hazard_cue_without_active_count",
    "wdl_delayed_hazard_active_without_prior_cue_count",
    "wdl_delayed_hazard_cue_to_active_delay_sum",
    "wdl_delayed_hazard_cue_to_active_delay_count",
};

const char* const kGenerationSummaryFields[kGenerationSummaryFieldCount] = {
    "wdl_delayed_hazard_event_count",
    "wdl_delayed_hazard_cue_cell_count_mean",
    "wdl_delayed_hazard_active_cell_count_mean",
    "wdl_delayed_hazard_agent_cue_exposure_count",
    "wdl_delayed_hazard_agent_active_exposure_count",
    "wdl_delayed_hazard_damage_total",
    "wdl_delayed_hazard_damage_max",
    "wdl_delayed_hazard_agents_damaged",
    "wdl_delayed_hazard_cue_to_active_delay_mean",
    "wdl_delayed_hazard_cue_without_active_count",
    "wdl_delayed_hazard_active_without_prior_cue_count",
    "wdl_delayed_hazard_enable",
    "wdl_delayed_hazard_tick_count",
};

namespace {

constexpr std::array<std::uint32_t, 64> kSha256RoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
    0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
    0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
    0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

std::uint32_t rotate_right(std::uint32_t value, std::uint32_t distance) noexcept {
    return (value >> distance) | (value << (32U - distance));
}

void sha256(const std::uint8_t* input, std::size_t size, std::uint8_t output[32]) noexcept {
    std::uint32_t state[8] = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    std::array<std::uint8_t, 128> tail{};
    const std::size_t full_blocks = size / 64U;
    auto transform = [&state](const std::uint8_t* block) {
        std::uint32_t words[64]{};
        for (std::size_t i = 0; i < 16; ++i) {
            words[i] = (static_cast<std::uint32_t>(block[4 * i]) << 24U)
                | (static_cast<std::uint32_t>(block[4 * i + 1]) << 16U)
                | (static_cast<std::uint32_t>(block[4 * i + 2]) << 8U)
                | static_cast<std::uint32_t>(block[4 * i + 3]);
        }
        for (std::size_t i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotate_right(words[i - 15], 7U)
                ^ rotate_right(words[i - 15], 18U) ^ (words[i - 15] >> 3U);
            const std::uint32_t s1 = rotate_right(words[i - 2], 17U)
                ^ rotate_right(words[i - 2], 19U) ^ (words[i - 2] >> 10U);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }
        std::uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
        std::uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
        for (std::size_t i = 0; i < 64; ++i) {
            const std::uint32_t s1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
            const std::uint32_t choice = (e & f) ^ (~e & g);
            const std::uint32_t t1 = h + s1 + choice + kSha256RoundConstants[i] + words[i];
            const std::uint32_t s0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = s0 + majority;
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    };
    for (std::size_t block = 0; block < full_blocks; ++block) {
        transform(input + block * 64U);
    }
    const std::size_t remainder = size % 64U;
    if (remainder != 0) {
        std::memcpy(tail.data(), input + full_blocks * 64U, remainder);
    }
    tail[remainder] = 0x80U;
    const std::size_t tail_blocks = remainder >= 56U ? 2U : 1U;
    const std::uint64_t bits = static_cast<std::uint64_t>(size) * 8U;
    for (std::size_t i = 0; i < 8; ++i) {
        tail[tail_blocks * 64U - 1U - i] = static_cast<std::uint8_t>(bits >> (8U * i));
    }
    for (std::size_t block = 0; block < tail_blocks; ++block) {
        transform(tail.data() + block * 64U);
    }
    for (std::size_t i = 0; i < 8; ++i) {
        output[4 * i] = static_cast<std::uint8_t>(state[i] >> 24U);
        output[4 * i + 1] = static_cast<std::uint8_t>(state[i] >> 16U);
        output[4 * i + 2] = static_cast<std::uint8_t>(state[i] >> 8U);
        output[4 * i + 3] = static_cast<std::uint8_t>(state[i]);
    }
}

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
T read_little(const std::uint8_t*& cursor) noexcept {
    std::uint8_t bytes[sizeof(T)];
    std::memcpy(bytes, cursor, sizeof(T));
    cursor += sizeof(T);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    std::reverse(bytes, bytes + sizeof(T));
#endif
    T value{};
    std::memcpy(&value, bytes, sizeof(T));
    return value;
}

ConstStateView as_const(StateView state) noexcept {
    return {state.cue, state.timer, state.active, state.intensity, state.age,
            state.event_id, state.prior_cue, state.cell_count};
}

bool valid_bool(std::uint8_t value) noexcept { return value <= 1U; }

}  // namespace

Status validate_config(const Config& c) noexcept {
    if (c.schema_version != kSchemaVersion || c.height <= 0 || c.width <= 0
        || c.delay_steps < 1 || c.cue_duration_steps < 1
        || c.cue_duration_steps > c.delay_steps || c.active_duration_steps < 1
        || static_cast<std::int64_t>(c.event_period_steps)
            < static_cast<std::int64_t>(c.delay_steps)
                + static_cast<std::int64_t>(c.active_duration_steps) + 1
        || c.radius < 0 || !std::isfinite(c.intensity) || c.intensity < 0.0F || c.intensity > 1.0F
        || !std::isfinite(c.observation_dropout_rate) || c.observation_dropout_rate < 0.0F
        || c.observation_dropout_rate > 1.0F || !valid_bool(c.enabled) || !valid_bool(c.cue_enabled)
        || !valid_bool(c.observation_enabled) || !valid_bool(c.damage_enabled)
        || !valid_bool(c.deterministic_schedule_enabled)) {
        return Status::kInvalidConfig;
    }
    return Status::kOk;
}

Status validate_state(const Config& c, ConstStateView s) noexcept {
    if (validate_config(c) != Status::kOk || s.cell_count != static_cast<std::size_t>(c.height) * c.width
        || s.cue == nullptr || s.timer == nullptr || s.active == nullptr || s.intensity == nullptr
        || s.age == nullptr || s.event_id == nullptr || s.prior_cue == nullptr) {
        return Status::kInvalidState;
    }
    for (std::size_t i = 0; i < s.cell_count; ++i) {
        if (!std::isfinite(s.cue[i]) || s.cue[i] < 0.0F || s.cue[i] > 1.0F
            || !std::isfinite(s.active[i]) || s.active[i] < 0.0F || s.active[i] > 1.0F
            || !std::isfinite(s.intensity[i]) || s.intensity[i] < 0.0F || s.intensity[i] > 1.0F
            || s.timer[i] < -1 || s.age[i] < 0 || s.event_id[i] < 0
            || (s.prior_cue[i] != 0 && s.prior_cue[i] != 1)) {
            return Status::kInvalidState;
        }
    }
    return Status::kOk;
}

Status initialize_state(const Config& c, StateView s) noexcept {
    if (validate_config(c) != Status::kOk || s.cell_count != static_cast<std::size_t>(c.height) * c.width
        || s.cue == nullptr || s.timer == nullptr || s.active == nullptr || s.intensity == nullptr
        || s.age == nullptr || s.event_id == nullptr || s.prior_cue == nullptr) {
        return Status::kInvalidState;
    }
    for (std::size_t i = 0; i < s.cell_count; ++i) {
        s.cue[i] = 0.0F; s.timer[i] = -1; s.active[i] = 0.0F; s.intensity[i] = 0.0F;
        s.age[i] = 0; s.event_id[i] = 0; s.prior_cue[i] = 0;
    }
    return Status::kOk;
}

Phase phase_at(const Config& c, ConstStateView s, std::size_t i) noexcept {
    if (i >= s.cell_count || s.timer == nullptr || s.cue == nullptr || s.active == nullptr
        || s.prior_cue == nullptr || s.age == nullptr) {
        return Phase::kBaseline;
    }
    if (s.active[i] > 0.0F) return Phase::kActiveHazard;
    if (s.cue[i] > 0.0F) return Phase::kCue;
    if (s.prior_cue[i] > 0 && s.timer[i] >= 0 && s.age[i] >= c.cue_duration_steps) {
        return Phase::kHiddenGap;
    }
    return Phase::kBaseline;
}

EventCoordinate stable_event_coordinate(
    std::int64_t seed, std::int64_t event_index, std::int32_t height, std::int32_t width
) noexcept {
    char material[160]{};
    const int length = std::snprintf(material, sizeof(material), "wdl1:%lld:%lld:%d:%d",
        static_cast<long long>(seed), static_cast<long long>(event_index), height, width);
    std::uint8_t digest[32]{};
    sha256(reinterpret_cast<const std::uint8_t*>(material), static_cast<std::size_t>(std::max(0, length)), digest);
    auto u64_big = [&digest](std::size_t offset) {
        std::uint64_t value = 0;
        for (std::size_t i = 0; i < 8; ++i) value = (value << 8U) | digest[offset + i];
        return value;
    };
    return {
        static_cast<std::int32_t>(u64_big(0) % static_cast<std::uint64_t>(std::max(1, height))),
        static_cast<std::int32_t>(u64_big(8) % static_cast<std::uint64_t>(std::max(1, width))),
        static_cast<std::int64_t>(u64_big(16) & 0x7FFFFFFFFFFFFFFFULL),
    };
}

float cue_observation_value(
    float cue, float dropout, std::int64_t seed, std::int32_t y, std::int32_t x
) noexcept {
    cue = std::isfinite(cue) ? std::clamp(cue, 0.0F, 1.0F) : 0.0F;
    dropout = std::isfinite(dropout) ? std::clamp(dropout, 0.0F, 1.0F) : 0.0F;
    if (dropout <= 0.0F) return cue;
    if (dropout >= 1.0F) return 0.0F;
    const std::uint32_t mixed = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(static_cast<std::int64_t>(y) * 1103515245LL)
        + static_cast<std::uint64_t>(static_cast<std::int64_t>(x) * 12345LL)
        + static_cast<std::uint32_t>(seed) + 0x9E3779B9ULL);
    const double unit = static_cast<double>(mixed) / 4294967296.0;
    return unit >= static_cast<double>(dropout) ? cue : 0.0F;
}

Status advance_tick(
    const Config& c, StateView s, ScheduleState* schedule, std::int64_t abs_step,
    bool cancel_requested, TickMetrics* out, ScheduledEventRow* event
) noexcept {
    if (out == nullptr || schedule == nullptr) return Status::kInvalidState;
    if (cancel_requested) return Status::kCancelled;
    if (schedule->next_abs_step != abs_step
        || abs_step < 0
        || abs_step == std::numeric_limits<std::int64_t>::max()) {
        return Status::kInvalidState;
    }
    const Status state_status = validate_state(c, as_const(s));
    if (state_status != Status::kOk) return state_status;
    TickMetrics pending{};
    ScheduledEventRow pending_event{};
    if (!c.enabled) {
        schedule->next_abs_step = abs_step + 1;
        *out = pending;
        return Status::kOk;
    }

    for (std::size_t i = 0; i < s.cell_count; ++i) {
        const bool occupied = s.timer[i] >= 0;
        if (occupied) { ++s.age[i]; --s.timer[i]; }
        s.cue[i] = 0.0F;
        s.active[i] = 0.0F;
        const bool cue_phase = occupied && s.age[i] < c.cue_duration_steps;
        const bool hidden_phase = occupied && s.age[i] >= c.cue_duration_steps && s.age[i] < c.delay_steps;
        const bool active_phase = occupied && s.age[i] >= c.delay_steps
            && s.age[i] < c.delay_steps + c.active_duration_steps;
        const bool done = occupied && s.age[i] >= c.delay_steps + c.active_duration_steps;
        if (cue_phase && c.cue_enabled != 0) s.cue[i] = 1.0F;
        if (cue_phase || hidden_phase || active_phase) s.prior_cue[i] = 1;
        if (active_phase) {
            s.active[i] = 1.0F;
            s.intensity[i] = c.intensity;
            pending.cue_to_active_delay_sum += static_cast<double>(c.delay_steps);
            ++pending.cue_to_active_delay_count;
        }
        if (!active_phase && !cue_phase && !hidden_phase) s.intensity[i] = 0.0F;
        if (done) {
            s.cue[i] = 0.0F; s.active[i] = 0.0F; s.intensity[i] = 0.0F;
            s.age[i] = 0; s.timer[i] = -1; s.event_id[i] = 0; s.prior_cue[i] = 0;
        }
    }

    if (c.deterministic_schedule_enabled && abs_step % c.event_period_steps == 0) {
        const EventCoordinate event = stable_event_coordinate(
            schedule->base_seed, abs_step / c.event_period_steps, c.height, c.width);
        pending_event.event_id = event.event_id;
        pending_event.abs_step = abs_step;
        pending_event.center_y = event.y;
        pending_event.center_x = event.x;
        pending_event.intensity = c.intensity;
        const std::int32_t y0 = std::max(0, event.y - c.radius);
        const std::int32_t y1 = std::min(c.height, event.y + c.radius + 1);
        const std::int32_t x0 = std::max(0, event.x - c.radius);
        const std::int32_t x1 = std::min(c.width, event.x + c.radius + 1);
        for (std::int32_t y = y0; y < y1; ++y) {
            for (std::int32_t x = x0; x < x1; ++x) {
                const std::size_t i = static_cast<std::size_t>(y) * c.width + x;
                if (s.timer[i] >= 0) continue;
                s.cue[i] = c.cue_enabled != 0 ? 1.0F : 0.0F;
                s.timer[i] = c.delay_steps + c.active_duration_steps;
                s.active[i] = 0.0F; s.intensity[i] = c.intensity;
                s.age[i] = 0; s.event_id[i] = event.event_id; s.prior_cue[i] = 1;
                ++pending.event_count;
                ++pending_event.scheduled_cell_count;
            }
        }
        if (pending_event.scheduled_cell_count == 0) pending_event = {};
    }
    for (std::size_t i = 0; i < s.cell_count; ++i) {
        pending.cue_cell_count += s.cue[i] > 0.0F;
        pending.active_cell_count += s.active[i] > 0.0F;
        pending.cue_without_active_count += s.prior_cue[i] > 0 && s.active[i] <= 0.0F;
        pending.active_without_prior_cue_count += s.active[i] > 0.0F && s.prior_cue[i] <= 0;
    }
    schedule->next_abs_step = abs_step + 1;
    *out = pending;
    if (event != nullptr) *event = pending_event;
    return Status::kOk;
}

Status apply_delayed_damage(
    const Config& c, ConstStateView s, AgentStateView a, bool cancel_requested, DamageMetrics* out
) noexcept {
    if (out == nullptr) return Status::kInvalidState;
    if (cancel_requested) return Status::kCancelled;
    if (validate_state(c, s) != Status::kOk || a.alive == nullptr || a.pos_y == nullptr
        || a.pos_x == nullptr || a.health == nullptr) return Status::kInvalidState;
    for (std::size_t i = 0; i < a.agent_count; ++i) {
        if (a.alive[i] > 1U || !std::isfinite(a.health[i]) || a.health[i] < 0.0F
            || (a.damage_scale != nullptr && (!std::isfinite(a.damage_scale[i]) || a.damage_scale[i] < 0.0F))) {
            return Status::kInvalidState;
        }
        if (a.alive[i] && (a.pos_y[i] < 0 || a.pos_y[i] >= c.height || a.pos_x[i] < 0 || a.pos_x[i] >= c.width)) {
            return Status::kInvalidState;
        }
    }
    DamageMetrics pending{};
    if (!c.enabled || !c.damage_enabled) {
        *out = pending;
        return Status::kOk;
    }
    for (std::size_t i = 0; i < a.agent_count; ++i) {
        if (!a.alive[i]) continue;
        const std::size_t cell = static_cast<std::size_t>(a.pos_y[i]) * c.width + a.pos_x[i];
        if (s.active[cell] <= 0.0F) continue;
        ++pending.active_exposure_count;
        const float scale = a.damage_scale == nullptr ? 1.0F : a.damage_scale[i];
        const float damage = std::max(0.0F, s.intensity[cell] * scale);
        if (damage <= 0.0F) continue;
        ++pending.agents_damaged;
        pending.damage_total += damage;
        pending.damage_max = std::max(pending.damage_max, damage);
        a.health[i] = std::max(0.0F, a.health[i] - damage);
        if (a.health[i] <= 0.0F) { a.alive[i] = 0; ++pending.deaths; }
    }
    *out = pending;
    return Status::kOk;
}

std::size_t checkpoint_size_bytes(const Config& c) noexcept {
    if (validate_config(c) != Status::kOk) return 0;
    const std::size_t cells = static_cast<std::size_t>(c.height) * c.width;
    return 32U + cells * (4U + 4U + 4U + 4U + 4U + 8U + 1U);
}

Status write_checkpoint(
    const Config& c, ConstStateView s, const ScheduleState& schedule,
    std::uint8_t* output, std::size_t capacity, std::size_t* bytes_written
) noexcept {
    const std::size_t required = checkpoint_size_bytes(c);
    if (validate_state(c, s) != Status::kOk) return Status::kInvalidState;
    if (output == nullptr || capacity < required) return Status::kInsufficientCapacity;
    std::uint8_t* cursor = output;
    write_little(cursor, static_cast<std::uint32_t>(0x314C4457U));  // WDL1
    write_little(cursor, static_cast<std::int32_t>(kSchemaVersion));
    write_little(cursor, c.height); write_little(cursor, c.width);
    write_little(cursor, schedule.base_seed); write_little(cursor, schedule.next_abs_step);
    for (std::size_t i = 0; i < s.cell_count; ++i) write_little(cursor, s.cue[i]);
    for (std::size_t i = 0; i < s.cell_count; ++i) write_little(cursor, s.timer[i]);
    for (std::size_t i = 0; i < s.cell_count; ++i) write_little(cursor, s.active[i]);
    for (std::size_t i = 0; i < s.cell_count; ++i) write_little(cursor, s.intensity[i]);
    for (std::size_t i = 0; i < s.cell_count; ++i) write_little(cursor, s.age[i]);
    for (std::size_t i = 0; i < s.cell_count; ++i) write_little(cursor, s.event_id[i]);
    for (std::size_t i = 0; i < s.cell_count; ++i) write_little(cursor, s.prior_cue[i]);
    if (bytes_written != nullptr) *bytes_written = required;
    return Status::kOk;
}

Status restore_checkpoint(
    const Config& c, const std::uint8_t* input, std::size_t size,
    StateView s, ScheduleState* schedule
) noexcept {
    const std::size_t required = checkpoint_size_bytes(c);
    if (input == nullptr || schedule == nullptr || size != required
        || s.cell_count != static_cast<std::size_t>(c.height) * c.width || s.cue == nullptr
        || s.timer == nullptr || s.active == nullptr || s.intensity == nullptr || s.age == nullptr
        || s.event_id == nullptr || s.prior_cue == nullptr) return Status::kInvalidCheckpoint;
    const std::uint8_t* cursor = input;
    if (read_little<std::uint32_t>(cursor) != 0x314C4457U
        || read_little<std::int32_t>(cursor) != kSchemaVersion
        || read_little<std::int32_t>(cursor) != c.height
        || read_little<std::int32_t>(cursor) != c.width) return Status::kInvalidCheckpoint;
    ScheduleState candidate{};
    candidate.base_seed = read_little<std::int64_t>(cursor);
    candidate.next_abs_step = read_little<std::int64_t>(cursor);
    if (candidate.next_abs_step < 0) return Status::kInvalidCheckpoint;

    // Validate the serialized state before the first destination write.
    const std::uint8_t* probe = cursor;
    for (std::size_t i = 0; i < s.cell_count; ++i) { const float v = read_little<float>(probe); if (!std::isfinite(v) || v < 0.0F || v > 1.0F) return Status::kInvalidCheckpoint; }
    for (std::size_t i = 0; i < s.cell_count; ++i) { if (read_little<std::int32_t>(probe) < -1) return Status::kInvalidCheckpoint; }
    for (std::size_t i = 0; i < s.cell_count; ++i) { const float v = read_little<float>(probe); if (!std::isfinite(v) || v < 0.0F || v > 1.0F) return Status::kInvalidCheckpoint; }
    for (std::size_t i = 0; i < s.cell_count; ++i) { const float v = read_little<float>(probe); if (!std::isfinite(v) || v < 0.0F || v > 1.0F) return Status::kInvalidCheckpoint; }
    for (std::size_t i = 0; i < s.cell_count; ++i) { if (read_little<std::int32_t>(probe) < 0) return Status::kInvalidCheckpoint; }
    for (std::size_t i = 0; i < s.cell_count; ++i) { if (read_little<std::int64_t>(probe) < 0) return Status::kInvalidCheckpoint; }
    for (std::size_t i = 0; i < s.cell_count; ++i) { const auto v = read_little<std::int8_t>(probe); if (v != 0 && v != 1) return Status::kInvalidCheckpoint; }

    for (std::size_t i = 0; i < s.cell_count; ++i) s.cue[i] = read_little<float>(cursor);
    for (std::size_t i = 0; i < s.cell_count; ++i) s.timer[i] = read_little<std::int32_t>(cursor);
    for (std::size_t i = 0; i < s.cell_count; ++i) s.active[i] = read_little<float>(cursor);
    for (std::size_t i = 0; i < s.cell_count; ++i) s.intensity[i] = read_little<float>(cursor);
    for (std::size_t i = 0; i < s.cell_count; ++i) s.age[i] = read_little<std::int32_t>(cursor);
    for (std::size_t i = 0; i < s.cell_count; ++i) s.event_id[i] = read_little<std::int64_t>(cursor);
    for (std::size_t i = 0; i < s.cell_count; ++i) s.prior_cue[i] = read_little<std::int8_t>(cursor);
    *schedule = candidate;
    return Status::kOk;
}

}  // namespace eiel::wdl::native_v1
