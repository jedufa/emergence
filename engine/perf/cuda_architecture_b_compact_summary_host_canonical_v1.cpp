#include "cuda_architecture_b_compact_summary_host_canonical_v1.hpp"
#include "cuda_architecture_b_compact_summary_reporting_authority_v1.cuh"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <system_error>

namespace eiel::cuda_architecture_b::compact_summary_host_canonical_v1 {

namespace {

inline constexpr std::uint32_t kLedgerSealedV1 = 2;
inline constexpr std::int32_t kMaximumRowCapacityV1 =
    compact_summary_reporting_authority_v1::kMaximumDynamicRowCapacityV1;
inline constexpr std::uint32_t kMaximumLineageGroupCapacityV1 = 384;
inline constexpr std::uint8_t kBehaviorAliveEndV1 = 1U << 0;
inline constexpr std::uint8_t kBehaviorEligibleV1 = 1U << 1;
inline constexpr std::uint8_t kBehaviorKeptV1 = 1U << 2;
inline constexpr std::uint8_t kBehaviorTerminalSealedV1 = 1U << 3;
inline constexpr std::uint8_t kBehaviorKnownFlagsV1 =
    kBehaviorAliveEndV1 | kBehaviorEligibleV1 | kBehaviorKeptV1
    | kBehaviorTerminalSealedV1;

struct alignas(8) HeaderLayoutV1 {
    std::int64_t start_id_ceiling;
    std::uint16_t behavior_row_count;
    std::uint16_t lineage_record_count;
    std::uint32_t status;
};

struct alignas(8) BehaviorLayoutV1 {
    std::int64_t agent_id;
    double energy_collected_exact;
    double base_fitness;
    float injury_start;
    float injury_end;
    float aggression_bias_f32;
    std::int32_t lifespan_steps;
    std::int32_t offspring;
    std::int32_t attacks_made;
    std::uint8_t action_counts[7];
    std::uint8_t flags;
};

struct alignas(8) LineageLayoutV1 {
    std::int64_t agent_id;
    double energy;
    double energy_collected_exact;
    double episode0_pred_err_mean;
    double identity_drift_ema;
    float health;
    float injury;
    float drives[4];
    float pred_err_sum;
    float drive_jitter;
    float aggression_bias_f32;
    float exploration_bias_f32;
    std::int32_t y;
    std::int32_t x;
    std::int32_t founder_id;
    std::int32_t birth_generation;
    std::int32_t birth_tick;
    std::int32_t lineage_depth;
    std::int32_t offspring_count;
    std::int32_t survival_steps;
    std::int32_t first_repro_step;
    std::int32_t pred_err_count;
    std::int32_t patch_habitat_lineage_id;
    std::int32_t hidden_size;
    std::int32_t memory_dim;
    std::uint8_t use_gru;
    std::uint8_t reserved[3];
};

static_assert(sizeof(HeaderLayoutV1) == 16);
static_assert(sizeof(BehaviorLayoutV1) == 56);
static_assert(sizeof(LineageLayoutV1) == 136);
static_assert(offsetof(BehaviorLayoutV1, flags) == 55);
static_assert(offsetof(LineageLayoutV1, reserved) == 133);

template <typename Destination, typename Source>
Destination load_layout(const Source* source, const std::size_t index = 0) noexcept {
    Destination destination{};
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(source);
    std::memcpy(&destination, bytes + index * sizeof(Destination), sizeof(Destination));
    return destination;
}

template <typename Layout, typename Source>
bool inactive_rows_are_zero(
    const Source* source,
    const std::uint32_t active_count,
    const std::int32_t row_capacity
) noexcept {
    for (std::uint32_t row = active_count;
         row < static_cast<std::uint32_t>(row_capacity);
         ++row) {
        const Layout value = load_layout<Layout>(source, row);
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
        for (std::size_t offset = 0; offset < sizeof(Layout); ++offset) {
            if (bytes[offset] != 0U) {
                return false;
            }
        }
    }
    return true;
}

struct Sha256ContextV1 {
    std::uint64_t total_bytes;
    std::uint32_t state[8];
    std::uint32_t buffered_bytes;
    std::uint8_t buffer[64];
};

inline constexpr std::uint32_t kRoundConstantsV1[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

std::uint32_t rotate_right(const std::uint32_t value, const std::uint32_t bits) noexcept {
    return (value >> bits) | (value << (32U - bits));
}

std::uint32_t load_big_endian_u32(const std::uint8_t* bytes) noexcept {
    return (static_cast<std::uint32_t>(bytes[0]) << 24U)
        | (static_cast<std::uint32_t>(bytes[1]) << 16U)
        | (static_cast<std::uint32_t>(bytes[2]) << 8U)
        | static_cast<std::uint32_t>(bytes[3]);
}

void transform_block(Sha256ContextV1* context, const std::uint8_t* block) noexcept {
    std::uint32_t words[64];
    for (std::uint32_t index = 0; index < 16; ++index) {
        words[index] = load_big_endian_u32(block + 4U * index);
    }
    for (std::uint32_t index = 16; index < 64; ++index) {
        const std::uint32_t word_minus_15 = words[index - 15U];
        const std::uint32_t word_minus_2 = words[index - 2U];
        const std::uint32_t sigma0 = rotate_right(word_minus_15, 7U)
            ^ rotate_right(word_minus_15, 18U) ^ (word_minus_15 >> 3U);
        const std::uint32_t sigma1 = rotate_right(word_minus_2, 17U)
            ^ rotate_right(word_minus_2, 19U) ^ (word_minus_2 >> 10U);
        words[index] = words[index - 16U] + sigma0 + words[index - 7U] + sigma1;
    }

    std::uint32_t a = context->state[0];
    std::uint32_t b = context->state[1];
    std::uint32_t c = context->state[2];
    std::uint32_t d = context->state[3];
    std::uint32_t e = context->state[4];
    std::uint32_t f = context->state[5];
    std::uint32_t g = context->state[6];
    std::uint32_t h = context->state[7];
    for (std::uint32_t round = 0; round < 64; ++round) {
        const std::uint32_t sum1 = rotate_right(e, 6U) ^ rotate_right(e, 11U)
            ^ rotate_right(e, 25U);
        const std::uint32_t choose = (e & f) ^ (~e & g);
        const std::uint32_t temporary1 = h + sum1 + choose
            + kRoundConstantsV1[round] + words[round];
        const std::uint32_t sum0 = rotate_right(a, 2U) ^ rotate_right(a, 13U)
            ^ rotate_right(a, 22U);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temporary2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }
    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}

void sha256_init(Sha256ContextV1* context) noexcept {
    context->total_bytes = 0;
    context->state[0] = 0x6a09e667U;
    context->state[1] = 0xbb67ae85U;
    context->state[2] = 0x3c6ef372U;
    context->state[3] = 0xa54ff53aU;
    context->state[4] = 0x510e527fU;
    context->state[5] = 0x9b05688cU;
    context->state[6] = 0x1f83d9abU;
    context->state[7] = 0x5be0cd19U;
    context->buffered_bytes = 0;
}

void sha256_update(
    Sha256ContextV1* context,
    const char* input,
    std::size_t input_bytes
) noexcept {
    context->total_bytes += input_bytes;
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(input);
    if (context->buffered_bytes != 0U) {
        const std::size_t available = 64U - context->buffered_bytes;
        const std::size_t copied = input_bytes < available ? input_bytes : available;
        std::memcpy(context->buffer + context->buffered_bytes, bytes, copied);
        context->buffered_bytes += static_cast<std::uint32_t>(copied);
        bytes += copied;
        input_bytes -= copied;
        if (context->buffered_bytes == 64U) {
            transform_block(context, context->buffer);
            context->buffered_bytes = 0;
        }
    }
    while (input_bytes >= 64U) {
        transform_block(context, bytes);
        bytes += 64U;
        input_bytes -= 64U;
    }
    if (input_bytes != 0U) {
        std::memcpy(context->buffer, bytes, input_bytes);
        context->buffered_bytes = static_cast<std::uint32_t>(input_bytes);
    }
}

void store_big_endian_u32(std::uint8_t* output, const std::uint32_t value) noexcept {
    output[0] = static_cast<std::uint8_t>(value >> 24U);
    output[1] = static_cast<std::uint8_t>(value >> 16U);
    output[2] = static_cast<std::uint8_t>(value >> 8U);
    output[3] = static_cast<std::uint8_t>(value);
}

void sha256_final(Sha256ContextV1* context, std::uint8_t output[32]) noexcept {
    const std::uint64_t message_bits = context->total_bytes * 8U;
    std::uint32_t cursor = context->buffered_bytes;
    context->buffer[cursor++] = 0x80U;
    if (cursor > 56U) {
        std::memset(context->buffer + cursor, 0, 64U - cursor);
        transform_block(context, context->buffer);
        cursor = 0;
    }
    std::memset(context->buffer + cursor, 0, 56U - cursor);
    for (std::uint32_t index = 0; index < 8; ++index) {
        context->buffer[56U + index] = static_cast<std::uint8_t>(
            message_bits >> (56U - 8U * index)
        );
    }
    transform_block(context, context->buffer);
    for (std::uint32_t index = 0; index < 8; ++index) {
        store_big_endian_u32(output + 4U * index, context->state[index]);
    }
}

struct CanonicalSinkV1 {
    Sha256ContextV1 context;
    bool valid;
};

void sink_init(CanonicalSinkV1* sink) noexcept {
    sha256_init(&sink->context);
    sink->valid = true;
}

void sink_write(
    CanonicalSinkV1* sink,
    const char* bytes,
    const std::size_t length
) noexcept {
    if (sink->valid) {
        sha256_update(&sink->context, bytes, length);
    }
}

template <std::size_t Size>
void sink_literal(CanonicalSinkV1* sink, const char (&literal)[Size]) noexcept {
    sink_write(sink, literal, Size - 1U);
}

template <typename Integer>
void sink_integer(CanonicalSinkV1* sink, const Integer value) noexcept {
    char buffer[32];
    const auto converted = std::to_chars(buffer, buffer + sizeof(buffer), value);
    if (converted.ec != std::errc{}) {
        sink->valid = false;
        return;
    }
    sink_write(sink, buffer, static_cast<std::size_t>(converted.ptr - buffer));
}

void sink_float(CanonicalSinkV1* sink, const double value) noexcept;

double canonical_behavior_aggression_bias(const float value) noexcept {
    constexpr double kDecimalScale = 100000000.0;
    const double scaled = static_cast<double>(value) * kDecimalScale;
    const double lower = std::floor(scaled);
    const double fraction = scaled - lower;
    double rounded = lower;
    if (fraction > 0.5
        || (fraction == 0.5 && std::fmod(lower, 2.0) != 0.0)) {
        rounded = lower + 1.0;
    }
    if (rounded == 0.0) {
        return std::copysign(0.0, static_cast<double>(value));
    }
    return rounded / kDecimalScale;
}

HostCanonicalStatusV1 fail(
    HostCanonicalResultV1* result,
    const HostCanonicalStatusV1 status,
    const HostCanonicalFieldV1 field,
    const std::uint32_t row = kNoFailedRowV1
) noexcept {
    result->status = static_cast<std::uint32_t>(status);
    result->failed_field = static_cast<std::uint32_t>(field);
    result->failed_row = row;
    return status;
}

bool host_is_little_endian() noexcept {
    const std::uint16_t marker = 1;
    return *reinterpret_cast<const std::uint8_t*>(&marker) == 1U;
}

HostCanonicalStatusV1 validate_authority(
    const std::int64_t generation,
    const HeaderLayoutV1& header,
    const BehaviorAuthorityRowV1* behavior_rows,
    const LineageAuthorityRowV1* lineage_rows,
    const std::int32_t row_capacity,
    HostCanonicalResultV1* result
) noexcept {
    if (generation < 0) {
        return fail(
            result, HostCanonicalStatusV1::kInvalidGeneration,
            HostCanonicalFieldV1::kHeader
        );
    }
    if (header.status != kLedgerSealedV1) {
        return fail(
            result, HostCanonicalStatusV1::kLedgerNotSealed,
            HostCanonicalFieldV1::kHeader
        );
    }
    if (header.start_id_ceiling < 0
        || header.behavior_row_count > static_cast<std::uint32_t>(row_capacity)
        || header.lineage_record_count > header.behavior_row_count) {
        return fail(
            result, HostCanonicalStatusV1::kInconsistentHeader,
            HostCanonicalFieldV1::kHeader
        );
    }
    if (!inactive_rows_are_zero<BehaviorLayoutV1>(
            behavior_rows, header.behavior_row_count, row_capacity
        )
        || !inactive_rows_are_zero<LineageLayoutV1>(
            lineage_rows, header.lineage_record_count, row_capacity
        )) {
        return fail(
            result, HostCanonicalStatusV1::kInactiveStorageNotZero,
            HostCanonicalFieldV1::kHeader
        );
    }

    std::int64_t previous_behavior_id = -1;
    for (std::uint32_t index = 0; index < header.behavior_row_count; ++index) {
        const BehaviorLayoutV1 row = load_layout<BehaviorLayoutV1>(behavior_rows, index);
        if (row.agent_id < 0 || row.agent_id <= previous_behavior_id) {
            return fail(
                result, HostCanonicalStatusV1::kBehaviorOrder,
                HostCanonicalFieldV1::kBehaviorAgentId, index
            );
        }
        if ((row.flags & static_cast<std::uint8_t>(~kBehaviorKnownFlagsV1)) != 0U
            || (row.flags & kBehaviorTerminalSealedV1) == 0U
            || ((row.flags & kBehaviorEligibleV1) != 0U
                && row.agent_id >= header.start_id_ceiling)
            || ((row.flags & kBehaviorKeptV1) != 0U
                && (row.flags & kBehaviorEligibleV1) == 0U)) {
            return fail(
                result, HostCanonicalStatusV1::kBehaviorFlags,
                HostCanonicalFieldV1::kBehaviorFlags, index
            );
        }
        if (row.lifespan_steps < 0 || row.offspring < 0 || row.attacks_made < 0) {
            return fail(
                result, HostCanonicalStatusV1::kBehaviorCounter,
                HostCanonicalFieldV1::kBehaviorCounters, index
            );
        }
        const struct {
            double value;
            HostCanonicalFieldV1 field;
        } finite_values[] = {
            {row.energy_collected_exact, HostCanonicalFieldV1::kBehaviorEnergyCollected},
            {row.base_fitness, HostCanonicalFieldV1::kBehaviorFitness},
            {static_cast<double>(row.injury_start), HostCanonicalFieldV1::kBehaviorInjuryStart},
            {static_cast<double>(row.injury_end), HostCanonicalFieldV1::kBehaviorInjuryEnd},
            {static_cast<double>(row.aggression_bias_f32), HostCanonicalFieldV1::kBehaviorAggressionBias},
        };
        for (const auto& candidate : finite_values) {
            if (!std::isfinite(candidate.value)) {
                return fail(
                    result, HostCanonicalStatusV1::kBehaviorNonFinite,
                    candidate.field, index
                );
            }
        }
        previous_behavior_id = row.agent_id;
    }

    std::int64_t previous_lineage_id = -1;
    std::uint32_t alive_index = 0;
    for (std::uint32_t index = 0; index < header.lineage_record_count; ++index) {
        const LineageLayoutV1 row = load_layout<LineageLayoutV1>(lineage_rows, index);
        if (row.agent_id < 0 || row.founder_id < 0 || row.agent_id <= previous_lineage_id) {
            return fail(
                result, HostCanonicalStatusV1::kLineageOrder,
                HostCanonicalFieldV1::kLineageAgentId, index
            );
        }
        if (row.use_gru > 1U || row.reserved[0] != 0U || row.reserved[1] != 0U
            || row.reserved[2] != 0U) {
            return fail(
                result, HostCanonicalStatusV1::kLineageStorage,
                HostCanonicalFieldV1::kLineageStorage, index
            );
        }
        if (row.y < 0 || row.x < 0 || row.birth_generation < 0 || row.birth_tick < 0
            || row.lineage_depth < 0 || row.offspring_count < 0
            || row.survival_steps < 0 || row.first_repro_step < -1
            || row.pred_err_count < 0 || row.patch_habitat_lineage_id < 0
            || row.hidden_size < 0 || row.memory_dim < 0) {
            return fail(
                result, HostCanonicalStatusV1::kLineageCounter,
                HostCanonicalFieldV1::kLineageCounters, index
            );
        }
        const struct {
            double value;
            HostCanonicalFieldV1 field;
        } finite_values[] = {
            {row.energy, HostCanonicalFieldV1::kLineageEnergy},
            {row.energy_collected_exact, HostCanonicalFieldV1::kLineageEnergyCollected},
            {row.episode0_pred_err_mean, HostCanonicalFieldV1::kLineageEpisode0PredErrMean},
            {row.identity_drift_ema, HostCanonicalFieldV1::kLineageIdentityDriftEma},
            {static_cast<double>(row.health), HostCanonicalFieldV1::kLineageHealth},
            {static_cast<double>(row.injury), HostCanonicalFieldV1::kLineageInjury},
            {static_cast<double>(row.drives[0]), HostCanonicalFieldV1::kLineageDrives},
            {static_cast<double>(row.drives[1]), HostCanonicalFieldV1::kLineageDrives},
            {static_cast<double>(row.drives[2]), HostCanonicalFieldV1::kLineageDrives},
            {static_cast<double>(row.drives[3]), HostCanonicalFieldV1::kLineageDrives},
            {static_cast<double>(row.pred_err_sum), HostCanonicalFieldV1::kLineagePredErrSum},
            {static_cast<double>(row.drive_jitter), HostCanonicalFieldV1::kLineageDriveJitter},
            {static_cast<double>(row.aggression_bias_f32), HostCanonicalFieldV1::kLineageAggressionBias},
            {static_cast<double>(row.exploration_bias_f32), HostCanonicalFieldV1::kLineageExplorationBias},
        };
        for (const auto& candidate : finite_values) {
            if (!std::isfinite(candidate.value)) {
                return fail(
                    result, HostCanonicalStatusV1::kLineageNonFinite,
                    candidate.field, index
                );
            }
        }
        bool matched_alive_row = false;
        while (alive_index < header.behavior_row_count) {
            const BehaviorLayoutV1 behavior =
                load_layout<BehaviorLayoutV1>(behavior_rows, alive_index);
            ++alive_index;
            if ((behavior.flags & kBehaviorAliveEndV1) != 0U) {
                if (behavior.agent_id != row.agent_id) {
                    return fail(
                        result, HostCanonicalStatusV1::kLineageBehaviorMismatch,
                        HostCanonicalFieldV1::kLineageAgentId, index
                    );
                }
                matched_alive_row = true;
                break;
            }
        }
        if (!matched_alive_row) {
            return fail(
                result, HostCanonicalStatusV1::kLineageBehaviorMismatch,
                HostCanonicalFieldV1::kLineageAgentId, index
            );
        }
        previous_lineage_id = row.agent_id;
    }
    while (alive_index < header.behavior_row_count) {
        const BehaviorLayoutV1 behavior =
            load_layout<BehaviorLayoutV1>(behavior_rows, alive_index);
        ++alive_index;
        if ((behavior.flags & kBehaviorAliveEndV1) != 0U) {
            return fail(
                result, HostCanonicalStatusV1::kLineageBehaviorMismatch,
                HostCanonicalFieldV1::kLineageAgentId
            );
        }
    }
    return HostCanonicalStatusV1::kSuccess;
}

void write_behavior_digest(
    const std::int64_t generation,
    const HeaderLayoutV1& header,
    const BehaviorAuthorityRowV1* behavior_rows,
    CanonicalSinkV1* sink
) noexcept {
    sink_literal(
        sink,
        "{\"fields\":[\"generation\",\"agent_id\",\"cohort\",\"move_n\","
        "\"move_s\",\"move_e\",\"move_w\",\"gather\",\"rest\",\"attack\","
        "\"lifespan_steps\",\"energy_collected\",\"offspring\",\"attacks_made\","
        "\"fitness\",\"alive_end\",\"eligible_for_selection\",\"kept_next\","
        "\"injury_end\",\"injury_delta\",\"aggression_bias\"],\"rows\":["
    );
    for (std::uint32_t index = 0; index < header.behavior_row_count; ++index) {
        const BehaviorLayoutV1 row = load_layout<BehaviorLayoutV1>(behavior_rows, index);
        if (index != 0U) {
            sink_literal(sink, ",");
        }
        sink_literal(sink, "[");
        sink_integer(sink, generation);
        sink_literal(sink, ",");
        sink_integer(sink, row.agent_id);
        if (row.agent_id < header.start_id_ceiling) {
            sink_literal(sink, ",\"start\",");
        } else {
            sink_literal(sink, ",\"born\",");
        }
        for (std::uint32_t action = 0; action < 7; ++action) {
            sink_integer(sink, static_cast<std::uint32_t>(row.action_counts[action]));
            sink_literal(sink, ",");
        }
        sink_integer(sink, row.lifespan_steps);
        sink_literal(sink, ",");
        sink_float(sink, row.energy_collected_exact);
        sink_literal(sink, ",");
        sink_integer(sink, row.offspring);
        sink_literal(sink, ",");
        sink_integer(sink, row.attacks_made);
        sink_literal(sink, ",");
        sink_float(sink, row.base_fitness);
        sink_literal(sink, ",");
        sink_integer(sink, (row.flags & kBehaviorAliveEndV1) != 0U ? 1 : 0);
        sink_literal(sink, ",");
        sink_integer(sink, (row.flags & kBehaviorEligibleV1) != 0U ? 1 : 0);
        sink_literal(sink, ",");
        sink_integer(sink, (row.flags & kBehaviorKeptV1) != 0U ? 1 : 0);
        sink_literal(sink, ",");
        sink_float(sink, static_cast<double>(row.injury_end));
        sink_literal(sink, ",");
        sink_float(
            sink,
            static_cast<double>(row.injury_end) - static_cast<double>(row.injury_start)
        );
        sink_literal(sink, ",");
        sink_float(sink, canonical_behavior_aggression_bias(row.aggression_bias_f32));
        sink_literal(sink, "]");
    }
    sink_literal(
        sink,
        "],\"schema_version\":\"eiel_behavior_rows_digest_v1\"}"
    );
}

void write_lineage_stats(const LineageLayoutV1& row, CanonicalSinkV1* sink) noexcept {
    sink_literal(sink, "{\"birth_generation\":");
    sink_integer(sink, row.birth_generation);
    sink_literal(sink, ",\"birth_tick\":");
    sink_integer(sink, row.birth_tick);
    sink_literal(sink, ",\"drive_jitter\":");
    sink_float(sink, static_cast<double>(row.drive_jitter));
    sink_literal(sink, ",\"ecological_role\":\"general_agent\",\"energy_collected\":");
    sink_float(sink, row.energy_collected_exact);
    sink_literal(sink, ",\"episode0_pred_err_mean\":");
    sink_float(sink, row.episode0_pred_err_mean);
    sink_literal(sink, ",\"first_repro_step\":");
    if (row.first_repro_step < 0) {
        sink_literal(sink, "null");
    } else {
        sink_integer(sink, row.first_repro_step);
    }
    sink_literal(sink, ",\"founder_id\":");
    sink_integer(sink, row.founder_id);
    sink_literal(sink, ",\"identity_drift_ema\":");
    sink_float(sink, row.identity_drift_ema);
    sink_literal(sink, ",\"lineage_depth\":");
    sink_integer(sink, row.lineage_depth);
    sink_literal(sink, ",\"lineage_id\":\"L");
    char founder[16];
    const auto converted = std::to_chars(founder, founder + sizeof(founder), row.founder_id);
    if (converted.ec != std::errc{}) {
        sink->valid = false;
        return;
    }
    const std::size_t founder_length = static_cast<std::size_t>(converted.ptr - founder);
    for (std::size_t padding = founder_length; padding < 8U; ++padding) {
        sink_literal(sink, "0");
    }
    sink_write(sink, founder, founder_length);
    sink_literal(sink, "\",\"offspring_count\":");
    sink_integer(sink, row.offspring_count);
    sink_literal(sink, ",\"parent_ids\":[],\"parent_lineage_ids\":[],\"patch_habitat_lineage_id\":");
    sink_integer(sink, row.patch_habitat_lineage_id);
    sink_literal(sink, ",\"pred_err_count\":");
    sink_integer(sink, row.pred_err_count);
    sink_literal(sink, ",\"pred_err_sum\":");
    sink_float(sink, static_cast<double>(row.pred_err_sum));
    sink_literal(sink, ",\"role_confidence\":null,\"role_source\":\"default\",\"survival_steps\":");
    sink_float(sink, static_cast<double>(row.survival_steps));
    sink_literal(sink, "}");
}

void write_lineage_digest(
    const std::int64_t generation,
    const HeaderLayoutV1& header,
    const LineageAuthorityRowV1* lineage_rows,
    CanonicalSinkV1* sink
) noexcept {
    sink_literal(sink, "{\"generation\":");
    sink_integer(sink, generation);
    sink_literal(sink, ",\"records\":[");
    for (std::uint32_t index = 0; index < header.lineage_record_count; ++index) {
        const LineageLayoutV1 row = load_layout<LineageLayoutV1>(lineage_rows, index);
        if (index != 0U) {
            sink_literal(sink, ",");
        }
        sink_literal(sink, "{\"agent_id\":");
        sink_integer(sink, row.agent_id);
        sink_literal(sink, ",\"drives\":[");
        for (std::uint32_t drive = 0; drive < 4; ++drive) {
            if (drive != 0U) {
                sink_literal(sink, ",");
            }
            sink_float(sink, static_cast<double>(row.drives[drive]));
        }
        sink_literal(sink, "],\"energy\":");
        sink_float(sink, row.energy);
        sink_literal(sink, ",\"genome\":{\"aggression_bias\":");
        sink_float(sink, canonical_behavior_aggression_bias(row.aggression_bias_f32));
        sink_literal(sink, ",\"exploration_bias\":");
        sink_float(sink, canonical_behavior_aggression_bias(row.exploration_bias_f32));
        sink_literal(sink, ",\"hidden_size\":");
        sink_integer(sink, row.hidden_size);
        sink_literal(sink, ",\"memory_dim\":");
        sink_integer(sink, row.memory_dim);
        sink_literal(sink, ",\"use_gru\":");
        if (row.use_gru != 0U) {
            sink_literal(sink, "true");
        } else {
            sink_literal(sink, "false");
        }
        sink_literal(sink, "},\"health\":");
        sink_float(sink, static_cast<double>(row.health));
        sink_literal(sink, ",\"injury\":");
        sink_float(sink, static_cast<double>(row.injury));
        sink_literal(sink, ",\"stats\":");
        write_lineage_stats(row, sink);
        sink_literal(sink, ",\"x\":");
        sink_integer(sink, row.x);
        sink_literal(sink, ",\"y\":");
        sink_integer(sink, row.y);
        sink_literal(sink, "}");
    }
    sink_literal(
        sink,
        "],\"schema_version\":\"eiel_lineage_records_digest_v1\"}"
    );
}

}  // namespace

namespace detail {

static bool format_python_json_float_v1(
    const double value,
    char* output,
    const std::size_t capacity,
    std::size_t* output_length
) noexcept {
    if (output == nullptr || output_length == nullptr || !std::isfinite(value)) {
        return false;
    }
    char shortest[64];
    const auto converted = std::to_chars(
        shortest, shortest + sizeof(shortest), value, std::chars_format::general
    );
    if (converted.ec != std::errc{}) {
        return false;
    }
    const std::string_view lexeme(
        shortest, static_cast<std::size_t>(converted.ptr - shortest)
    );
    const std::size_t exponent_marker = lexeme.find('e');
    if (exponent_marker == std::string_view::npos) {
        const bool needs_fraction = lexeme.find('.') == std::string_view::npos;
        const std::size_t required = lexeme.size() + (needs_fraction ? 2U : 0U);
        if (required > capacity) {
            return false;
        }
        std::memcpy(output, lexeme.data(), lexeme.size());
        if (needs_fraction) {
            output[lexeme.size()] = '.';
            output[lexeme.size() + 1U] = '0';
        }
        *output_length = required;
        return true;
    }

    std::int32_t exponent = 0;
    const char* exponent_begin = lexeme.data() + exponent_marker + 1U;
    const char* exponent_end = lexeme.data() + lexeme.size();
    if (exponent_begin != exponent_end && *exponent_begin == '+') {
        ++exponent_begin;
    }
    const auto exponent_result = std::from_chars(exponent_begin, exponent_end, exponent);
    if (exponent_result.ec != std::errc{} || exponent_result.ptr != exponent_end) {
        return false;
    }
    if (exponent < -4 || exponent >= 16) {
        if (lexeme.size() > capacity) {
            return false;
        }
        std::memcpy(output, lexeme.data(), lexeme.size());
        *output_length = lexeme.size();
        return true;
    }

    const bool negative = lexeme[0] == '-';
    const std::size_t mantissa_begin = negative ? 1U : 0U;
    char digits[32];
    std::size_t digit_count = 0;
    std::int32_t digits_before_point = 0;
    bool saw_point = false;
    for (std::size_t index = mantissa_begin; index < exponent_marker; ++index) {
        if (lexeme[index] == '.') {
            saw_point = true;
            continue;
        }
        if (!saw_point) {
            ++digits_before_point;
        }
        digits[digit_count++] = lexeme[index];
    }
    const std::int32_t decimal_position = digits_before_point + exponent;
    std::size_t cursor = 0;
    if (negative) {
        if (cursor == capacity) {
            return false;
        }
        output[cursor++] = '-';
    }
    if (decimal_position <= 0) {
        const std::size_t required = cursor + 2U
            + static_cast<std::size_t>(-decimal_position) + digit_count;
        if (required > capacity) {
            return false;
        }
        output[cursor++] = '0';
        output[cursor++] = '.';
        for (std::int32_t zero = 0; zero < -decimal_position; ++zero) {
            output[cursor++] = '0';
        }
        std::memcpy(output + cursor, digits, digit_count);
        cursor += digit_count;
    } else if (static_cast<std::size_t>(decimal_position) >= digit_count) {
        const std::size_t zero_count =
            static_cast<std::size_t>(decimal_position) - digit_count;
        const std::size_t required = cursor + digit_count + zero_count + 2U;
        if (required > capacity) {
            return false;
        }
        std::memcpy(output + cursor, digits, digit_count);
        cursor += digit_count;
        for (std::size_t zero = 0; zero < zero_count; ++zero) {
            output[cursor++] = '0';
        }
        output[cursor++] = '.';
        output[cursor++] = '0';
    } else {
        const std::size_t point = static_cast<std::size_t>(decimal_position);
        const std::size_t required = cursor + digit_count + 1U;
        if (required > capacity) {
            return false;
        }
        std::memcpy(output + cursor, digits, point);
        cursor += point;
        output[cursor++] = '.';
        std::memcpy(output + cursor, digits + point, digit_count - point);
        cursor += digit_count - point;
    }
    *output_length = cursor;
    return true;
}

}  // namespace detail

namespace {

void sink_float(CanonicalSinkV1* sink, const double value) noexcept {
    char buffer[96];
    std::size_t length = 0;
    if (!detail::format_python_json_float_v1(value, buffer, sizeof(buffer), &length)) {
        sink->valid = false;
        return;
    }
    sink_write(sink, buffer, length);
}

bool validate_lineage_metrics_input(
    const LineageMetricsInputV1& input,
    HostCanonicalResultV1* result
) noexcept {
    const LineageMetricsConfigV1& config = input.config;
    for (const std::uint8_t byte : config.reserved) {
        if (byte != 0U) {
            fail(
                result, HostCanonicalStatusV1::kLineageMetricsConfig,
                HostCanonicalFieldV1::kLineageMetricsConfig
            );
            return false;
        }
    }
    if (config.metrics_defined > 1U
        || !std::isfinite(config.population_dominance_threshold)
        || !std::isfinite(config.entropy_minimum)
        || !std::isfinite(config.reproduction_dominance_threshold)) {
        fail(
            result, HostCanonicalStatusV1::kLineageMetricsConfig,
            HostCanonicalFieldV1::kLineageMetricsConfig
        );
        return false;
    }
    if (input.initial_state.collapse_streak < 0
        || input.initial_state.population_dominance_streak < 0
        || input.initial_state.reproduction_dominance_streak < 0) {
        fail(
            result, HostCanonicalStatusV1::kLineageMetricsState,
            HostCanonicalFieldV1::kLineageMetricsState
        );
        return false;
    }
    return true;
}

void compute_behavior_aggregates(
    const HeaderLayoutV1& header,
    const BehaviorAuthorityRowV1* behavior_rows,
    BehaviorAggregateOutputV1* output
) noexcept {
    output->row_count = header.behavior_row_count;
    for (std::uint32_t index = 0; index < header.behavior_row_count; ++index) {
        const BehaviorLayoutV1 row = load_layout<BehaviorLayoutV1>(behavior_rows, index);
        if (row.agent_id < header.start_id_ceiling) {
            ++output->cohort_size;
            if ((row.flags & kBehaviorAliveEndV1) != 0U) {
                ++output->cohort_survivors;
            }
        }
        output->move_n += row.action_counts[0];
        output->move_s += row.action_counts[1];
        output->move_e += row.action_counts[2];
        output->move_w += row.action_counts[3];
        output->gather += row.action_counts[4];
        output->rest += row.action_counts[5];
        output->attack += row.action_counts[6];
    }
    output->move_total = output->move_n + output->move_s
        + output->move_e + output->move_w;
    output->action_total = output->move_total + output->gather
        + output->rest + output->attack;
}

struct LineageGroupV1 {
    std::int32_t founder_id;
    std::uint32_t population_count;
    std::uint64_t reproduction_count;
    std::int32_t maximum_depth;
    std::uint8_t text_record[kDominantLineageTextRecordBytesV1];
};

bool make_lineage_text_record(
    const std::int32_t founder_id,
    std::uint8_t output[kDominantLineageTextRecordBytesV1]
) noexcept {
    char digits[16];
    const auto converted = std::to_chars(digits, digits + sizeof(digits), founder_id);
    if (converted.ec != std::errc{}) {
        return false;
    }
    const std::size_t digit_count = static_cast<std::size_t>(converted.ptr - digits);
    const std::size_t zero_count = digit_count < 8U ? 8U - digit_count : 0U;
    const std::size_t text_length = 1U + zero_count + digit_count;
    if (text_length > kDominantLineageTextRecordBytesV1 - 1U) {
        return false;
    }
    output[0] = static_cast<std::uint8_t>(text_length);
    output[1] = static_cast<std::uint8_t>('L');
    std::memset(output + 2U, '0', zero_count);
    std::memcpy(output + 2U + zero_count, digits, digit_count);
    return true;
}

int compare_lineage_text_records(
    const std::uint8_t left[kDominantLineageTextRecordBytesV1],
    const std::uint8_t right[kDominantLineageTextRecordBytesV1]
) noexcept {
    const std::size_t left_length = left[0];
    const std::size_t right_length = right[0];
    const std::size_t common_length =
        left_length < right_length ? left_length : right_length;
    const int common = std::memcmp(left + 1U, right + 1U, common_length);
    if (common != 0) {
        return common;
    }
    if (left_length < right_length) {
        return -1;
    }
    return left_length > right_length ? 1 : 0;
}

bool increment_streak(const std::int64_t initial, std::int64_t* output) noexcept {
    if (initial == std::numeric_limits<std::int64_t>::max()) {
        return false;
    }
    *output = initial + 1;
    return true;
}

HostCanonicalStatusV1 compute_lineage_summary(
    const HeaderLayoutV1& header,
    const LineageAuthorityRowV1* lineage_rows,
    const LineageMetricsInputV1& input,
    HostCanonicalResultV1* result
) noexcept {
    LineageSummaryOutputV1* const output = &result->lineage;
    output->record_count = header.lineage_record_count;
    result->lineage_state_after = input.initial_state;
    if (input.config.metrics_defined == 0U) {
        return HostCanonicalStatusV1::kSuccess;
    }

    LineageGroupV1 groups[kMaximumLineageGroupCapacityV1]{};
    std::uint32_t group_count = 0;
    for (std::uint32_t index = 0; index < header.lineage_record_count; ++index) {
        const LineageLayoutV1 row = load_layout<LineageLayoutV1>(lineage_rows, index);
        std::uint32_t group_index = 0;
        while (group_index < group_count
               && groups[group_index].founder_id != row.founder_id) {
            ++group_index;
        }
        if (group_index == group_count) {
            if (group_count >= kMaximumLineageGroupCapacityV1) {
                return fail(
                    result,
                    HostCanonicalStatusV1::kAggregateOverflow,
                    HostCanonicalFieldV1::kLineageAggregates,
                    index
                );
            }
            LineageGroupV1& group = groups[group_count];
            group.founder_id = row.founder_id;
            if (!make_lineage_text_record(row.founder_id, group.text_record)) {
                return fail(
                    result, HostCanonicalStatusV1::kNumberFormatting,
                    HostCanonicalFieldV1::kLineageAggregates, index
                );
            }
            ++group_count;
        }
        LineageGroupV1& group = groups[group_index];
        ++group.population_count;
        group.reproduction_count += static_cast<std::uint32_t>(row.offspring_count);
        if (row.lineage_depth > group.maximum_depth) {
            group.maximum_depth = row.lineage_depth;
        }
    }

    output->active_lineages = group_count;
    output->compact_abi_v2_defined_mask =
        kCompactAbiV2DefinedLineageActiveV1
        | kCompactAbiV2DefinedLineageDominantPopulationShareV1
        | kCompactAbiV2DefinedLineageTop3ShareV1
        | kCompactAbiV2DefinedLineagePopulationEntropyV1
        | kCompactAbiV2DefinedLineageRarePersistentV1
        | kCompactAbiV2DefinedLineageCollapseWarningV1
        | kCompactAbiV2DefinedLineageCollapseStreakV1;

    std::uint32_t dominant_group = 0;
    std::uint32_t top_population_counts[3]{};
    std::uint64_t reproduction_total = 0;
    std::uint64_t dominant_reproduction_count = 0;
    const double population_total = static_cast<double>(header.lineage_record_count);
    for (std::uint32_t index = 0; index < group_count; ++index) {
        const LineageGroupV1& group = groups[index];
        if (index == 0U
            || group.population_count > groups[dominant_group].population_count
            || (group.population_count == groups[dominant_group].population_count
                && compare_lineage_text_records(
                       group.text_record, groups[dominant_group].text_record
                   ) < 0)) {
            dominant_group = index;
        }
        for (std::uint32_t position = 0; position < 3U; ++position) {
            if (group.population_count > top_population_counts[position]) {
                for (std::uint32_t shift = 2U; shift > position; --shift) {
                    top_population_counts[shift] = top_population_counts[shift - 1U];
                }
                top_population_counts[position] = group.population_count;
                break;
            }
        }
        const double probability =
            static_cast<double>(group.population_count) / population_total;
        output->population_entropy -= probability * std::log(probability);
        if (probability <= 0.10 && group.maximum_depth >= 2) {
            ++output->rare_persistent_lineages;
        }
        reproduction_total += group.reproduction_count;
        if (group.reproduction_count > dominant_reproduction_count) {
            dominant_reproduction_count = group.reproduction_count;
        }
    }

    if (group_count != 0U) {
        const LineageGroupV1& dominant = groups[dominant_group];
        output->dominant_population_share =
            static_cast<double>(dominant.population_count) / population_total;
        const std::uint64_t top3_count =
            static_cast<std::uint64_t>(top_population_counts[0])
            + top_population_counts[1] + top_population_counts[2];
        output->top3_population_share =
            static_cast<double>(top3_count) / population_total;
        std::memcpy(
            output->dominant_lineage_text_record,
            dominant.text_record,
            kDominantLineageTextRecordBytesV1
        );
        output->compact_abi_v2_defined_mask |=
            kCompactAbiV2DefinedLineageDominantTextV1;
    }

    if (reproduction_total != 0U) {
        const double total = static_cast<double>(reproduction_total);
        for (std::uint32_t index = 0; index < group_count; ++index) {
            if (groups[index].reproduction_count == 0U) {
                continue;
            }
            const double probability =
                static_cast<double>(groups[index].reproduction_count) / total;
            output->reproduction_entropy -= probability * std::log(probability);
        }
        output->dominant_reproduction_share =
            static_cast<double>(dominant_reproduction_count) / total;
        output->compact_abi_v2_defined_mask |=
            kCompactAbiV2DefinedLineageReproductionEntropyV1
            | kCompactAbiV2DefinedLineageDominantReproductionShareV1;
    }

    const LineageMetricsConfigV1& config = input.config;
    const std::int64_t population_streak_required =
        config.population_dominance_streak_required < 1
        ? 1 : config.population_dominance_streak_required;
    const std::int64_t reproduction_streak_required =
        config.reproduction_dominance_streak_required < 1
        ? 1 : config.reproduction_dominance_streak_required;
    const std::int64_t minimum_active =
        config.minimum_active_lineages < 0 ? 0 : config.minimum_active_lineages;
    const bool population_dominant_now = group_count != 0U
        && output->dominant_population_share >= config.population_dominance_threshold;
    if (population_dominant_now) {
        if (!increment_streak(
                input.initial_state.population_dominance_streak,
                &result->lineage_state_after.population_dominance_streak
            )) {
            return fail(
                result, HostCanonicalStatusV1::kAggregateOverflow,
                HostCanonicalFieldV1::kLineageMetricsState
            );
        }
    } else {
        result->lineage_state_after.population_dominance_streak = 0;
    }
    const bool reproduction_dominant_now = reproduction_total != 0U
        && output->dominant_reproduction_share
            >= config.reproduction_dominance_threshold;
    if (reproduction_dominant_now) {
        if (!increment_streak(
                input.initial_state.reproduction_dominance_streak,
                &result->lineage_state_after.reproduction_dominance_streak
            )) {
            return fail(
                result, HostCanonicalStatusV1::kAggregateOverflow,
                HostCanonicalFieldV1::kLineageMetricsState
            );
        }
    } else {
        result->lineage_state_after.reproduction_dominance_streak = 0;
    }

    const bool low_active_lineages =
        static_cast<std::int64_t>(group_count) < minimum_active;
    const bool low_entropy = group_count != 0U
        && output->population_entropy < config.entropy_minimum;
    const bool population_dominant =
        result->lineage_state_after.population_dominance_streak
        >= population_streak_required;
    const bool reproduction_dominant =
        result->lineage_state_after.reproduction_dominance_streak
        >= reproduction_streak_required;
    const bool collapse_warning = low_active_lineages || low_entropy
        || population_dominant || reproduction_dominant;
    output->collapse_warning_code = collapse_warning
        ? kCompactAbiV2StatusTrueV1 : kCompactAbiV2StatusFalseV1;
    if (collapse_warning) {
        if (!increment_streak(
                input.initial_state.collapse_streak,
                &result->lineage_state_after.collapse_streak
            )) {
            return fail(
                result, HostCanonicalStatusV1::kAggregateOverflow,
                HostCanonicalFieldV1::kLineageMetricsState
            );
        }
    } else {
        result->lineage_state_after.collapse_streak = 0;
    }
    output->collapse_streak = result->lineage_state_after.collapse_streak;
    return HostCanonicalStatusV1::kSuccess;
}

}  // namespace

HostCanonicalStatusV1 canonicalize_retained_summary_v1(
    const std::int64_t generation,
    const GenerationAuthorityHeaderV1* header,
    const BehaviorAuthorityRowV1* behavior_rows,
    const LineageAuthorityRowV1* lineage_rows,
    const std::int32_t row_capacity,
    const LineageMetricsInputV1* lineage_metrics_input,
    HostCanonicalResultV1* result
) noexcept {
    if (result == nullptr) {
        return HostCanonicalStatusV1::kNullArgument;
    }
    std::memset(result, 0, sizeof(*result));
    result->failed_row = kNoFailedRowV1;
    if (header == nullptr || behavior_rows == nullptr || lineage_rows == nullptr
        || lineage_metrics_input == nullptr) {
        return fail(
            result, HostCanonicalStatusV1::kNullArgument,
            HostCanonicalFieldV1::kHeader
        );
    }
    if (!host_is_little_endian()) {
        return fail(
            result, HostCanonicalStatusV1::kUnsupportedHostEndian,
            HostCanonicalFieldV1::kHeader
        );
    }
    if (row_capacity <= 0 || row_capacity > kMaximumRowCapacityV1) {
        return fail(
            result, HostCanonicalStatusV1::kInvalidRowCapacity,
            HostCanonicalFieldV1::kHeader
        );
    }
    if (!validate_lineage_metrics_input(*lineage_metrics_input, result)) {
        return static_cast<HostCanonicalStatusV1>(result->status);
    }

    const HeaderLayoutV1 header_value = load_layout<HeaderLayoutV1>(header);
    const HostCanonicalStatusV1 validation = validate_authority(
        generation, header_value, behavior_rows, lineage_rows, row_capacity, result
    );
    if (validation != HostCanonicalStatusV1::kSuccess) {
        return validation;
    }

    compute_behavior_aggregates(header_value, behavior_rows, &result->behavior);
    const HostCanonicalStatusV1 summary_status = compute_lineage_summary(
        header_value, lineage_rows, *lineage_metrics_input, result
    );
    if (summary_status != HostCanonicalStatusV1::kSuccess) {
        std::memset(result->behavior_rows_digest_sha256, 0, kDigestBytesV1);
        std::memset(result->lineage_records_digest_sha256, 0, kDigestBytesV1);
        return summary_status;
    }

    CanonicalSinkV1 behavior_sink{};
    sink_init(&behavior_sink);
    write_behavior_digest(generation, header_value, behavior_rows, &behavior_sink);
    if (!behavior_sink.valid) {
        return fail(
            result, HostCanonicalStatusV1::kNumberFormatting,
            HostCanonicalFieldV1::kNone
        );
    }
    sha256_final(&behavior_sink.context, result->behavior_rows_digest_sha256);

    CanonicalSinkV1 lineage_sink{};
    sink_init(&lineage_sink);
    write_lineage_digest(generation, header_value, lineage_rows, &lineage_sink);
    if (!lineage_sink.valid) {
        std::memset(result->behavior_rows_digest_sha256, 0, kDigestBytesV1);
        return fail(
            result, HostCanonicalStatusV1::kNumberFormatting,
            HostCanonicalFieldV1::kNone
        );
    }
    sha256_final(&lineage_sink.context, result->lineage_records_digest_sha256);
    result->status = static_cast<std::uint32_t>(HostCanonicalStatusV1::kSuccess);
    return HostCanonicalStatusV1::kSuccess;
}

HostCanonicalStatusV1 canonicalize_retained_digests_v1(
    const std::int64_t generation,
    const GenerationAuthorityHeaderV1* header,
    const BehaviorAuthorityRowV1* behavior_rows,
    const LineageAuthorityRowV1* lineage_rows,
    const std::int32_t row_capacity,
    HostCanonicalResultV1* result
) noexcept {
    return canonicalize_retained_summary_v1(
        generation, header, behavior_rows, lineage_rows, row_capacity,
        &kDefaultLineageMetricsInputV1, result
    );
}

void canonicalize_retained_summary_callback_v1(void* user_data) noexcept {
    if (user_data == nullptr) {
        return;
    }
    auto* request = static_cast<HostCanonicalRequestV1*>(user_data);
    canonicalize_retained_summary_v1(
        request->generation,
        request->header,
        request->behavior_rows,
        request->lineage_rows,
        request->row_capacity,
        &request->lineage_metrics_input,
        request->result
    );
}

void canonicalize_retained_digests_callback_v1(void* user_data) noexcept {
    canonicalize_retained_summary_callback_v1(user_data);
}

}  // namespace eiel::cuda_architecture_b::compact_summary_host_canonical_v1
