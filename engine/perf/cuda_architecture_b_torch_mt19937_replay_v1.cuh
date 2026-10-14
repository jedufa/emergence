#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define EIEL_TORCH_MT19937_REPLAY_HD __host__ __device__ __forceinline__
#else
#define EIEL_TORCH_MT19937_REPLAY_HD inline
#endif

namespace eiel::cuda_architecture_b::torch_mt19937_replay_v1 {

inline constexpr char kReplaySchemaVersion[] =
    "eiel_torch_cpu_mt19937_replay_v1";
inline constexpr char kQualifiedTorchVersion[] = "2.10.0+cu128";
inline constexpr char kQualifiedTorchGitVersion[] =
    "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0";

inline constexpr std::int64_t kFullStateBytes = 5056;
inline constexpr std::int64_t kStateWords = 624;
inline constexpr std::int64_t kStateM = 397;
inline constexpr std::int64_t kInitialSeedOffset = 0;
inline constexpr std::int64_t kLeftOffset = 8;
inline constexpr std::int64_t kSeededOffset = 12;
inline constexpr std::int64_t kNextOffset = 16;
inline constexpr std::int64_t kStateOffset = 24;
inline constexpr std::int64_t kStateLaneBytes = 8;

inline constexpr std::uint32_t kMatrixA = 0x9908b0dfU;
inline constexpr std::uint32_t kUpperMask = 0x80000000U;
inline constexpr std::uint32_t kLowerMask = 0x7fffffffU;
inline constexpr std::uint32_t kFloat32MantissaMask = 0x00ffffffU;
inline constexpr std::uint32_t kFloat32HalfThreshold = 0x00800000U;

struct TorchMt19937StateV1 {
    std::uint32_t words[kStateWords];
    std::uint64_t initial_seed;
    std::int32_t left;
    std::uint32_t next;
};

EIEL_TORCH_MT19937_REPLAY_HD std::uint32_t load_u32_le_v1(
    const std::uint8_t* const bytes
) noexcept {
    return static_cast<std::uint32_t>(bytes[0])
        | (static_cast<std::uint32_t>(bytes[1]) << 8)
        | (static_cast<std::uint32_t>(bytes[2]) << 16)
        | (static_cast<std::uint32_t>(bytes[3]) << 24);
}

EIEL_TORCH_MT19937_REPLAY_HD std::uint64_t load_u64_le_v1(
    const std::uint8_t* const bytes
) noexcept {
    std::uint64_t value = 0;
    for (std::int64_t index = 7; index >= 0; --index) {
        value = (value << 8) | static_cast<std::uint64_t>(bytes[index]);
    }
    return value;
}

EIEL_TORCH_MT19937_REPLAY_HD void store_u32_le_v1(
    std::uint8_t* const bytes,
    const std::uint32_t value
) noexcept {
    bytes[0] = static_cast<std::uint8_t>(value & 0xffU);
    bytes[1] = static_cast<std::uint8_t>((value >> 8) & 0xffU);
    bytes[2] = static_cast<std::uint8_t>((value >> 16) & 0xffU);
    bytes[3] = static_cast<std::uint8_t>((value >> 24) & 0xffU);
}

EIEL_TORCH_MT19937_REPLAY_HD void store_u64_le_v1(
    std::uint8_t* const bytes,
    const std::uint64_t value
) noexcept {
    for (std::int64_t index = 0; index < 8; ++index) {
        bytes[index] = static_cast<std::uint8_t>((value >> (8 * index)) & 0xffU);
    }
}

EIEL_TORCH_MT19937_REPLAY_HD bool is_canonical_cursor_v1(
    const std::int32_t left,
    const std::uint32_t next
) noexcept {
    return (left == 1 && next == 0)
        || (
            left >= 1 && left <= kStateWords
            && next >= 1 && next <= kStateWords
            && static_cast<std::uint32_t>(left) + next == kStateWords + 1
        );
}

EIEL_TORCH_MT19937_REPLAY_HD std::uint32_t mix_bits_v1(
    const std::uint32_t first,
    const std::uint32_t second
) noexcept {
    return (first & kUpperMask) | (second & kLowerMask);
}

EIEL_TORCH_MT19937_REPLAY_HD std::uint32_t twist_word_v1(
    const std::uint32_t first,
    const std::uint32_t second
) noexcept {
    return (mix_bits_v1(first, second) >> 1)
        ^ ((second & 1U) != 0U ? kMatrixA : 0U);
}

EIEL_TORCH_MT19937_REPLAY_HD std::uint32_t temper_v1(
    std::uint32_t value
) noexcept {
    value ^= value >> 11;
    value ^= (value << 7) & 0x9d2c5680U;
    value ^= (value << 15) & 0xefc60000U;
    value ^= value >> 18;
    return value;
}

EIEL_TORCH_MT19937_REPLAY_HD void next_state_v1(
    TorchMt19937StateV1* const state
) noexcept {
    for (std::int64_t index = 0; index < kStateWords - kStateM; ++index) {
        state->words[index] =
            state->words[index + kStateM]
            ^ twist_word_v1(state->words[index], state->words[index + 1]);
    }
    for (std::int64_t index = kStateWords - kStateM; index < kStateWords - 1;
         ++index) {
        state->words[index] =
            state->words[index + kStateM - kStateWords]
            ^ twist_word_v1(state->words[index], state->words[index + 1]);
    }
    state->words[kStateWords - 1] =
        state->words[kStateM - 1]
        ^ twist_word_v1(state->words[kStateWords - 1], state->words[0]);
    state->left = static_cast<std::int32_t>(kStateWords);
    state->next = 0;
}

EIEL_TORCH_MT19937_REPLAY_HD std::uint32_t next_uint32_v1(
    TorchMt19937StateV1* const state
) noexcept {
    state->left -= 1;
    if (state->left == 0) {
        next_state_v1(state);
    }
    const std::uint32_t raw = state->words[state->next];
    state->next += 1;
    return temper_v1(raw);
}

EIEL_TORCH_MT19937_REPLAY_HD void discard_uint32_v1(
    TorchMt19937StateV1* const state,
    const std::uint64_t raw_words
) noexcept {
    for (std::uint64_t index = 0; index < raw_words; ++index) {
        state->left -= 1;
        if (state->left == 0) {
            next_state_v1(state);
        }
        state->next += 1;
    }
}

EIEL_TORCH_MT19937_REPLAY_HD std::uint32_t torch_rand_float32_bits_v1(
    const std::uint32_t raw
) noexcept {
    const std::uint32_t mantissa = raw & kFloat32MantissaMask;
    if (mantissa == 0U) {
        return 0U;
    }
    // Every k * 2^-24 value is exactly representable as binary32. Constructing
    // the bits avoids relying on host/device floating-point library behavior.
    std::uint32_t highest = 31U;
    while (((mantissa >> highest) & 1U) == 0U) {
        highest -= 1U;
    }
    const std::uint32_t exponent = highest + 103U;
    const std::uint32_t fraction =
        (mantissa << (23U - highest)) & 0x007fffffU;
    return (exponent << 23U) | fraction;
}

EIEL_TORCH_MT19937_REPLAY_HD bool torch_rand_float32_less_than_half_v1(
    const std::uint32_t raw
) noexcept {
    return (raw & kFloat32MantissaMask) < kFloat32HalfThreshold;
}

EIEL_TORCH_MT19937_REPLAY_HD std::int64_t packed_lsb0_row_bytes_v1(
    const std::int64_t logical_width
) noexcept {
    constexpr std::int64_t kMaximumSafeWidth =
        static_cast<std::int64_t>(0x7fffffffffffffffLL) - 7;
    return logical_width < 0 || logical_width > kMaximumSafeWidth
        ? -1
        : (logical_width + 7) / 8;
}

EIEL_TORCH_MT19937_REPLAY_HD bool generate_packed_crossover_mask_lsb0_v1(
    TorchMt19937StateV1* const state,
    std::uint8_t* const packed_row,
    const std::int64_t logical_width
) noexcept {
    const std::int64_t row_bytes = packed_lsb0_row_bytes_v1(logical_width);
    if (
        state == nullptr || row_bytes < 0
        || (logical_width > 0 && packed_row == nullptr)
    ) {
        return false;
    }
    for (std::int64_t index = 0; index < row_bytes; ++index) {
        packed_row[index] = 0U;
    }
    for (std::int64_t column = 0; column < logical_width; ++column) {
        const std::uint32_t raw = next_uint32_v1(state);
        if (torch_rand_float32_less_than_half_v1(raw)) {
            packed_row[column / 8] = static_cast<std::uint8_t>(
                packed_row[column / 8] | static_cast<std::uint8_t>(1U << (column % 8))
            );
        }
    }
    return true;
}

EIEL_TORCH_MT19937_REPLAY_HD bool load_canonical_state_v1(
    const std::uint8_t* const bytes,
    const std::int64_t byte_count,
    TorchMt19937StateV1* const state
) noexcept {
    if (bytes == nullptr || state == nullptr || byte_count != kFullStateBytes) {
        return false;
    }
    const std::uint32_t seeded = load_u32_le_v1(bytes + kSeededOffset);
    const std::int32_t left =
        static_cast<std::int32_t>(load_u32_le_v1(bytes + kLeftOffset));
    const std::uint64_t next_u64 = load_u64_le_v1(bytes + kNextOffset);
    if (seeded != 1U || next_u64 > static_cast<std::uint64_t>(kStateWords)) {
        return false;
    }
    const std::uint32_t next = static_cast<std::uint32_t>(next_u64);
    if (!is_canonical_cursor_v1(left, next)) {
        return false;
    }
    for (std::int64_t index = 0; index < kStateWords; ++index) {
        const std::int64_t offset = kStateOffset + index * kStateLaneBytes;
        if (
            bytes[offset + 4] != 0 || bytes[offset + 5] != 0
            || bytes[offset + 6] != 0 || bytes[offset + 7] != 0
        ) {
            return false;
        }
        state->words[index] = load_u32_le_v1(bytes + offset);
    }
    state->initial_seed = load_u64_le_v1(bytes + kInitialSeedOffset);
    state->left = left;
    state->next = next;
    return true;
}

EIEL_TORCH_MT19937_REPLAY_HD bool store_engine_state_v1(
    const TorchMt19937StateV1* const state,
    std::uint8_t* const bytes,
    const std::int64_t byte_count
) noexcept {
    if (state == nullptr || bytes == nullptr || byte_count != kFullStateBytes) {
        return false;
    }
    if (!is_canonical_cursor_v1(state->left, state->next)) {
        return false;
    }
    store_u64_le_v1(bytes + kInitialSeedOffset, state->initial_seed);
    store_u32_le_v1(bytes + kLeftOffset, static_cast<std::uint32_t>(state->left));
    store_u32_le_v1(bytes + kSeededOffset, 1U);
    store_u64_le_v1(bytes + kNextOffset, static_cast<std::uint64_t>(state->next));
    for (std::int64_t index = 0; index < kStateWords; ++index) {
        const std::int64_t offset = kStateOffset + index * kStateLaneBytes;
        store_u32_le_v1(bytes + offset, state->words[index]);
        store_u32_le_v1(bytes + offset + 4, 0U);
    }
    return true;
}

EIEL_TORCH_MT19937_REPLAY_HD bool advance_canonical_state_v1(
    std::uint8_t* const bytes,
    const std::int64_t byte_count,
    const std::uint64_t raw_words
) noexcept {
    TorchMt19937StateV1 state{};
    if (!load_canonical_state_v1(bytes, byte_count, &state)) {
        return false;
    }
    for (std::uint64_t index = 0; index < raw_words; ++index) {
        (void)next_uint32_v1(&state);
    }
    return store_engine_state_v1(&state, bytes, byte_count);
}

}  // namespace eiel::cuda_architecture_b::torch_mt19937_replay_v1

#undef EIEL_TORCH_MT19937_REPLAY_HD
