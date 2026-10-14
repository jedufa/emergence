#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define EIEL_TORCH_RNG_STATE_PACK_HD __host__ __device__ __forceinline__
#define EIEL_TORCH_RNG_STATE_PACK_DEVICE __device__ __forceinline__
#else
#define EIEL_TORCH_RNG_STATE_PACK_HD inline
#define EIEL_TORCH_RNG_STATE_PACK_DEVICE inline
#endif

namespace eiel::cuda_architecture_b::torch_rng_state_pack_v1 {

inline constexpr char kEncoding[] =
    "torch_cpu_rng_state_5056_u32_middle_v1";
inline constexpr char kAbiSchemaVersion[] =
    "eiel_torch_cpu_mt19937_state_2560_abi_v1";
inline constexpr char kAbiFingerprintSha256[] =
    "dd3966ffb664fc6bb7b6c99a42436a8945e8b0fc4a0adf15238de16d07ddf5b4";
inline constexpr char kQualifiedTorchVersion[] = "2.10.0+cu128";
inline constexpr char kQualifiedTorchGitVersion[] =
    "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0";
inline constexpr char kQualifiedTorchBuildConfigSha256[] =
    "62a3328b6570991ff3dd369f547254c82ca716de947cd8a66374b52903f6793e";
inline constexpr char kQualifiedByteOrder[] = "little";
inline constexpr std::int64_t kFullStateBytes = 5056;
inline constexpr std::int64_t kPackedStateBytes = 2560;
inline constexpr std::int64_t kLeadingFullUint64Lanes = 3;
inline constexpr std::int64_t kMiddleLowUint32Lanes = 624;
inline constexpr std::int64_t kTrailingFullUint64Lanes = 5;

inline constexpr std::int64_t kUint64Bytes = 8;
inline constexpr std::int64_t kUint32Bytes = 4;
inline constexpr std::int64_t kLeadingFullBytes =
    kLeadingFullUint64Lanes * kUint64Bytes;
inline constexpr std::int64_t kMiddleFullBytes =
    kMiddleLowUint32Lanes * kUint64Bytes;
inline constexpr std::int64_t kMiddlePackedBytes =
    kMiddleLowUint32Lanes * kUint32Bytes;
inline constexpr std::int64_t kTrailingFullByteOffset =
    kLeadingFullBytes + kMiddleFullBytes;
inline constexpr std::int64_t kTrailingPackedByteOffset =
    kLeadingFullBytes + kMiddlePackedBytes;
inline constexpr std::int64_t kOmittedPackedByteIndex = -1;

EIEL_TORCH_RNG_STATE_PACK_HD constexpr bool compatible_extents_v1(
    const std::int64_t full_state_bytes,
    const std::int64_t packed_state_bytes
) noexcept {
    return full_state_bytes == kFullStateBytes
        && packed_state_bytes == kPackedStateBytes;
}

// Returns -1 for invalid full indices and for the omitted high half of each
// middle uint64 lane. Every non-negative result is within the packed row.
EIEL_TORCH_RNG_STATE_PACK_HD constexpr std::int64_t packed_byte_index_v1(
    const std::int64_t full_byte_index
) noexcept {
    if (full_byte_index < 0 || full_byte_index >= kFullStateBytes) {
        return kOmittedPackedByteIndex;
    }
    if (full_byte_index < kLeadingFullBytes) {
        return full_byte_index;
    }
    if (full_byte_index < kTrailingFullByteOffset) {
        const std::int64_t middle_byte_index =
            full_byte_index - kLeadingFullBytes;
        const std::int64_t byte_in_lane = middle_byte_index % kUint64Bytes;
        if (byte_in_lane >= kUint32Bytes) {
            return kOmittedPackedByteIndex;
        }
        return kLeadingFullBytes
            + (middle_byte_index / kUint64Bytes) * kUint32Bytes
            + byte_in_lane;
    }
    return kTrailingPackedByteOffset
        + (full_byte_index - kTrailingFullByteOffset);
}

EIEL_TORCH_RNG_STATE_PACK_HD constexpr std::uint8_t decoded_byte_v1(
    const std::uint8_t* const packed_state,
    const std::int64_t full_byte_index
) noexcept {
    const std::int64_t packed_byte_index =
        packed_byte_index_v1(full_byte_index);
    return packed_state != nullptr && packed_byte_index >= 0
        ? packed_state[packed_byte_index]
        : std::uint8_t{0};
}

EIEL_TORCH_RNG_STATE_PACK_DEVICE bool copy_unpacked_state_v1(
    const std::uint8_t* const packed_state,
    const std::int64_t packed_state_bytes,
    std::uint8_t* const full_state,
    const std::int64_t full_state_bytes
) noexcept {
    if (
        packed_state == nullptr || full_state == nullptr
        || !compatible_extents_v1(full_state_bytes, packed_state_bytes)
    ) {
        return false;
    }
    for (std::int64_t index = 0; index < kFullStateBytes; ++index) {
        full_state[index] = decoded_byte_v1(packed_state, index);
    }
    return true;
}

static_assert(
    kLeadingFullUint64Lanes + kMiddleLowUint32Lanes
        + kTrailingFullUint64Lanes
    == kFullStateBytes / kUint64Bytes
);
static_assert(
    kLeadingFullBytes + kMiddlePackedBytes
        + kTrailingFullUint64Lanes * kUint64Bytes
    == kPackedStateBytes
);
static_assert(sizeof(std::uint64_t) == kUint64Bytes);
static_assert(sizeof(std::uint32_t) == kUint32Bytes);
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
static_assert(
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
    "Torch CPU RNG compact state requires a little-endian build"
);
#else
#error "Torch CPU RNG compact state requires compile-time byte-order metadata"
#endif
static_assert(packed_byte_index_v1(0) == 0);
static_assert(packed_byte_index_v1(kLeadingFullBytes - 1) == 23);
static_assert(packed_byte_index_v1(kLeadingFullBytes) == 24);
static_assert(packed_byte_index_v1(kLeadingFullBytes + 3) == 27);
static_assert(
    packed_byte_index_v1(kLeadingFullBytes + 4)
    == kOmittedPackedByteIndex
);
static_assert(
    packed_byte_index_v1(kTrailingFullByteOffset - 1)
    == kOmittedPackedByteIndex
);
static_assert(
    packed_byte_index_v1(kTrailingFullByteOffset)
    == kTrailingPackedByteOffset
);
static_assert(
    packed_byte_index_v1(kFullStateBytes - 1) == kPackedStateBytes - 1
);

}  // namespace eiel::cuda_architecture_b::torch_rng_state_pack_v1

#undef EIEL_TORCH_RNG_STATE_PACK_DEVICE
#undef EIEL_TORCH_RNG_STATE_PACK_HD
