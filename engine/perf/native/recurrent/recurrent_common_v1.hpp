#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__)
#define EIEL_RECURRENT_HOST_DEVICE __host__ __device__
#else
#define EIEL_RECURRENT_HOST_DEVICE
#endif

namespace eiel::native::recurrent_v1 {

inline constexpr std::uint32_t kRecurrentContractVersionV1 = 1;
inline constexpr std::int64_t kAcceptedStableGruHiddenV1 = 64;
inline constexpr std::int64_t kAcceptedActionCountV1 = 7;

enum class StatusV1 : std::uint32_t {
    kSuccess = 0,
    kNullPointer = 1,
    kContractVersionMismatch = 2,
    kInvalidDimension = 3,
    kUnsupportedStableShape = 4,
    kIntegerOverflow = 5,
    kExtentMismatch = 6,
    kStrideInvalid = 7,
    kIndexOutOfRange = 8,
    kTokenMismatch = 9,
    kTokenInvalid = 10,
    kNonFinite = 11,
    kScratchTooSmall = 12,
    kCheckpointInvalid = 13,
    kMutationTapeInvalid = 14,
    kCounterOverflow = 15,
};

struct StableGruShapeV1 {
    std::uint32_t contract_version;
    std::int64_t observation_size;
    std::int64_t hidden_size;
    std::int64_t action_count;
};

struct StableGruPackedLayoutV1 {
    std::int64_t encoder_weight;
    std::int64_t encoder_bias;
    std::int64_t input_weight;
    std::int64_t input_bias;
    std::int64_t hidden_weight;
    std::int64_t hidden_bias;
    std::int64_t head_weight;
    std::int64_t head_bias;
    std::int64_t total_elements;
};

struct RecurrentStateViewV1 {
    float* hidden;
    std::int64_t hidden_element_count;
    std::uint64_t* tokens;
    std::int64_t token_count;
    std::int64_t cell_count;
    std::int64_t agents_per_cell;
    std::int64_t hidden_cell_stride;
    std::int64_t hidden_agent_stride;
    std::int64_t token_cell_stride;
    std::int64_t token_agent_stride;
};

struct ConstRecurrentStateViewV1 {
    const float* hidden;
    std::int64_t hidden_element_count;
    const std::uint64_t* tokens;
    std::int64_t token_count;
    std::int64_t cell_count;
    std::int64_t agents_per_cell;
    std::int64_t hidden_cell_stride;
    std::int64_t hidden_agent_stride;
    std::int64_t token_cell_stride;
    std::int64_t token_agent_stride;
};

struct RecurrentRowAddressV1 {
    std::int64_t hidden_offset;
    std::int64_t token_offset;
};

struct StableGruCompactCountersV1 {
    std::uint64_t committed_steps;
    std::uint64_t hidden_elements_updated;
    std::uint64_t resets;
    std::uint64_t births;
    std::uint64_t deaths;
    std::uint64_t turnovers;
    std::uint64_t mutation_draws_consumed;
    std::uint64_t checkpoint_rows_projected;
    std::uint64_t checkpoint_rows_restored;
};

enum class LifecycleOperationV1 : std::uint32_t {
    kReset = 1,
    kBirthInitialize = 2,
    kDeathCleanup = 3,
    kTurnoverReset = 4,
};

EIEL_RECURRENT_HOST_DEVICE constexpr bool ok(const StatusV1 status) noexcept {
    return status == StatusV1::kSuccess;
}

EIEL_RECURRENT_HOST_DEVICE constexpr bool checked_add_i64(
    const std::int64_t left,
    const std::int64_t right,
    std::int64_t& result
) noexcept {
    if (left < 0 || right < 0 || left > INT64_MAX - right) {
        return false;
    }
    result = left + right;
    return true;
}

EIEL_RECURRENT_HOST_DEVICE constexpr bool checked_mul_i64(
    const std::int64_t left,
    const std::int64_t right,
    std::int64_t& result
) noexcept {
    if (left < 0 || right < 0 || (left != 0 && right > INT64_MAX / left)) {
        return false;
    }
    result = left * right;
    return true;
}

EIEL_RECURRENT_HOST_DEVICE constexpr bool checked_add_u64(
    const std::uint64_t left,
    const std::uint64_t right,
    std::uint64_t& result
) noexcept {
    if (left > UINT64_MAX - right) {
        return false;
    }
    result = left + right;
    return true;
}

static_assert(std::is_standard_layout_v<StableGruShapeV1>);
static_assert(std::is_trivially_copyable_v<StableGruShapeV1>);
static_assert(std::is_standard_layout_v<StableGruPackedLayoutV1>);
static_assert(std::is_trivially_copyable_v<StableGruPackedLayoutV1>);
static_assert(std::is_standard_layout_v<RecurrentStateViewV1>);
static_assert(std::is_trivially_copyable_v<RecurrentStateViewV1>);
static_assert(std::is_standard_layout_v<StableGruCompactCountersV1>);
static_assert(std::is_trivially_copyable_v<StableGruCompactCountersV1>);

}  // namespace eiel::native::recurrent_v1

#undef EIEL_RECURRENT_HOST_DEVICE
