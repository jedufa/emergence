#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace eiel::native::cpu_compact_summary_digests_v1 {

inline constexpr std::size_t kDigestBytesV1 = 32;

enum class ScalarTypeV1 : std::uint8_t {
    kBool = 0,
    kInt8 = 1,
    kUInt8 = 2,
    kInt32 = 3,
    kFloat32 = 4,
    kInt64 = 5,
    kFloat64 = 6,
    kUnknown = 255,
};

// The native owner populates these views from already-contiguous CPU tensors.
// storage_bytes makes dtype/rank mistakes fail before any payload is read.
struct TensorViewV1 {
    const void* data;
    ScalarTypeV1 dtype;
    std::int32_t rank;
    std::int64_t extent0;
    std::int64_t extent1;
    std::uint64_t storage_bytes;
    bool c_contiguous;
};

struct DigestV1 {
    std::array<std::uint8_t, kDigestBytesV1> bytes;
};

enum class DigestStatusV1 : std::uint32_t {
    kSuccess = 0,
    kNonLittleEndianHost = 1,
    kDtypeMismatch = 2,
    kRankMismatch = 3,
    kNegativeExtent = 4,
    kNonContiguous = 5,
    kStorageSizeMismatch = 6,
    kNullData = 7,
    kSizeOverflow = 8,
    kColumnLengthMismatch = 9,
    kCursorOutOfRange = 10,
    kSequenceMismatch = 11,
    kRowOrderInvalid = 12,
    kNonFiniteValue = 13,
    kActiveAgentIdInvalid = 14,
    kPolicyMaskShapeMismatch = 15,
    kBooleanValueInvalid = 16,
    kInvalidMode = 17,
    kInvalidRowLayout = 18,
    kNullActiveSource = 19,
    kActiveLengthOutOfRange = 20,
};

// These values intentionally match the CUDA checkpoint descriptor contract.
enum class TensorCanonicalizationModeV1 : std::int32_t {
    kExactStateRawBytes = 0,
    kLiveRowsDeadZero = 1,
    kActivePrefix = 2,
};

// A descriptor covers one contiguous CPU tensor. Rank-one tensors have one
// scalar per row; rank-two tensors have extent1 scalars per row.
struct CheckpointTensorDigestDescriptorV1 {
    TensorViewV1 tensor;
    TensorCanonicalizationModeV1 mode;
    const TensorViewV1* alive_mask;
    std::int64_t active_length;
};

struct CheckpointTensorDigestResultV1 {
    DigestV1 raw;
    DigestV1 canonical;
    DigestStatusV1 status;
    std::uint8_t valid;
};

struct TypedLifecycleEventsViewV1 {
    TensorViewV1 event_type;
    TensorViewV1 generation;
    TensorViewV1 tick;
    TensorViewV1 sequence;
    TensorViewV1 actor_agent_id;
    TensorViewV1 target_agent_id;
    TensorViewV1 actor_slot;
    TensorViewV1 target_slot;
    TensorViewV1 parent_agent_id;
    TensorViewV1 child_agent_id;
    TensorViewV1 lineage_founder_id;
    TensorViewV1 lineage_depth;
    TensorViewV1 subtype_code;
    TensorViewV1 reason_code;
    TensorViewV1 success;
    TensorViewV1 value;
    std::int64_t active_count;
};

struct PolicyMutationTapeViewV1 {
    TensorViewV1 crossover_mask;
    TensorViewV1 delta;
    TensorViewV1 mutation_mask;
    TensorViewV1 post_rng_state;
    TensorViewV1 semantic_draws;
    std::int64_t cursor;
    bool post_rng_state_is_packed = false;
};

// The caller owns all storage. On failure both digests are zero and valid is
// zero. On success raw covers all tensor bytes and canonical follows mode.
[[nodiscard]] DigestStatusV1 checkpoint_tensor_descriptor_digests(
    const CheckpointTensorDigestDescriptorV1& descriptor,
    CheckpointTensorDigestResultV1& output
) noexcept;

// Every producer clears output before validation. A non-success result therefore
// never leaves a plausible stale digest in the caller's authority plane.
[[nodiscard]] DigestStatusV1 typed_lifecycle_active_prefix_digest(
    const TypedLifecycleEventsViewV1& events,
    DigestV1& output
) noexcept;

[[nodiscard]] DigestStatusV1 active_agent_ids_digest(
    const TensorViewV1& active_agent_ids,
    DigestV1& output
) noexcept;

[[nodiscard]] DigestStatusV1 policy_mutation_tape_consumed_prefix_digest(
    const PolicyMutationTapeViewV1& tape,
    DigestV1& output
) noexcept;

[[nodiscard]] const char* digest_status_name(DigestStatusV1 status) noexcept;

}  // namespace eiel::native::cpu_compact_summary_digests_v1
