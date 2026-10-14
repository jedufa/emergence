#ifndef EIEL_CUDA_ARCHITECTURE_B_COMPACT_SUMMARY_LIFECYCLE_DIGEST_V1_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_COMPACT_SUMMARY_LIFECYCLE_DIGEST_V1_CUH_

#include <cuda_runtime.h>

#include <cstdint>

namespace eiel::cuda_architecture_b::compact_summary_lifecycle_digest_v1 {

inline constexpr char kContractVersionV1[] =
    "eiel_cuda_architecture_b_compact_summary_lifecycle_digest_v1";
inline constexpr char kLifecycleSchemaVersionV1[] =
    "eiel_typed_lifecycle_event_v1";
inline constexpr std::int32_t kDigestBytesV1 = 32;
inline constexpr std::int32_t kFieldCountV1 = 16;
inline constexpr std::int32_t kThreadsPerBlockV1 = 128;

enum LifecycleDigestStatusV1 : std::int32_t {
    kLifecycleDigestSuccessV1 = 0,
    kLifecycleDigestNullCursorV1 = 1,
    kLifecycleDigestInvalidCapacityV1 = 2,
    kLifecycleDigestCountOutOfRangeV1 = 3,
    kLifecycleDigestNullColumnV1 = 4,
    kLifecycleDigestAddressOverflowV1 = 5,
    kLifecycleDigestSequenceMismatchV1 = 6,
    kLifecycleDigestNonfiniteValueV1 = 7,
    kLifecycleDigestRowOrderInvalidV1 = 8,
};

// Names mirror the typed subset of flattened_controller_v1::FlatEventState.
// Every column is cell-major contiguous [cell_count, capacity].
struct FlatTypedLifecycleColumnsV1 {
    const std::int32_t* typed_event_type;
    const std::int64_t* typed_generation;
    const std::int64_t* typed_tick;
    const std::int64_t* typed_sequence;
    const std::int64_t* typed_actor_id;
    const std::int64_t* typed_target_id;
    const std::int64_t* typed_actor_slot;
    const std::int64_t* typed_target_slot;
    const std::int64_t* typed_parent_id;
    const std::int64_t* typed_child_id;
    const std::int64_t* typed_founder_id;
    const std::int32_t* typed_lineage_depth;
    const std::int32_t* typed_subtype;
    const std::int32_t* typed_reason;
    const std::int8_t* typed_success;
    const float* typed_value;
};

struct FlatLifecycleDigestInputsV1 {
    FlatTypedLifecycleColumnsV1 columns;
    const std::int64_t* event_cursor;
    std::int64_t capacity;
};

// digest is [generation_count, cell_count, 32]; status and validity are
// [generation_count, cell_count], all contiguous in generation-major order.
struct LifecycleDigestOutputsV1 {
    std::uint8_t* digest;
    std::int32_t* status;
    std::uint8_t* validity;
    std::int64_t generation_count;
    std::int64_t cell_count;
    std::int64_t generation_index;
};

struct KernelResourceReceiptV1 {
    std::int32_t binary_version;
    std::int32_t ptx_version;
    std::int32_t max_threads_per_block;
    std::int32_t registers_per_thread;
    std::int32_t occupancy_blocks_per_multiprocessor;
    std::int64_t constant_bytes;
    std::int64_t local_bytes_per_thread;
    std::int64_t shared_bytes_per_block;
};

void launch_compact_summary_lifecycle_digest_v1(
    FlatLifecycleDigestInputsV1 inputs,
    LifecycleDigestOutputsV1 outputs,
    cudaStream_t stream
);

KernelResourceReceiptV1 query_compact_summary_lifecycle_digest_v1_resources();

}  // namespace eiel::cuda_architecture_b::compact_summary_lifecycle_digest_v1

#endif
