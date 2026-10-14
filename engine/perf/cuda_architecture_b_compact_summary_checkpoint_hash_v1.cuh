#ifndef EIEL_CUDA_ARCHITECTURE_B_COMPACT_SUMMARY_CHECKPOINT_HASH_V1_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_COMPACT_SUMMARY_CHECKPOINT_HASH_V1_CUH_

#if defined(__CUDACC__)
#include "cuda_architecture_b_sha256_v1.cuh"
#endif
#include "cuda_architecture_b_torch_rng_state_pack_v1.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace eiel::cuda_architecture_b::compact_summary_checkpoint_hash_v1 {

inline constexpr char kContractVersionV1[] =
    "eiel_cuda_architecture_b_compact_summary_checkpoint_hash_v1";
inline constexpr std::int32_t kDigestBytesV1 = 32;
inline constexpr std::int32_t kPointerBatchCapacityV1 = 16;
inline constexpr std::int32_t kThreadsPerBlockV1 = 128;
inline constexpr std::uint64_t kMaximumUint64V1 =
    ~static_cast<std::uint64_t>(0U);

enum CanonicalizationModeV1 : std::int32_t {
    kExactStateRawBytesV1 = 0,
    kLiveRowsDeadZeroV1 = 1,
    kActivePrefixV1 = 2,
    kTorchCpuMt19937PackedRowsToDenseActivePrefixV1 = 3,
};

enum SnapshotStatusV1 : std::int32_t {
    kSnapshotSuccessV1 = 0,
    kSnapshotNullDataV1 = 1,
    kSnapshotInvalidModeV1 = 2,
    kSnapshotInvalidDtypeWidthV1 = 3,
    kSnapshotInvalidRowLayoutV1 = 4,
    kSnapshotNullActiveSourceV1 = 5,
    kSnapshotActiveLengthOutOfRangeV1 = 6,
    kSnapshotInvalidOutputIndexV1 = 7,
    kSnapshotAddressOverflowV1 = 8,
};

// One immutable pointer-table entry describes one tensor authority. The table
// is passed in CUDA kernel-parameter memory in fixed-size batches, so snapshot
// execution requires no allocation and no descriptor copy or host sync.
struct TensorDigestPointerEntryV1 {
    const std::uint8_t* data;
    const std::uint8_t* alive_mask;
    const std::int64_t* active_lengths;
    std::uint64_t cell_stride_bytes;
    std::uint64_t bytes_per_cell;
    std::uint64_t row_capacity;
    std::uint64_t row_bytes;
    std::uint64_t alive_mask_cell_stride;
    std::uint32_t dtype_bytes;
    std::int32_t canonicalization_mode;
    std::int32_t raw_output_index;
    std::int32_t canonical_output_index;
    std::int32_t descriptor_index;
};

struct TensorDigestPointerBatchV1 {
    TensorDigestPointerEntryV1 entries[kPointerBatchCapacityV1];
    std::int32_t count;
};

struct SnapshotOutputsV1 {
    std::uint8_t* raw_digests;
    std::uint8_t* canonical_digests;
    std::int32_t* status;
    std::uint8_t* valid;
    std::int64_t generation_count;
    std::int64_t cell_count;
    std::int64_t output_field_count;
    std::int64_t descriptor_field_count;
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

void launch_checkpoint_tensor_hash_snapshot_v1(
    TensorDigestPointerBatchV1 pointer_batch,
    SnapshotOutputsV1 outputs,
    cudaStream_t stream
);

KernelResourceReceiptV1 query_checkpoint_tensor_hash_snapshot_v1_resources();

#if defined(__CUDACC__)

namespace detail {

static __device__ __constant__ const std::uint8_t kZeroBytesV1[64] = {};

__device__ __forceinline__ bool valid_dtype_width(
    const std::uint32_t width
) noexcept {
    return width == 1U || width == 2U || width == 4U || width == 8U;
}

__device__ __forceinline__ bool multiply_exact(
    const std::uint64_t left,
    const std::uint64_t right,
    std::uint64_t* const product
) noexcept {
    if (left != 0U && right > kMaximumUint64V1 / left) {
        return false;
    }
    *product = left * right;
    return true;
}

__device__ __forceinline__ void update_zeros(
    sha256_v1::Sha256ContextV1* const context,
    std::uint64_t byte_count
) noexcept {
    while (byte_count != 0U) {
        const std::uint64_t chunk = byte_count < 64U ? byte_count : 64U;
        sha256_v1::sha256_update_v1(context, kZeroBytesV1, chunk);
        byte_count -= chunk;
    }
}

__device__ __forceinline__ void copy_digest(
    const std::uint8_t* const source,
    std::uint8_t* const destination
) noexcept {
#pragma unroll
    for (std::int32_t index = 0; index < kDigestBytesV1; ++index) {
        destination[index] = source[index];
    }
}

__device__ __forceinline__ void zero_digest(
    std::uint8_t* const destination
) noexcept {
#pragma unroll
    for (std::int32_t index = 0; index < kDigestBytesV1; ++index) {
        destination[index] = 0U;
    }
}

__device__ __forceinline__ SnapshotStatusV1 validate_entry(
    const TensorDigestPointerEntryV1& entry,
    const SnapshotOutputsV1& outputs
) noexcept {
    if (entry.data == nullptr) {
        return kSnapshotNullDataV1;
    }
    if (!valid_dtype_width(entry.dtype_bytes)) {
        return kSnapshotInvalidDtypeWidthV1;
    }
    if (
        entry.canonicalization_mode < kExactStateRawBytesV1
        || entry.canonicalization_mode
            > kTorchCpuMt19937PackedRowsToDenseActivePrefixV1
    ) {
        return kSnapshotInvalidModeV1;
    }
    std::uint64_t expected_bytes = 0U;
    if (
        entry.row_capacity == 0U
        || entry.row_bytes == 0U
        || !multiply_exact(entry.row_capacity, entry.row_bytes, &expected_bytes)
        || expected_bytes != entry.bytes_per_cell
        || entry.cell_stride_bytes < entry.bytes_per_cell
        || entry.row_bytes % entry.dtype_bytes != 0U
        || entry.bytes_per_cell % entry.dtype_bytes != 0U
    ) {
        return kSnapshotInvalidRowLayoutV1;
    }
    if (
        entry.raw_output_index < 0
        || entry.raw_output_index >= outputs.output_field_count
        || entry.canonical_output_index < 0
        || entry.canonical_output_index >= outputs.output_field_count
    ) {
        return kSnapshotInvalidOutputIndexV1;
    }
    if (entry.canonicalization_mode == kLiveRowsDeadZeroV1) {
        if (entry.alive_mask == nullptr) {
            return kSnapshotNullActiveSourceV1;
        }
        if (entry.alive_mask_cell_stride < entry.row_capacity) {
            return kSnapshotInvalidRowLayoutV1;
        }
    }
    if (
        (entry.canonicalization_mode == kActivePrefixV1
         || entry.canonicalization_mode
             == kTorchCpuMt19937PackedRowsToDenseActivePrefixV1)
        && entry.active_lengths == nullptr
    ) {
        return kSnapshotNullActiveSourceV1;
    }
    return kSnapshotSuccessV1;
}

__device__ __forceinline__ bool validate_packed_torch_rng_rows(
    const TensorDigestPointerEntryV1& entry
) noexcept {
    return entry.dtype_bytes == 1U
        && entry.row_bytes
            == static_cast<std::uint64_t>(
                torch_rng_state_pack_v1::kPackedStateBytes
            );
}

__device__ __forceinline__ void update_unpacked_torch_rng_row(
    sha256_v1::Sha256ContextV1* const context,
    const std::uint8_t* const packed_row
) noexcept {
    namespace pack = torch_rng_state_pack_v1;
    sha256_v1::sha256_update_v1(
        context,
        packed_row,
        static_cast<std::uint64_t>(pack::kLeadingFullBytes)
    );
    for (std::int64_t lane = 0; lane < pack::kMiddleLowUint32Lanes; ++lane) {
        const std::int64_t offset =
            pack::kLeadingFullBytes + lane * pack::kUint32Bytes;
        sha256_v1::sha256_update_v1(
            context,
            packed_row + offset,
            static_cast<std::uint64_t>(pack::kUint32Bytes)
        );
        update_zeros(context, static_cast<std::uint64_t>(pack::kUint32Bytes));
    }
    sha256_v1::sha256_update_v1(
        context,
        packed_row + pack::kTrailingPackedByteOffset,
        static_cast<std::uint64_t>(
            pack::kTrailingFullUint64Lanes * pack::kUint64Bytes
        )
    );
}

}  // namespace detail

__device__ __forceinline__ SnapshotStatusV1 hash_entry_for_cell(
    const TensorDigestPointerEntryV1& entry,
    const SnapshotOutputsV1& outputs,
    const std::int64_t cell,
    std::uint8_t raw_digest[kDigestBytesV1],
    std::uint8_t canonical_digest[kDigestBytesV1]
) noexcept {
    const SnapshotStatusV1 validation = detail::validate_entry(entry, outputs);
    if (validation != kSnapshotSuccessV1) {
        return validation;
    }
    if (
        static_cast<std::uint64_t>(cell)
        > kMaximumUint64V1 / entry.cell_stride_bytes
    ) {
        return kSnapshotAddressOverflowV1;
    }
    const std::uint64_t cell_offset =
        static_cast<std::uint64_t>(cell) * entry.cell_stride_bytes;
    const std::uint8_t* const source = entry.data + cell_offset;
    sha256_v1::sha256_digest_v1(source, entry.bytes_per_cell, raw_digest);

    if (entry.canonicalization_mode == kExactStateRawBytesV1) {
        detail::copy_digest(raw_digest, canonical_digest);
        return kSnapshotSuccessV1;
    }

    if (
        entry.canonicalization_mode == kActivePrefixV1
        || entry.canonicalization_mode
            == kTorchCpuMt19937PackedRowsToDenseActivePrefixV1
    ) {
        const std::int64_t active_length = entry.active_lengths[cell];
        if (
            active_length < 0
            || static_cast<std::uint64_t>(active_length) > entry.row_capacity
        ) {
            return kSnapshotActiveLengthOutOfRangeV1;
        }
        std::uint64_t active_bytes = 0U;
        if (!detail::multiply_exact(
                static_cast<std::uint64_t>(active_length),
                entry.row_bytes,
                &active_bytes
            )) {
            return kSnapshotAddressOverflowV1;
        }
        if (
            entry.canonicalization_mode
            == kTorchCpuMt19937PackedRowsToDenseActivePrefixV1
        ) {
            if (!detail::validate_packed_torch_rng_rows(entry)) {
                return kSnapshotInvalidRowLayoutV1;
            }
            sha256_v1::Sha256ContextV1 context;
            sha256_v1::sha256_init_v1(&context);
            for (std::int64_t row = 0; row < active_length; ++row) {
                detail::update_unpacked_torch_rng_row(
                    &context,
                    source
                        + static_cast<std::uint64_t>(row) * entry.row_bytes
                );
            }
            sha256_v1::sha256_final_v1(&context, canonical_digest);
            return kSnapshotSuccessV1;
        }
        sha256_v1::sha256_digest_v1(
            active_bytes == 0U ? nullptr : source,
            active_bytes,
            canonical_digest
        );
        return kSnapshotSuccessV1;
    }

    if (
        static_cast<std::uint64_t>(cell)
        > kMaximumUint64V1 / entry.alive_mask_cell_stride
    ) {
        return kSnapshotAddressOverflowV1;
    }
    const std::uint64_t mask_offset =
        static_cast<std::uint64_t>(cell) * entry.alive_mask_cell_stride;
    sha256_v1::Sha256ContextV1 context;
    sha256_v1::sha256_init_v1(&context);
    for (std::uint64_t row = 0U; row < entry.row_capacity; ++row) {
        if (entry.alive_mask[mask_offset + row] != 0U) {
            sha256_v1::sha256_update_v1(
                &context,
                source + row * entry.row_bytes,
                entry.row_bytes
            );
        } else {
            detail::update_zeros(&context, entry.row_bytes);
        }
    }
    sha256_v1::sha256_final_v1(&context, canonical_digest);
    return kSnapshotSuccessV1;
}

#endif  // defined(__CUDACC__)

}  // namespace eiel::cuda_architecture_b::compact_summary_checkpoint_hash_v1

#endif
