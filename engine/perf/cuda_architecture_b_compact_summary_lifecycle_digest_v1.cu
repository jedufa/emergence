#include "cuda_architecture_b_compact_summary_lifecycle_digest_v1.cuh"

#include "cuda_architecture_b_sha256_v1.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace eiel::cuda_architecture_b::compact_summary_lifecycle_digest_v1 {

namespace {

namespace sha256 = eiel::cuda_architecture_b::sha256_v1;

template <std::size_t Size>
__device__ __forceinline__ void update_ascii_literal(
    sha256::Sha256ContextV1* const context,
    const char (&literal)[Size]
) noexcept {
    sha256::sha256_update_v1(
        context,
        reinterpret_cast<const std::uint8_t*>(literal),
        static_cast<std::uint64_t>(Size - 1U)
    );
}

__device__ __forceinline__ std::uint32_t compact_shape_bytes(
    const std::int64_t count,
    std::uint8_t output[22]
) noexcept {
    output[0] = static_cast<std::uint8_t>('[');
    if (count == 0) {
        output[1] = static_cast<std::uint8_t>('0');
        output[2] = static_cast<std::uint8_t>(']');
        return 3U;
    }

    std::uint8_t reversed[19];
    std::uint32_t digits = 0U;
    std::uint64_t remaining = static_cast<std::uint64_t>(count);
    while (remaining != 0U) {
        reversed[digits++] = static_cast<std::uint8_t>(
            '0' + static_cast<std::uint8_t>(remaining % 10U)
        );
        remaining /= 10U;
    }
    for (std::uint32_t index = 0U; index < digits; ++index) {
        output[1U + index] = reversed[digits - index - 1U];
    }
    output[1U + digits] = static_cast<std::uint8_t>(']');
    return digits + 2U;
}

__device__ __forceinline__ void zero_digest(std::uint8_t* const digest) noexcept {
#pragma unroll
    for (std::int32_t index = 0; index < kDigestBytesV1; ++index) {
        digest[index] = 0U;
    }
}

__device__ __forceinline__ bool all_columns_present(
    const FlatTypedLifecycleColumnsV1& columns
) noexcept {
    return columns.typed_event_type != nullptr
        && columns.typed_generation != nullptr
        && columns.typed_tick != nullptr
        && columns.typed_sequence != nullptr
        && columns.typed_actor_id != nullptr
        && columns.typed_target_id != nullptr
        && columns.typed_actor_slot != nullptr
        && columns.typed_target_slot != nullptr
        && columns.typed_parent_id != nullptr
        && columns.typed_child_id != nullptr
        && columns.typed_founder_id != nullptr
        && columns.typed_lineage_depth != nullptr
        && columns.typed_subtype != nullptr
        && columns.typed_reason != nullptr
        && columns.typed_success != nullptr
        && columns.typed_value != nullptr;
}

__device__ __forceinline__ bool byte_extent(
    const std::uint64_t element_offset,
    const std::uint64_t element_count,
    const std::uint64_t element_bytes,
    std::uint64_t* const byte_offset,
    std::uint64_t* const byte_count
) noexcept {
    constexpr std::uint64_t maximum = ~static_cast<std::uint64_t>(0U);
    if (
        element_offset > maximum / element_bytes
        || element_count > maximum / element_bytes
    ) {
        return false;
    }
    *byte_offset = element_offset * element_bytes;
    *byte_count = element_count * element_bytes;
    return *byte_offset <= maximum - *byte_count;
}

template <typename T, std::size_t FieldSize, std::size_t DtypeSize>
__device__ __forceinline__ bool update_field(
    sha256::Sha256ContextV1* const context,
    const char (&field_name)[FieldSize],
    const char (&dtype_name)[DtypeSize],
    const std::uint8_t* const shape,
    const std::uint32_t shape_size,
    const T* const column,
    const std::uint64_t element_offset,
    const std::uint64_t count
) noexcept {
    update_ascii_literal(context, field_name);
    update_ascii_literal(context, dtype_name);
    sha256::sha256_update_v1(context, shape, shape_size);
    if (count == 0U) {
        return true;
    }
    std::uint64_t byte_offset = 0U;
    std::uint64_t byte_count = 0U;
    if (!byte_extent(
            element_offset,
            count,
            static_cast<std::uint64_t>(sizeof(T)),
            &byte_offset,
            &byte_count
        )) {
        return false;
    }
    sha256::sha256_update_v1(
        context,
        reinterpret_cast<const std::uint8_t*>(column) + byte_offset,
        byte_count
    );
    return true;
}

__device__ __forceinline__ LifecycleDigestStatusV1 validate_active_prefix(
    const FlatLifecycleDigestInputsV1& inputs,
    const std::int64_t cell,
    std::int64_t* const count_out,
    std::uint64_t* const element_offset_out
) noexcept {
    if (inputs.event_cursor == nullptr) {
        return kLifecycleDigestNullCursorV1;
    }
    if (inputs.capacity < 0) {
        return kLifecycleDigestInvalidCapacityV1;
    }

    const std::int64_t count = inputs.event_cursor[cell];
    if (count < 0 || count > inputs.capacity) {
        return kLifecycleDigestCountOutOfRangeV1;
    }
    if (count != 0 && !all_columns_present(inputs.columns)) {
        return kLifecycleDigestNullColumnV1;
    }

    const std::uint64_t capacity = static_cast<std::uint64_t>(inputs.capacity);
    const std::uint64_t cell_index = static_cast<std::uint64_t>(cell);
    constexpr std::uint64_t maximum = ~static_cast<std::uint64_t>(0U);
    if (capacity != 0U && cell_index > maximum / capacity) {
        return kLifecycleDigestAddressOverflowV1;
    }
    const std::uint64_t element_offset = cell_index * capacity;
    const std::uint64_t count_u64 = static_cast<std::uint64_t>(count);
    if (element_offset > maximum - count_u64) {
        return kLifecycleDigestAddressOverflowV1;
    }

    std::int64_t previous_generation = 0;
    std::int64_t previous_tick = 0;
    std::int64_t previous_sequence = -1;
    for (std::int64_t row = 0; row < count; ++row) {
        const std::uint64_t index = element_offset + static_cast<std::uint64_t>(row);
        const std::int64_t sequence = inputs.columns.typed_sequence[index];
        if (sequence != row) {
            return kLifecycleDigestSequenceMismatchV1;
        }
        const float value = inputs.columns.typed_value[index];
        const std::uint32_t value_bits = __float_as_uint(value);
        if ((value_bits & 0x7f800000U) == 0x7f800000U) {
            return kLifecycleDigestNonfiniteValueV1;
        }

        const std::int64_t generation = inputs.columns.typed_generation[index];
        const std::int64_t tick = inputs.columns.typed_tick[index];
        if (
            row != 0
            && (
                generation < previous_generation
                || (generation == previous_generation && tick < previous_tick)
                || (
                    generation == previous_generation
                    && tick == previous_tick
                    && sequence <= previous_sequence
                )
            )
        ) {
            return kLifecycleDigestRowOrderInvalidV1;
        }
        previous_generation = generation;
        previous_tick = tick;
        previous_sequence = sequence;
    }
    *count_out = count;
    *element_offset_out = element_offset;
    return kLifecycleDigestSuccessV1;
}

__device__ __forceinline__ LifecycleDigestStatusV1 digest_active_prefix(
    const FlatLifecycleDigestInputsV1& inputs,
    const std::int64_t cell,
    std::uint8_t digest[kDigestBytesV1]
) noexcept {
    std::int64_t count = 0;
    std::uint64_t element_offset = 0U;
    const LifecycleDigestStatusV1 validation = validate_active_prefix(
        inputs,
        cell,
        &count,
        &element_offset
    );
    if (validation != kLifecycleDigestSuccessV1) {
        return validation;
    }

    std::uint8_t shape[22];
    const std::uint32_t shape_size = compact_shape_bytes(count, shape);
    const std::uint64_t count_u64 = static_cast<std::uint64_t>(count);
    sha256::Sha256ContextV1 context;
    sha256::sha256_init_v1(&context);
    update_ascii_literal(&context, "eiel_typed_lifecycle_event_v1");

    // FIELD_DTYPES lexicographic name order from lifecycle_events.py.
    bool extent_valid = true;
    extent_valid = extent_valid && update_field(
        &context, "actor_agent_id", "int64", shape, shape_size,
        inputs.columns.typed_actor_id, element_offset, count_u64
    );
    extent_valid = extent_valid && update_field(
        &context, "actor_slot", "int64", shape, shape_size,
        inputs.columns.typed_actor_slot, element_offset, count_u64
    );
    extent_valid = extent_valid && update_field(
        &context, "child_agent_id", "int64", shape, shape_size,
        inputs.columns.typed_child_id, element_offset, count_u64
    );
    extent_valid = extent_valid && update_field(
        &context, "event_type", "int32", shape, shape_size,
        inputs.columns.typed_event_type, element_offset, count_u64
    );
    extent_valid = extent_valid && update_field(
        &context, "generation", "int64", shape, shape_size,
        inputs.columns.typed_generation, element_offset, count_u64
    );
    extent_valid = extent_valid && update_field(
        &context, "lineage_depth", "int32", shape, shape_size,
        inputs.columns.typed_lineage_depth, element_offset, count_u64
    );
    extent_valid = extent_valid && update_field(
        &context, "lineage_founder_id", "int64", shape, shape_size,
        inputs.columns.typed_founder_id, element_offset, count_u64
    );
    extent_valid = extent_valid && update_field(
        &context, "parent_agent_id", "int64", shape, shape_size,
        inputs.columns.typed_parent_id, element_offset, count_u64
    );
    extent_valid = extent_valid && update_field(
        &context, "reason_code", "int32", shape, shape_size,
        inputs.columns.typed_reason, element_offset, count_u64
    );
    extent_valid = extent_valid && update_field(
        &context, "sequence", "int64", shape, shape_size,
        inputs.columns.typed_sequence, element_offset, count_u64
    );
    extent_valid = extent_valid && update_field(
        &context, "subtype_code", "int32", shape, shape_size,
        inputs.columns.typed_subtype, element_offset, count_u64
    );
    extent_valid = extent_valid && update_field(
        &context, "success", "int8", shape, shape_size,
        inputs.columns.typed_success, element_offset, count_u64
    );
    extent_valid = extent_valid && update_field(
        &context, "target_agent_id", "int64", shape, shape_size,
        inputs.columns.typed_target_id, element_offset, count_u64
    );
    extent_valid = extent_valid && update_field(
        &context, "target_slot", "int64", shape, shape_size,
        inputs.columns.typed_target_slot, element_offset, count_u64
    );
    extent_valid = extent_valid && update_field(
        &context, "tick", "int64", shape, shape_size,
        inputs.columns.typed_tick, element_offset, count_u64
    );
    extent_valid = extent_valid && update_field(
        &context, "value", "float32", shape, shape_size,
        inputs.columns.typed_value, element_offset, count_u64
    );
    if (!extent_valid) {
        return kLifecycleDigestAddressOverflowV1;
    }
    sha256::sha256_final_v1(&context, digest);
    return kLifecycleDigestSuccessV1;
}

__global__ void compact_summary_lifecycle_digest_kernel_v1(
    const FlatLifecycleDigestInputsV1 inputs,
    const LifecycleDigestOutputsV1 outputs
) {
    const std::int64_t cell =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cell >= outputs.cell_count) {
        return;
    }
    const std::int64_t row = outputs.generation_index * outputs.cell_count + cell;
    std::uint8_t* const output_digest = outputs.digest + row * kDigestBytesV1;
    outputs.validity[row] = 0U;
    zero_digest(output_digest);

    std::uint8_t local_digest[kDigestBytesV1];
    const LifecycleDigestStatusV1 status = digest_active_prefix(
        inputs,
        cell,
        local_digest
    );
    outputs.status[row] = static_cast<std::int32_t>(status);
    if (status != kLifecycleDigestSuccessV1) {
        return;
    }
#pragma unroll
    for (std::int32_t index = 0; index < kDigestBytesV1; ++index) {
        output_digest[index] = local_digest[index];
    }
    // Validity publishes a completed digest and status row.
    outputs.validity[row] = 1U;
}

}  // namespace

void launch_compact_summary_lifecycle_digest_v1(
    const FlatLifecycleDigestInputsV1 inputs,
    const LifecycleDigestOutputsV1 outputs,
    cudaStream_t stream
) {
    if (
        outputs.digest == nullptr
        || outputs.status == nullptr
        || outputs.validity == nullptr
        || outputs.generation_count <= 0
        || outputs.cell_count <= 0
        || outputs.generation_index < 0
        || outputs.generation_index >= outputs.generation_count
    ) {
        return;
    }
    const unsigned int blocks = static_cast<unsigned int>(
        (outputs.cell_count + kThreadsPerBlockV1 - 1) / kThreadsPerBlockV1
    );
    compact_summary_lifecycle_digest_kernel_v1<<<
        blocks,
        kThreadsPerBlockV1,
        0,
        stream
    >>>(inputs, outputs);
}

KernelResourceReceiptV1 query_compact_summary_lifecycle_digest_v1_resources() {
    cudaFuncAttributes attributes{};
    cudaError_t error = cudaFuncGetAttributes(
        &attributes,
        compact_summary_lifecycle_digest_kernel_v1
    );
    if (error != cudaSuccess) {
        return {-1, -1, -1, -1, -1, -1, -1, -1};
    }
    std::int32_t occupancy = -1;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy,
        compact_summary_lifecycle_digest_kernel_v1,
        kThreadsPerBlockV1,
        0
    );
    if (error != cudaSuccess) {
        occupancy = -1;
    }
    return {
        attributes.binaryVersion,
        attributes.ptxVersion,
        attributes.maxThreadsPerBlock,
        attributes.numRegs,
        occupancy,
        static_cast<std::int64_t>(attributes.constSizeBytes),
        static_cast<std::int64_t>(attributes.localSizeBytes),
        static_cast<std::int64_t>(attributes.sharedSizeBytes),
    };
}

}  // namespace eiel::cuda_architecture_b::compact_summary_lifecycle_digest_v1
