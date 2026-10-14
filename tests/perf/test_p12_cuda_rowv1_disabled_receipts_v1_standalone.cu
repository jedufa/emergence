#include "p12_cuda_rowv1_disabled_receipts_v1.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace rowv1_disabled = eiel::p12::cuda_rowv1_disabled_receipts_v1;

__global__ void test_kernel(
    std::int64_t* rows,
    std::int64_t* counts,
    std::int32_t* overflow,
    const std::int32_t* reasons,
    const std::int64_t* parent_ids,
    const std::int64_t* parent_slots,
    const std::int64_t* parent_tokens,
    const std::int64_t* child_ids,
    const std::int64_t* child_slots,
    const std::int64_t* child_tokens,
    const std::int64_t* generation,
    const std::int64_t* tick,
    const std::int64_t capacity,
    const std::int64_t attempts
) {
    const std::int64_t cell = blockIdx.x;
    rowv1_disabled::write_cell(
        rows,
        counts,
        overflow,
        reasons,
        parent_ids,
        parent_slots,
        parent_tokens,
        child_ids,
        child_slots,
        child_tokens,
        generation,
        tick,
        cell,
        capacity,
        attempts,
        rowv1_disabled::kRowV1Width
    );
}

template <typename T>
bool allocate_managed(T** pointer, const std::size_t count) {
    return cudaMallocManaged(pointer, count * sizeof(T)) == cudaSuccess;
}

int main() {
    constexpr std::int64_t cells = 2;
    constexpr std::int64_t attempts = 3;
    constexpr std::int64_t capacity = 3;
    constexpr std::int64_t row_width = rowv1_disabled::kRowV1Width;
    constexpr std::int64_t row_count = cells * capacity * row_width;
    std::int64_t* rows = nullptr;
    std::int64_t* counts = nullptr;
    std::int32_t* overflow = nullptr;
    std::int32_t* reasons = nullptr;
    std::int64_t* parent_ids = nullptr;
    std::int64_t* parent_slots = nullptr;
    std::int64_t* parent_tokens = nullptr;
    std::int64_t* child_ids = nullptr;
    std::int64_t* child_slots = nullptr;
    std::int64_t* child_tokens = nullptr;
    std::int64_t* generation = nullptr;
    std::int64_t* tick = nullptr;
    if (!allocate_managed(&rows, row_count)
        || !allocate_managed(&counts, cells)
        || !allocate_managed(&overflow, cells)
        || !allocate_managed(&reasons, cells * attempts)
        || !allocate_managed(&parent_ids, cells * attempts)
        || !allocate_managed(&parent_slots, cells * attempts)
        || !allocate_managed(&parent_tokens, cells * attempts)
        || !allocate_managed(&child_ids, cells * attempts)
        || !allocate_managed(&child_slots, cells * attempts)
        || !allocate_managed(&child_tokens, cells * attempts)
        || !allocate_managed(&generation, cells)
        || !allocate_managed(&tick, cells)) {
        std::fprintf(stderr, "allocation failed\n");
        return 10;
    }
    for (std::int64_t index = 0; index < row_count; ++index) rows[index] = -1;
    for (std::int64_t cell = 0; cell < cells; ++cell) {
        counts[cell] = -9;
        overflow[cell] = -9;
        generation[cell] = 40 + cell;
        tick[cell] = 80 + cell;
    }
    for (std::int64_t index = 0; index < cells * attempts; ++index) {
        reasons[index] = static_cast<std::int32_t>(index + 1);
        parent_ids[index] = 1000 + index;
        parent_slots[index] = index;
        parent_tokens[index] = 2000 + index;
        child_ids[index] = 3000 + index;
        child_slots[index] = 10 + index;
        child_tokens[index] = 4000 + index;
    }
    test_kernel<<<cells, 1>>>(
        rows,
        counts,
        overflow,
        reasons,
        parent_ids,
        parent_slots,
        parent_tokens,
        child_ids,
        child_slots,
        child_tokens,
        generation,
        tick,
        capacity,
        attempts
    );
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "kernel failed\n");
        return 20;
    }
    for (std::int64_t cell = 0; cell < cells; ++cell) {
        if (counts[cell] != attempts || overflow[cell] != 0) return 30;
        for (std::int64_t sequence = 0; sequence < attempts; ++sequence) {
            const std::int64_t* const row =
                rows + (cell * capacity + sequence) * row_width;
            const std::int64_t input = cell * attempts + sequence;
            if (row[0] != 3 || row[1] != 2 || row[2] != sequence) return 40;
            if (row[3] != generation[cell] || row[4] != tick[cell]) return 41;
            if (row[5] != parent_ids[input] || row[8] != child_ids[input]) return 42;
            if (row[11] != sequence || row[21] != 0) return 43;
            if (row[22] != reasons[input]) return 44;
            if (row[23] != -1 || row[63] != -1) return 45;
        }
    }
    return 0;
}

