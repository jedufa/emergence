#include "cuda_architecture_b_flattened_modular_controller_v1.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace modular = eiel::cuda_architecture_b::flattened_modular_controller_v1;
namespace device = eiel::cuda_architecture_b::flattened_modular_controller_v1::device;

__global__ void rowv1_device_writer_probe(
    std::int64_t* rows,
    std::int64_t* counts,
    std::int32_t* overflow,
    std::int32_t* flags
) {
    modular::FlatModularContextV1 context{};
    context.ordinary_birth_rowv1_receipt_rows = rows;
    context.ordinary_birth_rowv1_receipt_counts = counts;
    context.ordinary_birth_rowv1_receipt_overflow = overflow;
    context.cells = 2;
    context.capacity = 4;

    flags[0] = device::write_ordinary_birth_rowv1_receipt_v3(
        context,
        0,
        0,
        11,
        22,
        1001,
        2,
        2001,
        3001,
        3,
        4001,
        0,
        101,
        202,
        2,
        1,
        1,
        5,
        6,
        7,
        8,
        1,
        device::kOrdinaryBirthRowV1DisabledReasonAbsent
    ) ? 1 : 0;
    flags[1] = device::write_ordinary_birth_rowv1_disabled_event_v3(
        context,
        0,
        1,
        11,
        23,
        1002,
        2,
        2001,
        3002,
        4,
        4002,
        1,
        device::kOrdinaryBirthRowV1DisabledReasonStructuralDisabled
    ) ? 1 : 0;
    flags[2] = device::write_ordinary_birth_rowv1_disabled_event_v3(
        context,
        0,
        2,
        11,
        24,
        1003,
        2,
        2001,
        3003,
        5,
        4003,
        2,
        device::kOrdinaryBirthRowV1DisabledReasonAbsent
    ) ? 1 : 0;
    flags[3] = device::write_ordinary_birth_rowv1_receipt_v3(
        context,
        1,
        0,
        12,
        22,
        1101,
        1,
        2101,
        3101,
        2,
        4101,
        0,
        0,
        0,
        -1,
        3,
        2,
        -1,
        -1,
        -1,
        0,
        0,
        7
    ) ? 1 : 0;
    flags[4] = device::write_ordinary_birth_rowv1_disabled_event_v3(
        context,
        1,
        0,
        12,
        23,
        1102,
        1,
        2101,
        3102,
        3,
        4102,
        1,
        device::kOrdinaryBirthRowV1DisabledReasonAttemptBudgetZero
    ) ? 1 : 0;
    flags[5] = device::write_ordinary_birth_rowv1_receipt_v3(
        context,
        1,
        2,
        12,
        24,
        1103,
        1,
        2101,
        3103,
        4,
        4103,
        2,
        0,
        0,
        2,
        1,
        1,
        5,
        6,
        7,
        8,
        1,
        device::kOrdinaryBirthRowV1DisabledReasonAbsent
    ) ? 1 : 0;
}

template <typename T>
bool allocate_managed(T** pointer, const std::size_t count) {
    return cudaMallocManaged(pointer, count * sizeof(T)) == cudaSuccess;
}

int main() {
    constexpr std::int64_t cells = 2;
    constexpr std::int64_t capacity = modular::kTurnoverMutationSequenceCapacityV1;
    constexpr std::int64_t width = modular::kOrdinaryBirthRowV1ReceiptWidthV3;
    std::int64_t* rows = nullptr;
    std::int64_t* counts = nullptr;
    std::int32_t* overflow = nullptr;
    std::int32_t* flags = nullptr;
    if (!allocate_managed(&rows, cells * capacity * width)
        || !allocate_managed(&counts, cells)
        || !allocate_managed(&overflow, cells)
        || !allocate_managed(&flags, 6)) {
        std::fprintf(stderr, "allocation failed\n");
        return 10;
    }
    for (std::int64_t index = 0; index < cells * capacity * width; ++index) {
        rows[index] = -1;
    }
    for (std::int64_t cell = 0; cell < cells; ++cell) {
        counts[cell] = 0;
        overflow[cell] = 0;
    }
    for (std::int64_t index = 0; index < 6; ++index) flags[index] = -1;

    rowv1_device_writer_probe<<<1, 1>>>(rows, counts, overflow, flags);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "kernel failed\n");
        return 20;
    }

    const std::int32_t expected_flags[6] = {1, 1, 0, 0, 1, 0};
    for (std::int64_t index = 0; index < 6; ++index) {
        if (flags[index] != expected_flags[index]) return 30 + static_cast<int>(index);
    }
    if (counts[0] != 2 || overflow[0] != 0) return 40;
    if (counts[1] != 1 || overflow[1] != 1) return 41;

    const std::int64_t* accepted = rows;
    if (accepted[0] != 3 || accepted[1] != 2 || accepted[2] != 0) return 50;
    if (accepted[3] != 11 || accepted[4] != 22) return 51;
    if (accepted[5] != 1001 || accepted[7] != 2001) return 52;
    if (accepted[8] != 3001 || accepted[10] != 4001) return 53;
    if (accepted[11] != 0 || accepted[12] != 101 || accepted[13] != 202) return 54;
    if (accepted[14] != 2 || accepted[15] != 1 || accepted[16] != 1) return 55;
    if (accepted[21] != 1 || accepted[22] != 0 || accepted[23] != -1 || accepted[63] != -1) {
        return 56;
    }

    const std::int64_t* disabled = rows + width;
    if (disabled[0] != 3 || disabled[1] != 2 || disabled[2] != 1) return 60;
    if (disabled[3] != 11 || disabled[4] != 23) return 61;
    if (disabled[5] != 1002 || disabled[8] != 3002) return 62;
    if (disabled[11] != 1 || disabled[14] != -1) return 63;
    if (disabled[15] != 3 || disabled[16] != 2 || disabled[21] != 0) return 64;
    if (disabled[22] != device::kOrdinaryBirthRowV1DisabledReasonStructuralDisabled) {
        return 65;
    }
    if (disabled[23] != -1 || disabled[63] != -1) return 66;

    const std::int64_t* disabled_budget = rows + capacity * width;
    if (disabled_budget[0] != 3 || disabled_budget[1] != 2 || disabled_budget[2] != 0) {
        return 70;
    }
    if (disabled_budget[22] != device::kOrdinaryBirthRowV1DisabledReasonAttemptBudgetZero) {
        return 71;
    }
    if (disabled_budget[23] != -1 || disabled_budget[63] != -1) return 72;
    return 0;
}
