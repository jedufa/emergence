#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "p12_cuda_rowv1_disabled_receipts_v1.cuh"

#include <cstdint>
#include <vector>

namespace {

namespace rowv1_disabled = eiel::p12::cuda_rowv1_disabled_receipts_v1;

void check_tensor(
    const torch::Tensor& tensor,
    const char* label,
    const c10::ScalarType dtype,
    const c10::Device& device
) {
    TORCH_CHECK(tensor.is_cuda(), label, " must be CUDA resident");
    TORCH_CHECK(tensor.device() == device, label, " must share one CUDA device");
    TORCH_CHECK(tensor.is_contiguous(), label, " must be contiguous");
    TORCH_CHECK(!tensor.requires_grad(), label, " must be gradient-free");
    TORCH_CHECK(tensor.scalar_type() == dtype, label, " has an invalid dtype");
}

__global__ void write_disabled_rowv1_kernel(
    std::int64_t* __restrict__ rows,
    std::int64_t* __restrict__ counts,
    std::int32_t* __restrict__ overflow,
    const std::int32_t* __restrict__ disabled_reason_codes,
    const std::int64_t* __restrict__ parent_agent_ids,
    const std::int64_t* __restrict__ parent_slots,
    const std::int64_t* __restrict__ parent_slot_tokens,
    const std::int64_t* __restrict__ child_agent_ids,
    const std::int64_t* __restrict__ child_slots,
    const std::int64_t* __restrict__ child_slot_tokens,
    const std::int64_t* __restrict__ generation,
    const std::int64_t* __restrict__ tick,
    const std::int64_t cells,
    const std::int64_t capacity,
    const std::int64_t attempts,
    const std::int64_t row_width
) {
    const std::int64_t cell = blockIdx.x;
    if (cell >= cells || threadIdx.x != 0) {
        return;
    }
    rowv1_disabled::write_cell(
        rows,
        counts,
        overflow,
        disabled_reason_codes,
        parent_agent_ids,
        parent_slots,
        parent_slot_tokens,
        child_agent_ids,
        child_slots,
        child_slot_tokens,
        generation,
        tick,
        cell,
        capacity,
        attempts,
        row_width
    );
}

}  // namespace

std::vector<torch::Tensor> eiel_p12_cuda_rowv1_disabled_receipts_write_v1(
    torch::Tensor rows,
    torch::Tensor counts,
    torch::Tensor overflow,
    torch::Tensor disabled_reason_codes,
    torch::Tensor parent_agent_ids,
    torch::Tensor parent_slots,
    torch::Tensor parent_slot_tokens,
    torch::Tensor child_agent_ids,
    torch::Tensor child_slots,
    torch::Tensor child_slot_tokens,
    torch::Tensor generation,
    torch::Tensor tick
) {
    TORCH_CHECK(rows.dim() == 3, "rows must be rank-3");
    const c10::Device device = rows.device();
    check_tensor(rows, "rows", torch::kInt64, device);
    check_tensor(counts, "counts", torch::kInt64, device);
    check_tensor(overflow, "overflow", torch::kInt32, device);
    check_tensor(disabled_reason_codes, "disabled_reason_codes", torch::kInt32, device);
    check_tensor(parent_agent_ids, "parent_agent_ids", torch::kInt64, device);
    check_tensor(parent_slots, "parent_slots", torch::kInt64, device);
    check_tensor(parent_slot_tokens, "parent_slot_tokens", torch::kInt64, device);
    check_tensor(child_agent_ids, "child_agent_ids", torch::kInt64, device);
    check_tensor(child_slots, "child_slots", torch::kInt64, device);
    check_tensor(child_slot_tokens, "child_slot_tokens", torch::kInt64, device);
    check_tensor(generation, "generation", torch::kInt64, device);
    check_tensor(tick, "tick", torch::kInt64, device);
    TORCH_CHECK(disabled_reason_codes.dim() == 2, "disabled reason codes must be rank-2");
    const std::int64_t cells = disabled_reason_codes.size(0);
    const std::int64_t attempts = disabled_reason_codes.size(1);
    const std::int64_t capacity = rows.size(1);
    TORCH_CHECK(
        rows.size(0) == cells && rows.size(2) == rowv1_disabled::kRowV1Width,
        "rows shape mismatch"
    );
    TORCH_CHECK(counts.dim() == 1 && counts.size(0) == cells, "counts shape mismatch");
    TORCH_CHECK(overflow.dim() == 1 && overflow.size(0) == cells, "overflow shape mismatch");
    TORCH_CHECK(generation.dim() == 1 && generation.size(0) == cells, "generation shape mismatch");
    TORCH_CHECK(tick.dim() == 1 && tick.size(0) == cells, "tick shape mismatch");
    TORCH_CHECK(capacity >= 0 && attempts >= 0, "capacity and attempts must be nonnegative");
    const auto same_attempt_shape = [&](const torch::Tensor& tensor, const char* label) {
        TORCH_CHECK(
            tensor.dim() == 2 && tensor.size(0) == cells && tensor.size(1) == attempts,
            label,
            " shape mismatch"
        );
    };
    same_attempt_shape(parent_agent_ids, "parent_agent_ids");
    same_attempt_shape(parent_slots, "parent_slots");
    same_attempt_shape(parent_slot_tokens, "parent_slot_tokens");
    same_attempt_shape(child_agent_ids, "child_agent_ids");
    same_attempt_shape(child_slots, "child_slots");
    same_attempt_shape(child_slot_tokens, "child_slot_tokens");

    const c10::cuda::CUDAGuard guard(device);
    write_disabled_rowv1_kernel<<<cells, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        rows.data_ptr<std::int64_t>(),
        counts.data_ptr<std::int64_t>(),
        overflow.data_ptr<std::int32_t>(),
        disabled_reason_codes.data_ptr<std::int32_t>(),
        parent_agent_ids.data_ptr<std::int64_t>(),
        parent_slots.data_ptr<std::int64_t>(),
        parent_slot_tokens.data_ptr<std::int64_t>(),
        child_agent_ids.data_ptr<std::int64_t>(),
        child_slots.data_ptr<std::int64_t>(),
        child_slot_tokens.data_ptr<std::int64_t>(),
        generation.data_ptr<std::int64_t>(),
        tick.data_ptr<std::int64_t>(),
        cells,
        capacity,
        attempts,
        rowv1_disabled::kRowV1Width
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {rows, counts, overflow};
}

#ifndef EIEL_CUDA_ARCHITECTURE_B_NO_PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "eiel_p12_cuda_rowv1_disabled_receipts_write_v1",
        &eiel_p12_cuda_rowv1_disabled_receipts_write_v1,
        "Write P12 CUDA RowV1 disabled-event receipts"
    );
}
#endif
