#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int32_t STATUS_OK = 0;
constexpr int32_t STATUS_CAPACITY_EXCEEDED = 1;
constexpr int32_t STATUS_INVALID_COUNT = 2;

void check_cuda_vector(
    const torch::Tensor& tensor,
    const char* label,
    const c10::ScalarType dtype,
    const c10::Device& device
) {
    TORCH_CHECK(tensor.is_cuda(), label, " must be CUDA resident");
    TORCH_CHECK(tensor.device() == device, label, " must share one CUDA device");
    TORCH_CHECK(tensor.is_contiguous(), label, " must be contiguous");
    TORCH_CHECK(tensor.dim() == 1, label, " must be one-dimensional");
    TORCH_CHECK(tensor.scalar_type() == dtype, label, " has an invalid dtype");
}

void check_count(
    const torch::Tensor& tensor,
    const char* label,
    const c10::Device& device
) {
    check_cuda_vector(tensor, label, torch::kInt64, device);
    TORCH_CHECK(tensor.numel() == 1, label, " must contain exactly one value");
}

__global__ void death_class_concat_kernel(
    const int64_t* __restrict__ action_agent_ids,
    const int64_t* __restrict__ action_slots,
    const int32_t* __restrict__ action_reason_codes,
    const int64_t* __restrict__ action_count,
    const int64_t* __restrict__ residual_agent_ids,
    const int64_t* __restrict__ residual_slots,
    const int32_t* __restrict__ residual_reason_codes,
    const int64_t* __restrict__ residual_count,
    int64_t* __restrict__ combined_agent_ids,
    int64_t* __restrict__ combined_slots,
    int32_t* __restrict__ combined_reason_codes,
    int64_t* __restrict__ combined_count,
    int32_t* __restrict__ status,
    const int64_t capacity
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    const int64_t action_rows = action_count[0];
    const int64_t residual_rows = residual_count[0];
    if (action_rows < 0 || action_rows > capacity
        || residual_rows < 0 || residual_rows > capacity) {
        status[0] = STATUS_INVALID_COUNT;
        return;
    }
    if (residual_rows > capacity - action_rows) {
        status[0] = STATUS_CAPACITY_EXCEEDED;
        return;
    }

    for (int64_t row = 0; row < action_rows; ++row) {
        combined_agent_ids[row] = action_agent_ids[row];
        combined_slots[row] = action_slots[row];
        combined_reason_codes[row] = action_reason_codes[row];
    }
    for (int64_t row = 0; row < residual_rows; ++row) {
        const int64_t output_row = action_rows + row;
        combined_agent_ids[output_row] = residual_agent_ids[row];
        combined_slots[output_row] = residual_slots[row];
        combined_reason_codes[output_row] = residual_reason_codes[row];
    }
    combined_count[0] = action_rows + residual_rows;
    status[0] = STATUS_OK;
}

}  // namespace

std::vector<torch::Tensor> eiel_cuda_architecture_b_death_class_concat_v1(
    torch::Tensor action_agent_ids,
    torch::Tensor action_slots,
    torch::Tensor action_reason_codes,
    torch::Tensor action_count,
    torch::Tensor residual_agent_ids,
    torch::Tensor residual_slots,
    torch::Tensor residual_reason_codes,
    torch::Tensor residual_count
) {
    TORCH_CHECK(action_agent_ids.is_cuda(), "action_agent_ids must be CUDA resident");
    const auto device = action_agent_ids.device();
    c10::cuda::CUDAGuard device_guard(device);

    check_cuda_vector(
        action_agent_ids,
        "action_agent_ids",
        torch::kInt64,
        device
    );
    const int64_t capacity = action_agent_ids.numel();
    for (const auto& item : std::vector<std::pair<const char*, torch::Tensor>>{
             {"action_slots", action_slots},
             {"residual_agent_ids", residual_agent_ids},
             {"residual_slots", residual_slots},
         }) {
        check_cuda_vector(item.second, item.first, torch::kInt64, device);
        TORCH_CHECK(
            item.second.numel() == capacity,
            "death class tensors must share one fixed capacity"
        );
    }
    for (const auto& item : std::vector<std::pair<const char*, torch::Tensor>>{
             {"action_reason_codes", action_reason_codes},
             {"residual_reason_codes", residual_reason_codes},
         }) {
        check_cuda_vector(item.second, item.first, torch::kInt32, device);
        TORCH_CHECK(
            item.second.numel() == capacity,
            "death class tensors must share one fixed capacity"
        );
    }
    check_count(action_count, "action_count", device);
    check_count(residual_count, "residual_count", device);

    const auto int64_options = action_agent_ids.options().dtype(torch::kInt64);
    const auto int32_options = action_agent_ids.options().dtype(torch::kInt32);
    auto combined_agent_ids = torch::full({capacity}, -1, int64_options);
    auto combined_slots = torch::full({capacity}, -1, int64_options);
    auto combined_reason_codes = torch::zeros({capacity}, int32_options);
    auto combined_count = torch::zeros({1}, int64_options);
    auto status = torch::zeros({1}, int32_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    death_class_concat_kernel<<<1, 1, 0, stream>>>(
        action_agent_ids.data_ptr<int64_t>(),
        action_slots.data_ptr<int64_t>(),
        action_reason_codes.data_ptr<int32_t>(),
        action_count.data_ptr<int64_t>(),
        residual_agent_ids.data_ptr<int64_t>(),
        residual_slots.data_ptr<int64_t>(),
        residual_reason_codes.data_ptr<int32_t>(),
        residual_count.data_ptr<int64_t>(),
        combined_agent_ids.data_ptr<int64_t>(),
        combined_slots.data_ptr<int64_t>(),
        combined_reason_codes.data_ptr<int32_t>(),
        combined_count.data_ptr<int64_t>(),
        status.data_ptr<int32_t>(),
        capacity
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        combined_agent_ids,
        combined_slots,
        combined_reason_codes,
        combined_count,
        status,
    };
}

#ifndef EIEL_CUDA_ARCHITECTURE_B_NO_PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Fixed-capacity CUDA Architecture B death-class concatenation";
    module.def(
        "eiel_cuda_architecture_b_death_class_concat_v1",
        &eiel_cuda_architecture_b_death_class_concat_v1
    );
}
#endif
