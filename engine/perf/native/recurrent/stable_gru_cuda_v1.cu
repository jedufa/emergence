#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cstdint>

#include "stable_gru_v1.cpp"

namespace stable = eiel::native::recurrent_v1;

namespace {

constexpr std::int64_t kHidden = 64;
constexpr std::int64_t kActions = 7;
constexpr std::int64_t kScratchElements = kHidden + 3 * kHidden
    + 3 * kHidden + kHidden + kActions;

__global__ void stable_gru_forward_rows_kernel(
    const float* observations,
    const float* hidden_before,
    const float* packed_parameters,
    const std::int64_t* active_memory_size,
    const float* aggression,
    const float* temperature,
    const float* drives,
    const float* gain,
    const float hunger_gather_weight,
    const float pain_rest_weight,
    const float fatigue_rest_weight,
    const float pain_attack_weight,
    const float fear_attack_weight,
    const std::int64_t row_count,
    const std::int64_t observation_count,
    const std::int64_t packed_parameter_count,
    float* scratch,
    float* hidden_after,
    float* logits,
    std::int32_t* actions,
    std::int32_t* status
) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (row >= row_count) return;

    float* const row_scratch = scratch + row * kScratchElements;
    stable::StableGruStepScratchV1 scratch_view{
        row_scratch,
        kHidden,
        row_scratch + kHidden,
        3 * kHidden,
        row_scratch + kHidden + 3 * kHidden,
        3 * kHidden,
        row_scratch + kHidden + 6 * kHidden,
        kHidden,
        row_scratch + kHidden + 6 * kHidden + kHidden,
        kActions,
    };
    stable::StableGruStepInputV1 input{
        {stable::kRecurrentContractVersionV1, observation_count, kHidden, kActions},
        observations + row * observation_count,
        observation_count,
        hidden_before + row * kHidden,
        kHidden,
        active_memory_size[row],
        packed_parameters + row * packed_parameter_count,
        packed_parameter_count,
        aggression[row],
        temperature[row],
        drives + row * 4,
        4,
        gain[row],
        hunger_gather_weight,
        pain_rest_weight,
        fatigue_rest_weight,
        pain_attack_weight,
        fear_attack_weight,
    };
    stable::StableGruStepOutputV1 output{
        hidden_after + row * kHidden,
        kHidden,
        logits + row * kActions,
        kActions,
        actions + row,
    };
    status[row] = static_cast<std::int32_t>(
        stable::stable_gru_forward_one(input, scratch_view, output, nullptr)
    );
}

void require_cuda_contiguous(
    const torch::Tensor& reference,
    const torch::Tensor& tensor,
    const char* name
) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.device() == reference.device(), name, " device mismatch");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(!tensor.requires_grad(), name, " must not require gradients");
}

void require_rows(
    const torch::Tensor& reference,
    const torch::Tensor& tensor,
    const torch::ScalarType dtype,
    const std::int64_t rows,
    const std::int64_t columns,
    const char* name
) {
    require_cuda_contiguous(reference, tensor, name);
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " dtype mismatch");
    TORCH_CHECK(
        tensor.dim() == 2 && tensor.size(0) == rows && tensor.size(1) == columns,
        name,
        " shape mismatch"
    );
}

}  // namespace

void eiel_stable_gru_cuda_forward_into_v1(
    const torch::Tensor& observations,
    const torch::Tensor& hidden_before,
    const torch::Tensor& packed_parameters,
    const torch::Tensor& active_memory_size,
    const torch::Tensor& aggression,
    const torch::Tensor& temperature,
    const torch::Tensor& drives,
    const torch::Tensor& gain,
    const double hunger_gather_weight,
    const double pain_rest_weight,
    const double fatigue_rest_weight,
    const double pain_attack_weight,
    const double fear_attack_weight,
    const torch::Tensor& scratch,
    const torch::Tensor& hidden_after,
    const torch::Tensor& logits,
    const torch::Tensor& actions,
    const torch::Tensor& status
) {
    require_cuda_contiguous(observations, observations, "observations");
    TORCH_CHECK(
        observations.scalar_type() == torch::kFloat32 && observations.dim() == 2,
        "observations must be CUDA float32[rows, observations]"
    );
    const std::int64_t rows = observations.size(0);
    const std::int64_t observation_count = observations.size(1);
    TORCH_CHECK(rows > 0 && observation_count > 0, "observations must not be empty");
    const std::int64_t packed_count = 64 * observation_count + 64
        + 3 * 64 * 64 + 3 * 64 + 3 * 64 * 64 + 3 * 64 + 7 * 64 + 7;

    require_rows(observations, hidden_before, torch::kFloat32, rows, kHidden, "hidden_before");
    require_rows(
        observations, packed_parameters, torch::kFloat32, rows, packed_count,
        "packed_parameters"
    );
    require_rows(observations, drives, torch::kFloat32, rows, 4, "drives");
    require_rows(observations, scratch, torch::kFloat32, rows, kScratchElements, "scratch");
    require_rows(observations, hidden_after, torch::kFloat32, rows, kHidden, "hidden_after");
    require_rows(observations, logits, torch::kFloat32, rows, kActions, "logits");

    require_cuda_contiguous(observations, active_memory_size, "active_memory_size");
    require_cuda_contiguous(observations, aggression, "aggression");
    require_cuda_contiguous(observations, temperature, "temperature");
    require_cuda_contiguous(observations, gain, "gain");
    require_cuda_contiguous(observations, actions, "actions");
    require_cuda_contiguous(observations, status, "status");
    TORCH_CHECK(active_memory_size.dim() == 1 && active_memory_size.numel() == rows,
        "active_memory_size must have shape [rows]");
    TORCH_CHECK(aggression.dim() == 1 && aggression.numel() == rows,
        "aggression must have shape [rows]");
    TORCH_CHECK(temperature.dim() == 1 && temperature.numel() == rows,
        "temperature must have shape [rows]");
    TORCH_CHECK(gain.dim() == 1 && gain.numel() == rows,
        "gain must have shape [rows]");
    TORCH_CHECK(actions.dim() == 1 && actions.numel() == rows,
        "actions must have shape [rows]");
    TORCH_CHECK(status.dim() == 1 && status.numel() == rows,
        "status must have shape [rows]");
    TORCH_CHECK(active_memory_size.scalar_type() == torch::kInt64, "active_memory_size must be int64");
    TORCH_CHECK(aggression.scalar_type() == torch::kFloat32, "aggression must be float32");
    TORCH_CHECK(temperature.scalar_type() == torch::kFloat32, "temperature must be float32");
    TORCH_CHECK(gain.scalar_type() == torch::kFloat32, "gain must be float32");
    TORCH_CHECK(actions.scalar_type() == torch::kInt32, "actions must be int32");
    TORCH_CHECK(status.scalar_type() == torch::kInt32, "status must be int32");

    c10::cuda::CUDAGuard device_guard(observations.device());
    const auto stream = at::cuda::getCurrentCUDAStream(observations.device().index());
    constexpr int threads = 128;
    const int blocks = static_cast<int>((rows + threads - 1) / threads);
    stable_gru_forward_rows_kernel<<<blocks, threads, 0, stream.stream()>>>(
        observations.data_ptr<float>(),
        hidden_before.data_ptr<float>(),
        packed_parameters.data_ptr<float>(),
        active_memory_size.data_ptr<std::int64_t>(),
        aggression.data_ptr<float>(),
        temperature.data_ptr<float>(),
        drives.data_ptr<float>(),
        gain.data_ptr<float>(),
        static_cast<float>(hunger_gather_weight),
        static_cast<float>(pain_rest_weight),
        static_cast<float>(fatigue_rest_weight),
        static_cast<float>(pain_attack_weight),
        static_cast<float>(fear_attack_weight),
        rows,
        observation_count,
        packed_count,
        scratch.data_ptr<float>(),
        hidden_after.data_ptr<float>(),
        logits.data_ptr<float>(),
        actions.data_ptr<std::int32_t>(),
        status.data_ptr<std::int32_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

pybind11::dict eiel_stable_gru_cuda_resource_receipt_v1() {
    cudaFuncAttributes attributes{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, stable_gru_forward_rows_kernel));
    pybind11::dict receipt;
    receipt["kernel_symbol"] = "stable_gru_forward_rows_kernel";
    receipt["binary_version"] = attributes.binaryVersion;
    receipt["ptx_version"] = attributes.ptxVersion;
    receipt["num_registers"] = attributes.numRegs;
    receipt["local_size_bytes"] = attributes.localSizeBytes;
    receipt["shared_size_bytes"] = attributes.sharedSizeBytes;
    receipt["max_threads_per_block"] = attributes.maxThreadsPerBlock;
    receipt["scratch_elements_per_row"] = kScratchElements;
    return receipt;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward_into", &eiel_stable_gru_cuda_forward_into_v1);
    module.def("resource_receipt", &eiel_stable_gru_cuda_resource_receipt_v1);
}
