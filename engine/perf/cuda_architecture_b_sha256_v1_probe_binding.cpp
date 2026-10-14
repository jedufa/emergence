#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cstdint>

void eiel_cuda_architecture_b_sha256_v1_launch_probe(
    const std::uint8_t* input,
    std::uint64_t input_bytes,
    std::uint64_t split_bytes,
    std::uint8_t* output,
    std::int32_t* status,
    cudaStream_t stream
);

cudaError_t eiel_cuda_architecture_b_sha256_v1_probe_attributes(
    cudaFuncAttributes* attributes
);

namespace {

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

}  // namespace

void eiel_cuda_architecture_b_sha256_v1_hash_into(
    const torch::Tensor& input,
    const torch::Tensor& output,
    const torch::Tensor& status,
    const std::int64_t split_bytes
) {
    require_cuda_contiguous(input, input, "input");
    require_cuda_contiguous(input, output, "output");
    require_cuda_contiguous(input, status, "status");
    TORCH_CHECK(input.scalar_type() == torch::kUInt8, "input must be uint8");
    TORCH_CHECK(output.scalar_type() == torch::kUInt8, "output must be uint8");
    TORCH_CHECK(output.numel() == 32, "output must be uint8[32]");
    TORCH_CHECK(status.scalar_type() == torch::kInt32, "status must be int32");
    TORCH_CHECK(status.numel() == 1, "status must be int32[1]");
    TORCH_CHECK(
        split_bytes >= 0 && split_bytes <= input.numel(),
        "split_bytes must be within the input extent"
    );

    c10::cuda::CUDAGuard device_guard(input.device());
    const auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
    eiel_cuda_architecture_b_sha256_v1_launch_probe(
        input.data_ptr<std::uint8_t>(),
        static_cast<std::uint64_t>(input.numel()),
        static_cast<std::uint64_t>(split_bytes),
        output.data_ptr<std::uint8_t>(),
        status.data_ptr<std::int32_t>(),
        stream.stream()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

pybind11::dict eiel_cuda_architecture_b_sha256_v1_resource_receipt() {
    cudaFuncAttributes attributes{};
    C10_CUDA_CHECK(
        eiel_cuda_architecture_b_sha256_v1_probe_attributes(&attributes)
    );
    pybind11::dict receipt;
    receipt["kernel_symbol"] = "cuda_architecture_b_sha256_v1_probe_kernel";
    receipt["binary_version"] = attributes.binaryVersion;
    receipt["ptx_version"] = attributes.ptxVersion;
    receipt["num_registers"] = attributes.numRegs;
    receipt["local_size_bytes"] = attributes.localSizeBytes;
    receipt["shared_size_bytes"] = attributes.sharedSizeBytes;
    receipt["constant_size_bytes"] = attributes.constSizeBytes;
    receipt["max_threads_per_block"] = attributes.maxThreadsPerBlock;
    receipt["max_dynamic_shared_size_bytes"] = attributes.maxDynamicSharedSizeBytes;
    return receipt;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "hash_into",
        &eiel_cuda_architecture_b_sha256_v1_hash_into,
        pybind11::arg("input"),
        pybind11::arg("output"),
        pybind11::arg("status"),
        pybind11::arg("split_bytes")
    );
    module.def(
        "resource_receipt",
        &eiel_cuda_architecture_b_sha256_v1_resource_receipt
    );
}
