#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kTransportThreads = 256;

__device__ __forceinline__ bool batch_status_ok(
    const int32_t* status, int64_t batch) {
    __shared__ int32_t ok;
    if (threadIdx.x == 0) {
        ok = 1;
    }
    __syncthreads();
    for (int64_t world = threadIdx.x; world < batch; world += blockDim.x) {
        if (status[world] != 0) {
            atomicExch(&ok, 0);
        }
    }
    __syncthreads();
    return ok != 0;
}

template <bool Guarded>
__global__ void raw_transport_kernel(
    const int64_t* source_addresses,
    const int64_t* destination_addresses,
    const int64_t* byte_counts,
    int64_t chunk_count,
    const int32_t* status,
    int64_t batch) {
    const int64_t chunk = blockIdx.x;
    if (chunk >= chunk_count) {
        return;
    }
    if constexpr (Guarded) {
        if (!batch_status_ok(status, batch)) {
            return;
        }
    }
    const auto source_address = static_cast<uintptr_t>(source_addresses[chunk]);
    const auto destination_address =
        static_cast<uintptr_t>(destination_addresses[chunk]);
    const int64_t bytes = byte_counts[chunk];
    const auto* source = reinterpret_cast<const uint4*>(source_address);
    auto* destination = reinterpret_cast<uint4*>(destination_address);
    const int64_t vectors = bytes / static_cast<int64_t>(sizeof(uint4));
    for (int64_t index = threadIdx.x; index < vectors; index += blockDim.x) {
        destination[index] = source[index];
    }
    if (threadIdx.x == 0) {
        const auto* source_tail = reinterpret_cast<const uint8_t*>(source_address);
        auto* destination_tail = reinterpret_cast<uint8_t*>(destination_address);
        for (int64_t index = vectors * static_cast<int64_t>(sizeof(uint4));
             index < bytes; ++index) {
            destination_tail[index] = source_tail[index];
        }
    }
}

void validate_manifest(
    const torch::Tensor& source_addresses,
    const torch::Tensor& destination_addresses,
    const torch::Tensor& byte_counts) {
    const auto valid = [](const torch::Tensor& tensor) {
        return tensor.is_cuda() && tensor.is_contiguous()
            && tensor.scalar_type() == torch::kInt64 && tensor.dim() == 1;
    };
    TORCH_CHECK(
        valid(source_addresses) && valid(destination_addresses)
            && valid(byte_counts),
        "R13 transport manifest ABI mismatch"
    );
    TORCH_CHECK(
        source_addresses.device() == destination_addresses.device()
            && source_addresses.device() == byte_counts.device(),
        "R13 transport manifest tensors must share one CUDA device"
    );
    TORCH_CHECK(
        source_addresses.numel() > 0
            && source_addresses.numel() == destination_addresses.numel()
            && source_addresses.numel() == byte_counts.numel(),
        "R13 transport manifest lengths differ"
    );
}

void copy_to_shadow(
    torch::Tensor source_addresses,
    torch::Tensor destination_addresses,
    torch::Tensor byte_counts) {
    validate_manifest(source_addresses, destination_addresses, byte_counts);
    const int64_t chunks = source_addresses.numel();
    raw_transport_kernel<false><<<
        chunks, kTransportThreads, 0, at::cuda::getCurrentCUDAStream()
    >>>(
        source_addresses.data_ptr<int64_t>(),
        destination_addresses.data_ptr<int64_t>(),
        byte_counts.data_ptr<int64_t>(), chunks, nullptr, 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void publish_if_all_ok(
    torch::Tensor source_addresses,
    torch::Tensor destination_addresses,
    torch::Tensor byte_counts,
    torch::Tensor status) {
    validate_manifest(source_addresses, destination_addresses, byte_counts);
    TORCH_CHECK(
        status.is_cuda() && status.is_contiguous()
            && status.scalar_type() == torch::kInt32 && status.dim() == 1
            && status.numel() > 0
            && status.device() == source_addresses.device(),
        "R13 transport status ABI mismatch"
    );
    const int64_t chunks = source_addresses.numel();
    raw_transport_kernel<true><<<
        chunks, kTransportThreads, 0, at::cuda::getCurrentCUDAStream()
    >>>(
        source_addresses.data_ptr<int64_t>(),
        destination_addresses.data_ptr<int64_t>(),
        byte_counts.data_ptr<int64_t>(), chunks,
        status.data_ptr<int32_t>(), status.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t captured_graph_node_count(int64_t raw_graph) {
    const auto graph = reinterpret_cast<cudaGraph_t>(raw_graph);
    TORCH_CHECK(graph != nullptr, "R13 raw CUDA graph handle is null");
    size_t count = 0;
    C10_CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &count));
    return static_cast<int64_t>(count);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("copy_to_shadow", &copy_to_shadow);
    module.def("publish_if_all_ok", &publish_if_all_ok);
    module.def("captured_graph_node_count", &captured_graph_node_count);
}
