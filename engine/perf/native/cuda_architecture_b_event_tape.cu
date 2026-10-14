#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int32_t STATUS_OK = 0;
constexpr int32_t STATUS_CAPACITY_EXCEEDED = 1;

__global__ void validate_append_kernel(
    const int64_t* __restrict__ active_count,
    int32_t* __restrict__ status,
    const int64_t row_count,
    const int64_t capacity
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int64_t cursor = active_count[0];
        status[0] = cursor >= 0 && row_count >= 0 && cursor <= capacity - row_count
            ? STATUS_OK
            : STATUS_CAPACITY_EXCEEDED;
    }
}

template <typename scalar_t>
__global__ void append_column_kernel(
    scalar_t* __restrict__ destination,
    const scalar_t* __restrict__ source,
    const int64_t* __restrict__ active_count,
    const int32_t* __restrict__ status,
    const int64_t row_count
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < row_count && status[0] == STATUS_OK) {
        destination[active_count[0] + row] = source[row];
    }
}

__global__ void commit_append_kernel(
    int64_t* __restrict__ active_count,
    const int32_t* __restrict__ status,
    const int64_t row_count
) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && status[0] == STATUS_OK) {
        active_count[0] += row_count;
    }
}

void check_cuda_vector(const torch::Tensor& tensor, const char* label) {
    TORCH_CHECK(tensor.is_cuda(), label, " must be CUDA resident");
    TORCH_CHECK(tensor.is_contiguous(), label, " must be contiguous");
    TORCH_CHECK(tensor.dim() == 1, label, " must be one-dimensional");
}

}  // namespace

std::vector<torch::Tensor> eiel_cuda_architecture_b_event_tape_append_v1(
    const std::vector<torch::Tensor>& tape_columns,
    const std::vector<torch::Tensor>& event_rows,
    torch::Tensor active_count,
    torch::Tensor status
) {
    TORCH_CHECK(!tape_columns.empty(), "event tape requires at least one column");
    TORCH_CHECK(tape_columns.size() == event_rows.size(), "event tape row/column count mismatch");
    check_cuda_vector(active_count, "active_count");
    check_cuda_vector(status, "status");
    TORCH_CHECK(active_count.scalar_type() == torch::kInt64 && active_count.numel() == 1, "active_count must be int64[1]");
    TORCH_CHECK(status.scalar_type() == torch::kInt32 && status.numel() == 1, "status must be int32[1]");

    const auto device = active_count.device();
    const int64_t capacity = tape_columns.front().numel();
    const int64_t row_count = event_rows.front().numel();
    TORCH_CHECK(status.device() == device, "event tape status device mismatch");
    for (std::size_t index = 0; index < tape_columns.size(); ++index) {
        const auto& destination = tape_columns[index];
        const auto& source = event_rows[index];
        check_cuda_vector(destination, "event tape column");
        check_cuda_vector(source, "event row column");
        TORCH_CHECK(destination.device() == device && source.device() == device, "event tape device mismatch");
        TORCH_CHECK(destination.numel() == capacity, "event tape destination capacity mismatch");
        TORCH_CHECK(source.numel() == row_count, "event tape source row-count mismatch");
        TORCH_CHECK(destination.scalar_type() == source.scalar_type(), "event tape column dtype mismatch");
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    validate_append_kernel<<<1, 1, 0, stream>>>(
        active_count.data_ptr<int64_t>(),
        status.data_ptr<int32_t>(),
        row_count,
        capacity
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (row_count > 0) {
        constexpr int threads = 128;
        const int blocks = static_cast<int>((row_count + threads - 1) / threads);
        for (std::size_t index = 0; index < tape_columns.size(); ++index) {
            const auto& destination = tape_columns[index];
            const auto& source = event_rows[index];
            AT_DISPATCH_ALL_TYPES(
                destination.scalar_type(),
                "eiel_cuda_event_tape_append_column",
                [&] {
                    append_column_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                        destination.data_ptr<scalar_t>(),
                        source.data_ptr<scalar_t>(),
                        active_count.data_ptr<int64_t>(),
                        status.data_ptr<int32_t>(),
                        row_count
                    );
                }
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
    commit_append_kernel<<<1, 1, 0, stream>>>(
        active_count.data_ptr<int64_t>(),
        status.data_ptr<int32_t>(),
        row_count
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {active_count, status};
}
