#include "cuda_architecture_b_sha256_v1.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace sha256 = eiel::cuda_architecture_b::sha256_v1;

namespace {

__global__ void cuda_architecture_b_sha256_v1_probe_kernel(
    const std::uint8_t* input,
    const std::uint64_t input_bytes,
    const std::uint64_t split_bytes,
    std::uint8_t* output,
    std::int32_t* status
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    sha256::Sha256ContextV1 context;
    sha256::sha256_init_v1(&context);
    if (input_bytes == 0) {
        sha256::sha256_update_v1(&context, nullptr, 0);
    } else {
        sha256::sha256_update_v1(&context, input, split_bytes);
        sha256::sha256_update_v1(
            &context,
            input + split_bytes,
            input_bytes - split_bytes
        );
    }
    sha256::sha256_final_v1(&context, output);
    status[0] = 0;
}

}  // namespace

void eiel_cuda_architecture_b_sha256_v1_launch_probe(
    const std::uint8_t* input,
    const std::uint64_t input_bytes,
    const std::uint64_t split_bytes,
    std::uint8_t* output,
    std::int32_t* status,
    cudaStream_t stream
) {
    cuda_architecture_b_sha256_v1_probe_kernel<<<1, 1, 0, stream>>>(
        input,
        input_bytes,
        split_bytes,
        output,
        status
    );
}

cudaError_t eiel_cuda_architecture_b_sha256_v1_probe_attributes(
    cudaFuncAttributes* attributes
) {
    return cudaFuncGetAttributes(
        attributes,
        cuda_architecture_b_sha256_v1_probe_kernel
    );
}
