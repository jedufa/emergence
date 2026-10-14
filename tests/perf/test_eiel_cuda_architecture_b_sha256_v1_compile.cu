#include "../../engine/perf/cuda_architecture_b_sha256_v1.cuh"

#include <cstdint>
#include <type_traits>

namespace sha256 = eiel::cuda_architecture_b::sha256_v1;

static_assert(sha256::kBlockBytesV1 == 64);
static_assert(sha256::kDigestBytesV1 == 32);
static_assert(sizeof(sha256::Sha256ContextV1) == 112);
static_assert(std::is_standard_layout_v<sha256::Sha256ContextV1>);
static_assert(std::is_trivially_copyable_v<sha256::Sha256ContextV1>);

__global__ void compile_cuda_architecture_b_sha256_v1_probe(
    const std::uint8_t* bytes,
    const std::uint64_t byte_count,
    std::uint8_t* digest
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    sha256::Sha256ContextV1 context;
    sha256::sha256_init_v1(&context);
    sha256::sha256_update_v1(&context, nullptr, 0);
    if (byte_count != 0) {
        const std::uint64_t first = byte_count / 3;
        const std::uint64_t second = byte_count / 3;
        sha256::sha256_update_v1(&context, bytes, first);
        sha256::sha256_update_v1(&context, bytes + first, second);
        sha256::sha256_update_v1(
            &context,
            bytes + first + second,
            byte_count - first - second
        );
    }
    sha256::sha256_final_v1(&context, digest);
}
