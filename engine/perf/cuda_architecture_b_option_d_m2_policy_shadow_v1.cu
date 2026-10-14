#include "cuda_architecture_b_option_d_m2_policy_shadow_v1.cuh"

#include "cuda_architecture_b_torch_mt19937_replay_v1.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace eiel::cuda_architecture_b::option_d_m2_policy_shadow_v1 {

namespace replay = eiel::cuda_architecture_b::torch_mt19937_replay_v1;

namespace {

enum CounterIndexV1 : int {
    kCheckedRows = 0,
    kCheckedBytes = 1,
    kMismatchCount = 2,
    kInvalidStateCount = 3,
    kInvalidSemanticCount = 4,
    kCounterCount = 5,
};

__device__ __forceinline__ void add_counter_v1(
    unsigned long long* const counters,
    const CounterIndexV1 index,
    const unsigned long long value
) {
    atomicAdd(counters + static_cast<int>(index), value);
}

__global__ void policy_segment_shadow_kernel_v1(
    const std::uint8_t* __restrict__ source_state,
    const std::uint8_t* __restrict__ expected_crossover_mask_packed,
    const std::int64_t* __restrict__ semantic_tape,
    unsigned long long* __restrict__ counters,
    const std::int64_t cells,
    const std::int64_t episodes,
    const std::int64_t row_bytes,
    const std::int64_t policy_row_width
) {
    if (threadIdx.x != 0) {
        return;
    }
    const std::int64_t cell = static_cast<std::int64_t>(blockIdx.x);
    if (cell < 0 || cell >= cells) {
        return;
    }

    replay::TorchMt19937StateV1 state{};
    if (!replay::load_canonical_state_v1(
            source_state + cell * replay::kFullStateBytes,
            replay::kFullStateBytes,
            &state
        )) {
        add_counter_v1(counters, kInvalidStateCount, 1ULL);
        return;
    }

    unsigned long long checked_rows = 0ULL;
    unsigned long long checked_bytes = 0ULL;
    unsigned long long mismatches = 0ULL;
    for (std::int64_t episode = 0; episode < episodes; ++episode) {
        const std::int64_t semantic =
            semantic_tape[cell * episodes + episode];
        if (semantic != policy_row_width) {
            add_counter_v1(counters, kCheckedRows, checked_rows);
            add_counter_v1(counters, kCheckedBytes, checked_bytes);
            add_counter_v1(counters, kMismatchCount, mismatches);
            add_counter_v1(counters, kInvalidSemanticCount, 1ULL);
            return;
        }
        for (std::int64_t byte_index = 0; byte_index < row_bytes; ++byte_index) {
            std::uint8_t generated = 0U;
            for (int bit = 0; bit < 8; ++bit) {
                const std::int64_t column = byte_index * 8 + bit;
                if (column >= policy_row_width) {
                    break;
                }
                const std::uint32_t raw = replay::next_uint32_v1(&state);
                if (replay::torch_rand_float32_less_than_half_v1(raw)) {
                    generated = static_cast<std::uint8_t>(
                        generated | static_cast<std::uint8_t>(1U << bit)
                    );
                }
            }
            const std::uint8_t expected = expected_crossover_mask_packed[
                (cell * episodes + episode) * row_bytes + byte_index
            ];
            if (generated != expected) {
                ++mismatches;
            }
        }
        ++checked_rows;
        checked_bytes += static_cast<unsigned long long>(row_bytes);
    }

    add_counter_v1(counters, kCheckedRows, checked_rows);
    add_counter_v1(counters, kCheckedBytes, checked_bytes);
    add_counter_v1(counters, kMismatchCount, mismatches);
}

}  // namespace

PolicySegmentShadowResultV1 verify_policy_segment_shadow_v1(
    const torch::Tensor& source_state,
    const torch::Tensor& expected_crossover_mask_packed,
    const torch::Tensor& semantic_tape,
    const std::int64_t policy_row_width
) {
    TORCH_CHECK(source_state.defined(), "M2 policy shadow source state is undefined");
    TORCH_CHECK(source_state.is_cuda(), "M2 policy shadow source state must be CUDA resident");
    TORCH_CHECK(source_state.is_contiguous(), "M2 policy shadow source state must be contiguous");
    TORCH_CHECK(source_state.scalar_type() == torch::kUInt8, "M2 policy shadow source state must be uint8");
    TORCH_CHECK(source_state.dim() == 2, "M2 policy shadow source state must be rank-2");
    TORCH_CHECK(source_state.size(1) == replay::kFullStateBytes, "M2 policy shadow source state must use canonical 5056-byte rows");
    TORCH_CHECK(expected_crossover_mask_packed.defined(), "M2 policy shadow crossover tape is undefined");
    TORCH_CHECK(expected_crossover_mask_packed.is_cuda(), "M2 policy shadow crossover tape must be CUDA resident");
    TORCH_CHECK(expected_crossover_mask_packed.is_contiguous(), "M2 policy shadow crossover tape must be contiguous");
    TORCH_CHECK(expected_crossover_mask_packed.scalar_type() == torch::kUInt8, "M2 policy shadow crossover tape must be uint8");
    TORCH_CHECK(expected_crossover_mask_packed.dim() == 3, "M2 policy shadow crossover tape must be rank-3");
    TORCH_CHECK(semantic_tape.defined(), "M2 policy shadow semantic tape is undefined");
    TORCH_CHECK(semantic_tape.is_cuda(), "M2 policy shadow semantic tape must be CUDA resident");
    TORCH_CHECK(semantic_tape.is_contiguous(), "M2 policy shadow semantic tape must be contiguous");
    TORCH_CHECK(semantic_tape.scalar_type() == torch::kInt64, "M2 policy shadow semantic tape must be int64");
    TORCH_CHECK(semantic_tape.dim() == 2, "M2 policy shadow semantic tape must be rank-2");
    TORCH_CHECK(policy_row_width > 0, "M2 policy shadow row width must be positive");
    const std::int64_t cells = source_state.size(0);
    const std::int64_t episodes = expected_crossover_mask_packed.size(1);
    const std::int64_t row_bytes = replay::packed_lsb0_row_bytes_v1(policy_row_width);
    TORCH_CHECK(cells > 0, "M2 policy shadow requires at least one cell");
    TORCH_CHECK(episodes > 0, "M2 policy shadow requires at least one episode");
    TORCH_CHECK(cells <= std::numeric_limits<int>::max(), "M2 policy shadow cell count exceeds CUDA grid limit");
    TORCH_CHECK(expected_crossover_mask_packed.size(0) == cells, "M2 policy shadow crossover cell count mismatch");
    TORCH_CHECK(expected_crossover_mask_packed.size(2) == row_bytes, "M2 policy shadow crossover row width mismatch");
    TORCH_CHECK(semantic_tape.size(0) == cells, "M2 policy shadow semantic cell count mismatch");
    TORCH_CHECK(semantic_tape.size(1) == episodes, "M2 policy shadow semantic episode count mismatch");
    TORCH_CHECK(source_state.device() == expected_crossover_mask_packed.device(), "M2 policy shadow tensors must share one CUDA device");
    TORCH_CHECK(source_state.device() == semantic_tape.device(), "M2 policy shadow semantic tape must share the CUDA device");

    const c10::cuda::CUDAGuard device_guard(source_state.device());
    const auto counters = torch::zeros(
        {kCounterCount},
        source_state.options().dtype(torch::kInt64).requires_grad(false)
    );
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    policy_segment_shadow_kernel_v1<<<
        static_cast<unsigned int>(cells),
        1,
        0,
        stream
    >>>(
        source_state.data_ptr<std::uint8_t>(),
        expected_crossover_mask_packed.data_ptr<std::uint8_t>(),
        semantic_tape.data_ptr<std::int64_t>(),
        reinterpret_cast<unsigned long long*>(counters.data_ptr<std::int64_t>()),
        cells,
        episodes,
        row_bytes,
        policy_row_width
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const auto host = counters.cpu().contiguous();
    const auto* const values = host.data_ptr<std::int64_t>();
    return {
        values[kCheckedRows],
        values[kCheckedBytes],
        values[kMismatchCount],
        values[kInvalidStateCount],
        values[kInvalidSemanticCount],
    };
}

}  // namespace eiel::cuda_architecture_b::option_d_m2_policy_shadow_v1
