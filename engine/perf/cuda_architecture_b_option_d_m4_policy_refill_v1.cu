#include "cuda_architecture_b_option_d_m4_policy_refill_v1.cuh"

#include "cuda_architecture_b_torch_mt19937_replay_v1.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace eiel::cuda_architecture_b::option_d_m4_policy_refill_v1 {

namespace replay = eiel::cuda_architecture_b::torch_mt19937_replay_v1;

namespace {

enum CounterIndexV1 : int {
    kGeneratedRows = 0,
    kGeneratedBytes = 1,
    kInvalidStateCount = 2,
    kInvalidCounterCount = 3,
    kCounterCount = 4,
};

__device__ __forceinline__ void add_counter_v1(
    unsigned long long* const counters,
    const CounterIndexV1 index,
    const unsigned long long value
) {
    atomicAdd(counters + static_cast<int>(index), value);
}

__global__ void compact_policy_window_refill_kernel_v1(
    const std::uint8_t* __restrict__ source_state,
    float* __restrict__ delta,
    bool* __restrict__ mutation_mask,
    std::uint8_t* __restrict__ crossover_mask_packed,
    std::int64_t* __restrict__ semantic_tape,
    const std::int64_t* __restrict__ absolute_cursor,
    const std::int64_t* __restrict__ semantic_counter,
    std::int64_t* __restrict__ window_base,
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

    const std::int64_t cursor = absolute_cursor[cell];
    const std::int64_t semantic = semantic_counter[cell];
    const std::int64_t base = window_base[cell];
    if (cursor < 0 || semantic < 0 || base < 0 || base > cursor) {
        add_counter_v1(counters, kInvalidCounterCount, 1ULL);
        return;
    }

    unsigned long long generated_rows = 0ULL;
    unsigned long long generated_bytes = 0ULL;
    for (std::int64_t episode = 0; episode < episodes; ++episode) {
        const std::int64_t row = cell * episodes + episode;
        delta[row] = 0.0F;
        mutation_mask[row] = false;
        semantic_tape[row] = policy_row_width;
        if (!replay::generate_packed_crossover_mask_lsb0_v1(
                &state,
                crossover_mask_packed + row * row_bytes,
                policy_row_width
            )) {
            add_counter_v1(counters, kInvalidCounterCount, 1ULL);
            return;
        }
        ++generated_rows;
        generated_bytes += static_cast<unsigned long long>(
            sizeof(float) + sizeof(bool) + sizeof(std::int64_t) + row_bytes
        );
    }
    window_base[cell] = cursor;
    generated_bytes += static_cast<unsigned long long>(sizeof(std::int64_t));

    add_counter_v1(counters, kGeneratedRows, generated_rows);
    add_counter_v1(counters, kGeneratedBytes, generated_bytes);
}

}  // namespace

PolicyWindowRefillResultV1 generate_compact_policy_window_v1(
    const torch::Tensor& source_state,
    torch::Tensor delta,
    torch::Tensor mutation_mask,
    torch::Tensor crossover_mask_packed,
    torch::Tensor post_state_packed,
    torch::Tensor semantic_tape,
    const torch::Tensor& absolute_cursor,
    const torch::Tensor& semantic_counter,
    torch::Tensor window_base,
    const std::int64_t policy_row_width
) {
    TORCH_CHECK(source_state.defined(), "M4 policy refill source state is undefined");
    TORCH_CHECK(source_state.is_cuda(), "M4 policy refill source state must be CUDA resident");
    TORCH_CHECK(source_state.is_contiguous(), "M4 policy refill source state must be contiguous");
    TORCH_CHECK(source_state.scalar_type() == torch::kUInt8, "M4 policy refill source state must be uint8");
    TORCH_CHECK(source_state.dim() == 2, "M4 policy refill source state must be rank-2");
    TORCH_CHECK(source_state.size(1) == replay::kFullStateBytes, "M4 policy refill source state must use canonical 5056-byte rows");
    TORCH_CHECK(delta.defined(), "M4 policy refill delta tape is undefined");
    TORCH_CHECK(delta.is_cuda(), "M4 policy refill delta tape must be CUDA resident");
    TORCH_CHECK(delta.is_contiguous(), "M4 policy refill delta tape must be contiguous");
    TORCH_CHECK(delta.scalar_type() == torch::kFloat32, "M4 policy refill delta tape must be float32");
    TORCH_CHECK(delta.dim() == 3, "M4 policy refill delta tape must be rank-3");
    TORCH_CHECK(delta.size(2) == 1, "M4 policy refill requires compact disabled-mutation delta width 1");
    TORCH_CHECK(mutation_mask.defined(), "M4 policy refill mutation mask is undefined");
    TORCH_CHECK(mutation_mask.is_cuda(), "M4 policy refill mutation mask must be CUDA resident");
    TORCH_CHECK(mutation_mask.is_contiguous(), "M4 policy refill mutation mask must be contiguous");
    TORCH_CHECK(mutation_mask.scalar_type() == torch::kBool, "M4 policy refill mutation mask must be bool");
    TORCH_CHECK(mutation_mask.dim() == 3, "M4 policy refill mutation mask must be rank-3");
    TORCH_CHECK(crossover_mask_packed.defined(), "M4 policy refill crossover tape is undefined");
    TORCH_CHECK(crossover_mask_packed.is_cuda(), "M4 policy refill crossover tape must be CUDA resident");
    TORCH_CHECK(crossover_mask_packed.is_contiguous(), "M4 policy refill crossover tape must be contiguous");
    TORCH_CHECK(crossover_mask_packed.scalar_type() == torch::kUInt8, "M4 policy refill crossover tape must be uint8");
    TORCH_CHECK(crossover_mask_packed.dim() == 3, "M4 policy refill crossover tape must be rank-3");
    TORCH_CHECK(post_state_packed.defined(), "M4 policy refill post-state tape is undefined");
    TORCH_CHECK(post_state_packed.is_cuda(), "M4 policy refill post-state tape must be CUDA resident");
    TORCH_CHECK(post_state_packed.is_contiguous(), "M4 policy refill post-state tape must be contiguous");
    TORCH_CHECK(post_state_packed.scalar_type() == torch::kUInt8, "M4 policy refill post-state tape must be uint8");
    TORCH_CHECK(post_state_packed.dim() == 3, "M4 policy refill post-state tape must be rank-3");
    TORCH_CHECK(post_state_packed.size(2) == 0, "M4 policy refill requires replay-only packed post-state rows");
    TORCH_CHECK(semantic_tape.defined(), "M4 policy refill semantic tape is undefined");
    TORCH_CHECK(semantic_tape.is_cuda(), "M4 policy refill semantic tape must be CUDA resident");
    TORCH_CHECK(semantic_tape.is_contiguous(), "M4 policy refill semantic tape must be contiguous");
    TORCH_CHECK(semantic_tape.scalar_type() == torch::kInt64, "M4 policy refill semantic tape must be int64");
    TORCH_CHECK(semantic_tape.dim() == 2, "M4 policy refill semantic tape must be rank-2");
    TORCH_CHECK(absolute_cursor.defined(), "M4 policy refill absolute cursor is undefined");
    TORCH_CHECK(absolute_cursor.is_cuda(), "M4 policy refill absolute cursor must be CUDA resident");
    TORCH_CHECK(absolute_cursor.is_contiguous(), "M4 policy refill absolute cursor must be contiguous");
    TORCH_CHECK(absolute_cursor.scalar_type() == torch::kInt64, "M4 policy refill absolute cursor must be int64");
    TORCH_CHECK(absolute_cursor.dim() == 1, "M4 policy refill absolute cursor must be rank-1");
    TORCH_CHECK(semantic_counter.defined(), "M4 policy refill semantic counter is undefined");
    TORCH_CHECK(semantic_counter.is_cuda(), "M4 policy refill semantic counter must be CUDA resident");
    TORCH_CHECK(semantic_counter.is_contiguous(), "M4 policy refill semantic counter must be contiguous");
    TORCH_CHECK(semantic_counter.scalar_type() == torch::kInt64, "M4 policy refill semantic counter must be int64");
    TORCH_CHECK(semantic_counter.dim() == 1, "M4 policy refill semantic counter must be rank-1");
    TORCH_CHECK(window_base.defined(), "M4 policy refill window base is undefined");
    TORCH_CHECK(window_base.is_cuda(), "M4 policy refill window base must be CUDA resident");
    TORCH_CHECK(window_base.is_contiguous(), "M4 policy refill window base must be contiguous");
    TORCH_CHECK(window_base.scalar_type() == torch::kInt64, "M4 policy refill window base must be int64");
    TORCH_CHECK(window_base.dim() == 1, "M4 policy refill window base must be rank-1");
    TORCH_CHECK(policy_row_width > 0, "M4 policy refill row width must be positive");

    const std::int64_t cells = source_state.size(0);
    const std::int64_t episodes = delta.size(1);
    const std::int64_t row_bytes = replay::packed_lsb0_row_bytes_v1(policy_row_width);
    TORCH_CHECK(cells > 0, "M4 policy refill requires at least one cell");
    TORCH_CHECK(episodes > 0, "M4 policy refill requires at least one episode");
    TORCH_CHECK(row_bytes > 0, "M4 policy refill row width is outside the bounded range");
    TORCH_CHECK(cells <= std::numeric_limits<int>::max(), "M4 policy refill cell count exceeds CUDA grid limit");
    TORCH_CHECK(delta.size(0) == cells, "M4 policy refill delta cell count mismatch");
    TORCH_CHECK(mutation_mask.size(0) == cells, "M4 policy refill mutation mask cell count mismatch");
    TORCH_CHECK(mutation_mask.size(1) == episodes, "M4 policy refill mutation mask episode count mismatch");
    TORCH_CHECK(mutation_mask.size(2) == 1, "M4 policy refill mutation mask compact width mismatch");
    TORCH_CHECK(crossover_mask_packed.size(0) == cells, "M4 policy refill crossover cell count mismatch");
    TORCH_CHECK(crossover_mask_packed.size(1) == episodes, "M4 policy refill crossover episode count mismatch");
    TORCH_CHECK(crossover_mask_packed.size(2) == row_bytes, "M4 policy refill crossover row width mismatch");
    TORCH_CHECK(post_state_packed.size(0) == cells, "M4 policy refill post-state cell count mismatch");
    TORCH_CHECK(post_state_packed.size(1) == episodes, "M4 policy refill post-state episode count mismatch");
    TORCH_CHECK(semantic_tape.size(0) == cells, "M4 policy refill semantic cell count mismatch");
    TORCH_CHECK(semantic_tape.size(1) == episodes, "M4 policy refill semantic episode count mismatch");
    TORCH_CHECK(absolute_cursor.size(0) == cells, "M4 policy refill absolute cursor cell count mismatch");
    TORCH_CHECK(semantic_counter.size(0) == cells, "M4 policy refill semantic counter cell count mismatch");
    TORCH_CHECK(window_base.size(0) == cells, "M4 policy refill window base cell count mismatch");
    TORCH_CHECK(source_state.device() == delta.device(), "M4 policy refill tensors must share one CUDA device");
    TORCH_CHECK(source_state.device() == mutation_mask.device(), "M4 policy refill mutation mask must share the CUDA device");
    TORCH_CHECK(source_state.device() == crossover_mask_packed.device(), "M4 policy refill crossover tape must share the CUDA device");
    TORCH_CHECK(source_state.device() == post_state_packed.device(), "M4 policy refill post-state tape must share the CUDA device");
    TORCH_CHECK(source_state.device() == semantic_tape.device(), "M4 policy refill semantic tape must share the CUDA device");
    TORCH_CHECK(source_state.device() == absolute_cursor.device(), "M4 policy refill absolute cursor must share the CUDA device");
    TORCH_CHECK(source_state.device() == semantic_counter.device(), "M4 policy refill semantic counter must share the CUDA device");
    TORCH_CHECK(source_state.device() == window_base.device(), "M4 policy refill window base must share the CUDA device");

    const c10::cuda::CUDAGuard device_guard(source_state.device());
    const auto counters = torch::zeros(
        {kCounterCount},
        source_state.options().dtype(torch::kInt64).requires_grad(false)
    );
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    compact_policy_window_refill_kernel_v1<<<
        static_cast<unsigned int>(cells),
        1,
        0,
        stream
    >>>(
        source_state.data_ptr<std::uint8_t>(),
        delta.data_ptr<float>(),
        mutation_mask.data_ptr<bool>(),
        crossover_mask_packed.data_ptr<std::uint8_t>(),
        semantic_tape.data_ptr<std::int64_t>(),
        absolute_cursor.data_ptr<std::int64_t>(),
        semantic_counter.data_ptr<std::int64_t>(),
        window_base.data_ptr<std::int64_t>(),
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
        values[kGeneratedRows],
        values[kGeneratedBytes],
        values[kInvalidStateCount],
        values[kInvalidCounterCount],
    };
}

}  // namespace eiel::cuda_architecture_b::option_d_m4_policy_refill_v1
