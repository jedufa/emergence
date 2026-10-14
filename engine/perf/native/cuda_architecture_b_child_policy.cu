#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "cuda_architecture_b_torch_mt19937_replay_v1.cuh"
#include "cuda_architecture_b_torch_rng_state_pack_v1.cuh"

namespace {

namespace torch_rng_pack =
    eiel::cuda_architecture_b::torch_rng_state_pack_v1;
namespace torch_rng_replay =
    eiel::cuda_architecture_b::torch_mt19937_replay_v1;

constexpr int32_t STATUS_OK = 0;
constexpr int32_t STATUS_INVALID_PLAN = 1;
constexpr int32_t STATUS_TAPE_EXHAUSTED = 2;

__global__ void fixed_mlp_child_policy_kernel(
    const int64_t* __restrict__ slot_to_agent_id,
    bool* __restrict__ alive_mask,
    const int64_t* __restrict__ parent_slots,
    const int64_t* __restrict__ mate_slots,
    const int64_t* __restrict__ child_slots,
    float* __restrict__ mlp_pack,
    float* __restrict__ hmem,
    bool* __restrict__ policy_use_gru,
    uint8_t* __restrict__ mutation_torch_state,
    const float* __restrict__ delta_tape,
    const bool* __restrict__ mutation_mask_tape,
    const uint8_t* __restrict__ crossover_mask_packed_tape,
    const uint8_t* __restrict__ packed_post_state_tape,
    const int64_t* __restrict__ semantic_draws_tape,
    int64_t* __restrict__ tape_cursor,
    int64_t* __restrict__ semantic_draws_total,
    int64_t* __restrict__ committed_count,
    int64_t* __restrict__ semantic_draws_added,
    int64_t* __restrict__ mutation_events,
    int64_t* __restrict__ mutated_elements,
    double* __restrict__ mutation_delta_total,
    bool* __restrict__ mutation_occurred_by_birth,
    float* __restrict__ mutation_delta_by_birth,
    int32_t* __restrict__ status,
    const int64_t* __restrict__ planned_count,
    const int64_t plan_capacity,
    const int64_t slot_count,
    const int64_t row_width,
    const int64_t hmem_width,
    const int64_t state_bytes,
    const int64_t tape_capacity,
    const int64_t mutation_tape_width,
    const bool turnover_mode
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const int64_t birth_count = planned_count[0];
    if (
        birth_count < 0 || birth_count > plan_capacity
        || birth_count > slot_count || tape_cursor[0] < 0
    ) {
        status[0] = STATUS_INVALID_PLAN;
        return;
    }
    if (tape_cursor[0] + birth_count > tape_capacity) {
        status[0] = STATUS_TAPE_EXHAUSTED;
        return;
    }

    int64_t previous_parent_id = -1;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = parent_slots[birth];
        const int64_t child_slot = child_slots[birth];
        if (
            parent_slot < 0 || parent_slot >= slot_count
            || child_slot < 0 || child_slot >= slot_count
            || parent_slot == child_slot
            || !alive_mask[parent_slot] || !alive_mask[child_slot]
            || slot_to_agent_id[parent_slot] < 0 || slot_to_agent_id[child_slot] < 0
            || policy_use_gru[parent_slot]
        ) {
            status[0] = STATUS_INVALID_PLAN;
            return;
        }
        const int64_t parent_id = slot_to_agent_id[parent_slot];
        if (turnover_mode) {
            const int64_t mate_slot = mate_slots[birth];
            if (
                mate_slot < 0 || mate_slot >= slot_count
                || mate_slot == parent_slot || mate_slot == child_slot
                || !alive_mask[mate_slot] || slot_to_agent_id[mate_slot] < 0
                || policy_use_gru[mate_slot]
                || slot_to_agent_id[mate_slot] == parent_id
                || slot_to_agent_id[child_slot] == parent_id
                || slot_to_agent_id[child_slot] == slot_to_agent_id[mate_slot]
            ) {
                status[0] = STATUS_INVALID_PLAN;
                return;
            }
            for (int64_t destination = 0; destination < birth_count; ++destination) {
                if (
                    parent_slot == child_slots[destination]
                    || mate_slot == child_slots[destination]
                ) {
                    status[0] = STATUS_INVALID_PLAN;
                    return;
                }
            }
        } else {
            if (birth > 0 && parent_id <= previous_parent_id) {
                status[0] = STATUS_INVALID_PLAN;
                return;
            }
            previous_parent_id = parent_id;
        }
        for (int64_t previous = 0; previous < birth; ++previous) {
            if (child_slots[previous] == child_slot) {
                status[0] = STATUS_INVALID_PLAN;
                return;
            }
        }
    }

    const int64_t initial_cursor = tape_cursor[0];
    int64_t semantic_added = 0;
    int64_t event_count = 0;
    int64_t element_count = 0;
    double delta_total = 0.0;
    const int64_t packed_crossover_row_bytes = (row_width + 7) / 8;
    const bool replay_terminal_state = packed_post_state_tape == nullptr;
    if (
        state_bytes != torch_rng_pack::kFullStateBytes
        || mutation_tape_width < 1
        || (mutation_tape_width != 1 && mutation_tape_width != row_width)
        || (replay_terminal_state && mutation_tape_width != 1)
    ) {
        status[0] = STATUS_INVALID_PLAN;
        return;
    }
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = parent_slots[birth];
        const int64_t mate_slot = turnover_mode ? mate_slots[birth] : parent_slot;
        const int64_t child_slot = child_slots[birth];
        const int64_t episode = initial_cursor + birth;
        bool row_mutated = false;
        double row_delta_total = 0.0;
        for (int64_t column = 0; column < row_width; ++column) {
            const int64_t tape_index = episode * mutation_tape_width
                + (mutation_tape_width == 1 ? 0 : column);
            const int64_t packed_index =
                episode * packed_crossover_row_bytes + column / 8;
            const bool inherit_from_mate =
                turnover_mode
                && (
                    (
                        crossover_mask_packed_tape[packed_index]
                        >> (column & 7)
                    ) & 1U
                ) == 0U;
            const float parent_value = mlp_pack[parent_slot * row_width + column];
            const float inherited_value = inherit_from_mate
                ? mlp_pack[mate_slot * row_width + column]
                : parent_value;
            const float delta = delta_tape[tape_index];
            if (
                replay_terminal_state
                && (mutation_mask_tape[tape_index] || delta != 0.0f)
            ) {
                status[0] = STATUS_INVALID_PLAN;
                return;
            }
            mlp_pack[child_slot * row_width + column] = __fadd_rn(inherited_value, delta);
            if (mutation_mask_tape[tape_index]) {
                row_mutated = true;
                element_count += 1;
                delta_total += static_cast<double>(delta);
                row_delta_total += static_cast<double>(delta);
            }
        }
        for (int64_t column = 0; column < hmem_width; ++column) {
            hmem[child_slot * hmem_width + column] = 0.0f;
        }
        policy_use_gru[child_slot] = false;
        mutation_occurred_by_birth[birth] = row_mutated;
        mutation_delta_by_birth[birth] = static_cast<float>(row_delta_total);
        semantic_added += semantic_draws_tape[episode];
        event_count += row_mutated ? 1 : 0;
        if (replay_terminal_state) {
            if (semantic_draws_tape[episode] != row_width || row_mutated) {
                status[0] = STATUS_INVALID_PLAN;
                return;
            }
        } else if (!torch_rng_pack::copy_unpacked_state_v1(
                packed_post_state_tape
                    + episode * torch_rng_pack::kPackedStateBytes,
                torch_rng_pack::kPackedStateBytes,
                mutation_torch_state,
                state_bytes
            )) {
            status[0] = STATUS_INVALID_PLAN;
            return;
        }
    }
    if (
        replay_terminal_state
        && !torch_rng_replay::advance_canonical_state_v1(
            mutation_torch_state,
            state_bytes,
            static_cast<std::uint64_t>(semantic_added)
        )
    ) {
        status[0] = STATUS_INVALID_PLAN;
        return;
    }

    tape_cursor[0] = initial_cursor + birth_count;
    semantic_draws_total[0] += semantic_added;
    committed_count[0] = birth_count;
    semantic_draws_added[0] = semantic_added;
    mutation_events[0] = event_count;
    mutated_elements[0] = element_count;
    mutation_delta_total[0] = delta_total;
    status[0] = STATUS_OK;
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA-resident");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::vector<torch::Tensor> eiel_cuda_architecture_b_fixed_mlp_child_policy_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor alive_mask,
    torch::Tensor parent_slots,
    torch::Tensor child_slots,
    torch::Tensor planned_count,
    torch::Tensor mlp_pack,
    torch::Tensor hmem,
    torch::Tensor policy_use_gru,
    torch::Tensor rng_manager_mutation_torch_cpu_state,
    torch::Tensor policy_birth_delta_tape,
    torch::Tensor policy_birth_mutation_mask_tape,
    torch::Tensor policy_birth_post_rng_state_packed_tape,
    torch::Tensor policy_birth_semantic_draws_tape,
    torch::Tensor policy_birth_tape_cursor,
    torch::Tensor mutation_torch_cpu_semantic_draws
) {
    check_cuda_contiguous(slot_to_agent_id, "slot_to_agent_id");
    TORCH_CHECK(slot_to_agent_id.scalar_type() == torch::kInt64, "slot_to_agent_id must be int64");
    TORCH_CHECK(slot_to_agent_id.dim() == 1 && slot_to_agent_id.numel() > 0, "slot_to_agent_id must be nonempty rank 1");
    const int64_t slot_count = slot_to_agent_id.numel();
    const auto device = slot_to_agent_id.device();

    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"alive_mask", alive_mask},
             {"parent_slots", parent_slots},
             {"child_slots", child_slots},
             {"planned_count", planned_count},
             {"mlp_pack", mlp_pack},
             {"hmem", hmem},
             {"policy_use_gru", policy_use_gru},
             {"rng_manager_mutation_torch_cpu_state", rng_manager_mutation_torch_cpu_state},
             {"policy_birth_delta_tape", policy_birth_delta_tape},
             {"policy_birth_mutation_mask_tape", policy_birth_mutation_mask_tape},
             {"policy_birth_post_rng_state_packed_tape", policy_birth_post_rng_state_packed_tape},
             {"policy_birth_semantic_draws_tape", policy_birth_semantic_draws_tape},
             {"policy_birth_tape_cursor", policy_birth_tape_cursor},
             {"mutation_torch_cpu_semantic_draws", mutation_torch_cpu_semantic_draws},
         }) {
        check_cuda_contiguous(named.second, named.first);
        TORCH_CHECK(named.second.device() == device, named.first, " must share the child-policy CUDA device");
    }
    TORCH_CHECK(alive_mask.scalar_type() == torch::kBool && alive_mask.numel() == slot_count, "alive_mask must be bool[capacity]");
    TORCH_CHECK(policy_use_gru.scalar_type() == torch::kBool && policy_use_gru.numel() == slot_count, "policy_use_gru must be bool[capacity]");
    TORCH_CHECK(parent_slots.scalar_type() == torch::kInt64 && child_slots.scalar_type() == torch::kInt64, "birth slots must be int64");
    TORCH_CHECK(parent_slots.dim() == 1 && child_slots.sizes() == parent_slots.sizes(), "birth slots must be aligned rank 1");
    TORCH_CHECK(planned_count.scalar_type() == torch::kInt64 && planned_count.dim() == 1 && planned_count.numel() == 1, "planned_count must be int64[1]");
    TORCH_CHECK(mlp_pack.scalar_type() == torch::kFloat32 && mlp_pack.dim() == 2 && mlp_pack.size(0) == slot_count, "mlp_pack must be float32[capacity,width]");
    TORCH_CHECK(hmem.scalar_type() == torch::kFloat32 && hmem.dim() == 2 && hmem.size(0) == slot_count, "hmem must be float32[capacity,width]");
    TORCH_CHECK(rng_manager_mutation_torch_cpu_state.scalar_type() == torch::kUInt8 && rng_manager_mutation_torch_cpu_state.dim() == 1 && rng_manager_mutation_torch_cpu_state.numel() == torch_rng_pack::kFullStateBytes, "mutation Torch state must be canonical 5056-byte rank-1 uint8");
    TORCH_CHECK(policy_birth_delta_tape.scalar_type() == torch::kFloat32 && policy_birth_delta_tape.dim() == 2, "policy delta tape must be rank-2 float32");
    TORCH_CHECK(policy_birth_mutation_mask_tape.scalar_type() == torch::kBool && policy_birth_mutation_mask_tape.sizes() == policy_birth_delta_tape.sizes(), "policy mutation mask must align with delta tape");
    TORCH_CHECK(policy_birth_delta_tape.size(1) == 1 || policy_birth_delta_tape.size(1) == mlp_pack.size(1), "policy mutation tape width must be compact width 1 or match mlp_pack");
    TORCH_CHECK(policy_birth_post_rng_state_packed_tape.scalar_type() == torch::kUInt8 && policy_birth_post_rng_state_packed_tape.dim() == 2, "packed post-state tape must be rank-2 uint8");
    const bool replay_terminal_state = policy_birth_post_rng_state_packed_tape.numel() == 0;
    if (replay_terminal_state) {
        TORCH_CHECK(policy_birth_delta_tape.size(1) == 1, "post-state replay requires compact disabled-mutation tape");
    } else {
        TORCH_CHECK(policy_birth_post_rng_state_packed_tape.size(0) == policy_birth_delta_tape.size(0) && policy_birth_post_rng_state_packed_tape.size(1) == torch_rng_pack::kPackedStateBytes, "packed post-state tape must align with policy episodes and use 2560-byte rows");
    }
    TORCH_CHECK(policy_birth_semantic_draws_tape.scalar_type() == torch::kInt64 && policy_birth_semantic_draws_tape.dim() == 1 && policy_birth_semantic_draws_tape.size(0) == policy_birth_delta_tape.size(0), "semantic-draw tape must align with policy episodes");
    TORCH_CHECK(policy_birth_tape_cursor.scalar_type() == torch::kInt64 && policy_birth_tape_cursor.numel() == 1, "policy_birth_tape_cursor must be int64[1]");
    TORCH_CHECK(mutation_torch_cpu_semantic_draws.scalar_type() == torch::kInt64 && mutation_torch_cpu_semantic_draws.numel() == 1, "mutation_torch_cpu_semantic_draws must be int64[1]");

    c10::cuda::CUDAGuard device_guard(device);
    auto int64_options = slot_to_agent_id.options().dtype(torch::kInt64);
    auto int32_options = slot_to_agent_id.options().dtype(torch::kInt32);
    auto float64_options = slot_to_agent_id.options().dtype(torch::kFloat64);
    auto committed_count = torch::zeros({1}, int64_options);
    auto semantic_draws_added = torch::zeros({1}, int64_options);
    auto mutation_events = torch::zeros({1}, int64_options);
    auto mutated_elements = torch::zeros({1}, int64_options);
    auto mutation_delta_total = torch::zeros({1}, float64_options);
    auto mutation_occurred_by_birth = torch::zeros(
        {parent_slots.numel()},
        slot_to_agent_id.options().dtype(torch::kBool)
    );
    auto mutation_delta_by_birth = torch::zeros(
        {parent_slots.numel()},
        slot_to_agent_id.options().dtype(torch::kFloat32)
    );
    auto status = torch::zeros({1}, int32_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fixed_mlp_child_policy_kernel<<<1, 1, 0, stream>>>(
        slot_to_agent_id.data_ptr<int64_t>(),
        alive_mask.data_ptr<bool>(),
        parent_slots.data_ptr<int64_t>(),
        nullptr,
        child_slots.data_ptr<int64_t>(),
        mlp_pack.data_ptr<float>(),
        hmem.data_ptr<float>(),
        policy_use_gru.data_ptr<bool>(),
        rng_manager_mutation_torch_cpu_state.data_ptr<uint8_t>(),
        policy_birth_delta_tape.data_ptr<float>(),
        policy_birth_mutation_mask_tape.data_ptr<bool>(),
        nullptr,
        replay_terminal_state
            ? nullptr
            : policy_birth_post_rng_state_packed_tape.data_ptr<uint8_t>(),
        policy_birth_semantic_draws_tape.data_ptr<int64_t>(),
        policy_birth_tape_cursor.data_ptr<int64_t>(),
        mutation_torch_cpu_semantic_draws.data_ptr<int64_t>(),
        committed_count.data_ptr<int64_t>(),
        semantic_draws_added.data_ptr<int64_t>(),
        mutation_events.data_ptr<int64_t>(),
        mutated_elements.data_ptr<int64_t>(),
        mutation_delta_total.data_ptr<double>(),
        mutation_occurred_by_birth.data_ptr<bool>(),
        mutation_delta_by_birth.data_ptr<float>(),
        status.data_ptr<int32_t>(),
        planned_count.data_ptr<int64_t>(),
        parent_slots.numel(),
        slot_count,
        mlp_pack.size(1),
        hmem.size(1),
        rng_manager_mutation_torch_cpu_state.numel(),
        policy_birth_delta_tape.size(0),
        policy_birth_delta_tape.size(1),
        false
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        committed_count,
        semantic_draws_added,
        mutation_events,
        mutated_elements,
        mutation_delta_total,
        mutation_occurred_by_birth,
        mutation_delta_by_birth,
        status,
    };
}

std::vector<torch::Tensor> eiel_cuda_architecture_b_fixed_mlp_turnover_child_policy_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor alive_mask,
    torch::Tensor parent_slots,
    torch::Tensor mate_slots,
    torch::Tensor child_slots,
    torch::Tensor planned_count,
    torch::Tensor mlp_pack,
    torch::Tensor hmem,
    torch::Tensor policy_use_gru,
    torch::Tensor rng_manager_mutation_torch_cpu_state,
    torch::Tensor policy_birth_delta_tape,
    torch::Tensor policy_birth_mutation_mask_tape,
    torch::Tensor policy_birth_crossover_mask_packed_tape,
    torch::Tensor policy_birth_post_rng_state_packed_tape,
    torch::Tensor policy_birth_semantic_draws_tape,
    torch::Tensor policy_birth_tape_cursor,
    torch::Tensor mutation_torch_cpu_semantic_draws
) {
    check_cuda_contiguous(slot_to_agent_id, "slot_to_agent_id");
    TORCH_CHECK(slot_to_agent_id.scalar_type() == torch::kInt64, "slot_to_agent_id must be int64");
    TORCH_CHECK(slot_to_agent_id.dim() == 1 && slot_to_agent_id.numel() > 0, "slot_to_agent_id must be nonempty rank 1");
    const int64_t slot_count = slot_to_agent_id.numel();
    const auto device = slot_to_agent_id.device();

    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"alive_mask", alive_mask},
             {"parent_slots", parent_slots},
             {"mate_slots", mate_slots},
             {"child_slots", child_slots},
             {"planned_count", planned_count},
             {"mlp_pack", mlp_pack},
             {"hmem", hmem},
             {"policy_use_gru", policy_use_gru},
             {"rng_manager_mutation_torch_cpu_state", rng_manager_mutation_torch_cpu_state},
             {"policy_birth_delta_tape", policy_birth_delta_tape},
             {"policy_birth_mutation_mask_tape", policy_birth_mutation_mask_tape},
             {"policy_birth_crossover_mask_packed_tape", policy_birth_crossover_mask_packed_tape},
             {"policy_birth_post_rng_state_packed_tape", policy_birth_post_rng_state_packed_tape},
             {"policy_birth_semantic_draws_tape", policy_birth_semantic_draws_tape},
             {"policy_birth_tape_cursor", policy_birth_tape_cursor},
             {"mutation_torch_cpu_semantic_draws", mutation_torch_cpu_semantic_draws},
         }) {
        check_cuda_contiguous(named.second, named.first);
        TORCH_CHECK(named.second.device() == device, named.first, " must share the child-policy CUDA device");
    }
    TORCH_CHECK(alive_mask.scalar_type() == torch::kBool && alive_mask.numel() == slot_count, "alive_mask must be bool[capacity]");
    TORCH_CHECK(policy_use_gru.scalar_type() == torch::kBool && policy_use_gru.numel() == slot_count, "policy_use_gru must be bool[capacity]");
    TORCH_CHECK(parent_slots.scalar_type() == torch::kInt64 && mate_slots.scalar_type() == torch::kInt64 && child_slots.scalar_type() == torch::kInt64, "turnover slots must be int64");
    TORCH_CHECK(parent_slots.dim() == 1 && mate_slots.sizes() == parent_slots.sizes() && child_slots.sizes() == parent_slots.sizes(), "turnover slots must be aligned rank 1");
    TORCH_CHECK(planned_count.scalar_type() == torch::kInt64 && planned_count.dim() == 1 && planned_count.numel() == 1, "planned_count must be int64[1]");
    TORCH_CHECK(mlp_pack.scalar_type() == torch::kFloat32 && mlp_pack.dim() == 2 && mlp_pack.size(0) == slot_count, "mlp_pack must be float32[capacity,width]");
    TORCH_CHECK(hmem.scalar_type() == torch::kFloat32 && hmem.dim() == 2 && hmem.size(0) == slot_count, "hmem must be float32[capacity,width]");
    TORCH_CHECK(rng_manager_mutation_torch_cpu_state.scalar_type() == torch::kUInt8 && rng_manager_mutation_torch_cpu_state.dim() == 1 && rng_manager_mutation_torch_cpu_state.numel() == torch_rng_pack::kFullStateBytes, "mutation Torch state must be canonical 5056-byte rank-1 uint8");
    TORCH_CHECK(policy_birth_delta_tape.scalar_type() == torch::kFloat32 && policy_birth_delta_tape.dim() == 2, "policy delta tape must be rank-2 float32");
    TORCH_CHECK(policy_birth_mutation_mask_tape.scalar_type() == torch::kBool && policy_birth_mutation_mask_tape.sizes() == policy_birth_delta_tape.sizes(), "policy mutation mask must align with delta tape");
    const int64_t packed_crossover_row_bytes = (mlp_pack.size(1) + 7) / 8;
    TORCH_CHECK(policy_birth_crossover_mask_packed_tape.scalar_type() == torch::kUInt8 && policy_birth_crossover_mask_packed_tape.dim() == 2, "packed policy crossover mask must be rank-2 uint8");
    TORCH_CHECK(policy_birth_crossover_mask_packed_tape.size(0) == policy_birth_delta_tape.size(0), "packed policy crossover mask capacity must align with delta tape");
    TORCH_CHECK(policy_birth_crossover_mask_packed_tape.size(1) == packed_crossover_row_bytes, "packed policy crossover mask row bytes must match mlp_pack width");
    TORCH_CHECK(policy_birth_delta_tape.size(1) == 1 || policy_birth_delta_tape.size(1) == mlp_pack.size(1), "policy mutation tape width must be compact width 1 or match mlp_pack");
    TORCH_CHECK(policy_birth_post_rng_state_packed_tape.scalar_type() == torch::kUInt8 && policy_birth_post_rng_state_packed_tape.dim() == 2, "packed post-state tape must be rank-2 uint8");
    const bool replay_terminal_state = policy_birth_post_rng_state_packed_tape.numel() == 0;
    if (replay_terminal_state) {
        TORCH_CHECK(policy_birth_delta_tape.size(1) == 1, "post-state replay requires compact disabled-mutation tape");
    } else {
        TORCH_CHECK(policy_birth_post_rng_state_packed_tape.size(0) == policy_birth_delta_tape.size(0) && policy_birth_post_rng_state_packed_tape.size(1) == torch_rng_pack::kPackedStateBytes, "packed post-state tape must align with policy episodes and use 2560-byte rows");
    }
    TORCH_CHECK(policy_birth_semantic_draws_tape.scalar_type() == torch::kInt64 && policy_birth_semantic_draws_tape.dim() == 1 && policy_birth_semantic_draws_tape.size(0) == policy_birth_delta_tape.size(0), "semantic-draw tape must align with policy episodes");
    TORCH_CHECK(policy_birth_tape_cursor.scalar_type() == torch::kInt64 && policy_birth_tape_cursor.numel() == 1, "policy_birth_tape_cursor must be int64[1]");
    TORCH_CHECK(mutation_torch_cpu_semantic_draws.scalar_type() == torch::kInt64 && mutation_torch_cpu_semantic_draws.numel() == 1, "mutation_torch_cpu_semantic_draws must be int64[1]");

    c10::cuda::CUDAGuard device_guard(device);
    auto int64_options = slot_to_agent_id.options().dtype(torch::kInt64);
    auto int32_options = slot_to_agent_id.options().dtype(torch::kInt32);
    auto float64_options = slot_to_agent_id.options().dtype(torch::kFloat64);
    auto committed_count = torch::zeros({1}, int64_options);
    auto semantic_draws_added = torch::zeros({1}, int64_options);
    auto mutation_events = torch::zeros({1}, int64_options);
    auto mutated_elements = torch::zeros({1}, int64_options);
    auto mutation_delta_total = torch::zeros({1}, float64_options);
    auto mutation_occurred_by_birth = torch::zeros(
        {parent_slots.numel()},
        slot_to_agent_id.options().dtype(torch::kBool)
    );
    auto mutation_delta_by_birth = torch::zeros(
        {parent_slots.numel()},
        slot_to_agent_id.options().dtype(torch::kFloat32)
    );
    auto status = torch::zeros({1}, int32_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fixed_mlp_child_policy_kernel<<<1, 1, 0, stream>>>(
        slot_to_agent_id.data_ptr<int64_t>(),
        alive_mask.data_ptr<bool>(),
        parent_slots.data_ptr<int64_t>(),
        mate_slots.data_ptr<int64_t>(),
        child_slots.data_ptr<int64_t>(),
        mlp_pack.data_ptr<float>(),
        hmem.data_ptr<float>(),
        policy_use_gru.data_ptr<bool>(),
        rng_manager_mutation_torch_cpu_state.data_ptr<uint8_t>(),
        policy_birth_delta_tape.data_ptr<float>(),
        policy_birth_mutation_mask_tape.data_ptr<bool>(),
        policy_birth_crossover_mask_packed_tape.data_ptr<uint8_t>(),
        replay_terminal_state
            ? nullptr
            : policy_birth_post_rng_state_packed_tape.data_ptr<uint8_t>(),
        policy_birth_semantic_draws_tape.data_ptr<int64_t>(),
        policy_birth_tape_cursor.data_ptr<int64_t>(),
        mutation_torch_cpu_semantic_draws.data_ptr<int64_t>(),
        committed_count.data_ptr<int64_t>(),
        semantic_draws_added.data_ptr<int64_t>(),
        mutation_events.data_ptr<int64_t>(),
        mutated_elements.data_ptr<int64_t>(),
        mutation_delta_total.data_ptr<double>(),
        mutation_occurred_by_birth.data_ptr<bool>(),
        mutation_delta_by_birth.data_ptr<float>(),
        status.data_ptr<int32_t>(),
        planned_count.data_ptr<int64_t>(),
        parent_slots.numel(),
        slot_count,
        mlp_pack.size(1),
        hmem.size(1),
        rng_manager_mutation_torch_cpu_state.numel(),
        policy_birth_delta_tape.size(0),
        policy_birth_delta_tape.size(1),
        true
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        committed_count,
        semantic_draws_added,
        mutation_events,
        mutated_elements,
        mutation_delta_total,
        mutation_occurred_by_birth,
        mutation_delta_by_birth,
        status,
    };
}
