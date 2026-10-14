#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include "cuda_architecture_b_flattened_birth_identity_v2.cuh"

#include <cstdint>
#include <vector>

namespace {

namespace identity =
    eiel::cuda_architecture_b::flattened_birth_identity_v2;
namespace persistent_identity =
    eiel::cuda_architecture_b::persistent_birth_identity;

template <typename T>
void check_tensor(
    const torch::Tensor& tensor,
    const torch::Device& device,
    const char* name,
    torch::ScalarType dtype) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == device, name, " must share the CUDA device");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
}

__global__ void identity_preflight_kernel(
    const int64_t* slot_to_agent_id,
    const bool* alive_mask,
    const int32_t* agent_grid,
    const float* energy,
    const float* health,
    const int32_t* repro_cooldown,
    const float* repro_threshold,
    const double* policy_energy_observation,
    const int64_t* slot_free_queue,
    const int64_t* slot_free_queue_count,
    const int64_t* population_next_id,
    const int64_t* parent_slots,
    const int32_t* spawn_positions,
    const int64_t* planned_count,
    int64_t* staged_parent_slots,
    int64_t* staged_child_slots,
    int64_t* staged_child_agent_ids,
    double* staged_child_energy,
    int64_t* staged_count,
    int64_t* queue_head,
    int64_t* queue_count,
    int32_t* status,
    const int64_t batch,
    const int64_t capacity,
    const int64_t grid_height,
    const int64_t grid_width,
    const int64_t agent_id_capacity,
    const double parent_energy_share,
    const double minimum_energy_buffer,
    const float start_health,
    const int64_t cooldown_steps,
    const int64_t generation,
    const int64_t tick) {
    if (threadIdx.x != 0) {
        return;
    }
    const int64_t world = static_cast<int64_t>(blockIdx.x);
    if (world >= batch) {
        return;
    }
    const int64_t slot_offset = world * capacity;
    const int64_t grid_offset = world * grid_height * grid_width;

    // The source preflight writes scratch only after its cheap global checks.
    // Initialize the complete output row so every failure status has a stable
    // and auditable sentinel payload.
    for (int64_t birth = 0; birth < capacity; ++birth) {
        staged_parent_slots[slot_offset + birth] = -1;
        staged_child_slots[slot_offset + birth] = -1;
        staged_child_agent_ids[slot_offset + birth] = -1;
        staged_child_energy[slot_offset + birth] = 0.0;
    }
    staged_count[world] = 0;
    queue_head[world] = 0;
    queue_count[world] = 0;
    status[world] = persistent_identity::kStatusOk;

    const identity::FlatOrdinaryBirthIdentityPreflightState state{
        slot_to_agent_id + slot_offset,
        alive_mask + slot_offset,
        agent_grid + grid_offset,
        energy + slot_offset,
        health + slot_offset,
        repro_cooldown + slot_offset,
        repro_threshold + slot_offset,
        policy_energy_observation + slot_offset,
        slot_free_queue + slot_offset,
        slot_free_queue_count + world,
        population_next_id + world,
    };
    const persistent_identity::FlatOrdinaryBirthPlan plan{
        parent_slots + slot_offset,
        spawn_positions + slot_offset * 2,
        planned_count + world,
    };
    const identity::FlatOrdinaryBirthIdentityPreflightScratch scratch{
        staged_parent_slots + slot_offset,
        staged_child_slots + slot_offset,
        staged_child_agent_ids + slot_offset,
        staged_child_energy + slot_offset,
        staged_count + world,
        queue_head + world,
        queue_count + world,
    };
    const persistent_identity::OrdinaryBirthIdentityConfig config{
        capacity,
        capacity,
        agent_id_capacity,
        grid_height,
        grid_width,
        capacity,
        parent_energy_share,
        minimum_energy_buffer,
        start_health,
        static_cast<int32_t>(cooldown_steps),
        static_cast<int32_t>(generation),
        static_cast<int32_t>(tick),
        persistent_identity::parent_order::BirthParentOrderModeV1::
            ascending_agent_id_v1,
    };
    status[world] = static_cast<int32_t>(
        identity::preflight_ordinary_birth_identity_external_scratch_v2(
            state, plan, scratch, config));
}

void check_inputs(
    const std::vector<torch::Tensor>& tensors,
    const int64_t batch,
    const int64_t capacity,
    const int64_t grid_height,
    const int64_t grid_width) {
    TORCH_CHECK(!tensors.empty(), "C4 identity preflight requires tensors");
    const auto device = tensors.front().device();
    check_tensor<int64_t>(tensors[0], device, "slot_to_agent_id", torch::kInt64);
    check_tensor<bool>(tensors[1], device, "alive_mask", torch::kBool);
    check_tensor<int32_t>(tensors[2], device, "agent_grid", torch::kInt32);
    check_tensor<float>(tensors[3], device, "energy", torch::kFloat32);
    check_tensor<float>(tensors[4], device, "health", torch::kFloat32);
    check_tensor<int32_t>(tensors[5], device, "repro_cooldown", torch::kInt32);
    check_tensor<float>(tensors[6], device, "repro_threshold", torch::kFloat32);
    check_tensor<double>(
        tensors[7], device, "policy_energy_observation", torch::kFloat64);
    check_tensor<int64_t>(tensors[8], device, "slot_free_queue", torch::kInt64);
    check_tensor<int64_t>(
        tensors[9], device, "slot_free_queue_count", torch::kInt64);
    check_tensor<int64_t>(
        tensors[10], device, "population_next_id", torch::kInt64);
    check_tensor<int64_t>(tensors[11], device, "parent_slots", torch::kInt64);
    check_tensor<int32_t>(
        tensors[12], device, "spawn_positions", torch::kInt32);
    check_tensor<int64_t>(tensors[13], device, "planned_count", torch::kInt64);
    TORCH_CHECK(
        tensors[0].sizes() == torch::IntArrayRef({batch, capacity}),
        "slot_to_agent_id shape is invalid");
    TORCH_CHECK(
        tensors[1].sizes() == torch::IntArrayRef({batch, capacity}),
        "alive_mask shape is invalid");
    TORCH_CHECK(
        tensors[2].sizes() == torch::IntArrayRef({batch, grid_height, grid_width}),
        "agent_grid shape is invalid");
    for (const auto index : {3, 4, 5, 6, 7, 8, 11}) {
        TORCH_CHECK(
            tensors[index].sizes() == torch::IntArrayRef({batch, capacity}),
            "C4 identity slot tensor has an invalid shape");
    }
    TORCH_CHECK(
        tensors[9].sizes() == torch::IntArrayRef({batch}),
        "slot_free_queue_count shape is invalid");
    TORCH_CHECK(
        tensors[10].sizes() == torch::IntArrayRef({batch}),
        "population_next_id shape is invalid");
    TORCH_CHECK(
        tensors[12].sizes() == torch::IntArrayRef({batch, capacity, 2}),
        "spawn_positions shape is invalid");
    TORCH_CHECK(
        tensors[13].sizes() == torch::IntArrayRef({batch}),
        "planned_count shape is invalid");
}

}  // namespace

std::vector<torch::Tensor> tensor_world_c4_identity_preflight_cuda(
    torch::Tensor slot_to_agent_id,
    torch::Tensor alive_mask,
    torch::Tensor agent_grid,
    torch::Tensor energy,
    torch::Tensor health,
    torch::Tensor repro_cooldown,
    torch::Tensor repro_threshold,
    torch::Tensor policy_energy_observation,
    torch::Tensor slot_free_queue,
    torch::Tensor slot_free_queue_count,
    torch::Tensor population_next_id,
    torch::Tensor parent_slots,
    torch::Tensor spawn_positions,
    torch::Tensor planned_count,
    double parent_energy_share,
    double minimum_energy_buffer,
    float start_health,
    int64_t cooldown_steps,
    int64_t generation,
    int64_t tick) {
    const std::vector<torch::Tensor> inputs{
        slot_to_agent_id,
        alive_mask,
        agent_grid,
        energy,
        health,
        repro_cooldown,
        repro_threshold,
        policy_energy_observation,
        slot_free_queue,
        slot_free_queue_count,
        population_next_id,
        parent_slots,
        spawn_positions,
        planned_count,
    };
    TORCH_CHECK(slot_to_agent_id.dim() == 2, "C4 identity state must be rank two");
    const int64_t batch = slot_to_agent_id.size(0);
    const int64_t capacity = slot_to_agent_id.size(1);
    TORCH_CHECK(batch > 0 && capacity > 0, "C4 identity dimensions must be positive");
    TORCH_CHECK(agent_grid.dim() == 3, "agent_grid must be rank three");
    const int64_t grid_height = agent_grid.size(1);
    const int64_t grid_width = agent_grid.size(2);
    check_inputs(inputs, batch, capacity, grid_height, grid_width);
    TORCH_CHECK(
        parent_energy_share >= 0.0 && parent_energy_share <= 1.0,
        "parent_energy_share must be in [0, 1]");
    TORCH_CHECK(minimum_energy_buffer >= 0.0, "minimum_energy_buffer must be nonnegative");
    TORCH_CHECK(start_health >= 0.0f, "start_health must be nonnegative");
    TORCH_CHECK(cooldown_steps >= 0 && generation >= 0 && tick >= 0,
                "C4 identity scalar configuration is invalid");

    auto options_i64 = slot_to_agent_id.options().dtype(torch::kInt64);
    auto options_f64 = slot_to_agent_id.options().dtype(torch::kFloat64);
    auto options_i32 = slot_to_agent_id.options().dtype(torch::kInt32);
    auto staged_parent_slots = torch::full({batch, capacity}, -1, options_i64);
    auto staged_child_slots = torch::full({batch, capacity}, -1, options_i64);
    auto staged_child_agent_ids = torch::full({batch, capacity}, -1, options_i64);
    auto staged_child_energy = torch::zeros({batch, capacity}, options_f64);
    auto staged_count = torch::zeros({batch}, options_i64);
    auto queue_head = torch::zeros({batch}, options_i64);
    auto queue_count = torch::zeros({batch}, options_i64);
    auto status = torch::zeros({batch}, options_i32);

    identity_preflight_kernel<<<
        static_cast<unsigned int>(batch),
        1,
        0,
        at::cuda::getDefaultCUDAStream(slot_to_agent_id.device().index())>>>(
        slot_to_agent_id.data_ptr<int64_t>(),
        alive_mask.data_ptr<bool>(),
        agent_grid.data_ptr<int32_t>(),
        energy.data_ptr<float>(),
        health.data_ptr<float>(),
        repro_cooldown.data_ptr<int32_t>(),
        repro_threshold.data_ptr<float>(),
        policy_energy_observation.data_ptr<double>(),
        slot_free_queue.data_ptr<int64_t>(),
        slot_free_queue_count.data_ptr<int64_t>(),
        population_next_id.data_ptr<int64_t>(),
        parent_slots.data_ptr<int64_t>(),
        spawn_positions.data_ptr<int32_t>(),
        planned_count.data_ptr<int64_t>(),
        staged_parent_slots.data_ptr<int64_t>(),
        staged_child_slots.data_ptr<int64_t>(),
        staged_child_agent_ids.data_ptr<int64_t>(),
        staged_child_energy.data_ptr<double>(),
        staged_count.data_ptr<int64_t>(),
        queue_head.data_ptr<int64_t>(),
        queue_count.data_ptr<int64_t>(),
        status.data_ptr<int32_t>(),
        batch,
        capacity,
        grid_height,
        grid_width,
        capacity,
        parent_energy_share,
        minimum_energy_buffer,
        start_health,
        cooldown_steps,
        generation,
        tick);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        staged_parent_slots,
        staged_child_slots,
        staged_child_agent_ids,
        staged_child_energy,
        staged_count,
        queue_head,
        queue_count,
        status,
    };
}
