#include <torch/extension.h>

#include <cstdint>
#include <vector>

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
    int64_t tick);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Tensor World C4 resident identity preflight";
    module.def(
        "identity_preflight",
        &tensor_world_c4_identity_preflight_cuda,
        "Run the source-bound C4 identity/slot reservation preflight");
}
