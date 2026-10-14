#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> eiel_cuda_scripted_action_generation_v1(
    torch::Tensor agent_ids,
    torch::Tensor actions_by_tick_agent,
    torch::Tensor agent_pos_y,
    torch::Tensor agent_pos_x,
    torch::Tensor agent_grid,
    torch::Tensor food_grid,
    torch::Tensor patch_food_amount,
    torch::Tensor patch_scent_emit_pending,
    torch::Tensor patch_waste_emit_pending,
    torch::Tensor patch_movement_cost,
    torch::Tensor energies,
    torch::Tensor healths,
    bool patch_enabled,
    double move_cost,
    double gather_cost,
    double rest_cost,
    double attack_cost,
    double action_cost_mult,
    double rest_cost_mult,
    double max_energy,
    double food_energy,
    double patch_food_gather_amount,
    double patch_food_presence_threshold,
    double patch_scent_move_emit,
    double patch_waste_move_emit,
    double patch_scent_gather_emit,
    double patch_waste_consume_emit,
    bool patch_movement_cost_apply_enable,
    double patch_movement_cost_energy_scale
);

std::vector<torch::Tensor> eiel_cuda_scripted_owner_action_generation_v1(
    torch::Tensor agent_ids_by_alive_order,
    torch::Tensor agent_id_to_slot,
    torch::Tensor actions_by_tick_slot,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor food_grid,
    torch::Tensor patch_food_amount,
    torch::Tensor patch_scent_emit_pending,
    torch::Tensor patch_waste_emit_pending,
    torch::Tensor patch_movement_cost,
    torch::Tensor energy,
    torch::Tensor health,
    bool patch_enabled,
    double move_cost,
    double gather_cost,
    double rest_cost,
    double attack_cost,
    double action_cost_mult,
    double rest_cost_mult,
    double max_energy,
    double food_energy,
    double patch_food_gather_amount,
    double patch_food_presence_threshold,
    double patch_scent_move_emit,
    double patch_waste_move_emit,
    double patch_scent_gather_emit,
    double patch_waste_consume_emit,
    bool patch_movement_cost_apply_enable,
    double patch_movement_cost_energy_scale
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Prebuilt CUDA Architecture B scripted-generation primitives";
    module.def(
        "eiel_cuda_scripted_action_generation_v1",
        &eiel_cuda_scripted_action_generation_v1
    );
    module.def(
        "eiel_cuda_scripted_owner_action_generation_v1",
        &eiel_cuda_scripted_owner_action_generation_v1
    );
}
