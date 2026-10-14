#include <torch/extension.h>

void tensor_world_c1_observe_cuda(
    torch::Tensor observations, torch::Tensor alive, torch::Tensor pos_y,
    torch::Tensor pos_x, torch::Tensor agent_id_grid, torch::Tensor energy,
    torch::Tensor health, torch::Tensor food_amount, torch::Tensor cover,
    torch::Tensor shelter, torch::Tensor scent, torch::Tensor waste,
    torch::Tensor movement_cost, double max_energy, double start_health,
    double food_max, double food_threshold, double scent_max, double waste_max,
    int64_t density_radius, bool density_include_self,
    bool density_normalize, double movement_min, double movement_max);

void tensor_world_c1_policy_logits_cuda(
    torch::Tensor logits, torch::Tensor packed_weights,
    torch::Tensor observations, torch::Tensor drives,
    torch::Tensor attack_bias, torch::Tensor action_temperature,
    torch::Tensor drive_logit_gain, double w_hunger_gather,
    double w_pain_rest, double w_fatigue_rest,
    double w_pain_attack, double w_fear_attack);

void tensor_world_c1_select_cuda(
    torch::Tensor actions, torch::Tensor logits, torch::Tensor uniforms,
    torch::Tensor alive, torch::Tensor action_cursor);

void tensor_world_c1_settle_cuda(
    torch::Tensor actions, torch::Tensor alive, torch::Tensor agent_ids,
    torch::Tensor pos_y, torch::Tensor pos_x, torch::Tensor agent_id_grid,
    torch::Tensor slot_grid, torch::Tensor food_amount, torch::Tensor movement_cost,
    torch::Tensor energy, torch::Tensor energy_collected, torch::Tensor moved,
    torch::Tensor gathered, torch::Tensor food_gained, torch::Tensor moves_attempted,
    torch::Tensor moves_committed, torch::Tensor gathers_committed,
    double move_cost, double gather_cost, double rest_cost, double attack_cost,
    double action_cost_mult, double rest_cost_mult, double max_energy,
    double food_energy, double gather_amount, double food_threshold,
    bool movement_cost_apply, double movement_cost_scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Tensor World C1 persistent vertical-slice primitives";
    module.def("observe", &tensor_world_c1_observe_cuda);
    module.def("policy_logits", &tensor_world_c1_policy_logits_cuda);
    module.def("select_actions", &tensor_world_c1_select_cuda);
    module.def("settle", &tensor_world_c1_settle_cuda);
}
