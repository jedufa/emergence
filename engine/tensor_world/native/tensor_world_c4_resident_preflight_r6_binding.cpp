#include <torch/extension.h>

#include <vector>

void tensor_world_c4_resident_preflight_r6_cuda(
    std::vector<torch::Tensor> primary, std::vector<torch::Tensor> death_tensors,
    std::vector<torch::Tensor> inherit, std::vector<torch::Tensor> zero,
    std::vector<torch::Tensor> minus_one, std::vector<torch::Tensor> half,
    std::vector<torch::Tensor> one, std::vector<torch::Tensor> scalar_rng,
    std::vector<torch::Tensor> replay, std::vector<torch::Tensor> staging,
    std::vector<torch::Tensor> turnover, torch::Tensor status,
    int64_t max_births, double parent_energy_share,
    double minimum_energy_buffer, int64_t cooldown_steps,
    double start_health, double max_energy, int64_t life_history_mode,
    int64_t maturity_start_ticks, int64_t maturity_rise_ticks,
    int64_t fertility_decline_start_ticks, int64_t fertility_decline_ticks,
    double elder_fertility_floor, int64_t generation, int64_t tick,
    int64_t environment_stream, int64_t runner_stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Tensor World C4 R6 all-live preflight over frozen R5";
    module.def(
        "population_transaction",
        &tensor_world_c4_resident_preflight_r6_cuda);
}
