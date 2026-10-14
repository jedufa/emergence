#include <torch/extension.h>

#include <cstdint>
#include <vector>

void tensor_world_c4_resident_population_r5_cuda(
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>, int64_t, double, double, int64_t, double, double,
    int64_t, int64_t, int64_t, int64_t, int64_t, double, int64_t, int64_t,
    int64_t, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Tensor World C4 R5 B4 resident population transaction";
    module.def("population_transaction", &tensor_world_c4_resident_population_r5_cuda);
}
