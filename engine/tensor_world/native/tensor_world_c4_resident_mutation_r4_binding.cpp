#include <torch/extension.h>

#include <cstdint>
#include <vector>

void tensor_world_c4_resident_mutation_r4_cuda(
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>, int64_t, double, double, int64_t, double, double,
    int64_t, int64_t, int64_t, int64_t, int64_t, double, int64_t, int64_t,
    int64_t, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Tensor World C4 R4 resident mate/scalar mutation transaction";
    module.def("mutation_transaction", &tensor_world_c4_resident_mutation_r4_cuda);
}
