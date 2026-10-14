#include <torch/extension.h>

#include <cstdint>
#include <vector>

void tensor_world_c4_resident_maxload_r8_cuda(
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    std::vector<torch::Tensor>, torch::Tensor,
    int64_t, double, double, int64_t, double, double,
    int64_t, int64_t, int64_t, int64_t, int64_t, double, int64_t, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Tensor World C4 R8 B4/C32 maximum ordinary turnover probe";
    module.def("maxload_transaction", &tensor_world_c4_resident_maxload_r8_cuda);
}
