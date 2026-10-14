#include <torch/extension.h>

#include <cstdint>
#include <vector>

std::vector<torch::Tensor> tensor_world_c5_rngv2_mutate_cuda(
    torch::Tensor world,
    torch::Tensor tick,
    torch::Tensor subject,
    torch::Tensor parent_genes,
    torch::Tensor mate_genes,
    torch::Tensor parent_rate,
    torch::Tensor mate_rate,
    uint64_t namespace_key,
    int64_t semantic_epoch
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("mutate", &tensor_world_c5_rngv2_mutate_cuda, "V2 resident mutation kernel");
}
