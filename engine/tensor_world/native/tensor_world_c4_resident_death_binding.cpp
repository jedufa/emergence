#include <torch/extension.h>

#include <cstdint>
#include <vector>

void tensor_world_c4_resident_death_transaction_cuda(
    std::vector<torch::Tensor> authoritative,
    std::vector<torch::Tensor> staging,
    int64_t generation,
    int64_t tick);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Tensor World C4 resident death-only transaction";
    module.def(
        "death_transaction",
        &tensor_world_c4_resident_death_transaction_cuda,
        "Preflight and atomically commit the batched resident death transaction");
}
