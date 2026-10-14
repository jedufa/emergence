#include <torch/extension.h>

#include <cstdint>
#include <vector>

std::vector<torch::Tensor> hapb_rngv2_generate_cuda(
    torch::Tensor world,
    torch::Tensor domain,
    torch::Tensor epoch,
    torch::Tensor subject,
    torch::Tensor tick,
    torch::Tensor operation,
    torch::Tensor n_values,
    uint64_t namespace_key
);

torch::Tensor hapb_rngv2_philox_cuda(
    torch::Tensor counters,
    torch::Tensor keys,
    int64_t rounds
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("generate", &hapb_rngv2_generate_cuda, "HAPB-FRAG-RNGV2 generate kernel");
    module.def("philox", &hapb_rngv2_philox_cuda, "HAPB-FRAG-RNGV2 Philox helper");
}
