#include <torch/extension.h>

#include "cuda_architecture_b_modular_tick.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "eiel_cuda_architecture_b_modular_tick_v1",
        &eiel::cuda_architecture_b::modular_tick_v1
    );
}
