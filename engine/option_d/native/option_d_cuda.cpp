#include <torch/extension.h>

#include <cstdint>
#include <vector>


std::vector<torch::Tensor> option_d_execute_cuda(
    const torch::Tensor& words,
    const torch::Tensor& indices,
    const torch::Tensor& has_gauss,
    const torch::Tensor& gauss_next,
    const torch::Tensor& counters,
    const torch::Tensor& opcodes,
    const torch::Tensor& arguments,
    const torch::Tensor& mus,
    const torch::Tensor& sigmas,
    const torch::Tensor& predicate_sources,
    const torch::Tensor& predicate_kinds,
    const torch::Tensor& predicate_thresholds,
    const torch::Tensor& operation_counts,
    int64_t gaussian_strategy
);


std::vector<torch::Tensor> option_d_execute_tape_baseline_cuda(
    const torch::Tensor& words,
    const torch::Tensor& indices,
    const torch::Tensor& has_gauss,
    const torch::Tensor& gauss_next,
    const torch::Tensor& counters,
    const torch::Tensor& opcodes,
    const torch::Tensor& arguments,
    const torch::Tensor& mus,
    const torch::Tensor& sigmas,
    const torch::Tensor& predicate_sources,
    const torch::Tensor& predicate_kinds,
    const torch::Tensor& predicate_thresholds,
    const torch::Tensor& operation_counts,
    const torch::Tensor& tape_primary,
    const torch::Tensor& tape_secondary,
    const torch::Tensor& tape_base_raw_words
);


torch::Tensor option_d_math_probe_cuda(
    const torch::Tensor& inputs,
    int64_t gaussian_strategy
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "execute",
        &option_d_execute_cuda,
        "Standalone Option D transactional CUDA schedule execution"
    );
    module.def(
        "execute_tape_baseline",
        &option_d_execute_tape_baseline_cuda,
        "Current C5 host-tape baseline schedule execution"
    );
    module.def(
        "math_probe",
        &option_d_math_probe_cuda,
        "Option D bounded CUDA sin/cos/log exactness probe"
    );
}
