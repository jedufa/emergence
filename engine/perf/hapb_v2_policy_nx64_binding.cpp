#include "hapb_v2_policy_nx64_kernel.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

void check_cuda_f32(const torch::Tensor& value, const char* name, int64_t rank) {
    TORCH_CHECK(
        value.defined() && value.is_cuda() && value.is_contiguous()
            && value.scalar_type() == torch::kFloat32 && value.dim() == rank,
        "HAPB V2 experimental ", name, " must be contiguous CUDA float32"
    );
}

}  // namespace

torch::Tensor launch_experimental_logits_v1(
    const torch::Tensor& packs,
    const torch::Tensor& observations,
    const torch::Tensor& logical_slots,
    const torch::Tensor& drives,
    const torch::Tensor& attack_bias,
    const torch::Tensor& action_temperature,
    const torch::Tensor& drive_logit_gain,
    const double w_hunger_gather,
    const double w_pain_rest,
    const double w_fatigue_rest,
    const double w_pain_attack,
    const double w_fear_attack,
    const bool neuron_parallel
) {
    check_cuda_f32(packs, "packs", 2);
    check_cuda_f32(observations, "observations", 2);
    TORCH_CHECK(
        logical_slots.defined() && logical_slots.is_cuda()
            && logical_slots.is_contiguous()
            && logical_slots.scalar_type() == torch::kInt64
            && logical_slots.dim() == 1,
        "HAPB V2 experimental logical slots must be contiguous CUDA int64"
    );
    check_cuda_f32(drives, "drives", 2);
    check_cuda_f32(attack_bias, "attack_bias", 1);
    check_cuda_f32(action_temperature, "action_temperature", 1);
    check_cuda_f32(drive_logit_gain, "drive_logit_gain", 1);
    const auto rows = packs.size(0);
    TORCH_CHECK(rows > 0 && packs.size(1) == 9991, "HAPB V2 pack shape is invalid");
    TORCH_CHECK(
        observations.size(0) == rows && observations.size(1) == 83
            && logical_slots.size(0) == rows,
        "HAPB V2 observation/slot ranks do not match packs"
    );
    auto logits = torch::empty({rows, 7}, packs.options());
    const auto stream = at::cuda::getCurrentCUDAStream(packs.get_device());
    void* const raw_stream = static_cast<void*>(
        static_cast<cudaStream_t>(stream)
    );
    if (neuron_parallel) {
        eiel::cuda_architecture_b::hapb_v2_experimental::
            hapb_v2_nx64_policy_logits_cuda_v1(
                packs.data_ptr<float>(),
                observations.data_ptr<float>(),
                logical_slots.data_ptr<std::int64_t>(),
                drives.data_ptr<float>(),
                attack_bias.data_ptr<float>(),
                action_temperature.data_ptr<float>(),
                drive_logit_gain.data_ptr<float>(),
                logits.data_ptr<float>(),
                rows,
                drives.size(0),
                drives.size(1),
                static_cast<float>(w_hunger_gather),
                static_cast<float>(w_pain_rest),
                static_cast<float>(w_fatigue_rest),
                static_cast<float>(w_pain_attack),
                static_cast<float>(w_fear_attack),
                raw_stream
            );
    } else {
        eiel::cuda_architecture_b::hapb_v2_experimental::
            hapb_v2_nx1_policy_logits_cuda_v1(
                packs.data_ptr<float>(),
                observations.data_ptr<float>(),
                logical_slots.data_ptr<std::int64_t>(),
                drives.data_ptr<float>(),
                attack_bias.data_ptr<float>(),
                action_temperature.data_ptr<float>(),
                drive_logit_gain.data_ptr<float>(),
                logits.data_ptr<float>(),
                rows,
                drives.size(0),
                drives.size(1),
                static_cast<float>(w_hunger_gather),
                static_cast<float>(w_pain_rest),
                static_cast<float>(w_fatigue_rest),
                static_cast<float>(w_pain_attack),
                static_cast<float>(w_fear_attack),
                raw_stream
            );
    }
    const auto status = cudaGetLastError();
    TORCH_CHECK(
        status == cudaSuccess,
        neuron_parallel
            ? "HAPB V2 N×64 experimental kernel launch failed: "
            : "HAPB V2 N×1 experimental kernel launch failed: ",
        cudaGetErrorString(status)
    );
    return logits;
}

torch::Tensor eiel_hapb_v2_nx1_policy_logits_experimental_v1(
    const torch::Tensor& packs,
    const torch::Tensor& observations,
    const torch::Tensor& logical_slots,
    const torch::Tensor& drives,
    const torch::Tensor& attack_bias,
    const torch::Tensor& action_temperature,
    const torch::Tensor& drive_logit_gain,
    const double w_hunger_gather,
    const double w_pain_rest,
    const double w_fatigue_rest,
    const double w_pain_attack,
    const double w_fear_attack
) {
    return launch_experimental_logits_v1(
        packs, observations, logical_slots, drives, attack_bias,
        action_temperature, drive_logit_gain, w_hunger_gather, w_pain_rest,
        w_fatigue_rest, w_pain_attack, w_fear_attack, false
    );
}

torch::Tensor eiel_hapb_v2_nx64_policy_logits_experimental_v1(
    const torch::Tensor& packs,
    const torch::Tensor& observations,
    const torch::Tensor& logical_slots,
    const torch::Tensor& drives,
    const torch::Tensor& attack_bias,
    const torch::Tensor& action_temperature,
    const torch::Tensor& drive_logit_gain,
    const double w_hunger_gather,
    const double w_pain_rest,
    const double w_fatigue_rest,
    const double w_pain_attack,
    const double w_fear_attack
) {
    return launch_experimental_logits_v1(
        packs, observations, logical_slots, drives, attack_bias,
        action_temperature, drive_logit_gain, w_hunger_gather, w_pain_rest,
        w_fatigue_rest, w_pain_attack, w_fear_attack, true
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "eiel_hapb_v2_nx1_policy_logits_experimental_v1",
        &eiel_hapb_v2_nx1_policy_logits_experimental_v1,
        "Isolated experimental V1-launch N×1 logits for bake-off. Not production."
    );
    module.def(
        "eiel_hapb_v2_nx64_policy_logits_experimental_v1",
        &eiel_hapb_v2_nx64_policy_logits_experimental_v1,
        "Isolated experimental HAPB V2 N×64 exact policy logits. Not production."
    );
}
