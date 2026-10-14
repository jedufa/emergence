#include "cuda_architecture_b_stackless_birth_microproof.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace sb = eiel::cuda_architecture_b::stackless_birth_microproof;

namespace {

template <typename T>
T* ptr(const torch::Tensor& tensor) {
    return tensor.data_ptr<T>();
}

void require_cuda_contiguous(
    const torch::Tensor& reference,
    const torch::Tensor& tensor,
    const char* name
) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.device() == reference.device(), name, " device mismatch");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(!tensor.requires_grad(), name, " must not require gradients");
}

}  // namespace

void eiel_cuda_architecture_b_stackless_birth_microproof_v1(
    std::vector<torch::Tensor> scalar_state,
    std::vector<torch::Tensor> scalar_rng,
    std::vector<torch::Tensor> plan,
    std::vector<torch::Tensor> scalar_outputs,
    torch::Tensor trial_python_words,
    std::vector<torch::Tensor> drive_state,
    std::vector<torch::Tensor> drive_outputs,
    std::vector<torch::Tensor> policy_state,
    std::vector<torch::Tensor> policy_tape,
    std::vector<torch::Tensor> policy_outputs,
    torch::Tensor payload_i64,
    torch::Tensor payload_i32,
    torch::Tensor payload_f32,
    torch::Tensor status,
    const int64_t sensor_radius_max,
    const double max_energy,
    const int64_t parent_agent_id,
    const int64_t child_agent_id,
    const int64_t source_parent_slot,
    const int64_t source_child_slot,
    const int64_t spawn_y,
    const int64_t spawn_x,
    const int64_t parent_founder_id,
    const int64_t parent_lineage_depth,
    const int64_t generation,
    const int64_t tick,
    const double parent_energy_before,
    const double child_energy,
    const double parent_energy_after,
    const double start_health,
    const int64_t cooldown_steps
) {
    TORCH_CHECK(scalar_state.size() == 36, "scalar_state must contain 36 tensors");
    TORCH_CHECK(scalar_rng.size() == 8, "scalar_rng must contain 8 tensors");
    TORCH_CHECK(plan.size() == 3, "plan must contain 3 tensors");
    TORCH_CHECK(scalar_outputs.size() == 3, "scalar_outputs must contain 3 tensors");
    TORCH_CHECK(drive_state.size() == 16, "drive_state must contain 16 tensors");
    TORCH_CHECK(drive_outputs.size() == 7, "drive_outputs must contain 7 tensors");
    TORCH_CHECK(policy_state.size() == 7, "policy_state must contain 7 tensors");
    TORCH_CHECK(policy_tape.size() == 5, "policy_tape must contain 5 tensors");
    TORCH_CHECK(policy_outputs.size() == 8, "policy_outputs must contain 8 tensors");

    for (const auto* group : {&scalar_state, &scalar_rng, &plan, &scalar_outputs,
                              &drive_state, &drive_outputs, &policy_state,
                              &policy_tape, &policy_outputs}) {
        for (const auto& tensor : *group) {
            require_cuda_contiguous(status, tensor, "microproof tensor");
        }
    }
    for (const auto& tensor : {trial_python_words, payload_i64, payload_i32,
                               payload_f32, status}) {
        require_cuda_contiguous(status, tensor, "microproof output");
    }
    TORCH_CHECK(trial_python_words.scalar_type() == torch::kUInt32
                    && trial_python_words.numel() == 624,
                "trial_python_words must be uint32[624]");
    TORCH_CHECK(payload_i64.scalar_type() == torch::kInt64
                    && payload_i64.numel() == 16,
                "payload_i64 must be int64[16]");
    TORCH_CHECK(payload_i32.scalar_type() == torch::kInt32
                    && payload_i32.numel() == 24,
                "payload_i32 must be int32[24]");
    TORCH_CHECK(payload_f32.scalar_type() == torch::kFloat32
                    && payload_f32.numel() == 16,
                "payload_f32 must be float32[16]");
    TORCH_CHECK(status.scalar_type() == torch::kInt32 && status.numel() == 1,
                "status must be int32[1]");

    c10::cuda::CUDAGuard guard(status.device());
    const auto stream = at::cuda::getCurrentCUDAStream(status.device().index());
    sb::FlatStacklessBirthArguments arguments{};
    for (size_t index = 0; index < scalar_state.size(); ++index) {
        arguments.scalar_state[index] = scalar_state[index].data_ptr();
    }
    for (size_t index = 0; index < scalar_rng.size(); ++index) {
        arguments.scalar_rng[index] = scalar_rng[index].data_ptr();
    }
    for (size_t index = 0; index < plan.size(); ++index) {
        arguments.plan[index] = plan[index].data_ptr();
    }
    for (size_t index = 0; index < scalar_outputs.size(); ++index) {
        arguments.scalar_outputs[index] = scalar_outputs[index].data_ptr();
    }
    arguments.trial_python_words = ptr<uint32_t>(trial_python_words);
    for (size_t index = 0; index < drive_state.size(); ++index) {
        arguments.drive_state[index] = drive_state[index].data_ptr();
    }
    for (size_t index = 0; index < drive_outputs.size(); ++index) {
        arguments.drive_outputs[index] = drive_outputs[index].data_ptr();
    }
    for (size_t index = 0; index < policy_state.size(); ++index) {
        arguments.policy_state[index] = policy_state[index].data_ptr();
    }
    for (size_t index = 0; index < policy_tape.size(); ++index) {
        arguments.policy_tape[index] = policy_tape[index].data_ptr();
    }
    for (size_t index = 0; index < policy_outputs.size(); ++index) {
        arguments.policy_outputs[index] = policy_outputs[index].data_ptr();
    }
    arguments.gauss_tape_capacity = scalar_rng[4].numel();
    arguments.sensor_radius_max = static_cast<int32_t>(sensor_radius_max);
    arguments.drive_width = drive_state[13].size(1);
    arguments.chemistry_width = drive_state[15].size(1);
    arguments.max_energy = static_cast<float>(max_energy);
    arguments.packed_width = policy_state[2].size(1);
    arguments.hmem_width = policy_state[3].size(1);
    arguments.torch_state_bytes = policy_state[5].numel();
    arguments.policy_tape_capacity = policy_tape[0].size(0);
    sb::launch_stackless_birth_microproof_v1(
        arguments,
        {ptr<int64_t>(payload_i64), ptr<int32_t>(payload_i32),
         ptr<float>(payload_f32), ptr<int32_t>(status)},
        {parent_agent_id, child_agent_id, source_parent_slot, source_child_slot,
         static_cast<int32_t>(spawn_y), static_cast<int32_t>(spawn_x),
         static_cast<int32_t>(parent_founder_id),
         static_cast<int32_t>(parent_lineage_depth),
         static_cast<int32_t>(generation), static_cast<int32_t>(tick),
         static_cast<float>(parent_energy_before), static_cast<float>(child_energy),
         static_cast<float>(parent_energy_after), static_cast<float>(start_health),
         static_cast<int32_t>(cooldown_steps)},
        stream.stream()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<int64_t>
eiel_cuda_architecture_b_stackless_birth_microproof_resources_v1() {
    const auto receipt = sb::query_stackless_birth_microproof_resources_v1();
    return {receipt.registers, receipt.static_stack_bytes, receipt.local_bytes,
            receipt.shared_bytes, receipt.max_threads_per_block,
            receipt.max_blocks_per_multiprocessor};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("prepare", &eiel_cuda_architecture_b_stackless_birth_microproof_v1);
    module.def("resource_receipt",
               &eiel_cuda_architecture_b_stackless_birth_microproof_resources_v1);
}
