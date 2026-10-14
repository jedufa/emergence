#include "eed_resource_impact_native_v1.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace eiel::eed_resource_impact_native_v1 {
namespace {
thread_local ResidentContextV1 active_resident_context{};
}

void install_resident_context_v1(const ResidentContextV1& context) {
    active_resident_context = context;
}

void clear_resident_context_v1() noexcept {
    active_resident_context = {};
}

const ResidentContextV1& current_resident_context_v1() noexcept {
    return active_resident_context;
}

namespace {

__device__ __forceinline__ float bounded(
    const float value,
    const float low,
    const float high
) {
    return fminf(high, fmaxf(low, value));
}

__device__ __forceinline__ float fixed_axis(
    const Config config,
    const std::int64_t index
) {
    if (config.fixed_axis_mode == 0) {
        return 0.0F;
    }
    if (config.fixed_axis_mode == 1) {
        return 1.0F;
    }
    if (config.fixed_axis_mode == 2) {
        return -1.0F;
    }
    return index % 2 == 0 ? 1.0F : -1.0F;
}

__global__ void transition_kernel_v1(
    float* const a_amount,
    float* const b_amount,
    const float* const a_capacity,
    const float* const b_capacity,
    const float* const a_renewal,
    const float* const b_renewal,
    float* const a_pending,
    float* const b_pending,
    float* const a_load,
    float* const b_load,
    const std::int32_t* const resource_classes,
    const std::int64_t* const flat_indices,
    const float* const handling_axes,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const Config config
) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= state_count) {
        return;
    }

    if (config.ecology_frozen) {
        return;
    }

    if (config.impact_applied) {
        for (std::int64_t event = 0; event < event_count; ++event) {
            if (flat_indices[event] != index) {
                continue;
            }
            const auto resource_class = resource_classes[event];
            float axis = handling_axes[event];
            if (config.phenotype_fixed) {
                axis = fixed_axis(config, index);
            }
            const float signed_axis = resource_class == kResourceA
                ? axis
                : -axis;
            const float impact =
                config.impact_per_success * fmaxf(0.0F, signed_axis);
            if (resource_class == kResourceA) {
                a_pending[index] += impact;
            } else if (resource_class == kResourceB) {
                b_pending[index] += impact;
            }
        }
    }

    const float retained = 1.0F - config.decay_per_step;
    a_load[index] = config.impact_applied
        ? bounded(
              a_load[index] * retained + a_pending[index],
              0.0F,
              config.maximum_load
          )
        : 0.0F;
    b_load[index] = config.impact_applied
        ? bounded(
              b_load[index] * retained + b_pending[index],
              0.0F,
              config.maximum_load
          )
        : 0.0F;
    a_pending[index] = 0.0F;
    b_pending[index] = 0.0F;
    const float a_effective =
        a_renewal[index] * (1.0F - bounded(a_load[index], 0.0F, 1.0F));
    const float b_effective =
        b_renewal[index] * (1.0F - bounded(b_load[index], 0.0F, 1.0F));
    a_amount[index] = fminf(
        a_capacity[index],
        fmaxf(0.0F, a_amount[index] + a_effective)
    );
    b_amount[index] = fminf(
        b_capacity[index],
        fmaxf(0.0F, b_amount[index] + b_effective)
    );
}

}  // namespace

void transition_cuda_v1(
    const std::vector<torch::Tensor>& state,
    const torch::Tensor& resource_classes,
    const torch::Tensor& flat_indices,
    const torch::Tensor& handling_axes,
    const Config config
) {
    c10::cuda::CUDAGuard guard(state.front().device());
    const auto stream =
        at::cuda::getCurrentCUDAStream(state.front().device().index());
    constexpr std::int64_t threads = 256;
    const auto count = state.front().numel();
    const auto blocks = static_cast<unsigned int>(
        (count + threads - 1) / threads
    );
    transition_kernel_v1<<<blocks, threads, 0, stream>>>(
        state.at(0).data_ptr<float>(),
        state.at(1).data_ptr<float>(),
        state.at(2).data_ptr<float>(),
        state.at(3).data_ptr<float>(),
        state.at(4).data_ptr<float>(),
        state.at(5).data_ptr<float>(),
        state.at(6).data_ptr<float>(),
        state.at(7).data_ptr<float>(),
        state.at(8).data_ptr<float>(),
        state.at(9).data_ptr<float>(),
        resource_classes.data_ptr<std::int32_t>(),
        flat_indices.data_ptr<std::int64_t>(),
        handling_axes.data_ptr<float>(),
        resource_classes.numel(),
        count,
        config
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace eiel::eed_resource_impact_native_v1
