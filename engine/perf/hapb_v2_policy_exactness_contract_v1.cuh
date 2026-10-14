#pragma once

// Frozen V1 per-neuron exactness contract.
// Source of truth: fixed_mlp_policy_logits_from_pack_exact_v1
// in cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu
//
// A later N×64 kernel may assign one hidden neuron to one thread, but the
// accumulation inside that neuron MUST remain sequential left-to-right
// multiply-then-add starting from the bias.  GEMM / tree reductions are
// outside this contract.

#include <cstdint>

namespace eiel::cuda_architecture_b::hapb_v2_exactness_contract_v1 {

constexpr int kObservationSizeV1 = 83;
constexpr int kHiddenSizeV1 = 64;
constexpr int kActionCountV1 = 7;
constexpr int kPolicyWidthV1 = 9991;
constexpr int kNx64ThreadsV1 = 64;
constexpr float kTemperatureFloorV1 = 0.05f;
constexpr float kGainFloorV1 = 0.0f;
constexpr int kActionGatherV1 = 4;
constexpr int kActionRestV1 = 5;
constexpr int kActionAttackV1 = 6;
constexpr int kPackOffsetW1V1 = 0;
constexpr int kPackOffsetB1V1 = kHiddenSizeV1 * kObservationSizeV1;
constexpr int kPackOffsetW2V1 = kPackOffsetB1V1 + kHiddenSizeV1;
constexpr int kPackOffsetB2V1 = kPackOffsetW2V1 + kHiddenSizeV1 * kHiddenSizeV1;
constexpr int kPackOffsetWOutV1 = kPackOffsetB2V1 + kHiddenSizeV1;
constexpr int kPackOffsetBOutV1 = kPackOffsetWOutV1 + kActionCountV1 * kHiddenSizeV1;

static_assert(kPackOffsetBOutV1 + kActionCountV1 == kPolicyWidthV1,
    "HAPB V2 exactness contract pack offsets must reconstruct width 9991");

__device__ __host__ __forceinline__ float sequential_dot_plus_bias_exact_v1(
    const float* const weights,
    const float* const inputs,
    const float bias,
    const int width
) {
    float acc = bias;
    for (int index = 0; index < width; ++index) {
        acc += weights[index] * inputs[index];
    }
    return acc;
}

}  // namespace eiel::cuda_architecture_b::hapb_v2_exactness_contract_v1
