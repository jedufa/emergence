#include "hapb_v2_policy_exactness_contract_v1.cuh"
#include "hapb_v2_policy_nx64_kernel.hpp"

#include <cuda_runtime.h>

namespace eiel::cuda_architecture_b::hapb_v2_experimental {
namespace contract = eiel::cuda_architecture_b::hapb_v2_exactness_contract_v1;

// One policy row per block.  64 useful threads.  Each thread owns one hidden
// neuron and preserves the frozen sequential per-neuron accumulation.
__global__ void hapb_v2_nx64_policy_logits_kernel_v1(
    const float* const packs,
    const float* const observations,
    const long long* const logical_slots,
    const float* const drives,
    const float* const attack_bias,
    const float* const action_temperature,
    const float* const drive_logit_gain,
    float* const logits,
    const long long rows,
    const long long slot_count,
    const long long drive_width,
    const float w_hunger_gather,
    const float w_pain_rest,
    const float w_fatigue_rest,
    const float w_pain_attack,
    const float w_fear_attack
) {
    const long long row = static_cast<long long>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    if (row >= rows || thread >= contract::kNx64ThreadsV1) {
        return;
    }
    const long long slot = logical_slots[row];
    if (slot < 0 || slot >= slot_count) {
        return;
    }
    const float* const pack = packs + row * contract::kPolicyWidthV1;
    const float* const observation =
        observations + row * contract::kObservationSizeV1;
    __shared__ float h1[contract::kHiddenSizeV1];
    __shared__ float h2[contract::kHiddenSizeV1];

    if (thread < contract::kHiddenSizeV1) {
        const float* const weights =
            pack + contract::kPackOffsetW1V1 + thread * contract::kObservationSizeV1;
        const float bias = pack[contract::kPackOffsetB1V1 + thread];
        h1[thread] = tanhf(contract::sequential_dot_plus_bias_exact_v1(
            weights, observation, bias, contract::kObservationSizeV1
        ));
    }
    __syncthreads();

    if (thread < contract::kHiddenSizeV1) {
        const float* const weights =
            pack + contract::kPackOffsetW2V1 + thread * contract::kHiddenSizeV1;
        const float bias = pack[contract::kPackOffsetB2V1 + thread];
        h2[thread] = tanhf(contract::sequential_dot_plus_bias_exact_v1(
            weights, h1, bias, contract::kHiddenSizeV1
        ));
    }
    __syncthreads();

    if (thread >= contract::kActionCountV1) {
        return;
    }
    const float raw_temperature = action_temperature[slot];
    const float temperature = raw_temperature < contract::kTemperatureFloorV1
        ? contract::kTemperatureFloorV1
        : raw_temperature;
    const float raw_gain = drive_logit_gain[slot];
    const float gain = raw_gain < contract::kGainFloorV1
        ? contract::kGainFloorV1
        : raw_gain;
    const float* const slot_drives = drives + slot * drive_width;
    const float* const weights =
        pack + contract::kPackOffsetWOutV1 + thread * contract::kHiddenSizeV1;
    const float bias = pack[contract::kPackOffsetBOutV1 + thread];
    float score = contract::sequential_dot_plus_bias_exact_v1(
        weights, h2, bias, contract::kHiddenSizeV1
    );
    if (thread == contract::kActionAttackV1) {
        score += attack_bias[slot];
    }
    score /= temperature;
    if (thread == contract::kActionGatherV1) {
        score += gain * (w_hunger_gather * slot_drives[0]);
    } else if (thread == contract::kActionRestV1) {
        score += gain * (w_pain_rest * slot_drives[1]
            + w_fatigue_rest * slot_drives[2]);
    } else if (thread == contract::kActionAttackV1) {
        score -= gain * (w_pain_attack * slot_drives[1]
            + w_fear_attack * slot_drives[3]);
    }
    logits[row * contract::kActionCountV1 + thread] = score;
}

// Isolated V1-launch twin for bake-off only.  Same sequential helper, one
// useful thread per row.  Not registered on the production HAPB artifact.
__global__ void hapb_v2_nx1_policy_logits_kernel_v1(
    const float* const packs,
    const float* const observations,
    const long long* const logical_slots,
    const float* const drives,
    const float* const attack_bias,
    const float* const action_temperature,
    const float* const drive_logit_gain,
    float* const logits,
    const long long rows,
    const long long slot_count,
    const long long drive_width,
    const float w_hunger_gather,
    const float w_pain_rest,
    const float w_fatigue_rest,
    const float w_pain_attack,
    const float w_fear_attack
) {
    const long long row = static_cast<long long>(blockIdx.x);
    if (threadIdx.x != 0 || row >= rows) {
        return;
    }
    const long long slot = logical_slots[row];
    if (slot < 0 || slot >= slot_count) {
        return;
    }
    const float* const pack = packs + row * contract::kPolicyWidthV1;
    const float* const observation =
        observations + row * contract::kObservationSizeV1;
    float h1[contract::kHiddenSizeV1];
    float h2[contract::kHiddenSizeV1];
    for (int hidden = 0; hidden < contract::kHiddenSizeV1; ++hidden) {
        const float* const weights =
            pack + contract::kPackOffsetW1V1 + hidden * contract::kObservationSizeV1;
        h1[hidden] = tanhf(contract::sequential_dot_plus_bias_exact_v1(
            weights, observation, pack[contract::kPackOffsetB1V1 + hidden],
            contract::kObservationSizeV1
        ));
    }
    for (int hidden = 0; hidden < contract::kHiddenSizeV1; ++hidden) {
        const float* const weights =
            pack + contract::kPackOffsetW2V1 + hidden * contract::kHiddenSizeV1;
        h2[hidden] = tanhf(contract::sequential_dot_plus_bias_exact_v1(
            weights, h1, pack[contract::kPackOffsetB2V1 + hidden],
            contract::kHiddenSizeV1
        ));
    }
    const float raw_temperature = action_temperature[slot];
    const float temperature = raw_temperature < contract::kTemperatureFloorV1
        ? contract::kTemperatureFloorV1
        : raw_temperature;
    const float raw_gain = drive_logit_gain[slot];
    const float gain = raw_gain < contract::kGainFloorV1
        ? contract::kGainFloorV1
        : raw_gain;
    const float* const slot_drives = drives + slot * drive_width;
    for (int action = 0; action < contract::kActionCountV1; ++action) {
        const float* const weights =
            pack + contract::kPackOffsetWOutV1 + action * contract::kHiddenSizeV1;
        float score = contract::sequential_dot_plus_bias_exact_v1(
            weights, h2, pack[contract::kPackOffsetBOutV1 + action],
            contract::kHiddenSizeV1
        );
        if (action == contract::kActionAttackV1) {
            score += attack_bias[slot];
        }
        score /= temperature;
        if (action == contract::kActionGatherV1) {
            score += gain * (w_hunger_gather * slot_drives[0]);
        } else if (action == contract::kActionRestV1) {
            score += gain * (w_pain_rest * slot_drives[1]
                + w_fatigue_rest * slot_drives[2]);
        } else if (action == contract::kActionAttackV1) {
            score -= gain * (w_pain_attack * slot_drives[1]
                + w_fear_attack * slot_drives[3]);
        }
        logits[row * contract::kActionCountV1 + action] = score;
    }
}

void hapb_v2_nx1_policy_logits_cuda_v1(
    const float* packs,
    const float* observations,
    const std::int64_t* logical_slots,
    const float* drives,
    const float* attack_bias,
    const float* action_temperature,
    const float* drive_logit_gain,
    float* logits,
    std::int64_t rows,
    std::int64_t slot_count,
    std::int64_t drive_width,
    float w_hunger_gather,
    float w_pain_rest,
    float w_fatigue_rest,
    float w_pain_attack,
    float w_fear_attack,
    void* stream
) {
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    hapb_v2_nx1_policy_logits_kernel_v1<<<
        static_cast<unsigned int>(rows),
        1,
        0,
        cuda_stream
    >>>(
        packs, observations,
        reinterpret_cast<const long long*>(logical_slots),
        drives, attack_bias, action_temperature, drive_logit_gain, logits,
        static_cast<long long>(rows),
        static_cast<long long>(slot_count),
        static_cast<long long>(drive_width),
        w_hunger_gather, w_pain_rest, w_fatigue_rest, w_pain_attack,
        w_fear_attack
    );
}

void hapb_v2_nx64_policy_logits_cuda_v1(
    const float* packs,
    const float* observations,
    const std::int64_t* logical_slots,
    const float* drives,
    const float* attack_bias,
    const float* action_temperature,
    const float* drive_logit_gain,
    float* logits,
    std::int64_t rows,
    std::int64_t slot_count,
    std::int64_t drive_width,
    float w_hunger_gather,
    float w_pain_rest,
    float w_fatigue_rest,
    float w_pain_attack,
    float w_fear_attack,
    void* stream
) {
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    hapb_v2_nx64_policy_logits_kernel_v1<<<
        static_cast<unsigned int>(rows),
        static_cast<unsigned int>(contract::kNx64ThreadsV1),
        0,
        cuda_stream
    >>>(
        packs, observations,
        reinterpret_cast<const long long*>(logical_slots),
        drives, attack_bias, action_temperature, drive_logit_gain, logits,
        static_cast<long long>(rows),
        static_cast<long long>(slot_count),
        static_cast<long long>(drive_width),
        w_hunger_gather, w_pain_rest, w_fatigue_rest, w_pain_attack,
        w_fear_attack
    );
}

}  // namespace eiel::cuda_architecture_b::hapb_v2_experimental
