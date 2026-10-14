#pragma once

#include <cstdint>

namespace eiel::cuda_architecture_b::hapb_v2_experimental {

// Isolated experimental launcher.  Not a production HAPB export.
// Requires EIEL_HAPB_V2_EXPERIMENTAL=1 at the Python entrypoint.
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
);

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
);

}  // namespace eiel::cuda_architecture_b::hapb_v2_experimental
