#include "../../engine/perf/cuda_architecture_b_persistent_stochastic_policy.cuh"

#include <cstdint>

namespace stochastic_policy = eiel::cuda_architecture_b::persistent_stochastic_policy;

__global__ void compile_persistent_stochastic_policy_probe(
    const int64_t* ordered_agent_ids,
    const int32_t* ordered_slots,
    const float* observations,
    const float* mlp_pack,
    const float* drives,
    const float* attack_bias,
    const float* action_temperature,
    const float* drive_logit_gain,
    const int64_t* slot_tokens,
    int32_t* action_cache_actions,
    int64_t* action_cache_token,
    int32_t* action_cache_cursor,
    int32_t* action_cache_next,
    float* logits_out,
    int32_t* row_actions_out,
    int64_t* miss_rows,
    uint64_t* pcg64_state_words,
    int64_t* pcg64_buffer_words,
    stochastic_policy::PolicyCacheCounters* counters_out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const stochastic_policy::FixedMlpPolicyConfig config{
        3,
        8,
        11,
        8,
        7,
        231,
        4,
        4,
        2,
        1.2f,
        0.9f,
        1.0f,
        1.2f,
        1.4f,
        false,
    };
    stochastic_policy::service_fixed_mlp_gumbel_policy_serial(
        ordered_agent_ids,
        ordered_slots,
        observations,
        mlp_pack,
        drives,
        attack_bias,
        action_temperature,
        drive_logit_gain,
        slot_tokens,
        action_cache_actions,
        action_cache_token,
        action_cache_cursor,
        action_cache_next,
        logits_out,
        row_actions_out,
        miss_rows,
        pcg64_state_words,
        pcg64_buffer_words,
        counters_out,
        config
    );
}
