#pragma once

#include "native/cuda_architecture_b_pcg64.cuh"
#include "native/recurrent/stable_gru_v1.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace eiel::cuda_architecture_b::persistent_stochastic_policy {

constexpr int64_t kMaxHiddenSize = 64;
constexpr int64_t kMaxObservationSize = 128;
constexpr int64_t kMaxActionCount = 16;
constexpr int32_t kGatherAction = 4;
constexpr int32_t kRestAction = 5;
constexpr int32_t kAttackAction = 6;
constexpr float kGumbelEps = 1.0e-6f;

struct FixedMlpPolicyConfig {
    int64_t active_rows;
    int64_t capacity;
    int64_t observation_size;
    int64_t hidden_size;
    int64_t action_count;
    int64_t packed_width;
    int64_t drive_width;
    int64_t cache_capacity;
    int64_t cache_width;
    float w_hunger_gather;
    float w_pain_rest;
    float w_fatigue_rest;
    float w_pain_attack;
    float w_fear_attack;
    bool deterministic_actions;
};

struct PolicyCacheCounters {
    int64_t requested_rows;
    int64_t hit_rows;
    int64_t missed_rows;
    int64_t refilled_rows;
    int64_t refill_calls;
    int64_t full_refill_ticks;
    int64_t partial_refill_ticks;
    int64_t gumbel_draws;
};

struct StableGruPolicyConfig {
    int64_t active_rows;
    int64_t capacity;
    int64_t observation_size;
    int64_t packed_width;
    int64_t drive_width;
    int64_t cache_capacity;
    int64_t cache_width;
    float w_hunger_gather;
    float w_pain_rest;
    float w_fatigue_rest;
    float w_pain_attack;
    float w_fear_attack;
    bool deterministic_actions;
    bool immutable_shared_policy;
};

__device__ __forceinline__ void service_stable_gru_gumbel_policy_serial(
    const int64_t* __restrict__ ordered_agent_ids,
    const int32_t* __restrict__ ordered_slots,
    const float* __restrict__ observations,
    const float* __restrict__ gru_pack,
    float* __restrict__ hmem,
    const int64_t* __restrict__ policy_mem_dim,
    const bool* __restrict__ policy_use_gru,
    const float* __restrict__ drives,
    const float* __restrict__ attack_bias,
    const float* __restrict__ action_temperature,
    const float* __restrict__ drive_logit_gain,
    const int64_t* __restrict__ slot_tokens,
    int32_t* __restrict__ action_cache_actions,
    int64_t* __restrict__ action_cache_token,
    int32_t* __restrict__ action_cache_cursor,
    int32_t* __restrict__ action_cache_next,
    float* __restrict__ logits_out,
    int32_t* __restrict__ row_actions_out,
    int64_t* __restrict__ miss_rows,
    float* __restrict__ scratch,
    uint64_t* __restrict__ pcg64_state_words,
    int64_t* __restrict__ pcg64_buffer_words,
    PolicyCacheCounters* __restrict__ counters_out,
    int32_t* __restrict__ status_out,
    const StableGruPolicyConfig config
) {
    namespace stable = eiel::native::recurrent_v1;
    PolicyCacheCounters counters{};
    status_out[0] = 0;
    if (
        config.active_rows < 0 || config.active_rows > config.capacity
        || config.observation_size <= 0
        || config.packed_width <= 0
        || config.drive_width < 4
        || config.cache_width <= 0
        || config.cache_width > config.cache_capacity
        || !gru_pack || !hmem || !policy_mem_dim || !policy_use_gru || !scratch
    ) {
        status_out[0] = 1;
        *counters_out = counters;
        return;
    }

    for (int64_t row = 0; row < config.active_rows; ++row) {
        ++counters.requested_rows;
        const int32_t slot = ordered_slots[row];
        if (
            slot < 0 || static_cast<int64_t>(slot) >= config.capacity
            || !policy_use_gru[slot]
        ) {
            status_out[0] = 2;
            *counters_out = counters;
            return;
        }
        const int32_t cursor = action_cache_cursor[slot];
        const bool cache_hit = action_cache_token[slot] == slot_tokens[slot]
            && cursor >= 0 && static_cast<int64_t>(cursor) < config.cache_width;
        if (!cache_hit) {
            miss_rows[counters.missed_rows++] = row;
            continue;
        }
        int32_t action = action_cache_actions[
            static_cast<int64_t>(slot) * config.cache_capacity + cursor
        ];
        if (action < 0 || action >= stable::kAcceptedActionCountV1) {
            action = action_cache_next[slot];
        }
        if (action < 0 || action >= stable::kAcceptedActionCountV1) {
            action = kRestAction;
        }
        action_cache_next[slot] = action;
        action_cache_cursor[slot] = cursor + 1;
        row_actions_out[row] = action;
        ++counters.hit_rows;
    }

    if (config.cache_width > 1) {
        for (int64_t left = 0; left < counters.missed_rows; ++left) {
            int64_t selected = left;
            for (int64_t candidate = left + 1; candidate < counters.missed_rows; ++candidate) {
                const int64_t selected_row = miss_rows[selected];
                const int64_t candidate_row = miss_rows[candidate];
                if (
                    ordered_agent_ids[candidate_row] < ordered_agent_ids[selected_row]
                    || (
                        ordered_agent_ids[candidate_row] == ordered_agent_ids[selected_row]
                        && candidate_row < selected_row
                    )
                ) {
                    selected = candidate;
                }
            }
            const int64_t temporary = miss_rows[left];
            miss_rows[left] = miss_rows[selected];
            miss_rows[selected] = temporary;
        }
    }

    Pcg64State rng{};
    if (!config.deterministic_actions) {
        rng = Pcg64State{
            pcg64_state_words[0], pcg64_state_words[1],
            pcg64_state_words[2], pcg64_state_words[3],
            static_cast<uint32_t>(pcg64_buffer_words[0]),
            static_cast<uint32_t>(pcg64_buffer_words[1]),
        };
    }
    const stable::StableGruShapeV1 shape{
        stable::kRecurrentContractVersionV1,
        config.observation_size,
        stable::kAcceptedStableGruHiddenV1,
        stable::kAcceptedActionCountV1,
    };
    for (int64_t miss = 0; miss < counters.missed_rows; ++miss) {
        const int64_t row = miss_rows[miss];
        const int32_t slot = ordered_slots[row];
        float* const row_scratch = scratch + static_cast<int64_t>(slot) * 519;
        stable::StableGruStepScratchV1 scratch_view{
            row_scratch, 64,
            row_scratch + 64, 192,
            row_scratch + 256, 192,
            row_scratch + 448, 64,
            row_scratch + 512, 7,
        };
        int32_t deterministic_action = 0;
        stable::StableGruStepInputV1 input{
            shape,
            observations + row * config.observation_size,
            config.observation_size,
            hmem + static_cast<int64_t>(slot) * 64,
            64,
            policy_mem_dim[slot],
            gru_pack + (config.immutable_shared_policy
                ? 0
                : static_cast<int64_t>(slot) * config.packed_width),
            config.packed_width,
            attack_bias[slot],
            action_temperature[slot],
            drives + static_cast<int64_t>(slot) * config.drive_width,
            config.drive_width,
            drive_logit_gain[slot] < 0.0f ? 0.0f : drive_logit_gain[slot],
            config.w_hunger_gather,
            config.w_pain_rest,
            config.w_fatigue_rest,
            config.w_pain_attack,
            config.w_fear_attack,
        };
        stable::StableGruStepOutputV1 output{
            hmem + static_cast<int64_t>(slot) * 64,
            64,
            logits_out + row * 7,
            7,
            &deterministic_action,
        };
        const auto stable_status = stable::stable_gru_forward_one(
            input, scratch_view, output, nullptr
        );
        if (!stable::ok(stable_status)) {
            status_out[0] = 100 + static_cast<int32_t>(stable_status);
            *counters_out = counters;
            return;
        }

        int32_t action = deterministic_action;
        if (!config.deterministic_actions) {
            float best_score = 0.0f;
            for (int64_t candidate = 0; candidate < 7; ++candidate) {
                float uniform = pcg64_numpy_float32(rng);
                uniform = uniform < kGumbelEps ? kGumbelEps
                    : (uniform > 1.0f - kGumbelEps ? 1.0f - kGumbelEps : uniform);
                const float score = logits_out[row * 7 + candidate]
                    - logf(-logf(uniform));
                if (candidate == 0 || score > best_score) {
                    best_score = score;
                    action = static_cast<int32_t>(candidate);
                }
            }
        }
        const int64_t cache_offset = static_cast<int64_t>(slot) * config.cache_capacity;
        for (int64_t cache = 0; cache < config.cache_width; ++cache) {
            action_cache_actions[cache_offset + cache] = action;
        }
        action_cache_token[slot] = slot_tokens[slot];
        action_cache_cursor[slot] = 1;
        action_cache_next[slot] = action;
        row_actions_out[row] = action;
        ++counters.refilled_rows;
    }
    counters.refill_calls = counters.refilled_rows > 0 ? 1 : 0;
    counters.full_refill_ticks = counters.refilled_rows > 0 && counters.hit_rows == 0 ? 1 : 0;
    counters.partial_refill_ticks = counters.refilled_rows > 0 && counters.hit_rows > 0 ? 1 : 0;
    counters.gumbel_draws = config.deterministic_actions
        ? 0 : counters.refilled_rows * 7;
    if (!config.deterministic_actions) {
        pcg64_state_words[0] = rng.state_lo;
        pcg64_state_words[1] = rng.state_hi;
        pcg64_state_words[2] = rng.inc_lo;
        pcg64_state_words[3] = rng.inc_hi;
        pcg64_buffer_words[0] = static_cast<int64_t>(rng.has_uint32);
        pcg64_buffer_words[1] = static_cast<int64_t>(rng.uinteger);
    }
    *counters_out = counters;
}

// The caller supplies active rows in the accepted active-agent order. `miss_rows`
// must have at least `active_rows` elements. This serial helper deliberately
// mirrors the CPU cache-service sequence: consume hits first, then refill misses
// in agent-id order when the cache spans more than one tick.
__device__ __forceinline__ void service_fixed_mlp_gumbel_policy_serial(
    const int64_t* __restrict__ ordered_agent_ids,
    const int32_t* __restrict__ ordered_slots,
    const float* __restrict__ observations,
    const float* __restrict__ mlp_pack,
    const float* __restrict__ drives,
    const float* __restrict__ attack_bias,
    const float* __restrict__ action_temperature,
    const float* __restrict__ drive_logit_gain,
    const int64_t* __restrict__ slot_tokens,
    int32_t* __restrict__ action_cache_actions,
    int64_t* __restrict__ action_cache_token,
    int32_t* __restrict__ action_cache_cursor,
    int32_t* __restrict__ action_cache_next,
    float* __restrict__ logits_out,
    int32_t* __restrict__ row_actions_out,
    int64_t* __restrict__ miss_rows,
    uint64_t* __restrict__ pcg64_state_words,
    int64_t* __restrict__ pcg64_buffer_words,
    PolicyCacheCounters* __restrict__ counters_out,
    const FixedMlpPolicyConfig config
) {
    PolicyCacheCounters counters{};
    const int64_t active_rows = config.active_rows < 0 ? 0 : config.active_rows;

    // Phase 1 is intentionally fail-closed at the helper boundary. The primary
    // persistent kernel keeps its established ABI and validates its own inputs.
    if (
        active_rows > config.capacity
        || config.observation_size <= 0
        || config.observation_size > kMaxObservationSize
        || config.hidden_size <= 0
        || config.hidden_size > kMaxHiddenSize
        || config.action_count <= 0
        || config.action_count > kMaxActionCount
        || config.drive_width < 4
        || config.cache_width <= 0
        || config.cache_width > config.cache_capacity
    ) {
        *counters_out = counters;
        return;
    }

    for (int64_t row = 0; row < active_rows; ++row) {
        ++counters.requested_rows;
        const int32_t slot = ordered_slots[row];
        if (slot < 0 || static_cast<int64_t>(slot) >= config.capacity) {
            miss_rows[counters.missed_rows++] = row;
            continue;
        }
        const int32_t cursor = action_cache_cursor[slot];
        const bool cache_hit = action_cache_token[slot] == slot_tokens[slot]
            && cursor >= 0
            && static_cast<int64_t>(cursor) < config.cache_width;
        if (!cache_hit) {
            miss_rows[counters.missed_rows++] = row;
            continue;
        }

        int32_t action = action_cache_actions[
            static_cast<int64_t>(slot) * config.cache_capacity + cursor
        ];
        if (action < 0 || action > kAttackAction) {
            action = action_cache_next[slot];
        }
        if (action < 0 || action > kAttackAction) {
            action = kRestAction;
        }
        action_cache_next[slot] = action;
        action_cache_cursor[slot] = cursor + 1;
        row_actions_out[row] = action;
        ++counters.hit_rows;
    }

    // The CPU oracle only sorts misses for multi-tick caches. Agent IDs are
    // unique under the active-row contract; row is a deterministic tie fallback.
    if (config.cache_width > 1) {
        for (int64_t left = 0; left < counters.missed_rows; ++left) {
            int64_t selected = left;
            for (int64_t candidate = left + 1; candidate < counters.missed_rows; ++candidate) {
                const int64_t selected_row = miss_rows[selected];
                const int64_t candidate_row = miss_rows[candidate];
                const int64_t selected_id = ordered_agent_ids[selected_row];
                const int64_t candidate_id = ordered_agent_ids[candidate_row];
                if (
                    candidate_id < selected_id
                    || (candidate_id == selected_id && candidate_row < selected_row)
                ) {
                    selected = candidate;
                }
            }
            const int64_t tmp = miss_rows[left];
            miss_rows[left] = miss_rows[selected];
            miss_rows[selected] = tmp;
        }
    }

    Pcg64State rng{};
    if (!config.deterministic_actions) {
        rng = Pcg64State{
            pcg64_state_words[0],
            pcg64_state_words[1],
            pcg64_state_words[2],
            pcg64_state_words[3],
            static_cast<uint32_t>(pcg64_buffer_words[0]),
            static_cast<uint32_t>(pcg64_buffer_words[1]),
        };
    }

    for (int64_t miss_index = 0; miss_index < counters.missed_rows; ++miss_index) {
        const int64_t row = miss_rows[miss_index];
        const int32_t slot = ordered_slots[row];
        if (slot < 0 || static_cast<int64_t>(slot) >= config.capacity) {
            row_actions_out[row] = -1;
            continue;
        }

        const float* observation = observations + row * config.observation_size;
        const float* pack = mlp_pack + static_cast<int64_t>(slot) * config.packed_width;
        float h1[kMaxHiddenSize];
        float h2[kMaxHiddenSize];
        int64_t pack_offset = 0;
        for (int64_t hidden = 0; hidden < config.hidden_size; ++hidden) {
            float value = pack[config.hidden_size * config.observation_size + hidden];
            const float* weights = pack + hidden * config.observation_size;
            for (int64_t feature = 0; feature < config.observation_size; ++feature) {
                value += weights[feature] * observation[feature];
            }
            h1[hidden] = tanhf(value);
        }
        pack_offset += config.hidden_size * config.observation_size + config.hidden_size;
        for (int64_t hidden = 0; hidden < config.hidden_size; ++hidden) {
            float value = pack[pack_offset + config.hidden_size * config.hidden_size + hidden];
            const float* weights = pack + pack_offset + hidden * config.hidden_size;
            for (int64_t input = 0; input < config.hidden_size; ++input) {
                value += weights[input] * h1[input];
            }
            h2[hidden] = tanhf(value);
        }
        pack_offset += config.hidden_size * config.hidden_size + config.hidden_size;

        const float temperature = action_temperature[slot] < 0.05f
            ? 0.05f
            : action_temperature[slot];
        const float gain = drive_logit_gain[slot] < 0.0f ? 0.0f : drive_logit_gain[slot];
        const float* slot_drives = drives + static_cast<int64_t>(slot) * config.drive_width;
        float* row_logits = logits_out + row * config.action_count;
        for (int64_t action = 0; action < config.action_count; ++action) {
            const float* weights = pack + pack_offset + action * config.hidden_size;
            float score = pack[pack_offset + config.action_count * config.hidden_size + action];
            for (int64_t hidden = 0; hidden < config.hidden_size; ++hidden) {
                score += weights[hidden] * h2[hidden];
            }
            if (action == kAttackAction) {
                score += attack_bias[slot];
            }
            score /= temperature;
            if (action == kGatherAction) {
                score += gain * (config.w_hunger_gather * slot_drives[0]);
            } else if (action == kRestAction) {
                score += gain * (
                    config.w_pain_rest * slot_drives[1]
                    + config.w_fatigue_rest * slot_drives[2]
                );
            } else if (action == kAttackAction) {
                score -= gain * (
                    config.w_pain_attack * slot_drives[1]
                    + config.w_fear_attack * slot_drives[3]
                );
            }
            row_logits[action] = score;
        }

        const int64_t cache_offset = static_cast<int64_t>(slot) * config.cache_capacity;
        for (int64_t cache_index = 0; cache_index < config.cache_width; ++cache_index) {
            int32_t best_action = 0;
            float best_score = row_logits[0];
            if (!config.deterministic_actions) {
                float uniform = pcg64_numpy_float32(rng);
                uniform = uniform < kGumbelEps ? kGumbelEps
                    : (uniform > (1.0f - kGumbelEps) ? 1.0f - kGumbelEps : uniform);
                best_score += -logf(-logf(uniform));
            }
            for (int64_t action = 1; action < config.action_count; ++action) {
                float score = row_logits[action];
                if (!config.deterministic_actions) {
                    float uniform = pcg64_numpy_float32(rng);
                    uniform = uniform < kGumbelEps ? kGumbelEps
                        : (uniform > (1.0f - kGumbelEps) ? 1.0f - kGumbelEps : uniform);
                    score += -logf(-logf(uniform));
                }
                if (score > best_score) {
                    best_score = score;
                    best_action = static_cast<int32_t>(action);
                }
            }
            action_cache_actions[cache_offset + cache_index] = best_action;
            if (cache_index == 0) {
                row_actions_out[row] = best_action;
                action_cache_next[slot] = best_action;
            }
        }
        action_cache_token[slot] = slot_tokens[slot];
        action_cache_cursor[slot] = 1;
        ++counters.refilled_rows;
    }

    counters.refill_calls = counters.refilled_rows > 0 ? 1 : 0;
    counters.full_refill_ticks = counters.refilled_rows > 0 && counters.hit_rows == 0 ? 1 : 0;
    counters.partial_refill_ticks = counters.refilled_rows > 0 && counters.hit_rows > 0 ? 1 : 0;
    counters.gumbel_draws = config.deterministic_actions
        ? 0
        : counters.refilled_rows * config.cache_width * config.action_count;
    if (!config.deterministic_actions) {
        pcg64_state_words[0] = rng.state_lo;
        pcg64_state_words[1] = rng.state_hi;
        pcg64_state_words[2] = rng.inc_lo;
        pcg64_state_words[3] = rng.inc_hi;
        pcg64_buffer_words[0] = static_cast<int64_t>(rng.has_uint32);
        pcg64_buffer_words[1] = static_cast<int64_t>(rng.uinteger);
    }
    *counters_out = counters;
}

}  // namespace eiel::cuda_architecture_b::persistent_stochastic_policy
