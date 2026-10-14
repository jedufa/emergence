#ifndef EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_GMNW_V1_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_GMNW_V1_CUH_

#include "cuda_architecture_b_flattened_gmnw_controller_v1.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace eiel::cuda_architecture_b::persistent_gmnw_v1 {

namespace controller = flattened_gmnw_controller_v1;

inline constexpr std::int8_t kCuePhase = 0;
inline constexpr std::int8_t kDelayPhase = 1;
inline constexpr std::int8_t kDecisionPhase = 2;
inline constexpr std::int8_t kConsequencePhase = 3;

__device__ __forceinline__ std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

__device__ __forceinline__ std::uint64_t episode_rng_state(
    const std::int64_t task_seed,
    const std::int64_t stream_id,
    const std::int64_t episode_index
) {
    const std::uint64_t mixed = static_cast<std::uint64_t>(task_seed)
        ^ (static_cast<std::uint64_t>(stream_id + 1) * 0xD6E8FEB86659FD93ULL)
        ^ (static_cast<std::uint64_t>(episode_index + 1) * 0xA5A3564E27F8862FULL);
    return splitmix64(mixed);
}

__device__ __forceinline__ std::int8_t balanced_cue(
    const std::int32_t cue_alphabet,
    const std::int64_t task_seed,
    const std::int64_t stream_id,
    const std::int64_t episode_index
) {
    const std::int64_t block = episode_index / cue_alphabet;
    const std::int64_t ordinal = episode_index % cue_alphabet;
    std::int8_t values[4] = {0, 1, 2, 3};
    std::uint64_t state = episode_rng_state(task_seed, stream_id, block);
    for (std::int32_t upper = cue_alphabet - 1; upper > 0; --upper) {
        state = splitmix64(state);
        const std::int32_t swap = static_cast<std::int32_t>(
            state % static_cast<std::uint64_t>(upper + 1)
        );
        const std::int8_t temporary = values[upper];
        values[upper] = values[swap];
        values[swap] = temporary;
    }
    return values[ordinal];
}

__device__ __forceinline__ void initialize_episode(
    const controller::FlatGmnwContextV1 context,
    const std::int64_t cell,
    const std::int64_t index,
    const std::int64_t next_episode,
    const std::int64_t stream
) {
    constexpr std::int8_t actions[4] = {0, 2, 1, 3};
    const std::int8_t cue = balanced_cue(
        context.cue_alphabet[cell], context.task_seed[cell], stream, next_episode
    );
    context.episode_index[index] = next_episode;
    context.cue_identity[index] = cue;
    context.cue_phase[index] = kCuePhase;
    context.phase_tick[index] = 0;
    context.delay_remaining[index] = 0;
    context.decision_due[index] = false;
    context.correct_action[index] = actions[cue];
    context.last_response[index] = -1;
    context.correctness[index] = -1;
    context.consequence_status[index] = 0;
    context.task_rng_state[index] = static_cast<std::int64_t>(
        episode_rng_state(context.task_seed[cell], stream, next_episode)
    );
    context.task_rng_cursor[index] = next_episode + 1;
}

__device__ __forceinline__ void clear_slot(
    const controller::FlatGmnwContextV1 context,
    const std::int64_t index
) {
    context.stream_id[index] = -1;
    context.episode_index[index] = -1;
    context.cue_identity[index] = -1;
    context.cue_phase[index] = 0;
    context.phase_tick[index] = 0;
    context.delay_remaining[index] = 0;
    context.decision_due[index] = false;
    context.correct_action[index] = -1;
    context.last_response[index] = -1;
    context.correctness[index] = -1;
    context.consequence_status[index] = 0;
    context.task_rng_state[index] = 0;
    context.task_rng_cursor[index] = 0;
    context.reproduction_scale[index] = 1.0f;
}

__device__ __forceinline__ bool context_valid(
    const controller::FlatGmnwContextV1 context,
    const std::int64_t cells,
    const std::int64_t capacity,
    const std::int64_t generation
) {
    if (!context.enabled) return true;
    if (
        context.cells != cells || context.capacity != capacity
        || context.generation_rows_per_cell <= 0
        || generation < context.generation_base
        || generation - context.generation_base >= context.generation_rows_per_cell
        || context.cue_ticks <= 0 || context.consequence_ticks <= 0
        || context.grace_generation_count <= 0
        || context.severe_generation_count <= 0
        || !isfinite(context.grace_damage_fraction)
        || !isfinite(context.severe_damage_fraction)
        || !isfinite(context.lethal_damage_fraction)
        || !isfinite(context.grace_reproduction_scale)
        || context.grace_damage_fraction < 0.0f
        || context.severe_damage_fraction < 0.0f
        || context.lethal_damage_fraction < 0.0f
        || context.grace_reproduction_scale < 0.0f
        || context.grace_reproduction_scale > 1.0f
        || context.stream_id == nullptr || context.episode_index == nullptr
        || context.cue_identity == nullptr || context.cue_phase == nullptr
        || context.phase_tick == nullptr || context.delay_remaining == nullptr
        || context.decision_due == nullptr || context.correct_action == nullptr
        || context.last_response == nullptr || context.correctness == nullptr
        || context.consequence_status == nullptr
        || context.task_rng_state == nullptr || context.task_rng_cursor == nullptr
        || context.reproduction_scale == nullptr || context.counters == nullptr
        || context.damage_total == nullptr || context.status == nullptr
        || context.cue_alphabet == nullptr || context.delay_ticks == nullptr
        || context.distractors_enabled == nullptr
        || context.reset_hidden_each_tick == nullptr || context.task_seed == nullptr
        || context.venv_consequence_profile == nullptr
        || (
            context.selection_metrics.enabled
            && (
                context.selection_metrics.counts == nullptr
                || context.selection_metrics.recurrent_upkeep_charge_total == nullptr
                || context.selection_metrics.status == nullptr
            )
        )
    ) {
        return false;
    }
    for (std::int64_t cell = 0; cell < cells; ++cell) {
        if (
            (context.cue_alphabet[cell] != 2 && context.cue_alphabet[cell] != 4)
            || context.delay_ticks[cell] <= 0
            || context.venv_consequence_profile[cell]
                < controller::kVenvConsequenceLegacyV1
            || context.venv_consequence_profile[cell]
                > controller::kVenvConsequenceP3V1
        ) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ std::int64_t selection_generation_row(
    const controller::FlatGmnwContextV1 context,
    const std::int64_t cell,
    const std::int64_t generation
) {
    const std::int64_t generation_offset = generation - context.generation_base;
    return cell * context.generation_rows_per_cell + generation_offset;
}

__device__ __forceinline__ std::int64_t* selection_count_row(
    const controller::FlatGmnwContextV1 context,
    const std::int64_t cell,
    const std::int64_t generation,
    const bool policy_use_gru
) {
    const std::int64_t generation_row = selection_generation_row(
        context, cell, generation
    );
    const std::int64_t family = policy_use_gru ? 1 : 0;
    return context.selection_metrics.counts
        + (
            generation_row * controller::kSelectionPolicyFamilyCountV1 + family
        ) * controller::kSelectionCountMetricCountV1;
}

__device__ __forceinline__ double* selection_upkeep_row(
    const controller::FlatGmnwContextV1 context,
    const std::int64_t cell,
    const std::int64_t generation,
    const bool policy_use_gru
) {
    const std::int64_t generation_row = selection_generation_row(
        context, cell, generation
    );
    const std::int64_t family = policy_use_gru ? 1 : 0;
    return context.selection_metrics.recurrent_upkeep_charge_total
        + generation_row * controller::kSelectionPolicyFamilyCountV1 + family;
}

__device__ __forceinline__ bool sync_slots(
    const controller::FlatGmnwContextV1 context,
    const std::int64_t cell,
    const bool* const alive_mask,
    const std::int64_t* const slot_to_agent_id
) {
    const std::int64_t offset = cell * context.capacity;
    for (std::int64_t slot = 0; slot < context.capacity; ++slot) {
        const std::int64_t index = offset + slot;
        if (!alive_mask[index]) {
            clear_slot(context, index);
            continue;
        }
        const std::int64_t agent_id = slot_to_agent_id[index];
        if (agent_id < 0) return false;
        if (context.stream_id[index] != agent_id) {
            context.stream_id[index] = agent_id;
            context.reproduction_scale[index] = 1.0f;
            initialize_episode(context, cell, index, 0, agent_id);
        }
        if (
            context.episode_index[index] < 0
            || context.cue_identity[index] < 0
            || context.cue_identity[index] >= context.cue_alphabet[cell]
            || context.cue_phase[index] < kCuePhase
            || context.cue_phase[index] > kConsequencePhase
            || !isfinite(context.reproduction_scale[index])
            || context.reproduction_scale[index] < 0.0f
            || context.reproduction_scale[index] > 1.0f
        ) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ void write_observation(
    const controller::FlatGmnwContextV1 context,
    const std::int64_t cell,
    const std::int64_t index,
    float* const observation
) {
    for (std::int64_t column = 0; column < controller::kObservationColumnsV1; ++column) {
        observation[column] = 0.0f;
    }
    const std::int8_t phase = context.cue_phase[index];
    const std::int8_t arm = context.control_matrix.enabled
            && context.control_matrix.policy_arm != nullptr
        ? context.control_matrix.policy_arm[cell]
        : static_cast<std::int8_t>(-1);
    const bool cue_visible_at_decision =
        arm == gmnw_control_matrix_v1::kPersistentGruImmediateArmV1
        || arm == gmnw_control_matrix_v1::kMatchedMlpImmediateArmV1
        || arm == gmnw_control_matrix_v1::kMixedSelectionImmediateArmV1;
    if (
        phase == kCuePhase
        || (phase == kDecisionPhase && cue_visible_at_decision)
    ) {
        observation[context.cue_identity[index]] = 1.0f;
    }
    observation[4 + phase] = 1.0f;
    if (phase == kDelayPhase && context.distractors_enabled[cell]) {
        const std::int64_t elapsed =
            context.delay_ticks[cell] - context.delay_remaining[index];
        const std::uint64_t distractor = splitmix64(
            static_cast<std::uint64_t>(context.task_rng_state[index])
            ^ (static_cast<std::uint64_t>(elapsed + 1) * 0x8CB92BA72F3D8DD7ULL)
            ^ 0x47524D4E575F4453ULL
        );
        observation[8 + static_cast<std::int64_t>(distractor % 4ULL)] = 1.0f;
    }
}

__device__ __forceinline__ bool advance_slot(
    const controller::FlatGmnwContextV1 context,
    const std::int64_t cell,
    const std::int64_t generation,
    const std::int64_t index,
    const bool policy_use_gru,
    const std::int32_t private_action,
    const float start_health,
    float* const health
) {
    const std::int8_t phase = context.cue_phase[index];
    if (phase == kCuePhase) {
        context.phase_tick[index] += 1;
        if (context.phase_tick[index] >= context.cue_ticks) {
            context.cue_phase[index] = kDelayPhase;
            context.phase_tick[index] = 0;
            context.delay_remaining[index] = context.delay_ticks[cell];
        }
        return false;
    }
    if (phase == kDelayPhase) {
        context.delay_remaining[index] -= 1;
        if (context.delay_remaining[index] > 0) {
            context.phase_tick[index] += 1;
        } else {
            context.cue_phase[index] = kDecisionPhase;
            context.phase_tick[index] = 0;
            context.delay_remaining[index] = 0;
            context.decision_due[index] = true;
        }
        return false;
    }
    if (phase == kDecisionPhase) {
        const bool correct = private_action == context.correct_action[index];
        std::int64_t* selection_counts = nullptr;
        if (context.selection_metrics.enabled) {
            selection_counts = selection_count_row(
                context, cell, generation, policy_use_gru
            );
            selection_counts[controller::kSelectionDecisionCountV1] += 1;
            selection_counts[controller::kSelectionCorrectCountV1] +=
                correct ? 1 : 0;
        }
        context.cue_phase[index] = kConsequencePhase;
        context.phase_tick[index] = 0;
        context.decision_due[index] = false;
        context.last_response[index] = static_cast<std::int8_t>(private_action);
        context.correctness[index] = correct ? 1 : 0;
        context.consequence_status[index] = correct ? 1 : 2;
        const std::int64_t generation_row =
            generation - context.generation_base;
        std::int64_t* const counters = context.counters
            + (cell * context.generation_rows_per_cell + generation_row)
                * controller::kCounterColumnsV1;
        counters[0] += 1;
        counters[correct ? 1 : 2] += 1;
        if (correct) return false;

        float damage = 0.0f;
        float scale = 0.0f;
        const std::int32_t consequence_profile =
            context.venv_consequence_profile[cell];
        if (consequence_profile == controller::kVenvConsequenceLegacyV1) {
            damage = static_cast<float>(
                context.lethal_damage_fraction * static_cast<double>(start_health)
            );
            if (generation < context.grace_generation_count) {
                damage = static_cast<float>(
                    context.grace_damage_fraction * static_cast<double>(start_health)
                );
                scale = context.grace_reproduction_scale;
                counters[3] += 1;
            } else if (
                generation
                    < context.grace_generation_count
                        + context.severe_generation_count
            ) {
                damage = static_cast<float>(
                    context.severe_damage_fraction * static_cast<double>(start_health)
                );
                counters[4] += 1;
            } else {
                counters[4] += 1;
            }
        } else {
            if (generation < context.grace_generation_count) {
                counters[3] += 1;
            } else {
                counters[4] += 1;
            }
            if (consequence_profile == controller::kVenvConsequenceP2V1) {
                damage = static_cast<float>(
                    0.25 * context.lethal_damage_fraction
                        * static_cast<double>(start_health)
                );
            } else if (
                consequence_profile == controller::kVenvConsequenceP3V1
            ) {
                damage = static_cast<float>(
                    0.5 * context.lethal_damage_fraction
                        * static_cast<double>(start_health)
                );
            }
        }
        health[index] -= damage;
        context.reproduction_scale[index] = fminf(
            context.reproduction_scale[index], scale
        );
        context.damage_total[
            cell * context.generation_rows_per_cell + generation_row
        ] += static_cast<double>(damage);
        if (health[index] <= 0.0f) {
            health[index] = 0.0f;
            context.reproduction_scale[index] = 0.0f;
            counters[5] += 1;
            if (selection_counts != nullptr) {
                selection_counts[controller::kSelectionTaskDeathCountV1] += 1;
            }
            return true;
        }
        return false;
    }
    context.phase_tick[index] += 1;
    if (context.phase_tick[index] >= context.consequence_ticks) {
        context.reproduction_scale[index] = 1.0f;
        initialize_episode(
            context,
            cell,
            index,
            context.episode_index[index] + 1,
            context.stream_id[index]
        );
    }
    return false;
}

}  // namespace eiel::cuda_architecture_b::persistent_gmnw_v1

#endif
