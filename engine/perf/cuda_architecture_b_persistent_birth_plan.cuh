#ifndef EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_BIRTH_PLAN_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_BIRTH_PLAN_CUH_

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "native/cuda_architecture_b_pcg64.cuh"

namespace eiel::cuda_architecture_b::persistent_birth_plan {

constexpr int64_t kInt64Max = 9223372036854775807LL;
constexpr int64_t kInt64Min = -kInt64Max - 1;

enum OrdinaryBirthPlanStatus : int32_t {
    kStatusOk = 0,
    kStatusInvalidResidentState = 1,
    kStatusInvalidAgentIds = 2,
    kStatusUnsupportedGru = 3,
    kStatusInvalidPcg64State = 4,
    kStatusInvalidConfig = 5,
    kStatusCounterOverflow = 6,
};

enum OrdinaryBirthPlanReason : int32_t {
    kReasonAccepted = 0,
    kReasonNoSpace = 1,
    kReasonBuffer = 2,
    kReasonNoFreeSlot = 3,
    kReasonOpportunity = 4,
    kReasonConception = 5,
    kReasonGestation = 6,
    kReasonHealth = 7,
    kReasonCooldown = 8,
    kReasonNoEnergy = 9,
    kReasonCount = 10,
};

// Every pointer is already offset to the simulation cell owned by the caller.
// Pointer validity, element types, and physical extents remain launch-boundary
// obligations, matching the host checks around the authoritative Stage2 kernel.
struct FlatOrdinaryBirthPlanState {
    const int64_t* slot_to_agent_id;
    const bool* alive_mask;
    const bool* policy_use_gru;
    const int32_t* pos_y;
    const int32_t* pos_x;
    const int32_t* agent_grid;
    const float* energy;
    const double* policy_energy_observation;
    const float* health;
    const float* injury;
    const float* scar;
    int32_t* repro_cooldown;
    const float* repro_threshold;
    const int64_t* slot_free_queue;
    const int64_t* slot_free_queue_count;
    int64_t* slot_free_queue_head;
    uint64_t* pcg64_state_words;
    int64_t* pcg64_buffer_words;
    int64_t* policy_np_semantic_draws;
    uint64_t* runner_pcg64_state_words;
    int64_t* runner_pcg64_buffer_words;
    int64_t* metrics_np_semantic_draws;
    const float* reproduction_scale;
    const int32_t* survival_steps;
};

// All storage is fixed-capacity caller-owned device memory. Plan arrays have
// max_births rows, decision arrays have slot_count rows, reason_counts has
// kReasonCount rows, and each scalar output has one row.
struct FlatOrdinaryBirthPlanBuffers {
    int32_t* planned_repro_cooldown;
    int64_t* planned_parent_slots;
    int64_t* planned_child_slots;
    int32_t* planned_spawn_positions;
    float* planned_child_energy;
    int64_t* decision_parent_slots;
    int32_t* decision_reason_codes;
    int32_t* decision_candidate_counts;
    int64_t* decision_draw_starts;
    int64_t* decision_draw_ends;
    int32_t* decision_selected_positions;
    bool* decision_selected_reserved;
    int64_t* reason_counts;
    int64_t* planned_count;
    int64_t* decision_count;
    int64_t* spawn_rng_draws_added;
    int64_t* spawn_rng_semantic_draws_added;
    int64_t* runner_rng_draws_added;
    int64_t* runner_rng_semantic_draws_added;
    int32_t* status;
};

struct OrdinaryBirthPlanConfig {
    int64_t slot_count;
    int64_t grid_height;
    int64_t grid_width;
    int64_t free_queue_capacity;
    int64_t max_births;
    double parent_energy_share;
    float minimum_energy_buffer;
    double injury_repro_k_injury;
    double injury_repro_k_scar;
    double injury_birth_fail_scar;
    double injury_birth_fail_injury;
    bool env_policy_alias;
    bool requires_mate;
    bool injury_gate_active;
    bool conception_gate_active;
    bool gestation_gate_active;
    bool reproduction_scale_gate_active;
    int32_t life_history_mode;
    int32_t maturity_start_ticks;
    int32_t maturity_rise_ticks;
    int32_t fertility_decline_start_ticks;
    int32_t fertility_decline_ticks;
    float elder_fertility_floor;
};

// NumPy Generator.integers(0, max_value + 1) rejection contract used by the
// accepted Stage2 Fisher-Yates path. Each call is one planner draw even when
// rejection consumes more than one raw uint32 from the PCG64 cursor.
__device__ __forceinline__ uint32_t pcg64_bounded_uint32_exact(
    Pcg64State& rng,
    const uint32_t max_value
) {
    if (max_value == 0xffffffffU) {
        return pcg64_raw32(rng);
    }
    uint32_t mask = max_value;
    mask |= mask >> 1U;
    mask |= mask >> 2U;
    mask |= mask >> 4U;
    mask |= mask >> 8U;
    mask |= mask >> 16U;
    uint32_t value = 0;
    do {
        value = pcg64_raw32(rng) & mask;
    } while (value > max_value);
    return value;
}

__device__ __forceinline__ bool selected_position_is_reserved(
    const int32_t y,
    const int32_t x,
    const int32_t* planned_spawn_positions,
    const int64_t planned_count
) {
    for (int64_t row = 0; row < planned_count; ++row) {
        if (
            planned_spawn_positions[row * 2] == y
            && planned_spawn_positions[row * 2 + 1] == x
        ) {
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ bool child_slot_already_planned(
    const int64_t slot,
    const int64_t* planned_child_slots,
    const int64_t planned_count
) {
    for (int64_t row = 0; row < planned_count; ++row) {
        if (planned_child_slots[row] == slot) {
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ float life_history_fertility_scale_v1(
    const int32_t age_ticks,
    const OrdinaryBirthPlanConfig config
) {
    if (config.life_history_mode == 0) {
        return 1.0f;
    }
    if (age_ticks < config.maturity_start_ticks) {
        return 0.0f;
    }
    float scale = 1.0f;
    if (config.maturity_rise_ticks > 0) {
        const int32_t elapsed = age_ticks - config.maturity_start_ticks + 1;
        scale = static_cast<float>(elapsed)
            / static_cast<float>(config.maturity_rise_ticks);
        if (scale > 1.0f) {
            scale = 1.0f;
        }
    }
    if (
        config.fertility_decline_start_ticks > 0
        && age_ticks >= config.fertility_decline_start_ticks
    ) {
        if (config.fertility_decline_ticks <= 0) {
            scale = fminf(scale, config.elder_fertility_floor);
        } else {
            const int32_t elapsed =
                age_ticks - config.fertility_decline_start_ticks + 1;
            float decline = static_cast<float>(elapsed)
                / static_cast<float>(config.fertility_decline_ticks);
            if (decline > 1.0f) {
                decline = 1.0f;
            }
            const float aged_scale = 1.0f
                - decline * (1.0f - config.elder_fertility_floor);
            scale = fminf(scale, aged_scale);
        }
    }
    if (scale < 0.0f) {
        return 0.0f;
    }
    return scale > 1.0f ? 1.0f : scale;
}

__device__ __forceinline__ void initialize_plan_outputs(
    const FlatOrdinaryBirthPlanBuffers buffers,
    const OrdinaryBirthPlanConfig config
) {
    const int64_t plan_rows = config.max_births >= 0
            && config.max_births <= config.slot_count
        ? config.max_births
        : 0;
    for (int64_t birth = 0; birth < plan_rows; ++birth) {
        buffers.planned_parent_slots[birth] = -1;
        buffers.planned_child_slots[birth] = -1;
        buffers.planned_spawn_positions[birth * 2] = -1;
        buffers.planned_spawn_positions[birth * 2 + 1] = -1;
        buffers.planned_child_energy[birth] = __int_as_float(0x7fc00000);
    }
    for (int64_t decision = 0; decision < config.slot_count; ++decision) {
        buffers.decision_parent_slots[decision] = -1;
        buffers.decision_reason_codes[decision] = -1;
        buffers.decision_candidate_counts[decision] = 0;
        buffers.decision_draw_starts[decision] = 0;
        buffers.decision_draw_ends[decision] = 0;
        buffers.decision_selected_positions[decision * 2] = -1;
        buffers.decision_selected_positions[decision * 2 + 1] = -1;
        buffers.decision_selected_reserved[decision] = false;
    }
    for (int32_t reason = 0; reason < kReasonCount; ++reason) {
        buffers.reason_counts[reason] = 0;
    }
    buffers.planned_count[0] = 0;
    buffers.decision_count[0] = 0;
    buffers.spawn_rng_draws_added[0] = 0;
    buffers.spawn_rng_semantic_draws_added[0] = 0;
    buffers.runner_rng_draws_added[0] = 0;
    buffers.runner_rng_semantic_draws_added[0] = 0;
    buffers.status[0] = kStatusOk;
}

// Exact raw-device transcription of the authoritative Stage2 ordinary-birth
// planner in native/cuda_architecture_b_asexual_birth_plan.cu. Exactly one
// device thread invokes this allocation-free function for one simulation cell;
// the function performs no synchronization and is neutral to Flat/Patch state.
__device__ __forceinline__ void plan_ordinary_births_serial(
    const FlatOrdinaryBirthPlanState state,
    const FlatOrdinaryBirthPlanBuffers buffers,
    const OrdinaryBirthPlanConfig config
) {
    initialize_plan_outputs(buffers, config);

    if (
        !::isfinite(config.parent_energy_share)
        || config.parent_energy_share < 0.0
        || config.parent_energy_share > 1.0
        || !::isfinite(config.minimum_energy_buffer)
        || config.minimum_energy_buffer < 0.0f
        || config.max_births < 0
        || config.max_births > config.slot_count
        || !::isfinite(config.injury_repro_k_injury)
        || config.injury_repro_k_injury < 0.0
        || !::isfinite(config.injury_repro_k_scar)
        || config.injury_repro_k_scar < 0.0
        || !::isfinite(config.injury_birth_fail_scar)
        || config.injury_birth_fail_scar < 0.0
        || !::isfinite(config.injury_birth_fail_injury)
        || config.injury_birth_fail_injury < 0.0
        || config.requires_mate
        || config.injury_gate_active
        || config.conception_gate_active
            != (
                config.injury_repro_k_injury > 0.0
                || config.injury_repro_k_scar > 0.0
            )
        || config.gestation_gate_active
            != (
                config.injury_birth_fail_scar > 0.0
                || config.injury_birth_fail_injury > 0.0
            )
        || config.life_history_mode < 0
        || config.life_history_mode > 1
        || config.maturity_start_ticks < 0
        || config.maturity_rise_ticks < 0
        || config.fertility_decline_start_ticks < 0
        || config.fertility_decline_ticks < 0
        || !::isfinite(config.elder_fertility_floor)
        || config.elder_fertility_floor < 0.0f
        || config.elder_fertility_floor > 1.0f
    ) {
        buffers.status[0] = kStatusInvalidConfig;
        return;
    }

    const int64_t free_queue_size = state.slot_free_queue_count[0];
    const int64_t initial_free_head = state.slot_free_queue_head[0];
    if (
        free_queue_size < 0
        || free_queue_size > config.free_queue_capacity
        || initial_free_head < 0
        || initial_free_head > free_queue_size
        || state.policy_np_semantic_draws[0] < 0
        || state.metrics_np_semantic_draws[0] < 0
    ) {
        buffers.status[0] = kStatusInvalidResidentState;
        return;
    }
    if (
        state.pcg64_buffer_words[0] < 0
        || state.pcg64_buffer_words[0] > 1
        || state.pcg64_buffer_words[1] < 0
        || static_cast<uint64_t>(state.pcg64_buffer_words[1]) > 0xffffffffULL
        || (state.pcg64_state_words[2] & 1ULL) == 0ULL
    ) {
        buffers.status[0] = kStatusInvalidPcg64State;
        return;
    }
    if (
        state.runner_pcg64_buffer_words[0] < 0
        || state.runner_pcg64_buffer_words[0] > 1
        || state.runner_pcg64_buffer_words[1] < 0
        || static_cast<uint64_t>(state.runner_pcg64_buffer_words[1])
            > 0xffffffffULL
        || (state.runner_pcg64_state_words[2] & 1ULL) == 0ULL
    ) {
        buffers.status[0] = kStatusInvalidPcg64State;
        return;
    }

    int64_t alive_rows = 0;
    for (int64_t slot = 0; slot < config.slot_count; ++slot) {
        buffers.planned_repro_cooldown[slot] = state.repro_cooldown[slot];
        if (!state.alive_mask[slot]) {
            continue;
        }
        ++alive_rows;
        const int64_t agent_id = state.slot_to_agent_id[slot];
        if (
            agent_id < 0
            || state.pos_y[slot] < 0
            || state.pos_y[slot] >= config.grid_height
            || state.pos_x[slot] < 0
            || state.pos_x[slot] >= config.grid_width
            || !::isfinite(state.energy[slot])
            || !::isfinite(state.health[slot])
            || !::isfinite(state.injury[slot])
            || state.injury[slot] < 0.0f
            || !::isfinite(state.scar[slot])
            || state.scar[slot] < 0.0f
            || !::isfinite(state.repro_threshold[slot])
            || state.repro_cooldown[slot] < 0
            || (
                config.life_history_mode == 1
                && (
                    state.survival_steps == nullptr
                    || state.survival_steps[slot] < 0
                )
            )
            || (
                config.reproduction_scale_gate_active
                && (
                    state.reproduction_scale == nullptr
                    || !::isfinite(state.reproduction_scale[slot])
                    || state.reproduction_scale[slot] < 0.0f
                    || state.reproduction_scale[slot] > 1.0f
                )
            )
        ) {
            buffers.status[0] = kStatusInvalidResidentState;
            return;
        }
        for (int64_t other = 0; other < slot; ++other) {
            if (
                state.alive_mask[other]
                && state.slot_to_agent_id[other] == agent_id
            ) {
                buffers.status[0] = kStatusInvalidAgentIds;
                return;
            }
        }
    }

    Pcg64State rng{
        state.pcg64_state_words[0],
        state.pcg64_state_words[1],
        state.pcg64_state_words[2],
        state.pcg64_state_words[3],
        static_cast<uint32_t>(state.pcg64_buffer_words[0]),
        static_cast<uint32_t>(state.pcg64_buffer_words[1]),
    };
    Pcg64State runner_rng{
        state.runner_pcg64_state_words[0],
        state.runner_pcg64_state_words[1],
        state.runner_pcg64_state_words[2],
        state.runner_pcg64_state_words[3],
        static_cast<uint32_t>(state.runner_pcg64_buffer_words[0]),
        static_cast<uint32_t>(state.runner_pcg64_buffer_words[1]),
    };
    int64_t free_head = initial_free_head;
    int64_t planned_count = 0;
    int64_t decision_count = 0;
    int64_t spawn_rng_draws = 0;
    int64_t spawn_rng_semantic_draws = 0;
    int64_t runner_rng_draws = 0;
    int64_t runner_rng_semantic_draws = 0;
    int64_t previous_agent_id = kInt64Min;
    int64_t previous_slot = -1;

    for (int64_t order = 0; order < alive_rows; ++order) {
        if (planned_count >= config.max_births) {
            break;
        }

        int64_t parent_slot = -1;
        int64_t parent_agent_id = kInt64Max;
        for (int64_t slot = 0; slot < config.slot_count; ++slot) {
            if (!state.alive_mask[slot]) {
                continue;
            }
            const int64_t agent_id = state.slot_to_agent_id[slot];
            const bool follows_previous = agent_id > previous_agent_id
                || (agent_id == previous_agent_id && slot > previous_slot);
            if (!follows_previous) {
                continue;
            }
            if (
                parent_slot < 0
                || agent_id < parent_agent_id
                || (agent_id == parent_agent_id && slot < parent_slot)
            ) {
                parent_slot = slot;
                parent_agent_id = agent_id;
            }
        }
        if (parent_slot < 0) {
            buffers.status[0] = kStatusInvalidAgentIds;
            return;
        }
        previous_agent_id = parent_agent_id;
        previous_slot = parent_slot;

        const int64_t decision = decision_count++;
        buffers.decision_parent_slots[decision] = parent_slot;
        buffers.decision_draw_starts[decision] = spawn_rng_draws;
        buffers.decision_draw_ends[decision] = spawn_rng_draws;

        int32_t reason = kReasonAccepted;
        if (state.health[parent_slot] <= 0.0f) {
            reason = kReasonHealth;
        } else {
            const int32_t parent_age_ticks = state.survival_steps == nullptr
                ? 0
                : state.survival_steps[parent_slot];
            const float fertility_scale = life_history_fertility_scale_v1(
                parent_age_ticks,
                config
            );
            if (fertility_scale <= 0.0f) {
                reason = kReasonOpportunity;
            } else if (buffers.planned_repro_cooldown[parent_slot] > 0) {
                buffers.planned_repro_cooldown[parent_slot] -= 1;
            }
            if (
                reason == kReasonAccepted
                && buffers.planned_repro_cooldown[parent_slot] != 0
            ) {
                reason = kReasonCooldown;
            } else if (
                reason == kReasonAccepted
                && state.energy[parent_slot]
                    < state.repro_threshold[parent_slot] / fertility_scale
            ) {
                reason = kReasonNoEnergy;
            }
        }

        int32_t candidate_y[4];
        int32_t candidate_x[4];
        int32_t candidate_count = 0;
        int32_t selected_y = -1;
        int32_t selected_x = -1;
        bool selected_reserved = false;

        if (reason == kReasonAccepted) {
            constexpr int32_t kDy[4] = {-1, 1, 0, 0};
            constexpr int32_t kDx[4] = {0, 0, -1, 1};
            const int32_t y = state.pos_y[parent_slot];
            const int32_t x = state.pos_x[parent_slot];
            for (int32_t direction = 0; direction < 4; ++direction) {
                const int32_t ny = y + kDy[direction];
                const int32_t nx = x + kDx[direction];
                if (
                    ny < 0
                    || ny >= config.grid_height
                    || nx < 0
                    || nx >= config.grid_width
                    || state.agent_grid[
                        static_cast<int64_t>(ny) * config.grid_width + nx
                    ] != -1
                ) {
                    continue;
                }
                candidate_y[candidate_count] = ny;
                candidate_x[candidate_count] = nx;
                ++candidate_count;
            }

            if (
                spawn_rng_semantic_draws
                > kInt64Max - candidate_count
            ) {
                buffers.status[0] = kStatusCounterOverflow;
                return;
            }
            spawn_rng_semantic_draws += candidate_count;
            for (int32_t index = candidate_count - 1; index > 0; --index) {
                const int32_t selected = static_cast<int32_t>(
                    pcg64_bounded_uint32_exact(
                        rng,
                        static_cast<uint32_t>(index)
                    )
                );
                ++spawn_rng_draws;
                const int32_t tmp_y = candidate_y[index];
                const int32_t tmp_x = candidate_x[index];
                candidate_y[index] = candidate_y[selected];
                candidate_x[index] = candidate_x[selected];
                candidate_y[selected] = tmp_y;
                candidate_x[selected] = tmp_x;
            }
            if (candidate_count > 0) {
                selected_y = candidate_y[0];
                selected_x = candidate_x[0];
                selected_reserved = selected_position_is_reserved(
                    selected_y,
                    selected_x,
                    buffers.planned_spawn_positions,
                    planned_count
                );
            }
            if (candidate_count == 0 || selected_reserved) {
                reason = kReasonNoSpace;
            }
        }

        if (reason == kReasonAccepted) {
            const double parent_energy = ::isfinite(
                state.policy_energy_observation[parent_slot]
            )
                ? state.policy_energy_observation[parent_slot]
                : static_cast<double>(state.energy[parent_slot]);
            const double parent_share =
                parent_energy * config.parent_energy_share;
            if (
                parent_energy - parent_share
                < static_cast<double>(config.minimum_energy_buffer)
            ) {
                reason = kReasonBuffer;
            }
            if (reason == kReasonAccepted && config.conception_gate_active) {
                if (
                    state.metrics_np_semantic_draws[0]
                    > kInt64Max - runner_rng_semantic_draws - 1
                ) {
                    buffers.status[0] = kStatusCounterOverflow;
                    return;
                }
                const double parent_injury =
                    static_cast<double>(state.injury[parent_slot]);
                const double parent_scar =
                    static_cast<double>(state.scar[parent_slot]);
                const double conception_probability = ::exp(-(
                    config.injury_repro_k_injury * parent_injury
                    + config.injury_repro_k_scar * parent_scar
                ));
                const double conception_draw = pcg64_double01(runner_rng);
                ++runner_rng_draws;
                ++runner_rng_semantic_draws;
                if (conception_draw > conception_probability) {
                    reason = kReasonConception;
                }
            }
            if (reason == kReasonAccepted && config.gestation_gate_active) {
                if (
                    state.metrics_np_semantic_draws[0]
                    > kInt64Max - runner_rng_semantic_draws - 1
                ) {
                    buffers.status[0] = kStatusCounterOverflow;
                    return;
                }
                const double parent_injury =
                    static_cast<double>(state.injury[parent_slot]);
                const double parent_scar =
                    static_cast<double>(state.scar[parent_slot]);
                const double gestation_failure_probability = ::fmin(
                    0.95,
                    config.injury_birth_fail_scar * parent_scar
                        + config.injury_birth_fail_injury * parent_injury
                );
                const double gestation_draw = pcg64_double01(runner_rng);
                ++runner_rng_draws;
                ++runner_rng_semantic_draws;
                if (gestation_draw < gestation_failure_probability) {
                    reason = kReasonGestation;
                }
            }
            if (
                reason == kReasonAccepted
                && config.reproduction_scale_gate_active
            ) {
                const double probability = static_cast<double>(
                    state.reproduction_scale[parent_slot]
                );
                if (probability <= 0.0) {
                    reason = kReasonOpportunity;
                } else if (probability < 1.0) {
                    if (
                        state.metrics_np_semantic_draws[0]
                        > kInt64Max - runner_rng_semantic_draws - 1
                    ) {
                        buffers.status[0] = kStatusCounterOverflow;
                        return;
                    }
                    const double reproduction_draw = pcg64_double01(runner_rng);
                    ++runner_rng_draws;
                    ++runner_rng_semantic_draws;
                    if (reproduction_draw > probability) {
                        reason = kReasonOpportunity;
                    }
                }
            }
            if (reason == kReasonAccepted) {
                int64_t child_slot = -1;
                while (free_head < free_queue_size) {
                    const int64_t candidate_slot =
                        state.slot_free_queue[free_head++];
                    if (
                        candidate_slot >= 0
                        && candidate_slot < config.slot_count
                        && !state.alive_mask[candidate_slot]
                        && state.slot_to_agent_id[candidate_slot] < 0
                        && !child_slot_already_planned(
                            candidate_slot,
                            buffers.planned_child_slots,
                            planned_count
                        )
                    ) {
                        child_slot = candidate_slot;
                        break;
                    }
                }
                if (child_slot < 0) {
                    reason = kReasonNoFreeSlot;
                } else {
                    buffers.planned_parent_slots[planned_count] = parent_slot;
                    buffers.planned_child_slots[planned_count] = child_slot;
                    buffers.planned_spawn_positions[planned_count * 2] =
                        selected_y;
                    buffers.planned_spawn_positions[planned_count * 2 + 1] =
                        selected_x;
                    buffers.planned_child_energy[planned_count] =
                        static_cast<float>(parent_share);
                    ++planned_count;
                }
            }
        }

        buffers.decision_reason_codes[decision] = reason;
        buffers.decision_candidate_counts[decision] = candidate_count;
        buffers.decision_draw_ends[decision] = spawn_rng_draws;
        buffers.decision_selected_positions[decision * 2] = selected_y;
        buffers.decision_selected_positions[decision * 2 + 1] = selected_x;
        buffers.decision_selected_reserved[decision] = selected_reserved;
        if (reason >= 0 && reason < kReasonCount) {
            buffers.reason_counts[reason] += 1;
        }
        if (reason == kReasonNoFreeSlot) {
            break;
        }
    }

    if (
        config.env_policy_alias
        && state.policy_np_semantic_draws[0]
            > kInt64Max - spawn_rng_semantic_draws
    ) {
        buffers.status[0] = kStatusCounterOverflow;
        return;
    }
    if (
        state.metrics_np_semantic_draws[0]
        > kInt64Max - runner_rng_semantic_draws
    ) {
        buffers.status[0] = kStatusCounterOverflow;
        return;
    }

    for (int64_t slot = 0; slot < config.slot_count; ++slot) {
        state.repro_cooldown[slot] = buffers.planned_repro_cooldown[slot];
    }
    state.slot_free_queue_head[0] = free_head;
    state.pcg64_state_words[0] = rng.state_lo;
    state.pcg64_state_words[1] = rng.state_hi;
    state.pcg64_state_words[2] = rng.inc_lo;
    state.pcg64_state_words[3] = rng.inc_hi;
    state.pcg64_buffer_words[0] = static_cast<int64_t>(rng.has_uint32);
    state.pcg64_buffer_words[1] = static_cast<int64_t>(rng.uinteger);
    if (config.env_policy_alias) {
        state.policy_np_semantic_draws[0] += spawn_rng_semantic_draws;
    }
    state.runner_pcg64_state_words[0] = runner_rng.state_lo;
    state.runner_pcg64_state_words[1] = runner_rng.state_hi;
    state.runner_pcg64_state_words[2] = runner_rng.inc_lo;
    state.runner_pcg64_state_words[3] = runner_rng.inc_hi;
    state.runner_pcg64_buffer_words[0] =
        static_cast<int64_t>(runner_rng.has_uint32);
    state.runner_pcg64_buffer_words[1] =
        static_cast<int64_t>(runner_rng.uinteger);
    state.metrics_np_semantic_draws[0] += runner_rng_semantic_draws;
    buffers.planned_count[0] = planned_count;
    buffers.decision_count[0] = decision_count;
    buffers.spawn_rng_draws_added[0] = spawn_rng_draws;
    buffers.spawn_rng_semantic_draws_added[0] = spawn_rng_semantic_draws;
    buffers.runner_rng_draws_added[0] = runner_rng_draws;
    buffers.runner_rng_semantic_draws_added[0] =
        runner_rng_semantic_draws;
    buffers.status[0] = kStatusOk;
}

}  // namespace eiel::cuda_architecture_b::persistent_birth_plan

#endif  // EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_BIRTH_PLAN_CUH_
