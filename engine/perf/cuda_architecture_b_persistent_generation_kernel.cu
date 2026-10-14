#include "cuda_architecture_b_persistent_generation.hpp"
#include "cuda_architecture_b_persistent_active_mechanics.cuh"
#include "cuda_architecture_b_persistent_birth_action_history.cuh"
#include "cuda_architecture_b_persistent_birth_drives.cuh"
#include "cuda_architecture_b_persistent_birth_events.cuh"
#include "cuda_architecture_b_persistent_birth_identity.cuh"
#include "cuda_architecture_b_persistent_birth_plan.cuh"
#include "cuda_architecture_b_persistent_birth_policy.cuh"
#include "cuda_architecture_b_persistent_birth_scalar_genome.cuh"
#include "cuda_architecture_b_persistent_chemistry.cuh"
#include "cuda_architecture_b_persistent_food_spawn.cuh"
#include "cuda_architecture_b_persistent_lifecycle_events.cuh"
#include "cuda_architecture_b_persistent_newborn_state.cuh"
#include "cuda_architecture_b_persistent_stochastic_policy.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace eiel::cuda_architecture_b::persistent_generation {
namespace {

namespace torch_rng_pack =
    eiel::cuda_architecture_b::torch_rng_state_pack_v1;

constexpr int64_t kMaxHiddenSize = 64;
constexpr int64_t kMaxObservationSize = 128;
constexpr int64_t kMaxActionCount = 16;
constexpr int32_t kEmpty = 0;
constexpr int32_t kFood = 1;
constexpr int32_t kGatherAction = 4;
constexpr int32_t kRestAction = 5;
constexpr int32_t kAttackAction = 6;
constexpr int32_t kLifecycleOk = 0;
constexpr int32_t kLifecycleInvalidPendingCount = 1;
constexpr int32_t kLifecyclePendingCapacityExceeded = 2;

__device__ __forceinline__ void record_status(
    int32_t* status,
    const int64_t cell,
    const int32_t code
) {
    atomicCAS(status + cell, 0, code);
}

__device__ __forceinline__ float action_cost(
    const int32_t action,
    const double move_cost,
    const double gather_cost,
    const double rest_cost,
    const double attack_cost,
    const double action_cost_mult,
    const double rest_cost_mult
) {
    float cost = static_cast<float>(move_cost);
    if (action == kGatherAction) {
        cost = static_cast<float>(gather_cost);
    } else if (action == kRestAction) {
        cost = static_cast<float>(rest_cost);
    } else if (action == kAttackAction) {
        cost = static_cast<float>(attack_cost);
    }
    if (action_cost_mult != 1.0) {
        cost = static_cast<float>(static_cast<double>(cost) * action_cost_mult);
    }
    if (rest_cost_mult != 1.0 && action == kRestAction) {
        cost = static_cast<float>(static_cast<double>(cost) * rest_cost_mult);
    }
    return cost;
}

__device__ __forceinline__ bool is_death_candidate_v3(
    const int64_t slot,
    const int64_t* __restrict__ slot_to_agent_id,
    const bool* __restrict__ alive_mask,
    const float* __restrict__ energy,
    const float* __restrict__ health
) {
    return alive_mask[slot]
        && slot_to_agent_id[slot] >= 0
        && (health[slot] <= 0.0f || energy[slot] <= 0.0f);
}

__device__ __forceinline__ int32_t lifecycle_death_cull_cell_v3(
    int64_t* __restrict__ active_agent_ids,
    int64_t* __restrict__ active_count_state,
    int64_t* __restrict__ slot_to_agent_id,
    int32_t* __restrict__ agent_id_to_slot,
    bool* __restrict__ alive_mask,
    int32_t* __restrict__ pos_y,
    int32_t* __restrict__ pos_x,
    int32_t* __restrict__ agent_grid,
    float* __restrict__ energy,
    float* __restrict__ health,
    double* __restrict__ policy_energy_observation,
    int32_t* __restrict__ rest_streak,
    int64_t* __restrict__ action_cache_token,
    int32_t* __restrict__ action_cache_cursor,
    int32_t* __restrict__ action_cache_next,
    int64_t* __restrict__ pending_free_agent_ids,
    int64_t* __restrict__ pending_free_slots,
    int64_t* __restrict__ pending_free_count,
    int64_t* __restrict__ death_agent_ids,
    int64_t* __restrict__ death_slots,
    int32_t* __restrict__ death_reason_codes,
    int64_t* __restrict__ death_count,
    int64_t* __restrict__ survivor_agent_ids,
    int64_t* __restrict__ survivor_count,
    const int64_t slot_count,
    const int64_t active_capacity,
    const int64_t agent_id_capacity,
    const int64_t grid_height,
    const int64_t grid_width,
    const int64_t pending_capacity
) {
    const int64_t pending_before = pending_free_count[0];
    if (pending_before < 0 || pending_before > pending_capacity) {
        return kLifecycleInvalidPendingCount;
    }

    int64_t candidate_count = 0;
    for (int64_t slot = 0; slot < slot_count; ++slot) {
        candidate_count += is_death_candidate_v3(
            slot,
            slot_to_agent_id,
            alive_mask,
            energy,
            health
        ) ? 1 : 0;
    }
    if (pending_before + candidate_count > pending_capacity) {
        return kLifecyclePendingCapacityExceeded;
    }

    for (int64_t death_index = 0; death_index < candidate_count; ++death_index) {
        int64_t selected_slot = -1;
        int64_t selected_agent_id = INT64_MAX;
        for (int64_t slot = 0; slot < slot_count; ++slot) {
            if (!is_death_candidate_v3(
                    slot,
                    slot_to_agent_id,
                    alive_mask,
                    energy,
                    health
                )) {
                continue;
            }
            const int64_t agent_id = slot_to_agent_id[slot];
            if (
                agent_id < selected_agent_id
                || (agent_id == selected_agent_id && slot < selected_slot)
            ) {
                selected_agent_id = agent_id;
                selected_slot = slot;
            }
        }

        const int32_t reason_code = health[selected_slot] <= 0.0f ? 2 : 1;
        const int32_t y = pos_y[selected_slot];
        const int32_t x = pos_x[selected_slot];
        if (
            y >= 0 && y < grid_height
            && x >= 0 && x < grid_width
            && agent_grid[static_cast<int64_t>(y) * grid_width + x]
                == static_cast<int32_t>(selected_agent_id)
        ) {
            agent_grid[static_cast<int64_t>(y) * grid_width + x] = -1;
        }
        if (selected_agent_id >= 0 && selected_agent_id < agent_id_capacity) {
            agent_id_to_slot[selected_agent_id] = -1;
        }

        alive_mask[selected_slot] = false;
        slot_to_agent_id[selected_slot] = -1;
        pos_y[selected_slot] = -1;
        pos_x[selected_slot] = -1;
        energy[selected_slot] = 0.0f;
        health[selected_slot] = 0.0f;
        policy_energy_observation[selected_slot] = __longlong_as_double(
            static_cast<long long>(0x7ff8000000000000ULL)
        );
        rest_streak[selected_slot] = 0;
        action_cache_token[selected_slot] = -1;
        action_cache_cursor[selected_slot] = -1;
        action_cache_next[selected_slot] = -1;

        const int64_t pending_index = pending_before + death_index;
        pending_free_agent_ids[pending_index] = selected_agent_id;
        pending_free_slots[pending_index] = selected_slot;
        death_agent_ids[death_index] = selected_agent_id;
        death_slots[death_index] = selected_slot;
        death_reason_codes[death_index] = reason_code;
    }
    pending_free_count[0] = pending_before + candidate_count;
    death_count[0] = candidate_count;

    for (int64_t index = 0; index < active_capacity; ++index) {
        active_agent_ids[index] = -1;
    }
    int64_t remaining = 0;
    int64_t previous_agent_id = INT64_MIN;
    int64_t previous_slot = -1;
    while (true) {
        int64_t selected_slot = -1;
        int64_t selected_agent_id = INT64_MAX;
        for (int64_t slot = 0; slot < slot_count; ++slot) {
            if (!alive_mask[slot] || slot_to_agent_id[slot] < 0) {
                continue;
            }
            const int64_t agent_id = slot_to_agent_id[slot];
            const bool follows_previous = agent_id > previous_agent_id
                || (agent_id == previous_agent_id && slot > previous_slot);
            if (
                follows_previous
                && (
                    agent_id < selected_agent_id
                    || (agent_id == selected_agent_id && slot < selected_slot)
                )
            ) {
                selected_agent_id = agent_id;
                selected_slot = slot;
            }
        }
        if (selected_slot < 0) {
            break;
        }
        if (remaining >= active_capacity) {
            return kLifecyclePendingCapacityExceeded;
        }
        active_agent_ids[remaining] = selected_agent_id;
        survivor_agent_ids[remaining] = selected_agent_id;
        previous_agent_id = selected_agent_id;
        previous_slot = selected_slot;
        ++remaining;
    }
    active_count_state[0] = remaining;
    survivor_count[0] = remaining;
    return kLifecycleOk;
}

__device__ __forceinline__ int32_t release_residual_free_slots_v4(
    const int64_t* __restrict__ slot_to_agent_id,
    const bool* __restrict__ alive_mask,
    int64_t* __restrict__ slot_free_queue,
    int64_t* __restrict__ slot_free_queue_count,
    const int64_t* __restrict__ residual_slots,
    const int64_t residual_count,
    int64_t* __restrict__ appended_count,
    const int64_t slot_count,
    const int64_t queue_capacity
) {
    constexpr int32_t kInvalidState = 30;
    constexpr int32_t kCapacityExceeded = 31;
    constexpr int32_t kContradiction = 32;
    const int64_t queue_before = slot_free_queue_count[0];
    if (
        queue_before < 0 || queue_before > queue_capacity
        || residual_count < 0 || residual_count > slot_count
    ) {
        return kInvalidState;
    }
    if (queue_before + residual_count > queue_capacity) {
        return kCapacityExceeded;
    }
    for (int64_t row = 0; row < residual_count; ++row) {
        const int64_t slot = residual_slots[row];
        if (
            slot < 0 || slot >= slot_count || alive_mask[slot]
            || slot_to_agent_id[slot] >= 0
        ) {
            return kContradiction;
        }
        for (int64_t queued = 0; queued < queue_before; ++queued) {
            if (slot_free_queue[queued] == slot) {
                return kContradiction;
            }
        }
        for (int64_t prior = 0; prior < row; ++prior) {
            if (residual_slots[prior] == slot) {
                return kContradiction;
            }
        }
    }
    for (int64_t row = 0; row < residual_count; ++row) {
        slot_free_queue[queue_before + row] = residual_slots[row];
    }
    slot_free_queue_count[0] = queue_before + residual_count;
    appended_count[0] = residual_count;
    return kLifecycleOk;
}

__device__ __forceinline__ void append_death_event_class_v4(
    const FlatEventState& event_state,
    const int64_t* __restrict__ death_agent_ids,
    const int64_t* __restrict__ death_slots,
    const int32_t* __restrict__ death_reason_codes,
    const int64_t* __restrict__ death_count,
    const int64_t ticks,
    const int64_t capacity,
    const int64_t cell,
    const int64_t tick,
    const int64_t generation,
    const int64_t absolute_tick
) {
    using namespace persistent_lifecycle_events;
    const FlatEventTapeColumns columns{
        {event_state.event_type, event_state.capacity, 1},
        {event_state.event_generation, event_state.capacity, 1},
        {event_state.event_tick, event_state.capacity, 1},
        {event_state.event_actor_id, event_state.capacity, 1},
        {event_state.event_target_id, event_state.capacity, 1},
        {event_state.event_value, event_state.capacity, 1},
        {event_state.typed_event_type, event_state.capacity, 1},
        {event_state.typed_generation, event_state.capacity, 1},
        {event_state.typed_tick, event_state.capacity, 1},
        {event_state.typed_sequence, event_state.capacity, 1},
        {event_state.typed_actor_id, event_state.capacity, 1},
        {event_state.typed_target_id, event_state.capacity, 1},
        {event_state.typed_actor_slot, event_state.capacity, 1},
        {event_state.typed_target_slot, event_state.capacity, 1},
        {event_state.typed_parent_id, event_state.capacity, 1},
        {event_state.typed_child_id, event_state.capacity, 1},
        {event_state.typed_founder_id, event_state.capacity, 1},
        {event_state.typed_lineage_depth, event_state.capacity, 1},
        {event_state.typed_subtype, event_state.capacity, 1},
        {event_state.typed_reason, event_state.capacity, 1},
        {event_state.typed_success, event_state.capacity, 1},
        {event_state.typed_value, event_state.capacity, 1},
    };
    const FlatEventTapeState tape{
        {event_state.active_count, 1},
        {event_state.status, 1},
        event_state.capacity,
    };
    const FlatDeathRows deaths{
        {death_agent_ids + tick * capacity, ticks * capacity, 1},
        {death_slots + tick * capacity, ticks * capacity, 1},
        {death_reason_codes + tick * capacity, ticks * capacity, 1},
        {death_count + tick, ticks},
        capacity,
    };
    const FlatLineageRows lineage{
        {event_state.lineage_founder_id, capacity, 1},
        {event_state.lineage_depth, capacity, 1},
        capacity,
    };
    append_death_lifecycle_events(
        columns,
        tape,
        deaths,
        lineage,
        cell,
        generation,
        absolute_tick
    );
}

__device__ __forceinline__ void accumulate_action_terminal_v4(
    const FlatEventState& event_state,
    const int64_t* __restrict__ death_agent_ids,
    const int64_t* __restrict__ death_slots,
    const int32_t* __restrict__ death_reason_codes,
    const int64_t* __restrict__ death_count,
    const int32_t* __restrict__ survival_steps,
    const int64_t ticks,
    const int64_t capacity,
    const int64_t agent_id_capacity,
    const int64_t cell,
    const int64_t tick
) {
    using namespace persistent_lifecycle_events;
    const FlatDeathRows deaths{
        {death_agent_ids + tick * capacity, ticks * capacity, 1},
        {death_slots + tick * capacity, ticks * capacity, 1},
        {death_reason_codes + tick * capacity, ticks * capacity, 1},
        {death_count + tick, ticks},
        capacity,
    };
    const FlatActionTerminalReporting reporting{
        {event_state.initial_id_to_row, agent_id_capacity, 1},
        {survival_steps, capacity, 1},
        {event_state.terminal_survival, event_state.reporting_rows, 1},
        {event_state.terminal_rows, event_state.reporting_rows, 1},
        agent_id_capacity,
        capacity,
        event_state.reporting_rows,
    };
    accumulate_action_terminal_reporting(deaths, reporting, cell);
}

__device__ __forceinline__ int32_t execute_ordinary_birth_microcycle_v7(
    int64_t* __restrict__ active_agent_ids,
    int64_t* __restrict__ active_count_state,
    int32_t* __restrict__ ordered_slots_out,
    int64_t* __restrict__ slot_to_agent_id,
    int32_t* __restrict__ agent_id_to_slot,
    int64_t* __restrict__ slot_tokens,
    bool* __restrict__ alive_mask,
    int32_t* __restrict__ pos_y,
    int32_t* __restrict__ pos_x,
    int32_t* __restrict__ agent_grid,
    float* __restrict__ energy,
    float* __restrict__ health,
    float* __restrict__ injury,
    double* __restrict__ policy_energy_observation,
    int32_t* __restrict__ rest_streak,
    int32_t* __restrict__ action_cache_actions,
    int64_t* __restrict__ action_cache_token,
    int32_t* __restrict__ action_cache_cursor,
    int32_t* __restrict__ action_cache_next,
    int32_t* __restrict__ last_action,
    int32_t* __restrict__ last_last_action,
    int32_t* __restrict__ action_switches,
    int32_t* __restrict__ action_oscillations,
    int32_t* __restrict__ episode0_switches,
    int32_t* __restrict__ episode0_oscillations,
    int32_t* __restrict__ survival_steps,
    float* __restrict__ mlp_pack,
    float* __restrict__ drives,
    float* __restrict__ attack_bias,
    float* __restrict__ action_temperature,
    float* __restrict__ drive_logit_gain,
    float* __restrict__ energy_collected,
    double* __restrict__ energy_collected_exact,
    const FlatFullTickState full_tick_state,
    const FlatEventState event_state,
    const FlatOrdinaryBirthState birth_state,
    const FlatOrdinaryBirthOutputs birth_outputs,
    const FlatOrdinaryBirthConfig birth_config,
    const FlatMechanicsConfig mechanics_config,
    const int64_t cell,
    const int64_t tick,
    const int64_t ticks,
    const int64_t capacity,
    const int64_t active_capacity,
    const int64_t agent_id_capacity,
    const int64_t height,
    const int64_t width,
    const int64_t cache_capacity,
    const int64_t drive_width,
    const int64_t packed_width,
    const int64_t chemistry_width,
    const float max_energy,
    const float start_health,
    const int32_t external_gate_status
) {
    using namespace persistent_birth_action_history;
    using namespace persistent_birth_drives;
    using namespace persistent_birth_events;
    using namespace persistent_birth_identity;
    using namespace persistent_birth_plan;
    using namespace persistent_birth_policy;
    using namespace persistent_birth_scalar_genome;
    using namespace persistent_newborn_state;

    const int64_t cell_capacity_offset = cell * capacity;
    const int64_t cell_active_offset = cell * active_capacity;
    const int64_t cell_id_map_offset = cell * agent_id_capacity;
    const int64_t cell_grid_offset = cell * height * width;
    const int64_t cell_tick = cell * ticks + tick;
    const int64_t tick_capacity_offset = cell_tick * capacity;
    const int64_t birth_capacity = birth_config.max_births_per_tick;
    const int64_t tick_birth_offset = cell_tick * birth_capacity;
    const int64_t absolute_tick = mechanics_config.start_tick + tick;

    birth_outputs.combined_status[cell_tick] = 0;
    if (external_gate_status != 0) {
        birth_outputs.combined_status[cell_tick] = 4;
        return 4;
    }

    birth_outputs.slot_free_queue_head[cell] = 0;
    for (int64_t word = 0; word < 4; ++word) {
        birth_outputs.planner_pcg64_state_before[cell_tick * 4 + word] =
            full_tick_state.shared_env_policy_pcg64_state_words[cell * 4 + word];
    }
    for (int64_t word = 0; word < 2; ++word) {
        birth_outputs.planner_pcg64_buffer_before[cell_tick * 2 + word] =
            full_tick_state.shared_env_policy_pcg64_buffer_words[cell * 2 + word];
    }

    const FlatOrdinaryBirthPlanState plan_state{
        slot_to_agent_id + cell_capacity_offset,
        alive_mask + cell_capacity_offset,
        birth_state.policy_use_gru + cell_capacity_offset,
        pos_y + cell_capacity_offset,
        pos_x + cell_capacity_offset,
        agent_grid + cell_grid_offset,
        energy + cell_capacity_offset,
        policy_energy_observation + cell_capacity_offset,
        health + cell_capacity_offset,
        injury + cell_capacity_offset,
        birth_state.scar + cell_capacity_offset,
        birth_state.repro_cooldown + cell_capacity_offset,
        birth_state.repro_threshold + cell_capacity_offset,
        full_tick_state.slot_free_queue + cell_capacity_offset,
        full_tick_state.slot_free_queue_count + cell,
        birth_outputs.slot_free_queue_head + cell,
        full_tick_state.shared_env_policy_pcg64_state_words + cell * 4,
        full_tick_state.shared_env_policy_pcg64_buffer_words + cell * 2,
        full_tick_state.policy_np_semantic_draws + cell,
        full_tick_state.runner_pcg64_state_words + cell * 4,
        full_tick_state.runner_pcg64_buffer_words + cell * 2,
        full_tick_state.metrics_np_semantic_draws + cell,
        nullptr,
        survival_steps + cell_capacity_offset,
    };
    const FlatOrdinaryBirthPlanBuffers plan_buffers{
        birth_outputs.planned_repro_cooldown + cell_capacity_offset,
        birth_outputs.planned_parent_slots + tick_birth_offset,
        birth_outputs.planned_child_slots + tick_birth_offset,
        birth_outputs.planned_spawn_positions + tick_birth_offset * 2,
        birth_outputs.planned_child_energy + tick_birth_offset,
        birth_outputs.decision_parent_slots + tick_capacity_offset,
        birth_outputs.decision_reason_codes + tick_capacity_offset,
        birth_outputs.decision_candidate_counts + tick_capacity_offset,
        birth_outputs.decision_draw_starts + tick_capacity_offset,
        birth_outputs.decision_draw_ends + tick_capacity_offset,
        birth_outputs.decision_selected_positions + tick_capacity_offset * 2,
        birth_outputs.decision_selected_reserved + tick_capacity_offset,
        birth_outputs.reason_counts + cell_tick * kReasonCount,
        birth_outputs.planned_count + cell_tick,
        birth_outputs.decision_count + cell_tick,
        birth_outputs.spawn_rng_draws + cell_tick,
        birth_outputs.spawn_rng_semantic_draws + cell_tick,
        birth_outputs.runner_rng_draws + cell_tick,
        birth_outputs.runner_rng_semantic_draws + cell_tick,
        birth_outputs.plan_status + cell_tick,
    };
    plan_ordinary_births_serial(
        plan_state,
        plan_buffers,
        {
            capacity,
            height,
            width,
            capacity,
            birth_capacity,
            birth_config.parent_energy_share,
            birth_config.minimum_energy_buffer,
            birth_config.injury_repro_k_injury,
            birth_config.injury_repro_k_scar,
            birth_config.injury_birth_fail_scar,
            birth_config.injury_birth_fail_injury,
            birth_config.env_policy_alias,
            birth_config.requires_mate,
            false,
            birth_config.injury_repro_k_injury > 0.0
                || birth_config.injury_repro_k_scar > 0.0,
            birth_config.injury_birth_fail_scar > 0.0
                || birth_config.injury_birth_fail_injury > 0.0,
            false,
            0,
            0,
            0,
            0,
            0,
            1.0f,
        }
    );
    if (birth_outputs.plan_status[cell_tick] != 0) {
        birth_outputs.combined_status[cell_tick] = 1;
        return 1;
    }

    commit_ordinary_birth_identity_serial(
        {
            slot_to_agent_id + cell_capacity_offset,
            agent_id_to_slot + cell_id_map_offset,
            slot_tokens + cell_capacity_offset,
            alive_mask + cell_capacity_offset,
            pos_y + cell_capacity_offset,
            pos_x + cell_capacity_offset,
            agent_grid + cell_grid_offset,
            energy + cell_capacity_offset,
            health + cell_capacity_offset,
            injury + cell_capacity_offset,
            birth_state.scar + cell_capacity_offset,
            full_tick_state.age + cell_capacity_offset,
            birth_state.repro_cooldown + cell_capacity_offset,
            birth_state.repro_threshold + cell_capacity_offset,
            birth_state.offspring_count + cell_capacity_offset,
            event_state.lineage_founder_id + cell_capacity_offset,
            event_state.lineage_depth + cell_capacity_offset,
            birth_state.lineage_birth_generation + cell_capacity_offset,
            birth_state.lineage_birth_tick + cell_capacity_offset,
            birth_state.patch_habitat_lineage_id + cell_capacity_offset,
            policy_energy_observation + cell_capacity_offset,
            survival_steps + cell_capacity_offset,
            rest_streak + cell_capacity_offset,
            action_cache_token + cell_capacity_offset,
            action_cache_cursor + cell_capacity_offset,
            action_cache_next + cell_capacity_offset,
            full_tick_state.slot_free_queue + cell_capacity_offset,
            full_tick_state.slot_free_queue_count + cell,
            birth_state.population_next_id + cell,
        },
        {
            birth_outputs.planned_parent_slots + tick_birth_offset,
            birth_outputs.planned_spawn_positions + tick_birth_offset * 2,
            birth_outputs.planned_count + cell_tick,
        },
        {
            birth_outputs.committed_parent_agent_ids + tick_birth_offset,
            birth_outputs.committed_child_agent_ids + tick_birth_offset,
            birth_outputs.committed_child_slots + tick_birth_offset,
            birth_outputs.identity_committed_count + cell_tick,
            active_agent_ids + cell_active_offset,
            ordered_slots_out + tick_capacity_offset,
            active_count_state + cell,
            birth_outputs.identity_status + cell_tick,
        },
        {
            birth_capacity,
            capacity,
            agent_id_capacity,
            height,
            width,
            capacity,
            birth_config.parent_energy_share,
            static_cast<double>(birth_config.minimum_energy_buffer),
            start_health,
            birth_config.cooldown_steps,
            static_cast<int32_t>(mechanics_config.generation),
            static_cast<int32_t>(absolute_tick),
        }
    );

    int32_t action_history_status = 0;
    reset_ordinary_newborn_action_history_serial(
        {
            last_action + cell_capacity_offset,
            last_last_action + cell_capacity_offset,
            action_switches + cell_capacity_offset,
            action_oscillations + cell_capacity_offset,
            episode0_switches + cell_capacity_offset,
            episode0_oscillations + cell_capacity_offset,
        },
        {
            birth_outputs.committed_child_slots + tick_birth_offset,
            birth_outputs.identity_committed_count + cell_tick,
            birth_outputs.identity_status + cell_tick,
        },
        {&action_history_status},
        {birth_capacity, capacity}
    );
    if (
        birth_outputs.identity_status[cell_tick] != 0
        || action_history_status != 0
    ) {
        birth_outputs.combined_status[cell_tick] = 2;
        return 2;
    }

    commit_ordinary_fixed_mlp_child_scalar_genome_serial(
        {
            slot_to_agent_id + cell_capacity_offset,
            alive_mask + cell_capacity_offset,
            birth_state.genome_hidden_size + cell_capacity_offset,
            birth_state.genome_memory_dim + cell_capacity_offset,
            birth_state.genome_sensor_channel_mask + cell_capacity_offset,
            birth_state.genome_sensor_radius + cell_capacity_offset,
            birth_state.genome_introspection_channel_mask + cell_capacity_offset,
            birth_state.genome_self_regulation_control_mask + cell_capacity_offset,
            birth_state.genome_self_modification_target_mask + cell_capacity_offset,
            birth_state.genome_directed_inheritance_channel_mask + cell_capacity_offset,
            birth_state.policy_mem_dim + cell_capacity_offset,
            birth_state.policy_use_gru + cell_capacity_offset,
            birth_state.genome_exploration_bias + cell_capacity_offset,
            birth_state.genome_aggression_bias + cell_capacity_offset,
            birth_state.genome_mutation_rate + cell_capacity_offset,
            birth_state.genome_reproduction_threshold + cell_capacity_offset,
            birth_state.genome_action_temperature + cell_capacity_offset,
            birth_state.genome_drive_inertia + cell_capacity_offset,
            birth_state.genome_drive_noise + cell_capacity_offset,
            birth_state.genome_drive_logit_gain + cell_capacity_offset,
            birth_state.genome_pred_lr + cell_capacity_offset,
            birth_state.genome_drive_sp_hunger + cell_capacity_offset,
            birth_state.genome_drive_sp_pain + cell_capacity_offset,
            birth_state.genome_drive_sp_fatigue + cell_capacity_offset,
            birth_state.genome_drive_sp_fear + cell_capacity_offset,
            birth_state.genome_exp_prior_strength + cell_capacity_offset,
            birth_state.genome_exp_prior_noise + cell_capacity_offset,
            birth_state.genome_drive_birth_noise + cell_capacity_offset,
            birth_state.repro_threshold + cell_capacity_offset,
            full_tick_state.gru_upkeep_cost + cell_capacity_offset,
            attack_bias + cell_capacity_offset,
            action_temperature + cell_capacity_offset,
            drive_logit_gain + cell_capacity_offset,
            full_tick_state.drive_inertia + cell_capacity_offset,
            full_tick_state.drive_noise + cell_capacity_offset,
            birth_state.pred_lr + cell_capacity_offset,
        },
        {
            birth_outputs.planned_parent_slots + tick_birth_offset,
            birth_outputs.committed_child_slots + tick_birth_offset,
            birth_outputs.identity_committed_count + cell_tick,
        },
        {
            birth_state.python_random_words + cell * 624,
            birth_state.python_random_index + cell,
            birth_state.python_random_has_gauss + cell,
            birth_state.python_random_gauss_next + cell,
            birth_state.python_random_gauss_primary_tape
                + cell * birth_config.gauss_tape_capacity,
            birth_state.python_random_gauss_secondary_tape
                + cell * birth_config.gauss_tape_capacity,
            birth_state.python_random_raw_draws + cell,
            birth_state.mutation_py_semantic_draws + cell,
        },
        {
            birth_outputs.scalar_committed_count + cell_tick,
            birth_outputs.scalar_semantic_draws + cell_tick,
            birth_outputs.scalar_status + cell_tick,
        },
        {
            birth_capacity,
            capacity,
            birth_config.gauss_tape_capacity,
            birth_config.sensor_radius_max,
        }
    );
    if (birth_outputs.scalar_status[cell_tick] != 0) {
        birth_outputs.combined_status[cell_tick] = 2;
        return 2;
    }

    initialize_ordinary_newborn_drives_serial(
        {
            slot_to_agent_id + cell_capacity_offset,
            alive_mask + cell_capacity_offset,
            birth_state.policy_use_gru + cell_capacity_offset,
            policy_energy_observation + cell_capacity_offset,
            birth_state.genome_drive_sp_hunger + cell_capacity_offset,
            birth_state.genome_drive_sp_pain + cell_capacity_offset,
            birth_state.genome_drive_sp_fatigue + cell_capacity_offset,
            birth_state.genome_drive_sp_fear + cell_capacity_offset,
            birth_state.genome_exp_prior_noise + cell_capacity_offset,
            birth_state.genome_drive_birth_noise + cell_capacity_offset,
        },
        {
            birth_outputs.planned_parent_slots + tick_birth_offset,
            birth_outputs.committed_child_slots + tick_birth_offset,
            birth_outputs.identity_committed_count + cell_tick,
        },
        {
            birth_state.population_pcg64_state_words + cell * 4,
            birth_state.population_pcg64_buffer_words + cell * 2,
            birth_state.mutation_np_semantic_draws + cell,
        },
        {
            drives + cell_capacity_offset * drive_width,
            birth_state.birth_reset_shock + cell_capacity_offset,
            birth_outputs.drive_committed_count + cell_tick,
            birth_outputs.birth_noise_rows_consumed + cell_tick,
            birth_outputs.exp_prior_rows_consumed + cell_tick,
            birth_outputs.standard_normal_rows + cell_tick,
            birth_outputs.drive_semantic_draws + cell_tick,
            birth_outputs.drive_status + cell_tick,
        },
        {capacity, birth_capacity, drive_width, max_energy}
    );
    initialize_newborn_chemistry_serial(
        {
            slot_to_agent_id + cell_capacity_offset,
            alive_mask + cell_capacity_offset,
            policy_energy_observation + cell_capacity_offset,
            full_tick_state.chemistry + cell_capacity_offset * chemistry_width,
        },
        {
            birth_outputs.committed_child_slots + tick_birth_offset,
            birth_outputs.identity_committed_count + cell_tick,
        },
        {birth_outputs.chemistry_status + cell_tick},
        {capacity, birth_capacity, chemistry_width, max_energy}
    );
    if (
        birth_outputs.drive_status[cell_tick] != 0
        || birth_outputs.chemistry_status[cell_tick] != 0
    ) {
        birth_outputs.combined_status[cell_tick] = 2;
        return 2;
    }

    inherit_ordinary_fixed_mlp_child_policy_serial(
        {
            slot_to_agent_id + cell_capacity_offset,
            alive_mask + cell_capacity_offset,
            mlp_pack + cell_capacity_offset * packed_width,
            birth_state.hmem + cell_capacity_offset * birth_config.hmem_width,
            birth_state.policy_use_gru + cell_capacity_offset,
            birth_state.mutation_torch_cpu_state
                + cell * birth_config.torch_state_bytes,
            birth_state.mutation_torch_cpu_semantic_draws + cell,
        },
        {
            birth_outputs.planned_parent_slots + tick_birth_offset,
            birth_outputs.committed_child_slots + tick_birth_offset,
            birth_outputs.identity_committed_count + cell_tick,
        },
        {
            birth_state.policy_birth_delta_tape
                + cell * birth_config.policy_tape_capacity * packed_width,
            birth_state.policy_birth_mutation_mask_tape
                + cell * birth_config.policy_tape_capacity * packed_width,
            birth_state.policy_birth_post_rng_state_packed_tape
                + cell * birth_config.policy_tape_capacity
                    * torch_rng_pack::kPackedStateBytes,
            birth_state.policy_birth_semantic_draws_tape
                + cell * birth_config.policy_tape_capacity,
            birth_state.policy_birth_tape_cursor + cell,
        },
        {
            birth_outputs.policy_committed_count + cell_tick,
            birth_outputs.policy_semantic_draws + cell_tick,
            birth_outputs.policy_mutation_events + cell_tick,
            birth_outputs.policy_mutated_elements + cell_tick,
            birth_outputs.policy_mutation_delta_total + cell_tick,
            birth_outputs.policy_mutation_occurred + tick_birth_offset,
            birth_outputs.policy_mutation_delta + tick_birth_offset,
            birth_outputs.policy_status + cell_tick,
        },
        {
            birth_capacity,
            capacity,
            packed_width,
            birth_config.hmem_width,
            birth_config.torch_state_bytes,
            birth_config.policy_tape_capacity,
        }
    );
    if (birth_outputs.policy_status[cell_tick] != 0) {
        birth_outputs.combined_status[cell_tick] = 2;
        return 2;
    }

    const persistent_lifecycle_events::FlatEventTapeColumns columns{
        {event_state.event_type, event_state.capacity, 1},
        {event_state.event_generation, event_state.capacity, 1},
        {event_state.event_tick, event_state.capacity, 1},
        {event_state.event_actor_id, event_state.capacity, 1},
        {event_state.event_target_id, event_state.capacity, 1},
        {event_state.event_value, event_state.capacity, 1},
        {event_state.typed_event_type, event_state.capacity, 1},
        {event_state.typed_generation, event_state.capacity, 1},
        {event_state.typed_tick, event_state.capacity, 1},
        {event_state.typed_sequence, event_state.capacity, 1},
        {event_state.typed_actor_id, event_state.capacity, 1},
        {event_state.typed_target_id, event_state.capacity, 1},
        {event_state.typed_actor_slot, event_state.capacity, 1},
        {event_state.typed_target_slot, event_state.capacity, 1},
        {event_state.typed_parent_id, event_state.capacity, 1},
        {event_state.typed_child_id, event_state.capacity, 1},
        {event_state.typed_founder_id, event_state.capacity, 1},
        {event_state.typed_lineage_depth, event_state.capacity, 1},
        {event_state.typed_subtype, event_state.capacity, 1},
        {event_state.typed_reason, event_state.capacity, 1},
        {event_state.typed_success, event_state.capacity, 1},
        {event_state.typed_value, event_state.capacity, 1},
    };
    const persistent_lifecycle_events::FlatEventTapeState tape{
        {event_state.active_count, 1},
        {event_state.status, 1},
        event_state.capacity,
    };
    append_ordinary_birth_events_serial(
        columns,
        tape,
        {
            {slot_to_agent_id, capacity, 1},
            {event_state.lineage_founder_id, capacity, 1},
            {event_state.lineage_depth, capacity, 1},
            capacity,
        },
        {
            {birth_outputs.decision_parent_slots, ticks * capacity, 1},
            {birth_outputs.decision_reason_codes, ticks * capacity, 1},
            {birth_outputs.decision_count + tick, ticks},
            capacity,
        },
        {
            {birth_outputs.committed_parent_agent_ids,
             ticks * birth_capacity, 1},
            {birth_outputs.committed_child_agent_ids,
             ticks * birth_capacity, 1},
            {birth_outputs.planned_parent_slots, ticks * birth_capacity, 1},
            {birth_outputs.committed_child_slots,
             ticks * birth_capacity, 1},
            {birth_outputs.policy_mutation_occurred,
             ticks * birth_capacity, 1},
            {birth_outputs.policy_mutation_delta,
             ticks * birth_capacity, 1},
            {birth_outputs.planned_child_energy,
             ticks * birth_capacity, 1},
            {birth_outputs.identity_committed_count + tick, ticks},
            birth_capacity,
        },
        cell,
        mechanics_config.generation,
        absolute_tick
    );
    if (event_state.status[cell] != 0) {
        birth_outputs.combined_status[cell_tick] = 3;
        return 3;
    }

    const FlatFreshNewbornSelectionState newborn_state{
        {energy_collected + cell_capacity_offset, 1},
        {energy_collected_exact + cell_capacity_offset, 1},
        {birth_state.attacks + cell_capacity_offset, 1},
        {birth_state.early_survival_steps + cell_capacity_offset, 1},
        {birth_state.early_energy_collected + cell_capacity_offset, 1},
        {birth_state.early_energy_collected_exact + cell_capacity_offset, 1},
        {birth_state.early_injury_gained + cell_capacity_offset, 1},
        {birth_state.early_life_weight + cell_capacity_offset, 1},
        {birth_state.pred_err_sum + cell_capacity_offset, 1},
        {birth_state.pred_err_count + cell_capacity_offset, 1},
        {birth_state.first_repro_step + cell_capacity_offset, 1},
        {birth_state.dev_stage_initialized + cell_capacity_offset, 1},
        {birth_state.episode0_pred_err_sum + cell_capacity_offset, 1},
        {birth_state.episode0_pred_err_count + cell_capacity_offset, 1},
        {birth_state.late_pred_err_sum + cell_capacity_offset, 1},
        {birth_state.late_pred_err_count + cell_capacity_offset, 1},
        {birth_state.chemistry_observation_fidelity + cell_capacity_offset, 1},
        {birth_state.chemistry_observation_bias + cell_capacity_offset, 1},
        {action_cache_actions + cell * capacity * cache_capacity,
         cache_capacity, 1},
        {full_tick_state.prev_drive_delta
             + cell_capacity_offset * drive_width,
         drive_width, 1},
        {full_tick_state.drive_volatility + cell_capacity_offset, 1},
        {full_tick_state.drive_jitter + cell_capacity_offset, 1},
        {full_tick_state.episode0_drive_jitter_sum + cell_capacity_offset, 1},
        {full_tick_state.last_energy_delta + cell_capacity_offset, 1},
        {full_tick_state.last_health_delta + cell_capacity_offset, 1},
        {full_tick_state.last_injury_delta + cell_capacity_offset, 1},
        {full_tick_state.selfharm_events + cell_capacity_offset, 1},
        {full_tick_state.episode0_selfharm_events + cell_capacity_offset, 1},
    };
    const int64_t committed_count =
        birth_outputs.identity_committed_count[cell_tick];
    for (int64_t birth = 0; birth < committed_count; ++birth) {
        const int64_t child_slot =
            birth_outputs.committed_child_slots[tick_birth_offset + birth];
        reset_ordinary_newborn_selection_state(
            newborn_state,
            child_slot,
            {cache_capacity, drive_width}
        );
        const int64_t parent_slot =
            birth_outputs.planned_parent_slots[tick_birth_offset + birth];
        if (birth_state.first_repro_step[cell_capacity_offset + parent_slot] < 0) {
            birth_state.first_repro_step[cell_capacity_offset + parent_slot] =
                survival_steps[cell_capacity_offset + parent_slot];
        }
    }
    return 0;
}

__global__ void flat_dynamic_policy_food_birth_full_tick_persistent_kernel_v7(
    int64_t* __restrict__ active_agent_ids,
    int64_t* __restrict__ active_count_state,
    int64_t* __restrict__ slot_to_agent_id,
    int32_t* __restrict__ agent_id_to_slot,
    bool* __restrict__ alive_mask,
    int64_t* __restrict__ slot_tokens,
    int32_t* __restrict__ action_cache_actions,
    int64_t* __restrict__ action_cache_token,
    int32_t* __restrict__ action_cache_cursor,
    int32_t* __restrict__ action_cache_next,
    int32_t* __restrict__ pos_y,
    int32_t* __restrict__ pos_x,
    int32_t* __restrict__ agent_grid,
    int32_t* __restrict__ food_grid,
    float* __restrict__ energy,
    float* __restrict__ health,
    float* __restrict__ injury,
    double* __restrict__ policy_energy_observation,
    int32_t* __restrict__ rest_streak,
    int64_t* __restrict__ pending_free_agent_ids,
    int64_t* __restrict__ pending_free_slots,
    int64_t* __restrict__ pending_free_count,
    float* __restrict__ energy_collected,
    double* __restrict__ energy_collected_exact,
    int32_t* __restrict__ last_action,
    int32_t* __restrict__ last_last_action,
    int32_t* __restrict__ action_switches,
    int32_t* __restrict__ action_oscillations,
    int32_t* __restrict__ episode0_switches,
    int32_t* __restrict__ episode0_oscillations,
    int32_t* __restrict__ survival_steps,
    float* __restrict__ mlp_pack,
    float* __restrict__ drives,
    float* __restrict__ attack_bias,
    float* __restrict__ action_temperature,
    float* __restrict__ drive_logit_gain,
    float* __restrict__ food_gained_scratch,
    int64_t* __restrict__ ordered_agent_ids_out,
    int32_t* __restrict__ ordered_slots_out,
    float* __restrict__ observations_out,
    float* __restrict__ logits_out,
    int32_t* __restrict__ row_actions_out,
    int32_t* __restrict__ actions_by_slot_out,
    float* __restrict__ pre_energy_out,
    float* __restrict__ pre_health_out,
    float* __restrict__ pre_injury_out,
    int32_t* __restrict__ pre_survival_steps_out,
    int64_t* __restrict__ active_count_out,
    int64_t* __restrict__ policy_refill_rows_out,
    int64_t* __restrict__ policy_cache_hit_rows_out,
    int64_t* __restrict__ death_agent_ids_out,
    int64_t* __restrict__ death_slots_out,
    int32_t* __restrict__ death_reason_codes_out,
    int64_t* __restrict__ death_count_out,
    int64_t* __restrict__ survivor_agent_ids_out,
    int64_t* __restrict__ survivor_count_out,
    int32_t* __restrict__ lifecycle_status_out,
    int32_t* __restrict__ status_out,
    const FlatFullTickState full_tick_state,
    const FlatFullTickOutputs full_tick_outputs,
    const FlatEventState event_state,
    const FlatOrdinaryBirthState ordinary_birth_state,
    const FlatOrdinaryBirthOutputs ordinary_birth_outputs,
    const FlatOrdinaryBirthConfig ordinary_birth_config,
    const FlatMechanicsConfig mechanics_config,
    const int64_t ticks,
    const int64_t active_capacity,
    const int64_t agent_id_capacity,
    const int64_t capacity,
    const int64_t pending_capacity,
    const int64_t height,
    const int64_t width,
    const int64_t view_radius,
    const int64_t observation_size,
    const int64_t hidden_size,
    const int64_t action_count,
    const int64_t packed_width,
    const int64_t drive_width,
    const int64_t cache_capacity,
    const int64_t cache_width,
    const bool deterministic_actions,
    const int64_t max_food,
    const int64_t effective_food_spawn_count,
    const double max_energy,
    const double start_health,
    const double move_cost,
    const double gather_cost,
    const double rest_cost,
    const double attack_cost,
    const double action_cost_mult,
    const double rest_cost_mult,
    const double food_energy,
    const int64_t episode0_k,
    const float w_hunger_gather,
    const float w_pain_rest,
    const float w_fatigue_rest,
    const float w_pain_attack,
    const float w_fear_attack
) {
    const int64_t cell = static_cast<int64_t>(blockIdx.x);
    const int64_t thread = static_cast<int64_t>(threadIdx.x);
    const int64_t side = 2 * view_radius + 1;
    const int64_t local_size = side * side;
    const int64_t cell_capacity_offset = cell * capacity;
    const int64_t cell_grid_offset = cell * height * width;
    const int64_t cell_id_map_offset = cell * agent_id_capacity;
    const int64_t cell_active_offset = cell * active_capacity;
    const int64_t cell_cache_offset = cell_capacity_offset * cache_capacity;
    const int64_t cell_pending_offset = cell * pending_capacity;
    __shared__ int64_t current_active_rows;

    if (
        hidden_size > kMaxHiddenSize
        || observation_size > kMaxObservationSize
        || action_count > kMaxActionCount
        || observation_size != local_size + 2
        || drive_width < 4
        || cache_width <= 0
        || cache_width > cache_capacity
        || pending_capacity < capacity
        || full_tick_state.chemistry_width < 6
        || mechanics_config.start_tick < 0
        || mechanics_config.start_tick + ticks
            > full_tick_state.drive_noise_sample_ticks
        || event_state.capacity < 0
        || event_state.reporting_rows < 0
        || ordinary_birth_config.enabled
            != (ordinary_birth_config.max_births_per_tick > 0)
        || ordinary_birth_config.max_births_per_tick < 0
        || ordinary_birth_config.max_births_per_tick > 1
        || (
            ordinary_birth_config.enabled
            && (
                ordinary_birth_config.gauss_tape_capacity <= 0
                || ordinary_birth_config.policy_tape_capacity <= 0
                || ordinary_birth_config.torch_state_bytes <= 0
                || ordinary_birth_config.hmem_width < 0
                || ordinary_birth_config.sensor_radius_max < 0
                || !::isfinite(ordinary_birth_config.parent_energy_share)
                || ordinary_birth_config.parent_energy_share < 0.0
                || ordinary_birth_config.parent_energy_share > 1.0
                || !::isfinite(ordinary_birth_config.minimum_energy_buffer)
                || ordinary_birth_config.minimum_energy_buffer < 0.0f
            )
        )
    ) {
        if (thread == 0) {
            record_status(status_out, cell, 1);
        }
        return;
    }

    if (thread == 0) {
        current_active_rows = active_count_state[cell];
        if (current_active_rows < 0 || current_active_rows > active_capacity) {
            record_status(status_out, cell, 4);
        }
    }
    __syncthreads();
    if (status_out[cell] != 0) {
        return;
    }

    // Stage 2 resets the generation-local action cache when start_tick is zero.
    for (int64_t slot = thread; slot < capacity; slot += blockDim.x) {
        const int64_t slot_index = cell_capacity_offset + slot;
        action_cache_token[slot_index] = -1;
        action_cache_cursor[slot_index] = -1;
        action_cache_next[slot_index] = -1;
    }
    for (
        int64_t index = thread;
        index < capacity * cache_capacity;
        index += blockDim.x
    ) {
        action_cache_actions[cell_cache_offset + index] = -1;
    }
    __syncthreads();

    // One block owns the complete Flat tick. Thread zero preserves the accepted
    // serial lifecycle/RNG order while the block performs data-parallel policy
    // materialization and snapshots.
    for (int64_t tick = 0; tick < ticks; ++tick) {
        const int64_t cell_tick = cell * ticks + tick;
        const int64_t pre_action_rows = current_active_rows;
        if (thread == 0) {
            active_count_out[cell_tick] = pre_action_rows;
            policy_refill_rows_out[cell_tick] = 0;
            policy_cache_hit_rows_out[cell_tick] = 0;
            full_tick_outputs.runner_shuffle_draws[cell_tick] = 0;
            full_tick_outputs.runner_shuffle_status[cell_tick] = 0;
            full_tick_outputs.drive_noise_draws[cell_tick] = 0;
            full_tick_outputs.drive_noise_status[cell_tick] = 0;
            full_tick_outputs.chemistry_draws[cell_tick] = 0;
            full_tick_outputs.chemistry_status[cell_tick] = 0;
            full_tick_outputs.active_mechanics_status[cell_tick] = 0;
            full_tick_outputs.residual_death_count[cell_tick] = 0;
            full_tick_outputs.residual_release_count[cell_tick] = 0;
            full_tick_outputs.residual_release_status[cell_tick] = 0;
            for (int64_t stat = 0; stat < 3; ++stat) {
                full_tick_outputs.food_spawn_stats[cell_tick * 3 + stat] = 0;
            }
            for (int64_t count = 0; count < 5; ++count) {
                full_tick_outputs.active_mechanics_counts[
                    cell_tick * 5 + count
                ] = 0;
            }
        }
        for (int64_t slot = thread; slot < capacity; slot += blockDim.x) {
            actions_by_slot_out[cell_tick * capacity + slot] = kRestAction;
            food_gained_scratch[cell_capacity_offset + slot] = 0.0f;
            full_tick_outputs.chemistry_order_slots[
                cell_tick * capacity + slot
            ] = -1;
            full_tick_outputs.residual_death_agent_ids[
                cell_tick * capacity + slot
            ] = -1;
            full_tick_outputs.residual_death_slots[
                cell_tick * capacity + slot
            ] = -1;
            full_tick_outputs.residual_death_reason_codes[
                cell_tick * capacity + slot
            ] = 0;
            full_tick_state.residual_pending_free_agent_ids[
                cell_capacity_offset + slot
            ] = -1;
            full_tick_state.residual_pending_free_slots[
                cell_capacity_offset + slot
            ] = -1;
        }
        if (thread == 0) {
            full_tick_state.residual_pending_free_count[cell] = 0;
        }
        __syncthreads();

        if (thread == 0) {
            using namespace persistent_food_spawn;
            spawn_flat_food_exact(
                {
                    food_grid + cell_grid_offset,
                    nullptr,
                },
                {
                    full_tick_state.shared_env_policy_pcg64_state_words + cell * 4,
                    full_tick_state.shared_env_policy_pcg64_buffer_words + cell * 2,
                    full_tick_state.policy_np_semantic_draws + cell,
                },
                {
                    full_tick_outputs.food_spawn_stats + cell_tick * 3,
                },
                {
                    height * width,
                    max_food,
                    effective_food_spawn_count,
                    false,
                    0.0f,
                    0.0f,
                }
            );
        }
        __syncthreads();

        for (int64_t row = thread; row < pre_action_rows; row += blockDim.x) {
            const int64_t agent_id = active_agent_ids[cell_active_offset + row];
            if (agent_id < 0 || agent_id >= agent_id_capacity) {
                record_status(status_out, cell, 2);
                continue;
            }
            const int32_t slot = agent_id_to_slot[
                cell_id_map_offset + static_cast<int64_t>(agent_id)
            ];
            if (slot < 0 || static_cast<int64_t>(slot) >= capacity) {
                record_status(status_out, cell, 3);
                continue;
            }

            const int64_t row_out = cell_tick * capacity + row;
            const int64_t slot_index = cell_capacity_offset + slot;
            ordered_agent_ids_out[row_out] = agent_id;
            ordered_slots_out[row_out] = slot;

            const int32_t cursor = action_cache_cursor[slot_index];
            const bool cache_hit = action_cache_token[slot_index] == slot_tokens[slot_index]
                && cursor >= 0
                && static_cast<int64_t>(cursor) < cache_width;
            if (cache_hit) {
                if (deterministic_actions) {
                    int32_t action = action_cache_actions[
                        cell_cache_offset
                        + static_cast<int64_t>(slot) * cache_capacity
                        + cursor
                    ];
                    if (action < 0 || action > kAttackAction) {
                        action = action_cache_next[slot_index];
                    }
                    if (action < 0 || action > kAttackAction) {
                        action = kRestAction;
                    }
                    action_cache_next[slot_index] = action;
                    action_cache_cursor[slot_index] = cursor + 1;
                    row_actions_out[row_out] = action;
                    actions_by_slot_out[cell_tick * capacity + slot] = action;
                    atomicAdd(
                        reinterpret_cast<unsigned long long*>(
                            policy_cache_hit_rows_out + cell_tick
                        ),
                        1ULL
                    );
                }
                continue;
            }

            const int32_t cy = pos_y[slot_index];
            const int32_t cx = pos_x[slot_index];
            float* observation = observations_out + row_out * observation_size;
            int64_t observation_offset = 0;
            for (int64_t dy = -view_radius; dy <= view_radius; ++dy) {
                for (int64_t dx = -view_radius; dx <= view_radius; ++dx) {
                    const int64_t y = static_cast<int64_t>(cy) + dy;
                    const int64_t x = static_cast<int64_t>(cx) + dx;
                    float value = 0.0f;
                    if (y >= 0 && y < height && x >= 0 && x < width) {
                        const int64_t grid_index = cell_grid_offset + y * width + x;
                        if (agent_grid[grid_index] != -1) {
                            value = -1.0f;
                        } else if (food_grid[grid_index] == kFood) {
                            value = 1.0f;
                        }
                    }
                    observation[observation_offset++] = value;
                }
            }
            observation[local_size] = energy[slot_index] / static_cast<float>(max_energy);
            observation[local_size + 1] = health[slot_index] / static_cast<float>(start_health);

            // Stochastic cache service must consume Gumbel draws serially in
            // compact miss order. Thread zero performs its MLP/refill after
            // every missing observation has been materialized.
            if (!deterministic_actions) {
                continue;
            }

            const float* pack = mlp_pack + slot_index * packed_width;
            float h1[kMaxHiddenSize];
            float h2[kMaxHiddenSize];
            int64_t pack_offset = 0;
            for (int64_t hidden = 0; hidden < hidden_size; ++hidden) {
                float value = pack[hidden_size * observation_size + hidden];
                const float* weights = pack + hidden * observation_size;
                for (int64_t feature = 0; feature < observation_size; ++feature) {
                    value += weights[feature] * observation[feature];
                }
                h1[hidden] = tanhf(value);
            }
            pack_offset += hidden_size * observation_size + hidden_size;
            for (int64_t hidden = 0; hidden < hidden_size; ++hidden) {
                float value = pack[pack_offset + hidden_size * hidden_size + hidden];
                const float* weights = pack + pack_offset + hidden * hidden_size;
                for (int64_t input = 0; input < hidden_size; ++input) {
                    value += weights[input] * h1[input];
                }
                h2[hidden] = tanhf(value);
            }
            pack_offset += hidden_size * hidden_size + hidden_size;

            const float temperature = action_temperature[slot_index] < 0.05f
                ? 0.05f
                : action_temperature[slot_index];
            const float gain = drive_logit_gain[slot_index] < 0.0f
                ? 0.0f
                : drive_logit_gain[slot_index];
            const float* slot_drives = drives + slot_index * drive_width;
            int32_t best_action = 0;
            float best_score = 0.0f;
            for (int64_t action = 0; action < action_count; ++action) {
                const float* weights = pack + pack_offset + action * hidden_size;
                float score = pack[
                    pack_offset + action_count * hidden_size + action
                ];
                for (int64_t hidden = 0; hidden < hidden_size; ++hidden) {
                    score += weights[hidden] * h2[hidden];
                }
                if (action == kAttackAction) {
                    score += attack_bias[slot_index];
                }
                score /= temperature;
                if (action == kGatherAction) {
                    score += gain * (w_hunger_gather * slot_drives[0]);
                } else if (action == kRestAction) {
                    score += gain * (
                        w_pain_rest * slot_drives[1]
                        + w_fatigue_rest * slot_drives[2]
                    );
                } else if (action == kAttackAction) {
                    score -= gain * (
                        w_pain_attack * slot_drives[1]
                        + w_fear_attack * slot_drives[3]
                    );
                }
                logits_out[row_out * action_count + action] = score;
                if (action == 0 || score > best_score) {
                    best_score = score;
                    best_action = static_cast<int32_t>(action);
                }
            }

            const int64_t cache_offset = cell_cache_offset
                + static_cast<int64_t>(slot) * cache_capacity;
            for (int64_t cache_index = 0; cache_index < cache_width; ++cache_index) {
                action_cache_actions[cache_offset + cache_index] = best_action;
            }
            action_cache_token[slot_index] = slot_tokens[slot_index];
            action_cache_cursor[slot_index] = 1;
            action_cache_next[slot_index] = best_action;
            row_actions_out[row_out] = best_action;
            actions_by_slot_out[cell_tick * capacity + slot] = best_action;
            atomicAdd(
                reinterpret_cast<unsigned long long*>(policy_refill_rows_out + cell_tick),
                1ULL
            );
        }
        __syncthreads();

        if (!deterministic_actions && status_out[cell] == 0 && thread == 0) {
            using namespace persistent_stochastic_policy;
            PolicyCacheCounters counters{};
            const int64_t tick_capacity_offset = cell_tick * capacity;
            service_fixed_mlp_gumbel_policy_serial(
                ordered_agent_ids_out + tick_capacity_offset,
                ordered_slots_out + tick_capacity_offset,
                observations_out + tick_capacity_offset * observation_size,
                mlp_pack + cell_capacity_offset * packed_width,
                drives + cell_capacity_offset * drive_width,
                attack_bias + cell_capacity_offset,
                action_temperature + cell_capacity_offset,
                drive_logit_gain + cell_capacity_offset,
                slot_tokens + cell_capacity_offset,
                action_cache_actions + cell_cache_offset,
                action_cache_token + cell_capacity_offset,
                action_cache_cursor + cell_capacity_offset,
                action_cache_next + cell_capacity_offset,
                logits_out + tick_capacity_offset * action_count,
                row_actions_out + tick_capacity_offset,
                death_slots_out + tick_capacity_offset,
                full_tick_state.shared_env_policy_pcg64_state_words + cell * 4,
                full_tick_state.shared_env_policy_pcg64_buffer_words + cell * 2,
                &counters,
                {
                    pre_action_rows,
                    capacity,
                    observation_size,
                    hidden_size,
                    action_count,
                    packed_width,
                    drive_width,
                    cache_capacity,
                    cache_width,
                    w_hunger_gather,
                    w_pain_rest,
                    w_fatigue_rest,
                    w_pain_attack,
                    w_fear_attack,
                    false,
                }
            );
            if (
                counters.requested_rows != pre_action_rows
                || counters.missed_rows != counters.refilled_rows
            ) {
                record_status(status_out, cell, 33);
            } else {
                policy_refill_rows_out[cell_tick] = counters.refilled_rows;
                policy_cache_hit_rows_out[cell_tick] = counters.hit_rows;
                full_tick_state.policy_np_semantic_draws[cell] +=
                    counters.gumbel_draws;
                for (int64_t row = 0; row < pre_action_rows; ++row) {
                    const int32_t slot = ordered_slots_out[
                        tick_capacity_offset + row
                    ];
                    actions_by_slot_out[tick_capacity_offset + slot] =
                        row_actions_out[tick_capacity_offset + row];
                }
            }
        }
        __syncthreads();
        if (status_out[cell] != 0) {
            return;
        }
        if (!deterministic_actions) {
            const int64_t tick_capacity_offset = cell_tick * capacity;
            for (int64_t row = thread; row < capacity; row += blockDim.x) {
                death_slots_out[tick_capacity_offset + row] = -1;
            }
        }
        __syncthreads();

        if (thread == 0) {
            int32_t* chemistry_order = full_tick_outputs.chemistry_order_slots
                + cell_tick * capacity;
            for (int64_t row = 0; row < pre_action_rows; ++row) {
                chemistry_order[row] = ordered_slots_out[
                    cell_tick * capacity + row
                ];
            }
            persistent_chemistry::runner_shuffle_counted(
                chemistry_order,
                pre_action_rows,
                full_tick_state.runner_pcg64_state_words + cell * 4,
                full_tick_state.runner_pcg64_buffer_words + cell * 2,
                full_tick_outputs.runner_shuffle_draws + cell_tick,
                full_tick_outputs.runner_shuffle_status + cell_tick,
                capacity
            );
            if (full_tick_outputs.runner_shuffle_status[cell_tick] == 0) {
                full_tick_state.metrics_np_semantic_draws[cell] +=
                    full_tick_outputs.runner_shuffle_draws[cell_tick];
            } else {
                record_status(status_out, cell, 20);
            }
            for (int64_t word = 0; word < 4; ++word) {
                full_tick_outputs.policy_pcg64_state_after_action[
                    cell_tick * 4 + word
                ] = full_tick_state.shared_env_policy_pcg64_state_words[
                    cell * 4 + word
                ];
            }
            for (int64_t word = 0; word < 2; ++word) {
                full_tick_outputs.policy_pcg64_buffer_after_action[
                    cell_tick * 2 + word
                ] = full_tick_state.shared_env_policy_pcg64_buffer_words[
                    cell * 2 + word
                ];
            }
        }
        __syncthreads();
        if (status_out[cell] != 0) {
            return;
        }

        // Stage 2 commits coherence counters before cloning pre-action state.
        if (thread == 0) {
            for (int64_t row = 0; row < pre_action_rows; ++row) {
                const int64_t agent_id = active_agent_ids[cell_active_offset + row];
                if (agent_id < 0 || agent_id >= agent_id_capacity) {
                    continue;
                }
                const int32_t slot = agent_id_to_slot[cell_id_map_offset + agent_id];
                if (slot < 0 || static_cast<int64_t>(slot) >= capacity) {
                    continue;
                }
                const int64_t slot_index = cell_capacity_offset + slot;
                const int32_t action = row_actions_out[cell_tick * capacity + row];
                const int32_t old_last = last_action[slot_index];
                const int32_t old_last_last = last_last_action[slot_index];
                if (old_last != -1 && action != old_last) {
                    action_switches[slot_index] += 1;
                    if (survival_steps[slot_index] < episode0_k) {
                        episode0_switches[slot_index] += 1;
                    }
                }
                if (old_last_last != -1 && action == old_last_last && action != old_last) {
                    action_oscillations[slot_index] += 1;
                    if (survival_steps[slot_index] < episode0_k) {
                        episode0_oscillations[slot_index] += 1;
                    }
                }
                last_last_action[slot_index] = old_last;
                last_action[slot_index] = action;
            }
        }
        __syncthreads();

        for (int64_t slot = thread; slot < capacity; slot += blockDim.x) {
            const int64_t source = cell_capacity_offset + slot;
            const int64_t target = cell_tick * capacity + slot;
            pre_energy_out[target] = energy[source];
            pre_health_out[target] = health[source];
            pre_injury_out[target] = injury[source];
            pre_survival_steps_out[target] = survival_steps[source];
        }
        __syncthreads();

        // Occupancy changes are order-dependent and must follow ascending agent ID.
        if (thread == 0) {
            for (int64_t row = 0; row < pre_action_rows; ++row) {
                const int64_t agent_id = active_agent_ids[cell_active_offset + row];
                if (agent_id < 0 || agent_id >= agent_id_capacity) {
                    continue;
                }
                const int32_t slot = agent_id_to_slot[cell_id_map_offset + agent_id];
                if (slot < 0 || static_cast<int64_t>(slot) >= capacity) {
                    continue;
                }
                const int32_t action = row_actions_out[cell_tick * capacity + row];
                if (action < 0 || action > 3) {
                    continue;
                }
                const int64_t slot_index = cell_capacity_offset + slot;
                const int32_t y = pos_y[slot_index];
                const int32_t x = pos_x[slot_index];
                int32_t ny = y;
                int32_t nx = x;
                if (action == 0) {
                    --ny;
                } else if (action == 1) {
                    ++ny;
                } else if (action == 2) {
                    ++nx;
                } else {
                    --nx;
                }
                if (ny < 0 || ny >= height || nx < 0 || nx >= width) {
                    continue;
                }
                const int64_t target = cell_grid_offset
                    + static_cast<int64_t>(ny) * width + nx;
                if (agent_grid[target] != -1) {
                    continue;
                }
                const int64_t source = cell_grid_offset
                    + static_cast<int64_t>(y) * width + x;
                if (agent_grid[source] == static_cast<int32_t>(agent_id)) {
                    agent_grid[source] = -1;
                }
                agent_grid[target] = static_cast<int32_t>(agent_id);
                pos_y[slot_index] = ny;
                pos_x[slot_index] = nx;
            }
        }
        __syncthreads();

        if (thread == 0) {
            for (int64_t row = 0; row < pre_action_rows; ++row) {
                const int64_t agent_id = active_agent_ids[cell_active_offset + row];
                if (agent_id < 0 || agent_id >= agent_id_capacity) {
                    continue;
                }
                const int32_t slot = agent_id_to_slot[cell_id_map_offset + agent_id];
                if (slot < 0 || static_cast<int64_t>(slot) >= capacity) {
                    continue;
                }
                if (row_actions_out[cell_tick * capacity + row] != kGatherAction) {
                    continue;
                }
                const int64_t slot_index = cell_capacity_offset + slot;
                const int64_t grid_index = cell_grid_offset
                    + static_cast<int64_t>(pos_y[slot_index]) * width
                    + pos_x[slot_index];
                if (food_grid[grid_index] == kFood) {
                    food_grid[grid_index] = kEmpty;
                    food_gained_scratch[slot_index] = static_cast<float>(food_energy);
                }
            }
        }
        __syncthreads();

        if (thread == 0) {
            for (int64_t row = 0; row < pre_action_rows; ++row) {
                const int64_t agent_id = active_agent_ids[cell_active_offset + row];
                if (agent_id < 0 || agent_id >= agent_id_capacity) {
                    continue;
                }
                const int32_t slot = agent_id_to_slot[cell_id_map_offset + agent_id];
                if (slot < 0 || static_cast<int64_t>(slot) >= capacity) {
                    continue;
                }
                const int64_t slot_index = cell_capacity_offset + slot;
                const int32_t action = row_actions_out[cell_tick * capacity + row];
                const float cost = action_cost(
                    action,
                    move_cost,
                    gather_cost,
                    rest_cost,
                    attack_cost,
                    action_cost_mult,
                    rest_cost_mult
                );
                float updated = energy[slot_index]
                    - cost
                    + food_gained_scratch[slot_index];
                if (updated > static_cast<float>(max_energy)) {
                    updated = static_cast<float>(max_energy);
                }
                if (updated < 0.0f) {
                    updated = 0.0f;
                }
                energy[slot_index] = updated;
                policy_energy_observation[slot_index] = static_cast<double>(updated);
                if (food_gained_scratch[slot_index] > 0.0f) {
                    energy_collected_exact[slot_index] += static_cast<double>(
                        food_gained_scratch[slot_index]
                    );
                    energy_collected[slot_index] = static_cast<float>(
                        energy_collected_exact[slot_index]
                    );
                }
            }
        }
        __syncthreads();

        if (thread == 0) {
            const int64_t tick_capacity_offset = cell_tick * capacity;
            const int32_t lifecycle_status = lifecycle_death_cull_cell_v3(
                active_agent_ids + cell_active_offset,
                active_count_state + cell,
                slot_to_agent_id + cell_capacity_offset,
                agent_id_to_slot + cell_id_map_offset,
                alive_mask + cell_capacity_offset,
                pos_y + cell_capacity_offset,
                pos_x + cell_capacity_offset,
                agent_grid + cell_grid_offset,
                energy + cell_capacity_offset,
                health + cell_capacity_offset,
                policy_energy_observation + cell_capacity_offset,
                rest_streak + cell_capacity_offset,
                action_cache_token + cell_capacity_offset,
                action_cache_cursor + cell_capacity_offset,
                action_cache_next + cell_capacity_offset,
                pending_free_agent_ids + cell_pending_offset,
                pending_free_slots + cell_pending_offset,
                pending_free_count + cell,
                death_agent_ids_out + tick_capacity_offset,
                death_slots_out + tick_capacity_offset,
                death_reason_codes_out + tick_capacity_offset,
                death_count_out + cell_tick,
                survivor_agent_ids_out + tick_capacity_offset,
                survivor_count_out + cell_tick,
                capacity,
                active_capacity,
                agent_id_capacity,
                height,
                width,
                pending_capacity
            );
            lifecycle_status_out[cell_tick] = lifecycle_status;
            if (lifecycle_status != kLifecycleOk) {
                record_status(status_out, cell, 4 + lifecycle_status);
            } else {
                current_active_rows = active_count_state[cell];
                accumulate_action_terminal_v4(
                    event_state,
                    death_agent_ids_out,
                    death_slots_out,
                    death_reason_codes_out,
                    death_count_out,
                    survival_steps,
                    ticks,
                    capacity,
                    agent_id_capacity,
                    cell,
                    tick
                );
                append_death_event_class_v4(
                    event_state,
                    death_agent_ids_out,
                    death_slots_out,
                    death_reason_codes_out,
                    death_count_out,
                    ticks,
                    capacity,
                    cell,
                    tick,
                    mechanics_config.generation,
                    mechanics_config.start_tick + tick
                );
            }
        }
        __syncthreads();
        if (status_out[cell] != 0) {
            return;
        }

        if (thread == 0) {
            using namespace persistent_active_mechanics;
            const int64_t global_tick = mechanics_config.start_tick + tick;
            const int64_t sample_cell_offset =
                (cell * full_tick_state.drive_noise_sample_ticks + global_tick)
                * capacity;
            const FlatActiveMechanicsCellState mechanics_state{
                {slot_to_agent_id + cell_capacity_offset, 1},
                {alive_mask + cell_capacity_offset, 1},
                {energy + cell_capacity_offset, 1},
                {health + cell_capacity_offset, 1},
                {injury + cell_capacity_offset, 1},
                {full_tick_state.age + cell_capacity_offset, 1},
                {policy_energy_observation + cell_capacity_offset, 1},
                {rest_streak + cell_capacity_offset, 1},
                {survival_steps + cell_capacity_offset, 1},
                {full_tick_state.gru_upkeep_cost + cell_capacity_offset, 1},
                {full_tick_state.cost_charge_scalar + cell_capacity_offset, 1},
                {
                    full_tick_state.cost_charge_maturation_scalar
                        + cell_capacity_offset,
                    1
                },
                {drives + cell_capacity_offset * drive_width, drive_width, 1},
                {full_tick_state.drive_inertia + cell_capacity_offset, 1},
                {full_tick_state.drive_noise + cell_capacity_offset, 1},
                {
                    full_tick_state.prev_drive_delta
                        + cell_capacity_offset * drive_width,
                    drive_width,
                    1
                },
                {full_tick_state.drive_volatility + cell_capacity_offset, 1},
                {full_tick_state.drive_jitter + cell_capacity_offset, 1},
                {
                    full_tick_state.episode0_drive_jitter_sum
                        + cell_capacity_offset,
                    1
                },
                {full_tick_state.last_energy_delta + cell_capacity_offset, 1},
                {full_tick_state.last_health_delta + cell_capacity_offset, 1},
                {full_tick_state.last_injury_delta + cell_capacity_offset, 1},
                {full_tick_state.selfharm_events + cell_capacity_offset, 1},
                {
                    full_tick_state.episode0_selfharm_events
                        + cell_capacity_offset,
                    1
                },
            };
            const FlatActiveMechanicsPreAction pre_action{
                {actions_by_slot_out + cell_tick * capacity, 1},
                {pre_energy_out + cell_tick * capacity, 1},
                {pre_health_out + cell_tick * capacity, 1},
                {pre_injury_out + cell_tick * capacity, 1},
                {
                    full_tick_state.drive_noise_samples
                        + sample_cell_offset * drive_width,
                    drive_width,
                    1
                },
                {
                    full_tick_state.drive_noise_samples_valid
                        + sample_cell_offset,
                    1
                },
            };
            run_flat_active_mechanics(
                mechanics_state,
                pre_action,
                full_tick_outputs.active_mechanics_status + cell_tick,
                full_tick_outputs.active_mechanics_counts + cell_tick * 5,
                1,
                capacity,
                mechanics_config.enable_rest_charge,
                mechanics_config.rest_streak_free,
                mechanics_config.rest_charge_energy,
                mechanics_config.rest_charge_growth,
                mechanics_config.rest_deficit_to_health,
                mechanics_config.developmental_maturity_age,
                mechanics_config.life_history_mode,
                mechanics_config.life_history_maturity_start_ticks,
                mechanics_config.life_history_maturity_rise_ticks,
                mechanics_config.life_history_fertility_decline_start_ticks,
                mechanics_config.life_history_fertility_decline_ticks,
                mechanics_config.life_history_elder_fertility_floor,
                mechanics_config.life_history_senescence_start_ticks,
                mechanics_config.life_history_senescence_ramp_ticks,
                mechanics_config.life_history_senescence_energy_cost,
                mechanics_config.life_history_senescence_health_cost,
                mechanics_config.life_history_max_age_ticks,
                static_cast<int32_t>(episode0_k),
                static_cast<float>(max_energy),
                static_cast<float>(start_health),
                mechanics_config.drive_hunger_gain,
                mechanics_config.drive_hunger_decay,
                mechanics_config.drive_pain_gain,
                mechanics_config.drive_pain_decay,
                mechanics_config.drive_fatigue_gain,
                mechanics_config.drive_fatigue_rest_relief,
                mechanics_config.drive_fatigue_decay,
                mechanics_config.drive_fear_gain,
                mechanics_config.drive_fear_decay
            );
            if (full_tick_outputs.active_mechanics_status[cell_tick] != 0) {
                record_status(status_out, cell, 30);
            }

            int32_t* chemistry_order = full_tick_outputs.chemistry_order_slots
                + cell_tick * capacity;
            persistent_chemistry::advance_precomputed_drive_noise_rng(
                chemistry_order,
                pre_action_rows,
                full_tick_state.drive_noise + cell_capacity_offset,
                full_tick_state.runner_pcg64_state_words + cell * 4,
                full_tick_state.runner_pcg64_buffer_words + cell * 2,
                full_tick_outputs.drive_noise_draws + cell_tick,
                full_tick_outputs.drive_noise_status + cell_tick,
                capacity
            );
            if (full_tick_outputs.drive_noise_status[cell_tick] == 0) {
                full_tick_state.metrics_np_semantic_draws[cell] +=
                    full_tick_outputs.drive_noise_draws[cell_tick];
            } else {
                record_status(status_out, cell, 40);
            }

            persistent_chemistry::chemistry_update_counted(
                full_tick_state.chemistry
                    + cell_capacity_offset * full_tick_state.chemistry_width,
                energy + cell_capacity_offset,
                health + cell_capacity_offset,
                injury + cell_capacity_offset,
                drives + cell_capacity_offset * drive_width,
                alive_mask + cell_capacity_offset,
                chemistry_order,
                pre_action_rows,
                full_tick_state.shared_env_policy_pcg64_state_words + cell * 4,
                full_tick_state.shared_env_policy_pcg64_buffer_words + cell * 2,
                full_tick_state.policy_np_semantic_draws + cell,
                full_tick_outputs.chemistry_draws + cell_tick,
                full_tick_outputs.chemistry_status + cell_tick,
                capacity,
                {
                    full_tick_state.chemistry_width,
                    drive_width,
                    static_cast<float>(max_energy),
                    static_cast<float>(start_health),
                }
            );
            if (full_tick_outputs.chemistry_status[cell_tick] != 0) {
                record_status(status_out, cell, 50);
            }

            const int64_t tick_capacity_offset = cell_tick * capacity;
            const int32_t residual_status = lifecycle_death_cull_cell_v3(
                active_agent_ids + cell_active_offset,
                active_count_state + cell,
                slot_to_agent_id + cell_capacity_offset,
                agent_id_to_slot + cell_id_map_offset,
                alive_mask + cell_capacity_offset,
                pos_y + cell_capacity_offset,
                pos_x + cell_capacity_offset,
                agent_grid + cell_grid_offset,
                energy + cell_capacity_offset,
                health + cell_capacity_offset,
                policy_energy_observation + cell_capacity_offset,
                rest_streak + cell_capacity_offset,
                action_cache_token + cell_capacity_offset,
                action_cache_cursor + cell_capacity_offset,
                action_cache_next + cell_capacity_offset,
                full_tick_state.residual_pending_free_agent_ids
                    + cell_capacity_offset,
                full_tick_state.residual_pending_free_slots
                    + cell_capacity_offset,
                full_tick_state.residual_pending_free_count + cell,
                full_tick_outputs.residual_death_agent_ids
                    + tick_capacity_offset,
                full_tick_outputs.residual_death_slots + tick_capacity_offset,
                full_tick_outputs.residual_death_reason_codes
                    + tick_capacity_offset,
                full_tick_outputs.residual_death_count + cell_tick,
                survivor_agent_ids_out + tick_capacity_offset,
                survivor_count_out + cell_tick,
                capacity,
                active_capacity,
                agent_id_capacity,
                height,
                width,
                capacity
            );
            if (residual_status != kLifecycleOk) {
                record_status(status_out, cell, 60 + residual_status);
            } else {
                current_active_rows = active_count_state[cell];
                append_death_event_class_v4(
                    event_state,
                    full_tick_outputs.residual_death_agent_ids,
                    full_tick_outputs.residual_death_slots,
                    full_tick_outputs.residual_death_reason_codes,
                    full_tick_outputs.residual_death_count,
                    ticks,
                    capacity,
                    cell,
                    tick,
                    mechanics_config.generation,
                    global_tick
                );
            }

            const int32_t release_status = release_residual_free_slots_v4(
                slot_to_agent_id + cell_capacity_offset,
                alive_mask + cell_capacity_offset,
                full_tick_state.slot_free_queue + cell_capacity_offset,
                full_tick_state.slot_free_queue_count + cell,
                full_tick_outputs.residual_death_slots + tick_capacity_offset,
                full_tick_outputs.residual_death_count[cell_tick],
                full_tick_outputs.residual_release_count + cell_tick,
                capacity,
                capacity
            );
            full_tick_outputs.residual_release_status[cell_tick] = release_status;
            if (release_status != kLifecycleOk) {
                record_status(status_out, cell, 70 + release_status);
            }

            if (ordinary_birth_config.enabled) {
                const int32_t birth_status = execute_ordinary_birth_microcycle_v7(
                    active_agent_ids,
                    active_count_state,
                    ordered_slots_out,
                    slot_to_agent_id,
                    agent_id_to_slot,
                    slot_tokens,
                    alive_mask,
                    pos_y,
                    pos_x,
                    agent_grid,
                    energy,
                    health,
                    injury,
                    policy_energy_observation,
                    rest_streak,
                    action_cache_actions,
                    action_cache_token,
                    action_cache_cursor,
                    action_cache_next,
                    last_action,
                    last_last_action,
                    action_switches,
                    action_oscillations,
                    episode0_switches,
                    episode0_oscillations,
                    survival_steps,
                    mlp_pack,
                    drives,
                    attack_bias,
                    action_temperature,
                    drive_logit_gain,
                    energy_collected,
                    energy_collected_exact,
                    full_tick_state,
                    event_state,
                    ordinary_birth_state,
                    ordinary_birth_outputs,
                    ordinary_birth_config,
                    mechanics_config,
                    cell,
                    tick,
                    ticks,
                    capacity,
                    active_capacity,
                    agent_id_capacity,
                    height,
                    width,
                    cache_capacity,
                    drive_width,
                    packed_width,
                    full_tick_state.chemistry_width,
                    static_cast<float>(max_energy),
                    static_cast<float>(start_health),
                    release_status
                );
                if (birth_status != 0) {
                    record_status(status_out, cell, 90 + birth_status);
                } else {
                    current_active_rows = active_count_state[cell];
                }
            }

            // The authoritative checkpoint value is the action at the
            // post-service cursor, not the action consumed this tick.
            for (int64_t slot = 0; slot < capacity; ++slot) {
                const int64_t slot_index = cell_capacity_offset + slot;
                const int32_t cursor = action_cache_cursor[slot_index];
                if (
                    !alive_mask[slot_index]
                    || action_cache_token[slot_index] < 0
                    || cursor < 0
                ) {
                    action_cache_next[slot_index] = -1;
                    continue;
                }
                const int64_t projected_index = cursor < cache_width
                    ? static_cast<int64_t>(cursor)
                    : cache_width - 1;
                action_cache_next[slot_index] = action_cache_actions[
                    cell_cache_offset + slot * cache_capacity + projected_index
                ];
            }
            event_state.status_by_tick[cell_tick] = event_state.status[cell];
            if (event_state.status[cell] != 0) {
                record_status(status_out, cell, 80 + event_state.status[cell]);
            }
            lifecycle_status_out[cell_tick] = status_out[cell];
        }
        __syncthreads();
        if (status_out[cell] != 0) {
            return;
        }
    }
}

}  // namespace

void launch_flat_dynamic_policy_food_birth_full_tick_v7(
    int64_t* active_agent_ids,
    int64_t* active_count_state,
    int64_t* slot_to_agent_id,
    int32_t* agent_id_to_slot,
    bool* alive_mask,
    int64_t* slot_tokens,
    int32_t* action_cache_actions,
    int64_t* action_cache_token,
    int32_t* action_cache_cursor,
    int32_t* action_cache_next,
    int32_t* pos_y,
    int32_t* pos_x,
    int32_t* agent_grid,
    int32_t* food_grid,
    float* energy,
    float* health,
    float* injury,
    double* policy_energy_observation,
    int32_t* rest_streak,
    int64_t* pending_free_agent_ids,
    int64_t* pending_free_slots,
    int64_t* pending_free_count,
    float* energy_collected,
    double* energy_collected_exact,
    int32_t* last_action,
    int32_t* last_last_action,
    int32_t* action_switches,
    int32_t* action_oscillations,
    int32_t* episode0_switches,
    int32_t* episode0_oscillations,
    int32_t* survival_steps,
    float* mlp_pack,
    float* drives,
    float* attack_bias,
    float* action_temperature,
    float* drive_logit_gain,
    float* food_gained_scratch,
    int64_t* ordered_agent_ids_out,
    int32_t* ordered_slots_out,
    float* observations_out,
    float* logits_out,
    int32_t* row_actions_out,
    int32_t* actions_by_slot_out,
    float* pre_energy_out,
    float* pre_health_out,
    float* pre_injury_out,
    int32_t* pre_survival_steps_out,
    int64_t* active_count_out,
    int64_t* policy_refill_rows_out,
    int64_t* policy_cache_hit_rows_out,
    int64_t* death_agent_ids_out,
    int64_t* death_slots_out,
    int32_t* death_reason_codes_out,
    int64_t* death_count_out,
    int64_t* survivor_agent_ids_out,
    int64_t* survivor_count_out,
    int32_t* lifecycle_status_out,
    int32_t* status_out,
    const FlatFullTickState full_tick_state,
    const FlatFullTickOutputs full_tick_outputs,
    const FlatEventState event_state,
    const FlatOrdinaryBirthState ordinary_birth_state,
    const FlatOrdinaryBirthOutputs ordinary_birth_outputs,
    const FlatOrdinaryBirthConfig ordinary_birth_config,
    const FlatMechanicsConfig mechanics_config,
    const int64_t cells,
    const int64_t ticks,
    const int64_t active_capacity,
    const int64_t agent_id_capacity,
    const int64_t capacity,
    const int64_t pending_capacity,
    const int64_t height,
    const int64_t width,
    const int64_t view_radius,
    const int64_t observation_size,
    const int64_t hidden_size,
    const int64_t action_count,
    const int64_t packed_width,
    const int64_t drive_width,
    const int64_t cache_capacity,
    const int64_t cache_width,
    const bool deterministic_actions,
    const int64_t max_food,
    const int64_t effective_food_spawn_count,
    const double max_energy,
    const double start_health,
    const double move_cost,
    const double gather_cost,
    const double rest_cost,
    const double attack_cost,
    const double action_cost_mult,
    const double rest_cost_mult,
    const double food_energy,
    const int64_t episode0_k,
    const float w_hunger_gather,
    const float w_pain_rest,
    const float w_fatigue_rest,
    const float w_pain_attack,
    const float w_fear_attack,
    cudaStream_t stream
) {
    flat_dynamic_policy_food_birth_full_tick_persistent_kernel_v7<<<
        static_cast<unsigned int>(cells),
        kThreadsPerCell,
        0,
        stream
    >>>(
        active_agent_ids,
        active_count_state,
        slot_to_agent_id,
        agent_id_to_slot,
        alive_mask,
        slot_tokens,
        action_cache_actions,
        action_cache_token,
        action_cache_cursor,
        action_cache_next,
        pos_y,
        pos_x,
        agent_grid,
        food_grid,
        energy,
        health,
        injury,
        policy_energy_observation,
        rest_streak,
        pending_free_agent_ids,
        pending_free_slots,
        pending_free_count,
        energy_collected,
        energy_collected_exact,
        last_action,
        last_last_action,
        action_switches,
        action_oscillations,
        episode0_switches,
        episode0_oscillations,
        survival_steps,
        mlp_pack,
        drives,
        attack_bias,
        action_temperature,
        drive_logit_gain,
        food_gained_scratch,
        ordered_agent_ids_out,
        ordered_slots_out,
        observations_out,
        logits_out,
        row_actions_out,
        actions_by_slot_out,
        pre_energy_out,
        pre_health_out,
        pre_injury_out,
        pre_survival_steps_out,
        active_count_out,
        policy_refill_rows_out,
        policy_cache_hit_rows_out,
        death_agent_ids_out,
        death_slots_out,
        death_reason_codes_out,
        death_count_out,
        survivor_agent_ids_out,
        survivor_count_out,
        lifecycle_status_out,
        status_out,
        full_tick_state,
        full_tick_outputs,
        event_state,
        ordinary_birth_state,
        ordinary_birth_outputs,
        ordinary_birth_config,
        mechanics_config,
        ticks,
        active_capacity,
        agent_id_capacity,
        capacity,
        pending_capacity,
        height,
        width,
        view_radius,
        observation_size,
        hidden_size,
        action_count,
        packed_width,
        drive_width,
        cache_capacity,
        cache_width,
        deterministic_actions,
        max_food,
        effective_food_spawn_count,
        max_energy,
        start_health,
        move_cost,
        gather_cost,
        rest_cost,
        attack_cost,
        action_cost_mult,
        rest_cost_mult,
        food_energy,
        episode0_k,
        w_hunger_gather,
        w_pain_rest,
        w_fatigue_rest,
        w_pain_attack,
        w_fear_attack
    );
}

KernelResourceReceipt
query_flat_dynamic_policy_food_birth_full_tick_resources_v7() {
    cudaFuncAttributes attributes{};
    cudaError_t error = cudaFuncGetAttributes(
        &attributes,
        flat_dynamic_policy_food_birth_full_tick_persistent_kernel_v7
    );
    if (error != cudaSuccess) {
        return {-1, -1, -1, -1, -1, -1, -1, -1};
    }
    int32_t occupancy = -1;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy,
        flat_dynamic_policy_food_birth_full_tick_persistent_kernel_v7,
        kThreadsPerCell,
        0
    );
    if (error != cudaSuccess) {
        occupancy = -1;
    }
    return {
        attributes.binaryVersion,
        occupancy,
        attributes.maxThreadsPerBlock,
        attributes.numRegs,
        attributes.ptxVersion,
        static_cast<int64_t>(attributes.constSizeBytes),
        static_cast<int64_t>(attributes.localSizeBytes),
        static_cast<int64_t>(attributes.sharedSizeBytes),
    };
}

}  // namespace eiel::cuda_architecture_b::persistent_generation
