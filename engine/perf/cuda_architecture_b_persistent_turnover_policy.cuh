#ifndef EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_TURNOVER_POLICY_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_TURNOVER_POLICY_CUH_

#include <cuda_runtime.h>

#include <cstdint>

#include "cuda_architecture_b_persistent_birth_scalar_genome.cuh"
#include "cuda_architecture_b_torch_mt19937_replay_v1.cuh"
#include "cuda_architecture_b_torch_rng_state_pack_v1.cuh"
#include "../tw_rngv2/native/tw_rngv2.cuh"

namespace eiel::cuda_architecture_b::persistent_turnover_policy {

namespace torch_rng_pack =
    eiel::cuda_architecture_b::torch_rng_state_pack_v1;
namespace torch_rng_replay =
    eiel::cuda_architecture_b::torch_mt19937_replay_v1;

enum TurnoverPolicyStatus : int32_t {
    kStatusOk = 0,
    kStatusInvalidPlan = 1,
    kStatusTapeExhausted = 2,
};

// Every pointer is already offset to the simulation cell owned by the caller.
// Pointer validity, element types, and physical extents remain launch-boundary
// obligations, matching the host checks around the immutable Stage2 kernel.
struct FlatTurnoverPolicyState {
    const int64_t* slot_to_agent_id;
    const bool* alive_mask;
    float* mlp_pack;
    float* hmem;
    bool* policy_use_gru;
    uint8_t* mutation_torch_cpu_state;
    int64_t* mutation_torch_cpu_semantic_draws;
};

// Parent, mate, and child rows are supplied by the completed turnover refill
// plan. This helper validates and consumes them but performs no planning or
// identity assignment.
struct FlatTurnoverPolicyPlan {
    const int64_t* parent_slots;
    const int64_t* mate_slots;
    const int64_t* child_slots;
    const int64_t* planned_count;
};

struct FlatTurnoverPolicyTape {
    const float* mutation_delta;
    const bool* mutation_mask;
    const uint8_t* crossover_mask_packed;
    const uint8_t* packed_post_torch_cpu_state;
    const int64_t* semantic_draws;
    int64_t* cursor;
};

// Per-birth arrays have plan_capacity rows. Every other output has one row.
struct FlatTurnoverPolicyOutputs {
    int64_t* committed_count;
    int64_t* semantic_draws_added;
    int64_t* mutation_events;
    int64_t* mutated_elements;
    double* mutation_delta_total;
    bool* mutation_occurred_by_birth;
    float* mutation_delta_by_birth;
    int32_t* status;
};

struct TurnoverPolicyConfig {
    int64_t plan_capacity;
    int64_t slot_count;
    int64_t mlp_row_width;
    int64_t hmem_width;
    int64_t torch_state_bytes;
    int64_t tape_capacity;
    int64_t mutation_tape_width;
};

struct StableGruTurnoverPolicyConfig {
    int64_t plan_capacity;
    int64_t slot_count;
    int64_t gru_row_width;
    int64_t hmem_width;
};

__device__ __forceinline__ bool valid_turnover_policy_config(
    const TurnoverPolicyConfig config
) {
    return config.plan_capacity >= 0
        && config.slot_count > 0
        && config.mlp_row_width >= 0
        && config.hmem_width >= 0
        && config.torch_state_bytes == torch_rng_pack::kFullStateBytes
        && config.tape_capacity >= 0
        && (
            config.mutation_tape_width == 1
            || config.mutation_tape_width == config.mlp_row_width
        );
}

__device__ __forceinline__ void initialize_turnover_policy_outputs(
    const FlatTurnoverPolicyOutputs outputs,
    const int64_t plan_capacity
) {
    for (int64_t birth = 0; birth < plan_capacity; ++birth) {
        outputs.mutation_occurred_by_birth[birth] = false;
        outputs.mutation_delta_by_birth[birth] = 0.0f;
    }
    outputs.committed_count[0] = 0;
    outputs.semantic_draws_added[0] = 0;
    outputs.mutation_events[0] = 0;
    outputs.mutated_elements[0] = 0;
    outputs.mutation_delta_total[0] = 0.0;
    outputs.status[0] = kStatusOk;
}

// Exact turnover_mode=true raw-device transcription of immutable Stage2
// commit 250823059692a938992540e7042dc8660d0cefc4,
// native/cuda_architecture_b_child_policy.cu. Authority mapping:
//   56-67   count/cursor validation and atomic tape-exhaustion return
//   69-107  turnover parent, mate, child, liveness, and fixed-MLP validation
//   115-121 unique child-destination validation
//   123-161 crossover-mask inheritance, taped mutation, hmem/GRU reset, and
//           Torch post-state restoration
//   163-170 tape cursor, semantic counters, and mutation output commits
// Stage2 lines 373-386 supplied fresh-zero outputs through Torch allocation;
// initialize_turnover_policy_outputs reproduces that behavior for reused
// caller-owned storage. Exactly one device thread invokes this allocation-free
// generation-boundary helper for one simulation cell. It performs no planning,
// identity assignment, allocation, launch, or synchronization.
// source_mlp_pack optionally keeps immutable parent/mate reads separate from
// the state.mlp_pack child-row destination; null preserves in-place use.
__device__ __forceinline__ void inherit_fixed_mlp_turnover_child_policy_serial(
    const FlatTurnoverPolicyState state,
    const FlatTurnoverPolicyPlan plan,
    const FlatTurnoverPolicyTape tape,
    const FlatTurnoverPolicyOutputs outputs,
    const TurnoverPolicyConfig config,
    const float* const source_mlp_pack = nullptr,
    const bool compact_mlp_destination = false,
    const bool defer_mlp_destination = false,
    const bool host_authoritative_mlp = false
) {
    initialize_turnover_policy_outputs(
        outputs,
        config.plan_capacity >= 0 ? config.plan_capacity : 0
    );

    if (!valid_turnover_policy_config(config)) {
        outputs.status[0] = kStatusInvalidPlan;
        return;
    }

    const int64_t birth_count = plan.planned_count[0];
    const int64_t initial_cursor = tape.cursor[0];
    if (
        birth_count < 0 || birth_count > config.plan_capacity
        || birth_count > config.slot_count || initial_cursor < 0
    ) {
        outputs.status[0] = kStatusInvalidPlan;
        return;
    }
    if (
        initial_cursor > config.tape_capacity
        || birth_count > config.tape_capacity - initial_cursor
    ) {
        outputs.status[0] = kStatusTapeExhausted;
        return;
    }

    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = plan.parent_slots[birth];
        const int64_t mate_slot = plan.mate_slots[birth];
        const int64_t child_slot = plan.child_slots[birth];
        if (
            parent_slot < 0 || parent_slot >= config.slot_count
            || mate_slot < 0 || mate_slot >= config.slot_count
            || child_slot < 0 || child_slot >= config.slot_count
            || parent_slot == child_slot || mate_slot == parent_slot
            || mate_slot == child_slot
            || !state.alive_mask[parent_slot]
            || !state.alive_mask[mate_slot]
            || !state.alive_mask[child_slot]
            || state.slot_to_agent_id[parent_slot] < 0
            || state.slot_to_agent_id[mate_slot] < 0
            || state.slot_to_agent_id[child_slot] < 0
            || state.policy_use_gru[parent_slot]
            || state.policy_use_gru[mate_slot]
        ) {
            outputs.status[0] = kStatusInvalidPlan;
            return;
        }
        const int64_t parent_id = state.slot_to_agent_id[parent_slot];
        const int64_t mate_id = state.slot_to_agent_id[mate_slot];
        const int64_t child_id = state.slot_to_agent_id[child_slot];
        if (
            mate_id == parent_id || child_id == parent_id
            || child_id == mate_id
        ) {
            outputs.status[0] = kStatusInvalidPlan;
            return;
        }
        for (int64_t destination = 0; destination < birth_count; ++destination) {
            if (
                parent_slot == plan.child_slots[destination]
                || mate_slot == plan.child_slots[destination]
            ) {
                outputs.status[0] = kStatusInvalidPlan;
                return;
            }
        }
        for (int64_t previous = 0; previous < birth; ++previous) {
            if (plan.child_slots[previous] == child_slot) {
                outputs.status[0] = kStatusInvalidPlan;
                return;
            }
        }
    }

    int64_t semantic_added = 0;
    int64_t event_count = 0;
    int64_t element_count = 0;
    double delta_total = 0.0;
    const bool replay_terminal_state =
        tape.packed_post_torch_cpu_state == nullptr
        || config.mutation_tape_width == 1;
    if (replay_terminal_state && config.mutation_tape_width != 1) {
        outputs.status[0] = kStatusInvalidPlan;
        return;
    }
    const int64_t packed_crossover_row_bytes = (config.mlp_row_width + 7) / 8;
    const float* const inheritance_mlp_pack = source_mlp_pack == nullptr
        ? state.mlp_pack
        : source_mlp_pack;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = plan.parent_slots[birth];
        const int64_t mate_slot = plan.mate_slots[birth];
        const int64_t child_slot = plan.child_slots[birth];
        const int64_t episode = initial_cursor + birth;
        bool row_mutated = false;
        double row_delta_total = 0.0;
        for (int64_t column = 0; column < config.mlp_row_width; ++column) {
            const int64_t crossover_tape_index =
                episode * packed_crossover_row_bytes + column / 8;
            const bool inherit_parent = (
                (
                    tape.crossover_mask_packed[crossover_tape_index]
                    >> (column & 7)
                ) & 1U
            ) != 0U;
            const int64_t mutation_tape_index =
                episode * config.mutation_tape_width
                + (config.mutation_tape_width == 1 ? 0 : column);
            float inherited_value = 0.0f;
            if (!host_authoritative_mlp) {
                inherited_value = inherit_parent
                    ? inheritance_mlp_pack[
                        parent_slot * config.mlp_row_width + column
                    ]
                    : inheritance_mlp_pack[
                        mate_slot * config.mlp_row_width + column
                    ];
            }
            const float delta = tape.mutation_delta[mutation_tape_index];
            if (
                replay_terminal_state
                && (tape.mutation_mask[mutation_tape_index] || delta != 0.0f)
            ) {
                outputs.status[0] = kStatusInvalidPlan;
                return;
            }
            const int64_t destination_row = compact_mlp_destination
                ? birth
                : child_slot;
            if (!defer_mlp_destination && !host_authoritative_mlp) {
                state.mlp_pack[
                    destination_row * config.mlp_row_width + column
                ] = __fadd_rn(inherited_value, delta);
            }
            if (tape.mutation_mask[mutation_tape_index]) {
                row_mutated = true;
                element_count += 1;
                delta_total += static_cast<double>(delta);
                row_delta_total += static_cast<double>(delta);
            }
        }
        for (int64_t column = 0; column < config.hmem_width; ++column) {
            state.hmem[child_slot * config.hmem_width + column] = 0.0f;
        }
        state.policy_use_gru[child_slot] = false;
        outputs.mutation_occurred_by_birth[birth] = row_mutated;
        outputs.mutation_delta_by_birth[birth] =
            static_cast<float>(row_delta_total);
        semantic_added += tape.semantic_draws[episode];
        event_count += row_mutated ? 1 : 0;
        if (replay_terminal_state) {
            if (tape.semantic_draws[episode] != config.mlp_row_width) {
                outputs.status[0] = kStatusInvalidPlan;
                return;
            }
        } else if (!torch_rng_pack::copy_unpacked_state_v1(
                tape.packed_post_torch_cpu_state
                    + episode * torch_rng_pack::kPackedStateBytes,
                torch_rng_pack::kPackedStateBytes,
                state.mutation_torch_cpu_state,
                config.torch_state_bytes
            )) {
            outputs.status[0] = kStatusInvalidPlan;
            return;
        }
    }
    if (
        replay_terminal_state
        && !torch_rng_replay::advance_canonical_state_v1(
            state.mutation_torch_cpu_state,
            config.torch_state_bytes,
            static_cast<std::uint64_t>(semantic_added)
        )
    ) {
        outputs.status[0] = kStatusInvalidPlan;
        return;
    }

    tape.cursor[0] = initial_cursor + birth_count;
    state.mutation_torch_cpu_semantic_draws[0] += semantic_added;
    outputs.committed_count[0] = birth_count;
    outputs.semantic_draws_added[0] = semantic_added;
    outputs.mutation_events[0] = event_count;
    outputs.mutated_elements[0] = element_count;
    outputs.mutation_delta_total[0] = delta_total;
    outputs.status[0] = kStatusOk;
}

// RNG V2 owns policy crossover through compact deterministic coordinates.
// This path deliberately has no future tape, cursor, mutable RNG state, CPU
// generator, or host fallback. Policy-weight mutation remains fail-closed at
// source preparation until an explicit native RNG V2 mutation contract exists.
__device__ __forceinline__ void inherit_fixed_mlp_turnover_child_policy_rngv2_serial(
    const FlatTurnoverPolicyState state,
    const FlatTurnoverPolicyPlan plan,
    const FlatTurnoverPolicyOutputs outputs,
    const TurnoverPolicyConfig config,
    const uint64_t rng_namespace_key,
    const uint32_t world_uid,
    const uint32_t semantic_epoch,
    const uint64_t absolute_tick,
    const float* const source_mlp_pack = nullptr,
    const bool compact_mlp_destination = false,
    const bool defer_mlp_destination = false
) {
    initialize_turnover_policy_outputs(
        outputs,
        config.plan_capacity >= 0 ? config.plan_capacity : 0
    );
    if (
        !valid_turnover_policy_config(config)
        || config.tape_capacity != 0
        || config.mutation_tape_width != 1
        || config.mlp_row_width > 65536
    ) {
        outputs.status[0] = kStatusInvalidPlan;
        return;
    }
    const int64_t birth_count = plan.planned_count[0];
    if (
        birth_count < 0 || birth_count > config.plan_capacity
        || birth_count > config.slot_count
    ) {
        outputs.status[0] = kStatusInvalidPlan;
        return;
    }
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = plan.parent_slots[birth];
        const int64_t mate_slot = plan.mate_slots[birth];
        const int64_t child_slot = plan.child_slots[birth];
        if (
            parent_slot < 0 || parent_slot >= config.slot_count
            || mate_slot < 0 || mate_slot >= config.slot_count
            || child_slot < 0 || child_slot >= config.slot_count
            || parent_slot == child_slot || mate_slot == parent_slot
            || mate_slot == child_slot
            || !state.alive_mask[parent_slot]
            || !state.alive_mask[mate_slot]
            || !state.alive_mask[child_slot]
            || state.slot_to_agent_id[parent_slot] < 0
            || state.slot_to_agent_id[mate_slot] < 0
            || state.slot_to_agent_id[child_slot] < 0
            || state.policy_use_gru[parent_slot]
            || state.policy_use_gru[mate_slot]
        ) {
            outputs.status[0] = kStatusInvalidPlan;
            return;
        }
        const int64_t parent_id = state.slot_to_agent_id[parent_slot];
        const int64_t mate_id = state.slot_to_agent_id[mate_slot];
        const int64_t child_id = state.slot_to_agent_id[child_slot];
        if (
            mate_id == parent_id || child_id == parent_id
            || child_id == mate_id
        ) {
            outputs.status[0] = kStatusInvalidPlan;
            return;
        }
        for (int64_t destination = 0; destination < birth_count; ++destination) {
            if (
                parent_slot == plan.child_slots[destination]
                || mate_slot == plan.child_slots[destination]
            ) {
                outputs.status[0] = kStatusInvalidPlan;
                return;
            }
        }
        for (int64_t previous = 0; previous < birth; ++previous) {
            if (plan.child_slots[previous] == child_slot) {
                outputs.status[0] = kStatusInvalidPlan;
                return;
            }
        }
    }

    const float* const inheritance_mlp_pack = source_mlp_pack == nullptr
        ? state.mlp_pack
        : source_mlp_pack;
    int64_t semantic_added = 0;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = plan.parent_slots[birth];
        const int64_t mate_slot = plan.mate_slots[birth];
        const int64_t child_slot = plan.child_slots[birth];
        const uint64_t child_id = static_cast<uint64_t>(
            state.slot_to_agent_id[child_slot]
        );
        for (int64_t column = 0; column < config.mlp_row_width; ++column) {
            const int32_t field_status = tw_rngv2::validate_fields(
                world_uid,
                tw_rngv2::DOMAIN_POLICY_BIRTH_DRAW,
                semantic_epoch,
                child_id,
                absolute_tick,
                static_cast<uint32_t>(column),
                0u
            );
            if (field_status != tw_rngv2::RNG_OK) {
                outputs.status[0] = kStatusInvalidPlan;
                return;
            }
            const uint32_t word = tw_rngv2::word_at(
                rng_namespace_key,
                world_uid,
                tw_rngv2::DOMAIN_POLICY_BIRTH_DRAW,
                semantic_epoch,
                child_id,
                absolute_tick,
                static_cast<uint32_t>(column),
                0
            );
            const bool inherit_parent =
                tw_rngv2::u32_to_f32_half_open(word) < 0.5f;
            const float inherited_value = inherit_parent
                ? inheritance_mlp_pack[
                    parent_slot * config.mlp_row_width + column
                ]
                : inheritance_mlp_pack[
                    mate_slot * config.mlp_row_width + column
                ];
            const int64_t destination_row = compact_mlp_destination
                ? birth
                : child_slot;
            if (!defer_mlp_destination) {
                state.mlp_pack[
                    destination_row * config.mlp_row_width + column
                ] = inherited_value;
            }
        }
        for (int64_t column = 0; column < config.hmem_width; ++column) {
            state.hmem[child_slot * config.hmem_width + column] = 0.0f;
        }
        state.policy_use_gru[child_slot] = false;
        semantic_added += config.mlp_row_width;
    }
    outputs.committed_count[0] = birth_count;
    outputs.semantic_draws_added[0] = semantic_added;
    outputs.status[0] = kStatusOk;
}

__device__ __forceinline__ bool materialize_fixed_mlp_turnover_policy_rows_rngv2_v1(
    float* const destination_mlp_pack,
    const float* const source_mlp_pack,
    const int64_t* const slot_to_agent_id,
    const FlatTurnoverPolicyPlan plan,
    const TurnoverPolicyConfig config,
    const uint64_t rng_namespace_key,
    const uint32_t world_uid,
    const uint32_t semantic_epoch,
    const uint64_t absolute_tick,
    const int64_t lane,
    const int64_t lanes
) {
    if (
        destination_mlp_pack == nullptr || source_mlp_pack == nullptr
        || slot_to_agent_id == nullptr || plan.parent_slots == nullptr
        || plan.mate_slots == nullptr || plan.child_slots == nullptr
        || plan.planned_count == nullptr || lane < 0 || lanes <= 0
        || !valid_turnover_policy_config(config)
        || config.tape_capacity != 0 || config.mutation_tape_width != 1
        || config.mlp_row_width > 65536
    ) {
        return false;
    }
    const int64_t birth_count = plan.planned_count[0];
    if (
        birth_count < 0 || birth_count > config.plan_capacity
        || birth_count > config.slot_count
    ) {
        return false;
    }
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = plan.parent_slots[birth];
        const int64_t mate_slot = plan.mate_slots[birth];
        const int64_t child_slot = plan.child_slots[birth];
        if (
            parent_slot < 0 || parent_slot >= config.slot_count
            || mate_slot < 0 || mate_slot >= config.slot_count
            || child_slot < 0 || child_slot >= config.slot_count
            || slot_to_agent_id[child_slot] < 0
        ) {
            return false;
        }
        const uint64_t child_id = static_cast<uint64_t>(
            slot_to_agent_id[child_slot]
        );
        for (int64_t column = lane; column < config.mlp_row_width; column += lanes) {
            if (tw_rngv2::validate_fields(
                    world_uid,
                    tw_rngv2::DOMAIN_POLICY_BIRTH_DRAW,
                    semantic_epoch,
                    child_id,
                    absolute_tick,
                    static_cast<uint32_t>(column),
                    0u
                ) != tw_rngv2::RNG_OK) {
                return false;
            }
            const uint32_t word = tw_rngv2::word_at(
                rng_namespace_key,
                world_uid,
                tw_rngv2::DOMAIN_POLICY_BIRTH_DRAW,
                semantic_epoch,
                child_id,
                absolute_tick,
                static_cast<uint32_t>(column),
                0
            );
            const bool inherit_parent =
                tw_rngv2::u32_to_f32_half_open(word) < 0.5f;
            destination_mlp_pack[
                child_slot * config.mlp_row_width + column
            ] = inherit_parent
                ? source_mlp_pack[
                    parent_slot * config.mlp_row_width + column
                ]
                : source_mlp_pack[
                    mate_slot * config.mlp_row_width + column
                ];
        }
    }
    return true;
}

__device__ __forceinline__ bool materialize_fixed_mlp_turnover_policy_rows_v1(
    float* const destination_mlp_pack,
    const float* const source_mlp_pack,
    const FlatTurnoverPolicyPlan plan,
    const FlatTurnoverPolicyTape tape,
    const TurnoverPolicyConfig config,
    const int64_t initial_cursor,
    const int64_t lane,
    const int64_t lanes,
    const bool host_authoritative_mlp = false
) {
    if (
        destination_mlp_pack == nullptr || source_mlp_pack == nullptr
        || tape.mutation_delta == nullptr || tape.crossover_mask_packed == nullptr
        || plan.parent_slots == nullptr || plan.mate_slots == nullptr
        || plan.child_slots == nullptr || plan.planned_count == nullptr
        || lane < 0 || lanes <= 0 || initial_cursor < 0
        || !valid_turnover_policy_config(config)
    ) {
        return false;
    }
    const int64_t birth_count = plan.planned_count[0];
    if (
        birth_count < 0 || birth_count > config.plan_capacity
        || birth_count > config.slot_count
        || initial_cursor > config.tape_capacity
        || birth_count > config.tape_capacity - initial_cursor
    ) {
        return false;
    }
    const int64_t packed_crossover_row_bytes = (config.mlp_row_width + 7) / 8;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = plan.parent_slots[birth];
        const int64_t mate_slot = plan.mate_slots[birth];
        const int64_t child_slot = plan.child_slots[birth];
        if (
            parent_slot < 0 || parent_slot >= config.slot_count
            || mate_slot < 0 || mate_slot >= config.slot_count
            || child_slot < 0 || child_slot >= config.slot_count
            || parent_slot == child_slot || mate_slot == parent_slot
            || mate_slot == child_slot
        ) {
            return false;
        }
        const int64_t episode = initial_cursor + birth;
        for (int64_t column = lane; column < config.mlp_row_width; column += lanes) {
            const int64_t crossover_tape_index =
                episode * packed_crossover_row_bytes + column / 8;
            const bool inherit_parent = (
                (
                    tape.crossover_mask_packed[crossover_tape_index]
                    >> (column & 7)
                ) & 1U
            ) != 0U;
            const int64_t mutation_tape_index =
                episode * config.mutation_tape_width
                + (config.mutation_tape_width == 1 ? 0 : column);
            float inherited_value = 0.0f;
            if (!host_authoritative_mlp) {
                inherited_value = inherit_parent
                    ? source_mlp_pack[parent_slot * config.mlp_row_width + column]
                    : source_mlp_pack[mate_slot * config.mlp_row_width + column];
            }
            const float delta = tape.mutation_delta[mutation_tape_index];
            if (!host_authoritative_mlp) {
                destination_mlp_pack[
                    child_slot * config.mlp_row_width + column
                ] = __fadd_rn(inherited_value, delta);
            }
        }
    }
    return true;
}

// Fixed-GRU turnover uses the same selected parent/mate rows as scalar
// inheritance. The donor draw is part of the authoritative CPython RNG stream.
// Parameter rows are copied by the publish kernel only after every turnover
// stage succeeds, preserving the transaction boundary.
__device__ __forceinline__ void prepare_stable_gru_turnover_child_policy_serial(
    const FlatTurnoverPolicyState state,
    const FlatTurnoverPolicyPlan plan,
    persistent_birth_scalar_genome::FlatCpythonRandomState python_random,
    int64_t* const donor_slots,
    const FlatTurnoverPolicyOutputs outputs,
    const StableGruTurnoverPolicyConfig config
) {
    initialize_turnover_policy_outputs(outputs, config.plan_capacity);
    const int64_t birth_count = plan.planned_count[0];
    if (
        birth_count < 0 || birth_count > config.plan_capacity
        || birth_count > config.slot_count || config.gru_row_width <= 0
        || config.hmem_width != 64 || donor_slots == nullptr
        || python_random.python_random_words == nullptr
        || python_random.python_random_index == nullptr
        || python_random.python_random_raw_draws == nullptr
        || python_random.mutation_py_semantic_draws == nullptr
        || python_random.python_random_index[0] < 0
        || python_random.python_random_index[0] > 624
        || python_random.python_random_raw_draws[0] < 0
        || python_random.mutation_py_semantic_draws[0] < 0
    ) {
        outputs.status[0] = kStatusInvalidPlan;
        return;
    }
    for (int64_t birth = 0; birth < config.plan_capacity; ++birth) {
        donor_slots[birth] = -1;
    }
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = plan.parent_slots[birth];
        const int64_t mate_slot = plan.mate_slots[birth];
        const int64_t child_slot = plan.child_slots[birth];
        if (
            parent_slot < 0 || parent_slot >= config.slot_count
            || mate_slot < 0 || mate_slot >= config.slot_count
            || child_slot < 0 || child_slot >= config.slot_count
            || parent_slot == mate_slot || parent_slot == child_slot
            || mate_slot == child_slot || !state.alive_mask[parent_slot]
            || !state.alive_mask[mate_slot] || !state.alive_mask[child_slot]
            || state.slot_to_agent_id[parent_slot] < 0
            || state.slot_to_agent_id[mate_slot] < 0
            || state.slot_to_agent_id[child_slot] < 0
            || !state.policy_use_gru[parent_slot]
            || !state.policy_use_gru[mate_slot]
            || !state.policy_use_gru[child_slot]
        ) {
            outputs.status[0] = kStatusInvalidPlan;
            return;
        }
    }

    persistent_birth_scalar_genome::DeviceCpythonRandom rng{
        python_random.python_random_words,
        python_random.python_random_index[0],
        python_random.python_random_has_gauss[0],
        python_random.python_random_gauss_next[0],
        python_random.python_random_gauss_primary_tape,
        python_random.python_random_gauss_secondary_tape,
        0,
        python_random.python_random_raw_draws[0],
    };
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = plan.parent_slots[birth];
        const int64_t mate_slot = plan.mate_slots[birth];
        donor_slots[birth] = rng.random_double() < 0.5
            ? parent_slot
            : mate_slot;
        const int64_t child_slot = plan.child_slots[birth];
        for (int64_t column = 0; column < config.hmem_width; ++column) {
            state.hmem[child_slot * config.hmem_width + column] = 0.0f;
        }
    }
    python_random.python_random_index[0] = rng.index;
    python_random.python_random_raw_draws[0] = rng.raw_draws;
    python_random.mutation_py_semantic_draws[0] += birth_count;
    outputs.committed_count[0] = birth_count;
    outputs.status[0] = kStatusOk;
}

}  // namespace eiel::cuda_architecture_b::persistent_turnover_policy

#endif  // EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_TURNOVER_POLICY_CUH_
