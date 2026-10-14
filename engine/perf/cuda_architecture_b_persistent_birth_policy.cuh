#ifndef EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_BIRTH_POLICY_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_BIRTH_POLICY_CUH_

#include <cuda_runtime.h>

#include <cstdint>

#include "cuda_architecture_b_torch_mt19937_replay_v1.cuh"
#include "cuda_architecture_b_torch_rng_state_pack_v1.cuh"

namespace eiel::cuda_architecture_b::persistent_birth_policy {

namespace torch_rng_pack =
    eiel::cuda_architecture_b::torch_rng_state_pack_v1;
namespace torch_rng_replay =
    eiel::cuda_architecture_b::torch_mt19937_replay_v1;

enum OrdinaryBirthPolicyStatus : int32_t {
    kStatusOk = 0,
    kStatusInvalidPlan = 1,
    kStatusTapeExhausted = 2,
};

// Every pointer is already offset to the simulation cell owned by the caller.
// Pointer validity, element types, and physical extents remain launch-boundary
// obligations, matching the host checks around the immutable Stage2 kernel.
struct FlatOrdinaryBirthPolicyState {
    const int64_t* slot_to_agent_id;
    const bool* alive_mask;
    float* mlp_pack;
    float* hmem;
    bool* policy_use_gru;
    uint8_t* mutation_torch_cpu_state;
    int64_t* mutation_torch_cpu_semantic_draws;
};

struct FlatOrdinaryBirthPolicyPlan {
    const int64_t* parent_slots;
    const int64_t* child_slots;
    const int64_t* planned_count;
};

struct FlatOrdinaryBirthPolicyTape {
    const float* mutation_delta;
    const bool* mutation_mask;
    const uint8_t* packed_post_torch_cpu_state;
    const int64_t* semantic_draws;
    int64_t* cursor;
};

// Per-birth arrays have plan_capacity rows. Every other output has one row.
struct FlatOrdinaryBirthPolicyOutputs {
    int64_t* committed_count;
    int64_t* semantic_draws_added;
    int64_t* mutation_events;
    int64_t* mutated_elements;
    double* mutation_delta_total;
    bool* mutation_occurred_by_birth;
    float* mutation_delta_by_birth;
    int32_t* status;
};

struct OrdinaryBirthPolicyConfig {
    int64_t plan_capacity;
    int64_t slot_count;
    int64_t mlp_row_width;
    int64_t hmem_width;
    int64_t torch_state_bytes;
    int64_t tape_capacity;
    int64_t mutation_tape_width;
};

struct StableGruBirthPolicyConfig {
    int64_t plan_capacity;
    int64_t slot_count;
    int64_t gru_row_width;
    int64_t hmem_width;
};

__device__ __forceinline__ void initialize_ordinary_birth_policy_outputs(
    const FlatOrdinaryBirthPolicyOutputs outputs,
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

// Exact ordinary (turnover_mode=false) raw-device transcription of
// native/cuda_architecture_b_child_policy.cu. Source mapping in that immutable
// Stage2 file:
//   56-67   plan/cursor validation and atomic tape-exhaustion return
//   69-121  ordinary parent order, slot, liveness, and uniqueness validation
//   123-161 taped mutation, MLP inheritance, hmem/GRU reset, and Torch state
//   163-170 cursor, semantic-draw, and mutation aggregate commits
// Stage2 lines 243-256 supplied fresh-zero outputs through Torch allocation;
// initialize_ordinary_birth_policy_outputs reproduces that behavior for reused
// caller-owned storage. Exactly one device thread invokes this allocation-free
// function for one simulation cell. It performs no synchronization.
// source_mlp_pack optionally keeps immutable parent reads separate from the
// state.mlp_pack child-row destination; null preserves in-place use.
__device__ __forceinline__ void inherit_ordinary_fixed_mlp_child_policy_serial(
    const FlatOrdinaryBirthPolicyState state,
    const FlatOrdinaryBirthPolicyPlan plan,
    const FlatOrdinaryBirthPolicyTape tape,
    const FlatOrdinaryBirthPolicyOutputs outputs,
    const OrdinaryBirthPolicyConfig config,
    const float* const source_mlp_pack = nullptr
) {
    initialize_ordinary_birth_policy_outputs(outputs, config.plan_capacity);

    const int64_t birth_count = plan.planned_count[0];
    if (
        birth_count < 0 || birth_count > config.plan_capacity
        || birth_count > config.slot_count || tape.cursor[0] < 0
        || config.torch_state_bytes != torch_rng_pack::kFullStateBytes
        || (
            config.mutation_tape_width != 1
            && config.mutation_tape_width != config.mlp_row_width
        )
    ) {
        outputs.status[0] = kStatusInvalidPlan;
        return;
    }
    if (tape.cursor[0] + birth_count > config.tape_capacity) {
        outputs.status[0] = kStatusTapeExhausted;
        return;
    }

    int64_t previous_parent_id = -1;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = plan.parent_slots[birth];
        const int64_t child_slot = plan.child_slots[birth];
        if (
            parent_slot < 0 || parent_slot >= config.slot_count
            || child_slot < 0 || child_slot >= config.slot_count
            || parent_slot == child_slot
            || !state.alive_mask[parent_slot]
            || !state.alive_mask[child_slot]
            || state.slot_to_agent_id[parent_slot] < 0
            || state.slot_to_agent_id[child_slot] < 0
            || state.policy_use_gru[parent_slot]
        ) {
            outputs.status[0] = kStatusInvalidPlan;
            return;
        }
        const int64_t parent_id = state.slot_to_agent_id[parent_slot];
        if (birth > 0 && parent_id <= previous_parent_id) {
            outputs.status[0] = kStatusInvalidPlan;
            return;
        }
        previous_parent_id = parent_id;
        for (int64_t previous = 0; previous < birth; ++previous) {
            if (plan.child_slots[previous] == child_slot) {
                outputs.status[0] = kStatusInvalidPlan;
                return;
            }
        }
    }

    const int64_t initial_cursor = tape.cursor[0];
    int64_t semantic_added = 0;
    int64_t event_count = 0;
    int64_t element_count = 0;
    double delta_total = 0.0;
    const bool replay_terminal_state =
        tape.packed_post_torch_cpu_state == nullptr;
    if (replay_terminal_state && config.mutation_tape_width != 1) {
        outputs.status[0] = kStatusInvalidPlan;
        return;
    }
    const float* const inheritance_mlp_pack = source_mlp_pack == nullptr
        ? state.mlp_pack
        : source_mlp_pack;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = plan.parent_slots[birth];
        const int64_t child_slot = plan.child_slots[birth];
        const int64_t episode = initial_cursor + birth;
        bool row_mutated = false;
        double row_delta_total = 0.0;
        for (int64_t column = 0; column < config.mlp_row_width; ++column) {
            const int64_t mutation_tape_index =
                episode * config.mutation_tape_width
                + (
                    config.mutation_tape_width == 1
                    ? 0
                    : column
                );
            const float parent_value = inheritance_mlp_pack[
                parent_slot * config.mlp_row_width + column
            ];
            const float delta = tape.mutation_delta[mutation_tape_index];
            if (
                replay_terminal_state
                && (tape.mutation_mask[mutation_tape_index] || delta != 0.0f)
            ) {
                outputs.status[0] = kStatusInvalidPlan;
                return;
            }
            state.mlp_pack[child_slot * config.mlp_row_width + column] =
                __fadd_rn(parent_value, delta);
            if (tape.mutation_mask[mutation_tape_index]) {
                if (replay_terminal_state) {
                    outputs.status[0] = kStatusInvalidPlan;
                    return;
                }
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
            if (
                tape.semantic_draws[episode] != config.mlp_row_width
                || delta_total != 0.0
            ) {
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

// Stable-GRU births inherit one complete parent parameter row and always reset
// recurrent memory. Policy-weight mutation is handled by the GRU-specific
// authority layer; this helper is the exact disabled-mutation path.
__device__ __forceinline__ void inherit_ordinary_stable_gru_child_policy_serial(
    const FlatOrdinaryBirthPolicyState state,
    float* const gru_pack,
    const FlatOrdinaryBirthPolicyPlan plan,
    const FlatOrdinaryBirthPolicyOutputs outputs,
    const StableGruBirthPolicyConfig config
) {
    initialize_ordinary_birth_policy_outputs(outputs, config.plan_capacity);
    const int64_t birth_count = plan.planned_count[0];
    if (
        birth_count < 0 || birth_count > config.plan_capacity
        || birth_count > config.slot_count || config.gru_row_width <= 0
        || config.hmem_width != 64 || gru_pack == nullptr
    ) {
        outputs.status[0] = kStatusInvalidPlan;
        return;
    }
    int64_t previous_parent_id = -1;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = plan.parent_slots[birth];
        const int64_t child_slot = plan.child_slots[birth];
        if (
            parent_slot < 0 || parent_slot >= config.slot_count
            || child_slot < 0 || child_slot >= config.slot_count
            || parent_slot == child_slot || !state.alive_mask[parent_slot]
            || !state.alive_mask[child_slot]
            || state.slot_to_agent_id[parent_slot] < 0
            || state.slot_to_agent_id[child_slot] < 0
            || !state.policy_use_gru[parent_slot]
            || !state.policy_use_gru[child_slot]
        ) {
            outputs.status[0] = kStatusInvalidPlan;
            return;
        }
        const int64_t parent_id = state.slot_to_agent_id[parent_slot];
        if (birth > 0 && parent_id <= previous_parent_id) {
            outputs.status[0] = kStatusInvalidPlan;
            return;
        }
        previous_parent_id = parent_id;
        for (int64_t previous = 0; previous < birth; ++previous) {
            if (plan.child_slots[previous] == child_slot) {
                outputs.status[0] = kStatusInvalidPlan;
                return;
            }
        }
    }
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = plan.parent_slots[birth];
        const int64_t child_slot = plan.child_slots[birth];
        for (int64_t column = 0; column < config.gru_row_width; ++column) {
            gru_pack[child_slot * config.gru_row_width + column] =
                gru_pack[parent_slot * config.gru_row_width + column];
        }
        for (int64_t column = 0; column < config.hmem_width; ++column) {
            state.hmem[child_slot * config.hmem_width + column] = 0.0f;
        }
    }
    outputs.committed_count[0] = birth_count;
    outputs.status[0] = kStatusOk;
}

}  // namespace eiel::cuda_architecture_b::persistent_birth_policy

#endif  // EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_BIRTH_POLICY_CUH_
