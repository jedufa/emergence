#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "cuda_architecture_b_birth_parent_order_v1.cuh"
#include "cuda_architecture_b_persistent_birth_policy.cuh"
#include "cuda_architecture_b_torch_mt19937_replay_v1.cuh"

namespace eiel::cuda_architecture_b::flattened_birth_policy_v2 {

namespace persistent_policy =
    eiel::cuda_architecture_b::persistent_birth_policy;
namespace parent_order =
    eiel::cuda_architecture_b::birth_parent_order_v1;
namespace torch_rng_pack =
    eiel::cuda_architecture_b::torch_rng_state_pack_v1;
namespace torch_rng_replay =
    eiel::cuda_architecture_b::torch_mt19937_replay_v1;

constexpr int32_t kStatusOk = persistent_policy::kStatusOk;
constexpr int32_t kStatusInvalidPlan = persistent_policy::kStatusInvalidPlan;
constexpr int32_t kStatusTapeExhausted =
    persistent_policy::kStatusTapeExhausted;

static_assert(kStatusOk == 0);
static_assert(kStatusInvalidPlan == 1);
static_assert(kStatusTapeExhausted == 2);

// Allocation-free external-scratch transcription of
// persistent_policy::inherit_ordinary_fixed_mlp_child_policy_serial for zero
// through a runtime-sized ordered ordinary-birth plan. All input pointers are already offset
// to one simulation cell and are read-only. slot_to_agent_id and alive_mask
// must expose the identity-committed view used by the persistent helper; that
// view may itself live in transaction scratch. The MLP, policy-mode, Torch RNG,
// and tape inputs remain authoritative. Child MLP and hmem scratch rows are
// packed by birth ordinal, not by child slot. The hmem rows are reset to zero;
// parent hmem is intentionally not read, matching the persistent helper's
// fixed-MLP newborn semantics.
//
// Scratch extents are:
//   scratch_child_mlp_rows                    [2, mlp_row_width]
//   scratch_child_hmem_rows                   [2, hmem_width]
//   scratch_final_mutation_torch_cpu_state    [torch_state_bytes]
//   scratch_mutation_occurred_by_birth        [2]
//   scratch_mutation_delta_by_birth           [2]
// Only child rows [0, birth_count) are written. Both mutation-output rows are
// initialized on success, as if the persistent helper had plan_capacity=2.
// The final Torch state is the authoritative input state for zero births and
// the last consumed episode's post-state otherwise.
//
// Pointer validity, physical extents, nonnegative dimensions/capacities, and
// non-overlap between authoritative inputs and scratch outputs are
// launch-boundary obligations. On validation or tape-exhaustion failure, only
// scratch_status is changed. No authoritative state is ever mutated.
__device__ __forceinline__ void
prepare_ordinary_fixed_mlp_child_policy_external_scratch_v2(
    const int64_t* __restrict__ slot_to_agent_id,
    const bool* __restrict__ alive_mask,
    const float* __restrict__ authoritative_mlp_pack,
    const bool host_authoritative_mlp,
    const bool* __restrict__ policy_use_gru,
    const uint8_t* __restrict__ authoritative_mutation_torch_cpu_state,
    const int64_t authoritative_mutation_torch_cpu_semantic_draws,
    const float* __restrict__ policy_birth_delta_tape,
    const bool* __restrict__ policy_birth_mutation_mask_tape,
    const uint8_t* __restrict__ policy_birth_post_rng_state_packed_tape,
    const int64_t* __restrict__ policy_birth_semantic_draws_tape,
    const int64_t authoritative_policy_birth_tape_cursor,
    const int64_t* __restrict__ planned_parent_slots,
    const int64_t* __restrict__ planned_child_slots,
    const int64_t* __restrict__ authoritative_parent_slots,
    const int64_t birth_count,
    const int64_t slot_count,
    const int64_t authoritative_mlp_slot_count,
    const int64_t mlp_row_width,
    const int64_t hmem_width,
    const int64_t torch_state_bytes,
    const int64_t policy_tape_capacity,
    const int64_t policy_mutation_tape_width,
    float* __restrict__ scratch_child_mlp_rows,
    float* __restrict__ scratch_child_hmem_rows,
    uint8_t* __restrict__ scratch_final_mutation_torch_cpu_state,
    int64_t& scratch_final_policy_birth_tape_cursor,
    int64_t& scratch_final_mutation_torch_cpu_semantic_draws,
    int64_t& scratch_committed_count,
    int64_t& scratch_semantic_draws_added,
    int64_t& scratch_mutation_events,
    int64_t& scratch_mutated_elements,
    double& scratch_mutation_delta_total,
    bool* __restrict__ scratch_mutation_occurred_by_birth,
    float* __restrict__ scratch_mutation_delta_by_birth,
    int32_t& scratch_status,
    const parent_order::BirthParentOrderModeV1 parent_order_mode =
        parent_order::BirthParentOrderModeV1::ascending_agent_id_v1
) {
    if (birth_count < 0 || birth_count > slot_count) {
        scratch_status = kStatusInvalidPlan;
        return;
    }
    if (policy_tape_capacity == 0) {
        if (host_authoritative_mlp || authoritative_mlp_pack == nullptr) {
            scratch_status = kStatusInvalidPlan;
            return;
        }
        scratch_committed_count = 0;
        scratch_semantic_draws_added = 0;
        scratch_mutation_events = 0;
        scratch_mutated_elements = 0;
        scratch_mutation_delta_total = 0.0;
        scratch_final_policy_birth_tape_cursor = 0;
        scratch_final_mutation_torch_cpu_semantic_draws =
            authoritative_mutation_torch_cpu_semantic_draws;
        for (int64_t index = 0; index < torch_state_bytes; ++index) {
            scratch_final_mutation_torch_cpu_state[index] =
                authoritative_mutation_torch_cpu_state[index];
        }
        for (int64_t birth = 0; birth < birth_count; ++birth) {
            const int64_t parent_slot = authoritative_parent_slots[birth];
            if (
                parent_slot < 0
                || parent_slot >= authoritative_mlp_slot_count
            ) {
                scratch_status = kStatusInvalidPlan;
                return;
            }
            const float* const parent_row =
                authoritative_mlp_pack + parent_slot * mlp_row_width;
            float* const child_row =
                scratch_child_mlp_rows + birth * mlp_row_width;
            for (int64_t column = 0; column < mlp_row_width; ++column) {
                child_row[column] = parent_row[column];
            }
            if (scratch_child_hmem_rows != nullptr && hmem_width > 0) {
                float* const child_hmem =
                    scratch_child_hmem_rows + birth * hmem_width;
                for (int64_t column = 0; column < hmem_width; ++column) {
                    child_hmem[column] = 0.0f;
                }
            }
            scratch_mutation_occurred_by_birth[birth] = false;
            scratch_mutation_delta_by_birth[birth] = 0.0f;
            scratch_committed_count += 1;
        }
        scratch_status = kStatusOk;
        return;
    }
    if (
        authoritative_policy_birth_tape_cursor < 0
        || torch_state_bytes != torch_rng_pack::kFullStateBytes
        || (
            policy_mutation_tape_width != 1
            && policy_mutation_tape_width != mlp_row_width
        )
    ) {
        scratch_status = kStatusInvalidPlan;
        return;
    }
    if (
        authoritative_policy_birth_tape_cursor > policy_tape_capacity
        || birth_count
            > policy_tape_capacity - authoritative_policy_birth_tape_cursor
    ) {
        scratch_status = kStatusTapeExhausted;
        return;
    }
    const bool replay_terminal_state =
        policy_birth_post_rng_state_packed_tape == nullptr;
    if (replay_terminal_state && policy_mutation_tape_width != 1) {
        scratch_status = kStatusInvalidPlan;
        return;
    }

    int64_t previous_parent_id = -1;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = planned_parent_slots[birth];
        const int64_t child_slot = planned_child_slots[birth];
        const int64_t authoritative_parent_slot =
            authoritative_parent_slots[birth];
        if (
            parent_slot < 0 || parent_slot >= slot_count
            || child_slot < 0 || child_slot >= slot_count
            || authoritative_parent_slot < 0
            || authoritative_parent_slot >= authoritative_mlp_slot_count
            || parent_slot == child_slot
            || !alive_mask[parent_slot]
            || !alive_mask[child_slot]
            || slot_to_agent_id[parent_slot] < 0
            || slot_to_agent_id[child_slot] < 0
            || policy_use_gru[parent_slot]
        ) {
            scratch_status = kStatusInvalidPlan;
            return;
        }
        const int64_t parent_id = slot_to_agent_id[parent_slot];
        if (!parent_order::valid_parent_ordinal_v1(
                parent_order_mode,
                birth,
                parent_id,
                previous_parent_id,
                planned_parent_slots,
                slot_to_agent_id
            )) {
            scratch_status = kStatusInvalidPlan;
            return;
        }
        previous_parent_id = parent_id;
        for (int64_t previous = 0; previous < birth; ++previous) {
            if (planned_child_slots[previous] == child_slot) {
                scratch_status = kStatusInvalidPlan;
                return;
            }
        }
    }

    scratch_committed_count = 0;
    scratch_semantic_draws_added = 0;
    scratch_mutation_events = 0;
    scratch_mutated_elements = 0;
    scratch_mutation_delta_total = 0.0;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        scratch_mutation_occurred_by_birth[birth] = false;
        scratch_mutation_delta_by_birth[birth] = 0.0f;
    }
    for (int64_t index = 0; index < torch_state_bytes; ++index) {
        scratch_final_mutation_torch_cpu_state[index] =
            authoritative_mutation_torch_cpu_state[index];
    }

    int64_t semantic_added = 0;
    int64_t event_count = 0;
    int64_t element_count = 0;
    double delta_total = 0.0;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = authoritative_parent_slots[birth];
        const int64_t episode =
            authoritative_policy_birth_tape_cursor + birth;
        bool row_mutated = false;
        double row_delta_total = 0.0;
        for (int64_t column = 0; column < mlp_row_width; ++column) {
            const int64_t mutation_tape_index =
                episode * policy_mutation_tape_width
                + (policy_mutation_tape_width == 1 ? 0 : column);
            const float delta =
                policy_birth_delta_tape[mutation_tape_index];
            if (
                replay_terminal_state
                && (
                    policy_birth_mutation_mask_tape[mutation_tape_index]
                    || delta != 0.0f
                )
            ) {
                scratch_status = kStatusInvalidPlan;
                return;
            }
            // HAPB executes this exact add in its bounded host-authoritative
            // workbench at the prepare/commit boundary.  Preserve every
            // resident tape/RNG accounting action here, but never dereference
            // the nonresident ABI sentinel as a policy bank.
            if (!host_authoritative_mlp) {
                const float parent_value = authoritative_mlp_pack[
                    parent_slot * mlp_row_width + column
                ];
                scratch_child_mlp_rows[birth * mlp_row_width + column] =
                    __fadd_rn(parent_value, delta);
            }
            if (policy_birth_mutation_mask_tape[mutation_tape_index]) {
                row_mutated = true;
                ++element_count;
                delta_total += static_cast<double>(delta);
                row_delta_total += static_cast<double>(delta);
            }
        }
        for (int64_t column = 0; column < hmem_width; ++column) {
            scratch_child_hmem_rows[birth * hmem_width + column] = 0.0f;
        }
        scratch_mutation_occurred_by_birth[birth] = row_mutated;
        scratch_mutation_delta_by_birth[birth] =
            static_cast<float>(row_delta_total);
        semantic_added += policy_birth_semantic_draws_tape[episode];
        event_count += row_mutated ? 1 : 0;
        if (replay_terminal_state) {
            if (
                policy_birth_semantic_draws_tape[episode] != mlp_row_width
                || row_mutated
            ) {
                scratch_status = kStatusInvalidPlan;
                return;
            }
        } else if (!torch_rng_pack::copy_unpacked_state_v1(
                policy_birth_post_rng_state_packed_tape
                    + episode * torch_rng_pack::kPackedStateBytes,
                torch_rng_pack::kPackedStateBytes,
                scratch_final_mutation_torch_cpu_state,
                torch_state_bytes
            )) {
            scratch_status = kStatusInvalidPlan;
            return;
        }
    }
    if (
        replay_terminal_state
        && !torch_rng_replay::advance_canonical_state_v1(
            scratch_final_mutation_torch_cpu_state,
            torch_state_bytes,
            static_cast<std::uint64_t>(semantic_added)
        )
    ) {
        scratch_status = kStatusInvalidPlan;
        return;
    }

    scratch_final_policy_birth_tape_cursor =
        authoritative_policy_birth_tape_cursor + birth_count;
    scratch_final_mutation_torch_cpu_semantic_draws =
        authoritative_mutation_torch_cpu_semantic_draws + semantic_added;
    scratch_committed_count = birth_count;
    scratch_semantic_draws_added = semantic_added;
    scratch_mutation_events = event_count;
    scratch_mutated_elements = element_count;
    scratch_mutation_delta_total = delta_total;
    scratch_status = kStatusOk;
}

}  // namespace eiel::cuda_architecture_b::flattened_birth_policy_v2
