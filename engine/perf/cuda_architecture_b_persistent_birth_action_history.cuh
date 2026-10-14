#ifndef EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_BIRTH_ACTION_HISTORY_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_BIRTH_ACTION_HISTORY_CUH_

#include <cuda_runtime.h>

#include <cstdint>

namespace eiel::cuda_architecture_b::persistent_birth_action_history {

enum OrdinaryBirthActionHistoryStatus : int32_t {
    kStatusOk = 0,
    kStatusIdentityNotCommitted = 1,
    kStatusInvalidCommittedCount = 2,
    kStatusInvalidChildSlot = 3,
    kStatusDuplicateChildSlot = 4,
};

// Every pointer is already offset to the simulation cell owned by the caller.
// Pointer validity and physical slot-bank extents are launch-boundary
// obligations, matching the host checks around immutable Stage2. The six
// fields are the ordinary-birth action-history tensors at identity indices
// 29-34.
struct FlatOrdinaryBirthActionHistoryState {
    int32_t* last_action;
    int32_t* last_last_action;
    int32_t* action_switches;
    int32_t* action_oscillations;
    int32_t* episode0_switches;
    int32_t* episode0_oscillations;
};

// These are the successful identity transaction's committed outputs. The
// identity status is consumed explicitly because Stage2 gates the newborn
// mask on identity[6] == 0 before touching action history.
struct FlatCommittedOrdinaryBirthSlots {
    const int64_t* child_slots;
    const int64_t* committed_count;
    const int32_t* identity_status;
};

struct FlatOrdinaryBirthActionHistoryOutputs {
    int32_t* status;
};

struct OrdinaryBirthActionHistoryConfig {
    int64_t plan_capacity;
    int64_t slot_count;
};

// Exact allocation-free serial transcription of immutable Stage2
// native/cuda_architecture_b_birth_transaction.cpp:374-408:
//   374-394  derive the successful identity transaction's newborn slot mask
//   395-401  reset last_action and last_last_action to the -1 sentinel
//   402-408  zero ordinary and episode-0 switch/oscillation counters
// The source identity call ends at line 373 and scalar genome starts at 410.
// Exactly one device thread invokes this helper for one simulation cell after
// identity commit and before scalar-genome inheritance. Action-cache and fresh
// newborn selection-state resets are intentionally owned by their other stages.
// All inputs are preflighted before the first state write so malformed
// committed rows fail closed without a partial reset.
__device__ __forceinline__ void reset_ordinary_newborn_action_history_serial(
    const FlatOrdinaryBirthActionHistoryState state,
    const FlatCommittedOrdinaryBirthSlots committed,
    const FlatOrdinaryBirthActionHistoryOutputs outputs,
    const OrdinaryBirthActionHistoryConfig config
) {
    outputs.status[0] = kStatusOk;

    if (committed.identity_status[0] != 0) {
        outputs.status[0] = kStatusIdentityNotCommitted;
        return;
    }

    const int64_t birth_count = committed.committed_count[0];
    if (
        birth_count < 0 || birth_count > config.plan_capacity
        || birth_count > config.slot_count
    ) {
        outputs.status[0] = kStatusInvalidCommittedCount;
        return;
    }

    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t child_slot = committed.child_slots[birth];
        if (child_slot < 0 || child_slot >= config.slot_count) {
            outputs.status[0] = kStatusInvalidChildSlot;
            return;
        }
        for (int64_t previous = 0; previous < birth; ++previous) {
            if (committed.child_slots[previous] == child_slot) {
                outputs.status[0] = kStatusDuplicateChildSlot;
                return;
            }
        }
    }

    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t child_slot = committed.child_slots[birth];
        state.last_action[child_slot] = -1;
        state.last_last_action[child_slot] = -1;
        state.action_switches[child_slot] = 0;
        state.action_oscillations[child_slot] = 0;
        state.episode0_switches[child_slot] = 0;
        state.episode0_oscillations[child_slot] = 0;
    }
}

}  // namespace eiel::cuda_architecture_b::persistent_birth_action_history

#endif  // EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_BIRTH_ACTION_HISTORY_CUH_
