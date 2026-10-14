#ifndef EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_TURNOVER_NEWBORN_STATE_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_TURNOVER_NEWBORN_STATE_CUH_

#include <cuda_runtime.h>

#include <cstdint>

namespace eiel::cuda_architecture_b::persistent_turnover_newborn_state {

template <typename T>
struct FlatStridedSlotField {
    T* data;
    int64_t element_stride;

    __device__ __forceinline__ T& at(const int64_t slot) const {
        return data[slot * element_stride];
    }
};

template <typename T>
struct FlatStridedRowField {
    T* data;
    int64_t slot_stride;
    int64_t element_stride;

    __device__ __forceinline__ T& at(
        const int64_t slot,
        const int64_t element
    ) const {
        return data[slot * slot_stride + element * element_stride];
    }
};

// The first 18 fields are the immutable Stage2 turnover_state fields in their
// generation-owner order. The remaining fields are the separately-owned
// action history, action cache, drive history, and reproduction statistics.
struct FlatTurnoverNewbornState {
    FlatStridedSlotField<float> energy_collected;
    FlatStridedSlotField<double> energy_collected_exact;
    FlatStridedSlotField<int32_t> attacks;
    FlatStridedSlotField<int32_t> early_survival_steps;
    FlatStridedSlotField<float> early_energy_collected;
    FlatStridedSlotField<double> early_energy_collected_exact;
    FlatStridedSlotField<float> early_injury_gained;
    FlatStridedSlotField<float> early_life_weight;
    FlatStridedSlotField<float> pred_err_sum;
    FlatStridedSlotField<int32_t> pred_err_count;
    FlatStridedSlotField<int32_t> first_repro_step;
    FlatStridedSlotField<bool> dev_stage_initialized;
    FlatStridedSlotField<float> episode0_pred_err_sum;
    FlatStridedSlotField<int32_t> episode0_pred_err_count;
    FlatStridedSlotField<float> late_pred_err_sum;
    FlatStridedSlotField<int32_t> late_pred_err_count;
    FlatStridedSlotField<float> chemistry_observation_fidelity;
    FlatStridedSlotField<float> chemistry_observation_bias;

    FlatStridedSlotField<int32_t> last_action;
    FlatStridedSlotField<int32_t> last_last_action;
    FlatStridedSlotField<int32_t> action_switches;
    FlatStridedSlotField<int32_t> action_oscillations;
    FlatStridedSlotField<int32_t> episode0_switches;
    FlatStridedSlotField<int32_t> episode0_oscillations;
    FlatStridedRowField<int32_t> action_cache_actions;
    FlatStridedRowField<float> prev_drive_delta;
    FlatStridedSlotField<float> drive_volatility;
    FlatStridedSlotField<float> drive_jitter;
    FlatStridedSlotField<float> episode0_drive_jitter_sum;
    FlatStridedSlotField<float> last_energy_delta;
    FlatStridedSlotField<float> last_health_delta;
    FlatStridedSlotField<float> last_injury_delta;
    FlatStridedSlotField<int32_t> selfharm_events;
    FlatStridedSlotField<int32_t> episode0_selfharm_events;
    FlatStridedSlotField<int32_t> offspring_count;
};

struct TurnoverNewbornStateConfig {
    int64_t action_cache_width;
    int64_t drive_width;
};

enum TurnoverReproductionStatus : int32_t {
    kStatusOk = 0,
    kStatusInvalidConfig = 1,
    kStatusInvalidCommittedCount = 2,
    kStatusInvalidParentSlot = 3,
    kStatusInvalidMateSlot = 4,
    kStatusParentMateAlias = 5,
    kStatusInvalidChildSlot = 6,
    kStatusDuplicateChildSlot = 7,
};

// These rows are the already-validated turnover identity rows. The child rows
// are retained for fail-closed validation before parent counters are changed.
struct FlatTurnoverReproductionRows {
    const int64_t* parent_slots;
    const int64_t* mate_slots;
    const int64_t* child_slots;
    const int64_t* committed_count;
};

struct FlatTurnoverReproductionState {
    int32_t* offspring_count;
};

struct FlatTurnoverReproductionOutputs {
    int32_t* status;
};

struct TurnoverReproductionConfig {
    int64_t plan_capacity;
    int64_t slot_count;
};

// Exact allocation-free composition of the immutable Stage2 newborn reset
// (native/cuda_architecture_b_generation.cpp:475-559), its action-history
// defaults (native/cuda_architecture_b_birth_transaction.cpp:395-408), and
// child reproduction-stat initialization (native/cuda_architecture_b_lifecycle.cu:
// 825-829). It is applied after the turnover identity transaction has committed
// each child slot. It preserves the generation-boundary defaults:
// early_life_weight and chemistry fidelity start at one, first_repro_step and
// action history start at their sentinels, and all accumulated per-slot
// counters/totals start at zero.
//
// Turnover identity, genome, policy, drives/chemistry, and lifecycle-event
// publication remain separate owners. All pointers, strides, slot validity,
// and widths are launch-boundary obligations.
__device__ __forceinline__ void reset_turnover_newborn_state_serial(
    const FlatTurnoverNewbornState& state,
    const int64_t child_slot,
    const TurnoverNewbornStateConfig config
) {
    state.energy_collected.at(child_slot) = 0.0f;
    state.energy_collected_exact.at(child_slot) = 0.0;
    state.attacks.at(child_slot) = 0;
    state.early_survival_steps.at(child_slot) = 0;
    state.early_energy_collected.at(child_slot) = 0.0f;
    state.early_energy_collected_exact.at(child_slot) = 0.0;
    state.early_injury_gained.at(child_slot) = 0.0f;
    state.early_life_weight.at(child_slot) = 1.0f;
    state.pred_err_sum.at(child_slot) = 0.0f;
    state.pred_err_count.at(child_slot) = 0;
    state.first_repro_step.at(child_slot) = -1;
    state.dev_stage_initialized.at(child_slot) = false;
    state.episode0_pred_err_sum.at(child_slot) = 0.0f;
    state.episode0_pred_err_count.at(child_slot) = 0;
    state.late_pred_err_sum.at(child_slot) = 0.0f;
    state.late_pred_err_count.at(child_slot) = 0;
    state.chemistry_observation_fidelity.at(child_slot) = 1.0f;
    state.chemistry_observation_bias.at(child_slot) = 0.0f;

    state.last_action.at(child_slot) = -1;
    state.last_last_action.at(child_slot) = -1;
    state.action_switches.at(child_slot) = 0;
    state.action_oscillations.at(child_slot) = 0;
    state.episode0_switches.at(child_slot) = 0;
    state.episode0_oscillations.at(child_slot) = 0;
    for (int64_t action = 0; action < config.action_cache_width; ++action) {
        state.action_cache_actions.at(child_slot, action) = -1;
    }
    for (int64_t drive = 0; drive < config.drive_width; ++drive) {
        state.prev_drive_delta.at(child_slot, drive) = 0.0f;
    }
    state.drive_volatility.at(child_slot) = 0.0f;
    state.drive_jitter.at(child_slot) = 0.0f;
    state.episode0_drive_jitter_sum.at(child_slot) = 0.0f;
    state.last_energy_delta.at(child_slot) = 0.0f;
    state.last_health_delta.at(child_slot) = 0.0f;
    state.last_injury_delta.at(child_slot) = 0.0f;
    state.selfharm_events.at(child_slot) = 0;
    state.episode0_selfharm_events.at(child_slot) = 0;
    state.offspring_count.at(child_slot) = 0;
}

// Exact generation-turnover branch from immutable Stage2
// native/cuda_architecture_b_lifecycle.cu:799-802. A turnover refill credits
// both selected parents and does not debit energy or set cooldown. This is
// deliberately separate from the newborn reset above: at a generation
// boundary, Stage2 does not set parents' first_repro_step (that bookkeeping is
// owned by the ordinary mid-generation birth path).
//
// Validation is allocation-free and completes before the first counter write,
// preserving the identity transaction's fail-before-commit behavior.
__device__ __forceinline__ void update_turnover_parent_reproduction_stats_serial(
    const FlatTurnoverReproductionState& state,
    const FlatTurnoverReproductionRows& rows,
    const FlatTurnoverReproductionOutputs& outputs,
    const TurnoverReproductionConfig config
) {
    outputs.status[0] = kStatusOk;

    if (config.plan_capacity < 0 || config.slot_count <= 0) {
        outputs.status[0] = kStatusInvalidConfig;
        return;
    }
    const int64_t birth_count = rows.committed_count[0];
    if (birth_count < 0 || birth_count > config.plan_capacity) {
        outputs.status[0] = kStatusInvalidCommittedCount;
        return;
    }
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = rows.parent_slots[birth];
        const int64_t mate_slot = rows.mate_slots[birth];
        const int64_t child_slot = rows.child_slots[birth];
        if (parent_slot < 0 || parent_slot >= config.slot_count) {
            outputs.status[0] = kStatusInvalidParentSlot;
            return;
        }
        if (mate_slot < 0 || mate_slot >= config.slot_count) {
            outputs.status[0] = kStatusInvalidMateSlot;
            return;
        }
        if (parent_slot == mate_slot) {
            outputs.status[0] = kStatusParentMateAlias;
            return;
        }
        if (child_slot < 0 || child_slot >= config.slot_count) {
            outputs.status[0] = kStatusInvalidChildSlot;
            return;
        }
        for (int64_t previous = 0; previous < birth; ++previous) {
            if (rows.child_slots[previous] == child_slot) {
                outputs.status[0] = kStatusDuplicateChildSlot;
                return;
            }
        }
    }

    for (int64_t birth = 0; birth < birth_count; ++birth) {
        state.offspring_count[rows.parent_slots[birth]] += 1;
        state.offspring_count[rows.mate_slots[birth]] += 1;
    }
}

}  // namespace eiel::cuda_architecture_b::persistent_turnover_newborn_state

#endif  // EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_TURNOVER_NEWBORN_STATE_CUH_
