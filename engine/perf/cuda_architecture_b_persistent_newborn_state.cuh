#ifndef EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_NEWBORN_STATE_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_NEWBORN_STATE_CUH_

#include <cuda_runtime.h>

#include <cstdint>

namespace eiel::cuda_architecture_b::persistent_newborn_state {

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

// Every field is already offset to the simulation cell owned by the caller.
// The first 18 members preserve TURNOVER_STATE_FIELDS order from the immutable
// Stage2 generation owner.
struct FlatFreshNewbornSelectionState {
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
};

struct FreshNewbornSelectionConfig {
    int64_t action_cache_width;
    int64_t drive_width;
};

// Exact raw-device transcription of Stage2 reset_newborn_slot_state for one
// already-committed ordinary child slot. Identity, planning, scalar genome,
// drives/chemistry, policy inheritance, and lifecycle events remain external.
// The launch boundary must validate the slot, pointers, strides, and widths.
__device__ __forceinline__ void reset_ordinary_newborn_selection_state(
    const FlatFreshNewbornSelectionState& state,
    const int64_t child_slot,
    const FreshNewbornSelectionConfig config
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
}

}  // namespace eiel::cuda_architecture_b::persistent_newborn_state

#endif  // EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_NEWBORN_STATE_CUH_
