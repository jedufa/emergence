#ifndef EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_ACTIVE_MECHANICS_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_ACTIVE_MECHANICS_CUH_

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace eiel::cuda_architecture_b::persistent_active_mechanics {

constexpr int32_t kActionGather = 4;
constexpr int32_t kActionRest = 5;
constexpr int32_t kActionAttack = 6;
constexpr int32_t kInt32Max = 2147483647;

enum FlatActiveMechanicsStatus : int32_t {
    kStatusOk = 0,
    kStatusInvalidSlotPartition = 1,
    kStatusNonfiniteLiveState = 2,
    kStatusInvalidDriveControl = 3,
    kStatusMissingDriveNoise = 4,
    kStatusCounterOverflow = 5,
};

enum FlatActiveMechanicsCount : int64_t {
    kCountProcessedRows = 0,
    kCountSurvivalAdvances = 1,
    kCountRestCharges = 2,
    kCountRestDeficitDamage = 3,
    kCountDriveNoiseRows = 4,
    kCountFields = 5,
};

template <typename T>
struct FlatStridedSlotField {
    T* data;
    int64_t element_stride;

    __device__ __forceinline__ T& at(const int64_t slot) const {
        return data[slot * element_stride];
    }
};

template <typename T>
struct FlatStridedDriveField {
    T* data;
    int64_t slot_stride;
    int64_t drive_stride;

    __device__ __forceinline__ T& at(
        const int64_t slot,
        const int64_t drive
    ) const {
        return data[slot * slot_stride + drive * drive_stride];
    }
};

struct FlatActiveMechanicsCellState {
    FlatStridedSlotField<const int64_t> slot_to_agent_id;
    FlatStridedSlotField<const bool> alive_mask;
    FlatStridedSlotField<float> energy;
    FlatStridedSlotField<float> health;
    FlatStridedSlotField<const float> injury;
    // Age is a resident lifecycle counter.  C3 advances it at the same
    // active-row boundary as survival_steps; keeping it mutable here avoids
    // silently dropping the source lifecycle progression at the native ABI.
    FlatStridedSlotField<int32_t> age;
    FlatStridedSlotField<double> policy_energy_observation;
    FlatStridedSlotField<int32_t> rest_streak;
    FlatStridedSlotField<int32_t> survival_steps;
    FlatStridedSlotField<const float> gru_upkeep_cost;
    FlatStridedSlotField<const float> cost_charge_scalar;
    FlatStridedSlotField<const float> cost_charge_maturation_scalar;
    FlatStridedDriveField<float> drives;
    FlatStridedSlotField<const float> drive_inertia;
    FlatStridedSlotField<const float> drive_noise;
    FlatStridedDriveField<float> prev_drive_delta;
    FlatStridedSlotField<float> drive_volatility;
    FlatStridedSlotField<float> drive_jitter;
    FlatStridedSlotField<float> episode0_drive_jitter_sum;
    FlatStridedSlotField<float> last_energy_delta;
    FlatStridedSlotField<float> last_health_delta;
    FlatStridedSlotField<float> last_injury_delta;
    FlatStridedSlotField<int32_t> selfharm_events;
    FlatStridedSlotField<int32_t> episode0_selfharm_events;
};

struct FlatActiveMechanicsPreAction {
    FlatStridedSlotField<const int32_t> actions_by_slot;
    FlatStridedSlotField<const float> energy;
    FlatStridedSlotField<const float> health;
    FlatStridedSlotField<const float> injury;
    FlatStridedDriveField<const float> drive_noise_samples;
    FlatStridedSlotField<const bool> drive_noise_sample_valid;
};

__device__ __forceinline__ float f32_add(
    const float left,
    const float right
) {
    return __fadd_rn(left, right);
}

__device__ __forceinline__ float f32_sub(
    const float left,
    const float right
) {
    return __fsub_rn(left, right);
}

__device__ __forceinline__ float f32_mul(
    const float left,
    const float right
) {
    return __fmul_rn(left, right);
}

__device__ __forceinline__ float f32_div(
    const float left,
    const float right
) {
    return __fdiv_rn(left, right);
}

__device__ __forceinline__ float clamp01(const float value) {
    return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
}

__device__ __forceinline__ float max_zero(const float value) {
    return value < 0.0f ? 0.0f : value;
}

__device__ __forceinline__ int32_t normalized_action(const int32_t action) {
    return action < 0 || action > kActionAttack ? kActionRest : action;
}

__device__ __forceinline__ float bounded_life_history_ramp_v1(
    const int32_t age_ticks,
    const int32_t start_ticks,
    const int32_t ramp_ticks
) {
    if (age_ticks < start_ticks) {
        return 0.0f;
    }
    if (ramp_ticks <= 0) {
        return 1.0f;
    }
    const int32_t elapsed = age_ticks - start_ticks + 1;
    const float scaled = static_cast<float>(elapsed)
        / static_cast<float>(ramp_ticks);
    return clamp01(scaled);
}

// Exact raw-device transcription of the accepted Stage2 active-mechanics kernel;
// invoke once per cell from thread 0 with pre-action snapshots and noise samples.
__device__ __forceinline__ void run_flat_active_mechanics(
    const FlatActiveMechanicsCellState& state,
    const FlatActiveMechanicsPreAction& pre_action,
    int32_t* status,
    int64_t* counts,
    const int64_t count_stride,
    const int64_t capacity,
    const bool enable_rest_charge,
    const int32_t rest_streak_free,
    const float rest_charge_energy,
    const float rest_charge_growth,
    const float rest_deficit_to_health,
    const int32_t developmental_maturity_age,
    const int32_t life_history_mode,
    const int32_t life_history_maturity_start_ticks,
    const int32_t life_history_maturity_rise_ticks,
    const int32_t life_history_fertility_decline_start_ticks,
    const int32_t life_history_fertility_decline_ticks,
    const float life_history_elder_fertility_floor,
    const int32_t life_history_senescence_start_ticks,
    const int32_t life_history_senescence_ramp_ticks,
    const float life_history_senescence_energy_cost,
    const float life_history_senescence_health_cost,
    const int32_t life_history_max_age_ticks,
    const int32_t episode0_k,
    const float max_energy,
    const float start_health,
    const float drive_hunger_gain,
    const float drive_hunger_decay,
    const float drive_pain_gain,
    const float drive_pain_decay,
    const float drive_fatigue_gain,
    const float drive_fatigue_rest_relief,
    const float drive_fatigue_decay,
    const float drive_fear_gain,
    const float drive_fear_decay
) {
    status[0] = kStatusOk;
    for (int64_t index = 0; index < kCountFields; ++index) {
        counts[index * count_stride] = 0;
    }

    if (
        life_history_mode < 0 || life_history_mode > 1
        || life_history_maturity_start_ticks < 0
        || life_history_maturity_rise_ticks < 0
        || life_history_fertility_decline_start_ticks < 0
        || life_history_fertility_decline_ticks < 0
        || !isfinite(life_history_elder_fertility_floor)
        || life_history_elder_fertility_floor < 0.0f
        || life_history_elder_fertility_floor > 1.0f
        || life_history_senescence_start_ticks < 0
        || life_history_senescence_ramp_ticks < 0
        || !isfinite(life_history_senescence_energy_cost)
        || life_history_senescence_energy_cost < 0.0f
        || !isfinite(life_history_senescence_health_cost)
        || life_history_senescence_health_cost < 0.0f
        || life_history_max_age_ticks < 0
    ) {
        status[0] = kStatusInvalidDriveControl;
        return;
    }

    // Validate every assigned row before the first authoritative write. Dead
    // rows may retain NaN diagnostic shadows and are deliberately ignored.
    for (int64_t slot = 0; slot < capacity; ++slot) {
        const bool assigned = state.slot_to_agent_id.at(slot) >= 0;
        if (state.alive_mask.at(slot) != assigned) {
            status[0] = kStatusInvalidSlotPartition;
            return;
        }
        if (!assigned) {
            continue;
        }
        if (
            !isfinite(state.energy.at(slot))
            || !isfinite(state.health.at(slot))
            || !isfinite(state.injury.at(slot))
            || !isfinite(pre_action.energy.at(slot))
            || !isfinite(pre_action.health.at(slot))
            || !isfinite(pre_action.injury.at(slot))
            || !isfinite(state.gru_upkeep_cost.at(slot))
            || !isfinite(state.cost_charge_scalar.at(slot))
            || !isfinite(state.cost_charge_maturation_scalar.at(slot))
            || !isfinite(state.drive_volatility.at(slot))
            || !isfinite(state.drive_jitter.at(slot))
            || !isfinite(state.episode0_drive_jitter_sum.at(slot))
            || !isfinite(state.last_energy_delta.at(slot))
            || !isfinite(state.last_health_delta.at(slot))
            || !isfinite(state.last_injury_delta.at(slot))
        ) {
            status[0] = kStatusNonfiniteLiveState;
            return;
        }
        for (int64_t drive = 0; drive < 4; ++drive) {
            if (
                !isfinite(state.drives.at(slot, drive))
                || !isfinite(state.prev_drive_delta.at(slot, drive))
            ) {
                status[0] = kStatusNonfiniteLiveState;
                return;
            }
        }
        if (
            !isfinite(state.drive_inertia.at(slot))
            || state.drive_inertia.at(slot) < 0.0f
            || !isfinite(state.drive_noise.at(slot))
            || state.drive_noise.at(slot) < 0.0f
        ) {
            status[0] = kStatusInvalidDriveControl;
            return;
        }
        if (state.drive_noise.at(slot) > 0.0f) {
            if (!pre_action.drive_noise_sample_valid.at(slot)) {
                status[0] = kStatusMissingDriveNoise;
                return;
            }
            for (int64_t drive = 0; drive < 4; ++drive) {
                if (!isfinite(pre_action.drive_noise_samples.at(slot, drive))) {
                    status[0] = kStatusMissingDriveNoise;
                    return;
                }
            }
        }
        if (
            state.rest_streak.at(slot) == kInt32Max
            || state.survival_steps.at(slot) == kInt32Max
            || state.selfharm_events.at(slot) == kInt32Max
            || state.episode0_selfharm_events.at(slot) == kInt32Max
        ) {
            status[0] = kStatusCounterOverflow;
            return;
        }
    }

    for (int64_t slot = 0; slot < capacity; ++slot) {
        if (
            !state.alive_mask.at(slot)
            || state.slot_to_agent_id.at(slot) < 0
        ) {
            continue;
        }
        const int32_t action = normalized_action(
            pre_action.actions_by_slot.at(slot)
        );

        // The accepted CPU owner performs this rest gate after base action
        // mutation and before its action-time death decision.
        if (enable_rest_charge) {
            if (action == kActionRest) {
                state.rest_streak.at(slot) += 1;
            } else {
                state.rest_streak.at(slot) = 0;
            }
            if (state.rest_streak.at(slot) > rest_streak_free) {
                const int32_t over = state.rest_streak.at(slot) - rest_streak_free;
                const float growth = f32_mul(
                    rest_charge_growth,
                    static_cast<float>(over - 1)
                );
                const float charge = f32_add(rest_charge_energy, growth);
                state.energy.at(slot) = f32_sub(state.energy.at(slot), charge);
                counts[kCountRestCharges * count_stride] += 1;
                if (state.energy.at(slot) < 0.0f) {
                    const float damage = f32_mul(
                        -state.energy.at(slot),
                        rest_deficit_to_health
                    );
                    state.health.at(slot) = f32_sub(
                        state.health.at(slot),
                        damage
                    );
                    state.energy.at(slot) = 0.0f;
                    counts[kCountRestDeficitDamage * count_stride] += 1;
                }
                state.policy_energy_observation.at(slot) = static_cast<double>(
                    state.energy.at(slot)
                );
            }
        }

        // Rows terminal at the CPU action boundary do not receive upkeep or a
        // drive update. The following lifecycle slice owns their cull.
        if (state.energy.at(slot) <= 0.0f || state.health.at(slot) <= 0.0f) {
            continue;
        }

        const float gru_charge = max_zero(state.gru_upkeep_cost.at(slot));
        if (gru_charge > 0.0f) {
            state.energy.at(slot) = max_zero(
                f32_sub(state.energy.at(slot), gru_charge)
            );
        }
        float charge = max_zero(state.cost_charge_scalar.at(slot));
        if (
            developmental_maturity_age > 0
            && state.survival_steps.at(slot) < developmental_maturity_age
        ) {
            charge = f32_add(
                charge,
                max_zero(state.cost_charge_maturation_scalar.at(slot))
            );
        }
        if (charge > 0.0f) {
            state.energy.at(slot) = max_zero(
                f32_sub(state.energy.at(slot), charge)
            );
        }
        if (life_history_mode == 1) {
            const int32_t age_ticks = state.survival_steps.at(slot);
            const float senescence = bounded_life_history_ramp_v1(
                age_ticks,
                life_history_senescence_start_ticks,
                life_history_senescence_ramp_ticks
            );
            if (senescence > 0.0f) {
                const float energy_cost = f32_mul(
                    max_zero(life_history_senescence_energy_cost),
                    senescence
                );
                const float health_cost = f32_mul(
                    max_zero(life_history_senescence_health_cost),
                    senescence
                );
                if (energy_cost > 0.0f) {
                    state.energy.at(slot) = max_zero(
                        f32_sub(state.energy.at(slot), energy_cost)
                    );
                }
                if (health_cost > 0.0f) {
                    state.health.at(slot) = f32_sub(
                        state.health.at(slot),
                        health_cost
                    );
                }
            }
            if (
                life_history_max_age_ticks > 0
                && age_ticks >= life_history_max_age_ticks
            ) {
                state.health.at(slot) = 0.0f;
            }
        }
        state.survival_steps.at(slot) += 1;
        state.age.at(slot) += 1;
        state.policy_energy_observation.at(slot) = static_cast<double>(
            state.energy.at(slot)
        );

        const float prev0 = state.drives.at(slot, 0);
        const float prev1 = state.drives.at(slot, 1);
        const float prev2 = state.drives.at(slot, 2);
        const float prev3 = state.drives.at(slot, 3);
        const float e_normed = clamp01(
            f32_div(state.energy.at(slot), max_energy)
        );
        const float h_normed = clamp01(
            f32_div(state.health.at(slot), start_health)
        );
        const float de = f32_sub(
            state.energy.at(slot),
            pre_action.energy.at(slot)
        );
        const float dh = f32_sub(
            state.health.at(slot),
            pre_action.health.at(slot)
        );
        const float dinj = f32_sub(
            state.injury.at(slot),
            pre_action.injury.at(slot)
        );
        state.last_energy_delta.at(slot) = de;
        state.last_health_delta.at(slot) = dh;
        state.last_injury_delta.at(slot) = dinj;

        const float hunger_gap = f32_sub(1.0f, e_normed);
        const float hunger_gain_term = f32_mul(drive_hunger_gain, hunger_gap);
        const float target_hunger_start = f32_add(prev0, hunger_gain_term);
        const float hunger_delta_norm = f32_div(-de, max_energy);
        const float hunger_delta_term = f32_mul(0.04f, hunger_delta_norm);
        const float target_hunger_with_delta = f32_add(
            target_hunger_start,
            hunger_delta_term
        );
        float target_hunger = f32_sub(
            target_hunger_with_delta,
            drive_hunger_decay
        );
        if (action == kActionGather && de > 0.0f) {
            const float gather_norm = f32_div(de, max_energy);
            const float gather_term = f32_mul(0.20f, gather_norm);
            target_hunger = f32_sub(target_hunger, gather_term);
        }
        target_hunger = clamp01(target_hunger);

        float target_pain = prev1;
        if (dinj > 0.0f) {
            target_pain = f32_add(
                target_pain,
                f32_mul(drive_pain_gain, dinj < 1.0f ? dinj : 1.0f)
            );
        }
        if (dh < 0.0f) {
            const float damage_norm = f32_div(-dh, start_health);
            target_pain = f32_add(
                target_pain,
                f32_mul(
                    0.10f,
                    damage_norm < 1.0f ? damage_norm : 1.0f
                )
            );
        }
        target_pain = clamp01(f32_sub(target_pain, drive_pain_decay));

        float target_fatigue = prev2;
        if (action == kActionRest) {
            target_fatigue = f32_sub(
                target_fatigue,
                drive_fatigue_rest_relief
            );
        } else {
            target_fatigue = f32_add(target_fatigue, drive_fatigue_gain);
        }
        target_fatigue = clamp01(
            f32_sub(target_fatigue, drive_fatigue_decay)
        );

        float target_fear = prev3;
        if (dinj > 0.0f) {
            target_fear = f32_add(
                target_fear,
                f32_mul(drive_fear_gain, dinj < 1.0f ? dinj : 1.0f)
            );
        }
        target_fear = clamp01(
            f32_sub(
                f32_add(
                    target_fear,
                    f32_mul(0.05f, f32_sub(1.0f, h_normed))
                ),
                drive_fear_decay
            )
        );

        float inertia = state.drive_inertia.at(slot);
        inertia = inertia < 0.0f
            ? 0.0f
            : (inertia > 0.999f ? 0.999f : inertia);
        const float inverse = f32_sub(1.0f, inertia);
        float next0 = f32_add(
            f32_mul(inertia, prev0),
            f32_mul(inverse, target_hunger)
        );
        float next1 = f32_add(
            f32_mul(inertia, prev1),
            f32_mul(inverse, target_pain)
        );
        float next2 = f32_add(
            f32_mul(inertia, prev2),
            f32_mul(inverse, target_fatigue)
        );
        float next3 = f32_add(
            f32_mul(inertia, prev3),
            f32_mul(inverse, target_fear)
        );
        const float noise_scale = state.drive_noise.at(slot);
        if (noise_scale > 0.0f) {
            next0 = f32_add(
                next0,
                f32_mul(
                    noise_scale,
                    pre_action.drive_noise_samples.at(slot, 0)
                )
            );
            next1 = f32_add(
                next1,
                f32_mul(
                    noise_scale,
                    pre_action.drive_noise_samples.at(slot, 1)
                )
            );
            next2 = f32_add(
                next2,
                f32_mul(
                    noise_scale,
                    pre_action.drive_noise_samples.at(slot, 2)
                )
            );
            next3 = f32_add(
                next3,
                f32_mul(
                    noise_scale,
                    pre_action.drive_noise_samples.at(slot, 3)
                )
            );
            counts[kCountDriveNoiseRows * count_stride] += 1;
        }
        next0 = clamp01(next0);
        next1 = clamp01(next1);
        next2 = clamp01(next2);
        next3 = clamp01(next3);

        const float delta0 = f32_sub(next0, prev0);
        const float delta1 = f32_sub(next1, prev1);
        const float delta2 = f32_sub(next2, prev2);
        const float delta3 = f32_sub(next3, prev3);
        const float volatility_instant = static_cast<float>(
            fabs(static_cast<double>(delta0))
            + fabs(static_cast<double>(delta1))
            + fabs(static_cast<double>(delta2))
            + fabs(static_cast<double>(delta3))
        );
        state.drive_volatility.at(slot) = f32_add(
            state.drive_volatility.at(slot),
            volatility_instant
        );
        const float jitter_instant = static_cast<float>(
            fabs(
                static_cast<double>(delta0)
                - static_cast<double>(state.prev_drive_delta.at(slot, 0))
            )
            + fabs(
                static_cast<double>(delta1)
                - static_cast<double>(state.prev_drive_delta.at(slot, 1))
            )
            + fabs(
                static_cast<double>(delta2)
                - static_cast<double>(state.prev_drive_delta.at(slot, 2))
            )
            + fabs(
                static_cast<double>(delta3)
                - static_cast<double>(state.prev_drive_delta.at(slot, 3))
            )
        );
        state.drive_jitter.at(slot) = f32_add(
            state.drive_jitter.at(slot),
            jitter_instant
        );
        if (state.survival_steps.at(slot) <= episode0_k) {
            state.episode0_drive_jitter_sum.at(slot) = f32_add(
                state.episode0_drive_jitter_sum.at(slot),
                jitter_instant
            );
        }
        state.prev_drive_delta.at(slot, 0) = delta0;
        state.prev_drive_delta.at(slot, 1) = delta1;
        state.prev_drive_delta.at(slot, 2) = delta2;
        state.prev_drive_delta.at(slot, 3) = delta3;
        state.drives.at(slot, 0) = next0;
        state.drives.at(slot, 1) = next1;
        state.drives.at(slot, 2) = next2;
        state.drives.at(slot, 3) = next3;
        if (action == kActionAttack && dinj > 0.0f) {
            state.selfharm_events.at(slot) += 1;
            if (state.survival_steps.at(slot) <= episode0_k) {
                state.episode0_selfharm_events.at(slot) += 1;
            }
        }
        counts[kCountProcessedRows * count_stride] += 1;
        counts[kCountSurvivalAdvances * count_stride] += 1;
    }
}

__device__ __forceinline__ void run_flat_active_mechanics(
    const FlatActiveMechanicsCellState& state,
    const FlatActiveMechanicsPreAction& pre_action,
    int32_t* status,
    int64_t* counts,
    const int64_t count_stride,
    const int64_t capacity,
    const bool enable_rest_charge,
    const int32_t rest_streak_free,
    const float rest_charge_energy,
    const float rest_charge_growth,
    const float rest_deficit_to_health,
    const int32_t developmental_maturity_age,
    const int32_t episode0_k,
    const float max_energy,
    const float start_health,
    const float drive_hunger_gain,
    const float drive_hunger_decay,
    const float drive_pain_gain,
    const float drive_pain_decay,
    const float drive_fatigue_gain,
    const float drive_fatigue_rest_relief,
    const float drive_fatigue_decay,
    const float drive_fear_gain,
    const float drive_fear_decay
) {
    run_flat_active_mechanics(
        state,
        pre_action,
        status,
        counts,
        count_stride,
        capacity,
        enable_rest_charge,
        rest_streak_free,
        rest_charge_energy,
        rest_charge_growth,
        rest_deficit_to_health,
        developmental_maturity_age,
        0,
        0,
        0,
        0,
        0,
        0.0f,
        0,
        0,
        0.0f,
        0.0f,
        0,
        episode0_k,
        max_energy,
        start_health,
        drive_hunger_gain,
        drive_hunger_decay,
        drive_pain_gain,
        drive_pain_decay,
        drive_fatigue_gain,
        drive_fatigue_rest_relief,
        drive_fatigue_decay,
        drive_fear_gain,
        drive_fear_decay
    );
}

}  // namespace eiel::cuda_architecture_b::persistent_active_mechanics

#endif  // EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_ACTIVE_MECHANICS_CUH_
