#ifndef EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_BIRTH_DRIVES_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_BIRTH_DRIVES_CUH_

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "cuda_architecture_b_persistent_chemistry.cuh"

namespace eiel::cuda_architecture_b::persistent_birth_drives {

// Immutable Stage2 source: commit 11f58fa7,
// native/cuda_architecture_b_child_drives.cu:372-379.
enum OrdinaryBirthDriveStatus : int32_t {
    kStatusOk = 0,
    kStatusInvalidBirthPlan = 1,
    kStatusInvalidParentOrder = 2,
    kStatusChildIdentityNotCommitted = 3,
    kStatusUnsupportedGru = 4,
    kStatusInvalidChildScalarState = 5,
    kStatusInvalidPcg64State = 6,
};

constexpr int64_t kActionCount = 7;
constexpr int64_t kInt64Max = 9223372036854775807LL;

// Every pointer is already offset to the simulation cell owned by the caller.
struct FlatOrdinaryBirthDriveState {
    const int64_t* slot_to_agent_id;
    const bool* alive_mask;
    const bool* policy_use_gru;
    const double* policy_energy_observation;
    const float* genome_drive_sp_hunger;
    const float* genome_drive_sp_pain;
    const float* genome_drive_sp_fatigue;
    const float* genome_drive_sp_fear;
    const float* genome_exp_prior_noise;
    const float* genome_drive_birth_noise;
};

struct FlatOrdinaryBirthPlan {
    const int64_t* parent_slots;
    const int64_t* child_slots;
    const int64_t* planned_count;
};

// state_words, buffer_words, and mutation_np_semantic_draws have extents 4,
// 2, and 1 respectively. This is the serialized Population.np_rng cursor.
struct FlatPopulationPcg64Cursor {
    uint64_t* state_words;
    int64_t* buffer_words;
    int64_t* mutation_np_semantic_draws;
};

// Scalar counter/status pointers each have one element. The Stage2 launch
// owner zero-initializes these outputs. On a rejected transaction only status
// is changed, preserving the source kernel's fail-before-commit behavior.
struct FlatOrdinaryBirthDriveOutputs {
    float* drives;
    float* birth_reset_shock;
    int64_t* committed_count;
    int64_t* birth_noise_rows_consumed;
    int64_t* exp_prior_noise_rows_consumed;
    int64_t* standard_normal_rows_generated;
    int64_t* semantic_draws_added;
    int32_t* status;
};

// Pointer validity, physical extents, widths, and positive finite max_energy
// are launch-boundary obligations, matching the immutable Stage2 host wrapper.
struct OrdinaryBirthDriveConfig {
    int64_t slot_count;
    int64_t plan_capacity;
    int64_t drive_width;
    float max_energy;
};

struct FlatNewbornChemistryState {
    const int64_t* slot_to_agent_id;
    const bool* alive_mask;
    const double* policy_energy_observation;
    float* chemistry;
};

struct FlatNewbornChemistryOutputs {
    int32_t* status;
};

struct NewbornChemistryConfig {
    int64_t slot_count;
    int64_t plan_capacity;
    int64_t chemistry_width;
    float max_energy;
};

namespace detail {

using Pcg64State = eiel::cuda_architecture_b::Pcg64State;
namespace chemistry_detail =
    eiel::cuda_architecture_b::persistent_chemistry::detail;

// Immutable Stage2 source: child_drives.cu:423-454. The standard-normal
// primitive and tables are the exact extraction already shared by the
// persistent chemistry path from child_drives.cu:21-421.
__device__ __forceinline__ float clamp01(const float value) {
    if (value < 0.0f) {
        return 0.0f;
    }
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

__device__ __forceinline__ double canon_genome_double(const float value) {
    const double scaled = static_cast<double>(value) * 100000000.0;
    return ::nearbyint(scaled) / 100000000.0;
}

__device__ __forceinline__ bool finite_unit_scalar(const float value) {
    return ::isfinite(value) && value >= 0.0f && value <= 1.0f;
}

}  // namespace detail

// Exact allocation-free serial transcription of immutable Stage2
// fixed_mlp_child_drives_kernel (child_drives.cu:770-1012). Exactly one device
// thread invokes this helper for one simulation cell after identity commit.
// The helper deliberately executes both the preflight and commit RNG walks.
__device__ __forceinline__ void initialize_ordinary_newborn_drives_serial(
    const FlatOrdinaryBirthDriveState state,
    const FlatOrdinaryBirthPlan plan,
    const FlatPopulationPcg64Cursor cursor,
    const FlatOrdinaryBirthDriveOutputs outputs,
    const OrdinaryBirthDriveConfig config
) {
    const int64_t birth_count = plan.planned_count[0];
    if (
        birth_count < 0 || birth_count > config.plan_capacity
        || birth_count > config.slot_count
    ) {
        outputs.status[0] = kStatusInvalidBirthPlan;
        return;
    }

    if (
        cursor.buffer_words[0] < 0 || cursor.buffer_words[0] > 1
        || cursor.buffer_words[1] < 0
        || static_cast<uint64_t>(cursor.buffer_words[1]) > 0xffffffffULL
        || (cursor.state_words[2] & 1ULL) == 0ULL
        || cursor.mutation_np_semantic_draws[0] < 0
    ) {
        outputs.status[0] = kStatusInvalidPcg64State;
        return;
    }

    int64_t previous_parent_id = -1;
    int64_t previous_child_id = -1;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = plan.parent_slots[birth];
        const int64_t child_slot = plan.child_slots[birth];
        if (
            parent_slot < 0 || parent_slot >= config.slot_count
            || child_slot < 0 || child_slot >= config.slot_count
            || parent_slot == child_slot
            || !state.alive_mask[parent_slot]
            || state.slot_to_agent_id[parent_slot] < 0
        ) {
            outputs.status[0] = kStatusInvalidBirthPlan;
            return;
        }
        if (
            !state.alive_mask[child_slot]
            || state.slot_to_agent_id[child_slot] < 0
        ) {
            outputs.status[0] = kStatusChildIdentityNotCommitted;
            return;
        }
        const int64_t parent_id = state.slot_to_agent_id[parent_slot];
        const int64_t child_id = state.slot_to_agent_id[child_slot];
        if (
            (birth > 0 && parent_id <= previous_parent_id)
            || (birth > 0 && child_id <= previous_child_id)
        ) {
            outputs.status[0] = kStatusInvalidParentOrder;
            return;
        }
        previous_parent_id = parent_id;
        previous_child_id = child_id;
        for (int64_t other = 0; other < birth_count; ++other) {
            if (other != birth && plan.child_slots[other] == child_slot) {
                outputs.status[0] = kStatusInvalidBirthPlan;
                return;
            }
            if (plan.parent_slots[other] == child_slot) {
                outputs.status[0] = kStatusInvalidBirthPlan;
                return;
            }
        }
        const double child_energy = state.policy_energy_observation[child_slot];
        const float sp_hunger = state.genome_drive_sp_hunger[child_slot];
        const float sp_pain = state.genome_drive_sp_pain[child_slot];
        const float sp_fatigue = state.genome_drive_sp_fatigue[child_slot];
        const float sp_fear = state.genome_drive_sp_fear[child_slot];
        const float exp_prior_noise = state.genome_exp_prior_noise[child_slot];
        const float birth_noise = state.genome_drive_birth_noise[child_slot];
        if (
            !::isfinite(child_energy)
            || !detail::finite_unit_scalar(sp_hunger)
            || !detail::finite_unit_scalar(sp_pain)
            || !detail::finite_unit_scalar(sp_fatigue)
            || !detail::finite_unit_scalar(sp_fear)
            || !::isfinite(exp_prior_noise)
            || exp_prior_noise < 0.0f || exp_prior_noise > 0.5f
            || !::isfinite(birth_noise)
            || birth_noise < 0.0f || birth_noise > 0.2f
        ) {
            outputs.status[0] = kStatusInvalidChildScalarState;
            return;
        }
    }

    detail::Pcg64State preflight_rng = detail::chemistry_detail::load_rng(
        cursor.state_words,
        cursor.buffer_words
    );
    int64_t planned_birth_noise_rows = 0;
    int64_t planned_exp_prior_rows = 0;
    int64_t planned_semantic_draws = 0;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t child_slot = plan.child_slots[birth];
        const double clipped_birth_noise =
            detail::chemistry_detail::clamp_double(
                detail::canon_genome_double(
                    state.genome_drive_birth_noise[child_slot]
                ),
                0.0,
                0.20
            );
        if (clipped_birth_noise > 0.0) {
            detail::chemistry_detail::consume_standard_normals(
                preflight_rng,
                4
            );
            planned_birth_noise_rows += 1;
            planned_semantic_draws += 4;
        }
        const double clipped_exp_prior_noise =
            detail::chemistry_detail::clamp_double(
                static_cast<double>(
                    state.genome_exp_prior_noise[child_slot]
                ),
                0.0,
                0.5
            );
        if (clipped_exp_prior_noise > 0.0) {
            detail::chemistry_detail::consume_standard_normals(
                preflight_rng,
                2 * kActionCount
            );
            planned_exp_prior_rows += 2;
            planned_semantic_draws += 2 * kActionCount;
        }
    }
    if (
        cursor.mutation_np_semantic_draws[0]
        > kInt64Max - planned_semantic_draws
    ) {
        outputs.status[0] = kStatusInvalidPcg64State;
        return;
    }

    detail::Pcg64State commit_rng = detail::chemistry_detail::load_rng(
        cursor.state_words,
        cursor.buffer_words
    );
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t child_slot = plan.child_slots[birth];
        const float child_sp_hunger =
            state.genome_drive_sp_hunger[child_slot];
        const float child_sp_pain = state.genome_drive_sp_pain[child_slot];
        const float child_sp_fatigue =
            state.genome_drive_sp_fatigue[child_slot];
        const float child_sp_fear = state.genome_drive_sp_fear[child_slot];
        const double child_sp_hunger_scalar =
            detail::canon_genome_double(child_sp_hunger);
        const double hunger_from_energy =
            detail::chemistry_detail::clamp_double(
                1.0
                    - state.policy_energy_observation[child_slot]
                        / static_cast<double>(config.max_energy),
                0.0,
                1.0
            );
        const int64_t drive_offset = child_slot * config.drive_width;
        outputs.drives[drive_offset] = detail::clamp01(static_cast<float>(
            0.60 * child_sp_hunger_scalar + 0.40 * hunger_from_energy
        ));
        outputs.drives[drive_offset + 1] = detail::clamp01(child_sp_pain);
        outputs.drives[drive_offset + 2] = detail::clamp01(child_sp_fatigue);
        outputs.drives[drive_offset + 3] = detail::clamp01(child_sp_fear);

        const double clipped_birth_noise =
            detail::chemistry_detail::clamp_double(
                detail::canon_genome_double(
                    state.genome_drive_birth_noise[child_slot]
                ),
                0.0,
                0.20
            );
        if (clipped_birth_noise > 0.0) {
            for (int64_t drive = 0; drive < 4; ++drive) {
                const float normal = static_cast<float>(
                    detail::chemistry_detail::pcg64_numpy_standard_normal(
                        commit_rng
                    )
                );
                const float noise_delta = static_cast<float>(
                    static_cast<float>(clipped_birth_noise) * normal
                );
                outputs.drives[drive_offset + drive] =
                    detail::clamp01(static_cast<float>(
                        outputs.drives[drive_offset + drive] + noise_delta
                    ));
            }
        }

        const float shock0 = ::fabsf(
            outputs.drives[drive_offset] - child_sp_hunger
        );
        const float shock1 = ::fabsf(
            outputs.drives[drive_offset + 1] - child_sp_pain
        );
        const float shock2 = ::fabsf(
            outputs.drives[drive_offset + 2] - child_sp_fatigue
        );
        const float shock3 = ::fabsf(
            outputs.drives[drive_offset + 3] - child_sp_fear
        );
        const float shock_sum01 = static_cast<float>(shock0 + shock1);
        const float shock_sum012 = static_cast<float>(shock_sum01 + shock2);
        const float shock_sum0123 = static_cast<float>(
            shock_sum012 + shock3
        );
        outputs.birth_reset_shock[child_slot] = static_cast<float>(
            shock_sum0123 / 4.0f
        );

        if (state.genome_exp_prior_noise[child_slot] > 0.0f) {
            detail::chemistry_detail::consume_standard_normals(
                commit_rng,
                2 * kActionCount
            );
        }
    }

    detail::chemistry_detail::store_rng(
        cursor.state_words,
        cursor.buffer_words,
        commit_rng
    );
    cursor.mutation_np_semantic_draws[0] += planned_semantic_draws;
    outputs.committed_count[0] = birth_count;
    outputs.birth_noise_rows_consumed[0] = planned_birth_noise_rows;
    outputs.exp_prior_noise_rows_consumed[0] = planned_exp_prior_rows;
    outputs.standard_normal_rows_generated[0] = planned_birth_noise_rows;
    outputs.semantic_draws_added[0] = planned_semantic_draws;
    outputs.status[0] = kStatusOk;
}

// Exact allocation-free serial transcription of immutable Stage2
// newborn_chemistry_kernel (child_drives.cu:716-768). The energy denominator
// remains the host-normalized float32 max_energy widened back to float64.
__device__ __forceinline__ void initialize_newborn_chemistry_serial(
    const FlatNewbornChemistryState state,
    const FlatOrdinaryBirthPlan plan,
    const FlatNewbornChemistryOutputs outputs,
    const NewbornChemistryConfig config
) {
    const int64_t count = plan.planned_count[0];
    if (
        count < 0 || count > config.plan_capacity || count > config.slot_count
        || !::isfinite(config.max_energy) || config.max_energy <= 0.0f
    ) {
        outputs.status[0] = kStatusInvalidBirthPlan;
        return;
    }
    for (int64_t row = 0; row < count; ++row) {
        const int64_t slot = plan.child_slots[row];
        if (
            slot < 0 || slot >= config.slot_count || !state.alive_mask[slot]
            || state.slot_to_agent_id[slot] < 0
            || !::isfinite(state.policy_energy_observation[slot])
        ) {
            outputs.status[0] = kStatusInvalidChildScalarState;
            return;
        }
    }

    const double max_energy_f64 = static_cast<double>(config.max_energy);
    for (int64_t row = 0; row < count; ++row) {
        const int64_t slot = plan.child_slots[row];
        const double energy_n = detail::chemistry_detail::clamp_double(
            state.policy_energy_observation[slot] / max_energy_f64,
            0.0,
            1.0
        );
        const int64_t offset = slot * config.chemistry_width;
        state.chemistry[offset + 0] = static_cast<float>(
            0.40 + 0.40 * energy_n
        );
        state.chemistry[offset + 1] = 0.55f;
        state.chemistry[offset + 2] = 0.0f;
        state.chemistry[offset + 3] = 0.0f;
        state.chemistry[offset + 4] = 0.50f;
        state.chemistry[offset + 5] = 0.25f;
    }
}

}  // namespace eiel::cuda_architecture_b::persistent_birth_drives

#endif  // EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_BIRTH_DRIVES_CUH_
