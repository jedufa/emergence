#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "cuda_architecture_b_birth_parent_order_v1.cuh"
#include "cuda_architecture_b_flattened_birth_scratch_v1.cuh"
#include "cuda_architecture_b_persistent_chemistry.cuh"

namespace eiel::cuda_architecture_b::flattened_birth_drives_v1 {

namespace parent_order =
    eiel::cuda_architecture_b::birth_parent_order_v1;

// Exact status identity from immutable Stage2 commit 11f58fa7,
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

constexpr int64_t kMiniRowCount = 2;
constexpr int64_t kParentRow = 0;
constexpr int64_t kChildRow = 1;
constexpr int64_t kMaximumBirthCount = 1;
constexpr int64_t kDriveCount = 4;
constexpr int64_t kActionCount = 7;
constexpr int64_t kPopulationPcg64StateWordCount = 4;
constexpr int64_t kPopulationPcg64BufferWordCount = 2;
constexpr int64_t kBirthNoiseStandardNormalCount = kDriveCount;
constexpr int64_t kExpPriorStandardNormalCount = 2 * kActionCount;
constexpr int64_t kMaximumSemanticDrawsPerBirth =
    kBirthNoiseStandardNormalCount + kExpPriorStandardNormalCount;
constexpr int64_t kInt64Max = 9223372036854775807LL;

namespace scratch =
    eiel::cuda_architecture_b::flattened_birth_scratch_v1;

constexpr int64_t kScratchPopulationPcg64StateWordCapacity =
    (scratch::kPopulationPcgBuffer - scratch::kPopulationPcgState)
    / static_cast<int64_t>(sizeof(uint64_t));
constexpr int64_t kScratchPopulationPcg64BufferWordCapacity =
    (scratch::kPopulationSemanticDraws - scratch::kPopulationPcgBuffer)
    / static_cast<int64_t>(sizeof(int64_t));
constexpr int64_t kScratchMiniSlotCapacity =
    (scratch::kMiniAlive - scratch::kMiniSlotIds)
    / static_cast<int64_t>(sizeof(int64_t));
constexpr int64_t kScratchMiniEnergyCapacity =
    (scratch::kMiniDrives - scratch::kMiniEnergy)
    / static_cast<int64_t>(sizeof(double));
constexpr int64_t kScratchMiniDriveFloatCapacity =
    (scratch::kMiniChemistry - scratch::kMiniDrives)
    / static_cast<int64_t>(sizeof(float));
constexpr int64_t kScratchMiniBirthShockCapacity =
    (scratch::kMiniDriveCommitted - scratch::kMiniBirthShock)
    / static_cast<int64_t>(sizeof(float));

static_assert(kMiniRowCount == 2);
static_assert(kParentRow == 0 && kChildRow == 1);
static_assert(kMaximumBirthCount == 1);
static_assert(kDriveCount == 4);
static_assert(kActionCount == 7);
static_assert(kMaximumSemanticDrawsPerBirth == 18);
static_assert(
    kScratchPopulationPcg64StateWordCapacity
    == kPopulationPcg64StateWordCount
);
static_assert(
    kScratchPopulationPcg64BufferWordCapacity
    == kPopulationPcg64BufferWordCount
);
static_assert(kScratchMiniSlotCapacity == kMiniRowCount);
static_assert(kScratchMiniEnergyCapacity == kMiniRowCount);
static_assert(kScratchMiniDriveFloatCapacity >= kMiniRowCount * kDriveCount);
static_assert(kScratchMiniBirthShockCapacity == kMiniRowCount);
static_assert(
    scratch::kPopulationSemanticDraws
        + static_cast<int64_t>(sizeof(int64_t))
    <= scratch::kMiniSlotIds
);
static_assert(
    scratch::kMiniDriveStatus + static_cast<int64_t>(sizeof(int32_t))
    <= scratch::kMiniChemistryStatus
);

namespace detail {

using Pcg64State = eiel::cuda_architecture_b::Pcg64State;
namespace pcg64_detail =
    eiel::cuda_architecture_b::persistent_chemistry::detail;

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

// Exact flattened scratch transcription of
// persistent_birth_drives::initialize_ordinary_newborn_drives_serial, whose
// source is immutable Stage2 fixed_mlp_child_drives_kernel (commit 11f58fa7,
// native/cuda_architecture_b_child_drives.cu:770-1012). This primitive does not
// call the persistent helper and has no authoritative output parameter.
//
// Each mini-state pointer has exactly two provisional rows: parent row 0 and
// child row 1. The Population PCG64 arrays have extents 4 and 2. All mutable
// parameters must point into external transaction scratch. Scratch counters are
// zero-initialized by the caller; rejection changes only scratch_status, exactly
// preserving the source fail-before-commit boundary. Pointer validity, extents,
// and positive finite max_energy remain launch-boundary obligations.
//
// The preflight walk starts from the scratch cursor and consumes birth-noise
// normals before experience-prior normals. Commit reloads that same initial
// cursor and repeats the identical order, storing only the committed scratch
// continuation. The bounded route admits zero or one birth and contains no
// cell, birth, or authoritative-array loop.
__device__ __forceinline__ void prepare_ordinary_newborn_drives_scratch_v1(
    const int64_t* __restrict__ mini_slot_to_agent_id,
    const bool* __restrict__ mini_alive_mask,
    const bool* __restrict__ mini_policy_use_gru,
    const double* __restrict__ mini_policy_energy_observation,
    const float* __restrict__ mini_genome_drive_sp_hunger,
    const float* __restrict__ mini_genome_drive_sp_pain,
    const float* __restrict__ mini_genome_drive_sp_fatigue,
    const float* __restrict__ mini_genome_drive_sp_fear,
    const float* __restrict__ mini_genome_exp_prior_noise,
    const float* __restrict__ mini_genome_drive_birth_noise,
    const int64_t birth_count,
    const float max_energy,
    uint64_t* __restrict__ scratch_population_pcg64_state_words,
    int64_t* __restrict__ scratch_population_pcg64_buffer_words,
    int64_t& scratch_population_np_semantic_draws,
    float* __restrict__ scratch_child_drives,
    float& scratch_child_birth_reset_shock,
    int64_t& scratch_committed_count,
    int64_t& scratch_birth_noise_rows_consumed,
    int64_t& scratch_exp_prior_noise_rows_consumed,
    int64_t& scratch_standard_normal_rows_generated,
    int64_t& scratch_semantic_draws_added,
    int32_t& scratch_status
) {
    if (birth_count < 0 || birth_count > kMaximumBirthCount) {
        scratch_status = kStatusInvalidBirthPlan;
        return;
    }

    if (
        scratch_population_pcg64_buffer_words[0] < 0
        || scratch_population_pcg64_buffer_words[0] > 1
        || scratch_population_pcg64_buffer_words[1] < 0
        || static_cast<uint64_t>(
               scratch_population_pcg64_buffer_words[1]
           ) > 0xffffffffULL
        || (scratch_population_pcg64_state_words[2] & 1ULL) == 0ULL
        || scratch_population_np_semantic_draws < 0
    ) {
        scratch_status = kStatusInvalidPcg64State;
        return;
    }

    if (birth_count == 1) {
        if (
            !mini_alive_mask[kParentRow]
            || mini_slot_to_agent_id[kParentRow] < 0
        ) {
            scratch_status = kStatusInvalidBirthPlan;
            return;
        }
        if (
            !mini_alive_mask[kChildRow]
            || mini_slot_to_agent_id[kChildRow] < 0
        ) {
            scratch_status = kStatusChildIdentityNotCommitted;
            return;
        }
        const double child_energy =
            mini_policy_energy_observation[kChildRow];
        const float sp_hunger = mini_genome_drive_sp_hunger[kChildRow];
        const float sp_pain = mini_genome_drive_sp_pain[kChildRow];
        const float sp_fatigue = mini_genome_drive_sp_fatigue[kChildRow];
        const float sp_fear = mini_genome_drive_sp_fear[kChildRow];
        const float exp_prior_noise =
            mini_genome_exp_prior_noise[kChildRow];
        const float birth_noise =
            mini_genome_drive_birth_noise[kChildRow];
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
            scratch_status = kStatusInvalidChildScalarState;
            return;
        }
    }

    detail::Pcg64State preflight_rng = detail::pcg64_detail::load_rng(
        scratch_population_pcg64_state_words,
        scratch_population_pcg64_buffer_words
    );
    int64_t planned_birth_noise_rows = 0;
    int64_t planned_exp_prior_rows = 0;
    int64_t planned_semantic_draws = 0;
    if (birth_count == 1) {
        const double clipped_birth_noise =
            detail::pcg64_detail::clamp_double(
                detail::canon_genome_double(
                    mini_genome_drive_birth_noise[kChildRow]
                ),
                0.0,
                0.20
            );
        if (clipped_birth_noise > 0.0) {
            detail::pcg64_detail::consume_standard_normals(
                preflight_rng,
                kBirthNoiseStandardNormalCount
            );
            planned_birth_noise_rows = 1;
            planned_semantic_draws += kBirthNoiseStandardNormalCount;
        }

        const double clipped_exp_prior_noise =
            detail::pcg64_detail::clamp_double(
                static_cast<double>(
                    mini_genome_exp_prior_noise[kChildRow]
                ),
                0.0,
                0.5
            );
        if (clipped_exp_prior_noise > 0.0) {
            detail::pcg64_detail::consume_standard_normals(
                preflight_rng,
                kExpPriorStandardNormalCount
            );
            planned_exp_prior_rows = 2;
            planned_semantic_draws += kExpPriorStandardNormalCount;
        }
    }
    if (
        scratch_population_np_semantic_draws
        > kInt64Max - planned_semantic_draws
    ) {
        scratch_status = kStatusInvalidPcg64State;
        return;
    }

    detail::Pcg64State commit_rng = detail::pcg64_detail::load_rng(
        scratch_population_pcg64_state_words,
        scratch_population_pcg64_buffer_words
    );
    if (birth_count == 1) {
        const float child_sp_hunger =
            mini_genome_drive_sp_hunger[kChildRow];
        const float child_sp_pain = mini_genome_drive_sp_pain[kChildRow];
        const float child_sp_fatigue =
            mini_genome_drive_sp_fatigue[kChildRow];
        const float child_sp_fear = mini_genome_drive_sp_fear[kChildRow];
        const double child_sp_hunger_scalar =
            detail::canon_genome_double(child_sp_hunger);
        const double hunger_from_energy =
            detail::pcg64_detail::clamp_double(
                1.0
                    - mini_policy_energy_observation[kChildRow]
                        / static_cast<double>(max_energy),
                0.0,
                1.0
            );
        scratch_child_drives[0] = detail::clamp01(static_cast<float>(
            0.60 * child_sp_hunger_scalar + 0.40 * hunger_from_energy
        ));
        scratch_child_drives[1] = detail::clamp01(child_sp_pain);
        scratch_child_drives[2] = detail::clamp01(child_sp_fatigue);
        scratch_child_drives[3] = detail::clamp01(child_sp_fear);

        const double clipped_birth_noise =
            detail::pcg64_detail::clamp_double(
                detail::canon_genome_double(
                    mini_genome_drive_birth_noise[kChildRow]
                ),
                0.0,
                0.20
            );
        if (clipped_birth_noise > 0.0) {
            for (int64_t drive = 0; drive < kDriveCount; ++drive) {
                const float normal = static_cast<float>(
                    detail::pcg64_detail::pcg64_numpy_standard_normal(
                        commit_rng
                    )
                );
                const float noise_delta = static_cast<float>(
                    static_cast<float>(clipped_birth_noise) * normal
                );
                scratch_child_drives[drive] = detail::clamp01(
                    static_cast<float>(
                        scratch_child_drives[drive] + noise_delta
                    )
                );
            }
        }

        const float shock0 = ::fabsf(
            scratch_child_drives[0] - child_sp_hunger
        );
        const float shock1 = ::fabsf(
            scratch_child_drives[1] - child_sp_pain
        );
        const float shock2 = ::fabsf(
            scratch_child_drives[2] - child_sp_fatigue
        );
        const float shock3 = ::fabsf(
            scratch_child_drives[3] - child_sp_fear
        );
        const float shock_sum01 = static_cast<float>(shock0 + shock1);
        const float shock_sum012 = static_cast<float>(shock_sum01 + shock2);
        const float shock_sum0123 = static_cast<float>(
            shock_sum012 + shock3
        );
        scratch_child_birth_reset_shock = static_cast<float>(
            shock_sum0123 / 4.0f
        );

        if (mini_genome_exp_prior_noise[kChildRow] > 0.0f) {
            detail::pcg64_detail::consume_standard_normals(
                commit_rng,
                kExpPriorStandardNormalCount
            );
        }
    }

    detail::pcg64_detail::store_rng(
        scratch_population_pcg64_state_words,
        scratch_population_pcg64_buffer_words,
        commit_rng
    );
    scratch_population_np_semantic_draws += planned_semantic_draws;
    scratch_committed_count = birth_count;
    scratch_birth_noise_rows_consumed = planned_birth_noise_rows;
    scratch_exp_prior_noise_rows_consumed = planned_exp_prior_rows;
    scratch_standard_normal_rows_generated = planned_birth_noise_rows;
    scratch_semantic_draws_added = planned_semantic_draws;
    scratch_status = kStatusOk;
}

// Additive multi-birth route. The fixed B=1 mini-state primitive above remains
// the authoritative legacy path; this API reads cell-offset authoritative
// state directly and writes birth-ordered payloads into external scratch.
// Allocation-free bounded transcription of
// persistent_birth_drives::initialize_ordinary_newborn_drives_serial for zero
// through two ordered births. All state pointers and plan slots are offset to
// one simulation cell. scratch_child_drives has shape [2, 4], packed by birth
// ordinal rather than child slot, and scratch_child_birth_reset_shock has two
// elements. The caller copies committed rows to the authoritative child slots.
//
// The Population PCG64 arrays have extents 4 and 2. They are caller-owned
// transaction scratch initialized from the authoritative population cursor.
// On success they contain the committed continuation; on rejection only
// scratch_status changes. Counter/status scalars must be zero-initialized by
// the caller. Pointer validity, physical extents, and positive finite
// max_energy remain launch-boundary obligations. The implementation has no
// per-thread arrays and keeps only one PCG64 cursor live across the two walks.
__device__ __forceinline__ void
prepare_ordinary_newborn_drives_external_scratch_v2(
    const int64_t* __restrict__ slot_to_agent_id,
    const bool* __restrict__ alive_mask,
    const bool* __restrict__ policy_use_gru,
    const double* __restrict__ policy_energy_observation,
    const float* __restrict__ genome_drive_sp_hunger,
    const float* __restrict__ genome_drive_sp_pain,
    const float* __restrict__ genome_drive_sp_fatigue,
    const float* __restrict__ genome_drive_sp_fear,
    const float* __restrict__ genome_exp_prior_noise,
    const float* __restrict__ genome_drive_birth_noise,
    const int64_t* __restrict__ planned_parent_slots,
    const int64_t* __restrict__ planned_child_slots,
    const int64_t birth_count,
    const int64_t slot_count,
    const float max_energy,
    uint64_t* __restrict__ scratch_population_pcg64_state_words,
    int64_t* __restrict__ scratch_population_pcg64_buffer_words,
    int64_t& scratch_population_np_semantic_draws,
    float* __restrict__ scratch_child_drives,
    float* __restrict__ scratch_child_birth_reset_shock,
    int64_t& scratch_committed_count,
    int64_t& scratch_birth_noise_rows_consumed,
    int64_t& scratch_exp_prior_noise_rows_consumed,
    int64_t& scratch_standard_normal_rows_generated,
    int64_t& scratch_semantic_draws_added,
    int32_t& scratch_status,
    const parent_order::BirthParentOrderModeV1 parent_order_mode =
        parent_order::BirthParentOrderModeV1::ascending_agent_id_v1
) {
    if (
        birth_count < 0 || birth_count > slot_count
    ) {
        scratch_status = kStatusInvalidBirthPlan;
        return;
    }

    if (
        scratch_population_pcg64_buffer_words[0] < 0
        || scratch_population_pcg64_buffer_words[0] > 1
        || scratch_population_pcg64_buffer_words[1] < 0
        || static_cast<uint64_t>(
               scratch_population_pcg64_buffer_words[1]
           ) > 0xffffffffULL
        || (scratch_population_pcg64_state_words[2] & 1ULL) == 0ULL
        || scratch_population_np_semantic_draws < 0
    ) {
        scratch_status = kStatusInvalidPcg64State;
        return;
    }

    int64_t previous_parent_id = -1;
    int64_t previous_child_id = -1;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t parent_slot = planned_parent_slots[birth];
        const int64_t child_slot = planned_child_slots[birth];
        if (
            parent_slot < 0 || parent_slot >= slot_count
            || child_slot < 0 || child_slot >= slot_count
            || parent_slot == child_slot
            || !alive_mask[parent_slot]
            || slot_to_agent_id[parent_slot] < 0
        ) {
            scratch_status = kStatusInvalidBirthPlan;
            return;
        }
        if (!alive_mask[child_slot] || slot_to_agent_id[child_slot] < 0) {
            scratch_status = kStatusChildIdentityNotCommitted;
            return;
        }

        const int64_t parent_id = slot_to_agent_id[parent_slot];
        const int64_t child_id = slot_to_agent_id[child_slot];
        if (
            !parent_order::valid_parent_ordinal_v1(
                parent_order_mode,
                birth,
                parent_id,
                previous_parent_id,
                planned_parent_slots,
                slot_to_agent_id
            )
            || (birth > 0 && child_id <= previous_child_id)
        ) {
            scratch_status = kStatusInvalidParentOrder;
            return;
        }
        previous_parent_id = parent_id;
        previous_child_id = child_id;

        for (int64_t other = 0; other < birth_count; ++other) {
            if (
                other != birth
                && planned_child_slots[other] == child_slot
            ) {
                scratch_status = kStatusInvalidBirthPlan;
                return;
            }
            if (planned_parent_slots[other] == child_slot) {
                scratch_status = kStatusInvalidBirthPlan;
                return;
            }
        }
        const double child_energy = policy_energy_observation[child_slot];
        const float sp_hunger = genome_drive_sp_hunger[child_slot];
        const float sp_pain = genome_drive_sp_pain[child_slot];
        const float sp_fatigue = genome_drive_sp_fatigue[child_slot];
        const float sp_fear = genome_drive_sp_fear[child_slot];
        const float exp_prior_noise = genome_exp_prior_noise[child_slot];
        const float birth_noise = genome_drive_birth_noise[child_slot];
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
            scratch_status = kStatusInvalidChildScalarState;
            return;
        }
    }

    detail::Pcg64State rng = detail::pcg64_detail::load_rng(
        scratch_population_pcg64_state_words,
        scratch_population_pcg64_buffer_words
    );
    int64_t planned_birth_noise_rows = 0;
    int64_t planned_exp_prior_rows = 0;
    int64_t planned_semantic_draws = 0;
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t child_slot = planned_child_slots[birth];
        const double clipped_birth_noise =
            detail::pcg64_detail::clamp_double(
                detail::canon_genome_double(
                    genome_drive_birth_noise[child_slot]
                ),
                0.0,
                0.20
            );
        if (clipped_birth_noise > 0.0) {
            detail::pcg64_detail::consume_standard_normals(
                rng,
                kBirthNoiseStandardNormalCount
            );
            ++planned_birth_noise_rows;
            planned_semantic_draws += kBirthNoiseStandardNormalCount;
        }

        const double clipped_exp_prior_noise =
            detail::pcg64_detail::clamp_double(
                static_cast<double>(genome_exp_prior_noise[child_slot]),
                0.0,
                0.5
            );
        if (clipped_exp_prior_noise > 0.0) {
            detail::pcg64_detail::consume_standard_normals(
                rng,
                kExpPriorStandardNormalCount
            );
            planned_exp_prior_rows += 2;
            planned_semantic_draws += kExpPriorStandardNormalCount;
        }
    }
    if (
        scratch_population_np_semantic_draws
        > kInt64Max - planned_semantic_draws
    ) {
        scratch_status = kStatusInvalidPcg64State;
        return;
    }

    rng = detail::pcg64_detail::load_rng(
        scratch_population_pcg64_state_words,
        scratch_population_pcg64_buffer_words
    );
    for (int64_t birth = 0; birth < birth_count; ++birth) {
        const int64_t child_slot = planned_child_slots[birth];
        const int64_t drive_offset = birth * kDriveCount;
        const float child_sp_hunger = genome_drive_sp_hunger[child_slot];
        const float child_sp_pain = genome_drive_sp_pain[child_slot];
        const float child_sp_fatigue = genome_drive_sp_fatigue[child_slot];
        const float child_sp_fear = genome_drive_sp_fear[child_slot];
        const double child_sp_hunger_scalar =
            detail::canon_genome_double(child_sp_hunger);
        const double hunger_from_energy =
            detail::pcg64_detail::clamp_double(
                1.0
                    - policy_energy_observation[child_slot]
                        / static_cast<double>(max_energy),
                0.0,
                1.0
            );
        scratch_child_drives[drive_offset] = detail::clamp01(
            static_cast<float>(
                0.60 * child_sp_hunger_scalar + 0.40 * hunger_from_energy
            )
        );
        scratch_child_drives[drive_offset + 1] =
            detail::clamp01(child_sp_pain);
        scratch_child_drives[drive_offset + 2] =
            detail::clamp01(child_sp_fatigue);
        scratch_child_drives[drive_offset + 3] =
            detail::clamp01(child_sp_fear);

        const double clipped_birth_noise =
            detail::pcg64_detail::clamp_double(
                detail::canon_genome_double(
                    genome_drive_birth_noise[child_slot]
                ),
                0.0,
                0.20
            );
        if (clipped_birth_noise > 0.0) {
            for (int64_t drive = 0; drive < kDriveCount; ++drive) {
                const float normal = static_cast<float>(
                    detail::pcg64_detail::pcg64_numpy_standard_normal(rng)
                );
                const float noise_delta = static_cast<float>(
                    static_cast<float>(clipped_birth_noise) * normal
                );
                scratch_child_drives[drive_offset + drive] =
                    detail::clamp01(static_cast<float>(
                        scratch_child_drives[drive_offset + drive]
                            + noise_delta
                    ));
            }
        }

        const float shock0 = ::fabsf(
            scratch_child_drives[drive_offset] - child_sp_hunger
        );
        const float shock1 = ::fabsf(
            scratch_child_drives[drive_offset + 1] - child_sp_pain
        );
        const float shock2 = ::fabsf(
            scratch_child_drives[drive_offset + 2] - child_sp_fatigue
        );
        const float shock3 = ::fabsf(
            scratch_child_drives[drive_offset + 3] - child_sp_fear
        );
        const float shock_sum01 = static_cast<float>(shock0 + shock1);
        const float shock_sum012 = static_cast<float>(shock_sum01 + shock2);
        const float shock_sum0123 = static_cast<float>(
            shock_sum012 + shock3
        );
        scratch_child_birth_reset_shock[birth] = static_cast<float>(
            shock_sum0123 / 4.0f
        );

        if (genome_exp_prior_noise[child_slot] > 0.0f) {
            detail::pcg64_detail::consume_standard_normals(
                rng,
                kExpPriorStandardNormalCount
            );
        }
    }

    detail::pcg64_detail::store_rng(
        scratch_population_pcg64_state_words,
        scratch_population_pcg64_buffer_words,
        rng
    );
    scratch_population_np_semantic_draws += planned_semantic_draws;
    scratch_committed_count = birth_count;
    scratch_birth_noise_rows_consumed = planned_birth_noise_rows;
    scratch_exp_prior_noise_rows_consumed = planned_exp_prior_rows;
    scratch_standard_normal_rows_generated = planned_birth_noise_rows;
    scratch_semantic_draws_added = planned_semantic_draws;
    scratch_status = kStatusOk;
}

}  // namespace eiel::cuda_architecture_b::flattened_birth_drives_v1
