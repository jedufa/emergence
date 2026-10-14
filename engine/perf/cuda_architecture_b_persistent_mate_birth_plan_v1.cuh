#ifndef EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_MATE_BIRTH_PLAN_V1_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_MATE_BIRTH_PLAN_V1_CUH_

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "cuda_architecture_b_resident_reproduction_exposure_v1.cuh"

#if defined(__CUDACC__)
#define EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD \
    __host__ __device__ __forceinline__
#else
#define EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD inline
#endif

namespace eiel::cuda_architecture_b::persistent_mate_birth_plan_v1 {

constexpr int64_t kInt64Max = 9223372036854775807LL;
constexpr int64_t kInt64Min = -kInt64Max - 1;
constexpr int64_t kMaximumGridDimension = 0x7fffffffLL;
constexpr int32_t kAdjacentDirectionCount = 4;
constexpr int32_t kMateSpawnCandidateCount = 8;

enum PlanStatusV1 : int32_t {
    kStatusOk = 0,
    kStatusInvalidConfig = 1,
    kStatusInvalidResidentState = 2,
    kStatusInvalidAgentIds = 3,
    kStatusInvalidPcg64State = 4,
    kStatusCounterOverflow = 5,
    kStatusExposureOverflow = 6,
};

enum DecisionReasonV1 : int32_t {
    kReasonAccepted = 0,
    kReasonNoSpace = 1,
    kReasonBuffer = 2,
    kReasonNoFreeSlot = 3,
    kReasonNoMate = 4,
    kReasonConception = 5,
    kReasonGestation = 6,
    kReasonHealth = 7,
    kReasonCooldown = 8,
    kReasonNoEnergy = 9,
    kReasonAlreadyMated = 10,
    kReasonTryProbability = 11,
    kReasonCount = 12,
};

struct Pcg64StateV1 {
    uint64_t state_lo;
    uint64_t state_hi;
    uint64_t inc_lo;
    uint64_t inc_hi;
    uint32_t has_uint32;
    uint32_t uinteger;
#if defined(EIEL_C4_R11_BOUNDED_RNG_V1)
    int64_t raw64_transitions;
    int64_t output32_returns;
    bool retry_exhausted;
#endif
};

// Every pointer is already offset to one simulation cell. State pointers are
// authoritative resident state. The planner does not write any of them until
// the complete serial pass and all counter-overflow checks have succeeded.
struct FlatMateBirthPlanStateV1 {
    const int64_t* slot_to_agent_id;
    const bool* alive_mask;
    const int32_t* pos_y;
    const int32_t* pos_x;
    const int32_t* agent_grid;
    const double* energy;
    const float* health;
    const float* injury;
    const float* scar;
    const int32_t* survival_steps;
    const float* chemistry_repro_viability;
    const bool* chemistry_repro_blocked;
    int32_t* repro_cooldown;
    const float* repro_threshold;
    const int64_t* slot_free_queue;
    const int64_t* slot_free_queue_count;
    int64_t* slot_free_queue_head;
    uint64_t* env_pcg64_state_words;
    int64_t* env_pcg64_buffer_words;
    int64_t* policy_np_semantic_draws;
    uint64_t* runner_pcg64_state_words;
    int64_t* runner_pcg64_buffer_words;
    int64_t* metrics_np_semantic_draws;
#if defined(EIEL_C4_R11_BOUNDED_RNG_V1)
    int64_t* env_raw64_transition_count;
    int64_t* env_output32_return_count;
    int64_t* runner_raw64_transition_count;
    int64_t* runner_output32_return_count;
#endif
    // Optional source-current identity projections.  Legacy callers may omit
    // these trailing fields; the planner then retains its serial scans.
    const int32_t* agent_id_to_slot;
    const int64_t* active_agent_ids;
    const int32_t* active_agent_slots;
    const int64_t* active_count;
};

// Every plan and decision array has capacity rows. Spawn positions are
// interleaved [y, x] pairs. planned_mate_slots is -1 for accepted rows only
// when requires_mate is false; every accepted mate-required row has a slot.
struct FlatMateBirthPlanBuffersV1 {
    int32_t* planned_repro_cooldown;
    double* planned_energy;
    int64_t* planned_parent_slots;
    int64_t* planned_mate_slots;
    int64_t* planned_child_slots;
    int32_t* planned_spawn_positions;
    double* planned_child_energy;
    double* planned_parent_energy_after;
    double* planned_mate_energy_after;
    int64_t* decision_parent_slots;
    int64_t* decision_mate_slots;
    int32_t* decision_reason_codes;
    int32_t* decision_adjacent_counts;
    int32_t* decision_spawn_candidate_counts;
    int64_t* reason_counts;
    int64_t* planned_count;
    int64_t* decision_count;
    int64_t* env_rng_bounded_draws_added;
    int64_t* env_rng_semantic_draws_added;
    int64_t* runner_rng_bounded_draws_added;
    int64_t* runner_rng_semantic_draws_added;
    int32_t* status;
};

struct MateBirthPlanConfigV1 {
    int64_t capacity;
    int64_t grid_height;
    int64_t grid_width;
    int64_t free_queue_capacity;
    int64_t max_births;
    double parent_energy_share;
    double minimum_energy_buffer;
    int32_t reproduce_cooldown_steps;
    double reproduce_try_probability;
    double injury_gate;
    double scar_gate;
    double injury_repro_k_injury;
    double injury_repro_k_scar;
    double injury_birth_fail_scar;
    double injury_birth_fail_injury;
    bool env_policy_alias;
    bool requires_mate;
    bool injury_gate_active;
    bool scar_gate_active;
    bool chemistry_viability_active;
    int32_t life_history_mode;
    int32_t maturity_start_ticks;
    int32_t maturity_rise_ticks;
    int32_t fertility_decline_start_ticks;
    int32_t fertility_decline_ticks;
    float elder_fertility_floor;
};

enum MateBirthSelectionModeV1 : int32_t {
    kMateBirthSelectionLegacyAgentIdV1 = 0,
    kMateBirthSelectionUniformNeutralV1 = 1,
};

struct MateBirthSelectionConfigV1 {
    int32_t version;
    int32_t mode;
    uint64_t cell_seed;
    int64_t generation;
    int64_t absolute_tick;
    bool trait_authority_enabled;
    double neutral_repro_threshold;
    resident_reproduction_exposure_v1::FlatResidentReproductionExposureV1
        exposure{};
    int64_t exposure_cell{-1};
    int8_t exposure_path{-1};
};

namespace detail {

EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD uint64_t multiply_high_u64(
    const uint64_t left,
    const uint64_t right
) {
    constexpr uint64_t kLow32Mask = 0xffffffffULL;
    const uint64_t left_low = left & kLow32Mask;
    const uint64_t left_high = left >> 32U;
    const uint64_t right_low = right & kLow32Mask;
    const uint64_t right_high = right >> 32U;
    uint64_t product = left_low * right_low;
    const uint64_t carry = product >> 32U;
    product = left_high * right_low + carry;
    const uint64_t middle_low = product & kLow32Mask;
    const uint64_t middle_high = product >> 32U;
    product = left_low * right_high + middle_low;
    return left_high * right_high + middle_high + (product >> 32U);
}

EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD bool finite(const double value) {
    return value == value && value <= DBL_MAX && value >= -DBL_MAX;
}

EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD bool finite(const float value) {
    return value == value && value <= FLT_MAX && value >= -FLT_MAX;
}

EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD uint64_t neutral_mix_v1(
    uint64_t value
) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD uint64_t neutral_parent_rank_v1(
    const MateBirthSelectionConfigV1 selection,
    const int64_t agent_id
) {
    uint64_t value = selection.cell_seed;
    value ^= neutral_mix_v1(
        static_cast<uint64_t>(selection.generation)
        ^ 0xd1b54a32d192ed03ULL
    );
    value ^= neutral_mix_v1(
        static_cast<uint64_t>(selection.absolute_tick)
        ^ 0x8cb92baa7f3d8dd7ULL
    );
    value ^= neutral_mix_v1(
        static_cast<uint64_t>(agent_id)
        ^ 0xdb4f0b9175ae2165ULL
    );
    return neutral_mix_v1(value);
}

EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD uint64_t pcg64_raw64(
    Pcg64StateV1& rng
) {
#if defined(EIEL_C4_R11_BOUNDED_RNG_V1)
    ++rng.raw64_transitions;
#endif
    constexpr uint64_t kMultiplierLo = 4865540595714422341ULL;
    constexpr uint64_t kMultiplierHi = 2549297995355413924ULL;
    const uint64_t old_lo = rng.state_lo;
    const uint64_t old_hi = rng.state_hi;
    uint64_t next_lo = old_lo * kMultiplierLo;
    uint64_t next_hi = multiply_high_u64(old_lo, kMultiplierLo)
        + old_lo * kMultiplierHi + old_hi * kMultiplierLo;
    const uint64_t before_increment = next_lo;
    next_lo += rng.inc_lo;
    next_hi += rng.inc_hi
        + static_cast<uint64_t>(next_lo < before_increment);
    rng.state_lo = next_lo;
    rng.state_hi = next_hi;

    const uint64_t xorshifted = next_hi ^ next_lo;
    const uint32_t rotation = static_cast<uint32_t>(next_hi >> 58U);
    if (rotation == 0U) {
        return xorshifted;
    }
    return (xorshifted >> rotation)
        | (xorshifted << ((64U - rotation) & 63U));
}

EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD uint32_t pcg64_raw32(
    Pcg64StateV1& rng
) {
#if defined(EIEL_C4_R11_BOUNDED_RNG_V1)
    ++rng.output32_returns;
#endif
    if (rng.has_uint32 != 0U) {
        rng.has_uint32 = 0U;
        return rng.uinteger;
    }
    const uint64_t raw = pcg64_raw64(rng);
    rng.uinteger = static_cast<uint32_t>(raw >> 32U);
    rng.has_uint32 = 1U;
    return static_cast<uint32_t>(raw);
}

EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD double pcg64_double01(
    Pcg64StateV1& rng
) {
    return static_cast<double>(pcg64_raw64(rng) >> 11U) * 0x1.0p-53;
}

// NumPy Generator.shuffle uses an inclusive interval [0, max_value] for each
// reverse Fisher-Yates row. Rejections advance PCG64 but remain one semantic
// bounded draw, matching the resident counter convention.
EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD uint32_t pcg64_bounded_uint32(
    Pcg64StateV1& rng,
    const uint32_t max_value
) {
#if defined(EIEL_C4_R11_BOUNDED_RNG_V1)
    if (rng.retry_exhausted) return 0U;
#endif
    if (max_value == 0xffffffffU) {
        return pcg64_raw32(rng);
    }
    uint32_t mask = max_value;
    mask |= mask >> 1U;
    mask |= mask >> 2U;
    mask |= mask >> 4U;
    mask |= mask >> 8U;
    mask |= mask >> 16U;
    uint32_t value = 0;
#if defined(EIEL_C4_R11_BOUNDED_RNG_V1)
    for (int32_t attempt = 0; attempt < EIEL_C4_R11_PCG_MAX_RAW32_ATTEMPTS;
         ++attempt) {
        value = pcg64_raw32(rng) & mask;
        if (value <= max_value) return value;
    }
    rng.retry_exhausted = true;
    return 0U;
#else
    do {
        value = pcg64_raw32(rng) & mask;
    } while (value > max_value);
    return value;
#endif
}

EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD Pcg64StateV1 load_rng(
    const uint64_t* state_words,
    const int64_t* buffer_words
) {
    return Pcg64StateV1{
        state_words[0],
        state_words[1],
        state_words[2],
        state_words[3],
        static_cast<uint32_t>(buffer_words[0]),
        static_cast<uint32_t>(buffer_words[1]),
#if defined(EIEL_C4_R11_BOUNDED_RNG_V1)
        0,
        0,
        false,
#endif
    };
}

EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD void store_rng(
    uint64_t* state_words,
    int64_t* buffer_words,
    const Pcg64StateV1& rng
) {
    state_words[0] = rng.state_lo;
    state_words[1] = rng.state_hi;
    state_words[2] = rng.inc_lo;
    state_words[3] = rng.inc_hi;
    buffer_words[0] = static_cast<int64_t>(rng.has_uint32);
    buffer_words[1] = static_cast<int64_t>(rng.uinteger);
}

EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD bool counter_add_overflows(
    const int64_t base,
    const int64_t addition
) {
    return base < 0 || addition < 0 || base > kInt64Max - addition;
}

EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD bool position_is_reserved(
    const int32_t y,
    const int32_t x,
    const int32_t* planned_spawn_positions,
    const int64_t planned_count
) {
    for (int64_t row = 0; row < planned_count; ++row) {
        if (planned_spawn_positions[row * 2] == y
            && planned_spawn_positions[row * 2 + 1] == x) {
            return true;
        }
    }
    return false;
}

EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD bool slot_is_mated(
    const int64_t slot,
    const int64_t* planned_parent_slots,
    const int64_t* planned_mate_slots,
    const int64_t planned_count
) {
    for (int64_t row = 0; row < planned_count; ++row) {
        if (planned_parent_slots[row] == slot
            || planned_mate_slots[row] == slot) {
            return true;
        }
    }
    return false;
}

    EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD bool child_slot_is_planned(
    const int64_t slot,
    const int64_t* planned_child_slots,
    const int64_t planned_count
) {
    for (int64_t row = 0; row < planned_count; ++row) {
        if (planned_child_slots[row] == slot) {
            return true;
        }
    }
    return false;
}

EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD double life_history_fertility_scale_v1(
    const int32_t age_ticks,
    const MateBirthPlanConfigV1 config
) {
    if (config.life_history_mode == 0) {
        return 1.0;
    }
    if (age_ticks < config.maturity_start_ticks) {
        return 0.0;
    }
    double scale = 1.0;
    if (config.maturity_rise_ticks > 0) {
        const int32_t elapsed = age_ticks - config.maturity_start_ticks + 1;
        scale = static_cast<double>(elapsed)
            / static_cast<double>(config.maturity_rise_ticks);
        if (scale > 1.0) {
            scale = 1.0;
        }
    }
    if (
        config.fertility_decline_start_ticks > 0
        && age_ticks >= config.fertility_decline_start_ticks
    ) {
        if (config.fertility_decline_ticks <= 0) {
            scale = scale < static_cast<double>(config.elder_fertility_floor)
                ? scale
                : static_cast<double>(config.elder_fertility_floor);
        } else {
            const int32_t elapsed =
                age_ticks - config.fertility_decline_start_ticks + 1;
            double decline = static_cast<double>(elapsed)
                / static_cast<double>(config.fertility_decline_ticks);
            if (decline > 1.0) {
                decline = 1.0;
            }
            const double aged_scale = 1.0
                - decline
                    * (1.0 - static_cast<double>(config.elder_fertility_floor));
            scale = scale < aged_scale ? scale : aged_scale;
        }
    }
    if (scale < 0.0) {
        return 0.0;
    }
    return scale > 1.0 ? 1.0 : scale;
}

EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD double fertility_adjusted_threshold_v1(
    const FlatMateBirthPlanStateV1 state,
    const MateBirthPlanConfigV1 config,
    const MateBirthSelectionConfigV1 selection,
    const int64_t slot
) {
    const double threshold = selection.trait_authority_enabled
        ? static_cast<double>(state.repro_threshold[slot])
        : selection.neutral_repro_threshold;
    const double fertility_scale = life_history_fertility_scale_v1(
        state.survival_steps[slot],
        config
    );
    return fertility_scale <= 0.0 ? DBL_MAX : threshold / fertility_scale;
}

EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD int64_t slot_for_agent_id(
    const FlatMateBirthPlanStateV1 state,
    const MateBirthPlanConfigV1 config,
    const int64_t agent_id
) {
    if (state.agent_id_to_slot != nullptr
        && agent_id >= 0 && agent_id < config.capacity) {
        const int32_t projected = state.agent_id_to_slot[agent_id];
        if (projected >= 0 && projected < config.capacity
            && state.alive_mask[projected]
            && state.slot_to_agent_id[projected] == agent_id) {
            return projected;
        }
        // A source-current dense projection is only an acceleration.  If it
        // is absent or not populated for this ID, preserve the old exact
        // scan/fail-closed path rather than trusting a stale row.
    }
    for (int64_t slot = 0; slot < config.capacity; ++slot) {
        if (state.alive_mask[slot]
            && state.slot_to_agent_id[slot] == agent_id) {
            return slot;
        }
    }
    return -1;
}

EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD bool within_gate(
    const FlatMateBirthPlanStateV1 state,
    const MateBirthPlanConfigV1 config,
    const int64_t slot
) {
    return (!config.injury_gate_active
            || static_cast<double>(state.injury[slot]) <= config.injury_gate)
        && (!config.scar_gate_active
            || static_cast<double>(state.scar[slot]) <= config.scar_gate)
        && (!config.chemistry_viability_active
            || !state.chemistry_repro_blocked[slot]);
}

EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD bool viable_mate(
    const FlatMateBirthPlanStateV1 state,
    const FlatMateBirthPlanBuffersV1 buffers,
    const MateBirthPlanConfigV1 config,
    const MateBirthSelectionConfigV1 selection,
    const int64_t slot,
    const int64_t planned_count
) {
    return slot >= 0
        && state.alive_mask[slot]
        && state.health[slot] > 0.0f
        && buffers.planned_repro_cooldown[slot] == 0
        && !slot_is_mated(
            slot,
            buffers.planned_parent_slots,
            buffers.planned_mate_slots,
            planned_count
        )
        && buffers.planned_energy[slot]
            >= fertility_adjusted_threshold_v1(
                state,
                config,
                selection,
                slot
            )
        && within_gate(state, config, slot);
}

}  // namespace detail

EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD void initialize_outputs(
    const FlatMateBirthPlanStateV1 state,
    const FlatMateBirthPlanBuffersV1 buffers,
    const MateBirthPlanConfigV1 config
) {
    const int64_t rows = config.capacity >= 0 ? config.capacity : 0;
    for (int64_t row = 0; row < rows; ++row) {
        buffers.planned_repro_cooldown[row] = state.repro_cooldown[row];
        buffers.planned_energy[row] = state.energy[row];
        buffers.decision_parent_slots[row] = -1;
        buffers.decision_mate_slots[row] = -1;
        buffers.decision_reason_codes[row] = -1;
        buffers.decision_adjacent_counts[row] = 0;
        buffers.decision_spawn_candidate_counts[row] = 0;
    }
    const int64_t plan_rows = config.max_births >= 0
        ? config.max_births
        : 0;
    for (int64_t row = 0; row < plan_rows; ++row) {
        buffers.planned_parent_slots[row] = -1;
        buffers.planned_mate_slots[row] = -1;
        buffers.planned_child_slots[row] = -1;
        buffers.planned_spawn_positions[row * 2] = -1;
        buffers.planned_spawn_positions[row * 2 + 1] = -1;
        buffers.planned_child_energy[row] = NAN;
        buffers.planned_parent_energy_after[row] = NAN;
        buffers.planned_mate_energy_after[row] = NAN;
    }
    for (int32_t reason = 0; reason < kReasonCount; ++reason) {
        buffers.reason_counts[reason] = 0;
    }
    buffers.planned_count[0] = 0;
    buffers.decision_count[0] = 0;
    buffers.env_rng_bounded_draws_added[0] = 0;
    buffers.env_rng_semantic_draws_added[0] = 0;
    buffers.runner_rng_bounded_draws_added[0] = 0;
    buffers.runner_rng_semantic_draws_added[0] = 0;
    buffers.status[0] = kStatusOk;
}

// Serial, allocation-free extraction of mate-required ordinary reproduction
// from runner_ticks_reproduction.py. Parent order is the tick-start alive set
// sorted by agent ID. Mate adjacency is N/S/W/E, shuffled on runner.np_rng;
// the eight raw parent/mate spawn coordinates are shuffled on env.rng before
// bounds, occupancy, duplicate, and reservation filtering. Plan buffers are
// non-authoritative scratch. Cooldown aging happens before every later parent
// gate, and accepted energy debits feed later scratch-only eligibility/share
// calculations. Resident cooldown, free-head, RNG, and stream counters commit
// together only after the complete pass succeeds; resident energy is left for
// the downstream birth commit represented by the accepted plan rows.
EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD void plan_births_serial_v1(
    const FlatMateBirthPlanStateV1 state,
    const FlatMateBirthPlanBuffersV1 buffers,
    const MateBirthPlanConfigV1 config,
    const MateBirthSelectionConfigV1 selection = {
        1,
        kMateBirthSelectionLegacyAgentIdV1,
        0ULL,
        0,
        0,
        true,
        0.0,
    }
) {
    initialize_outputs(state, buffers, config);

    const bool invalid_grid_product = config.grid_height > 0
        && config.grid_width > kInt64Max / config.grid_height;
    if (config.capacity < 0
        || config.max_births < 0
        || config.max_births > config.capacity
        || config.grid_height <= 0
        || config.grid_width <= 0
        || config.grid_height > kMaximumGridDimension
        || config.grid_width > kMaximumGridDimension
        || invalid_grid_product
        || config.free_queue_capacity < 0
        || !detail::finite(config.parent_energy_share)
        || config.parent_energy_share < 0.0
        || config.parent_energy_share > 1.0
        || !detail::finite(config.minimum_energy_buffer)
        || config.minimum_energy_buffer < 0.0
        || config.reproduce_cooldown_steps < 0
        || !detail::finite(config.reproduce_try_probability)
        || config.reproduce_try_probability < 0.0
        || config.reproduce_try_probability > 1.0
        || !detail::finite(config.injury_gate)
        || !detail::finite(config.scar_gate)
        || !detail::finite(config.injury_repro_k_injury)
        || config.injury_repro_k_injury < 0.0
        || !detail::finite(config.injury_repro_k_scar)
        || config.injury_repro_k_scar < 0.0
        || !detail::finite(config.injury_birth_fail_scar)
        || config.injury_birth_fail_scar < 0.0
        || !detail::finite(config.injury_birth_fail_injury)
        || config.injury_birth_fail_injury < 0.0
        || config.life_history_mode < 0
        || config.life_history_mode > 1
        || config.maturity_start_ticks < 0
        || config.maturity_rise_ticks < 0
        || config.fertility_decline_start_ticks < 0
        || config.fertility_decline_ticks < 0
        || !detail::finite(static_cast<double>(config.elder_fertility_floor))
        || config.elder_fertility_floor < 0.0f
        || config.elder_fertility_floor > 1.0f
        || selection.version != 1
        || (
            selection.mode != kMateBirthSelectionLegacyAgentIdV1
            && selection.mode != kMateBirthSelectionUniformNeutralV1
        )
        || !detail::finite(selection.neutral_repro_threshold)
        || selection.neutral_repro_threshold < 0.0) {
        buffers.status[0] = kStatusInvalidConfig;
        return;
    }

    const int64_t free_count = state.slot_free_queue_count[0];
    const int64_t initial_free_head = state.slot_free_queue_head[0];
    if (free_count < 0
        || free_count > config.free_queue_capacity
        || initial_free_head < 0
        || initial_free_head > free_count
        || state.policy_np_semantic_draws[0] < 0
        || state.metrics_np_semantic_draws[0] < 0) {
        buffers.status[0] = kStatusInvalidResidentState;
        return;
    }
    if (state.env_pcg64_buffer_words[0] < 0
        || state.env_pcg64_buffer_words[0] > 1
        || state.env_pcg64_buffer_words[1] < 0
        || static_cast<uint64_t>(state.env_pcg64_buffer_words[1])
            > 0xffffffffULL
        || (state.env_pcg64_state_words[2] & 1ULL) == 0ULL
        || state.runner_pcg64_buffer_words[0] < 0
        || state.runner_pcg64_buffer_words[0] > 1
        || state.runner_pcg64_buffer_words[1] < 0
        || static_cast<uint64_t>(state.runner_pcg64_buffer_words[1])
            > 0xffffffffULL
        || (state.runner_pcg64_state_words[2] & 1ULL) == 0ULL) {
        buffers.status[0] = kStatusInvalidPcg64State;
        return;
    }

    int64_t alive_count = 0;
    for (int64_t slot = 0; slot < config.capacity; ++slot) {
        if (!state.alive_mask[slot]) {
            if (state.slot_to_agent_id[slot] >= 0) {
                buffers.status[0] = kStatusInvalidResidentState;
                return;
            }
            continue;
        }
        ++alive_count;
        const int64_t agent_id = state.slot_to_agent_id[slot];
        if (agent_id < 0
            || agent_id > 0x7fffffffLL
            || state.pos_y[slot] < 0
            || state.pos_y[slot] >= config.grid_height
            || state.pos_x[slot] < 0
            || state.pos_x[slot] >= config.grid_width
            || state.agent_grid[
                static_cast<int64_t>(state.pos_y[slot]) * config.grid_width
                    + state.pos_x[slot]
            ] != static_cast<int32_t>(agent_id)
            || !detail::finite(state.energy[slot])
            || !detail::finite(state.health[slot])
            || !detail::finite(state.injury[slot])
            || state.injury[slot] < 0.0f
            || !detail::finite(state.scar[slot])
            || state.scar[slot] < 0.0f
            || !detail::finite(state.repro_threshold[slot])
            || state.survival_steps[slot] < 0
            || state.repro_cooldown[slot] < 0
            || (config.chemistry_viability_active
                && (!detail::finite(state.chemistry_repro_viability[slot])
                    || state.chemistry_repro_viability[slot] < 0.0f))) {
            buffers.status[0] = kStatusInvalidResidentState;
            return;
        }
        for (int64_t previous = 0; previous < slot; ++previous) {
            if (state.alive_mask[previous]
                && state.slot_to_agent_id[previous] == agent_id) {
                buffers.status[0] = kStatusInvalidAgentIds;
                return;
            }
        }
    }

    Pcg64StateV1 env_rng = detail::load_rng(
        state.env_pcg64_state_words,
        state.env_pcg64_buffer_words
    );
    Pcg64StateV1 runner_rng = detail::load_rng(
        state.runner_pcg64_state_words,
        state.runner_pcg64_buffer_words
    );
    int64_t free_head = initial_free_head;
    int64_t planned_count = 0;
    int64_t decision_count = 0;
    int64_t env_bounded_draws = 0;
    int64_t env_semantic_draws = 0;
    int64_t runner_bounded_draws = 0;
    int64_t runner_semantic_draws = 0;
    int64_t previous_agent_id = kInt64Min;
    int64_t previous_slot = -1;
    uint64_t previous_neutral_rank = ~static_cast<uint64_t>(0);

    // R11's sorted active projection is validated by the preceding semantic
    // preflight.  A post-death shadow may still point at the pre-death list,
    // so require count, slot, liveness, and strict ID order to agree before
    // using it.  Any mismatch falls back to the original serial parent scan.
    bool indexed_parent_order = selection.mode
        == kMateBirthSelectionLegacyAgentIdV1
        && state.active_agent_ids != nullptr
        && state.active_agent_slots != nullptr
        && state.active_count != nullptr
        && state.active_count[0] == alive_count;
    if (indexed_parent_order) {
        int64_t prior_id = kInt64Min;
        for (int64_t row = 0; row < alive_count; ++row) {
            const int32_t candidate_slot = state.active_agent_slots[row];
            const int64_t candidate_id = state.active_agent_ids[row];
            if (candidate_slot < 0 || candidate_slot >= config.capacity
                || !state.alive_mask[candidate_slot]
                || state.slot_to_agent_id[candidate_slot] != candidate_id
                || (row > 0 && candidate_id <= prior_id)) {
                indexed_parent_order = false;
                break;
            }
            prior_id = candidate_id;
        }
    }

    constexpr int32_t kDy[kAdjacentDirectionCount] = {-1, 1, 0, 0};
    constexpr int32_t kDx[kAdjacentDirectionCount] = {0, 0, -1, 1};

    for (int64_t order = 0; order < alive_count; ++order) {
        if (planned_count >= config.max_births) {
            break;
        }

        int64_t parent_slot = -1;
        int64_t parent_agent_id = kInt64Max;
        uint64_t parent_neutral_rank = 0;
        if (indexed_parent_order) {
            parent_slot = state.active_agent_slots[order];
            parent_agent_id = state.active_agent_ids[order];
        } else {
            for (int64_t slot = 0; slot < config.capacity; ++slot) {
                if (!state.alive_mask[slot]) {
                    continue;
                }
                const int64_t agent_id = state.slot_to_agent_id[slot];
                const uint64_t neutral_rank =
                    selection.mode == kMateBirthSelectionUniformNeutralV1
                    ? detail::neutral_parent_rank_v1(selection, agent_id)
                    : 0;
                const bool follows_previous =
                    selection.mode == kMateBirthSelectionLegacyAgentIdV1
                    ? (
                        agent_id > previous_agent_id
                        || (
                            agent_id == previous_agent_id
                            && slot > previous_slot
                        )
                    )
                    : (
                        order == 0
                        || neutral_rank < previous_neutral_rank
                        || (
                            neutral_rank == previous_neutral_rank
                            && (
                                agent_id < previous_agent_id
                                || (
                                    agent_id == previous_agent_id
                                    && slot < previous_slot
                                )
                            )
                        )
                    );
                if (follows_previous
                    && (
                        parent_slot < 0
                        || (
                            selection.mode
                                == kMateBirthSelectionLegacyAgentIdV1
                            && (
                                agent_id < parent_agent_id
                                || (
                                    agent_id == parent_agent_id
                                    && slot < parent_slot
                                )
                            )
                        )
                        || (
                            selection.mode
                                == kMateBirthSelectionUniformNeutralV1
                            && (
                                neutral_rank > parent_neutral_rank
                                || (
                                    neutral_rank == parent_neutral_rank
                                    && (
                                        agent_id > parent_agent_id
                                        || (
                                            agent_id == parent_agent_id
                                            && slot > parent_slot
                                        )
                                    )
                                )
                            )
                        )
                    )) {
                    parent_slot = slot;
                    parent_agent_id = agent_id;
                    parent_neutral_rank = neutral_rank;
                }
            }
        }
        if (parent_slot < 0) {
            buffers.status[0] = kStatusInvalidAgentIds;
            return;
        }
        previous_agent_id = parent_agent_id;
        previous_slot = parent_slot;
        previous_neutral_rank = parent_neutral_rank;

        const int64_t decision = decision_count++;
        buffers.decision_parent_slots[decision] = parent_slot;
        int32_t reason = kReasonAccepted;
        const double parent_fertility_scale =
            detail::life_history_fertility_scale_v1(
                state.survival_steps[parent_slot],
                config
            );

        if (parent_fertility_scale <= 0.0) {
            reason = kReasonNoMate;
        } else if (buffers.planned_repro_cooldown[parent_slot] > 0) {
            --buffers.planned_repro_cooldown[parent_slot];
        }
        if (
            reason == kReasonAccepted
            && buffers.planned_repro_cooldown[parent_slot] != 0
        ) {
            reason = kReasonCooldown;
        } else if (detail::slot_is_mated(
                       parent_slot,
                       buffers.planned_parent_slots,
                       buffers.planned_mate_slots,
                       planned_count
                   )) {
            reason = kReasonAlreadyMated;
        } else if (state.health[parent_slot] <= 0.0f) {
            reason = kReasonHealth;
        } else if (
            buffers.planned_energy[parent_slot]
            < (
                selection.trait_authority_enabled
                ? static_cast<double>(state.repro_threshold[parent_slot])
                : selection.neutral_repro_threshold
            ) / parent_fertility_scale
        ) {
            reason = kReasonNoEnergy;
        } else if ((config.injury_gate_active
                    && static_cast<double>(state.injury[parent_slot])
                        > config.injury_gate)
                   || (config.scar_gate_active
                       && static_cast<double>(state.scar[parent_slot])
                           > config.scar_gate)
                   || (config.chemistry_viability_active
                       && state.chemistry_repro_blocked[parent_slot])) {
            reason = kReasonHealth;
        }

        int64_t mate_slot = -1;
        int64_t exposure_attempt = -1;
        int64_t exposure_primary_snapshot = -1;
        int64_t exposure_mate_snapshot = -1;
        int64_t adjacent_ids[kAdjacentDirectionCount] = {-1, -1, -1, -1};
        int32_t adjacent_count = 0;
        if (reason == kReasonAccepted && config.requires_mate) {
            const int32_t parent_y = state.pos_y[parent_slot];
            const int32_t parent_x = state.pos_x[parent_slot];
            for (int32_t direction = 0;
                 direction < kAdjacentDirectionCount;
                 ++direction) {
                const int32_t y = parent_y + kDy[direction];
                const int32_t x = parent_x + kDx[direction];
                if (y < 0
                    || y >= config.grid_height
                    || x < 0
                    || x >= config.grid_width) {
                    continue;
                }
                const int64_t candidate_id = static_cast<int64_t>(
                    state.agent_grid[
                        static_cast<int64_t>(y) * config.grid_width + x
                    ]
                );
                if (candidate_id >= 0 && candidate_id != parent_agent_id) {
                    adjacent_ids[adjacent_count++] = candidate_id;
                }
            }
            const auto& exposure = selection.exposure;
            if (
                resident_reproduction_exposure_v1::enabled_v1(exposure)
                && exposure.cell_enabled != nullptr
                && exposure.cell_enabled[selection.exposure_cell]
            ) {
                int32_t viable_mate_count = 0;
                for (int32_t index = 0; index < adjacent_count; ++index) {
                    const int64_t candidate_slot = detail::slot_for_agent_id(
                        state, config, adjacent_ids[index]
                    );
                    viable_mate_count += detail::viable_mate(
                        state,
                        buffers,
                        config,
                        selection,
                        candidate_slot,
                        planned_count
                    );
                }
                if (
                    viable_mate_count > 0
                    && !resident_reproduction_exposure_v1::
                        reserve_attempt_with_two_snapshots_v1(
                            exposure,
                            selection.exposure_cell,
                            &exposure_attempt,
                            &exposure_primary_snapshot,
                            &exposure_mate_snapshot
                        )
                ) {
                    buffers.status[0] = kStatusExposureOverflow;
                    return;
                }
                if (viable_mate_count > 0) {
                    resident_reproduction_exposure_v1::begin_snapshot_v1(
                        exposure,
                        selection.exposure_cell,
                        exposure_primary_snapshot,
                        selection.exposure_path,
                        selection.generation,
                        selection.absolute_tick
                    );
                    resident_reproduction_exposure_v1::begin_snapshot_v1(
                        exposure,
                        selection.exposure_cell,
                        exposure_mate_snapshot,
                        selection.exposure_path,
                        selection.generation,
                        selection.absolute_tick
                    );
                    for (
                        int64_t candidate_slot = 0;
                        candidate_slot < config.capacity;
                        ++candidate_slot
                    ) {
                        if (!state.alive_mask[candidate_slot]) {
                            continue;
                        }
                        const int64_t candidate_id =
                            state.slot_to_agent_id[candidate_slot];
                        const uint64_t candidate_rank =
                            selection.mode == kMateBirthSelectionUniformNeutralV1
                            ? detail::neutral_parent_rank_v1(
                                selection, candidate_id
                            )
                            : 0;
                        const bool not_yet_processed =
                            candidate_slot == parent_slot
                            || (
                                selection.mode
                                    == kMateBirthSelectionLegacyAgentIdV1
                                ? candidate_id > parent_agent_id
                                : (
                                    candidate_rank < parent_neutral_rank
                                    || (
                                        candidate_rank == parent_neutral_rank
                                        && (
                                            candidate_id < parent_agent_id
                                            || (
                                                candidate_id
                                                    == parent_agent_id
                                                && candidate_slot < parent_slot
                                            )
                                        )
                                    )
                                )
                            );
                        const bool eligible =
                            not_yet_processed
                            && buffers.planned_repro_cooldown[candidate_slot] == 0
                            && !detail::slot_is_mated(
                                candidate_slot,
                                buffers.planned_parent_slots,
                                buffers.planned_mate_slots,
                                planned_count
                            )
                            && state.health[candidate_slot] > 0.0f
                            && buffers.planned_energy[candidate_slot]
                                >= detail::fertility_adjusted_threshold_v1(
                                    state,
                                    config,
                                    selection,
                                    candidate_slot
                                )
                            && !(
                                config.injury_gate_active
                                && static_cast<double>(
                                    state.injury[candidate_slot]
                                ) > config.injury_gate
                            )
                            && !(
                                config.scar_gate_active
                                && static_cast<double>(
                                    state.scar[candidate_slot]
                                ) > config.scar_gate
                            )
                            && !(
                                config.chemistry_viability_active
                                && state.chemistry_repro_blocked[candidate_slot]
                            );
                        if (
                            eligible
                            && !resident_reproduction_exposure_v1::
                                append_role_eligible_parent_v1(
                                    exposure,
                                    selection.exposure_cell,
                                    exposure_primary_snapshot,
                                    resident_reproduction_exposure_v1::
                                        kPrimaryParentRole,
                                    candidate_id
                                )
                        ) {
                            buffers.status[0] = kStatusExposureOverflow;
                            return;
                        }
                    }
                    for (int32_t index = 0; index < adjacent_count; ++index) {
                        const int64_t candidate_slot =
                            detail::slot_for_agent_id(
                                state, config, adjacent_ids[index]
                            );
                        if (
                            detail::viable_mate(
                                state,
                                buffers,
                                config,
                                selection,
                                candidate_slot,
                                planned_count
                            )
                            && !resident_reproduction_exposure_v1::
                                append_role_eligible_parent_v1(
                                    exposure,
                                    selection.exposure_cell,
                                    exposure_mate_snapshot,
                                    resident_reproduction_exposure_v1::
                                        kMateParentRole,
                                    adjacent_ids[index]
                                )
                        ) {
                            buffers.status[0] = kStatusExposureOverflow;
                            return;
                        }
                    }
                }
            }
            if (runner_semantic_draws > kInt64Max - adjacent_count) {
                buffers.status[0] = kStatusCounterOverflow;
                return;
            }
            runner_semantic_draws += adjacent_count;
            for (int32_t index = adjacent_count - 1; index > 0; --index) {
                const int32_t selected = static_cast<int32_t>(
                    detail::pcg64_bounded_uint32(
                        runner_rng,
                        static_cast<uint32_t>(index)
                    )
                );
                ++runner_bounded_draws;
                const int64_t value = adjacent_ids[index];
                adjacent_ids[index] = adjacent_ids[selected];
                adjacent_ids[selected] = value;
            }
            for (int32_t index = 0; index < adjacent_count; ++index) {
                const int64_t candidate_slot = detail::slot_for_agent_id(
                    state,
                    config,
                    adjacent_ids[index]
                );
                if (detail::viable_mate(
                        state,
                        buffers,
                        config,
                        selection,
                        candidate_slot,
                        planned_count
                    )) {
                    mate_slot = candidate_slot;
                    break;
                }
            }
            if (mate_slot < 0) {
                reason = kReasonNoMate;
            }
        }
        buffers.decision_adjacent_counts[decision] = adjacent_count;
        buffers.decision_mate_slots[decision] = mate_slot;

        int32_t selected_y = -1;
        int32_t selected_x = -1;
        int32_t spawn_candidate_count = 0;
        if (reason == kReasonAccepted) {
            int32_t candidate_y[kMateSpawnCandidateCount];
            int32_t candidate_x[kMateSpawnCandidateCount];
            int32_t raw_candidate_count = 0;
            const int32_t parent_y = state.pos_y[parent_slot];
            const int32_t parent_x = state.pos_x[parent_slot];
            for (int32_t direction = 0;
                 direction < kAdjacentDirectionCount;
                 ++direction) {
                candidate_y[raw_candidate_count] = parent_y + kDy[direction];
                candidate_x[raw_candidate_count] = parent_x + kDx[direction];
                ++raw_candidate_count;
            }
            if (config.requires_mate) {
                const int32_t mate_y = state.pos_y[mate_slot];
                const int32_t mate_x = state.pos_x[mate_slot];
                for (int32_t direction = 0;
                     direction < kAdjacentDirectionCount;
                     ++direction) {
                    candidate_y[raw_candidate_count] = mate_y + kDy[direction];
                    candidate_x[raw_candidate_count] = mate_x + kDx[direction];
                    ++raw_candidate_count;
                }
            } else {
                // Asexual mode follows find_adjacent_empty: only currently empty,
                // in-bounds coordinates enter the environment shuffle.
                int32_t filtered_count = 0;
                for (int32_t index = 0;
                     index < raw_candidate_count;
                     ++index) {
                    const int32_t y = candidate_y[index];
                    const int32_t x = candidate_x[index];
                    if (y >= 0
                        && y < config.grid_height
                        && x >= 0
                        && x < config.grid_width
                        && state.agent_grid[
                            static_cast<int64_t>(y) * config.grid_width + x
                        ] == -1) {
                        candidate_y[filtered_count] = y;
                        candidate_x[filtered_count] = x;
                        ++filtered_count;
                    }
                }
                raw_candidate_count = filtered_count;
            }

            if (env_semantic_draws > kInt64Max - raw_candidate_count) {
                buffers.status[0] = kStatusCounterOverflow;
                return;
            }
            env_semantic_draws += raw_candidate_count;
            for (int32_t index = raw_candidate_count - 1;
                 index > 0;
                 --index) {
                const int32_t selected = static_cast<int32_t>(
                    detail::pcg64_bounded_uint32(
                        env_rng,
                        static_cast<uint32_t>(index)
                    )
                );
                ++env_bounded_draws;
                const int32_t value_y = candidate_y[index];
                const int32_t value_x = candidate_x[index];
                candidate_y[index] = candidate_y[selected];
                candidate_x[index] = candidate_x[selected];
                candidate_y[selected] = value_y;
                candidate_x[selected] = value_x;
            }

            for (int32_t index = 0; index < raw_candidate_count; ++index) {
                const int32_t y = candidate_y[index];
                const int32_t x = candidate_x[index];
                if (y < 0
                    || y >= config.grid_height
                    || x < 0
                    || x >= config.grid_width
                    || state.agent_grid[
                        static_cast<int64_t>(y) * config.grid_width + x
                    ] != -1
                    || detail::position_is_reserved(
                        y,
                        x,
                        buffers.planned_spawn_positions,
                        planned_count
                    )) {
                    continue;
                }
                bool duplicate = false;
                for (int32_t previous = 0; previous < index; ++previous) {
                    if (candidate_y[previous] == y
                        && candidate_x[previous] == x) {
                        const bool previous_valid = candidate_y[previous] >= 0
                            && candidate_y[previous] < config.grid_height
                            && candidate_x[previous] >= 0
                            && candidate_x[previous] < config.grid_width
                            && state.agent_grid[
                                static_cast<int64_t>(candidate_y[previous])
                                    * config.grid_width
                                    + candidate_x[previous]
                            ] == -1
                            && !detail::position_is_reserved(
                                candidate_y[previous],
                                candidate_x[previous],
                                buffers.planned_spawn_positions,
                                planned_count
                            );
                        duplicate = previous_valid;
                        if (duplicate) {
                            break;
                        }
                    }
                }
                if (duplicate) {
                    continue;
                }
                ++spawn_candidate_count;
                if (selected_y < 0) {
                    selected_y = y;
                    selected_x = x;
                }
            }
            if (selected_y < 0) {
                reason = kReasonNoSpace;
            }
        }
        buffers.decision_spawn_candidate_counts[decision] =
            spawn_candidate_count;

        if (reason == kReasonAccepted
            && config.reproduce_try_probability < 1.0) {
            if (runner_semantic_draws == kInt64Max) {
                buffers.status[0] = kStatusCounterOverflow;
                return;
            }
            ++runner_semantic_draws;
            if (detail::pcg64_double01(runner_rng)
                > config.reproduce_try_probability) {
                reason = kReasonTryProbability;
            }
        }

        const double parent_energy = buffers.planned_energy[parent_slot];
        const double parent_share =
            parent_energy * config.parent_energy_share;
        double mate_energy = 0.0;
        double mate_share = 0.0;
        if (mate_slot >= 0) {
            mate_energy = buffers.planned_energy[mate_slot];
            mate_share = mate_energy * config.parent_energy_share;
        }
        if (reason == kReasonAccepted
            && (parent_energy - parent_share < config.minimum_energy_buffer
                || (mate_slot >= 0
                    && mate_energy - mate_share
                        < config.minimum_energy_buffer))) {
            reason = kReasonBuffer;
        }

        if (reason == kReasonAccepted
            && (config.injury_repro_k_injury > 0.0
                || config.injury_repro_k_scar > 0.0)) {
            const double parent_probability = ::exp(-(
                config.injury_repro_k_injury
                    * static_cast<double>(state.injury[parent_slot])
                + config.injury_repro_k_scar
                    * static_cast<double>(state.scar[parent_slot])
            )) * (config.chemistry_viability_active
                    ? static_cast<double>(
                        state.chemistry_repro_viability[parent_slot]
                    )
                    : 1.0);
            if (runner_semantic_draws == kInt64Max) {
                buffers.status[0] = kStatusCounterOverflow;
                return;
            }
            ++runner_semantic_draws;
            if (detail::pcg64_double01(runner_rng) > parent_probability) {
                reason = kReasonConception;
            }
            if (reason == kReasonAccepted && mate_slot >= 0) {
                const double mate_probability = ::exp(-(
                    config.injury_repro_k_injury
                        * static_cast<double>(state.injury[mate_slot])
                    + config.injury_repro_k_scar
                        * static_cast<double>(state.scar[mate_slot])
                )) * (config.chemistry_viability_active
                        ? static_cast<double>(
                            state.chemistry_repro_viability[mate_slot]
                        )
                        : 1.0);
                if (runner_semantic_draws == kInt64Max) {
                    buffers.status[0] = kStatusCounterOverflow;
                    return;
                }
                ++runner_semantic_draws;
                if (detail::pcg64_double01(runner_rng) > mate_probability) {
                    reason = kReasonConception;
                }
            }
        }

        if (reason == kReasonAccepted
            && (config.injury_birth_fail_scar > 0.0
                || config.injury_birth_fail_injury > 0.0)) {
            double failure_probability = config.injury_birth_fail_scar
                    * static_cast<double>(state.scar[parent_slot])
                + config.injury_birth_fail_injury
                    * static_cast<double>(state.injury[parent_slot]);
            if (mate_slot >= 0) {
                failure_probability += config.injury_birth_fail_scar
                        * static_cast<double>(state.scar[mate_slot])
                    + config.injury_birth_fail_injury
                        * static_cast<double>(state.injury[mate_slot]);
            }
            if (failure_probability > 0.95) {
                failure_probability = 0.95;
            }
            if (runner_semantic_draws == kInt64Max) {
                buffers.status[0] = kStatusCounterOverflow;
                return;
            }
            ++runner_semantic_draws;
            if (detail::pcg64_double01(runner_rng) < failure_probability) {
                reason = kReasonGestation;
            }
        }

        if (reason == kReasonAccepted) {
            int64_t child_slot = -1;
            while (free_head < free_count) {
                const int64_t candidate_slot =
                    state.slot_free_queue[free_head++];
                if (candidate_slot >= 0
                    && candidate_slot < config.capacity
                    && !state.alive_mask[candidate_slot]
                    && state.slot_to_agent_id[candidate_slot] < 0
                    && !detail::child_slot_is_planned(
                        candidate_slot,
                        buffers.planned_child_slots,
                        planned_count
                    )) {
                    child_slot = candidate_slot;
                    break;
                }
            }
            if (child_slot < 0) {
                reason = kReasonNoFreeSlot;
            } else {
                buffers.planned_parent_slots[planned_count] = parent_slot;
                buffers.planned_mate_slots[planned_count] =
                    config.requires_mate ? mate_slot : -1;
                buffers.planned_child_slots[planned_count] = child_slot;
                buffers.planned_spawn_positions[planned_count * 2] =
                    selected_y;
                buffers.planned_spawn_positions[planned_count * 2 + 1] =
                    selected_x;
                buffers.planned_child_energy[planned_count] =
                    parent_share + mate_share;
                buffers.planned_parent_energy_after[planned_count] =
                    parent_energy - parent_share;
                buffers.planned_mate_energy_after[planned_count] =
                    mate_slot >= 0 ? mate_energy - mate_share : NAN;
                buffers.planned_energy[parent_slot] =
                    parent_energy - parent_share;
                if (mate_slot >= 0) {
                    buffers.planned_energy[mate_slot] =
                        mate_energy - mate_share;
                }
                buffers.planned_repro_cooldown[parent_slot] =
                    config.reproduce_cooldown_steps;
                if (mate_slot >= 0) {
                    buffers.planned_repro_cooldown[mate_slot] =
                        config.reproduce_cooldown_steps;
                }
                ++planned_count;
            }
        }

        buffers.decision_reason_codes[decision] = reason;
        if (
            exposure_attempt >= 0
            && !resident_reproduction_exposure_v1::commit_reserved_attempt_v1(
                selection.exposure,
                selection.exposure_cell,
                exposure_attempt,
                exposure_primary_snapshot,
                exposure_mate_snapshot,
                selection.exposure_path,
                selection.generation,
                selection.absolute_tick,
                parent_agent_id,
                state.slot_to_agent_id[mate_slot],
                reason == kReasonAccepted,
                reason,
                reason == kReasonAccepted ? planned_count - 1 : -1
            )
        ) {
            buffers.status[0] = kStatusExposureOverflow;
            return;
        }
        if (reason >= 0 && reason < kReasonCount) {
            ++buffers.reason_counts[reason];
        }
        if (reason == kReasonNoFreeSlot) {
            break;
        }
    }

#if defined(EIEL_C4_R11_BOUNDED_RNG_V1)
    if (env_rng.retry_exhausted || runner_rng.retry_exhausted) {
        buffers.status[0] = kStatusInvalidPcg64State;
        return;
    }
    if (detail::counter_add_overflows(
            state.env_raw64_transition_count[0], env_rng.raw64_transitions)
        || detail::counter_add_overflows(
            state.env_output32_return_count[0], env_rng.output32_returns)
        || detail::counter_add_overflows(
            state.runner_raw64_transition_count[0], runner_rng.raw64_transitions)
        || detail::counter_add_overflows(
            state.runner_output32_return_count[0], runner_rng.output32_returns)) {
        buffers.status[0] = kStatusCounterOverflow;
        return;
    }
#endif

    if ((config.env_policy_alias
         && detail::counter_add_overflows(
             state.policy_np_semantic_draws[0],
             env_semantic_draws
         ))
        || detail::counter_add_overflows(
            state.metrics_np_semantic_draws[0],
            runner_semantic_draws
        )) {
        buffers.status[0] = kStatusCounterOverflow;
        return;
    }

    for (int64_t slot = 0; slot < config.capacity; ++slot) {
        state.repro_cooldown[slot] =
            buffers.planned_repro_cooldown[slot];
    }
    state.slot_free_queue_head[0] = free_head;
    detail::store_rng(
        state.env_pcg64_state_words,
        state.env_pcg64_buffer_words,
        env_rng
    );
    detail::store_rng(
        state.runner_pcg64_state_words,
        state.runner_pcg64_buffer_words,
        runner_rng
    );
#if defined(EIEL_C4_R11_BOUNDED_RNG_V1)
    state.env_raw64_transition_count[0] += env_rng.raw64_transitions;
    state.env_output32_return_count[0] += env_rng.output32_returns;
    state.runner_raw64_transition_count[0] += runner_rng.raw64_transitions;
    state.runner_output32_return_count[0] += runner_rng.output32_returns;
#endif
    if (config.env_policy_alias) {
        state.policy_np_semantic_draws[0] += env_semantic_draws;
    }
    state.metrics_np_semantic_draws[0] += runner_semantic_draws;
    buffers.planned_count[0] = planned_count;
    buffers.decision_count[0] = decision_count;
    buffers.env_rng_bounded_draws_added[0] = env_bounded_draws;
    buffers.env_rng_semantic_draws_added[0] = env_semantic_draws;
    buffers.runner_rng_bounded_draws_added[0] = runner_bounded_draws;
    buffers.runner_rng_semantic_draws_added[0] = runner_semantic_draws;
    buffers.status[0] = kStatusOk;
}

}  // namespace eiel::cuda_architecture_b::persistent_mate_birth_plan_v1

#undef EIEL_PERSISTENT_MATE_BIRTH_PLAN_V1_HD

#endif  // EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_MATE_BIRTH_PLAN_V1_CUH_
