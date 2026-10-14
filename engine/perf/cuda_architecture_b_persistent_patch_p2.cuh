#ifndef EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_PATCH_P2_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_PATCH_P2_CUH_

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace eiel::cuda_architecture_b::persistent_patch_p2 {

constexpr int64_t kStatCount = 25;
constexpr int64_t kMaximumExactPairwiseCells = 256;
constexpr int64_t kMaximumExactSerialCells = 1024;
constexpr int32_t kEmpty = 0;
constexpr int32_t kFood = 1;

enum PatchP2Stat : int64_t {
    kScentEmittedTotal = 0,
    kScentDecayedTotal = 1,
    kWasteEmittedTotal = 2,
    kWasteDecayedTotal = 3,
    kFoodRegrowthTotal = 4,
    kRegrowthBaseTotal = 5,
    kRegrowthEffectiveTotal = 6,
    kRegrowthSuppressedTotal = 7,
    kRegrowthMultiplierMean = 8,
    kRegrowthMultiplierMin = 9,
    kRegrowthMultiplierP10 = 10,
    kRegrowthMultiplierP90 = 11,
    kWasteRegrowthPenaltyMean = 12,
    kWasteRegrowthPenaltyMax = 13,
    kDensityRegrowthPenaltyMean = 14,
    kDensityRegrowthPenaltyMax = 15,
    kHighWasteSuppressionFrac = 16,
    kHighDensitySuppressionFrac = 17,
    kRegrowthAfterWasteTotal = 18,
    kRegrowthAfterRecoveryTotal = 19,
    kRecoveryAddedRegrowthTotal = 20,
    kFoodDepletedFrac = 21,
    kFoodFullFrac = 22,
    kDensityHotspotFrac = 23,
    kRegrowthConsequenceSteps = 24,
};

// Every pointer is already offset to the simulation cell owned by the caller.
struct PatchP2State {
    float* patch_food_amount;
    const float* patch_fertility;
    float* patch_scent;
    float* patch_waste;
    float* patch_scent_emit_pending;
    float* patch_waste_emit_pending;
    const int32_t* agent_grid;
    float* patch_density;
    int32_t* food_grid;
};

// Each scratch array has height * width float32 elements and must not alias.
struct PatchP2Scratch {
    float* multiplier;
    float* waste_penalty;
    float* density_penalty;
    float* sorted_multiplier;
};

// stats points to exactly kStatCount float64 values for the current tick.
struct PatchP2Outputs {
    double* stats;
};

// Values must be validated and clamped by the host binding exactly as in the
// immutable Stage2 Patch P2 owner before this device helper is called.
struct PatchP2Config {
    int64_t height;
    int64_t width;
    float scent_decay_factor;
    float waste_decay_factor;
    double scent_presence_value;
    double waste_presence_value;
    float waste_density_emit_scale;
    float scent_max;
    float waste_max;
    float food_max;
    float food_regrow_per_step;
    bool food_regrow_use_fertility;
    bool food_regrow_cap_by_fertility;
    float food_presence_threshold;
    int64_t density_radius;
    bool density_include_self;
    bool density_normalize;
    bool waste_consequence_enable;
    bool waste_penalty_enable;
    float waste_penalty_scale;
    float waste_penalty_max;
    bool density_penalty_enable;
    float density_penalty_scale;
    float density_penalty_max;
    float regrowth_min_multiplier;
};

constexpr int64_t kRecoveryStatCount = 7;

enum PatchP2RecoveryStat : int64_t {
    kRecoveryPositiveTotal = 0,
    kRecoveryNegativeTotal,
    kRecoveryDecayTotal,
    kRecoveryBoostTotal,
    kRecoveryBoostMean,
    kRecoveryBoostMax,
    kRecoveryBoostP90,
};

struct PatchP2RecoveryContext {
    bool enabled;
    float* activity;
    const float* cover;
    float* boost_scratch;
    float* sorted_boost_scratch;
    double* stats;
    float activity_min;
    float activity_max;
    float fertility_min;
    float fertility_max;
    float decay_factor;
    float base_growth;
    float cover_weight;
    float fertility_weight;
    float waste_penalty_weight;
    float density_penalty_weight;
    float scent_penalty_weight;
    bool regrowth_boost_enabled;
    float regrowth_boost_scale;
    float regrowth_boost_max;
};

#if defined(__CUDACC__)

namespace detail {

__device__ __forceinline__ float f32_add(const float left, const float right) {
    return __fadd_rn(left, right);
}

__device__ __forceinline__ float f32_sub(const float left, const float right) {
    return __fsub_rn(left, right);
}

__device__ __forceinline__ float f32_mul(const float left, const float right) {
    return __fmul_rn(left, right);
}

__device__ __forceinline__ float f32_div(const float left, const float right) {
    return __fdiv_rn(left, right);
}

__device__ __forceinline__ float clamp01(const float value) {
    return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
}

__device__ __forceinline__ float max_zero(const float value) {
    return value < 0.0f ? 0.0f : value;
}

__device__ __forceinline__ float clamp_nonnegative(
    const float value,
    const float upper
) {
    const float nonnegative = max_zero(value);
    return nonnegative > upper ? upper : nonnegative;
}

__device__ __forceinline__ void insertion_sort_f32_ascending_exact(
    float* values,
    const int64_t count
) {
    for (int64_t index = 1; index < count; ++index) {
        const float value = values[index];
        int64_t insert = index;
        while (insert > 0 && values[insert - 1] > value) {
            values[insert] = values[insert - 1];
            --insert;
        }
        values[insert] = value;
    }
}

// Sorting changes no float values, so percentile arithmetic retains its exact
// accepted topology. The caller's finite-state contract applies here.
// Power-of-two inputs use a bounded bitonic network; other shapes retain the
// established insertion-sort behavior.
__device__ __forceinline__ void sort_f32_ascending_exact(
    float* values,
    const int64_t count
) {
    const bool bitonic_eligible = count > 0 && (count & (count - 1)) == 0;
    if (!bitonic_eligible) {
        insertion_sort_f32_ascending_exact(values, count);
        return;
    }
    for (int64_t span = 2; span <= count; span <<= 1) {
        for (int64_t stride = span >> 1; stride > 0; stride >>= 1) {
            for (int64_t left = 0; left < count; ++left) {
                const int64_t right = left ^ stride;
                if (right <= left) {
                    continue;
                }
                const bool ascending = (left & span) == 0;
                const float left_value = values[left];
                const float right_value = values[right];
                if (
                    (ascending && left_value > right_value)
                    || (!ascending && left_value < right_value)
                ) {
                    values[left] = right_value;
                    values[right] = left_value;
                }
            }
        }
    }
}

// This reproduces NumPy's accepted float32 pairwise reduction topology.
__device__ __forceinline__ float numpy_pairwise_sum_f32_base(
    const float* values,
    const int64_t count
) {
    float accumulators[8] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };
    int64_t index = 0;
    for (; index + 8 <= count; index += 8) {
        #pragma unroll
        for (int64_t lane = 0; lane < 8; ++lane) {
            accumulators[lane] = f32_add(
                accumulators[lane],
                values[index + lane]
            );
        }
    }
    float result = f32_add(
        f32_add(
            f32_add(accumulators[0], accumulators[1]),
            f32_add(accumulators[2], accumulators[3])
        ),
        f32_add(
            f32_add(accumulators[4], accumulators[5]),
            f32_add(accumulators[6], accumulators[7])
        )
    );
    for (; index < count; ++index) {
        result = f32_add(result, values[index]);
    }
    return result;
}

__device__ __forceinline__ float numpy_pairwise_sum_f32(
    const float* values,
    const int64_t count
) {
    constexpr int64_t block_size = 128;
    if (count <= block_size) {
        return numpy_pairwise_sum_f32_base(values, count);
    }
    int64_t left_count = count / 2;
    left_count -= left_count % 8;
    return f32_add(
        numpy_pairwise_sum_f32_base(values, left_count),
        numpy_pairwise_sum_f32_base(values + left_count, count - left_count)
    );
}

__device__ inline double percentile_sorted_f32(
    const float* sorted_values,
    const int64_t count,
    const float q
) {
    if (count <= 0) {
        return 0.0;
    }
    const float position = f32_mul(q, static_cast<float>(count - 1));
    const int64_t lower = static_cast<int64_t>(floorf(position));
    const int64_t upper = static_cast<int64_t>(ceilf(position));
    if (lower == upper) {
        return static_cast<double>(sorted_values[lower]);
    }
    const float fraction = f32_sub(position, static_cast<float>(lower));
    const float difference = f32_sub(
        sorted_values[upper],
        sorted_values[lower]
    );
    const float result = fraction < 0.5f
        ? f32_add(
            sorted_values[lower],
            f32_mul(difference, fraction)
        )
        : f32_sub(
            sorted_values[upper],
            f32_mul(difference, f32_sub(1.0f, fraction))
        );
    return static_cast<double>(result);
}

}  // namespace detail

__device__ __noinline__ void refresh_patch_density_exact(
    const int32_t* __restrict__ agent_grid,
    float* __restrict__ patch_density,
    const PatchP2Config config
) {
    using detail::f32_add;
    using detail::f32_div;
    using detail::f32_sub;

    const int64_t cell_count = config.height * config.width;
    const int64_t diameter = 2 * config.density_radius + 1;
    int64_t denominator = diameter * diameter;
    if (!config.density_include_self) {
        --denominator;
    }
    for (int64_t cell = 0; cell < cell_count; ++cell) {
        patch_density[cell] = 0.0f;
    }
    for (int64_t y = 0; y < config.height; ++y) {
        for (int64_t x = 0; x < config.width; ++x) {
            const int64_t occupied_index = y * config.width + x;
            if (agent_grid[occupied_index] == -1) {
                continue;
            }
            const int64_t y0 = y > config.density_radius
                ? y - config.density_radius
                : 0;
            const int64_t y1 = y + config.density_radius + 1 < config.height
                ? y + config.density_radius + 1
                : config.height;
            const int64_t x0 = x > config.density_radius
                ? x - config.density_radius
                : 0;
            const int64_t x1 = x + config.density_radius + 1 < config.width
                ? x + config.density_radius + 1
                : config.width;
            for (int64_t yy = y0; yy < y1; ++yy) {
                for (int64_t xx = x0; xx < x1; ++xx) {
                    const int64_t index = yy * config.width + xx;
                    patch_density[index] = f32_add(
                        patch_density[index], 1.0f
                    );
                }
            }
            if (!config.density_include_self) {
                patch_density[occupied_index] = f32_sub(
                    patch_density[occupied_index], 1.0f
                );
            }
        }
    }
    if (config.density_normalize && denominator > 0) {
        const float denominator_f32 = static_cast<float>(denominator);
        for (int64_t cell = 0; cell < cell_count; ++cell) {
            patch_density[cell] = f32_div(
                patch_density[cell], denominator_f32
            );
        }
    }
}

// Caller contract: exactly one device thread invokes this helper for one cell.
// The serial row-major loops preserve the immutable Stage2 Patch P2 operation
// order and deliberately avoid reassociation or parallel reductions.
__device__ inline void patch_p2_single_tick_exact(
    const PatchP2State state,
    const PatchP2Scratch scratch,
    const PatchP2Outputs outputs,
    const PatchP2Config config,
    const PatchP2RecoveryContext recovery = {}
) {
    using detail::clamp01;
    using detail::clamp_nonnegative;
    using detail::f32_add;
    using detail::f32_div;
    using detail::f32_mul;
    using detail::f32_sub;
    using detail::max_zero;
    using detail::numpy_pairwise_sum_f32;
    using detail::percentile_sorted_f32;
    using detail::sort_f32_ascending_exact;

    const int64_t cell_count = config.height * config.width;
    float density_denom = static_cast<float>(
        (2 * config.density_radius + 1) *
        (2 * config.density_radius + 1)
    );
    if (!config.density_include_self) {
        density_denom = f32_sub(density_denom, 1.0f);
    }
    if (density_denom <= 0.0f) {
        density_denom = 1.0f;
    }

    double scent_emitted_total = 0.0;
    double scent_decayed_total = 0.0;
    double waste_emitted_total = 0.0;
    double waste_decayed_total = 0.0;
    for (int64_t cell = 0; cell < cell_count; ++cell) {
        const float scent_before = state.patch_scent[cell];
        const float waste_before = state.patch_waste[cell];
        const float scent_after_decay = max_zero(
            f32_mul(scent_before, config.scent_decay_factor)
        );
        const float waste_after_decay = max_zero(
            f32_mul(waste_before, config.waste_decay_factor)
        );
        scent_decayed_total += static_cast<double>(
            f32_sub(scent_before, scent_after_decay)
        );
        waste_decayed_total += static_cast<double>(
            f32_sub(waste_before, waste_after_decay)
        );
        scent_emitted_total += static_cast<double>(
            state.patch_scent_emit_pending[cell]
        );
        waste_emitted_total += static_cast<double>(
            state.patch_waste_emit_pending[cell]
        );
        state.patch_scent[cell] = f32_add(
            scent_after_decay,
            state.patch_scent_emit_pending[cell]
        );
        state.patch_waste[cell] = f32_add(
            waste_after_decay,
            state.patch_waste_emit_pending[cell]
        );
        state.patch_scent_emit_pending[cell] = 0.0f;
        state.patch_waste_emit_pending[cell] = 0.0f;
        state.patch_density[cell] = 0.0f;
    }

    for (int64_t y = 0; y < config.height; ++y) {
        for (int64_t x = 0; x < config.width; ++x) {
            const int64_t occupied_index = y * config.width + x;
            if (state.agent_grid[occupied_index] == -1) {
                continue;
            }
            const int64_t y0 = y > config.density_radius
                ? y - config.density_radius
                : 0;
            const int64_t y1 = y + config.density_radius + 1 < config.height
                ? y + config.density_radius + 1
                : config.height;
            const int64_t x0 = x > config.density_radius
                ? x - config.density_radius
                : 0;
            const int64_t x1 = x + config.density_radius + 1 < config.width
                ? x + config.density_radius + 1
                : config.width;
            for (int64_t yy = y0; yy < y1; ++yy) {
                for (int64_t xx = x0; xx < x1; ++xx) {
                    const int64_t index = yy * config.width + xx;
                    state.patch_density[index] = f32_add(
                        state.patch_density[index],
                        1.0f
                    );
                }
            }
            if (!config.density_include_self) {
                state.patch_density[occupied_index] = f32_sub(
                    state.patch_density[occupied_index],
                    1.0f
                );
            }
        }
    }

    int64_t occupied_count = 0;
    int64_t density_hotspot_count = 0;
    for (int64_t cell = 0; cell < cell_count; ++cell) {
        const float raw_density = state.patch_density[cell];
        if (config.density_normalize) {
            state.patch_density[cell] = clamp01(
                f32_div(raw_density, density_denom)
            );
        }
        if (state.agent_grid[cell] != -1) {
            occupied_count += 1;
            state.patch_scent[cell] = f32_add(
                state.patch_scent[cell],
                static_cast<float>(config.scent_presence_value)
            );
            state.patch_waste[cell] = f32_add(
                state.patch_waste[cell],
                static_cast<float>(config.waste_presence_value)
            );
        }
        if (config.waste_density_emit_scale > 0.0f) {
            const float density_value = config.density_normalize
                ? clamp01(f32_div(raw_density, density_denom))
                : raw_density;
            const float density_emit = f32_mul(
                density_value,
                config.waste_density_emit_scale
            );
            state.patch_waste[cell] = f32_add(
                state.patch_waste[cell],
                density_emit
            );
            waste_emitted_total += static_cast<double>(density_emit);
        }
        state.patch_scent[cell] = clamp_nonnegative(
            state.patch_scent[cell],
            config.scent_max
        );
        state.patch_waste[cell] = clamp_nonnegative(
            state.patch_waste[cell],
            config.waste_max
        );
        const float density_value = config.density_normalize
            ? clamp01(f32_div(raw_density, density_denom))
            : raw_density;
        density_hotspot_count += density_value > 1.0f ? 1 : 0;
    }
    scent_emitted_total += config.scent_presence_value *
        static_cast<double>(occupied_count);
    waste_emitted_total += config.waste_presence_value *
        static_cast<double>(occupied_count);

    double recovery_positive_total = 0.0;
    double recovery_negative_total = 0.0;
    double recovery_decay_total = 0.0;
    double recovery_boost_total = 0.0;
    float recovery_boost_max_seen = 0.0f;
    if (recovery.enabled) {
        const float fertility_span = f32_sub(
            recovery.fertility_max, recovery.fertility_min
        );
        for (int64_t cell = 0; cell < cell_count; ++cell) {
            const float fertility_norm = fertility_span > 0.0f
                ? clamp01(f32_div(
                      f32_sub(
                          state.patch_fertility[cell], recovery.fertility_min
                      ),
                      fertility_span
                  ))
                : 0.0f;
            const float waste_norm = config.waste_max > 0.0f
                ? clamp01(f32_div(state.patch_waste[cell], config.waste_max))
                : 0.0f;
            const float scent_norm = config.scent_max > 0.0f
                ? clamp01(f32_div(state.patch_scent[cell], config.scent_max))
                : 0.0f;
            const float density_norm = clamp01(f32_div(
                state.patch_density[cell], density_denom
            ));
            const float positive = f32_add(
                f32_add(
                    recovery.base_growth,
                    f32_mul(clamp01(recovery.cover[cell]), recovery.cover_weight)
                ),
                f32_mul(fertility_norm, recovery.fertility_weight)
            );
            const float negative = f32_add(
                f32_add(
                    f32_mul(waste_norm, recovery.waste_penalty_weight),
                    f32_mul(density_norm, recovery.density_penalty_weight)
                ),
                f32_mul(scent_norm, recovery.scent_penalty_weight)
            );
            const float before = recovery.activity[cell];
            const float pre_decay = f32_sub(f32_add(before, positive), negative);
            const float post_decay = f32_mul(pre_decay, recovery.decay_factor);
            recovery.activity[cell] = max(
                recovery.activity_min,
                min(recovery.activity_max, post_decay)
            );
            recovery_positive_total += static_cast<double>(positive);
            recovery_negative_total += static_cast<double>(negative);
            recovery_decay_total += static_cast<double>(
                f32_sub(pre_decay, post_decay)
            );
            float boost = 0.0f;
            if (
                recovery.regrowth_boost_enabled
                && recovery.activity_max > 0.0f
            ) {
                boost = min(
                    recovery.regrowth_boost_max,
                    f32_mul(
                        clamp01(f32_div(
                            recovery.activity[cell], recovery.activity_max
                        )),
                        recovery.regrowth_boost_scale
                    )
                );
            }
            recovery.boost_scratch[cell] = boost;
            recovery.sorted_boost_scratch[cell] = boost;
            recovery_boost_total += static_cast<double>(boost);
            recovery_boost_max_seen = max(recovery_boost_max_seen, boost);
        }
        for (int64_t index = 1; index < cell_count; ++index) {
            const float value = recovery.sorted_boost_scratch[index];
            int64_t insert = index;
            while (
                insert > 0
                && recovery.sorted_boost_scratch[insert - 1] > value
            ) {
                recovery.sorted_boost_scratch[insert] =
                    recovery.sorted_boost_scratch[insert - 1];
                --insert;
            }
            recovery.sorted_boost_scratch[insert] = value;
        }
        recovery.stats[kRecoveryPositiveTotal] = recovery_positive_total;
        recovery.stats[kRecoveryNegativeTotal] = recovery_negative_total;
        recovery.stats[kRecoveryDecayTotal] = max(0.0, recovery_decay_total);
        recovery.stats[kRecoveryBoostTotal] = recovery_boost_total;
        recovery.stats[kRecoveryBoostMean] = cell_count > 0
            ? recovery_boost_total / static_cast<double>(cell_count)
            : 0.0;
        recovery.stats[kRecoveryBoostMax] =
            static_cast<double>(recovery_boost_max_seen);
        recovery.stats[kRecoveryBoostP90] = percentile_sorted_f32(
            recovery.sorted_boost_scratch, cell_count, 0.90f
        );
    }

    float multiplier_min = 1.0f;
    float waste_penalty_max_seen = 0.0f;
    float density_penalty_max_seen = 0.0f;
    int64_t high_waste_count = 0;
    int64_t high_density_count = 0;
    for (int64_t cell = 0; cell < cell_count; ++cell) {
        scratch.multiplier[cell] = 1.0f;
        scratch.waste_penalty[cell] = 0.0f;
        scratch.density_penalty[cell] = 0.0f;
        if (config.waste_consequence_enable) {
            const float waste_norm = config.waste_max > 0.0f
                ? clamp01(f32_div(state.patch_waste[cell], config.waste_max))
                : 0.0f;
            const float raw_density = config.density_normalize
                ? f32_mul(state.patch_density[cell], density_denom)
                : state.patch_density[cell];
            const float density_norm = clamp01(
                f32_div(raw_density, density_denom)
            );
            if (config.waste_penalty_enable) {
                scratch.waste_penalty[cell] = min(
                    config.waste_penalty_max,
                    f32_mul(waste_norm, config.waste_penalty_scale)
                );
            }
            if (config.density_penalty_enable) {
                scratch.density_penalty[cell] = min(
                    config.density_penalty_max,
                    f32_mul(density_norm, config.density_penalty_scale)
                );
            }
            const float total_penalty = clamp01(f32_add(
                scratch.waste_penalty[cell],
                scratch.density_penalty[cell]
            ));
            scratch.multiplier[cell] = max(
                config.regrowth_min_multiplier,
                min(1.0f, f32_sub(1.0f, total_penalty))
            );
        }
        scratch.sorted_multiplier[cell] = scratch.multiplier[cell];
        multiplier_min = min(multiplier_min, scratch.multiplier[cell]);
        waste_penalty_max_seen = max(
            waste_penalty_max_seen,
            scratch.waste_penalty[cell]
        );
        density_penalty_max_seen = max(
            density_penalty_max_seen,
            scratch.density_penalty[cell]
        );
        high_waste_count += scratch.waste_penalty[cell] > 0.0f ? 1 : 0;
        high_density_count += scratch.density_penalty[cell] > 0.0f ? 1 : 0;
    }
    sort_f32_ascending_exact(scratch.sorted_multiplier, cell_count);

    double regrowth_base_total = 0.0;
    double regrowth_effective_total = 0.0;
    double regrowth_after_waste_total = 0.0;
    double recovery_added_regrowth_total = 0.0;
    int64_t depleted_count = 0;
    int64_t full_count = 0;
    for (int64_t cell = 0; cell < cell_count; ++cell) {
        const float fertility_nonnegative = max_zero(
            state.patch_fertility[cell]
        );
        const float cap = config.food_regrow_cap_by_fertility
            ? f32_mul(config.food_max, fertility_nonnegative)
            : config.food_max;
        const float base_delta_raw = config.food_regrow_use_fertility
            ? f32_mul(config.food_regrow_per_step, fertility_nonnegative)
            : config.food_regrow_per_step;
        const float before = state.patch_food_amount[cell];
        const float base_amount = clamp_nonnegative(
            f32_add(before, base_delta_raw),
            cap
        );
        const float effective_delta_raw = f32_mul(
            base_delta_raw,
            scratch.multiplier[cell]
        );
        const float recovered_delta_raw = recovery.enabled
            ? f32_mul(
                  effective_delta_raw,
                  f32_add(1.0f, recovery.boost_scratch[cell])
              )
            : effective_delta_raw;
        const float after_waste_amount = clamp_nonnegative(
            f32_add(before, effective_delta_raw),
            cap
        );
        const float effective_amount = clamp_nonnegative(
            f32_add(before, recovered_delta_raw),
            cap
        );
        const float base_delta = f32_sub(base_amount, before);
        const float after_waste_delta = f32_sub(after_waste_amount, before);
        const float effective_delta = f32_sub(effective_amount, before);
        state.patch_food_amount[cell] = effective_amount;
        state.food_grid[cell] = effective_amount > config.food_presence_threshold
            ? kFood
            : kEmpty;
        regrowth_base_total += static_cast<double>(base_delta);
        regrowth_effective_total += static_cast<double>(effective_delta);
        regrowth_after_waste_total += static_cast<double>(after_waste_delta);
        recovery_added_regrowth_total += static_cast<double>(max_zero(
            f32_sub(effective_delta, after_waste_delta)
        ));
        depleted_count += effective_amount <= config.food_presence_threshold
            ? 1
            : 0;
        full_count += effective_amount >= f32_sub(cap, 1.0e-6f) ? 1 : 0;
    }

    const double count = static_cast<double>(cell_count);
    const float count_f32 = static_cast<float>(cell_count);
    outputs.stats[kScentEmittedTotal] = scent_emitted_total;
    outputs.stats[kScentDecayedTotal] = max(0.0, scent_decayed_total);
    outputs.stats[kWasteEmittedTotal] = waste_emitted_total;
    outputs.stats[kWasteDecayedTotal] = max(0.0, waste_decayed_total);
    outputs.stats[kFoodRegrowthTotal] = regrowth_effective_total;
    outputs.stats[kRegrowthBaseTotal] = regrowth_base_total;
    outputs.stats[kRegrowthEffectiveTotal] = regrowth_effective_total;
    outputs.stats[kRegrowthSuppressedTotal] = max(
        0.0,
        regrowth_base_total - regrowth_effective_total
    );
    outputs.stats[kRegrowthMultiplierMean] = static_cast<double>(f32_div(
        numpy_pairwise_sum_f32(scratch.multiplier, cell_count),
        count_f32
    ));
    outputs.stats[kRegrowthMultiplierMin] = static_cast<double>(multiplier_min);
    outputs.stats[kRegrowthMultiplierP10] = percentile_sorted_f32(
        scratch.sorted_multiplier,
        cell_count,
        0.10f
    );
    outputs.stats[kRegrowthMultiplierP90] = percentile_sorted_f32(
        scratch.sorted_multiplier,
        cell_count,
        0.90f
    );
    outputs.stats[kWasteRegrowthPenaltyMean] = static_cast<double>(f32_div(
        numpy_pairwise_sum_f32(scratch.waste_penalty, cell_count),
        count_f32
    ));
    outputs.stats[kWasteRegrowthPenaltyMax] = static_cast<double>(
        waste_penalty_max_seen
    );
    outputs.stats[kDensityRegrowthPenaltyMean] = static_cast<double>(f32_div(
        numpy_pairwise_sum_f32(scratch.density_penalty, cell_count),
        count_f32
    ));
    outputs.stats[kDensityRegrowthPenaltyMax] = static_cast<double>(
        density_penalty_max_seen
    );
    outputs.stats[kHighWasteSuppressionFrac] =
        static_cast<double>(high_waste_count) / count;
    outputs.stats[kHighDensitySuppressionFrac] =
        static_cast<double>(high_density_count) / count;
    outputs.stats[kRegrowthAfterWasteTotal] = regrowth_after_waste_total;
    outputs.stats[kRegrowthAfterRecoveryTotal] = regrowth_effective_total;
    outputs.stats[kRecoveryAddedRegrowthTotal] =
        recovery_added_regrowth_total;
    outputs.stats[kFoodDepletedFrac] =
        static_cast<double>(depleted_count) / count;
    outputs.stats[kFoodFullFrac] = static_cast<double>(full_count) / count;
    outputs.stats[kDensityHotspotFrac] =
        static_cast<double>(density_hotspot_count) / count;
    outputs.stats[kRegrowthConsequenceSteps] = 1.0;
}

// Caller contract: the accepted 16x16 Patch P2 shape launches exactly one
// 128-thread block for one simulation cell. Each lane owns two row-major patch
// cells. Scratch planes are reused only across the explicit block barriers;
// row-order reductions and accepted scalar arithmetic remain on lane zero.
__device__ __noinline__ void patch_p2_single_tick_exact_block_256(
    const PatchP2State state,
    const PatchP2Scratch scratch,
    const PatchP2Outputs outputs,
    const PatchP2Config config
) {
    using detail::clamp01;
    using detail::clamp_nonnegative;
    using detail::f32_add;
    using detail::f32_div;
    using detail::f32_mul;
    using detail::f32_sub;
    using detail::max_zero;
    using detail::numpy_pairwise_sum_f32;
    using detail::percentile_sorted_f32;

    constexpr int64_t cell_count = 256;
    constexpr int64_t cells_per_lane = 2;
    const int64_t lane = static_cast<int64_t>(threadIdx.x);
    float density_denom = static_cast<float>(
        (2 * config.density_radius + 1) *
        (2 * config.density_radius + 1)
    );
    if (!config.density_include_self) {
        density_denom = f32_sub(density_denom, 1.0f);
    }
    if (density_denom <= 0.0f) {
        density_denom = 1.0f;
    }

    double scent_emitted_total = 0.0;
    double scent_decayed_total = 0.0;
    double waste_emitted_total = 0.0;
    double waste_decayed_total = 0.0;
    double regrowth_base_total = 0.0;
    double regrowth_effective_total = 0.0;
    double regrowth_multiplier_mean = 0.0;
    double regrowth_multiplier_p10 = 0.0;
    double regrowth_multiplier_p90 = 0.0;
    double waste_penalty_mean = 0.0;
    double density_penalty_mean = 0.0;
    float multiplier_min = 1.0f;
    float waste_penalty_max_seen = 0.0f;
    float density_penalty_max_seen = 0.0f;
    int64_t occupied_count = 0;
    int64_t density_hotspot_count = 0;
    int64_t high_waste_count = 0;
    int64_t high_density_count = 0;
    int64_t depleted_count = 0;
    int64_t full_count = 0;
    __shared__ float regrowth_base_delta_rows[cell_count];
    __shared__ float regrowth_effective_delta_rows[cell_count];
    __shared__ int32_t regrowth_depleted_rows[cell_count];
    __shared__ int32_t regrowth_full_rows[cell_count];

    // Phase 1: update decay/emission state in parallel. The four scratch
    // planes hold the exact row-order reduction inputs for lane zero.
    for (int64_t owned = 0; owned < cells_per_lane; ++owned) {
        const int64_t cell = lane + owned * 128;
        const float scent_before = state.patch_scent[cell];
        const float waste_before = state.patch_waste[cell];
        const float scent_after_decay = max_zero(
            f32_mul(scent_before, config.scent_decay_factor)
        );
        const float waste_after_decay = max_zero(
            f32_mul(waste_before, config.waste_decay_factor)
        );
        scratch.multiplier[cell] = f32_sub(
            scent_before,
            scent_after_decay
        );
        scratch.waste_penalty[cell] = f32_sub(
            waste_before,
            waste_after_decay
        );
        scratch.density_penalty[cell] =
            state.patch_scent_emit_pending[cell];
        scratch.sorted_multiplier[cell] =
            state.patch_waste_emit_pending[cell];
        state.patch_scent[cell] = f32_add(
            scent_after_decay,
            state.patch_scent_emit_pending[cell]
        );
        state.patch_waste[cell] = f32_add(
            waste_after_decay,
            state.patch_waste_emit_pending[cell]
        );
        state.patch_scent_emit_pending[cell] = 0.0f;
        state.patch_waste_emit_pending[cell] = 0.0f;
    }
    __syncthreads();
    if (lane == 0) {
        for (int64_t cell = 0; cell < cell_count; ++cell) {
            scent_decayed_total += static_cast<double>(
                scratch.multiplier[cell]
            );
            waste_decayed_total += static_cast<double>(
                scratch.waste_penalty[cell]
            );
            scent_emitted_total += static_cast<double>(
                scratch.density_penalty[cell]
            );
            waste_emitted_total += static_cast<double>(
                scratch.sorted_multiplier[cell]
            );
        }
    }
    __syncthreads();

    // Destination ownership avoids density atomics. Integer-valued float32
    // additions are exact for this bounded grid, including the self removal.
    for (int64_t owned = 0; owned < cells_per_lane; ++owned) {
        const int64_t cell = lane + owned * 128;
        const int64_t y = cell / config.width;
        const int64_t x = cell - y * config.width;
        const int64_t y0 = y > config.density_radius
            ? y - config.density_radius
            : 0;
        const int64_t y1 = y + config.density_radius + 1 < config.height
            ? y + config.density_radius + 1
            : config.height;
        const int64_t x0 = x > config.density_radius
            ? x - config.density_radius
            : 0;
        const int64_t x1 = x + config.density_radius + 1 < config.width
            ? x + config.density_radius + 1
            : config.width;
        float raw_density = 0.0f;
        for (int64_t yy = y0; yy < y1; ++yy) {
            for (int64_t xx = x0; xx < x1; ++xx) {
                const int64_t source = yy * config.width + xx;
                if (state.agent_grid[source] != -1) {
                    raw_density = f32_add(raw_density, 1.0f);
                }
            }
        }
        if (!config.density_include_self && state.agent_grid[cell] != -1) {
            raw_density = f32_sub(raw_density, 1.0f);
        }
        state.patch_density[cell] = raw_density;
    }
    __syncthreads();

    // Phase 2: apply density/presence effects independently. Scratch now holds
    // density emission plus two integer-valued row-order counters.
    for (int64_t owned = 0; owned < cells_per_lane; ++owned) {
        const int64_t cell = lane + owned * 128;
        const float raw_density = state.patch_density[cell];
        if (config.density_normalize) {
            state.patch_density[cell] = clamp01(
                f32_div(raw_density, density_denom)
            );
        }
        const bool occupied = state.agent_grid[cell] != -1;
        if (occupied) {
            state.patch_scent[cell] = f32_add(
                state.patch_scent[cell],
                static_cast<float>(config.scent_presence_value)
            );
            state.patch_waste[cell] = f32_add(
                state.patch_waste[cell],
                static_cast<float>(config.waste_presence_value)
            );
        }
        float density_emit = 0.0f;
        if (config.waste_density_emit_scale > 0.0f) {
            const float density_value = config.density_normalize
                ? clamp01(f32_div(raw_density, density_denom))
                : raw_density;
            density_emit = f32_mul(
                density_value,
                config.waste_density_emit_scale
            );
            state.patch_waste[cell] = f32_add(
                state.patch_waste[cell],
                density_emit
            );
        }
        state.patch_scent[cell] = clamp_nonnegative(
            state.patch_scent[cell],
            config.scent_max
        );
        state.patch_waste[cell] = clamp_nonnegative(
            state.patch_waste[cell],
            config.waste_max
        );
        const float density_value = config.density_normalize
            ? clamp01(f32_div(raw_density, density_denom))
            : raw_density;
        scratch.multiplier[cell] = density_emit;
        scratch.waste_penalty[cell] = occupied ? 1.0f : 0.0f;
        scratch.density_penalty[cell] = density_value > 1.0f ? 1.0f : 0.0f;
    }
    __syncthreads();
    if (lane == 0) {
        for (int64_t cell = 0; cell < cell_count; ++cell) {
            waste_emitted_total += static_cast<double>(
                scratch.multiplier[cell]
            );
            occupied_count += scratch.waste_penalty[cell] != 0.0f ? 1 : 0;
            density_hotspot_count +=
                scratch.density_penalty[cell] != 0.0f ? 1 : 0;
        }
        scent_emitted_total += config.scent_presence_value *
            static_cast<double>(occupied_count);
        waste_emitted_total += config.waste_presence_value *
            static_cast<double>(occupied_count);
    }
    __syncthreads();

    // Phase 3: materialize the four persistent penalty/sort planes.
    for (int64_t owned = 0; owned < cells_per_lane; ++owned) {
        const int64_t cell = lane + owned * 128;
        float multiplier = 1.0f;
        float waste_penalty = 0.0f;
        float density_penalty = 0.0f;
        if (config.waste_consequence_enable) {
            const float waste_norm = config.waste_max > 0.0f
                ? clamp01(f32_div(state.patch_waste[cell], config.waste_max))
                : 0.0f;
            const float raw_density = config.density_normalize
                ? f32_mul(state.patch_density[cell], density_denom)
                : state.patch_density[cell];
            const float density_norm = clamp01(
                f32_div(raw_density, density_denom)
            );
            if (config.waste_penalty_enable) {
                waste_penalty = min(
                    config.waste_penalty_max,
                    f32_mul(waste_norm, config.waste_penalty_scale)
                );
            }
            if (config.density_penalty_enable) {
                density_penalty = min(
                    config.density_penalty_max,
                    f32_mul(density_norm, config.density_penalty_scale)
                );
            }
            const float total_penalty = clamp01(f32_add(
                waste_penalty,
                density_penalty
            ));
            multiplier = max(
                config.regrowth_min_multiplier,
                min(1.0f, f32_sub(1.0f, total_penalty))
            );
        }
        scratch.multiplier[cell] = multiplier;
        scratch.waste_penalty[cell] = waste_penalty;
        scratch.density_penalty[cell] = density_penalty;
        scratch.sorted_multiplier[cell] = multiplier;
    }
    __syncthreads();
    if (lane == 0) {
        for (int64_t cell = 0; cell < cell_count; ++cell) {
            multiplier_min = min(multiplier_min, scratch.multiplier[cell]);
            waste_penalty_max_seen = max(
                waste_penalty_max_seen,
                scratch.waste_penalty[cell]
            );
            density_penalty_max_seen = max(
                density_penalty_max_seen,
                scratch.density_penalty[cell]
            );
            high_waste_count += scratch.waste_penalty[cell] > 0.0f ? 1 : 0;
            high_density_count +=
                scratch.density_penalty[cell] > 0.0f ? 1 : 0;
        }
        const float count_f32 = static_cast<float>(cell_count);
        regrowth_multiplier_mean = static_cast<double>(f32_div(
            numpy_pairwise_sum_f32(scratch.multiplier, cell_count),
            count_f32
        ));
        waste_penalty_mean = static_cast<double>(f32_div(
            numpy_pairwise_sum_f32(scratch.waste_penalty, cell_count),
            count_f32
        ));
        density_penalty_mean = static_cast<double>(f32_div(
            numpy_pairwise_sum_f32(scratch.density_penalty, cell_count),
            count_f32
        ));
    }
    __syncthreads();

    // One compare/exchange per lane and a block barrier per bitonic stage.
    for (int64_t span = 2; span <= cell_count; span <<= 1) {
        for (int64_t stride = span >> 1; stride > 0; stride >>= 1) {
            const int64_t group = lane / stride;
            const int64_t offset = lane - group * stride;
            const int64_t left = group * (stride << 1) + offset;
            const int64_t right = left + stride;
            const bool ascending = (left & span) == 0;
            const float left_value = scratch.sorted_multiplier[left];
            const float right_value = scratch.sorted_multiplier[right];
            if (
                (ascending && left_value > right_value)
                || (!ascending && left_value < right_value)
            ) {
                scratch.sorted_multiplier[left] = right_value;
                scratch.sorted_multiplier[right] = left_value;
            }
            __syncthreads();
        }
    }
    if (lane == 0) {
        regrowth_multiplier_p10 = percentile_sorted_f32(
            scratch.sorted_multiplier,
            cell_count,
            0.10f
        );
        regrowth_multiplier_p90 = percentile_sorted_f32(
            scratch.sorted_multiplier,
            cell_count,
            0.90f
        );
    }
    __syncthreads();

    // Phase 4: preserve all canonical scratch planes while lanes independently
    // materialize exact per-cell regrowth results in block-shared row storage.
    // Lane zero then retains the original row-major double additions/counts.
    for (int64_t owned = 0; owned < cells_per_lane; ++owned) {
        const int64_t cell = lane + owned * 128;
        const float fertility_nonnegative = max_zero(
            state.patch_fertility[cell]
        );
        const float cap = config.food_regrow_cap_by_fertility
            ? f32_mul(config.food_max, fertility_nonnegative)
            : config.food_max;
        const float base_delta_raw = config.food_regrow_use_fertility
            ? f32_mul(config.food_regrow_per_step, fertility_nonnegative)
            : config.food_regrow_per_step;
        const float before = state.patch_food_amount[cell];
        const float base_amount = clamp_nonnegative(
            f32_add(before, base_delta_raw),
            cap
        );
        const float effective_delta_raw = f32_mul(
            base_delta_raw,
            scratch.multiplier[cell]
        );
        const float effective_amount = clamp_nonnegative(
            f32_add(before, effective_delta_raw),
            cap
        );
        const float base_delta = f32_sub(base_amount, before);
        const float effective_delta = f32_sub(effective_amount, before);
        state.patch_food_amount[cell] = effective_amount;
        state.food_grid[cell] = effective_amount > config.food_presence_threshold
            ? kFood
            : kEmpty;
        regrowth_base_delta_rows[cell] = base_delta;
        regrowth_effective_delta_rows[cell] = effective_delta;
        regrowth_depleted_rows[cell] =
            effective_amount <= config.food_presence_threshold ? 1 : 0;
        regrowth_full_rows[cell] =
            effective_amount >= f32_sub(cap, 1.0e-6f) ? 1 : 0;
    }
    __syncthreads();
    if (lane == 0) {
        for (int64_t cell = 0; cell < cell_count; ++cell) {
            regrowth_base_total += static_cast<double>(
                regrowth_base_delta_rows[cell]
            );
            regrowth_effective_total += static_cast<double>(
                regrowth_effective_delta_rows[cell]
            );
            depleted_count += regrowth_depleted_rows[cell];
            full_count += regrowth_full_rows[cell];
        }

        const double count = static_cast<double>(cell_count);
        outputs.stats[kScentEmittedTotal] = scent_emitted_total;
        outputs.stats[kScentDecayedTotal] = max(0.0, scent_decayed_total);
        outputs.stats[kWasteEmittedTotal] = waste_emitted_total;
        outputs.stats[kWasteDecayedTotal] = max(0.0, waste_decayed_total);
        outputs.stats[kFoodRegrowthTotal] = regrowth_effective_total;
        outputs.stats[kRegrowthBaseTotal] = regrowth_base_total;
        outputs.stats[kRegrowthEffectiveTotal] = regrowth_effective_total;
        outputs.stats[kRegrowthSuppressedTotal] = max(
            0.0,
            regrowth_base_total - regrowth_effective_total
        );
        outputs.stats[kRegrowthMultiplierMean] = regrowth_multiplier_mean;
        outputs.stats[kRegrowthMultiplierMin] =
            static_cast<double>(multiplier_min);
        outputs.stats[kRegrowthMultiplierP10] = regrowth_multiplier_p10;
        outputs.stats[kRegrowthMultiplierP90] = regrowth_multiplier_p90;
        outputs.stats[kWasteRegrowthPenaltyMean] = waste_penalty_mean;
        outputs.stats[kWasteRegrowthPenaltyMax] = static_cast<double>(
            waste_penalty_max_seen
        );
        outputs.stats[kDensityRegrowthPenaltyMean] = density_penalty_mean;
        outputs.stats[kDensityRegrowthPenaltyMax] = static_cast<double>(
            density_penalty_max_seen
        );
        outputs.stats[kHighWasteSuppressionFrac] =
            static_cast<double>(high_waste_count) / count;
        outputs.stats[kHighDensitySuppressionFrac] =
            static_cast<double>(high_density_count) / count;
        outputs.stats[kRegrowthAfterWasteTotal] = regrowth_effective_total;
        outputs.stats[kRegrowthAfterRecoveryTotal] = regrowth_effective_total;
        outputs.stats[kRecoveryAddedRegrowthTotal] = 0.0;
        outputs.stats[kFoodDepletedFrac] =
            static_cast<double>(depleted_count) / count;
        outputs.stats[kFoodFullFrac] = static_cast<double>(full_count) / count;
        outputs.stats[kDensityHotspotFrac] =
            static_cast<double>(density_hotspot_count) / count;
        outputs.stats[kRegrowthConsequenceSteps] = 1.0;
    }
    __syncthreads();
}

#endif  // defined(__CUDACC__)

}  // namespace eiel::cuda_architecture_b::persistent_patch_p2

#endif  // EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_PATCH_P2_CUH_
