#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int32_t EMPTY = 0;
constexpr int32_t FOOD = 1;
constexpr int64_t STAT_COUNT = 10;

enum PatchContextStat : int64_t {
    SCENT_EMITTED_TOTAL = 0,
    SCENT_DECAYED_TOTAL = 1,
    WASTE_EMITTED_TOTAL = 2,
    WASTE_DECAYED_TOTAL = 3,
    FOOD_REGROWTH_TOTAL = 4,
    REGROWTH_BASE_TOTAL = 5,
    REGROWTH_EFFECTIVE_TOTAL = 6,
    REGROWTH_SUPPRESSED_TOTAL = 7,
    FOOD_DEPLETED_FRAC = 8,
    FOOD_FULL_FRAC = 9,
};

constexpr int64_t P2_STAT_COUNT = 25;

enum PatchContextP2Stat : int64_t {
    P2_SCENT_EMITTED_TOTAL = 0,
    P2_SCENT_DECAYED_TOTAL = 1,
    P2_WASTE_EMITTED_TOTAL = 2,
    P2_WASTE_DECAYED_TOTAL = 3,
    P2_FOOD_REGROWTH_TOTAL = 4,
    P2_REGROWTH_BASE_TOTAL = 5,
    P2_REGROWTH_EFFECTIVE_TOTAL = 6,
    P2_REGROWTH_SUPPRESSED_TOTAL = 7,
    P2_REGROWTH_MULTIPLIER_MEAN = 8,
    P2_REGROWTH_MULTIPLIER_MIN = 9,
    P2_REGROWTH_MULTIPLIER_P10 = 10,
    P2_REGROWTH_MULTIPLIER_P90 = 11,
    P2_WASTE_REGROWTH_PENALTY_MEAN = 12,
    P2_WASTE_REGROWTH_PENALTY_MAX = 13,
    P2_DENSITY_REGROWTH_PENALTY_MEAN = 14,
    P2_DENSITY_REGROWTH_PENALTY_MAX = 15,
    P2_HIGH_WASTE_SUPPRESSION_FRAC = 16,
    P2_HIGH_DENSITY_SUPPRESSION_FRAC = 17,
    P2_REGROWTH_AFTER_WASTE_TOTAL = 18,
    P2_REGROWTH_AFTER_RECOVERY_TOTAL = 19,
    P2_RECOVERY_ADDED_REGROWTH_TOTAL = 20,
    P2_FOOD_DEPLETED_FRAC = 21,
    P2_FOOD_FULL_FRAC = 22,
    P2_DENSITY_HOTSPOT_FRAC = 23,
    P2_REGROWTH_CONSEQUENCE_STEPS = 24,
};

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

__device__ __forceinline__ float clamp_nonnegative(const float value, const float upper) {
    const float nonnegative = max_zero(value);
    return nonnegative > upper ? upper : nonnegative;
}

__global__ void patch_context_steps_kernel(
    float* __restrict__ patch_food_amount,
    const float* __restrict__ patch_fertility,
    float* __restrict__ patch_scent,
    float* __restrict__ patch_waste,
    float* __restrict__ patch_scent_emit_pending,
    float* __restrict__ patch_waste_emit_pending,
    int32_t* __restrict__ food_grid,
    double* __restrict__ stats_by_step,
    const int64_t cell_count,
    const int64_t steps,
    const float scent_decay_factor,
    const float waste_decay_factor,
    const float scent_max,
    const float waste_max,
    const float food_max,
    const float food_regrow_per_step,
    const bool food_regrow_use_fertility,
    const bool food_regrow_cap_by_fertility,
    const float food_presence_threshold
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    // This first correctness slice intentionally preserves the CPU owner's
    // row-major scalar operation order. Later parallel kernels must prove the
    // same boundary state before replacing it.
    for (int64_t step = 0; step < steps; ++step) {
        double scent_emitted_total = 0.0;
        double scent_decayed_total = 0.0;
        double waste_emitted_total = 0.0;
        double waste_decayed_total = 0.0;

        for (int64_t cell = 0; cell < cell_count; ++cell) {
            const float scent_before = patch_scent[cell];
            const float waste_before = patch_waste[cell];
            const float scent_after_decay = max_zero(f32_mul(scent_before, scent_decay_factor));
            const float waste_after_decay = max_zero(f32_mul(waste_before, waste_decay_factor));
            const float scent_pending = patch_scent_emit_pending[cell];
            const float waste_pending = patch_waste_emit_pending[cell];

            scent_decayed_total += static_cast<double>(f32_sub(scent_before, scent_after_decay));
            waste_decayed_total += static_cast<double>(f32_sub(waste_before, waste_after_decay));
            scent_emitted_total += static_cast<double>(scent_pending);
            waste_emitted_total += static_cast<double>(waste_pending);

            patch_scent[cell] = f32_add(scent_after_decay, scent_pending);
            patch_waste[cell] = f32_add(waste_after_decay, waste_pending);
            patch_scent_emit_pending[cell] = 0.0f;
            patch_waste_emit_pending[cell] = 0.0f;
        }

        // The CPU owner clamps traces after all emissions for the step.
        for (int64_t cell = 0; cell < cell_count; ++cell) {
            patch_scent[cell] = clamp_nonnegative(patch_scent[cell], scent_max);
            patch_waste[cell] = clamp_nonnegative(patch_waste[cell], waste_max);
        }

        double regrowth_base_total = 0.0;
        double regrowth_effective_total = 0.0;
        int64_t depleted_count = 0;
        int64_t full_count = 0;
        for (int64_t cell = 0; cell < cell_count; ++cell) {
            const float fertility_nonnegative = max_zero(patch_fertility[cell]);
            const float cap = food_regrow_cap_by_fertility
                ? f32_mul(food_max, fertility_nonnegative)
                : food_max;
            const float base_delta_raw = food_regrow_use_fertility
                ? f32_mul(food_regrow_per_step, fertility_nonnegative)
                : food_regrow_per_step;
            const float before = patch_food_amount[cell];
            const float base_amount = clamp_nonnegative(f32_add(before, base_delta_raw), cap);

            // Base Patch has no waste/density consequence multiplier, so the
            // effective delta is exactly the base delta in this bounded slice.
            const float effective_amount = base_amount;
            const float base_delta = f32_sub(base_amount, before);
            const float effective_delta = f32_sub(effective_amount, before);

            patch_food_amount[cell] = effective_amount;
            food_grid[cell] = effective_amount > food_presence_threshold ? FOOD : EMPTY;
            regrowth_base_total += static_cast<double>(base_delta);
            regrowth_effective_total += static_cast<double>(effective_delta);
            depleted_count += effective_amount <= food_presence_threshold ? 1 : 0;
            full_count += effective_amount >= f32_sub(cap, 1.0e-6f) ? 1 : 0;
        }

        double* stats = stats_by_step + step * STAT_COUNT;
        stats[SCENT_EMITTED_TOTAL] = scent_emitted_total;
        stats[SCENT_DECAYED_TOTAL] = scent_decayed_total < 0.0 ? 0.0 : scent_decayed_total;
        stats[WASTE_EMITTED_TOTAL] = waste_emitted_total;
        stats[WASTE_DECAYED_TOTAL] = waste_decayed_total < 0.0 ? 0.0 : waste_decayed_total;
        stats[FOOD_REGROWTH_TOTAL] = regrowth_effective_total;
        stats[REGROWTH_BASE_TOTAL] = regrowth_base_total;
        stats[REGROWTH_EFFECTIVE_TOTAL] = regrowth_effective_total;
        const double suppressed = regrowth_base_total - regrowth_effective_total;
        stats[REGROWTH_SUPPRESSED_TOTAL] = suppressed < 0.0 ? 0.0 : suppressed;
        stats[FOOD_DEPLETED_FRAC] = cell_count > 0
            ? static_cast<double>(depleted_count) / static_cast<double>(cell_count)
            : 0.0;
        stats[FOOD_FULL_FRAC] = cell_count > 0
            ? static_cast<double>(full_count) / static_cast<double>(cell_count)
            : 0.0;
    }
}

__device__ float numpy_pairwise_sum_f32(
    const float* values,
    const int64_t count
) {
    constexpr int64_t block_size = 128;
    if (count <= block_size) {
        float accumulators[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        int64_t index = 0;
        for (; index + 8 <= count; index += 8) {
            #pragma unroll
            for (int64_t lane = 0; lane < 8; ++lane) {
                accumulators[lane] = f32_add(accumulators[lane], values[index + lane]);
            }
        }
        float result = f32_add(
            f32_add(f32_add(accumulators[0], accumulators[1]), f32_add(accumulators[2], accumulators[3])),
            f32_add(f32_add(accumulators[4], accumulators[5]), f32_add(accumulators[6], accumulators[7]))
        );
        for (; index < count; ++index) {
            result = f32_add(result, values[index]);
        }
        return result;
    }
    int64_t left_count = count / 2;
    left_count -= left_count % 8;
    return f32_add(
        numpy_pairwise_sum_f32(values, left_count),
        numpy_pairwise_sum_f32(values + left_count, count - left_count)
    );
}

__device__ double percentile_sorted_f32(
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
    const float difference = f32_sub(sorted_values[upper], sorted_values[lower]);
    const float result = fraction < 0.5f
        ? f32_add(sorted_values[lower], f32_mul(difference, fraction))
        : f32_sub(
            sorted_values[upper],
            f32_mul(difference, f32_sub(1.0f, fraction))
        );
    return static_cast<double>(result);
}

__global__ void patch_context_p2_steps_kernel(
    float* __restrict__ patch_food_amount,
    const float* __restrict__ patch_fertility,
    float* __restrict__ patch_scent,
    float* __restrict__ patch_waste,
    float* __restrict__ patch_scent_emit_pending,
    float* __restrict__ patch_waste_emit_pending,
    const int32_t* __restrict__ agent_grid,
    float* __restrict__ patch_density,
    int32_t* __restrict__ food_grid,
    float* __restrict__ multiplier,
    float* __restrict__ waste_penalty,
    float* __restrict__ density_penalty,
    float* __restrict__ sorted_multiplier,
    double* __restrict__ stats_by_step,
    const int64_t height,
    const int64_t width,
    const int64_t steps,
    const float scent_decay_factor,
    const float waste_decay_factor,
    const double scent_presence_value,
    const double waste_presence_value,
    const float waste_density_emit_scale,
    const float scent_max,
    const float waste_max,
    const float food_max,
    const float food_regrow_per_step,
    const bool food_regrow_use_fertility,
    const bool food_regrow_cap_by_fertility,
    const float food_presence_threshold,
    const int64_t density_radius,
    const bool density_include_self,
    const bool density_normalize,
    const bool waste_consequence_enable,
    const bool waste_penalty_enable,
    const float waste_penalty_scale,
    const float waste_penalty_max,
    const bool density_penalty_enable,
    const float density_penalty_scale,
    const float density_penalty_max,
    const float regrowth_min_multiplier
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const int64_t cell_count = height * width;
    float density_denom = static_cast<float>((2 * density_radius + 1) * (2 * density_radius + 1));
    if (!density_include_self) {
        density_denom = f32_sub(density_denom, 1.0f);
    }
    if (density_denom <= 0.0f) {
        density_denom = 1.0f;
    }

    for (int64_t step = 0; step < steps; ++step) {
        double scent_emitted_total = 0.0;
        double scent_decayed_total = 0.0;
        double waste_emitted_total = 0.0;
        double waste_decayed_total = 0.0;
        for (int64_t cell = 0; cell < cell_count; ++cell) {
            const float scent_before = patch_scent[cell];
            const float waste_before = patch_waste[cell];
            const float scent_after_decay = max_zero(f32_mul(scent_before, scent_decay_factor));
            const float waste_after_decay = max_zero(f32_mul(waste_before, waste_decay_factor));
            scent_decayed_total += static_cast<double>(f32_sub(scent_before, scent_after_decay));
            waste_decayed_total += static_cast<double>(f32_sub(waste_before, waste_after_decay));
            scent_emitted_total += static_cast<double>(patch_scent_emit_pending[cell]);
            waste_emitted_total += static_cast<double>(patch_waste_emit_pending[cell]);
            patch_scent[cell] = f32_add(scent_after_decay, patch_scent_emit_pending[cell]);
            patch_waste[cell] = f32_add(waste_after_decay, patch_waste_emit_pending[cell]);
            patch_scent_emit_pending[cell] = 0.0f;
            patch_waste_emit_pending[cell] = 0.0f;
            patch_density[cell] = 0.0f;
        }

        for (int64_t y = 0; y < height; ++y) {
            for (int64_t x = 0; x < width; ++x) {
                const int64_t occupied_index = y * width + x;
                if (agent_grid[occupied_index] == -1) {
                    continue;
                }
                const int64_t y0 = y > density_radius ? y - density_radius : 0;
                const int64_t y1 = y + density_radius + 1 < height
                    ? y + density_radius + 1
                    : height;
                const int64_t x0 = x > density_radius ? x - density_radius : 0;
                const int64_t x1 = x + density_radius + 1 < width
                    ? x + density_radius + 1
                    : width;
                for (int64_t yy = y0; yy < y1; ++yy) {
                    for (int64_t xx = x0; xx < x1; ++xx) {
                        const int64_t index = yy * width + xx;
                        patch_density[index] = f32_add(patch_density[index], 1.0f);
                    }
                }
                if (!density_include_self) {
                    patch_density[occupied_index] = f32_sub(patch_density[occupied_index], 1.0f);
                }
            }
        }

        int64_t occupied_count = 0;
        int64_t density_hotspot_count = 0;
        for (int64_t cell = 0; cell < cell_count; ++cell) {
            const float raw_density = patch_density[cell];
            if (density_normalize) {
                patch_density[cell] = clamp01(f32_div(raw_density, density_denom));
            }
            if (agent_grid[cell] != -1) {
                occupied_count += 1;
                patch_scent[cell] = f32_add(patch_scent[cell], static_cast<float>(scent_presence_value));
                patch_waste[cell] = f32_add(patch_waste[cell], static_cast<float>(waste_presence_value));
            }
            if (waste_density_emit_scale > 0.0f) {
                const float density_value = density_normalize
                    ? clamp01(f32_div(raw_density, density_denom))
                    : raw_density;
                const float density_emit = f32_mul(density_value, waste_density_emit_scale);
                patch_waste[cell] = f32_add(patch_waste[cell], density_emit);
                waste_emitted_total += static_cast<double>(density_emit);
            }
            patch_scent[cell] = clamp_nonnegative(patch_scent[cell], scent_max);
            patch_waste[cell] = clamp_nonnegative(patch_waste[cell], waste_max);
            const float density_value = density_normalize
                ? clamp01(f32_div(raw_density, density_denom))
                : raw_density;
            density_hotspot_count += density_value > 1.0f ? 1 : 0;
        }
        scent_emitted_total += scent_presence_value * static_cast<double>(occupied_count);
        waste_emitted_total += waste_presence_value * static_cast<double>(occupied_count);

        float multiplier_min = 1.0f;
        float waste_penalty_max_seen = 0.0f;
        float density_penalty_max_seen = 0.0f;
        int64_t high_waste_count = 0;
        int64_t high_density_count = 0;
        for (int64_t cell = 0; cell < cell_count; ++cell) {
            multiplier[cell] = 1.0f;
            waste_penalty[cell] = 0.0f;
            density_penalty[cell] = 0.0f;
            if (waste_consequence_enable) {
                const float waste_norm = waste_max > 0.0f
                    ? clamp01(f32_div(patch_waste[cell], waste_max))
                    : 0.0f;
                const float raw_density = density_normalize
                    ? f32_mul(patch_density[cell], density_denom)
                    : patch_density[cell];
                const float density_norm = clamp01(f32_div(raw_density, density_denom));
                if (waste_penalty_enable) {
                    waste_penalty[cell] = min(waste_penalty_max, f32_mul(waste_norm, waste_penalty_scale));
                }
                if (density_penalty_enable) {
                    density_penalty[cell] = min(
                        density_penalty_max,
                        f32_mul(density_norm, density_penalty_scale)
                    );
                }
                const float total_penalty = clamp01(f32_add(waste_penalty[cell], density_penalty[cell]));
                multiplier[cell] = max(
                    regrowth_min_multiplier,
                    min(1.0f, f32_sub(1.0f, total_penalty))
                );
            }
            sorted_multiplier[cell] = multiplier[cell];
            multiplier_min = min(multiplier_min, multiplier[cell]);
            waste_penalty_max_seen = max(waste_penalty_max_seen, waste_penalty[cell]);
            density_penalty_max_seen = max(density_penalty_max_seen, density_penalty[cell]);
            high_waste_count += waste_penalty[cell] > 0.0f ? 1 : 0;
            high_density_count += density_penalty[cell] > 0.0f ? 1 : 0;
        }
        for (int64_t index = 1; index < cell_count; ++index) {
            const float value = sorted_multiplier[index];
            int64_t insert = index;
            while (insert > 0 && sorted_multiplier[insert - 1] > value) {
                sorted_multiplier[insert] = sorted_multiplier[insert - 1];
                --insert;
            }
            sorted_multiplier[insert] = value;
        }

        double regrowth_base_total = 0.0;
        double regrowth_effective_total = 0.0;
        int64_t depleted_count = 0;
        int64_t full_count = 0;
        for (int64_t cell = 0; cell < cell_count; ++cell) {
            const float fertility_nonnegative = max_zero(patch_fertility[cell]);
            const float cap = food_regrow_cap_by_fertility
                ? f32_mul(food_max, fertility_nonnegative)
                : food_max;
            const float base_delta_raw = food_regrow_use_fertility
                ? f32_mul(food_regrow_per_step, fertility_nonnegative)
                : food_regrow_per_step;
            const float before = patch_food_amount[cell];
            const float base_amount = clamp_nonnegative(f32_add(before, base_delta_raw), cap);
            const float effective_delta_raw = f32_mul(base_delta_raw, multiplier[cell]);
            const float effective_amount = clamp_nonnegative(f32_add(before, effective_delta_raw), cap);
            const float base_delta = f32_sub(base_amount, before);
            const float effective_delta = f32_sub(effective_amount, before);
            patch_food_amount[cell] = effective_amount;
            food_grid[cell] = effective_amount > food_presence_threshold ? FOOD : EMPTY;
            regrowth_base_total += static_cast<double>(base_delta);
            regrowth_effective_total += static_cast<double>(effective_delta);
            depleted_count += effective_amount <= food_presence_threshold ? 1 : 0;
            full_count += effective_amount >= f32_sub(cap, 1.0e-6f) ? 1 : 0;
        }

        double* stats = stats_by_step + step * P2_STAT_COUNT;
        const double count = static_cast<double>(cell_count);
        const float count_f32 = static_cast<float>(cell_count);
        stats[P2_SCENT_EMITTED_TOTAL] = scent_emitted_total;
        stats[P2_SCENT_DECAYED_TOTAL] = max(0.0, scent_decayed_total);
        stats[P2_WASTE_EMITTED_TOTAL] = waste_emitted_total;
        stats[P2_WASTE_DECAYED_TOTAL] = max(0.0, waste_decayed_total);
        stats[P2_FOOD_REGROWTH_TOTAL] = regrowth_effective_total;
        stats[P2_REGROWTH_BASE_TOTAL] = regrowth_base_total;
        stats[P2_REGROWTH_EFFECTIVE_TOTAL] = regrowth_effective_total;
        stats[P2_REGROWTH_SUPPRESSED_TOTAL] = max(0.0, regrowth_base_total - regrowth_effective_total);
        stats[P2_REGROWTH_MULTIPLIER_MEAN] = static_cast<double>(
            f32_div(numpy_pairwise_sum_f32(multiplier, cell_count), count_f32)
        );
        stats[P2_REGROWTH_MULTIPLIER_MIN] = static_cast<double>(multiplier_min);
        stats[P2_REGROWTH_MULTIPLIER_P10] = percentile_sorted_f32(sorted_multiplier, cell_count, 0.10f);
        stats[P2_REGROWTH_MULTIPLIER_P90] = percentile_sorted_f32(sorted_multiplier, cell_count, 0.90f);
        stats[P2_WASTE_REGROWTH_PENALTY_MEAN] = static_cast<double>(
            f32_div(numpy_pairwise_sum_f32(waste_penalty, cell_count), count_f32)
        );
        stats[P2_WASTE_REGROWTH_PENALTY_MAX] = static_cast<double>(waste_penalty_max_seen);
        stats[P2_DENSITY_REGROWTH_PENALTY_MEAN] = static_cast<double>(
            f32_div(numpy_pairwise_sum_f32(density_penalty, cell_count), count_f32)
        );
        stats[P2_DENSITY_REGROWTH_PENALTY_MAX] = static_cast<double>(density_penalty_max_seen);
        stats[P2_HIGH_WASTE_SUPPRESSION_FRAC] = static_cast<double>(high_waste_count) / count;
        stats[P2_HIGH_DENSITY_SUPPRESSION_FRAC] = static_cast<double>(high_density_count) / count;
        stats[P2_REGROWTH_AFTER_WASTE_TOTAL] = regrowth_effective_total;
        stats[P2_REGROWTH_AFTER_RECOVERY_TOTAL] = regrowth_effective_total;
        stats[P2_RECOVERY_ADDED_REGROWTH_TOTAL] = 0.0;
        stats[P2_FOOD_DEPLETED_FRAC] = static_cast<double>(depleted_count) / count;
        stats[P2_FOOD_FULL_FRAC] = static_cast<double>(full_count) / count;
        stats[P2_DENSITY_HOTSPOT_FRAC] = static_cast<double>(density_hotspot_count) / count;
        stats[P2_REGROWTH_CONSEQUENCE_STEPS] = 1.0;
    }
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA-resident");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.dim() == 2, name, " must be a two-dimensional single-cell grid");
}

double checked_finite(const double value, const char* name) {
    TORCH_CHECK(std::isfinite(value), name, " must be finite");
    return value;
}

}  // namespace

torch::Tensor eiel_cuda_architecture_b_patch_context_v1(
    torch::Tensor patch_food_amount,
    torch::Tensor patch_fertility,
    torch::Tensor patch_scent,
    torch::Tensor patch_waste,
    torch::Tensor patch_scent_emit_pending,
    torch::Tensor patch_waste_emit_pending,
    torch::Tensor food_grid,
    const int64_t steps,
    const double patch_scent_decay_per_step,
    const double patch_waste_decay_per_step,
    const double patch_scent_max,
    const double patch_waste_max,
    const double patch_food_amount_max,
    const double patch_food_regrow_per_step,
    const bool patch_food_regrow_use_fertility,
    const bool patch_food_regrow_cap_by_fertility,
    const double patch_food_presence_threshold
) {
    const std::vector<std::pair<const char*, torch::Tensor>> tensors = {
        {"patch_food_amount", patch_food_amount},
        {"patch_fertility", patch_fertility},
        {"patch_scent", patch_scent},
        {"patch_waste", patch_waste},
        {"patch_scent_emit_pending", patch_scent_emit_pending},
        {"patch_waste_emit_pending", patch_waste_emit_pending},
        {"food_grid", food_grid},
    };
    for (const auto& named : tensors) {
        check_cuda_contiguous(named.second, named.first);
        TORCH_CHECK(
            named.second.device() == patch_food_amount.device(),
            named.first,
            " must share the authoritative Patch CUDA device"
        );
        TORCH_CHECK(
            named.second.sizes() == patch_food_amount.sizes(),
            named.first,
            " must align with patch_food_amount"
        );
    }
    TORCH_CHECK(patch_food_amount.numel() > 0, "Patch context grid must not be empty");
    for (const auto& tensor : {
             patch_food_amount,
             patch_fertility,
             patch_scent,
             patch_waste,
             patch_scent_emit_pending,
             patch_waste_emit_pending,
         }) {
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "Patch context float grids must be float32");
    }
    TORCH_CHECK(food_grid.scalar_type() == torch::kInt32, "food_grid must be int32");
    TORCH_CHECK(steps > 0, "steps must be positive");

    for (size_t left = 0; left < tensors.size(); ++left) {
        for (size_t right = left + 1; right < tensors.size(); ++right) {
            TORCH_CHECK(
                tensors[left].second.data_ptr() != tensors[right].second.data_ptr(),
                tensors[left].first,
                " and ",
                tensors[right].first,
                " must not alias"
            );
        }
    }

    const double scent_decay = std::clamp(
        checked_finite(patch_scent_decay_per_step, "patch_scent_decay_per_step"),
        0.0,
        1.0
    );
    const double waste_decay = std::clamp(
        checked_finite(patch_waste_decay_per_step, "patch_waste_decay_per_step"),
        0.0,
        1.0
    );
    const double scent_max_value = std::max(
        0.0,
        checked_finite(patch_scent_max, "patch_scent_max")
    );
    const double waste_max_value = std::max(
        0.0,
        checked_finite(patch_waste_max, "patch_waste_max")
    );
    const double food_max_value = std::max(
        0.0,
        checked_finite(patch_food_amount_max, "patch_food_amount_max")
    );
    const double regrow_value = std::max(
        0.0,
        checked_finite(patch_food_regrow_per_step, "patch_food_regrow_per_step")
    );
    const double threshold_value = std::clamp(
        checked_finite(patch_food_presence_threshold, "patch_food_presence_threshold"),
        0.0,
        food_max_value
    );

    c10::cuda::CUDAGuard device_guard(patch_food_amount.device());
    auto stats_by_step = torch::empty(
        {steps, STAT_COUNT},
        patch_food_amount.options().dtype(torch::kFloat64)
    );
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    patch_context_steps_kernel<<<1, 1, 0, stream>>>(
        patch_food_amount.data_ptr<float>(),
        patch_fertility.data_ptr<float>(),
        patch_scent.data_ptr<float>(),
        patch_waste.data_ptr<float>(),
        patch_scent_emit_pending.data_ptr<float>(),
        patch_waste_emit_pending.data_ptr<float>(),
        food_grid.data_ptr<int32_t>(),
        stats_by_step.data_ptr<double>(),
        patch_food_amount.numel(),
        steps,
        static_cast<float>(1.0 - scent_decay),
        static_cast<float>(1.0 - waste_decay),
        static_cast<float>(scent_max_value),
        static_cast<float>(waste_max_value),
        static_cast<float>(food_max_value),
        static_cast<float>(regrow_value),
        patch_food_regrow_use_fertility,
        patch_food_regrow_cap_by_fertility,
        static_cast<float>(threshold_value)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return stats_by_step;
}

torch::Tensor eiel_cuda_architecture_b_patch_context_p2_v1(
    torch::Tensor patch_food_amount,
    torch::Tensor patch_fertility,
    torch::Tensor patch_scent,
    torch::Tensor patch_waste,
    torch::Tensor patch_scent_emit_pending,
    torch::Tensor patch_waste_emit_pending,
    torch::Tensor agent_grid,
    torch::Tensor patch_density,
    torch::Tensor food_grid,
    const int64_t steps,
    const double patch_scent_decay_per_step,
    const double patch_waste_decay_per_step,
    const double patch_scent_presence_emit,
    const double patch_waste_presence_emit,
    const double patch_waste_density_emit_scale,
    const double patch_scent_max,
    const double patch_waste_max,
    const double patch_food_amount_max,
    const double patch_food_regrow_per_step,
    const bool patch_food_regrow_use_fertility,
    const bool patch_food_regrow_cap_by_fertility,
    const double patch_food_presence_threshold,
    const int64_t patch_density_radius,
    const bool patch_density_include_self,
    const bool patch_density_normalize,
    const bool patch_world_waste_consequence_enable,
    const bool patch_waste_regrowth_penalty_enable,
    const double patch_waste_regrowth_penalty_scale,
    const double patch_waste_regrowth_penalty_max,
    const bool patch_density_regrowth_penalty_enable,
    const double patch_density_regrowth_penalty_scale,
    const double patch_density_regrowth_penalty_max,
    const double patch_regrowth_min_multiplier
) {
    const std::vector<std::pair<const char*, torch::Tensor>> tensors = {
        {"patch_food_amount", patch_food_amount},
        {"patch_fertility", patch_fertility},
        {"patch_scent", patch_scent},
        {"patch_waste", patch_waste},
        {"patch_scent_emit_pending", patch_scent_emit_pending},
        {"patch_waste_emit_pending", patch_waste_emit_pending},
        {"agent_grid", agent_grid},
        {"patch_density", patch_density},
        {"food_grid", food_grid},
    };
    for (const auto& named : tensors) {
        check_cuda_contiguous(named.second, named.first);
        TORCH_CHECK(
            named.second.device() == patch_food_amount.device(),
            named.first,
            " must share the authoritative Patch CUDA device"
        );
        TORCH_CHECK(
            named.second.sizes() == patch_food_amount.sizes(),
            named.first,
            " must align with patch_food_amount"
        );
    }
    TORCH_CHECK(patch_food_amount.numel() > 0, "Patch P2 context grid must not be empty");
    TORCH_CHECK(steps > 0, "steps must be positive");
    TORCH_CHECK(patch_density_radius >= 0, "patch_density_radius must be nonnegative");
    for (const auto& tensor : {
             patch_food_amount,
             patch_fertility,
             patch_scent,
             patch_waste,
             patch_scent_emit_pending,
             patch_waste_emit_pending,
             patch_density,
         }) {
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "Patch P2 float grids must be float32");
    }
    TORCH_CHECK(agent_grid.scalar_type() == torch::kInt32, "agent_grid must be int32");
    TORCH_CHECK(food_grid.scalar_type() == torch::kInt32, "food_grid must be int32");
    for (size_t left = 0; left < tensors.size(); ++left) {
        for (size_t right = left + 1; right < tensors.size(); ++right) {
            TORCH_CHECK(
                tensors[left].second.data_ptr() != tensors[right].second.data_ptr(),
                tensors[left].first,
                " and ",
                tensors[right].first,
                " must not alias"
            );
        }
    }

    const double scent_decay = std::clamp(
        checked_finite(patch_scent_decay_per_step, "patch_scent_decay_per_step"), 0.0, 1.0
    );
    const double waste_decay = std::clamp(
        checked_finite(patch_waste_decay_per_step, "patch_waste_decay_per_step"), 0.0, 1.0
    );
    const double scent_presence = std::max(
        0.0, checked_finite(patch_scent_presence_emit, "patch_scent_presence_emit")
    );
    const double waste_presence = std::max(
        0.0, checked_finite(patch_waste_presence_emit, "patch_waste_presence_emit")
    );
    const double density_emit = std::max(
        0.0, checked_finite(patch_waste_density_emit_scale, "patch_waste_density_emit_scale")
    );
    const double scent_max_value = std::max(
        0.0, checked_finite(patch_scent_max, "patch_scent_max")
    );
    const double waste_max_value = std::max(
        0.0, checked_finite(patch_waste_max, "patch_waste_max")
    );
    const double food_max_value = std::max(
        0.0, checked_finite(patch_food_amount_max, "patch_food_amount_max")
    );
    const double regrow_value = std::max(
        0.0, checked_finite(patch_food_regrow_per_step, "patch_food_regrow_per_step")
    );
    const double threshold_value = std::clamp(
        checked_finite(patch_food_presence_threshold, "patch_food_presence_threshold"),
        0.0,
        food_max_value
    );
    const double waste_scale = std::max(
        0.0,
        checked_finite(patch_waste_regrowth_penalty_scale, "patch_waste_regrowth_penalty_scale")
    );
    const double waste_cap = std::clamp(
        checked_finite(patch_waste_regrowth_penalty_max, "patch_waste_regrowth_penalty_max"),
        0.0,
        1.0
    );
    const double density_scale = std::max(
        0.0,
        checked_finite(patch_density_regrowth_penalty_scale, "patch_density_regrowth_penalty_scale")
    );
    const double density_cap = std::clamp(
        checked_finite(patch_density_regrowth_penalty_max, "patch_density_regrowth_penalty_max"),
        0.0,
        1.0
    );
    const double min_multiplier = std::clamp(
        checked_finite(patch_regrowth_min_multiplier, "patch_regrowth_min_multiplier"),
        0.0,
        1.0
    );

    c10::cuda::CUDAGuard device_guard(patch_food_amount.device());
    auto scratch = torch::empty(
        {4, patch_food_amount.numel()},
        patch_food_amount.options()
    );
    auto stats_by_step = torch::empty(
        {steps, P2_STAT_COUNT},
        patch_food_amount.options().dtype(torch::kFloat64)
    );
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    patch_context_p2_steps_kernel<<<1, 1, 0, stream>>>(
        patch_food_amount.data_ptr<float>(),
        patch_fertility.data_ptr<float>(),
        patch_scent.data_ptr<float>(),
        patch_waste.data_ptr<float>(),
        patch_scent_emit_pending.data_ptr<float>(),
        patch_waste_emit_pending.data_ptr<float>(),
        agent_grid.data_ptr<int32_t>(),
        patch_density.data_ptr<float>(),
        food_grid.data_ptr<int32_t>(),
        scratch.select(0, 0).data_ptr<float>(),
        scratch.select(0, 1).data_ptr<float>(),
        scratch.select(0, 2).data_ptr<float>(),
        scratch.select(0, 3).data_ptr<float>(),
        stats_by_step.data_ptr<double>(),
        patch_food_amount.size(0),
        patch_food_amount.size(1),
        steps,
        static_cast<float>(1.0 - scent_decay),
        static_cast<float>(1.0 - waste_decay),
        scent_presence,
        waste_presence,
        static_cast<float>(density_emit),
        static_cast<float>(scent_max_value),
        static_cast<float>(waste_max_value),
        static_cast<float>(food_max_value),
        static_cast<float>(regrow_value),
        patch_food_regrow_use_fertility,
        patch_food_regrow_cap_by_fertility,
        static_cast<float>(threshold_value),
        patch_density_radius,
        patch_density_include_self,
        patch_density_normalize,
        patch_world_waste_consequence_enable,
        patch_waste_regrowth_penalty_enable,
        static_cast<float>(waste_scale),
        static_cast<float>(waste_cap),
        patch_density_regrowth_penalty_enable,
        static_cast<float>(density_scale),
        static_cast<float>(density_cap),
        static_cast<float>(min_multiplier)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return stats_by_step;
}
