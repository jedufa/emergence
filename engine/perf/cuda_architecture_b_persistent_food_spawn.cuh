#ifndef EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_FOOD_SPAWN_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_FOOD_SPAWN_CUH_

#include <cuda_runtime.h>

#include <cstdint>

#include "native/cuda_architecture_b_pcg64.cuh"

namespace eiel::cuda_architecture_b::persistent_food_spawn {

constexpr int32_t kFood = 1;
constexpr int64_t kStatCount = 3;

enum FoodSpawnStat : int64_t {
    kAttempts = 0,
    kPlaced = 1,
    kPatchSyncRows = 2,
};

// Every pointer is already offset to the simulation cell owned by the caller.
// patch_food_amount is null exactly when config.patch_enabled is false.
struct FlatFoodSpawnState {
    int32_t* food_grid;
    float* patch_food_amount;
};

// Fixed-capacity cursor contract: state_words has exactly four uint64 values,
// buffer_words has exactly two int64 values, and semantic_draws has one int64.
// The helper loads this cursor only after the immutable Stage2 early-return
// gates, then writes it back once after all spawn attempts have completed.
struct FlatFoodSpawnRngCursor {
    uint64_t* state_words;
    int64_t* buffer_words;
    int64_t* semantic_draws;
};

// stats has exactly kStatCount int64 values for the current tick.
struct FlatFoodSpawnOutputs {
    int64_t* stats;
};

// All fields are host-validated immutable Stage2 values. cell_count must be
// positive, and food_grid contains exactly cell_count row-major Flat cells.
struct FlatFoodSpawnConfig {
    int64_t cell_count;
    int64_t max_food;
    int64_t effective_spawn_count;
    bool patch_enabled;
    float patch_food_presence_threshold;
    float patch_food_initial_value;
};

// Caller contract: exactly one device thread invokes this helper for one cell.
// It is allocation-free and reproduces the Stage2 Flat weighted-choice route:
// the positive flat density map normalizes to uniform 1/cell_count weights;
// candidates are implicit row-major indices [0, cell_count), selected with
// floor(PCG64 double01 * cell_count). An occupied draw is a retry, not a
// failure status. If every candidate is occupied, it consumes max_tries draws,
// records zero placements, and still persists the advanced cursor.
__device__ __forceinline__ void spawn_flat_food_exact(
    const FlatFoodSpawnState state,
    const FlatFoodSpawnRngCursor cursor,
    const FlatFoodSpawnOutputs outputs,
    const FlatFoodSpawnConfig config
) {
    outputs.stats[kAttempts] = 0;
    outputs.stats[kPlaced] = 0;
    outputs.stats[kPatchSyncRows] = 0;

    int64_t current_food = 0;
    for (int64_t cell = 0; cell < config.cell_count; ++cell) {
        current_food += state.food_grid[cell] == kFood ? 1 : 0;
    }
    if (
        config.max_food <= 0
        || config.effective_spawn_count <= 0
        || current_food >= config.max_food
    ) {
        return;
    }

    const int64_t target = config.effective_spawn_count < config.max_food - current_food
        ? config.effective_spawn_count
        : config.max_food - current_food;
    const int64_t target_budget = target * 10;
    const int64_t max_tries = target_budget > 80 ? target_budget : 80;
    Pcg64State rng{
        cursor.state_words[0],
        cursor.state_words[1],
        cursor.state_words[2],
        cursor.state_words[3],
        static_cast<uint32_t>(cursor.buffer_words[0]),
        static_cast<uint32_t>(cursor.buffer_words[1]),
    };

    int64_t placed = 0;
    int64_t tries = 0;
    while (placed < target && tries < max_tries) {
        ++tries;
        int64_t cell = static_cast<int64_t>(
            pcg64_double01(rng) * static_cast<double>(config.cell_count)
        );
        if (cell < 0) {
            cell = 0;
        } else if (cell >= config.cell_count) {
            cell = config.cell_count - 1;
        }
        if (state.food_grid[cell] != kFood) {
            state.food_grid[cell] = kFood;
            ++placed;
        }
    }

    int64_t patch_sync_rows = 0;
    if (config.patch_enabled) {
        for (int64_t cell = 0; cell < config.cell_count; ++cell) {
            if (
                state.food_grid[cell] == kFood
                && state.patch_food_amount[cell]
                    <= config.patch_food_presence_threshold
            ) {
                state.patch_food_amount[cell] = config.patch_food_initial_value;
                ++patch_sync_rows;
            }
        }
        for (int64_t cell = 0; cell < config.cell_count; ++cell) {
            state.food_grid[cell] = state.patch_food_amount[cell]
                    > config.patch_food_presence_threshold
                ? kFood
                : 0;
        }
    }

    cursor.state_words[0] = rng.state_lo;
    cursor.state_words[1] = rng.state_hi;
    cursor.state_words[2] = rng.inc_lo;
    cursor.state_words[3] = rng.inc_hi;
    cursor.buffer_words[0] = static_cast<int64_t>(rng.has_uint32);
    cursor.buffer_words[1] = static_cast<int64_t>(rng.uinteger);
    cursor.semantic_draws[0] += tries;
    outputs.stats[kAttempts] = tries;
    outputs.stats[kPlaced] = placed;
    outputs.stats[kPatchSyncRows] = patch_sync_rows;
}

}  // namespace eiel::cuda_architecture_b::persistent_food_spawn

#endif  // EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_FOOD_SPAWN_CUH_
