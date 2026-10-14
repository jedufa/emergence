#include "../../engine/perf/cuda_architecture_b_persistent_food_spawn.cuh"

namespace food_spawn = eiel::cuda_architecture_b::persistent_food_spawn;

__global__ void compile_persistent_food_spawn_probe(
    const food_spawn::FlatFoodSpawnState state,
    const food_spawn::FlatFoodSpawnRngCursor cursor,
    const food_spawn::FlatFoodSpawnOutputs outputs,
    const food_spawn::FlatFoodSpawnConfig config
) {
    if (threadIdx.x != 0) {
        return;
    }
    food_spawn::spawn_flat_food_exact(state, cursor, outputs, config);
}
