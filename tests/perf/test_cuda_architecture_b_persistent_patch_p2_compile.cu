#include "../../engine/perf/cuda_architecture_b_persistent_patch_p2.cuh"

namespace patch_p2 = eiel::cuda_architecture_b::persistent_patch_p2;

__global__ void compile_persistent_patch_p2_probe(
    const patch_p2::PatchP2State state,
    const patch_p2::PatchP2Scratch scratch,
    const patch_p2::PatchP2Outputs outputs,
    const patch_p2::PatchP2Config config
) {
    if (threadIdx.x != 0) {
        return;
    }
    patch_p2::patch_p2_single_tick_exact(state, scratch, outputs, config);
}
