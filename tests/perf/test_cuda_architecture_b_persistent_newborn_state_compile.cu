#include "../../engine/perf/cuda_architecture_b_persistent_newborn_state.cuh"

namespace newborn_state =
    eiel::cuda_architecture_b::persistent_newborn_state;

__global__ void compile_persistent_newborn_state_probe(
    const newborn_state::FlatFreshNewbornSelectionState state,
    const int64_t child_slot,
    const newborn_state::FreshNewbornSelectionConfig config
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    newborn_state::reset_ordinary_newborn_selection_state(
        state,
        child_slot,
        config
    );
}
