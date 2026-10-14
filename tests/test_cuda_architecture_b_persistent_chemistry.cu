#include "engine/perf/cuda_architecture_b_persistent_chemistry.cuh"

#include <cstdint>

namespace persistent_chemistry =
    eiel::cuda_architecture_b::persistent_chemistry;

__global__ void test_cuda_architecture_b_persistent_chemistry_compile_probe(
    int32_t* ordered_slots,
    const float* drive_noise,
    float* chemistry,
    const float* energy,
    const float* health,
    const float* injury,
    const float* drives,
    const bool* alive_mask,
    uint64_t* shuffle_state_words,
    int64_t* shuffle_buffer_words,
    uint64_t* drive_state_words,
    int64_t* drive_buffer_words,
    uint64_t* chemistry_state_words,
    int64_t* chemistry_buffer_words,
    int64_t* shuffle_semantic_draws,
    int64_t* drive_semantic_draws,
    int64_t* policy_np_semantic_draws,
    int64_t* chemistry_semantic_draws,
    int32_t* status,
    const int64_t count,
    const int64_t capacity,
    const persistent_chemistry::ChemistryConfig config
) {
    if (threadIdx.x != 0) {
        return;
    }
    persistent_chemistry::runner_shuffle_counted(
        ordered_slots,
        count,
        shuffle_state_words,
        shuffle_buffer_words,
        shuffle_semantic_draws,
        status,
        capacity
    );
    persistent_chemistry::advance_precomputed_drive_noise_rng(
        ordered_slots,
        count,
        drive_noise,
        drive_state_words,
        drive_buffer_words,
        drive_semantic_draws,
        status,
        capacity
    );
    persistent_chemistry::chemistry_update_counted(
        chemistry,
        energy,
        health,
        injury,
        drives,
        alive_mask,
        ordered_slots,
        count,
        chemistry_state_words,
        chemistry_buffer_words,
        policy_np_semantic_draws,
        chemistry_semantic_draws,
        status,
        capacity,
        config
    );
}
