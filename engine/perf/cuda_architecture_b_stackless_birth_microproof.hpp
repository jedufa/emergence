#ifndef EIEL_CUDA_ARCHITECTURE_B_STACKLESS_BIRTH_MICROPROOF_HPP_
#define EIEL_CUDA_ARCHITECTURE_B_STACKLESS_BIRTH_MICROPROOF_HPP_

#include <cuda_runtime.h>

#include <cstdint>

namespace eiel::cuda_architecture_b::stackless_birth_microproof {

struct FlatStacklessBirthArguments {
    void* scalar_state[36];
    void* scalar_rng[8];
    void* plan[3];
    void* scalar_outputs[3];
    uint32_t* trial_python_words;
    void* drive_state[16];
    void* drive_outputs[7];
    void* policy_state[7];
    void* policy_tape[5];
    void* policy_outputs[8];
    int64_t gauss_tape_capacity;
    int32_t sensor_radius_max;
    int64_t drive_width;
    int64_t chemistry_width;
    float max_energy;
    int64_t packed_width;
    int64_t hmem_width;
    int64_t torch_state_bytes;
    int64_t policy_tape_capacity;
};

struct FlatStacklessBirthPayload {
    // Fixed-layout, caller-preallocated receipt rows. The binding validates
    // extents: i64[16], i32[24], f32[16], status[1].
    int64_t* i64;
    int32_t* i32;
    float* f32;
    int32_t* status;
};

struct StacklessBirthFixtureConfig {
    int64_t parent_agent_id;
    int64_t child_agent_id;
    int64_t source_parent_slot;
    int64_t source_child_slot;
    int32_t spawn_y;
    int32_t spawn_x;
    int32_t parent_founder_id;
    int32_t parent_lineage_depth;
    int32_t generation;
    int32_t tick;
    float parent_energy_before;
    float child_energy;
    float parent_energy_after;
    float start_health;
    int32_t cooldown_steps;
};

struct StacklessBirthResourceReceipt {
    int32_t registers;
    int64_t static_stack_bytes;
    int64_t local_bytes;
    int64_t shared_bytes;
    int32_t max_threads_per_block;
    int32_t max_blocks_per_multiprocessor;
};

void launch_stackless_birth_microproof_v1(
    FlatStacklessBirthArguments arguments,
    FlatStacklessBirthPayload payload,
    StacklessBirthFixtureConfig fixture,
    cudaStream_t stream
);

StacklessBirthResourceReceipt query_stackless_birth_microproof_resources_v1();

}  // namespace eiel::cuda_architecture_b::stackless_birth_microproof

#endif
