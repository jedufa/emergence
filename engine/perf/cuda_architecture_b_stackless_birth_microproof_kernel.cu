#include "cuda_architecture_b_stackless_birth_microproof.hpp"
#include "cuda_architecture_b_persistent_birth_drives.cuh"
#include "cuda_architecture_b_persistent_birth_policy.cuh"
#include "cuda_architecture_b_persistent_birth_scalar_genome.cuh"

#include <cuda_runtime.h>

namespace eiel::cuda_architecture_b::stackless_birth_microproof {
namespace scalar = persistent_birth_scalar_genome;
namespace drives = persistent_birth_drives;
namespace policy = persistent_birth_policy;
namespace {

enum MicroproofStatus : int32_t {
    kOk = 0,
    kScalarFailed = 1,
    kDrivesFailed = 2,
    kChemistryFailed = 3,
    kPolicyFailed = 4,
};

__global__ void stackless_birth_microproof_kernel_v1(
    const FlatStacklessBirthArguments arguments,
    const FlatStacklessBirthPayload payload,
    const StacklessBirthFixtureConfig fixture
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    const scalar::FlatFixedMlpScalarGenomeState scalar_state{
        static_cast<int64_t*>(arguments.scalar_state[0]),
        static_cast<bool*>(arguments.scalar_state[1]),
        static_cast<int32_t*>(arguments.scalar_state[2]),
        static_cast<int32_t*>(arguments.scalar_state[3]),
        static_cast<int32_t*>(arguments.scalar_state[4]),
        static_cast<int32_t*>(arguments.scalar_state[5]),
        static_cast<int32_t*>(arguments.scalar_state[6]),
        static_cast<int32_t*>(arguments.scalar_state[7]),
        static_cast<int32_t*>(arguments.scalar_state[8]),
        static_cast<int32_t*>(arguments.scalar_state[9]),
        static_cast<int64_t*>(arguments.scalar_state[10]),
        static_cast<bool*>(arguments.scalar_state[11]),
        static_cast<float*>(arguments.scalar_state[12]),
        static_cast<float*>(arguments.scalar_state[13]),
        static_cast<float*>(arguments.scalar_state[14]),
        static_cast<float*>(arguments.scalar_state[15]),
        static_cast<float*>(arguments.scalar_state[16]),
        static_cast<float*>(arguments.scalar_state[17]),
        static_cast<float*>(arguments.scalar_state[18]),
        static_cast<float*>(arguments.scalar_state[19]),
        static_cast<float*>(arguments.scalar_state[20]),
        static_cast<float*>(arguments.scalar_state[21]),
        static_cast<float*>(arguments.scalar_state[22]),
        static_cast<float*>(arguments.scalar_state[23]),
        static_cast<float*>(arguments.scalar_state[24]),
        static_cast<float*>(arguments.scalar_state[25]),
        static_cast<float*>(arguments.scalar_state[26]),
        static_cast<float*>(arguments.scalar_state[27]),
        static_cast<float*>(arguments.scalar_state[28]),
        static_cast<float*>(arguments.scalar_state[29]),
        static_cast<float*>(arguments.scalar_state[30]),
        static_cast<float*>(arguments.scalar_state[31]),
        static_cast<float*>(arguments.scalar_state[32]),
        static_cast<float*>(arguments.scalar_state[33]),
        static_cast<float*>(arguments.scalar_state[34]),
        static_cast<float*>(arguments.scalar_state[35]),
    };
    const scalar::FlatOrdinaryScalarGenomePlan scalar_plan{
        static_cast<int64_t*>(arguments.plan[0]),
        static_cast<int64_t*>(arguments.plan[1]),
        static_cast<int64_t*>(arguments.plan[2]),
    };
    const drives::FlatOrdinaryBirthPlan drive_plan{
        static_cast<int64_t*>(arguments.plan[0]),
        static_cast<int64_t*>(arguments.plan[1]),
        static_cast<int64_t*>(arguments.plan[2]),
    };
    const policy::FlatOrdinaryBirthPolicyPlan policy_plan{
        static_cast<int64_t*>(arguments.plan[0]),
        static_cast<int64_t*>(arguments.plan[1]),
        static_cast<int64_t*>(arguments.plan[2]),
    };
    const scalar::FlatCpythonRandomState python_random{
        static_cast<uint32_t*>(arguments.scalar_rng[0]),
        static_cast<int64_t*>(arguments.scalar_rng[1]),
        static_cast<bool*>(arguments.scalar_rng[2]),
        static_cast<double*>(arguments.scalar_rng[3]),
        static_cast<double*>(arguments.scalar_rng[4]),
        static_cast<double*>(arguments.scalar_rng[5]),
        static_cast<int64_t*>(arguments.scalar_rng[6]),
        static_cast<int64_t*>(arguments.scalar_rng[7]),
    };
    const scalar::FlatOrdinaryScalarGenomeOutputs scalar_outputs{
        static_cast<int64_t*>(arguments.scalar_outputs[0]),
        static_cast<int64_t*>(arguments.scalar_outputs[1]),
        static_cast<int32_t*>(arguments.scalar_outputs[2]),
    };
    const drives::FlatOrdinaryBirthDriveState drive_state{
        static_cast<int64_t*>(arguments.drive_state[0]),
        static_cast<bool*>(arguments.drive_state[1]),
        static_cast<bool*>(arguments.drive_state[2]),
        static_cast<double*>(arguments.drive_state[3]),
        static_cast<float*>(arguments.drive_state[4]),
        static_cast<float*>(arguments.drive_state[5]),
        static_cast<float*>(arguments.drive_state[6]),
        static_cast<float*>(arguments.drive_state[7]),
        static_cast<float*>(arguments.drive_state[8]),
        static_cast<float*>(arguments.drive_state[9]),
    };
    const drives::FlatPopulationPcg64Cursor drive_cursor{
        static_cast<uint64_t*>(arguments.drive_state[10]),
        static_cast<int64_t*>(arguments.drive_state[11]),
        static_cast<int64_t*>(arguments.drive_state[12]),
    };
    const drives::FlatOrdinaryBirthDriveOutputs drive_outputs{
        static_cast<float*>(arguments.drive_state[13]),
        static_cast<float*>(arguments.drive_state[14]),
        static_cast<int64_t*>(arguments.drive_outputs[0]),
        static_cast<int64_t*>(arguments.drive_outputs[1]),
        static_cast<int64_t*>(arguments.drive_outputs[2]),
        static_cast<int64_t*>(arguments.drive_outputs[3]),
        static_cast<int64_t*>(arguments.drive_outputs[4]),
        static_cast<int32_t*>(arguments.drive_outputs[5]),
    };
    const drives::FlatNewbornChemistryState chemistry_state{
        static_cast<int64_t*>(arguments.drive_state[0]),
        static_cast<bool*>(arguments.drive_state[1]),
        static_cast<double*>(arguments.drive_state[3]),
        static_cast<float*>(arguments.drive_state[15]),
    };
    const drives::FlatNewbornChemistryOutputs chemistry_outputs{
        static_cast<int32_t*>(arguments.drive_outputs[6]),
    };
    const policy::FlatOrdinaryBirthPolicyState policy_state{
        static_cast<int64_t*>(arguments.policy_state[0]),
        static_cast<bool*>(arguments.policy_state[1]),
        static_cast<float*>(arguments.policy_state[2]),
        static_cast<float*>(arguments.policy_state[3]),
        static_cast<bool*>(arguments.policy_state[4]),
        static_cast<uint8_t*>(arguments.policy_state[5]),
        static_cast<int64_t*>(arguments.policy_state[6]),
    };
    const policy::FlatOrdinaryBirthPolicyTape policy_tape{
        static_cast<float*>(arguments.policy_tape[0]),
        static_cast<bool*>(arguments.policy_tape[1]),
        static_cast<uint8_t*>(arguments.policy_tape[2]),
        static_cast<int64_t*>(arguments.policy_tape[3]),
        static_cast<int64_t*>(arguments.policy_tape[4]),
    };
    const policy::FlatOrdinaryBirthPolicyOutputs policy_outputs{
        static_cast<int64_t*>(arguments.policy_outputs[0]),
        static_cast<int64_t*>(arguments.policy_outputs[1]),
        static_cast<int64_t*>(arguments.policy_outputs[2]),
        static_cast<int64_t*>(arguments.policy_outputs[3]),
        static_cast<double*>(arguments.policy_outputs[4]),
        static_cast<bool*>(arguments.policy_outputs[5]),
        static_cast<float*>(arguments.policy_outputs[6]),
        static_cast<int32_t*>(arguments.policy_outputs[7]),
    };

    payload.status[0] = kOk;
    scalar::commit_ordinary_fixed_mlp_child_scalar_genome_with_trial_scratch_serial(
        scalar_state,
        scalar_plan,
        python_random,
        scalar_outputs,
        {1, 2, arguments.gauss_tape_capacity, arguments.sensor_radius_max},
        arguments.trial_python_words
    );
    if (scalar_outputs.status[0] != 0) {
        payload.status[0] = kScalarFailed;
        return;
    }

    drives::initialize_ordinary_newborn_drives_serial(
        drive_state,
        drive_plan,
        drive_cursor,
        drive_outputs,
        {2, 1, arguments.drive_width, arguments.max_energy}
    );
    if (drive_outputs.status[0] != 0) {
        payload.status[0] = kDrivesFailed;
        return;
    }

    drives::initialize_newborn_chemistry_serial(
        chemistry_state,
        drive_plan,
        chemistry_outputs,
        {2, 1, arguments.chemistry_width, arguments.max_energy}
    );
    if (chemistry_outputs.status[0] != 0) {
        payload.status[0] = kChemistryFailed;
        return;
    }

    policy::inherit_ordinary_fixed_mlp_child_policy_serial(
        policy_state,
        policy_plan,
        policy_tape,
        policy_outputs,
        {1, 2, arguments.packed_width, arguments.hmem_width,
         arguments.torch_state_bytes, arguments.policy_tape_capacity}
    );
    if (policy_outputs.status[0] != 0) {
        payload.status[0] = kPolicyFailed;
        return;
    }

    // Prepared identity, lineage, history, and event payload. These rows are
    // global scratch only; the microproof never publishes authoritative state.
    payload.i64[0] = fixture.parent_agent_id;
    payload.i64[1] = fixture.child_agent_id;
    payload.i64[2] = fixture.source_parent_slot;
    payload.i64[3] = fixture.source_child_slot;
    payload.i64[4] = fixture.child_agent_id + 1;
    payload.i64[5] = 1;  // planned count
    payload.i64[6] = 1;  // prepared count
    payload.i64[7] = 1;  // birth event rows
    payload.i64[8] = policy_outputs.mutation_events[0] != 0 ? 1 : 0;
    payload.i64[9] = scalar_outputs.semantic_draws_added[0];
    payload.i64[10] = drive_outputs.semantic_draws_added[0];
    payload.i64[11] = policy_outputs.semantic_draws_added[0];
    payload.i64[12] = python_random.python_random_raw_draws[0];
    payload.i64[13] = drive_cursor.mutation_np_semantic_draws[0];
    payload.i64[14] = policy_state.mutation_torch_cpu_semantic_draws[0];
    payload.i64[15] = 1;  // valid transaction stamp

    payload.i32[0] = fixture.spawn_y;
    payload.i32[1] = fixture.spawn_x;
    payload.i32[2] = fixture.parent_founder_id;
    payload.i32[3] = fixture.parent_lineage_depth + 1;
    payload.i32[4] = fixture.generation;
    payload.i32[5] = fixture.tick;
    payload.i32[6] = fixture.cooldown_steps;
    payload.i32[7] = -1;  // last action
    payload.i32[8] = -1;  // last-last action
    payload.i32[9] = 0;   // switches
    payload.i32[10] = 0;  // oscillations
    payload.i32[11] = 0;  // episode-0 switches
    payload.i32[12] = 0;  // episode-0 oscillations
    payload.i32[13] = 1;  // birth event type
    payload.i32[14] = policy_outputs.mutation_events[0] != 0 ? 2 : 0;
    payload.i32[15] = 0;  // success reason
    for (int index = 16; index < 24; ++index) {
        payload.i32[index] = 0;
    }

    payload.f32[0] = fixture.parent_energy_before;
    payload.f32[1] = fixture.parent_energy_after;
    payload.f32[2] = fixture.child_energy;
    payload.f32[3] = fixture.start_health;
    payload.f32[4] = 0.0f;  // child injury
    payload.f32[5] = 0.0f;  // child scar
    payload.f32[6] = drive_outputs.birth_reset_shock[1];
    payload.f32[7] = policy_outputs.mutation_delta_by_birth[0];
    for (int index = 8; index < 16; ++index) {
        payload.f32[index] = 0.0f;
    }
}

}  // namespace

void launch_stackless_birth_microproof_v1(
    FlatStacklessBirthArguments arguments,
    FlatStacklessBirthPayload payload,
    StacklessBirthFixtureConfig fixture,
    cudaStream_t stream
) {
    stackless_birth_microproof_kernel_v1<<<1, 128, 0, stream>>>(
        arguments,
        payload,
        fixture
    );
}

StacklessBirthResourceReceipt query_stackless_birth_microproof_resources_v1() {
    cudaFuncAttributes attributes{};
    if (cudaFuncGetAttributes(
            &attributes,
            stackless_birth_microproof_kernel_v1
        ) != cudaSuccess) {
        return {-1, -1, -1, -1, -1, -1};
    }
    int32_t blocks = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks,
            stackless_birth_microproof_kernel_v1,
            128,
            0
        ) != cudaSuccess) {
        blocks = -1;
    }
    return {
        attributes.numRegs,
        static_cast<int64_t>(attributes.localSizeBytes),
        static_cast<int64_t>(attributes.localSizeBytes),
        static_cast<int64_t>(attributes.sharedSizeBytes),
        attributes.maxThreadsPerBlock,
        blocks,
    };
}

}  // namespace eiel::cuda_architecture_b::stackless_birth_microproof
