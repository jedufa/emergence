#include "../../engine/perf/cuda_architecture_b_flattened_turnover_v1.cuh"

#include <type_traits>

namespace turnover =
    eiel::cuda_architecture_b::flattened_turnover_v1;

using TensorTriplet = turnover::TurnoverMutableTensorTriplet;
using TensorTable = turnover::FlatTurnoverMutableTensorTable;
using Scratch = turnover::FlatTurnoverScratch;
using Layout = turnover::FlatTurnoverWorkLayout;
using Outputs = turnover::FlatTurnoverOutputs;
using Config = turnover::FlatTurnoverConfig;

static_assert(turnover::kActiveAgentIds == 0);
static_assert(turnover::kActiveAgentSlots == 1);
static_assert(turnover::kPolicyEnergyObservation == 13);
static_assert(turnover::kPythonRandomWords == 42);
static_assert(turnover::kPopulationPcg64StateWords == 83);
static_assert(turnover::kEnergyCollected == 92);
static_assert(turnover::kPolicyNpSemanticDraws == 121);
static_assert(turnover::kTurnoverMutableTensorCount == 122);
static_assert(turnover::kTurnoverOutputCount == 8);
static_assert(turnover::kTurnoverFailureDiagnosticStrideV1 == 1000);
static_assert(
    turnover::encode_turnover_failure_diagnostic_v1(
        turnover::kScalarStatus + 1,
        28
    ) == 5028
);
static_assert(turnover::turnover_failure_diagnostic_phase_v1(5028) == 5);
static_assert(turnover::turnover_failure_diagnostic_substatus_v1(5028) == 28);
static_assert(turnover::kMaximumCapacity == 768);
static_assert(turnover::kMaximumPending == 768);
static_assert(turnover::kMaximumReleaseRows == 1536);
static_assert(Layout::kRequiredBytesPerCell == 132096);
static_assert(
    Layout::kRequiredBytesPerCell
    <= turnover::kMaximumScratchBytesPerCell
);
static_assert(Layout::kRequiredBytesPerCell % 256 == 0);
static_assert(std::is_trivially_copyable<TensorTriplet>::value);
static_assert(std::is_trivially_copyable<TensorTable>::value);
static_assert(std::is_trivially_copyable<Scratch>::value);
static_assert(std::is_trivially_copyable<Outputs>::value);
static_assert(std::is_trivially_copyable<Config>::value);
static_assert(
    std::is_same<decltype(Outputs::ranked_agent_ids), int64_t*>::value
);
static_assert(std::is_same<decltype(Outputs::counts), int64_t*>::value);
static_assert(std::is_same<decltype(Outputs::status), int32_t*>::value);

__global__ void compile_flattened_turnover_v1_probe(
    const TensorTable tensors,
    const Scratch scratch,
    const Outputs outputs,
    const Config config
) {
    const int64_t cell = static_cast<int64_t>(blockIdx.x);
    const int64_t lane = static_cast<int64_t>(threadIdx.x);
    const int64_t lanes = static_cast<int64_t>(blockDim.x);

    tensors.copy_source_to_shadow(
        turnover::kActiveAgentIds, cell, lane, lanes
    );
    tensors.copy_shadow_to_source(
        turnover::kPolicyNpSemanticDraws, cell, lane, lanes
    );

    int64_t* const source_ids = tensors.source<int64_t>(
        turnover::kActiveAgentIds, cell
    );
    int64_t* const shadow_ids = tensors.shadow<int64_t>(
        turnover::kActiveAgentIds, cell
    );
    uint8_t* const source_raw = tensors.source_bytes(
        turnover::kMutationTorchCpuState, cell
    );
    uint8_t* const shadow_raw = tensors.shadow_bytes(
        turnover::kMutationTorchCpuState, cell
    );

    uint32_t* const python_a = scratch.at<uint32_t>(
        cell, Layout::kCpythonWordsA
    );
    uint32_t* const python_b = turnover::at<uint32_t>(
        scratch, cell, Layout::kCpythonWordsB
    );
    int64_t* const ranked_slots = scratch.at<int64_t>(
        cell, Layout::kRankedSlots
    );
    double* const ranked_fitness = scratch.at<double>(
        cell, Layout::kRankedFitness
    );
    int32_t* const spawn_positions = scratch.at<int32_t>(
        cell, Layout::kRefillSpawnPositions
    );
    int64_t* const release_ids = scratch.at<int64_t>(
        cell, Layout::kReleaseAgentIds
    );
    bool* const policy_mutated = scratch.at<bool>(
        cell, Layout::kPolicyMutationOccurred
    );
    float* const policy_delta = scratch.at<float>(
        cell, Layout::kPolicyMutationDelta
    );
    int64_t* const counters = scratch.at<int64_t>(
        cell, Layout::kScalarCounters
    );
    int32_t* const statuses = scratch.at<int32_t>(
        cell, Layout::kPhaseStatuses
    );

    if (lane == 0 && config.enabled) {
        shadow_ids[0] = source_ids[0];
        shadow_raw[0] = source_raw[0];
        python_b[0] = python_a[0];
        ranked_slots[0] = 0;
        ranked_fitness[0] = 0.0;
        spawn_positions[0] = 0;
        release_ids[0] = -1;
        policy_mutated[0] = false;
        policy_delta[0] = 0.0f;
        counters[turnover::kAggregateCommittedCount] = 0;
        statuses[turnover::kAggregateStatus] = 0;
        outputs.counts[turnover::kOutputFinalLiveCount] = 0;
        outputs.status[0] = 0;
        outputs.next_generation[0] = 1;
        outputs.next_tick[0] = 0;
    }
}
