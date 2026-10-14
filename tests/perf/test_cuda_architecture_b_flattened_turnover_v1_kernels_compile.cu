#include "../../engine/perf/cuda_architecture_b_flattened_turnover_v1_kernels.cuh"

#include <type_traits>

namespace turnover =
    eiel::cuda_architecture_b::flattened_turnover_v1;

using ExecutionConfig = turnover::FlatTurnoverExecutionConfig;
using LaunchReceipt = turnover::TurnoverLaunchReceipt;
using ResourceReceipt = turnover::KernelResourceReceipt;

static_assert(turnover::kTurnoverThreadsPerCell == 64);
static_assert(std::is_standard_layout_v<ExecutionConfig>);
static_assert(std::is_trivially_copyable_v<ExecutionConfig>);
static_assert(std::is_standard_layout_v<LaunchReceipt>);
static_assert(std::is_trivially_copyable_v<LaunchReceipt>);
static_assert(
    std::is_same_v<decltype(LaunchReceipt::shadow_plan_launches), int64_t>
);
static_assert(
    std::is_same_v<
        decltype(LaunchReceipt::cull_refill_plan_launches),
        int64_t
    >
);
static_assert(std::is_same_v<decltype(LaunchReceipt::commit_launches), int64_t>);
static_assert(std::is_same_v<decltype(LaunchReceipt::publish_launches), int64_t>);
static_assert(std::is_same_v<decltype(LaunchReceipt::cuda_status), int32_t>);
static_assert(std::is_same_v<decltype(ResourceReceipt::local_size_bytes), int64_t>);

void compile_flattened_turnover_v1_host_api(
    const cudaStream_t stream,
    const int64_t cells,
    const turnover::FlatTurnoverMutableTensorTable tensors,
    const turnover::FlatTurnoverScratch scratch,
    const turnover::FlatTurnoverOutputs outputs,
    const turnover::FlatTurnoverConfig config,
    const ExecutionConfig execution,
    const turnover::FlatOrdinaryBirthState birth_state,
    const turnover::FlatOrdinaryBirthConfig birth_config,
    LaunchReceipt* const launch_receipt,
    ResourceReceipt* const resources
) {
    launch_receipt[0] = turnover::run_flattened_turnover_v1(
        stream,
        cells,
        tensors,
        scratch,
        outputs,
        config,
        execution,
        birth_state,
        birth_config,
        {}
    );
    resources[0] =
        turnover::query_flattened_turnover_shadow_plan_resources_v1();
    resources[1] =
        turnover::query_flattened_turnover_cull_refill_plan_resources_v1();
    resources[2] = turnover::query_flattened_turnover_commit_resources_v1();
    resources[3] = turnover::query_flattened_turnover_publish_resources_v1();
}
