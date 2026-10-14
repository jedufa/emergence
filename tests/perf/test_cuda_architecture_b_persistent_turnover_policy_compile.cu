#include "../../engine/perf/cuda_architecture_b_persistent_turnover_policy.cuh"

#include <type_traits>

namespace turnover_policy =
    eiel::cuda_architecture_b::persistent_turnover_policy;

using PolicyState = turnover_policy::FlatTurnoverPolicyState;
using PolicyPlan = turnover_policy::FlatTurnoverPolicyPlan;
using PolicyTape = turnover_policy::FlatTurnoverPolicyTape;
using PolicyOutputs = turnover_policy::FlatTurnoverPolicyOutputs;
using PolicyConfig = turnover_policy::TurnoverPolicyConfig;

static_assert(turnover_policy::kStatusOk == 0);
static_assert(turnover_policy::kStatusInvalidPlan == 1);
static_assert(turnover_policy::kStatusTapeExhausted == 2);

static_assert(std::is_trivially_copyable_v<PolicyState>);
static_assert(std::is_trivially_copyable_v<PolicyPlan>);
static_assert(std::is_trivially_copyable_v<PolicyTape>);
static_assert(std::is_trivially_copyable_v<PolicyOutputs>);
static_assert(std::is_standard_layout_v<PolicyConfig>);
static_assert(std::is_trivially_copyable_v<PolicyConfig>);
static_assert(
    std::is_same_v<decltype(PolicyState::slot_to_agent_id), const int64_t*>
);
static_assert(
    std::is_same_v<decltype(PolicyState::alive_mask), const bool*>
);
static_assert(std::is_same_v<decltype(PolicyState::mlp_pack), float*>);
static_assert(std::is_same_v<decltype(PolicyState::hmem), float*>);
static_assert(
    std::is_same_v<decltype(PolicyState::policy_use_gru), bool*>
);
static_assert(
    std::is_same_v<decltype(PolicyState::mutation_torch_cpu_state), uint8_t*>
);
static_assert(
    std::is_same_v<
        decltype(PolicyState::mutation_torch_cpu_semantic_draws),
        int64_t*
    >
);
static_assert(
    std::is_same_v<decltype(PolicyPlan::parent_slots), const int64_t*>
);
static_assert(
    std::is_same_v<decltype(PolicyPlan::mate_slots), const int64_t*>
);
static_assert(
    std::is_same_v<decltype(PolicyPlan::child_slots), const int64_t*>
);
static_assert(
    std::is_same_v<decltype(PolicyTape::mutation_delta), const float*>
);
static_assert(
    std::is_same_v<decltype(PolicyTape::mutation_mask), const bool*>
);
static_assert(
    std::is_same_v<decltype(PolicyTape::crossover_mask_packed), const uint8_t*>
);
static_assert(
    std::is_same_v<decltype(PolicyTape::packed_post_torch_cpu_state), const uint8_t*>
);
static_assert(std::is_same_v<decltype(PolicyTape::cursor), int64_t*>);
static_assert(
    std::is_same_v<decltype(PolicyTape::semantic_draws), const int64_t*>
);
static_assert(
    std::is_same_v<decltype(PolicyOutputs::committed_count), int64_t*>
);
static_assert(
    std::is_same_v<decltype(PolicyOutputs::semantic_draws_added), int64_t*>
);
static_assert(
    std::is_same_v<decltype(PolicyOutputs::mutation_events), int64_t*>
);
static_assert(
    std::is_same_v<decltype(PolicyOutputs::mutated_elements), int64_t*>
);
static_assert(
    std::is_same_v<decltype(PolicyOutputs::mutation_delta_total), double*>
);
static_assert(
    std::is_same_v<decltype(PolicyOutputs::mutation_occurred_by_birth), bool*>
);
static_assert(
    std::is_same_v<decltype(PolicyOutputs::mutation_delta_by_birth), float*>
);
static_assert(std::is_same_v<decltype(PolicyOutputs::status), int32_t*>);
static_assert(
    std::is_same_v<decltype(PolicyConfig::plan_capacity), int64_t>
);
static_assert(std::is_same_v<decltype(PolicyConfig::slot_count), int64_t>);
static_assert(
    std::is_same_v<decltype(PolicyConfig::mlp_row_width), int64_t>
);
static_assert(std::is_same_v<decltype(PolicyConfig::hmem_width), int64_t>);
static_assert(
    std::is_same_v<decltype(PolicyConfig::torch_state_bytes), int64_t>
);
static_assert(
    std::is_same_v<decltype(PolicyConfig::tape_capacity), int64_t>
);

__global__ void compile_persistent_turnover_policy_probe(
    const turnover_policy::FlatTurnoverPolicyState state,
    const turnover_policy::FlatTurnoverPolicyPlan plan,
    const turnover_policy::FlatTurnoverPolicyTape tape,
    const turnover_policy::FlatTurnoverPolicyOutputs outputs,
    const turnover_policy::TurnoverPolicyConfig config
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    turnover_policy::inherit_fixed_mlp_turnover_child_policy_serial(
        state,
        plan,
        tape,
        outputs,
        config
    );
}
