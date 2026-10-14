#include "../../engine/perf/cuda_architecture_b_persistent_birth_policy.cuh"

#include <type_traits>

namespace birth_policy =
    eiel::cuda_architecture_b::persistent_birth_policy;

using PolicyState = birth_policy::FlatOrdinaryBirthPolicyState;
using PolicyTape = birth_policy::FlatOrdinaryBirthPolicyTape;
using PolicyOutputs = birth_policy::FlatOrdinaryBirthPolicyOutputs;

static_assert(birth_policy::kStatusOk == 0);
static_assert(birth_policy::kStatusInvalidPlan == 1);
static_assert(birth_policy::kStatusTapeExhausted == 2);

static_assert(
    std::is_same_v<decltype(PolicyState::mlp_pack), float*>
);
static_assert(
    std::is_same_v<decltype(PolicyState::hmem), float*>
);
static_assert(
    std::is_same_v<decltype(PolicyState::mutation_torch_cpu_state), uint8_t*>
);
static_assert(
    std::is_same_v<decltype(PolicyTape::mutation_delta), const float*>
);
static_assert(
    std::is_same_v<decltype(PolicyTape::mutation_mask), const bool*>
);
static_assert(
    std::is_same_v<decltype(PolicyTape::packed_post_torch_cpu_state), const uint8_t*>
);
static_assert(
    std::is_same_v<decltype(PolicyTape::cursor), int64_t*>
);
static_assert(
    std::is_same_v<decltype(PolicyOutputs::mutation_occurred_by_birth), bool*>
);
static_assert(
    std::is_same_v<decltype(PolicyOutputs::mutation_delta_by_birth), float*>
);

__global__ void compile_persistent_birth_policy_probe(
    const birth_policy::FlatOrdinaryBirthPolicyState state,
    const birth_policy::FlatOrdinaryBirthPolicyPlan plan,
    const birth_policy::FlatOrdinaryBirthPolicyTape tape,
    const birth_policy::FlatOrdinaryBirthPolicyOutputs outputs,
    const birth_policy::OrdinaryBirthPolicyConfig config
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    birth_policy::inherit_ordinary_fixed_mlp_child_policy_serial(
        state,
        plan,
        tape,
        outputs,
        config
    );
}
