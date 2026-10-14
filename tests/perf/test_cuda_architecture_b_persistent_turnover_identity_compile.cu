#include "../../engine/perf/cuda_architecture_b_persistent_turnover_identity.cuh"

#include <type_traits>

namespace turnover_identity =
    eiel::cuda_architecture_b::persistent_turnover_identity;

using CullState = turnover_identity::FlatTurnoverCullReleaseState;
using IdentityState = turnover_identity::FlatTurnoverRefillIdentityState;

static_assert(turnover_identity::kCullStatusOk == 0);
static_assert(turnover_identity::kCullStatusPlanFailed == 1);
static_assert(turnover_identity::kCullStatusInvalidResidentState == 2);
static_assert(turnover_identity::kCullStatusQueueCapacityExceeded == 3);
static_assert(turnover_identity::kCullStatusReleaseContradiction == 4);
static_assert(turnover_identity::kIdentityStatusOk == 0);
static_assert(turnover_identity::kIdentityStatusInvalidBirthPlan == 10);
static_assert(turnover_identity::kIdentityStatusInvalidParentOrder == 11);
static_assert(turnover_identity::kIdentityStatusParentNotViable == 12);
static_assert(turnover_identity::kIdentityStatusNoFreeSlot == 13);
static_assert(turnover_identity::kIdentityStatusAgentIdCapacity == 14);
static_assert(turnover_identity::kIdentityStatusSpawnReservation == 15);
static_assert(turnover_identity::kIdentityStatusInvalidFreeQueueCount == 16);

static_assert(std::is_same_v<decltype(CullState::slot_to_agent_id), int64_t*>);
static_assert(std::is_same_v<decltype(CullState::agent_id_to_slot), int32_t*>);
static_assert(std::is_same_v<decltype(CullState::slot_free_queue), int64_t*>);
static_assert(
    std::is_same_v<decltype(CullState::pending_free_agent_ids), const int64_t*>
);
static_assert(
    std::is_same_v<decltype(CullState::pending_free_slots), const int64_t*>
);
static_assert(std::is_same_v<decltype(IdentityState::slot_tokens), int64_t*>);
static_assert(std::is_same_v<decltype(IdentityState::slot_to_agent_id), int64_t*>);
static_assert(std::is_same_v<decltype(IdentityState::agent_id_to_slot), int32_t*>);
static_assert(std::is_same_v<decltype(IdentityState::alive_mask), bool*>);
static_assert(std::is_same_v<decltype(IdentityState::pos_y), int32_t*>);
static_assert(std::is_same_v<decltype(IdentityState::pos_x), int32_t*>);
static_assert(std::is_same_v<decltype(IdentityState::agent_grid), int32_t*>);
static_assert(std::is_same_v<decltype(IdentityState::energy), float*>);
static_assert(std::is_same_v<decltype(IdentityState::health), float*>);
static_assert(std::is_same_v<decltype(IdentityState::injury), float*>);
static_assert(std::is_same_v<decltype(IdentityState::scar), float*>);
static_assert(std::is_same_v<decltype(IdentityState::age), int32_t*>);
static_assert(std::is_same_v<decltype(IdentityState::repro_cooldown), int32_t*>);
static_assert(std::is_same_v<decltype(IdentityState::repro_threshold), float*>);
static_assert(std::is_same_v<decltype(IdentityState::offspring_count), int32_t*>);
static_assert(
    std::is_same_v<decltype(IdentityState::population_next_id), int64_t*>
);
static_assert(
    std::is_same_v<decltype(IdentityState::lineage_founder_id), int32_t*>
);
static_assert(
    std::is_same_v<decltype(IdentityState::lineage_depth), int32_t*>
);
static_assert(
    std::is_same_v<decltype(IdentityState::lineage_birth_generation), int32_t*>
);
static_assert(
    std::is_same_v<decltype(IdentityState::lineage_birth_tick), int32_t*>
);
static_assert(
    std::is_same_v<decltype(IdentityState::patch_habitat_lineage_id), int32_t*>
);
static_assert(
    std::is_same_v<decltype(IdentityState::slot_free_queue_count), int64_t*>
);
static_assert(
    std::is_same_v<decltype(IdentityState::policy_energy_observation), double*>
);
static_assert(
    std::is_same_v<decltype(IdentityState::survival_steps), int32_t*>
);
static_assert(std::is_same_v<decltype(IdentityState::rest_streak), int32_t*>);
static_assert(
    std::is_same_v<decltype(IdentityState::action_cache_token), int64_t*>
);
static_assert(
    std::is_same_v<decltype(IdentityState::action_cache_cursor), int32_t*>
);
static_assert(
    std::is_same_v<decltype(IdentityState::action_cache_next), int32_t*>
);
static_assert(std::is_same_v<decltype(IdentityState::slot_free_queue), int64_t*>);

__global__ void compile_persistent_turnover_cull_release_probe(
    const turnover_identity::FlatTurnoverCullReleaseState state,
    const turnover_identity::FlatTurnoverCullPlan plan,
    const turnover_identity::FlatTurnoverCullReleaseOutputs outputs,
    const turnover_identity::TurnoverCullReleaseConfig config
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    turnover_identity::commit_turnover_cull_release_serial(
        state,
        plan,
        outputs,
        config
    );
}

__global__ void compile_persistent_turnover_refill_identity_probe(
    const turnover_identity::FlatTurnoverRefillIdentityState state,
    const turnover_identity::FlatTurnoverRefillIdentityPlan plan,
    const turnover_identity::FlatTurnoverRefillIdentityOutputs outputs,
    const turnover_identity::TurnoverRefillIdentityConfig config
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    turnover_identity::commit_turnover_refill_identity_serial(
        state,
        plan,
        outputs,
        config
    );
}
