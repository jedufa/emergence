#include "../../engine/perf/cuda_architecture_b_persistent_birth_identity.cuh"

#include <type_traits>

namespace birth_identity =
    eiel::cuda_architecture_b::persistent_birth_identity;

using BirthState = birth_identity::FlatOrdinaryBirthIdentityState;

static_assert(birth_identity::kStatusOk == 0);
static_assert(birth_identity::kStatusInvalidBirthPlan == 10);
static_assert(birth_identity::kStatusInvalidParentOrder == 11);
static_assert(birth_identity::kStatusParentNotViable == 12);
static_assert(birth_identity::kStatusNoFreeSlot == 13);
static_assert(birth_identity::kStatusAgentIdCapacity == 14);
static_assert(birth_identity::kStatusSpawnReservation == 15);
static_assert(birth_identity::kStatusInvalidFreeQueueCount == 16);

static_assert(std::is_same_v<decltype(BirthState::slot_to_agent_id), int64_t*>);
static_assert(std::is_same_v<decltype(BirthState::agent_id_to_slot), int32_t*>);
static_assert(std::is_same_v<decltype(BirthState::slot_tokens), int64_t*>);
static_assert(std::is_same_v<decltype(BirthState::slot_free_queue), int64_t*>);
static_assert(
    std::is_same_v<decltype(BirthState::slot_free_queue_count), int64_t*>
);
static_assert(std::is_same_v<decltype(BirthState::population_next_id), int64_t*>);
static_assert(std::is_same_v<decltype(BirthState::lineage_founder_id), int32_t*>);
static_assert(std::is_same_v<decltype(BirthState::lineage_depth), int32_t*>);
static_assert(
    std::is_same_v<decltype(BirthState::lineage_birth_generation), int32_t*>
);
static_assert(
    std::is_same_v<decltype(BirthState::lineage_birth_tick), int32_t*>
);
static_assert(
    std::is_same_v<decltype(BirthState::patch_habitat_lineage_id), int32_t*>
);

__global__ void compile_persistent_birth_identity_probe(
    const birth_identity::FlatOrdinaryBirthIdentityState state,
    const birth_identity::FlatOrdinaryBirthPlan plan,
    const birth_identity::FlatOrdinaryBirthIdentityOutputs outputs,
    const birth_identity::OrdinaryBirthIdentityConfig config
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    birth_identity::commit_ordinary_birth_identity_serial(
        state,
        plan,
        outputs,
        config
    );
}
