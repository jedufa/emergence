#include "../../engine/perf/cuda_architecture_b_persistent_turnover_newborn_state.cuh"

#include <cstdint>
#include <type_traits>

namespace turnover_newborn_state =
    eiel::cuda_architecture_b::persistent_turnover_newborn_state;

using NewbornState = turnover_newborn_state::FlatTurnoverNewbornState;
using ReproductionState =
    turnover_newborn_state::FlatTurnoverReproductionState;
using ReproductionRows = turnover_newborn_state::FlatTurnoverReproductionRows;
using ReproductionOutputs =
    turnover_newborn_state::FlatTurnoverReproductionOutputs;

static_assert(turnover_newborn_state::kStatusOk == 0);
static_assert(turnover_newborn_state::kStatusInvalidConfig == 1);
static_assert(turnover_newborn_state::kStatusInvalidCommittedCount == 2);
static_assert(turnover_newborn_state::kStatusParentMateAlias == 5);

static_assert(std::is_trivially_copyable_v<NewbornState>);
static_assert(std::is_trivially_copyable_v<ReproductionState>);
static_assert(std::is_trivially_copyable_v<ReproductionRows>);
static_assert(std::is_trivially_copyable_v<ReproductionOutputs>);
static_assert(
    std::is_same_v<decltype(NewbornState::energy_collected),
                   turnover_newborn_state::FlatStridedSlotField<float>>
);
static_assert(
    std::is_same_v<decltype(NewbornState::action_cache_actions),
                   turnover_newborn_state::FlatStridedRowField<int32_t>>
);
static_assert(
    std::is_same_v<decltype(NewbornState::offspring_count),
                   turnover_newborn_state::FlatStridedSlotField<int32_t>>
);
static_assert(
    std::is_same_v<decltype(ReproductionState::offspring_count), int32_t*>
);
static_assert(
    std::is_same_v<decltype(ReproductionRows::parent_slots), const int64_t*>
);
static_assert(
    std::is_same_v<decltype(ReproductionRows::committed_count),
                   const int64_t*>
);
static_assert(
    std::is_same_v<decltype(ReproductionOutputs::status), int32_t*>
);

__global__ void compile_persistent_turnover_newborn_state_probe(
    const NewbornState state,
    const int64_t child_slot,
    const turnover_newborn_state::TurnoverNewbornStateConfig config
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    turnover_newborn_state::reset_turnover_newborn_state_serial(
        state,
        child_slot,
        config
    );
}

__global__ void compile_persistent_turnover_reproduction_stats_probe(
    const ReproductionState state,
    const ReproductionRows rows,
    const ReproductionOutputs outputs,
    const turnover_newborn_state::TurnoverReproductionConfig config
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    turnover_newborn_state::update_turnover_parent_reproduction_stats_serial(
        state,
        rows,
        outputs,
        config
    );
}
