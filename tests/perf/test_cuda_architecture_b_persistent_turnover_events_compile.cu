#include "../../engine/perf/cuda_architecture_b_persistent_turnover_events.cuh"

#include <cstdint>
#include <type_traits>

namespace turnover_events =
    eiel::cuda_architecture_b::persistent_turnover_events;

static_assert(turnover_events::kStatusOk == 0);
static_assert(turnover_events::kStatusCapacityExceeded == 1);
static_assert(turnover_events::kStatusInvalidSemantics == 2);
static_assert(turnover_events::kBirthCommittedEventType == 1);
static_assert(turnover_events::kMutationEventType == 2);
static_assert(
    std::is_same_v<
        decltype(turnover_events::FlatTurnoverPostcommitRows::child_agent_ids),
        turnover_events::FlatStridedCellRows<const int64_t>
    >
);
static_assert(
    std::is_same_v<
        decltype(turnover_events::FlatTurnoverPostcommitState::active_agent_ids),
        turnover_events::FlatStridedCellRows<int64_t>
    >
);

__global__ void compile_persistent_turnover_events_probe(
    const turnover_events::FlatEventTapeColumns columns,
    const turnover_events::FlatEventTapeState tape,
    const turnover_events::FlatTurnoverPostcommitRows rows,
    const turnover_events::FlatTurnoverPostcommitState state,
    const turnover_events::TurnoverLifecycleConfig config,
    const int64_t generation,
    const int64_t tick
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    turnover_events::append_turnover_lifecycle_events_serial(
        columns,
        tape,
        rows,
        state,
        config,
        0,
        generation,
        tick
    );
}

__global__ void compile_persistent_turnover_active_postcommit_probe(
    const turnover_events::FlatTurnoverPostcommitState state,
    const turnover_events::FlatTurnoverPostcommitRows rows,
    const int64_t active_count
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    turnover_events::update_active_ids_from_turnover_postcommit_serial(
        state,
        rows,
        0,
        active_count
    );
}
