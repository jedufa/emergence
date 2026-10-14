#include "../../engine/perf/cuda_architecture_b_persistent_birth_events.cuh"

#include <type_traits>

namespace birth_events =
    eiel::cuda_architecture_b::persistent_birth_events;

static_assert(birth_events::kStatusOk == 0);
static_assert(birth_events::kStatusCapacityExceeded == 1);
static_assert(birth_events::kStatusInvalidSemantics == 2);
static_assert(birth_events::kBirthCommittedEventType == 1);
static_assert(birth_events::kMutationEventType == 2);
static_assert(birth_events::kFailedBirthEventType == 3);

using Identity = birth_events::FlatOrdinaryBirthEventIdentity;
using Decisions = birth_events::FlatOrdinaryBirthDecisionRows;
using Commits = birth_events::FlatOrdinaryBirthCommitRows;

static_assert(std::is_same_v<
    decltype(Identity::slot_to_agent_id),
    birth_events::FlatStridedCellRows<const int64_t>
>);
static_assert(std::is_same_v<
    decltype(Identity::lineage_founder_id),
    birth_events::FlatStridedCellRows<const int32_t>
>);
static_assert(std::is_same_v<
    decltype(Decisions::reason_codes),
    birth_events::FlatStridedCellRows<const int32_t>
>);
static_assert(std::is_same_v<
    decltype(Commits::mutation_occurred),
    birth_events::FlatStridedCellRows<const bool>
>);
static_assert(std::is_same_v<
    decltype(Commits::child_energy),
    birth_events::FlatStridedCellRows<const float>
>);

__global__ void compile_persistent_birth_events_probe(
    const birth_events::FlatEventTapeColumns columns,
    const birth_events::FlatEventTapeState tape,
    const Identity identity,
    const Decisions decisions,
    const Commits commits,
    const int64_t generation,
    const int64_t tick
) {
    if (threadIdx.x != 0) {
        return;
    }
    birth_events::append_ordinary_birth_events_serial(
        columns,
        tape,
        identity,
        decisions,
        commits,
        static_cast<int64_t>(blockIdx.x),
        generation,
        tick
    );
}
