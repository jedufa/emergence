#include "../../engine/perf/cuda_architecture_b_persistent_lifecycle_events.cuh"

#include <cstdint>

namespace lifecycle_events =
    eiel::cuda_architecture_b::persistent_lifecycle_events;

__global__ void compile_persistent_lifecycle_events_probe(
    lifecycle_events::FlatEventTapeColumns columns,
    lifecycle_events::FlatEventTapeState tape,
    lifecycle_events::FlatDeathRows action_deaths,
    lifecycle_events::FlatDeathRows residual_deaths,
    lifecycle_events::FlatLineageRows lineage,
    lifecycle_events::FlatActionTerminalReporting reporting,
    const int64_t generation,
    const int64_t absolute_tick
) {
    if (threadIdx.x != 0) {
        return;
    }
    lifecycle_events::publish_persistent_lifecycle_death_boundary(
        columns,
        tape,
        action_deaths,
        residual_deaths,
        lineage,
        reporting,
        static_cast<int64_t>(blockIdx.x),
        generation,
        absolute_tick
    );
}
