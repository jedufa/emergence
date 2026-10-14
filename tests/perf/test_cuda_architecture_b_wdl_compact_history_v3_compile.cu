#include "../../engine/perf/cuda_architecture_b_flattened_wdl_controller_v2.hpp"

#include <type_traits>

namespace wdl =
    eiel::cuda_architecture_b::flattened_wdl_controller_v2;

static_assert(wdl::kAbiVersionV3 == 3);
static_assert(wdl::kWdlAgentFlagCuedV3 == 1U);
static_assert(wdl::kWdlAgentFlagDamagedV3 == 2U);
static_assert(wdl::kWdlAgentFlagMaskV3 == 3U);
static_assert(wdl::kWdlAgentHistoryFullStatusV3 == 250);
static_assert(wdl::kWdlLineageHistoryFullStatusV3 == 251);
static_assert(std::is_trivially_copyable<wdl::FlatWdlContextV2>::value);

__global__ void compile_wdl_compact_history_v3_probe(
    const wdl::FlatWdlContextV2 context
) {
    if (threadIdx.x == 0 && context.enabled && context.history_capacity > 0) {
        context.agent_history_keys[0] = -1;
        context.agent_history_flags[0] = 0U;
        context.lineage_history_keys[0] = -1;
        context.lineage_history_damage[0] = 0.0;
    }
}
