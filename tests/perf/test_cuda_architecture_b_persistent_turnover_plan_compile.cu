#include "../../engine/perf/cuda_architecture_b_persistent_turnover_plan.cuh"

#include <type_traits>

namespace turnover =
    eiel::cuda_architecture_b::persistent_turnover_plan;

using TurnoverState = turnover::FlatGenerationTurnoverState;
using TurnoverBuffers = turnover::FlatGenerationTurnoverBuffers;
using TurnoverConfig = turnover::GenerationTurnoverConfig;
using TurnoverSelectionConfig =
    turnover::GenerationTurnoverSelectionConfigV1;
using RefillState = turnover::FlatGenerationTurnoverRefillState;
using RefillBuffers = turnover::FlatGenerationTurnoverRefillBuffers;
using RefillConfig = turnover::GenerationTurnoverRefillConfig;

static_assert(std::is_trivially_copyable_v<TurnoverState>);
static_assert(std::is_trivially_copyable_v<TurnoverBuffers>);
static_assert(std::is_trivially_copyable_v<TurnoverConfig>);
static_assert(std::is_trivially_copyable_v<TurnoverSelectionConfig>);
static_assert(std::is_trivially_copyable_v<RefillState>);
static_assert(std::is_trivially_copyable_v<RefillBuffers>);
static_assert(std::is_trivially_copyable_v<RefillConfig>);

static_assert(turnover::kTurnoverStatusOk == 0);
static_assert(turnover::kTurnoverStatusInvalidConfig == 1);
static_assert(turnover::kTurnoverStatusInvalidResidentState == 2);
static_assert(turnover::kTurnoverStatusNonuniformEarlyLifeWeight == 3);
static_assert(turnover::kTurnoverStatusLowPopulationUnsupported == 4);
static_assert(turnover::kTurnoverStatusPostCullLowPopulationUnsupported == 5);
static_assert(turnover::kTurnoverStatusInsufficientParentPool == 6);
static_assert(turnover::kTurnoverSelectionFitnessV1 == 0);
static_assert(turnover::kTurnoverSelectionUniformNeutralV1 == 1);

static_assert(turnover::kRefillStatusOk == 0);
static_assert(turnover::kRefillStatusTurnoverFailed == 1);
static_assert(turnover::kRefillStatusInvalidState == 2);
static_assert(turnover::kRefillStatusInvalidParentPool == 3);
static_assert(turnover::kRefillStatusNoSpawnSpace == 4);

static_assert(turnover::kCpythonStateWordCount == 624);
static_assert(turnover::kMaximumParentPairAttempts == 1000000);
static_assert(turnover::kMaximumSpawnAttempts == 200);
static_assert(turnover::kMaximumGridDimension == 0x7fffffffLL);

static_assert(
    std::is_same_v<decltype(TurnoverState::slot_to_agent_id), const int64_t*>
);
static_assert(
    std::is_same_v<decltype(TurnoverState::energy_collected_exact), const double*>
);
static_assert(
    std::is_same_v<decltype(TurnoverState::early_life_weight), const float*>
);
static_assert(
    std::is_same_v<decltype(RefillState::python_random_words), uint32_t*>
);
static_assert(
    std::is_same_v<decltype(RefillState::python_random_trial_words), uint32_t*>
);
static_assert(
    std::is_same_v<decltype(RefillState::env_pcg64_state_words), uint64_t*>
);
static_assert(
    std::is_same_v<decltype(RefillState::turnover_status), const int32_t*>
);
static_assert(
    std::is_same_v<decltype(RefillState::env_pcg64_buffer_words), int64_t*>
);
static_assert(
    std::is_same_v<decltype(RefillBuffers::planned_spawn_positions), int32_t*>
);

__global__ void compile_persistent_turnover_plan_probe(
    const turnover::FlatGenerationTurnoverState turnover_state,
    const turnover::FlatGenerationTurnoverBuffers turnover_buffers,
    const turnover::GenerationTurnoverConfig turnover_config,
    const turnover::FlatGenerationTurnoverRefillState refill_state,
    const turnover::FlatGenerationTurnoverRefillBuffers refill_buffers,
    const turnover::GenerationTurnoverRefillConfig refill_config
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    turnover::plan_generation_turnover_serial(
        turnover_state,
        turnover_buffers,
        turnover_config
    );
    turnover::plan_generation_turnover_serial(
        turnover_state,
        turnover_buffers,
        turnover_config,
        TurnoverSelectionConfig{
            1,
            turnover::kTurnoverSelectionUniformNeutralV1,
            9101ULL,
            7,
        }
    );
    turnover::plan_generation_turnover_refill_serial(
        refill_state,
        refill_buffers,
        refill_config
    );
}
