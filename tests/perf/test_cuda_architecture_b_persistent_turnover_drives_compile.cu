#include "../../engine/perf/cuda_architecture_b_persistent_turnover_drives.cuh"

#include <type_traits>

namespace turnover_drives =
    eiel::cuda_architecture_b::persistent_turnover_drives;

using DriveState = turnover_drives::FlatTurnoverDriveState;
using DrivePlan = turnover_drives::FlatTurnoverDrivePlan;
using DriveCursor = turnover_drives::FlatPopulationPcg64Cursor;
using DriveOutputs = turnover_drives::FlatTurnoverDriveOutputs;
using ChemistryState = turnover_drives::FlatTurnoverChemistryState;
using ChemistryOutputs = turnover_drives::FlatTurnoverChemistryOutputs;

static_assert(turnover_drives::kStatusOk == 0);
static_assert(turnover_drives::kStatusInvalidBirthPlan == 1);
static_assert(turnover_drives::kStatusInvalidParentOrder == 2);
static_assert(turnover_drives::kStatusChildIdentityNotCommitted == 3);
static_assert(turnover_drives::kStatusUnsupportedGru == 4);
static_assert(turnover_drives::kStatusInvalidChildScalarState == 5);
static_assert(turnover_drives::kStatusInvalidPcg64State == 6);
static_assert(turnover_drives::kActionCount == 7);

static_assert(std::is_trivially_copyable_v<DriveState>);
static_assert(std::is_trivially_copyable_v<DrivePlan>);
static_assert(std::is_trivially_copyable_v<DriveCursor>);
static_assert(std::is_trivially_copyable_v<DriveOutputs>);
static_assert(std::is_trivially_copyable_v<ChemistryState>);
static_assert(std::is_trivially_copyable_v<ChemistryOutputs>);

static_assert(
    std::is_same_v<decltype(DriveState::slot_to_agent_id), const int64_t*>
);
static_assert(
    std::is_same_v<decltype(DriveState::policy_energy_observation), const double*>
);
static_assert(
    std::is_same_v<decltype(DrivePlan::child_slots), const int64_t*>
);
static_assert(std::is_same_v<decltype(DriveCursor::state_words), uint64_t*>);
static_assert(std::is_same_v<decltype(DriveCursor::buffer_words), int64_t*>);
static_assert(
    std::is_same_v<decltype(DriveCursor::mutation_np_semantic_draws), int64_t*>
);
static_assert(std::is_same_v<decltype(DriveOutputs::drives), float*>);
static_assert(
    std::is_same_v<decltype(DriveOutputs::birth_reset_shock), float*>
);
static_assert(std::is_same_v<decltype(ChemistryState::chemistry), float*>);

__global__ void compile_persistent_turnover_drives_probe(
    const DriveState state,
    const DrivePlan plan,
    const DriveCursor cursor,
    const DriveOutputs outputs,
    const turnover_drives::TurnoverDriveConfig config
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    turnover_drives::initialize_turnover_newborn_drives_serial(
        state, plan, cursor, outputs, config
    );
}

__global__ void compile_persistent_turnover_chemistry_probe(
    const ChemistryState state,
    const DrivePlan plan,
    const ChemistryOutputs outputs,
    const turnover_drives::TurnoverChemistryConfig config
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    turnover_drives::initialize_turnover_newborn_chemistry_serial(
        state, plan, outputs, config
    );
}
