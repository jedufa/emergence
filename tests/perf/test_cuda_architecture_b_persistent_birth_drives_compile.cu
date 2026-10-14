#include "../../engine/perf/cuda_architecture_b_persistent_birth_drives.cuh"

#include <type_traits>

namespace birth_drives =
    eiel::cuda_architecture_b::persistent_birth_drives;

using BirthState = birth_drives::FlatOrdinaryBirthDriveState;
using BirthCursor = birth_drives::FlatPopulationPcg64Cursor;
using BirthOutputs = birth_drives::FlatOrdinaryBirthDriveOutputs;
using ChemistryState = birth_drives::FlatNewbornChemistryState;

static_assert(birth_drives::kStatusOk == 0);
static_assert(birth_drives::kStatusInvalidBirthPlan == 1);
static_assert(birth_drives::kStatusInvalidParentOrder == 2);
static_assert(birth_drives::kStatusChildIdentityNotCommitted == 3);
static_assert(birth_drives::kStatusUnsupportedGru == 4);
static_assert(birth_drives::kStatusInvalidChildScalarState == 5);
static_assert(birth_drives::kStatusInvalidPcg64State == 6);
static_assert(birth_drives::kActionCount == 7);

static_assert(
    std::is_same_v<decltype(BirthState::slot_to_agent_id), const int64_t*>
);
static_assert(
    std::is_same_v<decltype(BirthState::policy_energy_observation), const double*>
);
static_assert(
    std::is_same_v<decltype(BirthState::genome_drive_birth_noise), const float*>
);
static_assert(std::is_same_v<decltype(BirthCursor::state_words), uint64_t*>);
static_assert(std::is_same_v<decltype(BirthCursor::buffer_words), int64_t*>);
static_assert(
    std::is_same_v<
        decltype(BirthCursor::mutation_np_semantic_draws),
        int64_t*
    >
);
static_assert(std::is_same_v<decltype(BirthOutputs::drives), float*>);
static_assert(
    std::is_same_v<decltype(BirthOutputs::birth_reset_shock), float*>
);
static_assert(std::is_same_v<decltype(ChemistryState::chemistry), float*>);

__global__ void compile_persistent_birth_drives_probe(
    const birth_drives::FlatOrdinaryBirthDriveState state,
    const birth_drives::FlatOrdinaryBirthPlan plan,
    const birth_drives::FlatPopulationPcg64Cursor cursor,
    const birth_drives::FlatOrdinaryBirthDriveOutputs outputs,
    const birth_drives::OrdinaryBirthDriveConfig config
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    birth_drives::initialize_ordinary_newborn_drives_serial(
        state,
        plan,
        cursor,
        outputs,
        config
    );
}

__global__ void compile_persistent_newborn_chemistry_probe(
    const birth_drives::FlatNewbornChemistryState state,
    const birth_drives::FlatOrdinaryBirthPlan plan,
    const birth_drives::FlatNewbornChemistryOutputs outputs,
    const birth_drives::NewbornChemistryConfig config
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    birth_drives::initialize_newborn_chemistry_serial(
        state,
        plan,
        outputs,
        config
    );
}
