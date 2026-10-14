#include "../../engine/perf/cuda_architecture_b_persistent_birth_action_history.cuh"

#include <type_traits>

namespace birth_action_history =
    eiel::cuda_architecture_b::persistent_birth_action_history;

using ActionHistoryState =
    birth_action_history::FlatOrdinaryBirthActionHistoryState;
using CommittedBirthSlots =
    birth_action_history::FlatCommittedOrdinaryBirthSlots;
using ActionHistoryOutputs =
    birth_action_history::FlatOrdinaryBirthActionHistoryOutputs;
using ActionHistoryConfig =
    birth_action_history::OrdinaryBirthActionHistoryConfig;

static_assert(birth_action_history::kStatusOk == 0);
static_assert(birth_action_history::kStatusIdentityNotCommitted == 1);
static_assert(birth_action_history::kStatusInvalidCommittedCount == 2);
static_assert(birth_action_history::kStatusInvalidChildSlot == 3);
static_assert(birth_action_history::kStatusDuplicateChildSlot == 4);

static_assert(std::is_standard_layout_v<ActionHistoryState>);
static_assert(std::is_trivially_copyable_v<ActionHistoryState>);
static_assert(sizeof(ActionHistoryState) == 6 * sizeof(int32_t*));
static_assert(std::is_standard_layout_v<CommittedBirthSlots>);
static_assert(std::is_trivially_copyable_v<CommittedBirthSlots>);
static_assert(std::is_standard_layout_v<ActionHistoryOutputs>);
static_assert(std::is_trivially_copyable_v<ActionHistoryOutputs>);
static_assert(std::is_standard_layout_v<ActionHistoryConfig>);
static_assert(std::is_trivially_copyable_v<ActionHistoryConfig>);
static_assert(
    std::is_same_v<decltype(ActionHistoryState::last_action), int32_t*>
);
static_assert(
    std::is_same_v<decltype(ActionHistoryState::last_last_action), int32_t*>
);
static_assert(
    std::is_same_v<decltype(ActionHistoryState::action_switches), int32_t*>
);
static_assert(
    std::is_same_v<decltype(ActionHistoryState::action_oscillations), int32_t*>
);
static_assert(
    std::is_same_v<decltype(ActionHistoryState::episode0_switches), int32_t*>
);
static_assert(
    std::is_same_v<decltype(ActionHistoryState::episode0_oscillations), int32_t*>
);
static_assert(
    std::is_same_v<decltype(CommittedBirthSlots::child_slots), const int64_t*>
);
static_assert(
    std::is_same_v<
        decltype(CommittedBirthSlots::committed_count),
        const int64_t*
    >
);
static_assert(
    std::is_same_v<decltype(CommittedBirthSlots::identity_status), const int32_t*>
);
static_assert(
    std::is_same_v<decltype(ActionHistoryOutputs::status), int32_t*>
);
static_assert(
    std::is_same_v<decltype(ActionHistoryConfig::plan_capacity), int64_t>
);
static_assert(
    std::is_same_v<decltype(ActionHistoryConfig::slot_count), int64_t>
);

__global__ void compile_persistent_birth_action_history_probe(
    const ActionHistoryState state,
    const CommittedBirthSlots committed,
    const ActionHistoryOutputs outputs,
    const ActionHistoryConfig config
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    birth_action_history::reset_ordinary_newborn_action_history_serial(
        state,
        committed,
        outputs,
        config
    );
}
