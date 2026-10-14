#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace eiel::cuda_architecture_b::flattened_birth_scratch_v1 {

constexpr int64_t kBytesPerCell = 163840;
constexpr int64_t kDynamicOffset = 8192;
// Turnover owns [0, 132096). Resident tick noise uses a separate scratch tail so
// generated values for missing newborn tape rows never mutate the reusable tape.
constexpr int64_t kResidentDriveNoiseOffset = 132096;

constexpr int64_t kPlannerPcgState = 0;
constexpr int64_t kPlannerPcgBuffer = 32;
constexpr int64_t kPlannerPolicyDraws = 48;
constexpr int64_t kRunnerPcgState = 56;
constexpr int64_t kRunnerPcgBuffer = 88;
constexpr int64_t kRunnerMetricDraws = 104;

constexpr int64_t kPythonPostWords = 128;
constexpr int64_t kPythonTrialWords = 2624;
constexpr int64_t kPythonIndex = 5120;
constexpr int64_t kPythonHasGauss = 5128;
constexpr int64_t kPythonGaussNext = 5136;
constexpr int64_t kPythonRawDraws = 5144;
constexpr int64_t kPythonSemanticDraws = 5152;

constexpr int64_t kPopulationPcgState = 5160;
constexpr int64_t kPopulationPcgBuffer = 5192;
constexpr int64_t kPopulationSemanticDraws = 5208;

constexpr int64_t kMiniSlotIds = 5216;
constexpr int64_t kMiniAlive = 5232;
constexpr int64_t kMiniGenomeInt32 = 5240;
constexpr int64_t kMiniPolicyMemDim = 5304;
constexpr int64_t kMiniPolicyUseGru = 5320;
constexpr int64_t kMiniGenomeFloat32 = 5328;
constexpr int64_t kMiniPlanParent = 5520;
constexpr int64_t kMiniPlanChild = 5528;
constexpr int64_t kMiniPlanCount = 5536;
constexpr int64_t kMiniScalarCommitted = 5544;
constexpr int64_t kMiniScalarSemanticDraws = 5552;
constexpr int64_t kMiniScalarStatus = 5560;
constexpr int64_t kMiniEnergy = 5568;
constexpr int64_t kMiniDrives = 5584;
constexpr int64_t kMiniChemistry = 5648;
constexpr int64_t kMiniBirthShock = 5712;
constexpr int64_t kMiniDriveCommitted = 5720;
constexpr int64_t kMiniBirthNoiseRows = 5728;
constexpr int64_t kMiniExpPriorRows = 5736;
constexpr int64_t kMiniStandardNormalRows = 5744;
constexpr int64_t kMiniDriveSemanticDraws = 5752;
constexpr int64_t kMiniDriveStatus = 5760;
constexpr int64_t kMiniChemistryStatus = 5764;

constexpr int64_t kTransactionStatus = 5792;
constexpr int64_t kMiniPolicyCommitted = 5800;
constexpr int64_t kMiniPolicyStateSemanticDraws = 5808;
constexpr int64_t kMiniPolicySemanticDraws = 5816;
constexpr int64_t kMiniPolicyMutationEvents = 5824;
constexpr int64_t kMiniPolicyMutatedElements = 5832;
constexpr int64_t kMiniPolicyMutationDeltaTotal = 5840;
constexpr int64_t kMiniPolicyMutationOccurred = 5848;
constexpr int64_t kMiniPolicyMutationDelta = 5852;
constexpr int64_t kMiniPolicyStatus = 5856;
constexpr int64_t kMiniPolicyCursor = 5864;
constexpr int64_t kRequiredEventRows = 5872;
constexpr int64_t kPreparedParentSlot = 5880;
constexpr int64_t kPreparedChildSlot = 5888;
constexpr int64_t kPreparedChildId = 5896;
constexpr int64_t kPreparedStamp = 5904;

// Failure-atomic B=2 staging. These fixed rows hold only the two births in the
// current tick and are reused by every subsequent tick.
constexpr int64_t kB2IdentityParentSlots = 5920;
constexpr int64_t kB2IdentityChildSlots = 5936;
constexpr int64_t kB2IdentityChildIds = 5952;
constexpr int64_t kB2IdentityChildEnergy = 5968;
constexpr int64_t kB2IdentityCount = 5984;
constexpr int64_t kB2QueueHead = 5992;
constexpr int64_t kB2QueueCount = 6000;
constexpr int64_t kB2MiniSlotIds = 6008;
constexpr int64_t kB2MiniAlive = 6040;
constexpr int64_t kB2MiniGenomeInt32 = 6048;
constexpr int64_t kB2MiniPolicyMemDim = 6176;
constexpr int64_t kB2MiniPolicyUseGru = 6208;
constexpr int64_t kB2MiniGenomeFloat32 = 6224;
constexpr int64_t kB2MiniPlanParent = 6608;
constexpr int64_t kB2MiniPlanChild = 6624;
constexpr int64_t kB2MiniPlanCount = 6640;
constexpr int64_t kB2ScalarGenomeInt32 = 6648;
constexpr int64_t kB2ScalarPolicyMemDim = 6712;
constexpr int64_t kB2ScalarPolicyUseGru = 6728;
constexpr int64_t kB2ScalarGenomeFloat32 = 6736;
constexpr int64_t kB2ChildLineageFounder = 6928;
constexpr int64_t kB2ChildLineageDepth = 6936;
constexpr int64_t kB2EventStatus = 6944;
constexpr int64_t kB2PolicyMutationOccurred = 6952;
constexpr int64_t kB2PolicyMutationDelta = 6960;
constexpr int64_t kB2MiniEnergy = 6976;
constexpr int64_t kB2ParentAgentIds = 7008;

constexpr int64_t align_up(const int64_t value, const int64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

struct DynamicLayout {
    int64_t mlp_rows;
    int64_t hmem_rows;
    int64_t torch_state;
    int64_t required_bytes;
};

struct B2DynamicLayout {
    int64_t child_mlp_rows;
    int64_t child_hmem_rows;
    int64_t torch_state;
    int64_t required_bytes;
};

struct ResidentDriveNoiseLayout {
    int64_t samples;
    int64_t valid;
    int64_t required_bytes;
};

constexpr ResidentDriveNoiseLayout make_resident_drive_noise_layout(
    const int64_t capacity
) {
    const int64_t samples = kResidentDriveNoiseOffset;
    const int64_t valid = align_up(
        samples + capacity * 4 * static_cast<int64_t>(sizeof(float)),
        16
    );
    return {
        samples,
        valid,
        align_up(valid + capacity * static_cast<int64_t>(sizeof(bool)), 256),
    };
}

constexpr DynamicLayout make_dynamic_layout(
    const int64_t packed_width,
    const int64_t hmem_width,
    const int64_t torch_state_bytes
) {
    const int64_t mlp_rows = kDynamicOffset;
    const int64_t hmem_rows = align_up(
        mlp_rows + 2 * packed_width * static_cast<int64_t>(sizeof(float)),
        16
    );
    const int64_t torch_state = align_up(
        hmem_rows + 2 * hmem_width * static_cast<int64_t>(sizeof(float)),
        16
    );
    return {
        mlp_rows,
        hmem_rows,
        torch_state,
        align_up(torch_state + torch_state_bytes, 256),
    };
}

constexpr B2DynamicLayout make_b2_dynamic_layout(
    const int64_t packed_width,
    const int64_t hmem_width,
    const int64_t torch_state_bytes
) {
    const int64_t child_mlp_rows = kDynamicOffset;
    const int64_t child_hmem_rows = align_up(
        child_mlp_rows + 2 * packed_width * static_cast<int64_t>(sizeof(float)),
        16
    );
    const int64_t torch_state = align_up(
        child_hmem_rows + 2 * hmem_width * static_cast<int64_t>(sizeof(float)),
        16
    );
    return {
        child_mlp_rows,
        child_hmem_rows,
        torch_state,
        align_up(torch_state + torch_state_bytes, 256),
    };
}

template <typename T, typename Scratch>
__device__ __forceinline__ T* at(
    const Scratch scratch,
    const int64_t cell,
    const int64_t byte_offset
) {
    return reinterpret_cast<T*>(
        scratch.storage + cell * scratch.bytes_per_cell + byte_offset
    );
}

static_assert(kPythonPostWords + 624 * 4 == kPythonTrialWords);
static_assert(kPythonTrialWords + 624 * 4 <= kPythonIndex);
static_assert(kMiniPolicyStateSemanticDraws % alignof(int64_t) == 0);
static_assert(kMiniPolicyCursor % alignof(int64_t) == 0);
static_assert(kPreparedStamp + static_cast<int64_t>(sizeof(int64_t)) <= kDynamicOffset);
static_assert(kB2IdentityParentSlots % alignof(int64_t) == 0);
static_assert(kB2ScalarGenomeFloat32 % alignof(float) == 0);
static_assert(kB2ParentAgentIds + 2 * static_cast<int64_t>(sizeof(int64_t)) <= kDynamicOffset);
static_assert(kResidentDriveNoiseOffset >= 132096);

}  // namespace eiel::cuda_architecture_b::flattened_birth_scratch_v1
