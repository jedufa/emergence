#include "../../engine/perf/cuda_architecture_b_flattened_birth_scratch_v2.cuh"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace scratch =
    eiel::cuda_architecture_b::flattened_birth_scratch_v2;

constexpr bool segments_are_aligned_and_disjoint(
    const scratch::FlatBirthScratchLayoutV2 layout
) {
    if (!layout.ok() || layout.required_bytes % scratch::kScratchArenaAlignment != 0) {
        return false;
    }
    int64_t previous_end = 0;
    for (int32_t index = 0; index < scratch::kBirthScratchSegmentCount; ++index) {
        const auto segment = layout.segments[index];
        if (
            segment.offset % scratch::kScratchSegmentAlignment != 0
            || segment.offset < previous_end
            || segment.bytes < 0
            || segment.offset > layout.required_bytes - segment.bytes
        ) {
            return false;
        }
        previous_end = segment.offset + segment.bytes;
    }
    return previous_end <= layout.required_bytes;
}

constexpr bool required_bytes_are_monotonic() {
    int64_t previous = 0;
    for (int64_t births = 0; births <= 64; ++births) {
        const auto layout = scratch::make_layout(births, 64, 2048, 32, 5056);
        if (!segments_are_aligned_and_disjoint(layout)) {
            return false;
        }
        if (births != 0 && layout.required_bytes < previous) {
            return false;
        }
        previous = layout.required_bytes;
    }
    return true;
}

constexpr auto kB0 = scratch::make_layout(0, 0, 2048, 32, 5056);
constexpr auto kB1 = scratch::make_layout(1, 64, 2048, 32, 5056);
constexpr auto kB2 = scratch::make_layout(2, 64, 2048, 32, 5056);
constexpr auto kB3 = scratch::make_layout(3, 64, 2048, 32, 5056);
constexpr auto kB64 = scratch::make_layout(64, 64, 2048, 32, 5056);

constexpr scratch::BirthScratchSegmentLayoutV2 kPatchP2B64Segments[] = {
    {0, 1536},       // kMiniSlotRows
    {1536, 192},     // kMiniAliveRows
    {1728, 6144},    // kMiniGenomeInt32Planes
    {7872, 1536},    // kMiniPolicyMemoryDimensionRows
    {9408, 192},     // kMiniPolicyUseGruRows
    {9600, 18432},   // kMiniGenomeFloatPlanes
    {28032, 1536},   // kMiniEnergyRows
    {29568, 512},    // kParentPlanRows
    {30080, 512},    // kMatePlanRows
    {30592, 512},    // kChildPlanRows
    {31104, 512},    // kIdentityParentSlots
    {31616, 512},    // kIdentityChildSlots
    {32128, 512},    // kIdentityChildIds
    {32640, 512},    // kIdentityChildEnergy
    {33152, 8},      // kIdentityCount
    {33168, 8},      // kQueueHead
    {33184, 8},      // kQueueCount
    {33200, 512},    // kParentAgentIds
    {33712, 256},    // kChildLineageFounderIds
    {33968, 256},    // kChildLineageDepths
    {34224, 2048},   // kGenomeInt32Planes
    {36272, 512},    // kPolicyMemoryDimensionRows
    {36784, 64},     // kPolicyUseGruRows
    {36848, 6144},   // kGenomeFloatPlanes
    {42992, 1024},   // kDriveRows
    {44016, 1536},   // kChemistryRows
    {45552, 256},    // kBirthResetShockRows
    {45808, 524288}, // kPolicyMlpRows
    {570096, 8192},  // kHmemRows
    {578288, 2496},  // kPythonRandomPostWords
    {580784, 2496},  // kPythonRandomTrialWords
    {583280, 80},    // kPythonRandomSnapshots
    {583360, 48},    // kPlannerPcg64Snapshot
    {583408, 48},    // kRunnerPcg64Snapshot
    {583456, 48},    // kPopulationPcg64Snapshot
    {583504, 5056},  // kTorchState
    {588560, 64},    // kMutationFlags
    {588624, 256},   // kMutationDeltas
    {588880, 448},   // kLinkedDonorOrdinals
    {589328, 16384}, // kDiploidStagedChildren
    {605712, 62464}, // kGenomeUpgradeV2StagedChildren
    {668176, 13312}, // kEventStagingRows
    {681488, 4},     // kGenomeUpgradeV2QualificationFaultStage
    {681504, 176},   // kScalarStatuses
};

constexpr bool patch_p2_b64_segment_map_is_exact() {
    if (
        sizeof(kPatchP2B64Segments) / sizeof(kPatchP2B64Segments[0])
        != scratch::kBirthScratchSegmentCount
    ) {
        return false;
    }
    for (int32_t index = 0; index < scratch::kBirthScratchSegmentCount; ++index) {
        if (
            kB64.segments[index].offset != kPatchP2B64Segments[index].offset
            || kB64.segments[index].bytes != kPatchP2B64Segments[index].bytes
        ) {
            return false;
        }
    }
    return kB64.required_bytes == 681728;
}

static_assert(segments_are_aligned_and_disjoint(kB0));
static_assert(segments_are_aligned_and_disjoint(kB1));
static_assert(segments_are_aligned_and_disjoint(kB2));
static_assert(segments_are_aligned_and_disjoint(kB3));
static_assert(segments_are_aligned_and_disjoint(kB64));
static_assert(required_bytes_are_monotonic());
static_assert(kB0.required_bytes <= kB1.required_bytes);
static_assert(kB1.required_bytes <= kB2.required_bytes);
static_assert(kB2.required_bytes <= kB3.required_bytes);
static_assert(kB3.required_bytes <= kB64.required_bytes);
static_assert(kB0.required_bytes == 10752);
static_assert(kB1.required_bytes == 27904);
static_assert(kB2.required_bytes == 38144);
static_assert(kB3.required_bytes == 48640);
static_assert(kB64.required_bytes == 681728);
static_assert(kB0.mini_state_rows == 0);
static_assert(kB1.mini_state_rows == 3);
static_assert(kB2.mini_state_rows == 6);
static_assert(kB3.mini_state_rows == 9);
static_assert(kB64.mini_state_rows == 192);
static_assert(kB2.mini_row(scratch::kMiniParent, 0) == 0);
static_assert(kB2.mini_row(scratch::kMiniMate, 0) == 1);
static_assert(kB2.mini_row(scratch::kMiniChild, 0) == 2);
static_assert(kB2.mini_row(scratch::kMiniParent, 1) == 3);
static_assert(kB2.mini_row(scratch::kMiniMate, 1) == 4);
static_assert(kB2.mini_row(scratch::kMiniChild, 1) == 5);
static_assert(kB2.mini_plane_element(1, scratch::kMiniParent, 0) == 6);
static_assert(kB2.mini_plane_element(1, scratch::kMiniChild, 1) == 11);
static_assert(kB2.child_plane_element(1, 0) == 2);
static_assert(kB2.child_plane_element(1, 1) == 3);
static_assert(kB2.segment(scratch::kMiniSlotRows).bytes == 3 * 2 * 8);
static_assert(kB2.segment(scratch::kMiniAliveRows).bytes == 3 * 2);
static_assert(
    kB2.segment(scratch::kMiniGenomeInt32Planes).bytes == 8 * 3 * 2 * 4
);
static_assert(
    kB2.segment(scratch::kMiniPolicyMemoryDimensionRows).bytes == 3 * 2 * 8
);
static_assert(kB2.segment(scratch::kMiniPolicyUseGruRows).bytes == 3 * 2);
static_assert(
    kB2.segment(scratch::kMiniGenomeFloatPlanes).bytes == 24 * 3 * 2 * 4
);
static_assert(kB2.segment(scratch::kMiniEnergyRows).bytes == 3 * 2 * 8);
static_assert(kB2.segment(scratch::kParentPlanRows).bytes == 2 * 8);
static_assert(kB2.segment(scratch::kMatePlanRows).bytes == 2 * 8);
static_assert(kB2.segment(scratch::kChildPlanRows).bytes == 2 * 8);
static_assert(kB2.segment(scratch::kIdentityParentSlots).bytes == 2 * 8);
static_assert(kB2.segment(scratch::kIdentityChildSlots).bytes == 2 * 8);
static_assert(kB2.segment(scratch::kIdentityChildIds).bytes == 2 * 8);
static_assert(kB2.segment(scratch::kIdentityChildEnergy).bytes == 2 * 8);
static_assert(kB2.segment(scratch::kIdentityCount).bytes == 8);
static_assert(kB2.segment(scratch::kQueueHead).bytes == 8);
static_assert(kB2.segment(scratch::kQueueCount).bytes == 8);
static_assert(kB2.segment(scratch::kParentAgentIds).bytes == 2 * 8);
static_assert(kB2.segment(scratch::kChildLineageFounderIds).bytes == 2 * 4);
static_assert(kB2.segment(scratch::kChildLineageDepths).bytes == 2 * 4);
static_assert(kB2.segment(scratch::kGenomeInt32Planes).bytes == 2 * 8 * 4);
static_assert(
    kB2.segment(scratch::kPolicyMemoryDimensionRows).bytes == 2 * 8
);
static_assert(kB2.segment(scratch::kPolicyUseGruRows).bytes == 2);
static_assert(kB2.segment(scratch::kGenomeFloatPlanes).bytes == 2 * 24 * 4);
static_assert(kB2.segment(scratch::kDriveRows).bytes == 2 * 4 * 4);
static_assert(kB2.segment(scratch::kChemistryRows).bytes == 2 * 6 * 4);
static_assert(kB2.segment(scratch::kBirthResetShockRows).bytes == 2 * 4);
static_assert(kB2.segment(scratch::kPolicyMlpRows).bytes == 2 * 2048 * 4);
static_assert(kB2.segment(scratch::kHmemRows).bytes == 2 * 32 * 4);
static_assert(kB2.segment(scratch::kPythonRandomPostWords).bytes == 624 * 4);
static_assert(kB2.segment(scratch::kPythonRandomTrialWords).bytes == 624 * 4);
static_assert(kB2.segment(scratch::kPythonRandomSnapshots).bytes == 2 * 40);
static_assert(kB2.segment(scratch::kPlannerPcg64Snapshot).bytes == 48);
static_assert(kB2.segment(scratch::kRunnerPcg64Snapshot).bytes == 48);
static_assert(kB2.segment(scratch::kPopulationPcg64Snapshot).bytes == 48);
static_assert(kB2.segment(scratch::kTorchState).bytes == 5056);
static_assert(kB2.segment(scratch::kMutationFlags).bytes == 2);
static_assert(kB2.segment(scratch::kMutationDeltas).bytes == 2 * 4);
static_assert(kB2.segment(scratch::kLinkedDonorOrdinals).bytes == 2 * 7);
static_assert(kB2.segment(scratch::kDiploidStagedChildren).bytes == 2 * 256);
static_assert(
    kB2.segment(scratch::kGenomeUpgradeV2StagedChildren).bytes == 2 * 976
);
static_assert(
    kB2.segment(scratch::kEventStagingRows).bytes == (64 + 2) * 104
);
static_assert(
    kB2.segment(scratch::kGenomeUpgradeV2QualificationFaultStage).bytes == 4
);
static_assert(kB2.segment(scratch::kScalarStatuses).bytes == 176);
static_assert(
    offsetof(scratch::BirthScalarStatusesV2, population_np_semantic_draws) == 72
);
static_assert(
    offsetof(scratch::BirthScalarStatusesV2, policy_birth_tape_cursor) == 120
);
static_assert(
    offsetof(scratch::BirthScalarStatusesV2, required_event_rows) == 128
);
static_assert(offsetof(scratch::BirthScalarStatusesV2, event_status) == 168);
static_assert(
    offsetof(scratch::BirthScalarStatusesV2, transaction_status) == 172
);
static_assert(patch_p2_b64_segment_map_is_exact());

static_assert(
    scratch::make_layout(-1, 64, 1, 1, 1).status
    == scratch::kLayoutStatusNegativeConfiguration
);
static_assert(scratch::make_layout(-1, 64, 1, 1, 1).required_bytes == 0);
static_assert(
    scratch::make_layout(1, -1, 1, 1, 1).status
    == scratch::kLayoutStatusNegativeConfiguration
);
static_assert(
    scratch::make_layout(1, 64, -1, 1, 1).status
    == scratch::kLayoutStatusNegativeConfiguration
);
static_assert(
    scratch::make_layout(1, 64, 1, -1, 1).status
    == scratch::kLayoutStatusNegativeConfiguration
);
static_assert(
    scratch::make_layout(1, 64, 1, 1, -1).status
    == scratch::kLayoutStatusNegativeConfiguration
);
static_assert(
    scratch::make_layout(65, 64, 1, 1, 1).status
    == scratch::kLayoutStatusBirthCountExceedsCapacity
);
static_assert(
    scratch::make_layout(scratch::kInt64Max, scratch::kInt64Max, 0, 0, 0).status
    == scratch::kLayoutStatusInt64Overflow
);
static_assert(
    scratch::make_layout(1, 1, scratch::kInt64Max, 0, 0).status
    == scratch::kLayoutStatusInt64Overflow
);
static_assert(
    scratch::make_layout(1, 1, scratch::kInt64Max, 0, 0).required_bytes == 0
);
static_assert(
    scratch::make_layout(0, 0, 0, 0, scratch::kInt64Max).status
    == scratch::kLayoutStatusInt64Overflow
);

int main() {
    for (const int64_t births : {0, 1, 2, 3, 64}) {
        const auto layout = scratch::make_layout(births, 64, 2048, 32, 5056);
        assert(segments_are_aligned_and_disjoint(layout));
        assert(layout.required_bytes > 0);
        assert(layout.mini_state_rows == 3 * births);
    }

    const auto layout = scratch::make_layout(3, 64, 2048, 32, 5056);
    std::vector<uint8_t> birth_storage(
        static_cast<std::size_t>(layout.required_bytes * 2)
    );
    std::vector<uint8_t> turnover_storage(65536 * 2);
    std::vector<uint8_t> resident_noise_storage(4096 * 2);
    scratch::FlatBirthScratchArenaV2 arena{
        birth_storage.data(),
        layout.required_bytes,
    };
    assert(arena.holds(layout));
    assert(arena.at<int64_t>(-1, layout, scratch::kParentPlanRows) == nullptr);
    assert(
        arena.at<int64_t>(scratch::kInt64Max, layout, scratch::kParentPlanRows)
        == nullptr
    );
    scratch::FlatBirthScratchArenaV2 misaligned_stride{
        birth_storage.data(),
        layout.required_bytes + scratch::kScratchSegmentAlignment,
    };
    assert(!misaligned_stride.holds(layout));
    scratch::FlatBirthScratchArenaV2 undersized{
        birth_storage.data(),
        layout.required_bytes - scratch::kScratchArenaAlignment,
    };
    assert(!undersized.holds(layout));
    assert(arena.at<int64_t>(0, layout, scratch::kParentPlanRows)
        != reinterpret_cast<int64_t*>(turnover_storage.data()));
    assert(arena.at<int64_t>(0, layout, scratch::kParentPlanRows)
        != reinterpret_cast<int64_t*>(resident_noise_storage.data()));
    assert(
        reinterpret_cast<uint8_t*>(
            arena.at<float>(1, layout, scratch::kPolicyMlpRows)
        )
        == birth_storage.data() + layout.required_bytes
            + layout.segment(scratch::kPolicyMlpRows).offset
    );
    return 0;
}
