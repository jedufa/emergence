#pragma once

#include <cstdint>
#include <type_traits>

#include "cuda_diploid_scalar_genome_v1.cuh"
#include "genome_upgrade_v2_q10_v1.cuh"

#if defined(__CUDACC__)
#define EIEL_FLAT_BIRTH_SCRATCH_HD __host__ __device__
#else
#define EIEL_FLAT_BIRTH_SCRATCH_HD
#endif

namespace eiel::cuda_architecture_b::flattened_birth_scratch_v2 {

constexpr int64_t kScratchSegmentAlignment = 16;
constexpr int64_t kScratchArenaAlignment = 256;
constexpr int64_t kPythonRandomWordCount = 624;
constexpr int64_t kMiniStateRoleCount = 3;
constexpr int64_t kGenomeInt32PlaneCount = 8;
constexpr int64_t kGenomeFloatPlaneCount = 24;
constexpr int64_t kDriveWidth = 4;
constexpr int64_t kChemistryWidth = 6;
constexpr int64_t kPcg64StateWordCount = 4;
constexpr int64_t kPcg64BufferWordCount = 2;
constexpr int64_t kInt64Max = 9223372036854775807LL;

enum BirthScratchLayoutStatusV2 : int32_t {
    kLayoutStatusOk = 0,
    kLayoutStatusNegativeConfiguration = 1,
    kLayoutStatusBirthCountExceedsCapacity = 2,
    kLayoutStatusInt64Overflow = 3,
};

enum BirthScratchSegmentV2 : int32_t {
    kMiniSlotRows = 0,
    kMiniAliveRows,
    kMiniGenomeInt32Planes,
    kMiniPolicyMemoryDimensionRows,
    kMiniPolicyUseGruRows,
    kMiniGenomeFloatPlanes,
    kMiniEnergyRows,
    kParentPlanRows,
    kMatePlanRows,
    kChildPlanRows,
    kIdentityParentSlots,
    kIdentityChildSlots,
    kIdentityChildIds,
    kIdentityChildEnergy,
    kIdentityCount,
    kQueueHead,
    kQueueCount,
    kParentAgentIds,
    kChildLineageFounderIds,
    kChildLineageDepths,
    kGenomeInt32Planes,
    kPolicyMemoryDimensionRows,
    kPolicyUseGruRows,
    kGenomeFloatPlanes,
    kDriveRows,
    kChemistryRows,
    kBirthResetShockRows,
    kPolicyMlpRows,
    kHmemRows,
    kPythonRandomPostWords,
    kPythonRandomTrialWords,
    kPythonRandomSnapshots,
    kPlannerPcg64Snapshot,
    kRunnerPcg64Snapshot,
    kPopulationPcg64Snapshot,
    kTorchState,
    kMutationFlags,
    kMutationDeltas,
    kLinkedDonorOrdinals,
    kDiploidStagedChildren,
    kGenomeUpgradeV2StagedChildren,
    kEventStagingRows,
    kGenomeUpgradeV2QualificationFaultStage,
    kScalarStatuses,
    kBirthScratchSegmentCount,
};

enum GenomeUpgradeV2QualificationFaultStageV1 : int32_t {
    kGenomeUpgradeV2FaultDisabled = 0,
    kGenomeUpgradeV2FaultAfterDestinationReservation = 1,
    kGenomeUpgradeV2FaultAfterGenotypeStaging = 2,
    kGenomeUpgradeV2FaultAfterProductStaging = 3,
    kGenomeUpgradeV2FaultAfterPhenotypeStaging = 4,
    kGenomeUpgradeV2FaultAfterParentDebit = 5,
    kGenomeUpgradeV2FaultAfterChildEnergy = 6,
    kGenomeUpgradeV2FaultAfterLineageAllocation = 7,
    kGenomeUpgradeV2FaultAfterShadowWrite = 8,
    kGenomeUpgradeV2FaultImmediatelyBeforeCommit = 9,
    kGenomeUpgradeV2FaultImmediatelyAfterCommitBeforeReporting = 10,
    kGenomeUpgradeV2FaultTurnoverMidCull = 101,
    kGenomeUpgradeV2FaultTurnoverAfterDestinationAllocation = 102,
    kGenomeUpgradeV2FaultTurnoverAfterChildStaging = 103,
    kGenomeUpgradeV2FaultTurnoverAfterMutation = 104,
    kGenomeUpgradeV2FaultTurnoverAfterPhenotypeProjection = 105,
    kGenomeUpgradeV2FaultTurnoverBeforeCommit = 106,
    kGenomeUpgradeV2FaultTurnoverAfterCommitBeforeCheckpoint = 107,
};

enum MiniStateRoleV2 : int32_t {
    kMiniParent = 0,
    kMiniMate = 1,
    kMiniChild = 2,
};

struct PythonRandomSnapshotV2 {
    int64_t index;
    int64_t raw_draws;
    int64_t semantic_draws;
    double gauss_next;
    bool has_gauss;
    uint8_t reserved[7];
};

struct Pcg64SnapshotV2 {
    uint64_t state_words[kPcg64StateWordCount];
    int64_t buffer_words[kPcg64BufferWordCount];
};

struct BirthEventStagingRowV2 {
    int64_t generation;
    int64_t tick;
    int64_t sequence;
    int64_t actor_id;
    int64_t target_id;
    int64_t actor_slot;
    int64_t target_slot;
    int64_t parent_id;
    int64_t child_id;
    int32_t event_type;
    int32_t lineage_founder_id;
    int32_t lineage_depth;
    int32_t subtype;
    int32_t reason;
    float value;
    bool success;
    uint8_t reserved[7];
};

// One row per cell. Counters are int64_t so they can be accumulated without
// narrowing; phase outcomes retain the int32_t status ABI used by birth code.
struct BirthScalarStatusesV2 {
    int64_t planned_count;
    int64_t identity_committed_count;
    int64_t scalar_committed_count;
    int64_t scalar_semantic_draws;
    int64_t drive_committed_count;
    int64_t birth_noise_rows;
    int64_t exp_prior_rows;
    int64_t standard_normal_rows;
    int64_t drive_semantic_draws;
    int64_t population_np_semantic_draws;
    int64_t policy_committed_count;
    int64_t policy_state_semantic_draws;
    int64_t policy_semantic_draws;
    int64_t policy_mutation_events;
    int64_t policy_mutated_elements;
    int64_t policy_birth_tape_cursor;
    int64_t required_event_rows;
    double policy_mutation_delta_total;
    int32_t plan_status;
    int32_t identity_status;
    int32_t scalar_status;
    int32_t drive_status;
    int32_t chemistry_status;
    int32_t policy_status;
    int32_t event_status;
    int32_t transaction_status;
};

struct BirthScratchSegmentLayoutV2 {
    int64_t offset;
    int64_t bytes;
};

struct FlatBirthScratchLayoutV2 {
    int64_t birth_count;
    int64_t capacity;
    int64_t mini_state_rows;
    int64_t packed_width;
    int64_t hmem_width;
    int64_t torch_state_bytes;
    BirthScratchLayoutStatusV2 status;
    BirthScratchSegmentLayoutV2 segments[kBirthScratchSegmentCount];
    int64_t required_bytes;

    EIEL_FLAT_BIRTH_SCRATCH_HD constexpr bool ok() const {
        return status == kLayoutStatusOk;
    }

    EIEL_FLAT_BIRTH_SCRATCH_HD constexpr BirthScratchSegmentLayoutV2 segment(
        const BirthScratchSegmentV2 value
    ) const {
        return segments[static_cast<int32_t>(value)];
    }

    // Mini-state planes are column-major. Within a column, each birth owns one
    // contiguous parent/mate/child triplet.
    EIEL_FLAT_BIRTH_SCRATCH_HD constexpr int64_t mini_row(
        const MiniStateRoleV2 role,
        const int64_t birth
    ) const {
        return birth * kMiniStateRoleCount + static_cast<int64_t>(role);
    }

    EIEL_FLAT_BIRTH_SCRATCH_HD constexpr int64_t mini_plane_element(
        const int64_t column,
        const MiniStateRoleV2 role,
        const int64_t birth
    ) const {
        return column * mini_state_rows + mini_row(role, birth);
    }

    EIEL_FLAT_BIRTH_SCRATCH_HD constexpr int64_t child_plane_element(
        const int64_t column,
        const int64_t birth
    ) const {
        return column * birth_count + birth;
    }
};

EIEL_FLAT_BIRTH_SCRATCH_HD constexpr bool checked_multiply_nonnegative(
    const int64_t left,
    const int64_t right,
    int64_t& result
) {
    if (left < 0 || right < 0 || (right != 0 && left > kInt64Max / right)) {
        return false;
    }
    result = left * right;
    return true;
}

EIEL_FLAT_BIRTH_SCRATCH_HD constexpr bool checked_add_nonnegative(
    const int64_t left,
    const int64_t right,
    int64_t& result
) {
    if (left < 0 || right < 0 || left > kInt64Max - right) {
        return false;
    }
    result = left + right;
    return true;
}

EIEL_FLAT_BIRTH_SCRATCH_HD constexpr bool checked_align_up(
    const int64_t value,
    const int64_t alignment,
    int64_t& result
) {
    if (
        value < 0 || alignment <= 0
        || value > kInt64Max - (alignment - 1)
    ) {
        return false;
    }
    result = (value + alignment - 1) / alignment * alignment;
    return true;
}

EIEL_FLAT_BIRTH_SCRATCH_HD constexpr bool append_segment(
    FlatBirthScratchLayoutV2& layout,
    const BirthScratchSegmentV2 segment,
    const int64_t rows,
    const int64_t row_width,
    const int64_t element_bytes,
    int64_t& cursor
) {
    int64_t aligned_cursor = 0;
    int64_t elements = 0;
    int64_t bytes = 0;
    if (
        !checked_align_up(cursor, kScratchSegmentAlignment, aligned_cursor)
        || !checked_multiply_nonnegative(rows, row_width, elements)
        || !checked_multiply_nonnegative(elements, element_bytes, bytes)
        || aligned_cursor > kInt64Max - bytes
    ) {
        return false;
    }
    layout.segments[static_cast<int32_t>(segment)] = {aligned_cursor, bytes};
    cursor = aligned_cursor + bytes;
    return true;
}

// The returned extent is for one cell. The caller supplies the configured
// maximum birth count separately so a current B cannot silently exceed it.
// Every arithmetic operation that can affect an offset is checked in int64_t.
EIEL_FLAT_BIRTH_SCRATCH_HD constexpr FlatBirthScratchLayoutV2
make_flattened_birth_scratch_layout_v2(
    const int64_t birth_count,
    const int64_t capacity,
    const int64_t packed_width,
    const int64_t hmem_width,
    const int64_t torch_state_bytes
) {
    FlatBirthScratchLayoutV2 layout{};
    layout.birth_count = birth_count;
    layout.capacity = capacity;
    layout.packed_width = packed_width;
    layout.hmem_width = hmem_width;
    layout.torch_state_bytes = torch_state_bytes;
    if (
        birth_count < 0 || capacity < 0 || packed_width < 0
        || hmem_width < 0 || torch_state_bytes < 0
    ) {
        layout.status = kLayoutStatusNegativeConfiguration;
        return layout;
    }
    if (birth_count > capacity) {
        layout.status = kLayoutStatusBirthCountExceedsCapacity;
        return layout;
    }

    if (!checked_multiply_nonnegative(
            birth_count,
            kMiniStateRoleCount,
            layout.mini_state_rows
        )) {
        layout.status = kLayoutStatusInt64Overflow;
        return layout;
    }

    int64_t event_staging_rows = 0;
    if (!checked_add_nonnegative(capacity, birth_count, event_staging_rows)) {
        layout.status = kLayoutStatusInt64Overflow;
        return layout;
    }

    int64_t cursor = 0;
    const bool complete =
        append_segment(layout, kMiniSlotRows, layout.mini_state_rows, 1, sizeof(int64_t), cursor)
        && append_segment(layout, kMiniAliveRows, layout.mini_state_rows, 1, sizeof(bool), cursor)
        && append_segment(layout, kMiniGenomeInt32Planes, kGenomeInt32PlaneCount, layout.mini_state_rows, sizeof(int32_t), cursor)
        && append_segment(layout, kMiniPolicyMemoryDimensionRows, layout.mini_state_rows, 1, sizeof(int64_t), cursor)
        && append_segment(layout, kMiniPolicyUseGruRows, layout.mini_state_rows, 1, sizeof(bool), cursor)
        && append_segment(layout, kMiniGenomeFloatPlanes, kGenomeFloatPlaneCount, layout.mini_state_rows, sizeof(float), cursor)
        && append_segment(layout, kMiniEnergyRows, layout.mini_state_rows, 1, sizeof(double), cursor)
        && append_segment(layout, kParentPlanRows, birth_count, 1, sizeof(int64_t), cursor)
        && append_segment(layout, kMatePlanRows, birth_count, 1, sizeof(int64_t), cursor)
        && append_segment(layout, kChildPlanRows, birth_count, 1, sizeof(int64_t), cursor)
        && append_segment(layout, kIdentityParentSlots, birth_count, 1, sizeof(int64_t), cursor)
        && append_segment(layout, kIdentityChildSlots, birth_count, 1, sizeof(int64_t), cursor)
        && append_segment(layout, kIdentityChildIds, birth_count, 1, sizeof(int64_t), cursor)
        && append_segment(layout, kIdentityChildEnergy, birth_count, 1, sizeof(double), cursor)
        && append_segment(layout, kIdentityCount, 1, 1, sizeof(int64_t), cursor)
        && append_segment(layout, kQueueHead, 1, 1, sizeof(int64_t), cursor)
        && append_segment(layout, kQueueCount, 1, 1, sizeof(int64_t), cursor)
        && append_segment(layout, kParentAgentIds, birth_count, 1, sizeof(int64_t), cursor)
        && append_segment(layout, kChildLineageFounderIds, birth_count, 1, sizeof(int32_t), cursor)
        && append_segment(layout, kChildLineageDepths, birth_count, 1, sizeof(int32_t), cursor)
        && append_segment(layout, kGenomeInt32Planes, kGenomeInt32PlaneCount, birth_count, sizeof(int32_t), cursor)
        && append_segment(layout, kPolicyMemoryDimensionRows, birth_count, 1, sizeof(int64_t), cursor)
        && append_segment(layout, kPolicyUseGruRows, birth_count, 1, sizeof(bool), cursor)
        && append_segment(layout, kGenomeFloatPlanes, kGenomeFloatPlaneCount, birth_count, sizeof(float), cursor)
        && append_segment(layout, kDriveRows, birth_count, kDriveWidth, sizeof(float), cursor)
        && append_segment(layout, kChemistryRows, birth_count, kChemistryWidth, sizeof(float), cursor)
        && append_segment(layout, kBirthResetShockRows, birth_count, 1, sizeof(float), cursor)
        && append_segment(layout, kPolicyMlpRows, birth_count, packed_width, sizeof(float), cursor)
        && append_segment(layout, kHmemRows, birth_count, hmem_width, sizeof(float), cursor)
        && append_segment(layout, kPythonRandomPostWords, 1, kPythonRandomWordCount, sizeof(uint32_t), cursor)
        && append_segment(layout, kPythonRandomTrialWords, 1, kPythonRandomWordCount, sizeof(uint32_t), cursor)
        && append_segment(layout, kPythonRandomSnapshots, 2, 1, sizeof(PythonRandomSnapshotV2), cursor)
        && append_segment(layout, kPlannerPcg64Snapshot, 1, 1, sizeof(Pcg64SnapshotV2), cursor)
        && append_segment(layout, kRunnerPcg64Snapshot, 1, 1, sizeof(Pcg64SnapshotV2), cursor)
        && append_segment(layout, kPopulationPcg64Snapshot, 1, 1, sizeof(Pcg64SnapshotV2), cursor)
        && append_segment(layout, kTorchState, 1, torch_state_bytes, sizeof(uint8_t), cursor)
        && append_segment(layout, kMutationFlags, birth_count, 1, sizeof(bool), cursor)
        && append_segment(layout, kMutationDeltas, birth_count, 1, sizeof(float), cursor)
        && append_segment(layout, kLinkedDonorOrdinals, birth_count, 7, sizeof(uint8_t), cursor)
        && append_segment(
            layout,
            kDiploidStagedChildren,
            birth_count,
            1,
            sizeof(cuda_diploid_scalar_genome_v1::StagedChildV1),
            cursor
        )
        && append_segment(
            layout,
            kGenomeUpgradeV2StagedChildren,
            birth_count,
            1,
            sizeof(genome_upgrade_v2_q10_v1::StagedChildV1),
            cursor
        )
        && append_segment(layout, kEventStagingRows, event_staging_rows, 1, sizeof(BirthEventStagingRowV2), cursor)
        && append_segment(
            layout,
            kGenomeUpgradeV2QualificationFaultStage,
            1,
            1,
            sizeof(int32_t),
            cursor
        )
        && append_segment(layout, kScalarStatuses, 1, 1, sizeof(BirthScalarStatusesV2), cursor);

    if (!complete || !checked_align_up(cursor, kScratchArenaAlignment, layout.required_bytes)) {
        for (int32_t index = 0; index < kBirthScratchSegmentCount; ++index) {
            layout.segments[index] = {};
        }
        layout.required_bytes = 0;
        layout.status = kLayoutStatusInt64Overflow;
        return layout;
    }
    layout.status = kLayoutStatusOk;
    return layout;
}

EIEL_FLAT_BIRTH_SCRATCH_HD constexpr FlatBirthScratchLayoutV2 make_layout(
    const int64_t birth_count,
    const int64_t capacity,
    const int64_t packed_width,
    const int64_t hmem_width,
    const int64_t torch_state_bytes
) {
    return make_flattened_birth_scratch_layout_v2(
        birth_count,
        capacity,
        packed_width,
        hmem_width,
        torch_state_bytes
    );
}

// This view deliberately owns only the birth allocation contract. Turnover and
// resident-noise scratch must be passed through different storage objects.
struct FlatBirthScratchArenaV2 {
    uint8_t* storage;
    int64_t bytes_per_cell;

    EIEL_FLAT_BIRTH_SCRATCH_HD constexpr bool holds(
        const FlatBirthScratchLayoutV2 layout
    ) const {
        return storage != nullptr && layout.ok()
            && bytes_per_cell >= layout.required_bytes
            && bytes_per_cell % kScratchArenaAlignment == 0;
    }

    template <typename T>
    EIEL_FLAT_BIRTH_SCRATCH_HD T* at(
        const int64_t cell,
        const FlatBirthScratchLayoutV2 layout,
        const BirthScratchSegmentV2 segment
    ) const {
        const BirthScratchSegmentLayoutV2 selected = layout.segment(segment);
        if (
            !holds(layout) || cell < 0
            || (bytes_per_cell != 0 && cell > kInt64Max / bytes_per_cell)
        ) {
            return nullptr;
        }
        const int64_t cell_offset = cell * bytes_per_cell;
        if (cell_offset > kInt64Max - selected.offset) {
            return nullptr;
        }
        return reinterpret_cast<T*>(
            storage + cell_offset + selected.offset
        );
    }
};

static_assert(kMiniStateRoleCount == 3);
static_assert(kBirthScratchSegmentCount == 44);
static_assert(sizeof(cuda_diploid_scalar_genome_v1::StagedChildV1) == 256);
static_assert(sizeof(genome_upgrade_v2_q10_v1::StagedChildV1) == 976);
static_assert(sizeof(bool) == 1);
static_assert(sizeof(PythonRandomSnapshotV2) == 40);
static_assert(sizeof(Pcg64SnapshotV2) == 48);
static_assert(sizeof(BirthEventStagingRowV2) == 104);
static_assert(sizeof(BirthScalarStatusesV2) == 176);
static_assert(std::is_standard_layout<BirthEventStagingRowV2>::value);
static_assert(std::is_trivially_copyable<BirthEventStagingRowV2>::value);
static_assert(std::is_standard_layout<BirthScalarStatusesV2>::value);
static_assert(std::is_trivially_copyable<BirthScalarStatusesV2>::value);
static_assert(std::is_standard_layout<FlatBirthScratchLayoutV2>::value);
static_assert(std::is_trivially_copyable<FlatBirthScratchLayoutV2>::value);
static_assert(std::is_standard_layout<FlatBirthScratchArenaV2>::value);
static_assert(std::is_trivially_copyable<FlatBirthScratchArenaV2>::value);

}  // namespace eiel::cuda_architecture_b::flattened_birth_scratch_v2

#undef EIEL_FLAT_BIRTH_SCRATCH_HD
