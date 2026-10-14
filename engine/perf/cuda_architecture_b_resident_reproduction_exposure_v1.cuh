#ifndef EIEL_CUDA_ARCHITECTURE_B_RESIDENT_REPRODUCTION_EXPOSURE_V1_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_RESIDENT_REPRODUCTION_EXPOSURE_V1_CUH_

#include <cstdint>

namespace eiel::cuda_architecture_b::resident_reproduction_exposure_v1 {

inline constexpr std::int8_t kNoPath = -1;
inline constexpr std::int8_t kPrimaryParentRole = 0;
inline constexpr std::int8_t kMateParentRole = 1;
inline constexpr std::int32_t kStatusOk = 0;
inline constexpr std::int32_t kStatusDisabled = 1;
inline constexpr std::int32_t kStatusAttemptCapacityExceeded = 2;
inline constexpr std::int32_t kStatusSnapshotCapacityExceeded = 3;
inline constexpr std::int32_t kStatusAggregateCapacityExceeded = 4;
inline constexpr std::int32_t kFailureSuccess = 0;
inline constexpr std::int32_t kFailurePendingCommit = INT32_MAX;
inline constexpr std::int32_t kFailureAuthoritativeCommitAborted =
    INT32_MAX - 1;
inline constexpr std::int32_t kStatusAuthoritativeCommitMismatch = 5;
inline constexpr std::uint8_t kNoDonorOrdinal = UINT8_MAX;
inline constexpr std::int64_t kDonorOrdinalCapacity = 7;
inline constexpr std::int8_t kStorageExactFull = 0;
inline constexpr std::int8_t kStorageCompactComplete = 1;

// Generic append-only, per-cell exposure owner. Attempt rows and eligible-set
// snapshots are independently bounded. Every pointer addresses cell-major
// contiguous CUDA storage and every cursor/status has one row per cell.
struct FlatResidentReproductionExposureV1 {
    const bool* cell_enabled;
    const std::int8_t* storage_mode_codes;
    std::int8_t* attempt_path_codes;
    std::int64_t* attempt_generations;
    std::int64_t* attempt_ticks;
    std::int64_t* attempt_eligible_snapshot_refs;
    std::int64_t* attempt_eligible_counts;
    std::int64_t* attempt_selected_parent_ids;
    bool* attempt_success;
    std::int32_t* attempt_failure_codes;
    std::int64_t* attempt_cursors;
    std::int8_t* eligible_path_codes;
    std::int64_t* eligible_generations;
    std::int64_t* eligible_ticks;
    std::int64_t* eligible_parent_counts;
    std::int64_t* eligible_parent_agent_ids;
    bool* aggregate_agent_seen;
    std::int64_t* aggregate_agent_ids;
    std::int64_t* aggregate_primary_opportunity_counts;
    std::int64_t* aggregate_mate_opportunity_counts;
    std::int64_t* eligible_snapshot_cursors;
    std::int64_t* aggregate_seen_counts;
    std::int32_t* status;
    std::int64_t* attempt_commit_ordinals;
    std::int64_t* attempt_child_agent_ids;
    std::uint8_t* attempt_donor_ordinals;
    std::int64_t* attempt_donor_draw_counts;
    const bool* qualification_attempt_capacity_enabled;
    const std::int64_t* qualification_effective_attempt_capacity;
    const std::int8_t* registered_path_codes;
    std::int64_t* path_rng_counter_states;
    std::int64_t* path_counter_write_counts;
    std::int64_t* path_event_row_write_counts;
    std::int64_t* path_rng_state_write_counts;
    std::int64_t attempt_capacity;
    std::int64_t snapshot_capacity;
    std::int64_t eligible_parent_capacity;
    std::int64_t aggregate_capacity;
    std::int64_t registered_path_count;
};

#if defined(__CUDACC__)
#define EIEL_REPRO_EXPOSURE_DEVICE __host__ __device__ __forceinline__
#else
#define EIEL_REPRO_EXPOSURE_DEVICE inline
#endif

EIEL_REPRO_EXPOSURE_DEVICE bool enabled_v1(
    const FlatResidentReproductionExposureV1& value
) noexcept {
    return value.attempt_path_codes != nullptr;
}

EIEL_REPRO_EXPOSURE_DEVICE bool complete_v1(
    const FlatResidentReproductionExposureV1& value
) noexcept {
    return !enabled_v1(value)
        || (
            value.cell_enabled != nullptr
            && value.storage_mode_codes != nullptr
            && value.attempt_generations != nullptr
            && value.attempt_ticks != nullptr
            && value.attempt_eligible_snapshot_refs != nullptr
            && value.attempt_eligible_counts != nullptr
            && value.attempt_selected_parent_ids != nullptr
            && value.attempt_success != nullptr
            && value.attempt_failure_codes != nullptr
            && value.attempt_cursors != nullptr
            && value.eligible_path_codes != nullptr
            && value.eligible_generations != nullptr
            && value.eligible_ticks != nullptr
            && value.eligible_parent_counts != nullptr
            && value.eligible_parent_agent_ids != nullptr
            && value.aggregate_agent_seen != nullptr
            && value.aggregate_agent_ids != nullptr
            && value.aggregate_primary_opportunity_counts != nullptr
            && value.aggregate_mate_opportunity_counts != nullptr
            && value.eligible_snapshot_cursors != nullptr
            && value.aggregate_seen_counts != nullptr
            && value.status != nullptr
            && value.attempt_commit_ordinals != nullptr
            && value.attempt_child_agent_ids != nullptr
            && value.attempt_donor_ordinals != nullptr
            && value.attempt_donor_draw_counts != nullptr
            && value.qualification_attempt_capacity_enabled != nullptr
            && value.qualification_effective_attempt_capacity != nullptr
            && value.registered_path_codes != nullptr
            && value.path_rng_counter_states != nullptr
            && value.path_counter_write_counts != nullptr
            && value.path_event_row_write_counts != nullptr
            && value.path_rng_state_write_counts != nullptr
            && value.attempt_capacity > 0
            && value.snapshot_capacity > 0
            && value.eligible_parent_capacity > 0
            && value.aggregate_capacity > 0
            && value.registered_path_count > 0
        );
}

EIEL_REPRO_EXPOSURE_DEVICE std::int64_t registered_path_index_v1(
    const FlatResidentReproductionExposureV1& value,
    const std::int8_t path
) noexcept {
    if (!complete_v1(value)) {
        return -1;
    }
    for (std::int64_t index = 0; index < value.registered_path_count; ++index) {
        if (value.registered_path_codes[index] == path) {
            return index;
        }
    }
    return -1;
}

EIEL_REPRO_EXPOSURE_DEVICE bool record_path_rng_commit_v1(
    const FlatResidentReproductionExposureV1& value,
    const std::int64_t cell,
    const std::int8_t path,
    const std::int64_t counter_delta
) noexcept {
    if (!enabled_v1(value)) {
        return true;
    }
    const std::int64_t path_index = registered_path_index_v1(value, path);
    if (
        cell < 0 || path_index < 0 || counter_delta < 0
        || !value.cell_enabled[cell] || value.status[cell] != kStatusOk
    ) {
        return false;
    }
    if (counter_delta == 0) {
        return true;
    }
    const std::int64_t row =
        cell * value.registered_path_count + path_index;
    if (
        value.path_rng_counter_states[row] > INT64_MAX - counter_delta
        || value.path_counter_write_counts[row] == INT64_MAX
        || value.path_rng_state_write_counts[row] == INT64_MAX
    ) {
        value.status[cell] = kStatusAggregateCapacityExceeded;
        return false;
    }
    value.path_rng_counter_states[row] += counter_delta;
    ++value.path_counter_write_counts[row];
    ++value.path_rng_state_write_counts[row];
    return true;
}

EIEL_REPRO_EXPOSURE_DEVICE bool record_path_event_rows_v1(
    const FlatResidentReproductionExposureV1& value,
    const std::int64_t cell,
    const std::int8_t path,
    const std::int64_t rows
) noexcept {
    if (!enabled_v1(value)) {
        return true;
    }
    const std::int64_t path_index = registered_path_index_v1(value, path);
    if (
        cell < 0 || path_index < 0 || rows < 0
        || !value.cell_enabled[cell] || value.status[cell] != kStatusOk
    ) {
        return false;
    }
    const std::int64_t row =
        cell * value.registered_path_count + path_index;
    if (value.path_event_row_write_counts[row] > INT64_MAX - rows) {
        value.status[cell] = kStatusAggregateCapacityExceeded;
        return false;
    }
    value.path_event_row_write_counts[row] += rows;
    return true;
}

EIEL_REPRO_EXPOSURE_DEVICE bool apply_attempt_capacity_qualification_v1(
    const FlatResidentReproductionExposureV1& value,
    const std::int64_t cell
) noexcept {
    if (
        !enabled_v1(value)
        || !complete_v1(value)
        || cell < 0
        || !value.cell_enabled[cell]
        || !value.qualification_attempt_capacity_enabled[cell]
    ) {
        return false;
    }
    if (value.status[cell] == kStatusAttemptCapacityExceeded) {
        return true;
    }
    if (value.status[cell] != kStatusOk) {
        return false;
    }
    const std::int64_t effective_capacity =
        value.qualification_effective_attempt_capacity[cell];
    if (
        effective_capacity < 0
        || effective_capacity >= value.attempt_capacity
        || value.attempt_cursors[cell] < effective_capacity
    ) {
        return false;
    }
    value.status[cell] = kStatusAttemptCapacityExceeded;
    return true;
}

EIEL_REPRO_EXPOSURE_DEVICE bool attempt_capacity_qualification_faulted_v1(
    const FlatResidentReproductionExposureV1& value,
    const std::int64_t cell
) noexcept {
    return enabled_v1(value)
        && complete_v1(value)
        && cell >= 0
        && value.cell_enabled[cell]
        && value.qualification_attempt_capacity_enabled[cell]
        && value.status[cell] == kStatusAttemptCapacityExceeded;
}

EIEL_REPRO_EXPOSURE_DEVICE bool reserve_attempt_with_two_snapshots_v1(
    const FlatResidentReproductionExposureV1& value,
    const std::int64_t cell,
    std::int64_t* const attempt,
    std::int64_t* const primary_snapshot,
    std::int64_t* const mate_snapshot
) noexcept {
    if (!enabled_v1(value)) {
        return true;
    }
    if (!complete_v1(value) || cell < 0) {
        return false;
    }
    if (!value.cell_enabled[cell] || value.status[cell] != kStatusOk) {
        return false;
    }
    *attempt = value.attempt_cursors[cell];
    if (value.storage_mode_codes[cell] == kStorageCompactComplete) {
        *primary_snapshot = -1;
        *mate_snapshot = -1;
        if (*attempt < 0 || *attempt >= value.attempt_capacity) {
            value.status[cell] = kStatusAttemptCapacityExceeded;
            return false;
        }
        return true;
    }
    *primary_snapshot = value.eligible_snapshot_cursors[cell];
    *mate_snapshot = *primary_snapshot + 1;
    if (*attempt < 0 || *attempt >= value.attempt_capacity) {
        value.status[cell] = kStatusAttemptCapacityExceeded;
        return false;
    }
    if (
        *primary_snapshot < 0
        || *primary_snapshot > value.snapshot_capacity - 2
    ) {
        value.status[cell] = kStatusSnapshotCapacityExceeded;
        return false;
    }
    return true;
}

EIEL_REPRO_EXPOSURE_DEVICE bool begin_snapshot_v1(
    const FlatResidentReproductionExposureV1& value,
    const std::int64_t cell,
    const std::int64_t snapshot,
    const std::int8_t path,
    const std::int64_t generation,
    const std::int64_t tick
) noexcept {
    if (!enabled_v1(value)) {
        return true;
    }
    if (!complete_v1(value) || cell < 0) {
        return false;
    }
    if (value.storage_mode_codes[cell] == kStorageCompactComplete) {
        return true;
    }
    if (snapshot < 0 || snapshot >= value.snapshot_capacity) {
        return false;
    }
    const std::int64_t row = cell * value.snapshot_capacity + snapshot;
    value.eligible_path_codes[row] = path;
    value.eligible_generations[row] = generation;
    value.eligible_ticks[row] = tick;
    value.eligible_parent_counts[row] = 0;
    const std::int64_t base = row * value.eligible_parent_capacity;
    for (
        std::int64_t index = 0;
        index < value.eligible_parent_capacity;
        ++index
    ) {
        value.eligible_parent_agent_ids[base + index] = -1;
    }
    return true;
}

EIEL_REPRO_EXPOSURE_DEVICE bool append_aggregate_opportunity_v1(
    const FlatResidentReproductionExposureV1& value,
    const std::int64_t cell,
    const std::int64_t agent_id,
    const std::int8_t role,
    const std::int64_t delta
) noexcept {
    if (!enabled_v1(value)) {
        return true;
    }
    if (
        !complete_v1(value)
        || cell < 0
        || agent_id < 0
        || (role != kPrimaryParentRole && role != kMateParentRole)
        || delta < 0
    ) {
        return false;
    }
    const std::int64_t seen_count = value.aggregate_seen_counts[cell];
    if (seen_count < 0 || seen_count > value.aggregate_capacity) {
        value.status[cell] = kStatusAggregateCapacityExceeded;
        return false;
    }
    const std::int64_t base = cell * value.aggregate_capacity;
    const std::int64_t first_row = agent_id % value.aggregate_capacity;
    std::int64_t row = -1;
    for (
        std::int64_t probe = 0;
        probe < value.aggregate_capacity;
        ++probe
    ) {
        const std::int64_t candidate =
            (first_row + probe) % value.aggregate_capacity;
        if (value.aggregate_agent_seen[base + candidate]) {
            if (value.aggregate_agent_ids[base + candidate] == agent_id) {
                row = candidate;
                break;
            }
            continue;
        }
        value.aggregate_agent_seen[base + candidate] = true;
        value.aggregate_agent_ids[base + candidate] = agent_id;
        value.aggregate_seen_counts[cell] = seen_count + 1;
        row = candidate;
        break;
    }
    if (row < 0) {
        value.status[cell] = kStatusAggregateCapacityExceeded;
        return false;
    }
    std::int64_t* const counts = role == kPrimaryParentRole
        ? value.aggregate_primary_opportunity_counts
        : value.aggregate_mate_opportunity_counts;
    if (counts[base + row] > INT64_MAX - delta) {
        value.status[cell] = kStatusAggregateCapacityExceeded;
        return false;
    }
    counts[base + row] += delta;
    return true;
}

EIEL_REPRO_EXPOSURE_DEVICE bool append_eligible_parent_v1(
    const FlatResidentReproductionExposureV1& value,
    const std::int64_t cell,
    const std::int64_t snapshot,
    const std::int64_t agent_id
) noexcept;

EIEL_REPRO_EXPOSURE_DEVICE bool append_role_eligible_parent_v1(
    const FlatResidentReproductionExposureV1& value,
    const std::int64_t cell,
    const std::int64_t snapshot,
    const std::int8_t role,
    const std::int64_t agent_id
) noexcept {
    if (!enabled_v1(value)) {
        return true;
    }
    if (
        complete_v1(value)
        && cell >= 0
        && value.storage_mode_codes[cell] == kStorageCompactComplete
    ) {
        const std::int64_t attempt = value.attempt_cursors[cell];
        if (
            attempt < 0
            || attempt >= value.attempt_capacity
            || !append_aggregate_opportunity_v1(
                value, cell, agent_id, role, 1
            )
        ) {
            return false;
        }
        ++value.attempt_eligible_counts[
            (cell * value.attempt_capacity + attempt) * 2 + role
        ];
        return true;
    }
    return append_eligible_parent_v1(value, cell, snapshot, agent_id);
}

EIEL_REPRO_EXPOSURE_DEVICE bool append_eligible_parent_v1(
    const FlatResidentReproductionExposureV1& value,
    const std::int64_t cell,
    const std::int64_t snapshot,
    const std::int64_t agent_id
) noexcept {
    if (!enabled_v1(value)) {
        return true;
    }
    if (
        !complete_v1(value)
        || cell < 0
        || snapshot < 0
        || snapshot >= value.snapshot_capacity
        || agent_id < 0
    ) {
        return false;
    }
    const std::int64_t row = cell * value.snapshot_capacity + snapshot;
    const std::int64_t count = value.eligible_parent_counts[row];
    if (count < 0 || count >= value.eligible_parent_capacity) {
        value.status[cell] = kStatusSnapshotCapacityExceeded;
        return false;
    }
    value.eligible_parent_agent_ids[
        row * value.eligible_parent_capacity + count
    ] = agent_id;
    value.eligible_parent_counts[row] = count + 1;
    return true;
}

EIEL_REPRO_EXPOSURE_DEVICE bool commit_reserved_attempt_v1(
    const FlatResidentReproductionExposureV1& value,
    const std::int64_t cell,
    const std::int64_t attempt,
    const std::int64_t primary_snapshot,
    const std::int64_t mate_snapshot,
    const std::int8_t path,
    const std::int64_t generation,
    const std::int64_t tick,
    const std::int64_t selected_primary_id,
    const std::int64_t selected_mate_id,
    const bool success,
    const std::int32_t failure_code,
    const std::int64_t commit_ordinal
) noexcept {
    if (!enabled_v1(value)) {
        return true;
    }
    if (
        !complete_v1(value)
        || cell < 0
        || value.status[cell] != kStatusOk
        || attempt != value.attempt_cursors[cell]
        || (
            value.storage_mode_codes[cell] == kStorageExactFull
            && (
                primary_snapshot != value.eligible_snapshot_cursors[cell]
                || mate_snapshot != primary_snapshot + 1
            )
        )
    ) {
        return false;
    }
    const std::int64_t attempt_row =
        cell * value.attempt_capacity + attempt;
    value.attempt_path_codes[attempt_row] = path;
    value.attempt_generations[attempt_row] = generation;
    value.attempt_ticks[attempt_row] = tick;
    if (value.storage_mode_codes[cell] == kStorageExactFull) {
        value.attempt_eligible_snapshot_refs[attempt_row * 2] =
            primary_snapshot;
        value.attempt_eligible_snapshot_refs[attempt_row * 2 + 1] =
            mate_snapshot;
        const std::int64_t primary_row =
            cell * value.snapshot_capacity + primary_snapshot;
        const std::int64_t mate_row =
            cell * value.snapshot_capacity + mate_snapshot;
        value.attempt_eligible_counts[attempt_row * 2] =
            value.eligible_parent_counts[primary_row];
        value.attempt_eligible_counts[attempt_row * 2 + 1] =
            value.eligible_parent_counts[mate_row];
    }
    value.attempt_selected_parent_ids[attempt_row * 2] =
        selected_primary_id;
    value.attempt_selected_parent_ids[attempt_row * 2 + 1] =
        selected_mate_id;
    value.attempt_success[attempt_row] = false;
    value.attempt_failure_codes[attempt_row] =
        success ? kFailurePendingCommit : failure_code;
    value.attempt_commit_ordinals[attempt_row] =
        success ? commit_ordinal : -1;
    if (value.storage_mode_codes[cell] == kStorageExactFull) {
        value.eligible_snapshot_cursors[cell] = mate_snapshot + 1;
    }
    value.attempt_cursors[cell] = attempt + 1;
    return record_path_event_rows_v1(value, cell, path, 1);
}

EIEL_REPRO_EXPOSURE_DEVICE bool finalize_staged_attempt_v1(
    const FlatResidentReproductionExposureV1& value,
    const std::int64_t cell,
    const std::int8_t path,
    const std::int64_t generation,
    const std::int64_t tick,
    const std::int64_t commit_ordinal,
    const std::int64_t committed_parent_id,
    const std::int64_t committed_mate_id,
    const std::int64_t child_agent_id,
    const std::uint8_t* const donor_ordinals,
    const std::int64_t donor_count
) noexcept {
    if (!enabled_v1(value)) {
        return true;
    }
    if (!complete_v1(value) || cell < 0) {
        return false;
    }
    if (
        value.status[cell] != kStatusOk
        || commit_ordinal < 0
        || child_agent_id < 0
        || donor_count < 0
        || donor_count > kDonorOrdinalCapacity
    ) {
        return false;
    }
    const std::int64_t base = cell * value.attempt_capacity;
    for (std::int64_t attempt = value.attempt_cursors[cell]; attempt > 0;) {
        --attempt;
        const std::int64_t row = base + attempt;
        if (
            value.attempt_path_codes[row] != path
            || value.attempt_generations[row] != generation
            || value.attempt_ticks[row] != tick
            || value.attempt_commit_ordinals[row] != commit_ordinal
        ) {
            continue;
        }
        if (
            value.attempt_failure_codes[row] != kFailurePendingCommit
            || value.attempt_success[row]
            || value.attempt_selected_parent_ids[row * 2]
                != committed_parent_id
            || value.attempt_selected_parent_ids[row * 2 + 1]
                != committed_mate_id
        ) {
            value.status[cell] = kStatusAuthoritativeCommitMismatch;
            return false;
        }
        for (std::int64_t index = 0; index < donor_count; ++index) {
            if (donor_ordinals == nullptr || donor_ordinals[index] > 1) {
                value.status[cell] = kStatusAuthoritativeCommitMismatch;
                return false;
            }
        }
        value.attempt_success[row] = true;
        value.attempt_failure_codes[row] = kFailureSuccess;
        value.attempt_child_agent_ids[row] = child_agent_id;
        value.attempt_donor_draw_counts[row] = donor_count;
        const std::int64_t donor_base = row * kDonorOrdinalCapacity;
        for (std::int64_t index = 0; index < donor_count; ++index) {
            const std::uint8_t donor = donor_ordinals[index];
            value.attempt_donor_ordinals[donor_base + index] = donor;
        }
        return true;
    }
    value.status[cell] = kStatusAuthoritativeCommitMismatch;
    return false;
}

// Some mechanics reserve a successful attempt from a provisional allocation
// pair and select the authoritative genetic pair later inside the scalar RNG
// transaction. Rebind only that pending row before finalization; eligibility
// snapshots and the commit ordinal remain unchanged.
EIEL_REPRO_EXPOSURE_DEVICE bool rebind_staged_attempt_selected_parents_v1(
    const FlatResidentReproductionExposureV1& value,
    const std::int64_t cell,
    const std::int8_t path,
    const std::int64_t generation,
    const std::int64_t tick,
    const std::int64_t commit_ordinal,
    const std::int64_t authoritative_parent_id,
    const std::int64_t authoritative_mate_id
) noexcept {
    if (!enabled_v1(value)) {
        return true;
    }
    if (!complete_v1(value) || cell < 0) {
        return false;
    }
    if (
        value.status[cell] != kStatusOk
        || commit_ordinal < 0
        || authoritative_parent_id < 0
        || authoritative_mate_id < 0
        || authoritative_parent_id == authoritative_mate_id
    ) {
        return false;
    }
    const std::int64_t base = cell * value.attempt_capacity;
    for (std::int64_t attempt = value.attempt_cursors[cell]; attempt > 0;) {
        --attempt;
        const std::int64_t row = base + attempt;
        if (
            value.attempt_path_codes[row] != path
            || value.attempt_generations[row] != generation
            || value.attempt_ticks[row] != tick
            || value.attempt_commit_ordinals[row] != commit_ordinal
        ) {
            continue;
        }
        if (
            value.attempt_failure_codes[row] != kFailurePendingCommit
            || value.attempt_success[row]
        ) {
            value.status[cell] = kStatusAuthoritativeCommitMismatch;
            return false;
        }
        value.attempt_selected_parent_ids[row * 2] =
            authoritative_parent_id;
        value.attempt_selected_parent_ids[row * 2 + 1] =
            authoritative_mate_id;
        return true;
    }
    value.status[cell] = kStatusAuthoritativeCommitMismatch;
    return false;
}

EIEL_REPRO_EXPOSURE_DEVICE std::int64_t staged_attempt_count_v1(
    const FlatResidentReproductionExposureV1& value,
    const std::int64_t cell,
    const std::int8_t path,
    const std::int64_t generation,
    const std::int64_t tick
) noexcept {
    if (!enabled_v1(value)) {
        return 0;
    }
    if (!complete_v1(value) || cell < 0) {
        return -1;
    }
    std::int64_t count = 0;
    const std::int64_t base = cell * value.attempt_capacity;
    for (std::int64_t attempt = 0; attempt < value.attempt_cursors[cell]; ++attempt) {
        const std::int64_t row = base + attempt;
        if (
            value.attempt_path_codes[row] == path
            && value.attempt_generations[row] == generation
            && value.attempt_ticks[row] == tick
            && value.attempt_failure_codes[row] == kFailurePendingCommit
        ) {
            ++count;
        }
    }
    return count;
}

EIEL_REPRO_EXPOSURE_DEVICE bool abort_staged_attempts_v1(
    const FlatResidentReproductionExposureV1& value,
    const std::int64_t cell,
    const std::int8_t path,
    const std::int64_t generation,
    const std::int64_t tick
) noexcept {
    if (!enabled_v1(value)) {
        return true;
    }
    if (!complete_v1(value) || cell < 0) {
        return false;
    }
    const std::int64_t base = cell * value.attempt_capacity;
    for (std::int64_t attempt = 0; attempt < value.attempt_cursors[cell]; ++attempt) {
        const std::int64_t row = base + attempt;
        if (
            value.attempt_path_codes[row] == path
            && value.attempt_generations[row] == generation
            && value.attempt_ticks[row] == tick
            && value.attempt_failure_codes[row] == kFailurePendingCommit
        ) {
            value.attempt_success[row] = false;
            value.attempt_failure_codes[row] =
                kFailureAuthoritativeCommitAborted;
            value.attempt_commit_ordinals[row] = -1;
        }
    }
    return true;
}

EIEL_REPRO_EXPOSURE_DEVICE bool append_shared_set_successes_v1(
    const FlatResidentReproductionExposureV1& value,
    const std::int64_t cell,
    const std::int8_t path,
    const std::int64_t generation,
    const std::int64_t tick,
    const std::int64_t* const eligible_ids,
    const std::int64_t eligible_count,
    const std::int64_t* const selected_primary_slots,
    const std::int64_t* const selected_mate_slots,
    const std::int64_t* const slot_to_agent_id,
    const std::int64_t attempt_count
) noexcept {
    if (!enabled_v1(value)) {
        return true;
    }
    if (!complete_v1(value) || cell < 0) {
        return false;
    }
    if (!value.cell_enabled[cell]) {
        return true;
    }
    const std::int64_t attempt = value.attempt_cursors[cell];
    const std::int64_t snapshot = value.eligible_snapshot_cursors[cell];
    if (
        value.status[cell] != kStatusOk
        || attempt_count < 0
        || attempt < 0
        || attempt_count > value.attempt_capacity - attempt
    ) {
        value.status[cell] = kStatusAttemptCapacityExceeded;
        return false;
    }
    if (value.storage_mode_codes[cell] == kStorageCompactComplete) {
        if (eligible_count < 1) {
            value.status[cell] = kStatusAggregateCapacityExceeded;
            return false;
        }
        for (std::int64_t index = 0; index < eligible_count; ++index) {
            if (
                !append_aggregate_opportunity_v1(
                    value,
                    cell,
                    eligible_ids[index],
                    kPrimaryParentRole,
                    attempt_count
                )
                || !append_aggregate_opportunity_v1(
                    value,
                    cell,
                    eligible_ids[index],
                    kMateParentRole,
                    attempt_count
                )
            ) {
                return false;
            }
        }
        for (std::int64_t index = 0; index < attempt_count; ++index) {
            const std::int64_t row =
                cell * value.attempt_capacity + attempt + index;
            value.attempt_path_codes[row] = path;
            value.attempt_generations[row] = generation;
            value.attempt_ticks[row] = tick;
            value.attempt_eligible_counts[row * 2] = eligible_count;
            value.attempt_eligible_counts[row * 2 + 1] = eligible_count;
            value.attempt_selected_parent_ids[row * 2] =
                slot_to_agent_id[selected_primary_slots[index]];
            value.attempt_selected_parent_ids[row * 2 + 1] =
                slot_to_agent_id[selected_mate_slots[index]];
            value.attempt_success[row] = false;
            value.attempt_failure_codes[row] = kFailurePendingCommit;
            value.attempt_commit_ordinals[row] = index;
        }
        value.attempt_cursors[cell] = attempt + attempt_count;
        return record_path_event_rows_v1(
            value, cell, path, attempt_count
        );
    }
    if (
        snapshot < 0
        || snapshot >= value.snapshot_capacity
        || eligible_count < 1
        || eligible_count > value.eligible_parent_capacity
    ) {
        value.status[cell] = kStatusSnapshotCapacityExceeded;
        return false;
    }
    begin_snapshot_v1(
        value, cell, snapshot, path, generation, tick
    );
    for (std::int64_t index = 0; index < eligible_count; ++index) {
        if (!append_eligible_parent_v1(
                value, cell, snapshot, eligible_ids[index]
            )) {
            return false;
        }
    }
    for (std::int64_t index = 0; index < attempt_count; ++index) {
        const std::int64_t row =
            cell * value.attempt_capacity + attempt + index;
        value.attempt_path_codes[row] = path;
        value.attempt_generations[row] = generation;
        value.attempt_ticks[row] = tick;
        value.attempt_eligible_snapshot_refs[row * 2] = snapshot;
        value.attempt_eligible_snapshot_refs[row * 2 + 1] = snapshot;
        value.attempt_eligible_counts[row * 2] = eligible_count;
        value.attempt_eligible_counts[row * 2 + 1] = eligible_count;
        value.attempt_selected_parent_ids[row * 2] =
            slot_to_agent_id[selected_primary_slots[index]];
        value.attempt_selected_parent_ids[row * 2 + 1] =
            slot_to_agent_id[selected_mate_slots[index]];
        value.attempt_success[row] = false;
        value.attempt_failure_codes[row] = kFailurePendingCommit;
        value.attempt_commit_ordinals[row] = index;
    }
    value.eligible_snapshot_cursors[cell] = snapshot + 1;
    value.attempt_cursors[cell] = attempt + attempt_count;
    return record_path_event_rows_v1(value, cell, path, attempt_count);
}

#undef EIEL_REPRO_EXPOSURE_DEVICE

}  // namespace eiel::cuda_architecture_b::resident_reproduction_exposure_v1

#endif
