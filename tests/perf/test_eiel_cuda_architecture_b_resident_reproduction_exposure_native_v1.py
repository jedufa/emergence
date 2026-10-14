from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_native_exposure_append_disabled_and_overflow_are_cell_local(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++ compiler is required")
    source = tmp_path / "exposure.cpp"
    executable = tmp_path / "exposure"
    source.write_text(
        r'''
#include "engine/perf/cuda_architecture_b_resident_reproduction_exposure_v1.cuh"
#include <cstdint>

namespace exposure =
    eiel::cuda_architecture_b::resident_reproduction_exposure_v1;

int main() {
    exposure::FlatResidentReproductionExposureV1 disabled{};
    if (exposure::attempt_capacity_qualification_faulted_v1(disabled, 0)) {
        return 34;
    }
    std::int64_t disabled_attempt = -1;
    std::int64_t disabled_primary_snapshot = -1;
    std::int64_t disabled_mate_snapshot = -1;
    if (!exposure::record_path_rng_commit_v1(disabled, 0, 0, 1)
        || !exposure::record_path_event_rows_v1(disabled, 0, 0, 1)
        || !exposure::reserve_attempt_with_two_snapshots_v1(
            disabled, 0, &disabled_attempt, &disabled_primary_snapshot,
            &disabled_mate_snapshot)
        || !exposure::begin_snapshot_v1(disabled, 0, 0, 0, 0, 0)
        || !exposure::append_aggregate_opportunity_v1(
            disabled, 0, 0, exposure::kPrimaryParentRole, 1)
        || !exposure::append_role_eligible_parent_v1(
            disabled, 0, 0, exposure::kPrimaryParentRole, 0)
        || !exposure::append_eligible_parent_v1(disabled, 0, 0, 0)
        || !exposure::commit_reserved_attempt_v1(
            disabled, 0, 0, 0, 1, 0, 0, 0, 0, 1, true, 0, 0)
        || !exposure::finalize_staged_attempt_v1(
            disabled, 0, 0, 0, 0, 0, 0, 1, 2, nullptr, 0)
        || !exposure::rebind_staged_attempt_selected_parents_v1(
            disabled, 0, 0, 0, 0, 0, 0, 1)
        || exposure::staged_attempt_count_v1(disabled, 0, 0, 0, 0) != 0
        || !exposure::abort_staged_attempts_v1(disabled, 0, 0, 0, 0)
        || !exposure::append_shared_set_successes_v1(
            disabled, 0, 0, 0, 0, nullptr, 0, nullptr, nullptr, nullptr, 0)) {
        return 37;
    }
    constexpr std::int64_t B = 5, E = 2, S = 4, C = 3, K = 2;
    bool enabled[B] = {true, false, true, true, true};
    std::int8_t storage_modes[B] = {
        exposure::kStorageExactFull,
        exposure::kStorageExactFull,
        exposure::kStorageExactFull,
        exposure::kStorageCompactComplete,
        exposure::kStorageExactFull,
    };
    std::int8_t attempt_paths[B * E];
    std::int64_t attempt_generations[B * E];
    std::int64_t attempt_ticks[B * E];
    std::int64_t refs[B * E * 2];
    std::int64_t attempt_eligible_counts[B * E * 2] = {};
    std::int64_t selected[B * E * 2];
    bool success[B * E] = {};
    std::int32_t failures[B * E];
    std::int64_t attempt_cursors[B] = {};
    std::int8_t snapshot_paths[B * S];
    std::int64_t generations[B * S];
    std::int64_t ticks[B * S];
    std::int64_t counts[B * S] = {};
    std::int64_t eligible[B * S * C];
    bool aggregate_seen[B * K] = {};
    std::int64_t aggregate_agent_ids[B * K];
    std::int64_t aggregate_primary[B * K] = {};
    std::int64_t aggregate_mate[B * K] = {};
    std::int64_t snapshot_cursors[B] = {};
    std::int64_t aggregate_seen_counts[B] = {};
    std::int32_t status[B] = {0, exposure::kStatusDisabled, 0, 0, 0};
    std::int64_t commit_ordinals[B * E];
    std::int64_t child_ids[B * E];
    std::uint8_t donor_ordinals[B * E * 7];
    std::int64_t donor_counts[B * E] = {};
    bool qualification_enabled[B] = {};
    std::int64_t qualification_capacity[B] = {E, E, E, E, E};
    constexpr std::int64_t P = 2;
    std::int8_t registered_paths[P] = {0, 1};
    std::int64_t path_rng_counters[B * P] = {};
    std::int64_t path_counter_writes[B * P] = {};
    std::int64_t path_event_writes[B * P] = {};
    std::int64_t path_rng_writes[B * P] = {};
    for (auto& value : attempt_paths) value = -1;
    for (auto& value : attempt_generations) value = -1;
    for (auto& value : attempt_ticks) value = -1;
    for (auto& value : refs) value = -1;
    for (auto& value : selected) value = -1;
    for (auto& value : failures) value = -1;
    for (auto& value : snapshot_paths) value = -1;
    for (auto& value : generations) value = -1;
    for (auto& value : ticks) value = -1;
    for (auto& value : eligible) value = -1;
    for (auto& value : aggregate_agent_ids) value = -1;
    for (auto& value : commit_ordinals) value = -1;
    for (auto& value : child_ids) value = -1;
    for (auto& value : donor_ordinals) value = exposure::kNoDonorOrdinal;
    exposure::FlatResidentReproductionExposureV1 owner{
        enabled, storage_modes, attempt_paths, attempt_generations,
        attempt_ticks, refs, attempt_eligible_counts, selected, success,
        failures, attempt_cursors, snapshot_paths, generations, ticks, counts,
        eligible, aggregate_seen, aggregate_agent_ids, aggregate_primary,
        aggregate_mate,
        snapshot_cursors, aggregate_seen_counts, status, commit_ordinals,
        child_ids, donor_ordinals, donor_counts, qualification_enabled,
        qualification_capacity, registered_paths, path_rng_counters,
        path_counter_writes, path_event_writes, path_rng_writes,
        E, S, C, K, P
    };
    const std::int64_t pool[3] = {10, 20, 30};
    const std::int64_t parent_slots[1] = {0};
    const std::int64_t mate_slots[1] = {1};
    const std::int64_t slot_to_id[3] = {10, 20, 30};
    if (!exposure::append_shared_set_successes_v1(
            owner, 0, 0, 4, 80, pool, 3, parent_slots, mate_slots,
            slot_to_id, 1)) return 1;
    if (attempt_cursors[0] != 1 || snapshot_cursors[0] != 1) return 2;
    if (refs[0] != 0 || refs[1] != 0) return 3;
    if (selected[0] != 10 || selected[1] != 20 || success[0]
        || failures[0] != exposure::kFailurePendingCommit) return 4;
    if (exposure::staged_attempt_count_v1(owner, 0, 0, 4, 80) != 1) {
        return 28;
    }
    if (!exposure::finalize_staged_attempt_v1(
            owner, 0, 0, 4, 80, 0, 10, 20, 31, nullptr, 0)) return 19;
    if (!success[0] || child_ids[0] != 31 || failures[0] != 0) return 20;
    if (path_event_writes[0] != 1) return 30;
    if (!exposure::record_path_rng_commit_v1(owner, 0, 0, 1)) return 31;
    if (path_rng_counters[0] != 1 || path_counter_writes[0] != 1
        || path_rng_writes[0] != 1 || path_rng_counters[1] != 0) return 32;
    if (exposure::staged_attempt_count_v1(owner, 0, 0, 4, 80) != 0) {
        return 29;
    }
    if (!exposure::append_shared_set_successes_v1(
            owner, 0, 0, 5, 80, pool, 3, parent_slots, mate_slots,
            slot_to_id, 1)) return 21;
    if (!exposure::abort_staged_attempts_v1(owner, 0, 0, 5, 80)) return 22;
    if (success[1]
        || failures[1] != exposure::kFailureAuthoritativeCommitAborted
        || commit_ordinals[1] != -1) return 23;
    if (!exposure::append_shared_set_successes_v1(
            owner, 1, 0, 4, 80, pool, 3, parent_slots, mate_slots,
            slot_to_id, 1)) return 5;
    if (attempt_cursors[1] != 0 || snapshot_cursors[1] != 0) return 6;
    const std::int64_t two_parent_slots[2] = {0, 1};
    const std::int64_t two_mate_slots[2] = {1, 2};
    if (exposure::append_shared_set_successes_v1(
            owner, 2, 0, 4, 80, pool, 3, two_parent_slots, two_mate_slots,
            slot_to_id, 3)) return 7;
    if (status[2] != exposure::kStatusAttemptCapacityExceeded) return 8;
    if (attempt_cursors[2] != 0 || snapshot_cursors[2] != 0) return 9;
    std::int64_t attempt = -1, primary_snapshot = -2, mate_snapshot = -2;
    constexpr std::int64_t high_id_0 = INT32_MAX - 2;
    constexpr std::int64_t high_id_1 = INT32_MAX - 1;
    constexpr std::int64_t high_id_2 = INT32_MAX;
    if (!exposure::reserve_attempt_with_two_snapshots_v1(
            owner, 3, &attempt, &primary_snapshot, &mate_snapshot)) return 10;
    if (attempt != 0 || primary_snapshot != -1 || mate_snapshot != -1) return 11;
    if (!exposure::append_role_eligible_parent_v1(
            owner, 3, primary_snapshot, exposure::kPrimaryParentRole,
            high_id_0)) {
        return 12;
    }
    if (!exposure::append_role_eligible_parent_v1(
            owner, 3, mate_snapshot, exposure::kMateParentRole,
            high_id_1)) {
        return 13;
    }
    if (!exposure::append_role_eligible_parent_v1(
            owner, 3, primary_snapshot, exposure::kPrimaryParentRole,
            high_id_0)) return 38;
    if (!exposure::commit_reserved_attempt_v1(
            owner, 3, attempt, primary_snapshot, mate_snapshot, 1, 7, 11,
            high_id_0, high_id_1, true, 0, 0)) return 14;
    const std::uint8_t linked_donors[7] = {0, 1, 0, 1, 0, 1, 0};
    if (!exposure::finalize_staged_attempt_v1(
            owner, 3, 1, 7, 11, 0, high_id_0, high_id_1, 60,
            linked_donors, 7)) return 24;
    const std::int64_t compact_row = 3 * E;
    if (attempt_generations[compact_row] != 7
        || attempt_ticks[compact_row] != 11
        || attempt_eligible_counts[compact_row * 2] != 2
        || attempt_eligible_counts[compact_row * 2 + 1] != 1
        || snapshot_cursors[3] != 0
        || aggregate_seen_counts[3] != 2
        || aggregate_agent_ids[3 * K] != high_id_1
        || aggregate_agent_ids[3 * K + 1] != high_id_0
        || aggregate_primary[3 * K + 1] != 2
        || aggregate_mate[3 * K] != 1
        || !success[compact_row]
        || child_ids[compact_row] != 60
        || donor_counts[compact_row] != 7) return 15;
    if (!exposure::reserve_attempt_with_two_snapshots_v1(
            owner, 3, &attempt, &primary_snapshot, &mate_snapshot)) return 16;
    if (exposure::append_role_eligible_parent_v1(
            owner, 3, primary_snapshot, exposure::kPrimaryParentRole,
            high_id_2)) {
        return 17;
    }
    if (status[3] != exposure::kStatusAggregateCapacityExceeded
        || attempt_cursors[3] != 1
        || status[0] != exposure::kStatusOk) return 18;
    if (!exposure::append_shared_set_successes_v1(
            owner, 4, 0, 9, 80, pool, 3, parent_slots, mate_slots,
            slot_to_id, 1)) return 25;
    if (!exposure::rebind_staged_attempt_selected_parents_v1(
            owner, 4, 0, 9, 80, 0, 20, 30)) return 33;
    if (selected[4 * E * 2] != 20 || selected[4 * E * 2 + 1] != 30) {
        return 34;
    }
    if (!exposure::finalize_staged_attempt_v1(
            owner, 4, 0, 9, 80, 0, 20, 30, 70, nullptr, 0)) return 35;
    if (!exposure::append_shared_set_successes_v1(
            owner, 4, 0, 10, 80, pool, 3, parent_slots, mate_slots,
            slot_to_id, 1)) return 36;
    const std::uint8_t invalid_donors[7] = {0, 1, 0, 2, 0, 1, 0};
    const std::int64_t invalid_row = 4 * E + 1;
    if (exposure::finalize_staged_attempt_v1(
            owner, 4, 0, 10, 80, 0, 10, 20, 71,
            invalid_donors, 7)) return 26;
    if (success[invalid_row] || child_ids[invalid_row] != -1
        || donor_counts[invalid_row] != 0
        || failures[invalid_row] != exposure::kFailurePendingCommit
        || status[4] != exposure::kStatusAuthoritativeCommitMismatch) return 27;
    return 0;
}
''',
        encoding="utf-8",
    )
    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT),
            str(source),
            "-o",
            str(executable),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run = subprocess.run([str(executable)], check=False)
    assert run.returncode == 0
