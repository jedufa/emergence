#ifndef EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_BIRTH_EVENTS_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_BIRTH_EVENTS_CUH_

#include "cuda_architecture_b_persistent_lifecycle_events.cuh"

#include <cstdint>

namespace eiel::cuda_architecture_b::persistent_birth_events {

using persistent_lifecycle_events::FlatEventTapeColumns;
using persistent_lifecycle_events::FlatEventTapeState;
using persistent_lifecycle_events::FlatStridedCellRows;
using persistent_lifecycle_events::FlatStridedCellScalars;

constexpr int32_t kStatusOk = persistent_lifecycle_events::kStatusOk;
constexpr int32_t kStatusCapacityExceeded =
    persistent_lifecycle_events::kStatusCapacityExceeded;
constexpr int32_t kStatusInvalidSemantics =
    persistent_lifecycle_events::kStatusInvalidSemantics;
constexpr int32_t kBirthCommittedEventType = 1;
constexpr int32_t kMutationEventType = 2;
constexpr int32_t kFailedBirthEventType = 3;
constexpr int64_t kSentinelId = persistent_lifecycle_events::kSentinelId;

// All rows are fixed-capacity, per-cell views. The caller owns pointer and
// stride validity, matching the host checks around immutable Stage2.
struct FlatOrdinaryBirthEventIdentity {
    FlatStridedCellRows<const int64_t> slot_to_agent_id;
    FlatStridedCellRows<const int32_t> lineage_founder_id;
    FlatStridedCellRows<const int32_t> lineage_depth;
    int64_t slot_count;
};

struct FlatOrdinaryBirthDecisionRows {
    FlatStridedCellRows<const int64_t> parent_slots;
    FlatStridedCellRows<const int32_t> reason_codes;
    FlatStridedCellScalars<const int64_t> count;
    int64_t capacity;
};

struct FlatOrdinaryBirthCommitRows {
    FlatStridedCellRows<const int64_t> parent_agent_ids;
    FlatStridedCellRows<const int64_t> child_agent_ids;
    FlatStridedCellRows<const int64_t> parent_slots;
    FlatStridedCellRows<const int64_t> child_slots;
    FlatStridedCellRows<const bool> mutation_occurred;
    FlatStridedCellRows<const float> mutation_delta_total;
    FlatStridedCellRows<const float> child_energy;
    FlatStridedCellScalars<const int64_t> count;
    int64_t capacity;
};

namespace detail {

// Immutable Stage2 source:
// native/cuda_architecture_b_birth_lifecycle_events.cu:92-133.
__device__ __forceinline__ void publish_event(
    const FlatEventTapeColumns& columns,
    const int64_t cell,
    const int64_t row,
    const int32_t event_type,
    const int64_t generation,
    const int64_t tick,
    const int64_t actor_id,
    const int64_t target_id,
    const int64_t actor_slot,
    const int64_t target_slot,
    const int64_t parent_id,
    const int64_t child_id,
    const int64_t founder_id,
    const int32_t lineage_depth,
    const int32_t subtype,
    const int32_t reason,
    const int8_t success,
    const float value
) {
    columns.event_type.at(cell, row) = event_type;
    columns.generation.at(cell, row) = generation;
    columns.tick.at(cell, row) = tick;
    columns.actor_id.at(cell, row) = actor_id;
    columns.target_id.at(cell, row) = target_id;
    columns.value.at(cell, row) = value;
    columns.typed_event_type.at(cell, row) = event_type;
    columns.typed_generation.at(cell, row) = generation;
    columns.typed_tick.at(cell, row) = tick;
    columns.typed_sequence.at(cell, row) = row;
    columns.typed_actor_id.at(cell, row) = actor_id;
    columns.typed_target_id.at(cell, row) = target_id;
    columns.typed_actor_slot.at(cell, row) = actor_slot;
    columns.typed_target_slot.at(cell, row) = target_slot;
    columns.typed_parent_id.at(cell, row) = parent_id;
    columns.typed_child_id.at(cell, row) = child_id;
    columns.typed_founder_id.at(cell, row) = founder_id;
    columns.typed_lineage_depth.at(cell, row) = lineage_depth;
    columns.typed_subtype.at(cell, row) = subtype;
    columns.typed_reason.at(cell, row) = reason;
    columns.typed_success.at(cell, row) = success;
    columns.typed_value.at(cell, row) = value;
}

}  // namespace detail

// Exact dynamic validation and event-count pass from immutable Stage2
// native/cuda_architecture_b_birth_lifecycle_events.cu:135-225. Status is
// deliberately reset to OK before validation; capacity failure publishes no
// rows and leaves the active cursor unchanged.
__device__ __forceinline__ bool preflight_ordinary_birth_events_serial(
    const FlatEventTapeState& tape,
    const FlatOrdinaryBirthEventIdentity& identity,
    const FlatOrdinaryBirthDecisionRows& decisions,
    const FlatOrdinaryBirthCommitRows& commits,
    const int64_t cell
) {
    int32_t& status = tape.status.at(cell);
    status = kStatusOk;

    const int64_t decision_count = decisions.count.at(cell);
    const int64_t success_count = commits.count.at(cell);
    if (
        decision_count < 0 || decision_count > decisions.capacity
        || success_count < 0 || success_count > commits.capacity
    ) {
        status = kStatusInvalidSemantics;
        return false;
    }

    int64_t success_ordinal = 0;
    int64_t event_count = 0;
    for (int64_t decision = 0; decision < decision_count; ++decision) {
        const int64_t parent_slot = decisions.parent_slots.at(cell, decision);
        const int32_t reason = decisions.reason_codes.at(cell, decision);
        if (
            parent_slot < 0 || parent_slot >= identity.slot_count
            || identity.slot_to_agent_id.at(cell, parent_slot) < 0
            || identity.lineage_depth.at(cell, parent_slot) < 0
        ) {
            status = kStatusInvalidSemantics;
            return false;
        }

        if (reason == 0) {
            if (success_ordinal >= success_count) {
                status = kStatusInvalidSemantics;
                return false;
            }
            const int64_t committed_parent_slot = commits.parent_slots.at(
                cell,
                success_ordinal
            );
            const int64_t committed_child_slot = commits.child_slots.at(
                cell,
                success_ordinal
            );
            if (
                committed_parent_slot != parent_slot
                || committed_parent_slot < 0
                || committed_parent_slot >= identity.slot_count
                || committed_child_slot < 0
                || committed_child_slot >= identity.slot_count
                || commits.parent_agent_ids.at(cell, success_ordinal) < 0
                || commits.child_agent_ids.at(cell, success_ordinal) < 0
                || identity.slot_to_agent_id.at(cell, committed_parent_slot)
                    != commits.parent_agent_ids.at(cell, success_ordinal)
                || identity.slot_to_agent_id.at(cell, committed_child_slot)
                    != commits.child_agent_ids.at(cell, success_ordinal)
                || identity.lineage_founder_id.at(cell, committed_child_slot)
                    < 0
                || identity.lineage_depth.at(cell, committed_child_slot) < 0
            ) {
                status = kStatusInvalidSemantics;
                return false;
            }
            event_count += commits.mutation_occurred.at(cell, success_ordinal)
                ? 2
                : 1;
            ++success_ordinal;
            continue;
        }
        if (
            (reason >= 1 && reason <= 3)
            || (reason >= 5 && reason <= 6)
        ) {
            ++event_count;
            continue;
        }
        if (reason == 4 || (reason >= 7 && reason <= 11)) {
            continue;
        }
        status = kStatusInvalidSemantics;
        return false;
    }
    if (success_ordinal != success_count) {
        status = kStatusInvalidSemantics;
        return false;
    }

    const int64_t cursor = tape.active_count.at(cell);
    if (
        cursor < 0 || cursor > tape.capacity
        || event_count > tape.capacity - cursor
    ) {
        status = kStatusCapacityExceeded;
        return false;
    }
    return true;
}

// Append a transaction whose complete event semantics and capacity were
// validated before authoritative state mutation. Exactly one device thread
// invokes this function for one cell; it performs no synchronization and has
// no failure path.
//
// Row order is decision order. Failed reasons 1..3 and 5..6 emit one failure
// row, reason 4 and reasons 7..11 emit no row, and a successful decision emits mutation first
// when present followed by its committed birth row.
__device__ __forceinline__ void append_preflighted_ordinary_birth_events_serial(
    const FlatEventTapeColumns& columns,
    const FlatEventTapeState& tape,
    const FlatOrdinaryBirthEventIdentity& identity,
    const FlatOrdinaryBirthDecisionRows& decisions,
    const FlatOrdinaryBirthCommitRows& commits,
    const int64_t cell,
    const int64_t generation,
    const int64_t tick
) {
    int64_t output_row = tape.active_count.at(cell);
    int64_t success_ordinal = 0;
    const int64_t decision_count = decisions.count.at(cell);
    for (int64_t decision = 0; decision < decision_count; ++decision) {
        const int64_t decision_parent_slot = decisions.parent_slots.at(
            cell,
            decision
        );
        const int64_t decision_parent_id = identity.slot_to_agent_id.at(
            cell,
            decision_parent_slot
        );
        const int32_t reason = decisions.reason_codes.at(cell, decision);
        if (
            (reason >= 1 && reason <= 3)
            || (reason >= 5 && reason <= 6)
        ) {
            const int32_t raw_founder = identity.lineage_founder_id.at(
                cell,
                decision_parent_slot
            );
            detail::publish_event(
                columns,
                cell,
                output_row++,
                kFailedBirthEventType,
                generation,
                tick,
                decision_parent_id,
                kSentinelId,
                decision_parent_slot,
                kSentinelId,
                kSentinelId,
                kSentinelId,
                raw_founder >= 0
                    ? static_cast<int64_t>(raw_founder)
                    : decision_parent_id,
                identity.lineage_depth.at(cell, decision_parent_slot),
                kFailedBirthEventType,
                reason,
                0,
                static_cast<float>(reason)
            );
            continue;
        }
        if (reason != 0) {
            continue;
        }

        const int64_t parent_id = commits.parent_agent_ids.at(
            cell,
            success_ordinal
        );
        const int64_t child_id = commits.child_agent_ids.at(
            cell,
            success_ordinal
        );
        const int64_t parent_slot = commits.parent_slots.at(
            cell,
            success_ordinal
        );
        const int64_t child_slot = commits.child_slots.at(
            cell,
            success_ordinal
        );
        const int64_t founder = static_cast<int64_t>(
            identity.lineage_founder_id.at(cell, child_slot)
        );
        const int32_t depth = identity.lineage_depth.at(cell, child_slot);
        if (commits.mutation_occurred.at(cell, success_ordinal)) {
            detail::publish_event(
                columns,
                cell,
                output_row++,
                kMutationEventType,
                generation,
                tick,
                child_id,
                child_id,
                child_slot,
                child_slot,
                parent_id,
                child_id,
                founder,
                depth,
                kMutationEventType,
                -1,
                1,
                commits.mutation_delta_total.at(cell, success_ordinal)
            );
        }
        detail::publish_event(
            columns,
            cell,
            output_row++,
            kBirthCommittedEventType,
            generation,
            tick,
            parent_id,
            child_id,
            parent_slot,
            child_slot,
            parent_id,
            child_id,
            founder,
            depth,
            kBirthCommittedEventType,
            -1,
            1,
            commits.child_energy.at(cell, success_ordinal)
        );
        ++success_ordinal;
    }
    tape.active_count.at(cell) = output_row;
}

// Exact allocation-free transcription of immutable Stage2's ordinary-birth
// preflight/append pair. Existing callers retain the checked behavior.
__device__ __forceinline__ void append_ordinary_birth_events_serial(
    const FlatEventTapeColumns& columns,
    const FlatEventTapeState& tape,
    const FlatOrdinaryBirthEventIdentity& identity,
    const FlatOrdinaryBirthDecisionRows& decisions,
    const FlatOrdinaryBirthCommitRows& commits,
    const int64_t cell,
    const int64_t generation,
    const int64_t tick
) {
    if (!preflight_ordinary_birth_events_serial(
            tape,
            identity,
            decisions,
            commits,
            cell
        )) {
        return;
    }
    append_preflighted_ordinary_birth_events_serial(
        columns,
        tape,
        identity,
        decisions,
        commits,
        cell,
        generation,
        tick
    );
}

}  // namespace eiel::cuda_architecture_b::persistent_birth_events

#endif  // EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_BIRTH_EVENTS_CUH_
