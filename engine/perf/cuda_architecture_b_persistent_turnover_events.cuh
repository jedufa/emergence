#ifndef EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_TURNOVER_EVENTS_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_TURNOVER_EVENTS_CUH_

#include "cuda_architecture_b_active_identity_pairs_v1.cuh"
#include "cuda_architecture_b_persistent_lifecycle_events.cuh"

#include <cstdint>

namespace eiel::cuda_architecture_b::persistent_turnover_events {

namespace active_identity =
    eiel::cuda_architecture_b::active_identity_pairs_v1;

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
constexpr int64_t kSentinelId = persistent_lifecycle_events::kSentinelId;

// All pointers are already offset to one simulation cell. Identity commit and
// policy inheritance have completed before these rows are consumed.
struct FlatTurnoverPostcommitRows {
    FlatStridedCellRows<const int64_t> parent_agent_ids;
    FlatStridedCellRows<const int64_t> mate_agent_ids;
    FlatStridedCellRows<const int64_t> child_agent_ids;
    FlatStridedCellRows<const int64_t> parent_slots;
    FlatStridedCellRows<const int64_t> mate_slots;
    FlatStridedCellRows<const int64_t> child_slots;
    FlatStridedCellRows<const bool> mutation_occurred;
    FlatStridedCellRows<const float> mutation_delta_total;
    FlatStridedCellRows<const float> child_energy;
    FlatStridedCellScalars<const int64_t> committed_count;
    int64_t capacity;

    // These are the sorted postcommit outputs of the Stage2 identity commit.
    FlatStridedCellRows<const int64_t> alive_agent_ids_by_order;
    FlatStridedCellScalars<const int64_t> alive_count;
    int64_t active_capacity;
};

struct FlatTurnoverPostcommitState {
    FlatStridedCellRows<const int64_t> slot_to_agent_id;
    FlatStridedCellRows<const int32_t> agent_id_to_slot;
    FlatStridedCellRows<const bool> alive_mask;
    FlatStridedCellRows<const int32_t> lineage_founder_id;
    FlatStridedCellRows<const int32_t> lineage_depth;
    int64_t slot_count;
    int64_t agent_id_capacity;

    FlatStridedCellRows<int64_t> active_agent_ids;
    FlatStridedCellScalars<int64_t> active_count;
    int64_t active_capacity;
    FlatStridedCellRows<int32_t> active_agent_slots;
};

struct TurnoverLifecycleConfig {
    int64_t slot_count;
    int64_t agent_id_capacity;
    int64_t active_capacity;
};

namespace detail {

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

__device__ __forceinline__ bool valid_postcommit_rows(
    const FlatTurnoverPostcommitRows& rows,
    const TurnoverLifecycleConfig config
) {
    return rows.capacity >= 0
        && rows.active_capacity >= 0
        && rows.active_capacity <= config.slot_count
        && rows.active_capacity <= config.active_capacity
        && persistent_lifecycle_events::valid_rows(rows.parent_agent_ids)
        && persistent_lifecycle_events::valid_rows(rows.mate_agent_ids)
        && persistent_lifecycle_events::valid_rows(rows.child_agent_ids)
        && persistent_lifecycle_events::valid_rows(rows.parent_slots)
        && persistent_lifecycle_events::valid_rows(rows.mate_slots)
        && persistent_lifecycle_events::valid_rows(rows.child_slots)
        && persistent_lifecycle_events::valid_rows(rows.mutation_occurred)
        && persistent_lifecycle_events::valid_rows(rows.mutation_delta_total)
        && persistent_lifecycle_events::valid_rows(rows.child_energy)
        && persistent_lifecycle_events::valid_scalars(rows.committed_count)
        && persistent_lifecycle_events::valid_rows(
            rows.alive_agent_ids_by_order
        )
        && persistent_lifecycle_events::valid_scalars(rows.alive_count);
}

__device__ __forceinline__ bool valid_postcommit_state(
    const FlatTurnoverPostcommitState& state,
    const TurnoverLifecycleConfig config
) {
    return state.slot_count == config.slot_count
        && state.agent_id_capacity == config.agent_id_capacity
        && state.active_capacity == config.active_capacity
        && state.slot_count > 0
        && state.agent_id_capacity > 0
        && state.active_capacity >= state.slot_count
        && persistent_lifecycle_events::valid_rows(state.slot_to_agent_id)
        && persistent_lifecycle_events::valid_rows(state.alive_mask)
        && persistent_lifecycle_events::valid_rows(state.lineage_founder_id)
        && persistent_lifecycle_events::valid_rows(state.lineage_depth)
        && persistent_lifecycle_events::valid_rows(state.active_agent_ids)
        && persistent_lifecycle_events::valid_rows(state.active_agent_slots)
        && persistent_lifecycle_events::valid_scalars(state.active_count);
}

}  // namespace detail

// The Stage2 identity commit returns sorted live IDs. Copying those rows is the
// persistent active-id/count postcommit, with no host round trip or allocation.
__device__ __forceinline__ bool update_active_ids_from_turnover_postcommit_serial(
    const FlatTurnoverPostcommitState& state,
    const FlatTurnoverPostcommitRows& rows,
    const int64_t cell,
    const int64_t active_count
) {
    for (int64_t row = 0; row < state.active_capacity; ++row) {
        const int64_t agent_id = row < active_count
            ? rows.alive_agent_ids_by_order.at(cell, row) : kSentinelId;
        state.active_agent_ids.at(cell, row) = agent_id;
        state.active_agent_slots.at(cell, row) = -1;
        if (row >= active_count) {
            continue;
        }
        const active_identity::ResolveResultV1 resolved =
            active_identity::resolve_resident_agent_slot_v1(
                state.slot_to_agent_id.data + cell
                    * state.slot_to_agent_id.cell_stride,
                state.alive_mask.data + cell * state.alive_mask.cell_stride,
                state.slot_count,
                agent_id
            );
        if (resolved.status != active_identity::kResolveFound) {
            return false;
        }
        state.active_agent_slots.at(cell, row) =
            static_cast<int32_t>(resolved.slot);
    }
    state.active_count.at(cell) = active_count;
    const active_identity::ActiveIdentityPairsV1 active_pairs{
        state.active_agent_ids.data + cell
            * state.active_agent_ids.cell_stride,
        state.active_agent_slots.data + cell
            * state.active_agent_slots.cell_stride,
        state.slot_to_agent_id.data + cell
            * state.slot_to_agent_id.cell_stride,
        state.alive_mask.data + cell * state.alive_mask.cell_stride,
        active_count,
        state.active_capacity,
        state.slot_count,
    };
    return active_identity::valid_active_identity_pairs_v1(active_pairs);
}

// Validate the identity and lineage state after turnover parent/child commit.
// This keeps event publication fail-closed if a preceding postcommit stage did
// not publish the exact parent, mate, child, or lineage mapping it reported.
__device__ __forceinline__ bool preflight_turnover_lifecycle_events_serial(
    const FlatEventTapeColumns& columns,
    const FlatEventTapeState& tape,
    const FlatTurnoverPostcommitRows& rows,
    const FlatTurnoverPostcommitState& state,
    const TurnoverLifecycleConfig config,
    const int64_t cell,
    const int64_t generation,
    const int64_t tick
) {
    if (cell < 0 || !persistent_lifecycle_events::valid_scalars(tape.status)) {
        return false;
    }
    int32_t& status = tape.status.at(cell);
    status = kStatusOk;
    if (
        tape.capacity < 0
        || !persistent_lifecycle_events::valid_scalars(tape.active_count)
        || !persistent_lifecycle_events::valid_event_columns(columns)
        || !detail::valid_postcommit_rows(rows, config)
        || !detail::valid_postcommit_state(state, config)
        || generation < 0
        || tick < 0
    ) {
        status = kStatusInvalidSemantics;
        return false;
    }

    const int64_t committed_count = rows.committed_count.at(cell);
    const int64_t active_count = rows.alive_count.at(cell);
    if (
        committed_count < 0 || committed_count > rows.capacity
        || active_count < 0 || active_count > rows.active_capacity
        || active_count > config.active_capacity
    ) {
        status = kStatusInvalidSemantics;
        return false;
    }

    int64_t event_count = 0;
    int64_t previous_active_id = kSentinelId;
    for (int64_t row = 0; row < active_count; ++row) {
        const int64_t agent_id = rows.alive_agent_ids_by_order.at(cell, row);
        if (!active_identity::agent_id_fits_grid_v1(agent_id)
            || (row > 0 && agent_id <= previous_active_id)) {
            status = kStatusInvalidSemantics;
            return false;
        }
        const active_identity::ResolveResultV1 resolved =
            active_identity::resolve_resident_agent_slot_v1(
                state.slot_to_agent_id.data
                    + cell * state.slot_to_agent_id.cell_stride,
                state.alive_mask.data + cell * state.alive_mask.cell_stride,
                state.slot_count,
                agent_id
            );
        if (resolved.status != active_identity::kResolveFound) {
            status = kStatusInvalidSemantics;
            return false;
        }
        previous_active_id = agent_id;
    }
    for (int64_t row = active_count; row < rows.active_capacity; ++row) {
        if (rows.alive_agent_ids_by_order.at(cell, row) >= 0) {
            status = kStatusInvalidSemantics;
            return false;
        }
    }

    for (int64_t birth = 0; birth < committed_count; ++birth) {
        const int64_t parent_id = rows.parent_agent_ids.at(cell, birth);
        const int64_t mate_id = rows.mate_agent_ids.at(cell, birth);
        const int64_t child_id = rows.child_agent_ids.at(cell, birth);
        const int64_t parent_slot = rows.parent_slots.at(cell, birth);
        const int64_t mate_slot = rows.mate_slots.at(cell, birth);
        const int64_t child_slot = rows.child_slots.at(cell, birth);
        if (
            !active_identity::agent_id_fits_grid_v1(parent_id)
            || !active_identity::agent_id_fits_grid_v1(mate_id)
            || !active_identity::agent_id_fits_grid_v1(child_id)
            || parent_slot < 0 || parent_slot >= config.slot_count
            || mate_slot < 0 || mate_slot >= config.slot_count
            || child_slot < 0 || child_slot >= config.slot_count
            || parent_slot == mate_slot || parent_slot == child_slot
            || mate_slot == child_slot
            || state.slot_to_agent_id.at(cell, parent_slot) != parent_id
            || state.slot_to_agent_id.at(cell, mate_slot) != mate_id
            || state.slot_to_agent_id.at(cell, child_slot) != child_id
            || !state.alive_mask.at(cell, parent_slot)
            || !state.alive_mask.at(cell, mate_slot)
            || !state.alive_mask.at(cell, child_slot)
            || state.lineage_founder_id.at(cell, child_slot) < 0
            || state.lineage_depth.at(cell, child_slot) < 0
        ) {
            status = kStatusInvalidSemantics;
            return false;
        }
        for (int64_t previous = 0; previous < birth; ++previous) {
            if (
                rows.child_slots.at(cell, previous) == child_slot
                || rows.child_agent_ids.at(cell, previous) == child_id
            ) {
                status = kStatusInvalidSemantics;
                return false;
            }
        }
        event_count += rows.mutation_occurred.at(cell, birth) ? 2 : 1;
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

// Exact allocation-free turnover publication. Row order is committed turnover
// order; mutation is emitted immediately before its birth row. The event cursor
// and active-id/count state change only after all postcommit checks pass.
__device__ __forceinline__ void append_turnover_lifecycle_events_serial(
    const FlatEventTapeColumns& columns,
    const FlatEventTapeState& tape,
    const FlatTurnoverPostcommitRows& rows,
    const FlatTurnoverPostcommitState& state,
    const TurnoverLifecycleConfig config,
    const int64_t cell,
    const int64_t generation,
    const int64_t tick
) {
    if (!preflight_turnover_lifecycle_events_serial(
            columns, tape, rows, state, config, cell, generation, tick
        )) {
        return;
    }

    const int64_t active_count = rows.alive_count.at(cell);
    int64_t output_row = tape.active_count.at(cell);
    for (int64_t birth = 0; birth < rows.committed_count.at(cell); ++birth) {
        const int64_t parent_id = rows.parent_agent_ids.at(cell, birth);
        const int64_t child_id = rows.child_agent_ids.at(cell, birth);
        const int64_t parent_slot = rows.parent_slots.at(cell, birth);
        const int64_t child_slot = rows.child_slots.at(cell, birth);
        const int64_t founder = static_cast<int64_t>(
            state.lineage_founder_id.at(cell, child_slot)
        );
        const int32_t depth = state.lineage_depth.at(cell, child_slot);
        if (rows.mutation_occurred.at(cell, birth)) {
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
                rows.mutation_delta_total.at(cell, birth)
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
            rows.child_energy.at(cell, birth)
        );
    }
    tape.active_count.at(cell) = output_row;
    if (!update_active_ids_from_turnover_postcommit_serial(
            state, rows, cell, active_count
        )) {
        tape.status.at(cell) = kStatusInvalidSemantics;
        return;
    }
}

}  // namespace eiel::cuda_architecture_b::persistent_turnover_events

#endif  // EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_TURNOVER_EVENTS_CUH_
