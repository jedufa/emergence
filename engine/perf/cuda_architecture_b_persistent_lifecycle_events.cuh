#ifndef EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_LIFECYCLE_EVENTS_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_LIFECYCLE_EVENTS_CUH_

#include <cuda_runtime.h>

#include <cstdint>

namespace eiel::cuda_architecture_b::persistent_lifecycle_events {

constexpr int32_t kStatusOk = 0;
constexpr int32_t kStatusCapacityExceeded = 1;
constexpr int32_t kStatusInvalidSemantics = 2;
constexpr int32_t kDeathCommittedEventType = 4;
constexpr int64_t kSentinelId = -1;

template <typename T>
struct FlatStridedCellRows {
    T* data;
    int64_t cell_stride;
    int64_t row_stride;

    __device__ __forceinline__ T& at(
        const int64_t cell,
        const int64_t row
    ) const {
        return data[cell * cell_stride + row * row_stride];
    }
};

template <typename T>
struct FlatStridedCellScalars {
    T* data;
    int64_t cell_stride;

    __device__ __forceinline__ T& at(const int64_t cell) const {
        return data[cell * cell_stride];
    }
};

struct FlatEventTapeColumns {
    FlatStridedCellRows<int32_t> event_type;
    FlatStridedCellRows<int64_t> generation;
    FlatStridedCellRows<int64_t> tick;
    FlatStridedCellRows<int64_t> actor_id;
    FlatStridedCellRows<int64_t> target_id;
    FlatStridedCellRows<float> value;
    FlatStridedCellRows<int32_t> typed_event_type;
    FlatStridedCellRows<int64_t> typed_generation;
    FlatStridedCellRows<int64_t> typed_tick;
    FlatStridedCellRows<int64_t> typed_sequence;
    FlatStridedCellRows<int64_t> typed_actor_id;
    FlatStridedCellRows<int64_t> typed_target_id;
    FlatStridedCellRows<int64_t> typed_actor_slot;
    FlatStridedCellRows<int64_t> typed_target_slot;
    FlatStridedCellRows<int64_t> typed_parent_id;
    FlatStridedCellRows<int64_t> typed_child_id;
    FlatStridedCellRows<int64_t> typed_founder_id;
    FlatStridedCellRows<int32_t> typed_lineage_depth;
    FlatStridedCellRows<int32_t> typed_subtype;
    FlatStridedCellRows<int32_t> typed_reason;
    FlatStridedCellRows<int8_t> typed_success;
    FlatStridedCellRows<float> typed_value;
};

struct FlatEventTapeState {
    FlatStridedCellScalars<int64_t> active_count;
    FlatStridedCellScalars<int32_t> status;
    int64_t capacity;
};

struct FlatDeathRows {
    FlatStridedCellRows<const int64_t> agent_ids;
    FlatStridedCellRows<const int64_t> slots;
    FlatStridedCellRows<const int32_t> reason_codes;
    FlatStridedCellScalars<const int64_t> count;
    int64_t capacity;
};

struct FlatLineageRows {
    FlatStridedCellRows<const int32_t> founder_id;
    FlatStridedCellRows<const int32_t> depth;
    int64_t slot_count;
};

struct FlatActionTerminalReporting {
    FlatStridedCellRows<const int64_t> initial_id_to_row;
    FlatStridedCellRows<const int32_t> survival_steps;
    FlatStridedCellRows<int32_t> terminal_survival;
    FlatStridedCellRows<int32_t> terminal_rows;
    int64_t agent_id_capacity;
    int64_t slot_count;
    int64_t reporting_row_count;
};

template <typename T>
__device__ __forceinline__ bool valid_rows(
    const FlatStridedCellRows<T>& rows
) {
    return rows.data != nullptr
        && rows.cell_stride >= 0
        && rows.row_stride > 0;
}

template <typename T>
__device__ __forceinline__ bool valid_scalars(
    const FlatStridedCellScalars<T>& scalars
) {
    return scalars.data != nullptr && scalars.cell_stride >= 0;
}

__device__ __forceinline__ bool valid_event_columns(
    const FlatEventTapeColumns& columns
) {
    return valid_rows(columns.event_type)
        && valid_rows(columns.generation)
        && valid_rows(columns.tick)
        && valid_rows(columns.actor_id)
        && valid_rows(columns.target_id)
        && valid_rows(columns.value)
        && valid_rows(columns.typed_event_type)
        && valid_rows(columns.typed_generation)
        && valid_rows(columns.typed_tick)
        && valid_rows(columns.typed_sequence)
        && valid_rows(columns.typed_actor_id)
        && valid_rows(columns.typed_target_id)
        && valid_rows(columns.typed_actor_slot)
        && valid_rows(columns.typed_target_slot)
        && valid_rows(columns.typed_parent_id)
        && valid_rows(columns.typed_child_id)
        && valid_rows(columns.typed_founder_id)
        && valid_rows(columns.typed_lineage_depth)
        && valid_rows(columns.typed_subtype)
        && valid_rows(columns.typed_reason)
        && valid_rows(columns.typed_success)
        && valid_rows(columns.typed_value);
}

__device__ __forceinline__ bool valid_death_rows(
    const FlatDeathRows& deaths
) {
    return deaths.capacity >= 0
        && valid_rows(deaths.agent_ids)
        && valid_rows(deaths.slots)
        && valid_rows(deaths.reason_codes)
        && valid_scalars(deaths.count);
}

__device__ __forceinline__ bool valid_lineage_rows(
    const FlatLineageRows& lineage
) {
    return lineage.slot_count > 0
        && valid_rows(lineage.founder_id)
        && valid_rows(lineage.depth);
}

__device__ __forceinline__ bool is_sticky_failure(const int32_t status) {
    return status == kStatusCapacityExceeded
        || status == kStatusInvalidSemantics;
}

// Provenance: exact field values and write order from
// native/cuda_architecture_b_death_lifecycle_events.cu:155-188.
__device__ __forceinline__ void publish_death_event(
    const FlatEventTapeColumns& columns,
    const int64_t cell,
    const int64_t row,
    const int64_t generation,
    const int64_t absolute_tick,
    const int64_t agent_id,
    const int64_t slot,
    const int64_t founder_id,
    const int32_t lineage_depth,
    const int32_t reason
) {
    const float value = static_cast<float>(reason);
    columns.event_type.at(cell, row) = kDeathCommittedEventType;
    columns.generation.at(cell, row) = generation;
    columns.tick.at(cell, row) = absolute_tick;
    columns.actor_id.at(cell, row) = agent_id;
    columns.target_id.at(cell, row) = kSentinelId;
    columns.value.at(cell, row) = value;
    columns.typed_event_type.at(cell, row) = kDeathCommittedEventType;
    columns.typed_generation.at(cell, row) = generation;
    columns.typed_tick.at(cell, row) = absolute_tick;
    columns.typed_sequence.at(cell, row) = row;
    columns.typed_actor_id.at(cell, row) = agent_id;
    columns.typed_target_id.at(cell, row) = kSentinelId;
    columns.typed_actor_slot.at(cell, row) = slot;
    columns.typed_target_slot.at(cell, row) = kSentinelId;
    columns.typed_parent_id.at(cell, row) = kSentinelId;
    columns.typed_child_id.at(cell, row) = kSentinelId;
    columns.typed_founder_id.at(cell, row) = founder_id;
    columns.typed_lineage_depth.at(cell, row) = lineage_depth;
    columns.typed_subtype.at(cell, row) = kDeathCommittedEventType;
    columns.typed_reason.at(cell, row) = reason;
    columns.typed_success.at(cell, row) = 1;
    columns.typed_value.at(cell, row) = value;
}

// Preflight one accepted death class before any event column or cursor write.
// Status values 1 and 2 remain sticky; another incoming value is the accepted
// standalone-call sentinel and is normalized to zero.
__device__ __forceinline__ bool preflight_death_lifecycle_events(
    const FlatEventTapeColumns& columns,
    const FlatEventTapeState& tape,
    const FlatDeathRows& deaths,
    const FlatLineageRows& lineage,
    const int64_t cell,
    const int64_t generation,
    const int64_t absolute_tick
) {
    if (cell < 0 || !valid_scalars(tape.status)) {
        return false;
    }
    int32_t& status = tape.status.at(cell);
    if (is_sticky_failure(status)) {
        return false;
    }
    status = kStatusOk;

    if (
        tape.capacity < 0
        || !valid_scalars(tape.active_count)
        || !valid_event_columns(columns)
        || !valid_death_rows(deaths)
        || !valid_lineage_rows(lineage)
        || generation < 0
        || absolute_tick < 0
    ) {
        status = kStatusInvalidSemantics;
        return false;
    }

    const int64_t count = deaths.count.at(cell);
    if (count < 0 || count > deaths.capacity) {
        status = kStatusInvalidSemantics;
        return false;
    }

    for (int64_t death = 0; death < count; ++death) {
        const int64_t agent_id = deaths.agent_ids.at(cell, death);
        const int64_t slot = deaths.slots.at(cell, death);
        const int32_t reason = deaths.reason_codes.at(cell, death);
        if (
            agent_id < 0
            || slot < 0
            || slot >= lineage.slot_count
            || (reason != 1 && reason != 2)
            || lineage.founder_id.at(cell, slot) < 0
            || lineage.depth.at(cell, slot) < 0
        ) {
            status = kStatusInvalidSemantics;
            return false;
        }
        for (int64_t previous = 0; previous < death; ++previous) {
            if (
                deaths.agent_ids.at(cell, previous) == agent_id
                || deaths.slots.at(cell, previous) == slot
            ) {
                status = kStatusInvalidSemantics;
                return false;
            }
        }
    }

    const int64_t cursor = tape.active_count.at(cell);
    if (
        cursor < 0
        || cursor > tape.capacity
        || count > tape.capacity - cursor
    ) {
        status = kStatusCapacityExceeded;
        return false;
    }
    return true;
}

// Exact raw-device transcription of the accepted preflight/append pair in
// native/cuda_architecture_b_death_lifecycle_events.cu:94-224. Invoke from
// thread 0; the function performs no synchronization.
__device__ __forceinline__ void append_death_lifecycle_events(
    const FlatEventTapeColumns& columns,
    const FlatEventTapeState& tape,
    const FlatDeathRows& deaths,
    const FlatLineageRows& lineage,
    const int64_t cell,
    const int64_t generation,
    const int64_t absolute_tick
) {
    if (!preflight_death_lifecycle_events(
            columns,
            tape,
            deaths,
            lineage,
            cell,
            generation,
            absolute_tick
        )) {
        return;
    }

    const int64_t cursor = tape.active_count.at(cell);
    const int64_t count = deaths.count.at(cell);
    for (int64_t death = 0; death < count; ++death) {
        const int64_t slot = deaths.slots.at(cell, death);
        publish_death_event(
            columns,
            cell,
            cursor + death,
            generation,
            absolute_tick,
            deaths.agent_ids.at(cell, death),
            slot,
            static_cast<int64_t>(lineage.founder_id.at(cell, slot)),
            lineage.depth.at(cell, slot),
            deaths.reason_codes.at(cell, death)
        );
    }
    tape.active_count.at(cell) = cursor + count;
}

// Provenance: action-terminal reporting from
// native/cuda_architecture_b_generation.cpp:1474-1511. Deaths outside the
// initial reporting cohort do not contribute, matching the accepted mask.
__device__ __forceinline__ void accumulate_action_terminal_reporting(
    const FlatDeathRows& action_deaths,
    const FlatActionTerminalReporting& reporting,
    const int64_t cell
) {
    if (
        cell < 0
        || !valid_death_rows(action_deaths)
        || !valid_rows(reporting.initial_id_to_row)
        || !valid_rows(reporting.survival_steps)
        || !valid_rows(reporting.terminal_survival)
        || !valid_rows(reporting.terminal_rows)
        || reporting.agent_id_capacity <= 0
        || reporting.slot_count <= 0
        || reporting.reporting_row_count < 0
    ) {
        return;
    }

    const int64_t count = action_deaths.count.at(cell);
    if (count <= 0 || count > action_deaths.capacity) {
        return;
    }
    for (int64_t death = 0; death < count; ++death) {
        const int64_t agent_id = action_deaths.agent_ids.at(cell, death);
        if (agent_id < 0 || agent_id >= reporting.agent_id_capacity) {
            continue;
        }
        const int64_t reporting_row = reporting.initial_id_to_row.at(
            cell,
            agent_id
        );
        const int64_t slot = action_deaths.slots.at(cell, death);
        if (
            reporting_row < 0
            || reporting_row >= reporting.reporting_row_count
            || slot < 0
            || slot >= reporting.slot_count
        ) {
            continue;
        }
        reporting.terminal_survival.at(cell, reporting_row) +=
            reporting.survival_steps.at(cell, slot) + 1;
        reporting.terminal_rows.at(cell, reporting_row) += 1;
    }
}

// Accepted boundary order from cuda_architecture_b_generation.cpp:1465-1529
// and 1727-1751: action terminal report, action event class, residual event
// class. Each class has its own atomic fail-closed preflight; therefore an
// accepted action append remains committed if the residual append later fails.
__device__ __forceinline__ void publish_persistent_lifecycle_death_boundary(
    const FlatEventTapeColumns& columns,
    const FlatEventTapeState& tape,
    const FlatDeathRows& action_deaths,
    const FlatDeathRows& residual_deaths,
    const FlatLineageRows& lineage,
    const FlatActionTerminalReporting& reporting,
    const int64_t cell,
    const int64_t generation,
    const int64_t absolute_tick
) {
    accumulate_action_terminal_reporting(action_deaths, reporting, cell);
    append_death_lifecycle_events(
        columns,
        tape,
        action_deaths,
        lineage,
        cell,
        generation,
        absolute_tick
    );
    append_death_lifecycle_events(
        columns,
        tape,
        residual_deaths,
        lineage,
        cell,
        generation,
        absolute_tick
    );
}

}  // namespace eiel::cuda_architecture_b::persistent_lifecycle_events

#endif  // EIEL_CUDA_ARCHITECTURE_B_PERSISTENT_LIFECYCLE_EVENTS_CUH_
