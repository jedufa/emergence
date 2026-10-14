#pragma once

#if defined(__CUDACC__)
#include <cuda_runtime_api.h>
#else
struct CUstream_st;
using cudaStream_t = CUstream_st*;
#endif

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace eiel::cuda_architecture_b::compact_summary_reporting_authority_v1 {

inline constexpr char kContractVersionV1[] =
    "eiel_cuda_architecture_b_compact_summary_reporting_authority_v1";
inline constexpr char kRetainedLayoutVersionV2[] =
    "eiel_cuda_architecture_b_compact_summary_reporting_retained_layout_v2";
inline constexpr std::int32_t kActionCountV1 = 7;
inline constexpr std::int32_t kStrictRowCapacityV1 = 64;
inline constexpr std::int32_t kDynamicRowCapacityV1 = 224;
inline constexpr std::int32_t kMaximumDynamicRowCapacityV1 = 31104;
inline constexpr std::int32_t kThreadsPerBlockV1 = 32;

enum LedgerStatusV1 : std::uint32_t {
    kLedgerEmptyV1 = 0,
    kLedgerOpenV1 = 1,
    kLedgerSealedV1 = 2,
    kLedgerInvalidStorageV1 = 100,
    kLedgerInvalidInputV1 = 101,
    kLedgerOverflowV1 = 102,
    kLedgerDuplicateAgentIdV1 = 103,
    kLedgerIdentityOrderV1 = 104,
    kLedgerSlotAliasV1 = 105,
    kLedgerUnknownAgentV1 = 106,
    kLedgerInvalidActionV1 = 107,
    kLedgerCounterOverflowV1 = 108,
    kLedgerWriteAfterSealV1 = 109,
    kLedgerIncompleteTerminalV1 = 110,
    kLedgerInvalidFitnessConfigV1 = 111,
    kLedgerInvalidLineageSourceV1 = 112,
};

enum BehaviorFlagsV1 : std::uint8_t {
    kBehaviorAliveEndV1 = 1U << 0,
    kBehaviorEligibleForSelectionV1 = 1U << 1,
    kBehaviorKeptNextV1 = 1U << 2,
    kBehaviorTerminalSealedV1 = 1U << 3,
};

struct alignas(8) GenerationAuthorityHeaderV1 {
    std::int64_t start_id_ceiling;
    std::uint16_t behavior_row_count;
    std::uint16_t lineage_record_count;
    std::uint32_t status;
};

struct alignas(8) BehaviorAuthorityRowV1 {
    std::int64_t agent_id;
    double energy_collected_exact;
    double base_fitness;
    float injury_start;
    float injury_end;
    float aggression_bias_f32;
    std::int32_t lifespan_steps;
    std::int32_t offspring;
    std::int32_t attacks_made;
    std::uint8_t action_counts[kActionCountV1];
    std::uint8_t flags;
};

struct alignas(8) LineageAuthorityRowV1 {
    std::int64_t agent_id;
    double energy;
    double energy_collected_exact;
    double episode0_pred_err_mean;
    double identity_drift_ema;
    float health;
    float injury;
    float drives[4];
    float pred_err_sum;
    float drive_jitter;
    float aggression_bias_f32;
    float exploration_bias_f32;
    std::int32_t y;
    std::int32_t x;
    std::int32_t founder_id;
    std::int32_t birth_generation;
    std::int32_t birth_tick;
    std::int32_t lineage_depth;
    std::int32_t offspring_count;
    std::int32_t survival_steps;
    std::int32_t first_repro_step;
    std::int32_t pred_err_count;
    std::int32_t patch_habitat_lineage_id;
    std::int32_t hidden_size;
    std::int32_t memory_dim;
    std::uint8_t use_gru;
    std::uint8_t reserved[3];
};

struct BehaviorTerminalValuesV1 {
    double energy_collected_exact;
    double early_energy_collected_exact;
    float injury_end;
    float aggression_bias_f32;
    float early_injury_gained;
    float early_life_weight;
    float drive_jitter;
    float pred_err_sum;
    float birth_reset_shock;
    float episode0_pred_err_sum;
    float late_pred_err_sum;
    std::int32_t lifespan_steps;
    std::int32_t offspring;
    std::int32_t attacks_made;
    std::int32_t early_survival_steps;
    std::int32_t action_switches;
    std::int32_t action_oscillations;
    std::int32_t selfharm_events;
    std::int32_t pred_err_count;
    std::int32_t episode0_pred_err_count;
    std::int32_t episode0_selfharm_events;
    std::int32_t late_pred_err_count;
};

struct ReportingFitnessConfigV1 {
    double parent_fraction;
    double episode0_fitness_weight;
    double episode0_pred_err_fitness_weight;
    double episode0_thrashing_fitness_weight;
    double episode0_drive_jitter_fitness_weight;
    double episode0_selfharm_fitness_weight;
    double coherence_thrashing_weight;
    double coherence_drive_jitter_weight;
    double coherence_selfharm_weight;
    double coherence_pred_err_weight;
    double coherence_reset_shock_weight;
    double coherence_episode0_pred_err_weight;
    double late_pred_cost_weight;
    std::uint8_t late_pred_cost_enable;
    std::uint8_t disable_prediction_error_selection_pressure;
    std::uint8_t patch_habitat_lineage_available;
    std::uint8_t reserved[5];
};

struct ResidentGenerationSourcesV1 {
    const std::int64_t* slot_to_agent_id;
    const double* energy_collected_exact;
    const double* early_energy_collected_exact;
    const float* injury;
    const float* aggression_bias;
    const float* early_injury_gained;
    const float* early_life_weight;
    const float* drive_jitter;
    const float* pred_err_sum;
    const float* birth_reset_shock;
    const float* episode0_pred_err_sum;
    const float* late_pred_err_sum;
    const std::int32_t* lifespan_steps;
    const std::int32_t* offspring_count;
    const std::int32_t* attacks_made;
    const std::int32_t* early_survival_steps;
    const std::int32_t* action_switches;
    const std::int32_t* action_oscillations;
    const std::int32_t* selfharm_events;
    const std::int32_t* pred_err_count;
    const std::int32_t* episode0_pred_err_count;
    const std::int32_t* episode0_selfharm_events;
    const std::int32_t* late_pred_err_count;
    const double* policy_energy_observation;
    const float* health;
    const float* drives;
    const float* identity_drift_ema;
    const float* exploration_bias;
    const std::int32_t* pos_y;
    const std::int32_t* pos_x;
    const std::int32_t* lineage_founder_id;
    const std::int32_t* lineage_birth_generation;
    const std::int32_t* lineage_birth_tick;
    const std::int32_t* lineage_depth;
    const std::int32_t* first_repro_step;
    const std::int32_t* patch_habitat_lineage_id;
    const std::int32_t* genome_hidden_size;
    const std::int32_t* genome_memory_dim;
    const std::uint8_t* policy_use_gru;
    std::uint8_t identity_drift_ema_available;
    std::int32_t slot_capacity;
};

struct GenerationAuthorityAccessV1 {
    GenerationAuthorityHeaderV1* header;
    BehaviorAuthorityRowV1* behavior_rows;
    LineageAuthorityRowV1* lineage_rows;
    std::int32_t* slot_to_behavior_row;
    double* effective_fitness_by_row;
    std::int32_t row_capacity;
    std::int32_t slot_capacity;
};

struct KernelResourceReceiptV1 {
    std::int32_t binary_version;
    std::int32_t ptx_version;
    std::int32_t max_threads_per_block;
    std::int32_t registers_per_thread;
    std::int32_t occupancy_blocks_per_multiprocessor;
    std::int64_t constant_bytes;
    std::int64_t local_bytes_per_thread;
    std::int64_t shared_bytes_per_block;
};

enum QualificationFixtureV1 : std::int32_t {
    kFixtureStartBornDiedReuseV1 = 0,
    kFixtureEmptyV1 = 1,
    kFixtureOverflowV1 = 2,
    kFixtureDuplicateV1 = 3,
    kFixtureSignedZeroV1 = 4,
    kFixtureG2IsolationV1 = 5,
    kFixtureWriteAfterSealV1 = 6,
    kFixtureIdentityDriftUnavailableV1 = 7,
    kFixtureMissingLineageSourceV1 = 8,
    kFixtureMalformedLineageSourceV1 = 9,
};

inline constexpr std::size_t retained_bytes_per_generation_cell_v1(
    const std::int32_t row_capacity
) noexcept {
    return sizeof(GenerationAuthorityHeaderV1)
        + static_cast<std::size_t>(row_capacity)
            * (sizeof(BehaviorAuthorityRowV1) + sizeof(LineageAuthorityRowV1));
}

inline constexpr std::size_t scratch_bytes_per_cell_v1(
    const std::int32_t row_capacity,
    const std::int32_t slot_capacity
) noexcept {
    return static_cast<std::size_t>(slot_capacity) * sizeof(std::int32_t)
        + static_cast<std::size_t>(row_capacity) * sizeof(double);
}

#if defined(__CUDACC__)
#define EIEL_REPORTING_AUTHORITY_INLINE __device__ __forceinline__
#else
#define EIEL_REPORTING_AUTHORITY_INLINE inline
#endif

EIEL_REPORTING_AUTHORITY_INLINE GenerationAuthorityAccessV1 access_at_row_v1(
    std::uint8_t* const retained_bytes,
    std::int32_t* const slot_to_behavior_row,
    double* const effective_fitness_by_row,
    const std::int64_t row_index,
    const std::int32_t row_capacity,
    const std::int32_t slot_capacity
) noexcept {
    const std::size_t retained_stride = sizeof(GenerationAuthorityHeaderV1)
        + static_cast<std::size_t>(row_capacity)
            * (sizeof(BehaviorAuthorityRowV1) + sizeof(LineageAuthorityRowV1));
    std::uint8_t* const row_base = retained_bytes
        + static_cast<std::size_t>(row_index) * retained_stride;
    auto* const header =
        reinterpret_cast<GenerationAuthorityHeaderV1*>(row_base);
    auto* const behavior = reinterpret_cast<BehaviorAuthorityRowV1*>(
        row_base + sizeof(GenerationAuthorityHeaderV1)
    );
    auto* const lineage = reinterpret_cast<LineageAuthorityRowV1*>(
        row_base + sizeof(GenerationAuthorityHeaderV1)
        + static_cast<std::size_t>(row_capacity)
            * sizeof(BehaviorAuthorityRowV1)
    );
    return {
        header,
        behavior,
        lineage,
        slot_to_behavior_row + row_index * slot_capacity,
        effective_fitness_by_row + row_index * row_capacity,
        row_capacity,
        slot_capacity,
    };
}

void launch_reporting_authority_fixture_v1(
    std::uint8_t* retained_bytes,
    std::int32_t* slot_to_behavior_row,
    double* effective_fitness_by_row,
    std::int64_t generations,
    std::int32_t row_capacity,
    std::int32_t slot_capacity,
    std::int32_t fixture,
    cudaStream_t stream
);

KernelResourceReceiptV1 query_reporting_authority_fixture_v1_resources();

#define EIEL_REPORTING_AUTHORITY_DEVICE EIEL_REPORTING_AUTHORITY_INLINE

EIEL_REPORTING_AUTHORITY_DEVICE bool finite_double_v1(const double value) noexcept {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

EIEL_REPORTING_AUTHORITY_DEVICE bool finite_float_v1(const float value) noexcept {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

EIEL_REPORTING_AUTHORITY_DEVICE bool sticky_error_v1(
    const std::uint32_t status
) noexcept {
    return status >= kLedgerInvalidStorageV1;
}

EIEL_REPORTING_AUTHORITY_DEVICE void record_failure_v1(
    GenerationAuthorityHeaderV1* const header,
    const LedgerStatusV1 status
) noexcept {
    if (header != nullptr && !sticky_error_v1(header->status)) {
        header->status = static_cast<std::uint32_t>(status);
    }
}

EIEL_REPORTING_AUTHORITY_DEVICE bool require_open_v1(
    GenerationAuthorityHeaderV1* const header
) noexcept {
    if (header == nullptr) {
        return false;
    }
    if (sticky_error_v1(header->status)) {
        return false;
    }
    if (header->status == kLedgerSealedV1) {
        header->status = kLedgerWriteAfterSealV1;
        return false;
    }
    return header->status == kLedgerOpenV1;
}

template <typename T>
EIEL_REPORTING_AUTHORITY_DEVICE void clear_pod_v1(T* const value) noexcept {
    std::uint8_t* const bytes = reinterpret_cast<std::uint8_t*>(value);
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        bytes[index] = 0;
    }
}

EIEL_REPORTING_AUTHORITY_DEVICE bool valid_access_v1(
    const GenerationAuthorityAccessV1& access
) noexcept {
    return access.header != nullptr
        && access.behavior_rows != nullptr
        && access.lineage_rows != nullptr
        && access.slot_to_behavior_row != nullptr
        && access.effective_fitness_by_row != nullptr
        && access.row_capacity > 0
        && access.row_capacity <= kMaximumDynamicRowCapacityV1
        && access.slot_capacity > 0;
}

EIEL_REPORTING_AUTHORITY_DEVICE bool initialize_generation_v1(
    const GenerationAuthorityAccessV1& access,
    const std::int64_t start_id_ceiling,
    const std::int64_t* const ascending_active_agent_ids,
    const std::int32_t* const active_slots,
    const float* const start_injury,
    const std::int32_t active_count
) noexcept {
    if (!valid_access_v1(access)) {
        record_failure_v1(access.header, kLedgerInvalidStorageV1);
        return false;
    }
    clear_pod_v1(access.header);
    access.header->start_id_ceiling = start_id_ceiling;
    access.header->status = kLedgerOpenV1;
    for (std::int32_t slot = 0; slot < access.slot_capacity; ++slot) {
        access.slot_to_behavior_row[slot] = -1;
    }
    for (std::int32_t row = 0; row < access.row_capacity; ++row) {
        clear_pod_v1(access.behavior_rows + row);
        clear_pod_v1(access.lineage_rows + row);
        access.effective_fitness_by_row[row] = 0.0;
    }
    if (start_id_ceiling < 0 || active_count < 0
        || active_count > access.row_capacity
        || (active_count > 0
            && (ascending_active_agent_ids == nullptr || active_slots == nullptr
                || start_injury == nullptr))) {
        record_failure_v1(access.header, kLedgerInvalidInputV1);
        return false;
    }
    std::int64_t previous_id = -1;
    for (std::int32_t row = 0; row < active_count; ++row) {
        const std::int64_t agent_id = ascending_active_agent_ids[row];
        const std::int32_t slot = active_slots[row];
        if (agent_id == previous_id) {
            record_failure_v1(access.header, kLedgerDuplicateAgentIdV1);
            return false;
        }
        if (agent_id < 0 || agent_id >= start_id_ceiling
            || agent_id < previous_id) {
            record_failure_v1(access.header, kLedgerIdentityOrderV1);
            return false;
        }
        if (slot < 0 || slot >= access.slot_capacity
            || access.slot_to_behavior_row[slot] != -1
            || !finite_float_v1(start_injury[row])) {
            record_failure_v1(access.header, kLedgerSlotAliasV1);
            return false;
        }
        BehaviorAuthorityRowV1& target = access.behavior_rows[row];
        target.agent_id = agent_id;
        target.injury_start = start_injury[row];
        target.flags = kBehaviorAliveEndV1;
        access.slot_to_behavior_row[slot] = row;
        previous_id = agent_id;
    }
    access.header->behavior_row_count = static_cast<std::uint16_t>(active_count);
    return true;
}

EIEL_REPORTING_AUTHORITY_DEVICE bool initialize_generation_from_slot_state_v1(
    const GenerationAuthorityAccessV1& access,
    const std::int64_t start_id_ceiling,
    const std::int64_t* const ascending_active_agent_ids,
    const std::int32_t* const agent_id_to_slot,
    const float* const injury_by_slot,
    const std::int32_t active_count,
    const std::int32_t agent_id_capacity
) noexcept {
    if (!valid_access_v1(access)) {
        record_failure_v1(access.header, kLedgerInvalidStorageV1);
        return false;
    }
    clear_pod_v1(access.header);
    access.header->start_id_ceiling = start_id_ceiling;
    access.header->status = kLedgerOpenV1;
    for (std::int32_t slot = 0; slot < access.slot_capacity; ++slot) {
        access.slot_to_behavior_row[slot] = -1;
    }
    for (std::int32_t row = 0; row < access.row_capacity; ++row) {
        clear_pod_v1(access.behavior_rows + row);
        clear_pod_v1(access.lineage_rows + row);
        access.effective_fitness_by_row[row] = 0.0;
    }
    if (start_id_ceiling < 0 || active_count < 0
        || active_count > access.row_capacity || agent_id_capacity <= 0
        || (active_count > 0
            && (ascending_active_agent_ids == nullptr
                || agent_id_to_slot == nullptr || injury_by_slot == nullptr))) {
        record_failure_v1(access.header, kLedgerInvalidInputV1);
        return false;
    }
    std::int64_t previous_id = -1;
    for (std::int32_t row = 0; row < active_count; ++row) {
        const std::int64_t agent_id = ascending_active_agent_ids[row];
        if (agent_id < 0 || agent_id >= start_id_ceiling
            || agent_id >= agent_id_capacity || agent_id <= previous_id) {
            record_failure_v1(access.header, kLedgerIdentityOrderV1);
            return false;
        }
        const std::int32_t slot = agent_id_to_slot[agent_id];
        if (slot < 0 || slot >= access.slot_capacity
            || access.slot_to_behavior_row[slot] != -1
            || !finite_float_v1(injury_by_slot[slot])) {
            record_failure_v1(access.header, kLedgerSlotAliasV1);
            return false;
        }
        BehaviorAuthorityRowV1& target = access.behavior_rows[row];
        target.agent_id = agent_id;
        target.injury_start = injury_by_slot[slot];
        target.flags = kBehaviorAliveEndV1;
        access.slot_to_behavior_row[slot] = row;
        previous_id = agent_id;
    }
    access.header->behavior_row_count = static_cast<std::uint16_t>(active_count);
    return true;
}

EIEL_REPORTING_AUTHORITY_DEVICE bool initialize_generation_from_active_slot_state_v1(
    const GenerationAuthorityAccessV1& access,
    const std::int64_t start_id_ceiling,
    const std::int64_t* const ascending_active_agent_ids,
    const std::int32_t* const active_agent_slots,
    const float* const injury_by_slot,
    const std::int32_t active_count
) noexcept {
    if (!valid_access_v1(access)) {
        record_failure_v1(access.header, kLedgerInvalidStorageV1);
        return false;
    }
    clear_pod_v1(access.header);
    access.header->start_id_ceiling = start_id_ceiling;
    access.header->status = kLedgerOpenV1;
    for (std::int32_t slot = 0; slot < access.slot_capacity; ++slot) {
        access.slot_to_behavior_row[slot] = -1;
    }
    for (std::int32_t row = 0; row < access.row_capacity; ++row) {
        clear_pod_v1(access.behavior_rows + row);
        clear_pod_v1(access.lineage_rows + row);
        access.effective_fitness_by_row[row] = 0.0;
    }
    if (start_id_ceiling < 0 || active_count < 0
        || active_count > access.row_capacity
        || (active_count > 0
            && (ascending_active_agent_ids == nullptr
                || active_agent_slots == nullptr || injury_by_slot == nullptr))) {
        record_failure_v1(access.header, kLedgerInvalidInputV1);
        return false;
    }
    std::int64_t previous_id = -1;
    for (std::int32_t row = 0; row < active_count; ++row) {
        const std::int64_t agent_id = ascending_active_agent_ids[row];
        const std::int32_t slot = active_agent_slots[row];
        if (agent_id < 0 || agent_id >= start_id_ceiling
            || agent_id <= previous_id) {
            record_failure_v1(access.header, kLedgerIdentityOrderV1);
            return false;
        }
        if (slot < 0 || slot >= access.slot_capacity
            || access.slot_to_behavior_row[slot] != -1
            || !finite_float_v1(injury_by_slot[slot])) {
            record_failure_v1(access.header, kLedgerSlotAliasV1);
            return false;
        }
        BehaviorAuthorityRowV1& target = access.behavior_rows[row];
        target.agent_id = agent_id;
        target.injury_start = injury_by_slot[slot];
        target.flags = kBehaviorAliveEndV1;
        access.slot_to_behavior_row[slot] = row;
        previous_id = agent_id;
    }
    access.header->behavior_row_count = static_cast<std::uint16_t>(active_count);
    return true;
}

EIEL_REPORTING_AUTHORITY_DEVICE std::int32_t find_agent_row_v1(
    const GenerationAuthorityAccessV1& access,
    const std::int64_t agent_id
) noexcept {
    const std::int32_t count = access.header->behavior_row_count;
    for (std::int32_t row = 0; row < count; ++row) {
        if (access.behavior_rows[row].agent_id == agent_id) {
            return row;
        }
    }
    return -1;
}

EIEL_REPORTING_AUTHORITY_DEVICE std::int32_t find_live_slot_for_row_v1(
    const GenerationAuthorityAccessV1& access,
    const std::int32_t behavior_row
) noexcept {
    std::int32_t found = -1;
    for (std::int32_t slot = 0; slot < access.slot_capacity; ++slot) {
        if (access.slot_to_behavior_row[slot] == behavior_row) {
            if (found != -1) {
                return -2;
            }
            found = slot;
        }
    }
    return found;
}

EIEL_REPORTING_AUTHORITY_DEVICE bool append_ordinary_birth_v1(
    const GenerationAuthorityAccessV1& access,
    const std::int64_t agent_id,
    const std::int32_t slot,
    const float start_injury
) noexcept {
    if (!require_open_v1(access.header)) {
        return false;
    }
    const std::int32_t count = access.header->behavior_row_count;
    if (find_agent_row_v1(access, agent_id) >= 0) {
        record_failure_v1(access.header, kLedgerDuplicateAgentIdV1);
        return false;
    }
    if (agent_id < access.header->start_id_ceiling || slot < 0
        || slot >= access.slot_capacity || !finite_float_v1(start_injury)) {
        record_failure_v1(access.header, kLedgerInvalidInputV1);
        return false;
    }
    if (count > 0 && agent_id <= access.behavior_rows[count - 1].agent_id) {
        record_failure_v1(access.header, kLedgerIdentityOrderV1);
        return false;
    }
    if (access.slot_to_behavior_row[slot] != -1) {
        record_failure_v1(access.header, kLedgerSlotAliasV1);
        return false;
    }
    if (count >= access.row_capacity) {
        record_failure_v1(access.header, kLedgerOverflowV1);
        return false;
    }
    BehaviorAuthorityRowV1& row = access.behavior_rows[count];
    clear_pod_v1(&row);
    row.agent_id = agent_id;
    row.injury_start = start_injury;
    row.flags = kBehaviorAliveEndV1;
    access.slot_to_behavior_row[slot] = count;
    access.header->behavior_row_count = static_cast<std::uint16_t>(count + 1);
    return true;
}

EIEL_REPORTING_AUTHORITY_DEVICE bool count_action_v1(
    const GenerationAuthorityAccessV1& access,
    const std::int32_t slot,
    const std::int64_t agent_id,
    const std::int32_t action
) noexcept {
    if (!require_open_v1(access.header)) {
        return false;
    }
    if (slot < 0 || slot >= access.slot_capacity || action < 0
        || action >= kActionCountV1) {
        record_failure_v1(access.header, kLedgerInvalidActionV1);
        return false;
    }
    const std::int32_t row_index = access.slot_to_behavior_row[slot];
    if (row_index < 0 || row_index >= access.header->behavior_row_count
        || access.behavior_rows[row_index].agent_id != agent_id) {
        record_failure_v1(access.header, kLedgerSlotAliasV1);
        return false;
    }
    std::uint8_t& count = access.behavior_rows[row_index].action_counts[action];
    if (count == 255U) {
        record_failure_v1(access.header, kLedgerCounterOverflowV1);
        return false;
    }
    ++count;
    return true;
}

EIEL_REPORTING_AUTHORITY_DEVICE bool valid_terminal_values_v1(
    const BehaviorTerminalValuesV1& source
) noexcept {
    return source.lifespan_steps >= 0 && source.offspring >= 0
        && source.attacks_made >= 0 && source.early_survival_steps >= 0
        && source.action_switches >= 0 && source.action_oscillations >= 0
        && source.selfharm_events >= 0 && source.pred_err_count >= 0
        && source.episode0_pred_err_count >= 0
        && source.episode0_selfharm_events >= 0
        && source.late_pred_err_count >= 0
        && finite_double_v1(source.energy_collected_exact)
        && finite_double_v1(source.early_energy_collected_exact)
        && finite_float_v1(source.injury_end)
        && finite_float_v1(source.aggression_bias_f32)
        && finite_float_v1(source.early_injury_gained)
        && finite_float_v1(source.drive_jitter)
        && finite_float_v1(source.pred_err_sum)
        && finite_float_v1(source.birth_reset_shock)
        && finite_float_v1(source.episode0_pred_err_sum)
        && finite_float_v1(source.late_pred_err_sum);
}

EIEL_REPORTING_AUTHORITY_DEVICE double base_fitness_v1(
    const BehaviorTerminalValuesV1& source,
    const bool alive_end
) noexcept {
    const double survival = static_cast<double>(source.lifespan_steps);
    double score = (survival < 120.0 ? survival : 120.0) * 0.05;
    if (source.offspring > 0) {
        score += 180.0;
    }
    if (source.offspring > 1) {
        score += 120.0;
    }
    if (source.offspring > 2) {
        score += 60.0 * static_cast<double>(source.offspring - 2);
    }
    score += (source.energy_collected_exact / (survival > 1.0 ? survival : 1.0))
        * 80.0;
    score -= static_cast<double>(source.attacks_made) * 2.5;
    const double early_survival = static_cast<double>(source.early_survival_steps);
    const double early_weight = finite_float_v1(source.early_life_weight)
        ? static_cast<double>(source.early_life_weight)
        : 1.0;
    double early_component = 0.10 * early_survival;
    early_component += 80.0
        * (source.early_energy_collected_exact
           / (early_survival > 1.0 ? early_survival : 1.0));
    early_component -= 10.0 * static_cast<double>(source.early_injury_gained);
    score += early_weight * early_component;
    if (alive_end) {
        score += 10.0;
    }
    return score;
}

EIEL_REPORTING_AUTHORITY_DEVICE bool valid_fitness_config_v1(
    const ReportingFitnessConfigV1& config
) noexcept {
    const double values[] = {
        config.parent_fraction,
        config.episode0_fitness_weight,
        config.episode0_pred_err_fitness_weight,
        config.episode0_thrashing_fitness_weight,
        config.episode0_drive_jitter_fitness_weight,
        config.episode0_selfharm_fitness_weight,
        config.coherence_thrashing_weight,
        config.coherence_drive_jitter_weight,
        config.coherence_selfharm_weight,
        config.coherence_pred_err_weight,
        config.coherence_reset_shock_weight,
        config.coherence_episode0_pred_err_weight,
        config.late_pred_cost_weight,
    };
    for (const double value : values) {
        if (!finite_double_v1(value)) {
            return false;
        }
    }
    return config.parent_fraction >= 0.0 && config.parent_fraction <= 1.0
        && config.late_pred_cost_enable <= 1
        && config.disable_prediction_error_selection_pressure <= 1
        && config.patch_habitat_lineage_available <= 1;
}

EIEL_REPORTING_AUTHORITY_DEVICE double effective_fitness_v1(
    const double base,
    const BehaviorTerminalValuesV1& source,
    const ReportingFitnessConfigV1& config
) noexcept {
    const double episode0_pred = static_cast<double>(source.episode0_pred_err_sum)
        / (source.episode0_pred_err_count > 1
               ? static_cast<double>(source.episode0_pred_err_count)
               : 1.0);
    const double thrash = 0.5 * static_cast<double>(source.action_switches)
        + static_cast<double>(source.action_oscillations);
    const double drive_jitter = static_cast<double>(source.drive_jitter);
    const double selfharm = static_cast<double>(source.selfharm_events);
    const double episode0_selfharm =
        static_cast<double>(source.episode0_selfharm_events);
    double effective = base;
    const bool split_episode0 = config.episode0_pred_err_fitness_weight != 0.0
        || config.episode0_thrashing_fitness_weight != 0.0
        || config.episode0_drive_jitter_fitness_weight != 0.0
        || config.episode0_selfharm_fitness_weight != 0.0;
    if (split_episode0) {
        const double pred_weight = config.disable_prediction_error_selection_pressure
            ? 0.0
            : config.episode0_pred_err_fitness_weight;
        effective += pred_weight * episode0_pred;
        effective += config.episode0_thrashing_fitness_weight * thrash;
        effective += config.episode0_drive_jitter_fitness_weight * drive_jitter;
        effective += config.episode0_selfharm_fitness_weight * episode0_selfharm;
    } else {
        effective += config.episode0_fitness_weight
            * (episode0_pred + thrash + drive_jitter + episode0_selfharm);
    }
    const bool disable_prediction =
        config.disable_prediction_error_selection_pressure != 0;
    const double pred_weight = disable_prediction ? 0.0 : config.coherence_pred_err_weight;
    const double episode0_pred_weight = disable_prediction
        ? 0.0
        : config.coherence_episode0_pred_err_weight;
    const double pred_mean = static_cast<double>(source.pred_err_sum)
        / (source.pred_err_count > 1
               ? static_cast<double>(source.pred_err_count)
               : 1.0);
    double legacy_penalty = config.coherence_thrashing_weight * thrash;
    legacy_penalty += config.coherence_drive_jitter_weight * drive_jitter;
    legacy_penalty += config.coherence_selfharm_weight * selfharm;
    legacy_penalty += pred_weight * pred_mean;
    legacy_penalty += config.coherence_reset_shock_weight
        * static_cast<double>(source.birth_reset_shock);
    legacy_penalty += episode0_pred_weight * episode0_pred;
    double late_cost = 0.0;
    if (config.late_pred_cost_enable && config.late_pred_cost_weight > 0.0
        && source.late_pred_err_count > 0) {
        const double late_mean = static_cast<double>(source.late_pred_err_sum)
            / static_cast<double>(source.late_pred_err_count);
        if (late_mean > 0.0) {
            late_cost = config.late_pred_cost_weight * late_mean;
        }
    }
    const double result = effective - legacy_penalty - late_cost;
    return result > 0.0 ? result : 0.0;
}

EIEL_REPORTING_AUTHORITY_DEVICE bool apply_terminal_values_v1(
    GenerationAuthorityHeaderV1* const header,
    BehaviorAuthorityRowV1* const row,
    const BehaviorTerminalValuesV1& source,
    const bool alive_end
) noexcept {
    if (!valid_terminal_values_v1(source)) {
        record_failure_v1(header, kLedgerIncompleteTerminalV1);
        return false;
    }
    row->energy_collected_exact = source.energy_collected_exact;
    row->base_fitness = base_fitness_v1(source, alive_end);
    row->injury_end = source.injury_end;
    row->aggression_bias_f32 = source.aggression_bias_f32;
    row->lifespan_steps = source.lifespan_steps;
    row->offspring = source.offspring;
    row->attacks_made = source.attacks_made;
    row->flags = static_cast<std::uint8_t>(
        (row->flags & ~(kBehaviorAliveEndV1 | kBehaviorEligibleForSelectionV1
                        | kBehaviorKeptNextV1))
        | kBehaviorTerminalSealedV1
        | (alive_end ? kBehaviorAliveEndV1 : 0)
    );
    if (!finite_double_v1(row->base_fitness)) {
        record_failure_v1(header, kLedgerIncompleteTerminalV1);
        return false;
    }
    return true;
}

EIEL_REPORTING_AUTHORITY_DEVICE bool seal_death_before_slot_reuse_v1(
    const GenerationAuthorityAccessV1& access,
    const std::int32_t slot,
    const std::int64_t agent_id,
    const BehaviorTerminalValuesV1& terminal
) noexcept {
    if (!require_open_v1(access.header)) {
        return false;
    }
    if (slot < 0 || slot >= access.slot_capacity) {
        record_failure_v1(access.header, kLedgerInvalidInputV1);
        return false;
    }
    const std::int32_t row_index = access.slot_to_behavior_row[slot];
    if (row_index < 0 || row_index >= access.header->behavior_row_count
        || access.behavior_rows[row_index].agent_id != agent_id) {
        record_failure_v1(access.header, kLedgerSlotAliasV1);
        return false;
    }
    if (!apply_terminal_values_v1(
            access.header, access.behavior_rows + row_index, terminal, false
        )) {
        return false;
    }
    access.slot_to_behavior_row[slot] = -1;
    return true;
}

EIEL_REPORTING_AUTHORITY_DEVICE bool resident_sources_valid_v1(
    const ResidentGenerationSourcesV1& sources,
    const std::int32_t expected_slot_capacity
) noexcept {
    return sources.slot_capacity == expected_slot_capacity
        && sources.slot_to_agent_id != nullptr
        && sources.energy_collected_exact != nullptr
        && sources.early_energy_collected_exact != nullptr
        && sources.injury != nullptr && sources.aggression_bias != nullptr
        && sources.early_injury_gained != nullptr
        && sources.early_life_weight != nullptr && sources.drive_jitter != nullptr
        && sources.pred_err_sum != nullptr && sources.birth_reset_shock != nullptr
        && sources.episode0_pred_err_sum != nullptr
        && sources.late_pred_err_sum != nullptr && sources.lifespan_steps != nullptr
        && sources.offspring_count != nullptr && sources.attacks_made != nullptr
        && sources.early_survival_steps != nullptr
        && sources.action_switches != nullptr && sources.action_oscillations != nullptr
        && sources.selfharm_events != nullptr && sources.pred_err_count != nullptr
        && sources.episode0_pred_err_count != nullptr
        && sources.episode0_selfharm_events != nullptr
        && sources.late_pred_err_count != nullptr
        && sources.policy_energy_observation != nullptr && sources.health != nullptr
        && sources.drives != nullptr && sources.exploration_bias != nullptr
        && sources.identity_drift_ema_available <= 1
        && (sources.identity_drift_ema_available == 0
            || sources.identity_drift_ema != nullptr)
        && sources.pos_y != nullptr && sources.pos_x != nullptr
        && sources.lineage_founder_id != nullptr
        && sources.lineage_birth_generation != nullptr
        && sources.lineage_birth_tick != nullptr && sources.lineage_depth != nullptr
        && sources.first_repro_step != nullptr
        && sources.patch_habitat_lineage_id != nullptr
        && sources.genome_hidden_size != nullptr && sources.genome_memory_dim != nullptr
        && sources.policy_use_gru != nullptr;
}

EIEL_REPORTING_AUTHORITY_DEVICE BehaviorTerminalValuesV1 terminal_at_v1(
    const ResidentGenerationSourcesV1& source,
    const std::int32_t slot
) noexcept {
    return {
        source.energy_collected_exact[slot],
        source.early_energy_collected_exact[slot],
        source.injury[slot],
        source.aggression_bias[slot],
        source.early_injury_gained[slot],
        source.early_life_weight[slot],
        source.drive_jitter[slot],
        source.pred_err_sum[slot],
        source.birth_reset_shock[slot],
        source.episode0_pred_err_sum[slot],
        source.late_pred_err_sum[slot],
        source.lifespan_steps[slot],
        source.offspring_count[slot],
        source.attacks_made[slot],
        source.early_survival_steps[slot],
        source.action_switches[slot],
        source.action_oscillations[slot],
        source.selfharm_events[slot],
        source.pred_err_count[slot],
        source.episode0_pred_err_count[slot],
        source.episode0_selfharm_events[slot],
        source.late_pred_err_count[slot],
    };
}

EIEL_REPORTING_AUTHORITY_DEVICE bool write_lineage_v1(
    GenerationAuthorityHeaderV1* const header,
    LineageAuthorityRowV1* const target,
    const ResidentGenerationSourcesV1& source,
    const std::int32_t slot,
    const std::uint8_t patch_habitat_lineage_available
) noexcept {
    const float* const drives = source.drives + static_cast<std::int64_t>(slot) * 4;
    const bool valid = finite_double_v1(source.policy_energy_observation[slot])
        && finite_double_v1(source.energy_collected_exact[slot])
        && finite_float_v1(source.health[slot])
        && finite_float_v1(source.injury[slot])
        && finite_float_v1(drives[0]) && finite_float_v1(drives[1])
        && finite_float_v1(drives[2]) && finite_float_v1(drives[3])
        && finite_float_v1(source.pred_err_sum[slot])
        && finite_float_v1(source.episode0_pred_err_sum[slot])
        && finite_float_v1(source.drive_jitter[slot])
        && finite_float_v1(source.aggression_bias[slot])
        && finite_float_v1(source.exploration_bias[slot])
        && source.lineage_founder_id[slot] >= 0
        && source.lineage_birth_generation[slot] >= 0
        && source.lineage_birth_tick[slot] >= 0
        && source.lineage_depth[slot] >= 0
        && source.offspring_count[slot] >= 0
        && source.lifespan_steps[slot] >= 0
        && source.first_repro_step[slot] >= -1
        && source.pred_err_count[slot] >= 0
        && source.episode0_pred_err_count[slot] >= 0
        && (patch_habitat_lineage_available == 0
            || source.patch_habitat_lineage_id[slot] >= 0)
        && source.genome_hidden_size[slot] >= 0
        && source.genome_memory_dim[slot] >= 0
        && source.policy_use_gru[slot] <= 1
        && (source.identity_drift_ema_available == 0
            || finite_float_v1(source.identity_drift_ema[slot]));
    if (!valid) {
        record_failure_v1(header, kLedgerInvalidLineageSourceV1);
        return false;
    }
    clear_pod_v1(target);
    const std::int32_t episode0_count =
        source.episode0_pred_err_count[slot];
    target->agent_id = source.slot_to_agent_id[slot];
    target->energy = source.policy_energy_observation[slot];
    target->energy_collected_exact = source.energy_collected_exact[slot];
    target->episode0_pred_err_mean = episode0_count > 0
        ? static_cast<double>(source.episode0_pred_err_sum[slot])
            / static_cast<double>(episode0_count)
        : 0.0;
    target->identity_drift_ema = source.identity_drift_ema_available != 0
        ? static_cast<double>(source.identity_drift_ema[slot])
        : 0.0;
    target->health = source.health[slot];
    target->injury = source.injury[slot];
    for (std::int32_t index = 0; index < 4; ++index) {
        target->drives[index] = drives[index];
    }
    target->pred_err_sum = source.pred_err_sum[slot];
    target->drive_jitter = source.drive_jitter[slot];
    target->aggression_bias_f32 = source.aggression_bias[slot];
    target->exploration_bias_f32 = source.exploration_bias[slot];
    target->y = source.pos_y[slot];
    target->x = source.pos_x[slot];
    target->founder_id = source.lineage_founder_id[slot];
    target->birth_generation = source.lineage_birth_generation[slot];
    target->birth_tick = source.lineage_birth_tick[slot];
    target->lineage_depth = source.lineage_depth[slot];
    target->offspring_count = source.offspring_count[slot];
    target->survival_steps = source.lifespan_steps[slot];
    target->first_repro_step = source.first_repro_step[slot];
    target->pred_err_count = source.pred_err_count[slot];
    target->patch_habitat_lineage_id = patch_habitat_lineage_available != 0
        ? source.patch_habitat_lineage_id[slot]
        : 0;
    target->hidden_size = source.genome_hidden_size[slot];
    target->memory_dim = source.genome_memory_dim[slot];
    target->use_gru = source.policy_use_gru[slot];
    return true;
}

EIEL_REPORTING_AUTHORITY_DEVICE bool finalize_pre_turnover_v1(
    const GenerationAuthorityAccessV1& access,
    const ResidentGenerationSourcesV1& sources,
    const ReportingFitnessConfigV1& config
) noexcept {
    if (!require_open_v1(access.header)) {
        return false;
    }
    if (!resident_sources_valid_v1(sources, access.slot_capacity)) {
        record_failure_v1(access.header, kLedgerInvalidStorageV1);
        return false;
    }
    if (!valid_fitness_config_v1(config)) {
        record_failure_v1(access.header, kLedgerInvalidFitnessConfigV1);
        return false;
    }
    const std::int32_t count = access.header->behavior_row_count;
    std::int32_t lineage_count = 0;
    std::int32_t eligible_count = 0;
    for (std::int32_t row_index = 0; row_index < count; ++row_index) {
        BehaviorAuthorityRowV1& row = access.behavior_rows[row_index];
        if ((row.flags & kBehaviorTerminalSealedV1) != 0
            && (row.flags & kBehaviorAliveEndV1) == 0) {
            if (find_live_slot_for_row_v1(access, row_index) != -1) {
                record_failure_v1(access.header, kLedgerSlotAliasV1);
                return false;
            }
            access.effective_fitness_by_row[row_index] = 0.0;
            continue;
        }
        const std::int32_t slot = find_live_slot_for_row_v1(access, row_index);
        if (slot < 0 || sources.slot_to_agent_id[slot] != row.agent_id) {
            record_failure_v1(access.header, kLedgerSlotAliasV1);
            return false;
        }
        const BehaviorTerminalValuesV1 terminal = terminal_at_v1(sources, slot);
        if (!apply_terminal_values_v1(access.header, &row, terminal, true)) {
            return false;
        }
        access.effective_fitness_by_row[row_index] =
            effective_fitness_v1(row.base_fitness, terminal, config);
        if (!finite_double_v1(access.effective_fitness_by_row[row_index])) {
            record_failure_v1(access.header, kLedgerInvalidFitnessConfigV1);
            return false;
        }
        if (row.agent_id < access.header->start_id_ceiling) {
            row.flags |= kBehaviorEligibleForSelectionV1;
            ++eligible_count;
        }
        if (!write_lineage_v1(
                access.header,
                access.lineage_rows + lineage_count,
                sources,
                slot,
                config.patch_habitat_lineage_available
            )) {
            return false;
        }
        ++lineage_count;
    }
    const std::int32_t scaled = static_cast<std::int32_t>(
        static_cast<double>(eligible_count) * config.parent_fraction
    );
    std::int32_t parent_count = scaled > 2 ? scaled : 2;
    if (parent_count > eligible_count) {
        parent_count = eligible_count;
    }
    for (std::int32_t row_index = 0; row_index < count; ++row_index) {
        BehaviorAuthorityRowV1& row = access.behavior_rows[row_index];
        if ((row.flags & kBehaviorEligibleForSelectionV1) == 0) {
            continue;
        }
        std::int32_t rank = 0;
        const double fitness = access.effective_fitness_by_row[row_index];
        for (std::int32_t other = 0; other < count; ++other) {
            const BehaviorAuthorityRowV1& candidate = access.behavior_rows[other];
            if ((candidate.flags & kBehaviorEligibleForSelectionV1) == 0) {
                continue;
            }
            const double other_fitness = access.effective_fitness_by_row[other];
            if (other_fitness > fitness
                || (other_fitness == fitness && other < row_index)) {
                ++rank;
            }
        }
        if (rank < parent_count) {
            row.flags |= kBehaviorKeptNextV1;
        }
    }
    access.header->lineage_record_count =
        static_cast<std::uint16_t>(lineage_count);
    access.header->status = kLedgerSealedV1;
    return true;
}

#undef EIEL_REPORTING_AUTHORITY_DEVICE
#undef EIEL_REPORTING_AUTHORITY_INLINE

static_assert(sizeof(GenerationAuthorityHeaderV1) == 16);
static_assert(sizeof(BehaviorAuthorityRowV1) == 56);
static_assert(sizeof(LineageAuthorityRowV1) == 136);
static_assert(offsetof(BehaviorAuthorityRowV1, agent_id) == 0);
static_assert(offsetof(BehaviorAuthorityRowV1, energy_collected_exact) == 8);
static_assert(offsetof(BehaviorAuthorityRowV1, base_fitness) == 16);
static_assert(offsetof(BehaviorAuthorityRowV1, injury_start) == 24);
static_assert(offsetof(BehaviorAuthorityRowV1, injury_end) == 28);
static_assert(offsetof(BehaviorAuthorityRowV1, aggression_bias_f32) == 32);
static_assert(offsetof(BehaviorAuthorityRowV1, lifespan_steps) == 36);
static_assert(offsetof(BehaviorAuthorityRowV1, offspring) == 40);
static_assert(offsetof(BehaviorAuthorityRowV1, attacks_made) == 44);
static_assert(offsetof(BehaviorAuthorityRowV1, action_counts) == 48);
static_assert(offsetof(BehaviorAuthorityRowV1, flags) == 55);
static_assert(offsetof(LineageAuthorityRowV1, agent_id) == 0);
static_assert(offsetof(LineageAuthorityRowV1, energy) == 8);
static_assert(offsetof(LineageAuthorityRowV1, energy_collected_exact) == 16);
static_assert(offsetof(LineageAuthorityRowV1, episode0_pred_err_mean) == 24);
static_assert(offsetof(LineageAuthorityRowV1, identity_drift_ema) == 32);
static_assert(offsetof(LineageAuthorityRowV1, health) == 40);
static_assert(offsetof(LineageAuthorityRowV1, injury) == 44);
static_assert(offsetof(LineageAuthorityRowV1, drives) == 48);
static_assert(offsetof(LineageAuthorityRowV1, pred_err_sum) == 64);
static_assert(offsetof(LineageAuthorityRowV1, drive_jitter) == 68);
static_assert(offsetof(LineageAuthorityRowV1, aggression_bias_f32) == 72);
static_assert(offsetof(LineageAuthorityRowV1, exploration_bias_f32) == 76);
static_assert(offsetof(LineageAuthorityRowV1, y) == 80);
static_assert(offsetof(LineageAuthorityRowV1, x) == 84);
static_assert(offsetof(LineageAuthorityRowV1, founder_id) == 88);
static_assert(offsetof(LineageAuthorityRowV1, birth_generation) == 92);
static_assert(offsetof(LineageAuthorityRowV1, birth_tick) == 96);
static_assert(offsetof(LineageAuthorityRowV1, lineage_depth) == 100);
static_assert(offsetof(LineageAuthorityRowV1, offspring_count) == 104);
static_assert(offsetof(LineageAuthorityRowV1, survival_steps) == 108);
static_assert(offsetof(LineageAuthorityRowV1, first_repro_step) == 112);
static_assert(offsetof(LineageAuthorityRowV1, pred_err_count) == 116);
static_assert(offsetof(LineageAuthorityRowV1, patch_habitat_lineage_id) == 120);
static_assert(offsetof(LineageAuthorityRowV1, hidden_size) == 124);
static_assert(offsetof(LineageAuthorityRowV1, memory_dim) == 128);
static_assert(offsetof(LineageAuthorityRowV1, use_gru) == 132);
static_assert(offsetof(LineageAuthorityRowV1, reserved) == 133);
static_assert(retained_bytes_per_generation_cell_v1(kStrictRowCapacityV1) == 12304);
static_assert(retained_bytes_per_generation_cell_v1(kDynamicRowCapacityV1) == 43024);
static_assert(
    retained_bytes_per_generation_cell_v1(kMaximumDynamicRowCapacityV1)
        == 5971984
);
static_assert(std::is_standard_layout_v<GenerationAuthorityHeaderV1>);
static_assert(std::is_trivially_copyable_v<GenerationAuthorityHeaderV1>);
static_assert(std::is_standard_layout_v<BehaviorAuthorityRowV1>);
static_assert(std::is_trivially_copyable_v<BehaviorAuthorityRowV1>);
static_assert(std::is_standard_layout_v<LineageAuthorityRowV1>);
static_assert(std::is_trivially_copyable_v<LineageAuthorityRowV1>);

}  // namespace eiel::cuda_architecture_b::compact_summary_reporting_authority_v1
