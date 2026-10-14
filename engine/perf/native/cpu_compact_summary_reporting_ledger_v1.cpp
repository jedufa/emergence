#include "cpu_compact_summary_reporting_ledger_v1.hpp"

#include <cstring>

namespace eiel::native::cpu_compact_summary_reporting_ledger_v1 {
namespace {

bool valid_lineage_state(
    const canonical::LineageMetricsStateV1& state
) noexcept {
    return state.collapse_streak >= 0
        && state.population_dominance_streak >= 0
        && state.reproduction_dominance_streak >= 0;
}

}  // namespace

CpuCompactSummaryReportingLedgerV1::CpuCompactSummaryReportingLedgerV1(
    const std::int32_t row_capacity,
    const std::int32_t slot_capacity,
    const canonical::LineageMetricsStateV1 initial_lineage_metrics_state
) noexcept
    : lineage_metrics_state_(initial_lineage_metrics_state),
      row_capacity_(row_capacity),
      slot_capacity_(slot_capacity) {
    clear_canonical_result();
    capacity_valid_ = row_capacity > 0
        && row_capacity <= kMaximumRowCapacityV1
        && slot_capacity > 0
        && slot_capacity <= kMaximumSlotCapacityV1;
    if (!capacity_valid_) {
        header_.status = authority::kLedgerInvalidStorageV1;
        wrapper_status_ = LedgerWrapperStatusV1::kInvalidCapacity;
        return;
    }
    if (!valid_lineage_state(lineage_metrics_state_)) {
        header_.status = authority::kLedgerInvalidStorageV1;
        wrapper_status_ = LedgerWrapperStatusV1::kInvalidInitialLineageState;
        return;
    }
    wrapper_status_ = LedgerWrapperStatusV1::kSuccess;
}

authority::GenerationAuthorityAccessV1
CpuCompactSummaryReportingLedgerV1::access() noexcept {
    return {
        &header_,
        behavior_rows_.data(),
        lineage_rows_.data(),
        slot_to_behavior_row_.data(),
        effective_fitness_by_row_.data(),
        row_capacity_,
        slot_capacity_,
    };
}

bool CpuCompactSummaryReportingLedgerV1::authority_call_permitted() noexcept {
    if (!capacity_valid_) {
        wrapper_status_ = LedgerWrapperStatusV1::kInvalidCapacity;
        clear_canonical_result();
        committed_ = false;
        return false;
    }
    if (!valid_lineage_state(lineage_metrics_state_)) {
        wrapper_status_ = LedgerWrapperStatusV1::kInvalidInitialLineageState;
        clear_canonical_result();
        committed_ = false;
        return false;
    }
    if (!initialized_) {
        wrapper_status_ = LedgerWrapperStatusV1::kNotInitialized;
        clear_canonical_result();
        committed_ = false;
        return false;
    }
    return true;
}

bool CpuCompactSummaryReportingLedgerV1::record_authority_result(
    const bool success
) noexcept {
    if (success) {
        wrapper_status_ = LedgerWrapperStatusV1::kSuccess;
        return true;
    }
    wrapper_status_ = LedgerWrapperStatusV1::kAuthorityFailure;
    finalized_ = false;
    committed_ = false;
    clear_canonical_result();
    return false;
}

void CpuCompactSummaryReportingLedgerV1::clear_canonical_result() noexcept {
    std::memset(&canonical_result_, 0, sizeof(canonical_result_));
    canonical_result_.failed_row = canonical::kNoFailedRowV1;
    last_canonical_status_ = canonical::HostCanonicalStatusV1::kSuccess;
}

bool CpuCompactSummaryReportingLedgerV1::initialize_generation_from_slot_state_v1(
    const std::int64_t start_id_ceiling,
    const std::int64_t* const ascending_active_agent_ids,
    const std::int32_t* const agent_id_to_slot,
    const float* const injury_by_slot,
    const std::int32_t active_count,
    const std::int32_t agent_id_capacity
) noexcept {
    initialized_ = false;
    finalized_ = false;
    committed_ = false;
    clear_canonical_result();
    if (!capacity_valid_) {
        wrapper_status_ = LedgerWrapperStatusV1::kInvalidCapacity;
        return false;
    }
    if (!valid_lineage_state(lineage_metrics_state_)) {
        wrapper_status_ = LedgerWrapperStatusV1::kInvalidInitialLineageState;
        return false;
    }
    const bool success = authority::initialize_generation_from_slot_state_v1(
        access(), start_id_ceiling, ascending_active_agent_ids,
        agent_id_to_slot, injury_by_slot, active_count, agent_id_capacity
    );
    if (!success) {
        wrapper_status_ = LedgerWrapperStatusV1::kAuthorityFailure;
        return false;
    }
    initialized_ = true;
    wrapper_status_ = LedgerWrapperStatusV1::kSuccess;
    return true;
}

bool CpuCompactSummaryReportingLedgerV1::
initialize_generation_from_active_slot_state_v1(
    const std::int64_t start_id_ceiling,
    const std::int64_t* const ascending_active_agent_ids,
    const std::int32_t* const active_agent_slots,
    const float* const injury_by_slot,
    const std::int32_t active_count
) noexcept {
    initialized_ = false;
    finalized_ = false;
    committed_ = false;
    clear_canonical_result();
    if (!capacity_valid_) {
        wrapper_status_ = LedgerWrapperStatusV1::kInvalidCapacity;
        return false;
    }
    if (!valid_lineage_state(lineage_metrics_state_)) {
        wrapper_status_ = LedgerWrapperStatusV1::kInvalidInitialLineageState;
        return false;
    }
    const bool success =
        authority::initialize_generation_from_active_slot_state_v1(
            access(), start_id_ceiling, ascending_active_agent_ids,
            active_agent_slots, injury_by_slot, active_count
        );
    if (!success) {
        wrapper_status_ = LedgerWrapperStatusV1::kAuthorityFailure;
        return false;
    }
    initialized_ = true;
    wrapper_status_ = LedgerWrapperStatusV1::kSuccess;
    return true;
}

bool CpuCompactSummaryReportingLedgerV1::count_action_v1(
    const std::int32_t slot,
    const std::int64_t agent_id,
    const std::int32_t action
) noexcept {
    if (!authority_call_permitted()) {
        return false;
    }
    return record_authority_result(
        authority::count_action_v1(access(), slot, agent_id, action)
    );
}

bool CpuCompactSummaryReportingLedgerV1::append_ordinary_birth_v1(
    const std::int64_t agent_id,
    const std::int32_t slot,
    const float start_injury
) noexcept {
    if (!authority_call_permitted()) {
        return false;
    }
    return record_authority_result(
        authority::append_ordinary_birth_v1(
            access(), agent_id, slot, start_injury
        )
    );
}

bool CpuCompactSummaryReportingLedgerV1::seal_death_before_slot_reuse_v1(
    const std::int32_t slot,
    const std::int64_t agent_id,
    const authority::BehaviorTerminalValuesV1& terminal
) noexcept {
    if (!authority_call_permitted()) {
        return false;
    }
    return record_authority_result(
        authority::seal_death_before_slot_reuse_v1(
            access(), slot, agent_id, terminal
        )
    );
}

bool CpuCompactSummaryReportingLedgerV1::finalize_pre_turnover_v1(
    const authority::ResidentGenerationSourcesV1& sources,
    const authority::ReportingFitnessConfigV1& fitness_config
) noexcept {
    if (!authority_call_permitted()) {
        return false;
    }
    const bool success = authority::finalize_pre_turnover_v1(
        access(), sources, fitness_config
    );
    if (!record_authority_result(success)) {
        return false;
    }
    finalized_ = true;
    committed_ = false;
    return true;
}

bool CpuCompactSummaryReportingLedgerV1::canonicalize_retained_summary_v1(
    const std::int64_t generation,
    const canonical::LineageMetricsConfigV1& lineage_metrics_config
) noexcept {
    committed_ = false;
    clear_canonical_result();
    if (!capacity_valid_) {
        wrapper_status_ = LedgerWrapperStatusV1::kInvalidCapacity;
        return false;
    }
    if (!finalized_ || header_.status != authority::kLedgerSealedV1) {
        wrapper_status_ = LedgerWrapperStatusV1::kFinalizeRequired;
        return false;
    }

    const canonical::LineageMetricsInputV1 input{
        lineage_metrics_state_,
        lineage_metrics_config,
    };
    canonical::HostCanonicalResultV1 candidate{};
    const canonical::HostCanonicalStatusV1 status =
        canonical::canonicalize_retained_summary_v1(
            generation,
            &header_,
            behavior_rows_.data(),
            lineage_rows_.data(),
            row_capacity_,
            &input,
            &candidate
        );
    last_canonical_status_ = status;
    if (status != canonical::HostCanonicalStatusV1::kSuccess) {
        canonical_result_.status = candidate.status;
        canonical_result_.failed_field = candidate.failed_field;
        canonical_result_.failed_row = candidate.failed_row;
        wrapper_status_ = LedgerWrapperStatusV1::kCanonicalFailure;
        return false;
    }

    canonical_result_ = candidate;
    lineage_metrics_state_ = candidate.lineage_state_after;
    wrapper_status_ = LedgerWrapperStatusV1::kSuccess;
    committed_ = true;
    return true;
}

const char* ledger_wrapper_status_name(
    const LedgerWrapperStatusV1 status
) noexcept {
    switch (status) {
        case LedgerWrapperStatusV1::kSuccess:
            return "success";
        case LedgerWrapperStatusV1::kInvalidCapacity:
            return "invalid_capacity";
        case LedgerWrapperStatusV1::kInvalidInitialLineageState:
            return "invalid_initial_lineage_state";
        case LedgerWrapperStatusV1::kNotInitialized:
            return "not_initialized";
        case LedgerWrapperStatusV1::kAuthorityFailure:
            return "authority_failure";
        case LedgerWrapperStatusV1::kFinalizeRequired:
            return "finalize_required";
        case LedgerWrapperStatusV1::kCanonicalFailure:
            return "canonical_failure";
    }
    return "unknown";
}

std::size_t CpuCompactSummaryReportingLedgerV1::retained_row_bytes_v1()
    const noexcept {
    if (!capacity_valid_) {
        return 0U;
    }
    return sizeof(header_)
        + static_cast<std::size_t>(row_capacity_)
            * (sizeof(authority::BehaviorAuthorityRowV1)
               + sizeof(authority::LineageAuthorityRowV1));
}

bool CpuCompactSummaryReportingLedgerV1::copy_retained_row_v1(
    std::uint8_t* const output,
    const std::size_t output_bytes
) const noexcept {
    const std::size_t required_bytes = retained_row_bytes_v1();
    if (
        !committed_ || output == nullptr || required_bytes == 0U
        || output_bytes != required_bytes
    ) {
        return false;
    }
    std::size_t offset = 0U;
    std::memcpy(output + offset, &header_, sizeof(header_));
    offset += sizeof(header_);
    const std::size_t behavior_bytes = static_cast<std::size_t>(row_capacity_)
        * sizeof(authority::BehaviorAuthorityRowV1);
    std::memcpy(output + offset, behavior_rows_.data(), behavior_bytes);
    offset += behavior_bytes;
    const std::size_t lineage_bytes = static_cast<std::size_t>(row_capacity_)
        * sizeof(authority::LineageAuthorityRowV1);
    std::memcpy(output + offset, lineage_rows_.data(), lineage_bytes);
    return true;
}

}  // namespace eiel::native::cpu_compact_summary_reporting_ledger_v1
