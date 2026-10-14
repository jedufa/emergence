#pragma once

#include "../cuda_architecture_b_compact_summary_host_canonical_v1.hpp"
#include "../cuda_architecture_b_compact_summary_reporting_authority_v1.cuh"

#include <array>
#include <cstddef>
#include <cstdint>

namespace eiel::native::cpu_compact_summary_reporting_ledger_v1 {

namespace authority =
    eiel::cuda_architecture_b::compact_summary_reporting_authority_v1;
namespace canonical =
    eiel::cuda_architecture_b::compact_summary_host_canonical_v1;

inline constexpr std::int32_t kMaximumRowCapacityV1 =
    authority::kMaximumDynamicRowCapacityV1;
inline constexpr std::int32_t kMaximumSlotCapacityV1 = 1024;

enum class LedgerWrapperStatusV1 : std::uint32_t {
    kSuccess = 0,
    kInvalidCapacity = 1,
    kInvalidInitialLineageState = 2,
    kNotInitialized = 3,
    kAuthorityFailure = 4,
    kFinalizeRequired = 5,
    kCanonicalFailure = 6,
};

class CpuCompactSummaryReportingLedgerV1 {
  public:
    explicit CpuCompactSummaryReportingLedgerV1(
        std::int32_t row_capacity,
        std::int32_t slot_capacity,
        canonical::LineageMetricsStateV1 initial_lineage_metrics_state = {}
    ) noexcept;

    [[nodiscard]] bool initialize_generation_from_slot_state_v1(
        std::int64_t start_id_ceiling,
        const std::int64_t* ascending_active_agent_ids,
        const std::int32_t* agent_id_to_slot,
        const float* injury_by_slot,
        std::int32_t active_count,
        std::int32_t agent_id_capacity
    ) noexcept;

    [[nodiscard]] bool initialize_generation_from_active_slot_state_v1(
        std::int64_t start_id_ceiling,
        const std::int64_t* ascending_active_agent_ids,
        const std::int32_t* active_agent_slots,
        const float* injury_by_slot,
        std::int32_t active_count
    ) noexcept;

    [[nodiscard]] bool count_action_v1(
        std::int32_t slot,
        std::int64_t agent_id,
        std::int32_t action
    ) noexcept;

    [[nodiscard]] bool append_ordinary_birth_v1(
        std::int64_t agent_id,
        std::int32_t slot,
        float start_injury
    ) noexcept;

    [[nodiscard]] bool seal_death_before_slot_reuse_v1(
        std::int32_t slot,
        std::int64_t agent_id,
        const authority::BehaviorTerminalValuesV1& terminal
    ) noexcept;

    [[nodiscard]] bool finalize_pre_turnover_v1(
        const authority::ResidentGenerationSourcesV1& sources,
        const authority::ReportingFitnessConfigV1& fitness_config
    ) noexcept;

    [[nodiscard]] bool canonicalize_retained_summary_v1(
        std::int64_t generation,
        const canonical::LineageMetricsConfigV1& lineage_metrics_config
    ) noexcept;

    [[nodiscard]] bool capacity_valid() const noexcept {
        return capacity_valid_;
    }

    [[nodiscard]] bool finalized() const noexcept {
        return finalized_;
    }

    [[nodiscard]] bool committed() const noexcept {
        return committed_;
    }

    [[nodiscard]] std::int32_t row_capacity() const noexcept {
        return row_capacity_;
    }

    [[nodiscard]] std::int32_t slot_capacity() const noexcept {
        return slot_capacity_;
    }

    [[nodiscard]] LedgerWrapperStatusV1 wrapper_status() const noexcept {
        return wrapper_status_;
    }

    [[nodiscard]] canonical::HostCanonicalStatusV1
    last_canonical_status() const noexcept {
        return last_canonical_status_;
    }

    [[nodiscard]] const authority::GenerationAuthorityHeaderV1& header()
        const noexcept {
        return header_;
    }

    [[nodiscard]] const authority::BehaviorAuthorityRowV1* behavior_rows()
        const noexcept {
        return behavior_rows_.data();
    }

    [[nodiscard]] const authority::LineageAuthorityRowV1* lineage_rows()
        const noexcept {
        return lineage_rows_.data();
    }

    [[nodiscard]] const canonical::HostCanonicalResultV1& canonical_result()
        const noexcept {
        return canonical_result_;
    }

    [[nodiscard]] const canonical::LineageMetricsStateV1&
    lineage_metrics_state() const noexcept {
        return lineage_metrics_state_;
    }

    [[nodiscard]] std::size_t retained_row_bytes_v1() const noexcept;

    [[nodiscard]] bool copy_retained_row_v1(
        std::uint8_t* output,
        std::size_t output_bytes
    ) const noexcept;

  private:
    [[nodiscard]] authority::GenerationAuthorityAccessV1 access() noexcept;
    [[nodiscard]] bool authority_call_permitted() noexcept;
    [[nodiscard]] bool record_authority_result(bool success) noexcept;
    void clear_canonical_result() noexcept;

    authority::GenerationAuthorityHeaderV1 header_{};
    std::array<
        authority::BehaviorAuthorityRowV1,
        static_cast<std::size_t>(kMaximumRowCapacityV1)
    > behavior_rows_{};
    std::array<
        authority::LineageAuthorityRowV1,
        static_cast<std::size_t>(kMaximumRowCapacityV1)
    > lineage_rows_{};
    std::array<
        std::int32_t,
        static_cast<std::size_t>(kMaximumSlotCapacityV1)
    > slot_to_behavior_row_{};
    std::array<
        double,
        static_cast<std::size_t>(kMaximumRowCapacityV1)
    > effective_fitness_by_row_{};
    canonical::HostCanonicalResultV1 canonical_result_{};
    canonical::LineageMetricsStateV1 lineage_metrics_state_{};
    std::int32_t row_capacity_ = 0;
    std::int32_t slot_capacity_ = 0;
    LedgerWrapperStatusV1 wrapper_status_ =
        LedgerWrapperStatusV1::kInvalidCapacity;
    canonical::HostCanonicalStatusV1 last_canonical_status_ =
        canonical::HostCanonicalStatusV1::kSuccess;
    bool capacity_valid_ = false;
    bool initialized_ = false;
    bool finalized_ = false;
    bool committed_ = false;
};

[[nodiscard]] const char* ledger_wrapper_status_name(
    LedgerWrapperStatusV1 status
) noexcept;

}  // namespace eiel::native::cpu_compact_summary_reporting_ledger_v1
