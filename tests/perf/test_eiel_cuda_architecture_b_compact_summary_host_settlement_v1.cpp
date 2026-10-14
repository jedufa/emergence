#include "cuda_architecture_b_compact_summary_host_settlement_v1.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace settlement =
    eiel::cuda_architecture_b::compact_summary_host_settlement_v1;
namespace canonical =
    eiel::cuda_architecture_b::compact_summary_host_canonical_v1;
namespace metadata =
    eiel::cuda_architecture_b::compact_summary_metadata_retention_v1;
namespace projection =
    eiel::cuda_architecture_b::compact_summary_source_projection_v1;
namespace reductions =
    eiel::cuda_architecture_b::compact_summary_reductions_v1;
namespace reporting =
    eiel::cuda_architecture_b::compact_summary_reporting_authority_v1;
namespace renderer =
    eiel::cuda_architecture_b::compact_summary_host_template_renderer_v1;
namespace runtime =
    eiel::cuda_architecture_b::compact_summary_host_template_runtime_v1;
namespace resolver =
    eiel::cuda_architecture_b::compact_summary_host_token_resolver_v1;
namespace abi = eiel::cuda_architecture_b::compact_summary_v2;

namespace eiel::cuda_architecture_b::compact_summary_host_canonical_v1 {

HostCanonicalStatusV1 canonicalize_retained_summary_v1(
    const std::int64_t,
    const GenerationAuthorityHeaderV1*,
    const BehaviorAuthorityRowV1*,
    const LineageAuthorityRowV1*,
    const std::int32_t,
    const LineageMetricsInputV1* const input,
    HostCanonicalResultV1* const result
) noexcept {
    if (input == nullptr || result == nullptr) {
        return HostCanonicalStatusV1::kNullArgument;
    }
    result->status = static_cast<std::uint32_t>(HostCanonicalStatusV1::kSuccess);
    result->lineage_state_after = input->initial_state;
    ++result->lineage_state_after.collapse_streak;
    ++result->lineage_state_after.population_dominance_streak;
    ++result->lineage_state_after.reproduction_dominance_streak;
    result->lineage.compact_abi_v2_defined_mask =
        kCompactAbiV2DefinedLineageCollapseStreakV1;
    result->lineage.collapse_streak =
        result->lineage_state_after.collapse_streak;
    return HostCanonicalStatusV1::kSuccess;
}

}  // namespace eiel::cuda_architecture_b::compact_summary_host_canonical_v1

namespace eiel::cuda_architecture_b::compact_summary_host_token_resolver_v1 {

HostTokenResolverStatusV1 resolve_compact_summary_host_tokens_v1(
    const HostTokenResolverRequestV1*
) noexcept {
    return HostTokenResolverStatusV1::kSuccess;
}

}  // namespace eiel::cuda_architecture_b::compact_summary_host_token_resolver_v1

namespace eiel::cuda_architecture_b::compact_summary_host_template_runtime_v1 {

HostTemplateProgramBindingV1 lookup_host_template_program_v1(
    HostTemplateRuntimeV1*,
    const std::uint32_t,
    const std::uint32_t,
    const std::uint32_t
) noexcept {
    static renderer::HostTemplateProgramV1 program{};
    static renderer::HostTemplateTokenValueV1 token{};
    return {&program, &token, 1U, 0U};
}

}  // namespace eiel::cuda_architecture_b::compact_summary_host_template_runtime_v1

namespace eiel::cuda_architecture_b::compact_summary_host_template_renderer_v1 {

HostTemplateStatusV1 render_and_hash_host_template_v1(
    const HostTemplateRequestV1* const request
) noexcept {
    if (request == nullptr || request->result == nullptr) {
        return HostTemplateStatusV1::kNullArgument;
    }
    request->result->status =
        static_cast<std::uint32_t>(HostTemplateStatusV1::kSuccess);
    std::memset(request->result->digest_sha256, 0x5a, kDigestBytesV1);
    return HostTemplateStatusV1::kSuccess;
}

}  // namespace eiel::cuda_architecture_b::compact_summary_host_template_renderer_v1

namespace {

constexpr std::int64_t kGenerations = 2;
constexpr std::int64_t kCells = 2;
constexpr std::int64_t kRows = kGenerations * kCells;
constexpr std::int64_t kSummaryWidth = 28;

struct FixtureV1 {
    std::array<projection::CompactSummaryGenerationCellAuthorityV1, kRows>
        authority{};
    std::vector<std::uint8_t> reporting_bytes;
    std::array<double, kRows * kSummaryWidth> summaries{};
    std::array<std::uint8_t, kRows * 32> lifecycle_digests{};
    std::array<std::int32_t, kRows> lifecycle_status{};
    std::array<std::uint8_t, kRows> lifecycle_validity{};
    std::array<std::uint8_t, kRows * 32> checkpoint_raw{};
    std::array<std::uint8_t, kRows * 32> checkpoint_canonical{};
    std::array<std::int32_t, kRows> checkpoint_status{};
    std::array<std::uint8_t, kRows> checkpoint_validity{};
    std::array<reductions::ReductionResultV1, kRows> reductions_rows{};
    std::array<metadata::CompactSummaryMetadataRetentionV1, kRows>
        metadata_rows{};
    std::array<bool, kRows> checkpoint_due{};
    std::array<std::uint8_t, kCells * 32> import_digests{};
    runtime::HostTemplateRuntimeV1 template_runtime{};
    std::array<std::int64_t, kCells> completed_generations{
        kGenerations,
        kGenerations,
    };
    std::array<std::uint64_t, 1> resolver_rows{};
    std::array<std::int64_t, 1> descriptor_specs{};
    std::array<canonical::LineageMetricsStateV1, kCells> lineage_states{
        canonical::LineageMetricsStateV1{3, 4, 5},
        canonical::LineageMetricsStateV1{30, 40, 50},
    };
    settlement::HostSettlementContextV1 context{};

    FixtureV1()
        : reporting_bytes(
              static_cast<std::size_t>(kRows)
              * reporting::retained_bytes_per_generation_cell_v1(
                  reporting::kDynamicRowCapacityV1
              ),
              0U
          ) {
        for (std::int64_t row = 0; row < kRows; ++row) {
            summaries[static_cast<std::size_t>(row) * kSummaryWidth] =
                static_cast<double>(row / kCells);
            lifecycle_validity[row] = 1U;
            checkpoint_validity[row] = 1U;
            reductions_rows[row].status = reductions::kReductionSuccessV1;
            reductions_rows[row].valid = 1U;
            metadata_rows[row].status = metadata::kRetentionSuccessV1;
            metadata_rows[row].validity = 1U;
            metadata_rows[row].flags = metadata::kRequiredRetentionFlagsV1;
            metadata_rows[row].configured_steps_per_generation = 1;
            metadata_rows[row].capacity = 1;
            metadata_rows[row].generation = row / kCells + 1;
        }
        context.authority_rows = authority.data();
        context.reporting_retained_bytes = reporting_bytes.data();
        context.generation_summary_values = summaries.data();
        context.patch_metrics_values = nullptr;
        context.lifecycle_digest_values = lifecycle_digests.data();
        context.lifecycle_status_values = lifecycle_status.data();
        context.lifecycle_validity_values = lifecycle_validity.data();
        context.checkpoint_raw_digests = checkpoint_raw.data();
        context.checkpoint_canonical_digests = checkpoint_canonical.data();
        context.checkpoint_status_values = checkpoint_status.data();
        context.checkpoint_validity_values = checkpoint_validity.data();
        context.reduction_rows = reductions_rows.data();
        context.metadata_rows = metadata_rows.data();
        context.checkpoint_due_values = checkpoint_due.data();
        context.import_digest_values = import_digests.data();
        context.host_template_runtime = &template_runtime;
        context.lineage_state_values = lineage_states.data();
        context.completed_generation_values = completed_generations.data();
        context.host_token_resolver_rows = resolver_rows.data();
        context.checkpoint_descriptor_specs = descriptor_specs.data();
        context.host_token_resolver_row_count = 1;
        context.initial_generation = 0;
        context.generation_count = kGenerations;
        context.cell_count = kCells;
        context.checkpoint_field_count = 1;
        context.checkpoint_descriptor_count = 1;
        context.reporting_retained_row_bytes = static_cast<std::int64_t>(
            reporting::retained_bytes_per_generation_cell_v1(
                reporting::kDynamicRowCapacityV1
            )
        );
        context.generation_summary_width = kSummaryWidth;
        context.patch_metrics_width = 0;
        context.lineage_state_count = kCells;
        context.world_code = abi::kWorldFlat;
        context.reporting_row_capacity = reporting::kDynamicRowCapacityV1;
    }
};

bool all_authority_zero(const FixtureV1& fixture) {
    const std::array<projection::CompactSummaryGenerationCellAuthorityV1, kRows>
        zero{};
    return std::memcmp(
               fixture.authority.data(),
               zero.data(),
               sizeof(zero)
           ) == 0;
}

bool check_b2_isolation_and_order() {
    FixtureV1 fixture;
    settlement::settle_compact_summary_rows_callback_v1(&fixture.context);
    const std::array<std::int64_t, kRows> expected_streak{4, 31, 5, 32};
    for (std::int64_t row = 0; row < kRows; ++row) {
        const auto& authority = fixture.authority[row];
        if (authority.generation_index != row / kCells
            || authority.cell_index != row % kCells
            || authority.integer_values[
                   abi::kIntegerLineageMetricsLineageCollapseStreak
               ] != expected_streak[row]
            || authority.validity != 1U) {
            return false;
        }
    }
    return fixture.context.status == settlement::kHostSettlementSuccessV1
        && fixture.context.completed_rows == kRows
        && fixture.lineage_states[0].collapse_streak == 5
        && fixture.lineage_states[1].collapse_streak == 32;
}

bool check_invalid_state_fails_closed() {
    FixtureV1 wrong_dimension;
    std::memset(
        wrong_dimension.authority.data(),
        0xa5,
        sizeof(wrong_dimension.authority)
    );
    wrong_dimension.context.lineage_state_count = 1;
    settlement::settle_compact_summary_rows_callback_v1(
        &wrong_dimension.context
    );
    if (wrong_dimension.context.status
            != settlement::kHostSettlementInvalidArgumentsV1
        || !all_authority_zero(wrong_dimension)) {
        return false;
    }

    FixtureV1 malformed;
    std::memset(
        malformed.authority.data(),
        0xa5,
        sizeof(malformed.authority)
    );
    malformed.lineage_states[1].collapse_streak = -1;
    settlement::settle_compact_summary_rows_callback_v1(&malformed.context);
    return malformed.context.status
            == settlement::kHostSettlementInvalidArgumentsV1
        && all_authority_zero(malformed);
}

bool check_b1_legacy_equivalence() {
    FixtureV1 explicit_state;
    explicit_state.context.cell_count = 1;
    explicit_state.context.lineage_state_count = 1;
    explicit_state.lineage_states[0] = canonical::kDefaultLineageMetricsInputV1
        .initial_state;
    explicit_state.completed_generations[0] = kGenerations;
    settlement::settle_compact_summary_rows_callback_v1(
        &explicit_state.context
    );

    FixtureV1 legacy;
    legacy.context.cell_count = 1;
    legacy.context.lineage_state_values = nullptr;
    legacy.context.lineage_state_count = 0;
    legacy.completed_generations[0] = kGenerations;
    settlement::settle_compact_summary_rows_callback_v1(&legacy.context);

    return explicit_state.context.status == settlement::kHostSettlementSuccessV1
        && legacy.context.status == settlement::kHostSettlementSuccessV1
        && std::memcmp(
               explicit_state.authority.data(),
               legacy.authority.data(),
               static_cast<std::size_t>(kGenerations)
                   * sizeof(projection::CompactSummaryGenerationCellAuthorityV1)
           ) == 0;
}

}  // namespace

int main() {
    if (!check_b2_isolation_and_order()) {
        std::cerr << "B2 lineage state isolation or row order failed\n";
        return 1;
    }
    if (!check_invalid_state_fails_closed()) {
        std::cerr << "invalid lineage state did not fail closed\n";
        return 2;
    }
    if (!check_b1_legacy_equivalence()) {
        std::cerr << "B1 compatibility changed\n";
        return 3;
    }
    std::cout
        << "b2=isolated row_order=generation_major "
        << "invalid_state=closed b1=unchanged\n";
    return 0;
}
