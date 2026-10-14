#include "../../engine/perf/cuda_architecture_b_compact_summary_v2.cuh"

#include <cstdint>
#include <type_traits>

namespace abi = eiel::cuda_architecture_b::compact_summary_v2;

using BytePlane = abi::StridedPlaneViewV2<std::uint8_t>;
using MaskPlane = abi::StridedPlaneViewV2<abi::DefinedMaskV1>;
using SummaryView = abi::CompactSummaryViewV2;

static_assert(abi::kNativeAbiVersionV2 == 2);
static_assert(abi::kWireSchemaVersionV1 == 1);
static_assert(abi::kCodebookVersionV1 == 1);
static_assert(abi::kDefinedMaskVersionV1 == 1);
static_assert(abi::kPlaneLayoutVersionV2 == 2);
static_assert(abi::kProjectionContractVersionV2 == 2);
static_assert(abi::kInvalidRowContractVersionV2 == 2);

static_assert(abi::kPlaneValidity == 0);
static_assert(abi::kPlaneTextValues == 6);
static_assert(abi::kPlaneCountV2 == 7);
static_assert(abi::kStatusSummaryStatus == 0);
static_assert(abi::kStatusTerminalStickyFailure == 7);
static_assert(abi::kStatusLineageMetricsLineageCollapseWarning == 9);
static_assert(abi::kStatusTerminalRecoverableErrorCode == 10);
static_assert(abi::kStatusTerminalFailureStage == 11);
static_assert(abi::kStatusFieldCountV2 == 12);
static_assert(abi::kIntegerGenerationStartGeneration == 0);
static_assert(abi::kIntegerTerminalFailureTick == 35);
static_assert(abi::kIntegerPatchMetricsEnvRngPaddingDraws == 75);
static_assert(abi::kIntegerFieldCountV2 == 76);
static_assert(abi::kFloatGenerationMetricsPeakInjury == 0);
static_assert(abi::kFloatPatchMetricsPatchDensityHotspotFrac == 20);
static_assert(abi::kFloatFieldCountV2 == 21);
static_assert(abi::kDigestEventsActivePrefixDigestSha256 == 0);
static_assert(abi::kDigestLineageMetricsRecordsDigestSha256 == 9);
static_assert(abi::kDigestFieldCountV2 == 10);
static_assert(abi::kTextLineageMetricsDominantLineageId == 0);
static_assert(abi::kTextFieldCountV2 == 1);
static_assert(abi::kDefinedIdentityActiveAgentIdMin == 0);
static_assert(abi::kDefinedPatchMetrics == 17);
static_assert(abi::kDefinedPatchMetricsPatchDensityHotspotFrac == 31);
static_assert(abi::kDefinedBitCountV1 == 32);
static_assert(
    abi::maximum_known_status_code(abi::kStatusTerminalFailureStage) == 16
);
static_assert(
    abi::is_known_status_code(abi::kStatusTerminalRecoverableErrorCode, 3)
);
static_assert(
    !abi::is_known_status_code(abi::kStatusTerminalRecoverableErrorCode, 4)
);
static_assert(
    abi::is_reserved_or_known_status_code(abi::kStatusWorld, 0)
);

static_assert(abi::kValidityLogicalBytesPerRowV2 == 1);
static_assert(abi::kDefinedMaskLogicalBytesPerRowV2 == 8);
static_assert(abi::kStatusLogicalBytesPerRowV2 == 48);
static_assert(abi::kIntegerLogicalBytesPerRowV2 == 608);
static_assert(abi::kFloatLogicalBytesPerRowV2 == 168);
static_assert(abi::kDigestLogicalBytesPerRowV2 == 320);
static_assert(abi::kTextLogicalBytesPerRowV2 == 65);
static_assert(abi::kLogicalBytesPerGenerationCellV2 == 1218);

static_assert(abi::contiguous_row_index(3, 2, 5) == 17);
static_assert(abi::validity_row_offset(3, 2, 5) == 17);
static_assert(abi::defined_mask_row_offset(3, 2, 5) == 17);
static_assert(abi::status_row_offset(3, 2, 5) == 17 * 12);
static_assert(abi::integer_row_offset(3, 2, 5) == 17 * 76);
static_assert(abi::float_row_offset(3, 2, 5) == 17 * 21);
static_assert(abi::digest_row_offset(3, 2, 5) == 17 * 10 * 32);
static_assert(abi::text_row_offset(3, 2, 5) == 17 * 65);
static_assert(
    abi::digest_field_offset(abi::kDigestLineageMetricsRecordsDigestSha256)
    == 9 * 32
);
static_assert(
    abi::text_field_offset(abi::kTextLineageMetricsDominantLineageId) == 0
);

constexpr BytePlane kTestPlane = {nullptr, 130, 65};
static_assert(abi::row_offset(kTestPlane, 2, 1) == 325);
static_assert(abi::is_defined_bit(0));
static_assert(abi::is_defined_bit(31));
static_assert(!abi::is_defined_bit(32));
static_assert(
    abi::set_defined_bit(0, abi::kDefinedTerminalFailureStage) == (1ULL << 6)
);
static_assert(
    abi::test_defined_bit(
        abi::set_defined_bit(0, abi::kDefinedPatchMetrics),
        abi::kDefinedPatchMetrics
    )
);
static_assert(abi::defined_mask_has_only_frozen_bits(0x00000000FFFFFFFFULL));
static_assert(!abi::defined_mask_has_only_frozen_bits(1ULL << 32));
static_assert(
    abi::set_defined_bit(
        0,
        static_cast<abi::DefinedBitV1>(abi::kDefinedBitCountV1)
    ) == 0
);
static_assert(
    abi::reserved_invalid_status_code(abi::kStatusWorld)
    == abi::kReservedInvalidStatusCodeV2
);

static_assert(abi::kInvalidRowContractV2.validity == 0);
static_assert(abi::kInvalidRowContractV2.defined_mask == 0);
static_assert(abi::kInvalidRowContractV2.status_code == 0);
static_assert(abi::kInvalidRowContractV2.integer_value == 0);
static_assert(abi::kInvalidRowContractV2.float_value == 0.0);
static_assert(abi::kInvalidRowContractV2.digest_byte == 0);
static_assert(abi::kInvalidRowContractV2.text_length == 0);
static_assert(abi::kInvalidRowContractV2.text_tail_byte == 0);
static_assert(abi::kInvalidRowContractV2.float_is_positive_zero);
static_assert(abi::kInvalidRowContractV2.validity_written_last);
static_assert(!abi::kInvalidRowContractV2.canonical_projection_permitted);
static_assert(
    !abi::kInvalidRowContractV2.completed_generation_count_includes_row
);
static_assert(
    abi::kInvalidRowContractV2
        .completed_rows_before_failure_remain_valid_and_immutable
);
static_assert(
    !abi::kInvalidRowContractV2
         .later_simulation_mutation_after_sticky_failure
);
static_assert(!abi::kInvalidRowContractV2.default_values_imply_definedness);
static_assert(
    abi::kInvalidRowContractV2.phase_a_nan_telemetry_is_separate_authority
);

static_assert(std::is_standard_layout_v<BytePlane>);
static_assert(std::is_trivially_copyable_v<BytePlane>);
static_assert(std::is_standard_layout_v<MaskPlane>);
static_assert(std::is_trivially_copyable_v<MaskPlane>);
static_assert(std::is_standard_layout_v<SummaryView>);
static_assert(std::is_trivially_copyable_v<SummaryView>);
static_assert(
    std::is_same_v<decltype(SummaryView::status_codes.data), std::int32_t*>
);
static_assert(
    std::is_same_v<decltype(SummaryView::integer_values.data), std::int64_t*>
);
static_assert(std::is_same_v<decltype(SummaryView::float_values.data), double*>);

__device__ void compile_compact_summary_v2_device_probe(
    const SummaryView view,
    const std::int64_t generation,
    const std::int64_t cell
) {
    abi::DefinedMaskV1* const mask =
        abi::row_pointer(view.defined_mask, generation, cell);
    std::int32_t* const statuses =
        abi::row_pointer(view.status_codes, generation, cell);
    std::uint8_t* const validity =
        abi::row_pointer(view.validity, generation, cell);

    *mask = abi::set_defined_bit(*mask, abi::kDefinedIdentityActiveAgentIdMin);
    statuses[abi::kStatusWorld] = abi::reserved_invalid_status_code(
        abi::kStatusWorld
    );
    validity[0] = abi::test_defined_bit(
        *mask,
        abi::kDefinedIdentityActiveAgentIdMin
    );
}
