#include "cpu_compact_summary_v1.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace py = pybind11;
namespace encoder = eiel::native::cpu_compact_summary_v1;
namespace compact_abi = eiel::cuda_architecture_b::compact_summary_v2;

namespace {

template <typename T, std::size_t Size>
py::bytes plane_bytes(const std::array<T, Size>& plane) {
    return py::bytes(
        reinterpret_cast<const char*>(plane.data()),
        static_cast<py::ssize_t>(sizeof(plane))
    );
}

template <typename T, std::size_t Size>
bool plane_is_bit_zero(const std::array<T, Size>& plane) {
    const auto* const bytes =
        reinterpret_cast<const unsigned char*>(plane.data());
    for (std::size_t index = 0; index < sizeof(plane); ++index) {
        if (bytes[index] != 0U) {
            return false;
        }
    }
    return true;
}

bool row_is_bit_zero(const encoder::CpuCompactSummaryRowV1& row) {
    return plane_is_bit_zero(row.validity)
        && plane_is_bit_zero(row.defined_mask)
        && plane_is_bit_zero(row.status_codes)
        && plane_is_bit_zero(row.integer_values)
        && plane_is_bit_zero(row.float_values)
        && plane_is_bit_zero(row.digest_values)
        && plane_is_bit_zero(row.text_values);
}

void dirty_row(encoder::CpuCompactSummaryRowV1& row) {
    row.validity.fill(0xA5U);
    row.defined_mask.fill(0xA5A5A5A5A5A5A5A5ULL);
    row.status_codes.fill(-1);
    row.integer_values.fill(-1);
    row.float_values.fill(1.25);
    row.digest_values.fill(0xA5U);
    row.text_values.fill(0xA5U);
}

encoder::CpuCompactSummarySourceAuthorityV1 flat_source() {
    encoder::CpuCompactSummarySourceAuthorityV1 source{};
    source.authority_version = encoder::kSourceAuthorityVersionV1;

    constexpr std::array<compact_abi::DefinedBitV1, 12> kDefinedFields = {
        compact_abi::kDefinedIdentityActiveAgentIdMin,
        compact_abi::kDefinedIdentityActiveAgentIdMax,
        compact_abi::kDefinedLineageMetricsActiveLineages,
        compact_abi::kDefinedLineageMetricsDominantLineageId,
        compact_abi::kDefinedLineageMetricsDominantLineagePopulationShare,
        compact_abi::kDefinedLineageMetricsTop3LineagePopulationShare,
        compact_abi::kDefinedLineageMetricsLineagePopulationEntropy,
        compact_abi::kDefinedLineageMetricsLineageReproductionEntropy,
        compact_abi::kDefinedLineageMetricsDominantLineageReproductionShare,
        compact_abi::kDefinedLineageMetricsRarePersistentLineages,
        compact_abi::kDefinedLineageMetricsLineageCollapseWarning,
        compact_abi::kDefinedLineageMetricsLineageCollapseStreak,
    };
    for (const compact_abi::DefinedBitV1 bit : kDefinedFields) {
        source.defined_mask = compact_abi::set_defined_bit(source.defined_mask, bit);
    }

    source.status_codes[compact_abi::kStatusSummaryStatus] =
        compact_abi::kSummaryStatusComplete;
    source.status_codes[compact_abi::kStatusWorld] = compact_abi::kWorldFlat;
    source.status_codes[compact_abi::kStatusTurnoverStatus] =
        compact_abi::kTurnoverStatusCommitted;
    source.status_codes[compact_abi::kStatusRngAuthorityComplete] =
        compact_abi::kBooleanStatusTrue;
    source.status_codes[compact_abi::kStatusRngExportNonConsuming] =
        compact_abi::kBooleanStatusTrue;
    source.status_codes[compact_abi::kStatusTerminalStatus] =
        compact_abi::kTerminalStatusNotTerminal;
    source.status_codes[compact_abi::kStatusTerminalRecoverableErrorStatus] =
        compact_abi::kRecoverableErrorStatusNone;
    source.status_codes[compact_abi::kStatusTerminalStickyFailure] =
        compact_abi::kBooleanStatusFalse;
    source.status_codes[compact_abi::kStatusCheckpointDue] =
        compact_abi::kBooleanStatusFalse;
    source.status_codes[compact_abi::kStatusLineageMetricsLineageCollapseWarning] =
        compact_abi::kBooleanStatusFalse;

    source.integer_values[compact_abi::kIntegerGenerationStartGeneration] = 3;
    source.integer_values[compact_abi::kIntegerGenerationStartTick] = 0;
    source.integer_values[compact_abi::kIntegerGenerationEndGeneration] = 3;
    source.integer_values[compact_abi::kIntegerGenerationEndTickExclusive] = 80;
    source.integer_values[compact_abi::kIntegerNextGenerationStartGeneration] = 4;
    source.integer_values[compact_abi::kIntegerNextGenerationStartTick] = 0;
    source.integer_values[compact_abi::kIntegerAlive] = 6;
    source.integer_values[compact_abi::kIntegerBirths] = 2;
    source.integer_values[compact_abi::kIntegerFailedBirths] = 1;
    source.integer_values[compact_abi::kIntegerDeaths] = 1;
    source.integer_values[compact_abi::kIntegerDeathsByClassEnergyDepleted] = 1;
    source.integer_values[compact_abi::kIntegerDeathsByClassHealthDepleted] = 0;
    source.integer_values[compact_abi::kIntegerTurnoverCount] = 1;
    source.integer_values[compact_abi::kIntegerEventsCursorStart] = 10;
    source.integer_values[compact_abi::kIntegerEventsCursorPreTurnover] = 14;
    source.integer_values[compact_abi::kIntegerEventsCursorEnd] = 16;
    source.integer_values[compact_abi::kIntegerEventsAuthoritativeCursor] = 16;
    source.integer_values[compact_abi::kIntegerEventsEventRowCount] = 6;
    source.integer_values[compact_abi::kIntegerEventsGenerationEventCount] = 4;
    source.integer_values[compact_abi::kIntegerEventsTurnoverEventCount] = 2;
    source.integer_values[
        compact_abi::kIntegerEventsGenerationCountsByTypeBirthCommitted
    ] = 2;
    source.integer_values[
        compact_abi::kIntegerEventsGenerationCountsByTypeMutation
    ] = 0;
    source.integer_values[
        compact_abi::kIntegerEventsGenerationCountsByTypeBirthFailed
    ] = 1;
    source.integer_values[
        compact_abi::kIntegerEventsGenerationCountsByTypeDeathCommitted
    ] = 1;
    source.integer_values[
        compact_abi::kIntegerEventsTurnoverCountsByTypeBirthCommitted
    ] = 0;
    source.integer_values[
        compact_abi::kIntegerEventsTurnoverCountsByTypeMutation
    ] = 2;
    source.integer_values[
        compact_abi::kIntegerEventsTurnoverCountsByTypeBirthFailed
    ] = 0;
    source.integer_values[
        compact_abi::kIntegerEventsTurnoverCountsByTypeDeathCommitted
    ] = 0;
    source.integer_values[compact_abi::kIntegerIdentityPopulationNextId] = 10;
    source.integer_values[compact_abi::kIntegerIdentityActiveAgentCount] = 8;
    source.integer_values[compact_abi::kIntegerIdentityActiveAgentIdMin] = 1;
    source.integer_values[compact_abi::kIntegerIdentityActiveAgentIdMax] = 9;
    source.integer_values[compact_abi::kIntegerIdentityFreeSlotCount] = 2;
    source.integer_values[compact_abi::kIntegerRngPolicyMutationTapeCursor] = 1;
    source.integer_values[compact_abi::kIntegerGenerationMetricsCapacity] = 10;
    source.integer_values[compact_abi::kIntegerGenerationMetricsFreeSlots] = 4;
    source.integer_values[compact_abi::kIntegerGenerationMetricsBirthAttempts] = 3;
    source.integer_values[
        compact_abi::kIntegerGenerationMetricsNativeFailedBirthEventRows
    ] = 1;
    source.integer_values[
        compact_abi::kIntegerGenerationMetricsInjuryBlockedBirths
    ] = 0;
    source.integer_values[compact_abi::kIntegerGenerationMetricsMoved] = 4;
    source.integer_values[compact_abi::kIntegerGenerationMetricsGathered] = 2;
    source.integer_values[
        compact_abi::kIntegerGenerationMetricsMovementAttempts
    ] = 8;
    source.integer_values[
        compact_abi::kIntegerGenerationMetricsMovementBlockedOccupied
    ] = 3;
    source.integer_values[
        compact_abi::kIntegerGenerationMetricsMovementBlockedBounds
    ] = 1;
    source.integer_values[
        compact_abi::kIntegerGenerationMetricsPolicyLoopTickCalls
    ] = 80;
    source.integer_values[
        compact_abi::kIntegerGenerationMetricsPolicyRefillCalls
    ] = 2;
    source.integer_values[
        compact_abi::kIntegerGenerationMetricsPolicyRefillRows
    ] = 5;
    source.integer_values[
        compact_abi::kIntegerGenerationMetricsPolicyCacheHitCalls
    ] = 78;
    source.integer_values[
        compact_abi::kIntegerGenerationMetricsPolicyCacheRowsRequested
    ] = 20;
    source.integer_values[
        compact_abi::kIntegerGenerationMetricsPolicyCacheRowsHit
    ] = 15;
    source.integer_values[
        compact_abi::kIntegerGenerationMetricsPolicyCacheRowsMissed
    ] = 5;
    source.integer_values[
        compact_abi::kIntegerGenerationMetricsPolicyCacheRowsRefilled
    ] = 5;
    source.integer_values[
        compact_abi::kIntegerGenerationMetricsPolicyCacheFullRefillTicks
    ] = 1;
    source.integer_values[
        compact_abi::kIntegerGenerationMetricsPolicyCachePartialRefillTicks
    ] = 1;
    source.integer_values[compact_abi::kIntegerBehaviorMetricsRowCount] = 2;
    source.integer_values[compact_abi::kIntegerBehaviorMetricsCohortSize] = 1;
    source.integer_values[compact_abi::kIntegerBehaviorMetricsCohortSurvivors] = 1;
    source.integer_values[compact_abi::kIntegerBehaviorMetricsActionTotal] = 13;
    source.integer_values[compact_abi::kIntegerBehaviorMetricsMoveTotal] = 7;
    source.integer_values[compact_abi::kIntegerBehaviorMetricsMoveN] = 3;
    source.integer_values[compact_abi::kIntegerBehaviorMetricsMoveS] = 2;
    source.integer_values[compact_abi::kIntegerBehaviorMetricsMoveE] = 0;
    source.integer_values[compact_abi::kIntegerBehaviorMetricsMoveW] = 2;
    source.integer_values[compact_abi::kIntegerBehaviorMetricsGather] = 4;
    source.integer_values[compact_abi::kIntegerBehaviorMetricsRest] = 2;
    source.integer_values[compact_abi::kIntegerBehaviorMetricsAttack] = 0;
    source.integer_values[compact_abi::kIntegerLineageMetricsRecordCount] = 2;
    source.integer_values[compact_abi::kIntegerLineageMetricsActiveLineages] = 2;
    source.integer_values[
        compact_abi::kIntegerLineageMetricsRarePersistentLineages
    ] = 1;
    source.integer_values[
        compact_abi::kIntegerLineageMetricsLineageCollapseStreak
    ] = 0;

    source.float_values[compact_abi::kFloatGenerationMetricsPeakInjury] = 0.5;
    source.float_values[
        compact_abi::kFloatLineageMetricsDominantLineagePopulationShare
    ] = 0.5;
    source.float_values[
        compact_abi::kFloatLineageMetricsTop3LineagePopulationShare
    ] = 1.0;
    source.float_values[
        compact_abi::kFloatLineageMetricsLineagePopulationEntropy
    ] = 0.9;
    source.float_values[
        compact_abi::kFloatLineageMetricsLineageReproductionEntropy
    ] = 0.8;
    source.float_values[
        compact_abi::kFloatLineageMetricsDominantLineageReproductionShare
    ] = 0.5;

    for (
        std::int64_t field = 0;
        field < compact_abi::kDigestFieldCountV2;
        ++field
    ) {
        if (field == compact_abi::kDigestTerminalRecoverableErrorDigestSha256) {
            continue;
        }
        source.digest_values[field].bytes.fill(
            static_cast<std::uint8_t>(0xA0 + field)
        );
    }

    constexpr char kLineageId[] = "L00000001";
    source.text_values[compact_abi::kTextLineageMetricsDominantLineageId].length =
        static_cast<std::uint8_t>(sizeof(kLineageId) - 1U);
    std::memcpy(
        source.text_values[
            compact_abi::kTextLineageMetricsDominantLineageId
        ].payload.data(),
        kLineageId,
        sizeof(kLineageId) - 1U
    );
    return source;
}

py::dict row_to_dict(const encoder::CpuCompactSummaryRowV1& row) {
    py::dict result;
    result["validity"] = plane_bytes(row.validity);
    result["defined_mask"] = plane_bytes(row.defined_mask);
    result["status_codes"] = plane_bytes(row.status_codes);
    result["integer_values"] = plane_bytes(row.integer_values);
    result["float_values"] = plane_bytes(row.float_values);
    result["digest_values"] = plane_bytes(row.digest_values);
    result["text_values"] = plane_bytes(row.text_values);
    return result;
}

template <typename Mutator>
void append_failure(
    py::list& failures,
    const char* const name,
    Mutator mutator
) {
    encoder::CpuCompactSummarySourceAuthorityV1 source = flat_source();
    mutator(source);
    encoder::CpuCompactSummaryRowV1 row{};
    dirty_row(row);
    const encoder::EncodeStatusV1 status =
        encoder::encode_completed_row(source, row);
    py::dict result;
    result["name"] = name;
    result["status"] = encoder::encode_status_name(status);
    result["row_bit_zero"] = row_is_bit_zero(row);
    failures.append(result);
}

py::dict fixture_output() {
    encoder::CpuCompactSummaryRowV1 completed{};
    const encoder::EncodeStatusV1 status =
        encoder::encode_completed_row(flat_source(), completed);

    encoder::CpuCompactSummaryRowV1 invalid{};
    dirty_row(invalid);
    encoder::initialize_invalid_row(invalid);

    const compact_abi::CompactSummaryViewV2 view =
        encoder::compact_summary_view(completed);
    py::dict view_receipt;
    view_receipt["generation_count"] = view.generation_count;
    view_receipt["cell_count"] = view.cell_count;
    view_receipt["validity_stride"] = view.validity.generation_stride;
    view_receipt["status_stride"] = view.status_codes.generation_stride;
    view_receipt["integer_stride"] = view.integer_values.generation_stride;
    view_receipt["float_stride"] = view.float_values.generation_stride;
    view_receipt["digest_stride"] = view.digest_values.generation_stride;
    view_receipt["text_stride"] = view.text_values.generation_stride;
    view_receipt["pointers_match"] =
        view.validity.data == completed.validity.data()
        && view.defined_mask.data == completed.defined_mask.data()
        && view.status_codes.data == completed.status_codes.data()
        && view.integer_values.data == completed.integer_values.data()
        && view.float_values.data == completed.float_values.data()
        && view.digest_values.data == completed.digest_values.data()
        && view.text_values.data == completed.text_values.data();

    py::dict result;
    result["status"] = encoder::encode_status_name(status);
    result["completed"] = row_to_dict(completed);
    result["invalid"] = row_to_dict(invalid);
    result["view"] = view_receipt;
    return result;
}

py::dict abi_receipt() {
    py::dict plane_bytes_result;
    plane_bytes_result["validity"] = sizeof(encoder::ValidityPlaneRowV1);
    plane_bytes_result["defined_mask"] =
        sizeof(encoder::DefinedMaskPlaneRowV1);
    plane_bytes_result["status_codes"] = sizeof(encoder::StatusPlaneRowV1);
    plane_bytes_result["integer_values"] = sizeof(encoder::IntegerPlaneRowV1);
    plane_bytes_result["float_values"] = sizeof(encoder::FloatPlaneRowV1);
    plane_bytes_result["digest_values"] = sizeof(encoder::DigestPlaneRowV1);
    plane_bytes_result["text_values"] = sizeof(encoder::TextPlaneRowV1);

    py::dict result;
    result["plane_count"] = static_cast<std::int64_t>(compact_abi::kPlaneCountV2);
    result["status_count"] = static_cast<std::int64_t>(
        compact_abi::kStatusFieldCountV2
    );
    result["integer_count"] = static_cast<std::int64_t>(
        compact_abi::kIntegerFieldCountV2
    );
    result["float_count"] = static_cast<std::int64_t>(
        compact_abi::kFloatFieldCountV2
    );
    result["digest_count"] = static_cast<std::int64_t>(
        compact_abi::kDigestFieldCountV2
    );
    result["text_count"] = static_cast<std::int64_t>(
        compact_abi::kTextFieldCountV2
    );
    result["defined_bit_count"] = static_cast<std::int64_t>(
        compact_abi::kDefinedBitCountV1
    );
    result["raw_digest_bytes"] = sizeof(encoder::RawDigestV1);
    result["utf8_text_value_bytes"] = sizeof(encoder::Utf8TextValueV1);
    result["logical_row_bytes"] = compact_abi::kLogicalBytesPerGenerationCellV2;
    result["plane_bytes"] = plane_bytes_result;
    return result;
}

py::list validation_failures() {
    py::list failures;
    append_failure(failures, "source_authority_version_mismatch", [](auto& source) {
        source.authority_version = 0;
    });
    append_failure(failures, "reserved_defined_bit", [](auto& source) {
        source.defined_mask |= compact_abi::kReservedDefinedMaskV1;
    });
    append_failure(failures, "patch_child_without_object", [](auto& source) {
        source.defined_mask = compact_abi::set_defined_bit(
            source.defined_mask, compact_abi::kDefinedPatchMetricsPatchSpawnSyncRows
        );
    });
    append_failure(failures, "status_code_invalid", [](auto& source) {
        source.status_codes[compact_abi::kStatusSummaryStatus] = 0;
    });
    append_failure(failures, "world_patch_presence_mismatch", [](auto& source) {
        source.status_codes[compact_abi::kStatusWorld] = compact_abi::kWorldPatch;
    });
    append_failure(failures, "integer_range_invalid", [](auto& source) {
        source.integer_values[compact_abi::kIntegerAlive] = -1;
    });
    append_failure(failures, "float_non_finite", [](auto& source) {
        source.float_values[compact_abi::kFloatGenerationMetricsPeakInjury] =
            std::numeric_limits<double>::quiet_NaN();
    });
    append_failure(failures, "float_range_invalid", [](auto& source) {
        source.float_values[
            compact_abi::kFloatLineageMetricsDominantLineagePopulationShare
        ] = 1.25;
    });
    append_failure(failures, "undefined_storage_nonzero", [](auto& source) {
        source.integer_values[compact_abi::kIntegerTerminalFailureGeneration] = 1;
    });
    append_failure(failures, "flat_patch_storage_nonzero", [](auto& source) {
        source.integer_values[
            compact_abi::kIntegerPatchMetricsPatchMovementEnergyEventCount
        ] = 1;
    });
    append_failure(failures, "recoverable_state_inconsistent", [](auto& source) {
        source.status_codes[compact_abi::kStatusTerminalRecoverableErrorStatus] =
            compact_abi::kRecoverableErrorStatusPresent;
    });
    append_failure(failures, "terminal_state_inconsistent", [](auto& source) {
        source.status_codes[compact_abi::kStatusTerminalStatus] =
            compact_abi::kTerminalStatusFailed;
    });
    append_failure(failures, "text_length_invalid", [](auto& source) {
        source.text_values[compact_abi::kTextLineageMetricsDominantLineageId].length =
            static_cast<std::uint8_t>(compact_abi::kTextMaximumPayloadBytesV2 + 1);
    });
    append_failure(failures, "text_utf8_invalid", [](auto& source) {
        auto& text =
            source.text_values[compact_abi::kTextLineageMetricsDominantLineageId];
        text.length = 1;
        text.payload.fill(0);
        text.payload[0] = 0xFFU;
    });
    append_failure(failures, "text_tail_nonzero", [](auto& source) {
        auto& text =
            source.text_values[compact_abi::kTextLineageMetricsDominantLineageId];
        text.payload[text.length] = 1;
    });
    append_failure(failures, "required_digest_missing", [](auto& source) {
        source.digest_values[
            compact_abi::kDigestEventsActivePrefixDigestSha256
        ].bytes.fill(0);
    });
    return failures;
}

}  // namespace

PYBIND11_MODULE(eiel_cpu_compact_summary_v1_test, module) {
    module.def("abi_receipt", &abi_receipt);
    module.def("fixture_output", &fixture_output);
    module.def("validation_failures", &validation_failures);
}
