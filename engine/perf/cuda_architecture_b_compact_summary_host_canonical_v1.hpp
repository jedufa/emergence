#pragma once

#include <cstdint>
#include <type_traits>

namespace eiel::cuda_architecture_b::compact_summary_reporting_authority_v1 {

struct GenerationAuthorityHeaderV1;
struct BehaviorAuthorityRowV1;
struct LineageAuthorityRowV1;

}  // namespace eiel::cuda_architecture_b::compact_summary_reporting_authority_v1

namespace eiel::cuda_architecture_b::compact_summary_host_canonical_v1 {

using compact_summary_reporting_authority_v1::BehaviorAuthorityRowV1;
using compact_summary_reporting_authority_v1::GenerationAuthorityHeaderV1;
using compact_summary_reporting_authority_v1::LineageAuthorityRowV1;

inline constexpr std::uint32_t kDigestBytesV1 = 32;
inline constexpr std::uint32_t kDominantLineageTextRecordBytesV1 = 65;
inline constexpr std::uint32_t kNoFailedRowV1 = UINT32_MAX;

inline constexpr std::uint64_t kCompactAbiV2DefinedLineageActiveV1 = 1ULL << 7U;
inline constexpr std::uint64_t kCompactAbiV2DefinedLineageDominantTextV1 =
    1ULL << 8U;
inline constexpr std::uint64_t kCompactAbiV2DefinedLineageDominantPopulationShareV1 =
    1ULL << 9U;
inline constexpr std::uint64_t kCompactAbiV2DefinedLineageTop3ShareV1 = 1ULL << 10U;
inline constexpr std::uint64_t kCompactAbiV2DefinedLineagePopulationEntropyV1 =
    1ULL << 11U;
inline constexpr std::uint64_t kCompactAbiV2DefinedLineageReproductionEntropyV1 =
    1ULL << 12U;
inline constexpr std::uint64_t
    kCompactAbiV2DefinedLineageDominantReproductionShareV1 = 1ULL << 13U;
inline constexpr std::uint64_t kCompactAbiV2DefinedLineageRarePersistentV1 =
    1ULL << 14U;
inline constexpr std::uint64_t kCompactAbiV2DefinedLineageCollapseWarningV1 =
    1ULL << 15U;
inline constexpr std::uint64_t kCompactAbiV2DefinedLineageCollapseStreakV1 =
    1ULL << 16U;

inline constexpr std::int32_t kCompactAbiV2StatusFalseV1 = 1;
inline constexpr std::int32_t kCompactAbiV2StatusTrueV1 = 2;

enum class HostCanonicalStatusV1 : std::uint32_t {
    kSuccess = 0,
    kNullArgument = 1,
    kUnsupportedHostEndian = 2,
    kInvalidGeneration = 3,
    kInvalidRowCapacity = 4,
    kLedgerNotSealed = 5,
    kInconsistentHeader = 6,
    kInactiveStorageNotZero = 7,
    kBehaviorOrder = 8,
    kBehaviorFlags = 9,
    kBehaviorCounter = 10,
    kBehaviorNonFinite = 11,
    kLineageOrder = 12,
    kLineageStorage = 13,
    kLineageCounter = 14,
    kLineageNonFinite = 15,
    kLineageBehaviorMismatch = 16,
    kNumberFormatting = 17,
    kLineageMetricsConfig = 18,
    kLineageMetricsState = 19,
    kAggregateOverflow = 20,
};

enum class HostCanonicalFieldV1 : std::uint32_t {
    kNone = 0,
    kHeader = 1,
    kBehaviorAgentId = 2,
    kBehaviorFlags = 3,
    kBehaviorCounters = 4,
    kBehaviorEnergyCollected = 5,
    kBehaviorFitness = 6,
    kBehaviorInjuryStart = 7,
    kBehaviorInjuryEnd = 8,
    kBehaviorAggressionBias = 9,
    kLineageAgentId = 10,
    kLineageStorage = 11,
    kLineageCounters = 12,
    kLineageEnergy = 13,
    kLineageEnergyCollected = 14,
    kLineageEpisode0PredErrMean = 15,
    kLineageIdentityDriftEma = 16,
    kLineageHealth = 17,
    kLineageInjury = 18,
    kLineageDrives = 19,
    kLineagePredErrSum = 20,
    kLineageDriveJitter = 21,
    kLineageAggressionBias = 22,
    kLineageExplorationBias = 23,
    kLineageMetricsConfig = 24,
    kLineageMetricsState = 25,
    kBehaviorAggregates = 26,
    kLineageAggregates = 27,
};

struct LineageMetricsStateV1 {
    std::int64_t collapse_streak;
    std::int64_t population_dominance_streak;
    std::int64_t reproduction_dominance_streak;
};

struct LineageMetricsConfigV1 {
    double population_dominance_threshold;
    double entropy_minimum;
    double reproduction_dominance_threshold;
    std::int64_t population_dominance_streak_required;
    std::int64_t minimum_active_lineages;
    std::int64_t reproduction_dominance_streak_required;
    std::uint8_t metrics_defined;
    std::uint8_t reserved[7];
};

struct LineageMetricsInputV1 {
    LineageMetricsStateV1 initial_state;
    LineageMetricsConfigV1 config;
};

inline constexpr LineageMetricsInputV1 kDefaultLineageMetricsInputV1 = {
    {0, 0, 0},
    {0.80, 0.10, 0.80, 5, 2, 5, 1, {0, 0, 0, 0, 0, 0, 0}},
};

struct BehaviorAggregateOutputV1 {
    std::int64_t row_count;
    std::int64_t cohort_size;
    std::int64_t cohort_survivors;
    std::int64_t action_total;
    std::int64_t move_total;
    std::int64_t move_n;
    std::int64_t move_s;
    std::int64_t move_e;
    std::int64_t move_w;
    std::int64_t gather;
    std::int64_t rest;
    std::int64_t attack;
};

struct LineageSummaryOutputV1 {
    std::uint64_t compact_abi_v2_defined_mask;
    std::int64_t record_count;
    std::int64_t active_lineages;
    std::int64_t rare_persistent_lineages;
    std::int64_t collapse_streak;
    double dominant_population_share;
    double top3_population_share;
    double population_entropy;
    double reproduction_entropy;
    double dominant_reproduction_share;
    std::int32_t collapse_warning_code;
    std::uint32_t reserved;
    std::uint8_t dominant_lineage_text_record[kDominantLineageTextRecordBytesV1];
    std::uint8_t reserved_tail[7];
};

struct HostCanonicalResultV1 {
    std::uint32_t status;
    std::uint32_t failed_field;
    std::uint32_t failed_row;
    std::uint32_t reserved;
    std::uint8_t behavior_rows_digest_sha256[kDigestBytesV1];
    std::uint8_t lineage_records_digest_sha256[kDigestBytesV1];
    BehaviorAggregateOutputV1 behavior;
    LineageSummaryOutputV1 lineage;
    LineageMetricsStateV1 lineage_state_after;
};

struct HostCanonicalRequestV1 {
    std::int64_t generation;
    const GenerationAuthorityHeaderV1* header;
    const BehaviorAuthorityRowV1* behavior_rows;
    const LineageAuthorityRowV1* lineage_rows;
    std::int32_t row_capacity;
    std::uint32_t reserved;
    LineageMetricsInputV1 lineage_metrics_input;
    HostCanonicalResultV1* result;
};

HostCanonicalStatusV1 canonicalize_retained_summary_v1(
    std::int64_t generation,
    const GenerationAuthorityHeaderV1* header,
    const BehaviorAuthorityRowV1* behavior_rows,
    const LineageAuthorityRowV1* lineage_rows,
    std::int32_t row_capacity,
    const LineageMetricsInputV1* lineage_metrics_input,
    HostCanonicalResultV1* result
) noexcept;

HostCanonicalStatusV1 canonicalize_retained_digests_v1(
    std::int64_t generation,
    const GenerationAuthorityHeaderV1* header,
    const BehaviorAuthorityRowV1* behavior_rows,
    const LineageAuthorityRowV1* lineage_rows,
    std::int32_t row_capacity,
    HostCanonicalResultV1* result
) noexcept;

void canonicalize_retained_digests_callback_v1(void* user_data) noexcept;
void canonicalize_retained_summary_callback_v1(void* user_data) noexcept;

static_assert(std::is_standard_layout_v<LineageMetricsStateV1>);
static_assert(std::is_trivially_copyable_v<LineageMetricsStateV1>);
static_assert(sizeof(LineageMetricsStateV1) == 24);
static_assert(std::is_standard_layout_v<LineageMetricsConfigV1>);
static_assert(std::is_trivially_copyable_v<LineageMetricsConfigV1>);
static_assert(sizeof(LineageMetricsConfigV1) == 56);
static_assert(std::is_standard_layout_v<LineageMetricsInputV1>);
static_assert(std::is_trivially_copyable_v<LineageMetricsInputV1>);
static_assert(sizeof(LineageMetricsInputV1) == 80);
static_assert(std::is_standard_layout_v<BehaviorAggregateOutputV1>);
static_assert(std::is_trivially_copyable_v<BehaviorAggregateOutputV1>);
static_assert(sizeof(BehaviorAggregateOutputV1) == 96);
static_assert(std::is_standard_layout_v<LineageSummaryOutputV1>);
static_assert(std::is_trivially_copyable_v<LineageSummaryOutputV1>);
static_assert(sizeof(LineageSummaryOutputV1) == 160);
static_assert(std::is_standard_layout_v<HostCanonicalResultV1>);
static_assert(std::is_trivially_copyable_v<HostCanonicalResultV1>);
static_assert(sizeof(HostCanonicalResultV1) == 360);
static_assert(std::is_standard_layout_v<HostCanonicalRequestV1>);
static_assert(std::is_trivially_copyable_v<HostCanonicalRequestV1>);

}  // namespace eiel::cuda_architecture_b::compact_summary_host_canonical_v1
