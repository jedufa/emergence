#pragma once

#include <cstddef>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#if defined(__CUDACC__)
#define EIEL_CELL_DECISION_SUMMARY_V1_HOST_DEVICE __host__ __device__
#else
#define EIEL_CELL_DECISION_SUMMARY_V1_HOST_DEVICE
#endif

namespace eiel::cuda_architecture_b::cell_decision_summary_v1 {

inline constexpr std::uint32_t kContractVersionV1 = 1U;
inline constexpr char kContractSchemaVersionV1[] =
    "eiel_cuda_architecture_b_cell_decision_summary_v1";
inline constexpr std::int64_t kPolicyDigestBytesV1 = 32;

// Status zero preserves the native producer success convention. INT32_MIN is
// reserved so an unexecuted row cannot be mistaken for an arbitrary failure.
inline constexpr std::int32_t kCellStatusSuccessV1 = 0;
inline constexpr std::int32_t kCellStatusUnexecutedV1 =
    std::numeric_limits<std::int32_t>::min();
inline constexpr std::uint8_t kCellValidityExcludedV1 = 0U;
inline constexpr std::uint8_t kCellValidityRankableV1 = 1U;

inline constexpr bool kContainsFullSimulationStateV1 = false;
inline constexpr bool kFullStateTransferClaimedV1 = false;
inline constexpr bool kCompositeScientificScoreDefinedV1 = false;

struct ScorePolicyIdentityV1 {
    // The caller owns both byte ranges for at least the lifetime of the view.
    // policy_id is a nonempty stable identifier; policy_digest is exactly one
    // SHA-256 digest of the caller's frozen score-basis policy.
    const std::uint8_t* policy_id;
    std::int64_t policy_id_bytes;
    const std::uint8_t* policy_digest;
    std::int64_t policy_digest_bytes;
};

// All planes are caller-owned and C-contiguous. validity and status are [G,C],
// score_basis is [G,C,K], and stable_cell_ids is [C]. K is an ordered,
// higher-first lexicographic basis supplied by the identified policy. This ABI
// never normalizes, weights, combines, or assigns scientific meaning to it.
struct CellDecisionSummaryViewV1 {
    std::uint32_t contract_version;
    std::uint32_t reserved_zero;
    std::int64_t generation_count;
    std::int64_t cell_count;
    std::int64_t score_basis_count;
    ScorePolicyIdentityV1 score_policy;
    const std::uint8_t* validity;
    const std::int32_t* status;
    const double* score_basis;
    const std::int64_t* stable_cell_ids;
};

EIEL_CELL_DECISION_SUMMARY_V1_HOST_DEVICE constexpr bool valid_dimensions_v1(
    const std::int64_t generation_count,
    const std::int64_t cell_count,
    const std::int64_t score_basis_count
) noexcept {
    if (generation_count <= 0 || cell_count <= 0 || score_basis_count <= 0) {
        return false;
    }
    constexpr std::int64_t maximum = INT64_MAX;
    if (generation_count > maximum / cell_count) {
        return false;
    }
    const std::int64_t row_count = generation_count * cell_count;
    return row_count <= maximum / score_basis_count;
}

EIEL_CELL_DECISION_SUMMARY_V1_HOST_DEVICE constexpr std::int64_t
generation_cell_offset_v1(
    const std::int64_t generation,
    const std::int64_t cell,
    const std::int64_t cell_count
) noexcept {
    if (
        generation < 0 || cell < 0 || cell_count <= 0 || cell >= cell_count
        || generation
            > (INT64_MAX - cell) / cell_count
    ) {
        return -1;
    }
    return generation * cell_count + cell;
}

EIEL_CELL_DECISION_SUMMARY_V1_HOST_DEVICE constexpr std::int64_t
validity_offset_v1(
    const std::int64_t generation,
    const std::int64_t cell,
    const std::int64_t cell_count
) noexcept {
    return generation_cell_offset_v1(generation, cell, cell_count);
}

EIEL_CELL_DECISION_SUMMARY_V1_HOST_DEVICE constexpr std::int64_t
status_offset_v1(
    const std::int64_t generation,
    const std::int64_t cell,
    const std::int64_t cell_count
) noexcept {
    return generation_cell_offset_v1(generation, cell, cell_count);
}

EIEL_CELL_DECISION_SUMMARY_V1_HOST_DEVICE constexpr std::int64_t
score_basis_offset_v1(
    const std::int64_t generation,
    const std::int64_t cell,
    const std::int64_t basis,
    const std::int64_t cell_count,
    const std::int64_t score_basis_count
) noexcept {
    const std::int64_t row = generation_cell_offset_v1(
        generation, cell, cell_count
    );
    if (
        row < 0 || basis < 0 || score_basis_count <= 0
        || basis >= score_basis_count
        || row > (
            INT64_MAX - basis
        ) / score_basis_count
    ) {
        return -1;
    }
    return row * score_basis_count + basis;
}

EIEL_CELL_DECISION_SUMMARY_V1_HOST_DEVICE constexpr std::int64_t
stable_cell_id_offset_v1(const std::int64_t cell) noexcept {
    return cell >= 0 ? cell : -1;
}

EIEL_CELL_DECISION_SUMMARY_V1_HOST_DEVICE constexpr bool
cell_status_is_unexecuted_v1(const std::int32_t status) noexcept {
    return status == kCellStatusUnexecutedV1;
}

EIEL_CELL_DECISION_SUMMARY_V1_HOST_DEVICE constexpr bool
cell_status_is_failed_v1(const std::int32_t status) noexcept {
    return status != kCellStatusSuccessV1
        && status != kCellStatusUnexecutedV1;
}

EIEL_CELL_DECISION_SUMMARY_V1_HOST_DEVICE constexpr bool
validity_status_consistent_v1(
    const std::uint8_t validity,
    const std::int32_t status
) noexcept {
    return (
        validity == kCellValidityRankableV1
        && status == kCellStatusSuccessV1
    ) || (
        validity == kCellValidityExcludedV1
        && status != kCellStatusSuccessV1
    );
}

EIEL_CELL_DECISION_SUMMARY_V1_HOST_DEVICE constexpr bool
cell_is_rankable_v1(
    const std::uint8_t validity,
    const std::int32_t status
) noexcept {
    return validity == kCellValidityRankableV1
        && status == kCellStatusSuccessV1;
}

EIEL_CELL_DECISION_SUMMARY_V1_HOST_DEVICE constexpr std::size_t
logical_plane_bytes_v1(
    const std::size_t generation_count,
    const std::size_t cell_count,
    const std::size_t score_basis_count
) noexcept {
    constexpr std::size_t maximum = SIZE_MAX;
    if (
        generation_count == 0 || cell_count == 0 || score_basis_count == 0
        || generation_count > maximum / cell_count
        || score_basis_count > (
            maximum - sizeof(std::uint8_t) - sizeof(std::int32_t)
        ) / sizeof(double)
    ) {
        return 0;
    }
    const std::size_t row_count = generation_count * cell_count;
    const std::size_t row_bytes = sizeof(std::uint8_t)
        + sizeof(std::int32_t) + sizeof(double) * score_basis_count;
    if (
        row_count > maximum / row_bytes
        || cell_count > maximum / sizeof(std::int64_t)
    ) {
        return 0;
    }
    const std::size_t row_plane_bytes = row_count * row_bytes;
    const std::size_t id_bytes = cell_count * sizeof(std::int64_t);
    return row_plane_bytes <= maximum - id_bytes
        ? row_plane_bytes + id_bytes
        : 0;
}

EIEL_CELL_DECISION_SUMMARY_V1_HOST_DEVICE inline bool
finite_score_value_v1(const double value) noexcept {
    constexpr double maximum = DBL_MAX;
    return value == value && value <= maximum && value >= -maximum;
}

EIEL_CELL_DECISION_SUMMARY_V1_HOST_DEVICE inline bool
canonical_positive_zero_v1(const double value) noexcept {
#if defined(__CUDA_ARCH__)
    return __double_as_longlong(value) == 0LL;
#else
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits == 0U;
#endif
}

EIEL_CELL_DECISION_SUMMARY_V1_HOST_DEVICE inline bool
valid_view_metadata_v1(const CellDecisionSummaryViewV1& view) noexcept {
    return view.contract_version == kContractVersionV1
        && view.reserved_zero == 0U
        && valid_dimensions_v1(
            view.generation_count, view.cell_count, view.score_basis_count
        )
        && logical_plane_bytes_v1(
            static_cast<std::size_t>(view.generation_count),
            static_cast<std::size_t>(view.cell_count),
            static_cast<std::size_t>(view.score_basis_count)
        ) != 0U
        && view.score_policy.policy_id != nullptr
        && view.score_policy.policy_id_bytes > 0
        && view.score_policy.policy_id_bytes <= 256
        && view.score_policy.policy_digest != nullptr
        && view.score_policy.policy_digest_bytes == kPolicyDigestBytesV1
        && view.validity != nullptr
        && view.status != nullptr
        && view.score_basis != nullptr
        && view.stable_cell_ids != nullptr;
}

// A validation kernel or a host view over host-resident planes may call this
// serial oracle once at the generation boundary. It is not part of the tick loop.
EIEL_CELL_DECISION_SUMMARY_V1_HOST_DEVICE inline bool
valid_view_contents_serial_v1(const CellDecisionSummaryViewV1& view) noexcept {
    if (!valid_view_metadata_v1(view)) {
        return false;
    }
    bool digest_nonzero = false;
    for (std::int64_t index = 0; index < view.score_policy.policy_id_bytes; ++index) {
        const std::uint8_t value = view.score_policy.policy_id[index];
        if (value == 0U || value > 127U) {
            return false;
        }
    }
    for (std::int64_t index = 0; index < kPolicyDigestBytesV1; ++index) {
        digest_nonzero = digest_nonzero || view.score_policy.policy_digest[index] != 0U;
    }
    if (!digest_nonzero) {
        return false;
    }
    for (std::int64_t cell = 0; cell < view.cell_count; ++cell) {
        if (view.stable_cell_ids[cell] < 0) {
            return false;
        }
        for (std::int64_t previous = 0; previous < cell; ++previous) {
            if (view.stable_cell_ids[previous] == view.stable_cell_ids[cell]) {
                return false;
            }
        }
    }
    for (std::int64_t generation = 0; generation < view.generation_count; ++generation) {
        for (std::int64_t cell = 0; cell < view.cell_count; ++cell) {
            const std::int64_t row = generation_cell_offset_v1(
                generation, cell, view.cell_count
            );
            const std::uint8_t validity = view.validity[row];
            const std::int32_t status = view.status[row];
            if (!validity_status_consistent_v1(validity, status)) {
                return false;
            }
            for (std::int64_t basis = 0; basis < view.score_basis_count; ++basis) {
                const std::int64_t offset = score_basis_offset_v1(
                    generation,
                    cell,
                    basis,
                    view.cell_count,
                    view.score_basis_count
                );
                const double value = view.score_basis[offset];
                if (
                    (validity == kCellValidityRankableV1 && !finite_score_value_v1(value))
                    || (validity == kCellValidityExcludedV1
                        && !canonical_positive_zero_v1(value))
                ) {
                    return false;
                }
            }
        }
    }
    return true;
}

static_assert(sizeof(std::uint8_t) == 1);
static_assert(sizeof(std::int32_t) == 4);
static_assert(sizeof(std::int64_t) == 8);
static_assert(sizeof(double) == 8);
static_assert(std::numeric_limits<double>::is_iec559);
static_assert(std::is_standard_layout_v<ScorePolicyIdentityV1>);
static_assert(std::is_trivially_copyable_v<ScorePolicyIdentityV1>);
static_assert(sizeof(ScorePolicyIdentityV1) == 32);
static_assert(offsetof(ScorePolicyIdentityV1, policy_id) == 0);
static_assert(offsetof(ScorePolicyIdentityV1, policy_id_bytes) == 8);
static_assert(offsetof(ScorePolicyIdentityV1, policy_digest) == 16);
static_assert(offsetof(ScorePolicyIdentityV1, policy_digest_bytes) == 24);
static_assert(std::is_standard_layout_v<CellDecisionSummaryViewV1>);
static_assert(std::is_trivially_copyable_v<CellDecisionSummaryViewV1>);
static_assert(alignof(CellDecisionSummaryViewV1) == 8);
static_assert(sizeof(CellDecisionSummaryViewV1) == 96);
static_assert(offsetof(CellDecisionSummaryViewV1, contract_version) == 0);
static_assert(offsetof(CellDecisionSummaryViewV1, reserved_zero) == 4);
static_assert(offsetof(CellDecisionSummaryViewV1, generation_count) == 8);
static_assert(offsetof(CellDecisionSummaryViewV1, cell_count) == 16);
static_assert(offsetof(CellDecisionSummaryViewV1, score_basis_count) == 24);
static_assert(offsetof(CellDecisionSummaryViewV1, score_policy) == 32);
static_assert(offsetof(CellDecisionSummaryViewV1, validity) == 64);
static_assert(offsetof(CellDecisionSummaryViewV1, status) == 72);
static_assert(offsetof(CellDecisionSummaryViewV1, score_basis) == 80);
static_assert(offsetof(CellDecisionSummaryViewV1, stable_cell_ids) == 88);

}  // namespace eiel::cuda_architecture_b::cell_decision_summary_v1

#undef EIEL_CELL_DECISION_SUMMARY_V1_HOST_DEVICE
