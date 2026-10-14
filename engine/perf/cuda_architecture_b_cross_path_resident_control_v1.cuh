#ifndef EIEL_CUDA_ARCHITECTURE_B_CROSS_PATH_RESIDENT_CONTROL_V1_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_CROSS_PATH_RESIDENT_CONTROL_V1_CUH_

#include <cstdint>

#include "cuda_architecture_b_resident_reproduction_exposure_v1.cuh"

namespace eiel::cuda_architecture_b::cross_path_resident_control_v1 {

#if defined(__CUDACC__)
#define EIEL_CROSS_PATH_HD __host__ __device__ __forceinline__
#else
#define EIEL_CROSS_PATH_HD inline
#endif

inline constexpr std::int8_t kReproductionPathGenerationBoundaryRefill = 0;
inline constexpr std::int8_t kReproductionPathOrdinaryLiveBirth = 1;
inline constexpr std::int8_t kSelectionModeGenotypeBlindNeutral = 0;
inline constexpr std::int8_t kInheritanceModeLegacyMidpoint = 0;
inline constexpr std::int8_t kInheritanceModeLinkedParentCopy = 1;
inline constexpr std::uint8_t kNoDonorOrdinal = 255;
inline constexpr std::uint8_t kDonorStatusOk = 0;
inline constexpr std::uint8_t kDonorStatusNotCommitted = 1;

enum InheritanceTransactionStatusV1 : std::int32_t {
    kInheritanceStatusNotCommitted = -1,
    kInheritanceStatusOk = 0,
    kInheritanceStatusInvalidControl = 1,
    kInheritanceStatusInvalidMode = 2,
    kInheritanceStatusInvalidParentIdentity = 3,
    kInheritanceStatusScalarBase = 100,
    kInheritanceStatusControllerBase = 1000,
};

// Generic resident authority for a mixed physical batch. All control planes
// are cell ordered. Slot identity and rank rows remain resident across ticks;
// birth rows are reused by the current transaction. A null reproduction path
// plane disables the extension and preserves the accepted controller ABI.
struct FlatCrossPathResidentControlV1 {
    const std::int8_t* reproduction_path_by_cell;
    const std::int8_t* selection_mode_by_cell;
    const std::int8_t* inheritance_mode_by_cell;
    const bool* mutation_enabled_by_cell;
    const bool* neutral_fixture_enabled_by_cell;
    const bool* trait_authority_enabled_by_cell;
    const std::uint64_t* selection_seed_by_cell;
    const double* neutral_repro_threshold_by_cell;
    // Generation comes from the controller mechanics configuration; every
    // cell in one physical batch advances under that same generation scalar.
    std::int64_t* genotype_codes;
    const std::int64_t* digest_rank_by_code;
    std::int64_t genotype_count;
    std::int64_t inheritance_rows_per_cell;
    std::int64_t* canonical_parent_0_slots;
    std::int64_t* canonical_parent_1_slots;
    std::int64_t* child_slots;
    std::uint8_t* donor_ordinals;
    std::uint8_t* donor_statuses;
    std::int64_t* staged_counts;
    std::int64_t* committed_counts;
    std::int64_t* donor_draws_added;
    std::int32_t* inheritance_status;
    resident_reproduction_exposure_v1::FlatResidentReproductionExposureV1
        reproduction_exposure{};
};

EIEL_CROSS_PATH_HD bool has_cross_path_authority_v1(
    const FlatCrossPathResidentControlV1& control
) noexcept {
    return control.reproduction_path_by_cell != nullptr;
}

EIEL_CROSS_PATH_HD bool has_complete_ordinary_authority_v1(
    const FlatCrossPathResidentControlV1& control
) noexcept {
    return control.reproduction_path_by_cell != nullptr
        && control.selection_mode_by_cell != nullptr
        && control.inheritance_mode_by_cell != nullptr
        && control.mutation_enabled_by_cell != nullptr
        && control.neutral_fixture_enabled_by_cell != nullptr
        && control.trait_authority_enabled_by_cell != nullptr
        && control.selection_seed_by_cell != nullptr
        && control.neutral_repro_threshold_by_cell != nullptr
        && control.genotype_codes != nullptr
        && control.digest_rank_by_code != nullptr
        && control.genotype_count > 0
        && control.inheritance_rows_per_cell > 0
        && control.canonical_parent_0_slots != nullptr
        && control.canonical_parent_1_slots != nullptr
        && control.child_slots != nullptr
        && control.donor_ordinals != nullptr
        && control.donor_statuses != nullptr
        && control.staged_counts != nullptr
        && control.committed_counts != nullptr
        && control.donor_draws_added != nullptr
        && control.inheritance_status != nullptr;
}

EIEL_CROSS_PATH_HD bool ordinary_live_birth_cell_v1(
    const FlatCrossPathResidentControlV1& control,
    const std::int64_t cell
) noexcept {
    return !has_cross_path_authority_v1(control)
        || (
            cell >= 0
            && control.reproduction_path_by_cell[cell]
                == kReproductionPathOrdinaryLiveBirth
        );
}

EIEL_CROSS_PATH_HD bool valid_reproduction_path_v1(
    const FlatCrossPathResidentControlV1& control,
    const std::int64_t cell
) noexcept {
    if (!has_cross_path_authority_v1(control)) {
        return true;
    }
    if (cell < 0) {
        return false;
    }
    const std::int8_t path = control.reproduction_path_by_cell[cell];
    return path == kReproductionPathGenerationBoundaryRefill
        || path == kReproductionPathOrdinaryLiveBirth;
}

#undef EIEL_CROSS_PATH_HD

}  // namespace eiel::cuda_architecture_b::cross_path_resident_control_v1

#endif
