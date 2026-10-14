#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define EIEL_DIPLOID_HD __host__ __device__
#define EIEL_DIPLOID_FORCEINLINE __forceinline__
#else
#define EIEL_DIPLOID_HD
#define EIEL_DIPLOID_FORCEINLINE inline
#endif

namespace eiel::cuda_diploid_scalar_genome_v1 {

inline constexpr const char* kProfileId = "diploid_linkage_v1";
inline constexpr const char* kSchemaVersion = "diploid_genome_bundle_v1";
inline constexpr const char* kExpressionProfile =
    "diploid_additive_expression_v1";
inline constexpr const char* kRecombinationProfile =
    "fixed_linkage_homolog_donation_v1";
inline constexpr const char* kMutationProfile =
    "diploid_allele_mutation_after_assembly_v1";
inline constexpr const char* kLinkageMapSha256 =
    "2e31a544c7505aa072e91db929c895d9a47f1a7e67c359679b0998d9b856000c";
inline constexpr const char* kAdmittedLocusSha256 =
    "4e9d0a0d5a79fe64f9b50c554d803bcea4be89c90b867315b6f972fb8f36d629";

inline constexpr int32_t kLocusCount = 17;
inline constexpr int32_t kLinkageBlockCount = 5;
inline constexpr int32_t kHomologCount = 2;
inline constexpr int32_t kAllelePlaneCount =
    kLocusCount * kHomologCount;
inline constexpr int32_t kMeiosisDrawCount =
    kLinkageBlockCount * kHomologCount;
inline constexpr int32_t kMutationDrawCount = kLocusCount * 3;

enum LocusV1 : int32_t {
    kExplorationBias = 0,
    kAggressionBias,
    kActionTemperature,
    kDriveInertia,
    kDriveNoise,
    kDriveLogitGain,
    kPredLr,
    kDriveSpHunger,
    kDriveSpPain,
    kDriveSpFatigue,
    kDriveSpFear,
    kExpPriorStrength,
    kExpPriorNoise,
    kDriveBirthNoise,
    kReproductionThreshold,
    kMutationRate,
    kResourceHandlingAxis,
};

inline constexpr const char* kLocusNames[kLocusCount] = {
    "exploration_bias",
    "aggression_bias",
    "action_temperature",
    "drive_inertia",
    "drive_noise",
    "drive_logit_gain",
    "pred_lr",
    "drive_sp_hunger",
    "drive_sp_pain",
    "drive_sp_fatigue",
    "drive_sp_fear",
    "exp_prior_strength",
    "exp_prior_noise",
    "drive_birth_noise",
    "reproduction_threshold",
    "mutation_rate",
    "resource_handling_axis",
};

inline constexpr int32_t kLinkageBlockByLocus[kLocusCount] = {
    0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2,
    3,
    4,
};

inline constexpr double kLowerBounds[kLocusCount] = {
    -1.0, -2.0, 0.25,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    2.0, 0.001, -1.0,
};
inline constexpr double kUpperBounds[kLocusCount] = {
    1.0, 2.0, 2.5,
    0.999, 0.1, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.2,
    90.0, 0.2, 1.0,
};
inline constexpr double kMutationSigma[kLocusCount] = {
    0.10, 0.15, 0.10,
    0.05, 0.01, 0.20, 0.03,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    2.0, 0.005, 0.10,
};

enum class StatusV1 : int32_t {
    ok = 0,
    invalid_layout = 1,
    invalid_slot = 2,
    invalid_allele = 3,
    gaussian_tape_exhausted = 4,
};

// The two bases are SoA banks with layout [locus, slot]. Each locus slice is
// one independent contiguous float32 resident array. Together they own the
// required 34 allele arrays without per-agent objects or pointer chasing.
struct AllelePlanesV1 {
    float* allele_0;
    float* allele_1;
    int64_t slot_stride;
    int64_t slot_count;
};

// Existing phenotype arrays remain the simulation-facing scalar state. The
// resource-handling pointer is the one newly admitted expressed array.
struct PhenotypeArraysV1 {
    float* locus[kLocusCount];
};

struct StagedChildV1 {
    float allele_0[kLocusCount];
    float allele_1[kLocusCount];
    float phenotype[kLocusCount];
    uint8_t donor_homologs[kMeiosisDrawCount];
    uint8_t mutation_homologs[kLocusCount];
    uint8_t mutation_changed[kLocusCount];
    int64_t semantic_draws;
};

EIEL_DIPLOID_HD EIEL_DIPLOID_FORCEINLINE double clamp_double(
    const double value,
    const double lower,
    const double upper
) {
    return value < lower ? lower : (value > upper ? upper : value);
}

EIEL_DIPLOID_HD EIEL_DIPLOID_FORCEINLINE bool finite_double(
    const double value
) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

EIEL_DIPLOID_HD EIEL_DIPLOID_FORCEINLINE int32_t linkage_block(
    const int32_t locus
) {
    if (locus <= kActionTemperature) {
        return 0;
    }
    if (locus <= kDriveBirthNoise) {
        return 1;
    }
    return locus - kReproductionThreshold + 2;
}

EIEL_DIPLOID_HD EIEL_DIPLOID_FORCEINLINE double lower_bound(
    const int32_t locus
) {
    switch (locus) {
        case kExplorationBias:
        case kResourceHandlingAxis:
            return -1.0;
        case kAggressionBias:
            return -2.0;
        case kActionTemperature:
            return 0.25;
        case kReproductionThreshold:
            return 2.0;
        case kMutationRate:
            return 0.001;
        default:
            return 0.0;
    }
}

EIEL_DIPLOID_HD EIEL_DIPLOID_FORCEINLINE double upper_bound(
    const int32_t locus
) {
    switch (locus) {
        case kExplorationBias:
        case kPredLr:
        case kDriveSpHunger:
        case kDriveSpPain:
        case kDriveSpFatigue:
        case kDriveSpFear:
        case kExpPriorStrength:
        case kResourceHandlingAxis:
            return 1.0;
        case kAggressionBias:
            return 2.0;
        case kActionTemperature:
            return 2.5;
        case kDriveInertia:
            return 0.999;
        case kDriveNoise:
            return 0.1;
        case kDriveLogitGain:
            return 3.0;
        case kExpPriorNoise:
            return 0.5;
        case kDriveBirthNoise:
        case kMutationRate:
            return 0.2;
        case kReproductionThreshold:
            return 90.0;
        default:
            return 0.0;
    }
}

EIEL_DIPLOID_HD EIEL_DIPLOID_FORCEINLINE double mutation_sigma(
    const int32_t locus
) {
    switch (locus) {
        case kExplorationBias:
        case kActionTemperature:
        case kResourceHandlingAxis:
            return 0.10;
        case kAggressionBias:
            return 0.15;
        case kDriveInertia:
            return 0.05;
        case kDriveNoise:
            return 0.01;
        case kDriveLogitGain:
            return 0.20;
        case kPredLr:
            return 0.03;
        case kReproductionThreshold:
            return 2.0;
        case kMutationRate:
            return 0.005;
        default:
            return 0.0;
    }
}

EIEL_DIPLOID_HD EIEL_DIPLOID_FORCEINLINE double canon_double(
    const double value
) {
    if (!finite_double(value)) {
        return 0.0;
    }
    const float value32 = static_cast<float>(value);
    return ::nearbyint(static_cast<double>(value32) * 100000000.0)
        / 100000000.0;
}

EIEL_DIPLOID_HD EIEL_DIPLOID_FORCEINLINE float canon_float(
    const double value
) {
    return static_cast<float>(canon_double(value));
}

EIEL_DIPLOID_HD EIEL_DIPLOID_FORCEINLINE int64_t plane_index(
    const AllelePlanesV1& planes,
    const int32_t locus,
    const int64_t slot
) {
    return static_cast<int64_t>(locus) * planes.slot_stride + slot;
}

EIEL_DIPLOID_HD EIEL_DIPLOID_FORCEINLINE bool valid_layout(
    const AllelePlanesV1& planes
) {
    return planes.allele_0 != nullptr
        && planes.allele_1 != nullptr
        && planes.slot_stride >= planes.slot_count
        && planes.slot_count > 0;
}

EIEL_DIPLOID_HD EIEL_DIPLOID_FORCEINLINE StatusV1
stage_homozygous_migration_v1(
    const PhenotypeArraysV1& phenotype,
    const int64_t slot,
    StagedChildV1* staged
) {
    if (staged == nullptr || slot < 0) {
        return StatusV1::invalid_slot;
    }
    for (int32_t locus = 0; locus < kLocusCount; ++locus) {
        if (phenotype.locus[locus] == nullptr) {
            return StatusV1::invalid_layout;
        }
        const double value = static_cast<double>(phenotype.locus[locus][slot]);
        if (!finite_double(value)) {
            return StatusV1::invalid_allele;
        }
        const float canonical = canon_float(clamp_double(
            value, lower_bound(locus), upper_bound(locus)
        ));
        staged->allele_0[locus] = canonical;
        staged->allele_1[locus] = canonical;
        staged->phenotype[locus] = canonical;
        staged->mutation_homologs[locus] = 0;
        staged->mutation_changed[locus] = 0;
    }
    for (int32_t draw = 0; draw < kMeiosisDrawCount; ++draw) {
        staged->donor_homologs[draw] = 0;
    }
    staged->semantic_draws = 0;
    return StatusV1::ok;
}

template <typename RandomV1>
EIEL_DIPLOID_HD EIEL_DIPLOID_FORCEINLINE StatusV1 stage_child_v1(
    const AllelePlanesV1& planes,
    const int64_t canonical_parent_0_slot,
    const int64_t canonical_parent_1_slot,
    RandomV1& random,
    const bool mutation_enabled,
    const bool resource_handling_enabled,
    const double mutation_sigma_multiplier,
    StagedChildV1* staged
) {
    if (!valid_layout(planes) || staged == nullptr) {
        return StatusV1::invalid_layout;
    }
    if (
        canonical_parent_0_slot < 0
        || canonical_parent_0_slot >= planes.slot_count
        || canonical_parent_1_slot < 0
        || canonical_parent_1_slot >= planes.slot_count
    ) {
        return StatusV1::invalid_slot;
    }

    for (int32_t parent = 0; parent < kHomologCount; ++parent) {
        for (int32_t block = 0; block < kLinkageBlockCount; ++block) {
            staged->donor_homologs[
                parent * kLinkageBlockCount + block
            ] = random.random_double() < 0.5 ? 0 : 1;
        }
    }
    staged->semantic_draws = kMeiosisDrawCount;

    for (int32_t locus = 0; locus < kLocusCount; ++locus) {
        const int32_t block = linkage_block(locus);
        const uint8_t homolog_0 = staged->donor_homologs[block];
        const uint8_t homolog_1 =
            staged->donor_homologs[kLinkageBlockCount + block];
        const int64_t parent_0_index =
            plane_index(planes, locus, canonical_parent_0_slot);
        const int64_t parent_1_index =
            plane_index(planes, locus, canonical_parent_1_slot);
        const double allele_0 = homolog_0 == 0
            ? static_cast<double>(planes.allele_0[parent_0_index])
            : static_cast<double>(planes.allele_1[parent_0_index]);
        const double allele_1 = homolog_1 == 0
            ? static_cast<double>(planes.allele_0[parent_1_index])
            : static_cast<double>(planes.allele_1[parent_1_index]);
        if (!finite_double(allele_0) || !finite_double(allele_1)) {
            return StatusV1::invalid_allele;
        }
        staged->allele_0[locus] = canon_float(clamp_double(
            allele_0, lower_bound(locus), upper_bound(locus)
        ));
        staged->allele_1[locus] = canon_float(clamp_double(
            allele_1, lower_bound(locus), upper_bound(locus)
        ));
        staged->mutation_homologs[locus] = 0;
        staged->mutation_changed[locus] = 0;
    }

    const double pre_mutation_rate = canon_double(
        0.5 * static_cast<double>(staged->allele_0[kMutationRate])
        + 0.5 * static_cast<double>(staged->allele_1[kMutationRate])
    );
    if (mutation_enabled) {
        for (int32_t locus = 0; locus < kLocusCount; ++locus) {
            const double gate_draw = random.random_double();
            const double homolog_draw = random.random_double();
            double standard_normal = 0.0;
            if (!random.consume_gauss(1.0, &standard_normal)) {
                return StatusV1::gaussian_tape_exhausted;
            }
            const uint8_t homolog = homolog_draw < 0.5 ? 0 : 1;
            staged->mutation_homologs[locus] = homolog;
            float* selected = homolog == 0
                ? &staged->allele_0[locus]
                : &staged->allele_1[locus];
            const double before = static_cast<double>(*selected);
            const double gate_rate =
                locus == kMutationRate ? 0.2 : pre_mutation_rate;
            const bool operator_enabled =
                mutation_sigma(locus) > 0.0
                && (
                    locus != kResourceHandlingAxis
                    || resource_handling_enabled
                );
            const double delta =
                operator_enabled && gate_draw < gate_rate
                    ? standard_normal
                        * mutation_sigma(locus)
                        * mutation_sigma_multiplier
                    : 0.0;
            *selected = canon_float(clamp_double(
                before + delta,
                lower_bound(locus),
                upper_bound(locus)
            ));
            staged->mutation_changed[locus] =
                static_cast<double>(*selected) != before ? 1 : 0;
        }
        staged->semantic_draws += kMutationDrawCount;
    }

    for (int32_t locus = 0; locus < kLocusCount; ++locus) {
        staged->phenotype[locus] = canon_float(clamp_double(
            0.5 * static_cast<double>(staged->allele_0[locus])
                + 0.5 * static_cast<double>(staged->allele_1[locus]),
            lower_bound(locus),
            upper_bound(locus)
        ));
    }
    return StatusV1::ok;
}

EIEL_DIPLOID_HD EIEL_DIPLOID_FORCEINLINE StatusV1 commit_staged_child_v1(
    const AllelePlanesV1& planes,
    const PhenotypeArraysV1& phenotype,
    const int64_t child_slot,
    const StagedChildV1& staged
) {
    if (!valid_layout(planes)) {
        return StatusV1::invalid_layout;
    }
    if (child_slot < 0 || child_slot >= planes.slot_count) {
        return StatusV1::invalid_slot;
    }
    for (int32_t locus = 0; locus < kLocusCount; ++locus) {
        if (phenotype.locus[locus] == nullptr) {
            return StatusV1::invalid_layout;
        }
    }
    for (int32_t locus = 0; locus < kLocusCount; ++locus) {
        const int64_t index = plane_index(planes, locus, child_slot);
        planes.allele_0[index] = staged.allele_0[locus];
        planes.allele_1[index] = staged.allele_1[locus];
        phenotype.locus[locus][child_slot] = staged.phenotype[locus];
    }
    return StatusV1::ok;
}

}  // namespace eiel::cuda_diploid_scalar_genome_v1

#undef EIEL_DIPLOID_FORCEINLINE
#undef EIEL_DIPLOID_HD
