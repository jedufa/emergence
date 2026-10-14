#pragma once

#include "../../perf/genome_upgrade_v2_q10_v1.cuh"
#include "tw_rngv2.cuh"

#include <cstdint>

#if defined(__CUDACC__)
#define EIEL_GV2_HD __host__ __device__
#define EIEL_GV2_INLINE __forceinline__
#else
#define EIEL_GV2_HD
#define EIEL_GV2_INLINE inline
#endif

namespace eiel::genome_upgrade_v2_q10_v1 {

inline constexpr uint32_t kRngv2DomainRecombination = 0x0320u;
inline constexpr uint32_t kRngv2DomainGv2Homolog = 0x0340u;
inline constexpr uint32_t kRngv2DomainGv2Gate = 0x0341u;
inline constexpr uint32_t kRngv2DomainGv2Delta = 0x0342u;
inline constexpr uint32_t kRngv2SemanticEpoch = 1u;
inline constexpr uint64_t kRngv2NamespaceDefault = 0xCA05117E450A51C0ull;

EIEL_GV2_HD EIEL_GV2_INLINE bool rngv2_uniform_v1(
    const uint32_t world_uid,
    const uint32_t domain_id,
    const uint64_t subject_uid,
    const uint64_t absolute_tick,
    const uint32_t operation_id,
    const uint64_t namespace_key,
    double* const value
) {
    const int32_t field_status = tw_rngv2::validate_fields(
        world_uid,
        domain_id,
        kRngv2SemanticEpoch,
        subject_uid,
        absolute_tick,
        operation_id,
        0
    );
    if (field_status != tw_rngv2::RNG_OK) {
        return false;
    }
    uint32_t k0 = 0;
    uint32_t k1 = 0;
    uint32_t c0 = 0;
    uint32_t c1 = 0;
    uint32_t c2 = 0;
    uint32_t c3 = 0;
    tw_rngv2::encode_key(
        namespace_key, world_uid, domain_id, kRngv2SemanticEpoch, &k0, &k1
    );
    tw_rngv2::encode_counter(
        subject_uid, absolute_tick, operation_id, 0, &c0, &c1, &c2, &c3
    );
    const tw_rngv2::PhiloxWords words =
        tw_rngv2::philox4x32_10(c0, c1, c2, c3, k0, k1);
    const uint64_t packed =
        static_cast<uint64_t>(words.w0)
        | (static_cast<uint64_t>(words.w1) << 32);
    *value = tw_rngv2::u64_to_f64_half_open(packed);
    return true;
}

EIEL_GV2_HD EIEL_GV2_INLINE bool rngv2_bounded_v1(
    const uint64_t upper,
    const uint32_t world_uid,
    const uint32_t domain_id,
    const uint64_t subject_uid,
    const uint64_t absolute_tick,
    const uint32_t operation_id,
    const uint64_t namespace_key,
    uint32_t* const value
) {
    int32_t accepted = 0;
    int32_t status = tw_rngv2::RNG_OK;
    tw_rngv2::bounded_uint32(
        upper,
        namespace_key,
        world_uid,
        domain_id,
        kRngv2SemanticEpoch,
        subject_uid,
        absolute_tick,
        operation_id,
        value,
        &accepted,
        &status
    );
    return status == tw_rngv2::RNG_OK;
}

EIEL_GV2_HD EIEL_GV2_INLINE bool stage_child_from_genotypes_v2(
    const GenotypeV1& parent_a,
    const GenotypeV1& parent_b,
    const uint32_t world_uid,
    const uint64_t absolute_tick,
    const uint64_t subject_uid,
    const int64_t mutation_rate_q10,
    const int64_t mutation_step_q10,
    StagedChildV1* staged,
    const uint64_t namespace_key = kRngv2NamespaceDefault
) {
    if (
        staged == nullptr
        || mutation_rate_q10 < 0 || mutation_rate_q10 > kScale
        || mutation_step_q10 < 0 || mutation_step_q10 > kScale
    ) {
        return false;
    }
    uint8_t donors[kMeiosisSemanticDrawCount];
    for (int32_t draw = 0; draw < kMeiosisSemanticDrawCount; ++draw) {
        uint32_t value = 0;
        if (
            !rngv2_bounded_v1(
                2,
                world_uid,
                kRngv2DomainRecombination,
                subject_uid,
                absolute_tick,
                static_cast<uint32_t>(draw),
                namespace_key,
                &value
            )
        ) {
            return false;
        }
        donors[draw] = static_cast<uint8_t>(value);
    }
    const GenotypeV1* const parents[kHomologCount] = {&parent_a, &parent_b};
    for (int32_t child_homolog = 0; child_homolog < kHomologCount; ++child_homolog) {
        const int32_t donor_offset = child_homolog * kLinkageBlockCount;
        for (int32_t product = 0; product < kProductCount; ++product) {
            const int32_t donor = donors[donor_offset + product / 2];
            for (int32_t parameter = 0; parameter < kParameterCount; ++parameter) {
                staged->genotype.alleles[child_homolog][product][parameter] =
                    parents[child_homolog]->alleles[donor][product][parameter];
            }
            staged->genotype.modifiers[child_homolog][product] =
                parents[child_homolog]->modifiers[donor][product];
        }
    }
    staged->changed_loci = 0;
    const double threshold =
        static_cast<double>(mutation_rate_q10) / static_cast<double>(kScale);
    const uint64_t delta_width =
        static_cast<uint64_t>(mutation_step_q10 * 2 + 1);
    int32_t locus = 0;
    for (int32_t product = 0; product < kProductCount; ++product) {
        for (int32_t parameter = 0; parameter < kParameterCount; ++parameter) {
            uint32_t homolog_raw = 0;
            uint32_t delta_raw = 0;
            double gate = 0.0;
            if (
                !rngv2_bounded_v1(
                    2,
                    world_uid,
                    kRngv2DomainGv2Homolog,
                    subject_uid,
                    absolute_tick,
                    static_cast<uint32_t>(locus),
                    namespace_key,
                    &homolog_raw
                )
                || !rngv2_uniform_v1(
                    world_uid,
                    kRngv2DomainGv2Gate,
                    subject_uid,
                    absolute_tick,
                    static_cast<uint32_t>(locus),
                    namespace_key,
                    &gate
                )
                || !rngv2_bounded_v1(
                    delta_width,
                    world_uid,
                    kRngv2DomainGv2Delta,
                    subject_uid,
                    absolute_tick,
                    static_cast<uint32_t>(locus),
                    namespace_key,
                    &delta_raw
                )
            ) {
                return false;
            }
            if (gate < threshold) {
                int64_t* const value =
                    &staged->genotype.alleles[homolog_raw][product][parameter];
                const int64_t before = *value;
                *value = clamp_i64(
                    before + static_cast<int64_t>(delta_raw) - mutation_step_q10,
                    parameter_lower_bound(parameter),
                    parameter_upper_bound(parameter)
                );
                staged->changed_loci += *value != before ? 1 : 0;
            }
            ++locus;
        }
        uint32_t homolog_raw = 0;
        uint32_t delta_raw = 0;
        double gate = 0.0;
        if (
            !rngv2_bounded_v1(
                2,
                world_uid,
                kRngv2DomainGv2Homolog,
                subject_uid,
                absolute_tick,
                static_cast<uint32_t>(locus),
                namespace_key,
                &homolog_raw
            )
            || !rngv2_uniform_v1(
                world_uid,
                kRngv2DomainGv2Gate,
                subject_uid,
                absolute_tick,
                static_cast<uint32_t>(locus),
                namespace_key,
                &gate
            )
            || !rngv2_bounded_v1(
                delta_width,
                world_uid,
                kRngv2DomainGv2Delta,
                subject_uid,
                absolute_tick,
                static_cast<uint32_t>(locus),
                namespace_key,
                &delta_raw
            )
        ) {
            return false;
        }
        if (gate < threshold) {
            int64_t* const value =
                &staged->genotype.modifiers[homolog_raw][product];
            const int64_t before = *value;
            *value = clamp_i64(
                before + static_cast<int64_t>(delta_raw) - mutation_step_q10,
                -kScale,
                kScale
            );
            staged->changed_loci += *value != before ? 1 : 0;
        }
        ++locus;
    }
    staged->semantic_draws = kInheritanceSemanticDrawCount;
    return locus == kHereditaryLocusCount;
}

EIEL_GV2_HD EIEL_GV2_INLINE bool stage_child_v2(
    const ResidentPlanesV1& planes,
    const int64_t canonical_parent_0_slot,
    const int64_t canonical_parent_1_slot,
    const uint32_t world_uid,
    const uint64_t absolute_tick,
    const uint64_t subject_uid,
    const int64_t mutation_rate_q10,
    const int64_t mutation_step_q10,
    StagedChildV1* staged,
    const uint64_t namespace_key = kRngv2NamespaceDefault
) {
    if (
        !valid_planes(planes)
        || staged == nullptr
        || canonical_parent_0_slot < 0
        || canonical_parent_0_slot >= planes.slot_count
        || canonical_parent_1_slot < 0
        || canonical_parent_1_slot >= planes.slot_count
    ) {
        return false;
    }
    GenotypeV1 parents[kHomologCount];
    load_genotype_v1(planes, canonical_parent_0_slot, &parents[0]);
    load_genotype_v1(planes, canonical_parent_1_slot, &parents[1]);
    return stage_child_from_genotypes_v2(
        parents[0],
        parents[1],
        world_uid,
        absolute_tick,
        subject_uid,
        mutation_rate_q10,
        mutation_step_q10,
        staged,
        namespace_key
    );
}

}  // namespace eiel::genome_upgrade_v2_q10_v1

#undef EIEL_GV2_HD
#undef EIEL_GV2_INLINE
