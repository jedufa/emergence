#pragma once

#include "../../perf/cuda_diploid_scalar_genome_v1.cuh"
#include "tw_rngv2.cuh"

#include <cstdint>

#if defined(__CUDACC__)
#define EIEL_DIPLOID_HD __host__ __device__
#define EIEL_DIPLOID_FORCEINLINE __forceinline__
#else
#define EIEL_DIPLOID_HD
#define EIEL_DIPLOID_FORCEINLINE inline
#endif

namespace eiel::cuda_diploid_scalar_genome_v1 {

inline constexpr uint32_t kRngv2DomainBirthProposal = 0x0310u;
inline constexpr uint32_t kRngv2SemanticEpoch = 1u;
inline constexpr uint64_t kRngv2NamespaceDefault = 0xCA05117E450A51C0ull;
inline constexpr uint32_t kRngv2OpMeiosis = 0u;
inline constexpr uint32_t kRngv2OpGate = 32u;
inline constexpr uint32_t kRngv2OpHomolog = 96u;
inline constexpr uint32_t kRngv2OpDelta = 160u;

EIEL_DIPLOID_HD EIEL_DIPLOID_FORCEINLINE bool rngv2_uniform_v1(
    const uint32_t world_uid,
    const uint32_t operation_id,
    const uint64_t subject_uid,
    const uint64_t absolute_tick,
    const uint64_t namespace_key,
    double* const value
) {
    const int32_t field_status = tw_rngv2::validate_fields(
        world_uid,
        kRngv2DomainBirthProposal,
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
        namespace_key,
        world_uid,
        kRngv2DomainBirthProposal,
        kRngv2SemanticEpoch,
        &k0,
        &k1
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

EIEL_DIPLOID_HD EIEL_DIPLOID_FORCEINLINE bool rngv2_bounded_v1(
    const uint64_t upper,
    const uint32_t world_uid,
    const uint32_t operation_id,
    const uint64_t subject_uid,
    const uint64_t absolute_tick,
    const uint64_t namespace_key,
    uint32_t* const value
) {
    int32_t accepted = 0;
    int32_t status = tw_rngv2::RNG_OK;
    tw_rngv2::bounded_uint32(
        upper,
        namespace_key,
        world_uid,
        kRngv2DomainBirthProposal,
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

EIEL_DIPLOID_HD EIEL_DIPLOID_FORCEINLINE bool rngv2_normal_v1(
    const uint32_t world_uid,
    const uint32_t operation_id,
    const uint64_t subject_uid,
    const uint64_t absolute_tick,
    const uint64_t namespace_key,
    double* const z0
) {
    const int32_t field_status = tw_rngv2::validate_fields(
        world_uid,
        kRngv2DomainBirthProposal,
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
        namespace_key,
        world_uid,
        kRngv2DomainBirthProposal,
        kRngv2SemanticEpoch,
        &k0,
        &k1
    );
    tw_rngv2::encode_counter(
        subject_uid, absolute_tick, operation_id, 0, &c0, &c1, &c2, &c3
    );
    const tw_rngv2::PhiloxWords words =
        tw_rngv2::philox4x32_10(c0, c1, c2, c3, k0, k1);
    uint64_t z0_bits = 0;
    uint64_t z1_bits = 0;
    int32_t nstatus = tw_rngv2::RNG_OK;
    tw_rngv2::box_muller(words, &z0_bits, &z1_bits, &nstatus);
    if (nstatus != tw_rngv2::RNG_OK) {
        return false;
    }
    *z0 = tw_rngv2::f64_from_bits(z0_bits);
    return true;
}

EIEL_DIPLOID_HD EIEL_DIPLOID_FORCEINLINE StatusV1 stage_child_v2(
    const AllelePlanesV1& planes,
    const int64_t canonical_parent_0_slot,
    const int64_t canonical_parent_1_slot,
    const uint32_t world_uid,
    const uint64_t absolute_tick,
    const uint64_t subject_uid,
    const bool mutation_enabled,
    const bool resource_handling_enabled,
    const double mutation_sigma_multiplier,
    StagedChildV1* staged,
    const uint64_t namespace_key = kRngv2NamespaceDefault
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
    const int64_t parent_slots[kHomologCount] = {
        canonical_parent_0_slot,
        canonical_parent_1_slot,
    };
    int32_t draw_index = 0;
    uint8_t selected[kHomologCount][kLinkageBlockCount];
    for (int32_t parent = 0; parent < kHomologCount; ++parent) {
        for (int32_t block = 0; block < kLinkageBlockCount; ++block) {
            uint32_t homolog = 0;
            if (
                !rngv2_bounded_v1(
                    2,
                    world_uid,
                    kRngv2OpMeiosis + static_cast<uint32_t>(draw_index),
                    subject_uid,
                    absolute_tick,
                    namespace_key,
                    &homolog
                )
            ) {
                return StatusV1::invalid_allele;
            }
            selected[parent][block] = static_cast<uint8_t>(homolog);
            staged->donor_homologs[draw_index] = static_cast<uint8_t>(homolog);
            ++draw_index;
        }
    }
    for (int32_t locus = 0; locus < kLocusCount; ++locus) {
        const int32_t block = linkage_block(locus);
        const int64_t parent_0_index =
            plane_index(planes, locus, parent_slots[0]);
        const int64_t parent_1_index =
            plane_index(planes, locus, parent_slots[1]);
        const double allele_0 = selected[0][block] == 0
            ? static_cast<double>(planes.allele_0[parent_0_index])
            : static_cast<double>(planes.allele_1[parent_0_index]);
        const double allele_1 = selected[1][block] == 0
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
    staged->semantic_draws = kMeiosisDrawCount;
    const double pre_mutation_rate = canon_double(
        0.5 * static_cast<double>(staged->allele_0[kMutationRate])
        + 0.5 * static_cast<double>(staged->allele_1[kMutationRate])
    );
    if (mutation_enabled) {
        for (int32_t locus = 0; locus < kLocusCount; ++locus) {
            double gate_draw = 0.0;
            uint32_t homolog = 0;
            double delta_draw = 0.0;
            if (
                !rngv2_uniform_v1(
                    world_uid,
                    kRngv2OpGate + static_cast<uint32_t>(locus),
                    subject_uid,
                    absolute_tick,
                    namespace_key,
                    &gate_draw
                )
                || !rngv2_bounded_v1(
                    2,
                    world_uid,
                    kRngv2OpHomolog + static_cast<uint32_t>(locus),
                    subject_uid,
                    absolute_tick,
                    namespace_key,
                    &homolog
                )
                || !rngv2_normal_v1(
                    world_uid,
                    kRngv2OpDelta + static_cast<uint32_t>(locus),
                    subject_uid,
                    absolute_tick,
                    namespace_key,
                    &delta_draw
                )
            ) {
                return StatusV1::invalid_allele;
            }
            staged->mutation_homologs[locus] = static_cast<uint8_t>(homolog);
            float* selected_allele = homolog == 0
                ? &staged->allele_0[locus]
                : &staged->allele_1[locus];
            const double before = static_cast<double>(*selected_allele);
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
                    ? delta_draw
                        * mutation_sigma(locus)
                        * mutation_sigma_multiplier
                    : 0.0;
            *selected_allele = canon_float(clamp_double(
                before + delta,
                lower_bound(locus),
                upper_bound(locus)
            ));
            staged->mutation_changed[locus] =
                static_cast<double>(*selected_allele) != before ? 1 : 0;
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

}  // namespace eiel::cuda_diploid_scalar_genome_v1

#undef EIEL_DIPLOID_FORCEINLINE
#undef EIEL_DIPLOID_HD
