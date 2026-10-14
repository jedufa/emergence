#pragma once

#include <climits>
#include <cstdint>

#if defined(__CUDACC__)
#define EIEL_GV2_HD __host__ __device__
#define EIEL_GV2_INLINE __forceinline__
#else
#define EIEL_GV2_HD
#define EIEL_GV2_INLINE inline
#endif

namespace eiel::genome_upgrade_v2_q10_v1 {

inline constexpr const char* kProfile = "genome_upgrade_v2_fixed_q10_v1";
inline constexpr const char* kSchemaVersion = "genome_upgrade_v2_bundle_v1";
inline constexpr const char* kContractSha256 =
    "5a78ac2371efa398b7700d843b52d71cc6683f049077b9825ddd0a5a6013f896";
inline constexpr int32_t kScale = 1024;
inline constexpr int32_t kHomologCount = 2;
inline constexpr int32_t kProductCount = 12;
inline constexpr int32_t kPhenotypeCount = 10;
inline constexpr int32_t kContextCount = 8;
inline constexpr int32_t kParameterCount = 4;
inline constexpr int32_t kLinkageBlockCount = 6;
inline constexpr int32_t kHereditaryLocusCount = 60;
inline constexpr int32_t kMeiosisSemanticDrawCount = 12;
inline constexpr int32_t kMutationSemanticDrawCount = 180;
inline constexpr int32_t kInheritanceSemanticDrawCount = 192;
inline constexpr int32_t kRuntimeFactorDeltaDivisor = 4;
inline constexpr int32_t kRuntimeFactorMinQ10 = 3 * kScale / 4;
inline constexpr int32_t kRuntimeFactorMaxQ10 = 5 * kScale / 4;
inline constexpr int32_t kRuntimeMemoryLadderCount = 5;
inline constexpr int32_t kRuntimeMemoryLadder[5] = {0, 8, 16, 32, 64};
inline constexpr int32_t kContextSenescentMaturityMultiplier = 4;
inline constexpr int32_t kRegulatorCount = 4;
inline constexpr int32_t kRegulatoryActiveNodeCostQ10 = 32;
inline constexpr int32_t kRegulatoryActiveEdgeCostQ10 = 8;
inline constexpr int32_t kRegulatorySignalCostQ10 = 64;
inline constexpr int32_t kRegulatoryBudgetQ10 = 640;
inline constexpr int32_t kC1GenotypeWidth =
    kHomologCount * kProductCount * (kParameterCount + 1);
inline constexpr int32_t kC1ProductWidth = kProductCount + 2;
inline constexpr int32_t kC1RegulatoryWidth =
    kRegulatorCount + kProductCount + 4;
inline constexpr int32_t kC1RuntimeWidth = 12;
inline constexpr int32_t kC1CostWidth = 2;

inline constexpr int32_t kProductContextIndex[kProductCount] = {
    0, 2, 0, 1, 1, 3, 5, 0, 6, 5, 6, 4,
};
inline constexpr int32_t kProductCostQ10[kProductCount] = {
    96, 128, 144, 128, 144, 96, 96, 160, 160, 112, 128, 112,
};
inline constexpr int32_t
    kPolygenicWeightsQ10[kPhenotypeCount][kProductCount] = {
        {640, 512, 0, 0, 192, 0, 0, 0, 0, 0, 0, 448},
        {-256, 0, 768, 0, -192, 0, 0, 0, 0, 128, 0, 192},
        {-128, 512, 0, 0, 704, 0, 0, 0, 0, -128, 0, 128},
        {704, -128, 0, 0, 0, 0, 0, 0, 0, -128, 0, 448},
        {-192, 0, 0, 768, 256, 0, 0, 0, 0, -128, 128, 0},
        {-128, 0, 0, 0, 0, 640, 512, 0, 0, 0, 128, 0},
        {-256, 0, 0, -128, 0, 0, 0, 704, 256, 0, 0, 192},
        {-192, 128, 0, 0, 0, 0, 0, 256, 768, 0, 0, 0},
        {-128, 0, 256, 0, 0, 0, 576, 0, 0, 640, -128, 0},
        {-192, 0, 0, 0, 0, 128, 0, 0, 0, -128, 768, 256},
    };

#if defined(__CUDACC__)
static __device__ __constant__ int32_t
    kDeviceProductContextIndex[kProductCount] = {
        0, 2, 0, 1, 1, 3, 5, 0, 6, 5, 6, 4,
    };
static __device__ __constant__ int32_t
    kDeviceProductCostQ10[kProductCount] = {
        96, 128, 144, 128, 144, 96, 96, 160, 160, 112, 128, 112,
    };
static __device__ __constant__ int32_t
    kDevicePolygenicWeightsQ10[kPhenotypeCount][kProductCount] = {
        {640, 512, 0, 0, 192, 0, 0, 0, 0, 0, 0, 448},
        {-256, 0, 768, 0, -192, 0, 0, 0, 0, 128, 0, 192},
        {-128, 512, 0, 0, 704, 0, 0, 0, 0, -128, 0, 128},
        {704, -128, 0, 0, 0, 0, 0, 0, 0, -128, 0, 448},
        {-192, 0, 0, 768, 256, 0, 0, 0, 0, -128, 128, 0},
        {-128, 0, 0, 0, 0, 640, 512, 0, 0, 0, 128, 0},
        {-256, 0, 0, -128, 0, 0, 0, 704, 256, 0, 0, 192},
        {-192, 128, 0, 0, 0, 0, 0, 256, 768, 0, 0, 0},
        {-128, 0, 256, 0, 0, 0, 576, 0, 0, 640, -128, 0},
        {-192, 0, 0, 0, 0, 128, 0, 0, 0, -128, 768, 256},
    };
#endif

struct GenotypeV1 {
    int64_t alleles[kHomologCount][kProductCount][kParameterCount];
    int64_t modifiers[kHomologCount][kProductCount];
};

struct ExpressionV1 {
    int64_t allele_products[kHomologCount][kProductCount];
    int64_t products[kProductCount];
    int64_t phenotypes[kPhenotypeCount];
    int64_t expression_cost;
    int64_t active_product_count;
    int64_t saturated_product_count;
};
static_assert(sizeof(ExpressionV1) == 49 * sizeof(int64_t));

struct RegulatoryDiagnosticsV1 {
    int64_t regulator_nodes[kRegulatorCount];
    int64_t epistatic_delta[kProductCount];
    int64_t regulatory_cost;
    int64_t active_regulator_count;
    int64_t active_regulatory_edge_count;
    int64_t developmental_stage_code;
};
static_assert(sizeof(RegulatoryDiagnosticsV1) == 20 * sizeof(int64_t));

EIEL_GV2_HD EIEL_GV2_INLINE int32_t c1_input_status_v1(
    const GenotypeV1& genotype,
    const int64_t context_q10[kContextCount]
) {
    for (int32_t homolog = 0; homolog < kHomologCount; ++homolog) {
        for (int32_t product = 0; product < kProductCount; ++product) {
            for (
                int32_t parameter = 0;
                parameter < kParameterCount;
                ++parameter
            ) {
                const int64_t value =
                    genotype.alleles[homolog][product][parameter];
                const int64_t lower =
                    parameter == 1 || parameter == 2 ? 0 : -kScale;
                if (value < lower || value > kScale) {
                    return 1;
                }
            }
            if (
                genotype.modifiers[homolog][product] < -kScale
                || genotype.modifiers[homolog][product] > kScale
            ) {
                return 1;
            }
        }
    }
    for (int32_t index = 0; index < kContextCount; ++index) {
        if (context_q10[index] < -kScale || context_q10[index] > kScale) {
            return 2;
        }
    }
    return context_q10[6] > 0 && context_q10[7] > 0 ? 3 : 0;
}

struct RuntimeProjectionV1 {
    int64_t body_size_factor_q10;
    int64_t movement_capacity_factor_q10;
    int64_t damage_received_factor_q10;
    int64_t metabolic_efficiency_factor_q10;
    int64_t recovery_factor_q10;
    int64_t sensory_fidelity_factor_q10;
    int64_t reproduction_threshold_factor_q10;
    int64_t offspring_energy_factor_q10;
    int64_t threat_response_q10;
    int64_t memory_allocation_q10;
    int64_t memory_dimension;
    int64_t expression_energy_debit_q10;
};

struct RuntimeContextInputsV1 {
    double energy;
    double max_energy;
    double injury;
    double start_health;
    double crowding;
    double resource_a;
    double resource_b;
    double hazard;
    int64_t age;
    int64_t maturity_age;
};

struct ResidentPlanesV1 {
    int64_t* alleles;
    int64_t* modifiers;
    int64_t* products;
    int64_t* phenotypes;
    int64_t* expression_cost;
    int64_t* active_product_count;
    int64_t* saturated_product_count;
    int64_t slot_count;
};

struct StagedChildV1 {
    GenotypeV1 genotype;
    int64_t semantic_draws;
    int64_t changed_loci;
};

EIEL_GV2_HD EIEL_GV2_INLINE int64_t clamp_i64(
    const int64_t value,
    const int64_t lower,
    const int64_t upper
) {
    return value < lower ? lower : (value > upper ? upper : value);
}

EIEL_GV2_HD EIEL_GV2_INLINE int64_t q10_product_2(
    const int64_t left,
    const int64_t right
) {
    return (left * right) / kScale;
}

EIEL_GV2_HD EIEL_GV2_INLINE int64_t q10_product_3(
    const int64_t first,
    const int64_t second,
    const int64_t third
) {
    return q10_product_2(q10_product_2(first, second), third);
}

EIEL_GV2_HD EIEL_GV2_INLINE int64_t allele_product(
    const int64_t parameters[kParameterCount]
) {
    const int64_t quantity_output =
        (parameters[0] * parameters[1]) / kScale;
    return (quantity_output * parameters[2]) / kScale;
}

EIEL_GV2_HD EIEL_GV2_INLINE int32_t product_context_index(
    const int32_t product
) {
#if defined(__CUDA_ARCH__)
    return kDeviceProductContextIndex[product];
#else
    return kProductContextIndex[product];
#endif
}

EIEL_GV2_HD EIEL_GV2_INLINE int32_t product_cost_q10(
    const int32_t product
) {
#if defined(__CUDA_ARCH__)
    return kDeviceProductCostQ10[product];
#else
    return kProductCostQ10[product];
#endif
}

EIEL_GV2_HD EIEL_GV2_INLINE int32_t polygenic_weight_q10(
    const int32_t phenotype,
    const int32_t product
) {
#if defined(__CUDA_ARCH__)
    return kDevicePolygenicWeightsQ10[phenotype][product];
#else
    return kPolygenicWeightsQ10[phenotype][product];
#endif
}

EIEL_GV2_HD EIEL_GV2_INLINE void neutral_genotype(GenotypeV1* genotype) {
    for (int32_t homolog = 0; homolog < kHomologCount; ++homolog) {
        for (int32_t product = 0; product < kProductCount; ++product) {
            genotype->alleles[homolog][product][0] = 0;
            genotype->alleles[homolog][product][1] = kScale;
            genotype->alleles[homolog][product][2] = kScale;
            genotype->alleles[homolog][product][3] = 0;
            genotype->modifiers[homolog][product] = 0;
        }
    }
}

EIEL_GV2_HD EIEL_GV2_INLINE void express_v1(
    const GenotypeV1& genotype,
    const int64_t context_q10[kContextCount],
    ExpressionV1* output,
    RegulatoryDiagnosticsV1* diagnostics = nullptr
) {
    output->expression_cost = 0;
    output->active_product_count = 0;
    output->saturated_product_count = 0;
    if (diagnostics != nullptr) {
        diagnostics->regulatory_cost = 0;
        diagnostics->active_regulator_count = 0;
        diagnostics->active_regulatory_edge_count = 0;
        diagnostics->developmental_stage_code =
            context_q10[6] > 0 && context_q10[7] > 0
            ? -1
            : (context_q10[6] > 0 ? 1 : (context_q10[7] > 0 ? 2 : 0));
        for (int32_t regulator = 0; regulator < kRegulatorCount; ++regulator) {
            diagnostics->regulator_nodes[regulator] = 0;
        }
        for (int32_t product = 0; product < kProductCount; ++product) {
            diagnostics->epistatic_delta[product] = 0;
        }
    }
    for (int32_t product = 0; product < kProductCount; ++product) {
        for (int32_t homolog = 0; homolog < kHomologCount; ++homolog) {
            output->allele_products[homolog][product] =
                allele_product(genotype.alleles[homolog][product]);
        }
        const int64_t dosage =
            output->allele_products[0][product]
            + output->allele_products[1][product];
        const int64_t sensitivity =
            (
                genotype.alleles[0][product][3]
                + genotype.alleles[1][product][3]
            ) / 2;
        const int64_t modifier =
            (
                genotype.modifiers[0][product]
                + genotype.modifiers[1][product]
            ) / 2;
        const int64_t context_effect =
            sensitivity * context_q10[product_context_index(product)] / kScale;
        const int64_t unsaturated =
            dosage + context_effect + modifier / 4;
        const int64_t value =
            clamp_i64(unsaturated, -kScale, kScale);
        output->products[product] = value;
        output->saturated_product_count += value != unsaturated ? 1 : 0;
    }
    int64_t expressed_modifiers[kProductCount]{};
    for (int32_t product = 0; product < kProductCount; ++product) {
        expressed_modifiers[product] =
            (
                genotype.modifiers[0][product]
                + genotype.modifiers[1][product]
            ) / 2;
    }
    const int64_t stability_factor = clamp_i64(
        kScale + expressed_modifiers[2] / 4,
        3 * kScale / 4,
        5 * kScale / 4
    );
    output->products[2] = clamp_i64(
        q10_product_2(output->products[2], stability_factor),
        -kScale,
        kScale
    );
    const int64_t saturation_limit =
        kScale
        - (
            expressed_modifiers[11] < 0
                ? -expressed_modifiers[11]
                : expressed_modifiers[11]
        ) / 8;
    output->products[7] = clamp_i64(
        output->products[7], -saturation_limit, saturation_limit
    );
    int64_t regulatory_cost = 0;
    int64_t active_regulatory_edges = 0;
    const int64_t epistatic_9_to_3 = q10_product_3(
        output->products[9], expressed_modifiers[9], 512
    );
    if (epistatic_9_to_3 != 0) {
        const int64_t delta = epistatic_9_to_3 < 0
            ? -epistatic_9_to_3
            : epistatic_9_to_3;
        if (diagnostics != nullptr) {
            diagnostics->epistatic_delta[3] = clamp_i64(
                diagnostics->epistatic_delta[3] + delta, -kScale, kScale
            );
        }
        output->products[3] = clamp_i64(
            output->products[3] + delta,
            -kScale,
            kScale
        );
        ++active_regulatory_edges;
    }
    const int64_t epistatic_0_to_4 = q10_product_3(
        output->products[0], expressed_modifiers[0], 256
    );
    if (epistatic_0_to_4 != 0) {
        const int64_t delta = epistatic_0_to_4 < 0
            ? epistatic_0_to_4
            : -epistatic_0_to_4;
        if (diagnostics != nullptr) {
            diagnostics->epistatic_delta[4] = clamp_i64(
                diagnostics->epistatic_delta[4] + delta, -kScale, kScale
            );
        }
        output->products[4] = clamp_i64(
            output->products[4] + delta,
            -kScale,
            kScale
        );
        ++active_regulatory_edges;
    }

    int64_t regulator_accumulators[kRegulatorCount]{};
    const int64_t regulator_inputs[kRegulatorCount][2] = {
        {q10_product_2(
             q10_product_2(context_q10[0], 512),
             kScale + expressed_modifiers[8] / 4
         ),
         q10_product_2(output->products[0], 256)},
        {q10_product_2(context_q10[1], 640),
         -(
             q10_product_2(output->products[4], 256) < 0
                 ? -q10_product_2(output->products[4], 256)
                 : q10_product_2(output->products[4], 256)
         )},
        {q10_product_2(
             q10_product_2(context_q10[5], 768),
             kScale + expressed_modifiers[6] / 4
         ),
         -(
             q10_product_2(context_q10[2], 256) < 0
                 ? -q10_product_2(context_q10[2], 256)
                 : q10_product_2(context_q10[2], 256)
         )},
        {q10_product_2(context_q10[6], 768),
         -(
             q10_product_2(
                 q10_product_2(context_q10[7], 768),
                 kScale + expressed_modifiers[10] / 4
             ) < 0
                 ? -q10_product_2(
                     q10_product_2(context_q10[7], 768),
                     kScale + expressed_modifiers[10] / 4
                 )
                 : q10_product_2(
                     q10_product_2(context_q10[7], 768),
                     kScale + expressed_modifiers[10] / 4
                 )
         )},
    };
    for (int32_t regulator = 0; regulator < kRegulatorCount; ++regulator) {
        for (int32_t input = 0; input < 2; ++input) {
            regulator_accumulators[regulator] +=
                regulator_inputs[regulator][input];
            active_regulatory_edges +=
                regulator_inputs[regulator][input] != 0 ? 1 : 0;
        }
    }
    const int64_t regulator_3_product_input =
        q10_product_2(output->products[10], 1024);
    regulator_accumulators[3] += regulator_3_product_input;
    active_regulatory_edges += regulator_3_product_input != 0 ? 1 : 0;
    int64_t regulators[kRegulatorCount]{};
    int64_t active_regulators = 0;
    const int32_t threshold_modifier_indices[kRegulatorCount] = {
        1, 3, 5, 7,
    };
    for (int32_t regulator = 0; regulator < kRegulatorCount; ++regulator) {
        const int64_t threshold = clamp_i64(
            128
                + expressed_modifiers[
                    threshold_modifier_indices[regulator]
                ] / 8,
            0,
            kScale / 4
        );
        regulators[regulator] = clamp_i64(
            regulator_accumulators[regulator] - threshold, 0, kScale
        );
        if (diagnostics != nullptr) {
            diagnostics->regulator_nodes[regulator] =
                regulators[regulator];
        }
        active_regulators += regulators[regulator] != 0 ? 1 : 0;
    }
    const int32_t output_targets[7] = {0, 4, 3, 1, 9, 10, 10};
    const int32_t output_regulators[7] = {0, 0, 1, 1, 2, 2, 3};
    const int32_t output_weights[7] = {256, 128, 384, 256, 512, 256, 512};
    const int32_t output_effects[7] = {1, -1, 1, -1, 1, -1, 2};
    for (int32_t edge = 0; edge < 7; ++edge) {
        const int64_t signal = regulators[output_regulators[edge]];
        if (signal == 0) {
            continue;
        }
        const int32_t target = output_targets[edge];
        const int64_t contribution =
            q10_product_2(signal, output_weights[edge]);
        int64_t delta = contribution;
        if (output_effects[edge] < 0) {
            delta = contribution < 0 ? contribution : -contribution;
        } else if (output_effects[edge] == 2) {
            delta = q10_product_2(output->products[target], contribution);
        }
        output->products[target] = clamp_i64(
            output->products[target] + delta, -kScale, kScale
        );
        ++active_regulatory_edges;
    }
    int64_t signal_cost = 0;
    for (int32_t regulator = 0; regulator < kRegulatorCount; ++regulator) {
        signal_cost +=
            (regulators[regulator] < 0
                ? -regulators[regulator]
                : regulators[regulator])
            * kRegulatorySignalCostQ10 / kScale;
    }
    regulatory_cost =
        active_regulators * kRegulatoryActiveNodeCostQ10
        + active_regulatory_edges * kRegulatoryActiveEdgeCostQ10
        + signal_cost;
    if (regulatory_cost > kRegulatoryBudgetQ10) {
        regulatory_cost = kRegulatoryBudgetQ10;
    }
    if (diagnostics != nullptr) {
        diagnostics->regulatory_cost = regulatory_cost;
        diagnostics->active_regulator_count = active_regulators;
        diagnostics->active_regulatory_edge_count =
            active_regulatory_edges;
    }
    output->expression_cost = regulatory_cost;
    output->active_product_count = 0;
    for (int32_t product = 0; product < kProductCount; ++product) {
        output->active_product_count +=
            output->products[product] != 0 ? 1 : 0;
        const int64_t magnitude = output->products[product] < 0
            ? -output->products[product]
            : output->products[product];
        output->expression_cost +=
            magnitude * product_cost_q10(product) / kScale;
    }
    for (int32_t phenotype = 0; phenotype < kPhenotypeCount; ++phenotype) {
        int64_t accumulator = 0;
        for (int32_t product = 0; product < kProductCount; ++product) {
            accumulator +=
                output->products[product]
                * polygenic_weight_q10(phenotype, product)
                / kScale;
        }
        output->phenotypes[phenotype] =
            clamp_i64(accumulator, -kScale, kScale);
    }
}

EIEL_GV2_HD EIEL_GV2_INLINE int64_t runtime_factor_q10(
    const int64_t phenotype_q10,
    const bool inverse
) {
    const int64_t delta = phenotype_q10 / kRuntimeFactorDeltaDivisor;
    return clamp_i64(
        inverse ? kScale - delta : kScale + delta,
        kRuntimeFactorMinQ10,
        kRuntimeFactorMaxQ10
    );
}

EIEL_GV2_HD EIEL_GV2_INLINE bool finite_value(const double value) {
#if defined(__CUDA_ARCH__)
    return isfinite(value);
#else
    return __builtin_isfinite(value);
#endif
}

EIEL_GV2_HD EIEL_GV2_INLINE bool normalized_context_q10(
    const double value,
    const double scale,
    int64_t* output
) {
    if (
        output == nullptr || !finite_value(value) || !finite_value(scale)
        || scale <= 0.0
    ) {
        return false;
    }
    *output = clamp_i64(
        static_cast<int64_t>(value * kScale / scale), 0, kScale
    );
    return true;
}

EIEL_GV2_HD EIEL_GV2_INLINE bool build_runtime_context_q10_v1(
    const RuntimeContextInputsV1& inputs,
    int64_t context_q10[kContextCount]
) {
    if (
        context_q10 == nullptr || inputs.age < 0 || inputs.maturity_age < 0
        || inputs.maturity_age > INT64_MAX
            / kContextSenescentMaturityMultiplier
        || !normalized_context_q10(
            inputs.max_energy - inputs.energy,
            inputs.max_energy,
            &context_q10[0]
        )
        || !normalized_context_q10(
            inputs.injury, inputs.start_health, &context_q10[1]
        )
        || !normalized_context_q10(inputs.crowding, 1.0, &context_q10[2])
        || !normalized_context_q10(inputs.resource_a, 1.0, &context_q10[3])
        || !normalized_context_q10(inputs.resource_b, 1.0, &context_q10[4])
        || !normalized_context_q10(inputs.hazard, 1.0, &context_q10[5])
    ) {
        return false;
    }
    context_q10[6] = inputs.maturity_age > 0
            && inputs.age < inputs.maturity_age
        ? kScale
        : 0;
    context_q10[7] = inputs.maturity_age > 0
            && inputs.age
                >= inputs.maturity_age
                    * kContextSenescentMaturityMultiplier
        ? kScale
        : 0;
    return true;
}

EIEL_GV2_HD EIEL_GV2_INLINE RuntimeProjectionV1 project_runtime_traits_v1(
    const ExpressionV1& expression
) {
    const int64_t memory_q10 = expression.phenotypes[9];
    const int64_t ladder_index = clamp_i64(
        (
            (memory_q10 + kScale)
            * (kRuntimeMemoryLadderCount - 1)
        ) / (2 * kScale),
        0,
        kRuntimeMemoryLadderCount - 1
    );
    const int64_t memory_dimension = ladder_index == 0
        ? 0
        : (ladder_index == 1
            ? 8
            : (ladder_index == 2 ? 16 : (ladder_index == 3 ? 32 : 64)));
    return {
        runtime_factor_q10(expression.phenotypes[0], false),
        runtime_factor_q10(expression.phenotypes[1], false),
        runtime_factor_q10(expression.phenotypes[2], true),
        runtime_factor_q10(expression.phenotypes[3], false),
        runtime_factor_q10(expression.phenotypes[4], false),
        runtime_factor_q10(expression.phenotypes[5], false),
        runtime_factor_q10(expression.phenotypes[6], true),
        runtime_factor_q10(expression.phenotypes[7], false),
        expression.phenotypes[8],
        memory_q10,
        memory_dimension,
        expression.expression_cost,
    };
}

EIEL_GV2_HD EIEL_GV2_INLINE void write_c1_dimensions_v1(
    const GenotypeV1& genotype,
    const int64_t context_q10[kContextCount],
    const ExpressionV1& expression,
    const RegulatoryDiagnosticsV1& diagnostics,
    const RuntimeProjectionV1& runtime,
    int64_t genotype_out[kC1GenotypeWidth],
    int64_t context_out[kContextCount],
    int64_t product_out[kC1ProductWidth],
    int64_t regulatory_out[kC1RegulatoryWidth],
    int64_t phenotype_out[kPhenotypeCount],
    int64_t runtime_out[kC1RuntimeWidth],
    int64_t cost_out[kC1CostWidth]
) {
    int32_t genotype_index = 0;
    for (int32_t homolog = 0; homolog < kHomologCount; ++homolog) {
        for (int32_t product = 0; product < kProductCount; ++product) {
            for (int32_t parameter = 0; parameter < kParameterCount; ++parameter) {
                genotype_out[genotype_index++] =
                    genotype.alleles[homolog][product][parameter];
            }
        }
    }
    for (int32_t homolog = 0; homolog < kHomologCount; ++homolog) {
        for (int32_t product = 0; product < kProductCount; ++product) {
            genotype_out[genotype_index++] =
                genotype.modifiers[homolog][product];
        }
    }
    for (int32_t index = 0; index < kContextCount; ++index) {
        context_out[index] = context_q10[index];
    }
    for (int32_t index = 0; index < kProductCount; ++index) {
        product_out[index] = expression.products[index];
    }
    product_out[kProductCount] = expression.active_product_count;
    product_out[kProductCount + 1] = expression.saturated_product_count;
    for (int32_t index = 0; index < kRegulatorCount; ++index) {
        regulatory_out[index] = diagnostics.regulator_nodes[index];
    }
    for (int32_t index = 0; index < kProductCount; ++index) {
        regulatory_out[kRegulatorCount + index] =
            diagnostics.epistatic_delta[index];
    }
    regulatory_out[kRegulatorCount + kProductCount] =
        diagnostics.regulatory_cost;
    regulatory_out[kRegulatorCount + kProductCount + 1] =
        diagnostics.active_regulator_count;
    regulatory_out[kRegulatorCount + kProductCount + 2] =
        diagnostics.active_regulatory_edge_count;
    regulatory_out[kRegulatorCount + kProductCount + 3] =
        diagnostics.developmental_stage_code;
    for (int32_t index = 0; index < kPhenotypeCount; ++index) {
        phenotype_out[index] = expression.phenotypes[index];
    }
    runtime_out[0] = runtime.body_size_factor_q10;
    runtime_out[1] = runtime.movement_capacity_factor_q10;
    runtime_out[2] = runtime.damage_received_factor_q10;
    runtime_out[3] = runtime.metabolic_efficiency_factor_q10;
    runtime_out[4] = runtime.recovery_factor_q10;
    runtime_out[5] = runtime.sensory_fidelity_factor_q10;
    runtime_out[6] = runtime.reproduction_threshold_factor_q10;
    runtime_out[7] = runtime.offspring_energy_factor_q10;
    runtime_out[8] = runtime.threat_response_q10;
    runtime_out[9] = runtime.memory_allocation_q10;
    runtime_out[10] = runtime.memory_dimension;
    runtime_out[11] = runtime.expression_energy_debit_q10;
    cost_out[0] = expression.expression_cost;
    cost_out[1] = diagnostics.regulatory_cost;
}

EIEL_GV2_HD EIEL_GV2_INLINE bool scale_offspring_transfer_v1(
    const double parental_transfer,
    const int64_t offspring_energy_factor_q10,
    double* const adjusted_transfer
) {
    if (
        adjusted_transfer == nullptr || !finite_value(parental_transfer)
        || parental_transfer < 0.0
        || offspring_energy_factor_q10 < kRuntimeFactorMinQ10
        || offspring_energy_factor_q10 > kRuntimeFactorMaxQ10
    ) {
        return false;
    }
    const double adjusted =
        parental_transfer
        * static_cast<double>(offspring_energy_factor_q10)
        / static_cast<double>(kScale);
    if (!finite_value(adjusted) || adjusted < 0.0) {
        return false;
    }
    *adjusted_transfer = adjusted;
    return true;
}

EIEL_GV2_HD EIEL_GV2_INLINE bool valid_runtime_factor_q10_v1(
    const int64_t value
) {
    return value >= kRuntimeFactorMinQ10
        && value <= kRuntimeFactorMaxQ10;
}

EIEL_GV2_HD EIEL_GV2_INLINE bool successful_movement_cost_v1(
    const double baseline_cost,
    const RuntimeProjectionV1& runtime,
    double* const adjusted_cost
) {
    if (
        adjusted_cost == nullptr || !finite_value(baseline_cost)
        || baseline_cost < 0.0
        || !valid_runtime_factor_q10_v1(runtime.body_size_factor_q10)
        || !valid_runtime_factor_q10_v1(
            runtime.metabolic_efficiency_factor_q10
        )
        || !valid_runtime_factor_q10_v1(
            runtime.movement_capacity_factor_q10
        )
    ) {
        return false;
    }
    const double adjusted =
        baseline_cost
        * static_cast<double>(runtime.body_size_factor_q10)
        / static_cast<double>(runtime.metabolic_efficiency_factor_q10)
        * static_cast<double>(kScale)
        / static_cast<double>(runtime.movement_capacity_factor_q10);
    if (!finite_value(adjusted) || adjusted < 0.0) {
        return false;
    }
    *adjusted_cost = adjusted;
    return true;
}

EIEL_GV2_HD EIEL_GV2_INLINE bool expression_energy_debit_v1(
    const RuntimeProjectionV1& runtime,
    double* const debit
) {
    if (
        debit == nullptr || runtime.expression_energy_debit_q10 < 0
        || !valid_runtime_factor_q10_v1(runtime.body_size_factor_q10)
        || !valid_runtime_factor_q10_v1(
            runtime.metabolic_efficiency_factor_q10
        )
    ) {
        return false;
    }
    const double adjusted =
        static_cast<double>(runtime.expression_energy_debit_q10)
        * static_cast<double>(runtime.body_size_factor_q10)
        / (
            static_cast<double>(runtime.metabolic_efficiency_factor_q10)
            * static_cast<double>(kScale)
        );
    if (!finite_value(adjusted) || adjusted < 0.0) {
        return false;
    }
    *debit = adjusted;
    return true;
}

EIEL_GV2_HD EIEL_GV2_INLINE bool reproduction_threshold_v1(
    const double hereditary_baseline,
    const RuntimeProjectionV1& runtime,
    double* const threshold
) {
    if (
        threshold == nullptr || !finite_value(hereditary_baseline)
        || hereditary_baseline < 0.0
        || !valid_runtime_factor_q10_v1(
            runtime.reproduction_threshold_factor_q10
        )
    ) {
        return false;
    }
    const double adjusted =
        hereditary_baseline
        * static_cast<double>(runtime.reproduction_threshold_factor_q10)
        / static_cast<double>(kScale);
    if (!finite_value(adjusted) || adjusted < 0.0) {
        return false;
    }
    *threshold = adjusted;
    return true;
}

EIEL_GV2_HD EIEL_GV2_INLINE bool health_after_delta_v1(
    const double before,
    const double unscaled_after,
    const double maximum,
    const RuntimeProjectionV1& runtime,
    double* const adjusted_after
) {
    if (
        adjusted_after == nullptr || !finite_value(before)
        || !finite_value(unscaled_after) || !finite_value(maximum)
        || maximum <= 0.0
    ) {
        return false;
    }
    const double delta = unscaled_after - before;
    const int64_t factor = delta < 0.0
        ? runtime.damage_received_factor_q10
        : runtime.recovery_factor_q10;
    if (!valid_runtime_factor_q10_v1(factor)) {
        return false;
    }
    double adjusted =
        before + delta * static_cast<double>(factor) / kScale;
    adjusted = adjusted < 0.0 ? 0.0 : (adjusted > maximum ? maximum : adjusted);
    if (!finite_value(adjusted)) {
        return false;
    }
    *adjusted_after = adjusted;
    return true;
}

EIEL_GV2_HD EIEL_GV2_INLINE bool valid_planes(
    const ResidentPlanesV1& planes
) {
    return planes.alleles != nullptr
        && planes.modifiers != nullptr
        && planes.products != nullptr
        && planes.phenotypes != nullptr
        && planes.expression_cost != nullptr
        && planes.active_product_count != nullptr
        && planes.saturated_product_count != nullptr
        && planes.slot_count > 0;
}

EIEL_GV2_HD EIEL_GV2_INLINE int64_t allele_index(
    const ResidentPlanesV1& planes,
    const int32_t homolog,
    const int32_t product,
    const int32_t parameter,
    const int64_t slot
) {
    return (
        (
            (
                static_cast<int64_t>(homolog) * kProductCount
                + product
            ) * kParameterCount
            + parameter
        ) * planes.slot_count
        + slot
    );
}

EIEL_GV2_HD EIEL_GV2_INLINE int64_t modifier_index(
    const ResidentPlanesV1& planes,
    const int32_t homolog,
    const int32_t product,
    const int64_t slot
) {
    return (
        (
            static_cast<int64_t>(homolog) * kProductCount
            + product
        ) * planes.slot_count
        + slot
    );
}

EIEL_GV2_HD EIEL_GV2_INLINE void load_genotype_v1(
    const ResidentPlanesV1& planes,
    const int64_t slot,
    GenotypeV1* genotype
) {
    for (int32_t homolog = 0; homolog < kHomologCount; ++homolog) {
        for (int32_t product = 0; product < kProductCount; ++product) {
            for (int32_t parameter = 0; parameter < kParameterCount; ++parameter) {
                genotype->alleles[homolog][product][parameter] =
                    planes.alleles[
                        allele_index(
                            planes, homolog, product, parameter, slot
                        )
                    ];
            }
            genotype->modifiers[homolog][product] =
                planes.modifiers[
                    modifier_index(planes, homolog, product, slot)
                ];
        }
    }
}

EIEL_GV2_HD EIEL_GV2_INLINE bool express_resident_slot_v1(
    const ResidentPlanesV1& planes,
    const int64_t slot,
    const int64_t context_q10[kContextCount]
) {
    if (
        !valid_planes(planes)
        || slot < 0
        || slot >= planes.slot_count
        || context_q10 == nullptr
    ) {
        return false;
    }
    GenotypeV1 genotype{};
    ExpressionV1 expression{};
    load_genotype_v1(planes, slot, &genotype);
    express_v1(genotype, context_q10, &expression);
    for (int32_t product = 0; product < kProductCount; ++product) {
        planes.products[
            static_cast<int64_t>(product) * planes.slot_count + slot
        ] = expression.products[product];
    }
    for (int32_t phenotype = 0; phenotype < kPhenotypeCount; ++phenotype) {
        planes.phenotypes[
            static_cast<int64_t>(phenotype) * planes.slot_count + slot
        ] = expression.phenotypes[phenotype];
    }
    planes.expression_cost[slot] = expression.expression_cost;
    planes.active_product_count[slot] = expression.active_product_count;
    planes.saturated_product_count[slot] =
        expression.saturated_product_count;
    return true;
}

EIEL_GV2_HD EIEL_GV2_INLINE RuntimeProjectionV1
load_runtime_projection_v1(
    const ResidentPlanesV1& planes,
    const int64_t slot
) {
    ExpressionV1 expression{};
    for (int32_t phenotype = 0; phenotype < kPhenotypeCount; ++phenotype) {
        expression.phenotypes[phenotype] = planes.phenotypes[
            static_cast<int64_t>(phenotype) * planes.slot_count + slot
        ];
    }
    expression.expression_cost = planes.expression_cost[slot];
    expression.active_product_count = planes.active_product_count[slot];
    expression.saturated_product_count =
        planes.saturated_product_count[slot];
    return project_runtime_traits_v1(expression);
}

EIEL_GV2_HD EIEL_GV2_INLINE int64_t parameter_lower_bound(
    const int32_t parameter
) {
    return parameter == 1 || parameter == 2 ? 0 : -kScale;
}

EIEL_GV2_HD EIEL_GV2_INLINE int64_t parameter_upper_bound(
    const int32_t
) {
    return kScale;
}

template <typename RandomV1>
EIEL_GV2_HD EIEL_GV2_INLINE bool stage_child_v1(
    const ResidentPlanesV1& planes,
    const int64_t canonical_parent_0_slot,
    const int64_t canonical_parent_1_slot,
    RandomV1& random,
    const int64_t mutation_rate_q10,
    const int64_t mutation_step_q10,
    StagedChildV1* staged
) {
    if (
        !valid_planes(planes) || staged == nullptr
        || canonical_parent_0_slot < 0
        || canonical_parent_0_slot >= planes.slot_count
        || canonical_parent_1_slot < 0
        || canonical_parent_1_slot >= planes.slot_count
        || mutation_rate_q10 < 0 || mutation_rate_q10 > kScale
        || mutation_step_q10 < 0 || mutation_step_q10 > kScale
    ) {
        return false;
    }
    GenotypeV1 parents[kHomologCount];
    load_genotype_v1(planes, canonical_parent_0_slot, &parents[0]);
    load_genotype_v1(planes, canonical_parent_1_slot, &parents[1]);
    uint8_t donors[kMeiosisSemanticDrawCount];
    for (int32_t draw = 0; draw < kMeiosisSemanticDrawCount; ++draw) {
        donors[draw] = static_cast<uint8_t>(random.randbelow(2));
    }
    for (int32_t child_homolog = 0; child_homolog < kHomologCount; ++child_homolog) {
        const int32_t donor_offset = child_homolog * kLinkageBlockCount;
        for (int32_t product = 0; product < kProductCount; ++product) {
            const int32_t donor = donors[donor_offset + product / 2];
            for (int32_t parameter = 0; parameter < kParameterCount; ++parameter) {
                staged->genotype.alleles[child_homolog][product][parameter] =
                    parents[child_homolog].alleles[donor][product][parameter];
            }
            staged->genotype.modifiers[child_homolog][product] =
                parents[child_homolog].modifiers[donor][product];
        }
    }
    staged->changed_loci = 0;
    for (int32_t product = 0; product < kProductCount; ++product) {
        for (int32_t parameter = 0; parameter < kParameterCount; ++parameter) {
            const int32_t homolog = static_cast<int32_t>(random.randbelow(2));
            const bool enabled =
                random.random_double()
                < static_cast<double>(mutation_rate_q10)
                    / static_cast<double>(kScale);
            const uint64_t delta_width =
                static_cast<uint64_t>(mutation_step_q10 * 2 + 1);
            const int64_t delta =
                static_cast<int64_t>(random.randbelow(delta_width))
                - mutation_step_q10;
            if (enabled) {
                int64_t* const value =
                    &staged->genotype.alleles[homolog][product][parameter];
                const int64_t before = *value;
                *value = clamp_i64(
                    before + delta,
                    parameter_lower_bound(parameter),
                    parameter_upper_bound(parameter)
                );
                staged->changed_loci += *value != before ? 1 : 0;
            }
        }
        const int32_t homolog = static_cast<int32_t>(random.randbelow(2));
        const bool enabled =
            random.random_double()
            < static_cast<double>(mutation_rate_q10)
                / static_cast<double>(kScale);
        const uint64_t delta_width =
            static_cast<uint64_t>(mutation_step_q10 * 2 + 1);
        const int64_t delta =
            static_cast<int64_t>(random.randbelow(delta_width))
            - mutation_step_q10;
        if (enabled) {
            int64_t* const value =
                &staged->genotype.modifiers[homolog][product];
            const int64_t before = *value;
            *value = clamp_i64(before + delta, -kScale, kScale);
            staged->changed_loci += *value != before ? 1 : 0;
        }
    }
    staged->semantic_draws = kInheritanceSemanticDrawCount;
    return true;
}

EIEL_GV2_HD EIEL_GV2_INLINE bool commit_staged_child_v1(
    const ResidentPlanesV1& planes,
    const int64_t slot,
    const int64_t context_q10[kContextCount],
    const StagedChildV1& staged
) {
    if (!valid_planes(planes) || slot < 0 || slot >= planes.slot_count) {
        return false;
    }
    for (int32_t homolog = 0; homolog < kHomologCount; ++homolog) {
        for (int32_t product = 0; product < kProductCount; ++product) {
            for (int32_t parameter = 0; parameter < kParameterCount; ++parameter) {
                planes.alleles[
                    allele_index(
                        planes, homolog, product, parameter, slot
                    )
                ] = staged.genotype.alleles[homolog][product][parameter];
            }
            planes.modifiers[
                modifier_index(planes, homolog, product, slot)
            ] = staged.genotype.modifiers[homolog][product];
        }
    }
    return express_resident_slot_v1(planes, slot, context_q10);
}

}  // namespace eiel::genome_upgrade_v2_q10_v1

#undef EIEL_GV2_HD
#undef EIEL_GV2_INLINE
