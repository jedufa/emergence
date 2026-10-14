#include "cuda_architecture_b_jungle_primitives.cuh"

namespace jungle = eiel::cuda_architecture_b::jungle_primitives;

__global__ void jungle_primitive_compile_probe(float* const output) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    const jungle::PredatorRiskInput risk{
        0.5F,
        0.25F,
        3.0F,
        0.75F,
        0.25F,
        0.5F,
        1.0F,
        1.0F,
        9.0F,
        false,
        0.001F,
        0.02F,
        0.02F,
        0.02F,
        0.02F,
        0.02F,
        0.02F,
        0.0F,
        0.05F,
    };
    const jungle::PredatorDamageInput damage{
        0.25,
        3,
        10,
        1.5,
        true,
        true,
        0.25F,
        0.5F,
        0.25F,
        0.25F,
        0.25F,
        1.0F,
        0.0F,
    };
    const jungle::BirthEmissionInput birth{
        0.25F,
        0.5F,
        0.0F,
        0.0F,
        0.05,
        0.05,
        1.0,
        1.0,
        true,
    };

    const jungle::PredatorDamageResult damage_result =
        jungle::predator_damage_exact(damage);
    const jungle::BirthEmissionResult birth_result =
        jungle::birth_emission_exact(birth);
    output[0] = jungle::predator_risk_exact(risk);
    output[1] = damage_result.health_after;
    output[2] = damage_result.last_health_delta_after;
    output[3] = birth_result.scent_pending_after;
    output[4] = birth_result.waste_pending_after;
}
