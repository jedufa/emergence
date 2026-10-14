#include "cuda_architecture_b_jungle_primitives.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace jungle = eiel::cuda_architecture_b::jungle_primitives;

namespace {

void require(const bool condition, const char* const message) {
    if (!condition) {
        std::fprintf(stderr, "jungle primitive host probe failure: %s\n", message);
        std::exit(1);
    }
}

bool same_float(const float left, const float right) {
    return std::memcmp(&left, &right, sizeof(float)) == 0;
}

bool same_double(const double left, const double right) {
    return std::memcmp(&left, &right, sizeof(double)) == 0;
}

float f32_add(const float left, const float right) {
    volatile float result = left + right;
    return result;
}

float f32_sub(const float left, const float right) {
    volatile float result = left - right;
    return result;
}

float f32_mul(const float left, const float right) {
    volatile float result = left * right;
    return result;
}

float f32_div(const float left, const float right) {
    volatile float result = left / right;
    return result;
}

float clamp01(const float value) {
    if (value < 0.0F) {
        return 0.0F;
    }
    if (value > 1.0F) {
        return 1.0F;
    }
    return value;
}

float resident_predator_risk(const jungle::PredatorRiskInput& input) {
    const float scent_norm = input.scent_max > 0.0F
        ? clamp01(f32_div(input.scent, input.scent_max))
        : 0.0F;
    const float waste_norm = input.waste_max > 0.0F
        ? clamp01(f32_div(input.waste, input.waste_max))
        : 0.0F;
    const float density_norm = input.density_already_normalized
        ? clamp01(input.density)
        : clamp01(f32_div(input.density, input.density_denom));
    float risk = input.base_risk;
    risk = f32_add(risk, f32_mul(scent_norm, input.scent_weight));
    risk = f32_add(risk, f32_mul(waste_norm, input.waste_weight));
    risk = f32_add(risk, f32_mul(density_norm, input.density_weight));
    risk = f32_add(risk, f32_mul(clamp01(input.visibility), input.visibility_weight));
    risk = f32_sub(risk, f32_mul(clamp01(input.cover), input.cover_weight));
    risk = f32_sub(risk, f32_mul(clamp01(input.shelter), input.shelter_weight));
    float risk_min = input.risk_min;
    float risk_cap = input.risk_cap;
    if (risk_cap < risk_min) {
        std::swap(risk_min, risk_cap);
    }
    return std::max<float>(
        risk_min,
        std::min<float>(risk_cap, std::isfinite(risk) ? risk : 0.0F)
    );
}

jungle::PredatorDamageResult resident_predator_damage(
    const jungle::PredatorDamageInput& input
) {
    const bool juvenile = input.age <= input.juvenile_threshold;
    const double juvenile_scale = juvenile && input.juvenile_vulnerability
        ? input.juvenile_multiplier
        : 1.0;
    float terrain_scale = 1.0F;
    if (input.terrain_protection) {
        terrain_scale = f32_sub(
            f32_sub(1.0F, f32_mul(clamp01(input.cover), input.terrain_cover_weight)),
            f32_mul(clamp01(input.shelter), input.terrain_shelter_weight)
        );
        terrain_scale = std::max<float>(
            input.terrain_min,
            std::min<float>(1.0F, terrain_scale)
        );
    }
    const double raw_damage = input.base_damage * juvenile_scale;
    const double effective_damage = raw_damage * static_cast<double>(terrain_scale);
    const double prevented_damage = std::max<double>(0.0, raw_damage - effective_damage);
    const float health_after = static_cast<float>(std::max<double>(
        0.0,
        static_cast<double>(input.health_before) - effective_damage
    ));
    const float last_health_delta_after = static_cast<float>(
        static_cast<double>(input.last_health_delta_before) - effective_damage
    );
    return {
        raw_damage,
        effective_damage,
        prevented_damage,
        terrain_scale,
        health_after,
        last_health_delta_after,
        juvenile,
        health_after <= 0.0F,
    };
}

jungle::BirthEmissionResult resident_birth_emission(
    const jungle::BirthEmissionInput& input
) {
    double scent_emit = std::max<double>(0.0, input.scent_emit);
    double waste_emit = std::max<double>(0.0, input.waste_emit);
    if (input.clamp_enabled) {
        scent_emit = std::min<double>(
            scent_emit,
            std::max<double>(0.0, input.scent_max - static_cast<double>(input.scent))
        );
        waste_emit = std::min<double>(
            waste_emit,
            std::max<double>(0.0, input.waste_max - static_cast<double>(input.waste))
        );
    }
    float scent_pending = input.scent_pending;
    float waste_pending = input.waste_pending;
    if (scent_emit > 0.0) {
        scent_pending += static_cast<float>(scent_emit);
    }
    if (waste_emit > 0.0) {
        waste_pending += static_cast<float>(waste_emit);
    }
    return {
        scent_emit,
        waste_emit,
        scent_pending,
        waste_pending,
        scent_emit > 0.0 || waste_emit > 0.0,
    };
}

void require_same_damage(
    const jungle::PredatorDamageResult& actual,
    const jungle::PredatorDamageResult& expected
) {
    require(same_double(actual.raw_damage, expected.raw_damage), "raw damage differs");
    require(
        same_double(actual.effective_damage, expected.effective_damage),
        "effective damage differs"
    );
    require(
        same_double(actual.prevented_damage, expected.prevented_damage),
        "prevented damage differs"
    );
    require(
        same_float(actual.terrain_multiplier, expected.terrain_multiplier),
        "terrain multiplier differs"
    );
    require(same_float(actual.health_after, expected.health_after), "health differs");
    require(
        same_float(actual.last_health_delta_after, expected.last_health_delta_after),
        "health delta differs"
    );
    require(actual.juvenile == expected.juvenile, "juvenile classification differs");
    require(actual.lethal == expected.lethal, "lethal classification differs");
}

void require_same_birth(
    const jungle::BirthEmissionResult& actual,
    const jungle::BirthEmissionResult& expected
) {
    require(same_double(actual.scent_emitted, expected.scent_emitted), "scent emit differs");
    require(same_double(actual.waste_emitted, expected.waste_emitted), "waste emit differs");
    require(
        same_float(actual.scent_pending_after, expected.scent_pending_after),
        "pending scent differs"
    );
    require(
        same_float(actual.waste_pending_after, expected.waste_pending_after),
        "pending waste differs"
    );
    require(actual.emitted == expected.emitted, "emission classification differs");
}

}  // namespace

int main() {
    jungle::PredatorRiskInput risk{
        0.625F,
        0.25F,
        3.0F,
        0.8F,
        0.35F,
        0.15F,
        1.0F,
        0.5F,
        8.0F,
        false,
        0.003F,
        0.021F,
        0.019F,
        0.017F,
        0.023F,
        0.011F,
        0.013F,
        0.0F,
        0.05F,
    };
    require(
        same_float(jungle::predator_risk_exact(risk), resident_predator_risk(risk)),
        "mixed predator risk differs"
    );

    risk.density_already_normalized = true;
    risk.density = 1.75F;
    risk.scent_max = 0.0F;
    risk.waste_max = 0.0F;
    risk.risk_min = 0.04F;
    risk.risk_cap = 0.01F;
    require(
        same_float(jungle::predator_risk_exact(risk), resident_predator_risk(risk)),
        "normalized and swapped-bound predator risk differs"
    );

    risk.risk_min = 0.0125F;
    risk.risk_cap = 0.05F;
    risk.visibility = std::numeric_limits<float>::quiet_NaN();
    require(
        same_float(jungle::predator_risk_exact(risk), resident_predator_risk(risk)),
        "non-finite predator risk fallback differs"
    );
    require(
        same_float(jungle::predator_risk_exact(risk), risk.risk_min),
        "non-finite predator risk did not clamp to the minimum"
    );

    jungle::PredatorDamageInput damage{
        0.25,
        10,
        10,
        1.5,
        true,
        true,
        0.8F,
        0.6F,
        0.25F,
        0.25F,
        0.25F,
        0.2F,
        -0.1F,
    };
    require_same_damage(
        jungle::predator_damage_exact(damage),
        resident_predator_damage(damage)
    );
    require(jungle::predator_damage_exact(damage).lethal, "juvenile damage should be lethal");

    damage.age = 11;
    damage.terrain_protection = false;
    damage.health_before = 1.0F;
    require_same_damage(
        jungle::predator_damage_exact(damage),
        resident_predator_damage(damage)
    );
    require(!jungle::predator_damage_exact(damage).juvenile, "adult was classified juvenile");

    const jungle::BirthEmissionInput clamped_birth{
        0.98F,
        0.4F,
        0.125F,
        0.25F,
        0.05,
        0.7,
        1.0,
        1.0,
        true,
    };
    require_same_birth(
        jungle::birth_emission_exact(clamped_birth),
        resident_birth_emission(clamped_birth)
    );

    const jungle::BirthEmissionInput unclamped_birth{
        1.0F,
        1.0F,
        0.3F,
        0.4F,
        0.05,
        -0.1,
        1.0,
        1.0,
        false,
    };
    require_same_birth(
        jungle::birth_emission_exact(unclamped_birth),
        resident_birth_emission(unclamped_birth)
    );

    const jungle::BirthEmissionInput non_finite_birth{
        0.5F,
        0.5F,
        0.1F,
        0.2F,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        1.0,
        1.0,
        true,
    };
    require_same_birth(
        jungle::birth_emission_exact(non_finite_birth),
        resident_birth_emission(non_finite_birth)
    );
    require(!jungle::birth_emission_exact(non_finite_birth).emitted, "NaN emits were accepted");

    std::puts("jungle primitive host probe passed");
    return 0;
}
