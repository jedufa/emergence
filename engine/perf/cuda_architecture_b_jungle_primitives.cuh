#ifndef EIEL_CUDA_ARCHITECTURE_B_JUNGLE_PRIMITIVES_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_JUNGLE_PRIMITIVES_CUH_

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define EIEL_JUNGLE_HOST_DEVICE __host__ __device__
#define EIEL_JUNGLE_FORCEINLINE __forceinline__
#else
#define EIEL_JUNGLE_HOST_DEVICE
#define EIEL_JUNGLE_FORCEINLINE inline
#endif

namespace eiel::cuda_architecture_b::jungle_primitives {

struct PredatorRiskInput {
    float scent;
    float waste;
    float density;
    float visibility;
    float cover;
    float shelter;
    float scent_max;
    float waste_max;
    float density_denom;
    bool density_already_normalized;
    float base_risk;
    float scent_weight;
    float waste_weight;
    float density_weight;
    float visibility_weight;
    float cover_weight;
    float shelter_weight;
    float risk_min;
    float risk_cap;
};

struct PredatorDamageInput {
    double base_damage;
    std::int32_t age;
    std::int32_t juvenile_threshold;
    double juvenile_multiplier;
    bool juvenile_vulnerability;
    bool terrain_protection;
    float cover;
    float shelter;
    float terrain_cover_weight;
    float terrain_shelter_weight;
    float terrain_min;
    float health_before;
    float last_health_delta_before;
};

struct PredatorDamageResult {
    double raw_damage;
    double effective_damage;
    double prevented_damage;
    float terrain_multiplier;
    float health_after;
    float last_health_delta_after;
    bool juvenile;
    bool lethal;
};

struct BirthEmissionInput {
    float scent;
    float waste;
    float scent_pending;
    float waste_pending;
    double scent_emit;
    double waste_emit;
    double scent_max;
    double waste_max;
    bool clamp_enabled;
};

struct BirthEmissionResult {
    double scent_emitted;
    double waste_emitted;
    float scent_pending_after;
    float waste_pending_after;
    bool emitted;
};

namespace detail {

EIEL_JUNGLE_HOST_DEVICE EIEL_JUNGLE_FORCEINLINE float f32_add(
    const float left,
    const float right
) {
#if defined(__CUDA_ARCH__)
    return __fadd_rn(left, right);
#else
    volatile float result = left + right;
    return result;
#endif
}

EIEL_JUNGLE_HOST_DEVICE EIEL_JUNGLE_FORCEINLINE float f32_sub(
    const float left,
    const float right
) {
#if defined(__CUDA_ARCH__)
    return __fsub_rn(left, right);
#else
    volatile float result = left - right;
    return result;
#endif
}

EIEL_JUNGLE_HOST_DEVICE EIEL_JUNGLE_FORCEINLINE float f32_mul(
    const float left,
    const float right
) {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(left, right);
#else
    volatile float result = left * right;
    return result;
#endif
}

EIEL_JUNGLE_HOST_DEVICE EIEL_JUNGLE_FORCEINLINE float f32_div(
    const float left,
    const float right
) {
#if defined(__CUDA_ARCH__)
    return __fdiv_rn(left, right);
#else
    volatile float result = left / right;
    return result;
#endif
}

EIEL_JUNGLE_HOST_DEVICE EIEL_JUNGLE_FORCEINLINE double f64_sub(
    const double left,
    const double right
) {
#if defined(__CUDA_ARCH__)
    return __dsub_rn(left, right);
#else
    volatile double result = left - right;
    return result;
#endif
}

EIEL_JUNGLE_HOST_DEVICE EIEL_JUNGLE_FORCEINLINE double f64_mul(
    const double left,
    const double right
) {
#if defined(__CUDA_ARCH__)
    return __dmul_rn(left, right);
#else
    volatile double result = left * right;
    return result;
#endif
}

template <typename T>
EIEL_JUNGLE_HOST_DEVICE EIEL_JUNGLE_FORCEINLINE T minimum_exact(
    const T left,
    const T right
) {
    return right < left ? right : left;
}

template <typename T>
EIEL_JUNGLE_HOST_DEVICE EIEL_JUNGLE_FORCEINLINE T maximum_exact(
    const T left,
    const T right
) {
    return left < right ? right : left;
}

EIEL_JUNGLE_HOST_DEVICE EIEL_JUNGLE_FORCEINLINE bool finite_float(
    const float value
) {
#if defined(__CUDA_ARCH__)
    return isfinite(value);
#else
    return std::isfinite(value);
#endif
}

EIEL_JUNGLE_HOST_DEVICE EIEL_JUNGLE_FORCEINLINE float clamp01(
    const float value
) {
    if (value < 0.0F) {
        return 0.0F;
    }
    if (value > 1.0F) {
        return 1.0F;
    }
    return value;
}

}  // namespace detail

// Inputs are the metadata-resolved values used by the resident CPU owner.
EIEL_JUNGLE_HOST_DEVICE EIEL_JUNGLE_FORCEINLINE float predator_risk_exact(
    const PredatorRiskInput& input
) {
    const float scent_norm = input.scent_max > 0.0F
        ? detail::clamp01(detail::f32_div(input.scent, input.scent_max))
        : 0.0F;
    const float waste_norm = input.waste_max > 0.0F
        ? detail::clamp01(detail::f32_div(input.waste, input.waste_max))
        : 0.0F;
    const float density_norm = input.density_already_normalized
        ? detail::clamp01(input.density)
        : detail::clamp01(detail::f32_div(input.density, input.density_denom));

    float risk = input.base_risk;
    risk = detail::f32_add(risk, detail::f32_mul(scent_norm, input.scent_weight));
    risk = detail::f32_add(risk, detail::f32_mul(waste_norm, input.waste_weight));
    risk = detail::f32_add(risk, detail::f32_mul(density_norm, input.density_weight));
    risk = detail::f32_add(
        risk,
        detail::f32_mul(detail::clamp01(input.visibility), input.visibility_weight)
    );
    risk = detail::f32_sub(
        risk,
        detail::f32_mul(detail::clamp01(input.cover), input.cover_weight)
    );
    risk = detail::f32_sub(
        risk,
        detail::f32_mul(detail::clamp01(input.shelter), input.shelter_weight)
    );

    float risk_min = input.risk_min;
    float risk_cap = input.risk_cap;
    if (risk_cap < risk_min) {
        const float previous_min = risk_min;
        risk_min = risk_cap;
        risk_cap = previous_min;
    }
    const float finite_risk = detail::finite_float(risk) ? risk : 0.0F;
    return detail::maximum_exact(
        risk_min,
        detail::minimum_exact(risk_cap, finite_risk)
    );
}

EIEL_JUNGLE_HOST_DEVICE EIEL_JUNGLE_FORCEINLINE PredatorDamageResult
predator_damage_exact(const PredatorDamageInput& input) {
    const bool juvenile = input.age <= input.juvenile_threshold;
    const double juvenile_scale = juvenile && input.juvenile_vulnerability
        ? input.juvenile_multiplier
        : 1.0;
    float terrain_multiplier = 1.0F;
    if (input.terrain_protection) {
        terrain_multiplier = detail::f32_sub(
            detail::f32_sub(
                1.0F,
                detail::f32_mul(
                    detail::clamp01(input.cover),
                    input.terrain_cover_weight
                )
            ),
            detail::f32_mul(
                detail::clamp01(input.shelter),
                input.terrain_shelter_weight
            )
        );
        terrain_multiplier = detail::maximum_exact(
            input.terrain_min,
            detail::minimum_exact(1.0F, terrain_multiplier)
        );
    }

    const double raw_damage = detail::f64_mul(input.base_damage, juvenile_scale);
    const double effective_damage = detail::f64_mul(
        raw_damage,
        static_cast<double>(terrain_multiplier)
    );
    const double prevented_damage = detail::maximum_exact(
        0.0,
        detail::f64_sub(raw_damage, effective_damage)
    );
    const float health_after = static_cast<float>(detail::maximum_exact(
        0.0,
        detail::f64_sub(static_cast<double>(input.health_before), effective_damage)
    ));
    const float last_health_delta_after = static_cast<float>(detail::f64_sub(
        static_cast<double>(input.last_health_delta_before),
        effective_damage
    ));
    return {
        raw_damage,
        effective_damage,
        prevented_damage,
        terrain_multiplier,
        health_after,
        last_health_delta_after,
        juvenile,
        health_after <= 0.0F,
    };
}

EIEL_JUNGLE_HOST_DEVICE EIEL_JUNGLE_FORCEINLINE BirthEmissionResult
birth_emission_exact(const BirthEmissionInput& input) {
    double scent_emit = detail::maximum_exact(0.0, input.scent_emit);
    double waste_emit = detail::maximum_exact(0.0, input.waste_emit);
    if (input.clamp_enabled) {
        scent_emit = detail::minimum_exact(
            scent_emit,
            detail::maximum_exact(
                0.0,
                detail::f64_sub(input.scent_max, static_cast<double>(input.scent))
            )
        );
        waste_emit = detail::minimum_exact(
            waste_emit,
            detail::maximum_exact(
                0.0,
                detail::f64_sub(input.waste_max, static_cast<double>(input.waste))
            )
        );
    }

    float scent_pending_after = input.scent_pending;
    float waste_pending_after = input.waste_pending;
    if (scent_emit > 0.0) {
        scent_pending_after = detail::f32_add(
            scent_pending_after,
            static_cast<float>(scent_emit)
        );
    }
    if (waste_emit > 0.0) {
        waste_pending_after = detail::f32_add(
            waste_pending_after,
            static_cast<float>(waste_emit)
        );
    }
    return {
        scent_emit,
        waste_emit,
        scent_pending_after,
        waste_pending_after,
        scent_emit > 0.0 || waste_emit > 0.0,
    };
}

}  // namespace eiel::cuda_architecture_b::jungle_primitives

#undef EIEL_JUNGLE_HOST_DEVICE
#undef EIEL_JUNGLE_FORCEINLINE

#endif  // EIEL_CUDA_ARCHITECTURE_B_JUNGLE_PRIMITIVES_CUH_
