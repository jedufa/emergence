#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define EIEL_GMNW_HOST_DEVICE __host__ __device__
#else
#define EIEL_GMNW_HOST_DEVICE
#endif

EIEL_GMNW_HOST_DEVICE inline float eiel_gmnw_exact_expf_v1(
    const float value
) noexcept {
    return static_cast<float>(::exp(static_cast<double>(value)));
}

EIEL_GMNW_HOST_DEVICE inline float eiel_gmnw_exact_tanhf_v1(
    const float value
) noexcept {
    return static_cast<float>(::tanh(static_cast<double>(value)));
}

#if defined(EIEL_GMNW_RECURRENT_EXACT_MATH_REMAP_V1)
#define expf eiel_gmnw_exact_expf_v1
#define tanhf eiel_gmnw_exact_tanhf_v1
#endif

#undef EIEL_GMNW_HOST_DEVICE
