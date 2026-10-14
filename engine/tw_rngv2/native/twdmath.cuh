#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define TWRNGV2_MATH_HD __host__ __device__
#else
#define TWRNGV2_MATH_HD
#endif

namespace tw_rngv2 {

TWRNGV2_MATH_HD inline uint64_t f64_bits(double value) {
#if defined(__CUDA_ARCH__)
    return static_cast<uint64_t>(__double_as_longlong(value));
#else
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
#endif
}

TWRNGV2_MATH_HD inline double f64_from_bits(uint64_t bits) {
#if defined(__CUDA_ARCH__)
    return __longlong_as_double(static_cast<long long>(bits));
#else
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}

TWRNGV2_MATH_HD inline uint32_t hi32(double value) {
    return static_cast<uint32_t>(f64_bits(value) >> 32);
}

TWRNGV2_MATH_HD inline uint32_t lo32(double value) {
    return static_cast<uint32_t>(f64_bits(value));
}

TWRNGV2_MATH_HD inline double set_hi32(double value, uint32_t hi) {
    const uint64_t bits = (f64_bits(value) & 0xFFFFFFFFull) | (static_cast<uint64_t>(hi) << 32);
    return f64_from_bits(bits);
}

TWRNGV2_MATH_HD inline double from_words(uint32_t hi, uint32_t lo) {
    return f64_from_bits((static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo));
}

TWRNGV2_MATH_HD inline double twdmath_sqrt(double x) {
#if defined(__CUDA_ARCH__)
    return sqrt(x);
#else
    return std::sqrt(x);
#endif
}

TWRNGV2_MATH_HD inline double twdmath_log(double x) {
    const double ln2_hi = from_words(0x3FE62E42u, 0xFEE00000u);
    const double ln2_lo = from_words(0x3DEA39EFu, 0x35793C76u);
    const double two54 = from_words(0x43500000u, 0x00000000u);
    const double lg1 = from_words(0x3FE55555u, 0x55555593u);
    const double lg2 = from_words(0x3FD99999u, 0x9997FA04u);
    const double lg3 = from_words(0x3FD24924u, 0x94229359u);
    const double lg4 = from_words(0x3FCC71C5u, 0x1D8E78AFu);
    const double lg5 = from_words(0x3FC74664u, 0x96CB03DEu);
    const double lg6 = from_words(0x3FC39A09u, 0xD078C69Fu);
    const double lg7 = from_words(0x3FC2F112u, 0xDF3E5244u);

    uint32_t hx = hi32(x);
    uint32_t lx = lo32(x);
    int k = 0;
    if (static_cast<int32_t>(hx) < 0x00100000) {
        if (((hx & 0x7FFFFFFFu) | lx) == 0u) {
            return -1.0 / 0.0;
        }
        if (static_cast<int32_t>(hx) < 0) {
            return 0.0 / 0.0;
        }
        k -= 54;
        x *= two54;
        hx = hi32(x);
        lx = lo32(x);
    }
    if (hx >= 0x7FF00000u) {
        return x + x;
    }
    k += static_cast<int>(hx >> 20) - 1023;
    hx &= 0x000FFFFFu;
    const uint32_t i = (hx + 0x95F64u) & 0x100000u;
    x = set_hi32(x, hx | (i ^ 0x3FF00000u));
    k += static_cast<int>(i >> 20);
    const double f = x - 1.0;
    if ((0x000FFFFFu & (2u + hx)) < 3u) {
        if (f == 0.0) {
            if (k == 0) {
                return 0.0;
            }
            const double dk = static_cast<double>(k);
            return dk * ln2_hi + dk * ln2_lo;
        }
        const double r = f * f * (0.5 - 0.33333333333333333 * f);
        if (k == 0) {
            return f - r;
        }
        const double dk = static_cast<double>(k);
        return dk * ln2_hi - ((r - dk * ln2_lo) - f);
    }
    const double s = f / (2.0 + f);
    const double z = s * s;
    const double w = z * z;
    const double t1 = w * (lg2 + w * (lg4 + w * lg6));
    const double t2 = z * (lg1 + w * (lg3 + w * (lg5 + w * lg7)));
    const double r = t2 + t1;
    uint32_t i2 = hx - 0x6147Au;
    const uint32_t j = 0x6B851u - hx;
    i2 |= j;
    if (static_cast<int32_t>(i2) > 0) {
        const double hfsq = 0.5 * f * f;
        if (k == 0) {
            return f - (hfsq - s * (hfsq + r));
        }
        const double dk = static_cast<double>(k);
        return dk * ln2_hi - ((hfsq - (s * (hfsq + r) + dk * ln2_lo)) - f);
    }
    if (k == 0) {
        return f - s * (f - r);
    }
    const double dk = static_cast<double>(k);
    return dk * ln2_hi - ((s * (f - r) - dk * ln2_lo) - f);
}

TWRNGV2_MATH_HD inline double kernel_sin(double x, double y, int iy) {
    const double s1 = from_words(0xBFC55555u, 0x55555549u);
    const double s2 = from_words(0x3F811111u, 0x1110F8A6u);
    const double s3 = from_words(0xBF2A01A0u, 0x19C161D5u);
    const double s4 = from_words(0x3EC71DE3u, 0x57B1FE7Du);
    const double s5 = from_words(0xBE5AE5E6u, 0x8A2B9CEBu);
    const double s6 = from_words(0x3DE5D93Au, 0x5ACFD57Cu);
    const double z = x * x;
    const double v = z * x;
    const double r = s2 + z * (s3 + z * (s4 + z * (s5 + z * s6)));
    if (iy == 0) {
        return x + v * (s1 + z * r);
    }
    return x - ((z * (0.5 * y - v * r) - y) - v * s1);
}

TWRNGV2_MATH_HD inline double kernel_cos(double x, double y) {
    const double c1 = from_words(0x3FA55555u, 0x5555554Cu);
    const double c2 = from_words(0xBF56C16Cu, 0x16C15177u);
    const double c3 = from_words(0x3EFA01A0u, 0x19CB1590u);
    const double c4 = from_words(0xBE927E4Fu, 0x809C52ADu);
    const double c5 = from_words(0x3E21EE9Eu, 0xBDB4B1C4u);
    const double c6 = from_words(0xBDA8FAE9u, 0xBE8838D4u);
    const double z = x * x;
    const double zz = z * z;
    const double r = z * (c1 + z * (c2 + z * c3)) + zz * zz * (c4 + z * (c5 + z * c6));
    const double hz = 0.5 * z;
    const double w = 1.0 - hz;
    return w + (((1.0 - w) - hz) + (z * r - x * y));
}

TWRNGV2_MATH_HD inline void rem_pio2(double x, int* n_out, double* y0, double* y1) {
    const double invpio2 = from_words(0x3FE45F30u, 0x6DC9C883u);
    const double pio2_1 = from_words(0x3FF921FBu, 0x54400000u);
    const double pio2_1t = from_words(0x3DD0B461u, 0x1A626331u);
    const double pio2_2 = from_words(0x3DD0B461u, 0x1A600000u);
    const double pio2_2t = from_words(0x3BA3198Au, 0x2E037073u);
    const double pio2_3 = from_words(0x3BA3198Au, 0x2E000000u);
    const double pio2_3t = from_words(0x397B839Au, 0x252049C1u);
    const uint32_t hx = hi32(x);
    const uint32_t ix = hx & 0x7FFFFFFFu;
    const double sign = (static_cast<int32_t>(hx) < 0) ? -1.0 : 1.0;
    const double ax = (x < 0.0) ? -x : x;
    if (ix <= 0x3FE921FBu) {
        *n_out = 0;
        *y0 = x;
        *y1 = 0.0;
        return;
    }
    double fn = ax * invpio2 + 0.5;
    int n = static_cast<int>(fn);
    fn = static_cast<double>(n);
    double r = ax - fn * pio2_1;
    double w = fn * pio2_1t;
    double yy0 = r - w;
    double yy1 = (r - yy0) - w;
    r = yy0;
    w = fn * pio2_2;
    yy0 = r - w;
    yy1 += (r - yy0) - w;
    r = yy0;
    w = fn * pio2_2t;
    yy0 = r - w;
    yy1 += (r - yy0) - w;
    r = yy0;
    w = fn * pio2_3;
    yy0 = r - w;
    yy1 += (r - yy0) - w;
    r = yy0;
    w = fn * pio2_3t;
    yy0 = r - w;
    yy1 += (r - yy0) - w;
    if (sign < 0.0) {
        *n_out = -n;
        *y0 = -yy0;
        *y1 = -yy1;
    } else {
        *n_out = n;
        *y0 = yy0;
        *y1 = yy1;
    }
}

TWRNGV2_MATH_HD inline double twdmath_sin(double x) {
    int n = 0;
    double y0 = 0.0;
    double y1 = 0.0;
    rem_pio2(x, &n, &y0, &y1);
    const int quadrant = n & 3;
    if (quadrant == 0) {
        return kernel_sin(y0, y1, 1);
    }
    if (quadrant == 1) {
        return kernel_cos(y0, y1);
    }
    if (quadrant == 2) {
        return -kernel_sin(y0, y1, 1);
    }
    return -kernel_cos(y0, y1);
}

TWRNGV2_MATH_HD inline double twdmath_cos(double x) {
    int n = 0;
    double y0 = 0.0;
    double y1 = 0.0;
    rem_pio2(x, &n, &y0, &y1);
    const int quadrant = n & 3;
    if (quadrant == 0) {
        return kernel_cos(y0, y1);
    }
    if (quadrant == 1) {
        return -kernel_sin(y0, y1, 1);
    }
    if (quadrant == 2) {
        return -kernel_cos(y0, y1);
    }
    return kernel_sin(y0, y1, 1);
}

static constexpr uint64_t TAU_BITS = 0x401921fb54442d18ull;

TWRNGV2_MATH_HD inline double twdmath_tau() {
    return f64_from_bits(TAU_BITS);
}

}  // namespace tw_rngv2
