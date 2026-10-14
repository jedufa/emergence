# Upstream attribution and redistribution terms: LICENSES/fdlibm.txt
"""TWDMATH-1: V2-owned IEEE binary64 software math.

Coefficients and operation order are frozen here. glibc/libdevice are not
authorities. The same algorithm is implemented in native/twdmath.cuh.
"""

from __future__ import annotations

import math
import struct
from typing import Final

from .contract import CONTRACT


def _u64(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def _f64(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(bits) & ((1 << 64) - 1)))[0]


def _hi32(value: float) -> int:
    return (_u64(value) >> 32) & 0xFFFFFFFF


def _signed_hi32(value: float) -> int:
    unsigned = _hi32(value)
    return unsigned - 0x100000000 if unsigned >= 0x80000000 else unsigned


def _lo32(value: float) -> int:
    return _u64(value) & 0xFFFFFFFF


def _set_hi32(value: float, hi: int) -> float:
    return _f64((_u64(value) & 0xFFFFFFFF) | ((int(hi) & 0xFFFFFFFF) << 32))


def _from_words(hi: int, lo: int) -> float:
    return _f64(((int(hi) & 0xFFFFFFFF) << 32) | (int(lo) & 0xFFFFFFFF))


# Frozen hex constants (fdlibm-derived, V2-owned).
LN2_HI: Final[float] = _from_words(0x3FE62E42, 0xFEE00000)
LN2_LO: Final[float] = _from_words(0x3DEA39EF, 0x35793C76)
TWO54: Final[float] = _from_words(0x43500000, 0x00000000)
LG1: Final[float] = _from_words(0x3FE55555, 0x55555593)
LG2: Final[float] = _from_words(0x3FD99999, 0x9997FA04)
LG3: Final[float] = _from_words(0x3FD24924, 0x94229359)
LG4: Final[float] = _from_words(0x3FCC71C5, 0x1D8E78AF)
LG5: Final[float] = _from_words(0x3FC74664, 0x96CB03DE)
LG6: Final[float] = _from_words(0x3FC39A09, 0xD078C69F)
LG7: Final[float] = _from_words(0x3FC2F112, 0xDF3E5244)

INVPIO2: Final[float] = _from_words(0x3FE45F30, 0x6DC9C883)
PIO2_1: Final[float] = _from_words(0x3FF921FB, 0x54400000)
PIO2_1T: Final[float] = _from_words(0x3DD0B461, 0x1A626331)
PIO2_2: Final[float] = _from_words(0x3DD0B461, 0x1A600000)
PIO2_2T: Final[float] = _from_words(0x3BA3198A, 0x2E037073)
PIO2_3: Final[float] = _from_words(0x3BA3198A, 0x2E000000)
PIO2_3T: Final[float] = _from_words(0x397B839A, 0x252049C1)

S1: Final[float] = _from_words(0xBFC55555, 0x55555549)
S2: Final[float] = _from_words(0x3F811111, 0x1110F8A6)
S3: Final[float] = _from_words(0xBF2A01A0, 0x19C161D5)
S4: Final[float] = _from_words(0x3EC71DE3, 0x57B1FE7D)
S5: Final[float] = _from_words(0xBE5AE5E6, 0x8A2B9CEB)
S6: Final[float] = _from_words(0x3DE5D93A, 0x5ACFD57C)

C1: Final[float] = _from_words(0x3FA55555, 0x5555554C)
C2: Final[float] = _from_words(0xBF56C16C, 0x16C15177)
C3: Final[float] = _from_words(0x3EFA01A0, 0x19CB1590)
C4: Final[float] = _from_words(0xBE927E4F, 0x809C52AD)
C5: Final[float] = _from_words(0x3E21EE9E, 0xBDB4B1C4)
C6: Final[float] = _from_words(0xBDA8FAE9, 0xBE8838D4)

HALF: Final[float] = 0.5
ONE: Final[float] = 1.0
ZERO: Final[float] = 0.0
TWO: Final[float] = 2.0
TAU: Final[float] = _f64(int(CONTRACT["twdmath_1"]["tau_hex"], 16))


def twdmath_sqrt(x: float) -> float:
    """IEEE-754 correctly rounded binary64 square root."""

    return math.sqrt(float(x))


def twdmath_log(x: float) -> float:
    """fdlibm-derived __ieee754_log with frozen coefficients."""

    hx = _hi32(x)
    lx = _lo32(x)
    k = 0
    if _signed_hi32(x) < 0x00100000:
        if ((hx & 0x7FFFFFFF) | lx) == 0:
            return -math.inf
        if _signed_hi32(x) < 0:
            return math.nan
        k -= 54
        x = float(x) * TWO54
        hx = _hi32(x)
        lx = _lo32(x)
    if hx >= 0x7FF00000:
        return float(x) + float(x)
    k += (hx >> 20) - 1023
    hx &= 0x000FFFFF
    i = (hx + 0x95F64) & 0x100000
    x = _set_hi32(x, hx | (i ^ 0x3FF00000))
    k += i >> 20
    f = x - ONE
    if (0x000FFFFF & (2 + hx)) < 3:
        if f == ZERO:
            if k == 0:
                return ZERO
            dk = float(k)
            return dk * LN2_HI + dk * LN2_LO
        r = f * f * (HALF - 0.33333333333333333 * f)
        if k == 0:
            return f - r
        dk = float(k)
        return dk * LN2_HI - ((r - dk * LN2_LO) - f)
    s = f / (TWO + f)
    z = s * s
    w = z * z
    t1 = w * (LG2 + w * (LG4 + w * LG6))
    t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)))
    r = t2 + t1
    i2 = hx - 0x6147A
    j = 0x6B851 - hx
    i2 |= j
    if i2 > 0:
        hfsq = HALF * f * f
        if k == 0:
            return f - (hfsq - s * (hfsq + r))
        dk = float(k)
        return dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f)
    if k == 0:
        return f - s * (f - r)
    dk = float(k)
    return dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f)


def _kernel_sin(x: float, y: float, iy: int) -> float:
    z = x * x
    v = z * x
    r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)))
    if iy == 0:
        return x + v * (S1 + z * r)
    return x - ((z * (HALF * y - v * r) - y) - v * S1)


def _kernel_cos(x: float, y: float) -> float:
    z = x * x
    w = z * z
    r = z * (C1 + z * (C2 + z * C3)) + w * w * (C4 + z * (C5 + z * C6))
    hz = HALF * z
    w = ONE - hz
    return w + (((ONE - w) - hz) + (z * r - x * y))


def _rem_pio2(x: float) -> tuple[int, float, float]:
    """Cody-Waite reduction for |x| <= 2*pi + small slack."""

    hx = _hi32(x)
    ix = hx & 0x7FFFFFFF
    sign = -1.0 if _signed_hi32(x) < 0 else 1.0
    ax = abs(float(x))
    if ix <= 0x3FE921FB:
        return 0, float(x), 0.0
    fn = ax * INVPIO2 + HALF
    n = int(fn)
    fn = float(n)
    r = ax - fn * PIO2_1
    w = fn * PIO2_1T
    y0 = r - w
    y1 = (r - y0) - w
    r = y0
    w = fn * PIO2_2
    y0 = r - w
    y1 += (r - y0) - w
    r = y0
    w = fn * PIO2_2T
    y0 = r - w
    y1 += (r - y0) - w
    r = y0
    w = fn * PIO2_3
    y0 = r - w
    y1 += (r - y0) - w
    r = y0
    w = fn * PIO2_3T
    y0 = r - w
    y1 += (r - y0) - w
    if sign < 0.0:
        return -n, -y0, -y1
    return n, y0, y1


def twdmath_sin(x: float) -> float:
    n, y0, y1 = _rem_pio2(x)
    quadrant = n & 3
    if quadrant == 0:
        return _kernel_sin(y0, y1, 1)
    if quadrant == 1:
        return _kernel_cos(y0, y1)
    if quadrant == 2:
        return -_kernel_sin(y0, y1, 1)
    return -_kernel_cos(y0, y1)


def twdmath_cos(x: float) -> float:
    n, y0, y1 = _rem_pio2(x)
    quadrant = n & 3
    if quadrant == 0:
        return _kernel_cos(y0, y1)
    if quadrant == 1:
        return -_kernel_sin(y0, y1, 1)
    if quadrant == 2:
        return -_kernel_cos(y0, y1)
    return _kernel_sin(y0, y1, 1)
