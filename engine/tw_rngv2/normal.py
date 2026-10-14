"""Cacheless pair-valued Box-Muller under TWDMATH-1."""

from __future__ import annotations

import math

from .status import Status
from .twdmath import TAU, twdmath_cos, twdmath_log, twdmath_sin, twdmath_sqrt
from .uniforms import f64_bits, u64_from_lanes, u64_to_f64_half_open, u64_to_f64_open


def box_muller_from_uniforms(u_angle: float, u_radius: float) -> tuple[float, float, Status]:
    if not (0.0 <= u_angle < 1.0):
        return 0.0, 0.0, Status.RNG_NONFINITE
    if not (0.0 < u_radius < 1.0):
        return 0.0, 0.0, Status.RNG_NONFINITE
    log_r = twdmath_log(u_radius)
    radius = twdmath_sqrt(-2.0 * log_r)
    theta = TAU * u_angle
    z0 = radius * twdmath_cos(theta)
    z1 = radius * twdmath_sin(theta)
    if not (math.isfinite(z0) and math.isfinite(z1)):
        return z0, z1, Status.RNG_NONFINITE
    return z0, z1, Status.RNG_OK


def box_muller_pair(words: tuple[int, int, int, int]) -> tuple[float, float, Status]:
    u_angle = u64_to_f64_half_open(u64_from_lanes(words[0], words[1]))
    u_radius = u64_to_f64_open(u64_from_lanes(words[2], words[3]))
    return box_muller_from_uniforms(u_angle, u_radius)


def box_muller_bits(words: tuple[int, int, int, int]) -> tuple[int, int, Status]:
    z0, z1, status = box_muller_pair(words)
    return f64_bits(z0), f64_bits(z1), status


def scale_shift(z: float, mu: float, sigma: float) -> float:
    """round(sigma*z) then round(+mu). Contraction forbidden."""

    return (float(sigma) * float(z)) + float(mu)
