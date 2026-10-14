from __future__ import annotations

import math

from engine.tw_rngv2.contract import CONTRACT_HASH
from engine.tw_rngv2.normal import box_muller_from_uniforms, box_muller_pair
from engine.tw_rngv2.philox import philox4x32_10
from engine.tw_rngv2.status import Status
from engine.tw_rngv2.twdmath import TAU, twdmath_cos, twdmath_log, twdmath_sin, twdmath_sqrt
from engine.tw_rngv2.uniforms import f64_from_bits


def test_signed_log_and_trig_match_fdlibm_sign_rules() -> None:
    assert math.isnan(twdmath_log(-1.0))
    assert twdmath_log(-0.0) == float("-inf")
    assert twdmath_sin(-math.pi / 2) < 0.0
    assert twdmath_cos(-math.pi) < 0.0
    assert abs(twdmath_sin(-0.3) + twdmath_sin(0.3)) < 1e-15


def test_twdmath_adversarial_vectors_are_finite() -> None:
    assert CONTRACT_HASH
    radii = [
        (0.5 + 0.0) * 2.0**-52,
        2.0**-52,
        2.0**-26,
        0.25,
        0.5,
        math.ldexp(1.0, -1),
        1.0 - 2.0**-52,
        math.nextafter(1.0, 0.0),
    ]
    for radius in radii:
        value = twdmath_log(radius)
        assert math.isfinite(value)
        assert math.isfinite(twdmath_sqrt(-2.0 * value))
    for angle in (0.0, 0.125, 0.25, 0.5, 0.75, math.nextafter(1.0, 0.0)):
        theta = TAU * angle
        assert math.isfinite(twdmath_sin(theta))
        assert math.isfinite(twdmath_cos(theta))
    z0, z1, status = box_muller_from_uniforms(0.0, 0.5)
    assert status is Status.RNG_OK
    assert math.isfinite(z0) and math.isfinite(z1)


def test_box_muller_from_philox_block_is_cacheless_and_finite() -> None:
    words = philox4x32_10((1, 2, 3, 4), (5, 6))
    z0, z1, status = box_muller_pair(words)
    assert status is Status.RNG_OK
    assert math.isfinite(z0) and math.isfinite(z1)
    again = box_muller_pair(words)
    assert again == (z0, z1, status)
    assert f64_from_bits(0x3FF0000000000000) == 1.0
