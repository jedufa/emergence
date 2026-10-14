from __future__ import annotations

import math

from engine.tw_rngv2.address import EventAddress
from engine.tw_rngv2.normal import box_muller_pair
from engine.tw_rngv2.oracle import PrototypeOracle
from engine.tw_rngv2.philox import philox4x32_10
from engine.tw_rngv2.stats import chi_square_uniform, moments, pearson_r
from engine.tw_rngv2.uniforms import u64_from_lanes, u64_to_f64_half_open


def test_bounded_statistical_sanity() -> None:
    oracle = PrototypeOracle()
    uniforms = []
    normals = []
    buckets = [0] * 16
    discrete = [0] * 7
    domain_a = []
    domain_b = []
    for index in range(4096):
        address = EventAddress(0, 0x0331, index + 1, 1, 1)
        result = oracle.generate(address, n=7)
        uniforms.append(
            u64_to_f64_half_open(u64_from_lanes(result.words[0], result.words[1]))
        )
        z0, z1, _ = box_muller_pair(result.words)
        normals.extend((z0, z1))
        buckets[int(uniforms[-1] * 16)] += 1
        discrete[result.bounded] += 1
        domain_a.append(float(result.words[0]))
        other = PrototypeOracle().generate(EventAddress(0, 0x0330, index + 1, 1, 1))
        domain_b.append(float(other.words[0]))
    chi = chi_square_uniform(buckets)
    stats = moments(normals)
    corr = pearson_r(domain_a, domain_b)
    assert chi < 50.0
    assert abs(stats["mean"]) < 0.08
    assert 0.85 < stats["variance"] < 1.15
    assert abs(corr) < 0.08
    assert min(discrete) > 0
    assert math.isfinite(philox4x32_10((9, 8, 7, 6), (5, 4))[0])
