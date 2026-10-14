from __future__ import annotations

from engine.tw_rngv2.contract import CONTRACT_HASH
from engine.tw_rngv2.philox import PHILOX_KATS, philox4x32_10


def test_official_random123_kats_cpu() -> None:
    assert CONTRACT_HASH
    for rounds, counter, key, expected in PHILOX_KATS:
        got = philox4x32_10(counter, key, rounds=rounds)
        assert got == expected


def test_wrap_and_lane_order_boundaries() -> None:
    words = philox4x32_10((0xFFFFFFFF, 0, 0xFFFFFFFF, 1), (0xFFFFFFFF, 0))
    assert all(0 <= word <= 0xFFFFFFFF for word in words)
    other = philox4x32_10((0, 0, 0, 0), (1, 0))
    assert words != other
