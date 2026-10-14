"""Exact V2 integer-to-float maps."""

from __future__ import annotations

import struct


def u32_to_f32_half_open(u32: int) -> float:
    q24 = (int(u32) & 0xFFFFFFFF) >> 8
    return float(q24) * (2.0**-24)


def u32_to_f32_bits(u32: int) -> int:
    return struct.unpack("<I", struct.pack("<f", u32_to_f32_half_open(u32)))[0]


def u64_from_lanes(lane_lo: int, lane_hi: int) -> int:
    return (int(lane_lo) & 0xFFFFFFFF) | ((int(lane_hi) & 0xFFFFFFFF) << 32)


def u64_to_f64_half_open(u64: int) -> float:
    q53 = (int(u64) & ((1 << 64) - 1)) >> 11
    return float(q53) * (2.0**-53)


def u64_to_f64_open(u64: int) -> float:
    q52 = (int(u64) & ((1 << 64) - 1)) >> 12
    return (float(q52) + 0.5) * (2.0**-52)


def f64_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def f64_from_bits(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(bits) & ((1 << 64) - 1)))[0]
