# Upstream attribution and redistribution terms: LICENSES/Random123.txt
"""Official Random123 Philox4x32-10 integer core."""

from __future__ import annotations

from typing import Final, Sequence

from .contract import CONTRACT


_M0: Final[int] = int(CONTRACT["core"]["constants_hex"]["M0"], 16)
_M1: Final[int] = int(CONTRACT["core"]["constants_hex"]["M1"], 16)
_W0: Final[int] = int(CONTRACT["core"]["constants_hex"]["W0"], 16)
_W1: Final[int] = int(CONTRACT["core"]["constants_hex"]["W1"], 16)
_MASK32: Final[int] = 0xFFFFFFFF
_ROUNDS: Final[int] = 10


def _mulhilo32(multiplier: int, ctr: int) -> tuple[int, int]:
    product = (multiplier & _MASK32) * (ctr & _MASK32)
    return product & _MASK32, (product >> 32) & _MASK32


def _round(
    ctr: tuple[int, int, int, int], key: tuple[int, int]
) -> tuple[int, int, int, int]:
    lo0, hi0 = _mulhilo32(_M0, ctr[0])
    lo1, hi1 = _mulhilo32(_M1, ctr[2])
    return (
        (hi1 ^ ctr[1] ^ key[0]) & _MASK32,
        lo1,
        (hi0 ^ ctr[3] ^ key[1]) & _MASK32,
        lo0,
    )


def philox4x32_10(
    counter: Sequence[int],
    key: Sequence[int],
    *,
    rounds: int = _ROUNDS,
) -> tuple[int, int, int, int]:
    """Return official Philox4x32-R output words (lane0..lane3)."""

    if rounds < 0 or rounds > 16:
        raise ValueError("Philox rounds must be in 0..16")
    ctr = (
        int(counter[0]) & _MASK32,
        int(counter[1]) & _MASK32,
        int(counter[2]) & _MASK32,
        int(counter[3]) & _MASK32,
    )
    k0 = int(key[0]) & _MASK32
    k1 = int(key[1]) & _MASK32
    if rounds == 0:
        return ctr
    ctr = _round(ctr, (k0, k1))
    for _ in range(1, rounds):
        k0 = (k0 + _W0) & _MASK32
        k1 = (k1 + _W1) & _MASK32
        ctr = _round(ctr, (k0, k1))
    return ctr


# Official Random123 kat_vectors, commit 9545ff6413f258be2f04c1d319d99aaef7521150.
# Format: (rounds, ctr[4], key[2], expected[4])
PHILOX_KATS: Final[tuple[tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...]] = (
    (
        10,
        (0x00000000, 0x00000000, 0x00000000, 0x00000000),
        (0x00000000, 0x00000000),
        (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8),
    ),
    (
        10,
        (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
        (0xFFFFFFFF, 0xFFFFFFFF),
        (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
    ),
    (
        10,
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
        (0xA4093822, 0x299F31D0),
        (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
    ),
    (
        7,
        (0x00000000, 0x00000000, 0x00000000, 0x00000000),
        (0x00000000, 0x00000000),
        (0x5F6FB709, 0x0D893F64, 0x4F121F81, 0x4F730A48),
    ),
    (
        7,
        (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
        (0xFFFFFFFF, 0xFFFFFFFF),
        (0x5207DDC2, 0x45165E59, 0x4D8EE751, 0x8C52F662),
    ),
    (
        7,
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
        (0xA4093822, 0x299F31D0),
        (0x4DFCCABA, 0x190A87F0, 0xC47362BA, 0xB6B5242A),
    ),
)
