"""Lemire multiply-high bounded uint32 with addressed retries."""

from __future__ import annotations

from typing import Callable

from .contract import MAX_WORD_ORDINAL
from .status import Status


def lemire_threshold(n: int) -> int:
    """uint32 (-n) % n. Must match CUDA `uint32_t t = (0u - n) % n`."""

    return ((-n) & 0xFFFFFFFF) % n


def bounded_uint32(
    n: int,
    word_at_ordinal: Callable[[int], int],
) -> tuple[int, int, Status]:
    """Return (value, accepted_ordinal, status)."""

    if n == 1:
        return 0, 0, Status.RNG_OK
    if n == (1 << 32):
        return word_at_ordinal(0) & 0xFFFFFFFF, 0, Status.RNG_OK
    if n < 1 or n > (1 << 32):
        return 0, 0, Status.RNG_INVALID_N

    threshold = lemire_threshold(n)
    ordinal = 0
    while True:
        if ordinal > MAX_WORD_ORDINAL:
            return 0, ordinal, Status.RNG_ADDRESS_EXHAUSTED
        sample = word_at_ordinal(ordinal) & 0xFFFFFFFF
        product = sample * n
        low = product & 0xFFFFFFFF
        if low < n and low < threshold:
            ordinal += 1
            continue
        return (product >> 32) & 0xFFFFFFFF, ordinal, Status.RNG_OK
