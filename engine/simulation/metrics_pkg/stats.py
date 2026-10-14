from __future__ import annotations

import math
from typing import List


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def std(xs: List[float]) -> float:
    if not xs:
        return 0.0
    m = mean(xs)
    return math.sqrt(mean([(x - m) ** 2 for x in xs]))


def percentile(xs: List[float], p: float) -> float:
    """
    Percentile with p in [0, 100].
    Safe for empty lists and singletons.
    """
    if not xs:
        return 0.0

    ys = sorted(xs)
    n = len(ys)

    if n == 1:
        return float(ys[0])

    # Clamp p
    if p <= 0:
        return float(ys[0])
    if p >= 100:
        return float(ys[-1])

    # Convert percentile to position
    pos = (p / 100.0) * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))

    if lo == hi:
        return float(ys[lo])

    frac = pos - lo
    return float(ys[lo] * (1.0 - frac) + ys[hi] * frac)
