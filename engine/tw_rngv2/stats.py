"""Bounded statistical sanity. Not a substitute for bitwise parity."""

from __future__ import annotations

import math
from typing import Sequence


def chi_square_uniform(buckets: Sequence[int]) -> float:
    expected = sum(buckets) / len(buckets)
    if expected <= 0.0:
        raise ValueError("empty histogram")
    return sum((count - expected) ** 2 / expected for count in buckets)


def moments(values: Sequence[float]) -> dict[str, float]:
    n = len(values)
    if n < 2:
        raise ValueError("need at least two samples")
    mean = sum(values) / n
    var = sum((value - mean) ** 2 for value in values) / (n - 1)
    std = math.sqrt(var)
    skew = (
        sum(((value - mean) / std) ** 3 for value in values) / n if std else 0.0
    )
    kurt = (
        sum(((value - mean) / std) ** 4 for value in values) / n if std else 0.0
    )
    return {
        "n": float(n),
        "mean": mean,
        "variance": var,
        "skewness": skew,
        "kurtosis": kurt,
    }


def pearson_r(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("paired samples required")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)
