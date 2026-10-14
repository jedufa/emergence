"""simulation/metrics_pkg/adaptation_velocity.py

Phase 5a: Adaptation Velocity (AV).

AV is computed from EPI using:

1) Rolling mean smoothing of EPI (k=5 by default)
2) Difference of short-window mean and long-window mean over the smoothed series

Transient rule:
- If generation_index < long_window, AV is null.

Schema version must be frozen at "av_v2".
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional


AV_SCHEMA_VERSION = "av_v2"


def _mean(values) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


@dataclass
class AdaptationVelocityEngine:
    """Stateful AV engine that updates once per SafeBoundary."""

    smoothing_k: int = 5
    short_window: int = 20
    long_window: int = 100

    _epi_raw: Deque[float] = None  # type: ignore[assignment]
    _epi_smooth: Deque[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._epi_raw = deque(maxlen=max(self.long_window, self.smoothing_k) + 8)
        self._epi_smooth = deque(maxlen=self.long_window + 8)

    def update(self, generation_index: int, epi_value: float) -> Dict:
        """Add one EPI observation and compute AV for this generation."""

        epi_value_f = float(epi_value)
        self._epi_raw.append(epi_value_f)

        raw_tail = list(self._epi_raw)[-self.smoothing_k :]
        epi_smoothed = _mean(raw_tail)
        self._epi_smooth.append(epi_smoothed)

        av_value: Optional[float]
        transient = bool(generation_index < self.long_window)
        if transient:
            av_value = None
        else:
            smooth_tail = list(self._epi_smooth)
            short_tail = smooth_tail[-self.short_window :]
            long_tail = smooth_tail[-self.long_window :]
            av_value = _mean(short_tail) - _mean(long_tail)

        return {
            "schema_version": AV_SCHEMA_VERSION,
            "generation": int(generation_index),
            "av_value": None if av_value is None else float(av_value),
            "epi_smoothed": float(epi_smoothed),
            "smoothing_k": int(self.smoothing_k),
            "short_window": int(self.short_window),
            "long_window": int(self.long_window),
            "transient": bool(transient),
        }
