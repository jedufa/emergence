"""tests/integration/metrics/test_adaptation_velocity.py

Phase 5a validation tests for AV (Adaptation Velocity).
"""

from __future__ import annotations

import pytest

from simulation.metrics_pkg.adaptation_velocity import AdaptationVelocityEngine, AV_SCHEMA_VERSION


def test_av_transient_rules() -> None:
    eng = AdaptationVelocityEngine(smoothing_k=2, short_window=2, long_window=4)

    r1 = eng.update(generation_index=1, epi_value=0.0)
    r2 = eng.update(generation_index=2, epi_value=0.0)
    r3 = eng.update(generation_index=3, epi_value=0.0)
    r4 = eng.update(generation_index=4, epi_value=10.0)

    assert r1["transient"] is True and r1["av_value"] is None
    assert r2["transient"] is True and r2["av_value"] is None
    assert r3["transient"] is True and r3["av_value"] is None

    assert r4["transient"] is False
    assert r4["av_value"] == pytest.approx(1.25)


def test_av_schema_frozen() -> None:
    eng = AdaptationVelocityEngine(smoothing_k=2, short_window=2, long_window=4)
    r = eng.update(generation_index=1, epi_value=0.0)

    assert r["schema_version"] == AV_SCHEMA_VERSION
    assert set(r.keys()) == {
        "schema_version",
        "generation",
        "av_value",
        "epi_smoothed",
        "smoothing_k",
        "short_window",
        "long_window",
        "transient",
    }
