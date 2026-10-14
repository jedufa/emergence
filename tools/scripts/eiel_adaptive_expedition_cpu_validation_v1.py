"""CLI wrapper for the test-only CPU Adaptive Expedition validation harness."""

from __future__ import annotations

from pathlib import Path
import runpy


_HARNESS = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "eiel_adaptive_expedition_cpu_validation_v1.py"
)


if __name__ == "__main__":
    runpy.run_path(str(_HARNESS), run_name="__main__")
