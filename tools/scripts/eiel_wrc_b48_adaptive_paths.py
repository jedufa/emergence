"""Canonical repository paths for the frozen WRC B48 adaptive design inputs."""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DESIGN_ROOT = REPOSITORY_ROOT / "_spec/campaigns/v1/wrc_b48_adaptive"
C1_PLAN_PATH = DESIGN_ROOT / "B48_EXPLORATORY_C1_PLAN.json"
C1_SPEC_PATH = DESIGN_ROOT / "B48_EXPLORATORY_C1_SPEC.json"
DESIGN_BINDING_PATH = DESIGN_ROOT / "B48_EXPLORATORY_DESIGN_BINDING.json"
INITIAL_DESIGN_PATH = DESIGN_ROOT / "B48_EXPLORATORY_INITIAL_DESIGN.csv"


__all__ = [
    "C1_PLAN_PATH",
    "C1_SPEC_PATH",
    "DESIGN_BINDING_PATH",
    "DESIGN_ROOT",
    "INITIAL_DESIGN_PATH",
    "REPOSITORY_ROOT",
]
