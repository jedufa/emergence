"""Frozen semantic contract for the resident cross-path CUDA owner.

The compiled extension owns execution.  This module is the importable,
versioned registry counterpart used to prove that a declarative campaign and
the resident control ABI assign the same integer IDs and linkage ownership.
It intentionally contains no campaign shape, seed, generation, checkpoint, or
world-specific values.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

from engine.evolution.inheritance_registry_v1 import (
    LINKAGE_BLOCKS as _LINKAGE_BLOCKS,
    LINKED_PROFILE_DRAW_COUNT,
)


NATIVE_ABI: Final[str] = (
    "eiel_cuda_architecture_b_flattened_controller_cross_path_overlay_v15"
)
NATIVE_CHECKPOINT_SCHEMA: Final[str] = (
    "eiel_cuda_architecture_b_cross_path_resident_overlay_checkpoint_v2"
)

SEMANTIC_TYPES = MappingProxyType(
    {
        "inheritance_mode": "string",
        "founder_pattern": "string",
        "reproduction_path": "string",
    }
)
SEMANTIC_IDS = MappingProxyType(
    {
        "inheritance_mode": MappingProxyType(
            {
                "legacy_midpoint_v1": 0,
                "linked_parent_copy_v1": 1,
            }
        ),
        "founder_pattern": MappingProxyType(
            {
                "single_extreme_scalar": 0,
                "resource_specialization_extremes": 1,
                "correlated_multi_trait_packages": 2,
            }
        ),
        "reproduction_path": MappingProxyType(
            {
                "generation_boundary_refill": 0,
                "ordinary_live_birth": 1,
            }
        ),
    }
)
INHERITANCE_SEMANTIC: Final[str] = "inheritance_mode"
DRAW_BUDGETS = MappingProxyType(
    {
        "legacy_midpoint_v1": 0,
        "linked_parent_copy_v1": LINKED_PROFILE_DRAW_COUNT,
    }
)
LINKAGE_BLOCKS = MappingProxyType(
    {
        block_id: tuple(fields)
        for block_id, fields in _LINKAGE_BLOCKS
    }
)
AUDITED_LOCI: Final[tuple[str, ...]] = tuple(
    field
    for _block_id, fields in _LINKAGE_BLOCKS
    for field in fields
)


__all__ = [
    "AUDITED_LOCI",
    "DRAW_BUDGETS",
    "INHERITANCE_SEMANTIC",
    "LINKAGE_BLOCKS",
    "NATIVE_ABI",
    "NATIVE_CHECKPOINT_SCHEMA",
    "SEMANTIC_IDS",
    "SEMANTIC_TYPES",
]
