from __future__ import annotations

import pytest

from engine.evolution.inheritance_registry_v1 import (
    LINKAGE_BLOCKS as REGISTERED_LINKAGE_BLOCKS,
    LINKED_PROFILE_DRAW_COUNT,
)
from engine.perf import inheritance_cross_path_gpu_native as native


def test_native_semantic_ids_match_resident_cross_path_control_abi() -> None:
    assert native.SEMANTIC_IDS["inheritance_mode"] == {
        "legacy_midpoint_v1": 0,
        "linked_parent_copy_v1": 1,
    }
    assert native.SEMANTIC_IDS["reproduction_path"] == {
        "generation_boundary_refill": 0,
        "ordinary_live_birth": 1,
    }
    assert native.DRAW_BUDGETS == {
        "legacy_midpoint_v1": 0,
        "linked_parent_copy_v1": LINKED_PROFILE_DRAW_COUNT,
    }


def test_native_linkage_contract_is_complete_and_immutable() -> None:
    expected = {
        block_id: tuple(fields)
        for block_id, fields in REGISTERED_LINKAGE_BLOCKS
    }
    assert native.LINKAGE_BLOCKS == expected
    assert native.AUDITED_LOCI == tuple(
        field
        for _block_id, fields in REGISTERED_LINKAGE_BLOCKS
        for field in fields
    )
    with pytest.raises(TypeError):
        native.SEMANTIC_IDS["inheritance_mode"]["new"] = 9
