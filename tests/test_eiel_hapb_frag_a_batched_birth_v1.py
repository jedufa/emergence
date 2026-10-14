from __future__ import annotations

import os

import pytest
import torch

from engine.perf.cuda_architecture_b_hapb_v1 import (
    CELL_COUNT_V1,
    HAPB_FRAG_A_BATCHED_BIRTH_ENABLED_V1,
    POLICY_WIDTH_V1,
    SLOT_CAPACITY_V1,
    _hapb_frag_a_use_batched_birth_v1,
    commit_hapb_ordinary_fixed_mlp_birth_rows_v1,
    hapb_frag_a_birth_counters_snapshot_v1,
    initialize_selected_hapb_owner_v1,
)


def test_frag_a_production_birth_defaults_to_per_cell_after_213() -> None:
    os.environ.pop("EIEL_HAPB_FRAG_A_PER_CELL_BIRTH", None)
    os.environ.pop("EIEL_HAPB_FRAG_A_BATCHED_BIRTH", None)
    assert HAPB_FRAG_A_BATCHED_BIRTH_ENABLED_V1 is False
    assert _hapb_frag_a_use_batched_birth_v1() is False
    os.environ["EIEL_HAPB_FRAG_A_BATCHED_BIRTH"] = "1"
    try:
        assert _hapb_frag_a_use_batched_birth_v1() is True
    finally:
        os.environ.pop("EIEL_HAPB_FRAG_A_BATCHED_BIRTH", None)


def _mutate(parent: torch.Tensor, delta: torch.Tensor, child: torch.Tensor) -> None:
    child.copy_(parent + delta)


def _plan(*, max_births: int = 4) -> dict[str, torch.Tensor]:
    parents = torch.full((CELL_COUNT_V1, max_births), -1, dtype=torch.int64)
    children = torch.full((CELL_COUNT_V1, max_births), -1, dtype=torch.int64)
    counts = torch.zeros(CELL_COUNT_V1, dtype=torch.int64)
    cursor = torch.zeros(CELL_COUNT_V1, dtype=torch.int64)
    active = (0, 3, 7, 11, 20)
    for index, cell in enumerate(active):
        count = 1 + (index % 3)
        counts[cell] = count
        cursor[cell] = index
        for birth in range(count):
            parents[cell, birth] = birth
            children[cell, birth] = 8 + birth
    tape_capacity = 16
    delta = torch.zeros(
        (CELL_COUNT_V1, tape_capacity, 1), dtype=torch.float32
    )
    for index, cell in enumerate(active):
        delta[cell, index : index + int(counts[cell])] = float(index + 1)
    mask = torch.zeros((CELL_COUNT_V1, tape_capacity, 1), dtype=torch.bool)
    semantic = torch.zeros((CELL_COUNT_V1, tape_capacity), dtype=torch.int64)
    return {
        "parents": parents,
        "children": children,
        "counts": counts,
        "cursor": cursor,
        "delta": delta,
        "mask": mask,
        "semantic": semantic,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_frag_a_batched_birth_matches_per_cell_bytes_and_collapses_syncs() -> None:
    device = torch.device("cuda", 0)
    canonical = torch.zeros(
        (CELL_COUNT_V1, SLOT_CAPACITY_V1, POLICY_WIDTH_V1),
        dtype=torch.float32,
    )
    torch.manual_seed(11)
    for cell in range(CELL_COUNT_V1):
        canonical[cell, :8].normal_()
    owner_a = initialize_selected_hapb_owner_v1(
        canonical.clone(),
        device=device,
        populate_hot_prefix=False,
        content_alias_enabled=False,
    )
    owner_b = initialize_selected_hapb_owner_v1(
        canonical.clone(),
        device=device,
        populate_hot_prefix=False,
        content_alias_enabled=False,
    )
    plan_a = _plan()
    plan_b = {key: value.clone() for key, value in plan_a.items()}
    os.environ["EIEL_HAPB_FRAG_A_PER_CELL_BIRTH"] = "1"
    committed_a = commit_hapb_ordinary_fixed_mlp_birth_rows_v1(
        owner_a,
        cuda_mutate_export=_mutate,
        parent_slots=plan_a["parents"],
        child_slots=plan_a["children"],
        birth_counts=plan_a["counts"],
        policy_birth_delta_tape=plan_a["delta"],
        policy_birth_mutation_mask_tape=plan_a["mask"],
        policy_birth_semantic_draws_tape=plan_a["semantic"],
        policy_birth_tape_cursor=plan_a["cursor"],
    )
    per_cell = hapb_frag_a_birth_counters_snapshot_v1()
    os.environ.pop("EIEL_HAPB_FRAG_A_PER_CELL_BIRTH", None)
    os.environ["EIEL_HAPB_FRAG_A_BATCHED_BIRTH"] = "1"
    committed_b = commit_hapb_ordinary_fixed_mlp_birth_rows_v1(
        owner_b,
        cuda_mutate_export=_mutate,
        parent_slots=plan_b["parents"],
        child_slots=plan_b["children"],
        birth_counts=plan_b["counts"],
        policy_birth_delta_tape=plan_b["delta"],
        policy_birth_mutation_mask_tape=plan_b["mask"],
        policy_birth_semantic_draws_tape=plan_b["semantic"],
        policy_birth_tape_cursor=plan_b["cursor"],
    )
    batched = hapb_frag_a_birth_counters_snapshot_v1()
    assert torch.equal(committed_a, committed_b)
    assert torch.equal(owner_a.canonical_host_bank, owner_b.canonical_host_bank)
    assert torch.equal(plan_a["cursor"], plan_b["cursor"])
    assert per_cell["active_cells"] == batched["active_cells"] == 5
    assert per_cell["device_syncs"] == 20
    os.environ.pop("EIEL_HAPB_FRAG_A_BATCHED_BIRTH", None)
    assert batched["device_syncs"] == 3
    assert batched["launches"] == 1
    assert batched["device_syncs"] < per_cell["device_syncs"]
