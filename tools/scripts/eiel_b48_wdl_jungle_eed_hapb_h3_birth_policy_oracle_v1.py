#!/usr/bin/env python3
"""Bounded H3 oracle for host-authoritative ordinary fixed-MLP births."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import torch

from engine.perf.cuda_architecture_b_hapb_v1 import (
    CELL_COUNT_V1,
    HOT_PREFIX_ROWS_V1,
    POLICY_WIDTH_V1,
    SLOT_CAPACITY_V1,
    commit_hapb_ordinary_fixed_mlp_birth_rows_v1,
    initialize_selected_hapb_owner_v1,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(artifact: Path) -> object:
    spec = importlib.util.spec_from_file_location(
        "eiel_cuda_architecture_b_wdl_jungle_eed_hapb_v1", artifact
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("HAPB birth artifact cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _host_transaction_source_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError("refusing to overwrite immutable HAPB H3 receipt")
    torch.manual_seed(94_003)
    canonical = torch.randn(
        (CELL_COUNT_V1, SLOT_CAPACITY_V1, POLICY_WIDTH_V1), dtype=torch.float32
    )
    reference = canonical.clone()
    owner = initialize_selected_hapb_owner_v1(canonical, device="cuda")
    plan_width, tape_capacity = 2, 4
    parents = torch.tensor(
        [[300, 301] for _ in range(CELL_COUNT_V1)], dtype=torch.int64
    ).contiguous()
    # First children are P256 residents, second are overflow rows.
    children = torch.tensor(
        [[10, 400] for _ in range(CELL_COUNT_V1)], dtype=torch.int64
    ).contiguous()
    counts = torch.full((CELL_COUNT_V1,), plan_width, dtype=torch.int64)
    delta = torch.randn(
        (CELL_COUNT_V1, tape_capacity, POLICY_WIDTH_V1), dtype=torch.float32
    ).contiguous()
    mask = (
        torch.rand((CELL_COUNT_V1, tape_capacity, POLICY_WIDTH_V1)) < 0.125
    ).contiguous()
    semantic = torch.arange(
        CELL_COUNT_V1 * tape_capacity, dtype=torch.int64
    ).reshape(CELL_COUNT_V1, tape_capacity).contiguous()
    cursor = torch.zeros((CELL_COUNT_V1,), dtype=torch.int64)
    # Resident-layout arithmetic oracle only materializes the 96 planned parent
    # rows, never a CUDA C512 bank.
    resident_child_rows = torch.empty(
        (CELL_COUNT_V1, plan_width, POLICY_WIDTH_V1), device="cuda", dtype=torch.float32
    )
    for cell in range(CELL_COUNT_V1):
        resident_parent_rows = reference[cell, parents[cell]].to(device="cuda")
        resident_delta = delta[cell, :plan_width].to(device="cuda")
        resident_child_rows[cell].copy_(resident_parent_rows + resident_delta)
    cuda_mutate = getattr(
        _load(args.artifact),
        "eiel_cuda_architecture_b_hapb_ordinary_birth_mutate_rows_v1",
    )
    committed = commit_hapb_ordinary_fixed_mlp_birth_rows_v1(
        owner,
        cuda_mutate_export=cuda_mutate,
        parent_slots=parents,
        child_slots=children,
        birth_counts=counts,
        policy_birth_delta_tape=delta,
        policy_birth_mutation_mask_tape=mask,
        policy_birth_semantic_draws_tape=semantic,
        policy_birth_tape_cursor=cursor,
    )
    torch.cuda.synchronize()
    if not torch.equal(committed, children):
        raise RuntimeError("HAPB birth transaction did not return committed child slots")
    if not torch.equal(cursor, counts):
        raise RuntimeError("HAPB birth transaction did not advance taped cursors")
    host_rows = torch.stack(
        [canonical[cell, children[cell]] for cell in range(CELL_COUNT_V1)]
    )
    if not torch.equal(host_rows, resident_child_rows.cpu()):
        raise RuntimeError("HAPB host birth rows diverge from resident arithmetic")
    hot_rows = torch.stack(
        [owner.hot_prefix_cache[cell, 10] for cell in range(CELL_COUNT_V1)]
    )
    if not torch.equal(hot_rows.cpu(), canonical[:, 10]):
        raise RuntimeError("HAPB hot-prefix child rows were not refreshed after host commit")
    if bool((children[:, 0] >= HOT_PREFIX_ROWS_V1).any()) or bool(
        (children[:, 1] < HOT_PREFIX_ROWS_V1).any()
    ):
        raise RuntimeError("H3 fixture no longer covers hot and overflow child rows")

    # The policy transaction may have already committed early cells when a
    # later CUDA workbench fails.  Verify that it restores all authoritative
    # child rows and keeps all policy cursors unpublished in that case.
    before_rollback = canonical.clone()
    cursor_before_rollback = cursor.clone()
    mutation_calls = 0

    def fail_mid_b48_mutation(
        parent_rows: torch.Tensor,
        delta_rows: torch.Tensor,
        child_rows: torch.Tensor,
    ) -> None:
        nonlocal mutation_calls
        mutation_calls += 1
        if mutation_calls == 3:
            raise RuntimeError("injected HAPB B48 policy-workbench failure")
        cuda_mutate(parent_rows, delta_rows, child_rows)

    try:
        commit_hapb_ordinary_fixed_mlp_birth_rows_v1(
            owner,
            cuda_mutate_export=fail_mid_b48_mutation,
            parent_slots=parents,
            child_slots=children,
            birth_counts=counts,
            policy_birth_delta_tape=delta,
            policy_birth_mutation_mask_tape=mask,
            policy_birth_semantic_draws_tape=semantic,
            policy_birth_tape_cursor=cursor,
        )
    except RuntimeError as error:
        if "injected HAPB B48 policy-workbench failure" not in str(error):
            raise
    else:
        raise RuntimeError("HAPB rollback fixture did not fail mid-B48")
    if not torch.equal(canonical, before_rollback):
        raise RuntimeError("HAPB rollback did not restore canonical child rows")
    if not torch.equal(cursor, cursor_before_rollback):
        raise RuntimeError("HAPB rollback advanced policy cursors")
    args.output_root.mkdir(parents=True)
    path = args.output_root / "h3_birth_policy_oracle_receipt.json"
    receipt = {
        "schema_version": "eiel_b48_wdl_jungle_eed_hapb_h3_birth_policy_oracle_v1",
        "artifact_source_commit": json.loads(
            args.manifest.read_text(encoding="utf-8")
        )["committed_source_snapshot"]["build_source_commit"],
        "host_transaction_source_commit": _host_transaction_source_commit(),
        "artifact_sha256": _sha256(args.artifact),
        "manifest_sha256": _sha256(args.manifest),
        "physical_cells": CELL_COUNT_V1,
        "capacity": SLOT_CAPACITY_V1,
        "births_per_cell": plan_width,
        "host_commit_before_eligibility": True,
        "all_cell_policy_rollback_arena_bounded": True,
        "all_cell_policy_rollback_on_failure_verified": True,
        "resident_arithmetic_rows_exact": True,
        "hot_prefix_child_refresh_exact": True,
        "full_cuda_policy_bank_retained": False,
        "scope_limit": (
            "H3 fixed-MLP host-policy birth component only; full identity/scalar/"
            "drive/chemistry commit ordering, Genome V2/recombination birth parity, "
            "direct trajectory, checkpoint, and restore remain unproven"
        ),
    }
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["receipt_sha256"] = _sha256(path)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
