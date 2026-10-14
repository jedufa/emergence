#!/usr/bin/env python3
"""H3 component oracle for HAPB host-bank checkpoint and guarded cache restore."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.perf.cuda_architecture_b_hapb_v1 import (
    CELL_COUNT_V1,
    HAPBContractError,
    HOT_PREFIX_ROWS_V1,
    POLICY_WIDTH_V1,
    SLOT_CAPACITY_V1,
    export_hapb_policy_bank_checkpoint_v1,
    initialize_selected_hapb_owner_v1,
    restore_hapb_policy_owner_checkpoint_v1,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError("refusing to overwrite immutable HAPB H3 receipt")

    canonical = torch.zeros(
        (CELL_COUNT_V1, SLOT_CAPACITY_V1, POLICY_WIDTH_V1),
        dtype=torch.float32,
        device="cpu",
    )
    # Sparse sentinels force byte-sensitive coverage across both hot and
    # overflow rows without allocating a duplicate host reference bank.
    for cell in range(CELL_COUNT_V1):
        canonical[cell, cell % HOT_PREFIX_ROWS_V1, cell] = float(cell + 1)
        canonical[cell, 256 + (cell % 256), POLICY_WIDTH_V1 - 1 - cell] = -float(cell + 1)
    owner = initialize_selected_hapb_owner_v1(canonical, device="cuda")
    checkpoint = export_hapb_policy_bank_checkpoint_v1(owner)
    digest_before = str(checkpoint["sha256"])
    del owner
    torch.cuda.empty_cache()

    restored = restore_hapb_policy_owner_checkpoint_v1(checkpoint, device="cuda")
    if restored.hot_prefix_valid:
        raise RuntimeError("checkpoint restore incorrectly marks the new CUDA cache valid")
    try:
        restored.require_hot_prefix_cache_v1()
    except HAPBContractError:
        pass
    else:
        raise RuntimeError("checkpoint restore exposed invalid hot-prefix bytes")
    restored = restored.rehydrate_hot_prefix_cache_v1()
    torch.cuda.synchronize()
    if not restored.hot_prefix_valid:
        raise RuntimeError("guarded rehydrate did not mark hot-prefix cache valid")
    checkpoint_after = export_hapb_policy_bank_checkpoint_v1(restored)
    if str(checkpoint_after["sha256"]) != digest_before:
        raise RuntimeError("checkpoint round trip changed canonical host-policy bytes")
    for cell in range(CELL_COUNT_V1):
        hot_slot = cell % HOT_PREFIX_ROWS_V1
        if not torch.equal(
            restored.require_hot_prefix_cache_v1()[cell, hot_slot].cpu(),
            restored.canonical_host_bank[cell, hot_slot],
        ):
            raise RuntimeError("rehydrated hot-prefix sentinel differs from host authority")

    args.output_root.mkdir(parents=True)
    receipt_path = args.output_root / "h3_checkpoint_oracle_receipt.json"
    receipt = {
        "schema_version": "eiel_b48_wdl_jungle_eed_hapb_h3_checkpoint_oracle_v1",
        "source_commit": _source_commit(),
        "physical_cells": CELL_COUNT_V1,
        "capacity": SLOT_CAPACITY_V1,
        "policy_width": POLICY_WIDTH_V1,
        "checkpoint_sha256": digest_before,
        "canonical_host_policy_bytes_exact": True,
        "hot_prefix_invalid_after_restore": True,
        "hot_prefix_guarded_rehydrate_exact": True,
        "tile_and_staging_recreated_not_serialized": True,
        "full_cuda_policy_bank_retained": restored.full_cuda_policy_bank_retained,
        "scope_limit": (
            "H3 host-policy checkpoint component only; WDL/Jungle/EED state, "
            "birth-plan rollback, direct trajectory, and full checkpoint/restore "
            "parity remain unproven"
        ),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["receipt_sha256"] = _sha256(receipt_path)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
