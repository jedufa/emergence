#!/usr/bin/env python3
"""Retain the physical B48 selected-HAPB ownership smoke evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.perf.cuda_architecture_b_hapb_v1 import initialize_selected_hapb_owner_v1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    canonical = torch.zeros((48, 512, 9991), dtype=torch.float32, device="cpu")
    owner = initialize_selected_hapb_owner_v1(canonical, device="cuda:0")
    torch.cuda.synchronize()
    receipt = {
        "schema_version": "eiel_b48_wdl_jungle_eed_hapb_h2_owner_smoke_v1",
        "physical_cells": 48,
        "canonical_shape": list(owner.canonical_host_bank.shape),
        "canonical_pageable_cpu": (
            owner.canonical_host_bank.device.type == "cpu"
            and not owner.canonical_host_bank.is_pinned()
        ),
        "pinned_staging_count": len(owner.pinned_staging_buffers),
        "pinned_staging_shape": list(owner.pinned_staging_buffers[0].shape),
        "pinned_staging_admitted": owner.pinned_staging_buffers[0].is_pinned(),
        "hot_prefix_shape": list(owner.hot_prefix_cache.shape),
        "tile_arena_shape": list(owner.cuda_tile_arena.shape),
        "full_cuda_policy_bank_retained": owner.full_cuda_policy_bank_retained,
        "scope_limit": "H2 ownership initialization only; not trajectory, parity, or qualification",
    }
    path = args.output_root / "h2_owner_smoke_receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["receipt_sha256"] = _sha256(path)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
