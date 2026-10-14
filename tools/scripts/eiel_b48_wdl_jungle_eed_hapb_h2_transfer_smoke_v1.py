#!/usr/bin/env python3
"""Retain a physical B48 HAPB map/stage/arena/compact-kernel smoke receipt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.perf.cuda_architecture_b_hapb_v1 import (  # noqa: E402
    initialize_selected_hapb_owner_v1,
    stable_overflow_policy_map_v1,
)


MODULE_NAME = "eiel_cuda_architecture_b_wdl_jungle_eed_hapb_v1"
ENTRYPOINT = "eiel_cuda_architecture_b_hapb_compact_policy_logits_v1"
CONTEXT_ENTRYPOINT = "eiel_cuda_architecture_b_hapb_validate_policy_context_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(MODULE_NAME, args.artifact)
    if spec is None or spec.loader is None:
        raise RuntimeError("HAPB artifact cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry = getattr(module, ENTRYPOINT)
    validate_context = getattr(module, CONTEXT_ENTRYPOINT)
    canonical = torch.zeros((48, 512, 9991), dtype=torch.float32)
    owner = initialize_selected_hapb_owner_v1(canonical, device="cuda:0")
    jungle_cursor_state = torch.zeros((48, 4), dtype=torch.uint64, device="cuda:0")
    jungle_cursor_buffer = torch.zeros((48, 2), dtype=torch.int64, device="cuda:0")
    slots = torch.full((48, 256), -1, dtype=torch.int64)
    slots[0] = torch.arange(256, 512, dtype=torch.int64)
    slots[1] = torch.arange(256, 512, dtype=torch.int64)
    stable_map = stable_overflow_policy_map_v1(slots)
    rows = stable_map.global_slots
    if validate_context(
        owner.canonical_host_bank,
        owner.pinned_staging_buffers[0],
        owner.hot_prefix_cache,
        owner.cuda_tile_arena,
        rows,
        stable_map.active_row_indices,
        jungle_cursor_state,
        jungle_cursor_buffer,
    ) is not True:
        raise RuntimeError("HAPB native policy context validation did not pass")
    observations = torch.randn((512, 83), device="cuda", dtype=torch.float32)
    slot_count = 48 * 512
    zeros = torch.zeros((slot_count,), device="cuda", dtype=torch.float32)
    parts = owner.compact_logits_v1(
        entry, rows, observations,
        torch.zeros((slot_count, 4), device="cuda", dtype=torch.float32),
        zeros, torch.ones_like(zeros), zeros,
        hidden_size=64, w_hunger_gather=.1, w_pain_rest=.2,
        w_fatigue_rest=.3, w_pain_attack=.4, w_fear_attack=.5,
    )
    torch.cuda.synchronize()
    if len(parts) != 1 or tuple(parts[0].shape) != (512, 7) or not torch.isfinite(parts[0]).all().item():
        raise RuntimeError("HAPB one-tile transfer smoke did not pass")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema_version": "eiel_b48_wdl_jungle_eed_hapb_h2_transfer_smoke_v1",
        "source_commit": manifest["committed_source_snapshot"]["build_source_commit"],
        "artifact_sha256": _sha256(args.artifact),
        "manifest_sha256": _sha256(args.manifest),
        "physical_cells": 48,
        "overflow_rows": int(rows.numel()),
        "tile_rows": 512,
        "tile_count": len(parts),
        "compact_logits_shape": list(parts[0].shape),
        "canonical_pageable_cpu": not owner.canonical_host_bank.is_pinned(),
        "pinned_staging_count": len(owner.pinned_staging_buffers),
        "native_policy_context_validated": True,
        "jungle_pre_predator_policy_cursor_shape": [48, 4],
        "jungle_pre_predator_policy_buffer_shape": [48, 2],
        "full_cuda_policy_bank_retained": owner.full_cuda_policy_bank_retained,
        "scope_limit": "H2 transfer/kernel smoke only; not full route or resident-oracle parity",
    }
    path = args.output_root / "h2_transfer_smoke_receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["receipt_sha256"] = _sha256(path)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
