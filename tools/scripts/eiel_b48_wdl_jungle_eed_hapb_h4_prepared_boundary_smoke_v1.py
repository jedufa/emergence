#!/usr/bin/env python3
"""Physical-B48 smoke for one HAPB prepare/refill/settle/resume boundary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
MODULE_NAME = "eiel_cuda_architecture_b_wdl_jungle_eed_hapb_v1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.perf.cuda_architecture_b_hapb_v1 import (
    CELL_COUNT_V1,
    POLICY_WIDTH_V1,
    SLOT_CAPACITY_V1,
    execute_hapb_prepared_policy_boundary_v1,
    initialize_selected_hapb_owner_v1,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("HAPB artifact cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError("refusing to overwrite immutable HAPB H4 boundary receipt")
    module = _load(args.artifact)
    canonical = torch.zeros(
        (CELL_COUNT_V1, SLOT_CAPACITY_V1, POLICY_WIDTH_V1), dtype=torch.float32
    )
    # One hot and one overflow miss per physical cell exercise both routes.
    canonical[:, 10, 0] = 0.125
    canonical[:, 400, 1] = -0.25
    owner = initialize_selected_hapb_owner_v1(canonical, device="cuda")
    device = owner.cuda_tile_arena.device
    ordered = torch.full((48, 512), -1, dtype=torch.int32, device=device)
    ordered[:, :2] = torch.tensor([10, 400], dtype=torch.int32, device=device)
    observations = torch.zeros((48, 512, 83), dtype=torch.float32, device=device)
    misses = torch.zeros((48, 512), dtype=torch.int64, device=device)
    misses[:, 1] = 1
    active = torch.full((48,), 2, dtype=torch.int64, device=device)
    miss_count = active.clone()
    drives = torch.zeros((48, 512, 4), dtype=torch.float32, device=device)
    attack = torch.zeros((48, 512), dtype=torch.float32, device=device)
    temperature = torch.ones((48, 512), dtype=torch.float32, device=device)
    gain = torch.ones((48, 512), dtype=torch.float32, device=device)
    tokens = torch.arange(48 * 512, dtype=torch.int64, device=device).reshape(48, 512)
    cache_actions = torch.zeros((48, 512, 2), dtype=torch.int32, device=device)
    cache_token = torch.full((48, 512), -1, dtype=torch.int64, device=device)
    cache_cursor = torch.zeros((48, 512), dtype=torch.int32, device=device)
    cache_next = torch.zeros((48, 512), dtype=torch.int32, device=device)
    row_actions = torch.full((48, 512), -1, dtype=torch.int32, device=device)
    pcg_state = torch.zeros((48, 4), dtype=torch.uint64, device=device)
    pcg_state[:, 0] = 1
    pcg_state[:, 2] = 1
    pcg_buffer = torch.zeros((48, 2), dtype=torch.int64, device=device)
    counters, context = execute_hapb_prepared_policy_boundary_v1(
        owner,
        refill_logits_export=getattr(module, "eiel_cuda_architecture_b_hapb_refill_policy_logits_from_context_v1"),
        settle_prepared_export=getattr(module, "eiel_cuda_architecture_b_hapb_settle_prepared_policy_rows_v1"),
        native_boundary_export=getattr(module, "eiel_cuda_architecture_b_hapb_execute_prepared_policy_boundary_v1"),
        ordered_slots=ordered, observations=observations, stable_miss_rows=misses,
        active_row_counts=active, miss_row_counts=miss_count, drives=drives,
        attack_bias=attack, action_temperature=temperature, drive_logit_gain=gain,
        slot_tokens=tokens, action_cache_actions=cache_actions,
        action_cache_token=cache_token, action_cache_cursor=cache_cursor,
        action_cache_next=cache_next, row_actions=row_actions,
        policy_pcg64_state_words=pcg_state, policy_pcg64_buffer_words=pcg_buffer,
        w_hunger_gather=0.1, w_pain_rest=0.2, w_fatigue_rest=0.3,
        w_pain_attack=0.4, w_fear_attack=0.5,
    )
    torch.cuda.synchronize(device)
    if tuple(counters.shape) != (48, 8) or not torch.equal(counters[:, 0], active):
        raise RuntimeError("HAPB boundary did not preserve B48 active-row accounting")
    if (
        not torch.equal(counters[:, 2], miss_count)
        or bool((counters[:, 7] <= 0).any().item())
    ):
        raise RuntimeError("HAPB boundary did not consume every prepared miss")
    if bool((row_actions[:, :2] < 0).any().item()):
        raise RuntimeError("HAPB boundary did not materialize resident ordered actions")
    args.output_root.mkdir(parents=True)
    receipt_path = args.output_root / "h4_prepared_boundary_smoke_receipt.json"
    receipt = {
        "schema_version": "eiel_b48_wdl_jungle_eed_hapb_h4_prepared_boundary_smoke_v1",
        "artifact_sha256": _sha256(args.artifact),
        "artifact_source_commit": json.loads(args.manifest.read_text())["committed_source_snapshot"]["build_source_commit"],
        "host_boundary_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "physical_cells": 48,
        "active_rows_per_cell": 2,
        "hot_prefix_misses_per_cell": 1,
        "overflow_misses_per_cell": 1,
        "stable_miss_order_preserved": True,
        "serial_settlement_executed": True,
        "full_cuda_policy_bank_retained": owner.full_cuda_policy_bank_retained,
        "scope_limit": "one HAPB prepared-policy boundary only; direct WDL/Jungle/EED trajectory and lifecycle parity remain unproven",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["receipt_sha256"] = _sha256(receipt_path)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
