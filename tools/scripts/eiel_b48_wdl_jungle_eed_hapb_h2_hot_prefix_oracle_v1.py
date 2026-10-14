#!/usr/bin/env python3
"""Bounded B1 exact oracle for the HAPB P256 hot-prefix path.

The tested cache has the physical B48/P256 shape.  Rows are evaluated in the
selected 512 and 4096 row tiles and compared with the resident CUDA scalar
helper.  The resident rows are a test-only reference view, not a runtime bank.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch


MODULE_NAME = "eiel_cuda_architecture_b_wdl_jungle_eed_hapb_v1"
HOT_PREFIX_ENTRYPOINT = "eiel_cuda_architecture_b_hapb_hot_prefix_policy_logits_v1"
RESIDENT_ORACLE_ENTRYPOINT = "eiel_cuda_architecture_b_hapb_resident_policy_logits_oracle_v1"
CELLS, CAPACITY, HOT_PREFIX = 48, 512, 256
POLICY_WIDTH, OBSERVATION_WIDTH, HIDDEN_SIZE, ACTION_COUNT = 9_991, 83, 64, 7
SLOT_COUNT = CELLS * CAPACITY


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(artifact: Path) -> Any:
    spec = importlib.util.spec_from_file_location(MODULE_NAME, artifact)
    if spec is None or spec.loader is None:
        raise RuntimeError("HAPB artifact cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hot_slots(start: int, rows: int) -> torch.Tensor:
    # Iterate the complete B48/P256 physical hot cache in row-major order.
    indices = torch.arange(start, start + rows, device="cuda", dtype=torch.int64)
    return (indices // HOT_PREFIX) * CAPACITY + (indices % HOT_PREFIX)


def _case(
    hot_prefix: Any,
    resident: Any,
    cache: torch.Tensor,
    drives: torch.Tensor,
    attack: torch.Tensor,
    temperature: torch.Tensor,
    gain: torch.Tensor,
    *,
    tile_rows: int,
) -> dict[str, object]:
    total_rows = CELLS * HOT_PREFIX
    exact_rows = 0
    for start in range(0, total_rows, tile_rows):
        rows = min(tile_rows, total_rows - start)
        slots = _hot_slots(start, rows)
        cells = torch.div(slots, CAPACITY, rounding_mode="floor")
        local = torch.remainder(slots, CAPACITY)
        # This is explicitly an oracle-only bounded reference materialization.
        resident_rows = cache[cells, local].contiguous()
        observations = torch.randn(
            (rows, OBSERVATION_WIDTH), device="cuda", dtype=torch.float32
        )
        torch.cuda.synchronize()
        rng_before = torch.cuda.get_rng_state()
        resident_logits = resident(
            resident_rows, observations, slots, drives, attack, temperature, gain,
            HIDDEN_SIZE, ACTION_COUNT, 1.2, 0.9, 1.0, 1.2, 1.4,
        )
        hot_logits = hot_prefix(
            cache, observations, slots, drives, attack, temperature, gain,
            HIDDEN_SIZE, ACTION_COUNT, 1.2, 0.9, 1.0, 1.2, 1.4,
        )
        torch.cuda.synchronize()
        if not torch.equal(resident_logits, hot_logits):
            mismatch = int((resident_logits != hot_logits).sum().item())
            raise RuntimeError(
                f"hot-prefix/resident logits differ at {start}:{start + rows}: {mismatch}"
            )
        if not torch.equal(rng_before, torch.cuda.get_rng_state()):
            raise RuntimeError("policy logit evaluation advanced CUDA RNG")
        exact_rows += rows
    return {
        "tile_rows": tile_rows,
        "physical_hot_rows": total_rows,
        "physical_cells": CELLS,
        "bitwise_logits_equal": True,
        "cuda_rng_unchanged": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError("refusing to overwrite immutable HAPB hot-prefix receipt")
    torch.manual_seed(92_001)
    module = _load(args.artifact)
    hot_prefix = getattr(module, HOT_PREFIX_ENTRYPOINT)
    resident = getattr(module, RESIDENT_ORACLE_ENTRYPOINT)
    cache = torch.randn(
        (CELLS, HOT_PREFIX, POLICY_WIDTH), device="cuda", dtype=torch.float32
    )
    drives = torch.randn((SLOT_COUNT, 4), device="cuda", dtype=torch.float32)
    attack = torch.randn((SLOT_COUNT,), device="cuda", dtype=torch.float32)
    temperature = torch.rand((SLOT_COUNT,), device="cuda", dtype=torch.float32)
    gain = torch.randn((SLOT_COUNT,), device="cuda", dtype=torch.float32)
    cases = [
        _case(hot_prefix, resident, cache, drives, attack, temperature, gain, tile_rows=rows)
        for rows in (512, 4096)
    ]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True)
    path = args.output_root / "h2_hot_prefix_oracle_receipt.json"
    receipt = {
        "schema_version": "eiel_b48_wdl_jungle_eed_hapb_h2_hot_prefix_oracle_v1",
        "source_commit": manifest["committed_source_snapshot"]["build_source_commit"],
        "artifact_sha256": _sha256(args.artifact),
        "manifest_sha256": _sha256(args.manifest),
        "representation": "p256_hot_prefix",
        "cases": cases,
        "full_cuda_policy_bank_retained": False,
        "scope_limit": (
            "B1 hot-prefix logits/RNG component parity only; overflow refill, "
            "serial settlement composition, direct trajectory, birth, and checkpoint "
            "parity remain unproven"
        ),
    }
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["receipt_sha256"] = _sha256(path)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
