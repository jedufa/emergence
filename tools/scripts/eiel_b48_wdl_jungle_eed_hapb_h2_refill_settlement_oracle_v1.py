#!/usr/bin/env python3
"""B1/B2 component parity for guarded HAPB refill plus exact settlement."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


MODULE_NAME = "eiel_cuda_architecture_b_wdl_jungle_eed_hapb_v1"
REFILL = "eiel_cuda_architecture_b_hapb_refill_policy_logits_from_context_v1"
RESIDENT = "eiel_cuda_architecture_b_hapb_resident_policy_logits_oracle_v1"
SETTLE = "eiel_cuda_architecture_b_hapb_settle_prepared_policy_rows_v1"
CELLS, CAPACITY, HOT, WIDTH, OBS, ACTIONS = 48, 512, 256, 9_991, 83, 7
TOTAL = CELLS * CAPACITY


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(artifact: Path) -> Any:
    spec = importlib.util.spec_from_file_location(MODULE_NAME, artifact)
    if spec is None or spec.loader is None:
        raise RuntimeError("HAPB artifact cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state_words(state: dict[str, object]) -> list[int]:
    inner = state["state"]
    if not isinstance(inner, dict):
        raise RuntimeError("NumPy PCG64 state is malformed")
    raw_state, raw_inc = int(inner["state"]), int(inner["inc"])
    return [
        raw_state & ((1 << 64) - 1), raw_state >> 64,
        raw_inc & ((1 << 64) - 1), raw_inc >> 64,
    ]


def _consume(logits: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    bitgen = np.random.PCG64(seed)
    generator = np.random.Generator(bitgen)
    cache = np.full((CAPACITY, 2), -1, dtype=np.int32)
    actions = np.full((CAPACITY,), -1, dtype=np.int32)
    for row in range(CAPACITY):
        for cache_index in range(2):
            scores = np.empty(ACTIONS, dtype=np.float32)
            for action in range(ACTIONS):
                uniform = np.clip(
                    generator.random(dtype=np.float32),
                    np.float32(1.0e-6), np.float32(1.0 - 1.0e-6),
                )
                scores[action] = np.float32(
                    logits[row, action] + np.float32(-np.log(-np.log(uniform)))
                )
            choice = int(np.argmax(scores))
            cache[row, cache_index] = choice
            if cache_index == 0:
                actions[row] = choice
    return actions, cache, np.asarray(_state_words(bitgen.state), dtype=np.uint64), [
        int(bitgen.state["has_uint32"]), int(bitgen.state["uinteger"])
    ]


def _maps() -> tuple[torch.Tensor, torch.Tensor]:
    cells = torch.arange(CELLS, dtype=torch.int64).unsqueeze(1) * CAPACITY
    hot = (cells + torch.arange(HOT, dtype=torch.int64)).reshape(-1).contiguous()
    overflow = (
        cells + torch.arange(HOT, CAPACITY, dtype=torch.int64)
    ).reshape(-1).contiguous()
    return hot, overflow


def _resident_dense(
    resident: Any,
    canonical: torch.Tensor,
    observations: torch.Tensor,
    drives: torch.Tensor,
    attack: torch.Tensor,
    temperature: torch.Tensor,
    gain: torch.Tensor,
    *,
    tile_rows: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    flat = canonical.reshape(TOTAL, WIDTH)
    for start in range(0, TOTAL, tile_rows):
        count = min(tile_rows, TOTAL - start)
        slots = torch.arange(start, start + count, device="cuda", dtype=torch.int64)
        rows.append(resident(
            flat.narrow(0, start, count).to(device="cuda", non_blocking=False),
            observations.narrow(0, start, count),
            slots, drives, attack, temperature, gain,
            64, ACTIONS, 1.2, 0.9, 1.0, 1.2, 1.4,
        ))
    return torch.cat(rows).reshape(CELLS, CAPACITY, ACTIONS)


def _settle_exact(settle: Any, logits: torch.Tensor) -> None:
    ordered = torch.arange(CAPACITY, dtype=torch.int32, device="cuda").expand(
        CELLS, CAPACITY
    ).clone()
    tokens = torch.arange(TOTAL, dtype=torch.int64, device="cuda").reshape(CELLS, CAPACITY)
    cache = torch.full((CELLS, CAPACITY, 2), -1, dtype=torch.int32, device="cuda")
    cache_token = torch.full((CELLS, CAPACITY), -1, dtype=torch.int64, device="cuda")
    cache_cursor = torch.full((CELLS, CAPACITY), -1, dtype=torch.int32, device="cuda")
    cache_next = torch.full((CELLS, CAPACITY), -1, dtype=torch.int32, device="cuda")
    row_actions = torch.full((CELLS, CAPACITY), -1, dtype=torch.int32, device="cuda")
    miss_rows = torch.arange(CAPACITY, dtype=torch.int64, device="cuda").expand(
        CELLS, CAPACITY
    ).clone()
    counts = torch.full((CELLS,), CAPACITY, dtype=torch.int64, device="cuda")
    initial_words = np.asarray(
        [_state_words(np.random.PCG64(40_400 + cell).state) for cell in range(CELLS)],
        dtype=np.uint64,
    )
    states = torch.from_numpy(initial_words.copy()).to(device="cuda")
    buffers = torch.zeros((CELLS, 2), dtype=torch.int64, device="cuda")
    settle(
        ordered, tokens, cache, cache_token, cache_cursor, cache_next, logits,
        row_actions, miss_rows, counts, counts, states, buffers,
        1.2, 0.9, 1.0, 1.2, 1.4,
    )
    torch.cuda.synchronize()
    expected = [_consume(logits[cell].cpu().numpy(), 40_400 + cell) for cell in range(CELLS)]
    if not (
        np.array_equal(row_actions.cpu().numpy(), np.stack([item[0] for item in expected]))
        and np.array_equal(cache.cpu().numpy(), np.stack([item[1] for item in expected]))
        and np.array_equal(states.cpu().numpy(), np.stack([item[2] for item in expected]))
        and np.array_equal(
            buffers.cpu().numpy(),
            np.asarray([item[3] for item in expected], dtype=np.int64),
        )
    ):
        raise RuntimeError("guarded HAPB refill settlement diverges from PCG64 oracle")


def _case(
    module: Any,
    canonical: torch.Tensor,
    hot_cache: torch.Tensor,
    observations: torch.Tensor,
    hot_rows: torch.Tensor,
    overflow_rows: torch.Tensor,
    drives: torch.Tensor,
    attack: torch.Tensor,
    temperature: torch.Tensor,
    gain: torch.Tensor,
    *,
    tile_rows: int,
) -> dict[str, object]:
    pinned_tile = torch.empty((tile_rows, WIDTH), dtype=torch.float32, pin_memory=True)
    cuda_tile = torch.empty((tile_rows, WIDTH), dtype=torch.float32, device="cuda")
    cursor_state = torch.zeros((CELLS, 4), dtype=torch.int64, device="cuda")
    cursor_buffer = torch.zeros((CELLS, 2), dtype=torch.int64, device="cuda")
    refill = getattr(module, REFILL)
    resident = getattr(module, RESIDENT)
    rng_before = torch.cuda.get_rng_state()
    actual = refill(
        canonical, pinned_tile, hot_cache, cuda_tile, overflow_rows, overflow_rows,
        cursor_state, cursor_buffer, observations, hot_rows, hot_rows,
        drives, attack, temperature, gain, 64, ACTIONS, 1.2, 0.9, 1.0, 1.2, 1.4,
    )
    torch.cuda.synchronize()
    if not torch.equal(rng_before, torch.cuda.get_rng_state()):
        raise RuntimeError("guarded refill advanced CUDA RNG before settlement")
    expected = _resident_dense(
        resident, canonical, observations, drives, attack, temperature, gain,
        tile_rows=tile_rows,
    )
    torch.cuda.synchronize()
    if not torch.equal(actual, expected):
        mismatch = int((actual != expected).sum().item())
        hot_mismatch = int((actual[:, :HOT] != expected[:, :HOT]).sum().item())
        overflow_mismatch = int(
            (actual[:, HOT:] != expected[:, HOT:]).sum().item()
        )
        first = (actual != expected).nonzero(as_tuple=False)[0].tolist()
        cell, slot, action = (int(value) for value in first)
        raise RuntimeError(
            "guarded refill logits differ for tile "
            f"{tile_rows}: total={mismatch} hot={hot_mismatch} "
            f"overflow={overflow_mismatch} first={first} "
            f"actual={actual[cell, slot, action].item()} "
            f"expected={expected[cell, slot, action].item()}"
        )
    _settle_exact(getattr(module, SETTLE), actual)
    return {
        "tile_rows": tile_rows,
        "physical_cells": CELLS,
        "physical_capacity": CAPACITY,
        "hot_rows": int(hot_rows.numel()),
        "overflow_rows": int(overflow_rows.numel()),
        "bitwise_logits_equal": True,
        "actions_cache_pcg64_exact": True,
        "cuda_rng_unchanged_before_settlement": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError("refusing to overwrite immutable HAPB refill receipt")
    torch.manual_seed(93_002)
    canonical = torch.randn((CELLS, CAPACITY, WIDTH), dtype=torch.float32)
    if canonical.is_pinned():
        raise RuntimeError("canonical authority unexpectedly pinned")
    hot_cache = canonical[:, :HOT].to(device="cuda", non_blocking=False)
    observations = torch.randn((TOTAL, OBS), device="cuda", dtype=torch.float32)
    drives = torch.randn((TOTAL, 4), device="cuda", dtype=torch.float32)
    attack = torch.randn((TOTAL,), device="cuda", dtype=torch.float32)
    temperature = torch.rand((TOTAL,), device="cuda", dtype=torch.float32)
    gain = torch.randn((TOTAL,), device="cuda", dtype=torch.float32)
    hot_rows, overflow_rows = _maps()
    module = _load(args.artifact)
    cases = [
        _case(
            module, canonical, hot_cache, observations, hot_rows, overflow_rows,
            drives, attack, temperature, gain, tile_rows=tile_rows,
        )
        for tile_rows in (512, 4096)
    ]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True)
    path = args.output_root / "h2_refill_settlement_oracle_receipt.json"
    receipt = {
        "schema_version": "eiel_b48_wdl_jungle_eed_hapb_h2_refill_settlement_oracle_v1",
        "source_commit": manifest["committed_source_snapshot"]["build_source_commit"],
        "artifact_sha256": _sha256(args.artifact),
        "manifest_sha256": _sha256(args.manifest),
        "representation": "p256_hot_prefix_plus_pageable_overflow_tile",
        "cases": cases,
        "full_cuda_policy_bank_retained": False,
        "scope_limit": (
            "B1/B2 guarded refill and prepared-settlement component parity only; "
            "classifier integration, direct WDL/Jungle/EED trajectory, ordinary "
            "birth, checkpoint, and restore remain unproven"
        ),
    }
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["receipt_sha256"] = _sha256(path)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
