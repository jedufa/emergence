#!/usr/bin/env python3
"""Bounded B1 settlement parity for the HAPB prepare/resume seam.

The test keeps policy-logit authority outside its scope: it injects the
already-proven dense mapped logits and compares exact serial cache/Gumbel/PCG
settlement against independent NumPy PCG64 cell oracles.  This is deliberately
not direct trajectory, birth, or checkpoint evidence.
"""

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
ENTRYPOINT = "eiel_cuda_architecture_b_hapb_settle_prepared_policy_rows_v1"
CELLS, CAPACITY, CACHE_WIDTH, ACTIONS, ACTIVE = 48, 512, 2, 7, 256


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
    cache = np.full((CAPACITY, CACHE_WIDTH), -1, dtype=np.int32)
    actions = np.full((CAPACITY,), -1, dtype=np.int32)
    for row in range(ACTIVE):
        slot = 256 + row
        for cache_index in range(CACHE_WIDTH):
            scores = np.empty(ACTIONS, dtype=np.float32)
            for action in range(ACTIONS):
                uniform = np.clip(
                    generator.random(dtype=np.float32),
                    np.float32(1.0e-6), np.float32(1.0 - 1.0e-6),
                )
                scores[action] = np.float32(
                    logits[row, action] + np.float32(-np.log(-np.log(uniform)))
                )
            chosen = int(np.argmax(scores))
            cache[slot, cache_index] = chosen
            if cache_index == 0:
                actions[row] = chosen
    return actions, cache, np.asarray(_state_words(bitgen.state), dtype=np.uint64), [
        int(bitgen.state["has_uint32"]), int(bitgen.state["uinteger"])
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError("refusing to overwrite immutable HAPB settlement receipt")
    torch.manual_seed(2_003)
    logits_cpu = torch.randn((CELLS, CAPACITY, ACTIONS), dtype=torch.float32).numpy()
    expected = [_consume(logits_cpu[cell], 30_300 + cell) for cell in range(CELLS)]
    ordered_slots = torch.full((CELLS, CAPACITY), -1, dtype=torch.int32, device="cuda")
    ordered_slots[:, :ACTIVE] = torch.arange(256, 512, dtype=torch.int32, device="cuda")
    slot_tokens = torch.arange(CELLS * CAPACITY, dtype=torch.int64, device="cuda").reshape(CELLS, CAPACITY)
    cache_actions = torch.full((CELLS, CAPACITY, CACHE_WIDTH), -1, dtype=torch.int32, device="cuda")
    cache_token = torch.full((CELLS, CAPACITY), -1, dtype=torch.int64, device="cuda")
    cache_cursor = torch.full((CELLS, CAPACITY), -1, dtype=torch.int32, device="cuda")
    cache_next = torch.full((CELLS, CAPACITY), -1, dtype=torch.int32, device="cuda")
    row_actions = torch.full((CELLS, CAPACITY), -1, dtype=torch.int32, device="cuda")
    miss_rows = torch.full((CELLS, CAPACITY), -1, dtype=torch.int64, device="cuda")
    miss_rows[:, :ACTIVE] = torch.arange(ACTIVE, dtype=torch.int64, device="cuda")
    active_counts = torch.full((CELLS,), ACTIVE, dtype=torch.int64, device="cuda")
    miss_counts = torch.full((CELLS,), ACTIVE, dtype=torch.int64, device="cuda")
    initial_words = np.asarray([
        _state_words(np.random.PCG64(30_300 + cell).state) for cell in range(CELLS)
    ], dtype=np.uint64)
    states = torch.from_numpy(initial_words.copy()).to(device="cuda")
    buffers = torch.zeros((CELLS, 2), dtype=torch.int64, device="cuda")
    entry = getattr(_load(args.artifact), ENTRYPOINT)
    counters = entry(
        ordered_slots, slot_tokens, cache_actions, cache_token, cache_cursor,
        cache_next, torch.from_numpy(logits_cpu).to(device="cuda"), row_actions,
        miss_rows, active_counts, miss_counts, states, buffers,
        1.2, 0.9, 1.0, 1.2, 1.4,
    )
    torch.cuda.synchronize()
    actual_actions = row_actions.cpu().numpy()
    actual_cache = cache_actions.cpu().numpy()
    actual_states = states.cpu().numpy()
    actual_buffers = buffers.cpu().numpy()
    expected_actions = np.stack([value[0] for value in expected])
    expected_cache = np.stack([value[1] for value in expected])
    expected_states = np.stack([value[2] for value in expected])
    expected_buffers = np.asarray([value[3] for value in expected], dtype=np.int64)
    expected_counters = np.tile(
        np.asarray([ACTIVE, 0, ACTIVE, ACTIVE, 1, 1, 0, ACTIVE * CACHE_WIDTH * ACTIONS], dtype=np.int64),
        (CELLS, 1),
    )
    exact = (
        np.array_equal(actual_actions, expected_actions)
        and np.array_equal(actual_cache, expected_cache)
        and np.array_equal(actual_states, expected_states)
        and np.array_equal(actual_buffers, expected_buffers)
        and np.array_equal(counters.cpu().numpy(), expected_counters)
    )
    if not exact:
        raise RuntimeError("HAPB serial settlement diverges from NumPy PCG64 oracle")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True)
    path = args.output_root / "h2_settlement_oracle_receipt.json"
    receipt = {
        "schema_version": "eiel_b48_wdl_jungle_eed_hapb_h2_settlement_oracle_v1",
        "source_commit": manifest["committed_source_snapshot"]["build_source_commit"],
        "artifact_sha256": _sha256(args.artifact),
        "manifest_sha256": _sha256(args.manifest),
        "physical_cells": CELLS,
        "capacity": CAPACITY,
        "cache_width": CACHE_WIDTH,
        "active_overflow_rows_per_cell": ACTIVE,
        "actions_exact": True,
        "cache_exact": True,
        "pcg64_state_exact": True,
        "pcg64_buffer_exact": True,
        "policy_counters_exact": True,
        "scope_limit": "B1 prepared-settlement component only; classifier, compact scatter, route, birth, and checkpoint remain unproven",
    }
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["receipt_sha256"] = _sha256(path)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
