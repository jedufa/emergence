#!/usr/bin/env python3
"""Bounded B1 component parity for HAPB mapped policy logits.

This is deliberately not a route or settlement receipt.  It compares the
selected tiled invocation with a bounded resident-layout CUDA oracle on the
same rows, preserving the exact scalar helper and proving neither invocation
advances CUDA RNG.  Action-cache/PCG settlement remains a later direct-boundary
obligation.
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
HAPB_ENTRYPOINT = "eiel_cuda_architecture_b_hapb_compact_policy_logits_v1"
RESIDENT_ORACLE_ENTRYPOINT = (
    "eiel_cuda_architecture_b_hapb_resident_policy_logits_oracle_v1"
)
POLICY_WIDTH = 9_991
OBSERVATION_WIDTH = 83
HIDDEN_SIZE = 64
ACTION_COUNT = 7
SLOT_COUNT = 48 * 512


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(artifact: Path) -> Any:
    spec = importlib.util.spec_from_file_location(MODULE_NAME, artifact)
    if spec is None or spec.loader is None:
        raise RuntimeError("HAPB artifact cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _overflow_global_slots(rows: int) -> torch.Tensor:
    # The first P256 rows per physical cell are deliberately absent: this is
    # the selected hot-prefix overflow transaction, not a pure-HAPB relabel.
    values = [
        (index // 256) * 512 + 256 + index % 256
        for index in range(rows)
    ]
    return torch.tensor(values, device="cuda", dtype=torch.int64)


def _case(
    hapb: Any,
    resident: Any,
    *,
    rows: int,
) -> dict[str, object]:
    torch.manual_seed(91_000 + rows)
    resident_rows = torch.randn((rows, POLICY_WIDTH), device="cuda", dtype=torch.float32)
    # A copy here is bounded to exactly one tested tile and represents a staged
    # arena input; it is never a B48/C512 policy bank.
    mapped_rows = resident_rows.clone()
    observations = torch.randn((rows, OBSERVATION_WIDTH), device="cuda", dtype=torch.float32)
    slots = _overflow_global_slots(rows)
    drives = torch.randn((SLOT_COUNT, 4), device="cuda", dtype=torch.float32)
    attack = torch.randn((SLOT_COUNT,), device="cuda", dtype=torch.float32)
    temperature = torch.rand((SLOT_COUNT,), device="cuda", dtype=torch.float32)
    gain = torch.randn((SLOT_COUNT,), device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    rng_before = torch.cuda.get_rng_state()
    resident_logits = resident(
        resident_rows, observations, slots, drives, attack, temperature, gain,
        HIDDEN_SIZE, ACTION_COUNT, 1.2, 0.9, 1.0, 1.2, 1.4,
    )
    hapb_logits = hapb(
        mapped_rows, observations, slots, drives, attack, temperature, gain,
        HIDDEN_SIZE, ACTION_COUNT, 1.2, 0.9, 1.0, 1.2, 1.4,
    )
    torch.cuda.synchronize()
    rng_after = torch.cuda.get_rng_state()
    if not torch.equal(resident_logits, hapb_logits):
        mismatch = int((resident_logits != hapb_logits).sum().item())
        raise RuntimeError(f"HAPB/resident logits differ for tile {rows}: {mismatch}")
    if not torch.equal(rng_before, rng_after):
        raise RuntimeError(f"policy logit evaluation advanced CUDA RNG for tile {rows}")
    return {
        "tile_rows": rows,
        "mapped_rows": rows,
        "global_slot_min": int(slots.min().item()),
        "global_slot_max": int(slots.max().item()),
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
        raise RuntimeError("refusing to overwrite immutable HAPB parity root")
    module = _load(args.artifact)
    hapb = getattr(module, HAPB_ENTRYPOINT)
    resident = getattr(module, RESIDENT_ORACLE_ENTRYPOINT)
    results = [_case(hapb, resident, rows=rows) for rows in (512, 4096)]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True)
    path = args.output_root / "h2_policy_logit_oracle_receipt.json"
    receipt = {
        "schema_version": "eiel_b48_wdl_jungle_eed_hapb_h2_policy_logit_oracle_v1",
        "source_commit": manifest["committed_source_snapshot"]["build_source_commit"],
        "artifact_sha256": _sha256(args.artifact),
        "manifest_sha256": _sha256(args.manifest),
        "representation": "p256_hot_prefix_overflow",
        "cases": results,
        "full_cuda_policy_bank_retained": False,
        "scope_limit": (
            "B1 component logits/RNG parity only; serial PCG settlement, actions, "
            "cache, direct trajectory, birth, and checkpoint parity remain unproven"
        ),
    }
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["receipt_sha256"] = _sha256(path)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
