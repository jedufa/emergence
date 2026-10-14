#!/usr/bin/env python3
"""Physical-B48 smoke for the native HAPB direct-route owner context only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

from engine.perf.cuda_architecture_b_hapb_v1 import (
    CELL_COUNT_V1,
    POLICY_WIDTH_V1,
    SLOT_CAPACITY_V1,
    initialize_selected_hapb_owner_v1,
)
import torch


# The immutable root83 artifact exports the canonical extension module name;
# the direct-route context is a symbol within that module, not a second
# extension module.
MODULE_NAME = "eiel_cuda_architecture_b_wdl_jungle_eed_hapb_v1"
NATIVE_EXPORT = "eiel_cuda_architecture_b_hapb_validate_direct_route_context_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def _load(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("HAPB artifact cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError("refusing to overwrite immutable HAPB route-context receipt")
    module = _load(args.artifact)
    export = getattr(module, NATIVE_EXPORT, None)
    if not callable(export):
        raise RuntimeError("HAPB native direct-route context export is unavailable")
    canonical = torch.zeros(
        (CELL_COUNT_V1, SLOT_CAPACITY_V1, POLICY_WIDTH_V1), dtype=torch.float32
    )
    owner = initialize_selected_hapb_owner_v1(canonical, device="cuda")
    if export(
        owner.canonical_host_bank,
        owner.pinned_staging_buffers[0],
        owner.require_hot_prefix_cache_v1(),
        owner.cuda_tile_arena,
    ) is not True:
        raise RuntimeError("HAPB direct-route context did not validate")
    torch.cuda.synchronize(owner.cuda_tile_arena.device)
    args.output_root.mkdir(parents=True)
    receipt_path = args.output_root / "h4_direct_route_context_smoke_receipt.json"
    receipt = {
        "schema_version": "eiel_b48_wdl_jungle_eed_hapb_h4_direct_route_context_smoke_v1",
        "source_commit": _source_commit(),
        "artifact_sha256": _sha256(args.artifact),
        "artifact_source_commit": json.loads(
            args.manifest.read_text(encoding="utf-8")
        )["committed_source_snapshot"]["build_source_commit"],
        "manifest_sha256": _sha256(args.manifest),
        "physical_cells": CELL_COUNT_V1,
        "capacity": SLOT_CAPACITY_V1,
        "host_policy_authority": "pageable_exact_b48_c512_9991",
        "pinned_staging_buffers": 1,
        "hot_prefix_rows": 256,
        "tile_rows": owner.tile_rows,
        "full_cuda_policy_bank_retained": owner.full_cuda_policy_bank_retained,
        "scope_limit": (
            "native long-lived direct-route context only; no direct WDL/Jungle/EED "
            "tick dispatch, birth plan, checkpoint projection, or route parity"
        ),
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt["receipt_sha256"] = _sha256(receipt_path)
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
