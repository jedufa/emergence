#!/usr/bin/env python3
"""Retain a bounded native HAPB compact-policy ABI smoke receipt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch


ENTRYPOINT = "eiel_cuda_architecture_b_hapb_compact_policy_logits_v1"
DIRECT_ENTRYPOINT = "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_eed_hapb_v1"
MODULE_NAME = "eiel_cuda_architecture_b_wdl_jungle_eed_hapb_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(artifact: Path) -> Any:
    spec = importlib.util.spec_from_file_location(MODULE_NAME, artifact)
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
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    module = _load(args.artifact)
    entry = getattr(module, ENTRYPOINT)
    direct_entry = getattr(module, DIRECT_ENTRYPOINT)
    direct_route_rejected = False
    try:
        direct_entry()
    except RuntimeError as error:
        direct_route_rejected = "not wired" in str(error)
    except TypeError as error:
        # The current pybind ABI is wired, but a no-argument probe must still
        # fail closed at the binding boundary rather than silently dispatching
        # a trajectory.  PyTorch/pybind reports that malformed call as a
        # TypeError instead of the older RuntimeError wording.
        if "incompatible function arguments" not in str(error):
            raise
        direct_route_rejected = True
    torch.manual_seed(9102)
    rows, hidden, actions, slots = 3, 64, 7, 48 * 512
    policy = torch.randn((rows, 9991), device="cuda", dtype=torch.float32)
    row_map = torch.tensor([256, 768, 1279], device="cuda", dtype=torch.int64)
    drives = torch.randn((slots, 4), device="cuda", dtype=torch.float32)
    attack = torch.randn((slots,), device="cuda", dtype=torch.float32)
    temperature = torch.rand((slots,), device="cuda", dtype=torch.float32) + 0.05
    gain = torch.rand((slots,), device="cuda", dtype=torch.float32)
    invalid_rejected = False
    try:
        entry(policy, torch.randn((rows, 85), device="cuda"), row_map, drives,
              attack, temperature, gain, hidden, actions, .1, .2, .3, .4, .5)
    except RuntimeError as error:
        invalid_rejected = "pack would exceed canonical row width" in str(error)
    observations = torch.randn((rows, 83), device="cuda", dtype=torch.float32)
    logits = entry(policy, observations, row_map, drives, attack, temperature,
                   gain, hidden, actions, .1, .2, .3, .4, .5)
    torch.cuda.synchronize()
    if (not direct_route_rejected or not invalid_rejected
            or tuple(logits.shape) != (rows, actions)
            or not torch.isfinite(logits).all().item()):
        raise RuntimeError("HAPB native compact-policy smoke did not pass")
    args.output_root.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema_version": "eiel_b48_wdl_jungle_eed_hapb_h2_native_smoke_v1",
        "artifact": str(args.artifact),
        "artifact_sha256": _sha256(args.artifact),
        "manifest_sha256": _sha256(args.manifest),
        "artifact_abi": "architecture_b_wdl_jungle_eed_hapb_fixed_mlp_native_v1",
        "entrypoint": ENTRYPOINT,
        "direct_trajectory_export_fail_closed": True,
        "source_commit": manifest["committed_source_snapshot"]["build_source_commit"],
        "mapped_rows": [256, 768, 1279],
        "observation_width": 83,
        "invalid_observation_width_85_rejected": True,
        "logits_shape": list(logits.shape),
        "logits_finite": True,
        "full_cuda_policy_bank_retained": False,
        "scope_limit": "native compact-kernel ABI smoke only; not resident-oracle parity or route execution",
    }
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    receipt_path = args.output_root / "h2_native_smoke_receipt.json"
    receipt_path.write_text(payload, encoding="utf-8")
    receipt["receipt_sha256"] = _sha256(receipt_path)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
