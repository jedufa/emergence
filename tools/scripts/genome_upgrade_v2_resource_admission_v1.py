#!/usr/bin/env python3
"""Measure Genome V2 resident-array resources at the package envelope."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any

import torch

from engine.evolution.genome_upgrade_v2 import CONTEXT_COUNT
from engine.evolution.genome_upgrade_v2_package_v1 import (
    capacity_layout_contract_v1,
    validate_capacity_layout_admission_v1,
)
from engine.perf.cuda_genome_upgrade_v2_state_bank import (
    CudaGenomeUpgradeV2StateBank,
    restore_cuda_genome_upgrade_v2_state_bank,
)


SCHEMA_VERSION = "genome_v2_resource_admission_receipt_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _host_memory() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            name, raw = line.split(":", 1)
            values[name] = int(raw.strip().split()[0]) * 1024
    return {
        "rss_bytes": values.get("VmRSS", 0),
        "high_water_bytes": max(
            values.get("VmHWM", 0),
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        ),
    }


def _cuda_memory(device: torch.device) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "device_free_bytes": int(free),
        "device_total_bytes": int(total),
    }


def _all_exact(
    first: CudaGenomeUpgradeV2StateBank,
    second: CudaGenomeUpgradeV2StateBank,
) -> bool:
    return all(
        torch.equal(tensor, second.named_tensors()[name])
        for name, tensor in first.named_tensors().items()
    )


def execute(args: argparse.Namespace) -> int:
    artifact = args.artifact.resolve()
    manifest_path = artifact.with_suffix(artifact.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("artifact_sha256") != _sha256(artifact)
        or manifest.get("cuda_arch") != "sm_75"
        or manifest.get("sass_only") is not True
        or manifest.get("runtime_jit_compilation") is not False
    ):
        raise RuntimeError("resource admission artifact is invalid or stale")
    dimensions = validate_capacity_layout_admission_v1(
        cells=args.cells,
        capacity=args.capacity,
        total_ticks=args.total_ticks,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Genome V2 resource admission requires CUDA")
    device = torch.device("cuda")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    host_before = _host_memory()
    cuda_before = _cuda_memory(device)
    started_ns = time.perf_counter_ns()
    bank = CudaGenomeUpgradeV2StateBank.allocate(
        cells=args.cells, capacity=args.capacity, device=device
    )
    context = torch.zeros(
        (args.cells, args.capacity, CONTEXT_COUNT),
        dtype=torch.int64,
        device=device,
    )
    torch.cuda.synchronize(device)
    cuda_persistent = _cuda_memory(device)
    pointers = {
        name: tensor.data_ptr() for name, tensor in bank.named_tensors().items()
    }
    steady_allocations: list[int] = []
    for cycle in range(args.cycles):
        context[..., cycle % CONTEXT_COUNT].fill_(cycle % 2 * 1024)
        bank.express(context)
        torch.cuda.synchronize(device)
        if pointers != {
            name: tensor.data_ptr()
            for name, tensor in bank.named_tensors().items()
        }:
            raise RuntimeError("resident Genome V2 array pointer changed")
        steady_allocations.append(int(torch.cuda.memory_allocated(device)))
    cuda_after_cycles = _cuda_memory(device)
    checkpoint_started_ns = time.perf_counter_ns()
    checkpoint = bank.checkpoint_projection()
    checkpoint_projection_ns = time.perf_counter_ns() - checkpoint_started_ns
    host_after_projection = _host_memory()
    checkpoint_path = args.output_dir.resolve() / "genome_v2_resource_probe.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    write_started_ns = time.perf_counter_ns()
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)
    checkpoint_write_ns = time.perf_counter_ns() - write_started_ns
    read_started_ns = time.perf_counter_ns()
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored = restore_cuda_genome_upgrade_v2_state_bank(
        loaded, device=device
    )
    torch.cuda.synchronize(device)
    checkpoint_read_restore_ns = time.perf_counter_ns() - read_started_ns
    restore_exact = _all_exact(bank, restored)
    cuda_after_restore = _cuda_memory(device)
    host_after_restore = _host_memory()

    negative_rows = []
    for name, values in (
        ("cells_overflow", {"cells": args.cells + 1}),
        ("capacity_overflow", {"capacity": args.capacity + 1}),
        ("total_ticks_zero", {"total_ticks": 0}),
    ):
        candidate: dict[str, Any] = {
            "cells": args.cells,
            "capacity": args.capacity,
            "total_ticks": args.total_ticks,
            **values,
        }
        try:
            validate_capacity_layout_admission_v1(**candidate)
            rejected = False
        except ValueError:
            rejected = True
        negative_rows.append({"case_id": name, "rejected": rejected})

    layout = capacity_layout_contract_v1()
    exact_resident_bytes = bank.storage_bytes()
    expected_resident_bytes = (
        layout["bytes_per_cell_slot"] * args.cells * args.capacity
    )
    context_bytes = context.numel() * context.element_size()
    persistent_delta = (
        cuda_persistent["allocated_bytes"] - cuda_before["allocated_bytes"]
    )
    peak_delta = (
        cuda_after_restore["peak_allocated_bytes"]
        - cuda_before["allocated_bytes"]
    )
    turnover_shadow_headroom = exact_resident_bytes
    observability_headroom = args.cells * 64 * 1024
    required_free_headroom = (
        turnover_shadow_headroom + observability_headroom + context_bytes
    )
    allocation_growth = (
        max(steady_allocations) - min(steady_allocations)
        if steady_allocations
        else 0
    )
    passed = (
        exact_resident_bytes == expected_resident_bytes
        and persistent_delta >= exact_resident_bytes + context_bytes
        and allocation_growth == 0
        and restore_exact
        and cuda_after_restore["device_free_bytes"] > required_free_headroom
        and all(row["rejected"] for row in negative_rows)
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "qualification_status": "PASS" if passed else "FAIL",
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "global_genome_profile_id": "genome_upgrade_v2",
        "artifact_sha256": _sha256(artifact),
        "manifest_sha256": _sha256(manifest_path),
        "dimensions": dimensions,
        "launch_geometry": {
            "sanitizer_grid_blocks": args.cells,
            "sanitizer_threads_per_block": args.capacity,
            "resident_layout": layout["resident_layout"],
        },
        "device": {
            "name": torch.cuda.get_device_name(device),
            "compute_capability": list(
                torch.cuda.get_device_capability(device)
            ),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        },
        "resident_array_measurement": {
            "exact_resident_bytes": exact_resident_bytes,
            "expected_resident_bytes": expected_resident_bytes,
            "context_staging_bytes": context_bytes,
            "persistent_allocated_delta_bytes": persistent_delta,
            "peak_allocated_delta_bytes": peak_delta,
            "steady_cycle_allocated_bytes": steady_allocations,
            "hidden_allocation_growth_bytes": allocation_growth,
            "pointer_identity_stable": True,
        },
        "headroom": {
            "ordinary_birth_staging_bytes": exact_resident_bytes // args.capacity,
            "turnover_shadow_bytes": turnover_shadow_headroom,
            "observability_projection_bytes": observability_headroom,
            "required_free_headroom_bytes": required_free_headroom,
            "observed_free_after_restore_bytes": cuda_after_restore[
                "device_free_bytes"
            ],
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "projection_ns": checkpoint_projection_ns,
            "write_ns": checkpoint_write_ns,
            "read_restore_ns": checkpoint_read_restore_ns,
            "fresh_owner_arrays_exact": restore_exact,
            "atomic_temporary_publish": True,
        },
        "host_staging": {
            "before": host_before,
            "after_projection": host_after_projection,
            "after_restore": host_after_restore,
        },
        "cuda_snapshots": {
            "before": cuda_before,
            "persistent": cuda_persistent,
            "after_cycles": cuda_after_cycles,
            "after_restore": cuda_after_restore,
        },
        "outside_envelope_rejection": negative_rows,
        "elapsed_ns": time.perf_counter_ns() - started_ns,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "genome_v2_resource_admission_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=72)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--total-ticks", type=int, required=True)
    parser.add_argument("--cycles", type=int, default=4)
    return execute(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
