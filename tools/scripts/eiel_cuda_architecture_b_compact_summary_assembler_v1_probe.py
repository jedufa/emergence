#!/usr/bin/env python3
"""Build and probe the standalone CUDA compact-summary ABI-v2 assembler."""

from __future__ import annotations

import argparse
import json
import math
import os
import runpy
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import numpy as np
import torch

ROOT: Final[Path] = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "engine"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from engine.native_runtime import compact_summary_cpu as cpu
from engine.native_runtime import compact_summary_planes as host_abi
from engine.native_runtime import compact_summary_projection as projection
from engine.perf import cuda_architecture_b_compact_summary as torch_abi


CUDA_SOURCE: Final[Path] = (
    ROOT / "engine/perf/cuda_architecture_b_compact_summary_assembler_v1.cu"
)
BINDING_SOURCE: Final[Path] = (
    ROOT
    / "tests/perf/"
    "test_eiel_cuda_architecture_b_compact_summary_assembler_v1_binding.cpp"
)
FIXTURE_SOURCE: Final[Path] = ROOT / "tests/test_eiel_compact_summary_cpu.py"
MODULE_NAME: Final[str] = "eiel_cuda_compact_summary_assembler_v1_probe"
PLANE_NAMES: Final[tuple[str, ...]] = torch_abi.PLANE_NAMES
_MODULE_CACHE: dict[Path, ModuleType] = {}


def build_extension(build_directory: Path, *, verbose: bool = False) -> ModuleType:
    """Compile one sm_75 test-boundary extension without changing the engine."""

    if not torch.cuda.is_available():
        raise RuntimeError("an available CUDA device is required")
    build_directory = build_directory.expanduser().resolve()
    cached = _MODULE_CACHE.get(build_directory)
    if cached is not None:
        return cached
    build_directory.mkdir(parents=True, exist_ok=True)
    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    previous_jobs = os.environ.get("MAX_JOBS")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
    os.environ["MAX_JOBS"] = "1"
    try:
        from torch.utils.cpp_extension import load

        module = load(
            name=MODULE_NAME,
            sources=[str(BINDING_SOURCE), str(CUDA_SOURCE)],
            extra_include_paths=[str(ROOT / "engine/perf")],
            extra_cflags=["-O2"],
            extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v"],
            build_directory=str(build_directory),
            verbose=verbose,
            with_cuda=True,
            is_python_module=True,
        )
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch
        if previous_jobs is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = previous_jobs
    _MODULE_CACHE[build_directory] = module
    return module


def _torch_source_planes(
    source: host_abi.CompactSummaryPlanes,
    *,
    device: torch.device,
) -> list[torch.Tensor]:
    values = source.as_mapping()
    return [
        torch.from_numpy(values[name][0].copy()).to(device=device)
        for name in PLANE_NAMES[1:]
    ]


def _scratch_row_views(
    scratch: torch_abi.CompactSummaryTensors,
) -> list[torch.Tensor]:
    return [scratch.as_mapping()[name][0] for name in PLANE_NAMES]


def assemble_host_planes(
    module: ModuleType,
    source: host_abi.CompactSummaryPlanes,
    *,
    prefill_scratch: bool = False,
) -> tuple[host_abi.CompactSummaryPlanes, np.ndarray, np.ndarray]:
    """Assemble source primitive rows and copy once at the test boundary."""

    generations, cells = host_abi.validate_compact_summary_plane_layout(source)
    if generations != 1:
        raise ValueError("the standalone probe expects one source generation")
    device = torch.device("cuda", torch.cuda.current_device())
    source_tensors = _torch_source_planes(source, device=device)
    scratch = torch_abi.allocate_compact_summary_tensors(
        1, cells, device=device
    )
    if prefill_scratch:
        for tensor in scratch.as_mapping().values():
            tensor.fill_(1)
    ready = torch.ones((cells,), dtype=torch.uint8, device=device)
    status = torch.full((cells,), -1, dtype=torch.int32, device=device)
    module.assemble_into(
        source_tensors,
        _scratch_row_views(scratch),
        ready,
        status,
    )
    actual = torch_abi.to_numpy_compact_summary_planes(scratch)
    return (
        actual,
        ready.to(device="cpu", copy=True).numpy(),
        status.to(device="cpu", copy=True).numpy(),
    )


def assemble_canonical_row(
    module: ModuleType,
    row: Mapping[str, Any],
) -> tuple[host_abi.CompactSummaryPlanes, dict[str, Any]]:
    expected = host_abi.allocate_compact_summary_planes(1, 1)
    projection.encode_compact_summary_row(row, expected, generation=0, cell=0)
    actual, ready, status = assemble_host_planes(module, expected)
    projected = projection.project_compact_summary_row(
        actual, generation=0, cell=0
    )
    exact_planes = all(
        np.array_equal(
            expected.as_mapping()[name],
            actual.as_mapping()[name],
            equal_nan=True,
        )
        for name in PLANE_NAMES
    )
    return actual, {
        "exact_planes": exact_planes,
        "exact_row": projected == row,
        "canonical_bytes_exact": (
            projection.canonical_compact_summary_bytes(projected)
            == projection.canonical_compact_summary_bytes(row)
        ),
        "scratch_ready": int(ready[0]),
        "assemble_status": int(status[0]),
    }


def _fixture_source(*, world: str) -> dict[str, Any]:
    namespace = runpy.run_path(str(FIXTURE_SOURCE))
    source_factory = namespace.get("_source")
    if not callable(source_factory):
        raise RuntimeError("canonical CPU fixture factory is unavailable")
    return source_factory(world=world)


def run_probe(module: ModuleType) -> dict[str, Any]:
    """Exercise exact Flat, Patch, signed-zero, UTF-8, and invalid rows."""

    cases: dict[str, Any] = {}
    for world in ("flat", "patch"):
        row = cpu.build_compact_summary_row(_fixture_source(world=world))
        _, cases[world] = assemble_canonical_row(module, row)

    signed_source = _fixture_source(world="flat")
    signed_source["pre_turnover"]["generation_metrics"]["peak_injury"] = -0.0
    signed_row = cpu.build_compact_summary_row(signed_source)
    signed_planes, signed_receipt = assemble_canonical_row(module, signed_row)
    signed_value = float(
        signed_planes.float_values[
            0, 0, host_abi.FLOAT_FIELD_INDEX["generation_metrics.peak_injury"]
        ]
    )
    signed_receipt["negative_zero_preserved"] = (
        math.copysign(1.0, signed_value) == -1.0
    )
    cases["signed_zero"] = signed_receipt

    utf8_source = _fixture_source(world="flat")
    utf8_source["lineage"]["dominant_lineage_id"] = "lineage-\u03a9"
    utf8_row = cpu.build_compact_summary_row(utf8_source)
    _, cases["strict_utf8"] = assemble_canonical_row(module, utf8_row)

    invalid = host_abi.allocate_compact_summary_planes(1, 1)
    projection.encode_compact_summary_row(
        cpu.build_compact_summary_row(_fixture_source(world="flat")),
        invalid,
        generation=0,
        cell=0,
    )
    invalid.status_codes[
        0, 0, host_abi.STATUS_FIELD_INDEX["world"]
    ] = 99
    invalid.validity.fill(0)
    rejected, ready, status = assemble_host_planes(
        module, invalid, prefill_scratch=True
    )
    invalid_noop = all(
        not np.any(array.view(np.uint8))
        for array in rejected.as_mapping().values()
    )
    cases["invalid_status"] = {
        "assemble_status": int(status[0]),
        "scratch_ready": int(ready[0]),
        "bit_zero_noop": invalid_noop,
    }
    all_exact = all(
        receipt.get("exact_planes", True)
        and receipt.get("exact_row", True)
        and receipt.get("canonical_bytes_exact", True)
        and receipt.get("scratch_ready", 0) == (0 if name == "invalid_status" else 1)
        and receipt.get("assemble_status", -1) == (5 if name == "invalid_status" else 0)
        for name, receipt in cases.items()
    )
    all_exact = all_exact and cases["signed_zero"]["negative_zero_preserved"]
    all_exact = all_exact and cases["invalid_status"]["bit_zero_noop"]
    return {
        "schema_version": "eiel_cuda_compact_summary_assembler_v1_probe_v1",
        "all_exact": all_exact,
        "logical_bytes_per_generation_cell": (
            host_abi.LOGICAL_BYTES_PER_GENERATION_CELL
        ),
        "cases": cases,
        "resource_receipt": dict(module.resource_receipt()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-directory",
        type=Path,
        default=Path("/tmp/eiel_cuda_compact_summary_assembler_v1"),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    module = build_extension(args.build_directory, verbose=args.verbose)
    receipt = run_probe(module)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["all_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
