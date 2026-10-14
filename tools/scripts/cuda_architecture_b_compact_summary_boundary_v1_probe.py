#!/usr/bin/env python3
"""Build and qualify the standalone compact post-turnover boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import runpy
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Mapping, Sequence

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


ENGINE_PERF: Final[Path] = ROOT / "engine/perf"
BOUNDARY_SOURCE: Final[Path] = (
    ENGINE_PERF / "cuda_architecture_b_compact_summary_boundary_v1.cu"
)
BOUNDARY_HEADER: Final[Path] = (
    ENGINE_PERF / "cuda_architecture_b_compact_summary_boundary_v1.cuh"
)
BINDING_SOURCE: Final[Path] = (
    ROOT
    / "tests/perf/cuda_architecture_b_compact_summary_boundary_v1_binding.cpp"
)
DEPENDENCY_SOURCES: Final[tuple[Path, ...]] = (
    ENGINE_PERF / "cuda_architecture_b_compact_summary_reductions_v1.cu",
    ENGINE_PERF / "cuda_architecture_b_compact_summary_checkpoint_hash_v1.cu",
    ENGINE_PERF / "cuda_architecture_b_compact_summary_pipeline_v1.cu",
    ENGINE_PERF / "cuda_architecture_b_compact_summary_assembler_v1.cu",
    ENGINE_PERF / "cuda_architecture_b_compact_summary_finalizer_v2.cu",
)
PIPELINE_PROBE: Final[Path] = (
    ROOT
    / "tools/scripts/eiel_cuda_architecture_b_compact_summary_pipeline_v1_probe.py"
)
FIXTURE_SOURCE: Final[Path] = ROOT / "tests/test_eiel_compact_summary_cpu.py"
MODULE_NAME: Final[str] = "eiel_cuda_compact_summary_boundary_v1"
PLANE_NAMES: Final[tuple[str, ...]] = torch_abi.PLANE_NAMES
RESULT_BYTES: Final[int] = 208

TYPED_FIELDS: Final[tuple[str, ...]] = (
    "event_type",
    "generation",
    "tick",
    "sequence",
    "actor_agent_id",
    "target_agent_id",
    "actor_slot",
    "target_slot",
    "parent_agent_id",
    "child_agent_id",
    "lineage_founder_id",
    "lineage_depth",
    "subtype_code",
    "reason_code",
    "success",
    "value",
)
TYPED_DTYPES: Final[tuple[torch.dtype, ...]] = (
    torch.int32,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int32,
    torch.int32,
    torch.int32,
    torch.int8,
    torch.float32,
)
CURSOR_FIELDS: Final[tuple[str, ...]] = (
    "active_event_count",
    "typed_active_count",
    "tape_active_count",
    "event_cursor",
    "cursor_start",
    "cursor_pre_turnover",
    "cursor_end",
    "authoritative_cursor",
    "event_row_count",
)
IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "active_agent_count",
    "population_next_id",
    "free_slot_count",
)


def _load_python(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_extension(build_directory: Path, *, verbose: bool = False) -> ModuleType:
    """Build an sm75-only qualification extension from standalone modules."""

    if not torch.cuda.is_available():
        raise RuntimeError("an available CUDA device is required")
    build_directory = build_directory.expanduser().resolve()
    build_directory.mkdir(parents=True, exist_ok=True)
    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    previous_jobs = os.environ.get("MAX_JOBS")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
    os.environ["MAX_JOBS"] = "1"
    try:
        from torch.utils.cpp_extension import load

        return load(
            name=MODULE_NAME,
            sources=[
                str(BINDING_SOURCE),
                str(BOUNDARY_SOURCE),
                *(str(path) for path in DEPENDENCY_SOURCES),
            ],
            extra_include_paths=[str(ENGINE_PERF)],
            extra_cflags=["-O2", "-ffp-contract=off"],
            extra_cuda_cflags=[
                "-O2",
                "--fmad=false",
                "--prec-div=true",
                "--prec-sqrt=true",
                "-lineinfo",
                "-Xptxas=-v",
            ],
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


def _fixture_source(*, world: str = "flat") -> dict[str, Any]:
    namespace = runpy.run_path(str(FIXTURE_SOURCE))
    factory = namespace.get("_source")
    if not callable(factory):
        raise RuntimeError("compact-summary fixture factory is unavailable")
    return factory(world=world)


def _host_matrix(
    rows: Sequence[Sequence[Mapping[str, Any]]],
) -> host_abi.CompactSummaryPlanes:
    generations = len(rows)
    cells = len(rows[0])
    planes = host_abi.allocate_compact_summary_planes(generations, cells)
    for generation, generation_rows in enumerate(rows):
        if len(generation_rows) != cells:
            raise ValueError("fixture must be rectangular")
        for cell, row in enumerate(generation_rows):
            projection.encode_compact_summary_row(
                row, planes, generation=generation, cell=cell
            )
    return planes


def _matrix_fixture() -> list[list[dict[str, Any]]]:
    pipeline_probe = _load_python(PIPELINE_PROBE, "boundary_pipeline_fixture")
    return pipeline_probe._matrix_fixture()


def _sticky_matrix_fixture() -> list[list[dict[str, Any]]]:
    normal = cpu.build_compact_summary_row(_fixture_source())
    failed_source = _fixture_source()
    failed_source["settled"]["terminal"].update(
        status="failed",
        sticky_failure=True,
        failure_generation=0,
        failure_tick=17,
        failure_stage="generation",
    )
    failed = cpu.build_compact_summary_row(failed_source)
    return [
        [deepcopy(normal), failed, deepcopy(normal)],
        [deepcopy(normal), deepcopy(normal), deepcopy(normal)],
    ]


def _source_tensors(
    planes: host_abi.CompactSummaryPlanes,
    generation: int,
    device: torch.device,
) -> list[torch.Tensor]:
    values = planes.as_mapping()
    return [
        torch.from_numpy(values[name][generation].copy()).to(device=device)
        for name in PLANE_NAMES[1:]
    ]


def _scratch_views(scratch: torch_abi.CompactSummaryTensors) -> list[torch.Tensor]:
    return [scratch.as_mapping()[name][0] for name in PLANE_NAMES]


def _output_views(output: torch_abi.CompactSummaryTensors) -> list[torch.Tensor]:
    return [output.as_mapping()[name] for name in PLANE_NAMES]


def _event(kind: int, *, generation: int, cell: int) -> dict[str, int | float]:
    actor = 10 + cell
    if kind == 1:
        target = 20 + cell
        return {
            "event_type": 1,
            "generation": generation,
            "tick": generation + 1,
            "sequence": 0,
            "actor_agent_id": actor,
            "target_agent_id": target,
            "actor_slot": 0,
            "target_slot": 1,
            "parent_agent_id": actor,
            "child_agent_id": target,
            "lineage_founder_id": cell,
            "lineage_depth": 1,
            "subtype_code": 1,
            "reason_code": -1,
            "success": 1,
            "value": 0.25,
        }
    reason = 1 + (cell % 2)
    return {
        "event_type": 4,
        "generation": generation,
        "tick": generation + 1,
        "sequence": 0,
        "actor_agent_id": actor,
        "target_agent_id": -1,
        "actor_slot": 0,
        "target_slot": -1,
        "parent_agent_id": -1,
        "child_agent_id": -1,
        "lineage_founder_id": cell,
        "lineage_depth": 1,
        "subtype_code": 4,
        "reason_code": reason,
        "success": 1,
        "value": float(reason),
    }


def _reduction_inputs(
    generation: int,
    device: torch.device,
    *,
    malformed_cursor_cell: int | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
    cells = 3
    capacity = 2
    rows = [_event(1 if generation == 0 else 4, generation=generation, cell=cell) for cell in range(cells)]
    typed: list[torch.Tensor] = []
    for field, dtype in zip(TYPED_FIELDS, TYPED_DTYPES, strict=True):
        values = torch.zeros((cells, capacity), dtype=dtype)
        for cell, row in enumerate(rows):
            values[cell, 0] = row[field]
        typed.append(values.to(device=device))
    cursors = {field: [1, 1, 1] for field in CURSOR_FIELDS}
    cursors["cursor_start"] = [0, 0, 0]
    if malformed_cursor_cell is not None:
        cursors["typed_active_count"][malformed_cursor_cell] = 0
    cursor_tensors = [
        torch.tensor(cursors[field], dtype=torch.int64, device=device)
        for field in CURSOR_FIELDS
    ]
    active_rows: list[list[int]] = []
    counts = [1, 2, 3]
    next_ids: list[int] = []
    for cell, count in enumerate(counts):
        start = generation * 20 + cell * 4 + 1
        values = list(range(start, start + count))
        next_ids.append(values[-1] + 1)
        active_rows.append(values + [-1] * (4 - count))
    active_ids = torch.tensor(active_rows, dtype=torch.int64, device=device)
    identity = [
        torch.tensor(counts, dtype=torch.int64, device=device),
        torch.tensor(next_ids, dtype=torch.int64, device=device),
        torch.tensor([4 - count for count in counts], dtype=torch.int64, device=device),
    ]
    return typed, cursor_tensors, active_ids, identity


def _checkpoint_inputs(
    generation: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[int], list[int], list[torch.Tensor], list[torch.Tensor]]:
    values = torch.arange(12, dtype=torch.int32, device=device).reshape(3, 4)
    values = (values + generation * 100).contiguous()
    placeholder = torch.empty((0,), dtype=torch.uint8, device=device)
    return [values], [0], [4], [placeholder], [placeholder]


def _allocate_runtime(
    generations: int,
    cells: int,
    checkpoint_fields: int,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "scratch": torch_abi.allocate_compact_summary_tensors(1, cells, device=device),
        "output": torch_abi.allocate_compact_summary_tensors(generations, cells, device=device),
        "scratch_ready": torch.zeros((cells,), dtype=torch.uint8, device=device),
        "reduction": torch.zeros((generations, cells, RESULT_BYTES), dtype=torch.uint8, device=device),
        "raw": torch.zeros((generations, cells, checkpoint_fields, 32), dtype=torch.uint8, device=device),
        "canonical": torch.zeros((generations, cells, checkpoint_fields, 32), dtype=torch.uint8, device=device),
        "hash_status": torch.zeros((generations, cells, checkpoint_fields), dtype=torch.int32, device=device),
        "hash_valid": torch.zeros((generations, cells, checkpoint_fields), dtype=torch.uint8, device=device),
        "assemble_status": torch.zeros((cells,), dtype=torch.int32, device=device),
        "producer_status": torch.zeros((cells,), dtype=torch.int32, device=device),
    }


def _invoke(
    module: ModuleType,
    source: host_abi.CompactSummaryPlanes,
    runtime: Mapping[str, Any],
    generation: int,
    *,
    host_digest_mask: int,
    checkpoint_count: int = 1,
    malformed_cursor_cell: int | None = None,
    checkpoint_fault: int = 0,
) -> dict[str, Any]:
    device = runtime["producer_status"].device
    typed, cursors, active_ids, identity = _reduction_inputs(
        generation, device, malformed_cursor_cell=malformed_cursor_cell
    )
    if checkpoint_count:
        fields, modes, capacities, alive, active = _checkpoint_inputs(generation, device)
    else:
        fields, modes, capacities, alive, active = [], [], [], [], []
    return dict(
        module.run_into(
            typed,
            cursors,
            active_ids,
            identity,
            4,
            fields,
            modes,
            capacities,
            alive,
            active,
            _source_tensors(source, generation, device),
            _scratch_views(runtime["scratch"]),
            runtime["scratch_ready"],
            _output_views(runtime["output"]),
            runtime["reduction"],
            runtime["raw"],
            runtime["canonical"],
            runtime["hash_status"],
            runtime["hash_valid"],
            runtime["assemble_status"],
            runtime["producer_status"],
            generation,
            host_digest_mask,
            checkpoint_fault,
        )
    )


def _reduction_summary(row: np.ndarray) -> dict[str, Any]:
    data = row.tobytes()
    return {
        "status": int(np.frombuffer(data, dtype=np.int32, count=1, offset=0)[0]),
        "valid": int(np.frombuffer(data, dtype=np.uint8, count=1, offset=4)[0]),
        "generation_counts": tuple(
            int(value)
            for value in np.frombuffer(data, dtype=np.int64, count=4, offset=24)
        ),
    }


def qualify(module: ModuleType) -> dict[str, Any]:
    device = torch.device("cuda", torch.cuda.current_device())
    rows = _matrix_fixture()
    expected = _host_matrix(rows)
    runtime = _allocate_runtime(2, 3, 1, device)
    full_mask = int(module.required_host_digest_mask)
    receipts = [
        _invoke(module, expected, runtime, generation, host_digest_mask=full_mask)
        for generation in range(2)
    ]
    actual = torch_abi.to_numpy_compact_summary_planes(runtime["output"])
    exact_planes = all(
        actual.as_mapping()[name].tobytes(order="C")
        == expected.as_mapping()[name].tobytes(order="C")
        for name in PLANE_NAMES
    )
    reduction = runtime["reduction"].cpu().numpy()
    reduction_rows = [
        [_reduction_summary(reduction[generation, cell]) for cell in range(3)]
        for generation in range(2)
    ]
    checkpoint_exact = True
    raw = runtime["raw"].cpu().numpy()
    canonical = runtime["canonical"].cpu().numpy()
    for generation in range(2):
        field = _checkpoint_inputs(generation, torch.device("cpu"))[0][0].numpy()
        for cell in range(3):
            expected_digest = hashlib.sha256(field[cell].tobytes(order="C")).digest()
            checkpoint_exact &= bytes(raw[generation, cell, 0]) == expected_digest
            checkpoint_exact &= bytes(canonical[generation, cell, 0]) == expected_digest

    missing_runtime = _allocate_runtime(2, 3, 0, device)
    missing_receipt = _invoke(
        module,
        expected,
        missing_runtime,
        0,
        host_digest_mask=0,
        checkpoint_count=0,
    )
    missing_validity = missing_runtime["output"].validity.cpu().tolist()
    missing_producer = missing_runtime["producer_status"].cpu().tolist()

    malformed_reduction = _allocate_runtime(2, 3, 1, device)
    malformed_reduction_receipt = _invoke(
        module,
        expected,
        malformed_reduction,
        0,
        host_digest_mask=full_mask,
        malformed_cursor_cell=1,
    )
    malformed_hash = _allocate_runtime(2, 3, 1, device)
    malformed_hash_receipt = _invoke(
        module,
        expected,
        malformed_hash,
        0,
        host_digest_mask=full_mask,
        checkpoint_fault=1,
    )

    sticky_rows = _sticky_matrix_fixture()
    sticky_source = _host_matrix(sticky_rows)
    sticky_runtime = _allocate_runtime(2, 3, 1, device)
    for generation in range(2):
        _invoke(
            module,
            sticky_source,
            sticky_runtime,
            generation,
            host_digest_mask=full_mask,
        )
    sticky_actual = torch_abi.to_numpy_compact_summary_planes(sticky_runtime["output"])

    resources = dict(module.resource_receipt())
    return {
        "ok": all(
            (
                exact_planes,
                checkpoint_exact,
                reduction_rows[0][0]["generation_counts"] == (1, 0, 0, 0),
                reduction_rows[1][0]["generation_counts"] == (0, 0, 0, 1),
                all(receipt["status"] == 0 for receipt in receipts),
            )
        ),
        "g2_b3": {
            "exact_planes": exact_planes,
            "checkpoint_hashes_exact": checkpoint_exact,
            "reduction_rows": reduction_rows,
            "receipts": receipts,
        },
        "host_digest_finalization_required": {
            "receipt": missing_receipt,
            "producer_status": missing_producer,
            "output_validity": missing_validity,
        },
        "malformed_fail_closed": {
            "reduction_receipt": malformed_reduction_receipt,
            "reduction_producer_status": malformed_reduction["producer_status"].cpu().tolist(),
            "reduction_output_validity": malformed_reduction["output"].validity.cpu().tolist(),
            "hash_receipt": malformed_hash_receipt,
            "hash_producer_status": malformed_hash["producer_status"].cpu().tolist(),
            "hash_output_validity": malformed_hash["output"].validity.cpu().tolist(),
        },
        "sticky_tail": {
            "producer_status": sticky_runtime["producer_status"].cpu().tolist(),
            "generation_zero_validity": sticky_actual.validity[0].tolist(),
            "generation_one_validity": sticky_actual.validity[1].tolist(),
        },
        "resources": resources,
    }


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    module = build_extension(args.build_directory, verbose=args.verbose)
    receipt = qualify(module)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
