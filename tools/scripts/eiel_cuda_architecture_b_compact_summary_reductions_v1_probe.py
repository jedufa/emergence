#!/usr/bin/env python3
from __future__ import annotations

"""Build and exercise the standalone compact lifecycle/identity reductions."""

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import numpy as np
import torch


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
HEADER: Final[Path] = (
    ROOT / "engine/perf/cuda_architecture_b_compact_summary_reductions_v1.cuh"
)
CUDA_SOURCE: Final[Path] = (
    ROOT / "engine/perf/cuda_architecture_b_compact_summary_reductions_v1.cu"
)
BINDING_SOURCE: Final[Path] = (
    ROOT
    / "tests/perf/test_eiel_cuda_architecture_b_compact_summary_reductions_v1_binding.cpp"
)
MODULE_NAME: Final[str] = "eiel_cuda_architecture_b_compact_summary_reductions_v1"
CONTRACT_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_compact_summary_reductions_v1"
)
RESULT_BYTES: Final[int] = 208

EVENT_FIELDS: Final[tuple[str, ...]] = (
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
EVENT_DTYPES: Final[tuple[torch.dtype, ...]] = (
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
SOURCE_INVENTORY: Final[tuple[Path, ...]] = (
    HEADER,
    CUDA_SOURCE,
    BINDING_SOURCE,
    Path(__file__).resolve(),
)


class ReductionResultV1(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("valid", ctypes.c_uint8),
        ("active_id_bounds_defined", ctypes.c_uint8),
        ("reserved", ctypes.c_uint16),
        ("generation_event_count", ctypes.c_int64),
        ("turnover_event_count", ctypes.c_int64),
        ("generation_counts_by_type", ctypes.c_int64 * 4),
        ("turnover_counts_by_type", ctypes.c_int64 * 4),
        ("deaths_energy_depleted", ctypes.c_int64),
        ("deaths_health_depleted", ctypes.c_int64),
        ("cursor_start", ctypes.c_int64),
        ("cursor_pre_turnover", ctypes.c_int64),
        ("cursor_end", ctypes.c_int64),
        ("authoritative_cursor", ctypes.c_int64),
        ("event_row_count", ctypes.c_int64),
        ("active_event_count", ctypes.c_int64),
        ("typed_event_row_count", ctypes.c_int64),
        ("tape_event_row_count", ctypes.c_int64),
        ("population_next_id", ctypes.c_int64),
        ("active_agent_count", ctypes.c_int64),
        ("active_agent_id_min", ctypes.c_int64),
        ("active_agent_id_max", ctypes.c_int64),
        ("free_slot_count", ctypes.c_int64),
    ]


if ctypes.sizeof(ReductionResultV1) != RESULT_BYTES:
    raise RuntimeError("Python/CUDA compact reduction POD size disagreement")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_artifact(path: Path) -> ModuleType:
    import torch as _torch  # noqa: F401 - preloads libtorch dependencies.

    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load compact reduction artifact: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(output_dir: Path) -> dict[str, Any]:
    """Build one standalone sm_75 artifact outside measured execution."""

    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise RuntimeError("an available CUDA device is required")
    output_dir = output_dir.expanduser().resolve()
    build_dir = output_dir / "build"
    artifact = output_dir / f"{MODULE_NAME}.so"
    manifest_path = Path(f"{artifact}.manifest.json")
    if artifact.exists() or manifest_path.exists():
        raise FileExistsError("compact reduction build output already exists")
    output_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
    try:
        module = load(
            name=MODULE_NAME,
            sources=[str(BINDING_SOURCE), str(CUDA_SOURCE)],
            extra_include_paths=[str(ROOT / "engine/perf")],
            extra_cflags=["-O2", "-ffp-contract=off"],
            extra_cuda_cflags=[
                "-O2",
                "--fmad=false",
                "-lineinfo",
                "-Xptxas=-v",
            ],
            build_directory=str(build_dir),
            with_cuda=True,
            verbose=False,
        )
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch
    shutil.copy2(Path(module.__file__).resolve(), artifact)
    manifest = {
        "schema_version": "eiel_cuda_architecture_b_compact_summary_reductions_v1_artifact",
        "contract_version": CONTRACT_VERSION,
        "artifact": str(artifact),
        "artifact_sha256": _sha256(artifact),
        "cuda_arch": "sm_75",
        "result_bytes": RESULT_BYTES,
        "source_records": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in SOURCE_INVENTORY
        ],
        "resource_receipt": dict(module.resource_receipt()),
        "controller_integrated": False,
        "capability_promoted": False,
        "caller_preallocated_output": True,
        "kernel_host_synchronization": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _event_tensors(
    rows_by_cell: Sequence[Sequence[Mapping[str, Any]]],
    *,
    capacity: int,
    device: torch.device,
) -> list[torch.Tensor]:
    cells = len(rows_by_cell)
    if capacity < 0 or any(len(rows) > capacity for rows in rows_by_cell):
        raise ValueError("event rows exceed the requested capacity")
    tensors = [
        torch.zeros((cells, capacity), dtype=dtype, device=device)
        for dtype in EVENT_DTYPES
    ]
    for cell, rows in enumerate(rows_by_cell):
        for row_index, row in enumerate(rows):
            missing = sorted(set(EVENT_FIELDS) - set(row))
            extra = sorted(set(row) - set(EVENT_FIELDS))
            if missing or extra:
                raise ValueError(
                    f"typed event fields mismatch: missing={missing} extra={extra}"
                )
            for field_index, field in enumerate(EVENT_FIELDS):
                tensors[field_index][cell, row_index] = row[field]
    return tensors


def _decode_results(raw: torch.Tensor) -> list[dict[str, Any]]:
    bytes_by_cell = raw.detach().to("cpu").contiguous().numpy()
    decoded: list[dict[str, Any]] = []
    for row in bytes_by_cell:
        result = ReductionResultV1.from_buffer_copy(row.tobytes())
        decoded.append(
            {
                "status": int(result.status),
                "valid": bool(result.valid),
                "active_id_bounds_defined": bool(
                    result.active_id_bounds_defined
                ),
                "generation_event_count": int(result.generation_event_count),
                "turnover_event_count": int(result.turnover_event_count),
                "generation_counts_by_type": tuple(
                    int(value) for value in result.generation_counts_by_type
                ),
                "turnover_counts_by_type": tuple(
                    int(value) for value in result.turnover_counts_by_type
                ),
                "deaths_energy_depleted": int(result.deaths_energy_depleted),
                "deaths_health_depleted": int(result.deaths_health_depleted),
                "cursor_start": int(result.cursor_start),
                "cursor_pre_turnover": int(result.cursor_pre_turnover),
                "cursor_end": int(result.cursor_end),
                "authoritative_cursor": int(result.authoritative_cursor),
                "event_row_count": int(result.event_row_count),
                "active_event_count": int(result.active_event_count),
                "typed_event_row_count": int(result.typed_event_row_count),
                "tape_event_row_count": int(result.tape_event_row_count),
                "population_next_id": int(result.population_next_id),
                "active_agent_count": int(result.active_agent_count),
                "active_agent_id_min": int(result.active_agent_id_min),
                "active_agent_id_max": int(result.active_agent_id_max),
                "free_slot_count": int(result.free_slot_count),
            }
        )
    return decoded


def run_fixture(
    module: ModuleType,
    fixture: Mapping[str, Any],
    *,
    device: str | torch.device = "cuda",
) -> list[dict[str, Any]]:
    """Run a caller-preallocated fixture and synchronize only at probe output."""

    target = torch.device(device)
    rows_by_cell = fixture["events"]
    cells = len(rows_by_cell)
    event_capacity = int(fixture["event_capacity"])
    typed = _event_tensors(rows_by_cell, capacity=event_capacity, device=target)
    cursors = [
        torch.tensor(fixture["cursors"][field], dtype=torch.int64, device=target)
        for field in CURSOR_FIELDS
    ]
    active_ids = torch.tensor(
        fixture["active_agent_ids"], dtype=torch.int64, device=target
    ).contiguous()
    identity = [
        torch.tensor(fixture["identity"][field], dtype=torch.int64, device=target)
        for field in IDENTITY_FIELDS
    ]
    output = torch.zeros((cells, RESULT_BYTES), dtype=torch.uint8, device=target)
    module.reduce_into(
        typed,
        cursors,
        active_ids,
        identity,
        int(fixture["slot_capacity"]),
        output,
    )
    torch.cuda.synchronize(target)
    return _decode_results(output)


def cpu_oracle_cell(fixture: Mapping[str, Any], cell: int) -> dict[str, Any]:
    """Use the accepted lifecycle and identity helpers as the scalar oracle."""

    from engine.native_runtime import compact_summary_digests
    from engine.native_runtime import lifecycle_events

    cursor_values = {
        name: int(fixture["cursors"][name][cell]) for name in CURSOR_FIELDS
    }
    end = cursor_values["cursor_end"]
    if len(set(cursor_values[name] for name in CURSOR_FIELDS[:4])) != 1:
        raise ValueError("CPU oracle active event counts disagree")
    if any(cursor_values[name] != end for name in CURSOR_FIELDS[:4]):
        raise ValueError("CPU oracle active event count disagrees with cursor_end")
    if cursor_values["authoritative_cursor"] != end:
        raise ValueError("CPU oracle authoritative cursor disagrees")
    start = cursor_values["cursor_start"]
    pre = cursor_values["cursor_pre_turnover"]
    if not (0 <= start <= pre <= end):
        raise ValueError("CPU oracle cursor order is invalid")
    if cursor_values["event_row_count"] != end - start:
        raise ValueError("CPU oracle event row count disagrees")

    rows = fixture["events"][cell][:end]
    event_mapping = {
        field: torch.tensor(
            [row[field] for row in rows], dtype=EVENT_DTYPES[index]
        )
        for index, field in enumerate(EVENT_FIELDS)
    }
    typed_rows = lifecycle_events.typed_lifecycle_events_to_rows(event_mapping)
    lifecycle_events.committed_death_semantic_rows(event_mapping)
    generation_counts = [0, 0, 0, 0]
    turnover_counts = [0, 0, 0, 0]
    energy_deaths = 0
    health_deaths = 0
    for row_index, row in enumerate(typed_rows):
        if row_index < start:
            continue
        event_type = int(row["event_type"])
        counts = generation_counts if row_index < pre else turnover_counts
        counts[event_type - 1] += 1
        if row_index < pre and event_type == 4:
            if int(row["reason_code"]) == 1:
                energy_deaths += 1
            else:
                health_deaths += 1

    active_count = int(fixture["identity"]["active_agent_count"][cell])
    active_ids = np.asarray(
        fixture["active_agent_ids"][cell][:active_count], dtype=np.int64
    )
    compact_summary_digests.active_agent_ids_digest_sha256(active_ids)
    population_next_id = int(fixture["identity"]["population_next_id"][cell])
    free_slots = int(fixture["identity"]["free_slot_count"][cell])
    if active_count + free_slots != int(fixture["slot_capacity"]):
        raise ValueError("CPU oracle slot conservation disagrees")
    if active_count and int(active_ids[-1]) >= population_next_id:
        raise ValueError("CPU oracle population_next_id does not exceed active IDs")
    return {
        "status": 0,
        "valid": True,
        "active_id_bounds_defined": bool(active_count),
        "generation_event_count": pre - start,
        "turnover_event_count": end - pre,
        "generation_counts_by_type": tuple(generation_counts),
        "turnover_counts_by_type": tuple(turnover_counts),
        "deaths_energy_depleted": energy_deaths,
        "deaths_health_depleted": health_deaths,
        "cursor_start": start,
        "cursor_pre_turnover": pre,
        "cursor_end": end,
        "authoritative_cursor": end,
        "event_row_count": end - start,
        "active_event_count": end,
        "typed_event_row_count": end,
        "tape_event_row_count": end,
        "population_next_id": population_next_id,
        "active_agent_count": active_count,
        "active_agent_id_min": int(active_ids[0]) if active_count else 0,
        "active_agent_id_max": int(active_ids[-1]) if active_count else 0,
        "free_slot_count": free_slots,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.build_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
