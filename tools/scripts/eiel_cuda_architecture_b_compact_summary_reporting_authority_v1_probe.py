#!/usr/bin/env python3
"""Build and qualify the standalone compact reporting authority ledger."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "engine"
ENGINE_PERF = ROOT / "engine" / "perf"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))
from native_runtime.compact_summary_digests import (
    behavior_rows_digest_sha256,
    lineage_records_digest_sha256,
)
from native_runtime.compact_summary_reporting_projection import (
    REPORTING_RETAINED_LAYOUT_VERSION,
    project_retained_reporting_authority,
)


HEADER = ENGINE_PERF / "cuda_architecture_b_compact_summary_reporting_authority_v1.cuh"
CUDA_SOURCE = ENGINE_PERF / "cuda_architecture_b_compact_summary_reporting_authority_v1.cu"
BINDING_SOURCE = (
    ROOT
    / "tests"
    / "perf"
    / "test_eiel_cuda_architecture_b_compact_summary_reporting_authority_v1_binding.cpp"
)
COMPILE_TEST_SOURCE = (
    ROOT
    / "tests"
    / "perf"
    / "test_eiel_cuda_architecture_b_compact_summary_reporting_authority_v1_compile.cu"
)
PYTHON_TEST_SOURCE = (
    ROOT / "tests" / "test_eiel_cuda_architecture_b_compact_summary_reporting_authority_v1.py"
)
PROBE_SOURCE = Path(__file__).resolve()
HOST_PROJECTION_SOURCE = (
    ENGINE / "native_runtime" / "compact_summary_reporting_projection.py"
)
DIGEST_SOURCE = ENGINE / "native_runtime" / "compact_summary_digests.py"
PUBLIC_OVERLAY_SOURCE = ENGINE / "native_runtime" / "public_metrics_overlay.py"
MODULE_NAME = "eiel_cuda_architecture_b_compact_summary_reporting_authority_v1_probe"
STRICT_CAPACITY = 64
DYNAMIC_CAPACITY = 224
HEADER_STRUCT = struct.Struct("<qHHI")
BEHAVIOR_STRUCT = struct.Struct("<qddfffiii7BB")
LINEAGE_STRUCT = struct.Struct("<qdddd10f13i4B")
LINEAGE_OFFSETS = {
    "agent_id": 0,
    "energy": 8,
    "energy_collected_exact": 16,
    "episode0_pred_err_mean": 24,
    "identity_drift_ema": 32,
    "health": 40,
    "injury": 44,
    "drives": 48,
    "pred_err_sum": 64,
    "drive_jitter": 68,
    "aggression_bias_f32": 72,
    "exploration_bias_f32": 76,
    "y": 80,
    "x": 84,
    "founder_id": 88,
    "birth_generation": 92,
    "birth_tick": 96,
    "lineage_depth": 100,
    "offspring_count": 104,
    "survival_steps": 108,
    "first_repro_step": 112,
    "pred_err_count": 116,
    "patch_habitat_lineage_id": 120,
    "hidden_size": 124,
    "memory_dim": 128,
    "use_gru": 132,
    "reserved": 133,
}
ACCEPTED_LINEAGE_CONSTANTS = {
    "parent_ids": [],
    "parent_lineage_ids": [],
    "ecological_role": "general_agent",
    "role_source": "default",
    "role_confidence": None,
}
FIXTURES = {
    "start_born_died_reuse": (0, 1),
    "empty": (1, 1),
    "overflow": (2, 1),
    "duplicate": (3, 1),
    "signed_zero": (4, 1),
    "g2_isolation": (5, 2),
    "write_after_seal": (6, 1),
    "identity_drift_unavailable": (7, 1),
    "missing_lineage_source": (8, 1),
    "malformed_lineage_source": (9, 1),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retained_bytes_per_generation(row_capacity: int) -> int:
    if not 1 <= row_capacity <= DYNAMIC_CAPACITY:
        raise ValueError("row_capacity must be in [1, 224]")
    return HEADER_STRUCT.size + row_capacity * (
        BEHAVIOR_STRUCT.size + LINEAGE_STRUCT.size
    )


def _module_artifact(module: ModuleType) -> Path:
    artifact = Path(module.__file__).resolve()
    if not artifact.is_file():
        raise RuntimeError(f"compiled module is absent: {artifact}")
    return artifact


def _qualify_ptxas(output_dir: Path) -> dict[str, Any]:
    object_path = output_dir / "reporting_authority_v2_ptxas.cuda.o"
    command = [
        "nvcc",
        "-std=c++17",
        "-O3",
        "--fmad=false",
        "-lineinfo",
        "-Xptxas=-v",
        "-gencode=arch=compute_75,code=sm_75",
        f"-I{ENGINE_PERF}",
        "-c",
        str(CUDA_SOURCE),
        "-o",
        str(object_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    output = f"{completed.stdout}{completed.stderr}"
    log_path = output_dir / "reporting_authority_v2_ptxas.log"
    log_path.write_text(output, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"standalone ptxas qualification failed:\n{output}")
    match = re.search(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
        r"(\d+) bytes spill loads",
        output,
    )
    if match is None:
        raise RuntimeError("ptxas output omitted stack/spill accounting")
    stack_bytes, spill_store_bytes, spill_load_bytes = map(int, match.groups())
    if spill_store_bytes != 0 or spill_load_bytes != 0:
        raise RuntimeError(
            "reporting authority kernel spilled: "
            f"stores={spill_store_bytes} loads={spill_load_bytes}"
        )
    return {
        "command": command,
        "log": str(log_path),
        "object": str(object_path),
        "object_sha256": _sha256(object_path),
        "stack_frame_bytes": stack_bytes,
        "spill_store_bytes": spill_store_bytes,
        "spill_load_bytes": spill_load_bytes,
    }


def build(output_dir: Path, *, verbose: bool = False) -> tuple[ModuleType, dict[str, Any]]:
    missing = [
        str(path)
        for path in (
            HEADER,
            CUDA_SOURCE,
            BINDING_SOURCE,
            COMPILE_TEST_SOURCE,
            PYTHON_TEST_SOURCE,
            HOST_PROJECTION_SOURCE,
            DIGEST_SOURCE,
            PUBLIC_OVERLAY_SOURCE,
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"reporting authority sources are missing: {missing}")
    if shutil.which("nvcc") is None:
        raise RuntimeError("nvcc is required")
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"immutable build directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    ptxas_receipt = _qualify_ptxas(output_dir)

    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    previous_jobs = os.environ.get("MAX_JOBS")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
    os.environ["MAX_JOBS"] = "1"
    try:
        module = load(
            name=MODULE_NAME,
            sources=[str(BINDING_SOURCE), str(CUDA_SOURCE)],
            extra_include_paths=[str(ENGINE_PERF)],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "--fmad=false",
                "-lineinfo",
                "-Xptxas=-v",
            ],
            build_directory=str(output_dir),
            verbose=verbose,
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

    artifact = _module_artifact(module)
    identity = _inspect_artifact(artifact)
    manifest = {
        "schema_version": "eiel_cuda_architecture_b_reporting_authority_v2_build",
        "retained_layout_version": REPORTING_RETAINED_LAYOUT_VERSION,
        "module_name": MODULE_NAME,
        "artifact": str(artifact),
        "artifact_sha256": _sha256(artifact),
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (HEADER, CUDA_SOURCE, BINDING_SOURCE)
        },
        "qualification_source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (COMPILE_TEST_SOURCE, PYTHON_TEST_SOURCE, PROBE_SOURCE)
        },
        "reference_source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (HOST_PROJECTION_SOURCE, DIGEST_SOURCE, PUBLIC_OVERLAY_SOURCE)
        },
        "cuda_arch": "sm_75",
        "fmad": False,
        "artifact_identity": identity,
        "ptxas": ptxas_receipt,
        "runtime_jit_compilation": False,
    }
    manifest_path = output_dir / "reporting_authority_v2_build_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return module, manifest


def _inspect_artifact(artifact: Path) -> dict[str, Any]:
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        raise RuntimeError("cuobjdump is required")
    listed = subprocess.run(
        [cuobjdump, "--list-elf", str(artifact)],
        check=False,
        capture_output=True,
        text=True,
    )
    if listed.returncode != 0 or "sm_75" not in f"{listed.stdout}\n{listed.stderr}":
        raise RuntimeError("artifact has no sm_75 ELF image")
    ptx = subprocess.run(
        [cuobjdump, "--dump-ptx", str(artifact)],
        check=False,
        capture_output=True,
        text=True,
    )
    ptx_embedded = ".version" in f"{ptx.stdout}\n{ptx.stderr}"
    if ptx_embedded:
        raise RuntimeError("artifact unexpectedly embeds PTX")
    return {
        "sm75_elf_present": True,
        "sass_only": True,
        "ptx_embedded": False,
        "runtime_jit_required": False,
    }


def decode_retained(
    retained: torch.Tensor,
    *,
    row_capacity: int = STRICT_CAPACITY,
) -> list[dict[str, Any]]:
    if retained.device.type != "cpu" or retained.dtype != torch.uint8:
        raise ValueError("retained must be one CPU uint8 transfer")
    if not retained.is_contiguous() or retained.dim() != 2:
        raise ValueError("retained must be contiguous [G,bytes]")
    stride = retained_bytes_per_generation(row_capacity)
    if tuple(retained.shape)[1] != stride:
        raise ValueError("retained byte stride disagrees with the frozen layout")
    raw = retained.numpy().tobytes(order="C")
    decoded: list[dict[str, Any]] = []
    for generation in range(retained.shape[0]):
        base = generation * stride
        start_id_ceiling, behavior_count, lineage_count, status = (
            HEADER_STRUCT.unpack_from(raw, base)
        )
        if behavior_count > row_capacity or lineage_count > row_capacity:
            raise ValueError("device counts exceed retained row capacity")
        behavior_offset = base + HEADER_STRUCT.size
        lineage_offset = behavior_offset + row_capacity * BEHAVIOR_STRUCT.size
        behavior_rows = []
        for index in range(behavior_count):
            values = BEHAVIOR_STRUCT.unpack_from(
                raw, behavior_offset + index * BEHAVIOR_STRUCT.size
            )
            behavior_rows.append(_decode_behavior(values, generation, start_id_ceiling))
        lineage_rows = []
        alive_behavior_rows = [row for row in behavior_rows if row["alive_end"]]
        if status == 2 and len(alive_behavior_rows) != lineage_count:
            raise ValueError("sealed behavior and lineage counts disagree")
        for index in range(lineage_count):
            values = LINEAGE_STRUCT.unpack_from(
                raw, lineage_offset + index * LINEAGE_STRUCT.size
            )
            lineage_rows.append(_decode_lineage(values, alive_behavior_rows[index]))
        decoded.append(
            {
                "generation": generation,
                "start_id_ceiling": start_id_ceiling,
                "behavior_row_count": behavior_count,
                "lineage_record_count": lineage_count,
                "status": status,
                "behavior_rows": behavior_rows,
                "lineage_records": lineage_rows,
                "raw": raw[base : base + stride],
            }
        )
    return decoded


def _decode_behavior(
    values: tuple[Any, ...], generation: int, start_id_ceiling: int
) -> dict[str, Any]:
    agent_id, energy, fitness, injury_start, injury_end, aggression = values[:6]
    lifespan, offspring, attacks = values[6:9]
    actions = values[9:16]
    flags = values[16]
    return {
        "generation": generation,
        "agent_id": agent_id,
        "cohort": "start" if agent_id < start_id_ceiling else "born",
        "move_n": actions[0],
        "move_s": actions[1],
        "move_e": actions[2],
        "move_w": actions[3],
        "gather": actions[4],
        "rest": actions[5],
        "attack": actions[6],
        "lifespan_steps": lifespan,
        "energy_collected": energy,
        "offspring": offspring,
        "attacks_made": attacks,
        "fitness": fitness,
        "alive_end": int(bool(flags & 1)),
        "eligible_for_selection": int(bool(flags & 2)),
        "kept_next": int(bool(flags & 4)),
        "injury_end": injury_end,
        "injury_delta": injury_end - injury_start,
        "aggression_bias": float(round(aggression, 8)),
        "_injury_start": injury_start,
        "_raw_flags": flags,
    }


def _decode_lineage(
    values: tuple[Any, ...], behavior: dict[str, Any]
) -> dict[str, Any]:
    agent_id = values[0]
    energy = values[1]
    energy_collected = values[2]
    episode0_pred_err_mean = values[3]
    identity_drift_ema = values[4]
    health, injury = values[5:7]
    drives = values[7:11]
    pred_err_sum, drive_jitter, aggression, exploration = values[11:15]
    (
        y,
        x,
        founder_id,
        birth_generation,
        birth_tick,
        lineage_depth,
        offspring_count,
        survival_steps,
        first_repro_step,
        pred_err_count,
        patch_lineage,
        hidden_size,
        memory_dim,
    ) = values[15:28]
    use_gru_raw = values[28]
    reserved = values[29:32]
    if agent_id != behavior["agent_id"]:
        raise ValueError("lineage agent ID disagrees with its alive behavior row")
    if use_gru_raw not in (0, 1) or any(reserved):
        raise ValueError("lineage flag or reserved storage is malformed")
    from native_runtime.public_metrics_overlay import _native_lineage_record

    record = _native_lineage_record(
        agent_id=agent_id,
        founder_id=founder_id,
        lineage_depth=lineage_depth,
        birth_generation=birth_generation,
        birth_tick=birth_tick,
        parent_ids=(),
        energy=energy,
        health=health,
        injury=injury,
        drives=drives,
        offspring_count=offspring_count,
        survival_steps=float(survival_steps),
        energy_collected=energy_collected,
        first_repro_step=None if first_repro_step < 0 else first_repro_step,
        pred_err_count=pred_err_count,
        pred_err_sum=pred_err_sum,
        episode0_pred_err_mean=episode0_pred_err_mean,
        identity_drift_ema=identity_drift_ema,
        drive_jitter=drive_jitter,
        hidden_size=hidden_size,
        memory_dim=memory_dim,
        use_gru=bool(use_gru_raw),
        aggression_bias=float(round(aggression, 8)),
        exploration_bias=float(round(exploration, 8)),
        patch_habitat_lineage_id=patch_lineage,
        y=y,
        x=x,
    )
    stats = record["stats"]
    for field, expected in ACCEPTED_LINEAGE_CONSTANTS.items():
        if stats[field] != expected:
            raise RuntimeError(f"accepted lineage constant drifted: {field}")
    return record


def public_behavior_rows(generation: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in generation["behavior_rows"]
    ]


def run_probe(module: ModuleType) -> dict[str, Any]:
    anchor = torch.empty(0, dtype=torch.uint8, device="cuda")
    results: dict[str, Any] = {}
    for name, (fixture, generations) in FIXTURES.items():
        device_retained = module.run_fixture(
            anchor, fixture, generations, STRICT_CAPACITY, DYNAMIC_CAPACITY + 1
        )
        if device_retained.device.type != "cuda":
            raise RuntimeError("fixture output left the device before settlement")
        retained = device_retained.cpu()  # The single allowed segment transfer.
        decoded = decode_retained(retained)
        for generation in decoded:
            behavior_rows = public_behavior_rows(generation)
            generation["behavior_rows_digest_sha256"] = (
                behavior_rows_digest_sha256(behavior_rows)
            )
            generation["lineage_records_digest_sha256"] = (
                lineage_records_digest_sha256(
                    generation["generation"], generation["lineage_records"]
                )
            )
        projection_verified = False
        if all(generation["status"] == 2 for generation in decoded):
            projected = project_retained_reporting_authority(
                retained,
                first_generation=0,
                row_capacity=STRICT_CAPACITY,
            )
            if len(projected) != len(decoded):
                raise RuntimeError("independent projection generation count disagrees")
            for generation, projected_cells in zip(decoded, projected, strict=True):
                if len(projected_cells) != 1:
                    raise RuntimeError("fixture projection did not retain one cell")
                independent = projected_cells[0]
                if (
                    independent["retained_layout_version"]
                    != REPORTING_RETAINED_LAYOUT_VERSION
                    or independent["behavior_rows"]
                    != public_behavior_rows(generation)
                    or independent["lineage_records"] != generation["lineage_records"]
                    or independent["behavior_rows_digest_sha256"]
                    != generation["behavior_rows_digest_sha256"]
                    or independent["lineage_records_digest_sha256"]
                    != generation["lineage_records_digest_sha256"]
                ):
                    raise RuntimeError(
                        f"independent host projection disagrees for fixture {name}"
                    )
            projection_verified = True
        results[name] = {
            "host_transfers": 1,
            "independent_projection_verified": projection_verified,
            "generations": decoded,
        }
    receipt = dict(module.resource_receipt())
    receipt["lineage_field_offsets"] = dict(receipt["lineage_field_offsets"])
    if receipt["retained_layout_version"] != REPORTING_RETAINED_LAYOUT_VERSION:
        raise RuntimeError("C++ and host retained layout versions disagree")
    if receipt["lineage_field_offsets"] != LINEAGE_OFFSETS:
        raise RuntimeError("C++ and host lineage offsets disagree")
    if receipt["lineage_row_bytes"] != LINEAGE_STRUCT.size:
        raise RuntimeError("C++ and host lineage row sizes disagree")
    receipt["abi_layout_verified"] = True
    receipt["accepted_lineage_constants"] = ACCEPTED_LINEAGE_CONSTANTS
    receipt["fixtures"] = results
    receipt["one_host_transfer_per_fixture"] = all(
        result["host_transfers"] == 1 for result in results.values()
    )
    receipt["strict_g2_b1_bytes"] = 2 * retained_bytes_per_generation(64)
    receipt["dynamic_g32_b16_bytes"] = (
        32 * 16 * retained_bytes_per_generation(DYNAMIC_CAPACITY)
    )
    return receipt


def _json_safe_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    safe = dict(receipt)
    fixtures = {}
    for name, result in receipt["fixtures"].items():
        generations = []
        for generation in result["generations"]:
            generations.append(
                {
                    key: value
                    for key, value in generation.items()
                    if key != "raw"
                }
            )
        fixtures[name] = {**result, "generations": generations}
    safe["fixtures"] = fixtures
    return safe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    module, manifest = build(args.output_dir, verbose=args.verbose)
    receipt = run_probe(module)
    output = {
        "manifest": manifest,
        "probe": _json_safe_receipt(receipt),
    }
    receipt_path = args.output_dir / "reporting_authority_v2_acceptance_receipt.json"
    receipt_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
