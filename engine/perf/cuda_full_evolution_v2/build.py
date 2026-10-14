"""Explicit immutable build step for Full Evolution CUDA Native V2."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Final, Iterator, Sequence

from .contract import CUDA_NATIVE_ABI_V1, CUDA_NATIVE_PROFILE_V2, CUDA_NATIVE_ROUTE_V2
from .native_loader import (
    BOUND_SOURCE_RELATIVE_PATHS_V2,
    BUILD_MANIFEST_SCHEMA_V2,
    EXECUTION_PLAN_CLASS_V2,
    EXTENSION_VERSION_V2,
    NORMATIVE_EXECUTION_CONTRACT_PATHS_V2,
    PROJECT_ROOT,
    REQUIRED_EXPORTS_V2,
    canonical_sha256,
    cuda_device_identity_v2,
    expected_execution_plan_contract_v2,
    expected_layout_manifest_v2,
    file_sha256,
)

NATIVE_SOURCE_V2: Final[Path] = (
    Path(__file__).with_name("native") / "full_evolution_v2.cu"
)
BOUND_SOURCE_PATHS_V2: Final[tuple[Path, ...]] = (
    *(PROJECT_ROOT / relative for relative in BOUND_SOURCE_RELATIVE_PATHS_V2),
)
CUDA_FLAGS_V2: Final[tuple[str, ...]] = (
    "-O3",
    "--fmad=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--extended-lambda",
    "-lineinfo",
)
CXX_FLAGS_V2: Final[tuple[str, ...]] = ("-O3", "-std=c++17")


class CudaFullEvolutionNativeBuildError(RuntimeError):
    """Raised when an immutable prebuilt artifact cannot be produced exactly."""


def _tool_identity(command: Sequence[str]) -> str:
    completed = subprocess.run(
        list(command), check=False, capture_output=True, text=True
    )
    if completed.returncode:
        raise CudaFullEvolutionNativeBuildError(
            f"tool identity command failed: {' '.join(command)}"
        )
    lines = (completed.stdout or completed.stderr).strip().splitlines()
    if not lines:
        raise CudaFullEvolutionNativeBuildError("tool identity output is empty")
    return "\n".join(lines)


@contextmanager
def _temporary_environment(updates: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def build_prebuilt_cuda_full_evolution_v2(
    output_directory: str | Path,
    *,
    target_sm: str | None = None,
) -> Path:
    """Compile once outside execution and seal a source/toolchain/SM receipt."""
    try:
        import torch
        from torch.utils.cpp_extension import load
    except Exception as exc:
        raise CudaFullEvolutionNativeBuildError(
            f"PyTorch build support unavailable: {exc}"
        ) from exc
    if not torch.cuda.is_available():
        raise CudaFullEvolutionNativeBuildError("CUDA device is unavailable")
    capability = torch.cuda.get_device_capability()
    selected_device_index = torch.cuda.current_device()
    build_device_identity = cuda_device_identity_v2(torch, selected_device_index)
    current_sm = f"sm_{capability[0]}{capability[1]}"
    selected_sm = target_sm or current_sm
    if selected_sm != current_sm or re.fullmatch(r"sm_[0-9]{2,3}", selected_sm) is None:
        raise CudaFullEvolutionNativeBuildError(
            f"target SM must exactly match current device {current_sm}"
        )
    if any(not path.is_file() for path in BOUND_SOURCE_PATHS_V2):
        raise CudaFullEvolutionNativeBuildError(
            "bound CUDA Native V2 source is incomplete"
        )
    source_inventory = [
        {"path": _relative(path), "sha256": file_sha256(path)}
        for path in BOUND_SOURCE_PATHS_V2
    ]
    source_identity = canonical_sha256(source_inventory)
    module_name = (
        f"eiel_full_evolution_cuda_native_v2_{source_identity[:12]}_"
        f"{selected_sm.replace('_', '')}"
    )
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise CudaFullEvolutionNativeBuildError(
            "immutable output directory must be empty"
        )
    arch_list = f"{capability[0]}.{capability[1]}"
    with tempfile.TemporaryDirectory(prefix="eiel-cuda-native-v2-build-") as scratch:
        scratch_root = Path(scratch)
        snapshot_root = scratch_root / "sealed-source"
        build_root = scratch_root / "build"
        build_root.mkdir()
        for source, inventory_row in zip(
            BOUND_SOURCE_PATHS_V2, source_inventory, strict=True
        ):
            destination = snapshot_root / str(inventory_row["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            destination.chmod(0o444)
        snapshot_inventory = [
            {
                "path": relative,
                "sha256": file_sha256(snapshot_root / relative),
            }
            for relative in BOUND_SOURCE_RELATIVE_PATHS_V2
        ]
        if snapshot_inventory != source_inventory:
            raise CudaFullEvolutionNativeBuildError(
                "sealed source snapshot differs from pre-build inventory"
            )
        snapshot_native = snapshot_root / BOUND_SOURCE_RELATIVE_PATHS_V2[0]
        snapshot_directories = sorted(
            (
                snapshot_root,
                *(path for path in snapshot_root.rglob("*") if path.is_dir()),
            ),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in snapshot_directories:
            directory.chmod(0o555)
        try:
            with _temporary_environment(
                {"TORCH_CUDA_ARCH_LIST": arch_list, "MAX_JOBS": "1"}
            ):
                try:
                    module = load(
                        name=module_name,
                        sources=[str(snapshot_native)],
                        extra_cuda_cflags=list(CUDA_FLAGS_V2),
                        extra_cflags=list(CXX_FLAGS_V2),
                        build_directory=str(build_root),
                        verbose=True,
                    )
                except Exception as exc:
                    raise CudaFullEvolutionNativeBuildError(
                        f"CUDA Native V2 compilation failed: {exc}"
                    ) from exc
            post_build_snapshot_inventory = [
                {"path": relative, "sha256": file_sha256(snapshot_root / relative)}
                for relative in BOUND_SOURCE_RELATIVE_PATHS_V2
            ]
            if post_build_snapshot_inventory != source_inventory:
                raise CudaFullEvolutionNativeBuildError(
                    "sealed source snapshot changed during compilation"
                )
        finally:
            for directory in reversed(snapshot_directories):
                directory.chmod(0o755)
        missing = [name for name in REQUIRED_EXPORTS_V2 if not hasattr(module, name)]
        if missing:
            raise CudaFullEvolutionNativeBuildError(
                f"compiled extension is missing exports: {missing}"
            )
        if not isinstance(getattr(module, EXECUTION_PLAN_CLASS_V2, None), type):
            raise CudaFullEvolutionNativeBuildError(
                "compiled extension is missing its native execution-plan class"
            )
        if module.abi_version() != CUDA_NATIVE_ABI_V1:
            raise CudaFullEvolutionNativeBuildError("compiled extension ABI mismatch")
        if module.extension_version() != EXTENSION_VERSION_V2:
            raise CudaFullEvolutionNativeBuildError(
                "compiled extension version mismatch"
            )
        if module.architecture_id() != CUDA_NATIVE_PROFILE_V2:
            raise CudaFullEvolutionNativeBuildError("compiled architecture ID mismatch")
        if module.route_id() != CUDA_NATIVE_ROUTE_V2:
            raise CudaFullEvolutionNativeBuildError("compiled route ID mismatch")
        if module.layout_manifest() != expected_layout_manifest_v2():
            raise CudaFullEvolutionNativeBuildError("compiled layout manifest mismatch")
        post_build_live_inventory = [
            {"path": _relative(path), "sha256": file_sha256(path)}
            for path in BOUND_SOURCE_PATHS_V2
        ]
        if post_build_live_inventory != source_inventory:
            raise CudaFullEvolutionNativeBuildError(
                "bound live source changed while the sealed snapshot compiled"
            )
        source_artifact = Path(module.__file__).resolve()
        artifact = output / source_artifact.name
        shutil.copy2(source_artifact, artifact)
    artifact.chmod(0o444)
    layout_manifest = expected_layout_manifest_v2()
    manifest: dict[str, object] = {
        "schema_version": BUILD_MANIFEST_SCHEMA_V2,
        "profile": CUDA_NATIVE_PROFILE_V2,
        "route_id": CUDA_NATIVE_ROUTE_V2,
        "native_abi": CUDA_NATIVE_ABI_V1,
        "extension_version": EXTENSION_VERSION_V2,
        "module_name": module_name,
        "immutable_output": True,
        "prebuilt_outside_hot_advance": True,
        "runtime_jit": False,
        "compiled_from_sealed_snapshot": True,
        "sealed_snapshot": {
            "files_mode": "0444",
            "directories_mode": "0555",
            "post_compile_inventory_verified": True,
        },
        "source_inventory": source_inventory,
        "normative_execution_contracts": [
            row
            for row in source_inventory
            if row["path"] in NORMATIVE_EXECUTION_CONTRACT_PATHS_V2
        ],
        "source_inventory_sha256": source_identity,
        "post_build_live_source_inventory_sha256": canonical_sha256(
            post_build_live_inventory
        ),
        "target": {
            "sm": selected_sm,
            "cuda_runtime": str(torch.version.cuda),
            "compatibility": "exact Python/PyTorch/CUDA runtime and target SM",
            "build_device": build_device_identity,
        },
        "toolchain": {
            "nvcc": _tool_identity(("nvcc", "--version")),
            "cxx": _tool_identity(("c++", "--version")),
            "torch": str(torch.__version__),
            "cuda_flags": list(CUDA_FLAGS_V2),
            "cxx_flags": list(CXX_FLAGS_V2),
            "torch_cuda_arch_list": arch_list,
        },
        "python": {
            "cache_tag": sys.implementation.cache_tag,
            "version": platform.python_version(),
            "platform": platform.platform(),
        },
        "exported_functions": list(REQUIRED_EXPORTS_V2),
        "layout_manifest": layout_manifest,
        "layout_manifest_sha256": canonical_sha256(layout_manifest),
        "runtime": {
            "device_or_tensor_allocations_inside_hot_advance": 0,
            "python_pending_dispatch_objects_inside_hot_advance": 1,
            "host_device_transfers_inside_hot_advance": 0,
            "host_synchronizations_inside_hot_advance": 0,
            "optimized_global_scratch_bytes": 0,
            "production_owner": "CudaFullEvolutionDirectSparseOwnerV3",
            "production_event_emission": "direct_per_world_bounded_sparse_v3",
            "production_event_drain": "actual_count_only_sparse_v3_native_pack",
            "dense_event_drain_role": "tiny_gpu_differential_oracle_only",
            "event_drain_timing_reported_separately": True,
            "sparse_drain_allocation": "exact_committed_event_count",
            "public_advance_dispatches": "one_successful_attempt_plus_overflow_retries",
            "dispatch_tick_limit": "temporal_k",
            "overflow_transaction": (
                "full_resident_state_d2d_snapshot_restore_re_admit_grow_retry"
            ),
            "direct_sparse_resource_admission": (
                "state_plus_full_state_rollback_plus_direct_staging_plus_"
                "simultaneous_exact_pack_plus_controls_plus_margin"
            ),
            "horizon_dependent_resident_bytes": 0,
            "event_frontier_commit": (
                "exact_canonical_local_envelope_inode_digest_open_descriptor_"
                "receipt_v3"
            ),
            "dense_oracle_event_frontier_commit": (
                "exact_immutable_sparse_content_external_sink_receipt_v3"
            ),
            "checkpoint_requires_acknowledged_event_frontier": True,
            "checkpoint_schema": "eiel_full_evolution_cuda_checkpoint_v2",
            "checkpoint_canonical_state": "immutable_cpu_bytes_with_content_layout_and_metadata_seals",
            "acs2_capture_source": "producer_borrowed_resident_cuda_rows_v1",
            "acs2_device_compaction_bytes": 0,
            "acs2_full_state_host_materialization_bytes": 0,
            "acs2_capture_pool": "one_shared_preallocated_two_pinned_slabs",
            "acs2_worker_cuda_references": 0,
            "public_state_projection": "validated_detached_clone_only",
            "drain_validation_boundary": "non_timed",
            "execution_plan_contract": "eiel_full_evolution_cuda_execution_plan_contract_v2",
            "execution_plan_creation_boundary": "outside_timed_launch",
            "execution_plan_runtime_receipt": "eiel_full_evolution_cuda_execution_plan_receipt_v2",
            "execution_plan_stream_policy": (
                "immutable_construction_stream_including_default_stream_0"
            ),
            "native_execution_receipt_binding": (
                "exact_source_topology_pointers_device_artifact_and_null_timings"
            ),
            "checkpoint_selected_device_identity": "exact_runtime_device_identity_v2",
            "full_capacity_benchmark_fixture": (
                "fresh_p256_to_exact_c512_sealed_technical_fixture_v2"
            ),
        },
        "execution_plan_contract": expected_execution_plan_contract_v2(),
        "output": {
            "file_name": artifact.name,
            "bytes": artifact.stat().st_size,
            "sha256": file_sha256(artifact),
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = artifact.with_suffix(artifact.suffix + ".manifest.json")
    temporary_manifest = output / f".{manifest_path.name}.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    manifest_path.chmod(0o444)
    output.chmod(0o555)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-sm")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = build_prebuilt_cuda_full_evolution_v2(
        args.output_dir, target_sm=args.target_sm
    )
    print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOUND_SOURCE_PATHS_V2",
    "CUDA_FLAGS_V2",
    "CudaFullEvolutionNativeBuildError",
    "NATIVE_SOURCE_V2",
    "build_prebuilt_cuda_full_evolution_v2",
    "main",
]
