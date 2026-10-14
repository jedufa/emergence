"""Immutable per-shape, per-backend CUDA build for workhorse V2."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Final

from .authority import (
    ARCHITECTURE_CONTRACT_RELATIVE_PATH_V2,
    PERFORMANCE_CONTRACT_RELATIVE_PATH_V2,
    VALIDATION_MATRIX_RELATIVE_PATH_V2,
    workhorse_governing_contract_bindings_v2,
)
from .contract import (
    BACKEND_FUSED_CTA_TEMPORAL_V2,
    BACKEND_PHASE_GRAPH_V2,
    BACKEND_REFERENCE_PHASED_V2,
    WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2,
    canonical_sha256,
    workhorse_artifact_identity_v2,
    workhorse_backend_v2,
    workhorse_shape_v2,
)
from .layout import workhorse_layout_v2
from .native_loader import (
    WORKHORSE_BUILD_MANIFEST_SCHEMA_V2,
    WORKHORSE_REQUIRED_EXPORTS_V2,
    expected_native_shape_v2,
    file_sha256,
)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[4]
NATIVE_SOURCE_V2: Final[Path] = (
    Path(__file__).with_name("native") / "full_evolution_workhorse_v2.cu"
)
BOUND_SOURCE_RELATIVE_PATHS_V2: Final[tuple[str, ...]] = (
    "engine/perf/cuda_full_evolution_v2/native_loader.py",
    "engine/perf/cuda_full_evolution_v2/benchmark.py",
    "engine/perf/cuda_full_evolution_v2/science_extension.py",
    "engine/perf/cuda_full_evolution_v2/workhorse/__init__.py",
    "engine/perf/cuda_full_evolution_v2/workhorse/attempt_authority.py",
    "engine/perf/cuda_full_evolution_v2/workhorse/authority.py",
    "engine/perf/cuda_full_evolution_v2/workhorse/canonical_rows.py",
    "engine/perf/cuda_full_evolution_v2/workhorse/contract.py",
    "engine/perf/cuda_full_evolution_v2/workhorse/continuation.py",
    "engine/perf/cuda_full_evolution_v2/workhorse/dispatch_plan.py",
    "engine/perf/cuda_full_evolution_v2/workhorse/indexed_checkpoint.py",
    "engine/perf/cuda_full_evolution_v2/workhorse/layout.py",
    "engine/perf/cuda_full_evolution_v2/workhorse/native_loader.py",
    "engine/perf/cuda_full_evolution_v2/workhorse/build.py",
    "engine/perf/cuda_full_evolution_v2/workhorse/probe.py",
    "engine/perf/cuda_full_evolution_v2/workhorse/science_demo.py",
    "engine/perf/cuda_full_evolution_v2/workhorse/semantic_cases.py",
    "engine/perf/cuda_full_evolution_v2/workhorse/semantic_evidence.py",
    "engine/perf/cuda_full_evolution_v2/workhorse/native/full_evolution_workhorse_v2.cu",
    "_artifacts/tw-cuda-full-evolution-reform-20260825/science_consumer_extension_contract.json",
    ARCHITECTURE_CONTRACT_RELATIVE_PATH_V2,
    PERFORMANCE_CONTRACT_RELATIVE_PATH_V2,
    VALIDATION_MATRIX_RELATIVE_PATH_V2,
)
_BACKEND_KIND: Final[dict[str, int]] = {
    BACKEND_REFERENCE_PHASED_V2: 0,
    BACKEND_PHASE_GRAPH_V2: 1,
    BACKEND_FUSED_CTA_TEMPORAL_V2: 2,
}


class FullEvolutionWorkhorseBuildError(RuntimeError):
    """Raised when a sealed shape/backend build cannot be completed."""


def _define_text(name: str, value: str) -> str:
    return f'-D{name}=\\"{value}\\"'


def _bound_source_snapshot_v2() -> tuple[list[dict[str, str]], str, dict[str, object]]:
    """Read the exact source/contract bytes bound into one immutable build."""

    inventory = [
        {"path": relative, "sha256": file_sha256(PROJECT_ROOT / relative)}
        for relative in BOUND_SOURCE_RELATIVE_PATHS_V2
    ]
    return (
        inventory,
        canonical_sha256(inventory),
        workhorse_governing_contract_bindings_v2(PROJECT_ROOT),
    )


def _require_bound_source_snapshot_unchanged_v2(
    *,
    expected_inventory: list[dict[str, str]],
    expected_governing_contracts: dict[str, object],
) -> None:
    """Fail before publication if any build-bound byte changed during nvcc."""

    observed_inventory, _, observed_governing_contracts = _bound_source_snapshot_v2()
    if (
        observed_inventory != expected_inventory
        or observed_governing_contracts != expected_governing_contracts
    ):
        raise FullEvolutionWorkhorseBuildError(
            "workhorse V2 bound source changed during compilation"
        )


def build_workhorse_prebuilt_v2(
    output_directory: str | Path,
    *,
    shape_id: str,
    backend_id: str,
    target_sm: str | None = None,
) -> Path:
    shape = workhorse_shape_v2(shape_id)
    backend = workhorse_backend_v2(backend_id)
    identity = workhorse_artifact_identity_v2(shape, backend.backend_id)
    layout = workhorse_layout_v2(shape)
    try:
        import torch
        from torch.utils.cpp_extension import load
    except Exception as exc:
        raise FullEvolutionWorkhorseBuildError(
            f"PyTorch build support unavailable: {exc}"
        ) from exc
    if not torch.cuda.is_available():
        raise FullEvolutionWorkhorseBuildError("CUDA device is unavailable")
    capability = torch.cuda.get_device_capability()
    current_sm = f"sm_{capability[0]}{capability[1]}"
    selected_sm = target_sm or current_sm
    if selected_sm != current_sm:
        raise FullEvolutionWorkhorseBuildError(
            f"target SM must exactly match current device {current_sm}"
        )
    sources = tuple(
        PROJECT_ROOT / relative for relative in BOUND_SOURCE_RELATIVE_PATHS_V2
    )
    if any(not source.is_file() for source in sources):
        raise FullEvolutionWorkhorseBuildError(
            "workhorse V2 bound source is incomplete"
        )
    inventory, source_hash, governing_contracts = _bound_source_snapshot_v2()
    module_name = (
        f"eiel_fe_whv2_{shape.shape_id}_{backend.backend_id}_{source_hash[:12]}_"
        f"{selected_sm.replace('_', '')}"
    )
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FullEvolutionWorkhorseBuildError(
            "immutable output directory must be empty"
        )

    field_defines: list[str] = []
    for field in layout.state_fields:
        field_defines.append(f"-DEIEL_STATE_{field.name.upper()}_OFFSET={field.offset}")
    for field in layout.scratch_fields:
        field_defines.append(
            f"-DEIEL_SCRATCH_{field.name.upper()}_OFFSET={field.offset}"
        )
    cuda_flags = [
        "-O3",
        "--fmad=false",
        "--prec-div=true",
        "--prec-sqrt=true",
        "-lineinfo",
        f"-DEIEL_WORKHORSE_WIDTH={shape.width}",
        f"-DEIEL_WORKHORSE_HEIGHT={shape.height}",
        f"-DEIEL_WORKHORSE_INITIAL_POPULATION={shape.initial_population}",
        f"-DEIEL_WORKHORSE_CAPACITY={shape.capacity}",
        f"-DEIEL_WORKHORSE_STATE_BYTES={layout.state_bytes_per_world}",
        f"-DEIEL_WORKHORSE_SCRATCH_BYTES={layout.scratch_bytes_per_world}",
        f"-DEIEL_WORKHORSE_MAX_K={shape.temporal_k_candidates[-1]}",
        f"-DEIEL_WORKHORSE_BACKEND_KIND={_BACKEND_KIND[backend.backend_id]}",
        _define_text("EIEL_WORKHORSE_PROFILE_ID", shape.profile_id),
        _define_text("EIEL_WORKHORSE_ROUTE_ID", shape.route_id),
        _define_text("EIEL_WORKHORSE_BACKEND_ID", backend.backend_id),
        _define_text("EIEL_WORKHORSE_LAYOUT_HASH", layout.layout_sha256),
        _define_text(
            "EIEL_WORKHORSE_SEMANTIC_HASH",
            WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2,
        ),
        _define_text("EIEL_WORKHORSE_TOPOLOGY_HASH", backend.topology_sha256(shape)),
        *field_defines,
    ]
    try:
        nvcc_version = subprocess.run(
            ["nvcc", "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise FullEvolutionWorkhorseBuildError(
            f"cannot bind the CUDA compiler identity: {exc}"
        ) from exc
    toolchain = {
        "python_version": sys.version,
        "python_compiler": platform.python_compiler(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "nvcc_version": nvcc_version,
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    }
    build_identity_payload = {
        "schema_version": "eiel_full_evolution_cuda_workhorse_build_identity_v2",
        "artifact_identity_sha256": identity["identity_sha256"],
        "shape_id": shape.shape_id,
        "backend_id": backend.backend_id,
        "target_sm": selected_sm,
        "source_inventory_sha256": source_hash,
        "governing_contracts": governing_contracts,
        "cuda_flags": cuda_flags,
        "host_flags": ["-O3", "-std=c++17"],
        "toolchain": toolchain,
    }
    build_identity = {
        **build_identity_payload,
        "build_identity_sha256": canonical_sha256(build_identity_payload),
    }
    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    previous_jobs = os.environ.get("MAX_JOBS")
    try:
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{capability[0]}.{capability[1]}"
        os.environ["MAX_JOBS"] = "1"
        # Publication ends with atomic ``os.replace``.  Keep the private build
        # and staging tree on the output filesystem so that invariant also
        # holds when the system temporary directory is a different mount.
        with tempfile.TemporaryDirectory(
            prefix=".eiel-workhorse-v2-build-", dir=output.parent
        ) as build_root:
            try:
                module = load(
                    name=module_name,
                    sources=[str(NATIVE_SOURCE_V2)],
                    extra_cuda_cflags=cuda_flags,
                    extra_cflags=["-O3", "-std=c++17"],
                    build_directory=build_root,
                    verbose=True,
                )
            except Exception as exc:
                raise FullEvolutionWorkhorseBuildError(
                    f"workhorse V2 compilation failed: {exc}"
                ) from exc
            missing = [
                name
                for name in WORKHORSE_REQUIRED_EXPORTS_V2
                if not hasattr(module, name)
            ]
            if missing:
                raise FullEvolutionWorkhorseBuildError(
                    f"compiled workhorse V2 exports missing: {missing}"
                )
            expected_shape = expected_native_shape_v2(shape.shape_id)
            checks = (
                (module.compile_shape(), expected_shape, "shape/layout"),
                (module.profile_id(), shape.profile_id, "profile"),
                (module.route_id(), shape.route_id, "route"),
                (module.backend_id(), backend.backend_id, "backend"),
                (module.layout_sha256(), layout.layout_sha256, "layout hash"),
                (
                    module.semantic_transition_sha256(),
                    WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2,
                    "semantic hash",
                ),
                (
                    module.backend_topology_sha256(),
                    backend.topology_sha256(shape),
                    "topology hash",
                ),
            )
            for actual, expected, label in checks:
                if actual != expected:
                    raise FullEvolutionWorkhorseBuildError(
                        f"compiled workhorse V2 {label} mismatch"
                    )
            runtime_compile_shape = module.compile_shape()
            runtime_topology = module.kernel_topology()
            _require_bound_source_snapshot_unchanged_v2(
                expected_inventory=inventory,
                expected_governing_contracts=governing_contracts,
            )
            source_artifact = Path(module.__file__).resolve()
            artifact = output / source_artifact.name
            staging_root = Path(build_root) / "publication"
            staging_root.mkdir()
            staged_artifact = staging_root / artifact.name
            shutil.copy2(source_artifact, staged_artifact)
            staged_artifact.chmod(0o444)
            output_identity = {
                "file_name": artifact.name,
                "bytes": staged_artifact.stat().st_size,
                "sha256": file_sha256(staged_artifact),
            }
            sealed_identity_payload = {
                "schema_version": (
                    "eiel_full_evolution_cuda_workhorse_sealed_artifact_v2"
                ),
                "artifact_identity_sha256": identity["identity_sha256"],
                "build_identity_sha256": build_identity["build_identity_sha256"],
                "target_sm": selected_sm,
                "source_inventory_sha256": source_hash,
                "governing_contracts": governing_contracts,
                "output": output_identity,
            }
            sealed_artifact_identity = {
                **sealed_identity_payload,
                "sealed_artifact_identity_sha256": canonical_sha256(
                    sealed_identity_payload
                ),
            }
            manifest: dict[str, object] = {
                "schema_version": WORKHORSE_BUILD_MANIFEST_SCHEMA_V2,
                "shape_id": shape.shape_id,
                "backend_id": backend.backend_id,
                "identity": identity,
                "build_identity": build_identity,
                "sealed_artifact_identity": sealed_artifact_identity,
                "native_shape": runtime_compile_shape,
                "kernel_topology": runtime_topology,
                "module_name": module_name,
                "target_sm": selected_sm,
                "immutable_output": True,
                "runtime_allocations_inside_advance": 0,
                "host_tick_intervention": False,
                "source_inventory": inventory,
                "source_inventory_sha256": source_hash,
                "governing_contracts": governing_contracts,
                "exported_functions": list(WORKHORSE_REQUIRED_EXPORTS_V2),
                "output": output_identity,
            }
            manifest["manifest_sha256"] = canonical_sha256(manifest)
            manifest_path = artifact.with_suffix(artifact.suffix + ".manifest.json")
            staged_manifest = staging_root / manifest_path.name
            staged_manifest.write_text(
                json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            staged_manifest.chmod(0o444)
            # A second snapshot check covers mutation during artifact staging or
            # manifest materialization.  Neither final path exists before this
            # succeeds, so a mismatch leaves no publishable artifact/manifest.
            _require_bound_source_snapshot_unchanged_v2(
                expected_inventory=inventory,
                expected_governing_contracts=governing_contracts,
            )
            os.replace(staged_artifact, artifact)
            os.replace(staged_manifest, manifest_path)
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch
        if previous_jobs is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = previous_jobs
    output.chmod(0o555)
    return artifact


__all__ = [
    "BOUND_SOURCE_RELATIVE_PATHS_V2",
    "FullEvolutionWorkhorseBuildError",
    "NATIVE_SOURCE_V2",
    "build_workhorse_prebuilt_v2",
]
