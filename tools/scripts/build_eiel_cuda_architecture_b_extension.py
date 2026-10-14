from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.utils.cpp_extension as cpp_extension
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "engine") not in sys.path:
    sys.path.insert(0, str(ROOT / "engine"))

from perf.cuda_architecture_b_generation import (  # noqa: E402
    COMPILE_SOURCES as GENERATION_COMPILE_SOURCES,
    CUDA_GENERATION_PREBUILT_CAPABILITY_CONTRACT,
    HASH_SOURCES as GENERATION_HASH_SOURCES,
)


MANIFEST_SCHEMA_VERSION = "eiel_cuda_architecture_b_build_manifest_v2"
EXTENSION_ABI_VERSION = "eiel_cuda_architecture_b_abi_v3"
EXTENSION_VERSION = "eiel_cuda_architecture_b_resident_generation_v6"
EXPORTED_FUNCTIONS = (
    "eiel_cuda_architecture_b_generation_v3",
    "eiel_cuda_architecture_b_generation_multicell_v2",
    "eiel_cuda_architecture_b_generation_phase_e_v1",
)
CUDA_ARCHITECTURE_B_CAPABILITY_CONTRACT = dict(
    CUDA_GENERATION_PREBUILT_CAPABILITY_CONTRACT
)
CUDA_ARCHITECTURE_B_COMPILE_SOURCES = tuple(
    source.resolve() for source in GENERATION_COMPILE_SOURCES
)
CUDA_ARCHITECTURE_B_SOURCE_INVENTORY = tuple(
    source.resolve() for source in GENERATION_HASH_SOURCES
)
CXX_BUILD_FLAGS = (
    "-O2",
    "-ffp-contract=off",
    "-DEIEL_CUDA_ARCHITECTURE_B_NO_PYBIND",
    "-DEIEL_CUDA_ARCHITECTURE_B_NO_BIRTH_MICROCYCLE_PYBIND",
)
NVCC_BUILD_FLAGS = (
    "-O2",
    "--fmad=false",
    "-DEIEL_CUDA_ARCHITECTURE_B_NO_PYBIND",
    "-DEIEL_CUDA_ARCHITECTURE_B_NO_BIRTH_MICROCYCLE_PYBIND",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_module_name(module_name: str) -> None:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", module_name) is None:
        raise ValueError("module name must be a valid Python extension module identifier")


def _cuda_arch_from_capability(capability: tuple[int, int]) -> str:
    major, minor = capability
    if major < 1 or minor < 0:
        raise ValueError(f"invalid CUDA device capability: {capability!r}")
    return f"{major}.{minor}"


def _require_outputs_absent(output: Path) -> Path:
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    existing = [path for path in (output, manifest_path) if path.exists()]
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"immutable CUDA extension output already exists: {rendered}")
    return manifest_path


def _prepare_source_snapshots(
    sources: Sequence[Path],
    build_dir: Path,
) -> tuple[tuple[Path, ...], list[dict[str, str]]]:
    if not sources:
        raise ValueError("at least one CUDA Architecture B source is required")
    snapshot_dir = build_dir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshots: list[Path] = []
    records: list[dict[str, str]] = []
    snapshot_names: set[str] = set()
    for source in sources:
        if source.name in snapshot_names:
            raise ValueError(f"source snapshot names must be unique: {source.name}")
        snapshot_names.add(source.name)
        source_sha256 = _sha256(source)
        snapshot = snapshot_dir / source.name
        if snapshot.exists():
            if _sha256(snapshot) != source_sha256:
                raise FileExistsError(
                    f"build source snapshot already exists with different content: {snapshot}"
                )
        else:
            shutil.copy2(source, snapshot)
        if _sha256(snapshot) != source_sha256:
            raise RuntimeError(f"failed to capture stable build source snapshot: {source}")
        snapshots.append(snapshot.resolve())
        records.append({"path": str(source), "sha256": source_sha256})
    return tuple(snapshots), records


def _probe_version(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())


def _nvcc_command() -> list[str]:
    configured = os.environ.get("NVCC")
    if configured:
        return shlex.split(configured)
    cuda_home = cpp_extension.CUDA_HOME
    if cuda_home:
        candidate = Path(cuda_home) / "bin" / "nvcc"
        if candidate.is_file():
            return [str(candidate)]
    discovered = shutil.which("nvcc")
    if discovered:
        return [discovered]
    raise RuntimeError("nvcc was not found; a CUDA toolkit is required")


def _verify_build_ninja(
    *,
    build_ninja: Path,
    sources: Sequence[Path],
    cuda_arch: str,
) -> None:
    if not build_ninja.is_file():
        raise RuntimeError(f"Torch extension build did not produce {build_ninja}")
    build_text = build_ninja.read_text(encoding="utf-8", errors="replace")
    required_fragments = [
        str(cpp_extension.TORCH_LIB_PATH),
        *(str(path) for path in cpp_extension.include_paths()),
        *(str(path) for path in sources),
        *CXX_BUILD_FLAGS,
        *NVCC_BUILD_FLAGS,
        "nvcc",
        f"compute_{cuda_arch.replace('.', '')}",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in build_text]
    if missing:
        raise RuntimeError(f"CUDA extension build.ninja is missing required build identity: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the EIEL CUDA Architecture B extension outside measured execution."
    )
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--module-name", default="eiel_cuda_architecture_b_ext")
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()

    try:
        _validate_module_name(args.module_name)
    except ValueError as exc:
        parser.error(str(exc))

    sources = CUDA_ARCHITECTURE_B_SOURCE_INVENTORY
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"required CUDA extension source is missing: {source}")

    build_dir = Path(args.build_dir).resolve()
    output = Path(args.output).resolve()
    manifest_path = _require_outputs_absent(output)
    build_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_sources, source_records = _prepare_source_snapshots(sources, build_dir)
    snapshot_by_source = dict(zip(sources, build_sources, strict=True))
    compile_sources = tuple(
        snapshot_by_source[source] for source in CUDA_ARCHITECTURE_B_COMPILE_SOURCES
    )

    if not torch.cuda.is_available():
        raise RuntimeError("an available CUDA device is required to bind the immutable GPU architecture")
    if args.device_index < 0 or args.device_index >= torch.cuda.device_count():
        raise ValueError(
            f"device index {args.device_index} is outside available range 0..{torch.cuda.device_count() - 1}"
        )

    torch.cuda.set_device(args.device_index)
    device = torch.device("cuda", args.device_index)
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    cuda_arch = _cuda_arch_from_capability(capability)
    nvcc_command = _nvcc_command()
    compiler_command = shlex.split(os.environ.get("CXX", "c++"))
    compiler_path = shutil.which(compiler_command[0]) or compiler_command[0]
    nvcc_path = shutil.which(nvcc_command[0]) or nvcc_command[0]

    prior_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch
    try:
        module = load(
            name=args.module_name,
            sources=[str(path) for path in compile_sources],
            extra_include_paths=[str(build_dir / "source_snapshot")],
            extra_cflags=list(CXX_BUILD_FLAGS),
            extra_cuda_cflags=list(NVCC_BUILD_FLAGS),
            with_cuda=True,
            build_directory=str(build_dir),
            verbose=False,
        )
    finally:
        if prior_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prior_arch_list

    missing_exports = [name for name in EXPORTED_FUNCTIONS if not hasattr(module, name)]
    if missing_exports:
        raise RuntimeError(f"built CUDA extension is missing required exports: {missing_exports}")

    changed_sources = [
        record["path"]
        for record in source_records
        if _sha256(Path(record["path"])) != record["sha256"]
    ]
    if changed_sources:
        raise RuntimeError(f"CUDA extension sources changed during the build: {changed_sources}")

    built_path = Path(module.__file__).resolve()
    _verify_build_ninja(
        build_ninja=build_dir / "build.ninja",
        sources=compile_sources,
        cuda_arch=cuda_arch,
    )
    shutil.copy2(built_path, output)

    build_flags: dict[str, list[str]] = {
        "cxx": list(CXX_BUILD_FLAGS),
        "nvcc": list(NVCC_BUILD_FLAGS),
    }
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "extension_abi_version": EXTENSION_ABI_VERSION,
        "extension_version": EXTENSION_VERSION,
        "generation_binding_source": source_records[0]["path"],
        "generation_binding_source_sha256": source_records[0]["sha256"],
        "sources": source_records,
        "compiled_source_snapshots": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in build_sources
        ],
        "source_snapshot_stable": True,
        "exported_functions": list(EXPORTED_FUNCTIONS),
        "capability_contract": dict(CUDA_ARCHITECTURE_B_CAPABILITY_CONTRACT),
        "capability_contract_sha256": _canonical_json_sha256(
            CUDA_ARCHITECTURE_B_CAPABILITY_CONTRACT
        ),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "numpy_version": str(np.__version__),
        "torch_file": str(Path(torch.__file__).resolve()),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "torch_include_paths": [str(path) for path in cpp_extension.include_paths()],
        "torch_library_paths": [str(path) for path in cpp_extension.library_paths()],
        "torch_lib_path": str(cpp_extension.TORCH_LIB_PATH),
        "cuda_home": str(cpp_extension.CUDA_HOME or ""),
        "cuda_device_index": args.device_index,
        "cuda_device_name": properties.name,
        "cuda_device_capability": [int(capability[0]), int(capability[1])],
        "cuda_arch": f"sm_{cuda_arch.replace('.', '')}",
        "torch_cuda_arch_list": cuda_arch,
        "cuda_device_total_memory_bytes": int(properties.total_memory),
        "compiler_command": compiler_command,
        "compiler_path": compiler_path,
        "compiler_version": _probe_version([*compiler_command, "--version"]),
        "nvcc_command": nvcc_command,
        "nvcc_path": nvcc_path,
        "nvcc_version": _probe_version([*nvcc_command, "--version"]),
        "build_flags": build_flags,
        "build_dir": str(build_dir),
        "built_path": str(built_path),
        "output": str(output),
        "output_sha256": _sha256(output),
        "stable_module_name": args.module_name,
        "with_cuda": True,
        "immutable_output": True,
        "prebuilt_outside_measured_execution": True,
        "runtime_jit_in_measured_execution": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
