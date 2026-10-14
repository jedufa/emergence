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

import numpy as np
import torch
import torch.utils.cpp_extension as cpp_extension
from torch.utils.cpp_extension import load


EXTENSION_ABI_VERSION = "eiel_native_runtime_abi_v1"
EXTENSION_VERSION = "eiel_native_runtime_flat_v1"
ROOT = Path(__file__).resolve().parents[2]
NATIVE_RUNTIME_BINDING_SOURCE = (
    ROOT / "engine" / "native_runtime" / "native" / "eiel_native_runtime.cpp"
)
RESIDENT_OWNER_SOURCE = (
    ROOT / "engine" / "perf" / "native" / "cxxbackbone10_resident_state_owner.cpp"
)
CPU_COMPACT_SUMMARY_SOURCE = (
    ROOT / "engine" / "perf" / "native" / "cpu_compact_summary_v1.cpp"
)
CPU_COMPACT_SUMMARY_DIGESTS_SOURCE = (
    ROOT / "engine" / "perf" / "native" / "cpu_compact_summary_digests_v1.cpp"
)
CPU_COMPACT_SUMMARY_TORCH_ADAPTER_SOURCE = (
    ROOT / "engine" / "perf" / "native" / "cpu_compact_summary_torch_adapter_v1.cpp"
)
CPU_COMPACT_SUMMARY_LIFECYCLE_REDUCTION_SOURCE = (
    ROOT
    / "engine"
    / "perf"
    / "native"
    / "cpu_compact_summary_lifecycle_reduction_v1.cpp"
)
CPU_COMPACT_SUMMARY_IDENTITY_REDUCTION_SOURCE = (
    ROOT
    / "engine"
    / "perf"
    / "native"
    / "cpu_compact_summary_identity_reduction_v1.cpp"
)
CPU_COMPACT_SUMMARY_REPORTING_LEDGER_SOURCE = (
    ROOT
    / "engine"
    / "perf"
    / "native"
    / "cpu_compact_summary_reporting_ledger_v1.cpp"
)
CPU_COMPACT_SUMMARY_REPORTING_LEDGER_HEADER = (
    ROOT
    / "engine"
    / "perf"
    / "native"
    / "cpu_compact_summary_reporting_ledger_v1.hpp"
)
CPU_COMPACT_SUMMARY_SOURCE_COMPOSER_SOURCE = (
    ROOT
    / "engine"
    / "perf"
    / "native"
    / "cpu_compact_summary_source_composer_v1.cpp"
)
CPU_COMPACT_SUMMARY_HOST_CANONICAL_SOURCE = (
    ROOT
    / "engine"
    / "perf"
    / "cuda_architecture_b_compact_summary_host_canonical_v1.cpp"
)
WDL_NATIVE_CONTRACT_SOURCE = (
    ROOT / "engine" / "native_runtime" / "wdl" / "wdl_native_contract.cpp"
)
STABLE_GRU_SOURCE = (
    ROOT / "engine" / "perf" / "native" / "recurrent" / "stable_gru_v1.cpp"
)
ACS_NATIVE_STREAMED_CHECKPOINT_SOURCE = (
    ROOT
    / "engine"
    / "native_runtime"
    / "native"
    / "acs_native_streamed_checkpoint_v1.cpp"
)
ACS_NATIVE_STREAMED_CHECKPOINT_HEADER = (
    ROOT
    / "engine"
    / "native_runtime"
    / "native"
    / "acs_native_streamed_checkpoint_v1.hpp"
)
NATIVE_RUNTIME_COMPILE_SOURCES = (
    NATIVE_RUNTIME_BINDING_SOURCE,
    CPU_COMPACT_SUMMARY_SOURCE,
    CPU_COMPACT_SUMMARY_DIGESTS_SOURCE,
    CPU_COMPACT_SUMMARY_TORCH_ADAPTER_SOURCE,
    CPU_COMPACT_SUMMARY_LIFECYCLE_REDUCTION_SOURCE,
    CPU_COMPACT_SUMMARY_IDENTITY_REDUCTION_SOURCE,
    CPU_COMPACT_SUMMARY_REPORTING_LEDGER_SOURCE,
    CPU_COMPACT_SUMMARY_SOURCE_COMPOSER_SOURCE,
    CPU_COMPACT_SUMMARY_HOST_CANONICAL_SOURCE,
    ACS_NATIVE_STREAMED_CHECKPOINT_SOURCE,
    WDL_NATIVE_CONTRACT_SOURCE,
    STABLE_GRU_SOURCE,
)
NATIVE_RUNTIME_SOURCE_INVENTORY = (
    NATIVE_RUNTIME_BINDING_SOURCE,
    RESIDENT_OWNER_SOURCE,
    CPU_COMPACT_SUMMARY_SOURCE,
    CPU_COMPACT_SUMMARY_DIGESTS_SOURCE,
    CPU_COMPACT_SUMMARY_TORCH_ADAPTER_SOURCE,
    CPU_COMPACT_SUMMARY_LIFECYCLE_REDUCTION_SOURCE,
    CPU_COMPACT_SUMMARY_IDENTITY_REDUCTION_SOURCE,
    CPU_COMPACT_SUMMARY_REPORTING_LEDGER_SOURCE,
    CPU_COMPACT_SUMMARY_REPORTING_LEDGER_HEADER,
    CPU_COMPACT_SUMMARY_SOURCE_COMPOSER_SOURCE,
    CPU_COMPACT_SUMMARY_HOST_CANONICAL_SOURCE,
    ACS_NATIVE_STREAMED_CHECKPOINT_SOURCE,
    ACS_NATIVE_STREAMED_CHECKPOINT_HEADER,
    WDL_NATIVE_CONTRACT_SOURCE,
    STABLE_GRU_SOURCE,
)
RESIDENT_OWNER_SOURCE_INVENTORY = (
    RESIDENT_OWNER_SOURCE,
    CPU_COMPACT_SUMMARY_SOURCE,
    CPU_COMPACT_SUMMARY_DIGESTS_SOURCE,
    CPU_COMPACT_SUMMARY_TORCH_ADAPTER_SOURCE,
    CPU_COMPACT_SUMMARY_LIFECYCLE_REDUCTION_SOURCE,
    CPU_COMPACT_SUMMARY_IDENTITY_REDUCTION_SOURCE,
    CPU_COMPACT_SUMMARY_REPORTING_LEDGER_SOURCE,
    CPU_COMPACT_SUMMARY_SOURCE_COMPOSER_SOURCE,
    CPU_COMPACT_SUMMARY_HOST_CANONICAL_SOURCE,
    ACS_NATIVE_STREAMED_CHECKPOINT_SOURCE,
    WDL_NATIVE_CONTRACT_SOURCE,
    STABLE_GRU_SOURCE,
    CPU_COMPACT_SUMMARY_REPORTING_LEDGER_HEADER,
    ACS_NATIVE_STREAMED_CHECKPOINT_HEADER,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_records(paths: tuple[Path, ...]) -> list[dict[str, str]]:
    return [
        {"path": str(path.resolve()), "sha256": _sha256(path)}
        for path in paths
    ]


def _source_fingerprint(records: list[dict[str, str]]) -> str:
    payload = json.dumps(records, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_source_records_unchanged(
    paths: tuple[Path, ...],
    expected: list[dict[str, str]],
) -> None:
    current = _source_records(paths)
    if current != expected:
        raise RuntimeError("native extension sources changed during build")


def _load_native_runtime_extension(
    *,
    module_name: str,
    build_dir: Path,
    numpy_random_lib: Path,
) -> object:
    return load(
        name=module_name,
        sources=[str(path) for path in NATIVE_RUNTIME_COMPILE_SOURCES],
        extra_include_paths=[str(np.get_include())],
        extra_cflags=["-O3"],
        extra_ldflags=[str(numpy_random_lib)] if numpy_random_lib.exists() else [],
        with_cuda=False,
        build_directory=str(build_dir),
        verbose=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the EIEL native runtime extension outside measured execution.")
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--module-name", default="eiel_native_runtime_ext")
    args = parser.parse_args()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.module_name) is None:
        parser.error("--module-name must be a valid Python extension module identifier")

    numpy_random_lib = Path(np.__file__).resolve().parent / "random" / "lib" / "libnpyrandom.a"
    build_dir = Path(args.build_dir).resolve()
    output = Path(args.output).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    compile_source_records = _source_records(NATIVE_RUNTIME_COMPILE_SOURCES)
    source_inventory_records = _source_records(NATIVE_RUNTIME_SOURCE_INVENTORY)
    resident_owner_source_inventory_records = _source_records(
        RESIDENT_OWNER_SOURCE_INVENTORY
    )
    module = _load_native_runtime_extension(
        module_name=args.module_name,
        build_dir=build_dir,
        numpy_random_lib=numpy_random_lib,
    )
    built_path = Path(module.__file__).resolve()
    if built_path != output:
        shutil.copy2(built_path, output)
    build_ninja = build_dir / "build.ninja"
    torch_lib_path = str(cpp_extension.TORCH_LIB_PATH)
    torch_include_paths = [str(path) for path in cpp_extension.include_paths()]
    compiler_command = shlex.split(os.environ.get("CXX", "c++"))
    compiler_path = shutil.which(compiler_command[0]) or compiler_command[0]
    compiler_probe = subprocess.run(
        [*compiler_command, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    if build_ninja.exists():
        build_text = build_ninja.read_text(encoding="utf-8", errors="replace")
        if torch_lib_path not in build_text:
            raise RuntimeError(
                "native extension build used a torch library path outside the active Python torch package: "
                f"expected {torch_lib_path}"
            )
        missing_includes = [path for path in torch_include_paths if path not in build_text]
        if missing_includes:
            raise RuntimeError(
                "native extension build used torch includes outside the active Python torch package: "
                f"missing {missing_includes}"
            )
        missing_sources = [
            str(path) for path in NATIVE_RUNTIME_COMPILE_SOURCES
            if str(path) not in build_text
        ]
        if missing_sources:
            raise RuntimeError(
                "native extension build omitted required compile sources: "
                f"{missing_sources}"
            )
    _require_source_records_unchanged(
        NATIVE_RUNTIME_SOURCE_INVENTORY,
        source_inventory_records,
    )
    manifest = {
        "schema_version": "eiel_native_runtime_build_manifest_v1",
        "extension_abi_version": EXTENSION_ABI_VERSION,
        "extension_version": EXTENSION_VERSION,
        "source": str(NATIVE_RUNTIME_BINDING_SOURCE),
        "source_sha256": _sha256(NATIVE_RUNTIME_BINDING_SOURCE),
        "resident_owner_source": str(RESIDENT_OWNER_SOURCE),
        "resident_owner_source_sha256": _sha256(RESIDENT_OWNER_SOURCE),
        "cpu_compact_summary_source": str(CPU_COMPACT_SUMMARY_SOURCE),
        "cpu_compact_summary_source_sha256": _sha256(CPU_COMPACT_SUMMARY_SOURCE),
        "cpu_compact_summary_digests_source": str(
            CPU_COMPACT_SUMMARY_DIGESTS_SOURCE
        ),
        "cpu_compact_summary_digests_source_sha256": _sha256(
            CPU_COMPACT_SUMMARY_DIGESTS_SOURCE
        ),
        "cpu_compact_summary_torch_adapter_source": str(
            CPU_COMPACT_SUMMARY_TORCH_ADAPTER_SOURCE
        ),
        "cpu_compact_summary_torch_adapter_source_sha256": _sha256(
            CPU_COMPACT_SUMMARY_TORCH_ADAPTER_SOURCE
        ),
        "cpu_compact_summary_lifecycle_reduction_source": str(
            CPU_COMPACT_SUMMARY_LIFECYCLE_REDUCTION_SOURCE
        ),
        "cpu_compact_summary_lifecycle_reduction_source_sha256": _sha256(
            CPU_COMPACT_SUMMARY_LIFECYCLE_REDUCTION_SOURCE
        ),
        "cpu_compact_summary_identity_reduction_source": str(
            CPU_COMPACT_SUMMARY_IDENTITY_REDUCTION_SOURCE
        ),
        "cpu_compact_summary_identity_reduction_source_sha256": _sha256(
            CPU_COMPACT_SUMMARY_IDENTITY_REDUCTION_SOURCE
        ),
        "cpu_compact_summary_reporting_ledger_source": str(
            CPU_COMPACT_SUMMARY_REPORTING_LEDGER_SOURCE
        ),
        "cpu_compact_summary_reporting_ledger_source_sha256": _sha256(
            CPU_COMPACT_SUMMARY_REPORTING_LEDGER_SOURCE
        ),
        "cpu_compact_summary_reporting_ledger_header": str(
            CPU_COMPACT_SUMMARY_REPORTING_LEDGER_HEADER
        ),
        "cpu_compact_summary_reporting_ledger_header_sha256": _sha256(
            CPU_COMPACT_SUMMARY_REPORTING_LEDGER_HEADER
        ),
        "cpu_compact_summary_source_composer_source": str(
            CPU_COMPACT_SUMMARY_SOURCE_COMPOSER_SOURCE
        ),
        "cpu_compact_summary_source_composer_source_sha256": _sha256(
            CPU_COMPACT_SUMMARY_SOURCE_COMPOSER_SOURCE
        ),
        "cpu_compact_summary_host_canonical_source": str(
            CPU_COMPACT_SUMMARY_HOST_CANONICAL_SOURCE
        ),
        "cpu_compact_summary_host_canonical_source_sha256": _sha256(
            CPU_COMPACT_SUMMARY_HOST_CANONICAL_SOURCE
        ),
        "wdl_native_contract_source": str(WDL_NATIVE_CONTRACT_SOURCE),
        "wdl_native_contract_source_sha256": _sha256(WDL_NATIVE_CONTRACT_SOURCE),
        "compile_sources": compile_source_records,
        "source_inventory": source_inventory_records,
        "resident_owner_source_inventory": resident_owner_source_inventory_records,
        "compile_source_fingerprint_sha256": _source_fingerprint(compile_source_records),
        "source_inventory_fingerprint_sha256": _source_fingerprint(
            resident_owner_source_inventory_records
        ),
        "native_runtime_source_inventory_fingerprint_sha256": _source_fingerprint(
            source_inventory_records
        ),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "torch_file": str(Path(torch.__file__).resolve()),
        "torch_version": str(torch.__version__),
        "torch_include_paths": torch_include_paths,
        "torch_library_paths": [str(path) for path in cpp_extension.library_paths()],
        "torch_lib_path": torch_lib_path,
        "numpy_include": str(np.get_include()),
        "numpy_file": str(Path(np.__file__).resolve()),
        "numpy_version": str(np.__version__),
        "numpy_random_lib": str(numpy_random_lib) if numpy_random_lib.exists() else "",
        "compiler_command": compiler_command,
        "compiler_path": compiler_path,
        "compiler_version": compiler_probe.stdout.strip(),
        "build_dir": str(build_dir),
        "built_path": str(built_path),
        "output": str(output),
        "output_sha256": _sha256(output),
        "stable_module_name": args.module_name,
        "runtime_jit_in_measured_execution": False,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
