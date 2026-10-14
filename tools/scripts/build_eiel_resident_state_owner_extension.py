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
from typing import Any

import torch.utils.cpp_extension as cpp_extension
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "engine") not in sys.path:
    sys.path.insert(0, str(ROOT / "engine"))

from perf import cxxbackbone10_resident_state_owner as development_loader


SCHEMA_VERSION = "cxxbackbone10_resident_owner_extension_manifest_v1"
EXTENSION_ABI_VERSION = "eiel_native_runtime_abi_v1"
EXTENSION_VERSION = "eiel_native_runtime_flat_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _load_extension(
    *,
    module_name: str,
    build_dir: Path,
) -> object:
    return load(
        name=module_name,
        sources=[str(path) for path in development_loader.NATIVE_EXTENSION_SOURCES],
        extra_include_paths=[str(development_loader.NUMPY_INCLUDE)],
        build_directory=str(build_dir),
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_ldflags=(
            [str(development_loader.NUMPY_RANDOM_LIB)]
            if development_loader.NUMPY_RANDOM_LIB.exists()
            else []
        ),
        with_cuda=False,
        verbose=False,
    )


def _validate_build_ninja(build_dir: Path) -> dict[str, Any]:
    build_ninja = build_dir / "build.ninja"
    torch_lib_path = str(cpp_extension.TORCH_LIB_PATH)
    torch_include_paths = [str(path) for path in cpp_extension.include_paths()]
    if not build_ninja.exists():
        return {
            "build_ninja_path": str(build_ninja),
            "build_ninja_verified": False,
            "build_ninja_warning": "build.ninja was not present after torch extension build",
        }
    build_text = build_ninja.read_text(encoding="utf-8", errors="replace")
    if torch_lib_path not in build_text:
        raise RuntimeError(
            "resident owner extension build used a torch library path outside "
            f"the active Python torch package: expected {torch_lib_path}"
        )
    missing_includes = [path for path in torch_include_paths if path not in build_text]
    if missing_includes:
        raise RuntimeError(
            "resident owner extension build used torch includes outside the active "
            f"Python torch package: missing {missing_includes}"
        )
    missing_sources = [
        str(path)
        for path in development_loader.NATIVE_EXTENSION_SOURCES
        if str(path) not in build_text
    ]
    if missing_sources:
        raise RuntimeError(
            "resident owner extension build omitted required compile sources: "
            f"{missing_sources}"
        )
    return {
        "build_ninja_path": str(build_ninja),
        "build_ninja_verified": True,
        "torch_lib_path": torch_lib_path,
        "torch_include_paths": torch_include_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the EIEL resident-state-owner native extension outside measured "
            "CPU production execution."
        )
    )
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--module-name",
        default=development_loader.RESIDENT_STATE_OWNER_STABLE_MODULE_NAME,
    )
    args = parser.parse_args()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.module_name) is None:
        parser.error("--module-name must be a valid Python extension module identifier")

    build_dir = Path(args.build_dir).resolve()
    output = Path(args.output).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_manifest = development_loader.native_extension_compile_source_manifest()
    module = _load_extension(module_name=args.module_name, build_dir=build_dir)
    built_path = Path(module.__file__).resolve()
    if built_path != output:
        shutil.copy2(built_path, output)
    current_manifest = development_loader.native_extension_compile_source_manifest()
    if current_manifest != source_manifest:
        raise RuntimeError("resident owner extension sources changed during build")

    compiler_command = shlex.split(os.environ.get("CXX", "c++"))
    compiler_path = shutil.which(compiler_command[0]) or compiler_command[0]
    compiler_probe = subprocess.run(
        [*compiler_command, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    output_sha256 = _sha256(output)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stable_module_name": args.module_name,
        "extension_abi_version": EXTENSION_ABI_VERSION,
        "extension_version": EXTENSION_VERSION,
        "source_inventory_fingerprint_sha256": source_manifest[
            "compile_source_fingerprint_sha256"
        ],
        "sources": source_manifest["sources"],
        "output": str(output),
        "output_sha256": output_sha256,
        "runtime_jit_in_measured_execution": False,
        "with_cuda": False,
        "compiler": {
            "command": compiler_command,
            "path": compiler_path,
            "version": compiler_probe.stdout.strip(),
        },
        "torch": {
            "version": torch_version(),
        },
        **_validate_build_ninja(build_dir),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": output_sha256,
                "manifest_path": str(manifest_path),
                "source_inventory_fingerprint_sha256": manifest[
                    "source_inventory_fingerprint_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


def torch_version() -> str:
    import torch

    return str(torch.__version__)


if __name__ == "__main__":
    raise SystemExit(main())
