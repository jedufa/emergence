"""Explicit immutable build for the G2 native snapshot capsule extension."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Iterator, Sequence

import torch
from torch.utils.cpp_extension import load

from .format import (
    REQUIRED_NATIVE_EXPORTS_V1,
    SNAPSHOT_CAPSULE_ABI_V1,
    SNAPSHOT_EXTENSION_VERSION_V1,
    capsule_format_manifest_v1,
)
from .native_loader import (
    BOUND_SOURCE_RELATIVE_PATHS_V1,
    PROJECT_ROOT,
    SNAPSHOT_BUILD_MANIFEST_SCHEMA_V1,
)


class NativeSnapshotBuildError(RuntimeError):
    """The independently sealed native snapshot extension could not be built."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _tool_identity(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{exc}"
    return (result.stdout or result.stderr).strip()


@contextmanager
def _temporary_environment(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_native_snapshot_extension_v1(output_directory: str | Path) -> Path:
    """Compile from a sealed source copy and publish one immutable artifact."""

    source_paths = tuple(PROJECT_ROOT / item for item in BOUND_SOURCE_RELATIVE_PATHS_V1)
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise NativeSnapshotBuildError(
            f"native snapshot bound source inventory is incomplete: {missing}"
        )
    source_inventory = [
        {"path": relative, "sha256": _sha256_file(path)}
        for relative, path in zip(
            BOUND_SOURCE_RELATIVE_PATHS_V1, source_paths, strict=True
        )
    ]
    source_identity = _canonical_sha256(source_inventory)
    module_name = f"eiel_native_snapshot_v1_{source_identity[:16]}"
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise NativeSnapshotBuildError(
            "native snapshot immutable output directory must be empty"
        )
    with tempfile.TemporaryDirectory(
        prefix=".eiel-native-snapshot-v1-build-",
        dir=output.parent,
    ) as raw:
        scratch = Path(raw)
        snapshot_root = scratch / "sealed-source"
        build_root = scratch / "build"
        build_root.mkdir()
        for row, source in zip(source_inventory, source_paths, strict=True):
            destination = snapshot_root / str(row["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            destination.chmod(0o444)
        copied_inventory = [
            {
                "path": relative,
                "sha256": _sha256_file(snapshot_root / relative),
            }
            for relative in BOUND_SOURCE_RELATIVE_PATHS_V1
        ]
        if copied_inventory != source_inventory:
            raise NativeSnapshotBuildError("sealed snapshot source differs")
        compile_source = snapshot_root / BOUND_SOURCE_RELATIVE_PATHS_V1[0]
        try:
            with _temporary_environment(
                {
                    "MAX_JOBS": "1",
                    "TMPDIR": str(build_root),
                    "TEMP": str(build_root),
                    "TMP": str(build_root),
                }
            ):
                module = load(
                    name=module_name,
                    sources=[str(compile_source)],
                    extra_cflags=["-O3", "-std=c++17"],
                    extra_ldflags=["-lz", "-lcrypto"],
                    with_cuda=False,
                    build_directory=str(build_root),
                    verbose=False,
                )
        except Exception as exc:
            raise NativeSnapshotBuildError(
                f"native snapshot extension compilation failed: {exc}"
            ) from exc
        missing_exports = [
            name for name in REQUIRED_NATIVE_EXPORTS_V1 if not hasattr(module, name)
        ]
        if missing_exports:
            raise NativeSnapshotBuildError(
                f"native snapshot extension is missing exports: {missing_exports}"
            )
        if not isinstance(getattr(module, "NativeSnapshotPlanV1", None), type):
            raise NativeSnapshotBuildError(
                "native snapshot extension is missing NativeSnapshotPlanV1"
            )
        if (
            module.abi_version() != SNAPSHOT_CAPSULE_ABI_V1
            or module.extension_version() != SNAPSHOT_EXTENSION_VERSION_V1
            or module.capsule_format_manifest_v1() != capsule_format_manifest_v1()
            or module.supports_cuda_projection_v1() is not False
        ):
            raise NativeSnapshotBuildError("compiled native snapshot ABI differs")
        zlib_compile_version = module.zlib_compile_version()
        zlib_runtime_version = module.zlib_runtime_version()
        if (
            not isinstance(zlib_compile_version, str)
            or not zlib_compile_version
            or zlib_compile_version != zlib_runtime_version
        ):
            raise NativeSnapshotBuildError(
                "native snapshot zlib header/runtime versions differ"
            )
        live_inventory = [
            {"path": relative, "sha256": _sha256_file(path)}
            for relative, path in zip(
                BOUND_SOURCE_RELATIVE_PATHS_V1, source_paths, strict=True
            )
        ]
        if live_inventory != source_inventory:
            raise NativeSnapshotBuildError(
                "native snapshot live source changed during compilation"
            )
        built = Path(module.__file__).resolve()
        artifact = output / built.name
        shutil.copy2(built, artifact)
    artifact.chmod(0o444)
    manifest: dict[str, object] = {
        "schema_version": SNAPSHOT_BUILD_MANIFEST_SCHEMA_V1,
        "native_abi": SNAPSHOT_CAPSULE_ABI_V1,
        "extension_version": SNAPSHOT_EXTENSION_VERSION_V1,
        "module_name": module_name,
        "immutable_output": True,
        "runtime_jit": False,
        "compiled_from_sealed_snapshot": True,
        "projection_backend": "HOST_CONTIGUOUS_SYNC_V1",
        "supports_cuda_projection": False,
        "source_inventory": source_inventory,
        "source_inventory_sha256": source_identity,
        "exported_functions": list(REQUIRED_NATIVE_EXPORTS_V1),
        "native_plan_class": "NativeSnapshotPlanV1",
        "capsule_format": capsule_format_manifest_v1(),
        "toolchain": {
            "cxx": _tool_identity((os.environ.get("CXX", "c++"), "--version")),
            "cxx_flags": ["-O3", "-std=c++17"],
            "torch": str(torch.__version__),
            "with_cuda": False,
        },
        "zlib": {
            "compile_header_version": zlib_compile_version,
            "runtime_version": zlib_runtime_version,
            "pkg_config": _tool_identity(("pkg-config", "--modversion", "zlib")),
        },
        "openssl": {
            "pkg_config": _tool_identity(("pkg-config", "--modversion", "openssl")),
            "link_flags": ["-lcrypto"],
        },
        "python": {
            "cache_tag": sys.implementation.cache_tag,
            "version": platform.python_version(),
            "platform": platform.platform(),
        },
        "output": {
            "file_name": artifact.name,
            "bytes": artifact.stat().st_size,
            "sha256": _sha256_file(artifact),
        },
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    manifest_path = artifact.with_suffix(artifact.suffix + ".manifest.json")
    temporary = output / f".{manifest_path.name}.tmp"
    temporary.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    manifest_path.chmod(0o444)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(build_native_snapshot_extension_v1(args.output_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover - explicit operator entry point
    raise SystemExit(main())


__all__ = ["NativeSnapshotBuildError", "build_native_snapshot_extension_v1"]
