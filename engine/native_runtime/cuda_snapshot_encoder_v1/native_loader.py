"""Fail-closed loader for the separately sealed native V1 encoder proof."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
from threading import RLock
from types import ModuleType
from typing import Any, Final

from .contract import (
    NATIVE_SNAPSHOT_ENCODER_ABI_V1,
    NATIVE_SNAPSHOT_ENCODER_BUILD_MANIFEST_V1,
    NATIVE_SNAPSHOT_ENCODER_EXTENSION_V1,
    REQUIRED_NATIVE_ENCODER_EXPORTS_V1,
)


NATIVE_SNAPSHOT_ENCODER_EXTENSION_ENV_V1: Final[str] = (
    "EIEL_NATIVE_SNAPSHOT_ENCODER_EXTENSION_V1"
)
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
BOUND_SOURCE_RELATIVE_PATHS_V1: Final[tuple[str, ...]] = (
    "engine/native_runtime/cuda_snapshot_encoder_v1/native/eiel_snapshot_encoder_v1.cpp",
    "engine/native_runtime/cuda_snapshot_encoder_v1/contract.py",
    "engine/native_runtime/cuda_snapshot_encoder_v1/native_loader.py",
    "engine/native_runtime/cuda_snapshot_encoder_v1/build.py",
    "engine/native_runtime/cuda_snapshot_encoder_v1/__init__.py",
)


class NativeSnapshotEncoderLoadError(RuntimeError):
    """The configured encoder artifact cannot be authenticated or loaded."""


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


@dataclass(frozen=True, slots=True)
class LoadedNativeSnapshotEncoderV1:
    module: ModuleType
    artifact_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


_LOAD_LOCK = RLock()
_LOADED_BY_PATH: dict[Path, LoadedNativeSnapshotEncoderV1] = {}
_LOADED_NAME_TO_PATH: dict[str, Path] = {}


def validate_native_snapshot_encoder_manifest_v1(
    path: str | os.PathLike[str],
) -> tuple[Path, Path, dict[str, Any]]:
    supplied_artifact = Path(path).absolute()
    if supplied_artifact.is_symlink() or not supplied_artifact.is_file():
        raise NativeSnapshotEncoderLoadError(
            "native snapshot encoder extension is absent or a symlink"
        )
    artifact = supplied_artifact.resolve()
    manifest_path = artifact.with_suffix(artifact.suffix + ".manifest.json")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise NativeSnapshotEncoderLoadError(
            "native snapshot encoder build manifest is absent"
        )
    try:
        decoded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NativeSnapshotEncoderLoadError(
            f"native snapshot encoder manifest is invalid: {exc}"
        ) from exc
    if not isinstance(decoded, dict):
        raise NativeSnapshotEncoderLoadError(
            "native snapshot encoder manifest is not an object"
        )
    manifest = dict(decoded)
    recorded_manifest_sha256 = manifest.pop("manifest_sha256", None)
    if (
        manifest.get("schema_version")
        != NATIVE_SNAPSHOT_ENCODER_BUILD_MANIFEST_V1
        or manifest.get("native_abi") != NATIVE_SNAPSHOT_ENCODER_ABI_V1
        or manifest.get("extension_version")
        != NATIVE_SNAPSHOT_ENCODER_EXTENSION_V1
        or manifest.get("runtime_jit") is not False
        or manifest.get("compiled_from_sealed_snapshot") is not True
        or manifest.get("execution_backend")
        != "HOST_CONTIGUOUS_SYNCHRONOUS_V1"
        or manifest.get("supports_cuda") is not False
        or manifest.get("filesystem_capability")
        != {
            "stage_file_output": True,
            "stage_file_open_policy": "EXCLUSIVE_CREATE_NO_OVERWRITE_V1",
            "stage_file_durability": "FILE_FSYNC_AND_CLOSE_V1",
            "owned_tensor_stage_method": True,
            "owned_tensor_native_work_releases_gil": True,
            "atomic_final_publication": False,
            "directory_fsync": False,
        }
        or recorded_manifest_sha256 != _canonical_sha256(manifest)
    ):
        raise NativeSnapshotEncoderLoadError(
            "native snapshot encoder build identity differs"
        )
    module_name = manifest.get("module_name")
    if (
        not isinstance(module_name, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", module_name) is None
    ):
        raise NativeSnapshotEncoderLoadError(
            "native snapshot encoder module name is invalid"
        )
    output = manifest.get("output")
    if (
        not isinstance(output, dict)
        or output.get("file_name") != artifact.name
        or output.get("bytes") != artifact.stat().st_size
        or output.get("sha256") != _sha256_file(artifact)
    ):
        raise NativeSnapshotEncoderLoadError(
            "native snapshot encoder artifact digest differs"
        )
    if manifest.get("exported_functions") != list(
        REQUIRED_NATIVE_ENCODER_EXPORTS_V1
    ):
        raise NativeSnapshotEncoderLoadError(
            "native snapshot encoder export contract differs"
        )
    dependencies = manifest.get("dependencies")
    zlib_identity = (
        dependencies.get("zlib") if isinstance(dependencies, dict) else None
    )
    if (
        not isinstance(zlib_identity, dict)
        or not isinstance(zlib_identity.get("compile_header_version"), str)
        or not zlib_identity.get("compile_header_version")
        or zlib_identity.get("compile_header_version")
        != zlib_identity.get("runtime_version")
        or not isinstance(zlib_identity.get("pkg_config_version"), str)
    ):
        raise NativeSnapshotEncoderLoadError(
            "native snapshot encoder zlib build identity differs"
        )
    inventory = manifest.get("source_inventory")
    if not isinstance(inventory, list) or [
        row.get("path") if isinstance(row, dict) else None for row in inventory
    ] != list(BOUND_SOURCE_RELATIVE_PATHS_V1):
        raise NativeSnapshotEncoderLoadError(
            "native snapshot encoder source inventory differs"
        )
    live_inventory: list[dict[str, str]] = []
    for row in inventory:
        assert isinstance(row, dict)
        relative = row.get("path")
        recorded_sha256 = row.get("sha256")
        source = PROJECT_ROOT / str(relative)
        if (
            not isinstance(relative, str)
            or not isinstance(recorded_sha256, str)
            or source.is_symlink()
            or not source.is_file()
            or _sha256_file(source) != recorded_sha256
        ):
            raise NativeSnapshotEncoderLoadError(
                f"native snapshot encoder bound source differs: {relative!r}"
            )
        live_inventory.append({"path": relative, "sha256": recorded_sha256})
    if manifest.get("source_inventory_sha256") != _canonical_sha256(
        live_inventory
    ):
        raise NativeSnapshotEncoderLoadError(
            "native snapshot encoder inventory identity differs"
        )
    manifest["manifest_sha256"] = recorded_manifest_sha256
    return artifact, manifest_path, manifest


def load_native_snapshot_encoder_v1(
    path: str | os.PathLike[str] | None = None,
) -> LoadedNativeSnapshotEncoderV1:
    raw = path or os.environ.get(NATIVE_SNAPSHOT_ENCODER_EXTENSION_ENV_V1)
    if not raw:
        raise NativeSnapshotEncoderLoadError(
            f"{NATIVE_SNAPSHOT_ENCODER_EXTENSION_ENV_V1} is not set and no path was supplied"
        )
    artifact, manifest_path, manifest = (
        validate_native_snapshot_encoder_manifest_v1(raw)
    )
    module_name = str(manifest["module_name"])
    with _LOAD_LOCK:
        cached = _LOADED_BY_PATH.get(artifact)
        if cached is not None:
            return cached
        claimed = _LOADED_NAME_TO_PATH.get(module_name)
        if claimed is not None and claimed != artifact:
            raise NativeSnapshotEncoderLoadError(
                f"native snapshot encoder module {module_name!r} is already loaded from {claimed}"
            )
        spec = importlib.util.spec_from_file_location(module_name, artifact)
        if spec is None or spec.loader is None:
            raise NativeSnapshotEncoderLoadError(
                "could not create native snapshot encoder module spec"
            )
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise NativeSnapshotEncoderLoadError(
                f"native snapshot encoder load failed: {exc}"
            ) from exc
        missing = [
            name
            for name in REQUIRED_NATIVE_ENCODER_EXPORTS_V1
            if not hasattr(module, name)
        ]
        if missing:
            raise NativeSnapshotEncoderLoadError(
                f"native snapshot encoder exports are missing: {missing}"
            )
        if (
            module.abi_version() != NATIVE_SNAPSHOT_ENCODER_ABI_V1
            or module.extension_version() != NATIVE_SNAPSHOT_ENCODER_EXTENSION_V1
            or not isinstance(getattr(module, "NativeSnapshotEncoderPlanV1", None), type)
        ):
            raise NativeSnapshotEncoderLoadError(
                "loaded native snapshot encoder ABI differs"
            )
        zlib_identity = manifest["dependencies"]["zlib"]
        if (
            module.zlib_compile_version()
            != zlib_identity["compile_header_version"]
            or module.zlib_runtime_version() != zlib_identity["runtime_version"]
        ):
            raise NativeSnapshotEncoderLoadError(
                "loaded native snapshot encoder zlib runtime differs"
            )
        loaded = LoadedNativeSnapshotEncoderV1(
            module=module,
            artifact_path=artifact,
            manifest_path=manifest_path,
            manifest=manifest,
        )
        _LOADED_BY_PATH[artifact] = loaded
        _LOADED_NAME_TO_PATH[module_name] = artifact
        return loaded


__all__ = [
    "BOUND_SOURCE_RELATIVE_PATHS_V1",
    "LoadedNativeSnapshotEncoderV1",
    "NATIVE_SNAPSHOT_ENCODER_EXTENSION_ENV_V1",
    "NativeSnapshotEncoderLoadError",
    "PROJECT_ROOT",
    "load_native_snapshot_encoder_v1",
    "validate_native_snapshot_encoder_manifest_v1",
]
