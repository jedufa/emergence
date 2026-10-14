"""Fail-closed loader for the independently built snapshot capsule extension."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
from threading import RLock
from types import ModuleType
from typing import Any, Final

from .format import (
    REQUIRED_NATIVE_EXPORTS_V1,
    SNAPSHOT_CAPSULE_ABI_V1,
    SNAPSHOT_EXTENSION_VERSION_V1,
    capsule_format_manifest_v1,
)


SNAPSHOT_BUILD_MANIFEST_SCHEMA_V1: Final[str] = (
    "eiel_native_snapshot_build_manifest_v1"
)
SNAPSHOT_EXTENSION_ENV_V1: Final[str] = "EIEL_NATIVE_SNAPSHOT_EXTENSION_V1"
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

BOUND_SOURCE_RELATIVE_PATHS_V1: Final[tuple[str, ...]] = (
    "engine/native_runtime/cuda_snapshot_v1/native/eiel_snapshot_capsule_v1.cpp",
    "engine/native_runtime/cuda_snapshot_v1/native/eiel_snapshot_projection_v1.hpp",
    "engine/native_runtime/cuda_snapshot_v1/format.py",
    "engine/native_runtime/cuda_snapshot_v1/native_loader.py",
    "engine/native_runtime/cuda_snapshot_v1/build.py",
    "engine/native_runtime/cuda_snapshot_v1/__init__.py",
)


class NativeSnapshotLoadError(RuntimeError):
    """The configured native snapshot artifact cannot be trusted or loaded."""


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
class LoadedNativeSnapshotExtensionV1:
    module: ModuleType
    artifact_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


_LOADED_BY_PATH: dict[Path, LoadedNativeSnapshotExtensionV1] = {}
_LOADED_NAME_TO_PATH: dict[str, Path] = {}
_LOAD_LOCK = RLock()


def validate_snapshot_extension_manifest_v1(
    path: str | os.PathLike[str],
) -> tuple[Path, Path, dict[str, Any]]:
    """Validate artifact, manifest, exports contract, and bound live sources."""

    supplied_artifact = Path(path).absolute()
    if supplied_artifact.is_symlink() or not supplied_artifact.is_file():
        raise NativeSnapshotLoadError("native snapshot extension is absent or a symlink")
    artifact = supplied_artifact.resolve()
    manifest_path = artifact.with_suffix(artifact.suffix + ".manifest.json")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise NativeSnapshotLoadError("native snapshot extension manifest is absent")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NativeSnapshotLoadError(
            f"native snapshot extension manifest is invalid: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise NativeSnapshotLoadError("native snapshot build manifest is not an object")
    manifest = dict(value)
    recorded_manifest_sha = manifest.pop("manifest_sha256", None)
    if (
        manifest.get("schema_version") != SNAPSHOT_BUILD_MANIFEST_SCHEMA_V1
        or manifest.get("native_abi") != SNAPSHOT_CAPSULE_ABI_V1
        or manifest.get("extension_version") != SNAPSHOT_EXTENSION_VERSION_V1
        or manifest.get("runtime_jit") is not False
        or manifest.get("compiled_from_sealed_snapshot") is not True
    ):
        raise NativeSnapshotLoadError("native snapshot build identity differs")
    if recorded_manifest_sha != _canonical_sha256(manifest):
        raise NativeSnapshotLoadError("native snapshot build manifest digest differs")
    module_name = manifest.get("module_name")
    if (
        not isinstance(module_name, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", module_name) is None
    ):
        raise NativeSnapshotLoadError("native snapshot module name is invalid")
    output = manifest.get("output")
    if (
        not isinstance(output, dict)
        or output.get("file_name") != artifact.name
        or output.get("bytes") != artifact.stat().st_size
        or output.get("sha256") != _sha256_file(artifact)
    ):
        raise NativeSnapshotLoadError("native snapshot extension artifact digest differs")
    if manifest.get("exported_functions") != list(REQUIRED_NATIVE_EXPORTS_V1):
        raise NativeSnapshotLoadError("native snapshot extension export contract differs")
    if manifest.get("capsule_format") != capsule_format_manifest_v1():
        raise NativeSnapshotLoadError("native snapshot capsule format manifest differs")
    openssl_identity = manifest.get("openssl")
    if (
        not isinstance(openssl_identity, dict)
        or not isinstance(openssl_identity.get("pkg_config"), str)
        or not openssl_identity.get("pkg_config")
        or openssl_identity.get("link_flags") != ["-lcrypto"]
    ):
        raise NativeSnapshotLoadError("native snapshot OpenSSL build identity differs")
    inventory = manifest.get("source_inventory")
    if not isinstance(inventory, list) or [
        row.get("path") if isinstance(row, dict) else None for row in inventory
    ] != list(BOUND_SOURCE_RELATIVE_PATHS_V1):
        raise NativeSnapshotLoadError("native snapshot source inventory differs")
    expected_inventory: list[dict[str, str]] = []
    for row in inventory:
        assert isinstance(row, dict)
        relative = row.get("path")
        expected_sha = row.get("sha256")
        source = PROJECT_ROOT / str(relative)
        if (
            not isinstance(relative, str)
            or not isinstance(expected_sha, str)
            or source.is_symlink()
            or not source.is_file()
            or _sha256_file(source) != expected_sha
        ):
            raise NativeSnapshotLoadError(
                f"native snapshot bound source differs: {relative!r}"
            )
        expected_inventory.append({"path": relative, "sha256": expected_sha})
    source_identity = _canonical_sha256(expected_inventory)
    if (
        manifest.get("source_inventory_sha256") != source_identity
        or module_name != f"eiel_native_snapshot_v1_{source_identity[:16]}"
    ):
        raise NativeSnapshotLoadError("native snapshot source inventory digest differs")
    manifest["manifest_sha256"] = recorded_manifest_sha
    return artifact, manifest_path, manifest


def load_prebuilt_snapshot_extension_v1(
    path: str | os.PathLike[str] | None = None,
) -> LoadedNativeSnapshotExtensionV1:
    """Load one exact prebuilt extension; runtime JIT is never attempted."""

    raw = path or os.environ.get(SNAPSHOT_EXTENSION_ENV_V1)
    if not raw:
        raise NativeSnapshotLoadError(
            f"{SNAPSHOT_EXTENSION_ENV_V1} is not set and no extension path was supplied"
        )
    artifact, manifest_path, manifest = validate_snapshot_extension_manifest_v1(raw)
    module_name = str(manifest["module_name"])
    with _LOAD_LOCK:
        cached = _LOADED_BY_PATH.get(artifact)
        if cached is not None:
            return cached
        claimed = _LOADED_NAME_TO_PATH.get(module_name)
        if claimed is not None and claimed != artifact:
            raise NativeSnapshotLoadError(
                f"native snapshot module {module_name!r} is already loaded from {claimed}"
            )
        spec = importlib.util.spec_from_file_location(module_name, artifact)
        if spec is None or spec.loader is None:
            raise NativeSnapshotLoadError("could not create native snapshot module spec")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise NativeSnapshotLoadError(
                f"native snapshot extension load failed: {exc}"
            ) from exc
        missing = [name for name in REQUIRED_NATIVE_EXPORTS_V1 if not hasattr(module, name)]
        if missing:
            raise NativeSnapshotLoadError(
                f"native snapshot extension exports are missing: {missing}"
            )
        if (
            module.abi_version() != SNAPSHOT_CAPSULE_ABI_V1
            or module.extension_version() != SNAPSHOT_EXTENSION_VERSION_V1
            or module.capsule_format_manifest_v1() != capsule_format_manifest_v1()
            or module.supports_cuda_projection_v1() is not False
        ):
            raise NativeSnapshotLoadError("loaded native snapshot ABI differs")
        loaded = LoadedNativeSnapshotExtensionV1(
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
    "LoadedNativeSnapshotExtensionV1",
    "NativeSnapshotLoadError",
    "PROJECT_ROOT",
    "SNAPSHOT_BUILD_MANIFEST_SCHEMA_V1",
    "SNAPSHOT_EXTENSION_ENV_V1",
    "load_prebuilt_snapshot_extension_v1",
    "validate_snapshot_extension_manifest_v1",
]
