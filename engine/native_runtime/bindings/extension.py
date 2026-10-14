from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
from threading import RLock
from types import ModuleType
from typing import Any

from ..types import ABI_VERSION, EXTENSION_VERSION


ENV_EXTENSION_PATH = "EIEL_NATIVE_RUNTIME_EXTENSION"
STABLE_MODULE_NAME = "eiel_native_runtime_ext"
_PREBUILT_MODULES: dict[str, ModuleType] = {}
_PREBUILT_MODULES_LOCK = RLock()


@dataclass(frozen=True)
class NativeExtensionStatus:
    schema_version: str = "eiel_native_runtime_extension_status_v1"
    stable_module_name: str = STABLE_MODULE_NAME
    env_var: str = ENV_EXTENSION_PATH
    extension_path: str = ""
    manifest_path: str = ""
    extension_abi_version: str = ""
    extension_version: str = ""
    abi_compatible: bool = False
    output_sha256_verified: bool = False
    available: bool = False
    runtime_jit: bool = False
    loaded: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def native_extension_status(path: str | os.PathLike[str] | None = None) -> NativeExtensionStatus:
    raw = str(path or os.environ.get(ENV_EXTENSION_PATH, "") or "")
    if not raw:
        return NativeExtensionStatus(error=f"{ENV_EXTENSION_PATH} is not set")
    p = Path(raw)
    if not p.exists():
        return NativeExtensionStatus(extension_path=str(p), error="configured extension path does not exist")
    if not p.is_file():
        return NativeExtensionStatus(extension_path=str(p), error="configured extension path is not a file")
    p = p.resolve(strict=True)
    manifest_path = p.with_suffix(p.suffix + ".manifest.json")
    if not manifest_path.is_file():
        return NativeExtensionStatus(
            extension_path=str(p),
            manifest_path=str(manifest_path),
            error="configured extension build manifest is missing",
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        return NativeExtensionStatus(
            extension_path=str(p),
            manifest_path=str(manifest_path),
            error=f"configured extension build manifest is invalid: {exc}",
        )
    stable_module_name = str(manifest.get("stable_module_name", "") or "")
    extension_abi_version = str(manifest.get("extension_abi_version", "") or "")
    extension_version = str(manifest.get("extension_version", "") or "")
    abi_compatible = extension_abi_version == ABI_VERSION and extension_version == EXTENSION_VERSION
    expected_sha256 = str(manifest.get("output_sha256", "") or "")
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    output_sha256_verified = bool(expected_sha256) and digest == expected_sha256
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", stable_module_name) is None:
        error = "configured extension build manifest has an invalid stable module name"
    elif not abi_compatible:
        error = (
            "configured extension ABI/version mismatch: "
            f"abi={extension_abi_version or '<missing>'} version={extension_version or '<missing>'}"
        )
    elif not output_sha256_verified:
        error = "configured extension SHA-256 does not match its build manifest"
    else:
        error = ""
    with _PREBUILT_MODULES_LOCK:
        loaded = str(p) in _PREBUILT_MODULES
    return NativeExtensionStatus(
        stable_module_name=stable_module_name,
        extension_path=str(p),
        manifest_path=str(manifest_path),
        extension_abi_version=extension_abi_version,
        extension_version=extension_version,
        abi_compatible=abi_compatible,
        output_sha256_verified=output_sha256_verified,
        available=not error,
        loaded=loaded,
        error=error,
    )


def load_prebuilt_extension(path: str | os.PathLike[str] | None = None) -> ModuleType:
    status = native_extension_status(path)
    if not status.available:
        raise RuntimeError(status.error)
    with _PREBUILT_MODULES_LOCK:
        cached = _PREBUILT_MODULES.get(status.extension_path)
        if cached is not None:
            return cached
        claimed_path = next(
            (
                loaded_path
                for loaded_path, loaded_module in _PREBUILT_MODULES.items()
                if loaded_module.__name__ == status.stable_module_name
            ),
            None,
        )
        if claimed_path is not None:
            raise RuntimeError(
                f"stable module name {status.stable_module_name!r} is already loaded from "
                f"{claimed_path}; refusing distinct extension {status.extension_path}"
            )
        spec = importlib.util.spec_from_file_location(status.stable_module_name, status.extension_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not create module spec for {status.extension_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module_abi = str(getattr(module, "EIEL_NATIVE_RUNTIME_ABI_VERSION", "") or "")
        module_version = str(getattr(module, "EIEL_NATIVE_RUNTIME_EXTENSION_VERSION", "") or "")
        if module_abi != ABI_VERSION or module_version != EXTENSION_VERSION:
            raise RuntimeError(
                "loaded native extension ABI/version mismatch: "
                f"abi={module_abi or '<missing>'} version={module_version or '<missing>'}"
            )
        _PREBUILT_MODULES[status.extension_path] = module
        return module
