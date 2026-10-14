from __future__ import annotations

"""Fail-closed loader for immutable CUDA Architecture B extensions."""

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = "eiel_cuda_architecture_b_build_manifest_v2"
EXTENSION_ABI_VERSION = "eiel_cuda_architecture_b_abi_v3"
EXTENSION_VERSION = "eiel_cuda_architecture_b_resident_generation_v6"


@dataclass(frozen=True)
class CudaPrebuiltIdentity:
    artifact: Path
    manifest_path: Path
    artifact_sha256: str
    module_name: str
    extension_abi_version: str
    extension_version: str
    exported_functions: tuple[str, ...]
    capability_contract_sha256: str
    manifest: Mapping[str, Any]


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


def validate_cuda_prebuilt_manifest(
    artifact: str | Path,
    *,
    required_exports: Sequence[str] = (),
    expected_abi_version: str = EXTENSION_ABI_VERSION,
    expected_extension_version: str = EXTENSION_VERSION,
) -> CudaPrebuiltIdentity:
    artifact_path = Path(artifact).resolve()
    manifest_path = artifact_path.with_suffix(artifact_path.suffix + ".manifest.json")
    if not artifact_path.is_file():
        raise RuntimeError(f"CUDA Architecture B artifact is missing: {artifact_path}")
    if not manifest_path.is_file():
        raise RuntimeError(f"CUDA Architecture B manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("CUDA Architecture B manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("CUDA Architecture B manifest must be an object")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("CUDA Architecture B manifest schema mismatch")
    abi_version = str(manifest.get("extension_abi_version", "") or "")
    extension_version = str(manifest.get("extension_version", "") or "")
    if abi_version != expected_abi_version or extension_version != expected_extension_version:
        raise RuntimeError(
            "CUDA Architecture B extension contract mismatch: "
            f"abi={abi_version or '<missing>'} version={extension_version or '<missing>'}"
        )
    if manifest.get("immutable_output") is not True:
        raise RuntimeError("CUDA Architecture B artifact is not declared immutable")
    if manifest.get("prebuilt_outside_measured_execution") is not True:
        raise RuntimeError("CUDA Architecture B artifact was not built outside measured execution")
    if manifest.get("runtime_jit_in_measured_execution") is not False:
        raise RuntimeError("CUDA Architecture B artifact does not prove runtime_jit=false")
    expected_hash = str(manifest.get("output_sha256", "") or "")
    actual_hash = _sha256(artifact_path)
    if len(expected_hash) != 64 or actual_hash != expected_hash:
        raise RuntimeError("CUDA Architecture B artifact hash mismatch")
    manifest_output = str(manifest.get("output", "") or "")
    if not manifest_output or Path(manifest_output).resolve() != artifact_path:
        raise RuntimeError("CUDA Architecture B manifest output path mismatch")
    module_name = str(manifest.get("stable_module_name", "") or "")
    if not module_name.isidentifier():
        raise RuntimeError("CUDA Architecture B stable module name is invalid")
    raw_exports = manifest.get("exported_functions")
    if not isinstance(raw_exports, list) or any(not isinstance(name, str) for name in raw_exports):
        raise RuntimeError("CUDA Architecture B exported function manifest is invalid")
    exported_functions = tuple(raw_exports)
    missing_exports = sorted(set(required_exports) - set(exported_functions))
    if missing_exports:
        raise RuntimeError(f"CUDA Architecture B manifest is missing required exports: {missing_exports}")
    capability_contract = manifest.get("capability_contract")
    capability_contract_sha256 = str(
        manifest.get("capability_contract_sha256", "") or ""
    )
    if not isinstance(capability_contract, dict):
        raise RuntimeError("CUDA Architecture B capability contract is missing")
    if (
        len(capability_contract_sha256) != 64
        or _canonical_json_sha256(capability_contract)
        != capability_contract_sha256
    ):
        raise RuntimeError("CUDA Architecture B capability contract hash mismatch")
    if capability_contract.get("promotion_qualified") is not False:
        raise RuntimeError(
            "CUDA Architecture B capability contract must not claim promotion"
        )
    if capability_contract.get("throughput_qualified") is not False:
        raise RuntimeError(
            "CUDA Architecture B capability contract must not claim throughput"
        )
    return CudaPrebuiltIdentity(
        artifact=artifact_path,
        manifest_path=manifest_path,
        artifact_sha256=actual_hash,
        module_name=module_name,
        extension_abi_version=abi_version,
        extension_version=extension_version,
        exported_functions=exported_functions,
        capability_contract_sha256=capability_contract_sha256,
        manifest=manifest,
    )


def load_prebuilt_cuda_architecture_b_extension(
    artifact: str | Path,
    *,
    required_exports: Sequence[str] = (),
    expected_abi_version: str = EXTENSION_ABI_VERSION,
    expected_extension_version: str = EXTENSION_VERSION,
) -> ModuleType:
    identity = validate_cuda_prebuilt_manifest(
        artifact,
        required_exports=required_exports,
        expected_abi_version=expected_abi_version,
        expected_extension_version=expected_extension_version,
    )
    spec = importlib.util.spec_from_file_location(identity.module_name, identity.artifact)
    if spec is None or spec.loader is None:
        raise RuntimeError("CUDA Architecture B extension import specification failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing_exports = [name for name in required_exports if not hasattr(module, name)]
    if missing_exports:
        raise RuntimeError(f"CUDA Architecture B binary is missing required exports: {missing_exports}")
    return module


__all__ = [
    "CudaPrebuiltIdentity",
    "EXTENSION_ABI_VERSION",
    "EXTENSION_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "load_prebuilt_cuda_architecture_b_extension",
    "validate_cuda_prebuilt_manifest",
]
