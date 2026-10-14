"""Receipt-bound loader for the Tensor World C1 CUDA extension."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType


PREBUILT_ENV = "TENSOR_WORLD_C1_PREBUILT"
MANIFEST_SCHEMA = "tensor_world_c1_cuda_build_manifest_v1"
REQUIRED_EXPORTS = ("observe", "policy_logits", "select_actions", "settle")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tensor_world_c1_extension(path: str | Path | None = None) -> ModuleType:
    configured = path or os.environ.get(PREBUILT_ENV)
    if not configured:
        raise RuntimeError(f"{PREBUILT_ENV} must name a receipt-bound C1 extension")
    binary = Path(configured).resolve()
    manifest_path = binary.with_suffix(binary.suffix + ".manifest.json")
    if not binary.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Tensor World C1 prebuilt binary or manifest is missing"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise RuntimeError("Tensor World C1 build manifest schema is unsupported")
    if manifest.get("binary_sha256") != sha256_file(binary):
        raise RuntimeError("Tensor World C1 binary hash does not match its manifest")
    spec = importlib.util.spec_from_file_location("tensor_world_c1_ext", binary)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create Tensor World C1 extension loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [name for name in REQUIRED_EXPORTS if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"Tensor World C1 extension exports are missing: {missing}")
    return module
