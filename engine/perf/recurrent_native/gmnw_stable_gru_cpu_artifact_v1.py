"""Fail-closed prebuilt GMNW exact-math stable-GRU CPU oracle."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from .stable_gru_cpu_artifact_v1 import (
    ABI_VERSION,
    LoadedStableGruCpuOracle,
    _configure_library,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = Path(__file__).with_name(
    "gmnw_stable_gru_cpu_oracle_manifest_v1.json"
)
SCHEMA_VERSION = "eiel_gmnw_stable_gru_cpu_oracle_manifest_v1"


class GmnwStableGruCpuArtifactError(RuntimeError):
    """Raised when the GMNW exact-math artifact identity drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("runtime_jit") is not False
        or payload.get("production_routing_qualified") is not False
        or payload.get("qualification_scope")
        != "GMNW integrated CPU/CUDA recurrent exactness oracle only"
    ):
        raise GmnwStableGruCpuArtifactError("GMNW oracle manifest contract drifted")
    identity = subprocess.run(
        ["g++", "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    if identity != payload.get("compiler_identity"):
        raise GmnwStableGruCpuArtifactError("GMNW oracle compiler identity drifted")
    for entry in payload.get("sources", []):
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise GmnwStableGruCpuArtifactError("GMNW oracle source path is invalid")
        source = PROJECT_ROOT / relative
        if not source.is_file() or _sha256(source) != entry.get("sha256"):
            raise GmnwStableGruCpuArtifactError(
                f"GMNW oracle source hash mismatch: {relative}"
            )
    return payload


def build_prebuilt_gmnw_stable_gru_cpu_oracle(
    output_directory: str | Path,
) -> Path:
    payload = _manifest()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / payload["artifact"]["file_name"]
    command = [
        str(artifact) if value == "{artifact}" else value
        for value in payload["build_command"]
    ]
    environment = os.environ.copy()
    environment.update({"LC_ALL": "C", "SOURCE_DATE_EPOCH": "0"})
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise GmnwStableGruCpuArtifactError(completed.stderr.strip())
    _validate_artifact(artifact, payload)
    return artifact


def _validate_artifact(path: Path, payload: dict[str, Any]) -> str:
    receipt = payload["artifact"]
    if path.is_symlink() or not path.is_file() or path.name != receipt["file_name"]:
        raise GmnwStableGruCpuArtifactError("GMNW oracle artifact path is invalid")
    if path.stat().st_size != receipt["size_bytes"]:
        raise GmnwStableGruCpuArtifactError("GMNW oracle artifact size mismatch")
    digest = _sha256(path)
    if digest != receipt["sha256"]:
        raise GmnwStableGruCpuArtifactError("GMNW oracle artifact hash mismatch")
    symbols = subprocess.run(
        ["nm", "-D", "--defined-only", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    observed = sorted(line.split()[-1] for line in symbols.splitlines() if line.split())
    if observed != sorted(payload["exported_symbols"]):
        raise GmnwStableGruCpuArtifactError("GMNW oracle exported symbols mismatch")
    return digest


def load_prebuilt_gmnw_stable_gru_cpu_oracle(
    artifact_path: str | Path,
) -> LoadedStableGruCpuOracle:
    payload = _manifest()
    path = Path(artifact_path).resolve()
    digest = _validate_artifact(path, payload)
    library = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    _configure_library(library)
    if int(library.eiel_stable_gru_cpu_oracle_abi_version_v1()) != ABI_VERSION:
        raise GmnwStableGruCpuArtifactError("GMNW oracle ABI mismatch")
    return LoadedStableGruCpuOracle(library, path, digest, payload)


__all__ = [
    "DEFAULT_MANIFEST",
    "GmnwStableGruCpuArtifactError",
    "build_prebuilt_gmnw_stable_gru_cpu_oracle",
    "load_prebuilt_gmnw_stable_gru_cpu_oracle",
]
