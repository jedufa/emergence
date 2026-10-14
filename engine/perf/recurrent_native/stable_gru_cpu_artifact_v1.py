from __future__ import annotations

"""Fail-closed build and load boundary for the prebuilt stable-GRU CPU oracle."""

from dataclasses import dataclass
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

from ..recurrent_native_contract_v1 import load_recurrent_native_contract


SCHEMA_VERSION = "eiel_stable_gru_cpu_oracle_shared_build_manifest_v1"
ABI_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUILD_MANIFEST = Path(__file__).with_name(
    "stable_gru_cpu_oracle_shared_build_manifest_v1.json"
)


class StableGruCpuArtifactError(RuntimeError):
    """Raised when a build or artifact does not match the frozen admission receipt."""


class StepConfigV1(ctypes.Structure):
    _fields_ = [
        ("contract_version", ctypes.c_uint32),
        ("observation_size", ctypes.c_int64),
        ("active_memory_size", ctypes.c_int64),
        ("aggression", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("drives", ctypes.c_float * 4),
        ("gain", ctypes.c_float),
        ("hunger_gather_weight", ctypes.c_float),
        ("pain_rest_weight", ctypes.c_float),
        ("fatigue_rest_weight", ctypes.c_float),
        ("pain_attack_weight", ctypes.c_float),
        ("fear_attack_weight", ctypes.c_float),
    ]


class CheckpointRowV1(ctypes.Structure):
    _fields_ = [
        ("contract_version", ctypes.c_uint32),
        ("cell_index", ctypes.c_int64),
        ("agent_index", ctypes.c_int64),
        ("hidden_size", ctypes.c_int64),
        ("token", ctypes.c_uint64),
        ("hidden", ctypes.c_float * 64),
    ]


@dataclass(frozen=True)
class LoadedStableGruCpuOracle:
    library: ctypes.CDLL
    artifact_path: Path
    artifact_sha256: str
    manifest: Mapping[str, Any]


def build_prebuilt_stable_gru_cpu_oracle(
    output_directory: str | Path,
    *,
    manifest_path: str | Path = DEFAULT_BUILD_MANIFEST,
) -> Path:
    """Explicit build-time action; the production loader never calls this function."""
    payload, manifest = _validated_manifest(manifest_path)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / str(payload["artifact"]["file_name"])
    compiler = str(payload["build"]["command"][0])
    identity = subprocess.run(
        [compiler, "--version"], cwd=PROJECT_ROOT, check=False,
        capture_output=True, text=True,
    )
    first_line = identity.stdout.splitlines()[0] if identity.returncode == 0 and identity.stdout else ""
    if first_line != payload["build"]["compiler_identity"]:
        raise StableGruCpuArtifactError("stable GRU compiler identity mismatch")
    command = [
        str(artifact) if value == "{artifact}" else str(PROJECT_ROOT) if value == "{project_root}" else value
        for value in payload["build"]["command"]
    ]
    environment = os.environ.copy()
    environment.update({str(k): str(v) for k, v in payload["build"]["environment"].items()})
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise StableGruCpuArtifactError(
            f"stable GRU CPU artifact build failed: {completed.stderr.strip()}"
        )
    _verify_artifact(artifact, payload["artifact"], manifest)
    _verify_exported_symbols(artifact, payload["c_abi"]["exported_symbols"])
    return artifact


def load_prebuilt_stable_gru_cpu_oracle(
    artifact_path: str | Path,
    *,
    manifest_path: str | Path = DEFAULT_BUILD_MANIFEST,
) -> LoadedStableGruCpuOracle:
    payload, manifest = _validated_manifest(manifest_path)
    supplied_artifact = Path(artifact_path)
    if supplied_artifact.is_symlink():
        raise StableGruCpuArtifactError("prebuilt artifact must not be a symlink")
    artifact = supplied_artifact.resolve()
    digest = _verify_artifact(artifact, payload["artifact"], manifest)
    _verify_exported_symbols(artifact, payload["c_abi"]["exported_symbols"])
    try:
        library = ctypes.CDLL(str(artifact), mode=ctypes.RTLD_LOCAL)
    except OSError as exc:
        raise StableGruCpuArtifactError(f"cannot load stable GRU artifact: {exc}") from exc
    _configure_library(library)
    if int(library.eiel_stable_gru_cpu_oracle_abi_version_v1()) != ABI_VERSION:
        raise StableGruCpuArtifactError("stable GRU artifact ABI version mismatch")
    for symbol in payload["c_abi"]["exported_symbols"]:
        if not hasattr(library, symbol):
            raise StableGruCpuArtifactError(f"stable GRU artifact symbol missing: {symbol}")
    return LoadedStableGruCpuOracle(library, artifact, digest, payload)


def repack_production_gru_parameter_tape_v1(
    values: Any,
    *,
    observation_width: int,
) -> list[Any]:
    """Map `GRUPolicy.parameters()` order into the resident packed-bank order."""
    if isinstance(observation_width, bool) or not isinstance(observation_width, int) \
            or observation_width <= 0:
        raise StableGruCpuArtifactError("observation_width must be a positive integer")
    source = list(values)
    h = 64
    a = 7
    sizes = [h * observation_width, h, 3 * h * h, 3 * h * h, 3 * h, 3 * h, a * h, a]
    expected = sum(sizes)
    if len(source) != expected:
        raise StableGruCpuArtifactError(
            f"production GRU parameter tape has {len(source)} values, expected {expected}"
        )
    fields: list[list[Any]] = []
    offset = 0
    for size in sizes:
        fields.append(source[offset : offset + size])
        offset += size
    # Production module iteration: encW, encb, Wih, Whh, bih, bhh, headW, headb.
    # Resident pack: encW, encb, Wih, bih, Whh, bhh, headW, headb.
    order = (0, 1, 2, 4, 3, 5, 6, 7)
    return [value for field in order for value in fields[field]]


def _validated_manifest(
    manifest_path: str | Path,
) -> tuple[Mapping[str, Any], Path]:
    manifest = Path(manifest_path).resolve()
    try:
        payload = json.loads(
            manifest.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StableGruCpuArtifactError(f"cannot read stable GRU build manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise StableGruCpuArtifactError("stable GRU build manifest must be an object")
    _exact_keys(
        payload,
        {"schema", "schema_version", "runtime_jit", "production_routing_qualified",
         "target", "toolchain", "sources", "build", "artifact", "c_abi", "qualification"},
        "manifest",
    )
    if payload["schema"] != SCHEMA_VERSION or payload["schema_version"] != 1:
        raise StableGruCpuArtifactError("stable GRU build manifest schema mismatch")
    if payload["runtime_jit"] is not False or payload["production_routing_qualified"] is not False:
        raise StableGruCpuArtifactError("artifact must remain prebuilt and production-routing unqualified")
    target = _mapping(payload["target"], "target")
    if (
        target != {"platform": "linux", "architecture": "x86_64", "abi": "LP64",
                   "endianness": "little", "cxx_standard": "c++17"}
        or not sys.platform.startswith("linux")
        or platform.machine() != "x86_64"
        or sys.byteorder != "little"
        or ctypes.sizeof(ctypes.c_void_p) != 8
    ):
        raise StableGruCpuArtifactError("unsupported stable GRU CPU artifact target")
    if ctypes.sizeof(StepConfigV1) != 72 or ctypes.sizeof(CheckpointRowV1) != 296:
        raise StableGruCpuArtifactError("stable GRU ctypes ABI layout mismatch")
    toolchain = _mapping(payload["toolchain"], "toolchain")
    _exact_keys(toolchain, {"compiler", "identity"}, "toolchain")
    if toolchain != {
        "compiler": "g++",
        "identity": "g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0",
    }:
        raise StableGruCpuArtifactError("stable GRU toolchain receipt drifted")
    load_recurrent_native_contract(project_root=PROJECT_ROOT)
    sources = payload["sources"]
    if not isinstance(sources, list) or not sources:
        raise StableGruCpuArtifactError("artifact source inventory is empty")
    expected_source_paths = [
        "engine/perf/native/recurrent/recurrent_common_v1.hpp",
        "engine/perf/native/recurrent/stable_gru_v1.hpp",
        "engine/perf/native/recurrent/stable_gru_v1.cpp",
        "engine/perf/native/recurrent/stable_gru_cpu_oracle_c_api_v1.h",
        "engine/perf/native/recurrent/stable_gru_cpu_oracle_c_api_v1.cpp",
    ]
    if [entry.get("path") if isinstance(entry, dict) else None for entry in sources] != (
        expected_source_paths
    ):
        raise StableGruCpuArtifactError("stable GRU source inventory drifted")
    for index, entry_value in enumerate(sources):
        entry = _mapping(entry_value, f"sources[{index}]")
        _exact_keys(entry, {"path", "sha256"}, f"sources[{index}]")
        relative = Path(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise StableGruCpuArtifactError(f"invalid source path: {relative}")
        source = PROJECT_ROOT / relative
        if not source.is_file() or _sha256(source) != entry["sha256"]:
            raise StableGruCpuArtifactError(f"source hash mismatch: {relative}")
    build = _mapping(payload["build"], "build")
    _exact_keys(build, {"command", "environment", "compiler_identity"}, "build")
    expected_command = [
        "g++", "-std=c++17", "-O2", "-fPIC", "-shared", "-Wall", "-Wextra",
        "-Wconversion", "-Wsign-conversion", "-Wshadow", "-Werror", "-pedantic",
        "-fno-exceptions", "-fno-rtti", "-fvisibility=hidden", "-fno-ident",
        "-frandom-seed=eiel-stable-gru-cpu-oracle-v1", "-Wl,--build-id=none",
        "-Wl,-z,defs", "-Wl,-z,relro,-z,now",
        "engine/perf/native/recurrent/stable_gru_v1.cpp",
        "engine/perf/native/recurrent/stable_gru_cpu_oracle_c_api_v1.cpp",
        "-o", "{artifact}",
    ]
    if build["command"] != expected_command or build["environment"] != {
        "LC_ALL": "C", "SOURCE_DATE_EPOCH": "0",
    }:
        raise StableGruCpuArtifactError("stable GRU deterministic build receipt drifted")
    if build["compiler_identity"] != toolchain["identity"]:
        raise StableGruCpuArtifactError("build and toolchain compiler identities disagree")
    artifact = _mapping(payload["artifact"], "artifact")
    _exact_keys(artifact, {"file_name", "size_bytes", "sha256"}, "artifact")
    if (
        artifact["file_name"] != "libeiel_stable_gru_cpu_oracle_v1.so"
        or isinstance(artifact["size_bytes"], bool)
        or not isinstance(artifact["size_bytes"], int)
        or artifact["size_bytes"] <= 0
        or not isinstance(artifact["sha256"], str)
        or len(artifact["sha256"]) != 64
    ):
        raise StableGruCpuArtifactError("stable GRU artifact receipt is invalid")
    c_abi = _mapping(payload["c_abi"], "c_abi")
    _exact_keys(c_abi, {"version", "version_function", "exported_symbols"}, "c_abi")
    if c_abi.get("version") != ABI_VERSION or c_abi.get("version_function") != (
        "eiel_stable_gru_cpu_oracle_abi_version_v1"
    ):
        raise StableGruCpuArtifactError("stable GRU C ABI receipt mismatch")
    expected_symbols = [
        "eiel_stable_gru_checkpoint_project_v1",
        "eiel_stable_gru_checkpoint_restore_v1",
        "eiel_stable_gru_cpu_oracle_abi_version_v1",
        "eiel_stable_gru_forward_batch_v1",
        "eiel_stable_gru_forward_v1",
        "eiel_stable_gru_lifecycle_v1",
        "eiel_stable_gru_mutate_v1",
        "eiel_stable_gru_packed_parameter_count_v1",
    ]
    if c_abi.get("exported_symbols") != expected_symbols:
        raise StableGruCpuArtifactError("stable GRU exported symbol receipt drifted")
    qualification = _mapping(payload["qualification"], "qualification")
    _exact_keys(
        qualification,
        {"scope", "requires_source_hash_match", "requires_artifact_hash_match",
         "requires_two_build_byte_identity", "float_gate", "does_not_authorize"},
        "qualification",
    )
    if qualification["scope"] != "standalone stable-GRU CPU production-oracle parity only":
        raise StableGruCpuArtifactError("stable GRU qualification scope drifted")
    if qualification.get("requires_source_hash_match") is not True or qualification.get(
        "requires_artifact_hash_match"
    ) is not True or qualification.get("requires_two_build_byte_identity") is not True:
        raise StableGruCpuArtifactError("stable GRU qualification requirements weakened")
    if qualification.get("float_gate") != {
        "rtol": 0.0,
        "one_step_hidden_atol": 5e-7,
        "sequence_hidden_atol": 2e-6,
        "one_step_post_temperature_logits_atol": 5e-6,
        "sequence_post_temperature_logits_atol": 1e-5,
        "basis": (
            "bounded real-GRUPolicy production-weight qualification including temperature "
            "clamp 0.05; actions remain exact"
        ),
    }:
        raise StableGruCpuArtifactError("stable GRU CPU float gate drifted")
    if qualification["does_not_authorize"] != [
        "production routing",
        "resident state ownership",
        "CUDA capability publication",
        "runtime JIT",
        "WDL plus GRU integration",
        "Phase 1 integration",
    ]:
        raise StableGruCpuArtifactError("stable GRU authorization boundary drifted")
    return payload, manifest


def _verify_artifact(
    artifact: Path,
    receipt_value: Any,
    manifest: Path,
) -> str:
    receipt = _mapping(receipt_value, "artifact")
    if not artifact.is_file() or artifact.is_symlink():
        raise StableGruCpuArtifactError(f"prebuilt artifact is missing or not a regular file: {artifact}")
    if artifact.name != receipt["file_name"]:
        raise StableGruCpuArtifactError("prebuilt artifact filename mismatch")
    if artifact.stat().st_size != receipt["size_bytes"]:
        raise StableGruCpuArtifactError("prebuilt artifact size mismatch")
    digest = _sha256(artifact)
    if digest != receipt["sha256"]:
        raise StableGruCpuArtifactError(
            f"prebuilt artifact hash mismatch against {manifest.name}"
        )
    return digest


def _verify_exported_symbols(artifact: Path, expected_value: Any) -> None:
    completed = subprocess.run(
        ["nm", "-D", "--defined-only", str(artifact)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise StableGruCpuArtifactError(
            f"cannot inspect stable GRU artifact exports: {completed.stderr.strip()}"
        )
    actual = sorted(
        line.split()[-1]
        for line in completed.stdout.splitlines()
        if line.split()
    )
    if actual != sorted(expected_value):
        raise StableGruCpuArtifactError("stable GRU artifact export set mismatch")


def _configure_library(library: ctypes.CDLL) -> None:
    float_p = ctypes.POINTER(ctypes.c_float)
    int32_p = ctypes.POINTER(ctypes.c_int32)
    uint8_p = ctypes.POINTER(ctypes.c_uint8)
    uint64_p = ctypes.POINTER(ctypes.c_uint64)
    library.eiel_stable_gru_cpu_oracle_abi_version_v1.restype = ctypes.c_uint32
    library.eiel_stable_gru_packed_parameter_count_v1.argtypes = [
        ctypes.c_int64, ctypes.POINTER(ctypes.c_int64)
    ]
    library.eiel_stable_gru_packed_parameter_count_v1.restype = ctypes.c_int32
    library.eiel_stable_gru_forward_v1.argtypes = [
        ctypes.POINTER(StepConfigV1), float_p, ctypes.c_int64, float_p, ctypes.c_int64,
        float_p, ctypes.c_int64, float_p, ctypes.c_int64, float_p, ctypes.c_int64, int32_p,
    ]
    library.eiel_stable_gru_forward_v1.restype = ctypes.c_int32
    library.eiel_stable_gru_forward_batch_v1.argtypes = [
        ctypes.POINTER(StepConfigV1), ctypes.c_int64, float_p, ctypes.c_int64,
        float_p, ctypes.c_int64, float_p, ctypes.c_int64, ctypes.c_int64,
        float_p, ctypes.c_int64, float_p, ctypes.c_int64, int32_p, ctypes.c_int64,
        float_p, ctypes.c_int64, float_p, ctypes.c_int64, int32_p, ctypes.c_int64,
    ]
    library.eiel_stable_gru_forward_batch_v1.restype = ctypes.c_int32
    library.eiel_stable_gru_lifecycle_v1.argtypes = [
        float_p, ctypes.c_int64, uint64_p, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32,
    ]
    library.eiel_stable_gru_lifecycle_v1.restype = ctypes.c_int32
    library.eiel_stable_gru_mutate_v1.argtypes = [
        float_p, ctypes.c_int64, float_p, uint8_p, ctypes.c_float,
        float_p, ctypes.c_int64, float_p, ctypes.c_int64,
    ]
    library.eiel_stable_gru_mutate_v1.restype = ctypes.c_int32
    library.eiel_stable_gru_checkpoint_project_v1.argtypes = [
        float_p, ctypes.c_int64, uint64_p, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_uint64, ctypes.POINTER(CheckpointRowV1),
    ]
    library.eiel_stable_gru_checkpoint_project_v1.restype = ctypes.c_int32
    library.eiel_stable_gru_checkpoint_restore_v1.argtypes = [
        ctypes.POINTER(CheckpointRowV1), float_p, ctypes.c_int64, uint64_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_uint64,
    ]
    library.eiel_stable_gru_checkpoint_restore_v1.restype = ctypes.c_int32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StableGruCpuArtifactError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise StableGruCpuArtifactError(f"{name} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise StableGruCpuArtifactError(
            f"{name} keys mismatch: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


__all__ = [
    "ABI_VERSION",
    "CheckpointRowV1",
    "DEFAULT_BUILD_MANIFEST",
    "LoadedStableGruCpuOracle",
    "SCHEMA_VERSION",
    "StableGruCpuArtifactError",
    "StepConfigV1",
    "build_prebuilt_stable_gru_cpu_oracle",
    "load_prebuilt_stable_gru_cpu_oracle",
    "repack_production_gru_parameter_tape_v1",
]
