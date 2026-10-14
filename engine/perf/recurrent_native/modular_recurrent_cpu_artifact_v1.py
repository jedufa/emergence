from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any, Final, Mapping


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_BUILD_MANIFEST: Final[Path] = Path(__file__).with_name(
    "modular_recurrent_cpu_oracle_shared_build_manifest_v1.json"
)
ABI_VERSION: Final[int] = 1
PLANE_COUNT: Final[int] = 57
MODULE_SLOTS: Final[int] = 16
CONNECTION_SLOTS: Final[int] = 48
ACTION_WIDTH: Final[int] = 7
OBSERVATION_WIDTH: Final[int] = 27
COST_FIELDS: Final[int] = 10


class ModularRecurrentCpuArtifactError(RuntimeError):
    """The framework-free modular recurrent CPU artifact failed identity checks."""


class RowV1(ctypes.Structure):
    _fields_ = [
        ("agent_occupied", ctypes.c_uint8),
        ("agent_id", ctypes.c_int64),
        ("agent_slot_token", ctypes.c_int64),
        ("module_count", ctypes.c_int32),
        ("connection_count", ctypes.c_int32),
        ("module_occupied", ctypes.c_uint8 * 16),
        ("module_local_id_index", ctypes.c_int32 * 16),
        ("module_instance_id_index", ctypes.c_int32 * 16),
        ("module_homology_id_index", ctypes.c_int32 * 16),
        ("module_type_index", ctypes.c_int32 * 16),
        ("module_type_code", ctypes.c_int8 * 16),
        ("identity_origin_code", ctypes.c_int8 * 16),
        ("topology_position", ctypes.c_float * 32),
        ("module_execution_order", ctypes.c_int32 * 16),
        ("dependency_topological_order", ctypes.c_int32 * 16),
        ("operator_code", ctypes.c_int8 * 16),
        ("integrator_parameters", ctypes.c_float * 64),
        ("gated_parameters", ctypes.c_float * 160),
        ("module_dormant", ctypes.c_uint8 * 16),
        ("module_state_present", ctypes.c_uint8 * 16),
        ("module_state", ctypes.c_float * 16),
        ("module_output_present", ctypes.c_uint8 * 16),
        ("module_previous_output", ctypes.c_float * 16),
        ("input_binding_count", ctypes.c_int8 * 16),
        ("input_semantic_code", ctypes.c_int8 * 32),
        ("input_semantic_id_index", ctypes.c_int32 * 32),
        ("input_gain", ctypes.c_float * 32),
        ("input_enabled", ctypes.c_uint8 * 32),
        ("output_binding_count", ctypes.c_int8 * 16),
        ("output_semantic_code", ctypes.c_int8 * 32),
        ("output_semantic_id_index", ctypes.c_int32 * 32),
        ("output_action_index", ctypes.c_int8 * 32),
        ("output_gain", ctypes.c_float * 32),
        ("output_enabled", ctypes.c_uint8 * 32),
        ("development_explicit", ctypes.c_uint8 * 16),
        ("development_stage_gains", ctypes.c_float * 48),
        ("development_prerequisite_count", ctypes.c_int8 * 16),
        ("development_prerequisite_homology_index", ctypes.c_int32 * 32),
        ("development_prerequisite_module_slot", ctypes.c_int32 * 32),
        ("development_latch_enabled", ctypes.c_uint8 * 16),
        ("development_latch_present", ctypes.c_uint8 * 16),
        ("development_latch_state", ctypes.c_uint8 * 16),
        ("development_effective_gain_present", ctypes.c_uint8 * 16),
        ("development_effective_gain", ctypes.c_float * 16),
        ("connection_occupied", ctypes.c_uint8 * 48),
        ("connection_local_id_index", ctypes.c_int32 * 48),
        ("connection_src_module_slot", ctypes.c_int32 * 48),
        ("connection_dst_module_slot", ctypes.c_int32 * 48),
        ("connection_src_slot_index", ctypes.c_int32 * 48),
        ("connection_dst_slot_index", ctypes.c_int32 * 48),
        ("connection_value_type_index", ctypes.c_int32 * 48),
        ("connection_channel_index", ctypes.c_int32 * 48),
        ("connection_weight", ctypes.c_float * 48),
        ("connection_execution_order", ctypes.c_int32 * 48),
        ("structural_cost_counts", ctypes.c_int64 * 10),
        ("structural_cost_components", ctypes.c_double * 10),
        ("structural_cost_total", ctypes.c_double),
    ]


class OwnerViewV1(ctypes.Structure):
    _fields_ = [
        ("slot_to_agent_id", ctypes.POINTER(ctypes.c_int64)),
        ("slot_tokens", ctypes.POINTER(ctypes.c_int64)),
        ("alive_mask", ctypes.POINTER(ctypes.c_uint8)),
        ("row_count", ctypes.c_int64),
    ]


class ConstOwnerViewV1(ctypes.Structure):
    _fields_ = OwnerViewV1._fields_


class MutationV1(ctypes.Structure):
    _fields_ = [
        ("field", ctypes.c_uint32),
        ("outer_index", ctypes.c_int32),
        ("inner_index", ctypes.c_int32),
        ("additive_delta", ctypes.c_float),
    ]


class CanonicalReceiptBindingV1(ctypes.Structure):
    _fields_ = [
        ("binding_version", ctypes.c_uint32),
        ("canonical_receipt_contract_version", ctypes.c_uint32),
        ("canonical_receipt_sha256_words", ctypes.c_uint64 * 4),
        ("canonical_receipt_bytes_sha256_words", ctypes.c_uint64 * 4),
        ("draw0_float64_bits", ctypes.c_uint64),
        ("draw1_float64_bits", ctypes.c_uint64),
        ("transaction_ordinal", ctypes.c_int64),
        ("parent_agent_id", ctypes.c_int64),
        ("parent_slot", ctypes.c_int64),
        ("parent_slot_token", ctypes.c_int64),
        ("child_agent_id", ctypes.c_int64),
        ("child_slot", ctypes.c_int64),
        ("child_slot_token", ctypes.c_int64),
        ("selected_instance_id_index", ctypes.c_int32),
        ("selected_homology_id_index", ctypes.c_int32),
        ("operation_code", ctypes.c_int32),
        ("status_code", ctypes.c_int32),
        ("reason_code", ctypes.c_int32),
        ("changed", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 7),
        ("structural_payload_sha256_words", ctypes.c_uint64 * 4),
        ("changed_dimension_code", ctypes.c_int32),
        ("native_receipt_construction_required", ctypes.c_uint8),
        ("reserved_tail", ctypes.c_uint8 * 3),
    ]


class CanonicalReceiptSourceV1(ctypes.Structure):
    _fields_ = [
        ("source_version", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("source_packed_sha256_words", ctypes.c_uint64 * 4),
        ("ancestry_tape_sha256_words", ctypes.c_uint64 * 4),
    ]


class BirthReceiptV1(ctypes.Structure):
    _fields_ = [
        ("receipt_version", ctypes.c_uint32),
        ("lifecycle_status", ctypes.c_int32),
        ("committed", ctypes.c_uint8),
        ("owner_checked", ctypes.c_uint8),
        ("parent_preserved", ctypes.c_uint8),
        ("child_runtime_reset", ctypes.c_uint8),
        ("canonical_receipt_bound", ctypes.c_uint8),
        ("canonical_receipt_identity_checked", ctypes.c_uint8),
        ("recurrent_receipt_bytes_bound", ctypes.c_uint8),
        ("canonical_receipt_reserved", ctypes.c_uint8),
        ("destination_row", ctypes.c_int64),
        ("parent_a_row", ctypes.c_int64),
        ("parent_b_row", ctypes.c_int64),
        ("selected_parent_row", ctypes.c_int64),
        ("expected_destination_token", ctypes.c_int64),
        ("expected_parent_a_token", ctypes.c_int64),
        ("expected_parent_b_token", ctypes.c_int64),
        ("parent_a_agent_id", ctypes.c_int64),
        ("parent_b_agent_id", ctypes.c_int64),
        ("child_agent_id", ctypes.c_int64),
        ("publication_token", ctypes.c_int64),
        ("donor_draw_float32_bits", ctypes.c_uint32),
        ("mutation_count", ctypes.c_int64),
        ("draw0_float64_bits", ctypes.c_uint64),
        ("draw1_float64_bits", ctypes.c_uint64),
        ("canonical_recurrent_receipt_sha256_words", ctypes.c_uint64 * 4),
        ("canonical_recurrent_receipt_bytes_sha256_words", ctypes.c_uint64 * 4),
        ("transaction_ordinal", ctypes.c_int64),
        ("selected_instance_id_index", ctypes.c_int32),
        ("selected_homology_id_index", ctypes.c_int32),
        ("operation_code", ctypes.c_int32),
        ("status_code", ctypes.c_int32),
        ("reason_code", ctypes.c_int32),
        ("changed", ctypes.c_uint8),
        ("canonical_receipt_reserved_tail", ctypes.c_uint8 * 7),
        ("structural_payload_sha256_words", ctypes.c_uint64 * 4),
        ("structural_payload_sha256_bound", ctypes.c_uint8),
        ("structural_payload_identity_checked", ctypes.c_uint8),
        ("structural_payload_reserved_tail", ctypes.c_uint8 * 6),
    ]


class TickReceiptV1(ctypes.Structure):
    _fields_ = [
        ("committed_rows", ctypes.c_int64),
        ("structured_agents", ctypes.c_int64),
        ("neutral_agents", ctypes.c_int64),
        ("executing_modules", ctypes.c_int64),
        ("changing_state_modules", ctypes.c_int64),
        ("rng_draws", ctypes.c_int64),
        ("production_routing_qualified", ctypes.c_uint8),
        ("publication_qualified", ctypes.c_uint8),
    ]


class CostCapacityReceiptV1(ctypes.Structure):
    _fields_ = [
        ("logical_module_slots", ctypes.c_int32),
        ("logical_connection_slots", ctypes.c_int32),
        ("occupied_modules", ctypes.c_int32),
        ("occupied_connections", ctypes.c_int32),
        ("counts", ctypes.c_int64 * 10),
        ("components", ctypes.c_double * 10),
        ("total", ctypes.c_double),
        ("reporting_only", ctypes.c_uint8),
        ("at_module_capacity", ctypes.c_uint8),
        ("at_connection_capacity", ctypes.c_uint8),
    ]


ROW_PLANE_FIELDS: Final[tuple[str, ...]] = tuple(
    name for name, _ctype in RowV1._fields_
)
if len(ROW_PLANE_FIELDS) != PLANE_COUNT:
    raise RuntimeError("RowV1 does not expose the exact 57-plane schema")


@dataclass(frozen=True)
class ModularRecurrentCpuArtifactV1:
    library: ctypes.CDLL
    artifact_path: Path
    artifact_sha256: str
    manifest_path: Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModularRecurrentCpuArtifactError(f"{name} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ModularRecurrentCpuArtifactError(f"{name} keys drifted")


def _load_manifest(path: str | Path) -> tuple[Mapping[str, Any], Path]:
    manifest = Path(path).resolve()
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModularRecurrentCpuArtifactError(
            f"cannot read build manifest: {manifest}"
        ) from exc
    root = _mapping(payload, "manifest")
    _exact_keys(
        root,
        {
            "schema",
            "schema_version",
            "runtime_jit",
            "production_routing_qualified",
            "publication_qualified",
            "target",
            "toolchain",
            "sources",
            "build",
            "artifact",
            "c_abi",
            "qualification",
        },
        "manifest",
    )
    if root["schema"] != "eiel_modular_recurrent_cpu_oracle_shared_build_manifest_v1":
        raise ModularRecurrentCpuArtifactError("manifest schema drifted")
    if root["schema_version"] != 1 or root["runtime_jit"] is not False:
        raise ModularRecurrentCpuArtifactError(
            "manifest version or JIT boundary drifted"
        )
    if (
        root["production_routing_qualified"] is not False
        or root["publication_qualified"] is not False
    ):
        raise ModularRecurrentCpuArtifactError(
            "artifact authorization boundary weakened"
        )
    target = _mapping(root["target"], "target")
    if target != {
        "platform": "linux",
        "architecture": "x86_64",
        "abi": "LP64",
        "endianness": "little",
        "cxx_standard": "c++17",
    }:
        raise ModularRecurrentCpuArtifactError("target identity drifted")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ModularRecurrentCpuArtifactError(
            "host does not match the prebuilt target"
        )
    toolchain = _mapping(root["toolchain"], "toolchain")
    _exact_keys(toolchain, {"compiler", "identity"}, "toolchain")
    if toolchain["compiler"] != "g++":
        raise ModularRecurrentCpuArtifactError("compiler command drifted")
    compiler = subprocess.run(
        ["g++", "--version"], check=False, capture_output=True, text=True
    )
    identity = compiler.stdout.splitlines()[0] if compiler.returncode == 0 else ""
    if identity != toolchain["identity"]:
        raise ModularRecurrentCpuArtifactError(
            "compiler identity does not match manifest"
        )
    expected_sources = [
        "engine/perf/native/recurrent/modular_recurrent_cpu_oracle_c_api_v1.h",
        "engine/perf/native/recurrent/modular_recurrent_cpu_oracle_c_api_v1.cpp",
    ]
    sources = root["sources"]
    if (
        not isinstance(sources, list)
        or [entry.get("path") for entry in sources] != expected_sources
    ):
        raise ModularRecurrentCpuArtifactError("source inventory drifted")
    for index, source_value in enumerate(sources):
        source = _mapping(source_value, f"sources[{index}]")
        _exact_keys(source, {"path", "sha256"}, f"sources[{index}]")
        relative = Path(str(source["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ModularRecurrentCpuArtifactError(
                "source path is not repository relative"
            )
        absolute = PROJECT_ROOT / relative
        if not absolute.is_file() or _sha256(absolute) != source["sha256"]:
            raise ModularRecurrentCpuArtifactError(f"source hash mismatch: {relative}")
    build = _mapping(root["build"], "build")
    _exact_keys(build, {"command", "environment", "compiler_identity"}, "build")
    expected_command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        "-Wall",
        "-Wextra",
        "-Wconversion",
        "-Wsign-conversion",
        "-Wshadow",
        "-Werror",
        "-pedantic",
        "-fno-exceptions",
        "-fno-rtti",
        "-fvisibility=hidden",
        "-fno-ident",
        "-ffp-contract=off",
        "-frandom-seed=eiel-modular-recurrent-cpu-oracle-v1",
        "-Wl,--build-id=none",
        "-Wl,-z,defs",
        "-Wl,-z,relro,-z,now",
        "engine/perf/native/recurrent/modular_recurrent_cpu_oracle_c_api_v1.cpp",
        "-o",
        "{artifact}",
    ]
    if build["command"] != expected_command or build["environment"] != {
        "LC_ALL": "C",
        "SOURCE_DATE_EPOCH": "0",
    }:
        raise ModularRecurrentCpuArtifactError("deterministic build receipt drifted")
    if build["compiler_identity"] != toolchain["identity"]:
        raise ModularRecurrentCpuArtifactError("build compiler identity drifted")
    artifact = _mapping(root["artifact"], "artifact")
    _exact_keys(artifact, {"file_name", "size_bytes", "sha256"}, "artifact")
    if artifact["file_name"] != "libeiel_modular_recurrent_cpu_oracle_v1.so":
        raise ModularRecurrentCpuArtifactError("artifact filename drifted")
    c_abi = _mapping(root["c_abi"], "c_abi")
    _exact_keys(
        c_abi,
        {"version", "version_function", "row_size", "plane_count", "exported_symbols"},
        "c_abi",
    )
    if c_abi["version"] != ABI_VERSION or c_abi["plane_count"] != PLANE_COUNT:
        raise ModularRecurrentCpuArtifactError("C ABI version or plane count drifted")
    if c_abi["version_function"] != "eiel_modular_recurrent_cpu_oracle_abi_version_v1":
        raise ModularRecurrentCpuArtifactError("C ABI version function drifted")
    if c_abi["row_size"] != ctypes.sizeof(RowV1):
        raise ModularRecurrentCpuArtifactError("C ABI row size drifted")
    expected_symbols = [
        "eiel_modular_recurrent_checkpoint_project_v1",
        "eiel_modular_recurrent_checkpoint_restore_v1",
        "eiel_modular_recurrent_cost_capacity_v1",
        "eiel_modular_recurrent_cpu_oracle_abi_version_v1",
        "eiel_modular_recurrent_cpu_oracle_plane_count_v1",
        "eiel_modular_recurrent_cpu_oracle_row_size_v1",
        "eiel_modular_recurrent_canonical_receipt_sha256_v1",
        "eiel_modular_recurrent_death_cleanup_v1",
        "eiel_modular_recurrent_founder_init_v1",
        "eiel_modular_recurrent_newborn_inherit_v1",
        "eiel_modular_recurrent_newborn_inherit_with_canonical_receipt_v1",
        "eiel_modular_recurrent_newborn_inherit_with_receipt_v1",
        "eiel_modular_recurrent_row_structural_payload_sha256_v1",
        "eiel_modular_recurrent_same_slot_reuse_v1",
        "eiel_modular_recurrent_same_slot_reuse_with_canonical_receipt_v1",
        "eiel_modular_recurrent_same_slot_reuse_with_receipt_v1",
        "eiel_modular_recurrent_tick_v1",
        "eiel_modular_recurrent_validate_v1",
    ]
    if c_abi["exported_symbols"] != expected_symbols:
        raise ModularRecurrentCpuArtifactError("exported symbol receipt drifted")
    qualification = _mapping(root["qualification"], "qualification")
    _exact_keys(
        qualification,
        {
            "scope",
            "requires_source_hash_match",
            "requires_artifact_hash_match",
            "requires_two_build_byte_identity",
            "does_not_authorize",
        },
        "qualification",
    )
    if qualification != {
        "scope": "standalone framework-free modular recurrent CPU oracle only",
        "requires_source_hash_match": True,
        "requires_artifact_hash_match": True,
        "requires_two_build_byte_identity": True,
        "does_not_authorize": [
            "production routing",
            "capability publication",
            "resident production ownership",
            "CUDA",
            "runtime JIT",
            "campaign execution",
        ],
    }:
        raise ModularRecurrentCpuArtifactError("qualification boundary drifted")
    return root, manifest


def _verify_artifact(
    artifact: Path,
    receipt: Mapping[str, Any],
    *,
    require_final_name: bool = True,
) -> str:
    if not artifact.is_file() or artifact.is_symlink():
        raise ModularRecurrentCpuArtifactError(
            "artifact is missing or not a regular file"
        )
    if (
        require_final_name and artifact.name != receipt["file_name"]
    ) or artifact.stat().st_size != receipt["size_bytes"]:
        raise ModularRecurrentCpuArtifactError("artifact name or size mismatch")
    digest = _sha256(artifact)
    if digest != receipt["sha256"]:
        raise ModularRecurrentCpuArtifactError("artifact hash mismatch")
    return digest


def _verify_exported_symbols(artifact: Path, expected: Any) -> None:
    completed = subprocess.run(
        ["nm", "-D", "--defined-only", str(artifact)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ModularRecurrentCpuArtifactError("cannot inspect artifact exports")
    actual = sorted(
        line.split()[-1] for line in completed.stdout.splitlines() if line.split()
    )
    if actual != sorted(expected):
        raise ModularRecurrentCpuArtifactError("artifact export set mismatch")


def _configure_library(library: ctypes.CDLL) -> None:
    row_p = ctypes.POINTER(RowV1)
    owner_p = ctypes.POINTER(OwnerViewV1)
    const_owner_p = ctypes.POINTER(ConstOwnerViewV1)
    mutation_p = ctypes.POINTER(MutationV1)
    canonical_binding_p = ctypes.POINTER(CanonicalReceiptBindingV1)
    canonical_source_p = ctypes.POINTER(CanonicalReceiptSourceV1)
    uint64_p = ctypes.POINTER(ctypes.c_uint64)
    birth_receipt_p = ctypes.POINTER(BirthReceiptV1)
    library.eiel_modular_recurrent_cpu_oracle_abi_version_v1.restype = ctypes.c_uint32
    library.eiel_modular_recurrent_cpu_oracle_plane_count_v1.restype = ctypes.c_int32
    library.eiel_modular_recurrent_cpu_oracle_row_size_v1.restype = ctypes.c_int64
    library.eiel_modular_recurrent_validate_v1.argtypes = [
        row_p,
        ctypes.c_int64,
        ctypes.c_uint32,
        const_owner_p,
    ]
    library.eiel_modular_recurrent_tick_v1.argtypes = [
        row_p,
        ctypes.c_int64,
        ctypes.c_uint32,
        const_owner_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        row_p,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.POINTER(TickReceiptV1),
    ]
    library.eiel_modular_recurrent_founder_init_v1.argtypes = [
        row_p,
        ctypes.c_int64,
        ctypes.c_uint32,
        owner_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        row_p,
    ]
    library.eiel_modular_recurrent_newborn_inherit_v1.argtypes = [
        row_p,
        ctypes.c_int64,
        ctypes.c_uint32,
        owner_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        mutation_p,
        ctypes.c_int64,
    ]
    library.eiel_modular_recurrent_death_cleanup_v1.argtypes = [
        row_p,
        ctypes.c_int64,
        ctypes.c_uint32,
        owner_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    library.eiel_modular_recurrent_row_structural_payload_sha256_v1.argtypes = [
        ctypes.POINTER(RowV1),
        uint64_p,
    ]
    library.eiel_modular_recurrent_canonical_receipt_sha256_v1.argtypes = [
        ctypes.c_uint32,
        canonical_source_p,
        canonical_binding_p,
        uint64_p,
        uint64_p,
    ]
    library.eiel_modular_recurrent_same_slot_reuse_v1.argtypes = [
        row_p,
        ctypes.c_int64,
        ctypes.c_uint32,
        owner_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        mutation_p,
        ctypes.c_int64,
    ]
    library.eiel_modular_recurrent_newborn_inherit_with_canonical_receipt_v1.argtypes = [
        row_p,
        ctypes.c_int64,
        ctypes.c_uint32,
        owner_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        mutation_p,
        ctypes.c_int64,
        canonical_source_p,
        canonical_binding_p,
        birth_receipt_p,
    ]
    library.eiel_modular_recurrent_checkpoint_project_v1.argtypes = [
        row_p,
        ctypes.c_int64,
        ctypes.c_uint32,
        const_owner_p,
        ctypes.c_int64,
        ctypes.c_int64,
        row_p,
    ]
    library.eiel_modular_recurrent_checkpoint_restore_v1.argtypes = [
        row_p,
        row_p,
        ctypes.c_int64,
        ctypes.c_uint32,
        owner_p,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    library.eiel_modular_recurrent_cost_capacity_v1.argtypes = [
        row_p,
        ctypes.c_uint32,
        ctypes.POINTER(CostCapacityReceiptV1),
    ]
    library.eiel_modular_recurrent_same_slot_reuse_with_canonical_receipt_v1.argtypes = [
        row_p,
        ctypes.c_int64,
        ctypes.c_uint32,
        owner_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        mutation_p,
        ctypes.c_int64,
        canonical_source_p,
        canonical_binding_p,
        birth_receipt_p,
    ]
    for name in (
        "eiel_modular_recurrent_validate_v1",
        "eiel_modular_recurrent_tick_v1",
        "eiel_modular_recurrent_founder_init_v1",
        "eiel_modular_recurrent_newborn_inherit_v1",
        "eiel_modular_recurrent_newborn_inherit_with_canonical_receipt_v1",
        "eiel_modular_recurrent_row_structural_payload_sha256_v1",
        "eiel_modular_recurrent_canonical_receipt_sha256_v1",
        "eiel_modular_recurrent_death_cleanup_v1",
        "eiel_modular_recurrent_same_slot_reuse_v1",
        "eiel_modular_recurrent_same_slot_reuse_with_canonical_receipt_v1",
        "eiel_modular_recurrent_checkpoint_project_v1",
        "eiel_modular_recurrent_checkpoint_restore_v1",
        "eiel_modular_recurrent_cost_capacity_v1",
    ):
        getattr(library, name).restype = ctypes.c_int32


def build_prebuilt_modular_recurrent_cpu_oracle(
    output_directory: str | Path,
    *,
    manifest_path: str | Path = DEFAULT_BUILD_MANIFEST,
) -> Path:
    payload, _manifest = _load_manifest(manifest_path)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / payload["artifact"]["file_name"]
    temporary = output / f".{artifact.name}.tmp"
    command = [
        str(temporary) if item == "{artifact}" else item
        for item in payload["build"]["command"]
    ]
    environment = os.environ.copy()
    environment.update(payload["build"]["environment"])
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise ModularRecurrentCpuArtifactError(
            f"artifact build failed: {completed.stderr.strip()}"
        )
    try:
        _verify_artifact(temporary, payload["artifact"], require_final_name=False)
        _verify_exported_symbols(temporary, payload["c_abi"]["exported_symbols"])
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(artifact)
    return artifact


def load_prebuilt_modular_recurrent_cpu_oracle(
    artifact_path: str | Path,
    *,
    manifest_path: str | Path = DEFAULT_BUILD_MANIFEST,
) -> ModularRecurrentCpuArtifactV1:
    payload, manifest = _load_manifest(manifest_path)
    artifact = Path(artifact_path).resolve()
    digest = _verify_artifact(artifact, payload["artifact"])
    _verify_exported_symbols(artifact, payload["c_abi"]["exported_symbols"])
    try:
        library = ctypes.CDLL(str(artifact))
    except OSError as exc:
        raise ModularRecurrentCpuArtifactError("cannot load verified artifact") from exc
    _configure_library(library)
    if library.eiel_modular_recurrent_cpu_oracle_abi_version_v1() != ABI_VERSION:
        raise ModularRecurrentCpuArtifactError("loaded artifact ABI version mismatch")
    if library.eiel_modular_recurrent_cpu_oracle_plane_count_v1() != PLANE_COUNT:
        raise ModularRecurrentCpuArtifactError("loaded artifact plane count mismatch")
    if library.eiel_modular_recurrent_cpu_oracle_row_size_v1() != ctypes.sizeof(RowV1):
        raise ModularRecurrentCpuArtifactError("loaded artifact row layout mismatch")
    return ModularRecurrentCpuArtifactV1(library, artifact, digest, manifest)


def copy_prebuilt_modular_recurrent_cpu_oracle(
    source: str | Path,
    destination_directory: str | Path,
    *,
    manifest_path: str | Path = DEFAULT_BUILD_MANIFEST,
) -> Path:
    payload, _manifest = _load_manifest(manifest_path)
    source_path = Path(source).resolve()
    _verify_artifact(source_path, payload["artifact"])
    destination = (
        Path(destination_directory).resolve() / payload["artifact"]["file_name"]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination)
    _verify_artifact(destination, payload["artifact"])
    return destination


__all__ = [
    "ABI_VERSION",
    "ACTION_WIDTH",
    "BirthReceiptV1",
    "CanonicalReceiptBindingV1",
    "CONNECTION_SLOTS",
    "COST_FIELDS",
    "ConstOwnerViewV1",
    "CostCapacityReceiptV1",
    "DEFAULT_BUILD_MANIFEST",
    "MODULE_SLOTS",
    "MutationV1",
    "OBSERVATION_WIDTH",
    "OwnerViewV1",
    "PLANE_COUNT",
    "PROJECT_ROOT",
    "ROW_PLANE_FIELDS",
    "RowV1",
    "TickReceiptV1",
    "ModularRecurrentCpuArtifactError",
    "ModularRecurrentCpuArtifactV1",
    "build_prebuilt_modular_recurrent_cpu_oracle",
    "copy_prebuilt_modular_recurrent_cpu_oracle",
    "load_prebuilt_modular_recurrent_cpu_oracle",
]
