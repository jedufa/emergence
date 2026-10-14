"""Verified framework-free native primitives for compact science reduction.

This module is deliberately a control-plane binding.  Campaign execution may
load only an already-built, hash-bound shared object; compilation is an
explicit engineering/install action and never occurs while a campaign runs.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Final, Mapping, Sequence

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_RELATIVE_PATH: Final[Path] = Path(
    "_artifacts/native-compact-reducer-100x-v1/native-default-build-v1"
)
SOURCE_PATHS: Final[tuple[str, ...]] = (
    "emergence_evidence/native/compact_reducer_core_v1.h",
    "emergence_evidence/native/compact_reducer_internal_v1.h",
    "emergence_evidence/native/compact_reducer_core_v1.cpp",
    "emergence_evidence/native/compact_reducer_state_v1.cpp",
    "emergence_evidence/native/compact_reducer_exports_v1.map",
)
ARTIFACT_FILE_NAME: Final[str] = "libeiel_native_compact_reducer_v1.so"
MANIFEST_FILE_NAME: Final[str] = "eiel_native_compact_reducer_build_manifest_v1.json"
ABI_VERSION: Final[int] = 1
GENOME_ROW_BYTES: Final[int] = 288
PACKED_BIRTH_PROVENANCE_BYTES: Final[int] = 68
LINKED_LOCI: Final[int] = 128
RECOMBINATION_CHOICES: Final[int] = 32
DEATH_METRIC_FIELDS: Final[int] = 21
DEATH_METRIC_ROW_BYTES: Final[int] = 88
STRATEGY_MAX_IDS: Final[int] = 64
STRATEGY_MAX_THRESHOLDS: Final[int] = 16
STRATEGY_MAX_GROUPS: Final[int] = 256
STRATEGY_MAX_PARENT_PAIRS: Final[int] = 1024
SNAPSHOT_MAX_POLICY_LOCI: Final[int] = 112
SNAPSHOT_MAX_TRAIT_LOCI: Final[int] = 16
EXPORTED_SYMBOLS: Final[tuple[str, ...]] = (
    "eiel_native_compact_accumulate_death_metrics_v1",
    "eiel_native_compact_reconstruct_birth_batch_v1",
    "eiel_native_compact_reconstruct_birth_v1",
    "eiel_native_compact_reducer_abi_version_v1",
    "eiel_native_compact_reducer_active_record_size_v1",
    "eiel_native_compact_reducer_birth_receipt_size_v1",
    "eiel_native_compact_reducer_strategy_receipt_size_v1",
    "eiel_native_compact_reducer_world_receipt_size_v1",
    "eiel_native_compact_state_create_v1",
    "eiel_native_compact_state_configure_strategy_v1",
    "eiel_native_compact_state_configure_snapshot_world_v1",
    "eiel_native_compact_state_configure_genotype_attribution_v1",
    "eiel_native_compact_state_configure_genotype_focal_ids_v1",
    "eiel_native_compact_state_configure_hereditary_birth_output_v1",
    "eiel_native_compact_state_copy_snapshot_result_v1",
    "eiel_native_compact_state_destroy_v1",
    "eiel_native_compact_state_export_active_v1",
    "eiel_native_compact_state_failure_v1",
    "eiel_native_compact_state_finalize_strategy_v1",
    "eiel_native_compact_state_finalize_hereditary_birth_output_v1",
    "eiel_native_compact_state_genotype_result_size_v1",
    "eiel_native_compact_state_copy_genotype_result_v1",
    "eiel_native_compact_state_initialize_world_v1",
    "eiel_native_compact_state_observe_strategy_snapshot_v1",
    "eiel_native_compact_state_process_dispatch_v1",
    "eiel_native_compact_state_process_snapshot_v1",
    "eiel_native_compact_state_snapshot_result_size_v1",
    "eiel_native_compact_state_strategy_receipt_v1",
    "eiel_native_compact_state_world_receipt_v1",
)


class NativeCompactReducerError(RuntimeError):
    """The native reducer artifact or invocation failed closed."""


class BirthBatchReceiptV1(ctypes.Structure):
    _fields_ = [
        ("reconstructed_birth_count", ctypes.c_uint64),
        ("mutation_origin_count", ctypes.c_uint64),
        ("saturated_noop_count", ctypes.c_uint64),
        ("recombination_nonzero_pattern_count", ctypes.c_uint64),
        (
            "mutation_origin_counts_by_linked_locus",
            ctypes.c_uint64 * LINKED_LOCI,
        ),
        (
            "mutation_delta_totals_by_linked_locus",
            ctypes.c_int64 * LINKED_LOCI,
        ),
        (
            "recombination_choice_one_counts",
            ctypes.c_uint64 * RECOMBINATION_CHOICES,
        ),
    ]


class I128V1(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint64), ("high", ctypes.c_int64)]


class ActiveRecordV1(ctypes.Structure):
    _fields_ = [
        ("agent_id", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("strategy_mask", ctypes.c_uint64),
        ("birth_tick", ctypes.c_int64),
        ("mutation_lineage", ctypes.c_uint8),
        ("born", ctypes.c_uint8),
        ("reproduced", ctypes.c_uint8),
        ("founder", ctypes.c_uint8),
        ("latest_metrics_present", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
        ("genome", ctypes.c_uint8 * GENOME_ROW_BYTES),
    ]


class WorldReceiptV1(ctypes.Structure):
    _fields_ = [
        ("birth_count", ctypes.c_uint64),
        ("death_count", ctypes.c_uint64),
        ("recombination_birth_count", ctypes.c_uint64),
        ("carrier_reproduction_birth_count", ctypes.c_uint64),
        ("mutation_carrier_reproduction_birth_count", ctypes.c_uint64),
        ("unique_born_agents_reproduced_count", ctypes.c_uint64),
        ("unique_mutation_carriers_reproduced_count", ctypes.c_uint64),
        ("maximum_lineage_depth", ctypes.c_uint64),
        ("dispatch_count", ctypes.c_uint64),
        ("event_count", ctypes.c_uint64),
        ("dispatch_tick_count", ctypes.c_uint64),
        ("next_absolute_tick", ctypes.c_int64),
        ("active_agent_count", ctypes.c_uint64),
        ("births", BirthBatchReceiptV1),
        ("completed_lifetime_metric_totals", I128V1 * DEATH_METRIC_FIELDS),
    ]


class FailureV1(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("world_index", ctypes.c_uint32),
        ("absolute_tick", ctypes.c_int64),
        ("event_row_index", ctypes.c_int64),
        ("agent_id", ctypes.c_uint64),
        ("parent_a_id", ctypes.c_uint64),
        ("parent_b_id", ctypes.c_uint64),
    ]


class StrategyGroupCountV1(ctypes.Structure):
    _fields_ = [
        ("strategy_mask", ctypes.c_uint64),
        ("count", ctypes.c_uint64),
    ]


class StrategyParentPairCountV1(ctypes.Structure):
    _fields_ = [
        ("first_strategy_mask", ctypes.c_uint64),
        ("second_strategy_mask", ctypes.c_uint64),
        ("count", ctypes.c_uint64),
    ]


class StrategyReceiptV1(ctypes.Structure):
    _fields_ = [
        ("enabled", ctypes.c_uint8),
        ("finalized", ctypes.c_uint8),
        ("strategy_count", ctypes.c_uint8),
        ("threshold_count", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 4),
        ("early_life_tick", ctypes.c_int64),
        ("first_birth_tick", ctypes.c_int64),
        ("first_death_tick", ctypes.c_int64),
        ("extinction_tick", ctypes.c_int64),
        ("total_agent_count", ctypes.c_uint64),
        ("founder_count", ctypes.c_uint64),
        ("birth_count", ctypes.c_uint64),
        ("death_count", ctypes.c_uint64),
        ("maximum_lineage_depth", ctypes.c_uint64),
        ("mixed_parent_birth_count", ctypes.c_uint64),
        ("threshold_ticks", ctypes.c_int64 * STRATEGY_MAX_THRESHOLDS),
        ("agent_counts_by_strategy", ctypes.c_uint64 * STRATEGY_MAX_IDS),
        ("offspring_born_by_strategy", ctypes.c_uint64 * STRATEGY_MAX_IDS),
        (
            "offspring_survived_by_strategy_threshold",
            ctypes.c_uint64 * (STRATEGY_MAX_IDS * STRATEGY_MAX_THRESHOLDS),
        ),
        (
            "maximum_lineage_depth_by_strategy",
            ctypes.c_uint64 * STRATEGY_MAX_IDS,
        ),
        ("parent_birth_events_by_strategy", ctypes.c_uint64 * STRATEGY_MAX_IDS),
        ("death_events_by_strategy", ctypes.c_uint64 * STRATEGY_MAX_IDS),
        (
            "persistence_to_early_life_by_strategy",
            ctypes.c_uint64 * STRATEGY_MAX_IDS,
        ),
        ("terminal_alive_by_strategy", ctypes.c_uint64 * STRATEGY_MAX_IDS),
        (
            "terminal_metric_totals_by_strategy",
            I128V1 * (STRATEGY_MAX_IDS * DEATH_METRIC_FIELDS),
        ),
        ("offspring_group_count", ctypes.c_uint64),
        ("offspring_groups", StrategyGroupCountV1 * STRATEGY_MAX_GROUPS),
        ("parent_pair_count", ctypes.c_uint64),
        (
            "parent_pairs",
            StrategyParentPairCountV1 * STRATEGY_MAX_PARENT_PAIRS,
        ),
    ]


@dataclass(frozen=True)
class NativeBirthBatchResultV1:
    child_genomes: bytes
    reconstructed_birth_count: int
    mutation_origin_count: int
    saturated_noop_count: int
    recombination_nonzero_pattern_count: int
    mutation_origin_counts_by_linked_locus: tuple[int, ...]
    mutation_delta_totals_by_linked_locus: tuple[int, ...]
    recombination_choice_one_counts: tuple[int, ...]


@dataclass(frozen=True)
class NativeActiveRecordResultV1:
    agent_id: int
    generation: int
    strategy_mask: int
    birth_tick: int
    mutation_lineage: bool
    born: bool
    reproduced: bool
    founder: bool
    latest_metrics_present: bool
    genome: bytes


@dataclass(frozen=True)
class NativeStrategyReceiptResultV1:
    enabled: bool
    finalized: bool
    strategy_count: int
    threshold_ticks: tuple[int, ...]
    early_life_tick: int
    first_birth_tick: int | None
    first_death_tick: int | None
    extinction_tick: int | None
    total_agent_count: int
    founder_count: int
    birth_count: int
    death_count: int
    maximum_lineage_depth: int
    mixed_parent_birth_count: int
    agent_counts_by_strategy: tuple[int, ...]
    offspring_born_by_strategy: tuple[int, ...]
    offspring_survived_by_strategy_threshold: tuple[tuple[int, ...], ...]
    maximum_lineage_depth_by_strategy: tuple[int, ...]
    parent_birth_events_by_strategy: tuple[int, ...]
    death_events_by_strategy: tuple[int, ...]
    persistence_to_early_life_by_strategy: tuple[int, ...]
    terminal_alive_by_strategy: tuple[int, ...]
    terminal_metric_totals_by_strategy: tuple[tuple[int, ...], ...]
    offspring_group_counts: tuple[tuple[int, int], ...]
    parent_pair_counts: tuple[tuple[int, int, int], ...]
    genotype_reproductive_attribution: dict[str, Any] | None


@dataclass(frozen=True)
class NativeWorldReceiptResultV1:
    birth_count: int
    death_count: int
    recombination_birth_count: int
    carrier_reproduction_birth_count: int
    mutation_carrier_reproduction_birth_count: int
    unique_born_agents_reproduced_count: int
    unique_mutation_carriers_reproduced_count: int
    maximum_lineage_depth: int
    dispatch_count: int
    event_count: int
    dispatch_tick_count: int
    next_absolute_tick: int
    active_agent_count: int
    births: NativeBirthBatchResultV1
    completed_lifetime_metric_totals: tuple[int, ...]


@dataclass(frozen=True)
class NativeCompactReducerArtifactV1:
    library: ctypes.CDLL
    artifact_path: Path
    artifact_sha256: str
    manifest_path: Path
    manifest_sha256: str
    production_routing_qualified: bool


_STATUS_NAMES: Final[dict[int, str]] = {
    0: "success",
    1: "null_pointer",
    2: "invalid_extent",
    3: "invalid_mutation_step",
    4: "reserved_mutation_code",
    5: "counter_overflow",
    6: "allocation_failure",
    7: "invalid_state",
    8: "invalid_world",
    9: "noncontiguous_tick",
    10: "invalid_event",
    11: "inactive_agent",
    12: "duplicate_agent",
    13: "capacity_exceeded",
    14: "auxiliary_cursor",
    15: "poisoned",
    16: "thread_failure",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeCompactReducerError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise NativeCompactReducerError(f"{label} keys drifted")


def _compiler_identity() -> str:
    completed = subprocess.run(
        ["g++", "--version"], check=False, capture_output=True, text=True
    )
    if completed.returncode != 0 or not completed.stdout.splitlines():
        raise NativeCompactReducerError("g++ is unavailable")
    return completed.stdout.splitlines()[0]


def _build_command(artifact_token: str = "{artifact}") -> list[str]:
    return [
        "g++",
        "-std=c++17",
        "-O3",
        "-march=x86-64-v3",
        "-mtune=generic",
        "-fPIC",
        "-shared",
        "-Wall",
        "-Wextra",
        "-Wconversion",
        "-Wsign-conversion",
        "-Wshadow",
        "-Werror",
        "-fno-rtti",
        "-fvisibility=hidden",
        "-fvisibility-inlines-hidden",
        "-fno-ident",
        "-ffp-contract=off",
        "-frandom-seed=eiel-native-compact-reducer-v1",
        "-Wl,--build-id=none",
        "-Wl,-z,defs",
        "-Wl,-z,relro,-z,now",
        f"-Wl,--version-script={SOURCE_PATHS[4]}",
        SOURCE_PATHS[2],
        SOURCE_PATHS[3],
        "-pthread",
        "-lz",
        "-o",
        artifact_token,
    ]


def _verify_exported_symbols(artifact: Path, expected: Sequence[str]) -> None:
    completed = subprocess.run(
        ["nm", "-D", "--defined-only", str(artifact)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise NativeCompactReducerError("cannot inspect native reducer exports")
    actual = sorted(
        line.split()[-1] for line in completed.stdout.splitlines() if line.split()
    )
    if actual != sorted(expected):
        raise NativeCompactReducerError("native reducer export set drifted")


def _artifact_receipt(path: Path) -> dict[str, Any]:
    return {
        "file_name": ARTIFACT_FILE_NAME,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build_native_compact_reducer_v1(output_directory: str | Path) -> tuple[Path, Path]:
    """Build a deterministic development artifact and its fail-closed manifest."""

    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / ARTIFACT_FILE_NAME
    temporary_artifact = output / f".{ARTIFACT_FILE_NAME}.tmp"
    manifest = output / MANIFEST_FILE_NAME
    temporary_manifest = output / f".{MANIFEST_FILE_NAME}.tmp"
    temporary_artifact.unlink(missing_ok=True)
    temporary_manifest.unlink(missing_ok=True)
    command = _build_command(str(temporary_artifact))
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
        temporary_artifact.unlink(missing_ok=True)
        raise NativeCompactReducerError(
            f"native reducer build failed: {completed.stderr.strip()}"
        )
    try:
        _verify_exported_symbols(temporary_artifact, EXPORTED_SYMBOLS)
        compiler_identity = _compiler_identity()
        payload = {
            "schema": "eiel_native_compact_reducer_build_manifest_v1",
            "schema_version": 1,
            "runtime_jit": False,
            "production_routing_qualified": False,
            "publication_qualified": False,
            "target": {
                "platform": "linux",
                "architecture": "x86_64",
                "abi": "LP64",
                "endianness": "little",
                "cxx_standard": "c++17",
                "cpu_isa": "x86-64-v3",
            },
            "toolchain": {"compiler": "g++", "identity": compiler_identity},
            "sources": [
                {"path": relative, "sha256": _sha256(PROJECT_ROOT / relative)}
                for relative in SOURCE_PATHS
            ],
            "build": {
                "command": _build_command(),
                "environment": {"LC_ALL": "C", "SOURCE_DATE_EPOCH": "0"},
                "compiler_identity": compiler_identity,
            },
            "artifact": _artifact_receipt(temporary_artifact),
            "c_abi": {
                "version": ABI_VERSION,
                "birth_receipt_size": ctypes.sizeof(BirthBatchReceiptV1),
                "active_record_size": ctypes.sizeof(ActiveRecordV1),
                "world_receipt_size": ctypes.sizeof(WorldReceiptV1),
                "strategy_receipt_size": ctypes.sizeof(StrategyReceiptV1),
                "exported_symbols": list(EXPORTED_SYMBOLS),
            },
            "qualification": {
                "status": "development_only",
                "qualification_receipt": None,
                "does_not_authorize": [
                    "production campaign routing",
                    "scientific result publication",
                    "qualification inheritance",
                    "runtime compilation",
                ],
            },
        }
        temporary_manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_artifact.replace(artifact)
        temporary_manifest.replace(manifest)
    except Exception:
        temporary_artifact.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        raise
    return artifact, manifest


def default_native_compact_reducer_binding_v1() -> tuple[Path, Path, str]:
    """Return the checked-in, source-bound native reducer installation.

    The campaign path consumes this prebuilt artifact; it never compiles.  The
    manifest and shared object remain independently verified by
    :func:`load_native_compact_reducer_v1` in the reducer sidecar.
    """

    output = PROJECT_ROOT / DEFAULT_BUILD_RELATIVE_PATH
    if output.is_symlink():
        raise NativeCompactReducerError(
            "default native reducer build directory must not be a symlink"
        )
    artifact = output / ARTIFACT_FILE_NAME
    manifest = output / MANIFEST_FILE_NAME
    for path, label in ((artifact, "artifact"), (manifest, "manifest")):
        if path.is_symlink() or not path.is_file():
            raise NativeCompactReducerError(
                f"default native reducer {label} is not a regular file: {path}"
            )
    return artifact, manifest, _sha256(manifest)


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeCompactReducerError(f"cannot read build manifest: {path}") from exc
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
    if (
        root["schema"] != "eiel_native_compact_reducer_build_manifest_v1"
        or root["schema_version"] != 1
        or root["runtime_jit"] is not False
    ):
        raise NativeCompactReducerError("manifest identity or JIT boundary drifted")
    if type(root["production_routing_qualified"]) is not bool:
        raise NativeCompactReducerError("production qualification flag is invalid")
    if type(root["publication_qualified"]) is not bool:
        raise NativeCompactReducerError("publication qualification flag is invalid")
    target = _mapping(root["target"], "target")
    if target != {
        "platform": "linux",
        "architecture": "x86_64",
        "abi": "LP64",
        "endianness": "little",
        "cxx_standard": "c++17",
        "cpu_isa": "x86-64-v3",
    }:
        raise NativeCompactReducerError("native reducer target drifted")
    if (
        platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or os.sys.byteorder != "little"
    ):
        raise NativeCompactReducerError("host does not match native reducer target")
    toolchain = _mapping(root["toolchain"], "toolchain")
    _exact_keys(toolchain, {"compiler", "identity"}, "toolchain")
    if toolchain["compiler"] != "g++" or not isinstance(toolchain["identity"], str):
        raise NativeCompactReducerError("native reducer toolchain receipt drifted")
    sources = root["sources"]
    if (
        not isinstance(sources, list)
        or [entry.get("path") for entry in sources if isinstance(entry, Mapping)]
        != list(SOURCE_PATHS)
        or len(sources) != len(SOURCE_PATHS)
    ):
        raise NativeCompactReducerError("native reducer source inventory drifted")
    for index, source_value in enumerate(sources):
        source = _mapping(source_value, f"sources[{index}]")
        _exact_keys(source, {"path", "sha256"}, f"sources[{index}]")
        relative = Path(str(source["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise NativeCompactReducerError("native reducer source path is invalid")
        absolute = PROJECT_ROOT / relative
        if not absolute.is_file() or _sha256(absolute) != source["sha256"]:
            raise NativeCompactReducerError(f"source hash mismatch: {relative}")
    build = _mapping(root["build"], "build")
    _exact_keys(build, {"command", "environment", "compiler_identity"}, "build")
    if (
        build["command"] != _build_command()
        or build["environment"] != {"LC_ALL": "C", "SOURCE_DATE_EPOCH": "0"}
        or build["compiler_identity"] != toolchain["identity"]
    ):
        raise NativeCompactReducerError("deterministic build receipt drifted")
    artifact = _mapping(root["artifact"], "artifact")
    _exact_keys(artifact, {"file_name", "size_bytes", "sha256"}, "artifact")
    if (
        artifact["file_name"] != ARTIFACT_FILE_NAME
        or type(artifact["size_bytes"]) is not int
        or artifact["size_bytes"] < 1
        or not isinstance(artifact["sha256"], str)
        or len(artifact["sha256"]) != 64
    ):
        raise NativeCompactReducerError("native reducer artifact receipt drifted")
    c_abi = _mapping(root["c_abi"], "c_abi")
    _exact_keys(
        c_abi,
        {
            "version",
            "birth_receipt_size",
            "active_record_size",
            "world_receipt_size",
            "strategy_receipt_size",
            "exported_symbols",
        },
        "c_abi",
    )
    if (
        c_abi["version"] != ABI_VERSION
        or c_abi["birth_receipt_size"] != ctypes.sizeof(BirthBatchReceiptV1)
        or c_abi["active_record_size"] != ctypes.sizeof(ActiveRecordV1)
        or c_abi["world_receipt_size"] != ctypes.sizeof(WorldReceiptV1)
        or c_abi["strategy_receipt_size"] != ctypes.sizeof(StrategyReceiptV1)
        or c_abi["exported_symbols"] != list(EXPORTED_SYMBOLS)
    ):
        raise NativeCompactReducerError("native reducer C ABI receipt drifted")
    qualification = _mapping(root["qualification"], "qualification")
    _exact_keys(
        qualification,
        {"status", "qualification_receipt", "does_not_authorize"},
        "qualification",
    )
    if not isinstance(qualification["does_not_authorize"], list):
        raise NativeCompactReducerError("qualification boundary is invalid")
    return root


def _configure_library(library: ctypes.CDLL) -> None:
    byte_p = ctypes.POINTER(ctypes.c_uint8)
    receipt_p = ctypes.POINTER(BirthBatchReceiptV1)
    i128_p = ctypes.POINTER(I128V1)
    int64_p = ctypes.POINTER(ctypes.c_int64)
    uint16_p = ctypes.POINTER(ctypes.c_uint16)
    uint64_p = ctypes.POINTER(ctypes.c_uint64)
    library.eiel_native_compact_reducer_abi_version_v1.argtypes = []
    library.eiel_native_compact_reducer_abi_version_v1.restype = ctypes.c_uint32
    library.eiel_native_compact_reducer_birth_receipt_size_v1.argtypes = []
    library.eiel_native_compact_reducer_birth_receipt_size_v1.restype = ctypes.c_int64
    library.eiel_native_compact_reducer_active_record_size_v1.argtypes = []
    library.eiel_native_compact_reducer_active_record_size_v1.restype = ctypes.c_int64
    library.eiel_native_compact_reducer_world_receipt_size_v1.argtypes = []
    library.eiel_native_compact_reducer_world_receipt_size_v1.restype = ctypes.c_int64
    library.eiel_native_compact_reducer_strategy_receipt_size_v1.argtypes = []
    library.eiel_native_compact_reducer_strategy_receipt_size_v1.restype = (
        ctypes.c_int64
    )
    library.eiel_native_compact_reconstruct_birth_v1.argtypes = [
        byte_p,
        byte_p,
        byte_p,
        ctypes.c_int32,
        byte_p,
        receipt_p,
    ]
    library.eiel_native_compact_reconstruct_birth_v1.restype = ctypes.c_int32
    library.eiel_native_compact_reconstruct_birth_batch_v1.argtypes = [
        byte_p,
        byte_p,
        byte_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int64,
        byte_p,
        receipt_p,
    ]
    library.eiel_native_compact_reconstruct_birth_batch_v1.restype = ctypes.c_int32
    library.eiel_native_compact_accumulate_death_metrics_v1.argtypes = [
        byte_p,
        ctypes.c_int64,
        i128_p,
    ]
    library.eiel_native_compact_accumulate_death_metrics_v1.restype = ctypes.c_int32
    library.eiel_native_compact_state_create_v1.argtypes = [
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_uint32),
        int64_p,
        int64_p,
        int64_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.eiel_native_compact_state_create_v1.restype = ctypes.c_int32
    library.eiel_native_compact_state_destroy_v1.argtypes = [ctypes.c_void_p]
    library.eiel_native_compact_state_destroy_v1.restype = None
    library.eiel_native_compact_state_initialize_world_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        uint64_p,
        byte_p,
        ctypes.c_int64,
    ]
    library.eiel_native_compact_state_initialize_world_v1.restype = ctypes.c_int32
    library.eiel_native_compact_state_configure_snapshot_world_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        byte_p,
        ctypes.c_int64,
        uint16_p,
        ctypes.c_int32,
        uint16_p,
        ctypes.c_int32,
        ctypes.c_int32,
        byte_p,
        uint64_p,
        ctypes.c_int32,
        int64_p,
        ctypes.c_int32,
        ctypes.c_int64,
    ]
    library.eiel_native_compact_state_configure_snapshot_world_v1.restype = (
        ctypes.c_int32
    )
    library.eiel_native_compact_state_configure_genotype_attribution_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
    ]
    library.eiel_native_compact_state_configure_genotype_attribution_v1.restype = (
        ctypes.c_int32
    )
    library.eiel_native_compact_state_configure_genotype_focal_ids_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        byte_p,
        ctypes.c_int64,
    ]
    library.eiel_native_compact_state_configure_genotype_focal_ids_v1.restype = (
        ctypes.c_int32
    )
    library.eiel_native_compact_state_configure_hereditary_birth_output_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        byte_p,
        ctypes.c_int64,
    ]
    library.eiel_native_compact_state_configure_hereditary_birth_output_v1.restype = (
        ctypes.c_int32
    )
    library.eiel_native_compact_state_finalize_hereditary_birth_output_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
    ]
    library.eiel_native_compact_state_finalize_hereditary_birth_output_v1.restype = (
        ctypes.c_int32
    )
    library.eiel_native_compact_state_process_snapshot_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        byte_p,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    library.eiel_native_compact_state_process_snapshot_v1.restype = ctypes.c_int32
    library.eiel_native_compact_state_snapshot_result_size_v1.argtypes = [
        ctypes.c_void_p,
        int64_p,
    ]
    library.eiel_native_compact_state_snapshot_result_size_v1.restype = (
        ctypes.c_int32
    )
    library.eiel_native_compact_state_copy_snapshot_result_v1.argtypes = [
        ctypes.c_void_p,
        byte_p,
        ctypes.c_int64,
    ]
    library.eiel_native_compact_state_copy_snapshot_result_v1.restype = (
        ctypes.c_int32
    )
    library.eiel_native_compact_state_process_dispatch_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        byte_p,
        ctypes.c_int64,
        byte_p,
        ctypes.c_int64,
        byte_p,
        ctypes.c_int64,
        int64_p,
        ctypes.c_int64,
    ]
    library.eiel_native_compact_state_process_dispatch_v1.restype = ctypes.c_int32
    library.eiel_native_compact_state_world_receipt_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.POINTER(WorldReceiptV1),
    ]
    library.eiel_native_compact_state_world_receipt_v1.restype = ctypes.c_int32
    library.eiel_native_compact_state_export_active_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.POINTER(ActiveRecordV1),
        ctypes.c_int64,
        int64_p,
    ]
    library.eiel_native_compact_state_export_active_v1.restype = ctypes.c_int32
    library.eiel_native_compact_state_failure_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(FailureV1),
    ]
    library.eiel_native_compact_state_failure_v1.restype = ctypes.c_int32
    library.eiel_native_compact_state_configure_strategy_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int32,
        uint64_p,
        uint64_p,
        ctypes.c_int64,
        int64_p,
        ctypes.c_int32,
        ctypes.c_int64,
    ]
    library.eiel_native_compact_state_configure_strategy_v1.restype = ctypes.c_int32
    library.eiel_native_compact_state_observe_strategy_snapshot_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        uint64_p,
        byte_p,
        ctypes.c_int64,
    ]
    library.eiel_native_compact_state_observe_strategy_snapshot_v1.restype = (
        ctypes.c_int32
    )
    library.eiel_native_compact_state_finalize_strategy_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    library.eiel_native_compact_state_finalize_strategy_v1.restype = ctypes.c_int32
    library.eiel_native_compact_state_strategy_receipt_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.POINTER(StrategyReceiptV1),
    ]
    library.eiel_native_compact_state_strategy_receipt_v1.restype = ctypes.c_int32
    library.eiel_native_compact_state_genotype_result_size_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        int64_p,
    ]
    library.eiel_native_compact_state_genotype_result_size_v1.restype = ctypes.c_int32
    library.eiel_native_compact_state_copy_genotype_result_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        byte_p,
        ctypes.c_int64,
    ]
    library.eiel_native_compact_state_copy_genotype_result_v1.restype = ctypes.c_int32


def load_native_compact_reducer_v1(
    artifact_path: str | Path,
    manifest_path: str | Path,
    *,
    require_production_qualified: bool = True,
    expected_manifest_sha256: str | None = None,
) -> NativeCompactReducerArtifactV1:
    """Load only an exact prebuilt artifact; never compile on this path."""

    artifact = Path(artifact_path).resolve()
    manifest = Path(manifest_path).resolve()
    manifest_sha256 = _sha256(manifest)
    if expected_manifest_sha256 is not None and (
        len(expected_manifest_sha256) != 64
        or manifest_sha256 != expected_manifest_sha256
    ):
        raise NativeCompactReducerError("native reducer manifest hash mismatch")
    if require_production_qualified and expected_manifest_sha256 is None:
        raise NativeCompactReducerError(
            "production native reducer requires a plan-bound manifest hash"
        )
    payload = _load_manifest(manifest)
    production_qualified = payload["production_routing_qualified"]
    qualification = _mapping(payload["qualification"], "qualification")
    if require_production_qualified and (
        production_qualified is not True
        or qualification["status"] != "qualified"
        or not isinstance(qualification["qualification_receipt"], Mapping)
    ):
        raise NativeCompactReducerError(
            "native reducer artifact is not production-routing qualified"
        )
    receipt = _mapping(payload["artifact"], "artifact")
    if (
        not artifact.is_file()
        or artifact.is_symlink()
        or artifact.name != receipt["file_name"]
        or artifact.stat().st_size != receipt["size_bytes"]
    ):
        raise NativeCompactReducerError("native reducer artifact identity mismatch")
    digest = _sha256(artifact)
    if digest != receipt["sha256"]:
        raise NativeCompactReducerError("native reducer artifact hash mismatch")
    _verify_exported_symbols(artifact, payload["c_abi"]["exported_symbols"])
    try:
        library = ctypes.CDLL(str(artifact))
    except OSError as exc:
        raise NativeCompactReducerError("cannot load verified native reducer") from exc
    _configure_library(library)
    if library.eiel_native_compact_reducer_abi_version_v1() != ABI_VERSION:
        raise NativeCompactReducerError("loaded native reducer ABI version mismatch")
    if library.eiel_native_compact_reducer_birth_receipt_size_v1() != ctypes.sizeof(
        BirthBatchReceiptV1
    ):
        raise NativeCompactReducerError("loaded native reducer receipt layout mismatch")
    if library.eiel_native_compact_reducer_active_record_size_v1() != ctypes.sizeof(
        ActiveRecordV1
    ):
        raise NativeCompactReducerError("loaded native active record layout mismatch")
    if library.eiel_native_compact_reducer_world_receipt_size_v1() != ctypes.sizeof(
        WorldReceiptV1
    ):
        raise NativeCompactReducerError("loaded native world receipt layout mismatch")
    if library.eiel_native_compact_reducer_strategy_receipt_size_v1() != ctypes.sizeof(
        StrategyReceiptV1
    ):
        raise NativeCompactReducerError(
            "loaded native strategy receipt layout mismatch"
        )
    return NativeCompactReducerArtifactV1(
        library=library,
        artifact_path=artifact,
        artifact_sha256=digest,
        manifest_path=manifest,
        manifest_sha256=manifest_sha256,
        production_routing_qualified=production_qualified,
    )


def _byte_array(value: bytes) -> ctypes.Array[ctypes.c_uint8]:
    array_type = ctypes.c_uint8 * len(value)
    return array_type.from_buffer_copy(value)


def _raise_status(status: int, operation: str) -> None:
    if status != 0:
        name = _STATUS_NAMES.get(status, f"unknown_status_{status}")
        raise NativeCompactReducerError(f"{operation} failed: {name}")


def _birth_result(
    receipt: BirthBatchReceiptV1, *, child_genomes: bytes = b""
) -> NativeBirthBatchResultV1:
    return NativeBirthBatchResultV1(
        child_genomes=child_genomes,
        reconstructed_birth_count=int(receipt.reconstructed_birth_count),
        mutation_origin_count=int(receipt.mutation_origin_count),
        saturated_noop_count=int(receipt.saturated_noop_count),
        recombination_nonzero_pattern_count=int(
            receipt.recombination_nonzero_pattern_count
        ),
        mutation_origin_counts_by_linked_locus=tuple(
            int(value) for value in receipt.mutation_origin_counts_by_linked_locus
        ),
        mutation_delta_totals_by_linked_locus=tuple(
            int(value) for value in receipt.mutation_delta_totals_by_linked_locus
        ),
        recombination_choice_one_counts=tuple(
            int(value) for value in receipt.recombination_choice_one_counts
        ),
    )


class NativeCompactReducerStateV1:
    """Owned persistent native state; dispatch calls release the Python GIL."""

    def __init__(
        self,
        artifact: NativeCompactReducerArtifactV1,
        handle: ctypes.c_void_p,
        world_count: int,
    ) -> None:
        self.artifact = artifact
        self._handle = handle
        self.world_count = world_count

    def _require_open(self) -> ctypes.c_void_p:
        if not self._handle:
            raise NativeCompactReducerError("native compact reducer state is closed")
        return self._handle

    def _raise_state_status(self, status: int, operation: str) -> None:
        if status == 0:
            return
        details = FailureV1()
        handle = self._handle
        if (
            handle
            and self.artifact.library.eiel_native_compact_state_failure_v1(
                handle, ctypes.byref(details)
            )
            == 0
            and details.status != 0
        ):
            name = _STATUS_NAMES.get(status, f"unknown_status_{status}")
            raise NativeCompactReducerError(
                f"{operation} failed: {name}; world={details.world_index}, "
                f"tick={details.absolute_tick}, row={details.event_row_index}, "
                f"agent={details.agent_id}, parents=({details.parent_a_id},"
                f"{details.parent_b_id})"
            )
        _raise_status(status, operation)

    def initialize_world(
        self,
        world_index: int,
        *,
        agent_ids: Sequence[int],
        packed_genomes: bytes,
    ) -> None:
        if (
            type(world_index) is not int
            or not 0 <= world_index < self.world_count
            or len(packed_genomes) != len(agent_ids) * GENOME_ROW_BYTES
            or any(
                type(agent_id) is not int or not 0 < agent_id < 2**64 - 1
                for agent_id in agent_ids
            )
        ):
            raise NativeCompactReducerError("native founder inputs are invalid")
        ids = (ctypes.c_uint64 * len(agent_ids))(*agent_ids) if agent_ids else None
        genomes = _byte_array(packed_genomes) if packed_genomes else None
        status = self.artifact.library.eiel_native_compact_state_initialize_world_v1(
            self._require_open(), world_index, ids, genomes, len(agent_ids)
        )
        self._raise_state_status(status, "native founder initialization")

    def configure_snapshot_world(
        self,
        world_index: int,
        *,
        cell_id: str,
        tracked_policy_loci: Sequence[int],
        tracked_trait_loci: Sequence[int],
        strategy_count: int = 0,
        strategy_policy_masks: Mapping[str, int] | None = None,
        threshold_ticks: Sequence[int] = (),
        early_life_tick: int = 0,
    ) -> None:
        policy_masks = dict(strategy_policy_masks or {})
        maximum_mask = (
            2**strategy_count - 1
            if type(strategy_count) is int and 1 <= strategy_count <= STRATEGY_MAX_IDS
            else 0
        )
        try:
            cell = cell_id.encode("ascii")
            policy_digests = b"".join(
                bytes.fromhex(policy_id.removeprefix("sha256:"))
                for policy_id in policy_masks
            )
        except (UnicodeEncodeError, ValueError) as exc:
            raise NativeCompactReducerError(
                "native snapshot configuration identity is invalid"
            ) from exc
        if (
            type(world_index) is not int
            or not 0 <= world_index < self.world_count
            or not cell
            or len(cell) > 4096
            or tuple(tracked_policy_loci)
            != tuple(sorted(set(tracked_policy_loci)))
            or tuple(tracked_trait_loci)
            != tuple(sorted(set(tracked_trait_loci)))
            or any(
                type(locus) is not int or not 0 <= locus < SNAPSHOT_MAX_POLICY_LOCI
                for locus in tracked_policy_loci
            )
            or any(
                type(locus) is not int or not 0 <= locus < SNAPSHOT_MAX_TRAIT_LOCI
                for locus in tracked_trait_loci
            )
            or type(strategy_count) is not int
            or not 0 <= strategy_count <= STRATEGY_MAX_IDS
            or (
                strategy_count == 0
                and (policy_masks or tuple(threshold_ticks))
            )
            or (
                strategy_count > 0
                and (not policy_masks or not tuple(threshold_ticks))
            )
            or len(policy_masks) > STRATEGY_MAX_IDS
            or len(policy_digests) != len(policy_masks) * 32
            or any(
                not isinstance(policy_id, str)
                or not policy_id.startswith("sha256:")
                or len(policy_id) != 71
                for policy_id in policy_masks
            )
            or any(
                type(mask) is not int or not 0 < mask <= maximum_mask
                for mask in policy_masks.values()
            )
            or tuple(threshold_ticks) != tuple(sorted(set(threshold_ticks)))
            or len(threshold_ticks) > STRATEGY_MAX_THRESHOLDS
            or any(
                type(tick) is not int or not 0 <= tick < 2**63
                for tick in threshold_ticks
            )
            or type(early_life_tick) is not int
            or not 0 <= early_life_tick < 2**63
        ):
            raise NativeCompactReducerError(
                "native snapshot configuration is invalid"
            )
        cell_bytes = _byte_array(cell)
        policy_loci = (
            (ctypes.c_uint16 * len(tracked_policy_loci))(*tracked_policy_loci)
            if tracked_policy_loci
            else None
        )
        trait_loci = (
            (ctypes.c_uint16 * len(tracked_trait_loci))(*tracked_trait_loci)
            if tracked_trait_loci
            else None
        )
        digest_rows = _byte_array(policy_digests) if policy_digests else None
        masks = (
            (ctypes.c_uint64 * len(policy_masks))(*policy_masks.values())
            if policy_masks
            else None
        )
        thresholds = (
            (ctypes.c_int64 * len(threshold_ticks))(*threshold_ticks)
            if threshold_ticks
            else None
        )
        status = self.artifact.library.eiel_native_compact_state_configure_snapshot_world_v1(
            self._require_open(),
            world_index,
            cell_bytes,
            len(cell),
            policy_loci,
            len(tracked_policy_loci),
            trait_loci,
            len(tracked_trait_loci),
            strategy_count,
            digest_rows,
            masks,
            len(policy_masks),
            thresholds,
            len(threshold_ticks),
            early_life_tick,
        )
        self._raise_state_status(status, "native snapshot configuration")

    def configure_genotype_attribution(
        self,
        world_index: int,
        *,
        focal_ids: Sequence[str] | None = None,
    ) -> None:
        if type(world_index) is not int or not 0 <= world_index < self.world_count:
            raise NativeCompactReducerError(
                "native genotype-attribution world index is invalid"
            )
        status = (
            self.artifact.library.eiel_native_compact_state_configure_genotype_attribution_v1(
                self._require_open(), world_index
            )
        )
        self._raise_state_status(status, "native genotype-attribution configuration")
        if focal_ids is None:
            return
        if (
            not focal_ids
            or tuple(focal_ids) != tuple(sorted(set(focal_ids)))
            or any(
                not isinstance(value, str)
                or len(value) != 71
                or not value.startswith("sha256:")
                or any(character not in "0123456789abcdef" for character in value[7:])
                for value in focal_ids
            )
        ):
            raise NativeCompactReducerError(
                "native genotype focal identities are invalid"
            )
        encoded_ids = _byte_array(
            b"".join(bytes.fromhex(value[7:]) for value in focal_ids)
        )
        status = (
            self.artifact.library.eiel_native_compact_state_configure_genotype_focal_ids_v1(
                self._require_open(),
                world_index,
                encoded_ids,
                len(focal_ids),
            )
        )
        self._raise_state_status(status, "native genotype focal-identity configuration")

    def configure_hereditary_birth_output(
        self, world_index: int, path: str | Path
    ) -> None:
        if type(world_index) is not int or not 0 <= world_index < self.world_count:
            raise NativeCompactReducerError(
                "native hereditary-output world index is invalid"
            )
        path_bytes = os.fsencode(str(Path(path).expanduser().absolute()))
        if not 1 <= len(path_bytes) <= 4096 or b"\x00" in path_bytes:
            raise NativeCompactReducerError(
                "native hereditary-output path is invalid"
            )
        encoded = _byte_array(path_bytes)
        status = (
            self.artifact.library.eiel_native_compact_state_configure_hereditary_birth_output_v1(
                self._require_open(), world_index, encoded, len(path_bytes)
            )
        )
        self._raise_state_status(status, "native hereditary-output configuration")

    def finalize_hereditary_birth_output(self, world_index: int) -> None:
        if type(world_index) is not int or not 0 <= world_index < self.world_count:
            raise NativeCompactReducerError(
                "native hereditary-output world index is invalid"
            )
        status = (
            self.artifact.library.eiel_native_compact_state_finalize_hereditary_birth_output_v1(
                self._require_open(), world_index
            )
        )
        self._raise_state_status(status, "native hereditary-output finalization")

    def process_snapshot(
        self,
        *,
        absolute_tick: int,
        canonical_json_lines: bytes,
        row_count: int,
    ) -> Mapping[str, Any]:
        if (
            type(absolute_tick) is not int
            or not 0 <= absolute_tick < 2**63
            or not isinstance(canonical_json_lines, bytes)
            or not canonical_json_lines
            or type(row_count) is not int
            or row_count < 1
        ):
            raise NativeCompactReducerError("native snapshot input is invalid")
        source_owner = ctypes.c_char_p(canonical_json_lines)
        source = ctypes.cast(source_owner, ctypes.POINTER(ctypes.c_uint8))
        status = self.artifact.library.eiel_native_compact_state_process_snapshot_v1(
            self._require_open(),
            absolute_tick,
            source,
            len(canonical_json_lines),
            row_count,
        )
        self._raise_state_status(status, "native snapshot reduction")
        result_size = ctypes.c_int64()
        status = self.artifact.library.eiel_native_compact_state_snapshot_result_size_v1(
            self._require_open(), ctypes.byref(result_size)
        )
        self._raise_state_status(status, "native snapshot result size")
        if result_size.value < 1:
            raise NativeCompactReducerError("native snapshot result is empty")
        result_bytes = (ctypes.c_uint8 * result_size.value)()
        status = self.artifact.library.eiel_native_compact_state_copy_snapshot_result_v1(
            self._require_open(), result_bytes, result_size.value
        )
        self._raise_state_status(status, "native snapshot result copy")
        try:
            result = json.loads(bytes(result_bytes))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise NativeCompactReducerError(
                "native snapshot result JSON is invalid"
            ) from exc
        if (
            not isinstance(result, Mapping)
            or result.get("schema_version")
            != "eiel_native_compact_snapshot_result_v1"
            or result.get("absolute_tick") != absolute_tick
            or not isinstance(result.get("worlds"), list)
        ):
            raise NativeCompactReducerError("native snapshot result contract drifted")
        return result

    def process_dispatch(
        self,
        *,
        start_next_absolute_tick: int,
        end_next_absolute_tick: int,
        event_rows: bytes,
        death_metric_rows: bytes,
        packed_birth_provenance_rows: bytes,
        world_offsets: Sequence[int],
    ) -> None:
        if (
            type(start_next_absolute_tick) is not int
            or type(end_next_absolute_tick) is not int
            or len(world_offsets) != self.world_count + 1
            or any(type(value) is not int for value in world_offsets)
        ):
            raise NativeCompactReducerError("native dispatch metadata is invalid")
        events = _byte_array(event_rows) if event_rows else None
        metrics = _byte_array(death_metric_rows) if death_metric_rows else None
        provenance = (
            _byte_array(packed_birth_provenance_rows)
            if packed_birth_provenance_rows
            else None
        )
        offsets = (ctypes.c_int64 * len(world_offsets))(*world_offsets)
        status = self.artifact.library.eiel_native_compact_state_process_dispatch_v1(
            self._require_open(),
            start_next_absolute_tick,
            end_next_absolute_tick,
            events,
            len(event_rows),
            metrics,
            len(death_metric_rows),
            provenance,
            len(packed_birth_provenance_rows),
            offsets,
            len(world_offsets),
        )
        self._raise_state_status(status, "native dispatch reduction")

    def configure_strategy(
        self,
        world_index: int,
        *,
        strategy_count: int,
        founder_agent_ids: Sequence[int],
        founder_strategy_masks: Sequence[int],
        threshold_ticks: Sequence[int],
        early_life_tick: int,
    ) -> None:
        maximum_mask = (
            2**strategy_count - 1
            if type(strategy_count) is int and 1 <= strategy_count <= STRATEGY_MAX_IDS
            else 0
        )
        if (
            type(world_index) is not int
            or not 0 <= world_index < self.world_count
            or type(strategy_count) is not int
            or not 1 <= strategy_count <= STRATEGY_MAX_IDS
            or len(founder_agent_ids) != len(founder_strategy_masks)
            or not 1 <= len(threshold_ticks) <= STRATEGY_MAX_THRESHOLDS
            or tuple(threshold_ticks) != tuple(sorted(set(threshold_ticks)))
            or any(
                type(tick) is not int or not 0 <= tick < 2**63
                for tick in threshold_ticks
            )
            or type(early_life_tick) is not int
            or not 0 <= early_life_tick < 2**63
            or any(
                type(agent_id) is not int or not 0 < agent_id < 2**64 - 1
                for agent_id in founder_agent_ids
            )
            or any(
                type(mask) is not int or not 0 < mask <= maximum_mask
                for mask in founder_strategy_masks
            )
        ):
            raise NativeCompactReducerError("native strategy configuration is invalid")
        ids = (ctypes.c_uint64 * len(founder_agent_ids))(*founder_agent_ids)
        masks = (ctypes.c_uint64 * len(founder_strategy_masks))(*founder_strategy_masks)
        thresholds = (ctypes.c_int64 * len(threshold_ticks))(*threshold_ticks)
        status = self.artifact.library.eiel_native_compact_state_configure_strategy_v1(
            self._require_open(),
            world_index,
            strategy_count,
            ids,
            masks,
            len(founder_agent_ids),
            thresholds,
            len(threshold_ticks),
            early_life_tick,
        )
        self._raise_state_status(status, "native strategy configuration")

    def observe_strategy_snapshot(
        self,
        world_index: int,
        *,
        active_agent_ids: Sequence[int],
        active_metric_rows: bytes,
    ) -> None:
        if (
            type(world_index) is not int
            or not 0 <= world_index < self.world_count
            or len(active_metric_rows) != len(active_agent_ids) * DEATH_METRIC_ROW_BYTES
            or any(
                type(agent_id) is not int or not 0 < agent_id < 2**64 - 1
                for agent_id in active_agent_ids
            )
        ):
            raise NativeCompactReducerError("native strategy snapshot is invalid")
        ids = (
            (ctypes.c_uint64 * len(active_agent_ids))(*active_agent_ids)
            if active_agent_ids
            else None
        )
        metrics = _byte_array(active_metric_rows) if active_metric_rows else None
        status = self.artifact.library.eiel_native_compact_state_observe_strategy_snapshot_v1(
            self._require_open(),
            world_index,
            ids,
            metrics,
            len(active_agent_ids),
        )
        self._raise_state_status(status, "native strategy snapshot")

    def finalize_strategy(self, world_index: int, *, terminal_tick: int) -> None:
        if (
            type(world_index) is not int
            or not 0 <= world_index < self.world_count
            or type(terminal_tick) is not int
            or not 0 <= terminal_tick < 2**63
        ):
            raise NativeCompactReducerError("native strategy terminal is invalid")
        status = self.artifact.library.eiel_native_compact_state_finalize_strategy_v1(
            self._require_open(), world_index, terminal_tick
        )
        self._raise_state_status(status, "native strategy finalization")

    def strategy_receipt(self, world_index: int) -> NativeStrategyReceiptResultV1:
        if type(world_index) is not int or not 0 <= world_index < self.world_count:
            raise NativeCompactReducerError("native strategy world index is invalid")
        receipt = StrategyReceiptV1()
        status = self.artifact.library.eiel_native_compact_state_strategy_receipt_v1(
            self._require_open(), world_index, ctypes.byref(receipt)
        )
        self._raise_state_status(status, "native strategy receipt")
        strategy_count = int(receipt.strategy_count)
        threshold_count = int(receipt.threshold_count)
        return NativeStrategyReceiptResultV1(
            enabled=bool(receipt.enabled),
            finalized=bool(receipt.finalized),
            strategy_count=strategy_count,
            threshold_ticks=tuple(
                int(receipt.threshold_ticks[index]) for index in range(threshold_count)
            ),
            early_life_tick=int(receipt.early_life_tick),
            first_birth_tick=(
                None if receipt.first_birth_tick < 0 else int(receipt.first_birth_tick)
            ),
            first_death_tick=(
                None if receipt.first_death_tick < 0 else int(receipt.first_death_tick)
            ),
            extinction_tick=(
                None if receipt.extinction_tick < 0 else int(receipt.extinction_tick)
            ),
            total_agent_count=int(receipt.total_agent_count),
            founder_count=int(receipt.founder_count),
            birth_count=int(receipt.birth_count),
            death_count=int(receipt.death_count),
            maximum_lineage_depth=int(receipt.maximum_lineage_depth),
            mixed_parent_birth_count=int(receipt.mixed_parent_birth_count),
            agent_counts_by_strategy=tuple(
                int(receipt.agent_counts_by_strategy[index])
                for index in range(strategy_count)
            ),
            offspring_born_by_strategy=tuple(
                int(receipt.offspring_born_by_strategy[index])
                for index in range(strategy_count)
            ),
            offspring_survived_by_strategy_threshold=tuple(
                tuple(
                    int(
                        receipt.offspring_survived_by_strategy_threshold[
                            strategy * STRATEGY_MAX_THRESHOLDS + threshold
                        ]
                    )
                    for threshold in range(threshold_count)
                )
                for strategy in range(strategy_count)
            ),
            maximum_lineage_depth_by_strategy=tuple(
                int(receipt.maximum_lineage_depth_by_strategy[index])
                for index in range(strategy_count)
            ),
            parent_birth_events_by_strategy=tuple(
                int(receipt.parent_birth_events_by_strategy[index])
                for index in range(strategy_count)
            ),
            death_events_by_strategy=tuple(
                int(receipt.death_events_by_strategy[index])
                for index in range(strategy_count)
            ),
            persistence_to_early_life_by_strategy=tuple(
                int(receipt.persistence_to_early_life_by_strategy[index])
                for index in range(strategy_count)
            ),
            terminal_alive_by_strategy=tuple(
                int(receipt.terminal_alive_by_strategy[index])
                for index in range(strategy_count)
            ),
            terminal_metric_totals_by_strategy=tuple(
                tuple(
                    (int(value.high) << 64) + int(value.low)
                    for value in receipt.terminal_metric_totals_by_strategy[
                        strategy
                        * DEATH_METRIC_FIELDS : (strategy + 1)
                        * DEATH_METRIC_FIELDS
                    ]
                )
                for strategy in range(strategy_count)
            ),
            offspring_group_counts=tuple(
                (
                    int(receipt.offspring_groups[index].strategy_mask),
                    int(receipt.offspring_groups[index].count),
                )
                for index in range(int(receipt.offspring_group_count))
            ),
            parent_pair_counts=tuple(
                (
                    int(receipt.parent_pairs[index].first_strategy_mask),
                    int(receipt.parent_pairs[index].second_strategy_mask),
                    int(receipt.parent_pairs[index].count),
                )
                for index in range(int(receipt.parent_pair_count))
            ),
            genotype_reproductive_attribution=self._genotype_result(world_index),
        )

    def _genotype_result(self, world_index: int) -> dict[str, Any] | None:
        result_size = ctypes.c_int64()
        status = self.artifact.library.eiel_native_compact_state_genotype_result_size_v1(
            self._require_open(), world_index, ctypes.byref(result_size)
        )
        self._raise_state_status(status, "native genotype result size")
        if result_size.value < 1:
            raise NativeCompactReducerError("native genotype result is empty")
        result_bytes = (ctypes.c_uint8 * result_size.value)()
        status = self.artifact.library.eiel_native_compact_state_copy_genotype_result_v1(
            self._require_open(), world_index, result_bytes, result_size.value
        )
        self._raise_state_status(status, "native genotype result copy")
        try:
            result = json.loads(bytes(result_bytes))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise NativeCompactReducerError(
                "native genotype result JSON is invalid"
            ) from exc
        if (
            not isinstance(result, dict)
            or set(result)
            not in (
                {
                    "schema_version",
                    "identity_mode",
                    "identity_scope",
                    "group_count",
                    "groups",
                    "descendant_birth_counts_by_parent_tracked_identity",
                    "totals",
                },
                {
                    "schema_version",
                    "identity_mode",
                    "identity_scope",
                    "group_scope",
                    "group_count",
                    "groups",
                    "descendant_birth_counts_by_parent_tracked_identity",
                    "totals",
                },
            )
            or result.get("schema_version")
            != "eiel_science_genotype_reproductive_attribution_v1"
            or result.get("identity_mode")
            != "exact_tracked_genotype_state_id_v1"
            or result.get("group_scope")
            not in {None, "requested_focal_ids_v1"}
            or type(result.get("group_count")) is not int
            or not isinstance(result.get("groups"), list)
            or not isinstance(result.get("identity_scope"), dict)
            or set(result["identity_scope"]) != {"policy_loci", "trait_loci"}
        ):
            raise NativeCompactReducerError("native genotype result contract drifted")
        if result["group_count"] != len(result["groups"]):
            raise NativeCompactReducerError("native genotype result group count differs")
        return result

    def world_receipt(self, world_index: int) -> NativeWorldReceiptResultV1:
        if type(world_index) is not int or not 0 <= world_index < self.world_count:
            raise NativeCompactReducerError("native world index is invalid")
        receipt = WorldReceiptV1()
        status = self.artifact.library.eiel_native_compact_state_world_receipt_v1(
            self._require_open(), world_index, ctypes.byref(receipt)
        )
        self._raise_state_status(status, "native world receipt")
        return NativeWorldReceiptResultV1(
            birth_count=int(receipt.birth_count),
            death_count=int(receipt.death_count),
            recombination_birth_count=int(receipt.recombination_birth_count),
            carrier_reproduction_birth_count=int(
                receipt.carrier_reproduction_birth_count
            ),
            mutation_carrier_reproduction_birth_count=int(
                receipt.mutation_carrier_reproduction_birth_count
            ),
            unique_born_agents_reproduced_count=int(
                receipt.unique_born_agents_reproduced_count
            ),
            unique_mutation_carriers_reproduced_count=int(
                receipt.unique_mutation_carriers_reproduced_count
            ),
            maximum_lineage_depth=int(receipt.maximum_lineage_depth),
            dispatch_count=int(receipt.dispatch_count),
            event_count=int(receipt.event_count),
            dispatch_tick_count=int(receipt.dispatch_tick_count),
            next_absolute_tick=int(receipt.next_absolute_tick),
            active_agent_count=int(receipt.active_agent_count),
            births=_birth_result(receipt.births),
            completed_lifetime_metric_totals=tuple(
                (int(value.high) << 64) + int(value.low)
                for value in receipt.completed_lifetime_metric_totals
            ),
        )

    def export_active(self, world_index: int) -> tuple[NativeActiveRecordResultV1, ...]:
        receipt = self.world_receipt(world_index)
        count = receipt.active_agent_count
        records = (ActiveRecordV1 * count)() if count else None
        output_count = ctypes.c_int64()
        status = self.artifact.library.eiel_native_compact_state_export_active_v1(
            self._require_open(),
            world_index,
            records,
            count,
            ctypes.byref(output_count),
        )
        self._raise_state_status(status, "native active-state export")
        if output_count.value != count:
            raise NativeCompactReducerError("native active-state count drifted")
        return tuple(
            NativeActiveRecordResultV1(
                agent_id=int(record.agent_id),
                generation=int(record.generation),
                strategy_mask=int(record.strategy_mask),
                birth_tick=int(record.birth_tick),
                mutation_lineage=bool(record.mutation_lineage),
                born=bool(record.born),
                reproduced=bool(record.reproduced),
                founder=bool(record.founder),
                latest_metrics_present=bool(record.latest_metrics_present),
                genome=bytes(record.genome),
            )
            for record in (records or ())
        )

    def close(self) -> None:
        if self._handle:
            self.artifact.library.eiel_native_compact_state_destroy_v1(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "NativeCompactReducerStateV1":
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def create_native_compact_reducer_state_v1(
    artifact: NativeCompactReducerArtifactV1,
    *,
    resident_capacities: Sequence[int],
    start_next_absolute_ticks: Sequence[int],
    mutation_segments: Sequence[Sequence[tuple[int, int]]],
    max_worker_threads: int,
) -> NativeCompactReducerStateV1:
    world_count = len(resident_capacities)
    if (
        world_count < 1
        or len(start_next_absolute_ticks) != world_count
        or len(mutation_segments) != world_count
        or type(max_worker_threads) is not int
        or max_worker_threads < 1
        or any(
            type(capacity) is not int or not 0 < capacity < 2**32
            for capacity in resident_capacities
        )
        or any(
            type(tick) is not int or not 0 <= tick < 2**63
            for tick in start_next_absolute_ticks
        )
    ):
        raise NativeCompactReducerError("native state configuration is invalid")
    offsets = [0]
    segment_ticks: list[int] = []
    segment_steps: list[int] = []
    for world_index, segments in enumerate(mutation_segments):
        if (
            not segments
            or any(
                type(tick) is not int
                or not -(2**63) <= tick < 2**63
                or type(step) is not int
                or not 0 < step < 2**31
                for tick, step in segments
            )
            or tuple(tick for tick, _ in segments)
            != tuple(sorted(set(tick for tick, _ in segments)))
            or segments[0][0] > start_next_absolute_ticks[world_index]
        ):
            raise NativeCompactReducerError("native mutation segments are invalid")
        segment_ticks.extend(tick for tick, _ in segments)
        segment_steps.extend(step for _, step in segments)
        offsets.append(len(segment_ticks))
    capacities_array = (ctypes.c_uint32 * world_count)(*resident_capacities)
    starts_array = (ctypes.c_int64 * world_count)(*start_next_absolute_ticks)
    offsets_array = (ctypes.c_int64 * len(offsets))(*offsets)
    segment_ticks_array = (ctypes.c_int64 * len(segment_ticks))(*segment_ticks)
    segment_steps_array = (ctypes.c_int32 * len(segment_steps))(*segment_steps)
    handle = ctypes.c_void_p()
    status = artifact.library.eiel_native_compact_state_create_v1(
        world_count,
        capacities_array,
        starts_array,
        offsets_array,
        segment_ticks_array,
        segment_steps_array,
        len(segment_ticks),
        max_worker_threads,
        ctypes.byref(handle),
    )
    _raise_status(status, "native state creation")
    if not handle:
        raise NativeCompactReducerError("native state creation returned no handle")
    return NativeCompactReducerStateV1(artifact, handle, world_count)


def reconstruct_packed_birth_genomes_native_v1(
    artifact: NativeCompactReducerArtifactV1,
    *,
    parent_a_genomes: bytes,
    parent_b_genomes: bytes,
    packed_birth_provenance_rows: bytes,
    mutation_steps: Sequence[int],
) -> NativeBirthBatchResultV1:
    row_count = len(mutation_steps)
    if (
        len(parent_a_genomes) != row_count * GENOME_ROW_BYTES
        or len(parent_b_genomes) != row_count * GENOME_ROW_BYTES
        or len(packed_birth_provenance_rows)
        != row_count * PACKED_BIRTH_PROVENANCE_BYTES
        or any(
            type(step) is not int or step < 1 or step > 2**31 - 1
            for step in mutation_steps
        )
    ):
        raise NativeCompactReducerError(
            "native birth batch inputs have invalid extents"
        )
    receipt = BirthBatchReceiptV1()
    if row_count == 0:
        return NativeBirthBatchResultV1(
            b"",
            0,
            0,
            0,
            0,
            (0,) * LINKED_LOCI,
            (0,) * LINKED_LOCI,
            (0,) * RECOMBINATION_CHOICES,
        )
    parent_a = _byte_array(parent_a_genomes)
    parent_b = _byte_array(parent_b_genomes)
    provenance = _byte_array(packed_birth_provenance_rows)
    steps = (ctypes.c_int32 * row_count)(*mutation_steps)
    children = (ctypes.c_uint8 * (row_count * GENOME_ROW_BYTES))()
    status = artifact.library.eiel_native_compact_reconstruct_birth_batch_v1(
        parent_a,
        parent_b,
        provenance,
        steps,
        row_count,
        children,
        ctypes.byref(receipt),
    )
    _raise_status(status, "native birth reconstruction")
    return NativeBirthBatchResultV1(
        child_genomes=bytes(children),
        reconstructed_birth_count=int(receipt.reconstructed_birth_count),
        mutation_origin_count=int(receipt.mutation_origin_count),
        saturated_noop_count=int(receipt.saturated_noop_count),
        recombination_nonzero_pattern_count=int(
            receipt.recombination_nonzero_pattern_count
        ),
        mutation_origin_counts_by_linked_locus=tuple(
            int(value) for value in receipt.mutation_origin_counts_by_linked_locus
        ),
        mutation_delta_totals_by_linked_locus=tuple(
            int(value) for value in receipt.mutation_delta_totals_by_linked_locus
        ),
        recombination_choice_one_counts=tuple(
            int(value) for value in receipt.recombination_choice_one_counts
        ),
    )


def reconstruct_packed_birth_genome_native_v1(
    artifact: NativeCompactReducerArtifactV1,
    *,
    parent_a_genome: bytes,
    parent_b_genome: bytes,
    packed_birth_provenance: bytes,
    mutation_step: int,
) -> NativeBirthBatchResultV1:
    return reconstruct_packed_birth_genomes_native_v1(
        artifact,
        parent_a_genomes=parent_a_genome,
        parent_b_genomes=parent_b_genome,
        packed_birth_provenance_rows=packed_birth_provenance,
        mutation_steps=(mutation_step,),
    )


def accumulate_death_metrics_native_v1(
    artifact: NativeCompactReducerArtifactV1, metric_rows: bytes
) -> tuple[int, ...]:
    if len(metric_rows) % DEATH_METRIC_ROW_BYTES != 0:
        raise NativeCompactReducerError("native death metric rows have invalid extent")
    row_count = len(metric_rows) // DEATH_METRIC_ROW_BYTES
    totals = (I128V1 * DEATH_METRIC_FIELDS)()
    rows = _byte_array(metric_rows) if metric_rows else None
    status = artifact.library.eiel_native_compact_accumulate_death_metrics_v1(
        rows, row_count, totals
    )
    _raise_status(status, "native death metric accumulation")
    return tuple((int(value.high) << 64) + int(value.low) for value in totals)


__all__ = (
    "ABI_VERSION",
    "ARTIFACT_FILE_NAME",
    "DEFAULT_BUILD_RELATIVE_PATH",
    "ActiveRecordV1",
    "BirthBatchReceiptV1",
    "DEATH_METRIC_FIELDS",
    "DEATH_METRIC_ROW_BYTES",
    "GENOME_ROW_BYTES",
    "MANIFEST_FILE_NAME",
    "NativeBirthBatchResultV1",
    "NativeActiveRecordResultV1",
    "NativeCompactReducerArtifactV1",
    "NativeCompactReducerError",
    "NativeCompactReducerStateV1",
    "NativeStrategyReceiptResultV1",
    "NativeWorldReceiptResultV1",
    "PACKED_BIRTH_PROVENANCE_BYTES",
    "STRATEGY_MAX_IDS",
    "STRATEGY_MAX_THRESHOLDS",
    "accumulate_death_metrics_native_v1",
    "build_native_compact_reducer_v1",
    "create_native_compact_reducer_state_v1",
    "default_native_compact_reducer_binding_v1",
    "load_native_compact_reducer_v1",
    "reconstruct_packed_birth_genome_native_v1",
    "reconstruct_packed_birth_genomes_native_v1",
)
