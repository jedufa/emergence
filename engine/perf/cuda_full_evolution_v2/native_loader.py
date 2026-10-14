"""Fail-closed loader for the prebuilt Full Evolution CUDA Native V2 binary."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from types import ModuleType
from typing import Any, Final, Mapping, Sequence

from .contract import (
    CUDA_NATIVE_ABI_V1,
    CUDA_NATIVE_PROFILE_V2,
    CUDA_NATIVE_ROUTE_V2,
    MAX_CAPABILITY_COUNTER_INCREMENT_PER_TICK_V2,
    MAX_TOTAL_TICKS_COUNTER_SAFE_V2,
)
from .layout import (
    CUDA_NATIVE_ALIGNMENT_BYTES,
    CUDA_NATIVE_HEADER_NAMES_V2,
    CUDA_NATIVE_LIFECYCLE_EVENT_ROW_BYTES_V2,
    CUDA_NATIVE_LIFECYCLE_EVENT_ROW_FIELDS_V2,
    CUDA_NATIVE_LIFECYCLE_EVENT_ROWS_PER_WORLD_PER_TICK_V2,
    CUDA_NATIVE_SPARSE_EVENT_KEY_KIND_BITS_V3,
    CUDA_NATIVE_SPARSE_EVENT_KEY_KIND_SHIFT_V3,
    CUDA_NATIVE_SPARSE_EVENT_KEY_SITE_BITS_V3,
    CUDA_NATIVE_SPARSE_EVENT_KEY_SITE_SHIFT_V3,
    CUDA_NATIVE_SPARSE_EVENT_KEY_SUBJECT_BITS_V3,
    CUDA_NATIVE_SPARSE_EVENT_KEY_SUBJECT_SHIFT_V3,
    CUDA_NATIVE_SPARSE_EVENT_KEY_TICK_BITS_V3,
    CUDA_NATIVE_SPARSE_EVENT_KEY_USED_BITS_V3,
    CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3,
    CUDA_NATIVE_SPARSE_EVENT_WIRE_ABI_V3,
    state_layout_v2,
)

BUILD_MANIFEST_SCHEMA_V2: Final[str] = (
    "eiel_full_evolution_cuda_native_build_manifest_v2"
)
EXTENSION_VERSION_V2: Final[str] = "eiel_full_evolution_cuda_native_extension_v2"
PREBUILT_EXTENSION_ENV_V2: Final[str] = "EIEL_FULL_EVOLUTION_CUDA_V2_PREBUILT"
EXECUTION_PLAN_CLASS_V2: Final[str] = "CudaFullEvolutionExecutionPlanV2"
REQUIRED_EXPORTS_V2: Final[tuple[str, ...]] = (
    "abi_version",
    "extension_version",
    "architecture_id",
    "route_id",
    "layout_manifest",
    "create_execution_plan",
    "initialize",
    "initialize_science_worlds_v1",
    "advance_reference",
    "advance",
    "advance_direct_sparse_v3",
    "advance_direct_sparse_worlds_v1",
    "advance_direct_sparse_compact_science_worlds_v1",
    "advance_direct_sparse_science_v1",
    "advance_direct_sparse_science_worlds_v1",
    "compact_events_v3",
    "pack_direct_sparse_events_v3",
    "pack_direct_sparse_science_events_v1",
    "rebuild_derived_caches",
    "force_test_conditions",
    "force_full_capacity_benchmark_fixture_v2",
)
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
BOUND_SOURCE_RELATIVE_PATHS_V2: Final[tuple[str, ...]] = (
    "engine/perf/cuda_full_evolution_v2/native/full_evolution_v2.cu",
    "engine/perf/cuda_full_evolution_v2/contract.py",
    "engine/perf/cuda_full_evolution_v2/layout.py",
    "engine/perf/cuda_full_evolution_v2/native_loader.py",
    "engine/perf/cuda_full_evolution_v2/build.py",
    "engine/perf/cuda_full_evolution_v2/runtime.py",
    "engine/perf/cuda_full_evolution_v2/direct_sparse_runtime.py",
    "engine/perf/cuda_full_evolution_v2/science_native_world_runtime_v1.py",
    "engine/perf/cuda_full_evolution_v2/direct_sparse_autotune.py",
    "engine/perf/cuda_full_evolution_v2/direct_sparse_sustained_benchmark.py",
    "engine/perf/cuda_full_evolution_v2/science_extension.py",
    "emergence_contracts/science_compact_capture.py",
    "engine/campaigns/science_cuda_v2_native_world_v1.py",
    "engine/native_runtime/cuda_v2_science_capture_v1.py",
    "engine/native_runtime/cuda_v2_compact_science_capture_v1.py",
    "_spec/science_instrument/v1/native_founder_profile_registry_v1.json",
    "_artifacts/tw-cuda-full-evolution-reform-20260825/architecture_contract.json",
    "_artifacts/tw-cuda-full-evolution-reform-20260825/performance_restructure_contract.json",
    "_artifacts/tw-cuda-full-evolution-reform-20260825/science_consumer_extension_contract.json",
    "_artifacts/tw-cuda-full-evolution-reform-20260825/host_layout_receipt.json",
)
# The retained V2 manifest records the complete source snapshot used when the
# extension was built.  Only this smaller closure is a live dependency of the
# compiled binary/ABI.  Host owners, capture adapters, autotune tooling and
# reports retain their historical build provenance in ``source_inventory`` but
# are versioned independently by the execution-route contract; changing them
# must not pretend that the CUDA binary changed.
NATIVE_ARTIFACT_LIVE_DEPENDENCY_PATHS_V2: Final[tuple[str, ...]] = (
    "engine/perf/cuda_full_evolution_v2/native/full_evolution_v2.cu",
    "engine/perf/cuda_full_evolution_v2/contract.py",
    "engine/perf/cuda_full_evolution_v2/layout.py",
    "_artifacts/tw-cuda-full-evolution-reform-20260825/architecture_contract.json",
    "_artifacts/tw-cuda-full-evolution-reform-20260825/performance_restructure_contract.json",
    "_artifacts/tw-cuda-full-evolution-reform-20260825/science_consumer_extension_contract.json",
    "_artifacts/tw-cuda-full-evolution-reform-20260825/host_layout_receipt.json",
)
NORMATIVE_EXECUTION_CONTRACT_PATHS_V2: Final[tuple[str, ...]] = (
    "_artifacts/tw-cuda-full-evolution-reform-20260825/architecture_contract.json",
    "_artifacts/tw-cuda-full-evolution-reform-20260825/performance_restructure_contract.json",
    "_artifacts/tw-cuda-full-evolution-reform-20260825/science_consumer_extension_contract.json",
)


class CudaFullEvolutionNativeLoadError(RuntimeError):
    """Raised when a binary is not exactly bound to its immutable receipt."""


@dataclass(frozen=True, slots=True)
class LoadedCudaFullEvolutionNativeV2:
    module: ModuleType
    artifact_path: Path
    artifact_sha256: str
    manifest_path: Path
    manifest_sha256: str
    module_name: str
    target_sm: str
    device_identity: Mapping[str, Any]
    manifest: Mapping[str, Any]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_component_identity_v2(
    relative_paths: Sequence[str],
) -> tuple[str, tuple[dict[str, str], ...]]:
    """Return a dependency-accurate identity for one live source component."""

    if isinstance(relative_paths, (str, bytes, bytearray)):
        raise CudaFullEvolutionNativeLoadError(
            "source component paths must be a non-empty unique sequence"
        )
    try:
        paths = tuple(relative_paths)
    except TypeError as exc:
        raise CudaFullEvolutionNativeLoadError(
            "source component paths must be a non-empty unique sequence"
        ) from exc
    if not paths or any(not isinstance(value, str) or not value for value in paths):
        raise CudaFullEvolutionNativeLoadError(
            "source component paths must be a non-empty unique sequence"
        )
    if len(paths) != len(set(paths)):
        raise CudaFullEvolutionNativeLoadError(
            "source component paths must be a non-empty unique sequence"
        )
    rows: list[dict[str, str]] = []
    for value in paths:
        relative = Path(value)
        if relative.is_absolute() or "." in relative.parts or ".." in relative.parts:
            raise CudaFullEvolutionNativeLoadError(
                "source component path is unsafe"
            )
        source = PROJECT_ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise CudaFullEvolutionNativeLoadError(
                f"source component input is unavailable: {relative}"
            )
        rows.append({"path": relative.as_posix(), "sha256": file_sha256(source)})
    frozen = tuple(rows)
    return canonical_sha256(list(frozen)), frozen


def cuda_device_identity_v2(torch: Any, device_index: int) -> dict[str, Any]:
    """Exact selected-device identity for execution receipts, separate from SM compatibility."""
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "index": int(device_index),
        "name": str(properties.name),
        "uuid": str(properties.uuid),
        "pci_domain_id": int(properties.pci_domain_id),
        "pci_bus_id": int(properties.pci_bus_id),
        "pci_device_id": int(properties.pci_device_id),
        "total_memory_bytes": int(properties.total_memory),
        "multiprocessor_count": int(properties.multi_processor_count),
        "sm": f"sm_{int(properties.major)}{int(properties.minor)}",
        "cuda_runtime": str(torch.version.cuda),
    }


def expected_layout_manifest_v2() -> dict[str, Any]:
    """Canonical Python projection compared byte-for-byte with the CUDA export."""
    layout = state_layout_v2()
    return {
        "schema_version": "eiel_full_evolution_cuda_native_layout_manifest_v2",
        "alignment_bytes": CUDA_NATIVE_ALIGNMENT_BYTES,
        "state_bytes_per_world": layout.state_bytes_per_world,
        "canonical_checkpoint_bytes_per_world": (
            layout.canonical_checkpoint_bytes_per_world
        ),
        "derived_cache_bytes_per_world": layout.derived_cache_bytes_per_world,
        "reference_scratch_bytes_per_world": layout.reference_scratch_bytes_per_world,
        "optimized_kernel_global_scratch_bytes": (
            layout.optimized_kernel_global_scratch_bytes
        ),
        "counter_safety": {
            "max_capability_counter_increment_per_tick": (
                MAX_CAPABILITY_COUNTER_INCREMENT_PER_TICK_V2
            ),
            "max_total_ticks": MAX_TOTAL_TICKS_COUNTER_SAFE_V2,
            "counter_dtype": "signed_int64_projection_of_u64_header",
        },
        "rng_address": {
            "fields": [
                "rng_key",
                "rng_semantic_epoch",
                "world_uid",
                "absolute_tick",
                "domain",
                "subject",
                "operation",
                "draw",
            ],
            "semantic_epoch": 3,
            "generator": "philox4x32_10",
            "composition": "arx_folded_full_u64_counter_address_v3",
            "arithmetic": "integer_only",
            "output_width_bits": 32,
            "site_bundle_domain": "site_environment",
            "site_bundle_subject": "canonical_site_index",
            "site_bundle_outputs": [
                "patch_carrying",
                "patch_growth",
                "wdl_schedule",
                "jungle_transition",
            ],
            "child_genome_bundle": {
                "persistent_subject": "child_biological_uid",
                "recombination": (
                    "one_philox_block_per_linked_block_" "parent_uids_in_operation_draw"
                ),
                "mutation": ("one_philox_block_per_linked_locus_two_allele_outputs"),
                "forbidden_subject": "canonical_agent_slot",
            },
            "forbidden_inputs": [
                "agent_slot",
                "batch_index",
                "cta",
                "warp",
                "lane",
                "launch_order",
            ],
        },
        "header_names": list(CUDA_NATIVE_HEADER_NAMES_V2),
        "lifecycle_event_row": {
            "bytes": CUDA_NATIVE_LIFECYCLE_EVENT_ROW_BYTES_V2,
            "complete_identity_key": ["world_uid", "absolute_tick", "event_id"],
            "event_id_codec": {
                "kind_bits": 16,
                "ordinal_bits": 48,
                "subject_ordinal": "canonical_agent_slot",
            },
            "event_kinds": {"none": 0, "death": 1, "birth": 2},
            "fields": [
                {"name": name, "dtype": dtype}
                for name, dtype in CUDA_NATIVE_LIFECYCLE_EVENT_ROW_FIELDS_V2
            ],
            "rows_per_world_per_tick": (
                CUDA_NATIVE_LIFECYCLE_EVENT_ROWS_PER_WORLD_PER_TICK_V2
            ),
            "row_index": {"death": "slot", "birth": "512+slot"},
            "status": {"unused": 0, "committed": 1},
            "unused_row": "all_zero",
            "reserved_value": 0,
        },
        "sparse_lifecycle_event_v3": {
            "wire_abi": CUDA_NATIVE_SPARSE_EVENT_WIRE_ABI_V3,
            "bytes": CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3,
            "fields": [
                {"name": "agent_id", "dtype": "u64"},
                {"name": "parent_a_id", "dtype": "u64"},
                {"name": "parent_b_id", "dtype": "u64"},
                {"name": "packed_key_and_reserved", "dtype": "u64"},
            ],
            "packed_key": {
                "tick_offset": {
                    "bits": CUDA_NATIVE_SPARSE_EVENT_KEY_TICK_BITS_V3,
                    "shift": 0,
                },
                "event_kind": {
                    "bits": CUDA_NATIVE_SPARSE_EVENT_KEY_KIND_BITS_V3,
                    "shift": CUDA_NATIVE_SPARSE_EVENT_KEY_KIND_SHIFT_V3,
                },
                "subject_slot": {
                    "bits": CUDA_NATIVE_SPARSE_EVENT_KEY_SUBJECT_BITS_V3,
                    "shift": CUDA_NATIVE_SPARSE_EVENT_KEY_SUBJECT_SHIFT_V3,
                },
                "site_index": {
                    "bits": CUDA_NATIVE_SPARSE_EVENT_KEY_SITE_BITS_V3,
                    "shift": CUDA_NATIVE_SPARSE_EVENT_KEY_SITE_SHIFT_V3,
                },
                "used_bits": CUDA_NATIVE_SPARSE_EVENT_KEY_USED_BITS_V3,
                "high_reserved_value": 0,
            },
            "canonical_order": [
                "world_uid",
                "tick_offset",
                "death_before_birth",
                "subject_slot",
            ],
            "world_offsets": "u64[B+1]_zero_origin_exact_terminal_N",
        },
        "fields": [
            {
                "name": field.name,
                "dtype": field.dtype,
                "shape": list(field.shape),
                "offset_bytes": field.offset_bytes,
                "bytes": field.bytes,
            }
            for field in layout.fields
        ],
    }


def expected_execution_plan_contract_v2() -> dict[str, Any]:
    """Declared micro-plan law; only runtime inspection proves an actual graph."""
    return {
        "schema_version": "eiel_full_evolution_cuda_execution_plan_contract_v2",
        "profile_id": "eiel_full_evolution_cuda_native_v2",
        "execution_modes": ["eager_monolith", "cuda_graph_monolith"],
        "default_execution_mode": "eager_monolith",
        "kernel": "advance_optimized_kernel",
        "prepared_dispatch_temporal_k_variants": (
            "every integer in [1,event_capacity_k], created before timed launch"
        ),
        "eager": {
            "kernel_launches_per_dispatch": 1,
            "cuda_graph_nodes": 0,
            "dependency_edges": [],
        },
        "cuda_graph": {
            "construction": "explicit_cuda_graph_api_no_stream_capture",
            "real_kernel_nodes": 1,
            "kernel_order": ["advance_optimized_kernel"],
            "dependency_edges": [],
            "wrapper_or_noop_nodes": "forbidden",
        },
        "binding": {
            "strong_tensor_references": True,
            "immutable_pointer_shape_device_batch_temporal_k_capability": True,
            "immutable_launch_stream_including_default_stream_0": True,
            "profile_artifact_manifest_identity": True,
            "pointer_substitution": "forbidden",
            "restore_constructs_fresh_plan": True,
        },
        "timing_boundary": {
            "create_instantiate_inspect": "outside_timed_launch",
            "launch_only_inside_timed_scope": True,
            "per_node_or_gap_timing_claimed": False,
            "timing_fabrication": "forbidden",
        },
        "native_launch_receipt": {
            "complete_exact_key_set": True,
            "source_bound_topology_and_pointers": True,
            "committed_ticks_at_native_boundary": 0,
            "runtime_commit_requires_exact_content_sink_acknowledgement": True,
        },
        "technical_fixtures": {
            "full_capacity": {
                "id": "eiel_full_evolution_cuda_full_capacity_benchmark_fixture_v2",
                "required_frontier": "fresh_canonical_p256_before_any_tick",
                "active_agents_per_world": 512,
                "checkpoint_sealed": True,
                "causal_or_science_claim": False,
            }
        },
        "runtime_inspection": {
            "eager_binding_validated": True,
            "cuda_graph_topology_inspected": True,
            "declared_flag_alone_sufficient": False,
        },
        "declaration_is_execution_evidence": False,
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CudaFullEvolutionNativeLoadError(
                f"duplicate key in CUDA Native V2 build manifest: {key}"
            )
        result[key] = value
    return result


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CudaFullEvolutionNativeLoadError(
            f"cannot read CUDA Native V2 build manifest: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise CudaFullEvolutionNativeLoadError("build manifest must be a JSON object")
    return value


def _regular_non_symlink(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise CudaFullEvolutionNativeLoadError(f"{label} must not be a symlink")
    resolved = path.resolve()
    if not resolved.is_file():
        raise CudaFullEvolutionNativeLoadError(
            f"{label} is missing or not a regular file: {resolved}"
        )
    return resolved


def validate_prebuilt_cuda_full_evolution_v2(
    artifact_path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    artifact = _regular_non_symlink(Path(artifact_path), "prebuilt extension")
    manifest_path = _regular_non_symlink(
        artifact.with_suffix(artifact.suffix + ".manifest.json"), "build manifest"
    )
    if artifact.stat().st_mode & 0o222 or manifest_path.stat().st_mode & 0o222:
        raise CudaFullEvolutionNativeLoadError(
            "prebuilt extension and manifest must be read-only"
        )
    manifest = _load_manifest(manifest_path)
    if manifest.get("schema_version") != BUILD_MANIFEST_SCHEMA_V2:
        raise CudaFullEvolutionNativeLoadError("build manifest schema mismatch")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest_sha256 = str(manifest.get("manifest_sha256", "") or "")
    if len(manifest_sha256) != 64 or canonical_sha256(body) != manifest_sha256:
        raise CudaFullEvolutionNativeLoadError("build manifest hash mismatch")
    if manifest.get("immutable_output") is not True:
        raise CudaFullEvolutionNativeLoadError("artifact is not declared immutable")
    if manifest.get("prebuilt_outside_hot_advance") is not True:
        raise CudaFullEvolutionNativeLoadError("artifact is not declared prebuilt")
    if manifest.get("runtime_jit") is not False:
        raise CudaFullEvolutionNativeLoadError("runtime JIT must be false")
    if manifest.get("compiled_from_sealed_snapshot") is not True:
        raise CudaFullEvolutionNativeLoadError(
            "artifact was not compiled from a sealed source snapshot"
        )
    if manifest.get("sealed_snapshot") != {
        "files_mode": "0444",
        "directories_mode": "0555",
        "post_compile_inventory_verified": True,
    }:
        raise CudaFullEvolutionNativeLoadError(
            "sealed source snapshot receipt mismatch"
        )
    if manifest.get("profile") != CUDA_NATIVE_PROFILE_V2:
        raise CudaFullEvolutionNativeLoadError("native profile mismatch")
    if manifest.get("route_id") != CUDA_NATIVE_ROUTE_V2:
        raise CudaFullEvolutionNativeLoadError("native route mismatch")
    if manifest.get("native_abi") != CUDA_NATIVE_ABI_V1:
        raise CudaFullEvolutionNativeLoadError("native ABI mismatch")
    if manifest.get("extension_version") != EXTENSION_VERSION_V2:
        raise CudaFullEvolutionNativeLoadError("extension version mismatch")
    module_name = str(manifest.get("module_name", "") or "")
    if not module_name.isidentifier():
        raise CudaFullEvolutionNativeLoadError("module name is invalid")
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise CudaFullEvolutionNativeLoadError("output receipt is missing")
    if output.get("file_name") != artifact.name:
        raise CudaFullEvolutionNativeLoadError("output filename mismatch")
    actual_size = artifact.stat().st_size
    if output.get("bytes") != actual_size or output.get("sha256") != file_sha256(
        artifact
    ):
        raise CudaFullEvolutionNativeLoadError("output size or hash mismatch")
    exports = manifest.get("exported_functions")
    if exports != list(REQUIRED_EXPORTS_V2):
        raise CudaFullEvolutionNativeLoadError("exported-function receipt mismatch")
    if manifest.get("layout_manifest") != expected_layout_manifest_v2():
        raise CudaFullEvolutionNativeLoadError("host/native layout receipt mismatch")
    if manifest.get("layout_manifest_sha256") != canonical_sha256(
        manifest["layout_manifest"]
    ):
        raise CudaFullEvolutionNativeLoadError("layout manifest hash mismatch")
    runtime = manifest.get("runtime")
    # ACS2 capture is a host-side checkpoint transport.  The Science
    # Instrument's already sealed V2 binary predates the ACS2 manifest
    # annotations, but it has the same native ABI and science export.  Keep
    # that exact artifact usable for host-context/science validation while
    # retaining the stricter current contract for newly built artifacts.
    expected_runtime = {
        "device_or_tensor_allocations_inside_hot_advance": 0,
        "python_pending_dispatch_objects_inside_hot_advance": 1,
        "host_device_transfers_inside_hot_advance": 0,
        "host_synchronizations_inside_hot_advance": 0,
        "optimized_global_scratch_bytes": 0,
        "production_owner": "CudaFullEvolutionDirectSparseOwnerV3",
        "production_event_emission": "direct_per_world_bounded_sparse_v3",
        "production_event_drain": "actual_count_only_sparse_v3_native_pack",
        "dense_event_drain_role": "tiny_gpu_differential_oracle_only",
        "event_drain_timing_reported_separately": True,
        "sparse_drain_allocation": "exact_committed_event_count",
        "public_advance_dispatches": "one_successful_attempt_plus_overflow_retries",
        "dispatch_tick_limit": "temporal_k",
        "overflow_transaction": (
            "full_resident_state_d2d_snapshot_restore_re_admit_grow_retry"
        ),
        "direct_sparse_resource_admission": (
            "state_plus_full_state_rollback_plus_direct_staging_plus_"
            "simultaneous_exact_pack_plus_controls_plus_margin"
        ),
        "horizon_dependent_resident_bytes": 0,
        "event_frontier_commit": (
            "exact_canonical_local_envelope_inode_digest_open_descriptor_" "receipt_v3"
        ),
        "dense_oracle_event_frontier_commit": (
            "exact_immutable_sparse_content_external_sink_receipt_v3"
        ),
        "checkpoint_requires_acknowledged_event_frontier": True,
        "checkpoint_schema": "eiel_full_evolution_cuda_checkpoint_v2",
        "checkpoint_canonical_state": "immutable_cpu_bytes_with_content_layout_and_metadata_seals",
        "acs2_capture_source": "producer_borrowed_resident_cuda_rows_v1",
        "acs2_device_compaction_bytes": 0,
        "acs2_full_state_host_materialization_bytes": 0,
        "acs2_capture_pool": "one_shared_preallocated_two_pinned_slabs",
        "acs2_worker_cuda_references": 0,
        "public_state_projection": "validated_detached_clone_only",
        "drain_validation_boundary": "non_timed",
        "execution_plan_contract": "eiel_full_evolution_cuda_execution_plan_contract_v2",
        "execution_plan_creation_boundary": "outside_timed_launch",
        "execution_plan_runtime_receipt": "eiel_full_evolution_cuda_execution_plan_receipt_v2",
        "execution_plan_stream_policy": (
            "immutable_construction_stream_including_default_stream_0"
        ),
        "native_execution_receipt_binding": (
            "exact_source_topology_pointers_device_artifact_and_null_timings"
        ),
        "checkpoint_selected_device_identity": "exact_runtime_device_identity_v2",
        "full_capacity_benchmark_fixture": (
            "fresh_p256_to_exact_c512_sealed_technical_fixture_v2"
        ),
    }
    legacy_science_runtime = {
        key: value
        for key, value in expected_runtime.items()
        if key not in {
            "acs2_capture_source",
            "acs2_device_compaction_bytes",
            "acs2_full_state_host_materialization_bytes",
            "acs2_capture_pool",
            "acs2_worker_cuda_references",
        }
    }
    if runtime not in (expected_runtime, legacy_science_runtime):
        raise CudaFullEvolutionNativeLoadError("hot-advance contract mismatch")
    if manifest.get("execution_plan_contract") != expected_execution_plan_contract_v2():
        raise CudaFullEvolutionNativeLoadError("execution-plan contract mismatch")
    normative_contracts = manifest.get("normative_execution_contracts")
    expected_contract_rows = [
        entry
        for entry in manifest.get("source_inventory", [])
        if isinstance(entry, dict)
        and entry.get("path") in NORMATIVE_EXECUTION_CONTRACT_PATHS_V2
    ]
    if normative_contracts != expected_contract_rows:
        raise CudaFullEvolutionNativeLoadError(
            "normative execution-contract source binding mismatch"
        )
    python_receipt = manifest.get("python")
    if (
        not isinstance(python_receipt, dict)
        or python_receipt.get("cache_tag") != sys.implementation.cache_tag
    ):
        raise CudaFullEvolutionNativeLoadError("Python ABI tag mismatch")
    if python_receipt.get("version") != platform.python_version():
        raise CudaFullEvolutionNativeLoadError("Python runtime version mismatch")
    if python_receipt.get("platform") != platform.platform():
        raise CudaFullEvolutionNativeLoadError("platform receipt mismatch")
    source_inventory = manifest.get("source_inventory")
    if not isinstance(source_inventory, list) or len(source_inventory) != len(
        BOUND_SOURCE_RELATIVE_PATHS_V2
    ):
        raise CudaFullEvolutionNativeLoadError("source inventory is missing")
    if [
        entry.get("path") if isinstance(entry, dict) else None
        for entry in source_inventory
    ] != list(BOUND_SOURCE_RELATIVE_PATHS_V2):
        raise CudaFullEvolutionNativeLoadError("source inventory path set mismatch")
    if manifest.get("source_inventory_sha256") != canonical_sha256(source_inventory):
        raise CudaFullEvolutionNativeLoadError("source inventory hash mismatch")
    if manifest.get("post_build_live_source_inventory_sha256") != canonical_sha256(
        source_inventory
    ):
        raise CudaFullEvolutionNativeLoadError(
            "post-build live source inventory hash mismatch"
        )
    live_native_paths = set(NATIVE_ARTIFACT_LIVE_DEPENDENCY_PATHS_V2)
    if not live_native_paths.issubset(BOUND_SOURCE_RELATIVE_PATHS_V2):
        raise CudaFullEvolutionNativeLoadError(
            "native artifact live-dependency closure is not a manifest subset"
        )
    live_manifest_rows: list[dict[str, str]] = []
    for entry in source_inventory:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise CudaFullEvolutionNativeLoadError("source inventory row is invalid")
        relative = Path(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise CudaFullEvolutionNativeLoadError("source inventory path is unsafe")
        if relative.as_posix() in live_native_paths:
            source = PROJECT_ROOT / relative
            if source.is_symlink() or not source.is_file() or file_sha256(source) != entry["sha256"]:
                raise CudaFullEvolutionNativeLoadError(
                    f"prebuilt native dependency mismatch: {relative}"
                )
            live_manifest_rows.append(
                {"path": relative.as_posix(), "sha256": str(entry["sha256"])}
            )
    live_identity, live_rows = source_component_identity_v2(
        NATIVE_ARTIFACT_LIVE_DEPENDENCY_PATHS_V2
    )
    if tuple(live_manifest_rows) != live_rows or live_identity != canonical_sha256(
        live_manifest_rows
    ):
        raise CudaFullEvolutionNativeLoadError(
            "prebuilt native dependency identity mismatch"
        )
    return artifact, manifest_path, manifest


def load_prebuilt_cuda_full_evolution_v2(
    artifact_path: str | Path | None = None,
    *,
    device: str | Any | None = None,
) -> LoadedCudaFullEvolutionNativeV2:
    supplied = artifact_path or os.environ.get(PREBUILT_EXTENSION_ENV_V2)
    if not supplied:
        raise CudaFullEvolutionNativeLoadError(
            f"set {PREBUILT_EXTENSION_ENV_V2} to an explicitly built extension"
        )
    artifact, manifest_path, manifest = validate_prebuilt_cuda_full_evolution_v2(
        supplied
    )
    target = manifest.get("target")
    if not isinstance(target, dict) or not isinstance(target.get("sm"), str):
        raise CudaFullEvolutionNativeLoadError("target SM receipt is missing")
    try:
        import torch

        if not torch.cuda.is_available():
            raise CudaFullEvolutionNativeLoadError("CUDA is unavailable")
        selected_device = torch.device("cuda" if device is None else device)
        if selected_device.type != "cuda":
            raise CudaFullEvolutionNativeLoadError(
                "CUDA Native V2 artifact validation requires a CUDA device"
            )
        device_index = (
            torch.cuda.current_device()
            if selected_device.index is None
            else selected_device.index
        )
        device_identity = cuda_device_identity_v2(torch, device_index)
        current_sm = str(device_identity["sm"])
    except CudaFullEvolutionNativeLoadError:
        raise
    except Exception as exc:
        raise CudaFullEvolutionNativeLoadError(
            f"cannot inspect CUDA target: {exc}"
        ) from exc
    if target["sm"] != current_sm:
        raise CudaFullEvolutionNativeLoadError(
            f"prebuilt target {target['sm']} does not match current {current_sm}"
        )
    build_device = target.get("build_device")
    if not isinstance(build_device, dict) or build_device.get("sm") != target["sm"]:
        raise CudaFullEvolutionNativeLoadError("build-device receipt mismatch")
    if target.get("cuda_runtime") != str(torch.version.cuda):
        raise CudaFullEvolutionNativeLoadError("CUDA runtime receipt mismatch")
    toolchain = manifest.get("toolchain")
    if not isinstance(toolchain, dict) or toolchain.get("torch") != str(
        torch.__version__
    ):
        raise CudaFullEvolutionNativeLoadError("PyTorch binary receipt mismatch")
    module_name = str(manifest["module_name"])
    spec = importlib.util.spec_from_file_location(module_name, artifact)
    if spec is None or spec.loader is None:
        raise CudaFullEvolutionNativeLoadError("extension import specification failed")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise CudaFullEvolutionNativeLoadError(
            f"cannot import prebuilt CUDA Native V2 extension: {exc}"
        ) from exc
    missing = [name for name in REQUIRED_EXPORTS_V2 if not hasattr(module, name)]
    if missing:
        raise CudaFullEvolutionNativeLoadError(
            f"binary is missing required exports: {missing}"
        )
    if not isinstance(getattr(module, EXECUTION_PLAN_CLASS_V2, None), type):
        raise CudaFullEvolutionNativeLoadError(
            "binary is missing its native execution-plan class"
        )
    if module.abi_version() != CUDA_NATIVE_ABI_V1:
        raise CudaFullEvolutionNativeLoadError("loaded binary ABI mismatch")
    if module.extension_version() != EXTENSION_VERSION_V2:
        raise CudaFullEvolutionNativeLoadError("loaded extension version mismatch")
    if module.architecture_id() != CUDA_NATIVE_PROFILE_V2:
        raise CudaFullEvolutionNativeLoadError("loaded architecture ID mismatch")
    if module.route_id() != CUDA_NATIVE_ROUTE_V2:
        raise CudaFullEvolutionNativeLoadError("loaded route ID mismatch")
    if module.layout_manifest() != expected_layout_manifest_v2():
        raise CudaFullEvolutionNativeLoadError("loaded binary layout mismatch")
    return LoadedCudaFullEvolutionNativeV2(
        module=module,
        artifact_path=artifact,
        artifact_sha256=str(manifest["output"]["sha256"]),
        manifest_path=manifest_path,
        manifest_sha256=str(manifest["manifest_sha256"]),
        module_name=module_name,
        target_sm=current_sm,
        device_identity=device_identity,
        manifest=manifest,
    )


__all__ = [
    "BOUND_SOURCE_RELATIVE_PATHS_V2",
    "BUILD_MANIFEST_SCHEMA_V2",
    "CudaFullEvolutionNativeLoadError",
    "EXECUTION_PLAN_CLASS_V2",
    "EXTENSION_VERSION_V2",
    "LoadedCudaFullEvolutionNativeV2",
    "NATIVE_ARTIFACT_LIVE_DEPENDENCY_PATHS_V2",
    "PREBUILT_EXTENSION_ENV_V2",
    "NORMATIVE_EXECUTION_CONTRACT_PATHS_V2",
    "REQUIRED_EXPORTS_V2",
    "canonical_sha256",
    "cuda_device_identity_v2",
    "expected_layout_manifest_v2",
    "expected_execution_plan_contract_v2",
    "file_sha256",
    "load_prebuilt_cuda_full_evolution_v2",
    "source_component_identity_v2",
    "validate_prebuilt_cuda_full_evolution_v2",
]
